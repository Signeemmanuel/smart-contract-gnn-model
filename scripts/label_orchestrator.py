#!/usr/bin/env python3
"""Resumable labelling orchestrator (Workstream A).

Drives SmartBugs' four tools (slither, mythril, securify, osiris) over the Wild
pool with a persistent SQLite ledger, per-contract timeouts, retries, and
safe stop/resume. It sits BEFORE the existing parse->union pipeline:

    orchestrator (this) ->  SmartBugs results tree  ->  scripts/label.py
                            + ledger                     (collect_votes -> union)

Design (matches the handoff's Workstream A):
  * De-duplicates the pool by canonical content hash BEFORE labelling, so no
    tool time is wasted on copies (popular contracts are widely duplicated).
  * A ledger row per (contract, tool) records status: pending / ok / timeout /
    crash / skipped, with attempt count and wall-clock. Safe to Ctrl-C and
    resume: on restart, only pending/failed-under-retry work is re-run.
  * Contracts where ALL four tools fail are recorded (unlabellable), never
    silently dropped.
  * Runs tools on CPU workers in parallel (so it can label while a GPU trains
    elsewhere). One SmartBugs invocation per (contract, tool).
  * Emits a ledger summary (per-tool completion, class-independent) for the
    dissertation's dataset table.

This module does NOT parse findings into labels — that stays in the existing
run_tools.collect_votes / label.py, which read the results tree this produces.

SmartBugs invocation
--------------------
By default each task runs:  sb -t <tool> -f <contract.sol> --results <dir> --json
Override the base command with --sb-cmd if your SmartBugs entrypoint differs
(e.g. "python -m sb" or a wrapper). The orchestrator only cares about the exit
status, a per-task timeout, and that SmartBugs writes result.json under
<results>/<tool>/<runid>/<contract>/ (which collect_votes already expects).

Usage
-----
    # dedup + build the worklist, then label (resumable):
    python scripts/label_orchestrator.py \
        --wild-dir data/raw/wild \
        --results data/sb_results \
        --ledger data/labelling_ledger.sqlite \
        --workers 40 --timeout 120 --max-attempts 2

    # resume after interruption: identical command; done work is skipped.
    # report progress without running:
    python scripts/label_orchestrator.py --ledger data/labelling_ledger.sqlite --report
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import re
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

TOOLS = ["slither", "mythril", "securify", "osiris"]

# --- canonical content hash (must match training/data/firewall.content_hash) ---
_WS = re.compile(r"\s+")


def content_hash(source: str) -> str:
    """Whitespace-normalised sha256. Mirrors firewall.content_hash so dedup and
    the train/test firewall agree on what 'the same contract' means."""
    return hashlib.sha256(_WS.sub(" ", source).strip().encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    try:
        return content_hash(path.read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return f"__unreadable__:{path}"


# ----------------------------- ledger (SQLite) -----------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    cid        TEXT PRIMARY KEY,      -- contract stem (unique after dedup)
    path       TEXT NOT NULL,
    chash      TEXT NOT NULL,         -- canonical content hash
    added_at   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
    cid        TEXT NOT NULL,
    tool       TEXT NOT NULL,
    status     TEXT NOT NULL,         -- pending|ok|timeout|crash|skipped
    attempts   INTEGER NOT NULL DEFAULT 0,
    seconds    REAL NOT NULL DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (cid, tool)
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


class Ledger:
    """Thread-safe SQLite ledger. WAL mode so parallel workers can update."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        con = self._con()
        con.executescript(_SCHEMA)
        con.execute("PRAGMA journal_mode=WAL;")
        con.commit()

    def _con(self) -> sqlite3.Connection:
        c = getattr(self._local, "con", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=60)
            c.execute("PRAGMA busy_timeout=60000;")
            self._local.con = c
        return c

    def enrol(self, contracts: list[tuple[str, str, str]]) -> tuple[int, int]:
        """Insert contracts + their pending tasks. Idempotent (INSERT OR IGNORE),
        so re-running after adding more contracts only enrols the new ones.
        Returns (new_contracts, new_tasks)."""
        con = self._con()
        now = time.time()
        nc = nt = 0
        for cid, path, chash in contracts:
            cur = con.execute(
                "INSERT OR IGNORE INTO contracts(cid,path,chash,added_at) VALUES (?,?,?,?)",
                (cid, path, chash, now))
            nc += cur.rowcount
            for tool in TOOLS:
                cur = con.execute(
                    "INSERT OR IGNORE INTO tasks(cid,tool,status,attempts,updated_at) "
                    "VALUES (?,?,?,0,?)", (cid, tool, "pending", now))
                nt += cur.rowcount
        con.commit()
        return nc, nt

    def claim_pending(self, max_attempts: int) -> list[tuple[str, str, str]]:
        """Return (cid, tool, path) tasks still needing work: pending, or a
        transient failure (timeout/crash) under the retry cap."""
        con = self._con()
        rows = con.execute(
            "SELECT t.cid, t.tool, c.path FROM tasks t JOIN contracts c ON c.cid=t.cid "
            "WHERE t.status='pending' "
            "   OR (t.status IN ('timeout','crash') AND t.attempts < ?)",
            (max_attempts,)).fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

    def update(self, cid: str, tool: str, status: str, seconds: float) -> None:
        con = self._con()
        con.execute(
            "UPDATE tasks SET status=?, attempts=attempts+1, seconds=?, updated_at=? "
            "WHERE cid=? AND tool=?", (status, seconds, time.time(), cid, tool))
        con.commit()

    def summary(self) -> dict:
        con = self._con()
        total_c = con.execute("SELECT COUNT(*) FROM contracts").fetchone()[0]
        by_status = dict(con.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall())
        per_tool = {}
        for tool in TOOLS:
            row = dict(con.execute(
                "SELECT status, COUNT(*) FROM tasks WHERE tool=? GROUP BY status",
                (tool,)).fetchall())
            per_tool[tool] = row
        # contracts where every tool failed (no 'ok' among its four tasks)
        unlabellable = con.execute(
            "SELECT COUNT(*) FROM (SELECT cid FROM tasks GROUP BY cid "
            "HAVING SUM(status='ok')=0)").fetchone()[0]
        done = con.execute(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('ok','skipped')").fetchone()[0]
        total_t = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        secs = con.execute("SELECT COALESCE(SUM(seconds),0) FROM tasks").fetchone()[0]
        return {"contracts": total_c, "tasks_total": total_t, "tasks_done": done,
                "by_status": by_status, "per_tool": per_tool,
                "unlabellable_contracts": unlabellable, "tool_seconds_total": secs}


# --------------------------- SmartBugs task runner ---------------------------

def run_task(sb_cmd: list[str], tool: str, contract: str, results: str,
             timeout: float) -> tuple[str, float]:
    """Run one (tool, contract) SmartBugs task. Returns (status, seconds).

    status: 'ok' (exit 0), 'timeout', or 'crash' (non-zero / launch failure).
    We treat only a clean exit as ok; findings themselves are parsed later by
    collect_votes from the results tree.
    """
    cmd = list(sb_cmd) + ["-t", tool, "-f", contract, "--results", results, "--json"]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0
    except (FileNotFoundError, OSError):
        return "crash", time.time() - t0
    return ("ok" if proc.returncode == 0 else "crash"), time.time() - t0


# ------------------------------ worklist build ------------------------------

def build_worklist(wild_dir: Path, limit: int | None) -> list[tuple[str, str, str]]:
    """Scan .sol files, DEDUP by content hash, return (cid, path, chash).

    First occurrence of each hash wins; later copies are dropped (their tool
    labels would be identical). cid is the file stem.
    """
    seen: dict[str, str] = {}          # chash -> cid kept
    out: list[tuple[str, str, str]] = []
    n_scanned = n_dup = 0
    for sol in sorted(wild_dir.rglob("*.sol")):
        if "__MACOSX" in sol.parts or sol.name.startswith("._"):
            continue
        n_scanned += 1
        h = hash_file(sol)
        if h in seen:
            n_dup += 1
            continue
        seen[h] = sol.stem
        out.append((sol.stem, str(sol), h))
        if limit and len(out) >= limit:
            break
    print(f"scanned {n_scanned} .sol, {n_dup} duplicates dropped, "
          f"{len(out)} unique to label", flush=True)
    return out


# ---------------------------------- main ----------------------------------

def print_report(led: Ledger) -> None:
    s = led.summary()
    print("\n=== labelling ledger summary ===")
    print(f"contracts (unique):        {s['contracts']}")
    print(f"tasks done / total:        {s['tasks_done']} / {s['tasks_total']}")
    print(f"task status:               {s['by_status']}")
    print(f"unlabellable contracts:    {s['unlabellable_contracts']}  (all 4 tools failed)")
    print(f"tool-seconds (cumulative): {s['tool_seconds_total']:.0f}s "
          f"({s['tool_seconds_total']/3600:.2f} h)")
    print("per-tool status:")
    for tool, row in s["per_tool"].items():
        ok = row.get("ok", 0)
        print(f"  {tool:<10} ok={ok:<7} {row}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild-dir", help="Root of Wild .sol files (required unless --report).")
    ap.add_argument("--results", default="data/sb_results", help="SmartBugs results dir.")
    ap.add_argument("--ledger", default="data/labelling_ledger.sqlite")
    ap.add_argument("--sb-cmd", default="sb",
                    help="SmartBugs base command (e.g. 'sb' or 'python -m sb').")
    ap.add_argument("--workers", type=int, default=40, help="Parallel tool processes.")
    ap.add_argument("--timeout", type=float, default=120, help="Per-task timeout (s).")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Retry cap for timeout/crash tasks.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap unique contracts (for the timed smoke batch).")
    ap.add_argument("--report", action="store_true", help="Print ledger summary and exit.")
    args = ap.parse_args()

    led = Ledger(args.ledger)

    if args.report:
        print_report(led)
        return 0

    if not args.wild_dir:
        sys.exit("ERROR: --wild-dir is required (or use --report).")

    # 1. dedup + enrol (idempotent; safe to re-run as the pool grows)
    work = build_worklist(Path(args.wild_dir), args.limit)
    nc, nt = led.enrol(work)
    print(f"enrolled {nc} new contracts, {nt} new tasks", flush=True)

    # 2. claim outstanding tasks and run them in parallel
    tasks = led.claim_pending(args.max_attempts)
    print(f"{len(tasks)} tasks to run (workers={args.workers}, timeout={args.timeout}s)",
          flush=True)
    if not tasks:
        print("nothing to do; all tasks complete or exhausted retries.")
        print_report(led)
        return 0

    sb_cmd = args.sb_cmd.split()
    Path(args.results).mkdir(parents=True, exist_ok=True)
    done = 0
    t_start = time.time()
    lock = threading.Lock()

    def worker(task):
        cid, tool, path = task
        status, secs = run_task(sb_cmd, tool, path, args.results, args.timeout)
        led.update(cid, tool, status, secs)
        return cid, tool, status

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in cf.as_completed(futures):
            done += 1
            if done % 100 == 0 or done == len(tasks):
                rate = done / max(1e-9, time.time() - t_start)
                eta = (len(tasks) - done) / max(1e-9, rate)
                with lock:
                    print(f"  {done}/{len(tasks)} tasks  "
                          f"({rate:.1f}/s, ETA {eta/60:.1f} min)", flush=True)

    print(f"\nrun complete in {(time.time()-t_start)/60:.1f} min", flush=True)
    print_report(led)
    print("\nNext: run scripts/label.py --results", args.results,
          "--method union   to build labels.parquet from these results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())