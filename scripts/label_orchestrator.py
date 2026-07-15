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
    crash / no_output / skipped, with attempt count and wall-clock. Safe to
    Ctrl-C and resume: on restart, only pending/failed-under-retry work is
    re-run.
  * Contracts where ALL four tools fail are recorded (unlabellable), never
    silently dropped.
  * Runs tools on CPU workers in parallel (so it can label while a GPU trains
    elsewhere). One SmartBugs invocation per (contract, tool).
  * Enrolment order is SHUFFLED (seeded), so a partially completed run is an
    unbiased random sample of the pool: the staged targets (Stage 1 Test A,
    Stage 2 10k) can be cut at any point without alphabetical bias.
  * Emits a ledger summary (per-tool completion, mean seconds, class-independent)
    for the dissertation's dataset table.

This module does NOT parse findings into labels — that stays in the existing
run_tools.collect_votes / label.py, which read the results tree this produces.

SmartBugs invocation (CORRECTED)
--------------------------------
SmartBugs 2.x treats ``--results`` as a path TEMPLATE with ``${TOOL}``,
``${RUNID}`` and ``${FILENAME}`` variables (its default is
``results/${TOOL}/${RUNID}/${FILENAME}``). Passing a plain directory makes
every task write into the SAME directory: each run sees existing results
there, skips the analysis, and exits 0 in a few seconds — which a
previous full run did, producing a ledger full of 'ok' rows and exactly one
result triplet on disk. Two defences now exist:

  1. The orchestrator builds the template itself from ``--results`` (the base
     directory), producing ``<base>/${TOOL}/${RUNID}/${FILENAME}`` — exactly
     the layout ``collect_votes`` walks. Override with ``--results-template``
     if your SmartBugs differs.
  2. An exit code of 0 is no longer sufficient for 'ok': the task must have
     produced a FRESH ``result.json`` (SmartBugs' parsed output, from
     ``--json``) for this tool and contract. Exit 0 without output is recorded
     as ``no_output`` and retried like a crash. The ledger can therefore never
     again claim success for work that left nothing behind.

The per-task subprocess timeout still applies as a hard kill. SmartBugs' own
``--timeout`` is additionally passed (slightly below the hard kill) so the tool
container is stopped cleanly by SmartBugs rather than orphaned by our kill;
disable with ``--no-sb-timeout`` if your SmartBugs lacks the flag. After a run
with many hard kills, check ``docker ps`` for stragglers.

Usage
-----
    # dedup + build the worklist, then label (resumable):
    python scripts/label_orchestrator.py \
        --wild-dir data/raw/wild \
        --results data/sb_results \
        --ledger data/labelling_ledger.sqlite \
        --workers 40 --timeout 300 --max-attempts 2

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

import numpy as np

TOOLS = ["slither", "mythril", "securify", "osiris"]
PARSER_OUTPUT = "result.json"    # SmartBugs' parsed findings (what collect_votes reads)

# --- canonical content hash (must match training/data/firewall.content_hash) ---
_WS = re.compile(r"\s+")


def strip_comments(source: str) -> str:
    """Remove // and /* */ comments, respecting string literals.

    Mirrors training/data/firewall.strip_comments so dedup here and the
    train/test firewall agree on what 'the same contract' means. A tiny state
    machine rather than a regex, so ``"https://x"`` inside a string survives.
    """
    out: list[str] = []
    i, n = 0, len(source)
    in_line = in_block = False
    quote: str | None = None
    while i < n:
        c = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
            i += 1
            continue
        if in_block:
            if c == "*" and nxt == "/":
                in_block = False
                out.append(" ")
                i += 2
                continue
            i += 1
            continue
        if quote:
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out.append(c)
            i += 1
            continue
        if c == "/" and nxt == "/":
            in_line = True
            i += 2
            continue
        if c == "/" and nxt == "*":
            in_block = True
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def content_hash(source: str) -> str:
    """Comment-stripped, whitespace-normalised sha256. Mirrors
    firewall.content_hash so dedup and the train/test firewall agree on what
    'the same contract' means."""
    normalised = _WS.sub(" ", strip_comments(source)).strip()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


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
    status     TEXT NOT NULL,         -- pending|ok|timeout|crash|no_output|skipped
    attempts   INTEGER NOT NULL DEFAULT 0,
    seconds    REAL NOT NULL DEFAULT 0,
    updated_at REAL,
    PRIMARY KEY (cid, tool)
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""

RETRYABLE = ("timeout", "crash", "no_output")


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
        transient failure (timeout/crash/no_output) under the retry cap."""
        con = self._con()
        rows = con.execute(
            "SELECT t.cid, t.tool, c.path FROM tasks t JOIN contracts c ON c.cid=t.cid "
            "WHERE t.status='pending' "
            "   OR (t.status IN ('timeout','crash','no_output') AND t.attempts < ?)",
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
            mean_ok = con.execute(
                "SELECT COALESCE(AVG(seconds),0) FROM tasks WHERE tool=? AND status='ok'",
                (tool,)).fetchone()[0]
            per_tool[tool] = {"statuses": row, "mean_ok_seconds": float(mean_ok)}
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

def results_template(base: str | Path) -> str:
    """The SmartBugs results template rooted at ``base``.

    Produces ``<base>/${TOOL}/${RUNID}/${FILENAME}`` — the layout
    ``collect_votes`` expects. The ``${...}`` variables are substituted by
    SmartBugs itself (the command is exec'd, not shell-expanded, so they pass
    through literally).
    """
    return str(Path(base) / "${TOOL}" / "${RUNID}" / "${FILENAME}")


def _has_fresh_result(base: Path, tool: str, contract: str, since: float) -> bool:
    """Did this (tool, contract) task leave a NEW parsed result on disk?

    Looks under ``<base>/<tool-id>/<runid>/<contract>/result.json``, accepting
    versioned tool-id dirs (``mythril-0.24.7``) and contract dirs named with or
    without the ``.sol`` extension. Freshness (mtime >= task start) guards
    against a skipped run being validated by an older file.
    """
    name = Path(contract).name
    stem = Path(contract).stem
    patterns = {f"{tool}*/*/{name}/{PARSER_OUTPUT}",
                f"{tool}*/*/{stem}/{PARSER_OUTPUT}"}
    for pat in patterns:
        for p in base.glob(pat):
            try:
                if p.stat().st_mtime >= since - 2.0:
                    return True
            except OSError:
                continue
    return False


def run_task(sb_cmd: list[str], tool: str, contract: str, results_base: str,
             timeout: float, sb_timeout: int | None,
             template: str | None = None) -> tuple[str, float]:
    """Run one (tool, contract) SmartBugs task. Returns (status, seconds).

    status: 'ok' (exit 0 AND a fresh result.json exists), 'no_output' (exit 0
    but nothing on disk — the failure mode of the collided-results run),
    'timeout', or 'crash' (non-zero / launch failure). Findings themselves are
    parsed later by collect_votes from the results tree. ``template`` overrides
    the results template built from ``results_base``.
    """
    cmd = list(sb_cmd) + ["-t", tool, "-f", contract,
                          "--results", template or results_template(results_base),
                          "--json"]
    if sb_timeout:
        cmd += ["--timeout", str(int(sb_timeout))]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", time.time() - t0
    except (FileNotFoundError, OSError):
        return "crash", time.time() - t0
    dt = time.time() - t0
    if proc.returncode != 0:
        return "crash", dt
    if not _has_fresh_result(Path(results_base), tool, contract, t0):
        return "no_output", dt
    return "ok", dt


# ------------------------------ worklist build ------------------------------

def build_worklist(wild_dir: Path, limit: int | None, *,
                   shuffle: bool = True, seed: int = 42) -> list[tuple[str, str, str]]:
    """Scan .sol files, DEDUP by content hash, return (cid, path, chash).

    First occurrence of each hash wins; later copies are dropped (their tool
    labels would be identical). cid is the file stem; a stem carried by TWO
    different contents is skipped and counted (it would merge two contracts'
    votes downstream, where results and labels are keyed by filename).

    With ``shuffle`` (default), the deduplicated list is shuffled with a fixed
    seed BEFORE any limit is applied, so partial progress and ``--limit``
    batches are unbiased random samples of the pool rather than alphabetical
    prefixes.
    """
    seen: dict[str, str] = {}          # chash -> cid kept
    stem_hash: dict[str, str] = {}     # stem  -> chash kept under that stem
    out: list[tuple[str, str, str]] = []
    n_scanned = n_dup = n_stem_collision = 0
    for sol in sorted(wild_dir.rglob("*.sol")):
        if "__MACOSX" in sol.parts or sol.name.startswith("._"):
            continue
        n_scanned += 1
        h = hash_file(sol)
        if h in seen:
            n_dup += 1
            continue
        prev = stem_hash.get(sol.stem)
        if prev is not None and prev != h:
            n_stem_collision += 1
            print(f"  WARNING stem collision, skipping: {sol}", flush=True)
            continue
        seen[h] = sol.stem
        stem_hash[sol.stem] = h
        out.append((sol.stem, str(sol), h))

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(out)
    if limit:
        out = out[:limit]
    print(f"scanned {n_scanned} .sol, {n_dup} duplicates dropped, "
          f"{n_stem_collision} stem collisions skipped, "
          f"{len(out)} unique to label"
          + (" (shuffled, seed %d)" % seed if shuffle else ""), flush=True)
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
    suspicious = []
    for tool, row in s["per_tool"].items():
        ok = row["statuses"].get("ok", 0)
        mean = row["mean_ok_seconds"]
        print(f"  {tool:<10} ok={ok:<7} mean_ok={mean:6.1f}s  {row['statuses']}")
        if tool != "slither" and ok > 50 and mean < 10.0:
            suspicious.append(tool)
    if suspicious:
        print(f"\n  WARNING: {', '.join(suspicious)} averaging <10s per 'ok' task. "
              f"Symbolic-execution tools cannot be that fast: verify the results "
              f"tree (find <results> -name result.json | wc -l) before trusting "
              f"this ledger.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wild-dir", help="Root of Wild .sol files (required unless --report).")
    ap.add_argument("--results", default="data/sb_results",
                    help="BASE directory for SmartBugs results. The orchestrator "
                         "appends ${TOOL}/${RUNID}/${FILENAME} itself.")
    ap.add_argument("--results-template", default=None,
                    help="Full SmartBugs results template, overriding the one "
                         "built from --results. Only needed if your SmartBugs "
                         "uses different variables.")
    ap.add_argument("--ledger", default="data/labelling_ledger.sqlite")
    ap.add_argument("--sb-cmd", default="sb",
                    help="SmartBugs base command (e.g. 'sb' or 'python -m sb').")
    ap.add_argument("--workers", type=int, default=40, help="Parallel tool processes.")
    ap.add_argument("--timeout", type=float, default=300, help="Hard per-task kill (s).")
    ap.add_argument("--no-sb-timeout", action="store_true",
                    help="Do not pass SmartBugs' own --timeout (use if your "
                         "SmartBugs version rejects the flag).")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="Retry cap for timeout/crash/no_output tasks.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap unique contracts (for the timed smoke batch).")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="Enrol in path order instead of seeded random order.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    ap.add_argument("--report", action="store_true", help="Print ledger summary and exit.")
    args = ap.parse_args()

    led = Ledger(args.ledger)

    if args.report:
        print_report(led)
        return 0

    if not args.wild_dir:
        sys.exit("ERROR: --wild-dir is required (or use --report).")

    # 1. dedup + enrol (idempotent; safe to re-run as the pool grows)
    work = build_worklist(Path(args.wild_dir), args.limit,
                          shuffle=not args.no_shuffle, seed=args.seed)
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
    base = args.results
    template = args.results_template          # None -> built from base per task
    Path(base).mkdir(parents=True, exist_ok=True)
    sb_timeout = None if args.no_sb_timeout else max(30, int(args.timeout) - 30)
    done = 0
    t_start = time.time()
    lock = threading.Lock()

    def worker(task):
        cid, tool, path = task
        status, secs = run_task(sb_cmd, tool, path, base, args.timeout,
                                sb_timeout, template)
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
    print("\nNext: run scripts/label.py --results", base,
          "--method union   to build labels.parquet from these results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())