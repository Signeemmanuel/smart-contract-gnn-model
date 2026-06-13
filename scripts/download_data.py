#!/usr/bin/env python3
"""Download the SmartBugs datasets into the project's data/raw tree.

Pulls SmartBugs Curated (small; expert labels + gold lines) and SmartBugs Wild
(~47.4k contracts; the large one) into ``data/raw/curated`` and ``data/raw/wild``
so the build script finds them at its default paths. Standard library + git.

Idempotent: a target that already contains contracts is left alone (use --force
to re-fetch). The Wild clone is the big download — hundreds of MB — but you pay
for it once; it persists across Studio restarts.

Examples:
    python scripts/download_data.py                 # both datasets, via git
    python scripts/download_data.py --skip-wild     # just Curated, to start
    python scripts/download_data.py --wild-method tarball
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPOS = {
    "curated": {
        "git": "https://github.com/smartbugs/smartbugs-curated.git",
        "branch": "main",
        "tarball": "https://github.com/smartbugs/smartbugs-curated/archive/refs/heads/main.tar.gz",
        "marker": "dataset",            # a subdir that proves the content arrived
    },
    "wild": {
        "git": "https://github.com/smartbugs/smartbugs-wild.git",
        "branch": "master",
        "tarball": "https://github.com/smartbugs/smartbugs-wild/archive/refs/heads/master.tar.gz",
        "marker": "contracts",
    },
}


def _has_git() -> bool:
    return shutil.which("git") is not None


def _already_there(dest: Path, marker: str) -> bool:
    return (dest / marker).is_dir() and any((dest / marker).rglob("*.sol"))


def _clone(repo: dict, dest: Path) -> None:
    """Shallow git clone (no history) straight into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", repo["branch"], repo["git"], str(dest)],
        check=True,
    )


def _tarball(repo: dict, dest: Path) -> None:
    """Download the branch tarball and move its single top dir to ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "src.tar.gz"
        print(f"   downloading {repo['tarball']}")
        with urllib.request.urlopen(repo["tarball"]) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp)                       # nosec: trusted source
        top = next(p for p in tmp.iterdir() if p.is_dir() and p.name != archive.name)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(top), str(dest))


def fetch(name: str, dest: Path, method: str, force: bool) -> None:
    repo = REPOS[name]
    if not force and _already_there(dest, repo["marker"]):
        print(f"== {name}: already present at {dest} (use --force to refetch); skipping")
        return
    print(f"== {name}: fetching into {dest} via {method}")
    if method == "git":
        if not _has_git():
            sys.exit("error: git not found; rerun with --%s-method tarball" % name)
        _clone(repo, dest)
    else:
        _tarball(repo, dest)
    n = len(list((dest / repo["marker"]).rglob("*.sol")))
    print(f"   done: {n} .sol files under {dest / repo['marker']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--skip-curated", action="store_true")
    ap.add_argument("--skip-wild", action="store_true")
    ap.add_argument("--curated-method", choices=["git", "tarball"], default="git")
    ap.add_argument("--wild-method", choices=["git", "tarball"], default="git")
    ap.add_argument("--force", action="store_true", help="Refetch even if present.")
    args = ap.parse_args()

    if not args.skip_curated:
        fetch("curated", args.out_dir / "curated", args.curated_method, args.force)
    if not args.skip_wild:
        fetch("wild", args.out_dir / "wild", args.wild_method, args.force)

    print("\nNext: PYTHONPATH=. python scripts/build_dataset.py "
          "--wild-dir data/raw/wild --wild-labels data/processed/labels.parquet "
          "--curated-dir data/raw/curated --out data/processed --max-wild 500")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
