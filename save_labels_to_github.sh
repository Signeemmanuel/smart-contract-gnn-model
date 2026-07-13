#!/usr/bin/env bash
# save_labels_to_github.sh — commit data/processed/labels.parquet to GitHub,
# overriding the repo's .gitignore, with size checks and REAL verification.
#
# Why this is not just `git add`:
#   .gitignore blocks labels.parquet three ways (data/processed/*, *.parquet).
#   A plain `git add` SILENTLY IGNORES it — you'd push and think it saved when
#   it didn't. This forces the add, checks GitHub's 100 MB file limit first,
#   and verifies the blob is actually staged before committing.
#
# Usage:
#   bash save_labels_to_github.sh                    # default path
#   bash save_labels_to_github.sh path/to/labels.parquet
#
# After a successful run, ALSO keep an off-repo copy (laptop / release asset):
# you lost this file once already; redundancy is the point.

set -euo pipefail

FILE="${1:-data/processed/labels.parquet}"

if [[ ! -f "$FILE" ]]; then
  echo "ERROR: $FILE not found. Run labelling first, or pass the correct path."
  exit 1
fi

# --- size check against GitHub's hard 100 MB per-file limit ---
BYTES=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE")
MB=$(awk "BEGIN{printf \"%.1f\", $BYTES/1048576}")
echo "File: $FILE  (${MB} MB)"

if (( BYTES > 100*1048576 )); then
  echo "STOP: ${MB} MB exceeds GitHub's 100 MB per-file limit."
  echo "A normal commit WILL be rejected. Use a GitHub Release asset instead:"
  echo "  gh release create labels-v1 \"$FILE\" --title 'Wild labels' --notes 'labels.parquet'"
  echo "(or 'gh release upload <tag> \"$FILE\"' to add to an existing release)."
  exit 1
fi
if (( BYTES > 50*1048576 )); then
  echo "NOTE: ${MB} MB is large for a git commit (GitHub warns >50 MB). It will"
  echo "work but bloats history. A Release asset is cleaner. Continuing anyway."
fi

# --- force past .gitignore ---
echo "force-adding (overriding .gitignore) ..."
git add -f "$FILE"

# --- VERIFY it is actually staged (the silent-ignore trap) ---
if ! git diff --cached --name-only | grep -qxF "$FILE"; then
  echo "ERROR: $FILE did NOT stage. Something else is blocking it."
  echo "Check: git check-ignore -v \"$FILE\""
  exit 1
fi
echo "verified staged: $FILE"

# --- commit + show what will be pushed ---
git commit -m "Save Wild labels.parquet (${MB} MB) — expensive labelling artefact"
echo
echo "committed. To push:"
echo "  git push"
echo
echo "REMINDER: also copy this file OFF the repo (laptop / release asset)."
echo "You lost it once; keep at least two independent copies."
