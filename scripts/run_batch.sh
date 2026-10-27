#!/usr/bin/env bash
# Example: flatten every page found by find_pages.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LIST="${1:?usage: run_batch.sh pages.jsonl out_dir [workers]}"
OUT="${2:?usage: run_batch.sh pages.jsonl out_dir [workers]}"
WORKERS="${3:-1}"

python - "$LIST" > /tmp/pageflat_inputs.txt <<'PY'
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        print(json.loads(line)["path"])
    except json.JSONDecodeError:
        print(line)
PY

python -m pageflat.cli --list /tmp/pageflat_inputs.txt -o "$OUT" \
    --dewarp best --illum classical --workers "$WORKERS" --json "$OUT/report.json"
