#!/usr/bin/env bash
# scans/ -> OCR -> documents.json -> searchable PDFs
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SCANS="${1:?usage: make_documents.sh scans_dir [workers] [llm_url]}"
WORKERS="${2:-4}"
LLM="${3:-}"

export OMP_NUM_THREADS=1
python "$HERE/scripts/ocr_scans.py" "$SCANS" --workers "$WORKERS"
ARGS=(--out "$SCANS/documents.json")
[ -n "$LLM" ] && ARGS+=(--llm "$LLM")
python "$HERE/scripts/group_documents.py" "$SCANS/ocr" "${ARGS[@]}"
python "$HERE/scripts/build_pdfs.py" "$SCANS/documents.json" "$SCANS" \
    --out "$SCANS/pdfs" --ocr "$SCANS/ocr"
