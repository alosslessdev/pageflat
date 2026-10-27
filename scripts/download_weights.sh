#!/usr/bin/env bash
# Fetch the DocTr checkpoints (segmentation, geometric unwarping, illumination).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights
BASE="https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained"
for f in seg.pth geotr.pth illtr.pth; do
  if [ ! -s "weights/$f" ]; then
    echo "downloading $f"
    curl -L --fail -o "weights/$f" "$BASE/$f"
  fi
done
ls -la weights
