#!/usr/bin/env bash
# Fetch model weights: DocTr checkpoints, MobileSAM (recommended) and optionally SAM ViT-B.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p weights

DOC_TR="https://huggingface.co/spaces/HaoFeng2019/DocTr/resolve/main/model_pretrained"
for f in seg.pth geotr.pth illtr.pth; do
  if [ ! -s "weights/$f" ]; then
    echo "downloading DocTr $f"
    curl -L --fail -o "weights/$f" "$DOC_TR/$f"
  fi
done

if [ ! -s weights/mobile_sam.pt ]; then
  echo "downloading MobileSAM"
  curl -L --fail -o weights/mobile_sam.pt \
    https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
fi

if [ "${WITH_SAM_VIT_B:-0}" = "1" ] && [ ! -s weights/sam_vit_b.pth ]; then
  echo "downloading SAM ViT-B (375 MB)"
  curl -L --fail -o weights/sam_vit_b.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi

ls -la weights
