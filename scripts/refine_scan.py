#!/usr/bin/env python3
"""Residual refinement for already-flattened scans (see pageflat.refine)."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageflat.refine import crop_to_paper, straighten


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-straighten", action="store_true")
    ap.add_argument("--no-crop", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.scans)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.png"))
    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        if not args.no_crop:
            img = crop_to_paper(img)
        if not args.no_straighten:
            img = straighten(img)
        cv2.imwrite(str(out / path.name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
