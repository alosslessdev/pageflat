#!/usr/bin/env python3
"""Fix page orientation with docTR's page-orientation classifier.

Class index = number of 90 degree clockwise turns needed to make the page
upright (0, 1, 2, 3). The classifier is a MobileNetV3-small trained by Mindee,
which is far more reliable on image-heavy pages than Tesseract OSD or OCR
confidence scoring.

    python scripts/fix_orientation.py scans/ --workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_PREDICTOR = None


def _init(device: str = "cpu") -> None:
    global _PREDICTOR
    from doctr.models import page_orientation_predictor

    _PREDICTOR = page_orientation_predictor(pretrained=True)
    if device != "cpu":
        _PREDICTOR.model = _PREDICTOR.model.to(device)


def _fix_batch(path_strs: list[str]) -> list[dict]:
    imgs, keep = [], []
    for path_str in path_strs:
        img = cv2.imread(path_str)
        if img is None:
            continue
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        keep.append((Path(path_str), img))
    if not imgs:
        return [{"file": Path(p).name, "status": "skipped"} for p in path_strs]
    preds = _PREDICTOR(imgs)
    # doctr returns (class_indices, angles_in_degrees, confidences), each a list
    classes = np.asarray(preds[0]).ravel()
    confs = np.asarray(preds[2]).ravel() if len(preds) > 2 else np.ones(len(classes))
    out = []
    for (path, img), cls, conf in zip(keep, classes, confs):
        cls = int(cls)
        if conf < 0.5:
            out.append({"file": path.name, "class": cls, "rotated": 0,
                        "skipped_low_conf": round(float(conf), 3)})
            continue
        turns = {0: None, 1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180,
                 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[cls]
        if turns is not None:
            cv2.imwrite(str(path), cv2.rotate(img, turns), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        out.append({"file": path.name, "class": cls, "rotated": cls * 90})
    return out


def _fix(path_str: str) -> dict:
    return _fix_batch([path_str])[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    root = Path(args.scans)
    if root.is_file():
        files = [l.strip() for l in root.read_text().splitlines() if l.strip()]
    else:
        files = [str(p) for p in sorted(root.glob("*.png"))]
    batches = [files[i:i + args.batch] for i in range(0, len(files), args.batch)]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(args.device,)) as pool:
            rows = [r for chunk in pool.map(_fix_batch, batches) for r in chunk]
    else:
        _init(args.device)
        rows = []
        for i, batch in enumerate(batches, 1):
            rows.extend(_fix_batch(batch))
            if i % 20 == 0:
                print(f"{i * args.batch}/{len(files)}", flush=True)
    changed = sum(1 for r in rows if r.get("rotated"))
    print(f"{len(rows)} scans, {changed} rotated", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
