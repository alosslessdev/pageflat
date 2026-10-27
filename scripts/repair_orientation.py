#!/usr/bin/env python3
"""Orientation repair combining two independent signals.

* docTR's page-orientation classifier (MobileNetV3) — strong on ordinary pages,
  but it can return the same class regardless of rotation on very image-heavy
  children's pages;
* OCR word confidence at the four candidate rotations — strong whenever there is
  machine-readable text, weak on handwriting.

A file is only rotated when one of them is confident: classifier confidence
>= 0.9, or the best OCR rotation beats the current one by 3x with a non-trivial
absolute score.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

_PRED = None


def _init(device: str) -> None:
    global _PRED
    from doctr.models import page_orientation_predictor

    _PRED = page_orientation_predictor(pretrained=True)
    if device != "cpu":
        _PRED.model = _PRED.model.to(device)


def decide(img: np.ndarray) -> tuple[int, dict]:
    """Return (clockwise_degrees_to_apply, evidence)."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pageflat.orientation import _ocr_score

    out = _PRED([cv2.cvtColor(img, cv2.COLOR_BGR2RGB)])
    cls = int(np.asarray(out[0]).ravel()[0])
    conf = float(np.asarray(out[2]).ravel()[0])
    doctr_rot = cls * 90
    if conf >= 0.9:
        return doctr_rot, {"source": "doctr", "cls": cls, "conf": round(conf, 3)}

    variants = {0: img,
                90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
                180: cv2.rotate(img, cv2.ROTATE_180),
                270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    scores = {a: _ocr_score(v) for a, v in variants.items()}
    best_ocr = max(scores, key=scores.get)
    s_best, s_now = scores[best_ocr], scores[0]
    if s_best >= 3 * max(s_now, 1) and s_best > 2000:
        return (360 - best_ocr) % 360, {"source": "ocr", "cls": cls, "conf": round(conf, 3),
                                        "ocr_best": best_ocr,
                                        "ocr_scores": {k: round(v) for k, v in scores.items()}}
    return 0, {"source": "none", "cls": cls, "conf": round(conf, 3),
               "ocr_best": best_ocr, "ocr_scores": {k: round(v) for k, v in scores.items()}}


def _job(path_str: str) -> dict:
    path = Path(path_str)
    img = cv2.imread(str(path))
    if img is None:
        return {"file": path.name, "rotated": 0, "error": "unreadable"}
    deg, ev = decide(img)
    if deg:
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}[deg]
        cv2.imwrite(str(path), cv2.rotate(img, code), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return {"file": path.name, "rotated": deg, **ev}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    root = Path(args.scans)
    files = ([l.strip() for l in Path(args.scans).read_text().splitlines() if l.strip()]
             if root.is_file() else [str(p) for p in sorted(root.glob("*.png"))])
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                                 initargs=(args.device,)) as pool:
            rows = list(pool.map(_job, files, chunksize=4))
    else:
        _init(args.device)
        rows = [_job(f) for f in files]

    rotated = sum(1 for r in rows if r.get("rotated"))
    by_src = {}
    for r in rows:
        by_src[r.get("source", "?")] = by_src.get(r.get("source", "?"), 0) + 1
    print(f"{len(rows)} scans, {rotated} rotated, evidence {by_src}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
