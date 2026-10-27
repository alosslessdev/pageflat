#!/usr/bin/env python3
"""Measure how warped / clipped every finished scan is.

Two defects, two measurements:

* **warp** — the OCR already told us where each text line is. Inside that band
  the ink moves up and down across the line: its bow. A flat page has a bow of
  a pixel or two; a curled page bends by tens.
* **clipped** — ink that runs into the page border means the crop cut into the
  document, and a text line whose ink stops dead at the border is the signature.

    python scripts/audit_warp.py scans/ --json scans/warp_audit.jsonl [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORK_SIDE = 1600


def _ink(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 14)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    h, w = bw.shape
    keep = np.zeros(n, np.uint8)
    for i in range(1, n):
        _, _, cw, ch, area = stats[i]
        if area >= 4 and ch <= max(4, int(0.03 * h)) and cw <= 0.45 * w:
            keep[i] = 1
    return (keep[labels] * 255).astype(np.uint8)


def _bow(ink: np.ndarray, box) -> tuple[float, int]:
    x, y, w, h = (int(v) for v in box)
    x, y = max(0, x), max(0, y)
    band = ink[y:y + h, x:x + w]
    if band.size == 0:
        return 0.0, 0
    cols = np.nonzero((band > 0).any(axis=0))[0]
    if cols.size < 20:
        return 0.0, 0
    on = (band > 0).astype(np.float64)
    weights = on.sum(axis=0)[cols]
    rows = np.arange(band.shape[0], dtype=np.float64)[:, None]
    centroid = (on * rows).sum(axis=0)[cols] / np.maximum(1.0, weights)
    grid = np.arange(cols.min(), cols.max() + 1, dtype=np.float64)
    smooth = np.interp(grid, cols, centroid)
    k = max(5, (grid.size // 10) | 1)
    smooth = cv2.GaussianBlur(smooth.reshape(1, -1), (k, 1), 0).ravel()
    slope, icpt = np.polyfit(grid, smooth, 1)
    resid = smooth - (slope * grid + icpt)
    return float(resid.max() - resid.min()), int(grid.size)


def audit(job: tuple[str, str | None]) -> dict:
    page, ocr_json = job
    img = cv2.imread(page, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"page": Path(page).name, "error": "unreadable"}
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIDE / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    ink = _ink(small)
    sh, sw = small.shape[:2]
    boxes = []
    touches = 0
    if ocr_json and Path(ocr_json).exists():
        data = json.loads(Path(ocr_json).read_text())
        # the OCR ran on its own working resolution, not on the scan we opened
        sx = w / float(data.get("width") or w)
        sy = h / float(data.get("height") or h)
        for line in data.get("lines", []):
            if line.get("conf", 0) < 45:
                continue
            x, y, bw, bh = line["bbox"]
            if bw * sx < 0.15 * w:
                continue
            boxes.append([x * sx * scale, y * sy * scale,
                          bw * sx * scale, bh * sy * scale])
            # a real text line touching the image border is a crop that cut in
            bx, by = x * sx * scale, y * sy * scale
            bw_s, bh_s = bw * sx * scale, bh * sy * scale
            if (bx <= 2 or by <= 2 or bx + bw_s >= sw - 2 or by + bh_s >= sh - 2) \
                    and bw_s > 0.2 * sw:
                touches += 1
    measured = []
    for box in boxes:
        bow, span = _bow(ink, box)
        if span >= 20:
            measured.append((box[3], bow, bow / max(1.0, box[3])))
    if measured:
        heights = np.array([m[0] for m in measured], dtype=np.float64)
        keep = [m for m in measured if m[0] <= 2.0 * float(np.median(heights))]
        bow_px = float(np.median([m[1] for m in keep])) if keep else 0.0
        ratio = float(np.median([m[2] for m in keep])) if keep else 0.0
    else:
        bow_px, ratio = 0.0, 0.0
    edge = max(2, int(0.004 * sw))
    edge_density = float(max(
        (ink[:, :edge] > 0).mean(), (ink[:, -edge:] > 0).mean(),
        (ink[:edge, :] > 0).mean(), (ink[-edge:, :] > 0).mean()))
    return {
        "page": Path(page).name,
        "width": w,
        "height": h,
        "lines": len(boxes),
        "bow_px": round(bow_px, 2),
        "bow_frac": round(bow_px / sh, 5),
        "bow_ratio": round(ratio, 3),
        "edge_density": round(edge_density, 4),
        "warped": bool(len(boxes) >= 8 and ratio > 0.5),
        "clipped": bool(touches >= 1),
        "border_lines": touches,
    }


def _init() -> None:
    cv2.setNumThreads(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--json", default=None)
    ap.add_argument("-o", "--out", default=None, help="write the audit here (jsonl)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    ocr = scans / "ocr"
    pages = sorted(scans.glob("*.png"))
    if args.limit:
        pages = pages[:args.limit]
    jobs = [(str(p), str(ocr / (p.stem + ".json"))) for p in pages]
    out = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        for i, row in enumerate(pool.map(audit, jobs, chunksize=8), 1):
            out.append(row)
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", file=sys.stderr, flush=True)
    warped = sum(1 for r in out if r.get("warped"))
    clipped = sum(1 for r in out if r.get("clipped"))
    print(f"{len(out)} scans: {warped} warped, {clipped} with ink at the border")
    dest = args.out or args.json
    if dest:
        Path(dest).write_text("\n".join(json.dumps(r) for r in out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
