#!/usr/bin/env python3
"""Crop a scan to its printed area when the rest of the frame is surface.

A lot of the corpus is a sheet lying on cloth, a table, or a pile of papers.
The flattening keeps that surface, and in the PDF it reads as a warped page
even though the sheet itself is fine. The OCR already says where the print is;
the crop is taken only when

* the printed area is less than 60 % of the frame (else there is nothing to
  remove),
* the area outside it carries almost no dark ink (else something printed would
  be cut away), and
* enough lines were recognised for the box to mean something.

    python scripts/crop_scans.py scans/ [--dry-run] [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MIN_LINES = 3
MAX_AREA = 0.60
OUTSIDE_DARK = 0.02


def _init() -> None:
    cv2.setNumThreads(1)


def one(job: tuple[str, bool]) -> dict:
    path_s, apply = job
    page = Path(path_s)
    img = cv2.imread(str(page))
    if img is None:
        return {"page": page.name, "status": "unreadable"}
    ocr = page.parent / "ocr" / (page.stem + ".json")
    if not ocr.exists():
        return {"page": page.name, "status": "no-ocr"}
    data = json.loads(ocr.read_text())
    lines = [l for l in data.get("lines", []) if l.get("conf", 0) > 40]
    if len(lines) < MIN_LINES:
        return {"page": page.name, "status": "too-few-lines"}
    h, w = img.shape[:2]
    sx, sy = w / float(data.get("width") or w), h / float(data.get("height") or h)
    x0 = min(l["bbox"][0] for l in lines) * sx
    y0 = min(l["bbox"][1] for l in lines) * sy
    x1 = max(l["bbox"][0] + l["bbox"][2] for l in lines) * sx
    y1 = max(l["bbox"][1] + l["bbox"][3] for l in lines) * sy
    area = (x1 - x0) * (y1 - y0) / float(h * w)
    if area > MAX_AREA:
        return {"page": page.name, "status": "fills-frame", "area": round(area, 3)}
    pad = int(0.03 * max(x1 - x0, y1 - y0))
    cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    cx1, cy1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    outside = np.ones((h, w), bool)
    outside[cy0:cy1, cx0:cx1] = False
    dark = float((gray[outside] < 110).mean()) if outside.any() else 0.0
    if dark > OUTSIDE_DARK:
        return {"page": page.name, "status": "ink-outside", "dark": round(dark, 4),
                "area": round(area, 3)}
    if apply:
        cv2.imwrite(str(page), img[cy0:cy1, cx0:cx1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return {"page": page.name, "status": "cropped", "area": round(area, 3),
            "size": [cx1 - cx0, cy1 - cy0], "was": [w, h]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--journal", default=None)
    args = ap.parse_args(argv)
    scans = Path(args.scans)
    journal = Path(args.journal) if args.journal else scans / "crop_journal.jsonl"
    pages = sorted(scans.glob("*.png"))
    jobs = [(str(p), args.apply) for p in pages]
    tally: dict[str, int] = {}
    cropped: list[str] = []
    with open(journal, "w") as fh, ProcessPoolExecutor(max_workers=args.workers,
                                                      initializer=_init) as pool:
        for i, row in enumerate(pool.map(one, jobs, chunksize=8), 1):
            fh.write(json.dumps(row) + "\n")
            tally[row["status"]] = tally.get(row["status"], 0) + 1
            if row["status"] == "cropped":
                cropped.append(row["page"])
            if i % 1000 == 0:
                print(f"{i}/{len(jobs)} {tally}", file=sys.stderr, flush=True)
    if args.apply:
        (scans / "cropped_pages.txt").write_text("\n".join(cropped) + "\n")
    print(f"{tally}; {len(cropped)} pages to crop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
