#!/usr/bin/env python3
"""Score a directory of finished pages on every defect that has been found.

One command per iteration, one table, so a fix can be judged instead of argued
about. Each metric is independent and each one has a reason to exist:

* ``bow``      — text lines must be straight; the flattening goal (median <= 1 % of width,
                 nothing above 3 %).
* ``tilt``     — vertical rules must stand up; the defect a text-line ruler cannot see.
* ``band``     — no strip of table, floor or neighbouring page left at the frame edge.
* ``cut``      — no detected text line interrupted by the frame (content cut away).
* ``rot``      — the page is upright.

    python scripts/score_pages.py out100/ [--json scores.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from pageflat.charlines import detect_lines
from pageflat.orientation import detect_rotation
from pageflat.perspective import _paper_like
from pageflat.vlines import vertical_drift


def band_frac(im: np.ndarray, frac: float = 0.03) -> float:
    h, w = im.shape[:2]
    b = max(4, int(frac * min(h, w)))
    p = _paper_like(im)
    return max(1.0 - float(p[:b].mean()), 1.0 - float(p[-b:].mean()),
               1.0 - float(p[:, :b].mean()), 1.0 - float(p[:, -b:].mean()))


def bow_pct(im: np.ndarray, min_chars: int = 6) -> float | None:
    vals = []
    for l in detect_lines(im):
        if len(l.boxes) < min_chars:
            continue
        xs = np.array([b[0] + b[2] / 2 for b in l.boxes], float)
        ys = np.array([b[1] + b[3] for b in l.boxes], float)
        A = np.stack([xs, np.ones_like(xs)], 1)
        c, *_ = np.linalg.lstsq(A, ys, rcond=None)
        vals.append(float(np.median(np.abs(ys - A @ c))))
    if len(vals) < 5:
        return None
    return 100.0 * st.median(vals) / im.shape[1]


def cut_lines(im: np.ndarray, edge: int = 6) -> int:
    """Detected lines that touch the frame: their box is clipped by it."""
    n = 0
    for l in detect_lines(im):
        if len(l.boxes) < 3:
            continue
        if (l.left <= edge or l.right >= im.shape[1] - edge
                or min(b[1] for b in l.boxes) <= edge
                or max(b[1] + b[3] for b in l.boxes) >= im.shape[0] - edge):
            n += 1
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages")
    ap.add_argument("--json", default=None)
    ap.add_argument("--side", type=int, default=1400)
    args = ap.parse_args(argv)

    rows = []
    for f in sorted(Path(args.pages).glob("*.png")):
        im = cv2.imread(str(f))
        if im is None:
            continue
        s = args.side / max(im.shape[:2])
        small = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
        # 1600 px, INTER_LINEAR: the scale and resample the pipeline itself uses to
        # decide orientation, so the score cannot disagree with the decision.
        s2 = 1600 / max(im.shape[:2])
        orient_small = cv2.resize(im, (int(im.shape[1] * s2), int(im.shape[0] * s2)),
                                  interpolation=cv2.INTER_LINEAR) if s2 < 1 else im
        v = vertical_drift(small)
        rows.append({
            "page": f.name,
            "bow": bow_pct(im),
            "tilt": None if v["tilt_deg"] is None else round(v["tilt_deg"], 2),
            "rules": v["rules"],
            "band": round(band_frac(im), 3),
            "cut_lines": cut_lines(im),
            "rot": detect_rotation(orient_small),
        })

    bows = [r["bow"] for r in rows if r["bow"] is not None]
    tilts = [abs(r["tilt"]) for r in rows if r["tilt"] is not None]
    print(f"pages                {len(rows)}")
    print(f"bow median           {st.median(bows):.2f} %  (measured {len(bows)})" if bows else "bow n/a")
    print(f"bow above 3 %        {sum(1 for x in bows if x > 3)}")
    if tilts:
        print(f"tilt median |deg|    {st.median(tilts):.2f}   p90 {np.percentile(tilts, 90):.2f}")
        print(f"tilt above 1 deg     {sum(1 for x in tilts if x > 1)}")
    print(f"background band >.5  {sum(1 for r in rows if r['band'] > 0.5)}")
    print(f"pages with cut lines {sum(1 for r in rows if r['cut_lines'] > 0)}")
    print(f"pages misoriented    {sum(1 for r in rows if r['rot'] != 0)}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
