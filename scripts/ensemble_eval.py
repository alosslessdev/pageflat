#!/usr/bin/env python3
"""Score every dewarping resource on the benchmark and pick per page.

The candidates come from what the session assembled: DocTr's flow, UVDoc,
DewarpNet's 3D map, and the monocular-depth correction. Each is scored by the
rectangle rule (how far text blocks are from rectangles) and by OCR
confidence, and the page takes the straightest candidate that keeps its text.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2, numpy as np
from pageflat.detect import text_lines
from pageflat.blocks import blocks_of
from pageflat.pipeline import ocr_quality, Settings, process_image
from pageflat.models import DocTrPipeline, DewarpNetDewarper, UVDocDewarper
from pageflat import depthwarp


def drift(img):
    b = blocks_of(text_lines(img))
    if not b:
        return None
    e = sorted(x.rect_error for x in b)
    return float(e[len(e) // 2])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--pages", type=int, default=8)
    args = ap.parse_args(argv)
    D = Path(args.scans)
    bench = json.loads((D / "benchmark.json").read_text())["pages"][:args.pages]
    doctr = DocTrPipeline(device="cuda")
    dewnet = DewarpNetDewarper(device="cuda")
    uvdoc = UVDocDewarper()
    ks = (-400., -300., -200., -120., -80., -40., 40., 80., 120., 200., 300., 400.)
    rows = []
    for page in bench:
        img = cv2.imread(str(D / page))
        if img is None:
            continue
        cands = {"none": img}
        for name, fn in (("doctr", doctr.rectify), ("dewnet", dewnet.rectify),
                         ("uvdoc", uvdoc.rectify)):
            try:
                cands[name] = fn(img)
            except Exception:
                pass
        try:
            out, info = depthwarp.rectify(img, k_values=ks)
            if info.get("corrected"):
                cands["depth"] = out
        except Exception:
            pass
        scored = []
        for name, cand in cands.items():
            d = drift(cand)
            c = ocr_quality(cand)
            scored.append((name, None if d is None else round(d, 1), round(c, 1)))
        best_conf = max(s[2] for s in scored)
        pool = [s for s in scored if s[2] >= best_conf - 1.5 and s[1] is not None]
        pick = min(pool, key=lambda s: s[1]) if pool else scored[0]
        rows.append({"page": page, "scored": scored, "pick": pick[0]})
        print(f"{page[:24]:24} pick={pick[0]:7} drift={str(pick[1]):>6} conf={pick[2]:>5}  "
              + " ".join(f"{n}:{d}/{c}" for n, d, c in scored))
    picks = {}
    for r in rows:
        picks[r["pick"]] = picks.get(r["pick"], 0) + 1
    print("picks:", picks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
