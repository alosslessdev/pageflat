#!/usr/bin/env python3
"""Pick the straightest version of every scan, across all the methods.

For each scan the candidates are: the scan itself, DocTr's flow, DewarpNet's
3D map, UVDoc's grid, and the monocular-depth correction. Each is scored by
the rule (how far the text blocks are from rectangles) and by OCR, and the
page takes the straightest candidate that does not lose its text.

    python scripts/rectify_scans.py scans/ [--only-failing] [--apply]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2, numpy as np
from pageflat.detect import text_lines
from pageflat.blocks import blocks_of
from pageflat.pipeline import ocr_quality
from pageflat.models import DocTrPipeline, DewarpNetDewarper, UVDocDewarper
from pageflat import depthwarp


def drift(img):
    b = blocks_of(text_lines(img))
    if not b:
        return None
    e = sorted(x.rect_error for x in b)
    return float(e[len(e) // 2])


def candidates(img, doctr, dewnet, uvdoc):
    out = {"none": img}
    for name, fn in (("doctr", doctr.rectify), ("dewnet", dewnet.rectify),
                     ("uvdoc", uvdoc.rectify)):
        try:
            out[name] = fn(img)
        except Exception:  # noqa: BLE001
            pass
    try:
        cand, info = depthwarp.rectify(img)
        if info.get("corrected"):
            out["depth"] = cand
    except Exception:  # noqa: BLE001
        pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only-failing", action="store_true",
                    help="skip pages whose blocks are already rectangles")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--conf-budget", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pages-from", default=None,
                    help="JSONL of pages with a drift_frac, to avoid re-measuring all")
    ap.add_argument("--journal", default=None)
    args = ap.parse_args(argv)
    scans = Path(args.scans)
    journal = Path(args.journal) if args.journal else scans / "rectify_journal.jsonl"
    if args.pages_from:
        rows = [json.loads(l) for l in open(args.pages_from) if l.strip()]
        keep = [r["page"] for r in rows if r.get("drift_frac", 0) > args.threshold / 100.0]
        pages = [scans / p for p in keep if (scans / p).exists()]
    else:
        pages = sorted(scans.glob("*.png"))
    if args.limit:
        pages = pages[:args.limit]
    doctr = DocTrPipeline(device="cuda")
    dewnet = DewarpNetDewarper(device="cuda")
    uvdoc = UVDocDewarper()
    changed, skipped = [], 0
    with open(journal, "w") as fh:
        for i, page in enumerate(pages, 1):
            img = cv2.imread(str(page))
            if img is None:
                continue
            base = drift(img)
            if args.only_failing and (base is None or base <= args.threshold):
                skipped += 1
                continue
            cands = candidates(img, doctr, dewnet, uvdoc)
            scored = []
            for name, cand in cands.items():
                d = drift(cand)
                c = ocr_quality(cand)
                scored.append((name, d, c))
            confs = [s[2] for s in scored]
            best_conf = max(confs)
            pool = [s for s in scored if s[2] >= best_conf - args.conf_budget and s[1] is not None]
            pick = min(pool, key=lambda s: s[1]) if pool else None
            row = {"page": page.name, "base": base, "scored": scored,
                   "pick": pick[0] if pick else None,
                   "drift": pick[1] if pick else None,
                   "conf": pick[2] if pick else None}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if pick and pick[0] != "none" and (base is None or pick[1] < base * 0.8):
                if args.apply:
                    cv2.imwrite(str(page), cands[pick[0]])
                changed.append(page.name)
            if i % 50 == 0:
                print(f"{i}/{len(pages)} changed {len(changed)}", file=sys.stderr, flush=True)
    if args.apply:
        (scans / "rectified_pages.txt").write_text("\n".join(changed) + "\n")
    print(f"{len(changed)} pages improved, {skipped} skipped as already rectangular")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
