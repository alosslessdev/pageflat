#!/usr/bin/env python3
"""The same rectangle audit, measured with the detector the loop uses.

`audit_blocks.py` reads the OCR line boxes, which are a by-product of
recognition; the replacement decision in `rewarp_all.py` measures block edges
from docTR's detector. Two rulers gave two answers (58 rewrites looked 15 %
better by one and 10 % better by the other), so the audit now has the same
instrument as the decision.

    python scripts/audit_blocks2.py scans/ -o blocks_det.jsonl --workers 4
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _init() -> None:
    import cv2
    cv2.setNumThreads(1)


def one(path_s: str) -> dict:
    import cv2
    from pageflat.detect import text_lines
    from pageflat.blocks import blocks_of
    p = Path(path_s)
    img = cv2.imread(str(p))
    if img is None:
        return {"page": p.name, "error": "unreadable"}
    blocks = blocks_of(text_lines(img))
    if not blocks:
        return {"page": p.name, "drift_frac": None, "blocks": 0}
    e = sorted(b.rect_error for b in blocks)
    return {"page": p.name, "blocks": len(blocks),
            "drift_frac": float(e[len(e) // 2] / img.shape[1])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    pages = sorted(Path(args.scans).glob("*.png"))
    if args.limit:
        pages = pages[:args.limit]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as pool:
        for i, r in enumerate(pool.map(one, [str(p) for p in pages], chunksize=4), 1):
            rows.append(r)
            if i % 500 == 0:
                print(f"{i}/{len(pages)}", file=sys.stderr, flush=True)
    vals = [r["drift_frac"] for r in rows if r.get("drift_frac") is not None]
    vals.sort()
    def q(x): return round(vals[int(x * (len(vals) - 1))] * 100, 2) if vals else None
    print(f"{len(vals)} measurable of {len(rows)}: median {q(0.5)}%  "
          f">3% {sum(1 for v in vals if v > 0.03)}  >1% {sum(1 for v in vals if v > 0.01)}")
    Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
