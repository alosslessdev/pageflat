#!/usr/bin/env python3
"""How many pages fail the rule "text blocks are rectangles".

Measures, for every page with OCR lines, how far the left and right edges of
its paragraph blocks wander from a vertical line, as a percentage of the page
width. No image is opened: the OCR boxes already carry the geometry.

    python scripts/audit_blocks.py scans/ [-o blocks_audit.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageflat.blocks import load_lines, page_rect_error  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="drift, as a fraction of the page width, that makes a block warped")
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    rows = []
    for j in sorted((scans / "ocr").glob("*.json")):
        lines, width, _ = load_lines(j)
        if len(lines) < 8:
            continue
        r = page_rect_error(lines, scale=1.0)
        if r["blocks"] < 1:
            continue
        rows.append({"page": j.stem + ".png", "blocks": r["blocks"],
                     "drift_frac": round(r["error"] / width, 4),
                     "warped": r["error"] / width > args.threshold})
    warped = sum(1 for r in rows if r["warped"])
    print(f"{len(rows)} pages measured, {warped} fail the rectangle rule "
          f"({100.0 * warped / max(1, len(rows)):.0f} %)")
    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
