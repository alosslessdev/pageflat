#!/usr/bin/env python3
"""Post-process a folder of flattened scans.

Runs the content gate over finished scans, splits two-page spreads that were
kept whole, and writes a manifest. Useful when the expensive dewarping has
already been paid for and only cheap decisions remain.

    python scripts/finalize_scans.py scans/ --json scans/manifest.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageflat import quality, spine  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-split", action="store_true")
    ap.add_argument("--min-ink", type=float, default=0.0025)
    ap.add_argument("--min-lines", type=int, default=3)
    args = ap.parse_args(argv)

    root = Path(args.scans)
    rejected_dir = root / "_rejected"
    manifest = []
    for path in sorted(root.glob("*.png")):
        img = cv2.imread(str(path))
        if img is None:
            continue
        ok, metrics = quality.looks_like_page(img, args.min_ink, args.min_lines)
        entry = {"file": path.name, "status": "ok" if ok else "rejected", **metrics}
        if not ok:
            rejected_dir.mkdir(exist_ok=True)
            shutil.move(str(path), rejected_dir / path.name)
            manifest.append(entry)
            continue
        if not args.no_split:
            pages = spine.split_spread(img)
            if len(pages) == 2:
                path.unlink()
                for i, page in enumerate(pages):
                    out = root / f"{path.stem}_p{i + 1}.png"
                    cv2.imwrite(str(out), page, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                entry["split"] = True
        manifest.append(entry)
    print(f"{sum(1 for m in manifest if m['status'] == 'ok')} ok, "
          f"{sum(1 for m in manifest if m['status'] != 'ok')} rejected", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
