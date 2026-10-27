#!/usr/bin/env python3
"""Find photographed pages in a large photo library.

Scores every JPEG for "page-likeness" (one dominant, flat surface filling the
frame) so that a huge camera roll can be triaged before running the expensive
rectification pipeline.

Usage:
    python scripts/find_pages.py /run/media/user/PHONE --min-mp 60 \
        --out pages.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat

Image.MAX_IMAGE_PIXELS = None


def jpeg_size(path: str) -> tuple[int, int] | None:
    """Read JPEG dimensions from the SOF marker (no full decode)."""
    with open(path, "rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            return None
        while True:
            b = fh.read(1)
            if not b:
                return None
            if b != b"\xff":
                continue
            m = fh.read(1)
            while m == b"\xff":
                m = fh.read(1)
            if not m:
                return None
            if m[0] in (0xD8, 0x01) or 0xD0 <= m[0] <= 0xD7:
                continue
            ln = int.from_bytes(fh.read(2), "big")
            if m[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                d = fh.read(5)
                if len(d) < 5:
                    return None
                h = int.from_bytes(d[1:3], "big")
                w = int.from_bytes(d[3:5], "big")
                return w, h
            fh.seek(ln - 2, 1)


def page_score(path: str) -> float | None:
    """How much of the frame is one flat, low-saturation surface (a page)."""
    try:
        im = Image.open(path)
        im.draft("RGB", (256, 256))
        im = im.convert("RGB")
        im = im.resize((128, 128), Image.BILINEAR) if im.size[0] > 128 else im
        px = np.asarray(im, dtype=np.uint8).reshape(-1, 3)
        q = Counter((int(r) >> 4, int(g) >> 4, int(b) >> 4) for r, g, b in px)
        dom, _ = q.most_common(1)[0]
        dc = np.array([dom[0] * 16 + 8, dom[1] * 16 + 8, dom[2] * 16 + 8], np.int16)
        near = (np.abs(px.astype(np.int16) - dc) < 48).all(axis=1).mean()
        return float(near)
    except Exception:
        return None


def iter_images(roots: list[str], exts=(".jpg", ".jpeg")):
    seen = set()
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dev = os.stat(dirpath).st_dev
            dirnames[:] = [d for d in dirnames
                           if os.stat(os.path.join(dirpath, d)).st_dev == dev]
            for name in filenames:
                if name.lower().endswith(exts):
                    p = os.path.join(dirpath, name)
                    if p not in seen:
                        seen.add(p)
                        yield p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--min-mp", type=float, default=60.0)
    ap.add_argument("--min-score", type=float, default=0.75)
    ap.add_argument("--out", default="pages.jsonl")
    args = ap.parse_args(argv)

    total = kept = 0
    with open(args.out, "w") as out:
        for path in iter_images(args.roots):
            total += 1
            size = jpeg_size(path)
            if not size:
                continue
            w, h = size
            mp = w * h / 1e6
            if mp < args.min_mp:
                continue
            score = page_score(path)
            if score is None or score < args.min_score:
                continue
            kept += 1
            out.write(json.dumps({"path": path, "width": w, "height": h,
                                  "megapixels": round(mp, 2),
                                  "page_score": round(score, 3)}) + "\n")
            print(f"{score:.3f}  {mp:5.1f}MP  {path}", flush=True)
    print(f"scanned {total}, kept {kept} -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
