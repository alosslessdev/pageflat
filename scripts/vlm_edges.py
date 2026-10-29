#!/usr/bin/env python3
"""Flatten a page from the four edges a vision model traces on it.

The detector can only place control points where there is ink, which is why the
thin-plate spline extrapolates across the blank regions of a bowed page and
loses text there. A vision model can trace the *boundary* of the text area
everywhere, including the parts with no ink.

Four questions, one curve each: the top edge left-to-right, the bottom edge
left-to-right, the left edge top-to-bottom, the right edge top-to-bottom. Each
answer is a handful of points on a 0-1000 grid. The correspondences are then
immediate, because a flattened page is exactly a rectangle:

    every point on the top curve   -> its own x, one common y
    every point on the left curve  -> one common x, its own y

and the TPS that satisfies both is the flattening.

    python scripts/vlm_edges.py page.png --url http://127.0.0.1:8100 --out fixed.png
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

CURVES = {
    "top": "the top edge of the area that contains text (not the paper edge)",
    "bottom": "the bottom edge of the area that contains text",
    "left": "the left edge of the area that contains text",
    "right": "the right edge of the area that contains text",
}

QUESTION = (
    "This is a photograph of a printed page. Trace {what}. "
    "Give {n} points along it, in order {order}. Use a 0-1000 grid where (0,0) is the "
    "top-left of the image and (1000,1000) the bottom-right. Answer only with the points, "
    "one per line, as `x,y`."
)


def encode(path: Path, side: int = 896) -> str:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = side / float(max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def ask(url: str, image_b64: str, what: str, n: int, order: str) -> str:
    import requests

    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": QUESTION.format(what=what, n=n, order=order)},
            {"type": "image_url", "image_url": {"url": image_b64}},
        ]}],
        # the server caps thinking at --reasoning-budget; the answer comes after it
        "max_tokens": 2400,
        "temperature": 0.0,
    }
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def points_of(text: str, w: int, h: int) -> list[tuple[float, float]]:
    out = []
    for m in re.finditer(r"(\d{1,4})\s*[,;]\s*(\d{1,4})", text):
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            out.append((x / 1000.0 * w, y / 1000.0 * h))
    return out


def flatten(img: np.ndarray, curves: dict[str, list[tuple[float, float]]],
            smoothing: float = 8.0) -> tuple[np.ndarray, dict]:
    from scipy.interpolate import RBFInterpolator

    h, w = img.shape[:2]
    src, dst = [], []
    if curves.get("top"):
        ty = float(np.median([p[1] for p in curves["top"]]))
        for x, _y in curves["top"]:
            src.append((x, _y)); dst.append((x, ty))
    if curves.get("bottom"):
        by = float(np.median([p[1] for p in curves["bottom"]]))
        for x, _y in curves["bottom"]:
            src.append((x, _y)); dst.append((x, by))
    if curves.get("left"):
        lx = float(np.median([p[0] for p in curves["left"]]))
        for _x, y in curves["left"]:
            src.append((_x, y)); dst.append((lx, y))
    if curves.get("right"):
        rx = float(np.median([p[0] for p in curves["right"]]))
        for _x, y in curves["right"]:
            src.append((_x, y)); dst.append((rx, y))
    if len(src) < 8:
        return img, {"corrected": False, "pairs": len(src)}
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    back = RBFInterpolator(dst, src, kernel="thin_plate_spline",
                           smoothing=smoothing, degree=1)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    q = np.stack([gx.ravel(), gy.ravel()], -1)
    s_pts = back(q).reshape(h, w, 2)
    map_x = np.clip(s_pts[:, :, 0], 0, w - 1).astype(np.float32)
    map_y = np.clip(s_pts[:, :, 1], 0, h - 1).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "pairs": int(len(src))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--smoothing", type=float, default=8.0)
    args = ap.parse_args(argv)

    page = Path(args.page)
    img = cv2.imread(str(page))
    if img is None:
        print("cannot read", page, file=sys.stderr)
        return 2
    h, w = img.shape[:2]
    b64 = encode(page)
    curves, raw = {}, {}
    for name, what in CURVES.items():
        order = "left to right" if name in ("top", "bottom") else "top to bottom"
        raw[name] = ask(args.url, b64, what, args.n, order)
        curves[name] = points_of(raw[name], w, h)
        print(f"  {name:6} {len(curves[name])} points")
    out, info = flatten(img, curves, args.smoothing)
    if info.get("corrected"):
        cv2.imwrite(args.out, out)
    print(json.dumps(info), "->", args.out if info.get("corrected") else "(unchanged)")
    if args.json:
        Path(args.json).write_text(json.dumps({"raw": raw, "curves": curves}, indent=1))
    return 0 if info.get("corrected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
