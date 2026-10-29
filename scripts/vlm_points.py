#!/usr/bin/env python3
"""Ask a vision model for control points, then let the pixels refine them.

The detector only sees structure where there is ink: on a page whose bowed
region is blank, a photograph, or a marginal drawing, the thin-plate spline has
nothing to hold on to and either does nothing or extrapolates wildly. A vision
model can name points anywhere on the sheet, which is what the control-point
method needs.

Flow: page -> VLM proposes N points in a 0-1000 grid -> each proposal is snapped
to the nearest real gradient (a coarse answer is enough, the pixels sharpen it)
-> the points become TPS correspondences with their flattened positions.

    python scripts/vlm_points.py page.png --url http://127.0.0.1:8100 --n 30 \
        --out page_points.json
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

from pageflat import points as pts_mod

PROMPT = """This is a photograph of a printed page that is curved.

Give me {n} control points on the page surface. Use a 0-1000 grid where (0,0) is the
top-left and (1000,1000) the bottom-right of the image. Place them:
- along the top edge, the bottom edge, the left margin and the right margin of the text,
- and MORE of them where the page bends or curls (put extra points along the curvature).

Answer only with one point per line, exactly as `x,y`, no words."""


def encode(path: Path, side: int = 896) -> str:
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    s = side / float(max(h, w))
    small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def ask(url: str, image_path: Path, n: int, timeout: float = 300.0) -> str:
    import requests

    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(n=n)},
            {"type": "image_url", "image_url": {"url": encode(image_path)}},
        ]}],
        # The model must think to localise (its reasoning contains the actual
        # coordinates) but will think forever if left alone: the server runs with
        # --reasoning-budget 1500, so give it that much thought plus room to answer.
        "max_tokens": 2400,
        "temperature": 0.0,
    }
    r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def parse(text: str, w: int, h: int) -> list[tuple[float, float]]:
    out = []
    for m in re.finditer(r"(\d{1,4})\s*[,;]\s*(\d{1,4})", text):
        x, y = int(m.group(1)), int(m.group(2))
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            out.append((x / 1000.0 * w, y / 1000.0 * h))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", default=None, help="write the rectified page here")
    args = ap.parse_args(argv)

    page = Path(args.page)
    img = cv2.imread(str(page))
    if img is None:
        print("cannot read page", file=sys.stderr)
        return 2
    h, w = img.shape[:2]
    raw = ask(args.url, page, args.n)
    proposed = parse(raw, w, h)
    snapped = pts_mod.snap(proposed, img, radius=max(12, int(0.01 * max(h, w))))
    print(f"proposed {len(proposed)} points, snapped {len(snapped)}")
    Path(args.out).write_text(json.dumps({"raw": raw, "points": snapped}, indent=1))
    if args.apply and len(snapped) >= 6:
        # targets: each point keeps its own position but is pulled onto the
        # nearest straight grid line, which is what "rectangular" means here
        xs = np.array([p[0] for p in snapped])
        ys = np.array([p[1] for p in snapped])
        src, dst = [], []
        for (x, y) in snapped:
            src.append((x, y))
            dst.append((x, float(np.median(ys))))
        from scipy.interpolate import RBFInterpolator
        back = RBFInterpolator(np.array(dst), np.array(src),
                               kernel="thin_plate_spline", smoothing=1.0, degree=1)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        q = np.stack([gx.ravel(), gy.ravel()], -1)
        s_pts = back(q).reshape(h, w, 2)
        out = cv2.remap(img, s_pts[:, :, 0].astype(np.float32),
                        s_pts[:, :, 1].astype(np.float32), cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(args.apply, out)
        print("wrote", args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
