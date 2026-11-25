"""Tiled page-edge annotation.

A single view of a phone photograph is about 1100 px wide when it goes to a vision model,
so one pixel of error there is roughly four in the original. That error is the whole
reason the control-point outline trades background for content: the walk is accurate to
about a character's height, so it sometimes sits inside the print and sometimes outside
the paper.

Tiling fixes it by giving the model the page edge at high magnification. Each tile is
cropped from the full-resolution original and upscaled, the model marks points along
whatever page edge it can see in that tile, and `merge_tiles` maps those points back into
full-image coordinates and rebuilds one ordered walk around the page.

The two functions are independent of any model: `make_tiles` writes tiles and a manifest,
`merge_tiles` consumes `{tile: [[x, y], ...]}` in tile-normalised coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def make_tiles(image_path: str, out_dir: str, cols: int = 2, rows: int = 2,
               overlap: float = 0.12, tile_side: int = 900) -> dict:
    """Cut `image_path` into overlapping tiles and write them plus a manifest."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    h, w = img.shape[:2]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tw = 1.0 / cols + overlap
    th = 1.0 / rows + overlap
    manifest = {"source": str(image_path), "size": [w, h], "tiles": {}}
    for r in range(rows):
        for c in range(cols):
            fx = min(1.0 - tw, c / cols)
            fy = min(1.0 - th, r / rows)
            x0, y0 = int(fx * w), int(fy * h)
            x1, y1 = int((fx + tw) * w), int((fy + th) * h)
            crop = img[y0:y1, x0:x1]
            scale = tile_side / max(crop.shape[:2])
            tile = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                              interpolation=cv2.INTER_AREA)
            name = f"{Path(image_path).stem}__r{r}c{c}.jpg"
            cv2.imwrite(str(out / name), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest["tiles"][name] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    return manifest


def _to_full(points, tile, size) -> np.ndarray:
    """Tile-normalised points -> full-image normalised points."""
    w, h = size
    x0, y0, x1, y1 = tile["x0"], tile["y0"], tile["x1"], tile["y1"]
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    xs = (x0 + pts[:, 0] * (x1 - x0)) / w
    ys = (y0 + pts[:, 1] * (y1 - y0)) / h
    return np.stack([xs, ys], axis=1)


def merge_tiles(tile_points: dict, manifest: dict, sides: int = 24) -> np.ndarray:
    """Turn per-tile points into one clockwise walk around the page.

    Points are scattered over the page but each belongs to one of the four sides, so the
    walk is rebuilt by assigning every point to its nearest side of a coarse bounding
    box, sorting each side along its own direction, and concatenating. Concatenation
    order is what the warp cares about; even spacing is restored by the reference points
    in `edges.dewarp_with_control_points`, so uneven sampling is harmless.
    """
    size = manifest["size"]
    allpts = []
    for name, pts in tile_points.items():
        tile = manifest["tiles"].get(name)
        if not tile or len(pts) < 1:
            continue
        allpts.append(_to_full(pts, tile, size))
    if not allpts:
        raise ValueError("no tile points")
    pts = np.vstack(allpts)
    pts = pts[(pts[:, 0] > -0.05) & (pts[:, 0] < 1.05)]
    pts = pts[(pts[:, 1] > -0.05) & (pts[:, 1] < 1.05)]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = pts.mean(axis=0)
    groups = {k: [] for k in ("top", "right", "bottom", "left")}
    for x, y in pts:
        d = {"top": abs(y - lo[1]), "bottom": abs(y - hi[1]),
             "left": abs(x - lo[0]), "right": abs(x - hi[0])}
        groups[min(d, key=d.get)].append((x, y))
    # keep the walk clockwise from the top-left corner
    order = (("top", lambda p: p[0]), ("right", lambda p: p[1]),
             ("bottom", lambda p: -p[0]), ("left", lambda p: -p[1]))
    walk = []
    for side, key in order:
        walk.extend(sorted(groups[side], key=key))
    if len(walk) < 8:
        raise ValueError(f"only {len(walk)} points survived: {groups}")
    # resample to `sides` evenly spaced points along the closed walk
    arr = np.asarray(walk, np.float64)
    closed = np.vstack([arr, arr[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / max(total, 1e-9)
    want = np.linspace(0.0, 1.0, sides, endpoint=False)
    return np.stack([np.interp(want, cum, closed[:, 0]),
                     np.interp(want, cum, closed[:, 1])], axis=1)


def load_manifest(path: str) -> dict:
    return json.loads(Path(path).read_text())
