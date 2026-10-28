"""Dewarp a page through control points: the method a human can drive.

Xie et al., "Document Dewarping with Control Points" (CPD): a handful of point
correspondences — "this bit of the page belongs there" — define a smooth
surface, and the page is resampled through it. No training, no model, and a
person can place the points by hand on the pages where the automatic methods
fail.

Annotation format, one JSON per page:

    {"points": [[[x, y], [u, v]], ...]}

where (x, y) is a point on the photographed page and (u, v) is where it has to
end up in the flattened page. Four corners alone already rectify the page; more
points capture the curl between them.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def fit(points: Sequence[Sequence[Sequence[float]]], smoothing: float = 0.0):
    """Thin-plate spline through the control points."""
    from scipy.interpolate import RBFInterpolator

    src = np.asarray([p[0] for p in points], float)
    dst = np.asarray([p[1] for p in points], float)
    if src.shape[0] < 4:
        raise ValueError("at least four control points are needed")
    return src, dst, RBFInterpolator(src, dst, kernel="thin_plate_spline",
                                     smoothing=smoothing, degree=1)


def rectify(img: np.ndarray, points, smoothing: float = 0.0,
            out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Resample `img` so every control point lands on its target."""
    from scipy.interpolate import RBFInterpolator

    src, dst, spline = fit(points, smoothing)
    if out_size is None:
        w = int(round(np.ptp(dst[:, 0]))) or img.shape[1]
        h = int(round(np.ptp(dst[:, 1]))) or img.shape[0]
        out_size = (max(16, w), max(16, h))
    w, h = out_size
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    # the spline maps source -> target, so it is evaluated in reverse: for each
    # output pixel find the source pixel that belongs there
    back = RBFInterpolator(dst, src, kernel="thin_plate_spline",
                           smoothing=smoothing, degree=1)
    query = np.stack([xs.ravel(), ys.ravel()], -1)
    src_pts = back(query).reshape(h, w, 2)
    map_x = np.clip(src_pts[:, :, 0], 0, img.shape[1] - 1).astype(np.float32)
    map_y = np.clip(src_pts[:, :, 1], 0, img.shape[0] - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def load(path: Path):
    return json.loads(Path(path).read_text())["points"]


def annotate_interactively(path: Path, out_json: Path, grid: int = 4) -> None:
    """Click a grid of points on the page, then the same grid on a rectangle.

    Left click adds a point, `u` undoes, `s` saves, `q` quits. This needs a
    display and is meant for the handful of pages that matter, not the corpus.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"cannot read {path}")
    h, w = img.shape[:2]
    scale = min(1.0, 1400.0 / max(h, w))
    view = cv2.resize(img, (int(w * scale), int(h * scale)))
    pts: list[list[float]] = []
    win = "annotate: click points, u=undo s=save q=quit"

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x / scale, y / scale])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = view.copy()
        for i, p in enumerate(pts):
            cv2.circle(canvas, (int(p[0] * scale), int(p[1] * scale)), 5, (0, 0, 255), -1)
            cv2.putText(canvas, str(i), (int(p[0] * scale) + 6, int(p[1] * scale)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and pts:
            pts.pop()
        if key == ord("s") and pts:
            # the targets are the same points on a regular grid spanning the
            # bounding box of the clicks — good enough for a hand annotation
            arr = np.asarray(pts)
            x0, y0 = arr.min(axis=0)
            x1, y1 = arr.max(axis=0)
            n = int(np.ceil(np.sqrt(len(pts))))
            targets = [[x0 + (x1 - x0) * (i % n) / max(1, n - 1),
                        y0 + (y1 - y0) * (i // n) / max(1, n - 1)]
                       for i in range(len(pts))]
            Path(out_json).write_text(json.dumps(
                {"points": [[list(map(float, p)), list(map(float, t))]
                            for p, t in zip(pts, targets)]}, indent=1))
            print(f"saved {len(pts)} points -> {out_json}")
            break
    cv2.destroyAllWindows()
