"""Thirty control points, denser where the page bends, driving a TPS.

The lesson from the per-page results is that a handful of corners is not enough
and a dense free field is not trustworthy; what works is a set of *measured*
points: the four corners, the ends of every text line (which say where each
line should start and end), and extra samples on the lines that are most bent,
because that is where the surface curves fastest.

A VLM can propose these points; so can the detector. This module takes either,
snaps each proposal to the nearest real structure (so a coarse VLM answer is
refined by the pixels), and fits the thin-plate spline that flattens the page.
"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np


def line_points(lines: Iterable, shape: tuple[int, int], extra_on_bowed: bool = True,
                bow_threshold: float = 0.004) -> list[tuple[float, float]]:
    """Ends of each line, plus mid-line samples on the ones that bow most."""
    h, w = shape[:2]
    pts: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in lines:
        pts.append((x0, (y0 + y1) / 2.0))
        pts.append((x1, (y0 + y1) / 2.0))
        if extra_on_bowed and (x1 - x0) > 0.3 * w:
            pts.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    return pts


def snap(points: list[tuple[float, float]], img: np.ndarray, radius: int = 24,
         samples: int = 8) -> list[tuple[float, float]]:
    """Move each point onto the nearest strong gradient within `radius`.

    A coarse annotation — from a detector, or from a VLM that names a position
    to within a percent of the image — is only useful once it lands on the
    actual ink or paper edge, so every point is pulled to the local maximum of
    gradient magnitude.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = gray.shape[:2]
    out = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x1 <= x0 or y1 <= y0:
            out.append((x, y))
            continue
        patch = mag[y0:y1, x0:x1]
        yy, xx = np.unravel_index(np.argmax(patch), patch.shape)
        peak = float(patch[yy, xx])
        if peak < 20.0:                      # nothing to snap to: keep the proposal
            out.append((x, y))
        else:
            out.append((x0 + float(xx), y0 + float(yy)))
    return out


def pairs_from_lines(lines, shape: tuple[int, int], extra_on_bowed: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """source -> target for a TPS: every line end goes to its block margin.

    Targets come from the block structure, not from the corners: the left ends
    of a block should share one x, the right ends another, and each line keeps
    its own height. That is the control-point formulation with the points taken
    from the page instead of from a person.
    """
    from .blocks import blocks_of
    h, w = shape[:2]
    src, dst = [], []
    for block in blocks_of(lines):
        lefts = np.array([l[0] for l in block.lines])
        rights = np.array([l[2] for l in block.lines])
        tl, tr = float(np.median(lefts)), float(np.median(rights))
        for l in block.lines:
            y = (l[1] + l[3]) / 2.0
            src.append((l[0], y))
            dst.append((tl, y))
            src.append((l[2], y))
            dst.append((tr, y))
            if extra_on_bowed and (l[2] - l[0]) > 0.3 * w:
                src.append(((l[0] + l[2]) / 2.0, y))
                dst.append(((tl + tr) / 2.0, y))
    if len(src) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    return np.asarray(src, float), np.asarray(dst, float)


def rectify(img: np.ndarray, lines, smoothing: float = 1.0,
            out_size: Optional[tuple[int, int]] = None) -> tuple[np.ndarray, dict]:
    """Flatten through a TPS fitted to the detected control points."""
    from scipy.interpolate import RBFInterpolator
    from .detect import text_lines
    from .blocks import blocks_of

    if lines is None:
        lines = text_lines(img)
    src, dst = pairs_from_lines(lines, img.shape[:2])
    if src.shape[0] < 8:
        return img, {"corrected": False, "points": int(src.shape[0])}
    h, w = img.shape[:2]
    if out_size is None:
        out_size = (w, h)
    try:
        back = RBFInterpolator(dst, src, kernel="thin_plate_spline",
                               smoothing=smoothing, degree=1)
    except Exception as exc:  # noqa: BLE001
        return img, {"corrected": False, "error": repr(exc)[:60]}
    ow, oh = out_size
    xs, ys = np.meshgrid(np.arange(ow, dtype=np.float64), np.arange(oh, dtype=np.float64))
    query = np.stack([xs.ravel(), ys.ravel()], -1)
    src_pts = back(query).reshape(oh, ow, 2)
    map_x = np.clip(src_pts[:, :, 0], 0, w - 1).astype(np.float32)
    map_y = np.clip(src_pts[:, :, 1], 0, h - 1).astype(np.float32)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "points": int(src.shape[0])}
