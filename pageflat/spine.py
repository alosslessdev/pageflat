"""Two-page spread detection and spine splitting."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _column_profile(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    # downsample rows to keep it cheap
    small = cv2.resize(gray, (w, min(h, 400)), interpolation=cv2.INTER_AREA)
    dark = 255.0 - small.mean(axis=0)
    gx = np.abs(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3))
    edge = gx.mean(axis=0)
    k = max(3, (w // 200) | 1)
    return (cv2.GaussianBlur(dark.reshape(1, -1), (k, 1), 0).ravel(),
            cv2.GaussianBlur(edge.reshape(1, -1), (k, 1), 0).ravel())


def find_spine_x(img_bgr: np.ndarray, search=(0.35, 0.65)) -> Optional[int]:
    """Return the x of a plausible book gutter, or None."""
    h, w = img_bgr.shape[:2]
    if w / float(h) < 1.05:
        return None
    dark, edge = _column_profile(img_bgr)
    lo, hi = int(search[0] * w), int(search[1] * w)
    if hi <= lo:
        return None
    seg_dark = dark[lo:hi]
    seg_edge = edge[lo:hi]
    # gutter: darkest vertical band that is also relatively smooth
    cost = seg_dark - 0.15 * seg_edge
    idx = int(np.argmax(cost))
    x = lo + idx

    # contrast against the page interior must be meaningful
    inner = np.concatenate([dark[lo:x - max(3, w // 100)], dark[x + max(3, w // 100):hi]])
    if inner.size < 10:
        return None
    if dark[x] < inner.mean() + 0.15 * inner.std() + 2.0:
        return None
    return x


def content_crosses_spine(img_bgr: np.ndarray, x: int, band: int = 12,
                          white_frac: float = 0.18) -> bool:
    """True when a picture appears to run across the gutter.

    Two facing pages of text nearly always keep a light margin next to the
    gutter; a photograph or illustration that spans the spine does not. So the
    seam is treated as "crossed" when there is almost no paper-white next to it.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    x0 = max(0, x - band)
    x1 = min(w, x + band + 1)
    strip = gray[:, x0:x1]
    if strip.size == 0:
        return False
    bg = float(np.percentile(gray, 90))
    near_white = float((strip > 0.82 * bg).mean())
    return near_white < white_frac


def split_spread(img_bgr: np.ndarray, x: Optional[int] = None,
                 min_gutter: int = 8) -> list[np.ndarray]:
    """Split a spread at the gutter; returns one or two page images."""
    if x is None:
        x = find_spine_x(img_bgr)
    if x is None:
        return [img_bgr]
    if content_crosses_spine(img_bgr, x):
        return [img_bgr]
    h, w = img_bgr.shape[:2]
    left = img_bgr[:, : max(1, x - min_gutter // 2)]
    right = img_bgr[:, min(w - 1, x + min_gutter // 2):]
    pages = [p for p in (left, right) if p.shape[1] > w * 0.15]
    return pages or [img_bgr]
