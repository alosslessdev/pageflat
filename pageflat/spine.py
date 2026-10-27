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


def _gutter_candidates(img_bgr: np.ndarray, lo_frac: float = 0.15,
                       hi_frac: float = 0.85) -> list[tuple[int, float]]:
    """Shadow bands that look like a book gutter, strongest first.

    The column-darkness profile is compared against a wide moving baseline, so
    an unevenly lit book (one side in shadow) does not hide the gutter.
    """
    h, w = img_bgr.shape[:2]
    if w / float(h) < 0.85:
        return []
    dark, edge = _column_profile(img_bgr)
    lo, hi = int(lo_frac * w), int(hi_frac * w)
    if hi - lo < 20:
        return []
    k = max(9, int(0.15 * w) | 1)
    base = cv2.blur(dark.reshape(1, -1), (k, 1)).ravel()
    resid = dark - base
    out: list[tuple[int, float]] = []
    work = resid.copy()
    for _ in range(4):
        idx = int(np.argmax(work[lo:hi]))
        x = lo + idx
        peak = float(work[x])
        if peak < 3.0:
            break
        half = peak * 0.5
        left = x
        while left > lo and resid[left] > half:
            left -= 1
        right = x
        while right < hi - 1 and resid[right] > half:
            right += 1
        band_width = right - left
        if band_width >= max(3, 0.003 * w):
            out.append((x, peak))
        work[max(0, x - int(0.06 * w)):min(w, x + int(0.06 * w))] = -1e9
    return out


def find_spine_x(img_bgr: np.ndarray) -> Optional[int]:
    """Strongest gutter that is not crossed by a spanning image."""
    for x, _ in _gutter_candidates(img_bgr):
        if not content_crosses_spine(img_bgr, x):
            return x
    return None


def content_crosses_spine(img_bgr: np.ndarray, x: int, band: int | None = None,
                          white_frac: float = 0.18) -> bool:
    """True when a picture appears to run across the gutter.

    Two facing pages of text nearly always keep a light margin next to the
    gutter; a photograph or illustration that spans the spine does not. So the
    seam is treated as "crossed" when there is almost no paper-white next to it.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    if band is None:
        band = max(6, int(round(0.012 * w)))
    x0 = max(0, x - band)
    x1 = min(w, x + band + 1)
    strip = gray[:, x0:x1]
    if strip.size == 0:
        return False
    bg = float(np.percentile(gray, 90))
    near_white = float((strip > 0.82 * bg).mean())
    return near_white < white_frac


def split_spread(img_bgr: np.ndarray, x: Optional[int] = None,
                 min_width_frac: float = 0.15) -> list[np.ndarray]:
    """Cut a spread at every detected gutter; returns one or more pages."""
    h, w = img_bgr.shape[:2]
    cuts: list[int] = []
    if x is not None:
        cuts = [] if content_crosses_spine(img_bgr, x) else [x]
    else:
        for gx, _ in _gutter_candidates(img_bgr):
            if content_crosses_spine(img_bgr, gx):
                continue
            if all(abs(gx - c) > 0.12 * w for c in cuts):
                cuts.append(gx)
    if not cuts:
        return [img_bgr]
    cuts.sort()
    bounds = [0] + cuts + [w]
    pages = []
    for a, b in zip(bounds, bounds[1:]):
        if (b - a) < min_width_frac * w:
            continue
        pages.append(img_bgr[:, a:b])
    return pages if len(pages) > 1 else [img_bgr]
