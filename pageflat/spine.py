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
    """Cut a spread at the gutters that keep every piece page-shaped."""
    h, w = img_bgr.shape[:2]
    if x is not None:
        candidates = [] if content_crosses_spine(img_bgr, x) else [(x, 0.0)]
    else:
        candidates = [(gx, strength) for gx, strength in _gutter_candidates(img_bgr)
                      if not content_crosses_spine(img_bgr, gx)]
    if not candidates:
        return [img_bgr]

    accepted: list[int] = []
    for gx, _ in sorted(candidates, key=lambda t: -t[1]):
        if any(abs(gx - c) < 0.12 * w for c in accepted):
            continue
        trial = sorted(accepted + [gx])
        bounds = [0] + trial + [w]
        widths = [b - a for a, b in zip(bounds, bounds[1:])]
        if min(widths) < min_width_frac * w:
            continue
        aspects = [wd / float(h) for wd in widths]
        if min(aspects) < 0.35 or max(aspects) > 2.5:
            continue
        if max(widths) / float(min(widths)) > 1.6:
            continue
        accepted.append(gx)

    if not accepted:
        return [img_bgr]
    bounds = [0] + sorted(accepted) + [w]
    return [img_bgr[:, a:b] for a, b in zip(bounds, bounds[1:])]


def trace_spine(img_bgr: np.ndarray, expected_x: Optional[int] = None,
                band_frac: float = 0.12, step: int = 4) -> Optional[np.ndarray]:
    """Follow the gutter down the page: one x per row, or None.

    The spine of a book is a straight line in the world. Once the spread is
    rectified, whatever wander is left in that dark band is residual rotation
    and curl, so tracing it gives a per-row correction the homography could
    not express.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    cx = w // 2 if expected_x is None else int(expected_x)
    band = max(6, int(band_frac * w))
    lo, hi = max(1, cx - band), min(w - 1, cx + band)
    if hi - lo < 4:
        return None
    rows = np.arange(0, h, step)
    xs = np.full(rows.size, np.nan)
    k = max(3, w // 200 | 1)
    for i, y in enumerate(rows):
        strip = gray[y:y + step, lo:hi].mean(axis=0)
        if strip.size < 4:
            continue
        smooth = cv2.GaussianBlur(strip.reshape(1, -1), (k, 1), 0).ravel()
        j = int(np.argmin(smooth))
        shoulder = float(np.median(smooth))
        if shoulder - smooth[j] < 4.0:      # no shadow at this row
            continue
        xs[i] = lo + j
    if np.isnan(xs).mean() > 0.5:
        return None
    ok = ~np.isnan(xs)
    xs = np.interp(rows, rows[ok], xs[ok])
    xs = cv2.GaussianBlur(xs.reshape(-1, 1), (1, max(3, rows.size // 20 | 1)), 0).ravel()
    return xs


def align_to_spine(img_bgr: np.ndarray, curve: Optional[np.ndarray] = None,
                   step: int = 4) -> tuple[np.ndarray, dict]:
    """Shear rows so the traced spine becomes a vertical line."""
    h, w = img_bgr.shape[:2]
    if curve is None:
        curve = trace_spine(img_bgr)
    if curve is None:
        return img_bgr, {"spine": False}
    rows = np.arange(0, h, step)[: curve.size]
    if curve.size != rows.size:
        return img_bgr, {"spine": False}
    full = np.interp(np.arange(h), rows, curve)
    shift = float(np.median(full)) - full
    if float(np.abs(shift).max()) < 2.0:
        return img_bgr, {"spine": True, "drift": round(float(np.abs(shift).max()), 1)}
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(xs + shift[:, None].astype(np.float32), 0, w - 1)
    out = cv2.remap(img_bgr, map_x, ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"spine": True, "drift": round(float(np.abs(shift).max()), 1)}
