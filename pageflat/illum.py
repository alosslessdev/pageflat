"""Illumination flattening and paper-background normalisation."""
from __future__ import annotations

import cv2
import numpy as np


def _poly_terms(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1)


def fit_background_model(img_bgr: np.ndarray, grid: int = 24, pct: float = 90.0) -> np.ndarray:
    """Quadratic paper-lighting coefficients (6, 3) fitted to tile brightness percentiles."""
    h, w = img_bgr.shape[:2]
    img = img_bgr
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    pts, vals = [], []
    for i in range(grid):
        for j in range(grid):
            tile = img[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if tile.size == 0:
                continue
            pts.append(((xs[j] + xs[j + 1]) / 2.0 / w, (ys[i] + ys[i + 1]) / 2.0 / h))
            vals.append(np.percentile(tile.reshape(-1, 3), pct, axis=0))
    if len(pts) < 6:
        return np.zeros((6, 3), np.float32) + np.array([255.0, 255.0, 255.0], np.float32)
    A = _poly_terms(np.array([p[0] for p in pts]), np.array([p[1] for p in pts])).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, np.array(vals, np.float32), rcond=None)
    return coef


def background_from_model(coef: np.ndarray, shape: tuple[int, int], size: int = 160) -> np.ndarray:
    """Evaluate the lighting model on a small grid and upsample to full size."""
    h, w = shape
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float32)
    gx /= max(1, size - 1)
    gy /= max(1, size - 1)
    B = _poly_terms(gx.ravel(), gy.ravel())
    bg = (B @ coef).reshape(size, size, 3)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 8.0)


def flatten_illumination(img_bgr: np.ndarray, white: float = 245.0, strength: float = 1.0) -> np.ndarray:
    """Divide by the paper-lighting surface so the page reads like a flat scan."""
    coef = fit_background_model(img_bgr)
    bg = background_from_model(coef, img_bgr.shape[:2])
    scale = (white / bg).astype(np.float32)
    if strength != 1.0:
        scale = 1.0 + (scale - 1.0) * strength
    out = img_bgr.astype(np.float32, copy=False)
    out *= scale
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    means = np.array(cv2.mean(img_bgr)[:3], np.float32)
    target = float(means.mean())
    gains = (target / np.maximum(means, 1e-3)).astype(np.float32)
    out = img_bgr.astype(np.float32, copy=False)
    out *= gains.reshape(1, 1, 3)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def auto_contrast(img_bgr: np.ndarray, low: float = 0.5, high: float = 99.6) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lo, hi = np.percentile(gray, [low, high])
    if hi - lo < 1:
        return img_bgr
    lut = np.clip((np.arange(256) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


def scan_finish(img_bgr: np.ndarray, paper_white: bool = True) -> np.ndarray:
    """Turn a dewarped page into a flatbed-looking scan (one tonal pass + contrast)."""
    if paper_white:
        coef = fit_background_model(img_bgr)
        bg = background_from_model(coef, img_bgr.shape[:2])
        means = np.array(cv2.mean(img_bgr)[:3], np.float32)
        wb = float(means.mean()) / np.maximum(means, 1e-3)
        scale = (245.0 / bg) * wb.reshape(1, 1, 3)
        out = img_bgr.astype(np.float32, copy=False)
        out *= scale
        np.clip(out, 0, 255, out=out)
        out = out.astype(np.uint8)
    else:
        out = gray_world_white_balance(img_bgr)
    return auto_contrast(out, 0.3, 99.7)
