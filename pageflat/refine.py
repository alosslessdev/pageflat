"""Residual refinement: text-line straightening and paper cropping."""
from __future__ import annotations

import cv2
import numpy as np


def _line_offset_samples(gray: np.ndarray, max_width: int = 1100):
    """Sample (x, y, dy) triples on detected text baselines."""
    h, w = gray.shape
    scale = min(1.0, max_width / float(w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape

    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    kx = max(9, sw // 25)
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1)))
    lines = cv2.morphologyEx(lines, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)

    samples = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < sw * 0.2 or bh > sh * 0.08 or area < 60:
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        cols = np.where(comp.any(axis=0))[0]
        if len(cols) < 10:
            continue
        pts_x, pts_y = [], []
        for c in np.linspace(cols.min(), cols.max(), 24).astype(int):
            rows = np.where(comp[:, c])[0]
            if rows.size:
                pts_x.append(x + c)
                pts_y.append(y + rows.mean())
        if len(pts_x) < 8:
            continue
        pts_x = np.array(pts_x, np.float32)
        pts_y = np.array(pts_y, np.float32)
        # remove the line's own slope: offset relative to its mean
        fit = np.polyfit(pts_x, pts_y, 1)
        resid = pts_y - np.polyval(fit, pts_x)
        samples.extend((px / sw, py / sh, res / sh)
                       for px, py, res in zip(pts_x, pts_y, resid))
    return [(x, y, dy) for x, y, dy in samples], (sw, sh)


def _poly_basis(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    """Centred, scaled monomial basis (u, v in [-1, 1]) to keep the fit well conditioned."""
    u = 2.0 * np.asarray(x, np.float64) - 1.0
    v = 2.0 * np.asarray(y, np.float64) - 1.0
    cols = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            cols.append((u ** i) * (v ** j))
    return np.stack(cols, axis=1)


def straighten(img_bgr: np.ndarray, degree: int = 2, max_shift: float = 0.03,
               iters: int = 3, min_amplitude: float = 0.008) -> np.ndarray:
    """Flatten residual curvature using detected text baselines."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    samples, _ = _line_offset_samples(gray)
    if len(samples) < 40:
        return img_bgr
    pts = np.array([[x, y] for x, y, _ in samples], np.float64)
    vals = np.array([dy for _, _, dy in samples], np.float64)
    spread = float(np.percentile(vals, 95) - np.percentile(vals, 5))
    if spread < min_amplitude:
        return img_bgr

    # robust, smooth global fit: residuals are low-order, so a polynomial cannot
    # blow up the way a local interpolant does
    A = _poly_basis(pts[:, 0], pts[:, 1], degree)
    w = np.ones(len(vals))
    coef = np.zeros(A.shape[1])
    ridge = 1e-3 * len(vals)
    for _ in range(iters):
        Aw = A * w[:, None]
        AtA = Aw.T @ Aw + ridge * np.eye(A.shape[1])
        coef = np.linalg.solve(AtA, Aw.T @ (vals * w))
        resid = vals - A @ coef
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        w = 1.0 / (1.0 + (resid / (3 * sigma)) ** 2)

    h, w_img = img_bgr.shape[:2]
    gy, gx = np.mgrid[0:h, 0:w_img].astype(np.float32)
    B = _poly_basis((gx.ravel() / w_img).astype(np.float64),
                    (gy.ravel() / h).astype(np.float64), degree)
    field = (B @ coef).reshape(h, w_img).astype(np.float32)
    np.clip(field, -max_shift, max_shift, out=field)

    # only trust the correction where text actually exists
    y_lo, y_hi = pts[:, 1].min(), pts[:, 1].max()
    ramp = np.clip((gy / h - (y_lo - 0.03)) / 0.03, 0, 1) * \
        np.clip(((y_hi + 0.03) - gy / h) / 0.03, 0, 1)
    field *= ramp.astype(np.float32)



    map_x = gx
    map_y = gy - field * h
    return cv2.remap(img_bgr, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def crop_to_paper(img_bgr: np.ndarray, margin: float = 0.01) -> np.ndarray:
    """Keep the largest bright, low-saturation region (the sheet itself)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((sat < 60) & (val > 110)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return img_bgr
    h, w = mask.shape
    centre = labels[h // 2, w // 2]
    if centre == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        centre = 1 + int(np.argmax(areas))
    x, y, bw, bh, area = stats[centre]
    if area < 0.2 * h * w:
        return img_bgr
    mx, my = int(margin * w), int(margin * h)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    if (x1 - x0) * (y1 - y0) < 0.25 * h * w:
        return img_bgr
    return img_bgr[y0:y1, x0:x1]
