"""Page boundary estimation and perspective rectification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np


@dataclass
class PageQuad:
    points: np.ndarray  # (4, 2) float32, ordered tl, tr, br, bl
    area_frac: float  # quad area / image area
    rectangularity: float  # mask pixels inside quad / quad area
    score: float

    @property
    def valid(self) -> bool:
        return self.area_frac > 0.03 and self.rectangularity > 0.6


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out = np.zeros((4, 2), np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def _clean_mask(mask: np.ndarray, thresh: float = 0.5, min_area_frac: float = 0.02) -> np.ndarray:
    """Binary mask, keeping the most page-like connected component.

    Components are scored by mean page probability (not just size), so a small,
    confidently-detected sheet wins over a large, weakly-detected background
    surface such as a bed sheet.
    """
    binm = (mask > thresh).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    if n <= 1:
        return binm
    h, w = binm.shape
    flat = labels.ravel()
    prob = np.bincount(flat, weights=mask.ravel().astype(np.float64), minlength=n)
    best, best_score = 0, -1.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = area / float(h * w)
        if frac < min_area_frac:
            continue
        mean_prob = prob[i] / max(1, area)
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
        centre = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))
        fill = area / float(bw * bh)
        score = mean_prob * (0.5 + 0.5 * fill) * (0.7 + 0.3 * centre) * min(1.0, frac / 0.15)
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        return binm
    return (labels == best).astype(np.uint8)


def mask_to_quad(mask: np.ndarray) -> Optional[PageQuad]:
    """Fit a quadrilateral to the page mask."""
    binm = _clean_mask(mask)
    h, w = binm.shape
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.02 * h * w:
        return None

    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        box = cv2.boxPoints(cv2.minAreaRect(cnt))
        quad = box.astype(np.float32)
    quad = order_quad(quad)

    quad_area = abs(cv2.contourArea(quad))
    mask_area = float((binm > 0).sum())
    rectangularity = min(1.0, mask_area / quad_area) if quad_area > 0 else 0.0
    area_frac = quad_area / float(h * w)
    score = area_frac * rectangularity
    return PageQuad(quad, area_frac, rectangularity, score)


def quad_target_size(quad: np.ndarray, max_side: int = 4200) -> tuple[int, int]:
    """Output (width, height) preserving the quad's mean edge lengths."""
    tl, tr, br, bl = quad
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    scale = min(1.0, max_side / max(width, height))
    return max(16, int(round(width * scale))), max(16, int(round(height * scale)))


def warp_quad(img: np.ndarray, quad: np.ndarray, out_size: Optional[tuple[int, int]] = None,
              max_side: int = 4200) -> np.ndarray:
    w, h = out_size if out_size else quad_target_size(quad, max_side)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def possible_spread(rect: np.ndarray) -> bool:
    """Heuristic: landscape-ish page that may hold two facing pages."""
    h, w = rect.shape[:2]
    return w / float(h) > 1.05


def _fit_line(pts: np.ndarray):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    d = np.array([vx, vy], np.float64)
    d /= np.linalg.norm(d)
    return np.array([x0, y0], np.float64), d


def _intersect(p1, d1, p2, d2):
    A = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], np.float64)
    if abs(np.linalg.det(A)) < 1e-8:
        return None
    b = p2 - p1
    t = np.linalg.solve(A, b)
    return p1 + t[0] * d1


def refine_quad(gray: np.ndarray, quad: np.ndarray, search_frac: float = 0.03,
                samples: int = 48) -> np.ndarray:
    """Snap a rough page quad to the strongest nearby edges."""
    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    D = max(6, int(search_frac * max(h, w)))
    lines = []
    for i in range(4):
        p = quad[i].astype(np.float64)
        q = quad[(i + 1) % 4].astype(np.float64)
        d = q - p
        L = np.linalg.norm(d)
        if L < 4:
            return quad
        n = np.array([-d[1], d[0]]) / L
        pts = []
        for t in np.linspace(0.08, 0.92, samples):
            base = p + t * d
            best, bestv = None, 0.0
            for s in range(-D, D + 1):
                x, y = base + n * s
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    v = float(mag[iy, ix])
                    if v > bestv:
                        bestv, best = v, (x, y)
            if best is not None:
                pts.append(best)
        pts = np.array(pts)
        if len(pts) < 12:
            return quad
        med = np.median(pts, axis=0)
        keep = np.linalg.norm(pts - med, axis=1) < 3 * (np.median(np.linalg.norm(pts - med, axis=1)) + 3)
        if keep.sum() < 10:
            return quad
        lines.append(_fit_line(pts[keep]))

    corners = []
    for i in range(4):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        c = _intersect(p1, d1, p2, d2)
        if c is None:
            return quad
        corners.append(c)
    refined = np.array(corners, np.float32)
    if not np.isfinite(refined).all():
        return quad
    if not cv2.isContourConvex(refined.reshape(-1, 1, 2).astype(np.int32)):
        return quad
    a0 = abs(cv2.contourArea(quad.astype(np.float32)))
    a1 = abs(cv2.contourArea(refined))
    if a0 <= 0 or not (0.6 < a1 / a0 < 1.15):
        return quad
    # corners must stay close to the rough quad
    if np.max(np.linalg.norm(refined - quad, axis=1)) > 0.18 * max(h, w):
        return quad
    return order_quad(refined)


def edge_support(gray: np.ndarray, quad: np.ndarray, band_frac: float = 0.02,
                 samples: int = 60, thresh: float = 35.0) -> float:
    """Fraction of quad-side samples that sit on a real intensity edge."""
    h, w = gray.shape[:2]
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    D = max(5, int(band_frac * max(h, w)))
    hit = total = 0
    for i in range(4):
        p = np.asarray(quad[i], np.float64)
        q = np.asarray(quad[(i + 1) % 4], np.float64)
        d = q - p
        L = np.linalg.norm(d)
        if L < 4:
            continue
        n = np.array([-d[1], d[0]]) / L
        for t in np.linspace(0.1, 0.9, samples):
            base = p + t * d
            best = 0.0
            for s in range(-D, D + 1):
                x, y = base + n * s
                ix, iy = int(round(x)), int(round(y))
                if 0 <= ix < w and 0 <= iy < h:
                    v = float(mag[iy, ix])
                    if v > best:
                        best = v
            total += 1
            if best >= thresh:
                hit += 1
    return hit / total if total else 0.0
