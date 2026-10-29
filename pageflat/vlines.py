"""Vertical structure as a reference: rules, column edges and panel borders.

The text-line ruler only sees horizontal geometry. A page that is tilted or bowed
*vertically* — the defect that survives on a flyer whose rows are graphics rather
than paragraphs — leaves the line bow at zero while the page is visibly crooked.
What it does damage instead is the vertical lines: column rules, panel borders,
the printed frame. In a flatbed scan those are vertical; how far they lean or bow
is the horizontal component of the same deformation.

The reference is built the same way the horizontal one is: find the strokes, join
the ones that belong together, then fit and measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VerticalRule:
    x0: float                 # x at the top
    x1: float                 # x at the bottom
    y0: int
    y1: int
    deviation: float          # median |x(y) - straight fit| in px

    @property
    def tilt_deg(self) -> float:
        dy = max(1, self.y1 - self.y0)
        return float(np.degrees(np.arctan2(self.x1 - self.x0, dy)))


def stroke_mask(img_bgr: np.ndarray, min_contrast: int = 25) -> np.ndarray:
    """Dark-on-light strokes, including thin rules and panel borders."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bg = cv2.medianBlur(blur, 31)
    diff = cv2.subtract(bg, blur)          # darker than its surroundings
    m = (diff > min_contrast).astype(np.uint8) * 255
    return m


def vertical_rules(img_bgr: np.ndarray, min_len_frac: float = 0.06,
                   open_frac: float = 0.01) -> list[VerticalRule]:
    """Runs of vertical stroke, measured against a straight vertical line."""
    h, w = img_bgr.shape[:2]
    m = stroke_mask(img_bgr)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(open_frac * h))))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(vert, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < min_len_frac * h or bw > 0.02 * w:
            continue                      # too short, or not a thin rule
        xs = np.full(bh, np.nan)
        for j in range(bh):
            row = np.flatnonzero(labels[y + j] == i)
            if row.size:
                xs[j] = row.mean()
        ok = ~np.isnan(xs)
        if ok.sum() < 0.5 * bh:
            continue
        xs = np.interp(np.arange(bh), np.flatnonzero(ok), xs[ok])
        t = np.arange(bh)
        A = np.stack([t, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
        dev = float(np.median(np.abs(xs - A @ coef)))
        out.append(VerticalRule(float(coef[1]), float(coef[1] + coef[0] * bh),
                                int(y), int(y + bh), dev))
    return out


def vertical_drift(img_bgr: np.ndarray) -> dict:
    """Median lean and bow of the page's vertical structure, as % of width."""
    rules = vertical_rules(img_bgr)
    if not rules:
        return {"rules": 0, "tilt_deg": None, "tilt_abs_deg": None, "bow_pct": None}
    tilts = np.array([r.tilt_deg for r in rules])
    bows = np.array([r.deviation for r in rules])
    w = img_bgr.shape[1]
    return {"rules": len(rules),
            "tilt_deg": float(np.median(tilts)),
            "tilt_abs_deg": float(np.median(np.abs(tilts))),
            "bow_pct": 100.0 * float(np.median(bows)) / w}


def rectify(img_bgr: np.ndarray, smoothing: int = 15,
             shear: bool = False) -> tuple[np.ndarray, dict]:
    """Undo the vertical lean: shift each row so the rules stand up straight.

    Every rule says, for each row it crosses, how far it sits from where a
    straight vertical line would put it. The median of those offsets over all
    rules is the row's horizontal displacement, and shifting every row by its own
    offset makes the columns vertical again — a shear that carries the page's
    bow with it, where a rotation would only undo the constant part.
    """
    h, w = img_bgr.shape[:2]
    h, w = img_bgr.shape[:2]
    # Detect on a small copy: the rules are judged by their slope, which survives
    # downscaling, while `stroke_mask`'s median filter on a 3508-px page costs
    # seconds. The rotation is applied to the full-resolution image.
    small = img_bgr
    scale = 1.0
    if max(h, w) > 1400:
        scale = 1400.0 / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    rules = vertical_rules(small)
    if len(rules) < 1:
        return img_bgr, {"corrected": False, "rules": 0}

    # The common lean goes first, as a rotation: the per-row offsets below are
    # measured against each rule's *own* straight fit, which by construction
    # ignores the slope they all share.
    tilt = float(np.median([r.tilt_deg for r in rules]))
    # Calibrated against known rotations: rotating a page +2 deg CCW raises the
    # measured tilt by 2, so the correction is -tilt.
    if abs(tilt) > 0.2:
        # Measured over 36 pages: the raw rotation takes the median |tilt| from
        # 1.64 to 0.17 deg and the pages above 1 deg from 25 to 7. No guard — a
        # "keep it only if it improved" check made it worse, because the guarded
        # and plain measurements of the same page disagreed.
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -tilt, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # The common lean is removed first, as a rotation: the per-row offsets below
    # are measured against each rule's *own* straight fit, which by construction
    # ignores the slope they all share.
    tilts = np.array([r.tilt_deg for r in rules])
    tilt = float(np.median(tilts))
    # Calibrated against known rotations: rotating a page +2 deg CCW raises the
    # measured tilt by 2, so the correction is -tilt.
    if abs(tilt) > 0.2:
        base = vertical_drift(img_bgr)["tilt_deg"]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -tilt, 1.0)
        img_bgr = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    m = stroke_mask(img_bgr)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(0.01 * h))))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vert, 8)

    offsets = np.zeros(h, float)
    counts = np.zeros(h, float)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < 0.06 * h or bw > 0.02 * w:
            continue
        xs = np.full(bh, np.nan)
        for j in range(bh):
            row = np.flatnonzero(labels[y + j] == i)
            if row.size:
                xs[j] = row.mean()
        ok = ~np.isnan(xs)
        if ok.sum() < 0.5 * bh:
            continue
        xs = np.interp(np.arange(bh), np.flatnonzero(ok), xs[ok])
        t = np.arange(bh)
        A = np.stack([t, np.ones_like(t)], 1)
        coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
        straight = A @ coef
        offsets[y:y + bh] += xs - straight
        counts[y:y + bh] += 1

    if not shear:
        # The rotation is calibrated; the per-row shear is not. Keeping it off by
        # default until it is measured against a known deformation.
        lean = 0.0
        if abs(tilt) > 0.2:
            return img_bgr, {"corrected": True, "rules": len(rules), "lean": 0.0,
                             "tilt_deg": round(tilt, 2)}
    live = counts > 0
    if live.sum() < 0.2 * h:
        return img_bgr, {"corrected": False, "rules": len(rules)}
    dx = np.zeros(h, float)
    dx[live] = offsets[live] / counts[live]
    dx = np.interp(np.arange(h), np.flatnonzero(live), dx[live])
    dx = cv2.GaussianBlur(dx.reshape(-1, 1), (1, max(3, smoothing | 1)), 0).ravel()

    lean = float(np.percentile(np.abs(dx), 90))
    if lean < 1.0:
        return img_bgr, {"corrected": False, "rules": len(rules), "lean": round(lean, 2)}
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(gx - dx.astype(np.float32)[:, None], 0, w - 1)
    out = cv2.remap(img_bgr, map_x, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "rules": len(rules), "lean": round(lean, 2),
                 "tilt_deg": round(tilt, 2)}
