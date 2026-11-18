"""Axis-Aligned Distortion: the published metric, implemented from the paper.

Wang et al., *Axis-Aligned Document Dewarping* (AAAI 2026, arXiv 2507.15000) argue
that a well-dewarped document is defined by its **axis-aligned feature lines** — text
lines horizontal, rules and column edges vertical — and introduce Axis-Aligned
Distortion (AAD) as a metric with geometric meaning that tracks human perception
better than pixel registration against a reference.

This is an implementation of that idea with the detectors this repository already
has, not of their network: feature lines come from `charlines` (horizontal, with a
baseline fit) and from `vlines` (vertical rules). Each line contributes its angular
deviation from the axis it belongs to, weighted by its length, so a page with many
short labels cannot outvote its own body text.

The number is in degrees: 0 means every feature line is axis-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AAD:
    value: float                  # length-weighted mean |deviation| in degrees
    p90: float                    # the tail, so a few badly-angled lines show
    horizontal: float | None      # text lines only
    vertical: float | None        # rules only
    lines: int

    def __str__(self) -> str:
        return (f"AAD {self.value:.2f}° (p90 {self.p90:.2f}°, "
                f"H {self.horizontal if self.horizontal is None else round(self.horizontal, 2)}°, "
                f"V {self.vertical if self.vertical is None else round(self.vertical, 2)}°, "
                f"{self.lines} lines)")


def _horizontal_angles(img_bgr: np.ndarray, min_chars: int = 5,
                       min_len_frac: float = 0.06) -> list[tuple[float, float]]:
    """(angle_deg, weight) for every horizontal feature line."""
    from .charlines import detect_lines

    h, w = img_bgr.shape[:2]
    out = []
    for line in detect_lines(img_bgr):
        if len(line.boxes) < min_chars:
            continue
        x0 = min(b[0] for b in line.boxes)
        x1 = max(b[0] + b[2] for b in line.boxes)
        span = x1 - x0
        if span < min_len_frac * w:
            continue
        mid = (x0 + x1) / 2.0
        left = [b[1] + b[3] for b in line.boxes if b[0] < mid]
        right = [b[1] + b[3] for b in line.boxes if b[0] >= mid]
        if not left or not right:
            continue
        ang = float(np.degrees(np.arctan2(np.mean(right) - np.mean(left), span / 2.0)))
        if abs(ang) > 30:            # not a line we can attribute to an axis
            continue
        out.append((ang, float(span)))
    return out


def _vertical_angles(img_bgr: np.ndarray) -> list[tuple[float, float]]:
    """(angle_deg from vertical, weight) for every tall rule."""
    from .vlines import vertical_rules

    out = []
    for r in vertical_rules(img_bgr):
        length = float(r.y1 - r.y0)
        if length <= 0:
            continue
        # tilt_deg is measured from vertical already
        if abs(r.tilt_deg) > 30:
            continue
        out.append((r.tilt_deg, length))
    return out


def measure(img_bgr: np.ndarray) -> AAD:
    hor = _horizontal_angles(img_bgr)
    ver = _vertical_angles(img_bgr)
    pairs = hor + ver
    if not pairs:
        return AAD(float("nan"), float("nan"), None, None, 0)
    angles = np.array([abs(a) for a, _ in pairs], float)
    weights = np.array([w for _, w in pairs], float)
    total = weights.sum()
    value = float((angles * weights).sum() / total) if total > 0 else float("nan")
    p90 = float(np.percentile(angles, 90))
    h_only = [abs(a) for a, _ in hor]
    v_only = [abs(a) for a, _ in ver]
    return AAD(value, p90,
               float(np.mean(h_only)) if h_only else None,
               float(np.mean(v_only)) if v_only else None,
               len(pairs))


def rotate_to_align(img_bgr: np.ndarray, max_iters: int = 3,
                    tol: float = 0.15) -> tuple[np.ndarray, dict]:
    """Iteratively rotate until AAD stops falling.

    ArbDR (IEEE TMM 2026, arXiv 2511.23150) makes the point that a cascade of small,
    separately-measured corrections beats one large one; the same applies to the
    residual angle here, because each rotation changes the lines the next measurement
    sees. The loop stops when a pass fails to improve, so it cannot oscillate.
    """
    from .pipeline import _rotate_expand

    cur = img_bgr
    best = measure(cur)
    history = [round(best.value, 3) if best.value == best.value else None]
    for _ in range(max_iters):
        if best.value != best.value or best.value <= tol:
            break
        # rotate by the weighted mean signed angle of the horizontal lines
        hor = _horizontal_angles(cur)
        if not hor:
            break
        signed = np.array([a for a, _ in hor], float)
        weights = np.array([w for _, w in hor], float)
        angle = float((signed * weights).sum() / weights.sum())
        if abs(angle) < tol:
            break
        cand = _rotate_expand(cur, angle)
        got = measure(cand)
        history.append(round(got.value, 3) if got.value == got.value else None)
        if got.value != got.value or got.value >= best.value - 1e-3:
            break                      # no improvement: keep what we had
        cur, best = cand, got
    return cur, {"aad": round(best.value, 3) if best.value == best.value else None,
                 "iterations": len(history) - 1, "history": history}
