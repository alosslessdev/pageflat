"""Content checks that separate real page scans from accidental surfaces."""
from __future__ import annotations

import cv2
import numpy as np


def ink_fraction(img_bgr: np.ndarray, max_side: int = 900) -> tuple[float, int]:
    """Return (ink coverage, count of long horizontal/vertical rules).

    Ink is dark relative to a heavily blurred local background, which is what
    printed or written content looks like once illumination is flattened.
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, gray.shape[1] / 60))
    ink = (gray.astype(np.int16) < bg.astype(np.int16) - 28).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    frac = float(ink.mean())

    lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180, threshold=60,
                            minLineLength=int(gray.shape[1] * 0.25), maxLineGap=6)
    if lines is None:
        return frac, 0
    good = 0
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if ang < 6 or ang > 174 or 84 < ang < 96:
            good += 1
    return frac, good


def looks_like_page(img_bgr: np.ndarray, min_ink: float = 0.0025,
                    min_lines: int = 3) -> tuple[bool, dict]:
    frac, lines = ink_fraction(img_bgr)
    ok = frac >= min_ink or lines >= min_lines
    return ok, {"ink_fraction": round(frac, 5), "long_lines": lines}
