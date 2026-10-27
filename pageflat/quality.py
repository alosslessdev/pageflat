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


def paper_texture(img_bgr: np.ndarray, max_side: int = 900) -> float:
    """Median |Laplacian| — paper is smooth, fabric and other surfaces are not."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.median(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))


def paper_fraction(img_bgr: np.ndarray, max_side: int = 900) -> float:
    """Fraction of neutral, bright pixels — paper, as opposed to scenes or fabric."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # brightness, not neutrality: a page of a picture book is still a page even
    # when most of it is orange, and requiring low saturation threw those away
    return float((hsv[:, :, 2] > 140).mean())


def surface_checks(img_bgr: np.ndarray, max_texture: float = 30.0,
                   min_paper: float = 0.45) -> tuple[bool, dict]:
    """Paper-surface tests. Run these BEFORE illumination flattening, which
    normalises away exactly the colour and texture cues they rely on."""
    tex = paper_texture(img_bgr)
    paper = paper_fraction(img_bgr)
    return (tex <= max_texture and paper >= min_paper), {
        "texture": round(tex, 2), "paper_fraction": round(paper, 3)}


def content_checks(img_bgr: np.ndarray, min_ink: float = 0.0025,
                   min_lines: int = 3) -> tuple[bool, dict]:
    """Ink/rule tests. Run these AFTER flattening, when the page is even."""
    frac, lines = ink_fraction(img_bgr)
    return (frac >= min_ink or lines >= min_lines), {
        "ink_fraction": round(frac, 5), "long_lines": lines}


def looks_like_page(img_bgr: np.ndarray, min_ink: float = 0.0025,
                    min_lines: int = 3, max_texture: float = 30.0,
                    min_paper: float = 0.45) -> tuple[bool, dict]:
    """Convenience wrapper for images that are already final (no separate surface pass)."""
    ok_s, m1 = surface_checks(img_bgr, max_texture, min_paper)
    ok_c, m2 = content_checks(img_bgr, min_ink, min_lines)
    return ok_s and ok_c, {**m1, **m2}
