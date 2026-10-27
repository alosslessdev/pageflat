"""Page orientation detection (0/90/180/270) and deskew."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


def detect_rotation(img_bgr: np.ndarray) -> int:
    """Clockwise rotation in degrees (0/90/180/270) needed to make text upright."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    cv2.imwrite(tmp, small)
    try:
        out = subprocess.run(["tesseract", tmp, "-", "--psm", "0"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return 0
    finally:
        Path(tmp).unlink(missing_ok=True)
    m = re.search(r"Rotate:\s*(\d+)", out)
    if not m:
        return 0
    return int(m.group(1)) % 360


def rotate_upright(img_bgr: np.ndarray) -> tuple[np.ndarray, int]:
    angle = detect_rotation(img_bgr)
    if angle == 90:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE), angle
    if angle == 180:
        return cv2.rotate(img_bgr, cv2.ROTATE_180), angle
    if angle == 270:
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE), angle
    return img_bgr, 0


def skew_angle(img_bgr: np.ndarray, max_angle: float = 12.0) -> float:
    """Estimate residual skew from the dominant near-horizontal line direction."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, 1200.0 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=80,
                            minLineLength=small.shape[1] // 6, maxLineGap=12)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) <= max_angle:
            angles.append(a)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img_bgr: np.ndarray, max_angle: float = 12.0) -> tuple[np.ndarray, float]:
    a = skew_angle(img_bgr, max_angle)
    if abs(a) < 0.25:
        return img_bgr, a
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    out = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, a
