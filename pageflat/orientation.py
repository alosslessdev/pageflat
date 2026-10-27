"""Page orientation detection (0/90/180/270) and deskew."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

_LANG: str | None = None


def ocr_lang() -> str:
    """First usable Tesseract language (installs often ship without 'eng')."""
    global _LANG
    if _LANG is None:
        try:
            out = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                                 text=True, timeout=20).stdout.splitlines()[1:]
            langs = [l.strip() for l in out if l.strip() and l.strip() != "osd"]
        except Exception:
            langs = []
        for pref in ("spa+eng", "eng", "spa"):
            parts = pref.split("+")
            if all(p in langs for p in parts):
                _LANG = pref
                break
        else:
            _LANG = langs[0] if langs else ""
    return _LANG


def _osd_rotation(img_bgr: np.ndarray) -> int | None:
    """Rotation from Tesseract OSD, but only when it is confident."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    cv2.imwrite(tmp, small)
    try:
        out = subprocess.run(["tesseract", tmp, "-", "--psm", "0"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return None
    finally:
        Path(tmp).unlink(missing_ok=True)
    rot = re.search(r"Rotate:\s*(\d+)", out)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    if not rot:
        return None
    if conf and float(conf.group(1)) < 2.0:
        return None
    return int(rot.group(1)) % 360


def _ocr_score(img_bgr: np.ndarray) -> float:
    """Sum of confident word scores (0 when there is no machine-readable text)."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, 700.0 / max(h, w))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    cv2.imwrite(tmp, small)
    try:
        cmd = ["tesseract", tmp, "stdout", "--psm", "6"]
        if ocr_lang():
            cmd += ["-l", ocr_lang()]
        out = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=90).stdout
    except Exception:
        return 0.0
    finally:
        Path(tmp).unlink(missing_ok=True)
    total = 0.0
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 12:
            try:
                conf = float(parts[10])
            except ValueError:
                continue
            word = parts[11].strip()
            if len(word) > 2 and conf > 55:
                total += conf * min(len(word), 8)
    return total


def detect_rotation(img_bgr: np.ndarray) -> int:
    """Clockwise rotation in degrees (0/90/180/270) needed to make text upright.

    Tesseract OSD alone is unreliable (it reports low confidence or "too few
    characters" and callers silently keep the wrong rotation), so the four
    candidate rotations are scored by OCR confidence and the best wins.
    """
    osd = _osd_rotation(img_bgr)
    if osd is not None:
        return osd

    variants = {0: img_bgr,
                90: cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
                180: cv2.rotate(img_bgr, cv2.ROTATE_180),
                270: cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    scores = {a: _ocr_score(v) for a, v in variants.items()}
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return 0
    return best


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
