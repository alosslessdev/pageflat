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
    # OSD is a *fallback*, not a preference. It was taking precedence and calling
    # upright pages 180 on graphic paper: on IMG_20260506_224123, rendered and read
    # at 900 px, the page is plainly upright with its headline across the top, while
    # OSD said 180. The four-rotation OCR comparison gets that page right (2953
    # units of text upright against 190 rotated), so the scores decide whenever they
    # have anything to say.
    variants = {0: img_bgr,
                90: cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE),
                180: cv2.rotate(img_bgr, cv2.ROTATE_180),
                270: cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)}
    scores = {a: _ocr_score(v) for a, v in variants.items()}
    best = max(scores, key=scores.get)
    # A rotation has to be earned: the page shown at 900 px as a Conair flyer is
    # plainly upright, yet its best rotation scored 181 against 0 upright -- a vote
    # cast by a handful of fragments. Requiring an absolute floor and a margin over
    # the upright score keeps that page as it is; pages with real text clear both
    # easily (2953 upright, 1845 upright, 1464 against 173).
    if best != 0 and (scores[best] < 500 or scores[best] < 2.0 * max(1, scores[0])):
        return 0
    if scores[best] <= 0:
        osd = _osd_rotation(img_bgr)
        return osd if osd is not None else 0
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


_PREDICTOR = None


def detect_rotation_fast(img_bgr: np.ndarray, device: str = "cuda") -> int:
    """Page orientation from docTR's MobileNetV3 classifier (GPU, ~10 ms).

    Class index = number of 90 degree clockwise turns needed to make the page
    upright. Falls back to the OCR scoring path when the model is unavailable.
    """
    global _PREDICTOR
    try:
        if _PREDICTOR is None:
            from doctr.models import page_orientation_predictor

            _PREDICTOR = page_orientation_predictor(pretrained=True)
            if device != "cpu":
                _PREDICTOR.model = _PREDICTOR.model.to(device)
        out = _PREDICTOR([cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)])
        cls = int(np.asarray(out[0]).ravel()[0])
        conf = float(np.asarray(out[2]).ravel()[0]) if len(out) > 2 else 1.0
        if conf >= 0.6:
            return (cls * 90) % 360
    except Exception:
        pass
    return detect_rotation(img_bgr)


_ORIENT_QUESTION = (
    "Which one word describes this page? Answer with a single letter, nothing else.\n"
    "A) the page is upright\n"
    "B) the page needs a 90 degree clockwise turn\n"
    "C) the page needs a 180 degree turn\n"
    "D) the page needs a 90 degree counter-clockwise turn"
)
_ORIENT_ANSWER = {"A": 0, "B": 90, "C": 180, "D": 270}


def ask_vision(url: str, img_bgr: np.ndarray, timeout: float = 300.0) -> int | None:
    """Orientation from a vision model, for pages where OCR has nothing to read.

    Measured: raw OCR voting calls upright pages 180 on flyers, and 15 % of the
    benchmark has no readable prose at all. A four-way classification works where a
    coordinate request does not — but the model has to be a strong one: a local 4B
    answered 2 of 3 correctly with one confident wrong direction, a DeepSeek-class
    model 3 of 3.
    """
    import base64

    import requests

    h, w = img_bgr.shape[:2]
    s = 896 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * s), int(h * s))) if s < 1 else img_bgr
    ok, buf = cv2.imencode(".jpg", small)
    if not ok:
        return None
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _ORIENT_QUESTION},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}},
        ]}],
        "max_tokens": 900, "temperature": 0.0,
    }
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip().upper()
    except Exception:  # noqa: BLE001
        return None
    for ch in text:
        if ch in _ORIENT_ANSWER:
            return _ORIENT_ANSWER[ch]
    return None


def squareness(img_bgr: np.ndarray, min_chars: int = 6) -> float | None:
    """Angle of the text lines in degrees: is the page square in the frame.

    Different from the bow, which asks whether a line is straight. A page can have
    perfectly straight lines and still sit a degree or two askew, which is what a
    reader notices first and what a line-straightness measure cannot see.
    """
    from .charlines import detect_lines

    h, w = img_bgr.shape[:2]
    slopes = []
    for line in detect_lines(img_bgr):
        if len(line.boxes) < min_chars:
            continue
        x0 = min(b[0] for b in line.boxes)
        x1 = max(b[0] + b[2] for b in line.boxes)
        if x1 - x0 < 0.1 * w:
            continue
        mid = (x0 + x1) / 2.0
        y0 = np.mean([b[1] + b[3] for b in line.boxes if b[0] < mid])
        y1 = np.mean([b[1] + b[3] for b in line.boxes if b[0] >= mid])
        slopes.append(float(np.degrees(np.arctan2(y1 - y0, (x1 - x0) / 2.0))))
    if len(slopes) < 3:
        return None
    return float(np.median(slopes))
