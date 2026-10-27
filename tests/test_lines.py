"""Word-line detection and grid unwarping, on a real scan and on synthetic bends.

The page used here is a finished 1595x2679 scan: text lines, a dashed border
and a real (small) curl. Bending it by a known sinusoid is the only way to
know what a correct answer looks like, so the bow is always measured in
full-resolution pixels through the module's own detector.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from pageflat import lines

SCAN_DIR = Path(os.environ.get(
    "PAGEFLAT_SCANS",
    "/run/media/asdf/30620d0d-1e41-4d55-af08-b4de2fb9c517/pageflat_scans_v4"))
PAGE = "IMG_20251029_213411_p1.png"        # real curl, readable text


def _load(name: str = PAGE) -> np.ndarray:
    path = SCAN_DIR / name
    if not path.exists():
        pytest.skip(f"test scan not available: {path}")
    img = cv2.imread(str(path))
    assert img is not None
    return img


def _bend(img: np.ndarray, amplitude: float, waves: float = 1.0) -> np.ndarray:
    """Resample `img` so it appears bent by a sinusoid of `amplitude` pixels."""
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    map_x = np.broadcast_to(xs, (h, w)).copy()
    map_y = np.clip(ys + amplitude * np.sin(np.pi * waves * xs / w), 0, h - 1)
    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def _bow(img: np.ndarray) -> float:
    """Median line bow in the pixels of `img`."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found, back = lines.detect_word_lines(gray)
    assert found, "no word lines detected"
    return float(np.median([line.bow for line in found]) * back)


def _synthetic_page(rows: int = 24, w: int = 1400, h: int = 1500,
                    jitter: float = 0.0, seed: int = 7) -> np.ndarray:
    """A page of clean printed text: no paper, no camera, just lines."""
    img = np.full((h, w), 255, np.uint8)
    rng = np.random.default_rng(seed)
    for i in range(rows):
        off = float(rng.uniform(-jitter, jitter)) if jitter else 0.0
        y = int(70 + i * (h - 140) / rows + off)
        cv2.putText(img, f"The quick brown fox jumps over the lazy dog {i}.",
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def test_detects_lines_not_fragments():
    """Fragments (median box a fraction of the page) are what broke the old code."""
    img = _load()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, back = lines.detect_word_lines(gray)
    assert len(found) >= 8
    work_w = gray.shape[1] / back
    work_h = gray.shape[0] / back
    spans = np.array([line.x1 - line.x0 for line in found], dtype=float)
    assert float(np.median(spans)) > 0.25 * work_w
    for line in found:
        assert 0 <= line.x0 < line.x1 <= work_w + 1
        assert 0 <= line.top < line.bottom <= work_h + 1
        assert line.curve.size == lines.CURVE_POINTS


def test_slanted_line_scores_no_bow():
    """A line that is only slanted is not bent: the slant must not count."""
    img = _synthetic_page()
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :]
    ys = np.arange(h, dtype=np.float32)[:, None]
    slope = 0.06
    map_y = np.clip(ys + slope * xs, 0, h - 1)
    map_x = np.broadcast_to(xs, (h, w)).copy()
    slanted = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                        cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    assert _bow(img) < 0.75
    assert _bow(slanted) < 1.5          # a 6% tilt over 1400 px is 84 px of rise


@pytest.mark.parametrize("amplitude", [8.0, 16.0, 32.0])
def test_unwarp_straightens_a_bent_scan(amplitude):
    img = _load()
    bent = _bend(img, amplitude)
    before = _bow(bent)
    out, info = lines.unwarp(bent)
    after = _bow(out)
    assert info["corrected"] is True, info
    assert before > 2.0
    assert after <= 0.4 * before, f"bow {before:.2f} -> {after:.2f} px"
    assert after < 3.0, f"residual bow {after:.2f} px at full resolution"


def test_unwarp_leaves_a_straight_page_alone():
    img = _synthetic_page()
    out, info = lines.unwarp(img)
    assert info["corrected"] is False
    assert np.array_equal(out, img)


def test_unwarp_refuses_an_inconsistent_page():
    """A crumple bends every line differently; straightening it would shear it."""
    img = _synthetic_page(jitter=14.0)
    out, info = lines.unwarp(img)
    assert info["corrected"] is False
    assert np.array_equal(out, img)


def test_unwarp_is_idempotent():
    img = _load()
    once, info = lines.unwarp(img)
    assert info["corrected"] is True
    twice, again = lines.unwarp(once)
    assert again["corrected"] is False
    assert np.array_equal(twice, once)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_boxes_are_in_the_given_frame():
    """Boxes come back in the coordinates of the image handed in, and only once."""
    img = _load()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (800, 1343), interpolation=cv2.INTER_AREA)
    low = lines.tesseract_lines(small)
    full = lines.tesseract_lines(gray)
    if not low or not full:
        pytest.skip("tesseract found no text on this machine")
    for x, y, w, h in low:
        assert 0 <= x and x + w <= small.shape[1] + 1
        assert 0 <= y and y + h <= small.shape[0] + 1
    for x, y, w, h in full:
        assert 0 <= x and x + w <= gray.shape[1] + 1
        assert 0 <= y and y + h <= gray.shape[0] + 1
    ratio = gray.shape[1] / small.shape[1]
    # the widest real lines are the ones a frame mix-up would move; the
    # median is polluted by dashes and captions tesseract also calls lines
    wide_low = float(np.percentile([b[2] for b in low], 90))
    wide_full = float(np.percentile([b[2] for b in full], 90))
    assert 0.75 * ratio < wide_full / wide_low < 1.25 * ratio
