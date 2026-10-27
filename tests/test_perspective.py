"""Tests for the perspective stage: sheet splitting and paper-edge extension."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageflat import perspective  # noqa: E402


def _sheet(w, h, x0, y0, x1, y1, colour=240):
    img = np.full((h, w, 3), 60, np.uint8)
    img[y0:y1, x0:x1] = colour
    return img


def _mask(w, h, x0, y0, x1, y1):
    mask = np.zeros((h, w), np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def test_single_sheet_is_not_split():
    img = _sheet(400, 600, 80, 40, 320, 560)
    mask = _mask(400, 600, 80, 40, 320, 560)
    parts = perspective.sheet_masks(img, mask)
    assert len(parts) == 1
    assert parts[0].sum() == mask.sum()


def test_spread_splits_at_the_gutter():
    """Two pages meeting at a dark spine are one mask but two physical sheets."""
    w, h = 900, 600
    img = np.full((h, w, 3), 60, np.uint8)
    img[40:580, 120:486] = 240
    img[40:580, 494:860] = 240
    img[40:580, 486:494] = 30          # the gutter shadow
    rng = np.random.default_rng(0)
    for y in range(60, 560, 18):       # text-like strokes: real pages are not blank
        img[y:y + 5, 130:476] = rng.integers(20, 90, (5, 346, 1))
        img[y:y + 5, 504:850] = rng.integers(20, 90, (5, 346, 1))
    mask = np.zeros((h, w), np.float32)
    mask[40:580, 120:860] = 1.0        # one connected mask, as the segmenter returns
    parts = perspective.sheet_masks(img, mask)
    assert len(parts) == 2, f"expected two sheets, got {len(parts)}"
    left, right = sorted(parts, key=lambda m: np.nonzero(m)[1].mean())
    assert np.nonzero(left)[1].max() < np.nonzero(right)[1].min()


def test_landscape_page_without_shadow_is_not_split():
    """A wide page must not be mistaken for a spread just for being wide."""
    img = _sheet(600, 300, 40, 30, 560, 270)
    mask = _mask(600, 300, 40, 30, 560, 270)
    parts = perspective.sheet_masks(img, mask)
    assert len(parts) == 1


def test_extend_to_paper_grows_past_a_tight_mask():
    img = _sheet(400, 460, 60, 40, 340, 400)
    quad = np.array([[100, 80], [300, 80], [300, 380], [100, 380]], np.float32)
    grown = perspective.extend_to_paper(img, quad, max_frac=0.15)
    assert grown[0][0] < quad[0][0], "left edge should move towards the paper edge"
    assert grown[1][0] > quad[1][0], "right edge should move towards the paper edge"
    assert grown[0][0] >= 60 - 3 and grown[1][0] <= 340 + 3


def test_extend_to_paper_keeps_quad_when_there_is_no_paper():
    img = np.full((500, 500, 3), 40, np.uint8)
    quad = np.array([[100, 80], [300, 80], [300, 420], [100, 420]], np.float32)
    grown = perspective.extend_to_paper(img, quad, max_frac=0.15)
    assert np.allclose(grown, quad, atol=2.0)
