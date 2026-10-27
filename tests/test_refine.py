import cv2
import numpy as np

from pageflat.refine import straighten


def _text_page(w=900, h=1200):
    img = np.full((h, w, 3), 250, np.uint8)
    for i, y in enumerate(range(80, h - 80, 46)):
        cv2.putText(img, "the quick brown fox jumps over the lazy dog " * 2,
                    (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return img


def _warp(img, amp=28.0):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = amp * np.sin(2 * np.pi * xx / w) * (0.3 + 0.7 * yy / h)
    return cv2.remap(img, xx, yy + dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _line_wobble(img):
    """Mean vertical spread of detected text baselines (lower is straighter)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    lines = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (41, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
    spreads = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < gray.shape[1] * 0.3 or bh > 40:
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        cols = np.where(comp.any(axis=0))[0]
        if len(cols) < 50:
            continue
        ys = [np.where(comp[:, c])[0].mean() for c in cols]
        spreads.append(np.std(ys))
    return float(np.mean(spreads)) if spreads else 0.0


def test_straighten_reduces_baseline_wobble():
    flat = _text_page()
    warped = _warp(flat)
    before = _line_wobble(warped)
    after = _line_wobble(straighten(warped))
    assert before > 1.5, before
    assert after < before * 0.5, (before, after)


def test_straighten_is_noop_on_flat_page():
    flat = _text_page()
    out = straighten(flat)
    assert np.array_equal(out, flat)
