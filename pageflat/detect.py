"""Text lines from a learned detector, not from OCR layout.

Tesseract's line boxes are the by-product of recognition: they merge a heading
with the paragraph under it, split a line at a wide word gap, and jitter by a
few pixels, which is too noisy to define a block's edge. docTR's detector is
trained to find text and returns clean word boxes; grouped by baseline they
give the geometry the rectification needs.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_DETECTOR = None
_SIDE = 1400


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from doctr.models import detection_predictor
        _DETECTOR = detection_predictor(pretrained=True, assume_straight_pages=True)
    return _DETECTOR


def word_boxes(img: np.ndarray, side: int = _SIDE) -> list[tuple[float, float, float, float]]:
    """Word boxes in the input image's coordinates."""
    h, w = img.shape[:2]
    scale = min(1.0, side / float(max(h, w)))
    small = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img
    out = _detector()([small])[0]
    boxes = []
    for row in np.asarray(out["words"]).reshape(-1, 5):
        x0, y0, x1, y1, score = row
        if score < 0.3:
            continue
        boxes.append((x0 * w, y0 * h, x1 * w, y1 * h))
    return boxes


def lines_from_words(boxes: list[tuple[float, float, float, float]],
                     height_tol: float = 0.6) -> list[tuple[float, float, float, float]]:
    """Group word boxes into lines by baseline, returns line bounding boxes."""
    if not boxes:
        return []
    heights = np.array([b[3] - b[1] for b in boxes])
    tol = max(4.0, height_tol * float(np.median(heights)))
    order = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    groups: list[list] = []
    centre = None
    for box in order:
        cy = (box[1] + box[3]) / 2.0
        if groups and abs(cy - centre) <= tol:
            groups[-1].append(box)
            centre = float(np.median([(b[1] + b[3]) / 2.0 for b in groups[-1]]))
        else:
            groups.append([box])
            centre = cy
    lines = []
    for group in groups:
        x0 = min(b[0] for b in group)
        y0 = min(b[1] for b in group)
        x1 = max(b[2] for b in group)
        y1 = max(b[3] for b in group)
        lines.append((x0, y0, x1, y1))
    return lines


def text_lines(img: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Lines in the image's own coordinates, via detection + baseline grouping."""
    return lines_from_words(word_boxes(img))
