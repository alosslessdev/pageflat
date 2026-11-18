"""Bound the page mask with two witnesses: the sheet (SAM) and the print.

The learned page model calls anything paper-coloured "page", and a pale table is
paper-coloured: on `IMG_20251229_122443` its mask reaches y=893 of a 900-px frame
while the sheet's bottom edge is at ~780. Every single-signal rule was tried and
failed (see the end of this file); what works is asking two other witnesses and
requiring the mask to agree with at least one of them:

* **SAM, prompted with a grid of points**, follows the sheet's boundary — its bottom
  on that page is 780, and an independent vision model put the same edge at y≈867 of
  a 1000-grid, i.e. the same place;
* **the print**, which must not be cut: SAM is sometimes *tighter than the page* (its
  left boundary there is x=127 against the mask's x=12), so print is the floor.

So: `keep = docTR & (SAM | print)`. A strip of table is neither the sheet nor print,
and goes; a part of the page that SAM missed but print covers stays.
"""
from __future__ import annotations

import cv2
import numpy as np


def print_region(img_bgr: np.ndarray, dark: int = 110, close: int = 25) -> np.ndarray:
    """Dark strokes closed over the gaps *between lines*, not over blank margins.

    `crop.printed_mask` is a texture detector and closes over the whole sheet, so it
    would let the mask keep the table too. A 25-px closing bridges line spacing and
    stops short of a blank margin — which is exactly the distinction that matters at
    the bottom of a page.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ink = (gray < dark).astype(np.uint8) * 255
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                            np.ones((close, close), np.uint8)) > 0


def bound_mask(img_bgr: np.ndarray, mask: np.ndarray,
               sam_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Keep the part of `mask` that SAM or the print also claims."""
    base = mask >= 0.5
    print_area = print_region(img_bgr)
    witnesses = print_area if sam_mask is None else ((sam_mask >= 0.5) | print_area)
    keep = base & witnesses
    # never return something that lost most of the sheet
    if keep.sum() < 0.35 * base.sum():
        return mask, {"bounded": False, "reason": "would cut too much"}
    ys, xs = np.where(keep)
    ys0, xs0 = np.where(base)
    info = {"bounded": True,
            "before": [int(xs0.min()), int(ys0.min()), int(xs0.max()), int(ys0.max())],
            "after": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "sam": sam_mask is not None}
    return keep.astype(np.float32), info


def extend_to_lines(img_bgr: np.ndarray, mask: np.ndarray,
                    pad: int = 8) -> tuple[np.ndarray, dict]:
    """Grow the mask until no detected text line is cut by it.

    SAM follows the sheet better than the learned page model, but it can come back
    tighter than the page (on `IMG_20251229_122443` its left boundary is x=127 while
    the sheet starts near x=12). The one thing a page may not lose is its own text,
    so any line whose box crosses the mask boundary pulls the boundary out to the
    line's far side.
    """
    from .charlines import detect_lines

    h, w = img_bgr.shape[:2]
    keep = mask >= 0.5
    grown = 0
    for line in detect_lines(img_bgr):
        if len(line.boxes) < 3:
            continue
        x0 = min(b[0] for b in line.boxes)
        y0 = min(b[1] for b in line.boxes)
        x1 = max(b[0] + b[2] for b in line.boxes)
        y1 = max(b[1] + b[3] for b in line.boxes)
        inside = keep[max(0, int(y0)):int(y1), max(0, int(x0)):int(x1)]
        if inside.size and inside.mean() > 0.5:
            continue                     # already contained
        keep[max(0, int(y0) - pad):int(y1) + pad,
             max(0, int(x0) - pad):int(x1) + pad] = True
        grown += 1
    return keep.astype(np.float32), {"grown_for_lines": grown}
