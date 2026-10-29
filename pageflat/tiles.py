"""Text detection over tiles, so a 64 MP page is measured, not skimmed.

The detector resizes whatever it is given to its own input size, so handing it a
9248-px page is the same as handing it a 1024-px thumbnail of that page: the body
text is one or two pixels tall and comes back missing. On a full source page that
meant twenty-four lines and three blocks — the headlines — which is not a
measurement of the page's straightness, it is a measurement of its masthead.

Cutting the page into native-resolution tiles and detecting each one keeps the
body text legible: the same page gives 103 lines instead of 24.
"""
from __future__ import annotations

import numpy as np

TILE = 2048
OVERLAP = 96


def text_lines_tiled(img: np.ndarray, tile: int = TILE, overlap: int = OVERLAP) -> list:
    """Lines in full-image coordinates, detected on tiles of `tile` pixels."""
    from .detect import text_lines

    h, w = img.shape[:2]
    if max(h, w) <= tile:
        return text_lines(img)

    step = tile - overlap
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    out = []
    for y in ys:
        for x in xs:
            patch = img[max(0, y):min(h, y + tile), max(0, x):min(w, x + tile)]
            if patch.size == 0:
                continue
            for line in text_lines(patch):
                out.append((line[0] + x, line[1] + max(0, y),
                            line[2] + x, line[3] + max(0, y)))
    return out
