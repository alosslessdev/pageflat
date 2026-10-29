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
_CACHE: dict = {}
_CACHE_MAX = 64


def _fingerprint(img: np.ndarray) -> bytes:
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    h.update(str(img.shape).encode())
    h.update(np.ascontiguousarray(img[:8]).tobytes())
    h.update(np.ascontiguousarray(img[-8:]).tobytes())
    return h.digest()


def text_lines_tiled(img: np.ndarray, tile: int = TILE, overlap: int = OVERLAP) -> list:
    """Lines in full-image coordinates, detected on tiles of `tile` pixels."""
    from .detect import text_lines

    # The router asks for the same page's lines several times — base, every
    # candidate, the finalists — and each ask is a full pass over four tiles.
    key = (tile, overlap, _fingerprint(img))
    if key in _CACHE:
        return _CACHE[key]

    h, w = img.shape[:2]
    if max(h, w) <= tile:
        out = text_lines(img)
        _CACHE[key] = out
        return out

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
    out = _dedupe(out)
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def _dedupe(lines: list, iou: float = 0.5) -> list:
    """Drop the duplicate detections the tile overlap creates.

    Neighbouring tiles share a band, so a line in that band is found twice with
    slightly different boxes. Left in, the same line counts twice in a block and
    the page's margins look like they vary.
    """
    if not lines:
        return lines
    kept: list = []
    for line in sorted(lines, key=lambda l: -(l[2] - l[0]) * (l[3] - l[1])):
        cx, cy = (line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0
        dup = False
        for k in kept:
            if (abs(cx - (k[0] + k[2]) / 2.0) < 0.5 * (line[2] - line[0])
                    and abs(cy - (k[1] + k[3]) / 2.0) < 0.5 * (line[3] - line[1])):
                dup = True
                break
        if not dup:
            kept.append(line)
    return kept
