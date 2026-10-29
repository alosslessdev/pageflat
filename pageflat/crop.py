"""Crop away the surface the page was lying on.

Many of the photographs are a sheet lying on a cloth, a table, or a pile of
other papers. The page mask then covers the whole pile, the warp keeps the
cloth, and the result reads as "warped" when it is really "the background is
still there" — the printed sheet itself is fine.

Print has high local contrast; cloth, wood and paper stacks are smooth at the
scale of a letter. That distinguishes them without a model. The crop is only
taken when nothing printed lies outside it, so a title page with one line does
not get cut down to that line.
"""
from __future__ import annotations

import cv2
import numpy as np


def printed_mask(img: np.ndarray, window: int = 9, threshold: float = 18.0) -> np.ndarray:
    """Where the image carries print: high local variance at letter scale."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (window, window))
    sq = cv2.blur(gray * gray, (window, window))
    std = np.sqrt(np.clip(sq - mean * mean, 0.0, None))
    # print is dark *and* sharp. Texture alone (cloth, wood) is sharp but not
    # dark, which is what separates a sheet lying on fabric from the fabric.
    mask = ((std > threshold) & (gray < 150)).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))


def crop_to_content(img: np.ndarray, pad: float = 0.03, outside_tol: float = 0.02,
                    min_gain: float = 0.10) -> tuple[np.ndarray, dict]:
    """Crop to the printed area when the rest is a smooth surface.

    Returns the (possibly cropped) image and diagnostics. ``cropped`` is False
    when the print already fills the frame, when the crop would remove more
    than ``outside_tol`` of the printed pixels, or when it would remove less
    than ``min_gain`` of the area (not worth the change).
    """
    info = {"cropped": False}
    mask = printed_mask(img)
    total = int(mask.sum())
    if total == 0:
        return img, info
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return img, info
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[biggest]
    H, W = mask.shape
    m = int(pad * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    outside = total - int(mask[y0:y1, x0:x1].sum())
    info.update({"box": [int(x0), int(y0), int(x1), int(y1)],
                 "outside_frac": round(outside / total, 4),
                 "area_gain": round(1.0 - ((x1 - x0) * (y1 - y0)) / float(W * H), 3)})
    if outside > outside_tol * total:
        info["reason"] = "print outside the crop"
        return img, info
    if 1.0 - ((x1 - x0) * (y1 - y0)) / float(W * H) < min_gain:
        info["reason"] = "crop too small to matter"
        return img, info
    info["cropped"] = True
    return img[y0:y1, x0:x1], info


def trim_borders(img: np.ndarray, max_frac: float = 0.15, need_paper: float = 0.6,
                 verify: bool = True, min_gain: float = 0.9
                 ) -> tuple[np.ndarray, dict]:
    """Take off the sliver of table, floor or neighbouring page at the frame edge.

    `crop_to_content` works on where the *print* is, which leaves a strip of
    whatever the sheet lies on wherever the print stops short of the paper edge.
    This works on the other question: is the outermost line of cells paper? It
    walks in from each side while the band is mostly not paper, and stops at the
    first band that is — so a page whose print runs to its edge is untouched, and
    a page with a dark table strip loses exactly that strip.

    Bounded by `max_frac` of the dimension so a dark picture at the edge can
    never eat the page.
    """
    from .perspective import _paper_like

    paper = _paper_like(img)
    h, w = paper.shape
    lim_y, lim_x = int(max_frac * h), int(max_frac * w)

    def band_ok(sl: tuple[slice, slice]) -> bool:
        return float(paper[sl].mean()) >= need_paper

    top = bot = left = right = 0
    while top < lim_y and not band_ok((slice(top, top + 1), slice(0, w))):
        top += 1
    while bot < lim_y and not band_ok((slice(h - 1 - bot, h - bot), slice(0, w))):
        bot += 1
    body = slice(top, h - bot)
    while left < lim_x and not band_ok((body, slice(left, left + 1))):
        left += 1
    while right < lim_x and not band_ok((body, slice(w - 1 - right, w - right))):
        right += 1

    if top + bot >= h - 8 or left + right >= w - 8:
        return img, {"trimmed": False}
    out = img[top:h - bot, left:w - right]
    info = {"trimmed": bool(top or bot or left or right),
            "box": [left, top, w - right, h - bot]}
    if not info["trimmed"]:
        return out, info

    # A strip of table is worth removing; a strip of the page's own content is
    # not. The only test that knows the difference is what the page still reads,
    # so a trim that costs OCR is undone — the same rule the warp candidates are
    # held to. Cheap when the trim is small, so only large trims are checked.
    big = (top + bot) > 0.05 * h or (left + right) > 0.05 * w
    if verify and big:
        from .pipeline import ocr_quality

        before, after = ocr_quality(img), ocr_quality(out)
        if before and after is not None and after < min_gain * before:
            info["rejected"] = "ocr"
            info["trimmed"] = False
            return img, info
    return out, info
