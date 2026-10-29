"""How bent the sheet's own edge is, measured where it is still visible.

A page has at least one geometric fact in it even without text: the paper's edge.
On a flat sheet photographed flat, that edge is a straight line, so how far the
traced edge wanders from a straight line is a drift measurement that does not
depend on content.

Two things make it honest or useless, both learned the hard way on this corpus:

* the mask must be **paper**, not whatever the segmenter called page. A
  newspaper lying on red bedding has a page mask that covers the bedding too, and
  the "edge" then belongs to a crumpled duvet — 3126 px of drift on a flat sheet.
  `perspective._paper_like` removes it: paper is bright and unsaturated, bedding
  is one or the other but rarely both.
* the edge must be **inside the frame**. When the page has been cropped to its
  content the edge *is* the frame, and the number would describe the crop.
  That case returns None rather than a lie.

Note what the number means: on `IMG_20260613_192849` the sheet's left edge bows
by 57 % — because the newspaper is draped over a duvet. Its *printed text* came
out at 0.18 % of width. Both are true: the paper is bent, the print is straight.
The edge says how much physical work the page needs; the text says whether the
reader got a flat page.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeDrift:
    left: float | None
    right: float | None
    top: float | None
    bottom: float | None
    width: int
    height: int

    @property
    def worst(self) -> float | None:
        vals = [v for v in (self.left, self.right, self.top, self.bottom) if v is not None]
        return max(vals) if vals else None

    @property
    def percent_of_width(self) -> float | None:
        w = self.worst
        return None if w is None else 100.0 * w / self.width


def _deviation(curve: np.ndarray, step: int = 2) -> float:
    t = np.arange(curve.size) * step
    A = np.stack([t, np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(A, curve, rcond=None)
    resid = np.abs(curve - A @ coef)
    # The median, not a high percentile: a fanned page has a genuinely ragged
    # outline, and one notch would then stand for the whole edge. The median asks
    # how far the edge sits from a straight line for most of its length.
    return float(np.median(resid))


def _edge(mask: np.ndarray, side: str, border: int = 4) -> float | None:
    h, w = mask.shape
    xs = np.full(h, np.nan)
    for y in range(h):
        idx = np.flatnonzero(mask[y] > 0)
        if idx.size < 8:
            continue
        xs[y] = idx[0] if side == "left" else idx[-1]
    ok = ~np.isnan(xs)
    if ok.sum() < 0.3 * h:
        return None
    med = float(np.median(xs[ok]))
    if side == "left" and med <= border:
        return None          # cropped to the sheet: this is the frame, not paper
    if side == "right" and med >= w - 1 - border:
        return None
    xs = np.interp(np.arange(h), np.flatnonzero(ok), xs[ok])
    # smooth the pixel-level chatter; the thing being measured is the drape
    k = max(3, h // 100 | 1)
    xs = cv2.GaussianBlur(xs.reshape(-1, 1), (1, k), 0).ravel()
    return _deviation(xs)


def edge_drift(img_bgr: np.ndarray, mask: np.ndarray | None = None,
               min_area_frac: float = 0.2) -> EdgeDrift:
    """Drift of each sheet edge, in pixels of the image given.

    `mask` is the page mask from the detector; when absent, paper-likeness alone
    is used, which is enough when the sheet lies on a plain surface.
    """
    from .perspective import _paper_like

    if mask is None:
        m = _paper_like(img_bgr)
    else:
        m = (mask >= 0.5) & _paper_like(img_bgr)
    if m.mean() < min_area_frac:
        m = _paper_like(img_bgr)
    if m.mean() < min_area_frac:
        return EdgeDrift(None, None, None, None, img_bgr.shape[1], img_bgr.shape[0])

    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return EdgeDrift(None, None, None, None, img_bgr.shape[1], img_bgr.shape[0])
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)

    res = {"left": _edge(filled, "left"), "right": _edge(filled, "right")}
    t = cv2.transpose(filled)
    res["top"] = _edge(t, "left")
    res["bottom"] = _edge(t, "right")
    return EdgeDrift(res["left"], res["right"], res["top"], res["bottom"],
                     img_bgr.shape[1], img_bgr.shape[0])


def straighten_edge(img_bgr: np.ndarray, side: str = "left",
                    max_shift_frac: float = 0.06) -> tuple[np.ndarray, dict]:
    """Shear rows so the sheet's own edge becomes a straight vertical.

    A homography cannot remove a physical curl: the page's boundary is a curve and
    it stays one after the quad is mapped to a rectangle. Shifting each row by the
    amount its edge is off a straight line is the smallest motion that makes the
    boundary straight, and it carries the curl out of the content with it.
    """
    from .perspective import _paper_like

    h, w = img_bgr.shape[:2]
    paper = _paper_like(img_bgr).astype(np.uint8)
    m8 = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return img_bgr, {"corrected": False}
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)

    xs = np.full(h, np.nan)
    for y in range(h):
        idx = np.flatnonzero(filled[y] > 0)
        if idx.size < 8:
            continue
        xs[y] = idx[0] if side == "left" else idx[-1]
    ok = ~np.isnan(xs)
    if ok.sum() < 0.3 * h:
        return img_bgr, {"corrected": False}
    xs = np.interp(np.arange(h), np.flatnonzero(ok), xs[ok])
    t = np.arange(h)
    A = np.stack([t, np.ones_like(t)], 1)
    coef, *_ = np.linalg.lstsq(A, xs, rcond=None)
    dx = xs - (A @ coef)
    if np.abs(dx).max() > max_shift_frac * w:
        return img_bgr, {"corrected": False, "reason": "shift too large"}
    if np.median(np.abs(dx)) < 1.0:
        return img_bgr, {"corrected": False, "reason": "already straight"}

    k = max(3, h // 60 | 1)
    dx = cv2.GaussianBlur(dx.reshape(-1, 1), (1, k), 0).ravel()
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(gx - dx.astype(np.float32)[:, None], 0, w - 1)
    out = cv2.remap(img_bgr, map_x, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "median_shift": round(float(np.median(np.abs(dx))), 1),
                 "max_shift": round(float(np.abs(dx).max()), 1)}
