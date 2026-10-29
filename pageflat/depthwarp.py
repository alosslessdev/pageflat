"""Dewarp through monocular depth: the surface tells you how the page bends.

OCR gives you geometry only where there is text — between lines you are
guessing, and the guess chases OCR noise. A monocular depth model sees the
whole sheet, including the blank parts, and reports how far each pixel is from
the camera. Paper curl is exactly a variation of that distance.

The two are combined the way a measurement is normally combined with a model:
the depth supplies the *shape* of the deformation, and the detected text lines
supply its *scale* — one regression coefficient over all the line samples,
instead of a separate correction per block.

Depth Anything V2 (small) is used; the depth is relative, which is all the
shape needs.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

_MODEL = None
_PROC = None
SIDE = 1024


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = "depth-anything/Depth-Anything-V2-Small-hf"
        _PROC = AutoImageProcessor.from_pretrained(name)
        _MODEL = AutoModelForDepthEstimation.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL.to(device).eval()
    return _MODEL, _PROC


def depth_map(img_bgr: np.ndarray, side: int = SIDE,
              engine: Optional[str] = None) -> tuple[np.ndarray, float]:
    """Relative depth at `side` pixels, and the factor back to the input frame.

    `PAGEFLAT_DEPTH=da3` switches to Depth Anything 3, which returns the same
    relative depth *and* the camera pose; the depth alone is a drop-in.
    """
    import os

    engine = (engine or os.environ.get("PAGEFLAT_DEPTH", "v2")).lower()
    if engine in ("da3", "v3", "3"):
        from .da3 import surface_depth

        return surface_depth(img_bgr, side=side), 1.0

    import torch
    from PIL import Image

    model, proc = _load()
    h, w = img_bgr.shape[:2]
    scale = min(1.0, side / float(max(h, w)))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else img_bgr
    inputs = proc(images=Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)),
                  return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    depth = torch.nn.functional.interpolate(out.predicted_depth.unsqueeze(1),
                                            size=small.shape[:2], mode="bicubic",
                                            align_corners=False)[0, 0].cpu().numpy()
    return depth.astype(np.float32), (1.0 / scale if scale > 0 else 1.0)


def plane_residual(depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Depth minus the best-fit plane: the part that is curvature, not slant."""
    h, w = depth.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    sel = np.ones_like(depth, bool) if mask is None else mask.astype(bool)
    if sel.sum() < 100:
        return np.zeros_like(depth)
    A = np.stack([xs[sel].ravel(), ys[sel].ravel(), np.ones(int(sel.sum()))], axis=1)
    coef, *_ = np.linalg.lstsq(A, depth[sel].ravel(), rcond=None)
    plane = coef[0] * xs + coef[1] * ys + coef[2]
    return (depth - plane).astype(np.float32)


def rectify(img: np.ndarray, lines=None, depth: Optional[np.ndarray] = None,
            back: float = 1.0, k_values=None,
            precomputed: Optional[np.ndarray] = None,
            engine: Optional[str] = None) -> tuple[np.ndarray, dict]:
    """Flatten using the depth's shape, calibrated by the measured line bow.

    The depth says *how* the sheet bends; the text lines say *how much* that
    bend moves the image. One global scale `k` is chosen by search: the value
    that leaves the least bow behind. That keeps the correction on the depth's
    shape instead of chasing per-block OCR noise.
    """
    from .detect import text_lines
    from .blocks import blocks_of

    h, w = img.shape[:2]
    if precomputed is not None:
        unit = precomputed
    else:
        if depth is None:
            depth, back = depth_map(img, engine=engine)
        resid = plane_residual(depth)
        resid = cv2.resize(resid, (w, h), interpolation=cv2.INTER_CUBIC)
        resid = resid - float(np.median(resid))
        span = float(np.percentile(np.abs(resid), 95)) or 1.0
        unit = resid / span
    if lines is None:
        lines = text_lines(img)

    from .sharp import row_sharpness

    def bow(im) -> float:
        """Rank the k values by a cheap profile score; the winner is checked
        with the real block measure afterwards (twelve detector passes per page
        is what made this candidate the slowest of the set)."""
        return -row_sharpness(im)

    base = bow(img)
    if k_values is None:
        k_values = (-120.0, -60.0, -30.0, 30.0, 60.0, 120.0)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    best = (base, 0.0, img)
    for k in k_values:
        map_x = np.clip(gx + (unit * k).astype(np.float32), 0, w - 1)
        out = cv2.remap(img, map_x, gy, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        b = bow(out)
        if b < best[0]:
            best = (b, k, out)
    if best[1] == 0.0 or best[0] > 0.85 * base:
        return img, {"corrected": False, "bow": round(base, 1)}

    def true_drift(im):
        found = text_lines(im)
        blocks = blocks_of(found)
        if not blocks:
            return None
        vals = sorted(b.rect_error for b in blocks)
        return vals[len(vals) // 2]

    before, after = true_drift(img), true_drift(best[2])
    if before is None or after is None or after > 0.9 * before:
        return img, {"corrected": False, "bow": round(before or 0, 1)}
    return best[2], {"corrected": True, "k": best[1],
                     "bow_before": round(before, 1), "bow_after": round(after, 1)}


def developable_residual(resid: np.ndarray) -> np.ndarray:
    """Project the depth residual onto the developable (cylinder) subspace.

    A sheet of paper does not stretch (Ishikawa's MonoBFS/BFS uses exactly this
    constraint): it bends along a curve and stays ruled across it. So the
    residual is, to first order, one profile swept along one direction. Taking
    the dominant rank-one term of the residual keeps that bend and discards the
    rest of the depth noise — far fewer degrees of freedom than a free field.
    """
    r = resid - resid.mean()
    u, s, vt = np.linalg.svd(r, full_matrices=False)
    if s.size == 0:
        return np.zeros_like(resid)
    rank1 = np.outer(u[:, 0] * s[0], vt[0])
    return rank1.astype(np.float32)


def rectify_developable(img: np.ndarray, depth: Optional[np.ndarray] = None,
                        k_values=None) -> tuple[np.ndarray, dict]:
    """Depth shape, paper's physics, spline mapping, rectangle-rule scale.

    * Depth (Depth Anything) says how far each point is from the camera;
    * the plane is removed, and the remainder is forced onto the developable
      subspace (a sheet bends but does not stretch);
    * the resulting profile becomes a thin-plate-spline displacement, so the
      correction is smooth rather than per-row;
    * the single scale is chosen by the rectangle rule, and OCR has the last
      word.
    """
    if depth is None:
        depth, back = depth_map(img)
    h, w = img.shape[:2]
    resid = developable_residual(plane_residual(depth))
    resid = cv2.resize(resid, (w, h), interpolation=cv2.INTER_CUBIC)
    resid = resid - float(np.median(resid))
    span = float(np.percentile(np.abs(resid), 95)) or 1.0
    unit = resid / span
    if k_values is None:
        k_values = (-400., -300., -200., -120., -80., -40., 40., 80., 120., 200., 300., 400.)
    return rectify(img, depth=None, back=back, k_values=k_values,
                   precomputed=unit)
