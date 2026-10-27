"""GPU (torch) versions of pageflat's per-image pixel steps.

Every public function accepts either an ``(H, W, 3)`` uint8 BGR numpy array or
a uint8 torch tensor in the same layout, and returns the same kind of object on
the same device.  That lets a caller keep a page resident on the GPU across a
whole chain of steps while the numpy-based call sites keep working unchanged.

The implementations reproduce the CPU originals (``geom.warp_quad``,
``io.resize_long_side``, ``illum.flatten_illumination``, ``illum.scan_finish``)
to within rounding noise; ``tests/test_gpu_ops.py`` pins that down on real
scans.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from . import illum

__all__ = [
    "default_device",
    "warp_quad_gpu",
    "resize_long_side_gpu",
    "flatten_illumination_gpu",
    "scan_finish_gpu",
]

ImgLike = Union[np.ndarray, torch.Tensor]

_BG_GRID = 160          # grid size used by illum.background_from_model
_WARP_BAND = 1024       # output rows per grid_sample call (bounds grid memory)
_RESIZE_BAND_PX = 1200  # source rows per resize band (bounds cumsum memory)


# --------------------------------------------------------------------- helpers
def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve(device: Optional[Union[str, torch.device]]) -> torch.device:
    return default_device() if device is None else torch.device(device)


def _as_u8(img: ImgLike, device: torch.device):
    """``(H, W, 3)`` uint8 BGR tensor on `device`, plus "input was numpy" flag."""
    if isinstance(img, torch.Tensor):
        if img.dtype != torch.uint8:
            raise TypeError(f"expected a uint8 tensor, got {img.dtype}")
        return img.to(device), None
    return torch.from_numpy(np.ascontiguousarray(img)).to(device), True


def _as_numpy(img: ImgLike) -> np.ndarray:
    return img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img


def _restore(t: torch.Tensor, to_numpy) -> Union[np.ndarray, torch.Tensor]:
    return t.detach().cpu().numpy() if to_numpy else t


def _round_u8(x: torch.Tensor) -> torch.Tensor:
    """cv2's saturate_cast<uchar>: round half to even, then clamp."""
    return x.round_().clamp_(0, 255).to(torch.uint8)


# ------------------------------------------------------------------ perspective
def warp_quad_gpu(img_bgr: ImgLike, quad, out_w: int, out_h: int, *,
                  device: Optional[Union[str, torch.device]] = None) -> ImgLike:
    """Rectify `quad` (tl, tr, br, bl) to an ``out_h x out_w`` BGR image.

    Equivalent to ``geom.warp_quad(img, quad, out_size=(out_w, out_h))``:
    same destination corner convention (0 .. out-1), cubic interpolation and
    border replication.
    """
    out_w, out_h = int(out_w), int(out_h)
    if out_w < 2 or out_h < 2:
        raise ValueError("out_w and out_h must be >= 2")
    dev = _resolve(device)
    src, to_numpy = _as_u8(img_bgr, dev)
    h, w = src.shape[:2]
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)

    # Destination pixel -> source pixel.  cv2.warpPerspective inverts the
    # forward quad->rect matrix internally; solving for the inverse directly is
    # the same map with less floating-point round-trip.
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   np.float32)
    m = cv2.getPerspectiveTransform(dst, quad).astype(np.float32)
    a00, a01, a02 = (float(v) for v in m[0])
    a10, a11, a12 = (float(v) for v in m[1])
    a20, a21, a22 = (float(v) for v in m[2])

    # Only the quad's bounding box can be sampled, so upload just that (plus a
    # cubic kernel's reach).  Sampling never leaves the box, so BORDER_REPLICATE
    # at the crop edge is the same pixel as at the full-image edge.
    x0 = max(0, int(np.floor(quad[:, 0].min())) - 4)
    x1 = min(w, int(np.ceil(quad[:, 0].max())) + 4)
    y0 = max(0, int(np.floor(quad[:, 1].min())) - 4)
    y1 = min(h, int(np.ceil(quad[:, 1].max())) + 4)
    if (x1 - x0) * (y1 - y0) > 0.9 * w * h:  # cropping is not worth it
        x0, y0, x1, y1 = 0, 0, w, h
    cw, ch = x1 - x0, y1 - y0
    # .to(float) with an explicit memory format keeps the upload to a single
    # NCHW-contiguous copy (float() alone would preserve the permuted strides).
    crop = src[y0:y1, x0:x1].permute(2, 0, 1).unsqueeze(0).to(
        dtype=torch.float32, memory_format=torch.contiguous_format)

    sx = torch.arange(out_w, dtype=torch.float32, device=dev).view(1, -1)
    out = torch.empty((out_h, out_w, 3), dtype=torch.uint8, device=dev)
    for yb in range(0, out_h, _WARP_BAND):
        ye = min(out_h, yb + _WARP_BAND)
        sy = torch.arange(yb, ye, dtype=torch.float32, device=dev).view(-1, 1)
        shape = (ye - yb, out_w)
        den = torch.full(shape, a22, dtype=torch.float32, device=dev)
        den.add_(sx, alpha=a20)
        den.add_(sy, alpha=a21)
        u = torch.full(shape, a02, dtype=torch.float32, device=dev)
        u.add_(sx, alpha=a00)
        u.add_(sy, alpha=a01)
        v = torch.full(shape, a12, dtype=torch.float32, device=dev)
        v.add_(sx, alpha=a10)
        v.add_(sy, alpha=a11)
        grid = torch.empty((1, ye - yb, out_w, 2), dtype=torch.float32, device=dev)
        u.div_(den).sub_(x0).add_(0.5).mul_(2.0 / cw).sub_(1.0)
        v.div_(den).sub_(y0).add_(0.5).mul_(2.0 / ch).sub_(1.0)
        grid[0, :, :, 0] = u
        grid[0, :, :, 1] = v
        del u, v, den
        o = F.grid_sample(crop, grid, mode="bicubic", padding_mode="border",
                          align_corners=False)
        out[yb:ye] = _round_u8(o[0].permute(1, 2, 0))
        del grid, o
    del crop
    return _restore(out, to_numpy)


# ---------------------------------------------------------------------- resize
def _area_weights(n: int, out_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-resampling weights for n -> out_len as (i0, f0, i1, f1) index/frac.

    Output pixel j covers the source interval [j*r, (j+1)*r); every source pixel
    is weighted by how much of it falls inside, i.e. exactly cv2.INTER_AREA for
    a downscale.
    """
    r = n / float(out_len)
    j = np.arange(out_len, dtype=np.float64)
    x0 = j * r
    x1 = x0 + r
    i0 = np.floor(x0).astype(np.int64)
    i1 = np.floor(x1).astype(np.int64)  # may equal n: that is the closed end
    return i0, (x0 - i0).astype(np.float32), i1, (x1 - i1).astype(np.float32)


def _area_resample_1d(x: torch.Tensor, dim: int, i0: torch.Tensor,
                      f0: torch.Tensor, i1: torch.Tensor, f1: torch.Tensor,
                      width: float, length: int) -> torch.Tensor:
    """Area-average `x` along `dim` using prefix sums (O(n), no kernel weights)."""
    shape = [1] * x.dim()
    shape[dim] = length
    f0 = f0.view(shape)
    f1 = f1.view(shape)
    # Running sums on a signal centred on zero keep the prefix magnitudes small,
    # so the differences below stay exact to ~1e-3 of a level.
    c = torch.cumsum(x, dim=dim)
    zero = torch.zeros_like(c.narrow(dim, 0, 1))
    C = torch.cat([zero, c], dim=dim)
    del c, zero
    # A(x) = C[floor(x)] + frac(x) * pixel[floor(x)] is the integral of the
    # piecewise-constant signal; the mean over [x0, x1) is (A(x1)-A(x0))/r.
    left = C.index_select(dim, i0).add_(x.index_select(dim, i0).mul_(f0))
    last = i1.clamp(0, x.shape[dim] - 1)
    right = C.index_select(dim, i1).add_(x.index_select(dim, last).mul_(f1))
    right.sub_(left).div_(width)
    return right


def _resize_area_u8(src: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """uint8 HWC tensor -> uint8 (out_h, out_w, C) area-resampled downscale."""
    h, w = src.shape[:2]
    if (out_h, out_w) == (h, w):
        return src
    if out_w > w or out_h > h:  # INTER_AREA is only an area filter when shrinking
        raise ValueError("_resize_area_u8 only downsamples")
    dev = src.device
    wy = _area_weights(h, out_h)
    ry = h / float(out_h)
    wx = [torch.from_numpy(a).to(dev) for a in _area_weights(w, out_w)] if out_w != w else None
    band = max(16, int(_RESIZE_BAND_PX / ry))
    out = torch.empty((out_h, out_w, src.shape[2]), dtype=torch.uint8, device=dev)
    for ob in range(0, out_h, band):
        oe = min(out_h, ob + band)
        sy0 = int(wy[0][ob])
        sy1 = int(min(wy[2][oe - 1], h - 1)) + 1
        b = src[sy0:sy1].permute(2, 0, 1).float()
        b.sub_(128.0)
        if wx is not None:
            b = _area_resample_1d(b, 2, wx[0], wx[1], wx[2], wx[3], w / float(out_w), out_w)
        if out_h != h:
            i0 = torch.from_numpy(wy[0][ob:oe] - sy0).to(dev)
            f0 = torch.from_numpy(wy[1][ob:oe]).to(dev)
            i1 = torch.from_numpy(wy[2][ob:oe] - sy0).to(dev)
            f1 = torch.from_numpy(wy[3][ob:oe]).to(dev)
            b = _area_resample_1d(b, 1, i0, f0, i1, f1, ry, oe - ob)
        b.add_(128.0)
        out[ob:oe] = _round_u8(b.permute(1, 2, 0))
        del b
    return out


def resize_long_side_gpu(img_bgr: ImgLike, long_side: int, *,
                         device: Optional[Union[str, torch.device]] = None
                         ) -> Tuple[ImgLike, float]:
    """Downscale so the long side is `long_side`; returns (image, scale).

    Matches ``io.resize_long_side`` (INTER_AREA, same output size rounding).
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) <= long_side:
        return img_bgr, 1.0
    scale = long_side / float(max(h, w))
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    src, to_numpy = _as_u8(img_bgr, _resolve(device))
    return _restore(_resize_area_u8(src, out_h, out_w), to_numpy), scale


# --------------------------------------------------------------- illumination
def _background_gpu(coef: np.ndarray, h: int, w: int, device: torch.device,
                    size: int = _BG_GRID) -> torch.Tensor:
    """Paper-lighting surface (h, w, C), same field as illum.background_from_model.

    The model is fitted to a coarse grid, then that grid is bilinearly resized to
    the image, so a full-resolution evaluation at the corresponding source
    coordinates reproduces it (to <0.01 of a level) without a 160x160 upsample.
    """
    c = torch.from_numpy(np.asarray(coef, np.float32)).to(device)
    u = ((torch.arange(w, dtype=torch.float32, device=device) + 0.5) * (size / w) - 0.5)
    v = ((torch.arange(h, dtype=torch.float32, device=device) + 0.5) * (size / h) - 0.5)
    u.div_(size - 1).clamp_(0.0, 1.0).unsqueeze_(0).unsqueeze_(-1)     # (1, w, 1)
    v.div_(size - 1).clamp_(0.0, 1.0).unsqueeze_(-1).unsqueeze_(-1)    # (h, 1, 1)
    bg = torch.empty((h, w, c.shape[1]), dtype=torch.float32, device=device)
    bg.copy_(c[0])
    bg.addcmul_(c[1], u)
    bg.addcmul_(c[2], v)
    bg.addcmul_(c[3], u * u)
    bg.addcmul_(c[4], u * v)
    bg.addcmul_(c[5], v * v)
    return bg.clamp_min_(8.0)


def _percentiles(x: torch.Tensor, qs) -> list[float]:
    """``np.percentile(x, qs)`` (linear method) from GPU order statistics.

    ``torch.quantile`` refuses tensors above 2**24 elements; ``kthvalue`` has no
    such limit and gives the exact interpolation endpoint values.
    """
    x = x.reshape(-1)
    n = x.numel()
    out = []
    for q in qs:
        pos = q * (n - 1)
        lo = int(np.floor(pos))
        hi = min(lo + 1, n - 1)
        a = x.kthvalue(lo + 1)[0].item()
        b = x.kthvalue(hi + 1)[0].item() if hi != lo else a
        out.append(a + (b - a) * (pos - lo))
    return out


def flatten_illumination_gpu(img_bgr: ImgLike, white: float = 245.0,
                             strength: float = 1.0, *,
                             device: Optional[Union[str, torch.device]] = None
                             ) -> ImgLike:
    """``illum.flatten_illumination`` with the fit on CPU and the divide on GPU."""
    coef = illum.fit_background_model(_as_numpy(img_bgr))
    dev = _resolve(device)
    src, to_numpy = _as_u8(img_bgr, dev)
    h, w = src.shape[:2]
    scale = torch.div(white, _background_gpu(coef, h, w, dev))
    if strength != 1.0:
        scale.sub_(1.0).mul_(strength).add_(1.0)
    out = src.float().mul_(scale)
    del scale
    return _restore(out.clamp_(0, 255).to(torch.uint8), to_numpy)


# --------------------------------------------------------------- scan finish
def scan_finish_gpu(img_bgr: ImgLike, paper_white: bool = True, *,
                    device: Optional[Union[str, torch.device]] = None) -> ImgLike:
    """``illum.scan_finish`` with the tonal pass and the LUT applied on GPU."""
    dev = _resolve(device)
    src, to_numpy = _as_u8(img_bgr, dev)
    h, w = src.shape[:2]
    if paper_white:
        coef = illum.fit_background_model(_as_numpy(img_bgr))
        # grey-world gains, as cv2.mean does it: means over each channel
        means = src.reshape(-1, 3).sum(dim=0, dtype=torch.float32)
        means.div_(h * w)
        wb = means.mean() / means
        scale = torch.div(245.0, _background_gpu(coef, h, w, dev))
        scale.mul_(wb)
        out = src.float().mul_(scale).clamp_(0, 255).to(torch.uint8)
        del scale
    else:
        means = src.reshape(-1, 3).sum(dim=0, dtype=torch.float32)
        means.div_(h * w)
        gains = means.mean() / means.clamp_min(1e-3)
        out = src.float().mul_(gains).clamp_(0, 255).to(torch.uint8)

    f = out.float()
    # same fixed-point luma as cv2.cvtColor(BGR2GRAY); terms stay exact in f32
    gray = f.view(-1, 3).mv(torch.tensor([1868.0, 9617.0, 4899.0], device=dev))
    gray.add_(8192.0).div_(16384.0).floor_()
    lo, hi = _percentiles(gray, (0.003, 0.997))
    del gray
    if hi - lo < 1:
        return _restore(out, to_numpy)
    return _restore(f.sub_(lo).mul_(255.0 / (hi - lo)).clamp_(0, 255)
                    .to(torch.uint8), to_numpy)
