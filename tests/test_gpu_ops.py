"""Equivalence of the torch (GPU) page ops against the numpy/cv2 originals.

The real-data checks need the page-flattening scan corpus; point
``PAGEFLAT_SCANS`` at a directory of output PNGs to use a different one.
Everything here is skipped when there is no CUDA device.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pageflat import geom, gpu_ops, illum, io  # noqa: E402

SCAN_DIR = Path(os.environ.get(
    "PAGEFLAT_SCANS",
    "/run/media/asdf/30620d0d-1e41-4d55-af08-b4de2fb9c517/pageflat_scans_v4",
))

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _need_free_vram(mb: int = 512) -> None:
    """Skip when the shared GPU is too full for another consumer."""
    free, _ = torch.cuda.mem_get_info()
    if free < mb * 2 ** 20:
        pytest.skip(f"only {free / 2 ** 20:.0f} MB of GPU memory free")


@pytest.fixture(scope="module")
def scans():
    """Five real 3508-px scans (the pipeline's own output)."""
    if not SCAN_DIR.is_dir():
        pytest.skip(f"no scan corpus at {SCAN_DIR}")
    out = []
    for path in sorted(SCAN_DIR.glob("*.png")):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or max(img.shape[:2]) < 1200:
            continue
        out.append((path.name, img))
        if len(out) == 5:
            break
    if len(out) < 5:
        pytest.skip("fewer than 5 real scans available")
    return out


def _page_quad(shape, inset: float = 0.03) -> np.ndarray:
    """A photographed-page-like quad: rectangular with a slight perspective."""
    h, w = shape[:2]
    dx, dy = inset * w, inset * h
    return np.array([[dx, dy * 1.2], [w - dx * 1.1, dy],
                     [w - dx * 0.8, h - dy], [dx * 1.15, h - dy * 0.9]], np.float32)


def _assert_close(got, ref, label: str) -> None:
    assert got.shape == ref.shape, f"{label}: {got.shape} != {ref.shape}"
    assert got.dtype == ref.dtype == np.uint8, f"{label}: dtype {got.dtype}"
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    assert d.max() <= 2, f"{label}: max abs diff {d.max()}"
    assert d.mean() <= 0.5, f"{label}: mean abs diff {d.mean():.3f}"


# ------------------------------------------------------------------ warp
@needs_cuda
def test_warp_quad_gpu_matches_geom_on_real_scans(scans):
    _need_free_vram()
    for name, img in scans:
        quad = _page_quad(img.shape)
        out_w, out_h = geom.quad_target_size(quad, 3508)
        ref = geom.warp_quad(img, quad, out_size=(out_w, out_h))
        _assert_close(gpu_ops.warp_quad_gpu(img, quad, out_w, out_h), ref, f"warp {name}")


@needs_cuda
def test_warp_quad_gpu_keeps_corner_orientation():
    """Each input corner colour must land in the matching output corner."""
    img = np.zeros((400, 600, 3), np.uint8)
    quad = np.array([[50, 40], [560, 70], [540, 360], [30, 340]], np.float32)
    colours = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]  # tl tr br bl
    for (x, y), colour in zip(quad, colours):
        cv2.circle(img, (int(x), int(y)), 14, colour, -1)
    out = gpu_ops.warp_quad_gpu(img, quad, 300, 200)
    assert out.shape == (200, 300, 3) and out.dtype == np.uint8
    corners = [out[3, 3], out[3, -4], out[-4, -4], out[-4, 3]]
    for got, want in zip(corners, colours):
        assert np.abs(got.astype(np.int16) - np.array(want, np.int16)).max() < 30


@needs_cuda
def test_warp_quad_gpu_preserves_flat_colour():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = (30, 60, 90)
    quad = np.array([[50, 50], [550, 70], [540, 350], [40, 340]], np.float32)
    out = gpu_ops.warp_quad_gpu(img, quad, 300, 200)
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), (30, 60, 90), atol=1)


# ---------------------------------------------------------------- resize
@needs_cuda
def test_resize_long_side_gpu_matches_io_on_real_scans(scans):
    _need_free_vram()
    for name, img in scans:
        ref, ref_scale = io.resize_long_side(img, 1600)
        got, got_scale = gpu_ops.resize_long_side_gpu(img, 1600)
        assert got_scale == ref_scale
        _assert_close(got, ref, f"resize {name}")


@needs_cuda
def test_resize_area_reproduces_cv2_on_a_non_integer_ratio():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (257, 389, 3), np.uint8)
    got, scale = gpu_ops.resize_long_side_gpu(img, 155)
    ref = cv2.resize(img, (155, 102), interpolation=cv2.INTER_AREA)
    assert got.shape == (102, 155, 3)
    assert np.abs(got.astype(np.int16) - ref.astype(np.int16)).max() <= 1


@needs_cuda
def test_resize_long_side_gpu_is_a_noop_below_target():
    img = np.zeros((100, 200, 3), np.uint8)
    out, scale = gpu_ops.resize_long_side_gpu(img, 1000)
    assert out is img and scale == 1.0


# ----------------------------------------------------------- illumination
@needs_cuda
def test_flatten_illumination_gpu_matches_reference_on_real_scans(scans):
    _need_free_vram()
    for name, img in scans:
        ref = illum.flatten_illumination(img)
        _assert_close(gpu_ops.flatten_illumination_gpu(img), ref, f"flatten {name}")


@needs_cuda
def test_scan_finish_gpu_matches_reference_on_real_scans(scans):
    _need_free_vram()
    for name, img in scans:
        ref = illum.scan_finish(img)
        _assert_close(gpu_ops.scan_finish_gpu(img), ref, f"scan_finish {name}")


# ------------------------------------------------------------ tensor path
@needs_cuda
def test_tensors_stay_on_the_device_and_match_the_numpy_path(scans):
    _need_free_vram()
    name, img = scans[0]
    quad = _page_quad(img.shape)
    t = torch.from_numpy(img).cuda()
    out = gpu_ops.warp_quad_gpu(t, quad, 800, 600)
    assert isinstance(out, torch.Tensor) and out.is_cuda and out.dtype == torch.uint8
    np.testing.assert_array_equal(out.cpu().numpy(), gpu_ops.warp_quad_gpu(img, quad, 800, 600))

    small = gpu_ops.resize_long_side_gpu(t, 1200)[0]
    assert isinstance(small, torch.Tensor) and small.is_cuda
