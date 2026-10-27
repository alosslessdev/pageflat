"""End-to-end equivalence of the GPU pipeline path on small synthetic pages.

The real-data version of this check lives in the report; these two pages keep it
runnable anywhere with a GPU (skipped without one).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pageflat.models import DocTrPipeline  # noqa: E402
from pageflat.pipeline import Settings, process_batch_gpu, process_image  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _synthetic_page(path, seed: int = 0) -> None:
    """A photographed single page: bright sheet on a dark desk, text-like lines."""
    rng = np.random.default_rng(seed)
    w, h = 1200, 900
    img = np.full((h, w, 3), 38, np.uint8)
    # portrait sheet (width/height < 0.85 so the spine splitter leaves it alone)
    quad = np.array([[300, 55], [935, 45], [950, 880], [288, 890]], np.float32)
    cv2.fillPoly(img, [quad.astype(np.int32)], (238, 240, 242))
    for y in range(130, 830, 26):
        x0 = 330 + int(rng.integers(0, 15))
        x1 = 880 - int(rng.integers(0, 120))
        cv2.line(img, (x0, y), (x1, y), (32, 32, 38), 3)
    yy, xx = np.mgrid[0:h, 0:w]
    gain = (0.82 + 0.25 * xx / w)[..., None]          # side-lighting
    img = np.clip(img * gain + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 94])


@pytest.fixture(scope="module")
def detector():
    free, _ = torch.cuda.mem_get_info()
    if free < 700 * 2 ** 20:
        pytest.skip(f"only {free / 2 ** 20:.0f} MB of GPU memory free")
    return DocTrPipeline(device="cuda")


@pytest.fixture(scope="module")
def pages(tmp_path_factory):
    root = tmp_path_factory.mktemp("pages")
    out = []
    for i in range(2):
        p = root / f"page{i}.jpg"
        _synthetic_page(p, seed=i)
        out.append(p)
    return out


def _compare(ref_dir, got_dir) -> None:
    refs = sorted(ref_dir.glob("*.png"))
    assert refs, "the current path wrote no pages"
    for f in refs:
        g = got_dir / f.name
        assert g.exists(), f"{f.name} missing from the GPU run"
        ref, got = cv2.imread(str(f)), cv2.imread(str(g))
        assert ref.shape == got.shape, f"{f.name}: {ref.shape} vs {got.shape}"
        d = np.abs(ref.astype(np.int16) - got.astype(np.int16))
        assert d.max() <= 3, f"{f.name}: max abs diff {d.max()}"
        assert d.mean() <= 1.0, f"{f.name}: mean abs diff {d.mean():.3f}"


def test_gpu_path_matches_the_current_path(detector, pages, tmp_path):
    ref_dir, gpu_dir = tmp_path / "current", tmp_path / "gpu"
    ref = [process_image(p, ref_dir, detector, Settings()) for p in pages]
    got = [process_image(p, gpu_dir, detector, Settings(gpu_path=True)) for p in pages]
    assert [r.status for r in ref] == ["ok", "ok"], [r.reason for r in ref]
    assert [r.status for r in got] == ["ok", "ok"], [r.reason for r in got]
    assert [r.pages for r in got] == [r.pages for r in ref]
    _compare(ref_dir, gpu_dir)


def test_gpu_batch_matches_batch_one(detector, pages, tmp_path):
    one_dir, batch_dir = tmp_path / "b1", tmp_path / "b2"
    one = process_batch_gpu(pages, one_dir, detector, Settings(gpu_path=True, gpu_batch=1))
    batched = process_batch_gpu(pages, batch_dir, detector,
                                Settings(gpu_path=True, gpu_batch=2))
    assert [r.status for r in one] == ["ok", "ok"], [r.reason for r in one]
    assert [r.status for r in batched] == ["ok", "ok"], [r.reason for r in batched]
    _compare(one_dir, batch_dir)
