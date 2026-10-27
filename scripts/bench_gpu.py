#!/usr/bin/env python
"""Per-stage CPU vs GPU cost of the page-flattening pipeline.

Runs the same 64 MP camera JPEGs through both paths of every pixel stage
(decode, page-mask inference, perspective warp, dewarp flow + grid_sample,
illumination, PNG write) and reports ms per stage, the per-image wall clock of
each path, and the GPU utilisation nvidia-smi saw during each phase.

    python scripts/bench_gpu.py --n 10
    OMP_NUM_THREADS=2 python scripts/bench_gpu.py --n 10 --json /tmp/bench.json

Stage timings are means over N images; the mask/warp/illumination stages run the
real pipeline code on one side and pageflat.gpu_ops on the other, so the table
shows what a cutover would actually buy.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pageflat import geom, gpu_ops, illum, io  # noqa: E402
from pageflat.models import DocTrPipeline  # noqa: E402

CAMERA_DIR = Path("/run/media/asdf/30620d0d-1e41-4d55-af08-b4de2fb9c517/folder/Camera")
MASK_WORK_SIDE = 1600          # pipeline Settings.mask_work_side
MAX_INTERMEDIATE = 3508        # pipeline Settings.max_intermediate
STAGES = ["jpeg_decode", "mask_infer", "quad_fit", "perspective_warp",
          "dewarp_flow", "dewarp_grid_sample", "illumination", "png_write"]

HAS_CUDA = torch.cuda.is_available()


# ------------------------------------------------------------------ measuring
def _sync() -> None:
    if HAS_CUDA:
        torch.cuda.synchronize()


@contextmanager
def _timed(store: dict, name: str):
    _sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        store[name] = store.get(name, 0.0) + (time.perf_counter() - t0) * 1e3


@contextmanager
def _cpu_reference():
    """Force illum.scan_finish onto its numpy body (it dispatches to the GPU)."""
    with mock.patch.object(torch.cuda, "is_available", lambda: False):
        yield


class GpuSampler(threading.Thread):
    """Background nvidia-smi poll, tagged with the phase that was running."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.phase = "idle"
        self.samples: list[tuple[str, float, float]] = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                util, mem = (float(v) for v in out.splitlines()[0].split(","))
                self.samples.append((self.phase, util, mem))
            except Exception:  # noqa: BLE001 - no nvidia-smi, keep benchmarking
                pass
            time.sleep(self.interval)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2)

    def summary(self, phase: str) -> dict:
        vals = [(u, m) for p, u, m in self.samples if p == phase]
        if not vals:
            return {"n": 0}
        utils = [u for u, _ in vals]
        return {"n": len(vals), "util_mean": statistics.fmean(utils),
                "util_max": max(utils), "util_min": min(utils),
                "mem_peak_mb": max(m for _, m in vals)}


# ------------------------------------------------------------------- the work
def _quad_points(mask: np.ndarray, scale: float, shape) -> np.ndarray:
    """Page quad in full-image coordinates (the CPU geometry shared by both paths)."""
    page = geom.mask_to_quad(mask)
    if page is None:
        raise RuntimeError("no page found")
    pts = page.points / scale
    h, w = shape[:2]
    pts[:, 0] = pts[:, 0].clip(0, w - 1)
    pts[:, 1] = pts[:, 1].clip(0, h - 1)
    return pts.astype(np.float32)


def run_cpu(path: Path, det: DocTrPipeline, png: Path) -> dict:
    """Baseline: every stage on the CPU with the existing pageflat code."""
    t: dict = {}
    wall0 = time.perf_counter()
    with _timed(t, "jpeg_decode"):
        img = io.load_image(path)
    with _timed(t, "mask_infer"):
        small, scale = io.resize_long_side(img, MASK_WORK_SIDE)
        mask = det.page_mask(small)
    with _timed(t, "quad_fit"):
        pts = _quad_points(mask, scale, img.shape)
        out_w, out_h = geom.quad_target_size(pts, MAX_INTERMEDIATE)
    with _timed(t, "perspective_warp"):
        rect = geom.warp_quad(img, pts, out_size=(out_w, out_h))
    with _timed(t, "dewarp_flow"):
        flow = det.geo_flow(rect)
    with _timed(t, "dewarp_grid_sample"):
        h, w = rect.shape[:2]
        map_x = ((flow[:, :, 0] + 1.0) * (w - 1) * 0.5).astype(np.float32)
        map_y = ((flow[:, :, 1] + 1.0) * (h - 1) * 0.5).astype(np.float32)
        page = cv2.remap(rect, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        del map_x, map_y
    with _timed(t, "illumination"):
        with _cpu_reference():
            page = illum.scan_finish(page)
    with _timed(t, "png_write"):
        cv2.imwrite(str(png), page, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    t["wall_ms"] = (time.perf_counter() - wall0) * 1e3
    t["out_hw"] = page.shape[:2]
    del img, small, mask, rect, flow, page
    return t


def _gpu_mask(det: DocTrPipeline, small: torch.Tensor) -> np.ndarray:
    """U2NETP page mask at `small`'s size, staying on the device."""
    h, w = small.shape[:2]
    with torch.no_grad():
        x = F.interpolate(small.permute(2, 0, 1).unsqueeze(0).float().div_(255.0),
                          size=(512, 512), mode="area")
        m = det.seg(x[:, [2, 1, 0]])[0]
        m = F.interpolate(m[:, :1], size=(h, w), mode="bilinear", align_corners=False)
    return m[0, 0].float().cpu().numpy()


def _gpu_flow(det: DocTrPipeline, rect: torch.Tensor) -> torch.Tensor:
    """DocTr backward flow as a grid_sample grid (1, h, w, 2) on the device."""
    h, w = rect.shape[:2]
    with torch.no_grad():
        s = F.interpolate(rect.permute(2, 0, 1).unsqueeze(0).float().div_(255.0),
                          size=(288, 288), mode="area")
        rgb = s[:, [2, 1, 0]]
        msk = (det.seg(rgb)[0] > 0.5).float()
        bm = det.geotr(msk * rgb)[0]
        bm = (2 * (bm / 286.8) - 1) * 0.99
        g = F.interpolate(bm.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
        out = F.avg_pool2d(g, 3, stride=1, padding=1)
    return out.permute(0, 2, 3, 1)


def run_gpu(path: Path, det: DocTrPipeline, png: Path) -> dict:
    """Every stage on the device; only the PNG encoder and the quad fit stay on CPU."""
    t: dict = {}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()      # nvjpeg wants ~0.7 GB of workspace for 64 MP
    wall0 = time.perf_counter()
    with _timed(t, "jpeg_decode"):
        buf = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8)
        chw = torchvision_decode_jpeg(buf)
        img = chw.permute(1, 2, 0).flip(-1)          # RGB CHW -> BGR HWC, uint8
    with _timed(t, "mask_infer"):
        small, scale = gpu_ops.resize_long_side_gpu(img, MASK_WORK_SIDE)
        mask = _gpu_mask(det, small)
    with _timed(t, "quad_fit"):
        pts = _quad_points(mask, scale, img.shape)
        out_w, out_h = geom.quad_target_size(pts, MAX_INTERMEDIATE)
    with _timed(t, "perspective_warp"):
        rect = gpu_ops.warp_quad_gpu(img, pts, out_w, out_h)
    with _timed(t, "dewarp_flow"):
        grid = _gpu_flow(det, rect)
    with _timed(t, "dewarp_grid_sample"):
        src = rect.permute(2, 0, 1).unsqueeze(0).to(
            dtype=torch.float32, memory_format=torch.contiguous_format).div_(255.0)
        h, w = rect.shape[:2]
        page = torch.empty((1, 3, h, w), dtype=torch.float32, device=src.device)
        for y0 in range(0, h, 512):                  # banded, as models.rectify does
            y1 = min(h, y0 + 512)
            page[:, :, y0:y1] = F.grid_sample(
                src, grid[:, y0:y1].contiguous(), align_corners=True, mode="bilinear")
        del src
        page = (page[0].permute(1, 2, 0) * 255.0).clamp_(0, 255).to(torch.uint8)
    with _timed(t, "illumination"):
        page = gpu_ops.scan_finish_gpu(page)
    with _timed(t, "png_write"):
        cv2.imwrite(str(png), page.cpu().numpy(), [cv2.IMWRITE_PNG_COMPRESSION, 1])
    t["wall_ms"] = (time.perf_counter() - wall0) * 1e3
    t["out_hw"] = page.shape[:2]
    t["vram_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del img, small, mask, rect, grid, page
    torch.cuda.empty_cache()
    return t


def torchvision_decode_jpeg(buf: torch.Tensor) -> torch.Tensor:
    """nvjpeg decode on the GPU (torchvision ships it with CUDA builds)."""
    from torchvision.io import decode_jpeg

    return decode_jpeg(buf, device="cuda")


# -------------------------------------------------------------------- report
def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return statistics.fmean(vals) if vals else float("nan")


def _fmt(ms: float) -> str:
    return f"{ms:8.1f}" if ms == ms else "     n/a"


def report(cpu_rows: list[dict], gpu_rows: list[dict], sampler: GpuSampler,
           n: int, src: Path) -> str:
    lines = []
    lines.append(f"pageflat CPU vs GPU benchmark - {n} images from {src}")
    lines.append(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
                 f"device {torch.cuda.get_device_name(0) if HAS_CUDA else 'none'}")
    lines.append(f"CPU threads {torch.get_num_threads()}, mask work side {MASK_WORK_SIDE}, "
                 f"max intermediate {MAX_INTERMEDIATE}")
    out_hw = cpu_rows[0]["out_hw"]
    lines.append(f"rectified page {out_hw[1]}x{out_hw[0]}; times are ms per image, "
                 f"mean of {n}")
    lines.append("")
    lines.append(f"{'stage':22} {'CPU ms':>8} {'share':>6} {'GPU ms':>8} "
                 f"{'speedup':>8}  verdict")
    lines.append("-" * 78)
    cpu_total = _mean(cpu_rows, "wall_ms")
    verdicts = []
    for name in STAGES:
        c, g = _mean(cpu_rows, name), _mean(gpu_rows, name)
        share = 100.0 * c / cpu_total if cpu_total else 0.0
        if g != g:
            sp, verdict = float("nan"), "no GPU path"
        else:
            sp = c / g if g else float("nan")
            verdict = "MOVE TO GPU" if sp >= 1.3 else "keep on CPU"
            verdicts.append((name, sp, verdict))
        lines.append(f"{name:22} {_fmt(c)} {share:5.0f}% {_fmt(g)} "
                     f"{sp:8.2f}  {verdict}")
    lines.append("-" * 78)
    gpu_total = _mean(gpu_rows, "wall_ms")
    lines.append(f"{'pipeline total':22} {_fmt(cpu_total)} {100:5.0f}% "
                 f"{_fmt(gpu_total)} {cpu_total / gpu_total:8.2f}  "
                 f"({100 * (gpu_total / cpu_total - 1):+.0f}% wall clock)")
    lines.append("")
    lines.append("which stages are worth moving:")
    for name, sp, verdict in verdicts:
        if verdict.startswith("MOVE"):
            lines.append(f"  + {name:22} {sp:5.2f}x  {_mean(cpu_rows, name):7.0f} -> "
                         f"{_mean(gpu_rows, name):6.0f} ms/image")
    for name, sp, verdict in verdicts:
        if not verdict.startswith("MOVE"):
            lines.append(f"  - {name:22} {sp:5.2f}x  stays on the CPU")
    lines.append("")
    if HAS_CUDA:
        lines.append(f"GPU memory benchmark peak: {_mean(gpu_rows, 'vram_peak_mb'):.0f} MB "
                     f"allocated (torch), device memory-in-use peak "
                     f"{max((m for _, _, m in sampler.samples), default=0):.0f} MB")
    for phase in ("cpu", "gpu"):
        s = sampler.summary(phase)
        if s["n"]:
            lines.append(f"nvidia-smi during {phase.upper()} phase: "
                         f"{s['n']} samples, util mean {s['util_mean']:.0f}% "
                         f"(max {s['util_max']:.0f}%, min {s['util_min']:.0f}%)")
        else:
            lines.append(f"nvidia-smi during {phase.upper()} phase: no samples")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10, help="number of source photos")
    ap.add_argument("--src", type=Path, default=CAMERA_DIR, help="photo directory")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 2)),
                    help="torch CPU threads (keep low on a shared box)")
    ap.add_argument("--json", type=Path, help="also write the raw timings here")
    ap.add_argument("--tmp", type=Path, default=Path(os.environ.get("PAGEFLAT_BENCH_TMP", "/var/tmp")),
                    help="where to write benchmark PNGs (needs a few hundred MB)")
    ap.add_argument("--keep", action="store_true", help="keep the written PNGs")
    args = ap.parse_args(argv)

    if not HAS_CUDA:
        print("no CUDA device: nothing to compare", file=sys.stderr)
        return 2
    if not args.src.is_dir():
        print(f"no photos at {args.src}", file=sys.stderr)
        return 2

    torch.set_num_threads(args.threads)
    # no cudnn autotuning: benchmark=True spends minutes warm-selecting kernels
    # for every conv shape of U2NETP/GeoTr and caches the workspaces.
    torch.backends.cudnn.benchmark = False
    photos = sorted(p for p in args.src.glob("IMG_*.jpg"))[:args.n]
    if len(photos) < args.n:
        print(f"only {len(photos)} photos found in {args.src}", file=sys.stderr)
        return 2
    free_mb = torch.cuda.mem_get_info()[0] / 2 ** 20
    from PIL import Image

    mp = []
    for p in photos:
        with Image.open(p) as im:
            mp.append(im.size[0] * im.size[1] / 1e6)
    print(f"{len(photos)} photos of {statistics.fmean(mp):.1f} MP ({min(mp):.1f}-{max(mp):.1f} MP), "
          f"{free_mb:.0f} MB of GPU memory free", flush=True)
    if free_mb < 1600:
        print("warning: the GPU is busy - nvjpeg alone wants ~0.7 GB of workspace "
              "for a 64 MP frame; timings and allocations may suffer", flush=True)

    args.tmp.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="pageflat_bench_", dir=args.tmp))
    sampler = GpuSampler()
    sampler.start()
    cpu_rows, gpu_rows = [], []
    try:
        # GPU first: if the box has no room for nvjpeg, fail before spending
        # minutes on the (much slower) CPU pass.
        det_gpu = DocTrPipeline(device="cuda")
        torch.cuda.reset_peak_memory_stats()
        sampler.phase = "gpu"
        run_gpu(photos[0], det_gpu, out_dir / "warm_gpu.png")   # warm the kernels
        print("  gpu warmup done", flush=True)
        torch.cuda.reset_peak_memory_stats()
        for i, path in enumerate(photos, 1):
            row = run_gpu(path, det_gpu, out_dir / f"gpu_{i}.png")
            gpu_rows.append(row)
            print(f"  gpu [{i}/{len(photos)}] {path.name}: {row['wall_ms']:.0f} ms",
                  flush=True)
        del det_gpu
        torch.cuda.empty_cache()

        det_cpu = DocTrPipeline(device="cpu")
        sampler.phase = "cpu"
        run_cpu(photos[0], det_cpu, out_dir / "warm_cpu.png")   # warm the kernels
        print("  cpu warmup done", flush=True)
        for i, path in enumerate(photos, 1):
            row = run_cpu(path, det_cpu, out_dir / f"cpu_{i}.png")
            cpu_rows.append(row)
            print(f"  cpu [{i}/{len(photos)}] {path.name}: {row['wall_ms']:.0f} ms",
                  flush=True)
        del det_cpu
    except RuntimeError as exc:  # noqa: BLE001
        sampler.stop()
        if "memory" not in str(exc).lower() and "CUDA" not in str(exc):
            raise
        print(f"GPU phase aborted: {exc}\n"
              f"free VRAM now {torch.cuda.mem_get_info()[0] / 2 ** 20:.0f} MB - rerun "
              f"when the GPU is idle (64 MP nvjpeg decode needs ~0.7 GB of workspace)",
              file=sys.stderr)
        for p in out_dir.glob("*.png"):
            p.unlink()
        out_dir.rmdir()
        return 3
    finally:
        sampler.stop()

    text = report(cpu_rows, gpu_rows, sampler, len(photos), args.src)
    print("\n" + text)
    if args.json:
        args.json.write_text(json.dumps(
            {"cpu": cpu_rows, "gpu": gpu_rows, "samples": sampler.samples,
             "n": len(photos), "src": str(args.src)}, indent=2))
    if not args.keep:
        for p in out_dir.glob("*.png"):
            p.unlink()
        out_dir.rmdir()
    else:
        print(f"\nPNGs kept in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
