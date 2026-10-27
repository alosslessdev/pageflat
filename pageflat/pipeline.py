"""End-to-end page -> flat scan pipeline."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import geom, illum, orientation, quality, spine
from .io import load_image, resize_long_side


@dataclass
class Settings:
    scan_long_side: int = 3508          # A4 @ 300dpi
    mask_work_side: int = 1600
    use_sam: bool = False
    dewarp: str = "doctr"               # doctr | uvdoc | best | none
    illum_mode: str = "classical"       # classical | doctr | none
    split_spreads: bool = True
    auto_orient: bool = True
    deskew: bool = True
    save_debug: bool = False
    min_quad_area: float = 0.05
    min_rectangularity: float = 0.65
    min_ink: float = 0.0025
    min_lines: int = 3
    max_intermediate: int = 4200


@dataclass
class PageResult:
    source: str
    status: str                         # ok | rejected | error
    reason: str = ""
    quad: list = field(default_factory=list)
    area_frac: float = 0.0
    rectangularity: float = 0.0
    rotation: int = 0
    dewarp: str = ""
    content: dict = field(default_factory=dict)
    pages: int = 0
    outputs: list = field(default_factory=list)
    elapsed_s: float = 0.0


def detect_page(detector, img: np.ndarray, settings: Settings) -> Optional[geom.PageQuad]:
    small, scale = resize_long_side(img, settings.mask_work_side)
    mask = detector.page_mask(small)
    quad = geom.mask_to_quad(mask)
    if quad is None:
        return None
    sam = getattr(detector, "sam", None)
    if settings.use_sam and sam is not None:
        pts = quad.points
        cx, cy = pts.mean(axis=0)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
        m2 = sam.mask(small, [[cx, cy]], box=(x, y, x + w, y + h), prior=mask)
        if m2 is not None:
            q2 = geom.mask_to_quad(m2)
            if q2 is not None and q2.area_frac > 0.03:
                quad = q2
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    quad.points = geom.refine_quad(gray, quad.points, search_frac=0.02)
    quad.points = (quad.points / scale).astype(np.float32)
    h, w = img.shape[:2]
    quad.points[:, 0] = quad.points[:, 0].clip(0, w - 1)
    quad.points[:, 1] = quad.points[:, 1].clip(0, h - 1)
    return quad


def _finish_size(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def ocr_quality(img_bgr: np.ndarray, max_side: int = 1600) -> float:
    """Mean tesseract word confidence; higher means cleaner, straighter text."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    cv2.imwrite(tmp, small)
    try:
        out = subprocess.run(["tesseract", tmp, "stdout", "--psm", "3", "tsv"],
                             capture_output=True, text=True, timeout=120).stdout
    except Exception:
        return 0.0
    finally:
        Path(tmp).unlink(missing_ok=True)
    confs = []
    for line in out.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 12:
            try:
                c = float(parts[10])
            except ValueError:
                continue
            if len(parts[11].strip()) > 2:
                confs.append(c)
    if not confs:
        return 0.0
    return float(np.mean(confs))


def process_image(path: str | Path, out_dir: str | Path, detector,
                  settings: Settings) -> PageResult:
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = PageResult(source=str(path), status="error")

    try:
        img = load_image(path)
    except Exception as exc:  # noqa: BLE001
        res.reason = f"load failed: {exc}"
        return res

    try:
        quad = detect_page(detector, img, settings)
        if quad is None:
            res.status, res.reason = "rejected", "no page mask"
            return res
        res.quad = quad.points.round(2).tolist()
        res.area_frac = round(quad.area_frac, 4)
        res.rectangularity = round(quad.rectangularity, 4)
        if quad.area_frac < settings.min_quad_area:
            res.status, res.reason = "rejected", "page too small in frame"
            return res
        if quad.rectangularity < settings.min_rectangularity:
            res.status, res.reason = "rejected", "boundary not page-like"
            return res

        rect = geom.warp_quad(img, quad.points, max_side=settings.max_intermediate)

        if settings.auto_orient:
            rect, res.rotation = orientation.rotate_upright(rect)

        if settings.dewarp == "doctr":
            rect = detector.rectify(rect)
            res.dewarp = "doctr"
        elif settings.dewarp == "uvdoc":
            from .models import UVDocDewarper
            rect = UVDocDewarper().rectify(rect)
            res.dewarp = "uvdoc"
        elif settings.dewarp == "best":
            from .models import UVDocDewarper
            cand_a = detector.rectify(rect)
            cand_b = UVDocDewarper().rectify(rect)
            qa, qb = ocr_quality(cand_a), ocr_quality(cand_b)
            res.dewarp = "doctr" if qa >= qb else "uvdoc"
            rect = cand_a if qa >= qb else cand_b

        if settings.deskew:
            rect, _ = orientation.deskew(rect, max_angle=10.0)

        if settings.illum_mode == "classical":
            rect = illum.scan_finish(rect)
        elif settings.illum_mode == "doctr":
            rect = detector.illuminate(rect)
            rect = illum.auto_contrast(rect)

        ok_content, metrics = quality.looks_like_page(rect, settings.min_ink, settings.min_lines)
        res.content = metrics
        if not ok_content:
            res.status, res.reason = "rejected", "no page content (ink/lines)"
            return res

        pages = spine.split_spread(rect) if settings.split_spreads else [rect]
        for i, page in enumerate(pages):
            page = _finish_size(page, settings.scan_long_side)
            suffix = "" if len(pages) == 1 else f"_p{i + 1}"
            out_path = out_dir / f"{path.stem}{suffix}.png"
            if out_path.exists():
                tag = hashlib.sha1(str(path).encode()).hexdigest()[:6]
                out_path = out_dir / f"{path.stem}_{tag}{suffix}.png"
            cv2.imwrite(str(out_path), page, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            res.outputs.append(str(out_path))
            if settings.save_debug:
                dbg = img.copy()
                cv2.polylines(dbg, [quad.points.astype(np.int32)], True, (0, 0, 255), 12)
                cv2.imwrite(str(out_dir / f"{path.stem}{suffix}_debug.jpg"), dbg,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        res.pages = len(pages)
        res.status = "ok"
    except Exception as exc:  # noqa: BLE001
        res.status, res.reason = "error", repr(exc)
    finally:
        res.elapsed_s = round(time.time() - t0, 2)
        if detector.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    return res
