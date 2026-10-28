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
import torch

from . import geom, gpu_ops, illum, orientation, quality, router, spine
from . import lines, perspective
from .io import load_image, resize_long_side


@dataclass
class Settings:
    scan_long_side: int = 3508          # A4 @ 300dpi
    mask_work_side: int = 1600
    use_sam: bool = False
    route_url: str = ""            # llama-server with a Clef model, optional
    trim_background: bool = False  # opt-in: drop saturated background (bed sheets, table cloth)
    dewarp: str = "doctr"               # doctr | uvdoc | best | none
    illum_mode: str = "classical"       # classical | doctr | none
    split_spreads: bool = True
    auto_orient: bool = True
    deskew: bool = True
    split_sheets: bool = True           # one homography per physical sheet
    max_sheets: int = 3
    quad_margin: float = 0.012          # expand each quad so paper edges survive
    paper_reach: float = 0.0            # >0: reach past a crop that cut into the text
    content_route: bool = False         # classify the page and pick a strategy
    pick_metric: str = "ocr"            # "bow": choose the straightest candidate without OCR
    route_url: str = ""                 # clef (llama-server) endpoint for the classifier
    line_straighten: bool = True        # word-line bow removal (kept only if OCR agrees)
    line_min_gain: float = 0.5          # OCR points the grid must win to be kept
    save_debug: bool = False
    min_quad_area: float = 0.05
    min_rectangularity: float = 0.65
    min_ink: float = 0.0025
    min_lines: int = 3
    max_texture: float = 30.0
    min_paper: float = 0.45
    max_intermediate: int = 3508
    gpu_path: bool = False              # keep the pixel stages on the device
    gpu_batch: int = 1                  # images per GPU batch on that path


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
    lines: dict = field(default_factory=dict)
    kind: str = ""                         # page type from the content classifier
    pick: dict = field(default_factory=dict)  # candidate scores in "best" mode
    route: str = ""
    pages: int = 0
    outputs: list = field(default_factory=list)
    elapsed_s: float = 0.0


def _quad_from_mask(detector, small: np.ndarray, mask: np.ndarray, scale: float,
                    settings: Settings, shape: tuple[int, int],
                    use_sam: bool | None = None) -> Optional[geom.PageQuad]:
    """Page quad from a mask at the mask-work scale, in full-image coordinates."""
    quad = geom.mask_to_quad(mask)
    if quad is None:
        return None
    sam = getattr(detector, "sam", None)
    want_sam = settings.use_sam if use_sam is None else use_sam
    if want_sam and sam is not None:
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
    h, w = shape[:2]
    quad.points[:, 0] = quad.points[:, 0].clip(0, w - 1)
    quad.points[:, 1] = quad.points[:, 1].clip(0, h - 1)
    return quad


def detect_page(detector, img: np.ndarray, settings: Settings,
                use_sam: bool | None = None) -> Optional[geom.PageQuad]:
    small, scale = resize_long_side(img, settings.mask_work_side)
    mask = detector.page_mask(small)
    return _quad_from_mask(detector, small, mask, scale, settings, img.shape, use_sam)


def page_kind(rect: np.ndarray, settings: Settings) -> str:
    """What kind of page this is, from the content classifier.

    Returns ``"unknown"`` when routing is off or the classifier cannot answer,
    and callers then keep the conservative behaviour.
    """
    if not settings.content_route:
        return "unknown"
    try:
        from . import content
        return content.classify(rect, url=settings.route_url or None).kind
    except Exception:  # noqa: BLE001 - routing is an optimisation, never a hard dependency
        return "unknown"


def rectified_sheets(img: np.ndarray, detector, settings: Settings
                     ) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Perspective correction first: one homography per physical sheet.

    Returns ``(rectified, quad, area_frac, rectangularity)`` per sheet. Two
    sheets touching on the table used to share one hull quad, and the warp
    then clipped whichever sheet sat further inside the hull.
    """
    small, scale = resize_long_side(img, settings.mask_work_side)
    mask = detector.page_mask(small)
    if not mask.any():
        return []
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if settings.split_sheets:
        kind = page_kind(small, settings) if settings.content_route else "unknown"
        parts = perspective.sheet_masks(small, mask, max_sheets=settings.max_sheets,
                                        kind=kind)
    else:
        parts = [(mask >= 0.5).astype(np.uint8)]
    out = []
    for part in parts:
        quad = geom.mask_to_quad(part)
        if quad is None or not part.any():
            continue
        pts = geom.refine_quad(gray, quad.points, search_frac=0.02)
        pts = (pts / scale).astype(np.float32)
        # only reach past the mask when ink already touches it: that is a
        # crop inside the page. A page with its margin intact needs nothing,
        # and reaching out there drags in whatever lies next to it.
        if settings.paper_reach > 0 and perspective.edge_has_ink(img, pts):
            pts = perspective.extend_to_paper(img, pts, max_frac=settings.paper_reach)
        pts = perspective.expand_quad(pts, settings.quad_margin, img.shape)
        rect = perspective.warp_sheet(img, pts, max_side=settings.max_intermediate)
        out.append((rect, pts, quad.area_frac, quad.rectangularity))
    return out


def trim_background(img_bgr: np.ndarray, min_removed: float = 0.03) -> np.ndarray:
    """Trim saturated background that survived segmentation.

    A page photographed on a coloured bed sheet often keeps a strip of fabric
    around the sheet. The paper is neutral and bright, the fabric is neither, so
    keep the largest paper-like component that contains the centre.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    paper = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 110)).astype(np.uint8)
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, 8)
    if n <= 1:
        return img_bgr
    centre = labels[h // 2, w // 2] or (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    x, y, bw, bh, area = stats[centre]
    if area < 0.25 * h * w:
        return img_bgr
    crop = img_bgr[y:y + bh, x:x + bw]
    if crop.size == 0:
        return img_bgr
    removed = 1.0 - (bw * bh) / float(h * w)
    if removed < min_removed:
        return img_bgr
    if crop.shape[0] < 0.4 * h or crop.shape[1] < 0.3 * w:
        return img_bgr
    return crop


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
        cmd = ["tesseract", tmp, "stdout", "--psm", "3"]
        if orientation.ocr_lang():
            cmd += ["-l", orientation.ocr_lang()]
        out = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=120).stdout
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


_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
           270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def process_image_gpu(path: str | Path, out_dir: str | Path, detector,
                      settings: Settings) -> PageResult:
    """`process_image` with the pixel stages kept on the GPU."""
    return process_batch_gpu([path], out_dir, detector, settings)[0]


def process_batch_gpu(paths, out_dir: str | Path, detector,
                      settings: Settings) -> list[PageResult]:
    """`process_image` with the pixel stages batched onto the GPU.

    The U2NETP mask forward, the GeoTr flow forward, the perspective warp, the
    flow resample and the illumination pass run on the device; the parts that
    decide *what* to render (quad fit, page orientation, deskew, spine split,
    quality gates) and the PNG encoder stay on the CPU, unchanged.

    `paths` holds one file for a plain run, or up to ``settings.gpu_batch``
    files: the model forwards then go through as one batch and the per-image
    kernels are queued without syncing, so the device stays busy while the CPU
    finishes the previous image.  ``elapsed_s`` is the amortised per-image wall
    clock when the batch is larger than one.
    """
    paths = [Path(p) for p in paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_batch = time.time()
    dev = torch.device(detector.device)
    results = [PageResult(source=str(p), status="error") for p in paths]
    states: list[dict] = []
    live: list[dict] = []
    failure = ""

    for res, path in zip(results, paths):
        try:
            img = load_image(path, min_long_side=4200)
        except Exception as exc:  # noqa: BLE001
            res.reason = f"load failed: {exc}"
            continue
        try:
            if settings.route_url:
                route = router.decide(img, settings.route_url)
                res.route = route.kind
                if route.kind == "not_page" and route.confidence >= 0.6:
                    res.status, res.reason = "rejected", "router: not a page"
                    continue
            small, scale = resize_long_side(img, settings.mask_work_side)
        except Exception as exc:  # noqa: BLE001
            res.reason = repr(exc)
            continue
        states.append({"res": res, "path": path, "img": img, "small": small,
                       "scale": scale, "quad": None, "rect_t": None, "rect": None})

    try:
        # --- segmentation: one forward for the batch, quad fit on the CPU
        masks = gpu_ops.page_masks(detector, [s["small"] for s in states]) if states else []
        for s, mask in zip(states, masks):
            res = s["res"]
            try:
                quad = _quad_from_mask(detector, s["small"], mask, s["scale"], settings,
                                       s["img"].shape, settings.use_sam)
            except Exception as exc:  # noqa: BLE001
                res.reason = repr(exc)
                continue
            s["small"] = None
            if quad is None:
                res.status, res.reason = "rejected", "no page mask"
                continue
            res.quad = quad.points.round(2).tolist()
            res.area_frac = round(quad.area_frac, 4)
            res.rectangularity = round(quad.rectangularity, 4)
            if quad.area_frac < settings.min_quad_area:
                res.status, res.reason = "rejected", "page too small in frame"
                continue
            if quad.rectangularity < settings.min_rectangularity:
                res.status, res.reason = "rejected", "boundary not page-like"
                continue
            s["quad"] = quad
            live.append(s)

        # --- rectification: upload once, warp on the device, queue the batch
        for s in live:
            res = s["res"]
            try:
                out_w, out_h = geom.quad_target_size(s["quad"].points,
                                                     settings.max_intermediate)
                src = torch.from_numpy(np.ascontiguousarray(s["img"])).to(dev)
                s["rect_t"] = gpu_ops.warp_quad_gpu(src, s["quad"].points, out_w, out_h)
                del src
                if not settings.save_debug:
                    s["img"] = None            # free the host copy early
            except Exception as exc:  # noqa: BLE001
                res.status, res.reason = "error", repr(exc)

        # --- orientation needs the numpy page (one download per image)
        for s in live:
            if s["rect_t"] is None:
                continue
            rect = s["rect_t"].cpu().numpy()
            s["rect_t"] = None                     # free the device copy at once
            angle = 0
            if settings.auto_orient:
                angle = orientation.detect_rotation_fast(rect, device=str(detector.device))
                if angle in _ROTATE:
                    rect = cv2.rotate(rect, _ROTATE[angle])
            s["res"].rotation = angle
            s["rect"] = rect
        ready = [s for s in live if s["rect"] is not None]

        # --- unwarp: one GeoTr forward for the batch, then one page at a time so
        # only a single full-resolution grid and source tensor are ever resident
        doctr = [s for s in ready if settings.dewarp == "doctr"]
        if doctr:
            flows = gpu_ops.page_flow_small(detector, [s["rect"] for s in doctr])
            for flow, s in zip(flows, doctr):
                try:
                    h, w = s["rect"].shape[:2]
                    grid = gpu_ops.flow_grid(flow, h, w)
                    # numpy in, numpy out: the tail below is all cv2
                    s["rect"] = gpu_ops.dewarp_page(s["rect"], grid)
                    s["res"].dewarp = "doctr"
                    del grid
                except Exception as exc:  # noqa: BLE001
                    s["rect"] = None
                    s["res"].status, s["res"].reason = "error", repr(exc)
            del flows
        for s in ready:
            if settings.dewarp == "doctr":
                continue
            if settings.dewarp == "uvdoc":
                try:
                    from .models import UVDocDewarper
                    s["rect"] = UVDocDewarper().rectify(s["rect"])
                    s["res"].dewarp = "uvdoc"
                except Exception as exc:  # noqa: BLE001
                    s["rect"] = None
                    s["res"].status, s["res"].reason = "error", repr(exc)
            elif settings.dewarp == "best":
                try:
                    from .models import UVDocDewarper
                    cand_a = detector.rectify(s["rect"])
                    cand_b = UVDocDewarper().rectify(s["rect"])
                    qa, qb = ocr_quality(cand_a), ocr_quality(cand_b)
                    s["res"].dewarp = "doctr" if qa >= qb else "uvdoc"
                    s["rect"] = cand_a if qa >= qb else cand_b
                except Exception as exc:  # noqa: BLE001
                    s["rect"] = None
                    s["res"].status, s["res"].reason = "error", repr(exc)

        # --- gates, illumination and PNG writing: exactly the CPU path, per page
        for s in ready:
            res, rect = s["res"], s["rect"]
            if rect is None:            # a failed stage already set status/reason
                continue
            try:
                if settings.deskew:
                    rect, _ = orientation.deskew(rect, max_angle=10.0)
                if settings.illum_mode == "doctr":
                    rect = detector.illuminate(rect)
                if settings.trim_background:
                    rect = trim_background(rect)
                pages = spine.split_spread(rect) if settings.split_spreads else [rect]
                kept = []
                for page in pages:
                    surface_ok, m_surface = quality.surface_checks(
                        page, max_texture=settings.max_texture,
                        min_paper=settings.min_paper)
                    if settings.illum_mode == "classical":
                        page = gpu_ops.scan_finish_gpu(page, device=dev)
                    elif settings.illum_mode == "doctr":
                        page = illum.auto_contrast(page)
                    content_ok, m_content = quality.content_checks(
                        page, settings.min_ink, settings.min_lines)
                    res.content = {**m_surface, **m_content}
                    if surface_ok and content_ok:
                        kept.append(page)
                if not kept:
                    res.status = "rejected"
                    res.reason = res.reason or "no page content (ink/lines/paper)"
                    continue
                for i, page in enumerate(kept):
                    page = _finish_size(page, settings.scan_long_side)
                    suffix = "" if len(kept) == 1 else f"_p{i + 1}"
                    out_path = out_dir / f"{s['path'].stem}{suffix}.png"
                    if out_path.exists():
                        tag = hashlib.sha1(str(s["path"]).encode()).hexdigest()[:6]
                        out_path = out_dir / f"{s['path'].stem}_{tag}{suffix}.png"
                    cv2.imwrite(str(out_path), page, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                    res.outputs.append(str(out_path))
                    if settings.save_debug and s["img"] is not None:
                        dbg = s["img"].copy()
                        cv2.polylines(dbg, [s["quad"].points.astype(np.int32)], True,
                                      (0, 0, 255), 12)
                        cv2.imwrite(str(out_dir / f"{s['path'].stem}{suffix}_debug.jpg"), dbg,
                                    [cv2.IMWRITE_JPEG_QUALITY, 80])
                res.pages = len(kept)
                res.status = "ok"
            except Exception as exc:  # noqa: BLE001
                res.status, res.reason = "error", repr(exc)
    except Exception as exc:  # noqa: BLE001
        failure = repr(exc)
    finally:
        per_image = (time.time() - t_batch) / max(1, len(results))
        for res in results:
            res.elapsed_s = round(per_image, 2)
            if res.status == "error" and not res.reason:
                res.reason = failure or "gpu path produced no result"
        states.clear()
        live.clear()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return results


def process_image(path: str | Path, out_dir: str | Path, detector,
                  settings: Settings) -> PageResult:
    if settings.gpu_path and detector.device.type == "cuda":
        try:
            return process_image_gpu(path, out_dir, detector, settings)
        except Exception as exc:  # noqa: BLE001 - never take the batch down
            res = PageResult(source=str(path), status="error", reason=repr(exc))
            return res
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = PageResult(source=str(path), status="error")

    try:
        img = load_image(path, min_long_side=4200)
    except Exception as exc:  # noqa: BLE001
        res.reason = f"load failed: {exc}"
        return res

    try:
        route = None
        if settings.route_url:
            route = router.decide(img, settings.route_url)
            res.route = route.kind
            if route.kind == "not_page" and route.confidence >= 0.6:
                res.status, res.reason = "rejected", "router: not a page"
                return res
        sheets = rectified_sheets(img, detector, settings)
        if not sheets:
            res.status, res.reason = "rejected", "no page mask"
            return res
        first = sheets[0]
        res.quad = first[1].round(2).tolist()
        res.area_frac = round(first[2], 4)
        res.rectangularity = round(first[3], 4)
        if first[2] < settings.min_quad_area:
            res.status, res.reason = "rejected", "page too small in frame"
            return res
        if first[3] < settings.min_rectangularity:
            res.status, res.reason = "rejected", "boundary not page-like"
            return res

        outputs: list[tuple[np.ndarray, int, int]] = []
        for s_idx, (rect, quad_pts, _af, _rq) in enumerate(sheets):
            if settings.auto_orient:
                angle = orientation.detect_rotation_fast(rect, device=str(detector.device))
                if angle == 90:
                    rect = cv2.rotate(rect, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rect = cv2.rotate(rect, cv2.ROTATE_180)
                elif angle == 270:
                    rect = cv2.rotate(rect, cv2.ROTATE_90_COUNTERCLOCKWISE)
                res.rotation = angle

            if settings.dewarp == "doctr":
                rect = detector.rectify(rect)
                res.dewarp = "doctr"
            elif settings.dewarp == "dewnet":
                if not hasattr(detector, "dewnet"):
                    from .models import DewarpNetDewarper
                    detector.dewnet = DewarpNetDewarper(device=str(detector.device))
                rect = detector.dewnet.rectify(rect)
                res.dewarp = "dewnet"
            elif settings.dewarp == "uvdoc":
                from .models import UVDocDewarper
                rect = UVDocDewarper().rectify(rect)
                res.dewarp = "uvdoc"
            elif settings.dewarp == "best":
                # No single model wins: on three pages measured from source,
                # "none" won once, DocTr once, DewarpNet once. So every
                # candidate is produced, the ones that keep the text are kept,
                # and the straightest of those is used.
                from .models import DewarpNetDewarper, UVDocDewarper
                if not hasattr(detector, "dewnet"):
                    detector.dewnet = DewarpNetDewarper(device=str(detector.device))
                cands = {"none": rect}
                for name, fn in (("doctr", detector.rectify),
                                 ("dewnet", detector.dewnet.rectify)):
                    try:
                        cands[name] = fn(rect)
                    except Exception:  # noqa: BLE001
                        continue
                # Tesseract is the cost here, not the GPU (which sat at 6-10 %):
                # one line-detection pass per candidate, and the OCR confidence
                # only when it is the chosen criterion.
                from .sharp import row_sharpness
                scored = []
                for name, cand in cands.items():
                    gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
                    ink = float((255.0 - gray.astype(np.float32)).mean())
                    scored.append((name, cand, row_sharpness(cand), ink))
                # a candidate that cropped the page keeps less ink; a candidate
                # that blurred it loses sharpness. Text must be there first.
                most_ink = max(x[3] for x in scored)
                keep = [x for x in scored if x[3] >= 0.85 * most_ink] or scored
                if settings.pick_metric == "ocr" and len(keep) > 1:
                    confs = {n: ocr_quality(c) for n, c, _, _ in keep}
                    best_conf = max(confs.values())
                    keep = [x for x in keep if confs[x[0]] >= best_conf - 2.0]
                    name, rect, sharp, ink = max(keep, key=lambda x: x[2])
                    res.pick = {"ocr": round(confs[name], 1), "sharp": round(sharp, 1),
                                "of": {n: [round(confs[n], 1), round(v, 1)]
                                       for n, _, v, _ in scored}}
                else:
                    name, rect, sharp, ink = max(keep, key=lambda x: x[2])
                    res.pick = {"sharp": round(sharp, 1),
                                "of": {n: [0.0, round(v, 1)] for n, _, v, _ in scored}}
                res.dewarp = name

            if settings.deskew:
                rect, _ = orientation.deskew(rect, max_angle=10.0)

            # the sheet is flat now; what is left is the paper's own bow, which
            # only the text lines can measure — and a picture has no lines to
            # measure, so the grid must not be forced onto it
            kind = page_kind(rect, settings)
            res.kind = kind
            text_page = kind in ("single_text", "single_text_curled", "unknown")
            if settings.line_straighten and text_page and kind != "graphic":
                # The grid is only kept when tesseract agrees: the measured bow
                # can fall while the text gets worse (IMG_20251211_164318 lost
                # 14 OCR points that way), so the downstream metric decides.
                cand, line_info = lines.unwarp(rect)
                if line_info.get("corrected"):
                    if ocr_quality(cand) > ocr_quality(rect) + settings.line_min_gain:
                        rect = cand
                    else:
                        line_info["rejected"] = "ocr_no_gain"
                res.lines = line_info

            if settings.illum_mode == "doctr":
                rect = detector.illuminate(rect)

            # saturated background (bed sheet, table cloth) that survived segmentation
            if settings.trim_background:
                rect = trim_background(rect)

            # spreads were already split at the gutter, before the warp, so each
            # page got its own homography; splitting again here only re-cuts
            # pages whose shadow happens to look like a spine
            split_here = settings.split_spreads and not settings.split_sheets
            pages = spine.split_spread(rect) if split_here else [rect]
            kept = []
            for page in pages:
                surface_ok, m_surface = quality.surface_checks(
                    page, max_texture=settings.max_texture, min_paper=settings.min_paper)
                if settings.illum_mode == "classical":
                    page = illum.scan_finish(page)
                elif settings.illum_mode == "doctr":
                    page = illum.auto_contrast(page)
                content_ok, m_content = quality.content_checks(page, settings.min_ink,
                                                               settings.min_lines)
                res.content = {**m_surface, **m_content}
                if surface_ok and content_ok:
                    kept.append(page)
            for i, page in enumerate(kept):
                outputs.append((page, s_idx, i if len(kept) > 1 else -1))
        if not outputs:
            res.status, res.reason = "rejected", res.reason or "no page content (ink/lines/paper)"
            return res
        multi_sheet = len(sheets) > 1
        for page, s_idx, p_idx in outputs:
            page = _finish_size(page, settings.scan_long_side)
            suffix = ""
            if multi_sheet:
                suffix += f"_s{s_idx + 1}"
            if p_idx >= 0:
                suffix += f"_p{p_idx + 1}"
            out_path = out_dir / f"{path.stem}{suffix}.png"
            if out_path.exists():
                tag = hashlib.sha1(str(path).encode()).hexdigest()[:6]
                out_path = out_dir / f"{path.stem}_{tag}{suffix}.png"
            cv2.imwrite(str(out_path), page, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            res.outputs.append(str(out_path))
            if settings.save_debug:
                dbg = img.copy()
                q = sheets[s_idx][1]
                cv2.polylines(dbg, [q.astype(np.int32)], True, (0, 0, 255), 12)
                cv2.imwrite(str(out_dir / f"{path.stem}{suffix}_debug.jpg"), dbg,
                            [cv2.IMWRITE_JPEG_QUALITY, 80])
        res.pages = len(outputs)
        res.status = "ok"
    except Exception as exc:  # noqa: BLE001
        res.status, res.reason = "error", repr(exc)
    finally:
        res.elapsed_s = round(time.time() - t0, 2)
        if detector.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
    return res
