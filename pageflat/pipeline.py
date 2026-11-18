"""End-to-end page -> flat scan pipeline."""
from __future__ import annotations

import difflib
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
    trim_edges: bool = True         # take the strip of table / neighbour off the frame edge
    straighten_edges: bool = True   # shear rows so the sheet's own edge is a straight line
    orient_url: str = ""            # vision model for pages where OCR cannot decide
    quad_url: str = ""              # vision model for the sheet's four corners
    fix_tilt: bool = False          # OFF: the rule-based lean squares the *content*, not
                                    # the page (see docs/LOOP.md) and made the page edge
                                    # worse on 13 of 30 pages
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
    use_depth: bool = True              # include the monocular-depth candidate in "best"
    use_aligner: bool = True            # DocAligner corners first, mask as fallback
    aligner_conf: float = 0.4
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
    trim: dict = field(default_factory=dict)   # edge-trim result
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
    if settings.quad_url:
        # A strong vision model places the four corners where the learned mask
        # over-reaches onto pale furniture (measured: 782 px against a true edge of
        # 780, where the mask says 893). Only attempted when the endpoint answers;
        # the mask below is the fallback, so a missing model changes nothing.
        from .segment import ask_quad

        quad = ask_quad(settings.quad_url, small)
        if quad is not None and cv2.contourArea(quad.astype(np.float32)) > 0.05 * small.size / 3:
            pts = (quad / scale).astype(np.float32)
            rect = perspective.warp_sheet(img, pts, max_side=settings.max_intermediate)
            area = abs(cv2.contourArea(quad)) / float(small.shape[0] * small.shape[1])
            return [(rect, pts.astype(np.float32), area, 1.0)]
    if settings.use_aligner:
        # a trained corner regressor beats a mask hull on cluttered tables; it
        # also says how sure it is, so a bad frame is refused instead of guessed
        from . import align
        found = align.corners(img, min_conf=settings.aligner_conf)
        if found is not None:
            quad, conf = found
            quad = perspective.expand_quad(quad, settings.quad_margin, img.shape)
            rect = perspective.warp_sheet(img, quad, max_side=settings.max_intermediate)
            res_quad = quad.astype(np.float32)
            area = abs(cv2.contourArea(quad)) / float(img.shape[0] * img.shape[1])
            return [(rect, res_quad, area, 1.0)]
    mask = detector.page_mask(small)
    if not mask.any():
        return []
    alt_parts = None
    sam_mask = None
    if settings.use_sam and getattr(detector, "sam", None) is not None:
        # SAM follows the sheet where the learned model follows colour, and a pale
        # table is paper-coloured. Measured on IMG_20251229_122443: docTR's mask
        # reaches y=893 of 900, SAM's stops at 780 where the sheet actually ends
        # (an independent vision model put the same edge at 867 of a 1000-grid).
        # It is not a free win though — a tighter mask can clip text — so the two
        # candidates are arbitrated further down by how many lines each one cuts.
        from .segment import extend_to_lines

        sam_mask = detector.sam.sheet_mask(small, (mask >= 0.5).astype(np.uint8) * 255)
        if sam_mask is not None and float((sam_mask >= 0.5).mean()) > 0.05:
            # only what both witnesses call sheet, then grown back over the text:
            # neither the learned model's over-reach nor SAM's under-reach survives,
            # and a page may never lose a line to either.
            both = ((mask >= 0.5) & (sam_mask >= 0.5)).astype(np.float32)
            if float(both.mean()) > 0.03:
                # keep it as an *alternative*, not a replacement: which mask is right
                # is a per-page question, settled below by how many text lines each
                # one cuts (SAM alone clipped five lines on IMG_20251229_122456 where
                # the learned mask clipped none).
                grown, _grow = extend_to_lines(small, both)
                alt_parts = [(grown >= 0.5).astype(np.uint8)]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if settings.split_sheets:
        kind = page_kind(small, settings) if settings.content_route else "unknown"
        parts = perspective.sheet_masks(small, mask, max_sheets=settings.max_sheets,
                                        kind=kind)
    else:
        parts = [(mask >= 0.5).astype(np.uint8)]
    if alt_parts:
        # Warp both candidates and keep the one that cuts fewer text lines.
        def _cuts(part):
            from .charlines import detect_lines

            q = geom.mask_to_quad(part)
            if q is None or not part.any():
                return None
            w = perspective.warp_sheet(img, perspective.expand_quad(
                (geom.refine_quad(gray, q.points, search_frac=0.02) / scale).astype(np.float32),
                settings.quad_margin, img.shape), max_side=1400)
            n = 0
            for line in detect_lines(w):
                if len(line.boxes) < 3:
                    continue
                if (line.left <= 6 or line.right >= w.shape[1] - 6
                        or min(b[1] for b in line.boxes) <= 6
                        or max(b[1] + b[3] for b in line.boxes) >= w.shape[0] - 6):
                    n += 1
            return n
        a = _cuts(parts[0]) if len(parts) == 1 else None
        b = _cuts(alt_parts[0])
        if a is not None and b is not None and b < a:
            parts = alt_parts
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


def _tilt_final(page: np.ndarray) -> np.ndarray:
    """One more vertical pass on the finished page.

    Measured on the it1 output: the rotation plus the per-row shear takes the
    median tilt from 1.29 to 0.95 deg and the pages above 1 deg from 37 to 27 on a
    57-page sample, where the rotation alone left p90 worse than before. It is
    cheap here because the page is already at its final size and the rules are
    detected on a 1400-px copy.
    """
    from .vlines import rectify as _vrect

    # One pass. Two were measured offline as 1.15 -> 0.87 -> 0.79 deg on the same
    # pages, but in the pipeline (it3) they left the median at 1.17 against 1.10
    # with one, at twice the runtime. The offline gain does not survive the run.
    out, info = _vrect(page)
    return out if info.get("corrected") else page


def _deskew_final(page: np.ndarray, max_angle: float = 12.0) -> tuple[np.ndarray, float]:
    """Remove the fine rotation on the finished page.

    `deskew` is called earlier in the pipeline too, but the warp that follows puts
    the angle back: measured on three of the worst pages, the finished output still
    carried 9.5, 10.0 and 9.5 degrees of skew, and running the same function on
    those pages takes them to 0.4, 0.0 and 0.0. Same lesson as orientation — the
    stage that changes the pixels has to come after the stages that do.
    """
    h, w = page.shape[:2]
    small = page
    if max(h, w) > 1200:
        s_ = 1200 / float(max(h, w))
        small = cv2.resize(page, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_AREA)
    # `skew_angle`, not the text-line angle. The text-line version measured better on
    # its own six worst pages (3.69 -> 1.11 deg and five others), but on the whole
    # benchmark it made squareness *worse* (median 0.42 -> 0.54, pages above 1 deg
    # 11 -> 14) and cost nine pages to the quality gate. Reverted.
    out, angle = orientation.deskew(small, max_angle=max_angle)
    if abs(float(angle)) < 0.3:
        return page, float(angle)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(page, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE), float(angle)


def _orient_final(page: np.ndarray, detector, orient_url: str = "") -> tuple[np.ndarray, int]:
    """Turn the finished page upright, but only when two detectors agree.

    The fast classifier is a MobileNet that sees a thumbnail; on half a spread it
    will call an upright page upside down with confidence, and rotating on that
    call makes the page worse. The OCR-scored detector is slower but reads the
    text, so it is used as the second opinion — and only for the pages the fast
    one wants to turn, which is a quarter of them.
    """
    # The OCR-scored detector, on every page. Trusting the thumbnail classifier
    # when it says "upright" was tried and is not safe: on four pages it called a
    # sideways page upright, and the output kept 90 and 180 degree rotations. The
    # classifier is only a hint; the text decides.
    h, w = page.shape[:2]
    small = page
    if max(h, w) > 1600:
        s_ = 1600 / float(max(h, w))
        # INTER_LINEAR, matching the resample the benchmark uses to judge this:
        # detect_rotation scores OCR of four rotations, and INTER_AREA's
        # softening is enough to flip the answer on a graphic page (the same page
        # reads 180 with one interpolation and 0 with the other).
        small = cv2.resize(page, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_LINEAR)
    angle = orientation.detect_rotation(small)
    if angle not in _ROTATE and orient_url:
        # no rotation was earned: this is the class where OCR is blind (15 % of the
        # benchmark), and a vision model is the only thing that reads it
        from .charlines import detect_lines

        if len([l for l in detect_lines(small) if len(l.boxes) >= 6]) < 5:
            vis = orientation.ask_vision(orient_url, page)
            if vis in _ROTATE and vis:
                return cv2.rotate(page, _ROTATE[vis]), vis
        return page, angle
    if angle in _ROTATE:
        # A page with no text grid is a flyer, and a flyer's mirrored advertising
        # banner "reads" better upside down: three pages in it7 were flagged while
        # being perfectly upright. Rotate only when there is a text grid to justify
        # it and the fast classifier agrees.
        from .charlines import detect_lines

        grid = [l for l in detect_lines(small) if len(l.boxes) >= 6]
        alt = orientation.detect_rotation_fast(page, device=str(detector.device))
        if len(grid) < 5 or alt != angle:
            angle = 0
        # A detector that answers the same angle for a page and for that page
        # rotated by it is contradicting itself — three pages do exactly this. Ask
        # a different model rather than trusting an answer that cannot be acted on.
        probe = cv2.rotate(small, _ROTATE[angle])
        if orientation.detect_rotation(probe) == angle:
            alt = orientation.detect_rotation_fast(page, device=str(detector.device))
            if alt in _ROTATE and alt != angle:
                angle = alt
            else:
                return page, 0
        return cv2.rotate(page, _ROTATE[angle]), angle
    return page, angle


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
                    if settings.straighten_edges:
                        from .edges import straighten_edge

                        rect, _ei = straighten_edge(rect, side="left")
                        res.pick["edge_shear"] = _ei

                if settings.illum_mode == "doctr":
                    rect = detector.illuminate(rect)
                if settings.trim_background:
                    rect = trim_background(rect)
                pages = [rect]
                if settings.split_spreads:
                    pages, _split_info = spine.split_without_losing_text(rect)
                    res.pick["split"] = _split_info
                kept = []
                for page in pages:
                    surface_ok, m_surface = quality.surface_checks(
                        page, max_texture=settings.max_texture,
                        min_paper=settings.min_paper)
                    if settings.illum_mode == "classical":
                        page = gpu_ops.scan_finish_gpu(page, device=dev)
                    elif settings.illum_mode == "doctr":
                        page = illum.auto_contrast(page)
                    # strip of table / neighbouring page at the frame edge; the
                    # GPU path needs its own copy of this, as it has its own loop
                    if settings.trim_edges:
                        from .crop import trim_borders

                        page, edge_info = trim_borders(page)
                        res.trim = edge_info
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
                    # orientation on the page as written: the GPU path decided on
                    # the pre-warp sheet and left pages sideways while reporting 0
                    if settings.auto_orient:
                        # One pass, on the page as written. Two passes (an early one
                        # on the rectified sheet plus this) measured worse: 13 of 128
                        # misoriented in it6 against 7 with this alone.
                        if settings.fix_tilt:
                            page = _tilt_final(page)
                        page, _ang = _orient_final(page, detector, settings.orient_url)
                        if _ang in _ROTATE:
                            res.rotation = _ang
                        if settings.deskew:
                            page, _sk = _deskew_final(page)
                            res.pick["deskew_final"] = round(_sk, 2)
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
                from . import depthwarp
                names = [("doctr", detector.rectify), ("dewnet", detector.dewnet.rectify),
                         ("uvdoc", UVDocDewarper().rectify)]
                if settings.use_depth:
                    names.append(("depth", lambda im: depthwarp.rectify(im)[0]))
                for name, fn in names:
                    try:
                        cands[name] = fn(rect)
                    except Exception:  # noqa: BLE001
                        continue
                # Tesseract is the cost here, not the GPU (which sat at 6-10 %):
                # one line-detection pass per candidate, and the OCR confidence
                # only when it is the chosen criterion.
                from .sharp import row_sharpness
                from .detect import text_lines
                from .blocks import blocks_of
                scored = []
                for name, cand in cands.items():
                    gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
                    ink = float((255.0 - gray.astype(np.float32)).mean())
                    if settings.pick_metric == "drift":
                        # score by the rule itself: the median distance of the
                        # text blocks from rectangles, negated so higher is better
                        blocks = blocks_of(text_lines(cand))
                        drifts = sorted(b.rect_error for b in blocks)
                        score = -(drifts[len(drifts) // 2]) if drifts else -1e6
                    else:
                        score = row_sharpness(cand)
                    scored.append((name, cand, score, ink))
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
            if settings.straighten_edges:
                from .edges import straighten_edge

                rect, _edge_info = straighten_edge(rect, side="left")
                res.pick["edge_shear"] = _edge_info
            if settings.fix_tilt:
                from .vlines import vertical_drift, rectify as _vrect

                _t0 = vertical_drift(rect)["tilt_deg"]
                rect, _ti = _vrect(rect)
                if _ti.get("corrected"):
                    res.pick["tilt"] = {"before": round(_t0, 2) if _t0 is not None else None,
                                        "after": vertical_drift(rect)["tilt_deg"]}

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
            pages = [rect]
            if split_here:
                pages, _split_info = spine.split_without_losing_text(rect)
                res.pick["split"] = _split_info
            kept = []
            for page in pages:
                surface_ok, m_surface = quality.surface_checks(
                    page, max_texture=settings.max_texture, min_paper=settings.min_paper)
                if settings.illum_mode == "classical":
                    page = illum.scan_finish(page)
                elif settings.illum_mode == "doctr":
                    page = illum.auto_contrast(page)
                # whatever the sheet lies on, left at the frame edge by the crop:
                # a table strip or a sliver of the neighbouring page. Per page,
                # after the split, so a spread's two halves are trimmed on their
                # own edges. Undone when it costs OCR.
                if settings.trim_edges:
                    from .crop import trim_borders

                    page, edge_info = trim_borders(page)
                    res.trim = edge_info
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
            # Check orientation on the page as it will be written, not on an
            # intermediate buffer: the GPU path decides on the pre-warp sheet and
            # was leaving 34 of 148 pages sideways or upside down while reporting
            # rotation 0. One classifier pass here costs ~10 ms and cannot disagree
            # with the pixels that get saved.
            if settings.auto_orient:
                if settings.fix_tilt:
                    page = _tilt_final(page)
                page, final_angle = _orient_final(page, detector, settings.orient_url)
                if final_angle in _ROTATE:
                    res.rotation = final_angle
                if settings.deskew:
                    page, _sk = _deskew_final(page)
                    res.pick["deskew_final"] = round(_sk, 2)
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


def ocr_text(img_bgr: np.ndarray, max_side: int = 1600) -> tuple[str, float, int]:
    """Recognised text, its mean confidence, and the word count."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    small = (cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    cv2.imwrite(tmp, small)
    try:
        cmd = ["tesseract", tmp, "stdout", "--psm", "3"]
        if orientation.ocr_lang():
            cmd += ["-l", orientation.ocr_lang()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception:  # noqa: BLE001
        return "", 0.0, 0
    finally:
        Path(tmp).unlink(missing_ok=True)
    words = out.split()
    return out, ocr_quality(img_bgr, max_side), len(words)


def keeps_text(before: np.ndarray, after: np.ndarray, min_chars: float = 0.95,
               max_conf_drop: float = float("inf")) -> tuple[bool, dict]:
    """Does this correction still produce the page's words?

    Confidence is the wrong question: OCR reads a warped page perfectly well,
    and a resampled one slightly worse, so confidence measures legibility, not
    straightness — and it is what an earlier gate used, wrongly. What the
    correction must not do is *lose content*: crop a column, blank a region,
    smear a line out of recognition. So the test is the transcription itself —
    how many characters of the original survive.
    """
    text_a, conf_a, words_a = ocr_text(before)
    text_b, conf_b, words_b = ocr_text(after)
    a = " ".join(text_a.split())
    b = " ".join(text_b.split())
    common = 0
    if a and b:
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        common = sum(block.size for block in sm.get_matching_blocks())
    kept = common / max(1, len(a))
    # character-overlap is too strict on a resampled image (re-OCR jitters at
    # margins: a fully preserved page measured 0.755); what must not happen is
    # *losing* words — a cropped column or a blanked region shows up as a
    # collapse in the count, not as a shift in glyphs
    enough_words = words_b >= 0.85 * max(1, words_a)
    ok = enough_words and (len(b) >= 0.8 * len(a) if a else True)
    return ok, {"chars_kept": round(kept, 3), "words": [words_a, words_b],
                "conf": [round(conf_a, 1), round(conf_b, 1)]}
