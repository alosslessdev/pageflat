"""Text blocks must be rectangles.

The rule the corpus is judged by: a paragraph block has a left edge and a
right edge that are vertical, and a top and bottom that are horizontal. Paper
curl breaks that — the left margin drifts down the block, the right margin
drifts the other way, and the block reads as warped even when no single line
is obviously bent.

This module measures that directly from the OCR line boxes: group the lines
of a page into blocks by horizontal overlap, fit each block's left and right
edge, and report how far those edges wander from a vertical line. It also
applies the cheap repair: a per-row horizontal shift that pulls the edges
back onto verticals.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Block:
    lines: list = field(default_factory=list)      # (x0, y0, x1, y1)
    left_drift: float = 0.0
    right_drift: float = 0.0
    justified: bool = False

    @property
    def rect_error(self) -> float:
        """Distance from a rectangle.

        Only the margins that are *supposed* to be straight count. A paragraph
        justifies its left edge always, but its right edge only when the text
        is justified — ragged-right text varies by design, and counting that
        would report normal typography as warping.
        """
        return max(self.left_drift, self.right_drift if self.justified else 0.0)


def load_lines(ocr_json: Path) -> tuple[list[tuple[float, float, float, float]], float, float]:
    """(lines, scale_x, scale_y) in the OCR's own frame."""
    data = json.loads(Path(ocr_json).read_text())
    lines = []
    for line in data.get("lines", []):
        if line.get("conf", 0) < 45:
            continue
        x, y, w, h = line["bbox"]
        if w < 20:
            continue
        lines.append((float(x), float(y), float(x + w), float(y + h)))
    return lines, float(data.get("width") or 1), float(data.get("height") or 1)


def blocks_of(lines: list[tuple[float, float, float, float]], tolerance: float = 0.05
              ) -> list[Block]:
    """Lines sharing a horizontal extent *and* sitting next to each other.

    Horizontal overlap alone is not enough: the same margin runs down the whole
    page, so a block would swallow lines from every column and report a drift
    that is just the page's layout.
    """
    if not lines:
        return []
    # columns first: a page has margins at a few x positions, and merging lines
    # across them makes any block look bent when it is only a two-column layout
    lefts = np.array([l[0] for l in lines])
    width = max(1.0, float(np.percentile([l[2] for l in lines], 90) - np.percentile(lefts, 10)))
    tol = 0.02 * width
    order = np.argsort(lefts)
    columns: list[list[int]] = []
    for idx in order:
        if columns and lefts[idx] - lefts[columns[-1][-1]] <= tol:
            columns[-1].append(idx)
        else:
            columns.append([idx])
    out: list[Block] = []
    for col in columns:
        sub = [lines[i] for i in col]
        out.extend(_blocks_in_column(sub, tolerance))
    return out


def _blocks_in_column(lines: list[tuple[float, float, float, float]],
                      tolerance: float = 0.05) -> list[Block]:
    ordered = sorted(lines, key=lambda l: l[1])
    groups: list[list] = []
    for line in ordered:
        placed = False
        for group in groups:
            ref = group[-1]
            gap = line[1] - ref[3]
            height = max(1.0, ref[3] - ref[1])
            if not (-0.6 * height <= gap <= 1.2 * height):
                continue
            overlap = min(ref[2], line[2]) - max(ref[0], line[0])
            width = min(ref[2] - ref[0], line[2] - line[0])
            # a full-width heading overlaps every column beneath it, but it is
            # not part of that column: the two must be about the same width
            w_ref, w_line = ref[2] - ref[0], line[2] - line[0]
            similar = min(w_ref, w_line) / max(w_ref, w_line, 1.0) > 0.7
            if similar and width > 0 and overlap > (1.0 - tolerance) * width:
                group.append(line)
                placed = True
                break
        if not placed:
            groups.append([line])
    out = []
    for group in groups:
        if len(group) < 3:
            continue
        block = Block(lines=group)
        block.left_drift = _edge_error(group, 0)
        block.right_drift = _edge_error(group, 2)
        rights = np.array([l[2] for l in group])
        widths = np.array([l[2] - l[0] for l in group])
        # justified text reaches the same right margin line after line; ragged
        # text does not, and must not be judged by it
        block.justified = bool(
            np.ptp(np.percentile(rights, [10, 90])) < 0.015 * max(1.0, float(np.median(widths)) * 6)
            and float(np.percentile(widths, 75)) > 0.5 * float(max(widths)))
        out.append(block)
    return out


def _edge_error(group, index: int) -> float:
    """How far one edge of a block is from a single vertical line.

    Two faults matter and both break the rectangle: the edge may **lean**
    (a shear, which a slope captures) and it may **bow** (curl, which the
    residual captures). An indented first line is neither — it is one outlier —
    so the residual is measured by its interquartile range, which ignores it.
    """
    ys = np.array([(l[1] + l[3]) / 2.0 for l in group])
    xs = np.array([l[index] for l in group])
    if xs.size < 3:
        return 0.0
    slope, intercept = np.polyfit(ys, xs, 1)
    shear = abs(slope) * float(ys.max() - ys.min())
    residual = xs - (slope * ys + intercept)
    curl = float(np.percentile(residual, 75) - np.percentile(residual, 25))
    return shear + curl


def page_rect_error(lines, scale: float = 1.0) -> dict:
    """How far this page's blocks are from rectangles, in scaled pixels."""
    blocks = blocks_of(lines)
    if not blocks:
        return {"blocks": 0, "error": 0.0}
    errors = sorted(b.rect_error for b in blocks)
    return {"blocks": len(blocks),
            "error": round(errors[len(errors) // 2] * scale, 1),
            "worst": round(errors[-1] * scale, 1)}


def rectify_field(img: np.ndarray, lines, scale_x: float, scale_y: float,
                  max_shift: float = 0.05, grid: int = 24) -> tuple[np.ndarray, dict]:
    """Horizontal displacement field fitted to the block edges.

    Every block edge is a constraint: at (x_of_edge, y_of_line) the image must
    move so that the edge lands on the block's mean x. Left and right edges
    both constrain it, so columns at different heights are respected instead of
    being averaged into one row shift.
    """
    import scipy.interpolate as interp

    blocks = blocks_of(lines)
    if not blocks:
        return img, {"corrected": False}
    pts, vals = [], []
    for block in blocks:
        ys = np.array([(l[1] + l[3]) / 2.0 * scale_y for l in block.lines])
        lefts = np.array([l[0] * scale_x for l in block.lines])
        rights = np.array([l[2] * scale_x for l in block.lines])
        if ys.size < 3:
            continue
        for values in (lefts, rights):
            mean = float(np.mean(values))
            for y, v in zip(ys, values):
                pts.append((y, v))
                vals.append(mean - v)
    if len(pts) < 12:
        return img, {"corrected": False, "samples": len(pts)}
    pts = np.asarray(pts, float)
    vals = np.asarray(vals, float)
    h, w = img.shape[:2]
    gy = np.linspace(0, h - 1, grid)
    gx = np.linspace(0, w - 1, grid)
    field = interp.griddata(pts, vals, np.meshgrid(gy, gx, indexing="ij"),
                            method="linear", fill_value=0.0)
    field = cv2.GaussianBlur(field.astype(np.float32), (5, 5), 0)
    cap = max_shift * w
    field = np.clip(field, -cap, cap)
    if float(np.abs(field).max()) < 1.5:
        return img, {"corrected": False, "max_shift": round(float(np.abs(field).max()), 1)}
    full = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    gx_full, gy_full = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(gx_full + full, 0, w - 1)
    out = cv2.remap(img, map_x, gy_full, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "max_shift": round(float(np.abs(field).max()), 1),
                 "samples": int(len(pts))}


def rectify_blocks(img: np.ndarray, lines, scale_x: float, scale_y: float,
                   max_shift: float = 0.03) -> tuple[np.ndarray, dict]:
    """Shift each row horizontally so the block edges become vertical.

    The left and right edges are measured per row from the line boxes, a
    straight line is fitted to each, and the term that is *not* explained by
    the fit is the wander. Rows are shifted by the average of the two edges'
    wander, which squares the block without moving it off the page.
    """
    blocks = blocks_of(lines)
    if not blocks:
        return img, {"shifted": False}
    h, w = img.shape[:2]
    xs = np.arange(h, dtype=np.float64)
    samples = np.zeros(h)
    weight = np.zeros(h)
    for block in blocks:
        ys = np.array([(l[1] + l[3]) / 2.0 * scale_y for l in block.lines])
        lefts = np.array([l[0] * scale_x for l in block.lines])
        rights = np.array([l[2] * scale_x for l in block.lines])
        if ys.size < 3:
            continue
        for values in (lefts, rights):
            base = np.polyval(np.polyfit(ys, values, 0), ys)      # the edge's mean
            wander = values - base
            np.add.at(samples, np.clip(ys.astype(int), 0, h - 1), wander)
            np.add.at(weight, np.clip(ys.astype(int), 0, h - 1), 1.0)
    ok = weight > 0
    if ok.sum() < 4:
        return img, {"shifted": False}
    profile = np.zeros(h)
    profile[ok] = samples[ok] / weight[ok]
    xs_ok = np.nonzero(ok)[0]
    profile = np.interp(xs, xs_ok, profile[xs_ok])
    k = max(3, (h // 40) | 1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).ravel()
    cap = max_shift * w
    profile = np.clip(profile, -cap, cap)
    if float(np.abs(profile).max()) < 2.0:
        return img, {"shifted": False, "max_shift": round(float(np.abs(profile).max()), 1)}
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(grid_x + profile[:, None].astype(np.float32), 0, w - 1)
    out = cv2.remap(img, map_x, grid_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"shifted": True, "max_shift": round(float(np.abs(profile).max()), 1)}


def block_affines(lines, shape: tuple[int, int]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """One affine per block: its four fitted corners -> the same rectangle.

    Each block is described by four points — the left and right edges
    evaluated at the block's top and bottom. The destination is a rectangle
    with the block's mean margins. ``(M, centre, weight)`` is returned for the
    blend in :func:`rectify_blocks_affine`.
    """
    h, w = shape[:2]
    out = []
    for block in blocks_of(lines):
        ys = np.array([(l[1] + l[3]) / 2.0 for l in block.lines])
        lefts = np.array([l[0] for l in block.lines])
        rights = np.array([l[2] for l in block.lines])
        if ys.size < 3:
            continue
        top, bottom = float(ys.min()), float(ys.max())
        if bottom - top < 20:
            continue
        try:
            sl, il = np.polyfit(ys, lefts, 1)
            sr, ir = np.polyfit(ys, rights, 1)
        except Exception:
            continue
        src = np.array([[sl * top + il, top], [sr * top + ir, top],
                        [sr * bottom + ir, bottom], [sl * bottom + il, bottom]], np.float32)
        dst = np.array([[np.median(lefts), top], [np.median(rights), top],
                        [np.median(rights), bottom], [np.median(lefts), bottom]], np.float32)
        # a shear/curl correction is a translation per row; an affine fitted to
        # these four points also removes any residual scale/skew
        M = cv2.getPerspectiveTransform(src, dst)
        out.append((M, np.array([np.median(lefts), (top + bottom) / 2.0]), float(bottom - top)))
    return out


def rectify_blocks_affine(img: np.ndarray, lines, grid: int = 48,
                          sigma_frac: float = 0.35, max_shift: float = 0.05
                          ) -> tuple[np.ndarray, dict]:
    """Blend the per-block affines into one smooth field and resample."""
    h, w = img.shape[:2]
    affines = block_affines(lines, (h, w))
    if not affines:
        return img, {"corrected": False, "blocks": 0}
    gy = np.linspace(0, h - 1, grid)
    gx = np.linspace(0, w - 1, grid)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X, Y], -1).reshape(-1, 2).astype(np.float32)
    sig = sigma_frac * float(np.median([a[2] for a in affines])) or 50.0
    num = np.zeros_like(pts)
    den = np.zeros((pts.shape[0], 1), np.float32)
    for M, centre, _ in affines:
        d2 = ((pts[:, 0] - centre[0]) ** 2 + (pts[:, 1] - centre[1]) ** 2)
        wt = np.exp(-d2 / (2 * sig * sig)).reshape(-1, 1).astype(np.float32)
        moved = cv2.perspectiveTransform(pts.reshape(1, -1, 2), M).reshape(-1, 2)
        num += wt * (moved - pts)
        den += wt
    disp = num / np.maximum(den, 1e-6)
    field_x = disp[:, 0].reshape(grid, grid)
    field_y = disp[:, 1].reshape(grid, grid)
    cap_x, cap_y = max_shift * w, 0.02 * h
    field_x = np.clip(field_x, -cap_x, cap_x)
    field_y = np.clip(field_y, -cap_y, cap_y)
    if float(np.abs(field_x).max()) < 1.0 and float(np.abs(field_y).max()) < 1.0:
        return img, {"corrected": False, "blocks": len(affines)}
    fx = cv2.resize(field_x, (w, h), interpolation=cv2.INTER_CUBIC)
    fy = cv2.resize(field_y, (w, h), interpolation=cv2.INTER_CUBIC)
    gxx, gyy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(gxx + fx, 0, w - 1)
    map_y = np.clip(gyy + fy, 0, h - 1)
    out = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "blocks": len(affines),
                 "max_dx": round(float(np.abs(field_x).max()), 1),
                 "max_dy": round(float(np.abs(field_y).max()), 1)}


def rectify_rows(img: np.ndarray, lines, max_shift: float = 0.04) -> tuple[np.ndarray, dict]:
    """Shift each row so every block's left margin becomes a vertical line.

    A curled sheet moves the whole row of text sideways as it rises off the
    page; that is a per-row translation, not a perspective. Each block says
    where its margin *should* be (its median) and where it actually is at each
    of its lines; the row takes the average of those answers.
    """
    h, w = img.shape[:2]
    blocks = blocks_of(lines)
    if not blocks:
        return img, {"corrected": False, "blocks": 0}
    acc = np.zeros(h)
    cnt = np.zeros(h)
    for block in blocks:
        ys = np.array([(l[1] + l[3]) / 2.0 for l in block.lines])
        lefts = np.array([l[0] for l in block.lines])
        target = float(np.median(lefts))
        step = target - lefts
        idx = np.clip(ys.astype(int), 0, h - 1)
        np.add.at(acc, idx, step)
        np.add.at(cnt, idx, 1.0)
    ok = cnt > 0
    if ok.sum() < 4:
        return img, {"corrected": False, "blocks": len(blocks)}
    profile = np.zeros(h)
    profile[ok] = acc[ok] / cnt[ok]
    xs_ok = np.nonzero(ok)[0]
    profile = np.interp(np.arange(h), xs_ok, profile[xs_ok])
    k = max(3, (h // 30) | 1)
    profile = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).ravel()
    cap = max_shift * w
    profile = np.clip(profile, -cap, cap)
    if float(np.abs(profile).max()) < 1.5:
        return img, {"corrected": False, "max_shift": round(float(np.abs(profile).max()), 1)}
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.clip(gx + profile[:, None].astype(np.float32), 0, w - 1)
    out = cv2.remap(img, map_x, gy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, {"corrected": True, "max_shift": round(float(np.abs(profile).max()), 1),
                 "blocks": len(blocks)}
