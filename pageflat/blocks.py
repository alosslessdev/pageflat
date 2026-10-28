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

    @property
    def rect_error(self) -> float:
        return max(self.left_drift, self.right_drift)


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
        lefts = np.array([l[0] for l in group])
        rights = np.array([l[2] for l in group])
        block.left_drift = float(np.percentile(lefts, 90) - np.percentile(lefts, 10))
        block.right_drift = float(np.percentile(rights, 90) - np.percentile(rights, 10))
        out.append(block)
    return out


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
