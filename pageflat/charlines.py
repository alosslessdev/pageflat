"""Text lines built bottom-up from character boxes.

Morphological closing joins a line — but it also joins across a column gutter,
because the closing kernel is a fixed width. Grouping individual letters instead
never bridges a gap it was not told to bridge: two characters belong to the same
line when they sit at the same height and are close relative to their own size,
and a column gutter is a gap far wider than that.

That matters twice over. A line is the geometry signal the warp needs, and it is
the ruler the flattening is judged by: if a "line" spans two columns, both the
correction and the score are wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CharLine:
    boxes: list[tuple[int, int, int, int]]      # x, y, w, h in input pixels

    @property
    def left(self) -> float:
        return min(b[0] for b in self.boxes)

    @property
    def right(self) -> float:
        return max(b[0] + b[2] for b in self.boxes)

    @property
    def baseline(self) -> tuple[float, float, float]:
        """(slope, intercept, residual) of the bottom edges, y = slope*x + b."""
        xs = np.array([b[0] + b[2] / 2.0 for b in self.boxes], float)
        ys = np.array([b[1] + b[3] for b in self.boxes], float)
        if len(xs) < 2:
            return 0.0, float(ys[0] if len(ys) else 0.0), 0.0
        A = np.stack([xs, np.ones_like(xs)], 1)
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ coef
        return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(resid ** 2)))

    @property
    def height(self) -> float:
        return float(np.median([b[3] for b in self.boxes]))


def ink_mask(img_bgr: np.ndarray, dark_only: bool = True) -> np.ndarray:
    """Letter strokes: dark in every channel, which survives coloured paper.

    A coloured flyer panel is bright in at least one channel, so requiring all
    channels to be dark keeps the print and drops the background.
    """
    if dark_only:
        mx = img_bgr.max(axis=2)
        return ((mx < 110).astype(np.uint8) * 255)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 14)


def characters(mask: np.ndarray, min_h: int = 4, max_h_frac: float = 0.06) -> list:
    h = mask.shape[0]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, ch, area = stats[i]
        if ch < min_h or ch > max_h_frac * h:
            continue
        if area < 5 or w > 0.25 * mask.shape[1]:
            continue
        out.append((int(x), int(y), int(w), int(ch)))
    return out


def group_lines(boxes: list, max_gap: float = 1.6, overlap: float = 0.5) -> list[CharLine]:
    """Union characters that share a height band and are close to each other."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(len(boxes)):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, len(boxes)):
            xj, yj, wj, hj = boxes[j]
            if xj - (xi + wi) > max_gap * max(hi, hj):
                break                      # sorted by x: nothing further can be close
            # vertical overlap relative to the shorter box
            ov = max(0, min(yi + hi, yj + hj) - max(yi, yj))
            if ov < overlap * min(hi, hj):
                continue
            if max(hi, hj) > 2.5 * min(hi, hj):
                continue                   # different type sizes: different lines
            union(i, j)

    groups: dict[int, list] = {}
    for i, b in enumerate(boxes):
        groups.setdefault(find(i), []).append(b)
    lines = []
    for g in groups.values():
        if len(g) < 3:
            continue                       # a line needs at least three letters
        line = CharLine(g)
        # drop fragments: a "line" too wide relative to its height is a merge
        if line.right - line.left > 60 * line.height:
            continue
        lines.append(line)
    return sorted(lines, key=lambda l: (min(b[1] for b in l.boxes), l.left))


def detect_lines(img_bgr: np.ndarray, dark_only: bool = True) -> list[CharLine]:
    return group_lines(characters(ink_mask(img_bgr, dark_only=dark_only)))
