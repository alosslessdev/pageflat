"""Word-line grid unwarping on the rectified page.

A sheet that has been perspective-corrected is flat, but the paper is still
curved. The page tells you how much: printed text sits on a grid. Word lines
run horizontally whatever the page does, paragraph blocks keep their left and
right edges vertical, and the page frame is a rectangle. When the paper bends,
all three bend together.

So the correction is measured, not guessed:

* every word line gives ``y(x)`` — where its ink actually is across the line;
* every line's left and right ink edges give ``x(y)`` — where the block edge
  actually is down the page;
* a displacement field is interpolated between those samples and the page is
  resampled once.

Every coordinate inside this module lives in one of two frames, and every
function says which: *work* (the downscaled copy that detection runs on) or
*input* (the image the caller handed in). ``detect_word_lines`` returns lines
in the work frame plus the factor that takes them back to the input frame;
``unwarp`` scales its field by that factor and resamples the input frame.

The same word lines say whether the sheet holds two pages side by side: a
column no line crosses is a gutter.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .orientation import ocr_lang

WORK_SIDE = 1600

# a page whose median line bow is under this (work-frame pixels) is flat: the
# measurement itself is good to about a pixel, so a smaller bow is not worth
# resampling the page for
BOW_MIN = 2.0
# beyond this fraction of the page height the bow is not paper curl any more:
# a crumpled sheet bends each line differently and cannot be straightened
MAX_BOW = 0.006
# residual shape of one line is resampled to this many points across its span
CURVE_POINTS = 33


@dataclass
class WordLine:
    top: int
    bottom: int
    mean_y: float
    x0: int
    x1: int
    # y(x) - mean_y sampled at CURVE_POINTS points evenly across [x0, x1],
    # with the straight-line part of the line taken out
    curve: np.ndarray = field(default_factory=lambda: np.zeros(CURVE_POINTS))
    bow: float = 0.0            # sagitta: peak-to-peak of the residual


# --------------------------------------------------------------------- ink
def _text_ink(gray: np.ndarray) -> np.ndarray:
    """Dark strokes on light paper, with photographs and page tone removed."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 31, 14)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    h, w = bw.shape
    keep = ((stats[:, cv2.CC_STAT_AREA] >= 6)
            & (stats[:, cv2.CC_STAT_HEIGHT] <= max(4, int(0.03 * h)))
            & (stats[:, cv2.CC_STAT_WIDTH] <= 0.45 * w))
    keep[0] = False
    return (keep[labels].astype(np.uint8) * 255)


# ------------------------------------------------------------------ lines
def _line_boxes(ink: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Text lines as horizontal runs of ink: works when the closing joins words.

    The closing kernel is a little taller than it is wide so a line that is
    already bent still joins into one box: a purely horizontal closing leaves
    a sloped line in fragments, one per word.
    """
    h, w = ink.shape
    kw = max(5, int(0.02 * w))
    kv = max(3, int(0.006 * w))
    joined = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kv)))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 5 or ch > max(8, int(0.06 * h)):
            continue
        if cw < max(12, int(0.02 * w)) or cw < 1.2 * ch:
            continue
        if ink[y:y + ch, x:x + cw].mean() / 255.0 < 0.05:
            continue
        boxes.append((x, y, cw, ch))
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _robust_poly(x: np.ndarray, y: np.ndarray, deg: int = 3, iters: int = 3,
                 k: float = 2.5) -> np.ndarray:
    """Polynomial fit that ignores descenders, bullets and page numbers.

    Printed text sits on one baseline; a descender or a stray mark is a gross
    outlier under it. Plain least squares would follow those outliers, so the
    fit is repeated on the inliers only.
    """
    deg = min(deg, max(1, y.size - 2))
    coef = np.polyfit(x, y, deg)
    for _ in range(iters):
        resid = y - np.polyval(coef, x)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        sigma = max(1.4826 * mad, 0.6)
        inlier = np.abs(resid) <= k * sigma
        if inlier.sum() < deg + 2:
            break
        coef = np.polyfit(x[inlier], y[inlier], deg)
    return coef


def _line_curve(ink: np.ndarray, box: tuple[int, int, int, int]
                ) -> tuple[np.ndarray | None, float, int, int]:
    """Bow of one line: the sagitta left when its slant is taken out.

    The line's position across the page is its *baseline* — the bottom of the
    ink in each column. The baseline is what printed type shares: ascenders
    and x-height letters all sit on it, and the per-column ink centroid does
    not (a column of a capital sits higher than a column of ``x``, which alone
    is worth several pixels of fake bow). Descenders below it are outliers and
    are dropped by a robust polynomial fit. What is left after the straight
    line through that fit is removed is the paper's bend, and its peak-to-peak
    range is the bow: a line that is merely slanted scores zero.
    """
    x, y, cw, ch = (int(v) for v in box)
    x, y = max(0, x), max(0, y)
    pad = max(2, ch // 5)                    # tesseract boxes clip descenders
    y0, y1 = max(0, y - pad), min(ink.shape[0], y + ch + pad)
    cw = min(cw, ink.shape[1] - x)
    if cw < 16 or y1 - y0 < 6:
        return None, 0.0, x, x + max(0, cw)
    band = ink[y0:y1, x:x + cw] > 0
    rows = np.arange(band.shape[0], dtype=np.float64)
    bottom = np.where(band, rows[:, None], -1.0).max(axis=0)
    cols = np.nonzero(bottom >= 0)[0]
    if cols.size < 16:
        return None, 0.0, x, x + max(0, cw)
    # normalise the abscissa to [-1, 1] so the fit is conditioned and the
    # residual is readable at the line's own ends
    half = max(1.0, np.ptp(cols.astype(np.float64)) / 2.0)
    centre = float(cols.min() + cols.max()) / 2.0
    xs = (cols.astype(np.float64) - centre) / half
    coef = _robust_poly(xs, bottom[cols])
    grid = np.linspace(-1.0, 1.0, CURVE_POINTS)
    fit = np.polyval(coef, grid)
    slope, icpt = np.polyfit(grid, fit, 1)
    residual = fit - (slope * grid + icpt)
    residual -= np.median(residual)
    return residual.astype(np.float64), float(np.ptp(residual)), \
        x + int(cols.min()), x + int(cols.max())


def tesseract_lines(gray: np.ndarray, max_side: int = WORK_SIDE
                    ) -> list[tuple[int, int, int, int]]:
    """Text lines as tesseract sees them, in the coordinates of `gray`.

    Tesseract's layout analysis is the most reliable line detector available
    here. It runs on a downscaled copy, and the boxes are brought back to
    `gray`'s frame exactly once — callers must not scale them again.

    The installed tessdata often has no ``eng``; asking for it by name fails
    with empty output, so the language comes from `ocr_lang()`.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    small = (cv2.resize(gray, (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
        tmp = fh.name
    try:
        if not cv2.imwrite(tmp, small):
            return []
        cmd = ["tesseract", tmp, "stdout", "--psm", "3"]
        lang = ocr_lang()
        if lang:
            cmd += ["-l", lang]
        out = subprocess.run(cmd + ["tsv"], capture_output=True, text=True,
                             timeout=180).stdout
    except Exception:
        return []
    finally:
        Path(tmp).unlink(missing_ok=True)
    back = 1.0 / scale if scale > 0 else 1.0
    boxes = []
    for row in out.splitlines()[1:]:
        parts = row.split("\t")
        if len(parts) < 10 or parts[0] != "4":
            continue
        try:
            x, y, bw, bh = (int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]))
        except ValueError:
            continue
        if bh < 4 or bw < 12:
            continue
        boxes.append((int(round(x * back)), int(round(y * back)),
                      int(round(bw * back)), int(round(bh * back))))
    return _dedupe(boxes)


def _dedupe(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """One box per text line: tesseract reports blocks that nest around lines."""
    kept: list[tuple[int, int, int, int]] = []
    # smallest first: tesseract repeats a line as part of larger block boxes,
    # and the tight box is the one that follows a single baseline
    for box in sorted(boxes, key=lambda b: b[2] * b[3]):
        x, y, bw, bh = box
        for kx, ky, kw, kh in kept:
            iw = min(x + bw, kx + kw) - max(x, kx)
            ih = min(y + bh, ky + kh) - max(y, ky)
            if iw <= 0 or ih <= 0:
                continue
            if iw * ih > 0.6 * min(bw * bh, kw * kh):
                break
        else:
            kept.append(box)
    return kept


def detect_word_lines(gray: np.ndarray, work_side: int = WORK_SIDE,
                      boxes: list[tuple[int, int, int, int]] | None = None
                      ) -> tuple[list[WordLine], float]:
    """Text lines plus the scale that takes them back to the input frame.

    `boxes`, when given, are in `gray`'s frame. The returned `WordLine`s are in
    the work frame (the image shrunk to `work_side`); multiply by the returned
    factor to get back to `gray`.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, work_side / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(gray, (max(16, int(round(w * scale))),
                                  max(16, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray
    ink = _text_ink(small)
    sw = small.shape[1]
    if boxes:
        found = [(int(round(v[0] * scale)), int(round(v[1] * scale)),
                  int(round(v[2] * scale)), int(round(v[3] * scale))) for v in boxes]
    else:
        found = _line_boxes(ink)
        widths = [b[2] for b in found]
        # fragments, not lines: the closing never joined the words of one line
        if len(found) >= 4 and float(np.median(widths)) >= 0.25 * sw:
            pass
        elif ink.any() and ink.mean() / 255.0 >= 0.002:
            # too little ink joined into lines (handwriting, heavy paper
            # texture, or a bend that split the lines): ask tesseract. Its
            # boxes are already in the work frame because it was handed `small`.
            found = tesseract_lines(small, max_side=work_side)
        else:
            found = []
    if len(found) >= 8:
        # a box much taller than its neighbours holds two lines joined where
        # the paper bent them together; its baseline belongs to neither
        median_h = float(np.median([b[3] for b in found]))
        found = [b for b in found if b[3] <= 1.7 * median_h]
    out = []
    ceiling = 0.05 * small.shape[0]
    for box in found:
        if box[3] < 4 or box[2] < 12:
            continue
        curve, bow, x0, x1 = _line_curve(ink, box)
        if curve is None or x1 - x0 < 12 or bow > ceiling:
            continue
        out.append(WordLine(box[1], box[1] + box[3], box[1] + box[3] / 2.0,
                            x0, x1, curve, bow))
    return out, (1.0 / scale if scale > 0 else 1.0)


# ------------------------------------------------------------------ fields
def _dy_field(lines: list[WordLine], shape: tuple[int, int], bow: float
              ) -> np.ndarray | None:
    """Vertical shift per pixel, in work-frame pixels, so word lines become flat.

    Each line's residual curve says how far its ink sits from the line it
    should be on. The page has no text between lines, so the shift for a row
    is interpolated between the curves of the line above and the line below
    it: rows with text get that line's own measurement, rows without get the
    bend the paper carries from one line to the next.
    """
    h, w = shape
    active = sorted((l for l in lines
                     if l.x1 - l.x0 >= max(24, int(0.1 * w)) and l.curve.size >= 2),
                    key=lambda l: l.mean_y)
    if len(active) < 6:
        return None
    curves = np.zeros((len(active), w), np.float32)
    for i, line in enumerate(active):
        src = np.linspace(line.x0, line.x1, line.curve.size)
        curves[i] = np.interp(np.arange(w, dtype=np.float64), src, line.curve)
    rows = np.array([l.mean_y for l in active], dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    below = np.clip(np.searchsorted(rows, ys), 0, len(rows) - 1)
    above = np.clip(below - 1, 0, len(rows) - 1)
    gap = np.maximum(rows[below] - rows[above], 1e-6)
    t = np.clip((ys - rows[above]) / gap, 0.0, 1.0).astype(np.float32)
    shift = curves[above] * (1.0 - t)[:, None] + curves[below] * t[:, None]
    shift = cv2.GaussianBlur(shift, (3, 3), 0)
    # over-correction guard: no line may move further than the most bent line
    # asked for (with some slack for the interpolation between lines)
    biggest = float(max(l.bow for l in active))
    cap = min(0.06 * h, max(2.0, 1.5 * biggest, 2.0 * bow))
    shift = np.clip(shift, -cap, cap)
    if float(np.abs(shift).max()) < 1.0:
        return None
    return shift


def _clusters(values: np.ndarray, tol: float) -> list[np.ndarray]:
    order = np.argsort(values)
    groups: list[list[int]] = []
    for idx in order:
        if groups and values[idx] - values[groups[-1][-1]] <= tol:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return [np.array(g) for g in groups if len(g) >= 3]


def _margin_drift(edges: np.ndarray, ys: np.ndarray, tol: float, need: int
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Systematic drift of one margin, or None when the margin is just ragged.

    Paragraphs share a margin; when the paper curls the shared edge moves with
    the row. Multiple blocks (columns) give several margins, so the samples are
    clustered first and only the largest cluster is trusted. A margin that is
    merely ragged has no trend once its cluster is centred, and is rejected.
    """
    best = None
    for group in _clusters(edges, tol):
        if group.size < need:
            continue
        if best is None or group.size > best.size:
            best = group
    if best is None or best.size < 3:
        return None
    drift = edges[best] - float(np.median(edges[best]))
    rows = ys[best]
    order = np.argsort(rows)
    rows, drift = rows[order], drift[order]
    span = float(np.ptp(drift))
    if span < 1.0:
        return None
    slope, icpt = np.polyfit(rows, drift, 1)
    resid = drift - (slope * rows + icpt)
    if float(np.median(np.abs(resid))) > 0.35 * span:
        return None                     # ragged, not a drifting margin
    return rows, drift


def _dx_field(lines: list[WordLine], shape: tuple[int, int], n_rows: int = 32
              ) -> np.ndarray | None:
    """Horizontal shift per pixel, in work-frame pixels, so block edges stay vertical.

    Paragraph blocks hold a left and a right margin. When the paper bends those
    edges drift with the row, and the drift is interpolated across the page:
    the left margin controls the left of the text, the right margin the right.
    """
    h, w = shape
    wide = [l for l in lines if (l.x1 - l.x0) > 0.15 * w]
    if len(wide) < 8:
        return None
    ys = np.array([l.mean_y for l in wide], dtype=np.float64)
    lefts = np.array([l.x0 for l in wide], dtype=np.float64)
    rights = np.array([l.x1 for l in wide], dtype=np.float64)
    need = max(3, int(0.25 * len(wide)))
    left = _margin_drift(lefts, ys, 0.03 * w, need)
    right = _margin_drift(rights, ys, 0.03 * w, need)
    if left is None and right is None:
        return None
    rows = np.linspace(0, max(1, h - 1), n_rows)
    l_rows, l_drift = left if left is not None else (np.arange(2), np.zeros(2))
    r_rows, r_drift = right if right is not None else (np.arange(2), np.zeros(2))
    d_l = np.interp(rows, l_rows, l_drift) if left is not None else None
    d_r = np.interp(rows, r_rows, r_drift) if right is not None else None
    if d_l is None:
        d_l = np.zeros_like(rows)
    if d_r is None:
        d_r = np.zeros_like(rows)
    lx = float(np.median(lefts)) if left is not None else float(np.percentile(lefts, 10))
    rx = float(np.median(rights)) if right is not None else float(np.percentile(rights, 90))
    if rx <= lx + 1:
        return None
    gx = np.clip((np.linspace(0.0, float(w - 1), n_rows) - lx) / (rx - lx), 0.0, 1.0)
    drift = (d_l[:, None] * (1.0 - gx)[None, :] + d_r[:, None] * gx[None, :]).astype(np.float32)
    drift = cv2.GaussianBlur(drift, (3, 3), 0)
    drift = np.clip(drift, -0.02 * w, 0.02 * w)
    if float(np.median(np.abs(drift))) < 0.5:
        return None
    return cv2.resize(drift, (w, h), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------- unwarp
def unwarp(img: np.ndarray, boxes: list[tuple[int, int, int, int]] | None = None
           ) -> tuple[np.ndarray, dict]:
    """Make word lines horizontal and block edges vertical, like a grid in a frame.

    Returns the image unchanged (same array, ``corrected`` False) when the page
    is already flat, when too few lines could be measured, or when the bow is
    too large or too uneven to be paper curl. ``info`` carries the line count,
    the median ``bow`` and the p75-p25 ``spread`` in input pixels, and the
    ``rejected`` reason when nothing was done.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines, back = detect_word_lines(gray, boxes=boxes)
    info = {"lines": len(lines), "bow": 0.0, "corrected": False}
    if len(lines) < 6:
        return img, info
    h, w = gray.shape[:2]
    shape = (max(16, int(round(h / back))), max(16, int(round(w / back))))
    bows = np.array([l.bow for l in lines], dtype=np.float64)
    bow = float(np.median(bows))
    spread = float(np.percentile(bows, 75) - np.percentile(bows, 25))
    info["bow"] = round(bow * back, 2)
    info["spread"] = round(spread * back, 2)
    if bow < BOW_MIN:
        return img, info
    # Paper curl bends every line by about the same, moderate amount. A
    # crumpled sheet (or a photo of a pile of pages) bends lines by tens of
    # pixels and by different amounts from line to line; no single smooth
    # field can undo that, and trying shears the page and costs OCR.
    if bow > MAX_BOW * shape[0]:
        info["rejected"] = "bow_too_large"
        return img, info
    if spread > max(1.5, bow):
        info["rejected"] = "bow_inconsistent"
        return img, info
    dy = _dy_field(lines, shape, bow)
    dx = _dx_field(lines, shape)
    if dy is None and dx is None:
        return img, info
    size = (w, h)
    dy_full = cv2.resize(dy, size, interpolation=cv2.INTER_CUBIC) * back \
        if dy is not None else None
    dx_full = cv2.resize(dx, size, interpolation=cv2.INTER_CUBIC) * back \
        if dx is not None else None
    if dy_full is not None and float(np.abs(dy_full).max()) < 1.0:
        dy_full = None
    if dx_full is not None and float(np.abs(dx_full).max()) < 1.0:
        dx_full = None
    if dy_full is None and dx_full is None:
        return img, info
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + (dx_full if dx_full is not None else 0.0)
    map_y = ys + (dy_full if dy_full is not None else 0.0)
    np.clip(map_x, 0, w - 1, out=map_x)
    np.clip(map_y, 0, h - 1, out=map_y)
    out = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    info["corrected"] = True
    info["dx"] = dx_full is not None
    info["dy"] = dy_full is not None
    return out, info


def column_gap(lines: list[WordLine], width: int, min_frac: float = 0.05) -> int | None:
    """Column that most word lines leave empty: the gutter of a two-page spread."""
    if len(lines) < 8:
        return None
    votes = np.zeros(width, np.int32)
    for line in lines:
        votes[max(0, line.x0):min(width, line.x1 + 1)] += 1
    active = votes >= max(4, int(0.3 * len(lines)))
    if not active.any():
        return None
    lo = int(np.argmax(active))
    hi = int(len(active) - np.argmax(active[::-1]))
    if hi - lo < 0.2 * width:
        return None
    best, best_x = 0, -1
    min_gap = int(min_frac * width)
    x = lo
    while x < hi:
        if not active[x]:
            y = x
            while y < hi and not active[y]:
                y += 1
            if y - x > best and y - x >= min_gap and lo < x and y < hi:
                best, best_x = y - x, (x + y) // 2
            x = y
        else:
            x += 1
    return best_x if best else None
