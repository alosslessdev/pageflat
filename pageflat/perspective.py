"""Perspective correction as the first stage: one homography per physical sheet.

The rest of the pipeline assumed a single page per photograph. When two sheets
lie side by side on the floor, the page mask covers both, ``mask_to_quad``
fits one hull around the pair, and the resulting homography squeezes both
sheets onto one page: the sheet that is "more inside" the hull keeps its
margins, the other one loses them. That is the "cut off" page.

Recovering a homography per sheet before anything else fixes it: the mask is
split along the seam between the sheets, a quad is fitted to each part, and
each quad gets its own warp. The seam is a real paper edge, so it is found on
the image (long straight edge inside the mask), not guessed from the shape.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from . import geom


def _components(binary: np.ndarray, min_area_frac: float = 0.02) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    total = binary.size
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_frac * total:
            out.append((labels == i).astype(np.uint8))
    return out or ([binary] if binary.any() else [])


def _fit(mask: np.ndarray) -> Optional[np.ndarray]:
    quad = geom.mask_to_quad(mask)
    return None if quad is None else quad.points


def _fill(mask: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of the quad's area that the mask actually covers."""
    h, w = mask.shape[:2]
    canvas = np.zeros((h, w), np.uint8)
    cv2.fillPoly(canvas, [quad.astype(np.int32)], 1)
    area = int(canvas.sum())
    return float(mask[canvas == 1].sum()) / area if area else 0.0


def _side_masks(shape: tuple[int, int], line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = (float(v) for v in line)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = (x2 - x1) * (yy - y1) - (y2 - y1) * (xx - x1)
    return (d >= 0).astype(np.uint8), (d < 0).astype(np.uint8)


def _seams(img: np.ndarray, mask: np.ndarray, min_len_frac: float = 0.4) -> list[np.ndarray]:
    """Long straight edges inside the mask: candidate sheet boundaries."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        return []
    span = min(ys.max() - ys.min(), xs.max() - xs.min())
    if span < 40:
        return []
    found = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=50,
                            minLineLength=int(min_len_frac * span), maxLineGap=12)
    if found is None:
        return []
    out = []
    for line in found[:, 0]:
        x1, y1, x2, y2 = line
        if np.hypot(x2 - x1, y2 - y1) < min_len_frac * span:
            continue
        # the seam must sit inside the mask, not trace its outer border
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if not (0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and mask[my, mx]):
            continue
        out.append(line.astype(np.float32))
    return out


def _split_on_seam(img: np.ndarray, mask: np.ndarray, max_sheets: int = 3) -> list[np.ndarray]:
    parts = [mask]
    total = float(mask.sum())
    while len(parts) < max_sheets:
        best = None
        for idx, part in enumerate(parts):
            quad = _fit(part)
            if quad is None:
                continue
            gain0 = _fill(part, quad)
            if gain0 >= 0.80:
                continue
            for line in _seams(img, part):
                a_all, b_all = _side_masks(part.shape[:2], line)
                a, b = part & a_all, part & b_all
                if min(a.sum(), b.sum()) < 0.12 * total:
                    continue
                qa, qb = _fit(a), _fit(b)
                if qa is None or qb is None:
                    continue
                gain = gain0 - min(_fill(a, qa), _fill(b, qb))
                if best is None or gain > best[0]:
                    best = (gain, idx, a.astype(np.uint8), b.astype(np.uint8))
        if best is None or best[0] <= 0.04:
            break
        _, idx, a, b = best
        parts[idx:idx + 1] = [a, b]
    return parts


def gutter_x(img: np.ndarray, mask: np.ndarray, lo_frac: float = 0.3,
             hi_frac: float = 0.7) -> Optional[int]:
    """Column of the darkest band inside the mask: the spine of an open book.

    A spread is one connected mask — the two pages meet at the gutter, which is
    a shadow, not a gap. Splitting on that shadow before warping is what lets
    each page get its own homography; warping the pair as one quad bakes the V
    shape into both pages.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys, xs = np.nonzero(mask)
    if xs.size < 16:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if (x1 - x0) < 1.05 * (y1 - y0):
        return None                      # page-shaped, not a spread
    band = gray[y0:y1 + 1]
    keep = mask[y0:y1 + 1].astype(bool)
    dark = np.where(keep, 255.0 - band, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            profile = np.nanmean(dark, axis=0)
    profile = np.nan_to_num(profile, nan=0.0)
    k = max(5, ((x1 - x0) // 60) | 1)
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (k, 1), 0).ravel()
    band_gray = gray
    lo, hi = int(x0 + lo_frac * (x1 - x0)), int(x0 + hi_frac * (x1 - x0))
    if hi - lo < 10:
        return None
    x = lo + int(np.argmax(smooth[lo:hi]))
    shoulder = np.median(np.concatenate([smooth[lo:max(lo + 1, x - k)],
                                         smooth[min(hi, x + k):hi]]))
    if smooth[x] - shoulder < 6.0:       # no real shadow, no gutter
        return None
    # both halves must be page-shaped, otherwise this is a landscape page with
    # something dark down its middle, not a spread
    for a, b in ((x0, x), (x, x1)):
        width, height = b - a, y1 - y0
        if height <= 0 or not (0.45 <= width / float(height) <= 1.05):
            return None
    # and the gutter itself must be smooth: a fold is a shadow, while a
    # crumpled single sheet has text running straight across its middle. Ink
    # is what rises fast across a row, a shadow does not.
    band = max(2, int(0.015 * (x1 - x0)))
    # along the fold, not across it: text alternates light and dark down a
    # column, a shadow is the same shade all the way down
    gy = np.abs(cv2.Sobel(band_gray, cv2.CV_32F, 0, 1, ksize=3))
    inside = mask.astype(bool)
    page_energy = float(np.percentile(gy[inside], 90)) if inside.any() else 0.0
    strip = np.zeros_like(inside)
    strip[:, max(0, x - band):x + band] = True
    strip &= inside
    if page_energy <= 0 or not strip.any():
        return None
    if float(np.percentile(gy[strip], 90)) > 0.5 * page_energy:
        return None
    return x


def _split_on_gutter(img: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    x = gutter_x(img, mask)
    if x is None:
        return [mask]
    h, w = mask.shape[:2]
    left = mask.copy()
    right = mask.copy()
    left[:, x:] = 0
    right[:, :x] = 0
    total = float(mask.sum())
    if min(left.sum(), right.sum()) < 0.1 * total:
        return [mask]
    return [left, right]


def sheet_masks(img: np.ndarray, mask: np.ndarray, thresh: float = 0.5,
                min_area_frac: float = 0.02, max_sheets: int = 3,
                kind: str = "unknown") -> list[np.ndarray]:
    """Split a page probability mask into one mask per physical sheet.

    ``kind`` comes from the content classifier: a wide newspaper sheet has
    columns and must not be cut in half, an open book must be.
    """
    binary = (mask >= thresh).astype(np.uint8)
    if not binary.any():
        return []
    # split unless the classifier positively says this is one wide sheet: a
    # missed split leaves a spread in the PDF, a wrong split destroys a page
    allow_gutter = kind not in ("spread_newspaper", "graphic", "not_page")
    out: list[np.ndarray] = []
    for comp in _components(binary, min_area_frac):
        parts = _split_on_gutter(img, comp) if allow_gutter else [comp]
        for part in parts:
            out.extend(_split_on_seam(img, part, max_sheets) if len(parts) == 1 else [part])
    return out


def _paper_like(px: np.ndarray) -> np.ndarray:
    """Bright, unsaturated pixels: paper rather than table, bed or desk."""
    px = px.astype(np.int16)
    b, g, r = px[..., 0], px[..., 1], px[..., 2]
    high = np.minimum(np.minimum(b, g), r)
    low = np.maximum(np.maximum(b, g), r)
    return (high > 95) & ((high - low) < 70)


def edge_has_ink(img: np.ndarray, quad: np.ndarray, band_frac: float = 0.02,
                 samples: int = 32, dark: int = 110) -> bool:
    """True when strokes run right up to the quad's edge.

    That is the signature of a crop that cut into the page: a real page edge
    has a blank margin between the paper border and the first letter. Without
    strokes at the edge there is nothing to recover, and reaching further out
    only drags the surroundings in.
    """
    quad = np.asarray(quad, np.float32)
    h, w = img.shape[:2]
    hits = 0
    total = 0
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        inward = quad.mean(axis=0) - (p0 + p1) / 2.0
        n = np.linalg.norm(inward)
        if n < 1e-6:
            continue
        inward = inward / n
        depth = max(2.0, band_frac * np.linalg.norm(p1 - p0))
        for t in np.linspace(0.1, 0.9, samples):
            pt = p0 + t * (p1 - p0) + inward * depth
            x, y = int(round(pt[0])), int(round(pt[1]))
            if not (2 <= x < w - 2 and 2 <= y < h - 2):
                continue
            patch = img[y - 2:y + 3, x - 2:x + 3]
            total += 1
            if float(patch.min()) < dark:
                hits += 1
    return total > 0 and hits >= 0.25 * total


def extend_to_paper(img: np.ndarray, quad: np.ndarray, max_frac: float = 0.035,
                    samples: int = 40, step: int = 2) -> np.ndarray:
    """Grow each quad edge outwards until the paper really ends.

    A mask that stops at the text block, not at the paper edge, crops the page
    and no later stage can recover the lost margin. Marching out along each
    edge's normal while the pixels still look like paper finds the real edge;
    the median of the march distances keeps a thumb or a shadow from dragging
    one sample away. Falls back to the input quad when nothing is found.
    """
    quad = np.asarray(quad, np.float32)
    h, w = img.shape[:2]
    centre = quad.mean(axis=0)
    out = quad.copy()
    for i in range(4):
        p0, p1 = quad[i], quad[(i + 1) % 4]
        mid = (p0 + p1) / 2.0
        normal = mid - centre
        n = np.linalg.norm(normal)
        if n < 1e-6:
            continue
        normal = normal / n
        ts = np.linspace(0.12, 0.88, samples)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        reach = []
        for pt in pts:
            d = 0
            miss = 0
            limit = max_frac * np.linalg.norm(p1 - p0)
            py = int(np.clip(round(pt[1]), 0, h - 1))
            px = int(np.clip(round(pt[0]), 0, w - 1))
            prev = img[py, px].astype(np.int16)
            while d < limit:
                d += step
                q = pt + normal * d
                xq, yq = int(round(q[0])), int(round(q[1]))
                if not (0 <= xq < w and 0 <= yq < h):
                    break
                here = img[yq, xq].astype(np.int16)
                # a paper edge is a hard brightness step; without this test a
                # white bedsheet around the page reads as paper and is kept
                if int(np.abs(here - prev).mean()) > 16:
                    break
                # and paper next door is blank: a neighbour newspaper or a
                # textured surface must not be swallowed as margin
                patch = img[max(0, yq - 4):yq + 5, max(0, xq - 4):xq + 5]
                if float(patch.std()) > 20.0:
                    break
                prev = here
                if _paper_like(here[None, None, :])[0, 0]:
                    miss = 0
                else:
                    miss += 1
                    if miss >= 3:
                        break
            reach.append(max(0.0, d - step if miss == 0 else d - miss * step - step))
        if not reach:
            continue
        grow = float(np.median(reach))
        # a real margin is parallel to the mask edge: if the samples disagree,
        # the march wandered into whatever lies next to the page
        spread = float(np.percentile(reach, 75) - np.percentile(reach, 25))
        if spread > max(8.0, 0.4 * grow):
            continue
        if grow <= step:
            continue
        # the strip we are about to add must be blank margin: if there is ink
        # in it, the march walked onto the next sheet, not onto this one
        inked = 0
        touched = 0
        for pt in pts:
            for d in np.linspace(step, grow, max(2, int(grow // step))):
                q = pt + normal * d
                xq, yq = int(round(q[0])), int(round(q[1]))
                if not (0 <= xq < w and 0 <= yq < h):
                    continue
                touched += 1
                if int(img[yq, xq].min()) < 110:
                    inked += 1
        if touched and inked > 0.02 * touched:
            continue
        shift = normal * grow
        out[i] += shift
        out[(i + 1) % 4] += shift
    out[:, 0] = out[:, 0].clip(0, w - 1)
    out[:, 1] = out[:, 1].clip(0, h - 1)
    return out.astype(np.float32)


def expand_quad(quad: np.ndarray, frac: float, shape: tuple[int, int]) -> np.ndarray:
    """Push a quad outwards so the paper edge is never clipped.

    The margin is a fraction of the quad's mean edge length, applied along the
    vector from the centre to each corner, then clipped to the image. Warping
    with ``BORDER_REPLICATE`` fills the ring with the nearest edge pixels.
    """
    quad = np.asarray(quad, np.float32)
    if frac <= 0:
        return quad
    centre = quad.mean(axis=0)
    edge = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[3] - quad[0])) / 2.0
    out = quad + (quad - centre) * (frac * edge / max(1e-6, np.linalg.norm(quad - centre, axis=1).mean()))
    h, w = shape[:2]
    out[:, 0] = out[:, 0].clip(0, w - 1)
    out[:, 1] = out[:, 1].clip(0, h - 1)
    return out.astype(np.float32)


def warp_sheet(img: np.ndarray, quad: np.ndarray, max_side: int = 3508) -> np.ndarray:
    """Rectify one sheet with its own homography."""
    return geom.warp_quad(img, quad, max_side=max_side)
