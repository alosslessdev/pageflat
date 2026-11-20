"""Boundary-curve dewarping, after DRCCBI (Istomin et al., "Efficient Document Image
Dewarping via Hybrid Deep Learning and Cubic Polynomial Geometry").

The paper's geometry stage is not a homography and not a dense flow field: it fits each
of the document's four sides with a cubic polynomial, builds a grid from those four
curves, and remaps to a uniform rectangle. That is a Coons patch in closed form --

    S(u, v) = (1-v)·T(u) + v·B(u) + (1-u)·L(v) + u·R(v) - bilinear(corners)

-- where T, B, L, R are the four boundary curves resampled to the same parameter
(eq. 4 of the paper) and the bilinear term removes the double counting at the corners.
Sampling S on a uniform (u, v) grid and remapping the source through it is exactly
"ConstructGrid -> FindIntersections -> CreateUniformGrid -> BicubicRemap" without the
intermediate knot bookkeeping, and it preserves topology by construction: the grid
cannot cross itself while the four curves are simple and the corners are ordered.

The curves come from a vision model marking points along each page edge, which is what
a quad cannot express: a page photographed on a curved surface has bowed sides, and the
four-corner homography straightens the corners while leaving the bow in place.
"""

from __future__ import annotations

import numpy as np
import cv2

SIDES = ("top", "right", "bottom", "left")


def _resample(points: np.ndarray, n: int) -> np.ndarray:
    """n points evenly spaced along the polyline by arc length."""
    pts = np.asarray(points, np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    want = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(want, cum, pts[:, 0]), np.interp(want, cum, pts[:, 1])], axis=1)


def _cubic_fit(points: np.ndarray, along: str) -> None:
    """Kept for parity with the paper's eq. 5: each side is a cubic in one coordinate.
    The Coons patch below interpolates the same curves exactly, so this exists to make
    the correspondence explicit and to let `side_polynomials` answer "what cubic is
    this edge" for diagnostics."""
    xs, ys = points[:, 0], points[:, 1]
    t = xs if along == "x" else ys
    return np.polyfit(t, ys if along == "x" else xs, 3) if len(points) >= 4 else None


def coons_grid(sides: dict[str, np.ndarray], out_w: int, out_h: int,
               steps: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Map a uniform output grid back into the source image.

    `sides` holds four polylines in source pixel coordinates, each running from one
    corner to the next in the order top: tl->tr, right: tr->br, bottom: br->bl,
    left: bl->tl. Returns (map_x, map_y) for cv2.remap.
    """
    top = _resample(sides["top"], steps)
    right = _resample(sides["right"], steps)
    bottom = _resample(sides["bottom"][::-1], steps)   # br->bl becomes bl->br
    left = _resample(sides["left"][::-1], steps)       # bl->tl becomes tl->bl

    c_tl, c_tr = top[0], top[-1]
    c_br, c_bl = bottom[-1], bottom[0]
    if np.linalg.norm(top[-1] - right[0]) > np.linalg.norm(top[-1] - right[-1]):
        right = right[::-1]
    if np.linalg.norm(bottom[-1] - right[-1]) > np.linalg.norm(bottom[-1] - right[0]):
        right = right[::-1]

    u = np.linspace(0.0, 1.0, steps)
    v = np.linspace(0.0, 1.0, steps)
    U, V = np.meshgrid(u, v)                          # (steps, steps)

    Tu = np.stack([np.interp(U[0], u, top[:, 0]), np.interp(U[0], u, top[:, 1])], axis=1)
    Bu = np.stack([np.interp(U[0], u, bottom[:, 0]), np.interp(U[0], u, bottom[:, 1])], axis=1)
    Lv = np.stack([np.interp(V[:, 0], v, left[:, 0]), np.interp(V[:, 0], v, left[:, 1])], axis=1)
    Rv = np.stack([np.interp(V[:, 0], v, right[:, 0]), np.interp(V[:, 0], v, right[:, 1])], axis=1)

    X = ((1 - V) * Tu[None, :, 0].repeat(steps, 0) + V * Bu[None, :, 0].repeat(steps, 0)
         + (1 - U) * Lv[:, None, 0].repeat(steps, 1) + U * Rv[:, None, 0].repeat(steps, 1)
         - ((1 - U) * (1 - V) * c_tl[0] + U * (1 - V) * c_tr[0]
            + (1 - U) * V * c_bl[0] + U * V * c_br[0]))
    Y = ((1 - V) * Tu[None, :, 1].repeat(steps, 0) + V * Bu[None, :, 1].repeat(steps, 0)
         + (1 - U) * Lv[:, None, 1].repeat(steps, 1) + U * Rv[:, None, 1].repeat(steps, 1)
         - ((1 - U) * (1 - V) * c_tl[1] + U * (1 - V) * c_tr[1]
            + (1 - U) * V * c_bl[1] + U * V * c_br[1]))

    map_x = cv2.resize(X.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(Y.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return map_x, map_y


def dewarp_by_edges(img: np.ndarray, sides: dict[str, np.ndarray],
                    max_side: int = 3200) -> np.ndarray:
    """Dewarp `img` so the four annotated boundary curves become the rectangle's edge."""
    top = np.asarray(sides["top"], np.float64).reshape(-1, 2)
    bottom = np.asarray(sides["bottom"], np.float64).reshape(-1, 2)
    left = np.asarray(sides["left"], np.float64).reshape(-1, 2)
    right = np.asarray(sides["right"], np.float64).reshape(-1, 2)

    width = 0.5 * ((np.linalg.norm(np.diff(top, axis=0), axis=1).sum())
                   + (np.linalg.norm(np.diff(bottom, axis=0), axis=1).sum()))
    height = 0.5 * ((np.linalg.norm(np.diff(left, axis=0), axis=1).sum())
                    + (np.linalg.norm(np.diff(right, axis=0), axis=1).sum()))
    scale = min(1.0, max_side / max(width, height, 1.0))
    out_w = max(64, int(round(width * scale)))
    out_h = max(64, int(round(height * scale)))

    map_x, map_y = coons_grid(sides, out_w, out_h)
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def sides_from_annotation(entry: dict, w: int, h: int) -> dict[str, np.ndarray]:
    """Turn a `{corners, edges}` annotation in normalised coordinates into pixel sides."""
    corners = entry.get("corners") or []
    edges = entry.get("edges") or {}
    if len(corners) != 4:
        raise ValueError("annotation needs four corners")
    c = np.asarray(corners, np.float64) * np.array([w, h], np.float64)
    out = {}
    for i, side in enumerate(SIDES):
        pts = edges.get(side) or []
        pts = np.asarray(pts, np.float64).reshape(-1, 2) * np.array([w, h], np.float64) if len(pts) else np.zeros((0, 2))
        a, b = c[i], c[(i + 1) % 4]
        if len(pts) == 0:
            pts = np.stack([a, b])
        else:
            # keep the corner-to-corner order the four sides require
            if np.linalg.norm(pts[0] - b) < np.linalg.norm(pts[0] - a):
                pts = pts[::-1]
            pts = np.vstack([a, pts, b])
        out[side] = pts
    return out


def tps_dewarp(img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray,
               out_size: tuple[int, int], regulariser: float = 0.0) -> np.ndarray:
    """Thin-plate-spline dewarp through annotated control points.

    This is the control-point route the IEEE method (DocReal, WACV 2024) and the ACPR
    paper it descends from share: a set of points identified on the page, each with the
    position it should occupy, and a smooth interpolant between them. Unlike the Coons
    patch it also fixes *interior* deformation, which is what a page resting on a curved
    object actually needs -- boundary curves alone cannot straighten a text line that is
    flat at both ends and bowed in the middle.

    src_pts/dst_pts are (N, 2) in pixels; the spline is fitted on normalised
    coordinates for conditioning.
    """
    src = np.asarray(src_pts, np.float64)
    dst = np.asarray(dst_pts, np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("thin-plate spline needs at least four matched control points")

    def normalise(p, w, h):
        return p / np.array([w, h], np.float64) * 2.0 - 1.0

    h, w = img.shape[:2]
    s_n = normalise(src, w, h)
    d_n = normalise(dst, w, h)

    def kernel(r2):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r2 > 0, r2 * np.log(r2), 0.0)

    def fit(points):
        n = len(points)
        d2 = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
        K = kernel(d2) + regulariser * np.eye(n)
        P = np.hstack([np.ones((n, 1)), points])
        A = np.zeros((n + 3, n + 3))
        A[:n, :n] = K
        A[:n, n:] = P
        A[n:, :n] = P.T
        return A, P

    out_w, out_h = out_size
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, out_w), np.linspace(-1.0, 1.0, out_h))
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    maps = []
    for target in (d_n, s_n):
        A, P = fit(s_n)
        b = np.zeros(len(s_n) + 3)
        b[: len(s_n)] = target
        # solve for the coefficients that map control points -> target, then evaluate
        coeff = np.linalg.lstsq(A, b, rcond=None)[0]
        d2 = ((grid[:, None, :] - s_n[None, :, :]) ** 2).sum(-1)
        gk = kernel(d2)
        vals = gk @ coeff[: len(s_n)] + np.hstack([np.ones((len(grid), 1)), grid]) @ coeff[len(s_n):]
        maps.append(vals.reshape(out_h, out_w))

    # forward spline (src -> dst) gives the target coordinates of each source pixel;
    # invert by remapping from the forward grid to sample the source.
    fx, fy = maps[0], maps[1]
    xs = ((fx + 1.0) * 0.5 * out_w).astype(np.float32)
    ys = ((fy + 1.0) * 0.5 * out_h).astype(np.float32)
    return cv2.remap(img, xs, ys, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def dewarp_with_control_points(img: np.ndarray, control_points, out_size=None,
                               max_side: int = 3200) -> np.ndarray:
    """Faithful port of Xie et al., "Document Dewarping with Control Points" (ICDAR 2021,
    github.com/gwxie/Document-Dewarping-with-Control-Points).

    That repository keeps only the *border* of its fiducial grid (`get_matric_edge`),
    spaces the reference points evenly on a rectangle, scatters the control points at
    those references with linear interpolation, and remaps. Two consequences matter
    here: the annotation is a closed polygon walk around the page edge, and uneven
    spacing between annotated points is harmless because the reference spacing is what
    the remap is built on.

    `control_points` is that closed polygon, (N, 2) in source pixels, ordered around
    the page; `out_size` is (w, h) of the reference rectangle.
    """
    from scipy.interpolate import griddata

    ctrl = np.asarray(control_points, np.float64).reshape(-1, 2)
    n = len(ctrl)
    if n < 8:
        raise ValueError("need at least eight control points around the page edge")

    if out_size is None:
        # the reference rectangle keeps the annotated polygon's aspect ratio
        lo, hi = ctrl.min(axis=0), ctrl.max(axis=0)
        bw, bh = max(hi[0] - lo[0], 1.0), max(hi[1] - lo[1], 1.0)
        scale = min(1.0, max_side / max(bw, bh))
        out_w, out_h = max(64, int(bw * scale)), max(64, int(bh * scale))
    else:
        out_w, out_h = out_size

    # reference points: the same count, evenly spaced around the rectangle
    perim = 2.0 * (out_w + out_h)
    dist = np.linspace(0.0, perim, n, endpoint=False)
    ref = np.zeros((n, 2))
    for i, d in enumerate(dist):
        d = d % perim
        if d < out_w:                       ref[i] = (d, 0.0)
        elif d < out_w + out_h:             ref[i] = (out_w, d - out_w)
        elif d < 2 * out_w + out_h:         ref[i] = (out_w - (d - out_w - out_h), out_h)
        else:                               ref[i] = (0.0, out_h - (d - 2 * out_w - out_h))

    gy, gx = np.mgrid[0:out_h, 0:out_w]
    # same construction as the repository: interpolate the source position at every
    # reference (target) pixel, then remap
    src_x = griddata(ref, ctrl[:, 0], (gx, gy), method="linear").astype(np.float32)
    src_y = griddata(ref, ctrl[:, 1], (gx, gy), method="linear").astype(np.float32)
    src_x = np.nan_to_num(src_x, nan=float(np.median(ctrl[:, 0])))
    src_y = np.nan_to_num(src_y, nan=float(np.median(ctrl[:, 1])))
    return cv2.remap(img, src_x, src_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def control_points_from_mask(mask: np.ndarray, count: int = 24) -> np.ndarray | None:
    """Sample `count` control points evenly around the page mask's outer contour.

    The same construction the control-point paper uses after its network predicts the
    fiducial grid: what actually drives the warp is the ordered walk around the page
    border, so a binary mask is already enough to obtain it. Sampling by arc length
    (rather than by polygon vertex index) keeps the spacing even, which matters because
    the reference spacing is uniform.
    """
    m = (np.asarray(mask) > 0.5).astype(np.uint8)
    if m.ndim == 3:
        m = m[..., 0]
    cont, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cont:
        return None
    c = max(cont, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if len(c) < 8:
        return None
    closed = np.vstack([c, c[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg)]) / total
    want = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.stack([np.interp(want, cum, closed[:, 0]),
                     np.interp(want, cum, closed[:, 1])], axis=1)
