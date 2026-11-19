"""Perspective correction from a pointmap, before any dewarping.

The mask-quad homography asks "which four corners does the paper seem to have". A
pointmap asks the question the camera actually answers: where is each pixel in 3D?
Depth Anything 3 returns depth *and* the intrinsics for a single view, which is the
same family of model used to build pointmaps for Gaussian splatting, and from those
two the sheet is a plane to fit rather than a quad to guess.

    depth, K  ->  back-project  ->  RANSAC plane  ->  warp rays through it

The warp is exact under the pinhole model: for each output pixel, shoot a ray from the
camera through the rectified image plane, intersect the fitted plane, and read the
source pixel that ray came from. Nothing depends on the page boundary being visible,
which is the failure that defeated the mask on pages whose edges were cropped away.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PlaneFit:
    normal: np.ndarray          # unit normal in camera space
    offset: float               # plane: n . X + d = 0
    inliers: float              # fraction of sampled points on the plane
    warp: np.ndarray            # 3x3 homography, source -> rectified


def back_project(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-space points for every pixel, shape (h, w, 3)."""
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth.astype(np.float32)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def fit_plane(points: np.ndarray, mask: np.ndarray | None = None,
              iters: int = 400, tol_frac: float = 0.01) -> tuple[np.ndarray, float, float] | None:
    """RANSAC a plane through the dominant surface: (normal, d, inlier fraction)."""
    pts = points.reshape(-1, 3)
    if mask is not None:
        pts = pts[mask.reshape(-1) > 0]
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 1000:
        return None
    span = float(np.percentile(np.abs(pts), 95)) or 1.0
    tol = max(1e-6, tol_frac * span)
    idx = np.random.default_rng(0).integers(0, pts.shape[0], size=(iters, 3))
    best = (0, None, 0.0)
    for a, b, c in idx:
        p0, p1, p2 = pts[a], pts[b], pts[c]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -float(n @ p0)
        dist = np.abs(pts @ n + d)
        count = int((dist < tol).sum())
        if count > best[0]:
            best = (count, (n, d), float(np.median(dist[dist < tol])) if count else 0.0)
    if best[1] is None:
        return None
    n, d = best[1]
    # refine on the inliers
    dist = np.abs(pts @ n + d)
    inl = pts[dist < tol]
    if inl.shape[0] < 100:
        return None
    centroid = inl.mean(axis=0)
    _u, _s, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    n = vt[2] / np.linalg.norm(vt[2])
    d = -float(n @ centroid)
    return n, d, inl.shape[0] / pts.shape[0]


def rectifying_homography(normal: np.ndarray, d: float, K: np.ndarray) -> np.ndarray:
    """Homography taking the fitted plane to a fronto-parallel view: H = K R K^-1.

    `R` is the rotation that brings the plane's normal onto the camera axis, so the
    image is resampled as if the camera had been turned to face the sheet squarely.
    The plane's distance `d` cancels in the projection — only the orientation matters
    for perspective, which is why no scale term appears here.
    """
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) or 1.0)
    cam_axis = np.array([0.0, 0.0, -1.0])          # camera looks down -Z
    if float(n @ cam_axis) < 0:
        n = -n
    v = np.cross(n, cam_axis)
    s = float(np.linalg.norm(v))
    c = float(n @ cam_axis)
    if s < 1e-8:
        R = np.eye(3)
    else:
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    H = K @ R @ np.linalg.inv(K)
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def rectify(img_bgr: np.ndarray, side: int = 518,
            mask: np.ndarray | None = None) -> tuple[np.ndarray, dict, np.ndarray]:
    """Perspective-correct a page from its pointmap. Returns (warped, info, H)."""
    from .da3 import estimate

    est = estimate(img_bgr, side=side)
    depth, K = est["depth"], est["intrinsics"]
    pts = back_project(depth, K)
    got = fit_plane(pts, mask=mask)
    if got is None:
        return img_bgr, {"rectified": False, "reason": "no plane"}, np.eye(3)
    n, d, inl = got
    H = rectifying_homography(n, d, K)
    h, w = img_bgr.shape[:2]
    out = cv2.warpPerspective(img_bgr, H, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, {"rectified": True, "inliers": round(inl, 3),
                 "normal": [round(float(v), 3) for v in n]}, H
