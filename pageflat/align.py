"""Page corners from DocAligner (ONNX), with a confidence you can refuse.

The segmentation mask answers "which pixels are paper", which on a table with
two sheets, cloth or clutter answers the wrong question. DocAligner regresses
the four corners of *the* document and returns a heatmap per corner, so the
peak of each heatmap is both the coordinate and a confidence — and a page whose
corners are not found can be refused instead of being guessed.

Model: `kittypdf/DocAligner-ONNX` (Apache-2.0), fastvit-sa24, 256x256 input,
4x128x128 heatmaps, ~10 ms on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "docaligner-fastvit-sa24.onnx"
_SESSION = None
SIDE = 256


def _session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort
        _SESSION = ort.InferenceSession(str(WEIGHTS), providers=["CPUExecutionProvider"])
    return _SESSION


def corners(img: np.ndarray, min_conf: float = 0.35
            ) -> Optional[tuple[np.ndarray, float]]:
    """Four corners in the input frame, ordered tl, tr, br, bl, and a confidence."""
    sess = _session()
    h, w = img.shape[:2]
    small = cv2.resize(img, (SIDE, SIDE), interpolation=cv2.INTER_LINEAR)
    x = (small[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = sess.get_inputs()[0].name
    heat = sess.run(None, {name: x})[0][0]
    pts, confs = [], []
    for c in range(4):
        y, xc = np.unravel_index(np.argmax(heat[c]), heat[c].shape)
        confs.append(float(heat[c][y, xc]))
        pts.append((xc * (w / float(heat.shape[2])), y * (h / float(heat.shape[1]))))
    conf = float(min(confs))
    if conf < min_conf:
        return None
    quad = np.array(pts, np.float32)
    # walk the corners in angular order around their centre, then start at
    # the top-left one; the sum/difference trick mislabels rotated pages
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    quad = quad[np.argsort(angles)]
    quad = np.roll(quad, -int(np.argmin(quad.sum(axis=1))), axis=0)
    tol = 0.02 * max(h, w)
    if np.linalg.norm(quad[0] - quad[1]) < tol or np.linalg.norm(quad[1] - quad[2]) < tol:
        return None
    area = abs(cv2.contourArea(quad)) / float(h * w)
    if area < 0.05:
        return None
    return quad, conf
