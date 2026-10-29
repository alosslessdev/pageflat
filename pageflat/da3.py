"""Depth Anything 3 through onnxruntime: relative depth plus the camera pose.

V2 gives a shape with no camera basis, so the tilt of the camera and the curl of
the page cannot be told apart — one of the standing failure modes when the warp
absorbs the tilt into the sheet. V3 predicts the intrinsics and extrinsics along
with the depth, so the page can be measured in camera space instead of guessed
at in image space.

The exported graph takes a stack of views:
    pixel_values (B, N, 3, H, W)  ImageNet-normalised, H/W multiple of 14
and returns
    predicted_depth (B, N, H, W), confidence (B, N, H, W),
    extrinsics (B, N, 3, 4), intrinsics (B, N, 3, 3)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PATCH = 14
_SESSION = None
MODEL = Path(__file__).resolve().parents[2] / "vlm" / "da3" / "onnx" / "model.onnx"


def session():
    global _SESSION
    if _SESSION is None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            _SESSION = ort.InferenceSession(str(MODEL), providers=providers)
        except Exception:  # noqa: BLE001
            _SESSION = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    return _SESSION


def _fit(h: int, w: int, side: int) -> tuple[int, int]:
    s = side / float(max(h, w))
    hh = max(_PATCH, int(round(h * s / _PATCH)) * _PATCH)
    ww = max(_PATCH, int(round(w * s / _PATCH)) * _PATCH)
    return hh, ww


def estimate(img_bgr: np.ndarray, side: int = 518) -> dict:
    """Depth, confidence and pose for one view, at the image's own frame."""
    sess = session()
    h, w = img_bgr.shape[:2]
    hh, ww = _fit(h, w, side)
    small = cv2.resize(img_bgr, (ww, hh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None, None]  # (1,1,3,H,W)
    out = sess.run(None, {"pixel_values": x.astype(np.float32)})
    names = [o.name for o in sess.get_outputs()]
    res = dict(zip(names, out))
    depth = res["predicted_depth"][0, 0]
    depth_full = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return {
        "depth": depth_full,
        "confidence": cv2.resize(res["confidence"][0, 0], (w, h), interpolation=cv2.INTER_LINEAR),
        "intrinsics": res["intrinsics"][0, 0],
        "extrinsics": res["extrinsics"][0, 0],
    }


def surface_depth(img_bgr: np.ndarray, side: int = 518) -> np.ndarray:
    """Depth in the image frame, oriented so that *larger means closer*.

    V3 returns relative depth like V2 (larger = closer on the usual
    Depth-Anything convention); the sign is checked against the image so the
    caller does not have to know which way the export went.
    """
    return estimate(img_bgr, side)["depth"]


if __name__ == "__main__":
    import sys

    path = Path(sys.argv[1])
    im = cv2.imread(str(path))
    r = estimate(im)
    d = r["depth"]
    print("depth", d.shape, "range", round(float(d.min()), 3), round(float(d.max()), 3))
    print("intrinsics\n", np.round(r["intrinsics"], 1))
    print("extrinsics\n", np.round(r["extrinsics"], 3))
