"""Image IO helpers: EXIF-correct loading, high-quality resizing, output naming."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _jpeg_size(buf: np.ndarray) -> tuple[int, int] | None:
    """Width/height from the JPEG SOF marker without decoding."""
    if buf.size < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    i, n = 2, buf.size
    while i + 9 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seglen = int(buf[i + 2]) << 8 | int(buf[i + 3])
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = int(buf[i + 5]) << 8 | int(buf[i + 6])
            w = int(buf[i + 7]) << 8 | int(buf[i + 8])
            return w, h
        i += 2 + seglen
    return None


def load_image(path: str | Path, min_long_side: int = 0) -> np.ndarray:
    """Load an image as BGR uint8, applying EXIF orientation.

    With `min_long_side`, very large JPEGs are decoded at half or quarter scale
    (libjpeg's DCT scaling) when the result still exceeds that size — decoding a
    64 MP photo at full resolution costs ~2 s and is thrown away by the resize.
    """
    path = str(path)
    with open(path, "rb") as fh:
        buf = np.frombuffer(fh.read(), dtype=np.uint8)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    if min_long_side:
        size = _jpeg_size(buf)
        if size:
            long_side = max(size)
            if long_side >= min_long_side * 4:
                flags |= cv2.IMREAD_REDUCED_COLOR_4
            elif long_side >= min_long_side * 2:
                flags |= cv2.IMREAD_REDUCED_COLOR_2
    img = cv2.imdecode(buf, flags)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    return fix_orientation(buf, img, path)


def fix_orientation(buf: np.ndarray, img: np.ndarray, path: str = "") -> np.ndarray:
    """Apply the EXIF orientation tag to an already decoded BGR image."""
    try:
        import PIL.Image
        import io

        with PIL.Image.open(io.BytesIO(buf.tobytes())) as pim:
            exif = pim.getexif()
        orientation = int(exif.get(0x0112, 1))
    except Exception:
        orientation = 1
    if orientation == 1:
        return img
    # orientation -> cv2 transform
    if orientation == 2:
        return cv2.flip(img, 1)
    if orientation == 3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if orientation == 4:
        return cv2.flip(img, 0)
    if orientation == 5:
        return cv2.transpose(img)
    if orientation == 6:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:
        return cv2.flip(cv2.transpose(img), -1)
    if orientation == 8:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def resize_long_side(img: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """Downscale so the long side equals long_side; returns (image, scale)."""
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img, 1.0
    scale = long_side / max(h, w)
    out = cv2.resize(
        img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return out, scale


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def imwrite(path: str | Path, img: np.ndarray, quality: int = 95) -> None:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif suffix == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif suffix in (".tif", ".tiff"):
        params = []
    else:
        params = []
    ok = cv2.imwrite(path, img, params)
    if not ok:
        raise IOError(f"failed to write {path}")
