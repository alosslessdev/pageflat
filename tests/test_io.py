import numpy as np
import PIL.Image

from pageflat.io import load_image, resize_long_side
from pageflat.quality import looks_like_page


def _make_jpeg(path, w, h, orientation=1):
    img = PIL.Image.new("RGB", (w, h), (200, 200, 200))
    exif = img.getexif()
    exif[0x0112] = orientation
    img.save(path, exif=exif.tobytes())


def test_load_image_applies_exif_orientation(tmp_path):
    path = tmp_path / "rot.jpg"
    _make_jpeg(path, 60, 20, orientation=6)  # 90 deg clockwise
    img = load_image(path)
    assert img.shape[:2] == (60, 20)


def test_resize_long_side_reports_scale():
    img = np.zeros((1000, 2000, 3), np.uint8)
    out, scale = resize_long_side(img, 1000)
    assert out.shape[:2] == (500, 1000)
    assert abs(scale - 0.5) < 1e-9
    same, scale = resize_long_side(np.zeros((100, 200, 3), np.uint8), 1000)
    assert scale == 1.0 and same.shape[:2] == (100, 200)


def test_page_gate_accepts_text_and_rejects_blank_surface():
    import cv2

    blank = np.full((900, 700, 3), 240, np.uint8)
    ok_blank, _ = looks_like_page(blank)
    assert not ok_blank

    text = blank.copy()
    for y in range(60, 840, 26):
        cv2.putText(text, "lorem ipsum dolor sit amet", (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    ok_text, metrics = looks_like_page(text)
    assert ok_text and metrics["ink_fraction"] > 0.01
