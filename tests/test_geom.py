import numpy as np
import cv2

from pageflat import geom, illum, spine


def test_order_quad_puts_corners_in_tl_tr_br_bl_order():
    pts = np.array([[100, 100], [10, 10], [100, 10], [10, 100]], np.float32)
    ordered = geom.order_quad(pts)
    assert ordered.tolist() == [[10, 10], [100, 10], [100, 100], [10, 100]]


def test_mask_to_quad_finds_rotated_page():
    mask = np.zeros((600, 800), np.float32)
    quad = np.array([[200, 60], [560, 150], [520, 540], [160, 450]], np.int32)
    cv2.fillPoly(mask, [quad], 1.0)
    page = geom.mask_to_quad(mask)
    assert page is not None
    assert page.area_frac > 0.15
    assert page.rectangularity > 0.8
    # corners recovered within a few pixels
    for expected in quad:
        assert np.min(np.linalg.norm(page.points - expected, axis=1)) < 12


def test_warp_quad_returns_requested_rectangle():
    img = np.zeros((400, 600, 3), np.uint8)
    img[:] = (30, 60, 90)
    quad = np.array([[50, 50], [550, 70], [540, 350], [40, 340]], np.float32)
    out = geom.warp_quad(img, quad, out_size=(300, 200))
    assert out.shape[:2] == (200, 300)
    assert np.allclose(out.reshape(-1, 3).mean(axis=0), (30, 60, 90), atol=2)


def test_illumination_flattening_removes_vignette():
    h, w = 200, 300
    yy, xx = np.mgrid[0:h, 0:w]
    vignette = 1.0 - 0.25 * ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 4
    img = np.clip(vignette[:, :, None] * 235, 0, 255).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    flat = illum.flatten_illumination(img, white=245.0)
    corners = [flat[:20, :20].mean(), flat[:20, -20:].mean(),
               flat[-20:, :20].mean(), flat[-20:, -20:].mean()]
    assert max(corners) - min(corners) < 12


def test_spread_is_split_at_the_gutter():
    page = np.full((400, 800, 3), 245, np.uint8)
    for y in range(40, 360, 24):
        cv2.line(page, (30, y), (370, y), (40, 40, 40), 2)
        cv2.line(page, (430, y), (760, y), (40, 40, 40), 2)
    page[:, 395:405] = 90  # gutter shadow
    pages = spine.split_spread(page)
    assert len(pages) == 2
    assert pages[0].shape[1] < 420 and pages[1].shape[1] < 420
