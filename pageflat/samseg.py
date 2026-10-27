"""Optional SAM-based page segmentation.

DocTr's U2NETP page model treats every sheet of paper in the frame as one class,
so piles and overlapping sheets merge into a single mask. SAM, prompted with a
point on the page of interest, follows the boundary of the *topmost* sheet and
gives a much tighter mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


class SamPageSegmenter:
    def __init__(self, checkpoint: str | Path, device: str = "cuda",
                 model_type: str | None = None):
        if model_type is None:
            model_type = "vit_t" if "mobile" in Path(checkpoint).name.lower() else "vit_b"
        if model_type == "vit_t":
            from mobile_sam import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device).eval()
        self.predictor = SamPredictor(sam)
        self.device = device

    def mask(self, img_bgr: np.ndarray, points: Sequence[Sequence[int]],
             box: Optional[Sequence[int]] = None, prior: Optional[np.ndarray] = None,
             max_side: int = 1024) -> Optional[np.ndarray]:
        """Return a page mask at the input resolution, or None."""
        h, w = img_bgr.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else img_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        pts = np.array([[p[0] * scale, p[1] * scale] for p in points], np.float32)
        labels = np.ones(len(pts), np.int32)
        kwargs = {}
        if box is not None:
            kwargs["box"] = np.array([box[0] * scale, box[1] * scale,
                                      box[2] * scale, box[3] * scale], np.float32)
        masks, scores, _ = self.predictor.predict(point_coords=pts, point_labels=labels,
                                                  multimask_output=True, **kwargs)
        if masks is None or len(masks) == 0:
            return None

        prior_small = None
        if prior is not None:
            prior_small = cv2.resize(prior, (small.shape[1], small.shape[0])) > 0.5

        # Prefer the smallest confident object at the prompt point: when a sheet
        # of paper sits on another surface, SAM's top-scoring mask is often the
        # whole background, while the pointed-at page is a smaller mask.
        candidates = []
        for m, s in zip(masks, scores):
            area = float(m.mean())
            if area < 0.01 or float(s) < 0.45:
                continue
            if prior_small is not None:
                contain = (m & prior_small).sum() / max(1, m.sum())
                if contain < 0.7:
                    continue
            if prior_small is not None and area < 0.5 * float(prior_small.mean()):
                continue
            candidates.append((area, float(s), m))
        if not candidates:
            best = max(zip(masks, scores), key=lambda t: float(t[1]))[0]
        else:
            candidates.sort(key=lambda t: t[0])
            best = candidates[0][2]
        out = cv2.resize(best.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return out.astype(np.float32)
