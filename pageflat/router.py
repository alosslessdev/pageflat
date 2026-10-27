"""Route each photo to a segmentation strategy with a decision model.

Running SAM on every photo costs ~1-2 s and can hurt: on a page that already
fills the frame the DocTr mask is better. A decision model is a cheap way to
pick the strategy per image — and to throw out non-pages before any pixel work.

The router talks to Cloudflare Clef on llama-server (TypeSafe `/v1/systemone`
API). If no server is configured or it fails, the heuristic in `route_heuristic`
is used instead, so the pipeline never depends on it.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

ROUTES = {
    "single": "one sheet of paper fills most of the frame, edges visible",
    "cluttered": "a sheet of paper on a cluttered background, other papers or objects around it",
    "pile": "several sheets stacked or overlapping, the top one is the target",
    "not_page": "no sheet of paper: people, rooms, objects, screens, scenery, textile",
}

_QUESTION = {
    "scene": {
        "type": "choice",
        "instructions": (
            "What does this photograph show? Answer for the scene, not for the "
            "quality of the photograph."
        ),
        "criteria": ROUTES,
    }
}


@dataclass
class Route:
    kind: str            # single | cluttered | pile | not_page
    confidence: float
    source: str          # clef | heuristic


def _encode(img_bgr: np.ndarray, side: int = 448) -> str:
    h, w = img_bgr.shape[:2]
    s = side / float(max(h, w))
    small = cv2.resize(img_bgr, (max(32, int(w * s)), max(32, int(h * s))),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def route_clef(img_bgr: np.ndarray, url: str, timeout: float = 60.0) -> Optional[Route]:
    try:
        import requests

        r = requests.post(url.rstrip("/") + "/v1/systemone", json={
            "state": "Route this photograph to the right page-segmentation strategy.",
            "images": [_encode(img_bgr)],
            "questions": _QUESTION,
        }, timeout=timeout)
        r.raise_for_status()
        ans = r.json()["answers"]["scene"]
        probs = ans.get("probabilities") or ans.get("options") or {}
        if not probs:
            return None
        kind, p = max(probs.items(), key=lambda kv: kv[1])
        return Route(kind, float(p), "clef")
    except Exception:
        return None


def route_heuristic(img_bgr: np.ndarray, mask: np.ndarray | None = None) -> Route:
    """Cheap fallback: how much of the frame is one flat surface."""
    h, w = img_bgr.shape[:2]
    s = 256.0 / max(h, w)
    small = cv2.resize(img_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    paper = float(((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 140)).mean())
    if paper < 0.25:
        return Route("not_page", 0.6, "heuristic")
    if mask is not None:
        frac = float((mask > 0.5).mean())
        if frac > 0.55:
            return Route("single", 0.6, "heuristic")
        if frac > 0.2:
            return Route("cluttered", 0.6, "heuristic")
        return Route("pile", 0.4, "heuristic")
    return Route("cluttered", 0.3, "heuristic")


def decide(img_bgr: np.ndarray, url: str | None = None,
           mask: np.ndarray | None = None) -> Route:
    if url:
        r = route_clef(img_bgr, url)
        if r is not None:
            return r
    return route_heuristic(img_bgr, mask)
