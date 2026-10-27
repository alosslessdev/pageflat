"""Route each photo to a segmentation strategy with a decision model.

Running SAM on every photo costs ~1-2 s and can hurt: on a page that already
fills the frame the DocTr mask is better. A decision model is a cheap way to
pick the strategy per image — and to throw out non-pages before any pixel work.

The same endpoint also answers a second question: which flattening strategy
the page needs (``single_text``, ``single_text_curled``, ``spread_book``,
``spread_newspaper``, ``graphic``, ``mixed``, ``not_page``).  See
:mod:`pageflat.content` for the classifier and its local fallback.

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

# Content routing: which flattening strategy this page needs.  The pipeline
# switches on these labels, so each one names a strategy, not just a look.
CONTENT_ROUTES = {
    "single_text": "one page, mostly printed text; the paper lies flat after perspective correction",
    "single_text_curled": "one page, mostly text, but the paper is visibly bowed or curled (a book page near the spine, a crumpled sheet)",
    "spread_book": "an open book or magazine showing two facing pages that meet at a visible gutter",
    "spread_newspaper": "one wide newspaper sheet with several text columns; there is no gutter to split at",
    "graphic": "photographs, drawings, maps or free-form layout dominate; a text-line grid would corrupt it",
    "mixed": "substantial text and substantial graphic content on the same page",
    "not_page": "no document in the frame: an object, a person, scenery, or a surface without readable content",
}

_CONTENT_QUESTION = {
    "content": {
        "type": "choice",
        "instructions": (
            "Which page type is this? Choose exactly one. Answer for the page "
            "content, not for the quality of the photograph."
        ),
        "criteria": CONTENT_ROUTES,
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


def _systemone(img_bgr: np.ndarray, url: str, state: str,
               questions: dict, timeout: float = 60.0) -> Optional[dict]:
    """POST one image to a llama-server `/v1/systemone` endpoint.

    Returns the parsed ``answers`` mapping, or ``None`` when the endpoint is
    missing, slow or answers with an unexpected shape.  Callers treat ``None``
    as "use the local fallback".
    """
    try:
        import requests

        r = requests.post(url.rstrip("/") + "/v1/systemone", json={
            "state": state,
            "images": [_encode(img_bgr)],
            "questions": questions,
        }, timeout=timeout)
        r.raise_for_status()
        return r.json().get("answers")
    except Exception:
        return None


def _choice(answers: Optional[dict], key: str) -> Optional[tuple[str, float]]:
    if not answers:
        return None
    ans = answers.get(key)
    if not isinstance(ans, dict):
        return None
    probs = ans.get("probabilities")
    if not isinstance(probs, dict):
        opts = ans.get("options")
        probs = opts if isinstance(opts, dict) else {}
    if probs:
        kind, p = max(probs.items(), key=lambda kv: float(kv[1]))
        return str(kind), float(p)
    kind = ans.get("choice")
    if kind is None:
        return None
    return str(kind), float(ans.get("confidence", 0.5))


def route_clef(img_bgr: np.ndarray, url: str, timeout: float = 60.0) -> Optional[Route]:
    answers = _systemone(img_bgr, url,
                         "Route this photograph to the right page-segmentation strategy.",
                         _QUESTION, timeout)
    picked = _choice(answers, "scene")
    if picked is None:
        return None
    kind, p = picked
    return Route(kind, p, "clef")


def route_content_clef(img_bgr: np.ndarray, url: str,
                       timeout: float = 60.0) -> Optional[tuple[str, float]]:
    """Ask Clef which page type this is.

    Returns ``(kind, probability)`` with ``kind`` in ``CONTENT_ROUTES``, or
    ``None`` when the endpoint cannot be reached or answers with a label
    outside the contract (callers then use the local fallback).
    """
    answers = _systemone(img_bgr, url, "Classify the page type.",
                         _CONTENT_QUESTION, timeout)
    picked = _choice(answers, "content")
    if picked is None or picked[0] not in CONTENT_ROUTES:
        return None
    return picked


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


def decide_content(img_bgr: np.ndarray, url: str | None = None):
    """Page type for the flattening strategy (``CONTENT_ROUTES`` keys).

    Delegates to :func:`pageflat.content.classify`; kept here so the pipeline
    has one router module for both decisions.
    """
    from .content import classify

    return classify(img_bgr, url=url)
