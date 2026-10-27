"""Decide which flattening strategy a page needs.

The pipeline switches on the labels returned here:

=============== ==========================================================
``single_text`` perspective correction, then the word-line grid unwarp
``single_text_curled`` perspective, model dewarp, then the word-line grid
``spread_book`` split at the gutter first, then per-page homography
``spread_newspaper`` no split; model dewarp and per-column line unwarp
``graphic`` perspective only; a line grid would bend the picture
``mixed`` perspective and model dewarp; line unwarp only inside text blocks
``not_page`` no readable document in the frame
=============== ==========================================================

The decision is made by Clef (Cloudflare's decision model served by
``llama-server``) when a working endpoint URL is supplied, and by a local,
model-free classifier otherwise.  The local classifier is deliberately cheap:
one downscaled pass, ink density, connected-component size/alignment, column
runs per text band and a gutter test.  It is a fallback, not a replacement —
it cannot separate some of the labels:

* ``single_text`` vs ``single_text_curled``: paper curl that survives the crop
  is hard to measure; the fallback only emits ``single_text_curled`` when the
  residual line bow across text bands is clearly high.
* ``spread_newspaper`` vs a multi-column magazine page: the fallback uses the
  number of column runs per text band; a single-column-looking newspaper page
  is reported as ``single_text``.
* ``not_page`` vs ``graphic``: a full-bleed photograph has no page semantics
  to measure, so the fallback reports ``graphic`` (never rejects it) unless
  the frame is blank or crumpled with no readable content at all.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .router import CONTENT_ROUTES, route_content_clef

# the labels the pipeline switches on, in the order they are documented
PAGE_TYPES = tuple(CONTENT_ROUTES)

# analysis resolution: enough for line/column structure, small enough to stay
# well under the cost of one JPEG decode
_SIDE = 640


@dataclass
class ContentRoute:
    kind: str            # one of PAGE_TYPES
    confidence: float
    source: str          # clef | heuristic


# ------------------------------------------------------------------ measure
def _measure(img_bgr: np.ndarray, side: int = _SIDE) -> dict:
    """Make the measurements the local classifier needs, in one pass.

    Coordinates are work-frame pixels (the image resized so its long side is
    ``side``).  Nothing here assumes a language or a page orientation.
    """
    h, w = img_bgr.shape[:2]
    s = side / float(max(h, w))
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (max(2, int(w * s)), max(2, int(h * s))),
                             interpolation=cv2.INTER_AREA)
    h, w = img_bgr.shape[:2]
    page = float(w * h)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    # ink: darker than a heavily blurred local background, so shading and
    # paper colour drop out and only printed/written strokes remain
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=max(3.0, w / 40.0),
                          sigmaY=max(3.0, w / 40.0))
    ink = ((bg - g) > 24).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ink_frac = float(ink.mean())

    # connected components: text is many small pieces sitting on shared rows
    n, _lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    align = bincover = 0.0
    if n > 1:
        areas = st[1:, cv2.CC_STAT_AREA].astype(np.float64)
        hs = st[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        ws = st[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        cys = cen[1:, 1]
        sel = (areas >= 3) & (hs <= 0.045 * h) & (ws <= 0.5 * w)
        if sel.any():
            yy = cys[sel]
            bins = (yy / 2.0).astype(int)
            vals, counts = np.unique(bins, return_counts=True)
            good = counts >= 4
            per = np.zeros(len(yy), bool)
            for b in vals[good]:
                per |= (bins == b)
            align = float(per.mean())           # glyphs that sit on a row
            bincover = float(good.sum() / max(len(vals), 1))

    # horizontal closing turns words into line fragments
    kw = max(2, int(w * 0.035))
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, kw), np.uint8))
    n2, _l2, s2, _c2 = cv2.connectedComponentsWithStats(closed, 8)
    line_ink = 0.0
    if n2 > 1:
        a2 = s2[1:, cv2.CC_STAT_AREA].astype(np.float64)
        w2 = s2[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        h2 = s2[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        tot2 = max(float(a2.sum()), 1.0)
        lm = (w2 > 0.10 * w) & (h2 < 0.06 * h) & (w2 > 2.0 * h2) & (h2 >= 2)
        line_ink = float(a2[lm].sum() / tot2)

    # text rows: bands in the row profile; also their column runs (newspaper
    # columns) and how much each band bends across the page (curled paper)
    nb = 0
    med_runs = 0.0
    band_bow = 0.0
    ys = np.where(ink.any(axis=1))[0]
    xs = np.where(ink.any(axis=0))[0]
    if len(ys) > 4 and len(xs) > 4:
        crop = ink[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
        p = crop.mean(axis=1).astype(np.float32)
        ps = cv2.GaussianBlur(p.reshape(-1, 1), (1, 0), 0, sigmaY=1.2).ravel()
        mx = float(ps.max())
        if mx > 1e-6:
            active = ps > max(0.02, 0.12 * mx)
            d = np.diff(np.concatenate([[0], active.astype(int), [0]]))
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
            bands = []
            for a, b in zip(starts, ends):
                if bands and a - bands[-1][1] <= 2:
                    bands[-1] = (bands[-1][0], b)
                else:
                    bands.append((a, b))
            nb = len(bands)
            if bands:
                cl = cv2.morphologyEx(crop, cv2.MORPH_CLOSE,
                                      np.ones((1, kw), np.uint8))
                counts, bows = [], []
                for a, b in bands:
                    if b - a < 3:
                        continue
                    band = cl[a:b]
                    q = band.mean(axis=0).astype(np.float32)
                    qs = cv2.GaussianBlur(q.reshape(-1, 1), (1, 0), 0,
                                          sigmaY=1.0).ravel()
                    mq = float(qs.max())
                    if mq > 1e-6:
                        ac = qs > 0.15 * mq
                        dd = np.diff(np.concatenate([[0], ac.astype(int), [0]]))
                        r0, r1 = np.where(dd == 1)[0], np.where(dd == -1)[0]
                        runs = [(x, y) for x, y in zip(r0, r1)
                                if y - x >= 0.02 * crop.shape[1]]
                        merged = []
                        for x, y in runs:
                            if merged and x - merged[-1][1] < 0.012 * crop.shape[1]:
                                merged[-1] = (merged[-1][0], y)
                            else:
                                merged.append((x, y))
                        counts.append(len(merged))
                    # bow: band centre vs the two sides, rotation cancels
                    src = crop[a:b]
                    vals = []
                    for s0, s1 in ((0, max(4, crop.shape[1] // 3)),
                                   (crop.shape[1] // 3, 2 * crop.shape[1] // 3),
                                   (2 * crop.shape[1] // 3, crop.shape[1])):
                        part = src[:, s0:s1]
                        npx = float(part.sum())
                        if npx < 8:
                            vals = []
                            break
                        vals.append(float((part.sum(axis=1) *
                                           np.arange(b - a)).sum() / npx))
                    if len(vals) == 3:
                        bows.append(abs(vals[1] - 0.5 * (vals[0] + vals[2])) /
                                    crop.shape[0])
                med_runs = float(np.median(counts)) if counts else 0.0
                band_bow = float(np.median(bows)) if bows else 0.0

    # picture-like regions: large areas that are not white paper.  Text
    # strokes disappear under the 3x3 opening; illustrations and photos do not.
    nonwhite = ((val < 210) | (sat > 35)).astype(np.uint8)
    opened = cv2.morphologyEx(nonwhite, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    n3, _l3, s3, _c3 = cv2.connectedComponentsWithStats(opened, 8)
    region = 0.0
    if n3 > 1:
        a3 = s3[1:, cv2.CC_STAT_AREA].astype(np.float64)
        region = float(a3[a3 > page * 0.01].sum() / page)

    # texture and surface colour
    m = cv2.blur(g, (5, 5))
    m2 = cv2.blur(g * g, (5, 5))
    sd = np.sqrt(np.maximum(m2 - m * m, 0))
    hi_tex = float((sd > 25).mean())
    sat_hi = float((sat > 80).mean())

    # gutter: a narrow, persistent dark valley near the centre of the page.
    # Pictures also have dark areas, so require the valley to run down most of
    # the page rather than a broad dark region.
    dark = 255.0 - g
    prof = dark.mean(axis=0)
    sm = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), 0,
                          sigmaY=max(2.0, w * 0.01)).ravel()
    lo, hi = int(w * 0.32), int(w * 0.68)
    gutter_v = gutter_vp = 0.0
    if hi > lo + 3:
        c = lo + int(np.argmax(sm[lo:hi]))
        off = max(3, int(w * 0.05))
        neigh = []
        if c - off > 0:
            neigh.append(dark[:, c - off:c - off // 2])
        if c + off < w:
            neigh.append(dark[:, c + off // 2:c + off])
        if neigh:
            base = np.median(np.concatenate(neigh, axis=1), axis=1)
            outside = np.concatenate([sm[:lo], sm[hi:]])
            base_mean = float(np.median(outside)) if len(outside) else float(np.median(sm))
            gutter_v = max(0.0, float(sm[c] - base_mean) / 255.0)
            gutter_vp = float(((base - dark[:, c]) > 12).mean())

    return {
        "ar": w / float(h),
        "ink": ink_frac,
        "hi_tex": hi_tex,
        "sat_hi": sat_hi,
        "align": align,
        "bincover": bincover,
        "line_ink": line_ink,
        "nb": nb,
        "med_runs": med_runs,
        "band_bow": band_bow,
        "region": region,
        "gutter_v": gutter_v,
        "gutter_vp": gutter_vp,
    }


# ---------------------------------------------------------------- classify
def _text_present(m: dict) -> bool:
    return (
        (m["nb"] >= 6 and m["hi_tex"] >= 0.10 and m["region"] < 0.85)
        or (m["align"] >= 0.70 and m["bincover"] >= 0.30)
        or m["line_ink"] >= 0.40
    )


def classify_local(img_bgr: np.ndarray) -> ContentRoute:
    """Model-free page-type decision from the image alone.

    Thresholds were calibrated on 320 scans from the 7,489-page corpus:
    ink and texture for the no-content test, row alignment and line fragments
    for text, non-white component area for pictures, and gutter persistence
    for book spreads.
    """
    m = _measure(img_bgr)
    text = _text_present(m)

    # 1. no readable document: near-blank or crumpled paper with no ink
    #    structure at all.  Deliberately conservative — rejecting a real page
    #    costs more than letting a picture through as `graphic`.
    if ((m["ink"] < 0.010 and m["hi_tex"] < 0.10)
            or (m["hi_tex"] < 0.085 and m["line_ink"] < 0.20
                and m["align"] < 0.55)):
        return ContentRoute("not_page", 0.75, "heuristic")

    # 2. two facing pages with a persistent central gutter.  Only landscape
    #    pages: a dark spine shadow on a single portrait page must not be
    #    mistaken for a gutter, because splitting there would cut the page.
    spread = (m["ar"] > 1.25 and m["region"] < 0.68 and (
        m["gutter_v"] > 0.20
        or (m["gutter_v"] > 0.14 and m["gutter_vp"] > 0.10)
        or (m["ar"] > 1.7 and m["gutter_vp"] > 0.12 and text)))
    if spread:
        conf = min(0.92, 0.55 + 2.0 * m["gutter_v"])
        return ContentRoute("spread_book", round(conf, 2), "heuristic")

    # 3. a wide sheet whose text is laid out in several columns: no split
    if (text and m["med_runs"] >= 2.5 and m["align"] >= 0.70
            and m["region"] < 0.45):
        conf = min(0.90, 0.50 + 0.10 * m["med_runs"])
        return ContentRoute("spread_newspaper", round(conf, 2), "heuristic")

    picture = m["region"] >= 0.30 or m["sat_hi"] >= 0.25
    if text:
        if picture:
            return ContentRoute("mixed", round(min(0.90, 0.55 + 0.25 * m["region"]), 2),
                                "heuristic")
        if m["band_bow"] > 0.012:
            return ContentRoute("single_text_curled",
                                round(min(0.85, 0.50 + 15.0 * m["band_bow"]), 2),
                                "heuristic")
        strength = max(m["align"], min(1.0, m["nb"] / 20.0),
                       min(1.0, m["line_ink"]))
        return ContentRoute("single_text", round(min(0.92, 0.55 + 0.35 * strength), 2),
                            "heuristic")

    return ContentRoute("graphic", round(min(0.92, 0.55 + 0.35 * m["region"]), 2),
                        "heuristic")


def classify(img_bgr: np.ndarray, url: str | None = None) -> ContentRoute:
    """Return the page type, using Clef when ``url`` answers and is valid.

    ``url`` is the base URL of a llama-server serving Clef (for example
    ``http://127.0.0.1:8099``).  Any connection error, timeout, or label
    outside :data:`PAGE_TYPES` falls back to :func:`classify_local`.
    """
    if url:
        picked = route_content_clef(img_bgr, url)
        if picked is not None:
            kind, prob = picked
            return ContentRoute(kind, float(prob), "clef")
    return classify_local(img_bgr)
