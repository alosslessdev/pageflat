#!/usr/bin/env python3
"""Classify photos as "page" or "other" with Cloudflare Clef (decision model) on llama-server.

    llama-server -m clef-flash.gguf --mmproj mmproj.gguf -ngl 99 --port 8099 --no-webui

    python scripts/classify_pages.py inventory.tsv --url http://127.0.0.1:8099 \
        --out clef_pages.jsonl --workers 4

The inventory is a TSV whose 4th column is the image path (the format written by
find_pages.py's raw scan) or a plain list of paths.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import requests

QUESTION = {
    "page": {
        "type": "choice",
        "instructions": (
            "Look at this photograph. Is it primarily a photograph of a sheet of "
            "paper — a book page, notebook page, printed document, newspaper page, "
            "form, receipt, poster or loose note — rather than a scene, a person, "
            "an object, a screen or fabric?"
        ),
        "criteria": {
            "page": "the frame is dominated by a sheet of paper with text or printed content",
            "other": "anything else: people, rooms, objects, screens, scenery, textile, food",
        },
    }
}


def encode(path: str, side: int = 320) -> str | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = side / float(max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def classify(url: str, path: str, timeout: float = 120.0, side: int = 320) -> dict:
    data = encode(path, side)
    if data is None:
        return {"path": path, "error": "decode"}
    body = {"state": "Classify the photograph.",
            "images": [data],
            "questions": QUESTION}
    try:
        r = requests.post(url.rstrip("/") + "/v1/systemone", json=body, timeout=timeout)
        r.raise_for_status()
        ans = r.json()["answers"]["page"]
        # choice answers carry per-option probabilities
        probs = ans.get("probabilities") or ans.get("options") or {}
        return {"path": path, "page": float(probs.get("page", 0.0)),
                "answer": ans.get("answer") or ans.get("choice")}
    except Exception as exc:  # noqa: BLE001
        return {"path": path, "error": repr(exc)[:200]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--side", type=int, default=320)
    args = ap.parse_args(argv)

    import re
    paths = []
    for line in Path(args.inventory).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        found = re.findall(r"/[^\t]*?\.(?:jpe?g|png)", line, flags=re.I)
        if found:
            paths.append(found[-1])
        elif "/" in line:
            paths.append(line)
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} images", file=sys.stderr)

    done = 0
    t0 = time.time()
    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(lambda p: classify(args.url, p, side=args.side), paths):
            out.write(json.dumps(res) + "\n")
            out.flush()
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                print(f"{done}/{len(paths)}  {rate:.1f}/s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
