#!/usr/bin/env python3
"""Compare page-flow predictors against the only ground truth pages carry: their page numbers.

Pairs where B follows A numerically are "same document"; pairs where a page
numbered 1 follows a higher number are "different document". Each predictor is
scored on those pairs.

    python scripts/eval_flow.py scans/ocr --llm http://127.0.0.1:8100 \
        --embed-url http://127.0.0.1:8100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from group_documents import (cosine, embed_text, header_text,  # noqa: E402
                             llm_same_document, load_pages, similarity)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ocr_dir")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    pages = load_pages(Path(args.ocr_dir))
    pages.sort(key=lambda p: p["captured"] or 0)
    pairs = []
    for a, b in zip(pages, pages[1:]):
        if a["number"] is None or b["number"] is None:
            continue
        if b["number"] == a["number"] + 1:
            pairs.append((a, b, True))
        elif b["number"] == 1 and a["number"] > 1:
            pairs.append((a, b, False))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"{len(pairs)} labelled pairs", file=sys.stderr)
    if not pairs:
        return 1

    stats = {"header": [0, 0], "embedding": [0, 0], "llm": [0, 0]}
    embeds = {}
    if args.embed_url:
        for pa, pb, _same in pairs:
            for p in (pa, pb):
                if p["file"] not in embeds:
                    text = (p.get("head", "") + " " + p.get("tail", "")).strip()
                    embeds[p["file"]] = embed_text(args.embed_url, text) if text else None

    for a, b, same in pairs:
        pred = similarity(a["header"], b["header"]) >= 0.25
        stats["header"][0] += int(pred == same)
        stats["header"][1] += 1

        if args.embed_url:
            ea, eb = embeds.get(a["file"]), embeds.get(b["file"])
            if ea and eb:
                pred = cosine(ea, eb) >= 0.72
                stats["embedding"][0] += int(pred == same)
                stats["embedding"][1] += 1

        if args.llm:
            decision = llm_same_document(args.llm, a, b)
            if decision is not None:
                stats["llm"][0] += int(decision == same)
                stats["llm"][1] += 1

    for name, (ok, n) in stats.items():
        if n:
            print(f"{name:10s} {ok}/{n} = {ok / n:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
