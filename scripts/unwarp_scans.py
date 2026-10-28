#!/usr/bin/env python3
"""Straighten the scans whose text measurably improves, and only those.

Runs the word-line grid unwarp over every scan, keeps the result only when
tesseract mean word confidence rises by `--min-gain`, and rewrites the scan in
place when it does. Writes the list of replaced pages so OCR and the PDFs can
be refreshed for exactly those.

    python scripts/unwarp_scans.py scans/ --workers 8 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pageflat import lines  # noqa: E402
from pageflat.pipeline import ocr_quality  # noqa: E402

_MODELS = {}


def _dewarpers(names: str):
    """Load the requested dewarping models once per worker."""
    if names in _MODELS:
        return _MODELS[names]
    loaded = {}
    if "doctr" in names:
        from pageflat.models import DocTrPipeline
        loaded["doctr"] = DocTrPipeline(device="cuda")
    if "uvdoc" in names:
        from pageflat.models import UVDocDewarper
        loaded["uvdoc"] = UVDocDewarper()
    _MODELS[names] = loaded
    return loaded


def _init() -> None:
    # tesseract grabs every core by default; six of those on one box thrash
    os.environ["OMP_THREAD_LIMIT"] = "1"
    cv2.setNumThreads(1)


def one(job: tuple[str, float, bool, str]) -> dict:
    path_s, min_gain, apply, models = job
    path = Path(path_s)
    img = cv2.imread(str(path))
    if img is None:
        return {"page": path.name, "status": "unreadable"}
    try:
        grid, info = lines.unwarp(img)
    except Exception as exc:  # noqa: BLE001
        return {"page": path.name, "status": "error", "reason": repr(exc)[:120]}
    candidates: list[tuple[str, "object"]] = [("grid", grid if info.get("corrected") else img)]
    if models:
        for name, model in _dewarpers(models).items():
            try:
                candidates.append((name, model.rectify(img)))
            except Exception:  # noqa: BLE001 - a failed model is just not a candidate
                continue
    if all(np.array_equal(c, img) for _, c in candidates):
        return {"page": path.name, "status": "skipped", "reason": info.get("rejected", "")}
    before = ocr_quality(img)
    scored = [(name, cand, ocr_quality(cand)) for name, cand in candidates if cand is not img]
    scored.sort(key=lambda x: -x[2])
    best_name, best_img, best_conf = scored[0]
    if best_conf <= before + min_gain:
        return {"page": path.name, "status": "rejected", "before": round(before, 1),
                "after": round(best_conf, 1), "candidate": best_name}
    if apply:
        cv2.imwrite(str(path), best_img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return {"page": path.name, "status": "rewritten", "before": round(before, 1),
            "after": round(best_conf, 1), "candidate": best_name,
            "bow": info.get("bow"), "bow_after": info.get("bow_after")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=0.5)
    ap.add_argument("--models", default="doctr,uvdoc",
                    help="dewarping models to try as candidates ('' disables)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--journal", default=None, help="defaults to <scans>/unwarp_journal.jsonl")
    ap.add_argument("--changed-out", default=None)
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    journal = Path(args.journal) if args.journal else scans / "unwarp_journal.jsonl"
    changed_path = Path(args.changed_out) if args.changed_out else scans / "unwarped_pages.txt"
    pages = sorted(scans.glob("*.png"))
    if args.limit:
        pages = pages[:args.limit]
    jobs = [(str(p), args.min_gain, not args.dry_run, args.models) for p in pages]

    rewritten: list[str] = []
    tally: dict[str, int] = {}
    t0 = time.time()
    with open(journal, "w") as fh, ProcessPoolExecutor(max_workers=args.workers,
                                                      initializer=_init) as pool:
        for i, row in enumerate(pool.map(one, jobs, chunksize=4), 1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            status = row["status"]
            tally[status] = tally.get(status, 0) + 1
            if status == "rewritten":
                rewritten.append(row["page"])
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} {tally}", file=sys.stderr, flush=True)
    if not args.dry_run:
        changed_path.write_text("\n".join(rewritten) + "\n")
    print(f"done in {time.time() - t0:.0f}s: {tally}")
    print(f"{len(rewritten)} pages rewritten -> {changed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
