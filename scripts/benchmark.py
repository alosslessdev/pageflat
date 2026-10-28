#!/usr/bin/env python3
"""The benchmark: a fixed set of pages, measured the same way every time.

Thirty pages chosen once (seed 7) stand in for the corpus. Each run records,
for every page:

* **drift** — how far its paragraph blocks wander from a vertical edge, as a
  fraction of the page width (the "blocks are rectangles" rule);
* **sharp** — row-ink sharpness, which falls with paper curl;
* **conf**   — tesseract mean word confidence;
* **bow**    — median word-line bow from the OCR boxes.

`--watch` repeats the run every hour and appends one JSON line per hour to
`benchmark_log.jsonl`, so a change can be judged by whether the aggregate
moved, not by whether one page looked better.

    python scripts/benchmark.py scans/ --make            # freeze the page list
    python scripts/benchmark.py scans/                   # one run
    python scripts/benchmark.py scans/ --watch --hours 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pageflat.blocks import load_lines, page_rect_error  # noqa: E402
from pageflat.pipeline import ocr_quality  # noqa: E402
from pageflat.sharp import row_sharpness  # noqa: E402


def measure(scans: Path, page: str) -> dict | None:
    img = cv2.imread(str(scans / page))
    if img is None:
        return None
    row = {"page": page, "sharp": round(row_sharpness(img), 1)}
    j = scans / "ocr" / (Path(page).stem + ".json")
    if j.exists():
        lines, width, height = load_lines(j)
        if len(lines) >= 8:
            r = page_rect_error(lines, scale=1.0)
            if r["blocks"]:
                row["drift"] = round(r["error"] / width, 4)
                sx = img.shape[1] / width
                row["bow"] = round(float(np.median([l[3] - l[1] for l in lines])) * sx, 1)
    row["conf"] = round(ocr_quality(img), 1)
    return row


def run(scans: Path, pages: list[str]) -> dict:
    rows = [r for p in pages if (r := measure(scans, p))]
    drift = [r["drift"] for r in rows if "drift" in r]
    sharp = [r["sharp"] for r in rows]
    conf = [r["conf"] for r in rows]
    summary = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pages": len(rows),
        "drift_median": round(float(np.median(drift)), 4) if drift else None,
        "drift_fail_pct": round(100.0 * sum(1 for d in drift if d > 0.03) / max(1, len(drift)), 1),
        "sharp_median": round(float(np.median(sharp)), 1),
        "conf_mean": round(float(np.mean(conf)), 1),
    }
    return {"summary": summary, "pages": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--bench", default=None, help="defaults to <scans>/benchmark.json")
    ap.add_argument("--log", default=None, help="defaults to <scans>/benchmark_log.jsonl")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--period", type=float, default=3600.0)
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    bench_path = Path(args.bench) if args.bench else scans / "benchmark.json"
    log_path = Path(args.log) if args.log else scans / "benchmark_log.jsonl"

    if args.make:
        pages = sorted(p.name for p in scans.glob("*.png"))
        random.seed(7)
        chosen = random.sample(pages, min(30, len(pages)))
        bench_path.write_text(json.dumps({"pages": chosen}, indent=1))
        print(f"benchmark frozen: {len(chosen)} pages -> {bench_path}")
        return 0

    if not bench_path.exists():
        print("no benchmark; run with --make first", file=sys.stderr)
        return 2
    pages = json.loads(bench_path.read_text())["pages"]

    rounds = int(args.hours * 3600 / args.period) + 1 if args.watch else 1
    for i in range(rounds):
        result = run(scans, pages)
        s = result["summary"]
        print(f"{s['at']}  pages {s['pages']}  drift {s['drift_median']}  "
              f"fail {s['drift_fail_pct']}%  sharp {s['sharp_median']}  conf {s['conf_mean']}")
        with open(log_path, "a") as fh:
            fh.write(json.dumps(result) + "\n")
        if i + 1 < rounds:
            time.sleep(args.period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
