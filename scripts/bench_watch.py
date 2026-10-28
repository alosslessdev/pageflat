#!/usr/bin/env python3
"""Run the benchmark whenever the repository changes.

The 30 frozen pages in `benchmark.json` are the measure. This watcher polls
the repository's HEAD; every time it moves, it measures the pages again and
appends the result, tagged with the commit, to `benchmark_log.jsonl`. A change
that lowers drift or raises sharpness/confidence is a win; one that does the
opposite is caught in minutes instead of at the end of a rebuild.

    python scripts/bench_watch.py scans/ [--period 60]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark import run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--period", type=float, default=60.0)
    args = ap.parse_args(argv)
    scans = Path(args.scans)
    bench = json.loads((scans / "benchmark.json").read_text())["pages"]
    log = scans / "benchmark_log.jsonl"
    last = None
    while True:
        now = head()
        if now and now != last:
            result = run(scans, bench)
            result["commit"] = now[:10]
            s = result["summary"]
            print(f"{s['at']} {now[:10]} drift {s['drift_median']} fail "
                  f"{s['drift_fail_pct']}% sharp {s['sharp_median']} conf {s['conf_mean']}",
                  flush=True)
            with open(log, "a") as fh:
                fh.write(json.dumps(result) + "\n")
            last = now
        time.sleep(args.period)


if __name__ == "__main__":
    raise SystemExit(main())
