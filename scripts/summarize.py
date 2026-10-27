#!/usr/bin/env python3
"""Aggregate pageflat JSON reports into a markdown summary."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    rows = []
    for pattern in args.reports:
        base = Path(pattern)
        matches = sorted(base.parent.glob(base.name))
        for path in matches or [base]:
            if path.is_file():
                rows.extend(json.loads(path.read_text()))
    if not rows:
        print("no reports")
        return 1

    ok = [r for r in rows if r["status"] == "ok"]
    rejected = [r for r in rows if r["status"] == "rejected"]
    errors = [r for r in rows if r["status"] == "error"]
    pages = sum(r["pages"] for r in ok)
    splits = sum(1 for r in ok if r["pages"] > 1)
    times = sorted(r["elapsed_s"] for r in ok)

    print(f"inputs           {len(rows)}")
    print(f"accepted         {len(ok)}")
    print(f"rejected         {len(rejected)}")
    print(f"errors           {len(errors)}")
    print(f"pages written    {pages}")
    print(f"inputs split     {splits}")
    print(f"median seconds   {times[len(times) // 2]:.1f}" if times else "median seconds   n/a")
    reasons = collections.Counter(r["reason"].split(" (")[0] for r in rejected + errors)
    for reason, count in reasons.most_common():
        print(f"  {count:4d}  {reason}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
