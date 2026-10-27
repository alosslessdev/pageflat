#!/usr/bin/env python3
"""Cleanup stage: drop scans that Clef says are not pages.

Classifies every PNG in a scan directory with Cloudflare Clef served by
llama-server, appends verdicts to a JSONL file (resumable) and moves files whose
page probability is below the threshold into `_nonpage/`.

    llama-server -m clef-flash.gguf --mmproj mmproj.gguf -ngl 16 -np 1 --port 8099
    python scripts/clef_cleanup.py scans/ --verdicts clef_verdicts.jsonl --threshold 0.5
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_pages import classify  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--verdicts", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--side", type=int, default=320)
    args = ap.parse_args(argv)

    root = Path(args.scans)
    verdicts_path = Path(args.verdicts or (root.parent / "clef_verdicts.jsonl"))
    seen = {}
    if verdicts_path.exists():
        for line in verdicts_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                seen[rec["file"]] = rec
            except Exception:
                pass

    todo = [p for p in sorted(root.glob("*.png")) if p.name not in seen]
    print(f"{len(seen)} already judged, {len(todo)} to go", file=sys.stderr)
    moved = 0
    with open(verdicts_path, "a") as out:
        for i, path in enumerate(todo, 1):
            res = classify(args.url, str(path), side=args.side)
            rec = {"file": path.name, **{k: v for k, v in res.items() if k != "path"}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if "error" in rec:
                continue
            if float(rec.get("page", 0)) < args.threshold:
                (root / "_nonpage").mkdir(exist_ok=True)
                shutil.move(str(path), root / "_nonpage" / path.name)
                moved += 1
            if i % 50 == 0:
                print(f"{i}/{len(todo)}  moved {moved}", file=sys.stderr, flush=True)
    print(f"done: {moved} moved to _nonpage", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
