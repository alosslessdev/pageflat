#!/usr/bin/env python3
"""Re-flatten the whole corpus in chunks, keeping only what got straighter.

`reprocess.py` runs the pipeline over every source and only replaces scans at
the end, which needs space for all of the new pages at once (~100 GB here) and
loses the work if it is interrupted. This driver instead does chunks: flatten
100 sources, compare each page against the scan it would replace, rewrite the
winners, delete the chunk's scratch, record the chunk in a journal.

    python scripts/rewarp_all.py scans/ --inventory jpg_index.tsv [--chunk 100]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from scripts.reprocess import load_inventory, resolve, source_of  # noqa: E402


def sharpness(path: Path) -> float:
    import cv2
    from pageflat.sharp import row_sharpness
    img = cv2.imread(str(path))
    return row_sharpness(img) if img is not None else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dewarp", default="best")
    ap.add_argument("--pick-metric", default="bow")
    ap.add_argument("--gain", type=float, default=1.15,
                    help="how much sharper the re-run must be to replace")
    ap.add_argument("--journal", default=None)
    ap.add_argument("--changed-out", default=None)
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    journal = Path(args.journal) if args.journal else scans / "rewarp_done.txt"
    changed_path = Path(args.changed_out) if args.changed_out else scans / "rewarped_pages.txt"
    scratch = Path(args.scratch) if args.scratch else scans.parent / "rewarp_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    done = {l.strip() for l in journal.read_text().splitlines()} if journal.exists() else set()
    inventory = load_inventory(Path(args.inventory))
    stems = []
    for name in sorted(p.name for p in scans.glob("*.png")):
        key = resolve(source_of(name), inventory)
        if key and key not in done and key not in stems:
            stems.append(key)
    print(f"{len(stems)} sources to do ({len(done)} already done)", file=sys.stderr)

    changed = [l.strip() for l in changed_path.read_text().splitlines()] if changed_path.exists() else []
    total_replaced = 0
    failures = 0
    for i in range(0, len(stems), args.chunk):
        batch = stems[i:i + args.chunk]
        work = Path(tempfile.mkdtemp(prefix="chunk_", dir=str(scratch)))
        listing = work / "inputs.txt"
        listing.write_text("\n".join(inventory[s] for s in batch) + "\n")
        out = work / "flat"
        out.mkdir()
        cmd = [sys.executable, "-m", "pageflat.cli", "--list", str(listing), "-o", str(out),
               "--dewarp", args.dewarp, "--illum", "classical", "--workers", str(args.workers),
               "--content-route", "--pick-metric", args.pick_metric]
        rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        if rc != 0:
            # a worker died inside the pool: retry the same chunk serially, so
            # one bad photo cannot cost the whole chunk
            cmd = [c if c != str(args.workers) else "1" for c in cmd]
            print("retrying the chunk with one worker", file=sys.stderr)
            rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[1]))
        if rc != 0:
            print(f"batch {i // args.chunk} failed ({rc}); not journaling it", file=sys.stderr)
            shutil.rmtree(work, ignore_errors=True)
            failures += 1
            if failures >= 3:
                print("three batches in a row failed; stopping so the cause can be "
                      "looked at (usually GPU memory with too many workers)",
                      file=sys.stderr)
                return 1
            continue
        failures = 0

        produced: dict[str, list[Path]] = {}
        for png in sorted(out.glob("*.png")):
            produced.setdefault(source_of(png.name), []).append(png)
        for stem in batch:
            new = produced.get(stem, [])
            if not new:
                continue
            old_files = sorted(scans.glob(f"{stem}*.png"))
            new_s = float(np.median([sharpness(p) for p in new[:2]]))
            old_s = float(np.median([sharpness(p) for p in old_files[:2]])) if old_files else 0.0
            if old_files and new_s <= old_s * args.gain:
                continue
            for f in old_files:
                f.unlink()
            names = []
            for png in new:
                shutil.copy2(png, scans / png.name)
                names.append(png.name)
            changed.extend(names)
            total_replaced += len(names)
        with open(journal, "a") as fh:
            for stem in batch:
                fh.write(stem + "\n")
        changed_path.write_text("\n".join(sorted(set(changed))) + "\n")
        shutil.rmtree(work, ignore_errors=True)
        done_n = min(i + args.chunk, len(stems))
        print(f"{done_n}/{len(stems)} sources, {total_replaced} pages replaced", file=sys.stderr)
    print(f"done: {total_replaced} pages replaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
