#!/usr/bin/env python3
"""Re-flatten the scans that came out warped or clipped.

Reads the audit written by `audit_warp.py`, maps every flagged page back to the
photo it came from, re-runs the pipeline over that photo, replaces the old
outputs of that photo (they may have been named differently), and records the
pages that changed so OCR and the PDFs can be rebuilt for them.

    python scripts/reprocess.py scans/ --audit scans/warp_audit.jsonl \
        --inventory scans/pages_found.jsonl --workers 3 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUFFIX = re.compile(r"(_p\d+|_s\d+)*(_[0-9a-f]{6})?$")


def score(path: Path) -> float:
    """Tesseract mean word confidence: a cheap, comparable quality number."""
    import cv2
    from pageflat.pipeline import ocr_quality
    img = cv2.imread(str(path))
    return ocr_quality(img) if img is not None else 0.0


def bow_of(path: Path) -> float:
    """Median word-line bow in pixels: lower is straighter.

    DocTr's flow bends some pages that were already straight (measured: bow
    14 -> 67 px on IMG_20251110_020257), so straightness has to be measured,
    not assumed from the model that produced the image.
    """
    import cv2
    import numpy as np
    from pageflat import lines
    img = cv2.imread(str(path))
    if img is None:
        return 1e9
    found, back = lines.detect_word_lines(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if len(found) < 6:
        return 1e9
    return float(np.median([l.bow for l in found])) * back


def resolve(stem: str, index: dict[str, str]) -> str | None:
    """The inventory key this scan stem was built from.

    The pipeline appends `_p1`/`_s1` when a photo yields several pages and a
    short hash when a name would collide, so the scan stem is the source stem
    plus suffixes. Match exactly first, then by longest prefix, which never
    mistakes a timestamp for a hash.
    """
    base = re.sub(r"(_p\d+|_s\d+)+$", "", stem)
    if stem in index:
        return stem
    if base in index:
        return base
    best = None
    for key in index:
        if stem.startswith(key + "_") and (best is None or len(key) > len(best)):
            best = key
    return best


def source_of(page: str) -> str:
    """The scan stem without the page suffixes the pipeline adds."""
    return re.sub(r"(_p\d+|_s\d+)+$", "", Path(page).stem)


def load_inventory(path: Path) -> dict[str, str]:
    """stem -> source path, from either a JSONL of {"path": ...} or a TSV."""
    index: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = row.get("path")
            if p:
                index.setdefault(Path(p).stem, p)
        elif "\t" in line:
            stem, p = line.split("\t", 1)
            index.setdefault(stem, p)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--audit", required=True)
    ap.add_argument("--inventory", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["warped", "clipped", "both", "all"], default="both")
    ap.add_argument("--done-journal", default=None,
                    help="append every source handled here and skip it next time")
    ap.add_argument("--dewarp", choices=["none", "doctr", "uvdoc", "best"], default="doctr")
    ap.add_argument("--pick-metric", choices=["ocr", "bow"], default="ocr",
                    help="how the pipeline's best-of chooses its candidate")
    ap.add_argument("--metric", choices=["ocr", "bow"], default="ocr",
                    help="what decides whether the re-run replaces the old scan")
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="replace the old scan when the re-run is within this many "
                         "OCR points of it (a complete page beats a clipped one)")
    ap.add_argument("--work-dir", default=None,
                    help="scratch space for the re-run (put it on the big volume)")
    ap.add_argument("--keep-best", action="store_true", default=True,
                    help="keep the old scan when the re-run scores worse")
    ap.add_argument("--changed-out", default=None,
                    help="write the list of replaced page names here")
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    rows = [json.loads(l) for l in Path(args.audit).read_text().splitlines() if l.strip()]
    want = []
    for r in rows:
        if args.only == "all":
            want.append(r["page"])
        elif args.only in ("warped", "both") and r.get("warped"):
            want.append(r["page"])
        elif args.only in ("clipped", "both") and r.get("clipped"):
            want.append(r["page"])
    stems = sorted({source_of(p) for p in want})
    print(f"{len(want)} flagged pages -> {len(stems)} source photos", file=sys.stderr)

    inventory = load_inventory(Path(args.inventory)) if args.inventory else {}
    resolved = {}
    missing = []
    for stem in stems:
        key = resolve(stem, inventory)
        if key is None:
            missing.append(stem)
        else:
            resolved[key] = inventory[key]
    if missing:
        print(f"{len(missing)} sources not in the inventory, skipping: {missing[:5]}",
              file=sys.stderr)
    stems = sorted(resolved)
    done = set()
    if args.done_journal and Path(args.done_journal).exists():
        done = {l.strip() for l in Path(args.done_journal).read_text().splitlines() if l.strip()}
    skipped_done = [s for s in stems if s in done]
    stems = [s for s in stems if s not in done]
    if skipped_done:
        print(f"{len(skipped_done)} sources already handled, skipped", file=sys.stderr)
    if not stems:
        return 1
    if args.dry_run:
        print("\n".join(stems))
        return 0

    base = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(
        prefix="reprocess_", dir="/home/asdf/pagedewarp/tmp2"))
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="reprocess_", dir=str(base)))
    listing = tmp / "inputs.txt"
    listing.write_text("\n".join(resolved[s] for s in stems) + "\n")
    out = tmp / "flat"
    out.mkdir()
    cmd = [sys.executable, "-m", "pageflat.cli", "--list", str(listing), "-o", str(out),
           "--dewarp", args.dewarp, "--illum", "classical", "--workers", str(args.workers),
           "--content-route", "--paper-reach", "0.06",
           "--pick-metric", args.pick_metric]
    print(" ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    if rc != 0:
        print(f"pipeline exited {rc}", file=sys.stderr)
        if args.done_journal:
            print("journal not updated: this batch has to be retried", file=sys.stderr)
            return rc

    changed: list[str] = []
    kept_old: list[str] = []
    produced = {}
    for png in sorted(out.glob("*.png")):
        produced.setdefault(source_of(png.name), []).append(png)
    for stem in stems:
        new = produced.get(stem, [])
        if not new:
            continue
        old_files = sorted(scans.glob(f"{stem}*.png"))
        # Two pages per side is enough to tell straight from bent, and the
        # comparison runs in a pool: doing it serially made the parent the
        # bottleneck (one source per minute while the GPU sat idle).
        sample_new = new[:2]
        sample_old = old_files[:2]
        if args.metric == "bow":
            with ThreadPoolExecutor(max_workers=4) as pool:
                nb = list(pool.map(bow_of, sample_new))
                ob = list(pool.map(bow_of, sample_old)) if sample_old else [1e9]
            new_conf = float(np.median(nb)) if nb else 1e9
            old_conf = float(np.median(ob))
            replace = not old_files or new_conf < old_conf * 0.85
        else:
            new_conf = sum(score(p) for p in sample_new) / len(sample_new)
            old_conf = (sum(score(p) for p in sample_old) / len(sample_old)) if sample_old else -1.0
            replace = not old_files or new_conf >= old_conf - args.tolerance
        # never replace a scan with a worse one: the re-run is an offer, and the
        # old output already went through OCR and into a PDF
        if not replace:
            kept_old.append(stem)
            continue
        for old in old_files:
            old.unlink()
        for png in new:
            dest = scans / png.name
            shutil.copy2(png, dest)
            changed.append(png.name)
    if args.done_journal:
        with open(args.done_journal, "a") as fh:
            for stem in stems:
                fh.write(stem + "\n")
    if kept_old:
        print(f"{len(kept_old)} sources kept their old output (the re-run was not better)",
              file=sys.stderr)
    print(f"replaced {len(changed)} pages", file=sys.stderr)
    if args.changed_out:
        Path(args.changed_out).write_text("\n".join(sorted(changed)) + "\n")
    print(f"work dir kept at {tmp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
