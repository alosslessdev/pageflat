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
    ap.add_argument("--only", choices=["warped", "clipped", "both"], default="both")
    ap.add_argument("--dewarp", choices=["doctr", "uvdoc", "best"], default="doctr")
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
        if args.only in ("warped", "both") and r.get("warped"):
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
           "--dewarp", args.dewarp, "--illum", "classical", "--workers", "1",
           "--content-route", "--paper-reach", "0.06"]
    print(" ".join(cmd), file=sys.stderr)
    rc = subprocess.call(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    if rc != 0:
        print(f"pipeline exited {rc}", file=sys.stderr)

    changed: list[str] = []
    kept_old: list[str] = []
    produced = {}
    for png in sorted(out.glob("*.png")):
        produced.setdefault(source_of(png.name), []).append(png)
    for stem in stems:
        new = produced.get(stem, [])
        if not new:
            continue
        new_conf = sum(score(p) for p in new) / len(new)
        old_files = sorted(scans.glob(f"{stem}*.png"))
        old_conf = (sum(score(p) for p in old_files) / len(old_files)) if old_files else -1.0
        # never replace a scan with a worse one: the re-run is an offer, and the
        # old output already went through OCR and into a PDF
        if args.keep_best and old_files and new_conf < old_conf - args.tolerance:
            kept_old.append(stem)
            continue
        for old in old_files:
            old.unlink()
        for png in new:
            dest = scans / png.name
            shutil.copy2(png, dest)
            changed.append(png.name)
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
