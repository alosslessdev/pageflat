#!/usr/bin/env python3
"""Pick the straightest version of every scan, across all the methods.

For each scan the candidates are: the scan itself, DocTr's flow, DewarpNet's
3D map, UVDoc's grid, and the monocular-depth correction. Each is scored by
the rule (how far the text blocks are from rectangles) and by OCR, and the
page takes the straightest candidate that does not lose its text.

    python scripts/rectify_scans.py scans/ [--only-failing] [--apply]
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2, numpy as np
from pageflat.detect import text_lines
from pageflat.tiles import text_lines_tiled
from pageflat.blocks import blocks_of
from pageflat.pipeline import ocr_quality
from pageflat.models import DocTrPipeline, DewarpNetDewarper, UVDocDewarper
from pageflat import depthwarp


def drift(img):
    b = blocks_of(text_lines(img))
    if not b:
        return None
    e = sorted(x.rect_error for x in b)
    return float(e[len(e) // 2])


def candidates(img, doctr, dewnet, uvdoc, include_depth: bool = True):
    """The cheap models first; the depth pass (6-7 s) only if they fail.

    The depth correction is the slowest candidate by an order of magnitude, and
    on most pages one of the flow models already brings the blocks to a
    rectangle, so running it unconditionally is most of the wasted time.
    """
    out = {"none": img}
    for name, fn in (("doctr", doctr.rectify), ("dewnet", dewnet.rectify),
                     ("uvdoc", uvdoc.rectify)):
        try:
            out[name] = fn(img)
        except Exception:  # noqa: BLE001
            pass
    if not include_depth:
        return out
    # Both depth engines are offered: V2 and V3 are equal in aggregate but win
    # different pages, and the router takes whichever is straighter.
    for name, engine in (("depth", "v2"), ("depth_da3", "da3")):
        try:
            cand, info = depthwarp.rectify(img, engine=engine)
            if info.get("corrected"):
                out[name] = cand
        except Exception:  # noqa: BLE001
            pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only-failing", action="store_true",
                    help="skip pages whose blocks are already rectangles")
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--conf-budget", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None,
                    help="i/N: process every N-th page, so several workers can share the corpus")
    ap.add_argument("--all", action="store_true", help="every scan, not only the failing ones")
    ap.add_argument("--max-side", type=int, default=2600,
                    help="working resolution: 64 MP phone photos cannot be run through "
                         "four warp models in one process without swapping")
    ap.add_argument("--pages-from", default=None,
                    help="JSONL of pages with a drift_frac, to avoid re-measuring all")
    ap.add_argument("--journal", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="write the pages of this attempt here instead of rewriting the corpus")
    ap.add_argument("--resume", action="store_true",
                    help="skip pages already in the journal and append")
    args = ap.parse_args(argv)
    scans = Path(args.scans)
    journal = Path(args.journal) if args.journal else scans / "rectify_journal.jsonl"
    if args.all:
        pages = sorted(list(scans.glob("*.png")) + list(scans.glob("*.jpg")) + list(scans.glob("*.jpeg")))
    elif args.pages_from:
        rows = [json.loads(l) for l in open(args.pages_from) if l.strip()]
        keep = [r["page"] for r in rows if r.get("drift_frac", 0) > args.threshold / 100.0]
        pages = [scans / p for p in keep if (scans / p).exists()]
    else:
        pages = sorted(list(scans.glob("*.png")) + list(scans.glob("*.jpg")) + list(scans.glob("*.jpeg")))

    doctr = DocTrPipeline(device="cuda")
    dewnet = DewarpNetDewarper(device="cuda")
    uvdoc = UVDocDewarper()
    done = set()
    if args.resume and journal.exists():
        for line in journal.read_text().splitlines():
            try:
                done.add(json.loads(line)["page"])
            except Exception:
                pass
        pages = [p for p in pages if p.name not in done]
        print(f"resuming: {len(done)} already scored, {len(pages)} to go", file=sys.stderr)
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        pages = pages[i::n]
    if args.limit:
        pages = pages[:args.limit]
    changed, skipped = [], 0
    with open(journal, "a" if args.resume else "w") as fh:
        for i, page in enumerate(pages, 1):
            img = cv2.imread(str(page))
            if img is None:
                continue
            if args.max_side and max(img.shape[:2]) > args.max_side:
                s_ = args.max_side / float(max(img.shape[:2]))
                img = cv2.resize(img, (int(img.shape[1] * s_), int(img.shape[0] * s_)),
                                 interpolation=cv2.INTER_AREA)
            base = drift(img)
            if args.only_failing and (base is None or base <= args.threshold):
                skipped += 1
                # record it, or the supervisor can never see the list drained
                fh.write(json.dumps({"page": page.name, "base": base, "scored": [],
                                     "pick": None, "drift": None, "conf": None,
                                     "skipped": "already rectangular"}) + "\n")
                fh.flush()
                continue
            # cheap models first, then the block rule; the slow depth pass is
            # added only when the cheap ones leave the page bent
            # Build and score one candidate at a time, and stop as soon as one
            # already satisfies the goal: once a page is under 1 % drift no
            # further model can improve the outcome, and the four remaining
            # evaluations are pure cost.
            cands = {"none": img}
            from pageflat.pipeline import keeps_text as _keeps
            for name, fn in (("uvdoc", uvdoc.rectify), ("doctr", doctr.rectify),
                             ("dewnet", dewnet.rectify)):
                try:
                    cands[name] = fn(img)
                except Exception:  # noqa: BLE001
                    continue
                d = drift(cands[name])
                if d is not None and d <= 0.01:
                    ok, _ = _keeps(img, cands[name])
                    if ok:
                        break
            # Cheapest signal first: sharpness ranks the candidates for free,
            # the learned detector then measures the block rule only for the
            # two that look straightest, and OCR only for the finalists.
            from pageflat.sharp import row_sharpness
            ranked = sorted(cands.items(), key=lambda kv: -row_sharpness(kv[1]))[:2]
            if "none" not in [n for n, _ in ranked]:
                ranked.append(("none", cands["none"]))
            scored = []
            for name, cand in ranked:
                scored.append((name, drift(cand), ocr_quality(cand)))
            for name, cand in cands.items():
                if name not in [s[0] for s in scored]:
                    scored.append((name, None, None))
            known = [s for s in scored if s[1] is not None]
            if known and min(s[1] for s in known) > args.threshold and not any(
                    k.startswith("depth") for k in cands):
                extra = candidates(img, doctr, dewnet, uvdoc, include_depth=True)
                for name, cand in extra.items():
                    if not name.startswith("depth"):
                        continue
                    cands[name] = cand
                    d, c = drift(cand), ocr_quality(cand)
                    scored.append((name, d, c))
                    if d is not None and c is not None:
                        known.append((name, d, c))
            if not known:
                fh.write(json.dumps({"page": page.name, "base": base, "scored": scored,
                                     "pick": None, "drift": None, "conf": None}) + "\n")
                fh.flush()
                continue
            # A candidate is viable if it is straighter AND it still produces the
            # page's words. An absolute confidence floor rejects pages where
            # tesseract simply found nothing (OCR 0), which is a failed
            # measurement, not a failed correction — so the test is comparative.
            from pageflat.pipeline import keeps_text
            base_conf = next((s[2] for s in scored if s[0] == "none"), 0.0)
            viable = []
            for s_ in sorted(known, key=lambda x: x[1]):
                if s_[0] == "none":
                    continue
                if base is not None and s_[1] > base * 0.95:
                    continue
                if base_conf <= 1.0:            # nothing to lose: judge on geometry
                    viable.append(s_)
                    continue
                ok, _det = keeps_text(img, cands[s_[0]])
                if ok:
                    viable.append(s_)
            pick = viable[0] if viable else next(
                (s_ for s_ in known if s_[0] == "none"), None)
            row = {"page": page.name, "base": base, "scored": scored,
                   "pick": pick[0] if pick else None,
                   "drift": pick[1] if pick else None,
                   "conf": pick[2] if pick else None}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            keep_better = pick and pick[0] != "none" and (base is None or pick[1] < base * 0.8)
            if args.out_dir:
                # every attempt keeps its own copy of the page set, so attempts
                # can be compared and the corpus only changes when one wins
                import shutil
                target = Path(args.out_dir)
                target.mkdir(parents=True, exist_ok=True)
                if keep_better:
                    cv2.imwrite(str(target / page.name), cands[pick[0]])
                else:
                    shutil.copy2(page, target / page.name)
            if keep_better:
                if args.apply:
                    cv2.imwrite(str(page), cands[pick[0]])
                    # record each rewrite as it happens: a run that is killed
                    # halfway must still leave the list the rebuild needs
                    with open(scans / "rectified_pages.txt", "a") as cf:
                        cf.write(page.name + "\n")
                changed.append(page.name)
            if i % 50 == 0:
                print(f"{i}/{len(pages)} changed {len(changed)}", file=sys.stderr, flush=True)
    if args.apply:
        print(f"{len(changed)} pages rewritten in this pass")
    print(f"{len(changed)} pages improved, {skipped} skipped as already rectangular")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
