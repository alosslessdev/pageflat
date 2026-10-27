"""CLI: turn photos of pages into flat scans."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .pipeline import Settings, process_image

_DETECTOR = None


def _init_worker(args) -> None:
    global _DETECTOR
    import cv2
    import torch

    # every worker process is one of several: OpenCV's and torch's default
    # "use all cores" thread pools oversubscribe the machine badly
    cv2.setNumThreads(1)
    torch.set_num_threads(1 if args.device == "cuda" else 2)
    from .models import DocTrPipeline

    _DETECTOR = DocTrPipeline(device=args.device, weights_dir=args.weights)
    if args.sam:
        from .samseg import SamPageSegmenter

        _DETECTOR.sam = SamPageSegmenter(args.sam_weights, device=args.device)


def _work(item) -> dict:
    path, out_dir, settings = item
    res = process_image(path, out_dir, _DETECTOR, settings)
    return res.__dict__


def collect_inputs(inputs: list[str], recursive: bool = True) -> list[Path]:
    files: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            files.extend(sorted(q for q in p.glob(pattern) if q.suffix.lower() in
                               {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}))
        elif p.is_file():
            files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Flatten photographed pages into scanner-like images")
    ap.add_argument("inputs", nargs="*", help="image files or directories")
    ap.add_argument("-o", "--out", required=True, help="output directory")
    ap.add_argument("--list", help="text file with one input path per line")
    ap.add_argument("--dewarp", choices=["doctr", "uvdoc", "best", "none"], default="doctr")
    ap.add_argument("--illum", choices=["classical", "doctr", "none"], default="classical")
    ap.add_argument("--long-side", type=int, default=3508, help="output long side in px (3508 = A4@300dpi)")
    ap.add_argument("--no-split", action="store_true", help="keep two-page spreads together")
    ap.add_argument("--debug", action="store_true", help="save boundary overlays")
    ap.add_argument("--route-url", default="",
                    help="llama-server with a Clef model; routes each photo to a "
                         "segmentation strategy and rejects non-pages early")
    ap.add_argument("--sam", action="store_true", help="refine the page mask with SAM (topmost sheet)")
    ap.add_argument("--sam-weights", default=str(Path(__file__).resolve().parents[1] / "weights" / "sam_vit_b.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parents[2] / "weights"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=1,
                    help="concurrent images inside one process; keeps the GPU busy "
                         "while the CPU decodes/encodes the neighbouring images")
    ap.add_argument("--json", help="write a JSON report of all results here")
    args = ap.parse_args(argv)

    settings = Settings(
        scan_long_side=args.long_side,
        dewarp=args.dewarp,
        illum_mode=args.illum,
        split_spreads=not args.no_split,
        use_sam=args.sam,
        route_url=args.route_url,
        save_debug=args.debug,
    )

    files = collect_inputs(args.inputs)
    if args.list:
        files = [Path(line.strip()) for line in Path(args.list).read_text().splitlines() if line.strip()]
    if not files:
        print("no input images", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(f), str(out_dir), settings) for f in files]
    results = []

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                                 initargs=(args,)) as pool:
            for i, res in enumerate(pool.map(_work, jobs), 1):
                results.append(res)
                print(f"[{i}/{len(jobs)}] {res['status']:<8} {res['source']}", flush=True)
    elif args.threads > 1:
        from concurrent.futures import ThreadPoolExecutor

        _init_worker(args)
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            for i, res in enumerate(pool.map(_work, jobs), 1):
                results.append(res)
                print(f"[{i}/{len(jobs)}] {res['status']:<8} {res['source']}", flush=True)
    else:
        _init_worker(args)
        for i, job in enumerate(jobs, 1):
            res = _work(job)
            results.append(res)
            print(f"[{i}/{len(jobs)}] {res['status']:<8} {res['source']}", flush=True)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"done: {ok}/{len(results)} pages accepted")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
