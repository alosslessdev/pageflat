#!/usr/bin/env python3
"""OCR flat scans with Tesseract, keeping word boxes for searchable PDFs.

    python scripts/ocr_scans.py scans/ --out scans/ocr --workers 4

Writes, per scan:
  <name>.txt   plain text
  <name>.json  {"lines": [{"text":..., "bbox":[x,y,w,h], "conf":...}], "width":..., "height":...}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def _lang() -> str:
    try:
        out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True,
                             timeout=20).stdout.splitlines()[1:]
        langs = [l.strip() for l in out if l.strip() and l.strip() != "osd"]
    except Exception:
        return ""
    for pref in ("spa+eng", "eng", "spa"):
        parts = pref.split("+")
        if all(p in langs for p in parts):
            return pref
    return langs[0] if langs else ""


def ocr_one(path_str: str, out_dir: str, lang: str, psm: int = 3,
            max_side: int = 1800) -> dict:
    path = Path(path_str)
    out_dir_p = Path(out_dir)
    txt_path = out_dir_p / (path.stem + ".txt")
    json_path = out_dir_p / (path.stem + ".json")
    if json_path.exists():
        return {"file": path.name, "status": "cached"}

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"file": path.name, "status": "error", "error": "unreadable"}
    h, w = img.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    tmp = out_dir_p / (path.stem + ".ocr.png")
    out_dir_p.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp), img)

    cmd = ["tesseract", str(tmp), "stdout", "--psm", str(psm)]
    if lang:
        cmd += ["-l", lang]
    try:
        proc = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return {"file": path.name, "status": "error", "error": repr(exc)[:120]}
    tmp.unlink(missing_ok=True)

    words, lines = [], {}
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            conf = float(parts[10])
        except ValueError:
            continue
        text = parts[11].strip()
        if not text:
            continue
        key = (parts[2], parts[3], parts[4])  # block, par, line
        lines.setdefault(key, []).append((text, int(parts[6]), int(parts[7]),
                                          int(parts[8]), int(parts[9]), conf))

    out_lines = []
    for key in sorted(lines, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        ws = lines[key]
        text = " ".join(w[0] for w in ws)
        x0 = min(w[1] for w in ws)
        y0 = min(w[2] for w in ws)
        x1 = max(w[1] + w[3] for w in ws)
        y1 = max(w[2] + w[4] for w in ws)
        conf = sum(w[5] for w in ws) / len(ws)
        out_lines.append({"text": text, "bbox": [x0, y0, x1 - x0, y1 - y0],
                          "conf": round(conf, 1)})

    out_dir_p.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(l["text"] for l in out_lines) + "\n")
    json_path.write_text(json.dumps({"file": path.name, "width": w, "height": h,
                                     "lines": out_lines}))
    return {"file": path.name, "status": "ok", "lines": len(out_lines)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--psm", type=int, default=3)
    ap.add_argument("--max-side", type=int, default=1800)
    args = ap.parse_args(argv)

    root = Path(args.scans)
    if root.is_file():
        files = [l.strip() for l in root.read_text().splitlines() if l.strip()]
        out_dir = args.out or str(Path(files[0]).parent / "ocr")
    else:
        out_dir = args.out or str(root / "ocr")
        files = [str(p) for p in sorted(root.glob("*.png"))]
    lang = _lang()
    print(f"{len(files)} scans, lang={lang or '-'}", file=sys.stderr)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for res in pool.map(ocr_one, files, [out_dir] * len(files), [lang] * len(files),
                            [args.psm] * len(files), chunksize=2):
            done += 1
            if res["status"] == "error":
                print("ERR", res, file=sys.stderr)
            if done % 50 == 0:
                print(f"{done}/{len(files)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
