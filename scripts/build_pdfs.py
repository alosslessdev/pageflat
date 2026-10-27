#!/usr/bin/env python3
"""Assemble per-document searchable PDFs from flat scans + OCR.

    python scripts/build_pdfs.py documents.json scans/ --out pdfs/

Each page is placed at its scan resolution; the OCR lines are drawn invisibly
(text render mode 3) so the PDF is searchable and copyable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_CANDIDATES = [
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "DejaVu"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVu"),
]


def _font():
    for path, name in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                return name
            except Exception:
                pass
    return "Helvetica"


def build(doc: dict, scan_dir: Path, ocr_dir: Path, out_path: Path, font: str,
          max_side: int = 0, quality: int = 95) -> int:
    files = [scan_dir / f for f in doc["pages"]]
    files = [f for f in files if f.exists()]
    if not files:
        return 0
    first = cv2.imread(str(files[0]))
    if first is None:
        return 0
    c = canvas.Canvas(str(out_path))
    for scan in files:
        img = cv2.imread(str(scan))
        if img is None:
            continue
        h, w = img.shape[:2]
        # keep the scan's own resolution by default; max_side > 0 downsamples
        scale = min(1.0, max_side / max(h, w)) if max_side else 1.0
        page = img if scale >= 1 else cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        import io
        c.setPageSize((w, h))
        c.drawImage(ImageReader(io.BytesIO(buf.tobytes())), 0, 0, width=w, height=h)
        ocr_json = ocr_dir / (scan.stem + ".json")
        if ocr_json.exists():
            data = json.loads(ocr_json.read_text())
            sw = data.get("width") or w
            sh = data.get("height") or h
            sx, sy = w / float(sw or w), h / float(sh or h)
            text = c.beginText()
            text.setTextRenderMode(3)
            for line in data.get("lines", []):
                x, y, bw, bh = line["bbox"]
                size = max(4.0, bh * sy * 0.8)
                try:
                    text.setFont(font, size)
                except Exception:
                    text.setFont("Helvetica", size)
                text.setTextOrigin(x * sx, h - (y + bh) * sy)
                text.textLine(line["text"])
            c.drawText(text)
        c.showPage()
    c.save()
    return len(files)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("documents")
    ap.add_argument("scans")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ocr", default=None)
    ap.add_argument("--max-side", type=int, default=0,
                    help="downsample pages above this size (0 = keep the scan resolution)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args(argv)

    docs = json.loads(Path(args.documents).read_text())
    scan_dir = Path(args.scans)
    ocr_dir = Path(args.ocr) if args.ocr else scan_dir / "ocr"
    out_dir = Path(args.out) if args.out else scan_dir / "pdfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    font = _font()

    total = 0
    for doc in docs:
        name = f"document_{doc['document']:03d}.pdf"
        n = build(doc, scan_dir, ocr_dir, out_dir / name, font,
                  args.max_side, args.quality)
        total += n
        print(f"{name}: {n} pages", file=sys.stderr, flush=True)
    print(f"{total} pages in {len(docs)} PDFs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
