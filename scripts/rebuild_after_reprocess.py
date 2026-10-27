#!/usr/bin/env python3
"""After `reprocess.py`: refresh OCR and rebuild only the PDFs that changed.

Deletes the OCR of the replaced pages, re-runs tesseract on whatever is
missing, maps every changed page back to its document, and rebuilds just those
PDFs. Anything the re-run did not touch keeps its existing OCR and PDF.

    python scripts/rebuild_after_reprocess.py scans/ --changed scans/reprocessed_pages.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--changed", required=True)
    ap.add_argument("--documents", default=None, help="defaults to <scans>/documents.json")
    ap.add_argument("--pdfs", default=None, help="defaults to <scans>/pdfs")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    changed = [l.strip() for l in Path(args.changed).read_text().splitlines() if l.strip()]
    changed_set = set(changed)
    ocr = scans / "ocr"
    stale = 0
    for page in changed:
        for suffix in (".json", ".txt"):
            p = ocr / (Path(page).stem + suffix)
            if p.exists():
                stale += 1
                if not args.dry_run:
                    p.unlink()
    print(f"{len(changed)} pages replaced, {stale} stale OCR files removed", file=sys.stderr)

    if not args.dry_run:
        rc = subprocess.call([sys.executable, str(HERE / "ocr_scans.py"), str(scans),
                              "--out", str(ocr), "--workers", str(args.workers)])
        if rc != 0:
            print(f"ocr exited {rc}", file=sys.stderr)

    docs_path = Path(args.documents) if args.documents else scans / "documents.json"
    pdf_dir = Path(args.pdfs) if args.pdfs else scans / "pdfs"
    docs = json.loads(docs_path.read_text())
    touched = []
    for doc in docs:
        pages = set(doc.get("pages", []))
        if pages & changed_set or any(p in pages for p in changed):
            touched.append(doc)
    print(f"{len(touched)} of {len(docs)} documents contain a replaced page", file=sys.stderr)
    if not touched:
        return 0
    subset = Path(scans) / "documents_changed.json"
    if not args.dry_run:
        subset.write_text(json.dumps(touched))
        for doc in touched:
            pdf = pdf_dir / f"document_{doc['document']:03d}.pdf"
            if pdf.exists():
                pdf.unlink()
        rc = subprocess.call([sys.executable, str(HERE / "build_pdfs.py"), str(subset),
                              str(scans), "--out", str(pdf_dir)])
        if rc != 0:
            print(f"build_pdfs exited {rc}", file=sys.stderr)
    else:
        print("\n".join(f"document_{d['document']:03d}.pdf" for d in touched[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
