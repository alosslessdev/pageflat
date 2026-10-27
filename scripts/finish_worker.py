#!/usr/bin/env python3
"""Dynamic PDF worker: take the next unbuilt document, build it, repeat.

Static slices leave the CPU idle once the quick documents are done; this pulls
work from the shared document list under a lock so eight workers keep every core
busy until nothing is left.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pdfs import _font, build  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    docs_path = scans / "documents.json"
    out_dir = scans / "pdfs"
    ocr_dir = scans / "ocr"
    out_dir.mkdir(exist_ok=True)
    lock_path = scans / ".pdf_queue.lock"
    font = _font()

    while True:
        doc = None
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            docs = json.loads(docs_path.read_text())
            built = {p.name for p in out_dir.glob("document_*.pdf")}
            for d in docs:
                name = f"document_{d['document']:03d}.pdf"
                if name in built:
                    continue
                target = out_dir / name
                if target.exists():
                    continue
                # reserve the slot so the next worker picks a different doc
                target.write_bytes(b"")
                doc = (d, target)
                break
        if doc is None:
            return 0
        d, target = doc
        try:
            build(d, scans, ocr_dir, target, font, 0, 95)
        except Exception as exc:  # noqa: BLE001
            target.unlink(missing_ok=True)
            print(f"FAILED {d['document']}: {exc!r}", file=sys.stderr)
        time.sleep(args.sleep)


if __name__ == "__main__":
    raise SystemExit(main())
