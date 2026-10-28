#!/usr/bin/env python3
"""Point documents.json at the scans that exist after a re-flatten.

A re-run can rename a page (`IMG_x_p1` becomes `IMG_x_s1`, or a photo that
produced six duplicates now produces two), so document entries can reference
files that no longer exist. Dropping those pages silently shortens the PDFs,
and regrouping from scratch renumbers every document.

So a missing page is remapped to the existing page from the same source photo,
in order, and only pages with no successor at all are dropped. Every document
that changed is reported so only its PDF has to be rebuilt.

    python scripts/remap_documents.py scans/ [--apply]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SUFFIX = re.compile(r"(_p\d+|_s\d+)+$")


def source_of(page: str) -> str:
    return SUFFIX.sub("", Path(page).stem)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--documents", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    scans = Path(args.scans)
    docs_path = Path(args.documents) if args.documents else scans / "documents.json"
    docs = json.loads(docs_path.read_text())
    have = sorted(p.name for p in scans.glob("*.png"))
    have_set = set(have)
    by_source: dict[str, list[str]] = {}
    for name in have:
        by_source.setdefault(source_of(name), []).append(name)
    for key in by_source:
        by_source[key].sort()

    # One stable mapping per source photo, decided before any document is
    # touched: the n-th old page of a source becomes the n-th surviving page.
    # Deciding it per document would let the first document steal pages the
    # next one needs.
    old_by_source: dict[str, list[str]] = {}
    for doc in docs:
        for page in doc.get("pages", []):
            old_by_source.setdefault(source_of(page), []).append(page)
    mapping: dict[str, str] = {}
    dropped_pages: set[str] = set()
    for src, olds in old_by_source.items():
        news = [p for p in by_source.get(src, []) if p not in mapping.values()]
        for i, page in enumerate(olds):
            if i < len(news):
                mapping[page] = news[i]
            else:
                dropped_pages.add(page)
    remapped = sum(1 for k, v in mapping.items() if k != v)
    dropped = len(dropped_pages)
    changed_docs = []
    for doc in docs:
        before = list(doc.get("pages", []))
        out = []
        for page in before:
            if page in mapping:
                out.append(mapping[page])
        sources = {source_of(p) for p in before}
        for src in sources:
            for extra in by_source.get(src, []):
                if extra not in out and extra not in mapping.values():
                    out.append(extra)
        if out != before:
            doc["pages"] = out
            changed_docs.append(doc["document"])
    docs = [d for d in docs if d.get("pages")]
    print(f"{remapped} pages remapped, {dropped} dropped, "
          f"{len(changed_docs)} documents changed, {len(docs)} documents remain")
    if args.apply:
        backup = docs_path.with_suffix(".json.bak")
        backup.write_text(json.dumps(json.loads(docs_path.read_text()), indent=1))
        docs_path.write_text(json.dumps(docs, indent=1))
        Path(scans / "documents_changed.txt").write_text(
            "\n".join(str(d) for d in changed_docs) + "\n")
        print(f"written {docs_path} (backup {backup.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
