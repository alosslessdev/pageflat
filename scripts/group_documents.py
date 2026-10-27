#!/usr/bin/env python3
"""Group flat scans into documents and order their pages.

Uses only what the pages say about themselves:

* detected page numbers (first/last line, "12", "- 12 -", "Página 12", "12 of 40");
* header/title similarity between consecutive pages (difflib ratio on the first
  two lines);
* capture session (timestamp in the source file name) for tie-breaking;
* a hard split when numbering restarts at 1 or the gap between captures is long.

    python scripts/group_documents.py scans/ocr --out documents.json [--llm URL]

With `--llm` the ambiguous boundaries are adjudicated by a small instruct model
served by llama-server (any chat model works, e.g. Qwen3-4B or LFM2.5).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

NUM_PATTERNS = [
    re.compile(r"^\s*[-–—]?\s*(\d{1,3})\s*[-–—]?\s*$"),
    re.compile(r"^\s*(?:p[aá]g(?:ina)?\.?|pg\.?)\s*(\d{1,3})\s*$", re.I),
    re.compile(r"^\s*(\d{1,3})\s*(?:/|of|de)\s*(\d{1,3})\s*$", re.I),
]


def page_number(lines: list[dict]) -> int | None:
    if not lines:
        return None
    for cand in (lines[-1]["text"], lines[0]["text"]):
        for pat in NUM_PATTERNS:
            m = pat.match(cand)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    return None
    return None


def header_text(lines: list[dict]) -> str:
    texts = [l["text"].strip() for l in lines if l["text"].strip()]
    return " ".join(texts[:2])[:200]


def capture_time(name: str) -> datetime | None:
    m = re.search(r"(\d{8})_(\d{6})", name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_pages(ocr_dir: Path) -> list[dict]:
    pages = []
    for json_path in sorted(ocr_dir.glob("*.json")):
        data = json.loads(json_path.read_text())
        name = data["file"]
        texts = [l["text"].strip() for l in data["lines"] if l["text"].strip()]
        pages.append({
            "file": name,
            "number": page_number(data["lines"]),
            "header": header_text(data["lines"]),
            "head": " ".join(texts[:2]),
            "tail": " ".join(texts[-2:]),
            "n_lines": len(data["lines"]),
            "captured": capture_time(name),
        })
    return pages


def embed_text(url: str, text: str, model: str = "local") -> list[float] | None:
    try:
        import requests
    except ImportError:
        return None
    try:
        r = requests.post(url.rstrip("/") + "/v1/embeddings",
                          json={"model": model, "input": text[:2000]}, timeout=60)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]
    except Exception:
        return None


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def llm_same_document(url: str, a: dict, b: dict, model: str = "local") -> bool | None:
    """Ask a small instruct model whether two consecutive pages belong together."""
    try:
        import requests
    except ImportError:
        return None
    prompt = (
        "Dos páginas escaneadas consecutivas. ¿Pertenecen al mismo documento? "
        "Responde solo SI o NO.\n"
        f"Página A, final: {a['tail'][-300:]!r}\n"
        f"Página B, inicio: {b['head'][:300:]!r}\n"
    )
    try:
        r = requests.post(url.rstrip("/") + "/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0.0,
        }, timeout=60)
        r.raise_for_status()
        answer = r.json()["choices"][0]["message"]["content"].strip().upper()
        if answer.startswith("SI") or answer.startswith("SÍ"):
            return True
        if answer.startswith("NO"):
            return False
    except Exception:
        return None
    return None


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def session_split(pages: list[dict], gap_seconds: int = 300) -> list[list[dict]]:
    """Contiguous capture bursts: a book or magazine scanned in one sitting."""
    out: list[list[dict]] = []
    cur: list[dict] = []
    for page in pages:
        if cur and page["captured"] and cur[-1]["captured"]:
            if (page["captured"] - cur[-1]["captured"]).total_seconds() > gap_seconds:
                out.append(cur)
                cur = []
        cur.append(page)
    if cur:
        out.append(cur)
    return out


def split_session(session: list[dict]) -> list[list[dict]]:
    """Inside one burst, a page numbered 1 (or a cover) starts a new document."""
    docs: list[list[dict]] = []
    cur: list[dict] = []
    for page in session:
        starts = bool(cur) and (
            (page.get("number") == 1 and (cur[-1].get("number") or 0) > 1)
            or page.get("cover") and not cur[-1].get("cover")
        )
        if starts:
            docs.append(cur)
            cur = []
        cur.append(page)
    if cur:
        docs.append(cur)
    return docs


def group(pages: list[dict], threshold: float = 0.25, gap_minutes: int = 360,
          llm_url: str | None = None, embed_url: str | None = None,
          embed_threshold: float = 0.72) -> list[list[dict]]:
    if embed_url:
        for page in pages:
            if "embedding" not in page:
                text = (page.get("head", "") + " " + page.get("tail", "")).strip()
                if text:
                    page["embedding"] = embed_text(embed_url, text)
    docs: list[list[dict]] = []
    current: list[dict] = []
    for page in pages:
        if not current:
            current = [page]
            continue
        prev = current[-1]
        split = False
        if page["captured"] and prev["captured"]:
            gap = (page["captured"] - prev["captured"]).total_seconds() / 60.0
            if gap > gap_minutes:
                split = True
        if page["number"] is not None and prev["number"] is not None:
            if page["number"] == 1 and prev["number"] > 1:
                split = True
        same_burst = bool(page["captured"] and prev["captured"] and
                          (page["captured"] - prev["captured"]).total_seconds() < 600)
        if not split and embed_url and prev.get("embedding") and page.get("embedding"):
            sim = cosine(prev["embedding"], page["embedding"])
            if sim < embed_threshold:
                split = True
        if not split and prev["header"] and page["header"]:
            sim = similarity(prev["header"], page["header"])
            limit = 0.1 if same_burst else threshold
            if sim < limit:
                decision = None
                if llm_url and 0.25 <= sim < limit:
                    decision = llm_same_document(llm_url, prev, page)
                split = decision is not True
        if split:
            docs.append(current)
            current = [page]
        else:
            current.append(page)
    if current:
        docs.append(current)
    return docs


def order_document(pages: list[dict]) -> list[dict]:
    numbered = [p for p in pages if p["number"] is not None]
    if len(numbered) >= max(2, int(0.5 * len(pages))):
        return sorted(pages, key=lambda p: (p["number"] is None, p["number"] or 0))
    return sorted(pages, key=lambda p: (p["captured"] or datetime.max))


def _tokens(text: str) -> list[str]:
    import re as _re
    import unicodedata

    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _re.findall(r"[a-z]{4,}", text)]


def merge_similar(docs: list[list[dict]], ocr_dir: Path, threshold: float = 0.35) -> list[list[dict]]:
    """Merge documents that talk about the same thing.

    A book can be photographed across several sessions; sequential grouping then
    splits it. TF-IDF centroids over the OCR text are enough to pull those back
    together without an embedding server.
    """
    import math

    texts = []
    for doc in docs:
        buf = []
        for page in doc:
            j = ocr_dir / (Path(page["file"]).stem + ".json")
            if j.exists():
                data = json.loads(j.read_text())
                buf.extend(l["text"] for l in data["lines"])
        texts.append(buf)

    tokenised = [_tokens(" ".join(t)) for t in texts]
    df = {}
    for toks in tokenised:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(tokenised)
    idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}

    vecs = []
    for toks in tokenised:
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (c / max(1, len(toks))) * idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vecs.append({t: v / norm for t, v in vec.items()})

    def cos(a, b):
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    parent = list(range(len(docs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if cos(vecs[i], vecs[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    merged: dict[int, list[dict]] = {}
    for i, doc in enumerate(docs):
        merged.setdefault(find(i), []).extend(doc)
    out = list(merged.values())
    for doc in out:
        doc.sort(key=lambda p: (p["captured"] or datetime.max))
    return out


def _cover_score(scan_dir: Path | None, name: str) -> bool:
    """A cover is a mostly-saturated, low-text page (a book/magazine cover)."""
    if scan_dir is None:
        return False
    import cv2
    import numpy as np

    img = cv2.imread(str(scan_dir / name), cv2.IMREAD_REDUCED_COLOR_4)
    if img is None:
        return False
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    colourful = float(((sat > 90) & (val > 40)).mean())
    return colourful > 0.35


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ocr_dir")
    ap.add_argument("--out", default="documents.json")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--gap-minutes", type=int, default=360)
    ap.add_argument("--llm", default=None, help="llama-server URL for ambiguous boundaries")
    ap.add_argument("--embed-url", default=None, help="llama-server URL (with --embeddings) for page-topic similarity")
    ap.add_argument("--embed-threshold", type=float, default=0.72)
    ap.add_argument("--scan-dir", default=None, help="scan directory (enables cover detection)")
    ap.add_argument("--sessions", action="store_true",
                    help="group wholesale capture bursts as documents (books, magazines)")
    ap.add_argument("--merge-similar", type=float, default=0.0,
                    help="merge documents whose TF-IDF centroids are at least this similar")
    args = ap.parse_args(argv)

    pages = load_pages(Path(args.ocr_dir))
    scan_dir = Path(args.scan_dir) if args.scan_dir else None
    for p in pages:
        p["cover"] = _cover_score(scan_dir, p["file"])
    if not pages:
        print("no OCR output found", file=sys.stderr)
        return 1
    if args.sessions:
        docs = []
        for session in session_split(pages):
            docs.extend(split_session(session))
    else:
        docs = group(pages, args.threshold, args.gap_minutes, args.llm,
                     args.embed_url, args.embed_threshold)
    if args.merge_similar > 0:
        docs = merge_similar(docs, Path(args.ocr_dir), args.merge_similar)
    out = []
    for i, doc in enumerate(docs, 1):
        ordered = order_document(doc)
        out.append({
            "document": i,
            "pages": [p["file"] for p in ordered],
            "numbers": [p["number"] for p in ordered],
            "headers": [p["header"][:60] for p in ordered],
        })
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"{len(pages)} pages -> {len(out)} documents", file=sys.stderr)
    for d in out[:10]:
        print(f"  doc {d['document']}: {len(d['pages'])} pages {d['numbers'][:8]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
