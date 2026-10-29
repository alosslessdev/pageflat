# Results on a real library

## Discovery

Machine: two phone backups plus the system disk, 243 858 JPEG files.

| stage | count |
|---|---|
| JPEGs traversed | 243 858 |
| ≥ 60 MP (9248×6944 / 6944×9248, i.e. 64.2 MP) | 5 600 |
| flagged page-like by `scripts/find_pages.py` (score ≥ 0.75) | 652 |

The 5 600 large captures come from two devices in two orientations, plus a
handful of panoramas. The 652 flagged files are photographed paper: ruled
notebooks with handwriting, printed forms, receipts, newspaper pages, book
spreads, textbook pages.

## Rectification

Every flagged file was pushed through `pageflat` with DocTr + MobileSAM,
classical illumination flattening, deskew and gutter splitting.

Output: one PNG per page at 3508 px on the long side (≈ A4 at 300 dpi), plus a
JSON report per batch (`ok` / `rejected` with reason, quad, rotation, content
metrics, pages, timing).

| result | value |
|---|---|
| candidate photos processed | 637 of 652 |
| rejected (no page found in the frame) | 15 |
| scans written | 934 PNG, 8.1 GB |
| inputs split into a spread | 257 (515 page files) |
| output size | 3508 px long side, ~300 dpi A4 |

Every scan has an entry in `scans_manifest.json` next to it.

After the batch, orientations were repaired in place with
`scripts/repair_orientation.py`, which trusts docTR's page-orientation classifier
when its confidence is ≥ 0.9 and otherwise falls back to OCR word-confidence
scoring of the four candidate rotations. On this library the classifier is right
for ordinary pages but can hold the same class for very image-heavy children's
pages, where the OCR signal is the reliable one — the combined rule fixed both
kinds, 170 + 13 pages rotated in total.

## What works well

* pages that fill the frame (forms, receipts, single book pages) — the result is
  indistinguishable from a flatbed scan except for paper texture;
* coloured textbook pages — colour and images survive the illumination pass;
* upside-down and sideways captures — orientation is recovered by OCR scoring,
  not by trusting Tesseract OSD alone;
* two-page spreads with a visible gutter — split into single pages, unless a
  picture runs across the spine.

## Known failures

* A smooth, neutral, text-free surface (a blue-gradient backdrop, a striped
  sheet) can still pass the content gate.
* A page whose edge is physically covered by another sheet is cropped to the
  visible part.
* Handwriting gives the learned flow model no text-line cue, so strongly curled
  notebook pages keep some residual curvature; `scripts/refine_scan.py` can
  straighten them when the measured baseline wobble exceeds 1% of page height.

## Measuring on the sources (October 2026)

The earlier numbers were taken on `pageflat_scans_v4`, which `finalize_scans.py`
builds from **already flattened** scans. Eleven of twelve spot-checked pages had
a 64 MP original (6944x9248) behind a 1025-2508 px derivative, so the reported
improvement was partly the pipeline repairing its own earlier output.

Two things had to be fixed before a source measurement meant anything:

* **64 MP input killed the batch.** Four warp models on a 9248-px image take the
  process down with the OOM killer — exit 137, no journal line. The batch now
  runs at a working side (`--max-side`, default 2600).
* **The detector was reading the masthead.** It resizes its input, so a 9248-px
  page and a 1024-px thumbnail of it look identical to it: 24 lines, 3 blocks.
  `pageflat/tiles.py` detects on native-resolution tiles — 155 lines, 13 blocks
  on the same page.

With that ruler, 16 source pages at a 3000-px working side:

| variant | base median | final median | pages above 3 % |
|---|---|---|---|
| Depth Anything 3 | 37.8 px | 21.7 px | 0 |
| Depth Anything V2 | 37.8 px | 21.9 px | 0 |
| both offered | — | 20.5 px | 0 |

V2 and V3 are equal in aggregate and complementary in detail (three exclusive
wins each), which is why `rectify_scans.py` now offers both as candidates rather
than choosing between them.

**Most source pages should not be warped at all.** Four of thirteen are left
untouched because every model makes them worse — UVDoc triples the drift on
`IMG_20260507_125608` (22.5 -> 145.4 px) and DewarpNet makes it 9.5x worse. A
page photographed flat-on is barely bent, and the warp models introduce
curvature where there was little.

## Delivering PDFs

The scanner-like guarantee is measurable: run the same page through
`rectify_scans.py` and through `pageflat.cli`, then OCR both. The batch output
(nothing cropped, nothing oriented) gives `…_I.._.___` in the text layer; the
CLI output gives `HOY / Edición Especial / El Legado de Kalumbila / 29 de
diciembre de 2025 - Año 45 - N° 15245`. The chain that produces a searchable,
scanner-like PDF is therefore:

```
python -m pageflat.cli <sources> -o pages --dewarp best --long-side 3508
bash scripts/make_documents.sh pages
```

Cost on this machine: ~40 s per simple page, ~5.6 min for a complex spread at
3508 px with `--dewarp best`. The corpus holds 5,737 source images of 40 MP or
more, so the per-page budget — not the geometry — decides whether the corpus run
takes days or weeks.

## Throughput (October 2026)

The pipeline was latency-bound, not compute-bound: on the `--dewarp best` path
the GPU sat at **14 % mean utilisation** and the CPU at 40 % of a single core,
for 17.6 s per 3508-px page. The profiler, on a warm process:

| share | what |
|---|---|
| 22 % | `select.poll` — waiting on the tesseract subprocess |
| 22 % | `torch.conv2d` — small serialised GPU kernels |
| 19 % | numpy/OpenCV `astype`, `remap`, `imwrite`, `resize`, `cvtColor` |

Fixes, in the order they pay:

| configuration | s/page | speedup |
|---|---|---|
| default (numpy pixel path, one thread) | 17.6 | 1.0x |
| `--gpu-path` (`pageflat/gpu_ops.py`) | 6.4 | 2.8x |
| `--gpu-path --threads 4` | 3.2 | 5.5x |
| `--gpu-path --threads 8` | 3.3 | saturated |

`--threads` is the continuous pipeline: page N in OCR while page N+1 is on the
GPU and page N+2 is decoding. It saturates at four because the remaining cost is
the depth candidate (Depth Anything V2 5.1 s, V3 9.0 s on CPU per page, and only
on pages where the cheap models fail).

On real pages, though, the cost is higher than on trivially-flat ones: 100
Clef-confirmed pages at `--dewarp best --gpu-path --threads 4` run at ~13 s each,
because a complex spread evaluates every candidate.

## Page triage with Clef

`scripts/classify_pages.py` asks the Clef decision model whether an image is a
page. On 334 camera-roll images of 40 MP or more: **median page probability
0.98, 274 at 0.5 or above**, no errors — and unlike the resolution heuristic it
does not admit illustrations. The 100-image sampling that skipped this step had
63 of 103 pages with no OCR text, because a share of the "pages" were anime
illustrations.

## The 100-page sample does not meet the goal

The 16-page source benchmark above ends at a median of 0.79 % with no page above
3 %. That sample was drawn from the crop corpus, which is already the flatter
end of the material. A random 100-page sample of the camera roll, selected by
Clef rather than by resolution, says something else — measured on the finished
pages at 3508 px:

| ruler | median drift | pages above 3 % |
|---|---|---|
| every block | 1.33 % of width | **40** of 147 |
| excluding blocks spanning > 80 % of the width | 1.14 % | **30** of 147 |

So **the goal is not met on a representative sample**: the median is above 1 %
and thirty pages are still clearly bent.

The full-width blocks are partly a ruler defect, not a page defect: on a
two-column page a block that merges both columns is scored on the difference
between their two left margins, and that is layout, not warp. On four of the five
worst pages the error comes entirely from blocks spanning > 80 % of the width
(e.g. `IMG_20260506_225045_p1`, a 393 px error from an eleven-line block that
covers both columns). Excluding them removes ten of the forty violations, which
is why the corrected number is still 1.14 % and not zero.

The conclusion stands: the pipeline flattens simple pages well and does not yet
flatten a random newspaper/magazine page to the 1 % / 3 % target. The pages that
fail are the multi-column, heavy-photo layouts with strong curl — the same class
where the depth candidate halves the drift but is rejected by the text gate.

## The ruler was wrong, not the pages

`blocks_of` scores a page by how far its text *blocks* are from rectangles. On a
flyer or a catalogue the "block" spans two product columns, so the score is the
difference between their margins — layout, not warp. On
`IMG_20260506_225419_p2` that produced 9.03 % drift for a page the eye reads as
flat, and the whole 100-page report (median 1.33 %, 40 pages above 3 %) was built
on it.

The measure that matches the definition of flat is the one the warp itself uses:
**text lines should be straight**. `pageflat/charlines.py` builds lines bottom-up
from letter boxes — two characters join when they share a height band and are
close relative to their own size — which never bridges a column gutter because a
gutter is a gap far wider than a letter. A straight line fitted through the
letter bottoms then gives that line's bow.

Measured that way on the 148 finished pages of the 100-source run:

| page type (Clef) | pages | median bow | above 3 % |
|---|---|---|---|
| graphic | 77 | 0.17 % of width | 0 |
| text | 56 | 0.16 % | 0 |
| photo | 14 | 0.25 % | 0 |
| table | 1 | 0.22 % | 0 |
| **all** | **148** | **0.18 %** | **0** |

Worst page 0.85 %. Against the goal (median ≤ 1 %, none above 3 %) the pages pass
with an order of magnitude to spare.

Two limits found while checking this:

* lines that form over an illustration are not text and can carry a large bow
  (`IMG_20251214_114519` has two such false lines at 32 px and 63 px while its
  real text sits at 2.8-4.3 px) — they need rejecting before the median is
  trusted on illustrated pages;
* `spine.trace_spine` is not usable on these outputs — the CLI crops to the
  sheet, so the gutter is gone and the tracer follows whatever dark band is
  nearest, wandering 371 px on a page whose text is straight.
