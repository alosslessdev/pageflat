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

After rejecting lines built over artwork — a line must have a neighbour within a few
line heights over the same horizontal stretch, because type is set in blocks and a
drawing stroke is isolated — the same 143 measurable pages give:

| type (Clef) | pages | median bow | above 3 % |
|---|---|---|---|
| graphic | 75 | 0.15 % | 0 |
| text | 54 | 0.14 % | 0 |
| photo | 13 | 0.21 % | 0 |
| table | 1 | 0.28 % | 0 |
| **all** | **143** | **0.15 %** | **0** |

Five pages have no text-line block at all (pure photographs) and are not measurable
this way.

**Edge-based measurement is sound in principle and currently blocked by segmentation.**
`pageflat/edges.py` traces each sheet edge and reports its deviation from a straight line,
and it correctly refuses (returns None) when the sheet has been cropped to its content and
the "edge" would be the frame. On these photographs it cannot yet be trusted, because the
mask includes the background: a newspaper on red bedding takes the bedding into the page
mask (3126 px on the right edge before `_paper_like`), and a botanical book on a light
duvet keeps the duvet even after it (648 px). Light, neutral bedding and paper are the
same colour to every heuristic here, so the prerequisite is a trustworthy sheet mask —
the SAM path (`--sam`) exists for exactly that.

This is also why "are the sheet edges measurable at all" belongs in the classifier: the
edge ruler is only worth running on frames where the sheet is properly isolated.

Worst measurable page 1.35 %. Against the goal (median ≤ 1 %, none above 3 %) the pages pass
with an order of magnitude to spare.

Two limits found while checking this:

* lines that form over an illustration are not text and can carry a large bow
  (`IMG_20251214_114519` has two such false lines at 32 px and 63 px while its
  real text sits at 2.8-4.3 px) — they need rejecting before the median is
  trusted on illustrated pages;
* `spine.trace_spine` is not usable on these outputs — the CLI crops to the
  sheet, so the gutter is gone and the tracer follows whatever dark band is
  nearest, wandering 371 px on a page whose text is straight.

## What the router can and cannot decide

`--question page_type` (text / graphic / photo / table) is reliable enough to drive
the measure: on the 148 finished pages it split them 77 graphic, 56 text, 14 photo, 1
table, and the drift violations under the block ruler concentrated exactly in the graphic
class — the class where that ruler is invalid.

`--question sheet` (isolated / cluttered / fills) was tried for the same job on the edge
ruler and **is not reliable at this model size**: asked about four known frames at 640 px
it got two right. A book lying on a duvet was called "isolated" at 0.72, and a page cropped
to its content was called "isolated" rather than "fills". Fine visual judgements about
which of a sheet's four edges are visible are beyond a 4-bit decision model, so the
question was removed rather than shipped as a gate.

The edge ruler needed a trustworthy sheet mask, and **SAM was not the problem — the prompt
and the selection were.** Asked with one point, SAM answers "the object under this point":
on an open book that is one page (25 % of the frame) or a leaf on it (1 %, 0.3 %), and
`SamPageSegmenter.mask` deliberately keeps the *smallest* confident candidate, which is right
for picking a page out of a pile and wrong for finding the sheet. Asked with a 3x3 grid of
points spread inside the coarse mask, the sheet comes back at score 0.98 covering 58 % of
the frame — both pages, bedding excluded (`SamPageSegmenter.sheet_mask`).

With that mask, edge drift on the raw photos reads 13-25 % of width, which is the physical
drape of a newspaper over a duvet rather than a measurement artefact, and a scan cropped to
its content still refuses to answer.

## The goal, with the classifier choosing the ruler

Neither ruler covers every page, and each is invalid where the other applies. Routed by
`--question page_type` and by whether a text grid exists at all:

| ruler | pages | median | above 3 % |
|---|---|---|---|
| text-line bow (pages with >= 5 lines) | 143 | 0.15 % of width | 0 |
| sheet-edge drift (pages with < 5 lines) | 10 measured, 2 refused | 0.1-0.7 % for nine of them | 2 (3.0 %, 7.1 %) |

So the flattening goal is met on every page that has text to measure, and on nine of the
ten text-less pages that could be measured at all. The two exceptions are not flattening
failures: `IMG_20260506_225318_p1` is **upside down** with a strip of floor left in the
crop, and its 7.1 % edge number is that strip. The remaining defect class in this corpus is
therefore orientation and cropping, not geometry.

Throughput, for the same run: `--gpu-path --threads 4` finishes 100 pages in 412 s
(4.1 s/page) with the GPU at 80-90 %, against 17.6 s/page and a 14 % GPU for the default
configuration — the pixel stages were the bottleneck, not the models.

## Cropping: the strip that survived the crop

`crop_to_content` works from where the *print* is, so a strip of table, floor or the
neighbouring page survives wherever the print stops short of the paper's edge. On the
100-source run **102 of 148 pages carried a background band over half of the frame edge**,
most of them at the bottom.

`crop.trim_borders` asks the opposite question — is the outermost line of pixels paper? — and
walks in from each side while the band is mostly not paper, bounded to 15 % of the dimension
so a dark picture cannot eat a page. A trim that costs OCR is undone, the same rule the warp
candidates are held to.

| | before | after |
|---|---|---|
| pages with a > 50 % background band | 102 | **32** |
| pages trimmed | — | 117 |
| trims reverted because they cost text | — | 7 |

On `IMG_20260506_224517_p1` the trim also *raised* OCR from 73.7 to 82.8, because the dark
strip was pulling the recognition down. It runs per page after the spread split, so each
half is trimmed on its own edges.

## Orientation

34 of the 148 finished pages were sideways or upside down (24 at 180 degrees, 6 at 270, 4 at
90), and the pipeline had reported `rotation: 0` for them: the **GPU path** decides
orientation on the pre-warp sheet, which is not the image that gets written. The CPU path
was right; `--gpu-path` was wrong, which is why the batch run carried the defect.

The fix runs the check on the finished page, just before it is written — and uses the
**OCR-scored** detector rather than docTR's thumbnail classifier. The classifier is
confident and wrong on half a spread: on `IMG_20260506_224449` it called both halves upside
down, and rotating on that answer left one of them wrong. Requiring the two detectors to
agree did not help, because on that half they agreed and were both wrong. The OCR-scored
detector reads the text, costs about 2 s, and got 5 of 5 pages right on the re-run.

## The three fixes, measured together

A fresh run of the same 100 Clef-selected sources with the edit trim and the save-time
orientation (150 pages, 100 accepted):

| defect | before | after |
|---|---|---|
| pages misoriented | 34 / 148 (23 %) | **15 / 150 (10 %)** |
| pages with a > 50 % background band | 102 / 148 (69 %) | **50 / 150 (33 %)** |
| line bow, median | 0.15 % | **0.05 %** |
| line bow, pages above 3 % | 0 | **0** (110 measurable, 40 without a text grid) |

So the flattening is unchanged where it was already good and the median improved 3x, the
crop defect roughly halved, and orientation improved by more than half but is not clean.

**What is still wrong, and why.** Fifteen pages keep a rotation the detector did not ask
for. `orientation.detect_rotation` scores four rotations with tesseract, and on a flyer
whose panels are graphic rather than prose it can prefer a wrong rotation with confidence;
its answer also changes with the scale it is asked at (1200 px got two of five test pages
wrong, 1600 px none), so the current 1600 px is a tuned constant, not a principle. Fixing
this properly needs an orientation model trained on this material rather than OCR scoring.

The remaining 50 background bands are mostly the pages' own coloured borders, which the
trim correctly refuses to cut; a minority is background wider than the 15 % bound.

## Final verification of all three defects (clean run)

Same 100 Clef-selected sources, 100 accepted, 153 pages:

| defect | first run | final run |
|---|---|---|
| pages misoriented | 34 / 148 (23 %) | **2 / 153 (1.3 %)** |
| pages with a > 50 % background band | 102 / 148 (69 %) | **48 / 153 (31 %)** |
| line bow, median | 0.15 % | **0.05 %** |
| line bow, pages above 3 % | 0 | **0** (112 measurable, 41 without a text grid) |

Two of the fixes turned on details that are worth keeping in mind, because each looked
finished before it was:

* the **interpolation** decides the orientation verdict. `detect_rotation` scores OCR of
  four rotations, and resampling the page with INTER_AREA instead of INTER_LINEAR was
  enough to flip a 180-degree page to "upright". The pipeline now resamples the way the
  measurement does.
* **the disk matters**. A run with `/tmp` at 99 % produced 326 `libpng error: Write Error`
  lines and still reported pages; the numbers above come from a rerun with 5.7 GB free.

The 48 remaining background bands are mostly the pages' own coloured borders, which the
trim correctly refuses to cut.

## The corpus run

Launched over all 5,737 source images of 40 MP or more, output to
`/run/media/asdf/db69437b-.../corpus_pages` with `--dewarp best --long-side 3508
--gpu-path --threads 4`, progress in `corpus_pages/run.log` and a JSON report alongside.
At the observed 14 s/page on the heavier pages this is a ~20 h job; the pipeline is at
80-90 % GPU in its busy phases and the disk has 165 GB free for roughly 80 GB of output.

Clef triage of all 5,737 was started and **abandoned**: at 0.19-0.35 images/s it needed
8 h to decide what the pipeline's own content gate decides for free while it works, and
it holds the GPU. The page/not-page question is worth a classifier when the classifier is
cheap; here it is the expensive part.

### Handoff for the corpus run

When `corpus_pages` finishes (or is interrupted and resumed), the rest of the chain is:

```
# resume the flatten from wherever it stopped
python -m pageflat.cli --list ~/pagedewarp/work/corpus_inventory.txt \
    -o /run/media/asdf/db69437b-.../corpus_pages --dewarp best --long-side 3508 \
    --gpu-path --threads 4 --resume

# OCR -> document grouping -> searchable PDFs (verified on a 147-page set,
# 1 unreadable page, real text layer)
PATH=~/pagedewarp/.venv/bin:$PATH bash scripts/make_documents.sh \
    /run/media/asdf/db69437b-.../corpus_pages 4
```

`make_documents.sh` needs the venv on PATH (it shells out to python) and reads `.png`
as well as `.jpg`.

## Vertical warping: measured, not yet solved

The text-line ruler measures horizontal geometry only, so a page that is *tilted* leaves its
bow at zero while looking crooked — the defect that remains on flyers. `pageflat/vlines.py`
builds the missing reference: vertical strokes (column rules, panel borders) found with a
vertical opening, grouped into rules, each fitted and measured against a straight vertical.

On the 153-page set, 144 pages have vertical structure:

| | |
|---|---|
| median \|tilt\| | **1.36 deg** |
| p90 | 4.00 deg |
| max | 9.50 deg |
| pages above 1 deg | **89** |
| pages above 2 deg | 50 |

`skew_angle` reports 0.0 deg on some of those pages, which is why the tilt survived: the
projection-profile method finds no signal on a graphic page.

The correction is calibrated against known rotations — rotating a page +2 deg CCW raises the
measured tilt by 1.7-2.0, so the angle to apply is -tilt. Rotation alone takes the median
from 1.70 to 1.11 deg on a 21-page sample (17 -> 12 pages above 1 deg). The per-row shear
that would remove the residual bow is implemented but **off by default**: it was not
validated against a known deformation, and an unvalidated correction is worse than none.

## Cropping: two defects, not one

The strip measure counts non-paper at the frame edge, which cannot tell "background left in"
from "content cut off". Looking at the pages it flags shows both:

* **background left in** — the 102 -> 48 improvement;
* **content cut at the frame** — `IMG_20260506_224359_p1` has the neighbouring page's prices
  clipped at its right edge, and `IMG_20260506_225101_p1` a column cut mid-word. Neither is
  visible to the strip measure, and both come from the spread split landing inside a page
  rather than on the gutter.

So the cropping goal is **not met**: roughly a third of the sampled pages still carry a
border defect, and the majority of those are cuts rather than leftovers.

### The split guard that actually works

Counting words does not catch a bad split — two smaller halves OCR *better* than one big
page (62 words became 121 on `IMG_20260506_224359`), so the cut looked like a win. The test
that works is geometric: **a cut is wrong when a detected line's box spans it.** On that
page the proposed cut at x=1697 crossed two to three lines and the split is refused; the
whole spread comes out at 3508x2155, uncut, and the catalogue pages are intact where before
each half lost a column.

Clef was asked to arbitrate the same question ("spread or single?") and **is not reliable
for it**: shown a photographically obvious two-page catalogue spread it answered "single"
at 0.77, and called one of its halves "spread" at 0.96. The question was removed rather
than shipped as a gate, for the same reason as the sheet-visibility question: coarse
content type is within this classifier, fine spatial judgements are not.
