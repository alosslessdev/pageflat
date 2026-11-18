# The fix loop

State for whoever picks this up (including a later me): the goal, the ruler, the commands,
the numbers so far, and what is still broken.

## Goal

Pages photographed with a phone should look like flatbed scans and survive OCR. The
measurable target on the 100-source benchmark:

| metric | target | meaning |
|---|---|---|
| `bow` median | <= 1 % of width | text lines are straight (the flattening goal) |
| `bow` above 3 % | 0 pages | nothing obviously bent |
| `tilt` median \|deg\| | <= 0.5 | vertical structure (rules, column edges) stands up |
| `tilt` above 1 deg | 0 pages | |
| `band` > 0.5 | 0 pages | no strip of table / neighbour at the frame edge |
| `cut_lines` > 0 | 0 pages | no text line interrupted by the frame |
| `rot` != 0 | 0 pages | page is upright |

`scripts/score_pages.py <dir>` prints exactly that table. **Everything is judged there**,
not by argument.

## The benchmark set

`~/pagedewarp/work/pages100/` — 100 Clef-confirmed page photographs (≥ 98 % page
probability) drawn from the camera roll, symlinks to the originals.

## Commands

```bash
# rebuild the benchmark output (about 25-40 min at 4 threads on this machine)
cd ~/pagedewarp/repo
TMPDIR=~/pagedewarp/tmp2 ../.venv/bin/python -m pageflat.cli \
    ~/pagedewarp/work/pages100 -o ~/pagedewarp/work/outN \
    --dewarp best --long-side 3508 --gpu-path --threads 4

# score it
TMPDIR=~/pagedewarp/tmp2 ../.venv/bin/python scripts/score_pages.py \
    ~/pagedewarp/work/outN --json ~/pagedewarp/work/scoreN.json
```

Do not run the corpus job and an iteration at the same time: both want the 5.7 GB card and
the iteration fails with a CUDA OOM that looks like a code bug.

## Numbers

| run | pages | bow med | bow >3 % | tilt med | tilt >1° | band >0.5 | cut lines | misoriented |
|---|---|---|---|---|---|---|---|---|
| out100_v5 (baseline) | 153 | 0.05 % | 0 | 1.36° | 89 | 48 | 69 | 25 (at 1400 px) |
| **it1** (tilt rotation + split guard) | 127 | **0.04 %** | **0** | **1.13°** | **64** | **45** | **46** | **5** (at 1600 px) |
| **it2** (tilt twice, with per-row shear) | 130 | **0.04 %** | **0** | **1.10°** (p90 3.64) | **66** | **47** | **49** | **4** |
| **it3** (two vertical passes at save) | 128 | **0.04 %** | **0** | 1.17° (p90 3.87) | 68 | **39** | **46** | **3** |
| **it4** (edge shear with the sign fixed, tilt off) | 129 | **0.04 %** | **0** | 1.15° (p90 3.74) | 65 | 43 | **43** | 7 |

The page count drops from 153 to 127 because the split guard refuses cuts that land inside a
page: fewer half-pages, and the first is a complete spread instead of two damaged halves.
The `misoriented` row is only comparable when measured at the same scale — the scorer now
uses the pipeline's own 1600 px / INTER_LINEAR.

Iteration 1 changed three things and each is visible: tilt median 1.36 -> 1.13 and pages
above 1 deg 89 -> 64 (the rotation); cut lines 69 -> 46 (the split guard); band 48 -> 45.

**Iteration 2 (in progress in this transcript)**: the vertical pass now runs on the finished
page and includes the per-row shear, and the shear's detection was moved onto a 1400-px copy
because `stroke_mask`'s median filter on a 3508-px page cost tens of seconds per page (which
is why an iteration slowed to 35-66 s/page). Measured on it1's output over 57 pages,
`rectify(shear=True)` takes the median tilt 1.29 -> 0.95 deg, p90 3.19 -> 2.71, pages above
1 deg 37 -> 27, where a second rotation alone left p90 worse.

Smoke test of the whole pipeline on the page that measured 4.58 deg: now 0.70 deg.

**it2 result: tilt 1.13 -> 1.10, p90 3.79 -> 3.64.** Much less than the 1.29 -> 0.95 that the
same correction achieved on it1's finished pages. The reason is visible in the numbers: the
pipeline already applies the rotation before the split, so the second pass has only the
residual bow left, and a rotation plus a rule-driven shear cannot remove a bow that is not
in the rules. **That hypothesis was wrong, and the ruler now proves it.** `vertical_drift` reports lean
and bow separately: on it2 the vertical **bow is 0.05 % median, p90 0.11 %** — already
negligible — while the **lean is 1.10 deg** with 66 pages above 1 deg. So what is left is
rotation, not bow, and the rotation just converges slowly: over 45 finished pages the median
goes 1.15 -> 0.87 -> 0.79 deg across two extra passes, pages above 1 deg 24 -> 20 -> 14. Two
passes were then tried in a full run (it3) and **did not survive it**: median tilt 1.17 against
1.10 with one pass, at twice the runtime, while the background band improved to 39 and cut
lines to 46. The second pass is therefore reverted — a dead end, recorded so it is not
retried.

## What is fixed

* **flattening** — text-line bow, bottom-up line grouping with artwork rejected.
* **orientation** — checked on the finished page with the OCR-scored detector, resampled
  INTER_LINEAR at 1600 px (the interpolation decides the verdict).
* **edge trim** — walks in while the band is not paper, bounded to 15 %, undone when it
  costs OCR. Works in **both** the CPU and GPU paths.
* **split** — refused when a detected line's box spans the cut (a test that words cannot
  do: two smaller halves OCR *better* than one page).

## What is open, in order of evidence

1. **Vertical tilt** — median 1.36°, p90 4.0°, 89 pages above 1°. `vlines.rectify` removes
   the common tilt as a rotation (calibrated: rotating +2° CCW raises the measure by ~2, so
   apply `-tilt`) and took 1.70 → 1.11 on 21 pages. **Not wired into the pipeline yet.**
2. **Left-edge curl** — the sheet's own boundary bows; on
   `IMG_20260506_224359` the shear needed is up to 95 px. `edges.straighten_edge` is wired
   into both paths but has never been run end to end (the test hit a CUDA OOM caused by the
   corpus job, and the run was rejected).
3. **Residual band** — 48 pages; mostly the page's own coloured border, but the metric
   cannot tell that from real background.
4. **Worked examples that must not regress**: `IMG_20260506_224359` (spread, must stay whole
   and uncut), `IMG_20260506_225101_p1` (cut column), `IMG_20260506_224505_p2` (tilt 4.6°).

## Dead ends, recorded so they are not retried

* **Clef for fine spatial judgements** — asked whether a scan is a spread or a single page
  it called a visible two-page catalogue "single" at 0.77, and a half-page "spread" at 0.96.
  `page_type` (text / graphic / photo / table) *is* reliable.
* **Clef for sheet-edge visibility** — 2 of 4 right at 640 px.
* **Word counts as a split guard** — splitting increases OCR words.
* **SAM with a single point or a box** — segments one page, or a leaf; a 3×3 grid inside a
  coarse mask finds the whole sheet at 0.98.
* **`skew_angle` on graphic pages** — reports 0.00° exactly where the page is tilted.

### Dead end: a Hough estimate of the lean

Tried next: one angle from every vertical stroke at once, weighted by stroke length, so that
re-detecting a different rule set after a rotation could not move the answer. It gives a
**median of 7.45 deg over 40 pages** — absurd, because "vertical-ish" accepts every slanted
product photo and diagonal design element on a flyer, and the weighted median then lands on
them. Tightening the acceptance angle would bias the estimate toward the answer being
measured. Removed rather than tuned.

The stable reference is the **page's own long boundary** (the left or right edge), which
`edges.straighten_edge` already traces: one line, unambiguous, and its slope is a lean
measurement that cannot be moved by which internal strokes get detected.

### The lean plateau was mostly the ruler, and the correction was making it worse

Two independent measures of the same thing, on 35 pages of it3:

| measure | median |
|---|---|
| rule-based tilt (`vertical_rules` median slope) | **1.03 deg** |
| the page's own boundary slope (`_paper_like` edge fit) | **0.18 deg** |

Correlation between them: **-0.05** — none. The rules are measuring the slope of *printed
design* (product panels, banners, diagonals), not the lean of the sheet. Rotating by that
median therefore squares up the content and tilts the paper: measured over 30 pages the
page-edge slope went **0.26 -> 0.34 deg**, worse on 13 pages and better on 6.

So `fix_tilt` is **off by default**. The page's own boundary is the honest lean reference; a
correction driven by it is the next thing to build, and it needs the edge to be inside the
frame, which is exactly the case `edges.py` already refuses when it is not.

This also explains the whole plateau: it1 -> it2 -> it3 moved the rule-tilt by tenths of a
degree while the page was already square to 0.18 deg. The number that *should* be in the
target table is the edge slope, not the rule tilt.

### The edge shear works: the sign was inverted

`straighten_edge` measured the displacement as (edge - straight) and then remapped with
`gx - dx`, which moved every row *away* from its straight fit and doubled the curvature:
left-edge deviation 9.0 -> 19.6 px. With `gx + dx` the same page goes **9.0 -> 0.3 px**, and
the visible result is a straight, vertical left edge where the green side band used to bow
inward at the top.

Same class of bug as the tilt rotation, and the same lesson: a displacement measured against
a reference has to be applied *towards* it, and that direction has to be checked against a
known case rather than assumed.

**it4** runs it in the pipeline: cut lines reach their best value (43, from 69 at baseline)
and the band improves to 43. The rule-based tilt is now reported for what it is — content
design, not page lean — and `fix_tilt` is off.

### The orientation regression in it4: 7 pages, two different causes

it4 has 7 misoriented pages against 2 at baseline. Probing each one at the pipeline's own
scale (1600 px, INTER_LINEAR):

| page | detector says | after rotating by that | verdict |
|---|---|---|---|
| `IMG_20260129_202334` | 180 | 0 | **pipeline miss** |
| `IMG_20260506_224356` | 180 | 0 | **pipeline miss** |
| `IMG_20260506_224646` | 180 | 0 | **pipeline miss** |
| `IMG_20260506_225304_p1` | 90 | 0 | **pipeline miss** |
| `IMG_20260506_224606` | 180 | 180 | detector contradicts itself |
| `IMG_20260506_224643` | 180 | 180 | detector contradicts itself |
| `IMG_20260506_225358_p2` | 180 | 180 | detector contradicts itself |

So four are a real pipeline miss to chase — the same 1600 px / INTER_LINEAR check that is
supposed to run at save time did not turn them — and three are cases where the OCR-scored
detector answers 180 for the page *and* for its 180-degree rotation, which no threshold can
fix. The detector needs a second opinion that is not OCR-scored: the page's text baseline
direction from `charlines` (the lines are horizontal in an upright page and their ink
profile is asymmetric vertically — ascenders above the baseline, descenders below).

### The four orientation misses: stage order

Fixed. The check ran at save time, after the edge shear and the trim — and those change the
pixels the OCR-scored detector reads, so on four pages it saw an "upright" page and left it
alone. Running the same check on the rectified sheet, **before** the shear and the trim, turns
all four: `IMG_20260129_202334`, `IMG_20260506_224356`, `IMG_20260506_224646` and
`IMG_20260506_225304_p1` come out at `needs 0` on a re-run, where they were at 180/90.

The save-time check is kept as well — it is the one that cannot disagree with the pixels that
get written — but it is no longer the only one.

### The three contradictory pages: ask a different model

Also fixed. When the OCR-scored detector returns the same angle for a page *and* for that
page rotated by it, the answer cannot be acted on; the code now asks docTR's page-orientation
classifier instead, and if that disagrees it uses it, otherwise it leaves the page alone
rather than rotating on an answer the detector itself contradicts. All three —
`IMG_20260506_224606`, `IMG_20260506_224643`, `IMG_20260506_225358` — come out at `needs 0`.

### The cut metric was counting its own seam

`cut_lines` counted a line touching **any** frame side, but when a spread is split the seam is
a border the pipeline just invented: a line touching it is not cut, the page continues into
the other half. 21 of the 43 "cut" pages were such halves. Excluding the seam side (right for
`_p1`, left for `_p2`) drops the count to **37**, and the pages that matter are the **22
whole** ones where the frame is the photograph's own border. Worst unexamined: `225734` (6),
`225042` (6), `224606` (4 — and also an orientation-contradiction page, so possibly a
rotation consequence rather than a crop).

### Why `IMG_20251229_122443` keeps a strip at the bottom: the mask over-reaches

The page looks right except for a band of table along the bottom. It is not the trim's fault —
the trim cannot see it, because a pale table is *paper-like* (bottom band non-paper: 0.00 at
3 %, 0.00 at 6 %, 0.02 at 10 %). The fault is upstream, in the mask that defines the quad:

| mask | coverage | vertical extent (frame is 900 px) |
|---|---|---|
| docTR page mask | 0.831 | y 7 – **893** |
| SAM sheet mask | 0.501 | y 0 – **780** |
| printed area alone | — | ink stops at **~790** |

The page's own bottom edge is around 800; docTR's mask runs ~90-110 px past it onto the table,
and the homography then maps that table into the output. The print gives the decisive clue:
rows 700-740 carry 254-343 ink pixels, and by row 800 it is 34 and falling to 0 — so the sheet
ends where the ink ends plus a margin, and the mask does not.

**The fix belongs in segmentation, not in cropping**: bound the mask where the print stops.
A cheap, checkable version is to find the strongest horizontal edge below the last ink row —
the paper-to-table boundary — and use that as the mask's bottom.

### The mask fix that works: SAM, guarded by the text lines

Six single-signal rules were tried and all failed on `IMG_20251229_122443` — the page whose
bottom keeps a strip of table. What works is asking a second segmenter:

| witness | bottom edge (frame 900 px) |
|---|---|
| learned page mask (docTR) | **893** — past the sheet |
| SAM, grid-prompted | **780** |
| an independent vision model, 0-1000 grid | **867** → 780 in pixels |
| print ends | ~790 |

SAM and a vision model agree; the learned mask does not. And SAM's weakness is the opposite
one — it can be *tighter than the page* (its left boundary is x=127 against the sheet's x=12),
so the mask is grown until no detected text line is cut by it (`segment.extend_to_lines`).

Wired behind `--sam`. On the page that motivated it the output no longer shows table at the
bottom: what reaches the bottom edge is the magazine's own yellow band. Note that the
`_paper_like` band metric now reads *worse* (0.05 non-paper against 0.00) because a pale table
counts as paper and the yellow band does not — the metric cannot judge this fix, which is why
the mask extents and the vision model are the evidence.

The six failures, for the record: `printed_mask` (full frame), dark ink (catches the shadow),
row-mean brightness step (invisible: 140-153 across the edge), strongest gradient (lands on
content), outermost gradient (lands on page borders), per-side minimum of the two masks (cuts
115 px off the sheet).
