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
