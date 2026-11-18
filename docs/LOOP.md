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

### The SAM mask fixes its case but is not yet a general win

Measured on five benchmark sources, three configurations (band and cut-lines per output):

| page | docTR mask (default) | SAM alone | docTR ∩ SAM + lines |
|---|---|---|---|
| `122456` | 0.21 / **0** | 0.28 / 5 | 0.31 / **0** |
| `122504` | — | 0.27 / 1 | 0.18 / 5 |
| `122507_p1` | — | 0.03 / 0 | 0.03 / 0 |
| `122507_p2` | — | 0.14 / 4 | 0.14 / 4 |
| `122517_p1` | 0.37 / 0 | 0.34 / 0 | 0.34 / 0 |
| `122517_p2` | 0.40 / 2 | 0.37 / 0 | 0.37 / 0 |
| `122443` | 0.20 / 1 | accepted | **rejected by the quality gate** |

So each rule trades one failure for another: SAM alone clips text on `122456`, the
intersection cuts too much on `122504` and rejects `122443` outright. The mask is behind
`--sam` and **off by default**, because "better on the page that motivated it" is not the
same as "better".

What the numbers do establish: the bottom edge on `122443` really is at 780 of 900 (SAM and an
independent vision model agree), so the learned mask's 893 is wrong — the remaining work is
per-page arbitration by measured outcome (cut lines and acceptance), not another fixed rule.

### Per-page arbitration beats both fixed rules

Choosing the mask by measured outcome — warp both candidates, count the text lines each one
cuts, keep the better — on the same five sources:

| page | default mask | SAM alone | intersection | **arbitrated** |
|---|---|---|---|---|
| `122443` | 1 | accepted | rejected | **0** |
| `122456` | 0 | 5 | 0 | **0** |
| `122507_p1` | — | 0 | 0 | **0** |
| `122507_p2` | — | 4 | 4 | 4 |
| `122517_p1` | 0 | 0 | 0 | **0** |
| `122517_p2` | 2 | 0 | 0 | **0** |

Five of six outputs cut nothing, and the page that motivated the whole exercise goes from one
cut line and a strip of table to zero cuts. `122504` reported `error` in the threaded batch and
runs clean alone, which is the concurrency/OOM pattern seen before rather than a logic fault.

Still behind `--sam`: it costs one extra model pass per page, and the arbitration doubles the
warp for those pages. A full benchmark run with it is the next validation, and the number to
beat is the default's 37 cut pages.

### The SAM mask does not generalise: a full run says no

The five-page sample said yes; the whole benchmark says no. Same 100 sources, `--sam` with the
per-page arbitration:

| run | accepted | pages | bow med | tilt med | band | **cut lines** | misoriented |
|---|---|---|---|---|---|---|---|
| it4 (default mask) | 100/100 | 129 | 0.04 % | 1.15° | 43 | **37** | 7 |
| **it5 (`--sam` + arbitration)** | **95/100** | 120 | 0.04 % | 1.92° | 38 | **52** | **12** |

Worse on everything that matters: five pages rejected that the default accepts, fifteen more
pages with cut lines, five more misoriented, and the tilt worse by two thirds. The smaller page
set also means the two runs are not scoring the same pages, so even the band improvement is not
comparable.

Why the sample lied: on those five sources SAM's boundary was the right one. Across the corpus
SAM under-segments graphic pages, and the line-growth guard cannot repair that because
`charlines` has nothing to detect on a flyer — the same reason 28 pages have no text grid at all.

**Not adopted.** `--sam` stays opt-in and off; the default mask (it4) is the better configuration
on every metric. This is the third fix in this loop that looked right on its motivating example
and failed at scale — the benchmark, not the example, is the judge.

### Two orientation passes are worse than one: reverted

it6 ran the benchmark with the orientation check both on the rectified sheet and at save time.
It measured **13 of 128 misoriented** against it4's **7 of 129** with the save-time pass alone —
the two passes rotate some pages twice, because the shear, the trim and the deskew change the
pixels the detector reads in between. The early pass is removed again; the save-time pass is the
single authority, and on the nine hard pages (the four it had fixed, the three contradictions,
and two halves) it gives **0/9 misoriented**.

The deskew-at-save fix (`90ed125`) is kept: it is a different correction (fine angle, not 90°
multiples) and it was measured on its own metric, 9.5° → 0.4°.

**Current configuration, all measured**: it4's numbers are the reference (cut 37, misoriented 7,
band 43, bow 0.04 %), with skew now corrected on the finished page. The last full run in this
session, it6, measured a configuration that no longer exists.

The lesson repeats what the loop already showed three times: a change that fixes its sample can
lose at scale, and the only way to know is the 100-page run — which is also why the run has to be
affordable, and why the arbitration path (one extra model pass and a second warp per page) was
not adopted for a two-point gain that did not appear.

### The orientation metric is fooled by mirrored design, and so is the correction

it7 (the shipped configuration) scores 15 of 129 "misoriented", it4 scored 7 and it6 13, with
code changes far too small to explain a doubling. Looking at the pages the scorer flags:

| page | what it actually is |
|---|---|
| `IMG_20260506_224512` | a flyer with a **mirrored advertising banner** along the bottom |
| `IMG_20260506_224606` | same: a reversed `DIAaDIA BLACKBERRY` strip |
| `IMG_20260506_224643` | header and body upright and readable |

The pages are upright. The OCR-scored detector votes for a rotation because the mirrored banner
"reads" better upside down — and since the pipeline uses the same detector to decide, it is both
a bad metric and a bad corrector on graphic pages. That explains the run-to-run drift: every
stage that changes the pixels changes which side of the vote wins.

**Second check, and it corrects the first one.** Three of the flagged pages were rendered and
looked at, then three more:

| page | what the eye says |
|---|---|
| `224512`, `224606` | upright; a mirrored banner explains the vote |
| `224643` | upright and readable |
| `202334` | text runs vertically — **genuinely rotated** |
| `224123` | headline "Los mensajes ocultos de los eclipses de Sol" runs vertically — **genuinely rotated** |
| `224218` | a flyer with mirrored product rows — mixed |

So the flag is **not** mostly false: two of six are plainly misoriented and would be fixed by a
rotation. The honest statement is that the automatic verdict mixes real misses with mirrored-design
false positives at an unknown ratio, and two spot checks of three pages each produced opposite
impressions — which is the strongest possible argument that the next step must be a hand-labelled
sample, not another rule change.

**Consequence:** the orientation count cannot be tuned against, and rotating on it can make an
upright page worse. The robust rule is narrower than what is implemented — rotate only when the
page has a text grid *and* the fast classifier agrees with the OCR score — and everything else
should be left alone. That is the next change to make, and it needs a hand-labelled orientation
sample to measure against, because the automatic verdict is not trustworthy.

### The narrow rule, shipped, with its cost stated

Orientation now rotates only when **both** hold: the page has a text grid (>= 5 detected lines)
and docTR's fast classifier agrees with the OCR score. Rationale, measured above: a flyer with a
mirrored banner wins the OCR vote while being upright.

Results on the nine hard pages: **7 upright, 1 correctly left alone, 1 known miss.**

* correctly left alone — `IMG_20260506_224606`: the metric says 180, the page is upright, and its
  mirrored `DIAaDIA BLACKBERRY` banner is the reason.
* **known miss** — `IMG_20260506_224356`: genuinely upside down (the `Canon` wordmark is inverted
  in the render) and now left as it is, because it has fewer than five detected lines.

So the rule trades one real miss for three prevented false rotations. That is the right way round
— rotating an upright page destroys it, while leaving an upside-down one is visible and fixable —
but it is a trade, not a fix, and it cannot be closed without a hand-labelled orientation sample.

### The labelling attempt, and why it did not settle anything

Six flagged pages were rendered in montages at ~340 px and labelled by eye: two looked rotated,
four upright. Then the OCR scores behind each verdict were printed:

| page | my label | score 0 | 90 | 180 | 270 | what OCR prefers |
|---|---|---|---|---|---|---|
| `202334` | rotated | 249 | 0 | **442** | 0 | 180 |
| `224123` | rotated | **2953** | 0 | 190 | 0 | **upright, by 15x** |
| `224218` | upright | 173 | 336 | **1464** | 211 | 180 |
| `224512` | upright | 0 | 0 | **181** | 0 | 180 |
| `224551` | upright | **1845** | 0 | 0 | 0 | upright |
| `224633` | upright | 0 | 0 | 0 | **568** | 270 |

`224123` is the case that matters: I called it rotated, and its upright OCR score beats the best
rotation by **15x**. Either the page is upright and I misread a vertical design element at 340 px,
or OCR finds 2953 units of text in a rotated page — the first is far more likely.

So **the labels are unreliable at that scale**, not just the metric. Three attempts at this now —
metric, correction, and my own eye at 340 px — have disagreed with each other, and the only route
left is one page at a time at a readable size, with the OCR scores printed alongside so a
disagreement is visible rather than silent.

What the table *does* establish: on four of six pages every rotation scores near zero except one,
which means those pages carry almost no readable text either way and the vote among 0/90/180/270
is decided by a handful of OCR fragments. That is not a measurement to tune against.

### Orientation: the vision model is right where OCR is blind

The page that broke every automatic attempt is `IMG_20260506_224512`, a Conair flyer whose text
is product labels and prices. Its four-rotation OCR scores are **0:0, 90:0, 180:181, 270:0** — a
vote cast by a handful of fragments — so:

| judge | verdict | correct? |
|---|---|---|
| OCR-scored detector | 180 | **no** |
| my eye at 340 px, then at 900 px | upright | **no** |
| **you** | top on the left (rotate 90 clockwise) | **yes** |
| **a vision model** | `rotate_cw_90`, confidence 0.96 | **yes** |

The model's evidence: *"the `infiniti pro by CONAIR` wordmark runs horizontally across the top with
its letters right-side up, price tags read left-to-right, the model's face and `Curl Secret`
headline sit at the top"* — it rotated the page mentally and read it, which is exactly what OCR
scoring cannot do when the page has almost no prose.

**What shipped instead:** rotation must now be *earned* — the winning rotation needs an absolute
score of 500 and twice the upright score. That fixes the three mirrored-banner pages (3892 vs 394,
867 vs 253, 1387 vs 399 all now stay upright, and they are upright) and leaves `224512` alone,
which is a miss rather than damage — the right way to be wrong.

**The escalation is now obvious and costed:** a vision call per page decides what OCR cannot,
at ~40 s each. For the benchmark that is fine; for 5,737 pages it is ~63 hours of model time, so
it belongs behind a flag, applied to the pages the cheap rule refuses.

### The low-text class is 15 % of the benchmark, and a vision model gets it right

`max OCR score < 500` over all four rotations — the pages where the vote is noise — is **9 of 60
pages sampled (15 %)**, so roughly 19 of a 129-page benchmark. The earned-rotation rule leaves all
of them untouched, which is a miss whenever they are actually rotated.

Three of them were judged by a vision model and checked by eye where possible:

| page | vision model | checked | pipeline (earned rule) |
|---|---|---|---|
| `224351` (EPSON printer flyer) | `rotate_180` 0.97 | **confirmed**: the wordmark, every caption and every price reads inverted | leaves it — **miss** |
| `224512` (Conair flyer) | `rotate_cw_90` 0.96 | **confirmed** by eye, top on the left | leaves it — **miss** |
| `224602` (newspaper page) | `upright` 0.96 | consistent with its readable headings | leaves it — correct |

So for this class the ranking is: **vision model ≫ earned OCR scores ≫ raw OCR vote**. Two of three
were pages the pipeline would ship rotated.

**The routing that follows:** decide orientation with OCR where the score is earned (85 % of pages,
free), and hand the rest to a vision model. One 4-way question per page is enough — no coordinates,
no point lists — which is the form of question the local models failed at earlier and the reason the
failure did not generalise: asking for *one of four labels* is a classification, and asking for
twelve coordinates is not.

### Which vision model is enough: local 4B is not

The same four-way question, asked of two models, checked against the pages:

| page | truth (by eye) | local Qwen3.5-4B | a DeepSeek-class subagent |
|---|---|---|---|
| `224351` EPSON flyer | 180 | **C (180)** ✓ | **rotate_180** ✓ |
| `224512` Conair flyer | 90 clockwise | not asked | **rotate_cw_90** ✓ |
| `224505` Black+Decker flyer | 90 clockwise (BLACK+DECKER reads bottom-to-top on the left) | **D (90 counter-clockwise)** ✗ | **rotate_cw_90** ✓ |
| `224602` newspaper page | upright | **A (upright)** ✓ | **upright** ✓ |

Local 4B: **2 of 3**. Subagent: **3 of 3**. The local model's failure is the interesting one — not a
refusal, a confident wrong direction — so for this routing the model has to be the stronger one,
and that is ~40 s per page rather than a local call.

Recorded because the cost depends on it: 15 % of pages × 5,737 is ~860 calls. At 40 s that is ~10 h
of model time; at a local 4B's ~8 s it would be ~2 h, but 2 of 3 is not good enough to ship.

### The vision fallback is wired, and not default

`--orient-url` asks a four-way question about pages where no rotation was earned and there is no
text grid. Two things were measured while wiring it:

* **accuracy**: the local Qwen3.5-4B answered 2 of 3 (one confident wrong direction), a
  DeepSeek-class model 3 of 3. So the flag is useful only with a strong endpoint.
* **concurrency**: in a batch at `--threads 2` two of seven pages came back `error`; the same page
  run alone is `ok`. The local server has one slot and the requests queue past their timeout. Run
  this path with `--workers 1` or against an endpoint that can take concurrent calls.

Off by default (`orient_url=""`), because a 2-of-3 decider with confident wrong directions is worse
than the current behaviour of leaving the page alone: leaving it alone is right whenever the page is
already upright, and wrong in a way that is visible rather than destructive.

### Some "cut" pages were cut by the camera, not by the crop

`IMG_20260506_225042` is one of the two worst remaining cut pages. Its **source** photograph shows
the newspaper spread on a wooden table with **the page running off the top edge of the frame** and
other newspapers stacked to the left — the picture itself does not contain the whole sheet, and the
pipeline cannot recover what was never photographed.

That matters for the metric: `cut_lines` counts any text line touching the frame, and cannot tell
"the crop cut it" from "the camera cut it". So **37 is an upper bound on pages we damaged**, not a
census of our errors. Separating the two requires looking at each source frame, which is why the
number should be quoted with that caveat rather than as a defect count.

The same page is also upside down in the source — which is the case the routed vision model exists
for, and the case the OCR vote gets wrong.

**But not all of them are inherited.** `IMG_20260506_225300`, another 6-line cut page, has its
source spread **completely inside the frame** with margins on every side — just photographed 90
degrees round. Everything cut on that page is ours.

So of the three worst checked: one is the camera's (`225042`), one is ours (`225300`), and one
(`225734`) is a page the pipeline left rotated, so its "cuts" are a rotation symptom. The count of
37 is a mixture of three different defects wearing one number, and separating them needs exactly
what was done here, page by page.
