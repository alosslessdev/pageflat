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

Numbers are reported in `docs/PIPELINE.md` for throughput; acceptance and split
rates are recorded per run in the JSON reports next to the scans.

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
