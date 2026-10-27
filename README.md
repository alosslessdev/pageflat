# pageflat

Turn photographs of pages — books, notebooks, loose sheets, forms, receipts,
newspaper — into flat, scanner-like images.

```
photo of a page  ──▶  page detection  ──▶  perspective rectification
                 ──▶  dewarping (DocTr / UVDoc)  ──▶  illumination flattening
                 ──▶  spine split  ──▶  flat scan (PNG, 300 dpi)
```

The pipeline is designed for the messy end of camera-based scanning: pages lying
on patterned bed sheets, crumpled pages, piles with several pages in frame, books
whose spine is visible, and pages photographed upside down.

## What it does

1. **Page discovery** (`scripts/find_pages.py`) — triage a photo library by
   resolution and by how much of the frame is one flat, low-saturation surface.
   A 60 MP camera roll takes ~30 s for 250 000 files.
2. **Page segmentation** — DocTr's U2NETP page model produces a page probability
   mask; the largest, most page-like connected component is kept, so scattered
   pages and patterned backgrounds do not leak into the crop.
3. **Optional SAM refinement** (`--sam`) — DocTr's mask treats every sheet as the
   same class, so piles and overlapping sheets merge. Pointed at the page of
   interest, SAM follows the boundary of the *topmost* object; the smallest
   confident mask containing the point wins, which resolves "card on a bed
   sheet" cases that the page model alone cannot.
4. **Boundary estimation and perspective correction** — the mask is reduced to a
   quadrilateral (convex hull → `approxPolyDP`, minimum-area rectangle fallback)
   and then *edge-snapped*: every side is re-fitted to the strongest nearby
   intensity edge, and corners come from line intersections. The quad is warped
   to a rectangle with `cv2.getPerspectiveTransform` + `warpPerspective`.
5. **Orientation** — Tesseract OSD decides the 0/90/180/270 rotation of the
   rectified page, so upside-down captures come out upright.
6. **Dewarping** — Two interchangeable learned rectifiers:
   * [DocTr](https://github.com/fh2019ustc/DocTr) `GeoTr` — a transformer that
     predicts a dense backward flow at 288×288, upsampled and applied with a
     tiled `grid_sample`;
   * [UVDoc](https://github.com/tanguymagne/UVDoc) — grid-based unwarping via
     `py-reform`, which predicts a 45×31 3D grid and resamples through it.
   `--dewarp best` runs both and keeps the output with the better mean Tesseract
   word confidence.
7. **Residual deskew** — Hough-based median line angle, applied when it exceeds
   0.25°, removes the small rotations that survive the dewarp.
8. **Content gate** — a finished scan is rejected when it has neither ink-like
   strokes nor long rules, which catches "page-shaped" surfaces (bed sheets,
   tables) that the segmentation stage mistook for paper.
9. **Illumination flattening** — the paper-lighting surface is modelled as a
   quadratic fitted through per-tile 90th-percentile brightness, then divided out
   and white-balanced, which removes vignetting and page curl shadows without
   destroying the paper texture. DocTr's `IllTr` is available as an alternative.
10. **Spine handling** — for spreads, a gutter is located from the column-wise
   darkness profile; the spread is split unless ink crosses the gutter (i.e. the
   artwork runs across the spine).
11. **Output** — PNG at a scanner-like size (default long side 3508 px ≈ A4 at
   300 dpi), with a JSON report of what happened to every input.

## Install

```bash
git clone https://github.com/alosslessdev/pageflat && cd pageflat
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
bash scripts/download_weights.sh          # DocTr seg/GeoTr/IllTr weights
```

`tesseract` must be on `PATH` (orientation, quality selection). A CUDA GPU is
used when available; inference is tiled so a 6 GB card is enough.

## Usage

```bash
# find candidate page photos in a phone backup
python scripts/find_pages.py /run/media/phone --min-mp 60 --out pages.jsonl

# flatten them
python -m pageflat.cli --list pages.jsonl -o scans --dewarp doctr --sam --workers 1

# single file, with the boundary overlay for debugging
python -m pageflat.cli photo.jpg -o scans --debug
```

Options: `--dewarp {doctr,uvdoc,best,none}`, `--illum {classical,doctr,none}`,
`--sam`, `--long-side`, `--no-split`, `--debug`, `--json`, `--workers`.

After a batch, `scripts/finalize_scans.py` re-checks finished scans, splits any
spreads that were left whole, and writes a manifest.

## Output contract

* one PNG per detected page (`<name>.png`, `<name>_p1.png`, `<name>_p2.png` for splits)
* upright, perspective-corrected, dewarped, evenly lit
* no border beyond the page
* JSON report: `ok` / `rejected` (with reason) / `error`, quad coordinates,
  area fraction, rotation applied, content metrics, pages produced, timing

## Limitations

* Occluded pages (another sheet lying on top of the page) are cropped to the
  visible page area; the hidden strip is not reconstructed.
* Heavy 3D curl beyond what DocTr/UVDoc were trained on keeps residual curvature;
  text-line-based refinement is not yet part of the pipeline.
* Handwriting on blank paper carries no geometric cue, so dewarping relies
  entirely on the learned flow model.

## References

* Feng et al., *DocTr: Document Image Transformer for Geometric Unwarping and
  Illumination Correction*, ACM MM 2021.
* Feng et al., *Deep Unrestricted Document Image Rectification*, TMM 2023 (DocTr++).
* Verhoeven et al., *UVDoc: Neural Grid-based Document Unwarping*, SIGGRAPH Asia 2023.
* Xie et al., *Dewarping Document Image By Displacement Flow Estimation with FCN*, 2021.
* Shibayama et al., *Reconstruction of 3D Surface and Restoration of Flat Document
  Image from Monocular Image Sequence*, ACCV 2013.
* Zucker, *Page dewarping* (mzucker/page_dewarp).
* Liu et al., *BookNet: Book Image Rectification via Cross-Page Attention Network*, 2026.

Licences of the borrowed models/code remain with their authors (MIT for UVDoc and
page_dewarp, research licences for DocTr/BookNet).
