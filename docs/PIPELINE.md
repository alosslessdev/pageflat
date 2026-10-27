# Pipeline notes

Field notes from building this on a real 250 000-photo library.

## Discovery

Scanning the three ext4 volumes on the machine:

| stage | count |
|---|---|
| JPEG files traversed | 243 858 |
| ≥ 60 MP (9248×6944 = 64.2 MP phone captures) | 5 600 |
| flagged page-like by the flat-surface score | 652 |

The 5 600 large files are all from two phone backups, two sensor modes
(9248×6944 landscape, 6944×9248 portrait) plus a handful of panoramas. The 652
flagged files are dominated by photographed pages: ruled notebooks with
handwriting, printed forms, receipts, newspaper pages, book spreads. A small
minority are false positives — a card lying on a bed sheet, for example — which
the content gate is there to catch.

The page score is `fraction of pixels within ±48 of the dominant quantised
colour` on a 128×128 thumbnail. Paper pages score 0.75–0.99; ordinary scenes
score 0.1–0.6.

## Why this order of stages

* **Perspective before dewarping.** DocTr's flow model expects the page to fill
  the frame; feeding it a photo where the page occupies a third of the frame
  produces a flow that tries to pull the background in. Warping the quad to a
  rectangle first makes the learned model's job much smaller.
* **Dewarping before illumination.** Dividing out the lighting surface changes
  the appearance the flow model was trained on (synthetic Doc3D renders keep
  their shading), and flattening curvature afterwards would smear the corrected
  shading.
* **Deskew after dewarping.** The flow prediction is trained on the *content* of
  the page; a global rotation of a couple of degrees is outside its
  parameterisation, but easy to measure with Hough lines afterwards.
* **Content gate last.** It is the cheapest check and needs the final image.

## Segmentation

DocTr's U2NETP page model labels paper as a class, not instances. Two
consequences:

* components are scored by *mean page probability* rather than by area, so a
  small, confident sheet is not swallowed by a large, weakly-labelled
  background surface;
* anything that is paper — a pile, a second sheet, a newspaper behind the page —
  merges into one mask. SAM with a positive point at the quad centroid and the
  DocTr mask as a prior fixes most of this: the *smallest* confident mask that
  contains the point and is contained in the prior is the topmost sheet, which
  is what the photographer was aiming at.

  On a 6 GB laptop GPU shared with another process, SAM ViT-B (1024x1024
  encoder) does not fit next to DocTr — its encoder allocates ~1.5 GB regardless
  of the input size passed in. MobileSAM (`vit_t`) produces equivalent page
  masks here in 0.3 s and fits comfortably, on GPU or CPU. That is the default
  for batch work; ViT-B stays available for difficult frames.

## Dewarping quality

Observed on this library:

* flat pages, forms, receipts — DocTr removes the residual perspective; output
  is indistinguishable from a flatbed scan apart from paper texture;
* pages curved near a book spine — DocTr and UVDoc both flatten the global
  curvature; a residual "wave" survives on strongly curled pages;
* handwriting on ruled paper — the learned flow has no text-line cue and the
  ruled lines keep a slight bend; a text-line-based refinement (mzucker's
  approach, or a residual spline fitted to detected rules) is the obvious next
  step;
* two-page spreads — split at the gutter unless ink runs into the gutter on both
  sides (checked by comparing the columns immediately left and right of the
  gutter, not the gutter itself, whose shadow otherwise looks like content).

## Throughput

On this machine (RTX 3060 Laptop, 6 GB, partially occupied by another process,
12 cores):

| configuration | time per image |
|---|---|
| DocTr + MobileSAM, GPU, 3508 px output | ~8 s |
| DocTr + MobileSAM, CPU (3 threads) | ~20 s |
| DocTr without SAM, GPU | ~5 s |
| illumination before the quadratic-surface rewrite | +38 s |

Two GPU workers do not fit in 6 GB together with the NVidia context; one GPU
worker plus one CPU worker is the practical parallel setup. `grid_sample` is
tiled in 512-row bands so the peak allocation is bounded by the source image
rather than by the output.
