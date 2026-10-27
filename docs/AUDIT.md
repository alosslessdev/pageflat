# Audit notes

An independent pass over 47 scans from the third batch found the following
defects. Each entry lists the fix that went into the pipeline.

| finding | cause | fix |
|---|---|---|
| 16/47 single pages cut into `_p1`/`_p2` | `find_spine_x` accepted any dip in the column-darkness profile inside the central band | the profile peak must now be a *shadow band* (width ≥ 0.4% of the page, ≥ 10 levels darker than its neighbourhood), and every cut must survive the "paper-white next to the gutter" test |
| the one genuine spread in the sample stayed whole, and a three-page spread produced a single scan | gutters were searched only in 42–58% of the width, and splitting ran *after* illumination flattening had normalised away the gutter shadow | gutters are searched over 15–85%, several are allowed (a photo of three pages yields three scans), and splitting now happens before illumination |
| 10/47 scans not upright (2 upside down, 8 sideways) | Tesseract OSD was trusted even at low confidence, and returned nothing for handwriting | OSD is used only when its orientation confidence ≥ 2.0; otherwise the four candidate rotations are scored by Tesseract word confidence (with the installed language pack — this machine has `spa`, not `eng`) |
| 16/47 crops land inside the page, clipping words | SAM was allowed to pick any confident mask, including a text block inside the sheet | SAM candidates must cover at least half of the DocTr page mask |
| 3 non-page photos accepted (blue gradient, striped fabric, night scene) | the content gate ran after illumination, which neutralises exactly the colour and texture evidence that distinguishes a page | surface checks (paper colour fraction, Laplacian texture) now run on the un-flattened dewarped page; content checks (ink, rules) run after flattening |
| blank card white-balanced to solid cyan | same cause | as above, plus the paper-fraction floor |
| `straighten()` left curved handwriting uncorrected (up to 64 px) | the refinement existed but was not part of the batch | available via `scripts/refine_scan.py`; it only fires when the measured baseline wobble exceeds 1% of the page height |

Known limitations that were *not* fixed:

* A smooth, neutral surface with no text (a blue gradient backdrop, a striped
  sheet) still passes the surface gate — there is no cue left once the image is
  flat. Two such images out of 652 remain in the output.
* A page whose top edge is physically covered by another sheet is cropped to the
  visible part; the hidden strip is not reconstructed.
* Residual curvature below 1% of page height is deliberately left alone: on
  real scans measured baseline wobble is already ~1.5 px, and warping further
  only adds interpolation noise.
