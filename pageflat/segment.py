"""Notes on bounding a page mask at the sheet's edge.

Not implemented, and here is why, so it is not attempted blind again.

The learned page model calls anything paper-coloured "page" and a pale table is
paper-coloured, so on `IMG_20251229_122443` the mask reaches y=893 of a 900-px
frame while the sheet's edge is near 800 and SAM's mask stops at 780. The print
looks like a witness but is not: `printed_mask` covers the whole frame here, so it
bounds nothing.

Two candidate rules were measured and both fail:

* **Brightness step.** The row means from y=760 to y=899 sit between 140 and 153 —
  the page's white margin and the pale table have the same mean, so the
  paper-to-table step is invisible in the profile (max |grad| 0.67 over that band).
* **Per-side minimum of the two masks.** SAM is tighter on the side that matters
  (bottom 780 vs 893) but *tighter than the page* elsewhere: on the same frame its
  left boundary is x=127 against docTR's x=12, which would cut 115 px off the sheet.

What the numbers do support: docTR over-reaches, SAM's boundary is closer on the
over-reaching sides, and the two disagree by more than a margin. A safe rule needs to
know which mask is right per side, and that is a segmentation problem rather than a
threshold — a better sheet model, or SAM prompted per side with a check against the
print.
"""