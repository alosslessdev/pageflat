# References

Everything pageflat borrows from, builds on, or evaluated and left out, with
links and licences.

> **Licensing note.** pageflat itself is MIT, but the vendored DocTr weights and
> code are licensed for **non-commercial use only**. Read [NOTICE](../NOTICE)
> before shipping anything that includes `pageflat/vendor/doctr` or the
> `weights/*.pth` files. Third-party components below keep their own licences.

## Document dewarping and rectification

**DocTr** — <https://github.com/fh2019ustc/DocTr> — U2NETP page segmentation, GeoTr backward-flow rectification and IllTr illumination correction; vendored under `pageflat/vendor/doctr` and used for the default `--dewarp doctr` path. Paper: <https://arxiv.org/abs/2110.12942> (ACM MM 2021). — Non-commercial, share-alike (custom licence).

**DocTr++** — <https://github.com/fh2019ustc/DocTr-Plus> — Follow-up to DocTr for unrestricted document rectification (TMM 2023, <https://arxiv.org/abs/2304.08796>); the reference for the flow model and its training data. — Non-commercial, share-alike (same licence as DocTr).

**UVDoc** — <https://github.com/tanguymagne/UVDoc> — Neural 3D grid unwarping (SIGGRAPH Asia 2023, <https://arxiv.org/abs/2302.02887>); used through `py-reform` as the `--dewarp uvdoc` backend. — MIT.

**BookNet** — <https://github.com/Liu-Shaokai/BookNet> — Cross-page attention rectification for book spreads (arXiv 2601.21938); the reference for two-page handling, not bundled. — No licence file (research code).

**Dewarping by displacement flow estimation** — <https://arxiv.org/abs/2104.06815> — Fully convolutional network that predicts a displacement flow field; the basis of pageflat's flow-based warp and its tiled `grid_sample` application. — Paper only, no code released.

**DocUNet** — <https://openaccess.thecvf.com/content_cvpr_2018/html/Ma_DocUNet_Document_Image_CVPR_2018_paper.html> — CVPR 2018 stacked U-Net dewarping and the benchmark dataset; the dataset page (<https://www3.cs.stonybrook.edu/~cvl/docunet.html>) is where the evaluation images come from. — Dataset code MIT; images under Google Street View terms.

**page_dewarp** — <https://github.com/mzucker/page_dewarp> — Matt Zucker's cylindrical text-line dewarping with a bundled C++ implementation; the reference for the line-based refinement stage (`pageflat/refine.py`). — MIT.

**Shibayama et al.** — <https://doi.org/10.1007/978-3-642-37447-0_27> — "Reconstruction of 3D Surface and Restoration of Flat Document Image from Monocular Image Sequence", ACCV 2012 (LNCS 7727, 2013); the multi-frame surface-reconstruction view behind the spine/curvature reasoning. — Springer, paywalled.

## Page detection and segmentation

**U2-Net / U2NETP** — <https://github.com/xuebinqin/U-2-Net> — Salient-object detector whose small U2NETP variant produces DocTr's page-probability mask. — Apache-2.0.

**Segment Anything (SAM)** — <https://github.com/facebookresearch/segment-anything> — Promptable segmentation (arXiv 2304.02643); `--sam` points it at the page centre so overlapping sheets are split along the topmost sheet's boundary. — Apache-2.0.

**MobileSAM** — <https://github.com/ChaoningZhang/MobileSAM> — Tiny-ViT SAM that gives the same masks for a tenth of the VRAM; the default SAM weights (`weights/mobile_sam.pt`). — Apache-2.0.

## OCR and orientation

**Tesseract** — <https://github.com/tesseract-ocr/tesseract> — OCR engine: OSD rotation detection (0/90/180/270) and word confidence used to pick the better of two dewarp outputs. — Apache-2.0.

**Leptonica** — <http://www.leptonica.org/> — The image-processing library Tesseract is built on and the source of its page-layout primitives. — BSD-2-Clause.

**docTR (Mindee)** — <https://github.com/mindee/doctr> — Mindee's OCR library; its MobileNetV3 page-orientation classifier runs on GPU in about 10 ms as the second orientation check. — Apache-2.0.

**python-doctr** — <https://pypi.org/project/python-doctr/> — The PyPI wheel that ships Mindee's pretrained orientation model. — Apache-2.0.

**EasyOCR** — <https://github.com/JaidedAI/EasyOCR> — Deep-learning OCR toolkit evaluated as a Tesseract replacement; not adopted. — Apache-2.0.

**PaddleOCR** — <https://github.com/PaddlePaddle/PaddleOCR> — PaddlePaddle OCR stack evaluated as an alternative; not adopted. — Apache-2.0.

## Models and serving

**Cloudflare Clef** — <https://huggingface.co/cloudflare/clef-flash> — Multimodal decision model that scores a scan as page/not-page; served by `llama-server --mmproj` in `scripts/classify_pages.py` and `scripts/clef_cleanup.py`. — Apache-2.0.

**llama.cpp** — <https://github.com/ggml-org/llama.cpp> — `llama-server` hosts both Clef and the optional grouping model. — MIT.

**LFM2.5** — <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct> — Small instruct model suggested for adjudicating ambiguous document boundaries in `group_documents.py`. — LFM Open License v1.0.

**Qwen3** — <https://github.com/QwenLM/Qwen3> — Qwen3-4B is the other suggested boundary adjudicator. — Apache-2.0.

## GPU image processing (evaluated, not adopted)

**NVIDIA DALI** — <https://github.com/NVIDIA/DALI> — GPU decode/augment pipeline evaluated for the JPEG-heavy ingest stage; not adopted. — Apache-2.0.

**cuCIM** — <https://github.com/rapidsai/cucim> — RAPIDS GPU image I/O and processing library evaluated for decode/resize; not adopted. — Apache-2.0.

**kornia** — <https://github.com/kornia/kornia> — Differentiable computer-vision operators evaluated for GPU warping; not adopted. — Apache-2.0.

**torchvision nvJPEG decoding** — <https://github.com/pytorch/vision/blob/main/torchvision/io/image.py> — GPU JPEG decoder (`torchvision.io.decode_jpeg`) evaluated against the CPU scaled-decode path; not adopted. — BSD-3-Clause.

## Software

**PyTorch** — <https://pytorch.org/> — Model inference and the CUDA kernels behind the dewarp, illumination and area-resize stages. — BSD-3-Clause.

**OpenCV** — <https://github.com/opencv/opencv> — JPEG decoding, warping, colour conversion and contour geometry. — Apache-2.0.

**NumPy** — <https://numpy.org/> — Array plumbing between OpenCV, PyTorch and the geometry code. — BSD-3-Clause.

**py-reform** — <https://pypi.org/project/py-reform/> — Python wrapper for UVDoc, including its weights. — MIT.

**reportlab** — <https://www.reportlab.com/> — Writes the searchable PDFs: one page image plus an invisible text layer. — BSD-3-Clause.
