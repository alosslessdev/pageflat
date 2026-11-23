# Running the external rectifiers

This pipeline does its own geometry. The notes here record how the two external rectifiers were
actually run during development, because in both cases the published code does not run as shipped.

## DvD — runs locally

`hanquansanren/DvD` (SIGGRAPH Asia 2025), 151 M parameters, coordinate-level diffusion. Weights are a
public Google Drive folder linked from the repository README.

Measured on the development machine (RTX 3060 Laptop, 6 GB): **4.4 s/page**, output at the source
resolution (9248×6944 for a phone photograph).

### It does not run as shipped — four defects

1. **Loader discards the dataset root.** `datasets/doc_dataset/doc_benchmark.py` builds paths with
   `pjoin(dir, "/" + warp_type, name)`; because the second component is absolute, `os.path.join`
   returns `/curved/name` and every lookup fails. Join the subdirectory without the leading slash.
2. **`data_root` must contain images, not folders.** `init_img_parms` lists `data_root` directly, so a
   `dataset/warp_type/` layout yields directories where images are expected.
3. **`cross_model.py` calls an unbound loop variable.** The line `# for block in self.blocks:` is
   commented out while the next line uses `block`, so the first sampling step raises
   `UnboundLocalError`. Uncomment the loop.
4. **Weights land on a full filesystem.** `torch.hub` downloads VGG16 to `TORCH_HOME`; point it
   somewhere with space, and the same for `TMPDIR` — OpenMPI needs scratch space at init and aborts
   with a confusing `MPI_Init_thread` error when the temp filesystem is full.

### Two shims

`mmcv.cnn.ConvModule` and `mmengine.model.BaseModule` come from compiled OpenMMLab packages pinned to
older torch builds; only a conv-with-norm-and-activation and an empty base class are used, so provide
those. `mpi4py` is used only to learn rank and world size before
`torch.distributed.init_process_group`, so a single-process module reporting rank 0 of 1 avoids
starting OpenMPI at all.

```bash
TORCH_HOME=/path/with/space \
TMPDIR=/path/with/space \
PYTHONPATH=/path/to/shims:/path/to/DvD:. \
python run_sampling.py --train_module dvd --train_name val_TDiff --name run1
```

Set `EnvironmentSettings.eval_dataset` in `admin/local.py` to a directory of images and
`eval_dataset_name` to `docunet`. Rectified pages appear under
`vis_hp/<name>/<run name>/dewarped_pred/`.

**What it does not do:** it returns the whole rectified frame and never crops, and it reconstructs
geometry rather than ink — a page the camera cut comes back rectangular with the missing words still
missing.

## DocTr++ — cannot be run

`fh2019ustc/DocTr-Plus` is the method for documents with partially missing borders, which is exactly
the camera-cut case. Its code is public but **the pretrained weights are not**: the repository's
`model_pretrained/` contains only `readme.txt`, and issue #22 requesting the weights has no
substantive reply. `inference.py` expects `./model_pretrained/DocTrP.pth`.

The same applies to DocReal (`irisXcoding/DocReal`), whose README states the source cannot be released
for commercial reasons and publishes benchmark data only.

## DocTr — this is what the pipeline uses

`--dewarp doctr` uses the vendored `pageflat/vendor/doctr` with `weights/geotr.pth` and
`weights/illtr.pth`. It is the only rectifier integrated by default, and it is trained on fully-framed
pages, so it is also the reason the control-point and quad paths remain the honest default here.
