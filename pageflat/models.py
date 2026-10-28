"""Model wrappers: DocTr (U2NETP page segmentation + GeoTr flow + IllTr) and UVDoc."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "weights"


def _strip_prefix(state_dict, prefixes=("module.", "model.")):
    out = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def _load_state(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(_strip_prefix(sd), strict=True)
    return model


def _to_tensor(img_rgb01: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(img_rgb01.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


class DocTrPipeline:
    """DocTr geometric unwarping + page segmentation, loaded once."""

    def __init__(self, device: str = "cuda", weights_dir: Path | str = WEIGHTS_DIR):
        from .vendor.doctr.seg import U2NETP  # noqa: E402
        from .vendor.doctr.GeoTr import GeoTr  # noqa: E402
        from .vendor.doctr.IllTr import IllTr  # noqa: E402

        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        weights_dir = Path(weights_dir)
        self.seg = _load_state(U2NETP(3, 1), weights_dir / "seg.pth").to(self.device).eval()
        self.geotr = _load_state(GeoTr(num_attn_layers=6), weights_dir / "geotr.pth").to(self.device).eval()
        self._illtr = None
        self._illtr_cls = IllTr
        self.weights_dir = weights_dir

    @property
    def illtr(self):
        """Illumination transformer, loaded only when requested (saves VRAM)."""
        if self._illtr is None:
            self._illtr = _load_state(self._illtr_cls(), self.weights_dir / "illtr.pth").to(self.device).eval()
        return self._illtr

    # ---------------------------------------------------------------- mask
    @torch.no_grad()
    def page_mask(self, img_bgr: np.ndarray, size: int = 512) -> np.ndarray:
        """Page probability mask (float32, 0..1) at the input image resolution."""
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
        x = _to_tensor(small[:, :, ::-1].astype(np.float32) / 255.0, self.device)
        msk = self.seg(x)[0]
        m = msk[0, 0].detach().cpu().numpy()
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---------------------------------------------------------------- flow
    @torch.no_grad()
    def geo_flow(self, img_bgr: np.ndarray) -> np.ndarray:
        """DocTr backward flow (h, w, 2) in normalized grid_sample coordinates."""
        h, w = img_bgr.shape[:2]
        small = cv2.resize(img_bgr, (288, 288), interpolation=cv2.INTER_AREA)
        x = _to_tensor(small[:, :, ::-1].astype(np.float32) / 255.0, self.device)
        msk = (self.seg(x)[0] > 0.5).float()
        bm = self.geotr(msk * x)[0].detach().cpu().numpy()
        bm = (2 * (bm / 286.8) - 1) * 0.99
        flow = np.stack(
            [
                cv2.resize(bm[0], (w, h), interpolation=cv2.INTER_LINEAR),
                cv2.resize(bm[1], (w, h), interpolation=cv2.INTER_LINEAR),
            ],
            axis=2,
        )
        return cv2.blur(flow, (3, 3))

    @torch.no_grad()
    def rectify(self, img_bgr: np.ndarray, band: int = 2048) -> np.ndarray:
        """DocTr geometric unwarping.

        One grid_sample over the whole page when it fits in VRAM (a 3508 px page
        needs ~300 MB), otherwise row bands. Everything stays on the GPU until a
        single uint8 copy back.
        """
        flow = self.geo_flow(img_bgr)
        h, w = img_bgr.shape[:2]
        src = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
        src = src.float().div(255.0).unsqueeze(0).to(self.device)
        out = np.empty((h, w, 3), np.uint8)
        for y0 in range(0, h, band):
            y1 = min(h, y0 + band)
            grid = torch.from_numpy(flow[y0:y1]).unsqueeze(0).to(self.device)
            o = F.grid_sample(src, grid, align_corners=True, mode="bilinear")
            rgb = o[0].permute(1, 2, 0).mul_(255.0).clamp_(0, 255).to(torch.uint8).cpu().numpy()
            out[y0:y1] = rgb[:, :, ::-1]
            del grid, o
        del src
        return out

    @torch.no_grad()
    def illuminate_gainmap(self, img_bgr: np.ndarray, work_side: int = 512) -> np.ndarray:
        """DocTr IllTr on a small copy, applied back as a gain map at full resolution.

        Running the transformer over 128x128 patches of a 3508 px page means a
        thousand forward passes; on a 512 px copy it is ~16, and the correction
        is a smooth lighting field, so upsampling the gain is faithful.
        """
        h, w = img_bgr.shape[:2]
        scale = min(1.0, work_side / float(max(h, w)))
        small = cv2.resize(img_bgr, (max(64, int(w * scale)), max(64, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        corrected = self.illuminate(small) if False else self._illtr_pass(small)
        gain = (corrected.astype(np.float32) + 1.0) / (small.astype(np.float32) + 1.0)
        gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_CUBIC)
        out = img_bgr.astype(np.float32) * gain
        return np.clip(out, 0, 255).astype(np.uint8)

    @torch.no_grad()
    def _illtr_pass(self, img_bgr: np.ndarray, patch: int = 128) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        ph, pw = min(patch, h), min(patch, w)
        out = img_bgr.copy()
        for y in range(0, h - ph + 1, ph):
            for x in range(0, w - pw + 1, pw):
                tile = img_bgr[y:y + ph, x:x + pw]
                t = _to_tensor(tile[:, :, ::-1].astype(np.float32) / 255.0, self.device)
                o = self.illtr(t)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255.0
                out[y:y + ph, x:x + pw] = np.clip(o, 0, 255).astype(np.uint8)
        return out

    @torch.no_grad()
    def illuminate(self, img_bgr: np.ndarray, patch: int = 128) -> np.ndarray:
        """DocTr illumination correction, patch-wise on 128x128 tiles."""
        h, w = img_bgr.shape[:2]
        step = patch - patch // 8
        ys = list(range(0, max(1, h - patch + 1), step)) or [0]
        xs = list(range(0, max(1, w - patch + 1), step)) or [0]
        out = img_bgr.copy().astype(np.float32)
        for y in ys:
            for x in xs:
                y2 = min(y + patch, h)
                x2 = min(x + patch, w)
                tile = img_bgr[y2 - patch:y2, x2 - patch:x2]
                t = _to_tensor(tile[:, :, ::-1].astype(np.float32) / 255.0, self.device)
                o = self.illtr(t)[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255.0
                if y2 == y + patch and x2 == x + patch:
                    out[y:y2, x:x2] = o
                else:
                    out[y:y2, x:x2] = o[patch - (y2 - y):, patch - (x2 - x):]
        return out.clip(0, 255).astype(np.uint8)


class UVDocDewarper:
    """UVDoc (grid-based, SIGGRAPH Asia 2023) via py-reform's reference implementation."""

    def __init__(self, device: str = "cuda"):
        from py_reform.models.uvdoc_model import UVDocModel

        self.model = UVDocModel(device=device)

    def rectify(self, img_bgr: np.ndarray) -> np.ndarray:
        from PIL import Image

        pil = Image.fromarray(img_bgr[:, :, ::-1])
        out = self.model.process(pil)
        return np.array(out)[:, :, ::-1].copy()


class DewarpNetDewarper:
    """DewarpNet (ICCV 2019): predict the sheet's 3D surface, then flatten it.

    The shape network regresses a 3D coordinate map of the page; the texture
    mapping network turns that map into the sampling grid that undoes the
    deformation. This is the 3D-geometry answer to paper curl, as opposed to
    DocTr's learned flow, which bends pages that were already straight.
    """

    def __init__(self, weights_dir: Path | str = WEIGHTS_DIR, device: str = "cuda"):
        from .vendor.dewarpnet import UnetGenerator, dnetccnl

        self.device = torch.device(device)
        self.wc = UnetGenerator(input_nc=3, output_nc=3, num_downs=7)
        self.bm = dnetccnl(img_size=128, in_channels=3, out_channels=2, filters=32)
        for model, name in ((self.wc, "unetnc_doc3d.pkl"), (self.bm, "dnetccnl_doc3d.pkl")):
            state = torch.load(str(Path(weights_dir) / name), map_location="cpu",
                               weights_only=False)
            state = state.get("model_state", state) if isinstance(state, dict) else state
            state = _strip_prefix(state)
            # the released checkpoints were saved from the 2019 wrappers: the
            # unet keys sit one level shallower than today's module tree
            if all(k.startswith("model.") and not k.startswith("model.model.")
                   for k in state if k.startswith("model.")):
                state = {("model." + k) if k.startswith("model.") else k: v
                         for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            model.to(self.device).eval()

    @torch.no_grad()
    def rectify(self, img_bgr: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F

        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (256, 256))
        tensor = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        shape = torch.tanh(self.wc(tensor.to(self.device)))
        grid = self.bm(F.interpolate(shape, size=(128, 128), mode="nearest"))

        # the sampling grid is in [-1, 1]; the reference maps it to pixels
        bm = grid.permute(0, 2, 3, 1)[0].cpu().numpy()
        bm = cv2.blur(bm[:, :, 0], (3, 3)), cv2.blur(bm[:, :, 1], (3, 3))
        sample = np.stack([cv2.resize(b, (w, h)) for b in bm], axis=-1)[None]
        sample = torch.from_numpy(sample.astype(np.float32))
        src = torch.from_numpy((rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
        out = F.grid_sample(src, sample, align_corners=True)[0].permute(1, 2, 0).numpy()
        return cv2.cvtColor((out * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
