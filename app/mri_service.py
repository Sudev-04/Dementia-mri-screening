"""MRI inference for the clinic page. ONNX Runtime only, no PyTorch needed."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from common.preprocess import PREPROCESS_VERSION, PreprocessConfig, check_input, load_volume, preprocess


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


class MriModel:
    """A cross-validation ensemble exported by ``ml.export``."""

    def __init__(self, model_dir: str | Path):
        import onnxruntime as ort

        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "model_meta.json").read_text())
        self.cfg = PreprocessConfig.from_dict(self.meta["preprocess"])
        if self.cfg.version != PREPROCESS_VERSION:
            raise RuntimeError(
                f"Model was trained with preprocessing '{self.cfg.version}' but this server runs "
                f"'{PREPROCESS_VERSION}'. Re-export the model or check out the matching code.")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.sessions = [ort.InferenceSession(str(self.dir / f["file"]), opts, providers=["CPUExecutionProvider"])
                         for f in self.meta["folds"]]
        self.temperatures = [float(f["temperature"]) for f in self.meta["folds"]]
        self.threshold = float(self.meta["threshold"])

    # -- inference -----------------------------------------------------------
    def _probs(self, batch: np.ndarray) -> np.ndarray:
        """Calibrated probabilities, shape (n_models, batch)."""
        out = [_sigmoid(s.run(None, {"volume": batch})[0].reshape(-1) / t)
               for s, t in zip(self.sessions, self.temperatures)]
        return np.stack(out)

    def prepare(self, path: str | Path) -> tuple[np.ndarray, dict]:
        vol, zooms = load_volume(path)
        info = check_input(vol, zooms, self.cfg)   # raises InputRejected
        return preprocess(vol, self.cfg), info

    def predict(self, x: np.ndarray) -> dict:
        p = self._probs(x[None, None].astype(np.float32))[:, 0]
        mean = float(p.mean())
        above = p >= self.threshold
        return {
            "probability": round(mean, 4),
            "member_probabilities": [round(float(v), 4) for v in p],
            "spread": round(float(p.max() - p.min()), 4),
            "threshold": round(self.threshold, 4),
            "above_threshold": bool(mean >= self.threshold),
            "members_agree": bool(above.all() or (~above).all()),
            "model": self.meta["name"],
            "synthetic": bool(self.meta.get("synthetic")),
        }

    def occlusion_map(self, x: np.ndarray, patch: int = 20, batch_size: int = 8) -> np.ndarray:
        """Coarse occlusion sensitivity: how much the ensemble probability falls
        when each block of the brain is blanked. Positive = the block pushed the
        score up. Resolution is the patch size, so this localises to a region,
        not a structure."""
        base = float(self._probs(x[None, None])[:, 0].mean())
        heat = np.zeros_like(x, dtype=np.float32)
        boxes = [(i, j, k) for i in range(0, x.shape[0], patch) for j in range(0, x.shape[1], patch)
                 for k in range(0, x.shape[2], patch)
                 if x[i:i + patch, j:j + patch, k:k + patch].mean() > 0.02]
        for start in range(0, len(boxes), batch_size):
            chunk = boxes[start:start + batch_size]
            batch = np.repeat(x[None, None], len(chunk), axis=0).astype(np.float32)
            for n, (i, j, k) in enumerate(chunk):
                batch[n, 0, i:i + patch, j:j + patch, k:k + patch] = 0.0
            drop = base - self._probs(batch).mean(axis=0)
            for n, (i, j, k) in enumerate(chunk):
                heat[i:i + patch, j:j + patch, k:k + patch] = drop[n]
        return heat

    def status(self) -> dict:
        m = self.meta
        return {"loaded": True, "name": m["name"], "created": m["created"], "synthetic": bool(m.get("synthetic")),
                "task": m["task"], "members": len(self.sessions), "threshold": self.threshold,
                "training": m.get("training"), "evaluation": m.get("evaluation"), "limits": m.get("limits", [])}


def load_model(app_config) -> tuple[MriModel | None, str]:
    """Return (model, message). The demo model is only used when explicitly allowed."""
    current, demo = Path(app_config["MODEL_DIR"]), Path(app_config["DEMO_MODEL_DIR"])
    if (current / "model_meta.json").exists():
        return MriModel(current), "trained model loaded"
    if app_config.get("ALLOW_DEMO_MODEL") and (demo / "model_meta.json").exists():
        return MriModel(demo), "synthetic demo model loaded"
    return None, ("No trained model is installed. Train on OASIS-1 (notebooks/train_oasis_colab.ipynb), "
                  "then copy the exported files into models/current.")


def render_slices(x: np.ndarray, heat: np.ndarray | None = None, scale: int = 3) -> bytes:
    """Three orthogonal mid-slices side by side as a PNG, with an optional heat overlay."""
    from PIL import Image

    def panel(arr2d, heat2d):
        g = (np.clip(np.rot90(arr2d), 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], axis=-1).astype(np.float32)
        if heat2d is not None:
            h = np.rot90(heat2d)
            peak = float(np.abs(heat).max()) or 1.0
            a = np.clip(np.abs(h) / peak, 0, 1)[..., None] * 0.65
            # orange where the region raised the score, blue where it lowered it
            colour = np.where(h[..., None] >= 0, np.array([235, 120, 20.0]), np.array([40, 110, 220.0]))
            rgb = rgb * (1 - a) + colour * a
        return rgb.astype(np.uint8)

    c = [s // 2 for s in x.shape]
    views = [(x[c[0], :, :], None if heat is None else heat[c[0], :, :]),
             (x[:, c[1], :], None if heat is None else heat[:, c[1], :]),
             (x[:, :, c[2]], None if heat is None else heat[:, :, c[2]])]
    panels = [panel(a, h) for a, h in views]
    height = max(p.shape[0] for p in panels)
    canvas = np.zeros((height, sum(p.shape[1] for p in panels) + 8 * (len(panels) - 1), 3), dtype=np.uint8)
    xoff = 0
    for p in panels:
        yoff = (height - p.shape[0]) // 2
        canvas[yoff:yoff + p.shape[0], xoff:xoff + p.shape[1]] = p
        xoff += p.shape[1] + 8
    img = Image.fromarray(canvas)
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST if heat is None else Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
