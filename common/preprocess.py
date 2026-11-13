"""MRI loading and preprocessing shared by training and serving.

Training and the app server both import this one module, so the tensor a model
sees at inference is produced by exactly the code that produced its training
data. Only numpy and nibabel are needed (no PyTorch), which keeps the app light.

Expected input: an OASIS-1 style volume, i.e. T1-weighted, bias-corrected,
skull-stripped and registered to the Talairach (T88) atlas at 1 mm, shape
176 x 208 x 176. A raw scan straight off a hospital scanner is NOT in this
space and is rejected by ``check_input`` rather than silently mis-scored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

PREPROCESS_VERSION = "t88-crop-pool-v1"


@dataclass(frozen=True)
class PreprocessConfig:
    expected_shape: tuple[int, int, int] = (176, 208, 176)
    expected_voxel_mm: float = 1.0
    crop_shape: tuple[int, int, int] = (160, 192, 160)
    pool: int = 2                    # 2 -> 80 x 96 x 80 at 2 mm
    robust_max_percentile: float = 99.5
    min_background_fraction: float = 0.45
    max_background_fraction: float = 0.92
    version: str = PREPROCESS_VERSION

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return tuple(c // self.pool for c in self.crop_shape)  # type: ignore[return-value]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_shape"] = list(self.output_shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        keys = {f for f in cls.__dataclass_fields__}
        clean = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in keys}
        return cls(**clean)


class InputRejected(ValueError):
    """Raised when a volume does not look like the data the model was trained on."""


def load_volume(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load an Analyze (.img/.hdr) or NIfTI (.nii/.nii.gz) volume.

    Returns the array as stored (no reorientation: Analyze 7.5 carries no
    reliable orientation, and all OASIS T88 volumes share one orientation) and
    the voxel sizes in mm.
    """
    import nibabel as nib

    path = Path(path)
    if path.suffix == ".hdr":
        path = path.with_suffix(".img")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise InputRejected(f"Expected a 3D volume, got an array with shape {data.shape}.")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms  # type: ignore[return-value]


def check_input(vol: np.ndarray, zooms: tuple[float, float, float] | None,
                cfg: PreprocessConfig = PreprocessConfig()) -> dict:
    """Refuse volumes that are clearly outside the training distribution."""
    problems = []
    if tuple(vol.shape) != tuple(cfg.expected_shape):
        problems.append(
            f"shape is {tuple(vol.shape)}, expected {tuple(cfg.expected_shape)} "
            "(atlas-registered 1 mm volume)"
        )
    if zooms is not None and any(abs(z - cfg.expected_voxel_mm) > 0.05 for z in zooms):
        problems.append(f"voxel size is {tuple(round(z, 3) for z in zooms)} mm, expected 1 mm isotropic")
    if not np.isfinite(vol).all():
        problems.append("volume contains NaN or infinite values")
    background = float((vol <= 0).mean()) if vol.size else 1.0
    if not problems and not (cfg.min_background_fraction <= background <= cfg.max_background_fraction):
        problems.append(
            f"{background:.0%} of voxels are background; a skull-stripped brain in atlas space "
            f"has between {cfg.min_background_fraction:.0%} and {cfg.max_background_fraction:.0%}"
        )
    if problems:
        raise InputRejected(
            "This scan does not match what the model was trained on: " + "; ".join(problems) + ". "
            "The model only accepts T1 volumes that are bias-corrected, skull-stripped and "
            "registered to the T88 atlas, as in OASIS-1."
        )
    return {"background_fraction": round(background, 4)}


def preprocess(vol: np.ndarray, cfg: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    """Centre-crop, block-average and intensity-scale a T88 volume.

    Returns float32 with shape ``cfg.output_shape`` and values in [0, 1].
    """
    if tuple(vol.shape) != tuple(cfg.expected_shape):
        raise InputRejected(f"shape {tuple(vol.shape)} is not {tuple(cfg.expected_shape)}")
    vol = np.asarray(vol, dtype=np.float32)
    starts = [(s - c) // 2 for s, c in zip(vol.shape, cfg.crop_shape)]
    sl = tuple(slice(st, st + c) for st, c in zip(starts, cfg.crop_shape))
    x = vol[sl]

    p = cfg.pool
    if p > 1:
        a, b, c = (d // p for d in x.shape)
        x = x.reshape(a, p, b, p, c, p).mean(axis=(1, 3, 5))

    brain = x[x > 0]
    scale = float(np.percentile(brain, cfg.robust_max_percentile)) if brain.size else 0.0
    if scale <= 0:
        raise InputRejected("volume has no positive intensities after cropping")
    x = np.clip(x / scale, 0.0, 1.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_and_preprocess(path: str | Path, cfg: PreprocessConfig = PreprocessConfig(),
                        guard: bool = True) -> tuple[np.ndarray, dict]:
    vol, zooms = load_volume(path)
    info = check_input(vol, zooms, cfg) if guard else {}
    return preprocess(vol, cfg), info
