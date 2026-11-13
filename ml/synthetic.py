"""Generate a fake OASIS-1 style dataset for tests and for trying the app.

    python -m ml.synthetic --out data/synthetic --subjects 40

The volumes are noise-textured ellipsoids with a dark central blob that is, on
average, larger in the "dementia" group. They have the right shape, file layout
and file format to exercise every line of the pipeline. They contain no
anatomy: a model trained on them knows nothing about brains, and anything
exported from such a run is flagged ``synthetic`` and labelled as a demo in the app.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

SHAPE = (176, 208, 176)


def make_volume(rng: np.random.Generator, dementia: bool) -> tuple[np.ndarray, float]:
    gx, gy, gz = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in SHAPE), indexing="ij")
    cx, cy, cz = (s / 2 for s in SHAPE)
    brain = ((gx - cx) / 66) ** 2 + ((gy - cy) / 84) ** 2 + ((gz - cz) / 62) ** 2 <= 1.0
    radius = float(np.clip(rng.normal(16.0 if dementia else 11.0, 2.5), 5, 24))
    blob = ((gx - cx) / radius) ** 2 + ((gy - cy) / (radius * 1.6)) ** 2 + ((gz - cz) / radius) ** 2 <= 1.0
    coarse = rng.normal(0, 1, tuple(s // 8 for s in SHAPE)).astype(np.float32)
    texture = zoom(coarse, 8, order=1)
    vol = np.where(brain, 900 + 120 * texture + rng.normal(0, 25, SHAPE), 0)
    vol = np.where(blob, vol * 0.25, vol)
    nwbv = float(brain.sum() - blob.sum()) / float(brain.sum())
    return np.clip(vol, 0, 4000).astype(np.int16), nwbv


def generate(out: str | Path, subjects: int = 40, seed: int = 0) -> tuple[Path, Path]:
    out = Path(out); rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, subjects + 1):
        dementia = i % 2 == 0
        sid = f"OAS1_{i:04d}_MR1"
        vol, nwbv = make_volume(rng, dementia)
        d = out / "disc1" / sid / "PROCESSED" / "MPRAGE" / "T88_111"
        d.mkdir(parents=True, exist_ok=True)
        img = nib.AnalyzeImage(vol, np.eye(4))
        img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img, str(d / f"{sid}_mpr_n4_anon_111_t88_masked_gfc.img"))
        rows.append({"ID": sid, "M/F": rng.choice(["M", "F"]), "Hand": "R",
                     "Age": int(rng.integers(62, 92)), "Educ": int(rng.integers(1, 6)), "SES": 2,
                     "MMSE": 22 if dementia else 29, "CDR": float(rng.choice([0.5, 1.0])) if dementia else 0.0,
                     "eTIV": int(rng.normal(1480, 150)), "nWBV": round(nwbv * 0.75, 3), "ASF": 1.2, "Delay": "N/A"})
    # Repeat sessions (same "person", slightly different noise), as in the OASIS reliability set.
    for i in range(1, min(4, subjects) + 1):
        src = out / "disc1" / f"OAS1_{i:04d}_MR1" / "PROCESSED" / "MPRAGE" / "T88_111"
        vol = np.asarray(nib.load(str(next(src.glob("*.img")))).dataobj).squeeze().astype(np.float32)
        vol = np.where(vol > 0, np.clip(vol + rng.normal(0, 15, vol.shape), 1, 4000), 0).astype(np.int16)
        sid = f"OAS1_{i:04d}_MR2"
        d = out / "disc1" / sid / "PROCESSED" / "MPRAGE" / "T88_111"; d.mkdir(parents=True, exist_ok=True)
        img = nib.AnalyzeImage(vol, np.eye(4)); img.header.set_zooms((1.0, 1.0, 1.0))
        nib.save(img, str(d / f"{sid}_mpr_n4_anon_111_t88_masked_gfc.img"))
        rows.append({**rows[i - 1], "ID": sid, "Delay": 30})
    rows.append({**rows[0], "ID": "OAS1_9999_MR1", "Age": 23, "CDR": None})
    csv = out / "oasis_cross-sectional.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return out, csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--subjects", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, csv = generate(a.out, a.subjects, a.seed)
    print(f"wrote {a.subjects} synthetic subjects to {root} and {csv}")
