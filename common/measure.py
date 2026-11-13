"""Brain tissue fraction and its age-referenced z-score.

A single, reproducible number from a skull-stripped, atlas-registered T1 scan:
the share of brain-mask voxels that are tissue rather than fluid. In the paper
experiments it tracked OASIS's own normalised brain volume (r = 0.96), was
stable across rescans (ICC 0.98) and, with age, separated the groups as well as
any image model tried. It is a measurement, not a diagnosis.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "models" / "reference" / "brain_fraction_reference.json"


@lru_cache(maxsize=1)
def load_reference() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def brain_fraction(vol: np.ndarray, threshold: float) -> float:
    mask = vol > 0
    if not mask.any():
        raise ValueError("volume has no brain voxels")
    scale = float(np.percentile(vol[mask], 99.5))
    return float((vol > threshold * scale).sum() / mask.sum())


def band(z: float) -> str:
    if z >= -1.0:
        return "usual"
    return "lower" if z >= -2.0 else "well_below"


def assess(vol: np.ndarray, age: float, ref: dict | None = None) -> dict:
    """Measure the scan and place it against people of the same age without dementia."""
    ref = ref or load_reference()
    frac = brain_fraction(vol, ref["threshold"])
    a, b, c = ref["age_poly_coef"]
    expected = a * age * age + b * age + c
    sd = ref["resid_sd_60plus"] if age >= 60 else ref["resid_sd"]
    z = (frac - expected) / sd
    lo, hi = ref["age_range"]
    return {
        "brain_fraction": round(frac, 4),
        "expected_for_age": round(expected, 4),
        "z": round(z, 2),
        "percentile": round(100 * 0.5 * (1 + math.erf(z / math.sqrt(2))), 1),
        "band": band(z),
        "age": age,
        "age_in_reference_range": bool(lo <= age <= hi),
        "reference_n": ref["reference_n"],
    }
