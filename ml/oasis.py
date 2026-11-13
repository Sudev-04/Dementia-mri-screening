"""Build the study cohort from an OASIS-1 download.

Three decisions here protect the evaluation from the mistakes that inflate most
published Alzheimer's MRI results:

1. One row per SUBJECT. The 20 repeat sessions (``_MR2``) are dropped, so the
   same brain can never sit on both sides of a split.
2. Only people aged 60+ with a recorded CDR. In OASIS-1 nobody under 60 has
   dementia and most young adults have no CDR at all; including them lets a
   model score well by detecting age.
3. The label is the Clinical Dementia Rating, assigned by clinicians without the
   MRI: CDR 0 = no dementia, CDR 0.5 or more = dementia (very mild to moderate).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

IMAGE_GLOB = "*_t88_masked_gfc.img"
SESSION_RE = re.compile(r"^(OAS1_\d{4})_MR(\d)$")
TABULAR_FEATURES = ["age", "male", "educ", "etiv", "nwbv"]


def read_demographics(path: str | Path) -> pd.DataFrame:
    """Read the OASIS-1 spreadsheet (CSV or XLSX) into tidy columns."""
    path = Path(path)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "ID" not in raw.columns:
        raise ValueError(f"{path.name} has no 'ID' column; is this the OASIS-1 demographics file?")
    # The OASIS reliability sheet (repeat scans) has no Age or CDR columns; those rows
    # are completed from the same person's first session by the caller.
    is_repeat_sheet = raw["ID"].astype(str).str.contains("_MR2").all()
    absent = [c for c in ("Age", "CDR") if c not in raw.columns]
    if absent and not is_repeat_sheet:
        raise ValueError(f"{path.name} is missing required column(s) {absent}. Found: {list(raw.columns)}")

    def num(name: str) -> pd.Series:
        if name not in raw.columns:
            print(f"[cohort] column '{name}' not found in {path.name}; treating it as missing")
            return pd.Series([float("nan")] * len(raw))
        return pd.to_numeric(raw[name], errors="coerce")

    sex = raw["M/F"].astype(str).str.strip().str.upper() if "M/F" in raw.columns else pd.Series([""] * len(raw))
    parsed = raw["ID"].astype(str).str.strip().str.extract(SESSION_RE)
    df = pd.DataFrame({
        "session_id": raw["ID"].astype(str).str.strip(),
        "subject_id": parsed[0],
        "session_no": pd.to_numeric(parsed[1], errors="coerce"),
        "age": num("Age"),
        "male": (sex == "M").astype(int),
        "educ": num("Educ"),
        "mmse": num("MMSE"),
        "cdr": num("CDR"),
        "etiv": num("eTIV"),
        "nwbv": num("nWBV"),
    })
    return df.dropna(subset=["subject_id"]).reset_index(drop=True)


def find_images(data_root: str | Path) -> dict[str, Path]:
    """Map session id (e.g. OAS1_0001_MR1) to its atlas-registered masked image."""
    found: dict[str, Path] = {}
    for p in sorted(Path(data_root).rglob(IMAGE_GLOB)):
        session = next((part for part in p.parts if SESSION_RE.match(part)), None)
        if session and "T88_111" in p.parts and session not in found:
            found[session] = p
    return found


def build_cohort(csv_path: str | Path, data_root: str | Path, min_age: int = 60,
                 cached_subjects: set[str] | None = None) -> pd.DataFrame:
    """Return one row per subject with image path, label and tabular features.

    ``cached_subjects`` lists people whose preprocessed volume is already cached;
    they are kept even when the raw image is absent, so a reset machine can resume
    training from the cache without downloading the scans again. The cohort rules
    (one session per person, age, CDR) are applied identically either way.
    """
    demo = read_demographics(csv_path)
    images = find_images(data_root) if Path(data_root).exists() else {}
    cached_subjects = cached_subjects or set()

    df = demo[demo["session_no"] == 1].copy()            # drop repeat sessions
    df = df[(df["age"] >= min_age) & df["cdr"].notna()]  # clinically rated older adults
    df["label"] = (df["cdr"] > 0).astype(int)
    df["image"] = [str(images[s]) if s in images else ("<cached>" if subj in cached_subjects else None)
                   for s, subj in zip(df["session_id"], df["subject_id"])]

    missing = df[df["image"].isna()]
    if len(missing):
        print(f"[cohort] {len(missing)} eligible subjects have no image under {data_root} "
              f"(first few: {missing['session_id'].head(5).tolist()}). They are skipped.")
    df = df.dropna(subset=["image"]).sort_values("subject_id").reset_index(drop=True)

    if df["subject_id"].duplicated().any():
        raise AssertionError("A subject appears more than once in the cohort; this would leak.")
    if len(df) == 0:
        raise ValueError("No eligible subjects with images were found. Check --data-root and --csv.")
    return df


def describe(df: pd.DataFrame) -> str:
    rows = []
    for label, name in ((0, "No dementia (CDR 0)"), (1, "Dementia (CDR >= 0.5)")):
        g = df[df["label"] == label]
        if len(g) == 0:
            rows.append(f"  {name:<24} n=0")
            continue
        rows.append(
            f"  {name:<24} n={len(g):<4} age {g['age'].mean():.1f} +/- {g['age'].std():.1f}  "
            f"male {g['male'].mean():.0%}  nWBV {g['nwbv'].mean():.3f}"
        )
    cdr = df["cdr"].value_counts().sort_index().to_dict()
    return "\n".join([f"Cohort: {len(df)} subjects, one scan each"] + rows + [f"  CDR counts: {cdr}"])
