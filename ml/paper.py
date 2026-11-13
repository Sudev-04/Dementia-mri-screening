"""Paper experiments: a leakage-controlled model ladder with reliability and triage analyses.

    python -m ml.paper --data-root data/oasis1 --csv data/oasis1/oasis.xlsx --out results/paper

Questions this answers, in order:
  1. Model ladder. How much does each step of complexity add, from age alone up to
     image models, under repeated nested cross-validation split by person?
  2. Control. Does the same pipeline score at chance on shuffled labels?
  3. Test-retest. Using the OASIS repeat scans (same person, under 90 days apart),
     how stable are the brain-volume measure and the model score?
  4. Triage. If the model may abstain on uncertain scans, how accurate is it on the
     rest (risk-coverage)? Is acting on it better than referring everyone or no one
     (decision-curve net benefit)?
  5. Data hunger. How does performance change with the number of training people?
  6. Severity. Which dementia severities are detected?

Every hyperparameter is chosen by inner cross-validation on training people only.
All models tried are reported; none is dropped for scoring badly.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common.preprocess import PreprocessConfig, load_volume, preprocess
from ml import metrics as M
from ml.oasis import find_images, read_demographics

warnings.filterwarnings("ignore")
THRESHOLDS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
TAB = ["age", "male", "educ", "etiv", "nwbv"]


# ----------------------------------------------------------------- features
def slice_features(x: np.ndarray, net, device) -> np.ndarray:
    """Frozen ImageNet ResNet-18 features from 16 axial and 16 coronal slices."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(x)
    ax = t[:, :, np.linspace(20, 60, 16).astype(int)].permute(2, 0, 1)
    co = t[:, np.linspace(24, 72, 16).astype(int), :].permute(1, 0, 2)
    out = []
    for stack in (ax, co):
        b = F.interpolate(stack[:, None], size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        b = (b - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        with torch.no_grad():
            out.append(net(b.to(device)).flatten(1).mean(0).cpu().numpy())
    return np.concatenate(out)


def extract(demo: pd.DataFrame, images: dict, cache: Path, use_cnn_features: bool = True) -> dict:
    """Per-session features, cached so a disconnected session can resume."""
    cache.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig()
    net = device = None
    if use_cnn_features:
        try:
            import torch
            import torchvision
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            net = torchvision.models.resnet18(weights="IMAGENET1K_V1")
            net.fc = torch.nn.Identity()
            net = net.eval().to(device)
        except Exception as e:  # no torch / no internet for weights
            print(f"[features] pretrained slice features unavailable ({type(e).__name__}); skipping that model")
            net = None
    feats = {}
    for i, sid in enumerate(demo["session_id"]):
        if sid not in images:
            continue
        f = cache / f"{sid}.npz"
        if f.exists():
            z = np.load(f)
            feats[sid] = {k: z[k] for k in z.files}
            continue
        vol, _ = load_volume(images[sid])
        mask = vol > 0
        scale = np.percentile(vol[mask], 99.5)
        frac = np.array([(vol > t * scale).sum() / mask.sum() for t in THRESHOLDS], dtype=np.float32)
        x = preprocess(vol, cfg)
        a, b, c = (d // 2 for d in x.shape)
        vox = x.reshape(a, 2, b, 2, c, 2).mean(axis=(1, 3, 5)).ravel().astype(np.float16)
        d = {"frac": frac, "vox": vox, "mask": np.array([mask.sum()])}
        if net is not None:
            d["deep"] = slice_features(x, net, device).astype(np.float32)
        np.savez_compressed(f, **d)
        feats[sid] = d
        if (i + 1) % 50 == 0:
            print(f"[features] {i + 1}/{len(demo)} sessions")
    return feats


# ----------------------------------------------------------------- models
def lr(C_grid=(0.01, 0.1, 1, 10)):
    return LogisticRegression(max_iter=2000), {"clf__C": list(C_grid)}


def make_model(names: list[str], widths: list[int]):
    """Pipeline and hyperparameter grid for a feature list. Image blocks (vox, deep)
    come first and are reduced by PCA; any remaining columns are scaled tabular values."""
    clf, grid = lr()
    if names[0] not in ("vox", "deep"):
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)]), grid
    n_img, n_extra = widths[0], sum(widths[1:])
    parts = [("img", Pipeline([("sc", StandardScaler()), ("pca", PCA(random_state=0))]), np.arange(n_img))]
    if n_extra:
        parts.append(("tab", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                      np.arange(n_img, n_img + n_extra)))
    return Pipeline([("ct", ColumnTransformer(parts)), ("clf", clf)]), {**grid, "ct__img__pca__n_components": [5, 10, 20, 40]}


def nested_oof(est, grid, X, y, repeats, folds, seed, inner=3):
    """Repeated nested CV. Returns array (repeats, n) of out-of-fold probabilities."""
    out = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            gs = GridSearchCV(clone(est), grid, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(inner, shuffle=True, random_state=seed + 100 + r))
            gs.fit(X[tr], y[tr])
            out[r, te] = gs.predict_proba(X[te])[:, 1]
    return out


def sens_at_spec(y, p, spec=0.8):
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(fpr <= 1 - spec)
    i = ok[-1]
    return float(tpr[i]), float(min(thr[i], 1.0))


def risk_coverage(y, p):
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    correct = ((p >= 0.5).astype(int) == y)[order]
    cov = np.arange(1, len(y) + 1) / len(y)
    acc = np.cumsum(correct) / np.arange(1, len(y) + 1)
    return cov, acc


def net_benefit(y, p, thresholds):
    n = len(y)
    nb = []
    for t in thresholds:
        pred = p >= t
        tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
        nb.append(tp / n - fp / n * t / (1 - t))
    return np.array(nb)


def icc_agreement(a, b):
    """ICC(2,1), two-way random, absolute agreement, single measurement."""
    x = np.column_stack([a, b]).astype(float)
    n, k = x.shape
    grand = x.mean()
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


# ----------------------------------------------------------------- main
def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--csv", required=True, nargs="+", help="demographics file(s); add the reliability sheet for test-retest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--cnn-run", default=None, help="optional ml.train run folder to include in figures")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    demo = pd.concat([read_demographics(c) for c in args.csv]).drop_duplicates("session_id").reset_index(drop=True)
    # Repeat sessions inherit any missing descriptors from the same person's first session.
    base = demo[demo["session_no"] == 1].set_index("subject_id")
    for col in ("age", "male", "educ", "etiv", "nwbv", "cdr"):
        from_first = demo["subject_id"].map(base[col]) if col in base else None
        if from_first is not None:
            fill = (demo["session_no"] == 2) & (demo[col].isna() if col != "male" else True)
            demo.loc[fill, col] = from_first[fill]
    images = find_images(args.data_root)
    feats = extract(demo, images, Path(args.cache_dir or out / "features"), not args.no_deep)
    demo = demo[demo["session_id"].isin(feats)].reset_index(drop=True)
    F = {k: np.stack([feats[s][k] for s in demo["session_id"]]) for k in feats[demo["session_id"][0]]}
    has_deep = "deep" in F

    # Brain-fraction threshold: the one that best tracks OASIS's own nWBV. Uses no diagnosis labels.
    ok = demo["nwbv"].notna().to_numpy()
    corrs = [float(np.corrcoef(F["frac"][ok, j], demo.loc[ok, "nwbv"])[0, 1]) for j in range(len(THRESHOLDS))]
    j_best = int(np.argmax(corrs))
    demo["brain_fraction"] = F["frac"][:, j_best]

    first = demo["session_no"] == 1
    ref = demo[first & (demo["cdr"].fillna(0) == 0)]            # people without dementia, all ages
    coef = np.polyfit(ref["age"], ref["brain_fraction"], 2)
    resid_sd = float((ref["brain_fraction"] - np.polyval(coef, ref["age"])).std())
    demo["bf_z"] = (demo["brain_fraction"] - np.polyval(coef, demo["age"])) / resid_sd

    cohort = demo[first & (demo["age"] >= args.min_age) & demo["cdr"].notna()].reset_index(drop=True)
    idx = demo.index[first & (demo["age"] >= args.min_age) & demo["cdr"].notna()].to_numpy()
    y = (cohort["cdr"] > 0).astype(int).to_numpy()
    print(f"Cohort: {len(y)} people, {int(y.sum())} with dementia; reference group {len(ref)}; "
          f"brain-fraction threshold {THRESHOLDS[j_best]} (r={corrs[j_best]:.3f} with nWBV)")

    mats = {"age": cohort[["age"]].to_numpy(float), "frac": cohort[["brain_fraction"]].to_numpy(float),
            "tab": cohort[TAB].to_numpy(float), "vox": F["vox"][idx].astype(np.float32)}
    if has_deep:
        mats["deep"] = F["deep"][idx]

    def stacked(names):
        return np.hstack([mats[n] for n in names])

    def model_for(name):
        return make_model(spec[name], [mats[n].shape[1] for n in spec[name]])

    spec = {"A. Age": ["age"], "B. Brain fraction + age": ["frac", "age"],
            "C. Tabular (age, sex, educ, eTIV, nWBV)": ["tab"], "D. Voxel PCA + logistic": ["vox"],
            "F. Voxel PCA + tabular (fusion)": ["vox", "tab"]}
    if has_deep:
        spec["E. Pretrained ResNet-18 slice features"] = ["deep"]

    results, oof_mean, oof_all = {}, {}, {}
    for name in sorted(spec):
        names = spec[name]
        X = stacked(names)
        est, grid = model_for(name)
        P = nested_oof(est, grid, X, y, args.repeats, args.folds, args.seed)
        p = P.mean(0)
        aucs = [M.safe_auc(y, P[r]) for r in range(args.repeats)]
        lo, hi = M.bootstrap_ci(y, p, n_boot=args.bootstrap, seed=args.seed)
        s80, thr80 = sens_at_spec(y, p, 0.8)
        results[name] = {"auc": M.safe_auc(y, p), "auc_ci95": [lo, hi], "auc_repeat_mean": float(np.mean(aucs)),
                         "auc_repeat_sd": float(np.std(aucs)), "brier": M.brier(y, p),
                         "ece": M.expected_calibration_error(y, p), "sens_at_spec80": s80, "threshold_spec80": thr80}
        oof_mean[name], oof_all[name] = p, P
        print(f"{name:<44} AUC {results[name]['auc']:.3f} ({lo:.3f}-{hi:.3f})  repeats {np.mean(aucs):.3f}+/-{np.std(aucs):.3f}")

    ref_name = "C. Tabular (age, sex, educ, eTIV, nWBV)"
    for name in results:
        if name != ref_name:
            dlo, dhi = M.bootstrap_ci(y, oof_mean[name], n_boot=args.bootstrap, seed=args.seed, p2=oof_mean[ref_name])
            results[name]["delta_vs_tabular"] = results[name]["auc"] - results[ref_name]["auc"]
            results[name]["delta_ci95"] = [dlo, dhi]

    image_models = [n for n in results if n[0] in "DEF"]
    # "Best image model" is named by repeat-mean AUC for the later analyses; all models stay in the table.
    best = max(image_models, key=lambda n: results[n]["auc_repeat_mean"])

    # ---- 2. shuffled-label control on the best image model
    names = spec[best]; X = stacked(names)
    est, grid = model_for(best)
    y_perm = np.random.default_rng(args.seed).permutation(y)
    Pp = nested_oof(est, grid, X, y_perm, 1, args.folds, args.seed)
    control = {"model": best, "auc": M.safe_auc(y_perm, Pp[0]),
               "auc_ci95": list(M.bootstrap_ci(y_perm, Pp[0], n_boot=args.bootstrap, seed=args.seed))}
    print(f"Shuffled-label control ({best}): AUC {control['auc']:.3f}")

    # ---- 3. test-retest on repeat scans
    retest = {"n_pairs": 0}
    second = demo[demo["session_no"] == 2]
    pairs = [(demo.index[(demo["subject_id"] == s) & first][0], i) for i, s in zip(second.index, second["subject_id"])
             if ((demo["subject_id"] == s) & first).any()]
    if len(pairs) >= 3:
        i1, i2 = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
        bf1, bf2 = demo["brain_fraction"].to_numpy()[i1], demo["brain_fraction"].to_numpy()[i2]
        # Model stability: fit on cohort people who are NOT in a repeat pair, then score both scans.
        repeat_subjects = set(demo["subject_id"].to_numpy()[i1])
        keep = ~cohort["subject_id"].isin(repeat_subjects).to_numpy()
        gs = GridSearchCV(clone(est), grid, scoring="roc_auc", n_jobs=-1, cv=StratifiedKFold(3, shuffle=True, random_state=args.seed))
        gs.fit(X[keep], y[keep])
        full = {"vox": F["vox"].astype(np.float32), "tab": demo[TAB].to_numpy(float), **({"deep": F["deep"]} if has_deep else {})}
        Xall = np.hstack([full[n] for n in names])
        s1, s2 = gs.predict_proba(Xall[i1])[:, 1], gs.predict_proba(Xall[i2])[:, 1]
        retest = {"n_pairs": len(pairs), "brain_fraction_icc": icc_agreement(bf1, bf2),
                  "brain_fraction_mean_abs_diff": float(np.abs(bf1 - bf2).mean()),
                  "brain_fraction_cv_percent": float(100 * np.mean(np.abs(bf1 - bf2) / np.sqrt(2) / ((bf1 + bf2) / 2))),
                  "model": best, "score_icc": icc_agreement(s1, s2), "score_mean_abs_diff": float(np.abs(s1 - s2).mean()),
                  "score_max_abs_diff": float(np.abs(s1 - s2).max()),
                  "pairs": {"bf1": bf1.tolist(), "bf2": bf2.tolist(), "s1": s1.tolist(), "s2": s2.tolist()}}
        print(f"Test-retest ({len(pairs)} pairs): brain fraction ICC {retest['brain_fraction_icc']:.3f}; "
              f"model score ICC {retest['score_icc']:.3f}, mean |diff| {retest['score_mean_abs_diff']:.3f}")

    # ---- 4. triage: risk-coverage and decision curve
    triage = {}
    for name in (ref_name, best):
        cov, acc = risk_coverage(y, oof_mean[name])
        triage[name] = {"accuracy_full": float(acc[-1]),
                        "accuracy_at_coverage": {str(c): float(acc[int(np.ceil(c * len(y))) - 1]) for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)},
                        "aurc": float(getattr(np, "trapezoid", None)(1 - acc, cov) if hasattr(np, "trapezoid") else np.trapz(1 - acc, cov))}
    thr = np.round(np.arange(0.10, 0.71, 0.05), 2)
    prev = y.mean()
    decision = {"thresholds": thr.tolist(), "refer_all": (prev - (1 - prev) * thr / (1 - thr)).tolist(),
                **{name: net_benefit(y, oof_mean[name], thr).tolist() for name in (ref_name, best)}}

    # ---- 5. learning curve (one repeat, same test folds, training set subsampled)
    fractions = [0.25, 0.5, 0.75, 1.0]
    learning = {name: [] for name in (ref_name, best)}
    for name in learning:
        Xn = stacked(spec[name]); e, g = model_for(name)
        for frac in fractions:
            p = np.zeros(len(y)); sizes = []
            for tr, te in StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(Xn, y):
                rng = np.random.default_rng(args.seed)
                sub = np.concatenate([rng.choice(tr[y[tr] == c], max(6, int(frac * (y[tr] == c).sum())), replace=False) for c in (0, 1)])
                gs = GridSearchCV(clone(e), g, scoring="roc_auc", n_jobs=-1, cv=StratifiedKFold(3, shuffle=True, random_state=args.seed))
                gs.fit(Xn[sub], y[sub]); p[te] = gs.predict_proba(Xn[te])[:, 1]; sizes.append(len(sub))
            learning[name].append({"n_train": int(np.mean(sizes)), "auc": M.safe_auc(y, p)})

    # ---- 6. detection by severity at 80% specificity
    severity = {}
    for name in (ref_name, best):
        t = results[name]["threshold_spec80"]
        severity[name] = {str(c): {"n": int((cohort["cdr"] == c).sum()),
                                   "detected": float((oof_mean[name][(cohort["cdr"] == c).to_numpy()] >= t).mean())}
                          for c in sorted(cohort["cdr"].unique()) if c > 0}

    summary = {
        "synthetic": bool(args.synthetic), "n": int(len(y)), "n_positive": int(y.sum()),
        "protocol": f"{args.repeats}x repeated {args.folds}-fold nested cross-validation split by person; "
                    "hyperparameters by inner 3-fold CV; probabilities averaged over repeats for pooled estimates",
        "brain_fraction": {"threshold": THRESHOLDS[j_best], "corr_with_nwbv": corrs[j_best], "reference_n": int(len(ref)),
                           "age_poly_coef": [float(c) for c in coef], "resid_sd": resid_sd,
                           "reference_age_range": [float(ref["age"].min()), float(ref["age"].max())]},
        "models": results, "best_image_model": best, "shuffled_label_control": control, "test_retest": retest,
        "triage": triage, "decision_curve": decision, "learning_curve": learning, "severity": severity,
        "cohort": {"age_mean": [float(cohort.loc[y == c, "age"].mean()) for c in (0, 1)],
                   "age_sd": [float(cohort.loc[y == c, "age"].std()) for c in (0, 1)],
                   "male": [float(cohort.loc[y == c, "male"].mean()) for c in (0, 1)],
                   "cdr_counts": {str(k): int(v) for k, v in cohort["cdr"].value_counts().sort_index().items()}},
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.DataFrame({"subject_id": cohort["subject_id"], "label": y, "age": cohort["age"], "cdr": cohort["cdr"],
                  **{f"p_{n[0]}": oof_mean[n] for n in results}}).to_csv(out / "oof.csv", index=False)
    demo[first][["age", "cdr", "brain_fraction", "bf_z", "nwbv"]].to_csv(out / "brain_fraction.csv", index=False)

    from ml.figures import make_all
    make_all(out, summary, y, oof_mean, cohort, demo[first], coef, resid_sd, args.cnn_run)
    print(f"Saved results and figures to {out}")
    return summary


if __name__ == "__main__":
    main()
