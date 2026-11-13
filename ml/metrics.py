"""Evaluation helpers: discrimination, operating points, calibration."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def bootstrap_ci(y, p, fn=safe_auc, n_boot: int = 2000, seed: int = 0, p2=None):
    """Stratified bootstrap 95% interval. With ``p2`` given, returns the interval
    of ``fn(y, p) - fn(y, p2)`` on the same resamples (paired comparison)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v = fn(y[idx], p[idx])
        if p2 is not None:
            v -= fn(y[idx], np.asarray(p2)[idx])
        vals.append(v)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def confusion_at(y, p, threshold: float) -> dict:
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "sensitivity": sens, "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "ppv": tp / (tp + fp) if tp + fp else float("nan"),
        "npv": tn / (tn + fn) if tn + fn else float("nan"),
    }


def youden_threshold(y, p) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thr[best], 0.0, 1.0))


def threshold_for_sensitivity(y, p, target: float = 0.85) -> float:
    """Highest threshold whose sensitivity is at least ``target``."""
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(tpr >= target)
    return float(np.clip(thr[ok[0]], 0.0, 1.0)) if len(ok) else float(np.min(p))


def fit_temperature(logits, y) -> float:
    """Single-parameter temperature scaling, bounded so a tiny validation set
    cannot produce an extreme value."""
    logits, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return 1.0

    def nll(log_t):
        p = np.clip(sigmoid(logits / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(nll, bounds=(np.log(0.5), np.log(5.0)), method="bounded")
    return float(np.exp(res.x))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def expected_calibration_error(y, p, n_bins: int = 5) -> float:
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def summarise(y, p, threshold: float, n_boot: int = 2000, seed: int = 0) -> dict:
    lo, hi = bootstrap_ci(y, p, n_boot=n_boot, seed=seed)
    return {
        "n": int(len(y)), "n_positive": int(np.sum(y)),
        "auc": safe_auc(y, p), "auc_ci95": [lo, hi],
        "brier": brier(y, p), "ece": expected_calibration_error(y, p),
        "threshold": float(threshold), **confusion_at(y, p, threshold),
    }
