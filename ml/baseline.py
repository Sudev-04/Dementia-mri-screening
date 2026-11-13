"""Simple baselines the CNN has to beat to be worth its complexity.

* ``age_only``  - logistic regression on age. If this scores well, the cohort
  lets a model cheat on age.
* ``tabular``   - age, sex, education, head size (eTIV) and normalised whole
  brain volume (nWBV). nWBV is a single number already derived from the same
  MRI, so this asks: does a 3D CNN add anything beyond "how much brain is left"?

MMSE is deliberately excluded: it is a cognitive test closely tied to the CDR
label, so using it would answer a different (and much easier) question.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml import metrics as M
from ml.oasis import TABULAR_FEATURES


def run_baselines(df, y, splitter, seed: int, cnn_oof=None, n_boot: int = 2000) -> dict:
    out = {}
    for name, cols in (("age_only", ["age"]), ("tabular", TABULAR_FEATURES)):
        X = df[cols].to_numpy(dtype=float)
        oof = np.full(len(y), np.nan)
        for tr, te in splitter.split(X, y):      # identical outer folds to the CNN
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(max_iter=1000))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        lo, hi = M.bootstrap_ci(y, oof, n_boot=n_boot, seed=seed)
        entry = {"features": cols, "auc_oof": M.safe_auc(y, oof), "auc_ci95": [lo, hi]}
        if cnn_oof is not None:
            dlo, dhi = M.bootstrap_ci(y, cnn_oof, n_boot=n_boot, seed=seed, p2=oof)
            entry["cnn_minus_baseline_auc"] = M.safe_auc(y, cnn_oof) - entry["auc_oof"]
            entry["difference_ci95"] = [dlo, dhi]
        out[name] = entry
    return out
