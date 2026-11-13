"""From-scratch 3D CNN under the same repeated protocol as the model ladder.

    python -m ml.cnn_repeated --data-root data/oasis1 --csv data/oasis1/main.xlsx --out results/cnn --paper-oof results/paper/oof.csv

Runs ml.train once per repeat with the seeds the ladder used (seed, seed+1, ...),
averages each person's out-of-fold probability across repeats, and reports the
pooled AUC with a bootstrap interval, the per-repeat spread and, when the ladder's
predictions are supplied, the paired difference against the tabular baseline.
The architecture and training settings are the ones fixed before any results
were seen (ml.train defaults); nothing is tuned here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml import metrics as M
from ml import train


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--paper-oof", default=None, help="oof.csv from ml.paper, for the paired comparison")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("extra", nargs="*", help="extra arguments passed to ml.train (after --)")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    probs, aucs, label, runs = [], [], None, []
    for r in range(args.repeats):
        run = out / f"repeat{r}"
        done = run / "metrics.json"
        if not done.exists():                      # resumable after a disconnect
            train.main(["--data-root", args.data_root, "--csv", args.csv, "--out", str(run), "--cache-dir", str(out / "cache"),
                        "--seed", str(args.seed + r), "--bootstrap", "200", *(["--synthetic"] if args.synthetic else []), *args.extra])
        oof = pd.read_csv(run / "oof_predictions.csv").sort_values("subject_id").reset_index(drop=True)
        m = json.loads(done.read_text())
        label = oof[["subject_id", "label"]] if label is None else label
        assert (oof["subject_id"] == label["subject_id"]).all()
        probs.append(oof["prob"].to_numpy()); aucs.append(M.safe_auc(oof["label"], oof["prob"]))
        runs.append({"repeat": r, "seed": args.seed + r, "auc_oof": aucs[-1], "auc_fold_mean": m["cnn"]["auc_fold_mean"],
                     "fold_aucs": [f["test_auc"] for f in m["folds"]], "best_epochs": [f["best_epoch"] for f in m["folds"]]})
        print(f"[repeat {r}] pooled AUC {aucs[-1]:.3f}; per-fold mean {m['cnn']['auc_fold_mean']:.3f}")

    y = label["label"].to_numpy(); p = np.mean(probs, axis=0)
    lo, hi = M.bootstrap_ci(y, p, n_boot=args.bootstrap, seed=args.seed)
    fpr_ok = None
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p); fpr_ok = float(tpr[np.flatnonzero(fpr <= 0.2)[-1]])
    res = {"model": "G. From-scratch 3D CNN", "synthetic": bool(args.synthetic), "n": int(len(y)), "n_positive": int(y.sum()),
           "repeats": args.repeats, "auc": M.safe_auc(y, p), "auc_ci95": [lo, hi], "auc_repeat_mean": float(np.mean(aucs)),
           "auc_repeat_sd": float(np.std(aucs)), "auc_fold_mean_over_repeats": float(np.mean([r["auc_fold_mean"] for r in runs])),
           "brier": M.brier(y, p), "sens_at_spec80": fpr_ok, "runs": runs}
    if args.paper_oof and Path(args.paper_oof).exists():
        ref = pd.read_csv(args.paper_oof).sort_values("subject_id").reset_index(drop=True)
        if len(ref) == len(label) and (ref["subject_id"] == label["subject_id"]).all():
            dlo, dhi = M.bootstrap_ci(y, p, n_boot=args.bootstrap, seed=args.seed, p2=ref["p_C"].to_numpy())
            res["delta_vs_tabular"] = res["auc"] - M.safe_auc(y, ref["p_C"]); res["delta_ci95"] = [dlo, dhi]
        else:
            res["delta_note"] = "cohorts differ; paired comparison skipped"
    (out / "cnn_repeated.json").write_text(json.dumps(res, indent=2, default=float))
    pd.DataFrame({"subject_id": label["subject_id"], "label": y, "p_G": p}).to_csv(out / "cnn_oof.csv", index=False)

    # Training curves from the first repeat (IEEE single-column figure).
    from ml.figures import GREY, SERIES, STYLES, W, _save, _style
    import matplotlib.pyplot as plt
    _style(); fig, axs = plt.subplots(1, 2, figsize=(W, 1.9))
    for k, h in enumerate(sorted((out / "repeat0").glob("fold*/history.json"))):
        H = json.loads(h.read_text()); ep = [e["epoch"] for e in H]
        axs[0].plot(ep, [e["train_loss"] for e in H], color=SERIES[k % 6], ls=STYLES[k % 6][0], lw=0.9, label=f"Fold {k}")
        axs[1].plot(ep, [e["val_auc"] for e in H], color=SERIES[k % 6], ls=STYLES[k % 6][0], lw=0.9)
    axs[0].axhline(np.log(2), color=GREY, lw=0.7, ls=(0, (1, 2))); axs[1].axhline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)))
    axs[0].set(xlabel="Epoch", ylabel="Training loss"); axs[1].set(xlabel="Epoch", ylabel="Validation AUC")
    axs[0].legend(ncol=2, fontsize=5.5); fig.tight_layout(w_pad=1.0)
    _save(fig, out, "fig11_cnn_training")

    d = f"; vs tabular {res['delta_vs_tabular']:+.3f} ({res['delta_ci95'][0]:+.3f} to {res['delta_ci95'][1]:+.3f})" if "delta_vs_tabular" in res else ""
    print(f"CNN repeated: AUC {res['auc']:.3f} ({lo:.3f}-{hi:.3f}); repeats {res['auc_repeat_mean']:.3f}+/-{res['auc_repeat_sd']:.3f}{d}")
    print(f"Saved to {out}")
    return res


if __name__ == "__main__":
    main()
