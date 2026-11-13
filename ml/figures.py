"""Report figures (IEEE single-column width, 300 dpi PNG and vector PDF) and result tables.

Identity is never carried by colour alone: every series also has its own line
style or marker, so the figures survive greyscale printing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

BLUE, ORANGE, GREEN, INK, GREY = "#0072B2", "#D55E00", "#009E73", "#222222", "#8A8A8A"
STYLES = [("-", "o"), ("--", "s"), ("-.", "^"), (":", "D"), ((0, (5, 1)), "v"), ((0, (3, 1, 1, 1)), "P")]
SERIES = [GREY, GREEN, BLUE, ORANGE, "#CC79A7", INK]
W = 3.5  # inches, IEEE single column


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
        "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.5,
    })


def _save(fig, out: Path, name: str):
    d = out / "figures"; d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png"); fig.savefig(d / f"{name}.pdf"); plt.close(fig)


def short(name: str) -> str:
    return name.split(" (")[0]


def make_all(out, S, y, oof, cohort, first, coef, resid_sd, cnn_run=None):
    out = Path(out); _style()
    names = sorted(S["models"]); ref = "C. Tabular (age, sex, educ, eTIV, nWBV)"; best = S["best_image_model"]
    tag = "SYNTHETIC TEST DATA. " if S.get("synthetic") else ""

    # 1. cohort
    fig, ax = plt.subplots(figsize=(W, 2.3))
    bins = np.arange(60, 100, 4)
    ax.hist([cohort.loc[y == 0, "age"], cohort.loc[y == 1, "age"]], bins=bins, color=[BLUE, ORANGE],
            label=[f"No dementia (n={int((y == 0).sum())})", f"Dementia, CDR$\\geq$0.5 (n={int(y.sum())})"],
            edgecolor="white", linewidth=0.6)
    for patch in ax.patches[len(bins) - 1:]:
        patch.set_hatch("///")
    ax.set(xlabel="Age (years)", ylabel="People"); ax.legend()
    _save(fig, out, "fig01_cohort_age")

    # 2. brain fraction against age with the reference band
    fig, ax = plt.subplots(figsize=(W, 2.6))
    nd = first[first["cdr"].fillna(0) == 0]; de = first[first["cdr"].fillna(0) > 0]
    ax.scatter(nd["age"], nd["brain_fraction"], s=9, c=BLUE, marker="o", linewidths=0, alpha=0.75, label="No dementia")
    ax.scatter(de["age"], de["brain_fraction"], s=13, c=ORANGE, marker="^", linewidths=0, alpha=0.9, label="Dementia")
    xs = np.linspace(first["age"].min(), first["age"].max(), 100); mid = np.polyval(coef, xs)
    ax.plot(xs, mid, color=INK, lw=1.2, label="Reference (no dementia)")
    ax.fill_between(xs, mid - 2 * resid_sd, mid + 2 * resid_sd, color=GREY, alpha=0.18, lw=0, label="$\\pm$2 SD")
    ax.set(xlabel="Age (years)", ylabel="Brain tissue fraction"); ax.legend(loc="lower left")
    _save(fig, out, "fig02_brain_fraction_by_age")

    # 3. ROC
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=(0, (1, 2)))
    for i, n in enumerate(names):
        fpr, tpr, _ = roc_curve(y, oof[n])
        ax.plot(fpr, tpr, color=SERIES[i % 6], ls=STYLES[i % 6][0], label=f"{short(n)} ({S['models'][n]['auc']:.2f})")
    ax.set(xlabel="1 $-$ specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01)); ax.set_aspect("equal")
    ax.legend(loc="lower right", title="Model (AUC)", title_fontsize=6.5)
    _save(fig, out, "fig03_roc")

    # 4. AUC with intervals
    fig, ax = plt.subplots(figsize=(W, 0.36 * len(names) + 0.9))
    for i, n in enumerate(names[::-1]):
        m = S["models"][n]
        ax.errorbar(m["auc"], i, xerr=[[m["auc"] - m["auc_ci95"][0]], [m["auc_ci95"][1] - m["auc"]]], fmt="o",
                    color=ORANGE if n == best else BLUE if n == ref else INK, ms=4, capsize=2.5, lw=1)
        ax.text(m["auc_ci95"][1] + 0.012, i, f"{m['auc']:.3f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)))
    ax.set_yticks(range(len(names))); ax.set_yticklabels([short(n) for n in names[::-1]])
    ax.set(xlabel="Out-of-fold AUC (95% bootstrap interval)", xlim=(0.4, 1.0)); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig04_model_ladder")

    # 5. calibration
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=(0, (1, 2)), label="Perfect")
    for n, c, (ls, mk) in ((ref, BLUE, STYLES[0]), (best, ORANGE, STYLES[1])):
        frac, mean = calibration_curve(y, oof[n], n_bins=5, strategy="quantile")
        ax.plot(mean, frac, color=c, ls=ls, marker=mk, ms=4, label=f"{short(n)} (Brier {S['models'][n]['brier']:.3f})")
    ax.set(xlabel="Predicted probability", ylabel="Observed fraction with dementia", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    _save(fig, out, "fig05_calibration")

    # 6. risk-coverage
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for n, c, (ls, mk) in ((ref, BLUE, STYLES[0]), (best, ORANGE, STYLES[1])):
        conf = np.abs(oof[n] - 0.5); order = np.argsort(-conf)
        acc = np.cumsum(((oof[n] >= 0.5).astype(int) == y)[order]) / np.arange(1, len(y) + 1)
        cov = np.arange(1, len(y) + 1) / len(y); keep = cov >= 0.2
        ax.plot(cov[keep], acc[keep], color=c, ls=ls, label=short(n))
    ax.set(xlabel="Coverage (fraction of scans the model answers)", ylabel="Accuracy on answered scans", xlim=(0.2, 1))
    ax.legend(loc="lower left")
    _save(fig, out, "fig06_risk_coverage")

    # 7. decision curve
    D = S["decision_curve"]; t = np.array(D["thresholds"])
    fig, ax = plt.subplots(figsize=(W, 2.5))
    ax.plot(t, D["refer_all"], color=GREY, ls=(0, (1, 2)), label="Refer everyone")
    ax.axhline(0, color=INK, lw=0.7, label="Refer no one")
    ax.plot(t, D[ref], color=BLUE, ls=STYLES[0][0], marker="o", ms=3, label=short(ref))
    ax.plot(t, D[best], color=ORANGE, ls=STYLES[1][0], marker="s", ms=3, label=short(best))
    ax.set(xlabel="Threshold probability for referral", ylabel="Net benefit"); ax.set_ylim(bottom=-0.05); ax.legend()
    _save(fig, out, "fig07_decision_curve")

    # 8. learning curve
    fig, ax = plt.subplots(figsize=(W, 2.4))
    for n, c, (ls, mk) in ((ref, BLUE, STYLES[0]), (best, ORANGE, STYLES[1])):
        L = S["learning_curve"][n]
        ax.plot([p["n_train"] for p in L], [p["auc"] for p in L], color=c, ls=ls, marker=mk, ms=4, label=short(n))
    ax.set(xlabel="Training people per fold", ylabel="Out-of-fold AUC"); ax.legend(loc="lower right")
    _save(fig, out, "fig08_learning_curve")

    # 9. test-retest
    R = S["test_retest"]
    if R.get("n_pairs", 0) >= 3:
        P = R["pairs"]; fig, axs = plt.subplots(1, 2, figsize=(W, 1.9))
        for ax, a, b, lab, icc in ((axs[0], P["bf1"], P["bf2"], "Brain fraction", R["brain_fraction_icc"]),
                                   (axs[1], P["s1"], P["s2"], "Model score", R["score_icc"])):
            a, b = np.array(a), np.array(b); lo, hi = min(a.min(), b.min()), max(a.max(), b.max()); pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=0.7, ls=(0, (1, 2)))
            ax.scatter(a, b, s=12, c=BLUE, linewidths=0)
            ax.set(xlabel=f"{lab}, scan 1", ylabel=f"{lab}, scan 2", title=f"ICC {icc:.3f} (n={R['n_pairs']})")
        fig.tight_layout(w_pad=1.0)
        _save(fig, out, "fig09_test_retest")

    # 10. detection by severity
    sev = S["severity"]; cdrs = sorted(sev[ref], key=float)
    fig, ax = plt.subplots(figsize=(W, 2.2)); x = np.arange(len(cdrs)); w = 0.38
    for k, (n, c, hatch) in enumerate(((ref, BLUE, None), (best, ORANGE, "///"))):
        vals = [sev[n][c_]["detected"] for c_ in cdrs]
        bars = ax.bar(x + (k - 0.5) * w, vals, w - 0.03, color=c, hatch=hatch, edgecolor="white", linewidth=0.6, label=short(n))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"CDR {c_}\n(n={sev[ref][c_]['n']})" for c_ in cdrs])
    ax.set(ylabel="Detected at 80% specificity", ylim=(0, 1.12)); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    _save(fig, out, "fig10_detection_by_severity")

    # 11. CNN training curves, if a from-scratch run is supplied
    if cnn_run and Path(cnn_run).exists():
        fig, axs = plt.subplots(1, 2, figsize=(W, 1.9))
        for k, h in enumerate(sorted(Path(cnn_run).glob("fold*/history.json"))):
            H = json.loads(h.read_text()); ep = [e["epoch"] for e in H]
            axs[0].plot(ep, [e["train_loss"] for e in H], color=SERIES[k % 6], ls=STYLES[k % 6][0], lw=0.9, label=f"Fold {k}")
            axs[1].plot(ep, [e["val_auc"] for e in H], color=SERIES[k % 6], ls=STYLES[k % 6][0], lw=0.9)
        axs[0].axhline(np.log(2), color=GREY, lw=0.7, ls=(0, (1, 2))); axs[1].axhline(0.5, color=GREY, lw=0.7, ls=(0, (1, 2)))
        axs[0].set(xlabel="Epoch", ylabel="Training loss"); axs[1].set(xlabel="Epoch", ylabel="Validation AUC")
        axs[0].legend(ncol=2, fontsize=5.5); fig.tight_layout(w_pad=1.0)
        _save(fig, out, "fig11_cnn_training")

    # tables
    L = [f"# {tag}Results tables", "", f"N = {S['n']} ({S['n_positive']} with dementia). {S['protocol']}.", "",
         "## Table I. Model ladder", "", "| Model | AUC (95% CI) | Repeats mean (SD) | Diff. vs tabular (95% CI) | Brier | Sens. at 80% spec. |", "|---|---|---|---|---|---|"]
    for n in names:
        m = S["models"][n]
        d = f"{m['delta_vs_tabular']:+.3f} ({m['delta_ci95'][0]:+.3f} to {m['delta_ci95'][1]:+.3f})" if "delta_vs_tabular" in m else "reference"
        L.append(f"| {n} | {m['auc']:.3f} ({m['auc_ci95'][0]:.3f}-{m['auc_ci95'][1]:.3f}) | {m['auc_repeat_mean']:.3f} ({m['auc_repeat_sd']:.3f}) | {d} | {m['brier']:.3f} | {m['sens_at_spec80']:.3f} |")
    c = S["shuffled_label_control"]
    L += ["", f"Shuffled-label control ({c['model']}): AUC {c['auc']:.3f} ({c['auc_ci95'][0]:.3f}-{c['auc_ci95'][1]:.3f}).", ""]
    if R.get("n_pairs", 0) >= 3:
        L += ["## Table II. Test-retest (repeat scans of the same person)", "", "| Quantity | ICC(2,1) | Mean abs. difference |", "|---|---|---|",
              f"| Brain fraction | {R['brain_fraction_icc']:.3f} | {R['brain_fraction_mean_abs_diff']:.4f} (CV {R['brain_fraction_cv_percent']:.2f}%) |",
              f"| Model score ({short(R['model'])}) | {R['score_icc']:.3f} | {R['score_mean_abs_diff']:.3f} (max {R['score_max_abs_diff']:.3f}) |", ""]
    L += ["## Table III. Selective prediction (accuracy when the model may abstain)", "", "| Model | 50% | 60% | 70% | 80% | 90% | 100% coverage |", "|---|---|---|---|---|---|---|"]
    for n in (ref, best):
        a = S["triage"][n]["accuracy_at_coverage"]
        L.append(f"| {short(n)} | " + " | ".join(f"{a[k]:.3f}" for k in ("0.5", "0.6", "0.7", "0.8", "0.9", "1.0")) + " |")
    b = S["brain_fraction"]
    L += ["", f"Brain fraction: intensity threshold {b['threshold']} of the robust maximum; r = {b['corr_with_nwbv']:.3f} with OASIS nWBV; "
              f"reference from {b['reference_n']} people without dementia aged {b['reference_age_range'][0]:.0f}-{b['reference_age_range'][1]:.0f}."]
    (out / "tables.md").write_text("\n".join(L) + "\n")
