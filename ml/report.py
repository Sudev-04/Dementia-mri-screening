"""Turn a run's metrics.json into a Markdown results section.

    python -m ml.report runs/oasis_v1 > docs/RESULTS.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def render(metrics: dict) -> str:
    c, dep, b = metrics["cnn"], metrics["deployment_threshold"], metrics.get("baselines", {})
    v = c["at_validation_thresholds"]
    lines = ["# Results", ""]
    if metrics.get("synthetic"):
        lines += ["> **Synthetic data.** These numbers describe a pipeline test on generated volumes. "
                  "They say nothing about real brains.", ""]
    if metrics.get("labels_shuffled"):
        lines += ["> **Shuffled-label sanity check.** An AUC near 0.5 here is the expected, healthy result.", ""]
    lines += [
        f"Task: {metrics['task']}.  ",
        f"Subjects: {metrics['n_subjects']} ({metrics['n_positive']} with dementia), one scan per person.  ",
        f"Evaluation: {len(metrics['folds'])}-fold cross-validation split by person; every number below "
        "comes from predictions on people the model had not seen.", "",
        "| Model | AUC (95% CI) |", "|---|---|",
        f"| 3D CNN ({metrics['config']['arch']}, {metrics['config']['parameters']:,} weights) | "
        f"{c['auc_oof']:.3f} ({c['auc_ci95'][0]:.3f} to {c['auc_ci95'][1]:.3f}) |",
    ]
    for name, label in (("tabular", "Logistic regression: age, sex, education, eTIV, nWBV"),
                        ("age_only", "Logistic regression: age only")):
        if name in b:
            lines.append(f"| {label} | {b[name]['auc_oof']:.3f} "
                         f"({b[name]['auc_ci95'][0]:.3f} to {b[name]['auc_ci95'][1]:.3f}) |")
    if "tabular" in b:
        d = b["tabular"]
        lines += ["", f"CNN minus tabular baseline: {d['cnn_minus_baseline_auc']:+.3f} AUC "
                      f"(95% CI {d['difference_ci95'][0]:+.3f} to {d['difference_ci95'][1]:+.3f}). "
                      "If this interval includes zero, the CNN has not been shown to beat a five-number model."]
    lines += [
        "", f"Per-fold AUC: {c['auc_fold_mean']:.3f} +/- {c['auc_fold_sd']:.3f}.  ",
        f"At thresholds chosen on validation data: sensitivity {v['sensitivity']:.3f}, "
        f"specificity {v['specificity']:.3f}, balanced accuracy {v['balanced_accuracy']:.3f}.  ",
        f"Calibration: Brier score {c['brier']:.3f}, expected calibration error {c['ece']:.3f}.", "",
        f"Deployment threshold {dep['value']:.3f} (targets sensitivity {dep['target_sensitivity']:.2f}): "
        f"sensitivity {dep['sensitivity']:.3f}, specificity {dep['specificity']:.3f}. "
        "This threshold was picked on the same out-of-fold predictions, so treat these two figures as optimistic.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render(json.loads((Path(sys.argv[1]) / "metrics.json").read_text())))
