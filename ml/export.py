"""Package a finished training run for the app server.

    python -m ml.export --run runs/oasis_v1 --dest models/current

Writes one ONNX file per cross-validation fold plus ``model_meta.json``. The
server averages the fold models (a small deep ensemble); how much they disagree
is shown to the clinician as a measure of uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from common.preprocess import PreprocessConfig
from ml.model import build_model


def export_onnx(model: torch.nn.Module, shape: tuple[int, int, int], path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, *shape)
    kwargs = dict(input_names=["volume"], output_names=["logit"],
                  dynamic_axes={"volume": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older PyTorch without the dynamo argument
        torch.onnx.export(model, dummy, str(path), **kwargs)


def check_parity(model: torch.nn.Module, onnx_path: Path, shape) -> float:
    import onnxruntime as ort
    x = np.random.default_rng(0).random((2, 1, *shape), dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"volume": x})[0].reshape(-1)
    return float(np.max(np.abs(ref - got)))


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="training run folder (contains metrics.json)")
    ap.add_argument("--dest", default="models/current")
    ap.add_argument("--name", default="oasis1-small3d")
    args = ap.parse_args(argv)

    run, dest = Path(args.run), Path(args.dest)
    metrics = json.loads((run / "metrics.json").read_text())
    if metrics.get("labels_shuffled"):
        raise SystemExit("This run used shuffled labels (sanity check). It must not be deployed.")
    cfg = PreprocessConfig.from_dict(metrics["config"]["preprocess"])
    dest.mkdir(parents=True, exist_ok=True)

    folds = []
    for info in metrics["folds"]:
        ckpt = torch.load(run / f"fold{info['fold']}" / "best.pt", map_location="cpu")
        model = build_model(ckpt["arch"], ckpt["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        name = f"fold{info['fold']}.onnx"
        export_onnx(model, cfg.output_shape, dest / name)
        diff = check_parity(model, dest / name, cfg.output_shape)
        if diff > 1e-3:
            raise SystemExit(f"{name}: ONNX output differs from PyTorch by {diff:.2e}. Not exporting.")
        folds.append({"file": name, "temperature": info["temperature"],
                      "sha256": hashlib.sha256((dest / name).read_bytes()).hexdigest()})
        print(f"exported {name} (max difference vs PyTorch {diff:.1e})")

    cnn, dep = metrics["cnn"], metrics["deployment_threshold"]
    meta = {
        "name": args.name,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic": bool(metrics.get("synthetic")),
        "task": metrics["task"],
        "architecture": metrics["config"]["arch"],
        "parameters": metrics["config"]["parameters"],
        "preprocess": cfg.to_dict(),
        "folds": folds,
        "threshold": dep["value"],
        "training": {"n_subjects": metrics["n_subjects"], "n_positive": metrics["n_positive"],
                     "dataset": "synthetic volumes" if metrics.get("synthetic") else "OASIS-1 cross-sectional"},
        "evaluation": {
            "method": f"{len(folds)}-fold subject-level cross-validation, out-of-fold predictions",
            "auc": cnn["auc_oof"], "auc_ci95": cnn["auc_ci95"],
            "sensitivity": cnn["at_validation_thresholds"]["sensitivity"],
            "specificity": cnn["at_validation_thresholds"]["specificity"],
            "brier": cnn["brier"],
            "baseline_tabular_auc": metrics.get("baselines", {}).get("tabular", {}).get("auc_oof"),
            "baseline_age_auc": metrics.get("baselines", {}).get("age_only", {}).get("auc_oof"),
        },
        "limits": [
            "Research prototype. Not a medical device and not validated for clinical use.",
            "Trained on one research cohort scanned on one 1.5T scanner; performance on other scanners, "
            "sites and populations is unknown and likely lower.",
            "Separates people with and without clinically rated dementia in that cohort. It does not "
            "identify the cause of dementia and does not stage it.",
            "Accepts only atlas-registered, skull-stripped T1 volumes prepared like OASIS-1.",
        ],
    }
    (dest / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {dest / 'model_meta.json'}")
    return meta


if __name__ == "__main__":
    main()
