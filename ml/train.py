"""Train and evaluate the MRI model with subject-level cross-validation.

    python -m ml.train --data-root /path/to/oasis1 --csv oasis_cross-sectional.csv --out runs/oasis_v1

How the evaluation is kept honest:
  * the cohort has one scan per person (see ml/oasis.py), so folds are disjoint by person;
  * each outer fold is split again into train/validation; early stopping, the
    probability calibration and the decision threshold use validation only;
  * the outer test fold is scored once, after everything is frozen;
  * headline numbers are pooled out-of-fold predictions with bootstrap intervals;
  * --shuffle-labels reruns the whole pipeline on permuted labels, which must
    come out near AUC 0.5. If it does not, something is leaking.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common.preprocess import PreprocessConfig, load_and_preprocess
from ml import metrics as M
from ml.baseline import run_baselines
from ml.model import build_model, count_parameters
from ml.oasis import build_cohort, describe


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_volumes(df: pd.DataFrame, cache_dir: Path, cfg: PreprocessConfig) -> list[Path]:
    """Preprocess every scan once and keep it as a small .npy file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in enumerate(df.itertuples()):
        out = cache_dir / f"{row.subject_id}_{cfg.version}_p{cfg.pool}.npy"
        if not out.exists():
            x, _ = load_and_preprocess(row.image, cfg, guard=True)
            np.save(out, x.astype(np.float16))
        paths.append(out)
        if (i + 1) % 50 == 0:
            print(f"[cache] {i + 1}/{len(df)} volumes ready")
    return paths


def make_augment(seed: int):
    from monai.transforms import Compose, RandAffine, RandFlip, RandGaussianNoise, RandScaleIntensity
    aug = Compose([
        RandFlip(prob=0.5, spatial_axis=0),                       # left-right
        RandAffine(prob=0.7, rotate_range=(0.05, 0.05, 0.05), translate_range=(3, 3, 3),
                   scale_range=(0.05, 0.05, 0.05), padding_mode="zeros", mode="bilinear"),
        RandScaleIntensity(factors=0.1, prob=0.5),
        RandGaussianNoise(prob=0.2, std=0.02),
    ])
    aug.set_random_state(seed)
    return aug


class VolumeDataset(Dataset):
    def __init__(self, paths, labels, augment=None):
        self.paths, self.labels, self.augment = list(paths), np.asarray(labels, dtype=np.float32), augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = torch.from_numpy(np.load(self.paths[i]).astype(np.float32))[None]
        if self.augment is not None:
            x = torch.as_tensor(self.augment(x)).float().clamp_(0, 1.5)
        return x, torch.tensor(self.labels[i])


@torch.no_grad()
def predict_logits(model, paths, device, batch_size=8) -> np.ndarray:
    model.eval()
    loader = DataLoader(VolumeDataset(paths, np.zeros(len(paths))), batch_size=batch_size)
    out = [model(x.to(device)).float().cpu().numpy() for x, _ in loader]
    return np.concatenate(out) if out else np.zeros(0)


def train_one_fold(args, fold, tr_paths, y_tr, va_paths, y_va, device, out_dir: Path) -> dict:
    set_seed(args.seed + fold)
    model = build_model(args.arch, args.dropout).to(device)
    n_pos = float(y_tr.sum()); n_neg = float(len(y_tr) - n_pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_neg / max(n_pos, 1.0), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(VolumeDataset(tr_paths, y_tr, make_augment(args.seed + fold)),
                        batch_size=args.batch_size, shuffle=True, drop_last=len(tr_paths) > args.batch_size,
                        num_workers=args.workers)
    steps = max(1, len(loader)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc, best_epoch, best_state, history = -1.0, -1, None, []
    for epoch in range(args.epochs):
        model.train(); running, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            # The loss is computed in float32: half-precision losses are a classic
            # source of silent NaNs.
            loss = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at fold {fold}, epoch {epoch}. Aborting.")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            running += loss.item() * len(y); seen += len(y)

        va_logits = predict_logits(model, va_paths, device, args.batch_size)
        va_auc = M.safe_auc(y_va, va_logits)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), "val_auc": va_auc})
        if np.isfinite(va_auc) and va_auc > best_auc + 1e-4:
            best_auc, best_epoch = va_auc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  fold {fold} epoch {epoch:3d}  loss {running / max(seen, 1):.4f}  val AUC {va_auc:.3f}")
        if epoch - best_epoch >= args.patience:
            print(f"  fold {fold}: no validation gain for {args.patience} epochs, stopping at {epoch}")
            break

    if best_state is None:  # validation fold had one class only (tiny debug runs)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    va_logits = predict_logits(model, va_paths, device, args.batch_size)
    temperature = M.fit_temperature(va_logits, y_va)
    threshold = M.youden_threshold(y_va, M.sigmoid(va_logits / temperature))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "arch": args.arch, "dropout": args.dropout}, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=1))
    return {"model": model, "temperature": temperature, "threshold": threshold,
            "best_epoch": best_epoch, "val_auc": best_auc}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="folder that contains the extracted OASIS-1 discs")
    ap.add_argument("--csv", required=True, help="oasis_cross-sectional.csv or .xlsx")
    ap.add_argument("--out", required=True, help="output folder for this run")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--arch", default="small", choices=["small", "tiny", "densenet121"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--pool", type=int, default=2, help="downsampling factor (2 = 2 mm voxels)")
    ap.add_argument("--min-age", type=int, default=60)
    ap.add_argument("--target-sensitivity", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="sanity check: permute labels; the result must be near AUC 0.5")
    ap.add_argument("--synthetic", action="store_true",
                    help="mark this run as trained on synthetic data (never for real use)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    t0 = time.time()
    set_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = PreprocessConfig(pool=args.pool)

    cache_dir = Path(args.cache_dir or out / "cache")
    suffix = f"_{cfg.version}_p{cfg.pool}.npy"
    cached = {p.name[: -len(suffix)] for p in cache_dir.glob(f"*{suffix}")} if cache_dir.exists() else set()
    df = build_cohort(args.csv, args.data_root, args.min_age, cached_subjects=cached)
    print(describe(df)); print(f"Device: {device}")
    paths = np.array(cache_volumes(df, cache_dir, cfg))
    y = df["label"].to_numpy()
    if args.shuffle_labels:
        y = np.random.default_rng(args.seed).permutation(y)
        print("[sanity] labels shuffled: expect AUC close to 0.5")
    subjects = df["subject_id"].to_numpy()

    oof = np.full(len(df), np.nan); oof_fold = np.full(len(df), -1); oof_pred = np.zeros(len(df), dtype=int)
    fold_info, splits = [], []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (dev_idx, te_idx) in enumerate(skf.split(paths, y)):
        tr_idx, va_idx = train_test_split(dev_idx, test_size=0.2, stratify=y[dev_idx],
                                          random_state=args.seed + fold)
        sets = [set(subjects[i]) for i in (tr_idx, va_idx, te_idx)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "subject leak across splits"
        splits.append({"fold": fold, "train": sorted(sets[0]), "val": sorted(sets[1]), "test": sorted(sets[2])})

        print(f"\nFold {fold}: train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}")
        r = train_one_fold(args, fold, paths[tr_idx], y[tr_idx], paths[va_idx], y[va_idx],
                           device, out / f"fold{fold}")
        te_prob = M.sigmoid(predict_logits(r["model"], paths[te_idx], device, args.batch_size) / r["temperature"])
        oof[te_idx], oof_fold[te_idx] = te_prob, fold
        oof_pred[te_idx] = (te_prob >= r["threshold"]).astype(int)
        fold_info.append({"fold": fold, "best_epoch": r["best_epoch"], "val_auc": r["val_auc"],
                          "temperature": r["temperature"], "threshold": r["threshold"],
                          "test_auc": M.safe_auc(y[te_idx], te_prob), "n_test": int(len(te_idx))})
        print(f"  fold {fold} test AUC {fold_info[-1]['test_auc']:.3f} "
              f"(T={r['temperature']:.2f}, threshold={r['threshold']:.2f})")
        del r["model"]

    # Headline numbers: each person predicted once, by a model that never saw them.
    lo, hi = M.bootstrap_ci(y, oof, n_boot=args.bootstrap, seed=args.seed)
    tp = int(((oof_pred == 1) & (y == 1)).sum()); fn = int(((oof_pred == 0) & (y == 1)).sum())
    tn = int(((oof_pred == 0) & (y == 0)).sum()); fp = int(((oof_pred == 1) & (y == 0)).sum())
    fold_aucs = [f["test_auc"] for f in fold_info]
    deploy_thr = M.threshold_for_sensitivity(y, oof, args.target_sensitivity)

    baselines = {} if args.shuffle_labels else run_baselines(df, y, skf, args.seed, oof, args.bootstrap)

    result = {
        "task": ("SYNTHETIC TEST DATA, not real scans" if args.synthetic else
                 "dementia (CDR >= 0.5) vs no dementia (CDR 0), OASIS-1, age >= %d" % args.min_age),
        "synthetic": bool(args.synthetic), "labels_shuffled": bool(args.shuffle_labels),
        "n_subjects": int(len(df)), "n_positive": int(y.sum()),
        "cnn": {
            "auc_oof": M.safe_auc(y, oof), "auc_ci95": [lo, hi],
            "auc_fold_mean": float(np.nanmean(fold_aucs)), "auc_fold_sd": float(np.nanstd(fold_aucs)),
            "at_validation_thresholds": {
                "note": "each test fold scored with the threshold chosen on that fold's validation set",
                "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                "balanced_accuracy": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2,
                "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            },
            "brier": M.brier(y, oof), "ece": M.expected_calibration_error(y, oof),
        },
        "deployment_threshold": {
            "value": deploy_thr, "target_sensitivity": args.target_sensitivity,
            "note": "chosen on the pooled out-of-fold predictions, so the figures below are optimistic, "
                    "not an independent estimate",
            **M.confusion_at(y, oof, deploy_thr),
        },
        "baselines": baselines, "folds": fold_info,
        "config": {**vars(args), "preprocess": cfg.to_dict(), "device": str(device),
                   "parameters": count_parameters(build_model(args.arch, args.dropout)),
                   "torch": torch.__version__},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    pd.DataFrame({"subject_id": subjects, "label": y, "prob": oof, "fold": oof_fold,
                  "pred_at_val_threshold": oof_pred}).to_csv(out / "oof_predictions.csv", index=False)
    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float))

    c = result["cnn"]
    print(f"\nOut-of-fold AUC {c['auc_oof']:.3f} (95% CI {lo:.3f} to {hi:.3f}); "
          f"per-fold {c['auc_fold_mean']:.3f} +/- {c['auc_fold_sd']:.3f}")
    print(f"Sensitivity {c['at_validation_thresholds']['sensitivity']:.3f}, "
          f"specificity {c['at_validation_thresholds']['specificity']:.3f} at validation-chosen thresholds")
    for name, b in baselines.items():
        print(f"Baseline {name}: AUC {b['auc_oof']:.3f}; CNN minus baseline "
              f"{b['cnn_minus_baseline_auc']:+.3f} (95% CI {b['difference_ci95'][0]:+.3f} to {b['difference_ci95'][1]:+.3f})")
    print(f"Saved to {out}")
    return result


if __name__ == "__main__":
    main()
