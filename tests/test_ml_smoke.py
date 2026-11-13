"""End-to-end pipeline test on generated volumes: cohort -> training -> export -> ONNX.
Slow (about a minute on a laptop CPU). Skipped when PyTorch is not installed."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    from ml.synthetic import generate
    return generate(tmp_path_factory.mktemp("syn"), subjects=12, seed=1)


def test_cohort_drops_repeat_sessions_and_unrated_young_adults(synthetic):
    from ml.oasis import build_cohort, read_demographics
    root, csv = synthetic
    demo = read_demographics(csv)
    assert "OAS1_0001_MR2" in set(demo["session_id"]) and "OAS1_9999_MR1" in set(demo["session_id"])
    cohort = build_cohort(csv, root)
    assert len(cohort) == 12 and cohort["subject_id"].is_unique
    assert not cohort["session_id"].str.endswith("MR2").any()
    assert cohort["age"].min() >= 60 and set(cohort["label"]) == {0, 1}


def test_train_export_and_serve(synthetic, tmp_path):
    from app.mri_service import MriModel
    from ml import export, train
    root, csv = synthetic
    run = tmp_path / "run"
    result = train.main(["--data-root", str(root), "--csv", str(csv), "--out", str(run), "--arch", "tiny",
                         "--folds", "2", "--epochs", "1", "--batch-size", "4", "--workers", "0",
                         "--bootstrap", "50", "--synthetic"])
    assert result["synthetic"] is True and 0 <= result["cnn"]["auc_oof"] <= 1
    assert set(result["baselines"]) == {"age_only", "tabular"}

    splits = json.loads((run / "splits.json").read_text())
    tested = [s for f in splits for s in f["test"]]
    assert len(tested) == len(set(tested)) == 12                    # everyone tested exactly once
    for f in splits:
        assert not set(f["train"]) & set(f["test"]) and not set(f["val"]) & set(f["test"])
        assert not set(f["train"]) & set(f["val"])

    meta = export.main(["--run", str(run), "--dest", str(tmp_path / "model")])
    assert meta["synthetic"] is True and len(meta["folds"]) == 2

    model = MriModel(tmp_path / "model")
    scan = next(root.rglob("*_t88_masked_gfc.img"))
    x, _ = model.prepare(scan)
    out = model.predict(x)
    assert 0 <= out["probability"] <= 1 and out["synthetic"] is True
    heat = model.occlusion_map(x, patch=40)
    assert heat.shape == x.shape


def test_shuffled_label_runs_cannot_be_exported(tmp_path):
    from ml import export
    (tmp_path / "metrics.json").write_text(json.dumps({"labels_shuffled": True}))
    with pytest.raises(SystemExit):
        export.main(["--run", str(tmp_path), "--dest", str(tmp_path / "m")])
