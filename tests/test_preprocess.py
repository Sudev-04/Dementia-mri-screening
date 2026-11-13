import numpy as np
import pytest

from common.preprocess import InputRejected, PreprocessConfig, check_input, load_and_preprocess, load_volume, preprocess
from tests.conftest import make_t88_like

CFG = PreprocessConfig()


def test_output_shape_and_range(t88_file):
    x, info = load_and_preprocess(t88_file)
    assert x.shape == (80, 96, 80) and x.dtype == np.float32
    assert x.min() == 0.0 and 0.95 <= x.max() <= 1.0
    assert 0.45 < info["background_fraction"] < 0.92


def test_deterministic():
    v = make_t88_like(1)
    assert np.array_equal(preprocess(v), preprocess(v.copy()))


def test_intensity_scale_invariant():
    v = make_t88_like(2).astype(np.float32)
    assert np.allclose(preprocess(v), preprocess(v * 3.7), atol=1e-5)


def test_rejects_wrong_shape():
    with pytest.raises(InputRejected, match="shape"):
        check_input(np.ones((256, 256, 160), dtype=np.float32), (1, 1, 1), CFG)


def test_rejects_wrong_voxel_size():
    with pytest.raises(InputRejected, match="voxel size"):
        check_input(make_t88_like(), (1.0, 1.0, 1.25), CFG)


def test_rejects_scan_that_is_not_skull_stripped():
    raw_like = np.full((176, 208, 176), 500, dtype=np.float32)   # signal everywhere, like a raw head scan
    with pytest.raises(InputRejected, match="background"):
        check_input(raw_like, (1, 1, 1), CFG)


def test_hdr_path_resolves_to_img(t88_file):
    vol, zooms = load_volume(t88_file.with_suffix(".hdr"))
    assert vol.shape == (176, 208, 176) and zooms == (1.0, 1.0, 1.0)


def test_config_round_trip():
    assert PreprocessConfig.from_dict(CFG.to_dict()) == CFG


def test_brain_fraction_and_age_reference():
    from common.measure import assess, brain_fraction, load_reference
    ref = load_reference()
    v = make_t88_like(4).astype(np.float32)
    full = brain_fraction(v, ref["threshold"])
    assert 0.99 <= full <= 1.0                                  # solid "brain": all tissue
    holed = v.copy(); holed[70:106, 80:128, 70:106] *= 0.1       # dark fluid-like block inside the mask
    assert brain_fraction(holed, ref["threshold"]) < full - 0.02
    assert brain_fraction(v * 5, ref["threshold"]) == full       # independent of intensity scale
    r = assess(holed, 75)
    assert r["band"] in ("usual", "lower", "well_below") and 0 <= r["percentile"] <= 100
    assert abs(r["expected_for_age"] - (-2.15264e-05 * 75 * 75 + 0.000624121 * 75 + 0.890831)) < 1e-4
    assert assess(holed, 75)["z"] > assess(holed, 30)["z"]       # the same brain is less unusual at an older age
    assert assess(v, 120)["age_in_reference_range"] is False


def test_browser_scan_measurement_matches_python(tmp_path):
    """The server-less demo measures scans in JavaScript; it must agree with common/measure.py."""
    import json, shutil, subprocess
    import nibabel as nib
    from pathlib import Path
    from common.measure import REFERENCE_PATH, assess
    from common.preprocess import check_input, load_volume
    if shutil.which("node") is None:
        pytest.skip("Node.js not installed")
    v = make_t88_like(7).astype(np.float32)
    v[60:116, 70:138, 60:116] *= np.linspace(0.05, 0.6, 56, dtype=np.float32)[:, None, None]   # graded dark region
    img = nib.AnalyzeImage(v.astype(np.int16), np.eye(4)); img.header.set_zooms((1.0, 1.0, 1.0))
    nib.save(img, str(tmp_path / "s.img"))
    vol, zooms = load_volume(tmp_path / "s.img")
    want = {**assess(vol, 71), **check_input(vol, zooms)}
    (tmp_path / "want.json").write_text(json.dumps(want))
    root = Path(__file__).resolve().parent.parent
    run = subprocess.run(["node", str(root / "tests" / "js" / "scan_parity.mjs"), str(tmp_path / "s.hdr"), str(tmp_path / "s.img"),
                          str(REFERENCE_PATH), "71", str(tmp_path / "want.json")], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
