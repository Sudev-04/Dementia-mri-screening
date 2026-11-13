import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_t88_like(seed: int = 0) -> np.ndarray:
    """A skull-stripped-looking volume in the expected shape (not anatomy)."""
    rng = np.random.default_rng(seed)
    g = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in (176, 208, 176)), indexing="ij")
    brain = ((g[0] - 88) / 66) ** 2 + ((g[1] - 104) / 84) ** 2 + ((g[2] - 88) / 62) ** 2 <= 1
    return np.where(brain, 900 + rng.normal(0, 60, brain.shape), 0).astype(np.int16)


@pytest.fixture()
def t88_file(tmp_path):
    import nibabel as nib
    img = nib.AnalyzeImage(make_t88_like(), np.eye(4))
    img.header.set_zooms((1.0, 1.0, 1.0))
    path = tmp_path / "scan.img"
    nib.save(img, str(path))
    return path
