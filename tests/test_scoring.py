"""Referral rules checked against the published cut-offs, written out by hand
so the expectations do not depend on the code under test."""
import shutil
import subprocess
from pathlib import Path

import pytest

from common.scoring import decide, evaluate, load_bundle, score_instrument
from tests import make_vectors

RULES = load_bundle()["rules"]
INST = load_bundle()["instruments"]
ROOT = Path(__file__).resolve().parent.parent


def outcome(cog=None, inf=None, iq=None, phq=None, flags=(), conds=()):
    return decide({"csid_cog": cog, "csid_inf": inf, "iqcode16": iq, "phq2": phq}, list(flags), list(conds), RULES)


@pytest.mark.parametrize("cog,expected", [(0, "REFER"), (4, "REFER"), (5, "FOLLOW_UP"), (6, "FOLLOW_UP"),
                                          (7, "NO_REFERRAL"), (9, "NO_REFERRAL")])
def test_cognitive_only_bands(cog, expected):
    # Prince 2011: <=4 highly suggestive, 5-6 needs informant/reassessment, >=7 improbable
    assert outcome(cog=cog)["outcome"] == expected


@pytest.mark.parametrize("cog,inf,expected", [(9, 5, "REFER"), (6, 2, "REFER"), (4, 0, "REFER"),
                                              (5, 0, "NO_REFERRAL"), (6, 1, "NO_REFERRAL"),
                                              (9, 4, "FOLLOW_UP"), (9, 0, "NO_REFERRAL")])
def test_combined_score(cog, inf, expected):
    # combined = cognitive - informant; 4 or less is a positive screen
    r = outcome(cog=cog, inf=inf)
    assert r["outcome"] == expected and r["csid_combined"] == cog - inf


def test_informant_only():
    assert outcome(inf=1)["outcome"] == "NO_REFERRAL"
    assert outcome(inf=2)["outcome"] == "REFER"


def test_iqcode_cutoff_is_3_44():
    assert outcome(cog=9, iq=343)["outcome"] == "NO_REFERRAL"
    assert outcome(cog=9, iq=344)["outcome"] == "REFER"


def test_red_flag_always_urgent():
    r = outcome(cog=9, inf=0, flags=["stroke_signs"])
    assert r["outcome"] == "URGENT" and r["red_flags"] == ["stroke_signs"]
    assert outcome(flags=["made_up_flag"])["outcome"] == "INCOMPLETE"   # unknown ids are ignored


def test_nothing_scored_is_incomplete_not_clear():
    assert outcome(phq=0)["outcome"] == "INCOMPLETE"


def test_notes():
    assert "phq2_positive" in outcome(cog=9, phq=3)["notes"]
    assert "phq2_positive" not in outcome(cog=9, phq=2)["notes"]
    assert "test_conditions" in outcome(cog=3, conds=["hearing"])["notes"]
    assert "test_conditions" not in outcome(cog=9, conds=["hearing"])["notes"]
    assert "no_informant" in outcome(cog=9)["notes"]


def test_instrument_scoring():
    full = {"elbow": 1, "hammer": 1, "store": 1, "day": 1, "season": 1, "point": 1, "recall": 3}
    assert score_instrument(INST["csid_cog"], full)["score"] == 9
    partial = dict(full); partial.pop("day")
    assert score_instrument(INST["csid_cog"], partial)["status"] == "incomplete"
    assert score_instrument(INST["csid_cog"], {})["status"] == "not_done"
    assert score_instrument(INST["csid_cog"], {**full, "recall": True})["status"] == "incomplete"  # booleans are not answers


def test_iqcode_mean_and_missing_items():
    ids = [i["id"] for i in INST["iqcode16"]["items"]]
    assert score_instrument(INST["iqcode16"], {i: 3 for i in ids})["score"] == 300
    assert score_instrument(INST["iqcode16"], {i: 4 for i in ids[:14]})["score"] == 400
    assert score_instrument(INST["iqcode16"], {i: 4 for i in ids[:13]})["status"] == "incomplete"


def test_evaluate_full_record():
    rec = {"answers": {"csid_cog": {"elbow": 1, "hammer": 1, "store": 0, "day": 0, "season": 1, "point": 1, "recall": 0},
                       "csid_inf": {"decline": 1, "reason": 1, "misplace": 1, "yesterday": 0, "where": 0, "dressing": 0}},
           "red_flags": [], "conditions": []}
    r = evaluate(rec)
    assert r["scores"]["csid_cog"] == 4 and r["scores"]["csid_inf"] == 3
    assert r["outcome"] == "REFER" and r["reasons"] == ["csid_combined_positive"]


def test_every_threshold_documents_its_source():
    for name, t in RULES["thresholds"].items():
        assert t["basis"] in ("published", "project default"), name
        assert t["source"], name
    for key in ("reasons", "notes", "outcomes"):
        assert RULES[key]


def test_vector_file_is_current():
    assert make_vectors.OUT.read_text(encoding="utf-8") == make_vectors.dumps(make_vectors.build()), \
        "Run: python -m tests.make_vectors"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_scoring_matches_server():
    run = subprocess.run(["node", str(ROOT / "tests" / "js" / "scoring_parity.mjs")], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
