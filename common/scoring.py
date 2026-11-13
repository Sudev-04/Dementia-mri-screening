"""Scoring and referral logic for the field screening instruments.

This module is the server-side source of truth. The browser has a line-for-line
port in ``app/static/js/scoring.js`` so results can be shown offline; the two are
kept identical by shared test vectors (``instruments/test_vectors.json``).

Nothing here is machine learning. Every threshold lives in ``instruments/rules.json``
with its published source, so a clinician can read and change the rules without
touching code.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

INSTRUMENT_DIR = Path(__file__).resolve().parent.parent / "instruments"
INSTRUMENT_IDS = ("csid_cog", "csid_inf", "iqcode16", "phq2")
OUTCOME_RANK = {"NO_REFERRAL": 0, "INCOMPLETE": 1, "FOLLOW_UP": 2, "REFER": 3, "URGENT": 4}


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    """Instruments + rules as one JSON-serialisable bundle (also served to the app)."""
    instruments = {
        iid: json.loads((INSTRUMENT_DIR / f"{iid}.json").read_text(encoding="utf-8"))
        for iid in INSTRUMENT_IDS
    }
    rules = json.loads((INSTRUMENT_DIR / "rules.json").read_text(encoding="utf-8"))
    return {"instruments": instruments, "rules": rules}


def _allowed_values(instrument: dict, item: dict) -> set[int]:
    options = item.get("options") or instrument.get("options") or []
    return {int(o["value"]) for o in options}


def score_instrument(instrument: dict, answers: dict | None) -> dict:
    """Score one instrument.

    Returns ``{"status", "score", "answered", "total"}`` where status is
    ``not_done`` (nothing answered), ``incomplete`` (too many items missing) or
    ``complete``. ``score`` is an integer: the sum for ``sum`` instruments, and
    the mean times 100 (rounded half up) for ``mean`` instruments, which keeps
    Python and JavaScript arithmetic exactly equal.
    """
    answers = answers or {}
    scored = [it for it in instrument["items"] if it.get("type") != "info"]
    values: list[int] = []
    for item in scored:
        raw = answers.get(item["id"])
        # Only whole numbers that are one of the item's options count as answered.
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        if raw in _allowed_values(instrument, item):
            values.append(raw)

    total, answered = len(scored), len(values)
    result = {"status": "not_done", "score": None, "answered": answered, "total": total}
    if answered == 0:
        return result

    scoring = instrument["scoring"]
    missing = total - answered
    if missing > int(scoring.get("max_missing", 0)):
        result["status"] = "incomplete"
        return result

    s = sum(values)
    if scoring["method"] == "mean":
        result["score"] = (200 * s + answered) // (2 * answered)
    else:
        result["score"] = s
    result["status"] = "complete"
    return result


def decide(scores: dict, red_flags: list[str] | None, conditions: list[str] | None,
           rules: dict) -> dict:
    """Turn instrument scores into one outcome with reasons and notes.

    ``scores`` maps instrument id to an integer score or ``None``.
    """
    t = {k: v["value"] for k, v in rules["thresholds"].items()}
    known_flags = [f["id"] for f in rules["red_flags"]]
    known_conditions = [c["id"] for c in rules["conditions"]]
    flags = [f for f in known_flags if f in (red_flags or [])]
    conds = [c for c in known_conditions if c in (conditions or [])]

    cog = scores.get("csid_cog")
    inf = scores.get("csid_inf")
    iq = scores.get("iqcode16")
    phq = scores.get("phq2")

    outcome = "NO_REFERRAL"
    reasons: list[str] = []
    notes: list[str] = []
    combined = None
    cog_drove_result = False

    def raise_to(level: str, reason: str) -> None:
        nonlocal outcome
        if OUTCOME_RANK[level] > OUTCOME_RANK[outcome]:
            outcome = level
        reasons.append(reason)

    if flags:
        raise_to("URGENT", "red_flag")

    if cog is not None and inf is not None:
        combined = cog - inf
        if combined <= t["csid_combined_max"]:
            raise_to("REFER", "csid_combined_positive")
            cog_drove_result = True
        elif inf >= t["csid_inf_min"]:
            raise_to("FOLLOW_UP", "informant_concern")
    elif cog is not None:
        if cog <= t["csid_cog_positive_max"]:
            raise_to("REFER", "csid_cog_positive")
            cog_drove_result = True
        elif cog <= t["csid_cog_indeterminate_max"]:
            raise_to("FOLLOW_UP", "csid_cog_indeterminate")
            cog_drove_result = True
    elif inf is not None:
        if inf >= t["csid_inf_min"]:
            raise_to("REFER", "csid_inf_positive")

    if iq is not None and iq >= t["iqcode16_min_x100"]:
        raise_to("REFER", "iqcode_positive")

    if cog is None and inf is None and iq is None:
        raise_to("INCOMPLETE", "nothing_scored")

    if phq is not None and phq >= t["phq2_min"]:
        notes.append("phq2_positive")
    if conds and cog_drove_result:
        notes.append("test_conditions")
    if cog is not None and inf is None and iq is None:
        notes.append("no_informant")

    return {
        "outcome": outcome,
        "reasons": reasons,
        "notes": notes,
        "red_flags": flags,
        "conditions": conds,
        "csid_combined": combined,
        "followup_months": t["followup_months"] if outcome == "FOLLOW_UP" else None,
    }


def evaluate(record: dict, bundle: dict | None = None) -> dict:
    """Score a whole screening record (answers, red flags, conditions)."""
    bundle = bundle or load_bundle()
    answers = record.get("answers") or {}
    detail = {
        iid: score_instrument(inst, answers.get(iid))
        for iid, inst in bundle["instruments"].items()
    }
    scores = {iid: d["score"] for iid, d in detail.items()}
    decision = decide(scores, record.get("red_flags"), record.get("conditions"), bundle["rules"])
    return {
        "rules_version": bundle["rules"]["version"],
        "scores": scores,
        "detail": detail,
        **decision,
    }
