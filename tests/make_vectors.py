"""Regenerate instruments/test_vectors.json from the Python scoring code.

    python -m tests.make_vectors

The JavaScript port is checked against this file, and a test fails if the file
is out of date, so the browser and the server cannot drift apart unnoticed.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

from common.scoring import INSTRUMENT_DIR, decide, load_bundle, score_instrument

OUT = INSTRUMENT_DIR / "test_vectors.json"


def build() -> dict:
    bundle = load_bundle()
    inst, rules = bundle["instruments"], bundle["rules"]

    scoring_cases = []
    iq_ids = [i["id"] for i in inst["iqcode16"]["items"]]
    cog_ids = [i["id"] for i in inst["csid_cog"]["items"] if i.get("type") != "info"]
    samples = [
        ("csid_cog", {}),
        ("csid_cog", {i: 1 for i in cog_ids} | {"recall": 3}),
        ("csid_cog", {i: 0 for i in cog_ids}),
        ("csid_cog", {i: 1 for i in cog_ids[:-1]}),                     # one missing -> incomplete
        ("csid_cog", {i: 1 for i in cog_ids} | {"recall": 7}),          # out-of-range value ignored
        ("csid_cog", {i: 1 for i in cog_ids} | {"elbow": "1"}),         # wrong type ignored
        ("csid_inf", {"decline": 1, "reason": 1, "misplace": 0, "yesterday": 0, "where": 0, "dressing": 0}),
        ("phq2", {"interest": 3, "mood": 0}),
        ("phq2", {"interest": 1}),
        ("iqcode16", {i: 3 for i in iq_ids}),
        ("iqcode16", {i: 3 for i in iq_ids} | {q: 4 for q in iq_ids[:7]}),   # 55/16 = 3.4375 -> 344
        ("iqcode16", {i: 3 for i in iq_ids} | {q: 4 for q in iq_ids[:6]}),   # 54/16 = 3.375  -> 338
        ("iqcode16", {i: 4 for i in iq_ids[:14]}),                           # two missing: allowed
        ("iqcode16", {i: 4 for i in iq_ids[:13]}),                           # three missing: incomplete
        ("iqcode16", {i: 5 for i in iq_ids[:15]} | {iq_ids[15]: 4}),
    ]
    for iid, answers in samples:
        scoring_cases.append({"instrument": iid, "answers": answers, "expected": score_instrument(inst[iid], answers)})

    decide_cases = []
    grid = itertools.product([None, *range(10)], [None, *range(7)], [None, 343, 344], [None, 2, 3],
                             [[], ["stroke_signs"]], [[], ["hearing"]])
    for cog, inf, iq, phq, flags, conds in grid:
        r = decide({"csid_cog": cog, "csid_inf": inf, "iqcode16": iq, "phq2": phq}, flags, conds, rules)
        decide_cases.append([cog, inf, iq, phq, len(flags), len(conds), r["outcome"],
                             ",".join(r["reasons"]), ",".join(r["notes"]), r["csid_combined"]])
    return {"rules_version": rules["version"],
            "decide_columns": ["cog", "inf", "iq_x100", "phq", "stroke_flag", "hearing_condition",
                               "outcome", "reasons", "notes", "combined"],
            "scoring": scoring_cases, "decide": decide_cases}


def dumps(v: dict) -> str:
    head = {k: v[k] for k in ("rules_version", "decide_columns", "scoring")}
    body = ",\n".join(json.dumps(row, separators=(",", ":")) for row in v["decide"])
    return json.dumps(head, indent=1)[:-2] + ',\n "decide": [\n' + body + "\n ]\n}\n"


if __name__ == "__main__":
    OUT.write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {OUT} ({len(build()['decide'])} decision cases)")
