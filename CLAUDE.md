# Notes for AI coding assistants working in this repository

Research prototype: offline field screening for dementia plus an MRI model for the clinic.
It handles health data and makes health-adjacent claims, so correctness and honesty about
limits matter more than features.

## Commands

- Tests: `python -m pytest -q` (must stay green; the training test is skipped without PyTorch)
- Browser/JS scoring parity only: `node tests/js/scoring_parity.mjs`
- After changing `common/scoring.py`, any file in `instruments/`, or `app/static/js/scoring.js`:
  `python -m tests.make_vectors`, then rerun the tests
- App: `python run_app.py --demo-model`
- Browser test: see the header of `tests/e2e/flow.mjs`

## Rules that must not be broken

1. `common/scoring.py` and `app/static/js/scoring.js` are the same algorithm in two languages.
   Change both together and regenerate the vectors.
2. Every threshold lives in `instruments/rules.json` with `basis` ("published" or
   "project default") and `source`. Do not hard-code a cut-off, and do not label a value
   "published" without a citation that actually contains it.
3. `common/preprocess.py` is shared by training and serving. If its behaviour changes, bump
   `PREPROCESS_VERSION`; old models must then be retrained.
4. Splits are by person. Never add slice-level, patch-level or session-level splitting, and never
   tune anything on a test fold.
5. Never write real-data metrics into docs unless they come from a `metrics.json` produced by
   `ml.train` on real data. Synthetic results must be labelled synthetic.
6. No user-facing text may say or imply "diagnosis". Screening results and model scores are not diagnoses.
7. Do not add translations of cognitive test items. UI text may be translated; test items need
   validated versions.
8. Bump `VERSION` in `app/static/sw.js` whenever a file in its `SHELL` list changes, or phones
   keep the old copy.
9. Do not commit OASIS data, cached volumes, `instance/` or `models/current/`.

## Style

Python 3.10+, type hints, small modules. No inline styles or scripts in the web app (the
Content-Security-Policy forbids them). Escape everything rendered from data with `esc()`.
