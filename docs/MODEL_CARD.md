# Model card: OASIS-1 3D CNN ensemble

**Status: not yet trained on real data.** The only model in this repository (`models/demo`) was
trained on synthetic noise to test the software. Fill in the Results section from
`python -m ml.report runs/<your run>` after training.

## Intended use

Research and teaching. A second-opinion score for a clinician who has already decided an MRI is
warranted and who is looking at the images themselves. It answers one narrow question: does this
scan look more like the scans of people with clinically rated dementia, or more like those of
people without, in OASIS-1?

## Not intended for

- Diagnosis, or ruling dementia out.
- Telling Alzheimer's disease from other causes (vascular, Lewy body, frontotemporal, depression,
  normal-pressure hydrocephalus, tumours). Every person with dementia in OASIS-1 was diagnosed with
  probable Alzheimer's, so the model has never seen an alternative.
- Staging. OASIS-1 has, as far as its fact sheet describes, about 70 people at CDR 0.5, 28 at
  CDR 1 and 2 at CDR 2 (the training log prints the exact counts). A couple of examples of the
  most severe class cannot support a staging model, so this one is binary by design.
- Screening people without symptoms, or people under 60.
- Any scan that has not been skull-stripped and registered to the T88 atlas at 1 mm.

## Data

OASIS-1 cross-sectional (Marcus et al. 2007): right-handed adults scanned on one 1.5T Siemens
Vision scanner at Washington University. Used here: one session per person, age 60 or over, with
a recorded Clinical Dementia Rating. That is roughly 198 people (about 98 at CDR 0, 100 at CDR
0.5 or more); the training log prints the exact cohort. Input images are the dataset's own
gain-field-corrected, atlas-registered, brain-masked volumes.

Label: CDR 0 versus CDR 0.5 or more. The CDR is assigned by clinicians from interviews, without
reference to the MRI, which is what makes it usable as a label.

## Model

`SmallCNN3D`: four stages of two 3x3x3 convolutions with instance normalisation, global average
pooling, dropout, one logit. About 0.9 million weights. Input 80 x 96 x 80 at 2 mm, intensities
scaled to [0, 1] by the 99.5th percentile of brain voxels. Five models, one per cross-validation
fold, each with its own temperature for calibration; the app averages their probabilities.

## Evaluation protocol

Five-fold cross-validation stratified by label and split by person. Inside each training fold,
20% is held out for early stopping, temperature scaling and the per-fold decision threshold. The
test fold is scored once. Reported: pooled out-of-fold AUC with a stratified bootstrap 95%
interval, per-fold AUC, sensitivity and specificity at validation-chosen thresholds, Brier score,
expected calibration error, and the paired difference in AUC against two logistic-regression
baselines (age only; age, sex, education, eTIV, nWBV).

Two cautions about reading the numbers:

- Pooled out-of-fold AUC also penalises folds whose scores sit on different scales, so it can be
  lower than the mean per-fold AUC. Both are reported.
- The deployment threshold is chosen on the pooled out-of-fold predictions to target 85%
  sensitivity. Its sensitivity and specificity are therefore optimistic, not an independent estimate.

## Results

_To be filled in from a real run. Do not copy numbers from the synthetic demo._

## Limits and risks

- **Small sample.** About 200 people gives wide intervals; about 40 people per test fold.
- **One site, one scanner, one protocol, 1990s-2000s hardware.** Wen et al. (2020) saw balanced
  accuracy fall by 0.13 to 0.19 when models moved between ADNI and OASIS. Expect a similar or
  larger drop on scans from Indian hospitals, where scanners, protocols and the population differ.
- **Population.** Highly educated US volunteers. Brain-volume norms and dementia causes differ
  across populations.
- **Confounding.** Atrophy also comes with age and with other diseases. The age-only baseline
  shows how much of the signal age alone explains; the model is not adjusted for it.
- **Explanations are coarse.** The region map blanks 4 cm blocks. It shows which broad region
  moved the score, not which structure, and it has not been checked against radiologists.
- **Automation bias.** A confident-looking number can anchor a clinician. The page shows member
  disagreement and never uses the word "diagnosis" for this reason.

## What would have to be true before clinical use

External validation on local scans with local clinical diagnoses; a validated preprocessing
pipeline for raw scans; a prospective study of whether the score changes clinicians' decisions
for the better; regulatory review as software as a medical device; and ethics approval.
