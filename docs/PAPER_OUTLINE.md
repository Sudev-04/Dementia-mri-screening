# Paper outline (draft for discussion with your faculty guide)

Working title: **How Much Does Deep Learning Add? A Leakage-Controlled, Reliability-Aware Evaluation of MRI Models for Dementia Triage on OASIS-1**

All numbers below are from the run of 2026-11-12 (`results.json` in the Drive folder `alz-screen-results/paper`). Do not edit a number here without rerunning.

## Claim the evidence supports
On OASIS-1 (198 adults aged 60+, CDR 0 vs CDR >= 0.5), under repeated nested cross-validation split by person, a single reproducible brain-tissue fraction plus age (AUC 0.712) is statistically indistinguishable from voxel-based, ImageNet-pretrained and fusion models (0.706 to 0.750), while a from-scratch 3D CNN does worse (0.576). Scores are highly repeatable across rescans (ICC 0.98 to 0.99), but very mild dementia is mostly missed.

## Claims the evidence does NOT support (do not write these)
- That any model here diagnoses or stages Alzheimer's disease.
- That the system has been validated in the field or on patients. The screening app has never been used on a real person.
- That results transfer to other scanners, hospitals or populations.
- "State of the art" or any accuracy above the intervals reported.

## Contributions
1. A model ladder from age alone to fusion, with paired confidence intervals against a tabular baseline and a shuffled-label control.
2. Test-retest reliability of a volumetric marker and of model scores on the OASIS repeat-scan set (rarely reported).
3. Triage-oriented evaluation: selective prediction, decision-curve net benefit, detection by severity, and a learning curve.
4. An open, tested reference implementation of a field-to-clinic pathway (system description only).

## Sections
1. **Introduction.** Dementia burden in India (Lee et al. 2023: 7.4% of adults 60+); inflated CNN results from data leakage (Wen et al. 2020; Yagis et al. 2021); the question in the title.
2. **Related work.** OASIS (Marcus et al. 2007); CNN vs classical comparisons; brief screening in low-resource settings (Prince et al. 2011).
3. **Methods.** Cohort and exclusions (one scan per person; repeat sessions held out for reliability only; age 60+). Brain-fraction definition and age reference (316 people without dementia). Models A to F. Nested CV protocol. Control. Reliability (ICC(2,1)). Triage metrics. From-scratch CNN protocol.
4. **Results.** Table I (ladder), Fig. 3 (ROC), Fig. 4 (AUC with intervals), Fig. 5 (calibration), Table II + Fig. 9 (test-retest), Table III + Fig. 6 (selective prediction), Fig. 7 (decision curve), Fig. 8 (learning curve), Fig. 10 (severity), Fig. 2 (brain fraction by age), Fig. 1 (cohort).
5. **System.** Offline screening app, referral queue, scan-check page; safety choices (no diagnosis wording, input guard, server re-scoring). State plainly that it is unvalidated.
6. **Discussion.** Why simple wins at n = 198; image models still improving with data, tabular model flat; weak sensitivity for CDR 0.5; near-zero net benefit over "refer everyone" at low thresholds.
7. **Limitations.** Single site and scanner; small n; CDR-based label, all cases probable AD; the from-scratch CNN was a separate single 5-fold run with three settings examined; pretrained features were 2D slices, not a 3D medical pretraining; no external validation; reference SD for age 60+ is rounded.
8. **Conclusion.**

## Before submission
- A faculty guide as co-author or supervisor, and their review of every claim.
- OASIS acknowledgement text and citation (see `docs/REFERENCES.md`).
- Check the venue's rule on disclosing AI assistance and follow it. This code, analysis and outline were produced with an AI assistant.
- Rerun the from-scratch CNN inside the same repeated nested protocol so Table I has one consistent method (needs a GPU session).
- Ideally one external dataset (OASIS-2 or ADNI) to test whether the conclusion holds.
- Novelty check: search for prior work reporting test-retest stability of AD classifiers on OASIS before claiming it is new.
