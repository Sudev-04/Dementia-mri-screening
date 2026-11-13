# References and licence notes

Bibliographic details below were retrieved from PubMed.

## Screening instruments

| Instrument | Source | Terms, as found |
|---|---|---|
| Brief CSI-D (cognitive 0-9, informant 0-6) | Prince M, Acosta D, Ferri CP, et al. A brief dementia screener suitable for use by non-specialists in resource poor settings: the cross-cultural derivation and validation of the brief Community Screening Instrument for Dementia. *Int J Geriatr Psychiatry* 2011;26(9):899-907. [doi:10.1002/gps.2622](https://doi.org/10.1002/gps.2622), PMID 21845592, PMC3427892 | Article is open access under Creative Commons Attribution 2.5; commercial exploitation not permitted. The parent CSI-D was developed by K. S. Hall. **Confirm permission with the authors before any field use.** |
| Short IQCODE (16 items) | Jorm AF. A short form of the Informant Questionnaire on Cognitive Decline in the Elderly (IQCODE): development and cross-validation. *Psychol Med* 1994;24(1):145-153. [doi:10.1017/s003329170002691x](https://doi.org/10.1017/s003329170002691x), PMID 8208879 | The published form states "There is no copyright on the Short IQCODE"; the author asks to be told about research use. |
| PHQ-2 | Kroenke K, Spitzer RL, Williams JBW. The Patient Health Questionnaire-2: validity of a two-item depression screener. *Med Care* 2003;41(11):1284-1292. [doi:10.1097/01.MLR.0000093487.78664.3C](https://doi.org/10.1097/01.MLR.0000093487.78664.3C), PMID 14583691 | Generally distributed for use without permission. Not re-verified for this project: check phqscreeners.com. |

### Cut-offs used, and where they come from

| Rule | Value | Evidence |
|---|---|---|
| CSI-D combined (cognitive minus informant) | 4 or less is positive | Optimal in all four pilot regions; in the India sample sensitivity 97.3%, specificity 90.5% (Prince 2011, Table 2) |
| CSI-D cognitive alone | 4 or less positive; 5-6 indeterminate; 7+ unlikely | "Conservative strategy" in the Discussion of Prince 2011; India sample at <6: sensitivity 72.4%, specificity 93.5% |
| CSI-D informant alone | 2 or more is positive | Optimal in every region except Nigeria; India sample sensitivity 87.9%, specificity 99.7% |
| Short IQCODE mean | 3.44 or more is positive | Recommended as a screening compromise in Jorm's review of the IQCODE (copy at nceph.anu.edu.au/files/IQCODE_review.pdf); community studies range about 3.3 to 3.6 |
| PHQ-2 | 3 or more is positive | Sensitivity 83%, specificity 92% for major depression (Kroenke 2003) |

**Read these accuracy figures with care.** They come from comparisons of people already
diagnosed with dementia against selected controls, which flatters any test. The authors
themselves note that the brief CSI-D items were administered inside a longer interview by
research staff, and that validity of the stand-alone brief form in routine primary care "still
needs to be established". Sensitivity for mild dementia was lower than for moderate (90.3% vs
95.0% on the combined score), and specificity was lower in people with little education (89.9%
vs 95.0%).

Why no clock drawing, MMSE, MoCA or Mini-Cog: Prince et al. point out that clock-drawing tasks
are "not generally feasible for those with less than 5 years education", and the other three
are education-biased, copyrighted or licence-restricted.

## MRI data and methods

- Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL. Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. *J Cogn Neurosci* 2007;19(9):1498-1507. [doi:10.1162/jocn.2007.19.9.1498](https://doi.org/10.1162/jocn.2007.19.9.1498), PMID 17714011. 416 people aged 18 to 96; 100 of those over 60 have very mild to moderate Alzheimer's disease; 20 people were rescanned within 90 days.
- Wen J, Thibeau-Sutre E, Diaz-Melo M, et al. Convolutional neural networks for classification of Alzheimer's disease: overview and reproducible evaluation. *Med Image Anal* 2020;63:101694. [doi:10.1016/j.media.2020.101694](https://doi.org/10.1016/j.media.2020.101694), PMID 32417716, [arXiv:1904.07773](https://www.alphaxiv.org/abs/1904.07773). About half of 32 surveyed studies may have had data leakage; CNNs did not beat a linear SVM; models trained on ADNI lost 0.13 to 0.19 balanced accuracy on OASIS.
- Yagis E, Atnafu SW, García Seco de Herrera A, et al. Effect of data leakage in brain MRI classification using 2D convolutional neural networks. *Sci Rep* 2021;11:22544. [doi:10.1038/s41598-021-01681-w](https://doi.org/10.1038/s41598-021-01681-w), PMID 34799630. Slice-level splitting inflated test accuracy by 30% on OASIS; randomly labelled OASIS data reached about 96% "accuracy" with a slice-level split.

OASIS acknowledgement required in publications: "Data were provided by OASIS-1: Cross-Sectional:
Principal Investigators: D. Marcus, R. Buckner, J. Csernansky, J. Morris; P50 AG05681, P01 AG03991,
P01 AG026276, R01 AG021910, P20 MH071616, U24 RR021382." Terms: <https://sites.wustl.edu/oasisbrains/home/access/>.

## Context

- Lee J, Meijer E, Langa KM, et al. Prevalence of dementia in India: national and state estimates from a nationwide study. *Alzheimers Dement* 2023;19(7):2898-2912. [doi:10.1002/alz.12928](https://doi.org/10.1002/alz.12928), PMID 36637034. Estimated prevalence 7.4% among adults aged 60+, about 8.8 million people, higher in women and in rural areas.
