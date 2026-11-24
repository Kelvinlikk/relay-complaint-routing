# More training data; an unchanged routing rule

**The project did not start in July.** May 2018–December 2019 supplies training data. January–April 2020 tuned the original models (3,065); May–August 2020 set historical thresholds (3,000); September 2020–April 2021 supplied the original test (6,917). The routing team used 14,642 June 2026 cases for policy analysis. July 2026 (8,830) was the later holdout for the original June routing rule. The August file has only one eligible narrative, so it cannot establish forward performance. Counts are analytic cohorts, not population prevalence. The source is the [CFPB archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/).

**Expansion.** 30,229 screened complaints joined 23,779 original training cases for a 54,008-case text-only challenger. We also added 11,847 disjoint test narratives across eight historical months, and 9,741 May complaints not in the original 5,254-case retrospective. Across these three roles the extension adds 51,817 distinct complaints, not 51,817 new independent July observations. We held June thresholds, model hyperparameters and the original blend fixed. Only the challenger's TF-IDF vocabulary and logistic regression coefficients were refit. It is not the model used for the 928 July handoffs.

| Same cases for both models | N | Original text macro-F1 | Larger text macro-F1 | Difference [paired 95% interval] |
|---|---:|---:|---:|---:|
| original historical test | 6,917 | 0.754 | 0.772 | +0.018 [+0.010, +0.026] |
| additional historical test | 11,847 | 0.761 | 0.773 | +0.012 [+0.006, +0.018] |
| previously opened July | 8,830 | 0.658 | 0.681 | +0.023 [+0.018, +0.028] |

Macro-F1 weights the seven product labels equally. Intervals resample receipt days with replacement in pairs; correlations spanning days are not captured. The additional historical test was sampled equally by calendar month before duplicate removal. July is a descriptive, already-opened comparison for the new challenger. Its original role as a forward test **for the frozen June policy** is unchanged.

**Does the original routing rule travel?** Original May sample: 479/5,254 flagged, 13/479 proxy errors. New May cases alone: 944/9,741 flagged, 16/944 proxy errors. Full deduplicated May: **1,423/14,995** flagged, **29/1,423 = 2.04%** wrong (95% Wilson 1.42–2.91%). July: **928/8,830**, **17/928 = 1.83%** (1.15–2.91%). May is a backward test against product labels, not a prospective field evaluation. On full May, the proposed 50-word hold adds 43 human reviews and prevents 2 proxy errors; 27/1,380 remaining routes disagree. Among 13,615 held cases, the top two matched 12,511 labels (91.9%) versus 8,662 (63.6%) for keywords; difference 28.3 points, day-bootstrap interval 27.5–29.1. These figures do not measure staff time or actual service improvements.

**Reproduce.** Use Python 3.12 in the pinned [full research environment](../../docs/reproduce.md). In the checkpoint edition the frozen models are bundled. In the public source edition run `bash run.sh` first to build them. From the repository root:
```bash
python src/prepare.py
python experiments/external/july_holdout.py --repo . --output data/blind_july --stage prepare
python experiments/transfer/replay.py --fetch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python experiments/scale/expand.py
```
Output aggregates and confidence intervals are in [results/](results/); [protocol](protocol.md) defines the selection, fixed before scoring. [Original May](../transfer/README.md) remains available to reproduce the subset. No complaint rows or learned challenger vocabularies are redistributed. Public archives and locally fitted challenger are stored under ignored data/. Read [results/fit.json](results/fit.json) for optimization diagnostics; a different numerical platform can yield small differences.
