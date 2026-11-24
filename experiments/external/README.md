# Independent July 2026 evaluation cohort
The source is the [CFPB Narratives Archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), [July ZIP](https://files.consumerfinance.gov/f/documents/CCDB_Export_20_July_2026.zip), SHA-256 `aafebbaaf0962c3d29532411ac2133c01bae830b7ea6ed1931a6e42278437d18`.

The previously available data end June 30, 2026. All eligible July published narratives (at least ten whitespace-separated words) enter duplicate screening regardless of target category, client, or outcome. Cross-set and within-July near duplicates are screened via the original normalization, MinHash and exact candidate-Jaccard procedure, retaining the first July occurrence and rejecting a July duplicate against **any eligible June narrative** (including June records omitted by sampling/dedup), or an earlier retained reference. This is stronger than merely checking the adaptation sample. Approximate LSH does not catch every paraphrase. The file is the snapshot published in the archive, not a random sample of production mail. Some July narratives may be unpublished or too recent to have been cleared; conclusions describe only eligible published complaints.

From the repository root after reproducing the historical artifacts:
```bash
OPENBLAS_NUM_THREADS=1 BERT_THREADS=3 python experiments/external/july_holdout.py --repo . --output data/blind_july
```
No training or selection takes place here. `july_predictions.npz` contains raw probabilities, BERT logits, the existing calibrated blend and justified metadata. Labels are written separately to `july_labels.npz`; no outcome distribution, quality measure or policy inspection is performed during preparation. Row text, complaint IDs, labels and per-case predictions should remain local. The public manifest gives the archive hash, retention counts and comparison universe without disclosing individual records.

The standalone diagnostic script `july_slices.py` is intentionally separate from blind preparation. **Run only after checking the frozen policy**; it reports error and coverage by date, length, and institution history plus a percentile institution-cluster bootstrap. These are exploratory reliability checks, not new approvals or comparisons of protected groups.
