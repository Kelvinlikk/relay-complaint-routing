# What can Relay do for cases it will not send?
We evaluated a second decision: a reviewer receives two suggestions and may select either, any other queue, or escalation. The underlying June score and gates stayed frozen. July had already been opened when the short-text rule and ranking question were formulated. May was not used for model or policy development. Because it predates June, May is a **backward transport check**, not a later validation or a bank trial.

| Among cases held by the rule with the short-text safeguard | Model, first choice | Model, first two | Keyword rule, first two | Two most common June labels |
|---|---:|---:|---:|---:|
| May, initial subset (4,793 held of 5,254) | 77.5% | 4,416 / 4,793 = 92.1% | 3,064 / 4,793 = 63.9% | 46.3% |
| July (7,931 held of 8,830) | 77.8% | 7,233 / 7,931 = 91.2% | 5,023 / 7,931 = 63.3% | 42.0% |

The [expanded May analysis](../scale/README.md) uses the full 14,995-case cohort. Among its 13,615 held cases, the correct product appeared in the top two 91.9% of the time for the model versus 63.6% for keywords. The earlier subset is retained here so its original analysis can be reproduced.

The July lift over the keyword rule was **27.9 percentage points** (paired receipt-day bootstrap, 95% interval 26.3–29.1); May: **28.2 points** (27.1–29.4). In July, 2,405 cases were hits for the model alone and 195 for the keyword rule alone. The model’s two suggestions both missed in 698 July cases. The weakest July hit rate was Transfers, at 659 / 771 = 85.5%. “Other / review” is a legitimate choice. A hit only means the recorded public **product label** is on the list; it neither resolves the complaint nor shows how a person would respond. To measure speed, accuracy and possible anchoring, use the [blinded adjudication and randomized human trial](../../docs/field-protocol.md).

**Why 7,931 instead of 7,902?** The June policy routed 928 in July and left 7,902 for manual review. The proposed 50-word safeguard withholds an additional 29. This offline extension studies 7,931 held cases; it does not rewrite the independent July audit of the original policy.

**Test the new length guard, not just the stories that inspired it.** In the July diagnostic, it would remove five mistakes among 29 short routes, lowering error from 17/928 to 12/899. On the initial May subset, the original gates routed 479/5,254 with 13 errors. Requiring at least 50 words routed 461 with 12 errors: 18 additional reviews prevented **one** proxy misroute. Under the illustrative $2 first review, 10% QA and $25 rework assumptions, that change has a **−$7.40** total incremental variable value in May ($25 avoided − 18 × $1.80 forgone). A length guard can be a safety preference even when its short-run labor economics are unfavorable. Cards under the May length gate had 7 errors / 224 (95% Wilson upper 6.31%); mortgages had 5 / 237 (upper 4.84%). Do not grant a desk local permission based on a pooled rate.

**The next month could not answer the question.** The official August archive has 667,140 records but only one narrative with at least ten words. That case was held; misrouting among automatic releases is **undefined**, not zero. No August effectiveness estimate is possible. The CFPB [announced it had stopped publishing complaint narratives on August 14, 2026](https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/). This does not establish a change in bank demand, and the published archive should not be used as an ongoing live monitoring feed.

**Design.** The forward protocol and May protocol were saved as separate Git commits before inspecting the respective outcomes. The 6,000-case deterministic May draw was from 17,488 eligible narratives. Of those sampled, 746 were removed as exact or approximate template overlaps, leaving 5,254. Reference screening uses all 63,522 historical-study/June/July eligible unique reference IDs (including the historical study’s sampled June overlap only once), word-trigram MinHash LSH with verified Jaccard ≥0.80; near-candidate retrieval is approximate. The length rule was selected after July. The May test used the same model and probabilities with no threshold or queue search. The simple keyword baseline counts distinct predefined phrases; lexical rules were authored after the original project but fixed before May was opened. Ties and a naive fixed-rank comparator use raw eligible June 1–7 product counts. Inference assumes an independently known client mailbox; archive recipients remain an optimistic proxy. Bootstrap uncertainty excludes label ambiguity and future shifts.

**Replay.** In the public edition, run the [isolated complete refit](../../docs/reproduce.md) first. For a checkpoint workstation with fitted base models, the equivalent replay commands below use those existing weights:
```bash
python src/prepare.py
python experiments/external/july_holdout.py --repo . --output data/blind_july --stage prepare
python experiments/transfer/replay.py --fetch
python experiments/transfer/pilot.py design
```
The first time, official archives and derived inputs are held in the ignored `data/` folder. The latest experiment requires the demo dependencies plus `datasketch==2.0.0` and `requests`. Only aggregate results, code, hashes, the protocol and the keyword lexicon are published. [Full results](results/) include all seven product breakdowns, the two comparator lifts and both policies with distinct denominators.
