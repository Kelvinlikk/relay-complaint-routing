# Decision for the head of complaint operations

**Run a shadow pilot on card and mortgage complaints. Staff confirm every handoff.** Relay proposes a destination, never a remedy. For complaints that remain in intake, test whether two ranked suggestions help reviewers. The decisions require different evidence: a model can name the right desk without saving a reviewer any time.

## Which handoffs qualify?

Keep the historical text-and-metadata classifier fixed. Use June 1–7 to adapt its scores and June 8–11 to choose a cutoff for each destination. No desk qualified in the initial June 12–15 audit. After seeing that result, the team added a cumulative June 12–30 audit of the unchanged cutoffs. Exact binomial tests used Holm correction across six desks at 2.5% per look. Only cards (score ≥0.94) and mortgages (≥0.86) qualified. This second look was retrospective; both the rules and the opportunity to recheck them were fixed before July labels were opened. See the [audit protocol](../experiments/policy/README-policy.md).

All four rules below saw the same **8,830** deduplicated July complaints with published narratives of at least ten words. “Wrong” means the suggested desk disagreed with the recorded product; no specialist adjudicated the archive. The recorded recipient is only a proxy for a client independently known at intake.

| Rule | Would route | Disagreed / routed | Error among routed |
|---|---:|---:|---:|
| Historical 0.90 cutoff | 2,695 | 337 / 2,695 | 12.50% |
| Recalibrated scores, same 0.90 cutoff | 3,104 | 191 / 3,104 | 6.15% |
| Recalibrated scores, one audited 0.94 cutoff | 2,286 | 112 / 2,286 | 4.90% |
| Separate cutoff and audit per desk | 928 | 17 / 928 | 1.83% |

The pooled rule hid **37 disagreements among 106 credit-report recommendations** (34.9%). The separate rules flagged 486 card complaints (12 disagreements) and 442 mortgage complaints (5). This is 10.5% coverage, leaving 7,902 for review. The overall 95% Wilson interval for disagreement was 1.15–2.91%; resampling receipt dates gave 1.07–2.67% and resampling institutions gave 1.16–2.53%. These intervals do not capture errors in the labels or shifts in the local inbox. Reviewing every case produces no automated assignments and hence no automated-error rate.

## What should a reviewer see?

Among July's 29 flagged complaints of 10–49 words, five disagreed with the recorded product. **The proposed 50-word minimum was chosen after examining July.** It would have left 899 candidate handoffs and 7,931 human reviews. On those 7,931, the recorded product appeared among Relay's top two for 7,233 (91.2%) and among the keyword rule's top two for 5,023 (63.3%). July independently tests the original June handoff policy, not the later length or shortlist decision.

The [expanded May retrospective](../experiments/scale/README.md) gives an earlier check with unchanged rules. The original gate flagged 1,423 of 14,995, of which 29 disagreed (2.04%; 95% Wilson interval 1.42–2.91%). The 50-word hold sent another 43 to review, avoiding two proxy errors. On the resulting 13,615 held complaints, Relay's top two contained 12,511 recorded products (91.9%), versus 8,662 for keywords (63.6%). The paired gain was 28.3 percentage points (95% interval 27.5–29.1, resampling receipt dates). Staff must be able to dismiss both suggestions. This comparison does not measure whether hints improve real decisions.

## What might the pilot be worth?

At four minutes per first review, $30 per staff hour, $25 incremental rework per wrong route and a 10% full-cost spot audit, **future automation of the original July rule** would free an estimated 63 first-pass hours and yield $1,410 in net variable value per 10,000 eligible complaints. The high-rework scenario loses $17. Neither is a saving from the shadow trial: staff still review every case. The denominator excludes complaints without usable public narratives; implementation, local labelling, critical delays and the ability to redeploy staff are not priced. The [calculator](../experiments/research/pilot_economics.py) and [sensitivity table](../results/pilot_economics.csv) show the assumptions.

## What would justify a change in policy?

Require an independently known client and keep urgency, vulnerability, language and deadline checks active. Test the candidate handoffs against independent local adjudication. In a separate randomized comparison, measure reviewer accuracy and active time with and without suggestions. Audit held cases as well as candidate transfers; pause on serious delays or loss of reliable client identity. Follow the [fixed local release rule and trial plan](field-protocol.md). No public archive test grants authority to move a live complaint.
