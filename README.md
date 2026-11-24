# Relay: financial complaint routing

**Project team:** Marcello Fagotto and Kelvin Li  
MIT Sloan School of Management | 15.071 The AI Edge

Relay helps a financial services intake team decide which complaints to send to a product specialist and which to hold for review. The classifier ranks six specialist desks plus “Other / review.” Reviewers can accept a recommendation, choose another desk or escalate; Relay does not transfer live cases.

## Model and inputs

Use the complaint text, state and an independently identified client mailbox. In the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/), the recorded company stands in for the mailbox. The recorded product is a proxy for the correct desk. Product, issue and resolution fields are excluded from prediction. We split by receipt date and screened for exact and near-duplicate narratives.

The model used for routing blends TF-IDF logistic regression (80%) with CatBoost on intake fields (20%). On 6,917 historical test cases, adding intake fields raised fine-tuned BERT’s macro-F1 from 0.706 to 0.769; sparse text plus intake scored 0.791. These challengers did not change the frozen policy.

## What the evaluation supports

The June 2026 audit targeted at most 5% disagreement among flagged cases per desk. Only cards and mortgages qualified. On the later July holdout, the frozen rule flagged **928 of 8,830** eligible complaints (10.5%). **17 of 928** disagreed with the recorded product (1.83%; 95% Wilson interval: 1.15–2.91%).

In a separate May retrospective, a follow-up rule held messages under 50 words. Among the 13,615 cases retained for review, Relay’s first two choices included the recorded product in **91.9%**, versus **63.6%** for a keyword rule. This does not measure reviewer accuracy or time saved. Test staff-confirmed handoffs locally before enabling transfers.

## Run from the repository root

**Offline check.** Run these commands to verify the published counts and print the comparison tables. They use only Python’s standard library; no downloads or model weights are required:
```bash
python verify.py
python tools/read_results.py
```
Open `demo/field-walkthrough.html` in a browser to try six fictional cases. This walkthrough uses saved examples; it does not score new text.

**Full reproduction.** In a fresh Python 3.12 environment, install the CPU dependencies in [docs/reproduce.md](docs/reproduce.md). The script below refits the models and replays the evaluations using five CFPB archives and a pinned pretrained BERT checkpoint. It requires internet access; CPU and library differences can affect the refit.
```bash
bash tools/rebuild_public.sh
```
See [experiments/scale/README.md](experiments/scale/README.md) for the expanded training and May cohorts, and [docs/field-protocol.md](docs/field-protocol.md) for the proposed local trial. Keep downloaded complaints, fitted assets and case-level outputs out of Git.
