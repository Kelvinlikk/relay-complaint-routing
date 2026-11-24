#!/usr/bin/env bash
# Recompute published evidence from source archives. The supplied model fit is frozen.
# Run in your own checkout. Private simulations and complaint-level features stay in data/.
set -euo pipefail
cd "$(dirname "$0")"
export USE_TF=0 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export BERT_THREADS="${BERT_THREADS:-3}"
J=data/blind_july
P=data/replication/policy
E=data/replication/external
F=data/replication/deep
A=data/replication/research
mkdir -p "$P" "$J" "$E" "$F" "$A"
if [[ "${SKIP_ASSET_CHECK:-0}" != 1 ]]; then python tools/check_assets.py; fi
python src/prepare.py
python src/rescore.py
python src/evaluate.py
python src/ablate.py
python experiments/research/archive_funnel.py data/CCDB_Export_2_May_2018_through_April_2021.zip data/CCDB_Export_19_June_2026.zip "$A/archive_funnel.csv"
python experiments/research/mix_replay.py --results-dir results --out "$A"
python experiments/policy/prepare_june.py --repo . --out "$P"
python experiments/policy/shift_policy.py --june "$P/private/june_scores.npz" --out "$P"
python experiments/external/july_holdout.py --repo . --output "$J"
python experiments/deep/fusion.py --repo . --out "$F" --threads "$BERT_THREADS"
python experiments/deep/july_challenger.py score --repo . --inputs "$J/blind_inputs.pkl" --model-dir "$F" --scores "$F/private_july_scores.npz" --threads "$BERT_THREADS"
python experiments/policy/shift_policy.py --july-predictions "$J/july_predictions.npz" --july-labels "$J/july_labels.npz" --out "$P"
python experiments/external/july_slices.py --repo . --policy-dir experiments/policy --lock "$P/locked_policy.json" --blind "$J" --out "$E"
read -r N R W < <(python -c 'import json,sys; p=next(x for x in json.load(open(sys.argv[1]))["rows"] if x["policy"]=="certified_destinations"); print(p["n"],p["auto"],p["wrong"])' "$P/july_metrics.json")
python experiments/research/pilot_economics.py "$N" "$R" "$W" "$A/pilot_economics.csv"
python experiments/deep/july_challenger.py audit --repo . --labels "$J/july_labels.npz" --model-dir "$F" --scores "$F/private_july_scores.npz"
printf 'Finished. Row-level data and regenerated aggregate results are under data/. Reference aggregates are in experiments/ and results/.\n'
