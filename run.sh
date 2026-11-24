#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 USE_TF=0 BERT_THREADS="${BERT_THREADS:-3}"
mkdir -p data models results
python src/prepare.py
python src/fit_classic.py
python src/fit_text.py
python src/evaluate.py
python src/ablate.py
python src/examples.py
