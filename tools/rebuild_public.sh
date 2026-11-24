#!/usr/bin/env bash
# Isolate a complete source refit from the committed reference tables.
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-all}"
case "$mode" in
  --help|-h|--plan)
    cat <<'TEXT'
Usage: bash tools/rebuild_public.sh [all|stage|fit|replay|extensions]
stage       Copy only source and aggregate references into the ignored data/public-refit/ directory.
fit         Fit original models and select on historical validation data (run.sh).
replay      With fitted models, replay the June policy audit and July holdout (reproduce.sh).
extensions  Recompute May/August transfers, reviewer suggestions, and the 54,008-case challenger.
all         Stage if needed, then run fit, replay, and extensions in order (default).
Requires Python 3.12 + full dependencies and network access for training. The original
source directory is not modified. Stages may be rerun in the SAME worktree; remove or
rename data/public-refit yourself if you want a fresh independent refit.
TEXT
    exit ;;
  all|stage|fit|replay|extensions) ;;
  *) echo "Unknown stage: $mode. Run with --help." >&2; exit 2 ;;
esac
work="data/public-refit"
if [[ ! -f "$work/.relay-refit" ]]; then
    if [[ -e "$work" ]]; then
        echo "$work already exists without a Relay marker. Rename it before continuing." >&2; exit 2
    fi
    python - <<'PY'
import shutil
from pathlib import Path
root = Path.cwd(); dst = root / "data/public-refit"
dst.mkdir(parents=True)
ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.npz", "*.pkl", "*.joblib", "*.safetensors", "*.bin", "*.cbm", "*.zip", "*.bundle", "private", "private_*")
for name in ("src", "experiments", "results", "tools", "demo", "docs"):
    shutil.copytree(root / name, dst / name, ignore=ignore)
for name in ("verify.py", "reproduce.sh", "run.sh", "requirements.txt", "README.md", "SOURCE-EDITION.md"):
    shutil.copy2(root / name, dst / name)
(dst / ".relay-refit").write_text("Isolated local rerun of the public source.\n")
PY
fi
cd "$work"
if [[ "$mode" == "stage" ]]; then echo "Staged a clean, isolated research tree at $PWD"; exit; fi
python - <<'PY'
import importlib.metadata as im
from pathlib import Path
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit("Full reference refit requires Python 3.12; see docs/reproduce.md.")
problems = []
for line in Path("requirements.txt").read_text().splitlines():
    if "==" not in line: continue
    name, wanted = line.split("==")
    try: actual = im.version(name).split("+")[0]
    except im.PackageNotFoundError: actual = "missing"
    if actual != wanted: problems.append(f"{name}: need {wanted}, found {actual}")
if problems:
    sys.exit("Install the research environment in docs/reproduce.md before training:\n  " + "\n  ".join(problems))
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
if [[ "$mode" == "all" || "$mode" == "fit" ]]; then bash run.sh; fi
if [[ "$mode" == "all" || "$mode" == "replay" ]]; then
    [[ -f models/fine/config.json ]] || { echo "Run the fit stage first." >&2; exit 2; }
    SKIP_ASSET_CHECK=1 bash reproduce.sh
fi
if [[ "$mode" == "all" || "$mode" == "extensions" ]]; then
    [[ -f data/blind_july/july_labels.npz && -f models/linear.joblib ]] || { echo "Run fit and replay first." >&2; exit 2; }
    python experiments/transfer/replay.py --fetch --refit
    python experiments/transfer/pilot.py design > experiments/transfer/results/local_gate_design.json
    python experiments/scale/expand.py
fi
echo "Completed $mode in $PWD. Compare the regenerated tables with the original checkout; field outcomes were not measured."
