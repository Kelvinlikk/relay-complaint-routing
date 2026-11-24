"""Score an invented complaint with the shipped representation-level joint model, offline."""

from pathlib import Path
import os, sys, json

os.environ.setdefault("USE_TF", "0")
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "experiments/deep"))
from fusion import score_frame, LABELS
import pandas as pd

case = json.loads((root / "demo/examples.json").read_text())[0]
x = pd.DataFrame([{k: case[k] for k in ("text", "company", "state")}])
p = score_frame(x, root, root / "experiments/deep/joint.joblib", threads=2)[0]
assert len(p) == len(LABELS) and abs(float(sum(p)) - 1) < 1e-6
print("Synthetic case:", case["short"])
print(
    "Highest-scoring research classes:",
    [(LABELS[i], round(float(p[i]), 3)) for i in p.argsort()[-3:][::-1]],
)
print(
    "This historical challenger is a model comparison, not an authorization to transfer."
)
