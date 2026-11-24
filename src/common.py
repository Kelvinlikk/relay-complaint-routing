from pathlib import Path
import os, numpy as np, pandas as pd
from prepare import LABELS

BASE = Path(__file__).resolve().parents[1]
D = BASE / "data"
R = BASE / "results"
M = BASE / "models"


def load():
    a = pd.read_pickle(D / "cohort.pkl").reset_index(drop=True)
    return a, a.label.map(dict(zip(LABELS, range(len(LABELS))))).to_numpy()


def metafeatures(a):
    return pd.DataFrame(
        {
            "company": a.company.fillna("[unknown]").astype(str),
            "state": a.state.fillna("[unknown]").astype(str),
        },
        index=a.index,
    )
