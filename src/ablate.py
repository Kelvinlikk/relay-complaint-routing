"""Nested feature-family ablations: select blend weights and temperature only on tune."""

import json, numpy as np, pandas as pd
from scipy.special import softmax
from sklearn.metrics import log_loss, f1_score
from evaluate import calibrated, getpolicy
from common import *


def main():
    a, y = load()
    c = np.load(D / "classic.npz")
    b = np.load(D / "bert.npz")
    va = np.flatnonzero(a.split == "tune")
    ll = {
        "TF-IDF": c["text"],
        "BERT": softmax(b["logits"], axis=1),
        "metadata": c["tab"],
    }
    pp = {n: calibrated(p, y, va)[1] for n, p in ll.items()}
    out = []
    for n1, n2 in [("TF-IDF", "metadata"), ("BERT", "metadata"), ("TF-IDF", "BERT")]:
        x, z = pp[n1], pp[n2]
        w = min(
            [0, 0.2, 0.4, 0.6, 0.8, 1],
            key=lambda w: log_loss(y[va], w * x[va] + (1 - w) * z[va], labels=range(7)),
        )
        q = calibrated(w * x + (1 - w) * z, y, va)[1]
        for split in ["tune", "test", "shift"]:
            i = np.flatnonzero(a.split == split)
            out.append(
                dict(
                    model=n1 + " + " + n2,
                    weight_first=w,
                    split=split,
                    macro_f1=f1_score(y[i], q[i].argmax(1), average="macro"),
                    nll=log_loss(y[i], q[i], labels=range(7)),
                )
            )
    pd.DataFrame(out).to_csv(R / "ablations.csv", index=False)


if __name__ == "__main__":
    main()
