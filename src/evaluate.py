"""Model choices on tune plus a separate policy window; both later sets untouched."""

import json, itertools
import pandas as pd, numpy as np
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix
from common import *


def ci(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [max(0.0, ctr - half), min(1.0, ctr + half)]


def calibrated(p, y, idx):
    log = np.log(np.clip(p, 1e-9, 1))
    t = minimize_scalar(
        lambda t: log_loss(
            y[idx], softmax(log[idx] / t, axis=1), labels=range(len(LABELS))
        ),
        bounds=(0.3, 4.0),
        method="bounded",
    ).x
    return float(t), softmax(log / t, axis=1)


def getpolicy(p, y, ix, th):
    v = p[ix]
    pred = v.argmax(1)
    mask = (v.max(1) >= th) & (pred != LABELS.index("Other / review"))
    n = int(mask.sum())
    err = int((pred[mask] != y[ix][mask]).sum())
    return {
        "n": len(ix),
        "auto": n,
        "wrong": err,
        "coverage": n / len(ix),
        "risk": err / n if n else None,
        "risk_ci": ci(err, n),
        "coverage_ci": ci(n, len(ix)),
    }


def main():
    a, y = load()
    c = np.load(D / "classic.npz")
    b = np.load(D / "bert.npz")
    assert np.array_equal(c["id"], a.id) and np.array_equal(b["id"], a.id)
    ix = {s: np.flatnonzero(a.split == s) for s in a.split.unique()}
    va = ix["tune"]
    pi = ix["policy"]
    raw = {
        "TF-IDF": c["text"],
        "CatBoost": c["tab"],
        "TinyBERT": softmax(b["logits"], axis=1),
    }
    t = {}
    p = {}
    for k, v in raw.items():
        t[k], p[k] = calibrated(v, y, va)
    cand = []
    for l in range(6):
        for h in range(6 - l):
            g = 5 - l - h
            v = (l * p["TF-IDF"] + h * p["TinyBERT"] + g * p["CatBoost"]) / 5
            loss = log_loss(y[va], v[va], labels=range(7))
            cand.append((loss, l / 5, h / 5, g / 5))
    _, l, h, g = min(cand)
    comb = l * p["TF-IDF"] + h * p["TinyBERT"] + g * p["CatBoost"]
    t["Combined"], p["Combined"] = calibrated(comb, y, va)
    train = ix["train"]
    prior = np.bincount(y[train], minlength=7) / len(train)
    p["Majority"] = np.tile(prior, (len(y), 1))
    benchmark = []
    thresholds = {}
    curves = []
    pol = {}
    for k, prob in p.items():
        for s in ["tune", "policy", "test", "shift"]:
            i = ix[s]
            pred = prob[i].argmax(1)
            benchmark.append(
                {
                    "model": k,
                    "split": s,
                    "accuracy": accuracy_score(y[i], pred),
                    "macro_f1": f1_score(y[i], pred, average="macro", labels=range(7)),
                    "nll": log_loss(y[i], prob[i], labels=range(7)),
                }
            )
        # Predeclared operating rule: search fixed grid and select the largest feasible coverage.
        options = []
        for cut in np.arange(0.50, 0.996, 0.005):
            v = getpolicy(prob, y, pi, cut)
            if v["auto"] >= 100 and v["risk_ci"][1] <= 0.05:
                options.append((v["auto"], -float(cut)))
        cut = -max(options)[1] if options else 1.01
        thresholds[k] = cut
        pol[k] = {}
        for s in ["policy", "test", "shift"]:
            pol[k][s] = getpolicy(prob, y, ix[s], cut)
            for grid in np.arange(0.5, 0.996, 0.01):
                curves.append(
                    dict(
                        model=k,
                        split=s,
                        threshold=round(float(grid), 3),
                        **getpolicy(prob, y, ix[s], grid),
                    )
                )
    # Pick the highest policy coverage with a qualifying validation bound; use simpler model on an exact tie.
    best = max(
        ["TF-IDF", "CatBoost", "TinyBERT", "Combined"],
        key=lambda k: pol[k]["policy"]["auto"],
    )
    v = p[best]
    cut = thresholds[best]
    qs = []
    ms = []
    costs = []
    for s in ["policy", "test", "shift"]:
        i = ix[s]
        pred = v[i].argmax(1)
        ys = y[i]
        mask = (pred != 6) & (v[i].max(1) >= cut)
        for j, name in enumerate(LABELS):
            part = ys == j
            dest = mask & (pred == j)
            n = int(dest.sum())
            e = int((ys[dest] != j).sum())
            qs.append(
                {
                    "split": s,
                    "queue": name,
                    "true": int(part.sum()),
                    "recall": float((pred[part] == j).mean()),
                    "true_class_auto_coverage": float(mask[part].mean()),
                    "sent_here": n,
                    "wrong": e,
                    "risk": e / n if n else None,
                    "risk_upper": ci(e, n)[1],
                }
            )
        mat = confusion_matrix(ys[mask], pred[mask], labels=range(7))
        for j, jj in itertools.product(range(7), repeat=2):
            ms.append(
                {
                    "split": s,
                    "actual": LABELS[j],
                    "route": LABELS[jj],
                    "n": int(mat[j, jj]),
                }
            )
        for minutes, hour, cost in itertools.product([2, 4, 7], [30], [10, 25, 50]):
            saved = (
                (mask.sum() * minutes * hour / 60 - ((pred != ys) & mask).sum() * cost)
                / len(i)
                * 10000
            )
            costs.append(
                {
                    "split": s,
                    "review_minutes": minutes,
                    "loaded_dollars_hr": hour,
                    "incremental_wrong_cost": cost,
                    "net_per_10000": saved,
                }
            )
    # If company is not a pre-existing tenant field, test the locked fusion with company removed and text only.
    masked = softmax(np.log(np.clip(c["masked"], 1e-9, 1)) / t["CatBoost"], axis=1)
    masked = softmax(
        np.log(np.clip(l * p["TF-IDF"] + h * p["TinyBERT"] + g * masked, 1e-9, 1))
        / t["Combined"],
        axis=1,
    )
    mask_pol = {
        s: getpolicy(masked, y, ix[s], thresholds["Combined"])
        for s in ["test", "shift"]
    }
    tables = {
        "benchmarks": benchmark,
        "queue_metrics": qs,
        "auto_confusion": ms,
        "costs": costs,
        "risk_curve": curves,
    }
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(R / (name + ".csv"), index=False)
    # Conservative month-cluster bootstrap sensitivity: test contains eight calendar months.
    rng = np.random.default_rng(15071)
    ii = ix["test"]
    mons = a.date.iloc[ii].str[:7].to_numpy()
    bins = [ii[mons == m] for m in sorted(set(mons))]
    z = []
    for _ in range(1000):
        j = np.concatenate(
            [bins[b] for b in rng.integers(0, len(bins), size=len(bins))]
        )
        v0 = getpolicy(v, y, j, cut)
        z.append([v0["risk"], v0["coverage"]])
    res = {
        "labels": LABELS,
        "chosen": best,
        "thresholds": thresholds,
        "temperatures": t,
        "weights": {"TF-IDF": l, "TinyBERT": h, "CatBoost": g},
        "policies": pol,
        "masked_company_combined": mask_pol,
        "month_cluster_test_95pct": np.quantile(z, [0.025, 0.975], axis=0).tolist(),
    }
    (R / "policy.json").write_text(json.dumps(res, indent=2))
    print(best, thresholds[best], pol[best], res["weights"], flush=True)


if __name__ == "__main__":
    main()
