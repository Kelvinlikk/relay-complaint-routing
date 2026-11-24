"""Train classic benchmarks. The company field represents a known tenant only."""

import os, json, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from common import *


def main():
    a, y = load()
    tr = a.split.eq("train").to_numpy()
    va = a.split.eq("tune").to_numpy()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=90000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X = vec.fit_transform(a.text[tr])
    vt = vec.transform(a.text[va])
    scores = {}
    best = -1
    for C in [0.5, 2, 6]:
        mod = LogisticRegression(
            C=C, solver="lbfgs", max_iter=240, random_state=15071
        ).fit(X, y[tr])
        s = f1_score(y[va], mod.predict(vt), average="macro")
        scores[str(C)] = s
        if s > best:
            best = s
            winner = mod
    t = metafeatures(a)
    tab = CatBoostClassifier(
        iterations=320,
        depth=6,
        learning_rate=0.12,
        thread_count=2,
        loss_function="MultiClass",
        random_seed=15071,
        verbose=False,
        allow_writing_files=False,
    )
    tab.fit(t[tr], y[tr], cat_features=["company", "state"])
    tab.save_model(str(M / "tab.cbm"))
    pipe = make_pipeline(vec, winner)
    joblib.dump(pipe, M / "linear.joblib")
    pro = pipe.predict_proba(a.text)
    no = t.copy()
    no["company"] = "[unknown]"
    np.savez_compressed(
        D / "classic.npz",
        text=pro,
        tab=tab.predict_proba(t),
        masked=tab.predict_proba(no),
        id=a.id.values,
    )
    (R / "linear_tuning.json").write_text(json.dumps(scores, indent=2))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
