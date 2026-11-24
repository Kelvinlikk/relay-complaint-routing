"""Historical baseline only: replay a synthetic or locally approved example. Current decision desk: demo/desk.py."""

import os, sys, json, argparse

os.environ.setdefault("USE_TF", "0")
import numpy as np, joblib, torch
from scipy.special import softmax
from catboost import CatBoostClassifier
from transformers import BertTokenizerFast, BertForSequenceClassification
from common import *


def route(text, company=None, state=None):
    if len(text.split()) < 10:
        return {
            "decision": "Human review",
            "reason": "Fewer than 10 words; outside the training scope.",
        }
    pol = json.loads((R / "policy.json").read_text())
    k = pol["chosen"] if company else "TF-IDF"
    t = pol["temperatures"]
    p = {}
    threshold = pol["thresholds"][k]

    def temp(v, n):
        return softmax(np.log(np.clip(v, 1e-9, 1)) / t[n])

    if k in ["Combined", "TF-IDF"]:
        p["TF-IDF"] = temp(
            joblib.load(M / "linear.joblib").predict_proba([text])[0], "TF-IDF"
        )
    if k in ["Combined", "CatBoost"]:
        tab = CatBoostClassifier()
        tab.load_model(str(M / "tab.cbm"))
        p["CatBoost"] = temp(
            tab.predict_proba([[company or "[unknown]", state or "[unknown]"]])[0],
            "CatBoost",
        )
    if k == "TinyBERT" or (k == "Combined" and pol["weights"]["TinyBERT"] > 0):
        tok = BertTokenizerFast.from_pretrained(M / "fine")
        mdl = BertForSequenceClassification.from_pretrained(M / "fine")
        mdl.eval()
        token = tok(text, add_special_tokens=False)["input_ids"]
        token = token if len(token) <= 110 else token[:78] + token[-32:]
        x = torch.tensor([[101] + token + [102]])
        with torch.no_grad():
            q = mdl(input_ids=x, attention_mask=torch.ones_like(x)).logits.numpy()[0]
        p["TinyBERT"] = temp(softmax(q), "TinyBERT")
    prob = (
        temp(sum(w * p[s] for s, w in pol["weights"].items() if w > 0), "Combined")
        if k == "Combined"
        else p[k]
    )
    ix = prob.argmax()
    route = LABELS[ix]
    accept = ix != 6 and prob[ix] >= threshold
    return {
        "decision": route if accept else "Human review",
        "suggested_queue": route,
        "confidence": round(float(prob[ix]), 4),
        "threshold": round(threshold, 4),
        "model": k,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--company")
    parser.add_argument("--state")
    args = parser.parse_args()
    print(json.dumps(route(args.text, args.company, args.state), indent=2))
