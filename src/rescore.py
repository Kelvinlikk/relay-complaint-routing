"""Rebuild row-aligned baseline scores using the delivered fit. No model fitting."""

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import json
import joblib
import numpy as np
import torch
from catboost import CatBoostClassifier
from transformers import BertForSequenceClassification, BertTokenizerFast
from common import D, M, R, load, metafeatures


def score(model, tok, texts, batch=96):
    """Use the same head/tail rule and 112-token tensors as the training script."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            examples = tok(
                texts[start : start + batch],
                truncation=False,
                add_special_tokens=False,
                verbose=False,
            )["input_ids"]
            ids = np.zeros((len(examples), 112), np.int32)
            for row, seq in enumerate(examples):
                seq = seq if len(seq) <= 110 else seq[:78] + seq[-32:]
                ids[row, : len(seq) + 2] = [101] + seq + [102]
            x = torch.from_numpy(ids)
            chunks.append(model(input_ids=x, attention_mask=x != 0).logits.numpy())
    return np.concatenate(chunks)


def main():
    torch.set_num_threads(int(os.getenv("BERT_THREADS", "3")))
    a, _ = load()
    expected = json.loads((R / "cohort.json").read_text())
    assert len(a) == sum(expected["retained"].values())
    linear = joblib.load(M / "linear.joblib")
    tab = CatBoostClassifier()
    tab.load_model(str(M / "tab.cbm"))
    metadata = metafeatures(a)
    m = metadata.copy()
    m["company"] = "[unknown]"
    np.savez_compressed(
        D / "classic.npz",
        text=linear.predict_proba(a.text),
        tab=tab.predict_proba(metadata, thread_count=2),
        masked=tab.predict_proba(m, thread_count=2),
        id=a.id.values,
    )
    tok = BertTokenizerFast.from_pretrained(M / "fine", local_files_only=True)
    mdl = BertForSequenceClassification.from_pretrained(
        M / "fine", local_files_only=True
    )
    np.savez_compressed(
        D / "bert.npz", logits=score(mdl, tok, a.text.tolist()), id=a.id.values
    )
    print("Rebuilt baseline probability inputs for", len(a), "rows. No fitting.")


if __name__ == "__main__":
    main()
