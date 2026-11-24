"""Fine-tune all weights of a two-layer pretrained BERT; checkpoint on tuning macro-F1."""

import os

os.environ.update(USE_TF="0", TOKENIZERS_PARALLELISM="false")
import json, requests, torch, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    BertTokenizerFast,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score
from common import *

REV = "6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837"
SEED = 15071


def get_pretrained():
    M.mkdir(exist_ok=True)
    for file in ["config.json", "vocab.txt", "pytorch_model.bin"]:
        if not (M / file).exists():
            r = requests.get(
                "https://huggingface.co/prajjwal1/bert-tiny/resolve/"
                + REV
                + "/"
                + file,
                timeout=120,
            )
            r.raise_for_status()
            (M / file).write_bytes(r.content)


def main():
    get_pretrained()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(int(os.getenv("BERT_THREADS", "4")))
    a, y = load()
    tok = BertTokenizerFast(vocab_file=str(M / "vocab.txt"), do_lower_case=True)
    # Join beginning and end for long narratives to capture the requested remedy.
    texts = a.text.tolist()
    ids = np.zeros((len(a), 112), np.int32)
    for start in range(0, len(a), 128):
        enc = tok(
            texts[start : start + 128],
            truncation=False,
            add_special_tokens=False,
            verbose=False,
        )
        for i, t in enumerate(enc["input_ids"], start):
            t = t if len(t) <= 110 else t[:78] + t[-32:]
            seq = [101] + t + [102]
            ids[i, : len(seq)] = seq
    del enc, texts
    print("Tokenization complete", len(a), flush=True)
    x = torch.from_numpy(ids)
    yt = torch.from_numpy(y)
    cfg = BertConfig.from_json_file(str(M / "config.json"))
    cfg.num_labels = len(LABELS)
    model = BertForSequenceClassification.from_pretrained(str(M), config=cfg)
    train = np.flatnonzero(a.split == "train")
    tune = np.flatnonzero(a.split == "tune")
    loader = DataLoader(
        TensorDataset(x[train], yt[train]),
        batch_size=48,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=0.01)
    epochs = 4
    sched = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(0.08 * len(loader) * epochs),
        num_training_steps=len(loader) * epochs,
    )

    def predict(idx):
        model.eval()
        res = []
        with torch.no_grad():
            for b in x[idx].split(128):
                res.append(model(input_ids=b, attention_mask=(b != 0)).logits.numpy())
        return np.concatenate(res)

    best = -1
    hist = []
    for e in range(epochs):
        model.train()
        for step, (b, tar) in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                out = model(input_ids=b, attention_mask=b != 0, labels=tar)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            sched.step()
            if step in (2, 150, 350):
                print("epoch", e + 1, "step", step, flush=True)
        pred = predict(tune)
        score = f1_score(y[tune], pred.argmax(1), average="macro")
        hist.append(score)
        print("BERT epoch", e + 1, score, flush=True)
        if score > best:
            best = score
            model.save_pretrained(M / "fine")
            tok.save_pretrained(M / "fine")
    model = BertForSequenceClassification.from_pretrained(M / "fine")
    np.savez_compressed(
        D / "bert.npz", logits=predict(np.arange(len(a))), id=a.id.values
    )
    (R / "bert_history.json").write_text(
        json.dumps(
            {"tune_macro_f1": hist, "pretrained_revision": REV, "max_tokens": 112},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
