"""Acquire official public archives; time-ordered sample; keep private inputs out of git."""

from pathlib import Path
import hashlib, zipfile, re, os, json, sys, unicodedata, zlib
from collections import defaultdict, Counter
import requests, pandas as pd, numpy as np
from datasketch import MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
URL = "https://files.consumerfinance.gov/f/documents/"
FILES = {
    "historical": "CCDB_Export_2_May_2018_through_April_2021.zip",
    "shift": "CCDB_Export_19_June_2026.zip",
}
COLS = [
    "Date received",
    "Product",
    "Consumer complaint narrative",
    "Company",
    "State",
    "Complaint ID",
]
LABELS = [
    "Accounts",
    "Cards",
    "Collections",
    "Credit reports",
    "Mortgages",
    "Transfers",
    "Other / review",
]


def label(s):
    s = str(s).lower()
    if s.startswith("checking or savings"):
        return "Accounts"
    if s.startswith(("credit card", "prepaid card")):
        return "Cards"
    if s == "debt collection":
        return "Collections"
    if s.startswith("credit report"):
        return "Credit reports"
    if s == "mortgage":
        return "Mortgages"
    if s.startswith("money transfer"):
        return "Transfers"
    return "Other / review"


def clean(s):
    return " ".join(
        re.findall(
            r"[a-z]+",
            re.sub(r"\bx{2,}\b", " mask ", unicodedata.normalize("NFKC", s).lower()),
        )
    )


def shingles(s):
    w = s.split()
    return set(
        zlib.crc32((" ".join(w[i : i + 3])).encode()) for i in range(max(1, len(w) - 2))
    )


def get_file(fn):
    p = DATA / fn
    if not p.exists():
        temp = p.with_suffix(".download")
        with requests.get(URL + fn, stream=True, timeout=150) as response:
            response.raise_for_status()
            with temp.open("wb") as f:
                for b in response.iter_content(1 << 20):
                    f.write(b)
        temp.replace(p)
    expected = {
        "CCDB_Export_2_May_2018_through_April_2021.zip": "4b33ffc92fa0236836228c5618b7247f0f59ffef35760a45175c858a0d5aaf45",
        "CCDB_Export_19_June_2026.zip": "3021c8af9eeed4971a92f15be21c7ce08cf3c1337151de255e5b12bf782049df",
    }[fn]
    with p.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"Archive changed: {fn}. Expected {expected}; got {actual}. Check source vintage before rerunning."
        )
    return p


def main():
    dfs = []
    info = {}
    counts = {}
    for k, fn in FILES.items():
        path = get_file(fn)
        with open(path, "rb") as f:
            info[fn] = hashlib.file_digest(f, "sha256").hexdigest()
        with zipfile.ZipFile(path) as z:
            chunks = []
            for d in pd.read_csv(
                z.open(z.namelist()[0]), usecols=COLS, chunksize=50000, low_memory=False
            ):
                d = d.dropna(subset=[COLS[2]])
                d = d[d[COLS[2]].astype(str).str.split().str.len().ge(10)]
                chunks.append(d)
            d = pd.concat(chunks, ignore_index=True)
            d.columns = [
                dict(zip(COLS, ["date", "product", "text", "company", "state", "id"]))[
                    s
                ]
                for s in d
            ]
            d["label"] = d["product"].map(label)
            d["split"] = np.select(
                [
                    d.date.lt("2020-01-01"),
                    d.date.lt("2020-05-01"),
                    d.date.lt("2020-09-01"),
                    d.date.lt("2022-01-01"),
                ],
                ["train", "tune", "policy", "test"],
                "shift",
            )
            for split, n in [
                ("train", 26000),
                ("tune", 3500),
                ("policy", 3500),
                ("test", 8000),
                ("shift", 4000),
            ]:
                g = d[d.split == split]
                if not len(g):
                    continue
                counts[split] = {"eligible": len(g), "draw": min(n, len(g))}
                g = g.assign(
                    key=g.id.astype(str).map(
                        lambda x: hashlib.blake2b(
                            ("15071:" + x).encode(), digest_size=8
                        ).hexdigest()
                    )
                )
                dfs.append(g.sort_values("key").head(n))
        del d, chunks
    data = pd.concat(dfs).sort_values(["date", "id"]).reset_index(drop=True)
    exact = set()
    lsh = MinHashLSH(threshold=0.73, num_perm=64)
    signatures = {}
    kept = []
    for r in data.itertuples():
        s = clean(r.text)
        sig = hashlib.sha256(s.encode()).digest()
        if sig in exact:
            continue
        v = shingles(s)
        if not v:
            continue
        m = MinHash(num_perm=64, seed=15071)
        arr = np.array(sorted(v), dtype=np.uint64)
        # vectorized permutation of pre-hashed word trigrams, equivalent MinHash.update_hashvalues
        a, b = m.permutations
        m.hashvalues = np.minimum(
            m.hashvalues,
            (
                ((arr[:, None] * a + b) % np.uint64((1 << 61) - 1))
                & np.uint64((1 << 32) - 1)
            ).min(axis=0),
        )
        if any(
            len(v.intersection(signatures[j])) >= 0.80 * len(v.union(signatures[j]))
            for j in lsh.query(m)
        ):
            continue
        i = r.Index
        kept.append(i)
        exact.add(sig)
        lsh.insert(i, m)
        signatures[i] = v
    out = data.loc[
        kept, ["id", "date", "label", "text", "company", "state", "split"]
    ].copy()
    out.to_pickle(DATA / "cohort.pkl")
    out[["id", "date", "label", "split"]].to_csv(DATA / "audit.csv", index=False)
    result = {
        "sha256_archives": info,
        "sampling": counts,
        "retained": out.groupby("split").size().to_dict(),
        "by_queue": pd.crosstab(out.split, out.label).to_dict(),
        "sha256_retained_ids": hashlib.sha256(
            ",".join(map(str, out.id)).encode()
        ).hexdigest(),
        "removed_in_sample": len(data) - len(out),
    }
    (ROOT / "results" / "cohort.json").write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == "__main__":
    main()
