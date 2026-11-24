# Reproducing Relay

The public edition offers two independent paths. The first checks the published evidence exactly, without downloads. The second obtains the source data and refits the models. Training is not necessary to inspect or grade the recorded decision.

## 1. Check the evidence offline

From the repository root, with Python 3.12:

```bash
python verify.py
python tools/read_results.py
```

The verifier reconciles historical cohorts, July confusion matrices and release counts, May shortlist membership by product, the June policy hash, the cost scenarios, and expanded-data estimates. It compares aggregate tables; it does not reconstruct individual predictions from them. Open [the walkthrough](../demo/field-walkthrough.html) for six fictional, precomputed decisions. With NumPy, SciPy and pandas installed, `python verify.py --numerical` additionally exercises the live policy and randomization analysis on synthetic inputs.

## 2. Refit using public data

Use Bash and Python 3.12. The original runs used Linux CPU. Create a virtual environment and install the pinned versions. On Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
bash tools/rebuild_public.sh
```

On macOS install directly from `requirements.txt` inside the environment, then use the same Bash command; exact numerical parity across hardware is not guaranteed. The [source list](sources.md) identifies five CFPB narrative ZIPs: May 2018–April 2021 combined, and the May, June, July and August 2026 snapshots. The frozen upstream revision of `prajjwal1/bert-tiny` supplies the starting encoder. The scripts verify SHA-256 hashes for archived complaint files. Archive downloads, fitting and duplicate screening require internet access, CPU time, memory and disk. If an archive hash changes, stop and retrieve the named vintage rather than substituting the live CFPB download.

The runner creates **`data/public-refit/`** and runs only there. The committed tables in the original checkout remain unchanged. Run `bash tools/rebuild_public.sh --plan` to see all stages; `bash tools/rebuild_public.sh stage` creates the isolated directory without downloading or training. The remaining stages, in order, can be run separately:

| Command suffix | What runs | Local output inside `data/public-refit/` |
|---|---|---|
| `fit` | Sample chronological cohorts, screen duplicates, train historical TF-IDF, CatBoost and BERT, select hyperparameters and blend | `models/`, `data/`, `results/` |
| `replay` | Refit the June calibration, derive destination cutoffs, score July, audit challenger heads and costs | `data/replication/{policy,external,deep,research}/` |
| `extensions` | Apply the published June rule to locally refitted base scores; audit original/expanded May, August availability and the larger historical text model | `experiments/{transfer,scale}/results/` |

Compare regenerated tables with the corresponding committed reference (for example `data/public-refit/data/replication/policy/july_metrics.json` with `experiments/policy/july_metrics.json`). The extensions write `experiments/transfer/results/refit_comparison.json` with both observed and published July counts, without overwriting the reference or describing a fresh fit as the original checkpoint. Archive, protocol, label-order and cohort checks remain enforced. If the refit chooses a different underlying ensemble, stop: it is a different candidate, not an exact operational replay. Do not adjust thresholds to match the published July outcome. The [data expansion](../experiments/scale/README.md) refits a separate text challenger; its larger training set never changed the documented July rule.

For arbitrary **fictional** input after fitting, run the local desk from the isolated tree: `cd data/public-refit && python demo/desk.py --repo .`. Its scores depend on the new fit. The public source does not contain checkpoint files: `tools/check_assets.py` and `tools/try_deep.py` are for the separate private checkpoint edition. Never commit locally generated complaint rows, IDs, traces, weights or learned vocabularies.
