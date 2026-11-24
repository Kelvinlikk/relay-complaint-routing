# Public source edition

This repository contains executable training and evaluation code, aggregate evidence, and six fictional demo cases. It excludes complaint narratives, row identifiers, prediction logs, fitted vocabularies, and model weights.

- **Check the evidence:** `python verify.py` reconciles the headline results across independent aggregate tables; `python tools/read_results.py` prints the benchmarks and decision metrics. Both run offline with Python's standard library.
- **Try the demo:** open [the field walkthrough](demo/field-walkthrough.html) in a browser. It uses precomputed results for six fictional cases. Free-text scoring requires a locally fitted model.
- **Refit:** follow [the research environment and stage commands](docs/reproduce.md). The published aggregate evidence stays untouched while the scripts learn fresh local weights from the official source archives.
- **Publish:** put the contents of this edition at the root of the team's GitHub repository. See [publication and access](docs/repository-publishing.md). The private checkpoint archive is separate and is not required for grading.

The July result was a forward test of the June policy; tests or rules developed after July cannot claim a new forward evaluation on those same cases.
