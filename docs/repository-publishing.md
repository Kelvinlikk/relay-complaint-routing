# Publishing the technical submission

Use this public-source directory for the team's GitHub repository. Put `README.md`, `src/`, `experiments/`, `tools/`, and `demo/` at the repository root. Do not upload a ZIP as the only repository file: a grader should be able to open the README, inspect code, and clone the tree. Use a new repository or a branch in the team's existing project.

1. Upload this source tree, including the supplied `.gitignore`. Keep the private checkpoint bundle, archive downloads, model files, fitted vocabularies, and row-level data out of Git.
2. From a clean clone, run `python verify.py` and `python tools/read_results.py`. The first checks aggregate evidence; the second prints it. Neither needs the original complaints or network access.
3. Open [the one-page technical README](README-one-page.pdf) and [the six-case walkthrough](../demo/field-walkthrough.html) from the clone.
4. Submit the GitHub repository URL with the deck and video. A private repository needs explicit grader access; use the course's visibility policy. A local ZIP is a transfer format, not a GitHub address.

Only independently agreed team members should be credited. The repository contains no evidence of an operational deployment or partner endorsement.
