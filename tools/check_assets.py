"""Verify shipped model and policy files against their versioned SHA-256 list."""

from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for fname, h in json.loads((root / "results" / "manifest.json").read_text()).items():
    path = root / fname
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != h:
        raise SystemExit(f"Check failed for {fname}")
print("Confirmed package integrity for all listed files.")
