"""Constructed examples, not excerpts from real complaints. Text-only, unknown tenant."""

import json
from common import R
from route import route

EXAMPLES = [
    "I paid off my mortgage but the servicer still lists a balance on my home loan and has not released the lien on my house.",
    "I contacted customer support about a charge and a transfer. The payment is still missing and two departments have given me different answers.",
]
if __name__ == "__main__":
    vals = [{"text": s, **route(s)} for s in EXAMPLES]
    (R / "synthetic_examples.json").write_text(json.dumps(vals, indent=2))
    print(json.dumps(vals, indent=2))
