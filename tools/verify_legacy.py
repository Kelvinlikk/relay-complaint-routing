"""Cross-check historical counts, costs and disclosure denominators; stdlib only."""
import csv
import json
from math import isclose
from pathlib import Path


def verify(root):
    result = root / "results"

    def rows(name):
        with (result / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    model = json.loads((result / "policy.json").read_text())
    cohort = json.loads((result / "cohort.json").read_text())
    queues = rows("queue_metrics.csv")
    confusions = rows("auto_confusion.csv")
    costs = rows("costs.csv")
    for split in ("policy", "test", "shift"):
        s = model["policies"][model["chosen"]][split]
        q = [r for r in queues if r["split"] == split]
        assert len(q) == 7
        assert s["n"] == sum(int(r["true"]) for r in q) == cohort["retained"][split]
        assert s["auto"] == sum(int(r["sent_here"]) for r in q)
        assert s["wrong"] == sum(int(r["wrong"]) for r in q)
        assert not int(next(r["sent_here"] for r in q if r["queue"] == "Other / review"))
        matrix = [r for r in confusions if r["split"] == split]
        assert s["auto"] == sum(int(r["n"]) for r in matrix)
        assert s["wrong"] == sum(int(r["n"]) for r in matrix if r["actual"] != r["route"])
        assert isclose(s["wrong"] / s["auto"], s["risk"])
        assert isclose(s["auto"] / s["n"], s["coverage"])
        for r in (r for r in costs if r["split"] == split):
            savings = s["auto"] * float(r["loaded_dollars_hr"]) * float(r["review_minutes"]) / 60
            rework = s["wrong"] * float(r["incremental_wrong_cost"])
            assert isclose((savings - rework) * 10000 / s["n"], float(r["net_per_10000"]), abs_tol=.0001)
    assert isclose(sum(model["weights"].values()), 1)
    assert model["policies"][model["chosen"]]["policy"]["risk_ci"][1] <= .05
    funnel = rows("archive_funnel.csv")
    for period in {r["period"] for r in funnel}:
        group = [r for r in funnel if r["period"] == period]
        assert len(group) == 2
        for field in ["share_of_public_records", "share_of_usable_texts"]:
            assert isclose(sum(float(r[field]) for r in group), 1)
        for r in group:
            assert isclose(int(r["usable_narrative"]) / int(r["public_records"]), float(r["narrative_rate"]), abs_tol=1e-7)


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
    print("Historical outcomes, costs and disclosure funnel verified.")
