"""Check the reference evidence offline. Optional checks exercise policy and inference code."""

from pathlib import Path
import csv
import hashlib
import json
from math import isclose
import sys
from tools.verify_legacy import verify as verify_legacy

ROOT = Path(__file__).resolve().parent
P = ROOT / "experiments/policy"
EXT = ROOT / "experiments/external/results"
j = lambda p: json.loads(p.read_text())


def table(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def verify_policy():
    spec = j(P / "locked_policy.json")
    assert (ROOT / "demo/guard.json").read_bytes() == (
        P / "locked_policy.json"
    ).read_bytes()
    assert not any(
        x["enabled"] for x in j(P / "june_midmonth_policy.json")["certificates"]
    )
    assert {c["queue"] for c in spec["certificates"] if c["enabled"]} == {
        "Cards",
        "Mortgages",
    }
    assert sum(spec["counts"].values()) == j(P / "june_cohort.json")["retained"]
    man = j(EXT / "july_manifest.json")
    assert (
        man["eligible_published_text_min10words"] - sum(man["removed"].values())
        == man["retained"]
    )
    for period in ("june_late", "july"):
        rows = j(P / f"{period}_metrics.json")["rows"]
        queues = table(P / f"{period}_queues.csv")
        cm = table(P / f"{period}_confusion.csv")
        for k in rows:
            n, auto, wrong = k["n"], k["auto"], k["wrong"]
            q = [r for r in queues if r["policy"] == k["policy"]]
            matrix = [r for r in cm if r["policy"] == k["policy"]]
            assert (
                len(q) == 7
                and len(matrix) == 49
                and n == sum(int(r["n_actual"]) for r in q)
            )
            assert (
                auto
                == sum(int(r["auto"]) for r in q)
                == sum(int(r["n"]) for r in matrix)
            )
            assert (
                wrong
                == sum(int(r["wrong"]) for r in q)
                == sum(int(r["n"]) for r in matrix if r["truth"] != r["destination"])
            )
            assert (
                int(next(r["auto"] for r in q if r["destination"] == "Other / review"))
                == 0
            )
            assert (
                n == auto + k["manual"]
                and wrong + k["correct_auto"] == auto
                and isclose(auto / n, k["coverage"])
            )
            assert (
                isclose(wrong / auto, k["risk"])
                if auto
                else (k["risk"] is None and k["risk_ci"] == [None, None])
            )
    g = next(r for r in rows if r["policy"] == "certified_destinations")
    assert g["n"] == man["retained"] == 8830 and (g["auto"], g["wrong"]) == (928, 17)
    inst = j(EXT / "july_institution_bootstrap.json")
    assert (
        inst["locked_policy_sha256"]
        == hashlib.sha256((P / "locked_policy.json").read_bytes()).hexdigest()
    )
    assert (
        inst["all_July_n"],
        inst["all_July_autorouted"],
        inst["all_July_wrong"],
    ) == (g["n"], g["auto"], g["wrong"])
    q = [
        r
        for r in queues
        if r["policy"] == "certified_destinations" and int(r["auto"]) > 0
    ]
    assert {r["destination"] for r in q} == {
        x["queue"] for x in spec["certificates"] if x["enabled"]
    }
    short = next(
        r for r in table(EXT / "july_audit_slices.csv") if r["subgroup"] == "10–49"
    )
    assert (int(short["auto"]), int(short["wrong"])) == (29, 5)
    for r in table(ROOT / "results/pilot_economics.csv"):
        n, a, w = map(int, [r["n"], r["routed"], r["wrong"]])
        assert (n, a, w) == (g["n"], g["auto"], g["wrong"])
        benefit = (
            a
            * float(r["review_minutes"])
            * float(r["wage"])
            / 60
            * (1 - float(r["audit_rate"]))
        )
        net = (benefit - w * float(r["wrong_route_cost"])) * 1e4 / n
        assert isclose(float(r["net_per_10000"]), net, abs_tol=1e-8)
    for r in table(ROOT / "experiments/deep/july_challengers.csv"):
        n, a, w = map(int, [r["n"], r["auto"], r["wrong"]])
        assert (
            n == man["retained"]
            and isclose(a / n, float(r["coverage"]))
            and isclose(w / a, float(r["misroute"]))
        )


def verify_desk():
    sys.path.insert(0, str(ROOT / "demo"))
    from desk import Desk

    desk = Desk(ROOT, ROOT / "demo/guard.json")
    cases = {s["id"]: s for s in desk.samples}
    for ident, out in [
        ("S-101", "Cards"),
        ("S-102", "Human review"),
        ("S-103", "Mortgages"),
        ("S-106", "Human review"),
    ]:
        assert desk.route(cases[ident])["decision"] == out
    assert desk.route({"text": "Please help."})["state"] == "review"
    assert desk.route(dict(cases["S-101"], company=""))["state"] == "review"


def verify_model_comparison():
    folder = ROOT / "experiments/deep"
    metrics = {(r["model"], r["period"]): r for r in table(folder / "benchmarks.csv")}
    recalls = table(folder / "recall_by_queue.csv")
    for key, r in metrics.items():
        group = [v for v in recalls if (v["model"], v["period"]) == key]
        assert len(group) == 7
        n = int(r["n"])
        assert sum(int(v["n"]) for v in group) == n
        assert isclose(sum(int(v["n"]) * float(v["recall"]) for v in group) / n, float(r["accuracy"]))
        assert isclose(int(r["auto"]) / n, float(r["coverage"]))
        if int(r["auto"]):
            assert isclose(int(r["wrong"]) / int(r["auto"]), float(r["misroute"]))
    for name, interval in j(folder / "paired_differences.json").items():
        period, models = name.split(": ")
        a, b = models.split(" minus ")
        difference = float(metrics[a, period]["macro_f1"]) - float(metrics[b, period]["macro_f1"])
        assert isclose(difference, interval["delta_macro_f1"], abs_tol=1e-12)
    original_test = next(v for v in table(ROOT / "results/benchmarks.csv") if v["model"] == "TF-IDF" and v["split"] == "test")
    scaled = next(v for v in table(ROOT / "experiments/scale/results/learning.csv") if v["test"] == "original historical test")
    assert isclose(float(original_test["macro_f1"]), float(scaled["old_f1"]))


def verify_expanded_may():
    folder = ROOT / "experiments/scale/results"
    gates = table(folder / "all_gates.csv")
    ranks = table(folder / "all_rank.csv")
    by_product = table(folder / "all_per_product.csv")
    contrasts = table(folder / "all_paired_shortlist.csv")
    base, guarded = [next(r for r in gates if r["policy"] == name and r["group"] == "All")
                     for name in ["June rule", "June rule + length hold"]]
    assert (int(base["group_n"]), int(base["routed"]), int(base["wrong"])) == (14995, 1423, 29)
    assert (int(guarded["group_n"]), int(guarded["routed"]), int(guarded["wrong"])) == (14995, 1380, 27)
    held = int(guarded["group_n"]) - int(guarded["routed"])
    top = {r["method"]: r for r in ranks if r["k"] == "2"}
    for method, r in top.items():
        fields = [x for x in by_product if x["method"] == method]
        assert len(fields) == 7
        assert sum(int(x["n"]) for x in fields) == int(r["n"]) == held
        hits = sum(int(x["correct_in_top2"]) for x in fields)
        assert hits == int(r["correct_in_list"]) and isclose(hits / held, float(r["rate"]))
    for r in contrasts:
        gain = int(top["Model"]["correct_in_list"]) - int(top[r["versus"]]["correct_in_list"])
        assert gain == int(r["model_only"]) - int(r["baseline_only"])
        assert isclose(gain / held, float(r["paired_lift"]))


def main():
    import argparse
    if not __debug__:
        raise SystemExit("Run without python -O: the evidence checks use Python assertions.")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--numerical", action="store_true", help="also exercise the live policy and pilot design on synthetic inputs; requires NumPy, pandas and SciPy")
    p.add_argument("--inference", action="store_true", help="also exercise the local desk; requires a fitted model checkout and the research dependencies")
    args = p.parse_args()
    verify_legacy(ROOT)
    verify_policy()
    from tools.verify_extension import verify as verify_extension
    verify_extension(ROOT, trial=args.numerical or args.inference)
    from tools.verify_scale import verify as verify_scale
    verify_scale(ROOT)
    verify_expanded_may()
    verify_model_comparison()
    if args.numerical or args.inference:
        from experiments.policy.shift_policy import apply
        import numpy as np
        spec = j(P / "locked_policy.json")
        probs = np.eye(7) * .9993 + .0001
        _, destinations, selected = apply(probs, spec)
        assert list(destinations) == list(range(7))
        assert list(selected) == [False, True, False, False, True, False, False]
    if args.inference:
        if not (ROOT / "models/linear.joblib").is_file():
            p.exit(2, "The public source excludes model weights. Refit before passing --inference.\n")
        verify_desk()
    print("PASS: historical benchmarks, July release counts and costs, locked policy, May/July shortlists, August availability and expanded-data tables.")
    print("These checks reconcile archived aggregate evidence; they do not refit models or measure field outcomes.")


if __name__ == "__main__":
    main()
