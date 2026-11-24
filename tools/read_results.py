"""Report the public reference results using only the Python standard library."""
import csv
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
def rows(name):
    with (root / name).open() as stream:
        return list(csv.DictReader(stream))
print("JULY: simulated handoffs compared with the recorded public product")
for row in json.loads((root / "experiments/policy/july_metrics.json").read_text())["rows"]:
    n, k, e = row["n"], row["auto"], row["wrong"]
    rate = f"{e/k:.2%}" if k else "undefined"
    print(f"{row['policy']:28} {k:4}/{n} would route; {e:3} disagree ({rate})")
print("\nHISTORICAL: combined model comparison (test macro-F1)")
for row in rows("experiments/deep/benchmarks.csv"):
    if row["period"] == "test":
        print(f"{row['model']:28} {float(row['macro_f1']):.3f}")
print("\nJULY: top-two suggestions on cases retained under the short-text hold")
for row in rows("experiments/transfer/results/reviewer_shortlist.csv"):
    if row["period"] == "July" and row["k"] == "2":
        print(f"{row['method']:28} {row['correct_in_list']}/{row['n']} ({float(row['rate']):.1%})")

print("\nLARGER DATA: same text classifier specification, original vs expanded training")
for row in rows("experiments/scale/results/learning.csv"):
    print(f"{row['test']:28} n={int(row['n']):,}  {float(row['old_f1']):.3f} to {float(row['new_f1']):.3f}")

print("\nEXPANDED MAY: same June policy applied retrospectively")
for row in rows("experiments/scale/results/all_gates.csv"):
    if row["group"] == "All":
        print(f"{row['policy']:28} {row['routed']}/{row['group_n']} flagged; {row['wrong']} disagreements ({float(row['wrong_per_route']):.2%})")
for row in rows("experiments/scale/results/all_rank.csv"):
    if row["k"] == "2":
        print(f"Top two, {row['method']:19} {row['correct_in_list']}/{row['n']} held cases ({float(row['rate']):.1%})")
print("\nSCENARIO, not measured savings: July mix, per 10,000 eligible intakes, 10% re-review")
for row in rows("results/pilot_economics.csv"):
    if float(row["audit_rate"]) == .1:
        net = float(row["net_per_10000"])
        print(f"{row['scenario']:28} net {'-' if net < 0 else ''}${abs(net):,.0f}")
