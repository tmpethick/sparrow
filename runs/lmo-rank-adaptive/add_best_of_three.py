"""Adds the best_of_three_gap column (experiments/solvers.py's instance_summary, which postdates this run) to
results.csv: the same instances and the same tighter exact-SVD reference (tol 1e-7), exact DRS, which takes the
same iterates as the dual ADMM used when the run was made."""
import csv, sys
sys.path.insert(0, ".")
from experiments.solvers import instance_summary
from sparrow.iterative_solvers import low_rank_gradient, solve_idrs

path = "runs/lmo-rank-adaptive/results.csv"
rows = list(csv.DictReader(open(path)))
info = {}
for decay, seed, ratio in sorted({(float(r["decay"]), int(r["seed"]), float(r["ratio"])) for r in rows}):
    g = low_rank_gradient(1024, 256, decay, seed)
    ref = solve_idrs(g, 1 / ratio, 1.0, exact=True, tol=1e-7, max_iters=30000)
    assert ref["converged"]
    info[decay, seed, ratio] = instance_summary(g, 1 / ratio, 1.0, ref)
    print(decay, seed, ratio, {k: round(info[decay, seed, ratio][k], 4) for k in ("best_of_two_gap", "best_of_three_gap")}, flush=True)
for r in rows:
    i = info[float(r["decay"]), int(r["seed"]), float(r["ratio"])]
    assert abs(float(r["best_of_two_gap"]) - i["best_of_two_gap"]) < 1e-4, (r, i)  # same instances and reference
    r["best_of_three_gap"] = repr(float(i["best_of_three_gap"]))
fields = list(rows[0].keys())
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
