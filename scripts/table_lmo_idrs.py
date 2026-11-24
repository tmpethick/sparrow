"""Rows of the paper's joint-LMO solver table from a run of experiments/solvers.py (results.csv): per (decay nu,
s/r), the median over seeds of the solve time of iDRS (with its final rank), exact DRS and fixed-rank DRS, or
"stall (g)" when a fixed rank fails to certify on some seed (g: its largest final relative gap), and the relative
value lost by the best composed candidate (best_of_three_gap, or best_of_two_gap in runs that predate it).

    python scripts/table_lmo_idrs.py runs/lmo-rank-adaptive
"""
import csv, sys
from collections import defaultdict
import numpy as np

rows = list(csv.DictReader(open(f"{sys.argv[1]}/results.csv")))
gap_col = "best_of_three_gap" if "best_of_three_gap" in rows[0] else "best_of_two_gap"
cells, gaps = defaultdict(list), defaultdict(dict)
for r in rows:
    key = (float(r["decay"]), float(r["ratio"]))
    cells[key, r["method"]].append(r)
    gaps[key][r["seed"]] = float(r[gap_col])
methods = sorted({m for _, m in cells}, key=lambda m: (m != "idrs", m != "exact", int(m.split("_r")[-1]) if "_r" in m else 0))
print(f"% {gap_col}; methods: {', '.join(methods)}")
for key in sorted(gaps):
    out = [f"{key[0]:g}", f"{key[1]:g}", f"{100 * np.median(list(gaps[key].values())):.0f}\\%"]
    for m in methods:
        c = cells[key, m]
        if all(x["converged"] == "True" for x in c):
            t = f"{np.median([float(x['seconds']) for x in c]):.2f}"
            out.append(t + (f" ({np.median([float(x['rank']) for x in c]):.0f})" if m == "idrs" else ""))
        else:
            out.append(f"stall ({max(float(x['gap']) for x in c):.0e})".replace("e-0", "e-"))
    print(" & ".join(out) + r" \\")
