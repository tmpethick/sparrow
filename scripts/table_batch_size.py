"""Gap to Muon versus batch size for the no-repeat sweep on the Complete Works (runs/bs-complete).

    python scripts/table_batch_size.py [runs/bs-complete]

Runs are <opt>-b<batch>-s<seed> for opt in muon, aurora, sparrow; gaps are paired by seed. kappa is the
SwiGLU gate and value matrices' layer-mean max row norm over their layer-mean spectral norm.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "runs/bs-complete"
runs = defaultdict(dict)                   # (opt, batch) -> seed -> {step: record}
for log in glob.glob(os.path.join(root, "*", "log.jsonl")):
    m = re.fullmatch(r"(\w+)-b(\d+)-s(\d+)", os.path.basename(os.path.dirname(log)))
    if not m or not os.path.exists(os.path.join(os.path.dirname(log), "done")):
        continue
    recs = [json.loads(l) for l in open(log)]
    runs[m[1], int(m[2])][int(m[3])] = {r["step"]: r for r in recs if r.get("event") == "eval"}


def kappa(r):
    n = r["norms"]
    return np.mean([n[k]["row"] / n[k]["spec"] for k in ("w_gate", "w_in")])


batches = sorted({b for _, b in runs})
summary = []
for b in batches:
    muon = runs["muon", b]
    print(f"\n## batch {b} ({b * 64} characters per step)\n")
    print("| step | Muon val | Aurora − Muon | lower | Sparrow − Muon | lower | κ Muon / Aurora / Sparrow |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    steps = sorted(next(iter(muon.values())))
    for t in steps:
        if t % 100 and t != steps[-1] and t != 50:
            continue
        cells, ks = [], [np.mean([kappa(muon[s][t]) for s in muon])]
        for o in ("aurora", "sparrow"):
            seeds = sorted(set(muon) & set(runs[o, b]))
            d = np.array([runs[o, b][s][t]["val_loss"] - muon[s][t]["val_loss"] for s in seeds])
            cells += [f"{d.mean():+.4f} ± {d.std(ddof=1) / len(d) ** .5:.4f}" if len(d) > 1 else f"{d.mean():+.4f}",
                      f"{(d < 0).sum()}/{len(d)}"]
            ks.append(np.mean([kappa(runs[o, b][s][t]) for s in seeds]))
            if t == steps[-1]:
                summary.append((b, o, d.mean(), d.std(ddof=1) / len(d) ** .5 if len(d) > 1 else float("nan"),
                                (d < 0).sum(), len(d)))
        mv = np.mean([muon[s][t]["val_loss"] for s in muon])
        print(f"| {t} | {mv:.3f} | " + " | ".join(cells) + " | " + " / ".join(f"{k:.3f}" for k in ks) + " |")

print("\n## final gap to Muon\n")
print("| batch | Aurora − Muon | lower | Sparrow − Muon | lower |")
print("| --- | --- | --- | --- | --- |")
for b in batches:
    row = {o: (g, se, n, k) for bb, o, g, se, n, k in summary if bb == b}
    print(f"| {b} | " + " | ".join(f"{row[o][0]:+.4f} ± {row[o][1]:.4f} | {row[o][2]}/{row[o][3]}"
                                    for o in ("aurora", "sparrow")) + " |")
