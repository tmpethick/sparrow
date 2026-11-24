"""Looped vs plain models in the drop-in setting (runs/loopbs/*.sh): final validation loss per (architecture,
optimizer, step), mean +- s.e. over seeds, each optimizer's best step per architecture, and Sparrow - Muon at
those steps paired by seed; for looped models also the final model's loss at other iteration counts.

    python scripts/table_loopbs.py 'runs/loopbs/gate-tall-*'
"""
import glob, json, re, sys
from collections import defaultdict
import numpy as np

R, L = defaultdict(dict), defaultdict(dict)
for d in sorted(glob.glob(sys.argv[1])):
    m = re.search(r"-(spectral|sparrow|muon)-(\w+)-lr([\d.]+)-s(\d+)$", d)
    try:
        recs = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    except FileNotFoundError:
        continue
    if not m:
        continue
    ev = [r for r in recs if r["event"] == "eval"]
    cfg = next(r for r in recs if r["event"] == "config")
    if not ev or ev[-1]["step"] < cfg["steps"]:
        continue
    o = "muon" if m[1] in ("spectral", "muon") else "sparrow"
    R[m[2], o, float(m[3])][int(m[4])] = ev[-1]["val_loss"]
    le = next((r for r in recs if r["event"] == "loop_eval"), None)
    if le:
        L[m[2], o, float(m[3])][int(m[4])] = le["val_loss"]

se = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
print("| arch | optimizer | step | seeds | final val loss | per seed | val loss at K = 1 / 2 / 4 / 8 / 16 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for k, v in sorted(R.items()):
    a = np.array([v[s] for s in sorted(v)])
    lk = ""
    if k in L:
        Ks = sorted(next(iter(L[k].values())), key=int)
        lk = " / ".join(f"{np.mean([L[k][s][K] for s in L[k]]):.3f}" for K in Ks)
    print(f"| {k[0]} | {k[1]} | {k[2]} | {len(a)} | {a.mean():.4f} +- {se(a):.4f} | {np.round(a, 4).tolist()} | {lk} |")
print("\nAt each optimizer's best step per architecture (mean final loss), Sparrow - Muon paired by seed:")
for arch in sorted({k[0] for k in R}):
    best = {}
    for (a_, o, lr), v in R.items():
        if a_ == arch and len(v) >= 2 and (o not in best or np.mean(list(v.values())) < np.mean(list(best[o][1].values()))):
            best[o] = (lr, v)
    if len(best) == 2:
        s = sorted(set(best["muon"][1]) & set(best["sparrow"][1]))
        d = np.array([best["sparrow"][1][i] - best["muon"][1][i] for i in s])
        print(f"  {arch}: Muon {best['muon'][0]} {np.mean(list(best['muon'][1].values())):.4f}, Sparrow {best['sparrow'][0]} "
              f"{np.mean(list(best['sparrow'][1].values())):.4f}, difference {d.mean():+.4f} +- {se(d):.4f} {np.round(d, 4).tolist()}")
