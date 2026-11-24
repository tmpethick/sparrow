"""Final val loss (mean +- sd over seeds) per method and polar solver of runs/polar-approx; gap to Muon with the
exact SVD, and the largest gate/value spectral norm at the end. python runs/polar-approx/summarize.py"""
import glob, json, os, re
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
P = ["svd", "pe2", "pe3", "pe4", "pe5", "pe6", "pe8"]
res = {}
for d in glob.glob(os.path.join(here, "*-s[0-9]")):
    m, p, s = re.match(r"(\w+)-(\w+)-s(\d)", os.path.basename(d)).groups()
    evs = [json.loads(l) for l in open(os.path.join(d, "log.jsonl")) if '"eval"' in l]
    nan = any('"nan"' in l for l in open(os.path.join(d, "log.jsonl")))
    if not os.path.exists(os.path.join(d, "done")) and not nan:
        continue
    last = evs[-1] if evs else None
    res.setdefault((m, p), []).append(dict(val=float("nan") if nan else last["val_loss"],
                                            spec=last["norms"]["w_gate"]["spec"] if last else float("nan")))
base = np.mean([r["val"] for r in res.get(("muon", "svd"), [])]) if ("muon", "svd") in res else float("nan")
print(f"{'method':8} {'polar':5} {'n':>2} {'val':>8} {'sd':>6} {'-muon(svd)':>10} {'||W_g||':>8}")
for m in ("sparrow", "aurora", "muon"):
    for p in P:
        rs = res.get((m, p))
        if not rs:
            continue
        v = np.array([r["val"] for r in rs])
        print(f"{m:8} {p:5} {len(v):2d} {v.mean():8.4f} {v.std(ddof=1) if len(v) > 1 else 0:6.4f} "
              f"{v.mean() - base:+10.4f} {np.mean([r['spec'] for r in rs]):8.3f}")
