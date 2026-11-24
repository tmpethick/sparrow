"""Cost of Muon, Sparrow and Aurora (the paper's training-speed experiment), d = 128, m = 4d, s = 1,
r = sqrt(d / m). Run on an otherwise idle machine.

    python scripts/time_directions.py            # writes runs/speed/cost.json

direction  mean wall-clock time of one compiled direction for an m x d matrix (200 calls)
step       mean wall-clock time of one training step of the runs in runs/speed: short runs of
           experiments/nanogpt.py, one method after the other, timed between steps 200 and 600 with a
           single evaluation batch (median of three repeats)
"""
import json, math, statistics as st, subprocess, sys, tempfile, time

import jax

from sparrow.approx_oracles import aurora_update, row_max, sigma_max, sparrow, spectral

d, m = 128, 512
r = math.sqrt(d / m)
METHODS = {"Muon": ("muon", lambda G: spectral(G, 1.0)),       # name: (--method, direction of a gate/value matrix)
           "Sparrow": ("sparrow", lambda G: sparrow(G, r, 1.0)),
           "Aurora": ("aurora", lambda G: aurora_update(G, r, 1.0))}

out = {}
G = jax.random.normal(jax.random.PRNGKey(0), (m, d))
for name, (_, direction) in METHODS.items():
    f = jax.jit(direction)
    f(G).block_until_ready()
    t = time.perf_counter()
    for _ in range(200):
        W = f(G)
    W.block_until_ready()
    ms = (time.perf_counter() - t) / 200 * 1e3
    out[name] = dict(direction_ms=round(ms, 2))
    print(f"{name:8s} direction {ms:6.2f} ms   ||W||_op = {float(sigma_max(W)):.4f}   "
          f"max row = {float(row_max(W)):.4f} (r = {r:.4f})", flush=True)


def step_ms(flags):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "experiments/nanogpt.py", "--out", tmp,
                        "--update", "unconstrained", "--lmo_lr", "0.01", "--steps", "600",
                        "--eval_every", "200", "--eval_batches", "1", *flags],
                       check=True, capture_output=True)
        e = {x["step"]: x["sec"] for x in map(json.loads, open(f"{tmp}/log.jsonl"))
             if x.get("event") == "eval"}
    return (e[600] - e[200]) / 400 * 1e3


for rep in range(3):
    for name, (method, _) in METHODS.items():
        out[name].setdefault("step_runs_ms", []).append(round(step_ms(["--method", method]), 2))
for name in METHODS:
    out[name]["step_ms"] = st.median(out[name]["step_runs_ms"])
    print(f"{name:8s} step {out[name]['step_ms']:6.2f} ms   runs {out[name]['step_runs_ms']}")
json.dump(out, open("runs/speed/cost.json", "w"), indent=1)
print("wrote runs/speed/cost.json")
