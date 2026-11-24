"""Training speed of Muon, Aurora and Sparrow (the paper's training-speed experiment): nanoGPT
Shakespeare-char with SwiGLU width m = 4d, one pass over the training split, unconstrained
updates (runs/speed, commands in REPRODUCE.md section 3). Muon at its best step size 0.01, Sparrow at
0.01 (Muon's) and 0.02, and Aurora (Algorithm 3 of Dewulf et al. 2026) at the same two (0.02 is its best
of {0.01, 0.02, 0.04}; printed from each step size's mean final loss minus Muon's on its own seeds).
Only seeds 0-9 count, also where more seeds were run.

    python scripts/plot_speed.py figures/speed.pdf --paper

Plots the validation loss minus Muon's at the same step and seed (seeds share the
initialization and the data order), mean over seeds with a band of one standard error of the
paired difference. Prints, per method,
the final validation loss, its paired difference to Muon, and the step at which it first reaches
Muon's final loss on the same seed (linear interpolation between evaluations). Writes the
validation curves to runs/speed/results.jsonl. The second panel is the time of one direction for a
512 x 128 SwiGLU gate/value momentum, from runs/speed/cost.json (scripts/time_directions.py).
"""
import glob, json, os, sys, statistics as st
# colors: Okabe-Ito colorblind-safe palette
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
paper = "--paper" in sys.argv
dst = args[0] if args else sys.exit(__doc__)
ROOT, LR = "runs/speed", "0.01"
SEEDS = range(10)  # the comparison uses seeds 0-9
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
# (run key, step size, optimizer, color, marker, line style); Aurora's step size is filled in below
DASH = (0, (2.2, 1.4))  # Muon's step size: short dashes, so they also show in the legend
METHODS = [("spectral", "0.01", "Muon", "#D55E00", "D", "-"), ("aurora", "0.01", "Aurora", "#CC79A7", "^", DASH),
           ("aurora", "0.02", "Aurora", "#CC79A7", "^", "-"),
           ("sparrow", "0.01", "Sparrow", "#0072B2", "o", DASH), ("sparrow", "0.02", "Sparrow", "#0072B2", "o", "-")]


def load(name):
    e = [json.loads(l) for l in open(f"{ROOT}/{name}/log.jsonl")]
    e = [x for x in e if x.get("event") == "eval"]
    return np.array([x["step"] for x in e]), np.array([x["val_loss"] for x in e])


def runs(key, lr=LR):
    out = {}
    for d in glob.glob(f"{ROOT}/{key}-lr{lr}") + glob.glob(f"{ROOT}/{key}-lr{lr}-s*"):
        if os.path.exists(f"{d}/done"):
            name = os.path.basename(d)
            sd = int(name.rsplit("-s", 1)[1]) if "-s" in name else 0
            if sd in SEEDS:
                out[sd] = load(name)
    return out


def first(steps, val, t):
    i = np.where(val <= t)[0]
    if not len(i):
        return None
    i = i[0]
    if i == 0:
        return float(steps[0])
    return float(steps[i - 1] + (val[i - 1] - t) / (val[i - 1] - val[i]) * (steps[i] - steps[i - 1]))


cand = {lr: runs("aurora", lr) for lr in ("0.01", "0.02", "0.04")}
cand = {lr: v for lr, v in cand.items() if v}
mu01 = runs("spectral", "0.01")
gain = {lr: float(np.mean([v[sd][1][-1] - mu01[sd][1][-1] for sd in v])) for lr, v in cand.items()}  # paired, own seeds
aur = min(gain, key=gain.get)
print("Aurora step size: final loss minus Muon's, each on its own seeds:",
      {lr: (round(g, 4), len(cand[lr])) for lr, g in gain.items()}, "->", aur)
data = {(k, lr): runs(k, lr) for k, lr, *_ in METHODS}
seeds = sorted(set.intersection(*(set(v) for v in data.values())))
target = {sd: data["spectral", "0.01"][sd][1][-1] for sd in seeds}
base = [first(*data["spectral", "0.01"][sd], target[sd]) for sd in seeds]

print(f"{len(seeds)} seeds; Muon step-size check (final val loss by seed):")
for lr in ("0.005", "0.01", "0.02"):
    r = runs("spectral", lr)
    print(f"  eta = {lr:5s} " + "  ".join(f"s{sd}: {r[sd][1][-1]:.4f}" for sd in sorted(r)))
print(f"{'method':8s} {'final':>15s} {'- Muon (paired)':>18s} {'better':>7s} {'steps to Muon final':>20s}")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 1.7) if paper else (10, 2.9), facecolor=SURF,
                             layout="constrained", gridspec_kw=dict(width_ratios=[1.6, 1]))
for a in (ax, bx):
    a.set_facecolor(SURF); a.set_axisbelow(True)
    for s in a.spines.values(): s.set_color(AXIS)
    a.tick_params(colors=MUTED, labelsize=fs)
ax.grid(True, color=GRID, lw=0.6); bx.grid(True, axis="x", color=GRID, lw=0.6)
records = []
for j, (key, lr, opt, col, mk, ls) in enumerate(METHODS):
    lab = "Muon (best $\\eta$)" if key == "spectral" else \
        f"{opt} (Muon's $\\eta$)" if lr == "0.01" else f"{opt} (Muon's $\\eta\\times2$)"
    cs = [data[key, lr][sd] for sd in seeds]
    x = cs[0][0]
    Y = np.stack([v - data["spectral", "0.01"][sd][1] for sd, (_, v) in zip(seeds, cs)])
    se = Y.std(0, ddof=1) / np.sqrt(len(Y)) if key != "spectral" else 0 * Y[0]
    ax.fill_between(x, Y.mean(0) - se, Y.mean(0) + se, color=col, alpha=0.15, lw=0)
    ax.plot(x, Y.mean(0), color=col, lw=1.4, ls=ls, label=lab, zorder=3 if key != "spectral" else 2,
            marker=mk, ms=3.5 if paper else 5, mec=SURF, mew=0.6, markevery=(1 + j, 3))
    fin = [v[-1] for _, v in cs]
    diff = [f - target[sd] for f, sd in zip(fin, seeds)]
    hit = [first(s, v, target[sd]) for sd, (s, v) in zip(seeds, cs)]
    reach = f"{st.mean(hit):.0f} ({st.mean(base) / st.mean(hit):.2f}x)" if None not in hit else \
        f"{sum(h is not None for h in hit)}/{len(hit)} reach"
    print(f"{lab:8s} {st.mean(fin):.4f}±{st.stdev(fin):.4f} {st.mean(diff):+.4f}±{st.stdev(diff):.4f} "
          f"{sum(d < 0 for d in diff)}/{len(diff):>5} {reach:>20s}")
    records += [dict(method=opt, seed=sd, eta=float(lr), steps=s.tolist(), val_loss=v.round(4).tolist())
                for sd, (s, v) in zip(seeds, cs)]
T = int(x[-1])
ax.set_xlim(x[0], T)
ax.set_xlabel("step", color=MUTED, fontsize=fs)
ax.set_ylabel("val. loss $-$ Muon", color=MUTED, fontsize=fs)
hl = list(zip(*ax.get_legend_handles_labels()))
hl = hl[1:] + hl[:1]  # Muon last
fig.legend(*zip(*hl), frameon=False, fontsize=fs - 1, labelcolor=INK, ncol=3,
           loc="outside upper center", handlelength=3.2, columnspacing=0.8, handletextpad=0.4)
ax.set_title("(a) validation loss relative to Muon (same seed)", fontsize=fs + 0.5, color=INK, loc="left")

# (b) time per training step
cost = json.load(open("runs/speed/cost.json"))
opts = list(dict.fromkeys((opt, col) for _, _, opt, col, *_ in METHODS))[::-1]
labs = [o for o, _ in opts]
vals = [cost[lab]["direction_ms"] for lab in labs]
bx.barh(labs, vals, color=[c for _, c in opts], height=0.6)
for i, v in enumerate(vals):
    bx.text(v + 0.1, i, f"{v:.1f} ms", va="center", fontsize=fs, color=INK)
bx.set_xlim(0, max(vals) * 1.3)
bx.set_xlabel("ms per direction ($512\\times128$)", color=MUTED, fontsize=fs)
bx.tick_params(axis="y", colors=INK)
bx.set_title("(b) cost of the LMO", fontsize=fs + 0.5, color=INK, loc="left")
fig.savefig(dst, dpi=200, facecolor=SURF)
with open("runs/speed/results.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in records)
print("wrote", dst, "and runs/speed/results.jsonl")
