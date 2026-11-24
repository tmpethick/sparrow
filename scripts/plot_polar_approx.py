"""Robustness of Muon, Aurora and Sparrow to an inexact polar factor (runs/polar-approx, launched by
runs/polar-approx/sweep.sh): eta = 0.02 for all three on Tiny Shakespeare (one epoch, the training-speed setup), with
every polar factor (Muon's Spectral_s, Sparrow's, and both of Aurora's) computed by k steps of Polar Express
(sparrow_jax.polar_factor) or by the exact SVD.

    python scripts/plot_polar_approx.py figures/polar_approx.pdf --paper

Plots the final validation loss minus Muon's with the exact SVD on the same seed (seeds share the initialization
and the data order), mean over seeds with a bar of one standard error of the paired difference. Prints, per method
and k, the paired difference to Muon (SVD), to Muon with the same k, and to the method's own SVD run.
"""
import glob, json, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
paper = "--paper" in sys.argv
dst = args[0] if args else sys.exit(__doc__)
ROOT = "runs/polar-approx"
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
POLARS = ["pe2", "pe3", "pe4", "pe5", "pe6", "pe8", "svd"]
XS = [2, 3, 4, 5, 6, 8, 10]  # the SVD is drawn at 10, past a dotted break
METHODS = [("muon", "Muon", "#D55E00", "D"), ("aurora", "Aurora", "#CC79A7", "^"), ("sparrow", "Sparrow", "#0072B2", "o")]

val = {}
for d in glob.glob(f"{ROOT}/*-s[0-9]"):
    if os.path.exists(f"{d}/done"):
        m, p, s = re.match(r"(\w+)-(\w+)-s(\d+)$", os.path.basename(d)).groups()
        ev = [json.loads(l) for l in open(f"{d}/log.jsonl") if '"eval"' in l]
        val[m, p, int(s)] = ev[-1]["val_loss"]


def paired(a, b):
    """Mean and standard error over seeds of val[a] - val[b], and the number of seeds."""
    seeds = sorted(s for (m, p, s) in val if (m, p) == a and (*b, s) in val)
    d = np.array([val[(*a, s)] - val[(*b, s)] for s in seeds])
    return d.mean(), (d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0), len(d)


fig, ax = plt.subplots(figsize=(3.6, 2.0) if paper else (6, 3.4), facecolor=SURF, layout="constrained")
ax.set_facecolor(SURF)
ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
ax.axvline(9, color=GRID, lw=0.8, ls=(0, (2, 2)), zorder=1)
n_seeds = set()
for i, (m, label, col, mk) in enumerate(METHODS):
    pts = [(x, *paired((m, p), ("muon", "svd"))) for x, p in zip(XS, POLARS) if (m, p, 0) in val]
    x, y, e, n = map(np.array, zip(*pts))
    n_seeds.update(n.tolist())
    x = x + (i - 1) * 0.12  # dodge so the error bars do not overlap
    k = x < 9
    ax.plot(x[k], y[k], color=col, lw=1.3, zorder=3, label=label)
    ax.errorbar(x, y, yerr=e, fmt=mk, color=col, ms=3.5 if paper else 5, mec=SURF, mew=0.6, elinewidth=1.0,
                capsize=0, zorder=4)
ax.set_xticks(XS, ["2", "3", "4", "5", "6", "8", "SVD"])
ax.set_xlabel("Polar Express iterations", color=INK)
ax.set_ylabel("Val. loss $-$ Muon (SVD)", color=INK)
ax.tick_params(colors=MUTED, labelcolor=INK)
ax.grid(axis="y", color=GRID, lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(AXIS)
ax.legend(frameon=False, loc="upper right", handlelength=1.6)
fig.savefig(dst, facecolor=SURF, **({} if dst.endswith(".pdf") else {"dpi": 200}))
print(dst, f"{len(val)} runs, seeds per point {sorted(n_seeds)}")

f = lambda r: f"{r[0]:+.3f}±{r[1]:.3f}"
print(f"{'':8} {'k':4} {'- Muon(SVD)':>14} {'- Muon(k)':>14} {'- own SVD':>14}")
for m, label, *_ in METHODS:
    for p in POLARS:
        if (m, p, 0) in val:
            print(f"{label:8} {p:4} {f(paired((m, p), ('muon', 'svd'))):>14} {f(paired((m, p), ('muon', p))):>14} "
                  f"{f(paired((m, p), (m, 'svd'))):>14}")
