"""Plot the looped drop-in runs (runs/loopbs/gate_grid.sh, gate_pcN.sh, gate_steps.sh, gate_depth.sh): final validation
loss for Muon and Sparrow at each one's best step (mean +- s.e. over seeds, paired difference annotated) for
(a) 1 + 1 x N + 1 with full backprop, N = 1, 2, 4, 8, 16, and (b) N = 16 with gradients through the last H iterations.

    python scripts/plot_loopbs.py --paper figures/loop_shakespeare.pdf   # paper colors and sizes
"""
import glob, json, re, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import numpy as np

R = defaultdict(dict)
for d in sorted(glob.glob("runs/loopbs/gate-tall-*")):
    m = re.search(r"-(spectral|sparrow)-(\w+)-lr([\d.]+)-s(\d+)$", d)
    recs = [json.loads(l) for l in open(f"{d}/log.jsonl")]
    ev = [r for r in recs if r["event"] == "eval"]
    if not m or not ev or ev[-1]["step"] < next(r for r in recs if r["event"] == "config")["steps"]:
        continue
    R[m[2], "Muon" if m[1] == "spectral" else "Sparrow", float(m[3])][int(m[4])] = ev[-1]["val_loss"]

PCN = [("PC1", "1"), ("L1x2", "2"), ("PC4", "4"), ("PC8", "8"), ("PC16", "16")]
TB16 = [("PC16H4", "4"), ("PC16H8", "8"), ("PC16", "16")]
paper = "--paper" in sys.argv
dst = [a for a in sys.argv[1:] if not a.startswith("--")][0]
fs = 7.5 if paper else 9
OPT = {"Muon": "#D55E00", "Sparrow": "#0072B2"} if paper else {"Muon": "#2a78d6", "Sparrow": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURF = "white" if paper else "#fcfcfb"
best = {}
for a in {a for a, _, _ in R}:
    for o in OPT:
        cands = [(np.mean(list(v.values())), lr) for (a_, o_, lr), v in R.items()
                 if a_ == a and o_ == o and len(v) >= 3]  # seed-0-only bracket checks do not compete
        if cands:
            best[a, o] = min(cands)[1]

plt.rcParams.update({"font.size": fs, "pdf.fonttype": 42, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
if paper:
    paper_rc(fs)


def panel(ax, groups, title, xlabel=None):
    """Mean +- s.e. per optimizer at its best step, seeds as faint dots, paired Sparrow - Muon above."""
    groups = [(a, lab) for a, lab in groups if all((a, o) in best for o in OPT)]
    for j, o in enumerate(OPT):
        for i, (a, _) in enumerate(groups):
            v = R[a, o, best[a, o]]
            y = np.array([v[s] for s in sorted(v)])
            xi = i + (j - 0.5) * 0.24
            ax.scatter(np.full(len(y), xi), y, s=10, color=OPT[o], alpha=0.35, linewidths=0, zorder=2)
            ax.errorbar(xi, y.mean(), yerr=y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0, fmt="o", ms=6,
                        color=OPT[o], capsize=0, elinewidth=2, mec=SURF, mew=1.5, zorder=3,
                        label=o if i == 0 else None)
    for i, (a, _) in enumerate(groups):
        mu, sp = R[a, "Muon", best[a, "Muon"]], R[a, "Sparrow", best[a, "Sparrow"]]
        d = np.mean([sp[k] - mu[k] for k in sorted(set(mu) & set(sp))])
        top = max(max(mu.values()), max(sp.values()))
        ax.text(i, top + 0.004 * (ax.get_ylim()[1] - ax.get_ylim()[0]) / 0.13, f"${d:+.3f}$", ha="center", va="bottom",
                fontsize=fs - 1, color=MUTED)
    ax.set_xticks(np.arange(len(groups)), [lab for _, lab in groups], fontsize=fs)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    lo, hi = ax.get_ylim(); ax.set_ylim(top=hi + 0.08 * (hi - lo))
    ax.set_title(title, fontsize=fs, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)


fig = plt.figure(figsize=(6.5, 1.95) if paper else (9.5, 3.6), facecolor=SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.22)
ax = fig.add_subplot(gs[0])
panel(ax, PCN, "(a) full backprop", xlabel="iterations $N$")
ax.set_ylabel("final val. loss")
ax.legend(frameon=False, loc="lower left", fontsize=fs - 0.5)
panel(fig.add_subplot(gs[1]), TB16, "(b) $N=16$, truncated", xlabel="backprop window $H$")
fig.savefig(dst, dpi=200, bbox_inches="tight", facecolor=SURF)
print("wrote", dst)
