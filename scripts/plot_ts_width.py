"""Width at a fixed certificate for the teacher-student classification runs (experiments/synthetic.py --widths).

    python scripts/plot_ts_width.py runs figures/ts_width_cls.pdf [--paper]

Teacher gauss_16 (m* = 16d); students of width m in the spectral ball (Muon) or the joint set
with kappa = sqrt(d / m) (ours). (a-c) clean accuracy against m at a fixed budget beta, best step
size per seed; (d) the frontier R(0.70) over these budgets and all step sizes. Mean over seeds, with a
bar of one standard deviation. Prints the numbers.
"""
import glob, json, os, sys
import numpy as np
# colors: Okabe-Ito colorblind-safe palette
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import matplotlib.lines

args = [a for a in sys.argv[1:] if not a.startswith("--")]
paper = "--paper" in sys.argv
root = args[0] if args else "runs"
dst = args[1] if len(args) > 1 else sys.exit(__doc__)
R = [json.loads(l) for f in sorted(glob.glob(os.path.join(root, "tsr-width-s*", "results.jsonl")))
     for l in open(f)]

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
FAM = [("muon_spec", "Muon + spectral", "#D55E00", "D"), ("ours_rowspec", "Ours + row/spectral", "#0072B2", "o")]
BETAS = [(1.0, r"(a) $\zeta = 1$ (binding)"), (4.0, r"(b) $\zeta = 4$"), (16.0, r"(c) $\zeta = 16$ (loose)")]
R = [r for r in R if r["beta"] in {b for b, _ in BETAS}]  # the zeta = 1/2 runs are not shown
LVL = 0.70
fs = 7.5 if paper else 9
lw, ms = (1.6, 4.5) if paper else (2, 7)
if paper:
    paper_rc(fs)
d = 64
widths = sorted({r["m"] // d for r in R})
seeds = sorted({r["seed"] for r in R})


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=fs)
    ax.set_xscale("log", base=2); ax.set_xticks(widths)
    ax.set_xticklabels(["$d$" if w == 1 else f"${w}d$" for w in widths])
    ax.minorticks_off()
    ax.set_xlabel("student width $m$", color=MUTED, fontsize=fs)


def series(fam, value):
    """Mean and std over seeds of value(runs of one seed and width), per width."""
    out = []
    for w in widths:
        v = [value([r for r in R if r["set"] == fam and r["m"] == w * d and r["seed"] == s]) for s in seeds]
        out.append((np.mean(v), np.std(v)))
    return np.array(out)


def plot(ax, fam, col, mk, ys):
    ok = np.isfinite(ys[:, 0])
    x = np.array(widths)[ok]
    ax.errorbar(x, ys[ok, 0], yerr=ys[ok, 1], color=col, lw=lw, marker=mk, ms=ms, mec=SURF, mew=0.8,
                elinewidth=0.8, capsize=0, zorder=3)


fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.1) if paper else (13, 3.7), facecolor=SURF)
for ax, (beta, title) in zip(axes[:3], BETAS):
    for fam, lab, col, mk in FAM:
        ys = series(fam, lambda rs: max(r["acc"] for r in rs if r["beta"] == beta))
        plot(ax, fam, col, mk, ys)
        print(f"beta={beta:<4} {lab:20s} " + " ".join(f"{m:.3f}±{s:.3f}" for m, s in ys))
    ax.set_title(title, loc="left", fontsize=fs + 0.5, color=INK)
    style(ax)
lo = min(a.get_ylim()[0] for a in axes[:3]); hi = max(a.get_ylim()[1] for a in axes[:3])
for a in axes[:3]:
    a.set_ylim(lo, hi)
for a in axes[1:3]:
    a.tick_params(labelleft=False)
axes[0].set_ylabel("clean accuracy", color=MUTED, fontsize=fs)

ax = axes[3]
for fam, lab, col, mk in FAM:
    ys = series(fam, lambda rs: 1e3 * max([r["radius_median"] for r in rs if r["acc"] >= LVL], default=np.nan))
    plot(ax, fam, col, mk, ys)
    print(f"R({LVL})     {lab:20s} " + " ".join(f"{m:.2f}±{s:.2f}" for m, s in ys))
ax.set_ylim(0, None)
ax.set_title(f"(d) frontier $R({LVL:.2f})$", loc="left", fontsize=fs + 0.5, color=INK)
ax.set_ylabel(r"median certified radius ($\times 10^{-3}$)", color=MUTED, fontsize=fs)
style(ax)

handles = [matplotlib.lines.Line2D([], [], color=col, lw=lw, marker=mk, ms=ms, mec=SURF, mew=0.8, label=lab)
           for _, lab, col, mk in FAM]
fig.legend(handles=handles, frameon=False, fontsize=fs, loc="upper center", ncol=len(FAM), labelcolor=INK,
           handlelength=2, columnspacing=2.5, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(pad=0.4 if paper else 1.0, w_pad=0.6, rect=(0, 0, 1, 0.9))
fig.savefig(dst, dpi=160, facecolor=SURF)
print(dst)
