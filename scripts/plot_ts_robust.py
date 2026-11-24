"""Robustness frontiers R_M(a) for the teacher-student classification runs (experiments/synthetic.py).

    python scripts/plot_ts_robust.py runs figures/ts_robust.pdf [--paper]

(a) spread teacher, (b) concentrated teacher: every run as a faint point (clean accuracy,
median certified radius) and the frontier R_M(a) = max radius over runs with accuracy >= a,
pooled over student seeds. (c) the split kappa of the joint set on the spread teacher:
R(0.70) against the student's kappa (seed 0); kappa = 1 is the spectral ball (Muon).
"""
import json, os, sys
import numpy as np
# colors: Okabe-Ito colorblind-safe palette
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import matplotlib.ticker
import matplotlib.lines

args = [a for a in sys.argv[1:] if not a.startswith("--")]
paper = "--paper" in sys.argv
root = args[0] if args else "runs"
dst = args[1] if len(args) > 1 else sys.exit(__doc__)
load = lambda d: [json.loads(l) for l in open(os.path.join(root, d, "results.jsonl"))]
R = load("tsr-fam")
K = load("tsr-kappa")

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
FAM = [("sgd_frob", "SGD + Frobenius", "#E69F00", "v"), ("adam_linf", r"Adam + $\ell_\infty$", "#009E73", "s"),
       ("muon_spec", "Muon + spectral", "#D55E00", "D"), ("ours_rowspec", "Ours + row/spectral", "#0072B2", "o")]
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=fs)


def frontier_panel(ax, teacher, title, xmin):
    for fam, lab, col, mk in FAM:
        rs = [r for r in R if r["teacher"] == teacher and r["set"] == fam]
        acc = np.array([r["acc"] for r in rs]); rad = np.array([r["radius_median"] for r in rs]) * 1e3
        ax.scatter(acc, rad, s=7 if paper else 14, color=col, alpha=0.25, marker=mk, lw=0, zorder=2)
        grid = np.linspace(xmin, acc.max(), 200)
        env = [rad[acc >= a].max() for a in grid]
        ax.plot(grid, env, color=col, lw=1.6 if paper else 2, zorder=3)
        ax.plot(acc.max(), rad[acc >= acc.max()].max(), marker=mk, color=col, ms=5 if paper else 7,
                mec=SURF, mew=0.8, zorder=4)
    ax.set_xlim(xmin, None); ax.set_ylim(0, None)
    ax.set_title(title, loc="left", fontsize=fs + 0.5, color=INK)
    ax.set_xlabel("clean accuracy", color=MUTED, fontsize=fs)
    style(ax)


fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.1) if paper else (13, 4.2), facecolor=SURF,
                         gridspec_kw=dict(width_ratios=[1.15, 1.15, 0.9]))
frontier_panel(axes[0], "gauss_4", r"(a) spread teacher ($\kappa^\star = 0.43$)", 0.45)
frontier_panel(axes[1], "few_8", r"(b) concentrated teacher ($\kappa^\star = 0.90$)", 0.60)
axes[0].set_ylabel(r"median certified radius ($\times 10^{-3}$)", color=MUTED, fontsize=fs)
handles = [matplotlib.lines.Line2D([], [], color=col, lw=1.6 if paper else 2, marker=mk, ms=5 if paper else 7,
                                   mec=SURF, mew=0.8, label=lab) for _, lab, col, mk in FAM]
axes[0].legend(handles=handles, frameon=False, fontsize=fs - 0.5, loc="lower left", labelcolor=INK, handlelength=1.5)

# (c) split sweep on the spread teacher, seed 0: R(0.70)
ax = axes[2]
lvl = 0.70
pool = [r for r in K + R if r["teacher"] == "gauss_4" and r["seed"] == 0 and r.get("steps", 1000) == 1000]
def R_at(sel):
    return 1e3 * max([r["radius_median"] for r in pool if sel(r) and r["acc"] >= lvl], default=np.nan)
kap_star = 0.4254
pts = [(0.21, R_at(lambda r: r.get("kappa_student") == 0.21)),
       (kap_star, R_at(lambda r: r["set"] == "ours_rowspec" and not r.get("kappa_student"))),
       (0.5, R_at(lambda r: r.get("kappa_student") == 0.5)),
       (0.85, R_at(lambda r: r.get("kappa_student") == 0.85)),
       (1.0, R_at(lambda r: r["set"] == "muon_spec"))]
xs, ys = zip(*pts)
ax.plot(xs[:-1], ys[:-1], color="#0072B2", lw=1.6 if paper else 2, marker="o", ms=4.5 if paper else 7, mec=SURF, zorder=3)
ax.plot([xs[-2], xs[-1]], [ys[-2], ys[-1]], color=AXIS, lw=1, ls=(0, (3, 2)), zorder=2)
ax.plot(xs[-1], ys[-1], marker="D", color="#D55E00", ms=4.5 if paper else 7, mec=SURF, zorder=4)
ax.annotate("Muon", (1.0, ys[-1]), xytext=(-4, -10), textcoords="offset points",
            ha="right", fontsize=fs - 1, color=MUTED)
for x, lab, ha in ((kap_star, r"$\kappa^\star$ ", "right"), (0.5, r" $\sqrt{d/m}$", "left")):
    ax.axvline(x, color=AXIS, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(x, 0.03, lab, transform=ax.get_xaxis_transform(), ha=ha, fontsize=fs - 1, color=MUTED)
for fam, lab, col, mk in FAM[:2]:
    v = 1e3 * max([r["radius_median"] for r in pool if r["set"] == fam and r["acc"] >= lvl], default=np.nan)
    if np.isfinite(v):
        ax.axhline(v, color=col, lw=1, ls=(0, (4, 2)), zorder=1)
        ax.text(0.98, v, lab + " ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=fs - 1, color=MUTED)
ax.set_xscale("log"); ax.set_xticks([0.2, 0.3, 0.5, 1.0]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_ylim(0, None)
ax.set_xlabel(r"split $\kappa = r/s$ of the joint set", color=MUTED, fontsize=fs)
ax.set_title(f"(c) choosing $\\kappa$: $R({lvl:.2f})$", loc="left", fontsize=fs + 0.5, color=INK)
style(ax)
fig.tight_layout(pad=0.4 if paper else 1.0, w_pad=0.8)
fig.savefig(dst, dpi=160, facecolor=SURF)
print(dst, [f"{x:.2f}:{y:.2f}" for x, y in pts])
