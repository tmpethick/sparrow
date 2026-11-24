"""Robustness frontiers R_M(a) of the joint-set direction U_1 = RowClip_r(Spectral_s(M)) against
Sparrow's U_3 = RowClip_r(Spectral_s(Row_r(M))), with Muon for reference.

    python scripts/plot_ts_sparrow.py runs figures/ts_sparrow.pdf [--paper]

All three methods use the same budgets, step sizes and seeds, run with JAX 0.11.2 on macOS:
Muon and U_1 from runs/tsr-fam-local-* (eta 0.01, 0.03, 0.1) and runs/tsr-fam-ext-* (eta 0.001,
0.003), U_3 from runs/tsr-sp-* (--impl sparrow_jax). As in plot_ts_robust.py, every run is a faint
point (clean accuracy, median certified radius) and the line is the frontier R_M(a) = max radius
over runs with accuracy >= a, pooled over student seeds. Only runs at the teacher's split kappa*
(--kappa_student 0, the default) count. Prints R(0.70) (spread) and R(0.85)
(concentrated) and the maximal accuracy per method.
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
load = lambda pat: [r for p in sorted(glob.glob(os.path.join(root, pat, "results.jsonl")))
                    for r in map(json.loads, open(p)) if not r.get("kappa_student")]   # kappa* only
base = load("tsr-fam-local-*") + load("tsr-fam-ext-*")
sp = load("tsr-sp-*")

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
FAM = [("muon", "Muon + spectral", "#D55E00", "D"), ("w1", "$U_1$ + row/spectral", "#0072B2", "o"),
       ("w3", "$U_3$ (Sparrow) + row/spectral", "#009E73", "s")]
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)


def runs(teacher, fam):
    if fam == "w3":
        return [r for r in sp if r["teacher"] == teacher]
    key = "muon_spec" if fam == "muon" else "ours_rowspec"
    return [r for r in base if r["teacher"] == teacher and r["set"] == key]


def panel(ax, teacher, title, xmin, lvl):
    for fam, lab, col, mk in FAM:
        rs = runs(teacher, fam)
        acc = np.array([r["acc"] for r in rs]); rad = np.array([r["radius_median"] for r in rs]) * 1e3
        ax.scatter(acc, rad, s=7 if paper else 14, color=col, alpha=0.25, marker=mk, lw=0, zorder=2)
        grid = np.linspace(xmin, acc.max(), 200)
        ax.plot(grid, [rad[acc >= a].max() for a in grid], color=col, lw=1.6 if paper else 2, zorder=3)
        ax.plot(acc.max(), rad[acc >= acc.max()].max(), marker=mk, color=col, ms=5 if paper else 7,
                mec=SURF, mew=0.8, zorder=4)
        R = rad[acc >= lvl].max() if (acc >= lvl).any() else float("nan")
        print(f"{teacher:8s} {lab:18s} runs={len(rs):3d}  max acc {acc.max():.3f}  R({lvl:.2f}) {R:.2f}e-3")
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=fs)
    ax.set_xlim(xmin, None); ax.set_ylim(0, None)
    ax.set_title(title, loc="left", fontsize=fs + 0.5, color=INK)
    ax.set_xlabel("clean accuracy", color=MUTED, fontsize=fs)


fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.3) if paper else (10, 3.8), facecolor=SURF)
panel(axes[0], "gauss_4", r"(a) spread teacher ($\kappa^\star = 0.43$)", 0.45, 0.70)
panel(axes[1], "few_8", r"(b) concentrated teacher ($\kappa^\star = 0.90$)", 0.60, 0.85)
axes[0].set_ylabel(r"median certified radius ($\times 10^{-3}$)", color=MUTED, fontsize=fs)
handles = [matplotlib.lines.Line2D([], [], color=col, lw=1.6 if paper else 2, marker=mk, ms=5 if paper else 7,
                                   mec=SURF, mew=0.8, label=lab) for _, lab, col, mk in FAM]
axes[0].legend(handles=handles, frameon=False, fontsize=fs - 0.5, loc="lower left", labelcolor=INK, handlelength=1.5)
fig.tight_layout(pad=0.4 if paper else 1.0, w_pad=0.8)
fig.savefig(dst, dpi=160, facecolor=SURF)
print(dst)
