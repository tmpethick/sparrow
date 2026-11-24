"""The approximate oracles against the joint LMO along Muon's training trajectory:
nanoGPT of the training-speed experiment with Muon at step 0.01 and --lmo_track 1 (runs/speed-track/spectral-lr0.01-s*).
At every evaluation, the momentum M of each SwiGLU gate and value matrix (512 x 128, 4 layers) is mapped by
RowClip(Spectral) (U_1), Sparrow (U_3) and Aurora (Algorithm 3 of Dewulf et al. 2026), and compared with the joint
LMO D* (DRS, 300 iterations, relative duality gap below 1e-3): (a) <M, D> / <M, D*>, (b) cosine to D*,
(c) ||D||_F / ||D*||_F, (d) largest row norm / r. Mean over the 8 matrices, then mean +- s.e. over seeds.

    python scripts/plot_lmo_exact.py figures/lmo_exact.pdf --paper
"""
import glob, json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paper_style import paper_rc  # --paper: LaTeX fonts; add --mpl-font to opt out
import numpy as np

args = [a for a in sys.argv[1:] if not a.startswith("--")]
paper = "--paper" in sys.argv
dst = args[0] if args else sys.exit(__doc__)
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
METHODS = [("rowclip", "$U_1$ (RowClip$\\circ$Spectral)", "#009E73", "s"), ("sparrow", "Sparrow ($U_3$)", "#0072B2", "o"),
           ("aurora", "Aurora", "#CC79A7", "^")]
PANELS = [("value", "(a) $\\langle M,D\\rangle/\\langle M,D^\\star\\rangle$"), ("cos", "(b) cosine to $D^\\star$"),
          ("fro", "(c) $\\|D\\|_F/\\|D^\\star\\|_F$"), ("row", "(d) largest row norm / $r$")]

runs = sorted(glob.glob("runs/speed-track/spectral-lr0.01-s*/log.jsonl"))
ev = [[json.loads(l) for l in open(f) if '"eval"' in l] for f in runs]
steps = np.array([e["step"] for e in ev[0]])
gap = max(e["lmo_track"]["gap"] for r in ev for e in r)
print(f"{len(runs)} seeds, max duality gap {gap:.1e}")


def series(k, q):
    if q == "fro":
        return np.array([[e["lmo_track"][k]["fro"] / e["lmo_track"]["exact"]["fro"] for e in r] for r in ev])
    return np.array([[e["lmo_track"][k][q] for e in r] for r in ev])


fig, axes = plt.subplots(1, 4, figsize=(6.5, 1.75) if paper else (11, 2.8), facecolor=SURF, layout="constrained")
for ax, (q, title) in zip(axes, PANELS):
    ax.set_facecolor(SURF); ax.set_axisbelow(True); ax.grid(True, color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=fs - 0.5)
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    for j, (k, lab, col, mk) in enumerate(METHODS):
        Y = series(k, q)
        m, se = Y.mean(0), (Y.std(0, ddof=1) / np.sqrt(len(Y)) if len(Y) > 1 else 0 * Y[0])
        ax.fill_between(steps, m - se, m + se, color=col, alpha=0.2, lw=0)
        ax.plot(steps, m, color=col, lw=1.3, marker=mk, ms=3, mec=SURF, mew=0.5, markevery=(j, 3), label=lab)
        print(f"{q:5s} {k:8s} " + " ".join(f"{v:.3f}" for v in m[[0, 4, 8, -1]]) + "   (steps " +
              " ".join(str(s) for s in steps[[0, 4, 8, -1]]) + ")")
    ax.set_title(title, fontsize=fs, color=INK, loc="left")
    ax.set_xlabel("step", color=MUTED, fontsize=fs - 0.5)
    ax.set_xticks([400, 800, 1200])
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=fs - 0.5, labelcolor=INK,
           loc="outside upper center", ncol=3)
fig.savefig(dst, dpi=200, facecolor=SURF)
print("wrote", dst)
