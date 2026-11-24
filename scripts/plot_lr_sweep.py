"""Step-size sweep of Muon, RowClip(Polar) (U_1), Aurora (Algorithm 3 of Dewulf et al. 2026) and Sparrow on the
training-speed setup (the paper's step-size sweep): Tiny Shakespeare, one pass, unconstrained updates,
all differing only on the SwiGLU gate and value matrices (runs/lr-sweep, commands in REPRODUCE.md section 3).

    python scripts/plot_lr_sweep.py figures/speed_lr.pdf --paper

Plots the final validation loss against the step size, mean over seeds with one standard error. Prints each
method's best step size and the paired differences between methods, each at its own best step size.
"""
import glob, json, os, re, sys
# colors: Okabe-Ito colorblind-safe palette, as in scripts/plot_speed.py
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
METHODS = [("muon", "Muon", "#D55E00", "D"), ("rowclip", r"$U_1$ (RowClip$\circ$Spectral)", "#009E73", "s"),
           ("aurora", "Aurora", "#CC79A7", "^"), ("sparrow", "Sparrow", "#0072B2", "o")]

final = {}
for d in glob.glob("runs/lr-sweep/*/done"):
    run = os.path.dirname(d)
    m = re.fullmatch(r"(\w+)-lr([\d.]+)-s(\d+)", os.path.basename(run))
    if not m:                                           # other variants in the same folder, e.g. rowclipc0.5-*
        continue
    ev = [r for r in map(json.loads, open(f"{run}/log.jsonl")) if r.get("event") == "eval"]
    final.setdefault((m[1], float(m[2])), {})[int(m[3])] = ev[-1]["val_loss"]
lrs = sorted({lr for _, lr in final})
best = {k: min(lrs, key=lambda lr: np.mean(list(final[k, lr].values()))) for k, *_ in METHODS}

fig, ax = plt.subplots(figsize=(3.6, 2.0) if paper else (6, 3.4), facecolor=SURF, layout="constrained")
ax.set_facecolor(SURF); ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=fs)
ax.grid(True, color=GRID, lw=0.6)
for i, (key, lab, col, mk) in enumerate(METHODS):
    v = [np.array(list(final[key, lr].values())) for lr in lrs]
    mean = np.array([x.mean() for x in v])
    se = np.array([x.std(ddof=1) / np.sqrt(len(x)) for x in v])
    x = np.array(lrs) * 2 ** (0.05 * (i - 1.5))         # small horizontal offset so the error bars do not overlap
    ax.errorbar(x, mean, yerr=se, color=col, lw=1.3, marker=mk, ms=3.5 if paper else 5, mec=SURF, mew=0.6,
                capsize=1.5, elinewidth=0.8, label=lab, zorder=3)
    j = lrs.index(best[key])
    ax.plot(x[j], mean[j], marker=mk, ms=6.5 if paper else 9, mfc="none", mec=col, mew=1.0, zorder=4)
ax.set_xscale("log", base=2)
ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
ax.minorticks_off()
ax.set_xlabel(r"step size $\eta$", color=MUTED, fontsize=fs)
ax.set_ylabel("final val. loss", color=MUTED, fontsize=fs)
ax.legend(frameon=False, fontsize=fs, labelcolor=INK, loc="upper center")
fig.savefig(dst, dpi=200, facecolor=SURF)

print("best step size:", {k: best[k] for k, *_ in METHODS})
pairs = [("sparrow", "muon"), ("rowclip", "muon"), ("aurora", "muon"), ("sparrow", "rowclip"), ("sparrow", "aurora")]
for a, b in pairs:
    seeds = sorted(set(final[a, best[a]]) & set(final[b, best[b]]))
    d = np.array([final[a, best[a]][s] - final[b, best[b]][s] for s in seeds])
    print(f"{a} - {b} (each at its best step size): {d.mean():+.4f} ± {d.std(ddof=1) / np.sqrt(len(d)):.4f}, "
          f"lower on {(d < 0).sum()}/{len(d)} seeds")
print("wrote", dst)
