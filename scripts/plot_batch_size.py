"""Batch-size sweep of Muon, Aurora and Sparrow without repeated data (the paper's
batch-size figure): the training-speed setup on the Complete Works, 750 steps at batch 12, 24, 48 and 96,
Aurora and Sparrow differing from Muon only on the SwiGLU gate and value matrices (commands in
REPRODUCE.md section 3).

    python scripts/plot_batch_size.py figures/batch_size.pdf --paper [--lr 0.01]

Step size eta = 0.01 comes from runs/bs-complete (<opt>-b<batch>-s<seed>, seeds 0-4), the other step sizes
from runs/bs-lr (<opt>-b<batch>-s<seed>-lr<eta>): 0.02 everywhere, 0.04 at batch 96 (it loses there, and
the best eta grows with the batch size) and 0.005 for Muon at batch 12, on seeds 0-2. By default every
method uses its own best eta per batch size (lowest mean final validation loss over the seeds of its
eta = 0.02 runs; single-seed probes are ignored), drawn filled for 0.02 and hollow for 0.01, and the plot
shows its final validation loss minus Muon's at Muon's best eta, paired by seed, mean over seeds with one
standard error; dashed lines show every method at eta = 0.01 over all of its seeds. With --lr, every
method uses that eta and all of its seeds. Prints the same numbers, the chosen eta and
Muon's final validation loss.
"""
import glob, json, os, re, statistics as st, sys
# colors: Okabe-Ito colorblind-safe palette, as in scripts/plot_gate.py
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
MUON = "#D55E00"
METHODS = [("aurora", "Aurora", "#CC79A7", "s"), ("sparrow", "Sparrow", "#0072B2", "o")]

final = {}                                 # (opt, batch, eta) -> seed -> final validation loss
for root, pat in (("runs/bs-complete", r"(\w+)-b(\d+)-s(\d+)()"), ("runs/bs-lr", r"(\w+)-b(\d+)-s(\d+)-lr([\d.]+)")):
    for d in glob.glob(f"{root}/*/done"):
        run = os.path.dirname(d)
        m = re.fullmatch(pat, os.path.basename(run))
        if not m:
            continue
        ev = [r for r in map(json.loads, open(f"{run}/log.jsonl")) if r.get("event") == "eval"]
        final.setdefault((m[1], int(m[2]), float(m[4] or 0.01)), {})[int(m[3])] = ev[-1]["val_loss"]
fixed = None
if "--lr" in sys.argv:
    fixed = float(sys.argv[sys.argv.index("--lr") + 1])
    args = [a for a in args if a != sys.argv[sys.argv.index("--lr") + 1]]
    dst = args[0] if args else sys.exit(__doc__)
ETAS = (0.005, 0.01, 0.02, 0.04)


def pick(o, b):
    """eta and seed -> final loss for opt o at batch b: the fixed eta, or the best of ETAS that were run."""
    if fixed is not None:
        return fixed, final[o, b, fixed]
    seeds = set(final[o, b, 0.02])             # the seeds of the sweep; single-seed probes are ignored
    etas = [e for e in ETAS if seeds <= set(final.get((o, b, e), {}))]
    best = min(etas, key=lambda e: st.mean(final[o, b, e][s] for s in seeds))
    return best, {s: final[o, b, best][s] for s in seeds}


batches = sorted({b for _, b, _ in final})
best = {(o, b): pick(o, b) for o in ("muon", "aurora", "sparrow") for b in batches}

fig, ax = plt.subplots(figsize=(3.6, 2.0) if paper else (6, 3.4), facecolor=SURF, layout="constrained")
ax.set_facecolor(SURF); ax.set_axisbelow(True); ax.grid(True, color=GRID, lw=0.6)
for s in ax.spines.values():
    s.set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=fs)
ax.axhline(0, color=MUON, lw=1.2, zorder=1)
FILL = {0.02: True}                         # filled marker: best eta 0.02, hollow: 0.01


def marker_kw(col, eta):
    return dict(color=col, mec=col, mfc=col if FILL.get(eta) else SURF, mew=0.9)


ax.plot(batches, [0] * len(batches), ls="none", marker="D", ms=3.5, zorder=3,
        **{k: v for k, v in marker_kw(MUON, None).items() if k != "color"})
for b in batches:                           # Muon's own best eta, on the reference line
    ax.plot([b], [0], ls="none", marker="D", ms=3.5, zorder=3, **marker_kw(MUON, best["muon", b][0]))
print(f"{'batch':>5s} {'Muon (eta)':>14s} " + " ".join(f"{lab + ' - Muon (eta)':>30s}" for _, lab, _, _ in METHODS))
rows = {b: [] for b in batches}
for j, (key, lab, col, mk) in enumerate(METHODS):
    m, e, etas, mf = [], [], [], []
    for b in batches:
        (_, fm), (eta, fo) = best["muon", b], best[key, b]
        sd = sorted(set(fm) & set(fo))
        d = [fo[s] - fm[s] for s in sd]
        m.append(st.mean(d)); e.append(st.stdev(d) / len(d) ** 0.5); etas.append(eta)
        rows[b].append(f"{m[-1]:+.4f}±{e[-1]:.4f} {sum(x < 0 for x in d)}/{len(d)} ({eta:g})")
        if fixed is None:                   # reference: every method at eta = 0.01, all of its seeds
            g = final[key, b, 0.01]; h = final["muon", b, 0.01]
            mf.append(st.mean(g[s] - h[s] for s in set(g) & set(h)))
    xs = np.array(batches) * 2 ** ((j - 0.5) * 0.08)   # small offset so the error bars do not overlap
    if mf:
        ax.plot(xs, mf, color=col, lw=0.9, ls=(0, (3, 2)), alpha=0.55, zorder=1)
    ax.errorbar(xs, m, yerr=e, color=col, lw=1.4, capsize=2.5, elinewidth=0.8, zorder=2)
    for x, y, eta in zip(xs, m, etas):
        ax.plot([x], [y], ls="none", marker=mk, ms=4, zorder=3, **marker_kw(col, eta))
for b in batches:
    eta, fm = best["muon", b]
    print(f"{b:5d} {st.mean(fm.values()):7.4f} ({eta:g}) " + " ".join(f"{c:>30s}" for c in rows[b]))
ax.set_xscale("log", base=2)
ax.set_xticks(batches); ax.set_xticklabels([str(b) for b in batches], color=INK)
ax.minorticks_off()
ax.set_xlabel("batch size (windows of 64 characters)", color=MUTED, fontsize=fs)
ax.set_ylabel("val. loss minus Muon\n(same seed)", color=MUTED, fontsize=fs)
lo = min(ax.get_ylim()[0], -0.016)
ax.set_ylim(lo, 0.003)
ax.set_xlim(batches[0] / 1.25, batches[-1] * 1.6)   # room on the right for the legend
import matplotlib.lines as ml
handles = [ml.Line2D([], [], color=MUON, lw=1.2, marker="D", ms=3.5, mfc=MUON, label="Muon (reference)")]
handles += [ml.Line2D([], [], color=c, lw=1.4, marker=mk, ms=4, mfc=c, label=lab) for _, lab, c, mk in METHODS]
if fixed is None:
    handles += [ml.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (3, 2)), label=r"all at $\eta=0.01$")]
ax.legend(handles=handles, frameon=False, fontsize=fs - 0.5, labelcolor=INK, loc="lower right",
          handlelength=1.6, borderaxespad=0.2)
fig.savefig(dst, dpi=200, facecolor=SURF)
print("wrote", dst)
