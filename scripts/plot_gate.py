"""Muon vs Sparrow with an elementwise attention gate (runs/gate-attn, commands in REPRODUCE.md
section 4): nanoGPT Shakespeare-char, SwiGLU width 4d, one pass, unconstrained updates at
Muon's step size. The gate W_ag has one row per attention channel: square (4 heads x 32 = d) or
tall (4 heads x 128 = 4d).

    python scripts/plot_gate.py figures/gate.pdf --paper --with-hyperball

--with-hyperball adds, hatched and next to the tall-gate bars, MuonH, SparrowH and Sparrow with
weight normalization only (runs/gate-h, REPRODUCE.md section 4, fan-in initialization). They share
the data order with the tall-gate Muon runs, so they are also paired with tall-gate Muon; the
differences from MuonH are printed.

Bars: final validation loss minus Muon's on the same seed, mean over seeds, for Aurora (Algorithm 3 of
Dewulf et al. 2026 on every tall matrix, Muon otherwise), Sparrow on the attention gate only (Row_r; every other matrix
Muon) and Sparrow on every parameter group (the Sparrow algorithm), for each gate shape. Seeds share
the initialization and the data order, so the difference removes the seed-to-seed spread (std about
0.016, against a standard error of the difference of about 0.002). Error bars: one standard error of
the difference. Muon is the zero line, drawn in its color.
Prints the table, including the gate's maximal row norm (the norm in the gated-attention certificate) and spectral norm.
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
ROOT = "runs/gate-attn"
fs = 7.5 if paper else 9
if paper:
    paper_rc(fs)
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "white" if paper else "#fcfcfb"
SHAPES = [("square", "square gate\n$128 \\times 128$"), ("tall", "tall gate\n$512 \\times 128$")]
METHODS = [("muon", "Muon", "#D55E00"), ("aurora", "Aurora", "#CC79A7"),
           ("sparrowgate", "Sparrow (gates only)", "#56B4E9"), ("sparrow", "Sparrow (all)", "#0072B2")]
KEYS = [k for k, _, _ in METHODS]


def final(name):
    e = [json.loads(l) for l in open(f"{ROOT}/{name}/log.jsonl")]
    return [x for x in e if x.get("event") == "eval"][-1]


def seeds(shape, key):
    return {int(d.rsplit("-s", 1)[1]) for d in glob.glob(f"{ROOT}/{shape}-{key}-s*")
            if os.path.exists(f"{d}/done")}


print(f"{'gate':7s} {'method':12s} {'n':>2s} {'final':>8s} {'- Muon':>16s} {'better':>7s} "
      f"{'gate row':>9s} {'gate spec':>9s}")
res = {}
for shape, _ in SHAPES:
    sd = sorted(set.intersection(*(seeds(shape, k) for k in KEYS)))
    base = {s: final(f"{shape}-muon-s{s}") for s in sd}
    for key in KEYS:
        E = {s: final(f"{shape}-{key}-s{s}") for s in sd}
        diff = [E[s]["val_loss"] - base[s]["val_loss"] for s in sd]
        se = st.stdev(diff) / len(diff) ** 0.5 if key != "muon" else 0.0
        res[shape, key] = (st.mean(diff), se, st.mean(E[s]["val_loss"] for s in sd))
        print(f"{shape:7s} {key:12s} {len(sd):2d} {st.mean(E[s]['val_loss'] for s in sd):8.4f} "
              f"{st.mean(diff):+.4f}±{se:.4f} {sum(x < 0 for x in diff):>4}/{len(diff)} "
              f"{st.mean(E[s]['gate']['row_max'] for s in sd):9.3f} "
              f"{st.mean(E[s]['gate']['spec_max'] for s in sd):9.3f}")

hyper = "--with-hyperball" in sys.argv
fig, ax = plt.subplots(figsize=(6.5, 2.0) if paper else (9, 3.6), facecolor=SURF, layout="constrained")
ax.set_facecolor(SURF); ax.set_axisbelow(True); ax.grid(True, axis="y", color=GRID, lw=0.6)
for sp in ax.spines.values(): sp.set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=fs)
w = 0.2
vals = [0.0]
ax.axhline(0, color=METHODS[0][2], lw=1.2, zorder=1, label="Muon (reference)")


def bar(x, d, col, lab, hatch=None):
    """Mean paired difference d (one entry per seed) with one standard error."""
    m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    ax.bar(x, m, w, color=col, hatch=hatch, edgecolor=SURF, lw=0, label=lab, yerr=se,
           error_kw=dict(ecolor=INK, lw=0.8, capsize=2.5), zorder=2)
    vals.extend([m - se, m + se])


# Aurora and Sparrow minus Muon on the same seed, for each gate shape
centers = []
for i, (shape, _) in enumerate(SHAPES):
    x0 = i * (3 * w + 0.35)
    sd = sorted(set.intersection(*(seeds(shape, k) for k in KEYS)))
    for j, (key, lab, col) in enumerate(METHODS[1:]):
        d = [final(f"{shape}-{key}-s{s}")["val_loss"] - final(f"{shape}-muon-s{s}")["val_loss"] for s in sd]
        bar(x0 + (j - 1) * w, d, col, lab if i == 0 else None)
    centers.append(x0)
if hyper:  # tall gate with Hyperball weight normalization (fan-in init), minus tall-gate Muon on the same seed
    # hatch: xxx = update and weights normalized (MuonH, SparrowH), /// = weights only
    H = [("MuonH (update + weight norm.)", "muonh", "#D55E00", "xxx"),
         ("SparrowH (update + weight norm.)", "sparrowh", "#0072B2", "xxx"),
         ("Sparrow (weight norm.)", "sparrowhu", "#0072B2", "///")]
    fh = lambda k, s: final(f"../gate-h/{k}-lr0.01-s{s}")["val_loss"]
    hs = [s for s in range(10) if all(os.path.exists(f"runs/gate-h/{k}-lr0.01-s{s}/done") for _, k, _, _ in H)
          and os.path.exists(f"{ROOT}/tall-muon-s{s}/done")]
    x0 = centers[1] + 2 * w + 0.08
    for i, (lab, k, col, hatch) in enumerate(H):
        d = [fh(k, s) - final(f"tall-muon-s{s}")["val_loss"] for s in hs]
        bar(x0 + i * w, d, col, lab, hatch)
        dh = [fh(k, s) - fh("muonh", s) for s in hs]
        print(f"hyperball {k:10s} {len(hs):2d} {st.mean(fh(k, s) for s in hs):.4f} "
              f"- Muon {st.mean(d):+.4f}±{st.stdev(d) / len(d) ** 0.5:.4f} {sum(x < 0 for x in d):>2}/{len(d)}  "
              + (f"- MuonH {st.mean(dh):+.4f}±{st.stdev(dh) / len(dh) ** 0.5:.4f} {sum(x < 0 for x in dh):>2}/{len(dh)}"
                 if i else ""))
    centers[1] = (centers[1] - w + x0 + 2 * w) / 2
lo, hi = min(vals), max(vals)
ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo))
ax.set_xticks(centers); ax.set_xticklabels([l for _, l in SHAPES], color=INK, fontsize=fs)
ax.set_ylabel("val. loss minus Muon\n(same seed)", color=MUTED, fontsize=fs)
hl = list(zip(*ax.get_legend_handles_labels()))
hl = [x for x in hl if not x[1].startswith("MuonH")] + [x for x in hl if x[1].startswith("MuonH")]  # MuonH last
fig.legend(*zip(*hl), frameon=False, fontsize=fs - 0.5, labelcolor=INK, loc="outside upper center", ncol=4 if hyper else 4,
           handlelength=1.2, columnspacing=1.0)
fig.savefig(dst, dpi=200, facecolor=SURF)
print("wrote", dst)
