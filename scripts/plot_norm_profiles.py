"""Normalization profiles h of N_RMS and N_tanh (the paper's normalization profiles), N_h(x) =
h(r) / r x with r = rms(x), tau = 1 and a = 1/2:

    h_RMS(r)  = tau r / sqrt(tau^2 + r^2)
    h_tanh(r) = r for r <= a, else a + (tau - a) tanh((r - a) / (tau - a))

(a) the profile h(r), with the identity h = r and the saturation level tau; (b) the gain h(r) / r, which is
||D N_h(x)||_op (eq. profile-jacobian): N_tanh is the identity up to rms a, N_RMS shrinks every input.

    python scripts/plot_norm_profiles.py figures/norm_profiles.pdf --paper
"""
import sys
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
TAU, A = 1.0, 0.5

r = np.linspace(1e-4, 3.0, 600)
h_rms = TAU * r / np.sqrt(TAU ** 2 + r ** 2)
h_tanh = np.where(r <= A, r, A + (TAU - A) * np.tanh(np.maximum(r - A, 0) / (TAU - A)))
# color, marker and line style differ, so the two curves stay apart without color
PROFILES = [(r"$\mathbb{N}_{\mathrm{RMS}}$", h_rms, "#0072B2", "o", "-"),
            (r"$\mathbb{N}_{\tanh}$", h_tanh, "#D55E00", "s", (0, (4, 1.5)))]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 1.9) if paper else (10, 3), facecolor=SURF, layout="constrained")
for ax, (title, gain) in zip(axes, [(r"(a) profile $h(r)$", False), (r"(b) gain $h(r)/r$", True)]):
    ax.set_facecolor(SURF); ax.set_axisbelow(True); ax.grid(True, color=GRID, lw=0.6)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=fs - 0.5)
    if gain:
        ax.axhline(1.0, color=MUTED, lw=0.8, ls=":", zorder=1)
    else:
        ax.plot(r, r, color=MUTED, lw=0.8, ls=":", zorder=1, label=r"identity $h(r)=r$")
        ax.axhline(TAU, color=MUTED, lw=0.8, ls="--", zorder=1, label=r"level $\tau$")
    ax.axvline(A, color=AXIS, lw=0.8, zorder=1)
    for j, (lab, h, col, mk, ls) in enumerate(PROFILES):
        ax.plot(r, h / r if gain else h, color=col, lw=1.4, ls=ls, marker=mk, ms=3.2, mec=SURF, mew=0.5,
                markevery=(25 + 25 * j, 50), label=lab)
    ax.set_xlim(0, 3); ax.set_ylim(0, 1.15 if not gain else 1.08)
    ax.set_xticks([0, A, TAU, 2, 3], ["0", "$a$", r"$\tau$", "2", "3"])
    ax.set_xlabel(r"$r=\mathrm{rms}(x)$", color=MUTED, fontsize=fs - 0.5)
    ax.set_title(title, fontsize=fs, color=INK, loc="left")
handles, labels = axes[0].get_legend_handles_labels()
order = [2, 3, 0, 1]
fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=fs - 0.5,
           labelcolor=INK, loc="outside upper center", ncol=4)
fig.savefig(dst, dpi=200, facecolor=SURF)
print("wrote", dst)
