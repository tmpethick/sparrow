"""Shared figure fonts for the plot scripts.

With --paper, text is typeset by LaTeX in Computer Modern so figures match the paper's body text
(needs a LaTeX installation). Add --mpl-font to keep matplotlib's own font (DejaVu Sans) instead.
Labels are written as LaTeX/mathtext ($\\kappa$, $\\times$, ...), which both modes render.
"""
import sys
import matplotlib.pyplot as plt


def paper_rc(fs, **extra):
    """rcParams for --paper figures at base font size fs; extra entries are added as is."""
    rc = {"font.size": fs, "pdf.fonttype": 42, **extra}
    if "--mpl-font" not in sys.argv:
        rc.update({"text.usetex": True, "font.family": "serif",
                   "text.latex.preamble": r"\usepackage{amsmath,amssymb}"})
    plt.rcParams.update(rc)
