"""Robustness frontiers of the teacher-student classification runs (experiments/synthetic.py).

R_M(a) = best median certified radius over all runs of family M (budgets x step sizes) with
clean accuracy >= a, computed per student seed and then averaged (mean +- std over seeds).

    python scripts/table_ts_robust.py runs/tsr-fam gauss_4 0.65,0.70,0.75
    python scripts/table_ts_robust.py runs/tsr-fam few_8 0.80,0.85,0.90 --latex
"""
import json, os, statistics as st, sys

FAMS = [("sgd_frob", "SGD + Frobenius"), ("adam_linf", "Adam + $\\ell_\\infty$"),
        ("muon_spec", "Muon + spectral"), ("ours_rowspec", "Ours + row/spectral")]


def load(out):
    return [json.loads(l) for l in open(os.path.join(out, "results.jsonl"))]


def frontier(R, teacher, fam, levels, **match):
    rows = []
    seeds = sorted({r["seed"] for r in R if r["teacher"] == teacher and r["set"] == fam
                    and all(r.get(k) == v for k, v in match.items())})
    for s in seeds:
        rs = [r for r in R if r["teacher"] == teacher and r["set"] == fam and r["seed"] == s
              and all(r.get(k) == v for k, v in match.items())]
        best = lambda a: max([r["radius_median"] for r in rs if r["acc"] >= a], default=None)
        rows.append([max(r["acc"] for r in rs)] + [best(a) for a in levels])
    return rows


def fmt(vals, n, latex, scale=1.0, digits=4):
    v = [x * scale for x in vals if x is not None]
    if not v or len(v) < n:
        return "--" if latex else "-"
    m = f"{st.mean(v):.{digits}f}"
    if len(v) > 1:
        m += (f" \\pm {st.stdev(v):.{digits}f}" if latex else f" ± {st.stdev(v):.{digits}f}")
    return f"${m}$" if latex else m


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    latex = "--latex" in sys.argv
    out, teacher, levels = args[0], args[1], [float(x) for x in args[2].split(",")]
    match = {"steps": int(args[3])} if len(args) > 3 else {}
    R = load(out)
    for fam, name in FAMS:
        rows = frontier(R, teacher, fam, levels, **match)
        if not rows:
            continue
        n = len(rows)
        cells = [fmt([r[0] for r in rows], n, latex, digits=3)] + [
            fmt([r[i + 1] for r in rows], n, latex, scale=1e3, digits=2) for i in range(len(levels))]
        if latex:
            print(f"{name} & " + " & ".join(cells) + " \\\\")
        else:
            print(f"{fam:13s} seeds={n}  max acc {cells[0]:16s} " +
                  "  ".join(f"R(>={a:.2f})x1e3 {c}" for a, c in zip(levels, cells[1:])))


if __name__ == "__main__":
    main()
