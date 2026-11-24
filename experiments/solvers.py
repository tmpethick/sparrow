"""Rank-adaptive iDRS vs fixed-rank DRS for the joint row/spectral LMO.

DRS on min_W f(W) + g(W), with f = -<G,.> + indicator(row norms <= r) and
g = indicator(||W||_op <= s). The row step prox_{gamma f}(Z) = RowClip_r(Z + gamma G)
is exact. The spectral step SpectralClip_s is computed inexactly by a shared
warm-started rank-k subspace iteration, which returns an EXACT normal-cone
pair (w, n) and an explicitly measured resolvent error. Fixed-rank DRS accepts
every step. iDRS (Algorithm II of iFedDR) accepts a pair only under its
relative-error test and otherwise refines: one more subspace sweep, or a
doubled rank when the unresolved tail dominates the error. Exact DRS uses a
full SVD. Exact DRS in its separate implementation (solve_idrs with exact=True), solved
more tightly, is the reference for the certificates.

Run: python experiments/solvers.py --out runs/lmo-rank-adaptive
"""
import argparse
import csv
import json
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from sparrow.iterative_solvers import (best_of_three, best_of_two, low_rank_gradient, metrics, solve,
                                 solve_idrs)


def instance_summary(g, r, s, reference):
    """How strongly each constraint binds, measured against the reference."""
    row_lmo = r * g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
    left, _, right_t = np.linalg.svd(g, full_matrices=False)
    spectral_lmo = s * left @ right_t
    best = metrics(g, best_of_two(g, r, s), r, s)['score']
    best3 = metrics(g, best_of_three(g, r, s), r, s)['score']
    w_rank = int(np.sum(np.linalg.svd(reference['w'], compute_uv=False) > s * (1 - 1e-4)))
    return dict(row_lmo_spectral_ratio=float(np.linalg.svd(row_lmo, compute_uv=False)[0] / s),
                spectral_lmo_row_ratio=float(np.linalg.norm(spectral_lmo, axis=1).max() / r),
                best_of_two_gap=(reference['upper'] - best) / reference['upper'],
                best_of_three_gap=(reference['upper'] - best3) / reference['upper'],
                solution_rank_at_cap=w_rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('runs/lmo-rank-adaptive'))
    parser.add_argument('--rows', type=int, default=1024)
    parser.add_argument('--cols', type=int, default=256)
    parser.add_argument('--decays', type=float, nargs='+', default=[1.0, 1.5])
    parser.add_argument('--ratios', type=float, nargs='+', default=[6, 8, 12])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--ranks', type=int, nargs='+', default=[16, 32, 64, 128])
    parser.add_argument('--initial-rank', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--max-iters', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3,
                        help='Fresh solves per certified method; median time reported')
    parser.add_argument('--replot', action='store_true',
                        help='Only redraw summary/figure from an existing --out run')
    args = parser.parse_args()
    if args.replot:
        return replot(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    records, traces = [], []
    for decay in args.decays:
        for seed in args.seeds:
            g = low_rank_gradient(args.rows, args.cols, decay, seed)
            for ratio in args.ratios:
                r, s = 1 / ratio, 1.0
                identity = dict(decay=decay, seed=seed, ratio=ratio)
                # Separate implementation of exact DRS, solved more tightly.
                reference = solve_idrs(g, r, s, exact=True, tol=args.tol / 10,
                                       max_iters=10 * args.max_iters)
                if not reference['converged']:
                    raise RuntimeError('exact-SVD reference did not certify convergence')
                info = instance_summary(g, r, s, reference)
                methods = [('exact', dict(method='exact')),
                           ('idrs', dict(method='idrs', rank=args.initial_rank))]
                methods += [(f'drs_r{k}', dict(method='fixed', rank=k)) for k in args.ranks]
                for name, options in methods:
                    result = solve(g, r, s, gamma=args.gamma, sigma=args.sigma, tol=args.tol,
                                   max_iters=args.max_iters, seed=seed, **options)
                    # Weak duality against the independent reference.
                    if result['lower'] > reference['upper'] + 1e-10 or \
                            reference['lower'] > result['upper'] + 1e-10:
                        raise RuntimeError('certificates disagree with the reference')
                    if name == 'exact' and not result['converged']:
                        raise RuntimeError('exact DRS did not certify convergence')
                    if result['converged']:
                        # Uncertified runs stop at the iteration cap; one timing suffices.
                        result['seconds'] = float(np.median([result['seconds']] + [
                            solve(g, r, s, gamma=args.gamma, sigma=args.sigma, tol=args.tol,
                                  max_iters=args.max_iters, seed=seed, **options)['seconds']
                            for _ in range(args.repeats - 1)]))
                    records.append(dict(**identity, **info, method=name, rank=result['rank'],
                        converged=result['converged'], gap=result['gap'],
                        iterations=result['iterations'], seconds=result['seconds'],
                        products=result['products'], rejected=result['rejected']))
                    traces.append(dict(**identity, method=name, history=result['history']))
                print(f'decay={decay} seed={seed} s/r={ratio}: ' + ', '.join(
                    f"{x['method']} {'ok' if x['converged'] else 'FAIL'} {x['seconds']:.2f}s"
                    for x in records if all(x[key] == identity[key] for key in identity)),
                    flush=True)
    with (args.out / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.out / 'history.json').write_text(json.dumps(traces))
    config = {key: (str(value) if isinstance(value, Path) else value)
              for key, value in vars(args).items()}
    config.update(numpy_version=np.__version__, platform=platform.platform(),
                  OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS'))
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    summary = summarize(records)
    (args.out / 'summary.md').write_text(summary)
    print(summary)
    plot(records, traces, args.out)


def summarize(records):
    names = list(dict.fromkeys(x['method'] for x in records))
    cases = sorted({(x['decay'], x['seed'], x['ratio']) for x in records})
    lines = ['| Method | Certified | Total time (s) | Median time (s) | Median final rank |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name in names:
        group = [x for x in records if x['method'] == name]
        lines.append(f"| {name} | {sum(x['converged'] for x in group)}/{len(group)} | "
                     f"{sum(x['seconds'] for x in group):.1f} | "
                     f"{np.median([x['seconds'] for x in group]):.2f} | "
                     f"{np.median([x['rank'] for x in group]):.0f} |")
    best = []
    for case in cases:
        group = [x for x in records if (x['decay'], x['seed'], x['ratio']) == case
                 and x['method'].startswith('drs_r') and x['converged']]
        best.append(min(x['seconds'] for x in group) if group else np.nan)
    lines += ['', f'Per-instance fastest certified fixed rank (oracle tuning): total '
              f'{np.nansum(best):.1f}s, {int(np.isnan(best).sum())} instances with none.']
    return '\n'.join(lines) + '\n'


def plot(records, traces, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(x['method'] for x in records))
    settings = sorted({(x['decay'], x['ratio']) for x in records})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        for j, setting in enumerate(settings):
            group = [x for x in records if x['method'] == name
                     and (x['decay'], x['ratio']) == setting]
            position = j + (i - len(names) / 2 + 0.5) * width
            if all(x['converged'] for x in group):
                axes[0].bar(position, np.median([x['seconds'] for x in group]), width,
                            color=f'C{i}', label=name if j == 0 else None)
            else:
                axes[0].plot(position, 0, 'x', color=f'C{i}', markersize=7,
                             transform=axes[0].get_xaxis_transform(), clip_on=False,
                             label=name if j == 0 else None)
    axes[0].set(yscale='log', ylabel='Median solve time over seeds (s)',
                xticks=np.arange(len(settings)),
                xticklabels=[f'decay {d:g}\ns/r={q:g}' for d, q in settings],
                title='x on the axis: fails to certify on some seed')
    axes[0].legend(fontsize=7, ncol=2)
    case = max(sorted({(x['decay'], x['seed'], x['ratio']) for x in records}),
               key=lambda c: next(x['rank'] for x in records if x['method'] == 'idrs'
                                  and (x['decay'], x['seed'], x['ratio']) == c))
    for i, name in enumerate(names):
        trace = next(t for t in traces if t['method'] == name and
                     (t['decay'], t['seed'], t['ratio']) == case)
        axes[1].plot([h['seconds'] for h in trace['history']],
                     np.maximum([h['gap'] for h in trace['history']], 1e-12),
                     color=f'C{i}', label=name)
    axes[1].set(xlabel='Solve time (s)', ylabel='Certified relative gap', yscale='log',
                title=f'decay {case[0]:g}, seed {case[1]}, s/r={case[2]:g}')
    axes[1].legend(fontsize=7)
    fig.savefig(out / 'comparison.png', dpi=160)
    plt.close(fig)


def replot(run):
    """Regenerate summary.md and comparison.png from a finished run directory."""
    records = []
    with (run / 'results.csv').open() as handle:
        for row in csv.DictReader(handle):
            for key in ('decay', 'ratio', 'gap', 'seconds'):
                row[key] = float(row[key])
            for key in ('seed', 'rank', 'iterations', 'rejected'):
                row[key] = int(row[key])
            row['converged'] = row['converged'] == 'True'
            records.append(row)
    traces = json.loads((run / 'history.json').read_text())
    (run / 'summary.md').write_text(summarize(records))
    plot(records, traces, run)


if __name__ == '__main__':
    main()
