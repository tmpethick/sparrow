"""Exact and inexact solvers for the joint row/spectral LMO (float64 NumPy).

    max_W <G, W>  subject to  ||W||_{2->inf} <= r,  ||W||_op <= s

- Clipping candidates: `spectral_then_row`, `row_then_spectral`, `best_of_two`, `best_of_three`.
- Certified gaps (`metrics`, `dual_value`).
- iDRS, Algorithm II of iFedDR, with a graph-certified inexact nuclear prox, or exact DRS with
  `exact=True` (`solve_idrs`); exact DRS is ADMM on the dual min_{A+B=G} r||A||_{2,1} + s||B||_*.
- DRS with a warm-started rank-k subspace clip, fixed-rank or rank-adaptive iDRS (`solve`).
- Test instances: `synthetic` and `low_rank_gradient`.

The training oracles (U_1 = RowClip_r Spectral_s, Sparrow, Aurora) are in `sparrow.approx_oracles`.
"""
from time import perf_counter

import numpy as np


# =========================================================================== clipping

def row_clip(x, r):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x * np.minimum(1.0, r / np.maximum(norms, np.finfo(float).tiny))


def spectral_clip(x, s):
    u, v, vt = np.linalg.svd(x, full_matrices=False)
    return (u * np.minimum(v, s)) @ vt


def row_shrink(x, threshold):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x * np.maximum(0.0, 1.0 - threshold / np.maximum(norms, np.finfo(float).tiny))


def spectral_then_row(g, r, s):
    # Compact SVD also gives a valid polar LMO for rank-deficient matrices.
    u, _, vt = np.linalg.svd(g, full_matrices=False)
    return row_clip(s * (u @ vt), r)


def row_then_spectral(g, r, s):
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    z = r * np.divide(g, norms, out=np.zeros_like(g), where=norms > 0)
    return spectral_clip(z, s)


def best_of_two(g, r, s):
    a, b = spectral_then_row(g, r, s), row_then_spectral(g, r, s)
    return a if np.sum(g * a) >= np.sum(g * b) else b


def row_spectral_row(g, r, s):
    norms = np.linalg.norm(g, axis=1, keepdims=True)
    return spectral_then_row(r * np.divide(g, norms, out=np.zeros_like(g), where=norms > 0), r, s)


def best_of_three(g, r, s):
    candidates = (spectral_then_row(g, r, s), row_then_spectral(g, r, s),
                  row_spectral_row(g, r, s))
    return max(candidates, key=lambda w: np.sum(g * w))


def dual_value(g, a, r, s):
    # Any split A + B = G gives an upper bound on the LMO value (weak duality).
    return float(r * np.linalg.norm(a, axis=1).sum()
                 + s * np.linalg.svd(g - a, compute_uv=False).sum())


def metrics(g, w, r, s):
    return dict(score=float(np.sum(g * w)),
                row_ratio=float(np.linalg.norm(w, axis=1).max() / r),
                spectral_ratio=float(np.linalg.svd(w, compute_uv=False)[0] / s))


def synthetic(family, rows, cols, seed):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((rows, cols))
    if family == 'heterogeneous':
        g *= np.geomspace(0.05, 20, rows)[:, None]
    elif family == 'low_rank':
        u, _, vt = np.linalg.svd(g, full_matrices=False)
        g = (u * np.geomspace(1, 1e-3, min(rows, cols))) @ vt
    elif family != 'gaussian':
        raise ValueError(family)
    return g / np.linalg.norm(g)


# =========================================================================== iDRS (Algorithm II of iFedDR)

def nuclear_graph_pair(z, gamma, s, *, rank=None, omega=None, basis=None,
                       complete_tail=False):
    """Return x, u in partial(s||.||_*)(x), epsilon=(z-x)/gamma-u.

    Compute the EXACT prox at z_tilde=Q Q^T z, using the SVD of Q^T z.
    Include inactive singular directions in u, not only the polar of x.
    Thus ||u||_op<=s and <u,x>=s||x||_*. Optional tail completion adds
    a norm-bounded subgradient component orthogonal to both projected
    singular spaces. epsilon is always evaluated from the actual pair.
    rank=None uses the full exact SVD (the DRS control and fallback).
    """
    full_rank = min(z.shape)
    if rank is None or rank >= full_rank:
        left, values, vt = np.linalg.svd(z, full_matrices=False)
        used_rank = full_rank
    else:
        if rank < 1 or (omega is None and basis is None):
            raise ValueError('positive rank and a sketch matrix or basis required')
        if basis is None:
            q, _ = np.linalg.qr(z @ omega[:, :rank], mode='reduced')
        else:
            q = basis
            if q.shape != (z.shape[0], rank):
                raise ValueError('basis has the wrong shape')
        small_left, values, vt = np.linalg.svd(q.T @ z, full_matrices=False)
        left = q @ small_left
        used_rank = rank
    shrunk = np.maximum(values - gamma * s, 0.0)
    x = (left * shrunk) @ vt
    u = (left * np.minimum(values / gamma, s)) @ vt
    if complete_tail and used_rank < full_rank:
        # Orthogonal to ALL projected singular directions on BOTH sides.
        # Its spectral norm is certified by its Frobenius norm. This adds
        # sub-threshold modes without assuming approximate triplets exact.
        tail = z / gamma - left @ (left.T @ (z / gamma))
        tail -= (tail @ vt.T) @ vt
        tail_norm = np.linalg.norm(tail)
        u += tail * min(1.0, s / max(tail_norm, np.finfo(float).tiny))
    epsilon = (z - x) / gamma - u
    return dict(x=x, u=u, epsilon=epsilon, rank=used_rank,
                right_basis=vt.T, nuclear_x=float(shrunk.sum()))


def idrs_step(z, g, r, s, *, gamma=1.0, sigma=0.5, relaxation=1.0,
              initial_rank=2, omega=None, exact=False, inner_solver="power",
              max_power_iters=5, warm_basis=None):
    """One accepted Algorithm II step; reject/refine until its error test holds.

    q=x-gamma*u; p=prox_gamma_f(q); y=(q-p)/gamma;
    z_bar=x-gamma*y; v=u+y.
    Accept ||epsilon|| <= sigma max(||v||,||z_bar-z||/gamma).
    alpha=<v,z-z_bar>/||v||^2; z_next=z-relaxation*alpha*v.
    With epsilon=0, alpha=gamma and this is ordinary relaxed DRS.
    The power backend refines a fixed block until acceptance, enlarging
    it after max_power_iters failed sweeps. The subspace backend retains
    the original rank-doubling range sketches as a comparison control.
    """
    if gamma <= 0 or not np.isfinite(gamma):
        raise ValueError('gamma must be finite and positive')
    if not 0 <= sigma < 1 or not 0 < relaxation < 2 or initial_rank < 1:
        raise ValueError('require 0<=sigma<1, 0<relaxation<2, initial_rank>=1')
    if inner_solver not in ('power', 'subspace') or max_power_iters < 1:
        raise ValueError('inner_solver must be power/subspace and max_power_iters>=1')
    exact = exact or sigma == 0
    full_rank = min(z.shape)
    rank = full_rank if exact else min(initial_rank, full_rank)
    if not exact and inner_solver == 'power' and warm_basis is not None:
        rank = min(max(initial_rank, warm_basis.shape[1]), full_rank)
    if omega is None:
        omega = np.random.default_rng(0).standard_normal((z.shape[1], full_rank))
    attempts, svd_work, power_iters, powers_at_rank, rank_expansions = 0, 0, 0, 0, 0
    q = None
    if inner_solver == 'power' and rank < full_rank:
        start_vectors = omega[:, :rank]
        if warm_basis is not None and warm_basis.shape[1] == rank:
            start_vectors = warm_basis
        q, _ = np.linalg.qr(z @ start_vectors, mode='reduced')
    while True:
        pair = nuclear_graph_pair(z, gamma, s, rank=rank, omega=omega,
            basis=q if inner_solver == 'power' else None,
            complete_tail=inner_solver == 'power')
        x, u, epsilon = pair['x'], pair['u'], pair['epsilon']
        reflected = x - gamma * u
        p = g - row_shrink(g - reflected, gamma * r)
        y = (reflected - p) / gamma
        z_bar = x - gamma * y
        v = u + y
        norm_v = np.linalg.norm(v)
        error_norm = float(np.linalg.norm(epsilon))
        denominator = max(norm_v, np.linalg.norm(z_bar - z) / gamma)
        attempts += 1
        svd_work += rank  # ranks summed over all projected/full SVD attempts
        # Full fallback is the exact proximal calculation to floating-point
        # precision. Never relax the criterion for a truncated sketch.
        roundoff = 100 * np.finfo(float).eps * max(np.linalg.norm(z) / gamma, s, 1.0)
        accepted = error_norm <= sigma * denominator
        exact_roundoff = rank == full_rank and error_norm <= roundoff
        if accepted or exact_roundoff:
            break
        if pair['rank'] == full_rank:
            raise RuntimeError('full proximal fallback failed the graph residual check')
        if inner_solver == 'power' and powers_at_rank < max_power_iters:
            # Stabilized block power iteration: QR between z^T and z to
            # avoid squaring the conditioning in floating-point arithmetic.
            right, _ = np.linalg.qr(z.T @ q, mode='reduced')
            q, _ = np.linalg.qr(z @ right, mode='reduced')
            powers_at_rank += 1
            power_iters += 1
            continue
        new_rank = min(full_rank, max(rank + 1, 2 * rank))
        if inner_solver == 'power' and new_rank < full_rank:
            # More powers cannot remove the fixed-rank approximation floor.
            extra = z @ omega[:, rank:new_rank]
            q, _ = np.linalg.qr(np.column_stack([q, extra]), mode='reduced')
        rank = new_rank
        rank_expansions += 1
        powers_at_rank = 0
    if norm_v <= roundoff and error_norm <= roundoff:
        alpha, z_next = gamma, z.copy()
    else:
        alpha = float(np.sum(v * (z - z_bar)) / (norm_v * norm_v))
        if not np.isfinite(alpha) or alpha <= 0:
            raise RuntimeError('invalid iDRS correction: alpha must be positive')
        z_next = z - relaxation * alpha * v
    return dict(z_next=z_next, x=x, u=u, y=y, epsilon=epsilon, v=v,
                z_bar=z_bar, alpha=alpha, error_norm=error_norm,
                error_denominator=float(denominator),
                relative_error=error_norm / denominator if denominator else 0.0,
                exact_roundoff=bool(exact_roundoff and not accepted),
                attempts=attempts, svd_rank_work=svd_work,
                power_iters=power_iters, rank_expansions=rank_expansions,
                right_basis=pair['right_basis'], **{
                    k: pair[k] for k in ['rank', 'nuclear_x']})


def solve_idrs(g, r, s, *, gamma=1.0, sigma=0.5, relaxation=1.0,
               initial_rank=2, tol=1e-7, max_iters=5000, check_every=10,
               seed=0, exact=False, inner_solver="power", max_power_iters=5,
               warm_start=True):
    """Solve using graph-certified inexact spectral prox and Algorithm II.

    u is spectrally feasible; row clipping gives a primal LMO candidate.
    h(x)+f(x) is a valid dual upper bound at EACH iterate. Stop on the
    certified gap relative to the initial upper bound.
    Diagnostics use full SVDs, are timed separately, and are not evidence
    for a speedup from partial SVDs. Every accepted inner test is logged.
    """
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) == 0 or not np.isfinite(g).all():
        raise ValueError('g must be a nonempty finite matrix')
    if not all(np.isfinite(q) and q > 0 for q in (r, s, gamma, tol)):
        raise ValueError('r, s, gamma and tol must be finite and positive')
    if not 0 <= sigma < 1 or not 0 < relaxation < 2 or initial_rank < 1:
        raise ValueError('require 0<=sigma<1, 0<relaxation<2, initial_rank>=1')
    if inner_solver not in ('power', 'subspace') or max_power_iters < 1:
        raise ValueError('inner_solver must be power/subspace and max_power_iters>=1')
    if max_iters < 1 or check_every < 1:
        raise ValueError('iteration counts must be positive')
    start = perf_counter()
    scale = min(s * np.linalg.svd(g, compute_uv=False).sum(),
                r * np.linalg.norm(g, axis=1).sum())
    certificate_seconds = perf_counter() - start
    w_best, lower, upper = np.zeros_like(g), 0.0, scale
    z = np.zeros_like(g)
    omega = np.random.default_rng(seed).standard_normal((g.shape[1], min(g.shape)))
    history, inner_history, solve_seconds = [], [], 0.0
    total_attempts, full_fallbacks, svd_rank_work = 0, 0, 0
    total_power_iters, total_rank_expansions, warm_basis = 0, 0, None
    if scale == 0:
        return dict(w=w_best, lower=0.0, upper=0.0, relative_gap=0.0,
                    iterations=0, converged=True, history=[], inner_history=[],
                    solve_seconds=0.0, certificate_seconds=certificate_seconds,
                    total_attempts=0, full_fallbacks=0, svd_rank_work=0,
                    total_power_iters=0, total_rank_expansions=0)
    converged = False
    for k in range(1, max_iters + 1):
        start = perf_counter()
        nonzero_input = np.linalg.norm(z) > 0
        step = idrs_step(z, g, r, s, gamma=gamma, sigma=sigma,
                         relaxation=relaxation, initial_rank=initial_rank,
                         omega=omega, exact=exact, inner_solver=inner_solver,
                         max_power_iters=max_power_iters,
                         warm_basis=warm_basis if warm_start else None)
        z = step['z_next']
        total_attempts += step['attempts']
        full_fallbacks += step['rank'] == min(g.shape)
        svd_rank_work += step['svd_rank_work']
        total_power_iters += step['power_iters']
        total_rank_expansions += step['rank_expansions']
        if warm_start and inner_solver == 'power' and not exact and nonzero_input:
            # Carry dominant right directions; avoid immediately locking into
            # full rank after a fallback. Enrichment remains error-controlled.
            keep = min(step['rank'], max(1, min(g.shape) - 1))
            warm_basis = step['right_basis'][:, :keep]
        inner_history.append(dict(iteration=k, rank=step['rank'],
            attempts=step['attempts'], power_iters=step['power_iters'],
            rank_expansions=step['rank_expansions'], error_norm=step['error_norm'],
            error_denominator=step['error_denominator'],
            relative_error=step['relative_error'], alpha=step['alpha'],
            exact_roundoff=step['exact_roundoff']))
        solve_seconds += perf_counter() - start
        if k == 1 or k % check_every == 0 or k == max_iters:
            start = perf_counter()
            # u belongs to partial h(x); it has spectral norm <=s already.
            w = row_clip(step['u'], r)
            candidate = metrics(g, w, r, s)
            if candidate['row_ratio'] > 1 + 1e-10 or candidate['spectral_ratio'] > 1 + 1e-10:
                raise RuntimeError('iDRS graph pair or feasibility repair failed')
            if candidate['score'] > lower:
                w_best, lower = w.copy(), candidate['score']
            upper = min(upper, s * step['nuclear_x']
                        + r * np.linalg.norm(g - step['x'], axis=1).sum())
            gap = upper - lower
            if gap < -1e-10 * scale:
                raise RuntimeError('iDRS negative primal-dual gap')
            relative_gap = max(0.0, gap) / scale
            history.append(dict(iteration=k, lower=lower, upper=upper,
                relative_gap=relative_gap, solve_seconds=solve_seconds,
                natural_residual=float(np.linalg.norm(step['v'])),
                row_ratio=candidate['row_ratio'], spectral_ratio=candidate['spectral_ratio']))
            certificate_seconds += perf_counter() - start
            if relative_gap <= tol:
                converged = True
                break
    return dict(w=w_best, lower=lower, upper=upper, relative_gap=relative_gap,
                iterations=k, converged=converged, history=history,
                inner_history=inner_history, solve_seconds=solve_seconds,
                certificate_seconds=certificate_seconds, total_attempts=total_attempts,
                full_fallbacks=full_fallbacks, svd_rank_work=svd_rank_work,
                total_power_iters=total_power_iters, total_rank_expansions=total_rank_expansions)


# =========================================================================== DRS with a rank-k subspace clip

def low_rank_gradient(rows, cols, decay, seed, row_range=(0.2, 5.0)):
    """Unit-Frobenius G with singular values i**-decay and uneven row norms."""
    rng = np.random.default_rng(seed)
    width = min(rows, cols)
    left = np.linalg.qr(rng.standard_normal((rows, width)))[0]
    right = np.linalg.qr(rng.standard_normal((cols, width)))[0]
    g = (left * np.arange(1, width + 1) ** (-decay)) @ right.T
    g *= rng.permutation(np.geomspace(*row_range, rows))[:, None]
    return g / np.linalg.norm(g)


class SubspaceClip:
    """Inexact SpectralClip_s, the resolvent of the normal cone N of {||W||_op <= s}.

    With Q = qr(Z V) and the small SVD Q^T Z = U_k diag(a) R^T, set L = Q U_k,
    w = L diag(min(a, s)) R^T + c T and n = L diag((a - s)_+) R^T / gamma,
    where T = Z (I - R R^T) is orthogonal to L and R on both sides and
    c = min(1, s/||T||_F). Then ||w||_op <= s and <n, w> = s||n||_*, so
    n is in N(w) exactly for ANY basis. The resolvent error
    eps = (Z - w)/gamma - n = [(I - Q Q^T) Z R R^T + (1 - c) T] / gamma
    has a cross part (reduced by subspace sweeps) and a tail part (reduced only
    by more rank); the parts are orthogonal. At rank min(shape) both vanish
    and w = SpectralClip_s(Z).
    """

    def __init__(self, shape, gamma, s, rank, seed=0):
        if not 1 <= rank <= min(shape):
            raise ValueError('rank must satisfy 1 <= rank <= min(shape)')
        self.gamma, self.s, self.full = gamma, s, min(shape)
        self.rng = np.random.default_rng(seed)
        self.basis = np.linalg.qr(self.rng.standard_normal((shape[1], rank)))[0]
        self.products = 0  # matrix-vector products with Z or Z^T

    @property
    def rank(self):
        return self.basis.shape[1]

    def pair(self, z):
        gamma, s = self.gamma, self.s
        q = np.linalg.qr(z @ self.basis)[0]
        small_left, values, right_t = np.linalg.svd(q.T @ z, full_matrices=False)
        left, right = q @ small_left, right_t.T
        z_right = z @ right
        self.products += 3 * self.rank
        excess = np.maximum(values - s, 0.0)
        tail = z - z_right @ right.T
        tail_norm = np.sqrt(max(np.sum(z * z) - np.sum(z_right * z_right), 0.0))
        c = min(1.0, s / tail_norm) if tail_norm > 0 else 1.0
        self._next = right
        return dict(w=(left * np.minimum(values, s)) @ right.T + c * tail,
                    n=(left * (excess / gamma)) @ right.T,
                    cross_error=float(np.linalg.norm(z_right - left * values)) / gamma,
                    tail_error=(1 - c) * tail_norm / gamma,
                    nuclear_n=float(excess.sum()) / gamma, rank=self.rank)

    def sweep(self):
        """Advance the basis by the right Ritz vectors of the last pair."""
        self.basis = self._next

    def grow(self):
        """Keep the current Ritz basis and double the rank with random directions."""
        self.sweep()
        extra = min(self.rank, self.full - self.rank)
        new = self.rng.standard_normal((self.basis.shape[0], extra))
        new -= self.basis @ (self.basis.T @ new)
        self.basis = np.linalg.qr(np.hstack([self.basis, new]))[0]


def exact_pair(z, gamma, s):
    """Exact SpectralClip_s by a full SVD, in the same pair format."""
    left, values, right_t = np.linalg.svd(z, full_matrices=False)
    excess = np.maximum(values - s, 0.0)
    return dict(w=(left * np.minimum(values, s)) @ right_t,
                n=(left * (excess / gamma)) @ right_t,
                cross_error=0.0, tail_error=0.0,
                nuclear_n=float(excess.sum()) / gamma, rank=min(z.shape))


def error_norm(pair):
    return float(np.hypot(pair['cross_error'], pair['tail_error']))


def row_step(y, g, r, gamma):
    """prox_{gamma f}(Y) = RowClip_r(Y + gamma G), exact."""
    return row_clip(y + gamma * g, r)


def idrs_quantities(z, pair, g, r, gamma):
    """Exact row resolvent at the reflected point and Algorithm II's test."""
    w, n = pair['w'], pair['n']
    reflected = w - gamma * n
    y = (reflected - row_step(reflected, g, r, gamma)) / gamma
    v = n + y
    z_bar = w - gamma * y
    denominator = max(np.linalg.norm(v), np.linalg.norm(z - z_bar) / gamma)
    return v, z_bar, float(denominator)


def solve(g, r, s, *, method, rank=None, gamma=1.0, sigma=0.9, tol=1e-6,
          max_iters=3000, max_trials=100, check_every=10, seed=0):
    """method='idrs' (adaptive rank), 'fixed' (fixed rank) or 'exact' (full SVD).

    All report RowClip_r(w) as the feasible candidate. The dual bound
    s||X||_* + r||G - X||_{2,1}, valid for any X by weak duality, is evaluated
    at X = n, the normal-cone element (equal to the dual solution at a fixed
    point). All stop on the same certified relative gap. Certificates are
    timed separately.
    """
    g = np.asarray(g, dtype=np.float64)
    if method not in ('idrs', 'fixed', 'exact'):
        raise ValueError('method must be idrs/fixed/exact')
    if not all(np.isfinite(v) and v > 0 for v in (r, s, gamma, tol)):
        raise ValueError('r, s, gamma, tol must be finite and positive')
    if not 0 < sigma < 1 or min(max_iters, max_trials, check_every) < 1:
        raise ValueError('0 < sigma < 1 and positive limits required')
    kernel = None if method == 'exact' else SubspaceClip(g.shape, gamma, s, rank, seed)
    scale = min(s * np.linalg.svd(g, compute_uv=False).sum(),
                r * np.linalg.norm(g, axis=1).sum())
    z, w_best = np.zeros_like(g), np.zeros_like(g)
    lower, upper, gap = 0.0, scale, 1.0
    seconds, rejected, history = 0.0, 0, []
    status = 'outer_limit'
    roundoff = 100 * np.finfo(float).eps
    if scale == 0:
        return dict(status='converged', converged=True, iterations=0, lower=0.0,
                    upper=0.0, gap=0.0, w=w_best, seconds=0.0, products=0,
                    rank=rank or min(g.shape), rejected=0, history=[])
    for k in range(1, max_iters + 1):
        start = perf_counter()
        if method == 'exact':
            pair = exact_pair(z, gamma, s)
        elif method == 'fixed':
            pair = kernel.pair(z)
            kernel.sweep()
        else:
            for _ in range(max_trials):
                pair = kernel.pair(z)
                v, z_bar, denominator = idrs_quantities(z, pair, g, r, gamma)
                error = error_norm(pair)
                exact = kernel.rank == kernel.full and \
                    error <= roundoff * max(np.linalg.norm(z) / gamma, s, 1.0)
                if error <= sigma * denominator or exact:
                    kernel.sweep()
                    break
                rejected += 1
                if pair['tail_error'] >= pair['cross_error'] and kernel.rank < kernel.full:
                    kernel.grow()
                else:
                    kernel.sweep()
            else:
                status = 'trial_limit'
        seconds += perf_counter() - start
        stationary = method == 'idrs' and status == 'outer_limit' and \
            denominator <= roundoff * max(np.linalg.norm(z) / gamma, s, 1.0)
        if k == 1 or k % check_every == 0 or k == max_iters or stationary or \
                status != 'outer_limit':
            candidate = row_clip(pair['w'], r)
            score = float(np.sum(g * candidate))
            if score > lower:
                lower, w_best = score, candidate
            upper = min(upper, s * pair['nuclear_n']
                        + r * np.linalg.norm(g - pair['n'], axis=1).sum())
            if upper - lower < -1e-10 * scale:
                raise RuntimeError('invalid objective certificate')
            gap = max(0.0, upper - lower) / scale
            history.append(dict(iteration=k, gap=gap, seconds=seconds,
                                products=kernel.products if kernel else None,
                                rank=pair['rank']))
            if gap <= tol:
                status = 'converged'
            if status != 'outer_limit' or stationary:
                break
        start = perf_counter()
        if method == 'idrs':
            z = z - (float(np.sum(v * (z - z_bar))) / float(np.sum(v * v))) * v
        else:
            # Plain DRS; with an inexact clip this is z - gamma n, not the graph point w.
            clipped = z - gamma * pair['n']
            z = z + row_step(2 * clipped - z, g, r, gamma) - clipped
        seconds += perf_counter() - start
    return dict(status=status, converged=status == 'converged', iterations=k,
                lower=lower, upper=upper, gap=gap, w=w_best, seconds=seconds,
                products=kernel.products if kernel else None,
                rank=pair['rank'], rejected=rejected, history=history)
