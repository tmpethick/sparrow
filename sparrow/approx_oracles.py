"""Oracles for the joint row-norm and spectral-norm set K_{r,s} (the paper's section "The joint
row-norm and spectral-norm oracle" and its Sparrow algorithm).

    K_{r,s} = {W : ||W||_{2->inf} <= r, ||W||_op <= s}

The directions, each the (approximate) maximizer of <G, D> over its set, for G a momentum:

    spectral(G, s)        Spectral_s(G) = s U V^T              LMO of the spectral ball (Muon)
    row(G, r)             Row_r(G): every row scaled to norm r  LMO of the row-norm ball
    sparrow(G, r, s)      RowClip_r Spectral_s Row_r (G)       Sparrow
    rowclip_polar(G, r, s)  RowClip_r Spectral_s (G)           U_1
    aurora_update(G, r, s)  Aurora (Algorithm 3 of Dewulf et al. 2026), not in K_{r,s} exactly
    drs_lmo(G, r, s)      the joint LMO of K_{r,s} by Douglas-Rachford splitting, warm-startable
    sparrow_c(G, r, s, rho)  RowClip_r Spectral_s RowClip_c (G), rho in [0, 1] sets c (rho = 0: Sparrow, 1: U_1)

`DenseGeometry` holds one set for projected training (experiments/synthetic.py): `lmo(g)` returns
argmin_{W in K} <g, W> and `project(w)` lands in K in one pass (spectral clip, then row clip). Accurate
solvers of the joint LMO for benchmarking are in `sparrow.iterative_solvers`.
"""

import math

import jax
import jax.numpy as jnp

SVD_MARGIN = 1 + 1e-5  # float32 slack on computed singular values


def sigma_max(m):
    return jnp.linalg.svd(m.astype(jnp.float32), compute_uv=False)[0] * SVD_MARGIN


def row_max(m):
    return jnp.max(jnp.linalg.norm(m.astype(jnp.float32), axis=1))


def _svd(m):
    """Thin SVD. LAPACK's default divide-and-conquer (gesdd) occasionally fails to converge
    on ill-conditioned inputs and returns NaN (seen on momentum matrices during training), so
    fall back to the slower QR-based gesvd in that case."""
    out = jnp.linalg.svd(m, full_matrices=False)
    ok = jnp.all(jnp.array([jnp.all(jnp.isfinite(a)) for a in out]))
    return jax.lax.cond(ok, lambda: tuple(out), lambda: tuple(jax.lax.linalg.svd(
        m, full_matrices=False, algorithm=jax.lax.linalg.SvdAlgorithm.QR)))


def _spectral_clip(m, cap):
    u, s, vt = _svd(m)
    return (u * jnp.minimum(s, cap)) @ vt


def _row_clip(m, cap):
    return m * jnp.minimum(1.0, cap / jnp.maximum(jnp.linalg.norm(m, axis=1, keepdims=True), 1e-12))


def _row_normalize(m, r=1.0):
    return r * m / jnp.maximum(jnp.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def _polar(m):
    u, _, vt = _svd(m)
    return u @ vt


def spectral(G, s):
    """Spectral_s(G) = s Polar(G), the LMO of the spectral ball of radius s (Muon's direction)."""
    return s * _polar(G)


def row(G, r):
    """Row_r(G): every row of G rescaled to norm r, the LMO of the row-norm ball ||W||_{2->inf} <= r."""
    return _row_normalize(G, r)


def rowclip_polar(G, r, s):
    """RowClip_r(s Polar(G)) (the paper's one-shot approximation U_1): start from the spectral
    LMO and clip its rows. It is in K_{r,s} because row clipping never increases the spectral
    norm."""
    return _row_clip(s * _polar(G), r)


def specclip_rownormalize(G, r, s):
    """SpecClip_s(r RowNormalize(G)): the mirror image of rowclip_polar. It starts from the row
    LMO and clips its singular values. It is in K_{r,s} because a spectral clip never
    increases a row norm."""
    return _spectral_clip(_row_normalize(G, r), s)


def aurora_update(G, r, s, iters=2, beta=0.5):
    """Aurora as in Algorithm 3 of Dewulf et al. (2026), arXiv 2606.27715: from X_0 = G / ||G||_F and D = I,
    repeat r_k = row norms of X_k, D <- D^beta diag(r_k)^(1 - beta), X_{k+1} = Polar(sqrt(n / m) D^{-1} X_k);
    return s X_K. Their speedrun default is K = 2, beta = 0.5. Ends on Stiefel, so it is not in K_{r,s} exactly
    (r enters only through the target of equal rows, which the iteration does not use explicitly)."""
    m, n = G.shape
    X = G / jnp.maximum(jnp.linalg.norm(G), 1e-12)
    D = jnp.ones((m, 1), G.dtype)
    for _ in range(iters):
        D = D ** beta * jnp.maximum(jnp.linalg.norm(X, axis=1, keepdims=True), 1e-12) ** (1 - beta)
        X = _polar(math.sqrt(n / m) * X / D)
    return s * X


def sparrow_c(G, r, s, rho):
    """RowClip_r(Spectral_s(RowClip_c(G))) with c = rho / (1 - rho) * gbar, gbar = ||G||_F / sqrt(m) the RMS row norm
    (Sparrow with a row-clipping threshold): one step of forward-backward on the dual min_{A+B=G} r||A||_{2,1} + s||B||_*, started from
    A = RowShrink_c(G). Only the rows of G above c are equalized before the polar factor. Since Spectral_s ignores
    positive scaling, RowClip_c(G) is replaced by its multiple with rows G_i / max((1 - rho) ||G_i||, rho gbar), which is
    defined on all of [0, 1] and has norm of order one whatever the scale of G: rho = 0 is Sparrow (Row(G)), rho = 1/2
    is c = gbar, rho = 1 is U_1 = RowClip_r Spectral_s (G). Also returns its score <G, D> / (r ||G||_{2,1}), the value
    relative to the row-ball LMO's, which does not depend on rho and so peaks where <G, D> does (for adapting rho)."""
    rn = jnp.linalg.norm(G, axis=1, keepdims=True)
    gbar = jnp.sqrt(jnp.mean(rn ** 2))
    B = G / jnp.maximum(jnp.maximum((1 - rho) * rn, rho * gbar), 1e-30)
    D = _row_clip(s * _polar(B), r)
    return D, jnp.sum(G * D) / jnp.maximum(r * jnp.sum(rn), 1e-30)


def sparrow(G, r, s):
    """Sparrow direction: RowClip_r(s Polar(Row_r(G))). One polar
    factor; the row maps are cheap. In K_{r,s}, since row clipping never increases the spectral
    norm."""
    return rowclip_polar(_row_normalize(G, r), r, s)


def drs_lmo(G, r, s, Z=None, iters=20):
    """Joint LMO argmax_{W in K_{r,s}} <G, W> by Douglas-Rachford splitting (DRS, gamma = 1),
    warm-started from the state Z of a previous call. G is first scaled to unit spectral norm (the LMO is
    invariant to positive scaling), so warm starts carry over between nearby gradients.

        W <- SpectralClip_s(Z),  Z <- Z + RowClip_r(2W - Z + G) - W

    This is ADMM on the dual min_{A+B=G} r||A||_{2,1} + s||B||_* (the paper's dual problem), with Z = B - U.
    The SVD of Z also gives the dual split: B = Z - SpectralClip_s(Z) = SVT_s(Z) and A = G - B.
    Returns W = RowClip_r(SpectralClip_s(Z)) (in K_{r,s}), the state Z, and the certified relative gap
    (dual - <G, W>) / dual with dual = r||G - B||_{2,1} + s||B||_*."""
    Gn = G / jnp.maximum(sigma_max(G), 1e-30)
    Z = jnp.zeros_like(Gn) if Z is None else Z

    def step(_, Z):
        W = _spectral_clip(Z, s)
        return Z + _row_clip(2 * W - Z + Gn, r) - W
    Z = jax.lax.fori_loop(0, iters, step, Z)
    u, sv, vt = _svd(Z)
    W = _row_clip((u * jnp.minimum(sv, s)) @ vt, r)
    B = (u * jnp.maximum(sv - s, 0.0)) @ vt
    dual = r * jnp.sum(jnp.linalg.norm(Gn - B, axis=1)) + s * jnp.sum(jnp.maximum(sv - s, 0.0))
    gap = (dual - jnp.sum(Gn * W)) / jnp.maximum(dual, 1e-30)
    return W, Z, gap


class DenseGeometry:
    """K_{r,s} = {W : ||W||_{2->inf} <= row, ||W||_op <= spec} for a (fanout, fanin) matrix,
    following the paper's section "The joint row-norm and spectral-norm oracle". At least one
    constraint is required. A layer without a row constraint is a spectral ball whose radius sets its scale.

    lmo(g) = argmin_{W in K} <g, W> = -LMO_paper(g). The regime comes from the norm chain
    ||W||_{2->inf} <= ||W||_op <= sqrt(d) ||W||_{2->inf} (d = number of rows):
      spectral  (row is None, or s <= r)          -s Polar(g)
      row       (spec is None, or s >= sqrt(d) r) -r RowNormalize(g)
      mixed, oracle:
        "rowclip"  -rowclip_polar(g, r, s)            the paper's one-shot approximation
        "specclip" -specclip_rownormalize(g, r, s)    the mirror candidate
        "aurora"   -aurora_update(g, r, s)            Aurora, Algorithm 3 of Dewulf et al. (2026), K = 2, beta = 1/2
                                                      (near-equal rows on Stiefel, not in K exactly)
        "sparrow"  -sparrow(g, r, s)                  RowClip_r(s Polar(Row_r(g)))
        "sparrow_c" -sparrow_c(g, r, s, rho)          RowClip_r(s Polar(RowClip_c(g))), rho in [0, 1] sets c
        "best"     the better of rowclip_polar and specclip_rownormalize. Both are feasible,
                   and the choice is exact whenever one of the two constraints is inactive at the
                   optimum.
    project: spectral clip, then row clip. It lands in K_{r,s} in one pass."""

    def __init__(self, spec=1.0, row=None, oracle="rowclip", rho=0.5):
        assert spec is not None or row is not None, "give a spectral and/or row radius"
        assert oracle in ("rowclip", "specclip", "best", "aurora", "sparrow", "sparrow_c"), oracle
        self.spec, self.row, self.oracle, self.rho = spec, row, oracle, rho

    def regime(self, rows):
        s, r = self.spec, self.row
        if r is None or (s is not None and s <= r):
            return "spectral"
        if s is None or s >= math.sqrt(rows) * r:
            return "row"
        return "mixed"

    def lmo(self, g, rho=None):
        """rho: overrides self.rho of oracle "sparrow_c" (e.g. a per-step schedule); the other oracles ignore it."""
        s, r = self.spec, self.row
        regime = self.regime(g.shape[0])
        if regime == "spectral":
            return -s * _polar(g)
        if regime == "row":
            return -_row_normalize(g, r)
        if self.oracle == "specclip":
            return -specclip_rownormalize(g, r, s)
        if self.oracle == "aurora":
            return -aurora_update(g, r, s)
        if self.oracle == "sparrow":
            return -sparrow(g, r, s)
        if self.oracle == "sparrow_c":
            return -sparrow_c(g, r, s, self.rho if rho is None else rho)[0]
        a = rowclip_polar(g, r, s)
        if self.oracle == "best":
            b = specclip_rownormalize(g, r, s)
            a = jnp.where(jnp.sum(g * a) >= jnp.sum(g * b), a, b)
        return -a

    def project(self, w):
        if self.spec is not None:
            w = _spectral_clip(w, self.spec)
        if self.row is not None:
            w = _row_clip(w, self.row)
        return w
