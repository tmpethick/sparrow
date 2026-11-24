"""Sparrow: minimal JAX reference.

For a weight W of shape (m, d) = (fanout, fanin), applied as x @ W.T,
`sparrow(G, kind)` is

    kind="spectral"    Spectral_s(G) = s U V^T,  G = U S V^T
                       Muon; attention, MLP, SwiGLU W_o
    kind="row"         Row_r(G): each row rescaled to norm r
                       attention and SSM gates W_a
    kind="gate_value"  RowClip_r(Spectral_s(Row_r(G)))
                       SwiGLU W_g, W_v; ReLU^2 W_1 (Sparrow)

with r = s min(1, sqrt(d / m)) and RowClip_r capping each row at norm r
("gate_value" is Spectral_s for m <= d).

With rho in (0, 1], "gate_value" is RowClip_r(Spectral_s(RowClip_c(G)))
instead (Sparrow with a row-clipping threshold), with c = rho / (1 - rho) times the RMS
row norm of G: only the rows above c are equalized. rho = 0 is Sparrow
and rho = 1 is RowClip_r(Spectral_s(G)).

s is a number, a function (m, d) -> s, or a name in SCALES.
polar="polarexpress" (default), "newtonschulz5" or "svd" (exact) computes U V^T;
the first two meet the spectral constraint only approximately.

`sparrow_optimizer` is an optax-style optimizer for the hidden matrices;
embeddings, head, gains and biases stay with AdamW (e.g. via
optax.multi_transform):

    M <- mu M + (1 - mu) grad
    D = sparrow(M, kind)
        (nesterov: D = sparrow(grad + mu (M - grad), kind))
    W <- (1 - lr wd) W - lr D

or, with weight_norm=True (Hyperball without normalizing D),

    W <- R N((1 - lr wd) W - lr c D),
    c = R / (s sqrt(min(m, d))),  R = ||W_0||_F,  N(X) = X / ||X||_F.

The paper's runs use nesterov=False.
"""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

EPS = 1e-12

# Muon's quintic Newton-Schulz coefficients (Jordan et al. 2024). They
# maximize the slope at zero, so the singular values end up near 1
# (roughly in [0.7, 1.2]) rather than at 1.
NS5 = (3.4445, -4.7750, 2.0315)

# Polar Express (Amsel, Persson, Musco, Gower 2025;
# github.com/NoahAmsel/PolarExpress, MIT): the minimax-optimal odd
# quintic per step, so it converges to U V^T. Coefficients from
#   optimal_composition(l=1e-3, num_iters=10, degree=5,
#                       safety_factor_eps=1e-2, cushion=0.02);
# steps past the end reuse the last one. The composition stays <= 1.131
# on [0, 1] after 5 steps and <= 1 + 2e-6 after 8.
POLAR_EXPRESS = [
    (8.237312490495558, -23.15774741455821, 16.68056841144592),
    (4.082441999064829, -2.8930477353325825, 0.525284925697564),
    (3.926347992254658, -2.854746803476532, 0.5318022422894996),
    (3.2982187133085143, -2.424541981026707, 0.48632008358844103),
    (2.297036943455258, -1.636625581259032, 0.4002628455953631),
    (1.8763805351440401, -1.234789657772224, 0.3589188750166845),
    (1.8564423485617227, -1.2132449880934026, 0.3568003487825125),
    (1.8564374057003257, -1.2132411397238774, 0.356801326215331),
    (1.856431167495673, -1.2132289091858204, 0.3567953314592478),
    (1.8749954775672242, -1.2499909551562645, 0.3749954775890401),
]
POLAR_METHODS = ("svd", "newtonschulz5", "polarexpress")
KINDS = ("gate_value", "row", "spectral")

# Shape-dependent spectral radii s(m, d) for a weight of shape
# (m, d) = (out, in).
SCALES = {
    # the newest Muon: max(1, sqrt(out / in))
    "muon": lambda m, d: max(1.0, (m / d) ** 0.5),
    # sqrt(out / in), the spectral condition of muP
    "mup": lambda m, d: (m / d) ** 0.5,
    # unit RMS entries (modded-nanoGPT record)
    "rms": lambda m, d: max(m, d) ** 0.5,
}


def spectral_radius(s, shape):
    """s for a matrix of this shape: s itself if a number, s(m, d) if a
    function, SCALES[s] if a name."""
    if isinstance(s, str):
        assert s in SCALES, (
            f"unknown scale {s!r}; use a number, a function (m, d) -> s, "
            f"or one of {list(SCALES)}")
        s = SCALES[s]
    return float(s(*shape)) if callable(s) else s


def _qr_polar(G):
    U, _, Vt = jax.lax.linalg.svd(
        G, full_matrices=False,
        algorithm=jax.lax.linalg.SvdAlgorithm.QR)
    return U @ Vt


def polar_factor(G, method="polarexpress", steps=5):
    """U V^T for G = U S V^T.

    "newtonschulz5" and "polarexpress" use matrix products only: `steps`
    odd quintic iterations X <- a X + (b A + c A^2) X, A = X X^T, on the
    wide orientation of G.
    """
    if method == "svd":
        # LAPACK's default SVD (gesdd) occasionally returns NaN on
        # ill-conditioned momentum matrices on CPU; fall back to the
        # slower QR-based algorithm then.
        U, _, Vt = jnp.linalg.svd(G, full_matrices=False)
        P = U @ Vt
        return jax.lax.cond(jnp.all(jnp.isfinite(P)),
                            lambda: P, lambda: _qr_polar(G))
    assert method in POLAR_METHODS, method
    if method == "newtonschulz5":
        coeffs, margin = [NS5] * steps, 1.0
    else:
        padded = POLAR_EXPRESS + POLAR_EXPRESS[-1:] * steps
        coeffs, margin = padded[:steps], 1.01
    tall = G.shape[0] > G.shape[1]
    X = G.T if tall else G
    # every singular value <= 1 / margin
    X = X / (margin * jnp.linalg.norm(X) + EPS)
    for a, b, c in coeffs:
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


def spectral(G, s, polar="polarexpress", polar_steps=5):
    return s * polar_factor(G, polar, polar_steps)


def _row_norms(G):
    return jnp.maximum(jnp.linalg.norm(G, axis=1, keepdims=True), EPS)


def row(G, r):
    return r * G / _row_norms(G)


def row_clip(G, r):
    return G * jnp.minimum(1.0, r / _row_norms(G))


def rowclip_c(G, rho):
    """RowClip_c(G), c = rho / (1 - rho) gbar with gbar the RMS row norm,
    up to a positive factor (which the polar factor ignores).

    Returns the rows G_i / max((1 - rho) ||G_i||, rho gbar), of norm about
    one at any rho in [0, 1].
    """
    rn = jnp.linalg.norm(G, axis=1, keepdims=True)
    gbar = jnp.sqrt(jnp.mean(rn ** 2))
    return G / jnp.maximum(jnp.maximum((1 - rho) * rn, rho * gbar), EPS)


def sparrow(G, kind="spectral", s=1.0, polar="polarexpress", polar_steps=5,
            r=None, rho=0.0):
    """The Sparrow direction of G.

    kind: "gate_value" (SwiGLU gate and value), "row" (attention or SSM
        gate) or "spectral".
    polar: how the polar factor is computed, "polarexpress" (default)
        or "newtonschulz5" (polar_steps iterations), "svd" (exact; see
        polar_factor).
    s: a number, a function (m, d) -> s, or a name in SCALES.
    r: the row radius, s * min(1, sqrt(d / m)) by default.
    rho in [0, 1]: the clip level of "gate_value" (rowclip_c; 0: Sparrow,
        1: RowClip_r Spectral_s).
    """
    m, d = G.shape
    s = spectral_radius(s, G.shape)
    r = s * min(1.0, (d / m) ** 0.5) if r is None else r
    if kind == "gate_value" and r < s:
        Y = row(G, r) if rho == 0 else rowclip_c(G, rho)
        return row_clip(spectral(Y, s, polar, polar_steps), r)
    if kind == "row":
        return row(G, r)
    assert kind in ("gate_value", "spectral"), kind
    return spectral(G, s, polar, polar_steps)


class GradientTransformation(NamedTuple):
    """The fields of optax.GradientTransformation."""
    init: Callable
    update: Callable


class SparrowState(NamedTuple):
    count: Any     # steps taken, for a schedule lr(count)
    momentum: Any  # M, a pytree like params
    radius: Any    # R = ||W_0||_F of every matrix (for weight_norm=True)


def sparrow_optimizer(lr=0.01, momentum=0.9, nesterov=True,
                      kind="spectral", s=1.0, weight_decay=0.0,
                      weight_norm=False, polar="polarexpress", polar_steps=5):
    """Sparrow for a pytree of 2-D weights stored (out, in).

    Transpose (in, out) kernels first, as in Flax. Returns
    init(params) -> state and update(grads, state, params) ->
    (updates, state), with params <- params + updates
    (optax.apply_updates), so it composes with optax.

    kind is one kind for every matrix, a pytree of kinds whose leaves
    line up with the matrices (the structure of params, or of the
    matrices alone when other leaves are masked out, as in
    optax.multi_transform), or a function params -> such a pytree, e.g.

        kind={"w_gate": "gate_value", "w_value": "gate_value",
              "w_out": "spectral"}

    s is the spectral radius (a number, a function (m, d) -> s of each
    matrix's shape, or a name in SCALES), and lr the step in units of it
    (as for Muon), or a schedule count -> lr. weight_decay is decoupled
    and scaled by lr, as in AdamW. polar and polar_steps choose how the
    polar factor is computed (see sparrow).
    """
    assert polar in POLAR_METHODS, polar
    spectral_radius(s, (1, 1))  # fail early on an unknown name

    def kinds(tree):
        leaves, treedef = jax.tree.flatten(tree)
        k = kind(tree) if callable(kind) else kind
        k = [k] * len(leaves) if isinstance(k, str) else jax.tree.leaves(k)
        assert len(k) == len(leaves), (
            f"{len(k)} kinds for {len(leaves)} matrices")
        return jax.tree.unflatten(treedef, k)

    def init(params):
        for k, p in zip(jax.tree.leaves(kinds(params)),
                        jax.tree.leaves(params)):
            assert k in KINDS, k
            assert p.ndim == 2, (
                "Sparrow is for matrices; "
                "use AdamW for the other parameters")
        return SparrowState(
            count=jnp.zeros([], jnp.int32),
            momentum=jax.tree.map(jnp.zeros_like, params),
            radius=jax.tree.map(jnp.linalg.norm, params))

    needs_params = weight_norm or weight_decay != 0

    def update(grads, state, params=None):
        assert params is not None or not needs_params, (
            "weight_norm or weight_decay needs params")
        step = lr(state.count) if callable(lr) else lr
        buf = jax.tree.map(lambda b, g: b + (1 - momentum) * (g - b),
                           state.momentum, grads)

        def direction(k, grad, b, R, p):
            g = grad + momentum * (b - grad) if nesterov else b
            # shapes are static, so this is a Python float under jit
            sk = spectral_radius(s, grad.shape)
            D = sparrow(g.astype(jnp.float32), k, sk, polar, polar_steps)
            decay = step * weight_decay * p if weight_decay else 0.0
            if not weight_norm:
                return (-step * D - decay).astype(grad.dtype)
            c = step * R / (sk * min(p.shape) ** 0.5)
            v = p - decay - c * D
            v = R * v / jnp.maximum(jnp.linalg.norm(v), EPS)
            return (v - p).astype(p.dtype)

        updates = jax.tree.map(direction, kinds(grads), grads, buf,
                               state.radius,
                               params if needs_params else grads)
        return updates, SparrowState(state.count + 1, buf, state.radius)

    return GradientTransformation(init, update)


if __name__ == "__main__":
    d, m = 64, 256  # SwiGLU block: m = 4d
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    params = dict(w_gate=jax.random.normal(k[0], (m, d)) / d ** 0.5,
                  w_value=jax.random.normal(k[1], (m, d)) / d ** 0.5,
                  w_out=jax.random.normal(k[2], (d, m)) / m ** 0.5)
    opt = sparrow_optimizer(
        lr=0.02,
        kind=dict(w_gate="gate_value", w_value="gate_value",
                  w_out="spectral"))  # W_o: Muon
    x = jax.random.normal(k[3], (512, d))
    y = jnp.roll(x, 1, axis=1)

    def loss_fn(p):
        h = jax.nn.silu(x @ p["w_gate"].T) * (x @ p["w_value"].T)
        return jnp.mean((h @ p["w_out"].T - y) ** 2)

    @jax.jit
    def train_step(p, state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, state = opt.update(grads, state, p)
        return jax.tree.map(jnp.add, p, updates), state, loss

    state = opt.init(params)
    for step in range(301):
        params, state, loss = train_step(params, state)
        if step % 100 == 0:
            print(f"step {step:3d}  loss {float(loss):.4f}")
