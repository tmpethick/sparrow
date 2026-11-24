"""Checks for the training oracles of the joint row/spectral set (sparrow.approx_oracles)."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from sparrow import approx_oracles as O


def _mixed_setup():
    rows, cols, c = 64, 16, 0.9
    r, s = c / math.sqrt(cols), 1.0  # SwiGLU symmetric budget: mixed since m/d > 1/c^2
    G = jax.random.normal(jax.random.PRNGKey(7), (rows, cols))
    return G / O.sigma_max(G), r, s


def test_joint_oracle_regimes_and_feasibility():
    """lmo(g) = argmin_K <g, W> = -LMO_paper(g); the paper's G is -gradient."""
    G, r, s = _mixed_setup()
    assert O.DenseGeometry(spec=s, row=r).regime(G.shape[0]) == "mixed"
    assert np.allclose(O.DenseGeometry(spec=0.5, row=1.0).lmo(-G), 0.5 * O._polar(G), atol=1e-5)
    assert np.allclose(jnp.linalg.norm(O.DenseGeometry(spec=100.0, row=0.1).lmo(G), axis=1), 0.1,
                       atol=1e-5)
    assert np.allclose(O.DenseGeometry(spec=None, row=0.1).lmo(G),
                       O.DenseGeometry(spec=100.0, row=0.1).lmo(G), atol=1e-6)
    for oracle in ("rowclip", "best", "sparrow"):
        W = O.DenseGeometry(spec=s, row=r, oracle=oracle).lmo(-G)
        assert float(O.sigma_max(W)) <= s * (1 + 1e-3) and float(O.row_max(W)) <= r * (1 + 1e-4)
    assert np.allclose(O.DenseGeometry(spec=s, row=r).lmo(-G), O.rowclip_polar(G, r, s))
    assert np.allclose(O.DenseGeometry(spec=s, row=r, oracle="sparrow").lmo(-G),
                       O.rowclip_polar(O._row_normalize(G), r, s), atol=1e-6)


def test_joint_oracle_against_exact_drs():
    """<G, W>: rowclip <= best <= exact (DRS) <= dual objective (weak duality)."""
    G, r, s = _mixed_setup()
    val = {o: float(jnp.sum(G * O.DenseGeometry(spec=s, row=r, oracle=o).lmo(-G)))
           for o in ("rowclip", "best")}
    W, _, gap = O.drs_lmo(G, r, s, iters=500)  # G has unit spectral norm, so drs_lmo leaves its scale
    exact = float(jnp.sum(G * W))
    dual = exact / (1 - float(gap))
    assert float(O.sigma_max(W)) <= s * (1 + 1e-3) and float(O.row_max(W)) <= r * (1 + 1e-4)
    assert val["rowclip"] <= val["best"] * (1 + 1e-4) <= exact * (1 + 1e-3)
    assert exact <= dual * (1 + 1e-4) and (dual - exact) / dual < 1e-2


def test_project_is_feasible():
    w = 3 * jax.random.normal(jax.random.PRNGKey(0), (64, 16))
    for geo in (O.DenseGeometry(spec=1.0), O.DenseGeometry(spec=None, row=0.2),
                O.DenseGeometry(spec=1.0, row=0.2)):
        p = geo.project(w)
        assert geo.spec is None or float(O.sigma_max(p)) <= geo.spec * (1 + 1e-3)
        assert geo.row is None or float(O.row_max(p)) <= geo.row * (1 + 1e-4)


def test_aurora_ends_on_stiefel_with_more_equal_rows():
    """Aurora (Algorithm 3 of Dewulf et al. 2026) returns s times a polar factor (all singular values s) whose row norms are
    closer to equal than those of Muon's s Polar(G); one iteration with beta = 0 is Polar(Row(G))."""
    m, d, s = 256, 64, 1.0
    r = s * math.sqrt(d / m)
    G = jax.random.normal(jax.random.PRNGKey(5), (m, d)) * jax.random.gamma(jax.random.PRNGKey(6), 0.5, (m, 1))
    W = O.aurora_update(G, r, s)
    np.testing.assert_allclose(jnp.linalg.svd(W, compute_uv=False), s, atol=1e-4)
    spread = lambda X: float(jnp.std(jnp.linalg.norm(X, axis=1)))
    assert spread(W) < 0.5 * spread(s * O._polar(G))
    np.testing.assert_allclose(O.aurora_update(G, r, s, iters=1, beta=0.0), s * O._polar(O._row_normalize(G)), atol=1e-5)


def test_sparrow_c_spans_sparrow_to_u1():
    """sparrow_c is Sparrow at rho = 0 and U_1 at rho = 1, without a branch, and agrees with RowClip_c up to scaling
    in between."""
    m, d, s = 256, 64, 1.0
    r = s * math.sqrt(d / m)
    G = jax.random.normal(jax.random.PRNGKey(10), (m, d)) @ jnp.diag(jnp.linspace(1.0, 0.05, d))
    np.testing.assert_allclose(O.sparrow_c(G, r, s, 0.0)[0], O.sparrow(G, r, s), atol=1e-5)
    np.testing.assert_allclose(O.sparrow_c(G, r, s, 1.0)[0], O.rowclip_polar(G, r, s), atol=1e-5)
    gbar = float(jnp.linalg.norm(G)) / math.sqrt(m)
    for rho in (0.3, 0.5, 0.7):
        np.testing.assert_allclose(O.sparrow_c(G, r, s, rho)[0],
                                   O.rowclip_polar(O._row_clip(G, rho / (1 - rho) * gbar), r, s), atol=1e-5)
    for scale in (1e-8, 1e8):  # the polar factor's input does not depend on the scale of G
        np.testing.assert_allclose(O.sparrow_c(scale * G, r, s, 0.3)[0], O.sparrow_c(G, r, s, 0.3)[0], atol=1e-5)
