"""Checks for the synthetic classification experiment (experiments/synthetic.py)."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments import synthetic as R

D = 16


def test_squash():
    x, y = jax.random.normal(jax.random.PRNGKey(0), (2, 200, D)) * 5
    sx, sy = R.squash(x, 0.7), R.squash(y, 0.7)
    assert jnp.all(jnp.linalg.norm(sx - sy, axis=-1) <= jnp.linalg.norm(x - y, axis=-1) + 1e-6)
    assert jnp.all(jnp.linalg.norm(sx, axis=-1) <= math.sqrt(D) * 0.7 + 1e-5)


def test_teachers_and_student_width():
    Wt, kappa, b0 = R.make_teacher("gauss_4", 16, jax.random.PRNGKey(1))
    assert [w.shape for w in Wt] == [(10, 16), (16, 64), (64, 16), (64, 16)]
    assert 0.3 < kappa < 0.6 and b0 == 1 / 4
    assert R.make_teacher("few_8", 16, jax.random.PRNGKey(1))[1] > kappa   # concentrated
    gv, wo = R.family_sets("ours_rowspec", b0, 0.5, 16, 32)
    W = R.train(Wt, gv, jax.random.PRNGKey(0), 16, 3, 0.1, batch=32, wo_geo=wo, m=32)
    assert [w.shape for w in W[1:]] == [(16, 32), (32, 16), (32, 16)]
    assert float(R.row_max(W[2])) <= gv.row * (1 + 1e-4)
    assert np.isfinite(R.certificate(W))


def test_sparrow_geometry():
    """--impl sparrow_jax: Muon's direction is unchanged, the joint set gets the library's Sparrow
    oracle at the experiment's own radii (r = sqrt(B kappa), not sparrow_jax's default), and the
    projection is DenseGeometry's."""
    from sparrow.approx_oracles import DenseGeometry
    g = jax.random.normal(jax.random.PRNGKey(0), (256, 64))
    for spec, row, ref in [(0.3, None, "rowclip"), (0.5, 0.2, "sparrow"), (0.5, 0.45, "sparrow")]:
        geo = DenseGeometry(spec=spec, row=row)
        sp = R.SparrowGeometry(geo)
        assert np.allclose(sp.lmo(g), DenseGeometry(spec=spec, row=row, oracle=ref).lmo(g), atol=1e-5)
        assert np.allclose(sp.project(g), geo.project(g))
