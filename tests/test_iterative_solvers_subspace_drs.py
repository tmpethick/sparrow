"""Inexact clip pairs, the DRS recurrence, and certificates against exact solves."""
import numpy as np
import pytest

from sparrow.iterative_solvers import solve_idrs, spectral_clip
from sparrow.iterative_solvers import (SubspaceClip, error_norm,
                                                low_rank_gradient, solve)


@pytest.mark.parametrize('shape,rank', [((30, 12), 3), ((12, 30), 3), ((30, 12), 12)])
@pytest.mark.parametrize('gamma', [0.4, 1.0])
def test_clip_pair_is_exact_normal_cone_point_with_measured_error(shape, rank, gamma):
    rng = np.random.default_rng(3)
    z = rng.normal(size=shape)
    z[:, -1] = 0  # rank deficient
    s = 0.9
    kernel = SubspaceClip(shape, gamma, s, rank, seed=1)
    for _ in range(3):
        pair = kernel.pair(z)
        kernel.sweep()
    w, n = pair['w'], pair['n']
    # n in N(w) for the spectral ball: w feasible and <n, w> = s ||n||_*.
    assert np.linalg.svd(w, compute_uv=False)[0] <= s + 1e-12
    nuclear = np.linalg.svd(n, compute_uv=False).sum()
    assert np.sum(n * w) == pytest.approx(s * nuclear, abs=1e-10)
    assert pair['nuclear_n'] == pytest.approx(nuclear, abs=1e-10)
    epsilon = (z - w) / gamma - n
    assert np.linalg.norm(epsilon) == pytest.approx(error_norm(pair), abs=1e-10)
    assert kernel.products == 3 * 3 * rank


def test_full_rank_pair_is_exact_clip_and_sweeps_reduce_cross_error():
    rng = np.random.default_rng(5)
    left, right = (np.linalg.qr(rng.normal(size=(n, 8)))[0] for n in (20, 8))
    z = (left * np.geomspace(10, 0.1, 8)) @ right.T
    gamma, s = 0.5, 1.0
    pair = SubspaceClip(z.shape, gamma, s, 8).pair(z)
    assert np.allclose(pair['w'], spectral_clip(z, s), atol=1e-12)
    assert error_norm(pair) < 1e-12
    kernel = SubspaceClip(z.shape, gamma, s, 3)
    errors = []
    for _ in range(6):
        errors.append(kernel.pair(z)['cross_error'])
        kernel.sweep()
    assert errors[-1] < 0.1 * errors[0]
    kernel.grow()
    assert kernel.rank == 6
    assert np.allclose(kernel.basis.T @ kernel.basis, np.eye(6), atol=1e-12)


def test_fixed_rank_uses_plain_drs_recurrence(monkeypatch):
    g, gamma, r, s = low_rank_gradient(15, 6, 1.0, 0), 0.7, 0.3, 1.0
    calls = []
    original = SubspaceClip.pair

    def record(self, z):
        pair = original(self, z)
        calls.append((z.copy(), pair['n'].copy()))
        return pair
    monkeypatch.setattr(SubspaceClip, 'pair', record)
    solve(g, r, s, method='fixed', rank=2, gamma=gamma, max_iters=5, tol=1e-14)
    assert np.allclose(calls[0][0], 0)
    for (z, n), (z_next, _) in zip(calls, calls[1:]):
        clipped = z - gamma * n
        reflected = 2 * clipped - z + gamma * g
        norms = np.linalg.norm(reflected, axis=1, keepdims=True)
        row = reflected / np.maximum(1, norms / r)
        assert np.allclose(z_next, z + row - clipped, atol=1e-13)


@pytest.mark.parametrize('gamma', [0.5, 1.0, 2.0])
def test_exact_drs_matches_separate_exact_drs(gamma):
    # solve's exact and full-rank DRS against the separate implementation in solve_idrs, which steps
    # with the nuclear-norm prox of the dual split: primal step gamma is its step 1/gamma.
    g, r, s = low_rank_gradient(24, 8, 1.0, 2), 1 / 3, 1.0
    reference = solve_idrs(g, r, s, exact=True, gamma=1 / gamma, tol=1e-8, max_iters=5000)
    drs = solve(g, r, s, method='exact', gamma=gamma, tol=1e-8, max_iters=5000)
    full_rank = solve(g, r, s, method='fixed', rank=8, gamma=gamma, tol=1e-8, max_iters=5000)
    assert reference['converged'] and drs['converged'] and full_rank['converged']
    assert drs['iterations'] == reference['iterations'] == full_rank['iterations']
    assert drs['lower'] == pytest.approx(reference['lower'], abs=1e-9)
    assert full_rank['lower'] == pytest.approx(reference['lower'], abs=1e-9)


@pytest.mark.parametrize('shape', [(40, 12), (12, 40)])
@pytest.mark.parametrize('sigma,gamma', [(0.5, 1.0), (0.9, 1.0), (0.9, 0.5)])
def test_idrs_certifies_the_exact_optimum_from_low_rank(shape, sigma, gamma):
    g, r, s = low_rank_gradient(*shape, 1.0, 4), 1 / 4, 1.0
    reference = solve_idrs(g, r, s, exact=True, tol=1e-9, max_iters=20000)
    result = solve(g, r, s, method='idrs', rank=1, sigma=sigma, gamma=gamma, tol=1e-7)
    assert result['converged'] and result['rejected'] > 0
    assert result['lower'] <= reference['upper'] + 1e-10
    assert reference['lower'] <= result['upper'] + 1e-10
    assert abs(result['lower'] - reference['lower']) <= 2e-7 * reference['upper']


def test_too_small_fixed_rank_stalls_but_idrs_grows():
    g, r, s = low_rank_gradient(120, 30, 1.0, 0), 1 / 4, 1.0
    stalled = solve(g, r, s, method='fixed', rank=2, tol=1e-6, max_iters=400)
    adaptive = solve(g, r, s, method='idrs', rank=2, tol=1e-6)
    assert not stalled['converged'] and stalled['gap'] > 1e-3
    assert adaptive['converged'] and adaptive['rank'] > 2


def test_zero_and_invalid_inputs():
    result = solve(np.zeros((4, 3)), 0.5, 1, method='idrs', rank=2)
    assert result['converged'] and result['iterations'] == 0
    for options in [dict(rank=0), dict(rank=4), dict(gamma=0), dict(sigma=1),
                    dict(max_iters=0), dict(method='wrong')]:
        kwargs = dict(method='fixed', rank=2)
        kwargs.update(options)
        with pytest.raises(ValueError):
            solve(np.ones((4, 3)), 0.5, 1, **kwargs)
