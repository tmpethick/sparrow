"""Independent optima and certificate checks for the synthetic LMO benchmark."""
import numpy as np
import pytest
from sparrow.iterative_solvers import (best_of_two, dual_value, metrics,
    row_then_spectral, solve_idrs, spectral_then_row)


def exact_drs(g, r, s, **kwargs):
    return solve_idrs(g, r, s, exact=True, **kwargs)


def vector_optimum(g, r, s):
    # Independent KKT solution: w_i=min(r,t*g_i), choose t by bisection.
    lo, hi = 0.0, 1.0
    while np.linalg.norm(np.minimum(r, hi * g)) < s:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.linalg.norm(np.minimum(r, mid * g)) < s:
            lo = mid
        else:
            hi = mid
    return np.minimum(r, 0.5 * (lo + hi) * g)


def test_exact_drs_joint_vector_independent_optimum():
    g = np.array([[4.0], [2.0], [1.0], [0.2]])
    r, s = 1.0, 1.5
    optimum = float(np.sum(g * vector_optimum(g, r, s)))
    for gamma in [0.3, 1.0, 3.0]:
        result = exact_drs(g, r, s, gamma=gamma, tol=1e-9)
        assert result['converged']
        assert result['lower'] <= optimum + 1e-9
        assert result['upper'] >= optimum - 1e-9
        assert abs(result['lower'] - optimum) < 1e-7
        assert np.all(np.diff([h['lower'] for h in result['history']]) >= -1e-12)
        assert np.all(np.diff([h['upper'] for h in result['history']]) <= 1e-12)


@pytest.mark.parametrize('shape', [(7, 3), (3, 7)])
@pytest.mark.parametrize('rank_deficient', [False, True])
def test_clipping_feasibility_and_best_selection(shape, rank_deficient):
    rng = np.random.default_rng(17)
    g = rng.standard_normal(shape)
    if rank_deficient:
        g[0] = 0
        g[:, -1] = 0
    r, s = 0.6, 1.0
    a, b = spectral_then_row(g, r, s), row_then_spectral(g, r, s)
    best = best_of_two(g, r, s)
    for w in [a, b, best]:
        m = metrics(g, w, r, s)
        assert m['row_ratio'] <= 1 + 1e-12
        assert m['spectral_ratio'] <= 1 + 1e-12
    assert np.sum(g * best) == max(np.sum(g * a), np.sum(g * b))
    reference = exact_drs(g, r, s, tol=1e-8)
    assert reference['converged']
    assert reference['upper'] >= np.sum(g * best) - 1e-10
    inexact = solve_idrs(g, r, s, gamma=0.3, sigma=0.8, tol=1e-8)
    assert inexact['converged']
    assert inexact['lower'] <= reference['upper'] + 1e-9
    assert reference['lower'] <= inexact['upper'] + 1e-9
    for step in inexact['inner_history']:
        assert step['relative_error'] <= 0.8 + 1e-10 or step['exact_roundoff']


@pytest.mark.parametrize('r,s', [(1.0, 0.5), (0.1, 1.0)])
def test_redundant_constraint_independent_optimum(r, s):
    g = np.diag([3.0, 2.0, 1.0])
    optimum = min(r, s) * 6.0
    result = exact_drs(g, r, s, tol=1e-10)
    assert result['converged']
    assert abs(result['lower'] - optimum) < 1e-8
    assert result['upper'] >= optimum - 1e-10


def test_dual_bound_at_arbitrary_exact_split_and_iteration_limit():
    rng = np.random.default_rng(4)
    g, a = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    assert np.sum(g * best_of_two(g, 0.5, 1.0)) <= dual_value(g, a, 0.5, 1.0)
    result = exact_drs(g, 0.5, 1.0, tol=1e-14, max_iters=1)
    assert not result['converged']
    assert result['upper'] >= result['lower'] - 1e-12


def test_zero_and_input_validation():
    result = exact_drs(np.zeros((4, 2)), 0.5, 1.0)
    assert result['converged'] and result['iterations'] == 0
    assert np.all(result['w'] == 0)
    for kwargs in [dict(r=0, s=1), dict(r=1, s=1, gamma=-1),
                   dict(r=1, s=1, max_iters=0)]:
        with pytest.raises(ValueError):
            exact_drs(np.ones((4, 2)), **kwargs)


@pytest.mark.parametrize('shape,rank', [((8, 5), 2), ((4, 7), 2), ((8, 5), 5)])
def test_idrs_nuclear_pair_is_on_exact_subgradient_graph(shape, rank):
    from sparrow.iterative_solvers import nuclear_graph_pair
    rng = np.random.default_rng(3)
    z = rng.normal(size=shape)
    omega = rng.normal(size=(shape[1], min(shape)))
    gamma, s = 0.3, 0.7
    pair = nuclear_graph_pair(z, gamma, s, rank=rank, omega=omega)
    x, u = pair['x'], pair['u']
    assert np.linalg.svd(u, compute_uv=False)[0] <= s + 1e-12
    assert np.sum(u * x) == pytest.approx(s*np.linalg.svd(x, compute_uv=False).sum(), abs=1e-10)
    assert np.allclose(z-x, gamma*(u+pair['epsilon']), atol=1e-12)


def test_idrs_exact_limit_equals_classical_drs():
    from sparrow.iterative_solvers import idrs_step
    from sparrow.iterative_solvers import row_shrink
    rng = np.random.default_rng(8)
    g, z = rng.normal(size=(9, 4)), rng.normal(size=(9, 4))
    gamma, r, s, relaxation = 0.3, 0.5, 0.8, 1.2
    left, values, vt = np.linalg.svd(z, full_matrices=False)
    x = (left*np.maximum(values-gamma*s, 0))@vt
    p = g-row_shrink(g-(2*x-z), gamma*r)
    step = idrs_step(z, g, r, s, gamma=gamma, sigma=0, relaxation=relaxation)
    assert step['alpha'] == pytest.approx(gamma, abs=1e-12)
    assert np.allclose(step['z_next'], z+relaxation*(p-x), atol=1e-12)


def test_idrs_nonzero_error_correction_and_refinement():
    from sparrow.iterative_solvers import idrs_step
    rng = np.random.default_rng(19)
    g, z = rng.normal(size=(10, 6)), rng.normal(size=(10, 6))
    omega = rng.normal(size=(6, 6))
    loose = idrs_step(z, g, 0.4, 0.7, gamma=0.2, sigma=0.8, initial_rank=1, omega=omega, inner_solver="subspace")
    tight = idrs_step(z, g, 0.4, 0.7, gamma=0.2, sigma=0.01, initial_rank=1, omega=omega, inner_solver="subspace")
    assert tight['attempts'] > 1
    assert loose['relative_error'] <= 0.8 + 1e-12
    assert tight['relative_error'] <= 0.01 + 1e-12
    v = loose['v']
    alpha = np.sum(v*(z-loose['z_bar']))/np.sum(v*v)
    assert loose['alpha'] == pytest.approx(alpha)
    assert np.allclose(loose['z_next'], z-alpha*v)
    p = loose['x'] - 0.2 * loose['v']
    y = loose['y']
    assert np.max(np.linalg.norm(y, axis=1)) <= 0.4 + 1e-12
    assert -np.sum(y*(g-p)) == pytest.approx(
        0.4*np.linalg.norm(g-p, axis=1).sum(), abs=1e-10)


@pytest.mark.parametrize('gamma,sigma', [(0.1, 0.8), (0.3, 0.5), (1.0, 0)])
def test_idrs_vector_optimum_and_fejer_descent(gamma, sigma):
    from sparrow.iterative_solvers import idrs_step, solve_idrs
    g = np.array([[4.0], [2.0], [1.0], [0.2]])
    r, s = 1.0, 1.5
    w_star = vector_optimum(g, r, s)
    value = float(np.sum(g*w_star))
    result = solve_idrs(g, r, s, gamma=gamma, sigma=sigma, tol=1e-9)
    assert result['converged']
    assert result['lower'] <= value + 1e-10 <= result['upper'] + 2e-10
    assert value-result['lower'] < 1e-7
    x_star = (g[-1, 0]/w_star[-1, 0])*w_star
    z_star = x_star+gamma*w_star
    z = np.zeros_like(g)
    for _ in range(40):
        step = idrs_step(z, g, r, s, gamma=gamma, sigma=sigma)
        assert np.linalg.norm(step['z_next']-z_star) <= np.linalg.norm(z-z_star)+1e-10
        z = step['z_next']


def test_idrs_genuine_partial_step_and_failures():
    from sparrow.iterative_solvers import idrs_step, solve_idrs
    z = np.diag([5., 0.1, 0.01, 0.001])
    g = np.eye(4)
    step = idrs_step(z, g, 1, 1.5, gamma=0.1, sigma=0.8,
                     initial_rank=1, omega=np.eye(4), inner_solver="subspace")
    assert step['rank'] < 4 and step['error_norm'] > 1e-4
    assert step['relative_error'] <= 0.8
    assert not np.isclose(step['alpha'], 0.1, atol=1e-12)
    # Here x*=0 and W*=I give the independent DRS fixed point z*=gamma I.
    assert np.linalg.norm(step['z_next']-0.1*g) < np.linalg.norm(z-0.1*g)
    zero = solve_idrs(np.zeros((4, 2)), 0.5, 1)
    assert zero['converged'] and zero['iterations'] == 0
    result = solve_idrs(np.random.default_rng(6).normal(size=(8, 3)),
                        0.5, 1, tol=1e-14, max_iters=1)
    assert not result['converged']
    for kwargs in [dict(sigma=1), dict(gamma=0), dict(relaxation=2),
                   dict(initial_rank=0), dict(max_iters=0)]:
        with pytest.raises(ValueError):
            solve_idrs(g, 0.5, 1, **kwargs)


@pytest.mark.parametrize('shape', [(9, 5), (5, 9)])
def test_power_tail_completion_preserves_nuclear_subgradient(shape):
    from sparrow.iterative_solvers import nuclear_graph_pair
    rng = np.random.default_rng(51)
    z = rng.normal(size=shape)
    q, _ = np.linalg.qr(rng.normal(size=(shape[0], 2)), mode='reduced')
    pair = nuclear_graph_pair(z, 0.3, 0.7, rank=2, basis=q, complete_tail=True)
    x, u = pair['x'], pair['u']
    assert np.linalg.svd(u, compute_uv=False)[0] <= 0.7 + 1e-12
    assert np.sum(u*x) == pytest.approx(0.7*np.linalg.svd(x, compute_uv=False).sum(), abs=1e-10)
    assert np.allclose(z-x, 0.3*(u+pair['epsilon']), atol=1e-12)


def test_power_loops_more_at_same_rank_until_relative_test_passes():
    from sparrow.iterative_solvers import idrs_step
    z, g = np.diag([5., 1., 0.03, 0.01]), np.eye(4)
    loose = idrs_step(z, g, 1, 1.5, gamma=0.1, sigma=0.001, initial_rank=2)
    tight = idrs_step(z, g, 1, 1.5, gamma=0.1, sigma=0.0001, initial_rank=2)
    assert loose['rank'] == tight['rank'] == 2
    assert loose['rank_expansions'] == tight['rank_expansions'] == 0
    assert 0 < loose['power_iters'] < tight['power_iters']
    assert tight['relative_error'] <= 0.0001
    assert tight['error_norm'] < loose['error_norm']/100


def test_power_rank_floor_triggers_enrichment_and_exact_fallback():
    from sparrow.iterative_solvers import idrs_step
    step = idrs_step(np.diag([5., 4., 3., 2.]), np.eye(4), 1, 1.5,
                     gamma=0.1, sigma=0.001, initial_rank=1, max_power_iters=2)
    assert step['rank'] == 4
    assert step['rank_expansions'] == 2
    assert step['power_iters'] == 4
    assert step['error_norm'] < 1e-10


def test_power_completion_handles_small_modes_without_full_svd():
    from sparrow.iterative_solvers import idrs_step
    step = idrs_step(np.diag([5., 0.1, 0.01, 0.001]), np.eye(4), 1, 1.5,
                     gamma=0.1, sigma=0.5, initial_rank=1, omega=np.eye(4))
    assert step['rank'] == 1
    assert step['error_norm'] < 1e-12  # tail is below the threshold and captured in u
    assert step['power_iters'] == step['rank_expansions'] == 0


def test_power_warm_start_reproducibility_and_parameter_checks():
    from sparrow.iterative_solvers import solve_idrs
    g = np.random.default_rng(11).normal(size=(8, 4))
    options = dict(gamma=0.3, sigma=0.5, tol=1e-8, seed=6)
    first, second = solve_idrs(g, 0.6, 1, **options), solve_idrs(g, 0.6, 1, **options)
    assert first['converged'] and second['converged']
    assert np.allclose(first['w'], second['w'])
    assert first['inner_history'] == second['inner_history']
    cold = solve_idrs(g, 0.6, 1, warm_start=False, **options)
    assert cold['converged']
    assert first['lower'] <= cold['upper'] + 1e-9
    assert cold['lower'] <= first['upper'] + 1e-9
    for options in [dict(inner_solver='wrong'), dict(max_power_iters=0)]:
        with pytest.raises(ValueError):
            solve_idrs(g, 0.6, 1, **options)


@pytest.mark.parametrize('rho', [0.3, 1.0, 3.0])
def test_exact_drs_is_dual_admm(rho):
    # ADMM on min_{A+B=G} r||A||_{2,1} + s||B||_* with penalty rho takes the DRS iterates z = B - U, gamma = 1/rho.
    from sparrow.iterative_solvers import row_shrink, synthetic
    from sparrow.iterative_solvers import idrs_step
    g, r, s = synthetic('gaussian', 12, 4, 0), 0.6, 1.0
    b, multiplier, z = (np.zeros_like(g) for _ in range(3))
    for _ in range(30):
        a = row_shrink(g-b-multiplier, r/rho)
        left, values, vt = np.linalg.svd(g-a-multiplier, full_matrices=False)
        b = (left*np.maximum(values-s/rho, 0))@vt
        multiplier = multiplier+a+b-g
        step = idrs_step(z, g, r, s, gamma=1/rho, sigma=0)
        z = step['z_next']
        assert np.allclose(z, b-multiplier, atol=1e-12)
