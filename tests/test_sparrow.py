"""The stand-alone Sparrow references (sparrow_jax.py, sparrow_torch.py) agree with the
library's LMO (sparrow.approx_oracles.DenseGeometry)."""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sparrow import approx_oracles as O


def load(name):
    return importlib.import_module(name)   # sparrow/ is on the test path (pyproject.toml)


SHAPES = [(256, 64), (64, 64), (64, 256)]   # tall (SwiGLU gate at m = 4d), square, wide


KINDS = ["gate_value", "row", "spectral"]


def radii(kind, shape, s):
    """The radii (s, r) of each kind's constraints (None: no such constraint), at r = s * min(1, sqrt(d / m))."""
    r = s * min(1.0, (shape[1] / shape[0]) ** 0.5)
    return {"gate_value": (s, r), "row": (None, r), "spectral": (s, None)}[kind]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_jax_reference_matches_library_lmo(shape, kind):
    ref = load("sparrow_jax")
    G = jax.random.normal(jax.random.PRNGKey(0), shape)
    s, r = radii(kind, shape, 0.7)
    # DenseGeometry.lmo(g) minimizes <g, W>, so the direction for G is lmo(-G).
    lib = O.DenseGeometry(spec=s, row=r, oracle="sparrow").lmo(-G)
    Z = ref.sparrow(G, kind, 0.7, "svd")
    assert np.allclose(Z, lib, atol=1e-5)
    assert s is None or float(O.sigma_max(Z)) <= s * (1 + 1e-4)
    assert r is None or float(O.row_max(Z)) <= r * (1 + 1e-5)


def test_swiglu_output_and_square_gate_value_get_muon():
    ref = load("sparrow_jax")
    for shape in SHAPES:
        G = jax.random.normal(jax.random.PRNGKey(1), shape)
        assert np.allclose(ref.sparrow(G, "spectral", polar="svd"), O._polar(G), atol=1e-5)
        joint = ref.sparrow(G, "gate_value", polar="svd")
        assert np.allclose(joint, O._polar(G), atol=1e-5) == (shape[0] <= shape[1])


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_iterative_polar(shape, kind):
    """Polar Express converges to the SVD's polar factor; Newton-Schulz only gets the singular
    values near 1. Where RowClip or Row comes last, the row constraint holds exactly either way."""
    ref = load("sparrow_jax")
    G = jax.random.normal(jax.random.PRNGKey(2), shape)
    s, r = radii(kind, shape, 0.7)
    exact = ref.sparrow(G, kind, 0.7, "svd")
    assert np.allclose(ref.sparrow(G, kind, 0.7, "polarexpress", 8), exact, atol=1e-4)
    for polar, steps in [("polarexpress", 5), ("newtonschulz5", 5)]:
        Z = ref.sparrow(G, kind, 0.7, polar, steps)
        sv = np.linalg.svd(np.asarray(ref.polar_factor(G, polar, steps)), compute_uv=False)
        assert 0.5 < sv.min() and sv.max() < 1.25
        assert r is None or r == s or float(O.row_max(Z)) <= r * (1 + 1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("polar", ["svd", "newtonschulz5", "polarexpress"])
def test_torch_reference_matches_jax(shape, kind, polar):
    torch = pytest.importorskip("torch")
    ref_t, ref_j = load("sparrow_torch"), load("sparrow_jax")
    G = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    zt = ref_t.sparrow(torch.from_numpy(G), kind, 0.7, polar).numpy()
    assert np.allclose(zt, np.asarray(ref_j.sparrow(G, kind, 0.7, polar)), atol=1e-4)


def _torch_params(torch, shapes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(*sh, generator=g) / sh[1] ** 0.5) for sh in shapes]


@pytest.mark.parametrize("nesterov,polar", [(False, "svd"), (True, "svd"), (True, "polarexpress")])
def test_torch_optimizer_update_rule(nesterov, polar):
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    (p,) = _torch_params(torch, [(256, 64)])
    opt = ref.Sparrow([dict(params=[p], kind="gate_value")], lr=0.1, momentum=0.9, nesterov=nesterov,
                      polar=polar)
    w, buf = p.detach().clone(), torch.zeros_like(p)
    for t in range(3):
        grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(10 + t))
        p.grad = grad.clone()
        opt.step()
        buf = 0.9 * buf + 0.1 * grad
        g = grad + 0.9 * (buf - grad) if nesterov else buf
        w = w - 0.1 * ref.sparrow(g, "gate_value", polar=polar)
        assert torch.allclose(p.detach(), w, atol=1e-5)


def test_torch_optimizer_weight_norm_keeps_radius():
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    ps = _torch_params(torch, [(256, 64), (64, 256)])
    radii = [float(p.detach().norm()) for p in ps]
    opt = ref.Sparrow([dict(params=ps[:1], kind="gate_value"), dict(params=ps[1:])], lr=0.05,
                      weight_norm=True)
    for t in range(5):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(t))
        opt.step()
    assert all(float(p.detach().norm()) == pytest.approx(R, rel=1e-5) for p, R in zip(ps, radii))


def test_torch_optimizer_weight_norm_step():
    """W <- R N(W - lr c D), c = R / (s sqrt(min(m, d))): MuonH for "spectral", and the
    RowClip-shortened step is kept for "gate_value"."""
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    for shape, kind in [((256, 64), "gate_value"), ((64, 256), "spectral")]:
        (p,) = _torch_params(torch, [shape])
        w0 = p.detach().clone()
        R = float(w0.norm())
        opt = ref.Sparrow([dict(params=[p], kind=kind)], lr=0.05, momentum=0.0, nesterov=False,
                          s=0.8, weight_norm=True, polar="svd")
        grad = torch.randn(shape, generator=torch.Generator().manual_seed(3))
        p.grad = grad.clone()
        opt.step()
        D = ref.sparrow(grad, kind, 0.8, "svd")
        v = w0 - 0.05 * R / (0.8 * min(shape) ** 0.5) * D
        assert torch.allclose(p.detach(), R * v / v.norm(), atol=1e-6)
        if kind == "spectral":   # MuonH: the step has norm lr R exactly
            assert float((R / (0.8 * min(shape) ** 0.5) * D).norm()) == pytest.approx(R, rel=1e-5)


@pytest.mark.parametrize("weight_norm", [False, True])
def test_torch_optimizer_weight_decay_step(weight_norm):
    """W <- (1 - lr wd) W - lr D, and W <- R N((1 - lr wd) W - lr c D) with weight_norm."""
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    (p,) = _torch_params(torch, [(256, 64)])
    w0 = p.detach().clone()
    R = float(w0.norm())
    opt = ref.Sparrow([dict(params=[p], kind="gate_value")], lr=0.05, momentum=0.0,
                      nesterov=False, s=0.8, weight_decay=0.5, weight_norm=weight_norm)
    grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(3))
    p.grad = grad.clone()
    opt.step()
    D = ref.sparrow(grad, "gate_value", 0.8)
    if weight_norm:
        v = (1 - 0.05 * 0.5) * w0 - 0.05 * R / (0.8 * 64 ** 0.5) * D
        expected = R * v / v.norm()
    else:
        expected = (1 - 0.05 * 0.5) * w0 - 0.05 * D
    assert torch.allclose(p.detach(), expected, atol=1e-6)


def test_torch_optimizer_trains_a_swiglu_block():
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    torch.manual_seed(0)
    d, m = 16, 64
    gate, value, out = (torch.nn.Linear(d, m, bias=False), torch.nn.Linear(d, m, bias=False),
                        torch.nn.Linear(m, d, bias=False))
    opt = ref.Sparrow([dict(params=[gate.weight, value.weight], kind="gate_value"),
                       dict(params=[out.weight])], lr=0.02)
    x = torch.randn(256, d)
    y = torch.roll(x, 1, dims=1)
    f = lambda: ((out(torch.nn.functional.silu(gate(x)) * value(x)) - y) ** 2).mean()
    start = f().item()
    for _ in range(200):
        opt.zero_grad()
        f().backward()
        opt.step()
    assert f().item() < 0.5 * start


def _jax_params(shapes, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), len(shapes))
    return [jax.random.normal(k, sh) / sh[1] ** 0.5 for k, sh in zip(keys, shapes)]


@pytest.mark.parametrize("nesterov,polar", [(False, "svd"), (True, "svd"), (True, "polarexpress")])
def test_jax_optimizer_update_rule(nesterov, polar):
    ref = load("sparrow_jax")
    (w,) = _jax_params([(256, 64)])
    opt = ref.sparrow_optimizer(lr=0.1, momentum=0.9, nesterov=nesterov, kind="gate_value",
                                polar=polar)
    params, state = {"w": w}, opt.init({"w": w})
    buf = np.zeros(w.shape, np.float32)
    for t in range(3):
        grad = jax.random.normal(jax.random.PRNGKey(10 + t), w.shape)
        updates, state = opt.update({"w": grad}, state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates)
        buf = 0.9 * buf + 0.1 * np.asarray(grad)
        g = grad + 0.9 * (buf - grad) if nesterov else buf
        w = w - 0.1 * ref.sparrow(g, "gate_value", polar=polar)
        assert np.allclose(params["w"], w, atol=1e-5)


def test_jax_optimizer_weight_norm_step():
    """W <- R N(W - lr c D), c = R / (s sqrt(min(m, d))), and W stays at R."""
    ref = load("sparrow_jax")
    for shape, kind in [((256, 64), "gate_value"), ((64, 256), "spectral")]:
        (w0,) = _jax_params([shape])
        R = float(np.linalg.norm(w0))
        opt = ref.sparrow_optimizer(lr=0.05, momentum=0.0, nesterov=False, kind=kind, s=0.8,
                                    weight_norm=True)
        grad = jax.random.normal(jax.random.PRNGKey(3), shape)
        updates, _ = opt.update(grad, opt.init(w0), w0)
        D = ref.sparrow(grad, kind, 0.8)
        v = w0 - 0.05 * R / (0.8 * min(shape) ** 0.5) * D
        assert np.allclose(w0 + updates, R * v / np.linalg.norm(v), atol=1e-6)
        assert float(np.linalg.norm(w0 + updates)) == pytest.approx(R, rel=1e-5)


@pytest.mark.parametrize("weight_norm", [False, True])
def test_jax_optimizer_weight_decay_step(weight_norm):
    """W <- (1 - lr wd) W - lr D, and W <- R N((1 - lr wd) W - lr c D) with weight_norm."""
    ref = load("sparrow_jax")
    (w0,) = _jax_params([(256, 64)])
    R = float(np.linalg.norm(w0))
    opt = ref.sparrow_optimizer(lr=0.05, momentum=0.0, nesterov=False, kind="gate_value", s=0.8,
                                weight_decay=0.5, weight_norm=weight_norm)
    grad = jax.random.normal(jax.random.PRNGKey(3), w0.shape)
    updates, _ = opt.update(grad, opt.init(w0), w0)
    D = ref.sparrow(grad, "gate_value", 0.8)
    if weight_norm:
        v = (1 - 0.05 * 0.5) * w0 - 0.05 * R / (0.8 * 64 ** 0.5) * D
        expected = R * v / np.linalg.norm(v)
    else:
        expected = (1 - 0.05 * 0.5) * w0 - 0.05 * D
    assert np.allclose(w0 + updates, expected, atol=1e-6)


@pytest.mark.parametrize("weight_norm,weight_decay", [(False, 0.0), (True, 0.0), (False, 0.1),
                                                      (True, 0.1)])
def test_jax_optimizer_matches_torch(weight_norm, weight_decay):
    torch = pytest.importorskip("torch")
    ref_t, ref_j = load("sparrow_torch"), load("sparrow_jax")
    shapes, kinds = [(256, 64), (64, 256), (64, 128)], ["gate_value", "spectral", "row"]
    ws = [np.asarray(w) for w in _jax_params(shapes)]
    ps = [torch.nn.Parameter(torch.from_numpy(w.copy())) for w in ws]
    opt_t = ref_t.Sparrow([dict(params=[p], kind=k) for p, k in zip(ps, kinds)], lr=0.05,
                          weight_decay=weight_decay, weight_norm=weight_norm)
    opt_j = ref_j.sparrow_optimizer(lr=0.05, kind=kinds, weight_decay=weight_decay,
                                    weight_norm=weight_norm)
    params = [jnp.asarray(w) for w in ws]
    state = opt_j.init(params)
    for t in range(4):
        grads = [np.random.default_rng(t * 10 + i).standard_normal(sh).astype(np.float32)
                 for i, sh in enumerate(shapes)]
        for p, g in zip(ps, grads):
            p.grad = torch.from_numpy(g)
        opt_t.step()
        updates, state = opt_j.update([jnp.asarray(g) for g in grads], state, params)
        params = [p + u for p, u in zip(params, updates)]
    for p, q in zip(ps, params):
        assert np.allclose(p.detach().numpy(), np.asarray(q), atol=1e-5)


@pytest.mark.parametrize("name", ["sparrow_jax", "sparrow_torch"])
def test_spectral_radius_presets(name):
    if name == "sparrow_torch":
        pytest.importorskip("torch")
    ref = load(name)
    assert ref.spectral_radius("muon", (3072, 768)) == 2.0      # max(1, sqrt(out / in))
    assert ref.spectral_radius("muon", (768, 3072)) == 1.0
    assert ref.spectral_radius("mup", (768, 3072)) == 0.5         # sqrt(out / in)
    assert ref.spectral_radius("rms", (768, 3072)) == 3072 ** 0.5
    assert ref.spectral_radius(lambda m, d: m / d, (6, 3)) == 2.0
    assert ref.spectral_radius(0.7, (6, 3)) == 0.7
    with pytest.raises(AssertionError, match="unknown scale"):
        ref.spectral_radius("Muon", (6, 3))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("kind", KINDS)
def test_named_s_is_evaluated_at_the_shape(shape, kind):
    torch = pytest.importorskip("torch")
    ref_t, ref_j = load("sparrow_torch"), load("sparrow_jax")
    G = np.array(jax.random.normal(jax.random.PRNGKey(3), shape))
    s = max(1.0, (shape[0] / shape[1]) ** 0.5)
    assert np.allclose(ref_j.sparrow(jnp.asarray(G), kind, "muon"), ref_j.sparrow(jnp.asarray(G), kind, s),
                       atol=1e-6)
    assert np.allclose(ref_t.sparrow(torch.from_numpy(G), kind, lambda m, d: max(1.0, (m / d) ** 0.5)).numpy(),
                       ref_t.sparrow(torch.from_numpy(G), kind, s).numpy(), atol=1e-6)


@pytest.mark.parametrize("weight_norm", [False, True])
def test_named_s_optimizers_match_per_matrix_s(weight_norm):
    """One group with s="muon" steps every matrix as a group of its own with s = max(1, sqrt(m / d))
    would, and the JAX optimizer with s="muon" matches it."""
    torch = pytest.importorskip("torch")
    ref_t, ref_j = load("sparrow_torch"), load("sparrow_jax")
    shapes, kinds = [(256, 64), (64, 256), (64, 128)], ["gate_value", "spectral", "row"]
    ws = [np.asarray(w) for w in _jax_params(shapes)]
    ps = [torch.nn.Parameter(torch.from_numpy(w.copy())) for w in ws]
    qs = [torch.nn.Parameter(torch.from_numpy(w.copy())) for w in ws]
    opt_named = ref_t.Sparrow([dict(params=[p], kind=k) for p, k in zip(ps, kinds)], lr=0.05, s="muon",
                              weight_norm=weight_norm)
    opt_float = ref_t.Sparrow([dict(params=[q], kind=k, s=max(1.0, (q.shape[0] / q.shape[1]) ** 0.5))
                               for q, k in zip(qs, kinds)], lr=0.05, weight_norm=weight_norm)
    opt_j = ref_j.sparrow_optimizer(lr=0.05, kind=kinds, s="muon", weight_norm=weight_norm)
    params = [jnp.asarray(w) for w in ws]
    state = opt_j.init(params)
    for t in range(3):
        grads = [np.random.default_rng(t * 10 + i).standard_normal(sh).astype(np.float32)
                 for i, sh in enumerate(shapes)]
        for p, q, g in zip(ps, qs, grads):
            p.grad, q.grad = torch.from_numpy(g), torch.from_numpy(g.copy())
        opt_named.step()
        opt_float.step()
        updates, state = opt_j.update([jnp.asarray(g) for g in grads], state, params)
        params = [w + u for w, u in zip(params, updates)]
    for p, q, w in zip(ps, qs, params):
        assert np.allclose(p.detach().numpy(), q.detach().numpy(), atol=1e-6)
        assert np.allclose(p.detach().numpy(), np.asarray(w), atol=1e-5)


def test_unknown_scale_name_fails_at_construction():
    torch = pytest.importorskip("torch")
    with pytest.raises(AssertionError, match="unknown scale"):
        load("sparrow_torch").Sparrow([torch.nn.Parameter(torch.zeros(4, 2))], s="nope")
    with pytest.raises(AssertionError, match="unknown scale"):
        load("sparrow_jax").sparrow_optimizer(s="nope")


def test_jax_optimizer_in_optax_trains_a_swiglu_block():
    """Hidden matrices on Sparrow (with a schedule) and the rest on AdamW, via multi_transform."""
    optax = pytest.importorskip("optax")
    ref = load("sparrow_jax")
    d, m = 16, 64
    w_gate, w_value, w_out = _jax_params([(m, d), (m, d), (d, m)])
    params = dict(w_gate=w_gate, w_value=w_value, w_out=w_out, gain=jnp.ones(d))
    hidden = ref.sparrow_optimizer(lr=optax.cosine_decay_schedule(0.02, 200),
                                   kind=dict(w_gate="gate_value", w_value="gate_value",
                                             w_out="spectral"))
    opt = optax.multi_transform(dict(sparrow=hidden, adamw=optax.adamw(1e-3)),
                                dict(w_gate="sparrow", w_value="sparrow", w_out="sparrow",
                                     gain="adamw"))
    x = jax.random.normal(jax.random.PRNGKey(5), (256, d))
    y = jnp.roll(x, 1, axis=1)

    def f(p):
        h = jax.nn.silu(x @ p["w_gate"].T) * (x @ p["w_value"].T)
        return jnp.mean((p["gain"] * (h @ p["w_out"].T) - y) ** 2)

    @jax.jit
    def step(p, s):
        updates, s = opt.update(jax.grad(f)(p), s, p)
        return optax.apply_updates(p, updates), s

    start, state = float(f(params)), opt.init(params)
    for _ in range(200):
        params, state = step(params, state)
    assert float(f(params)) < 0.5 * start


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.5, 1.0])
def test_rho_references_match_library_sparrow_c(rho):
    """gate_value with rho (RowClip_r Spectral_s RowClip_c): torch = jax = approx_oracles.sparrow_c; rho = 1 is U_1."""
    torch = pytest.importorskip("torch")
    ref_t, ref_j = load("sparrow_torch"), load("sparrow_jax")
    G = np.random.default_rng(1).standard_normal((256, 64)).astype(np.float32) * np.geomspace(0.1, 10, 256)[:, None]
    s, r = 0.7, 0.35
    lib = np.asarray(O.sparrow_c(jnp.asarray(G), r, s, rho)[0])
    zt = ref_t.sparrow(torch.from_numpy(G), "gate_value", s, "svd", rho=rho).numpy()
    assert np.allclose(zt, lib, atol=1e-4)
    assert np.allclose(np.asarray(ref_j.sparrow(G, "gate_value", s, "svd", rho=rho)), lib, atol=1e-4)
    if rho == 1.0:
        assert np.allclose(zt, ref_t.row_clip(ref_t.spectral(torch.from_numpy(G), s, "svd"), r).numpy(), atol=1e-4)


def test_torch_optimizer_uses_group_rho():
    torch = pytest.importorskip("torch")
    ref = load("sparrow_torch")
    (p,) = _torch_params(torch, [(256, 64)])
    opt = ref.Sparrow([dict(params=[p], kind="gate_value")], lr=0.1, momentum=0.0, nesterov=False)
    grad = torch.randn(p.shape, generator=torch.Generator().manual_seed(3))
    w = p.detach().clone()
    opt.param_groups[0]["rho"] = 0.4   # as a schedule sets it per step
    p.grad = grad.clone()
    opt.step()
    assert torch.allclose(p.detach(), w - 0.1 * ref.sparrow(grad, "gate_value", rho=0.4), atol=1e-6)
