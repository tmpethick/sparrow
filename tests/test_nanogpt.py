"""Checks for the nanoGPT Shakespeare experiment (experiments/nanogpt.py)."""
import numpy as np


def test_epoch_sampler_one_pass():
    """--sampler epoch: every non-overlapping training window appears exactly once per pass,
    and the next pass is a different permutation."""
    from experiments.nanogpt import Cfg, epoch_batch
    ids = np.arange(1000, dtype=np.int32)
    cfg = Cfg(ctx=8, batch=4, sampler="epoch")
    per_pass = (len(ids) - 1) // cfg.ctx // cfg.batch
    first = [np.asarray(epoch_batch(ids, cfg, t)[0])[:, 0] for t in range(2 * per_pass)]
    a, b = np.concatenate(first[:per_pass]), np.concatenate(first[per_pass:])
    assert len(set(a.tolist())) == len(a) == per_pass * cfg.batch
    assert set(a.tolist()) == set(b.tolist()) and not np.array_equal(a, b)
    assert all(x % cfg.ctx == 0 for x in a)
    x, y = epoch_batch(ids, cfg, 0)
    assert np.array_equal(np.asarray(y), np.asarray(x) + 1)


def test_method_directions_match_sparrow_jax():
    """--method maps every hidden matrix to its direction as in the docstring's table, and the directions agree
    with the stand-alone reference sparrow_jax.sparrow (kinds spectral, row, gate_value)."""
    import jax
    import sparrow_jax
    from experiments.nanogpt import DIRECTIONS, Cfg, direction_kind, radius
    d, m = 128, 512
    shapes = dict(wq=(d, d), wo=(d, d), w_gate=(m, d), w_in=(m, d), w_down=(d, m), w_ag=(m, d))
    expect = {"muon": dict(w_gate="spectral", w_ag="spectral"), "sparrow": dict(w_gate="sparrow", w_ag="row"),
              "sparrow_gate": dict(w_gate="spectral", w_ag="row"), "aurora": dict(w_gate="aurora", w_ag="aurora"),
              "exact": dict(w_gate="exact", w_ag="spectral")}
    for method, e in expect.items():
        cfg = Cfg(method=method, attn_gate="elem", head_dim=128)
        kinds = {n: direction_kind(n, sh, cfg) for n, sh in shapes.items()}
        assert kinds["w_gate"] == kinds["w_in"] == e["w_gate"] and kinds["w_ag"] == e["w_ag"]
        assert kinds["wq"] == kinds["wo"] == kinds["w_down"] == "spectral"  # square or wide
    assert direction_kind("w_gate", (d, d), Cfg(method="sparrow")) == "spectral"  # r = s: a spectral ball
    assert direction_kind("w_in", (m, d), Cfg(method="adamw")) is None
    reference = dict(spectral="spectral", row="row", sparrow="gate_value")
    for i, kind in enumerate(reference):
        G = jax.random.normal(jax.random.PRNGKey(i), (m, d))
        D = DIRECTIONS[kind](G, radius(G.shape, 0.8), 0.8)
        assert np.allclose(D, sparrow_jax.sparrow(G, reference[kind], 0.8, "svd"), atol=1e-5)


def test_loop_forward():
    """--loop K: z^{k+1} = beta u + (1 - beta) T(z^k) between prelude and coda. beta = 0 with K = 1
    is the plain stack of all layers; beta = 1 skips the core (z^K = u); no loop is the plain stack."""
    import dataclasses
    import jax
    from experiments.nanogpt import Cfg, forward, init_params
    cfg = Cfg(n_layer=3, d=32, n_head=2, ctx=8)
    params = init_params(cfg, 11, jax.random.PRNGKey(0))
    x = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, 11)
    plain = forward(params, x, cfg)
    looped = lambda **kw: forward(params, x, dataclasses.replace(cfg, n_prelude=1, n_coda=1, **kw))
    assert np.allclose(looped(loop=1, beta=0.0), plain, atol=1e-5)
    skip = dict(params, layers=[params["layers"][0], params["layers"][2]])
    assert np.allclose(looped(loop=3, beta=1.0), forward(skip, x, dataclasses.replace(cfg, n_layer=2)), atol=1e-5)
    assert not np.allclose(looped(loop=2, beta=0.5), looped(loop=4, beta=0.5))


def test_train_logs_evals(tmp_path):
    """A short unconstrained Muon run logs its config and an eval with validation and training losses."""
    import json
    from experiments.nanogpt import Cfg, train
    train(Cfg(method="muon", update="unconstrained", n_layer=1, d=32, n_head=2, ctx=16, batch=2, steps=3,
              warmup=1, eval_every=3, eval_batches=1, lmo_lr=0.1), str(tmp_path))
    recs = [json.loads(l) for l in open(tmp_path / "log.jsonl")]
    assert recs[0]["event"] == "config" and recs[-1]["event"] == "eval"
    assert recs[-1]["val_loss"] > 0 and recs[-1]["train_eval"] > 0


def test_loop_tbptt():
    """--tbptt H: the same forward pass, but the gradient only flows through the last H iterations. With a
    single core layer and no prelude/coda, H = 1 drops every term of the gradient that passes through an
    earlier application of the core, so it differs from full backprop, and H = K equals it."""
    import dataclasses
    import jax
    from experiments.nanogpt import Cfg, forward, init_params, loss_fn
    cfg = Cfg(n_layer=1, d=32, n_head=2, ctx=8, loop=4, n_prelude=0, n_coda=0, beta=0.5)
    params = init_params(cfg, 11, jax.random.PRNGKey(0))
    params["layers"] = [jax.tree_util.tree_map(lambda w: w + 0.05 * jax.random.normal(jax.random.PRNGKey(2), w.shape),
                                               params["layers"][0])]
    x = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, 11)
    grad = lambda H: jax.grad(loss_fn)(params, x, x, dataclasses.replace(cfg, tbptt=H))["layers"][0]["w_in"]
    assert np.allclose(forward(params, x, cfg), forward(params, x, dataclasses.replace(cfg, tbptt=1)), atol=1e-6)
    assert np.allclose(grad(0), grad(4), atol=1e-6)
    assert not np.allclose(grad(0), grad(1), atol=1e-6)
    assert np.isfinite(np.asarray(grad(2))).all()


def test_exact_oracle_and_lmo_track(tmp_path):
    """--method exact steps the SwiGLU gate/value along the warm-started DRS LMO (descending, so the loss
    falls like Sparrow's) and logs its duality gap; --lmo_track logs Sparrow's and Aurora's agreement with it."""
    import json
    from experiments.nanogpt import Cfg, train
    base = dict(update="unconstrained", n_layer=1, d=32, n_head=2, ctx=16, batch=4, steps=20, warmup=2,
                eval_every=20, eval_batches=1, lmo_lr=0.02)
    rec = {}
    for o in ("exact", "sparrow"):
        train(Cfg(**base, method=o, drs_iters=30, lmo_track=1), str(tmp_path / o))
        rec[o] = [json.loads(l) for l in open(tmp_path / o / "log.jsonl")][-1]
    assert rec["exact"]["drs_gap"]["max"] < 0.05
    t = rec["sparrow"]["lmo_track"]
    assert t["gap"] < 0.01 and t["exact"]["cos"] > 0.999 and t["sparrow"]["row"] <= 1.001 and t["aurora"]["cos"] > 0.8
    assert abs(rec["exact"]["val_loss"] - rec["sparrow"]["val_loss"]) < 0.05
