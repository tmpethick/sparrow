"""nanoGPT on Shakespeare-char with Muon, Sparrow, Aurora and the exact joint LMO as drop-in updates
(the paper's training-speed, gated-attention and looped experiments).

Model: nanoGPT's CPU Shakespeare-char setup (4 layers, 4 heads, width d = 128, context 64, batch 12) with
RMSNorm, a SwiGLU (or ReLU^2) MLP of width m = 4d, an optional sigmoid gate on the attention output
(--attn_gate) and an optional looped core (--loop). The embeddings and the head are trained with AdamW. Every
other ("hidden") matrix W, stored (out, in), keeps a momentum M <- mu M + (1 - mu) grad and steps along a
direction D(M) with ||D||_op <= s (--spec):

    W <- W - eta D                                         --update unconstrained
    W <- R N(W - eta R N(D)),      N(X) = X / ||X||_F      --update hyperball (MuonH, SparrowH; Wen et al. 2026)
    W <- R N(W - eta c D),         c = R / (s sqrt(min(out, in)))   --update hyperball_unnorm

where R is the Frobenius norm of W at initialization. --method chooses D for each matrix, with the row radius
r = s min(1, sqrt(in / out)):

    method        SwiGLU gate and value         attention gate   other matrices   paper
    adamw         AdamW for every matrix
    muon          Spectral_s                    Spectral_s       Spectral_s       Muon
    sparrow       RowClip_r Spectral_s Row_r    Row_r            Spectral_s       Sparrow
    sparrow_gate  Spectral_s                    Row_r            Spectral_s       Sparrow on the gates only
    rowclip       RowClip_r Spectral_s          Spectral_s       Spectral_s       U_1
    aurora        Aurora on every tall matrix, Spectral_s on the others          Algorithm 3 of Dewulf et al. 2026
                  (without an attention gate or tall heads: the gate and value only)
    exact         joint LMO of K_{r,s} (DRS)    Spectral_s       Spectral_s       exact-LMO comparison
    sparrow_rho   RowClip_r Spectral_s RowClip_c, Row_r          Spectral_s       Sparrow with a row-clipping
                  threshold c = rho / (1 - rho) RMS row norm: --rho in [0, 1] fixed (0: Sparrow, 1/2: c = RMS row
                  norm, 1: U_1), or --rho sched: rho = u, the decay of the step size so far (rho_schedule; Sparrow
                  through the warmup and at the peak, U_1 at the end)

The gate and value are the matrices of the joint set K_{r,s} (for ReLU^2: its input matrix). Where one is square
or wide, r = s and K_{r,s} is the spectral ball, so it takes Spectral_s. The directions are in
sparrow.approx_oracles; --method exact warm-starts its DRS from the previous step (--drs_iters per step).

Recurrence (--loop K > 0): the first --n_prelude layers map the embeddings to the anchor u, the middle layers form
the core T, iterated with fixed-step Halpern z^{k+1} = beta u + (1 - beta) T(z^k) from z^0 = u, and the last
--n_coda layers read out z^K. --tbptt H backpropagates through the last H iterations only (z^{K-H} is detached,
the anchor u is not).

    python experiments/nanogpt.py --method sparrow --sampler epoch --epochs 1 --out runs/lm-sparrow
Re-running with the same --out resumes; --budget SECONDS pauses cleanly.
"""

import argparse
import dataclasses
import json
import math
import os
import time
import urllib.request

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from sparrow.approx_oracles import (aurora_update, drs_lmo, row, row_max, rowclip_polar, sigma_max, sparrow, sparrow_c,
                                    spectral)

DATA_DIR = os.environ.get(
    "SPARROW_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Complete Works of Shakespeare (Project Gutenberg eBook #100, 5.4M characters after stripping the
# Gutenberg header and license), ~5x Tiny Shakespeare, for runs that must not repeat data
URL_COMPLETE = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


@dataclasses.dataclass
class Cfg:
    method: str = "muon"        # adamw | muon | sparrow | sparrow_gate | rowclip | aurora | exact
    update: str = "unconstrained"  # unconstrained | hyperball | hyperball_unnorm (see the docstring)
    lmo_lr: float = 0.02        # eta of the hidden matrices, same schedule shape as lr
    spec: float = 1.0           # spectral radius s
    momentum: float = 0.9
    drs_iters: int = 50         # --method exact: DRS iterations per step
    rho: str = "sched"          # --method sparrow_rho: a number in [0, 1], c = rho / (1 - rho) * (RMS row norm of M)
                                # (0: Sparrow, 1: U_1), or "sched": rho = u, the decay of the step size so far
    lr: float = 1e-3            # AdamW (nanoGPT shakespeare_char): embeddings and head
    wd: float = 0.1
    warmup: int = 100           # linear warmup, then --decay
    decay: str = "cosine"       # cosine: to 10% at the end (nanoGPT) | linear0: constant, then linear to 0 over the
                                # last --warmdown fraction of the steps (the modded-nanoGPT record: 1450 of 5100)
    warmdown: float = 0.28
    steps: int = 2000
    epochs: float = 0.0         # > 0 (--sampler epoch): steps = epochs * windows / batch, overriding steps
    sampler: str = "random"     # random: windows drawn with replacement (nanoGPT) | epoch: shuffled
                                # non-overlapping windows, each training character predicted once per pass
    batch: int = 12
    ctx: int = 64
    data: str = "tiny"          # tiny: Tiny Shakespeare | complete: Complete Works (URL_COMPLETE), characters |
                                # shk_bpe: Complete Works | fineweb: a FineWeb shard, both GPT-2 BPE (vocabulary 50304)
    seed: int = 0
    n_layer: int = 4
    n_head: int = 4
    d: int = 128
    head_dim: int = 0           # attention head width (0: d / n_head); n_head * head_dim > d makes W_Q, W_K, W_V tall
    mlp: str = "swiglu"         # swiglu | relu2
    mlp_mult: int = 4           # MLP width m = mlp_mult * d
    attn_gate: str = "none"     # none | head | elem: sigma(RMSNorm(z) W_a^T) on the attention output, one gate per
                                # head (W_a: n_head x d) or per channel (W_a: n_head * head_dim x d)
    init: str = "nanogpt"       # nanogpt (std 0.02, residual projections 0.02 / sqrt(2 n_layer)) | fanin (std
                                # 1 / sqrt(in), as for the hyperball radii of Wen et al. 2026)
    loop: int = 0               # > 0: iterations K of the looped core, 0: no recurrence
    n_prelude: int = 1
    n_coda: int = 1
    beta: float = 0.5           # anchor weight of the loop
    tbptt: int = 0              # gradients through the last tbptt iterations of the loop only (0: all)
    eval_every: int = 250
    eval_batches: int = 20
    eval_batch: int = 0         # windows per eval batch (0: batch), so the eval set can be fixed across batch sizes
    lmo_track: int = 0          # 1: at every eval, compare Sparrow, Aurora, RowClip Spectral and Sparrow_rho (at --rho)
                                # on the gate and value momentum with the joint LMO (DRS, 300 iterations)


# --------------------------------------------------------------------------- data

BPE_DATA = {"shk_bpe": "shk_bpe", "fineweb": "fineweb_local"}  # token shards of experiments/nanogpt_gpu/make_local_data.py


def load_text(data="tiny"):
    if data in BPE_DATA:  # GPT-2 BPE tokens (vocabulary 50304, as in the GPU script) instead of characters
        shard = lambda split: np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanogpt_gpu", "data",
                                                       BPE_DATA[data], f"fineweb_{split}_000000.bin"),
                                          dtype=np.uint16, offset=256 * 4).astype(np.int32)
        return shard("train"), shard("val"), 50304
    assert data in ("tiny", "complete"), data
    path = os.path.join(DATA_DIR, "tinyshakespeare.txt" if data == "tiny" else "shakespeare_complete.txt")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        if data == "tiny":
            urllib.request.urlretrieve(URL, path)
        else:
            raw = urllib.request.urlopen(URL_COMPLETE).read().decode("utf-8").replace("\r\n", "\n")
            a = raw.index("\n", raw.index("*** START OF")) + 1
            open(path, "w", encoding="utf-8").write(raw[a:raw.index("*** END OF")].strip() + "\n")
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    ids = np.array([chars.index(c) for c in text], dtype=np.int32)
    n = int(0.9 * len(ids))  # nanoGPT split: first 90% train, last 10% val
    return ids[:n], ids[n:], len(chars)


def epoch_batch(ids, cfg, step):
    """Batch `step` of the epoch sampler: pass p = step // per_pass uses permutation p of the
    non-overlapping windows starting at multiples of ctx (seeded by cfg.seed)."""
    n_win = (len(ids) - 1) // cfg.ctx
    per_pass = n_win // cfg.batch
    p, k = divmod(step, per_pass)
    perm = np.random.default_rng([cfg.seed, 7, p]).permutation(n_win)
    ix = perm[k * cfg.batch:(k + 1) * cfg.batch] * cfg.ctx
    x = np.stack([ids[i:i + cfg.ctx] for i in ix])
    y = np.stack([ids[i + 1:i + cfg.ctx + 1] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


def batch(ids, cfg, rng, n=None):
    ix = rng.integers(0, len(ids) - cfg.ctx - 1, n or cfg.batch)
    x = np.stack([ids[i:i + cfg.ctx] for i in ix])
    y = np.stack([ids[i + 1:i + cfg.ctx + 1] for i in ix])
    return jnp.asarray(x), jnp.asarray(y)


# --------------------------------------------------------------------------- model

def init_params(cfg, vocab, key):
    d, h = cfg.d, cfg.mlp_mult * cfg.d
    da = cfg.n_head * (cfg.head_dim or d // cfg.n_head)  # attention inner width
    keys = iter(jax.random.split(key, 3 + 7 * cfg.n_layer))
    nrm = lambda shape, std=0.02: std * jax.random.normal(next(keys), shape)
    proj_std = 0.02 / math.sqrt(2 * cfg.n_layer)  # nanoGPT residual-projection init
    layers = []
    for i in range(cfg.n_layer):
        fan = lambda shape, std: std if cfg.init == "nanogpt" else 1 / math.sqrt(shape[1])
        hid = lambda shape, std=0.02: nrm(shape, fan(shape, std))
        layer = dict(wq=hid((da, d)), wk=hid((da, d)), wv=hid((da, d)), wo=hid((d, da), proj_std),
                     w_in=hid((h, d)), w_down=hid((d, h), proj_std))
        if cfg.mlp == "swiglu":
            layer["w_gate"] = hid((h, d))
        if cfg.attn_gate != "none":
            # own key, so the other matrices are initialized as without the gate
            rows = cfg.n_head if cfg.attn_gate == "head" else da
            layer["w_ag"] = fan((rows, d), 0.02) * jax.random.normal(jax.random.fold_in(key, 1000 + i),
                                                                     (rows, d))
        layers.append(layer)
    return dict(tok=nrm((vocab, d)), pos=nrm((cfg.ctx, d)), head=nrm((vocab, d)), layers=layers)


def rmsnorm(x):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6)


def forward(params, x, cfg):
    B, T = x.shape
    hd = cfg.head_dim or cfg.d // cfg.n_head
    z = params["tok"][x] + params["pos"][:T]
    mask = jnp.tril(jnp.ones((T, T), bool))
    layers = params["layers"]
    if cfg.loop == 0:
        for p in layers:
            z = block(z, p, cfg, hd, mask)
        return rmsnorm(z) @ params["head"].T
    P, C = cfg.n_prelude, len(layers) - cfg.n_prelude - cfg.n_coda
    for p in layers[:P]:
        z = block(z, p, cfg, hd, mask)
    u = z
    for k in range(cfg.loop):
        if 0 < cfg.tbptt < cfg.loop and k == cfg.loop - cfg.tbptt:  # truncated backprop
            z = jax.lax.stop_gradient(z)
        t = z
        for p in layers[P:P + C]:
            t = block(t, p, cfg, hd, mask)
        z = cfg.beta * u + (1 - cfg.beta) * t
    for p in layers[P + C:]:
        z = block(z, p, cfg, hd, mask)
    return rmsnorm(z) @ params["head"].T


def block(z, p, cfg, hd, mask):
    B, T, _ = z.shape
    a = rmsnorm(z)
    q, k, v = [(a @ p[n].T).reshape(B, T, cfg.n_head, hd) for n in ("wq", "wk", "wv")]
    s = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(hd)
    att = jax.nn.softmax(jnp.where(mask, s, -jnp.inf), -1)
    y = jnp.einsum("bhts,bshd->bthd", att, v)
    if cfg.attn_gate == "head":
        y = y * jax.nn.sigmoid(a @ p["w_ag"].T)[..., None]
    elif cfg.attn_gate == "elem":
        y = y * jax.nn.sigmoid(a @ p["w_ag"].T).reshape(B, T, cfg.n_head, hd)
    z = z + y.reshape(B, T, cfg.n_head * hd) @ p["wo"].T
    a = rmsnorm(z)
    u = a @ p["w_in"].T
    u = jax.nn.silu(a @ p["w_gate"].T) * u if cfg.mlp == "swiglu" else jax.nn.relu(u) ** 2
    z = z + u @ p["w_down"].T
    return z


def xent(logits, y):
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))


def loss_fn(params, x, y, cfg):
    return xent(forward(params, x, cfg), y)


# --------------------------------------------------------------------------- optimizer

GATE_VALUE = {"swiglu": ("w_gate", "w_in"), "relu2": ("w_in",)}  # the matrices of the joint set K_{r,s}
METHODS = {  # method: direction of (the gate and value, the attention gate W_a, every other hidden matrix)
    "muon": ("spectral", "spectral", "spectral"),
    "sparrow": ("sparrow", "row", "spectral"),
    "sparrow_gate": ("spectral", "row", "spectral"),
    "rowclip": ("rowclip", "spectral", "spectral"),
    "aurora": ("aurora", "aurora", "aurora"),  # only where tall, see direction_kind
    "exact": ("exact", "spectral", "spectral"),
    "sparrow_rho": ("sparrow_rho", "row", "spectral"),  # rho from rho_schedule
}
# ascent directions D(M, r, s) with <M, D> > 0; the step is W - eta D. "exact" is stateful (see hidden_step): it
# keeps its DRS state between steps. "sparrow_rho" takes its rho per step (rho_schedule).
DIRECTIONS = {"spectral": lambda M, r, s: spectral(M, s), "row": lambda M, r, s: row(M, r),
              "sparrow": sparrow, "rowclip": rowclip_polar, "aurora": aurora_update}
JOINT = ("sparrow", "rowclip", "aurora", "exact", "sparrow_rho")
STATEFUL = ("exact",)  # directions for the joint set, which is a spectral ball if r = s


def radius(shape, s):
    """Row radius r = s min(1, sqrt(in / out)) of a matrix stored (out, in)."""
    return s * min(1.0, math.sqrt(shape[1] / shape[0]))


def direction_kind(name, shape, cfg):
    """The direction of one hidden matrix under cfg.method, or None for AdamW."""
    if cfg.method == "adamw":
        return None
    assert cfg.method in METHODS, cfg.method
    gv, gate, other = METHODS[cfg.method]
    kind = gv if name in GATE_VALUE[cfg.mlp] else gate if name == "w_ag" else other
    return "spectral" if kind in JOINT and shape[0] <= shape[1] else kind


def schedule(step, cfg):
    """Linear warmup, then cosine decay to 10% (nanoGPT; --decay cosine) or constant and a linear decay to 0 over the
    last --warmdown fraction of the steps (the modded-nanoGPT record; --decay linear0)."""
    if step < cfg.warmup:
        return (step + 1) / cfg.warmup
    if cfg.decay == "linear0":
        return min(1.0, (cfg.steps - step) / max(1.0, cfg.warmdown * cfg.steps))
    assert cfg.decay == "cosine", cfg.decay
    t = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


def rho_schedule(step, cfg):
    """rho of the sparrow_rho direction at this step: --rho, or for --rho sched u, the decay of the step size so far (0
    through the warmup and at the peak, 1 at the end: cosine to 10% or linear0 to 0), so Sparrow first and U_1 last."""
    if cfg.rho != "sched":
        return float(cfg.rho)
    f = schedule(step, cfg)
    return min(1.0, 0.0 if step < cfg.warmup else (1 - f) / (0.9 if cfg.decay == "cosine" else 1.0))


def make_optimizer(cfg, params):
    """The direction of every hidden matrix, AdamW for the rest, and the per-step hidden update."""
    assert cfg.update in ("unconstrained", "hyperball", "hyperball_unnorm"), cfg.update
    kinds = [{n: direction_kind(n, w.shape, cfg) for n, w in layer.items()} for layer in params["layers"]]
    adam_mask = jax.tree_util.tree_map(lambda _: True, params)
    adam_mask["layers"] = [{n: k is None for n, k in layer.items()} for layer in kinds]
    # AdamW (decoupled decay; every parameter here is a matrix, as nanoGPT decays 2-D params)
    adam = optax.masked(optax.chain(optax.scale_by_adam(b2=0.99), optax.add_decayed_weights(cfg.wd)), adam_mask)
    radii = [{n: float(jnp.linalg.norm(w)) for n, w in layer.items()} for layer in params["layers"]]  # hyperball R
    unit = lambda x: x / jnp.maximum(jnp.linalg.norm(x), 1e-30)

    def step(W, D, eta, R):
        if cfg.update == "hyperball":
            return R * unit(W - eta * R * unit(D))
        if cfg.update == "hyperball_unnorm":
            return R * unit(W - eta * R / (cfg.spec * math.sqrt(min(W.shape))) * D)
        return W - eta * D

    @jax.jit
    def hidden_step(params, new, mom, drs, eta, rho):
        """Steps every hidden matrix of `new` (the AdamW-updated params) from `params`; returns the DRS state and
        the relative duality gaps of --method exact. rho: of the sparrow_rho direction (rho_schedule)."""
        gaps = []
        for i, layer in enumerate(kinds):
            for n, kind in layer.items():
                if kind is None:
                    continue
                W, M = params["layers"][i][n], mom["layers"][i][n]
                s = cfg.spec
                r = radius(W.shape, s)
                if kind == "exact":
                    D, drs[i][n], gap = drs_lmo(M, r, s, drs[i][n], iters=cfg.drs_iters)
                    gaps.append(gap)
                elif kind == "sparrow_rho":
                    D = sparrow_c(M, r, s, rho)[0]
                else:
                    D = DIRECTIONS[kind](M, r, s)
                new["layers"][i][n] = step(W, D, eta, radii[i][n])
        return new, drs, jnp.stack(gaps) if gaps else jnp.zeros(0)

    drs = [{n: jnp.zeros_like(params["layers"][i][n])
            for n, k in layer.items() if k in STATEFUL}
           for i, layer in enumerate(kinds)]
    return kinds, adam, hidden_step, drs


def track_lmo(mom, cfg, rho):
    """Sparrow, Aurora, RowClip Spectral and Sparrow_rho (sparrow_c at rho) against the joint LMO X on the momentum of every gate and value matrix:
    cosine to X, ||D||_F, largest row norm / r, and <M, D> / <M, X> (the paper's comparison with the exact LMO)."""
    out = []
    for layer in mom["layers"]:
        for n in GATE_VALUE[cfg.mlp]:
            M = layer[n]
            s = cfg.spec
            r = radius(M.shape, s)
            X, _, gap = drs_lmo(M, r, s, iters=300)
            rec = dict(gap=gap)
            for k in ("sparrow", "aurora", "rowclip", "sparrow_rho"):
                D = sparrow_c(M, r, s, rho)[0] if k == "sparrow_rho" else DIRECTIONS[k](M, r, s)
                rec[k] = dict(cos=jnp.sum(D * X) / jnp.linalg.norm(D) / jnp.linalg.norm(X), fro=jnp.linalg.norm(D),
                              row=jnp.max(jnp.linalg.norm(D, axis=1)) / r, value=jnp.sum(M * D) / jnp.sum(M * X))
            rec["exact"] = dict(cos=jnp.ones(()), fro=jnp.linalg.norm(X), row=jnp.max(jnp.linalg.norm(X, axis=1)) / r,
                                value=jnp.ones(()))
            out.append(rec)
    return out


# --------------------------------------------------------------------------- training

def train(cfg, out):
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "log.jsonl"), "a")

    def emit(**rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    train_ids, val_ids, vocab = load_text(cfg.data)
    assert cfg.loop == 0 or 1 <= cfg.n_layer - cfg.n_prelude - cfg.n_coda, "--loop needs a core layer"
    assert cfg.sampler in ("random", "epoch"), cfg.sampler
    if cfg.epochs > 0:
        assert cfg.sampler == "epoch", "--epochs needs --sampler epoch"
        cfg = dataclasses.replace(cfg, steps=int(round(cfg.epochs * ((len(train_ids) - 1) // cfg.ctx // cfg.batch))))
    params = init_params(cfg, vocab, jax.random.PRNGKey(cfg.seed))
    kinds, adam, hidden_step, drs = make_optimizer(cfg, params)
    state = dict(adam=adam.init(params), mom=jax.tree_util.tree_map(jnp.zeros_like, params), drs=drs)

    # Two jits: on XLA:CPU, fusing the LAPACK SVDs with the backward pass was ~2x slower.
    @jax.jit
    def grad_step(params, state, x, y, lr):
        loss, g = jax.value_and_grad(loss_fn)(params, x, y, cfg)
        g, _ = optax.clip_by_global_norm(1.0).update(g, optax.EmptyState())
        upd, adam_state = adam.update(g, state["adam"], params)
        new = jax.tree_util.tree_map(lambda p, u: p - lr * u, params, upd)
        mom = jax.tree_util.tree_map(lambda m, gi: cfg.momentum * m + (1 - cfg.momentum) * gi, state["mom"], g)
        return new, dict(state, adam=adam_state, mom=mom), loss

    eval_fn = jax.jit(lambda p, x, y: xent(forward(p, x, cfg), y))
    track_fn = jax.jit(lambda mom, rho: track_lmo(mom, cfg, rho))

    def evaluate(params, ids, seed):
        r = np.random.default_rng(seed)
        return round(float(np.mean([eval_fn(params, *batch(ids, cfg, r, cfg.eval_batch))
                                    for _ in range(cfg.eval_batches)])), 4)

    ckpt = os.path.join(out, "state.eqx")
    start = 0
    if os.path.exists(ckpt + ".json"):
        start = json.load(open(ckpt + ".json"))["step"]
        params, state = eqx.tree_deserialise_leaves(ckpt, (params, state))
    else:
        emit(event="config", **dataclasses.asdict(cfg), params=int(sum(w.size for w in jax.tree_util.tree_leaves(params))),
             directions=kinds[0])

    t0, gaps = time.time(), None
    for step in range(start, cfg.steps):
        if args_budget and time.time() - t0 > args_budget:
            eqx.tree_serialise_leaves(ckpt, (params, state))
            json.dump(dict(step=step), open(ckpt + ".json", "w"))
            emit(event="paused", step=step)
            raise SystemExit(3)
        x, y = (epoch_batch(train_ids, cfg, step) if cfg.sampler == "epoch"
                else batch(train_ids, cfg, np.random.default_rng([cfg.seed, step])))
        f = schedule(step, cfg)
        new, state, loss = grad_step(params, state, x, y, cfg.lr * f)
        params, state["drs"], gaps = hidden_step(params, new, state["mom"], state["drs"], cfg.lmo_lr * f,
                                                    rho_schedule(step, cfg))
        if not all(bool(jnp.all(jnp.isfinite(w))) for w in jax.tree_util.tree_leaves(params)):
            emit(event="nan", step=step, loss=float(loss))
            raise SystemExit(4)
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            layers = params["layers"]
            rec = dict(step=step + 1, train_loss=round(float(loss), 4), val_loss=evaluate(params, val_ids, [cfg.seed, 10**6]),
                       train_eval=evaluate(params, train_ids, [cfg.seed, 10**6 + 1]), sec=round(time.time() - t0, 1),
                       norms={n: dict(spec=round(float(np.mean([sigma_max(l[n]) for l in layers])), 3),
                                      row=round(float(np.mean([row_max(l[n]) for l in layers])), 4)) for n in layers[0]})
            if cfg.mlp == "swiglu":  # max over layers of the SwiGLU certificate on the RMSNorm sphere (the
                # paper's SwiGLU Lipschitz bound without omega): sqrt(d) ||W_down|| (1.0998 ||W_v||_2->inf ||W_g|| + ||W_g||_2->inf ||W_v||)
                rec["cert"] = round(max(float(math.sqrt(cfg.d) * sigma_max(l["w_down"]) * (
                    1.0998 * row_max(l["w_in"]) * sigma_max(l["w_gate"]) + row_max(l["w_gate"]) * sigma_max(l["w_in"])))
                    for l in layers), 3)
            if cfg.attn_gate != "none":  # the gated-attention certificate uses max over layers of ||W_a||_2->inf
                rec["gate"] = dict(row_max=round(max(float(row_max(l["w_ag"])) for l in layers), 4),
                                   spec_max=round(max(float(sigma_max(l["w_ag"])) for l in layers), 4))
            # structure of the gate and value momentum, averaged over layers and matrices: effective rank (sum sigma)^2 /
            # sum sigma^2 as a fraction of min(out, in); largest over median row norm; and how uniform the rows of
            # Polar(M) are (their row norms are M's leverage scores): 10% and 90% quantiles in units of r, and the share
            # of ||Polar(M)||_F that RowClip_r keeps (1: Muon is already in K_{r,s}, so U_1 is the LMO); and
            # <M, Sparrow(M)> / <M, U_1(M)>, both in K_{r,s} (below 1: U_1 = RowClip_r Spectral_s is the better step)
            mstats = []
            for l in state["mom"]["layers"]:
                for n in GATE_VALUE[cfg.mlp]:
                    M = np.asarray(l[n], np.float64)
                    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
                    r = radius(M.shape, 1.0)
                    rn, lev = np.linalg.norm(M, axis=1), np.linalg.norm(U, axis=1) / r
                    u1 = np.sum(M * (U @ Vt) * np.minimum(1.0, 1.0 / np.maximum(lev, 1e-30))[:, None])
                    Us, _, Vts = np.linalg.svd(M / np.maximum(rn, 1e-300)[:, None], full_matrices=False)
                    Ps = Us @ Vts
                    u3 = np.sum(M * Ps * np.minimum(1.0, r / np.maximum(np.linalg.norm(Ps, axis=1), 1e-30))[:, None])
                    mstats.append((sv.sum() ** 2 / max((sv ** 2).sum(), 1e-300) / min(M.shape),
                                   rn.max() / max(np.median(rn), 1e-300), *np.quantile(lev, [0.1, 0.9]),
                                   math.sqrt(np.sum(np.minimum(lev, 1) ** 2) / np.sum(lev ** 2)), u3 / u1))
            m = np.mean(mstats, 0)
            rec["momentum"] = dict(eff_rank=round(float(m[0]), 3), row_spread=round(float(m[1]), 2),
                                   polar_rows=[round(float(m[2]), 2), round(float(m[3]), 2)], rowclip_keeps=round(float(m[4]), 3),
                                   sparrow_over_u1=round(float(m[5]), 3))
            if gaps.size:
                rec["drs_gap"] = dict(mean=float(gaps.mean()), max=float(gaps.max()))
            if cfg.method == "sparrow_rho":
                rec["rho"] = round(rho_schedule(step, cfg), 4)
            if cfg.lmo_track:
                rho_t = rho_schedule(step, cfg)
                rows = jax.tree_util.tree_map(float, track_fn(state["mom"], rho_t))
                rec["lmo_track"] = dict(gap=max(r["gap"] for r in rows), rho=round(rho_t, 4), **{
                    k: {q: round(float(np.mean([r[k][q] for r in rows])), 4) for q in ("cos", "fro", "row", "value")}
                    for k in ("sparrow", "aurora", "rowclip", "sparrow_rho", "exact")})
            emit(event="eval", **rec)
            eqx.tree_serialise_leaves(ckpt, (params, state))  # checkpoint at every eval
            json.dump(dict(step=step + 1), open(ckpt + ".json", "w"))
    json.dump(dict(step=cfg.steps), open(ckpt + ".json", "w"))
    eqx.tree_serialise_leaves(ckpt, (params, state))


args_budget = 0.0


def main():
    global args_budget
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.0)
    for f in dataclasses.fields(Cfg):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = vars(ap.parse_args())
    args_budget = a.pop("budget")
    out = a.pop("out")
    train(Cfg(**a), out)


if __name__ == "__main__":
    main()
