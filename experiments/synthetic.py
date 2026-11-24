"""Teacher-student classification: certified robustness at matched accuracy.

    logits(z) = H G(S_tau(z)),   G(u) = W_o [SiLU(W_g u) * W_v u],   z ~ N(0, I_d),   tau = 1.
Labels are the argmax of a teacher of the same form. The certificate (Theorem
nonexpansive-swiglu, from the actual weights) is
    L = ||H|| (sqrt(d) tau / 2) ||W_o|| (L_SiLU rho_v sigma_g + rho_g sigma_v),
and a test point is certified at radius margin / (sqrt(2) L), 0 if misclassified.

Teachers (W_o unit spectral, W_g, W_v with common sigma, budget b0 = sigma max rho = 1/sqrt(d)):
  gauss_4   i.i.d. Gaussian W_g, W_v with m = 4d: spread, kappa = rho / sigma ~ 0.43
  few_8     only 8 of the m hidden units are nonzero: concentrated, kappa ~ 1
Students: W_g, W_v in the spectral ball {sigma <= sqrt(B)} or in K_{sqrt(B kappa), sqrt(B / kappa)}
with the teacher's kappa, B = beta b0; W_o in the unit spectral ball. All three are trained
with the projected LMO step (momentum 0.9); the head H is trained with Adam and unconstrained
(the certified radius does not depend on the logit scale).

    python experiments/synthetic.py --out runs/tsr
    python experiments/synthetic.py --out runs/tsr-fam --families sgd_frob,adam_linf,muon_spec,ours_rowspec
Families pair each optimizer with its own geometry, at the same worst-case certificate
sup_{W in set} rho sigma = B (and ||W_o|| <= 1):
  sgd_frob      normalized momentum SGD, W <- Pi_F(W - eta c_F M / ||M||_F), c_F = sqrt(B)
  adam_linf     Adam direction, W <- clip_{c_inf}(W - eta c_inf adam(g)),   c_inf = sqrt(B / (d sqrt(m)))
  muon_spec     W <- SpecClip_s(W - eta s Polar(M)),                        s = sqrt(B)
  ours_rowspec  W <- RowClip_r(SpecClip_s(W - eta RowClip_r(s Polar(M)))),  s = sqrt(B / kappa), r = sqrt(B kappa)
--impl sparrow_jax keeps every projected step W <- Pi_K(W - eta D) (spec, rowspec, muon_spec,
ours_rowspec and W_o) but takes D from the stand-alone sparrow_jax.sparrow: s Polar(M) on a
spectral ball (as before) and Sparrow's RowClip_r(s Polar(Row_r(M))) on the joint set, in place of
RowClip_r(s Polar(M)). --polar newtonschulz5 | polarexpress (--polar_steps) replace the SVD in D
(not in the projection).
--oracle (ours_rowspec, library directions) picks the joint-set direction: rowclip (U_1, default), sparrow (U_3) or
sparrow_rho (RowClip_r Spectral_s RowClip_c, c = rho / (1 - rho) RMS row norm), with --rho a number in [0, 1] or sched:
rho = u = (1 - eta / eta0) / 0.9, the decay of the cosine step size (Sparrow at the start, U_1 at the end).
"""

import argparse
import itertools
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sparrow.approx_oracles import DenseGeometry, row_max, sigma_max

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sparrow_jax  # noqa: E402  (sparrow/sparrow_jax.py, the stand-alone reference)

L_SILU = 1.0998


def squash(x, tau):
    """N_RMS(x) = x / sqrt(1 + ||x||^2 / (d tau^2)) over the last axis (the paper's
    normalization profile; RMSNorm with gain tau and stabilizer tau^2): 1-Lipschitz and
    ||N_RMS(x)|| <= sqrt(d) tau."""
    d = x.shape[-1]
    return x * jax.lax.rsqrt(1 + jnp.sum(x * x, -1, keepdims=True) / (d * tau ** 2))


def raw_teacher(name, d, key):
    """gauss_M: i.i.d. Gaussian W_g, W_v with m = M d, and W_o with orthonormal rows."""
    if not name.startswith("gauss_"):
        raise ValueError(name)
    ks = jax.random.split(key, 6)
    m = int(name.split("_")[1]) * d
    g = jax.random.normal(ks[0], (m, d))
    v = jax.random.normal(ks[1], (m, d))
    q, _ = jnp.linalg.qr(jax.random.normal(ks[4], (m, d)))
    return q.T, g, v


def normalize(Wraw, c):
    """W_o -> unit spectral; W_g, W_v -> common sigma, then scaled so sigma * max rho = c / sqrt(d)."""
    wo, g, v = Wraw
    d = g.shape[1]
    wo = wo / sigma_max(wo)
    g, v = g / sigma_max(g), v / sigma_max(v)
    kappa = float(max(row_max(g), row_max(v)))            # rho / sigma with sigma = 1
    b0 = c / math.sqrt(d)
    t = math.sqrt(b0 / kappa)                             # sigma = t, rho = t kappa, product b0
    return [wo, t * g, t * v], kappa, b0


class FrobGeometry:
    """{||W||_F <= c_F}: lmo = -c_F M / ||M||_F (normalized SGD), projection = radial rescale."""

    def __init__(self, c_F):
        self.c_F = c_F

    def lmo(self, g):
        return -self.c_F * g / jnp.maximum(jnp.linalg.norm(g), 1e-12)

    def project(self, w):
        return w * jnp.minimum(1.0, self.c_F / jnp.maximum(jnp.linalg.norm(w), 1e-12))


class BoxGeometry:
    """{max_ij |W_ij| <= c_inf}: projection = entrywise clip. Stepped with the Adam direction."""

    def __init__(self, c_inf):
        self.c_inf = c_inf

    def project(self, w):
        return jnp.clip(w, -self.c_inf, self.c_inf)


class SparrowGeometry(DenseGeometry):
    """K_{r,s} with DenseGeometry's projection, stepped with the stand-alone sparrow_jax direction
    (--impl sparrow_jax): s Polar(g) on a spectral ball, RowClip_r(s Polar(Row_r(g))) on the
    joint set."""

    def __init__(self, geo, polar="svd", polar_steps=5):
        super().__init__(spec=geo.spec, row=geo.row)
        self.polar, self.polar_steps = polar, polar_steps

    def lmo(self, g, rho=None):   # rho: unused (sparrow_jax's Sparrow direction)
        kind = dict(spectral="spectral", mixed="gate_value", row="row")[self.regime(g.shape[0])]
        return -sparrow_jax.sparrow(g, kind, self.spec, self.polar, self.polar_steps, r=self.row)


def family_sets(fam, B, kappa, d, m, oracle="rowclip", rho="sched"):
    """(gate/value set, W_o set) with worst-case certificate sup rho sigma = B (and ||W_o|| <= 1). oracle and
    rho choose the joint-set direction of ours_rowspec (--oracle, --rho; DenseGeometry)."""
    if fam == "sgd_frob":
        return FrobGeometry(math.sqrt(B)), FrobGeometry(1.0)
    if fam == "adam_linf":
        return BoxGeometry(math.sqrt(B / (d * math.sqrt(m)))), BoxGeometry(1 / math.sqrt(d * m))
    if fam == "muon_spec":
        return DenseGeometry(spec=math.sqrt(B)), DenseGeometry(spec=1.0)
    if fam == "ours_rowspec":
        if oracle == "sparrow_rho":   # --rho sched: rho per step from train(rho_sched=True)
            oracle, rho = "sparrow_c", 0.5 if rho == "sched" else float(rho)
        return (DenseGeometry(spec=math.sqrt(B / kappa), row=math.sqrt(B * kappa), oracle=oracle,
                              rho=rho if oracle == "sparrow_c" else 0.5),
                DenseGeometry(spec=1.0))
    raise ValueError(fam)


def logits(W, z):
    H, wo, wg, wv = W
    u = squash(z, 1.0)
    return ((jax.nn.silu(u @ wg.T) * (u @ wv.T)) @ wo.T) @ H.T


def certificate(W):
    H, wo, wg, wv = W
    d = wg.shape[1]
    return float(sigma_max(H) * math.sqrt(d) / 2 * sigma_max(wo)
                 * (L_SILU * row_max(wv) * sigma_max(wg) + row_max(wg) * sigma_max(wv)))


def make_teacher(name, d, key, n_cls=10):
    if name.startswith("few_"):
        k = int(name.split("_")[1])
        wo, g, v = raw_teacher("gauss_4", d, key)
        mask = jnp.zeros(g.shape[0]).at[:k].set(1.0)[:, None]
        raw = (wo, g * mask, v * mask)
    else:
        raw = raw_teacher(name, d, key)
    (wo, wg, wv), kappa, b0 = normalize(raw, 1.0)
    H = jax.random.normal(jax.random.fold_in(key, 9), (n_cls, d))
    return [H / sigma_max(H), wo, wg, wv], kappa, b0


def radii(W, z, y):
    lg = np.asarray(logits(W, z))
    L = certificate(W)
    correct = lg.argmax(1) == y
    top = np.take_along_axis(lg, y[:, None], 1)[:, 0]
    other = np.where(np.eye(lg.shape[1], dtype=bool)[y], -np.inf, lg).max(1)
    return np.where(correct, (top - other) / (math.sqrt(2) * L), 0.0), correct, L


def train(Wt, gv, key, d, steps, eta0, batch=512, mu=0.9, wo_geo=None, fam="lmo", m=None,
          impl="library", polar="svd", polar_steps=5, rho_sched=False):
    """m: student width (default: the teacher's). impl: see the docstring (--impl). rho_sched: --oracle sparrow_rho
    --rho sched, the rho of the joint-set direction is u, the decay of the step size (False: fixed)."""
    ks = jax.random.split(key, 5)
    geos = [None, wo_geo or DenseGeometry(spec=1.0), gv, gv]
    if impl == "sparrow_jax":
        geos = [SparrowGeometry(g, polar, polar_steps) if isinstance(g, DenseGeometry) else g
                for g in geos]
    shapes = [w.shape for w in Wt[1:]] if m is None else [(d, m), (m, d), (m, d)]
    W = [jax.random.normal(ks[0], Wt[0].shape) / math.sqrt(d)] + [
        g.project(0.02 * jax.random.normal(kk, sh)) for g, kk, sh in zip(geos[1:], ks[1:4], shapes)]
    sadam = optax.scale_by_adam()
    sst = sadam.init(W[1:])
    M = [jnp.zeros_like(w) for w in W]
    adam = optax.adam(1e-2)
    ast = adam.init(W[0])

    # Wt is an argument, not a captured constant: with JAX 0.11.2 on CPU (macOS), jitted
    # closures over large arrays returned wrong values.
    @jax.jit
    def grad_fn(W, M, ast, sst, Wt, key):
        z = jax.random.normal(key, (batch, d))
        y = jnp.argmax(logits(Wt, z), 1)
        g = jax.grad(lambda W: optax.softmax_cross_entropy_with_integer_labels(logits(W, z), y).mean())(W)
        uH, ast = adam.update(g[0], ast, W[0])
        if fam == "adam_linf":                        # Adam direction for the block matrices
            u, sst = sadam.update(g[1:], sst, W[1:])
            return [None] + list(u), uH, ast, sst
        return [mu * mi + (1 - mu) * gi for mi, gi in zip(M, g)], uH, ast, sst

    @jax.jit
    def upd_fn(W, M, uH, eta, rho):
        if fam == "adam_linf":                        # step of eta * (box radius) per unit Adam direction
            return [W[0] + uH] + [g.project(w - eta * g.c_inf * u) for g, w, u in zip(geos[1:], W[1:], M[1:])]
        lmo = lambda g, mi: g.lmo(mi, rho) if isinstance(g, DenseGeometry) else g.lmo(mi)
        return [W[0] + uH] + [g.project(w + eta * lmo(g, mi)) for g, w, mi in zip(geos[1:], W[1:], M[1:])]

    for t in range(steps):
        f = 0.1 + 0.45 * (1 + math.cos(math.pi * t / steps))   # cosine from 1 to 0.1
        eta = eta0 * f
        u = (1 - f) / 0.9
        rho = u if rho_sched else None
        out, uH, ast, sst = grad_fn(W, M, ast, sst, Wt, jax.random.fold_in(ks[4], t))
        if fam != "adam_linf":
            M = out
        W = upd_fn(W, out, uH, eta, rho)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--teachers", default="gauss_4,few_8")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--betas", default="0.5,1,2,4")
    ap.add_argument("--etas", default="0.03,0.1")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", default="")   # e.g. sgd_frob,adam_linf,muon_spec,ours_rowspec
    ap.add_argument("--kappa_student", type=float, default=0.0)  # ours_rowspec split (0: teacher's kappa)
    ap.add_argument("--widths", default="")     # student widths m / d (default: the teacher's width);
                                                # ours_rowspec then uses kappa = sqrt(d / m)
    ap.add_argument("--impl", default="library", choices=["library", "sparrow_jax"])  # direction (docstring)
    ap.add_argument("--polar", default="svd", choices=list(sparrow_jax.POLAR_METHODS))
    ap.add_argument("--polar_steps", type=int, default=5)
    ap.add_argument("--oracle", default="rowclip", choices=["rowclip", "sparrow", "sparrow_rho"])
    ap.add_argument("--rho", default="sched")   # sparrow_rho: a number in [0, 1] (c = rho / (1 - rho) RMS row norm) or sched
    a = ap.parse_args()
    assert a.impl == "sparrow_jax" or a.polar == "svd", "--polar needs --impl sparrow_jax"
    assert a.impl == "library" or a.oracle == "rowclip", "--oracle needs --impl library"
    variant = {} if a.impl == "library" else dict(impl=a.impl, polar=a.polar, polar_steps=a.polar_steps)
    if a.oracle != "rowclip":
        variant.update(oracle=a.oracle, **({"rho": a.rho if a.rho == "sched" else float(a.rho)}
                                           if a.oracle == "sparrow_rho" else {}))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["teacher"], r["set"], r["beta"], r["eta"], r["seed"], r.get("kappa_student", 0.0),
                 r.get("steps", 1000), r.get("m") if a.widths else None,
                 tuple((k, r[k]) for k in ("impl", "polar", "polar_steps", "oracle", "rho") if k in r))
                for r in map(json.loads, open(path))}
    zte = jax.random.normal(jax.random.PRNGKey(123), (4096, a.d))
    for name in a.teachers.split(","):
        Wt, kappa, b0 = make_teacher(name, a.d, jax.random.PRNGKey(1))
        yte = np.asarray(jnp.argmax(logits(Wt, zte), 1))
        rt, _, Lt = radii(Wt, zte, yte)
        widths = [int(float(w) * a.d) for w in a.widths.split(",")] if a.widths else [Wt[2].shape[0]]
        for m, beta in itertools.product(widths, map(float, a.betas.split(","))):
            B = beta * b0
            ks = a.kappa_student or (math.sqrt(a.d / m) if a.widths else kappa)
            if a.families:
                cells = [(f,) + family_sets(f, B, ks, a.d, m, a.oracle, a.rho)
                         for f in a.families.split(",")]
            else:
                cells = [("spec", DenseGeometry(spec=math.sqrt(B)), None),
                         ("rowspec", DenseGeometry(spec=math.sqrt(B / ks), row=math.sqrt(B * ks)), None)]
            for sname, gv, wo_geo in cells:
                for eta in map(float, a.etas.split(",")):
                    if (name, sname, beta, eta, a.seed, a.kappa_student, a.steps, m if a.widths else None,
                            tuple(variant.items())) in done:
                        continue
                    t0 = time.time()
                    W = train(Wt, gv, jax.random.PRNGKey(100 + a.seed), a.d, a.steps, eta,
                              wo_geo=wo_geo, fam=sname, m=m if a.widths else None, impl=a.impl,
                              polar=a.polar, polar_steps=a.polar_steps,
                              rho_sched=a.oracle == "sparrow_rho" and a.rho == "sched" and sname == "ours_rowspec")
                    r, correct, L = radii(W, zte, yte)
                    u = squash(zte[:2048], 1.0)
                    h = np.abs(np.asarray(jax.nn.silu(u @ W[2].T) * (u @ W[3].T)))
                    keff = float(np.mean(h.sum(1) ** 2 / np.maximum((h ** 2).sum(1), 1e-30)))
                    rec = dict(teacher=name, set=sname, beta=beta, eta=eta, seed=a.seed, kappa=round(kappa, 4),
                               m=m, kappa_split=round(ks, 4), kappa_student=a.kappa_student, steps=a.steps, k_eff=round(keff, 1),
                               acc=round(float(correct.mean()), 4), L=round(L, 3),
                               radius_median=round(float(np.median(r)), 5), radius_mean=round(float(r.mean()), 5),
                               cert={str(e): round(float((r > e).mean()), 4) for e in (0.01, 0.03, 0.1)},
                               teacher_radius_median=round(float(np.median(rt)), 5), teacher_L=round(Lt, 3),
                               rho_over_sigma=round(float(row_max(W[2]) / sigma_max(W[2])), 3),
                               sec=round(time.time() - t0, 1), **variant)
                    print(json.dumps(rec), flush=True)
                    open(path, "a").write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
