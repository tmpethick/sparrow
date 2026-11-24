"""Sparrow: minimal PyTorch reference.

For a weight W of shape (m, d) = (out_features, in_features),
`sparrow(G, kind)` is the direction

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
polar="polarexpress" (default), "newtonschulz5" or "svd" (exact) computes U V^T
in float32; the first two meet the spectral constraint only approximately.

`Sparrow` is the optimizer for the hidden matrices; embeddings, head,
gains and biases stay with AdamW:

    M <- mu M + (1 - mu) grad
    D = sparrow(M, kind)
        (nesterov: D = sparrow(grad + mu (M - grad), kind))
    W <- (1 - lr wd) W - lr D

or, with weight_norm=True (Hyperball without normalizing D),

    W <- R N((1 - lr wd) W - lr c D),
    c = R / (s sqrt(min(m, d))),  R = ||W_0||_F,  N(X) = X / ||X||_F.

The paper's runs use nesterov=False.
"""
import torch

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


def polar_factor(G, method="polarexpress", steps=5):
    """U V^T for G = U S V^T.

    "newtonschulz5" and "polarexpress" use matrix products only: `steps`
    odd quintic iterations X <- a X + (b A + c A^2) X, A = X X^T, on the
    wide orientation of G. method can also be a function
    (G, steps) -> U V^T, e.g. the Muon implementation the other matrices
    use, so that both share one solver; its output is cast back to G's
    dtype.
    """
    if callable(method):
        return method(G, steps=steps).to(G.dtype)
    if method == "svd":
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        return U @ Vh
    assert method in POLAR_METHODS, method
    if method == "newtonschulz5":
        coeffs, margin = [NS5] * steps, 1.0
    else:
        padded = POLAR_EXPRESS + POLAR_EXPRESS[-1:] * steps
        coeffs, margin = padded[:steps], 1.01
    tall = G.shape[0] > G.shape[1]
    X = G.T if tall else G
    # every singular value <= 1 / margin
    X = X / (margin * X.norm() + EPS)
    for a, b, c in coeffs:
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


def spectral(G, s, polar="polarexpress", polar_steps=5):
    return s * polar_factor(G, polar, polar_steps)


def row(G, r):
    return r * G / G.norm(dim=1, keepdim=True).clamp_min(EPS)


def row_clip(G, r):
    scale = r / G.norm(dim=1, keepdim=True).clamp_min(EPS)
    return G * scale.clamp(max=1.0)


def rowclip_c(G, rho):
    """RowClip_c(G), c = rho / (1 - rho) gbar with gbar the RMS row norm,
    up to a positive factor (which the polar factor ignores).

    Returns the rows G_i / max((1 - rho) ||G_i||, rho gbar), of norm about
    one at any rho in [0, 1].
    """
    rn = G.norm(dim=1, keepdim=True)
    gbar = rn.pow(2).mean().sqrt()
    return G / torch.maximum((1 - rho) * rn, rho * gbar).clamp_min(EPS)


@torch.no_grad()
def sparrow(G, kind="spectral", s=1.0, polar="polarexpress", polar_steps=5,
            r=None, rho=0.0):
    """The Sparrow direction of G.

    kind: "gate_value" (SwiGLU gate and value), "row" (attention or SSM
        gate) or "spectral".
    polar: how the polar factor is computed, "polarexpress" (default)
        or "newtonschulz5" (polar_steps iterations), "svd" (exact), or a
        function (see polar_factor).
    s: a number, a function (m, d) -> s, or a name in SCALES.
    r: the row radius, s * min(1, sqrt(d / m)) by default.
    rho in [0, 1]: the clip level of "gate_value" (rowclip_c; 0: Sparrow,
        1: RowClip_r Spectral_s).
    """
    m, d = G.shape
    s = spectral_radius(s, G.shape)
    r = s * min(1.0, (d / m) ** 0.5) if r is None else r
    X = G.float()
    if kind == "gate_value" and r < s:
        Y = row(X, r) if rho == 0 else rowclip_c(X, rho)
        Z = row_clip(spectral(Y, s, polar, polar_steps), r)
    elif kind == "row":
        Z = row(X, r)
    else:
        assert kind in ("gate_value", "spectral"), kind
        Z = spectral(X, s, polar, polar_steps)
    return Z.to(G.dtype)


class Sparrow(torch.optim.Optimizer):
    """Sparrow for 2-D weights stored (out, in), as in torch.nn.Linear.

    Give each parameter group its kind, e.g.

        Sparrow([dict(params=gates_and_values, kind="gate_value"),
                 dict(params=attention_gates, kind="row"),
                 dict(params=other_matrices)],  # kind="spectral"
                lr=0.01)

    s is the spectral radius (a number, a function (m, d) -> s of each
    matrix's shape, or a name in SCALES, e.g. s="muon"), and lr the step
    in units of it (as for Muon). weight_decay is decoupled and scaled by
    lr, as in AdamW. polar and polar_steps choose how the polar factor is
    computed, and rho the clip level of "gate_value" (see sparrow; a
    schedule can set group["rho"] per step).
    """

    def __init__(self, params, lr=0.01, momentum=0.9, nesterov=True,
                 kind="spectral", s=1.0, weight_decay=0.0,
                 weight_norm=False, polar="polarexpress", polar_steps=5, rho=0.0):
        assert kind in ("gate_value", "row", "spectral"), kind
        assert 0 <= rho <= 1, rho
        assert callable(polar) or polar in POLAR_METHODS, polar
        spectral_radius(s, (1, 1))  # fail early on an unknown name
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        kind=kind, s=s, weight_decay=weight_decay,
                        weight_norm=weight_norm, polar=polar,
                        polar_steps=polar_steps, rho=rho)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mu = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, (
                    "Sparrow is for matrices; "
                    "use AdamW for the other parameters")
                state = self.state[p]
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                    if group["weight_norm"]:
                        state["radius"] = p.norm()
                buf = state["momentum"]
                buf.lerp_(p.grad, 1 - mu)
                g = p.grad.lerp(buf, mu) if group["nesterov"] else buf
                s = spectral_radius(group["s"], p.shape)
                D = sparrow(g, group["kind"], s, group["polar"],
                            group["polar_steps"], rho=group["rho"])
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                if group["weight_norm"]:
                    R = state["radius"]
                    alpha = lr * float(R) / (s * min(p.shape) ** 0.5)
                    p.sub_(D, alpha=alpha)
                    p.mul_(R / p.norm().clamp_min(EPS))
                else:
                    p.sub_(D, alpha=lr)
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    d, m = 64, 256  # SwiGLU block: m = 4d
    gate = torch.nn.Linear(d, m, bias=False)
    value = torch.nn.Linear(d, m, bias=False)
    out = torch.nn.Linear(m, d, bias=False)
    opt = Sparrow([dict(params=[gate.weight, value.weight],
                        kind="gate_value"),
                   dict(params=[out.weight])],  # W_o: "spectral" (Muon)
                  lr=0.02)
    x = torch.randn(512, d)
    y = torch.roll(x, 1, dims=1)
    for step in range(301):
        h = torch.nn.functional.silu(gate(x)) * value(x)
        loss = ((out(h) - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")
