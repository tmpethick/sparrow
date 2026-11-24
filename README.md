# Sparrow

Code for the experiments of ["How to build and train your recurrent-depth model: A fixed-point perspective"](https://arxiv.org/abs/2611.31513).

## Stand-alone reference implementation

[`sparrow_jax.py`](sparrow_jax.py) and
[`sparrow_torch.py`](sparrow_torch.py) implement the Sparrow direction
and optimizer, each in one short file that imports only JAX or PyTorch (copy either into your project). Each matrix gets
an update direction based on its associated constraint set, given by its `kind`:

| Block | Spectral constraint: `"spectral"` | Row constraint: `"row"` | Both constraints: `"gate_value"` |
| --- | --- | --- | --- |
| MLP | W_1, W_2 | – | – |
| ReLU² MLP | W_2 | – | W_1 |
| SwiGLU | W_o | – | W_g, W_v |
| Attention | W_Q, W_K, W_V, W_O | – | – |
| Gated attention | W_Q, W_K, W_V, W_O | W_a (gate) | – |
| Gated SSM | W_V, W_O | W_a (gate) | – |

Embeddings, the head, gains and biases are not hidden matrices and stay with AdamW. Each kind
has its own direction:

| `kind` | Constraint | Direction |
| --- | --- | --- |
| `"spectral"` | ‖W‖_op ≤ s | Spectral_s |
| `"row"` | ‖W‖_{2→∞} ≤ r | Row_r |
| `"gate_value"` | both | RowClip_r ∘ Spectral_s ∘ Row_r |

### Sparrow as drop-in replacement of Muon/MuonH

If you already train with Scion, Muon or MuonH, keep everything and change only the optimizer:

1. **Swap.** Use `Sparrow` on the matrices Muon trained. Everything else stays on AdamW.
2. **Set s** to the factor your Muon multiplies UVᵀ by. The implementation
   supports a number, a function `(out, in) -> s`, or a preset name (`"muon"`, `"mup"`, `"rms"`).
   The row radius r is set automatically from s, as r = s·min(1, √(in/out)).
3. **Pick kinds** from the table above: `"gate_value"` for SwiGLU W_g, W_v, `"row"` for the gate
   matrix W_a of gated attention or a gated SSM, and `"spectral"` for the rest.
4. **If using MuonH**, set `weight_norm=True`. Weight normalization keeps each matrix at its
   initial Frobenius norm R, so a matrix that starts at zero has R = 0 and stays at zero.
   So make sure that no matrix is zero-initialized, such as zero-initialized output projections.

Sparrow should already improve on Muon without retuning, but doubling the step size may improve it
further.

In PyTorch, with Polar Express (Amsel et al. 2025) for the polar factor:

```python
from sparrow_torch import Sparrow
groups = [dict(params=gates_and_values, kind="gate_value"),   # SwiGLU W_g, W_v
          dict(params=attention_gates, kind="row"),            # attention or SSM gates W_a
          dict(params=other_matrices)]                         # kind="spectral" (default): Muon's step

# with weight decay (as MuonW / Scion)
opt = Sparrow(groups, lr=lr, momentum=0.95, s="muon", weight_decay=wd,
              polar="polarexpress", polar_steps=5)

# weight normalized (as MuonH)
opt = Sparrow(groups, lr=lr, momentum=0.95, s="muon", weight_norm=True,
              polar="polarexpress", polar_steps=5)
```

In JAX, `sparrow_optimizer` is the same optimizer as an optax-style gradient transformation (it needs
only JAX). `kind` is one kind for every matrix, or a pytree of kinds whose leaves line up with the
matrices, and `lr` can be a schedule:

```python
import optax
from sparrow_jax import sparrow_optimizer
kinds = dict(w_gate="gate_value", w_value="gate_value", w_attn_gate="row", w_out="spectral")

# with weight decay (as MuonW / Scion)
hidden = sparrow_optimizer(lr=lr, momentum=0.95, s="muon", weight_decay=wd, kind=kinds,
                           polar="polarexpress", polar_steps=5)

# weight normalized (as MuonH)
hidden = sparrow_optimizer(lr=lr, momentum=0.95, s="muon", weight_norm=True, kind=kinds,
                           polar="polarexpress", polar_steps=5)

opt = optax.multi_transform(dict(sparrow=hidden, adamw=optax.adamw(1e-3)), labels)
```

Notes on both files:

- **Weight decay** is decoupled and scaled by `lr`: W ← (1 − lr·wd) W − lr·D.
- **Weight normalization** keeps each matrix at its initial Frobenius norm R (as in Hyperball), with the
  step scaled by R/(s·√min(out, in)) instead of normalizing the update direction, which we found to not change performance.
- **Polar factor**: `"polarexpress"` by default, or `"newtonschulz5"` (Muon's); both use matrix products
  only, so the spectral constraint holds only approximately, while the row constraints hold exactly. `"svd"` is exact. In PyTorch, `polar` can also be
  your own Muon function `(G, steps) -> UVᵀ`.
- **Shapes**: weights are (out, in) as in `torch.nn.Linear`, with r = s·min(1, √(in/out)); a square or wide
  `"gate_value"` matrix therefore just gets Spectral_s.
- The paper's nanoGPT runs use `nesterov=False`, `weight_decay=0` and `polar="svd"`.
- `tests/test_sparrow.py` checks both files against the library's LMO and against each other.

## Experiments

Everything runs on a CPU except `experiments/nanogpt_gpu/`. [REPRODUCE.md](REPRODUCE.md) regenerates the
paper's results.

- **`experiments/synthetic.py`**: certified robustness of one SwiGLU block on teacher-student classification.
- **`experiments/nanogpt.py`**: Muon, Sparrow, Aurora and the exact joint LMO as drop-in updates on nanoGPT Shakespeare
  (`--method`, see its docstring), plain or looped (`--loop`).
- **`experiments/nanogpt_gpu/`**: Sparrow versus Muon on the modded-nanoGPT GPT-2 record (not in the paper).
- **`experiments/solvers.py`**: exact and inexact solvers of the joint LMO.

## Layout

```text
sparrow/              library
  approx_oracles.py   the directions of the joint row/spectral set K_{r,s}: Spectral_s (Muon), Row_r,
                      RowClip∘Spectral, Sparrow, Aurora, the exact LMO by DRS. DenseGeometry for
                      projected training (LMO and projection, JAX)
  iterative_solvers.py
                      accurate iterative solvers of the joint LMO for benchmarking: exact DRS, iDRS,
                      rank-k subspace DRS, and the clipping candidates as baselines (NumPy)
sparrow_jax.py        minimal stand-alone Sparrow direction and optimizer (JAX)
sparrow_torch.py      minimal stand-alone Sparrow direction and optimizer (PyTorch)
experiments/          one file per experiment, run as `python experiments/<name>.py` (except nanogpt_gpu/)
  solvers.py          rank-adaptive iDRS versus fixed-rank DRS
  synthetic.py        synthetic teacher-student classification
  nanogpt.py          nanoGPT on Shakespeare (CPU)
  nanogpt_gpu/        PyTorch modded-nanoGPT GPT-2: Muon, Sparrow and Aurora on 4×H100 (Modal) or one local device
scripts/              figures and tables from the run records
runs/                 records of the paper's runs, one folder per sweep (logs, results, launch scripts)
tests/                test_approx_oracles, test_iterative_solvers*, test_sparrow, test_<experiment>
```

## Setup

```sh
cd sparrow
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,plot]'
.venv/bin/python -m pytest -q
```

Run from the repository root, for example
`.venv/bin/python experiments/synthetic.py --out runs/tsr`. Run outputs go to `runs/`, which
holds the records of the paper's runs, and downloaded data to `data/` (override with `SPARROW_DATA`).
