# Reproducing the experiments

Commands for the experiments of [*How to build and train your recurrent-depth model: A fixed-point
perspective*](https://arxiv.org/abs/2611.31513):

| Section | Figures and tables |
| --- | --- |
| [1. Setup](#1-setup) | `norm_profiles.pdf` |
| [2. Synthetic](#2-synthetic-certified-robustness) | `ts_robust.pdf` and the robustness tables, `ts_width_cls.pdf`, `ts_sparrow.pdf` |
| [3. Training speed](#3-training-speed) | `speed.pdf`, `lmo_exact.pdf`, `speed_lr.pdf`, `batch_size.pdf`, `polar_approx.pdf` |
| [4. Gated attention](#4-gated-attention) | `gate.pdf` |
| [5. Looped nanoGPT](#5-looped-nanogpt) | `loop_shakespeare.pdf` |
| [6. Joint LMO solvers](#6-joint-lmo-solvers) | the iDRS table |

Run all commands from the repository root.

The repository includes the records of the reported runs in `runs/`. The last code block of each section
only reads these records: it writes the figures to `figures/` and prints the tables, so you can regenerate
every figure and table without rerunning an experiment.

To rerun an experiment yourself, first move the records out of the way, for example with
`mv runs runs-reported`. The experiment scripts write to the same folders and would otherwise trip over
the old records: `synthetic.py` would skip every setting it already finds in a `results.jsonl`, and
`nanogpt.py` would try to continue each finished run from its checkpoint, which is not included in the
repository, and fail.

`--paper` makes the plot scripts typeset text with LaTeX to match the paper (this needs LaTeX installed).
Add `--mpl-font` to use matplotlib's default font instead.

## 1. Setup

Everything runs on a CPU. The reported runs used the versions below on a 4-core Linux VM on an
Apple-silicon MacBook, except where a section names JAX 0.11.2 on macOS.

| Package | Version |
| --- | --- |
| Python | 3.10 |
| JAX | 0.6.2 |
| Equinox | 0.13.8 |
| Optax | 0.2.8 |
| NumPy | 2.2.6 |
| Matplotlib (plots only) | 3.10 |

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev,plot]'
python -m pytest -q
mkdir -p figures
python scripts/plot_norm_profiles.py figures/norm_profiles.pdf --paper   # normalization profiles (no runs needed)
```

## 2. Synthetic: certified robustness

[`experiments/synthetic.py`](experiments/synthetic.py). Each command appends to `<out>/results.jsonl`.
The methods (`--families`) and teachers (`--teachers`) are named as follows:

| Code | Paper |
| --- | --- |
| `sgd_frob`, `adam_linf`, `muon_spec` | SGD + Frobenius ball, Adam + entrywise box, Muon + spectral ball |
| `ours_rowspec` | U_1 (RowClip∘Spectral) + joint row/spectral set; U_3 (Sparrow) with `--impl sparrow_jax` |
| `gauss_4`, `few_8` | spread and concentrated teacher |

**Robustness frontiers and tables.** A run takes 1–10 s (about 50 s at 3000
steps); the whole block takes about 1.5 h on the 4-core VM. The step sizes 0.001, 0.003 and 0.3 were
added later with JAX 0.11.2 on macOS; rerunning the grid on another platform moves the per-seed
frontiers by up to 11%.

```sh
E3=0.01,0.03,0.1
EA=0.001,0.003,0.01,0.03,0.1,0.3; EM=0.001,0.003,0.01,0.03,0.1   # SGD/Adam, Muon/ours
# spread teacher, 1000 steps
python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --families sgd_frob,adam_linf --etas $EA --betas 0.5,1,2,4
python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --families muon_spec,ours_rowspec --etas $EM --betas 0.5,1,2,4
python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --families sgd_frob,adam_linf --etas $EA --betas 16,64,256,1024,4096
python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --families muon_spec,ours_rowspec --etas $EM --betas 16,64
for s in 1 2; do
  python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --seed $s --families sgd_frob,adam_linf --etas $EA --betas 64,256,1024,4096
  python experiments/synthetic.py --out runs/tsr-fam --teachers gauss_4 --seed $s --families muon_spec,ours_rowspec --etas $EM --betas 0.5,1,2,4,16
done
# concentrated teacher
python experiments/synthetic.py --out runs/tsr-fam --teachers few_8 --families sgd_frob,adam_linf --etas $EA --betas 1,4,16,64,256,1024,4096
python experiments/synthetic.py --out runs/tsr-fam --teachers few_8 --families muon_spec,ours_rowspec --etas $EM --betas 0.5,1,2,4,16,64
for s in 1 2; do
  python experiments/synthetic.py --out runs/tsr-fam --teachers few_8 --seed $s --families sgd_frob,adam_linf --etas $EA --betas 16,64,256,1024,4096
  python experiments/synthetic.py --out runs/tsr-fam --teachers few_8 --seed $s --families muon_spec,ours_rowspec --etas $EM --betas 1,4,16,64
done
# spread teacher, 3000 steps, seed 0
python experiments/synthetic.py --out runs/tsr-long --teachers gauss_4 --steps 3000 --families sgd_frob,adam_linf --etas $E3 --betas 64,256,1024,4096
python experiments/synthetic.py --out runs/tsr-long --teachers gauss_4 --steps 3000 --families muon_spec,ours_rowspec --etas 0.01,0.03 --betas 0.5,1
python experiments/synthetic.py --out runs/tsr-long --teachers gauss_4 --steps 3000 --families muon_spec,ours_rowspec --etas 0.01 --betas 2,4,16
# choice of the split kappa, seed 0
for k in 0.21 0.5 0.85; do
  python experiments/synthetic.py --out runs/tsr-kappa --teachers gauss_4 --families ours_rowspec --kappa_student $k --etas 0.01,0.03 --betas 0.5,1,2,4,16
done
python experiments/synthetic.py --out runs/tsr-kappa --teachers few_8 --families ours_rowspec --kappa_student 0.5 --etas 0.01,0.03 --betas 1,4,16,64
```

```sh
python scripts/plot_ts_robust.py runs figures/ts_robust.pdf --paper
python scripts/table_ts_robust.py runs/tsr-fam gauss_4 0.65,0.70,0.75 --latex
python scripts/table_ts_robust.py runs/tsr-long gauss_4 0.70,0.75,0.80 3000 --latex
python scripts/table_ts_robust.py runs/tsr-fam few_8 0.80,0.85,0.90 --latex
```

**Width at a fixed certificate.** About 35 min per seed with JAX 0.11.2 on macOS;
seeds can run in parallel.

```sh
for s in 0 1 2; do
  python experiments/synthetic.py --out runs/tsr-width-s$s --teachers gauss_16 --widths 1,2,4,8,16 \
    --betas 0.5,1,4,16 --etas 0.01,0.03,0.1 --families muon_spec,ours_rowspec --seed $s
done
python scripts/plot_ts_width.py runs figures/ts_width_cls.pdf --paper
```

**U_1 versus Sparrow's U_3.** All runs of this figure use JAX 0.11.2 on macOS: Muon
and U_1 are the `muon_spec,ours_rowspec` commands of `runs/tsr-fam` above, written to
`runs/tsr-fam-local-{gauss,few}` (η₀ ∈ {0.01, 0.03, 0.1}) and `runs/tsr-fam-ext-{gauss,few}`
(η₀ ∈ {0.001, 0.003}). U_3 takes about 10 min with the jobs in parallel:

```sh
R="python experiments/synthetic.py --families ours_rowspec --impl sparrow_jax --etas 0.001,0.003,0.01,0.03,0.1"
$R --out runs/tsr-sp-g0 --teachers gauss_4 --seed 0 --betas 0.5,1,2,4,16,64
for s in 1 2; do $R --out runs/tsr-sp-g12 --teachers gauss_4 --seed $s --betas 0.5,1,2,4,16; done
$R --out runs/tsr-sp-f0 --teachers few_8 --seed 0 --betas 0.5,1,2,4,16,64
for s in 1 2; do $R --out runs/tsr-sp-f12 --teachers few_8 --seed $s --betas 1,4,16,64; done
python scripts/plot_ts_sparrow.py runs figures/ts_sparrow.pdf --paper
```

All runs of the figure use the teacher's split κ* (the default of `--kappa_student`).

## 3. Training speed

[`experiments/nanogpt.py`](experiments/nanogpt.py) on Tiny Shakespeare (downloaded to `data/` on the
first run). One pass over the training split is 1307 steps. A run takes about 2 min alone on an
Apple-silicon CPU (Aurora about 10% longer), and runs can go in parallel. `--method` chooses the update of each
hidden matrix (the table in the docstring of `nanogpt.py`): `muon`, `sparrow`, `aurora` and `rowclip` (U_1)
differ only on the SwiGLU gate and value unless the model has an attention gate. The logs of the reported runs
record the flags of the earlier interface (`--opt spectral` for `--method muon`, `--opt targeted --row_kappa 1
--oracle X` for `--method X`, `--opt aurora` for `--method aurora` with an attention gate, `--opt spectral --gate_opt sparrow` for
`--method sparrow_gate`); the refactor reproduces their losses.

**Muon, Aurora and Sparrow.** Aurora is Algorithm 3 of Dewulf et al. (2026) with K = 2 and
β = 1/2 (`--method aurora`). The figure shows Muon at its best step size 0.01, Sparrow at 0.01 and 0.02, and
Aurora at its best of {0.01, 0.02, 0.04} (0.02), chosen by `plot_speed.py` from each step
size's mean final loss minus Muon's. `time_directions.py` measures the cost of each direction and needs an idle machine.

```bash
run() { mkdir -p runs/speed/$1; python experiments/nanogpt.py --out runs/speed/$1 \
          --update unconstrained --sampler epoch --epochs 1 --eval_every 100 "${@:2}" \
          > runs/speed/$1/stdout.log 2>&1 && touch runs/speed/$1/done; }
for s in $(seq 0 9); do sfx=$([ $s = 0 ] || echo -s$s)
  run spectral-lr0.01$sfx --method muon    --lmo_lr 0.01 --seed $s
  run aurora-lr0.01$sfx   --method aurora  --lmo_lr 0.01 --seed $s
  run sparrow-lr0.01$sfx  --method sparrow --lmo_lr 0.01 --seed $s
done
for s in 0 1; do sfx=$([ $s = 0 ] || echo -s$s); for lr in 0.005 0.02; do                     # Muon step size
  run spectral-lr$lr$sfx --method muon --lmo_lr $lr --seed $s
done; done
for s in $(seq 0 9); do sfx=$([ $s = 0 ] || echo -s$s); for o in sparrow aurora; do           # step size 0.02
  run $o-lr0.02$sfx --method $o --lmo_lr 0.02 --seed $s
done; done
for s in 0 1 2 3 4; do sfx=$([ $s = 0 ] || echo -s$s)                                         # Aurora at 0.04
  run aurora-lr0.04$sfx --method aurora --lmo_lr 0.04 --seed $s
done
python scripts/time_directions.py      # -> runs/speed/cost.json
```

```bash
python scripts/plot_speed.py figures/speed.pdf --paper   # also writes runs/speed/results.jsonl
```

The Aurora runs in `runs/speed` record the oracle under its earlier name `aurora3` (the same map), and
`runs/speed/run_*.sh` are the scripts that launched them.

**Comparison with the exact LMO.** Muon's runs with `--lmo_track 1`, which at every
evaluation compares RowClip(Spectral), Sparrow and Aurora on the momentum of each SwiGLU gate and value
matrix with the joint LMO (DRS, 300 iterations). 3 runs, about 5 min with 3 in parallel.

```bash
for s in 0 1 2; do out=runs/speed-track/spectral-lr0.01-s$s; mkdir -p $out
  python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method muon --lmo_lr 0.01 --seed $s --lmo_track 1 > $out/stdout.log 2>&1 && touch $out/done
done
python scripts/plot_lmo_exact.py figures/lmo_exact.pdf --paper
```

**Step-size sweep.** 60 runs, about 2.5 h with 5 in parallel. The Aurora runs at 0.01, 0.02
and 0.04 are copies of the identical `runs/speed/aurora-lr*` runs (`runs/lr-sweep/run_aurora.sh`).

```bash
run() { out=runs/lr-sweep/$1-lr$2-s$3; mkdir -p $out
  O="--method $1"
  python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $2 --seed $3 $O > $out/stdout.log 2>&1 && touch $out/done; }
for lr in 0.04 0.02 0.01 0.005 0.0025; do for s in 0 1 2; do for o in muon sparrow rowclip aurora; do run $o $lr $s; done; done; done
```

```bash
python scripts/plot_lr_sweep.py figures/speed_lr.pdf --paper
```

**Batch size**, on the Complete Works (`--data complete`, downloaded on the first
run), 750 steps per run: 60 runs at η = 0.01 (about 75 min with 5 in parallel) and 48 runs at the other
step sizes (about 70 min with 4 in parallel).

```bash
run() { root=runs/bs-complete o=$1 b=$2 s=$3; out=$root/$o-b$b-s$s; mkdir -p $out
  O="--method $o"
  python experiments/nanogpt.py --out $out --data complete --update unconstrained --sampler epoch \
    --lmo_lr 0.01 --eval_batch 96 --batch $b --seed $s --steps 750 --eval_every 50 $O \
    > $out/stdout.log 2>&1 && touch $out/done; }
for b in 96 48 24 12; do for s in 0 1 2 3 4; do for o in muon aurora sparrow; do run $o $b $s; done; done; done
runlr() { o=$1 b=$2 s=$3 lr=$4; out=runs/bs-lr/$o-b$b-s$s-lr$lr; mkdir -p $out
  O="--method $o"
  python experiments/nanogpt.py --out $out --data complete --update unconstrained --sampler epoch \
    --lmo_lr $lr --eval_batch 96 --batch $b --seed $s --steps 750 --eval_every 50 $O \
    > $out/stdout.log 2>&1 && touch $out/done; }
for b in 96 48 24 12; do for s in 0 1 2; do for o in muon aurora sparrow; do runlr $o $b $s 0.02; done; done; done
for s in 0 1 2; do for o in muon aurora sparrow; do runlr $o 96 $s 0.04; done; runlr muon 12 $s 0.005; done
```

```bash
python scripts/plot_batch_size.py figures/batch_size.pdf --paper
python scripts/table_batch_size.py runs/bs-complete     # per-step gaps
```

**Inexact polar factor.** Muon, Aurora and Sparrow at η = 0.02 in the setup above,
with every polar factor computed by the exact SVD or by k ∈ {2, 3, 4, 5, 6, 8} iterations of Polar Express
(Amsel et al. 2025) in float32. 63 runs (seeds 0, 1, 2), about 30 min with 5 in parallel.
[`runs/polar-approx/run_polar.py`](runs/polar-approx/run_polar.py) runs `nanogpt.py` with
`sparrow.approx_oracles._polar` replaced by `sparrow_jax.polar_factor(G, "polarexpress", k)`, which every
direction calls; the launcher skips finished runs.

```bash
bash runs/polar-approx/sweep.sh
```

```bash
python scripts/plot_polar_approx.py figures/polar_approx.pdf --paper   # also prints the paired gaps
```

## 4. Gated attention

The model of section 3 with a channelwise attention gate (`--attn_gate elem`), 4 heads of width 32
(square gate) or 128 (tall gate). A tall-gate run takes about 3 min alone on an Apple-silicon CPU.

**Muon, Aurora and Sparrow** (solid bars of `gate.pdf`). `--method aurora` is Aurora (Algorithm 3) on
every tall matrix, here also the tall gate W_a and W_Q, W_K, W_V, and Muon elsewhere; `--method sparrow_gate` is Sparrow on the gate only.

```bash
run() { mkdir -p runs/gate-attn/$1; python experiments/nanogpt.py --out runs/gate-attn/$1 \
          --update unconstrained --sampler epoch --epochs 1 --eval_every 100 --lmo_lr 0.01 \
          --attn_gate elem "${@:2}" > runs/gate-attn/$1/stdout.log 2>&1 && touch runs/gate-attn/$1/done; }
for s in $(seq 0 9); do for shape in square:32 tall:128; do sh=${shape%%:*}; hd=${shape##*:}
  run $sh-muon-s$s        --head_dim $hd --method muon --seed $s
  run $sh-aurora-s$s      --head_dim $hd --method aurora --seed $s
  run $sh-sparrowgate-s$s --head_dim $hd --method sparrow_gate --seed $s
  run $sh-sparrow-s$s     --head_dim $hd --method sparrow --seed $s
done; done
```

**Weight normalization** (hatched bars of `gate.pdf`), on the tall gate with fan-in initialization.
`--update hyperball` is MuonH and SparrowH; `--update hyperball_unnorm` is Sparrow with weight
normalization only. MuonH's step size is tuned first, and the Sparrow variants reuse it.

```bash
run() { mkdir -p runs/gate-h/$1; python experiments/nanogpt.py --out runs/gate-h/$1 \
          --init fanin --sampler epoch --epochs 1 --eval_every 100 \
          --attn_gate elem --head_dim 128 "${@:2}" > runs/gate-h/$1/stdout.log 2>&1 && touch runs/gate-h/$1/done; }
for lr in 0.005 0.01 0.02 0.04 0.08; do for s in 0 1; do
  run muonh-lr$lr-s$s --method muon --update hyperball --lmo_lr $lr --seed $s
done; done
for s in $(seq 0 9); do
  run muonh-lr0.01-s$s     --method muon    --update hyperball        --lmo_lr 0.01 --seed $s
  run sparrowh-lr0.01-s$s  --method sparrow --update hyperball        --lmo_lr 0.01 --seed $s
  run sparrowhu-lr0.01-s$s --method sparrow --update hyperball_unnorm --lmo_lr 0.01 --seed $s
done
```

```bash
python scripts/plot_gate.py figures/gate.pdf --paper --with-hyperball   # both blocks; prints the table
```

## 5. Looped nanoGPT

The tall-gate model of section 4 with 3 layers, the middle one iterated N times (`--loop N --beta
0.5`); `--tbptt H` backpropagates through the last H iterations only. Cells are named `PC<N>[H<H>]`,
except N = 2, which is `L1x2`. The 98 runs take about 2.5 h with 4 in parallel (2 min at N = 1 and
9 min at N = 16 under that load).

```bash
run() { opt=$1 arch=$2 lr=$3 s=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$s; mkdir -p $out
  case $arch in L1x2) N=2 H=0;; PC*H*) N=${arch#PC}; H=${N#*H}; N=${N%H*};; PC*) N=${arch#PC} H=0;; esac
  python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --attn_gate elem --head_dim 128 --n_layer 3 --beta 0.5 --loop $N --tbptt $H --lmo_lr $lr \
    --method $([ $opt = spectral ] && echo muon || echo sparrow) \
    --seed $s > $out/stdout.log 2>&1 && touch $out/done; }
for arch in PC1 L1x2 PC4 PC8 PC16 PC16H8 PC16H4; do for s in 0 1 2; do for opt in spectral sparrow; do
  for lr in 0.01 0.02; do run $opt $arch $lr $s; done; done; done; done
for s in 0 1 2; do for arch in L1x2 PC4; do run spectral $arch 0.005 $s; run sparrow $arch 0.04 $s; done; done
run spectral PC16 0.005 0; run sparrow PC16 0.005 0                     # step-size brackets
```

```bash
python scripts/plot_loopbs.py --paper figures/loop_shakespeare.pdf
python scripts/table_loopbs.py 'runs/loopbs/gate-tall-*'                # per-cell losses
```

The launch scripts of the reported runs are `runs/loopbs/gate_{grid,pcN,steps,pc16,depth}.sh`.

## 6. Joint LMO solvers

[`experiments/solvers.py`](experiments/solvers.py) (NumPy, CPU) compares exact DRS, iDRS and fixed-rank DRS
(k ∈ {16, 32, 64, 128}) on the 18 instances of the paper's iDRS table, until a certified relative duality gap of
1e-6 or 3000 iterations. Times are the median of 3 solves with single-threaded BLAS. The run writes
`results.csv`, `history.json`, `config.json`, `summary.md` and `comparison.png`; `--replot` redraws from an
existing run.

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/sparrow-mpl \
  python experiments/solvers.py --out runs/lmo-rank-adaptive
python scripts/table_lmo_idrs.py runs/lmo-rank-adaptive   # rows of the iDRS table
```

Expected totals (`runs/lmo-rank-adaptive/summary.md`): iDRS certifies 18/18 in 65 s, exact DRS in 186 s, and
k = 128, the smallest fixed rank that certifies every instance, in 120 s.
