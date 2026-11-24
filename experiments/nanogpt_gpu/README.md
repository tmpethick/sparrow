# GPT-2 speedrun: Muon, Sparrow and Aurora (modded-nanoGPT)

`train_gpt2.py` is the training script of the modded-nanoGPT
[2024-10-17 DistributedMuon record](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_1_short/2024-10-17_DistributedMuon)
(GPT-2 small on FineWeb, 5,100 steps, val loss 3.2747 on 8×H100). Its defaults are the record's hyperparameters,
and our changes are marked `[sparrow]` in the source. `--optimizer` picks the update of the block matrices
(the embeddings and head stay on AdamW):

| `--optimizer` | Update |
| --- | --- |
| `muon` (default) | the record's Muon |
| `sparrow` | `Sparrow` from [`sparrow_torch.py`](../../sparrow_torch.py): Sparrow on the tall ReLU² input `mlp.c_fc`, Muon's step on the rest |
| `aurora` | Muon with Aurora (Algorithm 3 of Dewulf et al. 2026) on `mlp.c_fc` |
| `sparrow_rho` | `sparrow` with the row-clipping threshold ρ on `mlp.c_fc`: `--rho` a number in [0, 1], or `sched` (default) to follow the warmdown |

Other flags: `--seed` (model init; the record is unseeded), `--lr` (block matrices, default 0.00036), `--head-lr`
(default 0.0036), `--run-name`, `--wandb-project`, `--no-wandb`, and `--num-iterations` / `--val-loss-every` for
smoke tests. On 4 GPUs the script keeps the global batch of 512 with two gradient-accumulation steps.

## Run on Modal (4×H100)

Setup, once: a W&B secret and a volume with the FineWeb shards (28 train shards and the validation shard,
about 5.8 GB; the volume name defaults to `accmoda-fineweb10b`, or set `MODAL_DATA_VOLUME_NAME`):

```bash
pip install modal huggingface_hub && modal setup
modal secret create wandb-secret WANDB_API_KEY=<your key>
python download_data.py --train-shards 28                  # -> data/fineweb10B
modal volume create accmoda-fineweb10b
modal volume put accmoda-fineweb10b data/fineweb10B/ /
```

Runs (about 33 min and $8.4 each; `-d` detaches):

```bash
modal run modal_app.py --action smoke --optimizers muon,sparrow,aurora   # 10 steps each on one H100
MODAL_TRAIN_TIMEOUT_MINUTES=38 modal run -d modal_app.py --train-args "--seed 0 --run-name muon-s0"
MODAL_TRAIN_TIMEOUT_MINUTES=38 modal run -d modal_app.py --train-args "--optimizer sparrow --seed 0 --run-name sparrow-s0"
MODAL_TRAIN_TIMEOUT_MINUTES=42 modal run -d modal_app.py --train-args "--optimizer aurora --seed 0 --run-name aurora-s0"
# twice the record's step size for the block matrices (the head's AdamW step size is unchanged)
MODAL_TRAIN_TIMEOUT_MINUTES=38 modal run -d modal_app.py --train-args "--optimizer sparrow --seed 0 --lr 0.00072 --run-name sparrow-lr2x-s0"
MODAL_TRAIN_TIMEOUT_MINUTES=42 modal run -d modal_app.py --train-args "--optimizer aurora --seed 0 --lr 0.00072 --run-name aurora-lr2x-s0"
```

Runs with the same `--seed` share the initialization and data order, so compare `val/loss` at the same `step`.
Logs and the final checkpoint go to the Modal volume `halpern-tbptt-nanogpt-logs` (`logs/<uuid>.txt`,
`logs/<uuid>/state_step005100.pt`).

## Run on a laptop (one device)

`--device mps` (or `cpu`) runs one process without torchrun, NCCL or `torch.compile`. `make_local_data.py` writes
small shards, and the model and batch shrink with `--n-layer`, `--n-head`, `--n-embd`, `--vocab-size`,
`--batch-size`, `--device-batch-size`, `--sequence-length`, `--warmdown-iters` and `--val-tokens`:

```bash
pip install torch numpy wandb tiktoken
python make_local_data.py bpe      # Complete Works of Shakespeare in GPT-2 BPE -> data/shk_bpe
python train_gpt2.py --device mps --data-dir data/shk_bpe --optimizer sparrow --no-wandb \
  --n-layer 4 --n-head 4 --n-embd 256 --batch-size 16 --device-batch-size 16 --sequence-length 256 \
  --num-iterations 500 --warmdown-iters 150 --val-tokens 65536 --val-loss-every 50
```

`make_local_data.py` reads `sparrow/data/shakespeare_complete.txt`, which `experiments/nanogpt.py` downloads on
its first run with `--data complete`.
