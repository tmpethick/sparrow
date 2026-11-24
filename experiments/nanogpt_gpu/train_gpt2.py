import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
# [sparrow] the stand-alone Sparrow optimizer: next to this file on Modal, two levels up in the repo
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import sparrow_torch
from sparrow_torch import Sparrow
with open(sparrow_torch.__file__) as f:
    code += '\n# ' + '=' * 98 + '\n# sparrow_torch.py\n' + f.read() # logged with this file
import contextlib
import json
# [sparrow] --device mps|cpu: one local process, without torchrun, NCCL or torch.compile (default cuda: the record)
LOCAL = '--device' in sys.argv and sys.argv[sys.argv.index('--device') + 1] != 'cuda'
maybe_compile = (lambda f: f) if LOCAL else torch.compile

def all_reduce(t, op=dist.ReduceOp.SUM):
    # [sparrow] a no-op in a single local process
    if dist.is_initialized():
        dist.all_reduce(t, op=op)

# -----------------------------------------------------------------------------
# Muon optimizer

def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T

@maybe_compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    r"""
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

zeropower_backends = dict(svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5)

# [sparrow] Aurora (Algorithm 3 of Dewulf et al. 2026), as sparrow.approx_oracles.aurora_update with s = 1: from
# X = G / ||G||_F, repeat D <- D^beta diag(row norms of X)^(1 - beta), X <- Polar(sqrt(n/m) D^{-1} X) (polar factor by the
# given backend); their speedrun default K = 2, beta = 1/2, so two polar factors.
def aurora(G, zeropower_backend, steps, iters=2, beta=0.5):
    m, n = G.shape
    X = G.float() / G.norm().clamp_min(1e-12)
    D = torch.ones(m, 1, device=G.device)
    for _ in range(iters):
        D = D**beta * X.norm(dim=1, keepdim=True).clamp_min(1e-12)**(1 - beta)
        X = zeropower_backend((n / m)**0.5 * X / D, steps=steps).float()
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
        dualizer: [sparrow] 'muon' (the record) or 'aurora' (see `aurora`), per parameter group.
    """
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, backend='newtonschulz5', backend_steps=5,
                 rank=0, world_size=1):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend=backend, backend_steps=backend_steps,
                        dualizer='muon')
        super().__init__(params, defaults)
        self.rank = rank
        self.world_size = world_size

    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group['params'])
            updates_flat = torch.zeros(total_params, device=group['params'][0].device, dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(group['params']):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    # [sparrow] momentum as a convex combination, as in later modded-nanogpt records (the
                    # record sums, buf <- momentum*buf + g). The direction is scale-invariant, so only rounding
                    # changes, and Muon and Sparrow then feed the same input to Newton-Schulz.
                    buf.lerp_(g, 1 - momentum)
                    if group['nesterov']:
                        g = g.lerp(buf, momentum)
                    else:
                        g = buf
                    if group['dualizer'] == 'aurora': # [sparrow]
                        g = aurora(g, zeropower_backend, group['backend_steps'])
                    else:
                        g = zeropower_backend(g, steps=group['backend_steps'])
                    g *= max(g.size(0), g.size(1))**0.5 # scale to have update.square().mean() == 1
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            all_reduce(updates_flat)

            # deserialize and apply updates
            curr_idx = 0
            for p in group['params']:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

# -----------------------------------------------------------------------------
# [sparrow] Sparrow (sparrow_torch.py) as a drop-in replacement for Muon

# The constraints each block matrix needs (the paper's table of block constraints), as a sparrow_torch kind. Keys are
# parameter names within a block. This model's MLP is ReLU^2, whose W_1 needs both constraints, as the SwiGLU W_g
# and W_v do; the rest need the spectral constraint only, where Sparrow's direction is Muon's.
SPARROW_KINDS = {
    'attn.c_q.weight': 'spectral',     # attention W_Q
    'attn.c_k.weight': 'spectral',     # attention W_K
    'attn.c_v.weight': 'spectral',     # attention W_V
    'attn.c_proj.weight': 'spectral',  # attention W_O
    'mlp.c_fc.weight': 'gate_value',   # ReLU^2 W_1 (3072x768): RowClip_r(Spectral_s(Row_r(G))), r = s/2
    'mlp.c_proj.weight': 'spectral',   # ReLU^2 W_2
}

def sparrow_param_groups(named_params):
    # one group per kind. s='rms' is sqrt(max(out, in)) per matrix, the record's Muon update scale, so lr means
    # what it means for Muon. A matrix missing from SPARROW_KINDS is an error rather than a silent default.
    groups = {}
    for name, p in named_params:
        kind = SPARROW_KINDS['.'.join(name.split('.')[-3:])]
        groups.setdefault(kind, dict(params=[], kind=kind, s='rms'))['params'].append(p)
    return list(groups.values())

class DistributedSparrow(Sparrow):
    """
    sparrow_torch.Sparrow, spread over ranks as Muon is: each rank steps every world_size-th matrix, then the
    updates are summed across ranks in bfloat16 and every rank applies them. With momentum=0.95, nesterov=True,
    s='rms' (sqrt(max(out, in)) per matrix) and polar=zeropower_via_newtonschulz5 it takes Muon's step on every 'spectral' matrix
    (Sparrow's momentum is an average rather than a sum, which the direction does not see).
    """
    def __init__(self, params, rank=0, world_size=1, **kwargs):
        super().__init__(params, **kwargs)
        self.rank = rank
        self.world_size = world_size

    @torch.no_grad()
    def step(self):
        params = [p for group in self.param_groups for p in group['params']]
        old = [None] * len(params)
        for i, p in enumerate(params):
            if i % self.world_size == self.rank:
                old[i] = p.detach().clone()
            else:
                p.grad = None # Sparrow skips matrices without a gradient; zero_grad follows the step anyway
        super().step()
        updates_flat = torch.zeros(sum(p.numel() for p in params), device=params[0].device, dtype=torch.bfloat16)
        curr_idx = 0
        for p, p_old in zip(params, old):
            if p_old is not None:
                updates_flat[curr_idx:curr_idx+p.numel()] = (p_old - p).flatten()
                p.copy_(p_old)
            curr_idx += p.numel()
        all_reduce(updates_flat)
        curr_idx = 0
        for p in params:
            p.sub_(updates_flat[curr_idx:curr_idx+p.numel()].view_as(p).type_as(p))
            curr_idx += p.numel()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6 # head dim 128 suggested by @Grad62304977
    n_embd : int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data hyperparams
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin' # input .bin to train on
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin' # input .bin to eval validation loss on
    # optimization hyperparams
    batch_size : int = 8*64 # batch size, in sequences, across all devices
    device_batch_size : int = 64 # batch size, in sequences, per device
    sequence_length : int = 1024 # sequence length, in tokens
    num_iterations : int = 5100 # number of iterations to run
    learning_rate : float = 0.0036
    warmup_iters : int = 0
    warmdown_iters : int = 1450 # number of iterations of linear warmup/warmdown for triangular or trapezoidal schedule
    weight_decay : float = 0
    # evaluation and logging hyperparams
    val_loss_every : int = 125 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens : int = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    save_every : int = 0 # every how many steps to save the checkpoint? 0 for only at the end
args = Hyperparameters()

# [sparrow] runtime flags. Defaults reproduce the 2024-10-17 DistributedMuon record exactly;
# --num-iterations / --val-loss-every exist only for short smoke tests.
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/fineweb10B')
parser.add_argument('--num-iterations', type=int, default=args.num_iterations)
parser.add_argument('--val-loss-every', type=int, default=args.val_loss_every)
parser.add_argument('--wandb-project', default='halpern-tbptt')
parser.add_argument('--wandb-entity', default=None)
parser.add_argument('--run-name', default='muon-baseline')
parser.add_argument('--no-wandb', action='store_true')
# [sparrow] the optimizer of the hidden (block) matrices; the head stays on AdamW either way
parser.add_argument('--optimizer', default='muon', choices=['muon', 'sparrow', 'sparrow_rho', 'aurora'],
                    help='muon: the record. sparrow: DistributedSparrow, routed by SPARROW_KINDS. '
                         'sparrow_rho: sparrow with mlp.c_fc RowClip_r(Spectral_s(RowClip_c)), c set by --rho '
                         '(--rho sched: Sparrow before the warmdown, U_1 = RowClip_r Spectral_s at the end). '
                         'aurora: the record\'s Muon with the Aurora direction on the tall mlp.c_fc only')
parser.add_argument('--rho', default='sched',
                    help='sparrow_rho: a number in [0, 1] (c = rho / (1 - rho) RMS row norm; 0: Sparrow, 1: RowClip_r '
                         'Spectral_s) or sched: rho = u, the fraction of the warmdown done')
parser.add_argument('--seed', type=int, default=None, help='seed for the model init (the record is unseeded)')
parser.add_argument('--lr', type=float, default=0.1 * args.learning_rate, # [sparrow]
                    help='step size of the block matrices (Muon, Sparrow, Aurora); the record: 0.00036')
parser.add_argument('--head-lr', type=float, default=args.learning_rate,
                    help='AdamW step size of the head; the record: 0.0036')
# [sparrow] small single-device runs; the defaults are the record
parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'],
                    help='mps or cpu: one process without torchrun, NCCL or torch.compile')
parser.add_argument('--n-layer', type=int, default=12)
parser.add_argument('--n-head', type=int, default=6)
parser.add_argument('--n-embd', type=int, default=768)
parser.add_argument('--vocab-size', type=int, default=50304)
parser.add_argument('--batch-size', type=int, default=args.batch_size, help='sequences per step, over all devices')
parser.add_argument('--device-batch-size', type=int, default=args.device_batch_size)
parser.add_argument('--sequence-length', type=int, default=args.sequence_length)
parser.add_argument('--warmdown-iters', type=int, default=args.warmdown_iters)
parser.add_argument('--val-tokens', type=int, default=args.val_tokens)
parser.add_argument('--diag-every', type=int, default=0,
                    help='log the mlp.c_fc momentum diagnostics (momentum_diag) every N steps; 0: off')
cli = parser.parse_args()
if cli.seed is not None:
    torch.manual_seed(cli.seed)
args.learning_rate = cli.head_lr # [sparrow] logged as learning_rate
args.input_bin = os.path.join(cli.data_dir, 'fineweb_train_*.bin')
args.input_val_bin = os.path.join(cli.data_dir, 'fineweb_val_*.bin')
args.num_iterations = cli.num_iterations
args.val_loss_every = cli.val_loss_every
for k in ('batch_size', 'device_batch_size', 'sequence_length', 'warmdown_iters', 'val_tokens'):
    setattr(args, k, getattr(cli, k))

# set up DDP (distributed data parallel). torchrun sets this env variable
if cli.device == 'cuda':
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
else: # [sparrow]
    ddp_rank, ddp_local_rank, ddp_world_size, device = 0, 0, 1, cli.device
print(f"using device: {device}")

def sync(): # [sparrow] wait for the device, for timing
    if cli.device == 'cuda':
        torch.cuda.synchronize()
    elif cli.device == 'mps':
        torch.mps.synchronize()
master_process = (ddp_rank == 0) # this process will do logging, checkpointing etc.

# convenience variables
B, T = args.device_batch_size, args.sequence_length
# calculate the number of steps to take in the val loop.
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
# calculate the steps of gradient accumulation required to attain the desired global batch size.
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

# load tokens
train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

# there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. suggested to me by @Grad62304977.
# this originates from Karpathy's experiments.
num_vocab = cli.vocab_size # [sparrow] default 50304
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=cli.n_layer, n_head=cli.n_head, n_embd=cli.n_embd))
model = model.to(device)
if cli.device == 'cuda':
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    model = torch.compile(model)
    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model
else:
    raw_model = model
ctx = torch.amp.autocast(device_type=cli.device, dtype=torch.bfloat16)

# init the optimizer(s)
optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=cli.head_lr, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=cli.device == 'cuda')
named_block_params = list(raw_model.transformer.h.named_parameters())
if cli.optimizer == 'muon':
    optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=cli.lr, momentum=0.95,
                      rank=ddp_rank, world_size=ddp_world_size)
elif cli.optimizer == 'aurora':
    # [sparrow] the record's Muon; only the direction of the one tall matrix, mlp.c_fc (3072x768), changes
    is_fc = lambda name: name.endswith('mlp.c_fc.weight')
    optimizer2 = Muon([dict(params=[p for n, p in named_block_params if not is_fc(n)]),
                       dict(params=[p for n, p in named_block_params if is_fc(n)], dualizer='aurora')],
                      lr=cli.lr, momentum=0.95, rank=ddp_rank, world_size=ddp_world_size)
else:
    # [sparrow] Muon's step size, momentum, Nesterov and Newton-Schulz solver (the same compiled function)
    optimizer2 = DistributedSparrow(sparrow_param_groups(named_block_params), lr=cli.lr,
                                    momentum=0.95, nesterov=True, polar=zeropower_via_newtonschulz5, polar_steps=5,
                                    rank=ddp_rank, world_size=ddp_world_size)
optimizers = [optimizer1, optimizer2]

def weight_stats():
    # spectral norm and kappa = (max row norm) / (spectral norm) of the block matrices, by layer type
    stats = {}
    for name, p in named_block_params:
        kind = '.'.join(name.split('.')[1:3]) # e.g. attn.c_q, mlp.c_fc
        w = p.data.float()
        w = w if w.is_cuda else w.cpu() # [sparrow] no SVD on MPS
        s = torch.linalg.matrix_norm(w, ord=2).item()
        stats.setdefault(kind, []).append((s, w.norm(dim=1).max().item() / max(s, 1e-12)))
    out = {}
    for kind, vals in stats.items():
        out[f'weights/spectral_max/{kind}'] = max(v[0] for v in vals)
        out[f'weights/spectral_mean/{kind}'] = sum(v[0] for v in vals) / len(vals)
        out[f'weights/kappa_mean/{kind}'] = sum(v[1] for v in vals) / len(vals)
    return out

def momentum_diag():
    # [sparrow] on the momentum M of each mlp.c_fc this rank steps, with s = 1 and r = sqrt(in/out):
    #   clip_loss             1 - <M, RowClip_r(Polar(M))> / ||M||_*, the mass RowClip(Polar) loses to clipping
    #   sparrow_over_rowclip  <M, Sparrow(M)> / <M, RowClip_r(Polar(M))>; both lie in K_{r,1}, larger is the better step
    #   top10_rows            share of ||M||_F^2 in the 10% largest rows
    #   polar_maxrow_over_r   max row norm of Polar(M) over r
    acc = {}
    for name, p in named_block_params:
        st = optimizer2.state.get(p, {})
        M = st.get('momentum_buffer', st.get('momentum'))
        if not name.endswith('mlp.c_fc.weight') or M is None:
            continue
        M = M.detach().cpu().double() # [sparrow] float64 on the host (MPS has none)
        m, d = M.shape
        r = (d / m)**0.5
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        P = U @ Vh
        W1 = sparrow_torch.row_clip(P, r)
        W3 = sparrow_torch.sparrow(M, 'gate_value', 1.0, 'svd').double()
        rows2 = M.pow(2).sum(dim=1)
        vals = dict(clip_loss=1 - (M * W1).sum() / S.sum(),
                    sparrow_over_rowclip=(M * W3).sum() / (M * W1).sum(),
                    top10_rows=rows2.sort(descending=True).values[:m // 10].sum() / rows2.sum(),
                    polar_maxrow_over_r=P.norm(dim=1).max() / r)
        for k, v in vals.items():
            acc.setdefault(k, []).append(float(v))
    return {f'diag/{k}': sum(v) / len(v) for k, v in acc.items()}
# learning rate decay scheduler (linear warmup and warmdown)
def get_lr(it):
    assert it <= args.num_iterations
    # 1) linear warmup for warmup_iters steps
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    # 2) constant lr for a while
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    # 3) linear warmdown
    else:
        decay_ratio = (args.num_iterations - it) / args.warmdown_iters
        return decay_ratio
schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

def rho_schedule(it):
    # [sparrow] --optimizer sparrow_rho: the clip level rho of mlp.c_fc (sparrow_torch.rowclip_c), --rho or for
    # --rho sched the decay u = 1 - get_lr(it) of the step size (0 before the warmdown, 1 at its end), so 0 (Sparrow)
    # at the peak and 1 (RowClip_r Spectral_s) at the end
    if cli.rho != 'sched':
        return float(cli.rho)
    return 0.0 if it < args.warmup_iters else min(1.0, 1.0 - get_lr(it))

# begin logging
if master_process:
    run_id = str(uuid.uuid4())
    logdir = 'logs/%s/' % run_id
    os.makedirs(logdir, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    # create the log file
    with open(logfile, "w") as f:
        # begin the log by printing this file (the Python code)
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        # log information about the hardware/software environment this is running on
        # and print the full `nvidia-smi` to file
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            f.write(f'{result.stdout}\n')
        except FileNotFoundError: # [sparrow] local device
            f.write(f'(no nvidia-smi; device {device})\n')
        f.write('='*100 + '\n')

# [sparrow] W&B logging (rank 0 only)
use_wandb = master_process and not cli.no_wandb
if use_wandb:
    import wandb
    wandb.init(
        project=cli.wandb_project,
        entity=cli.wandb_entity,
        name=cli.run_name,
        save_code=True,
        config=dict(
            **asdict(args),
            record='modded-nanogpt/records/track_1_short/2024-10-17_DistributedMuon',
            optimizer=cli.optimizer,
            head_optimizer='adamw',
            muon_lr=cli.lr,
            seed=cli.seed,
            muon_momentum=0.95,
            muon_nesterov=True,
            muon_backend='newtonschulz5',
            muon_backend_steps=5,
            **({'sparrow_kinds': SPARROW_KINDS} if cli.optimizer.startswith('sparrow') else {}),
            **({'rho': cli.rho} if cli.optimizer == 'sparrow_rho' else {}),
            adam_betas=(0.9, 0.95),
            n_layer=cli.n_layer, n_head=cli.n_head, n_embd=cli.n_embd, vocab_size=num_vocab, device=cli.device,
            world_size=ddp_world_size,
            train_accumulation_steps=train_accumulation_steps,
            val_steps=val_steps,
            local_run_id=run_id,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),
            torch_version=torch.__version__,
            gpu=torch.cuda.get_device_name() if cli.device == 'cuda' else cli.device,
        ),
    )
    wandb.define_metric('step')
    wandb.define_metric('*', step_metric='step')
tokens_per_step = args.batch_size * args.sequence_length

training_time_ms = 0
# start the clock
sync()
t0 = time.time()
# begin training
train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # once in a while evaluate the validation dataset
    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        # stop the clock
        sync()
        training_time_ms += 1000 * (time.time() - t0)
        # run validation batches
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx: # of course, we'd like to use no_grad() here too, but that creates a torch.compile error for some reason
                _, loss = model(x_val, y_val, return_logits=False)
                val_loss += loss.detach()
                del loss
        all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= val_steps
        # log val loss to console and to logfile
        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')
            if use_wandb:
                wandb.log({'step': step, 'val/loss': val_loss.item(), 'tokens': step * tokens_per_step,
                           'time/train_time_ms': training_time_ms, **weight_stats()})
        # start the clock again
        sync()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        # stop the clock
        sync()
        training_time_ms += 1000 * (time.time() - t0)
        # save the state of the training process
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        # start the clock again
        sync()
        t0 = time.time()

    # bit confusing: we want to make sure to eval on 0th iteration
    # but also after the very last iteration. so we loop for step <= num_iterations
    # instead of just < num_iterations (one extra due to <=), only to do
    # the validation/sampling one last time, and then we break right here as we're done.
    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    for i in range(1, train_accumulation_steps+1):
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
            train_loss = loss.detach()
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        if i < train_accumulation_steps:
            with (model.no_sync() if isinstance(model, DDP) else contextlib.nullcontext()): # there's no need to sync gradients every accumulation step
                loss.backward()
        else:
            loss.backward() # just sync on the last step
    for p in model.parameters():
        p.grad /= train_accumulation_steps
    if cli.optimizer == 'sparrow_rho': # [sparrow] the clip level of this step, as the schedulers set lr
        for group in optimizer2.param_groups:
            if group['kind'] == 'gate_value':
                group['rho'] = rho_schedule(step)
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------
    # everything that follows now is just diagnostics, prints, logging, etc.

    #dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # all-reducing the training loss would be more correct in terms of logging, but slower
    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")
        if use_wandb:
            wandb.log({'step': step + 1, 'train/loss': train_loss.item(), 'tokens': (step + 1) * tokens_per_step,
                       'lr/adam_head': optimizer1.param_groups[0]['lr'], 'lr/muon': optimizer2.param_groups[0]['lr'],
                       **({'rho': rho_schedule(step)} if cli.optimizer == 'sparrow_rho' else {}),
                       'time/train_time_ms': approx_time, 'time/step_avg_ms': approx_time / timed_steps})
        # [sparrow] stop paying for a diverged run; torchrun then tears down the other ranks
        if not torch.isfinite(train_loss):
            if use_wandb:
                wandb.finish(exit_code=1)
            raise RuntimeError(f'non-finite train loss at step {step+1}')
        if cli.diag_every and (step + 1) % cli.diag_every == 0: # [sparrow]
            diag = momentum_diag()
            if cli.optimizer == 'sparrow_rho':
                diag['rho'] = rho_schedule(step) # the clip level this step used
            line = f'step:{step+1}/{args.num_iterations} diag {json.dumps({k: round(v, 5) for k, v in diag.items()})}'
            print(line)
            with open(logfile, "a") as f:
                f.write(line + '\n')
            if use_wandb:
                wandb.log({'step': step + 1, **diag})

if master_process:
    peak_mib = torch.cuda.max_memory_allocated() // 1024 // 1024 if cli.device == 'cuda' else -1
    print(f"peak memory consumption: {peak_mib} MiB")
    if use_wandb:
        wandb.run.summary['final_val_loss'] = val_loss.item()
        wandb.run.summary['peak_memory_mib'] = peak_mib
        wandb.finish()

# -------------------------------------------------------------------------
# clean up nice
if dist.is_initialized():
    dist.destroy_process_group()
