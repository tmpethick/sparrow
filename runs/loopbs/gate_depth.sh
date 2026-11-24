#!/bin/bash
# 1 + 1 x N + 1 on the tall gated model: (a) full backprop at N = 1, 2, 4, 8, 16; (b) N = 16 with gradients through
# the last H = 4 / 8 iterations. Muon (--method muon) and Sparrow at steps 0.01 / 0.02 (the optima at every N <= 6,
# where 0.005 and 0.04 were worse), seeds 0-2; plus 0.005 at N = 16 full on seed 0 as a bracket check.
# N = 2 and 4 are gate_grid.sh's L1x2 and gate_pcN.sh's PC4 (already run). Arch PC<N>[H<H>] = --loop N [--tbptt H].
cd "$(dirname "$0")/../.."
job() { opt=$1; arch=$2; lr=$3; seed=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$seed; mkdir -p $out; [ -f $out/done ] && return
  N=${arch#PC}; H=0; [[ $N == *H* ]] && { H=${N#*H}; N=${N%H*}; }
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --attn_gate elem --head_dim 128 --method $([ $opt = spectral ] && echo muon || echo $opt) --seed $seed --beta 0.5 --n_layer 3 --loop $N --tbptt $H \
    > $out/stdout.log 2>&1 && touch $out/done
  echo "$opt $arch $lr s$seed exit $? $(date +%H:%M)"; }
export -f job
{ for arch in PC16 PC16H8 PC16H4 PC8 PC1; do for seed in 0 1 2; do for opt in spectral sparrow; do for lr in 0.01 0.02; do
    echo "$opt $arch $lr $seed"; done; done; done; done
  echo "spectral PC16 0.005 0"; echo "sparrow PC16 0.005 0"; } | xargs -P 4 -L 1 bash -c 'job "$@"' _
