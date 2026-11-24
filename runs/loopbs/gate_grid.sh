#!/bin/bash
# (The reported runs also evaluated L1x2, L2x2 and L1x4 at other iteration counts, an option since removed.)
# Tall gated-attention model of the gated-attention experiment (Tiny Shakespeare, one epoch, --attn_gate elem, 4 heads of width 128)
# at 4 layers' compute: plain4 | L1x2 (prelude 1 + core 1 x 2 + coda 1) | L2x2 (2 layers x 2) | L1x4 (1 layer x 4), fixed-step
# Halpern beta 0.5. Muon (--method muon) vs Sparrow on every matrix (--method sparrow), steps 0.01 / 0.02, seeds 0-2.
cd "$(dirname "$0")/../.."
job() { opt=$1; arch=$2; lr=$3; seed=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$seed; mkdir -p $out; [ -f $out/done ] && return
  case $arch in plain4) A="--n_layer 4";; L1x2) A="--n_layer 3 --loop 2";;
    L2x2) A="--n_layer 2 --loop 2 --n_prelude 0 --n_coda 0";;
    L1x4) A="--n_layer 1 --loop 4 --n_prelude 0 --n_coda 0";; esac
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --attn_gate elem --head_dim 128 --method $([ $opt = spectral ] && echo muon || echo $opt) --seed $seed --beta 0.5 $A > $out/stdout.log 2>&1 && touch $out/done
  echo "$opt $arch $lr s$seed exit $? $(date +%H:%M)"; }
export -f job
for seed in 0 1 2; do for arch in plain4 L1x2 L2x2 L1x4; do for opt in spectral sparrow; do for lr in 0.01 0.02; do
  echo "$opt $arch $lr $seed"; done; done; done; done | xargs -P 4 -L 1 bash -c 'job "$@"' _
