#!/bin/bash
# 1 + 1 x N + 1 on the tall gated-attention model of gate_grid.sh: prelude, one core layer iterated N times (fixed-step
# Halpern, beta 0.5), coda; always 3 distinct layers. plain3 (no loop) is the parameter-matched baseline; N = 2 is
# gate_grid.sh's L1x2. Then N = 6 with truncated backprop through the last H = 2 / 4 iterations.
# Muon (--method muon) vs Sparrow (--method sparrow), steps 0.01 / 0.02, seeds 0-2.
cd "$(dirname "$0")/../.."
job() { opt=$1; arch=$2; lr=$3; seed=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$seed; mkdir -p $out; [ -f $out/done ] && return
  case $arch in plain3) A="--n_layer 3";; PC4) A="--n_layer 3 --loop 4";; PC6) A="--n_layer 3 --loop 6";;
    PC6H2) A="--n_layer 3 --loop 6 --tbptt 2";; PC6H4) A="--n_layer 3 --loop 6 --tbptt 4";; esac
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --attn_gate elem --head_dim 128 --method $([ $opt = spectral ] && echo muon || echo $opt) --seed $seed --beta 0.5 $A > $out/stdout.log 2>&1 && touch $out/done
  echo "$opt $arch $lr s$seed exit $? $(date +%H:%M)"; }
export -f job
for arch in plain3 PC4 PC6 PC6H2 PC6H4; do for seed in 0 1 2; do for opt in spectral sparrow; do for lr in 0.01 0.02; do
  echo "$opt $arch $lr $seed"; done; done; done; done | xargs -P 4 -L 1 bash -c 'job "$@"' _
