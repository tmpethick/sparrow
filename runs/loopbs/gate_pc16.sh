#!/bin/bash
# Pilot: does truncated backprop hurt less with more iterations? 1 + 1 x 16 + 1 (3 distinct layers, 18 layer
# applications), gradients through the last H = 2 / 4 / 8 / 16 (full) iterations. Seed 0, each optimizer at its best
# step at N = 6 (Muon 0.01, Sparrow 0.02); same flags as gate_pcN.sh.
cd "$(dirname "$0")/../.."
job() { opt=$1; arch=$2; lr=$3; seed=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$seed; mkdir -p $out; [ -f $out/done ] && return
  case $arch in PC16) A="--n_layer 3 --loop 16";; PC16H*) A="--n_layer 3 --loop 16 --tbptt ${arch#PC16H}";; esac
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --attn_gate elem --head_dim 128 --method $([ $opt = spectral ] && echo muon || echo $opt) --seed $seed --beta 0.5 $A > $out/stdout.log 2>&1 && touch $out/done
  echo "$opt $arch $lr s$seed exit $? $(date +%H:%M)"; }
export -f job
for arch in PC16 PC16H8 PC16H4 PC16H2; do echo "spectral $arch 0.01 0"; echo "sparrow $arch 0.02 0"; done |
  xargs -P 4 -L 1 bash -c 'job "$@"' _
