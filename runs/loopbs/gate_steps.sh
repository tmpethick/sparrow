#!/bin/bash
# Bracket the LMO step for the 1 + 1 x N + 1 cells (gate_grid.sh, gate_pcN.sh), which only tried 0.01 / 0.02:
# Muon (best at 0.01 everywhere) at 0.005; Sparrow at 0.005 where it was best at 0.01 (plain4, PC6H4) and at 0.04
# where it was best at 0.02 (plain3, L1x2, PC4, PC6, PC6H2). Seeds 0-2, same flags as before.
cd "$(dirname "$0")/../.."
job() { opt=$1; arch=$2; lr=$3; seed=$4; out=runs/loopbs/gate-tall-$opt-$arch-lr$lr-s$seed; mkdir -p $out; [ -f $out/done ] && return
  case $arch in plain3) A="--n_layer 3";; plain4) A="--n_layer 4";; L1x2) A="--n_layer 3 --loop 2";;
    PC4) A="--n_layer 3 --loop 4";; PC6) A="--n_layer 3 --loop 6";;
    PC6H2) A="--n_layer 3 --loop 6 --tbptt 2";; PC6H4) A="--n_layer 3 --loop 6 --tbptt 4";; esac
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --attn_gate elem --head_dim 128 --method $([ $opt = spectral ] && echo muon || echo $opt) --seed $seed --beta 0.5 $A > $out/stdout.log 2>&1 && touch $out/done
  echo "$opt $arch $lr s$seed exit $? $(date +%H:%M)"; }
export -f job
for seed in 0 1 2; do
  for arch in plain3 plain4 L1x2 PC4 PC6 PC6H2 PC6H4; do echo "spectral $arch 0.005 $seed"; done
  for arch in plain4 PC6H4; do echo "sparrow $arch 0.005 $seed"; done
  for arch in plain3 L1x2 PC4 PC6 PC6H2; do echo "sparrow $arch 0.04 $seed"; done
done | xargs -P 4 -L 1 bash -c 'job "$@"' _
