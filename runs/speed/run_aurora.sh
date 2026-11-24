#!/bin/bash
# Aurora's Algorithm 3 (Dewulf et al. 2026, K = 2, beta = 1/2) on the SwiGLU gate and value, in the speed setup of
# REPRODUCE.md (Muon's step 0.01, seeds 0-14), next to the runs of Muon, our Aurora variant and Sparrow.
cd "$(dirname "$0")/../.."
job() { s=$1; sfx=$([ $s = 0 ] || echo -s$s); out=runs/speed/aurora-lr0.01$sfx; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method aurora --lmo_lr 0.01 --seed $s > $out/stdout.log 2>&1 && touch $out/done
  echo "s$s exit $? $(date +%H:%M)"; }
export -f job
seq 0 14 | xargs -P 5 -L 1 bash -c 'job "$@"' _
