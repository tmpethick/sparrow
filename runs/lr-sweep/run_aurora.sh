#!/bin/bash
# Aurora (Algorithm 3) in the step-size sweep: eta 0.0025 and 0.005 here; 0.01, 0.02 and 0.04 are copies of the
# identical runs/speed/aurora-lr* runs (same command and seed).
cd "$(dirname "$0")/../.."
job() { lr=$1 s=$2; out=runs/lr-sweep/aurora-lr$lr-s$s; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --lmo_lr $lr --seed $s --method aurora > $out/stdout.log 2>&1 && touch $out/done
  echo "aurora $lr s$s exit $? $(date +%H:%M)"; }
export -f job
for lr in 0.005 0.0025; do for s in 0 1 2; do echo "$lr $s"; done; done | xargs -P 6 -L 1 bash -c 'job "$@"' _
