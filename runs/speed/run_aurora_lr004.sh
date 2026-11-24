#!/bin/bash
# Aurora (Algorithm 3) at step 0.04, seeds 0-4: brackets Aurora's step size from above (0.02 beat 0.01).
cd "$(dirname "$0")/../.."
job() { s=$1; sfx=$([ $s = 0 ] || echo -s$s); out=runs/speed/aurora-lr0.04$sfx; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method aurora --lmo_lr 0.04 --seed $s > $out/stdout.log 2>&1 && touch $out/done
  echo "s$s exit $? $(date +%H:%M)"; }
export -f job
seq 0 4 | xargs -P 5 -L 1 bash -c 'job "$@"' _
