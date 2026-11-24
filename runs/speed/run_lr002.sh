#!/bin/bash
# Sparrow and Aurora (Algorithm 3) at step 0.02 in the speed setup, seeds 0-14: does Sparrow's lead over Aurora
# depend on the step size? (Sparrow's best step in the lr sweep is 0.02; Muon's is 0.01.)
cd "$(dirname "$0")/../.."
job() { o=$1 s=$2; sfx=$([ $s = 0 ] || echo -s$s); out=runs/speed/$o-lr0.02$sfx; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method $o --lmo_lr 0.02 --seed $s > $out/stdout.log 2>&1 && touch $out/done
  echo "$o s$s exit $? $(date +%H:%M)"; }
export -f job
for s in $(seq 0 14); do echo "sparrow $s"; echo "aurora $s"; done | xargs -P 5 -L 1 bash -c 'job "$@"' _
