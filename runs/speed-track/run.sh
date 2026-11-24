#!/bin/bash
# Exact joint LMO (warm-started ADMM, --admm_iters per step) on the SwiGLU gate/value in the speed setup, and
# Sparrow / Aurora (Alg. 3) / Muon runs, all with --lmo_track 1: at every eval (100 steps) Sparrow's and Aurora's
# directions on the current momentum are compared with the joint LMO (cosine, ||D||_F, max row / r, value).
cd "$(dirname "$0")/../.."
job() { o=$1 lr=$2 s=$3; out=runs/speed-track/$o-lr$lr-s$s; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method $([ $o = spectral ] && echo muon || echo $o) --lmo_lr $lr --seed $s --lmo_track 1 > $out/stdout.log 2>&1 && touch $out/done
  echo "$o $lr s$s exit $? $(date +%H:%M)"; }
export -f job
printf "%s\n" "${@:-exact 0.01 0}" | xargs -P 4 -L 1 bash -c 'job "$@"' _
