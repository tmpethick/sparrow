#!/bin/bash
# sparrow_rho --rho sched (RowClip_r Spectral_s RowClip_c, c = rho / (1 - rho) RMS row norm, rho = u with u the decay of the step size) on the SwiGLU gate and value in the
# speed setup, step 0.02, seeds 0-2: Sparrow until the decay, U_1 at the end.
cd "$(dirname "$0")/../.."
job() { rho=$1 s=$2 lr=0.02; sfx=$([ $s = 0 ] || echo -s$s); out=runs/speed/sparrow_sched$rho-lr$lr$sfx; mkdir -p $out; [ -f $out/done ] && return
  .venv/bin/python experiments/nanogpt.py --out $out --update unconstrained --sampler epoch --epochs 1 --eval_every 100 \
    --method sparrow_rho --rho sched --lmo_lr $lr --seed $s > $out/stdout.log 2>&1 && touch $out/done
  echo "sparrow_rho sched s$s exit $? $(date +%H:%M)"; }
export -f job
for s in 0 1 2; do echo "0.5 $s"; done | xargs -P 3 -L 1 bash -c 'job "$@"' _
