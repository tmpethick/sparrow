#!/bin/bash
# Muon, Sparrow, Aurora at eta = 0.02 on Tiny Shakespeare (1 epoch, the training-speed setup), polar factor by the
# exact SVD or PolarExpress with k steps. Run from sparrow/: bash runs/polar-approx/sweep.sh
run() { m=$1 p=$2 s=$3; out=runs/polar-approx/$m-$p-s$s; [ -f $out/done ] && return; mkdir -p $out
  case $p in svd) P="--polar svd";; pe*) P="--polar polarexpress --polar_steps ${p#pe}";; esac
  python runs/polar-approx/run_polar.py $P --out $out --method $m --lmo_lr 0.02 \
    --update unconstrained --sampler epoch --epochs 1 --eval_every 100 --seed $s \
    > $out/stdout.log 2>&1 && touch $out/done; }
export -f run
for s in 0 1 2; do for p in svd pe2 pe3 pe4 pe5 pe6 pe8; do for m in sparrow aurora muon; do
  echo "$m $p $s"; done; done; done | xargs -P 5 -L 1 bash -c 'run $0 $1 $2'
