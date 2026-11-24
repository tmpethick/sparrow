#!/bin/bash
# The U_1 versus U_3 teacher-student grid with the joint-set direction sparrow_rho --rho sched (rho = u, the cosine step-size decay: Sparrow at the
# start, U_1 at the end) instead of Sparrow's U_3: same teachers, step sizes, budgets and seeds as runs/tsr-sp-*.
cd "$(dirname "$0")/.."
R() { .venv/bin/python experiments/synthetic.py --families ours_rowspec --oracle sparrow_rho --rho sched \
        --etas 0.001,0.003,0.01,0.03,0.1 "$@" > /dev/null 2>&1; echo "done $* exit $?"; }
R --out runs/tsr-sched-g0 --teachers gauss_4 --seed 0 --betas 0.5,1,2,4,16,64 &
for s in 1 2; do R --out runs/tsr-sched-g$s --teachers gauss_4 --seed $s --betas 0.5,1,2,4,16 & done
R --out runs/tsr-sched-f0 --teachers few_8 --seed 0 --betas 0.5,1,2,4,16,64 &
for s in 1 2; do R --out runs/tsr-sched-f$s --teachers few_8 --seed $s --betas 1,4,16,64 & done
wait
