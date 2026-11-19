#!/bin/bash
# MARL-only rerun after the idle fix (src/strategies/marl.py: nothing affordable -> idle, so the controller parks the drone, as for every other strategy). Same missions and seeds as every earlier batch; the other strategies are unchanged and are not rerun.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
ALL=($(cat scripts/paper_runs/grid64.txt))
P=experiments/peer_model
$V -m src.run_batch --configs "${ALL[@]}" --arms marl_ippo_limited --replicates 400 --only-replicates 300 301 302 303 304 --out $P/marlfix_grid --jobs 16 --mem-gb 30 > $P/marlfix_grid.log 2>&1
echo grid-done > $P/marlfix_grid.done
$V -m src.run_batch --configs A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16 --only-replicates 470 471 472 --arms marl_ippo_limited_hs marl_ippo_limited_hs3 --out $P/marlfix_speed --jobs 6 > $P/marlfix_speed.log 2>&1
$V -m src.run_batch --configs A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16 --only-replicates 490 491 492 --arms marl_ippo_limited marl_ippo_limited_n2 marl_ippo_limited_l50 marl_ippo_limited_n2l50 --out $P/marlfix_sensing --jobs 6 > $P/marlfix_sensing.log 2>&1
mkdir -p experiments/compute
export DRONE_CPU_LOG=$PWD/experiments/compute/think_cpu_marlfix.jsonl; rm -f $DRONE_CPU_LOG
$V -m src.run_batch --configs A4_P24 A8_P24 A12_P24 A10_P12 A15_P12 A20_P24 --only-replicates 480 481 --arms marl_ippo_limited --out $P/marlfix_energy --jobs 3 > $P/marlfix_energy.log 2>&1
unset DRONE_CPU_LOG
$V -m src.run_batch --configs A10_P24 A20_P24 --only-replicates 450 --arms marl_ippo_limited --out $P/marlfix_large --jobs 2 > $P/marlfix_large.log 2>&1
echo all-done > $P/marlfix_all.done
