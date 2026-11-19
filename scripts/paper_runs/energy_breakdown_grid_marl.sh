#!/bin/bash
# MARL on the same missions as energy_breakdown_grid.sh (6 configs x replicates 480-481), same CPU log (appended), combined-change checkpoint.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
mkdir -p experiments/compute
export DRONE_CPU_LOG=$PWD/experiments/compute/think_cpu_grid.jsonl
V=${PYTHON:-python3}
$V -m src.run_batch --configs A4_P24 A8_P24 A12_P24 A10_P12 A15_P12 A20_P24 --only-replicates 480 481 --arms marl_ippo_limited --out experiments/peer_model/energy_breakdown_grid_marl --jobs 2 > experiments/peer_model/energy_breakdown_grid_marl.log 2>&1
echo finished > experiments/peer_model/energy_breakdown_grid_marl.done
