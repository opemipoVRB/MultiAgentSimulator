#!/bin/bash
# Energy by source (motion, hover, radio) and CPU time of every strategy, including the centralised planner and the capability-sized CBBA bundle: 6 configs across the load regimes (A4_P24 low, A8_P24 and A12_P24 middle, A10_P12, A15_P12 and A20_P24 high) x fresh replicates 480-481, realistic energy model.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
mkdir -p experiments/compute
export DRONE_CPU_LOG=$PWD/experiments/compute/think_cpu_grid.jsonl
V=${PYTHON:-python3}
rm -f $DRONE_CPU_LOG
$V -m src.run_batch --configs A4_P24 A8_P24 A12_P24 A10_P12 A15_P12 A20_P24 --only-replicates 480 481 --arms sppdec_consensus_rule_nogamble_limited cbba_single_task_cbba_single cbba_full_cbba_bundle cbba_full_capability_bundle decentralized_auction_auction_baseline naive_decentralized_reactive centralized_cc_hungarian_hungarian --out experiments/peer_model/energy_breakdown_grid --jobs 3 > experiments/peer_model/energy_breakdown_grid.log 2>&1
echo finished > experiments/peer_model/energy_breakdown_grid.done
