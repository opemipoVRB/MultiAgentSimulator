#!/bin/bash
# Fresh-seed full-grid comparison, rerun in the only mode of the system (one process per drone, each on its own core), all six strategies plus the capability-sized CBBA bundle and the trained MARL baseline, replicates 300-304 (the same missions as the submitted-model grid) under the realistic energy model. Jobs are admitted only while the drones of all running jobs fit the physical cores. Runs the configs in chunks of 8; a chunk whose five replicates are finished is skipped, so the script can be started again after an interruption.
# Runs the configs in chunks of 8; a chunk whose five replicates are all finished is skipped, so the script can be started again after an interruption.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
V=${PYTHON:-python3}
OUT=experiments/peer_model/fresh_grid_realistic
mkdir -p $OUT/logs
ALL=($(cat scripts/paper_runs/grid64.txt))
export MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
ARMS="sppdec_consensus_rule_nogamble_limited centralized_cc_hungarian_hungarian cbba_single_task_cbba_single cbba_full_cbba_bundle decentralized_auction_auction_baseline naive_decentralized_reactive cbba_full_capability_bundle marl_ippo_limited"
for ((i=0; i<${#ALL[@]}; i+=8)); do
  CH=("${ALL[@]:i:8}"); PEND=()
  for c in "${CH[@]}"; do
    n=0; for r in 300 301 302 303 304; do grep -qs "finished in" $OUT/logs/${c}_r$r.log && n=$((n+1)); done
    [ $n -lt 5 ] && PEND+=($c)
  done
  [ ${#PEND[@]} -eq 0 ] && continue
  echo "[GRID] chunk $((i/8+1)): ${PEND[*]}"
  $V -m src.run_batch --configs "${PEND[@]}" --arms $ARMS --replicates 400 --only-replicates 300 301 302 303 304 --out $OUT --jobs 24 --mem-gb 30 >> $OUT.log 2>&1
done
echo "[GRID] done"
