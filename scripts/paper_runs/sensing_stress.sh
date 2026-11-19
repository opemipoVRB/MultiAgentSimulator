#!/bin/bash
# Sensing stress test: SPPDEC, CBBA single, Centralised and MARL under position noise (n2), lost readings (l50) and both (n2l50), plus the reference, on 7 development configs x fresh replicates 490-492, realistic energy model.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
ARMS=""
for b in sppdec_consensus_rule_nogamble_limited cbba_single_task_cbba_single marl_ippo_limited; do ARMS="$ARMS $b ${b}_n2 ${b}_l50 ${b}_n2l50"; done
ARMS="$ARMS centralized_cc_hungarian_hungarian centralized_cc_hungarian_n2_hungarian centralized_cc_hungarian_l50_hungarian centralized_cc_hungarian_n2l50_hungarian"
$V -m src.run_batch --configs A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16 --only-replicates 490 491 492 --arms $ARMS --out experiments/peer_model/sensing_stress --jobs 4 > experiments/peer_model/sensing_stress.log 2>&1
echo finished > experiments/peer_model/sensing_stress.done
