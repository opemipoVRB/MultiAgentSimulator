#!/bin/bash
# Radio stress (checklist 2): the strategies that pass messages, under the ideal radio and under lossy50 (50% loss, 1-3 steps delay), plus SPPDEC and MARL (no messages) as controls. 7 configs x replicates 490-492.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
ARMS="cbba_single_task_cbba_single cbba_full_cbba_bundle cbba_full_capability_bundle decentralized_auction_auction_baseline centralized_cc_hungarian_hungarian"
CFG="A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16"
RADIO_PROFILE=ideal $V -m src.run_batch --configs $CFG --only-replicates 490 491 492 --arms $ARMS --out experiments/peer_model/radio_stress/ideal --jobs 4 > experiments/peer_model/radio_stress_ideal.log 2>&1
RADIO_PROFILE=lossy50 $V -m src.run_batch --configs $CFG --only-replicates 490 491 492 --arms $ARMS sppdec_consensus_rule_nogamble_limited marl_ippo_limited --out experiments/peer_model/radio_stress/lossy50 --jobs 4 > experiments/peer_model/radio_stress_lossy50.log 2>&1
echo finished > experiments/peer_model/radio_stress.done
