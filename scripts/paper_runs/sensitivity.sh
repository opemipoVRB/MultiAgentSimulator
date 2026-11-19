#!/bin/bash
# Sensitivity study of the arm of record: 11 configs (7 development + 4 heavy-load) x replicates 500-502, reference plus beta 0 / 0.1, w-bar 0 / 0.5, belief window 10 / 60, belief off.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
V=${PYTHON:-python3}
B=sppdec_consensus_rule_nogamble_limited
$V -m src.run_batch --configs A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16 A2_P30 A2_P36 A3_P36 A4_P36 --only-replicates 500 501 502 --arms $B ${B}_b0 ${B}_b10 ${B}_w0 ${B}_w50 ${B}_h10 ${B}_h60 ${B}_boff --out experiments/peer_model/sensitivity --jobs 4 > experiments/peer_model/sensitivity.log 2>&1
echo finished > experiments/peer_model/sensitivity.done
