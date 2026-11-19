#!/bin/bash
# Gap-closing batch, phase 3 (revision, 2026-11-02): the language-model ablation re-run on the current simulator and the same missions: rule list against the free language model at three sizes
# (no decision time charged), with the arm of record as reference. Starts when phase 2 has finished.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
V=${PYTHON:-python3}
L=experiments/peer_model/gaps
while [ ! -f $L/phase2.done ]; do sleep 30; done
echo "[GAPS3] slm start $(date +%T)"
RADIO_PROFILE=ideal $V -m src.run_batch --jobs 3 --mem-gb 30 --configs A4_P12 A8_P24 A12_P24 --only-replicates 490 491 492 \
  --arms sppdec_consensus_rule_nogamble_limited sppdec_ladder_limited_rules sppdec_ladder_limited_slm_vl8b sppdec_ladder_limited_slm_q8b sppdec_ladder_limited_slm_q1_7b \
  --out $L/slm_rerun > $L/slm_rerun.log 2>&1
echo finished > $L/slm_rerun.done
echo "[GAPS3] finished $(date +%T)" > $L/phase3.done
