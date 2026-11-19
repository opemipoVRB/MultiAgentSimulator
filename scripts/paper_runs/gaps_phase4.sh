#!/bin/bash
# Retransmission batches re-run (2026-11-02): the first run applied the repeat only to point-to-point messages, not to broadcasts (CBBA, auction); fixed in src/resources.py.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
V=${PYTHON:-python3}
L=experiments/peer_model/gaps
while [ ! -f $L/phase2.done ]; do sleep 30; done
CFG="A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16"
for P in lossy50_retx3 lossy80_retx5; do
  RADIO_PROFILE=$P $V -m src.run_batch --jobs 12 --mem-gb 30 --configs $CFG --only-replicates 490 491 492 --arms cbba_single_task_cbba_single cbba_full_cbba_bundle cbba_full_capability_bundle decentralized_auction_auction_baseline centralized_cc_hungarian_hungarian --out $L/radio_$P > $L/radio_$P.log 2>&1
  echo finished > $L/radio_$P.done
done
echo done > $L/phase4.done
