#!/bin/bash
# Gap-closing batches, phase 2 (revision, 2026-11-02): dynamic order arrival, obstacles, belief drift. Originally run from a frozen copy of src (identical to this release for the options used); starts when phase 1 has finished.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
O=experiments/peer_model/gaps; L=experiments/peer_model/gaps
while [ ! -f $L/phase1.done ]; do sleep 30; done
CFG="A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16"; REP="490 491 492"
RUN="$V -m src.run_batch --jobs 12 --mem-gb 30"
S=sppdec_consensus_rule_nogamble_limited; C=cbba_single_task_cbba_single; H=centralized_cc_hungarian_hungarian; M=marl_ippo_limited
CB=cbba_full_cbba_bundle; CK=cbba_full_capability_bundle; A=decentralized_auction_auction_baseline; N=naive_decentralized_reactive
ALL="$S $H $C $CB $A $N $CK $M"
step() { echo "[GAPS2] $1 start $(date +%T)"; }
# 6. Dynamic arrival: the last half of the orders are released over the first 600 steps (shared order feed); the reference is the static run of the unequal_batteries batch
[ -f $L/dynamic.done ] || { step dynamic; ORDER_ARRIVAL=0.5,600 RADIO_PROFILE=ideal $RUN --configs $CFG --only-replicates $REP --arms $ALL --out $O/dynamic > $L/dynamic.log 2>&1; echo finished > $L/dynamic.done; }
# 7. Obstacles: 8 blocks of 3 x 3 cells (18% of the map) and 14 blocks (32%)
for OB in 8,3,3 14,3,3; do
  T=obstacles_$(echo $OB | tr , _)
  [ -f $L/$T.done ] || { step $T; OBSTACLES=$OB RADIO_PROFILE=ideal $RUN --configs $CFG --only-replicates $REP --arms $ALL --out $O/$T > $L/$T.log 2>&1; echo finished > $L/$T.done; }
done
# 8. Peer-energy belief under sensing stress (SPPDEC only; the belief is logged at every decision)
[ -f $L/belief_drift.done ] || { step belief_drift; ARMS="$S"; for t in n05 n1 n2 l50 d120 bo; do ARMS="$ARMS ${S}_$t"; done
  RADIO_PROFILE=ideal $RUN --configs $CFG --only-replicates $REP --arms $ARMS --out $O/belief_drift > $L/belief_drift.log 2>&1; echo finished > $L/belief_drift.done; }
echo "[GAPS2] phase 2 finished $(date +%T)" > $L/phase2.done
