#!/bin/bash
# Gap-closing batches, phase 1 (revision, 2026-11-02). Originally run from a frozen copy of src; that copy is identical to src in this release for every option these batches use.
# Seeds: replicates 490-492 (the stress-test seeds); 7 configs unless stated. One batch at a time (one process per drone, each on its own core).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
O=experiments/peer_model/gaps; L=experiments/peer_model/gaps
CFG="A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16"
REP="490 491 492"
RUN="$V -m src.run_batch --jobs 12 --mem-gb 30"
S=sppdec_consensus_rule_nogamble_limited; C=cbba_single_task_cbba_single; H=centralized_cc_hungarian_hungarian; M=marl_ippo_limited
CB=cbba_full_cbba_bundle; CK=cbba_full_capability_bundle; A=decentralized_auction_auction_baseline; N=naive_decentralized_reactive
ALL="$S $H $C $CB $A $N $CK $M"
step() { [ -f $L/$1.done ] && return; echo "[GAPS] $1 start $(date +%T)"; }
# 1. Unequal batteries (capacities 100 and 60), all eight strategies against the equal-battery reference
if [ ! -f $L/unequal_batteries.done ]; then step unequal_batteries
  ARMS=""; for b in $ALL; do ARMS="$ARMS $b"; done
  ARMS="$ARMS ${S}_ub ${C}_ub centralized_cc_hungarian_ub_hungarian ${CB}_ub ${A}_ub ${N}_ub ${CK}_ub ${M}_ub"
  RADIO_PROFILE=ideal $RUN --configs $CFG --only-replicates $REP --arms $ARMS --out $O/unequal_batteries > $L/unequal_batteries.log 2>&1; echo finished > $L/unequal_batteries.done; fi
# 2. Sensing: every stress setting for the three strategies that read peers; the worst cases for the five that do not
if [ ! -f $L/sensing_all.done ]; then step sensing_all
  ARMS="$ALL"
  for t in n05 n1 n2 l50 n2l50 d30 d120 bo; do for b in $S $N $M; do ARMS="$ARMS ${b}_$t"; done; done
  for t in n2l50 d120 bo; do for b in $C $CB $A $CK; do ARMS="$ARMS ${b}_$t"; done; ARMS="$ARMS centralized_cc_hungarian_${t}_hungarian"; done
  RADIO_PROFILE=ideal $RUN --configs $CFG --only-replicates $REP --arms $ARMS --out $O/sensing_all > $L/sensing_all.log 2>&1; echo finished > $L/sensing_all.done; fi
# 3. Radio: the five strategies that send messages under seven profiles
for P in ideal lossy50 lossy80 slow outage lossy50_retx3 lossy80_retx5; do
  if [ ! -f $L/radio_$P.done ]; then step radio_$P
    X="$C $CB $CK $A $H"; [ $P = ideal ] && X="$X $S $N $M"; [ $P = lossy50 ] && X="$X $S $N $M"
    RADIO_PROFILE=$P $RUN --configs $CFG --only-replicates $REP --arms $X --out $O/radio_$P > $L/radio_$P.log 2>&1; echo finished > $L/radio_$P.done; fi
done
# 4. Reserve floor: heavy-load configurations, floor 3 (reference), 0 and 6
if [ ! -f $L/reserve_floor.done ]; then step reserve_floor
  ARMS="$ALL"; for t in rf0 rf6; do for b in $S $C $CB $A $N $CK $M; do ARMS="$ARMS ${b}_$t"; done; ARMS="$ARMS centralized_cc_hungarian_${t}_hungarian"; done; ARMS="$ARMS ${S}_rf0_boff"
  RADIO_PROFILE=ideal $RUN --configs A2_P20 A3_P24 A4_P36 A6_P36 A8_P36 A12_P36 --only-replicates 490 491 --arms $ARMS --out $O/reserve_floor > $L/reserve_floor.log 2>&1; echo finished > $L/reserve_floor.done; fi
# 5. Other map sizes (40 x 25 and 16 x 10), all eight strategies
for G in 40x25 16x10; do
  if [ ! -f $L/map_$G.done ]; then step map_$G
    GRID_DIMS=$G RADIO_PROFILE=ideal $RUN --configs A4_P12 A8_P24 A12_P24 --only-replicates $REP --arms $ALL --out $O/map_$G > $L/map_$G.log 2>&1; echo finished > $L/map_$G.done; fi
done
echo "[GAPS] phase 1 finished $(date +%T)" > $L/phase1.done
