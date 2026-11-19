#!/bin/bash
# Mixed-speed fleet (nominal and half speed) for every strategy, same missions (7 development configs x fresh replicates 470-472, realistic energy model); MARL also on the three-speed fleet [1.0, 0.75, 0.5], a profile it never trained on (transfer test, MARL only: the rule arms have no three-speed arm).
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0 MARL_WEIGHTS=$PWD/data/marl/v3_shape_mb_best.npz
V=${PYTHON:-python3}
$V -m src.run_batch --configs A2_P8 A3_P12 A4_P12 A6_P18 A8_P24 A12_P24 A16_P16 --only-replicates 470 471 472 --arms sppdec_consensus_rule_nogamble_hs_limited centralized_cc_hungarian_hs_hungarian cbba_single_task_hs_limited cbba_full_hs_limited decentralized_auction_hs_limited naive_decentralized_hs_limited marl_ippo_limited_hs marl_ippo_limited_hs3 --out experiments/peer_model/speed_transfer --jobs 4 > experiments/peer_model/speed_transfer.log 2>&1
echo finished > experiments/peer_model/speed_transfer.done
