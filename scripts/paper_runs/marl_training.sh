#!/bin/bash
# MARL v3 test of the two helpful changes together against v3_shape and v3_minibatch (same seed 98, lr 1e-3, entropy 0.003, local critic, same frozen current world): --shape 1.0 and --minibatch 512, from scratch, 25 rounds x 150 missions. Resumable.
cd "$(dirname "$0")/../.."
export PYTHONPATH=.:src SDL_VIDEODRIVER=dummy PYTHONHASHSEED=0 PARCEL_WEIGHT_RANGE=0.2,8.0
V=${PYTHON:-python3}
setsid nohup $V -m src.marl.train --out experiments/marl2/v3_shape_mb --seed 98 --max-rounds 25 --hours 3 --episodes 150 --jobs 8 --critic local --lr 1e-3 --entropy 0.003 --shape 1.0 --minibatch 512 --validate-every 5 --repo $PWD >> experiments/marl2/v3_shape_mb.log 2>&1 < /dev/null &
echo started
