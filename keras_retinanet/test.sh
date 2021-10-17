#!/bin/bash
# you might wanna specify WANDB env variables here
# export WANDB_MODE="dryrun"

# fixes some errors (maybe)
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

wandb on
# wandb agent --count 1 hypersphere/masters-thesis/bwtfliv6
# hypersphere/masters-thesis/lw526gt7
# --count 10 hypersphere/masters-thesis/mliy096l
# hypersphere/masters-thesis/qqnxugpm # model selection train eval sweep
# hypersphere/masters-thesis/da4z3et9 # noise resnet152 sweep
python3 $WANDB_DIR/keras_retinanet/bin/evaluate.py "$@"
