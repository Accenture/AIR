#!/bin/bash
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  echo 'Usage: bash '$(basename $0)' [CONTAINER_TYPE="docker"]'
  exit 0
fi

CONTAINER_TYPE="docker"

CONTAINER_TYPE="${1:-$CONTAINER_TYPE}"
if [[ "$CONTAINER_TYPE" == "docker" ]]; then
    docker run --gpus all --env WANDB_API_KEY=$WANDB_API_KEY -it --network host -v $PWD:/home -w /home intrinsick/keras-retinanet-gpu:latest /bin/bash
elif [[ "$CONTAINER_TYPE" == "singularity" ]]; then
    # YOU MIGHT WANNA CHANGE WHERE SINGULARITY STORES ITS CACHES (THEY TAKE UP A LOT OF SPACE)
    # IT'S LIKELY NOT A GOOD IDEA TO STORE THEM INSIDE THE MOUNTED VOLUME EITHER
    export SINGULARITY_CACHEDIR=$PWD/../.singularity
    # singularity build keras-retinanet-gpu.simg docker://intrinsick/keras-retinanet-gpu:latest
    singularity shell -B $PWD:/home --writable --nv docker://intrinsick/keras-retinanet-gpu:latest
else
    echo "Invalid container type: ${CONTAINER_TYPE}"
fi
