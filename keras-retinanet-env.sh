#!/bin/bash
# usage: bash tf-gpu-env.sh
docker run --env WANDB_API_KEY=$WANDB_API_KEY -it --network host -v $PWD:/home -w /home intrinsick/keras-retinanet-gpu:latest /bin/bash
