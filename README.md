# AIR: Aerial Inspection RetinaNet for supporting Land Search and Rescue Missions

 A deep learning based object detection solution to automate the aerial drone footage inspection task carried out during SAR operations with drone units.


## Hardware Requirements
- x86 or x64 processor architecture (atleast for Docker installation)
- Nvidia GPU (recommended) with CUDA capability 6.0 or higher

## Software Requirements

Either 
- Docker

or
- Python 3.6
- pip
- Linux or Mac OS (not tested on Windows)


## Quick install instructions using Docker
- Clone this repo 
- Download data set and model
- Start docker environment by running: `bash keras-retinanet-env.sh`
- Inside docker container, build the AIR project by running: `pip3 install .`
- evaluate the AIR detector by running: `bash evaluate.sh`
- check `data/predictions/dauntless-sweep-2_resnet152_pascal-eval` for the output images

## Quick native install instructions 
- Clone this repo 
- Download data set and model
- build the AIR project by running: `pip3 install .`
- evaluate the AIR detector by running: `bash evaluate.sh`
- check `data/predictions/dauntless-sweep-2_resnet152_pascal-eval` for the output images

## Wandb support
- Experiment tracking software by [Weight & Biases](https://wandb.ai/home)
- Controlled with the following environment variables:
    - `WANDB_MODE="disabled"`
        - disables all wandb functionality, required if not logged in to wandb
    - `WANDB_MODE="dryrun"`
        - wandb logging works as normal, but nothing is uploaded to cloud
    - `WANDB_API_KEY=<your_key>`
        - needed to interact with wandb web UI
    - `WANDB_DIR=~/wandb` 
        - select where local log files are stored
    - `WANDB_PROJECT="air-testing"`
        - select your wandb cloud project
    - `WANDB_ENTITY="ML-Mike"`
        - select your wandb account username

## Project folder structure

- **airutils:** air utility functions that augment the standard `keras-retinanet` framework to incorporate *aerial person detection* (APD) support to it.
- **config:**  files for detection scripts
- **data:** input and output data for computer vision algorithms
- **keras-retinanet:** a fork of the [keras-retinanet](https://github.com/fizyr/keras-retinanet) repository with some AIR modifications
- **models:** contains trained computer vision models
