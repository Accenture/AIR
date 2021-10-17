# AIR: Aerial Inspection RetinaNet for supporting Land Search and Rescue Missions

 A deep learning based object detection solution to automate the aerial drone footage inspection task carried out during SAR operations with drone units.


## Hardware Requirements
- x86 or x64 processor architecture (at least for Docker installation)
- NVIDIA® GPU with CUDA® capability 3.5 or higher (recommended)

## Software Requirements

Either
- Docker

or
- Python 3.6
- `pip` and `setuptools`
- Linux or Mac OS (not tested on Windows)

## Quick install instructions using Docker
- Clone this repo 
- Download data set and model
- Either start docker CPU environment by running: `bash keras-retinanet-env.sh`
- Or start docker GPU environment by running: `bash keras-retinanet-gpu-env.sh`
- Inside docker container, build the AIR project by running: `pip3 install .`
- evaluate the AIR detector by running: `bash evaluate.sh`
- check `data/predictions/dauntless-sweep-2_resnet152_pascal-enclose-sar-apd-eval` for the output images

## Quick native install instructions 
- Clone this repo 
- Download data set and model
- To build the AIR project for CPU, run the command: `/usr/bin/python3 -m pip install air-detector[cpu]`
- To build the AIR project for GPU, run the command: `/usr/bin/python3 -m pip install air-detector[gpu]`
- Test the AIR detector by running: `bash evaluate.sh`
- Check `data/predictions/dauntless-sweep-2_resnet152_pascal-enclose-sar-apd-eval` for the output images

## Wandb support
- Experiment tracking software by [Weight & Biases](https://wandb.ai/home)
- Controlled with the following environment variables:
    - `WANDB_MODE="disabled"`
        - disables all wandb functionality, required if not logged in to wandb
    - `WANDB_MODE="dryrun"`
        - wandb logging works as normal, but nothing is uploaded to cloud
    - `WANDB_API_KEY=<your_key>`
        - needed to interact with wandb web UI, you might want to put this in your ``~/.bashrc`` or ``~/.zshrc``, it is also automatically included into the Docker envs this way. To do this run this with your Wandb API key:
            ```bash
            echo "export WANDB_API_KEY=<your_key>" >> ".${SHELL/\/bin\//}rc"; exec $SHELL
            ```
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
