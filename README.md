<img src="keras_retinanet/images/air-logo-large-blue.png" alt="AIR-logo" width="250"/><br><br>

# AIR: Aerial Inspection RetinaNet for supporting Land Search and Rescue Missions

 AIR is a deep learning based object detection solution to automate the aerial drone footage inspection task carried out during SAR operations with drone units. It provides a fast, convenient and reliable way to augment aerial, high-resolution image inspection for clues about human presence by highligthing relevant image regions with bounding boxes, as done in the image below. 

<img src="keras_retinanet/images/air-example.png" alt="AIR-example"/><br><br>

## Key features
- Supports both video and images
- Highly accurate deep learning detector custom built for this application
- Decent processing speed despite the high-resolution images and a very deep neural network used
- Supports object tracking
- Implements the novel MOB algorithm
- Plethora of options for customization (e.g., in bounding box postprocessing)

## Results on HERIDAL test set

In the table below are listed the results of the AIR detector and other notable state-of-the-art methods on the [HERIDAL](http://ipsar.fesb.unist.hr/HERIDAL%20database.html) test set. Note that the FPS metric is a rough estimate based on the reported inference speed and hardware setup by the original authors, and should be taken as mostly directional. We did not test any of the other methods ourselves.

| Method  | Precision  | Recall  | AP  | FPS  |
|---|---|---|---|---|
| Mean shift segmentation method [[1](#references)]|  18.7 | 74.7 | - | -
|  Saliency guided VGG16 [[2](#references)] |   | 34.8 | 88.9 | - | - 
|  Faster R-CNN [[3](#references)] |  67.3 | 88.3 | **86.1** | **1** 
|  Two-stage multimodel CNN [[4](#references)] |  68.9 | **94.7** | - | 0.1 
|  SSD [[4](#references)] | 4.3 | 94.4 | - | -
|  AIR with NMS (ours) | **90.1** | 86.1 | 84.6 | **1**

It turns out AIR achieves state-of-the-art results in precison and inference speed while having comparable recall to the strongest competitors!

## Hardware Requirements
- x86 or x64 processor architecture (at least for the Docker installation)
- NVIDIA® GPU with CUDA® capability 3.5 or higher (recommended)

## Software Requirements

If using containers:
- [Docker](https://www.docker.com/) or [Singularity](https://singularity.hpcng.org/) container system

If using native installation:
- Python 3.6
- `pip` and `setuptools`
- FFmpeg
- Linux or Mac OS (not tested on Windows)

## Quick install instructions using Docker
- Clone this repo
- Either start docker CPU environment by running: `bash keras-retinanet-env.sh`
- Or start docker GPU environment by running: `bash keras-retinanet-gpu-env.sh`
- Inside docker container, build the AIR project by running: `pip3 install .`

## Quick native install instructions 
- Clone this repo 
- To build the AIR project for CPU, run the command: `/usr/bin/python3 -m pip install air-detector[cpu]`
- To build the AIR project for GPU, run the command: `/usr/bin/python3 -m pip install air-detector[gpu]`

## Download example data and the trained model
- Download links coming soon
- Save the demo image folder under `data/images`
- Save the demo videos under `data/videos`
- Save the trained model in `models/` folder
- In the project root, convert the training model to inference model by running: `/bin/bash convert-model.sh`

## Quick demos

Once everything is setup (installation and asset downloads), you might wanna try out these couple simple demos to get the hang of the AIR detector.

### Running inference on HERIDAL test image folder
- In the project root, run the command: `bash infer.sh`
- Check `data/predictions/dauntless-sweep-2_resnet152_pascal-enclose-inference` for the output images

### Evaluating AIR performance on HERIDAL test set (can be slow on CPU)
- In the project root, run the command: `bash evaluate.sh`
- Check `data/predictions/dauntless-sweep-2_resnet152_pascal-enclose-sar-apd-eval` for the output images

### SAR-APD video detection
- In the project root, run the command: `/bin/bash convert-model.sh`
- Then start the inferencing by running: `/usr/bin/python3 video_detect.py -c test_config`
- Check `data/videos/Ylojarvi-gridiajo-two-guys-moving_air_output_compressed.mov` for the output video

## Wandb support
- Experiment tracking software by [Weight & Biases](https://wandb.ai/home)
- AIR experiments can be controlled with the following environment variables:
    - `WANDB_MODE="disabled"`
        - disables all wandb functionality, required if not logged in to wandb
    - `WANDB_MODE="dryrun"`
        - wandb logging works as normal, but nothing is uploaded to cloud
    - `WANDB_API_KEY=<your_api_key>`
        - needed to interact with wandb web UI, you might want to put this in your ``~/.bashrc`` or ``~/.zshrc``, it is also automatically included into the Docker envs this way. To do this run this command with your Wandb API key:
            ```bash
            echo "export WANDB_API_KEY=<your_api_key>" >> ".${SHELL/\/bin\//}rc"; exec $SHELL
            ```
    - `WANDB_DIR=~/wandb` 
        - select where local log files are stored
    - `WANDB_PROJECT="air-testing"`
        - select your wandb cloud project
    - `WANDB_ENTITY="ML-Mike"`
        - select your wandb account username
    - `WANDB_RUN_GROUP="cool-experiments"`
        - Semantic group within a project to log experiments runs to
- More info in [wandb documentation](https://docs.wandb.ai/guides/track/advanced/environment-variables#optional-environment-variables) (not all env variables are supported though)
 
## Project folder structure

- **airutils:** air utility functions that augment the standard `keras-retinanet` framework to incorporate *aerial person detection* (APD) support to it.
- **config:**  files for detection scripts
- **data:** input and output data for computer vision algorithms
- **keras-retinanet:** a fork of the [keras-retinanet](https://github.com/fizyr/keras-retinanet) repository with some AIR modifications
- **models:** contains trained computer vision models

## General toubleshooting
- Try setting the `AIR_VERBOSE=1` enviroment variable to see full TF logs
- If using docker, make sure you're allocating enough memory for the container (like at least 8 GB)
- Tracker options in `video_detect.py` might need to be recalibrated for each use case for the best performance

## Acknowledgements
- Kudos to [Aalto Science-IT project](https://scicomp.aalto.fi/) for providing the compute hardware for training and testing the AIR detector

## References

[1] TURIC, H., DUJMIC, H., AND PAPIC, V. Two-stage segmentation of aerial
images for search and rescue. Information Technology and Control 39, 2 (2010).

[2] BOŽIC-ŠTULIC, D., MARUŠIC, Ž., AND GOTOVAC, S. Deep learning approach in
aerial imagery for supporting land search and rescue missions. International Journal of Computer Vision 127, 9 (2019), 1256–1278.

[3] MARUŠIC, Ž., BOŽIC-ŠTULIC, D., GOTOVAC, S., AND MARUŠIC, T. Region
proposal approach for human detection on aerial imagery. In 2018 3rd International Conference on Smart and Sustainable Technologies (SpliTech) (2018), IEEE, pp. 1–6.

[4] VASIC, M. K., AND PAPIC, V. Multimodel deep learning for person detection in aerial images. Electronics 9, 9 (2020), 1459.