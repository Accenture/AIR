#!/usr/bin/env python3

"""
Copyright 2021 Accenture

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import numpy as np
import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.inference import InferenceGenerator
from ..utils.anchors import make_shapes_callback
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval import get_detections
from ..utils.image import preprocess_image_caffe_fast
from ..utils.keras_version import check_keras_version
from ..utils.tf_version import check_tf_version

from ..utils.dataset import compute_dataset_metadata
from ..utils import seed, optimize_tf_parallel_processing

import multiprocessing

NUM_PARALLEL_EXEC_UNITS = multiprocessing.cpu_count()


def default_args():
    return {
        'convert_model' : True,
        'image_tiling_dim': 0,
        'backbone': 'resnet50',
        'gpu': None,
        'score_threshold' : 0.05,
        'iou_threshold' : 0.5,
        'nms_threshold' : 0.5,
        'max_detections' : 100,
        'save_path' : None,
        'image_min_side' : 800,
        'image_max_side' : 1333,
        'config' : None,
        'eval_mode' : "voc2012",

        # misc
        'name' : None,
        'group': None,
        'seed' : None,
    }


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Inference script for the AIR detector.')

    def bool_str(string):
        return string.lower() in {"true", "yes", "y", "on"}

    parser.add_argument('input_image_folder',  help='Input folder for images to inference.')
    parser.add_argument('output_image_folder', help='Output folder to store inferenced images in.')
    parser.add_argument('--model',             help='Path to RetinaNet model.', required=True)
    parser.add_argument('--convert_model',     help='Convert the model to an inference model (ie. the input is a training model).', type=bool_str, default=False)
    parser.add_argument('--image_tiling_dim',  help='Split input image into <this param>^2 overlapping tiles before feeding it into the network.', type=int, default=0)
    parser.add_argument('--backbone',          help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',               help='Id of the GPU to use (as reported by nvidia-smi).', type=int)
    parser.add_argument('--score_threshold',   help='Threshold on score to filter detections with (defaults to 0.05).', default=0.05, type=float)
    parser.add_argument('--nms_threshold',     help='NMS Threshold for two overlapping detections (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--nms_mode',          help='How to merge two overlapping detections in NMS (defaults to "argmax").', default="argmax")
    parser.add_argument('--top_k',             help='Number of top scoring bboxes to keep in merge cluster when nms_mode is not "argmax"', type=int, default=-1)
    parser.add_argument('--max_detections',    help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--image_min_side',    help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image_max_side',    help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',            help='Path to a configuration parameters .ini file (only used with --convert-model).')
    parser.add_argument('--anchor_scale',      help='Scale the anchor boxes by this constant (e.g. if your objects are very small)', type=float, default=1.0)
    parser.add_argument('--profile',           help="ADVANCED: Enable execution profiling to find bottlenecks in the program performance", 
                                               default=False, action="store_true")

    # misc
    parser.add_argument("--seed", help="Fix various random seeds for more reproducible results.", type=int)

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

        if args.backbone == "infer":
            import re
            base_name = os.path.basename(args.model)
            try:
                args.backbone = re.search(r"\_([A-Za-z]*[^\-]+[A-Za-z]+\d+?)\_", base_name).group(1)
            except AttributeError:
                raise ValueError("Cannot infer backbone from model name: " + base_name)

        if "SLURM_JOB_ID" in os.environ:
            print("Slurm Job ID is", os.environ["SLURM_JOB_ID"])
            args.input_image_folder = args.input_image_folder.replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
            args.output_image_folder = args.output_image_folder.replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
        else:
            args.input_image_folder =  args.input_image_folder.replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")
            args.output_image_folder =  args.output_image_folder.replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")
        
    # optionally seed random number generators
    if args.seed:
        seed(args.seed)

    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    # optimize parallel thread usage on CPU and configure GPU
    optimize_tf_parallel_processing(NUM_PARALLEL_EXEC_UNITS, gpu_id=args.gpu)

    # make save path if it doesn't exist
    if args.output_image_folder is not None:
        os.makedirs(args.output_image_folder, exist_ok=True)

    # optionally load config parameters
    if isinstance(args.config, str):
        args.config = read_config_file(args.config)

    # create the generator
    backbone = models.backbone(args.backbone)

    # create an inference generator to hold the images (with no labels)
    generator = InferenceGenerator(
        args.input_image_folder,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        config=args.config,
        shuffle_groups=False,
        preprocess_image=backbone.preprocess_image
    )
    
    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
        anchor_params.scales *= args.anchor_scale

    # load the model
    print('Loading model, this may take a second...')
    # using compile=False might fix some model loading problems (especially with seresnext101 backbone)
    # but inference performance may drop
    model = models.load_model(args.model, backbone_name=args.backbone, compile=True)
    generator.compute_shapes = make_shapes_callback(model)

    # optionally convert the model
    if args.convert_model:
        model = models.convert_model(
                    model,
                    nms=args.nms_mode == "argmax",
                    anchor_params=anchor_params,
                    nms_threshold=args.nms_threshold,
                    max_detections=args.max_detections # // 4 if args.image_tiling_dim else args.max_detections
                )

    print(f"Running inference on image folder: {args.input_image_folder}")
    all_detections, all_inference_times = get_detections(
        generator,
        model,
        score_threshold=args.score_threshold,
        top_k=args.top_k,
        nms_threshold=args.nms_threshold,
        nms_mode=args.nms_mode,
        max_detections=args.max_detections,
        save_path=args.output_image_folder,
        tiling=args.image_tiling_dim,
        profile=args.profile
    )
    print("Inference done! Reporting results...\n")

    # iterate over images
    for i in range(generator.size()):
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            detections = all_detections[i][label]
            img_name = os.path.basename(generator.image_path(i))
            class_name = generator.label_to_name(label)
            num_instances = len(detections)
            if num_instances > 0:
                print(f"Detected {num_instances} instances of class '{class_name}' in image: {img_name}")

    print(f'\nInferenced {generator.size():.0f} images')
    print(f'Average inference time per image: {np.mean(all_inference_times):.4f} s')
    print("-------------------------------------------")
    print(f"Saved output to {args.output_image_folder}")

if __name__ == '__main__':
    main()
