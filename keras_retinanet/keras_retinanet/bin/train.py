#!/usr/bin/env python

"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications copyright (c) 2020-2021 Accenture
"""

"""
Modified by Pasi Pyrrö 2020

Changes:
- integration to wandb library
"""

# For 2x8 core Intel Xeon Gold 6134 @ 3.2GHz processor
NUM_PARALLEL_EXEC_UNITS = 8

import argparse
import os
import sys
import warnings
import numpy as np
import random
import functools

# if "SLURM_ARRAY_TASK_ID" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_ARRAY_TASK_ID"]

import keras
import keras.preprocessing.image
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from . import evaluate
from .. import layers  # noqa: F401
from .. import losses
from .. import models
from ..callbacks import RedirectModel
from ..callbacks.eval import Evaluate
from ..models.retinanet import retinanet_bbox
from ..preprocessing.csv_generator import CSVGenerator
from ..preprocessing.kitti import KittiGenerator
from ..preprocessing.open_images import OpenImagesGenerator
from ..preprocessing.pascal_voc import PascalVocGenerator
from ..utils.anchors import make_shapes_callback, anchor_targets_bbox
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.gpu import setup_gpu
from ..utils.image import random_visual_effect_generator
from ..utils.keras_version import check_keras_version
from ..utils.model import freeze as freeze_model
from ..utils.tf_version import check_tf_version
from ..utils.transform import random_transform_generator

from ..utils.dataset import compute_dataset_metadata
from ..utils import seed, optimize_tf_parallel_processing

import wandb

def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
        model         : The model to load weights for.
        weights       : The weights to load.
        skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(backbone_retinanet, num_classes, weights, args):
# multi_gpu=0,
#                   freeze_backbone=False, lr=1e-5, config=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    modifier = freeze_model if args.freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors   = None
    num_pyramid_levels = 5
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)
        num_anchors   = anchor_params.num_anchors()
        num_pyramid_levels = len(anchor_params.sizes)
        # modified by Pasi Pyrrö
        # scale anchor scales if needed, easier than modifying exact sizes
        # by default scale them by 1
        # useful for making anchors smaller for small objects
        anchor_params.scales *= args.anchor_scale

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    if args.multi_gpu > 1:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, num_pyramid_levels=num_pyramid_levels, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = multi_gpu_model(model, gpus=args.multi_gpu)
    else:
        model          = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, num_pyramid_levels=num_pyramid_levels, modifier=modifier), weights=weights, skip_mismatch=True)
        training_model = model

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model
    training_model.compile(
        loss={
            'regression'    : losses.smooth_l1(),
            'classification': losses.focal(alpha=args.loss_alpha_factor)
        },
        optimizer=keras.optimizers.adam(lr=args.lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        makedirs(args.tensorboard_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir                = args.tensorboard_dir,
            histogram_freq         = 0,
            batch_size             = args.batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from ..callbacks.coco import CocoEval

            # use prediction model for evaluation
            evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
        else:
            evaluation = Evaluate(validation_generator, 
                                  tensorboard=tensorboard_callback, 
                                  weighted_average=args.weighted_average)
                                #   max_detections=1000,
                                #   score_threshold=0.01)
                                # better to keep these as default so validation scores are comparable
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{run_name}_{backbone}_{dataset_type}.h5'.format(run_name=wandb.run.name, 
                    backbone=args.backbone, dataset_type=args.dataset_type)
            ),
            verbose=1,
            period=args.snapshot_interval,
            save_best_only=True,
            monitor="mAP",
            mode='max'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'mAP',
        factor     = args.reduce_lr_factor,
        patience   = args.reduce_lr_patience,
        verbose    = 1,
        mode       = 'max',
        min_delta  = 0.001,
        cooldown   = 0,
        min_lr     = 0
    ))

    callbacks.append(keras.callbacks.EarlyStopping(
        monitor    = 'mAP',
        patience   = args.early_stop_patience,
        mode       = 'max',
        min_delta  = 0.001,
        verbose    = 1
    ))

    if args.tensorboard_dir:
        callbacks.append(tensorboard_callback)

    return callbacks


def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'config'           : args.config,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image,
        'compute_anchor_targets' : functools.partial(
            anchor_targets_bbox,
            negative_overlap=args.foreground_overlap_threshold - 0.1,
            positive_overlap=args.foreground_overlap_threshold
        )
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            prng=np.random.RandomState(args.seed) if args.seed else None,
            min_rotation=-0.5,
            max_rotation=0.5,
            min_translation=(-0.01, 0),
            max_translation=(0.01, 0),
            min_shear=-0.18,
            max_shear=0.18,
            min_scaling=(0.9, 1.0),
            max_scaling=(1.1, 1.0),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
            max_transforms=1
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.8, 1.2),
            brightness_range=(-0.2, 0.2),
            # hue_range=(-0.05, 0.05),
            # saturation_range=(0.95, 1.05),
            color_range=(0.8, 1.2),
            noise_range=(2.0, 5.0),
            equalize_chance=0.8,
            max_effects=1
        )
    else:
        transform_generator = None # random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    if args.dataset_type == 'coco':
        # import here to prevent unnecessary dependency on cocoapi
        from ..preprocessing.coco import CocoGenerator

        train_generator = CocoGenerator(
            args.coco_path,
            'train2017',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = CocoGenerator(
            args.coco_path,
            'val2017',
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'pascal':
        train_generator = PascalVocGenerator(
            args.pascal_path,
            'train',
            image_extension=args.image_extension,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = PascalVocGenerator(
            args.pascal_path,
            'val',
            image_extension=args.image_extension,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'csv':
        train_generator = CSVGenerator(
            args.annotations,
            args.classes,
            "train",
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        if args.val_annotations:
            validation_generator = CSVGenerator(
                args.val_annotations,
                args.classes,
                "val",
                shuffle_groups=False,
                **common_args
            )
        else:
            validation_generator = None
    elif args.dataset_type == 'oid':
        train_generator = OpenImagesGenerator(
            args.main_dir,
            subset='train',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = OpenImagesGenerator(
            args.main_dir,
            subset='validation',
            version=args.version,
            labels_filter=args.labels_filter,
            annotation_cache_dir=args.annotation_cache_dir,
            parent_label=args.parent_label,
            shuffle_groups=False,
            **common_args
        )
    elif args.dataset_type == 'kitti':
        train_generator = KittiGenerator(
            args.kitti_path,
            subset='train',
            transform_generator=transform_generator,
            visual_effect_generator=visual_effect_generator,
            **common_args
        )

        validation_generator = KittiGenerator(
            args.kitti_path,
            subset='val',
            shuffle_groups=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    coco_parser = subparsers.add_parser('coco')
    coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    pascal_parser = subparsers.add_parser('pascal')
    pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')
    pascal_parser.add_argument('--image_extension',   help='Declares the dataset images\' extension.', default='.jpg')

    kitti_parser = subparsers.add_parser('kitti')
    kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')

    def csv_list(string):
        return string.split(',')

    def bool_str(string):
        return string.lower() in {"true", "yes", "y", "on"}

    oid_parser = subparsers.add_parser('oid')
    oid_parser.add_argument('main_dir', help='Path to dataset directory.')
    oid_parser.add_argument('--version',  help='The current dataset version is v4.', default='v4')
    oid_parser.add_argument('--labels_filter',  help='A list of labels to filter.', type=csv_list, default=None)
    oid_parser.add_argument('--annotation_cache_dir', help='Path to store annotation cache.', default='.')
    oid_parser.add_argument('--parent_label', help='Use the hierarchy children of this label.', default=None)

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    csv_parser.add_argument('--val_annotations', help='Path to CSV file containing annotations for validation (optional).')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet_weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.')
    group.add_argument('--no_weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)
    parser.add_argument('--backbone',         help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch_size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi_gpu',        help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi_gpu_force',  help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--initial_epoch',    help='Epoch from which to begin the train, useful if resuming from snapshot.', type=int, default=0)
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--loss_alpha_factor',help='Class balancing factor for the classification loss', type=float, default=0.25)
    parser.add_argument('--snapshot_path',    help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard_dir',  help='Log directory for Tensorboard output', default='')  # default='./logs') => https://github.com/tensorflow/tensorflow/pull/34870
    parser.add_argument('--snapshot_interval',help='Interval in epochs between saving snapshots.', type=int, default=2)
    parser.add_argument('--no_snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no_evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze_backbone',  help='Freeze training of backbone layers.', type=bool_str, default=False)
    parser.add_argument('--random_transform', help='Randomly transform image and annotations.', type=bool_str, default=False)
    parser.add_argument('--image_min_side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image_max_side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--no_resize',        help='Don''t rescale the image.', type=bool_str, default=False)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted_average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute_val_loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')
    parser.add_argument('--reduce_lr_patience', help='Reduce learning rate after validation loss decreases over reduce_lr_patience epochs', type=int, default=2)
    parser.add_argument('--reduce_lr_factor', help='When learning rate is reduced due to reduce_lr_patience, multiply by reduce_lr_factor', type=float, default=0.1)
    parser.add_argument('--early_stop_patience', help='Stop training early if mAP has not increased for early_stop_patience epochs', type=int, default=5)
    parser.add_argument('--anchor_scale',     help='Scale the anchor boxes by this constant (e.g. if your objects are very small)', type=float, default=1.0)
    parser.add_argument('--test_after_training', help="Evaluate model on test set after training", type=bool_str, default=True)

    parser.add_argument("--foreground_overlap_threshold", help="Minimum overlap (IoU 0.1-0.9) with Anchor and GT box to consider"
                        "the anchor as positive (i.e. contributes to foreground class learning)", type=float, default=0.5)

    # Fit generator arguments
    parser.add_argument('--multiprocessing',  help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers',          help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max_queue_size',   help='Queue length for multiprocessing workers in fit_generator.', type=int, default=10)

    # misc
    parser.add_argument('--name',  help='Name for this run (default: auto-generated by wandb)')
    parser.add_argument('--group', help='W&B run group for this run (default: retinanet-train)')
    parser.add_argument('--tags',  help='Add custom tags to wandb run', type=csv_list, default=[])
    parser.add_argument("--seed", help="Fix various random seeds for more reproducible results.", type=int)

    return check_args(parser.parse_args(args))


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
        args = parse_args(args)

    # optionally seed random number generators
    if args.seed:
        seed(args.seed)

    # save arguments to weights and biases
    wandb.config.update(args)

    if args.name is not None:
        wandb.run.name = args.name
    
    if args.group is not None:
        wandb.run.group = args.group

    # puts some tags on the run to distinguish it from other runs
    wandb.run.tags += ["train", args.backbone]

    if args.config and "3_levels" in args.config:
        wandb.run.tags.append("FPN-3")

    if args.foreground_overlap_threshold:
        wandb.run.tags.append(f"AFGR-{args.foreground_overlap_threshold:.2f}".rstrip("0"))
    
    if args.loss_alpha_factor:
        wandb.run.tags.append(f"LAF-{args.loss_alpha_factor:.2f}".rstrip("0"))
    
    if args.random_transform:
        wandb.run.tags.append("data-augmentation")
    
    wandb.run.tags += args.tags

    wandb.run.save()

    print(f"Starting RetinaNet training session '{wandb.run.name}' "
          f"with '{args.backbone}' backbone inside group '{wandb.run.group}'...")
    print(f"Running {args.epochs} epochs with {args.steps} steps each...")

    # create object that stores backbone information
    backbone = models.backbone(args.backbone)

    # make sure keras and tensorflow are the minimum required version
    check_keras_version()
    check_tf_version()

    if "pascal_path" in vars(args):
        path_key = "pascal_path"
    else:
        path_key = "coco_path"

    arg_dict = vars(args)

    # if "SLURM_ARRAY_JOB_ID" in os.environ:
    #     arg_dict[path_key] = arg_dict[path_key].replace("UNIQUE_JOB_FOLDER", 
    #         "{}_{}".format(os.environ["SLURM_ARRAY_JOB_ID"], os.environ["SLURM_ARRAY_TASK_ID"]))
    
    if "SLURM_JOB_ID" in os.environ:
        print("Slurm Job ID is", os.environ["SLURM_JOB_ID"])
        try:
            arg_dict[path_key] = arg_dict[path_key].replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
        except KeyError: # csv dataset
            arg_dict["annotations"] = arg_dict["annotations"].replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
            arg_dict["classes"] = arg_dict["classes"].replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
            arg_dict["val_annotations"] = arg_dict["val_annotations"].replace("UNIQUE_JOB_FOLDER", os.environ["SLURM_JOB_ID"])
    else:
        try:
            arg_dict[path_key] = arg_dict[path_key].replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")
        except KeyError: # csv dataset
            arg_dict["annotations"] = arg_dict["annotations"].replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")
            arg_dict["classes"] = arg_dict["classes"].replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")
            arg_dict["val_annotations"] = arg_dict["val_annotations"].replace("UNIQUE_JOB_FOLDER", "sar-uav-cv")


    print("Computing and saving dataset metadata, this may take a while...")
    try:
        dataset_folder = arg_dict[path_key]
    except KeyError:
        dataset_folder = os.path.dirname(arg_dict["annotations"])
    metadata = compute_dataset_metadata(dataset_folder)
    print("\n".join([f"{k}: {v}" for k, v in metadata.items()]))
    wandb.config.update(metadata)

    if "tiled" in wandb.config.dataset_name:
        wandb.run.tags.append("tiled-images")

    config = None

    # optionally choose specific GPU
    if args.gpu is not None:
        if args.gpu == "array":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print(f"Using array task assigned GPU:{os.environ.get('CUDA_VISIBLE_DEVICES')}")
            config = tf.compat.v1.ConfigProto(device_count={"GPU": 1})
            config.gpu_options.allow_growth = True
            # tf.keras.backend.set_session(tf.Session(config=config))
            
        else:
            print(f"Setting up user specified GPU:{args.gpu}")
            setup_gpu(args.gpu)
    
    # optimize parallel thread usage
    optimize_tf_parallel_processing(NUM_PARALLEL_EXEC_UNITS, config)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generators
    train_generator, validation_generator = create_generators(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model            = models.load_model(args.snapshot, backbone_name=args.backbone)
        training_model   = model
        anchor_params    = None
        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
            anchor_params.scales *= args.anchor_scale
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = backbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            args=args
            # multi_gpu=args.multi_gpu,
            # freeze_backbone=args.freeze_backbone,
            # lr=args.lr,
            # config=args.config
        )

    # print model summary
    print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    print("Creating callbacks...")
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    callbacks += [wandb.keras.WandbCallback(
                    data_type="image",
                    monitor="mAP",
                    mode="max",
                    labels=['aeroplane', 'bicycle', 'bird',
                            'boat', 'bottle', 'bus',
                            'car', 'cat', 'chair',
                            'cow', 'diningtable', 'dog',
                            'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa',
                            'train', 'tvmonitor'],
                    save_model=False)]
                    # generator=validation_generator,
                    # validation_steps=1)]

    print("Callbacks ready, starting training...")
    train_history = training_model.fit_generator(
        generator=train_generator,
        # x=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=1, # args.workers,
        use_multiprocessing=False, #args.multiprocessing,
        max_queue_size=16, #args.max_queue_size,
        validation_data=validation_generator,
        initial_epoch=args.initial_epoch
    )

    if args.test_after_training:
        from copy import deepcopy
        trained_model_path = os.path.join(os.path.dirname(__file__), "..", 
            "snapshots", f"{wandb.run.name}_{args.backbone}_{args.dataset_type}.h5")
        if os.path.exists(trained_model_path):
            test_args = deepcopy(args)
            vars(test_args)["model"] = trained_model_path
            default_args = evaluate.default_args()
            for key in default_args:
                if key not in vars(test_args):
                    vars(test_args)[key] = default_args[key]
            evaluate.main(test_args)
        else:
            print("No trained model found to evaluate...")

    return train_history


if __name__ == '__main__':
    # initialize weights and biases
    if WANDB_ENABLED:
        wandb.init(entity=os.environ.get("WANDB_ENTITY"),
                   group=os.environ.get("WANDB_RUN_GROUP"),
                   project=os.environ.get("WANDB_PROJECT"))
    main()
