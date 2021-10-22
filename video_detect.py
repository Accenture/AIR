"""
Copyright 2020-2021 Accenture

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

'''
Aerial Inspection RetinaNet based people detector for finding lost people from topdown drone video footage in a forest terrain.
Possible output types are: video and json. The output type is inferenced from the given -o (--output) parameter.

Example usage:
python detect_people_retinanet.py --input path/to/video.mp4

Author: Pasi Pyrrö
Date: 7.2.2020
'''

import os
import re
import gc
import cv2
import sys
import time
import argparse
import functools
import itertools
import numpy as np
from collections import deque

# define absolute folder locations
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, "data", "videos")

# make utils and keras-retinanet folders available for python
sys.path.insert(0, os.path.join(current_dir, 'airutils'))
sys.path.insert(0, os.path.join(current_dir, 'keras-retinanet'))

valid_output_types = ("video", "json", "exporter")

class Params(object):
    ''' Holds people detection algorithm parameters and output options '''

    # load parameters from a config file instead
    CONFIG_FILE = None

    # output options
    OUTPUT_TYPE = "video"
    LABEL_MAPPING = "pascal"
    VIDEO_FILE = None
    OUT_RESOLUTION = None
    OUTPUT_PATH = None
    FRAME_OFFSET = 0
    PROCESS_NUM_FRAMES = None
    COMPRESS_VIDEO = True

    # algorithm parameters
    MODEL = "dauntless-sweep-2_resnet152_pascal-nms-inference.h5"
    BACKBONE = "resnet152"
    CONFIDENCE_THRES = 0.25
    DETECT_EVERY_NTH_FRAME = 60
    MAX_DETECTIONS_PER_FRAME = 20
    USE_TRACKING = True
    SHOW_DETECTION_N_FRAMES = 20
    USE_GPU = False
    PROFILE = False
    IMAGE_TILING_DIM = 1
    IMAGE_MIN_SIDE = 1525
    IMAGE_MAX_SIDE = 2025
    MERGE_MODE = "argmax"
    MOB_ITERS = 1
    BBA_IOU_THRES = 0.5
    TOP_K=-1


def resolve_output_type(output):
    if output is None:
        return Params.OUTPUT_TYPE
    elif output.endswith(".json"):
        return valid_output_types[1] # json
    elif re.search(r"\.[a-zA-Z0-9]{3,4}$", output):
        return valid_output_types[0] # video
    return valid_output_types[2]     # exporter object


def parse_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', default=Params.CONFIG_FILE, 
                        help="Load settings from a configuration file in the /config/ folder")
    parser.add_argument('-i', '--input', help="Input video path")
    parser.add_argument('-o', '--output', default=Params.OUTPUT_PATH, help="Output video path")
    parser.add_argument('-lm', '--label-mapping', default=Params.LABEL_MAPPING,
                        const=Params.LABEL_MAPPING, nargs="?", choices=("pascal", "coco"),
                        help=f'Select label mapping from object id to name (default: "{Params.LABEL_MAPPING}")')
    parser.add_argument('-r', '--resolution', type=int,
                        nargs=2, help="Output video resolution")
    parser.add_argument('--image-tiling-dim', 
                        help='Split input image into <this param>^2 overlapping tiles before feeding it into the network.', 
                        type=int, default=Params.IMAGE_TILING_DIM)
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', 
                        type=int, default=Params.IMAGE_MIN_SIDE)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', 
                        type=int, default=Params.IMAGE_MAX_SIDE)
    parser.add_argument('-n', '--num-frames', type=int,
                        default=Params.PROCESS_NUM_FRAMES, help="How many frames to process?")
    parser.add_argument('-fo', '--frame-offset', type=int,
                        default=Params.FRAME_OFFSET, help="Where to start processing?")
    parser.add_argument('-g', '--gpu', action="store_true", default=Params.USE_GPU,
                        help="Use GPU hardware accerelation")
    parser.add_argument('-nc', '--no-compress', action="store_false", default=Params.COMPRESS_VIDEO,
                        help="Don't compress the output video, "
                             "but it can become huge (NOT RECOMMENDED)")
    parser.add_argument('-t', '--confidence-thres', type=float, default=Params.CONFIDENCE_THRES,
                        help="ADVANCED: Confidence threshold for people classification, "
                             "increasing this improves precision but lowers recall")
    parser.add_argument('-d', '--detect-every-nth-frame', type=int, default=Params.DETECT_EVERY_NTH_FRAME,
                        help="ADVANCED: Performance parameter, decides how often we run retinanet object "
                        "detection, low value increases detection accuracy but is much slower")
    parser.add_argument('-nt', '--no-tracking', action="store_false", default=Params.USE_TRACKING,
                        help="ADVANCED: Don't interpolate between detection frames so the bounding "
                             "box tracking looks smooth, has no effect if -d flag is set to 1. If interpolation is not "
                             "enabled bounding boxes flash on top of video at (maximum) frequency specified by -d flag, "
                             "useful to disable when testing retinanet detection performance")
    parser.add_argument('-bb', '--backbone', default=Params.BACKBONE,
                        help="ADVANCED: Select backbone network for the inference model, "
                             "check the available options here: https://github.com/fizyr/keras-retinanet/tree/master/keras_retinanet/models")
    parser.add_argument('-m', '--model', default=Params.MODEL,
                        help="ADVANCED: Select the trained and converted inference model for detection, "
                             "it should be located in the ../models/ directory!")
    parser.add_argument('-p', '--profile', default=Params.PROFILE, action="store_true",
                        help="ADVANCED: Enable execution profiling to find bottlenecks in the program performance")
    parser.add_argument('--merge-mode', 
                        help=f'ADVANCED: How to merge two overlapping detections in BBA (defaults to "{Params.MERGE_MODE}").', default=Params.MERGE_MODE)
    parser.add_argument('--top_k',            
                        help='ADVANCED: Number of top scoring bboxes to keep in merge cluster when nms_mode is not "argmax"', type=int, default=Params.TOP_K)
    parser.add_argument('--bba_iou_threshold',    
                        help=f'ADVANCED: BBA IoU threshold for two overlapping detections (defaults to {Params.BBA_IOU_THRES}).', default=Params.BBA_IOU_THRES, type=float)

    args = parser.parse_args()
    Params.CONFIG_FILE = args.config_file

    if Params.CONFIG_FILE:
        import importlib
        config = importlib.import_module("config." + Params.CONFIG_FILE.replace(".py",
                                         "").replace("config", "").replace(os.path.sep, ""))
        for key, value in config.__dict__.items():
            if not key.startswith("__"):
                setattr(Params, key, value)
    else:
        if not args.input:
            parser.print_help(sys.stderr)
            print("Missing input video! Please use -i/--input command line switch or specify input file in the configuration file.")
            sys.exit(1)
        if args.resolution:
            Params.OUT_RESOLUTION = tuple(args.resolution)
        Params.VIDEO_FILE = args.input
        Params.OUTPUT_PATH = args.output
        Params.OUTPUT_TYPE = resolve_output_type(args.output)
        Params.COMPRESS_VIDEO = args.no_compress
        Params.IMAGE_TILING_DIM = args.image_tiling_dim
        Params.IMAGE_MIN_SIDE = args.image_min_side
        Params.IMAGE_MAX_SIDE = args.image_max_side

        Params.MODEL = args.model
        Params.BACKBONE = args.backbone
        Params.CONFIDENCE_THRES = args.confidence_thres 
        Params.DETECT_EVERY_NTH_FRAME = args.detect_every_nth_frame
        Params.USE_TRACKING = args.no_interpolation
        Params.PROCESS_NUM_FRAMES = args.num_frames
        Params.FRAME_OFFSET = args.frame_offset
        Params.USE_GPU = args.gpu
        Params.PROFILE = args.profile
    
    return args


# display argparse help message before tensorflow starts to initialize
if __name__ == '__main__':
    print("""
      .o.        o8o  ooooooooo.   
     .888.       `"'  `888   `Y88. 
    .8"888.     oooo   888   .d88' 
   .8' `888.    `888   888ooo88P'  
  .88ooo8888.    888   888`88b.    
 .8'     `888.   888   888  `88b.  
o88o     o8888o o888o o888o  o888o 
================================== 
    """)
    parse_args()


import keras
from keras_retinanet.keras_retinanet import models
from keras_retinanet.keras_retinanet.utils.image import preprocess_image_caffe_fast, resize_image as resize_func
from keras_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.keras_retinanet.utils.eval import run_inference_on_image
from keras_retinanet.keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.keras_retinanet.utils import optimize_tf_parallel_processing

from dataset.detection_exporter import DetectionExporter
from video.video_iterator import VideoIterator
from video.videowriter import VideoWriter
from kalman_tracker import KalmanConfig
import video.vidtools as vid
import kalman_tracker as kt

# coco dataset labels
labels_to_names_coco = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
                        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
                        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
                        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
                        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
                        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
                        63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                        73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                        78: 'hair drier', 79: 'toothbrush'}

labels_to_names_pascal = {
    0  : 'aeroplane',
    1  : 'bicycle',
    2  : 'bird',
    3  : 'boat' ,
    4  : 'bottle',  
    5  : 'bus',     
    6  : 'car',
    7  : 'cat',
    8  : 'chair',
    9  : 'cow',     
    10 : 'diningtable',
    11 : 'dog',
    12 : 'horse',       
    13 : 'motorbike',  
    14 : 'person', 
    15 : 'pottedplant', 
    16 : 'sheep',
    17 : 'sofa',      
    18 : 'train',       
    19 : 'tvmonitor',
}

labels_to_names = labels_to_names_coco if Params.LABEL_MAPPING == "coco" else labels_to_names_pascal


def main(exporter=None):

    # tracking options
    KalmanConfig.HISTORY_SPAN = [3, 5]
    KalmanConfig.CONFIDENCE_BOUNDS = [Params.CONFIDENCE_THRES-0.01, Params.CONFIDENCE_THRES+0.05]
    KalmanConfig.TRACKING_DELTA_THRES_MULT = 2
    KalmanConfig.INITIAL_PROCESS_NOISE = 200
    KalmanConfig.INITIAL_COVARIANCE = 500
    KalmanConfig.INITIAL_MEASUREMENT_NOISE = 50
    KalmanConfig.TIMESTEP = 1

    # make sure we use absolute path
    if not os.path.isabs(Params.VIDEO_FILE):
        Params.VIDEO_FILE = os.path.join(current_dir, Params.VIDEO_FILE)

    assert os.path.exists(Params.VIDEO_FILE), f"Input video file '{Params.VIDEO_FILE}' does not exist!"

    if Params.USE_GPU:
        # use gpu:0
        setup_gpu(0)
        optimize_tf_parallel_processing(2)
    else:
        setup_gpu("cpu")
        optimize_tf_parallel_processing(8)

    if Params.PROFILE:
        import cProfile, pstats
        pr = cProfile.Profile(builtins=False)

    # load the inference model with selected backbone
    model_path = os.path.join(current_dir, 'models', Params.MODEL)
    model = models.load_model(model_path, backbone_name=Params.BACKBONE, compile=True)

    if Params.IMAGE_MIN_SIDE is None or Params.IMAGE_MAX_SIDE is None:
        resize_image = None
    else:
        resize_image = functools.partial(resize_func, min_side=Params.IMAGE_MIN_SIDE, max_side=Params.IMAGE_MAX_SIDE)

    detection_exporter = exporter

    running_id = 1
    seen_idx = 0
    detections = []
    start_time = time.time()

    running_id = 1
    detections = []
    detection_disp_counter = 0
    inference_count = 0

    old_bboxes, old_scores, old_labels = ([], [], [])
    bboxes, scores, labels = ([], [], [])

    if not Params.OUTPUT_PATH:
        dir_name, base_name = os.path.split(Params.VIDEO_FILE)
        vid_file_name, vid_file_ext = os.path.splitext(base_name)
        if Params.OUTPUT_TYPE == "video":
            out_path = os.path.join(
                dir_name, vid_file_name + "_air_output" + vid_file_ext)
        elif Params.OUTPUT_TYPE == "json":
            out_path = os.path.join(
                dir_name, vid_file_name + ".json")
        else:
            out_path = ""
    else:
        if not os.path.exists(Params.OUTPUT_PATH):
            os.makedirs(os.path.dirname(Params.OUTPUT_PATH), exist_ok=True)
        out_path = Params.OUTPUT_PATH

    real_fps = vid.read_video_fps(Params.VIDEO_FILE)
    fps = int(real_fps)
    in_res = vid.read_video_resolution(Params.VIDEO_FILE)

    if Params.USE_TRACKING:
        print("\nUsing kalman filter based tracking with settings:")
        print("=================================================")
        print("\n".join([f"{k}: {v}" for k, v in vars(KalmanConfig).items() if not k.startswith("__")]))
        print("")
    
    CHUNK_SIZE = Params.DETECT_EVERY_NTH_FRAME
    disable_video = Params.OUTPUT_TYPE.lower() != valid_output_types[0]
    if Params.OUT_RESOLUTION is None:
        Params.OUT_RESOLUTION = in_res
    if detection_exporter is None:
        detection_exporter = DetectionExporter(out_path, real_fps, in_res, Params.OUT_RESOLUTION, 
                                               Params.DETECT_EVERY_NTH_FRAME, placeholder=Params.OUTPUT_TYPE == "video")
    # 'placeholder=True' keyword argument disables writing but retains context manager for syntactical reasons
    with detection_exporter:
        with VideoWriter(out_path, Params.OUT_RESOLUTION, fps=fps, codec="mp4v", compress=Params.COMPRESS_VIDEO, placeholder=disable_video) as writer:
            with VideoIterator(Params.VIDEO_FILE, max_slice=CHUNK_SIZE) as vi:
                print("\n* * * * *")
                print(f"Starting object detection from frame number {Params.FRAME_OFFSET}")
                print(f"Using inference model '{Params.MODEL}'' ({Params.BACKBONE} backbone) for detection")
                print(f"Inference interval is {Params.DETECT_EVERY_NTH_FRAME} frames")
                print("Output type is", Params.OUTPUT_TYPE.upper())
                info_str = "all remaining frames..." if Params.PROCESS_NUM_FRAMES is None else f"{Params.PROCESS_NUM_FRAMES} frames in total..."
                if Params.PROFILE:
                    pr.enable()
                    print("Execution profiler is turned ON")
                print("Processing", info_str)
                print("* * * * *\n")
                
                fps_timer = time.time()   # time since last FPS measurement
                fps_counter = 0           # frame count since last FPS measurement
                i = Params.FRAME_OFFSET   # offsetted frame index
                detections = []           # list of tracked detections from kalman filter

                vi.seek(Params.FRAME_OFFSET)      
                gen = vi if Params.OUTPUT_TYPE == "video" else itertools.repeat(None)
                if Params.OUTPUT_TYPE != "video":
                    fps_counter =  -Params.DETECT_EVERY_NTH_FRAME

                # main loop - iterate over all (specified) video frames
                for j, frame in enumerate(gen):

                    if Params.OUTPUT_TYPE != "video":
                        i = Params.FRAME_OFFSET + j * Params.DETECT_EVERY_NTH_FRAME
                        try:
                            frame = vi[i]
                        except IndexError:
                            break
                        fps_counter += Params.DETECT_EVERY_NTH_FRAME - 1
                    else:
                        i = j + Params.FRAME_OFFSET

                    if Params.PROCESS_NUM_FRAMES is not None and i >= Params.PROCESS_NUM_FRAMES + Params.FRAME_OFFSET:
                        break

                    if j % Params.DETECT_EVERY_NTH_FRAME == 0 or Params.OUTPUT_TYPE != "video":

                        image = preprocess_image_caffe_fast(frame)
                        bboxes, scores, labels = run_inference_on_image(model, image, resize_image,
                            score_threshold=Params.CONFIDENCE_THRES,
                            max_detections=Params.MAX_DETECTIONS_PER_FRAME // Params.IMAGE_TILING_DIM,
                            top_k=Params.TOP_K,
                            tiling_dim=Params.IMAGE_TILING_DIM,
                            nms_threshold=Params.BBA_IOU_THRES,
                            nms_mode=Params.MERGE_MODE,
                            tiling_overlap=100,
                            mob_iterations=Params.MOB_ITERS
                        )
                        inference_count += 1

                        if bboxes.shape[0] > 0:
                            valid_detections = True
                            detection_disp_counter = 0
                            labels = [labels_to_names[l] for l in labels]
                            old_bboxes, old_scores, old_labels = bboxes, scores, labels
                        else:
                            valid_detections = False
                            bboxes, scores, labels = ([], [], [])

                        new_detections = kt.get_detections_from_bboxes(labels, bboxes, scores)
                            
                        if Params.USE_TRACKING:
                            detections, running_id = kt.match_and_update_detections(
                                new_detections, detections, running_id)
                            if Params.OUTPUT_TYPE != "video":
                                unseen_detections = [d for d in detections if d.object_id > seen_idx and d.is_valid]
                                if unseen_detections:
                                    seen_idx = max(unseen_detections, key=lambda d: d.object_id).object_id
                                    detection_exporter.add_detections_at_frame(unseen_detections, i)
                            else:
                                frame = kt.visualize_detections(frame, detections, uncertain_color="blue")
                            detection_exporter.update_timeseries(detections, i)
                        elif Params.OUTPUT_TYPE != "video":
                            detection_exporter.add_detections_at_frame(new_detections, i)
                            detection_exporter.update_timeseries(new_detections, i)
                            
                        for new_det in new_detections:
                            print(f"[Frame {i}] Detected: '{new_det.object_class}' ({100.*new_det.confidence:.1f} % confidence)")
                            
                        for det in detections:
                            if det.object_id < running_id:
                                print(f"[Frame {i}] Tracking: '{det.object_class} {det.object_id}' ({100.*det.confidence:.1f} % confidence){', passed tracker validation!' if det.is_valid else ''}")

                    elif Params.OUTPUT_TYPE == "video":
                        if Params.USE_TRACKING:
                            interp_detections = kt.interpolate_detections(
                                detections, (i % Params.DETECT_EVERY_NTH_FRAME) / Params.DETECT_EVERY_NTH_FRAME)
                            frame = kt.visualize_detections(
                                frame, interp_detections, uncertain_color="blue")
                        elif valid_detections and detection_disp_counter < Params.SHOW_DETECTION_N_FRAMES:
                            frame = kt.visualize_bboxes(
                                frame, old_labels, old_bboxes, old_scores)
                            detection_disp_counter += 1

                    writer.write(frame, bgr_to_rgb=False)
                    fps_counter += 1
                    if fps_counter >= 100:
                        processing_fps = fps_counter / (time.time() - fps_timer)
                        if Params.OUTPUT_TYPE == "video":
                            print(f"[Time {time.time() - start_time:.1f} s] Processed {j+1} frames ({(j+1)/(fps+1e-6):.1f} seconds of video) @ {processing_fps:.1f} FPS, inferred {inference_count} frames")
                        else:
                            print(f"[Time {time.time() - start_time:.1f} s] Inferred {inference_count} frames @ {processing_fps:.1f} FPS")
                        fps_counter = 0
                        fps_timer = time.time()

    if Params.PROFILE:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats("cumulative")
        # ps.reverse_order()
        ps.print_stats(50)

    print(f"Overall processing time: {time.time() - start_time:.3f} s")
    return detection_exporter if Params.OUTPUT_TYPE == "exporter" else None


if __name__ == '__main__':
    main()