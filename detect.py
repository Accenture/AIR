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
RetinaNet based people detector for finding lost people from topdown drone video footage in a forest terrain.
Possible output types are: video and json. The output type is inferenced from the given -o (--output) parameter.

Example usage:
python detect_people_retinanet.py --input path/to/video.mp4

Author: Pasi Pyrrö
Date: 7.2.2020
'''

# %%
import os
import re
import gc
import cv2
import sys
import time
import argparse
import itertools
import numpy as np
from collections import deque

# define absolute folder locations
current_dir = os.path.dirname(os.path.abspath(__file__))
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
    INTERPOLATE_BETWEEN_DETECTIONS = True
    SHOW_DETECTION_N_FRAMES = 20
    USE_GPU = False
    PROFILE = False
    IMAGE_TILING_DIM = 0
    IMAGE_MIN_SIDE = 1525
    IMAGE_MAX_SIDE = 2025


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
    parser.add_argument('-ni', '--no-interpolation', action="store_false", default=Params.INTERPOLATE_BETWEEN_DETECTIONS,
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

    args = parser.parse_args()
    Params.CONFIG_FILE = args.config_file

    if Params.CONFIG_FILE:
        import importlib
        config = importlib.import_module("config." + Params.CONFIG_FILE.replace(".py", ""))
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
        Params.INTERPOLATE_BETWEEN_DETECTIONS = args.no_interpolation
        Params.PROCESS_NUM_FRAMES = args.num_frames
        Params.FRAME_OFFSET = args.frame_offset
        Params.USE_GPU = args.gpu
        Params.PROFILE = args.profile
    
    return args


# display argparse help message before tensorflow starts to initialize
if __name__ == '__main__':
    parse_args()


import keras
from keras_retinanet.keras_retinanet import models
from keras_retinanet.keras_retinanet.utils.image import read_image_bgr, preprocess_image_caffe_fast, resize_image, compute_resize_scale
from keras_retinanet.keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.keras_retinanet.utils.gpu import setup_gpu

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


# tracking options (you shouldn't need to touch these)
KalmanConfig.DEFAULT_HISTORY_SPAN = [3, 5]
KalmanConfig.DEFAULT_CONFIDENCE_BOUNDS = [0.5, 0.7]
KalmanConfig.TRACKING_DELTA_THRES_MULT = 4
KalmanConfig.INITIAL_PROCESS_NOISE = 200
KalmanConfig.INITIAL_COVARIANCE = 500
KalmanConfig.INITIAL_MEASUREMENT_NOISE = 50
KalmanConfig.TIMESTEP = 1


def main(exporter=None):

    # make sure we use absolute path
    if not os.path.isabs(Params.VIDEO_FILE):
        Params.VIDEO_FILE = os.path.join(current_dir, Params.VIDEO_FILE)

    assert os.path.exists(Params.VIDEO_FILE), f"Input video file '{Params.VIDEO_FILE}' does not exist!"

    if Params.USE_GPU:
        # use gpu:0
        setup_gpu(0)
    else:
        setup_gpu("cpu")

    if Params.PROFILE:
        import cProfile, pstats
        pr = cProfile.Profile(builtins=False)

    # load the inference model with selected backbone
    model_path = os.path.join(current_dir, 'models', Params.MODEL)
    model = models.load_model(model_path, backbone_name=Params.BACKBONE)

    detection_exporter = exporter

    running_id = 1
    seen_idx = 0
    detections = []
    start_time = time.time()

    running_id = 1
    detections = []
    detection_disp_counter = 0

    old_bboxes, old_scores, old_labels = ([], [], [])
    bboxes, scores, labels = ([], [], [])

    width, height = vid.read_video_resolution(Params.VIDEO_FILE)
    frame_shape = (height, width, 3)
    scale = compute_resize_scale(frame_shape)
    resized_shape = (round(scale*height), round(scale*width), 3)
    imagenet_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

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
    
    CHUNK_SIZE = Params.DETECT_EVERY_NTH_FRAME
    disable_video = Params.OUTPUT_TYPE.lower() != valid_output_types[0]
    if Params.OUT_RESOLUTION is None:
        Params.OUT_RESOLUTION = in_res
    if detection_exporter is None:
        detection_exporter = DetectionExporter(out_path, real_fps, in_res, Params.OUT_RESOLUTION, 
                                               Params.DETECT_EVERY_NTH_FRAME, placeholder=Params.OUTPUT_TYPE == "video")
    with detection_exporter:
        with VideoWriter(out_path, Params.OUT_RESOLUTION, fps=fps, codec="mp4v", placeholder=disable_video) as writer:
            with VideoIterator(Params.VIDEO_FILE, max_slice=CHUNK_SIZE) as vi:
                print("* * * * *")
                print(f"Starting object detection from frame #{Params.FRAME_OFFSET}")
                print(f"Using inference model {Params.MODEL} ({Params.BACKBONE} backbone) for detection")
                print("Output type is", Params.OUTPUT_TYPE.upper())
                info_str = "all remaining frames..." if Params.PROCESS_NUM_FRAMES is None else f"{Params.PROCESS_NUM_FRAMES} frames in total..."
                if Params.PROFILE:
                    pr.enable()
                    print("Execution Profiler is turned ON")
                print("Processing", info_str)
                print("* * * * *")
                vi.seek(Params.FRAME_OFFSET)
                fps_timer = time.time()
                fps_counter = 0
                                
                preprocessed_float32 = np.empty(resized_shape, dtype=np.float32)
                gen = vi if Params.OUTPUT_TYPE == "video" else itertools.repeat(None)
                fps_counter = 0 if Params.OUTPUT_TYPE == "video" else -Params.DETECT_EVERY_NTH_FRAME
                for j, frame in enumerate(gen):

                    if Params.OUTPUT_TYPE != "video":
                        i = Params.FRAME_OFFSET + j * Params.DETECT_EVERY_NTH_FRAME
                        try:
                            frame = vi[i]
                        except IndexError:
                            break
                        fps_counter += Params.DETECT_EVERY_NTH_FRAME - 1
                    else:
                        i = j

                    if Params.PROCESS_NUM_FRAMES is not None and i >= Params.PROCESS_NUM_FRAMES + Params.FRAME_OFFSET:
                        break

                    if i % Params.DETECT_EVERY_NTH_FRAME == 0 or Params.OUTPUT_TYPE != "video":
                        
                        image = cv2.resize(frame, None, fx=scale, fy=scale)  
                        
                        # Subtract ImageNet mean (vectorized version of preprocess_image())
                        np.subtract(image, imagenet_mean, out=preprocessed_float32, casting="unsafe")

                        bboxes, scores, labels = model.predict_on_batch(
                            np.expand_dims(preprocessed_float32, axis=0))

                        bboxes /= scale

                        # extract valid detections
                        valid_detections = [(b, s, l) for b, s, l in list(zip(bboxes[0], scores[0], labels[0]))[
                            :Params.MAX_DETECTIONS_PER_FRAME] if s > Params.CONFIDENCE_THRES]
                        if valid_detections:
                            detection_disp_counter = 0
                            bboxes, scores, labels = list(zip(*valid_detections))
                            labels = [labels_to_names[l] for l in labels]
                            old_bboxes, old_scores, old_labels = bboxes, scores, labels
                        else:
                            bboxes, scores, labels = ([], [], [])
                        if Params.INTERPOLATE_BETWEEN_DETECTIONS:
                            new_detections = kt.get_detections_from_bboxes(
                                labels, bboxes, scores)
                            detections, running_id = kt.match_and_update_detections(
                                new_detections, detections, running_id)
                            if Params.OUTPUT_TYPE != "video":
                                unseen_detections = [d for d in detections if d.object_id > seen_idx and d.is_valid]
                                if unseen_detections:
                                    seen_idx = max(unseen_detections, key=lambda d: d.object_id).object_id
                                    detection_exporter.add_detections_at_frame(unseen_detections, i)
                            else:
                                frame = kt.visualize_detections(frame, detections, uncertain_color=None)
                            detection_exporter.update_timeseries(detections, i)
                        else:
                            if Params.OUTPUT_TYPE != "video":
                                new_detections = kt.get_detections_from_bboxes(
                                    labels, bboxes, scores)
                                detection_exporter.add_detections_at_frame(new_detections, i) # - 2*Params.DETECT_EVERY_NTH_FRAME)
                                detection_exporter.update_timeseries(new_detections, i)
                            else:
                                frame = kt.visualize_bboxes(
                                    frame, labels, bboxes, scores)

                        for label, score in zip(labels, scores):
                            print(f"[Frame {i + Params.FRAME_OFFSET}] Detected: {label} ({score:.3f})")

                    elif Params.OUTPUT_TYPE == "video":
                        if Params.INTERPOLATE_BETWEEN_DETECTIONS:
                            interp_detections = kt.interpolate_detections(
                                detections, (i % Params.DETECT_EVERY_NTH_FRAME) / Params.DETECT_EVERY_NTH_FRAME)
                            frame = kt.visualize_detections(
                                frame, interp_detections, uncertain_color=None)
                        elif valid_detections and detection_disp_counter < Params.SHOW_DETECTION_N_FRAMES:
                            frame = kt.visualize_bboxes(
                                frame, old_labels, old_bboxes, old_scores)
                            detection_disp_counter += 1

                    writer.write(frame, bgr_to_rgb=False)
                    fps_counter += 1
                    if fps_counter >= 100:
                        processing_fps = fps_counter / (time.time() - fps_timer)
                        print(f"Processed {i} frames ({(i)/(fps+1e-6):.1f} seconds of video), processing speed: {processing_fps:.1f} fps")
                        fps_counter = 0
                        fps_counter = 0
                        fps_timer = time.time()

    if Params.PROFILE:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.sort_stats(pstats.SortKey.CUMULATIVE)
        ps.reverse_order()
        ps.print_stats()

    if Params.COMPRESS_VIDEO and Params.OUTPUT_TYPE == "video":
        print("Compressing video...")
        new_output_path = vid.compress_video(out_path, create_copy=True)
        os.remove(out_path)
        out_path = new_output_path

    if Params.OUTPUT_TYPE == "video":
        print("Wrote output to:", out_path)

    print(f"Processing time: {time.time() - start_time:.3f} s")
    return detection_exporter if Params.OUTPUT_TYPE == "exporter" else None


if __name__ == '__main__':
    main()