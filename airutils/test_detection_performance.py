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
Test object detection algorithm performance on a video file and
tagging tool groundtruth tags. Computes accuracy, recall and running time of the algorithm
with given parameter ranges and plots the results.

Example usage:
python test_detection_performance.py

Author: Pasi Pyrrö
Date: 17.3.2020
'''
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# define absolute folder locations
current_dir = os.path.dirname(os.path.abspath(__file__))

from detect_people_retinanet import main as detect, Params, parse_args
from kalman_tracker import KalmanConfig
import video.vidtools as vid

# algorithm parameters
TEST_TYPE = "speed"
BATCH_SIZE_RANGE = [1, 32]
CONFIDENCE_THRESHOLD_RANGE = [0.8, 0.8, 0]
DETECT_EVERY_NTH_FRAME_RANGE = [15, 150, 15]
KALMAN_CONFIDENCE_LOWER_BOUND_RANGE = [0.5, 0.6, 0.5]
KALMAN_CONFIDENCE_UPPER_BOUND_RANGE = [0.7, 0.8, 0.5]

Params.INTERPOLATE_BETWEEN_DETECTIONS = True
Params.MODEL = "resnet50_coco_60_inference.h5"
Params.BACKBONE = "resnet50"

def calc_range_size(start, stop, step, include_end=True):
    return (stop - start) // step + (1 if include_end else 0)

def main(args):
    EPS = 1e-6
    Params.OUTPUT_TYPE = "json"

    DETECT_EVERY_NTH_FRAME_RANGE[1] += 1
    CONFIDENCE_THRESHOLD_RANGE[1] += EPS
    KALMAN_CONFIDENCE_LOWER_BOUND_RANGE[1] += EPS
    KALMAN_CONFIDENCE_UPPER_BOUND_RANGE[1] += EPS

    k = calc_range_size(*KALMAN_CONFIDENCE_LOWER_BOUND_RANGE)
    assert k == calc_range_size(*KALMAN_CONFIDENCE_UPPER_BOUND_RANGE), \
        "Kalman filter confidence bound ranges must have equal size!"

    total_frames = vid.read_video_total_frame_count_fast(args.input)

    if TEST_TYPE == "speed":
        batch_plots = []
        bsize = BATCH_SIZE_RANGE[0]
        rate_size = calc_range_size(*DETECT_EVERY_NTH_FRAME_RANGE)
        i = 0
        fig = plt.figure(figsize=(11, 7))
        try:
            while bsize <= BATCH_SIZE_RANGE[1]:
                batch_plots.append((bsize, np.empty([rate_size, 2])))
                for j, rate in enumerate(range(*DETECT_EVERY_NTH_FRAME_RANGE)):
                    Params.BATCH_SIZE = bsize
                    Params.DETECT_EVERY_NTH_FRAME = rate
                    start_time = time.time()
                    detect()
                    duration = time.time() - start_time
                    mean_fps = total_frames / duration
                    batch_plots[i][1][j, :] = np.array([rate, mean_fps])
                bsize <<= 1 # fancy and fast way to multiply by two
                i += 1
        finally:
            for bsize, b_plot in batch_plots:
                plt.plot(b_plot[:, 0], b_plot[:, 1], "x-", label=f"batch size {bsize}")
            plt.xlabel("Interval between processed frames (#frames)")
            plt.ylabel("Average FPS")
            plt.grid(axis="y")
            plt.legend()
            plt.title(f"Performance test on {os.path.basename(args.input)}")
            output_path = os.path.join(current_dir, "figures", f"performance_stats_{int(time.time())}.png")
            plt.savefig(output_path, bbox_inches="tight")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Input video path")
    parser.add_argument('-a', '--anns', help="Groundtruth annotations path", default=None)

    args = parse_args(parser)
    if not os.path.isabs(args.input):
        args.input = os.path.join(current_dir, args.input)
    assert os.path.exists(args.input), "Input video path does not exist!"
    if args.anns:
        if not os.path.isabs(args.anns):
            args.anns = os.path.join(current_dir, args.anns)
        assert os.path.exists(args.anns), "Input annotations path does not exist!"
    # if os.path.isfile(args.input):
    #     args.input = os.path.dirname(os.path.abspath(args.input))
    main(args)