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
Optical Flow based algorithm to detect moving objects (people) from drone video.
Assumes constant movement of the drone, video brightness consistency, 
object movement magnitude at some reasonable range and movement direction not parallel to the drone movement.
Threshold parameter might need to be adjusted for good detection performance in different videos.

Example usage:
python detect_moving_objects.py --input path/to/video.mp4

Author: Pasi Pyrrö
Date: 13.1.2020
'''

import os
import sys
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

# make utils folder available for python
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'video'))

from video.keyframe_extractor import KeyframeExtractor
from video.videowriter import VideoWriter
import video.vidtools as vid
import imtools as im
import kalman_tracker as kt


class Params(object):
    # default input parameters
    DATA_DIR = os.path.join(os.path.dirname(current_dir), "..", "data", "videos")
    VIDEO_FILE = None

    # default output parameters
    OUT_RESOLUTION = (3840, 2024)
    PLOT_INTERMEDIATE_RESULTS = True
    PROCESS_FULL_VIDEO = True
    PROCESS_NUM_FRAMES = 1500
    FRAME_OFFSET = 0
    QUIVER_STEP = 60
    OUTPUT_PATH = None

    # default model parameters
    SPEED_DIFF_THRES = 0.075
    MORPH_FILTER_SIZE = 15
    ALPHA = 0.5


def put_title(img, title, color="white"):
    return im.overlay_text(img, title, (40, 100), color=color)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', default=Params.VIDEO_FILE, help="Input video path")
    parser.add_argument(
        '-o', '--output', default=Params.OUTPUT_PATH, help="Output video path")
    parser.add_argument('-r', '--resolution', type=int,
                        nargs=2, help="Output video resolution")
    parser.add_argument('-p', '--plot', type=bool, default=Params.PLOT_INTERMEDIATE_RESULTS,
                        help="Determines whether to plot intermediate results alongside output video")
    parser.add_argument('-f', '--full', type=bool, default=Params.PROCESS_FULL_VIDEO,
                        help="Determines whether to process all the frames of the video or only keyframes (full processing is much slower).")
    parser.add_argument('-n', '--num-frames', type=int,
                        default=Params.PROCESS_NUM_FRAMES, help="How many frames to process?")
    parser.add_argument('-fo', '--frame-offset', type=int,
                        default=Params.FRAME_OFFSET, help="Where to start processing?")
    parser.add_argument('-t', '--speed-diff-thres', type=float, default=Params.SPEED_DIFF_THRES,
                        help="ADVANCED: Final threshold for movement classification.")
    parser.add_argument('-a', '--alpha', type=float, default=Params.ALPHA,
                        help="ADVANCED: Defines the hue + saliency blend ratio.")
    parser.add_argument('-s', '--filter-size', type=int, default=Params.MORPH_FILTER_SIZE,
                        help="ADVANCED: Defines size of the morphological filter kernel. Needs to be an odd positive integer!")
    args = parser.parse_args()
    if args.resolution:
        Params.OUT_RESOLUTION = tuple(args.resolution)
    Params.VIDEO_FILE = args.input
    Params.OUTPUT_PATH = args.output
    Params.PLOT_INTERMEDIATE_RESULTS = args.plot
    Params.PROCESS_FULL_VIDEO = args.full
    Params.PROCESS_NUM_FRAMES = args.num_frames
    Params.FRAME_OFFSET = args.frame_offset
    Params.SPEED_DIFF_THRES = args.speed_diff_thres
    Params.MORPH_FILTER_SIZE = args.filter_size
    Params.ALPHA = args.alpha


if __name__ == '__main__':
    parse_args()
    if not Params.OUTPUT_PATH:
        vid_file_name, vid_file_ext = os.path.splitext(
            os.path.basename(Params.VIDEO_FILE))
        out_video_file = os.path.join(
            Params.DATA_DIR, vid_file_name + "_out" + vid_file_ext)
    else:
        out_video_file = Params.OUTPUT_PATH
    running_id = 1
    delta_idx = 0
    detections = []
    start_time = time.time()
    prev_field = None
    global_displacement = None
    frame_type = "I"
    key_frames = KeyframeExtractor(
        Params.VIDEO_FILE, return_delta_frames=Params.PROCESS_FULL_VIDEO)
    delta_width = key_frames.delta_width
    if not Params.PROCESS_FULL_VIDEO:
        prev_frame = key_frames[max(0, Params.FRAME_OFFSET)]
        # slice video frames
        key_frames = key_frames[Params.FRAME_OFFSET + 1:]
    else:
        key_frames.seek(Params.FRAME_OFFSET)
        i, prev_frame = key_frames.next_keyframe
        key_frames.seek(i)
    fps = 2 if not Params.PROCESS_FULL_VIDEO else int(
        vid.read_video_fps(Params.VIDEO_FILE))
    saliency_filter = cv2.saliency.StaticSaliencyFineGrained_create()
    prev_saliency = (saliency_filter.computeSaliency(
        prev_frame)[1] * 255).astype("uint8")

    # define plot images
    composite_plot, motion_field_plot, thres_plot, bboxes_plot = None, None, None, None

    with VideoWriter(out_video_file, Params.OUT_RESOLUTION, fps) as writer:
        for i, frame in enumerate(key_frames):
            if Params.PROCESS_FULL_VIDEO:
                frame_type, frame = frame
            if frame is None or i > Params.PROCESS_NUM_FRAMES:
                break
            # process keyframe
            if not Params.PROCESS_FULL_VIDEO or frame_type == "I":
                delta_idx = 0
                if Params.PROCESS_FULL_VIDEO:
                    try:
                        _, frame = key_frames.next_keyframe
                    except TypeError:
                        continue
                saliency = (saliency_filter.computeSaliency(
                    frame)[1] * 255).astype("uint8")

                # transform frames to HSV color space and pick the hue channel
                im_from_hue = im.color2gray(prev_frame, mode="hue")
                im_to_hue = im.color2gray(frame, mode="hue")

                # combine saliency map and hue channel for both frames
                composite_prev = ((1 - Params.ALPHA) * prev_saliency +
                                  Params.ALPHA * im_from_hue).astype("uint8")
                composite = ((1 - Params.ALPHA) * saliency + Params.ALPHA *
                             im_to_hue).astype("uint8")

                motion_field, global_displacement = im.get_motion_field_between_two_frames(
                    prev_frame, frame, composite_prev, composite, prev_field)

                motion_field_img = im.get_flow_map(
                    motion_field, None, Params.QUIVER_STEP, scale=1, display=False, color="magenta")

                ydim, xdim = motion_field.shape[:2]
                mid_p = (xdim // 2, ydim // 2)
                dx, dy = global_displacement
                # draw general drone speed vector on top of the vector field
                motion_field_img = cv2.arrowedLine(
                    motion_field_img, mid_p, (mid_p[0] + int(3 * dx), mid_p[1] + int(3 * dy)), (0, 255, 0), 10)
                speed_diff = im.get_motion_mask(
                    motion_field, global_displacement)

                # do thresholding and morphological filtering
                thres_map = speed_diff
                mask = im.filter_space_morph_seq(
                    (thres_map > Params.SPEED_DIFF_THRES).astype("uint8") * 255, Params.MORPH_FILTER_SIZE)
                mask = cv2.dilate(
                    mask, (2 * Params.MORPH_FILTER_SIZE, 2 * Params.MORPH_FILTER_SIZE), iterations=3)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                mean_speeds = im.get_contour_mean_speeds(
                    contours, motion_field)
                new_detections = kt.get_detections_from_contours(
                    "Person", contours, mean_speeds)
                detections, running_id = kt.match_and_update_detections(
                    new_detections, detections, running_id)
                bboxes = kt.visualize_detections(prev_frame, detections)

                # prepare frames for plotting
                if Params.PLOT_INTERMEDIATE_RESULTS:
                    mask_plot = put_title(
                        np.stack((mask,) * 3, axis=-1), "Output Mask")
                    bboxes_plot = put_title(bboxes, "Detections", color="blue")
                    thres_plot = put_title(
                        (thres_map * 255).astype("uint8"), "Score Map")
                    motion_field_plot = put_title(
                        motion_field_img, "Motion Field")
                    composite_plot = put_title(
                        composite_prev, "Saliency & Hue Filter")

                    # write output frame
                    out_frame = im.plot_tiled_imgs(
                        [composite_plot, motion_field_plot,
                            thres_plot, bboxes_plot],
                        display=False, title="Saliency, Thres Map,\nSpeed Diff, Mask", figsize=(8, 4))
                else:
                    out_frame = bboxes
                writer.write(out_frame, bgr_to_rgb=False)

                prev_field = motion_field.copy()
                prev_frame = frame.copy()
                prev_saliency = saliency.copy()
                print("Processed key frame", i)

            # process delta frame
            else:
                delta_idx += 1
                # interpolate detections if any
                interp_detections = kt.interpolate_detections(
                    detections, delta_idx / delta_width, global_displacement)
                # prepare frames for plotting
                bboxes_plot = put_title(kt.visualize_detections(
                    frame, interp_detections), "Detections", color="blue")
                if Params.PLOT_INTERMEDIATE_RESULTS and motion_field_plot is not None:
                    # write output frame
                    out_frame = im.plot_tiled_imgs(
                        [composite_plot, motion_field_plot,
                            thres_plot, bboxes_plot],
                        display=False, title="Saliency, Thres Map,\nSpeed Diff, Mask", figsize=(8, 4))
                else:
                    out_frame = kt.visualize_detections(
                        frame, interp_detections)
                writer.write(out_frame, bgr_to_rgb=False)

    print("Wrote output to:", out_video_file)
    print(f"Processing time: {time.time() - start_time:.3f} s")
    if not isinstance(key_frames, list):
        key_frames.close()
