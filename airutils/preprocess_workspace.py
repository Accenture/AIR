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

import os
import sys
import argparse
import subprocess as sp

# define absolute folder locations
current_dir = os.path.dirname(os.path.abspath(__file__))

# make detect.py available for python
sys.path.insert(0, os.path.join(current_dir, '..'))

from detect import main as detect, Params, parse_args
import video.vidtools as vid
import imtools as im


def main(args):
    detection_exporter = None
    Params.OUTPUT_TYPE = "exporter"
    Params.OUTPUT_PATH = None
    Params.FRAME_OFFSET = 0
    Params.PROCESS_NUM_FRAMES = None
    Params.INTERPOLATE_BETWEEN_DETECTIONS = False
    Params.DETECT_EVERY_NTH_FRAME = 60
    for f in sorted(os.listdir(args.input)):
        vid_file = os.path.join(args.input, f)
        if vid.is_supported_video_file(vid_file):
            try:
                vid.extract_subtitles(vid_file, verbose=False)
            except sp.CalledProcessError:
                print(f"'{vid_file}' does not contain subtitles track, skipping extraction...")
            if args.use_artificial_intelligence:
                Params.VIDEO_FILE = vid_file
                detection_exporter = detect(detection_exporter)
                num_frames = vid.read_video_total_frame_count_fast(vid_file)
                detection_exporter.frame_offset += num_frames
                if args.plot_timeseries:
                    timeseries_file = os.path.join(args.input, 
                        os.path.splitext(f)[0] + ".png")
                    im.plot_detection_timeseries(detection_exporter.detection_timeseries,
                        timeseries_file, num_frames=num_frames)
                detection_exporter.switch_video()
    if detection_exporter is not None and detection_exporter.detection_list:
        detection_exporter.save(os.path.join(args.input, 
            os.path.basename(args.input) + "_AI_HAVAINNOT.json"))
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Input video folder path")
    parser.add_argument('-ai', '--use_artificial_intelligence', action="store_true",
        help="Try to automatically detect lost people from the" 
        "video files in the given input folder and store" 
        "detections in a saved session file")
    parser.add_argument('-pl', '--plot_timeseries', action="store_true",
        help="Produce time series plot of detections")
    args = parse_args(parser)
    if not os.path.isabs(args.input):
        args.input = os.path.join(current_dir, args.input)
    assert os.path.exists(args.input), "Input path does not exist!"
    if os.path.isfile(args.input):
        args.input = os.path.dirname(os.path.abspath(args.input))
    main(args)