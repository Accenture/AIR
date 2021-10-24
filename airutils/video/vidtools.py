"""
Copyright 2018-2021 Accenture

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
import re
import cv2
import sys
import subprocess

supported_video_formats = ("mp4", "mov", "avi")


def is_supported_video_file(path):
    return path.lower().endswith(supported_video_formats)


def read_video_fps(video_file):
    fps = None
    try:
        video = cv2.VideoCapture(video_file)
        fps = video.get(cv2.CAP_PROP_FPS)
    finally:
        video.release()
    return fps


def read_video_resolution(video_file):
    res = None
    try:
        video = cv2.VideoCapture(video_file)
        res = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        video.release()
    return res


def read_video_total_frame_count_fast(video_file):
    count = None
    try:
        video = cv2.VideoCapture(video_file)
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        video.release()
    return count
    

def read_video_total_frame_count_accurate(video_file):
    ''' slow but accurate (should return exact count) 
        version of the above function '''
    count = 0
    try:
        video = cv2.VideoCapture(video_file)
        while True:
            grabbed, _ = video.read()
            if not grabbed:
                break
            count += 1
    finally:
        video.release()
    return count


def compress_video(video_path, target_kbps=1200, res_downscale_factor=None, create_copy=False):
    ''' compresses video to smaller filesize, requires ffmpeg installed '''
    if create_copy:
        output_path = [os.path.join(b, n + "_compressed." + e) for b, f in [os.path.split(video_path)] for n, e in [f.split(".")]][0]
    else:
        output_path = video_path
    if res_downscale_factor is None:
        cmd = f"ffmpeg -threads 4 -i {video_path} -y -acodec copy -b:v {int(target_kbps*10e3)} {output_path}".split()
    else:
        f = res_downscale_factor
        cmd = f"ffmpeg -threads 4 -i {video_path} -y -vf \"scale=iw/{f}:ih/{f}\" -acodec copy -b:v {int(target_kbps*10e3)} {output_path}".split()
    completed = subprocess.run(cmd)
    completed.check_returncode()
    return output_path


def extract_subtitles(video_path, output_path=None, verbose=True):
    ''' extracts subtitles track from a video file into srt file, 
        requires ffmpeg installed'''
    folder, video_file = os.path.split(video_path)
    temp = os.getcwd()
    if output_path is None:
        os.chdir(folder)
        out_file = re.sub(r"\.[a-zA-Z0-9]{3,4}$", ".srt", video_file)

    cmd = f"ffmpeg -y -i {video_file} {out_file}"
    if not verbose:
        with open(os.devnull, "w") as f:
            completed = subprocess.run(cmd, shell=True, stdout=f, stderr=f)
    else:
        completed = subprocess.run(cmd, shell=True)
    completed.check_returncode()
    os.chdir(temp)
    return os.path.join(folder, out_file)