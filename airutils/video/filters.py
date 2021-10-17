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

import cv2
import numpy as np
import functools

def crop_frame_square(frame):
    if type(frame) is np.ndarray and len(frame.shape) == 3:
        height, width, _ = frame.shape
        mid_x = width // 2
        mid_y = height // 2
        frame = frame[:,(mid_x-mid_y):(mid_x+mid_y),:]
    # frame = np.flip(frame, 0)
    return frame

def half_resolution(frame):
    if type(frame) is np.ndarray and len(frame.shape) == 3:
        return frame[::2,::2,:]
    else:
        return frame

def resize_frame(frame, resolution):
    if type(resolution) not in (list, tuple) or len(resolution) != 2:
        raise TypeError("resolution should be a list of two integers")
    if type(frame) is np.ndarray and len(frame.shape) == 3:
        return cv2.resize(frame, tuple(resolution), interpolation = cv2.INTER_CUBIC)
    else:
        return frame

def get_filter_with_args(ft, **kwargs):
    return functools.partial(ft, **kwargs)