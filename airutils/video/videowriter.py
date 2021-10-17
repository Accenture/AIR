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
import cv2

class VideoWriter(object):
    ''' OpenCV VideoWriter wrapper '''

    def __init__(self, filepath, resolution, fps, codec="avc1", placeholder=False):
        self.resolution = resolution
        self.filepath = filepath
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.fps = fps
        self.writer = None
        # makes this object an empty shell (placeholder) 
        # to ensure syntactic correctness without the need
        # to modify the "with" block when conditionally disabling video writing
        self.placeholder = placeholder

    def __enter__(self):
        if not self.placeholder:
            self.init()
        return self

    def __exit__(self, type, value, traceback):
        if not self.placeholder:
            self.close()

    def init(self):
        os.makedirs(os.path.split(self.filepath)[0], exist_ok=True)
        self.writer = cv2.VideoWriter(
            self.filepath, self.fourcc, int(self.fps), self.resolution)

    def close(self):
        self.writer.release()

    def write(self, frame, bgr_to_rgb=False):
        if not self.placeholder:
            if bgr_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.writer.write(cv2.resize(frame, self.resolution, interpolation=cv2.INTER_CUBIC))