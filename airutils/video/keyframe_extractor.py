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
import json
import subprocess
from video_iterator import VideoIterator 


class KeyframeExtractor(object):
    ''' 
    Extracts keyframes of the given video 
    file and provides an iterator over them.
    '''

    def __init__(self, video_path, no_cache=False, return_delta_frames=False):
        self.file_path = video_path
        self.video_path = video_path
        self.no_cache = no_cache
        self.directory = os.path.dirname(video_path)
        self.video_name, self.extension = os.path.splitext(
            os.path.basename(video_path))
        self.iframes = []
        self.all_frames = None
        self.return_delta_frames = return_delta_frames
        self.frame_types = []
        self.frame_idx = 0
        self.iframe_folder = os.path.join(
            self.directory, ".iframes_" + self.video_name)
        self.cmd = f'ffmpeg -i {video_path} -q:v 2 -pred 1 -vf select=eq(pict_type\,PICT_TYPE_I) -vsync 0 {self.iframe_folder}/frame_%03d.jpg'.split()
        self.cmd_extract_all = f'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1 {video_path}'.split()
        if not return_delta_frames:
            self.__extract()
        else:
            self.__extract_all()
        print("Extraction complete!")

    @property
    def __frame_folder_present(self):
        return False if self.no_cache else os.path.isdir(self.iframe_folder) and os.listdir(self.iframe_folder)

    def __extract(self):
        print("Extracting key video frames...")
        if not self.__frame_folder_present:
            if not os.path.exists(self.iframe_folder):
                os.mkdir(self.iframe_folder)
            completed = subprocess.run(self.cmd)
            completed.check_returncode()
        frames_files = sorted(os.listdir(self.iframe_folder))
        for frame_file in frames_files:
            if frame_file.endswith(".jpg"):
                self.iframes.append(cv2.imread(
                    os.path.join(self.iframe_folder, frame_file)))

    def __extract_all(self):
        print("Extracting all video frames...")
        frame_type_file = os.path.join(self.directory, "." + self.video_name + "_frame_types.json")
        if not os.path.exists(frame_type_file) or self.no_cache:
            out = subprocess.check_output(self.cmd_extract_all).decode()
            self.frame_types = out.replace('pict_type=','').split()
            if not self.no_cache:
                with open(frame_type_file, "w") as f:
                    f.write(json.dumps(self.frame_types))
        else:
            print("Using cached frame type file...")
            with open(frame_type_file, "r") as f:
                self.frame_types = json.loads(f.read())
        self.all_frames = VideoIterator(self.video_path)
        self.all_frames.init()

    def __iter__(self):
        return self

    def __len__(self):
        if self.return_delta_frames:
            return len(self.all_frames)
        else:
            return len(self.iframes)

    def __next__(self):
        try:
            if self.return_delta_frames:
                frame_type = self.frame_types[self.frame_idx]
                frame = self.all_frames[self.frame_idx]
                self.frame_idx += 1
                return (frame_type, frame)
            else:
                frame = self.iframes[self.frame_idx]
                self.frame_idx += 1
                return frame
        except IndexError:
            raise StopIteration
    
    def __getitem__(self, idx):
        if self.return_delta_frames:
            # raise NotImplementedError("Full video extraction does not currently support indexing!")
            # if isinstance(idx, slice):
            #     self.video_client.consume(idx.start)
            #     return list(zip(*(self.frame_types[idx], self.video_client.consume(idx.stop - idx.start))))
            # return self.frame_types[idx], self.video_client.consume(1)
            if isinstance(idx, slice):
                return list(zip(*(self.frame_types[idx], self.all_frames[idx])))
            return (self.frame_types[idx], self.all_frames[idx])
        else:
            return self.iframes[idx]

    def seek(self, frame_no):
        if frame_no >= len(self):
            self.frame_idx = len(self) - 1
        elif frame_no < 0:
            self.frame_idx = 0
        else:
            self.frame_idx = frame_no

    @property
    def next_keyframe(self):
        if not self.return_delta_frames:
            return next(self)
        for i in range(self.frame_idx, len(self)):
            if self.frame_types[i] == "I":
                return i, self.all_frames[i]

    @property
    def delta_width(self):
        if not self.return_delta_frames:
            return 0
        else:
            j = 0
            key_frames = 0
            for i in range(len(self)):
                if self.frame_types[i] == "I":
                    key_frames += 1
                if key_frames == 2:
                    break
                if key_frames == 1:
                    j += 1
            return j

    def close(self):
        if self.all_frames is not None:
            self.all_frames.close()


if __name__ == "__main__":
    extractor = KeyframeExtractor("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/demo_video/DJI_demo.mov",
                                  no_cache=False, return_delta_frames=True)
    # cv2.namedWindow(f"iframe", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(f"iframe", 1280, 720)
    # for frame_type, iframe in extractor[:10]:
    #     cv2.imshow(f"frametype: {frame_type}", iframe)
    #     cv2.waitKey(30)
    extractor.seek(1000)
    for asd in extractor:
        print(asd)