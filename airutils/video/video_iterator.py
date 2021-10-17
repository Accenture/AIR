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

class VideoIterator(object):
    ''' 
    Provides simple memory efficient 
    interface for working with video frames. 

    ```yml
    Parameters:
        src: Video file path
        max_slice: maximum number of frames to keep in the memory, 
                   increasing this allows more memory usage but can 
                   be really slow

    Usage:
        with VideoIterator("my_video.mp4") as vi:
            first_ten_frames = vi[0:10:1]
    ```
    '''

    def __init__(self, src, max_slice=100, backend = cv2.CAP_ANY):
        self.video_capture = None
        self.src = src
        self.max_slice = max_slice
        self.__frame_idx = 0
        self.backend = backend

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, type, value, traceback):
        self.close()
    
    def init(self):
        self.video_capture = cv2.VideoCapture(self.src)
        self.video_capture.open(self.src, self.backend)

    def close(self):
        self.video_capture.release()
        self.video_capture = None

    def _read_next_frame(self):
        if self.video_capture is None:
            raise RuntimeError("You have to initialize this object first!")
        ret, frame = self.video_capture.read()
        if not ret or frame is None:
            raise IndexError("No frame received!")
        return frame

    def __iter__(self):
        return self

    def __len__(self):
        if self.video_capture is None:
            raise RuntimeError("You have to initialize this object first!")
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __next__(self):
        try:
            frames = self[self.__frame_idx]
            self.__frame_idx += 1
            return frames
        except IndexError:
            print("VideoIterator exhausted, stopping iteration...")
            raise StopIteration
    
    def __getitem__(self, idx):
        frames = []
        if isinstance(idx, slice):
            if idx.start != self.__frame_idx:
                self.seek(idx.start)
            start = 0 if idx.start is None else idx.start
            stop = len(self) - 2 if idx.stop is None else idx.stop
            step = 1 if idx.step is None else idx.step
            if ((stop - start) // step) > self.max_slice:
                print(f"RuntimeWarning: Configured max_slice ({self.max_slice}) exceeded! Reducing slice size to that...")
                start, stop, step = start, start + self.max_slice * step, step
            for i in range(start, stop, step):
                try:
                    if i != self.__frame_idx:
                        self.seek(i)
                    frames.append(self._read_next_frame())
                    self.__frame_idx += 1
                except IndexError:
                    print("VideoIterator has reached the end of the video!")
                    break
        else:
            if idx != self.__frame_idx:
                self._seek_no_check(idx)
            frames = self._read_next_frame()
            self.__frame_idx += 1
        return frames
    
    def seek(self, frame_no):
        if frame_no >= len(self):
            frame_no = len(self) - 2
        elif frame_no < 0:
            frame_no = 0
        return self._seek_no_check(frame_no)
    
    def _seek_no_check(self,frame_no):
        self.__frame_idx = frame_no
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        return frame_no


if __name__ == "__main__":
    with VideoIterator("/Users/pasi.pyrro/Downloads/2018_12_14_15_45_59.MP4") as vi:
        vi.seek(200)
        print(len(vi[10:100]))
        print(vi._VideoIterator__frame_idx)

    