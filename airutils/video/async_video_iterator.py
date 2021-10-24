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

import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir))

import cv2
import time
from threading import Thread, RLock
from queue import Queue, Full, Empty

from video_iterator import VideoIterator


class AsyncVideoIterator(VideoIterator):
    ''' 
    Provides simple, memory efficient and asynchronous
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

    def __init__(self, src, start_idx=0, end_idx=-1, skip_rate=1, max_size=100, timeout=None, backend = cv2.CAP_ANY):
        super().__init__(src, max_size, backend)
        self.start_idx = max(0, start_idx)
        self.end_idx = end_idx
        # user options
        self.skip_rate = max(1, skip_rate)
        self.timeout = timeout
        self.skip_rate = skip_rate
        # internal variables
        self._lock = RLock()
        self._buffer = Queue(maxsize = max_size)
        self._producer = None
        self._running = False

        
    def __enter__(self):
        super().init()
        if not isinstance(self.end_idx, int) or self.end_idx < 0:
            self.end_idx = super().__len__() - 1
        self.start_stream()
        return self


    def __exit__(self, type, value, traceback):
        self.stop_stream()
        super().close()

    
    def start_stream(self):
        if self._running:
            self.stop_stream()
        with self._buffer.mutex:
            self._buffer.queue.clear()
        with self._lock:
            print(f"AsyncVideoIterator: Consuming video source '{self.src}'")
            self._running = True
            self._producer = Thread(target = self.produce)
            self._producer.daemon = True
            self._producer.start()

    
    def stop_stream(self):
        with self._lock:
            self._running = False
        self._producer.join()


    def produce(self):
        for i in range(self.start_idx, self.end_idx, self.skip_rate):
            with self._lock:
                frame = super().__getitem__(i)
            put_success = False
            while not put_success:
                if not self._running:
                    return
                try:
                    self._buffer.put_nowait(frame)
                except Full:
                    pass
                else:
                    put_success = True


    def __iter__(self):
        return self


    def __len__(self):
        return (self.end_idx - self.start_idx) // self.skip_rate


    def __next__(self):
        if self._running:
            try:
                frame = self._buffer.get(timeout = self.timeout)
                return frame
            except Empty:
                print("AsyncVideoIterator: Video buffer is empty!")
                raise StopIteration
        else:
            raise StopIteration

            
    def __getitem__(self, idx):
        raise NotImplementedError("Use the synchronous VideoIterator class to access '__getitem__' functionality")


    def seek(self, frame_no):
        raise NotImplementedError("Use the synchronous VideoIterator class to access 'seek' functionality")


if __name__ == "__main__":
    with AsyncVideoIterator("/Users/pasi.pyrro/Documents/repos/computer_vision/air/data/videos/Ylojarvi-gridiajo-two-guys-moving.mov", 
            max_size=100, skip_rate=30) as avi:
        print(len(avi))
        for frame in avi:
            cv2.namedWindow("AsyncVideoIterator", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("AsyncVideoIterator", 1280, 720)
            cv2.imshow("AsyncVideoIterator", frame)
            cv2.waitKey(1)
            time.sleep(0.5)
            print(avi[0])
            

    