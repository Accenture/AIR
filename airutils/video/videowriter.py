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
import video.vidtools as vid


class AsyncVideoWriter(VideoWriter):
    
    def __init__(self, output_path, resolution, fps, codec="avc1", compress=True, placeholder=False, bgr_to_rgb=False):
        super().__init__(output_path, resolution, fps, codec, compress, placeholder, bgr_to_rgb)
        self._lock = RLock()
        self._buffer = Queue(maxsize = max_size)
        self._writer = None
        self._running = False


    def __enter__(self):
        super().init()
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
            print(f"AsyncWriter: writing to sink '{self.output_path}'")
            self._running = True
            self._writer = Thread(target = self._write_async)
            self._writer.daemon = False
            self._writer.start()

    
    def stop_stream(self):
        with self._lock:
            self._running = False
        self._writer.join()
    

    def write(self):
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

        with self._lock:
            self._running = False


    def _write_async(self):
        while self._running:
            get_success = False
            while not get_success:
                if not self._running and self._buffer.empty():
                    return
                try:
                    frame = self._buffer.get_nowait()
                    super().write(frame)
                except Empty:
                    pass
                else:
                    get_success = True



class VideoWriter(object):
    ''' OpenCV VideoWriter wrapper '''

    def __init__(self, output_path, resolution, fps, codec="avc1", compress=True, placeholder=False, bgr_to_rgb=False):
        self.write_mode = "images" if os.path.isdir(output_path) else "video"
        self.resolution = resolution
        self.output_path = output_path
        self.compress = compress
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.fps = fps
        self.writer = None
        # makes this object an empty shell (placeholder) 
        # to ensure syntactic correctness without the need
        # to modify the "with" block when conditionally disabling video writing
        self.placeholder = placeholder
        self.bgr_to_rgb = bgr_to_rgb
        self.frames_written = 0

    def __enter__(self):
        if not self.placeholder:
            print(f"VideoWriter: writing output {self.write_mode} to {self.filepath}")
            self.init()
        return self

    def __exit__(self, type, value, traceback):
        if not self.placeholder:
            self.close()

    def init(self):
        os.makedirs(os.path.split(self.filepath)[0], exist_ok=True)
        self.writer = cv2.VideoWriter(
            self.output_path, self.fourcc, int(self.fps), self.resolution)

    def close(self):
        self.writer.release()
        if self.compress and self.write_mode == "video":
            print("Compressing video...")
            new_output_path = vid.compress_video(self.output_path, create_copy=True)
            os.remove(self.filepath)
            self.filepath = new_output_path
        print("Wrote output to:", self.output_path)

    def write(self, frame):
        if not self.placeholder:
            if self.write_mode == "video":
                if self.bgr_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.writer.write(cv2.resize(frame, self.resolution, interpolation=cv2.INTER_CUBIC))
            else:
                cv2.imwrite(os.path.join(save_path, f'frame-{self.frames_written + 1}.png'), save_image)
            self.frames_written += 1
