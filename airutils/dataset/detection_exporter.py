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
Author: Pasi Pyrrö
Date: 7.2.2020
'''

import json
import math
import numpy as np

LANG = "en"

DEFAULT_LABELS = {
    "person" : "Person" if LANG == "en" else "Henkilö"
}

VIDAR_CANVAS_WIDTH = 4096

class DetectionExporter(object):
    '''
    Module for converting detections into tagging tool displayable format
    '''

    def __init__(self, output_path, fps, in_resolution, out_resolution, inverse_sampling_rate, frame_offset=0,
                 map_labels=DEFAULT_LABELS, margin=20, placeholder=False):
        self.output_path = output_path
        self.frame_offset = frame_offset
        self.fps = fps
        self.detection_list = []
        self.map_labels = map_labels
        self.running_id = 1
        self.margin = margin
        vidar_scale_factor = VIDAR_CANVAS_WIDTH / in_resolution[0]
        self.width_coeff = vidar_scale_factor * out_resolution[0] / in_resolution[0]
        self.height_coeff = vidar_scale_factor * out_resolution[1] / in_resolution[1]
        self.placeholder = placeholder
        self.inverse_sampling_rate = inverse_sampling_rate
        self._detection_timeseries = [[[0,0]]]
        self._video_idx = 0
        self._num_detections = 0


    @property
    def detection_timeseries(self):
        # TODO: don't scale x-axis (irrelevant in ViDAR plots)
        return np.array(self._detection_timeseries[self._video_idx]) / self._num_detections


    def __enter__(self):
        return self


    def __exit__(self, type, value, traceback):
        if not self.placeholder:
            self.save(self.output_path)


    def update_timeseries(self, detections, frame_no, frame_offset=0):
        sum_confidence = np.sum([d.confidence if not math.isnan(d.confidence)
                                else d.detection_history[-1] 
                                for d in detections])
        self._num_detections += len(detections)
        current_timestamp = (frame_no + frame_offset) / self.fps 
        last_time_point = self._detection_timeseries[self._video_idx][-1][0] \
            if self._detection_timeseries[self._video_idx] else 0
        if math.isclose(last_time_point, current_timestamp):
            # increment existing point in time series
            self._detection_timeseries[self._video_idx][-1][1] += sum_confidence
        else:
            # add new detection to the time series
            self._detection_timeseries[self._video_idx].append([current_timestamp, sum_confidence])


    def add_detections_at_frame(self, detections, frame_no):
        if not isinstance(detections, list):
            detections = [detections]
        for d in detections:
            w_half = self.width_coeff * d.width / 2 + self.margin
            h_half = self.height_coeff * d.height / 2 + self.margin
            if LANG == "en":
                note = f"AI generated observation.\nConfidence: {100. * d.confidence:.1f} %\n"
            else:
                note = f"Tekoälyn generoima havainto.\nHavainnon varmuus: {100. * d.confidence:.1f} %\n"
            self.detection_list.append({
                "a": {
                    "x": self.width_coeff * d.position[0] - w_half,
                    "y": self.height_coeff * d.position[1] - h_half
                },
                "b": {
                    "x": self.width_coeff * d.position[0] + w_half,
                    "y": self.height_coeff * d.position[1] + h_half
                },
                "label": self.map_labels.get(d.object_class, d.object_class),
                "id": self.running_id,
                "timestamp": 1000. * (self.frame_offset + frame_no) / self.fps,
                "note" : note
            })
            self.running_id += 1


    def switch_video(self):
        self._video_idx += 1
        self._detection_timeseries.append([[0,0]])


    def save(self, output_path=None):
        if not output_path:
            output_path = self.output_path
        if output_path:
            with open(output_path, "w") as f:
                f.write(json.dumps(self.detection_list, ensure_ascii=False))
            print("Exported detections to", output_path)
        