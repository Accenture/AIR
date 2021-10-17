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

import cv2
import copy
import numpy as np
from PIL import ImageColor
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class KalmanConfig(object):
    DEFAULT_CONFIDENCE_BOUNDS = [0.4, 0.65]
    DEFAULT_HISTORY_SPAN = [3, 6]
    TRACKING_DELTA_THRES_MULT = 5
    TIMESTEP = 1
    INITIAL_COVARIANCE = 100
    INITIAL_MEASUREMENT_NOISE = 100
    INITIAL_PROCESS_NOISE = 400
    # DEFAULT_HIT_THRES = 2


class TrackerNotAssignedError(Exception):
    ''' Thrown when Detection class method 
        requiring a tracker is called without 
        initiating the tracker first '''


class Detection(object):
    ''' Represent trackable detection and it's meta information '''

    def __init__(self, object_class, confidence_bounds, history_span, center, 
                 speed=None, width=0, height=0, initial_confidence=float("NaN")):
        if speed is not None:
            self.initial_measurement = np.array(center + speed)
        else:
            self.initial_measurement = np.array(center)
        self.assigned_tracker = False
        self.tracker = None
        self.object_id = None
        self.object_class = object_class
        self.hitcount = 0
        self.width = width
        self.height = height
        self.initial_confidence = initial_confidence
        self.detection_history = [initial_confidence]
        self.position_history = [center]
        self.history_min_len, self.history_max_len = history_span
        self.confidence_bounds = confidence_bounds

    def __update_detection_history(self, detection_kind):
        self.detection_history.insert(0, detection_kind)
        if self.history_max_len < len(self.detection_history):
            self.detection_history.pop()

    def missed(self):
        self.__update_detection_history(0.0)

    def successfully_tracked(self, confidence=1.0):
        self.__update_detection_history(confidence)

    def update_position_history(self):
        self.position_history.insert(0, self.position)
        if len(self.position_history) > self.history_max_len:
            self.position_history.pop()

    @property
    def is_valid(self):
        # do we have enough history data to make the decision?
        if len(self.detection_history) < self.history_min_len:
            return False
        return self.confidence > self.confidence_bounds[1]

    @property
    def has_expired(self):
        # do we have enough history data to make the decision?
        if len(self.detection_history) < self.history_min_len:
            return False
        return self.confidence < self.confidence_bounds[0]

    @property
    def confidence(self):
        return np.mean(self.detection_history, dtype=float)

    @property
    def speed(self):
        if self.tracker is None:
            if self.initial_measurement.size == 4:
                return self.initial_measurement[2:4]
            return None
        else:
            return self.tracker.x[2:].flatten()

    @property
    def speed_variance(self):
        if self.tracker is None:
            raise TrackerNotAssignedError("Tracker not initialized!")
        else:
            return self.tracker.P.diagonal()[2:]

    @property
    def position(self):
        if self.tracker is None:
            return self.initial_measurement[:2]
        else:
            return self.tracker.x[:2].flatten()
        
    @position.setter
    def position(self, value):
        self.tracker.x[:2] = np.array(value).reshape(2, 1)

    @property
    def position_variance(self):
        if self.tracker is None:
            raise TrackerNotAssignedError("Tracker not initialized!")
        else:
            return self.tracker.P.diagonal()[:2]

    @property
    def current_state(self):
        if self.tracker is None:
            raise TrackerNotAssignedError("Tracker not initialized!")
        else:
            return self.tracker.x.flatten()
    
    @property
    def current_measurement(self):
        if self.tracker is None:
            return self.initial_measurement
        else:
            return self.tracker.z.flatten()


def interpolate_detections(detections, t, direction=None):
    interp_detections = copy.deepcopy(detections)
    for interp, detection in zip(interp_detections, detections):
        if detection.speed is not None:
            interp.position = t * detection.speed + detection.position
        elif len(detection.position_history) >= 2:
            interp.position = t * detection.position + (1 - t) * detection.position_history[0]
        elif direction is not None:
            interp.position = t * direction + detection.position
    return interp_detections


def visualize_bboxes(background_image, object_classes, bboxes, scores=None, speeds=None, 
                     confidence_bounds=KalmanConfig.DEFAULT_CONFIDENCE_BOUNDS, 
                     history_span=KalmanConfig.DEFAULT_HISTORY_SPAN,
                     valid_color="red", uncertain_color="blue", margin=40, 
                     line_width=8, fontsize=1.5):
    ''' Wrapper function for directly visualizing bounding boxes in format [xmin, ymin, xmax, ymax] '''
    detections = get_detections_from_bboxes(object_classes, bboxes, scores, speeds, confidence_bounds, history_span)
    return visualize_detections(background_image, detections, valid_color, uncertain_color, margin, line_width, fontsize)


def visualize_detections(background_image, detections, valid_color="red", uncertain_color="blue", margin=20, line_width=8, fontsize=1.5):
    alpha = 0.7
    y_offset = 10

    bg_img = background_image.copy()
    title_boxes = background_image.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    try:
        valid_color = ImageColor.getrgb(valid_color)[::-1]
        if uncertain_color is not None:
            uncertain_color = ImageColor.getrgb(uncertain_color)[::-1]
    except ValueError:
        valid_color = (0, 0, 255)
        if uncertain_color is not None:
            uncertain_color = (255, 0, 0)
    
    bbox_infos = []

    ymax, xmax = bg_img.shape[:2]
    for detection in detections:
        if (uncertain_color is None and detection.is_valid) or uncertain_color is not None:
            x, y = detection.position
            w = detection.width if detection.width > 0 else 50
            h = detection.height if detection.height > 0 else 50
            x, y = int(max(x - margin - w // 2, 0)), int(max(y - margin - h // 2, 0))
            w, h = int(min(w + 2 * margin, xmax)), int(min(h + 2 * margin, ymax))
            color = valid_color if detection.is_valid else uncertain_color
            if detection.object_id is not None:
                title = f"{detection.object_class} {detection.object_id} ({100.*detection.confidence:.1f} %)"
            elif not np.isnan(detection.initial_confidence): 
                title = f"{detection.object_class} ({100.*detection.confidence:.1f} %)"
            else:
                title = detection.object_class
            (title_w, title_h), baseline = cv2.getTextSize(
                title, font, fontsize, line_width)
            if y > (title_h + 2*y_offset):
                text_pos = (x - title_w // 2 + w // 2, y - title_h - y_offset)
            else:
                text_pos = (x - title_w // 2 + w // 2, y + h + title_h + baseline + y_offset)
            cv2.rectangle(title_boxes, (text_pos[0] - baseline, text_pos[1] - title_h - baseline),
                        (text_pos[0] + title_w + baseline // 2, text_pos[1] + title_h - y_offset), (0, 0, 0), -1)
            bbox_infos.append((x, y, w, h, color, title, title_w, title_h, baseline, text_pos))
        else:
            bbox_infos.append(None)

    if detections:
        cv2.addWeighted(title_boxes, alpha, bg_img, 1 - alpha, 0, bg_img)

    for detection, bbox_info in zip(detections, bbox_infos):
        if (uncertain_color is None and detection.is_valid) or uncertain_color is not None:
            x, y, w, h, color, title, title_w, title_h, baseline, text_pos = bbox_info
            cv2.rectangle(title_boxes, (text_pos[0] - baseline, text_pos[1] - title_h - baseline),
                        (text_pos[0] + title_w, text_pos[1] + title_h - y_offset), (0, 0, 0), -1)
            cv2.putText(bg_img, title, text_pos, font,
                        fontsize, (255,)*3, line_width, cv2.LINE_8)
            cv2.rectangle(bg_img, (x, y), (x + w, y + h), color, line_width)
    
    return bg_img

def get_detections_from_bboxes(object_classes, bboxes, scores=None, speeds=None, 
                               confidence_bounds=KalmanConfig.DEFAULT_CONFIDENCE_BOUNDS, 
                               history_span=KalmanConfig.DEFAULT_HISTORY_SPAN):
    '''
    Assumes bboxes in format [xmin, ymin, xmax, ymax]
    '''
    detections = []
    if speeds is None:
        speeds = [None] * len(bboxes)
    if scores is None:
        scores = [float("NaN")] * len(bboxes)
    if isinstance(object_classes, str):
        object_classes = [object_classes] * len(bboxes)
    for object_class, bbox, score, speed in zip(object_classes, bboxes, scores, speeds):
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        width = max(bbox[2] - bbox[0], 0)
        height = max(bbox[3] - bbox[1], 0)
        detections.append(Detection(object_class, confidence_bounds, history_span, np.array(center), 
            speed=speed, width=width, height=height, initial_confidence=score))
    return detections


def get_detections_from_centers(object_classes, centers, speeds=None, 
                                confidence_bounds=KalmanConfig.DEFAULT_CONFIDENCE_BOUNDS, 
                                history_span=KalmanConfig.DEFAULT_HISTORY_SPAN):
    detections = []
    if speeds is None:
        speeds = [None] * len(centers)
    if isinstance(object_classes, str):
        object_classes = [object_classes] * len(centers)
    for object_class, center, speed in zip(object_classes, centers, speeds):
        detections.append(Detection(object_class, confidence_bounds, history_span, np.array(center), speed=speed))
    return detections


def get_detections_from_contours(object_classes, contours, speeds=None, 
                                 confidence_bounds=KalmanConfig.DEFAULT_CONFIDENCE_BOUNDS, 
                                 history_span=KalmanConfig.DEFAULT_HISTORY_SPAN):
    detections = []
    if speeds is None:
        speeds = [None] * len(contours)
    if isinstance(object_classes, str):
        object_classes = [object_classes] * len(contours)
    for object_class, blob, speed in zip(object_classes, contours, speeds):
        (x, y), r = cv2.minEnclosingCircle(blob)
        detections.append(Detection(object_class, confidence_bounds, history_span, np.array([x, y]), speed=speed, width=2*r, height=2*r))
    return detections


def initiate_tracker(detection, location, velocity=None):
    dim_x = 6 if velocity is not None else 4 # number of state variables
    dim_z = 4 if velocity is not None else 2 # number of observed variables
    dt = KalmanConfig.TIMESTEP # time step
    f = KalmanFilter(dim_x, dim_z)

    # x : ndarray (dim_x, 1)
    # initial value for the state (position and velocity).
    # f.x = np.array([[0.],  # position - x
    #                 [0.],  # position - y
    #                 [0.],  # velocity - x
    #                 [0.],  # velocity - y
    #                 [0.],  # acceleration - x
    #                 [0.]]) # acceleration - y
    f.x = np.zeros([dim_x, 1])
    f.x[0:2] = np.array([location]).T
    if velocity is not None:
        f.x[2:4] = np.array(velocity).T

    # F : ndarray (dim_x, dim_x)
    # state transition matrix:
    if velocity is not None:
        f.F = np.array([[1.,        0.,        dt,        0., 0.5*dt*dt,        0.],
                        [0.,        1.,        0.,        dt,        0., 0.5*dt*dt],
                        [0.,        0.,        1.,        0.,        dt,        0.],
                        [0.,        0.,        0.,        1.,        0.,        dt],
                        [0.,        0.,        0.,        0.,        1.,        0.],
                        [0.,        0.,        0.,        0.,        0.,        1.]])
    else:
        f.F = np.array([[1., 0., dt, 0.],
                        [0., 1., 0., dt],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])


    # H : ndarray (dim_z, dim_x)
    # measurement function:
    if velocity is not None:
        f.H = np.array([[1., 0., 0., 0., 0., 0.],   # measuring location - x
                        [0., 1., 0., 0., 0., 0.],   # measuring location - y
                        [0., 0., 1., 0., 0., 0.],   # measuring velocity - x
                        [0., 0., 0., 1., 0., 0.]])  # measuring velocity - y
    else:
        f.H = np.array([[1., 0., 0., 0.],           # measuring location - x
                        [0., 1., 0., 0.]])          # measuring location - y


    # P : ndarray (dim_x, dim_x)
    # initial covariance matrix:
    f.P = float(KalmanConfig.INITIAL_COVARIANCE) * np.eye(dim_x)

    # R : ndarray (dim_z, dim_z), default eye(dim_z)
    # measurement noise:
    # f.R = 1.0 * np.ones([dim_z, dim_z])
    f.R = float(KalmanConfig.INITIAL_MEASUREMENT_NOISE) * np.eye(dim_z)

    # Q : ndarray (dim_x, dim_x), default eye(dim_x)
    # process noise:
    # f.Q = Q_discrete_white_noise(dim=4, dt=0.05, var=0.05)
    noise_dim = 3 if velocity is not None else 2
    f.Q = Q_discrete_white_noise(dim=noise_dim, dt=dt, var=float(KalmanConfig.INITIAL_PROCESS_NOISE), block_size=2)

    # B : ndarray (dim_x, dim_u), default 0
    # control transition matrix

    # instantiate tracker component in detection class
    # all property methods are now available
    detection.tracker = f

    return detection


def match_and_update_detections(new_detections, tracked_detections, running_id):
    updated_detections = []
    for old_detection in tracked_detections:
        old_detection.tracker.predict()
        center_predict = old_detection.tracker.x[:2].flatten()
        covariance = old_detection.tracker.P
        threshold_dist = KalmanConfig.TRACKING_DELTA_THRES_MULT * \
            np.sqrt(np.sum(np.diag(covariance)))
        candid_old_detections = []
        candid_old_distances = []
        for detection in new_detections:
            center_measured = detection.position
            dist = np.linalg.norm(center_measured - center_predict)
            if dist <= threshold_dist and \
                    detection.object_class == old_detection.object_class:
                candid_old_detections.append(detection)
                candid_old_distances.append(dist)
        
        # Did old tracker got updated?
        if candid_old_detections: 
            d_idx_select = np.argmin(candid_old_distances)
            winner_detection = candid_old_detections[d_idx_select]
            winner_detection.assigned_tracker = True
            old_detection.tracker.update(winner_detection.current_measurement)
            # Increase HIT count for the object (measure of confidence)
            # if not old_detection.is_valid:
                # old_detection.hitcount += 1
            old_detection.successfully_tracked(winner_detection.initial_confidence)
        else:
            # Decrease HIT count for the object (measure of confidence)
            # old_detection.hitcount -= 1
            old_detection.missed()

        old_detection.update_position_history()

        if not old_detection.has_expired:
            updated_detections.append(old_detection)

    # Deal with remaining unassigned detections
    for detection in new_detections:
        if not detection.assigned_tracker:
            detection.object_id = running_id
            updated_detections.append(initiate_tracker(
                detection, detection.position,
                detection.speed))
            running_id += 1

    return updated_detections, running_id
