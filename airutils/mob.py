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
Tools for post-processing bounding box detections 
from typical object detectors, implements the
MOB (Merging of Overlapping Boxes) algorithm.
See details below.

Date: 16.10.2020
Author: Pasi Pyrrö
'''

import numpy as np

from sklearn.cluster import AgglomerativeClustering
try:
    from compute_overlap import compute_overlap # needs C binary to work
except ImportError:
    raise ImportError("Please compile compute_overlap binary before using "
                      "this module by running 'python setup.py build_ext --inplace'")


''' AABB (axis aligned bounding box) operations '''

def aabb_dims(aabbs, dim):
    return aabbs[:, dim+2] - aabbs[:, dim]


def aabb_centroids(aabbs):
    x_means = (aabbs[:,2] + aabbs[:,0]) / 2
    y_means = (aabbs[:,3] + aabbs[:,1]) / 2
    return np.concatenate((x_means[:, None], y_means[:, None]), axis=1)


def max_aabb_area(aabbs):
    x_dims = aabb_dims(aabbs, 0)
    y_dims = aabb_dims(aabbs, 1)
    return np.max(x_dims, 0) * np.max(y_dims, 0)


def compute_aabb_area(aabb):
    return abs(aabb[2] - aabb[0]) * abs(aabb[3] - aabb[1])


def get_aabb_max_axis(aabb):
    ''' split heuristic to obtain good subdivision in minimal steps '''
    return np.argmax([abs(aabb[2] - aabb[0]), abs(aabb[3] - aabb[1])])


def enclosing_aabb(aabbs, scores=None):
    return np.array([
        np.min(aabbs[:,0]),
        np.min(aabbs[:,1]),
        np.max(aabbs[:,2]),
        np.max(aabbs[:,3]),
    ])


def mean_aabb(aabbs, scores=None):
    return np.mean(aabbs, axis=0)


def median_aabb(aabbs, scores=None):
    return np.median(aabbs, axis=0)


def score_vote_aabb(aabbs, scores):
    normalized_scores = scores / np.sum(scores)
    return np.average(aabbs, axis=0, weights=normalized_scores)


def spatial_vote_aabb(aabbs, scores=None):
    def _gaussian_weights(points):
        if len(points) == 1:
            return np.array([1.])
        mu = np.mean(points)
        sigma = np.std(points)
        log_weights = np.abs(-(points-mu)**2 / (2 * sigma**2))
        weights = np.exp(log_weights - log_weights.max())
        return weights / np.sum(weights)

    centroids = aabb_centroids(aabbs)
    x_weights = _gaussian_weights(centroids[:, 0])
    y_weights = _gaussian_weights(centroids[:, 1])

    return np.array([
        np.average(aabbs[:,0], weights=x_weights),
        np.average(aabbs[:,1], weights=y_weights),
        np.average(aabbs[:,2], weights=x_weights),
        np.average(aabbs[:,3], weights=y_weights),
    ])


def mixture_vote_aabb(aabbs, scores):
    score_box = score_vote_aabb(aabbs, scores)
    spatial_box = spatial_vote_aabb(aabbs, scores)
    mixture_box = np.vstack([score_box, spatial_box])
    return np.mean(mixture_box, axis=0)


def score_median_vote_aabb(aabbs, scores):
    score_box = score_vote_aabb(aabbs, scores)
    median_box = median_aabb(aabbs, scores)
    mixture_box = np.vstack([score_box, median_box])
    return np.mean(mixture_box, axis=0)


select_merge = {
    "enclose" : enclosing_aabb,
    "mean" : mean_aabb,
    "median" : median_aabb,
    "score_vote" : score_vote_aabb,
    "spatial_vote" : spatial_vote_aabb,
    "mixture_vote" : mixture_vote_aabb,
    "score_median_vote" : score_median_vote_aabb
}


def subdivide_bboxes(max_area, all_bboxes, all_scores=None, merge_mode="enclose"):
    ''' Splits box cluster recursively into halves until the areas of
        enclosing boxes of each cluster do not exceed ``max_area`` '''

    ret_boxes = []
    ret_scores = []

    def _subdivide(bboxes, scores, sort_axis):
        if bboxes.shape[0] > 1: # stop recursion if there's nothing to split
            half = len(bboxes) // 2
            centroids = aabb_centroids(bboxes)
            sorted_indices = np.argsort(centroids[:, sort_axis % 2])
            sub_bboxes = bboxes[sorted_indices, :]
            if scores is not None:
                sub_scores = scores[sorted_indices]
                left_score_cluster = sub_scores[:half]
                right_score_cluster = sub_scores[half:]
            else:
                left_score_cluster = None
                right_score_cluster = None
            left_cluster = sub_bboxes[:half]
            right_cluster = sub_bboxes[half:]
            left_enc_box = select_merge[merge_mode](left_cluster, left_score_cluster)
            right_enc_box = select_merge[merge_mode](right_cluster, right_score_cluster)
            if compute_aabb_area(left_enc_box) > max_area:
                _subdivide(left_cluster, left_score_cluster, get_aabb_max_axis(left_enc_box))
            else:
                ret_boxes.append(left_enc_box[None, :])
                if scores is not None:
                    ret_scores.append(np.mean(left_score_cluster[None, :], axis=1))

            if compute_aabb_area(right_enc_box) > max_area:
                _subdivide(right_cluster, right_score_cluster, get_aabb_max_axis(right_enc_box))
            else:
                ret_boxes.append(right_enc_box[None, :])
                if scores is not None:
                    ret_scores.append(np.mean(right_score_cluster[None, :], axis=1))
        else:
            ret_boxes.append(bboxes)
            if scores is not None:
                ret_scores.append(scores)

    split_axis = get_aabb_max_axis(select_merge[merge_mode](all_bboxes, all_scores))
    _subdivide(all_bboxes, all_scores, split_axis)

    if len(ret_boxes) > 0:
        if None in ret_scores:
            return np.concatenate(ret_boxes), None
        return np.concatenate(ret_boxes), np.concatenate(ret_scores)
    return all_bboxes, all_scores


def merge_overlapping_boxes(boxes, scores=None, iou_threshold=0, max_iterations=2, 
                            top_k=50, max_inflation_factor=100, merge_mode="enclose"):
    ''' 
    Implements the MOB-algorithm (Merging of Overlapping Bounding boxes)

    Merges bounding boxes that are very similar in size and location.
    Works in similar way as NMS but instead of bbox elimination merges
    everything into one enclosing bbox. If ``scores`` is provided, calculates new
    scores for merged detections based on the mean of the merged detections'
    scores. Number of ``max_iterations`` specifies how many times to do the merging, 
    usually the default value 2 merges everything (e.g. small disjoint boxes left
    inside merged boxes).

    ### IMPORTANT! 
    To use this function you must first run
    ```
    python setup.py build_ext --inplace
    ```
    in this directory and install ``sklearn`` module.
    
    The ``ìou_threshold`` parameter controls the minimum required overlap
    between two boxes to merge them:
    - 1 means no merging happens
    - 0 means merge if there's any overlap at all
    - any negative value means merge all boxes regardless of overlap

    The ``top_k``parameter gives an option for eliminating some of the lowest scoring bboxes:
    - 1 means this algorithm is the same as regular NMS
    - 0 (or negative number) means keep all boxes for merging
    - any positive integer means that many top scoring boxes are kept at maximum for merging

    The ``max_inflation_factor`` is the maximum relative size of a merged box compared to the 
    largest input bounding box size. If a larger merged box were to be created, it is subdivided
    until the maximum size of any merged box does not exceed the size limit posed by this parameter.
    - Value larger than 1 is the intended usage, good values would be something above 2
    - Values less than 2 usually results in weird merging of the boxes
    - Value 0 results in no merging happening at all (given ``top_k`` is zero)
    
    The ``merge_mode`` parameter defines how merging of overlapping bounding box clusters happens.
    It can be one of the following options:
    - "enclose" (default) mode draws bounding box around all boxes in the cluster, which "encloses" them
    - "mean" mode merges the boxes by taking the mean of them
    - "median" mode merges the oxes by taking the median of them
    - "score_vote" mode takes the weighted average of the boxes based on their relative scores
    - "spatial_vote" mode assumes the boxes were sampled from a gaussian distribution centered 
        at cluster mean and weights them according to their density while taking the average
    - "mixture_vote" is the mean of "score_vote" and "spatial_vote", which takes into account both 
        score based and location based votes.
 
    NOTE: Assumes all boxes are from the same class!

    Designed to be used with keras-retinanet predictions, the converted
    model should have NMS disabled for this to be useful.

    Returns: merged_boxes (np.ndarray), merged_scores (np.ndarray or None)
    '''

    assert isinstance(boxes, np.ndarray) and \
        len(boxes.shape) == 2 and boxes.shape[1] == 4, \
            "Input bounding boxes need to be in a numpy ndarray of shape (N, 4)"

    assert merge_mode in select_merge, f"Invalid merge mode '{merge_mode}'"

    assert merge_mode not in {"score_vote", "mixture_vote"} or scores is not None, \
        f"Cannot use merge mode '{merge_mode}' without specifying scores parameter"

    merged_boxes = boxes
    merged_scores = scores
    
    # do we have anything to merge?
    if boxes.shape[0] > 1:

        # optionally calculate max merged box area from max_inflation_factor
        if max_inflation_factor is not None:
            max_merged_area = max_inflation_factor * max_aabb_area(boxes)

        # performs recursive merging of bounding boxes
        # based on jaccard distance metric
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="single",
            distance_threshold=1. - iou_threshold
        )

        # Many max_iterations ensures that everything gets merged
        # at the cost of slightly longer computation time
        for j in range(max_iterations):

            # compute jaccard distance matrix for detected boxes
            # clustered boxes should have small distances
            X = 1. - compute_overlap(boxes, boxes)
            
            # divide boxes into merge groups
            clustering.fit(X)

            merged_boxes = np.empty((clustering.n_clusters_, 4))
            if scores is not None:
                merged_scores = np.empty(clustering.n_clusters_)

            offset = 0

            for i in range(clustering.n_clusters_):
                cluster_indices = np.squeeze(np.argwhere(clustering.labels_ == i))
                if len(cluster_indices.shape) == 0:
                    cluster_indices = np.array([cluster_indices])
                cluster = boxes[cluster_indices, :]
                score_cluster = None
                if scores is not None:
                    score_cluster = scores[cluster_indices]
                    if top_k > 0: # pick top-k scoring boxes for merging only
                        score_indices = np.argsort(score_cluster)[::-1]
                        score_cluster = score_cluster[score_indices][:top_k]
                        cluster = cluster[score_indices][:top_k]
                    merged_scores[i + offset] = np.mean(score_cluster)

                merged_bbox = select_merge[merge_mode](cluster, score_cluster)
                enc_area = compute_aabb_area(merged_bbox)
        
                # do we have an upper bound for merged box area?
                if max_inflation_factor is not None and enc_area > max_merged_area:
                    enc_scores = score_cluster if scores is not None else None
                    split_bboxes, split_scores = subdivide_bboxes(max_merged_area, cluster, 
                                                                  enc_scores, merge_mode)
                    split_len = len(split_bboxes)
                    new_len = len(merged_boxes) + split_len - 1
                    merged_boxes.resize((new_len, 4), refcheck=False)
                    merged_boxes[i + offset:i + offset + split_len, :] = split_bboxes
                    if scores is not None:
                        merged_scores.resize(new_len, refcheck=False)
                        merged_scores[i + offset:i + offset + split_len] = split_scores
                    offset += split_len - 1
                else:
                    merged_boxes[i + offset, :] = merged_bbox

            if j < max_iterations - 1:
                if merged_boxes.shape[0] <= 1:
                    # no boxes left to merge, we stop iteration here
                    break
                # update boxes (and scores) for the next iteration
                boxes = merged_boxes.copy()
                if scores is not None:
                    scores = merged_scores.copy()

    if scores is not None:
        return merged_boxes, merged_scores
    return merged_boxes, None


def merge_boxes_per_label(boxes, scores, labels, iou_threshold=0, max_iterations=2, 
                          top_k=-1, max_inflation_factor=100, merge_mode="enclose"):
    ''' 
    Same as ``merge_overlapping_boxes`` but accepts ``labels`` argument.
    Overlapping boxes from two different classes will not be merged.
    '''

    assert isinstance(boxes, np.ndarray) and \
        len(boxes.shape) == 2 and boxes.shape[1] == 4, \
            "Input bounding boxes need to be in a numpy ndarray of shape (N, 4)"

    if boxes.shape[0] <= 1:
        # nothing to merge here
        return boxes, scores, labels
    
    if labels is None or len(labels) == 0:
        merged_boxes, merged_scores = merge_overlapping_boxes(boxes, scores, 
            iou_threshold, max_iterations, top_k, max_inflation_factor, merge_mode)
        return merged_boxes, merged_scores, labels

    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for l in np.unique(labels):
        label_indices = np.squeeze(np.argwhere(labels == l))
        if len(label_indices.shape) == 0:
            label_indices = np.array([label_indices])
        lboxes = boxes[label_indices, :]
        lscores = scores[label_indices] if scores is not None else None
        mboxes, mscores = merge_overlapping_boxes(lboxes, lscores, 
            iou_threshold, max_iterations, top_k, max_inflation_factor, merge_mode)
        mlabels = np.ones(len(mboxes), dtype=int) * l
        merged_boxes.append(mboxes)
        merged_scores.append(mscores)
        merged_labels.append(mlabels)

    merged_boxes = np.concatenate(merged_boxes, axis=0)
    merged_labels = np.concatenate(merged_labels, axis=0)
    if scores is None:
        return merged_boxes, None, merged_labels
    merged_scores = np.concatenate(merged_scores, axis=0)
    return merged_boxes, merged_scores, merged_labels


if __name__ == "__main__":
    ''' test code goes here '''

    from time import perf_counter
    import cv2

    def draw_box(image, box, color, thickness=2):
        """ Draws a box on an image with a given color.

        # Arguments
            image     : The image to draw on.
            box       : A list of 4 elements (x1, y1, x2, y2).
            color     : The color of the box.
            thickness : The thickness of the lines to draw a box with.
        """
        b = np.clip(np.array(box), 0, None).astype(np.uint32)
        return cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


    def draw_boxes(image, boxes, color, thickness=2):
        """ Draws boxes on an image with a given color.

        # Arguments
            image     : The image to draw on.
            boxes     : A [N, 4] matrix (x1, y1, x2, y2).
            color     : The color of the boxes.
            thickness : The thickness of the lines to draw boxes with.
        """
        image = np.array(image).astype(np.uint8)
        for b in boxes:
            image = draw_box(image, b, color, thickness=thickness)
        return image

    N_LABELS = 1

    np.random.seed(1234)

    bbox0 = np.array([10,20,110,120])
    bbox1 = np.array([100,200,200,300])
    bbox2 = np.array([1900,2900,2000,3000])
    bbox3 = np.array([1950,2950,2050,3050])

    # bboxes0 = bbox0.astype(float)
    bboxes0 = np.tile(bbox0, (101, 1)).astype(float) + np.random.normal(scale=3, size=(101,4))    
    for i in range(len(bboxes0)):
        # if i < 40 or i > 50:
        bboxes0[i, :] += np.array([10, 10, 10, 10]) * i * 2

    bboxes1 = np.tile(bbox1, (100, 1)).astype(float) + np.random.normal(scale=10, size=(100,4))
    bboxes2 = np.tile(bbox2, (500, 1)).astype(float) + np.random.normal(scale=10, size=(500,4))
    bboxes3 = np.tile(bbox3, (200, 1)).astype(float) + np.random.normal(scale=10, size=(200,4))
    boxes = np.concatenate((bboxes0, bboxes1, bboxes2, bboxes3), axis=0) #[:, [0,2,1,3]]

    enc_box = enclosing_aabb(boxes)
    img_shape = np.array([enc_box[3] - enc_box[1] + 100, enc_box[2] - enc_box[0] + 100, 3], dtype=int)
    img_before = np.zeros(img_shape, dtype=np.uint8)
    img_before = draw_boxes(img_before, boxes, (255,0,0))
    cv2.imwrite("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/misc/MOB_test_before.jpg", img_before)
    
    scores = np.random.rand(len(boxes))

    labels = np.random.randint(0, N_LABELS, len(boxes))

    merged_labels = []

    t1 = perf_counter()
    merged_boxes, merged_scores, merged_labels = merge_boxes_per_label(boxes, scores, labels, max_inflation_factor=None, top_k=-1, merge_mode="score_vote") #, iou_threshold=0.4) # max_iterations=1, top_k=10, max_inflation_factor=50)
    # merged_boxes, merged_scores = merge_overlapping_boxes(boxes, scores, iou_threshold=0, max_iterations=2, top_k=10, max_inflation_factor=None)
    # merged_boxes = merge_overlapping_boxes(boxes, None, iou_threshold=0, max_iterations=1, top_k=10, max_inflation_factor=2)

    # merged_boxes, merged_scores, merged_labels = merge_boxes_per_label(*[np.array([])]*3)
    # merged_boxes, merged_scores = merge_close_detections(np.array([]), np.array([]), iou_threshold=-100.001)

    img_after = np.zeros(img_shape, dtype=np.uint8)
    img_after = draw_boxes(img_after, merged_boxes, (0,255,0))

    
    cv2.imwrite("/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/misc/MOB_test_after.jpg", img_after)

    print("Time elapsed", perf_counter()-t1)
    print("==boxes==")
    print(merged_boxes)
    print("==box-areas==")
    print([int(compute_aabb_area(b)) for b in merged_boxes])
    print("==scores==")
    print(merged_scores)
    print("==labels==")
    print(merged_labels)
    print("==merged-length==")
    print(len(merged_boxes))