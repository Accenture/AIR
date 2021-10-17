'''
Handles parsing PASCAL VOC dataset annotations and computes statistics 
about the dataset folder, such as its hash sum, which can be used to verify
the dataset version later.

Modifications copyright (c) 2021 Accenture
'''

import os
import sys

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import keras_retinanet.utils
    __package__ = "keras_retinanet.utils"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "airutils"))
import imtools

import hashlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import copy

from . import transform as tr
from . import image as im
from . import visualization as vis

def compute_dir_metadata(path, human_readable=True):
    ''' Credits for hashing go to: https://stackoverflow.com/a/49701019 
        
        Computes metadata from directory
        returns dir_name, sha1_hash, num_files, dir_size
    '''
    if not os.path.isdir(path):
        raise TypeError(f"'{path}' is not a directory!")
    
    dir_name = os.path.basename(path)
    digest = hashlib.sha1()
    num_files = 0
    dir_size = 0

    hash_files = []

    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            if os.path.isfile(file_path) and not name.startswith("."):
                hash_files.append((name, file_path))
                num_files += 1
                dir_size += os.path.getsize(file_path)
    
    # make sure we always hash in the same order as it affects the overall hash!
    for name, file_path in sorted(hash_files, key=lambda f: f[0]):
        # hash the file name
        digest.update(name.encode())

        # Hash the path and add to the digest to account for empty files/directories
        # This is no good as the path might change on every training run!
        # digest.update(hashlib.sha1(file_path[len(path):].encode()).digest())

        # hash file contents
        with open(file_path, 'rb') as f_obj:
            while True:
                buf = f_obj.read(1024 * 1024)
                if not buf:
                    break
                digest.update(buf)

    if human_readable:
        dir_size = f"{dir_size/1e6:.1f}M"
    else:
        dir_size = f"{dir_size}B"

    return dir_name, digest.hexdigest(), num_files, dir_size


def compute_dataset_metadata(path, human_readable=True):
    """ Wraps dir metadata in a dictionary that can be directly fed to wandb """
    dir_name, sha1_hash, num_files, dir_size = compute_dir_metadata(path, human_readable)
    return {
        "dataset_name" : dir_name,
        "dataset_checksum" : sha1_hash,
        "dataset_num_files" : num_files,
        "dataset_size" : dir_size
    }


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise ValueError('illegal value for \'{}\': {}'.format(debug_name, e))
    return result
    

def parse_voc_annotations(filename):
    def parse_annotation(element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name != "person":
            raise ValueError('class name \'{}\' not understood')

        box = np.zeros((4,))
        label = 14

        bndbox    = _findNode(element, 'bndbox')
        box[0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float)
        box[1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float)
        box[2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float)
        box[3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float)

        return truncated, difficult, box, label

    xml_root = ET.parse(filename).getroot()
    """ Parse all annotations under the xml_root.
    """
    num_objects = len(xml_root.findall('object'))
    annotations = {'labels': np.empty((num_objects,)), 'bboxes': np.empty((num_objects, 4))}
    for i, element in enumerate(xml_root.iter('object')):
        try:
            _, _, box, label = parse_annotation(element)
        except ValueError as e:
            raise ValueError('could not parse object #{}: {}'.format(i, e))

        annotations['bboxes'][i, :] = box
        annotations['labels'][i] = label

    return annotations


if __name__ == "__main__":
    ''' misc plotting code, can be safely ignored '''

    import time

    def augment_image(image, annotations, transform, visual_effect):
        params = im.TransformParameters()
        image = image.copy()
        annotations = copy.deepcopy(annotations)
        # do transformation
        transform = im.adjust_transform_for_image(transform, image, params.relative_translation)
        image = im.apply_transform(transform, image, params)
        for index in range(annotations['bboxes'].shape[0]):
            annotations['bboxes'][index, :] = tr.clip_aabb(image.shape, tr.transform_aabb(transform, annotations['bboxes'][index, :]))
        # do visual effect
        image = visual_effect(image)
        return image, annotations


    def show_image(img, title = '', bgr_to_rgb=True):
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # fig = plt.gcf()
        # fig.set_size_inches(*size)
        plt.clf()
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        # plt.pause(0.001)
        plt.show()
        a = time.time()

    # dataset_path = os.environ["HOME"] + "/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc_tiled"
    # print(compute_dataset_metadata(dataset_path))

    full_image_file = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc/JPEGImages/train_ZRI_3035.jpg"
    full_ann_file = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc/annotations/train_ZRI_3035.xml"

    full_image = im.read_image_bgr(full_image_file)
    full_annotations = parse_voc_annotations(full_ann_file)
    full_image = vis.draw_boxes(full_image, full_annotations["bboxes"], (255, 0, 0), thickness=4)
    full_image = vis.draw_boxes(full_image, np.array([
        [0, 1500, 2025, 3000],
        [2000, 0, 4000, 1525],
        [2000, 1500, 4000, 3000]
    ]), color=(0, 0, 255), thickness=4)
    full_image = vis.draw_box(full_image, [0, 0, 2025, 1525],
                              color=(0, 255, 0), thickness=4)

    image_file = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc_tiled/JPEGImages/train_ZRI_3035_1.jpg"
    ann_file = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc_tiled/annotations/train_ZRI_3035_1.xml"

    image = im.read_image_bgr(image_file)
    annotations = parse_voc_annotations(ann_file)
    bbox_image = vis.draw_boxes(image, annotations["bboxes"], (255, 0, 0), thickness=4)

    transform = tr.rotation(0.5)
    visual_effect = im.VisualEffect()

    transformed_image, transformed_annotations = augment_image(image, annotations, transform, visual_effect)
    transformed_image = vis.draw_boxes(transformed_image, transformed_annotations["bboxes"], (255, 0, 0), thickness=4)

    visual_effect = im.VisualEffect(equalize_chance=1)

    augmented_image, augmented_annotations = augment_image(image, annotations, transform, visual_effect)
    augmented_image = vis.draw_boxes(augmented_image, augmented_annotations["bboxes"], (255, 0, 0), thickness=4)

    print(f"Execution took {time.time()-a:.3f} s")

    fig = imtools.create_grid_plot([full_image, bbox_image, transformed_image, augmented_image], [
        "Input image tiling",
        "Tile cropping and bounding\nbox transformation",    
        "Geometric transformation to image\nand bounding boxes",
        "Color operation to image"
    ])
    fig.savefig("/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/misc/testfig.png", bbox_inches="tight", dpi=100)

    # show_image(image, "rotated & added noise")