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

import xml.etree.ElementTree as ET
import numpy as np
import sys
import os
import cv2

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))

import imtools

OVERLAP = 50       # how many pixels overlap between tiles?
SIZE_THRES = 0.9   # how much can we truncate objects at the tile border?

orig_voc_path = "/Users/pasi.pyrro/Documents/video_analytics/rescue-drone-tool/data/datasets/heridal_keras_retinanet_voc"
tiled_voc_path = orig_voc_path + "_tiled"

ann_path = os.path.join(orig_voc_path, "Annotations")
img_path = os.path.join(orig_voc_path, "JPEGImages")
img_set_path = os.path.join(orig_voc_path, "ImageSets", "Main")

new_ann_path = os.path.join(tiled_voc_path, "Annotations")
new_img_path = os.path.join(tiled_voc_path, "JPEGImages")
new_img_set_path = os.path.join(tiled_voc_path, "ImageSets", "Main")

os.makedirs(new_ann_path, exist_ok=True)
os.makedirs(new_img_path, exist_ok=True)
os.makedirs(new_img_set_path, exist_ok=True)

with open(os.path.join(img_set_path, "train.txt")) as f:
    train_ann_files = set([l.strip() + ".xml" for l in f.readlines()])

with open(os.path.join(img_set_path, "val.txt")) as f:
    val_ann_files = set([l.strip() + ".xml" for l in f.readlines()])

with open(os.path.join(img_set_path, "test.txt")) as f:
    test_ann_files = set([l.strip() + ".xml" for l in f.readlines()])

train_list = []
val_list = []
test_list = []

for ann_file in sorted(os.listdir(ann_path)):
    ann_file_path = os.path.join(ann_path, ann_file)
    img_file_path = os.path.join(img_path, ann_file.replace("xml", "JPG"))
    if not os.path.exists(img_file_path):
        print("No image file found for annotations in", ann_file)
    elif not ann_file.startswith("."):
        im = cv2.imread(img_file_path)
        tiles, offsets = imtools.create_overlapping_tiled_mosaic(im, overlap=OVERLAP, no_fill=True)
        ann_tree = ET.parse(ann_file_path)
        ann_tree_root = ann_tree.getroot()
        objects = ann_tree_root.findall("object")
        for i, (tile, offset_a) in enumerate(zip(tiles, offsets)):
            ann_basename = ann_file[:-4] + f"_{i+1}"
            root_xml_str = (
                "<annotation>"
                    "<folder>Annotations</folder>"
                    "<filename>{}</filename>"
                    "<source>"
                        "<database>HERIDAL database</database>"
                    "</source>"
                    "<size>"
                        "<height>{}</height>"
                        "<width>{}</width>"
                        "<depth>{}</depth>"
                    "</size>"
                "</annotation>"
            ).format(ann_basename, *tile.shape)
            new_xml_root = ET.fromstring(root_xml_str)
            tile_path = os.path.join(new_img_path, ann_file.replace(".xml", f"_{i+1}.jpg"))
            cv2.imwrite(tile_path, tile)
            for obj in objects:
                bbox_elem = obj.find("bndbox")
                bbox = np.array([[
                        int(bbox_elem.find("ymin").text),
                        int(bbox_elem.find("xmin").text),
                    ], [
                        int(bbox_elem.find("ymax").text),
                        int(bbox_elem.find("xmax").text),
                    ]
                ])
                centroid = np.mean(bbox, axis=0)
                offset_b = offset_a + np.array(tile.shape[:2])
                if np.alltrue(offset_a <= centroid) and np.alltrue(centroid <= offset_b):
                    obj_xml_str = (
                        "<object>"
                            "<name>person</name>"
                            "<pose>unspecified</pose>"
                            "<truncated>0</truncated>"
                            "<difficult>0</difficult>"
                            "<bndbox>"
                                "<xmin>{xmin}</xmin>"
                                "<xmax>{xmax}</xmax>"
                                "<ymin>{ymin}</ymin>"
                                "<ymax>{ymax}</ymax>"
                            "</bndbox>"
                        "</object>")
                    orig_area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
                    # translate bounding box
                    bbox -= offset_a
                    # clip negative coords to zero
                    bbox[bbox < 0] = 0
                    # clip overly large coords within tile bounds
                    if bbox[1,0] >= tile.shape[0]:
                        bbox[1,0] = tile.shape[0] - 1
                    if bbox[1,1] >= tile.shape[1]:
                        bbox[1,1] = tile.shape[1] - 1
                    truncated_area = (bbox[1,0] - bbox[0,0]) * (bbox[1,1] - bbox[0,1])
                    # only include bbox if it hasn't been truncated to less than 75% 
                    # of the original size
                    if truncated_area > SIZE_THRES * orig_area:
                        transformed_obj = ET.fromstring(
                            obj_xml_str.format(xmin=bbox[0,1], xmax=bbox[1,1],
                                            ymin=bbox[0,0], ymax=bbox[1,0]))
                        new_xml_root.append(transformed_obj)

            tile_ann_path = os.path.join(new_ann_path, ann_file.replace(".xml", f"_{i+1}.xml")) 
            ET.ElementTree(new_xml_root).write(tile_ann_path, encoding="utf-8", xml_declaration=True)

            if ann_file in train_ann_files:
                train_list.append(ann_basename + "\n")
            elif ann_file in val_ann_files:
                val_list.append(ann_basename + "\n")
            else:
                test_list.append(ann_basename + "\n")

        print("Processed", os.path.basename(img_file_path), f"into {len(tiles)} tiles")
        if len(tiles) != 4 or len(offsets) != 4:
            print("Warning! Invalid split for", os.path.basename(img_file_path))
        # break # TODO: remove me

with open(os.path.join(new_img_set_path, "train.txt"), "w") as f:
    f.writelines(train_list)

with open(os.path.join(new_img_set_path, "trainval.txt"), "w") as f:
    f.writelines(train_list + val_list)

with open(os.path.join(new_img_set_path, "val.txt"), "w") as f:
    f.writelines(val_list)

with open(os.path.join(new_img_set_path, "test.txt"), "w") as f:
    f.writelines(test_list)
