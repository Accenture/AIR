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

from bs4 import BeautifulSoup as bs
from collections import defaultdict
import os
import re

script_dir = os.path.dirname(os.path.realpath(__file__))

region_regex = re.compile(r"\_\w{2,3}\_", flags=re.IGNORECASE)

heridal_path = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal"
retinanet_heridal_path = "/Users/pasi.pyrro/Documents/repos/computer_vision/masters-thesis/data/datasets/heridal_keras_retinanet_voc"

def count_objects_in_xml(xml_path):
    objects = 0
    with open(xml_path, "r") as f:
        html = f.read()
        soup = bs(html, features="html.parser")
        objects += len(soup.find_all("object"))
    return objects


''' verify original heridal '''
print("Verifying original heridal dataset...\n")

train_ann_path = os.path.join(heridal_path, "trainImages", "labels")
train_img_path = os.path.join(heridal_path, "trainImages")
test_ann_path = os.path.join(heridal_path, "testImages", "labels")
test_img_path = os.path.join(heridal_path, "testImages")

''' train images '''
train_loc_images = defaultdict(int)
train_loc_objects = defaultdict(int)
train_objects = 0
train_files = 0
ann_files = [x for x in sorted(os.listdir(train_ann_path)) if ".xml" in x]
for ann_file in ann_files:
    region = region_regex.search(ann_file).group(0).strip("_")
    ann_file_path = os.path.join(train_ann_path, ann_file)
    img_file_path = os.path.join(train_img_path, ann_file.replace("xml", "JPG"))
    if not os.path.exists(img_file_path):
        print("No image file found for train annotations in", ann_file)
    else:
        obj_count = count_objects_in_xml(ann_file_path)
        train_objects += obj_count
        train_files += 1
        train_loc_images[region] += 1
        train_loc_objects[region] += obj_count
    # print(ann_file)

print("original train files:", train_files)
print("original train objects:", train_objects)
print("original train location images:", sorted(train_loc_images.items(), key=lambda x: x[0]))
print("original train location objects:", sorted(train_loc_objects.items(), key=lambda x: x[0]))
print()

''' test images '''
test_loc_images = defaultdict(int)
test_loc_objects = defaultdict(int)
test_objects = 0
test_files = 0
ann_test_files = [x for x in sorted(os.listdir(test_ann_path)) if ".xml" in x]
for ann_file in ann_test_files:
    region = region_regex.search(ann_file).group(0).strip("_")
    ann_file_path = os.path.join(test_ann_path, ann_file)
    img_file_path = os.path.join(test_img_path, ann_file.replace("xml", "JPG"))
    if not os.path.exists(img_file_path):
        print("No image file found for test annotations in", ann_file)
    else:
        obj_count = count_objects_in_xml(ann_file_path)
        test_objects += obj_count
        test_files += 1
        test_loc_images[region] += 1
        test_loc_objects[region] += obj_count

print("original test files:", test_files)
print("original test objects:", test_objects)
print("original test location images:", sorted(test_loc_images.items(), key=lambda x: x[0]))
print("original test location objects:", sorted(test_loc_objects.items(), key=lambda x: x[0]))


''' verify keras-retinanet heridal '''

print("\nVerifying keras-retinanet heridal dataset...\n")

ann_path = os.path.join(retinanet_heridal_path, "Annotations")
img_path = os.path.join(retinanet_heridal_path, "JPEGImages")

img_set_path = os.path.join(retinanet_heridal_path, "ImageSets", "Main")

with open(os.path.join(img_set_path, "trainval.txt")) as f:
    train_ann_files = set([l.strip() + ".xml" for l in f.readlines()])

with open(os.path.join(img_set_path, "test.txt")) as f:
    test_ann_files = set([l.strip() + ".xml" for l in f.readlines()])

''' train images '''
train_loc_images = defaultdict(int)
train_loc_objects = defaultdict(int)
train_objects = 0
train_files = 0
ann_files = [x for x in sorted(os.listdir(ann_path)) if x in train_ann_files]
for ann_file in ann_files:
    region = region_regex.search(ann_file).group(0).strip("_")
    ann_file_path = os.path.join(ann_path, ann_file)
    img_file_path = os.path.join(img_path, ann_file.replace("xml", "JPG"))
    if not os.path.exists(img_file_path):
        print("No image file found for train annotations in", ann_file)
    else:
        obj_count = count_objects_in_xml(ann_file_path)
        train_objects += obj_count
        train_files += 1
        train_loc_images[region] += 1
        train_loc_objects[region] += obj_count
    # print(ann_file)

print("keras-retinanet train files:", train_files)
print("keras-retinanet train objects:", train_objects)
print("keras-retinanet train location images:", sorted(train_loc_images.items(), key=lambda x: x[0]))
print("keras-retinanet train location objects:", sorted(train_loc_objects.items(), key=lambda x: x[0]))
print()

''' test images '''
test_loc_images = defaultdict(int)
test_loc_objects = defaultdict(int)
test_objects = 0
test_files = 0
ann_test_files = [x for x in sorted(os.listdir(ann_path)) if x in test_ann_files]
for ann_file in ann_test_files:
    region = region_regex.search(ann_file).group(0).strip("_")
    ann_file_path = os.path.join(ann_path, ann_file)
    img_file_path = os.path.join(img_path, ann_file.replace("xml", "JPG"))
    if not os.path.exists(img_file_path):
        print("No image file found for test annotations in", ann_file)
    else:
        obj_count = count_objects_in_xml(ann_file_path)
        test_objects += obj_count
        test_files += 1
        test_loc_images[region] += 1
        test_loc_objects[region] += obj_count

print("keras-retinanet test files:", test_files)
print("keras-retinanet test objects:", test_objects)
print("keras-retinanet test location images:", sorted(test_loc_images.items(), key=lambda x: x[0]))
print("keras-retinanet test location objects:", sorted(test_loc_objects.items(), key=lambda x: x[0]))
