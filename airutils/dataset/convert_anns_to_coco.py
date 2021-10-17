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
Use this script to standardize the cv_train_data produced by the 
tagging tool to COCO dataset format.

Author: Pasi Pyrrö
Date: 13.1.2020
'''

import argparse
import shutil
import time
import json
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split


def merge_annotations(file_tuples, dest_path, data_type, args):
    with open(file_tuples[0][1], "r") as f:
        out_json = json.loads(f.read())
    # set "the commons" lisence
    out_json["licenses"] = {
        "url": "http://flickr.com/commons/usage/",
        "id": 0,
        "name": "No known copyright restrictions"
    }
    out_json["info"]["description"] = re.sub(
        r" at time.*", "", out_json["info"]["description"])

    for img, ann in file_tuples:
        # copy images
        shutil.copyfile(img, os.path.join(dest_path, "images",
                                          f"{data_type}2017", os.path.basename(img)))
        with open(ann, "r") as f:
            orig_json = json.loads(f.read())
        out_json["images"] += orig_json["images"]
        out_json["annotations"] += orig_json["annotations"]

    # remove duplicate images
    out_json["images"] = list(
        {v["id"]: v for v in out_json["images"]}.values())

    delete_anns = []
    keep_cats = set(args.keep_cats)

    # reassign annotation ids
    ann_id = int(time.time())
    for ann in out_json["annotations"]:
        ann["id"] = ann_id
        ann_id += 1
        if ann["category_id"] not in keep_cats:
            delete_anns.append(ann)
    
    for dann in delete_anns:
        out_json["annotations"].remove(dann)

    for cat in out_json["categories"][:]:
        if cat["id"] not in keep_cats:
            out_json["categories"].remove(cat)

    # write the output json
    output_filename = os.path.join(
        dest_path, "annotations", f"instances_{data_type}2017.json")
    with open(output_filename, "w") as f:
        f.write(json.dumps(out_json, ensure_ascii=False))


def convert(args):
    dest_path = args.dest if args.dest else os.path.join(
        os.path.dirname(args.paths[0]), "cv_train_data_coco")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path, exist_ok=True)
    else:
        if not args.y:
            answer = input(
                "Destination path already exists, do you wish to override? (y/n) ")
        if args.y or answer in ("y", "yes"):
            shutil.rmtree(dest_path)
            os.makedirs(dest_path, exist_ok=True)
    img_files = []
    ann_files = []
    for path in args.paths:
        img_files += sorted([os.path.join(path, f)
                             for f in os.listdir(path) if f.endswith(".JPG")])
        ann_files += sorted([os.path.join(path, f)
                             for f in os.listdir(path) if f.endswith(".json")])
    file_tuples = list(zip(img_files, ann_files))

    try:
        train_split, val_split, test_split = [
            float(s) for s in args.split.split(":")]
    except Exception as e:
        print("Invalid split argument format, parsing resulted in error:", str(e))

    assert sum((train_split, val_split, test_split)
               ) == 1., "Invalid dataset split"

    if test_split > 0:
        train_tuples, test_tuples = train_test_split(
            file_tuples, test_size=test_split, random_state=args.seed)
    else:
        train_tuples = file_tuples
        test_tuples = []
    if val_split > 0:
        train_tuples, val_tuples = train_test_split(
            train_tuples, test_size=val_split, random_state=args.seed)
    else:
        train_tuples = train_tuples
        val_tuples = []
    N = len(file_tuples)
    ntrain, nval, ntest = len(train_tuples), len(val_tuples), len(test_tuples)
    rtrain, rval, rtest = ntrain / N, nval / N, ntest / N
    print(
        f"Using dataset split -> train: {rtrain:.2f} ({ntrain}) | val: {rval:.2f} ({nval}) | test: {rtest:.2f} ({ntest})")

    # create the folder structure
    os.makedirs(os.path.join(dest_path, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(dest_path, "images"), exist_ok=True)
    if train_tuples:
        os.makedirs(os.path.join(dest_path, "images",
                                 "train2017"), exist_ok=True)
    if val_tuples:
        os.makedirs(os.path.join(
            dest_path, "images", "val2017"), exist_ok=True)
    if test_tuples:
        os.makedirs(os.path.join(dest_path, "images",
                                 "test2017"), exist_ok=True)

    for data_type, tuples in [("train", train_tuples), ("val", val_tuples), ("test", test_tuples)]:
        if tuples:
            merge_annotations(tuples, dest_path, data_type, args)

    print("Conversion output wrote to", dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', required=True,
                        help="Input Dataset folder paths (can be more than one)", nargs="+")
    parser.add_argument('-d', '--dest', required=False,
                        help="Destination Dataset folder path")
    parser.add_argument('-s', '--split', default="0.7:0.1:0.2",
                        help="Defines the dataset split ratios for training, validation and testing data, for example -s 0.7:0.1:0.2")
    parser.add_argument('-se', '--seed', type=int,
                        help="Fix random seed for training data splitting")
    parser.add_argument('-y', '--y', action="store_true",
                        help="Do not prompt destination folder override.")
    parser.add_argument('-k', '--keep-cats', type=int, nargs="*",
                        help="Category ids to keep (if not given keep all)")
    args = parser.parse_args()
    print("Coco converting", *args.paths)
    for path in args.paths:
        assert os.path.exists(
            path), f"Input Dataset folder '{path}' doesn't exist!"
    convert(args)
