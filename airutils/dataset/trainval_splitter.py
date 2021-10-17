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
Used for splitting Pascal VOC trainval.txt into train.txt and val.txt

Author: Pasi Pyrrö
Date: 28.6.2020
'''

import os
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', required=True,
                    help="Input trainval.txt path")
parser.add_argument('-s', '--split', default="0.9:0.1",
                    help="Defines the dataset split ratios for training, validation and testing data, for example -s 0.9:0.1")
parser.add_argument('-se', '--seed', type=int, default=1,
                    help="Fix random seed for training data splitting")
args = parser.parse_args()

val_split = float(args.split.split(":")[-1])

script_dir = os.path.dirname(os.path.realpath(__file__))

if not os.path.isabs(args.path):
    args.path = os.path.join(script_dir, args.path)

input_dir = os.path.dirname(args.path)
train_path = os.path.join(input_dir, "train.txt")
val_path = os.path.join(input_dir, "val.txt")

with open(args.path, "r") as f:
    trainval_ids = f.readlines()

print("Splitting", args.path, "with ratio", args.split)

train_ids, val_ids = train_test_split(
    trainval_ids, test_size=val_split, random_state=args.seed)

with open(train_path, "w") as f:
    f.writelines(sorted(train_ids))

print("Wrote", train_path)

with open(val_path, "w") as f:
    f.writelines(sorted(val_ids))

print("Wrote", val_path)