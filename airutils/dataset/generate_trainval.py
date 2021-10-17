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

import sys
import os

def generate(ann_folder):
    files = sorted([x.replace(".xml", "") for x in os.listdir(ann_folder) if ".xml" in x])
    with open(os.path.join(ann_folder, "trainval.txt"), "w") as out:
        for f in files:
            out.write(f + "\n")
    print("Generated trainval.txt in", ann_folder)

if __name__ == "__main__":
    if not sys.argv[1:]:
        print("Usage: python generate_trainval.py [pascal_voc_annotations_folder]")
        sys.exit(-1)
    generate(sys.argv[1])
    sys.exit(0)