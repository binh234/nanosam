# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json


def filter_results_by_area(results, min=None, max=None):
    filtered = []
    for r in results:
        if min is not None and r["area"] < min:
            continue
        if max is not None and r["area"] > max:
            continue
        filtered.append(r)
    return filtered


def filter_results_by_category_id(results, category_id):
    return [r for r in results if r["category_id"] == category_id]


def compute_iou(results):
    return sum(r["iou"] for r in results) / len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("coco_results", type=str, default="data/mobile_sam_coco_results.json")
    parser.add_argument("--category_id", type=int, default=None)
    parser.add_argument(
        "--size", type=str, default="all", choices=["all", "small", "medium", "large"]
    )
    args = parser.parse_args()

    print(args)

    with open(args.coco_results, "r") as f:
        results = json.load(f)

    if args.category_id is not None:
        results = filter_results_by_category_id(results, args.category_id)

    if args.size == "small":
        results = filter_results_by_area(results, None, 32**2)
    elif args.size == "medium":
        results = filter_results_by_area(results, 32**2, 96**2)
    elif args.size == "large":
        results = filter_results_by_area(results, 96**2, None)

    miou = compute_iou(results)
    print(f"mIOU {args.size}: {miou}")

    if args.size == "all":
        size_names = ["small", "medium", "large"]
        size_areas = [None, 32**2, 96**2, None]
        for i, name in enumerate(size_names):
            results_filtered = filter_results_by_area(results, size_areas[i], size_areas[i + 1])
            miou_filtered = compute_iou(results_filtered)
            print(f"mIOU {name}: {miou_filtered}")
