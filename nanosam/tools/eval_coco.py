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


import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

from nanosam.utils import PROVIDERS_DICT, Predictor, get_provider_options

import argparse
import json
import os.path as osp
from PIL import Image
from tqdm import tqdm


def predict_box(predictor, image, box, set_image=True):
    if set_image:
        predictor.set_image(image)

    points = np.array([[box[0], box[1]], [box[2], box[3]]])
    point_labels = np.array([2, 3])

    mask, iou_preds, _ = predictor.predict(points=points, point_labels=point_labels)

    mask = mask[0, iou_preds.argmax()] > 0

    return mask


def box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def draw_box(box):
    x = [box[0], box[0], box[2], box[2], box[0]]
    y = [box[1], box[3], box[3], box[1], box[1]]
    plt.plot(x, y, "g-")


def iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return intersection / union


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="dataset root")
    parser.add_argument("--img_dir", default="data/coco/val2017", help="image folder path")
    parser.add_argument(
        "--ann_file", type=str, default="data/coco/annotations/instances_val2017.json"
    )
    parser.add_argument("--image_encoder", type=str, default="data/mobile_sam_image_encoder.onnx")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.onnx")
    parser.add_argument("--output", type=str, default="data/mobile_sam_coco_results.json")
    parser.add_argument(
        "--provider",
        type=str,
        default="cpu",
        choices=PROVIDERS_DICT.keys(),
    )
    parser.add_argument(
        "-opt",
        "--provider_options",
        type=str,
        nargs="+",
        default=None,
        help="Provider options for model to run",
    )
    args = parser.parse_args()

    if args.data_root is not None:
        coco = COCO(osp.join(args.data_root, args.ann_file))
    else:
        coco = COCO(args.ann_file)

    provider_options = get_provider_options(args.provider_options)
    predictor = Predictor(args.image_encoder, args.mask_decoder, args.provider, provider_options)

    results = []
    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids):
        image_data = coco.loadImgs(img_id)[0]
        if args.data_root is not None:
            image_path = osp.join(args.data_root, args.img_dir, image_data["file_name"])
        else:
            image_path = osp.join(args.img_dir, image_data["file_name"])

        annotation_ids = coco.getAnnIds(imgIds=image_data["id"])
        anns = coco.loadAnns(annotation_ids)

        image = Image.open(image_path).convert("RGB")

        for j, ann in enumerate(anns):
            id = ann["id"]
            area = ann["area"]
            category_id = ann["category_id"]
            iscrowd = ann["iscrowd"]
            image_id = ann["image_id"]
            box = box_xywh_to_xyxy(ann["bbox"])
            mask = coco.annToMask(ann)
            mask_coco = mask > 0
            mask_sam = predict_box(predictor, image, box, set_image=(j == 0))

            result = {
                "id": ann["id"],
                "area": ann["area"],
                "category_id": ann["category_id"],
                "iscrowd": ann["iscrowd"],
                "image_id": ann["image_id"],
                "box": box,
                "iou": iou(mask_sam, mask_coco),
            }

            results.append(result)

    with open(args.output, "w") as f:
        json.dump(results, f)
