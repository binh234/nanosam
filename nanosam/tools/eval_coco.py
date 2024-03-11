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

from nanosam.utils import Predictor, get_config
from nanosam.tools.compute_eval_coco_metrics import compute_miou_metric

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
    return intersection / union * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, help="dataset root")
    parser.add_argument("--img_dir", default="data/coco/val2017", help="image folder path")
    parser.add_argument(
        "--ann_file", type=str, default="data/coco/annotations/instances_val2017.json"
    )
    parser.add_argument(
        "--encoder_cfg",
        type=str,
        required=True,
        help="Path to image encoder config file",
    )
    parser.add_argument(
        "--decoder_cfg",
        type=str,
        default="configs/inference/decoder.yaml",
        help="Path to mask decoder config file",
    )
    parser.add_argument(
        "--encoder_opt",
        type=str,
        nargs="+",
        help="Overridding config for image encoder",
    )
    parser.add_argument(
        "--decoder_opt",
        type=str,
        nargs="+",
        help="Overridding config for mask decoder",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--log_step", type=int, default=200)
    args = parser.parse_args()

    if args.data_root is not None:
        coco = COCO(osp.join(args.data_root, args.ann_file))
    else:
        coco = COCO(args.ann_file)

    encoder_cfg = get_config(args.encoder_cfg, args.encoder_opt)
    decoder_cfg = get_config(args.decoder_cfg, args.decoder_opt)
    predictor = Predictor(encoder_cfg, decoder_cfg)

    results = []
    image_ids = coco.getImgIds()
    prog_bar = tqdm(enumerate(image_ids), total=len(image_ids))

    for step, img_id in prog_bar:
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

        if (step + 1) % args.log_step == 0:
            metric_dict = compute_miou_metric(results)
            prog_bar.set_postfix(metric_dict)

    metric_dict = compute_miou_metric(results)
    prog_bar.set_postfix(metric_dict)
    print(", ".join([f"{key}={val:.3f}" for key, val in metric_dict.items()]))
    prog_bar.close()

    with open(args.output, "w") as f:
        json.dump(results, f)
