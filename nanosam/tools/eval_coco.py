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
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nanosam.tools.compute_eval_coco_metrics import compute_miou_metric
from nanosam.utils import Predictor, get_config

import argparse
import copy
import json
import os.path as osp
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


def evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    cocoeval_fn=COCOeval,
    img_ids=None,
):
    """
    Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/coco_evaluation.py.
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)

        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = cocoeval_fn(coco_gt, coco_dt, iou_type)

    max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


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
    parser.add_argument("--data_root", default="data/coco/", help="dataset root")
    parser.add_argument("--img_dir", default="val2017", help="image folder path")
    parser.add_argument("--ann_file", type=str, default="annotations/instances_val2017.json")
    parser.add_argument("--src_det_file", type=str, default=None)
    parser.add_argument(
        "--encoder_cfg",
        type=str,
        default="configs/inference/encoder.yaml",
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

    if args.src_det_file is not None:
        detections = json.load(open(args.src_det_file))
        src_dets = defaultdict(list)
        for det in detections:
            src_dets[det["image_id"]].append(det)

    encoder_cfg = get_config(args.encoder_cfg, args.encoder_opt)
    decoder_cfg = get_config(args.decoder_cfg, args.decoder_opt)
    predictor = Predictor(encoder_cfg, decoder_cfg)

    results = []
    image_ids = coco.getImgIds()
    prog_bar = tqdm(enumerate(image_ids), total=len(image_ids))

    for step, img_id in prog_bar:
        image_data = coco.loadImgs(img_id)[0]
        image_id = image_data["id"]
        if args.data_root is not None:
            image_path = osp.join(args.data_root, args.img_dir, image_data["file_name"])
        else:
            image_path = osp.join(args.img_dir, image_data["file_name"])
        image = Image.open(image_path).convert("RGB")

        if args.src_det_file is None:
            annotation_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(annotation_ids)

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
        else:
            detections = src_dets[image_id]
            for j, det in enumerate(detections):
                box = box_xywh_to_xyxy(det["bbox"])
                mask_sam = predict_box(predictor, image, box, set_image=(j == 0))
                rle = mask_util.encode(np.array(mask_sam[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                det["segmentation"] = rle
            results.extend(detections)

    prog_bar.close()
    if args.src_det_file is None:
        metric_dict = compute_miou_metric(results)
        print(", ".join([f"{key}={val:.3f}" for key, val in metric_dict.items()]))
    else:
        evaluate_predictions_on_coco(coco_gt=coco, coco_results=results, iou_type="segm")

    with open(args.output, "w") as f:
        json.dump(results, f)
