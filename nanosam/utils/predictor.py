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
import cv2

from PIL import Image
from typing import Any

from .onnx_model import OnnxModel


def get_preprocess_shape(img_h: int, img_w: int, size: int = 512):
    scale = float(size) / max(img_w, img_h)
    new_h = int(round(img_h * scale))
    new_w = int(round(img_w * scale))
    return new_h, new_w


def preprocess_image(image, size: int = 512):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_mean = np.asarray([123.675, 116.28, 103.53])[:, None, None]
    image_std = np.asarray([58.395, 57.12, 57.375])[:, None, None]

    width, height = image.size
    resize_height, resize_width = get_preprocess_shape(height, width, size)

    image_resized = image.resize((resize_width, resize_height), resample=Image.BILINEAR)
    image_resized = np.asarray(image_resized)
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized_normalized = (image_resized - image_mean) / image_std
    image_tensor = np.zeros((1, 3, size, size), dtype=np.float32)
    image_tensor[0, :, :resize_height, :resize_width] = image_resized_normalized

    return image_tensor


def preprocess_points(points, image_size, size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points


def run_mask_decoder(mask_decoder, features, points=None, point_labels=None, mask_input=None):
    if points is not None:
        assert point_labels is not None
        assert len(points) == len(point_labels)

    image_point_coords = np.asarray([points], dtype=np.float32)
    image_point_labels = np.asarray([point_labels], dtype=np.float32)

    if mask_input is None:
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        has_mask_input = np.zeros(1, dtype=np.float32)
    else:
        has_mask_input = np.ones(1, dtype=np.float32)

    iou_predictions, low_res_masks = mask_decoder(
        features, image_point_coords, image_point_labels, mask_input, has_mask_input
    )

    return iou_predictions, low_res_masks


def upscale_mask(mask, image_shape, size=256, interpolation=cv2.INTER_LINEAR):
    """

    Args:
        mask (np.ndarray): Input mask with shape [B, C, H, W]
        image_shape (Union[int, Tuple[int, int]]): Desired output size in (H, W) format
        size (int, optional): Mask size. Defaults to 256.

    Returns:
        (np.ndarray): Upscaled mask
    """
    lim_y, lim_x = get_preprocess_shape(image_shape[0], image_shape[1], size)

    bs = mask.shape[0]
    mask = np.transpose(mask[:, :, :lim_y, :lim_x], (2, 3, 0, 1)).reshape(lim_y, lim_x, -1)
    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=interpolation)
    mask = mask.reshape(*image_shape, bs, -1)
    mask = np.transpose(mask, (2, 3, 0, 1))

    return mask


class Predictor(object):
    def __init__(
        self,
        image_encoder_cfg,
        mask_decoder_cfg,
    ):
        self.image_encoder = OnnxModel(**image_encoder_cfg)
        self.mask_decoder = OnnxModel(**mask_decoder_cfg)
        self.image_encoder_size = self.image_encoder.get_inputs()[0].shape[-1]

    def set_image(self, image):
        self.image = image
        self.image_tensor = preprocess_image(image, self.image_encoder_size)
        self.features = self.image_encoder(self.image_tensor)[0]

    def predict(self, points, point_labels, mask_input=None):
        points = preprocess_points(points, (self.image.height, self.image.width))
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder, self.features, points, point_labels, mask_input
        )

        hi_res_mask = upscale_mask(low_res_mask, (self.image.height, self.image.width))

        return hi_res_mask, mask_iou, low_res_mask
