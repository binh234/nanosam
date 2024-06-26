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
from typing import Optional, Union

from .onnx_model import OnnxModel
from .trt_model import TrtModel


def get_preprocess_shape(img_h: int, img_w: int, size: int = 512):
    scale = float(size) / max(img_w, img_h)
    new_h = int(round(img_h * scale))
    new_w = int(round(img_w * scale))
    return new_h, new_w


class SamPreprocess:
    def __init__(self, size: int = 512, normalize_input: bool = True):
        self.size = size
        self.normalize_input = normalize_input
        self.mean = np.asarray([123.675, 116.28, 103.53])[:, None, None]
        self.std = np.asarray([58.395, 57.12, 57.375])[:, None, None]

    def __call__(self, image) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        width, height = image.size
        resize_height, resize_width = get_preprocess_shape(height, width, self.size)

        image_resized = image.resize((resize_width, resize_height), resample=Image.BILINEAR)
        image_resized = np.asarray(image_resized)
        image_resized = np.transpose(image_resized, (2, 0, 1))
        if self.normalize_input:
            image_resized = (image_resized - self.mean) / self.std
        image_tensor = np.zeros((1, 3, self.size, self.size), dtype=np.float32)
        image_tensor[0, :, :resize_height, :resize_width] = image_resized

        return image_tensor


def preprocess_points(points, image_size, size: int = 1024):
    scale = size / max(*image_size)
    points = points * scale
    return points


def run_mask_decoder(mask_decoder, features, points, point_labels, mask_input=None):
    """Predict masks for the given input prompts, using the image features.

    Args:
        mask_decoder (Union[OnnxModel, TrtModel]): mask decoder model
        features (np.ndarray): image features extracted using image encoder
        points (Union[List[int], np.ndarray]): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
        point_labels (Union[List[int], np.ndarray]): A length N array of labels for
            the point prompts.
            0: background point
            1: foreground point
            2: top left box corner
            3: bootom right box corner
        mask_input (np.ndarray, optional): A low resolution mask input to the model,
            typically coming from a previous prediction iteration. Has form 1xHxW,
            where for SAM, H=W=256. Defaults to None.

    Returns:
        (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
        (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
    """
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
        self.normalize_input = image_encoder_cfg.pop("normalize_input", True)
        encoder_cls = image_encoder_cfg.pop("name", "OnnxModel")
        self.image_encoder = eval(encoder_cls)(**image_encoder_cfg)
        decoder_cls = mask_decoder_cfg.pop("name", "OnnxModel")
        self.mask_decoder = eval(decoder_cls)(**mask_decoder_cfg)
        self.image_encoder_size = self.image_encoder.get_input_shapes(0)[-1]
        self.preprocess = SamPreprocess(self.image_encoder_size, self.normalize_input)

    def reset_image(self) -> None:
        self.is_image_set = False
        self.features = None
        self.original_size = None

    def set_image(self, image: Union[np.ndarray, Image.Image]):
        if isinstance(image, np.ndarray):
            self.original_size = image.shape[:2]
        else:
            img_w, img_h = image.size
            self.original_size = (img_h, img_w)
        self.image_tensor = self.preprocess(image)
        self.features = self.image_encoder(self.image_tensor)
        self.is_image_set = True

    def predict(
        self, points: np.ndarray, point_labels: np.ndarray, mask_input: Optional[np.ndarray] = None
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        points = preprocess_points(points, self.original_size)
        mask_iou, low_res_mask = run_mask_decoder(
            self.mask_decoder, self.features, points, point_labels, mask_input
        )

        hi_res_mask = upscale_mask(low_res_mask, self.original_size)

        return hi_res_mask, mask_iou, low_res_mask
