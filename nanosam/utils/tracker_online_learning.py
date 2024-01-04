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
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

from .predictor import Predictor, upscale_mask
from .tracker import bbox2points, resize_mask, mask_to_box


@torch.no_grad()
def mask_to_centroid_soft(mask):
    mask = mask[0, 0]
    mask = torch.from_numpy(mask)
    weight = torch.sigmoid(mask)
    i, j = torch.meshgrid(torch.arange(mask.shape[0]), torch.arange(mask.shape[1]))
    ij = torch.stack([i, j]).cuda()
    center = torch.sum(weight[None, :] * ij, dim=(1, 2)) / torch.sum(weight)

    # mask_pts = np.argwhere(mask)
    # center_y = np.median(mask_pts[:, 0])
    # center_x = np.median(mask_pts[:, 1])
    center_y = float(center[0])
    center_x = float(center[1])
    return np.array([center_x, center_y])


class SelfAtt(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.layer = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(a, b, 1), nn.Sigmoid())
        self.feat = nn.Conv2d(a, b, 1)

    def forward(self, x):
        return torch.sum(self.layer(x) * self.feat(x), dim=1, keepdim=True)


class ObjectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.model = nn.Sequential(
            nn.ConvTranspose2d(256, 8, 3, 2, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(8, 8, 3, 2, 1, 1),
            nn.GELU(),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(8, 1, 1)
            # nn.Conv2d(256, 16, 1),
            # nn.GELU(),
            # nn.Conv2d(16, 1, 1),
            # nn.UpsamplingBilinear2d(scale_factor=4)
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            device = self.dummy_param.device
            x = torch.from_numpy(x).to(device)
        return self.model(x)


class TrackerOnline(object):
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self.target_mask = None
        self.token = None
        self._targets = []
        self._features = []
        self.object_model = None
        self._lb = None

    def set_image(self, image):
        self.predictor.set_image(image)

    def predict_mask(self, points=None, point_labels=None, box=None, mask_input=None):
        if box is not None:
            points, point_labels = bbox2points(box)

        mask_high, iou_pred, mask_raw = self.predictor.predict(
            points, point_labels, mask_input=mask_input
        )

        idx = int(iou_pred.argmax())
        mask_raw = mask_raw[:, idx : idx + 1, :, :]
        mask_high = mask_high[:, idx : idx + 1, :, :]
        return mask_high, mask_raw, resize_mask(mask_raw, (64, 64))

    def fit_token(self, features, mask_low):
        """
        Finds token that when dot-producted with features minimizes MSE with low
        resolution masks.

        Args:
            features (Nx256x64x64) (np.ndarray):
            mask (Nx1x64x64) - Should be logits type
        """
        N = features.shape[0]
        assert N == mask_low.shape[0]
        A = np.transpose(features, (0, 2, 3, 1)).reshape(N * 64 * 64, 256)
        B = np.transpose(mask_low, (0, 2, 3, 1)).reshape(N * 64 * 64, 1)
        X = np.linalg.lstsq(A, B, rcond=None)[0].reshape(1, 256, 1, 1)
        return X

    def apply_token(self, features, token):
        return resize_mask(np.sum(features * token, axis=1, keepdims=True), (256, 256))

    def init(self, image, point=None, box=None):
        with torch.no_grad():
            self.set_image(image)

            if point is not None:
                mask_high, mask_raw, mask_low = self.predict_mask(np.array([point]), np.array([1]))

            if box is not None:
                mask_high, mask_raw, mask_low = self.predict_mask(*bbox2points(box))

        self.object_model = ObjectModel().cuda()

        # self.optimizer = torch.optim.LBFGS(self.object_model.parameters(), history_size=1, max_iter=20, lr=1)
        self.optimizer = torch.optim.Adam(self.object_model.parameters(), lr=1e-3)
        self._first = (self.predictor.features, mask_raw)
        self.fit_model(mask_raw, 400)

        return mask_high

    def fit_model(self, target, iters):
        xf, yf = self._first
        x = self.predictor.features  # torch.cat(self._features, dim=0)
        y = target  # torch.cat(self._targets, dim=0)
        x = torch.from_numpy(np.concatenate([xf, x], axis=0))
        y = torch.from_numpy(np.concatenate([yf, y], axis=0))
        # def closure():
        #     self.optimizer.zero_grad()
        #     output = self.object_model(x)
        #     loss = sigmoid_focal_loss(output, torch.sigmoid(y), reduction="mean")
        #     loss.backward()
        #     return loss

        # self.optimizer.step(closure)
        for i in range(iters):
            self.optimizer.zero_grad()
            output = self.object_model(x)
            loss = sigmoid_focal_loss(output, torch.sigmoid(y), reduction="mean")
            loss.backward()
            self.optimizer.step()

    def reset(self):
        self._features = []
        self._targets = []
        self.token = None

    def update(self, image):
        self.set_image(image)

        with torch.no_grad():
            mask_token = self.object_model(self.predictor.features).detach().cpu().numpy()
        mask_token_up = upscale_mask(mask_token, (image.height, image.width))

        if np.count_nonzero(mask_token_up > 0) > 10:
            box = mask_to_box(mask_token_up)
            points_1, point_labels_1 = bbox2points(box)
            if self._lb is not None:
                box = 0.5 * box + 0.5 * self._lb
            self._lb = box
            mask_high, mask_raw, mask_low = self.predict_mask(
                points_1, point_labels_1, mask_input=mask_token
            )
            # points = np.array([mask_to_centroid_soft(mask_token_up)])
            # point_labels = np.array([1])
            # mask_high, mask_raw, mask_low = self.predict_mask(points, point_labels, mask_input=mask_raw)

            a = mask_token > 0
            b = mask_raw > 0
            inter = np.count_nonzero(a & b)
            union = np.count_nonzero(a | b)
            iou = float(inter) / (1e-3 + float(union))

            if iou > 0.5 and iou < 0.75:
                self.fit_model(mask_raw, 20)

            with torch.no_grad():
                mask_token = self.object_model(self.predictor.features).detach().cpu().numpy()
            mask_token_up = upscale_mask(mask_token, (image.height, image.width))

        return mask_token_up
