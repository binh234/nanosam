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


import torch
import torch.nn as nn

import math
import timm
from typing import Tuple

from .registry import register_model


class TimmImageEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        num_conv_layers: int = 3,
        pretrained: bool = False,
        feature_dim: int = 256,
        feature_shape: Tuple[int, int] = (64, 64),
        neck_channels: int = 256,
        pos_embedding: bool = True,
        img_size: int = 1024,
    ):
        super().__init__()

        num_upsample = int(math.log2(2048 // img_size))
        assert int(img_size * pow(2, num_upsample)) == 2048

        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)

        channels = self.backbone.feature_info.channels()

        up_layers = []
        for i in range(num_conv_layers):
            in_channels = channels[-1] if i == 0 else neck_channels
            up_layers.append(nn.Conv2d(in_channels, neck_channels, 3, padding=1))
            up_layers.append(nn.GELU())

        for _ in range(num_upsample):
            up_layers.append(nn.ConvTranspose2d(neck_channels, neck_channels, 3, 2, 1, 1))
            up_layers.append(nn.GELU())

        self.up_1 = nn.Sequential(*up_layers)

        self.proj = nn.Sequential(
            nn.Conv2d(neck_channels, neck_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(neck_channels, feature_dim, 1, padding=0),
        )

        if pos_embedding:
            self.register_parameter(
                "pos_embedding",
                nn.Parameter(1e-5 * torch.randn(1, feature_dim, *feature_shape)),
            )
        else:
            self.pos_embedding = None

    def forward(self, x):
        x = self.backbone(x)
        # z = torch.cat([x[-2], self.up_1(x[-1])], dim=1)
        x = self.proj(self.up_1(x[-1]))
        if self.pos_embedding is not None:
            x = x + self.pos_embedding
        return x


@register_model("resnet18")
def resnet18(img_size=1024):
    return TimmImageEncoder("resnet18", pretrained=True, img_size=img_size)


@register_model("resnet34")
def resnet34(img_size=1024):
    return TimmImageEncoder("resnet34", pretrained=True, img_size=img_size)


@register_model("resnet50")
def resnet50(img_size=1024):
    return TimmImageEncoder("resnet50", pretrained=True, img_size=img_size)


@register_model("efficientvit_b0")
def efficientvit_b0(img_size=1024):
    return TimmImageEncoder("efficientvit_b0", pretrained=True, img_size=img_size)


@register_model("efficientvit_b1")
def efficientvit_b0(img_size=1024):
    return TimmImageEncoder("efficientvit_b1", pretrained=True, img_size=img_size)
