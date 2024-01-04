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
import PIL.Image
import argparse
from nanosam.utils import PROVIDERS_DICT, Predictor, get_provider_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_encoder", type=str, default="data/resnet18_image_encoder.onnx")
    parser.add_argument("--mask_decoder", type=str, default="data/mobile_sam_mask_decoder.onnx")
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

    # Instantiate TensorRT predictor
    provider_options = get_provider_options(args.provider_options)
    predictor = Predictor(args.image_encoder, args.mask_decoder, args.provider, provider_options)

    # Read image and run image encoder
    # image = PIL.Image.open("assets/dogs.jpg")
    image = PIL.Image.open("app/assets/picture3.jpg")
    predictor.set_image(image)

    # Segment using bounding box
    bbox = [100, 100, 850, 759]  # x0, y0, x1, y1

    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])

    point_labels = np.array([2, 3])

    mask, _, _ = predictor.predict(points, point_labels)

    mask = mask[0, 0] > 0

    # Draw resykts
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    x = [bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]]
    y = [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]]
    plt.plot(x, y, "g-")
    plt.savefig("data/basic_usage_out.jpg")
    plt.show(block=False)
    plt.pause(4)
    plt.close()
