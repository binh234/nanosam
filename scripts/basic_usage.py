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
from nanosam.utils import Predictor, get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # Instantiate TensorRT predictor
    encoder_cfg = get_config(args.encoder_cfg, args.encoder_opt)
    decoder_cfg = get_config(args.decoder_cfg, args.decoder_opt)
    predictor = Predictor(encoder_cfg, decoder_cfg)

    # Read image and run image encoder
    image = PIL.Image.open("assets/dogs.jpg")
    predictor.set_image(image)

    # Segment using bounding box
    bbox = [100, 100, 770, 759]  # x1, y1, x2, y2

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
