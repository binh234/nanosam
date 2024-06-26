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
import PIL.Image
import argparse
import time
from nanosam.utils import Predictor, get_config


def benchmark_encoder(
    model, input_shape=(1024, 1024, 3), nwarmup=50, nruns=1000, log_steps=50, input_data=None
):
    if not input_data:
        input_data = np.random.randint(0, 255, input_shape).transpose(1, 2, 0)

    print("Warm up ...")
    for _ in range(nwarmup):
        _ = model.set_image(input_data)
    print("Start timing ...")
    timings = []
    for i in range(1, nruns + 1):
        start_time = time.perf_counter()
        _ = model.set_image(input_data)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        if i % log_steps == 0:
            print("Iteration %d/%d, avg batch time %.2f ms" % (i, nruns, np.mean(timings) * 1000))

    throughput = 1 / np.mean(timings)
    print("Average FPS: %.2f images/second" % (throughput))
    return throughput


def benchmark_decoder(
    model,
    points,
    point_labels,
    input_shape=(1024, 1024, 3),
    nwarmup=50,
    nruns=1000,
    log_steps=50,
    input_data=None,
):
    if not input_data:
        input_data = np.random.randint(0, 255, input_shape).transpose(1, 2, 0)

    model.set_image(input_data)
    print("Warm up ...")
    for _ in range(nwarmup):
        _ = model.predict(points, point_labels)
    print("Start timing ...")
    timings = []
    for i in range(1, nruns + 1):
        start_time = time.perf_counter()
        _ = model.predict(points, point_labels)
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
        if i % log_steps == 0:
            print("Iteration %d/%d, avg batch time %.2f ms" % (i, nruns, np.mean(timings) * 1000))

    throughput = 1 / np.mean(timings)
    print("Average FPS: %.2f images/second" % (throughput))
    return throughput


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
    parser.add_argument("--num_warmup", type=int, default=50)
    parser.add_argument("--num_run", type=int, default=500)
    args = parser.parse_args()

    # Instantiate TensorRT predictor
    encoder_cfg = get_config(args.encoder_cfg, args.encoder_opt)
    decoder_cfg = get_config(args.decoder_cfg, args.decoder_opt)
    predictor = Predictor(encoder_cfg, decoder_cfg)

    # Read image and run image encoder
    image = PIL.Image.open("assets/dogs.jpg")

    # Segment using bounding box
    bbox = [100, 100, 850, 759]  # x0, y0, x1, y1

    points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])

    point_labels = np.array([2, 3])

    print(f"Benchmarking encoder {encoder_cfg.path} ...")
    benchmark_encoder(predictor, nruns=args.num_run, nwarmup=args.num_warmup, input_data=image)
    print("-" * 80)
    print(f"Benchmarking decoder {decoder_cfg.path} ...")
    benchmark_decoder(
        predictor,
        points,
        point_labels,
        nruns=args.num_run,
        nwarmup=args.num_warmup,
        input_data=image,
    )
