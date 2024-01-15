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

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from nanosam.datasets.image_folder import ImageFolder
from nanosam.models.torch import create_model, list_models

import argparse
import os
import tensorrt as trt
from tqdm import tqdm
from torch2trt import TRTModule


def load_image_encoder_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine, input_names=["image"], output_names=["image_embeddings"]
    )

    return image_encoder_trt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, help="The path to images to use for distillation")
    parser.add_argument(
        "output_dir",
        type=str,
        help="The directory to store checkpoints and training visualizations.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        choices=list_models(),
        help="The NanoSAM model name.",
    )
    parser.add_argument(
        "--teacher_size",
        type=int,
        default=1024,
        help="The size of image to feed to the teacher during distillation.",
    )
    parser.add_argument(
        "--student_size",
        type=int,
        default=1024,
        help="The size of image to feed to the student during distillation.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Limit the number of images per epoch.  Helpful for quick training runs when experimenting.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=200, help="The number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=8, help="The number of data loader workers."
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="The learning rate.")
    parser.add_argument(
        "--loss",
        type=str,
        default="huber",
        choices=["huber", "l1", "mse"],
        help="The loss function to use for distillation.",
    )
    parser.add_argument("--log_step", type=int, default=20, help="The logging step.")
    parser.add_argument(
        "--teacher_image_encoder_engine",
        type=str,
        default="data/mobile_sam_image_encoder_bs16.engine",
        help="The path to the image encoder engine to use as a teacher model.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_encoder_trt = load_image_encoder_engine(args.teacher_image_encoder_engine)

    image_encoder_cnn = create_model(args.model_name, args.student_size).cuda()

    if args.loss == "huber":
        loss_function = F.huber_loss
    elif args.loss == "l1":
        loss_function = F.l1_loss
    elif args.loss == "mse":
        loss_function = F.mse_loss
    else:
        raise RuntimeError(f"Unsupported loss function {args.loss}")

    optimizer = torch.optim.Adam(image_encoder_cnn.parameters(), lr=args.learning_rate)

    dataset = ImageFolder(args.images, img_size=args.teacher_size)

    if args.num_images is not None:
        dataset, _ = random_split(dataset, [args.num_images, len(dataset) - args.num_images])

    loader = DataLoader(
        dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers
    )

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        image_encoder_cnn.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0.0

        prog_bar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch + 1}")
        for cnt, image in prog_bar:
            image = image.cuda()
            if len(image) != args.batch_size:
                continue
            
            with torch.no_grad():
                features = image_encoder_trt(image)
            if args.teacher_size != args.student_size:
                image = F.interpolate(image, (args.student_size, args.student_size), mode="area")

            optimizer.zero_grad()
            output = image_encoder_cnn(image)

            loss = loss_function(output, features)

            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
        
            if (cnt + 1) % args.log_step == 0:
                prog_bar.set_postfix_str(f"loss: {epoch_loss / (cnt + 1):.5f}")

        epoch_loss /= len(loader)
        print(f"{epoch} - {epoch_loss: .5f}")

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"{epoch} - {epoch_loss}\n")

        torch.save(
            {
                "model": image_encoder_cnn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            checkpoint_path,
        )

        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(features[0, 0].detach().cpu().numpy())
        plt.subplot(122)
        plt.imshow(output[0, 0].detach().cpu().numpy())
        plt.savefig(os.path.join(args.output_dir, f"epoch_{epoch}.png"))
        plt.close()
