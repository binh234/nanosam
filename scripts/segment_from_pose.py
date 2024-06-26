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

import PIL.Image
import matplotlib.pyplot as plt
from nanosam.utils.trt_pose import PoseDetector, pose_to_sam_points
from nanosam.utils import Predictor, get_config


def get_torso_points(pose):
    return pose_to_sam_points(
        pose,
        ["left_shoulder", "right_shoulder"],
        [
            "nose",
            "left_ear",
            "right_ear",
            "right_wrist",
            "left_wrist",
            "left_knee",
            "right_knee",
        ],
    )


def get_face_points(pose):
    return pose_to_sam_points(
        pose,
        ["left_eye", "right_eye", "nose", "left_ear", "right_ear"],
        ["left_shoulder", "right_shoulder", "neck", "left_wrist", "right_wrist"],
    )


def get_pants_points(pose):
    return pose_to_sam_points(pose, ["left_hip", "right_hip"], ["left_shoulder", "right_shoulder"])


def get_right_hand_points(pose):
    return pose_to_sam_points(
        pose,
        ["right_wrist"],
        [
            "left_wrist",
            "left_ankle",
            "right_ankle",
            "nose",
            "left_shoulder",
            "right_shoulder",
        ],
    )


def get_left_hand_points(pose):
    return pose_to_sam_points(
        pose,
        ["left_wrist"],
        [
            "right_wrist",
            "right_ankle",
            "left_ankle",
            "nose",
            "left_shoulder",
            "right_shoulder",
        ],
    )


def get_left_leg_points(pose):
    return pose_to_sam_points(
        pose, ["left_ankle"], ["left_hip", "right_hip", "left_wrist", "right_wrist"]
    )


def get_right_leg_points(pose):
    return pose_to_sam_points(
        pose, ["right_ankle"], ["left_hip", "right_hip", "left_wrist", "right_wrist"]
    )


pose_model = PoseDetector(
    "data/densenet121_baseline_att_256x256_B_epoch_160.pth", "assets/human_pose.json"
)

image = PIL.Image.open("assets/john_1.jpg")

detections = pose_model.predict(image)

encoder_cfg = get_config("configs/inference/encoder.yaml")
decoder_cfg = get_config("configs/inference/decoder.yaml")
sam_predictor = Predictor(encoder_cfg, decoder_cfg)

pose = detections[0]

points, point_labels = get_pants_points(detections[0])

sam_predictor.set_image(image)


def subplot_notick(a, b, c):
    ax = plt.subplot(a, b, c)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")


def predict_and_show(N, index, pose, fg_points, bg_points):
    subplot_notick(2, N, index + 1)
    points, point_labels = pose_to_sam_points(pose, fg_points, bg_points)
    mask, _, _ = sam_predictor.predict(points, point_labels)
    plt.imshow(image)
    plt.plot(points[point_labels == 1, 0], points[point_labels == 1, 1], "g.")
    plt.plot(points[point_labels != 1, 0], points[point_labels != 1, 1], "r.")
    subplot_notick(2, N, N + index + 1)
    plt.imshow(image)
    plt.imshow(mask[0, 0] > 0, alpha=0.5)


N = 4
AR = image.width / image.height
plt.figure(figsize=(10 / AR, 10))
predict_and_show(
    N,
    0,
    pose,
    ["left_shoulder", "right_shoulder"],
    ["nose", "left_knee", "right_knee", "left_hip", "right_hip"],
)
predict_and_show(
    N,
    1,
    pose,
    ["left_eye", "right_eye", "nose", "left_ear", "right_ear"],
    ["left_shoulder", "right_shoulder", "neck", "left_wrist", "right_wrist"],
)
predict_and_show(N, 2, pose, ["left_hip", "right_hip"], ["left_shoulder", "right_shoulder"])
predict_and_show(
    N,
    3,
    pose,
    ["nose", "left_wrist", "right_wrist", "left_ankle", "right_ankle"],
    ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("data/segment_from_pose_out.png", bbox_inches="tight")
