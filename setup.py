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

from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "pillow",
    "tqdm",
    "onnx",
    "pyyaml",
]

extras_require = {
    "dev": [
        "black",
        "flake8",
        "imagehash",
        "isort",
        "opencv-python",
        "pytest",
        "setuptools",
        "torch",
        "torchvision",
        "twine",
        "wheel",
        "onnxruntime",
    ],
    "gpu": ["onnxruntime-gpu"],
    "cpu": ["onnxruntime-openvino"],
}


setup(
    name="nanosam",
    packages=find_packages(),
    include_package_data=True,
    version="0.0.0",
    install_requires=install_requires,
    extras_require=extras_require,
    license="Apache License 2.0",
    description="NanoSAM is a Segment Anything (SAM) model variant that is capable of running in real-time",
    url="https://github.com/binh234/nanosam",
    download_url="https://github.com/binh234/nanosam.git",
    author="Binh Le",
    author_email="binhnd234@gmail.com",
    keywords=[
        "image-classification",
        "image-recognition",
        "pretrained-models",
        "knowledge-distillation",
        "product-recognition",
        "autoaugment",
        "cutmix",
        "randaugment",
        "gridmask",
        "deit",
        "repvgg",
        "swin-transformer",
        "image-retrieval-system",
    ],
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.6, <3.12",
)
