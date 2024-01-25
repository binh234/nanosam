import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

import argparse
import glob
import os
import tensorrt as trt
from typing import Tuple
from PIL import Image
from torch2trt import TRTModule
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


def load_image_encoder_engine(path: str):
    with trt.Logger() as logger, trt.Runtime(logger) as runtime:
        with open(path, "rb") as f:
            engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    image_encoder_trt = TRTModule(
        engine=engine, input_names=["image"], output_names=["image_embeddings"]
    )

    return image_encoder_trt


def transform(x, img_size=1024):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    transform = ResizeLongestSide(img_size)
    x = transform.apply_image(x)
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1).contiguous()

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def extract_feat_and_save(image_encoder, images, filenames, args):
    inp_tensor = torch.stack(images, 0)
    if len(filenames) < args.batch_size:
        pad_tensor = torch.zeros(
            (args.batch_size - len(filenames), *inp_tensor.shape[1:]), dtype=inp_tensor.dtype
        )
        inp_tensor = torch.cat((inp_tensor, pad_tensor), 0)
    inp_tensor = inp_tensor.cuda()

    with torch.no_grad():
        features = image_encoder(inp_tensor)

    features = features.cpu().numpy()
    if args.fp16:
        features = features.astype(np.float16)

    for name, feat in zip(filenames, features):
        save_path = os.path.join(args.out_dir, name + ".npy")
        np.save(save_path, feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export features from image folder using SAM image encoder"
    )
    parser.add_argument(
        "--image_encoder",
        "-enc",
        type=str,
        required=True,
        help="The path to the image encoder TensorRT engine.",
    )
    parser.add_argument(
        "--image_size",
        "-s",
        type=int,
        required=True,
        help="Input image size.",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--img_dir",
        "-i",
        type=str,
        required=True,
        help="Input image folder path.",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        required=True,
        help="Output folder to save extracted features.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to save features in FP16",
    )

    args = parser.parse_args()

    image_encoder = load_image_encoder_engine(args.image_encoder)

    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = glob.iglob(os.path.join(args.img_dir, "*"))
    batch_images = []
    batch_filenames = []
    for image_path in tqdm(image_paths):
        try:
            basename = os.path.basename(image_path)
            filename, image_ext = os.path.splitext(basename)
            if image_ext in IMG_EXTENSIONS:
                image = np.asarray(Image.open(image_path).convert("RGB"))
                image = transform(image, args.image_size)
                batch_images.append(image)
                batch_filenames.append(filename)

                if len(batch_filenames) == args.batch_size:
                    extract_feat_and_save(image_encoder, batch_images, batch_filenames, args)
                    batch_filenames = []
                    batch_images = []
        except Exception as ex:
            print("Exception occured when processing {} with msg: {}".format(image_path, ex))

    if len(batch_filenames) > 0:
        extract_feat_and_save(image_encoder, batch_images, batch_filenames, args)
