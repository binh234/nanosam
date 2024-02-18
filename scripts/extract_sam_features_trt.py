import numpy as np
import torch

import argparse
import glob
import os
import tensorrt as trt
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class TrtModel:
    def __init__(self, path, **kwargs):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.load_engine(path)

    def load_engine(self, path):
        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            with open(path, "rb") as f:
                engine_bytes = f.read()
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.get_inference_info(self.engine)

    def get_inference_info(self, engine):
        self.context = engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        self.input_shapes = []
        self.output_shapes = []
        self.binding_outputs = []
        self.outputs = []

        for binding in engine:
            shape = engine.get_tensor_shape(binding)
            dtype = trt.nptype(engine.get_tensor_dtype(binding))

            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                self.input_shapes.append(shape)
            else:
                tensor = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).cuda()
                self.binding_outputs.append(tensor.data_ptr())
                self.outputs.append(tensor)
                self.output_shapes.append(shape)

    def infer(self, *args):
        """Run inference on TensorRT engine."""
        assert len(args) == len(
            self.input_shapes
        ), "Number of arguments must match number of model's input"

        binding_outputs = [inp.data_ptr() for inp in args] + self.binding_outputs
        self.context.execute_async_v2(
            bindings=binding_outputs, stream_handle=self.stream.cuda_stream
        )
        self.stream.synchronize()

        return self.outputs  # torch tensors on GPU

    def __call__(self, *args):
        out = self.infer(*args)

        return out


def get_preprocess_shape(img_h: int, img_w: int, size: int = 512):
    percent = float(size) / max(img_w, img_h)
    new_h = int(round(img_h * percent))
    new_w = int(round(img_w * percent))
    return new_h, new_w


def preprocess_image(image, size: int = 512):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image_mean = np.asarray([123.675, 116.28, 103.53])[:, None, None]
    image_std = np.asarray([58.395, 57.12, 57.375])[:, None, None]

    width, height = image.size
    resize_height, resize_width = get_preprocess_shape(height, width, size)

    image_resized = image.resize((resize_width, resize_height), resample=Image.BILINEAR)
    image_resized = np.asarray(image_resized)
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_resized_normalized = (image_resized - image_mean) / image_std
    image_tensor = np.zeros((1, 3, size, size), dtype=np.float32)
    image_tensor[0, :, :resize_height, :resize_width] = image_resized_normalized

    return image_tensor


def extract_feat_and_save(image_encoder, images, filenames, args):
    inp_tensor = np.stack(images, 0)
    if len(filenames) < args.batch_size:
        pad_tensor = np.zeros(
            (args.batch_size - len(filenames), *inp_tensor.shape[1:]), dtype=inp_tensor.dtype
        )
        inp_tensor = np.concatenate((inp_tensor, pad_tensor), 0)
    inp_tensor = torch.as_tensor(inp_tensor).cuda()

    with torch.no_grad():
        features = image_encoder(inp_tensor)[0]

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

    image_encoder = TrtModel(args.image_encoder)

    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = glob.iglob(os.path.join(args.img_dir, "*"))
    batch_images = []
    batch_filenames = []
    for image_path in tqdm(image_paths):
        try:
            basename = os.path.basename(image_path)
            filename, image_ext = os.path.splitext(basename)
            save_path = os.path.join(args.out_dir, filename + ".npy")
            if image_ext in IMG_EXTENSIONS and not os.path.exists(save_path):
                image = Image.open(image_path).convert("RGB")
                image = preprocess_image(image, args.image_size)
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
