import os
import glob
import numpy as np
import PIL
from paddle.io import Dataset
from ppcls.data.preprocess import transform
from ppcls.data.dataloader.common_dataset import create_operators

from nanosam.utils.onnx_model import OnnxModel


class ImageFolderDataset(Dataset):
    def __init__(self, image_root, transform_ops=None, **kwargs):
        self._img_root = image_root
        self._transform_ops = create_operators(transform_ops)

        image_paths = glob.glob(os.path.join(self._img_root, "*.jpg"))
        image_paths += glob.glob(os.path.join(self._img_root, "*.png"))
        self.images = image_paths

    def __getitem__(self, index):
        image = PIL.Image.open(self.images[index]).convert("RGB")
        if self._transform_ops:
            image = transform(image, self._transform_ops)
        image = np.asarray(image)
        image = np.transpose(image, (2, 0, 1))
        return image

    def __len__(self):
        return len(self.images)
