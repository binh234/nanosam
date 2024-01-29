import numpy as np

import glob
import os
import PIL
from paddle.io import Dataset

from .preprocess import create_operators, transform


class ImageFolderDataset(Dataset):
    def __init__(self, image_root, transform_ops=None, **kwargs):
        self._img_root = image_root
        self._transform_ops = create_operators(transform_ops)

        image_paths = glob.glob(os.path.join(self._img_root, "*.jpg"))
        image_paths += glob.glob(os.path.join(self._img_root, "*.png"))
        self.images = image_paths

    def try_getitem(self, index):
        image = PIL.Image.open(self.images[index]).convert("RGB")
        if self._transform_ops:
            image = transform(image, self._transform_ops)
        image = np.asarray(image)
        image = np.transpose(image, (2, 0, 1))
        return image, np.zeros(1)

    def __getitem__(self, index):
        try:
            return self.try_getitem(index)
        except Exception as ex:
            print(
                "Exception occured when parse line: {} with msg: {}".format(self.images[index], ex)
            )
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)


class SamImageFolderDataset(Dataset):
    def __init__(self, image_root, feature_root, **kwargs):
        self._img_root = image_root
        self._feature_root = feature_root

        # Overwrite transform ops
        transform_ops = [
            {"SamResizeImage": {"resize_long": 512, "backend": "pil"}},
            {
                "NormalizeImage": {
                    "scale": 1.0 / 255,
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "order": "hwc",
                }
            },
            {"SamPad": {"size": 512}},
        ]
        self._transform_ops = create_operators(transform_ops)

        image_paths = glob.glob(os.path.join(self._img_root, "*.jpg"))
        image_paths += glob.glob(os.path.join(self._img_root, "*.png"))
        self.images = image_paths

    def try_getitem(self, index):
        image_path = self.images[index]
        image = PIL.Image.open(image_path).convert("RGB")
        if self._transform_ops:
            image = transform(image, self._transform_ops)
        image = np.asarray(image)
        image = np.transpose(image, (2, 0, 1))

        basename = os.path.basename(image_path)
        filename, _ = os.path.splitext(basename)
        feature = np.load(os.path.join(self._feature_root, filename + ".npy")).squeeze()
        return image, feature

    def __getitem__(self, index):
        try:
            return self.try_getitem(index)
        except Exception as ex:
            print(
                "Exception occured when parse line: {} with msg: {}".format(self.images[index], ex)
            )
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)
