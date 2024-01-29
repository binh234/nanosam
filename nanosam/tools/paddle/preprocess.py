from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import cv2

import math
import random
from functools import partial
from paddle.vision.transforms import RandomResizedCrop
from PIL import Image
from ppcls.utils import logger


def transform(data, ops=[]):
    """transform"""
    for op in ops:
        data = op(data)
    return data


def create_operators(params):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    if params is None:
        return None
    assert isinstance(params, list), "operator config should be a list"
    ops = []
    for operator in params:
        assert isinstance(operator, dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = eval(op_name)(**param)
        ops.append(op)


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        _cv2_interp_from_str = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "area": cv2.INTER_AREA,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "random": (cv2.INTER_LINEAR, cv2.INTER_CUBIC),
        }
        _pil_interp_from_str = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
            "random": (Image.BILINEAR, Image.BICUBIC),
        }

        def _cv2_resize(src, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(src, size, interpolation=resample)

        def _pil_resize(src, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            if isinstance(src, np.ndarray):
                pil_img = Image.fromarray(src)
            else:
                pil_img = src
            pil_img = pil_img.resize(size, resample)
            if return_numpy:
                return np.asarray(pil_img)
            return pil_img

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(_cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            elif interpolation is None:
                interpolation = Image.BILINEAR
            self.resize_func = partial(
                _pil_resize, resample=interpolation, return_numpy=return_numpy
            )
        else:
            logger.warning(
                f'The backend of Resize only support "cv2" or "PIL". "f{backend}" is unavailable. Use "cv2" instead.'
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(src, size)


class RandCropImage(object):
    """random crop image"""

    def __init__(
        self,
        scale=[0.08, 1.0],
        ratio=[3.0 / 4.0, 4.0 / 3.0],
        interpolation=None,
        use_log_aspect=False,
        backend="cv2",
        **kwargs,
    ):
        assert backend.lower() in ["cv2", "pil"], "Only cv2 and PIL backends are supported"
        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3.0 / 4.0, 4.0 / 3.0] if ratio is None else ratio
        self.use_log_aspect = use_log_aspect

        self._resize_func = UnifiedResize(interpolation=interpolation, backend=backend)

    def __call__(self, img):
        scale = self.scale
        ratio = self.ratio

        if self.use_log_aspect:
            log_ratio = list(map(math.log, ratio))
            aspect_ratio = math.exp(random.uniform(*log_ratio))
        else:
            aspect_ratio = random.uniform(*ratio)

        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_w, img_h = img.size
        bound = min((float(img_w) / img_h) / aspect_ratio, (float(img_h) / img_w) * aspect_ratio)
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        w = int(math.sqrt(target_area * aspect_ratio))
        h = int(math.sqrt(target_area / aspect_ratio))

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        if isinstance(img, np.ndarray):
            img_crop = img[j : j + h, i : i + w, :]
        else:
            img_crop = img.crop((i, j, i + w, j + h))
        return img_crop


class SamResizeImage(object):
    """SAM resize image"""

    def __init__(
        self, size=None, resize_long=None, interpolation=None, backend="cv2", return_numpy=True
    ):
        if resize_long is not None and resize_long > 0:
            self.resize_long = resize_long
            self.size = None
        elif size is not None:
            self.resize_long = None
            self.size = (size, size) if type(size) is int else size
        else:
            raise ValueError(
                "invalid params for SamResizeImage for '\
                'both 'size' and 'resize_long' are None"
            )

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend, return_numpy=return_numpy
        )

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = img.size

        if self.resize_long is not None:
            percent = float(self.resize_long) / max(img_w, img_h)
            w = img_w * percent
            h = img_h * percent
            w = int(w + 0.5)
            h = int(h + 0.5)
        else:
            w, h = self.size
        return self._resize_func(img, (w, h))


class RandFlipImage(object):
    """random flip image
    flip_code:
        1: Flipped Horizontally
        0: Flipped Vertically
        -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            if isinstance(img, np.ndarray):
                return cv2.flip(img, self.flip_code)
            else:
                if self.flip_code == 1:
                    return img.transpose(Image.FLIP_LEFT_RIGHT)
                elif self.flip_code == 0:
                    return img.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    return img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img


class SamPad(object):
    def __init__(self, size=None, fill_value=0):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, list): image target size, if None, pad to multiple of size_divisor, default None
            fill_value (bool): value of pad area, default 0
        """
        if isinstance(size, int):
            size = [size, size]

        self.size = size
        self.fill_value = fill_value

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y : y + im_h, x : x + im_w, :] = image.astype(np.float32)
        return canvas

    def __call__(self, img):
        im_h, im_w = img.shape[:2]
        w, h = self.size
        assert im_h <= h and im_w <= w, "(h, w) of target size should be greater than (im_h, im_w)"

        if h == im_h and w == im_w:
            return img.astype(np.float32)

        offset_y, offset_x = 0, 0
        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        return self.apply_image(img, offsets, im_size, size)


class NormalizeImage(object):
    """normalize image such as substract mean, divide std"""

    def __init__(
        self, scale=None, mean=None, std=None, order="chw", output_fp16=False, channel_num=3
    ):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [3, 4], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = "float16" if output_fp16 else "float32"
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype("float32") * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == "chw" else img.shape[0]
            img_w = img.shape[2] if self.order == "chw" else img.shape[1]
            pad_zeros = (
                np.zeros((1, img_h, img_w)) if self.order == "chw" else np.zeros((img_h, img_w, 1))
            )
            img = (
                np.concatenate((img, pad_zeros), axis=0)
                if self.order == "chw"
                else np.concatenate((img, pad_zeros), axis=2)
            )

        img = img.astype(self.output_dtype)
        return img
