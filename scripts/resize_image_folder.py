import cv2

import argparse
import asyncio
import glob
import os
from PIL import Image
from tqdm.asyncio import tqdm_asyncio


def get_resize_shape(img_w, img_h, size=512, mode="resize_short"):
    if mode == "resize":
        resize_w, resize_h = img_w, img_h
    else:
        if mode == "resize_short":
            percent = float(size) / min(img_w, img_h)
        elif mode == "resize_long":
            percent = float(size) / max(img_w, img_h)
        else:
            raise NotImplementedError

        resize_w = int(round(img_w * percent))
        resize_h = int(round(img_h * percent))

    return resize_w, resize_h


async def resize_and_save(img_path, out_dir, size=512, mode="resize_short", backend="pil"):
    filename = os.path.basename(img_path)
    if backend == "pil":
        img = Image.open(img_path)
        img_w, img_h = img.size
        if min(img_w, img_h) > size:
            w, h = get_resize_shape(img_w, img_h, size, mode)
            img = img.resize((w, h), Image.BILINEAR)
        img.save(os.path.join(out_dir, filename))
    elif backend == "cv2":
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[0], img.shape[1]
        if min(img_w, img_h) > size:
            w, h = get_resize_shape(img_w, img_h, size, mode)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, filename), img)


async def resize_image_folder(img_dir, out_dir=None, size=512, mode="resize_short", backend="pil"):
    if out_dir is None:
        out_dir = img_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    backend = backend.lower()
    img_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_paths += glob.glob(os.path.join(img_dir, "*.png"))

    tasks = []
    for img_path in img_paths:
        tasks.append(resize_and_save(img_path, out_dir, size, mode, backend))

    _ = await tqdm_asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export features from image folder using SAM image encoder"
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
        default=None,
        help="Output folder to save resized images. If not set, will overwrite images inside input folder",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=512,
        help="Target image size",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="resize_long",
        choices=["resize", "resize_short", "resize_long"],
        help="In ['resize', 'resize_short', 'resize_long']. Resize mode, defaults to 'resize_long'",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pil",
        choices=["pil", "cv2"],
        help="Resize backend, defaults to PIL",
    )

    args = parser.parse_args()

    asyncio.run(
        resize_image_folder(
            img_dir=args.img_dir,
            out_dir=args.out_dir,
            size=args.size,
            mode=args.mode,
            backend=args.backend,
        )
    )
