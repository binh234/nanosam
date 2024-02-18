import numpy as np
import torch

import argparse
import glob
import os
from PIL import Image
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def build_efficientvit_sam(model_type, checkpoint, device="cuda"):
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from efficientvit.sam_model_zoo import create_sam_model

    efficientvit_sam = create_sam_model(model_type, True, checkpoint)
    efficientvit_sam.to(device).eval()
    predictor = EfficientViTSamPredictor(efficientvit_sam)
    return predictor


def build_sam(model_type, checkpoint, device="cuda"):
    from nanosam.mobile_sam import SamPredictor, sam_model_registry

    mobile_sam = sam_model_registry[model_type](checkpoint=checkpoint)
    mobile_sam.to(device).eval()
    predictor = SamPredictor(mobile_sam)
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export features from image folder using SAM image encoder"
    )
    parser.add_argument(
        "--checkpoint",
        "-ckpt",
        type=str,
        required=True,
        help="The path to the SAM model checkpoint.",
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
        "--model_type",
        type=str,
        default="vit_h",
        choices=["default", "vit_h", "vit_l", "vit_b", "l0", "l1", "l2", "xl0", "xl1"],
        help="In ['default', 'vit_h', 'vit_l', 'vit_b', 'l0', 'l1', 'l2', 'xl0', 'xl1']. Which type of SAM model to export.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to save features in FP16",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type in ["default", "vit_h", "vit_l", "vit_b"]:
        predictor = build_sam(args.model_type, args.checkpoint, device)
    else:
        predictor = build_efficientvit_sam(args.model_type, args.checkpoint, device)

    os.makedirs(args.out_dir, exist_ok=True)
    image_paths = glob.iglob(os.path.join(args.img_dir, "*"))
    for image_path in tqdm(image_paths):
        try:
            basename = os.path.basename(image_path)
            filename, image_ext = os.path.splitext(basename)
            save_path = os.path.join(args.out_dir, filename + ".npy")
            if image_ext in IMG_EXTENSIONS and not os.path.exists(save_path):
                image = np.asarray(Image.open(image_path).convert("RGB"))

                predictor.set_image(image)
                features = predictor.features
                features = features.cpu().numpy()
                if args.fp16:
                    features = features.astype(np.float16)

                np.save(save_path, features)
        except Exception as ex:
            print("Exception occured when processing {} with msg: {}".format(image_path, ex))
