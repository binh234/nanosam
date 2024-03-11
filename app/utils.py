import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def point_prompt(masks, points, point_label, target_height, target_width):
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    if h != target_height or w != target_width:
        points = [
            [int(point[0] * w / target_width), int(point[1] * h / target_height)]
            for point in points
        ]
    onemask = np.zeros((h, w))
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation["segmentation"]
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1:
                if point_label[i] == 0:
                    onemask -= mask
                else:
                    onemask += mask
                break
    onemask = onemask > 0
    return onemask, 0


def format_results(masks, scores, logits, filter=0):
    annotations = []
    n = len(scores)
    for i in range(n):
        annotation = {}

        mask = masks[i] > 0
        tmp = np.where(mask)
        annotation["id"] = i
        annotation["segmentation"] = mask
        annotation["bbox"] = [
            np.min(tmp[0]),
            np.min(tmp[1]),
            np.max(tmp[1]),
            np.max(tmp[0]),
        ]
        annotation["score"] = scores[i]
        annotation["area"] = mask.sum()
        annotations.append(annotation)
    return annotations


def fast_process(
    annotations,
    image,
    scale,
    better_quality=False,
    mask_random_color=True,
    bbox=None,
    use_retina=True,
    withContours=True,
):
    if isinstance(annotations[0], dict):
        annotations = [annotation["segmentation"] for annotation in annotations]

    original_h = image.height
    original_w = image.width
    if better_quality:
        if isinstance(annotations[0], torch.Tensor):
            annotations = np.array(annotations.cpu())
        for i, mask in enumerate(annotations):
            mask = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
            )
            annotations[i] = cv2.morphologyEx(
                mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
            )
    annotations = np.asarray(annotations)
    inner_mask = fast_show_mask(
        annotations,
        plt.gca(),
        random_color=mask_random_color,
        bbox=bbox,
        retinamask=use_retina,
        target_height=original_h,
        target_width=original_w,
    )

    if withContours:
        contour_all = []
        temp = np.zeros((original_h, original_w, 1))
        for i, mask in enumerate(annotations):
            if type(mask) == dict:
                mask = mask["segmentation"]
            annotation = mask.astype(np.uint8)
            if use_retina == False:
                annotation = cv2.resize(
                    annotation,
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            contours, _ = cv2.findContours(annotation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_all.append(contour)
        cv2.drawContours(temp, contour_all, -1, (255, 255, 255), 2 // scale)
        color = np.array([0 / 255, 0 / 255, 255 / 255, 0.9])
        contour_mask = temp / 255 * color.reshape(1, 1, -1)

    image = image.convert("RGBA")
    overlay_inner = Image.fromarray((inner_mask * 255).astype(np.uint8), "RGBA")
    image.paste(overlay_inner, (0, 0), overlay_inner)

    if withContours:
        overlay_contour = Image.fromarray((contour_mask * 255).astype(np.uint8), "RGBA")
        image.paste(overlay_contour, (0, 0), overlay_contour)

    return image


# CPU post process
def fast_show_mask(
    annotation,
    ax,
    random_color=False,
    bbox=None,
    retinamask=True,
    target_height=960,
    target_width=960,
):
    mask_sum = annotation.shape[0]
    height = annotation.shape[1]
    weight = annotation.shape[2]
    areas = np.sum(annotation, axis=(1, 2))
    sorted_indices = np.argsort(areas)[::1]
    annotation = annotation[sorted_indices]

    index = (annotation != 0).argmax(axis=0)
    if random_color == True:
        color = np.random.random((mask_sum, 1, 1, 3))
    else:
        color = np.ones((mask_sum, 1, 1, 3)) * np.array([30 / 255, 144 / 255, 255 / 255])
    transparency = np.ones((mask_sum, 1, 1, 1)) * 0.6
    visual = np.concatenate([color, transparency], axis=-1)
    mask_image = np.expand_dims(annotation, -1) * visual

    mask = np.zeros((height, weight, 4))

    h_indices, w_indices = np.meshgrid(np.arange(height), np.arange(weight), indexing="ij")
    indices = (index[h_indices, w_indices], h_indices, w_indices, slice(None))

    mask[h_indices, w_indices, :] = mask_image[indices]
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="b", linewidth=1)
        )

    if retinamask == False:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return mask
