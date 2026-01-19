import numpy as np
from PIL import Image

def crop_to_mask(image: Image.Image, mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))


def categorize_masks_by_area(masks, size_mode="mixed"):
    areas = [mask.sum() for mask in masks]

    if size_mode == "s":
        return [i for i, a in enumerate(areas) if a < 2_000]
    if size_mode == "m":
        return [i for i, a in enumerate(areas) if 2_000 <= a < 20_000]
    if size_mode == "l":
        return [i for i, a in enumerate(areas) if a >= 20_000]

    return list(range(len(masks)))
