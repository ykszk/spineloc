from typing import Tuple

import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def trim_params(arr: np.ndarray, min_p=0.2, max_p=0.8) -> Tuple[int, int, int, int]:
    """
    Calculate cropping parameters for trimming nearly empty space from the array based on pixel value thresholds.
    """
    assert arr.ndim == 2, "Input array must be 2D (grayscale image)."
    min_value, max_value = np.min(arr), np.max(arr)
    min_allowed = min_p * (max_value - min_value)
    max_allowed = max_p * (max_value - min_value)
    inside = (arr >= min_allowed) & (arr <= max_allowed)
    coords = np.argwhere(inside)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    return y0, x0, y1, x1


def trim(img: np.ndarray, min_p=0.2, max_p=0.8) -> np.ndarray:
    """
    Trim nearly empty space from the image based on pixel value thresholds.
    """
    if img.ndim == 2:
        arr = img
    elif img.ndim == 3:
        arr = np.mean(img, axis=2)  # convert to grayscale
    else:
        raise ValueError("Input image must be 2D or 3D array.")

    y0, x0, y1, x1 = trim_params(arr, min_p=min_p, max_p=max_p)
    return arr[y0:y1, x0:x1]


class Trim(ImageOnlyTransform):
    """
    Albumentations transform that trims nearly empty space from the image
    based on pixel value thresholds.

    Args:
        min_p (float): Minimum percentile threshold (0.0-1.0). Default: 0.2
        max_p (float): Maximum percentile threshold (0.0-1.0). Default: 0.8
        always_apply (bool): Whether to always apply the transform. Default: True
        p (float): Probability of applying the transform. Default: 1.0
    """

    def __init__(self, min_p=0.2, max_p=0.8, p=1.0):
        super().__init__(p=p)
        self.min_p = min_p
        self.max_p = max_p

    def apply(self, img, **params):
        return trim(img, min_p=self.min_p, max_p=self.max_p)

    def get_transform_init_args_names(self):
        return ("min_p", "max_p")
