"""
Data augmentation transforms for spine X-ray images.
Uses albumentations for medical imaging augmentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..utils.transforms import Trim


def was_flipped(replay_dict):
    if "replay" not in replay_dict:
        return False
    return any(
        t["__class_fullname__"] == "HorizontalFlip" and t["applied"]
        for t in replay_dict["replay"]["transforms"]
    )


def get_train_transforms_with_replay(image_size=(256, 256)) -> A.ReplayCompose:
    """Transforms that track if horizontal flip was applied."""
    return A.ReplayCompose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),  # Track this
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size=(256, 256)) -> A.Compose:
    """
    Validation transforms (no augmentation).

    Args:
        image_size: (height, width) tuple

    Returns:
        albumentations.Compose object
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_predict_transforms(image_size=(256, 256)) -> A.Compose:
    """
    Prediction transforms (no augmentation).

    Args:
        image_size: (height, width) tuple

    Returns:
        albumentations.Compose object
    """
    return A.Compose(
        [
            Trim(min_p=0.2, max_p=0.8, p=1.0),
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_test_transforms(image_size=(256, 256)) -> A.Compose:
    """Test transforms (same as validation)."""
    return get_val_transforms(image_size)
