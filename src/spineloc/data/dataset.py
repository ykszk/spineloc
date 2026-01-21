from typing import Optional

import albumentations as A
import numpy as np
from torch.utils.data import Dataset

from ..data.spine_radiograph import SpineRadiograph
from ..data.transforms import was_flipped


class SpineCoordinateDataset(Dataset):
    """
    Dataset that generates cropped spine x-rays with coordinate labels, view type, and laterality.
    """

    def __init__(
        self,
        spine_images: list[SpineRadiograph],
        aspect_ratios: list[float] = [0.5, 0.75, 1.0, 1.5, 2.0],
        crop_height_min_max: tuple[float, float] = (10, 20),  # in normalized units
        transform: Optional[A.Compose] = None,
    ):
        self.spine_images = spine_images
        self.aspect_ratios = aspect_ratios
        self.num_crops_per_image = len(aspect_ratios)
        self.transform = transform
        self.crop_height_min_max = crop_height_min_max

        # Encode labels
        self.view_to_id = {"frontal": 0, "lateral": 1}
        self.laterality_to_id = {"left": 0, "right": 1, None: -1}

    def __len__(self):
        return len(self.spine_images) * self.num_crops_per_image

    def __getitem__(self, idx):
        # Determine which full image and which crop
        img_idx = idx // self.num_crops_per_image
        aspect_ratio = self.aspect_ratios[idx % self.num_crops_per_image]
        # random crop size
        crop_height_anat = np.random.uniform(
            self.crop_height_min_max[0], self.crop_height_min_max[1]
        )  # in normalized units
        crop_size_anat = (
            int(crop_height_anat * aspect_ratio),
            int(crop_height_anat),
        )

        spine_radiograph: SpineRadiograph = self.spine_images[img_idx]
        assert isinstance(spine_radiograph, SpineRadiograph)
        W, H = spine_radiograph.image_wh
        [crop_w, crop_h] = crop_size_anat * spine_radiograph.units

        # Random crop position
        max_y = max(0, H - crop_h)
        max_x = max(0, W - crop_w)
        start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
        end_y = int(start_y + crop_h)
        end_x = int(start_x + crop_w)

        crop_img, anat_coords = spine_radiograph.crop(start_y, start_x, end_y, end_x)
        view = spine_radiograph.view

        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=crop_img)
            crop_img = transformed["image"]
            if was_flipped(transformed):
                view = view.flip()

        # Encode view
        view_id = view.value

        return crop_img, anat_coords, view_id
