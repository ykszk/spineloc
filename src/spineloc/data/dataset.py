from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from spineloc.data import transforms
from spineloc.utils import pylogger

from ..data.spine_radiograph import SpineRadiograph
from ..data.transforms import was_flipped
from ..utils.labelme import LabelMe

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SpineCoordinateDataset(Dataset):
    """
    Dataset that generates cropped spine x-rays with coordinate labels, view type, and laterality.
    """

    def __init__(
        self,
        spine_images: list[SpineRadiograph],
        aspect_ratios: list[Optional[float]] = [0.5, 0.75, 1.0, 1.5, 2.0, None],
        crop_height_min_max: tuple[float, float] = (10, 30),  # in normalized units
        transform: Optional[A.Compose] = None,
    ):
        self.spine_images = spine_images
        self.aspect_ratios = aspect_ratios
        self.num_crops_per_image = len(aspect_ratios)
        self.transform = transform
        self.crop_height_min_max = crop_height_min_max

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

        spine_radiograph: SpineRadiograph = self.spine_images[img_idx]
        assert isinstance(spine_radiograph, SpineRadiograph)
        if aspect_ratio is None:
            # no cropping
            crop_img = spine_radiograph.load_image()
            pixel_coords = np.array(
                [[0, 0], [spine_radiograph.image_wh[0], spine_radiograph.image_wh[1]]]
            )
            anat_coords = spine_radiograph.pix_to_anat(pixel_coords).ravel()
        else:
            crop_size_anat = (
                int(crop_height_anat * aspect_ratio),
                int(crop_height_anat),
            )

            W, H = spine_radiograph.image_wh
            [crop_w, crop_h] = crop_size_anat * spine_radiograph.units

            # Random crop position
            max_y = max(0, H - crop_h)
            max_x = max(0, W - crop_w)
            start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
            start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0
            end_y = int(start_y + crop_h)
            end_x = int(start_x + crop_w)
            end_y = min(end_y, H)
            end_x = min(end_x, W)

            crop_img, anat_coords = spine_radiograph.crop(start_x, start_y, end_x, end_y)

        view = spine_radiograph.view
        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=crop_img)
            crop_img = transformed["image"]
            if was_flipped(transformed):
                view = view.flip()
                anat_coords[0], anat_coords[2] = -anat_coords[2], -anat_coords[0]  # Flip x coords

        # Encode view
        view_id = view.value

        return crop_img, anat_coords, view_id


class SpineDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for spine coordinate dataset."""

    def __init__(
        self,
        data_dirs=["data/"],
        cervical_data_dirs=[],
        image_size=(256, 256),
        batch_size=64,
        num_workers=0,
        val_split=0.2,
        shuffle_before_split=True,
        pin_memory=False,
    ):
        super().__init__()
        self.data_dirs = [Path(d) for d in data_dirs]
        self.cervical_data_dirs = [Path(d) for d in cervical_data_dirs]
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.shuffle_before_split = shuffle_before_split
        self.pin_memory = pin_memory
        self.data_loaded = False

    def setup(self, stage: Optional[str] = None) -> None:
        spine_images = []
        all_data_dirs = [(d, False) for d in self.data_dirs]
        if self.cervical_data_dirs:
            all_data_dirs.extend([(d, True) for d in self.cervical_data_dirs])
        for data_dir, is_cervical in all_data_dirs:
            data_dir = data_dir
            log.info(f"Loading data from {data_dir}")
            sub_images = []
            # ndjson
            if data_dir.suffix == ".ndjson":
                with open(data_dir, "r") as f:
                    for line in f:
                        lm = LabelMe.model_validate_json(line.strip())
                        if is_cervical:
                            sr = SpineRadiograph.from_cervical_labelme(lm)
                        else:
                            sr = SpineRadiograph.from_labelme(lm)
                        sub_images.append(sr)
                log.info(f"Loaded {len(sub_images)} json lines from {data_dir}")
                spine_images.extend(sub_images)
                continue
            if not data_dir.is_dir():
                raise ValueError(f"Data directory {data_dir} is not a directory or ndjson.")
            for json_path in data_dir.glob("*.json"):
                if is_cervical:
                    sr = SpineRadiograph.from_cervical_labelme_file(json_path)
                else:
                    sr = SpineRadiograph.from_labelme_file(json_path)
                sub_images.append(sr)
            log.info(f"Loaded {len(sub_images)} json files from {data_dir}")
            spine_images.extend(sub_images)
        log.info(f"Total loaded spine images: {len(spine_images)}")

        num_val = int(len(spine_images) * self.val_split)
        num_train = len(spine_images) - num_val
        log.info(f"Train/val split: {num_train}/{num_val}")

        if self.shuffle_before_split:
            np.random.shuffle(spine_images)
        train_images = spine_images[:num_train]
        val_images = spine_images[num_train:]

        self.train_dataset = SpineCoordinateDataset(
            train_images,
            transform=transforms.get_train_transforms_with_replay(self.image_size),
        )
        self.val_dataset = SpineCoordinateDataset(
            val_images,
            transform=transforms.get_val_transforms(self.image_size),
        )
        self.data_loaded = True

    def train_dataloader(self):
        if not self.data_loaded:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        if not self.data_loaded:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class InferenceImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_paths = sorted(
            [
                p
                for p in Path(image_dir).iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        log.info(f"Found {len(self.image_paths)} images in {image_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("L")  # Convert to grayscale
        if isinstance(self.transform, A.Compose):
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = self.transform(image)

        return image
