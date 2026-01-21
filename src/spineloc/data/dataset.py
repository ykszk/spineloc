import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class SpineCoordinateDataset(Dataset):
    """
    Dataset that generates cropped spine x-rays with coordinate labels, view type, and laterality.
    """

    def __init__(
        self,
        full_spine_images,  # List of full spine x-ray images
        full_spine_coords,  # List of (top_y, top_x, bottom_y, bottom_x) for each image
        view_labels,  # List of view types: 'frontal' or 'lateral'
        laterality_labels,  # List of facing direction: 'left', 'right', or None (for frontal)
        crop_size=(512, 256),  # (height, width)
        num_crops_per_image=10,
        transform=None,
    ):
        self.full_images = full_spine_images
        self.full_coords = full_spine_coords
        self.view_labels = view_labels
        self.laterality_labels = laterality_labels
        self.crop_size = crop_size
        self.num_crops_per_image = num_crops_per_image
        self.transform = transform

        # Encode labels
        self.view_to_id = {"frontal": 0, "lateral": 1}
        self.laterality_to_id = {"left": 0, "right": 1, None: -1}

    def __len__(self):
        return len(self.full_images) * self.num_crops_per_image

    def __getitem__(self, idx):
        # Determine which full image and which crop
        img_idx = idx // self.num_crops_per_image

        full_img = self.full_images[img_idx]
        full_top_y, full_top_x, full_bottom_y, full_bottom_x = self.full_coords[img_idx]
        view = self.view_labels[img_idx]
        laterality = self.laterality_labels[img_idx]

        # Get image dimensions
        if full_img.ndim == 2:
            H, W = full_img.shape
        else:
            _, H, W = full_img.shape

        # Generate random crop
        crop_h, crop_w = self.crop_size

        # Random crop position
        max_y = max(0, H - crop_h)
        max_x = max(0, W - crop_w)
        start_y = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        start_x = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        # Extract crop
        if full_img.ndim == 2:
            crop_img = full_img[start_y : start_y + crop_h, start_x : start_x + crop_w]
            crop_img = crop_img[np.newaxis, ...]
        else:
            crop_img = full_img[:, start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Calculate anatomical coordinates for this crop
        pixel_height = full_bottom_y - full_top_y
        pixel_width = full_bottom_x - full_top_x

        crop_top_y = full_top_y + (start_y / H) * pixel_height
        crop_top_x = full_top_x + (start_x / W) * pixel_width
        crop_bottom_y = full_top_y + ((start_y + crop_h) / H) * pixel_height
        crop_bottom_x = full_top_x + ((start_x + crop_w) / W) * pixel_width

        coords = torch.tensor(
            [crop_top_y, crop_top_x, crop_bottom_y, crop_bottom_x], dtype=torch.float32
        )

        # Encode view and laterality
        view_id = torch.tensor(self.view_to_id[view], dtype=torch.long)
        laterality_id = torch.tensor(self.laterality_to_id[laterality], dtype=torch.long)

        # Convert to tensor
        crop_img = torch.from_numpy(crop_img).float()

        # Apply transforms if any
        if self.transform:
            crop_img = self.transform(crop_img)

        return crop_img, coords, view_id, laterality_id


class SpineDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for spine coordinate dataset."""

    def __init__(
        self,
        train_images,
        train_coords,
        train_views,
        train_laterality,
        val_images=None,
        val_coords=None,
        val_views=None,
        val_laterality=None,
        crop_size=(512, 256),
        num_crops_per_image=10,
        batch_size=8,
        num_workers=4,
        val_split=0.2,
    ):
        super().__init__()
        self.train_images = train_images
        self.train_coords = train_coords
        self.train_views = train_views
        self.train_laterality = train_laterality

        self.val_images = val_images
        self.val_coords = val_coords
        self.val_views = val_views
        self.val_laterality = val_laterality

        self.crop_size = crop_size
        self.num_crops_per_image = num_crops_per_image
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        # If validation data not provided, split from training
        if self.val_images is None:
            n_train = len(self.train_images)
            n_val = int(n_train * self.val_split)
            indices = np.random.permutation(n_train)

            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            self.train_dataset = SpineCoordinateDataset(
                [self.train_images[i] for i in train_idx],
                [self.train_coords[i] for i in train_idx],
                [self.train_views[i] for i in train_idx],
                [self.train_laterality[i] for i in train_idx],
                crop_size=self.crop_size,
                num_crops_per_image=self.num_crops_per_image,
            )

            self.val_dataset = SpineCoordinateDataset(
                [self.train_images[i] for i in val_idx],
                [self.train_coords[i] for i in val_idx],
                [self.train_views[i] for i in val_idx],
                [self.train_laterality[i] for i in val_idx],
                crop_size=self.crop_size,
                num_crops_per_image=self.num_crops_per_image,
            )
        else:
            self.train_dataset = SpineCoordinateDataset(
                self.train_images,
                self.train_coords,
                self.train_views,
                self.train_laterality,
                crop_size=self.crop_size,
                num_crops_per_image=self.num_crops_per_image,
            )

            self.val_dataset = SpineCoordinateDataset(
                self.val_images,
                self.val_coords,
                self.val_views,
                self.val_laterality,
                crop_size=self.crop_size,
                num_crops_per_image=self.num_crops_per_image,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
