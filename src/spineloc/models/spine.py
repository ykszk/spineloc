from typing import Any, Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule

from spineloc.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SpineLoss:
    def __init__(
        self,
        coord_loss_type="l2",
        coord_weight=1.0,
        view_weight=1.0,
    ):
        assert coord_loss_type in ["l2", "smooth_l1", "l1"], "Unsupported coordinate loss type."
        self.coord_loss_type = coord_loss_type
        self.coord_weight = coord_weight
        self.view_weight = view_weight

        # coordinate loss function without uncertainty
        if self.coord_loss_type == "smooth_l1":
            self.coord_loss_func = F.smooth_l1_loss
        elif self.coord_loss_type == "l1":
            self.coord_loss_func = F.l1_loss
        else:
            self.coord_loss_func = F.mse_loss

        # coordinate loss function with uncertainty
        if self.coord_loss_type == "l2":

            def coord_loss_with_uncertainty(pred_coords, target_coords, coord_log_var):
                precision = torch.exp(-coord_log_var)
                sq_diff = (pred_coords - target_coords) ** 2
                return torch.mean(0.5 * precision * sq_diff + 0.5 * coord_log_var)

            self.coord_loss_with_uncertainty = coord_loss_with_uncertainty
        else:

            def coord_loss_with_uncertainty(pred_coords, target_coords, coord_log_var):
                precision = torch.exp(-coord_log_var)
                if self.coord_loss_type == "smooth_l1":
                    diff = F.smooth_l1_loss(pred_coords, target_coords, reduction="none")
                else:
                    diff = torch.abs(pred_coords - target_coords)
                return torch.mean(precision * diff + 0.5 * coord_log_var)

            self.coord_loss_with_uncertainty = coord_loss_with_uncertainty

    def __call__(
        self,
        pred_coords,
        target_coords,
        view_logits,
        target_views,
        coord_log_var=None,
    ):
        """Compute multi-task loss."""
        # Coordinate loss with uncertainty
        if coord_log_var is not None:
            coord_loss = self.coord_loss_with_uncertainty(pred_coords, target_coords, coord_log_var)
        else:
            coord_loss = self.coord_loss_func(pred_coords, target_coords)

        # View classification loss
        view_loss = F.cross_entropy(view_logits, target_views)

        total_loss = self.coord_weight * coord_loss + self.view_weight * view_loss

        return total_loss, coord_loss, view_loss


class MultiTaskSpineModel(LightningModule):
    """
    Multi-task model using timm backbone and PyTorch Lightning.
    Predicts:
    1. Anatomical coordinates at each spatial position (with uncertainty)
    2. View type (frontal/lateral_left/lateral_right) with confidence
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        backbone_name="resnet50",
        pretrained=True,
        in_channels=1,
        num_views=3,
        estimate_uncertainty=True,
        cls_mid_dim=128,
        loss=SpineLoss(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])

        # Create timm backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 512, 256)
            features = self.backbone(dummy_input)
            feature_dim = features[0].shape[0]
            log.info(f"Backbone feature dimension: {feature_dim}")

        # Coordinate regression head
        self.coord_head = nn.Conv2d(feature_dim, 2, kernel_size=1)

        # Uncertainty estimation for coordinates
        if estimate_uncertainty:
            self.coord_uncertainty_head = nn.Conv2d(feature_dim, 2, kernel_size=1)

        # Global pooling for classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # View classification head
        self.view_classifier = nn.Sequential(
            nn.Linear(feature_dim, cls_mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(cls_mid_dim, num_views),
        )

        # Loss function
        self.loss = loss

        # Metrics tracking
        self.train_metrics = {"coord": [], "view": []}
        self.val_metrics = {"coord": [], "view": []}

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            coord_map: Predicted coordinates (B, 2, h, w)
            coord_log_var: Coordinate uncertainty (B, 2, h, w) or None
            view_logits: View classification logits (B, num_views)
        """
        # Extract features using timm backbone
        features = self.backbone(x)  # (B, feature_dim, h, w)

        # Coordinate prediction
        coord_map = self.coord_head(features)

        # Coordinate uncertainty
        if self.hparams.estimate_uncertainty:
            coord_log_var = self.coord_uncertainty_head(features)
        else:
            coord_log_var = None

        # Global features for classification
        global_feat = self.global_pool(features)
        global_feat = global_feat.view(global_feat.size(0), -1)

        # Classifications
        view_logits = self.view_classifier(global_feat)

        return coord_map, coord_log_var, view_logits

    def predict_bounding_box(self, coord_map, coord_log_var=None):
        """Extract bounding box using min/max operations."""
        B = coord_map.shape[0]
        coord_flat = coord_map.view(B, 2, -1)

        top_y = coord_flat[:, 0, :].min(dim=1)[0]
        bottom_y = coord_flat[:, 0, :].max(dim=1)[0]
        top_x = coord_flat[:, 1, :].min(dim=1)[0]
        bottom_x = coord_flat[:, 1, :].max(dim=1)[0]

        boxes = torch.stack([top_y, top_x, bottom_y, bottom_x], dim=1)

        if coord_log_var is None:
            return boxes, None

        var = torch.exp(coord_log_var)
        var_flat = var.view(B, 2, -1)
        y_uncertainty = var_flat[:, 0, :].mean(dim=1)
        x_uncertainty = var_flat[:, 1, :].mean(dim=1)
        uncertainties = torch.stack(
            [y_uncertainty, x_uncertainty, y_uncertainty, x_uncertainty], dim=1
        )
        return boxes, uncertainties

    def shared_step(self, batch, batch_idx, log_prefix: str):
        images, coords, view_ids = batch

        # Forward pass
        coord_map, coord_log_var, view_logits = self(images)

        # Generate targets
        _, _, h, w = coord_map.shape
        target_coords = generate_coordinate_targets(coords, (h, w))

        # Compute loss
        (
            total_loss,
            coord_loss,
            view_loss,
        ) = self.loss(
            coord_map,
            target_coords,
            view_logits,
            view_ids,
            coord_log_var,
        )

        # Compute metrics
        view_acc = (view_logits.argmax(dim=1) == view_ids).float().mean()

        # Log metrics
        self.log(f"{log_prefix}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{log_prefix}/coord_loss", coord_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_loss", view_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_acc", view_acc, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, log_prefix="val")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def generate_coordinate_targets(coords, feature_map_size):
    """Generate interpolated coordinate targets.

    Args:
        coords: Ground truth bounding boxes (B, 4) in (top_y, top_x, bottom_y, bottom_x) format
        feature_map_size: Size of the feature map (h, w)

    Returns:
        targets: Interpolated coordinate targets (B, 2, h, w)
    """
    B = coords.shape[0]
    h, w = feature_map_size
    device = coords.device

    y_grid = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(B, h, w)
    x_grid = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(B, h, w)

    top_y = coords[:, 0].view(B, 1, 1)
    top_x = coords[:, 1].view(B, 1, 1)
    bottom_y = coords[:, 2].view(B, 1, 1)
    bottom_x = coords[:, 3].view(B, 1, 1)

    target_y = top_y + (bottom_y - top_y) * y_grid
    target_x = top_x + (bottom_x - top_x) * x_grid

    targets = torch.stack([target_y, target_x], dim=1)
    return targets
