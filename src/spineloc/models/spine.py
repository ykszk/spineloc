import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from ..models.losses import SpineLoss


class MultiTaskSpineModel(pl.LightningModule):
    """
    Multi-task model using timm backbone and PyTorch Lightning.
    Predicts:
    1. Anatomical coordinates at each spatial position (with uncertainty)
    2. View type (frontal/lateral) with confidence
    3. Laterality (left/right facing for lateral views) with confidence
    """

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        in_channels=1,
        num_views=2,
        num_laterality=2,
        estimate_uncertainty=True,
        coord_weight=1.0,
        view_weight=0.5,
        laterality_weight=0.5,
        coord_loss_type="smooth_l1",
        learning_rate=1e-4,
        weight_decay=1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

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
            feature_dim = features[0].shape[1]

        # Coordinate regression head
        self.coord_head = nn.Conv2d(feature_dim, 2, kernel_size=1)

        # Uncertainty estimation for coordinates
        if estimate_uncertainty:
            self.coord_uncertainty_head = nn.Conv2d(feature_dim, 2, kernel_size=1)

        # Global pooling for classification
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # View classification head
        self.view_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_views),
        )

        # Laterality classification head
        self.laterality_classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_laterality),
        )

        # Loss function
        self.compute_loss = SpineLoss(
            coord_loss_type=coord_loss_type,
            coord_weight=coord_weight,
            view_weight=view_weight,
            laterality_weight=laterality_weight,
        )

        # Metrics tracking
        self.train_metrics = {"coord": [], "view": [], "laterality": []}
        self.val_metrics = {"coord": [], "view": [], "laterality": []}

    def forward(self, x):
        """
        Args:
            x: Input image (B, C, H, W)

        Returns:
            coord_map: Predicted coordinates (B, 2, h, w)
            coord_log_var: Coordinate uncertainty (B, 2, h, w) or None
            view_logits: View classification logits (B, num_views)
            laterality_logits: Laterality classification logits (B, num_laterality)
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
        laterality_logits = self.laterality_classifier(global_feat)

        return coord_map, coord_log_var, view_logits, laterality_logits

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
        images, coords, view_ids, laterality_ids = batch

        # Forward pass
        coord_map, coord_log_var, view_logits, laterality_logits = self(images)

        # Generate targets
        _, _, h, w = coord_map.shape
        target_coords = generate_coordinate_targets(coords, (h, w))

        # Compute loss
        total_loss, coord_loss, view_loss, lat_loss = self.compute_loss(
            coord_map,
            target_coords,
            view_logits,
            view_ids,
            laterality_logits,
            laterality_ids,
            coord_log_var,
        )

        # Compute metrics
        view_acc = (view_logits.argmax(dim=1) == view_ids).float().mean()

        lateral_mask = laterality_ids >= 0
        if lateral_mask.sum() > 0:
            lat_acc = (
                (laterality_logits[lateral_mask].argmax(dim=1) == laterality_ids[lateral_mask])
                .float()
                .mean()
            )
        else:
            lat_acc = torch.tensor(0.0)

        # Log metrics
        self.log(f"{log_prefix}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{log_prefix}/coord_loss", coord_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_loss", view_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/lat_loss", lat_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_acc", view_acc, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/lat_acc", lat_acc, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, log_prefix="train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, log_prefix="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


def generate_coordinate_targets(coords, feature_map_size):
    """Generate interpolated coordinate targets."""
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
