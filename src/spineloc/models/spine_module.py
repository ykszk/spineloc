from typing import Any, Dict

import torch
import torch.nn.functional as F
from lightning import LightningModule

from spineloc.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


class SpineLoss:
    def __init__(
        self,
        coord_loss_type="l2",
        coord_weight=1.0,
        aux_weight=1.0,
    ):
        assert coord_loss_type in ["l2", "smooth_l1", "l1"], "Unsupported coordinate loss type."
        self.coord_loss_type = coord_loss_type
        self.coord_weight = coord_weight
        self.aux_weight = aux_weight

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
        photometric_logits,
        target_photometric,
        rotation_90_logits,
        target_rotation_90,
        rotation_angle,
        target_rotation_angle,
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
        photometric_loss = F.binary_cross_entropy_with_logits(
            photometric_logits.squeeze(), target_photometric.float()
        )
        rotation_90_loss = F.cross_entropy(rotation_90_logits, target_rotation_90)
        rotation_angle_loss = F.mse_loss(rotation_angle.squeeze(), target_rotation_angle.float())

        total_loss = self.coord_weight * coord_loss + self.aux_weight * (
            view_loss + photometric_loss + rotation_90_loss + rotation_angle_loss
        )

        return (
            total_loss,
            coord_loss,
            view_loss,
            photometric_loss,
            rotation_90_loss,
            rotation_angle_loss,
        )


class MultiTaskSpineModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loss=SpineLoss(),
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.net = net
        self.loss = loss

    def forward(self, x):
        return self.net(x)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def shared_step(self, batch, batch_idx, log_prefix: str):
        images, coords, rad_chars = batch
        view_ids = rad_chars["view"].long()
        pi = rad_chars["photometric_interpretation"]
        rotation_90 = rad_chars["rotation_90"]
        rotation_angle = rad_chars["rotation_angle"]

        # Forward pass
        (
            coord_map,
            coord_log_var,
            view_logits,
            photometric_logits,
            rotation_90_logits,
            rotation_angle,
        ) = self(images)

        # Generate targets
        _, _, h, w = coord_map.shape
        target_coords = generate_coordinate_targets(coords, (h, w))

        # Compute loss
        (
            total_loss,
            coord_loss,
            view_loss,
            photometric_loss,
            rotation_90_loss,
            rotation_angle_loss,
        ) = self.loss(
            coord_map,
            target_coords,
            view_logits,
            view_ids,
            photometric_logits,
            pi,
            rotation_90_logits,
            rotation_90,
            rotation_angle,
            rotation_angle,
            coord_log_var,
        )

        # Compute metrics
        view_acc = (view_logits.argmax(dim=1) == view_ids).float().mean()

        # Log metrics
        self.log(f"{log_prefix}/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{log_prefix}/coord_loss", coord_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_loss", view_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/view_acc", view_acc, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/photometric_loss", photometric_loss, on_step=False, on_epoch=True)
        self.log(f"{log_prefix}/rotation_90_loss", rotation_90_loss, on_step=False, on_epoch=True)
        self.log(
            f"{log_prefix}/rotation_angle_loss", rotation_angle_loss, on_step=False, on_epoch=True
        )

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


class SpineViewModule(LightningModule):
    """
    View only classification module for testing purposes.
    """

    def __init__(
        self,
        net: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        self.net = net

    def forward(self, x):
        """
        Return view logits.
        """
        return self.net(x)[2]

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        # Use dataloader_idx to handle different test scenarios
        return {"test_loss": loss}


def generate_coordinate_targets(coords, feature_map_size):
    """Generate interpolated coordinate targets.

    Args:
        coords: Ground truth bounding boxes (B, 4) in (top_x, top_y, bottom_x, bottom_y) format
        feature_map_size: Size of the feature map (h, w)

    Returns:
        targets: Interpolated coordinate targets (B, 2, h, w)
    """
    B = coords.shape[0]
    h, w = feature_map_size
    device = coords.device

    y_grid = torch.linspace(0, 1, h, device=device).view(1, h, 1).expand(B, h, w)
    x_grid = torch.linspace(0, 1, w, device=device).view(1, 1, w).expand(B, h, w)

    top_x = coords[:, 0].view(B, 1, 1)
    top_y = coords[:, 1].view(B, 1, 1)
    bottom_x = coords[:, 2].view(B, 1, 1)
    bottom_y = coords[:, 3].view(B, 1, 1)

    target_y = top_y + (bottom_y - top_y) * y_grid
    target_x = top_x + (bottom_x - top_x) * x_grid

    targets = torch.stack([target_x, target_y], dim=1)
    return targets


class InferenceModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = net

    def forward(self, x):
        return self.net(x)
        return self.net(x)
