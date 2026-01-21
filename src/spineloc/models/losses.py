import torch
import torch.nn.functional as F


class SpineLoss:
    def __init__(
        self,
        coord_loss_type="l2",
        coord_weight=1.0,
        view_weight=1.0,
        laterality_weight=1.0,
    ):
        assert coord_loss_type in ["l2", "smooth_l1", "l1"], "Unsupported coordinate loss type."
        self.coord_loss_type = coord_loss_type
        self.coord_weight = coord_weight
        self.view_weight = view_weight
        self.laterality_weight = laterality_weight

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
        laterality_logits,
        target_laterality,
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

        # Laterality classification loss (only for lateral views)
        lateral_mask = target_laterality >= 0
        if lateral_mask.sum() > 0:
            laterality_loss = F.cross_entropy(
                laterality_logits[lateral_mask], target_laterality[lateral_mask]
            )
        else:
            laterality_loss = torch.tensor(0.0, device=pred_coords.device)

        # Total loss
        total_loss = (
            self.coord_weight * coord_loss
            + self.view_weight * view_loss
            + self.laterality_weight * laterality_loss
        )

        return total_loss, coord_loss, view_loss, laterality_loss
