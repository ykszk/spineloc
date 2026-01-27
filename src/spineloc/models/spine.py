import timm
import torch
import torch.nn as nn


class MultiTaskSpineNet(torch.nn.Module):
    """
    Multi-task model using timm backbone
    Predicts:
    1. Anatomical coordinates at each spatial position (with uncertainty)
    2. View type (frontal/lateral_left/lateral_right)
    """

    def __init__(
        self,
        backbone_name="mobilenetv3_small_100",
        pretrained=True,
        in_channels=1,
        num_views=3,
        estimate_uncertainty=True,
        cls_mid_dim=128,
    ):
        super().__init__()

        self.estimate_uncertainty = estimate_uncertainty

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
        if self.estimate_uncertainty:
            coord_log_var = self.coord_uncertainty_head(features)
        else:
            coord_log_var = None

        # Global features for classification
        global_feat = self.global_pool(features)
        global_feat = global_feat.view(global_feat.size(0), -1)

        # Classifications
        view_logits = self.view_classifier(global_feat)

        return coord_map, coord_log_var, view_logits

    @staticmethod
    def predict_bounding_box(coord_map, coord_log_var=None):
        """Extract bounding box using min/max operations."""
        if coord_map.dim() == 3:  # (2, h, w)
            B = 1
            coord_flat = coord_map.view(1, 2, -1)
        elif coord_map.dim() == 4:  # (B, 2, h, w)
            B = coord_map.shape[0]
            coord_flat = coord_map.view(B, 2, -1)
        else:
            raise ValueError("coord_map must be 3D or 4D tensor.")

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
