from enum import Enum
from pathlib import Path

import numpy as np

from spineloc.utils.labelme import LabelMe


def load_image(image_path: str) -> np.ndarray:
    """Load image from the given path as a numpy array."""
    from PIL import Image

    image = Image.open(image_path).convert("L")
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return np.array(image)


class RadiographView(Enum):
    FRONTAL = 0
    LATERAL_LEFT = 1
    LATERAL_RIGHT = 2

    def flip(self) -> "RadiographView":
        """Return the view after a horizontal flip."""
        if self == RadiographView.FRONTAL:
            return RadiographView.FRONTAL
        elif self == RadiographView.LATERAL_LEFT:
            return RadiographView.LATERAL_RIGHT
        elif self == RadiographView.LATERAL_RIGHT:
            return RadiographView.LATERAL_LEFT
        else:
            raise ValueError(f"Unknown RadiographView: {self}")


class SpineRadiograph:
    """
    Spine radiograph with patient specific anatomical coordinate system.

    Coordinate system is defined by:
    origin: mid-point between C7 and S1 vertebrae in image coordinates (pixels)
    units: average width and height of vertebrae in pixels
    """

    def __init__(
        self,
        image_path: str,
        image_wh: np.ndarray,
        origin: np.ndarray,
        units: np.ndarray,
        view: RadiographView,
    ):
        self.image_path = image_path
        self.image_wh = image_wh
        self.origin = origin  # (x, y) in pixels
        self.units = units  # (unit_x, unit_y) in pixels
        self.view = view

    def crop(
        self, top_x: int, top_y: int, bottom_x: int, bottom_y: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Crop the SpineRadiograph to the specified bounding box.

        The coordinate system is adjusted accordingly.

        Returns:
            (cropped_image, coordinates)
            cropped_image: Cropped image as numpy array
            coordinates: New coordinates (top_x, top_y, bottom_x, bottom_y) in anatomical units
        """

        image = load_image(self.image_path)
        cropped_image = image[top_y:bottom_y, top_x:bottom_x]
        top_left = (np.array([top_x, top_y]) - self.origin) / self.units
        bottom_right = (np.array([bottom_x, bottom_y]) - self.origin) / self.units
        return cropped_image, np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]])

    def pix_to_anat(self, pix_coords: np.ndarray) -> np.ndarray:
        """
        Convert pixel coordinates to anatomical coordinates.

        Args:
            pix_coords: Pixel coordinates as (N, 2) array

        Returns:
            anat_coords: Anatomical coordinates as (N, 2) array
        """
        shifted = pix_coords - self.origin  # (N, 2)
        anat_coords = shifted / self.units  # (N, 2)
        return anat_coords

    def anat_to_pix(self, anat_coords: np.ndarray) -> np.ndarray:
        """
        Convert anatomical coordinates to pixel coordinates.

        Args:
            anat_coords: Anatomical coordinates as (N, 2) array

        Returns:
            pix_coords: Pixel coordinates as (N, 2) array
        """
        scaled = anat_coords * self.units  # (N, 2)
        pix_coords = scaled + self.origin  # (N, 2)
        return pix_coords

    def to_labelme(self, axis_length=5) -> LabelMe:
        """
        Convert SpineRadiograph to LabelMe format to visualize coordinate system.

        Creates origin and unit vectors.
        """
        shapes = []
        shapes.append(
            LabelMe.Shape(
                label="origin",
                points=[[float(self.origin[0]), float(self.origin[1])]],
                group_id=None,
                shape_type="point",
                flags={},
            )
        )
        # Unit vector along x-axis
        shapes.append(
            LabelMe.Shape(
                label="unit_x",
                points=[
                    [float(self.origin[0]), float(self.origin[1])],
                    [float(self.origin[0] + axis_length * self.units[0]), float(self.origin[1])],
                ],
                group_id=None,
                shape_type="line",
                flags={},
            )
        )
        # Unit vector along y-axis
        shapes.append(
            LabelMe.Shape(
                label="unit_y",
                points=[
                    [float(self.origin[0]), float(self.origin[1])],
                    [float(self.origin[0]), float(self.origin[1] + axis_length * self.units[1])],
                ],
                group_id=None,
                shape_type="line",
                flags={},
            )
        )
        lm = LabelMe(
            version="spineloc",
            shapes=shapes,
            imagePath=self.image_path,
            imageData=None,
            imageHeight=0,
            imageWidth=0,
            flags={},
        )
        return lm

    @classmethod
    def from_labelme(cls, lm: LabelMe) -> "SpineRadiograph":
        """
        Create a SpineRadiograph instance from LabelMe.

        LabelMe must contain points for C7 and S1 vertebrae to define the coordinate system.
        """

        shape_dict = lm.into_shape_dict()
        points = shape_dict.shapes["point"]
        n_tl = len(points["TL"])
        n_tr = len(points["TR"])
        n_bl = len(points["BL"])
        n_br = len(points["BR"])
        if n_tl != n_tr:
            raise ValueError("Number of TL and TR points must be the same.")
        if n_bl != n_br:
            raise ValueError("Number of BL and BR points must be the same.")
        if n_tl != n_bl + 1:  # +1 because S1 has only TL and TR points
            raise ValueError("Number of top and bottom vertebrae points do not match.")

        corner_points = np.array(
            [points["TL"][:-1], points["TR"][:-1], points["BL"], points["BR"]],
        )[:, :, 0]  # (tl/tr/bl/br, vertebrae, xy)
        corner_points = corner_points.transpose(1, 0, 2)  # (vertebrae, tl/tr/bl/br, xy)

        c7 = corner_points[0]
        c7_center = c7.mean(axis=0)
        s_center = np.array([*points["TL"][-1], *points["TR"][-1]]).mean(axis=0)
        origin = (c7_center + s_center) / 2.0
        mean_width = np.mean(
            np.concatenate(
                [
                    np.linalg.norm(corner_points[:, 1] - corner_points[:, 0], axis=1),  # TR - TL
                    np.linalg.norm(corner_points[:, 3] - corner_points[:, 2], axis=1),  # BR - BL
                ],
            )
        )
        mean_height = np.mean(
            np.concatenate(
                [
                    np.linalg.norm(corner_points[:, 2] - corner_points[:, 0], axis=1),  # BL - TL
                    np.linalg.norm(corner_points[:, 3] - corner_points[:, 1], axis=1),  # BR - TR
                ],
            )
        )
        units = np.array([mean_width, mean_height])
        image_wh = np.array([lm.imageWidth, lm.imageHeight])
        frontal = lm.flags.get("frontal", False)
        if frontal:
            view = RadiographView.FRONTAL
        else:
            lateral = lm.flags.get("lateral", False)
            if not lateral:
                raise ValueError("LabelMe flags must indicate 'frontal' or 'lateral' view.")
            right = lm.flags.get("right", False)
            if right:
                view = RadiographView.LATERAL_RIGHT
            else:
                view = RadiographView.LATERAL_LEFT
        return cls(lm.imagePath, image_wh, origin, units, view)

    @classmethod
    def from_labelme_file(cls, json_path: Path | str) -> "SpineRadiograph":
        """Create SpineRadiograph from LabelMe JSON file."""
        json_path = Path(json_path)
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        return cls.from_labelme(lm)
