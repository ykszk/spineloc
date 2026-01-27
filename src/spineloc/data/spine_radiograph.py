from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..utils.labelme import LabelMe


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


def determine_radiograph_view(flags: dict) -> RadiographView:
    frontal = flags.get("frontal", False)
    if frontal:
        view = RadiographView.FRONTAL
    else:
        lateral = flags.get("lateral", False)
        if not lateral:
            raise ValueError("LabelMe flags must indicate 'frontal' or 'lateral' view.")
        right = flags.get("right", False)
        if right:
            view = RadiographView.LATERAL_RIGHT
        else:
            view = RadiographView.LATERAL_LEFT
    return view


class SpineRadiograph:
    """
    Spine radiograph with patient specific anatomical coordinate system.

    Coordinate system is defined by:
    origin: mid-point between C7 and L4 vertebrae in image coordinates (pixels)
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
        self.origin = origin
        """(x, y) in pixels"""
        self.units = units
        """(unit_x, unit_y) in pixels"""
        self.view = view

    def load_image(self) -> np.ndarray:
        """Load the radiograph image as a numpy array."""
        return load_image(self.image_path)

    def flip(self) -> "SpineRadiograph":
        """
        Return a new SpineRadiograph instance with horizontal flip applied.
        Note that image is not flipped here.

        The view and origin are adjusted accordingly.
        """
        flipped_origin_x = self.image_wh[0] - self.origin[0]
        flipped_origin = np.array([flipped_origin_x, self.origin[1]])
        flipped_view = self.view.flip()
        return SpineRadiograph(
            image_path=self.image_path,
            image_wh=self.image_wh,
            origin=flipped_origin,
            units=self.units,
            view=flipped_view,
        )

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
            pix_coords: Pixel coordinates as (N, xy) array

        Returns:
            anat_coords: Anatomical coordinates as (N, xy) array
        """
        shifted = pix_coords - self.origin  # (N, xy)
        anat_coords = shifted / self.units  # (N, xy)
        return anat_coords

    def anat_to_pix(self, anat_coords: np.ndarray) -> np.ndarray:
        """
        Convert anatomical coordinates to pixel coordinates.

        Args:
            anat_coords: Anatomical coordinates as (N, xy) array

        Returns:
            pix_coords: Pixel coordinates as (N, xy) array
        """
        scaled = anat_coords * self.units  # (N, xy)
        pix_coords = scaled + self.origin  # (N, xy)
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
        if n_tl < 18:
            # C7, T1-T12, L1-L4, S1 => 17 vertebrae points
            # L5 and L6 are optional
            raise ValueError(
                "At least 18 vertebrae points (C7 to L4 and S1) are required to define coordinate system."
            )

        corner_points = np.array(
            [points["TL"][:-1], points["TR"][:-1], points["BL"], points["BR"]],
        )[:, :, 0]  # (tl/tr/bl/br, vertebrae, xy)
        corner_points = corner_points.transpose(1, 0, 2)  # (vertebrae, tl/tr/bl/br, xy)

        c7 = corner_points[0]
        c7_center = c7.mean(axis=0)
        l4 = corner_points[16]  # assuming 17 vertebrae from C7 to L4
        l4_center = l4.mean(axis=0)
        origin = (c7_center + l4_center) / 2.0
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
        view = determine_radiograph_view(lm.flags)
        return cls(lm.imagePath, image_wh, origin, units, view)

    @classmethod
    def from_labelme_file(cls, json_path: Path | str) -> "SpineRadiograph":
        """Create SpineRadiograph from LabelMe JSON file."""
        json_path = Path(json_path)
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        return cls.from_labelme(lm)

    @classmethod
    def from_cervical_labelme(cls, lm: LabelMe) -> "SpineRadiograph":
        """
        Create a SpineRadiograph instance from LabelMe annotation of cervical x-ray.

        LabelMe must contain points for C2(BL and BR) and C3 to T1 vertebrae to define the coordinate system.
        Origin (mid point between C7 and L4) are estimated baased on cervical vertebrae points.
        Units are estimated based on average vertebrae size in cervical spine.
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
        if n_tl != n_bl - 1:  # -1 because C2 has only BL and BR points
            raise ValueError("Number of top and bottom vertebrae points do not match.")
        if n_tl < 6:
            raise ValueError(
                "At least 6 vertebrae points (C2 to T1) are required to define coordinate system."
            )

        corner_points = np.array(
            [points["TL"], points["TR"], points["BL"][1:], points["BR"][1:]],
        )[:, :, 0]  # (tl/tr/bl/br, vertebrae, xy)
        corner_points = corner_points.transpose(1, 0, 2)  # (vertebrae, tl/tr/bl/br, xy)

        multiplier = 1.35  # compensation factor to estimate full spine size from cervical spine
        mean_width = multiplier * np.mean(
            np.concatenate(
                [
                    np.linalg.norm(corner_points[:, 1] - corner_points[:, 0], axis=1),  # TR - TL
                    np.linalg.norm(corner_points[:, 3] - corner_points[:, 2], axis=1),  # BR - BL
                ],
            )
        )
        mean_height = multiplier * np.mean(
            np.concatenate(
                [
                    np.linalg.norm(corner_points[:, 2] - corner_points[:, 0], axis=1),  # BL - TL
                    np.linalg.norm(corner_points[:, 3] - corner_points[:, 1], axis=1),  # BR - TR
                ],
            )
        )
        t1 = corner_points[-1]
        t1_center = t1.mean(axis=0)
        origin_y = (
            t1_center[1] + 1.05 * 8 * mean_height
        )  # estimate T9/T10 position. 1.05 is for disc height
        origin = np.array([t1_center[0], origin_y])

        units = np.array([mean_width, mean_height])
        image_wh = np.array([lm.imageWidth, lm.imageHeight])
        view = determine_radiograph_view(lm.flags)
        return cls(lm.imagePath, image_wh, origin, units, view)

    @classmethod
    def from_cervical_labelme_file(cls, json_path: Path | str) -> "SpineRadiograph":
        """Create SpineRadiograph from LabelMe JSON file for cervical x-ray."""
        json_path = Path(json_path)
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        return cls.from_cervical_labelme(lm)


class SpineRadiographAtlas:
    """
    Spine atlas show scan regions
    """

    def __init__(self, frontal: SpineRadiograph, lateral_left: SpineRadiograph):
        assert frontal.view == RadiographView.FRONTAL, "Frontal radiograph must have FRONTAL view."
        assert lateral_left.view == RadiographView.LATERAL_LEFT, (
            "Lateral radiograph must have LATERAL_LEFT view."
        )
        self.frontal = frontal
        self.lateral_left = lateral_left

    @staticmethod
    def built_in() -> "SpineRadiographAtlas":
        """Load built-in spine radiograph atlas."""

        data_dir = Path(__file__).parent / "../../../data/radiopaedia/raw"
        frontal = SpineRadiograph.from_labelme_file(data_dir / "case4_frontal.json")
        lateral_left = SpineRadiograph.from_labelme_file(data_dir / "case4_lateral.json")
        return SpineRadiographAtlas(frontal=frontal, lateral_left=lateral_left)

    def locate(
        self,
        bounding_box: np.ndarray,
        view: RadiographView,
        line_color: str = "red",
        line_width: int = 3,
    ) -> Image.Image:
        """
        Locate bounding box in anatomical coordinates.

        Args:
            bounding_box: (top_x, top_y, bottom_x, bottom_y) in anatomical units
            view: RadiographView

        Returns:
            image: PIL Image with bounding box drawn
        """
        if view == RadiographView.FRONTAL:
            radiograph = self.frontal
            flip_required = False
        elif view == RadiographView.LATERAL_LEFT:
            radiograph = self.lateral_left
            flip_required = False
        else:
            radiograph = self.lateral_left.flip()
            flip_required = True

        top_left_anat = np.array([[bounding_box[0], bounding_box[1]]])  # (x, y)
        bottom_right_anat = np.array([[bounding_box[2], bounding_box[3]]])  # (x, y)

        top_left_pix = radiograph.anat_to_pix(top_left_anat)
        bottom_right_pix = radiograph.anat_to_pix(bottom_right_anat)

        # draw rectangle on image
        image = Image.open(radiograph.image_path).convert("RGB")

        if flip_required:
            image = ImageOps.mirror(image)

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [
                (top_left_pix[0, 0], top_left_pix[0, 1]),
                (bottom_right_pix[0, 0], bottom_right_pix[0, 1]),
            ],
            outline=line_color,
            width=line_width,
        )
        return image
