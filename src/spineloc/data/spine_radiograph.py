import numpy as np

from spineloc.utils.labelme import LabelMe


class SpineRadiograph:
    """
    Spine radiograph with patient specific coordinate system.

    Coordinate system is defined by:
    origin: mid-point between C7 and S1 vertebrae in image coordinates (pixels)
    units: average width and height of vertebrae in pixels
    """

    def __init__(self, image_path: str, origin: np.ndarray, units: np.ndarray):
        self.image_path = image_path
        self.origin = origin  # np.ndarray of shape (2,)
        self.units = units  # np.ndarray of shape (2,)

    def to_labelme(self, length=10) -> LabelMe:
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
                    [float(self.origin[0] + length * self.units[0]), float(self.origin[1])],
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
                    [float(self.origin[0]), float(self.origin[1] + length * self.units[1])],
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
        )[:, :, 0]  # (tl/tr,bl/br, vertebrae, xy)
        corner_points = corner_points.transpose(1, 0, 2)  # (vertebrae, tl/tr/bl/br, xy)

        c7 = corner_points[0]
        c7_center = c7.mean(axis=0)
        s_center = np.array([*points["TL"][-1], *points["TR"][-1]]).mean(axis=0)
        origin = (c7_center + s_center) / 2.0
        mean_width = np.mean(
            np.concatenate(
                [
                    corner_points[:, 1] - corner_points[:, 0],  # TR - TL
                    corner_points[:, 3] - corner_points[:, 2],  # BR - BL
                ],
            )
        )
        mean_height = np.mean(
            np.concatenate(
                [
                    corner_points[:, 2] - corner_points[:, 0],  # BL - TL
                    corner_points[:, 3] - corner_points[:, 1],  # BR - TR
                ],
            )
        )
        units = np.array([mean_width, mean_height])
        return cls(lm.imagePath, origin, units)
