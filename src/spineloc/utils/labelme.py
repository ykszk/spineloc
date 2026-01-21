from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Union

import numpy as np
from pydantic import BaseModel
from szkspyne.type import LineMixin


class LabelMe(BaseModel):
    class Shape(BaseModel):
        label: str
        points: List[List[float]]
        group_id: Optional[int]
        shape_type: str
        flags: Dict[str, bool]

        @staticmethod
        def from_dict(data: Dict[str, Any]) -> "LabelMe.Shape":
            return LabelMe.Shape(
                label=data["label"],
                points=data["points"],
                group_id=data.get("group_id"),
                shape_type=data["shape_type"],
                flags=data.get("flags", {}),
            )

    version: str
    shapes: List[Shape]
    imagePath: str
    imageData: Optional[str]
    imageHeight: int
    imageWidth: int
    flags: Dict[str, bool]

    @staticmethod
    def from_file(filename: Union[str, Path]) -> "LabelMe":
        with open(filename) as f:
            return LabelMe.model_validate_json(f.read())

    def resolve_image_path(self, base_dir: Union[str, Path]) -> Path:
        base_dir = Path(base_dir)
        original_path = Path(self.imagePath)
        if original_path.is_absolute():
            return original_path
        resolved = (base_dir / self.imagePath).resolve()
        self.imagePath = str(resolved)
        return resolved

    def shift(self, tx: float, ty: float) -> "LabelMe":
        shapes = []
        for shape in self.shapes:
            points = [[p[0] + tx, p[1] + ty] for p in shape.points]
            shapes.append(shape.model_copy(update={"points": points}))
        return self.model_copy(update={"shapes": shapes})

    def scale(self, s: float) -> "LabelMe":
        shapes = []
        for shape in self.shapes:
            points = [[p[0] * s, p[1] * s] for p in shape.points]
            shapes.append(shape.model_copy(update={"points": points}))
        return self.model_copy(update={"shapes": shapes})

    def rotate(self, angle_deg: float, center: Optional[List[float]] = None) -> "LabelMe":
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        if center is None:
            center = [self.imageWidth / 2, self.imageHeight / 2]

        cx, cy = center
        shapes = []
        for shape in self.shapes:
            points = []
            for p in shape.points:
                x_shifted = p[0] - cx
                y_shifted = p[1] - cy
                x_rotated = x_shifted * cos_a - y_shifted * sin_a + cx
                y_rotated = x_shifted * sin_a + y_shifted * cos_a + cy
                points.append([x_rotated, y_rotated])
            shapes.append(shape.model_copy(update={"points": points}))
        return self.model_copy(update={"shapes": shapes})

    def _into_dict(self) -> Dict[str, Dict[str, list]]:
        """
        Dict[shape: str, Dict[label: str, points: list]]
        """
        shape_dict: DefaultDict[str, DefaultDict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        for shape in self.shapes:
            shape_dict[shape.shape_type][shape.label].append(shape.points)

        # defaultdict -> dict
        output = {k: {kk: vv for kk, vv in v.items()} for k, v in shape_dict.items()}
        return output

    def into_labelme_line(self, filename: str) -> "LabelMeLine":
        return LabelMeLine(content=self, filename=filename)

    def into_shape_dict(self) -> "ShapeDict":
        shapes = self._into_dict()
        shape_dict = ShapeDict(
            shapes=shapes,
            imagePath=self.imagePath,
            imageHeight=self.imageHeight,
            imageWidth=self.imageWidth,
            flags=self.flag_set(),
        )
        return shape_dict

    def flag_set(self) -> Set[str]:
        return {k for k, v in self.flags.items() if v}


class ShapeDict(BaseModel):
    shapes: Dict[str, Dict[str, list]]
    imagePath: str
    imageHeight: int
    imageWidth: int
    flags: Set[str]

    def into_labelme(self) -> "LabelMe":
        shapes = []
        for shape_type, labels in self.shapes.items():
            for label, points in labels.items():
                for p in points:
                    shapes.append(
                        LabelMe.Shape(
                            label=label,
                            points=p,
                            group_id=None,
                            shape_type=shape_type,
                            flags={},
                        )
                    )
        return LabelMe(
            version="4.5.7",
            shapes=shapes,
            imagePath=self.imagePath,
            imageData=None,
            imageHeight=self.imageHeight,
            imageWidth=self.imageWidth,
            flags={k: True for k in self.flags},
        )

    def transform(self, tf: Callable):
        """
        Apply a transformation to the points in the shape dict.

        Args:
            tf (Callable): A function that takes a list of points and returns a transformed list of points.
        """
        for shape_type, labels in self.shapes.items():
            for label, points in labels.items():
                for i, p in enumerate(points):
                    points[i] = tf(p)
        return self


class ShapeDictLine(BaseModel, LineMixin):
    content: ShapeDict
    filename: str

    @staticmethod
    def from_file(filename: Union[str, Path]) -> "ShapeDictLine":
        with open(filename) as f:
            return ShapeDictLine.model_validate_json(f.read())

    def into_labelme_line(self) -> "LabelMeLine":
        return LabelMeLine(content=self.content.into_labelme(), filename=self.filename)


class LabelMeLine(BaseModel, LineMixin):
    content: LabelMe
    filename: str

    def into_shape_dict_line(self) -> ShapeDictLine:
        shape_dict = self.content.into_shape_dict()
        return ShapeDictLine(content=shape_dict, filename=self.filename)
