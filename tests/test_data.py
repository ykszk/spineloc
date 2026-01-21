import os
from pathlib import Path

import pytest

from spineloc.data import spine_radiograph
from spineloc.utils.labelme import LabelMe


def data_dir() -> Path:
    return Path(__file__).parent / "../data"


def test_assert():
    assert True


@pytest.fixture(scope="package")
def spine_radiograph_json_path(tmp_path_factory) -> Path:
    fn = tmp_path_factory.mktemp("data") / "spine_coordinate_system.json"
    print("Temporary JSON path:", fn)
    return fn


@pytest.mark.skipif(
    not os.getenv("TEST_VIS"),
    reason="This test is for visual inspection. Run with `TEST_VIS=1` to enable.",
)
def test_spine_radiograph_from_labelme(spine_radiograph_json_path):
    json_path = data_dir() / "radiopedia/raw/frontal.json"
    lm = LabelMe.from_file(json_path)
    lm.imagePath = str(lm.resolve_image_path(json_path.parent))
    instance = spine_radiograph.SpineRadiograph.from_labelme(lm)
    assert isinstance(instance, spine_radiograph.SpineRadiograph)

    spine_lm = instance.to_labelme()
    assert isinstance(spine_lm, LabelMe)
    with open(spine_radiograph_json_path, "w") as f:
        f.write(spine_lm.model_dump_json(indent=2))
