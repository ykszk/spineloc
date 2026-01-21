import os
from pathlib import Path

import pytest
from PIL import Image

from spineloc.data import spine_radiograph
from spineloc.data.dataset import SpineCoordinateDataset
from spineloc.utils.labelme import LabelMe


def data_dir() -> Path:
    return Path(__file__).parent / "../data"


def test_assert():
    assert True


@pytest.fixture(scope="session")
def spine_radiograph_dir(tmp_path_factory) -> Path:
    fn = tmp_path_factory.mktemp("data")
    return fn


SKIP_REASON = "This test is for visual inspection. Run with `TEST_VIS=1` to enable."


@pytest.mark.skipif(not os.getenv("TEST_VIS"), reason=SKIP_REASON)
def test_spine_radiograph_from_labelme(spine_radiograph_dir):
    def inner(filename: str):
        json_path = data_dir() / f"radiopedia/raw/{filename}"
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = spine_radiograph.SpineRadiograph.from_labelme(lm)
        assert isinstance(instance, spine_radiograph.SpineRadiograph)

        spine_lm = instance.to_labelme()
        assert isinstance(spine_lm, LabelMe)
        output_path = spine_radiograph_dir / filename
        print(f"Writing output to {output_path}")
        with open(output_path, "w") as f:
            f.write(spine_lm.model_dump_json(indent=2))

    inner("frontal.json")
    inner("lateral.json")


@pytest.mark.skipif(not os.getenv("TEST_VIS"), reason=SKIP_REASON)
def test_spine_cropped_dataset(spine_radiograph_dir):
    json_dir = data_dir() / "radiopedia/raw"
    instances = []
    for json_path in json_dir.glob("*.json"):
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = spine_radiograph.SpineRadiograph.from_labelme(lm)
        instances.append(instance)

    dataset = SpineCoordinateDataset(instances)
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "crops"
    output_dir.mkdir(exist_ok=True)
    for i in range(len(dataset)):
        crop_img, anat_coords, view_id = dataset[i]
        image_path = output_dir / f"crop_{i:03d}.jpg"
        text_path = output_dir / f"crop_{i:03d}.txt"

        img = Image.fromarray(crop_img.numpy().astype("uint8"))
        img.save(image_path)

        with open(text_path, "w") as f:
            f.write(f"view: {view_id}\n")
            f.write(f"coord: {anat_coords}\n")
