import os
from pathlib import Path

import pytest
import torch
from PIL import Image

import spineloc.data.transforms as transforms
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
    def inner(filename: str, is_frontal: bool):
        json_path = data_dir() / f"radiopedia/raw/{filename}"
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = spine_radiograph.SpineRadiograph.from_labelme(lm)
        assert isinstance(instance, spine_radiograph.SpineRadiograph)
        view = (
            spine_radiograph.RadiographView.FRONTAL
            if is_frontal
            else spine_radiograph.RadiographView.LATERAL_LEFT
        )
        assert instance.view == view

        spine_lm = instance.to_labelme()
        assert isinstance(spine_lm, LabelMe)
        output_path = spine_radiograph_dir / filename
        print(f"Writing output to {output_path}")
        with open(output_path, "w") as f:
            f.write(spine_lm.model_dump_json(indent=2))

    inner("frontal.json", is_frontal=True)
    inner("lateral.json", is_frontal=False)


def load_instances():
    json_dir = data_dir() / "radiopedia/raw"
    instances = []
    for json_path in json_dir.glob("*.json"):
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = spine_radiograph.SpineRadiograph.from_labelme(lm)
        instances.append(instance)
    return instances


def save_dataset(dataset: SpineCoordinateDataset, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    for i in range(len(dataset)):
        crop_img, anat_coords, view_id = dataset[i]
        image_path = output_dir / f"crop_{i:03d}.jpg"
        text_path = output_dir / f"crop_{i:03d}.txt"

        if isinstance(crop_img, torch.Tensor):
            crop_img = crop_img.numpy()
            # normalize
            crop_img = (crop_img - crop_img.min()) / (crop_img.max() - crop_img.min()) * 255.0
        img = Image.fromarray(crop_img.astype("uint8")[0])
        img.save(image_path)

        with open(text_path, "w") as f:
            f.write(f"view: {view_id}\n")
            f.write(f"coord: {anat_coords}\n")


@pytest.mark.skipif(not os.getenv("TEST_VIS"), reason=SKIP_REASON)
def test_spine_cropped_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(instances)
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "crops"
    save_dataset(dataset, output_dir)


@pytest.mark.skipif(not os.getenv("TEST_VIS"), reason=SKIP_REASON)
def test_train_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(
        instances, transform=transforms.get_train_transforms_with_replay()
    )
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "train"
    save_dataset(dataset, output_dir)


@pytest.mark.skipif(not os.getenv("TEST_VIS"), reason=SKIP_REASON)
def test_val_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(instances, transform=transforms.get_val_transforms())
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "val"
    save_dataset(dataset, output_dir)
