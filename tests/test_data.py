from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import spineloc.data.transforms as transforms
from spineloc.data.dataset import (
    SpineCoordinateDataset,
    SpineDataModule,
    SpineRadiograph,
)
from spineloc.data.spine_radiograph import RadiographView, SpineRadiographAtlas
from spineloc.utils.labelme import LabelMe


def data_dir() -> Path:
    return Path(__file__).parent / "../data"


def test_assert():
    assert True


@pytest.fixture(scope="session")
def spine_radiograph_dir(tmp_path_factory) -> Path:
    fn = tmp_path_factory.mktemp("data")
    return fn


@pytest.mark.visualize
def test_spine_radiograph_from_labelme(spine_radiograph_dir):
    def inner(filename: str, is_frontal: bool):
        json_path = data_dir() / f"radiopaedia/raw/{filename}"
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = SpineRadiograph.from_labelme(lm)
        assert isinstance(instance, SpineRadiograph)
        view = RadiographView.FRONTAL if is_frontal else RadiographView.LATERAL_LEFT
        assert instance.view == view

        # test round trip of coordinates
        # TODO: move to separate test
        anat_coords = np.array([[-1, -1], [1, 1]])
        pixel_coords = instance.anat_to_pix(anat_coords)
        anat_coords_roundtrip = instance.pix_to_anat(pixel_coords)
        np.testing.assert_allclose(anat_coords, anat_coords_roundtrip, atol=1e-5)

        spine_lm = instance.to_labelme()
        assert isinstance(spine_lm, LabelMe)
        output_path = spine_radiograph_dir / filename
        print(f"Writing output to {output_path}")
        with open(output_path, "w") as f:
            f.write(spine_lm.model_dump_json(indent=2))

    for case_id in range(1, 5):
        inner(f"case{case_id}_frontal.json", is_frontal=True)
        inner(f"case{case_id}_lateral.json", is_frontal=False)


@pytest.mark.visualize
def test_cervical_labelme(spine_radiograph_dir):
    def inner(json_filename: Path, is_frontal: bool):
        json_path = Path(json_filename)
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = SpineRadiograph.from_cervical_labelme(lm)
        assert isinstance(instance, SpineRadiograph)
        view = RadiographView.FRONTAL if is_frontal else RadiographView.LATERAL_LEFT
        assert instance.view == view

        # test round trip of coordinates
        # TODO: move to separate test
        anat_coords = np.array([[-1, -1], [1, 1]])
        pixel_coords = instance.anat_to_pix(anat_coords)
        anat_coords_roundtrip = instance.pix_to_anat(pixel_coords)
        np.testing.assert_allclose(anat_coords, anat_coords_roundtrip, atol=1e-5)

        spine_lm = instance.to_labelme()
        assert isinstance(spine_lm, LabelMe)
        output_path = spine_radiograph_dir / ("neck_" + json_filename.name)
        print(f"Writing output to {output_path}")
        with open(output_path, "w") as f:
            f.write(spine_lm.model_dump_json(indent=2))

    json_path = data_dir() / "radiopaedia/neck/case1_lateral.json"
    inner(json_path, is_frontal=False)


def load_instances():
    json_dir = data_dir() / "radiopaedia/raw"
    instances = []
    for json_path in json_dir.glob("*.json"):
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = SpineRadiograph.from_labelme(lm)
        instances.append(instance)

    neck_json_dir = data_dir() / "radiopaedia/neck"
    for json_path in neck_json_dir.glob("*.json"):
        lm = LabelMe.from_file(json_path)
        lm.resolve_image_path(json_path.parent)
        instance = SpineRadiograph.from_cervical_labelme(lm)
        instances.append(instance)
    return instances


def save_dataset(dataset: SpineCoordinateDataset, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    atlas = SpineRadiographAtlas.built_in()
    for i in range(len(dataset)):
        crop_img, anat_coords, view_id = dataset[i]
        image_path = output_dir / f"crop_{i:03d}.jpg"
        text_path = output_dir / f"crop_{i:03d}.txt"
        loc_path = output_dir / f"crop_{i:03d}_loc.jpg"

        if isinstance(crop_img, torch.Tensor):
            crop_img = crop_img.numpy()[0]
            # normalize
            crop_img = (crop_img - crop_img.min()) / (crop_img.max() - crop_img.min()) * 255.0
        img = Image.fromarray(crop_img.astype("uint8"))
        img.save(image_path)

        with open(text_path, "w") as f:
            f.write(f"view: {view_id}\n")
            f.write(f"coord: {anat_coords}\n")
        loc_img = atlas.locate(anat_coords, RadiographView(view_id))
        loc_img.save(loc_path)


@pytest.mark.visualize
def test_spine_cropped_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(instances)
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "crops"
    save_dataset(dataset, output_dir)


@pytest.mark.visualize
def test_train_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(
        instances, transform=transforms.get_train_transforms_with_replay()
    )
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "train"
    save_dataset(dataset, output_dir)


@pytest.mark.visualize
def test_val_dataset(spine_radiograph_dir):
    instances = load_instances()

    dataset = SpineCoordinateDataset(instances, transform=transforms.get_val_transforms())
    assert len(dataset) == len(instances) * len(dataset.aspect_ratios)

    output_dir = spine_radiograph_dir / "val"
    save_dataset(dataset, output_dir)


def test_data_module():
    data_module = SpineDataModule(
        data_dirs=[str(data_dir() / "radiopaedia/raw")], batch_size=2, num_workers=0, val_split=0.5
    )
    data_module.setup()
    assert data_module.data_loaded
    assert len(data_module.train_dataset) > 0
    assert len(data_module.val_dataset) > 0

    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
