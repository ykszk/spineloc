import hydra
import lightning as L
import torch
from lightning import Trainer
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from spineloc.data import transforms
from spineloc.data.dataset import InferenceImageDataset
from spineloc.data.spine_radiograph import RadiographView
from spineloc.models.spine import InferenceModule, MultiTaskSpineNet


@hydra.main(version_base=None, config_path="../configs", config_name="predict")
def main(cfg: DictConfig):
    assert cfg.safetensors_path, (
        'set "safetensors_path" using "safetensors_path=<PATH_TO_SAFETENSORS>"'
    )
    assert cfg.image_dir, 'set "image_dir" using "image_dir=<PATH_TO_IMAGE_DIRECTORY>"'

    # Print configuration
    log.info(f"Starting {cfg.task_name} with configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Create Dataloader
    transform = transforms.get_val_transforms(cfg.data.image_size)
    dataset: InferenceImageDataset = hydra.utils.instantiate(cfg.dataset, transform=transform)
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

    # Create model
    log.info("Instantiating model...")
    model: MultiTaskSpineNet = hydra.utils.instantiate(cfg.model.net)
    model.load_state_dict(load_file(cfg.safetensors_path))

    module = InferenceModule(net=model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Predict
    outputs = trainer.predict(module, dataloaders=dataloader)
    assert outputs is not None
    # Flatten outputs
    coords_maps = torch.concat([b[0] for b in outputs], dim=0)
    coord_log_vars = torch.concat([b[1] for b in outputs], dim=0)
    view_logitss = torch.concat([b[2] for b in outputs], dim=0)
    outputs = list(zip(coords_maps, coord_log_vars, view_logitss))

    for path, output in zip(dataset.image_paths, outputs):
        coord_map, coord_log_var, view_logits = output
        view_id = view_logits.argmax(dim=0).item()
        view = RadiographView(view_id)
        bbox, uncert = MultiTaskSpineNet.predict_bounding_box(coord_map, coord_log_var)
        assert uncert is not None
        print(path, bbox[0].tolist(), uncert[0].tolist(), view.name)


if __name__ == "__main__":
    main()
