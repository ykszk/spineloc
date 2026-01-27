import hydra
import lightning as L
import numpy as np
from lightning import Trainer
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader

from spineloc.data import transforms
from spineloc.models.spine import MultiTaskSpineNet
from spineloc.models.spine_module import SpineViewModule


@hydra.main(version_base=None, config_path="../configs", config_name="validation")
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
    a_transform = transforms.get_predict_transforms(cfg.data.image_size)

    def transform(image):
        return a_transform(image=np.array(image))["image"]

    dataset = hydra.utils.instantiate(cfg.dataset, transform=transform)
    log.info(f"Dataset has {len(dataset)} images.")
    dataloader: DataLoader = hydra.utils.instantiate(cfg.dataloader, dataset=dataset)

    # Create model
    log.info("Instantiating model...")
    model: MultiTaskSpineNet = hydra.utils.instantiate(cfg.model.net)
    model.load_state_dict(load_file(cfg.safetensors_path))

    module = SpineViewModule(net=model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # Predict
    trainer.test(module, dataloaders=dataloader)


if __name__ == "__main__":
    main()
    main()
