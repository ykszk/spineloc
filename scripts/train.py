from typing import List

import hydra
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf

# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from spineloc.data.dataset import SpineDataModule
from spineloc.models.spine import MultiTaskSpineModel
from spineloc.utils import (
    instantiate_callbacks,
    instantiate_loggers,
)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # Print configuration
    log.info("Starting training with configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Create DataModule
    log.info("Instantiating data module...")
    datamodule: SpineDataModule = hydra.utils.instantiate(cfg.data)
    log.info("Training samples: %d", len(datamodule.train_dataloader()))
    log.info("Validation samples: %d", len(datamodule.val_dataloader()))

    # Create model
    log.info("Instantiating model...")
    model: MultiTaskSpineModel = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
    main()
