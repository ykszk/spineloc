from pathlib import Path
from typing import List

import hydra
import lightning as L
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from spineloc.data.dataset import SpineDataModule
from spineloc.models.spine import MultiTaskSpineNet
from spineloc.models.spine_module import MultiTaskSpineModule
from spineloc.utils.instantiators import instantiate_callbacks, instantiate_loggers
from spineloc.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

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
    net: MultiTaskSpineNet = hydra.utils.instantiate(cfg.model.net)
    module: MultiTaskSpineModule = hydra.utils.instantiate(
        cfg.model.module,
        net=net,
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # Train
    trainer.fit(module, datamodule)

    # Prevent multiple processes from saving the model
    if not trainer.is_global_zero:
        return 0

    ckpt_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Loading best model checkpoint saved at: {ckpt_path}")
    best_model: MultiTaskSpineModule = MultiTaskSpineModule.load_from_checkpoint(
        ckpt_path, weights_only=False
    )
    safetensor_path = Path(ckpt_path).with_name("best.safetensors")
    log.info(f"Saving best model weights in safetensors format at: {safetensor_path}")
    save_file(
        best_model.net.state_dict(),
        safetensor_path,
    )

    # Required for sweeping W&B runs to finish properly
    # https://github.com/wandb/wandb/issues/1314#issuecomment-2596424189
    if any(isinstance(lgr, WandbLogger) for lgr in logger):
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
