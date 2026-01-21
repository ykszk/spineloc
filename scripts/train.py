import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from spineloc.models.spine import MultiTaskSpineModel, SpineDataModule


def load_data(data_cfg):
    """Load your actual spine X-ray data here."""
    # Replace with your actual data loading logic
    num_images = 100
    full_images = [np.random.rand(1024, 512).astype(np.float32) for _ in range(num_images)]
    full_coords = [(0.0, 0.0, 1.0, 1.0) for _ in range(num_images)]
    view_labels = ["frontal" if i % 2 == 0 else "lateral" for i in range(num_images)]
    laterality_labels = [
        None if v == "frontal" else ("left" if i % 3 == 0 else "right")
        for i, v in enumerate(view_labels)
    ]

    return full_images, full_coords, view_labels, laterality_labels


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""

    # Print configuration
    print("=" * 60)
    print("Configuration")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load data
    images, coords, views, laterality = load_data(cfg.data)

    # Create DataModule
    dm = SpineDataModule(
        train_images=images,
        train_coords=coords,
        train_views=views,
        train_laterality=laterality,
        **cfg.data,
    )

    # Create model
    model = MultiTaskSpineModel(**cfg.model)

    # Setup logger
    if cfg.logger.name == "tensorboard":
        logger = TensorBoardLogger(save_dir=cfg.logger.save_dir, name=cfg.logger.experiment_name)
    elif cfg.logger.name == "wandb":
        logger = WandbLogger(
            project=cfg.logger.project,
            name=cfg.logger.experiment_name,
            save_dir=cfg.logger.save_dir,
        )
    else:
        logger = True  # Default logger

    # Setup callbacks
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.callbacks.checkpoint.dirpath,
        filename=cfg.callbacks.checkpoint.filename,
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        save_last=cfg.callbacks.checkpoint.save_last,
    )
    callbacks.append(checkpoint_callback)

    early_stop_callback = EarlyStopping(
        monitor=cfg.callbacks.early_stopping.monitor,
        patience=cfg.callbacks.early_stopping.patience,
        mode=cfg.callbacks.early_stopping.mode,
        verbose=cfg.callbacks.early_stopping.verbose,
    )
    callbacks.append(early_stop_callback)

    # Create trainer
    trainer = pl.Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)

    # Train
    trainer.fit(model, dm)

    # Test on best model if checkpoint callback was used
    if cfg.callbacks.checkpoint.enabled and cfg.get("test_after_training", False):
        best_model_path = checkpoint_callback.best_model_path
        print(f"\nLoading best model from: {best_model_path}")
        best_model = MultiTaskSpineModel.load_from_checkpoint(best_model_path)

        # Inference example
        print("\n" + "=" * 60)
        print("Inference Example")
        print("=" * 60)

        best_model.eval()
        dm.setup()
        sample_img, sample_coords, sample_view, sample_lat = dm.val_dataset[0]
        sample_img = sample_img.unsqueeze(0)

        with torch.no_grad():
            coord_map, coord_log_var, view_logits, lat_logits = best_model(sample_img)
            pred_box, uncertainties = best_model.predict_bounding_box(coord_map, coord_log_var)

            view_probs = F.softmax(view_logits, dim=1)
            lat_probs = F.softmax(lat_logits, dim=1)

            pred_view = view_logits.argmax(dim=1)
            pred_lat = lat_logits.argmax(dim=1)

            view_names = ["frontal", "lateral"]
            lat_names = ["left", "right"]

            print(f"\nTrue coords: {sample_coords}")
            print(f"Predicted coords: {pred_box.squeeze()}")

            if uncertainties is not None:
                print(f"Coordinate uncertainties (std): {torch.sqrt(uncertainties.squeeze())}")

            print(f"\nTrue view: {view_names[sample_view.item()]}")
            print(
                f"Predicted view: {view_names[pred_view.item()]} "
                f"(confidence: {view_probs[0, pred_view].item():.2%})"
            )

            if sample_lat.item() >= 0:
                print(f"\nTrue laterality: {lat_names[sample_lat.item()]}")
                print(
                    f"Predicted laterality: {lat_names[pred_lat.item()]} "
                    f"(confidence: {lat_probs[0, pred_lat].item():.2%})"
                )
            else:
                print("\nLaterality: N/A (frontal view)")


if __name__ == "__main__":
    main()
