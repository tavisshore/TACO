"""Example script for training image retrieval model."""

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner

from taco.sensors.cvgl import (
    CVUSADataset,
    CVUSADatasetConfig,
    ImageRetrievalModel,
    ImageRetrievalModelConfig,
)
from taco.sensors.cvgl.data import CVGLDataModule, DatasetShuffleCallback
from taco.utils.config import parse_args


def main():
    """Train a simple image retrieval model."""
    args = parse_args()

    file_path = Path(args.data_folder)

    # Configuration
    train_config = CVUSADatasetConfig(
        data_folder=file_path,
        mode="triplet",
        stage="train",
    )
    val_config = CVUSADatasetConfig(
        data_folder=file_path,
        mode="triplet",
        stage="val",
    )

    model_config = ImageRetrievalModelConfig(
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        scheduler_type=args.scheduler_type,
        scheduler_t_max=args.scheduler_t_max,
        scheduler_eta_min=args.scheduler_eta_min,
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = CVUSADataset(train_config)
    val_dataset = CVUSADataset(val_config)
    print(f"Train dataset: {len(train_dataset)} triplets")
    print(f"Val dataset: {len(val_dataset)} triplets")

    # Create data module for auto batch size support
    datamodule = CVGLDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    print("\nCreating model...")
    if args.model_name == "sample4geo":
        model = ImageRetrievalModel.from_sample4geo(
            config=model_config,
        )
    elif args.model_name == "convnext":
        model = ImageRetrievalModel.from_convnext(
            config=model_config,
            model_name=args.model_variant,
            pretrained=True,
            img_size=args.img_size,
            freeze=args.freeze_encoder,
        )
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create callbacks
    output_dir = Path(args.output_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="retrieval-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=args.early_stop_patience,
        mode="min",
    )

    # Custom shuffle callback for similarity-based batching
    shuffle_callback = DatasetShuffleCallback(
        train_dataset=train_dataset,
        update_frequency=args.shuffle_update_freq,
        neighbour_select=args.neighbour_select,
        neighbour_range=args.neighbour_range,
        cache_dir=output_dir / "embeddings_cache",
    )

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project="taco",
        name=args.experiment_name,
        log_model="all",  # Log model checkpoints to W&B
        save_dir=str(output_dir / "wandb"),
    )

    # Log hyperparameters to W&B
    wandb_logger.log_hyperparams(
        {
            "embedding_dim": model_config.embedding_dim,
            "learning_rate": model_config.learning_rate,
            "temperature": model_config.temperature,
            "margin": model_config.margin,
            "encoder_model": args.model_name,
            "encoder_img_size": args.img_size,
            "loss_type": model_config.loss_type,
            "scheduler_type": model_config.scheduler_type,
            "scheduler_t_max": model_config.scheduler_t_max,
            "scheduler_eta_min": model_config.scheduler_eta_min,
            "batch_size": args.batch_size,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "freeze_encoder": args.freeze_encoder,
        }
    )

    # Parse devices argument (handle "auto" and numeric strings)
    if args.devices == "auto":
        devices = "auto"
    else:
        try:
            devices = int(args.devices)
        except ValueError:
            devices = args.devices

    # Create trainer
    print("\nCreating trainer...")
    print(f"Using devices: {devices}")
    print(f"Using strategy: {args.strategy}")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, early_stop, shuffle_callback],
        logger=wandb_logger,
        # val_check_interval=1000,  # Validate every 1000 training steps
    )

    # Auto batch size tuning if requested
    if args.auto_batch_size:
        print("\nFinding optimal batch size...")
        tuner = Tuner(trainer)
        tuner.scale_batch_size(
            model,
            datamodule=datamodule,
            mode="power",  # Try powers of 2
            steps_per_trial=3,  # Number of steps to run for each trial
            init_val=2,  # Start with batch size of 2
            max_trials=25,  # Maximum number of trials
        )
        print(f"Optimal batch size found: {datamodule.batch_size}")

        # Update W&B config with the found batch size
        wandb_logger.log_hyperparams({"batch_size": datamodule.batch_size})
    else:
        print(f"Using batch size: {datamodule.batch_size}")

    # Train
    print("\nStarting training...")
    trainer.fit(model, datamodule=datamodule)

    print("\n✓ Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

    # Save final model
    final_model_path = output_dir / "weights" / f"retrieval_{args.experiment_name}_final.pth"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
