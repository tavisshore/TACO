"""Training script for image retrieval model."""

import argparse
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .dataset import TripletDataset
from .model import ImageRetrievalModel


def train_retrieval_model(
    train_data_dir: Path,
    train_triplets_file: Path,
    val_data_dir: Path,
    val_triplets_file: Path,
    output_dir: Path,
    embedding_dim: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    temperature: float = 0.07,
    margin: float = 0.2,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    gpus: int | None = None,
    resume_from_checkpoint: Path | None = None,
) -> None:
    """Train image retrieval model.

    Args:
        train_data_dir: Directory containing training images.
        train_triplets_file: File with training triplets.
        val_data_dir: Directory containing validation images.
        val_triplets_file: File with validation triplets.
        output_dir: Directory to save checkpoints and logs.
        embedding_dim: Dimension of output embeddings.
        batch_size: Training batch size.
        num_workers: Number of data loading workers.
        max_epochs: Maximum number of training epochs.
        learning_rate: Learning rate.
        temperature: Temperature for contrastive loss.
        margin: Margin for triplet loss.
        freeze_backbone: Whether to freeze backbone weights.
        pretrained: Whether to use pretrained weights.
        gpus: Number of GPUs to use (None for CPU).
        resume_from_checkpoint: Path to checkpoint to resume from.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    train_dataset = TripletDataset(
        image_dir=train_data_dir,
        triplets_file=train_triplets_file,
    )

    val_dataset = TripletDataset(
        image_dir=val_data_dir,
        triplets_file=val_triplets_file,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(gpus),
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(gpus),
        persistent_workers=num_workers > 0,
    )

    # Create model
    model = ImageRetrievalModel(
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        learning_rate=learning_rate,
        temperature=temperature,
        margin=margin,
        freeze_backbone=freeze_backbone,
    )

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="retrieval-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Create logger
    logger = TensorBoardLogger(
        save_dir=output_dir / "logs",
        name="image_retrieval",
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus else "cpu",
        devices=gpus if gpus else "auto",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        precision="16-mixed" if gpus else "32",
    )

    # Train model
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_from_checkpoint,
    )

    print(f"\nTraining complete! Best model saved to: {checkpoint_callback.best_model_path}")


def main() -> None:
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description="Train image retrieval model for CVGL localization"
    )

    # Data arguments
    parser.add_argument(
        "--train-data-dir",
        type=Path,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--train-triplets",
        type=Path,
        required=True,
        help="File with training triplets",
    )
    parser.add_argument(
        "--val-data-dir",
        type=Path,
        required=True,
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--val-triplets",
        type=Path,
        required=True,
        help="File with validation triplets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/retrieval"),
        help="Output directory for checkpoints and logs",
    )

    # Model arguments
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
        help="Dimension of output embeddings",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Don't use pretrained weights",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze ConvNeXt backbone weights",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive loss",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Margin for triplet loss",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )

    args = parser.parse_args()

    # Train model
    train_retrieval_model(
        train_data_dir=args.train_data_dir,
        train_triplets_file=args.train_triplets,
        val_data_dir=args.val_data_dir,
        val_triplets_file=args.val_triplets,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        margin=args.margin,
        freeze_backbone=args.freeze_backbone,
        pretrained=not args.no_pretrained,
        gpus=args.gpus,
        resume_from_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
