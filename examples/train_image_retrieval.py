"""Example script for training image retrieval model."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from taco.sensors.cvgl import ImageRetrievalModel, TripletDataset


def main():
    """Train a simple image retrieval model."""

    # Configuration
    config = {
        "train_data_dir": Path("data/images/train"),
        "val_data_dir": Path("data/images/val"),
        "train_triplets": Path("data/triplets/train.txt"),
        "val_triplets": Path("data/triplets/val.txt"),
        "output_dir": Path("outputs/retrieval"),
        "embedding_dim": 512,
        "batch_size": 16,
        "max_epochs": 50,
        "learning_rate": 1e-4,
        "num_workers": 4,
    }

    # Create datasets
    print("Loading datasets...")
    train_dataset = TripletDataset(
        image_dir=config["train_data_dir"],
        triplets_file=config["train_triplets"],
    )

    val_dataset = TripletDataset(
        image_dir=config["val_data_dir"],
        triplets_file=config["val_triplets"],
    )

    print(f"Train dataset: {len(train_dataset)} triplets")
    print(f"Val dataset: {len(val_dataset)} triplets")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = ImageRetrievalModel(
        embedding_dim=config["embedding_dim"],
        pretrained=True,
        learning_rate=config["learning_rate"],
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["output_dir"] / "checkpoints",
        filename="retrieval-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=10,
        mode="min",
    )

    # Create trainer
    print("\nStarting training...")
    trainer = L.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\n✓ Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
