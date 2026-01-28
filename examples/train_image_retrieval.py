"""Example script for training image retrieval model."""

from pathlib import Path

import cv2
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

from taco.sensors.cvgl import (
    CVUSADataset,
    CVUSADatasetConfig,
    ImageRetrievalModel,
    ImageRetrievalModelConfig,
)


class DatasetShuffleCallback(Callback):
    """Callback to shuffle dataset with similarity-based batching after each epoch.

    Computes embeddings for all training samples and creates a similarity dictionary
    for neighbor-based batch composition. This improves hard negative mining and
    training diversity.

    Args:
        train_dataset: The training dataset with shuffle() method
        update_frequency: How often to recompute similarities (every N epochs)
        neighbour_select: Number of neighbors to select for batch grouping
        neighbour_range: Range of top neighbors to sample from
    """

    def __init__(
        self,
        train_dataset: CVUSADataset,
        update_frequency: int = 5,
        neighbour_select: int = 64,
        neighbour_range: int = 128,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.update_frequency = update_frequency
        self.neighbour_select = neighbour_select
        self.neighbour_range = neighbour_range
        self.sim_dict = None

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Shuffle dataset before each epoch starts."""
        # First epoch or when similarity dict needs update
        if trainer.current_epoch == 0 or trainer.current_epoch % self.update_frequency == 0:
            print(f"\nEpoch {trainer.current_epoch}: Computing similarity dictionary...")
            self.sim_dict = self._compute_similarity_dict(pl_module)

        # Shuffle dataset with similarity dictionary
        self.train_dataset.shuffle(
            sim_dict=self.sim_dict,
            neighbour_select=self.neighbour_select,
            neighbour_range=self.neighbour_range,
        )

    @torch.no_grad()
    def _compute_similarity_dict(self, pl_module: L.LightningModule) -> dict:
        """Compute similarity dictionary for all training samples.

        Args:
            pl_module: The PyTorch Lightning model

        Returns:
            Dictionary mapping idx -> list of similar neighbor indices (sorted by similarity)
        """
        pl_module.eval()
        device = pl_module.device

        # Get all training indices
        train_ids = self.train_dataset.train_ids

        # Compute embeddings for all training samples (using satellite/reference images)
        print("Computing embeddings for training samples...")
        embeddings = []

        for idx in tqdm(train_ids, desc="Encoding images"):
            # Load satellite image for this sample
            sat_path = self.train_dataset.idx2sat[idx]
            img = self._load_and_preprocess_image(
                f"{self.train_dataset.config.data_folder}/{sat_path}"
            )
            img = img.unsqueeze(0).to(device)

            # Get embedding
            emb = pl_module(img)
            embeddings.append(emb.cpu())

        # Stack all embeddings (N, embedding_dim)
        embeddings = torch.cat(embeddings, dim=0)

        # Compute pairwise cosine similarities in batches to avoid OOM
        print("Computing pairwise similarities in batches...")
        batch_size = 1000  # Process 1000 queries at a time
        top_k = min(self.neighbour_range, len(train_ids) - 1)

        sim_dict = {}

        for batch_start in tqdm(
            range(0, len(train_ids), batch_size), desc="Computing similarities"
        ):
            batch_end = min(batch_start + batch_size, len(train_ids))
            batch_embeddings = embeddings[batch_start:batch_end]

            # Compute similarities between batch and all embeddings
            # Shape: (batch_size, N)
            similarities = torch.matmul(batch_embeddings, embeddings.T)

            # For each sample in batch, get top-k neighbors
            for i, global_idx in enumerate(range(batch_start, batch_end)):
                idx = train_ids[global_idx]

                # Get similarity scores for this sample
                sim_scores = similarities[i].clone()

                # Exclude self by setting to minimum
                sim_scores[global_idx] = -1

                # Get top-k most similar indices
                _, top_indices = torch.topk(sim_scores, top_k)

                # Map back to original training IDs
                similar_ids = [train_ids[j] for j in top_indices.tolist()]
                sim_dict[idx] = similar_ids

        pl_module.train()
        return sim_dict

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for embedding computation.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image tensor (C, H, W)
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply same transforms as dataset if available
        if self.train_dataset.config.transforms_reference is not None:
            img = self.train_dataset.config.transforms_reference(image=img)["image"]

        # Convert to tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        # Resize to network input size
        img = F.interpolate(
            img.unsqueeze(0),
            size=self.train_dataset.config.network_input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return img


def main():
    """Train a simple image retrieval model."""
    file_path = Path("/scratch/datasets/CVUSA/files")
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
        embedding_dim=512,
        learning_rate=1e-4,
        loss_type="ntxent",  # Options: "combined", "ntxent", "triplet"
    )

    # Create datasets
    print("Loading datasets...")
    train_dataset = CVUSADataset(train_config)
    val_dataset = CVUSADataset(val_config)
    print(f"Train dataset: {len(train_dataset)} triplets")
    print(f"Val dataset: {len(val_dataset)} triplets")

    # Create data loaders
    # NOTE: shuffle=False because DatasetShuffleCallback handles custom shuffling
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    print("\nCreating model...")
    model = ImageRetrievalModel(config=model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path("output") / "checkpoints",
        filename="retrieval-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=25,
        mode="min",
    )

    # Custom shuffle callback for similarity-based batching
    shuffle_callback = DatasetShuffleCallback(
        train_dataset=train_dataset,
        update_frequency=5,  # Recompute similarities every 5 epochs
        neighbour_select=64,  # Number of neighbors to select
        neighbour_range=128,  # Range of top neighbors to sample from
    )

    # Initialize Weights & Biases logger
    wandb_logger = WandbLogger(
        project="taco",
        name="initial",
        log_model="all",  # Log model checkpoints to W&B
        save_dir="output/wandb",
    )

    # Log hyperparameters to W&B
    wandb_logger.experiment.config.update(
        {
            "embedding_dim": model_config.embedding_dim,
            "learning_rate": model_config.learning_rate,
            "temperature": model_config.temperature,
            "margin": model_config.margin,
            "freeze_backbone": model_config.freeze_backbone,
            "loss_type": model_config.loss_type,
            "batch_size": 32,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        }
    )

    # Create trainer
    print("\nStarting training...")
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback, early_stop, shuffle_callback],
        logger=wandb_logger,
        check_val_every_n_epoch=2,
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\n✓ Training complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
