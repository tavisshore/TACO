"""Example script for training image retrieval model."""

import hashlib
import pickle
from pathlib import Path

import cv2
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from tqdm import tqdm

from taco.sensors.cvgl import CVUSADataset
from taco.utils.config import parse_args


class CVGLDataModule(L.LightningDataModule):
    """Lightning DataModule for CVGL training with auto batch size support."""

    def __init__(
        self,
        train_dataset: CVUSADataset,
        val_dataset: CVUSADataset,
        batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # DatasetShuffleCallback handles custom shuffling
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
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
        cache_dir: Directory to cache computed embeddings (None to disable caching)
    """

    def __init__(
        self,
        train_dataset: CVUSADataset,
        update_frequency: int = 5,
        neighbour_select: int = 64,
        neighbour_range: int = 128,
        cache_dir: Path | None = None,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.update_frequency = update_frequency
        self.neighbour_select = neighbour_select
        self.neighbour_range = neighbour_range
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.sim_dict = None

        # Create cache directory if it doesn't exist
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, pl_module) -> str:
        """Compute a cache key based on training data and model architecture.

        Args:
            pl_module: The PyTorch Lightning model

        Returns:
            A hash string that uniquely identifies the training configuration
        """
        # Create a string representation of the cache key components
        train_ids_str = str(sorted(self.train_dataset.train_ids))
        data_folder = str(self.train_dataset.config.data_folder)
        network_size = str(self.train_dataset.config.network_input_size)
        model_arch = pl_module.__class__.__name__
        embedding_dim = str(pl_module.config.embedding_dim)

        # Combine all components
        cache_key_data = (
            f"{train_ids_str}_{data_folder}_{network_size}_{model_arch}_{embedding_dim}"
        )

        # Create hash
        return hashlib.md5(cache_key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the path to the cache file for a given cache key."""
        return self.cache_dir / f"embeddings_{cache_key}.pkl"

    def _save_embeddings_cache(self, embeddings: torch.Tensor, cache_key: str) -> None:
        """Save embeddings to cache file.

        Args:
            embeddings: The computed embeddings tensor
            cache_key: The cache key for this computation
        """
        if self.cache_dir is None:
            return

        cache_path = self._get_cache_path(cache_key)
        cache_data = {
            "embeddings": embeddings,
            "train_ids": self.train_dataset.train_ids,
        }

        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"Embeddings cached to: {cache_path}")

    def _load_embeddings_cache(self, cache_key: str) -> torch.Tensor | None:
        """Load embeddings from cache file if it exists.

        Args:
            cache_key: The cache key for this computation

        Returns:
            The cached embeddings tensor, or None if cache doesn't exist or is invalid
        """
        if self.cache_dir is None:
            return None

        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Verify train_ids match
            if cache_data["train_ids"] != self.train_dataset.train_ids:
                print("Cache invalid: training IDs don't match")
                return None

            print(f"Loaded embeddings from cache: {cache_path}")
            return cache_data["embeddings"]

        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

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
    def _compute_similarity_dict(self, pl_module) -> dict:
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

        # Try to load embeddings from cache
        cache_key = self._get_cache_key(pl_module)
        embeddings = self._load_embeddings_cache(cache_key)

        if embeddings is None:
            # Compute embeddings for all training samples (using satellite/reference images)
            print("Computing embeddings for training samples...")
            embeddings = []

            for idx in tqdm(train_ids, desc="Encoding images"):
                # Load satellite image for this sample
                sat_path = self.train_dataset.idx2sat[idx]
                img = self._load_and_preprocess_image(
                    f"{self.train_dataset.config.data_folder}/{sat_path}"
                )
                img = img.to(device)

                # Get embedding
                emb = pl_module.encode_image(img, branch="satellite")
                embeddings.append(emb.cpu())

            # Stack all embeddings (N, embedding_dim)
            embeddings = torch.cat(embeddings, dim=0)

            # Save to cache
            self._save_embeddings_cache(embeddings, cache_key)

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
            similarities = torch.matmul(batch_embeddings, embeddings.T)

            # For each sample in batch, get top-k neighbors
            for i, global_idx in enumerate(range(batch_start, batch_end)):
                idx = train_ids[global_idx]
                sim_scores = similarities[i].clone()
                sim_scores[global_idx] = -1
                _, top_indices = torch.topk(sim_scores, top_k)
                similar_ids = [train_ids[j] for j in top_indices.tolist()]
                sim_dict[idx] = similar_ids

        pl_module.train()
        return sim_dict

    def _load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.train_dataset.config.transforms_reference is not None:
            img = self.train_dataset.config.transforms_reference(image=img)["image"]

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float()

        img = F.interpolate(
            img.unsqueeze(0),
            size=self.train_dataset.config.network_input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return img
