from dataclasses import dataclass
from typing import Any, Literal, Tuple

import gtsam
import lightning as L
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch import Tensor, nn
from torchvision import models

try:
    import utm

    UTM_AVAILABLE = True
except ImportError:
    UTM_AVAILABLE = False

from taco.sensors.cvgl.measurement import CVGLMeasurement


@dataclass
class ImageRetrievalModelConfig:
    """Configuration for ImageRetrievalModel.

    Args:
        embedding_dim: Dimension of the output embedding
        learning_rate: Learning rate for optimizer
        temperature: Temperature for contrastive loss
        margin: Margin for triplet loss
        loss_type: Type of loss function to use ('combined', 'ntxent', 'triplet')
        scheduler_type: Type of learning rate scheduler ('cosine', 'step', 'plateau', 'none')
        scheduler_t_max: T_max for cosine annealing (number of epochs for full cycle)
        scheduler_eta_min: Minimum learning rate for cosine annealing
        scheduler_step_size: Step size for StepLR scheduler
        scheduler_gamma: Multiplicative factor for StepLR and ReduceLROnPlateau
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
    """

    embedding_dim: int = 512
    learning_rate: float = 1e-4
    temperature: float = 0.07
    margin: float = 0.2
    loss_type: Literal["combined", "ntxent", "triplet"] = "ntxent"
    scheduler_type: Literal["cosine", "step", "plateau", "none"] = "cosine"
    scheduler_t_max: int = 100
    scheduler_eta_min: float = 1e-6
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    weights_path: str | None = None


class ImageRetrievalModel(L.LightningModule):
    def __init__(
        self,
        encoder: pl.LightningModule,
        config: ImageRetrievalModelConfig,
        backbone_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder

        if backbone_dim is None:
            backbone_dim = self._infer_backbone_dim()

        self.projection_head = nn.Sequential(
            nn.Flatten(),  # Flatten (B, D, H, W) -> (B, D) or keep (B, D)
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, config.embedding_dim),
        )

        self.l2_norm = nn.functional.normalize
        self.ntxent_loss = losses.NTXentLoss(temperature=config.temperature)
        self.reference_embeddings: np.ndarray | None = None
        self.reference_coordinates: np.ndarray | None = None  # GPS (lat, lon)
        self.reference_origin: tuple[float, float] | None = (
            None  # (lat, lon) origin for local frame
        )
        self.use_utm: bool = False  # Whether to use UTM coordinates
        self.utm_zone: int | None = None  # UTM zone number
        self.utm_letter: str | None = None

        self.example_input_array = (
            torch.randn(1, 3, 384, 384),
            torch.randn(1, 3, 384, 384),
        )

    def _infer_backbone_dim(self) -> int:
        was_training = self.encoder.training
        dummy_input = torch.randn(1, 3, 384, 384)
        self.encoder.eval()
        with torch.no_grad():
            street_features = self.encoder.branch1(dummy_input)

        if was_training:
            self.encoder.train()

        # Flatten and get the dimension
        if len(street_features.shape) == 4:  # (B, C, H, W)
            backbone_dim = (
                street_features.shape[1] * street_features.shape[2] * street_features.shape[3]
            )
        elif len(street_features.shape) == 2:  # (B, D)
            backbone_dim = street_features.shape[1]
        else:
            raise ValueError(
                f"Unexpected encoder output shape: {street_features.shape}. "
                f"Expected (B, D) or (B, C, H, W)"
            )

        return backbone_dim

    @classmethod
    def from_sample4geo(
        cls,
        config: ImageRetrievalModelConfig,
        model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        img_size: int = 384,
        freeze_encoder: bool = False,
    ) -> "ImageRetrievalModel":
        encoder = create_sample4geo_encoder(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            freeze=freeze_encoder,
            weights_path=config.weights_path,
        )
        return cls(encoder=encoder, config=config)

    @classmethod
    def from_convnext(
        cls,
        config: ImageRetrievalModelConfig,
        model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        img_size: int = 384,
        freeze: bool = False,
    ) -> "ImageRetrievalModel":
        from taco.sensors.cvgl.models.convnext import ConvNeXtEncoder

        encoder = ConvNeXtEncoder(
            model_name=model_name,
            pretrained=pretrained,
            img_size=img_size,
            freeze=freeze,
        )

        return cls(encoder=encoder, config=config)

    @staticmethod
    def _infer_encoder_from_checkpoint(state_dict: dict) -> tuple[str, bool]:
        encoder_keys = [k for k in state_dict.keys() if k.startswith("encoder.")]

        if encoder_keys:
            # Full model checkpoint with "encoder." prefix
            # Infer from key patterns
            if any("stem" in k for k in encoder_keys):
                model_name = "convnext_base.fb_in22k_ft_in1k_384"
                print(f"Inferred encoder architecture: {model_name} (ConvNeXt-based)")
            elif any("patch_embed" in k for k in encoder_keys):
                model_name = "vit_base_patch16_224"
                print(f"Inferred encoder architecture: {model_name} (ViT-based)")
            else:
                model_name = "resnet50"
                print(f"Could not infer encoder, using default: {model_name}")
            return model_name, False

        # Check if this is an encoder-only checkpoint (Sample4Geo style, no prefix)
        # Look for encoder-specific keys without the "encoder." prefix
        all_keys = list(state_dict.keys())

        if any("model.stem" in k for k in all_keys) or any("stem.0" in k for k in all_keys):
            model_name = "convnext_base.fb_in22k_ft_in1k_384"
            print(f"Detected Sample4Geo encoder checkpoint: {model_name} (ConvNeXt-based)")
            return model_name, True
        elif any("model.patch_embed" in k for k in all_keys) or any(
            "patch_embed" in k for k in all_keys
        ):
            model_name = "vit_base_patch16_224"
            print(f"Detected Sample4Geo encoder checkpoint: {model_name} (ViT-based)")
            return model_name, True
        elif any("model.layer" in k for k in all_keys) or any("layer1" in k for k in all_keys):
            model_name = "resnet50"
            print(f"Detected Sample4Geo encoder checkpoint: {model_name} (ResNet-based)")
            return model_name, True
        else:
            print("Warning: Could not infer encoder architecture, using default: resnet50")
            return "resnet50", False

    @staticmethod
    def _infer_config_from_state_dict(state_dict: dict) -> ImageRetrievalModelConfig:
        config = ImageRetrievalModelConfig()

        # Try to infer embedding_dim from projection head output layer
        if "projection_head.6.weight" in state_dict:
            # projection_head.6 is the final linear layer: Linear(1024, embedding_dim)
            embedding_dim = state_dict["projection_head.6.weight"].shape[0]
            config.embedding_dim = int(embedding_dim)
            print(f"Inferred embedding_dim={embedding_dim} from state dict")
        elif "projection_head.4.weight" in state_dict:
            # Alternative structure
            embedding_dim = state_dict["projection_head.4.weight"].shape[0]
            config.embedding_dim = int(embedding_dim)
            print(f"Inferred embedding_dim={embedding_dim} from state dict")

        return config

    @staticmethod
    def _extract_config_from_checkpoint(checkpoint: dict) -> ImageRetrievalModelConfig:
        if "hyper_parameters" in checkpoint and "config" in checkpoint["hyper_parameters"]:
            return ImageRetrievalModelConfig(**checkpoint["hyper_parameters"]["config"])
        return ImageRetrievalModelConfig()

    @staticmethod
    def _report_loading_status(
        missing_keys: list, unexpected_keys: list, checkpoint_path: str, strict: bool
    ) -> None:
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {len(missing_keys)} keys")
            for key in missing_keys[:10]:
                print(f"  - {key}")

        if unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            for key in unexpected_keys[:10]:
                print(f"  - {key}")

        if not missing_keys and not unexpected_keys:
            print(f"✓ Successfully loaded full model from {checkpoint_path}")
        else:
            print(f"⚠ Loaded model from {checkpoint_path} with warnings (strict={strict})")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        model_name: str | None = None,
        img_size: int = 384,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
        config: ImageRetrievalModelConfig | None = None,
    ) -> "ImageRetrievalModel":
        print(f"Loading from: {checkpoint_path}")
        model_state_dict = torch.load(checkpoint_path, map_location=map_location)

        if isinstance(model_state_dict, dict) and "state_dict" in model_state_dict:
            checkpoint_dict = model_state_dict
            model_state_dict = checkpoint_dict["state_dict"]
        else:
            checkpoint_dict = {}

        # Infer model name if not provided
        is_encoder_only = False
        if model_name is None:
            model_name, is_encoder_only = cls._infer_encoder_from_checkpoint(model_state_dict)
        else:
            # Check if it's encoder-only even when model_name is provided
            _, is_encoder_only = cls._infer_encoder_from_checkpoint(model_state_dict)

        # Get or create config
        if config is None:
            if checkpoint_dict and ("hyper_parameters" in checkpoint_dict):
                # PyTorch Lightning checkpoint with hyperparameters
                config = cls._extract_config_from_checkpoint(checkpoint_dict)
            else:
                # Raw state dict - infer config from weights if not encoder-only
                if not is_encoder_only:
                    config = cls._infer_config_from_state_dict(model_state_dict)
                else:
                    config = ImageRetrievalModelConfig()

        # Create encoder and model
        if is_encoder_only:
            # Load encoder-only checkpoint (Sample4Geo style)
            encoder = create_sample4geo_encoder(
                model_name=model_name,
                pretrained=False,
                img_size=img_size,
                freeze=False,
                weights_path=checkpoint_path,  # Load weights directly
            )
            model = cls(encoder=encoder, config=config)
            print(f"✓ Loaded encoder-only checkpoint from {checkpoint_path}")
            return model
        else:
            # Load full model checkpoint
            encoder = create_sample4geo_encoder(
                model_name=model_name,
                pretrained=False,
                img_size=img_size,
                freeze=False,
            )
            model = cls(encoder=encoder, config=config)

            # Load state dict - Sample4Geo style with strict=False by default
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=strict)
            cls._report_loading_status(missing_keys, unexpected_keys, checkpoint_path, strict)

        return model

    def forward(self, img_1: Tensor, img_2: Tensor = None) -> Tensor:
        # Extract features from encoder
        street_features, reference_features = self.encoder(img_1, img_2)
        # Project to embedding dimension
        street_embeddings = self.projection_head(street_features)
        reference_embeddings = self.projection_head(reference_features)
        # L2 normalize embeddings
        street_embeddings = self.l2_norm(street_embeddings, p=2, dim=1)
        reference_embeddings = self.l2_norm(reference_embeddings, p=2, dim=1)
        return street_embeddings, reference_embeddings

    def compute_triplet_loss(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tensor:
        """Compute triplet loss for metric learning.

        Args:
            anchor: Anchor image embeddings (B, D).
            positive: Positive image embeddings (B, D).
            negative: Negative image embeddings (B, D).

        Returns:
            Triplet loss value.
        """
        # Compute distances
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)

        # Triplet loss with margin
        loss = F.relu(pos_distance - neg_distance + self.config.margin)

        return loss.mean()

    def compute_contrastive_loss(
        self,
        embeddings: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            embeddings: Image embeddings (B, D).
            labels: Place labels (B,).

        Returns:
            Contrastive loss value.
        """
        similarity = torch.matmul(embeddings, embeddings.T) / self.config.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(mask.size(0), device=mask.device)

        # Compute log probabilities
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos.mean()

        return loss

    def compute_ntxent_loss(
        self,
        query_emb: Tensor,
        reference_emb: Tensor,
        labels: Tensor,
    ) -> Tensor:
        # Concatenate query and reference embeddings
        # This creates pairs where query[i] and reference[i] have the same label
        embeddings = torch.cat([query_emb, reference_emb], dim=0)

        # Duplicate labels for both query and reference
        # labels[i] corresponds to query_emb[i] and reference_emb[i]
        labels_combined = torch.cat([labels, labels], dim=0)
        loss = self.ntxent_loss(embeddings, labels_combined)
        return loss

    def on_train_epoch_start(self) -> None:
        self.train_query_embs: list[Tensor] = []
        self.train_reference_embs: list[Tensor] = []

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        query_img, reference_img, labels = batch
        query_emb, reference_emb = self.forward(query_img, reference_img)

        if self.config.loss_type == "ntxent":
            # Use NT-Xent loss only
            loss = self.compute_ntxent_loss(query_emb, reference_emb, labels)
            self.log("train/ntxent_loss", loss, sync_dist=True)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        elif self.config.loss_type == "triplet":
            # Use triplet loss only
            negative_emb = torch.roll(reference_emb, shifts=1, dims=0)
            loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)
            self.log("train/triplet_loss", loss, sync_dist=True)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        else:  # self.config.loss_type == "combined"
            # Sample negatives: shift reference embeddings to create mismatched pairs
            # Since the dataset shuffle ensures unique labels in each batch,
            # rolling by 1 gives us guaranteed negatives
            negative_emb = torch.roll(reference_emb, shifts=1, dims=0)

            # Compute triplet loss (query as anchor, reference as positive, rolled reference as negative)
            triplet_loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)

            # Compute contrastive loss on query embeddings
            contrastive_loss = self.compute_contrastive_loss(query_emb, labels)

            # Combined loss
            loss = triplet_loss + 0.5 * contrastive_loss

            # Log metrics
            self.log("train/triplet_loss", triplet_loss, sync_dist=True)
            self.log("train/contrastive_loss", contrastive_loss, sync_dist=True)
            self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        # Accumulate embeddings for epoch-end metrics
        self.train_query_embs.append(query_emb.detach())
        self.train_reference_embs.append(reference_emb.detach())

        return loss

    def on_train_epoch_end(self) -> None:
        query_emb = torch.cat(self.train_query_embs, dim=0)
        reference_emb = torch.cat(self.train_reference_embs, dim=0)

        # Compute distances over the full training epoch
        pos_distance = F.pairwise_distance(query_emb, reference_emb, p=2)
        negative_emb = torch.roll(reference_emb, shifts=1, dims=0)
        neg_distance = F.pairwise_distance(query_emb, negative_emb, p=2)
        accuracy = (pos_distance < neg_distance).float().mean()

        # Compute Top-K recall over the full epoch
        recall_at_k = self.compute_recall_at_k(query_emb, reference_emb, k_values=(1, 5, 10))

        self.log("train/accuracy", accuracy, sync_dist=True)
        self.log("train/pos_distance", pos_distance.mean(), sync_dist=True)
        self.log("train/neg_distance", neg_distance.mean(), sync_dist=True)
        self.log("train@1", recall_at_k[1] * 100, prog_bar=True, sync_dist=True)
        self.log("train@5", recall_at_k[5] * 100, prog_bar=True, sync_dist=True)
        self.log("train@10", recall_at_k[10] * 100, prog_bar=True, sync_dist=True)

    def compute_recall_at_k(
        self,
        query_emb: Tensor,
        reference_emb: Tensor,
        k_values: tuple[int, ...] = (1, 5, 10),
    ) -> dict[int, float]:
        """Compute recall@K metrics for image retrieval.

        For each query, computes similarity with all references and checks
        if the correct match (same index) appears in the top-K results.

        Args:
            query_emb: Query embeddings (B, D).
            reference_emb: Reference embeddings (B, D).
            k_values: List of K values to compute recall for.

        Returns:
            Dictionary mapping K to recall@K as a fraction in [0, 1].
        """
        batch_size = query_emb.size(0)

        # Compute similarity matrix: (B, B)
        # Each row i contains similarities between query i and all references
        similarity_matrix = torch.matmul(query_emb, reference_emb.T)

        # For each query (row), rank references by similarity (descending)
        # Get indices of references sorted by similarity
        _, sorted_indices = torch.sort(similarity_matrix, dim=1, descending=True)

        # Ground truth: correct match for query i is reference i
        correct_indices = torch.arange(batch_size, device=query_emb.device).unsqueeze(1)

        # Compute recall@K for each K value
        recall_at_k = {}
        for k in k_values:
            # Get top-K predictions for each query
            top_k_indices = sorted_indices[:, :k]

            # Check if correct index is in top-K for each query
            matches = (top_k_indices == correct_indices).any(dim=1).float()

            # Recall@K is the fraction of queries where correct match is in top-K
            recall_at_k[k] = matches.mean().item()

        return recall_at_k

    def on_validation_epoch_start(self) -> None:
        self.val_query_embs: list[Tensor] = []
        self.val_reference_embs: list[Tensor] = []

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Validation step.

        Args:
            batch: Tuple of (query_img, reference_img, labels) from CVUSADataset.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        query_img, reference_img, labels = batch

        # Compute embeddings
        query_emb, reference_emb = self(
            query_img, reference_img
        )  # anchor embeddings (ground view) and positive embeddings (satellite view)

        # Sample negatives: shift reference embeddings to create mismatched pairs
        negative_emb = torch.roll(reference_emb, shifts=1, dims=0)

        # Compute loss based on loss_type configuration
        if self.config.loss_type == "ntxent":
            # Use NT-Xent loss only
            loss = self.compute_ntxent_loss(query_emb, reference_emb, labels)
            self.log("val/ntxent_loss", loss, sync_dist=True)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        elif self.config.loss_type == "triplet":
            # Use triplet loss only
            loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)
            self.log("val/triplet_loss", loss, sync_dist=True)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        else:  # self.config.loss_type == "combined"
            # Compute triplet loss
            triplet_loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)

            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(query_emb, labels)

            # Combined loss
            loss = triplet_loss + 0.5 * contrastive_loss

            # Log metrics
            self.log("val/triplet_loss", triplet_loss, sync_dist=True)
            self.log("val/contrastive_loss", contrastive_loss, sync_dist=True)
            self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # Accumulate embeddings for epoch-end metrics
        self.val_query_embs.append(query_emb.detach())
        self.val_reference_embs.append(reference_emb.detach())

        return loss

    def on_validation_epoch_end(self) -> None:
        query_emb = torch.cat(self.val_query_embs, dim=0)
        reference_emb = torch.cat(self.val_reference_embs, dim=0)

        # Compute distances over the full validation set
        pos_distance = F.pairwise_distance(query_emb, reference_emb, p=2)
        negative_emb = torch.roll(reference_emb, shifts=1, dims=0)
        neg_distance = F.pairwise_distance(query_emb, negative_emb, p=2)
        accuracy = (pos_distance < neg_distance).float().mean()

        # Compute Top-K recall over the full gallery
        recall_at_k = self.compute_recall_at_k(query_emb, reference_emb, k_values=(1, 5, 10))

        self.log("val/accuracy", accuracy, sync_dist=True)
        self.log("val/pos_distance", pos_distance.mean(), sync_dist=True)
        self.log("val/neg_distance", neg_distance.mean(), sync_dist=True)
        self.log("val@1", recall_at_k[1] * 100, prog_bar=True, sync_dist=True)
        self.log("val@5", recall_at_k[5] * 100, prog_bar=True, sync_dist=True)
        self.log("val@10", recall_at_k[10] * 100, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01,
        )

        # Configure scheduler based on config
        if self.config.scheduler_type == "none":
            return {"optimizer": optimizer}

        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.scheduler_t_max,
                eta_min=self.config.scheduler_eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        if self.config.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        if self.config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.scheduler_gamma,
                patience=self.config.scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                },
            }

        raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def encode_image(
        self,
        image: np.ndarray,
        device: torch.device | None = None,
        branch: str = "street",
    ) -> np.ndarray:
        """Encode a single image to embedding.

        Args:
            image: Input image as numpy array (H, W, 3) in RGB, [0, 255].
            device: Device to run inference on.

        Returns:
            Image embedding as numpy array (embedding_dim,).
        """
        if device is None:
            device = next(self.parameters()).device

        # Convert to tensor and normalize
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        # Scale to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0

        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        image = (image - mean) / std

        # Add batch dimension and move to device
        image = image.unsqueeze(0).to(device)

        # Forward pass
        self.eval()

        if branch == "street":
            features = self.encoder.branch1(image)
        else:
            features = self.encoder.branch2(image)

        features = self.projection_head(features)

        # L2 normalize embeddings
        embedding = self.l2_norm(features, p=2, dim=1)

        return embedding.cpu()  # .numpy()[0]  # type: ignore[no-any-return]

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding (D,).
            embedding2: Second embedding (D,).

        Returns:
            Cosine similarity in [-1, 1].
        """
        # Convert to tensors
        emb1 = torch.from_numpy(embedding1)
        emb2 = torch.from_numpy(embedding2)

        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=0)

        return float(similarity.item())

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        database_embeddings: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k similar images from database.

        Args:
            query_embedding: Query image embedding (D,).
            database_embeddings: Database embeddings (N, D).
            top_k: Number of top results to return.

        Returns:
            Tuple of (indices, similarities) for top-k matches.
        """
        # Convert to tensors
        query = torch.from_numpy(query_embedding).unsqueeze(0)
        database = torch.from_numpy(database_embeddings)

        # Compute similarities
        similarities = F.cosine_similarity(query, database, dim=1)

        # Get top-k
        top_k = min(top_k, len(similarities))
        top_similarities, top_indices = torch.topk(similarities, top_k)

        return top_indices.numpy(), top_similarities.numpy()

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def build_reference_database(
        self,
        images: np.ndarray | list[np.ndarray],
        coordinates: np.ndarray,
        origin: tuple[float, float] | None = None,
        use_utm: bool = True,
        device: torch.device | None = None,
        batch_size: int = 32,
    ) -> None:
        """Build a reference database from input images and GPS coordinates.

        This method processes a collection of images, computes their embeddings,
        and stores them along with their corresponding GPS coordinates for later
        querying. CVGL provides position-only measurements; heading comes from IMU.

        Args:
            images: Reference images as either:
                - numpy array of shape (N, H, W, 3) in RGB, [0, 255], or
                - list of N numpy arrays each of shape (H, W, 3) in RGB, [0, 255]
            coordinates: GPS coordinates of shape (N, 2) where each row is [latitude, longitude].
            origin: Optional reference origin (latitude, longitude) for local coordinate conversion.
                If None, uses the first coordinate as origin. Only used if use_utm=False.
            use_utm: If True, use UTM coordinates for consistent global frame (recommended).
                If False, use local tangent plane approximation (only for small areas <10km).
            device: Device to run inference on. If None, uses model's current device.
            batch_size: Batch size for processing images.

        Raises:
            ValueError: If the number of images and coordinates don't match.
            ImportError: If use_utm=True but utm package is not installed.
        """
        if device is None:
            device = next(self.parameters()).device

        # Handle list of images
        if isinstance(images, list):
            num_images = len(images)
        else:
            num_images = len(images)

        # Validate inputs
        if len(coordinates) != num_images:
            raise ValueError(
                f"Number of images ({num_images}) must match number of coordinates ({len(coordinates)})"
            )

        # Store coordinates (headings not needed - CVGL is position-only)
        self.reference_coordinates = np.array(coordinates, dtype=np.float32)

        # Set coordinate system
        self.use_utm = use_utm

        if use_utm:
            # Convert first coordinate to determine UTM zone
            if not UTM_AVAILABLE:
                raise ImportError(
                    "utm package is required for UTM conversion. Install it with: pip install utm"
                )
            _, _, zone_number, zone_letter = utm.from_latlon(coordinates[0, 0], coordinates[0, 1])
            self.utm_zone = zone_number
            self.utm_letter = zone_letter
            # Origin not used for UTM (uses UTM zone origin)
            self.reference_origin = None
        else:
            # Store or compute origin for local coordinate frame
            if origin is not None:
                self.reference_origin = origin
            else:
                # Use first coordinate as origin
                self.reference_origin = (float(coordinates[0, 0]), float(coordinates[0, 1]))
            self.utm_zone = None
            self.utm_letter = None

        # Process images in batches to compute embeddings
        embeddings_list = []
        self.eval()

        for i in range(0, num_images, batch_size):
            batch_end = min(i + batch_size, num_images)

            # Get batch of images
            if isinstance(images, list):
                batch_images = images[i:batch_end]
            else:
                batch_images = images[i:batch_end]

            # Prepare batch tensor
            batch_tensors = []
            for img in batch_images:
                # Convert to tensor and normalize
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                img_tensor = img_tensor / 255.0

                # Normalize using ImageNet statistics
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std

                batch_tensors.append(img_tensor)

            # Stack into batch and move to device
            batch_tensor = torch.stack(batch_tensors).to(device)

            # Compute embeddings (using reference branch, pass same tensor twice)
            _, batch_embeddings = self(batch_tensor, batch_tensor)
            embeddings_list.append(batch_embeddings.cpu().numpy())

        # Concatenate all embeddings
        self.reference_embeddings = np.vstack(embeddings_list)

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def query_database(
        self,
        query_image: np.ndarray,
        top_k: int = 5,
        device: torch.device | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Query the reference database with an image.

        Embeds the query image, searches the reference database, and returns
        the GPS coordinates of the top-K most similar reference images.

        Args:
            query_image: Query image as numpy array (H, W, 3) in RGB, [0, 255].
            top_k: Number of top results to return.
            device: Device to run inference on. If None, uses model's current device.

        Returns:
            Tuple of (coordinates, similarities) where:
                - coordinates: GPS coordinates of top-K matches, shape (K, 2) [latitude, longitude]
                - similarities: Cosine similarities for top-K matches, shape (K,)

        Raises:
            RuntimeError: If the reference database has not been built yet.
        """
        if self.reference_embeddings is None or self.reference_coordinates is None:
            raise RuntimeError(
                "Reference database has not been built. Call build_reference_database() first."
            )

        # Encode query image
        query_embedding = self.encode_image(query_image, device=device)

        # Retrieve top-K similar embeddings
        top_indices, top_similarities = self.retrieve_similar(
            query_embedding,
            self.reference_embeddings,
            top_k=top_k,
        )

        # Get corresponding coordinates
        top_coordinates = self.reference_coordinates[top_indices]

        return top_coordinates, top_similarities

    @staticmethod
    def gps_to_utm(
        lat: float,
        lon: float,
    ) -> tuple[float, float, int, str]:
        """Convert GPS coordinates to UTM coordinates.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            Tuple of (easting, northing, zone_number, zone_letter) in meters.

        Raises:
            ImportError: If utm package is not installed.
        """
        if not UTM_AVAILABLE:
            raise ImportError(
                "utm package is required for UTM conversion. Install it with: pip install utm"
            )
        easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
        return float(easting), float(northing), zone_number, zone_letter

    @staticmethod
    def utm_to_gps(
        easting: float,
        northing: float,
        zone_number: int,
        zone_letter: str,
    ) -> tuple[float, float]:
        """Convert UTM coordinates to GPS coordinates.

        Args:
            easting: UTM easting in meters.
            northing: UTM northing in meters.
            zone_number: UTM zone number.
            zone_letter: UTM zone letter.

        Returns:
            Tuple of (latitude, longitude) in degrees.

        Raises:
            ImportError: If utm package is not installed.
        """
        if not UTM_AVAILABLE:
            raise ImportError(
                "utm package is required for UTM conversion. Install it with: pip install utm"
            )
        lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
        return float(lat), float(lon)

    @staticmethod
    def gps_to_local_xy(
        lat: float,
        lon: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        """Convert GPS coordinates to local XY coordinates.

        Uses a local tangent plane approximation (ENU frame) centered at the origin.
        This is accurate for distances up to ~10km from the origin.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.
            origin_lat: Origin latitude in degrees.
            origin_lon: Origin longitude in degrees.

        Returns:
            Tuple of (x, y) in meters, where:
                - x is positive East
                - y is positive North
        """
        # Earth radius in meters
        R = 6371000.0

        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        origin_lat_rad = np.radians(origin_lat)
        origin_lon_rad = np.radians(origin_lon)

        # Local tangent plane approximation
        dlat = lat_rad - origin_lat_rad
        dlon = lon_rad - origin_lon_rad

        # Convert to meters
        x = R * dlon * np.cos(origin_lat_rad)  # East
        y = R * dlat  # North

        return float(x), float(y)

    @staticmethod
    def local_xy_to_gps(
        x: float,
        y: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        """Convert local XY coordinates to GPS coordinates.

        Uses a local tangent plane approximation (ENU frame).

        Args:
            x: X coordinate in meters (East).
            y: Y coordinate in meters (North).
            origin_lat: Origin latitude in degrees.
            origin_lon: Origin longitude in degrees.

        Returns:
            Tuple of (latitude, longitude) in degrees.
        """
        # Earth radius in meters
        R = 6371000.0

        # Convert to radians
        origin_lat_rad = np.radians(origin_lat)
        origin_lon_rad = np.radians(origin_lon)

        # Convert from meters to radians
        dlat = y / R
        dlon = x / (R * np.cos(origin_lat_rad))

        # Compute final coordinates
        lat_rad = origin_lat_rad + dlat
        lon_rad = origin_lon_rad + dlon

        # Convert to degrees
        lat = np.degrees(lat_rad)
        lon = np.degrees(lon_rad)

        return float(lat), float(lon)

    def similarity_to_covariance(
        self,
        similarities: np.ndarray,
        base_position_std: float = 5.0,
        min_position_std: float = 1.0,
    ) -> np.ndarray:
        """Estimate position covariance matrix from similarity scores.

        Higher similarity scores result in lower uncertainty (smaller covariance).
        Uses the top match similarity to estimate uncertainty.

        Note: CVGL provides position-only measurements. Heading comes from IMU.

        Args:
            similarities: Array of similarity scores (K,) in range [-1, 1].
            base_position_std: Base standard deviation for position when similarity is 0 (meters).
            min_position_std: Minimum position std dev at perfect similarity (meters).

        Returns:
            2x2 covariance matrix for (x, y) position.
        """
        # Use top match similarity to estimate uncertainty
        top_similarity = similarities[0]

        # Map similarity [0, 1] to uncertainty
        # similarity = 1.0 → minimum uncertainty
        # similarity = 0.0 → base uncertainty
        # Clamp similarity to [0, 1] range (cosine similarity can be negative)
        confidence = np.clip(top_similarity, 0.0, 1.0)

        # Linear interpolation between base and minimum uncertainty
        position_std = base_position_std - confidence * (base_position_std - min_position_std)

        # Create diagonal 2x2 covariance matrix for position
        covariance = np.diag([position_std**2, position_std**2])

        return covariance

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def query_database_as_measurement(
        self,
        query_image: np.ndarray,
        timestamp: float,
        top_k: int = 5,
        device: torch.device | None = None,
        use_weighted_average: bool = True,
        base_position_std: float = 5.0,
        current_yaw: float | None = None,
    ) -> CVGLMeasurement:
        """Query database and return result as CVGLMeasurement for pose graph.

        This method combines query_database with coordinate conversion and
        covariance estimation to produce a POSITION-ONLY measurement ready
        for insertion into the GTSAM pose graph.

        Note: CVGL only provides position (x, y) measurements. Heading should
        come from IMU integration, not from visual localization.

        Args:
            query_image: Query image as numpy array (H, W, 3) in RGB, [0, 255].
            timestamp: Timestamp of the measurement.
            top_k: Number of top results to consider for position estimation.
            device: Device to run inference on. If None, uses model's current device.
            use_weighted_average: If True, compute position as weighted average of top-K matches.
                If False, use only the top match.
            base_position_std: Base standard deviation for position uncertainty (meters).
            current_yaw: Optional current heading from IMU (radians). Stored but not used
                for measurement (position-only measurement).

        Returns:
            CVGLMeasurement object with position-only measurement.

        Raises:
            RuntimeError: If the reference database has not been built yet.
        """
        if self.reference_embeddings is None or self.reference_coordinates is None:
            raise RuntimeError(
                "Reference database has not been built. Call build_reference_database() first."
            )

        # Query the database
        top_gps_coords, top_similarities = self.query_database(
            query_image=query_image,
            top_k=top_k,
            device=device,
        )

        # Estimate position in metric coordinates (UTM or local XY)
        if self.use_utm:
            # Use UTM coordinates
            if use_weighted_average and top_k > 1:
                # Compute weighted average of top-K matches
                weights = np.clip(top_similarities, 0.0, 1.0)
                weights = weights / (weights.sum() + 1e-8)  # Normalize

                # Convert all top-K GPS to UTM
                utm_positions = []
                for i in range(len(top_gps_coords)):
                    easting, northing, _, _ = self.gps_to_utm(
                        lat=top_gps_coords[i, 0],
                        lon=top_gps_coords[i, 1],
                    )
                    utm_positions.append([easting, northing])

                utm_positions = np.array(utm_positions)

                # Weighted average in UTM space
                position = np.average(utm_positions, axis=0, weights=weights)
            else:
                # Use only top match
                easting, northing, _, _ = self.gps_to_utm(
                    lat=top_gps_coords[0, 0],
                    lon=top_gps_coords[0, 1],
                )
                position = np.array([easting, northing])
        else:
            # Use local tangent plane coordinates
            if self.reference_origin is None:
                raise RuntimeError("Reference origin not set for local coordinate conversion.")

            if use_weighted_average and top_k > 1:
                # Compute weighted average of top-K matches
                weights = np.clip(top_similarities, 0.0, 1.0)
                weights = weights / (weights.sum() + 1e-8)  # Normalize

                # Convert all top-K GPS to local XY
                local_positions = []
                for i in range(len(top_gps_coords)):
                    x, y = self.gps_to_local_xy(
                        lat=top_gps_coords[i, 0],
                        lon=top_gps_coords[i, 1],
                        origin_lat=self.reference_origin[0],
                        origin_lon=self.reference_origin[1],
                    )
                    local_positions.append([x, y])

                local_positions = np.array(local_positions)

                # Weighted average
                position = np.average(local_positions, axis=0, weights=weights)
            else:
                # Use only top match
                position_x, position_y = self.gps_to_local_xy(
                    lat=top_gps_coords[0, 0],
                    lon=top_gps_coords[0, 1],
                    origin_lat=self.reference_origin[0],
                    origin_lon=self.reference_origin[1],
                )
                position = np.array([position_x, position_y])

        # Estimate position covariance from similarity scores
        # CVGL provides position-only measurements (no heading)
        position_covariance = self.similarity_to_covariance(
            similarities=top_similarities,
            base_position_std=base_position_std,
        )

        # Add geographic coordinates for debug
        coord = top_gps_coords[0]  # Use top match GPS for reference

        # Create CVGLMeasurement (position-only)
        measurement = CVGLMeasurement(
            timestamp=timestamp,
            position=position,
            position_covariance=position_covariance,
            coordinates=coord,
            confidence=float(np.clip(top_similarities[0], 0.0, 1.0)),
            num_inliers=top_k,  # Use top_k as a proxy for inliers
            yaw=current_yaw,  # Store current yaw from IMU (not estimated)
            image_id=None,
        )

        return measurement


def create_convnext_encoder(pretrained: bool = True, freeze: bool = False) -> nn.Module:
    """Create a ConvNeXt-Base encoder for image retrieval.

    Args:
        pretrained: Whether to use pretrained ConvNeXt weights
        freeze: If True, freeze encoder weights

    Returns:
        Encoder module that outputs features (B, 1024, 1, 1)
    """
    weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
    convnext = models.convnext_base(weights=weights)

    # Create encoder from ConvNeXt features and avgpool
    encoder = nn.Sequential(
        convnext.features,
        convnext.avgpool,
    )

    # Freeze encoder if requested
    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder


def create_sample4geo_encoder(
    model_name: str = "convnext_base.fb_in22k_ft_in1k_384",
    pretrained: bool = True,
    img_size: int = 384,
    freeze: bool = False,
    weights_path: str | None = None,
) -> nn.Module:
    from taco.sensors.cvgl.models.sample4geo import Sample4GeoEncoder

    # This is the two
    encoder = Sample4GeoEncoder(
        model_name=model_name,
        pretrained=pretrained,
        img_size=img_size,
        freeze=freeze,
    )

    # Load custom weights if provided
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location="cpu")

        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            # PyTorch Lightning checkpoint
            state_dict = state_dict["state_dict"]

        # Process state dict based on key structure
        encoder_state_dict = {}

        # Check if keys have "encoder." prefix (full model checkpoint)
        has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())

        if has_encoder_prefix:
            # Full model checkpoint - extract encoder weights
            for key, value in state_dict.items():
                if key.startswith("encoder."):
                    # Remove 'encoder.' prefix
                    new_key = key.replace("encoder.", "", 1)
                    encoder_state_dict[new_key] = value
        else:
            # Sample4Geo encoder-only checkpoint (no prefix) - use as-is
            encoder_state_dict = state_dict

        # Load with strict=False to allow partial loading
        missing_keys, unexpected_keys = encoder.load_state_dict(encoder_state_dict, strict=False)

        if missing_keys or unexpected_keys:
            print(f"Loaded encoder weights from {weights_path} (strict=False)")
            if missing_keys and len(missing_keys) <= 5:
                print(f"  Missing keys: {missing_keys[:5]}")
            if unexpected_keys and len(unexpected_keys) <= 5:
                print(f"  Unexpected keys: {unexpected_keys[:5]}")
        else:
            print(f"✓ Loaded encoder weights from {weights_path}")

    if freeze:
        encoder.eval()
    else:
        encoder.train()

    return encoder


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create configuration with NT-Xent loss
    config = ImageRetrievalModelConfig(
        embedding_dim=512,
        learning_rate=1e-4,
        temperature=0.07,
        margin=0.2,
        loss_type="ntxent",  # Options: "combined", "ntxent", "triplet"
    )

    print("=" * 60)
    print("Example 1: Sample4Geo Encoder (RECOMMENDED DEFAULT)")
    print("=" * 60)

    # Method 1: Using the convenience class method (easiest)
    model_default = ImageRetrievalModel.from_sample4geo(
        config=config,
        model_name="resnet50",
        pretrained=True,
        img_size=384,
    )
    print(f"Model created with embedding_dim={config.embedding_dim}, loss_type={config.loss_type}")

    # Test forward pass
    dummy_image = torch.randn(2, 3, 384, 384)
    embeddings = model_default(dummy_image)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test encoding a single image
    test_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    embedding = model_default.encode_image(test_image)
    print(f"Single image embedding shape: {embedding.shape}")

    print("\n" + "=" * 60)
    print("Example 2: Loading Custom Encoder Weights")
    print("=" * 60)

    # Load model with custom pre-trained encoder weights
    # Uncomment to use with your own encoder weights:
    # model_custom = ImageRetrievalModel.from_sample4geo(
    #     config=config,
    #     model_name="resnet50",
    #     pretrained=True,  # Start with ImageNet weights
    #     encoder_weights_path="/path/to/your/encoder_weights.pth",
    # )
    print("Example: Load custom encoder weights with encoder_weights_path parameter")

    # Or load a full model checkpoint (encoder + projection head):
    # model_full = ImageRetrievalModel.load_from_checkpoint(
    #     checkpoint_path="/path/to/your/model_checkpoint.ckpt",
    #     model_name="resnet50",
    #     img_size=384,
    # )
    print("Example: Load full model checkpoint with load_from_checkpoint()")

    print("\n" + "=" * 60)
    print("Example 3: Using Different Timm Models")
    print("=" * 60)

    # Method 2: Create encoder manually (more flexible)
    encoder_vit = create_sample4geo_encoder(
        model_name="vit_base_patch16_224", pretrained=True, img_size=224, freeze=False
    )

    model_vit = ImageRetrievalModel(encoder=encoder_vit, config=config)
    print("Model created with ViT-Base encoder")

    # Test forward pass
    dummy_image_224 = torch.randn(2, 3, 224, 224)
    embeddings = model_vit(dummy_image_224)
    print(f"Embeddings shape: {embeddings.shape}")

    print("\n" + "=" * 60)
    print("Example 4: ConvNeXt-Base Encoder (torchvision)")
    print("=" * 60)

    # Using torchvision ConvNeXt instead of timm
    encoder_convnext = create_convnext_encoder(pretrained=True, freeze=False)

    # backbone_dim=1024 for ConvNeXt-Base (can be inferred automatically if omitted)
    model_convnext = ImageRetrievalModel(encoder=encoder_convnext, config=config, backbone_dim=1024)
    print("Model created with ConvNeXt-Base encoder")

    # Test forward pass
    embeddings = model_convnext(dummy_image_224)
    print(f"Embeddings shape: {embeddings.shape}")
