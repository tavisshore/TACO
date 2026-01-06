"""PyTorch Lightning model for image retrieval using ConvNeXt."""

from typing import Any, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


class ImageRetrievalModel(L.LightningModule):
    """PyTorch Lightning model for image retrieval using ConvNeXt-Tiny.

    Uses a pre-trained ConvNeXt-Tiny backbone to encode images into
    low-dimensional descriptors for visual place recognition.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        pretrained: bool = True,
        learning_rate: float = 1e-4,
        temperature: float = 0.07,
        margin: float = 0.2,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the image retrieval model.

        Args:
            embedding_dim: Dimension of the output embedding.
            pretrained: Whether to use pretrained ConvNeXt weights.
            learning_rate: Learning rate for optimizer.
            temperature: Temperature for contrastive loss.
            margin: Margin for triplet loss.
            freeze_backbone: If True, freeze ConvNeXt backbone weights.
        """
        super().__init__()
        self.save_hyperparameters()

        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.margin = margin

        # Load pre-trained ConvNeXt-Tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        convnext = models.convnext_tiny(weights=weights)

        # ConvNeXt-Tiny structure:
        # - features: the convolutional backbone
        # - avgpool: adaptive average pooling
        # - classifier: Sequential(LayerNorm, Flatten, Linear)

        # We want to use everything except the final Linear layer
        # Keep: features + avgpool + LayerNorm + Flatten
        self.backbone = nn.Sequential(
            convnext.features,
            convnext.avgpool,
        )

        # Get the feature dimension after pooling
        # ConvNeXt-Tiny: after avgpool, features are (batch, 768, 1, 1)
        # After flatten: (batch, 768)
        backbone_dim = 768

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head to map to embedding dimension
        self.projection_head = nn.Sequential(
            nn.Flatten(),  # Flatten (B, 768, 1, 1) -> (B, 768)
            nn.LayerNorm(backbone_dim),  # LayerNorm like original ConvNeXt
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, embedding_dim),
        )

        # L2 normalization layer
        self.l2_norm = nn.functional.normalize

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass to compute image embeddings.

        Args:
            images: Batch of images (B, C, H, W).

        Returns:
            Normalized embeddings (B, embedding_dim).
        """
        # Extract features from backbone
        features = self.backbone(images)

        # Project to embedding dimension
        embeddings = self.projection_head(features)

        # L2 normalize embeddings
        embeddings = self.l2_norm(embeddings, p=2, dim=1)

        return embeddings

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
        loss = F.relu(pos_distance - neg_distance + self.margin)

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
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (same place)
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

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Training step.

        Args:
            batch: Tuple of (anchor, positive, negative, labels).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        anchor_img, positive_img, negative_img, labels = batch

        # Compute embeddings
        anchor_emb = self(anchor_img)
        positive_emb = self(positive_img)
        negative_emb = self(negative_img)

        # Compute triplet loss
        triplet_loss = self.compute_triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Compute contrastive loss on anchors
        contrastive_loss = self.compute_contrastive_loss(anchor_emb, labels)

        # Combined loss
        loss = triplet_loss + 0.5 * contrastive_loss

        # Log metrics
        self.log("train/triplet_loss", triplet_loss, prog_bar=True)
        self.log("train/contrastive_loss", contrastive_loss, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Validation step.

        Args:
            batch: Tuple of (anchor, positive, negative, labels).
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        anchor_img, positive_img, negative_img, labels = batch

        # Compute embeddings
        anchor_emb = self(anchor_img)
        positive_emb = self(positive_img)
        negative_emb = self(negative_img)

        # Compute triplet loss
        triplet_loss = self.compute_triplet_loss(anchor_emb, positive_emb, negative_emb)

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(anchor_emb, labels)

        # Combined loss
        loss = triplet_loss + 0.5 * contrastive_loss

        # Compute metrics
        pos_distance = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        neg_distance = F.pairwise_distance(anchor_emb, negative_emb, p=2)
        accuracy = (pos_distance < neg_distance).float().mean()

        # Log metrics
        self.log("val/triplet_loss", triplet_loss, prog_bar=True)
        self.log("val/contrastive_loss", contrastive_loss, prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/accuracy", accuracy, prog_bar=True)
        self.log("val/pos_distance", pos_distance.mean())
        self.log("val/neg_distance", neg_distance.mean())

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration.
        """
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def encode_image(
        self,
        image: np.ndarray,
        device: Optional[torch.device] = None,
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
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        img_tensor = img_tensor / 255.0

        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Forward pass
        self.eval()
        embedding = self(img_tensor)

        return embedding.cpu().numpy()[0]  # type: ignore[no-any-return]

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
