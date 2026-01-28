"""PyTorch Lightning model for image retrieval using ConvNeXt."""

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import gtsam
import lightning as L
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
        pretrained: Whether to use pretrained ConvNeXt weights
        learning_rate: Learning rate for optimizer
        temperature: Temperature for contrastive loss
        margin: Margin for triplet loss
        freeze_backbone: If True, freeze ConvNeXt backbone weights
        loss_type: Type of loss function to use ('combined', 'ntxent', 'triplet')
        scheduler_type: Type of learning rate scheduler ('cosine', 'step', 'plateau', 'none')
        scheduler_t_max: T_max for cosine annealing (number of epochs for full cycle)
        scheduler_eta_min: Minimum learning rate for cosine annealing
        scheduler_step_size: Step size for StepLR scheduler
        scheduler_gamma: Multiplicative factor for StepLR and ReduceLROnPlateau
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
    """

    embedding_dim: int = 512
    pretrained: bool = True
    learning_rate: float = 1e-4
    temperature: float = 0.07
    margin: float = 0.2
    freeze_backbone: bool = False
    loss_type: Literal["combined", "ntxent", "triplet"] = "ntxent"
    scheduler_type: Literal["cosine", "step", "plateau", "none"] = "cosine"
    scheduler_t_max: int = 100
    scheduler_eta_min: float = 1e-6
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10


class ImageRetrievalModel(L.LightningModule):
    """PyTorch Lightning model for image retrieval using ConvNeXt-Tiny.

    Uses a pre-trained ConvNeXt-Tiny backbone to encode images into
    low-dimensional descriptors for visual place recognition.

    Args:
        config: ImageRetrievalModelConfig object containing all configuration parameters
    """

    def __init__(self, config: ImageRetrievalModelConfig) -> None:
        """Initialize the image retrieval model.

        Args:
            config: ImageRetrievalModelConfig object containing all configuration parameters
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # Load pre-trained ConvNeXt-Tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if config.pretrained else None
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
        if config.freeze_backbone:
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
            nn.Linear(1024, config.embedding_dim),
        )

        # L2 normalization layer
        self.l2_norm = nn.functional.normalize

        # Initialize NT-Xent loss from pytorch_metric_learning
        self.ntxent_loss = losses.NTXentLoss(temperature=config.temperature)

        # Reference database storage
        self.reference_embeddings: np.ndarray | None = None
        self.reference_coordinates: np.ndarray | None = None  # GPS (lat, lon)
        self.reference_origin: tuple[float, float] | None = (
            None  # (lat, lon) origin for local frame
        )
        self.use_utm: bool = False  # Whether to use UTM coordinates
        self.utm_zone: int | None = None  # UTM zone number
        self.utm_letter: str | None = None  # UTM zone letter

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
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.config.temperature

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

    def compute_ntxent_loss(
        self,
        query_emb: Tensor,
        reference_emb: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Compute NT-Xent loss using pytorch_metric_learning.

        NT-Xent (Normalized Temperature-scaled Cross Entropy Loss) is a contrastive
        loss that treats each pair of query-reference images as positives and all
        other images in the batch as negatives.

        Args:
            query_emb: Query image embeddings (B, D).
            reference_emb: Reference image embeddings (B, D).
            labels: Place labels (B,).

        Returns:
            NT-Xent loss value.
        """
        # Concatenate query and reference embeddings
        # This creates pairs where query[i] and reference[i] have the same label
        embeddings = torch.cat([query_emb, reference_emb], dim=0)

        # Duplicate labels for both query and reference
        # labels[i] corresponds to query_emb[i] and reference_emb[i]
        labels_combined = torch.cat([labels, labels], dim=0)

        # Compute NT-Xent loss using pytorch_metric_learning
        loss = self.ntxent_loss(embeddings, labels_combined)

        return loss

    def training_step(
        self,
        batch: Tuple[Tensor, Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """Training step.

        Args:
            batch: Tuple of (query_img, reference_img, labels) from CVUSADataset.
            batch_idx: Batch index.

        Returns:
            Loss value.
        """
        query_img, reference_img, labels = batch

        # Compute embeddings
        query_emb = self(query_img)  # anchor embeddings (ground view)
        reference_emb = self(reference_img)  # positive embeddings (satellite view)

        # Compute loss based on loss_type configuration
        if self.config.loss_type == "ntxent":
            # Use NT-Xent loss only
            loss = self.compute_ntxent_loss(query_emb, reference_emb, labels)
            self.log("train/ntxent_loss", loss)
            self.log("train/loss", loss, prog_bar=True)

        elif self.config.loss_type == "triplet":
            # Use triplet loss only
            negative_emb = torch.roll(reference_emb, shifts=1, dims=0)
            loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)
            self.log("train/triplet_loss", loss)
            self.log("train/loss", loss, prog_bar=True)

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
            self.log("train/triplet_loss", triplet_loss)
            self.log("train/contrastive_loss", contrastive_loss)
            self.log("train/loss", loss, prog_bar=True)

        return loss

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
        query_emb = self(query_img)  # anchor embeddings (ground view)
        reference_emb = self(reference_img)  # positive embeddings (satellite view)

        # Sample negatives: shift reference embeddings to create mismatched pairs
        negative_emb = torch.roll(reference_emb, shifts=1, dims=0)

        # Compute loss based on loss_type configuration
        if self.config.loss_type == "ntxent":
            # Use NT-Xent loss only
            loss = self.compute_ntxent_loss(query_emb, reference_emb, labels)
            self.log("val/ntxent_loss", loss)
            self.log("val/loss", loss, prog_bar=True)

        elif self.config.loss_type == "triplet":
            # Use triplet loss only
            loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)
            self.log("val/triplet_loss", loss)
            self.log("val/loss", loss, prog_bar=True)

        else:  # self.config.loss_type == "combined"
            # Compute triplet loss
            triplet_loss = self.compute_triplet_loss(query_emb, reference_emb, negative_emb)

            # Compute contrastive loss
            contrastive_loss = self.compute_contrastive_loss(query_emb, labels)

            # Combined loss
            loss = triplet_loss + 0.5 * contrastive_loss

            # Log metrics
            self.log("val/triplet_loss", triplet_loss)
            self.log("val/contrastive_loss", contrastive_loss)
            self.log("val/loss", loss, prog_bar=True)

        # Compute metrics (independent of loss type)
        pos_distance = F.pairwise_distance(query_emb, reference_emb, p=2)
        neg_distance = F.pairwise_distance(query_emb, negative_emb, p=2)
        accuracy = (pos_distance < neg_distance).float().mean()

        # Compute Top-K recall metrics
        recall_at_k = self.compute_recall_at_k(query_emb, reference_emb, k_values=(1, 5, 10))

        # Log metrics
        self.log("val/accuracy", accuracy)
        self.log("val/pos_distance", pos_distance.mean())
        self.log("val/neg_distance", neg_distance.mean())

        # Log recall@K metrics as percentages in progress bar
        self.log("val@1", recall_at_k[1] * 100, prog_bar=True)
        self.log("val@5", recall_at_k[5] * 100, prog_bar=True)
        self.log("val@10", recall_at_k[10] * 100, prog_bar=True)

        return loss

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
                    "utm package is required for UTM conversion. "
                    "Install it with: pip install utm"
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

            # Compute embeddings
            batch_embeddings = self(batch_tensor)
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

        if self.reference_origin is None:
            raise RuntimeError("Reference origin not set. This should not happen.")

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

        # Create CVGLMeasurement (position-only)
        measurement = CVGLMeasurement(
            timestamp=timestamp,
            position=position,
            position_covariance=position_covariance,
            confidence=float(np.clip(top_similarities[0], 0.0, 1.0)),
            num_inliers=top_k,  # Use top_k as a proxy for inliers
            yaw=current_yaw,  # Store current yaw from IMU (not estimated)
            image_id=None,
        )

        return measurement


if __name__ == "__main__":
    # Example usage
    import numpy as np

    # Create configuration with NT-Xent loss
    config = ImageRetrievalModelConfig(
        embedding_dim=512,
        pretrained=True,
        learning_rate=1e-4,
        temperature=0.07,
        margin=0.2,
        freeze_backbone=False,
        loss_type="ntxent",  # Options: "combined", "ntxent", "triplet"
    )

    # Create model with config
    model = ImageRetrievalModel(config)
    print(f"Model created with embedding_dim={config.embedding_dim}, loss_type={config.loss_type}")

    # Test forward pass
    dummy_image = torch.randn(2, 3, 224, 224)
    embeddings = model(dummy_image)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test encoding a single image
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    embedding = model.encode_image(test_image)
    print(f"Single image embedding shape: {embedding.shape}")
