"""Tests for CVGL image retrieval model."""

import numpy as np
import pytest

# Skip tests if PyTorch not installed
pytorch_available = True
try:
    import torch

    from taco.sensors.cvgl import ImageRetrievalModel
except ImportError:
    pytorch_available = False


@pytest.mark.skipif(not pytorch_available, reason="PyTorch not installed")
class TestImageRetrievalModel:
    """Test ImageRetrievalModel class."""

    def test_model_creation(self) -> None:
        """Test creating the model."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)

        assert model.embedding_dim == 128
        assert model.learning_rate == 1e-4

    def test_forward_pass(self) -> None:
        """Test forward pass."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)
        model.eval()

        # Create dummy input
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)

        # Forward pass
        embeddings = model(images)

        # Check output shape
        assert embeddings.shape == (batch_size, 128)

        # Check L2 normalization
        norms = torch.norm(embeddings, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)

    def test_encode_image(self) -> None:
        """Test encoding a single image."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)
        model.eval()

        # Create dummy image
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Encode
        embedding = model.encode_image(image)

        # Check output
        assert embedding.shape == (128,)
        assert isinstance(embedding, np.ndarray)

        # Check normalization
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_compute_similarity(self) -> None:
        """Test computing similarity between embeddings."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)

        # Create dummy embeddings
        emb1 = np.random.randn(128)
        emb2 = np.random.randn(128)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 / np.linalg.norm(emb2)

        # Compute similarity
        similarity = model.compute_similarity(emb1, emb2)

        # Check output
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0

        # Identity should have similarity 1.0
        identity_sim = model.compute_similarity(emb1, emb1)
        assert np.isclose(identity_sim, 1.0, atol=1e-5)

    def test_retrieve_similar(self) -> None:
        """Test retrieving similar images."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)

        # Create query and database embeddings
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)

        database = np.random.randn(100, 128)
        database = database / np.linalg.norm(database, axis=1, keepdims=True)

        # Retrieve top-10
        indices, similarities = model.retrieve_similar(query, database, top_k=10)

        # Check outputs
        assert len(indices) == 10
        assert len(similarities) == 10
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)

        # Similarities should be sorted in descending order
        assert all(similarities[i] >= similarities[i + 1] for i in range(len(similarities) - 1))

    def test_triplet_loss(self) -> None:
        """Test triplet loss computation."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False, margin=0.2)

        batch_size = 8
        anchor = torch.randn(batch_size, 128)
        positive = torch.randn(batch_size, 128)
        negative = torch.randn(batch_size, 128)

        # Normalize
        anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
        positive = torch.nn.functional.normalize(positive, p=2, dim=1)
        negative = torch.nn.functional.normalize(negative, p=2, dim=1)

        # Compute loss
        loss = model.compute_triplet_loss(anchor, positive, negative)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0  # Loss should be non-negative

    def test_contrastive_loss(self) -> None:
        """Test contrastive loss computation."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)

        batch_size = 8
        embeddings = torch.randn(batch_size, 128)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Create labels (some same, some different)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

        # Compute loss
        loss = model.compute_contrastive_loss(embeddings, labels)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0  # Loss should be non-negative

    def test_configure_optimizers(self) -> None:
        """Test optimizer configuration."""
        model = ImageRetrievalModel(embedding_dim=128, pretrained=False)

        config = model.configure_optimizers()

        # Check optimizer
        assert "optimizer" in config
        assert isinstance(config["optimizer"], torch.optim.Optimizer)

        # Check scheduler
        assert "lr_scheduler" in config
        assert "scheduler" in config["lr_scheduler"]
