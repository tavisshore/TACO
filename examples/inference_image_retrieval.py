"""Example script for image retrieval inference."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from taco.sensors.cvgl import ImageRetrievalModel


def build_database(model, image_dir, device="cuda"):
    """Build embedding database from images.

    Args:
        model: Trained ImageRetrievalModel.
        image_dir: Directory containing database images.
        device: Device to run inference on.

    Returns:
        Tuple of (embeddings, image_paths).
    """
    image_paths = sorted(Path(image_dir).glob("*.jpg"))

    print(f"Building database from {len(image_paths)} images...")

    embeddings = []
    paths = []

    model.eval()
    with torch.no_grad():
        for img_path in image_paths:
            # Load image
            image = np.array(Image.open(img_path))

            # Encode
            embedding = model.encode_image(image, device=torch.device(device))

            embeddings.append(embedding)
            paths.append(str(img_path))

            if len(embeddings) % 100 == 0:
                print(f"  Processed {len(embeddings)}/{len(image_paths)}")

    embeddings = np.stack(embeddings)
    print(f"✓ Database built: {embeddings.shape}")

    return embeddings, paths


def query_image(model, query_path, database_embeddings, database_paths, top_k=10):
    """Query for similar images.

    Args:
        model: Trained ImageRetrievalModel.
        query_path: Path to query image.
        database_embeddings: Database embeddings (N, D).
        database_paths: Database image paths.
        top_k: Number of results to return.

    Returns:
        List of (path, similarity) tuples.
    """
    # Load query image
    query_image = np.array(Image.open(query_path))

    # Encode query
    query_embedding = model.encode_image(query_image)

    # Retrieve similar
    indices, similarities = model.retrieve_similar(
        query_embedding, database_embeddings, top_k=top_k
    )

    # Format results
    results = [(database_paths[idx], sim) for idx, sim in zip(indices, similarities)]

    return results


def main():
    """Run image retrieval inference example."""

    # Configuration
    checkpoint_path = "outputs/retrieval/checkpoints/best.ckpt"
    database_dir = "data/images/database"
    query_path = "data/images/query/test_001.jpg"
    top_k = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = ImageRetrievalModel.load_from_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded (device: {device})")

    # Build database
    database_embeddings, database_paths = build_database(
        model, database_dir, device=device
    )

    # Save database for later use
    database_file = Path("outputs/database.npz")
    np.savez(
        database_file,
        embeddings=database_embeddings,
        paths=database_paths,
    )
    print(f"✓ Database saved to {database_file}")

    # Query
    print(f"\nQuerying for: {query_path}")
    results = query_image(
        model, query_path, database_embeddings, database_paths, top_k=top_k
    )

    print(f"\nTop {top_k} matches:")
    for i, (path, similarity) in enumerate(results, 1):
        print(f"  {i}. {Path(path).name}: {similarity:.4f}")

    # Example: Load database from file
    print("\n--- Loading from saved database ---")
    data = np.load(database_file, allow_pickle=True)
    loaded_embeddings = data["embeddings"]
    loaded_paths = data["paths"]
    print(f"✓ Loaded database: {loaded_embeddings.shape}")

    # Query again
    results = query_image(model, query_path, loaded_embeddings, loaded_paths, top_k=5)
    print("\nTop 5 matches (from loaded database):")
    for i, (path, similarity) in enumerate(results, 1):
        print(f"  {i}. {Path(path).name}: {similarity:.4f}")


if __name__ == "__main__":
    main()
