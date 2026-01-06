"""Dataset classes for image retrieval training."""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class TripletDataset(Dataset):
    """Dataset for triplet-based image retrieval training.

    Each sample returns an anchor, positive, and negative image triplet
    for metric learning.
    """

    def __init__(
        self,
        image_dir: Path,
        triplets_file: Path,
        transform: Callable | None = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize triplet dataset.

        Args:
            image_dir: Directory containing images.
            triplets_file: File containing triplet annotations.
                Format: anchor_path,positive_path,negative_path,label
            transform: Optional image transformations.
            image_size: Target image size (height, width).
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Load triplets
        self.triplets = self._load_triplets(triplets_file)

        # Default transforms if none provided
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _load_triplets(self, triplets_file: Path) -> list[dict]:
        """Load triplet annotations from file.

        Args:
            triplets_file: Path to triplets file.

        Returns:
            List of triplet dictionaries.
        """
        triplets = []
        with open(triplets_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    triplets.append(
                        {
                            "anchor": parts[0],
                            "positive": parts[1],
                            "negative": parts[2],
                            "label": int(parts[3]),
                        }
                    )
        return triplets

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations.

        Returns:
            Composed transforms.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path.

        Args:
            image_path: Relative path to image.

        Returns:
            PIL Image.
        """
        full_path = self.image_dir / image_path
        image = Image.open(full_path).convert("RGB")
        return image

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of triplets.
        """
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a triplet sample.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (anchor, positive, negative, label).
        """
        triplet = self.triplets[idx]

        # Load images
        anchor_img = self._load_image(triplet["anchor"])
        positive_img = self._load_image(triplet["positive"])
        negative_img = self._load_image(triplet["negative"])

        # Apply transforms
        anchor = self.transform(anchor_img)
        positive = self.transform(positive_img)
        negative = self.transform(negative_img)

        label = triplet["label"]

        return anchor, positive, negative, label


class ImageDatabaseDataset(Dataset):
    """Dataset for image database (for inference/retrieval).

    Each sample returns a single image with its metadata.
    """

    def __init__(
        self,
        image_dir: Path,
        images_file: Path,
        transform: Callable | None = None,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize image database dataset.

        Args:
            image_dir: Directory containing images.
            images_file: File containing image paths and metadata.
                Format: image_path,latitude,longitude,heading
            transform: Optional image transformations.
            image_size: Target image size (height, width).
        """
        self.image_dir = Path(image_dir)
        self.image_size = image_size

        # Load image metadata
        self.images = self._load_images(images_file)

        # Default transforms if none provided
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform

    def _load_images(self, images_file: Path) -> list[dict]:
        """Load image metadata from file.

        Args:
            images_file: Path to images file.

        Returns:
            List of image metadata dictionaries.
        """
        images = []
        with open(images_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    images.append(
                        {
                            "path": parts[0],
                            "latitude": float(parts[1]),
                            "longitude": float(parts[2]),
                            "heading": float(parts[3]),
                        }
                    )
        return images

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations.

        Returns:
            Composed transforms.
        """
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_image(self, image_path: str) -> Image.Image:
        """Load image from path.

        Args:
            image_path: Relative path to image.

        Returns:
            PIL Image.
        """
        full_path = self.image_dir / image_path
        image = Image.open(full_path).convert("RGB")
        return image

    def __len__(self) -> int:
        """Get dataset size.

        Returns:
            Number of images.
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """Get an image sample.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple of (image_tensor, metadata).
        """
        image_data = self.images[idx]

        # Load image
        image = self._load_image(image_data["path"])

        # Apply transforms
        image_tensor = self.transform(image)

        # Return with metadata
        metadata = {
            "path": image_data["path"],
            "latitude": image_data["latitude"],
            "longitude": image_data["longitude"],
            "heading": image_data["heading"],
            "index": idx,
        }

        return image_tensor, metadata


def create_triplet_mining_dataset(
    image_dir: Path,
    annotations_file: Path,
    transform: Callable | None = None,
    num_negatives: int = 5,
) -> TripletDataset:
    """Create triplet dataset with online hard negative mining.

    Args:
        image_dir: Directory containing images.
        annotations_file: File containing image annotations.
        transform: Optional image transformations.
        num_negatives: Number of hard negatives to mine per anchor.

    Returns:
        TripletDataset instance.
    """
    # This is a placeholder for more sophisticated triplet mining
    # In practice, you would:
    # 1. Load all images and their GPS coordinates
    # 2. For each anchor, find positives (same place) and negatives (different places)
    # 3. Optionally use hard negative mining based on similarity

    # For now, just return a basic dataset
    # The triplets file should be pre-generated
    triplets_file = annotations_file.parent / "triplets.txt"

    return TripletDataset(
        image_dir=image_dir,
        triplets_file=triplets_file,
        transform=transform,
    )
