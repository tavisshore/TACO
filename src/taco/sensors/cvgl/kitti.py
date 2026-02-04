from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class KITTIValDatasetConfig:
    data_folder: Path = Path("/scratch/datasets/kitti_val/")
    network_input_size: Tuple[int, int] = (384, 384)
    transforms_query: object | None = None
    transforms_reference: object | None = None


class KITTIValDataset(Dataset):
    def __init__(self, config: KITTIValDatasetConfig):
        super().__init__()
        self.config = config

        # Load image file names
        self.pairings = [
            *((p, "street") for p in sorted((self.config.data_folder / "street").glob("*.jpg"))),
            *((p, "sat") for p in sorted((self.config.data_folder / "sat").glob("*.jpg"))),
            *((p, "sat_rot") for p in sorted((self.config.data_folder / "sat_rot").glob("*.jpg"))),
        ]

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, index):
        path, img_type = self.pairings[index]
        img = Image.open(path).convert("RGB")

        # image transforms
        if self.config.transforms_query is not None and img_type == "street":
            img = self.config.transforms_query(img)
        if self.config.transforms_reference is not None and img_type in ["sat", "sat_rot"]:
            img = self.config.transforms_reference(img)
        else:
            # Convert to tensor and permute from (H, W, C) to (C, H, W) if numpy array
            if isinstance(img, np.ndarray):
                img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            # Resize to network input size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.config.network_input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return img, img_type


if __name__ == "__main__":
    # Example usage
    config = KITTIValDatasetConfig(
        data_folder=Path("/scratch/datasets/kitti_val/"),
    )
    dataset = KITTIValDataset(config)

    print(f"Dataset size: {len(dataset)}")

    # Iterate through a few samples
    for i in range(5):
        query_img, reference_img, reference_rotated_img, label = dataset[i]
        print(
            f"Sample {i}: Query shape: {query_img.shape}, Reference shape: {reference_img.shape}, Reference Rotated shape: {reference_rotated_img.shape}, Label: {label}"
        )
