import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from taco.utils.image import panorama_horizontal_crop


@dataclass
class CVUSADatasetConfig:
    """Configuration for CVUSADataset.

    Args:
        data_folder: Path to CVUSA dataset root
        stage: 'train' or 'val' for data split
        mode: 'triplet' for training pairs, 'query' for ground images only,
              'reference' for satellite images only
        transforms_query: Albumentations transforms for ground images
        transforms_reference: Albumentations transforms for satellite images
        prob_flip: Probability of horizontal flip augmentation
        prob_rotate: Probability of rotation augmentation (90/180/270 degrees)
        shuffle_batch_size: Batch size for shuffle operation in triplet mode
        use_gnomonic_projection: Whether to crop panorama horizontally
        gnomonic_fov_deg: Horizontal field of view in degrees (determines crop width)
        gnomonic_output_shape: Output shape (H, W) for cropped images
        random_heading: If True, adds random offset to ground truth heading during training
        random_pitch: Not used (kept for compatibility)
        pitch_range: Not used (kept for compatibility)
        eval_heading_deg: Not used (heading from CSV is used instead)
        eval_pitch_deg: Not used (kept for compatibility)
        heading_csv_offset: Offset to add to image ID when indexing into all.csv
        rotate_reference_by_heading: If True, rotates satellite reference image by gt_heading_deg
        crop_rotated_reference: If True, crops rotated reference to remove black padding
        network_input_size: Output size (H, W) for network input after all processing
    """

    data_folder: Path = Path("/scratch/datasets/CVUSA/files")
    stage: str = "train"
    mode: str = "triplet"
    transforms_query: object | None = None
    transforms_reference: object | None = None
    prob_flip: float = 0.0
    prob_rotate: float = 0.0
    shuffle_batch_size: int = 128
    use_gnomonic_projection: bool = True
    gnomonic_fov_deg: float = 120.0
    gnomonic_output_shape: Tuple[int, int] = (224, 224)
    random_heading: bool = True
    random_pitch: bool = False
    pitch_range: Tuple[float, float] = (-10.0, 10.0)
    eval_heading_deg: float = 0.0
    eval_pitch_deg: float = 0.0
    heading_csv_offset: int = 0
    rotate_reference_by_heading: bool = True
    crop_rotated_reference: bool = True
    network_input_size: Tuple[int, int] = (224, 224)


class CVUSADataset(Dataset):
    """
    Unified CVUSA dataset for both training and evaluation.

    Loads heading metadata from split/all.csv and uses it to extract horizontal
    slices from street view panoramas at the correct vehicle direction.

    Args:
        config: CVUSADatasetConfig object containing all configuration parameters
    """

    def __init__(self, config: CVUSADatasetConfig):
        super().__init__()

        self.config = config

        assert config.stage in ["train", "val"], "stage must be 'train' or 'val'"
        assert config.mode in [
            "triplet",
            "query",
            "reference",
        ], "mode must be 'triplet', 'query', or 'reference'"
        # Load heading metadata from all.csv
        # Image IDs in filenames (e.g., 0041073.jpg -> ID 41073) map to row (ID + offset) in all.csv
        all_metadata = pd.read_csv(f"{config.data_folder}/split/all.csv", header=None)
        all_metadata.columns = ["street_lat", "street_lon", "sat_lat", "sat_lon", "heading"]
        # Create mapping: image_id -> heading
        # The row index in all.csv = image_id + offset
        self.idx2heading = {
            i: all_metadata.loc[i + config.heading_csv_offset, "heading"]
            for i in range(len(all_metadata) - config.heading_csv_offset)
            if i + config.heading_csv_offset < len(all_metadata)
        }

        # Load appropriate split
        if config.stage == "train":
            self.df = pd.read_csv(f"{config.data_folder}/splits/train-19zl.csv", header=None)
        else:
            self.df = pd.read_csv(f"{config.data_folder}/splits/val-19zl.csv", header=None)

        self.df = self.df.rename(columns={0: "satellite", 1: "street", 2: "ground_anno"})
        self.df = self.df.drop(columns=["ground_anno"])

        self.df["idx"] = self.df.satellite.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.satellite, strict=False))
        self.idx2ground = dict(zip(self.df.idx, self.df.street, strict=False))

        # Setup based on mode
        if config.mode == "triplet":
            # Training mode: return (query, reference, label) triplets
            self.pairs = list(zip(self.df.idx, self.df.satellite, self.df.street, strict=False))
            self.idx2pair = {}
            train_ids_list = []

            # for shuffle pool
            for pair in self.pairs:
                idx = pair[0]
                self.idx2pair[idx] = pair
                train_ids_list.append(idx)

            self.train_ids = train_ids_list
            self.samples = copy.deepcopy(self.train_ids)
        elif config.mode == "reference":
            # Reference/satellite images only
            self.images = self.df.satellite.values
            self.label = self.df.idx.values
        elif config.mode == "query":
            # Query/ground images only
            self.images = self.df.street.values
            self.label = self.df.idx.values

    def __getitem__(self, index):
        if self.config.mode == "triplet":
            return self._get_triplet(index)
        return self._get_single_image(index)

    def _get_triplet(self, index):
        """Get triplet for training: (query, reference, label)."""
        idx, sat, ground = self.idx2pair[self.samples[index]]

        # load query -> ground image (equirectangular panorama)
        query_img = cv2.imread(f"{self.config.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.config.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Get the ground truth heading from the metadata
        gt_heading_deg = (
            self.idx2heading.get(idx - 1, 0.0) + 180
        ) % 360.0  # Adjust for image ID offset

        # Optionally rotate reference image by ground truth heading
        if self.config.rotate_reference_by_heading:
            reference_img = self._rotate_image_by_heading(
                reference_img, gt_heading_deg, crop_to_fit=self.config.crop_rotated_reference
            )

        # Apply horizontal crop from panorama
        if self.config.use_gnomonic_projection:
            # Sample random heading (vehicle direction) if enabled (training augmentation)
            if self.config.random_heading and self.config.stage == "train":
                # Add random offset to the ground truth heading for augmentation
                heading_offset = np.random.uniform(-180.0, 180.0)
                heading_deg = (gt_heading_deg + heading_offset) % 360.0
            else:
                # Use ground truth heading for evaluation
                heading_deg = gt_heading_deg

            query_img = panorama_horizontal_crop(
                query_img,
                heading_deg=heading_deg,
                fov_deg=self.config.gnomonic_fov_deg,
                output_shape=self.config.gnomonic_output_shape,
            )
        # cv2.imwrite(f"output/{index}_r.jpg", cv2.cvtColor(reference_img, cv2.COLOR_RGB2BGR))
        # cv2.imwrite(f"output/{index}_q.jpg", cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))

        # Flip simultaneously query and reference (only during training)
        if self.config.stage == "train" and np.random.random() < self.config.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.config.transforms_query is not None:
            query_img = self.config.transforms_query(image=query_img)["image"]

        if self.config.transforms_reference is not None:
            reference_img = self.config.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference (only during training)
        if self.config.stage == "train" and np.random.random() < self.config.prob_rotate:
            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            # Note: If gnomonic projection is used, the query_img is already a perspective crop
            # so rolling might not make sense. Consider adjusting heading instead during projection.
            if not self.config.use_gnomonic_projection:
                w = query_img.shape[2]
                shifts = -w // 4 * r
                query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        # Convert to tensor and permute from (H, W, C) to (C, H, W)
        query_img = torch.tensor(query_img, dtype=torch.float32).permute(2, 0, 1)
        reference_img = torch.tensor(reference_img, dtype=torch.float32).permute(2, 0, 1)

        # Resize to network input size
        query_img = torch.nn.functional.interpolate(
            query_img.unsqueeze(0),
            size=self.config.network_input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        reference_img = torch.nn.functional.interpolate(
            reference_img.unsqueeze(0),
            size=self.config.network_input_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        return query_img, reference_img, label

    def _get_single_image(self, index):
        """Get single image for evaluation: (image, label)."""
        img = cv2.imread(f"{self.config.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the index/label for this image
        idx = self.label[index]

        # Apply horizontal crop for query (ground) images during evaluation
        if self.config.use_gnomonic_projection and self.config.mode == "query":
            # Get the ground truth heading from the metadata
            gt_heading_deg = self.idx2heading.get(idx, 0.0)

            img = panorama_horizontal_crop(
                img,
                heading_deg=gt_heading_deg,
                fov_deg=self.config.gnomonic_fov_deg,
                output_shape=self.config.gnomonic_output_shape,
            )

        # Optionally rotate reference (satellite) image by ground truth heading
        if self.config.rotate_reference_by_heading and self.config.mode == "reference":
            gt_heading_deg = self.idx2heading.get(idx, 0.0)
            img = self._rotate_image_by_heading(
                img, gt_heading_deg, crop_to_fit=self.config.crop_rotated_reference
            )

        # image transforms
        if self.config.mode == "query" and self.config.transforms_query is not None:
            img = self.config.transforms_query(image=img)["image"]
        elif self.config.mode == "reference" and self.config.transforms_reference is not None:
            img = self.config.transforms_reference(image=img)["image"]

        label = torch.tensor(idx, dtype=torch.long)

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

        return img, label

    def _rotate_image_by_heading(self, img, heading_deg, crop_to_fit=True):
        """
        Rotate image by heading angle (clockwise) with options to handle black padding.

        Args:
            img: Input image (H, W, C) as numpy array
            heading_deg: Rotation angle in degrees (0-360, clockwise from North)
            crop_to_fit: If True, crops to largest rectangle that fits without black padding

        Returns:
            Rotated image (cropped if crop_to_fit=True, otherwise same dimensions)
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        heading_deg += 180.0  # Adjust heading to match image rotation direction

        # Create rotation matrix (negative for clockwise rotation in image coordinates)
        rotation_matrix = cv2.getRotationMatrix2D(center, heading_deg, 1.0)

        # Rotate the image
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        if crop_to_fit:
            # Calculate the largest rectangle that fits inside the rotated image
            # without black padding. This uses the formula for inscribed rectangle.
            angle_rad = np.abs(np.radians(heading_deg % 90))

            if angle_rad != 0:
                # Calculate the dimensions of the largest axis-aligned rectangle
                # that fits inside the rotated image
                sin_a = np.sin(angle_rad)
                cos_a = np.cos(angle_rad)

                # Derived from geometry of rotated rectangle
                new_w = int((w * cos_a - h * sin_a) / (cos_a**2 - sin_a**2))
                new_h = int((h * cos_a - w * sin_a) / (cos_a**2 - sin_a**2))

                # Ensure dimensions are positive and within bounds
                new_w = max(1, min(new_w, w))
                new_h = max(1, min(new_h, h))

                # Crop to center
                x_start = (w - new_w) // 2
                y_start = (h - new_h) // 2
                rotated_img = rotated_img[y_start : y_start + new_h, x_start : x_start + new_w]

        return rotated_img

    def __len__(self):
        if self.config.mode == "triplet":
            return len(self.samples)
        return len(self.images)

    def _select_similar_neighbours(self, idx, similarity_pool, neighbour_split, neighbour_range):
        """Select similar neighbors for batch sampling."""
        near_similarity = similarity_pool[idx][:neighbour_range]
        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
        random.shuffle(far_neighbours)
        far_neighbours = far_neighbours[:neighbour_split]
        return near_neighbours + far_neighbours

    def _add_similar_neighbours_to_batch(
        self,
        idx,
        current_batch,
        idx_batch,
        idx_epoch,
        similarity_pool,
        neighbour_split,
        neighbour_range,
    ):
        """Add similar neighbors to the current batch."""
        near_similarity_select = self._select_similar_neighbours(
            idx, similarity_pool, neighbour_split, neighbour_range
        )

        for idx_near in near_similarity_select:
            if len(current_batch) >= self.config.shuffle_batch_size:
                break

            is_available = idx_near not in idx_batch and idx_near not in idx_epoch and idx_near
            if is_available:
                idx_batch.add(idx_near)
                current_batch.append(idx_near)
                idx_epoch.add(idx_near)
                similarity_pool[idx].remove(idx_near)

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        Custom shuffle function for unique class_id sampling in batch.
        Only applicable when mode='triplet'.
        """
        if self.config.mode != "triplet":
            raise ValueError("Shuffle is only supported in triplet mode")

        print("\nShuffle Dataset:")

        idx_pool = copy.deepcopy(self.train_ids)
        neighbour_split = neighbour_select // 2
        similarity_pool = copy.deepcopy(sim_dict) if sim_dict is not None else None

        random.shuffle(idx_pool)
        idx_epoch = set()
        idx_batch = set()
        batches = []
        current_batch = []
        break_counter = 0
        pbar = tqdm()

        while len(idx_pool) > 0 and break_counter < 1024:
            pbar.update()
            idx = idx_pool.pop(0)

            is_valid = (
                idx not in idx_batch
                and idx not in idx_epoch
                and len(current_batch) < self.config.shuffle_batch_size
            )

            if is_valid:
                idx_batch.add(idx)
                current_batch.append(idx)
                idx_epoch.add(idx)
                break_counter = 0

                # Add similar neighbors if similarity dictionary is provided
                has_space = len(current_batch) < self.config.shuffle_batch_size
                if similarity_pool is not None and has_space:
                    self._add_similar_neighbours_to_batch(
                        idx,
                        current_batch,
                        idx_batch,
                        idx_epoch,
                        similarity_pool,
                        neighbour_split,
                        neighbour_range,
                    )
            else:
                # if idx fits not in batch and is not already used in epoch -> back to pool
                if idx not in idx_batch and idx not in idx_epoch:
                    idx_pool.append(idx)
                break_counter += 1

            if len(current_batch) >= self.config.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        # print("idx_pool:", len(idx_pool))
        # print(f"Original Length: {len(self.train_ids)} - Length after Shuffle: {len(self.samples)}")
        # print("Break Counter:", break_counter)
        # print(
        #     f"Pairs left out of last batch to avoid creating noise: {len(self.train_ids) - len(self.samples)}"
        # )
        # print(f"First Element ID: {self.samples[0]} - Last Element ID: {self.samples[-1]}")


# Backward compatibility aliases
CVUSADatasetTrain = CVUSADataset
CVUSADatasetEval = CVUSADataset


if __name__ == "__main__":
    # Example usage
    config = CVUSADatasetConfig(
        data_folder=Path("/scratch/datasets/CVUSA/files"),
        stage="train",
        mode="triplet",
        use_gnomonic_projection=True,
        random_heading=False,
    )
    dataset = CVUSADataset(config)

    print(f"Dataset size: {len(dataset)}")

    config_val = CVUSADatasetConfig(
        data_folder=Path("/scratch/datasets/CVUSA/files"),
        stage="val",
        mode="triplet",
        use_gnomonic_projection=True,
        random_heading=False,
    )
    dataset_val = CVUSADataset(config_val)

    print(f"Dataset size: {len(dataset_val)}")

    # Iterate through a few samples
    for i in range(5):
        query_img, reference_img, label = dataset[i]
        print(
            f"Sample {i}: Query shape: {query_img.shape}, Reference shape: {reference_img.shape}, Label: {label}"
        )
