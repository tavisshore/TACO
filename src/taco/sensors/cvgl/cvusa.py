import copy
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from taco.utils.image import panorama_horizontal_crop


class CVUSADataset(Dataset):
    """
    Unified CVUSA dataset for both training and evaluation.

    Loads heading metadata from split/all.csv and uses it to extract horizontal
    slices from street view panoramas at the correct vehicle direction.

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
        use_gnomonic_projection: Whether to crop panorama horizontally (name kept for compatibility)
        gnomonic_fov_deg: Horizontal field of view in degrees (determines crop width)
        gnomonic_output_shape: Output shape (H, W) for cropped images
        random_heading: If True, adds random offset to ground truth heading during training
        random_pitch: Not used (kept for compatibility)
        pitch_range: Not used (kept for compatibility)
        eval_heading_deg: Not used (heading from CSV is used instead)
        eval_pitch_deg: Not used (kept for compatibility)
        heading_csv_offset: Offset to add to image ID when indexing into all.csv (default: 0)
        rotate_reference_by_heading: If True, rotates satellite reference image by gt_heading_deg
    """

    def __init__(
        self,
        data_folder: Path = Path("/scratch/datasets/CVUSA/files"),
        stage="train",
        mode="triplet",
        transforms_query=None,
        transforms_reference=None,
        prob_flip=0.0,
        prob_rotate=0.0,
        shuffle_batch_size=128,
        use_gnomonic_projection=True,
        gnomonic_fov_deg=90.0,
        gnomonic_output_shape=(224, 224),
        random_heading=True,
        random_pitch=False,
        pitch_range=(-10.0, 10.0),
        eval_heading_deg=0.0,
        eval_pitch_deg=0.0,
        heading_csv_offset=0,
        rotate_reference_by_heading=True,
    ):
        super().__init__()

        assert stage in ["train", "val"], "stage must be 'train' or 'val'"
        assert mode in [
            "triplet",
            "query",
            "reference",
        ], "mode must be 'triplet', 'query', or 'reference'"

        self.data_folder = data_folder
        self.stage = stage
        self.mode = mode
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size

        # Gnomonic projection parameters
        self.use_gnomonic_projection = use_gnomonic_projection
        self.gnomonic_fov_deg = gnomonic_fov_deg
        self.gnomonic_output_shape = gnomonic_output_shape
        self.random_heading = random_heading
        self.random_pitch = random_pitch
        self.pitch_range = pitch_range
        self.eval_heading_deg = eval_heading_deg
        self.eval_pitch_deg = eval_pitch_deg

        self.transforms_query = transforms_query  # ground
        self.transforms_reference = transforms_reference  # satellite
        self.heading_csv_offset = heading_csv_offset
        self.rotate_reference_by_heading = rotate_reference_by_heading

        # Load heading metadata from all.csv
        # Image IDs in filenames (e.g., 0041073.jpg -> ID 41073) map to row (ID + offset) in all.csv
        all_metadata = pd.read_csv(f"{data_folder}/split/all.csv", header=None)
        all_metadata.columns = ["street_lat", "street_lon", "sat_lat", "sat_lon", "heading"]
        print(all_metadata.head(101))
        # Create mapping: image_id -> heading
        # The row index in all.csv = image_id + offset
        self.idx2heading = {
            i: all_metadata.loc[i + heading_csv_offset, "heading"]
            for i in range(len(all_metadata) - heading_csv_offset)
            if i + heading_csv_offset < len(all_metadata)
        }

        # Load appropriate split
        if stage == "train":
            self.df = pd.read_csv(f"{data_folder}/splits/train-19zl.csv", header=None)
        else:
            self.df = pd.read_csv(f"{data_folder}/splits/val-19zl.csv", header=None)

        self.df = self.df.rename(columns={0: "satellite", 1: "street", 2: "ground_anno"})
        self.df = self.df.drop(columns=["ground_anno"])

        self.df["idx"] = self.df.satellite.map(lambda x: int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.satellite, strict=False))
        self.idx2ground = dict(zip(self.df.idx, self.df.street, strict=False))

        # Setup based on mode
        if mode == "triplet":
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
        elif mode == "reference":
            # Reference/satellite images only
            self.images = self.df.satellite.values
            self.label = self.df.idx.values
        elif mode == "query":
            # Query/ground images only
            self.images = self.df.street.values
            self.label = self.df.idx.values

    def __getitem__(self, index):
        if self.mode == "triplet":
            return self._get_triplet(index)
        return self._get_single_image(index)

    def _get_triplet(self, index):
        """Get triplet for training: (query, reference, label)."""
        idx, sat, ground = self.idx2pair[self.samples[index]]

        # load query -> ground image (equirectangular panorama)
        query_img = cv2.imread(f"{self.data_folder}/{ground}")
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)

        # load reference -> satellite image
        reference_img = cv2.imread(f"{self.data_folder}/{sat}")
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        # Get the ground truth heading from the metadata
        gt_heading_deg = (
            self.idx2heading.get(idx - 1, 0.0) + 180
        ) % 360.0  # Adjust for image ID offset

        # Optionally rotate reference image by ground truth heading
        if self.rotate_reference_by_heading:
            reference_img = self._rotate_image_by_heading(reference_img, gt_heading_deg)

        # Apply horizontal crop from panorama
        if self.use_gnomonic_projection:
            # Sample random heading (vehicle direction) if enabled (training augmentation)
            if self.random_heading and self.stage == "train":
                # Add random offset to the ground truth heading for augmentation
                heading_offset = np.random.uniform(-180.0, 180.0)
                heading_deg = (gt_heading_deg + heading_offset) % 360.0
            else:
                # Use ground truth heading for evaluation
                heading_deg = gt_heading_deg

            query_img = panorama_horizontal_crop(
                query_img,
                heading_deg=heading_deg,
                fov_deg=self.gnomonic_fov_deg,
                output_shape=self.gnomonic_output_shape,
            )
        cv2.imwrite(f"output/{index}_r.jpg", cv2.cvtColor(reference_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"output/{index}_q.jpg", cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))

        # Flip simultaneously query and reference (only during training)
        if self.stage == "train" and np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)

        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)["image"]

        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)["image"]

        # Rotate simultaneously query and reference (only during training)
        if self.stage == "train" and np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])

            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))

            # use roll for ground view if rotate sat view
            # Note: If gnomonic projection is used, the query_img is already a perspective crop
            # so rolling might not make sense. Consider adjusting heading instead during projection.
            if not self.use_gnomonic_projection:
                w = query_img.shape[2]
                shifts = -w // 4 * r
                query_img = torch.roll(query_img, shifts=shifts, dims=2)

        label = torch.tensor(idx, dtype=torch.long)

        return query_img, reference_img, label

    def _get_single_image(self, index):
        """Get single image for evaluation: (image, label)."""
        img = cv2.imread(f"{self.data_folder}/{self.images[index]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the index/label for this image
        idx = self.label[index]

        # Apply horizontal crop for query (ground) images during evaluation
        if self.use_gnomonic_projection and self.mode == "query":
            # Get the ground truth heading from the metadata
            gt_heading_deg = self.idx2heading.get(idx, 0.0)

            img = panorama_horizontal_crop(
                img,
                heading_deg=gt_heading_deg,
                fov_deg=self.gnomonic_fov_deg,
                output_shape=self.gnomonic_output_shape,
            )

        # Optionally rotate reference (satellite) image by ground truth heading
        if self.rotate_reference_by_heading and self.mode == "reference":
            gt_heading_deg = self.idx2heading.get(idx, 0.0)
            img = self._rotate_image_by_heading(img, gt_heading_deg)

        # image transforms
        if self.mode == "query" and self.transforms_query is not None:
            img = self.transforms_query(image=img)["image"]
        elif self.mode == "reference" and self.transforms_reference is not None:
            img = self.transforms_reference(image=img)["image"]

        label = torch.tensor(idx, dtype=torch.long)

        return img, label

    def _rotate_image_by_heading(self, img, heading_deg):
        """
        Rotate image by heading angle (clockwise).

        Args:
            img: Input image (H, W, C) as numpy array
            heading_deg: Rotation angle in degrees (0-360, clockwise from North)

        Returns:
            Rotated image with same dimensions
        """
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        heading_deg += 180.0  # Adjust heading to match image rotation direction
        # Create rotation matrix (negative for clockwise rotation in image coordinates)
        rotation_matrix = cv2.getRotationMatrix2D(center, heading_deg, 1.0)

        # Rotate the image
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        return rotated_img

    def __len__(self):
        if self.mode == "triplet":
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
            if len(current_batch) >= self.shuffle_batch_size:
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
        if self.mode != "triplet":
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
                and len(current_batch) < self.shuffle_batch_size
            )

            if is_valid:
                idx_batch.add(idx)
                current_batch.append(idx)
                idx_epoch.add(idx)
                break_counter = 0

                # Add similar neighbors if similarity dictionary is provided
                has_space = len(current_batch) < self.shuffle_batch_size
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

            if len(current_batch) >= self.shuffle_batch_size:
                # empty current_batch bucket to batches
                batches.extend(current_batch)
                idx_batch = set()
                current_batch = []

        pbar.close()

        # wait before closing progress bar
        time.sleep(0.3)

        self.samples = batches
        print("idx_pool:", len(idx_pool))
        print(f"Original Length: {len(self.train_ids)} - Length after Shuffle: {len(self.samples)}")
        print("Break Counter:", break_counter)
        print(
            f"Pairs left out of last batch to avoid creating noise: {len(self.train_ids) - len(self.samples)}"
        )
        print(f"First Element ID: {self.samples[0]} - Last Element ID: {self.samples[-1]}")


# Backward compatibility aliases
CVUSADatasetTrain = CVUSADataset
CVUSADatasetEval = CVUSADataset


if __name__ == "__main__":
    # Example usage
    dataset = CVUSADataset(
        data_folder=Path("/scratch/datasets/CVUSA/files"),
        stage="train",
        mode="triplet",
        use_gnomonic_projection=True,
        random_heading=False,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Loaded heading data for {len(dataset.idx2heading)} images")

    # Fetch a few samples to verify heading usage
    print("\nTesting first 3 samples:")
    for i in range(3):
        query_img, reference_img, label = dataset.__getitem__(i)
        idx = label.item()
        heading = dataset.idx2heading.get(idx, None)

        print(f"\nSample {i}:")
        print(f"  Image ID: {idx}")
        print(f"  Ground truth heading: {heading:.2f}°" if heading else "  Heading: Not found")
        print(f"  Query shape: {query_img.shape}")
        print(f"  Reference shape: {reference_img.shape}")
