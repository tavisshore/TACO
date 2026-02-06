import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from taco.utils.image import gnomonic_projection, panorama_horizontal_crop


@dataclass
class CVUSADatasetConfig:
    data_folder: Path = Path("/scratch/datasets/CVUSA/")
    dataset: str = "cvusa"
    stage: str = "train"
    mode: str = "triplet"  # Options: "triplet", "query", "reference"
    shuffle_batch_size: int = 128
    use_gnomonic_projection: bool = True
    gnomonic_fov_deg: float = 90.0
    gnomonic_output_shape: Tuple[int, int] = (384, 384)

    rotate_reference_by_heading: bool = True
    crop_rotated_reference: bool = True
    network_input_size: Tuple[int, int] = (384, 384)
    transforms_query: object | None = None
    transforms_reference: object | None = None

    random_heading: bool = False
    random_pitch: bool = False
    pitch_range: Tuple[float, float] = (-5.0, 5.0)

    # Albumentations augmentation pipelines
    # These should be A.Compose or ProgressiveAugmentation objects or None
    augmentations_train: A.Compose | None = None
    augmentations_val: A.Compose | None = None
    # For synchronized augmentations (query + reference together)
    augmentations_train_sync: A.Compose | None = None
    # Progressive augmentation settings
    use_progressive_augmentation: bool = True
    progressive_max_epochs: int = 100
    progressive_warmup_epochs: int = 10


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

        # Set up augmentations
        self.augmentations = self._setup_augmentations()
        self.augmentations_sync = self._setup_sync_augmentations()

        # Load heading metadata and splits
        self.idx2heading = self._load_heading_metadata()
        self.df = self._load_split()

        # Prepare dataframe mappings
        self.df = self.df.rename(columns={0: "satellite", 1: "street", 2: "ground_anno"})
        self.df = self.df.drop(columns=["ground_anno"])
        self.df["idx"] = self.df.satellite.map(lambda x: int(x.split("/")[-1].split(".")[0]))
        self.idx2sat = dict(zip(self.df.idx, self.df.satellite, strict=False))
        self.idx2ground = dict(zip(self.df.idx, self.df.street, strict=False))

        # Setup based on mode
        self._setup_mode_specific_data()

    def _setup_augmentations(self):
        """Set up the main augmentation pipeline based on config."""
        config = self.config
        if config.stage == "train" and config.augmentations_train is not None:
            if config.use_progressive_augmentation:
                return ProgressiveAugmentation(
                    config.augmentations_train,
                    max_epochs=config.progressive_max_epochs,
                    warmup_epochs=config.progressive_warmup_epochs,
                )
            return config.augmentations_train
        elif config.stage == "val" and config.augmentations_val is not None:
            return config.augmentations_val
        return None

    def _setup_sync_augmentations(self):
        """Set up synchronized augmentations for query+reference pairs."""
        config = self.config
        if config.stage == "train" and config.mode == "triplet":
            if config.augmentations_train_sync is not None and config.use_progressive_augmentation:
                return ProgressiveAugmentation(
                    config.augmentations_train_sync,
                    max_epochs=config.progressive_max_epochs,
                    warmup_epochs=config.progressive_warmup_epochs,
                )
            return config.augmentations_train_sync
        return None

    def _load_heading_metadata(self):
        """Load heading metadata from all.csv and create index-to-heading mapping."""
        all_metadata = pd.read_csv(f"{self.config.data_folder}/split/all.csv", header=None)
        all_metadata.columns = ["street_lat", "street_lon", "sat_lat", "sat_lon", "heading"]
        return {i: all_metadata.loc[i, "heading"] for i in range(len(all_metadata))}

    def _load_split(self):
        """Load the appropriate train or val split."""
        if self.config.stage == "train":
            return pd.read_csv(f"{self.config.data_folder}/splits/train-19zl.csv", header=None)
        return pd.read_csv(f"{self.config.data_folder}/splits/val-19zl.csv", header=None)

    def _setup_mode_specific_data(self):
        """Set up data structures based on dataset mode (triplet, query, or reference)."""
        if self.config.mode == "triplet":
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
        elif self.config.mode == "reference":
            # Reference/satellite images only
            self.images = self.df.satellite.values
            self.label = self.df.idx.values
        elif self.config.mode == "query":
            # Query/ground images only
            self.images = self.df.street.values
            self.label = self.df.idx.values

    def __getitem__(self, index):
        if self.config.mode == "triplet":
            return self._get_triplet(index)
        return self._get_single_image(index)

    def _get_triplet(self, index):
        """
        Get triplet for training: (query, reference, label).

        Processing order:
        1. Load query and reference images
        2. Rotate reference by heading (if enabled)
        3. Crop query with gnomonic projection (if enabled)
        4. Resize reference to match query size (for synchronized augmentations)
        5. Apply synchronized augmentations (flip, rotate) to both images
        6. Apply independent augmentations (color, blur, noise) to each image
        7. Apply timm transforms (resize, normalize, to tensor)
        """
        idx, sat, street = self.idx2pair[self.samples[index]]

        query_img = Image.open(f"{self.config.data_folder}/{street}").convert("RGB")
        reference_img = Image.open(f"{self.config.data_folder}/{sat}").convert("RGB")

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
            if self.config.random_heading and self.config.stage == "train":
                heading_offset = np.random.uniform(-180.0, 180.0)
                heading_deg = (gt_heading_deg + heading_offset) % 360.0
            else:
                heading_deg = gt_heading_deg

            # CVUSA doesn't have the full panos
            if self.config.dataset == "cvusa":
                query_img = panorama_horizontal_crop(
                    query_img,
                    heading_deg=float(heading_deg),
                    fov_deg=self.config.gnomonic_fov_deg,
                    output_shape=self.config.gnomonic_output_shape,
                )
            else:
                query_img = gnomonic_projection(
                    query_img,
                    heading_deg=float(heading_deg),
                    fov_deg=self.config.gnomonic_fov_deg,
                    output_shape=self.config.gnomonic_output_shape,
                )

        # Ensure both images are the same size before augmentation
        # Resize reference to match query size if needed
        if query_img.size != reference_img.size:
            reference_img = reference_img.resize(query_img.size, Image.BILINEAR)

        # Convert PIL images to numpy arrays for albumentations
        query_np = np.array(query_img)
        reference_np = np.array(reference_img)

        # Apply synchronized augmentations (query + reference together)
        # These are augmentations that need to be applied identically to both images
        if self.augmentations_sync is not None:
            # Use additional_targets to apply the same transform to both images
            augmented = self.augmentations_sync(image=query_np, reference=reference_np)
            query_np = augmented["image"]
            reference_np = augmented["reference"]

        # Apply independent augmentations to each image
        if self.augmentations is not None:
            query_np = self.augmentations(image=query_np)["image"]
            reference_np = self.augmentations(image=reference_np)["image"]

        # Convert back to PIL for the transform pipeline
        query_img = Image.fromarray(query_np)
        reference_img = Image.fromarray(reference_np)

        label = torch.tensor(idx, dtype=torch.long)

        # image transforms (timm pipeline handles PIL -> tensor + normalize)
        if self.config.transforms_query is not None:
            query_img = self.config.transforms_query(query_img)
        else:
            query_img = (
                torch.tensor(np.array(query_img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            )
            query_img = torch.nn.functional.interpolate(
                query_img.unsqueeze(0),
                size=self.config.network_input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.config.transforms_reference is not None:
            reference_img = self.config.transforms_reference(reference_img)
        else:
            reference_img = (
                torch.tensor(np.array(reference_img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

            reference_img = torch.nn.functional.interpolate(
                reference_img.unsqueeze(0),
                size=self.config.network_input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return query_img, reference_img, label

    def _get_single_image(self, index):
        """Get single image for evaluation: (image, label)."""
        img = Image.open(f"{self.config.data_folder}/{self.images[index]}").convert("RGB")

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

        # Apply albumentations augmentations
        if self.augmentations is not None:
            img_np = np.array(img)
            img_np = self.augmentations(image=img_np)["image"]
            img = Image.fromarray(img_np)

        # image transforms (timm pipeline handles PIL -> tensor + normalize)
        if self.config.mode == "query" and self.config.transforms_query is not None:
            img = self.config.transforms_query(img)
        elif self.config.mode == "reference" and self.config.transforms_reference is not None:
            img = self.config.transforms_reference(img)
        else:
            img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            # Resize to network input size
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.config.network_input_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        label = torch.tensor(idx, dtype=torch.long)

        return img, label

    def _rotate_image_by_heading(self, img, heading_deg, crop_to_fit=True):
        """
        Rotate image by heading angle (clockwise) with options to handle black padding.

        Args:
            img: Input PIL Image
            heading_deg: Rotation angle in degrees (0-360, clockwise from North)
            crop_to_fit: If True, crops to largest rectangle that fits without black padding

        Returns:
            Rotated PIL Image (cropped if crop_to_fit=True, otherwise same dimensions)
        """
        w, h = img.size
        heading_deg += 180.0  # Adjust heading to match image rotation direction

        # PIL rotate is counter-clockwise, same sign convention as cv2.getRotationMatrix2D
        rotated_img = img.rotate(heading_deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))

        if crop_to_fit:
            # Calculate the largest rectangle that fits inside the rotated image
            # without black padding. This uses the formula for inscribed rectangle.
            angle_rad = np.abs(np.radians(heading_deg % 90))

            if angle_rad != 0:
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
                rotated_img = rotated_img.crop((x_start, y_start, x_start + new_w, y_start + new_h))

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


def create_default_augmentations(
    stage: str = "train",
    strength: str = "progressive",  # "weak", "medium", "strong", or "progressive"
    # Color augmentations
    prob_color_jitter: float | None = None,
    brightness_limit: float | None = None,
    contrast_limit: float | None = None,
    saturation_limit: float | None = None,
    hue_limit: float | None = None,
    # Blur/noise augmentations
    prob_blur: float | None = None,
    prob_gaussian_noise: float | None = None,
    # Other augmentations
    prob_sharpen: float | None = None,
    prob_clahe: float | None = None,
) -> A.Compose:
    """
    Create a default albumentations augmentation pipeline with progressive strength.

    Args:
        stage: Either "train" or "val"
        strength: Augmentation strength preset: "weak", "medium", "strong", or "progressive"
                 - "weak": Light augmentations for early training
                 - "medium": Moderate augmentations for mid training
                 - "strong": Heavy augmentations for late training / regularization
                 - "progressive": Automatically increases strength (default, recommended)
        prob_color_jitter: Probability of applying color jitter (overrides strength preset)
        brightness_limit: Brightness variation range (overrides strength preset)
        contrast_limit: Contrast variation range (overrides strength preset)
        saturation_limit: Saturation variation range (overrides strength preset)
        hue_limit: Hue variation range (overrides strength preset)
        prob_blur: Probability of applying blur (overrides strength preset)
        prob_gaussian_noise: Probability of adding Gaussian noise (overrides strength preset)
        prob_sharpen: Probability of applying sharpening (overrides strength preset)
        prob_clahe: Probability of applying CLAHE (overrides strength preset)

    Returns:
        A.Compose object with the augmentation pipeline

    Note:
        When using strength="progressive", the augmentation strength will automatically
        increase during training. Use ProgressiveAugmentationCallback to enable this.
    """
    if stage != "train":
        # No augmentations for validation
        return A.Compose([])

    # Define strength presets
    strength_presets = {
        "weak": {
            "prob_color_jitter": 0.3,
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
            "saturation_limit": 0.1,
            "hue_limit": 0.05,
            "prob_blur": 0.1,
            "prob_gaussian_noise": 0.1,
            "prob_sharpen": 0.1,
            "prob_clahe": 0.1,
        },
        "medium": {
            "prob_color_jitter": 0.5,
            "brightness_limit": 0.2,
            "contrast_limit": 0.2,
            "saturation_limit": 0.2,
            "hue_limit": 0.1,
            "prob_blur": 0.3,
            "prob_gaussian_noise": 0.3,
            "prob_sharpen": 0.2,
            "prob_clahe": 0.2,
        },
        "strong": {
            "prob_color_jitter": 0.7,
            "brightness_limit": 0.3,
            "contrast_limit": 0.3,
            "saturation_limit": 0.3,
            "hue_limit": 0.15,
            "prob_blur": 0.5,
            "prob_gaussian_noise": 0.5,
            "prob_sharpen": 0.3,
            "prob_clahe": 0.3,
        },
        "progressive": {
            # Start with weak, will be updated progressively
            "prob_color_jitter": 0.3,
            "brightness_limit": 0.1,
            "contrast_limit": 0.1,
            "saturation_limit": 0.1,
            "hue_limit": 0.05,
            "prob_blur": 0.1,
            "prob_gaussian_noise": 0.1,
            "prob_sharpen": 0.1,
            "prob_clahe": 0.1,
        },
    }

    # Get preset values
    if strength not in strength_presets:
        raise ValueError(
            f"Invalid strength '{strength}'. Must be one of: {list(strength_presets.keys())}"
        )

    preset = strength_presets[strength]

    # Override with explicit parameters if provided
    params = {
        "prob_color_jitter": prob_color_jitter or preset["prob_color_jitter"],
        "brightness_limit": brightness_limit or preset["brightness_limit"],
        "contrast_limit": contrast_limit or preset["contrast_limit"],
        "saturation_limit": saturation_limit or preset["saturation_limit"],
        "hue_limit": hue_limit or preset["hue_limit"],
        "prob_blur": prob_blur or preset["prob_blur"],
        "prob_gaussian_noise": prob_gaussian_noise or preset["prob_gaussian_noise"],
        "prob_sharpen": prob_sharpen or preset["prob_sharpen"],
        "prob_clahe": prob_clahe or preset["prob_clahe"],
    }

    transforms = [
        # Color augmentations
        A.ColorJitter(
            brightness=params["brightness_limit"],
            contrast=params["contrast_limit"],
            saturation=params["saturation_limit"],
            hue=params["hue_limit"],
            p=params["prob_color_jitter"],
        ),
        # Blur and noise
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
            ],
            p=params["prob_blur"],
        ),
        A.GaussNoise(std_range=(0.04, 0.2), p=params["prob_gaussian_noise"]),
        # Sharpening and contrast
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=params["prob_sharpen"]),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=params["prob_clahe"]),
    ]

    return A.Compose(transforms)


class ProgressiveAugmentation:
    """
    Wrapper for albumentations that progressively increases augmentation strength.

    This class wraps an A.Compose pipeline and allows updating augmentation
    probabilities during training based on the current epoch or progress.

    Example:
        >>> base_aug = create_default_augmentations(stage="train", strength="progressive")
        >>> prog_aug = ProgressiveAugmentation(base_aug, max_epochs=100)
        >>> # During training loop:
        >>> prog_aug.update_strength(current_epoch=10)
    """

    def __init__(
        self,
        base_augmentation: A.Compose,
        max_epochs: int = 100,
        start_strength: float = 0.3,
        end_strength: float = 1.0,
        warmup_epochs: int = 10,
    ):
        """
        Initialize progressive augmentation.

        Args:
            base_augmentation: Base albumentations pipeline (should use "progressive" strength)
            max_epochs: Total number of training epochs
            start_strength: Starting strength multiplier (0.0-1.0)
            end_strength: Final strength multiplier (0.0-1.0)
            warmup_epochs: Number of epochs to reach end_strength
        """
        self.augmentation = base_augmentation
        self.max_epochs = max_epochs
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.warmup_epochs = warmup_epochs
        self.current_strength = start_strength

        # Store original probabilities
        self._original_probs = {}
        for i, transform in enumerate(self.augmentation.transforms):
            if hasattr(transform, "p"):
                self._original_probs[i] = transform.p

    def update_strength(self, current_epoch: int) -> float:
        """
        Update augmentation strength based on current epoch.

        Args:
            current_epoch: Current training epoch (0-indexed)

        Returns:
            Current strength multiplier
        """
        if current_epoch >= self.warmup_epochs:
            self.current_strength = self.end_strength
        else:
            # Linear warmup
            progress = current_epoch / self.warmup_epochs
            self.current_strength = (
                self.start_strength + (self.end_strength - self.start_strength) * progress
            )

        # Update transform probabilities
        for i, transform in enumerate(self.augmentation.transforms):
            if i in self._original_probs:
                transform.p = self._original_probs[i] * self.current_strength

        return self.current_strength

    def __call__(self, **kwargs):
        """Apply augmentation."""
        return self.augmentation(**kwargs)

    @property
    def transforms(self):
        """Access to transforms for compatibility."""
        return self.augmentation.transforms


def create_synchronized_augmentations(
    prob_flip: float = 0.5,
    prob_rotate: float = 0.5,
    rotate_limit: int = 20,
    use_90deg_rotations: bool = False,
) -> A.Compose:
    """
    Create synchronized augmentations that should be applied identically to both
    query and reference images in a pair.

    Args:
        prob_flip: Probability of horizontal flip
        prob_rotate: Probability of rotation
        rotate_limit: Maximum rotation angle in degrees (both directions).
                     Rotation angle is sampled uniformly from [-rotate_limit, rotate_limit].
        use_90deg_rotations: If True, use only 90-degree rotations instead of arbitrary angles

    Returns:
        A.Compose object configured for synchronized augmentations
        Use with additional_targets={'reference': 'image'}
    """
    transforms = [
        A.HorizontalFlip(p=prob_flip),
    ]

    if use_90deg_rotations:
        transforms.append(A.RandomRotate90(p=prob_rotate))
    else:
        # Rotate with arbitrary angles, sampled uniformly from [-rotate_limit, rotate_limit]
        # Border mode 'reflect' or 'constant' can be used to handle boundaries
        transforms.append(
            A.Rotate(
                limit=rotate_limit,
                p=prob_rotate,
                border_mode=0,  # cv2.BORDER_CONSTANT (black padding)
                interpolation=1,  # cv2.INTER_LINEAR
            )
        )

    return A.Compose(transforms, additional_targets={"reference": "image"})


# Backward compatibility aliases
CVUSADatasetTrain = CVUSADataset
CVUSADatasetEval = CVUSADataset


if __name__ == "__main__":
    # Example usage with progressive albumentations

    # Create augmentation pipelines with progressive strength (recommended)
    train_augs = create_default_augmentations(
        stage="train",
        strength="progressive",  # Starts weak, increases during training
    )

    # Create synchronized augmentations for query+reference pairs
    sync_augs = create_synchronized_augmentations(
        prob_flip=0.5,
        prob_rotate=0.5,
        rotate_limit=10,  # Rotate up to ±10 degrees
        use_90deg_rotations=False,  # Use arbitrary angles instead of 90-degree rotations
    )

    config = CVUSADatasetConfig(
        data_folder=Path("/scratch/datasets/CVUSA"),
        stage="train",
        mode="triplet",
        use_gnomonic_projection=True,
        random_heading=False,
        augmentations_train=train_augs,
        augmentations_train_sync=sync_augs,
        use_progressive_augmentation=True,  # Enable progressive augmentations
        progressive_max_epochs=100,
        progressive_warmup_epochs=10,
    )
    dataset = CVUSADataset(config)

    print(f"Dataset size: {len(dataset)}")

    # Validation dataset (no augmentations)
    val_augs = create_default_augmentations(stage="val")

    config_val = CVUSADatasetConfig(
        data_folder=Path("/scratch/datasets/CVUSA"),
        stage="val",
        mode="triplet",
        use_gnomonic_projection=True,
        random_heading=False,
        augmentations_val=val_augs,
    )
    dataset_val = CVUSADataset(config_val)

    print(f"Dataset size: {len(dataset_val)}")

    # Iterate through a few samples
    for i in range(5):
        query_img, reference_img, label = dataset[i]
        print(
            f"Sample {i}: Query shape: {query_img.shape}, Reference shape: {reference_img.shape}, Label: {label}"
        )
