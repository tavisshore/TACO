"""Downloads city graphs, then CVGL pairs for each junction, and creates a dataset for training."""

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import albumentations as A
import lightning.pytorch as pl
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streetview
import torch
from haversine import Unit, haversine
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from taco.data.graph_refine import simplify_sharp_turns
from taco.data.sat import download_satmap
from taco.utils.image import gnomonic_projection


def extract_bearing_from_geometry(graph, node, edge_data) -> float | None:
    if not edge_data or "geometry" not in edge_data:
        return None
    geom = edge_data["geometry"]

    if not hasattr(geom, "coords"):
        return None
    coords = list(geom.coords)
    if len(coords) < 2:
        return None

    # Get node position
    node_lat, node_lon = graph.nodes[node]["y"], graph.nodes[node]["x"]

    # Find which end of the geometry is closest to our node
    first_point = coords[0]
    last_point = coords[-1]
    dist_to_first = (node_lon - first_point[0]) ** 2 + (node_lat - first_point[1]) ** 2
    dist_to_last = (node_lon - last_point[0]) ** 2 + (node_lat - last_point[1]) ** 2

    # If node is closer to the end, reverse coords so node is at the start
    if dist_to_last < dist_to_first:
        coords = coords[::-1]

    if len(coords) > 2:
        # Find the first segment that is at least 3 meters long # Optimise
        accumulated_length = 0.0
        for i in range(1, len(coords)):
            pt1 = coords[i - 1]
            pt2 = coords[i]
            segment_length = haversine((pt1[1], pt1[0]), (pt2[1], pt2[0]), unit=Unit.METERS)
            accumulated_length += segment_length
            if accumulated_length >= 3.0:
                next_point_lat, next_point_lon = pt2[1], pt2[0]
                return calculate_bearing(node_lat, node_lon, next_point_lat, next_point_lon)

    # Now node is always at coords[0], use bearing from node to coords[1]
    next_point_lat, next_point_lon = coords[1][1], coords[1][0]
    return calculate_bearing(node_lat, node_lon, next_point_lat, next_point_lon)


def calculate_bearing(lat1, lon1, lat2, lon2):
    phi_1, lambda_1, phi_2, lambda_2 = map(math.radians, [lat1, lon1, lat2, lon2])
    lambda_d = lambda_2 - lambda_1
    x = math.sin(lambda_d) * math.cos(phi_2)
    y = math.cos(phi_1) * math.sin(phi_2) - math.sin(phi_1) * math.cos(phi_2) * math.cos(lambda_d)
    theta = math.atan2(x, y)
    bearing = (math.degrees(theta) + 360) % 360  # Normalize to [0, 360)
    bearing = np.deg2rad(bearing)  # Convert to radians
    return bearing


def _filter_nearby_panoramas(sv_images, node_coord, max_distance=30):
    """Filter panoramas by distance and return sorted by proximity."""
    panoramas_with_dist = []
    for pan in sv_images:
        if pan.heading is not None:  # Absolutely necessary
            dist = haversine((pan.lat, pan.lon), node_coord, unit=Unit.METERS)
            if dist < max_distance:
                panoramas_with_dist.append((dist, pan))

    # Sort by distance and return sorted indices and panoramas
    panoramas_with_dist.sort(key=lambda x: x[0])
    return [pan for _, pan in panoramas_with_dist[:10]]


def _download_single_panorama(pano, pano_path, node_coord):
    """Download a single panorama if it doesn't exist. Returns pano_id on success, None on failure."""
    if Path(pano_path / f"{pano.pano_id}.jpg").exists():
        return pano.pano_id

    try:
        image = streetview.get_panorama(pano_id=pano.pano_id, multi_threaded=True)
        if isinstance(image, Image.Image):
            image.save(pano_path / f"{pano.pano_id}.jpg", "jpeg")
            return pano.pano_id
    except Exception as e:
        print(f"Failed to download panorama {pano.pano_id} at {node_coord}. {e}")

    return None


def download_streetview(node_coord, cache_dir, number=5):
    # Setup directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    pano_path = cache_dir / "pano"
    pano_path.mkdir(parents=True, exist_ok=True)

    # Search and filter panoramas
    sv_images = streetview.search_panoramas(lat=node_coord[0], lon=node_coord[1])
    nearby_panos = _filter_nearby_panoramas(sv_images, node_coord)

    # Download panoramas until we reach the desired number
    panos = []
    for pano in tqdm(nearby_panos, desc="Downloading panoramas", leave=False, position=1):
        pano_id = _download_single_panorama(pano, pano_path, node_coord)
        if pano_id:
            panos.append(pano_id)
            if len(panos) >= number:
                break

    return panos


class JunctionData:
    def __init__(self, cache_dir: Path = Path("/scratch/datasets/junctions")) -> None:
        self.cache_dir = cache_dir
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pose_csv = self.cache_dir / "poses.csv"

        self.cities = [
            "karlsruhe",
            "heidelberg",
            "freiburg",
        ]

        self._download_city_graphs()
        # Divide into train/val/test splits - randomly
        samples = pd.read_csv(self.pose_csv, header=None)
        samples.columns = ["city", "node_id", "lat", "lon", "sat_image", "street_images", "azis"]
        samples = samples.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        n = len(samples)
        self.train_samples = samples[: int(0.8 * n)]
        self.val_samples = samples[int(0.8 * n) : int(0.9 * n)]
        self.test_samples = samples[int(0.9 * n) :]

    def _download_city_graphs(self):
        for city in self.cities:
            g = ox.graph.graph_from_place(
                query=city,
                network_type="drive",
                simplify=True,
                retain_all=False,
                truncate_by_edge=False,
                custom_filter=None,
            )
            g = ox.projection.project_graph(g, to_latlong=True)
            g.remove_edges_from(nx.selfloop_edges(g))
            g = simplify_sharp_turns(g, min_total_turn_deg=20)

            for n in tqdm(
                g.nodes(data=True),
                desc=f"Downloading data for {city}",
                leave=False,
                position=0,
            ):
                lat, lon = float(n[1]["y"]), float(n[1]["x"])

                # Download satellite and street-level imagery for this node
                street_images = download_streetview((lat, lon), self.cache_dir / city)
                sat_image = download_satmap((lat, lon), self.cache_dir / city)

                # For number of outgoing edges, get the north-aligned exist yaw
                azis = []
                outgoing_edges = g.out_edges(n[0], data=True)
                for edge in outgoing_edges:
                    _, _, edge_data = edge
                    azis.append(extract_bearing_from_geometry(g, n[0], edge_data))

                if len(street_images) == 0 or sat_image is None:
                    continue  # Skip if we couldn't get any image pairs

                line = f"{city},{n[0]},{lat},{lon},{sat_image},{';'.join(street_images)},{';'.join(map(str, azis))}\n"
                line_exists = False
                if self.pose_csv.exists():
                    with open(self.pose_csv) as f:
                        if line in f.read():
                            line_exists = True

                if not line_exists:
                    with open(self.pose_csv, "a") as f:
                        f.write(line)


@dataclass
class JunctionDatasetConfig:
    data: JunctionData | None = None
    stage: str = "train"  # "train", "val", or "test"
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
    augmentations_train: A.Compose | None = None
    augmentations_val: A.Compose | None = None
    augmentations_train_sync: A.Compose | None = None
    use_progressive_augmentation: bool = True
    progressive_max_epochs: int = 100
    progressive_warmup_epochs: int = 10


class JunctionDataset(Dataset):
    """
    Junction dataset for cross-view geo-localization training and evaluation.

    Loads panoramas and satellite imagery for each junction node, with multiple
    panoramas per junction and multiple exit headings per junction.

    Args:
        config: JunctionDatasetConfig object containing all configuration parameters
    """

    def __init__(self, config: JunctionDatasetConfig):
        super().__init__()
        self.config = config

        assert config.stage in ["train", "val", "test"], "stage must be 'train', 'val', or 'test'"
        assert config.mode in [
            "triplet",
            "query",
            "reference",
        ], "mode must be 'triplet', 'query', or 'reference'"

        # Load samples from JunctionData
        self.data = config.data
        self.samples = getattr(self.data, f"{config.stage}_samples")

        # Set up augmentations
        from taco.sensors.cvgl.cvusa import ProgressiveAugmentation

        self.ProgressiveAugmentation = ProgressiveAugmentation
        self.augmentations = self._setup_augmentations()
        self.augmentations_sync = self._setup_sync_augmentations()

        # Prepare index mappings
        self._setup_mode_specific_data()

    def _setup_augmentations(self):
        """Set up the main augmentation pipeline based on config."""
        config = self.config
        if config.stage == "train" and config.augmentations_train is not None:
            if config.use_progressive_augmentation:
                return self.ProgressiveAugmentation(
                    config.augmentations_train,
                    max_epochs=config.progressive_max_epochs,
                    warmup_epochs=config.progressive_warmup_epochs,
                )
            return config.augmentations_train
        elif config.stage in ["val", "test"] and config.augmentations_val is not None:
            return config.augmentations_val
        return None

    def _setup_sync_augmentations(self):
        """Set up synchronized augmentations for query+reference pairs."""
        config = self.config
        if config.stage == "train" and config.mode == "triplet":
            if config.augmentations_train_sync is not None and config.use_progressive_augmentation:
                return self.ProgressiveAugmentation(
                    config.augmentations_train_sync,
                    max_epochs=config.progressive_max_epochs,
                    warmup_epochs=config.progressive_warmup_epochs,
                )
            return config.augmentations_train_sync
        return None

    def _setup_mode_specific_data(self):
        """Set up data structures based on dataset mode (triplet, query, or reference)."""
        if self.config.mode == "triplet":
            self._setup_triplet_mode()
        elif self.config.mode == "reference":
            self._setup_reference_mode()
        elif self.config.mode == "query":
            self._setup_query_mode()

    def _setup_triplet_mode(self):
        """Set up triplet mode: expand samples to all panorama-heading pairs."""
        self.triplets = []
        self.idx2triplet = {}

        for _, row in self.samples.iterrows():
            city = row["city"]
            node_id = row["node_id"]
            lat = row["lat"]
            lon = row["lon"]
            sat_image = row["sat_image"]
            street_images = row["street_images"].split(";")
            azis = [float(azi) for azi in row["azis"].split(";")]

            # Create triplets for each (panorama, heading) pair
            for pano_id in street_images:
                for azi in azis:
                    triplet_idx = len(self.triplets)
                    triplet = {
                        "idx": triplet_idx,
                        "city": city,
                        "node_id": node_id,
                        "lat": lat,
                        "lon": lon,
                        "sat_image": sat_image,
                        "street_image": pano_id,
                        "heading": azi,
                    }
                    self.triplets.append(triplet)
                    self.idx2triplet[triplet_idx] = triplet

        self.train_ids = list(range(len(self.triplets)))
        self.samples_list = copy.deepcopy(self.train_ids)

    def _setup_reference_mode(self):
        """Set up reference mode: satellite images only."""
        self.images = []
        self.labels = []
        for idx, row in self.samples.iterrows():
            self.images.append((row["city"], row["sat_image"]))
            self.labels.append(idx)

    def _setup_query_mode(self):
        """Set up query mode: expand to all panorama-heading pairs."""
        self.images = []
        self.labels = []
        for idx, row in self.samples.iterrows():
            city = row["city"]
            street_images = row["street_images"].split(";")
            azis = [float(azi) for azi in row["azis"].split(";")]

            for pano_id in street_images:
                for azi in azis:
                    self.images.append((city, pano_id, azi))
                    self.labels.append(idx)

    def __len__(self):
        if self.config.mode == "triplet":
            return len(self.samples_list)
        return len(self.images)

    def __getitem__(self, index):
        if self.config.mode == "triplet":
            return self._get_triplet(index)
        return self._get_single_image(index)

    def _get_triplet(self, index):
        """
        Get triplet for training: (query, reference, label).

        Processing order:
        1. Load query panorama and reference satellite images
        2. Rotate reference by heading (if enabled)
        3. Crop query with gnomonic projection at heading (if enabled)
        4. Resize reference to match query size (for synchronized augmentations)
        5. Apply synchronized augmentations (flip, rotate) to both images
        6. Apply independent augmentations (color, blur, noise) to each image
        7. Apply timm transforms (resize, normalize, to tensor)
        """
        triplet = self.idx2triplet[self.samples_list[index]]
        city = triplet["city"]
        sat_image = triplet["sat_image"]
        street_image = triplet["street_image"]
        heading_rad = triplet["heading"]
        idx = triplet["idx"]

        # Load images
        query_img = Image.open(self.data.cache_dir / city / "pano" / f"{street_image}.jpg").convert(
            "RGB"
        )
        reference_img = Image.open(self.data.cache_dir / city / "sat" / sat_image).convert("RGB")

        # Convert heading from radians to degrees for processing
        heading_deg = np.rad2deg(heading_rad) % 360.0

        # Optionally rotate reference image by ground truth heading
        if self.config.rotate_reference_by_heading:
            reference_img = self._rotate_image_by_heading(
                reference_img, heading_deg, crop_to_fit=self.config.crop_rotated_reference
            )

        # Apply gnomonic projection to extract view at heading
        if self.config.use_gnomonic_projection:
            if self.config.random_heading and self.config.stage == "train":
                heading_offset = np.random.uniform(-180.0, 180.0)
                heading_deg = (heading_deg + heading_offset) % 360.0

            if self.config.random_pitch and self.config.stage == "train":
                pitch_deg = np.random.uniform(*self.config.pitch_range)
            else:
                pitch_deg = 0.0

            query_img = gnomonic_projection(
                query_img,
                heading_deg=float(heading_deg),
                pitch_deg=float(pitch_deg),
                fov_deg=self.config.gnomonic_fov_deg,
                output_shape=self.config.gnomonic_output_shape,
            )

        # Ensure both images are the same size before augmentation
        if query_img.size != reference_img.size:
            reference_img = reference_img.resize(query_img.size, Image.BILINEAR)

        # Temporarily save
        query_img.save(f"tmp/{index}_q.jpg")
        reference_img.save(f"tmp/{index}_r.jpg")

        # Convert PIL images to numpy arrays for albumentations
        query_np = np.array(query_img)
        reference_np = np.array(reference_img)

        # Apply synchronized augmentations (query + reference together)
        if self.augmentations_sync is not None:
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

        # Image transforms (timm pipeline handles PIL -> tensor + normalize)
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
        if self.config.mode == "reference":
            city, sat_image = self.images[index]
            img = Image.open(self.data.cache_dir / city / "sat" / sat_image).convert("RGB")
            idx = self.labels[index]

            # Optionally rotate reference by ground truth heading
            if self.config.rotate_reference_by_heading:
                # For reference mode, we need to get heading from the original sample
                row = self.samples.iloc[idx]
                azis = [float(azi) for azi in row["azis"].split(";")]
                # Use first heading as representative
                heading_deg = np.rad2deg(azis[0]) % 360.0
                img = self._rotate_image_by_heading(
                    img, heading_deg, crop_to_fit=self.config.crop_rotated_reference
                )

        elif self.config.mode == "query":
            city, pano_id, heading_rad = self.images[index]
            img = Image.open(self.data.cache_dir / city / "pano" / f"{pano_id}.jpg").convert("RGB")
            idx = self.labels[index]
            heading_deg = np.rad2deg(heading_rad) % 360.0

            # Apply gnomonic projection for query images
            if self.config.use_gnomonic_projection:
                img = gnomonic_projection(
                    img,
                    heading_deg=float(heading_deg),
                    pitch_deg=0.0,
                    fov_deg=self.config.gnomonic_fov_deg,
                    output_shape=self.config.gnomonic_output_shape,
                )

        # Apply albumentations augmentations
        if self.augmentations is not None:
            img_np = np.array(img)
            img_np = self.augmentations(image=img_np)["image"]
            img = Image.fromarray(img_np)

        # Image transforms (timm pipeline handles PIL -> tensor + normalize)
        if self.config.mode == "query" and self.config.transforms_query is not None:
            img = self.config.transforms_query(img)
        elif self.config.mode == "reference" and self.config.transforms_reference is not None:
            img = self.config.transforms_reference(img)
        else:
            img = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
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

    def shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        Custom shuffle function for unique class_id sampling in batch.
        Only applicable when mode='triplet'.
        """
        if self.config.mode != "triplet":
            raise ValueError("Shuffle is only supported in triplet mode")

        print("\nShuffle Junction Dataset:")

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

        self.samples_list = batches

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


if __name__ == "__main__":
    # Example: Download junction data
    data = JunctionData()

    # Example: Create training dataset with augmentations
    from taco.sensors.cvgl.cvusa import (
        create_default_augmentations,
        create_synchronized_augmentations,
    )

    # Create augmentation pipelines
    train_augs = create_default_augmentations(
        stage="train",
        strength="progressive",  # Starts weak, increases during training
    )

    # Create synchronized augmentations for query+reference pairs
    sync_augs = create_synchronized_augmentations(
        prob_flip=0.5,
        prob_rotate=0.5,
        rotate_limit=10,  # Rotate up to ±10 degrees
        use_90deg_rotations=False,
    )

    # Create training dataset config
    config = JunctionDatasetConfig(
        data=data,
        stage="train",
        mode="triplet",
        use_gnomonic_projection=True,
        gnomonic_fov_deg=90.0,
        gnomonic_output_shape=(384, 384),
        rotate_reference_by_heading=True,
        crop_rotated_reference=True,
        network_input_size=(384, 384),
        augmentations_train=train_augs,
        augmentations_train_sync=sync_augs,
        use_progressive_augmentation=True,
        progressive_max_epochs=100,
        progressive_warmup_epochs=10,
    )

    # Create dataset
    dataset = JunctionDataset(config)
    print(f"Training dataset size: {len(dataset)}")

    # Create validation dataset (no augmentations)
    val_config = JunctionDatasetConfig(
        data=data,
        stage="val",
        mode="triplet",
        use_gnomonic_projection=True,
        gnomonic_fov_deg=90.0,
        gnomonic_output_shape=(384, 384),
        rotate_reference_by_heading=True,
        crop_rotated_reference=True,
        network_input_size=(384, 384),
    )

    val_dataset = JunctionDataset(val_config)
    print(f"Validation dataset size: {len(val_dataset)}")

    # Iterate through a few samples
    if len(dataset) > 0:
        for i in range(min(3, len(dataset))):
            query_img, reference_img, label = dataset[i]
            print(
                f"Sample {i}: Query shape: {query_img.shape}, "
                f"Reference shape: {reference_img.shape}, Label: {label}"
            )
