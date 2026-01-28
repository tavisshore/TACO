import os
from dataclasses import dataclass

import torch


@dataclass
class TrainingConfiguration:
    # Model
    model: str = "convnext_base.fb_in22k_ft_in1k_384"

    # Override model image size
    img_size: int = 384

    # Training
    mixed_precision: bool = True
    seed = 1
    epochs: int = 40
    batch_size: int = 128  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0, 1, 2, 3, 4, 5, 6, 7)  # GPU ids for training

    # Similarity Sampling
    custom_sampling: bool = True  # use custom sampling instead of random
    gps_sample: bool = True  # use gps sampling
    sim_sample: bool = True  # use similarity sampling
    neighbour_select: int = 64  # max selection size from pool
    neighbour_range: int = 128  # pool size for selection
    gps_dict_path: str = "./data/VIGOR/gps_dict_same.pkl"  # gps_dict_cross.pkl | gps_dict_same.pkl

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 4  # eval every n Epoch
    normalize_features: bool = True

    # Optimizer
    clip_grad = 100.0  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 1
    lr_end: float = 0.0001  #  only for "polynomial"

    # Dataset
    data_folder = "./data/VIGOR"
    same_area: bool = True  # True: same | False: cross

    # pre extracted polar view for ground
    ground_extracted: bool = False

    # Inv_Polar Transformation
    ground_polar = False  # use warp polar inv for ground
    ground_polar_align: int = 1  # 1: start 12pm
    ground_polar_center_radius = 64
    ground_cutting = 0  # cut ground

    # Augment Images
    prob_rotate: float = 0.75  # rotates the sat image and ground images simultaneously
    prob_flip: float = 0.5  # flipping the sat image and ground images simultaneously
    satellite_aug: bool = False  # slightly shift sat image
    ground_aug: bool = False  # slightly change view for ground or rotate if pre-extracted

    # Savepath for model checkpoints
    model_path: str = "./vigor_same"

    # Eval before training
    zero_shot: bool = False

    # Checkpoint to start from
    checkpoint_start = None

    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == "nt" else 4

    # train on GPU if available
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # for better performance
    cudnn_benchmark: bool = True

    # make cudnn deterministic
    cudnn_deterministic: bool = False


# -----------------------------------------------------------------------------#
# Train Config                                                                #
# -----------------------------------------------------------------------------#

config = TrainingConfiguration()
