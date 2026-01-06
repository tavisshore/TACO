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
    custom_sampling: bool = True  # use custom sampling instead of random
    seed = 1
    epochs: int = 1
    batch_size: int = 128  # keep in mind real_batch_size = 2 * batch_size
    verbose: bool = True
    gpu_ids: tuple = (0, 1, 2, 3)  # GPU ids for training

    # Eval
    batch_size_eval: int = 128
    eval_every_n_epoch: int = 1  # eval every n Epoch
    normalize_features: bool = True
    eval_gallery_n: int = -1  # -1 for all or int

    # Optimizer
    clip_grad = 100.0  # None | float
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False  # Gradient Checkpointing

    # Loss
    label_smoothing: float = 0.1

    # Learning Rate
    lr: float = 0.001  # 1 * 10^-4 for ViT | 1 * 10^-1 for CNN
    scheduler: str = "cosine"  # "polynomial" | "cosine" | "constant" | None
    warmup_epochs: int = 0.1
    lr_end: float = 0.0001  #  only for "polynomial"

    # Dataset
    dataset: str = "U1652-S2D"  # 'U1652-D2S' | 'U1652-S2D'
    data_folder: str = "./data/U1652"

    # Augment Images
    prob_flip: float = 0.5  # flipping the sat image and drone image simultaneously

    # Savepath for model checkpoints
    model_path: str = "./university"

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

if config.dataset == "U1652-D2S":
    config.query_folder_train = "./data/U1652/train/satellite"
    config.gallery_folder_train = "./data/U1652/train/drone"
    config.query_folder_test = "./data/U1652/test/query_drone"
    config.gallery_folder_test = "./data/U1652/test/gallery_satellite"
elif config.dataset == "U1652-S2D":
    config.query_folder_train = "./data/U1652/train/satellite"
    config.gallery_folder_train = "./data/U1652/train/drone"
    config.query_folder_test = "./data/U1652/test/query_satellite"
    config.gallery_folder_test = "./data/U1652/test/gallery_drone"
