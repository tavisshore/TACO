import argparse


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train image retrieval model")

    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Dataset configuration
    parser.add_argument(
        "--data-folder",
        type=str,
        default="/scratch/datasets/CVUSA",
        help="Path to dataset folder",
    )
    # KITTI sequence
    parser.add_argument(
        "--sequence",
        type=int,
        default=0,
        help="KITTI odometry sequence to use (e.g., 0, 1, ...)",
    )

    # Verbose
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging during training"
    )

    # Training configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=28,
        help="Initial batch size for training (ignored if --auto-batch-size is set)",
    )
    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Automatically find maximum batch size that fits in GPU memory",
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")

    # Multi-GPU configuration
    parser.add_argument(
        "--devices", type=str, default="auto", help="Number of devices to use (e.g., 1, 2, 'auto')"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Training strategy for multi-GPU (auto, ddp, fsdp, etc.)",
    )

    # Model configuration
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ntxent",
        choices=["combined", "ntxent", "triplet"],
        help="Loss type to use",
    )

    # Scheduler configuration
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--scheduler-t-max", type=int, default=100, help="Cosine scheduler T_max parameter"
    )
    parser.add_argument(
        "--scheduler-eta-min",
        type=float,
        default=1e-6,
        help="Cosine scheduler minimum learning rate",
    )

    # Model architecture
    parser.add_argument(
        "--model-name",
        type=str,
        default="convnext",
        choices=["convnext", "sample4geo"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="timm/convnext_tiny.fb_in22k_ft_in1k_384",
        help="Model variant",
    )
    parser.add_argument("--img-size", type=int, default=384, help="Input image size")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights")

    # Training callbacks
    parser.add_argument(
        "--early-stop-patience", type=int, default=25, help="Early stopping patience in epochs"
    )
    parser.add_argument(
        "--shuffle-update-freq",
        type=int,
        default=5,
        help="How often to update similarity dictionary",
    )
    parser.add_argument(
        "--neighbour-select",
        type=int,
        default=64,
        help="Number of neighbors to select for batching",
    )
    parser.add_argument(
        "--neighbour-range", type=int, default=128, help="Range of top neighbors to sample from"
    )
    parser.add_argument(
        "--sim-shuffle",
        action="store_true",
        help="Enable similarity-based batch shuffling",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--experiment-name", type=str, default="initial", help="Experiment name for W&B"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/scratch/datasets/CVUSA/output",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose logging"
    )

    return parser.parse_args()
