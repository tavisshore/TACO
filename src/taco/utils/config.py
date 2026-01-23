import argparse


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TACO Full Pipeline Example")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with additional outputs",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        default="/scratch/datasets/KITTI/odometry/dataset/sequences_jpg",
        help="Path to KITTI dataset",
    )
    parser.add_argument(
        "-s",
        "--sequence",
        type=int,
        default=0,
        help="KITTI sequence to evaluate (0-10)",
    )
    parser.add_argument(
        "-p",
        "--pose_path",
        default="/scratch/datasets/KITTI/odometry/dataset/poses",
        help="Path to ground truth poses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    return parser.parse_args()
