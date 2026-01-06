"""Full pipeline example using TACO for KITTI pose graph optimization.

This example demonstrates:
1. Loading KITTI data with IMU and image sequences
2. Integrating IMU measurements using TACO's IMUIntegrator
3. Creating a pose graph with GTSAM factors
4. Adding CVGL localization measurements
5. Incremental optimization with iSAM2
6. Visualizing results
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from taco.data.kitti import Kitti
from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.sensors.cvgl import CVGLMeasurement
from taco.sensors.imu import IMUData, IMUIntegrator
from taco.utils.conversions import numpy_pose_to_gtsam, rotation_matrix_to_quaternion
from taco.visualization import plot_trajectory


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TACO Full Pipeline Example")
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
    return parser.parse_args()


def create_imu_measurements_from_kitti(
    kitti_data: Kitti, start_idx: int, end_idx: int
) -> list[IMUData]:
    """Convert KITTI IMU data to TACO IMUData format.

    Args:
        kitti_data: KITTI data loader instance.
        start_idx: Start frame index.
        end_idx: End frame index.

    Returns:
        List of IMUData measurements.
    """
    imu_measurements = []

    for i in range(start_idx, end_idx):
        # Get raw IMU data from KITTI
        acc = kitti_data.acc[i].numpy()
        gyro = kitti_data.gyro[i].numpy()

        # Create timestamp (cumulative from start)
        timestamp = sum(kitti_data.dt[start_idx:i].numpy()) if i > start_idx else 0.0

        imu = IMUData.from_raw(
            timestamp=timestamp,
            accel_x=acc[0],
            accel_y=acc[1],
            accel_z=acc[2],
            gyro_x=gyro[0],
            gyro_y=gyro[1],
            gyro_z=gyro[2],
        )
        imu_measurements.append(imu)

    return imu_measurements


def main() -> None:
    """Run the full TACO pipeline on KITTI data."""
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    print("TACO Full Pipeline - KITTI Pose Graph Optimization")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KITTI data
    print(f"\n1. Loading KITTI sequence {args.sequence}...")
    data = Kitti(args)
    print(f"   Loaded {len(data)} frames")

    # Initialize pose graph
    print("\n2. Initializing pose graph...")
    graph = PoseGraph()

    # Get initial pose from KITTI ground truth
    init_values = data.get_init_value()
    init_pos = init_values["pos"][0].numpy()
    init_rot = init_values["rot"][0].numpy()

    # Convert rotation matrix to quaternion for TACO
    # PyPose SO3 -> rotation matrix -> quaternion
    init_rot_matrix = init_rot[:3, :3] if init_rot.ndim == 2 else np.eye(3)
    init_quat = rotation_matrix_to_quaternion(init_rot_matrix)

    initial_pose_gtsam = numpy_pose_to_gtsam(init_pos, init_quat)

    # Add first pose with strong prior
    pose_id_0 = graph.add_pose_estimate(initial_pose_gtsam, timestamp=0.0)
    prior_noise = create_noise_model_diagonal(
        np.array([0.01, 0.01, 0.01, 0.005, 0.005, 0.005])  # Very tight prior
    )
    graph.add_prior_factor(pose_id_0, initial_pose_gtsam, prior_noise)
    print(f"   Added initial pose (ID: {pose_id_0}) with prior")

    # Initialize IMU integrator
    integrator = IMUIntegrator(gravity=np.array([0.0, 0.0, -9.81]))

    # Storage for trajectories
    trajectories = {
        "gt": [init_pos.tolist()],
        "imu": [init_pos.tolist()],
        "optimized": [init_pos.tolist()],
    }

    # Process frames
    print("\n3. Processing frames...")
    current_pose = initial_pose_gtsam
    current_pose_id = pose_id_0
    imu_integration_window = 5  # Integrate every N frames

    for idx in tqdm(range(1, len(data)), desc=f"Sequence {args.sequence}"):
        # Get ground truth for comparison
        _frame, gt_pose, _frame_id, _yaw = data.get_next_data()
        trajectories["gt"].append(gt_pose)

        # Get IMU data and integrate
        if idx >= imu_integration_window:
            # Create IMU measurements for this window
            imu_measurements = create_imu_measurements_from_kitti(
                data, idx - imu_integration_window, idx
            )

            if len(imu_measurements) > 1:
                # Integrate IMU to get next pose estimate
                next_pose_gtsam = integrator.integrate_to_gtsam_pose(imu_measurements, current_pose)

                # Store IMU trajectory
                next_t = next_pose_gtsam.translation()
                trajectories["imu"].append([next_t.x(), next_t.y(), next_t.z()])

                # Add new pose to graph
                timestamp = idx * 0.1  # Approximate timestamp
                next_pose_id = graph.add_pose_estimate(next_pose_gtsam, timestamp=timestamp)

                # Add between factor (odometry constraint from IMU)
                relative_pose = current_pose.between(next_pose_gtsam)
                odometry_noise = create_noise_model_diagonal(
                    np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
                )
                graph.add_between_factor(
                    current_pose_id, next_pose_id, relative_pose, odometry_noise
                )

                # Add CVGL measurement periodically (every 20 frames)
                if idx % 20 == 0:
                    # Get CVGL localization (using ground truth coords as proxy)
                    cvgl_coord = data.get_coord(idx)
                    cvgl_yaw = data.get_yaw(idx)

                    # Create CVGL measurement
                    # Convert lat/lon to local frame (simplified - in practice use proper projection)
                    start_coord = data.get_start_coord()
                    cvgl_pos = np.array(
                        [
                            (cvgl_coord[1] - start_coord[1])
                            * 111320
                            * np.cos(np.radians(start_coord[0])),
                            (cvgl_coord[0] - start_coord[0]) * 110540,
                            0.0,
                        ]
                    )

                    # Create quaternion from yaw
                    cvgl_quat = np.array(
                        [
                            np.cos(cvgl_yaw / 2),
                            0.0,
                            0.0,
                            np.sin(cvgl_yaw / 2),
                        ]
                    )

                    cvgl_measurement = CVGLMeasurement(
                        timestamp=timestamp,
                        position=cvgl_pos,
                        orientation=cvgl_quat,
                        covariance=np.eye(6) * 0.5,  # CVGL uncertainty
                        confidence=0.85,
                        num_inliers=100,
                    )

                    # Add CVGL factor
                    cvgl_pose_gtsam = cvgl_measurement.to_gtsam_pose()
                    cvgl_noise = cvgl_measurement.get_gtsam_noise_model()
                    graph.add_pose_factor(next_pose_id, cvgl_pose_gtsam, cvgl_noise)

                # Update current pose for next iteration
                current_pose = next_pose_gtsam
                current_pose_id = next_pose_id

        # Increment frame counter
        data.frame_id += 1

    # Optimize graph
    print("\n4. Optimizing pose graph...")
    print(f"   Number of factors: {graph.size()}")
    print(f"   Initial error: {graph.get_error():.4f}")

    graph.optimize(optimizer_type="LevenbergMarquardt", max_iterations=100)
    print(f"   Final error: {graph.get_error():.4f}")

    # Extract optimized trajectory
    print("\n5. Extracting optimized trajectory...")
    optimized_poses = graph.get_all_poses()
    for pose_id in sorted(optimized_poses.keys()):
        pose = optimized_poses[pose_id]
        t = pose.translation()
        trajectories["optimized"].append([t.x(), t.y(), t.z()])

    # Visualize results
    print("\n6. Visualizing results...")

    # Plot ground truth trajectory
    gt_positions = np.array(trajectories["gt"])
    plot_trajectory(
        gt_positions,
        title=f"KITTI Sequence {args.sequence} - Ground Truth",
        show=False,
    )

    # Plot optimized trajectory
    if len(trajectories["optimized"]) > 1:
        opt_positions = np.array(trajectories["optimized"])
        plot_trajectory(
            opt_positions,
            title=f"KITTI Sequence {args.sequence} - Optimized",
            show=True,
        )

    # Save trajectories
    output_file = output_dir / f"trajectory_seq{args.sequence}.npz"
    np.savez(
        output_file,
        gt=np.array(trajectories["gt"]),
        imu=np.array(trajectories["imu"]) if trajectories["imu"] else np.array([]),
        optimized=np.array(trajectories["optimized"]),
    )
    print(f"\n   Saved trajectories to {output_file}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
