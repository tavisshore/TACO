"""Full pipeline example using TACO for KITTI pose graph optimization.

This example demonstrates:
1. Loading KITTI data with IMU and image sequences
2. Integrating IMU measurements using TACO's IMUIntegrator
3. Creating a 2D pose graph with GTSAM Pose2 factors
4. Adding CVGL localization measurements
5. Optimization with Levenberg-Marquardt
6. Visualizing results

The pose graph uses Pose2 (x, y, yaw) for 2D vehicle trajectory estimation.
"""

import argparse
from pathlib import Path

import numpy as np
import pypose as pp
from tqdm import tqdm

from taco.data.kitti import Kitti
from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.sensors.imu import IMUData, IMUIntegrator
from taco.utils.conversions import numpy_pose_to_gtsam, quaternion_to_yaw, rotation_matrix_to_yaw
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
    timestamp = 0.0
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

    print("TACO Full Pipeline - KITTI Pose Graph Optimisation (Pose2)")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KITTI data
    print(f"\n1. Loading KITTI sequence {args.sequence}...")
    data = Kitti(args)
    print(f"   Loaded {len(data)} frames")

    # Initialize pose graph
    print("\n2. Initialising pose graph...")
    graph = PoseGraph()

    # Get initial pose from KITTI ground truth (already in meters)
    init_values = data.get_init_value()
    init_pos = init_values["pos"][0].numpy()
    init_rot = init_values["rot"][0].numpy()

    # Get 2D pose (x, y, yaw) - positions are already in meters
    init_pos_2d = np.array([init_pos[0], init_pos[1]])
    if init_rot.shape == (3, 3):
        init_yaw = rotation_matrix_to_yaw(init_rot)
    else:
        # If quaternion, extract yaw directly
        init_yaw = quaternion_to_yaw(init_rot)

    initial_pose_gtsam = numpy_pose_to_gtsam(init_pos_2d, init_yaw)

    # Add first pose with strong prior (3 DOF for Pose2: x, y, theta)
    pose_id_0 = graph.add_pose_estimate(initial_pose_gtsam, timestamp=0.0)
    prior_noise = create_noise_model_diagonal(
        np.array([0.01, 0.01, 0.005])  # Very tight prior for x, y, theta
    )
    graph.add_prior_factor(pose_id_0, initial_pose_gtsam, prior_noise)
    print(f"   Added initial pose (ID: {pose_id_0}) with prior")

    # Initialize IMU integrator
    # integrator = IMUIntegrator()
    integrator = pp.module.IMUPreintegrator(
        init_values["pos"], init_values["rot"], init_values["vel"], reset=False
    ).to("cpu")
    imu_state = {"pos": init_values["pos"], "rot": init_values["rot"], "vel": init_values["vel"]}

    # Storage for trajectories (store as 2D in meters)
    trajectories = {
        "gt": [[init_pos[0], init_pos[1]]],
        "imu": [[init_pos[0], init_pos[1]]],
        "optimised": [[init_pos[0], init_pos[1]]],
    }

    # Process frames
    print("\n3. Processing frames...")
    current_pose = initial_pose_gtsam
    current_pose_id = pose_id_0

    # Use length of gt_pos_meters to avoid index out of bounds
    num_frames = len(data.gt_pos_meters)
    for idx in tqdm(range(1, num_frames), desc=f"Sequence {args.sequence}"):
        timestamp = data.get_timestamp(idx)

        # Get ground truth position in meters
        gt_pos_meters = data.get_pos_meters(idx)
        trajectories["gt"].append([gt_pos_meters[0], gt_pos_meters[1]])

        # Get IMU data and integrate
        imu_data = data.get_imu(idx)

        imu_predict = integrator(
            dt=imu_data["dt"],
            gyro=imu_data["gyro"],
            acc=imu_data["acc"],
            rot=imu_data["init_rot"],
            init_state=imu_state,
        )
        imu_state = {
            "pos": imu_predict["pos"][..., -2, :],
            "rot": imu_predict["rot"][..., -2, :],
            "vel": imu_predict["vel"][..., -2, :],
        }
        # Extract 3D position and rotation from IMU prediction
        next_pos_3d = imu_predict["pos"][..., -1, :].cpu().numpy()[0]
        next_rot_quat = (
            imu_predict["rot"][..., -1, :].cpu().numpy().flatten()
        )  # pypose quaternion [x,y,z,w]
        # Convert pypose quaternion [x,y,z,w] to [w,x,y,z] for our conversion function
        quat_wxyz = np.array(
            [next_rot_quat[3], next_rot_quat[0], next_rot_quat[1], next_rot_quat[2]]
        )
        next_yaw = quaternion_to_yaw(quat_wxyz)
        # Create gtsam.Pose2 from x, y, yaw
        next_pose_gtsam = numpy_pose_to_gtsam(next_pos_3d[:2], next_yaw)
        # Store IMU trajectory (x, y from Pose2)
        trajectories["imu"].append([next_pose_gtsam.x(), next_pose_gtsam.y()])

        # Add new pose to graph
        # timestamp = idx * 0.1  # Approximate timestamp
        next_pose_id = graph.add_pose_estimate(next_pose_gtsam, timestamp=timestamp)

        # Add between factor (odometry constraint from IMU)
        # Use high uncertainty since IMU integration drifts significantly
        relative_pose = current_pose.between(next_pose_gtsam)
        odometry_noise = create_noise_model_diagonal(
            np.array([1.0, 1.0, 0.25])  # High uncertainty for x, y, theta
        )
        graph.add_between_factor(current_pose_id, next_pose_id, relative_pose, odometry_noise)

        # GT from GPS
        if idx % 50 == 0:
            # Create Pose2 from ground truth (x, y in meters, yaw)
            gt_pos_2d = np.array([gt_pos_meters[0], gt_pos_meters[1]])
            # Use current yaw estimate since GT doesn't provide it directly here
            gt_yaw = next_pose_gtsam.theta()
            gt_pose_gtsam = numpy_pose_to_gtsam(gt_pos_2d, gt_yaw)
            # Use pose factor with low noise (high confidence) to anchor trajectory
            pose_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.05]))
            graph.add_pose_factor(next_pose_id, gt_pose_gtsam, pose_noise)

        # Update current pose for next iteration
        current_pose = next_pose_gtsam
        current_pose_id = next_pose_id

        # Increment frame counter
        data.frame_id += 1

    # Optimise graph
    print("\n4. Optimising pose graph...")
    print(f"   Number of factors: {graph.size()}")
    print(f"   Initial error: {graph.get_error():.4f}")

    graph.optimize(optimizer_type="LevenbergMarquardt", max_iterations=100)
    print(f"   Final error: {graph.get_error():.4f}")

    # Extract optimised trajectory
    print("\n5. Extracting optimised trajectory...")
    optimized_poses = graph.get_all_poses()
    for pose_id in sorted(optimized_poses.keys()):
        pose = optimized_poses[pose_id]
        trajectories["optimised"].append([pose.x(), pose.y()])

    # Visualize results
    print("\n6. Visualising results...")

    # Plot ground truth trajectory (positions are already in meters)
    gt_positions = np.array(trajectories["gt"])
    fig1 = plot_trajectory(
        gt_positions,
        title=f"KITTI Sequence {args.sequence} - Ground Truth",
        show=False,
        convert_latlon=False,  # Data is already in meters
    )
    fig1.savefig(output_dir / f"gt_trajectory_seq{args.sequence}.png", dpi=150)
    print(f"   Saved gt_trajectory_seq{args.sequence}.png")

    # Plot IMU trajectory
    if len(trajectories["imu"]) > 1:
        imu_positions = np.array(trajectories["imu"])
        fig_imu = plot_trajectory(
            imu_positions,
            title=f"KITTI Sequence {args.sequence} - IMU Integrated",
            show=False,
            convert_latlon=False,  # Data is already in meters
        )
        fig_imu.savefig(output_dir / f"imu_trajectory_seq{args.sequence}.png", dpi=150)
        print(f"   Saved imu_trajectory_seq{args.sequence}.png")

    # Plot optimised trajectory
    if len(trajectories["optimised"]) > 1:
        opt_positions = np.array(trajectories["optimised"])
        fig2 = plot_trajectory(
            opt_positions,
            title=f"KITTI Sequence {args.sequence} - Optimised",
            show=False,
            convert_latlon=False,  # Data is already in meters
        )
        fig2.savefig(output_dir / f"optimised_trajectory_seq{args.sequence}.png", dpi=150)
        print(f"   Saved optimised_trajectory_seq{args.sequence}.png")

    # Save trajectories
    output_file = output_dir / f"trajectory_seq{args.sequence}.npz"
    np.savez(
        output_file,
        gt=np.array(trajectories["gt"]),
        imu=np.array(trajectories["imu"]) if trajectories["imu"] else np.array([]),
        optimised=np.array(trajectories["optimised"]),
    )
    print(f"   Saved trajectories to {output_file}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")


if __name__ == "__main__":
    main()
