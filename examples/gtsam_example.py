"""Example of using TACO with GTSAM Pose2 for 2D pose graph optimization.

This example demonstrates:
1. Creating a 2D pose graph with GTSAM Pose2
2. Adding IMU odometry factors
3. Adding CVGL localization factors
4. Optimizing the graph
5. Visualizing results

The pose graph represents a vehicle trajectory using only x, y, and yaw.
"""

import numpy as np

from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.sensors.cvgl import CVGLMeasurement
from taco.sensors.imu import IMUData, IMUIntegrator
from taco.utils.conversions import numpy_pose_to_gtsam
from taco.visualization import plot_gtsam_values, plot_pose_graph


def main() -> None:
    """Run GTSAM Pose2 pose graph optimization example."""
    print("TACO GTSAM Pose2 Example")
    print("=" * 50)

    # Create pose graph
    graph = PoseGraph()
    print("\n1. Created pose graph for 2D trajectory (x, y, yaw)")

    # Define initial pose (x, y, yaw)
    initial_position = np.array([0.0, 0.0])
    initial_yaw = 0.0  # radians

    initial_pose_gtsam = numpy_pose_to_gtsam(initial_position, initial_yaw)

    # Add first pose with strong prior
    pose_id_0 = graph.add_pose_estimate(initial_pose_gtsam, timestamp=0.0)

    # Noise model for Pose2: (x, y, theta) standard deviations
    prior_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.05]))

    graph.add_prior_factor(pose_id_0, initial_pose_gtsam, prior_noise)
    print(f"2. Added initial pose (ID: {pose_id_0}) with prior")

    # Simulate IMU measurements
    print("\n3. Simulating IMU measurements...")
    imu_measurements = []
    dt = 0.01  # 100 Hz
    for i in range(100):
        t = i * dt
        # Simulate moving forward with slight rotation
        accel_x = 1.0 if i < 50 else 0.0  # Accelerate then coast
        gyro_z = 0.1  # Slight yaw rotation

        imu = IMUData.from_raw(
            timestamp=t,
            accel_x=accel_x,
            accel_y=0.0,
            accel_z=-9.81,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=gyro_z,
        )
        imu_measurements.append(imu)

    # Integrate IMU to get odometry
    print("4. Integrating IMU measurements...")
    integrator = IMUIntegrator()
    next_pose_gtsam = integrator.integrate_to_gtsam_pose(imu_measurements, initial_pose_gtsam)

    # Add second pose
    pose_id_1 = graph.add_pose_estimate(next_pose_gtsam, timestamp=1.0)

    # Add between factor (odometry) - 3 DOF noise for Pose2
    relative_pose = initial_pose_gtsam.between(next_pose_gtsam)
    odometry_noise = create_noise_model_diagonal(np.array([0.5, 0.5, 0.1]))
    graph.add_between_factor(pose_id_0, pose_id_1, relative_pose, odometry_noise)
    print(f"5. Added odometry constraint between poses {pose_id_0} and {pose_id_1}")

    # Simulate CVGL measurement (2D localization)
    print("\n6. Simulating CVGL localization measurement...")
    cvgl_position = np.array([0.5, 0.1])  # Slightly offset from true position
    cvgl_yaw = 0.05  # Small rotation
    cvgl_measurement = CVGLMeasurement(
        timestamp=1.0,
        position=cvgl_position,
        yaw=cvgl_yaw,
        covariance=np.eye(3) * 0.1,  # 3x3 for Pose2
        confidence=0.95,
        num_inliers=150,
    )

    # Add CVGL factor
    cvgl_pose_gtsam = cvgl_measurement.to_gtsam_pose()
    cvgl_noise = cvgl_measurement.get_gtsam_noise_model()
    graph.add_pose_factor(pose_id_1, cvgl_pose_gtsam, cvgl_noise)
    print(f"7. Added CVGL localization factor to pose {pose_id_1}")

    # Print graph statistics
    print("\n8. Graph statistics:")
    print(f"   - Number of factors: {graph.size()}")
    print(f"   - Initial error: {graph.get_error():.4f}")

    # Optimize
    print("\n9. Optimizing pose graph...")
    result = graph.optimize(optimizer_type="LevenbergMarquardt", max_iterations=100)
    print(f"   - Final error: {graph.get_error():.4f}")

    # Get optimized poses
    print("\n10. Optimized poses (x, y, yaw):")
    for pose_id in range(2):
        pose = graph.get_pose(pose_id)
        if pose:
            print(f"   Pose {pose_id}: [{pose.x():.3f}, {pose.y():.3f}, {pose.theta():.3f}]")

    # Visualize
    print("\n11. Visualizing results...")
    fig1 = plot_pose_graph(graph, title="GTSAM Pose2 Graph Optimization", show=False)
    fig1.savefig("pose_graph.png", dpi=150)
    print("   Saved pose_graph.png")

    fig2 = plot_gtsam_values(result, [0, 1], title="Optimized 2D Trajectory", show=False)
    fig2.savefig("trajectory.png", dpi=150)
    print("   Saved trajectory.png")

    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
