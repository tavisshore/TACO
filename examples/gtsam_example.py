"""Example of using TACO with GTSAM for pose graph optimization.

This example demonstrates:
1. Creating a pose graph with GTSAM
2. Adding IMU odometry factors
3. Adding CVGL localization factors
4. Optimizing the graph
5. Visualizing results
"""

import numpy as np

from taco.pose_graph import PoseGraph, create_noise_model_diagonal
from taco.sensors.cvgl import CVGLMeasurement
from taco.sensors.imu import IMUData, IMUIntegrator
from taco.utils.conversions import numpy_pose_to_gtsam
from taco.visualization import plot_gtsam_values, plot_pose_graph


def main() -> None:
    """Run GTSAM pose graph optimization example."""
    print("TACO GTSAM Example")
    print("=" * 50)

    # Create pose graph
    graph = PoseGraph()
    print("\n1. Created pose graph")

    # Define initial pose
    initial_position = np.array([0.0, 0.0, 0.0])
    initial_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    initial_pose_gtsam = numpy_pose_to_gtsam(initial_position, initial_orientation)

    # Add first pose with strong prior
    pose_id_0 = graph.add_pose_estimate(initial_pose_gtsam, timestamp=0.0)
    prior_noise = create_noise_model_diagonal(np.array([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))
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
        gyro_z = 0.1  # Slight rotation

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
    integrator = IMUIntegrator(gravity=np.array([0.0, 0.0, -9.81]))
    next_pose_gtsam = integrator.integrate_to_gtsam_pose(imu_measurements, initial_pose_gtsam)

    # Add second pose
    pose_id_1 = graph.add_pose_estimate(next_pose_gtsam, timestamp=1.0)

    # Add between factor (odometry)
    relative_pose = initial_pose_gtsam.between(next_pose_gtsam)
    odometry_noise = create_noise_model_diagonal(np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]))
    graph.add_between_factor(pose_id_0, pose_id_1, relative_pose, odometry_noise)
    print(f"5. Added odometry constraint between poses {pose_id_0} and {pose_id_1}")

    # Simulate CVGL measurement
    print("\n6. Simulating CVGL localization measurement...")
    cvgl_position = np.array([0.5, 0.1, 0.0])  # Slightly offset from true position
    cvgl_orientation = np.array([0.9987, 0.0, 0.0, 0.0507])  # Small rotation
    cvgl_measurement = CVGLMeasurement(
        timestamp=1.0,
        position=cvgl_position,
        orientation=cvgl_orientation,
        covariance=np.eye(6) * 0.1,
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
    print("\n10. Optimized poses:")
    for pose_id in range(2):
        pose = graph.get_pose(pose_id)
        if pose:
            t = pose.translation()
            print(f"   Pose {pose_id}: [{t.x():.3f}, {t.y():.3f}, {t.z():.3f}]")

    # Visualize
    print("\n11. Visualizing results...")
    plot_pose_graph(graph, title="GTSAM Pose Graph Optimization", show=False)
    plot_gtsam_values(result, [0, 1], title="Optimized Trajectory", show=True)

    print("\n" + "=" * 50)
    print("Example complete!")


if __name__ == "__main__":
    main()
