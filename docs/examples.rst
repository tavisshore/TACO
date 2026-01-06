Examples
========

Basic Pose Graph
----------------

Creating and building a simple pose graph:

.. code-block:: python

    import numpy as np
    from taco.pose_graph import PoseGraph, PoseNode, Edge, EdgeType

    # Create graph
    graph = PoseGraph()

    # Add nodes
    for i in range(5):
        node = PoseNode(
            position=np.array([i * 1.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=float(i)
        )
        graph.add_node(node)

    # Add odometry edges
    for i in range(4):
        edge = Edge(
            from_node_id=i,
            to_node_id=i + 1,
            relative_transform=np.eye(4),
            information_matrix=np.eye(6) * 10.0,
            edge_type=EdgeType.IMU
        )
        graph.add_edge(edge)

IMU Processing
--------------

Loading and integrating IMU data:

.. code-block:: python

    from pathlib import Path
    import numpy as np
    from taco.sensors.imu import IMUIntegrator
    from taco.utils.io import load_imu_data

    # Load IMU data from CSV
    imu_measurements = load_imu_data(Path("imu_data.csv"))

    # Initialize integrator
    gravity = np.array([0.0, 0.0, -9.81])
    integrator = IMUIntegrator(gravity)

    # Integrate
    position, velocity, orientation = integrator.integrate(
        imu_measurements,
        initial_orientation=np.eye(3)
    )

    print(f"Final position: {position}")
    print(f"Final velocity: {velocity}")

IMU Preintegration
------------------

Using preintegration for efficient optimization:

.. code-block:: python

    from taco.sensors.imu import IMUPreintegrator
    import numpy as np

    # Initialize preintegrator
    gravity = np.array([0.0, 0.0, -9.81])
    preintegrator = IMUPreintegrator(
        gravity=gravity,
        gyro_noise=0.001,
        accel_noise=0.01
    )

    # Preintegrate between keyframes
    bias_accel = np.zeros(3)
    bias_gyro = np.zeros(3)
    preintegrator.integrate(imu_measurements, bias_accel, bias_gyro)

    # Access preintegrated measurements
    print(f"Delta position: {preintegrator.delta_p}")
    print(f"Delta velocity: {preintegrator.delta_v}")
    print(f"Delta rotation: {preintegrator.delta_R}")

CVGL Localization
-----------------

Using image-based localization:

.. code-block:: python

    from pathlib import Path
    from taco.localization.cvgl import CVGLLocalizer
    import cv2

    # Initialize localizer
    localizer = CVGLLocalizer(map_path=Path("visual_map.db"))

    # Load query image
    image = cv2.imread("query_image.jpg")

    # Camera intrinsics (example)
    K = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])

    # Localize
    measurement = localizer.localize(
        image,
        timestamp=1.0,
        camera_intrinsics=K
    )

    if measurement and measurement.confidence > 0.7:
        print(f"Localized at: {measurement.position}")
        print(f"Confidence: {measurement.confidence:.2f}")
        print(f"Inliers: {measurement.num_inliers}")

Complete SLAM Pipeline
-----------------------

Combining all components:

.. code-block:: python

    from pathlib import Path
    import numpy as np
    from taco.pose_graph import PoseGraph, PoseNode, Edge, EdgeType
    from taco.sensors.imu import IMUIntegrator
    from taco.localization.cvgl import CVGLLocalizer
    from taco.utils.io import load_imu_data
    from taco.visualization import plot_pose_graph

    # Initialize
    graph = PoseGraph()
    imu_integrator = IMUIntegrator(gravity=np.array([0.0, 0.0, -9.81]))
    cvgl_localizer = CVGLLocalizer(map_path=Path("map.db"))

    # Load data
    imu_data = load_imu_data(Path("imu.csv"))

    # Process IMU between keyframes
    keyframe_interval = 1.0
    last_keyframe_time = 0.0
    current_imu_buffer = []

    for imu in imu_data:
        current_imu_buffer.append(imu)

        if imu.timestamp - last_keyframe_time >= keyframe_interval:
            # Integrate IMU
            pos, vel, ori = imu_integrator.integrate(
                current_imu_buffer,
                initial_orientation=np.eye(3)
            )

            # Create node
            node = PoseNode(
                position=pos,
                orientation=ori,
                timestamp=imu.timestamp
            )
            node_id = graph.add_node(node)

            # Clear buffer
            current_imu_buffer = []
            last_keyframe_time = imu.timestamp

    # Visualize
    plot_pose_graph(graph, title="SLAM Trajectory")

Visualization
-------------

Plotting trajectories:

.. code-block:: python

    from taco.visualization import plot_trajectory, plot_pose_graph
    import numpy as np

    # Create trajectory
    t = np.linspace(0, 4 * np.pi, 100)
    positions = np.column_stack([
        np.cos(t),
        np.sin(t),
        t / 10
    ])

    # Plot
    plot_trajectory(positions, title="Spiral Trajectory")

    # Plot pose graph
    plot_pose_graph(graph, show_edges=True, title="Optimized Pose Graph")

Coordinate Transformations
---------------------------

Working with rotations and transformations:

.. code-block:: python

    from taco.utils.conversions import (
        quaternion_to_rotation_matrix,
        rotation_matrix_to_quaternion,
        pose_to_transform,
        transform_to_pose
    )
    import numpy as np

    # Quaternion to rotation matrix
    q = np.array([0.7071, 0.7071, 0.0, 0.0])  # 90° around x-axis
    R = quaternion_to_rotation_matrix(q)

    # Rotation matrix to quaternion
    q_recovered = rotation_matrix_to_quaternion(R)

    # Create transformation matrix
    position = np.array([1.0, 2.0, 3.0])
    T = pose_to_transform(position, q)

    # Extract pose from transform
    pos, quat = transform_to_pose(T)
