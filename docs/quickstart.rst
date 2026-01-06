Quickstart
==========

This guide will help you get started with TACO.

Creating a Pose Graph
----------------------

.. code-block:: python

    import numpy as np
    from taco.pose_graph import PoseGraph, PoseNode

    # Create a new pose graph
    graph = PoseGraph()

    # Add a pose node
    node = PoseNode(
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        timestamp=0.0
    )

    node_id = graph.add_node(node)
    print(f"Added node with ID: {node_id}")

Processing IMU Data
-------------------

.. code-block:: python

    from taco.sensors.imu import IMUData

    # Create an IMU measurement
    imu = IMUData.from_raw(
        timestamp=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=-9.81,  # Gravity
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.1  # Small rotation
    )

    print(f"Acceleration: {imu.linear_acceleration}")
    print(f"Angular velocity: {imu.angular_velocity}")

Adding Edges
------------

.. code-block:: python

    from taco.pose_graph import Edge, EdgeType

    # Create an edge between two nodes
    edge = Edge(
        from_node_id=0,
        to_node_id=1,
        relative_transform=np.eye(4),
        information_matrix=np.eye(6) * 10.0,  # Higher = more confident
        edge_type=EdgeType.IMU
    )

    graph.add_edge(edge)

Visualization
-------------

.. code-block:: python

    from taco.visualization import plot_pose_graph

    # Plot the pose graph
    plot_pose_graph(graph, show_edges=True)

Next Steps
----------

- Check out the :doc:`examples` page for more detailed examples
- Read the :doc:`api` reference for complete documentation
- See :doc:`contributing` if you want to contribute
