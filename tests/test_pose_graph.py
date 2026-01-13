"""Tests for pose graph functionality with Pose2."""

import numpy as np
import pytest

from taco.pose_graph import Edge, EdgeType, PoseGraph, PoseNode


class TestPoseNode:
    """Test PoseNode class."""

    def test_node_creation_with_yaw(self) -> None:
        """Test creating a node with 2D position and yaw."""
        position = np.array([1.0, 2.0])
        yaw = 0.5  # radians
        timestamp = 0.0

        node = PoseNode(position=position, yaw=yaw, timestamp=timestamp)

        assert np.allclose(node.position, position)
        assert node.yaw == yaw
        assert node.timestamp == timestamp

    def test_node_creation_with_covariance(self) -> None:
        """Test creating a node with covariance."""
        position = np.array([1.0, 2.0])
        yaw = 0.0
        timestamp = 0.0
        covariance = np.eye(3) * 0.1

        node = PoseNode(position=position, yaw=yaw, timestamp=timestamp, covariance=covariance)

        assert np.allclose(node.covariance, covariance)

    def test_invalid_position_shape(self) -> None:
        """Test that invalid position shape raises error."""
        position = np.array([1.0, 2.0, 3.0])  # Wrong shape - should be 2D
        yaw = 0.0

        with pytest.raises(ValueError, match="Position must be a 2D vector"):
            PoseNode(position=position, yaw=yaw, timestamp=0.0)

    def test_invalid_covariance_shape(self) -> None:
        """Test that invalid covariance shape raises error."""
        position = np.array([1.0, 2.0])
        yaw = 0.0
        covariance = np.eye(6)  # Wrong shape - should be 3x3

        with pytest.raises(ValueError, match="Covariance must be a 3x3 matrix"):
            PoseNode(position=position, yaw=yaw, timestamp=0.0, covariance=covariance)


class TestEdge:
    """Test Edge class."""

    def test_edge_creation(self) -> None:
        """Test creating an edge."""
        transform = np.eye(3)  # 3x3 SE(2) identity
        information = np.eye(3)

        edge = Edge(
            from_node_id=0,
            to_node_id=1,
            relative_transform=transform,
            information_matrix=information,
            edge_type=EdgeType.IMU,
        )

        assert edge.from_node_id == 0
        assert edge.to_node_id == 1
        assert edge.edge_type == EdgeType.IMU

    def test_invalid_transform_shape(self) -> None:
        """Test that invalid transform shape raises error."""
        transform = np.eye(4)  # Wrong shape - should be 3x3
        information = np.eye(3)

        with pytest.raises(ValueError, match="Relative transform must be a 3x3 SE\\(2\\) matrix"):
            Edge(
                from_node_id=0,
                to_node_id=1,
                relative_transform=transform,
                information_matrix=information,
                edge_type=EdgeType.IMU,
            )

    def test_invalid_information_shape(self) -> None:
        """Test that invalid information matrix shape raises error."""
        transform = np.eye(3)
        information = np.eye(6)  # Wrong shape - should be 3x3

        with pytest.raises(ValueError, match="Information matrix must be 3x3"):
            Edge(
                from_node_id=0,
                to_node_id=1,
                relative_transform=transform,
                information_matrix=information,
                edge_type=EdgeType.IMU,
            )


class TestPoseGraph:
    """Test PoseGraph class."""

    def test_add_node(self) -> None:
        """Test adding nodes to graph."""
        graph = PoseGraph()
        node = PoseNode(
            position=np.zeros(2),
            yaw=0.0,
            timestamp=0.0,
        )

        node_id = graph.add_node(node)

        assert node_id == 0
        assert len(graph.nodes) == 1
        assert graph.get_node(node_id) == node

    def test_add_multiple_nodes(self) -> None:
        """Test adding multiple nodes."""
        graph = PoseGraph()

        for i in range(5):
            node = PoseNode(
                position=np.array([float(i), 0.0]),
                yaw=0.0,
                timestamp=float(i),
            )
            node_id = graph.add_node(node)
            assert node_id == i

        assert len(graph.nodes) == 5

    def test_add_edge(self) -> None:
        """Test adding edges to graph."""
        graph = PoseGraph()

        # Add two nodes
        for i in range(2):
            node = PoseNode(
                position=np.array([float(i), 0.0]),
                yaw=0.0,
                timestamp=float(i),
            )
            graph.add_node(node)

        # Add edge with 3x3 SE(2) transform
        edge = Edge(
            from_node_id=0,
            to_node_id=1,
            relative_transform=np.eye(3),
            information_matrix=np.eye(3),
            edge_type=EdgeType.IMU,
        )
        graph.add_edge(edge)

        assert len(graph.edges) == 1
        assert graph.edges[0] == edge

    def test_get_nonexistent_node(self) -> None:
        """Test getting a node that doesn't exist."""
        graph = PoseGraph()
        assert graph.get_node(999) is None
