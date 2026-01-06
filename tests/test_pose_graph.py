"""Tests for pose graph functionality."""

import numpy as np
import pytest

from taco.pose_graph import Edge, EdgeType, PoseGraph, PoseNode


class TestPoseNode:
    """Test PoseNode class."""

    def test_node_creation_with_quaternion(self) -> None:
        """Test creating a node with quaternion orientation."""
        position = np.array([1.0, 2.0, 3.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        timestamp = 0.0

        node = PoseNode(position=position, orientation=orientation, timestamp=timestamp)

        assert np.allclose(node.position, position)
        assert np.allclose(node.orientation, orientation)
        assert node.timestamp == timestamp

    def test_node_creation_with_rotation_matrix(self) -> None:
        """Test creating a node with rotation matrix orientation."""
        position = np.array([1.0, 2.0, 3.0])
        orientation = np.eye(3)  # Identity rotation matrix
        timestamp = 0.0

        node = PoseNode(position=position, orientation=orientation, timestamp=timestamp)

        assert np.allclose(node.position, position)
        assert np.allclose(node.orientation, orientation)

    def test_invalid_position_shape(self) -> None:
        """Test that invalid position shape raises error."""
        position = np.array([1.0, 2.0])  # Wrong shape
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Position must be a 3D vector"):
            PoseNode(position=position, orientation=orientation, timestamp=0.0)


class TestEdge:
    """Test Edge class."""

    def test_edge_creation(self) -> None:
        """Test creating an edge."""
        transform = np.eye(4)
        information = np.eye(6)

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
        transform = np.eye(3)  # Wrong shape
        information = np.eye(6)

        with pytest.raises(ValueError, match="Relative transform must be a 4x4 matrix"):
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
            position=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
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
                position=np.array([i, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
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
                position=np.array([i, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=float(i),
            )
            graph.add_node(node)

        # Add edge
        edge = Edge(
            from_node_id=0,
            to_node_id=1,
            relative_transform=np.eye(4),
            information_matrix=np.eye(6),
            edge_type=EdgeType.IMU,
        )
        graph.add_edge(edge)

        assert len(graph.edges) == 1
        assert graph.edges[0] == edge

    def test_get_nonexistent_node(self) -> None:
        """Test getting a node that doesn't exist."""
        graph = PoseGraph()
        assert graph.get_node(999) is None
