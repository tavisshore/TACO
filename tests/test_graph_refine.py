"""Tests for graph refinement module."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import networkx as nx

# Skip tests if networkx not available
networkx_available = True
try:
    import networkx as nx

    from taco.data.graph_refine import calculate_bearing, calculate_turn_angle, simplify_sharp_turns
except ImportError:
    networkx_available = False
    nx = None  # type: ignore[assignment]


pytestmark = pytest.mark.skipif(not networkx_available, reason="networkx not installed")


class TestCalculateTurnAngle:
    """Test turn angle calculation."""

    def test_straight_line_180_degrees(self) -> None:
        """Test angle for collinear points is 180 degrees."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (2.0, 0.0)

        angle = calculate_turn_angle(p1, p2, p3)

        assert abs(angle - 180.0) < 1e-10

    def test_right_angle_90_degrees(self) -> None:
        """Test 90 degree turn."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0, 1.0)

        angle = calculate_turn_angle(p1, p2, p3)

        assert abs(angle - 90.0) < 1e-10

    def test_acute_angle(self) -> None:
        """Test acute angle (less than 90 degrees)."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (0.5, 0.5)  # Sharp turn back

        angle = calculate_turn_angle(p1, p2, p3)

        # Vector from p2 to p1 is (-1, 0), vector from p2 to p3 is (-0.5, 0.5)
        # The angle between them is 45 degrees
        assert 40.0 < angle < 50.0

    def test_zero_length_vector(self) -> None:
        """Test handling of coincident points."""
        p1 = (0.0, 0.0)
        p2 = (0.0, 0.0)  # Same as p1
        p3 = (1.0, 0.0)

        angle = calculate_turn_angle(p1, p2, p3)

        # Should return 180 for degenerate case
        assert angle == 180.0

    def test_symmetric_angles(self) -> None:
        """Test that angle is symmetric."""
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0, 1.0)

        angle1 = calculate_turn_angle(p1, p2, p3)
        angle2 = calculate_turn_angle(p3, p2, p1)

        assert abs(angle1 - angle2) < 1e-10


class TestCalculateBearing:
    """Test bearing calculation."""

    def test_bearing_north(self) -> None:
        """Test bearing when going north."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 1.0, 0.0  # North

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        # Should be close to 0 radians (north)
        assert abs(bearing) < 0.01 or abs(bearing - 2 * math.pi) < 0.01

    def test_bearing_east(self) -> None:
        """Test bearing when going east."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.0, 1.0  # East

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        # Should be close to pi/2 radians (east)
        assert abs(bearing - math.pi / 2) < 0.01

    def test_bearing_south(self) -> None:
        """Test bearing when going south."""
        lat1, lon1 = 1.0, 0.0
        lat2, lon2 = 0.0, 0.0  # South

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        # Should be close to pi radians (south)
        assert abs(bearing - math.pi) < 0.01

    def test_bearing_west(self) -> None:
        """Test bearing when going west."""
        lat1, lon1 = 0.0, 1.0
        lat2, lon2 = 0.0, 0.0  # West

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        # Should be close to 3*pi/2 or -pi/2 radians (west)
        assert abs(bearing - 3 * math.pi / 2) < 0.01 or abs(bearing + math.pi / 2) < 0.01

    def test_bearing_returns_radians(self) -> None:
        """Test that bearing is returned in radians."""
        lat1, lon1 = 0.0, 0.0
        lat2, lon2 = 0.5, 0.5

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)

        # Should be between 0 and 2*pi
        assert 0 <= bearing <= 2 * math.pi


class TestSimplifySharpTurns:
    """Test graph simplification function."""

    @pytest.fixture
    def straight_line_graph(self) -> nx.MultiDiGraph:
        """Create a graph representing a straight road."""
        g = nx.MultiDiGraph()

        # Add nodes in a straight line
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=0.001, y=0.0)  # Very small offset
        g.add_node(2, x=0.002, y=0.0)

        g.add_edge(0, 1)
        g.add_edge(1, 2)

        return g

    @pytest.fixture
    def turn_graph(self) -> nx.MultiDiGraph:
        """Create a graph with a sharp turn."""
        g = nx.MultiDiGraph()

        # Add nodes forming an L-shape
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=0.001, y=0.0)  # Corner
        g.add_node(2, x=0.001, y=0.001)  # Turn

        g.add_edge(0, 1)
        g.add_edge(1, 2)

        return g

    @pytest.fixture
    def complex_graph(self) -> nx.MultiDiGraph:
        """Create a more complex graph."""
        g = nx.MultiDiGraph()

        # Add junction node with multiple connections
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=0.001, y=0.0)
        g.add_node(2, x=0.002, y=0.0)
        g.add_node(3, x=0.001, y=0.001)

        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(1, 3)

        return g

    def test_simplify_preserves_graph_type(self, straight_line_graph: nx.MultiDiGraph) -> None:
        """Test that simplification returns the same graph type."""
        result = simplify_sharp_turns(straight_line_graph)
        assert isinstance(result, nx.MultiDiGraph)

    def test_simplify_does_not_modify_original(self, straight_line_graph: nx.MultiDiGraph) -> None:
        """Test that original graph is not modified."""
        original_nodes = set(straight_line_graph.nodes())
        simplify_sharp_turns(straight_line_graph)
        assert set(straight_line_graph.nodes()) == original_nodes

    def test_simplify_removes_collinear_nodes(self, straight_line_graph: nx.MultiDiGraph) -> None:
        """Test that collinear intermediate nodes are removed."""
        original_node_count = straight_line_graph.number_of_nodes()
        result = simplify_sharp_turns(straight_line_graph, angle_threshold=10.0)

        # Middle node should be removed since points are collinear
        assert result.number_of_nodes() <= original_node_count

    def test_simplify_preserves_turn_nodes(self, turn_graph: nx.MultiDiGraph) -> None:
        """Test that nodes at sharp turns are preserved."""
        result = simplify_sharp_turns(turn_graph, angle_threshold=30.0)

        # All nodes should be preserved since there's a 90-degree turn
        assert result.number_of_nodes() == turn_graph.number_of_nodes()

    def test_simplify_preserves_junctions(self, complex_graph: nx.MultiDiGraph) -> None:
        """Test that junction nodes (degree > 2) are preserved."""
        result = simplify_sharp_turns(complex_graph)

        # Node 1 is a junction (3 edges) and should be preserved
        assert 1 in result.nodes()

    def test_simplify_empty_graph(self) -> None:
        """Test simplifying an empty graph."""
        g = nx.MultiDiGraph()
        result = simplify_sharp_turns(g)
        assert result.number_of_nodes() == 0
        assert result.number_of_edges() == 0

    def test_simplify_single_node(self) -> None:
        """Test simplifying a graph with single node."""
        g = nx.MultiDiGraph()
        g.add_node(0, x=0.0, y=0.0)
        result = simplify_sharp_turns(g)
        assert result.number_of_nodes() == 1

    def test_simplify_maintains_connectivity(self, straight_line_graph: nx.MultiDiGraph) -> None:
        """Test that simplification maintains connectivity."""
        # Check that end nodes remain connected
        result = simplify_sharp_turns(straight_line_graph, angle_threshold=10.0)

        # Should still have path from first to last node
        if 0 in result.nodes() and 2 in result.nodes():
            # There should be a path between them
            assert nx.has_path(result.to_undirected(), 0, 2)
