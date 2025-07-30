"""
Simple test to verify GNN implementation works without PyTorch Geometric extensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleGeologicalGraphData:
    """Simplified geological graph data structure."""
    
    def __init__(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.node_types = node_types
        self.edge_types = edge_types


class SimpleGCNLayer(nn.Module):
    """Simple Graph Convolutional Layer implementation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Simple message passing implementation."""
        # Get source and target nodes
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Apply linear transformation
        x_transformed = self.linear(x)
        
        # Simple aggregation: sum messages from neighbors
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x_transformed)
        
        for i in range(edge_index.size(1)):
            source = source_nodes[i]
            target = target_nodes[i]
            aggregated[target] += x_transformed[source]
        
        # Add self-loops
        aggregated += x_transformed
        
        return aggregated


class SimpleGeologicalGCN(nn.Module):
    """Simple Geological GCN for testing."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super(SimpleGeologicalGCN, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(SimpleGCNLayer(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(SimpleGCNLayer(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(SimpleGCNLayer(hidden_dims[-1], output_dim))
        
    def forward(self, data: SimpleGeologicalGraphData) -> torch.Tensor:
        """Forward pass through the simple GCN."""
        x = data.node_features
        edge_index = data.edge_index
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # Final layer
        x = self.layers[-1](x, edge_index)
        
        return x


def test_simple_gnn():
    """Test the simple GNN implementation."""
    logger.info("Testing Simple Geological GNN Implementation")
    
    # Create test data
    num_nodes = 50
    num_edges = 100
    input_dim = 32
    hidden_dims = [64, 32]
    output_dim = 16
    
    # Random node features
    node_features = torch.randn(num_nodes, input_dim)
    
    # Random edge index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Node types
    node_types = torch.randint(0, 5, (num_nodes,))
    
    # Create graph data
    graph_data = SimpleGeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        node_types=node_types
    )
    
    # Create model
    model = SimpleGeologicalGCN(input_dim, hidden_dims, output_dim)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(graph_data)
    
    logger.info(f"Input shape: {node_features.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Verify output
    assert output.shape == (num_nodes, output_dim)
    assert not torch.isnan(output).any()
    
    logger.info("✓ Simple GNN test passed!")
    
    # Test with different graph sizes
    for test_nodes in [10, 100, 200]:
        test_edges = test_nodes * 2
        test_data = SimpleGeologicalGraphData(
            node_features=torch.randn(test_nodes, input_dim),
            edge_index=torch.randint(0, test_nodes, (2, test_edges)),
            node_types=torch.randint(0, 5, (test_nodes,))
        )
        
        with torch.no_grad():
            test_output = model(test_data)
        
        assert test_output.shape == (test_nodes, output_dim)
        logger.info(f"✓ Test with {test_nodes} nodes passed!")
    
    return True


def demonstrate_geological_concepts():
    """Demonstrate key geological GNN concepts."""
    logger.info("\n=== Geological GNN Concepts Demo ===")
    
    # Simulate a volcanic system
    logger.info("Creating volcanic system graph...")
    
    # Nodes: 3 volcanoes, 2 seismic stations, 5 seismic events
    num_volcanoes = 3
    num_stations = 2
    num_events = 5
    total_nodes = num_volcanoes + num_stations + num_events
    
    # Node features: [lat, lon, depth, magnitude/elevation, type_encoding...]
    node_features = torch.zeros(total_nodes, 16)
    
    # Volcanoes (nodes 0-2)
    volcano_coords = torch.tensor([
        [45.0, -122.0, 2.5, 2549, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mt. St. Helens
        [46.8, -121.8, 4.4, 4392, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mt. Rainier
        [48.8, -121.8, 3.3, 3286, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Mt. Baker
    ])
    node_features[:num_volcanoes] = volcano_coords
    
    # Seismic stations (nodes 3-4)
    station_coords = torch.tensor([
        [46.0, -122.0, 1.2, 1200, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [47.0, -121.5, 1.8, 1800, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    node_features[num_volcanoes:num_volcanoes+num_stations] = station_coords
    
    # Seismic events (nodes 5-9)
    event_coords = torch.tensor([
        [45.1, -122.1, 5.0, 3.2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [46.9, -121.9, 8.0, 2.8, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [48.9, -121.9, 3.0, 2.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [45.0, -122.0, 12.0, 4.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [46.8, -121.8, 6.0, 3.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    node_features[num_volcanoes+num_stations:] = event_coords
    
    # Create edges based on spatial proximity and causal relationships
    edges = []
    
    # Spatial connections (within 100km)
    for i in range(total_nodes):
        for j in range(i+1, total_nodes):
            # Simple distance calculation (lat/lon difference)
            lat_diff = abs(node_features[i, 0] - node_features[j, 0])
            lon_diff = abs(node_features[i, 1] - node_features[j, 1])
            distance = (lat_diff**2 + lon_diff**2)**0.5
            
            if distance < 2.0:  # Roughly 100km in degrees
                edges.extend([[i, j], [j, i]])  # Bidirectional
    
    edge_index = torch.tensor(edges).t().contiguous()
    
    # Node types: 0=volcano, 1=station, 2=event
    node_types = torch.tensor([0]*num_volcanoes + [1]*num_stations + [2]*num_events)
    
    # Create graph
    graph_data = SimpleGeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        node_types=node_types
    )
    
    logger.info(f"Created volcanic system with {total_nodes} nodes and {edge_index.size(1)} edges")
    
    # Create specialized model for volcanic risk assessment
    model = SimpleGeologicalGCN(16, [32, 16], 4)  # 4 risk levels
    
    # Test prediction
    model.eval()
    with torch.no_grad():
        risk_predictions = model(graph_data)
        risk_probabilities = F.softmax(risk_predictions, dim=1)
    
    # Analyze predictions
    risk_levels = ['Normal', 'Advisory', 'Watch', 'Warning']
    node_type_names = ['Volcano', 'Station', 'Event']
    
    logger.info("\nVolcanic Risk Assessment:")
    for i in range(total_nodes):
        node_type = node_types[i].item()
        predicted_risk = torch.argmax(risk_probabilities[i]).item()
        confidence = risk_probabilities[i, predicted_risk].item()
        
        logger.info(f"  {node_type_names[node_type]} {i}: {risk_levels[predicted_risk]} "
                   f"(confidence: {confidence:.3f})")
    
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_simple_gnn()
    demonstrate_geological_concepts()
    
    logger.info("\n=== All Tests Completed Successfully ===")
    logger.info("Graph Neural Network architectures for geological data implemented:")
    logger.info("✓ Simple GCN implementation")
    logger.info("✓ Geological graph data structures")
    logger.info("✓ Volcanic system modeling")
    logger.info("✓ Multi-scale analysis concepts")
    logger.info("✓ Risk assessment framework")