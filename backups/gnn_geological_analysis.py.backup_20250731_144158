"""
Example usage of Graph Neural Networks for geological analysis.

This script demonstrates how to use the geological GNN implementations
for various geological modeling tasks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging

# Import our GNN modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.graph_neural_networks import (
    GeologicalGCN, GeologicalGAT, GeologicalGraphSAGE,
    GeologicalGNNEnsemble, GeologicalGraphData
)
from core.geological_graph_utils import (
    GeologicalGraphBuilder, GeologicalGraphAnalyzer,
    create_volcanic_system_graph
)
from core.gnn_training import (
    GeologicalGNNTrainer, GeologicalGNNEvaluator,
    create_geological_gnn_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_volcanic_system():
    """
    Create a sample volcanic system for demonstration.
    """
    # Sample volcanoes
    volcanoes = [
        {
            'volcano_id': 'mount_st_helens',
            'latitude': 46.1914,
            'longitude': -122.1956,
            'elevation': 2549,
            'properties': {
                'last_eruption': 1980,
                'vei': 5,
                'rock_type': 1,  # andesite
                'tectonic_setting': 2  # subduction zone
            }
        },
        {
            'volcano_id': 'mount_rainier',
            'latitude': 46.8523,
            'longitude': -121.7603,
            'elevation': 4392,
            'properties': {
                'last_eruption': 1894,
                'vei': 2,
                'rock_type': 1,
                'tectonic_setting': 2
            }
        },
        {
            'volcano_id': 'mount_baker',
            'latitude': 48.7767,
            'longitude': -121.8144,
            'elevation': 3286,
            'properties': {
                'last_eruption': 1880,
                'vei': 2,
                'rock_type': 1,
                'tectonic_setting': 2
            }
        }
    ]
    
    # Sample seismic events
    seismic_events = [
        {
            'event_id': 1,
            'latitude': 46.2,
            'longitude': -122.2,
            'depth': 5.0,
            'magnitude': 3.2,
            'timestamp': 1640995200
        },
        {
            'event_id': 2,
            'latitude': 46.85,
            'longitude': -121.8,
            'depth': 8.0,
            'magnitude': 2.8,
            'timestamp': 1640995800
        },
        {
            'event_id': 3,
            'latitude': 48.8,
            'longitude': -121.8,
            'depth': 3.0,
            'magnitude': 2.1,
            'timestamp': 1641000000
        },
        {
            'event_id': 4,
            'latitude': 46.15,
            'longitude': -122.15,
            'depth': 12.0,
            'magnitude': 4.1,
            'timestamp': 1641002400
        }
    ]
    
    # Sample monitoring stations
    monitoring_stations = [
        {
            'station_id': 'PNSN_HSR',
            'latitude': 46.1,
            'longitude': -122.1,
            'elevation': 1200,
            'properties': {
                'installation_date': 1980,
                'sensor_type': 1,  # broadband seismometer
                'data_quality': 0.95
            }
        },
        {
            'station_id': 'PNSN_RCM',
            'latitude': 46.9,
            'longitude': -121.7,
            'elevation': 1800,
            'properties': {
                'installation_date': 1985,
                'sensor_type': 1,
                'data_quality': 0.92
            }
        },
        {
            'station_id': 'PNSN_MBW',
            'latitude': 48.8,
            'longitude': -121.9,
            'elevation': 1500,
            'properties': {
                'installation_date': 1990,
                'sensor_type': 2,  # short period
                'data_quality': 0.88
            }
        }
    ]
    
    return volcanoes, seismic_events, monitoring_stations


def demonstrate_graph_construction():
    """
    Demonstrate geological graph construction and analysis.
    """
    logger.info("=== Geological Graph Construction Demo ===")
    
    # Create sample data
    volcanoes, seismic_events, monitoring_stations = create_sample_volcanic_system()
    
    # Create volcanic system graph
    graph_data = create_volcanic_system_graph(
        volcanoes, seismic_events, monitoring_stations,
        max_spatial_distance=100.0
    )
    
    logger.info(f"Created graph with {graph_data.node_features.shape[0]} nodes")
    logger.info(f"Graph has {graph_data.edge_index.shape[1]} edges")
    
    # Analyze the graph
    analyzer = GeologicalGraphAnalyzer()
    
    structure_metrics = analyzer.analyze_graph_structure(graph_data)
    node_type_dist = analyzer.analyze_node_types(graph_data)
    spatial_stats = analyzer.calculate_spatial_statistics(graph_data)
    
    logger.info("Graph Structure Metrics:")
    for key, value in structure_metrics.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Node Type Distribution:")
    for key, value in node_type_dist.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Spatial Statistics:")
    for key, value in spatial_stats.items():
        logger.info(f"  {key}: {value:.4f}")
    
    return graph_data


def demonstrate_gnn_models(graph_data):
    """
    Demonstrate different GNN model architectures.
    """
    logger.info("\n=== GNN Model Architecture Demo ===")
    
    input_dim = graph_data.node_features.shape[1]
    hidden_dims = [128, 64]
    output_dim = 4  # 4 volcanic risk levels
    
    # Create different model types
    models = {
        'GCN': create_geological_gnn_model('gcn', input_dim, hidden_dims, output_dim),
        'GAT': create_geological_gnn_model('gat', input_dim, hidden_dims, output_dim, num_heads=[4, 2]),
        'GraphSAGE': create_geological_gnn_model('graphsage', input_dim, hidden_dims, output_dim),
        'Ensemble': create_geological_gnn_model('ensemble', input_dim, hidden_dims, output_dim)
    }
    
    # Test each model
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\nTesting {model_name} model:")
        
        model.eval()
        with torch.no_grad():
            if model_name == 'Ensemble':
                outputs = model(graph_data)
                predictions = outputs['ensemble_prediction']
                uncertainty = outputs['uncertainty']
                
                logger.info(f"  Prediction shape: {predictions.shape}")
                logger.info(f"  Uncertainty shape: {uncertainty.shape}")
                logger.info(f"  Mean uncertainty: {uncertainty.mean().item():.4f}")
                
                results[model_name] = {
                    'predictions': predictions,
                    'uncertainty': uncertainty,
                    'individual_predictions': outputs['individual_predictions']
                }
                
            elif model_name == 'GAT':
                predictions, attention_weights = model(graph_data)
                
                logger.info(f"  Prediction shape: {predictions.shape}")
                logger.info(f"  Attention layers: {list(attention_weights.keys())}")
                
                # Analyze relationship importance
                importance = model.get_relationship_importance(graph_data)
                logger.info("  Relationship importance:")
                for rel_type, importance_score in importance.items():
                    logger.info(f"    {rel_type}: {importance_score:.4f}")
                
                results[model_name] = {
                    'predictions': predictions,
                    'attention_weights': attention_weights,
                    'relationship_importance': importance
                }
                
            else:
                predictions = model(graph_data)
                logger.info(f"  Prediction shape: {predictions.shape}")
                
                results[model_name] = {'predictions': predictions}
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Total parameters: {num_params:,}")
    
    return results


def demonstrate_volcanic_risk_prediction(graph_data):
    """
    Demonstrate volcanic risk prediction using GNN ensemble.
    """
    logger.info("\n=== Volcanic Risk Prediction Demo ===")
    
    input_dim = graph_data.node_features.shape[1]
    output_dim = 4  # Risk levels: 0=Normal, 1=Advisory, 2=Watch, 3=Warning
    
    # Create ensemble model for robust predictions
    model = create_geological_gnn_model('ensemble', input_dim, [128, 64], output_dim)
    
    # Add dummy risk labels for demonstration
    # In practice, these would come from historical eruption data
    num_nodes = graph_data.node_features.shape[0]
    risk_labels = torch.randint(0, output_dim, (num_nodes,))
    graph_data.y = risk_labels
    
    # Create trainer
    trainer = GeologicalGNNTrainer(model, device='cpu', learning_rate=0.001)
    
    # Simple training demonstration (normally would use proper train/val split)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = [graph_data]  # Simplified for demo
    
    logger.info("Training volcanic risk prediction model...")
    
    # Train for a few epochs
    for epoch in range(5):
        loss, metrics = trainer.train_epoch(train_loader, criterion, 'node_classification')
        logger.info(f"  Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={metrics.get('accuracy', 0):.4f}")
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(graph_data)
        predictions = outputs['ensemble_prediction']
        uncertainty = outputs['uncertainty']
        
        # Convert to probabilities
        probabilities = torch.softmax(predictions, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Analyze predictions by node type
        risk_levels = ['Normal', 'Advisory', 'Watch', 'Warning']
        
        logger.info("\nVolcanic Risk Predictions:")
        for i in range(num_nodes):
            node_type = graph_data.node_types[i].item()
            predicted_risk = predicted_classes[i].item()
            confidence = probabilities[i, predicted_risk].item()
            uncertainty_score = uncertainty[i].mean().item()
            
            type_names = {0: 'Volcano', 1: 'Station', 2: 'Fault', 3: 'Chamber', 4: 'Event'}
            type_name = type_names.get(node_type, 'Unknown')
            
            logger.info(f"  Node {i} ({type_name}): {risk_levels[predicted_risk]} "
                       f"(confidence: {confidence:.3f}, uncertainty: {uncertainty_score:.3f})")
    
    return predictions, uncertainty


def demonstrate_seismic_pattern_analysis(graph_data):
    """
    Demonstrate seismic pattern analysis using GAT.
    """
    logger.info("\n=== Seismic Pattern Analysis Demo ===")
    
    input_dim = graph_data.node_features.shape[1]
    
    # Create GAT model for attention-based analysis
    model = create_geological_gnn_model('gat', input_dim, [128, 64], 32, num_heads=[8, 4])
    
    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model(graph_data)
        
        # Analyze attention patterns
        logger.info("Attention Analysis:")
        
        if 'final' in attention_weights:
            edge_index, att_weights = attention_weights['final']
            
            # Find most important connections
            top_k = 5
            top_indices = torch.topk(att_weights.squeeze(), top_k).indices
            
            logger.info(f"Top {top_k} most important geological connections:")
            for i, idx in enumerate(top_indices):
                source = edge_index[0, idx].item()
                target = edge_index[1, idx].item()
                weight = att_weights[idx].item()
                
                logger.info(f"  {i+1}. Node {source} -> Node {target}: {weight:.4f}")
        
        # Analyze relationship importance
        importance = model.get_relationship_importance(graph_data)
        logger.info("\nRelationship Type Importance:")
        for rel_type, score in importance.items():
            logger.info(f"  {rel_type}: {score:.4f}")
    
    return attention_weights


def demonstrate_multi_scale_analysis(graph_data):
    """
    Demonstrate multi-scale geological analysis using GraphSAGE.
    """
    logger.info("\n=== Multi-Scale Geological Analysis Demo ===")
    
    input_dim = graph_data.node_features.shape[1]
    
    # Create GraphSAGE model for multi-scale analysis
    model = create_geological_gnn_model('graphsage', input_dim, [128, 96, 64], 32)
    
    model.eval()
    with torch.no_grad():
        embeddings = model(graph_data)
        
        logger.info(f"Generated multi-scale embeddings: {embeddings.shape}")
        
        # Analyze embedding clusters (simplified clustering)
        from sklearn.cluster import KMeans
        
        # Convert to numpy for sklearn
        embeddings_np = embeddings.cpu().numpy()
        
        # Cluster nodes based on embeddings
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_np)
        
        logger.info(f"\nIdentified {n_clusters} geological clusters:")
        
        # Analyze cluster composition
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            cluster_nodes = np.where(cluster_mask)[0]
            
            # Get node types in this cluster
            cluster_node_types = graph_data.node_types[cluster_nodes].numpy()
            unique_types, counts = np.unique(cluster_node_types, return_counts=True)
            
            type_names = {0: 'Volcano', 1: 'Station', 2: 'Fault', 3: 'Chamber', 4: 'Event'}
            
            logger.info(f"  Cluster {cluster_id}: {len(cluster_nodes)} nodes")
            for node_type, count in zip(unique_types, counts):
                type_name = type_names.get(node_type, 'Unknown')
                logger.info(f"    {type_name}: {count}")
    
    return embeddings, clusters


def save_analysis_results(results, output_dir="gnn_analysis_results"):
    """
    Save analysis results to files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save results as JSON (convert tensors to lists)
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            serializable_results[key] = value.cpu().numpy().tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    serializable_results[key][subkey] = subvalue.cpu().numpy().tolist()
                else:
                    serializable_results[key][subkey] = subvalue
        else:
            serializable_results[key] = value
    
    results_file = output_path / "gnn_analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Analysis results saved to {results_file}")


def main():
    """
    Main demonstration function.
    """
    logger.info("Starting Geological Graph Neural Network Analysis Demo")
    
    try:
        # 1. Graph Construction
        graph_data = demonstrate_graph_construction()
        
        # 2. Model Architecture Comparison
        model_results = demonstrate_gnn_models(graph_data)
        
        # 3. Volcanic Risk Prediction
        risk_predictions, risk_uncertainty = demonstrate_volcanic_risk_prediction(graph_data)
        
        # 4. Seismic Pattern Analysis
        attention_analysis = demonstrate_seismic_pattern_analysis(graph_data)
        
        # 5. Multi-Scale Analysis
        embeddings, clusters = demonstrate_multi_scale_analysis(graph_data)
        
        # Compile results
        all_results = {
            'model_comparisons': model_results,
            'risk_predictions': risk_predictions,
            'risk_uncertainty': risk_uncertainty,
            'attention_analysis': attention_analysis,
            'multi_scale_embeddings': embeddings,
            'geological_clusters': clusters.tolist()
        }
        
        # Save results
        save_analysis_results(all_results)
        
        logger.info("\n=== Demo Completed Successfully ===")
        logger.info("Geological Graph Neural Networks demonstrated:")
        logger.info("✓ Graph construction from geological data")
        logger.info("✓ GCN, GAT, GraphSAGE, and Ensemble models")
        logger.info("✓ Volcanic risk prediction with uncertainty")
        logger.info("✓ Attention-based relationship analysis")
        logger.info("✓ Multi-scale geological clustering")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()