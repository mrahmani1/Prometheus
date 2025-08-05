"""
Training and evaluation module for geological graph neural networks.

This module provides training loops, evaluation metrics, and model management
specifically designed for geological GNN applications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
from pathlib import Path
import json
from datetime import datetime

from .graph_neural_networks import (
    GeologicalGCN, GeologicalGAT, GeologicalGraphSAGE, 
    GeologicalGNNEnsemble, GeologicalGraphData
)
from .geological_graph_utils import GeologicalGraphPreprocessor

logger = logging.getLogger(__name__)


class GeologicalGNNTrainer:
    """
    Trainer class for geological graph neural networks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Physics-informed loss components
        self.physics_loss_weight = 0.1
        
    def physics_informed_loss(self, predictions: torch.Tensor, data: GeologicalGraphData) -> torch.Tensor:
        """
        Calculate physics-informed loss for geological constraints.
        """
        physics_loss = torch.tensor(0.0, device=self.device)
        
        # Energy conservation constraint
        if hasattr(data, 'energy_values'):
            energy_pred = predictions[:, :3]  # Assume first 3 dims are energy-related
            energy_conservation = torch.sum(energy_pred, dim=1) - data.energy_values
            physics_loss += torch.mean(energy_conservation ** 2)
        
        # Mass conservation constraint
        if hasattr(data, 'mass_values'):
            mass_pred = predictions[:, 3:6]  # Next 3 dims for mass-related
            mass_conservation = torch.sum(mass_pred, dim=1) - data.mass_values
            physics_loss += torch.mean(mass_conservation ** 2)
        
        # Spatial continuity constraint
        if data.spatial_coords is not None:
            # Penalize large gradients in predictions for nearby nodes
            coords = data.spatial_coords
            distances = torch.cdist(coords, coords)
            nearby_mask = (distances < 10.0) & (distances > 0)  # Within 10km
            
            if nearby_mask.any():
                pred_diff = torch.cdist(predictions, predictions)
                spatial_penalty = torch.mean(pred_diff[nearby_mask] / (distances[nearby_mask] + 1e-6))
                physics_loss += spatial_penalty
        
        return physics_loss    

    def train_epoch(
        self, 
        train_loader: DataLoader, 
        criterion: nn.Module,
        task_type: str = 'node_classification'
    ) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            if isinstance(self.model, GeologicalGNNEnsemble):
                outputs = self.model(batch)
                predictions = outputs['ensemble_prediction']
            else:
                predictions = self.model(batch)
            
            # Calculate main loss
            if task_type == 'node_classification':
                targets = batch.y
                loss = criterion(predictions, targets)
            elif task_type == 'graph_classification':
                targets = batch.graph_y
                # Global pooling for graph-level prediction
                from torch_geometric.nn import global_mean_pool
                graph_pred = global_mean_pool(predictions, batch.batch)
                loss = criterion(graph_pred, targets)
            elif task_type == 'regression':
                targets = batch.y
                loss = criterion(predictions, targets)
            
            # Add physics-informed loss
            physics_loss = self.physics_informed_loss(predictions, batch)
            total_loss_value = loss + self.physics_loss_weight * physics_loss
            
            # Backward pass
            total_loss_value.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_loss_value.item()
            
            # Store predictions and targets for metrics
            if task_type == 'node_classification':
                all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            elif task_type == 'graph_classification':
                all_predictions.extend(torch.argmax(graph_pred, dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        
        # Calculate metrics
        metrics = {}
        if task_type in ['node_classification', 'graph_classification']:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        return avg_loss, metrics
    
    def validate_epoch(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module,
        task_type: str = 'node_classification'
    ) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(self.model, GeologicalGNNEnsemble):
                    outputs = self.model(batch)
                    predictions = outputs['ensemble_prediction']
                else:
                    predictions = self.model(batch)
                
                # Calculate loss
                if task_type == 'node_classification':
                    targets = batch.y
                    loss = criterion(predictions, targets)
                elif task_type == 'graph_classification':
                    targets = batch.graph_y
                    from torch_geometric.nn import global_mean_pool
                    graph_pred = global_mean_pool(predictions, batch.batch)
                    loss = criterion(graph_pred, targets)
                elif task_type == 'regression':
                    targets = batch.y
                    loss = criterion(predictions, targets)
                
                # Add physics loss
                physics_loss = self.physics_informed_loss(predictions, batch)
                total_loss_value = loss + self.physics_loss_weight * physics_loss
                
                total_loss += total_loss_value.item()
                
                # Store predictions and targets
                if task_type == 'node_classification':
                    all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                elif task_type == 'graph_classification':
                    all_predictions.extend(torch.argmax(graph_pred, dim=1).cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {}
        if task_type in ['node_classification', 'graph_classification']:
            metrics['accuracy'] = accuracy_score(all_targets, all_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted'
            )
            metrics['precision'] = precision
            metrics['recall'] = recall
            metrics['f1'] = f1
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        criterion: nn.Module,
        task_type: str = 'node_classification',
        early_stopping_patience: int = 20,
        save_path: Optional[str] = None
    ) -> Dict[str, List]:
        """
        Full training loop with validation and early stopping.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, criterion, task_type)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader, criterion, task_type)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'val_metrics': val_metrics
                    }, save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                if 'accuracy' in val_metrics:
                    logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }


class GeologicalGNNEvaluator:
    """
    Evaluator for geological GNN models with domain-specific metrics.
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
    def evaluate_volcanic_risk_prediction(
        self, 
        test_loader: DataLoader,
        risk_thresholds: List[float] = [0.3, 0.5, 0.7]
    ) -> Dict[str, float]:
        """
        Evaluate volcanic risk prediction performance.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                if isinstance(self.model, GeologicalGNNEnsemble):
                    outputs = self.model(batch)
                    predictions = outputs['ensemble_prediction']
                else:
                    predictions = self.model(batch)
                
                # Convert to probabilities
                probabilities = torch.softmax(predictions, dim=1)
                predicted_classes = torch.argmax(predictions, dim=1)
                
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics for different risk thresholds
        results = {}
        all_probabilities = np.array(all_probabilities)
        
        for threshold in risk_thresholds:
            # Binary classification at threshold
            binary_pred = (all_probabilities[:, 1] > threshold).astype(int)  # Assuming class 1 is high risk
            binary_target = (np.array(all_targets) > 0).astype(int)
            
            accuracy = accuracy_score(binary_target, binary_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                binary_target, binary_pred, average='binary'
            )
            
            results[f'threshold_{threshold}'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Overall metrics
        overall_accuracy = accuracy_score(all_targets, all_predictions)
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        results['overall'] = {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }
        
        # AUC if binary classification
        if len(np.unique(all_targets)) == 2:
            auc = roc_auc_score(all_targets, all_probabilities[:, 1])
            results['overall']['auc'] = auc
        
        return results
    
    def evaluate_seismic_event_detection(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate seismic event detection performance.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        detection_latencies = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                
                if isinstance(self.model, GeologicalGNNEnsemble):
                    outputs = self.model(batch)
                    predictions = outputs['ensemble_prediction']
                else:
                    predictions = self.model(batch)
                
                end_time.record()
                torch.cuda.synchronize()
                
                # Calculate detection latency
                latency = start_time.elapsed_time(end_time)  # milliseconds
                detection_latencies.append(latency)
                
                predicted_classes = torch.argmax(predictions, dim=1)
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        # Calculate detection metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted'
        )
        
        # Detection performance metrics
        avg_latency = np.mean(detection_latencies)
        detection_rate = recall  # Recall is the detection rate
        false_alarm_rate = 1 - precision  # Simplified false alarm rate
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'avg_detection_latency_ms': avg_latency
        }
    
    def evaluate_geological_relationship_discovery(
        self, 
        test_loader: DataLoader,
        ground_truth_relationships: Dict[Tuple[int, int], str]
    ) -> Dict[str, float]:
        """
        Evaluate geological relationship discovery accuracy.
        """
        self.model.eval()
        discovered_relationships = {}
        relationship_confidences = {}
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                if isinstance(self.model, GeologicalGAT):
                    predictions, attention_weights = self.model(batch)
                    
                    # Extract relationship strengths from attention weights
                    if 'final' in attention_weights:
                        edge_index, att_weights = attention_weights['final']
                        
                        for i in range(edge_index.size(1)):
                            source = edge_index[0, i].item()
                            target = edge_index[1, i].item()
                            confidence = att_weights[i].item()
                            
                            relationship_key = (source, target)
                            if relationship_key not in relationship_confidences:
                                relationship_confidences[relationship_key] = []
                            relationship_confidences[relationship_key].append(confidence)
        
        # Determine discovered relationships based on confidence threshold
        confidence_threshold = 0.5
        for rel_key, confidences in relationship_confidences.items():
            avg_confidence = np.mean(confidences)
            if avg_confidence > confidence_threshold:
                discovered_relationships[rel_key] = 'discovered'
        
        # Calculate discovery metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for rel_key in ground_truth_relationships:
            if rel_key in discovered_relationships:
                true_positives += 1
            else:
                false_negatives += 1
        
        for rel_key in discovered_relationships:
            if rel_key not in ground_truth_relationships:
                false_positives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'relationship_discovery_precision': precision,
            'relationship_discovery_recall': recall,
            'relationship_discovery_f1': f1,
            'total_discovered_relationships': len(discovered_relationships),
            'total_ground_truth_relationships': len(ground_truth_relationships)
        }


def create_geological_gnn_model(
    model_type: str,
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create geological GNN models.
    """
    if model_type == 'gcn':
        return GeologicalGCN(input_dim, hidden_dims, output_dim, **kwargs)
    elif model_type == 'gat':
        num_heads = kwargs.get('num_heads', [4] * len(hidden_dims))
        return GeologicalGAT(input_dim, hidden_dims, output_dim, num_heads, **kwargs)
    elif model_type == 'graphsage':
        return GeologicalGraphSAGE(input_dim, hidden_dims, output_dim, **kwargs)
    elif model_type == 'ensemble':
        return GeologicalGNNEnsemble(input_dim, hidden_dims, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_training_results(
    results: Dict,
    model_config: Dict,
    save_dir: str
):
    """
    Save training results and model configuration.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training results
    results_file = save_path / f"training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                serializable_results[key] = value
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value
        json.dump(serializable_results, f, indent=2)
    
    # Save model configuration
    config_file = save_path / f"model_config_{timestamp}.json"
    with open(config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    logger.info(f"Training results saved to {results_file}")
    logger.info(f"Model configuration saved to {config_file}")


# Example usage and testing
def test_gnn_training():
    """Test GNN training functionality."""
    # Create dummy data
    num_nodes = 100
    input_dim = 64
    output_dim = 3  # 3 risk levels
    
    # Create sample graph data
    from .graph_neural_networks import GeologicalGraphData
    
    node_features = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 200))
    node_types = torch.randint(0, 5, (num_nodes,))
    targets = torch.randint(0, output_dim, (num_nodes,))
    
    graph_data = GeologicalGraphData(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=None,
        node_types=node_types,
        edge_types=None
    )
    graph_data.y = targets
    
    # Create model
    model = create_geological_gnn_model(
        'gcn', input_dim, [128, 64], output_dim
    )
    
    # Create trainer
    trainer = GeologicalGNNTrainer(model)
    
    # Create dummy data loader
    train_loader = [graph_data]  # Simplified for testing
    val_loader = [graph_data]
    
    # Test training (just one epoch)
    criterion = nn.CrossEntropyLoss()
    train_loss, train_metrics = trainer.train_epoch(train_loader, criterion)
    
    print(f"Training completed. Loss: {train_loss:.4f}")
    if 'accuracy' in train_metrics:
        print(f"Training accuracy: {train_metrics['accuracy']:.4f}")
    
    return True


if __name__ == "__main__":
    test_gnn_training()
    logger.info("GNN training module implemented and tested successfully")