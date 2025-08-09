"""
Federated Learning System
Distributed machine learning for geological monitoring networks with privacy preservation
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import hashlib
import copy
import uuid
from pathlib import Path
import socket
import struct

# ML and deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim import Adam, SGD
import torch.distributed as dist
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from sklearn.cluster import KMeans

# Cryptography for secure aggregation
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    import secrets
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    warnings.warn("Cryptography not available. Secure aggregation will be limited.")

# Network communication
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False
    warnings.warn("ZeroMQ not available. Network communication will be limited.")

try:
    import requests
    import flask
    from flask import Flask, request, jsonify
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    warnings.warn("Web frameworks not available. HTTP communication will be limited.")

warnings.filterwarnings('ignore')

class FederatedRole(Enum):
    SERVER = "server"
    CLIENT = "client"
    AGGREGATOR = "aggregator"

class AggregationStrategy(Enum):
    FEDERATED_AVERAGING = "fedavg"
    WEIGHTED_AVERAGE = "weighted_avg"
    SECURE_AGGREGATION = "secure_agg"
    DIFFERENTIAL_PRIVACY = "diff_privacy"

class ModelType(Enum):
    EARTHQUAKE_DETECTION = "earthquake_detection"
    DEFORMATION_PREDICTION = "deformation_prediction"
    VOLCANIC_ACTIVITY = "volcanic_activity"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    model_type: ModelType
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING
    rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    min_clients: int = 2
    client_fraction: float = 1.0
    dp_noise_multiplier: float = 1.0  # For differential privacy
    dp_l2_norm_clip: float = 1.0
    secure_aggregation: bool = False
    model_checkpoint_freq: int = 10

@dataclass
class ClientInfo:
    """Information about federated learning client"""
    client_id: str
    device_type: str
    location: Optional[Tuple[float, float, float]]
    data_size: int
    last_seen: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelUpdate:
    """Model update from client"""
    client_id: str
    round_number: int
    parameters: Dict[str, torch.Tensor]
    data_size: int
    loss: float
    accuracy: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class GeologicalDataset(Dataset):
    """Dataset for geological monitoring data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 feature_names: List[str] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(features.shape[1])]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EarthquakeDetectionModel(nn.Module):
    """Neural network for earthquake detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)

class DeformationPredictionModel(nn.Module):
    """Neural network for deformation prediction"""
    
    def __init__(self, input_dim: int, sequence_length: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.sequence_length = sequence_length
        
        # LSTM for time series prediction
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # Predict x, y, z displacement
        )
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        lstm_out, _ = self.lstm(x)
        # Take last time step
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class VolcanicActivityModel(nn.Module):
    """Neural network for volcanic activity classification"""
    
    def __init__(self, input_dim: int, num_classes: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: str, config: FederatedConfig, 
                 local_data: GeologicalDataset, device: str = "cpu"):
        self.client_id = client_id
        self.config = config
        self.local_data = local_data
        self.device = torch.device(device)
        self.logger = logging.getLogger(f"{__name__}.client.{client_id}")
        
        # Model
        self.model = self._create_model()
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Training state
        self.current_round = 0
        self.local_losses = []
        self.local_accuracies = []
        
        # Communication
        self.server_address = None
        self.communication_manager = None
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        input_dim = self.local_data.features.shape[1]
        
        if self.config.model_type == ModelType.EARTHQUAKE_DETECTION:
            return EarthquakeDetectionModel(input_dim)
        elif self.config.model_type == ModelType.DEFORMATION_PREDICTION:
            return DeformationPredictionModel(input_dim)
        elif self.config.model_type == ModelType.VOLCANIC_ACTIVITY:
            return VolcanicActivityModel(input_dim, num_classes=5)
        else:
            # Generic classification/regression model
            output_dim = 1 if self.config.model_type == ModelType.REGRESSION else 2
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
    
    def update_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Update local model with global parameters"""
        try:
            self.model.load_state_dict(global_parameters)
            self.logger.info(f"Updated local model for round {self.current_round}")
        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")
    
    async def local_training(self) -> ModelUpdate:
        """Perform local training"""
        self.model.train()
        data_loader = DataLoader(self.local_data, batch_size=self.config.batch_size, shuffle=True)
        
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch_features, batch_labels in data_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                
                # Calculate loss
                if self.config.model_type in [ModelType.DEFORMATION_PREDICTION, ModelType.REGRESSION]:
                    loss = F.mse_loss(outputs, batch_labels)
                    # For regression, calculate R²
                    correct_predictions += len(batch_labels)
                    total_predictions += len(batch_labels)
                else:
                    # Classification
                    if batch_labels.dim() == 1:
                        batch_labels = batch_labels.long()
                    loss = F.cross_entropy(outputs, batch_labels)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += batch_labels.size(0)
                    correct_predictions += (predicted == batch_labels).sum().item()
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy noise if enabled
                if self.config.aggregation_strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
                    self._add_dp_noise()
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Calculate epoch metrics
            avg_epoch_loss = epoch_loss / len(data_loader)
            epoch_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            
            epoch_losses.append(avg_epoch_loss)
            epoch_accuracies.append(epoch_accuracy)
            
            self.logger.debug(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Accuracy={epoch_accuracy:.4f}")
        
        # Store training history
        self.local_losses.extend(epoch_losses)
        self.local_accuracies.extend(epoch_accuracies)
        
        # Create model update
        model_update = ModelUpdate(
            client_id=self.client_id,
            round_number=self.current_round,
            parameters=copy.deepcopy(self.model.state_dict()),
            data_size=len(self.local_data),
            loss=np.mean(epoch_losses),
            accuracy=np.mean(epoch_accuracies),
            timestamp=datetime.utcnow(),
            metadata={
                'local_epochs': self.config.local_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            }
        )
        
        self.current_round += 1
        
        return model_update
    
    def _add_dp_noise(self):
        """Add differential privacy noise to gradients"""
        if self.config.aggregation_strategy != AggregationStrategy.DIFFERENTIAL_PRIVACY:
            return
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.dp_l2_norm_clip)
        
        # Add noise to gradients
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(0, self.config.dp_noise_multiplier * self.config.dp_l2_norm_clip, param.grad.shape)
                param.grad += noise.to(param.device)
    
    def evaluate_model(self, test_data: GeologicalDataset) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        test_loader = DataLoader(test_data, batch_size=self.config.batch_size, shuffle=False)
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                
                # Calculate loss
                if self.config.model_type in [ModelType.DEFORMATION_PREDICTION, ModelType.REGRESSION]:
                    loss = F.mse_loss(outputs, batch_labels)
                else:
                    if batch_labels.dim() == 1:
                        batch_labels = batch_labels.long()
                    loss = F.cross_entropy(outputs, batch_labels)
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_predictions += batch_labels.size(0)
                    correct_predictions += (predicted == batch_labels).sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': len(test_data)
        }

class FederatedServer:
    """Federated learning server/aggregator"""
    
    def __init__(self, config: FederatedConfig, model_template: nn.Module):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.server")
        
        # Global model
        self.global_model = model_template
        self.global_parameters = self.global_model.state_dict()
        
        # Training state
        self.current_round = 0
        self.clients = {}
        self.round_updates = {}
        self.training_history = []
        
        # Communication
        self.communication_manager = None
        
        # Secure aggregation
        self.private_key = None
        self.public_key = None
        if self.config.secure_aggregation and CRYPTO_AVAILABLE:
            self._generate_keys()
    
    def _generate_keys(self):
        """Generate RSA key pair for secure aggregation"""
        try:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            self.public_key = self.private_key.public_key()
            self.logger.info("Generated RSA key pair for secure aggregation")
        except Exception as e:
            self.logger.error(f"Failed to generate keys: {e}")
    
    def register_client(self, client_info: ClientInfo):
        """Register a new client"""
        self.clients[client_info.client_id] = client_info
        self.logger.info(f"Registered client: {client_info.client_id}")
    
    def select_clients(self) -> List[str]:
        """Select clients for current round"""
        available_clients = [
            client_id for client_id, client_info in self.clients.items()
            if (datetime.utcnow() - client_info.last_seen) < timedelta(minutes=10)
        ]
        
        num_clients = max(
            self.config.min_clients,
            int(len(available_clients) * self.config.client_fraction)
        )
        
        selected = available_clients[:num_clients] if len(available_clients) >= num_clients else available_clients
        
        self.logger.info(f"Selected {len(selected)} clients for round {self.current_round}")
        return selected
    
    def receive_update(self, model_update: ModelUpdate):
        """Receive model update from client"""
        if self.current_round not in self.round_updates:
            self.round_updates[self.current_round] = {}
        
        self.round_updates[self.current_round][model_update.client_id] = model_update
        
        # Update client info
        if model_update.client_id in self.clients:
            self.clients[model_update.client_id].last_seen = datetime.utcnow()
            self.clients[model_update.client_id].data_size = model_update.data_size
            self.clients[model_update.client_id].performance_metrics = {
                'loss': model_update.loss,
                'accuracy': model_update.accuracy
            }
        
        self.logger.info(f"Received update from client {model_update.client_id} for round {self.current_round}")
    
    def aggregate_updates(self) -> Dict[str, torch.Tensor]:
        """Aggregate client updates"""
        if self.current_round not in self.round_updates:
            return self.global_parameters
        
        updates = self.round_updates[self.current_round]
        
        if not updates:
            return self.global_parameters
        
        if self.config.aggregation_strategy == AggregationStrategy.FEDERATED_AVERAGING:
            return self._federated_averaging(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_averaging(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.SECURE_AGGREGATION:
            return self._secure_aggregation(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.DIFFERENTIAL_PRIVACY:
            return self._dp_aggregation(updates)
        else:
            return self._federated_averaging(updates)
    
    def _federated_averaging(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Standard federated averaging (FedAvg)"""
        aggregated_params = {}
        total_samples = sum(update.data_size for update in updates.values())
        
        # Initialize aggregated parameters
        for param_name in self.global_parameters.keys():
            aggregated_params[param_name] = torch.zeros_like(self.global_parameters[param_name])
        
        # Aggregate parameters
        for client_id, update in updates.items():
            weight = update.data_size / total_samples
            
            for param_name, param_tensor in update.parameters.items():
                if param_name in aggregated_params:
                    aggregated_params[param_name] += weight * param_tensor
        
        self.logger.info(f"Aggregated {len(updates)} client updates using FedAvg")
        return aggregated_params
    
    def _weighted_averaging(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Weighted averaging based on client performance"""
        aggregated_params = {}
        
        # Calculate weights based on inverse loss (better performance = higher weight)
        weights = {}
        total_weight = 0
        
        for client_id, update in updates.items():
            # Use inverse loss as weight, add small epsilon to avoid division by zero
            weight = 1.0 / (update.loss + 1e-6)
            weights[client_id] = weight
            total_weight += weight
        
        # Normalize weights
        for client_id in weights:
            weights[client_id] /= total_weight
        
        # Initialize aggregated parameters
        for param_name in self.global_parameters.keys():
            aggregated_params[param_name] = torch.zeros_like(self.global_parameters[param_name])
        
        # Aggregate with performance weights
        for client_id, update in updates.items():
            weight = weights[client_id]
            
            for param_name, param_tensor in update.parameters.items():
                if param_name in aggregated_params:
                    aggregated_params[param_name] += weight * param_tensor
        
        self.logger.info(f"Aggregated {len(updates)} client updates using weighted averaging")
        return aggregated_params
    
    def _secure_aggregation(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Secure aggregation using homomorphic encryption (simplified)"""
        if not self.config.secure_aggregation or not CRYPTO_AVAILABLE:
            return self._federated_averaging(updates)
        
        # This is a simplified version - in practice would use more sophisticated
        # secure multi-party computation protocols
        self.logger.info("Secure aggregation not fully implemented, falling back to FedAvg")
        return self._federated_averaging(updates)
    
    def _dp_aggregation(self, updates: Dict[str, ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Differential privacy aggregation"""
        # Use federated averaging as base
        aggregated_params = self._federated_averaging(updates)
        
        # Add noise to aggregated parameters
        for param_name, param_tensor in aggregated_params.items():
            noise = torch.normal(
                0, 
                self.config.dp_noise_multiplier * self.config.dp_l2_norm_clip / len(updates),
                param_tensor.shape
            )
            aggregated_params[param_name] += noise
        
        self.logger.info(f"Applied differential privacy to aggregated parameters")
        return aggregated_params
    
    async def run_round(self) -> Dict[str, Any]:
        """Run a single federated learning round"""
        self.logger.info(f"Starting round {self.current_round}")
        
        # Select clients
        selected_clients = self.select_clients()
        
        if len(selected_clients) < self.config.min_clients:
            self.logger.warning(f"Not enough clients available: {len(selected_clients)} < {self.config.min_clients}")
            return {"status": "insufficient_clients"}
        
        # Wait for client updates (in real implementation, this would be event-driven)
        await asyncio.sleep(2)  # Simulate training time
        
        # Check if we have updates from selected clients
        if self.current_round in self.round_updates:
            received_updates = len(self.round_updates[self.current_round])
        else:
            received_updates = 0
        
        if received_updates == 0:
            self.logger.warning("No updates received from clients")
            return {"status": "no_updates"}
        
        # Aggregate updates
        new_global_params = self.aggregate_updates()
        self.global_parameters = new_global_params
        self.global_model.load_state_dict(new_global_params)
        
        # Calculate round statistics
        if self.current_round in self.round_updates:
            updates = self.round_updates[self.current_round]
            avg_loss = np.mean([update.loss for update in updates.values()])
            avg_accuracy = np.mean([update.accuracy for update in updates.values()])
            total_samples = sum([update.data_size for update in updates.values()])
        else:
            avg_loss, avg_accuracy, total_samples = 0, 0, 0
        
        round_stats = {
            "round": self.current_round,
            "selected_clients": len(selected_clients),
            "received_updates": received_updates,
            "avg_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
            "total_samples": total_samples,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.training_history.append(round_stats)
        
        # Save checkpoint
        if self.current_round % self.config.model_checkpoint_freq == 0:
            await self._save_checkpoint()
        
        self.current_round += 1
        
        self.logger.info(f"Completed round {self.current_round - 1}: "
                        f"Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        
        return round_stats
    
    async def _save_checkpoint(self):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'round': self.current_round,
                'model_state_dict': self.global_model.state_dict(),
                'training_history': self.training_history,
                'config': self.config.__dict__
            }
            
            checkpoint_path = f"federated_model_round_{self.current_round}.pt"
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters"""
        return copy.deepcopy(self.global_parameters)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'training_history': self.training_history,
            'client_stats': {
                client_id: {
                    'data_size': client.data_size,
                    'last_seen': client.last_seen.isoformat(),
                    'performance': client.performance_metrics
                }
                for client_id, client in self.clients.items()
            }
        }

class FederatedLearningOrchestrator:
    """Orchestrate federated learning across geological monitoring networks"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.orchestrator")
        
        # Create model template
        self.model_template = self._create_model_template()
        
        # Initialize server
        self.server = FederatedServer(config, self.model_template)
        
        # Track clients
        self.active_clients = {}
    
    def _create_model_template(self) -> nn.Module:
        """Create model template based on configuration"""
        # This would typically be determined by the specific use case
        # For demonstration, create a simple model
        if self.config.model_type == ModelType.EARTHQUAKE_DETECTION:
            return EarthquakeDetectionModel(input_dim=20)  # Example feature count
        elif self.config.model_type == ModelType.DEFORMATION_PREDICTION:
            return DeformationPredictionModel(input_dim=10)
        elif self.config.model_type == ModelType.VOLCANIC_ACTIVITY:
            return VolcanicActivityModel(input_dim=15, num_classes=5)
        else:
            return nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
    
    def add_client(self, client: FederatedClient):
        """Add client to federated learning"""
        self.active_clients[client.client_id] = client
        
        # Register with server
        client_info = ClientInfo(
            client_id=client.client_id,
            device_type="geological_sensor",
            location=None,  # Would be provided by client
            data_size=len(client.local_data),
            last_seen=datetime.utcnow()
        )
        
        self.server.register_client(client_info)
        
        self.logger.info(f"Added client {client.client_id} to federated learning")
    
    async def run_federated_learning(self) -> Dict[str, Any]:
        """Run complete federated learning process"""
        self.logger.info(f"Starting federated learning with {len(self.active_clients)} clients")
        
        results = {
            'config': self.config.__dict__,
            'rounds_completed': 0,
            'final_performance': {},
            'training_history': []
        }
        
        for round_num in range(self.config.rounds):
            # Get current global model
            global_params = self.server.get_global_model()
            
            # Update client models
            client_updates = []
            for client in self.active_clients.values():
                client.update_model(global_params)
                
                # Perform local training
                update = await client.local_training()
                client_updates.append(update)
                
                # Send update to server
                self.server.receive_update(update)
            
            # Run server round
            round_stats = await self.server.run_round()
            results['training_history'].append(round_stats)
            
            if round_stats.get('status') == 'insufficient_clients':
                self.logger.warning("Stopping federated learning due to insufficient clients")
                break
            
            results['rounds_completed'] += 1
            
            self.logger.info(f"Round {round_num + 1}/{self.config.rounds} completed")
        
        # Final evaluation
        if self.active_clients:
            final_performance = await self._evaluate_global_model()
            results['final_performance'] = final_performance
        
        results['server_stats'] = self.server.get_training_stats()
        
        self.logger.info(f"Federated learning completed after {results['rounds_completed']} rounds")
        
        return results
    
    async def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance"""
        # This would typically involve a separate test dataset
        # For demonstration, evaluate on one client's data
        if not self.active_clients:
            return {}
        
        client = list(self.active_clients.values())[0]
        
        # Create test data (in practice, would be separate)
        test_features = client.local_data.features[:100]  # First 100 samples
        test_labels = client.local_data.labels[:100]
        test_dataset = GeologicalDataset(test_features.numpy(), test_labels.numpy())
        
        # Evaluate
        performance = client.evaluate_model(test_dataset)
        
        return performance

# Example usage and testing
async def example_federated_learning():
    """Example of using federated learning system"""
    
    print("=== Federated Learning System Example ===")
    
    # Generate synthetic geological data for multiple clients
    def generate_geological_data(num_samples: int, noise_level: float = 0.1):
        # Simulate seismic features
        np.random.seed(42)
        
        # Features: magnitude, depth, distance, frequency components, etc.
        features = np.random.randn(num_samples, 20)
        
        # Labels: earthquake detection (binary classification)
        # Create some correlation with features
        earthquake_prob = 1 / (1 + np.exp(-(features[:, 0] + 0.5 * features[:, 1] - 0.3)))
        labels = (np.random.random(num_samples) < earthquake_prob).astype(float)
        
        return features, labels
    
    # Configuration
    config = FederatedConfig(
        model_type=ModelType.EARTHQUAKE_DETECTION,
        aggregation_strategy=AggregationStrategy.FEDERATED_AVERAGING,
        rounds=10,
        local_epochs=3,
        batch_size=16,
        learning_rate=0.001,
        min_clients=2,
        client_fraction=1.0
    )
    
    # Create orchestrator
    orchestrator = FederatedLearningOrchestrator(config)
    
    # Create federated clients with different data distributions
    client_configs = [
        ("seismic_station_1", 1000, 0.05),
        ("seismic_station_2", 800, 0.08),
        ("seismic_station_3", 1200, 0.06),
        ("gps_station_1", 600, 0.10)
    ]
    
    for client_id, num_samples, noise_level in client_configs:
        # Generate local data
        features, labels = generate_geological_data(num_samples, noise_level)
        local_dataset = GeologicalDataset(features, labels)
        
        # Create client
        client = FederatedClient(
            client_id=client_id,
            config=config,
            local_data=local_dataset,
            device="cpu"
        )
        
        orchestrator.add_client(client)
        print(f"Added client {client_id} with {num_samples} samples")
    
    # Run federated learning
    results = await orchestrator.run_federated_learning()
    
    print(f"\nFederated Learning Results:")
    print(f"Rounds completed: {results['rounds_completed']}")
    print(f"Final performance: {results['final_performance']}")
    
    # Show training history
    if results['training_history']:
        final_round = results['training_history'][-1]
        print(f"Final round statistics:")
        print(f"  Average loss: {final_round['avg_loss']:.4f}")
        print(f"  Average accuracy: {final_round['avg_accuracy']:.4f}")
        print(f"  Total samples: {final_round['total_samples']}")
    
    # Show client statistics
    server_stats = results.get('server_stats', {})
    if server_stats.get('client_stats'):
        print(f"\nClient Statistics:")
        for client_id, stats in server_stats['client_stats'].items():
            print(f"  {client_id}: {stats['data_size']} samples, "
                  f"Loss: {stats['performance'].get('loss', 0):.4f}")
    
    print("Federated learning example completed!")

if __name__ == "__main__":
    print("=== Federated Learning System ===")
    print(f"Available features:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Cryptography: {CRYPTO_AVAILABLE}")
    print(f"  ZeroMQ: {ZMQ_AVAILABLE}")
    print(f"  Web Framework: {WEB_AVAILABLE}")
    
    # Run example
    asyncio.run(example_federated_learning())