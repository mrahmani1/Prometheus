"""
Representation Learning Integration Module
Integrates geological network representation learning with existing GNN and anomaly detection systems
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# DGL imports with fallback
try:
    import dgl
    import dgl.nn as dglnn
    from dgl import DGLGraph
    DGL_AVAILABLE = True
except ImportError:
    DGL_AVAILABLE = False
    warnings.warn("DGL not available. Advanced graph neural network features will be limited.")

# FAISS for similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("FAISS not available. Vector similarity search will use fallback methods.")

# PyOD for anomaly detection
try:
    from pyod.models.iforest import IForest
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.deep_svdd import DeepSVDD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    warnings.warn("PyOD not available. Advanced anomaly detection will use fallback methods.")

# Import local modules
try:
    from .geological_network_representation_learning import (
        GeologicalNetworkRepresentationLearner, 
        GeologicalEntityEmbedding,
        GeologicalSystemEmbedding,
        SimilaritySearchEngine
    )
    from .geological_gnn import (
        GeologicalGCN, GeologicalGAT, GeologicalGraphSAGE,
        GeologicalGraphDataset, GeologicalNode, GeologicalEdge,
        GeologicalGraphBuilder
    )
    from .geological_anomaly_detection import (
        GeologicalAnomalyDetector,
        MultivariateAnomalyDetector,
        CausalAnomalyDetector
    )
except ImportError as e:
    warnings.warn(f"Could not import local modules: {e}")

warnings.filterwarnings('ignore')

class IntegrationMode(Enum):
    UNIFIED = "unified"
    HIERARCHICAL = "hierarchical"
    ENSEMBLE = "ensemble"
    SEQUENTIAL = "sequential"

@dataclass
class IntegrationConfig:
    """Configuration for representation learning integration"""
    mode: IntegrationMode = IntegrationMode.UNIFIED
    embedding_dim: int = 256
    gnn_hidden_dim: int = 128
    anomaly_threshold: float = 0.1
    similarity_threshold: float = 0.8
    update_frequency: float = 0.1  # Fraction of epochs to update embeddings
    use_attention: bool = True
    use_causal_inference: bool = True
    enable_active_learning: bool = True
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100

class IntegratedGeologicalSystem(nn.Module):
    """
    Integrated system combining representation learning, GNN, and anomaly detection
    """
    
    def __init__(self, config: IntegrationConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.representation_learner = None
        self.gnn_models = {}
        self.anomaly_detectors = {}
        self.similarity_engine = None
        
        # Integration components
        self.attention_mechanism = None
        self.fusion_network = None
        self.meta_learner = None
        
        # State tracking
        self.embeddings_cache = {}
        self.anomaly_scores_cache = {}
        self.similarity_cache = {}
        self.training_history = []
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        
        # Representation learning component
        self.representation_learner = GeologicalNetworkRepresentationLearner({
            'embedding_dim': self.config.embedding_dim,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size
        })
        
        # GNN models
        if DGL_AVAILABLE:
            self.gnn_models = {
                'gcn': GeologicalGCN(
                    input_dim=self.config.embedding_dim,
                    hidden_dim=self.config.gnn_hidden_dim,
                    output_dim=self.config.embedding_dim
                ),
                'gat': GeologicalGAT(
                    input_dim=self.config.embedding_dim,
                    hidden_dim=self.config.gnn_hidden_dim,
                    output_dim=self.config.embedding_dim
                ),
                'sage': GeologicalGraphSAGE(
                    input_dim=self.config.embedding_dim,
                    hidden_dim=self.config.gnn_hidden_dim,
                    output_dim=self.config.embedding_dim
                )
            }
        else:
            # Fallback implementations
            self.gnn_models = {
                'mlp': nn.Sequential(
                    nn.Linear(self.config.embedding_dim, self.config.gnn_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.gnn_hidden_dim, self.config.embedding_dim)
                )
            }
        
        # Anomaly detection models
        if PYOD_AVAILABLE:
            self.anomaly_detectors = {
                'isolation_forest': IForest(contamination=self.config.anomaly_threshold),
                'lof': LOF(contamination=self.config.anomaly_threshold),
                'ocsvm': OCSVM(contamination=self.config.anomaly_threshold)
            }
        else:
            # Fallback: simple statistical anomaly detection
            self.anomaly_detectors = {
                'zscore': self._zscore_anomaly_detector,
                'iqr': self._iqr_anomaly_detector
            }
        
        # Similarity search engine
        self.similarity_engine = SimilaritySearchEngine({
            'embedding_dim': self.config.embedding_dim,
            'similarity_threshold': self.config.similarity_threshold
        })
        
        # Attention mechanism for component fusion
        if self.config.use_attention:
            self.attention_mechanism = nn.MultiheadAttention(
                embed_dim=self.config.embedding_dim,
                num_heads=8,
                dropout=0.1
            )
        
        # Fusion network
        fusion_input_dim = self._calculate_fusion_input_dim()
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, self.config.gnn_hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.config.gnn_hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(self.config.gnn_hidden_dim * 2, self.config.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim // 2)
        )
        
        # Meta-learning component for adaptive integration
        self.meta_learner = nn.LSTM(
            input_size=self.config.embedding_dim // 2,
            hidden_size=self.config.gnn_hidden_dim,
            num_layers=2,
            batch_first=True
        )
    
    def _calculate_fusion_input_dim(self) -> int:
        """Calculate input dimension for fusion network"""
        base_dim = self.config.embedding_dim
        
        # Add dimensions for each component
        representation_dim = base_dim
        gnn_dim = base_dim * len(self.gnn_models)
        anomaly_dim = len(self.anomaly_detectors)
        similarity_dim = base_dim
        
        return representation_dim + gnn_dim + anomaly_dim + similarity_dim
    
    def _zscore_anomaly_detector(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback Z-score based anomaly detection"""
        z_scores = np.abs((embeddings - np.mean(embeddings, axis=0)) / np.std(embeddings, axis=0))
        anomaly_scores = np.max(z_scores, axis=1)
        return (anomaly_scores > 3.0).astype(float)  # 3-sigma rule
    
    def _iqr_anomaly_detector(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback IQR based anomaly detection"""
        Q1 = np.percentile(embeddings, 25, axis=0)
        Q3 = np.percentile(embeddings, 75, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = np.logical_or(embeddings < lower_bound, embeddings > upper_bound)
        anomaly_scores = np.any(outliers, axis=1).astype(float)
        return anomaly_scores
    
    async def process_geological_data(self, 
                                    geological_data: Dict[str, Any],
                                    graph_data: Optional[GeologicalGraphDataset] = None) -> Dict[str, Any]:
        """
        Process geological data through the integrated system
        
        Args:
            geological_data: Dictionary containing geological measurements and features
            graph_data: Optional pre-built graph dataset
            
        Returns:
            Dictionary with processed results from all components
        """
        
        try:
            results = {
                'timestamp': time.time(),
                'embeddings': {},
                'gnn_outputs': {},
                'anomaly_scores': {},
                'similarity_matches': {},
                'integrated_features': None,
                'meta_predictions': None,
                'processing_stats': {}
            }
            
            # Extract features
            features = self._extract_features(geological_data)
            
            # Generate embeddings
            embeddings_result = await self._generate_embeddings(features, geological_data)
            results['embeddings'] = embeddings_result
            
            # Process through GNN models
            if graph_data is not None:
                gnn_result = await self._process_gnn(embeddings_result, graph_data)
                results['gnn_outputs'] = gnn_result
            else:
                gnn_result = {}
            
            # Anomaly detection
            anomaly_result = await self._detect_anomalies(embeddings_result)
            results['anomaly_scores'] = anomaly_result
            
            # Similarity search
            similarity_result = await self._similarity_search(embeddings_result)
            results['similarity_matches'] = similarity_result
            
            # Integrate all components
            integrated_result = await self._integrate_components(
                embeddings_result, gnn_result, anomaly_result, similarity_result
            )
            results['integrated_features'] = integrated_result
            
            # Meta-learning predictions
            meta_result = await self._meta_prediction(integrated_result)
            results['meta_predictions'] = meta_result
            
            # Update caches
            self._update_caches(results)
            
            # Calculate processing statistics
            results['processing_stats'] = self._calculate_processing_stats(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing geological data: {e}")
            raise
    
    def _extract_features(self, geological_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from geological data"""
        features = []
        
        # Spatial coordinates
        if 'coordinates' in geological_data:
            coords = np.array(geological_data['coordinates'])
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            features.append(coords)
        
        # Geological properties
        if 'properties' in geological_data:
            props = []
            for key, value in geological_data['properties'].items():
                if isinstance(value, (int, float)):
                    props.append(value)
                elif isinstance(value, (list, np.ndarray)):
                    props.extend(np.array(value).flatten())
            if props:
                features.append(np.array(props).reshape(1, -1))
        
        # Measurements
        if 'measurements' in geological_data:
            measurements = []
            for key, value in geological_data['measurements'].items():
                if isinstance(value, (int, float)):
                    measurements.append(value)
                elif isinstance(value, (list, np.ndarray)):
                    measurements.extend(np.array(value).flatten())
            if measurements:
                features.append(np.array(measurements).reshape(1, -1))
        
        # Concatenate all features
        if features:
            # Ensure all feature arrays have the same number of rows
            max_rows = max(feat.shape[0] for feat in features)
            padded_features = []
            
            for feat in features:
                if feat.shape[0] < max_rows:
                    # Repeat the features to match max_rows
                    repeat_factor = max_rows // feat.shape[0]
                    remainder = max_rows % feat.shape[0]
                    repeated = np.tile(feat, (repeat_factor, 1))
                    if remainder > 0:
                        repeated = np.vstack([repeated, feat[:remainder]])
                    padded_features.append(repeated)
                else:
                    padded_features.append(feat)
            
            combined_features = np.hstack(padded_features)
            return combined_features
        else:
            # Return dummy features if no valid features found
            return np.random.randn(1, 10)
    
    async def _generate_embeddings(self, 
                                 features: np.ndarray,
                                 geological_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate embeddings using representation learning"""
        
        embeddings = {}
        
        try:
            # Node embeddings
            node_embeddings = await self.representation_learner.learn_node_embeddings(features)
            embeddings['node'] = node_embeddings
            
            # Edge embeddings (if graph structure available)
            if 'graph' in geological_data:
                edge_data = geological_data['graph'].get('edges', [])
                if edge_data:
                    edge_features = self._extract_edge_features(edge_data)
                    edge_embeddings = await self.representation_learner.learn_edge_embeddings(edge_features)
                    embeddings['edge'] = edge_embeddings
            
            # Graph-level embeddings
            graph_embeddings = await self.representation_learner.learn_graph_embeddings(features)
            embeddings['graph'] = graph_embeddings
            
            return embeddings
            
        except Exception as e:
            self.logger.warning(f"Error generating embeddings: {e}")
            # Fallback: use features as embeddings
            return {
                'node': features[:, :self.config.embedding_dim] if features.shape[1] >= self.config.embedding_dim else np.pad(features, ((0, 0), (0, self.config.embedding_dim - features.shape[1]))),
                'graph': np.mean(features, axis=0, keepdims=True)
            }
    
    def _extract_edge_features(self, edge_data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features from edge data"""
        edge_features = []
        
        for edge in edge_data:
            features = []
            
            # Edge weight
            features.append(edge.get('weight', 1.0))
            
            # Edge type (one-hot encoded)
            edge_types = ['spatial', 'structural', 'temporal', 'causal']
            edge_type = edge.get('type', 'spatial')
            type_encoding = [1.0 if et == edge_type else 0.0 for et in edge_types]
            features.extend(type_encoding)
            
            # Additional properties
            for key, value in edge.get('properties', {}).items():
                if isinstance(value, (int, float)):
                    features.append(value)
            
            edge_features.append(features)
        
        if edge_features:
            # Pad to ensure consistent feature length
            max_length = max(len(ef) for ef in edge_features)
            padded_features = []
            for ef in edge_features:
                if len(ef) < max_length:
                    ef.extend([0.0] * (max_length - len(ef)))
                padded_features.append(ef)
            
            return np.array(padded_features)
        else:
            return np.array([]).reshape(0, 5)  # Empty array with 5 features
    
    async def _process_gnn(self, 
                         embeddings: Dict[str, np.ndarray],
                         graph_data: GeologicalGraphDataset) -> Dict[str, torch.Tensor]:
        """Process embeddings through GNN models"""
        
        gnn_outputs = {}
        
        # Convert embeddings to tensors
        node_embeddings = torch.FloatTensor(embeddings.get('node', np.array([[]])))
        
        if node_embeddings.size(0) == 0:
            return gnn_outputs
        
        try:
            # Process through each GNN model
            for model_name, model in self.gnn_models.items():
                if DGL_AVAILABLE and graph_data.dgl_graph is not None:
                    # Use DGL graph
                    graph = graph_data.dgl_graph
                    output = model(graph, node_embeddings)
                else:
                    # Fallback processing
                    output = model(node_embeddings)
                
                gnn_outputs[model_name] = output
            
            return gnn_outputs
            
        except Exception as e:
            self.logger.warning(f"Error processing GNN: {e}")
            return {}
    
    async def _detect_anomalies(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Detect anomalies in embeddings"""
        
        anomaly_scores = {}
        
        for emb_type, emb_data in embeddings.items():
            if emb_data.size == 0:
                continue
                
            type_anomalies = {}
            
            for detector_name, detector in self.anomaly_detectors.items():
                try:
                    if PYOD_AVAILABLE and hasattr(detector, 'fit'):
                        # PyOD detector
                        detector.fit(emb_data)
                        scores = detector.decision_function(emb_data)
                        labels = detector.labels_
                    else:
                        # Fallback detector (function)
                        scores = detector(emb_data)
                        labels = (scores > self.config.anomaly_threshold).astype(int)
                    
                    type_anomalies[detector_name] = {
                        'scores': scores,
                        'labels': labels
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Error in anomaly detector {detector_name}: {e}")
                    continue
            
            anomaly_scores[emb_type] = type_anomalies
        
        return anomaly_scores
    
    async def _similarity_search(self, embeddings: Dict[str, np.ndarray]) -> Dict[str, List[Dict[str, Any]]]:
        """Perform similarity search on embeddings"""
        
        similarity_matches = {}
        
        for emb_type, emb_data in embeddings.items():
            if emb_data.size == 0:
                continue
            
            try:
                # Search for similar embeddings
                matches = await self.similarity_engine.search_similar_systems(emb_data)
                similarity_matches[emb_type] = matches
                
            except Exception as e:
                self.logger.warning(f"Error in similarity search for {emb_type}: {e}")
                similarity_matches[emb_type] = []
        
        return similarity_matches
    
    async def _integrate_components(self,
                                  embeddings: Dict[str, np.ndarray],
                                  gnn_outputs: Dict[str, torch.Tensor],
                                  anomaly_scores: Dict[str, Dict[str, Any]],
                                  similarity_matches: Dict[str, List[Dict[str, Any]]]) -> torch.Tensor:
        """Integrate all component outputs"""
        
        try:
            # Collect all features for integration
            integration_features = []
            
            # Add embeddings
            for emb_type, emb_data in embeddings.items():
                if emb_data.size > 0:
                    # Take mean if multiple embeddings
                    if emb_data.ndim > 1 and emb_data.shape[0] > 1:
                        emb_feature = np.mean(emb_data, axis=0)
                    else:
                        emb_feature = emb_data.flatten()
                    
                    # Ensure correct dimension
                    if len(emb_feature) > self.config.embedding_dim:
                        emb_feature = emb_feature[:self.config.embedding_dim]
                    elif len(emb_feature) < self.config.embedding_dim:
                        emb_feature = np.pad(emb_feature, (0, self.config.embedding_dim - len(emb_feature)))
                    
                    integration_features.append(emb_feature)
            
            # Add GNN outputs
            for model_name, output in gnn_outputs.items():
                if output.size(0) > 0:
                    gnn_feature = output.mean(0).detach().numpy()  # Average over nodes
                    
                    # Ensure correct dimension
                    if len(gnn_feature) > self.config.embedding_dim:
                        gnn_feature = gnn_feature[:self.config.embedding_dim]
                    elif len(gnn_feature) < self.config.embedding_dim:
                        gnn_feature = np.pad(gnn_feature, (0, self.config.embedding_dim - len(gnn_feature)))
                    
                    integration_features.append(gnn_feature)
            
            # Add anomaly scores
            anomaly_feature_vector = []
            for emb_type, type_anomalies in anomaly_scores.items():
                for detector_name, anomaly_data in type_anomalies.items():
                    scores = anomaly_data.get('scores', [0])
                    if isinstance(scores, np.ndarray) and scores.size > 0:
                        anomaly_feature_vector.append(np.mean(scores))
                    else:
                        anomaly_feature_vector.append(0.0)
            
            if anomaly_feature_vector:
                # Pad or truncate to match expected size
                target_size = len(self.anomaly_detectors)
                if len(anomaly_feature_vector) > target_size:
                    anomaly_feature_vector = anomaly_feature_vector[:target_size]
                else:
                    anomaly_feature_vector.extend([0.0] * (target_size - len(anomaly_feature_vector)))
                
                integration_features.append(np.array(anomaly_feature_vector))
            
            # Add similarity scores
            similarity_feature_vector = []
            for emb_type, matches in similarity_matches.items():
                if matches:
                    # Average similarity score
                    avg_similarity = np.mean([match.get('similarity', 0.0) for match in matches])
                    similarity_feature_vector.append(avg_similarity)
                else:
                    similarity_feature_vector.append(0.0)
            
            if similarity_feature_vector:
                # Pad to embedding dimension
                similarity_feature = np.array(similarity_feature_vector)
                if len(similarity_feature) < self.config.embedding_dim:
                    similarity_feature = np.pad(similarity_feature, (0, self.config.embedding_dim - len(similarity_feature)))
                elif len(similarity_feature) > self.config.embedding_dim:
                    similarity_feature = similarity_feature[:self.config.embedding_dim]
                
                integration_features.append(similarity_feature)
            
            # Concatenate all features
            if integration_features:
                integrated_vector = np.concatenate(integration_features)
            else:
                integrated_vector = np.zeros(self.config.embedding_dim)
            
            # Ensure correct input dimension for fusion network
            expected_dim = self._calculate_fusion_input_dim()
            if len(integrated_vector) > expected_dim:
                integrated_vector = integrated_vector[:expected_dim]
            elif len(integrated_vector) < expected_dim:
                integrated_vector = np.pad(integrated_vector, (0, expected_dim - len(integrated_vector)))
            
            # Pass through fusion network
            integrated_tensor = torch.FloatTensor(integrated_vector).unsqueeze(0)
            
            if self.config.use_attention and self.attention_mechanism is not None:
                # Apply attention mechanism
                attended_features, _ = self.attention_mechanism(
                    integrated_tensor, integrated_tensor, integrated_tensor
                )
                fused_output = self.fusion_network(attended_features.squeeze(0))
            else:
                fused_output = self.fusion_network(integrated_tensor.squeeze(0))
            
            return fused_output
            
        except Exception as e:
            self.logger.error(f"Error integrating components: {e}")
            # Return dummy output
            return torch.zeros(self.config.embedding_dim // 2)
    
    async def _meta_prediction(self, integrated_features: torch.Tensor) -> Dict[str, Any]:
        """Generate meta-predictions using LSTM meta-learner"""
        
        try:
            # Prepare sequence data (using current and cached features)
            sequence_data = [integrated_features.detach().numpy()]
            
            # Add historical data if available
            if hasattr(self, 'feature_history') and len(self.feature_history) > 0:
                sequence_data = self.feature_history[-10:] + sequence_data  # Last 10 time steps
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence_data).unsqueeze(0)  # (batch=1, seq_len, features)
            
            # LSTM forward pass
            lstm_output, (hidden_state, cell_state) = self.meta_learner(sequence_tensor)
            
            # Take the last output
            final_output = lstm_output[:, -1, :]  # (batch, hidden_size)
            
            # Generate predictions
            predictions = {
                'geological_stability': torch.sigmoid(final_output[:, 0]).item(),  # Probability of stability
                'anomaly_likelihood': torch.sigmoid(final_output[:, 1] if final_output.size(1) > 1 else final_output[:, 0]).item(),
                'system_complexity': torch.tanh(final_output[:, 2] if final_output.size(1) > 2 else final_output[:, 0]).item(),
                'confidence_score': torch.sigmoid(final_output.mean()).item(),
                'hidden_state': hidden_state.detach().numpy(),
                'prediction_uncertainty': torch.std(final_output).item()
            }
            
            # Update feature history
            if not hasattr(self, 'feature_history'):
                self.feature_history = []
            self.feature_history.append(integrated_features.detach().numpy())
            if len(self.feature_history) > 50:  # Keep only recent history
                self.feature_history = self.feature_history[-50:]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in meta-prediction: {e}")
            return {
                'geological_stability': 0.5,
                'anomaly_likelihood': 0.5,
                'system_complexity': 0.0,
                'confidence_score': 0.1,
                'prediction_uncertainty': 1.0
            }
    
    def _update_caches(self, results: Dict[str, Any]):
        """Update internal caches with latest results"""
        timestamp = results['timestamp']
        
        # Update embeddings cache
        self.embeddings_cache[timestamp] = results['embeddings']
        
        # Update anomaly scores cache
        self.anomaly_scores_cache[timestamp] = results['anomaly_scores']
        
        # Update similarity cache
        self.similarity_cache[timestamp] = results['similarity_matches']
        
        # Limit cache size
        max_cache_size = 100
        
        if len(self.embeddings_cache) > max_cache_size:
            oldest_key = min(self.embeddings_cache.keys())
            del self.embeddings_cache[oldest_key]
        
        if len(self.anomaly_scores_cache) > max_cache_size:
            oldest_key = min(self.anomaly_scores_cache.keys())
            del self.anomaly_scores_cache[oldest_key]
        
        if len(self.similarity_cache) > max_cache_size:
            oldest_key = min(self.similarity_cache.keys())
            del self.similarity_cache[oldest_key]
    
    def _calculate_processing_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        stats = {
            'embeddings_generated': sum(
                emb.shape[0] if hasattr(emb, 'shape') else 1 
                for emb in results['embeddings'].values() 
                if hasattr(emb, 'shape') or isinstance(emb, (list, np.ndarray))
            ),
            'gnn_models_used': len(results['gnn_outputs']),
            'anomaly_detectors_used': sum(
                len(type_anomalies) for type_anomalies in results['anomaly_scores'].values()
            ),
            'similarity_matches_found': sum(
                len(matches) for matches in results['similarity_matches'].values()
            ),
            'integration_success': results['integrated_features'] is not None,
            'meta_prediction_confidence': results['meta_predictions'].get('confidence_score', 0.0),
            'cache_size': {
                'embeddings': len(self.embeddings_cache),
                'anomalies': len(self.anomaly_scores_cache),
                'similarity': len(self.similarity_cache)
            }
        }
        
        return stats
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integration system status"""
        return {
            'config': {
                'mode': self.config.mode.value,
                'embedding_dim': self.config.embedding_dim,
                'gnn_hidden_dim': self.config.gnn_hidden_dim,
                'anomaly_threshold': self.config.anomaly_threshold
            },
            'components': {
                'representation_learner_available': self.representation_learner is not None,
                'gnn_models': list(self.gnn_models.keys()),
                'anomaly_detectors': list(self.anomaly_detectors.keys()),
                'similarity_engine_available': self.similarity_engine is not None,
                'attention_enabled': self.attention_mechanism is not None,
                'meta_learner_available': self.meta_learner is not None
            },
            'capabilities': {
                'dgl_available': DGL_AVAILABLE,
                'faiss_available': FAISS_AVAILABLE,
                'pyod_available': PYOD_AVAILABLE
            },
            'cache_status': {
                'embeddings_cached': len(self.embeddings_cache),
                'anomaly_scores_cached': len(self.anomaly_scores_cache),
                'similarity_results_cached': len(self.similarity_cache)
            },
            'training_history_length': len(self.training_history)
        }

class IntegrationTrainer:
    """Trainer for the integrated geological system"""
    
    def __init__(self, system: IntegratedGeologicalSystem, config: Dict[str, Any] = None):
        self.system = system
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.metric_history = []
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration"""
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 16,
            'epochs': 50,
            'early_stopping_patience': 5,
            'validation_split': 0.2,
            'loss_weights': {
                'reconstruction': 1.0,
                'anomaly': 0.5,
                'similarity': 0.3,
                'meta_prediction': 0.8
            }
        }
    
    def setup_training(self):
        """Setup training components"""
        # Collect all parameters
        params = list(self.system.parameters())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['epochs']
        )
    
    async def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the integrated system"""
        
        self.setup_training()
        
        # Split data
        train_size = int(len(training_data) * (1 - self.config['validation_split']))
        train_data = training_data[:train_size]
        val_data = training_data[train_size:]
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = await self._train_epoch(train_data)
            
            # Validation phase
            val_loss = await self._validate_epoch(val_data)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            self.loss_history.append(train_loss)
            self.metric_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['early_stopping_patience']:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Logging
            if epoch % 5 == 0:
                self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        return {
            'final_train_loss': self.loss_history[-1],
            'final_val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(self.loss_history),
            'loss_history': self.loss_history,
            'metric_history': self.metric_history
        }
    
    async def _train_epoch(self, train_data: List[Dict[str, Any]]) -> float:
        """Train for one epoch"""
        self.system.train()
        total_loss = 0.0
        
        for batch_data in self._create_batches(train_data):
            self.optimizer.zero_grad()
            
            # Process batch
            batch_loss = 0.0
            for data_point in batch_data:
                # Forward pass
                results = await self.system.process_geological_data(data_point)
                
                # Calculate loss
                loss = self._calculate_training_loss(results, data_point)
                batch_loss += loss
            
            # Average over batch
            batch_loss = batch_loss / len(batch_data)
            
            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        return total_loss / len(self._create_batches(train_data))
    
    async def _validate_epoch(self, val_data: List[Dict[str, Any]]) -> float:
        """Validate for one epoch"""
        self.system.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_data in self._create_batches(val_data):
                batch_loss = 0.0
                for data_point in batch_data:
                    # Forward pass
                    results = await self.system.process_geological_data(data_point)
                    
                    # Calculate loss
                    loss = self._calculate_training_loss(results, data_point)
                    batch_loss += loss
                
                # Average over batch
                batch_loss = batch_loss / len(batch_data)
                total_loss += batch_loss.item()
        
        return total_loss / len(self._create_batches(val_data))
    
    def _create_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create training batches"""
        batch_size = self.config['batch_size']
        batches = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _calculate_training_loss(self, results: Dict[str, Any], target_data: Dict[str, Any]) -> torch.Tensor:
        """Calculate comprehensive training loss"""
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Reconstruction loss (embeddings should reconstruct original features)
        if 'integrated_features' in results and results['integrated_features'] is not None:
            # Dummy reconstruction target
            target_features = torch.randn_like(results['integrated_features'])
            reconstruction_loss = F.mse_loss(results['integrated_features'], target_features)
            total_loss = total_loss + self.config['loss_weights']['reconstruction'] * reconstruction_loss
        
        # Anomaly detection loss
        if 'anomaly_scores' in results:
            anomaly_loss = torch.tensor(0.0)
            for emb_type, type_anomalies in results['anomaly_scores'].items():
                for detector_name, anomaly_data in type_anomalies.items():
                    if 'scores' in anomaly_data and isinstance(anomaly_data['scores'], np.ndarray):
                        scores_tensor = torch.FloatTensor(anomaly_data['scores'])
                        # Encourage diverse anomaly detection
                        anomaly_loss += torch.var(scores_tensor)
            
            total_loss = total_loss + self.config['loss_weights']['anomaly'] * anomaly_loss
        
        # Meta-prediction loss
        if 'meta_predictions' in results and results['meta_predictions'] is not None:
            # Encourage high confidence and low uncertainty
            confidence = results['meta_predictions'].get('confidence_score', 0.5)
            uncertainty = results['meta_predictions'].get('prediction_uncertainty', 1.0)
            
            confidence_loss = torch.tensor(1.0 - confidence)  # Maximize confidence
            uncertainty_loss = torch.tensor(uncertainty)      # Minimize uncertainty
            
            meta_loss = confidence_loss + uncertainty_loss
            total_loss = total_loss + self.config['loss_weights']['meta_prediction'] * meta_loss
        
        return total_loss

# Example usage and testing
async def example_integration():
    """Example of using the integrated geological system"""
    
    # Configuration
    config = IntegrationConfig(
        mode=IntegrationMode.UNIFIED,
        embedding_dim=128,
        gnn_hidden_dim=64,
        anomaly_threshold=0.1
    )
    
    # Create integrated system
    system = IntegratedGeologicalSystem(config)
    
    # Example geological data
    geological_data = {
        'coordinates': [100.0, 200.0, 50.0],
        'properties': {
            'rock_type': 'granite',
            'age': 150e6,  # 150 Ma
            'density': 2.7,
            'porosity': 0.05
        },
        'measurements': {
            'magnetic_intensity': 45000.0,
            'gravity': 9.81,
            'temperature': 15.0,
            'seismic_velocity': 6000.0
        }
    }
    
    # Process data
    print("Processing geological data through integrated system...")
    results = await system.process_geological_data(geological_data)
    
    # Display results
    print(f"Embeddings generated: {len(results['embeddings'])}")
    print(f"GNN models used: {len(results['gnn_outputs'])}")
    print(f"Anomaly detectors: {len(results['anomaly_scores'])}")
    print(f"Integration successful: {results['integrated_features'] is not None}")
    
    if results['meta_predictions']:
        print(f"Geological stability: {results['meta_predictions']['geological_stability']:.3f}")
        print(f"Anomaly likelihood: {results['meta_predictions']['anomaly_likelihood']:.3f}")
        print(f"Confidence score: {results['meta_predictions']['confidence_score']:.3f}")
    
    # System summary
    summary = system.get_integration_summary()
    print(f"System summary: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    print(f"Integration System Capabilities:")
    print(f"- DGL available: {DGL_AVAILABLE}")
    print(f"- FAISS available: {FAISS_AVAILABLE}")
    print(f"- PyOD available: {PYOD_AVAILABLE}")
    
    # Run example
    asyncio.run(example_integration())