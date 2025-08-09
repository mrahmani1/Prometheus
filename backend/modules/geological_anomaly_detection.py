"""
Advanced Geological Anomaly Detection System
Multi-variate anomaly detection with causal reinforcement learning for new phenomena discovery
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
from collections import defaultdict, deque
import pickle
import h5py
from datetime import datetime, timedelta

# Scientific computing
from scipy import signal, stats
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve

# Advanced ML libraries with fallbacks
try:
    import pyod
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.vae import VAE
    from pyod.models.deep_svdd import DeepSVDD
    from pyod.models.lof import LOF
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    warnings.warn("PyOD not available. Some anomaly detection models will be limited.")

try:
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    warnings.warn("Stable-baselines3 not available. Reinforcement learning features will be limited.")

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn("OpenAI Gym not available. RL environment features will be limited.")

warnings.filterwarnings('ignore')

class AnomalyType(Enum):
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SPATIO_TEMPORAL = "spatio_temporal"

class GeologicalAnomalyCategory(Enum):
    SEISMIC = "seismic"
    DEFORMATION = "deformation"
    GEOCHEMICAL = "geochemical"
    THERMAL = "thermal"
    MAGNETIC = "magnetic"
    GRAVITATIONAL = "gravitational"
    HYDROGEOLOGICAL = "hydrogeological"

@dataclass
class AnomalyEvent:
    """Represents a detected anomaly event"""
    timestamp: pd.Timestamp
    location: np.ndarray  # [x, y, z]
    anomaly_type: AnomalyType
    category: GeologicalAnomalyCategory
    severity_score: float
    confidence: float
    features: Dict[str, float]
    context: Dict[str, Any]
    explanation: str = ""
    significance: float = 0.0
    causality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'location': self.location.tolist(),
            'anomaly_type': self.anomaly_type.value,
            'category': self.category.value,
            'severity_score': self.severity_score,
            'confidence': self.confidence,
            'features': self.features,
            'context': self.context,
            'explanation': self.explanation,
            'significance': self.significance,
            'causality_score': self.causality_score
        }

class MultiVariateAnomalyDetector(nn.Module):
    """Advanced multi-variate anomaly detection using deep neural networks"""
    
    def __init__(self, input_dim: int, latent_dim: int = 64, 
                 architecture: str = 'autoencoder'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.architecture = architecture
        
        if architecture == 'autoencoder':
            self._build_autoencoder()
        elif architecture == 'vae':
            self._build_vae()
        elif architecture == 'deep_svdd':
            self._build_deep_svdd()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _build_autoencoder(self):
        """Build autoencoder architecture"""
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, self.latent_dim),
            nn.Tanh()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, self.input_dim)
        )
    
    def _build_vae(self):
        """Build Variational Autoencoder architecture"""
        # Encoder
        self.encoder_shared = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.mu_layer = nn.Linear(128, self.latent_dim)
        self.logvar_layer = nn.Linear(128, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, self.input_dim)
        )
    
    def _build_deep_svdd(self):
        """Build Deep SVDD architecture"""
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, self.latent_dim)
        )
        
        # Initialize center
        self.register_buffer('center', torch.zeros(self.latent_dim))
    
    def encode(self, x):
        """Encode input to latent space"""
        if self.architecture == 'vae':
            shared = self.encoder_shared(x)
            mu = self.mu_layer(shared)
            logvar = self.logvar_layer(shared)
            return mu, logvar
        elif self.architecture == 'deep_svdd':
            return self.network(x)
        else:
            return self.encoder(x)
    
    def decode(self, z):
        """Decode from latent space"""
        if self.architecture != 'deep_svdd':
            return self.decoder(z)
        else:
            return z  # Deep SVDD doesn't have decoder
    
    def forward(self, x):
        """Forward pass"""
        if self.architecture == 'vae':
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            reconstruction = self.decode(z)
            return reconstruction, mu, logvar
        elif self.architecture == 'deep_svdd':
            return self.encode(x)
        else:
            z = self.encode(x)
            reconstruction = self.decode(z)
            return reconstruction
    
    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_anomaly_score(self, x):
        """Compute anomaly score for input"""
        self.eval()
        with torch.no_grad():
            if self.architecture == 'vae':
                reconstruction, mu, logvar = self.forward(x)
                # Reconstruction error + KL divergence
                mse = F.mse_loss(reconstruction, x, reduction='none').sum(1)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                return mse + kld
            elif self.architecture == 'deep_svdd':
                features = self.forward(x)
                # Distance from center
                return torch.norm(features - self.center, dim=1)
            else:
                reconstruction = self.forward(x)
                return F.mse_loss(reconstruction, x, reduction='none').sum(1)

class ContextualAnomalyDetector:
    """Contextual anomaly detection using geological domain knowledge"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.context_models = {}
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'context_window': 100,  # Number of samples for context
            'similarity_threshold': 0.8,
            'seasonal_periods': [365.25, 30.44, 7.0],  # Daily, monthly, weekly patterns
            'geological_constraints': {
                'seismic': {'magnitude_range': [0, 9], 'depth_range': [0, 700]},
                'deformation': {'rate_range': [-50, 50], 'coherence_threshold': 0.7},
                'temperature': {'range': [-50, 1500], 'gradient_threshold': 100}
            }
        }
    
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """Initialize geological domain knowledge"""
        return {
            'fault_patterns': {
                'normal_fault': {'dip_range': [45, 90], 'stress_regime': 'extensional'},
                'reverse_fault': {'dip_range': [30, 60], 'stress_regime': 'compressional'},
                'strike_slip': {'dip_range': [70, 90], 'stress_regime': 'shear'}
            },
            'earthquake_scaling': {
                'gutenberg_richter': {'b_value_range': [0.8, 1.2]},
                'omori_law': {'p_value_range': [0.9, 1.5]}
            },
            'volcanic_indicators': {
                'pre_eruption': ['increased_seismicity', 'ground_deformation', 'gas_emissions'],
                'co_eruption': ['volcanic_tremor', 'ash_emissions', 'lava_flows'],
                'post_eruption': ['cooling', 'subsidence', 'reduced_activity']
            }
        }
    
    def detect_contextual_anomalies(self, data: pd.DataFrame, 
                                  context_features: List[str]) -> List[AnomalyEvent]:
        """Detect contextual anomalies using domain knowledge"""
        anomalies = []
        
        # Group data by geological context
        for context_group, group_data in data.groupby(context_features):
            # Build context model
            context_model = self._build_context_model(group_data)
            
            # Detect anomalies within context
            context_anomalies = self._detect_in_context(
                group_data, context_model, context_group
            )
            
            anomalies.extend(context_anomalies)
        
        return anomalies
    
    def _build_context_model(self, context_data: pd.DataFrame) -> Dict[str, Any]:
        """Build statistical model for specific geological context"""
        model = {
            'statistics': {},
            'patterns': {},
            'constraints': {}
        }
        
        # Basic statistics
        numeric_cols = context_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            values = context_data[col].dropna()
            if len(values) > 0:
                model['statistics'][col] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max()
                }
        
        # Temporal patterns
        if 'timestamp' in context_data.columns:
            model['patterns']['temporal'] = self._extract_temporal_patterns(context_data)
        
        # Spatial patterns
        if all(col in context_data.columns for col in ['x', 'y', 'z']):
            model['patterns']['spatial'] = self._extract_spatial_patterns(context_data)
        
        return model
    
    def _extract_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract temporal patterns from data"""
        patterns = {}
        
        if 'timestamp' not in data.columns:
            return patterns
        
        # Ensure timestamp is datetime
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        # Seasonal decomposition for each numeric feature
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data.columns and not data[col].isna().all():
                try:
                    # Resample to regular intervals
                    ts_data = data.set_index('timestamp')[col].resample('1D').mean()
                    ts_data = ts_data.dropna()
                    
                    if len(ts_data) > 14:  # Minimum data for pattern detection
                        # Simple trend analysis
                        x = np.arange(len(ts_data))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data.values)
                        
                        patterns[col] = {
                            'trend_slope': slope,
                            'trend_r_squared': r_value**2,
                            'trend_p_value': p_value,
                            'seasonality': self._detect_seasonality(ts_data.values),
                            'autocorrelation': self._compute_autocorrelation(ts_data.values)
                        }
                except Exception as e:
                    self.logger.warning(f"Could not extract temporal patterns for {col}: {e}")
        
        return patterns
    
    def _extract_spatial_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract spatial patterns from data"""
        patterns = {}
        
        try:
            # Spatial coordinates
            coords = data[['x', 'y', 'z']].dropna()
            
            if len(coords) > 1:
                # Compute spatial statistics
                patterns['centroid'] = coords.mean().to_dict()
                patterns['spread'] = coords.std().to_dict()
                
                # Clustering analysis
                if len(coords) >= 3:
                    kmeans = KMeans(n_clusters=min(3, len(coords)), random_state=42)
                    clusters = kmeans.fit_predict(coords)
                    patterns['clusters'] = {
                        'n_clusters': len(np.unique(clusters)),
                        'centers': kmeans.cluster_centers_.tolist(),
                        'silhouette_score': self._compute_silhouette_score(coords, clusters)
                    }
        except Exception as e:
            self.logger.warning(f"Could not extract spatial patterns: {e}")
        
        return patterns
    
    def _detect_seasonality(self, time_series: np.ndarray) -> Dict[str, float]:
        """Detect seasonal patterns in time series"""
        seasonality = {}
        
        for period in self.config['seasonal_periods']:
            if len(time_series) > 2 * period:
                try:
                    # Simple periodogram-based detection
                    freqs = np.fft.fftfreq(len(time_series))
                    fft = np.fft.fft(time_series)
                    power = np.abs(fft)**2
                    
                    # Find peak near expected period
                    target_freq = 1.0 / period
                    freq_idx = np.argmin(np.abs(freqs - target_freq))
                    seasonality[f'period_{period}'] = power[freq_idx]
                except:
                    seasonality[f'period_{period}'] = 0.0
        
        return seasonality
    
    def _compute_autocorrelation(self, time_series: np.ndarray, max_lag: int = 30) -> List[float]:
        """Compute autocorrelation function"""
        if len(time_series) <= max_lag:
            return []
        
        autocorr = []
        for lag in range(1, min(max_lag + 1, len(time_series) // 2)):
            corr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
            autocorr.append(corr if not np.isnan(corr) else 0.0)
        
        return autocorr
    
    def _compute_silhouette_score(self, data: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(data, labels)
        except:
            return 0.0
    
    def _detect_in_context(self, data: pd.DataFrame, context_model: Dict[str, Any], 
                          context_group: Any) -> List[AnomalyEvent]:
        """Detect anomalies within specific context"""
        anomalies = []
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for idx, row in data.iterrows():
            anomaly_scores = {}
            
            # Statistical anomaly detection
            for col in numeric_cols:
                if col in context_model['statistics'] and not pd.isna(row[col]):
                    stats = context_model['statistics'][col]
                    
                    # Z-score based detection
                    if stats['std'] > 0:
                        z_score = abs(row[col] - stats['mean']) / stats['std']
                        anomaly_scores[f'{col}_zscore'] = z_score
                    
                    # IQR based detection
                    iqr = stats['q75'] - stats['q25']
                    if iqr > 0:
                        lower_bound = stats['q25'] - 1.5 * iqr
                        upper_bound = stats['q75'] + 1.5 * iqr
                        if row[col] < lower_bound or row[col] > upper_bound:
                            anomaly_scores[f'{col}_iqr'] = 1.0
            
            # Domain knowledge validation
            domain_score = self._validate_domain_constraints(row, context_group)
            if domain_score > 0:
                anomaly_scores['domain_violation'] = domain_score
            
            # Aggregate anomaly score
            if anomaly_scores:
                max_score = max(anomaly_scores.values())
                avg_score = np.mean(list(anomaly_scores.values()))
                
                # Create anomaly event if significant
                if max_score > 3.0 or avg_score > 2.0:  # Thresholds for significance
                    anomaly = AnomalyEvent(
                        timestamp=pd.to_datetime(row.get('timestamp', pd.Timestamp.now())),
                        location=np.array([row.get('x', 0), row.get('y', 0), row.get('z', 0)]),
                        anomaly_type=AnomalyType.CONTEXTUAL,
                        category=self._infer_category(anomaly_scores),
                        severity_score=max_score,
                        confidence=min(avg_score / 5.0, 1.0),
                        features=anomaly_scores,
                        context={'context_group': str(context_group)},
                        explanation=self._generate_explanation(anomaly_scores, context_model)
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _validate_domain_constraints(self, row: pd.Series, context_group: Any) -> float:
        """Validate against geological domain constraints"""
        violation_score = 0.0
        
        # Check geological constraints
        constraints = self.config['geological_constraints']
        
        for param, constraint in constraints.items():
            if param in row and not pd.isna(row[param]):
                value = row[param]
                
                if 'range' in constraint:
                    min_val, max_val = constraint['range']
                    if value < min_val or value > max_val:
                        violation_score += 1.0
                
                if param == 'seismic' and 'magnitude' in row:
                    mag = row['magnitude']
                    depth = row.get('depth', 0)
                    
                    # Check Gutenberg-Richter relation violations
                    if mag > 8.0 and depth < 50:  # Very shallow large earthquakes are rare
                        violation_score += 0.5
        
        return violation_score
    
    def _infer_category(self, anomaly_scores: Dict[str, float]) -> GeologicalAnomalyCategory:
        """Infer geological category from anomaly scores"""
        # Simple heuristic based on feature names
        feature_names = list(anomaly_scores.keys())
        
        if any('seismic' in f or 'magnitude' in f or 'earthquake' in f for f in feature_names):
            return GeologicalAnomalyCategory.SEISMIC
        elif any('deform' in f or 'displacement' in f for f in feature_names):
            return GeologicalAnomalyCategory.DEFORMATION
        elif any('temp' in f or 'thermal' in f for f in feature_names):
            return GeologicalAnomalyCategory.THERMAL
        else:
            return GeologicalAnomalyCategory.SEISMIC  # Default
    
    def _generate_explanation(self, anomaly_scores: Dict[str, float], 
                            context_model: Dict[str, Any]) -> str:
        """Generate human-readable explanation for anomaly"""
        explanations = []
        
        for feature, score in anomaly_scores.items():
            if score > 2.0:
                if 'zscore' in feature:
                    param = feature.replace('_zscore', '')
                    explanations.append(f"{param} deviates significantly from normal pattern (z-score: {score:.2f})")
                elif 'iqr' in feature:
                    param = feature.replace('_iqr', '')
                    explanations.append(f"{param} is an outlier beyond expected range")
                elif 'domain_violation' in feature:
                    explanations.append(f"Violates geological domain constraints (score: {score:.2f})")
        
        return "; ".join(explanations) if explanations else "Anomalous pattern detected"

class CollectiveAnomalyDetector:
    """Detect collective anomalies in geological event sequences"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.sequence_models = {}
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'sequence_length': 50,
            'overlap': 0.5,
            'clustering_eps': 0.5,
            'min_samples': 3,
            'temporal_window': timedelta(hours=24),
            'spatial_radius': 10000  # meters
        }
    
    def detect_collective_anomalies(self, data: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect collective anomalies in event sequences"""
        anomalies = []
        
        # Sort data by timestamp
        data = data.sort_values('timestamp')
        
        # Create sliding windows
        sequences = self._create_sequences(data)
        
        # Analyze each sequence
        for seq_data in sequences:
            collective_anomaly = self._analyze_sequence(seq_data)
            if collective_anomaly:
                anomalies.append(collective_anomaly)
        
        return anomalies
    
    def _create_sequences(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """Create sliding window sequences from data"""
        sequences = []
        seq_len = self.config['sequence_length']
        step = int(seq_len * (1 - self.config['overlap']))
        
        for i in range(0, len(data) - seq_len + 1, step):
            sequence = data.iloc[i:i+seq_len].copy()
            sequences.append(sequence)
        
        return sequences
    
    def _analyze_sequence(self, sequence: pd.DataFrame) -> Optional[AnomalyEvent]:
        """Analyze a sequence for collective anomalies"""
        # Extract sequence features
        features = self._extract_sequence_features(sequence)
        
        # Compute sequence statistics
        stats = self._compute_sequence_statistics(sequence)
        
        # Check for known collective anomaly patterns
        anomaly_type, score, explanation = self._detect_anomaly_patterns(features, stats)
        
        if score > 0.5:  # Threshold for collective anomaly
            # Get representative timestamp and location
            center_time = sequence['timestamp'].median()
            center_location = sequence[['x', 'y', 'z']].mean().values
            
            return AnomalyEvent(
                timestamp=center_time,
                location=center_location,
                anomaly_type=AnomalyType.COLLECTIVE,
                category=GeologicalAnomalyCategory.SEISMIC,
                severity_score=score,
                confidence=min(score, 1.0),
                features=features,
                context={'sequence_length': len(sequence)},
                explanation=explanation
            )
        
        return None
    
    def _extract_sequence_features(self, sequence: pd.DataFrame) -> Dict[str, float]:
        """Extract features from event sequence"""
        features = {}
        
        numeric_cols = sequence.select_dtypes(include=[np.number]).columns
        
        # Statistical features
        for col in numeric_cols:
            if col in sequence.columns and not sequence[col].isna().all():
                values = sequence[col].dropna()
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std()
                features[f'{col}_trend'] = self._compute_trend(values)
        
        # Temporal features
        if 'timestamp' in sequence.columns:
            time_diffs = sequence['timestamp'].diff().dt.total_seconds().dropna()
            features['avg_time_interval'] = time_diffs.mean()
            features['time_interval_std'] = time_diffs.std()
        
        # Spatial features
        if all(col in sequence.columns for col in ['x', 'y', 'z']):
            coords = sequence[['x', 'y', 'z']].dropna()
            if len(coords) > 1:
                distances = cdist(coords, coords)
                features['spatial_spread'] = np.std(distances)
                features['spatial_clustering'] = self._compute_clustering_coefficient(distances)
        
        # Event rate features
        features['event_rate'] = len(sequence) / max(1, (sequence['timestamp'].max() - sequence['timestamp'].min()).total_seconds())
        
        return features
    
    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute trend in time series"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _compute_clustering_coefficient(self, distance_matrix: np.ndarray) -> float:
        """Compute spatial clustering coefficient"""
        # Simple clustering measure based on distance distribution
        mean_dist = np.mean(distance_matrix[distance_matrix > 0])
        std_dist = np.std(distance_matrix[distance_matrix > 0])
        
        if std_dist == 0:
            return 1.0
        
        return 1.0 / (1.0 + std_dist / mean_dist)
    
    def _compute_sequence_statistics(self, sequence: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistical properties of sequence"""
        stats = {}
        
        # Temporal statistics
        time_span = (sequence['timestamp'].max() - sequence['timestamp'].min()).total_seconds()
        stats['duration'] = time_span
        stats['event_count'] = len(sequence)
        stats['event_rate'] = len(sequence) / max(1, time_span)
        
        # Magnitude statistics (if available)
        if 'magnitude' in sequence.columns:
            mags = sequence['magnitude'].dropna()
            if len(mags) > 0:
                stats['magnitude_range'] = mags.max() - mags.min()
                stats['b_value'] = self._compute_b_value(mags)
        
        return stats
    
    def _compute_b_value(self, magnitudes: np.ndarray) -> float:
        """Compute Gutenberg-Richter b-value"""
        if len(magnitudes) < 10:
            return 1.0  # Default b-value
        
        # Maximum likelihood estimation
        min_mag = magnitudes.min()
        return 1.0 / (np.mean(magnitudes) - min_mag + 0.05) * np.log10(np.e)
    
    def _detect_anomaly_patterns(self, features: Dict[str, float], 
                               stats: Dict[str, Any]) -> Tuple[str, float, str]:
        """Detect known collective anomaly patterns"""
        anomaly_type = "unknown"
        score = 0.0
        explanation = ""
        
        # Earthquake swarm pattern
        if 'event_rate' in features and features['event_rate'] > 10:  # High event rate
            if 'magnitude_range' in stats and stats['magnitude_range'] < 2.0:  # Similar magnitudes
                anomaly_type = "earthquake_swarm"
                score = 0.8
                explanation = f"Earthquake swarm detected: {stats['event_count']} events in {stats['duration']:.1f}s"
        
        # Foreshock sequence
        if 'magnitude_mean' in features and 'magnitude_std' in features:
            if features['magnitude_mean'] > 3.0 and features['magnitude_std'] < 1.0:
                if 'magnitude_trend' in features and features['magnitude_trend'] > 0:
                    anomaly_type = "foreshock_sequence"
                    score = 0.7
                    explanation = "Potential foreshock sequence with increasing magnitudes"
        
        # Unusual b-value
        if 'b_value' in stats:
            if stats['b_value'] < 0.6 or stats['b_value'] > 1.4:
                anomaly_type = "unusual_b_value"
                score = 0.6
                explanation = f"Unusual b-value: {stats['b_value']:.2f} (normal range: 0.8-1.2)"
        
        return anomaly_type, score, explanation

class CausalReinforcementLearner:
    """Causal reinforcement learning for geological anomaly detection"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=10000)
        
        # Causal model
        self.causal_graph = None
        self.causal_effects = {}
        
        if RL_AVAILABLE and GYM_AVAILABLE:
            self.rl_agent = None
            self._initialize_rl_agent()
        else:
            self.logger.warning("RL libraries not available. Using simplified learning approach.")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'learning_rate': 0.001,
            'exploration_rate': 0.1,
            'gamma': 0.95,
            'update_frequency': 100,
            'causal_threshold': 0.5,
            'intervention_budget': 0.1
        }
    
    def _initialize_rl_agent(self):
        """Initialize reinforcement learning agent"""
        if not (RL_AVAILABLE and GYM_AVAILABLE):
            return
        
        try:
            # Create custom environment
            env = GeologicalAnomalyEnv(self.state_dim, self.action_dim)
            
            # Initialize PPO agent
            self.rl_agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config['learning_rate'],
                gamma=self.config['gamma'],
                verbose=0
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize RL agent: {e}")
    
    def learn_from_experience(self, state: np.ndarray, action: int, 
                            reward: float, next_state: np.ndarray, done: bool):
        """Learn from experience tuple"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': datetime.now()
        }
        
        self.experience_buffer.append(experience)
        
        # Update agent if enough experience
        if len(self.experience_buffer) >= self.config['update_frequency']:
            self._update_agent()
    
    def _update_agent(self):
        """Update RL agent with accumulated experience"""
        if self.rl_agent is None:
            return
        
        try:
            # Train agent with collected experience
            self.rl_agent.learn(total_timesteps=len(self.experience_buffer))
        except Exception as e:
            self.logger.warning(f"Could not update RL agent: {e}")
    
    def discover_causal_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover causal relationships in geological data"""
        causal_discovery = {}
        
        # Extract numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return causal_discovery
        
        # Pairwise causal discovery
        features = list(numeric_data.columns)
        causal_matrix = np.zeros((len(features), len(features)))
        
        for i, cause in enumerate(features):
            for j, effect in enumerate(features):
                if i != j:
                    causal_strength = self._test_causality(
                        numeric_data[cause].values,
                        numeric_data[effect].values
                    )
                    causal_matrix[i, j] = causal_strength
        
        causal_discovery['causal_matrix'] = causal_matrix
        causal_discovery['features'] = features
        causal_discovery['significant_relationships'] = self._extract_significant_relationships(
            causal_matrix, features
        )
        
        return causal_discovery
    
    def _test_causality(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Test causal relationship between two variables"""
        # Granger causality test (simplified)
        try:
            # Ensure sufficient data
            if len(cause) < 10 or len(effect) < 10:
                return 0.0
            
            # Cross-correlation with lags
            max_lag = min(5, len(cause) // 4)
            correlations = []
            
            for lag in range(1, max_lag + 1):
                if lag < len(cause):
                    corr = np.corrcoef(cause[:-lag], effect[lag:])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            return max(correlations) if correlations else 0.0
        
        except Exception:
            return 0.0
    
    def _extract_significant_relationships(self, causal_matrix: np.ndarray, 
                                         features: List[str]) -> List[Dict[str, Any]]:
        """Extract significant causal relationships"""
        relationships = []
        
        threshold = self.config['causal_threshold']
        
        for i in range(len(features)):
            for j in range(len(features)):
                if causal_matrix[i, j] > threshold:
                    relationships.append({
                        'cause': features[i],
                        'effect': features[j],
                        'strength': causal_matrix[i, j],
                        'type': 'positive' if causal_matrix[i, j] > 0 else 'negative'
                    })
        
        return relationships
    
    def recommend_interventions(self, anomaly: AnomalyEvent, 
                              causal_relationships: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend interventions based on causal analysis"""
        interventions = []
        
        if 'significant_relationships' not in causal_relationships:
            return interventions
        
        # Find potential intervention points
        for relationship in causal_relationships['significant_relationships']:
            if relationship['strength'] > 0.7:  # Strong causal relationship
                intervention = {
                    'type': 'causal_intervention',
                    'target_variable': relationship['cause'],
                    'expected_effect': relationship['effect'],
                    'strength': relationship['strength'],
                    'recommendation': f"Monitor {relationship['cause']} to predict changes in {relationship['effect']}",
                    'priority': relationship['strength']
                }
                interventions.append(intervention)
        
        # Sort by priority
        interventions.sort(key=lambda x: x['priority'], reverse=True)
        
        return interventions[:5]  # Top 5 interventions

# Custom Gym environment for geological anomaly detection
if GYM_AVAILABLE:
    class GeologicalAnomalyEnv(gym.Env):
        """Gymnasium environment for geological anomaly detection"""
        
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            
            # Define action and observation spaces
            self.action_space = spaces.Discrete(action_dim)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
            )
            
            # Environment state
            self.current_state = None
            self.step_count = 0
            self.max_steps = 1000
        
        def reset(self, seed=None, options=None):
            """Reset environment"""
            super().reset(seed=seed)
            
            self.current_state = np.random.randn(self.state_dim).astype(np.float32)
            self.step_count = 0
            
            return self.current_state, {}
        
        def step(self, action):
            """Take environment step"""
            # Simple dynamics: state evolves based on action
            self.current_state += 0.1 * (action - self.action_dim // 2) * np.random.randn(self.state_dim)
            
            # Compute reward (simplified)
            reward = -np.sum(np.abs(self.current_state))  # Reward for keeping state near origin
            
            self.step_count += 1
            done = self.step_count >= self.max_steps
            
            return self.current_state.astype(np.float32), reward, done, False, {}

class ContinuousLearningAnomalyDetector:
    """Continuous learning system for geological anomaly detection"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.learning_history = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize base models
        self._initialize_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'adaptation_rate': 0.01,
            'memory_size': 10000,
            'retraining_threshold': 100,
            'novelty_threshold': 0.8,
            'ensemble_size': 5,
            'multi_scale_levels': [1, 5, 25, 125]  # Time scales in days
        }
    
    def _initialize_models(self):
        """Initialize ensemble of anomaly detection models"""
        # Statistical models
        self.models['isolation_forest'] = IsolationForest(contamination=0.1, random_state=42)
        self.models['one_class_svm'] = OneClassSVM(nu=0.1, gamma='scale')
        self.models['local_outlier'] = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
        self.models['elliptic_envelope'] = EllipticEnvelope(contamination=0.1, random_state=42)
        
        # Neural network models
        if PYOD_AVAILABLE:
            try:
                self.models['autoencoder'] = AutoEncoder(contamination=0.1, random_state=42)
                self.models['deep_svdd'] = DeepSVDD(contamination=0.1, random_state=42)
            except Exception as e:
                self.logger.warning(f"Could not initialize PyOD models: {e}")
    
    def continuous_learn(self, new_data: pd.DataFrame, labels: Optional[np.ndarray] = None):
        """Continuously learn from new data"""
        # Preprocess new data
        processed_data = self._preprocess_data(new_data)
        
        # Update models incrementally
        self._incremental_update(processed_data, labels)
        
        # Adapt to distribution shifts
        self._detect_and_adapt_to_shifts(processed_data)
        
        # Record learning event
        self.learning_history.append({
            'timestamp': datetime.now(),
            'data_size': len(new_data),
            'has_labels': labels is not None
        })
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for anomaly detection"""
        # Select numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Handle missing values
        processed = numeric_data.fillna(numeric_data.mean())
        
        # Standardize
        if hasattr(self, 'scaler'):
            # Update scaler incrementally
            self.scaler.partial_fit(processed)
        else:
            self.scaler = StandardScaler()
            self.scaler.fit(processed)
        
        return self.scaler.transform(processed)
    
    def _incremental_update(self, data: np.ndarray, labels: Optional[np.ndarray] = None):
        """Update models incrementally with new data"""
        for name, model in self.models.items():
            try:
                if hasattr(model, 'partial_fit') and labels is not None:
                    # Supervised incremental learning
                    model.partial_fit(data, labels)
                elif name in ['isolation_forest', 'elliptic_envelope']:
                    # Retrain periodically for non-incremental models
                    if len(self.learning_history) % self.config['retraining_threshold'] == 0:
                        model.fit(data)
                elif hasattr(model, 'fit'):
                    # Fit on new data (will overwrite previous learning)
                    model.fit(data)
            except Exception as e:
                self.logger.warning(f"Could not update model {name}: {e}")
    
    def _detect_and_adapt_to_shifts(self, new_data: np.ndarray):
        """Detect distribution shifts and adapt accordingly"""
        if not hasattr(self, 'reference_distribution'):
            self.reference_distribution = {
                'mean': np.mean(new_data, axis=0),
                'std': np.std(new_data, axis=0),
                'cov': np.cov(new_data.T)
            }
            return
        
        # Compare current data to reference
        current_mean = np.mean(new_data, axis=0)
        current_std = np.std(new_data, axis=0)
        
        # Detect shift using KL divergence approximation
        mean_shift = np.mean(np.abs(current_mean - self.reference_distribution['mean']))
        std_shift = np.mean(np.abs(current_std - self.reference_distribution['std']))
        
        if mean_shift > 0.5 or std_shift > 0.5:  # Significant shift detected
            self.logger.info("Distribution shift detected, adapting models...")
            
            # Update reference distribution
            alpha = self.config['adaptation_rate']
            self.reference_distribution['mean'] = (1 - alpha) * self.reference_distribution['mean'] + alpha * current_mean
            self.reference_distribution['std'] = (1 - alpha) * self.reference_distribution['std'] + alpha * current_std
            
            # Retrain models with adapted parameters
            self._adapt_models(new_data)
    
    def _adapt_models(self, data: np.ndarray):
        """Adapt models to distribution changes"""
        for name, model in self.models.items():
            try:
                # Adjust model parameters based on shift
                if hasattr(model, 'contamination'):
                    # Adjust contamination rate based on recent anomaly rate
                    recent_anomalies = self._get_recent_anomaly_rate()
                    model.contamination = max(0.01, min(0.5, recent_anomalies))
                
                # Retrain with new data
                if hasattr(model, 'fit'):
                    model.fit(data)
            except Exception as e:
                self.logger.warning(f"Could not adapt model {name}: {e}")
    
    def _get_recent_anomaly_rate(self) -> float:
        """Estimate recent anomaly rate"""
        # Simple heuristic based on learning history
        recent_events = self.learning_history[-10:] if len(self.learning_history) >= 10 else self.learning_history
        if not recent_events:
            return 0.1  # Default rate
        
        # This is a placeholder - in practice, you would track actual anomaly detections
        return min(0.1 + len(recent_events) * 0.01, 0.3)
    
    def detect_multi_scale_anomalies(self, data: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect anomalies at multiple temporal scales"""
        anomalies = []
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Analyze at different time scales
        for scale in self.config['multi_scale_levels']:
            scale_anomalies = self._detect_at_scale(data, scale)
            anomalies.extend(scale_anomalies)
        
        # Remove duplicates and merge nearby anomalies
        merged_anomalies = self._merge_nearby_anomalies(anomalies)
        
        return merged_anomalies
    
    def _detect_at_scale(self, data: pd.DataFrame, scale_days: int) -> List[AnomalyEvent]:
        """Detect anomalies at specific temporal scale"""
        anomalies = []
        
        if 'timestamp' not in data.columns:
            return anomalies
        
        # Resample data to target scale
        try:
            data_resampled = data.set_index('timestamp').resample(f'{scale_days}D').mean()
            data_resampled = data_resampled.dropna()
            
            if len(data_resampled) < 10:  # Need sufficient data
                return anomalies
            
            # Preprocess for anomaly detection
            processed_data = self._preprocess_data(data_resampled.reset_index())
            
            # Apply ensemble of models
            ensemble_scores = []
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(processed_data)
                    elif hasattr(model, 'score_samples'):
                        scores = -model.score_samples(processed_data)
                    else:
                        continue
                    
                    ensemble_scores.append(scores)
                except Exception as e:
                    self.logger.warning(f"Could not apply model {name} at scale {scale_days}: {e}")
            
            if ensemble_scores:
                # Aggregate ensemble scores
                mean_scores = np.mean(ensemble_scores, axis=0)
                threshold = np.percentile(mean_scores, 95)  # Top 5% as anomalies
                
                # Create anomaly events
                anomaly_indices = np.where(mean_scores > threshold)[0]
                for idx in anomaly_indices:
                    row = data_resampled.iloc[idx]
                    
                    anomaly = AnomalyEvent(
                        timestamp=row.name,
                        location=np.array([row.get('x', 0), row.get('y', 0), row.get('z', 0)]),
                        anomaly_type=AnomalyType.TEMPORAL,
                        category=GeologicalAnomalyCategory.SEISMIC,
                        severity_score=mean_scores[idx],
                        confidence=min(mean_scores[idx] / threshold, 1.0),
                        features={'scale_days': scale_days, 'ensemble_score': mean_scores[idx]},
                        context={'temporal_scale': f'{scale_days} days'},
                        explanation=f"Multi-scale anomaly detected at {scale_days}-day resolution"
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            self.logger.warning(f"Could not detect anomalies at scale {scale_days}: {e}")
        
        return anomalies
    
    def _merge_nearby_anomalies(self, anomalies: List[AnomalyEvent]) -> List[AnomalyEvent]:
        """Merge anomalies that are close in time and space"""
        if len(anomalies) <= 1:
            return anomalies
        
        merged = []
        remaining = anomalies.copy()
        
        while remaining:
            current = remaining.pop(0)
            similar = [current]
            
            # Find similar anomalies
            to_remove = []
            for i, other in enumerate(remaining):
                time_diff = abs((current.timestamp - other.timestamp).total_seconds())
                spatial_diff = np.linalg.norm(current.location - other.location)
                
                if time_diff < 86400 and spatial_diff < 10000:  # Within 1 day and 10km
                    similar.append(other)
                    to_remove.append(i)
            
            # Remove similar anomalies from remaining
            for i in reversed(to_remove):
                remaining.pop(i)
            
            # Merge similar anomalies
            if len(similar) > 1:
                merged_anomaly = self._merge_anomaly_group(similar)
                merged.append(merged_anomaly)
            else:
                merged.append(current)
        
        return merged
    
    def _merge_anomaly_group(self, anomalies: List[AnomalyEvent]) -> AnomalyEvent:
        """Merge a group of similar anomalies"""
        # Use the anomaly with highest severity as base
        base_anomaly = max(anomalies, key=lambda x: x.severity_score)
        
        # Aggregate properties
        avg_location = np.mean([a.location for a in anomalies], axis=0)
        max_severity = max(a.severity_score for a in anomalies)
        avg_confidence = np.mean([a.confidence for a in anomalies])
        
        # Merge features
        merged_features = {}
        for anomaly in anomalies:
            merged_features.update(anomaly.features)
        
        merged_anomaly = AnomalyEvent(
            timestamp=base_anomaly.timestamp,
            location=avg_location,
            anomaly_type=base_anomaly.anomaly_type,
            category=base_anomaly.category,
            severity_score=max_severity,
            confidence=avg_confidence,
            features=merged_features,
            context={'merged_count': len(anomalies)},
            explanation=f"Merged anomaly from {len(anomalies)} similar events: {base_anomaly.explanation}"
        )
        
        return merged_anomaly

# Main anomaly detection coordinator
class GeologicalAnomalyDetectionSystem:
    """Main system coordinating all anomaly detection approaches"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-systems
        self.multivariate_detector = None
        self.contextual_detector = ContextualAnomalyDetector(config)
        self.collective_detector = CollectiveAnomalyDetector(config)
        self.continuous_learner = ContinuousLearningAnomalyDetector(config)
        self.causal_learner = None
        
        # Detection history
        self.detection_history = []
        self.performance_metrics = defaultdict(list)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'enable_multivariate': True,
            'enable_contextual': True,
            'enable_collective': True,
            'enable_continuous': True,
            'enable_causal_rl': True,
            'fusion_method': 'weighted_ensemble',
            'output_format': 'structured',
            'real_time_processing': True
        }
    
    def initialize(self, sample_data: pd.DataFrame):
        """Initialize the system with sample data"""
        self.logger.info("Initializing geological anomaly detection system...")
        
        # Initialize multivariate detector
        if self.config['enable_multivariate'] and len(sample_data) > 0:
            input_dim = len(sample_data.select_dtypes(include=[np.number]).columns)
            if input_dim > 0:
                self.multivariate_detector = MultiVariateAnomalyDetector(
                    input_dim=input_dim,
                    latent_dim=min(64, input_dim // 2),
                    architecture='autoencoder'
                )
        
        # Initialize causal RL learner
        if self.config['enable_causal_rl']:
            state_dim = min(50, len(sample_data.select_dtypes(include=[np.number]).columns))
            action_dim = 10  # Number of possible interventions
            if state_dim > 0:
                self.causal_learner = CausalReinforcementLearner(state_dim, action_dim)
        
        self.logger.info("System initialization completed.")
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        context_features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main anomaly detection pipeline"""
        results = {
            'anomalies': [],
            'summary': {},
            'recommendations': [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_size': len(data),
                'processing_time': 0
            }
        }
        
        start_time = time.time()
        
        try:
            # Multivariate anomaly detection
            if self.config['enable_multivariate'] and self.multivariate_detector:
                mv_anomalies = self._detect_multivariate_anomalies(data)
                results['anomalies'].extend(mv_anomalies)
            
            # Contextual anomaly detection
            if self.config['enable_contextual']:
                context_features = context_features or ['x', 'y', 'z']
                ctx_anomalies = self.contextual_detector.detect_contextual_anomalies(
                    data, context_features
                )
                results['anomalies'].extend(ctx_anomalies)
            
            # Collective anomaly detection
            if self.config['enable_collective']:
                collective_anomalies = self.collective_detector.detect_collective_anomalies(data)
                results['anomalies'].extend(collective_anomalies)
            
            # Multi-scale continuous learning
            if self.config['enable_continuous']:
                multiscale_anomalies = self.continuous_learner.detect_multi_scale_anomalies(data)
                results['anomalies'].extend(multiscale_anomalies)
            
            # Causal analysis and recommendations
            if self.config['enable_causal_rl'] and self.causal_learner:
                causal_relationships = self.causal_learner.discover_causal_relationships(data)
                results['metadata']['causal_relationships'] = causal_relationships
                
                # Generate recommendations for significant anomalies
                significant_anomalies = [a for a in results['anomalies'] if a.severity_score > 2.0]
                for anomaly in significant_anomalies:
                    interventions = self.causal_learner.recommend_interventions(
                        anomaly, causal_relationships
                    )
                    results['recommendations'].extend(interventions)
            
            # Fuse results and generate summary
            results['anomalies'] = self._fuse_anomaly_results(results['anomalies'])
            results['summary'] = self._generate_summary(results['anomalies'])
            
            # Update continuous learning
            if self.config['enable_continuous']:
                self.continuous_learner.continuous_learn(data)
        
        except Exception as e:
            self.logger.error(f"Error in anomaly detection pipeline: {e}")
            results['error'] = str(e)
        
        finally:
            results['metadata']['processing_time'] = time.time() - start_time
        
        # Record detection event
        self.detection_history.append(results)
        
        return results
    
    def _detect_multivariate_anomalies(self, data: pd.DataFrame) -> List[AnomalyEvent]:
        """Detect multivariate anomalies using neural networks"""
        anomalies = []
        
        # Preprocess data
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) == 0:
            return anomalies
        
        processed_data = StandardScaler().fit_transform(numeric_data.fillna(0))
        
        # Compute anomaly scores
        data_tensor = torch.FloatTensor(processed_data)
        anomaly_scores = self.multivariate_detector.compute_anomaly_score(data_tensor)
        
        # Identify anomalies
        threshold = torch.quantile(anomaly_scores, 0.95)  # Top 5% as anomalies
        anomaly_indices = torch.where(anomaly_scores > threshold)[0]
        
        for idx in anomaly_indices:
            row_idx = int(idx)
            row = data.iloc[row_idx]
            
            anomaly = AnomalyEvent(
                timestamp=pd.to_datetime(row.get('timestamp', pd.Timestamp.now())),
                location=np.array([row.get('x', 0), row.get('y', 0), row.get('z', 0)]),
                anomaly_type=AnomalyType.POINT,
                category=GeologicalAnomalyCategory.SEISMIC,
                severity_score=float(anomaly_scores[idx]),
                confidence=min(float(anomaly_scores[idx]) / float(threshold), 1.0),
                features={col: float(row[col]) for col in numeric_data.columns if col in row},
                context={'detection_method': 'multivariate_neural'},
                explanation="Neural network-based multivariate anomaly detection"
            )
            anomalies.append(anomaly)
        
        return anomalies
    
    def _fuse_anomaly_results(self, anomalies: List[AnomalyEvent]) -> List[AnomalyEvent]:
        """Fuse anomalies from different detection methods"""
        if self.config['fusion_method'] == 'simple_merge':
            return anomalies
        
        # Group by proximity in time and space
        fused_anomalies = []
        processed = set()
        
        for i, anomaly in enumerate(anomalies):
            if i in processed:
                continue
            
            # Find nearby anomalies
            nearby = [anomaly]
            for j, other in enumerate(anomalies[i+1:], start=i+1):
                if j in processed:
                    continue
                
                time_diff = abs((anomaly.timestamp - other.timestamp).total_seconds())
                spatial_diff = np.linalg.norm(anomaly.location - other.location)
                
                if time_diff < 3600 and spatial_diff < 5000:  # Within 1 hour and 5km
                    nearby.append(other)
                    processed.add(j)
            
            # Fuse nearby anomalies
            if len(nearby) > 1:
                fused_anomaly = self._fuse_anomaly_group(nearby)
                fused_anomalies.append(fused_anomaly)
            else:
                fused_anomalies.append(anomaly)
            
            processed.add(i)
        
        return fused_anomalies
    
    def _fuse_anomaly_group(self, anomalies: List[AnomalyEvent]) -> AnomalyEvent:
        """Fuse a group of nearby anomalies"""
        # Weight by confidence
        weights = np.array([a.confidence for a in anomalies])
        weights = weights / weights.sum()
        
        # Weighted average location
        locations = np.array([a.location for a in anomalies])
        avg_location = np.average(locations, axis=0, weights=weights)
        
        # Max severity and average confidence
        max_severity = max(a.severity_score for a in anomalies)
        avg_confidence = np.mean([a.confidence for a in anomalies])
        
        # Combine features
        combined_features = {}
        for anomaly in anomalies:
            for key, value in anomaly.features.items():
                if key not in combined_features:
                    combined_features[key] = []
                combined_features[key].append(value)
        
        # Average feature values
        averaged_features = {
            key: np.mean(values) for key, values in combined_features.items()
        }
        
        # Create fused anomaly
        base_anomaly = max(anomalies, key=lambda x: x.confidence)
        
        fused_anomaly = AnomalyEvent(
            timestamp=base_anomaly.timestamp,
            location=avg_location,
            anomaly_type=base_anomaly.anomaly_type,
            category=base_anomaly.category,
            severity_score=max_severity,
            confidence=avg_confidence,
            features=averaged_features,
            context={
                'fused_count': len(anomalies),
                'detection_methods': list(set(a.context.get('detection_method', 'unknown') for a in anomalies))
            },
            explanation=f"Fused anomaly from {len(anomalies)} detections",
            significance=max(a.significance for a in anomalies)
        )
        
        return fused_anomaly
    
    def _generate_summary(self, anomalies: List[AnomalyEvent]) -> Dict[str, Any]:
        """Generate summary statistics for detected anomalies"""
        if not anomalies:
            return {'total_anomalies': 0}
        
        # Basic counts
        summary = {
            'total_anomalies': len(anomalies),
            'by_type': defaultdict(int),
            'by_category': defaultdict(int),
            'severity_distribution': {
                'low': 0, 'medium': 0, 'high': 0, 'critical': 0
            },
            'temporal_range': {},
            'spatial_range': {}
        }
        
        # Categorize anomalies
        timestamps = []
        locations = []
        
        for anomaly in anomalies:
            # Type and category counts
            summary['by_type'][anomaly.anomaly_type.value] += 1
            summary['by_category'][anomaly.category.value] += 1
            
            # Severity distribution
            if anomaly.severity_score < 1.0:
                summary['severity_distribution']['low'] += 1
            elif anomaly.severity_score < 2.0:
                summary['severity_distribution']['medium'] += 1
            elif anomaly.severity_score < 4.0:
                summary['severity_distribution']['high'] += 1
            else:
                summary['severity_distribution']['critical'] += 1
            
            timestamps.append(anomaly.timestamp)
            locations.append(anomaly.location)
        
        # Temporal range
        if timestamps:
            summary['temporal_range'] = {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat(),
                'duration_hours': (max(timestamps) - min(timestamps)).total_seconds() / 3600
            }
        
        # Spatial range
        if locations:
            locations_array = np.array(locations)
            summary['spatial_range'] = {
                'centroid': np.mean(locations_array, axis=0).tolist(),
                'spread': np.std(locations_array, axis=0).tolist(),
                'bounding_box': {
                    'min': np.min(locations_array, axis=0).tolist(),
                    'max': np.max(locations_array, axis=0).tolist()
                }
            }
        
        return summary

# Example usage and testing
async def example_geological_anomaly_detection():
    """Example of using the geological anomaly detection system"""
    
    # Create synthetic geological data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'x': np.random.uniform(-50000, 50000, n_samples),  # UTM coordinates
        'y': np.random.uniform(-50000, 50000, n_samples),
        'z': np.random.uniform(0, 20000, n_samples),  # Depth in meters
        'magnitude': np.random.exponential(2.0, n_samples),  # Earthquake magnitudes
        'temperature': np.random.normal(25, 10, n_samples),  # Temperature
        'pressure': np.random.normal(101325, 1000, n_samples),  # Atmospheric pressure
        'deformation_rate': np.random.normal(0, 5, n_samples),  # mm/year
        'gas_concentration': np.random.lognormal(0, 1, n_samples)  # ppm
    })
    
    # Inject some anomalies
    anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
    data.loc[anomaly_indices, 'magnitude'] += np.random.uniform(2, 5, 50)  # Large earthquakes
    data.loc[anomaly_indices[:20], 'temperature'] += np.random.uniform(50, 200, 20)  # Thermal anomalies
    
    print(f"Created synthetic dataset with {len(data)} samples and {len(anomaly_indices)} injected anomalies")
    
    # Initialize detection system
    detection_system = GeologicalAnomalyDetectionSystem()
    detection_system.initialize(data)
    
    # Detect anomalies
    results = detection_system.detect_anomalies(data, context_features=['x', 'y'])
    
    print(f"\nDetection completed!")
    print(f"Processing time: {results['metadata']['processing_time']:.2f} seconds")
    print(f"Total anomalies detected: {results['summary']['total_anomalies']}")
    
    # Display summary
    print("\nAnomaly Summary:")
    for anomaly_type, count in results['summary']['by_type'].items():
        print(f"  {anomaly_type}: {count}")
    
    print("\nSeverity Distribution:")
    for severity, count in results['summary']['severity_distribution'].items():
        print(f"  {severity}: {count}")
    
    # Show top anomalies
    if results['anomalies']:
        top_anomalies = sorted(results['anomalies'], key=lambda x: x.severity_score, reverse=True)[:5]
        print(f"\nTop 5 anomalies:")
        for i, anomaly in enumerate(top_anomalies):
            print(f"  {i+1}. {anomaly.explanation} (severity: {anomaly.severity_score:.2f}, confidence: {anomaly.confidence:.2f})")
    
    # Show recommendations
    if results['recommendations']:
        print(f"\nRecommendations ({len(results['recommendations'])}):")
        for i, rec in enumerate(results['recommendations'][:3]):
            print(f"  {i+1}. {rec['recommendation']} (priority: {rec['priority']:.2f})")

if __name__ == "__main__":
    print("Advanced Geological Anomaly Detection System")
    print("=" * 50)
    
    # Check available libraries
    print(f"PyOD available: {PYOD_AVAILABLE}")
    print(f"Reinforcement Learning available: {RL_AVAILABLE}")
    print(f"Gym available: {GYM_AVAILABLE}")
    
    # Run example
    asyncio.run(example_geological_anomaly_detection())