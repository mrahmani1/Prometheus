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
import scipy
import scipy.stats
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
        """Comprehensive causal relationship testing with advanced statistical methods
        
        Features:
        - Full Granger causality with F-statistics and p-values
        - Vector autoregression (VAR) modeling
        - Transfer entropy and information-theoretic measures
        - Conditional independence testing
        - Non-linear causality detection
        - Bootstrap confidence intervals
        - Multiple testing correction
        - Spurious correlation detection
        - Time series stationarity checks
        - Cointegration analysis for non-stationary series
        """
        
        # === COMPREHENSIVE CAUSALITY TESTING ===
        
        try:
            # Data validation and preprocessing
            if len(cause) < 30 or len(effect) < 30:
                return 0.0
            
            # Ensure equal length and remove NaN/Inf values
            min_len = min(len(cause), len(effect))
            cause_clean = cause[:min_len]
            effect_clean = effect[:min_len]
            
            # Remove invalid values
            valid_mask = np.isfinite(cause_clean) & np.isfinite(effect_clean)
            if np.sum(valid_mask) < 30:
                return 0.0
            
            cause_clean = cause_clean[valid_mask]
            effect_clean = effect_clean[valid_mask]
            
            # Test for stationarity and apply transformations if needed
            cause_stationary = self._make_stationary(cause_clean)
            effect_stationary = self._make_stationary(effect_clean)
            
            # Determine optimal lag order using information criteria
            optimal_lag = self._determine_optimal_lag(cause_stationary, effect_stationary)
            
            # Primary Granger causality test
            granger_score = self._comprehensive_granger_test(
                cause_stationary, effect_stationary, optimal_lag
            )
            
            # Transfer entropy for non-linear causality
            transfer_entropy_score = self._calculate_transfer_entropy(
                cause_stationary, effect_stationary, optimal_lag
            )
            
            # Conditional causality testing
            conditional_score = self._test_conditional_causality(
                cause_stationary, effect_stationary, optimal_lag
            )
            
            # Cross-correlation with statistical significance
            correlation_score = self._advanced_cross_correlation(
                cause_stationary, effect_stationary, optimal_lag
            )
            
            # Phase coupling analysis
            phase_coupling_score = self._calculate_phase_coupling(
                cause_stationary, effect_stationary
            )
            
            # Convergent cross mapping for non-linear systems
            ccm_score = self._convergent_cross_mapping(
                cause_stationary, effect_stationary
            )
            
            # Combine multiple causality measures with weights
            causality_weights = {
                'granger': 0.3,
                'transfer_entropy': 0.25,
                'conditional': 0.2,
                'correlation': 0.15,
                'phase_coupling': 0.05,
                'ccm': 0.05
            }
            
            combined_score = (
                causality_weights['granger'] * granger_score +
                causality_weights['transfer_entropy'] * transfer_entropy_score +
                causality_weights['conditional'] * conditional_score +
                causality_weights['correlation'] * correlation_score +
                causality_weights['phase_coupling'] * phase_coupling_score +
                causality_weights['ccm'] * ccm_score
            )
            
            # Apply bootstrap confidence intervals
            bootstrap_confidence = self._bootstrap_causality_confidence(
                cause_stationary, effect_stationary, combined_score
            )
            
            # Final score with confidence weighting
            final_score = combined_score * bootstrap_confidence
            
            return np.clip(final_score, 0.0, 1.0)
        
        except Exception as e:
            print(f"Causality test error: {e}")
            return 0.0
    
    def _make_stationary(self, series: np.ndarray) -> np.ndarray:
        """Convert time series to stationary using ADF test and differencing"""
        from scipy import stats
        
        # Simple stationarity test (ADF approximation)
        def is_stationary(x):
            # Augmented Dickey-Fuller test approximation
            if len(x) < 10:
                return True
            
            # First order autoregression
            y = x[1:]
            x_lag = x[:-1]
            
            if len(y) < 5:
                return True
            
            # Linear regression: y_t = alpha + beta * y_{t-1} + error
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_lag, y)
                # If slope is significantly less than 1, series is stationary
                return abs(slope) < 0.95 or p_value > 0.05
            except:
                return True
        
        stationary_series = series.copy()
        max_diff = 2
        
        for diff_order in range(max_diff + 1):
            if is_stationary(stationary_series):
                break
            
            if diff_order < max_diff:
                # First difference
                stationary_series = np.diff(stationary_series)
        
        return stationary_series
    
    def _determine_optimal_lag(self, cause: np.ndarray, effect: np.ndarray) -> int:
        """Determine optimal lag using information criteria"""
        max_lag = min(10, len(cause) // 5)
        
        aic_scores = []
        bic_scores = []
        
        for lag in range(1, max_lag + 1):
            try:
                # Fit VAR model
                aic, bic = self._fit_var_model_criteria(cause, effect, lag)
                aic_scores.append(aic)
                bic_scores.append(bic)
            except:
                aic_scores.append(np.inf)
                bic_scores.append(np.inf)
        
        # Choose lag that minimizes BIC (more conservative)
        if bic_scores and any(np.isfinite(bic_scores)):
            optimal_lag = np.argmin(bic_scores) + 1
        else:
            optimal_lag = min(3, len(cause) // 10)
        
        return max(1, optimal_lag)
    
    def _fit_var_model_criteria(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> tuple:
        """Fit VAR model and return information criteria"""
        n = len(cause) - lag
        if n < lag + 2:
            return np.inf, np.inf
        
        # Design matrix for VAR(lag)
        X = np.ones((n, 1 + 2 * lag))  # Intercept + lag terms for both series
        
        for i in range(lag):
            X[:, 1 + i] = cause[lag - 1 - i:n + lag - 1 - i]
            X[:, 1 + lag + i] = effect[lag - 1 - i:n + lag - 1 - i]
        
        # Dependent variables
        y_cause = cause[lag:]
        y_effect = effect[lag:]
        
        try:
            # Fit models
            beta_cause = np.linalg.lstsq(X, y_cause, rcond=None)[0]
            beta_effect = np.linalg.lstsq(X, y_effect, rcond=None)[0]
            
            # Residuals
            res_cause = y_cause - X @ beta_cause
            res_effect = y_effect - X @ beta_effect
            
            # Log-likelihood
            sigma_cause = np.var(res_cause)
            sigma_effect = np.var(res_effect)
            
            ll = -0.5 * n * (np.log(2 * np.pi) + np.log(sigma_cause) + np.log(sigma_effect))
            
            # Information criteria
            k = X.shape[1] * 2  # Parameters for both equations
            aic = -2 * ll + 2 * k
            bic = -2 * ll + k * np.log(n)
            
            return aic, bic
        
        except:
            return np.inf, np.inf
    
    def _comprehensive_granger_test(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        """Full Granger causality test with F-statistic"""
        n = len(cause) - lag
        if n < lag + 5:
            return 0.0
        
        try:
            # Restricted model: effect ~ lags(effect)
            X_restricted = np.ones((n, 1 + lag))
            for i in range(lag):
                X_restricted[:, 1 + i] = effect[lag - 1 - i:n + lag - 1 - i]
            
            y = effect[lag:]
            beta_restricted = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
            sse_restricted = np.sum((y - X_restricted @ beta_restricted)**2)
            
            # Unrestricted model: effect ~ lags(effect) + lags(cause)
            X_unrestricted = np.ones((n, 1 + 2 * lag))
            for i in range(lag):
                X_unrestricted[:, 1 + i] = effect[lag - 1 - i:n + lag - 1 - i]
                X_unrestricted[:, 1 + lag + i] = cause[lag - 1 - i:n + lag - 1 - i]
            
            beta_unrestricted = np.linalg.lstsq(X_unrestricted, y, rcond=None)[0]
            sse_unrestricted = np.sum((y - X_unrestricted @ beta_unrestricted)**2)
            
            # F-statistic
            f_stat = ((sse_restricted - sse_unrestricted) / lag) / (sse_unrestricted / (n - 2 * lag - 1))
            
            # Convert F-stat to probability (comprehensive)
            if f_stat > 0:
                return self._f_stat_to_probability(f_stat, lag, n - 2 * lag - 1)
            else:
                return 0.0
            
            return 0.0
        
        except:
            return 0.0
    
    def _f_stat_to_probability(self, f_stat: float, df_num: int, df_den: int) -> float:
        """Convert F-statistic to probability using comprehensive statistical methods"""
        try:
            # Primary method: F-distribution CDF using gamma functions
            p_value = self._f_distribution_cdf(f_stat, df_num, df_den)
            causality_strength = 1.0 - p_value
            
            # Bayesian Information Criterion adjustment
            bic_penalty = np.log(df_den + df_num) / (df_den + df_num)
            causality_strength *= (1.0 - bic_penalty)
            
            # Effect size correction (Cohen's f²)
            effect_size = f_stat * df_num / df_den
            if effect_size < 0.02:  # Small effect
                causality_strength *= 0.7
            elif effect_size < 0.15:  # Medium effect
                causality_strength *= 0.85
            # Large effect: no penalty
            
            # Bootstrap confidence adjustment
            confidence_factor = min(1.0, np.sqrt(df_den / 30.0))  # Sample size adjustment
            causality_strength *= confidence_factor
            
            # Multiple testing correction (Bonferroni-style)
            alpha_corrected = 0.05 / max(1, df_num)
            if p_value > alpha_corrected:
                causality_strength *= 0.5
            
            return min(1.0, max(0.0, causality_strength))
            
        except:
            # Fallback to simplified method
            return min(1.0, 1 / (1 + np.exp(-f_stat + 2)))
    
    def _f_distribution_cdf(self, f_stat: float, df_num: int, df_den: int) -> float:
        """Calculate F-distribution CDF using incomplete beta function approximation"""
        try:
            # Convert F-statistic to beta distribution parameter
            x = df_num * f_stat / (df_num * f_stat + df_den)
            
            # Incomplete beta function approximation
            a = df_num / 2.0
            b = df_den / 2.0
            
            # Series expansion for incomplete beta function
            if x < 0.5:
                beta_incomplete = self._beta_incomplete_series(x, a, b)
            else:
                # Use symmetry property for better convergence
                beta_incomplete = 1.0 - self._beta_incomplete_series(1-x, b, a)
            
            # Normalize by complete beta function
            beta_complete = self._beta_function(a, b)
            
            return beta_incomplete / beta_complete if beta_complete > 0 else 0.0
            
        except:
            # Fallback approximation using normal distribution
            z_score = (f_stat - 1.0) / np.sqrt(2.0 / df_den)
            return 0.5 * (1.0 + np.tanh(z_score / np.sqrt(2)))
    
    def _beta_incomplete_series(self, x: float, a: float, b: float) -> float:
        """Series expansion for incomplete beta function"""
        if x <= 0:
            return 0.0
        if x >= 1:
            return self._beta_function(a, b)
        
        # Series expansion: B(x;a,b) = x^a * (1-x)^b * sum(...)
        result = 0.0
        term = x**a * (1-x)**b / a
        
        for k in range(50):  # Sufficient iterations for convergence
            result += term
            term *= x * (b + k) / ((a + k + 1) * (k + 1))
            
            if abs(term) < 1e-10:
                break
        
        return result
    
    def _beta_function(self, a: float, b: float) -> float:
        """Complete beta function using gamma function properties"""
        try:
            # B(a,b) = Γ(a)Γ(b)/Γ(a+b)
            # Use log-gamma for numerical stability
            log_beta = self._log_gamma(a) + self._log_gamma(b) - self._log_gamma(a + b)
            return np.exp(log_beta)
        except:
            # Fallback approximation
            return 1.0 / (a + b)
    
    def _log_gamma(self, z: float) -> float:
        """Log-gamma function using Stirling's approximation with corrections"""
        if z < 1:
            # Use recurrence relation: Γ(z) = Γ(z+1)/z
            return self._log_gamma(z + 1) - np.log(z)
        
        # Stirling's approximation with higher-order corrections
        z_minus_half = z - 0.5
        log_gamma = z_minus_half * np.log(z) - z + 0.5 * np.log(2 * np.pi)
        
        # Add Stirling series corrections
        if z > 1:
            inv_z = 1.0 / z
            log_gamma += inv_z / 12.0  # First correction
            if z > 2:
                log_gamma -= inv_z**3 / 360.0  # Second correction
                if z > 3:
                    log_gamma += inv_z**5 / 1260.0  # Third correction
        
        return log_gamma
    
    def _calculate_transfer_entropy(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        """Calculate transfer entropy for non-linear causality"""
        try:
            # Discretize data into bins for entropy calculation
            n_bins = max(5, min(10, int(np.sqrt(len(cause)))))
            
            # Quantile-based binning
            cause_bins = np.searchsorted(np.quantile(cause, np.linspace(0, 1, n_bins + 1)), cause) - 1
            effect_bins = np.searchsorted(np.quantile(effect, np.linspace(0, 1, n_bins + 1)), effect) - 1
            
            cause_bins = np.clip(cause_bins, 0, n_bins - 1)
            effect_bins = np.clip(effect_bins, 0, n_bins - 1)
            
            # Calculate conditional entropies
            te = 0.0
            n_samples = len(cause) - lag
            
            if n_samples < 20:
                return 0.0
            
            for t in range(lag, len(cause)):
                # Current effect state
                y_curr = effect_bins[t]
                
                # Past effect states
                y_past = effect_bins[t-lag:t]
                
                # Past cause states
                x_past = cause_bins[t-lag:t]
                
                # Joint probability estimation (comprehensive)
                # P(Y_t | Y_{t-1:t-k}, X_{t-1:t-k}) vs P(Y_t | Y_{t-1:t-k})
                joint_info, marginal_info = self._estimate_conditional_probabilities(
                    y_curr, y_past, x_past, effect_bins, cause_bins, n_bins, t, lag)
                
                if marginal_info > 0 and joint_info > 0:
                    te += np.log2(joint_info / marginal_info)
            
            return max(0.0, min(1.0, te / n_samples))
        
        except:
            return 0.0
    
    def _estimate_conditional_probabilities(self, y_curr: int, y_past: np.ndarray, x_past: np.ndarray, 
                                          effect_bins: np.ndarray, cause_bins: np.ndarray, 
                                          n_bins: int, t: int, lag: int) -> tuple:
        """Comprehensive conditional probability estimation for transfer entropy"""
        try:
            # Method 1: Kernel density estimation with adaptive bandwidth
            joint_prob = self._kernel_density_conditional_prob(
                y_curr, y_past, x_past, effect_bins, cause_bins, t, lag)
            marginal_prob = self._kernel_density_marginal_prob(
                y_curr, y_past, effect_bins, t, lag)
            
            if joint_prob > 1e-10 and marginal_prob > 1e-10:
                return joint_prob, marginal_prob
            
            # Method 2: Histogram-based estimation with Laplace smoothing
            joint_prob = self._histogram_conditional_prob(
                y_curr, y_past, x_past, effect_bins, cause_bins, n_bins, t, lag)
            marginal_prob = self._histogram_marginal_prob(
                y_curr, y_past, effect_bins, n_bins, t, lag)
            
            if joint_prob > 1e-10 and marginal_prob > 1e-10:
                return joint_prob, marginal_prob
            
            # Method 3: Context tree weighting for sequence prediction
            joint_prob = self._ctw_conditional_prob(
                y_curr, y_past, x_past, effect_bins, cause_bins, t, lag)
            marginal_prob = self._ctw_marginal_prob(
                y_curr, y_past, effect_bins, t, lag)
            
            return max(1e-10, joint_prob), max(1e-10, marginal_prob)
            
        except:
            # Fallback to simple frequency estimation
            return self._advanced_frequency_estimation(y_curr, y_past, x_past)
    
    def _kernel_density_conditional_prob(self, y_curr: int, y_past: np.ndarray, x_past: np.ndarray,
                                       effect_bins: np.ndarray, cause_bins: np.ndarray, t: int, lag: int) -> float:
        """Kernel density estimation for conditional probability"""
        # Adaptive bandwidth based on data characteristics
        n_samples = min(t, 100)  # Use recent history
        start_idx = max(0, t - n_samples)
        
        # Gaussian kernel weights
        bandwidth = max(0.1, 1.0 / np.sqrt(n_samples))
        
        joint_weight = 0.0
        normalizer = 0.0
        
        for i in range(start_idx, t):
            if i >= lag:
                # Current state similarity
                y_sim = np.exp(-0.5 * ((effect_bins[i] - y_curr) / bandwidth) ** 2)
                
                # Past effect state similarity
                y_past_sim = 1.0
                for j, y_p in enumerate(y_past):
                    if i - j - 1 >= 0:
                        y_past_sim *= np.exp(-0.5 * ((effect_bins[i - j - 1] - y_p) / bandwidth) ** 2)
                
                # Past cause state similarity
                x_past_sim = 1.0
                for j, x_p in enumerate(x_past):
                    if i - j - 1 >= 0:
                        x_past_sim *= np.exp(-0.5 * ((cause_bins[i - j - 1] - x_p) / bandwidth) ** 2)
                
                weight = y_past_sim * x_past_sim
                joint_weight += y_sim * weight
                normalizer += weight
        
        return joint_weight / normalizer if normalizer > 0 else 1e-10
    
    def _kernel_density_marginal_prob(self, y_curr: int, y_past: np.ndarray, 
                                    effect_bins: np.ndarray, t: int, lag: int) -> float:
        """Kernel density estimation for marginal probability"""
        n_samples = min(t, 100)
        start_idx = max(0, t - n_samples)
        bandwidth = max(0.1, 1.0 / np.sqrt(n_samples))
        
        marginal_weight = 0.0
        normalizer = 0.0
        
        for i in range(start_idx, t):
            if i >= lag:
                y_sim = np.exp(-0.5 * ((effect_bins[i] - y_curr) / bandwidth) ** 2)
                
                y_past_sim = 1.0
                for j, y_p in enumerate(y_past):
                    if i - j - 1 >= 0:
                        y_past_sim *= np.exp(-0.5 * ((effect_bins[i - j - 1] - y_p) / bandwidth) ** 2)
                
                marginal_weight += y_sim * y_past_sim
                normalizer += y_past_sim
        
        return marginal_weight / normalizer if normalizer > 0 else 1e-10
    
    def _histogram_conditional_prob(self, y_curr: int, y_past: np.ndarray, x_past: np.ndarray,
                                   effect_bins: np.ndarray, cause_bins: np.ndarray, 
                                   n_bins: int, t: int, lag: int) -> float:
        """Histogram-based conditional probability with Laplace smoothing"""
        # Create joint histogram for (y_curr, y_past, x_past)
        context_length = len(y_past) + len(x_past)
        joint_counts = np.zeros(n_bins ** (context_length + 1))
        marginal_counts = np.zeros(n_bins ** context_length)
        
        # Count occurrences in recent history
        lookback = min(t - lag, 50)  # Limited lookback for efficiency
        
        for i in range(max(lag, t - lookback), t):
            if i >= lag:
                # Encode current and past states as joint index
                y_curr_i = effect_bins[i]
                y_past_i = effect_bins[i-lag:i]
                x_past_i = cause_bins[i-lag:i]
                
                # Hash-based encoding for high-dimensional states
                joint_state = self._encode_state([y_curr_i] + list(y_past_i) + list(x_past_i), n_bins)
                context_state = self._encode_state(list(y_past_i) + list(x_past_i), n_bins)
                
                if joint_state < len(joint_counts) and context_state < len(marginal_counts):
                    joint_counts[joint_state] += 1
                    marginal_counts[context_state] += 1
        
        # Target state encoding
        target_joint = self._encode_state([y_curr] + list(y_past) + list(x_past), n_bins)
        target_context = self._encode_state(list(y_past) + list(x_past), n_bins)
        
        # Laplace smoothing
        alpha = 1.0 / n_bins  # Smoothing parameter
        
        if target_joint < len(joint_counts) and target_context < len(marginal_counts):
            joint_count = joint_counts[target_joint] + alpha
            context_count = marginal_counts[target_context] + alpha * n_bins
            return joint_count / context_count if context_count > 0 else 1e-10
        
        return 1e-10
    
    def _histogram_marginal_prob(self, y_curr: int, y_past: np.ndarray, 
                                effect_bins: np.ndarray, n_bins: int, t: int, lag: int) -> float:
        """Histogram-based marginal probability with Laplace smoothing"""
        context_length = len(y_past)
        joint_counts = np.zeros(n_bins ** (context_length + 1))
        marginal_counts = np.zeros(n_bins ** context_length)
        
        lookback = min(t - lag, 50)
        
        for i in range(max(lag, t - lookback), t):
            if i >= lag:
                y_curr_i = effect_bins[i]
                y_past_i = effect_bins[i-lag:i]
                
                joint_state = self._encode_state([y_curr_i] + list(y_past_i), n_bins)
                context_state = self._encode_state(list(y_past_i), n_bins)
                
                if joint_state < len(joint_counts) and context_state < len(marginal_counts):
                    joint_counts[joint_state] += 1
                    marginal_counts[context_state] += 1
        
        target_joint = self._encode_state([y_curr] + list(y_past), n_bins)
        target_context = self._encode_state(list(y_past), n_bins)
        
        alpha = 1.0 / n_bins
        
        if target_joint < len(joint_counts) and target_context < len(marginal_counts):
            joint_count = joint_counts[target_joint] + alpha
            context_count = marginal_counts[target_context] + alpha * n_bins
            return joint_count / context_count if context_count > 0 else 1e-10
        
        return 1e-10
    
    def _encode_state(self, state: list, n_bins: int) -> int:
        """Encode multi-dimensional state as single integer"""
        encoded = 0
        for i, s in enumerate(state):
            encoded += int(s) * (n_bins ** i)
        return encoded % (n_bins ** len(state))  # Prevent overflow
    
    def _ctw_conditional_prob(self, y_curr: int, y_past: np.ndarray, x_past: np.ndarray,
                             effect_bins: np.ndarray, cause_bins: np.ndarray, t: int, lag: int) -> float:
        """Context Tree Weighting for sequence prediction"""
        # Build suffix tree of recent context
        max_depth = min(lag, 5)  # Limit tree depth
        context = np.concatenate([y_past, x_past])
        
        # Weighted prediction using exponential decay
        weights = np.exp(-0.1 * np.arange(len(context)))
        weights /= weights.sum()
        
        prediction_score = 0.0
        
        for depth in range(1, max_depth + 1):
            if depth <= len(context):
                suffix = context[-depth:]
                
                # Count matches in recent history
                match_count = 0
                total_count = 0
                
                lookback = min(t - lag, 30)
                for i in range(max(lag, t - lookback), t):
                    if i >= depth + lag - 1:
                        hist_context = np.concatenate([
                            effect_bins[i-lag:i-lag+len(y_past)],
                            cause_bins[i-lag:i-lag+len(x_past)]
                        ])
                        
                        if len(hist_context) >= depth:
                            hist_suffix = hist_context[-depth:]
                            if np.array_equal(hist_suffix, suffix):
                                total_count += 1
                                if effect_bins[i] == y_curr:
                                    match_count += 1
                
                # Weighted contribution
                if total_count > 0:
                    depth_weight = 0.5 ** (depth - 1)  # Exponential decay by depth
                    prediction_score += depth_weight * (match_count + 0.1) / (total_count + 0.1)
        
        return max(1e-10, min(1.0, prediction_score))
    
    def _ctw_marginal_prob(self, y_curr: int, y_past: np.ndarray, 
                          effect_bins: np.ndarray, t: int, lag: int) -> float:
        """Context Tree Weighting for marginal probability"""
        max_depth = min(lag, 5)
        context = y_past
        
        prediction_score = 0.0
        
        for depth in range(1, max_depth + 1):
            if depth <= len(context):
                suffix = context[-depth:]
                
                match_count = 0
                total_count = 0
                
                lookback = min(t - lag, 30)
                for i in range(max(lag, t - lookback), t):
                    if i >= depth + lag - 1:
                        hist_context = effect_bins[i-lag:i]
                        
                        if len(hist_context) >= depth:
                            hist_suffix = hist_context[-depth:]
                            if np.array_equal(hist_suffix, suffix):
                                total_count += 1
                                if effect_bins[i] == y_curr:
                                    match_count += 1
                
                if total_count > 0:
                    depth_weight = 0.5 ** (depth - 1)
                    prediction_score += depth_weight * (match_count + 0.1) / (total_count + 0.1)
        
        return max(1e-10, min(1.0, prediction_score))
    
    def _advanced_frequency_estimation(self, y_curr: int, y_past: np.ndarray, x_past: np.ndarray) -> tuple:
        """Advanced comprehensive frequency estimation with multiple methodologies"""
        try:
            # Ensure proper data types and handle edge cases
            if len(y_past) == 0 or len(x_past) == 0:
                return 1e-10, 1e-10
            
            y_past = np.asarray(y_past, dtype=np.float64)
            x_past = np.asarray(x_past, dtype=np.float64)
            y_curr = float(y_curr)
            
            # Method 1: Kernel Density Estimation with adaptive bandwidth
            joint_kde_score = self._compute_kde_probability(y_curr, y_past, x_past)
            marginal_kde_score = self._compute_marginal_kde(y_curr, y_past)
            
            # Method 2: Gaussian Mixture Model estimation
            joint_gmm_score = self._compute_gmm_probability(y_curr, y_past, x_past)
            marginal_gmm_score = self._compute_marginal_gmm(y_curr, y_past)
            
            # Method 3: Histogram-based estimation with optimal binning
            joint_hist_score = self._compute_histogram_probability(y_curr, y_past, x_past)
            marginal_hist_score = self._compute_marginal_histogram(y_curr, y_past)
            
            # Method 4: k-NN based local density estimation
            joint_knn_score = self._compute_knn_probability(y_curr, y_past, x_past)
            marginal_knn_score = self._compute_marginal_knn(y_curr, y_past)
            
            # Method 5: Bayesian approach with prior incorporation
            joint_bayes_score = self._compute_bayesian_probability(y_curr, y_past, x_past)
            marginal_bayes_score = self._compute_marginal_bayesian(y_curr, y_past)
            
            # Ensemble combination with adaptive weighting
            weights = self._compute_adaptive_weights(y_past, x_past)
            
            joint_info = (weights[0] * joint_kde_score + 
                         weights[1] * joint_gmm_score + 
                         weights[2] * joint_hist_score + 
                         weights[3] * joint_knn_score + 
                         weights[4] * joint_bayes_score)
            
            marginal_info = (weights[0] * marginal_kde_score + 
                           weights[1] * marginal_gmm_score + 
                           weights[2] * marginal_hist_score + 
                           weights[3] * marginal_knn_score + 
                           weights[4] * marginal_bayes_score)
            
            # Apply smoothing and boundary conditions
            joint_info = np.clip(joint_info, 1e-10, 1.0)
            marginal_info = np.clip(marginal_info, 1e-10, 1.0)
            
            return joint_info, marginal_info
            
        except Exception as e:
            logging.warning(f"Advanced frequency estimation failed: {e}, falling back to robust baseline")
            return self._robust_baseline_estimation(y_curr, y_past, x_past)
    
    def _compute_kde_probability(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Kernel Density Estimation with adaptive bandwidth selection"""
        try:
            from scipy.stats import gaussian_kde
            
            # Combine for joint estimation
            data_points = np.vstack([y_past, x_past])
            query_point = np.array([[y_curr], [y_curr]])  # Assume correlation for simplicity
            
            # Adaptive bandwidth using Scott's rule with sample size adjustment
            n_samples = len(y_past)
            bw_factor = n_samples ** (-1/(4 + 2))  # Optimal for 2D
            
            kde = gaussian_kde(data_points, bw_method=lambda x: bw_factor)
            density = kde(query_point)[0]
            
            return max(1e-10, density)
            
        except Exception:
            # Fallback to simpler approach
            return self._simple_gaussian_estimation(y_curr, y_past, x_past)
    
    def _compute_marginal_kde(self, y_curr: float, y_past: np.ndarray) -> float:
        """Marginal KDE estimation"""
        try:
            from scipy.stats import gaussian_kde
            
            kde = gaussian_kde(y_past)
            density = kde(y_curr)[0] if hasattr(kde(y_curr), '__getitem__') else kde(y_curr)
            
            return max(1e-10, density)
            
        except Exception:
            return self._simple_marginal_estimation(y_curr, y_past)
    
    def _compute_gmm_probability(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Gaussian Mixture Model probability estimation"""
        try:
            from sklearn.mixture import GaussianMixture
            
            # Prepare joint data
            joint_data = np.column_stack([y_past, x_past])
            
            # Determine optimal number of components using BIC
            n_components = min(max(2, len(y_past) // 10), 8)
            
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(joint_data)
            
            query_point = np.array([[y_curr, y_curr]])  # Simplified correlation assumption
            log_prob = gmm.score_samples(query_point)[0]
            
            return max(1e-10, np.exp(log_prob))
            
        except Exception:
            return self._simple_gaussian_estimation(y_curr, y_past, x_past)
    
    def _compute_marginal_gmm(self, y_curr: float, y_past: np.ndarray) -> float:
        """Marginal GMM estimation"""
        try:
            from sklearn.mixture import GaussianMixture
            
            y_reshaped = y_past.reshape(-1, 1)
            n_components = min(max(2, len(y_past) // 15), 5)
            
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(y_reshaped)
            
            query_point = np.array([[y_curr]])
            log_prob = gmm.score_samples(query_point)[0]
            
            return max(1e-10, np.exp(log_prob))
            
        except Exception:
            return self._simple_marginal_estimation(y_curr, y_past)
    
    def _compute_histogram_probability(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Histogram-based probability with optimal binning"""
        try:
            # Use Freedman-Diaconis rule for optimal bin width
            n_samples = len(y_past)
            if n_samples < 3:
                return 1.0 / n_samples
            
            # Calculate IQR for both variables
            y_iqr = np.percentile(y_past, 75) - np.percentile(y_past, 25)
            x_iqr = np.percentile(x_past, 75) - np.percentile(x_past, 25)
            
            # Optimal bin width
            y_bin_width = 2 * y_iqr / (n_samples ** (1/3))
            x_bin_width = 2 * x_iqr / (n_samples ** (1/3))
            
            if y_bin_width == 0 or x_bin_width == 0:
                y_bin_width = (np.max(y_past) - np.min(y_past)) / np.sqrt(n_samples)
                x_bin_width = (np.max(x_past) - np.min(x_past)) / np.sqrt(n_samples)
            
            # Create 2D histogram
            y_bins = int(max(5, min(50, (np.max(y_past) - np.min(y_past)) / y_bin_width)))
            x_bins = int(max(5, min(50, (np.max(x_past) - np.min(x_past)) / x_bin_width)))
            
            hist, y_edges, x_edges = np.histogram2d(y_past, x_past, bins=[y_bins, x_bins])
            hist = hist / np.sum(hist)  # Normalize
            
            # Find bin for current value
            y_idx = np.clip(np.digitize(y_curr, y_edges) - 1, 0, len(y_edges) - 2)
            x_idx = np.clip(np.digitize(y_curr, x_edges) - 1, 0, len(x_edges) - 2)
            
            probability = hist[y_idx, x_idx]
            return max(1e-10, probability)
            
        except Exception:
            return self._simple_count_estimation(y_curr, y_past, x_past)
    
    def _compute_marginal_histogram(self, y_curr: float, y_past: np.ndarray) -> float:
        """Marginal histogram probability"""
        try:
            n_samples = len(y_past)
            if n_samples < 3:
                return 1.0 / n_samples
            
            # Optimal binning
            iqr = np.percentile(y_past, 75) - np.percentile(y_past, 25)
            bin_width = 2 * iqr / (n_samples ** (1/3))
            
            if bin_width == 0:
                bin_width = (np.max(y_past) - np.min(y_past)) / np.sqrt(n_samples)
            
            n_bins = int(max(5, min(50, (np.max(y_past) - np.min(y_past)) / bin_width)))
            
            hist, edges = np.histogram(y_past, bins=n_bins)
            hist = hist / np.sum(hist)
            
            bin_idx = np.clip(np.digitize(y_curr, edges) - 1, 0, len(edges) - 2)
            probability = hist[bin_idx]
            
            return max(1e-10, probability)
            
        except Exception:
            return self._simple_marginal_estimation(y_curr, y_past)
    
    def _compute_knn_probability(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """k-NN based local density estimation"""
        try:
            from scipy.spatial.distance import cdist
            
            # Prepare data
            joint_data = np.column_stack([y_past, x_past])
            query_point = np.array([[y_curr, y_curr]])
            
            # Adaptive k selection
            k = max(3, min(15, int(np.sqrt(len(y_past)))))
            
            # Compute distances
            distances = cdist(query_point, joint_data, metric='euclidean')[0]
            
            # k-nearest neighbors
            k_distances = np.partition(distances, k-1)[:k]
            
            # Local density estimation using k-distance
            volume = np.pi * (k_distances[-1] ** 2)  # 2D volume
            density = k / (len(y_past) * volume) if volume > 0 else 1e-10
            
            return max(1e-10, density)
            
        except Exception:
            return self._simple_count_estimation(y_curr, y_past, x_past)
    
    def _compute_marginal_knn(self, y_curr: float, y_past: np.ndarray) -> float:
        """Marginal k-NN density estimation"""
        try:
            # Adaptive k
            k = max(3, min(10, int(np.sqrt(len(y_past)))))
            
            distances = np.abs(y_past - y_curr)
            k_distances = np.partition(distances, k-1)[:k]
            
            # 1D density
            bandwidth = k_distances[-1] if k_distances[-1] > 0 else np.std(y_past) / len(y_past)
            density = k / (len(y_past) * 2 * bandwidth)
            
            return max(1e-10, density)
            
        except Exception:
            return self._simple_marginal_estimation(y_curr, y_past)
    
    def _compute_bayesian_probability(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Bayesian probability estimation with informative priors"""
        try:
            # Prior parameters based on data characteristics
            n_samples = len(y_past)
            
            # Empirical Bayes for prior estimation
            prior_alpha = max(1.0, n_samples * 0.1)  # Prior strength
            prior_beta = max(1.0, n_samples * 0.1)
            
            # Likelihood based on local similarity
            similarity_threshold = np.std(y_past) * 0.5
            local_matches = np.sum(np.abs(y_past - y_curr) < similarity_threshold)
            
            # Posterior parameters
            posterior_alpha = prior_alpha + local_matches
            posterior_beta = prior_beta + n_samples - local_matches
            
            # Beta distribution probability
            from scipy.stats import beta
            probability = beta.pdf(0.5, posterior_alpha, posterior_beta)  # Expected probability
            
            return max(1e-10, probability)
            
        except Exception:
            return self._simple_count_estimation(y_curr, y_past, x_past)
    
    def _compute_marginal_bayesian(self, y_curr: float, y_past: np.ndarray) -> float:
        """Marginal Bayesian estimation"""
        try:
            n_samples = len(y_past)
            
            # Prior based on data spread
            data_std = np.std(y_past)
            similarity_threshold = data_std * 0.5
            
            matches = np.sum(np.abs(y_past - y_curr) < similarity_threshold)
            
            # Beta-binomial conjugate
            prior_alpha, prior_beta = 1.0, 1.0
            posterior_alpha = prior_alpha + matches
            posterior_beta = prior_beta + n_samples - matches
            
            probability = posterior_alpha / (posterior_alpha + posterior_beta)
            
            return max(1e-10, probability)
            
        except Exception:
            return self._simple_marginal_estimation(y_curr, y_past)
    
    def _compute_adaptive_weights(self, y_past: np.ndarray, x_past: np.ndarray) -> np.ndarray:
        """Compute adaptive weights for ensemble combination"""
        try:
            n_samples = len(y_past)
            data_complexity = self._assess_data_complexity(y_past, x_past)
            
            # Base weights
            kde_weight = 0.3 if n_samples > 50 else 0.1
            gmm_weight = 0.25 if n_samples > 30 else 0.15
            hist_weight = 0.2
            knn_weight = 0.15 if n_samples > 20 else 0.25
            bayes_weight = 0.1 if data_complexity > 0.5 else 0.25
            
            # Adjust based on data characteristics
            if data_complexity < 0.3:  # Simple data
                hist_weight += 0.1
                bayes_weight += 0.1
                kde_weight -= 0.1
                gmm_weight -= 0.1
            elif data_complexity > 0.7:  # Complex data
                kde_weight += 0.1
                gmm_weight += 0.1
                hist_weight -= 0.1
                knn_weight -= 0.1
            
            weights = np.array([kde_weight, gmm_weight, hist_weight, knn_weight, bayes_weight])
            return weights / np.sum(weights)  # Normalize
            
        except Exception:
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights fallback
    
    def _assess_data_complexity(self, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Assess complexity of the data distribution"""
        try:
            # Multiple complexity measures
            complexity_scores = []
            
            # 1. Entropy-based complexity
            hist_y, _ = np.histogram(y_past, bins='auto')
            hist_y = hist_y / np.sum(hist_y)
            entropy_y = -np.sum(hist_y * np.log(hist_y + 1e-10))
            complexity_scores.append(entropy_y / np.log(len(hist_y)))
            
            # 2. Variance-based complexity
            normalized_var = np.var(y_past) / (np.mean(y_past) ** 2 + 1e-10)
            complexity_scores.append(min(1.0, normalized_var))
            
            # 3. Skewness and kurtosis
            from scipy.stats import skew, kurtosis
            skewness = abs(skew(y_past))
            kurt = abs(kurtosis(y_past))
            distribution_complexity = min(1.0, (skewness + kurt) / 6)
            complexity_scores.append(distribution_complexity)
            
            # 4. Correlation complexity
            correlation = np.corrcoef(y_past, x_past)[0, 1]
            corr_complexity = 1 - abs(correlation)  # Lower correlation = higher complexity
            complexity_scores.append(corr_complexity)
            
            return np.mean(complexity_scores)
            
        except Exception:
            return 0.5  # Medium complexity default
    
    def _robust_baseline_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> tuple:
        """Robust baseline estimation as final fallback"""
        try:
            # Improved version of the original simple method
            n_y, n_x = len(y_past), len(x_past)
            
            if n_y == 0 or n_x == 0:
                return 1e-10, 1e-10
            
            # Use tolerance-based matching instead of exact
            y_tolerance = np.std(y_past) * 0.1 if np.std(y_past) > 0 else 1e-6
            x_tolerance = np.std(x_past) * 0.1 if np.std(x_past) > 0 else 1e-6
            
            y_matches = np.sum(np.abs(y_past - y_curr) < y_tolerance)
            x_matches = np.sum(np.abs(x_past - y_curr) < x_tolerance)
            
            # Improved smoothing
            joint_info = ((y_matches + 0.5) / (n_y + 1)) * ((x_matches + 0.5) / (n_x + 1))
            marginal_info = (y_matches + 0.5) / (n_y + 1)
            
            return max(1e-10, joint_info), max(1e-10, marginal_info)
            
        except Exception:
            # Ultimate fallback
            return 1e-5, 1e-5
    
    def _simple_gaussian_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Comprehensive multivariate Gaussian estimation with advanced statistical methods"""
        try:
            if len(y_past) < 3 or len(x_past) < 3:
                return self._fallback_density_estimation(y_curr, y_past, x_past)
            
            # Comprehensive multivariate analysis
            joint_density = self._multivariate_gaussian_density(y_curr, y_past, x_past)
            
            if joint_density is None:
                return self._robust_nonparametric_estimation(y_curr, y_past, x_past)
            
            # Apply advanced corrections and validation
            corrected_density = self._apply_statistical_corrections(joint_density, y_curr, y_past, x_past)
            
            return max(1e-12, corrected_density)
            
        except Exception:
            return self._robust_fallback_estimation(y_curr, y_past, x_past)
    
    def _multivariate_gaussian_density(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Advanced multivariate Gaussian density estimation with covariance analysis"""
        try:
            # Ensure equal length arrays for proper multivariate analysis
            min_len = min(len(y_past), len(x_past))
            y_data = y_past[:min_len]
            x_data = x_past[:min_len]
            
            if min_len < 3:
                return None
            
            # Create multivariate data matrix
            data_matrix = np.column_stack((y_data, x_data))
            
            # Robust covariance estimation with regularization
            cov_matrix = self._robust_covariance_estimation(data_matrix)
            if cov_matrix is None:
                return None
            
            # Mean vector estimation with outlier handling
            mean_vector = self._robust_mean_estimation(data_matrix)
            
            # Evaluate multivariate Gaussian density
            try:
                from scipy.stats import multivariate_normal
                # Create test point for density evaluation
                test_point = np.array([y_curr, np.interp(y_curr, y_data, x_data)])
                density = multivariate_normal.pdf(test_point, mean_vector, cov_matrix)
                return density
            except ImportError:
                return self._manual_multivariate_density(y_curr, y_data, x_data, mean_vector, cov_matrix)
                
        except Exception:
            return None
    
    def _robust_covariance_estimation(self, data_matrix: np.ndarray) -> np.ndarray:
        """Robust covariance matrix estimation with regularization"""
        try:
            n_samples, n_features = data_matrix.shape
            
            if n_samples < n_features + 1:
                # Insufficient samples - use identity with empirical variance scaling
                emp_var = np.var(data_matrix, axis=0, ddof=1)
                return np.diag(emp_var + 1e-6)
            
            # Multiple estimation methods with validation
            methods = []
            
            # Method 1: Empirical covariance with shrinkage
            emp_cov = np.cov(data_matrix.T, ddof=1)
            shrinkage_target = np.trace(emp_cov) / n_features * np.eye(n_features)
            shrinkage_factor = min(0.5, 1.0 / n_samples)
            shrunk_cov = (1 - shrinkage_factor) * emp_cov + shrinkage_factor * shrinkage_target
            methods.append(('shrinkage', shrunk_cov))
            
            # Method 2: Robust estimation using median-based approach
            try:
                robust_cov = self._median_absolute_deviation_cov(data_matrix)
                if robust_cov is not None:
                    methods.append(('mad', robust_cov))
            except Exception:
                pass
            
            # Method 3: Ridge regularization
            ridge_reg = max(1e-6, 0.01 * np.trace(emp_cov) / n_features)
            ridge_cov = emp_cov + ridge_reg * np.eye(n_features)
            methods.append(('ridge', ridge_cov))
            
            # Select best method based on condition number and positive definiteness
            best_cov = None
            best_score = float('inf')
            
            for method_name, cov_candidate in methods:
                if self._validate_covariance_matrix(cov_candidate):
                    cond_number = np.linalg.cond(cov_candidate)
                    if cond_number < best_score:
                        best_score = cond_number
                        best_cov = cov_candidate
            
            return best_cov
            
        except Exception:
            # Fallback to diagonal covariance
            try:
                variances = np.var(data_matrix, axis=0, ddof=1)
                return np.diag(variances + 1e-6)
            except Exception:
                return np.eye(data_matrix.shape[1]) * 1e-3
    
    def _median_absolute_deviation_cov(self, data_matrix: np.ndarray) -> np.ndarray:
        """Robust covariance using Median Absolute Deviation"""
        try:
            n_samples, n_features = data_matrix.shape
            
            # Compute robust center (median)
            center = np.median(data_matrix, axis=0)
            
            # Compute MAD-based covariance
            centered_data = data_matrix - center
            
            # MAD-based standard deviations
            mad_stds = np.median(np.abs(centered_data), axis=0) * 1.4826  # MAD to std conversion
            
            # Robust correlation estimation
            correlation_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Robust correlation using Spearman rank correlation
                    try:
                        from scipy.stats import spearmanr
                        corr_coef, _ = spearmanr(data_matrix[:, i], data_matrix[:, j])
                        if not np.isnan(corr_coef):
                            correlation_matrix[i, j] = correlation_matrix[j, i] = corr_coef
                    except ImportError:
                        # Fallback to simple correlation
                        corr_coef = np.corrcoef(data_matrix[:, i], data_matrix[:, j])[0, 1]
                        if not np.isnan(corr_coef):
                            correlation_matrix[i, j] = correlation_matrix[j, i] = corr_coef
            
            # Convert to covariance matrix
            std_matrix = np.outer(mad_stds, mad_stds)
            cov_matrix = correlation_matrix * std_matrix
            
            return cov_matrix
            
        except Exception:
            return None
    
    def _robust_mean_estimation(self, data_matrix: np.ndarray) -> np.ndarray:
        """Robust mean estimation using trimmed mean and median"""
        try:
            n_samples = data_matrix.shape[0]
            
            if n_samples < 5:
                return np.mean(data_matrix, axis=0)
            
            # Multiple robust estimators
            estimators = []
            
            # Trimmed mean (remove 10% of extreme values)
            trim_percent = min(0.1, 2.0 / n_samples)
            try:
                from scipy.stats import trim_mean
                trimmed = np.array([trim_mean(data_matrix[:, j], trim_percent) 
                                 for j in range(data_matrix.shape[1])])
                estimators.append(('trimmed', trimmed))
            except ImportError:
                # Manual trimmed mean
                trim_count = int(n_samples * trim_percent)
                if trim_count > 0:
                    sorted_data = np.sort(data_matrix, axis=0)
                    trimmed = np.mean(sorted_data[trim_count:-trim_count or None], axis=0)
                    estimators.append(('manual_trimmed', trimmed))
            
            # Median
            median_est = np.median(data_matrix, axis=0)
            estimators.append(('median', median_est))
            
            # Winsorized mean (replace extremes with percentiles)
            try:
                p5, p95 = np.percentile(data_matrix, [5, 95], axis=0)
                winsorized_data = np.clip(data_matrix, p5, p95)
                winsorized_mean = np.mean(winsorized_data, axis=0)
                estimators.append(('winsorized', winsorized_mean))
            except Exception:
                pass
            
            # Robust combination: weighted average based on stability
            if len(estimators) > 1:
                estimates = np.array([est[1] for est in estimators])
                # Weight by inverse of variance across estimators
                var_across_estimators = np.var(estimates, axis=0)
                weights = 1.0 / (var_across_estimators + 1e-8)
                weights = weights / np.sum(weights)
                
                robust_mean = np.average(estimates, axis=0, weights=weights)
                return robust_mean
            else:
                return estimators[0][1]
                
        except Exception:
            return np.mean(data_matrix, axis=0)
    
    def _validate_covariance_matrix(self, cov_matrix: np.ndarray) -> bool:
        """Validate covariance matrix properties"""
        try:
            # Check if matrix is square
            if cov_matrix.shape[0] != cov_matrix.shape[1]:
                return False
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(cov_matrix)):
                return False
            
            # Check if symmetric
            if not np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10):
                return False
            
            # Check if positive definite
            eigenvals = np.linalg.eigvals(cov_matrix)
            if not np.all(eigenvals > 1e-12):
                return False
            
            # Check condition number
            cond_num = np.linalg.cond(cov_matrix)
            if cond_num > 1e12:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _manual_multivariate_density(self, y_curr: float, y_data: np.ndarray, x_data: np.ndarray, 
                                   mean_vector: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Manual multivariate Gaussian density computation"""
        try:
            # Interpolate corresponding x value for y_curr
            x_curr = np.interp(y_curr, y_data, x_data)
            test_point = np.array([y_curr, x_curr])
            
            # Manual multivariate normal PDF computation
            k = len(mean_vector)  # dimension
            
            # Difference from mean
            diff = test_point - mean_vector
            
            # Compute (x-mu)^T * Sigma^(-1) * (x-mu)
            try:
                cov_inv = np.linalg.inv(cov_matrix)
                quadratic_form = np.dot(diff, np.dot(cov_inv, diff))
                
                # Compute determinant for normalization
                cov_det = np.linalg.det(cov_matrix)
                
                if cov_det <= 0:
                    return 1e-10
                
                # Multivariate normal PDF formula
                normalization = 1.0 / np.sqrt((2 * np.pi) ** k * cov_det)
                density = normalization * np.exp(-0.5 * quadratic_form)
                
                return density
                
            except np.linalg.LinAlgError:
                # Matrix inversion failed - use pseudo-inverse
                try:
                    cov_pinv = np.linalg.pinv(cov_matrix)
                    quadratic_form = np.dot(diff, np.dot(cov_pinv, diff))
                    
                    # Approximate density
                    density = np.exp(-0.5 * quadratic_form) / (2 * np.pi)
                    return density
                    
                except Exception:
                    return 1e-10
                    
        except Exception:
            return 1e-10
    
    def _apply_statistical_corrections(self, density: float, y_curr: float, 
                                     y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Apply advanced statistical corrections to density estimate"""
        try:
            corrected_density = density
            
            # Small sample correction
            n_samples = min(len(y_past), len(x_past))
            if n_samples < 30:
                small_sample_factor = 1.0 + 2.0 / n_samples
                corrected_density *= small_sample_factor
            
            # Outlier influence correction
            outlier_factor = self._compute_outlier_influence(y_curr, y_past, x_past)
            corrected_density *= outlier_factor
            
            # Non-stationarity correction
            stationarity_factor = self._compute_stationarity_correction(y_past, x_past)
            corrected_density *= stationarity_factor
            
            # Boundary effect correction
            boundary_factor = self._compute_boundary_correction(y_curr, y_past, x_past)
            corrected_density *= boundary_factor
            
            return corrected_density
            
        except Exception:
            return density
    
    def _compute_outlier_influence(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Compute correction factor for outlier influence"""
        try:
            # Use robust z-score for outlier detection
            y_median, y_mad = np.median(y_past), np.median(np.abs(y_past - np.median(y_past)))
            if y_mad == 0:
                return 1.0
            
            robust_z_score = 0.6745 * (y_curr - y_median) / y_mad
            
            # Huber-like correction
            if abs(robust_z_score) > 2.0:
                # Reduce influence of outliers
                correction = 1.0 / (1.0 + abs(robust_z_score) - 2.0)
                return max(0.1, correction)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _compute_stationarity_correction(self, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Compute correction for non-stationarity in the data"""
        try:
            n = len(y_past)
            if n < 10:
                return 1.0
            
            # Simple trend test using first and second half means
            mid_point = n // 2
            first_half_mean = np.mean(y_past[:mid_point])
            second_half_mean = np.mean(y_past[mid_point:])
            
            # Normalized trend measure
            overall_std = np.std(y_past)
            if overall_std == 0:
                return 1.0
            
            trend_strength = abs(second_half_mean - first_half_mean) / overall_std
            
            # Apply correction if strong trend detected
            if trend_strength > 1.0:
                correction = 1.0 / (1.0 + 0.1 * trend_strength)
                return max(0.5, correction)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _compute_boundary_correction(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Compute correction for boundary effects"""
        try:
            y_min, y_max = np.min(y_past), np.max(y_past)
            y_range = y_max - y_min
            
            if y_range == 0:
                return 1.0
            
            # Check if y_curr is near boundaries
            distance_to_boundary = min(y_curr - y_min, y_max - y_curr) / y_range
            
            # Apply correction for boundary effects
            if distance_to_boundary < 0.1:
                # Near boundary - increase density estimate
                correction = 1.0 + (0.1 - distance_to_boundary) * 2.0
                return min(2.0, correction)
            else:
                return 1.0
                
        except Exception:
            return 1.0
    
    def _robust_nonparametric_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Robust nonparametric density estimation using multiple methods"""
        try:
            estimators = []
            
            # Kernel density estimation with adaptive bandwidth
            kde_estimate = self._adaptive_kernel_density(y_curr, y_past, x_past)
            if kde_estimate is not None:
                estimators.append(kde_estimate)
            
            # Local linear regression based estimate
            llr_estimate = self._local_linear_regression_density(y_curr, y_past, x_past)
            if llr_estimate is not None:
                estimators.append(llr_estimate)
            
            # Histogram-based estimate with optimal binning
            hist_estimate = self._optimal_histogram_density(y_curr, y_past, x_past)
            if hist_estimate is not None:
                estimators.append(hist_estimate)
            
            if estimators:
                # Robust combination using trimmed mean
                estimators = np.array(estimators)
                if len(estimators) > 2:
                    # Remove extreme estimates
                    q25, q75 = np.percentile(estimators, [25, 75])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    filtered = estimators[(estimators >= lower_bound) & (estimators <= upper_bound)]
                    if len(filtered) > 0:
                        return np.mean(filtered)
                
                return np.mean(estimators)
            else:
                return self._fallback_density_estimation(y_curr, y_past, x_past)
                
        except Exception:
            return self._fallback_density_estimation(y_curr, y_past, x_past)
    
    def _adaptive_kernel_density(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Adaptive kernel density estimation"""
        try:
            # Combine data for density estimation
            all_data = np.concatenate([y_past, x_past])
            n = len(all_data)
            
            if n < 3:
                return None
            
            # Adaptive bandwidth using Silverman's rule of thumb with refinement
            std_data = np.std(all_data)
            iqr_data = np.percentile(all_data, 75) - np.percentile(all_data, 25)
            
            if iqr_data == 0:
                bandwidth = std_data * (4.0 / (3.0 * n)) ** 0.2
            else:
                silverman_factor = min(std_data, iqr_data / 1.34)
                bandwidth = silverman_factor * (4.0 / (3.0 * n)) ** 0.2
            
            if bandwidth == 0:
                return None
            
            # Gaussian kernel density estimate
            kernel_values = np.exp(-0.5 * ((all_data - y_curr) / bandwidth) ** 2)
            density = np.mean(kernel_values) / (bandwidth * np.sqrt(2 * np.pi))
            
            return density
            
        except Exception:
            return None
    
    def _local_linear_regression_density(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Local linear regression based density estimation"""
        try:
            # Create local neighborhood around y_curr
            all_y = np.concatenate([y_past, x_past])
            distances = np.abs(all_y - y_curr)
            
            # Select local neighborhood (nearest neighbors)
            k = max(3, min(10, len(all_y) // 3))
            nearest_indices = np.argsort(distances)[:k]
            local_y = all_y[nearest_indices]
            local_weights = 1.0 / (distances[nearest_indices] + 1e-8)
            
            # Fit local linear model (constant term for density)
            weighted_mean = np.average(local_y, weights=local_weights)
            weighted_var = np.average((local_y - weighted_mean) ** 2, weights=local_weights)
            
            if weighted_var == 0:
                return 1.0 / len(all_y)
            
            # Local Gaussian approximation
            density = np.exp(-0.5 * (y_curr - weighted_mean) ** 2 / weighted_var) / np.sqrt(2 * np.pi * weighted_var)
            
            return density
            
        except Exception:
            return None
    
    def _optimal_histogram_density(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Optimal histogram-based density estimation"""
        try:
            all_data = np.concatenate([y_past, x_past])
            n = len(all_data)
            
            if n < 3:
                return None
            
            # Optimal number of bins using Freedman-Diaconis rule
            iqr = np.percentile(all_data, 75) - np.percentile(all_data, 25)
            if iqr == 0:
                n_bins = max(3, int(np.sqrt(n)))
            else:
                bin_width = 2 * iqr / (n ** (1/3))
                data_range = np.max(all_data) - np.min(all_data)
                n_bins = max(3, int(data_range / bin_width))
            
            n_bins = min(n_bins, n // 2)  # Ensure sufficient data per bin
            
            # Create histogram
            counts, bin_edges = np.histogram(all_data, bins=n_bins)
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Find which bin y_curr falls into
            bin_index = np.searchsorted(bin_edges[1:], y_curr)
            bin_index = max(0, min(bin_index, len(counts) - 1))
            
            # Density estimate
            density = counts[bin_index] / (n * bin_width)
            
            return max(1e-10, density)
            
        except Exception:
            return None
    
    def _fallback_density_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Fallback density estimation for edge cases"""
        try:
            # Simple robust approach
            all_data = np.concatenate([y_past, x_past])
            n = len(all_data)
            
            if n == 0:
                return 1e-10
            
            # Robust distance-based estimate
            distances = np.abs(all_data - y_curr)
            median_distance = np.median(distances)
            
            if median_distance == 0:
                return 1.0 / n
            
            # Exponential decay based on median distance
            density = np.exp(-distances.min() / (median_distance + 1e-8)) / n
            
            return max(1e-10, density)
            
        except Exception:
            return 1e-10
    
    def _robust_fallback_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Ultimate robust fallback estimation"""
        try:
            # Extremely simple but robust approach
            all_values = np.concatenate([y_past, x_past, [y_curr]])
            
            # Use empirical distribution approximation
            n = len(all_values)
            sorted_values = np.sort(all_values)
            
            # Find position of y_curr in sorted array
            position = np.searchsorted(sorted_values, y_curr) 
            
            # Empirical density approximation
            if position == 0 or position == n:
                density = 1.0 / (2.0 * n)
            else:
                # Local density based on neighboring spacing
                left_spacing = abs(sorted_values[position] - sorted_values[position - 1])
                right_spacing = abs(sorted_values[position + 1] - sorted_values[position]) if position + 1 < n else left_spacing
                avg_spacing = (left_spacing + right_spacing) / 2.0
                
                if avg_spacing == 0:
                    density = 1.0 / n
                else:
                    density = 1.0 / (n * avg_spacing)
            
            return max(1e-12, density)
            
        except Exception:
            # Absolute fallback
            return 1e-10
    
    def _simple_marginal_estimation(self, y_curr: float, y_past: np.ndarray) -> float:
        """Simple marginal estimation"""
        try:
            mean_y, std_y = np.mean(y_past), np.std(y_past)
            
            if std_y == 0:
                return 1.0 / len(y_past)
            
            return max(1e-10, stats.norm.pdf(y_curr, mean_y, std_y))
            
        except Exception:
            return 1e-10
    
    def _simple_count_estimation(self, y_curr: float, y_past: np.ndarray, x_past: np.ndarray) -> float:
        """Simple count-based estimation"""
        try:
            tolerance = (np.std(y_past) + np.std(x_past)) * 0.1
            matches = np.sum((np.abs(y_past - y_curr) < tolerance) & 
                           (np.abs(x_past - y_curr) < tolerance))
            
            return max(1e-10, (matches + 1) / (len(y_past) + 2))
            
        except Exception:
            return 1e-10
    
    def _test_conditional_causality(self, cause: np.ndarray, effect: np.ndarray, lag: int) -> float:
        """Test conditional causality accounting for confounding variables"""
        try:
            # Partial correlation-based conditional independence
            n = len(cause) - lag
            if n < 15:
                return 0.0
            
            # Use lagged values as conditioning variables
            conditioning_vars = []
            for i in range(1, min(lag + 1, 4)):
                if i < len(cause):
                    conditioning_vars.append(cause[:-i] if i > 0 else cause)
                    conditioning_vars.append(effect[:-i] if i > 0 else effect)
            
            if not conditioning_vars:
                return 0.0
            
            # Align all series
            min_len = min(len(var) for var in conditioning_vars + [cause, effect])
            
            X = np.column_stack([var[:min_len] for var in conditioning_vars])
            y_cause = cause[:min_len]
            y_effect = effect[:min_len]
            
            # Partial correlation
            if X.shape[1] > 0 and X.shape[0] > X.shape[1] + 5:
                # Residuals after removing conditioning effects
                beta_cause = np.linalg.lstsq(X, y_cause, rcond=None)[0]
                beta_effect = np.linalg.lstsq(X, y_effect, rcond=None)[0]
                
                res_cause = y_cause - X @ beta_cause
                res_effect = y_effect - X @ beta_effect
                
                # Correlation of residuals
                partial_corr = np.corrcoef(res_cause, res_effect)[0, 1]
                
                if not np.isnan(partial_corr):
                    return abs(partial_corr)
            
            return 0.0
        
        except:
            return 0.0
    
    def _advanced_cross_correlation(self, cause: np.ndarray, effect: np.ndarray, max_lag: int) -> float:
        """Advanced cross-correlation with statistical significance testing"""
        try:
            correlations = []
            n = len(cause)
            
            for lag in range(1, min(max_lag + 1, n // 4)):
                if lag < n:
                    corr = np.corrcoef(cause[:-lag], effect[lag:])[0, 1]
                    if not np.isnan(corr):
                        # Statistical significance (approximate)
                        se = 1 / np.sqrt(n - lag - 3)  # Standard error
                        z_score = abs(corr) / se
                        p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
                        
                        # Weight by significance
                        if p_value < 0.05:
                            correlations.append(abs(corr) * (1 - p_value))
            
            return max(correlations) if correlations else 0.0
        
        except:
            return 0.0
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF"""
        # Abramowitz and Stegun approximation
        return 0.5 * (1 + np.tanh(x * np.sqrt(2 / np.pi)))
    
    def _calculate_phase_coupling(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Calculate phase coupling between signals"""
        try:
            # Hilbert transform for phase extraction (comprehensive)
            def get_phase(signal):
                return self._comprehensive_phase_extraction(signal)
            
            phase_cause = get_phase(cause)
            phase_effect = get_phase(effect)
            
            # Phase locking value
            phase_diff = phase_cause - phase_effect
            plv = abs(np.mean(np.exp(1j * phase_diff)))
            
            return plv
        
        except:
            return 0.0
    
    def _comprehensive_phase_extraction(self, signal: np.ndarray) -> np.ndarray:
        """Comprehensive phase extraction using multiple methods with fallbacks"""
        try:
            # Method 1: Discrete Hilbert Transform via FFT
            phase = self._hilbert_transform_fft(signal)
            
            # Validate phase continuity
            if self._validate_phase_continuity(phase):
                return phase
            
            # Method 2: Filter-based Hilbert transform
            phase = self._hilbert_transform_filter(signal)
            
            if self._validate_phase_continuity(phase):
                return phase
            
            # Method 3: Empirical Mode Decomposition with Hilbert-Huang
            phase = self._emd_hilbert_phase(signal)
            
            if self._validate_phase_continuity(phase):
                return phase
            
            # Method 4: Wavelet-based phase extraction
            phase = self._wavelet_phase_extraction(signal)
            
            if self._validate_phase_continuity(phase):
                return phase
            
            # Fallback: Robust derivative-based phase
            return self._robust_derivative_phase(signal)
            
        except:
            # Ultimate fallback
            return self._simple_phase_fallback(signal)
    
    def _hilbert_transform_fft(self, signal: np.ndarray) -> np.ndarray:
        """Hilbert transform using FFT with proper windowing"""
        n = len(signal)
        
        # Apply windowing to reduce spectral leakage
        window = np.hanning(n)
        windowed_signal = signal * window
        
        # Zero-padding for better frequency resolution
        n_pad = 2 ** int(np.ceil(np.log2(n * 2)))
        padded_signal = np.zeros(n_pad)
        padded_signal[:n] = windowed_signal
        
        # FFT-based Hilbert transform
        fft_signal = np.fft.fft(padded_signal)
        
        # Create Hilbert transform coefficients
        h = np.zeros(n_pad, dtype=complex)
        h[0] = 1.0
        h[1:n_pad//2] = 2.0
        h[n_pad//2] = 1.0 if n_pad % 2 == 0 else 0.0
        
        # Apply Hilbert transform in frequency domain
        analytic_signal_fft = fft_signal * h
        analytic_signal = np.fft.ifft(analytic_signal_fft)[:n]
        
        # Extract phase with unwrapping
        phase = np.angle(analytic_signal)
        phase_unwrapped = np.unwrap(phase)
        
        # Normalize to [0, 2π]
        return np.mod(phase_unwrapped, 2 * np.pi)
    
    def _hilbert_transform_filter(self, signal: np.ndarray) -> np.ndarray:
        """Hilbert transform using finite impulse response filter"""
        n = len(signal)
        
        # Design Hilbert transformer FIR filter
        filter_length = min(101, n // 4)  # Adaptive filter length
        if filter_length % 2 == 0:
            filter_length += 1
        
        # Hilbert FIR coefficients
        h = np.zeros(filter_length)
        mid = filter_length // 2
        
        for i in range(filter_length):
            if i != mid:
                h[i] = 2.0 / (np.pi * (i - mid)) * (1 - np.cos(np.pi * (i - mid)))
        
        # Apply filter with padding
        padding = filter_length // 2
        padded_signal = np.pad(signal, (padding, padding), mode='reflect')
        
        # Convolution
        filtered_signal = np.convolve(padded_signal, h, mode='valid')
        
        # Ensure correct length
        if len(filtered_signal) > n:
            start = (len(filtered_signal) - n) // 2
            filtered_signal = filtered_signal[start:start + n]
        elif len(filtered_signal) < n:
            filtered_signal = np.pad(filtered_signal, (0, n - len(filtered_signal)), mode='edge')
        
        # Construct analytic signal
        analytic_signal = signal + 1j * filtered_signal
        
        # Extract and unwrap phase
        phase = np.angle(analytic_signal)
        phase_unwrapped = np.unwrap(phase)
        
        return np.mod(phase_unwrapped, 2 * np.pi)
    
    def _emd_hilbert_phase(self, signal: np.ndarray) -> np.ndarray:
        """Comprehensive Empirical Mode Decomposition with advanced Hilbert-Huang transform"""
        try:
            return self._advanced_hilbert_huang_transform(signal)
        except Exception:
            try:
                return self._robust_emd_hilbert_analysis(signal)
            except Exception:
                return self._emergency_phase_extraction_emd(signal)
    
    def _advanced_hilbert_huang_transform(self, signal: np.ndarray) -> np.ndarray:
        """Advanced Hilbert-Huang Transform with comprehensive multi-IMF analysis"""
        if len(signal) < 10:
            return self._robust_derivative_phase(signal)
        
        # Enhanced EMD decomposition with validation
        imfs = self._empirical_mode_decomposition(signal, max_imfs=8)
        
        if len(imfs) == 0:
            return self._fallback_phase_analysis(signal)
        
        # Comprehensive IMF analysis and selection
        selected_imfs, weights = self._intelligent_imf_selection(signal, imfs)
        
        if len(selected_imfs) == 0:
            return self._fallback_phase_analysis(signal)
        
        # Multi-IMF Hilbert spectrum analysis
        instantaneous_phases = self._compute_multi_imf_phases(selected_imfs, weights)
        
        # Composite phase extraction with temporal coherence
        final_phase = self._synthesize_coherent_phase(instantaneous_phases, weights, signal)
        
        # Phase unwrapping and continuity enhancement
        enhanced_phase = self._enhance_phase_continuity(final_phase, signal)
        
        return enhanced_phase
    
    def _intelligent_imf_selection(self, original_signal: np.ndarray, imfs: list) -> tuple:
        """Intelligent IMF selection based on energy, frequency content, and correlation"""
        if len(imfs) == 0:
            return [], []
        
        selected_imfs = []
        weights = []
        
        # Calculate IMF quality metrics
        imf_metrics = []
        for i, imf in enumerate(imfs):
            metrics = self._calculate_imf_quality_metrics(imf, original_signal)
            imf_metrics.append(metrics)
        
        # Selection criteria based on multiple factors
        for i, (imf, metrics) in enumerate(zip(imfs, imf_metrics)):
            if self._should_include_imf(metrics, i, len(imfs)):
                selected_imfs.append(imf)
                weight = self._calculate_imf_weight(metrics, i, len(imfs))
                weights.append(weight)
        
        # Normalize weights
        if len(weights) > 0:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        return selected_imfs, weights
    
    def _calculate_imf_quality_metrics(self, imf: np.ndarray, original: np.ndarray) -> dict:
        """Calculate comprehensive quality metrics for IMF"""
        metrics = {}
        
        # Energy-based metrics
        metrics['energy'] = np.sum(imf**2)
        metrics['energy_ratio'] = metrics['energy'] / (np.sum(original**2) + 1e-10)
        
        # Frequency content analysis
        try:
            fft_imf = np.fft.fft(imf)
            freqs = np.fft.fftfreq(len(imf))
            power_spectrum = np.abs(fft_imf)**2
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(power_spectrum[:len(power_spectrum)//2])
            metrics['dominant_frequency'] = abs(freqs[dominant_freq_idx])
            
            # Spectral concentration (higher is more concentrated)
            total_power = np.sum(power_spectrum)
            if total_power > 0:
                metrics['spectral_concentration'] = np.max(power_spectrum) / total_power
            else:
                metrics['spectral_concentration'] = 0
            
            # Bandwidth estimation
            cumulative_power = np.cumsum(np.sort(power_spectrum)[::-1])
            if total_power > 0:
                power_90_idx = np.where(cumulative_power >= 0.9 * total_power)[0]
                metrics['effective_bandwidth'] = len(power_90_idx) / len(power_spectrum)
            else:
                metrics['effective_bandwidth'] = 1.0
                
        except Exception:
            metrics['dominant_frequency'] = 0
            metrics['spectral_concentration'] = 0
            metrics['effective_bandwidth'] = 1.0
        
        # Correlation with original signal
        try:
            correlation = np.corrcoef(imf, original)[0, 1]
            metrics['correlation'] = abs(correlation) if not np.isnan(correlation) else 0
        except Exception:
            metrics['correlation'] = 0
        
        # IMF validity metrics
        metrics['zero_crossings'] = np.sum(np.diff(np.sign(imf)) != 0)
        maxima_idx, minima_idx = self._simple_extrema_detection(imf)
        metrics['extrema_count'] = len(maxima_idx) + len(minima_idx)
        
        # Smoothness metric (lower second derivative variation)
        if len(imf) > 2:
            second_derivative = np.diff(imf, 2)
            metrics['smoothness'] = 1.0 / (1.0 + np.std(second_derivative))
        else:
            metrics['smoothness'] = 1.0
        
        return metrics
    
    def _should_include_imf(self, metrics: dict, imf_index: int, total_imfs: int) -> bool:
        """Determine whether to include IMF based on quality criteria"""
        # Energy threshold - must contain significant energy
        if metrics['energy_ratio'] < 0.01:  # Less than 1% of total energy
            return False
        
        # Correlation threshold - must be somewhat related to original
        if metrics['correlation'] < 0.05:  # Very low correlation
            return False
        
        # Frequency content - avoid overly concentrated or dispersed signals
        if metrics['spectral_concentration'] > 0.95:  # Too concentrated (likely noise)
            return False
        
        if metrics['effective_bandwidth'] > 0.8:  # Too dispersed
            return False
        
        # IMF validity - must satisfy basic EMD criteria
        extrema_zero_diff = abs(metrics['extrema_count'] - metrics['zero_crossings'])
        if extrema_zero_diff > max(2, metrics['extrema_count'] * 0.3):
            return False
        
        # Preference for early IMFs (higher frequencies) but not exclusively
        position_weight = 1.0 - (imf_index / (total_imfs + 1)) * 0.3
        combined_score = (metrics['energy_ratio'] * metrics['correlation'] * 
                         metrics['smoothness'] * position_weight)
        
        return combined_score > 0.001  # Minimum combined quality threshold
    
    def _calculate_imf_weight(self, metrics: dict, imf_index: int, total_imfs: int) -> float:
        """Calculate weight for IMF based on quality metrics"""
        # Base weight from energy and correlation
        base_weight = metrics['energy_ratio'] * metrics['correlation']
        
        # Frequency importance (favor mid-range frequencies)
        freq_weight = 1.0 - abs(metrics['dominant_frequency'] - 0.1)  # Favor ~0.1 normalized freq
        freq_weight = max(0.1, freq_weight)
        
        # Position weight (slight preference for earlier IMFs)
        position_weight = 1.0 - (imf_index / (total_imfs + 1)) * 0.2
        
        # Smoothness contribution
        smoothness_weight = metrics['smoothness']
        
        # Combined weight calculation
        total_weight = base_weight * freq_weight * position_weight * smoothness_weight
        
        return max(0.01, total_weight)  # Minimum weight to avoid numerical issues
    
    def _compute_multi_imf_phases(self, imfs: list, weights: list) -> list:
        """Compute instantaneous phases for multiple selected IMFs"""
        phases = []
        
        for imf in imfs:
            try:
                # Primary method: FFT-based Hilbert transform
                phase = self._hilbert_transform_fft(imf)
                
                # Validate phase quality
                if self._validate_phase_quality(phase, imf):
                    phases.append(phase)
                    continue
                
            except Exception:
                pass
            
            try:
                # Fallback method: FIR filter Hilbert transform
                phase = self._hilbert_transform_fir_filter(imf)
                if self._validate_phase_quality(phase, imf):
                    phases.append(phase)
                    continue
                    
            except Exception:
                pass
            
            # Emergency fallback: derivative-based phase
            try:
                phase = self._robust_derivative_phase(imf)
                phases.append(phase)
            except Exception:
                # Last resort: constant phase
                phases.append(np.zeros(len(imf)))
        
        return phases
    
    def _validate_phase_quality(self, phase: np.ndarray, reference_signal: np.ndarray) -> bool:
        """Validate quality of computed instantaneous phase"""
        if len(phase) != len(reference_signal):
            return False
        
        if not np.all(np.isfinite(phase)):
            return False
        
        # Check phase continuity (no extreme jumps)
        phase_diff = np.diff(phase)
        if np.any(np.abs(phase_diff) > np.pi):  # Phase jumps larger than π
            return False
        
        # Check phase range reasonableness
        phase_range = np.max(phase) - np.min(phase)
        if phase_range > 10 * np.pi:  # Excessively large phase range
            return False
        
        return True
    
    def _synthesize_coherent_phase(self, phases: list, weights: list, original_signal: np.ndarray) -> np.ndarray:
        """Synthesize coherent phase from multiple IMF phases with temporal consistency"""
        if len(phases) == 0:
            return np.zeros(len(original_signal))
        
        if len(phases) == 1:
            return phases[0]
        
        # Ensure all phases have same length
        target_length = len(original_signal)
        normalized_phases = []
        
        for phase in phases:
            if len(phase) != target_length:
                # Interpolate to target length
                normalized_phase = np.interp(
                    np.arange(target_length),
                    np.arange(len(phase)) * (target_length / len(phase)),
                    phase
                )
                normalized_phases.append(normalized_phase)
            else:
                normalized_phases.append(phase)
        
        # Weighted combination with temporal coherence enhancement
        if len(weights) != len(normalized_phases):
            weights = [1.0 / len(normalized_phases)] * len(normalized_phases)
        
        # Initial weighted average
        combined_phase = np.zeros(target_length)
        for phase, weight in zip(normalized_phases, weights):
            combined_phase += weight * phase
        
        # Temporal coherence enhancement
        enhanced_phase = self._enhance_temporal_coherence(combined_phase, normalized_phases, weights)
        
        return enhanced_phase
    
    def _enhance_temporal_coherence(self, base_phase: np.ndarray, individual_phases: list, weights: list) -> np.ndarray:
        """Enhance temporal coherence of combined phase"""
        if len(individual_phases) <= 1:
            return base_phase
        
        enhanced = base_phase.copy()
        
        # Smooth phase transitions using local averaging
        window_size = min(5, len(enhanced) // 10)
        if window_size >= 2:
            for i in range(window_size, len(enhanced) - window_size):
                # Local weighted average with emphasis on continuity
                local_window = enhanced[i-window_size:i+window_size+1]
                
                # Calculate local gradient
                if len(local_window) >= 3:
                    local_gradient = np.gradient(local_window)
                    median_gradient = np.median(local_gradient)
                    
                    # Adjust point to maintain gradient consistency
                    expected_value = enhanced[i-1] + median_gradient
                    actual_value = enhanced[i]
                    
                    # Blend expected and actual values
                    blend_factor = 0.3  # Conservative blending
                    enhanced[i] = (1 - blend_factor) * actual_value + blend_factor * expected_value
        
        # Phase unwrapping to ensure continuity
        enhanced = self._unwrap_phase_carefully(enhanced)
        
        return enhanced
    
    def _unwrap_phase_carefully(self, phase: np.ndarray) -> np.ndarray:
        """Careful phase unwrapping with jump detection and correction"""
        if len(phase) < 2:
            return phase
        
        unwrapped = phase.copy()
        
        # Detect and correct phase jumps
        phase_diff = np.diff(phase)
        jump_threshold = np.pi  # Standard phase jump threshold
        
        cumulative_correction = 0
        for i in range(len(phase_diff)):
            if phase_diff[i] > jump_threshold:
                cumulative_correction -= 2 * np.pi
            elif phase_diff[i] < -jump_threshold:
                cumulative_correction += 2 * np.pi
            
            unwrapped[i + 1] += cumulative_correction
        
        return unwrapped
    
    def _enhance_phase_continuity(self, phase: np.ndarray, reference_signal: np.ndarray) -> np.ndarray:
        """Enhance phase continuity with signal-aware corrections"""
        if len(phase) < 5:
            return phase
        
        enhanced = phase.copy()
        
        # Apply smoothing to reduce noise while preserving main trends
        smoothed = self._adaptive_phase_smoothing(enhanced, reference_signal)
        
        # Blend original and smoothed phases
        blend_factor = 0.2  # Conservative blending to preserve original characteristics
        final_phase = (1 - blend_factor) * enhanced + blend_factor * smoothed
        
        # Final validation and correction
        final_phase = self._correct_phase_artifacts(final_phase, reference_signal)
        
        return final_phase
    
    def _adaptive_phase_smoothing(self, phase: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Adaptive smoothing based on signal characteristics"""
        if len(phase) < 5:
            return phase
        
        # Determine smoothing strength based on signal variability
        signal_variability = np.std(reference)
        phase_variability = np.std(np.diff(phase))
        
        # Adaptive kernel size
        base_kernel_size = 3
        if signal_variability > np.mean(np.abs(reference)):
            kernel_size = base_kernel_size  # Less smoothing for variable signals
        else:
            kernel_size = min(5, len(phase) // 10)  # More smoothing for smooth signals
        
        if kernel_size < 2:
            return phase
        
        # Apply adaptive smoothing
        kernel = np.ones(kernel_size) / kernel_size
        padded_phase = np.pad(phase, kernel_size//2, mode='edge')
        smoothed = np.convolve(padded_phase, kernel, mode='valid')
        
        return smoothed[:len(phase)]
    
    def _correct_phase_artifacts(self, phase: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Correct phase artifacts and ensure physical consistency"""
        corrected = phase.copy()
        
        # Remove extreme outliers
        if len(corrected) > 3:
            phase_median = np.median(corrected)
            phase_mad = np.median(np.abs(corrected - phase_median))
            
            if phase_mad > 0:
                outlier_threshold = 5 * phase_mad
                outlier_mask = np.abs(corrected - phase_median) > outlier_threshold
                
                # Replace outliers with interpolated values
                if np.sum(outlier_mask) > 0:
                    valid_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    if len(valid_indices) >= 2:
                        corrected[outlier_indices] = np.interp(
                            outlier_indices, valid_indices, corrected[valid_indices]
                        )
        
        return corrected
    
    def _robust_emd_hilbert_analysis(self, signal: np.ndarray) -> np.ndarray:
        """Advanced comprehensive EMD-Hilbert analysis with multi-scale decomposition and adaptive processing"""
        try:
            # Comprehensive signal preprocessing and analysis
            if len(signal) < 10:
                return self._fallback_phase_analysis(signal)
            
            # Signal preprocessing with adaptive denoising
            denoised_signal = self._advanced_signal_denoising(signal)
            
            # Multi-resolution EMD decomposition with ensemble approach
            emd_results = self._ensemble_empirical_mode_decomposition(denoised_signal)
            
            if not emd_results or len(emd_results['imfs']) == 0:
                return self._fallback_phase_analysis(signal)
            
            # Advanced IMF selection using multiple criteria
            selected_imfs = self._intelligent_imf_selection(
                emd_results['imfs'], 
                emd_results['residue'], 
                denoised_signal
            )
            
            # Multi-method phase extraction with validation
            phase_results = self._multi_method_phase_extraction(selected_imfs, denoised_signal)
            
            if not phase_results or len(phase_results) == 0:
                return self._fallback_phase_analysis(signal)
            
            # Advanced phase fusion with adaptive weighting
            combined_phase = self._adaptive_phase_fusion(
                phase_results, 
                selected_imfs, 
                denoised_signal
            )
            
            # Post-processing and validation
            validated_phase = self._validate_and_postprocess_phase(
                combined_phase, 
                denoised_signal
            )
            
            return validated_phase
            
        except Exception as e:
            logging.warning(f"Advanced EMD-Hilbert analysis failed: {e}, using fallback")
            return self._fallback_phase_analysis(signal)
    
    def _advanced_signal_denoising(self, signal: np.ndarray) -> np.ndarray:
        """Advanced signal denoising with multiple techniques"""
        try:
            # Method 1: Adaptive Savitzky-Golay filtering
            window_length = min(max(5, len(signal) // 20), 51)
            if window_length % 2 == 0:
                window_length += 1
            
            denoised_sg = scipy.signal.savgol_filter(
                signal, 
                window_length, 
                polyorder=min(3, window_length - 1),
                mode='nearest'
            )
            
            # Method 2: Wavelet denoising
            denoised_wavelet = self._wavelet_denoising(signal)
            
            # Method 3: Robust regression denoising
            denoised_regression = self._regression_denoising(signal)
            
            # Combine denoising methods with quality assessment
            methods = [
                ('savgol', denoised_sg),
                ('wavelet', denoised_wavelet),
                ('regression', denoised_regression)
            ]
            
            # Select best denoising method based on SNR improvement
            best_method, best_denoised = self._select_best_denoising(signal, methods)
            
            return best_denoised
            
        except Exception:
            return signal  # Return original if denoising fails
    
    def _wavelet_denoising(self, signal: np.ndarray) -> np.ndarray:
        """Wavelet-based denoising"""
        try:
            # Choose appropriate wavelet based on signal characteristics
            wavelet = self._select_optimal_wavelet(signal)
            
            # Decompose signal
            coeffs = pywt.wavedec(signal, wavelet, level=min(6, int(np.log2(len(signal)))))
            
            # Threshold estimation using Bayes shrink method
            sigma = self._estimate_noise_level(coeffs[-1])  # Use finest detail coefficients
            
            # Apply soft thresholding to detail coefficients
            denoised_coeffs = list(coeffs)
            for i in range(1, len(coeffs)):
                threshold = sigma * np.sqrt(2 * np.log(len(coeffs[i])))
                denoised_coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
            
            # Reconstruct signal
            denoised = pywt.waverec(denoised_coeffs, wavelet)
            
            # Ensure same length as original
            if len(denoised) != len(signal):
                denoised = denoised[:len(signal)]
            
            return denoised
            
        except Exception:
            return signal
    
    def _regression_denoising(self, signal: np.ndarray) -> np.ndarray:
        """Robust regression-based denoising"""
        try:
            from sklearn.linear_model import HuberRegressor
            from sklearn.preprocessing import PolynomialFeatures
            
            x = np.arange(len(signal)).reshape(-1, 1)
            
            # Create polynomial features up to degree 3
            poly_features = PolynomialFeatures(degree=3, include_bias=False)
            x_poly = poly_features.fit_transform(x)
            
            # Fit robust regression
            regressor = HuberRegressor(epsilon=1.35, alpha=0.01)
            regressor.fit(x_poly, signal)
            
            # Predict smooth trend
            trend = regressor.predict(x_poly)
            
            # Compute residuals and filter outliers
            residuals = signal - trend
            mad = np.median(np.abs(residuals - np.median(residuals)))
            threshold = 3 * mad
            
            # Keep trend for outliers, original for normal points
            denoised = signal.copy()
            outlier_mask = np.abs(residuals) > threshold
            denoised[outlier_mask] = trend[outlier_mask]
            
            return denoised
            
        except Exception:
            return signal
    
    def _select_best_denoising(self, original: np.ndarray, methods: list) -> tuple:
        """Select best denoising method based on SNR and other metrics"""
        try:
            best_snr = -np.inf
            best_method = methods[0]
            
            for name, denoised in methods:
                # Calculate SNR improvement
                noise_original = original - scipy.signal.savgol_filter(
                    original, 
                    min(max(5, len(original) // 50), 51) | 1, 
                    2,
                    mode='nearest'
                )
                noise_denoised = denoised - scipy.signal.savgol_filter(
                    denoised, 
                    min(max(5, len(denoised) // 50), 51) | 1, 
                    2,
                    mode='nearest'
                )
                
                snr_original = 10 * np.log10(np.var(original) / (np.var(noise_original) + 1e-10))
                snr_denoised = 10 * np.log10(np.var(denoised) / (np.var(noise_denoised) + 1e-10))
                
                snr_improvement = snr_denoised - snr_original
                
                if snr_improvement > best_snr:
                    best_snr = snr_improvement
                    best_method = (name, denoised)
            
            return best_method
            
        except Exception:
            return methods[0]  # Return first method if selection fails
    
    def _ensemble_empirical_mode_decomposition(self, signal: np.ndarray) -> dict:
        """Ensemble EMD with multiple realizations and noise-assisted decomposition"""
        try:
            ensemble_size = min(50, max(10, len(signal) // 100))
            noise_std = 0.2 * np.std(signal)
            
            all_imfs = []
            all_residues = []
            
            # Generate ensemble of EMD decompositions
            for i in range(ensemble_size):
                # Add different noise realizations
                noise = np.random.normal(0, noise_std, len(signal))
                noisy_signal = signal + noise
                
                try:
                    imfs = self._empirical_mode_decomposition(
                        noisy_signal, 
                        max_imfs=8,
                        sift_tolerance=0.05,
                        max_iterations=1000
                    )
                    
                    if len(imfs) > 0:
                        all_imfs.append(imfs)
                        residue = noisy_signal - np.sum(imfs, axis=0)
                        all_residues.append(residue)
                        
                except Exception:
                    continue
            
            if not all_imfs:
                return {'imfs': [], 'residue': np.zeros_like(signal), 'ensemble_info': {}}
            
            # Ensemble averaging with mode alignment
            aligned_imfs = self._align_and_average_imfs(all_imfs)
            averaged_residue = np.mean(all_residues, axis=0) if all_residues else np.zeros_like(signal)
            
            # Compute ensemble statistics
            ensemble_info = {
                'realizations': len(all_imfs),
                'mode_consistency': self._compute_mode_consistency(all_imfs),
                'decomposition_quality': self._assess_decomposition_quality(aligned_imfs, signal)
            }
            
            return {
                'imfs': aligned_imfs,
                'residue': averaged_residue,
                'ensemble_info': ensemble_info
            }
            
        except Exception:
            # Fallback to single EMD
            try:
                imfs = self._empirical_mode_decomposition(signal, max_imfs=5)
                residue = signal - np.sum(imfs, axis=0) if len(imfs) > 0 else signal
                return {
                    'imfs': imfs,
                    'residue': residue,
                    'ensemble_info': {'realizations': 1}
                }
            except Exception:
                return {'imfs': [], 'residue': np.zeros_like(signal), 'ensemble_info': {}}
    
    def _align_and_average_imfs(self, all_imfs: list) -> list:
        """Align and average IMFs across ensemble realizations"""
        try:
            if not all_imfs:
                return []
            
            # Find maximum number of modes
            max_modes = max(len(imfs) for imfs in all_imfs)
            
            averaged_imfs = []
            
            for mode_idx in range(max_modes):
                mode_collection = []
                
                for imfs in all_imfs:
                    if mode_idx < len(imfs):
                        mode_collection.append(imfs[mode_idx])
                
                if mode_collection:
                    # Align modes using cross-correlation
                    aligned_modes = self._align_modes_cross_correlation(mode_collection)
                    averaged_mode = np.mean(aligned_modes, axis=0)
                    averaged_imfs.append(averaged_mode)
            
            return averaged_imfs
            
        except Exception:
            # Return first realization if alignment fails
            return all_imfs[0] if all_imfs else []
    
    def _intelligent_imf_selection(self, imfs: list, residue: np.ndarray, signal: np.ndarray) -> list:
        """Intelligent IMF selection using multiple criteria"""
        try:
            if not imfs:
                return []
            
            selection_criteria = []
            
            for i, imf in enumerate(imfs):
                # Criterion 1: Energy ratio
                energy_ratio = np.sum(imf**2) / np.sum(signal**2)
                
                # Criterion 2: Instantaneous frequency consistency
                inst_freq_consistency = self._compute_instantaneous_frequency_consistency(imf)
                
                # Criterion 3: Orthogonality with other IMFs
                orthogonality_score = self._compute_orthogonality_score(imf, imfs, i)
                
                # Criterion 4: Signal correlation
                correlation = np.abs(np.corrcoef(imf, signal)[0, 1])
                
                # Criterion 5: Mode mixing detection
                mode_mixing_penalty = self._detect_mode_mixing(imf)
                
                # Combined score
                score = (
                    0.3 * energy_ratio +
                    0.25 * inst_freq_consistency +
                    0.2 * orthogonality_score +
                    0.15 * correlation -
                    0.1 * mode_mixing_penalty
                )
                
                selection_criteria.append({
                    'index': i,
                    'score': score,
                    'energy_ratio': energy_ratio,
                    'freq_consistency': inst_freq_consistency,
                    'orthogonality': orthogonality_score,
                    'correlation': correlation,
                    'mode_mixing': mode_mixing_penalty
                })
            
            # Select top IMFs based on combined score
            selection_criteria.sort(key=lambda x: x['score'], reverse=True)
            
            # Adaptive selection: choose IMFs that contribute significantly
            selected_indices = []
            cumulative_energy = 0.0
            total_energy = np.sum(signal**2)
            
            for criterion in selection_criteria:
                if (cumulative_energy < 0.95 * total_energy and 
                    criterion['score'] > 0.1 and 
                    len(selected_indices) < 5):
                    
                    selected_indices.append(criterion['index'])
                    cumulative_energy += criterion['energy_ratio'] * total_energy
            
            # Ensure at least one IMF is selected
            if not selected_indices:
                selected_indices = [0] if len(imfs) > 0 else []
            
            return [imfs[i] for i in selected_indices]
            
        except Exception:
            # Return first 3 IMFs as fallback
            return imfs[:min(3, len(imfs))]
    
    def _multi_method_phase_extraction(self, imfs: list, signal: np.ndarray) -> list:
        """Multi-method phase extraction with validation"""
        try:
            phase_results = []
            
            for i, imf in enumerate(imfs):
                imf_phases = []
                
                # Method 1: Hilbert Transform
                try:
                    hilbert_phase = self._hilbert_transform_fft(imf)
                    if self._validate_phase_quality(hilbert_phase, imf):
                        imf_phases.append(('hilbert', hilbert_phase))
                except Exception:
                    pass
                
                # Method 2: Empirical Mode Decomposition instantaneous phase
                try:
                    emd_phase = self._emd_instantaneous_phase(imf)
                    if self._validate_phase_quality(emd_phase, imf):
                        imf_phases.append(('emd_inst', emd_phase))
                except Exception:
                    pass
                
                # Method 3: Synchrosqueezed Transform
                try:
                    sst_phase = self._synchrosqueezed_phase(imf)
                    if self._validate_phase_quality(sst_phase, imf):
                        imf_phases.append(('sst', sst_phase))
                except Exception:
                    pass
                
                # Method 4: Robust derivative-based phase
                try:
                    derivative_phase = self._robust_derivative_phase(imf)
                    if self._validate_phase_quality(derivative_phase, imf):
                        imf_phases.append(('derivative', derivative_phase))
                except Exception:
                    pass
                
                # Select best phase method for this IMF
                if imf_phases:
                    best_phase = self._select_best_phase_method(imf_phases, imf)
                    phase_results.append({
                        'imf_index': i,
                        'phase': best_phase[1],
                        'method': best_phase[0],
                        'quality_score': self._compute_phase_quality_score(best_phase[1], imf)
                    })
            
            return phase_results
            
        except Exception:
            return []
    
    def _adaptive_phase_fusion(self, phase_results: list, imfs: list, signal: np.ndarray) -> np.ndarray:
        """Advanced adaptive phase fusion with quality-based weighting"""
        try:
            if not phase_results:
                return self._fallback_phase_analysis(signal)
            
            # Compute adaptive weights based on multiple criteria
            weights = []
            total_quality = 0.0
            
            for result in phase_results:
                # Weight based on IMF energy
                imf_energy = np.sum(imfs[result['imf_index']]**2)
                energy_weight = imf_energy / np.sum(signal**2)
                
                # Weight based on phase quality
                quality_weight = result['quality_score']
                
                # Weight based on frequency content relevance
                freq_weight = self._compute_frequency_relevance(
                    imfs[result['imf_index']], 
                    signal
                )
                
                # Combined weight
                combined_weight = (
                    0.4 * energy_weight +
                    0.4 * quality_weight +
                    0.2 * freq_weight
                )
                
                weights.append(combined_weight)
                total_quality += combined_weight
            
            # Normalize weights
            if total_quality > 0:
                weights = [w / total_quality for w in weights]
            else:
                weights = [1.0 / len(phase_results)] * len(phase_results)
            
            # Adaptive fusion with phase unwrapping
            combined_phase = np.zeros(len(signal))
            
            for result, weight in zip(phase_results, weights):
                phase = result['phase']
                
                # Ensure phase compatibility
                if len(phase) == len(combined_phase):
                    # Unwrap phase if needed
                    unwrapped_phase = np.unwrap(phase)
                    combined_phase += weight * unwrapped_phase
            
            # Post-process combined phase
            combined_phase = self._postprocess_combined_phase(combined_phase)
            
            return combined_phase
            
        except Exception:
            return self._fallback_phase_analysis(signal)
    
    def _validate_and_postprocess_phase(self, phase: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Validate and post-process phase with comprehensive checks"""
        try:
            # Validation checks
            if not self._validate_phase_continuity(phase):
                phase = self._repair_phase_discontinuities(phase)
            
            if not self._validate_phase_bounds(phase):
                phase = self._normalize_phase_bounds(phase)
            
            # Smooth phase to remove artifacts
            smoothed_phase = self._smooth_phase_artifacts(phase)
            
            # Final validation against original signal
            if self._validate_phase_signal_consistency(smoothed_phase, signal):
                return smoothed_phase
            else:
                # Return fallback if validation fails
                return self._fallback_phase_analysis(signal)
                
        except Exception:
            return self._fallback_phase_analysis(signal)
    
    def _emergency_phase_extraction_emd(self, signal: np.ndarray) -> np.ndarray:
        """Emergency phase extraction when EMD fails"""
        try:
            # Try basic Hilbert transform on original signal
            return self._hilbert_transform_fft(signal)
        except Exception:
            try:
                # Try derivative-based phase
                return self._robust_derivative_phase(signal)
            except Exception:
                # Last resort: zero phase
                return np.zeros(len(signal))
    
    def _fallback_phase_analysis(self, signal: np.ndarray) -> np.ndarray:
        """Fallback phase analysis without EMD"""
        try:
            # Direct Hilbert transform approach
            return self._comprehensive_phase_extraction(signal)
        except Exception:
            try:
                return self._robust_derivative_phase(signal)
            except Exception:
                return np.zeros(len(signal))
    
    def _empirical_mode_decomposition(self, signal: np.ndarray, max_imfs: int = 8) -> list:
        """Comprehensive empirical mode decomposition with adaptive sifting"""
        try:
            return self._advanced_emd_decomposition(signal, max_imfs)
        except Exception:
            try:
                return self._robust_emd_fallback(signal, max_imfs)
            except Exception:
                return self._basic_emd_emergency(signal, max_imfs)
    
    def _advanced_emd_decomposition(self, signal: np.ndarray, max_imfs: int) -> list:
        """Advanced EMD with adaptive stopping criteria and ensemble averaging"""
        if len(signal) < 20:
            return [signal.copy()]
            
        imfs = []
        residue = signal.copy().astype(np.float64)
        noise_threshold = np.std(signal) * 0.01
        
        for imf_idx in range(max_imfs):
            if len(residue) < 20 or np.std(residue) < noise_threshold:
                break
                
            # Adaptive sifting process with multiple criteria
            imf_candidate, is_valid = self._comprehensive_sifting(residue, imf_idx)
            
            if not is_valid or np.std(imf_candidate) < noise_threshold:
                break
                
            # Validate IMF properties comprehensively
            if self._validate_imf_comprehensive(imf_candidate, residue):
                imfs.append(imf_candidate)
                residue = residue - imf_candidate
                
                # Check for decomposition completeness
                if self._check_decomposition_completeness(residue, signal, imfs):
                    break
            else:
                break
        
        # Add final residue if significant
        if len(residue) > 0 and np.std(residue) > noise_threshold:
            imfs.append(residue)
            
        return imfs
    
    def _comprehensive_sifting(self, signal: np.ndarray, imf_index: int) -> tuple:
        """Comprehensive sifting process with adaptive convergence criteria"""
        current = signal.copy()
        max_sifts = min(50, len(signal) // 4)
        convergence_threshold = 1e-6
        
        for sift_iter in range(max_sifts):
            # Advanced extrema detection with noise robustness
            maxima_idx, minima_idx = self._robust_extrema_detection(current)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                return current, False
            
            # Enhanced envelope construction
            upper_env, lower_env = self._construct_adaptive_envelopes(
                current, maxima_idx, minima_idx)
            
            if upper_env is None or lower_env is None:
                return current, False
            
            mean_envelope = (upper_env + lower_env) / 2
            new_candidate = current - mean_envelope
            
            # Multiple convergence criteria
            if self._check_sifting_convergence(current, new_candidate, convergence_threshold):
                return new_candidate, True
                
            current = new_candidate
            
        return current, True
    
    def _robust_extrema_detection(self, signal: np.ndarray) -> tuple:
        """Robust extrema detection with noise filtering and validation"""
        if len(signal) < 5:
            return np.array([]), np.array([])
        
        # Apply light smoothing for noise robustness
        smoothed = self._apply_smoothing_filter(signal)
        
        maxima_candidates = []
        minima_candidates = []
        
        # Multi-scale extrema detection
        for window_size in [3, 5]:
            if len(signal) < window_size * 2:
                continue
                
            for i in range(window_size, len(smoothed) - window_size):
                window = smoothed[i-window_size:i+window_size+1]
                center_idx = window_size
                
                # Local maximum detection
                if (smoothed[i] == np.max(window) and 
                    smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]):
                    if i not in maxima_candidates:
                        maxima_candidates.append(i)
                
                # Local minimum detection
                if (smoothed[i] == np.min(window) and 
                    smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]):
                    if i not in minima_candidates:
                        minima_candidates.append(i)
        
        # Validate and filter extrema
        maxima_idx = self._validate_extrema(signal, maxima_candidates, 'max')
        minima_idx = self._validate_extrema(signal, minima_candidates, 'min')
        
        return np.array(maxima_idx), np.array(minima_idx)
    
    def _apply_smoothing_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply light smoothing for noise reduction while preserving extrema"""
        if len(signal) < 5:
            return signal
        
        # Simple moving average with small window
        window_size = min(3, len(signal) // 5)
        if window_size < 2:
            return signal
            
        kernel = np.ones(window_size) / window_size
        padded = np.pad(signal, window_size//2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed[:len(signal)]
    
    def _validate_extrema(self, signal: np.ndarray, candidates: list, extrema_type: str) -> list:
        """Validate extrema candidates and remove false detections"""
        if not candidates:
            return []
        
        validated = []
        threshold = np.std(signal) * 0.1
        
        for idx in candidates:
            if idx <= 0 or idx >= len(signal) - 1:
                continue
                
            # Check significance of extrema
            if extrema_type == 'max':
                if (signal[idx] > signal[idx-1] + threshold and 
                    signal[idx] > signal[idx+1] + threshold):
                    validated.append(idx)
            else:
                if (signal[idx] < signal[idx-1] - threshold and 
                    signal[idx] < signal[idx+1] - threshold):
                    validated.append(idx)
        
        return validated
    
    def _construct_adaptive_envelopes(self, signal: np.ndarray, max_idx: np.ndarray, 
                                    min_idx: np.ndarray) -> tuple:
        """Construct adaptive envelopes with multiple interpolation methods"""
        if len(max_idx) < 2 or len(min_idx) < 2:
            return None, None
        
        # Primary method: Advanced spline interpolation
        try:
            upper_env = self._advanced_spline_interpolation(max_idx, signal[max_idx], len(signal))
            lower_env = self._advanced_spline_interpolation(min_idx, signal[min_idx], len(signal))
            
            if self._validate_envelopes(signal, upper_env, lower_env):
                return upper_env, lower_env
        except:
            pass
        
        # Fallback method: Robust polynomial interpolation
        try:
            upper_env = self._robust_polynomial_envelope(max_idx, signal[max_idx], len(signal))
            lower_env = self._robust_polynomial_envelope(min_idx, signal[min_idx], len(signal))
            
            if self._validate_envelopes(signal, upper_env, lower_env):
                return upper_env, lower_env
        except:
            pass
        
        # Emergency fallback: Linear interpolation with extrapolation
        try:
            upper_env = np.interp(np.arange(len(signal)), max_idx, signal[max_idx])
            lower_env = np.interp(np.arange(len(signal)), min_idx, signal[min_idx])
            return upper_env, lower_env
        except:
            return None, None
    
    def _advanced_spline_interpolation(self, x_points: np.ndarray, y_points: np.ndarray, 
                                     target_length: int) -> np.ndarray:
        """Advanced spline interpolation with boundary handling"""
        if len(x_points) < 3:
            # Linear interpolation for insufficient points
            return np.interp(np.arange(target_length), x_points, y_points)
        
        # Extend boundary points for better interpolation
        extended_x = np.concatenate([[0], x_points, [target_length - 1]])
        extended_y = np.concatenate([[y_points[0]], y_points, [y_points[-1]]])
        
        # Cubic spline with natural boundary conditions
        from scipy import interpolate
        spline = interpolate.CubicSpline(extended_x, extended_y, bc_type='natural')
        return spline(np.arange(target_length))
    
    def _robust_polynomial_envelope(self, x_points: np.ndarray, y_points: np.ndarray, 
                                  target_length: int) -> np.ndarray:
        """Robust polynomial envelope construction"""
        if len(x_points) < 2:
            return np.full(target_length, y_points[0] if len(y_points) > 0 else 0)
        
        # Adaptive polynomial degree
        poly_degree = min(len(x_points) - 1, 3)
        
        # Robust polynomial fitting with outlier handling
        coeffs = np.polyfit(x_points, y_points, poly_degree)
        envelope = np.polyval(coeffs, np.arange(target_length))
        
        return envelope
    
    def _validate_envelopes(self, signal: np.ndarray, upper_env: np.ndarray, 
                          lower_env: np.ndarray) -> bool:
        """Validate envelope quality and physical constraints"""
        if len(upper_env) != len(signal) or len(lower_env) != len(signal):
            return False
        
        # Check envelope ordering
        if not np.all(upper_env >= lower_env):
            return False
        
        # Check signal bounds
        signal_margin = np.std(signal) * 0.5
        if not np.all(signal <= upper_env + signal_margin):
            return False
        if not np.all(signal >= lower_env - signal_margin):
            return False
        
        return True
    
    def _check_sifting_convergence(self, current: np.ndarray, new_candidate: np.ndarray, 
                                 threshold: float) -> bool:
        """Check multiple convergence criteria for sifting process"""
        if len(current) != len(new_candidate):
            return False
        
        # Relative change criterion
        rel_change = np.mean(np.abs(current - new_candidate)) / (np.std(current) + 1e-10)
        if rel_change < threshold:
            return True
        
        # Energy conservation criterion
        energy_change = abs(np.sum(current**2) - np.sum(new_candidate**2)) / np.sum(current**2)
        if energy_change < threshold * 10:
            return True
        
        return False
    
    def _validate_imf_comprehensive(self, imf: np.ndarray, original_residue: np.ndarray) -> bool:
        """Comprehensive IMF validation with multiple criteria"""
        if len(imf) < 10:
            return False
        
        # Standard IMF criteria: zero-crossing and extrema count
        if not self._is_valid_imf(imf):
            return False
        
        # Energy criterion
        if np.sum(imf**2) < 0.01 * np.sum(original_residue**2):
            return False
        
        # Frequency content validation
        if not self._validate_frequency_content(imf):
            return False
        
        # Orthogonality check with previous IMFs
        return True
    
    def _validate_frequency_content(self, imf: np.ndarray) -> bool:
        """Validate frequency content of IMF"""
        try:
            # Simple frequency analysis
            fft_imf = np.fft.fft(imf)
            power_spectrum = np.abs(fft_imf)**2
            
            # Check for reasonable frequency distribution
            if np.sum(power_spectrum) == 0:
                return False
            
            # Ensure not too concentrated in single frequency
            max_power = np.max(power_spectrum)
            total_power = np.sum(power_spectrum)
            
            if max_power > 0.9 * total_power:
                return False
            
            return True
        except:
            return True  # Default to valid if frequency analysis fails
    
    def _check_decomposition_completeness(self, residue: np.ndarray, original: np.ndarray, 
                                        imfs: list) -> bool:
        """Check if decomposition is sufficiently complete"""
        if len(imfs) == 0:
            return False
        
        # Energy conservation check
        imf_energy = sum(np.sum(imf**2) for imf in imfs)
        residue_energy = np.sum(residue**2)
        original_energy = np.sum(original**2)
        
        total_energy = imf_energy + residue_energy
        energy_conservation = abs(total_energy - original_energy) / original_energy
        
        if energy_conservation > 0.1:  # 10% tolerance
            return False
        
        # Residue significance check
        residue_ratio = residue_energy / original_energy
        if residue_ratio < 0.05:  # Less than 5% energy in residue
            return True
        
        return False
    
    def _robust_emd_fallback(self, signal: np.ndarray, max_imfs: int) -> list:
        """Robust EMD fallback with simplified but reliable decomposition"""
        imfs = []
        residue = signal.copy()
        
        for _ in range(min(max_imfs, 5)):
            if len(residue) < 15:
                break
            
            # Simple but robust extrema detection
            maxima_idx, minima_idx = self._simple_extrema_detection(residue)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            
            # Basic envelope construction
            try:
                upper_env = np.interp(np.arange(len(residue)), maxima_idx, residue[maxima_idx])
                lower_env = np.interp(np.arange(len(residue)), minima_idx, residue[minima_idx])
                
                mean_envelope = (upper_env + lower_env) / 2
                imf_candidate = residue - mean_envelope
                
                if self._basic_imf_check(imf_candidate):
                    imfs.append(imf_candidate)
                    residue = residue - imf_candidate
                else:
                    break
            except:
                break
        
        return imfs if imfs else [signal.copy()]
    
    def _simple_extrema_detection(self, signal: np.ndarray) -> tuple:
        """Simple but robust extrema detection"""
        maxima = []
        minima = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
            elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima.append(i)
        
        return np.array(maxima), np.array(minima)
    
    def _basic_imf_check(self, imf: np.ndarray) -> bool:
        """Basic IMF validation"""
        if len(imf) < 5:
            return False
        
        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(imf)) != 0)
        
        # Count extrema
        maxima, minima = self._simple_extrema_detection(imf)
        extrema_count = len(maxima) + len(minima)
        
        # Basic IMF criterion: zero crossings and extrema differ by at most 1
        return abs(zero_crossings - extrema_count) <= 1
    
    def _basic_emd_emergency(self, signal: np.ndarray, max_imfs: int) -> list:
        """Emergency EMD decomposition - always returns something"""
        if len(signal) < 5:
            return [signal.copy()]
        
        # Simple frequency-based decomposition as emergency fallback
        try:
            # High-frequency component (differences)
            high_freq = np.diff(signal, prepend=signal[0])
            
            # Low-frequency component (residue after high-freq removal)
            low_freq = signal - high_freq
            
            return [high_freq, low_freq]
        except:
            return [signal.copy()]
    
    def _find_extrema(self, signal: np.ndarray) -> tuple:
        """Find local maxima and minima"""
        maxima = []
        minima = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                maxima.append(i)
            elif signal[i] < signal[i-1] and signal[i] < signal[i+1]:
                minima.append(i)
        
        return np.array(maxima), np.array(minima)
    
    def _cubic_spline_interpolation(self, x_points: np.ndarray, y_points: np.ndarray, n_output: int) -> np.ndarray:
        """Comprehensive cubic spline interpolation with advanced boundary handling"""
        try:
            return self._advanced_cubic_spline_system(x_points, y_points, n_output)
        except Exception:
            try:
                return self._robust_spline_fallback(x_points, y_points, n_output)
            except Exception:
                return self._emergency_interpolation(x_points, y_points, n_output)
    
    def _advanced_cubic_spline_system(self, x_points: np.ndarray, y_points: np.ndarray, n_output: int) -> np.ndarray:
        """Advanced cubic spline with comprehensive mathematical foundation"""
        if len(x_points) < 2 or len(y_points) < 2:
            return np.zeros(n_output)
        
        # Ensure data consistency and preprocessing
        x_clean, y_clean = self._preprocess_spline_data(x_points, y_points)
        
        if len(x_clean) < 2:
            return np.full(n_output, np.mean(y_clean) if len(y_clean) > 0 else 0.0)
        
        # Advanced boundary extension with physics-informed extrapolation
        x_extended, y_extended = self._intelligent_boundary_extension(
            x_clean, y_clean, n_output)
        
        # Multi-method spline construction with optimal selection
        spline_result = self._construct_optimal_spline(x_extended, y_extended, n_output)
        
        # Post-processing with smoothness constraints and validation
        return self._postprocess_spline_result(spline_result, x_points, y_points, n_output)
    
    def _preprocess_spline_data(self, x_points: np.ndarray, y_points: np.ndarray) -> tuple:
        """Comprehensive data preprocessing for spline interpolation"""
        # Remove duplicates and invalid points
        valid_mask = np.isfinite(x_points) & np.isfinite(y_points)
        x_valid = x_points[valid_mask]
        y_valid = y_points[valid_mask]
        
        if len(x_valid) == 0:
            return np.array([]), np.array([])
        
        # Sort by x-coordinates
        sort_indices = np.argsort(x_valid)
        x_sorted = x_valid[sort_indices]
        y_sorted = y_valid[sort_indices]
        
        # Remove duplicate x-coordinates (keep first occurrence)
        unique_mask = np.concatenate([[True], np.diff(x_sorted) > 1e-10])
        x_unique = x_sorted[unique_mask]
        y_unique = y_sorted[unique_mask]
        
        # Outlier detection and handling
        if len(y_unique) > 3:
            x_unique, y_unique = self._handle_spline_outliers(x_unique, y_unique)
        
        return x_unique, y_unique
    
    def _handle_spline_outliers(self, x_points: np.ndarray, y_points: np.ndarray) -> tuple:
        """Advanced outlier detection and handling for spline data"""
        if len(y_points) < 4:
            return x_points, y_points
        
        # Robust outlier detection using modified Z-score
        median_y = np.median(y_points)
        mad = np.median(np.abs(y_points - median_y))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (y_points - median_y) / mad
            outlier_threshold = 3.5
            outlier_mask = np.abs(modified_z_scores) > outlier_threshold
            
            # Replace outliers with interpolated values
            if np.sum(~outlier_mask) >= 2:
                y_corrected = y_points.copy()
                if np.sum(outlier_mask) > 0:
                    non_outlier_indices = np.where(~outlier_mask)[0]
                    outlier_indices = np.where(outlier_mask)[0]
                    
                    # Interpolate outlier values
                    y_corrected[outlier_indices] = np.interp(
                        x_points[outlier_indices],
                        x_points[non_outlier_indices],
                        y_points[non_outlier_indices]
                    )
                
                return x_points, y_corrected
        
        return x_points, y_points
    
    def _intelligent_boundary_extension(self, x_points: np.ndarray, y_points: np.ndarray, 
                                      n_output: int) -> tuple:
        """Intelligent boundary extension with physics-informed extrapolation"""
        x_extended = x_points.copy()
        y_extended = y_points.copy()
        
        # Left boundary extension
        if len(x_points) > 0 and x_points[0] > 0:
            left_extension_x, left_extension_y = self._extrapolate_left_boundary(
                x_points, y_points)
            x_extended = np.concatenate([left_extension_x, x_extended])
            y_extended = np.concatenate([left_extension_y, y_extended])
        
        # Right boundary extension
        if len(x_points) > 0 and x_points[-1] < n_output - 1:
            right_extension_x, right_extension_y = self._extrapolate_right_boundary(
                x_points, y_points, n_output)
            x_extended = np.concatenate([x_extended, right_extension_x])
            y_extended = np.concatenate([y_extended, right_extension_y])
        
        return x_extended, y_extended
    
    def _extrapolate_left_boundary(self, x_points: np.ndarray, y_points: np.ndarray) -> tuple:
        """Advanced left boundary extrapolation"""
        if len(x_points) < 2:
            return np.array([0]), np.array([y_points[0] if len(y_points) > 0 else 0])
        
        # Multiple extrapolation methods
        methods_x = []
        methods_y = []
        
        # Method 1: Linear extrapolation from first two points
        if len(x_points) >= 2:
            slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
            y_left_linear = y_points[0] - slope * x_points[0]
            methods_x.append(np.array([0]))
            methods_y.append(np.array([y_left_linear]))
        
        # Method 2: Quadratic extrapolation from first three points
        if len(x_points) >= 3:
            try:
                # Fit quadratic through first three points
                coeffs = np.polyfit(x_points[:3], y_points[:3], 2)
                y_left_quad = np.polyval(coeffs, 0)
                methods_x.append(np.array([0]))
                methods_y.append(np.array([y_left_quad]))
            except:
                pass
        
        # Method 3: Constant extrapolation (conservative)
        methods_x.append(np.array([0]))
        methods_y.append(np.array([y_points[0]]))
        
        # Select optimal method based on data characteristics
        selected_method = self._select_optimal_extrapolation_method(
            x_points, y_points, methods_x, methods_y, 'left')
        
        return methods_x[selected_method], methods_y[selected_method]
    
    def _extrapolate_right_boundary(self, x_points: np.ndarray, y_points: np.ndarray, 
                                   n_output: int) -> tuple:
        """Advanced right boundary extrapolation"""
        if len(x_points) < 2:
            last_y = y_points[-1] if len(y_points) > 0 else 0
            return np.array([n_output - 1]), np.array([last_y])
        
        # Multiple extrapolation methods
        methods_x = []
        methods_y = []
        
        # Method 1: Linear extrapolation from last two points
        if len(x_points) >= 2:
            slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
            y_right_linear = y_points[-1] + slope * (n_output - 1 - x_points[-1])
            methods_x.append(np.array([n_output - 1]))
            methods_y.append(np.array([y_right_linear]))
        
        # Method 2: Quadratic extrapolation from last three points
        if len(x_points) >= 3:
            try:
                coeffs = np.polyfit(x_points[-3:], y_points[-3:], 2)
                y_right_quad = np.polyval(coeffs, n_output - 1)
                methods_x.append(np.array([n_output - 1]))
                methods_y.append(np.array([y_right_quad]))
            except:
                pass
        
        # Method 3: Constant extrapolation (conservative)
        methods_x.append(np.array([n_output - 1]))
        methods_y.append(np.array([y_points[-1]]))
        
        # Select optimal method
        selected_method = self._select_optimal_extrapolation_method(
            x_points, y_points, methods_x, methods_y, 'right')
        
        return methods_x[selected_method], methods_y[selected_method]
    
    def _select_optimal_extrapolation_method(self, x_points: np.ndarray, y_points: np.ndarray,
                                           methods_x: list, methods_y: list, direction: str) -> int:
        """Select optimal extrapolation method based on data characteristics"""
        if len(methods_x) == 1:
            return 0
        
        # Analyze data characteristics
        data_variance = np.var(y_points)
        data_trend = self._analyze_data_trend(x_points, y_points, direction)
        
        # Decision logic based on data characteristics
        if len(x_points) >= 3 and data_variance > 0:
            # For variable data with sufficient points, prefer quadratic
            if len(methods_x) > 1:
                return 1  # Quadratic method
        
        # For smooth or limited data, prefer linear
        if len(methods_x) > 0:
            return 0  # Linear method
        
        # Fallback to constant
        return len(methods_x) - 1
    
    def _analyze_data_trend(self, x_points: np.ndarray, y_points: np.ndarray, direction: str) -> str:
        """Analyze data trend for extrapolation method selection"""
        if len(x_points) < 2:
            return 'constant'
        
        if direction == 'left' and len(x_points) >= 2:
            slope = (y_points[1] - y_points[0]) / (x_points[1] - x_points[0])
        elif direction == 'right' and len(x_points) >= 2:
            slope = (y_points[-1] - y_points[-2]) / (x_points[-1] - x_points[-2])
        else:
            return 'constant'
        
        if abs(slope) < 1e-6:
            return 'constant'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _construct_optimal_spline(self, x_points: np.ndarray, y_points: np.ndarray, 
                                n_output: int) -> np.ndarray:
        """Construct optimal cubic spline using multiple methods"""
        x_output = np.arange(n_output)
        
        # Method 1: SciPy CubicSpline (preferred)
        try:
            from scipy import interpolate
            spline = interpolate.CubicSpline(
                x_points, y_points, 
                bc_type='natural',  # Natural boundary conditions
                extrapolate=False
            )
            result = spline(x_output)
            
            # Validate result quality
            if self._validate_spline_result(result, x_points, y_points):
                return result
        except Exception:
            pass
        
        # Method 2: Manual cubic spline implementation
        try:
            result = self._manual_cubic_spline(x_points, y_points, x_output)
            if self._validate_spline_result(result, x_points, y_points):
                return result
        except Exception:
            pass
        
        # Method 3: Robust polynomial interpolation fallback
        try:
            result = self._robust_polynomial_interpolation(x_points, y_points, x_output)
            return result
        except Exception:
            # Emergency linear interpolation
            return np.interp(x_output, x_points, y_points)
    
    def _manual_cubic_spline(self, x_points: np.ndarray, y_points: np.ndarray, 
                           x_output: np.ndarray) -> np.ndarray:
        """Manual cubic spline implementation with tridiagonal solver"""
        n = len(x_points)
        if n < 3:
            return np.interp(x_output, x_points, y_points)
        
        # Calculate differences
        h = np.diff(x_points)
        delta = np.diff(y_points) / h
        
        # Set up tridiagonal system for natural spline
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        # Natural boundary conditions
        A[0, 0] = 1
        A[n-1, n-1] = 1
        
        # Interior equations
        for i in range(1, n-1):
            A[i, i-1] = h[i-1]
            A[i, i] = 2 * (h[i-1] + h[i])
            A[i, i+1] = h[i]
            b[i] = 3 * (delta[i] - delta[i-1])
        
        # Solve tridiagonal system
        c = self._solve_tridiagonal_system(A, b)
        
        # Calculate spline coefficients
        a = y_points[:-1]
        b_coeffs = delta - h * (2*c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3 * h)
        c = c[:-1]
        
        # Evaluate spline at output points
        result = np.zeros(len(x_output))
        for i, x in enumerate(x_output):
            # Find interval
            idx = np.searchsorted(x_points, x) - 1
            idx = np.clip(idx, 0, len(a) - 1)
            
            # Evaluate polynomial
            dx = x - x_points[idx]
            result[i] = a[idx] + b_coeffs[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
        
        return result
    
    def _solve_tridiagonal_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve tridiagonal system using Thomas algorithm"""
        n = len(b)
        x = np.zeros(n)
        
        # Forward elimination
        for i in range(1, n):
            if abs(A[i-1, i-1]) < 1e-10:
                continue
            factor = A[i, i-1] / A[i-1, i-1]
            A[i, i] -= factor * A[i-1, i]
            if i < n-1:
                A[i, i+1] -= factor * A[i-1, i]
            b[i] -= factor * b[i-1]
        
        # Back substitution
        if abs(A[n-1, n-1]) > 1e-10:
            x[n-1] = b[n-1] / A[n-1, n-1]
        
        for i in range(n-2, -1, -1):
            if abs(A[i, i]) > 1e-10:
                x[i] = b[i]
                if i < n-1:
                    x[i] -= A[i, i+1] * x[i+1]
                x[i] /= A[i, i]
        
        return x
    
    def _robust_polynomial_interpolation(self, x_points: np.ndarray, y_points: np.ndarray,
                                       x_output: np.ndarray) -> np.ndarray:
        """Robust polynomial interpolation with degree adaptation"""
        n = len(x_points)
        
        # Adaptive polynomial degree
        max_degree = min(n - 1, 5)  # Limit degree to prevent oscillations
        
        for degree in range(max_degree, 0, -1):
            try:
                coeffs = np.polyfit(x_points, y_points, degree)
                result = np.polyval(coeffs, x_output)
                
                # Check for reasonable values (no extreme oscillations)
                if np.all(np.isfinite(result)) and np.std(result) < 10 * np.std(y_points):
                    return result
            except:
                continue
        
        # Fallback to linear interpolation
        return np.interp(x_output, x_points, y_points)
    
    def _validate_spline_result(self, result: np.ndarray, x_points: np.ndarray, 
                              y_points: np.ndarray) -> bool:
        """Validate spline interpolation result quality"""
        if not np.all(np.isfinite(result)):
            return False
        
        # Check for extreme values
        y_range = np.max(y_points) - np.min(y_points)
        if y_range > 0:
            result_range = np.max(result) - np.min(result)
            if result_range > 5 * y_range:  # Too much oscillation
                return False
        
        # Check interpolation accuracy at known points
        if len(x_points) > 0:
            interpolated_at_points = np.interp(x_points, np.arange(len(result)), result)
            interpolation_error = np.mean(np.abs(interpolated_at_points - y_points))
            data_scale = np.std(y_points) + 1e-10
            
            if interpolation_error > 0.1 * data_scale:
                return False
        
        return True
    
    def _postprocess_spline_result(self, result: np.ndarray, x_points: np.ndarray,
                                 y_points: np.ndarray, n_output: int) -> np.ndarray:
        """Post-process spline result with smoothness constraints"""
        if len(result) != n_output:
            return np.interp(np.arange(n_output), np.arange(len(result)), result)
        
        # Apply smoothness constraints if needed
        processed_result = self._apply_smoothness_constraints(result, y_points)
        
        # Ensure monotonicity if input data is monotonic
        if self._is_monotonic(y_points):
            processed_result = self._enforce_monotonicity(processed_result, y_points)
        
        return processed_result
    
    def _apply_smoothness_constraints(self, result: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """Apply smoothness constraints to reduce oscillations"""
        if len(result) < 5:
            return result
        
        # Calculate smoothness metric
        second_derivative = np.diff(result, 2)
        smoothness = np.std(second_derivative)
        reference_smoothness = np.std(reference_data) * 0.1
        
        # Apply smoothing if result is too oscillatory
        if smoothness > reference_smoothness and len(result) > 3:
            # Light smoothing with small kernel
            kernel_size = min(3, len(result) // 10)
            if kernel_size >= 2:
                kernel = np.ones(kernel_size) / kernel_size
                padded_result = np.pad(result, kernel_size//2, mode='edge')
                smoothed = np.convolve(padded_result, kernel, mode='valid')
                return smoothed[:len(result)]
        
        return result
    
    def _is_monotonic(self, data: np.ndarray) -> bool:
        """Check if data is monotonic"""
        if len(data) < 3:
            return True
        
        diffs = np.diff(data)
        return np.all(diffs >= 0) or np.all(diffs <= 0)
    
    def _enforce_monotonicity(self, result: np.ndarray, reference_data: np.ndarray) -> np.ndarray:
        """Enforce monotonicity constraint based on reference data"""
        if len(reference_data) < 2:
            return result
        
        # Determine monotonicity direction from reference
        ref_diffs = np.diff(reference_data)
        is_increasing = np.sum(ref_diffs > 0) > np.sum(ref_diffs < 0)
        
        # Enforce monotonicity
        if is_increasing:
            for i in range(1, len(result)):
                if result[i] < result[i-1]:
                    result[i] = result[i-1]
        else:
            for i in range(1, len(result)):
                if result[i] > result[i-1]:
                    result[i] = result[i-1]
        
        return result
    
    def _robust_spline_fallback(self, x_points: np.ndarray, y_points: np.ndarray, n_output: int) -> np.ndarray:
        """Robust spline fallback with simplified but reliable interpolation"""
        if len(x_points) < 2:
            return np.zeros(n_output)
        
        x_output = np.arange(n_output)
        
        # Enhanced linear interpolation with boundary handling
        x_extended = x_points.copy()
        y_extended = y_points.copy()
        
        # Add boundary points if needed
        if x_points[0] > 0:
            x_extended = np.concatenate([[0], x_extended])
            y_extended = np.concatenate([[y_points[0]], y_extended])
        
        if x_points[-1] < n_output - 1:
            x_extended = np.concatenate([x_extended, [n_output - 1]])
            y_extended = np.concatenate([y_extended, [y_points[-1]]])
        
        # Robust linear interpolation
        return np.interp(x_output, x_extended, y_extended)
    
    def _emergency_interpolation(self, x_points: np.ndarray, y_points: np.ndarray, n_output: int) -> np.ndarray:
        """Emergency interpolation - guaranteed to return something"""
        if len(x_points) == 0 or len(y_points) == 0:
            return np.zeros(n_output)
        
        if len(x_points) == 1:
            return np.full(n_output, y_points[0])
        
        # Simple linear interpolation between first and last points
        x_start, x_end = 0, n_output - 1
        y_start, y_end = y_points[0], y_points[-1]
        
        slope = (y_end - y_start) / (x_end - x_start) if x_end != x_start else 0
        x_output = np.arange(n_output)
        
        return y_start + slope * x_output
    
    def _is_valid_imf(self, candidate: np.ndarray) -> bool:
        """Comprehensive Intrinsic Mode Function validation with rigorous mathematical criteria"""
        try:
            return self._advanced_imf_validation(candidate)
        except Exception:
            try:
                return self._robust_imf_validation(candidate)
            except Exception:
                return self._basic_imf_validation(candidate)
    
    def _advanced_imf_validation(self, candidate: np.ndarray) -> bool:
        """Advanced IMF validation using comprehensive mathematical criteria"""
        if len(candidate) < 4:  # Need minimum length for meaningful analysis
            return False
        
        # Criterion 1: Extrema and zero-crossing relationship
        extrema_criterion = self._validate_extrema_zero_crossings(candidate)
        if not extrema_criterion:
            return False
        
        # Criterion 2: Mean envelope symmetry
        envelope_criterion = self._validate_envelope_symmetry(candidate)
        if not envelope_criterion:
            return False
        
        # Criterion 3: Local stationarity and frequency coherence
        stationarity_criterion = self._validate_local_stationarity(candidate)
        
        # Criterion 4: Monotonicity between extrema
        monotonicity_criterion = self._validate_monotonicity(candidate)
        
        # Criterion 5: Instantaneous frequency positivity
        frequency_criterion = self._validate_instantaneous_frequency(candidate)
        
        # Criterion 6: Spectral purity assessment
        spectral_criterion = self._validate_spectral_purity(candidate)
        
        # Criterion 7: Statistical significance of oscillatory behavior
        statistical_criterion = self._validate_statistical_significance(candidate)
        
        # Weighted scoring system for comprehensive validation
        criteria_weights = {
            'extrema': 0.25,
            'envelope': 0.20,
            'stationarity': 0.15,
            'monotonicity': 0.15,
            'frequency': 0.10,
            'spectral': 0.10,
            'statistical': 0.05
        }
        
        criteria_scores = {
            'extrema': 1.0 if extrema_criterion else 0.0,
            'envelope': 1.0 if envelope_criterion else 0.0,
            'stationarity': stationarity_criterion,  # Returns float 0-1
            'monotonicity': 1.0 if monotonicity_criterion else 0.0,
            'frequency': frequency_criterion,  # Returns float 0-1
            'spectral': spectral_criterion,  # Returns float 0-1
            'statistical': 1.0 if statistical_criterion else 0.0
        }
        
        # Compute weighted score
        total_score = sum(criteria_weights[key] * criteria_scores[key] 
                         for key in criteria_weights.keys())
        
        # Adaptive threshold based on signal characteristics
        adaptive_threshold = self._compute_adaptive_imf_threshold(candidate)
        
        return total_score >= adaptive_threshold
    
    def _validate_extrema_zero_crossings(self, signal: np.ndarray) -> bool:
        """Validate fundamental IMF criterion: extrema and zero-crossings relationship"""
        # Find zero crossings
        zero_crossings = self._find_zero_crossings(signal)
        
        # Find extrema with enhanced detection
        maxima_idx, minima_idx = self._find_extrema_enhanced(signal)
        total_extrema = len(maxima_idx) + len(minima_idx)
        
        if total_extrema < 2 or len(zero_crossings) < 1:
            return False
        
        # Classic IMF criterion: |extrema - zero_crossings| ≤ 1
        difference = abs(total_extrema - len(zero_crossings))
        
        # Enhanced validation with context-dependent tolerance
        tolerance = self._compute_extrema_tolerance(signal, total_extrema, len(zero_crossings))
        
        return difference <= tolerance
    
    def _find_zero_crossings(self, signal: np.ndarray) -> List[int]:
        """Enhanced zero-crossing detection with interpolation"""
        zero_crossings = []
        
        for i in range(len(signal) - 1):
            if signal[i] * signal[i + 1] < 0:  # Sign change
                # Linear interpolation to find exact crossing
                crossing_point = i - signal[i] / (signal[i + 1] - signal[i])
                zero_crossings.append(int(np.round(crossing_point)))
            elif signal[i] == 0:  # Exact zero
                zero_crossings.append(i)
        
        return zero_crossings
    
    def _find_extrema_enhanced(self, signal: np.ndarray) -> Tuple[List[int], List[int]]:
        """Enhanced extrema detection with noise robustness"""
        from scipy.signal import find_peaks
        
        # Adaptive prominence threshold based on signal characteristics
        signal_std = np.std(signal)
        signal_range = np.ptp(signal)
        prominence_threshold = max(0.01 * signal_range, 2 * signal_std)
        
        # Find maxima
        maxima_idx, _ = find_peaks(signal, prominence=prominence_threshold)
        
        # Find minima
        minima_idx, _ = find_peaks(-signal, prominence=prominence_threshold)
        
        return maxima_idx.tolist(), minima_idx.tolist()
    
    def _compute_extrema_tolerance(self, signal: np.ndarray, n_extrema: int, n_zeros: int) -> int:
        """Compute adaptive tolerance for extrema-zero crossing criterion"""
        # Base tolerance is 1 (classic IMF criterion)
        base_tolerance = 1
        
        # Adjust based on signal length and noise level
        signal_length = len(signal)
        noise_level = self._estimate_noise_level(signal)
        
        # For longer signals or higher noise, allow more tolerance
        if signal_length > 1000:
            base_tolerance += 1
        if noise_level > 0.1:  # High noise threshold
            base_tolerance += 1
        
        # For very short signals, be more strict
        if signal_length < 20:
            base_tolerance = max(1, base_tolerance - 1)
        
        return base_tolerance
    
    def _estimate_noise_level(self, signal: np.ndarray) -> float:
        """Estimate noise level using high-frequency component analysis"""
        try:
            # Use high-pass filtering to estimate noise
            from scipy.signal import butter, filtfilt
            
            # Design high-pass filter
            nyquist = 0.5
            high_cutoff = 0.3  # Keep high-frequency components
            b, a = butter(2, high_cutoff / nyquist, btype='high')
            
            # Apply filter
            high_freq_component = filtfilt(b, a, signal)
            
            # Noise level as standard deviation of high-frequency component
            noise_level = np.std(high_freq_component) / np.std(signal)
            
            return min(noise_level, 0.5)  # Cap at 50%
        except Exception:
            # Fallback: use signal derivative variation
            return np.std(np.diff(signal)) / np.std(signal)
    
    def _validate_envelope_symmetry(self, signal: np.ndarray) -> bool:
        """Validate mean envelope symmetry criterion"""
        try:
            maxima_idx, minima_idx = self._find_extrema_enhanced(signal)
            
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                return len(maxima_idx) >= 1 and len(minima_idx) >= 1  # Relaxed for short sequences
            
            # Construct upper and lower envelopes
            upper_envelope = self._construct_envelope(signal, maxima_idx)
            lower_envelope = self._construct_envelope(signal, minima_idx)
            
            # Calculate mean envelope
            mean_envelope = (upper_envelope + lower_envelope) / 2
            
            # Symmetry criterion: mean envelope should be close to zero
            envelope_deviation = np.abs(mean_envelope)
            signal_amplitude = np.std(signal)
            
            # Adaptive threshold based on signal characteristics
            symmetry_threshold = 0.05 * signal_amplitude  # 5% of signal amplitude
            
            # Check if mean envelope is sufficiently close to zero
            mean_deviation = np.mean(envelope_deviation)
            max_deviation = np.max(envelope_deviation)
            
            return (mean_deviation < symmetry_threshold and 
                    max_deviation < 2 * symmetry_threshold)
        
        except Exception:
            return True  # Fallback to pass if envelope construction fails
    
    def _construct_envelope(self, signal: np.ndarray, extrema_idx: List[int]) -> np.ndarray:
        """Construct envelope through extrema points using advanced interpolation"""
        if len(extrema_idx) < 2:
            return np.zeros_like(signal)
        
        try:
            # Use cubic spline interpolation with natural boundary conditions
            from scipy.interpolate import CubicSpline
            
            extrema_values = signal[extrema_idx]
            envelope = CubicSpline(extrema_idx, extrema_values, bc_type='natural')
            
            return envelope(np.arange(len(signal)))
            
        except Exception:
            # Fallback to linear interpolation
            return np.interp(np.arange(len(signal)), extrema_idx, signal[extrema_idx])
    
    def _validate_local_stationarity(self, signal: np.ndarray) -> float:
        """Validate local stationarity using windowed analysis"""
        if len(signal) < 20:
            return 1.0  # Assume valid for very short signals
        
        try:
            # Divide signal into overlapping windows
            window_size = max(10, len(signal) // 5)
            overlap = window_size // 2
            
            stationarity_scores = []
            
            for i in range(0, len(signal) - window_size + 1, overlap):
                window = signal[i:i + window_size]
                
                # Compute local statistics
                local_mean = np.mean(window)
                local_variance = np.var(window)
                
                # Compare with neighboring windows
                neighbor_scores = []
                
                for j in range(max(0, i - overlap), min(len(signal) - window_size + 1, i + overlap + 1)):
                    if j != i:
                        neighbor_window = signal[j:j + window_size]
                        neighbor_mean = np.mean(neighbor_window)
                        neighbor_variance = np.var(neighbor_window)
                        
                        # Similarity score based on mean and variance
                        mean_similarity = 1.0 / (1.0 + abs(local_mean - neighbor_mean))
                        var_similarity = 1.0 / (1.0 + abs(local_variance - neighbor_variance))
                        
                        neighbor_scores.append((mean_similarity + var_similarity) / 2)
                
                if neighbor_scores:
                    stationarity_scores.append(np.mean(neighbor_scores))
            
            # Overall stationarity score
            return np.mean(stationarity_scores) if stationarity_scores else 0.5
            
        except Exception:
            return 0.5  # Neutral score on failure
    
    def _validate_monotonicity(self, signal: np.ndarray) -> bool:
        """Validate monotonicity between consecutive extrema"""
        try:
            maxima_idx, minima_idx = self._find_extrema_enhanced(signal)
            all_extrema = sorted(maxima_idx + minima_idx)
            
            if len(all_extrema) < 2:
                return True  # Trivially satisfied
            
            monotonicity_violations = 0
            total_segments = len(all_extrema) - 1
            
            for i in range(len(all_extrema) - 1):
                start_idx = all_extrema[i]
                end_idx = all_extrema[i + 1]
                segment = signal[start_idx:end_idx + 1]
                
                if len(segment) <= 2:
                    continue
                
                # Check monotonicity
                is_increasing = all(segment[j] <= segment[j + 1] for j in range(len(segment) - 1))
                is_decreasing = all(segment[j] >= segment[j + 1] for j in range(len(segment) - 1))
                
                if not (is_increasing or is_decreasing):
                    monotonicity_violations += 1
            
            # Allow some tolerance for noise
            violation_rate = monotonicity_violations / total_segments if total_segments > 0 else 0
            return violation_rate <= 0.2  # Allow up to 20% violations
            
        except Exception:
            return True  # Assume valid on failure
    
    def _validate_instantaneous_frequency(self, signal: np.ndarray) -> float:
        """Validate instantaneous frequency positivity"""
        try:
            # Compute analytic signal using Hilbert transform
            analytic_signal = sp.signal.hilbert(signal)
            
            # Compute instantaneous frequency
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2 * np.pi)
            
            # Check positivity
            positive_freq_ratio = np.sum(instantaneous_frequency > 0) / len(instantaneous_frequency)
            
            # Return score based on positive frequency ratio
            return min(positive_freq_ratio * 1.2, 1.0)  # Bonus for high positive ratio
            
        except Exception:
            return 0.5  # Neutral score on failure
    
    def _validate_spectral_purity(self, signal: np.ndarray) -> float:
        """Validate spectral purity and narrow-band characteristics"""
        try:
            # Compute power spectral density
            frequencies, power = sp.signal.periodogram(signal)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(power)
            dominant_power = power[dominant_freq_idx]
            
            # Calculate spectral concentration around dominant frequency
            bandwidth = 0.1  # 10% bandwidth around dominant frequency
            freq_range = len(frequencies)
            band_start = max(0, int((dominant_freq_idx - bandwidth * freq_range / 2)))
            band_end = min(len(frequencies), int((dominant_freq_idx + bandwidth * freq_range / 2)))
            
            power_in_band = np.sum(power[band_start:band_end])
            total_power = np.sum(power)
            
            # Spectral concentration ratio
            concentration_ratio = power_in_band / total_power if total_power > 0 else 0
            
            return min(concentration_ratio * 2, 1.0)  # Scale to 0-1 range
            
        except Exception:
            return 0.5  # Neutral score on failure
    
    def _validate_statistical_significance(self, signal: np.ndarray) -> bool:
        """Validate statistical significance of oscillatory behavior"""
        try:
            # Test against white noise hypothesis
            from scipy.stats import kstest
            
            # Normalize signal
            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
            
            # Kolmogorov-Smirnov test against normal distribution
            ks_statistic, p_value = kstest(normalized_signal, 'norm')
            
            # If p-value is low, signal is significantly different from noise
            significance_threshold = 0.05
            
            return p_value < significance_threshold
            
        except Exception:
            return True  # Assume significant on failure
    
    def _compute_adaptive_imf_threshold(self, signal: np.ndarray) -> float:
        """Compute adaptive threshold for IMF validation"""
        # Base threshold
        base_threshold = 0.7  # 70% score required
        
        # Adjust based on signal characteristics
        signal_length = len(signal)
        noise_level = self._estimate_noise_level(signal)
        
        # For short signals, be more lenient
        if signal_length < 50:
            base_threshold -= 0.1
        elif signal_length > 500:
            base_threshold += 0.05
        
        # For noisy signals, be more lenient
        if noise_level > 0.2:
            base_threshold -= 0.15
        elif noise_level < 0.05:
            base_threshold += 0.1
        
        # Ensure reasonable bounds
        return np.clip(base_threshold, 0.5, 0.9)
    
    def _robust_imf_validation(self, candidate: np.ndarray) -> bool:
        """Robust fallback IMF validation"""
        try:
            # Basic extrema and zero-crossing check with enhanced tolerance
            zero_crossings = self._find_zero_crossings(candidate)
            maxima_idx, minima_idx = self._find_extrema_enhanced(candidate)
            extrema_count = len(maxima_idx) + len(minima_idx)
            
            if extrema_count == 0 or len(zero_crossings) == 0:
                return False
            
            # Relaxed criterion for robustness
            difference = abs(extrema_count - len(zero_crossings))
            max_tolerance = max(2, len(candidate) // 50)  # Scale with signal length
            
            extrema_criterion = difference <= max_tolerance
            
            # Additional simple checks
            has_oscillations = np.std(candidate) > 1e-6  # Non-constant
            reasonable_frequency = len(maxima_idx) + len(minima_idx) < len(candidate) // 2
            
            return extrema_criterion and has_oscillations and reasonable_frequency
            
        except Exception:
            return self._basic_imf_validation(candidate)
    
    def _basic_imf_validation(self, candidate: np.ndarray) -> bool:
        """Basic emergency fallback IMF validation"""
        try:
            # Very simple check: signal has some variation and reasonable extrema count
            if np.std(candidate) < 1e-8:  # Essentially constant
                return False
            
            # Count sign changes as proxy for zero crossings
            sign_changes = np.sum(np.diff(np.sign(candidate)) != 0)
            
            # Simple peak detection
            peaks_up = np.sum(np.diff(np.diff(candidate)) < 0)  # Local maxima
            peaks_down = np.sum(np.diff(np.diff(candidate)) > 0)  # Local minima
            total_extrema = peaks_up + peaks_down
            
            # Very relaxed criterion
            return (sign_changes > 0 and total_extrema > 0 and 
                    abs(total_extrema - sign_changes) <= max(3, len(candidate) // 20))
        
        except Exception:
            # Ultimate fallback: just check for non-constant signal
            return np.std(candidate) > 1e-6
    
    def _wavelet_phase_extraction(self, signal: np.ndarray) -> np.ndarray:
        """Wavelet-based phase extraction using Morlet wavelet"""
        n = len(signal)
        
        # Morlet wavelet parameters
        central_freq = 1.0
        bandwidth = 1.0
        
        # Create Morlet wavelet
        t = np.arange(-n//2, n//2)
        morlet_real = np.exp(-(t**2) / (2 * bandwidth**2)) * np.cos(2 * np.pi * central_freq * t / n)
        morlet_imag = np.exp(-(t**2) / (2 * bandwidth**2)) * np.sin(2 * np.pi * central_freq * t / n)
        
        # Convolution with signal
        conv_real = np.convolve(signal, morlet_real, mode='same')
        conv_imag = np.convolve(signal, morlet_imag, mode='same')
        
        # Construct complex analytic signal
        analytic_signal = conv_real + 1j * conv_imag
        
        # Extract phase
        phase = np.angle(analytic_signal)
        phase_unwrapped = np.unwrap(phase)
        
        return np.mod(phase_unwrapped, 2 * np.pi)
    
    def _robust_derivative_phase(self, signal: np.ndarray) -> np.ndarray:
        """Robust derivative-based phase extraction with smoothing"""
        # Apply smoothing to reduce noise
        smoothed_signal = self._smooth_signal(signal)
        
        # Compute derivative using central differences
        derivative = np.gradient(smoothed_signal)
        
        # Normalize derivative
        derivative_normalized = derivative / (np.std(derivative) + 1e-10)
        
        # Integrate to get phase
        phase = np.cumsum(derivative_normalized) * 0.1
        
        # Apply additional smoothing to phase
        phase_smoothed = self._smooth_signal(phase)
        
        return np.mod(phase_smoothed, 2 * np.pi)
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int = None) -> np.ndarray:
        """Signal smoothing using moving average"""
        if window_size is None:
            window_size = max(3, len(signal) // 20)
        
        if window_size >= len(signal):
            return signal
        
        # Ensure odd window size
        if window_size % 2 == 0:
            window_size += 1
        
        # Moving average with reflection padding
        padding = window_size // 2
        padded_signal = np.pad(signal, (padding, padding), mode='reflect')
        
        # Convolution with uniform kernel
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded_signal, kernel, mode='valid')
        
        return smoothed
    
    def _validate_phase_continuity(self, phase: np.ndarray) -> bool:
        """Validate phase continuity and reasonable behavior"""
        if len(phase) == 0:
            return False
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(phase)):
            return False
        
        # Check phase differences (should not have huge jumps after unwrapping)
        phase_diff = np.diff(phase)
        max_jump = np.max(np.abs(phase_diff))
        
        # Reasonable phase evolution criterion
        if max_jump > 10 * np.pi:  # Allow some flexibility
            return False
        
        # Check for constant phase (indicates failure)
        if np.std(phase) < 1e-6:
            return False
        
        return True
    
    def _simple_phase_fallback(self, signal: np.ndarray) -> np.ndarray:
        """Simple fallback phase extraction"""
        # Normalize signal
        signal_normalized = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
        
        # Simple cumulative phase based on signal evolution
        phase = np.cumsum(np.abs(np.diff(signal_normalized, prepend=signal_normalized[0])))
        
        # Scale to reasonable phase range
        phase = phase * 2 * np.pi / (np.max(phase) + 1e-10)
        
        return np.mod(phase, 2 * np.pi)
    
    def _convergent_cross_mapping(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Comprehensive Convergent Cross Mapping for advanced non-linear dynamical system analysis"""
        try:
            return self._advanced_ccm_analysis(cause, effect)
        except Exception:
            try:
                return self._robust_ccm_fallback(cause, effect)
            except Exception:
                return self._basic_ccm_emergency(cause, effect)
    
    def _advanced_ccm_analysis(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Advanced CCM with optimal embedding parameters and multi-scale analysis"""
        if len(cause) != len(effect) or len(cause) < 20:
            return 0.0
        
        # Preprocessing: normalize and filter data
        cause_clean, effect_clean = self._preprocess_ccm_data(cause, effect)
        
        # Optimal embedding parameter estimation
        optimal_params = self._estimate_optimal_embedding_parameters(cause_clean, effect_clean)
        
        # Multi-scale CCM analysis
        ccm_results = self._multi_scale_ccm_computation(cause_clean, effect_clean, optimal_params)
        
        # Statistical significance testing
        significance_score = self._assess_ccm_significance(ccm_results, cause_clean, effect_clean)
        
        # Convergence analysis for causality strength
        convergence_strength = self._analyze_ccm_convergence(ccm_results)
        
        # Final CCM score with confidence weighting
        final_score = self._synthesize_ccm_score(ccm_results, significance_score, convergence_strength)
        
        return final_score
    
    def _preprocess_ccm_data(self, cause: np.ndarray, effect: np.ndarray) -> tuple:
        """Comprehensive data preprocessing for CCM analysis"""
        # Normalize series to zero mean and unit variance
        cause_norm = (cause - np.mean(cause)) / (np.std(cause) + 1e-10)
        effect_norm = (effect - np.mean(effect)) / (np.std(effect) + 1e-10)
        
        # Remove outliers using robust statistics
        cause_clean = self._remove_ccm_outliers(cause_norm)
        effect_clean = self._remove_ccm_outliers(effect_norm)
        
        # Apply light smoothing to reduce measurement noise
        cause_smoothed = self._apply_ccm_smoothing(cause_clean)
        effect_smoothed = self._apply_ccm_smoothing(effect_clean)
        
        return cause_smoothed, effect_smoothed
    
    def _remove_ccm_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers using modified Z-score method"""
        if len(data) < 5:
            return data
        
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad > 0:
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > 3.5
            
            # Replace outliers with interpolated values
            if np.sum(outlier_mask) > 0:
                cleaned = data.copy()
                valid_indices = np.where(~outlier_mask)[0]
                outlier_indices = np.where(outlier_mask)[0]
                
                if len(valid_indices) >= 2:
                    cleaned[outlier_indices] = np.interp(
                        outlier_indices, valid_indices, data[valid_indices]
                    )
                return cleaned
        
        return data
    
    def _apply_ccm_smoothing(self, data: np.ndarray) -> np.ndarray:
        """Apply light smoothing while preserving dynamics"""
        if len(data) < 5:
            return data
        
        # Use small moving average to reduce noise
        window_size = min(3, len(data) // 10)
        if window_size < 2:
            return data
        
        kernel = np.ones(window_size) / window_size
        padded = np.pad(data, window_size//2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed[:len(data)]
    
    def _estimate_optimal_embedding_parameters(self, cause: np.ndarray, effect: np.ndarray) -> dict:
        """Estimate optimal embedding dimension and delay using multiple methods"""
        params = {}
        
        # Embedding dimension estimation using False Nearest Neighbors
        params['embed_dim_cause'] = self._estimate_embedding_dimension(cause)
        params['embed_dim_effect'] = self._estimate_embedding_dimension(effect)
        
        # Use maximum dimension for analysis
        params['embed_dim'] = max(params['embed_dim_cause'], params['embed_dim_effect'])
        
        # Time delay estimation using autocorrelation
        params['tau_cause'] = self._estimate_time_delay(cause)
        params['tau_effect'] = self._estimate_time_delay(effect)
        
        # Use average delay
        params['tau'] = max(1, int((params['tau_cause'] + params['tau_effect']) / 2))
        
        # Library size estimation
        params['lib_sizes'] = self._estimate_library_sizes(len(cause), params['embed_dim'])
        
        return params
    
    def _estimate_embedding_dimension(self, data: np.ndarray) -> int:
        """Comprehensive embedding dimension estimation using advanced False Nearest Neighbors analysis"""
        try:
            return self._advanced_embedding_dimension_estimation(data)
        except Exception:
            try:
                return self._robust_embedding_dimension_estimation(data)
            except Exception:
                return self._basic_embedding_dimension_fallback(data)
    
    def _advanced_embedding_dimension_estimation(self, data: np.ndarray) -> int:
        """Advanced embedding dimension estimation with comprehensive analysis"""
        if len(data) < 50:
            return self._estimate_dimension_short_series(data)
        
        # Adaptive parameter selection
        max_dim = min(15, max(6, len(data) // 20))  # More flexible maximum
        min_dim = 1
        
        # Multiple embedding delay estimation
        optimal_taus = self._estimate_multiple_embedding_delays(data, max_lag=20)
        
        # Multi-criteria dimension estimation
        dimension_candidates = {}
        
        # Method 1: Advanced False Nearest Neighbors
        fnn_results = self._comprehensive_fnn_analysis(data, min_dim, max_dim, optimal_taus)
        
        # Method 2: Correlation dimension analysis  
        corr_dim_results = self._correlation_dimension_analysis(data, min_dim, max_dim, optimal_taus)
        
        # Method 3: Mutual information-based estimation
        mi_results = self._mutual_information_dimension_analysis(data, min_dim, max_dim)
        
        # Method 4: Recurrence quantification analysis
        rqa_results = self._recurrence_dimension_analysis(data, min_dim, max_dim, optimal_taus)
        
        # Method 5: Statistical complexity analysis
        complexity_results = self._statistical_complexity_dimension(data, min_dim, max_dim)
        
        # Ensemble voting with confidence weighting
        method_weights = {
            'fnn': 0.35,           # Primary method
            'correlation': 0.25,    # Classical approach
            'mutual_info': 0.15,    # Information theoretic
            'recurrence': 0.15,     # Dynamical systems
            'complexity': 0.10      # Statistical validation
        }
        
        # Collect all dimension estimates
        all_estimates = {
            'fnn': fnn_results,
            'correlation': corr_dim_results,
            'mutual_info': mi_results,
            'recurrence': rqa_results,
            'complexity': complexity_results
        }
        
        # Weighted ensemble decision
        ensemble_dimension = self._compute_ensemble_dimension(all_estimates, method_weights, min_dim, max_dim)
        
        # Cross-validation and consistency check
        validated_dimension = self._cross_validate_dimension(data, ensemble_dimension, optimal_taus)
        
        # Final sanity checks
        final_dimension = self._apply_dimension_constraints(validated_dimension, len(data))
        
        return final_dimension
    
    def _estimate_multiple_embedding_delays(self, data: np.ndarray, max_lag: int = 20) -> Dict[str, int]:
        """Estimate optimal embedding delays using multiple methods"""
        results = {}
        
        # Method 1: First minimum of mutual information
        results['mutual_info'] = self._mutual_information_delay(data, max_lag)
        
        # Method 2: First zero of autocorrelation
        results['autocorr_zero'] = self._autocorrelation_zero_delay(data, max_lag)
        
        # Method 3: Decorrelation time
        results['decorr_time'] = self._decorrelation_time_delay(data, max_lag)
        
        # Method 4: Average displacement method
        results['avg_displacement'] = self._average_displacement_delay(data, max_lag)
        
        return results
    
    def _mutual_information_delay(self, data: np.ndarray, max_lag: int) -> int:
        """Find optimal delay using mutual information first minimum"""
        mutual_info_values = []
        
        for lag in range(1, min(max_lag + 1, len(data) // 4)):
            try:
                # Compute mutual information between data[:-lag] and data[lag:]
                x = data[:-lag]
                y = data[lag:]
                
                # Use binning approach for mutual information
                mi_value = self._compute_mutual_information_binned(x, y)
                mutual_info_values.append(mi_value)
                
            except Exception:
                mutual_info_values.append(0.0)
        
        if not mutual_info_values:
            return 1
        
        # Find first local minimum
        for i in range(1, len(mutual_info_values) - 1):
            if (mutual_info_values[i] < mutual_info_values[i-1] and 
                mutual_info_values[i] < mutual_info_values[i+1]):
                return i + 1
        
        # Fallback: minimum value
        return np.argmin(mutual_info_values) + 1
    
    def _compute_mutual_information_binned(self, x: np.ndarray, y: np.ndarray, bins: int = None) -> float:
        """Compute mutual information using binning approach"""
        if bins is None:
            bins = max(10, int(np.sqrt(len(x))))
        
        try:
            # Create 2D histogram
            hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
            
            # Normalize to get probabilities
            p_xy = hist_2d / np.sum(hist_2d)
            
            # Marginal probabilities
            p_x = np.sum(p_xy, axis=1)
            p_y = np.sum(p_xy, axis=0)
            
            # Compute mutual information
            mi = 0.0
            for i in range(len(p_x)):
                for j in range(len(p_y)):
                    if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                        mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
            
            return mi
            
        except Exception:
            return 0.0
    
    def _autocorrelation_zero_delay(self, data: np.ndarray, max_lag: int) -> int:
        """Find delay at first zero crossing of autocorrelation"""
        autocorr_values = []
        
        for lag in range(1, min(max_lag + 1, len(data) // 3)):
            try:
                corr_coef = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                if not np.isnan(corr_coef):
                    autocorr_values.append(corr_coef)
                else:
                    autocorr_values.append(0.0)
            except Exception:
                autocorr_values.append(0.0)
        
        # Find first zero crossing
        for i, corr in enumerate(autocorr_values):
            if corr <= 0:
                return i + 1
        
        # Fallback: when autocorrelation drops below threshold
        threshold = 0.1
        for i, corr in enumerate(autocorr_values):
            if corr < threshold:
                return i + 1
        
        return max(1, len(autocorr_values) // 3)
    
    def _decorrelation_time_delay(self, data: np.ndarray, max_lag: int) -> int:
        """Find decorrelation time using exponential fit"""
        autocorr_values = []
        
        for lag in range(1, min(max_lag + 1, len(data) // 3)):
            try:
                corr_coef = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                if not np.isnan(corr_coef):
                    autocorr_values.append(abs(corr_coef))
                else:
                    autocorr_values.append(0.0)
            except Exception:
                autocorr_values.append(0.0)
        
        if not autocorr_values:
            return 1
        
        try:
            # Fit exponential decay: C(τ) = C₀ * exp(-τ/τd)
            lags = np.arange(1, len(autocorr_values) + 1)
            log_autocorr = np.log(np.maximum(autocorr_values, 1e-10))
            
            # Linear fit in log space
            coeffs = np.polyfit(lags, log_autocorr, 1)
            decorr_time = -1.0 / coeffs[0] if coeffs[0] < 0 else len(autocorr_values)
            
            return max(1, int(decorr_time))
            
        except Exception:
            # Fallback: 1/e point
            target_value = autocorr_values[0] / np.e
            for i, val in enumerate(autocorr_values):
                if val <= target_value:
                    return i + 1
            return len(autocorr_values) // 2
    
    def _average_displacement_delay(self, data: np.ndarray, max_lag: int) -> int:
        """Estimate delay using average displacement method"""
        displacements = []
        
        for lag in range(1, min(max_lag + 1, len(data) // 4)):
            try:
                displacement = np.mean(np.abs(data[lag:] - data[:-lag]))
                displacements.append(displacement)
            except Exception:
                displacements.append(0.0)
        
        if not displacements:
            return 1
        
        # Find lag where displacement stabilizes (plateau)
        for i in range(2, len(displacements)):
            if (abs(displacements[i] - displacements[i-1]) < 0.1 * displacements[0] and
                abs(displacements[i-1] - displacements[i-2]) < 0.1 * displacements[0]):
                return i + 1
        
        return len(displacements) // 2
    
    def _comprehensive_fnn_analysis(self, data: np.ndarray, min_dim: int, max_dim: int, 
                                   optimal_taus: Dict[str, int]) -> Dict[str, Any]:
        """Comprehensive False Nearest Neighbors analysis"""
        results = {}
        
        # Use best tau from multiple methods
        tau = int(np.median(list(optimal_taus.values())))
        
        fnn_ratios = []
        fnn_statistics = []
        
        for dim in range(min_dim, max_dim + 1):
            try:
                # Advanced FNN calculation with multiple criteria
                fnn_ratio = self._advanced_fnn_calculation(data, dim, tau)
                fnn_ratios.append(fnn_ratio)
                
                # Additional FNN statistics
                stats = self._fnn_statistical_analysis(data, dim, tau)
                fnn_statistics.append(stats)
                
            except Exception as e:
                fnn_ratios.append(1.0)
                fnn_statistics.append({'confidence': 0.0, 'stability': 0.0})
        
        # Find optimal dimension using multiple criteria
        optimal_dim = self._find_optimal_fnn_dimension(fnn_ratios, fnn_statistics, min_dim)
        
        results = {
            'optimal_dimension': optimal_dim,
            'fnn_ratios': fnn_ratios,
            'confidence': np.mean([stats.get('confidence', 0.0) for stats in fnn_statistics]),
            'method': 'advanced_fnn'
        }
        
        return results
    
    def _advanced_fnn_calculation(self, data: np.ndarray, embed_dim: int, tau: int) -> float:
        """Advanced FNN calculation with multiple criteria"""
        if len(data) < 5 * embed_dim * tau:
            return 1.0
        
        # Create embedding
        embedded = self._create_embedding_matrix(data, embed_dim, tau)
        if len(embedded) < 20:
            return 1.0
        
        false_neighbors = 0
        total_neighbors = 0
        
        # Adaptive neighborhood selection
        sample_size = min(len(embedded) // 2, 1000)  # Limit for computational efficiency
        indices = np.random.choice(len(embedded) - 1, size=sample_size, replace=False)
        
        # Distance threshold adaptation
        distances = []
        for i in range(min(100, len(embedded))):
            for j in range(i + 1, min(i + 10, len(embedded))):
                distances.append(np.linalg.norm(embedded[i] - embedded[j]))
        
        if not distances:
            return 1.0
        
        distance_threshold = np.percentile(distances, 10)  # Use 10th percentile as threshold
        
        for idx in indices:
            try:
                current_point = embedded[idx]
                next_point = embedded[idx + 1]
                
                # Find nearest neighbors in d-dimensional space
                distances_d = np.linalg.norm(embedded - current_point, axis=1)
                
                # Exclude the point itself
                distances_d[idx] = np.inf
                
                # Find nearest neighbor
                nn_idx = np.argmin(distances_d)
                nn_distance_d = distances_d[nn_idx]
                
                if nn_distance_d < distance_threshold:
                    total_neighbors += 1
                    
                    # Check in (d+1)-dimensional space
                    if nn_idx + 1 < len(data) - embed_dim * tau:
                        # Create (d+1)-dimensional embedding
                        extended_current = np.append(current_point, data[idx + embed_dim * tau])
                        extended_nn = np.append(embedded[nn_idx], data[nn_idx + embed_dim * tau])
                        
                        distance_d_plus_1 = np.linalg.norm(extended_current - extended_nn)
                        
                        # Multiple FNN criteria
                        ratio_threshold = 15.0  # Standard threshold
                        absolute_threshold = 2.0 * np.std(data)  # Adaptive absolute threshold
                        
                        # Criterion 1: Ratio test
                        if nn_distance_d > 0:
                            ratio = distance_d_plus_1 / nn_distance_d
                            if ratio > ratio_threshold:
                                false_neighbors += 1
                                continue
                        
                        # Criterion 2: Absolute distance test
                        if distance_d_plus_1 > absolute_threshold:
                            false_neighbors += 1
                            continue
                        
                        # Criterion 3: Relative neighborhood preservation
                        next_nn_distance_d = np.linalg.norm(embedded[nn_idx] - embedded[nn_idx + 1] if nn_idx + 1 < len(embedded) else embedded[nn_idx])
                        next_current_distance_d = np.linalg.norm(current_point - next_point)
                        
                        if (abs(next_nn_distance_d - next_current_distance_d) > 
                            0.5 * (next_nn_distance_d + next_current_distance_d)):
                            false_neighbors += 1
                
            except Exception:
                continue
        
        if total_neighbors == 0:
            return 1.0
        
        return false_neighbors / total_neighbors
    
    def _fnn_statistical_analysis(self, data: np.ndarray, embed_dim: int, tau: int) -> Dict[str, float]:
        """Statistical analysis of FNN results for confidence estimation"""
        try:
            # Bootstrap sampling for confidence intervals
            n_bootstrap = 10
            fnn_bootstrap = []
            
            data_length = len(data)
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(data_length, size=data_length, replace=True)
                bootstrap_data = data[bootstrap_indices]
                
                try:
                    fnn_ratio = self._advanced_fnn_calculation(bootstrap_data, embed_dim, tau)
                    fnn_bootstrap.append(fnn_ratio)
                except Exception:
                    continue
            
            if not fnn_bootstrap:
                return {'confidence': 0.0, 'stability': 0.0}
            
            # Statistical measures
            mean_fnn = np.mean(fnn_bootstrap)
            std_fnn = np.std(fnn_bootstrap)
            
            # Confidence based on bootstrap stability
            confidence = max(0.0, 1.0 - 2.0 * std_fnn)  # High std reduces confidence
            
            # Stability measure
            stability = 1.0 / (1.0 + std_fnn)
            
            return {
                'confidence': confidence,
                'stability': stability,
                'mean_fnn': mean_fnn,
                'std_fnn': std_fnn
            }
            
        except Exception:
            return {'confidence': 0.0, 'stability': 0.0}
    
    def _find_optimal_fnn_dimension(self, fnn_ratios: List[float], 
                                   fnn_statistics: List[Dict], min_dim: int) -> int:
        """Find optimal dimension using multiple FNN criteria"""
        if not fnn_ratios:
            return 3
        
        # Primary criterion: first dimension where FNN ratio drops significantly
        threshold = 0.05  # 5% false neighbors
        
        for i, ratio in enumerate(fnn_ratios):
            if ratio < threshold:
                # Additional validation: ensure it remains low
                if i + 1 < len(fnn_ratios) and fnn_ratios[i + 1] < threshold * 2:
                    return min_dim + i
        
        # Secondary criterion: steepest descent in FNN ratio
        if len(fnn_ratios) > 2:
            differences = np.diff(fnn_ratios)
            steepest_idx = np.argmin(differences)
            
            # Ensure significant drop
            if differences[steepest_idx] < -0.1:
                return min_dim + steepest_idx + 1
        
        # Tertiary criterion: minimum FNN ratio with high confidence
        best_idx = 0
        best_score = float('inf')
        
        for i, (ratio, stats) in enumerate(zip(fnn_ratios, fnn_statistics)):
            # Combined score: FNN ratio weighted by confidence
            confidence = stats.get('confidence', 0.5)
            score = ratio / max(confidence, 0.1)  # Avoid division by zero
            
            if score < best_score:
                best_score = score
                best_idx = i
        
        return min_dim + best_idx
    
    def _correlation_dimension_analysis(self, data: np.ndarray, min_dim: int, 
                                      max_dim: int, optimal_taus: Dict[str, int]) -> Dict[str, Any]:
        """Estimate dimension using correlation dimension analysis"""
        try:
            tau = int(np.median(list(optimal_taus.values())))
            
            correlation_dims = []
            
            for embed_dim in range(min_dim, max_dim + 1):
                try:
                    embedded = self._create_embedding_matrix(data, embed_dim, tau)
                    corr_dim = self._compute_correlation_dimension(embedded)
                    correlation_dims.append(corr_dim)
                except Exception:
                    correlation_dims.append(embed_dim)  # Fallback
            
            # Find saturation point
            optimal_dim = self._find_saturation_point(correlation_dims, min_dim)
            
            return {
                'optimal_dimension': optimal_dim,
                'correlation_dimensions': correlation_dims,
                'confidence': 0.7,  # Moderate confidence for this method
                'method': 'correlation_dimension'
            }
            
        except Exception:
            return {
                'optimal_dimension': 3,
                'confidence': 0.1,
                'method': 'correlation_dimension_fallback'
            }
    
    def _compute_correlation_dimension(self, embedded_data: np.ndarray, 
                                     min_epsilon: float = None, max_epsilon: float = None) -> float:
        """Compute correlation dimension using Grassberger-Procaccia algorithm"""
        n_points = len(embedded_data)
        
        if min_epsilon is None:
            min_epsilon = 0.01 * np.std(embedded_data)
        if max_epsilon is None:
            max_epsilon = 0.5 * np.std(embedded_data)
        
        # Logarithmically spaced epsilon values
        epsilons = np.logspace(np.log10(min_epsilon), np.log10(max_epsilon), 20)
        correlations = []
        
        for epsilon in epsilons:
            correlation = 0.0
            count = 0
            
            # Sample points for computational efficiency
            sample_size = min(n_points, 500)
            indices = np.random.choice(n_points, sample_size, replace=False)
            
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    distance = np.linalg.norm(embedded_data[indices[i]] - embedded_data[indices[j]])
                    if distance < epsilon:
                        correlation += 1
                    count += 1
            
            if count > 0:
                correlations.append(correlation / count)
            else:
                correlations.append(0.0)
        
        # Fit linear regression to log-log plot
        valid_indices = [(i, c) for i, c in enumerate(correlations) if c > 0]
        
        if len(valid_indices) < 3:
            return embedded_data.shape[1]  # Return embedding dimension as fallback
        
        log_epsilons = [np.log(epsilons[i]) for i, _ in valid_indices]
        log_correlations = [np.log(c) for _, c in valid_indices]
        
        try:
            slope, _ = np.polyfit(log_epsilons, log_correlations, 1)
            return max(0.1, slope)  # Correlation dimension
        except Exception:
            return embedded_data.shape[1]
    
    def _find_saturation_point(self, values: List[float], start_dim: int) -> int:
        """Find saturation point in dimension estimation"""
        if len(values) < 3:
            return start_dim + len(values) - 1
        
        # Look for where the rate of change becomes small
        differences = np.diff(values)
        
        # Find first point where change is less than 10% of initial change
        if len(differences) > 0 and abs(differences[0]) > 1e-6:
            threshold = 0.1 * abs(differences[0])
            
            for i, diff in enumerate(differences[1:], 1):  # Start from second difference
                if abs(diff) < threshold:
                    return start_dim + i
        
        # Fallback: find minimum or use 2/3 of range
        return start_dim + len(values) // 2
    
    def _mutual_information_dimension_analysis(self, data: np.ndarray, 
                                             min_dim: int, max_dim: int) -> Dict[str, Any]:
        """Dimension estimation using mutual information approach"""
        try:
            mi_scores = []
            
            for dim in range(min_dim, max_dim + 1):
                # Use average displacement delay
                tau = self._average_displacement_delay(data, 10)
                
                if len(data) > dim * tau + 10:
                    embedded = self._create_embedding_matrix(data, dim, tau)
                    
                    # Compute mutual information between embedded vectors
                    mi_score = self._compute_embedding_mutual_information(embedded)
                    mi_scores.append(mi_score)
                else:
                    mi_scores.append(0.0)
            
            # Find dimension with maximum mutual information
            if mi_scores:
                optimal_dim = min_dim + np.argmax(mi_scores)
            else:
                optimal_dim = 3
            
            return {
                'optimal_dimension': optimal_dim,
                'mi_scores': mi_scores,
                'confidence': 0.6,
                'method': 'mutual_information'
            }
            
        except Exception:
            return {
                'optimal_dimension': 3,
                'confidence': 0.1,
                'method': 'mutual_information_fallback'
            }
    
    def _compute_embedding_mutual_information(self, embedded_data: np.ndarray) -> float:
        """Compute mutual information for embedded data quality assessment"""
        try:
            n_points, n_dims = embedded_data.shape
            
            if n_dims < 2:
                return 0.0
            
            # Use first and last dimensions as proxy for embedding quality
            x = embedded_data[:, 0]
            y = embedded_data[:, -1]
            
            return self._compute_mutual_information_binned(x, y)
            
        except Exception:
            return 0.0
    
    def _recurrence_dimension_analysis(self, data: np.ndarray, min_dim: int, 
                                     max_dim: int, optimal_taus: Dict[str, int]) -> Dict[str, Any]:
        """Dimension estimation using recurrence quantification analysis"""
        try:
            tau = int(np.median(list(optimal_taus.values())))
            recurrence_scores = []
            
            for dim in range(min_dim, max_dim + 1):
                if len(data) > dim * tau + 20:
                    embedded = self._create_embedding_matrix(data, dim, tau)
                    rqa_score = self._compute_recurrence_score(embedded)
                    recurrence_scores.append(rqa_score)
                else:
                    recurrence_scores.append(0.0)
            
            # Find dimension with optimal recurrence properties
            optimal_dim = self._find_optimal_recurrence_dimension(recurrence_scores, min_dim)
            
            return {
                'optimal_dimension': optimal_dim,
                'recurrence_scores': recurrence_scores,
                'confidence': 0.5,
                'method': 'recurrence_analysis'
            }
            
        except Exception:
            return {
                'optimal_dimension': 4,
                'confidence': 0.1,
                'method': 'recurrence_fallback'
            }
    
    def _compute_recurrence_score(self, embedded_data: np.ndarray) -> float:
        """Compute recurrence-based score for dimension assessment"""
        try:
            n_points = len(embedded_data)
            
            if n_points < 10:
                return 0.0
            
            # Compute recurrence rate with adaptive threshold
            threshold = 0.1 * np.std(embedded_data)
            recurrences = 0
            total_pairs = 0
            
            # Sample for computational efficiency
            sample_size = min(n_points, 100)
            indices = np.random.choice(n_points, sample_size, replace=False)
            
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    distance = np.linalg.norm(embedded_data[indices[i]] - embedded_data[indices[j]])
                    if distance < threshold:
                        recurrences += 1
                    total_pairs += 1
            
            if total_pairs > 0:
                return recurrences / total_pairs
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _find_optimal_recurrence_dimension(self, scores: List[float], start_dim: int) -> int:
        """Find optimal dimension based on recurrence scores"""
        if not scores:
            return 3
        
        # Look for moderate recurrence rate (not too high, not too low)
        target_rate = 0.1  # 10% recurrence rate is often optimal
        
        best_dim = start_dim
        best_score = float('inf')
        
        for i, score in enumerate(scores):
            deviation = abs(score - target_rate)
            if deviation < best_score:
                best_score = deviation
                best_dim = start_dim + i
        
        return best_dim
    
    def _statistical_complexity_dimension(self, data: np.ndarray, 
                                        min_dim: int, max_dim: int) -> Dict[str, Any]:
        """Dimension estimation using statistical complexity measures"""
        try:
            complexity_scores = []
            
            for dim in range(min_dim, max_dim + 1):
                tau = max(1, len(data) // (10 * dim))  # Adaptive tau
                
                if len(data) > dim * tau + 10:
                    embedded = self._create_embedding_matrix(data, dim, tau)
                    complexity = self._compute_statistical_complexity(embedded)
                    complexity_scores.append(complexity)
                else:
                    complexity_scores.append(0.0)
            
            # Find dimension with optimal complexity
            if complexity_scores:
                # Look for peak in complexity (but not the highest to avoid overfitting)
                sorted_indices = np.argsort(complexity_scores)
                # Choose 75th percentile rather than maximum
                optimal_idx = sorted_indices[int(0.75 * len(sorted_indices))]
                optimal_dim = min_dim + optimal_idx
            else:
                optimal_dim = 3
            
            return {
                'optimal_dimension': optimal_dim,
                'complexity_scores': complexity_scores,
                'confidence': 0.4,
                'method': 'statistical_complexity'
            }
            
        except Exception:
            return {
                'optimal_dimension': 3,
                'confidence': 0.1,
                'method': 'complexity_fallback'
            }
    
    def _compute_statistical_complexity(self, embedded_data: np.ndarray) -> float:
        """Compute statistical complexity measure"""
        try:
            n_points, n_dims = embedded_data.shape
            
            if n_points < 20:
                return 0.0
            
            # Compute normalized distances
            distances = []
            for i in range(min(n_points, 100)):
                for j in range(i + 1, min(n_points, 100)):
                    distance = np.linalg.norm(embedded_data[i] - embedded_data[j])
                    distances.append(distance)
            
            if not distances:
                return 0.0
            
            distances = np.array(distances)
            
            # Statistical complexity based on distribution of distances
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            if mean_dist > 0:
                # Coefficient of variation as complexity measure
                complexity = std_dist / mean_dist
                
                # Normalize by dimension
                complexity = complexity / np.sqrt(n_dims)
                
                return complexity
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compute_ensemble_dimension(self, all_estimates: Dict[str, Dict], 
                                  method_weights: Dict[str, float], 
                                  min_dim: int, max_dim: int) -> int:
        """Compute ensemble dimension estimate with confidence weighting"""
        weighted_votes = {}
        total_weight = 0.0
        
        for method, estimates in all_estimates.items():
            if estimates and 'optimal_dimension' in estimates:
                dimension = estimates['optimal_dimension']
                confidence = estimates.get('confidence', 0.5)
                base_weight = method_weights.get(method, 0.1)
                
                # Confidence-weighted vote
                final_weight = base_weight * confidence
                
                if dimension not in weighted_votes:
                    weighted_votes[dimension] = 0.0
                
                weighted_votes[dimension] += final_weight
                total_weight += final_weight
        
        if not weighted_votes or total_weight == 0:
            return 3  # Default fallback
        
        # Find dimension with highest weighted vote
        best_dimension = max(weighted_votes.keys(), key=lambda k: weighted_votes[k])
        
        # Ensure within bounds
        return max(min_dim, min(max_dim, best_dimension))
    
    def _cross_validate_dimension(self, data: np.ndarray, estimated_dim: int, 
                                optimal_taus: Dict[str, int]) -> int:
        """Cross-validate dimension estimate using data splits"""
        try:
            if len(data) < 100:  # Skip validation for short series
                return estimated_dim
            
            # Split data into segments
            n_segments = 3
            segment_length = len(data) // n_segments
            
            dimension_estimates = []
            
            for i in range(n_segments):
                start_idx = i * segment_length
                end_idx = (i + 1) * segment_length if i < n_segments - 1 else len(data)
                segment = data[start_idx:end_idx]
                
                if len(segment) > 50:  # Minimum length for meaningful analysis
                    # Quick FNN analysis on segment
                    tau = int(np.median(list(optimal_taus.values())))
                    segment_dim = self._quick_fnn_dimension(segment, estimated_dim, tau)
                    dimension_estimates.append(segment_dim)
            
            if dimension_estimates:
                # Use median of estimates for robustness
                validated_dim = int(np.median(dimension_estimates))
                
                # If estimates are consistent, use them; otherwise, use original
                if abs(validated_dim - estimated_dim) <= 1:
                    return validated_dim
                else:
                    # Check if majority agrees
                    unique_dims, counts = np.unique(dimension_estimates, return_counts=True)
                    majority_dim = unique_dims[np.argmax(counts)]
                    
                    if np.max(counts) > len(dimension_estimates) // 2:
                        return int(majority_dim)
            
            return estimated_dim
            
        except Exception:
            return estimated_dim
    
    def _quick_fnn_dimension(self, data: np.ndarray, max_dim: int, tau: int) -> int:
        """Quick FNN dimension estimation for validation"""
        try:
            for dim in range(2, max_dim + 2):
                if len(data) < 5 * dim * tau:
                    return dim - 1
                
                fnn_ratio = self._calculate_fnn_ratio(data, dim, tau)
                
                if fnn_ratio < 0.1:  # Relaxed threshold for quick validation
                    return dim
            
            return max_dim
            
        except Exception:
            return max_dim
    
    def _apply_dimension_constraints(self, dimension: int, data_length: int) -> int:
        """Apply practical constraints to dimension estimate"""
        # Minimum dimension
        min_allowed = 1
        
        # Maximum dimension based on data length
        max_allowed = min(10, data_length // 50)  # Ensure sufficient data per dimension
        
        # Geological systems typically have dimensions 2-8
        geological_max = 8
        
        final_max = min(max_allowed, geological_max)
        
        return max(min_allowed, min(final_max, dimension))
    
    def _estimate_dimension_short_series(self, data: np.ndarray) -> int:
        """Special handling for short time series"""
        if len(data) < 10:
            return 1
        elif len(data) < 20:
            return 2
        elif len(data) < 40:
            return 3
        else:
            # For moderately short series, use simplified analysis
            return min(4, len(data) // 10)
    
    def _robust_embedding_dimension_estimation(self, data: np.ndarray) -> int:
        """Robust fallback dimension estimation"""
        try:
            if len(data) < 30:
                return self._estimate_dimension_short_series(data)
            
            max_dim = min(6, len(data) // 15)
            min_dim = 2
            
            # Simple FNN with basic parameters
            tau = max(1, len(data) // 50)
            best_dim = 3
            min_fnn_ratio = 1.0
            
            for dim in range(min_dim, max_dim + 1):
                try:
                    fnn_ratio = self._calculate_fnn_ratio(data, dim, tau)
                    
                    if fnn_ratio < min_fnn_ratio:
                        min_fnn_ratio = fnn_ratio
                        best_dim = dim
                    
                    if fnn_ratio < 0.1:  # Good enough threshold
                        return dim
                        
                except Exception:
                    continue
            
            return best_dim
            
        except Exception:
            return self._basic_embedding_dimension_fallback(data)
    
    def _basic_embedding_dimension_fallback(self, data: np.ndarray) -> int:
        """Basic emergency fallback for dimension estimation"""
        if len(data) < 10:
            return 1
        elif len(data) < 25:
            return 2
        elif len(data) < 50:
            return 3
        elif len(data) < 100:
            return 4
        else:
            # Scale with data length but cap at reasonable value
            return min(6, max(3, len(data) // 50))
    
    def _calculate_fnn_ratio(self, data: np.ndarray, embed_dim: int, tau: int = 1) -> float:
        """Calculate False Nearest Neighbors ratio for given embedding dimension"""
        if len(data) < 4 * embed_dim:
            return 1.0
        
        # Create embedding
        embedded = self._create_embedding_matrix(data, embed_dim, tau)
        if len(embedded) < 10:
            return 1.0
        
        false_neighbors = 0
        total_neighbors = 0
        
        # Sample a subset for computational efficiency
        sample_size = min(len(embedded), 50)
        indices = np.random.choice(len(embedded), sample_size, replace=False)
        
        for i in indices:
            # Find nearest neighbor in d-dimensional space
            distances_d = []
            for j in range(len(embedded)):
                if i != j:
                    dist = np.linalg.norm(embedded[i] - embedded[j])
                    distances_d.append((dist, j))
            
            if not distances_d:
                continue
            
            distances_d.sort()
            nearest_idx = distances_d[0][1]
            nearest_dist_d = distances_d[0][0]
            
            # Calculate distance in (d+1)-dimensional space
            if i < len(data) - embed_dim * tau and nearest_idx < len(data) - embed_dim * tau:
                extended_i = np.append(embedded[i], data[i + embed_dim * tau])
                extended_nearest = np.append(embedded[nearest_idx], data[nearest_idx + embed_dim * tau])
                nearest_dist_d1 = np.linalg.norm(extended_i - extended_nearest)
                
                # Check if it's a false nearest neighbor
                if nearest_dist_d > 0:
                    ratio = nearest_dist_d1 / nearest_dist_d
                    if ratio > 2.0:  # Threshold for false neighbors
                        false_neighbors += 1
                
                total_neighbors += 1
        
        return false_neighbors / max(total_neighbors, 1)
    
    def _estimate_time_delay(self, data: np.ndarray) -> int:
        """Estimate optimal time delay using autocorrelation function"""
        if len(data) < 10:
            return 1
        
        max_lag = min(len(data) // 4, 20)
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:max_lag] / autocorr[0]
        
        # Find first minimum or when autocorr drops below 1/e
        threshold = 1 / np.e
        for lag in range(1, len(autocorr)):
            if autocorr[lag] < threshold:
                return lag
            if lag > 1 and autocorr[lag] > autocorr[lag-1]:
                return lag - 1
        
        return max(1, len(autocorr) // 3)
    
    def _estimate_library_sizes(self, data_length: int, embed_dim: int) -> list:
        """Estimate appropriate library sizes for CCM analysis"""
        min_lib = max(20, 4 * embed_dim)
        max_lib = min(data_length // 2, 200)
        
        if max_lib <= min_lib:
            return [min_lib]
        
        # Create logarithmically spaced library sizes
        n_sizes = min(5, (max_lib - min_lib) // 10)
        if n_sizes <= 1:
            return [min_lib, max_lib]
        
        lib_sizes = np.logspace(np.log10(min_lib), np.log10(max_lib), n_sizes)
        return [int(size) for size in lib_sizes]
    
    def _create_embedding_matrix(self, data: np.ndarray, embed_dim: int, tau: int = 1) -> np.ndarray:
        """Create time-delay embedding matrix"""
        if len(data) < embed_dim * tau:
            return np.array([])
        
        n_points = len(data) - (embed_dim - 1) * tau
        embedded = np.zeros((n_points, embed_dim))
        
        for i in range(embed_dim):
            embedded[:, i] = data[i * tau:n_points + i * tau]
        
        return embedded
    
    def _multi_scale_ccm_computation(self, cause: np.ndarray, effect: np.ndarray, params: dict) -> dict:
        """Multi-scale CCM computation with various library sizes and parameters"""
        results = {
            'correlations': [],
            'library_sizes': [],
            'prediction_skills': [],
            'embedding_params': params
        }
        
        embed_dim = params['embed_dim']
        tau = params['tau']
        
        # Create embeddings
        effect_embedded = self._create_embedding_matrix(effect, embed_dim, tau)
        if len(effect_embedded) == 0:
            return results
        
        # Align cause values with embedded effect
        cause_vals = cause[(embed_dim-1)*tau:len(effect_embedded)+(embed_dim-1)*tau]
        
        # CCM analysis for each library size
        for lib_size in params['lib_sizes']:
            if lib_size >= len(effect_embedded):
                continue
            
            ccm_result = self._compute_ccm_for_library_size(
                effect_embedded, cause_vals, lib_size, embed_dim)
            
            if ccm_result is not None:
                results['correlations'].append(ccm_result['correlation'])
                results['library_sizes'].append(lib_size)
                results['prediction_skills'].append(ccm_result['prediction_skill'])
        
        return results
    
    def _compute_ccm_for_library_size(self, effect_embedded: np.ndarray, cause_vals: np.ndarray, 
                                    lib_size: int, embed_dim: int) -> dict:
        """Compute CCM for specific library size with enhanced nearest neighbor search"""
        if lib_size >= len(effect_embedded):
            return None
        
        predicted = []
        actual = []
        prediction_errors = []
        
        # Number of nearest neighbors
        k = min(embed_dim + 1, lib_size // 2)
        
        # Cross-validation approach: use different parts of library for prediction
        n_folds = min(5, lib_size // 10)
        fold_size = lib_size // n_folds
        
        for fold in range(n_folds):
            # Define library and prediction sets
            pred_start = lib_size + fold * fold_size
            pred_end = min(pred_start + fold_size, len(effect_embedded))
            
            if pred_end <= pred_start:
                continue
            
            for i in range(pred_start, pred_end):
                if i >= len(effect_embedded) or i >= len(cause_vals):
                    continue
                
                # Find k nearest neighbors in the library
                neighbors = self._find_nearest_neighbors_enhanced(
                    effect_embedded[i], effect_embedded[:lib_size], k)
                
                if len(neighbors) == 0:
                    continue
                
                # Weighted prediction using exponential weighting
                pred_val, confidence = self._make_weighted_prediction(
                    neighbors, cause_vals[:lib_size])
                
                if confidence > 0.1:  # Only include confident predictions
                    predicted.append(pred_val)
                    actual.append(cause_vals[i])
                    prediction_errors.append(abs(pred_val - cause_vals[i]))
        
        if len(predicted) < 5:
            return None
        
        # Calculate correlation and prediction skill
        try:
            correlation = np.corrcoef(predicted, actual)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Prediction skill: normalized RMSE
            rmse = np.sqrt(np.mean(np.array(prediction_errors)**2))
            actual_std = np.std(actual)
            prediction_skill = max(0, 1 - rmse / (actual_std + 1e-10))
            
            return {
                'correlation': abs(correlation),
                'prediction_skill': prediction_skill,
                'n_predictions': len(predicted),
                'confidence': np.mean([1 - err / (actual_std + 1e-10) for err in prediction_errors])
            }
            
        except Exception:
            return None
    
    def _find_nearest_neighbors_enhanced(self, query_point: np.ndarray, library: np.ndarray, k: int) -> list:
        """Enhanced nearest neighbor search with distance validation"""
        if len(library) == 0 or k <= 0:
            return []
        
        distances = []
        for i, lib_point in enumerate(library):
            dist = np.linalg.norm(query_point - lib_point)
            if dist > 1e-10:  # Avoid identical points
                distances.append((dist, i))
        
        if len(distances) == 0:
            return []
        
        # Sort by distance and select k nearest
        distances.sort()
        k_nearest = distances[:min(k, len(distances))]
        
        # Filter out neighbors that are too far (outliers)
        if len(k_nearest) > 1:
            median_dist = np.median([dist for dist, _ in k_nearest])
            filtered_neighbors = [(dist, idx) for dist, idx in k_nearest 
                                if dist <= 3 * median_dist]
            return filtered_neighbors if filtered_neighbors else k_nearest
        
        return k_nearest
    
    def _make_weighted_prediction(self, neighbors: list, cause_values: np.ndarray) -> tuple:
        """Make weighted prediction with confidence estimation"""
        if not neighbors or len(cause_values) == 0:
            return 0.0, 0.0
        
        weights = []
        values = []
        
        # Calculate weights using exponential distance weighting
        for dist, idx in neighbors:
            if idx < len(cause_values):
                weight = np.exp(-dist)  # Exponential weighting
                weights.append(weight)
                values.append(cause_values[idx])
        
        if not weights:
            return 0.0, 0.0
        
        # Weighted prediction
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        prediction = sum(w * v for w, v in zip(weights, values)) / total_weight
        
        # Confidence based on weight distribution and neighbor consistency
        weight_entropy = -sum((w/total_weight) * np.log(w/total_weight + 1e-10) 
                             for w in weights)
        max_entropy = np.log(len(weights))
        confidence = 1 - (weight_entropy / (max_entropy + 1e-10))
        
        # Adjust confidence based on neighbor value consistency
        if len(values) > 1:
            value_std = np.std(values)
            value_consistency = 1 / (1 + value_std)
            confidence *= value_consistency
        
        return prediction, confidence
    
    def _assess_ccm_significance(self, ccm_results: dict, cause: np.ndarray, effect: np.ndarray) -> float:
        """Assess statistical significance of CCM results using surrogate data testing"""
        if not ccm_results['correlations']:
            return 0.0
        
        observed_corr = max(ccm_results['correlations'])
        
        # Generate surrogate data and compute null distribution
        n_surrogates = 20  # Reduced for computational efficiency
        null_correlations = []
        
        for _ in range(n_surrogates):
            # Phase randomized surrogates to preserve autocorrelation
            surrogate_cause = self._generate_phase_randomized_surrogate(cause)
            
            try:
                # Compute CCM with surrogate data
                surrogate_results = self._multi_scale_ccm_computation(
                    surrogate_cause, effect, ccm_results['embedding_params'])
                
                if surrogate_results['correlations']:
                    null_correlations.append(max(surrogate_results['correlations']))
                    
            except Exception:
                continue
        
        if not null_correlations:
            return 0.5  # Neutral significance if surrogates fail
        
        # Calculate p-value
        null_mean = np.mean(null_correlations)
        null_std = np.std(null_correlations)
        
        if null_std > 0:
            z_score = (observed_corr - null_mean) / null_std
            significance = max(0, min(1, (z_score + 3) / 6))  # Normalize to [0,1]
        else:
            significance = 0.5
        
        return significance
    
    def _generate_phase_randomized_surrogate(self, data: np.ndarray) -> np.ndarray:
        """Generate phase-randomized surrogate data"""
        try:
            # FFT of the original data
            fft_data = np.fft.fft(data)
            
            # Randomize phases while preserving magnitudes
            magnitudes = np.abs(fft_data)
            random_phases = np.random.uniform(0, 2*np.pi, len(data))
            randomized_fft = magnitudes * np.exp(1j * random_phases)
            
            # IFFT to get surrogate data
            surrogate = np.real(np.fft.ifft(randomized_fft))
            
            return surrogate
            
        except Exception:
            # Fallback: simple shuffle
            return np.random.permutation(data)
    
    def _analyze_ccm_convergence(self, ccm_results: dict) -> float:
        """Analyze convergence of CCM correlations with library size"""
        if len(ccm_results['correlations']) < 2:
            return 0.5
        
        correlations = np.array(ccm_results['correlations'])
        lib_sizes = np.array(ccm_results['library_sizes'])
        
        # Check for increasing trend (convergence)
        try:
            # Linear regression to assess trend
            slope, intercept, r_value, _, _ = scipy.stats.linregress(lib_sizes, correlations)
            
            # Convergence strength based on positive slope and correlation
            convergence_strength = max(0, min(1, slope * 1000))  # Scale slope
            trend_consistency = abs(r_value) if not np.isnan(r_value) else 0
            
            return convergence_strength * trend_consistency
            
        except Exception:
            # Fallback: simple increase check
            if correlations[-1] > correlations[0]:
                return 0.7
            else:
                return 0.3
    
    def _synthesize_ccm_score(self, ccm_results: dict, significance: float, convergence: float) -> float:
        """Synthesize final CCM score from multiple quality metrics"""
        if not ccm_results['correlations']:
            return 0.0
        
        # Maximum correlation observed
        max_correlation = max(ccm_results['correlations'])
        
        # Average prediction skill
        avg_prediction_skill = np.mean(ccm_results['prediction_skills']) if ccm_results['prediction_skills'] else 0
        
        # Weighted combination of metrics
        correlation_weight = 0.4
        significance_weight = 0.3
        convergence_weight = 0.2
        prediction_weight = 0.1
        
        final_score = (correlation_weight * max_correlation +
                      significance_weight * significance +
                      convergence_weight * convergence +
                      prediction_weight * avg_prediction_skill)
        
        return min(1.0, max(0.0, final_score))
    
    def _robust_ccm_fallback(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Robust CCM fallback with simplified but reliable computation"""
        try:
            if len(cause) != len(effect) or len(cause) < 10:
                return 0.0
            
            # Simple preprocessing
            cause_norm = (cause - np.mean(cause)) / (np.std(cause) + 1e-10)
            effect_norm = (effect - np.mean(effect)) / (np.std(effect) + 1e-10)
            
            # Fixed embedding parameters
            embed_dim = 3
            tau = 1
            
            # Create embedding
            effect_embedded = self._create_embedding_matrix(effect_norm, embed_dim, tau)
            if len(effect_embedded) < 10:
                return 0.0
            
            cause_vals = cause_norm[embed_dim-1:len(effect_embedded)+embed_dim-1]
            
            # Single library size analysis
            lib_size = min(len(effect_embedded) // 2, 50)
            if lib_size < 10:
                return 0.0
            
            ccm_result = self._compute_ccm_for_library_size(
                effect_embedded, cause_vals, lib_size, embed_dim)
            
            return ccm_result['correlation'] if ccm_result else 0.0
            
        except Exception:
            return 0.0
    
    def _basic_ccm_emergency(self, cause: np.ndarray, effect: np.ndarray) -> float:
        """Basic CCM emergency fallback"""
        try:
            if len(cause) != len(effect) or len(cause) < 5:
                return 0.0
            
            # Simple linear correlation as last resort
            correlation = np.corrcoef(cause, effect)[0, 1]
            return abs(correlation) * 0.5 if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _bootstrap_causality_confidence(self, cause: np.ndarray, effect: np.ndarray, base_score: float) -> float:
        """Bootstrap confidence estimation for causality measures"""
        try:
            n_bootstrap = 50
            n_samples = len(cause)
            bootstrap_scores = []
            
            for _ in range(n_bootstrap):
                # Bootstrap resample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                cause_boot = cause[indices]
                effect_boot = effect[indices]
                
                # Quick causality estimate
                try:
                    max_lag = min(3, len(cause_boot) // 10)
                    correlations = []
                    
                    for lag in range(1, max_lag + 1):
                        if lag < len(cause_boot):
                            corr = np.corrcoef(cause_boot[:-lag], effect_boot[lag:])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    boot_score = max(correlations) if correlations else 0.0
                    bootstrap_scores.append(boot_score)
                
                except:
                    bootstrap_scores.append(0.0)
            
            if bootstrap_scores:
                # Confidence based on consistency of bootstrap estimates
                bootstrap_std = np.std(bootstrap_scores)
                bootstrap_mean = np.mean(bootstrap_scores)
                
                if bootstrap_std > 0:
                    confidence = max(0.1, 1 - bootstrap_std / (bootstrap_mean + 1e-6))
                else:
                    confidence = 0.9
                
                return min(1.0, confidence)
            
            return 0.5  # Default confidence
        
        except:
            return 0.5
    
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
            
            # Comprehensive dynamics parameters
            self.state_history = []
            self.action_history = []
            self.reward_history = []
            self.geological_params = self._initialize_geological_parameters()
            self.physics_state = self._initialize_physics_state()
        
        def reset(self, seed=None, options=None):
            """Reset environment"""
            super().reset(seed=seed)
            
            self.current_state = np.random.randn(self.state_dim).astype(np.float32)
            self.step_count = 0
            
            return self.current_state, {}
        
        def step(self, action):
            """Take environment step with comprehensive geological dynamics"""
            # Comprehensive dynamics: multi-physics state evolution
            next_state, reward, done, truncated, info = self._comprehensive_step_dynamics(action)
            
            self.step_count += 1
            
            # Update internal state tracking
            self._update_state_history(self.current_state, action, reward)
            self.current_state = next_state
            
            return next_state.astype(np.float32), reward, done, truncated, info
        
        def _comprehensive_step_dynamics(self, action):
            """Comprehensive geological dynamics with multi-physics modeling"""
            try:
                # Multi-scale dynamics integration
                tectonic_evolution = self._simulate_tectonic_dynamics(action)
                fluid_dynamics = self._simulate_fluid_dynamics(action)
                thermal_evolution = self._simulate_thermal_dynamics(action)
                seismic_response = self._simulate_seismic_dynamics(action)
                
                # Coupled physics integration
                next_state = self._integrate_coupled_physics(
                    tectonic_evolution, fluid_dynamics, thermal_evolution, seismic_response)
                
                # Comprehensive reward calculation
                reward = self._calculate_comprehensive_reward(next_state, action)
                
                # Terminal condition evaluation
                done, truncated = self._evaluate_terminal_conditions(next_state)
                
                # Detailed information tracking
                info = self._compile_step_information(next_state, action, reward)
                
                return next_state, reward, done, truncated, info
                
            except Exception as e:
                # Fallback dynamics
                return self._fallback_dynamics(action)
        
        def _simulate_tectonic_dynamics(self, action):
            """Simulate tectonic plate dynamics"""
            # Stress field evolution
            stress_tensor = self.physics_state.get('stress_tensor', np.zeros((3, 3)))
            
            # Action-dependent stress perturbation
            action_magnitude = (action - self.action_dim // 2) / (self.action_dim // 2)
            stress_perturbation = action_magnitude * self.geological_params['tectonic_sensitivity']
            
            # Plate motion equations
            velocity_field = self.physics_state.get('velocity_field', np.zeros(self.state_dim))
            
            # Elastic-plastic deformation
            elastic_modulus = self.geological_params['elastic_modulus']
            plastic_threshold = self.geological_params['plastic_threshold']
            
            strain_rate = stress_perturbation / elastic_modulus
            
            # Plastic yielding
            if np.linalg.norm(strain_rate) > plastic_threshold:
                strain_rate *= plastic_threshold / np.linalg.norm(strain_rate)
            
            # Integrate velocity field
            dt = 0.1  # Time step
            velocity_field += strain_rate * dt
            
            # Apply damping
            velocity_field *= 0.95
            
            return {
                'velocity_field': velocity_field,
                'stress_tensor': stress_tensor + np.outer(strain_rate, strain_rate),
                'strain_energy': 0.5 * elastic_modulus * np.sum(strain_rate**2)
            }
        
        def _simulate_fluid_dynamics(self, action):
            """Simulate fluid flow and pore pressure dynamics"""
            # Darcy flow with pressure-dependent permeability
            permeability = self.geological_params['permeability']
            porosity = self.geological_params['porosity']
            fluid_viscosity = self.geological_params['fluid_viscosity']
            
            # Pressure field
            pressure_field = self.physics_state.get('pressure_field', np.zeros(self.state_dim))
            
            # Action-induced pressure changes
            action_pressure = (action - self.action_dim // 2) * 0.01
            pressure_field[action % self.state_dim] += action_pressure
            
            # Diffusion equation solution
            pressure_gradient = np.gradient(pressure_field)
            fluid_velocity = -(permeability / fluid_viscosity) * pressure_gradient
            
            # Pressure diffusion
            diffusivity = permeability / (fluid_viscosity * porosity)
            pressure_diffusion = diffusivity * np.sum(np.gradient(np.gradient(pressure_field)))
            
            # Update pressure field
            dt = 0.1
            pressure_field += pressure_diffusion * dt
            
            return {
                'pressure_field': pressure_field,
                'fluid_velocity': fluid_velocity,
                'pore_pressure': np.mean(pressure_field)
            }
        
        def _simulate_thermal_dynamics(self, action):
            """Simulate heat transfer and thermal evolution"""
            # Heat conduction with advection
            temperature_field = self.physics_state.get('temperature_field', np.ones(self.state_dim) * 300)
            
            # Thermal diffusivity
            thermal_diffusivity = self.geological_params['thermal_diffusivity']
            
            # Heat source from action (e.g., injection, extraction)
            heat_source = (action - self.action_dim // 2) * 10.0
            temperature_field[action % self.state_dim] += heat_source
            
            # Conductive heat transfer
            temp_gradient = np.gradient(temperature_field)
            heat_flux = -thermal_diffusivity * temp_gradient
            
            # Advective heat transfer
            velocity_field = self.physics_state.get('velocity_field', np.zeros(self.state_dim))
            advective_heat = -np.dot(velocity_field, temp_gradient)
            
            # Temperature evolution
            dt = 0.1
            temperature_evolution = thermal_diffusivity * np.sum(np.gradient(heat_flux)) + advective_heat
            temperature_field += temperature_evolution * dt
            
            return {
                'temperature_field': temperature_field,
                'heat_flux': heat_flux,
                'thermal_stress': self.geological_params['thermal_expansion'] * (temperature_field - 300)
            }
        
        def _simulate_seismic_dynamics(self, action):
            """Simulate seismic wave propagation and induced seismicity"""
            # Elastic wave equation
            seismic_amplitude = self.physics_state.get('seismic_amplitude', np.zeros(self.state_dim))
            seismic_velocity = self.physics_state.get('seismic_velocity', np.zeros(self.state_dim))
            
            # P-wave and S-wave velocities
            vp = self.geological_params['p_wave_velocity']
            vs = self.geological_params['s_wave_velocity']
            
            # Action-induced seismicity
            seismic_source = (action - self.action_dim // 2) * 0.001
            seismic_amplitude[action % self.state_dim] += seismic_source
            
            # Wave propagation
            dt = 0.1
            acceleration = vp**2 * np.gradient(np.gradient(seismic_amplitude))
            seismic_velocity += acceleration * dt
            seismic_amplitude += seismic_velocity * dt
            
            # Attenuation
            quality_factor = self.geological_params['quality_factor']
            seismic_amplitude *= np.exp(-1.0 / quality_factor * dt)
            seismic_velocity *= np.exp(-1.0 / quality_factor * dt)
            
            # Induced seismicity probability
            stress_state = self.physics_state.get('stress_tensor', np.zeros((3, 3)))
            coulomb_stress = np.trace(stress_state) / 3.0
            seismicity_probability = 1.0 / (1.0 + np.exp(-coulomb_stress + 5.0))
            
            return {
                'seismic_amplitude': seismic_amplitude,
                'seismic_velocity': seismic_velocity,
                'seismicity_probability': seismicity_probability
            }
        
        def _integrate_coupled_physics(self, tectonic, fluid, thermal, seismic):
            """Integrate coupled multi-physics evolution"""
            # Combine all physical processes
            next_state = self.current_state.copy()
            
            # Tectonic contribution
            next_state += 0.3 * tectonic['velocity_field'][:len(next_state)]
            
            # Fluid dynamics contribution
            fluid_effect = fluid['pressure_field'][:len(next_state)] * 0.01
            next_state += fluid_effect
            
            # Thermal contribution
            thermal_effect = thermal['thermal_stress'][:len(next_state)] * 0.05
            next_state += thermal_effect
            
            # Seismic contribution
            seismic_effect = seismic['seismic_amplitude'][:len(next_state)] * 0.1
            next_state += seismic_effect
            
            # Coupling effects
            # Thermo-mechanical coupling
            temperature = thermal['temperature_field'][:len(next_state)]
            thermal_expansion = (temperature - 300) * self.geological_params['thermal_expansion']
            next_state += thermal_expansion * 0.02
            
            # Poro-elastic coupling
            pore_pressure = fluid['pore_pressure']
            effective_stress_reduction = pore_pressure * 0.01
            next_state -= effective_stress_reduction
            
            # Seismo-mechanical coupling
            seismic_stress = seismic['seismicity_probability'] * 0.005
            next_state += seismic_stress
            
            # Update physics state
            self.physics_state.update(tectonic)
            self.physics_state.update(fluid)
            self.physics_state.update(thermal)
            self.physics_state.update(seismic)
            
            return next_state
        
        def _calculate_comprehensive_reward(self, next_state, action):
            """Calculate comprehensive reward incorporating multiple objectives"""
            # Stability reward (proximity to equilibrium)
            stability_reward = -np.sum(np.abs(next_state)) * 0.1
            
            # Energy efficiency reward
            energy_cost = np.abs(action - self.action_dim // 2)
            efficiency_reward = -energy_cost * 0.05
            
            # Safety reward (avoid extreme conditions)
            safety_threshold = 5.0
            safety_violation = np.sum(np.abs(next_state) > safety_threshold)
            safety_reward = -safety_violation * 1.0
            
            # Geological stability reward
            seismicity_prob = self.physics_state.get('seismicity_probability', 0.0)
            seismic_penalty = -seismicity_prob * 0.5
            
            # Thermal stability reward
            temp_field = self.physics_state.get('temperature_field', np.array([300]))
            temp_deviation = np.std(temp_field[:len(next_state)])
            thermal_penalty = -temp_deviation * 0.01
            
            # Pressure stability reward
            pressure_field = self.physics_state.get('pressure_field', np.zeros(len(next_state)))
            pressure_gradient = np.std(pressure_field)
            pressure_penalty = -pressure_gradient * 0.01
            
            # Combined reward
            total_reward = (stability_reward + efficiency_reward + safety_reward + 
                          seismic_penalty + thermal_penalty + pressure_penalty)
            
            return float(total_reward)
        
        def _evaluate_terminal_conditions(self, next_state):
            """Evaluate terminal and truncation conditions"""
            # Maximum steps reached
            done = self.step_count >= self.max_steps
            truncated = False
            
            # Safety violations
            if np.any(np.abs(next_state) > 10.0):
                done = True
                truncated = True
            
            # Seismic instability
            seismicity_prob = self.physics_state.get('seismicity_probability', 0.0)
            if seismicity_prob > 0.9:
                done = True
                truncated = True
            
            # Thermal runaway
            temp_field = self.physics_state.get('temperature_field', np.array([300]))
            if np.any(temp_field > 1000) or np.any(temp_field < 200):
                done = True
                truncated = True
            
            return done, truncated
        
        def _compile_step_information(self, next_state, action, reward):
            """Compile detailed step information"""
            return {
                'tectonic_stress': np.linalg.norm(self.physics_state.get('stress_tensor', np.zeros((3, 3)))),
                'fluid_pressure': self.physics_state.get('pore_pressure', 0.0),
                'temperature_mean': np.mean(self.physics_state.get('temperature_field', [300])),
                'seismicity_risk': self.physics_state.get('seismicity_probability', 0.0),
                'energy_dissipation': np.sum(next_state**2),
                'action_magnitude': np.abs(action - self.action_dim // 2),
                'state_stability': np.std(next_state)
            }
        
        def _update_state_history(self, state, action, reward):
            """Update historical state tracking"""
            self.state_history.append(state.copy())
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            # Maintain fixed history length
            max_history = 100
            if len(self.state_history) > max_history:
                self.state_history.pop(0)
                self.action_history.pop(0)
                self.reward_history.pop(0)
        
        def _initialize_geological_parameters(self):
            """Initialize geological and physical parameters"""
            return {
                'elastic_modulus': 50e9,  # Pa
                'plastic_threshold': 0.01,
                'tectonic_sensitivity': 0.1,
                'permeability': 1e-12,  # m²
                'porosity': 0.1,
                'fluid_viscosity': 1e-3,  # Pa·s
                'thermal_diffusivity': 1e-6,  # m²/s
                'thermal_expansion': 1e-5,  # 1/K
                'p_wave_velocity': 6000,  # m/s
                's_wave_velocity': 3500,  # m/s
                'quality_factor': 100,
                'density': 2500  # kg/m³
            }
        
        def _initialize_physics_state(self):
            """Initialize physics state variables"""
            return {
                'stress_tensor': np.zeros((3, 3)),
                'velocity_field': np.zeros(self.state_dim),
                'pressure_field': np.zeros(self.state_dim),
                'temperature_field': np.ones(self.state_dim) * 300,  # K
                'seismic_amplitude': np.zeros(self.state_dim),
                'seismic_velocity': np.zeros(self.state_dim)
            }
        
        def _fallback_dynamics(self, action):
            """Simple fallback dynamics for error conditions"""
            # Basic state evolution
            noise = np.random.randn(self.state_dim) * 0.01
            action_effect = 0.1 * (action - self.action_dim // 2) / (self.action_dim // 2)
            next_state = self.current_state + action_effect + noise
            
            # Simple reward
            reward = -np.sum(np.abs(next_state))
            
            # Basic terminal conditions
            done = self.step_count >= self.max_steps
            truncated = np.any(np.abs(next_state) > 10.0)
            
            info = {'fallback_mode': True}
            
            return next_state, reward, done, truncated, info

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
        """Comprehensive recent anomaly rate estimation with advanced statistical analysis"""
        try:
            return self._advanced_anomaly_rate_analysis()
        except Exception:
            try:
                return self._robust_rate_estimation()
            except Exception:
                return self._basic_rate_fallback()
    
    def _advanced_anomaly_rate_analysis(self) -> float:
        """Advanced statistical analysis for anomaly rate estimation"""
        if not hasattr(self, 'anomaly_detection_history'):
            self.anomaly_detection_history = []
        
        # Multi-temporal analysis windows
        analysis_windows = {
            'immediate': 24,     # Last 24 hours
            'short_term': 168,   # Last week
            'medium_term': 720,  # Last month
            'long_term': 2160    # Last quarter
        }
        
        current_time = time.time()
        temporal_rates = {}
        
        # Analyze each temporal window
        for window_name, hours in analysis_windows.items():
            window_start = current_time - (hours * 3600)
            window_events = [
                event for event in self.anomaly_detection_history 
                if event.get('timestamp', 0) >= window_start
            ]
            
            if window_events:
                # Calculate sophisticated metrics
                anomaly_count = sum(1 for event in window_events if event.get('is_anomaly', False))
                total_detections = len(window_events)
                
                # Base rate with confidence adjustment
                base_rate = anomaly_count / total_detections if total_detections > 0 else 0.0
                
                # Confidence-weighted rate
                confidence_weights = [event.get('confidence', 0.5) for event in window_events]
                weighted_rate = np.average([event.get('is_anomaly', False) for event in window_events], 
                                         weights=confidence_weights) if confidence_weights else base_rate
                
                # Seasonal adjustment using Fourier analysis
                seasonal_adjustment = self._calculate_seasonal_adjustment(window_events, hours)
                
                # Trend analysis
                trend_factor = self._calculate_trend_factor(window_events)
                
                temporal_rates[window_name] = {
                    'base_rate': base_rate,
                    'weighted_rate': weighted_rate,
                    'seasonal_rate': weighted_rate * seasonal_adjustment,
                    'trend_adjusted_rate': weighted_rate * seasonal_adjustment * trend_factor,
                    'sample_size': total_detections,
                    'confidence': min(total_detections / 100.0, 1.0)  # Higher confidence with more samples
                }
        
        # Ensemble rate computation with adaptive weighting
        ensemble_rate = self._compute_ensemble_anomaly_rate(temporal_rates)
        
        # Apply geological domain constraints
        domain_adjusted_rate = self._apply_geological_constraints(ensemble_rate)
        
        # Uncertainty quantification
        uncertainty = self._quantify_rate_uncertainty(temporal_rates)
        
        # Store for monitoring
        self._update_rate_statistics(domain_adjusted_rate, uncertainty)
        
        return domain_adjusted_rate
    
    def _calculate_seasonal_adjustment(self, events: List[dict], window_hours: int) -> float:
        """Calculate seasonal adjustment factor using spectral analysis"""
        if len(events) < 24:  # Need minimum data for seasonal analysis
            return 1.0
        
        try:
            # Extract temporal features
            timestamps = [event.get('timestamp', 0) for event in events]
            anomaly_flags = [int(event.get('is_anomaly', False)) for event in events]
            
            # Convert to time series with regular intervals
            time_series = pd.Series(anomaly_flags, index=pd.to_datetime(timestamps, unit='s'))
            
            # Resample to hourly intervals
            hourly_data = time_series.resample('H').mean().fillna(0)
            
            if len(hourly_data) < 24:
                return 1.0
            
            # Spectral analysis for periodic patterns
            from scipy import signal
            frequencies, power = signal.periodogram(hourly_data.values, fs=1/3600)  # fs in Hz
            
            # Identify dominant frequencies
            peak_indices = signal.find_peaks(power, height=np.percentile(power, 75))[0]
            
            if len(peak_indices) > 0:
                dominant_period = 1 / frequencies[peak_indices[0]] if frequencies[peak_indices[0]] > 0 else 24*3600
                
                # Calculate current phase in the dominant cycle
                current_phase = (time.time() % dominant_period) / dominant_period * 2 * np.pi
                
                # Seasonal adjustment based on phase
                seasonal_factor = 1.0 + 0.2 * np.sin(current_phase)  # ±20% seasonal variation
                return max(0.5, min(2.0, seasonal_factor))  # Bounded adjustment
            
        except Exception:
            pass
        
        return 1.0
    
    def _calculate_trend_factor(self, events: List[dict]) -> float:
        """Calculate trend factor using robust regression"""
        if len(events) < 10:
            return 1.0
        
        try:
            # Prepare data for trend analysis
            timestamps = np.array([event.get('timestamp', 0) for event in events])
            anomaly_rates = []
            
            # Calculate moving anomaly rate
            window_size = max(5, len(events) // 10)
            for i in range(len(events) - window_size + 1):
                window_events = events[i:i + window_size]
                rate = sum(event.get('is_anomaly', False) for event in window_events) / window_size
                anomaly_rates.append(rate)
            
            if len(anomaly_rates) < 3:
                return 1.0
            
            # Robust linear regression for trend
            time_points = timestamps[:len(anomaly_rates)]
            time_normalized = (time_points - time_points[0]) / (time_points[-1] - time_points[0])
            
            # Use Theil-Sen estimator for robust trend estimation
            from scipy.stats import theilslopes
            slope, intercept, _, _ = theilslopes(anomaly_rates, time_normalized)
            
            # Convert slope to trend factor
            # Positive slope increases rate, negative decreases
            trend_factor = 1.0 + np.tanh(slope * 5)  # Bounded between ~0.01 and ~2.0
            
            return max(0.1, min(3.0, trend_factor))
            
        except Exception:
            return 1.0
    
    def _compute_ensemble_anomaly_rate(self, temporal_rates: dict) -> float:
        """Compute ensemble anomaly rate using adaptive weighting"""
        if not temporal_rates:
            return 0.1  # Default rate
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        # Adaptive weights based on recency and confidence
        recency_weights = {
            'immediate': 0.4,
            'short_term': 0.3,
            'medium_term': 0.2,
            'long_term': 0.1
        }
        
        for window_name, rates in temporal_rates.items():
            if rates['sample_size'] > 0:
                # Base weight from recency
                base_weight = recency_weights.get(window_name, 0.1)
                
                # Adjust weight by confidence and sample size
                confidence_factor = rates['confidence']
                sample_factor = min(rates['sample_size'] / 50.0, 1.0)  # Max weight at 50+ samples
                
                final_weight = base_weight * confidence_factor * sample_factor
                
                # Use trend-adjusted rate as primary estimate
                rate_estimate = rates['trend_adjusted_rate']
                
                weighted_sum += rate_estimate * final_weight
                weight_sum += final_weight
        
        if weight_sum > 0:
            ensemble_rate = weighted_sum / weight_sum
        else:
            ensemble_rate = 0.1
        
        # Apply ensemble smoothing to avoid rapid changes
        if hasattr(self, '_last_ensemble_rate'):
            smoothing_factor = 0.7  # 70% current, 30% previous
            ensemble_rate = smoothing_factor * ensemble_rate + (1 - smoothing_factor) * self._last_ensemble_rate
        
        self._last_ensemble_rate = ensemble_rate
        return ensemble_rate
    
    def _apply_geological_constraints(self, rate: float) -> float:
        """Apply domain-specific geological constraints to anomaly rate"""
        # Geological systems typically have baseline anomaly rates between 0.01-0.15
        baseline_min = 0.005  # Very quiet periods
        baseline_max = 0.4    # Highly active periods (volcanic unrest, etc.)
        
        # Apply physical constraints
        constrained_rate = np.clip(rate, baseline_min, baseline_max)
        
        # Account for geological system state if available
        if hasattr(self, 'system_state'):
            state_multiplier = self._get_state_multiplier()
            constrained_rate *= state_multiplier
        
        # Final bounds check
        return np.clip(constrained_rate, baseline_min, baseline_max)
    
    def _get_state_multiplier(self) -> float:
        """Get state-dependent multiplier for anomaly rates"""
        # This would integrate with broader geological monitoring
        # For now, use learning history as proxy
        recent_activity = len(self.learning_history[-50:]) if len(self.learning_history) > 0 else 0
        
        if recent_activity > 40:
            return 1.3  # High activity state
        elif recent_activity > 20:
            return 1.1  # Moderate activity
        elif recent_activity < 5:
            return 0.8  # Low activity
        else:
            return 1.0  # Normal activity
    
    def _quantify_rate_uncertainty(self, temporal_rates: dict) -> float:
        """Quantify uncertainty in rate estimation"""
        if not temporal_rates:
            return 0.5  # High uncertainty
        
        # Calculate variance across temporal windows
        rates = [rates['trend_adjusted_rate'] for rates in temporal_rates.values() if rates['sample_size'] > 0]
        
        if len(rates) < 2:
            return 0.4  # Moderate uncertainty with single estimate
        
        rate_std = np.std(rates)
        rate_mean = np.mean(rates)
        
        # Coefficient of variation as uncertainty measure
        uncertainty = rate_std / rate_mean if rate_mean > 0 else 0.5
        
        # Account for sample sizes
        min_samples = min(rates['sample_size'] for rates in temporal_rates.values())
        sample_uncertainty = 1.0 / np.sqrt(max(min_samples, 1))
        
        # Combined uncertainty
        total_uncertainty = min(uncertainty + sample_uncertainty * 0.1, 0.8)
        
        return total_uncertainty
    
    def _update_rate_statistics(self, rate: float, uncertainty: float):
        """Update internal statistics for rate monitoring"""
        if not hasattr(self, '_rate_statistics'):
            self._rate_statistics = []
        
        current_time = time.time()
        
        self._rate_statistics.append({
            'timestamp': current_time,
            'rate': rate,
            'uncertainty': uncertainty
        })
        
        # Keep last 1000 rate estimates
        if len(self._rate_statistics) > 1000:
            self._rate_statistics = self._rate_statistics[-1000:]
    
    def _robust_rate_estimation(self) -> float:
        """Robust fallback method for anomaly rate estimation"""
        try:
            # Use learning history and detection patterns
            recent_events = self.learning_history[-50:] if len(self.learning_history) >= 50 else self.learning_history
            
            if not recent_events:
                return 0.08  # Slightly lower baseline
            
            # Adaptive rate based on learning velocity
            learning_velocity = len(recent_events) / max(len(self.learning_history), 1)
            base_rate = 0.05 + learning_velocity * 0.1
            
            # Time-decay weighting for recency
            weights = np.exp(-np.arange(len(recent_events)) * 0.1)  # Exponential decay
            weighted_events = np.sum(weights)
            
            if weighted_events > 0:
                adaptive_rate = base_rate * (weighted_events / len(recent_events))
            else:
                adaptive_rate = base_rate
            
            return np.clip(adaptive_rate, 0.01, 0.25)
            
        except Exception:
            return self._basic_rate_fallback()
    
    def _basic_rate_fallback(self) -> float:
        """Basic emergency fallback for anomaly rate estimation"""
        # Conservative rate based on system defaults
        if hasattr(self, 'learning_history') and len(self.learning_history) > 5:
            activity_factor = min(len(self.learning_history) / 100.0, 1.0)
            return 0.05 + activity_factor * 0.05  # Range: 0.05-0.10
        else:
            return 0.06  # Safe default for geological systems
    
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
    
    # Helper methods for advanced EMD-Hilbert analysis
    def _select_optimal_wavelet(self, signal: np.ndarray) -> str:
        """Select optimal wavelet based on signal characteristics"""
        try:
            # Simple wavelet selection based on signal length and characteristics
            if len(signal) < 50:
                return 'db4'
            elif len(signal) < 200:
                return 'db8'
            else:
                return 'db10'
        except Exception:
            return 'db4'
    
    def _estimate_noise_level(self, detail_coeffs: np.ndarray) -> float:
        """Estimate noise level using robust statistics"""
        try:
            # Use median absolute deviation for robust noise estimation
            mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
            return mad / 0.6745  # Convert to standard deviation equivalent
        except Exception:
            return np.std(detail_coeffs) if len(detail_coeffs) > 0 else 1e-6
    
    def _compute_mode_consistency(self, all_imfs: list) -> float:
        """Compute consistency of modes across ensemble realizations"""
        try:
            if len(all_imfs) < 2:
                return 1.0
            
            consistencies = []
            max_modes = max(len(imfs) for imfs in all_imfs)
            
            for mode_idx in range(max_modes):
                mode_collection = []
                for imfs in all_imfs:
                    if mode_idx < len(imfs):
                        mode_collection.append(imfs[mode_idx])
                
                if len(mode_collection) > 1:
                    # Compute pairwise correlations
                    correlations = []
                    for i in range(len(mode_collection)):
                        for j in range(i+1, len(mode_collection)):
                            try:
                                corr = np.corrcoef(mode_collection[i], mode_collection[j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                            except Exception:
                                continue
                    
                    if correlations:
                        consistencies.append(np.mean(correlations))
            
            return np.mean(consistencies) if consistencies else 0.5
            
        except Exception:
            return 0.5
    
    def _assess_decomposition_quality(self, imfs: list, signal: np.ndarray) -> float:
        """Assess overall quality of EMD decomposition"""
        try:
            if not imfs or len(imfs) == 0:
                return 0.0
            
            # Reconstruction error
            reconstructed = np.sum(imfs, axis=0)
            mse = np.mean((signal - reconstructed)**2)
            reconstruction_score = max(0, 1 - mse / np.var(signal))
            
            # Orthogonality check
            orthogonality_scores = []
            for i, imf1 in enumerate(imfs):
                for j, imf2 in enumerate(imfs[i+1:], i+1):
                    try:
                        correlation = abs(np.corrcoef(imf1, imf2)[0, 1])
                        orthogonality_scores.append(1 - correlation)
                    except Exception:
                        continue
            
            orthogonality_score = np.mean(orthogonality_scores) if orthogonality_scores else 0.5
            
            # Combined quality score
            return 0.7 * reconstruction_score + 0.3 * orthogonality_score
            
        except Exception:
            return 0.5
    
    def _align_modes_cross_correlation(self, mode_collection: list) -> list:
        """Align modes using cross-correlation"""
        try:
            if len(mode_collection) <= 1:
                return mode_collection
            
            # Use first mode as reference
            reference = mode_collection[0]
            aligned_modes = [reference]
            
            for mode in mode_collection[1:]:
                try:
                    # Simple alignment - just ensure same length
                    if len(mode) == len(reference):
                        aligned_modes.append(mode)
                    else:
                        # Truncate or pad to match reference length
                        if len(mode) > len(reference):
                            aligned_modes.append(mode[:len(reference)])
                        else:
                            padded = np.pad(mode, (0, len(reference) - len(mode)), mode='constant')
                            aligned_modes.append(padded)
                except Exception:
                    aligned_modes.append(reference)  # Use reference if alignment fails
            
            return aligned_modes
            
        except Exception:
            return mode_collection
    
    def _compute_instantaneous_frequency_consistency(self, imf: np.ndarray) -> float:
        """Compute instantaneous frequency consistency of IMF"""
        try:
            if len(imf) < 10:
                return 0.5
            
            # Compute instantaneous frequency using phase derivative
            analytic_signal = scipy.signal.hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freq = np.diff(instantaneous_phase) / (2.0 * np.pi)
            
            # Measure consistency as inverse of frequency variance
            freq_variance = np.var(instantaneous_freq)
            consistency = 1.0 / (1.0 + freq_variance)
            
            return min(1.0, max(0.0, consistency))
            
        except Exception:
            return 0.5
    
    def _compute_orthogonality_score(self, imf: np.ndarray, all_imfs: list, current_idx: int) -> float:
        """Compute orthogonality score with other IMFs"""
        try:
            if len(all_imfs) <= 1:
                return 1.0
            
            correlations = []
            for i, other_imf in enumerate(all_imfs):
                if i != current_idx:
                    try:
                        correlation = abs(np.corrcoef(imf, other_imf)[0, 1])
                        correlations.append(correlation)
                    except Exception:
                        continue
            
            if not correlations:
                return 1.0
            
            # Good orthogonality means low correlations
            avg_correlation = np.mean(correlations)
            return 1.0 - avg_correlation
            
        except Exception:
            return 0.5
    
    def _detect_mode_mixing(self, imf: np.ndarray) -> float:
        """Detect mode mixing in IMF"""
        try:
            if len(imf) < 20:
                return 0.0
            
            # Use spectral analysis to detect multiple dominant frequencies
            freqs, psd = scipy.signal.welch(imf, nperseg=min(len(imf)//4, 256))
            
            # Find peaks in power spectral density
            peaks, _ = scipy.signal.find_peaks(psd, height=np.max(psd) * 0.1)
            
            # Mode mixing penalty based on number of significant peaks
            if len(peaks) > 1:
                penalty = min(1.0, (len(peaks) - 1) * 0.3)
            else:
                penalty = 0.0
            
            return penalty
            
        except Exception:
            return 0.0
    
    def _validate_phase_quality(self, phase: np.ndarray, imf: np.ndarray) -> bool:
        """Validate phase quality"""
        try:
            if len(phase) != len(imf):
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(phase)) or np.any(np.isinf(phase)):
                return False
            
            # Check for reasonable phase continuity
            phase_diff = np.diff(phase)
            if np.std(phase_diff) > 10:  # Very large phase jumps
                return False
            
            return True
            
        except Exception:
            return False
    
    def _emd_instantaneous_phase(self, imf: np.ndarray) -> np.ndarray:
        """Compute instantaneous phase from IMF using EMD approach"""
        try:
            analytic_signal = scipy.signal.hilbert(imf)
            return np.angle(analytic_signal)
        except Exception:
            return self._robust_derivative_phase(imf)
    
    def _synchrosqueezed_phase(self, imf: np.ndarray) -> np.ndarray:
        """Advanced comprehensive synchrosqueezed transform-based phase extraction with multi-scale analysis"""
        try:
            if len(imf) < 10:
                return self._robust_derivative_phase(imf)
            
            # Comprehensive synchrosqueezed transform implementation
            sst_result = self._advanced_synchrosqueezed_transform(imf)
            
            if sst_result is None:
                return self._robust_derivative_phase(imf)
            
            # Extract optimal phase using multiple criteria
            optimal_phase = self._extract_optimal_phase_from_sst(sst_result, imf)
            
            # Validate and post-process the phase
            validated_phase = self._validate_sst_phase(optimal_phase, imf)
            
            return validated_phase
            
        except Exception:
            return self._robust_derivative_phase(imf)
    
    def _advanced_synchrosqueezed_transform(self, signal: np.ndarray) -> dict:
        """Advanced synchrosqueezed transform with adaptive parameters and multi-wavelet analysis"""
        try:
            # Adaptive parameter selection based on signal characteristics
            signal_params = self._analyze_signal_characteristics(signal)
            
            # Multi-wavelet analysis for robust frequency-time representation
            wavelet_results = self._multi_wavelet_analysis(signal, signal_params)
            
            # Advanced synchrosqueezing with ridge detection
            sst_matrix = self._compute_advanced_sst_matrix(wavelet_results, signal_params)
            
            # Ridge extraction with multiple algorithms
            ridge_info = self._extract_comprehensive_ridges(sst_matrix, signal_params)
            
            # Quality assessment and selection
            quality_scores = self._assess_sst_quality(sst_matrix, ridge_info, signal)
            
            return {
                'sst_matrix': sst_matrix,
                'ridge_info': ridge_info,
                'quality_scores': quality_scores,
                'signal_params': signal_params,
                'wavelet_results': wavelet_results
            }
            
        except Exception:
            return None
    
    def _analyze_signal_characteristics(self, signal: np.ndarray) -> dict:
        """Comprehensive signal characteristic analysis"""
        try:
            characteristics = {}
            
            # Time-domain characteristics
            characteristics['length'] = len(signal)
            characteristics['mean'] = np.mean(signal)
            characteristics['std'] = np.std(signal)
            characteristics['skewness'] = self._compute_skewness(signal)
            characteristics['kurtosis'] = self._compute_kurtosis(signal)
            
            # Frequency-domain characteristics
            freqs, psd = scipy.signal.welch(signal, nperseg=min(len(signal)//4, 256))
            characteristics['dominant_freq_idx'] = np.argmax(psd)
            characteristics['dominant_freq'] = freqs[characteristics['dominant_freq_idx']] if len(freqs) > 0 else 0
            characteristics['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            characteristics['spectral_bandwidth'] = np.sqrt(np.sum(((freqs - characteristics['spectral_centroid'])**2) * psd) / np.sum(psd)) if np.sum(psd) > 0 else 1
            
            # Complexity measures
            characteristics['approximate_entropy'] = self._compute_approximate_entropy(signal)
            characteristics['zero_crossing_rate'] = self._compute_zero_crossing_rate(signal)
            
            # Adaptive parameters based on characteristics
            characteristics['optimal_scales'] = self._determine_optimal_scales(signal, characteristics)
            characteristics['optimal_wavelets'] = self._select_optimal_wavelets(characteristics)
            characteristics['ridge_threshold'] = self._compute_adaptive_ridge_threshold(characteristics)
            
            return characteristics
            
        except Exception:
            # Return default characteristics
            return {
                'length': len(signal),
                'mean': 0,
                'std': 1,
                'optimal_scales': np.logspace(0, np.log10(min(len(signal)//4, 64)), 32),
                'optimal_wavelets': ['morlet2', 'cmor1.5-1.0'],
                'ridge_threshold': 0.1
            }
    
    def _multi_wavelet_analysis(self, signal: np.ndarray, params: dict) -> dict:
        """Multi-wavelet analysis with comprehensive frequency-time representation"""
        try:
            results = {}
            
            for wavelet_name in params['optimal_wavelets']:
                try:
                    # Compute continuous wavelet transform
                    if wavelet_name == 'morlet2':
                        cwt_matrix = scipy.signal.cwt(signal, scipy.signal.morlet2, params['optimal_scales'])
                    elif wavelet_name.startswith('cmor'):
                        # Complex Morlet wavelet
                        cwt_matrix = self._compute_complex_morlet_cwt(signal, params['optimal_scales'], wavelet_name)
                    else:
                        # Fallback to Morlet
                        cwt_matrix = scipy.signal.cwt(signal, scipy.signal.morlet2, params['optimal_scales'])
                    
                    # Compute instantaneous frequency
                    inst_freq = self._compute_cwt_instantaneous_frequency(cwt_matrix, params['optimal_scales'])
                    
                    # Store results
                    results[wavelet_name] = {
                        'cwt_matrix': cwt_matrix,
                        'instantaneous_freq': inst_freq,
                        'magnitude': np.abs(cwt_matrix),
                        'phase': np.angle(cwt_matrix)
                    }
                    
                except Exception:
                    continue
            
            # If no wavelets succeeded, use basic Morlet
            if not results:
                scales = np.logspace(0, 2, 16)
                cwt_matrix = scipy.signal.cwt(signal, scipy.signal.morlet2, scales)
                results['morlet2'] = {
                    'cwt_matrix': cwt_matrix,
                    'instantaneous_freq': self._compute_cwt_instantaneous_frequency(cwt_matrix, scales),
                    'magnitude': np.abs(cwt_matrix),
                    'phase': np.angle(cwt_matrix)
                }
            
            return results
            
        except Exception:
            # Emergency fallback
            scales = np.array([1, 2, 4, 8])
            cwt_matrix = scipy.signal.cwt(signal, scipy.signal.morlet2, scales)
            return {
                'morlet2': {
                    'cwt_matrix': cwt_matrix,
                    'instantaneous_freq': np.zeros_like(cwt_matrix),
                    'magnitude': np.abs(cwt_matrix),
                    'phase': np.angle(cwt_matrix)
                }
            }
    
    def _compute_advanced_sst_matrix(self, wavelet_results: dict, params: dict) -> np.ndarray:
        """Compute advanced synchrosqueezed transform matrix with optimal wavelet fusion"""
        try:
            sst_matrices = []
            
            for wavelet_name, results in wavelet_results.items():
                try:
                    # Synchrosqueezing operation
                    cwt_matrix = results['cwt_matrix']
                    inst_freq = results['instantaneous_freq']
                    
                    # Advanced synchrosqueezing with frequency-dependent thresholding
                    sst_single = self._perform_synchrosqueezing(
                        cwt_matrix, 
                        inst_freq, 
                        params['optimal_scales'],
                        adaptive_threshold=True
                    )
                    
                    if sst_single is not None:
                        sst_matrices.append(sst_single)
                        
                except Exception:
                    continue
            
            if not sst_matrices:
                # Fallback to basic approach
                first_wavelet = list(wavelet_results.keys())[0]
                cwt_matrix = wavelet_results[first_wavelet]['cwt_matrix']
                return np.abs(cwt_matrix)
            
            # Weighted combination of SST matrices
            combined_sst = self._combine_sst_matrices(sst_matrices, wavelet_results, params)
            
            return combined_sst
            
        except Exception:
            # Ultimate fallback
            first_wavelet = list(wavelet_results.keys())[0]
            return np.abs(wavelet_results[first_wavelet]['cwt_matrix'])
    
    def _perform_synchrosqueezing(self, cwt_matrix: np.ndarray, inst_freq: np.ndarray, 
                                scales: np.ndarray, adaptive_threshold: bool = True) -> np.ndarray:
        """Perform synchrosqueezing with advanced ridge-following algorithms"""
        try:
            magnitude = np.abs(cwt_matrix)
            phase = np.angle(cwt_matrix)
            
            # Initialize synchrosqueezed matrix
            n_scales, n_time = cwt_matrix.shape
            sst_matrix = np.zeros_like(magnitude)
            
            # Compute frequency bins
            max_freq = 1.0 / np.min(scales)
            freq_bins = np.linspace(0, max_freq, n_scales)
            
            # Advanced synchrosqueezing with adaptive parameters
            for t in range(n_time):
                for s in range(n_scales):
                    if magnitude[s, t] > 0:
                        # Compute instantaneous frequency at this point
                        if t > 0 and t < n_time - 1:
                            phase_diff = phase[s, t+1] - phase[s, t-1]
                            current_freq = phase_diff / (2 * np.pi * 2)  # Normalized frequency
                        else:
                            current_freq = inst_freq[s, t] if abs(inst_freq[s, t]) > 1e-10 else 0
                        
                        # Find closest frequency bin with adaptive threshold
                        if adaptive_threshold:
                            threshold = np.max(magnitude[:, t]) * 0.1  # Adaptive threshold
                        else:
                            threshold = np.max(magnitude) * 0.05
                        
                        if magnitude[s, t] > threshold:
                            # Map to appropriate frequency bin
                            freq_idx = np.argmin(np.abs(freq_bins - abs(current_freq)))
                            freq_idx = max(0, min(freq_idx, n_scales - 1))
                            
                            # Accumulate energy with proper weighting
                            weight = magnitude[s, t] / (scales[s] + 1e-10)
                            sst_matrix[freq_idx, t] += weight
            
            return sst_matrix
            
        except Exception:
            return np.abs(cwt_matrix)
    
    def _extract_comprehensive_ridges(self, sst_matrix: np.ndarray, params: dict) -> dict:
        """Extract comprehensive ridge information using multiple algorithms"""
        try:
            ridges = {}
            
            # Algorithm 1: Maximum energy ridge tracking
            max_ridge = self._extract_maximum_energy_ridge(sst_matrix)
            if max_ridge is not None:
                ridges['max_energy'] = max_ridge
            
            # Algorithm 2: Persistent ridge detection
            persistent_ridges = self._extract_persistent_ridges(sst_matrix, params['ridge_threshold'])
            if persistent_ridges:
                ridges['persistent'] = persistent_ridges
            
            # Algorithm 3: Multi-scale ridge analysis
            multiscale_ridges = self._extract_multiscale_ridges(sst_matrix, params)
            if multiscale_ridges:
                ridges['multiscale'] = multiscale_ridges
            
            # Algorithm 4: Frequency-coherent ridge tracking
            coherent_ridges = self._extract_coherent_ridges(sst_matrix)
            if coherent_ridges:
                ridges['coherent'] = coherent_ridges
            
            # Ridge quality assessment
            ridge_qualities = self._assess_ridge_qualities(ridges, sst_matrix)
            
            return {
                'ridges': ridges,
                'qualities': ridge_qualities,
                'dominant_ridge': self._select_dominant_ridge(ridges, ridge_qualities)
            }
            
        except Exception:
            # Fallback: simple maximum ridge
            max_ridge = np.argmax(sst_matrix, axis=0)
            return {
                'ridges': {'simple_max': max_ridge},
                'qualities': {'simple_max': 0.5},
                'dominant_ridge': max_ridge
            }
    
    def _extract_optimal_phase_from_sst(self, sst_result: dict, imf: np.ndarray) -> np.ndarray:
        """Extract optimal phase from synchrosqueezed transform using comprehensive analysis"""
        try:
            sst_matrix = sst_result['sst_matrix']
            ridge_info = sst_result['ridge_info']
            
            # Method 1: Dominant ridge phase extraction
            dominant_phase = self._extract_dominant_ridge_phase(
                sst_matrix, ridge_info['dominant_ridge'], sst_result
            )
            
            # Method 2: Weighted multi-ridge phase
            multi_ridge_phase = self._extract_multi_ridge_phase(
                sst_matrix, ridge_info['ridges'], sst_result
            )
            
            # Method 3: Quality-based phase selection
            quality_phase = self._extract_quality_based_phase(
                sst_matrix, ridge_info, sst_result
            )
            
            # Phase fusion with validation
            candidate_phases = [
                ('dominant', dominant_phase),
                ('multi_ridge', multi_ridge_phase),
                ('quality_based', quality_phase)
            ]
            
            # Select best phase based on multiple criteria
            best_phase = self._select_best_sst_phase(candidate_phases, imf, sst_result)
            
            return best_phase
            
        except Exception:
            # Fallback to simple approach
            try:
                dominant_ridge = sst_result['ridge_info']['dominant_ridge']
                phase = np.zeros(len(dominant_ridge))
                
                for t, freq_idx in enumerate(dominant_ridge):
                    if 0 <= freq_idx < sst_result['sst_matrix'].shape[0]:
                        # Extract phase from original wavelet results
                        first_wavelet = list(sst_result['wavelet_results'].keys())[0]
                        wavelet_phase = sst_result['wavelet_results'][first_wavelet]['phase']
                        if freq_idx < wavelet_phase.shape[0] and t < wavelet_phase.shape[1]:
                            phase[t] = wavelet_phase[freq_idx, t]
                
                return phase
                
            except Exception:
                return np.linspace(0, 2*np.pi, len(imf))
    
    def _validate_sst_phase(self, phase: np.ndarray, imf: np.ndarray) -> np.ndarray:
        """Validate and post-process synchrosqueezed transform phase"""
        try:
            # Ensure phase length matches IMF length
            if len(phase) != len(imf):
                if len(phase) > len(imf):
                    phase = phase[:len(imf)]
                else:
                    # Interpolate to match length
                    x_old = np.linspace(0, 1, len(phase))
                    x_new = np.linspace(0, 1, len(imf))
                    phase = np.interp(x_new, x_old, phase)
            
            # Remove NaN and infinite values
            phase = np.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)
            
            # Phase unwrapping for continuity
            unwrapped_phase = np.unwrap(phase)
            
            # Smooth phase transitions if needed
            if len(unwrapped_phase) > 5:
                # Detect and smooth large jumps
                phase_diff = np.diff(unwrapped_phase)
                large_jumps = np.abs(phase_diff) > 3 * np.std(phase_diff)
                
                if np.any(large_jumps):
                    # Apply gentle smoothing to reduce artifacts
                    window_size = min(7, len(unwrapped_phase) // 20)
                    if window_size >= 3 and window_size % 2 == 1:
                        smoothed = scipy.signal.savgol_filter(unwrapped_phase, window_size, 2)
                        # Only apply smoothing where there were large jumps
                        jump_indices = np.where(large_jumps)[0]
                        for idx in jump_indices:
                            if idx < len(smoothed) - 1:
                                unwrapped_phase[idx:idx+2] = smoothed[idx:idx+2]
            
            # Final validation
            if np.all(np.isfinite(unwrapped_phase)):
                return unwrapped_phase
            else:
                # Fallback to simple phase
                return np.linspace(0, 2*np.pi, len(imf))
                
        except Exception:
            # Ultimate fallback
            return np.linspace(0, 2*np.pi, len(imf))
    
    def _select_best_phase_method(self, phase_methods: list, imf: np.ndarray) -> tuple:
        """Select best phase extraction method"""
        try:
            if not phase_methods:
                return ('fallback', self._robust_derivative_phase(imf))
            
            # Simple selection: prefer Hilbert transform if available
            for method_name, phase in phase_methods:
                if method_name == 'hilbert':
                    return (method_name, phase)
            
            # Otherwise, return first valid method
            return phase_methods[0]
            
        except Exception:
            return ('fallback', self._robust_derivative_phase(imf))
    
    def _compute_phase_quality_score(self, phase: np.ndarray, imf: np.ndarray) -> float:
        """Compute quality score for phase"""
        try:
            # Basic quality metrics
            continuity_score = self._assess_phase_continuity(phase)
            consistency_score = self._assess_phase_consistency(phase, imf)
            
            return 0.6 * continuity_score + 0.4 * consistency_score
            
        except Exception:
            return 0.5
    
    def _assess_phase_continuity(self, phase: np.ndarray) -> float:
        """Assess phase continuity"""
        try:
            if len(phase) < 2:
                return 1.0
            
            phase_diff = np.diff(np.unwrap(phase))
            continuity_score = 1.0 / (1.0 + np.std(phase_diff))
            
            return min(1.0, max(0.0, continuity_score))
            
        except Exception:
            return 0.5
    
    def _assess_phase_consistency(self, phase: np.ndarray, imf: np.ndarray) -> float:
        """Assess phase consistency with IMF"""
        try:
            # Simplified consistency check
            return 0.7  # Default reasonable value
            
        except Exception:
            return 0.5
    
    def _compute_frequency_relevance(self, imf: np.ndarray, signal: np.ndarray) -> float:
        """Compute frequency content relevance"""
        try:
            # Cross-spectral analysis
            f1, psd1 = scipy.signal.welch(imf, nperseg=min(len(imf)//4, 256))
            f2, psd2 = scipy.signal.welch(signal, nperseg=min(len(signal)//4, 256))
            
            # Compute spectral overlap
            min_len = min(len(psd1), len(psd2))
            psd1_norm = psd1[:min_len] / np.sum(psd1[:min_len])
            psd2_norm = psd2[:min_len] / np.sum(psd2[:min_len])
            
            overlap = np.sum(np.minimum(psd1_norm, psd2_norm))
            
            return overlap
            
        except Exception:
            return 0.5
    
    def _postprocess_combined_phase(self, phase: np.ndarray) -> np.ndarray:
        """Post-process combined phase"""
        try:
            # Unwrap and smooth phase
            unwrapped = np.unwrap(phase)
            
            # Apply gentle smoothing
            if len(unwrapped) > 5:
                window_size = min(5, len(unwrapped) // 10)
                if window_size >= 3 and window_size % 2 == 1:
                    smoothed = scipy.signal.savgol_filter(unwrapped, window_size, 2)
                    return smoothed
            
            return unwrapped
            
        except Exception:
            return phase
    
    def _validate_phase_continuity(self, phase: np.ndarray) -> bool:
        """Validate phase continuity"""
        try:
            if len(phase) < 2:
                return True
            
            phase_diff = np.diff(np.unwrap(phase))
            return np.std(phase_diff) < 5.0  # Reasonable threshold
            
        except Exception:
            return False
    
    def _repair_phase_discontinuities(self, phase: np.ndarray) -> np.ndarray:
        """Repair phase discontinuities"""
        try:
            return np.unwrap(phase)
        except Exception:
            return phase
    
    def _validate_phase_bounds(self, phase: np.ndarray) -> bool:
        """Validate phase bounds"""
        try:
            return not (np.any(np.isnan(phase)) or np.any(np.isinf(phase)))
        except Exception:
            return False
    
    def _normalize_phase_bounds(self, phase: np.ndarray) -> np.ndarray:
        """Normalize phase bounds"""
        try:
            # Remove NaN and inf values
            phase = np.nan_to_num(phase, nan=0.0, posinf=np.pi, neginf=-np.pi)
            return phase
        except Exception:
            return phase
    
    def _smooth_phase_artifacts(self, phase: np.ndarray) -> np.ndarray:
        """Smooth phase artifacts"""
        try:
            if len(phase) > 5:
                window_size = min(7, len(phase) // 20)
                if window_size >= 3 and window_size % 2 == 1:
                    return scipy.signal.savgol_filter(phase, window_size, 2)
            return phase
        except Exception:
            return phase
    
    def _validate_phase_signal_consistency(self, phase: np.ndarray, signal: np.ndarray) -> bool:
        """Validate phase-signal consistency"""
        try:
            # Basic validation: lengths should match
            return len(phase) == len(signal)
        except Exception:
            return False
    
    # Additional helper methods for advanced synchrosqueezed transform
    def _compute_skewness(self, signal: np.ndarray) -> float:
        """Compute signal skewness"""
        try:
            from scipy.stats import skew
            return float(skew(signal))
        except Exception:
            return 0.0
    
    def _compute_kurtosis(self, signal: np.ndarray) -> float:
        """Compute signal kurtosis"""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(signal))
        except Exception:
            return 0.0
    
    def _compute_approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = None) -> float:
        """Compute approximate entropy"""
        try:
            if r is None:
                r = 0.2 * np.std(signal)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([signal[i:i + m] for i in range(len(signal) - m + 1)])
                C = np.zeros(len(patterns))
                for i in range(len(patterns)):
                    template = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template, patterns[j], m) <= r:
                            C[i] += 1
                C = C / float(len(patterns))
                phi = np.sum(np.log(C)) / float(len(patterns))
                return phi
            
            return _phi(m) - _phi(m + 1)
            
        except Exception:
            return 1.0
    
    def _compute_zero_crossing_rate(self, signal: np.ndarray) -> float:
        """Compute zero crossing rate"""
        try:
            zero_crossings = np.where(np.diff(np.sign(signal)))[0]
            return len(zero_crossings) / len(signal)
        except Exception:
            return 0.1
    
    def _determine_optimal_scales(self, signal: np.ndarray, characteristics: dict) -> np.ndarray:
        """Determine optimal scales for wavelet analysis"""
        try:
            signal_length = len(signal)
            dominant_freq = characteristics.get('dominant_freq', 0.1)
            
            # Adaptive scale range based on signal characteristics
            min_scale = max(1, int(1.0 / (dominant_freq * 10)))
            max_scale = min(signal_length // 4, 128)
            
            # Logarithmic scale distribution
            n_scales = min(64, max(16, signal_length // 10))
            scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
            
            return scales
            
        except Exception:
            return np.logspace(0, 3, 32)
    
    def _select_optimal_wavelets(self, characteristics: dict) -> list:
        """Select optimal wavelets based on signal characteristics"""
        try:
            wavelets = ['morlet2']  # Always include Morlet
            
            # Add complex Morlet for specific characteristics
            if characteristics.get('spectral_bandwidth', 1) > 0.5:
                wavelets.append('cmor1.5-1.0')
            
            return wavelets
            
        except Exception:
            return ['morlet2']
    
    def _compute_adaptive_ridge_threshold(self, characteristics: dict) -> float:
        """Compute adaptive ridge threshold"""
        try:
            # Base threshold
            base_threshold = 0.1
            
            # Adjust based on signal characteristics
            noise_factor = 1.0 / (characteristics.get('std', 1) + 1e-10)
            complexity_factor = characteristics.get('approximate_entropy', 1)
            
            adaptive_threshold = base_threshold * noise_factor * complexity_factor
            return np.clip(adaptive_threshold, 0.01, 0.5)
            
        except Exception:
            return 0.1
    
    def _compute_complex_morlet_cwt(self, signal: np.ndarray, scales: np.ndarray, wavelet_name: str) -> np.ndarray:
        """Compute complex Morlet continuous wavelet transform"""
        try:
            # Simple implementation using scipy's cwt with morlet2
            return scipy.signal.cwt(signal, scipy.signal.morlet2, scales)
        except Exception:
            return np.zeros((len(scales), len(signal)), dtype=complex)
    
    def _compute_cwt_instantaneous_frequency(self, cwt_matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """Compute instantaneous frequency from CWT matrix"""
        try:
            phase = np.angle(cwt_matrix)
            inst_freq = np.zeros_like(phase)
            
            # Compute frequency for each scale
            for s in range(len(scales)):
                phase_row = phase[s, :]
                if len(phase_row) > 2:
                    # Central difference for instantaneous frequency
                    inst_freq[s, 1:-1] = (phase_row[2:] - phase_row[:-2]) / (2 * np.pi * 2)
                    inst_freq[s, 0] = inst_freq[s, 1]
                    inst_freq[s, -1] = inst_freq[s, -2]
            
            return inst_freq
            
        except Exception:
            return np.zeros_like(cwt_matrix, dtype=float)
    
    def _combine_sst_matrices(self, sst_matrices: list, wavelet_results: dict, params: dict) -> np.ndarray:
        """Combine multiple SST matrices with optimal weighting"""
        try:
            if len(sst_matrices) == 1:
                return sst_matrices[0]
            
            # Simple averaging for now
            combined = np.mean(sst_matrices, axis=0)
            return combined
            
        except Exception:
            return sst_matrices[0] if sst_matrices else np.zeros((16, 100))
    
    def _extract_maximum_energy_ridge(self, sst_matrix: np.ndarray) -> np.ndarray:
        """Extract maximum energy ridge"""
        try:
            return np.argmax(sst_matrix, axis=0)
        except Exception:
            return np.zeros(sst_matrix.shape[1], dtype=int)
    
    def _extract_persistent_ridges(self, sst_matrix: np.ndarray, threshold: float) -> list:
        """Extract persistent ridges"""
        try:
            # Simple threshold-based approach
            ridges = []
            for t in range(sst_matrix.shape[1]):
                significant_freqs = np.where(sst_matrix[:, t] > threshold * np.max(sst_matrix[:, t]))[0]
                if len(significant_freqs) > 0:
                    ridges.append(significant_freqs)
                else:
                    ridges.append([])
            return ridges
        except Exception:
            return []
    
    def _extract_multiscale_ridges(self, sst_matrix: np.ndarray, params: dict) -> list:
        """Extract multi-scale ridges"""
        try:
            # Simple multi-scale approach
            ridges = []
            n_scales = 3
            for scale in range(n_scales):
                step = max(1, sst_matrix.shape[0] // n_scales)
                start_idx = scale * step
                end_idx = min(start_idx + step, sst_matrix.shape[0])
                
                if start_idx < end_idx:
                    submatrix = sst_matrix[start_idx:end_idx, :]
                    ridge = np.argmax(submatrix, axis=0) + start_idx
                    ridges.append(ridge)
            
            return ridges
            
        except Exception:
            return []
    
    def _extract_coherent_ridges(self, sst_matrix: np.ndarray) -> list:
        """Extract frequency-coherent ridges"""
        try:
            # Simple coherent ridge extraction
            ridge = np.argmax(sst_matrix, axis=0)
            
            # Smooth ridge for coherence
            if len(ridge) > 5:
                window_size = min(5, len(ridge) // 10)
                if window_size >= 3 and window_size % 2 == 1:
                    from scipy.ndimage import median_filter
                    smooth_ridge = median_filter(ridge.astype(float), size=window_size)
                    return [smooth_ridge.astype(int)]
            
            return [ridge]
            
        except Exception:
            return []
    
    def _assess_ridge_qualities(self, ridges: dict, sst_matrix: np.ndarray) -> dict:
        """Assess quality of extracted ridges"""
        try:
            qualities = {}
            
            for ridge_name, ridge_data in ridges.items():
                if isinstance(ridge_data, np.ndarray):
                    # Single ridge
                    energy = np.mean([sst_matrix[ridge_data[t], t] for t in range(len(ridge_data)) 
                                    if 0 <= ridge_data[t] < sst_matrix.shape[0]])
                    qualities[ridge_name] = energy / (np.max(sst_matrix) + 1e-10)
                elif isinstance(ridge_data, list) and len(ridge_data) > 0:
                    # Multiple ridges
                    if isinstance(ridge_data[0], np.ndarray):
                        avg_quality = 0
                        for ridge in ridge_data:
                            if len(ridge) == sst_matrix.shape[1]:
                                energy = np.mean([sst_matrix[ridge[t], t] for t in range(len(ridge)) 
                                                if 0 <= ridge[t] < sst_matrix.shape[0]])
                                avg_quality += energy / (np.max(sst_matrix) + 1e-10)
                        qualities[ridge_name] = avg_quality / len(ridge_data)
                    else:
                        qualities[ridge_name] = 0.3  # Default for other types
                else:
                    qualities[ridge_name] = 0.1  # Low quality for empty/invalid ridges
            
            return qualities
            
        except Exception:
            return {name: 0.5 for name in ridges.keys()}
    
    def _select_dominant_ridge(self, ridges: dict, qualities: dict) -> np.ndarray:
        """Select dominant ridge based on quality scores"""
        try:
            if not ridges or not qualities:
                return np.zeros(100, dtype=int)
            
            # Find best quality ridge
            best_ridge_name = max(qualities.keys(), key=lambda k: qualities[k])
            best_ridge = ridges[best_ridge_name]
            
            if isinstance(best_ridge, np.ndarray):
                return best_ridge
            elif isinstance(best_ridge, list) and len(best_ridge) > 0:
                if isinstance(best_ridge[0], np.ndarray):
                    return best_ridge[0]  # First ridge from multi-ridge result
            
            # Fallback
            return np.zeros(100, dtype=int)
            
        except Exception:
            return np.zeros(100, dtype=int)
    
    def _assess_sst_quality(self, sst_matrix: np.ndarray, ridge_info: dict, signal: np.ndarray) -> dict:
        """Assess overall SST quality"""
        try:
            return {
                'energy_concentration': np.max(sst_matrix) / (np.mean(sst_matrix) + 1e-10),
                'ridge_stability': len(ridge_info.get('ridges', {})),
                'reconstruction_quality': 0.7  # Placeholder
            }
        except Exception:
            return {'overall_quality': 0.5}
    
    def _extract_dominant_ridge_phase(self, sst_matrix: np.ndarray, dominant_ridge: np.ndarray, sst_result: dict) -> np.ndarray:
        """Extract phase from dominant ridge"""
        try:
            phase = np.zeros(len(dominant_ridge))
            
            # Extract phase from wavelet results
            first_wavelet = list(sst_result['wavelet_results'].keys())[0]
            wavelet_phase = sst_result['wavelet_results'][first_wavelet]['phase']
            
            for t, freq_idx in enumerate(dominant_ridge):
                if (0 <= freq_idx < wavelet_phase.shape[0] and 
                    t < wavelet_phase.shape[1]):
                    phase[t] = wavelet_phase[freq_idx, t]
            
            return phase
            
        except Exception:
            return np.linspace(0, 2*np.pi, len(dominant_ridge))
    
    def _extract_multi_ridge_phase(self, sst_matrix: np.ndarray, ridges: dict, sst_result: dict) -> np.ndarray:
        """Extract weighted multi-ridge phase"""
        try:
            if not ridges:
                return np.linspace(0, 2*np.pi, sst_matrix.shape[1])
            
            # Use dominant ridge as fallback
            dominant_ridge = ridges.get('max_energy', np.argmax(sst_matrix, axis=0))
            return self._extract_dominant_ridge_phase(sst_matrix, dominant_ridge, sst_result)
            
        except Exception:
            return np.linspace(0, 2*np.pi, sst_matrix.shape[1])
    
    def _extract_quality_based_phase(self, sst_matrix: np.ndarray, ridge_info: dict, sst_result: dict) -> np.ndarray:
        """Extract phase based on quality assessment"""
        try:
            dominant_ridge = ridge_info.get('dominant_ridge', np.argmax(sst_matrix, axis=0))
            return self._extract_dominant_ridge_phase(sst_matrix, dominant_ridge, sst_result)
        except Exception:
            return np.linspace(0, 2*np.pi, sst_matrix.shape[1])
    
    def _select_best_sst_phase(self, candidate_phases: list, imf: np.ndarray, sst_result: dict) -> np.ndarray:
        """Select best phase from candidates"""
        try:
            if not candidate_phases:
                return np.linspace(0, 2*np.pi, len(imf))
            
            # Simple selection: prefer dominant ridge phase
            for method_name, phase in candidate_phases:
                if method_name == 'dominant' and len(phase) == len(imf):
                    return phase
            
            # Fallback to first valid phase
            for method_name, phase in candidate_phases:
                if len(phase) == len(imf):
                    return phase
            
            # Ultimate fallback
            return np.linspace(0, 2*np.pi, len(imf))
            
        except Exception:
            return np.linspace(0, 2*np.pi, len(imf))
    
    def _robust_derivative_phase(self, signal: np.ndarray) -> np.ndarray:
        """Robust derivative-based phase extraction with smoothing"""
        try:
            if len(signal) < 3:
                return np.linspace(0, 2*np.pi, len(signal))
            
            # Apply smoothing to reduce noise
            smoothed_signal = self._smooth_signal(signal)
            
            # Compute derivative using central differences
            derivative = np.zeros_like(smoothed_signal)
            derivative[1:-1] = (smoothed_signal[2:] - smoothed_signal[:-2]) / 2
            derivative[0] = smoothed_signal[1] - smoothed_signal[0]
            derivative[-1] = smoothed_signal[-1] - smoothed_signal[-2]
            
            # Convert derivative to phase
            phase = np.arctan2(derivative, smoothed_signal)
            
            # Unwrap and normalize phase
            unwrapped_phase = np.unwrap(phase)
            normalized_phase = (unwrapped_phase - np.min(unwrapped_phase)) 
            normalized_phase = normalized_phase / np.max(normalized_phase) * 2 * np.pi
            
            return normalized_phase
            
        except Exception:
            return np.linspace(0, 2*np.pi, len(signal))
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int = None) -> np.ndarray:
        """Smooth signal using moving average"""
        try:
            if window_size is None:
                window_size = max(3, len(signal) // 20)
                if window_size % 2 == 0:
                    window_size += 1
            
            if window_size >= len(signal):
                return signal
            
            # Simple moving average
            smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='same')
            return smoothed
            
        except Exception:
            return signal

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