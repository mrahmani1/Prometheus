"""
Multivariate anomaly detection for geological data streams

This module implements advanced multivariate anomaly detection algorithms
specifically designed for geological data with multiple sensors and variables.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD

from core.config import Settings
from core.database import DatabaseManager
from models.schemas import (
    GeologicalDataStream, DetectedAnomaly, AnomalyType, AnomalySeverity
)

logger = logging.getLogger(__name__)


class GeologicalVAE(nn.Module):
    """Variational Autoencoder specifically designed for geological data"""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = None):
        super(GeologicalVAE, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
    
    def get_reconstruction_error(self, x):
        """Get reconstruction error for anomaly detection"""
        with torch.no_grad():
            recon_x, _, _ = self.forward(x)
            mse = torch.mean((x - recon_x) ** 2, dim=1)
            return mse.cpu().numpy()


class EnsembleAnomalyDetector:
    """Ensemble of multiple anomaly detection algorithms"""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.detectors = {}
        self.scalers = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """Fit ensemble of detectors"""
        # Isolation Forest
        self.detectors['isolation_forest'] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=200
        )
        
        # Elliptic Envelope (robust covariance)
        self.detectors['elliptic_envelope'] = EllipticEnvelope(
            contamination=self.contamination,
            random_state=42
        )
        
        # COPOD (Copula-based outlier detection)
        self.detectors['copod'] = COPOD(contamination=self.contamination)
        
        # ECOD (Empirical Cumulative Distribution)
        self.detectors['ecod'] = ECOD(contamination=self.contamination)
        
        # Fit scalers and detectors
        for name, detector in self.detectors.items():
            # Use different scalers for different methods
            if name in ['elliptic_envelope']:
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            
            X_scaled = scaler.fit_transform(X)
            self.scalers[name] = scaler
            
            detector.fit(X_scaled)
        
        self.is_fitted = True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities using ensemble voting"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = []
        
        for name, detector in self.detectors.items():
            X_scaled = self.scalers[name].transform(X)
            
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(X_scaled)
                # Convert to probabilities (higher score = more normal)
                probs = 1 / (1 + np.exp(-scores))
                predictions.append(1 - probs)  # Convert to anomaly probability
            else:
                # For detectors that only have predict
                binary_pred = detector.predict(X_scaled)
                predictions.append(binary_pred.astype(float))
        
        # Ensemble voting (average)
        ensemble_probs = np.mean(predictions, axis=0)
        return ensemble_probs


class MultivariateAnomalyDetector:
    """Advanced multivariate anomaly detector for geological data"""
    
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager
        
        # Models
        self.vae_model: Optional[GeologicalVAE] = None
        self.ensemble_detector: Optional[EnsembleAnomalyDetector] = None
        self.deep_svdd: Optional[DeepSVDD] = None
        
        # Data preprocessing
        self.feature_scaler: Optional[StandardScaler] = None
        self.pca_transformer: Optional[PCA] = None
        
        # Model state
        self.is_trained = False
        self.feature_names: List[str] = []
        self.baseline_statistics: Dict[str, Any] = {}
        
        # Device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def initialize(self):
        """Initialize the multivariate detector"""
        try:
            logger.info("Initializing multivariate anomaly detector...")
            
            # Load or train models
            await self._load_or_train_models()
            
            logger.info("Multivariate anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize multivariate detector: {e}")
            raise
    
    async def detect_anomalies(
        self, 
        data_stream: GeologicalDataStream,
        sensitivity: float
    ) -> List[DetectedAnomaly]:
        """Detect multivariate anomalies in geological data stream"""
        try:
            # Prepare data
            features_df = self._prepare_features(data_stream)
            if features_df.empty:
                return []
            
            # Get anomaly scores from different methods
            anomaly_scores = await self._compute_anomaly_scores(features_df)
            
            # Combine scores and identify anomalies
            detected_anomalies = await self._identify_anomalies(
                features_df, anomaly_scores, data_stream, sensitivity
            )
            
            return detected_anomalies
            
        except Exception as e:
            logger.error(f"Multivariate anomaly detection failed: {e}")
            return []
    
    def _prepare_features(self, data_stream: GeologicalDataStream) -> pd.DataFrame:
        """Prepare feature matrix from geological data stream"""
        try:
            # Convert data points to DataFrame
            data_records = []
            for point in data_stream.data_points:
                record = {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'quality_score': point.quality_score,
                    'sensor_type': point.sensor_type,
                    'measurement_type': point.measurement_type,
                    'latitude': point.location.get('lat', 0.0),
                    'longitude': point.location.get('lon', 0.0),
                    'elevation': point.location.get('elevation', 0.0),
                    'depth': point.location.get('depth', 0.0)
                }
                data_records.append(record)
            
            df = pd.DataFrame(data_records)
            if df.empty:
                return df
            
            # Create time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            
            # Create rolling statistics
            window_size = min(10, len(df))
            if window_size > 1:
                df['value_rolling_mean'] = df['value'].rolling(window=window_size).mean()
                df['value_rolling_std'] = df['value'].rolling(window=window_size).std()
                df['value_rolling_min'] = df['value'].rolling(window=window_size).min()
                df['value_rolling_max'] = df['value'].rolling(window=window_size).max()
            else:
                df['value_rolling_mean'] = df['value']
                df['value_rolling_std'] = 0.0
                df['value_rolling_min'] = df['value']
                df['value_rolling_max'] = df['value']
            
            # Create spatial features
            if len(df) > 1:
                df['spatial_distance'] = np.sqrt(
                    (df['latitude'] - df['latitude'].mean()) ** 2 +
                    (df['longitude'] - df['longitude'].mean()) ** 2
                )
            else:
                df['spatial_distance'] = 0.0
            
            # One-hot encode categorical variables
            sensor_dummies = pd.get_dummies(df['sensor_type'], prefix='sensor')
            measurement_dummies = pd.get_dummies(df['measurement_type'], prefix='measurement')
            
            # Combine all features
            feature_columns = [
                'value', 'quality_score', 'latitude', 'longitude', 'elevation', 'depth',
                'hour', 'day_of_week', 'month',
                'value_rolling_mean', 'value_rolling_std', 'value_rolling_min', 'value_rolling_max',
                'spatial_distance'
            ]
            
            features_df = df[feature_columns].copy()
            features_df = pd.concat([features_df, sensor_dummies, measurement_dummies], axis=1)
            
            # Fill NaN values
            features_df = features_df.fillna(features_df.mean())
            
            # Store feature names for later use
            self.feature_names = features_df.columns.tolist()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    async def _compute_anomaly_scores(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Compute anomaly scores using multiple methods"""
        scores = {}
        
        try:
            features_array = features_df.values
            
            # VAE-based anomaly detection
            if self.vae_model is not None:
                vae_scores = await self._compute_vae_scores(features_array)
                scores['vae'] = vae_scores
            
            # Ensemble detector scores
            if self.ensemble_detector is not None and self.ensemble_detector.is_fitted:
                ensemble_scores = self.ensemble_detector.predict_proba(features_array)
                scores['ensemble'] = ensemble_scores
            
            # Deep SVDD scores
            if self.deep_svdd is not None:
                try:
                    svdd_scores = self.deep_svdd.decision_function(features_array)
                    # Convert to probabilities
                    svdd_probs = 1 / (1 + np.exp(-svdd_scores))
                    scores['deep_svdd'] = 1 - svdd_probs
                except Exception as e:
                    logger.warning(f"Deep SVDD scoring failed: {e}")
            
            # Statistical anomaly detection
            statistical_scores = self._compute_statistical_scores(features_array)
            scores['statistical'] = statistical_scores
            
        except Exception as e:
            logger.error(f"Anomaly score computation failed: {e}")
        
        return scores
    
    async def _compute_vae_scores(self, features_array: np.ndarray) -> np.ndarray:
        """Compute VAE reconstruction error scores"""
        try:
            if self.feature_scaler is not None:
                features_scaled = self.feature_scaler.transform(features_array)
            else:
                features_scaled = features_array
            
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            
            with torch.no_grad():
                reconstruction_errors = self.vae_model.get_reconstruction_error(features_tensor)
            
            # Normalize scores to [0, 1]
            min_error = np.min(reconstruction_errors)
            max_error = np.max(reconstruction_errors)
            if max_error > min_error:
                normalized_scores = (reconstruction_errors - min_error) / (max_error - min_error)
            else:
                normalized_scores = np.zeros_like(reconstruction_errors)
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"VAE scoring failed: {e}")
            return np.zeros(len(features_array))
    
    def _compute_statistical_scores(self, features_array: np.ndarray) -> np.ndarray:
        """Compute statistical anomaly scores using Mahalanobis distance"""
        try:
            if len(self.baseline_statistics) == 0:
                # Use current data to compute baseline if not available
                mean = np.mean(features_array, axis=0)
                cov = np.cov(features_array.T)
            else:
                mean = self.baseline_statistics['mean']
                cov = self.baseline_statistics['covariance']
            
            # Compute Mahalanobis distance
            try:
                cov_inv = np.linalg.pinv(cov)
                diff = features_array - mean
                mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
                
                # Convert to probabilities using chi-square distribution
                from scipy.stats import chi2
                p_values = 1 - chi2.cdf(mahal_dist ** 2, df=features_array.shape[1])
                
                return p_values
                
            except np.linalg.LinAlgError:
                # Fallback to Euclidean distance if covariance is singular
                diff = features_array - mean
                euclidean_dist = np.sqrt(np.sum(diff ** 2, axis=1))
                normalized_dist = euclidean_dist / (np.max(euclidean_dist) + 1e-8)
                return normalized_dist
            
        except Exception as e:
            logger.error(f"Statistical scoring failed: {e}")
            return np.zeros(len(features_array))
    
    async def _identify_anomalies(
        self,
        features_df: pd.DataFrame,
        anomaly_scores: Dict[str, np.ndarray],
        data_stream: GeologicalDataStream,
        sensitivity: float
    ) -> List[DetectedAnomaly]:
        """Identify anomalies based on combined scores"""
        detected_anomalies = []
        
        try:
            if not anomaly_scores:
                return detected_anomalies
            
            # Combine scores using weighted average
            weights = {
                'vae': 0.3,
                'ensemble': 0.3,
                'deep_svdd': 0.2,
                'statistical': 0.2
            }
            
            combined_scores = np.zeros(len(features_df))
            total_weight = 0
            
            for method, scores in anomaly_scores.items():
                if method in weights and len(scores) == len(features_df):
                    combined_scores += weights[method] * scores
                    total_weight += weights[method]
            
            if total_weight > 0:
                combined_scores /= total_weight
            
            # Identify anomalies based on threshold
            threshold = 1 - sensitivity  # Higher sensitivity = lower threshold
            anomaly_indices = np.where(combined_scores > threshold)[0]
            
            # Create DetectedAnomaly objects
            for idx in anomaly_indices:
                anomaly_id = str(uuid.uuid4())
                data_point = data_stream.data_points[idx]
                
                # Determine severity based on score
                score = combined_scores[idx]
                if score > 0.95:
                    severity = AnomalySeverity.CRITICAL
                elif score > 0.85:
                    severity = AnomalySeverity.HIGH
                elif score > 0.75:
                    severity = AnomalySeverity.MEDIUM
                else:
                    severity = AnomalySeverity.LOW
                
                # Calculate deviation magnitude
                if len(self.baseline_statistics) > 0:
                    baseline_mean = self.baseline_statistics.get('feature_means', {})
                    current_value = data_point.value
                    expected_value = baseline_mean.get(data_point.measurement_type, current_value)
                    deviation_magnitude = abs(current_value - expected_value) / (abs(expected_value) + 1e-8)
                else:
                    deviation_magnitude = score
                
                # Create baseline comparison
                baseline_comparison = {}
                for method, scores in anomaly_scores.items():
                    if len(scores) > idx:
                        baseline_comparison[f"{method}_score"] = float(scores[idx])
                
                detected_anomaly = DetectedAnomaly(
                    anomaly_id=anomaly_id,
                    detection_timestamp=datetime.utcnow(),
                    anomaly_type=AnomalyType.POINT,
                    severity=severity,
                    temporal_extent={
                        "start": data_point.timestamp,
                        "end": data_point.timestamp
                    },
                    spatial_extent=data_point.location,
                    affected_variables=[data_point.measurement_type],
                    anomaly_score=float(score),
                    deviation_magnitude=float(deviation_magnitude),
                    baseline_comparison=baseline_comparison,
                    geological_context=f"Multivariate anomaly in {data_point.sensor_type} {data_point.measurement_type}",
                    detection_method="multivariate_ensemble",
                    model_version="1.0.0",
                    processing_time=0.0  # Will be updated by caller
                )
                
                detected_anomalies.append(detected_anomaly)
            
        except Exception as e:
            logger.error(f"Anomaly identification failed: {e}")
        
        return detected_anomalies
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        try:
            # For now, we'll initialize with default parameters
            # In production, this would load from saved models or train on historical data
            
            # Initialize VAE
            input_dim = 20  # Estimated feature dimension
            self.vae_model = GeologicalVAE(input_dim=input_dim, latent_dim=8)
            self.vae_model.to(self.device)
            
            # Initialize ensemble detector
            self.ensemble_detector = EnsembleAnomalyDetector(
                contamination=1 - self.settings.anomaly_threshold
            )
            
            # Initialize Deep SVDD
            self.deep_svdd = DeepSVDD(
                contamination=1 - self.settings.anomaly_threshold,
                random_state=42
            )
            
            # Initialize scalers
            self.feature_scaler = StandardScaler()
            self.pca_transformer = PCA(n_components=0.95)  # Keep 95% of variance
            
            # Load baseline statistics if available
            await self._load_baseline_statistics()
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    async def _load_baseline_statistics(self):
        """Load baseline statistics from historical data"""
        try:
            # Query recent historical data for baseline
            end_time = datetime.utcnow()
            start_time = end_time - pd.Timedelta(days=30)  # Last 30 days
            
            historical_data = await self.db_manager.query_geological_data(
                measurement_type="all",
                start_time=start_time,
                end_time=end_time
            )
            
            if historical_data:
                df = pd.DataFrame(historical_data)
                
                # Compute baseline statistics
                self.baseline_statistics = {
                    'mean': df.select_dtypes(include=[np.number]).mean().values,
                    'covariance': df.select_dtypes(include=[np.number]).cov().values,
                    'feature_means': df.groupby('measurement_type')['value'].mean().to_dict(),
                    'feature_stds': df.groupby('measurement_type')['value'].std().to_dict()
                }
                
                logger.info("Baseline statistics loaded from historical data")
            else:
                logger.warning("No historical data available for baseline statistics")
                
        except Exception as e:
            logger.warning(f"Failed to load baseline statistics: {e}")
    
    async def update_models(self, new_data: pd.DataFrame):
        """Update models with new training data"""
        try:
            if new_data.empty:
                return
            
            # Prepare features
            features_array = new_data.values
            
            # Update ensemble detector
            if self.ensemble_detector is not None:
                self.ensemble_detector.fit(features_array)
                logger.info("Ensemble detector updated")
            
            # Update Deep SVDD
            if self.deep_svdd is not None:
                self.deep_svdd.fit(features_array)
                logger.info("Deep SVDD updated")
            
            # Update VAE (would require more complex training loop)
            # For now, we'll skip VAE retraining
            
            # Update baseline statistics
            self.baseline_statistics['mean'] = np.mean(features_array, axis=0)
            self.baseline_statistics['covariance'] = np.cov(features_array.T)
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Model update failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current models"""
        return {
            "is_trained": self.is_trained,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "models": {
                "vae": self.vae_model is not None,
                "ensemble": self.ensemble_detector is not None and self.ensemble_detector.is_fitted,
                "deep_svdd": self.deep_svdd is not None,
            },
            "baseline_available": len(self.baseline_statistics) > 0
        }