"""
Multi-variate Anomaly Detection for Geological Data Streams

Implements advanced multi-variate anomaly detection algorithms specifically
designed for geological data with complex interdependencies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    timestamp: float
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    affected_variables: List[str]
    anomaly_type: str
    explanation: str

class DeepAutoencoderDetector(nn.Module):
    """Deep autoencoder for multi-variate anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                decoder_layers.append(nn.Linear(prev_dim, input_dim))
            else:
                decoder_layers.extend([
                    nn.Linear(prev_dim, hidden_dims[i-1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.BatchNorm1d(hidden_dims[i-1])
                ])
                prev_dim = hidden_dims[i-1]
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class VariationalAutoencoderDetector(nn.Module):
    """Variational autoencoder for probabilistic anomaly detection"""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
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

class MultivariateAnomalyDetector:
    """
    Comprehensive multi-variate anomaly detection for geological data streams
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.autoencoder = None
        self.vae = None
        self.isolation_forest = IsolationForest(
            contamination=config.get('contamination', 0.1),
            random_state=42
        )
        self.elliptic_envelope = EllipticEnvelope(
            contamination=config.get('contamination', 0.1),
            random_state=42
        )
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Geological domain knowledge
        self.geological_correlations = config.get('geological_correlations', {})
        self.variable_importance = config.get('variable_importance', {})
        
        logger.info("Initialized MultivariateAnomalyDetector")
    
    def fit(self, data: np.ndarray, variable_names: List[str]) -> None:
        """
        Fit the anomaly detection models on training data
        
        Args:
            data: Training data array of shape (n_samples, n_features)
            variable_names: Names of the geological variables
        """
        logger.info(f"Fitting anomaly detection models on data shape: {data.shape}")
        
        # Normalize data
        data_normalized = self.scaler.fit_transform(data)
        
        # Fit traditional models
        self.isolation_forest.fit(data_normalized)
        self.elliptic_envelope.fit(data_normalized)
        
        # Initialize and train deep learning models
        input_dim = data.shape[1]
        hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
        
        # Autoencoder
        self.autoencoder = DeepAutoencoderDetector(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=self.config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # VAE
        latent_dim = self.config.get('latent_dim', 8)
        self.vae = VariationalAutoencoderDetector(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Train deep models
        self._train_autoencoder(data_normalized)
        self._train_vae(data_normalized)
        
        self.variable_names = variable_names
        self.is_fitted = True
        
        logger.info("Successfully fitted all anomaly detection models")
    
    def _train_autoencoder(self, data: np.ndarray, epochs: int = 100) -> None:
        """Train the autoencoder model"""
        data_tensor = torch.FloatTensor(data).to(self.device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = self.autoencoder(data_tensor)
            loss = criterion(reconstructed, data_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Autoencoder epoch {epoch}, loss: {loss.item():.6f}")
    
    def _train_vae(self, data: np.ndarray, epochs: int = 100) -> None:
        """Train the variational autoencoder model"""
        data_tensor = torch.FloatTensor(data).to(self.device)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        
        self.vae.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            recon_x, mu, logvar = self.vae(data_tensor)
            
            # VAE loss
            recon_loss = nn.functional.mse_loss(recon_x, data_tensor, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kld_loss
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"VAE epoch {epoch}, loss: {loss.item():.6f}")
    
    def detect_anomalies(self, data: np.ndarray, timestamps: np.ndarray) -> List[AnomalyResult]:
        """
        Detect anomalies in new data using ensemble of methods
        
        Args:
            data: Data array of shape (n_samples, n_features)
            timestamps: Corresponding timestamps
            
        Returns:
            List of anomaly detection results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        logger.info(f"Detecting anomalies in data shape: {data.shape}")
        
        # Normalize data
        data_normalized = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_normalized).to(self.device)
        
        results = []
        
        for i, (sample, timestamp) in enumerate(zip(data_normalized, timestamps)):
            # Get anomaly scores from different methods
            scores = self._compute_anomaly_scores(sample.reshape(1, -1), data_tensor[i:i+1])
            
            # Ensemble scoring
            ensemble_score = self._compute_ensemble_score(scores)
            
            # Determine if anomaly
            is_anomaly = ensemble_score > self.config.get('anomaly_threshold', 0.7)
            
            # Identify affected variables
            affected_vars = self._identify_affected_variables(sample, scores)
            
            # Classify anomaly type
            anomaly_type = self._classify_anomaly_type(scores, affected_vars)
            
            # Generate explanation
            explanation = self._generate_explanation(scores, affected_vars, anomaly_type)
            
            result = AnomalyResult(
                timestamp=timestamp,
                anomaly_score=ensemble_score,
                is_anomaly=is_anomaly,
                confidence=self._compute_confidence(scores),
                affected_variables=affected_vars,
                anomaly_type=anomaly_type,
                explanation=explanation
            )
            
            results.append(result)
        
        logger.info(f"Detected {sum(r.is_anomaly for r in results)} anomalies out of {len(results)} samples")
        return results
    
    def _compute_anomaly_scores(self, sample: np.ndarray, sample_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute anomaly scores using different methods"""
        scores = {}
        
        # Isolation Forest
        scores['isolation_forest'] = -self.isolation_forest.decision_function(sample)[0]
        
        # Elliptic Envelope
        scores['elliptic_envelope'] = -self.elliptic_envelope.decision_function(sample)[0]
        
        # Autoencoder reconstruction error
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed, _ = self.autoencoder(sample_tensor)
            recon_error = torch.mean((sample_tensor - reconstructed) ** 2).item()
            scores['autoencoder'] = recon_error
        
        # VAE reconstruction probability
        self.vae.eval()
        with torch.no_grad():
            recon_x, mu, logvar = self.vae(sample_tensor)
            recon_loss = torch.mean((sample_tensor - recon_x) ** 2).item()
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()
            scores['vae'] = recon_loss + kld_loss
        
        return scores
    
    def _compute_ensemble_score(self, scores: Dict[str, float]) -> float:
        """Compute ensemble anomaly score"""
        # Normalize scores to [0, 1] range
        normalized_scores = {}
        for method, score in scores.items():
            if method in ['isolation_forest', 'elliptic_envelope']:
                # These are already normalized decision functions
                normalized_scores[method] = max(0, min(1, (score + 1) / 2))
            else:
                # Reconstruction errors - use sigmoid normalization
                normalized_scores[method] = 1 / (1 + np.exp(-score))
        
        # Weighted ensemble
        weights = self.config.get('ensemble_weights', {
            'isolation_forest': 0.25,
            'elliptic_envelope': 0.25,
            'autoencoder': 0.25,
            'vae': 0.25
        })
        
        ensemble_score = sum(weights.get(method, 0.25) * score 
                           for method, score in normalized_scores.items())
        
        return ensemble_score
    
    def _identify_affected_variables(self, sample: np.ndarray, scores: Dict[str, float]) -> List[str]:
        """Identify which geological variables contribute most to the anomaly"""
        affected_vars = []
        
        # Use autoencoder feature importance
        if self.autoencoder is not None:
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}
 torch.FloatTensor(sample.reshape(1, -1)).to(self.device)
            self.autoencoder.eval()
            
            with torch.no_grad():
                reconstructed, _ = self.autoencoder(sample_tensor)
                feature_errors = torch.abs(sample_tensor - reconstructed).squeeze().cpu().numpy()
                
                # Identify variables with highest reconstruction errors
                error_threshold = np.percentile(feature_errors, 75)
                anomalous_indices = np.where(feature_errors > error_threshold)[0]
                
                for idx in anomalous_indices:
                    if idx < len(self.variable_names):
                        affected_vars.append(self.variable_names[idx])
        
        return affected_vars
    
    def _classify_anomaly_type(self, scores: Dict[str, float], affected_vars: List[str]) -> str:
        """Classify the type of anomaly based on scores and affected variables"""
        # Determine dominant detection method
        max_method = max(scores.items(), key=lambda x: x[1])[0]
        
        if max_method == 'isolation_forest':
            return "point_anomaly"
        elif max_method == 'elliptic_envelope':
            return "distributional_anomaly"
        elif max_method in ['autoencoder', 'vae']:
            if len(affected_vars) > len(self.variable_names) * 0.5:
                return "global_anomaly"
            else:
                return "local_anomaly"
        
        return "unknown_anomaly"
    
    def _generate_explanation(self, scores: Dict[str, float], affected_vars: List[str], 
                            anomaly_type: str) -> str:
        """Generate human-readable explanation for the anomaly"""
        explanations = []
        
        if affected_vars:
            var_str = ", ".join(affected_vars[:3])  # Limit to top 3 variables
            explanations.append(f"Unusual patterns detected in: {var_str}")
        
        if anomaly_type == "point_anomaly":
            explanations.append("This appears to be an isolated unusual measurement")
        elif anomaly_type == "distributional_anomaly":
            explanations.append("The data distribution differs significantly from normal patterns")
        elif anomaly_type == "global_anomaly":
            explanations.append("Multiple geological parameters show coordinated unusual behavior")
        elif anomaly_type == "local_anomaly":
            explanations.append("Specific geological parameters show unusual behavior")
        
        # Add geological context if available
        if self.geological_correlations and affected_vars:
            for var in affected_vars[:2]:  # Check top 2 variables
                if var in self.geological_correlations:
                    correlations = self.geological_correlations[var]
                    explanations.append(f"This may indicate changes in {correlations}")
        
        return ". ".join(explanations) if explanations else "Anomalous pattern detected"
    
    def _compute_confidence(self, scores: Dict[str, float]) -> float:
        """Compute confidence in the anomaly detection"""
        # Higher confidence when multiple methods agree
        normalized_scores = [max(0, min(1, score)) for score in scores.values()]
        score_variance = np.var(normalized_scores)
        
        # Lower variance means higher agreement, thus higher confidence
        confidence = 1.0 - min(1.0, score_variance * 4)  # Scale variance to [0, 1]
        
        return confidence
    
    def update_geological_knowledge(self, correlations: Dict[str, str], 
                                  importance: Dict[str, float]) -> None:
        """Update geological domain knowledge"""
        self.geological_correlations.update(correlations)
        self.variable_importance.update(importance)
        logger.info("Updated geological domain knowledge")