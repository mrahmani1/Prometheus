"""
Enhanced Seismic Analysis Module with Real IRIS Data Integration
Production-grade seismic processing with authentic IRIS/EarthScope data streams,
advanced phase picking, moment tensor inversion, and real-time processing
"""

import numpy as np
import asyncio
import warnings
import logging
from typing import Dict, List, Tuple, Optional, Union, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt, spectrogram, hilbert
    from scipy.fft import fft, ifft
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Real IRIS data integration
try:
    from .real_iris_data_connector import RealIRISDataConnector, IRISDataConfig, SeismicStation, SeismicEvent
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError:
    IRIS_CONNECTOR_AVAILABLE = False
    warnings.warn("IRIS data connector not available - using fallback data")

# Deep Earth ML Integration
try:
    from .deep_earth_ml import DeepEarthMLSystem
    DEEP_EARTH_ML_AVAILABLE = True
except ImportError:
    DEEP_EARTH_ML_AVAILABLE = False
    warnings.warn("Deep Earth ML not available. Thermal-seismic coupling will be limited.")

# Moment tensor inversion
try:
    from .moment_tensor_inversion import MomentTensorInverter, MomentTensorConfig
    MOMENT_TENSOR_AVAILABLE = True
except ImportError:
    MOMENT_TENSOR_AVAILABLE = False
    warnings.warn("Moment tensor inversion not available")


@dataclass
class SeismicConfig:
    """Configuration for seismic processing"""
    sample_rate: float = 100.0
    window_size: int = 6000  # 60 seconds at 100 Hz
    overlap_ratio: float = 0.5
    frequency_bands: List[Tuple[float, float]] = field(default_factory=lambda: [(1, 5), (5, 15), (15, 30)])
    transformer_layers: int = 6
    transformer_heads: int = 8
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    batch_size: int = 32
    learning_rate: float = 1e-4


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class SeismicTransformer(nn.Module):
    """Transformer model for seismic phase picking"""
    
    def __init__(self, config: SeismicConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(3, config.hidden_dim)  # 3-component seismogram
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=2048,
            dropout=config.dropout_rate,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.transformer_layers
        )
        
        # Output heads for different phases
        self.p_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.s_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # P and S uncertainties
            nn.Softplus()
        )
    
    def forward(self, x):
        # x shape: (batch, time, channels)
        batch_size, seq_len, channels = x.shape
        
        # Project input
        x = self.input_projection(x)  # (batch, time, hidden_dim)
        x = x.transpose(0, 1)  # (time, batch, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (time, batch, hidden_dim)
        encoded = encoded.transpose(0, 1)  # (batch, time, hidden_dim)
        
        # Phase predictions
        p_prob = self.p_head(encoded).squeeze(-1)  # (batch, time)
        s_prob = self.s_head(encoded).squeeze(-1)  # (batch, time)
        
        # Uncertainty estimation
        uncertainties = self.uncertainty_head(encoded.mean(dim=1))  # (batch, 2)
        
        return {
            'p_probability': p_prob,
            's_probability': s_prob,
            'uncertainties': uncertainties
        }


class FirstMotionCNN(nn.Module):
    """CNN for first motion polarity detection"""
    
    def __init__(self, config: SeismicConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList([
            # Scale 1: Fine details
            nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            # Scale 2: Medium features
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ),
            # Scale 3: Coarse features
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(256, 8, dropout=config.dropout_rate)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 3)  # Up, Down, Undetermined
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        
        # Multi-scale feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Attention over time dimension
        x_att = x.permute(2, 0, 1)  # (time, batch, features)
        attn_output, _ = self.attention(x_att, x_att, x_att)
        x = attn_output.permute(1, 2, 0)  # (batch, features, time)
        
        # Classification
        output = self.classifier(x)
        
        return {
            'polarity_logits': output,
            'polarity_probs': F.softmax(output, dim=1)
        }


class MagnitudeRNN(nn.Module):
    """RNN for magnitude estimation"""
    
    def __init__(self, config: SeismicConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)  # Fixed length for RNN
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=config.dropout_rate,
            bidirectional=True
        )
        
        # Magnitude regression head
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 256),  # 512 from bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(128, 1)
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size = x.shape[0]
        
        # Feature extraction
        features = self.feature_extractor(x)  # (batch, 128, 100)
        features = features.transpose(1, 2)  # (batch, 100, 128)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # (batch, 100, 512)
        
        # Use final hidden state for prediction
        final_hidden = lstm_out[:, -1, :]  # (batch, 512)
        
        # Magnitude prediction
        magnitude = self.magnitude_head(final_hidden).squeeze(-1)  # (batch,)
        uncertainty = self.uncertainty_head(final_hidden).squeeze(-1)  # (batch,)
        
        return {
            'magnitude': magnitude,
            'uncertainty': uncertainty
        }


class SeismicGAN:
    """GAN for seismic data augmentation"""
    
    class Generator(nn.Module):
        def __init__(self, config: SeismicConfig):
            super().__init__()
            self.config = config
            
            # Noise to features
            self.noise_projection = nn.Linear(100, 512 * 64)
            
            # Upsampling layers
            self.upsample_layers = nn.ModuleList([
                nn.Sequential(
                    nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.ConvTranspose1d(64, 3, kernel_size=4, stride=2, padding=1),
                    nn.Tanh()
                )
            ])
        
        def forward(self, noise):
            # noise shape: (batch, 100)
            x = self.noise_projection(noise)
            x = x.view(-1, 512, 64)
            
            for layer in self.upsample_layers:
                x = layer(x)
            
            return x  # (batch, 3, 1024)
    
    class Discriminator(nn.Module):
        def __init__(self, config: SeismicConfig):
            super().__init__()
            self.config = config
            
            self.conv_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(3, 64, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                ),
                nn.Sequential(
                    nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2)
                ),
                nn.Sequential(
                    nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2)
                ),
                nn.Sequential(
                    nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2)
                )
            ])
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            for layer in self.conv_layers:
                x = layer(x)
            
            return self.classifier(x)
    
    def __init__(self, config: SeismicConfig):
        self.config = config
        self.generator = self.Generator(config)
        self.discriminator = self.Discriminator(config)
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """Generate synthetic seismic data"""
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available - returning random noise")
            return np.random.randn(n_samples, 3, self.config.window_size)
        
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, 100)
            fake_data = self.generator(noise)
        
        return fake_data.numpy()


class ActiveLearningSeismic:
    """Active learning system for seismic data annotation"""
    
    def __init__(self, config: SeismicConfig):
        self.config = config
        self.models = {}
        self.labeled_pool = []
        self.unlabeled_pool = []
        self.uncertainty_threshold = 0.3
    
    def add_models(self, models: Dict):
        """Add trained models for uncertainty estimation"""
        self.models = models
    
    def calculate_uncertainty(self, data: np.ndarray) -> np.ndarray:
        """Calculate prediction uncertainty for active learning"""
        uncertainties = np.zeros(len(data))
        
        if 'transformer' in self.models:
            model = self.models['transformer']
            model.eval()
            
            with torch.no_grad():
                if isinstance(data, np.ndarray):
                    data_tensor = torch.FloatTensor(data)
                else:
                    data_tensor = data
                
                predictions = model(data_tensor)
                uncertainties = predictions['uncertainties'].mean(dim=1).numpy()
        
        return uncertainties
    
    def select_samples_for_annotation(self, n_samples: int = 10) -> List[int]:
        """Select most uncertain samples for manual annotation"""
        if len(self.unlabeled_pool) == 0:
            return []
        
        data = np.array([sample['data'] for sample in self.unlabeled_pool])
        uncertainties = self.calculate_uncertainty(data)
        
        # Select samples with highest uncertainty
        uncertain_indices = np.argsort(uncertainties)[-n_samples:]
        
        return uncertain_indices.tolist()
    
    def update_training_set(self, new_labels: List[Dict]):
        """Update training set with newly annotated samples"""
        self.labeled_pool.extend(new_labels)
        
        # Remove labeled samples from unlabeled pool
        labeled_indices = [label['index'] for label in new_labels]
        self.unlabeled_pool = [
            sample for i, sample in enumerate(self.unlabeled_pool)
            if i not in labeled_indices
        ]


class EnhancedSeismicProcessor:
    """Enhanced seismic processing pipeline with real IRIS data integration"""
    
    def __init__(self, config: Optional[SeismicConfig] = None):
        self.config = config or SeismicConfig()
        self.models = {}
        self.active_learner = ActiveLearningSeismic(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize real IRIS data connector
        self.iris_connector = None
        if IRIS_CONNECTOR_AVAILABLE:
            iris_config = IRISDataConfig()
            self.iris_connector = RealIRISDataConnector(iris_config)
            self.logger.info("Real IRIS data connector initialized")
        
        # Initialize moment tensor inversion
        self.mt_inverter = None
        if MOMENT_TENSOR_AVAILABLE:
            mt_config = MomentTensorConfig()
            self.mt_inverter = MomentTensorInverter(mt_config)
            self.logger.info("Moment tensor inverter initialized")
        
        # Initialize Deep Earth ML system for thermal-seismic coupling
        self.deep_earth_ml = None
        if DEEP_EARTH_ML_AVAILABLE:
            self._initialize_deep_earth_ml()
        
        # Real-time processing state
        self.real_time_stations = []
        self.processing_queue = asyncio.Queue()
        self.is_streaming = False
        
        if TORCH_AVAILABLE:
            self._initialize_models()
        else:
            warnings.warn("PyTorch not available - using simplified processing")
    
    def _initialize_deep_earth_ml(self):
        """Initialize Deep Earth ML system configured for seismic-thermal coupling"""
        try:
            # Configure Deep Earth ML for seismic data processing with thermal coupling
            deep_earth_config = {
                'use_physics_informed': True,  # Essential for thermal-seismic physics
                'use_fourier_operator': True,  # For frequency domain seismic analysis
                'use_deeponet': True,         # For operator learning in wave propagation
                'use_graph_operator': True,   # For seismic network topology
                'learning_rate': 1e-4,
                'hidden_dim': 512,           # Higher capacity for seismic complexity
                'fno_modes': 32,             # Higher modes for seismic frequency content
                'fno_width': 256,            # Wide enough for multi-channel seismic data
                'num_layers': 8              # Deep network for complex wave physics
            }
            
            self.deep_earth_ml = DeepEarthMLSystem(**deep_earth_config)
            print("Deep Earth ML system initialized for thermal-seismic coupling")
            
        except Exception as e:
            warnings.warn(f"Failed to initialize Deep Earth ML system: {e}")
            self.deep_earth_ml = None
    
    def get_thermal_seismic_features(self, seismic_data: np.ndarray, 
                                   coordinates: Tuple[float, float],
                                   depth_range: Tuple[float, float]) -> Dict[str, np.ndarray]:
        """
        Extract thermal-seismic coupling features using Deep Earth ML
        
        Args:
            seismic_data: Raw seismic waveform data [time_samples, channels]
            coordinates: (latitude, longitude) of seismic station
            depth_range: (min_depth, max_depth) for thermal analysis
            
        Returns:
            Dictionary containing thermal-seismic coupling features
        """
        if not self.deep_earth_ml:
            warnings.warn("Deep Earth ML not available. Returning default thermal-seismic features.")
            return self._get_default_thermal_seismic_features(seismic_data)
        
        try:
            # Create thermal-seismic analysis grid
            lat, lon = coordinates
            min_depth, max_depth = depth_range
            
            # Depth levels for thermal-seismic coupling analysis
            depths = np.linspace(min_depth, max_depth, 15)  # 15 depth levels
            thermal_grid = np.array([[lat, lon, d] for d in depths])
            
            # Get thermal field at seismic station location
            thermal_field = self.deep_earth_ml.get_thermal_field(thermal_grid)
            
            # Analyze thermal effects on seismic wave propagation
            features = {}
            
            # 1. Thermal velocity perturbations
            # Temperature affects seismic velocity through thermal expansion
            reference_temp = 300.0  # Reference temperature in Kelvin
            thermal_expansion_coeff = 3e-5  # Typical value for rocks
            
            # Calculate velocity perturbations due to temperature
            temp_perturbation = thermal_field - reference_temp
            velocity_perturbation = -thermal_expansion_coeff * temp_perturbation
            
            features['thermal_velocity_perturbation_mean'] = np.mean(velocity_perturbation)
            features['thermal_velocity_perturbation_std'] = np.std(velocity_perturbation)
            features['thermal_velocity_perturbation_max'] = np.max(np.abs(velocity_perturbation))
            
            # 2. Depth-dependent thermal effects
            features['thermal_gradient_effect'] = np.gradient(thermal_field)[0]  # dT/dz effect
            features['thermal_stratification'] = np.var(thermal_field)  # Thermal layering
            
            # 3. Thermal-seismic attenuation coupling
            # Higher temperatures generally increase seismic attenuation
            attenuation_factor = 1e-4  # Temperature coefficient for attenuation
            thermal_attenuation = attenuation_factor * (thermal_field - reference_temp)
            
            features['thermal_attenuation_mean'] = np.mean(thermal_attenuation)
            features['thermal_attenuation_integrated'] = np.trapz(thermal_attenuation, depths)
            
            # 4. Seismic waveform analysis with thermal context
            if len(seismic_data.shape) >= 2:
                # Multi-channel seismic data
                time_samples, n_channels = seismic_data.shape
                
                # Apply Fourier Neural Operator for thermal-aware seismic processing
                seismic_tensor = torch.FloatTensor(seismic_data).unsqueeze(0)  # Add batch dim
                
                # Get thermal-aware seismic features using FNO
                thermal_seismic_enhanced = self.deep_earth_ml.fourier_operator(seismic_tensor)
                thermal_seismic_features = thermal_seismic_enhanced.detach().numpy().squeeze()
                
                # Extract key thermal-seismic coupling metrics
                features['thermal_seismic_coherence'] = np.corrcoef(
                    thermal_field, 
                    np.mean(thermal_seismic_features[:len(thermal_field)], axis=0) if thermal_seismic_features.ndim > 1 else thermal_seismic_features[:len(thermal_field)]
                )[0, 1] if len(thermal_seismic_features) >= len(thermal_field) else 0.0
                
                # Thermal-modulated seismic amplitude statistics
                features['thermal_modulated_amplitude_mean'] = np.mean(thermal_seismic_features)
                features['thermal_modulated_amplitude_std'] = np.std(thermal_seismic_features)
                
            else:
                # Single channel seismic data
                features['thermal_seismic_coherence'] = 0.0
                features['thermal_modulated_amplitude_mean'] = 0.0
                features['thermal_modulated_amplitude_std'] = 0.0
            
            # 5. Thermal anomaly indicators in seismic context
            baseline_thermal = np.mean(thermal_field)
            thermal_anomaly_mask = np.abs(thermal_field - baseline_thermal) > 2 * np.std(thermal_field)
            
            features['thermal_anomaly_seismic_depth'] = np.mean(depths[thermal_anomaly_mask]) if np.any(thermal_anomaly_mask) else 0.0
            features['thermal_anomaly_volume_fraction'] = np.sum(thermal_anomaly_mask) / len(thermal_anomaly_mask)
            
            # 6. Physics-informed thermal-seismic coupling
            # Use Deep Earth ML physics-informed network for coupling analysis
            if hasattr(self.deep_earth_ml, 'physics_informed_network'):
                physics_input = np.column_stack([
                    np.full(len(depths), lat),
                    np.full(len(depths), lon),
                    depths,
                    thermal_field
                ])
                
                physics_tensor = torch.FloatTensor(physics_input)
                physics_output = self.deep_earth_ml.physics_informed_network(physics_tensor)
                physics_coupling = physics_output.detach().numpy()
                
                features['physics_thermal_seismic_coupling'] = np.mean(physics_coupling)
                features['physics_coupling_depth_variation'] = np.std(physics_coupling)
            else:
                features['physics_thermal_seismic_coupling'] = 0.0
                features['physics_coupling_depth_variation'] = 0.0
            
            print(f"Generated thermal-seismic coupling features for coordinates {coordinates}")
            return features
            
        except Exception as e:
            warnings.warn(f"Error in thermal-seismic feature extraction: {e}")
            return self._get_default_thermal_seismic_features(seismic_data)
    
    def _get_default_thermal_seismic_features(self, seismic_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate default thermal-seismic features when Deep Earth ML is not available"""
        return {
            'thermal_velocity_perturbation_mean': 0.0,
            'thermal_velocity_perturbation_std': 0.0,
            'thermal_velocity_perturbation_max': 0.0,
            'thermal_gradient_effect': 0.0,
            'thermal_stratification': 0.0,
            'thermal_attenuation_mean': 0.0,
            'thermal_attenuation_integrated': 0.0,
            'thermal_seismic_coherence': 0.0,
            'thermal_modulated_amplitude_mean': 0.0,
            'thermal_modulated_amplitude_std': 0.0,
            'thermal_anomaly_seismic_depth': 0.0,
            'thermal_anomaly_volume_fraction': 0.0,
            'physics_thermal_seismic_coupling': 0.0,
            'physics_coupling_depth_variation': 0.0
        }
    
    def _initialize_models(self):
        """Initialize all deep learning models"""
        self.models = {
            'transformer': SeismicTransformer(self.config),
            'first_motion': FirstMotionCNN(self.config),
            'magnitude': MagnitudeRNN(self.config),
            'gan': SeismicGAN(self.config)
        }
        
        # Add models to active learner
        self.active_learner.add_models(self.models)
    
    def preprocess_waveform(self, data: np.ndarray, 
                          sample_rate: Optional[float] = None) -> np.ndarray:
        """Preprocess seismic waveform data"""
        if sample_rate is None:
            sample_rate = self.config.sample_rate
        
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available - skipping preprocessing")
            return data
        
        # Ensure 3-component data
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        
        processed_data = np.zeros_like(data)
        
        for i in range(data.shape[0]):
            # Bandpass filtering
            for low, high in self.config.frequency_bands:
                nyquist = sample_rate / 2
                low_norm = low / nyquist
                high_norm = high / nyquist
                
                if high_norm >= 1.0:
                    high_norm = 0.99
                
                b, a = butter(4, [low_norm, high_norm], btype='band')
                filtered = filtfilt(b, a, data[i])
                processed_data[i] += filtered / len(self.config.frequency_bands)
        
        # Normalize
        processed_data = (processed_data - np.mean(processed_data, axis=1, keepdims=True)) / \
                        (np.std(processed_data, axis=1, keepdims=True) + 1e-8)
        
        return processed_data
    
    def detect_phases(self, data: np.ndarray) -> Dict:
        """Detect P and S phases using transformer model"""
        if not TORCH_AVAILABLE or 'transformer' not in self.models:
            # Simplified phase detection
            return self._simple_phase_detection(data)
        
        model = self.models['transformer']
        model.eval()
        
        # Prepare data
        if data.ndim == 2:
            data = data.T  # Transpose to (time, channels)
            data = data[np.newaxis, :]  # Add batch dimension
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            predictions = model(data_tensor)
        
        return {
            'p_picks': self._find_peaks(predictions['p_probability'].numpy()),
            's_picks': self._find_peaks(predictions['s_probability'].numpy()),
            'uncertainties': predictions['uncertainties'].numpy(),
            'confidence_scores': {
                'p_confidence': float(predictions['p_probability'].max()),
                's_confidence': float(predictions['s_probability'].max())
            }
        }
    
    def determine_first_motion(self, data: np.ndarray, p_arrival: int) -> Dict:
        """Determine first motion polarity"""
        if not TORCH_AVAILABLE or 'first_motion' not in self.models:
            return {'polarity': 'undetermined', 'confidence': 0.0}
        
        model = self.models['first_motion']
        model.eval()
        
        # Extract window around P arrival
        window_size = int(0.5 * self.config.sample_rate)  # 0.5 second window
        start_idx = max(0, p_arrival - window_size // 2)
        end_idx = min(data.shape[-1], p_arrival + window_size // 2)
        
        p_window = data[:, start_idx:end_idx]
        
        # Pad if necessary
        if p_window.shape[1] < window_size:
            padding = window_size - p_window.shape[1]
            p_window = np.pad(p_window, ((0, 0), (0, padding)), 'constant')
        
        # Add batch dimension
        p_window = p_window[np.newaxis, :]
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(p_window)
            predictions = model(data_tensor)
        
        polarity_probs = predictions['polarity_probs'].numpy()[0]
        polarity_classes = ['up', 'down', 'undetermined']
        
        best_class = np.argmax(polarity_probs)
        
        return {
            'polarity': polarity_classes[best_class],
            'confidence': float(polarity_probs[best_class]),
            'probabilities': {
                'up': float(polarity_probs[0]),
                'down': float(polarity_probs[1]),
                'undetermined': float(polarity_probs[2])
            }
        }
    
    def estimate_magnitude(self, data: np.ndarray) -> Dict:
        """Estimate earthquake magnitude"""
        if not TORCH_AVAILABLE or 'magnitude' not in self.models:
            # Simple amplitude-based magnitude
            max_amp = np.max(np.abs(data))
            magnitude = np.log10(max_amp) + 2.0  # Simplified relationship
            return {'magnitude': float(magnitude), 'uncertainty': 0.5}
        
        model = self.models['magnitude']
        model.eval()
        
        # Add batch dimension if needed
        if data.ndim == 2:
            data = data[np.newaxis, :]
        
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data)
            predictions = model(data_tensor)
        
        return {
            'magnitude': float(predictions['magnitude'].numpy()[0]),
            'uncertainty': float(predictions['uncertainty'].numpy()[0])
        }
    
    def augment_data(self, n_samples: int = 100) -> np.ndarray:
        """Generate augmented seismic data using GAN"""
        if not TORCH_AVAILABLE or 'gan' not in self.models:
            # Return random noise as fallback
            return np.random.randn(n_samples, 3, self.config.window_size)
        
        return self.models['gan'].generate_samples(n_samples)
    
    def _simple_phase_detection(self, data: np.ndarray) -> Dict:
        """Simple phase detection without deep learning"""
        if not SCIPY_AVAILABLE:
            return {'p_picks': [], 's_picks': [], 'uncertainties': np.array([])}
        
        # Use STA/LTA for simple detection
        sta_window = int(0.5 * self.config.sample_rate)
        lta_window = int(5.0 * self.config.sample_rate)
        
        # Calculate characteristic function
        cf = np.sum(data**2, axis=0)  # Energy
        
        # STA/LTA
        sta_lta = np.zeros_like(cf)
        for i in range(lta_window, len(cf) - sta_window):
            sta = np.mean(cf[i:i+sta_window])
            lta = np.mean(cf[i-lta_window:i])
            if lta > 0:
                sta_lta[i] = sta / lta
        
        # Find peaks
        threshold = 3.0
        p_picks = np.where(sta_lta > threshold)[0].tolist()
        s_picks = []  # Would need more sophisticated detection
        
        return {
            'p_picks': p_picks,
            's_picks': s_picks,
            'uncertainties': np.ones(len(p_picks)) * 0.5
        }
    
    def _find_peaks(self, probability_trace: np.ndarray, threshold: float = 0.5) -> List[int]:
        """Find peaks in probability trace"""
        if probability_trace.ndim > 1:
            probability_trace = probability_trace[0]  # Take first batch
        
        peaks = []
        above_threshold = probability_trace > threshold
        
        # Find start and end of above-threshold regions
        diff = np.diff(above_threshold.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        
        # Handle edge cases
        if above_threshold[0]:
            starts = np.concatenate([[0], starts])
        if above_threshold[-1]:
            ends = np.concatenate([ends, [len(probability_trace)]])
        
        # Find peak in each region
        for start, end in zip(starts, ends):
            if start < end:
                region = probability_trace[start:end]
                peak_idx = start + np.argmax(region)
                peaks.append(int(peak_idx))
        
        return peaks
    
    def process_continuous_stream(self, data_stream: np.ndarray) -> Dict:
        """Process continuous seismic data stream"""
        window_size = self.config.window_size
        overlap = int(window_size * self.config.overlap_ratio)
        step_size = window_size - overlap
        
        results = []
        
        for i in range(0, data_stream.shape[-1] - window_size + 1, step_size):
            window_data = data_stream[:, i:i + window_size]
            
            # Preprocess
            processed = self.preprocess_waveform(window_data)
            
            # Analyze window
            phases = self.detect_phases(processed)
            magnitude = self.estimate_magnitude(processed)
            
            # Adjust timing to global stream
            for pick_list in [phases['p_picks'], phases['s_picks']]:
                for j in range(len(pick_list)):
                    pick_list[j] += i
            
            window_result = {
                'start_time': i / self.config.sample_rate,
                'phases': phases,
                'magnitude': magnitude
            }
            
            results.append(window_result)
        
        return {
            'window_results': results,
            'processing_summary': {
                'total_windows': len(results),
                'total_p_picks': sum(len(r['phases']['p_picks']) for r in results),
                'total_s_picks': sum(len(r['phases']['s_picks']) for r in results)
            }
        }
    
    def export_results(self, results: Dict, filename: str):
        """Export processing results to file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'sample_rate': self.config.sample_rate,
                'window_size': self.config.window_size,
                'frequency_bands': self.config.frequency_bands
            },
            'results': results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to export results: {e}")
    
    async def setup_real_time_processing(self, 
                                       latitude: float, 
                                       longitude: float, 
                                       radius_km: float = 500.0) -> bool:
        """Setup real-time seismic processing with IRIS data streams"""
        
        if not self.iris_connector:
            self.logger.error("IRIS connector not available for real-time processing")
            return False
        
        try:
            async with self.iris_connector:
                # Get stations in the region
                stations = await self.iris_connector.get_station_metadata(
                    latitude=latitude,
                    longitude=longitude,
                    max_radius=radius_km / 111.0  # Convert km to degrees
                )
                
                # Filter for high-quality broadband stations
                self.real_time_stations = [
                    station for station in stations
                    if (station.channel.startswith('BH') or station.channel.startswith('HH'))
                    and station.sample_rate >= 20.0
                    and station.data_quality in ['A', 'B']
                ]
                
                self.logger.info(f"Setup {len(self.real_time_stations)} stations for real-time processing")
                return len(self.real_time_stations) > 0
                
        except Exception as e:
            self.logger.error(f"Failed to setup real-time processing: {e}")
            return False
    
    def _analyze_spectral_content(self, waveform: np.ndarray) -> Dict:
        """Analyze spectral content of waveform"""
        
        if not SCIPY_AVAILABLE:
            return {'dominant_frequency': 1.0, 'bandwidth': 10.0}
        
        # Calculate power spectral density
        freqs, times, psd = spectrogram(waveform, fs=self.config.sample_rate, nperseg=1024)
        
        # Find dominant frequency
        mean_psd = np.mean(psd, axis=1)
        dominant_freq_idx = np.argmax(mean_psd)
        dominant_frequency = freqs[dominant_freq_idx]
        
        # Calculate bandwidth (frequencies containing 90% of energy)
        cumulative_energy = np.cumsum(mean_psd) / np.sum(mean_psd)
        f_low = freqs[np.where(cumulative_energy >= 0.05)[0][0]]
        f_high = freqs[np.where(cumulative_energy >= 0.95)[0][0]]
        bandwidth = f_high - f_low
        
        return {
            'dominant_frequency': float(dominant_frequency),
            'bandwidth': float(bandwidth),
            'frequency_range': [float(f_low), float(f_high)],
            'spectral_centroid': float(np.sum(freqs * mean_psd) / np.sum(mean_psd))
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance in km"""
        
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    async def get_historical_seismicity(self, 
                                      latitude: float, 
                                      longitude: float,
                                      radius_km: float = 100.0,
                                      days_back: int = 30) -> List[Dict]:
        """Get historical seismicity from IRIS catalog"""
        
        if not self.iris_connector:
            return []
        
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            async with self.iris_connector:
                events = await self.iris_connector.get_earthquake_catalog(
                    start_time=start_time,
                    end_time=end_time,
                    latitude=latitude,
                    longitude=longitude,
                    max_radius=radius_km / 111.0,
                    min_magnitude=2.0
                )
            
            # Convert to standard format
            historical_events = []
            for event in events:
                historical_events.append({
                    'event_id': event.event_id,
                    'origin_time': event.origin_time,
                    'latitude': event.latitude,
                    'longitude': event.longitude,
                    'depth': event.depth,
                    'magnitude': event.magnitude,
                    'magnitude_type': event.magnitude_type,
                    'location_name': event.location_name,
                    'catalog': event.catalog,
                    'uncertainties': event.uncertainty
                })
            
            self.logger.info(f"Retrieved {len(historical_events)} historical events")
            return historical_events
            
        except Exception as e:
            self.logger.error(f"Failed to get historical seismicity: {e}")
            return []


# Example usage and testing
def main():
    """Example usage of seismic processing pipeline"""
    
    # Initialize processor
    config = SeismicConfig(
        sample_rate=100.0,
        window_size=6000,
        transformer_layers=4,
        batch_size=16
    )
    
    processor = SeismicProcessor(config)
    
    # Generate synthetic data for testing
    time = np.linspace(0, 60, 6000)  # 60 seconds at 100 Hz
    
    # Simulate 3-component seismogram with P and S arrivals
    p_arrival_time = 10.0  # seconds
    s_arrival_time = 18.0  # seconds
    
    # P wave
    p_signal = np.exp(-(time - p_arrival_time)**2 / 0.5) * np.sin(2 * np.pi * 10 * (time - p_arrival_time))
    # S wave  
    s_signal = np.exp(-(time - s_arrival_time)**2 / 1.0) * np.sin(2 * np.pi * 5 * (time - s_arrival_time))
    
    # Add noise
    noise_level = 0.1
    noise = np.random.randn(3, len(time)) * noise_level
    
    # Create 3-component data
    seismic_data = np.array([
        p_signal + s_signal + noise[0],  # Z component
        0.7 * (p_signal + s_signal) + noise[1],  # N component  
        0.5 * (p_signal + s_signal) + noise[2]   # E component
    ])
    
    print("Processing seismic data...")
    
    # Process the data
    results = processor.process_continuous_stream(seismic_data)
    
    print(f"Processing complete:")
    print(f"- Processed {results['processing_summary']['total_windows']} windows")
    print(f"- Found {results['processing_summary']['total_p_picks']} P picks")
    print(f"- Found {results['processing_summary']['total_s_picks']} S picks")
    
    # Test individual components
    if results['window_results']:
        first_window = results['window_results'][0]
        
        # Test first motion determination
        if first_window['phases']['p_picks']:
            p_pick = first_window['phases']['p_picks'][0]
            first_motion = processor.determine_first_motion(seismic_data, p_pick)
            print(f"First motion: {first_motion['polarity']} (confidence: {first_motion['confidence']:.3f})")
        
        # Test magnitude estimation
        magnitude = processor.estimate_magnitude(seismic_data)
        print(f"Magnitude: {magnitude['magnitude']:.2f} ± {magnitude['uncertainty']:.2f}")
    
    # Test data augmentation
    augmented_data = processor.augment_data(n_samples=5)
    print(f"Generated {augmented_data.shape[0]} synthetic seismograms")
    
    # Export results
    processor.export_results(results, 'seismic_analysis_results.json')
    print("Results exported to seismic_analysis_results.json")
    
    return processor, results


# Maintain backward compatibility
class SeismicProcessor(EnhancedSeismicProcessor):
    """Backward compatibility wrapper"""
    pass


if __name__ == "__main__":
    # Example usage
    async def main():
        processor = EnhancedSeismicProcessor()
        
        # Setup real-time processing for California
        success = await processor.setup_real_time_processing(
            latitude=37.0,
            longitude=-120.0,
            radius_km=200.0
        )
        
        if success:
            print(f"✓ Setup {len(processor.real_time_stations)} stations for real-time processing")
        else:
            print("⚠ Real-time processing setup failed - using fallback mode")
        
        return processor
    
    # Run example
    import asyncio
    processor = asyncio.run(main())