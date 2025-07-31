"""
GAN Engine for Geological Scenario Generation and Uncertainty Quantification

This module implements Generative Adversarial Networks (GANs) specifically designed
for generating realistic geological scenarios and quantifying uncertainty in
geological predictions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import structlog
from pathlib import Path

from models.ai_models import ModelMetadata, UncertaintyQuantification, UncertaintyMethod
from models.geological import GeologicalLocation, MagmaReservoir, SeismicEvent

logger = structlog.get_logger()


@dataclass
class GANConfig:
    """Configuration for GAN models"""
    model_id: str
    latent_dim: int
    output_dim: int
    generator_layers: List[int]
    discriminator_layers: List[int]
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 64
    noise_type: str = "normal"  # normal, uniform
    spectral_normalization: bool = True
    gradient_penalty_weight: float = 10.0
    
    def __post_init__(self):
        if self.latent_dim <= 0:
            raise ValueError("Latent dimension must be positive")
        if self.output_dim <= 0:
            raise ValueError("Output dimension must be positive")
        if not self.generator_layers:
            raise ValueError("Generator must have at least one hidden layer")
        if not self.discriminator_layers:
            raise ValueError("Discriminator must have at least one hidden layer")


class SpectralNorm(nn.Module):
    """Spectral normalization for stable GAN training"""
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = nn.functional.normalize(torch.mv(torch.t(w.view(height,-1).data), u.data), dim=0)
            u.data = nn.functional.normalize(torch.mv(w.view(height,-1).data, v.data), dim=0)

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = nn.functional.normalize(u.data, dim=0)
        v.data = nn.functional.normalize(v.data, dim=0)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class GeologicalGenerator(nn.Module):
    """
    Generator network for creating realistic geological scenarios
    
    Generates geological parameters including:
    - Magma reservoir properties (volume, composition, temperature, pressure)
    - Seismic activity patterns
    - Deformation fields
    - Thermal distributions
    """
    
    def __init__(self, config: GANConfig):
        super(GeologicalGenerator, self).__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        
        # Build generator network
        layers = []
        prev_dim = self.latent_dim
        
        for i, hidden_dim in enumerate(config.generator_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer with appropriate activation
        layers.append(nn.Linear(prev_dim, self.output_dim))
        layers.append(nn.Tanh())  # Normalize outputs to [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Handle spectral normalization
                if hasattr(module, 'weight_bar'):
                    nn.init.xavier_uniform_(module.weight_bar)
                elif hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate geological scenarios from noise
        
        Args:
            noise: Random noise tensor [batch_size, latent_dim]
            
        Returns:
            Generated geological parameters [batch_size, output_dim]
        """
        return self.network(noise)
    
    def generate_scenarios(self, num_scenarios: int, device: torch.device) -> torch.Tensor:
        """Generate multiple geological scenarios"""
        if self.config.noise_type == "normal":
            noise = torch.randn(num_scenarios, self.latent_dim, device=device)
        elif self.config.noise_type == "uniform":
            noise = torch.rand(num_scenarios, self.latent_dim, device=device) * 2 - 1
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")
        
        with torch.no_grad():
            scenarios = self.forward(noise)
        
        return scenarios


class GeologicalDiscriminator(nn.Module):
    """
    Discriminator network for assessing geological plausibility
    
    Evaluates whether geological scenarios are realistic based on:
    - Physical constraints and relationships
    - Statistical distributions from real data
    - Geological domain knowledge
    """
    
    def __init__(self, config: GANConfig):
        super(GeologicalDiscriminator, self).__init__()
        self.config = config
        self.input_dim = config.output_dim
        
        # Build discriminator network
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(config.discriminator_layers[:-1]):
            linear = nn.Linear(prev_dim, hidden_dim)
            if config.spectral_normalization:
                linear = SpectralNorm(linear)
            layers.append(linear)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer (probability of being real)
        output_linear = nn.Linear(prev_dim, config.discriminator_layers[-1])
        if config.spectral_normalization:
            output_linear = SpectralNorm(output_linear)
        layers.append(output_linear)
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Handle spectral normalization
                if hasattr(module, 'weight_bar'):
                    nn.init.xavier_uniform_(module.weight_bar)
                elif hasattr(module, 'weight'):
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, scenarios: torch.Tensor) -> torch.Tensor:
        """
        Assess geological plausibility of scenarios
        
        Args:
            scenarios: Geological scenarios [batch_size, output_dim]
            
        Returns:
            Plausibility scores [batch_size, 1]
        """
        return self.network(scenarios)


class GeologicalGAN:
    """
    Complete GAN system for geological scenario generation
    """
    
    def __init__(self, config: GANConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize networks
        self.generator = GeologicalGenerator(config).to(device)
        self.discriminator = GeologicalDiscriminator(config).to(device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        
        # Training history
        self.training_history = {
            "g_loss": [],
            "d_loss": [],
            "d_real_acc": [],
            "d_fake_acc": [],
            "epoch": []
        }
        
        # Model metadata
        self.metadata = ModelMetadata(
            model_id=config.model_id,
            version="1.0.0",
            creation_timestamp=datetime.utcnow(),
            training_data_hash="untrained",
            performance_metrics={},
            hyperparameters=config.__dict__,
            tags=["gan", "geological", "uncertainty"]
        )
    
    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_samples.size(0)
        
        # Random interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)
        
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output for interpolated samples
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return penalty
    
    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            real_data: Real geological data [batch_size, output_dim]
            
        Returns:
            Training metrics
        """
        batch_size = real_data.size(0)
        
        # Labels for real and fake data
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Real data
        d_real = self.discriminator(real_data)
        d_real_loss = self.adversarial_loss(d_real, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        
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
 self.generator(noise).detach()
        d_fake = self.discriminator(fake_data)
        d_fake_loss = self.adversarial_loss(d_fake, fake_labels)
        
        # Gradient penalty (for WGAN-GP)
        gp = self.gradient_penalty(real_data, fake_data)
        
        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss + self.config.gradient_penalty_weight * gp
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        noise = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        
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
 self.generator(noise)
        d_fake = self.discriminator(fake_data)
        
        # Generator loss (wants discriminator to classify fake as real)
        g_loss = self.adversarial_loss(d_fake, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        # Calculate accuracies
        d_real_acc = (d_real > 0.5).float().mean().item()
        d_fake_acc = (d_fake <= 0.5).float().mean().item()
        
        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
            "gradient_penalty": gp.item()
        }
    
    def train(self, dataloader: torch.utils.data.DataLoader, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the GAN
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                "g_loss": [],
                "d_loss": [],
                "d_real_acc": [],
                "d_fake_acc": [],
                "gradient_penalty": []
            }
            
            for batch_idx, (real_data,) in enumerate(dataloader):
                real_data = real_data.to(self.device)
                
                # Training step
                metrics = self.train_step(real_data)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
            
            # Average metrics for epoch
            avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
            
            # Update training history
            self.training_history["epoch"].append(epoch)
            self.training_history["g_loss"].append(avg_metrics["g_loss"])
            self.training_history["d_loss"].append(avg_metrics["d_loss"])
            self.training_history["d_real_acc"].append(avg_metrics["d_real_acc"])
            self.training_history["d_fake_acc"].append(avg_metrics["d_fake_acc"])
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs}",
                    g_loss=f"{avg_metrics['g_loss']:.4f}",
                    d_loss=f"{avg_metrics['d_loss']:.4f}",
                    d_real_acc=f"{avg_metrics['d_real_acc']:.4f}",
                    d_fake_acc=f"{avg_metrics['d_fake_acc']:.4f}"
                )
        
        logger.info("GAN training completed")
        return self.training_history
    
    def generate_scenarios(self, num_scenarios: int) -> torch.Tensor:
        """Generate geological scenarios"""
        self.generator.eval()
        return self.generator.generate_scenarios(num_scenarios, self.device)
    
    def assess_plausibility(self, scenarios: torch.Tensor) -> torch.Tensor:
        """Assess geological plausibility of scenarios"""
        self.discriminator.eval()
        with torch.no_grad():
            plausibility = self.discriminator(scenarios)
        return plausibility
    
    def save_model(self, filepath: Path):
        """Save GAN model"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'metadata': self.metadata.to_dict()
        }, filepath)
        
        logger.info(f"GAN model saved to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load GAN model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.metadata = ModelMetadata.from_dict(checkpoint['metadata'])
        
        logger.info(f"GAN model loaded from {filepath}")


class GANEnsemble:
    """
    Ensemble of GAN models for robust uncertainty quantification
    """
    
    def __init__(self, configs: List[GANConfig], device: torch.device):
        self.device = device
        self.gans = []
        
        # Create ensemble of GANs with different configurations
        for config in configs:
            gan = GeologicalGAN(config, device)
            self.gans.append(gan)
        
        logger.info(f"Created GAN ensemble with {len(self.gans)} models")
    
    def train_ensemble(self, dataloader: torch.utils.data.DataLoader, num_epochs: int):
        """Train all GANs in the ensemble"""
        logger.info("Training GAN ensemble")
        
        for i, gan in enumerate(self.gans):
            logger.info(f"Training GAN {i+1}/{len(self.gans)}")
            gan.train(dataloader, num_epochs)
    
    def generate_ensemble_scenarios(self, num_scenarios: int) -> List[torch.Tensor]:
        """Generate scenarios from all GANs in ensemble"""
        scenarios_list = []
        
        for gan in self.gans:
            scenarios = gan.generate_scenarios(num_scenarios)
            scenarios_list.append(scenarios)
        
        return scenarios_list
    
    def quantify_uncertainty(self, num_scenarios: int) -> UncertaintyQuantification:
        """
        Quantify uncertainty using ensemble variance
        
        Args:
            num_scenarios: Number of scenarios to generate
            
        Returns:
            Uncertainty quantification results
        """
        # Generate scenarios from all models
        ensemble_scenarios = self.generate_ensemble_scenarios(num_scenarios)
        
        # Stack scenarios from all models
        all_scenarios = torch.stack(ensemble_scenarios, dim=0)  # [num_models, num_scenarios, output_dim]
        
        # Calculate ensemble statistics
        mean_scenarios = torch.mean(all_scenarios, dim=0)  # [num_scenarios, output_dim]
        var_scenarios = torch.var(all_scenarios, dim=0)    # [num_scenarios, output_dim]
        std_scenarios = torch.std(all_scenarios, dim=0)    # [num_scenarios, output_dim]
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = var_scenarios.cpu().numpy()
        
        # Aleatoric uncertainty (data uncertainty) - estimated from individual model variance
        aleatoric_uncertainty = np.zeros_like(epistemic_uncertainty)
        for scenarios in ensemble_scenarios:
            aleatoric_uncertainty += torch.var(scenarios, dim=0).cpu().numpy()
        aleatoric_uncertainty /= len(ensemble_scenarios)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence intervals (95%)
        confidence_intervals = {}
        for i in range(mean_scenarios.shape[1]):
            lower = torch.quantile(all_scenarios[:, :, i], 0.025).item()
            upper = torch.quantile(all_scenarios[:, :, i], 0.975).item()
            confidence_intervals[f"param_{i}"] = (lower, upper)
        
        return UncertaintyQuantification(
            method=UncertaintyMethod.ENSEMBLE_VARIANCE,
            confidence_intervals=confidence_intervals,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            total_uncertainty=total_uncertainty
        )


class GANEngine:
    """
    Main engine for managing GAN models for geological scenario generation
    """
    
    def __init__(self, device: torch.device, model_cache_dir: str = "models/gan"):
        self.device = device
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Registry of GAN models
        self.gans: Dict[str, GeologicalGAN] = {}
        self.ensembles: Dict[str, GANEnsemble] = {}
        
        logger.info("GAN Engine initialized")
    
    def create_gan(self, config: GANConfig) -> GeologicalGAN:
        """Create a new GAN model"""
        if config.model_id in self.gans:
            raise ValueError(f"GAN model {config.model_id} already exists")
        
        gan = GeologicalGAN(config, self.device)
        self.gans[config.model_id] = gan
        
        logger.info(f"Created GAN model: {config.model_id}")
        return gan
    
    def create_ensemble(self, ensemble_id: str, configs: List[GANConfig]) -> GANEnsemble:
        """Create a GAN ensemble"""
        if ensemble_id in self.ensembles:
            raise ValueError(f"GAN ensemble {ensemble_id} already exists")
        
        ensemble = GANEnsemble(configs, self.device)
        self.ensembles[ensemble_id] = ensemble
        
        logger.info(f"Created GAN ensemble: {ensemble_id}")
        return ensemble
    
    def get_gan(self, model_id: str) -> GeologicalGAN:
        """Get a GAN model by ID"""
        if model_id not in self.gans:
            raise ValueError(f"GAN model {model_id} not found")
        return self.gans[model_id]
    
    def get_ensemble(self, ensemble_id: str) -> GANEnsemble:
        """Get a GAN ensemble by ID"""
        if ensemble_id not in self.ensembles:
            raise ValueError(f"GAN ensemble {ensemble_id} not found")
        return self.ensembles[ensemble_id]
    
    def list_gans(self) -> List[str]:
        """List all available GAN models"""
        return list(self.gans.keys())
    
    def list_ensembles(self) -> List[str]:
        """List all available GAN ensembles"""
        return list(self.ensembles.keys())
    
    def save_gan(self, model_id: str):
        """Save a GAN model to disk"""
        gan = self.get_gan(model_id)
        filepath = self.model_cache_dir / f"{model_id}.pth"
        gan.save_model(filepath)
    
    def load_gan(self, model_id: str, filepath: Optional[Path] = None):
        """Load a GAN model from disk"""
        if filepath is None:
            filepath = self.model_cache_dir / f"{model_id}.pth"
        
        # Load checkpoint to get config
        checkpoint = torch.load(filepath, map_location=self.device)
        config = checkpoint['config']
        
        # Create and load GAN
        gan = GeologicalGAN(config, self.device)
        gan.load_model(filepath)
        
        self.gans[model_id] = gan
        logger.info(f"Loaded GAN model: {model_id}")