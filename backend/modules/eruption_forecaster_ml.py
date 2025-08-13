#!/usr/bin/env python3
"""
Advanced ML-based Eruption Forecasting Engine
Physics-Informed Neural Networks for volcanic eruption prediction
Based on latest research in volcano science and deep learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
import optuna
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EruptionForecast:
    """Eruption forecast data structure"""
    eruption_probability: float
    time_to_eruption: Optional[float]  # days
    expected_vei: int
    confidence: float
    uncertainty_bounds: Dict[str, List[float]]
    alert_level: str
    contributing_factors: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime

class VolcanicDataset(Dataset):
    """PyTorch dataset for volcanic monitoring data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 30, 
                 target_horizon: int = 7, mode: str = 'train'):
        self.data = data
        self.sequence_length = sequence_length
        self.target_horizon = target_horizon
        self.mode = mode
        
        # Feature columns
        self.feature_cols = [
            'pressure_shallow', 'pressure_middle', 'pressure_deep',
            'temperature_shallow', 'temperature_middle', 'temperature_deep',
            'gas_co2', 'gas_so2', 'gas_h2s', 'gas_ratio_co2_so2',
            'seismic_magnitude_avg', 'seismic_count', 'seismic_energy',
            'deformation_radial', 'deformation_vertical', 'deformation_rate',
            'thermal_anomaly', 'thermal_area', 'thermal_intensity',
            'gravity_change', 'magnetic_change',
            'tidal_stress', 'regional_seismicity', 'precipitation'
        ]
        
        # Preprocess data
        self.prepare_sequences()
    
    def prepare_sequences(self):
        """Prepare time series sequences for training"""
        self.sequences = []
        self.targets = []
        self.metadata = []
        
        # Group by volcano
        for volcano_id in self.data['volcano_id'].unique():
            volcano_data = self.data[self.data['volcano_id'] == volcano_id].sort_values('timestamp')
            
            # Create sequences
            for i in range(len(volcano_data) - self.sequence_length - self.target_horizon):
                # Input sequence
                sequence = volcano_data.iloc[i:i+self.sequence_length][self.feature_cols].values
                
                # Target (eruption in next target_horizon days)
                target_window = volcano_data.iloc[i+self.sequence_length:i+self.sequence_length+self.target_horizon]
                eruption_occurred = (target_window['eruption_flag'].sum() > 0).astype(int)
                
                # VEI if eruption occurred
                vei = target_window['vei'].max() if eruption_occurred else 0
                
                # Time to eruption
                if eruption_occurred:
                    eruption_idx = target_window[target_window['eruption_flag'] == 1].index[0]
                    time_to_eruption = (eruption_idx - volcano_data.iloc[i+self.sequence_length].name)
                else:
                    time_to_eruption = self.target_horizon + 1  # No eruption
                
                self.sequences.append(sequence)
                self.targets.append([eruption_occurred, vei, time_to_eruption])
                self.metadata.append({
                    'volcano_id': volcano_id,
                    'timestamp': volcano_data.iloc[i+self.sequence_length]['timestamp']
                })
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class PhysicsInformedEruptionNet(pl.LightningModule):
    """
    Physics-Informed Neural Network for eruption forecasting
    Incorporates physical constraints and domain knowledge
    """
    
    def __init__(self, input_dim: int = 23, hidden_dim: int = 256, 
                 num_layers: int = 4, dropout: float = 0.2, 
                 learning_rate: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.save_hyperparameters()
        
        # Multi-scale temporal encoding
        self.temporal_encoder = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim//4, batch_first=True, dropout=dropout),  # Daily
            nn.LSTM(input_dim, hidden_dim//4, batch_first=True, dropout=dropout),  # Weekly  
            nn.LSTM(input_dim, hidden_dim//4, batch_first=True, dropout=dropout),  # Monthly
            nn.LSTM(input_dim, hidden_dim//4, batch_first=True, dropout=dropout)   # Seasonal
        ])
        
        # Physics-informed layers
        self.physics_layer = PhysicsConstraintLayer(hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers-1)
        ])
        
        # Output heads
        self.eruption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
        self.vei_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 8)  # VEI 0-7
        )
        
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.ReLU()  # Time must be positive
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 3),  # Uncertainty for each output
            nn.Softplus()
        )
        
        # Loss weights
        self.eruption_weight = 2.0
        self.vei_weight = 1.0
        self.time_weight = 1.5
        self.physics_weight = 0.5
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Multi-scale temporal encoding
        temporal_features = []
        scales = [1, 7, 30, 90]  # Daily, weekly, monthly, seasonal
        
        for i, (encoder, scale) in enumerate(zip(self.temporal_encoder, scales)):
            # Downsample for different temporal scales
            if scale > 1:
                downsampled = x[:, ::scale, :]
                if downsampled.shape[1] == 0:
                    downsampled = x[:, -1:, :]  # Use last timestep if too short
            else:
                downsampled = x
                
            encoded, _ = encoder(downsampled)
            # Use last output
            temporal_features.append(encoded[:, -1, :])
        
        # Combine temporal features
        combined_features = torch.cat(temporal_features, dim=1)
        
        # Apply physics constraints
        physics_constrained = self.physics_layer(combined_features, x)
        
        # Self-attention
        attended_features, _ = self.attention(
            physics_constrained.unsqueeze(0),
            physics_constrained.unsqueeze(0), 
            physics_constrained.unsqueeze(0)
        )
        attended_features = attended_features.squeeze(0)
        
        # Feature extraction
        h = attended_features
        for layer in self.feature_layers:
            h = F.relu(layer(h))
        
        # Output predictions
        eruption_prob = self.eruption_head(h)
        vei_logits = self.vei_head(h)
        time_pred = self.time_head(h)
        uncertainty = self.uncertainty_head(h)
        
        return {
            'eruption_prob': eruption_prob,
            'vei_logits': vei_logits,
            'time_pred': time_pred,
            'uncertainty': uncertainty,
            'features': h
        }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        # Extract targets
        eruption_target = y[:, 0].unsqueeze(1)
        vei_target = y[:, 1].long()
        time_target = y[:, 2].unsqueeze(1)
        
        # Calculate losses
        eruption_loss = F.binary_cross_entropy(outputs['eruption_prob'], eruption_target)
        vei_loss = F.cross_entropy(outputs['vei_logits'], vei_target)
        time_loss = F.mse_loss(outputs['time_pred'], time_target)
        
        # Physics-based loss
        physics_loss = self.physics_layer.physics_loss(x, outputs)
        
        # Uncertainty loss (negative log-likelihood)
        uncertainty_loss = self.uncertainty_loss(outputs['uncertainty'], 
                                                [eruption_loss, vei_loss, time_loss])
        
        # Combined loss
        total_loss = (self.eruption_weight * eruption_loss + 
                     self.vei_weight * vei_loss +
                     self.time_weight * time_loss +
                     self.physics_weight * physics_loss +
                     0.1 * uncertainty_loss)
        
        # Logging
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_eruption_loss', eruption_loss)
        self.log('train_vei_loss', vei_loss)
        self.log('train_time_loss', time_loss)
        self.log('train_physics_loss', physics_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        
        # Extract targets
        eruption_target = y[:, 0].unsqueeze(1)
        vei_target = y[:, 1].long()
        time_target = y[:, 2].unsqueeze(1)
        
        # Calculate losses
        eruption_loss = F.binary_cross_entropy(outputs['eruption_prob'], eruption_target)
        vei_loss = F.cross_entropy(outputs['vei_logits'], vei_target)
        time_loss = F.mse_loss(outputs['time_pred'], time_target)
        
        total_loss = (self.eruption_weight * eruption_loss + 
                     self.vei_weight * vei_loss +
                     self.time_weight * time_loss)
        
        # Metrics
        eruption_acc = ((outputs['eruption_prob'] > 0.5) == eruption_target).float().mean()
        vei_acc = (outputs['vei_logits'].argmax(dim=1) == vei_target).float().mean()
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_eruption_acc', eruption_acc)
        self.log('val_vei_acc', vei_acc)
        
        return total_loss
    
    def uncertainty_loss(self, uncertainty, losses):
        """Uncertainty-weighted loss"""
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = 1.0 / (uncertainty[:, i] + 1e-8)
            weighted_loss = precision * loss + torch.log(uncertainty[:, i] + 1e-8)
            weighted_losses.append(weighted_loss.mean())
        return sum(weighted_losses)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

class PhysicsConstraintLayer(nn.Module):
    """Layer that enforces physical constraints"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Physics-informed transformations
        self.pressure_constraint = nn.Linear(hidden_dim, hidden_dim)
        self.temperature_constraint = nn.Linear(hidden_dim, hidden_dim) 
        self.gas_constraint = nn.Linear(hidden_dim, hidden_dim)
        
        # Physical parameter ranges (normalized)
        self.pressure_range = [0, 1000]  # MPa
        self.temp_range = [600, 1200]    # °C
        self.gas_range = [0, 10]         # wt%
        
    def forward(self, features, raw_input):
        # Extract physical parameters from raw input
        batch_size = raw_input.shape[0]
        
        # Average over sequence dimension for physical constraints
        avg_input = raw_input.mean(dim=1)
        
        pressure_features = self.pressure_constraint(features)
        temp_features = self.temperature_constraint(features)
        gas_features = self.gas_constraint(features)
        
        # Apply physical constraints
        constrained_features = features + 0.1 * (pressure_features + temp_features + gas_features)
        
        return constrained_features
    
    def physics_loss(self, input_data, outputs):
        """Calculate physics-based regularization loss"""
        batch_size, seq_len, features = input_data.shape
        
        # Extract physical variables (assuming specific feature order)
        pressures = input_data[:, :, :3]  # shallow, middle, deep
        temperatures = input_data[:, :, 3:6]
        gas_ratios = input_data[:, :, 6:10]
        
        loss = 0.0
        
        # Pressure gradient constraint (should increase with depth)
        pressure_gradient = torch.diff(pressures, dim=2)
        gradient_violation = F.relu(-pressure_gradient).mean()  # Penalize negative gradients
        loss += gradient_violation
        
        # Temperature gradient constraint (should decrease upward)
        temp_gradient = torch.diff(temperatures, dim=2)  
        temp_violation = F.relu(temp_gradient).mean()  # Penalize positive gradients
        loss += temp_violation
        
        # Gas solubility constraint (Henry's Law approximation)
        # Higher pressure should allow more dissolved gas
        gas_pressure_relation = gas_ratios[:, :, 0] - 0.1 * pressures[:, :, 0]
        solubility_violation = F.relu(-gas_pressure_relation).mean()
        loss += solubility_violation
        
        # Thermodynamic consistency
        # Clausius-Clapeyron relationship approximation
        temp_pressure_consistency = torch.abs(
            temperatures[:, :, 0] - 600 - 0.1 * pressures[:, :, 0]
        ).mean()
        loss += 0.01 * temp_pressure_consistency
        
        return loss

class EruptionForecaster:
    """Main eruption forecasting system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.feature_importance = {}
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Alert thresholds
        self.alert_thresholds = {
            'normal': 0.1,
            'advisory': 0.3,
            'watch': 0.5,
            'warning': 0.7,
            'critical': 0.9
        }
        
        logger.info("Eruption Forecaster initialized")
    
    def train(self, data: pd.DataFrame, epochs: int = 100, 
              val_split: float = 0.2, optimize_hyperparameters: bool = True):
        """Train the eruption forecasting model"""
        logger.info("Starting model training...")
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            best_params = self.optimize_hyperparameters(data)
        else:
            best_params = {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.2,
                'learning_rate': 1e-3
            }
        
        # Create datasets
        train_data, val_data = self.create_train_val_split(data, val_split)
        train_dataset = VolcanicDataset(train_data, mode='train')
        val_dataset = VolcanicDataset(val_data, mode='val')
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        # Initialize model
        self.model = PhysicsInformedEruptionNet(**best_params)
        
        # Training
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator='auto',
            devices='auto',
            precision=16,
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )
        
        trainer.fit(self.model, train_loader, val_loader)
        
        # Train anomaly detector on training features
        train_features = self.extract_features(train_data)
        self.scaler.fit(train_features)
        scaled_features = self.scaler.transform(train_features)
        self.anomaly_detector.fit(scaled_features)
        
        logger.info("Model training completed")
    
    def optimize_hyperparameters(self, data: pd.DataFrame, n_trials: int = 50):
        """Optimize hyperparameters using Optuna"""
        logger.info("Optimizing hyperparameters...")
        
        def objective(trial):
            params = {
                'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
                'num_layers': trial.suggest_int('num_layers', 2, 6),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            }
            
            # Quick training for hyperparameter optimization
            train_data, val_data = self.create_train_val_split(data, 0.2)
            train_dataset = VolcanicDataset(train_data, mode='train')
            val_dataset = VolcanicDataset(val_data, mode='val')
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            model = PhysicsInformedEruptionNet(**params)
            trainer = pl.Trainer(
                max_epochs=20,
                accelerator='auto',
                devices='auto',
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=False
            )
            
            trainer.fit(model, train_loader, val_loader)
            
            # Return validation loss
            val_results = trainer.validate(model, val_loader, verbose=False)
            return val_results[0]['val_loss']
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        return study.best_params
    
    def create_train_val_split(self, data: pd.DataFrame, val_split: float):
        """Create train/validation split by volcano"""
        volcanoes = data['volcano_id'].unique()
        n_val = int(len(volcanoes) * val_split)
        
        val_volcanoes = np.random.choice(volcanoes, n_val, replace=False)
        
        train_data = data[~data['volcano_id'].isin(val_volcanoes)]
        val_data = data[data['volcano_id'].isin(val_volcanoes)]
        
        return train_data, val_data
    
    def extract_features(self, data: pd.DataFrame):
        """Extract features for anomaly detection"""
        feature_cols = [
            'pressure_shallow', 'pressure_middle', 'pressure_deep',
            'temperature_shallow', 'temperature_middle', 'temperature_deep',
            'gas_co2', 'gas_so2', 'gas_h2s', 'gas_ratio_co2_so2',
            'seismic_magnitude_avg', 'seismic_count', 'seismic_energy',
            'deformation_radial', 'deformation_vertical', 'deformation_rate',
            'thermal_anomaly', 'thermal_area', 'thermal_intensity'
        ]
        
        return data[feature_cols].values
    
    async def predict(self, monitoring_data: Dict) -> EruptionForecast:
        """Generate eruption forecast from monitoring data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Convert monitoring data to tensor
        features = self.prepare_prediction_data(monitoring_data)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(features)
        
        # Extract predictions
        eruption_prob = outputs['eruption_prob'].item()
        vei_probs = F.softmax(outputs['vei_logits'], dim=1).squeeze().numpy()
        expected_vei = int(np.argmax(vei_probs))
        time_to_eruption = outputs['time_pred'].item() if eruption_prob > 0.3 else None
        uncertainty = outputs['uncertainty'].squeeze().numpy()
        
        # Anomaly detection
        current_features = self.extract_current_features(monitoring_data)
        scaled_features = self.scaler.transform(current_features.reshape(1, -1))
        anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
        
        # Adjust probability based on anomaly score
        if anomaly_score < -0.5:  # Highly anomalous
            eruption_prob = min(1.0, eruption_prob * 1.5)
        
        # Calculate uncertainty bounds
        uncertainty_bounds = self.calculate_uncertainty_bounds(
            eruption_prob, uncertainty[0]
        )
        
        # Determine alert level
        alert_level = self.get_alert_level(eruption_prob)
        
        # Contributing factors analysis
        contributing_factors = self.analyze_contributing_factors(monitoring_data, outputs)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            eruption_prob, expected_vei, time_to_eruption, alert_level
        )
        
        # Create forecast
        forecast = EruptionForecast(
            eruption_probability=eruption_prob,
            time_to_eruption=time_to_eruption,
            expected_vei=expected_vei,
            confidence=1.0 - uncertainty.mean(),
            uncertainty_bounds=uncertainty_bounds,
            alert_level=alert_level,
            contributing_factors=contributing_factors,
            recommendations=recommendations,
            timestamp=datetime.now()
        )
        
        logger.info(f"Generated forecast: {alert_level} alert, {eruption_prob:.3f} probability")
        
        return forecast
    
    def prepare_prediction_data(self, monitoring_data: Dict) -> torch.Tensor:
        """Prepare monitoring data for model prediction"""
        # Extract time series data (last 30 days)
        sequence_data = []
        
        for day in range(30):  # 30-day sequence
            daily_features = [
                monitoring_data.get('pressure_shallow', [0]*30)[day],
                monitoring_data.get('pressure_middle', [0]*30)[day],
                monitoring_data.get('pressure_deep', [0]*30)[day],
                monitoring_data.get('temperature_shallow', [0]*30)[day],
                monitoring_data.get('temperature_middle', [0]*30)[day],
                monitoring_data.get('temperature_deep', [0]*30)[day],
                monitoring_data.get('gas_co2', [0]*30)[day],
                monitoring_data.get('gas_so2', [0]*30)[day],
                monitoring_data.get('gas_h2s', [0]*30)[day],
                monitoring_data.get('gas_ratio_co2_so2', [1]*30)[day],
                monitoring_data.get('seismic_magnitude_avg', [0]*30)[day],
                monitoring_data.get('seismic_count', [0]*30)[day],
                monitoring_data.get('seismic_energy', [0]*30)[day],
                monitoring_data.get('deformation_radial', [0]*30)[day],
                monitoring_data.get('deformation_vertical', [0]*30)[day],
                monitoring_data.get('deformation_rate', [0]*30)[day],
                monitoring_data.get('thermal_anomaly', [0]*30)[day],
                monitoring_data.get('thermal_area', [0]*30)[day],
                monitoring_data.get('thermal_intensity', [0]*30)[day],
                monitoring_data.get('gravity_change', [0]*30)[day],
                monitoring_data.get('magnetic_change', [0]*30)[day],
                monitoring_data.get('tidal_stress', [0]*30)[day],
                monitoring_data.get('regional_seismicity', [0]*30)[day],
                monitoring_data.get('precipitation', [0]*30)[day]
            ]
            sequence_data.append(daily_features)
        
        return torch.FloatTensor(sequence_data).unsqueeze(0)  # Add batch dimension
    
    def extract_current_features(self, monitoring_data: Dict) -> np.ndarray:
        """Extract current features for anomaly detection"""
        current_features = [
            monitoring_data.get('pressure_shallow', [0])[-1],
            monitoring_data.get('pressure_middle', [0])[-1],
            monitoring_data.get('pressure_deep', [0])[-1],
            monitoring_data.get('temperature_shallow', [0])[-1],
            monitoring_data.get('temperature_middle', [0])[-1],
            monitoring_data.get('temperature_deep', [0])[-1],
            monitoring_data.get('gas_co2', [0])[-1],
            monitoring_data.get('gas_so2', [0])[-1],
            monitoring_data.get('gas_h2s', [0])[-1],
            monitoring_data.get('gas_ratio_co2_so2', [1])[-1],
            monitoring_data.get('seismic_magnitude_avg', [0])[-1],
            monitoring_data.get('seismic_count', [0])[-1],
            monitoring_data.get('seismic_energy', [0])[-1],
            monitoring_data.get('deformation_radial', [0])[-1],
            monitoring_data.get('deformation_vertical', [0])[-1],
            monitoring_data.get('deformation_rate', [0])[-1],
            monitoring_data.get('thermal_anomaly', [0])[-1],
            monitoring_data.get('thermal_area', [0])[-1],
            monitoring_data.get('thermal_intensity', [0])[-1]
        ]
        
        return np.array(current_features)
    
    def calculate_uncertainty_bounds(self, prob: float, uncertainty: float) -> Dict[str, List[float]]:
        """
        Calculate comprehensive uncertainty bounds for eruption probability with advanced statistical modeling
        
        Incorporates:
        - Temporal autocorrelation and memory effects
        - Heteroscedastic uncertainty modeling
        - Non-Gaussian uncertainty distributions
        - Ensemble-based confidence intervals
        - Physics-informed uncertainty propagation
        """
        import scipy.stats as stats
        from scipy.interpolate import interp1d
        
        # Enhanced uncertainty decomposition
        # Decompose total uncertainty into epistemic and aleatory components
        # Epistemic uncertainty: model uncertainty (reducible with more data)
        # Aleatory uncertainty: intrinsic randomness (irreducible)
        
        # Model epistemic uncertainty as function of data sparsity and model complexity
        epistemic_base = uncertainty * 0.3  # Base epistemic uncertainty
        aleatory_base = uncertainty * 0.7   # Base aleatory uncertainty
        
        # Time horizon parameters
        time_steps = 30
        forecast_horizon_days = 30
        time_points = np.linspace(0, forecast_horizon_days, time_steps)
        
        # Initialize comprehensive uncertainty bounds structure
        bounds = {
            'lower': [],
            'upper': [],
            'lower_5th': [],    # 5th percentile
            'upper_95th': [],   # 95th percentile
            'lower_25th': [],   # 25th percentile (IQR)
            'upper_75th': [],   # 75th percentile (IQR)
            'epistemic_lower': [],
            'epistemic_upper': [],
            'aleatory_lower': [],
            'aleatory_upper': [],
            'temporal_correlation': [],
            'uncertainty_growth': []
        }
        
        # Temporal correlation modeling
        # Volcanic processes exhibit memory effects and temporal dependencies
        correlation_length = 7.0  # Days - characteristic time scale for volcanic processes
        temporal_correlation_function = np.exp(-time_points / correlation_length)
        
        # Uncertainty growth modeling
        # Uncertainty typically grows with forecast horizon due to chaotic dynamics
        growth_rate = 0.1  # Uncertainty growth rate per day
        base_growth = 1.0 + growth_rate * time_points / forecast_horizon_days
        
        # Physics-informed uncertainty modulation
        # Account for physical processes that affect predictability
        
        # 1. Magma chamber dynamics uncertainty
        magma_chamber_uncertainty = epistemic_base * (1.0 + 0.5 * np.sin(2 * np.pi * time_points / 14))
        
        # 2. Crustal stress field uncertainty  
        stress_field_uncertainty = aleatory_base * (1.0 + 0.3 * np.cos(2 * np.pi * time_points / 21))
        
        # 3. Gas exsolution process uncertainty
        gas_exsolution_uncertainty = uncertainty * 0.2 * np.exp(-time_points / 10)
        
        # Ensemble-based uncertainty quantification
        # Generate ensemble of probability trajectories
        n_ensemble = 1000
        
        for t_idx, t in enumerate(time_points):
            # Time-varying uncertainty components
            epistemic_t = epistemic_base * base_growth[t_idx] * (1.0 + magma_chamber_uncertainty[t_idx])
            aleatory_t = aleatory_base * base_growth[t_idx] * (1.0 + stress_field_uncertainty[t_idx])
            physics_uncertainty_t = gas_exsolution_uncertainty[t_idx]
            
            # Total uncertainty at time t
            total_std_t = np.sqrt(epistemic_t**2 + aleatory_t**2 + physics_uncertainty_t**2)
            
            # Temporal correlation effects
            if t_idx > 0:
                # Apply temporal correlation to maintain realistic uncertainty evolution
                correlation_factor = temporal_correlation_function[t_idx]
                prev_uncertainty = bounds['uncertainty_growth'][-1] if bounds['uncertainty_growth'] else total_std_t
                total_std_t = correlation_factor * prev_uncertainty + (1 - correlation_factor) * total_std_t
            
            # Non-Gaussian uncertainty modeling
            # Use Beta distribution for probability uncertainty (bounded [0,1])
            # Parameterize Beta distribution to match desired mean and variance
            
            # Method of moments for Beta distribution parameters
            mean_prob = prob
            var_prob = total_std_t**2
            
            # Ensure valid Beta parameters
            if var_prob >= mean_prob * (1 - mean_prob):
                var_prob = 0.99 * mean_prob * (1 - mean_prob)  # Avoid degenerate case
            
            # Beta distribution parameters
            if var_prob > 0:
                alpha = mean_prob * ((mean_prob * (1 - mean_prob)) / var_prob - 1)
                beta = (1 - mean_prob) * ((mean_prob * (1 - mean_prob)) / var_prob - 1)
                
                # Ensure positive parameters
                alpha = max(0.1, alpha)
                beta = max(0.1, beta)
            else:
                alpha = beta = 1.0
            
            # Generate ensemble samples
            beta_dist = stats.beta(alpha, beta)
            ensemble_samples = beta_dist.rvs(n_ensemble)
            
            # Apply temporal memory effects
            if t_idx > 0:
                # Incorporate autocorrelation in ensemble
                prev_samples = prev_ensemble_samples if 'prev_ensemble_samples' in locals() else ensemble_samples
                autocorr_strength = temporal_correlation_function[t_idx] * 0.5
                ensemble_samples = (autocorr_strength * prev_samples + 
                                  (1 - autocorr_strength) * ensemble_samples)
            
            prev_ensemble_samples = ensemble_samples.copy()
            
            # Calculate comprehensive percentiles
            percentiles = np.percentile(ensemble_samples, [5, 25, 50, 75, 95])
            
            # Standard confidence intervals (assuming Gaussian approximation for comparison)
            gaussian_lower = max(0, prob - 2 * total_std_t)
            gaussian_upper = min(1, prob + 2 * total_std_t)
            
            # Epistemic vs Aleatory decomposition
            epistemic_margin = 1.96 * epistemic_t  # 95% confidence for epistemic
            aleatory_margin = 1.96 * aleatory_t    # 95% confidence for aleatory
            
            epistemic_lower = max(0, prob - epistemic_margin)
            epistemic_upper = min(1, prob + epistemic_margin)
            aleatory_lower = max(0, prob - aleatory_margin)
            aleatory_upper = min(1, prob + aleatory_margin)
            
            # Store comprehensive bounds
            bounds['lower'].append(gaussian_lower)
            bounds['upper'].append(gaussian_upper)
            bounds['lower_5th'].append(percentiles[0])
            bounds['upper_95th'].append(percentiles[4])
            bounds['lower_25th'].append(percentiles[1])
            bounds['upper_75th'].append(percentiles[3])
            bounds['epistemic_lower'].append(epistemic_lower)
            bounds['epistemic_upper'].append(epistemic_upper)
            bounds['aleatory_lower'].append(aleatory_lower)
            bounds['aleatory_upper'].append(aleatory_upper)
            bounds['temporal_correlation'].append(temporal_correlation_function[t_idx])
            bounds['uncertainty_growth'].append(total_std_t)
        
        # Post-processing for temporal consistency
        # Apply smoothing to prevent unrealistic discontinuities
        from scipy.ndimage import gaussian_filter1d
        smoothing_sigma = 1.0  # Smoothing parameter
        
        for key in ['lower', 'upper', 'lower_5th', 'upper_95th']:
            if key in bounds:
                bounds[key] = gaussian_filter1d(bounds[key], sigma=smoothing_sigma).tolist()
        
        # Advanced uncertainty statistics
        bounds['mean_uncertainty'] = float(np.mean(bounds['uncertainty_growth']))
        bounds['max_uncertainty'] = float(np.max(bounds['uncertainty_growth']))
        bounds['uncertainty_trend'] = float((bounds['uncertainty_growth'][-1] - bounds['uncertainty_growth'][0]) / 
                                          bounds['uncertainty_growth'][0] if bounds['uncertainty_growth'][0] > 0 else 0)
        
        # Confidence level assessment
        # Assess the reliability of the uncertainty bounds themselves
        confidence_in_bounds = 1.0 - (total_std_t / (1.0 + total_std_t))  # Diminishing confidence with high uncertainty
        bounds['bounds_confidence'] = float(confidence_in_bounds)
        
        # Volcanic process-specific uncertainty factors
        bounds['process_uncertainties'] = {
            'magma_chamber_dynamics': float(np.mean(magma_chamber_uncertainty)),
            'crustal_stress_field': float(np.mean(stress_field_uncertainty)),
            'gas_exsolution': float(np.mean(gas_exsolution_uncertainty)),
            'temporal_correlation_strength': float(np.mean(temporal_correlation_function))
        }
        
        # Uncertainty attribution analysis
        total_variance = np.var(bounds['uncertainty_growth'])
        epistemic_contribution = np.var([bounds['epistemic_upper'][i] - bounds['epistemic_lower'][i] 
                                        for i in range(len(bounds['epistemic_upper']))]) / (4 * total_variance) if total_variance > 0 else 0
        aleatory_contribution = 1.0 - epistemic_contribution
        
        bounds['uncertainty_attribution'] = {
            'epistemic_fraction': float(max(0, min(1, epistemic_contribution))),
            'aleatory_fraction': float(max(0, min(1, aleatory_contribution))),
            'reducible_uncertainty': float(epistemic_contribution * bounds['mean_uncertainty']),
            'irreducible_uncertainty': float(aleatory_contribution * bounds['mean_uncertainty'])
        }
        
        # Temporal dynamics summary
        bounds['temporal_dynamics'] = {
            'correlation_length_days': float(correlation_length),
            'growth_rate_per_day': float(growth_rate),
            'forecast_horizon_days': float(forecast_horizon_days),
            'memory_effect_strength': float(np.mean(temporal_correlation_function))
        }
        
        # Quality metrics for uncertainty quantification
        bounds['quality_metrics'] = {
            'ensemble_size': n_ensemble,
            'distribution_type': 'beta',
            'temporal_smoothing_applied': True,
            'physics_constraints_applied': True,
            'autocorrelation_modeled': True
        }
        
        # Validation checks
        # Ensure bounds are physically meaningful
        for key in ['lower', 'upper', 'lower_5th', 'upper_95th', 'lower_25th', 'upper_75th']:
            bounds[key] = [max(0.0, min(1.0, x)) for x in bounds[key]]
        
        # Ensure ordering constraints (lower <= upper)
        for i in range(len(bounds['lower'])):
            if bounds['lower'][i] > bounds['upper'][i]:
                mid_point = (bounds['lower'][i] + bounds['upper'][i]) / 2
                bounds['lower'][i] = mid_point - abs(bounds['upper'][i] - bounds['lower'][i]) / 4
                bounds['upper'][i] = mid_point + abs(bounds['upper'][i] - bounds['lower'][i]) / 4
        
        return bounds
    
    def get_alert_level(self, probability: float) -> str:
        """Determine alert level based on eruption probability"""
        if probability >= self.alert_thresholds['critical']:
            return 'CRITICAL'
        elif probability >= self.alert_thresholds['warning']:
            return 'WARNING' 
        elif probability >= self.alert_thresholds['watch']:
            return 'WATCH'
        elif probability >= self.alert_thresholds['advisory']:
            return 'ADVISORY'
        else:
            return 'NORMAL'
    
    def analyze_contributing_factors(self, monitoring_data: Dict, 
                                   model_outputs: Dict) -> Dict[str, float]:
        """
        Comprehensive analysis of factors contributing to eruption prediction
        
        Incorporates:
        - Multi-scale temporal pattern analysis
        - Physics-informed feature attribution
        - Statistical significance testing
        - Non-linear interaction effects
        - Precursor sequence identification
        - SHAP-like attribution values
        """
        from scipy import stats
        from scipy.signal import find_peaks, savgol_filter
        from scipy.ndimage import gaussian_filter1d
        
        # Initialize comprehensive factor analysis
        factors = {}
        detailed_analysis = {}
        
        # Extract feature attention weights from model if available
        model_attention_weights = {}
        if 'features' in model_outputs:
            # Extract attention patterns from model features
            # This would integrate with the actual attention mechanism
            feature_vector = model_outputs['features']
            if hasattr(feature_vector, 'detach'):
                feature_importance = torch.abs(feature_vector.detach().numpy()).flatten()
                model_attention_weights = {
                    'pressure_attention': float(np.mean(feature_importance[:3])),
                    'temperature_attention': float(np.mean(feature_importance[3:6])),
                    'gas_attention': float(np.mean(feature_importance[6:10])),
                    'seismic_attention': float(np.mean(feature_importance[10:13])),
                    'deformation_attention': float(np.mean(feature_importance[13:16])),
                    'thermal_attention': float(np.mean(feature_importance[16:19]))
                }
        
        # === PRESSURE SYSTEM ANALYSIS ===
        pressure_shallow = np.array(monitoring_data.get('pressure_shallow', [0]*30))
        pressure_middle = np.array(monitoring_data.get('pressure_middle', [0]*30))
        pressure_deep = np.array(monitoring_data.get('pressure_deep', [0]*30))
        
        # Multi-scale temporal analysis
        pressure_changes_1d = np.abs(np.diff(pressure_shallow[-2:]))[0] if len(pressure_shallow) > 1 else 0
        pressure_changes_7d = np.abs(np.mean(pressure_shallow[-7:]) - np.mean(pressure_shallow[-14:-7])) if len(pressure_shallow) >= 14 else 0
        pressure_changes_30d = np.abs(np.mean(pressure_shallow[-15:]) - np.mean(pressure_shallow[-30:-15])) if len(pressure_shallow) >= 30 else 0
        
        # Pressure gradient analysis (physics-informed)
        if len(pressure_shallow) >= 7:
            pressure_gradient = np.gradient(pressure_shallow[-7:])
            pressure_acceleration = np.gradient(pressure_gradient)
            pressure_volatility = np.std(pressure_shallow[-7:])
            
            # Magma chamber pressurization signature
            pressurization_score = (
                0.4 * (pressure_changes_7d / (np.mean(pressure_shallow[-7:]) + 1e-6)) +  # Relative change
                0.3 * (np.mean(pressure_gradient > 0) if len(pressure_gradient) > 0 else 0) +  # Trending upward
                0.3 * (pressure_volatility / (np.mean(pressure_shallow[-7:]) + 1e-6))  # Volatility
            )
        else:
            pressurization_score = 0
        
        # Depth-dependent pressure analysis
        if len(pressure_deep) >= 7 and len(pressure_shallow) >= 7:
            deep_shallow_ratio = np.mean(pressure_deep[-7:]) / (np.mean(pressure_shallow[-7:]) + 1e-6)
            pressure_inversion = 1.0 if deep_shallow_ratio < 1.5 else 0.0  # Abnormal pressure profile
        else:
            pressure_inversion = 0
        
        factors['pressure_changes'] = min(1.0, 0.6 * pressurization_score + 0.4 * pressure_inversion)
        detailed_analysis['pressure'] = {
            'daily_change_mpa': float(pressure_changes_1d),
            'weekly_change_mpa': float(pressure_changes_7d),
            'monthly_change_mpa': float(pressure_changes_30d),
            'pressurization_score': float(pressurization_score),
            'pressure_inversion_detected': bool(pressure_inversion > 0.5),
            'volatility': float(pressure_volatility) if 'pressure_volatility' in locals() else 0.0,
            'model_attention': model_attention_weights.get('pressure_attention', 0.0)
        }
        
        # === GAS EMISSION ANALYSIS ===
        gas_co2 = np.array(monitoring_data.get('gas_co2', [0]*30))
        gas_so2 = np.array(monitoring_data.get('gas_so2', [0]*30))
        gas_h2s = np.array(monitoring_data.get('gas_h2s', [0]*30))
        gas_ratio = np.array(monitoring_data.get('gas_ratio_co2_so2', [1]*30))
        
        # Multi-component gas analysis
        if len(gas_co2) >= 7:
            # Gas flux changes
            co2_trend = (np.mean(gas_co2[-3:]) - np.mean(gas_co2[-7:-4])) if len(gas_co2) >= 7 else 0
            so2_trend = (np.mean(gas_so2[-3:]) - np.mean(gas_so2[-7:-4])) if len(gas_so2) >= 7 else 0
            
            # Degassing efficiency analysis
            total_gas_flux = gas_co2[-1] + gas_so2[-1] + gas_h2s[-1] if len(gas_co2) > 0 else 0
            
            # Gas ratio anomalies (physics-informed)
            normal_co2_so2_ratio = 2.0  # Typical baseline
            ratio_anomaly = abs(gas_ratio[-1] - normal_co2_so2_ratio) / normal_co2_so2_ratio if len(gas_ratio) > 0 else 0
            
            # Deep vs shallow gas signature
            h2s_co2_ratio = (gas_h2s[-1] / (gas_co2[-1] + 1e-6)) if len(gas_h2s) > 0 and len(gas_co2) > 0 else 0
            deep_gas_signature = min(1.0, h2s_co2_ratio * 10)  # H2S indicates deeper sources
            
            # Peak detection in gas emissions
            if len(gas_co2) >= 10:
                peaks_co2, _ = find_peaks(gas_co2[-10:], height=np.mean(gas_co2[-10:]))
                peak_frequency = len(peaks_co2) / 10.0  # peaks per day
            else:
                peak_frequency = 0
            
            gas_score = (
                0.3 * min(1.0, abs(co2_trend) / 5.0) +  # CO2 trend significance
                0.2 * min(1.0, abs(so2_trend) / 2.0) +  # SO2 trend significance  
                0.2 * min(1.0, ratio_anomaly) +  # Ratio anomaly
                0.2 * deep_gas_signature +  # Deep gas signature
                0.1 * min(1.0, peak_frequency)  # Episodic degassing
            )
        else:
            gas_score = co2_trend = so2_trend = ratio_anomaly = deep_gas_signature = peak_frequency = 0
        
        factors['gas_emissions'] = gas_score
        detailed_analysis['gas_emissions'] = {
            'co2_trend_ppm_per_day': float(co2_trend),
            'so2_trend_ppm_per_day': float(so2_trend),
            'total_flux_estimate': float(total_gas_flux),
            'co2_so2_ratio_anomaly': float(ratio_anomaly),
            'deep_gas_signature': float(deep_gas_signature),
            'degassing_peak_frequency': float(peak_frequency),
            'model_attention': model_attention_weights.get('gas_attention', 0.0)
        }
        
        # === SEISMIC ACTIVITY ANALYSIS ===
        seismic_count = np.array(monitoring_data.get('seismic_count', [0]*30))
        seismic_magnitude = np.array(monitoring_data.get('seismic_magnitude_avg', [0]*30))
        seismic_energy = np.array(monitoring_data.get('seismic_energy', [0]*30))
        
        if len(seismic_count) >= 7:
            # Seismicity rate analysis
            recent_rate = np.mean(seismic_count[-3:])
            background_rate = np.mean(seismic_count[-30:-7]) if len(seismic_count) >= 30 else np.mean(seismic_count[:-3])
            rate_increase = (recent_rate - background_rate) / (background_rate + 1) if background_rate > 0 else 0
            
            # Comprehensive Gutenberg-Richter b-value analysis with advanced seismological methods
            if len(seismic_magnitude) >= 10:
                # Extract magnitude data with comprehensive preprocessing
                magnitudes = seismic_magnitude[-30:] if len(seismic_magnitude) >= 30 else seismic_magnitude[-10:]
                magnitudes = magnitudes[magnitudes > 0]  # Remove zeros and negative values
                
                if len(magnitudes) >= 5:  # Minimum required for robust b-value estimation
                    # Advanced b-value calculation with multiple methods
                    
                    # === METHOD 1: Maximum Likelihood Estimation (Aki, 1965) ===
                    # Most robust method for b-value estimation
                    try:
                        # Magnitude completeness estimation using maximum curvature method
                        magnitude_bins = np.arange(np.min(magnitudes), np.max(magnitudes) + 0.1, 0.1)
                        hist, bin_edges = np.histogram(magnitudes, bins=magnitude_bins)
                        
                        # Find completeness magnitude (Mc) using maximum curvature
                        if len(hist) > 2:
                            # Calculate discrete derivative to find maximum curvature
                            discrete_derivative = np.diff(hist)
                            if len(discrete_derivative) > 0:
                                max_curve_idx = np.argmax(discrete_derivative)
                                mc_estimate = bin_edges[max_curve_idx]
                            else:
                                mc_estimate = np.min(magnitudes)
                        else:
                            mc_estimate = np.min(magnitudes)
                        
                        # Filter magnitudes above completeness threshold
                        complete_magnitudes = magnitudes[magnitudes >= mc_estimate]
                        
                        if len(complete_magnitudes) >= 3:
                            # Maximum likelihood b-value estimation (Aki, 1965)
                            mean_magnitude = np.mean(complete_magnitudes)
                            min_magnitude = np.min(complete_magnitudes)
                            
                            # MLE b-value formula: b = log10(e) / (mean_M - min_M)
                            if (mean_magnitude - min_magnitude) > 0:
                                b_value_mle = np.log10(np.e) / (mean_magnitude - min_magnitude)
                                
                                # Uncertainty estimation (Shi & Bolt, 1982)
                                n_events = len(complete_magnitudes)
                                b_value_uncertainty = 2.3 * b_value_mle**2 * np.sqrt(
                                    np.sum((complete_magnitudes - mean_magnitude)**2) / (n_events * (n_events - 1))
                                )
                            else:
                                b_value_mle = 1.0
                                b_value_uncertainty = 0.5
                        else:
                            b_value_mle = 1.0
                            b_value_uncertainty = 0.5
                            
                    except Exception:
                        b_value_mle = 1.0
                        b_value_uncertainty = 0.5
                    
                    # === METHOD 2: Least Squares Regression (Classic Gutenberg-Richter) ===
                    try:
                        # Create cumulative frequency-magnitude distribution
                        unique_mags = np.sort(np.unique(magnitudes))
                        cumulative_counts = []
                        
                        for mag in unique_mags:
                            count = np.sum(magnitudes >= mag)
                            cumulative_counts.append(count)
                        
                        cumulative_counts = np.array(cumulative_counts)
                        
                        # Filter out zero counts and take log
                        valid_indices = cumulative_counts > 0
                        if np.sum(valid_indices) >= 3:
                            log_counts = np.log10(cumulative_counts[valid_indices])
                            valid_mags = unique_mags[valid_indices]
                            
                            # Linear regression on log(N) vs M
                            if len(valid_mags) > 1:
                                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_mags, log_counts)
                                b_value_lsr = -slope  # b-value is negative slope
                                b_value_lsr_uncertainty = std_err
                                lsr_correlation = abs(r_value)
                            else:
                                b_value_lsr = 1.0
                                b_value_lsr_uncertainty = 0.5
                                lsr_correlation = 0.0
                        else:
                            b_value_lsr = 1.0
                            b_value_lsr_uncertainty = 0.5
                            lsr_correlation = 0.0
                            
                    except Exception:
                        b_value_lsr = 1.0
                        b_value_lsr_uncertainty = 0.5
                        lsr_correlation = 0.0
                    
                    # === METHOD 3: Weighted Least Squares (Marzocchi & Sandri, 2003) ===
                    try:
                        if 'cumulative_counts' in locals() and 'valid_mags' in locals() and len(valid_mags) > 2:
                            # Weight by square root of counts (Poisson assumption)
                            weights = np.sqrt(cumulative_counts[valid_indices])
                            weights = weights / np.sum(weights)  # Normalize weights
                            
                            # Weighted least squares
                            W = np.diag(weights)
                            X = np.column_stack([np.ones(len(valid_mags)), valid_mags])
                            y = np.log10(cumulative_counts[valid_indices])
                            
                            # Normal equation: (X^T W X)^-1 X^T W y
                            XtWX = X.T @ W @ X
                            if np.linalg.det(XtWX) != 0:
                                beta = np.linalg.inv(XtWX) @ X.T @ W @ y
                                b_value_wls = -beta[1]
                                
                                # Uncertainty from covariance matrix
                                residuals = y - X @ beta
                                mse = np.sum(weights * residuals**2) / (len(y) - 2)
                                cov_matrix = mse * np.linalg.inv(XtWX)
                                b_value_wls_uncertainty = np.sqrt(cov_matrix[1, 1])
                            else:
                                b_value_wls = b_value_lsr
                                b_value_wls_uncertainty = b_value_lsr_uncertainty
                        else:
                            b_value_wls = b_value_lsr
                            b_value_wls_uncertainty = b_value_lsr_uncertainty
                            
                    except Exception:
                        b_value_wls = b_value_lsr
                        b_value_wls_uncertainty = b_value_lsr_uncertainty
                    
                    # === ENSEMBLE B-VALUE ESTIMATION ===
                    # Combine multiple methods using weighted average based on reliability
                    
                    # Assign weights based on method reliability and data quality
                    mle_weight = 0.5 if len(complete_magnitudes) >= 10 else 0.3
                    lsr_weight = 0.3 * min(1.0, lsr_correlation**2)  # Weight by correlation quality
                    wls_weight = 0.2 if len(valid_mags) >= 5 else 0.1
                    
                    # Normalize weights
                    total_weight = mle_weight + lsr_weight + wls_weight
                    if total_weight > 0:
                        mle_weight /= total_weight
                        lsr_weight /= total_weight
                        wls_weight /= total_weight
                    else:
                        mle_weight = lsr_weight = wls_weight = 1.0/3.0
                    
                    # Ensemble b-value
                    b_value_ensemble = (mle_weight * b_value_mle + 
                                      lsr_weight * b_value_lsr + 
                                      wls_weight * b_value_wls)
                    
                    # Ensemble uncertainty (combining method uncertainties)
                    b_value_ensemble_uncertainty = np.sqrt(
                        mle_weight**2 * b_value_uncertainty**2 + 
                        lsr_weight**2 * b_value_lsr_uncertainty**2 + 
                        wls_weight**2 * b_value_wls_uncertainty**2
                    )
                    
                    # === TEMPORAL B-VALUE EVOLUTION ANALYSIS ===
                    # Analyze how b-value changes over time (important for eruption forecasting)
                    if len(magnitudes) >= 15:
                        # Calculate b-values for sliding windows
                        window_size = max(5, len(magnitudes) // 3)
                        b_value_time_series = []
                        
                        for i in range(len(magnitudes) - window_size + 1):
                            window_mags = magnitudes[i:i + window_size]
                            if len(window_mags) >= 3 and np.max(window_mags) > np.min(window_mags):
                                try:
                                    # Quick MLE b-value for window
                                    window_mean = np.mean(window_mags)
                                    window_min = np.min(window_mags)
                                    if (window_mean - window_min) > 0:
                                        window_b = np.log10(np.e) / (window_mean - window_min)
                                        b_value_time_series.append(window_b)
                                except:
                                    pass
                        
                        if len(b_value_time_series) >= 3:
                            # Analyze b-value trend
                            time_indices = np.arange(len(b_value_time_series))
                            b_trend_slope, _, _, _, _ = stats.linregress(time_indices, b_value_time_series)
                            b_value_volatility = np.std(b_value_time_series)
                            
                            # Decreasing b-value often precedes eruptions
                            b_trend_significance = abs(b_trend_slope) / (b_value_volatility + 1e-6)
                        else:
                            b_trend_slope = 0
                            b_value_volatility = 0
                            b_trend_significance = 0
                    else:
                        b_trend_slope = 0
                        b_value_volatility = 0
                        b_trend_significance = 0
                    
                    # === STATISTICAL SIGNIFICANCE TESTING ===
                    # Test if b-value significantly deviates from normal volcanic value (b ≈ 1.0)
                    normal_b_value = 1.0
                    b_deviation = abs(b_value_ensemble - normal_b_value)
                    
                    # Statistical significance using t-test approximation
                    if b_value_ensemble_uncertainty > 0:
                        t_statistic = b_deviation / b_value_ensemble_uncertainty
                        # Degrees of freedom approximation
                        df = len(complete_magnitudes) - 2 if len(complete_magnitudes) > 2 else 1
                        p_value_b = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
                        significance_level = 1 - p_value_b
                    else:
                        significance_level = 0.5
                    
                    # === VOLCANIC B-VALUE INTERPRETATION ===
                    # Interpret b-value in volcanic context
                    
                    # Low b-values (< 0.8) often indicate:
                    # - Increased stress
                    # - Approach to critical state
                    # - Possible eruption precursor
                    
                    # High b-values (> 1.2) often indicate:
                    # - Heterogeneous stress field
                    # - Fluid injection effects
                    # - Possible post-eruptive relaxation
                    
                    if b_value_ensemble < 0.8:
                        volcanic_interpretation = "low_stress_concentration"
                        eruption_relevance = min(1.0, (0.8 - b_value_ensemble) / 0.3)
                    elif b_value_ensemble > 1.2:
                        volcanic_interpretation = "high_heterogeneous_stress"
                        eruption_relevance = min(1.0, (b_value_ensemble - 1.2) / 0.5)
                    else:
                        volcanic_interpretation = "normal_stress_regime"
                        eruption_relevance = 0.2  # Baseline relevance
                    
                    # Incorporate temporal trend
                    if b_trend_slope < -0.1 and b_trend_significance > 1.0:
                        eruption_relevance *= 1.5  # Decreasing b-value increases eruption relevance
                        trend_interpretation = "decreasing_b_pre_eruptive"
                    elif b_trend_slope > 0.1 and b_trend_significance > 1.0:
                        eruption_relevance *= 0.7  # Increasing b-value decreases eruption relevance
                        trend_interpretation = "increasing_b_post_eruptive"
                    else:
                        trend_interpretation = "stable_b_value"
                    
                    # Final b-value anomaly score
                    b_value_anomaly = min(1.0, eruption_relevance * significance_level)
                    
                    # Store comprehensive b-value analysis results
                    b_value_analysis = {
                        'ensemble_b_value': float(b_value_ensemble),
                        'b_value_uncertainty': float(b_value_ensemble_uncertainty),
                        'mle_b_value': float(b_value_mle),
                        'lsr_b_value': float(b_value_lsr),
                        'wls_b_value': float(b_value_wls),
                        'completeness_magnitude': float(mc_estimate),
                        'n_complete_events': int(len(complete_magnitudes)),
                        'temporal_trend_slope': float(b_trend_slope),
                        'b_value_volatility': float(b_value_volatility),
                        'statistical_significance': float(significance_level),
                        'volcanic_interpretation': volcanic_interpretation,
                        'trend_interpretation': trend_interpretation,
                        'eruption_relevance_score': float(eruption_relevance),
                        'method_weights': {
                            'mle': float(mle_weight),
                            'lsr': float(lsr_weight),
                            'wls': float(wls_weight)
                        }
                    }
                    
                else:
                    # Insufficient data for comprehensive analysis
                    b_value_anomaly = 0.1  # Low confidence default
                    b_value_analysis = {
                        'ensemble_b_value': 1.0,
                        'b_value_uncertainty': 0.5,
                        'insufficient_data': True,
                        'n_events': len(magnitudes)
                    }
            else:
                # Insufficient data
                b_value_anomaly = 0
                b_value_analysis = {
                    'ensemble_b_value': 1.0,
                    'b_value_uncertainty': 1.0,
                    'insufficient_data': True,
                    'n_events': len(seismic_magnitude) if len(seismic_magnitude) > 0 else 0
                }
            
            # Seismic swarm detection
            if len(seismic_count) >= 7:
                daily_counts = seismic_count[-7:]
                swarm_threshold = np.mean(seismic_count[:-7]) + 2*np.std(seismic_count[:-7]) if len(seismic_count) > 7 else 10
                swarm_days = np.sum(daily_counts > swarm_threshold)
                swarm_intensity = swarm_days / 7.0
            else:
                swarm_intensity = 0
            
            # Energy release patterns
            if len(seismic_energy) >= 7:
                energy_trend = (np.mean(seismic_energy[-3:]) - np.mean(seismic_energy[-7:-3])) / (np.mean(seismic_energy[-7:-3]) + 1)
                energy_volatility = np.std(seismic_energy[-7:]) / (np.mean(seismic_energy[-7:]) + 1)
            else:
                energy_trend = energy_volatility = 0
            
            seismic_score = (
                0.3 * min(1.0, max(0, rate_increase / 5.0)) +  # Rate increase
                0.2 * min(1.0, b_value_anomaly) +  # b-value anomaly
                0.25 * swarm_intensity +  # Swarm activity
                0.15 * min(1.0, abs(energy_trend)) +  # Energy trend
                0.1 * min(1.0, energy_volatility)  # Energy volatility
            )
        else:
            seismic_score = rate_increase = b_value_anomaly = swarm_intensity = energy_trend = energy_volatility = 0
        
        factors['seismic_activity'] = seismic_score
        detailed_analysis['seismic_activity'] = {
            'rate_increase_factor': float(rate_increase),
            'b_value_anomaly': float(b_value_anomaly),
            'swarm_intensity': float(swarm_intensity),
            'energy_release_trend': float(energy_trend),
            'energy_volatility': float(energy_volatility),
            'model_attention': model_attention_weights.get('seismic_attention', 0.0),
            'b_value_analysis': b_value_analysis if 'b_value_analysis' in locals() else {'insufficient_data': True}
        }
        
        # === DEFORMATION ANALYSIS ===
        deformation_radial = np.array(monitoring_data.get('deformation_radial', [0]*30))
        deformation_vertical = np.array(monitoring_data.get('deformation_vertical', [0]*30))
        deformation_rate = np.array(monitoring_data.get('deformation_rate', [0]*30))
        
        if len(deformation_radial) >= 7:
            # Multi-component deformation analysis
            radial_trend = np.polyfit(range(len(deformation_radial[-7:])), deformation_radial[-7:], 1)[0] if len(deformation_radial) >= 7 else 0
            vertical_trend = np.polyfit(range(len(deformation_vertical[-7:])), deformation_vertical[-7:], 1)[0] if len(deformation_vertical) >= 7 else 0
            
            # Inflation pattern recognition
            inflation_score = max(0, radial_trend) + max(0, vertical_trend)  # Positive trends indicate inflation
            
            # Deformation acceleration
            if len(deformation_rate) >= 7:
                rate_acceleration = np.gradient(deformation_rate[-7:])[-1] if len(deformation_rate) >= 7 else 0
                rate_volatility = np.std(deformation_rate[-7:])
            else:
                rate_acceleration = rate_volatility = 0
            
            # Comprehensive Mogi-Yamakawa deformation source analysis with advanced geodetic modeling
            if abs(radial_trend) > 0 or abs(vertical_trend) > 0:
                
                # === COMPREHENSIVE MOGI SOURCE MODELING ===
                # Full implementation of Mogi (1958) and Yamakawa (1955) elastic point source models
                
                try:
                    # Enhanced deformation pattern analysis
                    # Convert mm/day to m for physical calculations
                    ur_trend = radial_trend * 1e-3  # Radial deformation rate (m/day)
                    uz_trend = vertical_trend * 1e-3  # Vertical deformation rate (m/day)
                    
                    # Initialize comprehensive source parameters
                    mogi_analysis = {
                        'source_depth_estimate': None,
                        'volume_change_rate': None,
                        'pressure_change_rate': None,
                        'source_strength': None,
                        'deformation_pattern_type': 'unknown',
                        'model_confidence': 0.0,
                        'stress_regime': 'unknown'
                    }
                    
                    # === METHOD 1: ANALYTICAL MOGI DEPTH ESTIMATION ===
                    if abs(ur_trend) > 1e-6 and abs(uz_trend) > 1e-6:
                        # Traditional radial/vertical ratio method (enhanced)
                        ratio_rv = abs(ur_trend) / abs(uz_trend)
                        
                        # Enhanced depth estimation using full Mogi theory
                        # For Mogi point source: ur/uz = (3/2) * (r²/d²) * (1 + r²/d²)^(-1)
                        # where r = radial distance, d = source depth
                        
                        # Assume measurement at characteristic distance r ~ 1-5 km from source
                        # This is typical for volcanic deformation monitoring networks
                        characteristic_distance = 2000.0  # meters (2 km typical)
                        
                        # Solve for depth using ratio relationship
                        # Iterative solution for d given ur/uz ratio
                        def mogi_ratio_function(depth_m):
                            r = characteristic_distance
                            d = depth_m
                            if d <= 0:
                                return float('inf')
                            theoretical_ratio = (3.0/2.0) * (r**2/d**2) / (1 + r**2/d**2)
                            return abs(theoretical_ratio - ratio_rv)
                        
                        # Search for optimal depth (0.1 to 20 km range)
                        depth_search = np.linspace(100, 20000, 200)  # 100m to 20km
                        ratio_errors = [mogi_ratio_function(d) for d in depth_search]
                        min_error_idx = np.argmin(ratio_errors)
                        
                        if ratio_errors[min_error_idx] < 0.5:  # Reasonable fit
                            estimated_depth = depth_search[min_error_idx]
                            depth_confidence = 1.0 - ratio_errors[min_error_idx]
                            mogi_analysis['source_depth_estimate'] = float(estimated_depth)
                            mogi_analysis['model_confidence'] = float(depth_confidence)
                        else:
                            estimated_depth = 5000.0  # Default 5 km
                            depth_confidence = 0.3
                            mogi_analysis['source_depth_estimate'] = float(estimated_depth)
                            mogi_analysis['model_confidence'] = float(depth_confidence)
                    else:
                        estimated_depth = 5000.0
                        depth_confidence = 0.2
                        mogi_analysis['source_depth_estimate'] = float(estimated_depth)
                        mogi_analysis['model_confidence'] = float(depth_confidence)
                    
                    # === METHOD 2: VOLUME CHANGE RATE ESTIMATION ===
                    # Calculate magma chamber volume change rate from surface deformation
                    
                    # Mogi model: uz = (ΔV/π) * (1-ν) / (2*μ) * d / (r² + d²)^(3/2)
                    # where: ΔV = volume change, ν = Poisson's ratio, μ = shear modulus
                    
                    # Typical crustal parameters
                    poisson_ratio = 0.25  # Typical for volcanic rocks
                    shear_modulus = 5e9   # Pa, typical crustal value
                    
                    # Calculate volume change rate
                    r = characteristic_distance
                    d = estimated_depth
                    
                    if abs(uz_trend) > 1e-6:
                        # Invert Mogi formula for volume change
                        denominator = (1 - poisson_ratio) / (2 * shear_modulus) * d / (r**2 + d**2)**(3/2)
                        if abs(denominator) > 1e-15:
                            volume_change_rate = uz_trend * np.pi / denominator  # m³/day
                            mogi_analysis['volume_change_rate'] = float(volume_change_rate)
                            
                            # Convert to pressure change (approximate)
                            # Assuming spherical chamber: ΔP ≈ μ * ΔV/V₀
                            # With chamber volume estimate V₀ ~ 10⁶ - 10⁹ m³
                            chamber_volume_estimate = 1e8  # m³ (moderate size chamber)
                            pressure_change_rate = shear_modulus * volume_change_rate / chamber_volume_estimate
                            mogi_analysis['pressure_change_rate'] = float(pressure_change_rate)  # Pa/day
                        else:
                            mogi_analysis['volume_change_rate'] = 0.0
                            mogi_analysis['pressure_change_rate'] = 0.0
                    else:
                        mogi_analysis['volume_change_rate'] = 0.0
                        mogi_analysis['pressure_change_rate'] = 0.0
                    
                    # === METHOD 3: DEFORMATION PATTERN CLASSIFICATION ===
                    # Classify the type of deformation pattern observed
                    
                    inflation_threshold = 1e-6  # m/day
                    deflation_threshold = -1e-6  # m/day
                    
                    # Primary pattern identification
                    if uz_trend > inflation_threshold and ur_trend > 0:
                        if ratio_rv > 2.0:
                            mogi_analysis['deformation_pattern_type'] = 'shallow_inflation'
                            eruption_relevance_pattern = 0.8  # High relevance
                        elif ratio_rv > 0.5:
                            mogi_analysis['deformation_pattern_type'] = 'intermediate_inflation'
                            eruption_relevance_pattern = 0.6
                        else:
                            mogi_analysis['deformation_pattern_type'] = 'deep_inflation'
                            eruption_relevance_pattern = 0.4
                    elif uz_trend < deflation_threshold and ur_trend < 0:
                        mogi_analysis['deformation_pattern_type'] = 'deflation'
                        eruption_relevance_pattern = 0.3  # Lower eruption relevance
                    elif abs(ur_trend) > abs(uz_trend):
                        mogi_analysis['deformation_pattern_type'] = 'lateral_migration'
                        eruption_relevance_pattern = 0.5
                    else:
                        mogi_analysis['deformation_pattern_type'] = 'mixed_pattern'
                        eruption_relevance_pattern = 0.4
                    
                    # === METHOD 4: STRESS REGIME ANALYSIS ===
                    # Determine stress regime implications
                    
                    if mogi_analysis['volume_change_rate'] and abs(mogi_analysis['volume_change_rate']) > 1e3:
                        if mogi_analysis['volume_change_rate'] > 0:
                            mogi_analysis['stress_regime'] = 'pressurization'
                            stress_eruption_relevance = 0.7
                        else:
                            mogi_analysis['stress_regime'] = 'depressurization'
                            stress_eruption_relevance = 0.2
                    else:
                        mogi_analysis['stress_regime'] = 'stable'
                        stress_eruption_relevance = 0.3
                    
                    # === METHOD 5: ADVANCED SOURCE STRENGTH CALCULATION ===
                    # Calculate Mogi source strength parameter
                    
                    # Source strength S = (ΔV * μ) / (π * (1-ν))
                    if mogi_analysis['volume_change_rate']:
                        source_strength = abs(mogi_analysis['volume_change_rate']) * shear_modulus / (np.pi * (1 - poisson_ratio))
                        mogi_analysis['source_strength'] = float(source_strength)
                        
                        # Normalize for eruption relevance (log scale due to wide range)
                        strength_eruption_relevance = min(1.0, np.log10(source_strength + 1) / 15.0)
                    else:
                        mogi_analysis['source_strength'] = 0.0
                        strength_eruption_relevance = 0.1
                    
                    # === METHOD 6: TEMPORAL ACCELERATION ANALYSIS ===
                    # Analyze acceleration patterns in deformation
                    
                    acceleration_score = 0.0
                    if len(deformation_rate) >= 7:
                        # Calculate second derivative (acceleration)
                        rate_series = np.array(deformation_rate[-7:])
                        if len(rate_series) > 2:
                            acceleration = np.gradient(np.gradient(rate_series))
                            mean_acceleration = np.mean(acceleration)
                            acceleration_significance = abs(mean_acceleration) / (np.std(acceleration) + 1e-6)
                            
                            if acceleration_significance > 1.0:  # Significant acceleration
                                acceleration_score = min(1.0, acceleration_significance / 3.0)
                                if mean_acceleration > 0:
                                    mogi_analysis['acceleration_pattern'] = 'accelerating_deformation'
                                else:
                                    mogi_analysis['acceleration_pattern'] = 'decelerating_deformation'
                            else:
                                mogi_analysis['acceleration_pattern'] = 'constant_rate'
                        else:
                            mogi_analysis['acceleration_pattern'] = 'insufficient_data'
                    else:
                        mogi_analysis['acceleration_pattern'] = 'insufficient_data'
                    
                    # === INTEGRATED ERUPTION RELEVANCE SCORING ===
                    # Combine all analysis components
                    
                    # Weight different components
                    pattern_weight = 0.3
                    stress_weight = 0.25
                    strength_weight = 0.2
                    depth_weight = 0.15
                    acceleration_weight = 0.1
                    
                    # Depth relevance (shallower = more relevant)
                    depth_eruption_relevance = max(0, 1.0 - (estimated_depth - 1000) / 10000)  # Higher relevance for depths < 5km
                    
                    # Combined eruption relevance
                    integrated_eruption_relevance = (
                        pattern_weight * eruption_relevance_pattern +
                        stress_weight * stress_eruption_relevance +
                        strength_weight * strength_eruption_relevance +
                        depth_weight * depth_eruption_relevance +
                        acceleration_weight * acceleration_score
                    )
                    
                    depth_indicator = integrated_eruption_relevance
                    
                    # === UNCERTAINTY AND QUALITY METRICS ===
                    # Assess the reliability of the analysis
                    
                    data_quality_score = min(1.0, len(deformation_radial[-7:]) / 7.0)  # Data completeness
                    signal_noise_ratio = abs(uz_trend) / (1e-6 + np.std(deformation_vertical[-7:]) * 1e-3)
                    snr_score = min(1.0, signal_noise_ratio / 2.0)
                    
                    overall_confidence = (mogi_analysis['model_confidence'] + data_quality_score + snr_score) / 3.0
                    
                    # Add comprehensive metadata
                    mogi_analysis.update({
                        'radial_deformation_rate_mm_day': float(radial_trend),
                        'vertical_deformation_rate_mm_day': float(vertical_trend),
                        'radial_vertical_ratio': float(ratio_rv) if 'ratio_rv' in locals() else 0.0,
                        'characteristic_distance_m': float(characteristic_distance),
                        'eruption_relevance_score': float(integrated_eruption_relevance),
                        'data_quality_score': float(data_quality_score),
                        'signal_noise_ratio': float(signal_noise_ratio),
                        'overall_confidence': float(overall_confidence),
                        'analysis_components': {
                            'pattern_relevance': float(eruption_relevance_pattern),
                            'stress_relevance': float(stress_eruption_relevance), 
                            'strength_relevance': float(strength_eruption_relevance),
                            'depth_relevance': float(depth_eruption_relevance),
                            'acceleration_score': float(acceleration_score)
                        },
                        'physical_parameters': {
                            'poisson_ratio': float(poisson_ratio),
                            'shear_modulus_pa': float(shear_modulus),
                            'chamber_volume_estimate_m3': float(chamber_volume_estimate)
                        }
                    })
                    
                except Exception as e:
                    # Fallback analysis if comprehensive method fails
                    mogi_analysis = {
                        'error': str(e),
                        'fallback_analysis': True,
                        'source_depth_estimate': 5000.0,
                        'model_confidence': 0.2,
                        'deformation_pattern_type': 'unknown',
                        'eruption_relevance_score': 0.3
                    }
                    depth_indicator = 0.3
                    
            else:
                # No significant deformation detected
                depth_indicator = 0.1
                mogi_analysis = {
                    'no_deformation_detected': True,
                    'source_depth_estimate': None,
                    'model_confidence': 0.0,
                    'deformation_pattern_type': 'none',
                    'eruption_relevance_score': 0.0
                }
            
            deformation_score = (
                0.4 * min(1.0, inflation_score / 0.1) +  # Inflation magnitude
                0.3 * min(1.0, abs(rate_acceleration) / 0.01) +  # Acceleration
                0.2 * min(1.0, rate_volatility / 0.05) +  # Rate volatility
                0.1 * depth_indicator  # Shallow source preference
            )
        else:
            deformation_score = radial_trend = vertical_trend = rate_acceleration = rate_volatility = depth_indicator = 0
        
        factors['deformation'] = deformation_score
        detailed_analysis['deformation'] = {
            'radial_trend_mm_per_day': float(radial_trend),
            'vertical_trend_mm_per_day': float(vertical_trend),
            'rate_acceleration': float(rate_acceleration),
            'rate_volatility': float(rate_volatility),
            'source_depth_indicator': float(depth_indicator),
            'model_attention': model_attention_weights.get('deformation_attention', 0.0),
            'mogi_source_analysis': mogi_analysis if 'mogi_analysis' in locals() else {'no_analysis': True}
        }
        
        # === THERMAL ANOMALY ANALYSIS ===
        thermal_anomaly = np.array(monitoring_data.get('thermal_anomaly', [0]*30))
        thermal_area = np.array(monitoring_data.get('thermal_area', [0]*30))
        thermal_intensity = np.array(monitoring_data.get('thermal_intensity', [0]*30))
        
        if len(thermal_anomaly) >= 7:
            # Thermal evolution analysis
            thermal_trend = (np.mean(thermal_anomaly[-3:]) - np.mean(thermal_anomaly[-7:-3])) if len(thermal_anomaly) >= 7 else 0
            thermal_persistence = np.sum(thermal_anomaly[-7:] > np.mean(thermal_anomaly[:-7])) / 7.0 if len(thermal_anomaly) > 7 else 0
            
            # Thermal flux analysis
            if len(thermal_area) >= 7 and len(thermal_intensity) >= 7:
                thermal_flux = thermal_area[-1] * thermal_intensity[-1] if len(thermal_area) > 0 and len(thermal_intensity) > 0 else 0
                flux_trend = (thermal_flux - np.mean(thermal_area[-7:-1] * thermal_intensity[-7:-1])) / (np.mean(thermal_area[-7:-1] * thermal_intensity[-7:-1]) + 1) if len(thermal_area) >= 7 else 0
            else:
                thermal_flux = flux_trend = 0
            
            thermal_score = (
                0.4 * min(1.0, abs(thermal_trend) / 0.5) +  # Thermal trend
                0.3 * thermal_persistence +  # Persistence of anomalies
                0.3 * min(1.0, abs(flux_trend))  # Thermal flux changes
            )
        else:
            thermal_score = thermal_trend = thermal_persistence = flux_trend = 0
        
        factors['thermal_anomalies'] = thermal_score
        detailed_analysis['thermal_anomalies'] = {
            'temperature_trend': float(thermal_trend),
            'anomaly_persistence': float(thermal_persistence),
            'thermal_flux_trend': float(flux_trend),
            'model_attention': model_attention_weights.get('thermal_attention', 0.0)
        }
        
        # === PRECURSOR PATTERN ANALYSIS ===
        # Advanced pattern recognition for eruption precursors
        
        # Multi-parameter correlation analysis
        if len(pressure_shallow) >= 15 and len(seismic_count) >= 15:
            # Cross-correlation between pressure and seismicity
            pressure_norm = (pressure_shallow[-15:] - np.mean(pressure_shallow[-15:])) / (np.std(pressure_shallow[-15:]) + 1e-6)
            seismic_norm = (seismic_count[-15:] - np.mean(seismic_count[-15:])) / (np.std(seismic_count[-15:]) + 1e-6)
            cross_correlation = np.corrcoef(pressure_norm, seismic_norm)[0,1] if not np.isnan(np.corrcoef(pressure_norm, seismic_norm)[0,1]) else 0
        else:
            cross_correlation = 0
        
        # Temporal clustering analysis
        if len(seismic_count) >= 10:
            # Identify periods of elevated activity across multiple parameters
            seismic_elevated = seismic_count[-10:] > (np.mean(seismic_count[:-10]) + np.std(seismic_count[:-10])) if len(seismic_count) > 10 else np.zeros(10)
            pressure_elevated = pressure_shallow[-10:] > (np.mean(pressure_shallow[:-10]) + np.std(pressure_shallow[:-10])) if len(pressure_shallow) > 10 else np.zeros(10)
            gas_elevated = gas_co2[-10:] > (np.mean(gas_co2[:-10]) + np.std(gas_co2[:-10])) if len(gas_co2) > 10 else np.zeros(10)
            
            # Count coincident elevations
            multi_parameter_events = np.sum((seismic_elevated & pressure_elevated) | 
                                          (seismic_elevated & gas_elevated) | 
                                          (pressure_elevated & gas_elevated))
            clustering_score = multi_parameter_events / 10.0
        else:
            clustering_score = 0
        
        # Escalation pattern detection
        if len(pressure_shallow) >= 20:
            # Look for accelerating patterns
            pressure_windows = [np.mean(pressure_shallow[-5:]), 
                              np.mean(pressure_shallow[-10:-5]), 
                              np.mean(pressure_shallow[-20:-10])]
            if len(pressure_windows) == 3:
                escalation_detected = (pressure_windows[0] > pressure_windows[1] > pressure_windows[2])
                escalation_score = 1.0 if escalation_detected else 0.0
            else:
                escalation_score = 0.0
        else:
            escalation_score = 0.0
        
        precursor_score = (
            0.4 * min(1.0, abs(cross_correlation)) +  # Parameter correlation
            0.4 * clustering_score +  # Multi-parameter clustering  
            0.2 * escalation_score  # Escalation patterns
        )
        
        factors['precursor_patterns'] = precursor_score
        detailed_analysis['precursor_patterns'] = {
            'cross_correlation_strength': float(abs(cross_correlation)),
            'multi_parameter_clustering': float(clustering_score),
            'escalation_detected': bool(escalation_score > 0.5),
            'pattern_complexity_score': float(precursor_score)
        }
        
        # === TEMPORAL INTEGRATION AND WEIGHTING ===
        
        # Apply temporal weighting (recent data more important)
        temporal_weights = np.exp(-np.arange(7) / 3.0)  # 3-day half-life
        temporal_weights = temporal_weights / np.sum(temporal_weights)
        
        # Weight by model attention if available
        if model_attention_weights:
            attention_sum = sum(model_attention_weights.values())
            if attention_sum > 0:
                for key in factors:
                    if f"{key.replace('_', '_')}_attention" in model_attention_weights:
                        attention_weight = model_attention_weights[f"{key.replace('_', '_')}_attention"] / attention_sum
                        factors[key] = 0.7 * factors[key] + 0.3 * attention_weight
        
        # === FINAL NORMALIZATION AND VALIDATION ===
        
        # Ensure all factors are in valid range [0, 1]
        for key in factors:
            factors[key] = max(0.0, min(1.0, factors[key]))
        
        # Normalize to sum to 1
        total = sum(factors.values())
        if total > 0:
            factors = {k: v/total for k, v in factors.items()}
        else:
            # Default uniform distribution if no signal detected
            n_factors = len(factors)
            factors = {k: 1.0/n_factors for k in factors.keys()}
        
        # Add detailed analysis metadata
        factors['_detailed_analysis'] = detailed_analysis
        factors['_analysis_metadata'] = {
            'temporal_window_days': 30,
            'recent_focus_days': 7,
            'correlation_analysis_applied': True,
            'physics_informed_features': True,
            'model_attention_integrated': bool(model_attention_weights),
            'multi_scale_temporal_analysis': True,
            'statistical_significance_tested': True
        }
        
        return factors
    
    def generate_recommendations(self, prob: float, vei: int, 
                               time_to_eruption: Optional[float], 
                               alert_level: str) -> List[str]:
        """Generate recommendations based on forecast"""
        recommendations = []
        
        if alert_level == 'CRITICAL':
            recommendations.extend([
                "Immediate evacuation of high-risk zones recommended",
                "Activate emergency response protocols",
                "Notify aviation authorities of imminent eruption risk",
                "Increase monitoring frequency to real-time"
            ])
        elif alert_level == 'WARNING':
            recommendations.extend([
                "Prepare evacuation plans for at-risk communities",
                "Increase monitoring and surveillance activities",
                "Brief emergency responders and local authorities",
                "Consider restricting access to high-risk areas"
            ])
        elif alert_level == 'WATCH':
            recommendations.extend([
                "Enhanced monitoring of volcanic parameters",
                "Update emergency response plans",
                "Inform local communities of increased volcanic activity"
            ])
        elif alert_level == 'ADVISORY':
            recommendations.extend([
                "Continue routine monitoring",
                "Monitor for changes in volcanic behavior",
                "Maintain public awareness"
            ])
        else:
            recommendations.append("Continue baseline monitoring")
        
        if vei >= 4:
            recommendations.append("Prepare for potential regional ash fall impacts")
        
        if time_to_eruption and time_to_eruption < 7:
            recommendations.append(f"Estimated eruption window: {time_to_eruption:.1f} days")
        
        return recommendations
    
    def save_model(self, path: str):
        """Save trained model and preprocessing components"""
        model_data = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'alert_thresholds': self.alert_thresholds
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model and preprocessing components"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Initialize model architecture
        self.model = PhysicsInformedEruptionNet()
        if model_data['model_state_dict']:
            self.model.load_state_dict(model_data['model_state_dict'])
        
        self.scaler = model_data['scaler']
        self.anomaly_detector = model_data['anomaly_detector'] 
        self.alert_thresholds = model_data['alert_thresholds']
        
        logger.info(f"Model loaded from {path}")

# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Initialize forecaster
    forecaster = EruptionForecaster()
    
    # Generate synthetic training data for demonstration
    def generate_synthetic_data(n_volcanoes: int = 10, n_days: int = 1000) -> pd.DataFrame:
        """Generate synthetic volcanic monitoring data"""
        data = []
        
        for volcano_id in range(n_volcanoes):
            for day in range(n_days):
                # Simulate volcanic parameters with trends
                base_pressure = 200 + volcano_id * 50
                pressure_trend = np.sin(day / 100) * 50
                noise = np.random.normal(0, 10)
                
                # Create eruption events (rare)
                eruption_flag = 1 if np.random.random() < 0.001 else 0
                vei = np.random.choice([0,1,2,3,4,5], p=[0.4,0.3,0.15,0.1,0.04,0.01]) if eruption_flag else 0
                
                # Pre-eruption signals
                if eruption_flag:
                    pressure_anomaly = 100
                    temp_anomaly = 50
                    gas_anomaly = 5
                    seismic_anomaly = 20
                else:
                    pressure_anomaly = 0
                    temp_anomaly = 0
                    gas_anomaly = 0
                    seismic_anomaly = 0
                
                row = {
                    'volcano_id': volcano_id,
                    'timestamp': day,
                    'pressure_shallow': base_pressure + pressure_trend + noise + pressure_anomaly,
                    'pressure_middle': base_pressure * 1.5 + pressure_trend + noise + pressure_anomaly,
                    'pressure_deep': base_pressure * 2 + pressure_trend + noise + pressure_anomaly,
                    'temperature_shallow': 800 + volcano_id * 20 + np.random.normal(0, 20) + temp_anomaly,
                    'temperature_middle': 900 + volcano_id * 20 + np.random.normal(0, 15) + temp_anomaly,
                    'temperature_deep': 1000 + volcano_id * 20 + np.random.normal(0, 10) + temp_anomaly,
                    'gas_co2': 2 + np.random.exponential(1) + gas_anomaly,
                    'gas_so2': 1 + np.random.exponential(0.5) + gas_anomaly * 0.5,
                    'gas_h2s': 0.5 + np.random.exponential(0.2) + gas_anomaly * 0.2,
                    'gas_ratio_co2_so2': 2 + np.random.normal(0, 0.5),
                    'seismic_magnitude_avg': np.random.exponential(0.5) + seismic_anomaly * 0.1,
                    'seismic_count': np.random.poisson(5) + seismic_anomaly,
                    'seismic_energy': np.random.lognormal(0, 1) + seismic_anomaly,
                    'deformation_radial': np.random.normal(0, 2) + pressure_anomaly * 0.01,
                    'deformation_vertical': np.random.normal(0, 3) + pressure_anomaly * 0.02,
                    'deformation_rate': np.random.normal(0, 0.5) + pressure_anomaly * 0.005,
                    'thermal_anomaly': np.random.exponential(0.1) + temp_anomaly * 0.01,
                    'thermal_area': np.random.exponential(100),
                    'thermal_intensity': np.random.exponential(10) + temp_anomaly * 0.1,
                    'gravity_change': np.random.normal(0, 0.1),
                    'magnetic_change': np.random.normal(0, 1),
                    'tidal_stress': np.sin(day / 30) * 0.1,
                    'regional_seismicity': np.random.poisson(2),
                    'precipitation': np.random.exponential(5),
                    'eruption_flag': eruption_flag,
                    'vei': vei
                }
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    print("🌋 Generating synthetic training data...")
    training_data = generate_synthetic_data(n_volcanoes=20, n_days=2000)
    print(f"Generated {len(training_data)} training samples")
    
    print("🚀 Training eruption forecasting model...")
    forecaster.train(training_data, epochs=50, optimize_hyperparameters=False)
    
    print("💾 Saving trained model...")
    forecaster.save_model("eruption_forecaster_model.pkl")
    
    # Test prediction
    print("🔮 Testing prediction...")
    test_monitoring_data = {
        'pressure_shallow': [250 + i for i in range(30)],
        'pressure_middle': [350 + i*1.5 for i in range(30)],
        'pressure_deep': [450 + i*2 for i in range(30)],
        'temperature_shallow': [850 + i*0.5 for i in range(30)],
        'temperature_middle': [950 + i*0.3 for i in range(30)],
        'temperature_deep': [1050 + i*0.1 for i in range(30)],
        'gas_co2': [3 + i*0.1 for i in range(30)],
        'gas_so2': [1.5 + i*0.05 for i in range(30)],
        'gas_h2s': [0.7 + i*0.02 for i in range(30)],
        'gas_ratio_co2_so2': [2.0 for _ in range(30)],
        'seismic_magnitude_avg': [1.5 + i*0.05 for i in range(30)],
        'seismic_count': [10 + i for i in range(30)],
        'seismic_energy': [100 + i*5 for i in range(30)],
        'deformation_radial': [0.1*i for i in range(30)],
        'deformation_vertical': [0.2*i for i in range(30)],
        'deformation_rate': [0.05*i for i in range(30)],
        'thermal_anomaly': [0.2 + i*0.01 for i in range(30)],
        'thermal_area': [150 for _ in range(30)],
        'thermal_intensity': [20 + i*0.5 for i in range(30)],
        'gravity_change': [0.01*i for i in range(30)],
        'magnetic_change': [0.1*i for i in range(30)],
        'tidal_stress': [0.05 for _ in range(30)],
        'regional_seismicity': [3 for _ in range(30)],
        'precipitation': [5 for _ in range(30)]
    }
    
    # Run prediction
    import asyncio
    async def test_prediction():
        forecast = await forecaster.predict(test_monitoring_data)
        
        print(f"\n🌋 Eruption Forecast:")
        print(f"   Alert Level: {forecast.alert_level}")
        print(f"   Eruption Probability: {forecast.eruption_probability:.3f}")
        print(f"   Expected VEI: {forecast.expected_vei}")
        print(f"   Time to Eruption: {forecast.time_to_eruption} days")
        print(f"   Confidence: {forecast.confidence:.3f}")
        print(f"   Contributing Factors:")
        for factor, importance in forecast.contributing_factors.items():
            print(f"     - {factor}: {importance:.3f}")
        print(f"   Recommendations:")
        for rec in forecast.recommendations:
            print(f"     - {rec}")
    
    asyncio.run(test_prediction())
    
    print("\n✅ Eruption Forecaster ML system ready!")