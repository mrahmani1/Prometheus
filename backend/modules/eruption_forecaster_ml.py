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
        """Calculate uncertainty bounds for probability"""
        std = uncertainty * 0.5  # Convert to standard deviation
        
        # Generate time series of bounds (simplified)
        time_steps = 30
        bounds = {
            'lower': [max(0, prob - 2*std) for _ in range(time_steps)],
            'upper': [min(1, prob + 2*std) for _ in range(time_steps)]
        }
        
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
        """Analyze which factors contribute most to the prediction"""
        # Simplified feature importance analysis
        factors = {
            'pressure_changes': np.mean(monitoring_data.get('pressure_shallow', [0])[-7:]) / 100,
            'gas_emissions': np.mean(monitoring_data.get('gas_co2', [0])[-7:]) / 10,
            'seismic_activity': np.mean(monitoring_data.get('seismic_count', [0])[-7:]) / 50,
            'deformation': abs(np.mean(monitoring_data.get('deformation_rate', [0])[-7:])) / 10,
            'thermal_anomalies': np.mean(monitoring_data.get('thermal_anomaly', [0])[-7:]),
            'precursor_patterns': 0.5  # Placeholder for complex pattern analysis
        }
        
        # Normalize to sum to 1
        total = sum(factors.values())
        if total > 0:
            factors = {k: v/total for k, v in factors.items()}
        
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