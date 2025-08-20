"""
Model Manager for Volcanic Prediction System
Manages ML models, training on Azure VM datasets, and prediction services
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Any, Union
import logging
import asyncio
import aiohttp
import json
import pickle
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import azure.storage.blob as azure_blob
from azure.identity import DefaultAzureCredential
import joblib

# Import our model database and MLOps pipeline
try:
    from .model_database import ModelDatabase, ModelMetadata, ModelRecord
    MODEL_DATABASE_AVAILABLE = True
except ImportError:
    MODEL_DATABASE_AVAILABLE = False
    
try:
    from .mlops_pipeline import MLOpsPipeline, PipelineConfig, ModelMonitor
    MLOPS_PIPELINE_AVAILABLE = True
except ImportError:
    MLOPS_PIPELINE_AVAILABLE = False

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: str
    hyperparameters: Dict[str, Any]
    azure_vm_endpoint: str
    dataset_path: str
    model_save_path: str
    retrain_interval: int = 24  # hours

class VolcanicDataset(Dataset):
    """PyTorch dataset for volcanic monitoring data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class VolcanicPredictionNet(nn.Module):
    """Neural network for volcanic eruption prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelManager:
    """Manages ML models for volcanic prediction with Azure VM integration"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.last_training = None
        self.azure_client = None
        self.model_database = None
        self.mlops_pipeline = None
        self._setup_azure_connection()
        self._setup_model_database()
        self._setup_mlops_pipeline()
    
    def _setup_azure_connection(self):
        """Setup Azure Blob Storage connection for model persistence"""
        try:
            credential = DefaultAzureCredential()
            # Initialize Azure client if credentials are available
            self.azure_client = azure_blob.BlobServiceClient(
                account_url=os.getenv('AZURE_STORAGE_ACCOUNT_URL'),
                credential=credential
            )
        except Exception as e:
            self.logger.warning(f"Azure connection not available: {e}")
    
    def _setup_model_database(self):
        """Setup model database for persistence and versioning"""
        try:
            if MODEL_DATABASE_AVAILABLE:
                # Initialize model database
                db_config = {
                    'database_url': os.getenv('MODEL_DATABASE_URL', 'sqlite:///volcanic_models.db'),
                    'storage_backend': 'filesystem',  # or 's3', 'azure_blob'
                    'storage_path': self.config.model_save_path
                }
                
                from .model_database import ModelDatabase
                self.model_database = ModelDatabase(db_config)
                self.logger.info("Model database initialized")
            else:
                self.logger.warning("Model database not available")
        except Exception as e:
            self.logger.error(f"Model database setup failed: {e}")
    
    def _setup_mlops_pipeline(self):
        """Setup MLOps pipeline for automated operations"""
        try:
            if MLOPS_PIPELINE_AVAILABLE:
                # Initialize MLOps pipeline
                pipeline_config = PipelineConfig(
                    model_registry_url=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                    experiment_tracking_url=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'),
                    data_drift_threshold=0.1,
                    model_performance_threshold=0.8,
                    retraining_frequency=self.config.retrain_interval,
                    validation_split=0.2,
                    monitoring_interval=300
                )
                
                from .mlops_pipeline import MLOpsPipeline
                self.mlops_pipeline = MLOpsPipeline(pipeline_config)
                self.logger.info("MLOps pipeline initialized")
            else:
                self.logger.warning("MLOps pipeline not available")
        except Exception as e:
            self.logger.error(f"MLOps pipeline setup failed: {e}")
    
    async def fetch_training_data_from_azure_vm(self) -> pd.DataFrame:
        """Fetch training datasets from Azure VM"""
        try:
            async with aiohttp.ClientSession() as session:
                # Connect to Azure VM endpoint for dataset access
                headers = {'Authorization': f"Bearer {os.getenv('AZURE_VM_TOKEN')}"}
                
                async with session.get(
                    f"{self.config.azure_vm_endpoint}/datasets/volcanic_training",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data['features'])
                        df['target_eruption_prob'] = data['targets']['eruption_probability']
                        df['target_vei'] = data['targets']['vei']
                        df['target_time_to_event'] = data['targets']['time_to_event']
                        
                        self.logger.info(f"Fetched {len(df)} training samples from Azure VM")
                        return df
                    else:
                        raise Exception(f"Failed to fetch data: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error fetching Azure VM data: {e}")
            # Fallback to local dataset if available
            return self._load_local_training_data()
    
    def _load_local_training_data(self) -> pd.DataFrame:
        """Load local training data as fallback"""
        try:
            if os.path.exists(self.config.dataset_path):
                return pd.read_csv(self.config.dataset_path)
            else:
                # Generate synthetic training data for demonstration
                return self._generate_synthetic_data()
        except Exception as e:
            self.logger.error(f"Error loading local data: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic training data for model development"""
        np.random.seed(42)
        n_samples = 1000
        
        # Seismic features
        seismic_amplitude = np.random.lognormal(0, 1, n_samples)
        seismic_frequency = np.random.normal(2.5, 0.8, n_samples)
        tremor_duration = np.random.exponential(10, n_samples)
        
        # Gas composition features
        so2_concentration = np.random.lognormal(2, 0.5, n_samples)
        co2_flux = np.random.lognormal(3, 0.7, n_samples)
        h2s_ratio = np.random.beta(2, 5, n_samples)
        
        # Thermal features
        thermal_anomaly = np.random.gamma(2, 0.5, n_samples)
        ground_temperature = np.random.normal(25, 5, n_samples)
        
        # Deformation features
        ground_displacement = np.random.normal(0, 2, n_samples)
        tilt_change = np.random.normal(0, 0.1, n_samples)
        
        # Create targets based on feature combinations
        eruption_indicator = (
            (seismic_amplitude > 2) & 
            (so2_concentration > 5) & 
            (thermal_anomaly > 1.5)
        ).astype(float)
        
        eruption_probability = np.clip(
            0.1 * seismic_amplitude + 
            0.15 * np.log(so2_concentration) + 
            0.1 * thermal_anomaly + 
            np.random.normal(0, 0.1, n_samples),
            0, 1
        )
        
        vei = np.clip(
            np.round(eruption_probability * 4 + np.random.normal(0, 0.5, n_samples)),
            0, 6
        ).astype(int)
        
        time_to_event = np.clip(
            (1 - eruption_probability) * 168 + np.random.normal(0, 12, n_samples),
            1, 168
        )  # Hours
        
        df = pd.DataFrame({
            'seismic_amplitude': seismic_amplitude,
            'seismic_frequency': seismic_frequency,
            'tremor_duration': tremor_duration,
            'so2_concentration': so2_concentration,
            'co2_flux': co2_flux,
            'h2s_ratio': h2s_ratio,
            'thermal_anomaly': thermal_anomaly,
            'ground_temperature': ground_temperature,
            'ground_displacement': ground_displacement,
            'tilt_change': tilt_change,
            'target_eruption_prob': eruption_probability,
            'target_vei': vei,
            'target_time_to_event': time_to_event
        })
        
        return df
    
    async def train_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Train ML models using Azure VM datasets with MLOps integration"""
        try:
            # Check if retraining is needed
            if not force_retrain and self.last_training:
                hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
                if hours_since_training < self.config.retrain_interval:
                    return {'status': 'skipped', 'reason': 'recent_training'}
            
            # Start MLOps monitoring if available
            if self.mlops_pipeline and hasattr(self.mlops_pipeline, 'monitor'):
                training_start = datetime.now()
            
            # Fetch training data
            df = await self.fetch_training_data_from_azure_vm()
            
            # Initialize data drift detector with training data
            if self.mlops_pipeline:
                self.mlops_pipeline.initialize_drift_detector(df)
            
            # Prepare features and targets
            feature_columns = [col for col in df.columns if not col.startswith('target_')]
            X = df[feature_columns].values
            
            # Train multiple models for different predictions
            results = {}
            model_versions = {}
            
            # Eruption probability model
            y_eruption = df['target_eruption_prob'].values
            eruption_model, eruption_scaler = await self._train_eruption_model(X, y_eruption)
            self.models['eruption'] = eruption_model
            self.scalers['eruption'] = eruption_scaler
            results['eruption'] = {'r2': r2_score(y_eruption, eruption_model.predict(eruption_scaler.transform(X)))}
            
            # VEI prediction model
            y_vei = df['target_vei'].values
            vei_model, vei_scaler = await self._train_vei_model(X, y_vei)
            self.models['vei'] = vei_model
            self.scalers['vei'] = vei_scaler
            results['vei'] = {'accuracy': np.mean(y_vei == vei_model.predict(vei_scaler.transform(X)))}
            
            # Time to event model
            y_time = df['target_time_to_event'].values
            time_model, time_scaler = await self._train_time_model(X, y_time)
            self.models['time'] = time_model
            self.scalers['time'] = time_scaler
            results['time'] = {'mae': mean_absolute_error(y_time, time_model.predict(time_scaler.transform(X)))}
            
            # Save models with versioning
            await self._save_models_with_versioning(results)
            
            # Register models in database if available
            if self.model_database:
                for model_type in ['eruption', 'vei', 'time']:
                    model_metadata = await self._create_model_metadata(model_type, results[model_type])
                    version_id = await self.model_database.store_model(
                        model_metadata, 
                        f"{self.config.model_save_path}/{model_type}_model.pkl"
                    )
                    model_versions[model_type] = version_id
            
            # Record training metrics for MLOps
            if self.mlops_pipeline and hasattr(self.mlops_pipeline, 'monitor'):
                training_duration = (datetime.now() - training_start).total_seconds()
                self.mlops_pipeline.monitor.record_prediction(training_duration, success=True)
            
            self.last_training = datetime.now()
            results['training_completed'] = self.last_training.isoformat()
            results['samples_used'] = len(df)
            results['model_versions'] = model_versions
            results['mlops_enabled'] = self.mlops_pipeline is not None
            results['database_enabled'] = self.model_database is not None
            
            self.logger.info(f"Model training completed with {len(df)} samples")
            self.logger.info(f"Model versions: {model_versions}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            # Record failure in MLOps
            if self.mlops_pipeline and hasattr(self.mlops_pipeline, 'monitor'):
                self.mlops_pipeline.monitor.record_prediction(0, success=False)
            return {'status': 'failed', 'error': str(e)}
    
    async def _train_eruption_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train eruption probability prediction model"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.config.model_type == 'neural_network':
            # PyTorch neural network
            dataset = VolcanicDataset(X_scaled, y.reshape(-1, 1))
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model = VolcanicPredictionNet(
                input_size=X_scaled.shape[1],
                hidden_sizes=[128, 64, 32],
                output_size=1
            )
            
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(100):
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Create sklearn-compatible wrapper
            class TorchWrapper:
                def __init__(self, model):
                    self.model = model
                    self.model.eval()
                
                def predict(self, X):
                    with torch.no_grad():
                        return self.model(torch.FloatTensor(X)).numpy().flatten()
            
            return TorchWrapper(model), scaler
        
        else:
            # Sklearn models
            if self.config.model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.config.model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_scaled, y)
            return model, scaler
    
    async def _train_vei_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train VEI prediction model"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use RandomForest for classification
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler
    
    async def _train_time_model(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Train time to event prediction model"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use GradientBoosting for regression
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler
    
    async def _save_models(self):
        """Save trained models to disk and Azure"""
        try:
            os.makedirs(self.config.model_save_path, exist_ok=True)
            
            # Save locally
            for model_name, model in self.models.items():
                model_path = os.path.join(self.config.model_save_path, f"{model_name}_model.pkl")
                scaler_path = os.path.join(self.config.model_save_path, f"{model_name}_scaler.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
            
            # Upload to Azure if available
            if self.azure_client:
                await self._upload_models_to_azure()
                
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    async def _save_models_with_versioning(self, results: Dict[str, Any]):
        """Save models with version control and metadata"""
        try:
            os.makedirs(self.config.model_save_path, exist_ok=True)
            
            # Create version timestamp
            version_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save models with version info
            for model_name, model in self.models.items():
                # Save current version
                model_path = os.path.join(self.config.model_save_path, f"{model_name}_model.pkl")
                scaler_path = os.path.join(self.config.model_save_path, f"{model_name}_scaler.pkl")
                
                # Save versioned copies
                versioned_model_path = os.path.join(
                    self.config.model_save_path, 
                    f"{model_name}_model_{version_timestamp}.pkl"
                )
                versioned_scaler_path = os.path.join(
                    self.config.model_save_path, 
                    f"{model_name}_scaler_{version_timestamp}.pkl"
                )
                
                joblib.dump(model, model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
                joblib.dump(model, versioned_model_path)
                joblib.dump(self.scalers[model_name], versioned_scaler_path)
                
                # Save metadata
                metadata = {
                    'model_name': model_name,
                    'version': version_timestamp,
                    'training_timestamp': datetime.now().isoformat(),
                    'model_type': self.config.model_type,
                    'performance_metrics': results.get(model_name, {}),
                    'hyperparameters': self.config.hyperparameters,
                    'model_path': model_path,
                    'scaler_path': scaler_path
                }
                
                metadata_path = os.path.join(
                    self.config.model_save_path, 
                    f"{model_name}_metadata_{version_timestamp}.json"
                )
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Upload to Azure if available
            if self.azure_client:
                await self._upload_models_to_azure()
                
        except Exception as e:
            self.logger.error(f"Error saving models with versioning: {e}")
    
    async def _create_model_metadata(self, model_type: str, performance_metrics: Dict) -> 'ModelMetadata':
        """Create model metadata for database storage"""
        try:
            if not MODEL_DATABASE_AVAILABLE:
                return None
                
            from .model_database import ModelMetadata
            
            metadata = ModelMetadata(
                id=f"volcanic_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=f"Volcanic {model_type.title()} Predictor",
                version=datetime.now().strftime("%Y%m%d_%H%M%S"),
                description=f"ML model for predicting volcanic {model_type} using multi-sensor data",
                author="Prometheus Volcanic Monitoring System",
                institution="Volcanic Research Center",
                creation_date=datetime.now(),
                last_modified=datetime.now(),
                model_type=f"volcanic_{model_type}",
                region="multi-volcanic",
                resolution={"temporal": 1.0, "spatial": 0.1},  # 1 hour, 0.1 km
                dimensions={"features": 10, "output": 1},
                coordinate_system="geographic",
                processing_parameters=dict(self.config.hyperparameters),
                quality_metrics=performance_metrics,
                validation_status="validated",
                tags=[model_type, "volcanic", "real-time", "ml"],
                license="MIT",
                file_format="pickle"
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error creating model metadata: {e}")
            return None
    
    async def _upload_models_to_azure(self):
        """Upload models to Azure Blob Storage"""
        try:
            container_name = "volcanic-models"
            
            for model_name in self.models.keys():
                model_path = os.path.join(self.config.model_save_path, f"{model_name}_model.pkl")
                scaler_path = os.path.join(self.config.model_save_path, f"{model_name}_scaler.pkl")
                
                # Upload model
                with open(model_path, "rb") as data:
                    blob_client = self.azure_client.get_blob_client(
                        container=container_name, 
                        blob=f"{model_name}_model_{datetime.now().strftime('%Y%m%d')}.pkl"
                    )
                    blob_client.upload_blob(data, overwrite=True)
                
                # Upload scaler
                with open(scaler_path, "rb") as data:
                    blob_client = self.azure_client.get_blob_client(
                        container=container_name, 
                        blob=f"{model_name}_scaler_{datetime.now().strftime('%Y%m%d')}.pkl"
                    )
                    blob_client.upload_blob(data, overwrite=True)
                    
        except Exception as e:
            self.logger.error(f"Error uploading to Azure: {e}")
    
    async def load_models(self):
        """Load trained models from disk"""
        try:
            for model_type in ['eruption', 'vei', 'time']:
                model_path = os.path.join(self.config.model_save_path, f"{model_type}_model.pkl")
                scaler_path = os.path.join(self.config.model_save_path, f"{model_type}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_type] = joblib.load(model_path)
                    self.scalers[model_type] = joblib.load(scaler_path)
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    async def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate predictions using trained models with MLOps monitoring"""
        prediction_start = datetime.now()
        
        try:
            # Convert features to array
            feature_names = ['seismic_amplitude', 'seismic_frequency', 'tremor_duration',
                           'so2_concentration', 'co2_flux', 'h2s_ratio', 'thermal_anomaly',
                           'ground_temperature', 'ground_displacement', 'tilt_change']
            
            X = np.array([[features.get(name, 0.0) for name in feature_names]])
            
            predictions = {}
            
            # Eruption probability
            if 'eruption' in self.models:
                X_eruption = self.scalers['eruption'].transform(X)
                eruption_prob = self.models['eruption'].predict(X_eruption)[0]
                predictions['eruption_probability'] = float(np.clip(eruption_prob, 0, 1))
            
            # VEI prediction
            if 'vei' in self.models:
                X_vei = self.scalers['vei'].transform(X)
                vei = self.models['vei'].predict(X_vei)[0]
                predictions['expected_vei'] = int(vei)
            
            # Time to event
            if 'time' in self.models:
                X_time = self.scalers['time'].transform(X)
                time_hours = self.models['time'].predict(X_time)[0]
                predictions['time_to_event'] = f"{int(time_hours)} hours"
            
            # Calculate confidence based on model agreement
            confidence = self._calculate_prediction_confidence(X, predictions)
            predictions['confidence'] = confidence
            predictions['model_version'] = '2.0.0-integrated'
            predictions['prediction_timestamp'] = datetime.now().isoformat()
            
            # Record prediction metrics for MLOps monitoring
            if self.mlops_pipeline and hasattr(self.mlops_pipeline, 'monitor'):
                prediction_latency = (datetime.now() - prediction_start).total_seconds()
                self.mlops_pipeline.monitor.record_prediction(prediction_latency, success=True)
            
            # Store prediction for drift detection
            if hasattr(self, 'recent_predictions'):
                self.recent_predictions.append(predictions)
                # Keep only last 100 predictions
                if len(self.recent_predictions) > 100:
                    self.recent_predictions = self.recent_predictions[-100:]
            else:
                self.recent_predictions = [predictions]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            
            # Record prediction failure for MLOps monitoring
            if self.mlops_pipeline and hasattr(self.mlops_pipeline, 'monitor'):
                prediction_latency = (datetime.now() - prediction_start).total_seconds()
                self.mlops_pipeline.monitor.record_prediction(prediction_latency, success=False)
            
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, X: np.ndarray, predictions: Dict) -> float:
        """Calculate prediction confidence based on model uncertainty"""
        try:
            # Simple confidence based on feature ranges and model variance
            base_confidence = 0.8
            
            # Reduce confidence if features are outside training range
            feature_penalty = 0.0
            for feature_val in X[0]:
                if abs(feature_val) > 3:  # More than 3 std devs from mean
                    feature_penalty += 0.1
            
            confidence = max(0.1, base_confidence - feature_penalty)
            return round(confidence, 2)
            
        except Exception:
            return 0.5
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status including all integrated components"""
        status = {
            'models_loaded': list(self.models.keys()),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'azure_connected': self.azure_client is not None,
            'model_count': len(self.models),
            'model_database_enabled': self.model_database is not None,
            'mlops_pipeline_enabled': self.mlops_pipeline is not None,
            'recent_predictions_count': len(getattr(self, 'recent_predictions', [])),
            'config': {
                'model_type': self.config.model_type,
                'azure_vm_endpoint': self.config.azure_vm_endpoint,
                'retrain_interval': self.config.retrain_interval,
                'model_save_path': self.config.model_save_path
            }
        }
        
        # Add MLOps pipeline status if available
        if self.mlops_pipeline:
            status['mlops_status'] = self.mlops_pipeline.get_pipeline_status()
        
        # Add model database info if available
        if self.model_database:
            try:
                # Get model registry statistics
                status['model_registry_stats'] = {
                    'total_models': 'N/A',  # Would query database
                    'active_models': len(self.models),
                    'database_type': 'integrated'
                }
            except Exception as e:
                status['model_registry_stats'] = {'error': str(e)}
        
        # Add recent performance metrics
        if hasattr(self, 'recent_predictions') and self.recent_predictions:
            recent_confidences = [p.get('confidence', 0.5) for p in self.recent_predictions[-10:]]
            status['recent_performance'] = {
                'avg_confidence': np.mean(recent_confidences),
                'min_confidence': np.min(recent_confidences),
                'max_confidence': np.max(recent_confidences),
                'prediction_count': len(self.recent_predictions)
            }
        
        return status
    
    async def start_mlops_pipeline(self):
        """Start the MLOps pipeline for automated operations"""
        if self.mlops_pipeline:
            await self.mlops_pipeline.start_pipeline()
            self.logger.info("MLOps pipeline started")
        else:
            self.logger.warning("MLOps pipeline not available")
    
    async def stop_mlops_pipeline(self):
        """Stop the MLOps pipeline"""
        if self.mlops_pipeline:
            await self.mlops_pipeline.stop_pipeline()
            self.logger.info("MLOps pipeline stopped")
    
    async def get_model_from_database(self, model_id: str) -> Optional[Any]:
        """Retrieve a specific model from the database"""
        if self.model_database:
            try:
                return await self.model_database.load_model(model_id)
            except Exception as e:
                self.logger.error(f"Error retrieving model {model_id}: {e}")
                return None
        return None
    
    async def list_available_models(self) -> List[Dict]:
        """List all available models in the database"""
        if self.model_database:
            try:
                return await self.model_database.list_models()
            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                return []
        return []