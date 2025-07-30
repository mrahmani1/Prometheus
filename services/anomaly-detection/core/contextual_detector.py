"""
Contextual anomaly detection using geological domain knowledge

This module implements contextual anomaly detection that considers geological
domain knowledge, environmental conditions, and temporal/spatial context.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn

from core.config import Settings
from core.database import DatabaseManager
from models.schemas import (
    GeologicalDataStream, DetectedAnomaly, AnomalyType, AnomalySeverity
)

logger = logging.getLogger(__name__)


class GeologicalContextualModel(nn.Module):
    """Neural network model for contextual anomaly detection in geological data"""
    
    def __init__(self, feature_dim: int, context_dim: int, hidden_dim: int = 128):
        super(GeologicalContextualModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Combined processing
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, context):
        feature_encoded = self.feature_encoder(features)
        context_encoded = self.context_encoder(context)
        
        # Combine feature and context representations
        combined = torch.cat([feature_encoded, context_encoded], dim=1)
        
        # Output anomaly probability
        anomaly_prob = self.combined_processor(combined)
        return anomaly_prob


class GeologicalDomainKnowledge:
    """Repository of geological domain knowledge for contextual analysis"""
    
    def __init__(self):
        # Geological process relationships
        self.process_relationships = {
            'seismic': {
                'related_processes': ['volcanic', 'tectonic', 'landslide'],
                'typical_precursors': ['ground_deformation', 'gas_emissions'],
                'temporal_patterns': ['swarm', 'mainshock_aftershock', 'foreshock_mainshock']
            },
            'volcanic': {
                'related_processes': ['seismic', 'thermal', 'gas_emissions', 'deformation'],
                'typical_precursors': ['seismic_swarms', 'ground_uplift', 'gas_composition_change'],
                'temporal_patterns': ['precursory', 'eruptive', 'post_eruptive']
            },
            'thermal': {
                'related_processes': ['volcanic', 'hydrothermal', 'geothermal'],
                'typical_precursors': ['gas_emissions', 'ground_deformation'],
                'temporal_patterns': ['gradual_increase', 'sudden_spike', 'cyclic']
            },
            'deformation': {
                'related_processes': ['seismic', 'volcanic', 'tectonic'],
                'typical_precursors': ['seismic_activity', 'pressure_changes'],
                'temporal_patterns': ['linear', 'exponential', 'step_function']
            }
        }
        
        # Environmental context factors
        self.environmental_factors = {
            'seasonal': {
                'temperature_variations': {'winter': (-10, 10), 'summer': (15, 35)},
                'precipitation_effects': ['groundwater_changes', 'thermal_effects'],
                'atmospheric_pressure': {'low': (980, 1000), 'normal': (1000, 1020), 'high': (1020, 1040)}
            },
            'diurnal': {
                'thermal_cycles': {'day': 'heating', 'night': 'cooling'},
                'atmospheric_effects': ['pressure_variations', 'humidity_changes'],
                'tidal_effects': ['earth_tides', 'ocean_loading']
            },
            'geological_setting': {
                'volcanic_regions': {
                    'expected_activity': ['seismic_swarms', 'gas_emissions', 'thermal_anomalies'],
                    'background_levels': {'seismic': 0.1, 'thermal': 5.0, 'gas': 100}
                },
                'tectonic_regions': {
                    'expected_activity': ['earthquakes', 'ground_deformation'],
                    'background_levels': {'seismic': 0.05, 'deformation': 1.0}
                },
                'stable_regions': {
                    'expected_activity': ['minimal_seismic', 'stable_thermal'],
                    'background_levels': {'seismic': 0.01, 'thermal': 1.0}
                }
            }
        }
        
        # Anomaly significance thresholds based on context
        self.significance_thresholds = {
            'volcanic_regions': {
                'seismic': {'low': 0.5, 'medium': 2.0, 'high': 5.0, 'critical': 10.0},
                'thermal': {'low': 2.0, 'medium': 5.0, 'high': 10.0, 'critical': 20.0},
                'gas': {'low': 1.5, 'medium': 3.0, 'high': 6.0, 'critical': 12.0}
            },
            'tectonic_regions': {
                'seismic': {'low': 0.3, 'medium': 1.0, 'high': 3.0, 'critical': 6.0},
                'deformation': {'low': 0.5, 'medium': 2.0, 'high': 5.0, 'critical': 10.0}
            },
            'stable_regions': {
                'seismic': {'low': 0.1, 'medium': 0.3, 'high': 1.0, 'critical': 2.0},
                'thermal': {'low': 0.5, 'medium': 1.0, 'high': 2.0, 'critical': 5.0}
            }
        }
    
    def get_geological_context(self, location: Dict[str, float]) -> str:
        """Determine geological context based on location"""
        # Simplified geological context determination
        # In practice, this would use geological databases and maps
        
        lat = location.get('lat', 0.0)
        lon = location.get('lon', 0.0)
        
        # Example volcanic regions (simplified)
        volcanic_regions = [
            {'name': 'Ring_of_Fire', 'lat_range': (-60, 70), 'lon_range': (120, -60)},
            {'name': 'Mediterranean', 'lat_range': (30, 50), 'lon_range': (-10, 50)},
            {'name': 'Iceland', 'lat_range': (63, 67), 'lon_range': (-25, -13)}
        ]
        
        for region in volcanic_regions:
            if (region['lat_range'][0] <= lat <= region['lat_range'][1] and
                region['lon_range'][0] <= lon <= region['lon_range'][1]):
                return 'volcanic_regions'
        
        # Example tectonic regions
        if abs(lat) > 30:  # Simplified tectonic activity zones
            return 'tectonic_regions'
        
        return 'stable_regions'
    
    def get_expected_background(self, measurement_type: str, geological_context: str) -> float:
        """Get expected background level for measurement type in given context"""
        context_data = self.environmental_factors['geological_setting'].get(geological_context, {})
        background_levels = context_data.get('background_levels', {})
        return background_levels.get(measurement_type, 1.0)
    
    def get_significance_threshold(
        self, 
        measurement_type: str, 
        geological_context: str, 
        severity_level: str
    ) -> float:
        """Get significance threshold for anomaly classification"""
        context_thresholds = self.significance_thresholds.get(geological_context, {})
        measurement_thresholds = context_thresholds.get(measurement_type, {})
        return measurement_thresholds.get(severity_level, 1.0)
    
    def get_related_processes(self, measurement_type: str) -> List[str]:
        """Get processes related to a measurement type"""
        return self.process_relationships.get(measurement_type, {}).get('related_processes', [])
    
    def get_temporal_patterns(self, measurement_type: str) -> List[str]:
        """Get typical temporal patterns for a measurement type"""
        return self.process_relationships.get(measurement_type, {}).get('temporal_patterns', [])


class ContextualAnomalyDetector:
    """Contextual anomaly detector using geological domain knowledge"""
    
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager
        
        # Domain knowledge
        self.domain_knowledge = GeologicalDomainKnowledge()
        
        # Models
        self.contextual_model: Optional[GeologicalContextualModel] = None
        self.local_outlier_detector: Optional[LocalOutlierFactor] = None
        
        # Data preprocessing
        self.feature_scaler: Optional[StandardScaler] = None
        self.context_scaler: Optional[StandardScaler] = None
        
        # Context memory
        self.context_memory: Dict[str, Any] = {}
        self.memory_size = settings.contextual_memory_size
        
        # Device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    async def initialize(self):
        """Initialize the contextual detector"""
        try:
            logger.info("Initializing contextual anomaly detector...")
            
            # Initialize models
            await self._initialize_models()
            
            # Load context memory
            await self._load_context_memory()
            
            logger.info("Contextual anomaly detector initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual detector: {e}")
            raise
    
    async def detect_anomalies(
        self,
        data_stream: GeologicalDataStream,
        context_window: Optional[int] = None
    ) -> List[DetectedAnomaly]:
        """Detect contextual anomalies in geological data stream"""
        try:
            if not data_stream.data_points:
                return []
            
            # Prepare contextual features
            contextual_features = await self._prepare_contextual_features(
                data_stream, context_window
            )
            
            # Detect anomalies using multiple methods
            anomalies = []
            
            # Neural network-based contextual detection
            nn_anomalies = await self._detect_neural_contextual_anomalies(
                data_stream, contextual_features
            )
            anomalies.extend(nn_anomalies)
            
            # Local outlier factor with context
            lof_anomalies = await self._detect_lof_contextual_anomalies(
                data_stream, contextual_features
            )
            anomalies.extend(lof_anomalies)
            
            # Domain knowledge-based detection
            domain_anomalies = await self._detect_domain_knowledge_anomalies(
                data_stream, contextual_features
            )
            anomalies.extend(domain_anomalies)
            
            # Update context memory
            await self._update_context_memory(data_stream, anomalies)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Contextual anomaly detection failed: {e}")
            return []
    
    async def _prepare_contextual_features(
        self,
        data_stream: GeologicalDataStream,
        context_window: Optional[int] = None
    ) -> pd.DataFrame:
        """Prepare contextual features for anomaly detection"""
        try:
            context_data = []
            window_size = context_window or self.settings.contextual_memory_size
            
            for i, point in enumerate(data_stream.data_points):
                # Basic features
                features = {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'quality_score': point.quality_score,
                    'measurement_type': point.measurement_type,
                    'sensor_type': point.sensor_type
                }
                
                # Spatial context
                location = point.location
                features.update({
                    'latitude': location.get('lat', 0.0),
                    'longitude': location.get('lon', 0.0),
                    'elevation': location.get('elevation', 0.0),
                    'depth': location.get('depth', 0.0)
                })
                
                # Geological context
                geological_context = self.domain_knowledge.get_geological_context(location)
                features['geological_context'] = geological_context
                
                # Expected background level
                expected_background = self.domain_knowledge.get_expected_background(
                    point.measurement_type, geological_context
                )
                features['expected_background'] = expected_background
                features['deviation_from_background'] = abs(point.value - expected_background)
                
                # Temporal context
                features.update({
                    'hour': point.timestamp.hour,
                    'day_of_week': point.timestamp.weekday(),
                    'month': point.timestamp.month,
                    'season': self._get_season(point.timestamp.month)
                })
                
                # Historical context (if available)
                historical_context = await self._get_historical_context(
                    point, window_size
                )
                features.update(historical_context)
                
                # Neighboring data context
                if i > 0 and i < len(data_stream.data_points) - 1:
                    prev_point = data_stream.data_points[i-1]
                    next_point = data_stream.data_points[i+1]
                    
                    features.update({
                        'temporal_gradient_prev': (point.value - prev_point.value) / max(1, (point.timestamp - prev_point.timestamp).total_seconds()),
                        'temporal_gradient_next': (next_point.value - point.value) / max(1, (next_point.timestamp - point.timestamp).total_seconds()),
                        'spatial_distance_prev': self._calculate_spatial_distance(point.location, prev_point.location),
                        'spatial_distance_next': self._calculate_spatial_distance(point.location, next_point.location)
                    })
                else:
                    features.update({
                        'temporal_gradient_prev': 0.0,
                        'temporal_gradient_next': 0.0,
                        'spatial_distance_prev': 0.0,
                        'spatial_distance_next': 0.0
                    })
                
                # Environmental context
                env_context = self._get_environmental_context(point.timestamp, location)
                features.update(env_context)
                
                context_data.append(features)
            
            return pd.DataFrame(context_data)
            
        except Exception as e:
            logger.error(f"Contextual feature preparation failed: {e}")
            return pd.DataFrame()
    
    async def _detect_neural_contextual_anomalies(
        self,
        data_stream: GeologicalDataStream,
        contextual_features: pd.DataFrame
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using neural contextual model"""
        anomalies = []
        
        try:
            if self.contextual_model is None or contextual_features.empty:
                return anomalies
            
            # Prepare features for neural network
            feature_columns = [
                'value', 'quality_score', 'latitude', 'longitude', 'elevation', 'depth',
                'deviation_from_background', 'temporal_gradient_prev', 'temporal_gradient_next'
            ]
            
            context_columns = [
                'expected_background', 'hour', 'day_of_week', 'month', 'season',
                'spatial_distance_prev', 'spatial_distance_next'
            ]
            
            # Add environmental context columns
            env_columns = [col for col in contextual_features.columns if col.startswith('env_')]
            context_columns.extend(env_columns)
            
            # Ensure all columns exist
            for col in feature_columns + context_columns:
                if col not in contextual_features.columns:
                    contextual_features[col] = 0.0
            
            features = contextual_features[feature_columns].fillna(0).values
            context = contextual_features[context_columns].fillna(0).values
            
            # Scale features
            if self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)
            if self.context_scaler is not None:
                context = self.context_scaler.transform(context)
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(features).to(self.device)
            context_tensor = torch.FloatTensor(context).to(self.device)
            
            # Get anomaly probabilities
            with torch.no_grad():
                anomaly_probs = self.contextual_model(features_tensor, context_tensor)
                anomaly_probs = anomaly_probs.cpu().numpy().flatten()
            
            # Identify anomalies
            threshold = 1 - self.settings.anomaly_threshold
            anomaly_indices = np.where(anomaly_probs > threshold)[0]
            
            for idx in anomaly_indices:
                anomaly = self._create_contextual_anomaly(
                    data_stream.data_points[idx],
                    contextual_features.iloc[idx],
                    float(anomaly_probs[idx]),
                    "neural_contextual"
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Neural contextual detection failed: {e}")
        
        return anomalies
    
    async def _detect_lof_contextual_anomalies(
        self,
        data_stream: GeologicalDataStream,
        contextual_features: pd.DataFrame
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using Local Outlier Factor with contextual features"""
        anomalies = []
        
        try:
            if contextual_features.empty:
                return anomalies
            
            # Prepare features for LOF
            numeric_columns = contextual_features.select_dtypes(include=[np.number]).columns
            features = contextual_features[numeric_columns].fillna(0).values
            
            if len(features) < 5:  # LOF needs minimum samples
                return anomalies
            
            # Apply LOF
            lof = LocalOutlierFactor(
                n_neighbors=min(20, len(features) - 1),
                contamination='auto',
                novelty=False
            )
            
            outlier_labels = lof.fit_predict(features)
            outlier_scores = -lof.negative_outlier_factor_
            
            # Identify anomalies
            anomaly_indices = np.where(outlier_labels == -1)[0]
            
            for idx in anomaly_indices:
                anomaly = self._create_contextual_anomaly(
                    data_stream.data_points[idx],
                    contextual_features.iloc[idx],
                    float(outlier_scores[idx]),
                    "lof_contextual"
                )
                anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"LOF contextual detection failed: {e}")
        
        return anomalies
    
    async def _detect_domain_knowledge_anomalies(
        self,
        data_stream: GeologicalDataStream,
        contextual_features: pd.DataFrame
    ) -> List[DetectedAnomaly]:
        """Detect anomalies using geological domain knowledge"""
        anomalies = []
        
        try:
            for i, (point, context_row) in enumerate(zip(data_stream.data_points, contextual_features.itertuples())):
                # Get geological context
                geological_context = context_row.geological_context
                
                # Check against domain knowledge thresholds
                deviation = context_row.deviation_from_background
                measurement_type = point.measurement_type
                
                # Determine severity based on domain knowledge
                severity_levels = ['low', 'medium', 'high', 'critical']
                detected_severity = None
                
                for severity in severity_levels:
                    threshold = self.domain_knowledge.get_significance_threshold(
                        measurement_type, geological_context, severity
                    )
                    if deviation > threshold:
                        detected_severity = severity
                
                if detected_severity:
                    # Calculate anomaly score based on how much it exceeds threshold
                    base_threshold = self.domain_knowledge.get_significance_threshold(
                        measurement_type, geological_context, 'low'
                    )
                    anomaly_score = min(1.0, deviation / (base_threshold * 10))
                    
                    anomaly = self._create_contextual_anomaly(
                        point,
                        context_row,
                        anomaly_score,
                        "domain_knowledge",
                        detected_severity
                    )
                    anomalies.append(anomaly)
            
        except Exception as e:
            logger.error(f"Domain knowledge detection failed: {e}")
        
        return anomalies
    
    def _create_contextual_anomaly(
        self,
        data_point,
        context_row,
        anomaly_score: float,
        detection_method: str,
        severity_override: Optional[str] = None
    ) -> DetectedAnomaly:
        """Create a contextual anomaly object"""
        
        # Determine severity
        if severity_override:
            severity_map = {
                'low': AnomalySeverity.LOW,
                'medium': AnomalySeverity.MEDIUM,
                'high': AnomalySeverity.HIGH,
                'critical': AnomalySeverity.CRITICAL
            }
            severity = severity_map.get(severity_override, AnomalySeverity.MEDIUM)
        else:
            if anomaly_score > 0.9:
                severity = AnomalySeverity.CRITICAL
            elif anomaly_score > 0.8:
                severity = AnomalySeverity.HIGH
            elif anomaly_score > 0.7:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW
        
        # Create geological context description
        geological_context = getattr(context_row, 'geological_context', 'unknown')
        expected_background = getattr(context_row, 'expected_background', 0.0)
        deviation = getattr(context_row, 'deviation_from_background', 0.0)
        
        context_description = (
            f"Contextual anomaly in {geological_context} setting. "
            f"Expected background: {expected_background:.2f}, "
            f"Observed deviation: {deviation:.2f}"
        )
        
        return DetectedAnomaly(
            anomaly_id=str(uuid.uuid4()),
            detection_timestamp=datetime.utcnow(),
            anomaly_type=AnomalyType.CONTEXTUAL,
            severity=severity,
            temporal_extent={
                "start": data_point.timestamp,
                "end": data_point.timestamp
            },
            spatial_extent=data_point.location,
            affected_variables=[data_point.measurement_type],
            anomaly_score=anomaly_score,
            deviation_magnitude=deviation,
            baseline_comparison={
                "expected_background": expected_background,
                "geological_context": geological_context,
                "detection_method": detection_method
            },
            geological_context=context_description,
            detection_method=f"contextual_{detection_method}",
            model_version="1.0.0",
            processing_time=0.0
        )
    
    async def _get_historical_context(
        self, 
        point, 
        window_size: int
    ) -> Dict[str, float]:
        """Get historical context for a data point"""
        try:
            # Query historical data
            end_time = point.timestamp
            start_time = end_time - timedelta(days=7)  # Look back 7 days
            
            historical_data = await self.db_manager.query_geological_data(
                measurement_type=point.measurement_type,
                start_time=start_time,
                end_time=end_time,
                location_filter=point.location
            )
            
            if not historical_data:
                return {
                    'historical_mean': point.value,
                    'historical_std': 0.0,
                    'historical_trend': 0.0,
                    'historical_count': 0
                }
            
            values = [d['value'] for d in historical_data]
            
            return {
                'historical_mean': np.mean(values),
                'historical_std': np.std(values),
                'historical_trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0.0,
                'historical_count': len(values)
            }
            
        except Exception as e:
            logger.error(f"Historical context retrieval failed: {e}")
            return {
                'historical_mean': 0.0,
                'historical_std': 0.0,
                'historical_trend': 0.0,
                'historical_count': 0
            }
    
    def _get_season(self, month: int) -> int:
        """Get season from month (0=winter, 1=spring, 2=summer, 3=autumn)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def _calculate_spatial_distance(self, loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
        """Calculate spatial distance between two locations"""
        lat1, lon1 = loc1.get('lat', 0.0), loc1.get('lon', 0.0)
        lat2, lon2 = loc2.get('lat', 0.0), loc2.get('lon', 0.0)
        
        # Simple Euclidean distance (for small distances)
        return np.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2)
    
    def _get_environmental_context(
        self, 
        timestamp: datetime, 
        location: Dict[str, float]
    ) -> Dict[str, float]:
        """Get environmental context factors"""
        # Simplified environmental context
        # In practice, this would integrate with weather APIs, tidal data, etc.
        
        hour = timestamp.hour
        month = timestamp.month
        
        return {
            'env_diurnal_phase': np.sin(2 * np.pi * hour / 24),  # Diurnal cycle
            'env_seasonal_phase': np.sin(2 * np.pi * month / 12),  # Seasonal cycle
            'env_atmospheric_pressure': 1013.25,  # Standard pressure (would be real data)
            'env_temperature': 15.0 + 10 * np.sin(2 * np.pi * month / 12),  # Seasonal temp
            'env_tidal_phase': np.sin(2 * np.pi * (hour + timestamp.minute/60) / 12.42)  # Tidal cycle
        }
    
    async def _initialize_models(self):
        """Initialize contextual detection models"""
        try:
            # Initialize neural contextual model
            feature_dim = 9  # Basic features
            context_dim = 12  # Context features
            
            self.contextual_model = GeologicalContextualModel(
                feature_dim=feature_dim,
                context_dim=context_dim,
                hidden_dim=128
            )
            self.contextual_model.to(self.device)
            
            # Initialize scalers
            self.feature_scaler = StandardScaler()
            self.context_scaler = StandardScaler()
            
            logger.info("Contextual models initialized")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
    
    async def _load_context_memory(self):
        """Load context memory from database"""
        try:
            # Load recent anomalies for context
            recent_anomalies = await self.db_manager.get_recent_anomalies(
                hours=24 * 7,  # Last week
                anomaly_type="contextual"
            )
            
            self.context_memory = {
                'recent_anomalies': recent_anomalies,
                'loaded_at': datetime.utcnow()
            }
            
            logger.info(f"Context memory loaded with {len(recent_anomalies)} recent anomalies")
            
        except Exception as e:
            logger.warning(f"Context memory loading failed: {e}")
    
    async def _update_context_memory(self, data_stream, anomalies):
        """Update context memory with new data and anomalies"""
        try:
            # Update recent anomalies
            if 'recent_anomalies' not in self.context_memory:
                self.context_memory['recent_anomalies'] = []
            
            # Add new anomalies to memory
            for anomaly in anomalies:
                self.context_memory['recent_anomalies'].append({
                    'anomaly_id': anomaly.anomaly_id,
                    'timestamp': anomaly.detection_timestamp,
                    'location': anomaly.spatial_extent,
                    'measurement_type': anomaly.affected_variables[0] if anomaly.affected_variables else 'unknown',
                    'severity': anomaly.severity.value,
                    'score': anomaly.anomaly_score
                })
            
            # Keep memory size limited
            if len(self.context_memory['recent_anomalies']) > self.memory_size:
                self.context_memory['recent_anomalies'] = self.context_memory['recent_anomalies'][-self.memory_size:]
            
            self.context_memory['updated_at'] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Context memory update failed: {e}")
    
    def get_detector_info(self) -> Dict[str, Any]:
        """Get information about the contextual detector"""
        return {
            "models_initialized": self.contextual_model is not None,
            "context_memory_size": len(self.context_memory.get('recent_anomalies', [])),
            "domain_knowledge_processes": len(self.domain_knowledge.process_relationships),
            "geological_contexts": list(self.domain_knowledge.significance_thresholds.keys()),
            "last_memory_update": self.context_memory.get('updated_at', 'never')
        }