"""
Real Data Integration Pipeline for Multi-Source Geological Data Processing
Comprehensive system for real-time volcanic data enhancement and geological data integration

This module replaces all simulated/synthetic data with authentic geological observations
from authoritative sources including IRIS/EarthScope, USGS, NASA, Copernicus, EarthChem/GEOROC.

Requirements: 6.1, 6.2, 6.3, 2.1, 2.2
"""

import asyncio
import logging
import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal, stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
import json
import pickle
import hashlib
import time
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_manager import get_database_manager, DatabaseType
from core.real_data_connectors import get_real_data_connector_manager, GeologicalObservation, DataSourceType

warnings.filterwarnings('ignore')

# Configure advanced logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Enumeration of supported geological data types"""
    SEISMIC = "seismic"
    GRAVITY = "gravity"
    MAGNETIC = "magnetic"
    ELECTROMAGNETIC = "electromagnetic"
    DEFORMATION = "deformation"
    GEOCHEMICAL = "geochemical"
    THERMAL = "thermal"
    GEOLOGICAL = "geological"
    REMOTE_SENSING = "remote_sensing"
    DEM = "digital_elevation_model"

class ProcessingLevel(Enum):
    """Data processing levels"""
    RAW = "raw"
    CALIBRATED = "calibrated"
    PROCESSED = "processed"
    ANALYZED = "analyzed"
    INTEGRATED = "integrated"

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    enable_quality_control: bool = True
    enable_anomaly_detection: bool = True
    enable_cross_validation: bool = True
    enable_uncertainty_quantification: bool = True
    processing_level: ProcessingLevel = ProcessingLevel.PROCESSED
    temporal_resolution: timedelta = timedelta(minutes=1)
    spatial_resolution: float = 1.0  # km
    quality_threshold: float = 0.7
    max_processing_time: int = 300  # seconds

@dataclass
class DataLayer:
    """Geological data layer with metadata"""
    layer_id: str
    data_type: DataType
    data: Union[np.ndarray, pd.DataFrame, Dict]
    metadata: Dict[str, Any]
    quality_metrics: Dict[str, float]
    processing_history: List[str]
    timestamp: datetime
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    temporal_bounds: Optional[Tuple[datetime, datetime]] = None
    uncertainty: Optional[Dict[str, float]] = None

class RealDataIntegrationPipeline:
    """
    Advanced data integration pipeline for multi-source geological data processing.
    
    This pipeline integrates real geological data from multiple authoritative sources:
    - IRIS/EarthScope for seismic data
    - USGS for volcanic and earthquake data
    - NASA EarthData for satellite observations
    - EarthChem/GEOROC for geochemical data
    - National geological surveys for geological maps and surveys
    
    All synthetic/simulated data generation has been removed and replaced with
    real data connectors and processing algorithms.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.data_layers = {}
        self.processing_cache = {}
        self.quality_metrics = {}
        self.connector_manager = None
        self.db_manager = None
        
    async def initialize(self):
        """Initialize the data integration pipeline"""
        try:
            # Initialize real data connectors
            self.connector_manager = await get_real_data_connector_manager()
            
            # Initialize database manager
            self.db_manager = await get_database_manager()
            
            logger.info("Real data integration pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {e}")
            raise
    
    async def process_multi_source_data(self, 
                                      start_time: datetime, 
                                      end_time: datetime,
                                      latitude: float, 
                                      longitude: float, 
                                      radius_km: float = 100) -> Dict[str, DataLayer]:
        """
        Process multi-source geological data for specified region and time period.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            
        Returns:
            Dictionary of processed data layers by data type
        """
        try:
            logger.info(f"Processing multi-source data for region ({latitude}, {longitude}) ±{radius_km}km")
            
            # Fetch real data from all sources
            raw_data = await self.connector_manager.fetch_all_data(
                start_time=start_time,
                end_time=end_time,
                latitude=latitude,
                longitude=longitude,
                radius_km=radius_km
            )
            
            processed_layers = {}
            
            # Process each data type
            for data_type, observations in raw_data.items():
                if observations:
                    layer = await self._process_data_type(data_type, observations, start_time, end_time)
                    if layer:
                        processed_layers[data_type] = layer
                        
                        # Store processed data in database
                        await self._store_processed_layer(layer)
            
            # Perform cross-source validation and integration
            if len(processed_layers) > 1:
                integrated_layer = await self._integrate_multi_source_data(processed_layers)
                processed_layers['integrated'] = integrated_layer
            
            logger.info(f"Successfully processed {len(processed_layers)} data layers")
            return processed_layers
            
        except Exception as e:
            logger.error(f"Error processing multi-source data: {e}")
            return {}
    
    async def _process_data_type(self, data_type: str, observations: List[GeologicalObservation], 
                               start_time: datetime, end_time: datetime) -> Optional[DataLayer]:
        """Process observations for a specific data type"""
        try:
            if not observations:
                return None
            
            logger.info(f"Processing {len(observations)} {data_type} observations")
            
            # Convert observations to structured data
            processed_data = await self._convert_observations_to_structured_data(observations)
            
            # Apply quality control
            if self.config.enable_quality_control:
                processed_data = await self._apply_quality_control(processed_data, data_type)
            
            # Detect and handle anomalies
            if self.config.enable_anomaly_detection:
                anomaly_flags = await self._detect_anomalies(processed_data, data_type)
                processed_data['anomaly_flags'] = anomaly_flags
            
            # Calculate quality metrics
            quality_metrics = await self._calculate_quality_metrics(processed_data, observations)
            
            # Calculate uncertainty
            uncertainty = await self._calculate_uncertainty(processed_data, observations)
            
            # Create data layer
            layer = DataLayer(
                layer_id=f"{data_type}_{int(time.time())}",
                data_type=DataType(data_type.upper()) if hasattr(DataType, data_type.upper()) else DataType.SEISMIC,
                data=processed_data,
                metadata={
                    'source_count': len(observations),
                    'time_range': (start_time, end_time),
                    'processing_timestamp': datetime.now(timezone.utc),
                    'processing_config': self.config.__dict__
                },
                quality_metrics=quality_metrics,
                processing_history=[f"Processed {len(observations)} {data_type} observations"],
                timestamp=datetime.now(timezone.utc),
                uncertainty=uncertainty
            )
            
            return layer
            
        except Exception as e:
            logger.error(f"Error processing {data_type} data: {e}")
            return None
    
    async def _convert_observations_to_structured_data(self, observations: List[GeologicalObservation]) -> Dict[str, Any]:
        """Convert raw observations to structured data format"""
        try:
            # Create structured arrays for efficient processing
            timestamps = [obs.timestamp for obs in observations]
            latitudes = [obs.latitude for obs in observations]
            longitudes = [obs.longitude for obs in observations]
            elevations = [obs.elevation or 0 for obs in observations]
            
            # Extract parameters
            all_parameters = {}
            for obs in observations:
                for param, value in obs.parameters.items():
                    if param not in all_parameters:
                        all_parameters[param] = []
                    all_parameters[param].append(value)
            
            # Pad parameter arrays to same length
            max_length = len(observations)
            for param in all_parameters:
                while len(all_parameters[param]) < max_length:
                    all_parameters[param].append(np.nan)
            
            structured_data = {
                'timestamps': np.array(timestamps),
                'coordinates': np.column_stack([latitudes, longitudes, elevations]),
                'parameters': all_parameters,
                'observation_ids': [obs.observation_id for obs in observations],
                'sources': [obs.source for obs in observations],
                'quality_flags': [obs.quality_flags for obs in observations]
            }
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error converting observations to structured data: {e}")
            return {}
    
    async def _apply_quality_control(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """Apply quality control filters to data"""
        try:
            # Remove data points with poor quality flags
            quality_mask = np.ones(len(data['timestamps']), dtype=bool)
            
            for i, flags in enumerate(data['quality_flags']):
                if flags and len(flags) > 2:  # More than 2 quality issues
                    quality_mask[i] = False
            
            # Apply statistical outlier detection
            for param_name, param_values in data['parameters'].items():
                if param_name in ['magnitude', 'depth_km', 'temperature']:
                    param_array = np.array(param_values)
                    
                    # Remove statistical outliers (beyond 3 sigma)
                    if len(param_array) > 10:
                        z_scores = np.abs(stats.zscore(param_array, nan_policy='omit'))
                        outlier_mask = z_scores > 3
                        quality_mask &= ~outlier_mask
            
            # Apply quality mask to all data arrays
            if np.any(quality_mask):
                data['timestamps'] = data['timestamps'][quality_mask]
                data['coordinates'] = data['coordinates'][quality_mask]
                data['observation_ids'] = [data['observation_ids'][i] for i in range(len(quality_mask)) if quality_mask[i]]
                data['sources'] = [data['sources'][i] for i in range(len(quality_mask)) if quality_mask[i]]
                data['quality_flags'] = [data['quality_flags'][i] for i in range(len(quality_mask)) if quality_mask[i]]
                
                for param_name in data['parameters']:
                    param_array = np.array(data['parameters'][param_name])
                    data['parameters'][param_name] = param_array[quality_mask].tolist()
            
            logger.info(f"Quality control: retained {np.sum(quality_mask)}/{len(quality_mask)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error applying quality control: {e}")
            return data
    
    async def _detect_anomalies(self, data: Dict[str, Any], data_type: str) -> List[bool]:
        """Detect anomalies in the data using multiple algorithms"""
        try:
            anomaly_flags = [False] * len(data['timestamps'])
            
            # Prepare feature matrix for anomaly detection
            features = []
            for param_name, param_values in data['parameters'].items():
                if param_name in ['magnitude', 'depth_km', 'temperature', 'pressure']:
                    param_array = np.array(param_values)
                    # Handle NaN values
                    param_array = np.nan_to_num(param_array, nan=np.nanmean(param_array))
                    features.append(param_array)
            
            if features and len(features[0]) > 10:
                feature_matrix = np.column_stack(features)
                
                # Apply Isolation Forest for anomaly detection
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_scores = iso_forest.fit_predict(feature_matrix)
                
                # Convert to boolean flags
                anomaly_flags = (anomaly_scores == -1).tolist()
                
                logger.info(f"Anomaly detection: found {sum(anomaly_flags)} anomalies in {len(anomaly_flags)} data points")
            
            return anomaly_flags
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return [False] * len(data['timestamps'])
    
    async def _calculate_quality_metrics(self, data: Dict[str, Any], observations: List[GeologicalObservation]) -> Dict[str, float]:
        """Calculate quality metrics for the processed data"""
        try:
            metrics = {}
            
            # Data completeness
            total_expected = len(observations)
            actual_count = len(data['timestamps'])
            metrics['completeness'] = actual_count / total_expected if total_expected > 0 else 0
            
            # Temporal coverage
            if data['timestamps'].size > 0:
                time_span = (data['timestamps'][-1] - data['timestamps'][0]).total_seconds()
                metrics['temporal_coverage'] = min(1.0, time_span / 3600)  # Normalize to hours
            else:
                metrics['temporal_coverage'] = 0
            
            # Spatial distribution
            if data['coordinates'].size > 0:
                coords = data['coordinates']
                lat_range = np.ptp(coords[:, 0])
                lon_range = np.ptp(coords[:, 1])
                metrics['spatial_distribution'] = min(1.0, (lat_range + lon_range) / 2)
            else:
                metrics['spatial_distribution'] = 0
            
            # Source diversity
            unique_sources = len(set(data['sources']))
            metrics['source_diversity'] = min(1.0, unique_sources / 5)  # Normalize to max 5 sources
            
            # Overall quality score
            metrics['overall_quality'] = np.mean([
                metrics['completeness'],
                metrics['temporal_coverage'],
                metrics['spatial_distribution'],
                metrics['source_diversity']
            ])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {'overall_quality': 0.0}
    
    async def _calculate_uncertainty(self, data: Dict[str, Any], observations: List[GeologicalObservation]) -> Dict[str, float]:
        """Calculate uncertainty estimates for the data"""
        try:
            uncertainty = {}
            
            # Parameter-specific uncertainties
            for param_name, param_values in data['parameters'].items():
                if param_name in ['magnitude', 'depth_km', 'temperature']:
                    param_array = np.array(param_values)
                    param_array = param_array[~np.isnan(param_array)]
                    
                    if len(param_array) > 1:
                        # Calculate standard error
                        uncertainty[param_name] = np.std(param_array) / np.sqrt(len(param_array))
                    else:
                        uncertainty[param_name] = 0.1  # Default uncertainty
            
            # Spatial uncertainty
            if data['coordinates'].size > 0:
                coords = data['coordinates']
                lat_std = np.std(coords[:, 0])
                lon_std = np.std(coords[:, 1])
                uncertainty['spatial'] = np.sqrt(lat_std**2 + lon_std**2)
            
            # Temporal uncertainty
            if len(data['timestamps']) > 1:
                time_diffs = np.diff([t.timestamp() for t in data['timestamps']])
                uncertainty['temporal'] = np.std(time_diffs)
            
            return uncertainty
            
        except Exception as e:
            logger.error(f"Error calculating uncertainty: {e}")
            return {}
    
    async def _integrate_multi_source_data(self, layers: Dict[str, DataLayer]) -> DataLayer:
        """Integrate data from multiple sources into a unified layer"""
        try:
            logger.info(f"Integrating {len(layers)} data layers")
            
            # Combine all data into unified structure
            integrated_data = {
                'layers': {},
                'cross_correlations': {},
                'temporal_alignment': {},
                'spatial_alignment': {}
            }
            
            # Store individual layers
            for layer_name, layer in layers.items():
                integrated_data['layers'][layer_name] = {
                    'data': layer.data,
                    'quality_metrics': layer.quality_metrics,
                    'uncertainty': layer.uncertainty
                }
            
            # Calculate cross-correlations between layers
            layer_names = list(layers.keys())
            for i, layer1_name in enumerate(layer_names):
                for j, layer2_name in enumerate(layer_names[i+1:], i+1):
                    correlation = await self._calculate_cross_correlation(
                        layers[layer1_name], layers[layer2_name]
                    )
                    integrated_data['cross_correlations'][f"{layer1_name}_{layer2_name}"] = correlation
            
            # Calculate overall quality metrics
            overall_quality = {}
            for metric_name in ['completeness', 'temporal_coverage', 'spatial_distribution', 'overall_quality']:
                values = [layer.quality_metrics.get(metric_name, 0) for layer in layers.values()]
                overall_quality[metric_name] = np.mean(values)
            
            # Create integrated layer
            integrated_layer = DataLayer(
                layer_id=f"integrated_{int(time.time())}",
                data_type=DataType.GEOLOGICAL,
                data=integrated_data,
                metadata={
                    'integrated_layers': list(layers.keys()),
                    'integration_timestamp': datetime.now(timezone.utc),
                    'integration_method': 'multi_source_fusion'
                },
                quality_metrics=overall_quality,
                processing_history=[f"Integrated {len(layers)} data layers"],
                timestamp=datetime.now(timezone.utc)
            )
            
            return integrated_layer
            
        except Exception as e:
            logger.error(f"Error integrating multi-source data: {e}")
            return None
    
    async def _calculate_cross_correlation(self, layer1: DataLayer, layer2: DataLayer) -> Dict[str, float]:
        """Calculate cross-correlation between two data layers"""
        try:
            correlation = {}
            
            # Temporal correlation
            if 'timestamps' in layer1.data and 'timestamps' in layer2.data:
                # Find overlapping time periods
                times1 = [t.timestamp() for t in layer1.data['timestamps']]
                times2 = [t.timestamp() for t in layer2.data['timestamps']]
                
                if times1 and times2:
                    overlap_start = max(min(times1), min(times2))
                    overlap_end = min(max(times1), max(times2))
                    
                    if overlap_end > overlap_start:
                        correlation['temporal_overlap'] = (overlap_end - overlap_start) / (max(max(times1), max(times2)) - min(min(times1), min(times2)))
                    else:
                        correlation['temporal_overlap'] = 0
            
            # Spatial correlation
            if 'coordinates' in layer1.data and 'coordinates' in layer2.data:
                coords1 = layer1.data['coordinates']
                coords2 = layer2.data['coordinates']
                
                if coords1.size > 0 and coords2.size > 0:
                    # Calculate average distance between coordinate sets
                    center1 = np.mean(coords1, axis=0)
                    center2 = np.mean(coords2, axis=0)
                    
                    # Haversine distance for lat/lon
                    lat1, lon1 = np.radians(center1[:2])
                    lat2, lon2 = np.radians(center2[:2])
                    
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
                    
                    # Normalize distance to correlation (closer = higher correlation)
                    correlation['spatial_proximity'] = max(0, 1 - distance_km / 1000)  # Normalize to 1000km
            
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating cross-correlation: {e}")
            return {}
    
    async def _store_processed_layer(self, layer: DataLayer):
        """Store processed data layer in database"""
        try:
            # Store in PostgreSQL for structured queries
            await self.db_manager.execute_query(
                'postgresql',
                """
                INSERT INTO processed_data_layers 
                (layer_id, data_type, quality_score, processing_timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                {
                    'layer_id': layer.layer_id,
                    'data_type': layer.data_type.value,
                    'quality_score': layer.quality_metrics.get('overall_quality', 0),
                    'processing_timestamp': layer.timestamp,
                    'metadata': json.dumps(layer.metadata)
                }
            )
            
            # Store time-series data in InfluxDB
            if 'timestamps' in layer.data and 'parameters' in layer.data:
                for i, timestamp in enumerate(layer.data['timestamps']):
                    tags = {
                        'layer_id': layer.layer_id,
                        'data_type': layer.data_type.value
                    }
                    
                    fields = {}
                    for param_name, param_values in layer.data['parameters'].items():
                        if i < len(param_values) and not np.isnan(param_values[i]):
                            fields[param_name] = param_values[i]
                    
                    if fields:
                        await self.db_manager.write_time_series(
                            'influxdb',
                            measurement='processed_geological_data',
                            tags=tags,
                            fields=fields,
                            timestamp=timestamp
                        )
            
            # Cache layer metadata in Redis
            cache_key = f"layer:{layer.layer_id}"
            await self.db_manager.cache_set('redis', cache_key, {
                'layer_id': layer.layer_id,
                'data_type': layer.data_type.value,
                'quality_metrics': layer.quality_metrics,
                'timestamp': layer.timestamp.isoformat()
            }, ttl=3600)
            
            logger.info(f"Stored processed layer {layer.layer_id} in databases")
            
        except Exception as e:
            logger.error(f"Error storing processed layer: {e}")

# Global pipeline instance
_pipeline_instance: Optional[RealDataIntegrationPipeline] = None

async def get_data_integration_pipeline(config: Optional[ProcessingConfig] = None) -> RealDataIntegrationPipeline:
    """Get or create global data integration pipeline instance"""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        if config is None:
            config = ProcessingConfig()
        
        _pipeline_instance = RealDataIntegrationPipeline(config)
        await _pipeline_instance.initialize()
    
    return _pipeline_instance