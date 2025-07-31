"""
Multi-scale analyzer for geological anomaly detection

This module implements multi-scale analysis capabilities that detect anomalies
across different temporal and spatial scales in geological data.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import scipy.signal as signal
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform

from core.config import Settings
from core.database import DatabaseManager
from models.schemas import (
    GeologicalDataStream, DetectedAnomaly, AnomalyType, AnomalySeverity,
    MultiScaleAnomaly
)

logger = logging.getLogger(__name__)


class WaveletTransform:
    """Wavelet transform for multi-scale time series analysis"""
    
    def __init__(self, wavelet_type: str = 'morlet'):
        self.wavelet_type = wavelet_type
        self.scales = np.logspace(0, 3, 50)  # 50 scales from 1 to 1000
    
    def continuous_wavelet_transform(self, signal_data: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Perform continuous wavelet transform"""
        try:
            import pywt
            
            # Perform CWT
            coefficients, frequencies = pywt.cwt(
                signal_data, 
                self.scales, 
                self.wavelet_type,
                dt
            )
            
            return coefficients, frequencies
            
        except ImportError:
            logger.warning("PyWavelets not available, using simplified wavelet transform")
            return self._simple_wavelet_transform(signal_data, dt)
        except Exception as e:
            logger.error(f"Wavelet transform failed: {e}")
            return np.array([]), np.array([])
    
    def _simple_wavelet_transform(self, signal_data: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Simplified wavelet transform using Gabor wavelets"""
        try:
            n_samples = len(signal_data)
            n_scales = len(self.scales)
            
            coefficients = np.zeros((n_scales, n_samples), dtype=complex)
            frequencies = 1.0 / (self.scales * dt)
            
            for i, scale in enumerate(self.scales):
                # Create Gabor wavelet
                sigma = scale / 4.0
                t = np.arange(-3*sigma, 3*sigma, dt)
                wavelet = np.exp(1j * 2 * np.pi * t / scale) * np.exp(-t**2 / (2 * sigma**2))
                
                # Convolve with signal
                coefficients[i, :] = np.convolve(signal_data, wavelet, mode='same')
            
            return coefficients, frequencies
            
        except Exception as e:
            logger.error(f"Simple wavelet transform failed: {e}")
            return np.array([]), np.array([])
    
    def detect_scale_anomalies(self, coefficients: np.ndarray, threshold: float = 3.0) -> List[Dict[str, Any]]:
        """Detect anomalies across different scales"""
        anomalies = []
        
        try:
            # Calculate power spectrum
            power = np.abs(coefficients) ** 2
            
            # Detect anomalies at each scale
            for scale_idx, scale_power in enumerate(power):
                # Calculate z-scores
                mean_power = np.mean(scale_power)
                std_power = np.std(scale_power)
                
                if std_power > 0:
                    z_scores = (scale_power - mean_power) / std_power
                    anomaly_indices = np.where(np.abs(z_scores) > threshold)[0]
                    
                    for idx in anomaly_indices:
                        anomalies.append({
                            'scale_index': scale_idx,
                            'scale_value': self.scales[scale_idx],
                            'time_index': idx,
                            'power': scale_power[idx],
                            'z_score': z_scores[idx],
                            'anomaly_strength': abs(z_scores[idx])
                        })
            
        except Exception as e:
            logger.error(f"Scale anomaly detection failed: {e}")
        
        return anomalies


class SpatialScaleAnalyzer:
    """Analyzer for spatial scale anomalies"""
    
    def __init__(self):
        self.spatial_scales = {
            'local': 1.0,      # 1 km
            'regional': 10.0,   # 10 km
            'global': 100.0     # 100 km
        }
    
    def analyze_spatial_scales(
        self, 
        data_points: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze anomalies at different spatial scales"""
        scale_results = {}
        
        try:
            if len(data_points) < 3:
                return scale_results
            
            # Extract spatial coordinates
            coordinates = np.array([
                [point['location']['lat'], point['location']['lon']] 
                for point in data_points
            ])
            values = np.array([point['value'] for point in data_points])
            
            for scale_name, scale_km in self.spatial_scales.items():
                scale_anomalies = self._detect_spatial_anomalies_at_scale(
                    coordinates, values, scale_km, data_points
                )
                scale_results[scale_name] = scale_anomalies
            
        except Exception as e:
            logger.error(f"Spatial scale analysis failed: {e}")
        
        return scale_results
    
    def _detect_spatial_anomalies_at_scale(
        self,
        coordinates: np.ndarray,
        values: np.ndarray,
        scale_km: float,
        data_points: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect spatial anomalies at specific scale"""
        anomalies = []
        
        try:
            # Calculate distance matrix (simplified - assumes flat Earth)
            distances = pdist(coordinates, metric='euclidean')
            distance_matrix = squareform(distances)
            
            # Convert to kilometers (rough approximation)
            distance_matrix_km = distance_matrix * 111.0  # 1 degree ≈ 111 km
            
            # For each point, find neighbors within the scale
            for i, (coord, value) in enumerate(zip(coordinates, values)):
                neighbors_mask = distance_matrix_km[i] <= scale_km
                neighbor_values = values[neighbors_mask]
                
                if len(neighbor_values) > 1:
                    # Calculate local statistics
                    local_mean = np.mean(neighbor_values)
                    local_std = np.std(neighbor_values)
                    
                    if local_std > 0:
                        z_score = (value - local_mean) / local_std
                        
                        if abs(z_score) > 2.0:  # Anomaly threshold
                            anomalies.append({
                                'point_index': i,
                                'coordinates': coord.tolist(),
                                'value': value,
                                'local_mean': local_mean,
                                'local_std': local_std,
                                'z_score': z_score,
                                'neighbor_count': len(neighbor_values),
                                'scale_km': scale_km,
                                'anomaly_strength': abs(z_score)
                            })
            
        except Exception as e:
            logger.error(f"Spatial anomaly detection at scale {scale_km} failed: {e}")
        
        return anomalies


class CrossScaleInteractionDetector:
    """Detector for interactions between different scales"""
    
    def __init__(self):
        self.interaction_types = [
            'temporal_spatial',
            'scale_cascade',
            'emergent_behavior',
            'scale_coupling'
        ]
    
    def detect_cross_scale_interactions(
        self,
        temporal_anomalies: Dict[str, List[Dict[str, Any]]],
        spatial_anomalies: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect interactions between temporal and spatial scales"""
        interactions = []
        
        try:
            # Temporal-spatial interactions
            temp_spatial_interactions = self._detect_temporal_spatial_interactions(
                temporal_anomalies, spatial_anomalies
            )
            interactions.extend(temp_spatial_interactions)
            
            # Scale cascade effects
            cascade_interactions = self._detect_scale_cascades(
                temporal_anomalies, spatial_anomalies
            )
            interactions.extend(cascade_interactions)
            
            # Emergent behaviors
            emergent_interactions = self._detect_emergent_behaviors(
                temporal_anomalies, spatial_anomalies
            )
            interactions.extend(emergent_interactions)
            
        except Exception as e:
            logger.error(f"Cross-scale interaction detection failed: {e}")
        
        return interactions
    
    def _detect_temporal_spatial_interactions(
        self,
        temporal_anomalies: Dict[str, List[Dict[str, Any]]],
        spatial_anomalies: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect temporal-spatial scale interactions"""
        interactions = []
        
        try:
            # Look for correlations between temporal and spatial anomalies
            for temp_scale, temp_anomalies in temporal_anomalies.items():
                for spatial_scale, spat_anomalies in spatial_anomalies.items():
                    
                    if len(temp_anomalies) > 0 and len(spat_anomalies) > 0:
                        # Calculate correlation between anomaly strengths
                        temp_strengths = [a.get('anomaly_strength', 0) for a in temp_anomalies]
                        spat_strengths = [a.get('anomaly_strength', 0) for a in spat_anomalies]
                        
                        # Align lengths
                        min_len = min(len(temp_strengths), len(spat_strengths))
                        if min_len > 3:
                            correlation = np.corrcoef(
                                temp_strengths[:min_len], 
                                spat_strengths[:min_len]
                            )[0, 1]
                            
                            if not np.isnan(correlation) and abs(correlation) > 0.7:
                                interactions.append({
                                    'interaction_type': 'temporal_spatial',
                                    'temporal_scale': temp_scale,
                                    'spatial_scale': spatial_scale,
                                    'correlation': correlation,
                                    'strength': abs(correlation),
                                    'temporal_anomalies': len(temp_anomalies),
                                    'spatial_anomalies': len(spat_anomalies)
                                })
            
        except Exception as e:
            logger.error(f"Temporal-spatial interaction detection failed: {e}")
        
        return interactions
    
    def _detect_scale_cascades(
        self,
        temporal_anomalies: Dict[str, List[Dict[str, Any]]],
        spatial_anomalies: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect cascade effects between scales"""
        cascades = []
        
        try:
            # Look for patterns where small-scale anomalies lead to large-scale ones
            scale_order = ['seconds', 'minutes', 'hours', 'days']
            
            for i in range(len(scale_order) - 1):
                small_scale = scale_order[i]
                large_scale = scale_order[i + 1]
                
                if (small_scale in temporal_anomalies and 
                    large_scale in temporal_anomalies):
                    
                    small_anomalies = temporal_anomalies[small_scale]
                    large_anomalies = temporal_anomalies[large_scale]
                    
                    if len(small_anomalies) > 0 and len(large_anomalies) > 0:
                        # Check if small-scale anomalies precede large-scale ones
                        cascade_strength = self._calculate_cascade_strength(
                            small_anomalies, large_anomalies
                        )
                        
                        if cascade_strength > 0.6:
                            cascades.append({
                                'interaction_type': 'scale_cascade',
                                'source_scale': small_scale,
                                'target_scale': large_scale,
                                'cascade_strength': cascade_strength,
                                'source_anomalies': len(small_anomalies),
                                'target_anomalies': len(large_anomalies)
                            })
            
        except Exception as e:
            logger.error(f"Scale cascade detection failed: {e}")
        
        return cascades
    
    def _detect_emergent_behaviors(
        self,
        temporal_anomalies: Dict[str, List[Dict[str, Any]]],
        spatial_anomalies: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Detect emergent behaviors from scale interactions"""
        emergent = []
        
        try:
            # Look for patterns that only appear when multiple scales interact
            total_temporal_anomalies = sum(len(anomalies) for anomalies in temporal_anomalies.values())
            total_spatial_anomalies = sum(len(anomalies) for anomalies in spatial_anomalies.values())
            
            if total_temporal_anomalies > 5 and total_spatial_anomalies > 5:
                # Calculate emergent complexity
                scale_diversity = len(temporal_anomalies) + len(spatial_anomalies)
                interaction_complexity = total_temporal_anomalies * total_spatial_anomalies
                
                emergent_score = min(1.0, (scale_diversity * np.log(interaction_complexity)) / 100.0)
                
                if emergent_score > 0.7:
                    emergent.append({
                        'interaction_type': 'emergent_behavior',
                        'emergent_score': emergent_score,
                        'scale_diversity': scale_diversity,
                        'interaction_complexity': interaction_complexity,
                        'temporal_scales_involved': len(temporal_anomalies),
                        'spatial_scales_involved': len(spatial_anomalies)
                    })
            
        except Exception as e:
            logger.error(f"Emergent behavior detection failed: {e}")
        
        return emergent
    
    def _calculate_cascade_strength(
        self,
        source_anomalies: List[Dict[str, Any]],
        target_anomalies: List[Dict[str, Any]]
    ) -> float:
        """Calculate strength of cascade effect between scales"""
        try:
            
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

            source_strength = np.mean([a.get('anomaly_strength', 0) for a in source_anomalies])
            target_strength = np.mean([a.get('anomaly_strength', 0) for a in target_anomalies])
            
            # Cascade strength is higher when source precedes and correlates with target
            if source_strength > 0 and target_strength > 0:
                return min(1.0, target_strength / source_strength)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Cascade strength calculation failed: {e}")
            return 0.0


class MultiScaleAnalyzer:
    """Multi-scale analyzer for geological anomaly detection"""
    
    def __init__(self, settings: Settings, db_manager: DatabaseManager):
        self.settings = settings
        self.db_manager = db_manager
        
        # Analysis components
        self.wavelet_transform = WaveletTransform()
        self.spatial_analyzer = SpatialScaleAnalyzer()
        self.interaction_detector = CrossScaleInteractionDetector()
        
        # Scale definitions
        self.temporal_scales = settings.temporal_scales
        self.spatial_scales = settings.spatial_scales
        
        # Analysis parameters
        self.min_samples_per_scale = 10
        self.anomaly_threshold = 2.5
        self.interaction_threshold = 0.7
    
    async def initialize(self):
        """Initialize the multi-scale analyzer"""
        try:
            logger.info("Initializing multi-scale analyzer...")
            
            # Initialize analysis components
            # (Components are already initialized in __init__)
            
            logger.info("Multi-scale analyzer initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-scale analyzer: {e}")
            raise
    
    async def detect_anomalies(
        self,
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Detect multi-scale anomalies in geological data stream"""
        try:
            if len(data_stream.data_points) < self.min_samples_per_scale:
                return []
            
            # Prepare data for multi-scale analysis
            analysis_data = self._prepare_multiscale_data(data_stream)
            
            detected_anomalies = []
            
            # Temporal scale analysis
            temporal_anomalies = await self._analyze_temporal_scales(analysis_data)
            
            # Spatial scale analysis
            spatial_anomalies = await self._analyze_spatial_scales(analysis_data)
            
            # Cross-scale interaction analysis
            interactions = self.interaction_detector.detect_cross_scale_interactions(
                temporal_anomalies, spatial_anomalies
            )
            
            # Convert scale anomalies to DetectedAnomaly objects
            detected_anomalies.extend(
                await self._create_temporal_anomalies(temporal_anomalies, data_stream)
            )
            detected_anomalies.extend(
                await self._create_spatial_anomalies(spatial_anomalies, data_stream)
            )
            detected_anomalies.extend(
                await self._create_interaction_anomalies(interactions, data_stream)
            )
            
            return detected_anomalies
            
        except Exception as e:
            logger.error(f"Multi-scale anomaly detection failed: {e}")
            return []
    
    def _prepare_multiscale_data(self, data_stream: GeologicalDataStream) -> Dict[str, Any]:
        """Prepare data for multi-scale analysis"""
        try:
            # Convert data stream to structured format
            data_points = []
            for point in data_stream.data_points:
                data_point = {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'quality_score': point.quality_score,
                    'measurement_type': point.measurement_type,
                    'sensor_type': point.sensor_type,
                    'location': point.location
                }
                data_points.append(data_point)
            
            # Create time series
            df = pd.DataFrame(data_points)
            df = df.sort_values('timestamp')
            
            # Calculate time intervals
            time_diffs = df['timestamp'].diff().dt.total_seconds().fillna(1.0)
            avg_dt = np.mean(time_diffs[time_diffs > 0])
            
            return {
                'dataframe': df,
                'values': df['value'].values,
                'timestamps': df['timestamp'].values,
                'locations': df['location'].tolist(),
                'data_points': data_points,
                'avg_dt': avg_dt,
                'measurement_types': df['measurement_type'].unique().tolist()
            }
            
        except Exception as e:
            logger.error(f"Multi-scale data preparation failed: {e}")
            return {}
    
    async def _analyze_temporal_scales(self, analysis_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze anomalies at different temporal scales"""
        temporal_anomalies = {}
        
        try:
            values = analysis_data.get('values', np.array([]))
            avg_dt = analysis_data.get('avg_dt', 1.0)
            
            if len(values) < 10:
                return temporal_anomalies
            
            # Wavelet-based multi-scale analysis
            coefficients, frequencies = self.wavelet_transform.continuous_wavelet_transform(
                values, avg_dt
            )
            
            if coefficients.size > 0:
                scale_anomalies = self.wavelet_transform.detect_scale_anomalies(
                    coefficients, threshold=self.anomaly_threshold
                )
                
                # Group anomalies by temporal scale
                for anomaly in scale_anomalies:
                    scale_value = anomaly['scale_value']
                    
                    # Map scale to temporal category
                    if scale_value < 60:
                        scale_category = 'seconds'
                    elif scale_value < 3600:
                        scale_category = 'minutes'
                    elif scale_value < 86400:
                        scale_category = 'hours'
                    else:
                        scale_category = 'days'
                    
                    if scale_category not in temporal_anomalies:
                        temporal_anomalies[scale_category] = []
                    
                    temporal_anomalies[scale_category].append(anomaly)
            
            # Additional temporal scale analysis using rolling statistics
            for scale in self.temporal_scales:
                if scale in ['seconds', 'minutes', 'hours', 'days']:
                    scale_anomalies = self._detect_rolling_anomalies(
                        values, scale, avg_dt
                    )
                    
                    if scale not in temporal_anomalies:
                        temporal_anomalies[scale] = []
                    temporal_anomalies[scale].extend(scale_anomalies)
            
        except Exception as e:
            logger.error(f"Temporal scale analysis failed: {e}")
        
        return temporal_anomalies
    
    async def _analyze_spatial_scales(self, analysis_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze anomalies at different spatial scales"""
        try:
            data_points = analysis_data.get('data_points', [])
            
            if len(data_points) < 3:
                return {}
            
            return self.spatial_analyzer.analyze_spatial_scales(data_points)
            
        except Exception as e:
            logger.error(f"Spatial scale analysis failed: {e}")
            return {}
    
    def _detect_rolling_anomalies(
        self,
        values: np.ndarray,
        scale: str,
        avg_dt: float
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using rolling statistics at specific temporal scale"""
        anomalies = []
        
        try:
            # Define window size based on scale
            scale_windows = {
                'seconds': max(5, int(10 / avg_dt)),
                'minutes': max(10, int(300 / avg_dt)),
                'hours': max(20, int(3600 / avg_dt)),
                'days': max(50, int(86400 / avg_dt))
            }
            
            window_size = scale_windows.get(scale, 10)
            window_size = min(window_size, len(values) // 2)
            
            if window_size < 3:
                return anomalies
            
            # Calculate rolling statistics
            rolling_mean = pd.Series(values).rolling(window=window_size, center=True).mean()
            rolling_std = pd.Series(values).rolling(window=window_size, center=True).std()
            
            # Detect anomalies
            for i in range(len(values)):
                if not (np.isnan(rolling_mean.iloc[i]) or np.isnan(rolling_std.iloc[i])):
                    if rolling_std.iloc[i] > 0:
                        z_score = (values[i] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                        
                        if abs(z_score) > self.anomaly_threshold:
                            anomalies.append({
                                'time_index': i,
                                'value': values[i],
                                'rolling_mean': rolling_mean.iloc[i],
                                'rolling_std': rolling_std.iloc[i],
                                'z_score': z_score,
                                'anomaly_strength': abs(z_score),
                                'scale': scale,
                                'window_size': window_size
                            })
            
        except Exception as e:
            logger.error(f"Rolling anomaly detection for scale {scale} failed: {e}")
        
        return anomalies
    
    async def _create_temporal_anomalies(
        self,
        temporal_anomalies: Dict[str, List[Dict[str, Any]]],
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Create DetectedAnomaly objects from temporal scale anomalies"""
        detected_anomalies = []
        
        try:
            for scale, anomalies in temporal_anomalies.items():
                for anomaly in anomalies:
                    time_index = anomaly.get('time_index', 0)
                    
                    if time_index < len(data_stream.data_points):
                        data_point = data_stream.data_points[time_index]
                        
                        anomaly_score = min(1.0, anomaly.get('anomaly_strength', 0) / 5.0)
                        
                        detected_anomaly = DetectedAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            detection_timestamp=datetime.utcnow(),
                            anomaly_type=AnomalyType.MULTI_SCALE,
                            severity=self._determine_severity(anomaly_score),
                            temporal_extent={
                                "start": data_point.timestamp,
                                "end": data_point.timestamp
                            },
                            spatial_extent=data_point.location,
                            affected_variables=[data_point.measurement_type],
                            anomaly_score=anomaly_score,
                            deviation_magnitude=anomaly.get('anomaly_strength', 0),
                            baseline_comparison={
                                "temporal_scale": scale,
                                "scale_value": anomaly.get('scale_value', 0),
                                "z_score": anomaly.get('z_score', 0),
                                "rolling_mean": anomaly.get('rolling_mean', 0),
                                "rolling_std": anomaly.get('rolling_std', 0)
                            },
                            geological_context=f"Multi-scale temporal anomaly at {scale} scale",
                            detection_method=f"multiscale_temporal_{scale}",
                            model_version="1.0.0",
                            processing_time=0.0
                        )
                        
                        detected_anomalies.append(detected_anomaly)
            
        except Exception as e:
            logger.error(f"Temporal anomaly creation failed: {e}")
        
        return detected_anomalies
    
    async def _create_spatial_anomalies(
        self,
        spatial_anomalies: Dict[str, List[Dict[str, Any]]],
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Create DetectedAnomaly objects from spatial scale anomalies"""
        detected_anomalies = []
        
        try:
            for scale, anomalies in spatial_anomalies.items():
                for anomaly in anomalies:
                    point_index = anomaly.get('point_index', 0)
                    
                    if point_index < len(data_stream.data_points):
                        data_point = data_stream.data_points[point_index]
                        
                        anomaly_score = min(1.0, anomaly.get('anomaly_strength', 0) / 5.0)
                        
                        detected_anomaly = DetectedAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            detection_timestamp=datetime.utcnow(),
                            anomaly_type=AnomalyType.MULTI_SCALE,
                            severity=self._determine_severity(anomaly_score),
                            temporal_extent={
                                "start": data_point.timestamp,
                                "end": data_point.timestamp
                            },
                            spatial_extent=data_point.location,
                            affected_variables=[data_point.measurement_type],
                            anomaly_score=anomaly_score,
                            deviation_magnitude=anomaly.get('anomaly_strength', 0),
                            baseline_comparison={
                                "spatial_scale": scale,
                                "scale_km": anomaly.get('scale_km', 0),
                                "z_score": anomaly.get('z_score', 0),
                                "local_mean": anomaly.get('local_mean', 0),
                                "local_std": anomaly.get('local_std', 0),
                                "neighbor_count": anomaly.get('neighbor_count', 0)
                            },
                            geological_context=f"Multi-scale spatial anomaly at {scale} scale ({anomaly.get('scale_km', 0):.1f} km)",
                            detection_method=f"multiscale_spatial_{scale}",
                            model_version="1.0.0",
                            processing_time=0.0
                        )
                        
                        detected_anomalies.append(detected_anomaly)
            
        except Exception as e:
            logger.error(f"Spatial anomaly creation failed: {e}")
        
        return detected_anomalies
    
    async def _create_interaction_anomalies(
        self,
        interactions: List[Dict[str, Any]],
        data_stream: GeologicalDataStream
    ) -> List[DetectedAnomaly]:
        """Create DetectedAnomaly objects from cross-scale interactions"""
        detected_anomalies = []
        
        try:
            for interaction in interactions:
                if interaction.get('strength', 0) > self.interaction_threshold:
                    # Create anomaly representing the interaction
                    anomaly_score = interaction.get('strength', 0)
                    
                    # Use first data point as representative (simplified)
                    if data_stream.data_points:
                        data_point = data_stream.data_points[0]
                        
                        detected_anomaly = DetectedAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            detection_timestamp=datetime.utcnow(),
                            anomaly_type=AnomalyType.MULTI_SCALE,
                            severity=self._determine_severity(anomaly_score),
                            temporal_extent={
                                "start": data_stream.data_points[0].timestamp,
                                "end": data_stream.data_points[-1].timestamp
                            },
                            spatial_extent=data_point.location,
                            affected_variables=list(set(
                                point.measurement_type for point in data_stream.data_points
                            )),
                            anomaly_score=anomaly_score,
                            deviation_magnitude=anomaly_score,
                            baseline_comparison=interaction,
                            geological_context=f"Cross-scale interaction: {interaction.get('interaction_type', 'unknown')}",
                            detection_method=f"multiscale_interaction_{interaction.get('interaction_type', 'unknown')}",
                            model_version="1.0.0",
                            processing_time=0.0
                        )
                        
                        detected_anomalies.append(detected_anomaly)
            
        except Exception as e:
            logger.error(f"Interaction anomaly creation failed: {e}")
        
        return detected_anomalies
    
    def _determine_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Determine anomaly severity based on score"""
        if anomaly_score > 0.9:
            return AnomalySeverity.CRITICAL
        elif anomaly_score > 0.8:
            return AnomalySeverity.HIGH
        elif anomaly_score > 0.7:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about the multi-scale analyzer"""
        return {
            "temporal_scales": self.temporal_scales,
            "spatial_scales": self.spatial_scales,
            "wavelet_scales": len(self.wavelet_transform.scales),
            "spatial_scale_definitions": self.spatial_analyzer.spatial_scales,
            "interaction_types": self.interaction_detector.interaction_types,
            "anomaly_threshold": self.anomaly_threshold,
            "interaction_threshold": self.interaction_threshold,
            "min_samples_per_scale": self.min_samples_per_scale
        }