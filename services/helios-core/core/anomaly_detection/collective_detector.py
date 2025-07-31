"""
Collective Anomaly Detection for Geological Event Sequences

Implements collective anomaly detection to identify unusual patterns in
sequences of geological events that may indicate emerging phenomena.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Sequence
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class GeologicalEventSequence:
    """Sequence of geological events"""
    events: List[Dict[str, Any]]
    start_time: float
    end_time: float
    location: Tuple[float, float, float]
    sequence_type: str
    metadata: Dict[str, Any]

@dataclass
class CollectiveAnomalyResult:
    """Result of collective anomaly detection"""
    sequence_id: str
    start_time: float
    end_time: float
    anomaly_score: float
    is_collective_anomaly: bool
    sequence_pattern: str
    deviation_type: str
    affected_event_types: List[str]
    temporal_characteristics: Dict[str, float]
    spatial_characteristics: Dict[str, float]
    geological_interpretation: str
    confidence: float
    significance_level: str
    explanation: str

class SequenceEncoder(nn.Module):
    """LSTM-based encoder for geological event sequences"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.1
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, sequences, sequence_lengths):
        # Pack sequences for variable length handling
        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences, sequence_lengths, batch_first=True, enforce_sorted=False
        )
        
        # LSTM encoding
        packed_output, (hidden, cell) = self.lstm(packed_sequences)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Self-attention
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Global representation (mean pooling over time)
        sequence_representations = []
        for i, length in enumerate(sequence_lengths):
            seq_repr = torch.mean(attended_output[i, :length, :], dim=0)
            sequence_representations.append(seq_repr)
        
        sequence_representations = torch.stack(sequence_representations)
        
        return self.output_layer(sequence_representations), attention_weights

class TemporalPatternAnalyzer:
    """Analyzer for temporal patterns in geological sequences"""
    
    def __init__(self):
        self.pattern_templates = {
            'precursory_sequence': {
                'description': 'Increasing activity leading to main event',
                'characteristics': {
                    'trend': 'increasing',
                    'duration_range': (3600, 86400 * 30),  # 1 hour to 30 days
                    'event_rate_increase': 2.0
                }
            },
            'swarm_sequence': {
                'description': 'Clustered events in time and space',
                'characteristics': {
                    'trend': 'clustered',
                    'duration_range': (3600, 86400 * 7),  # 1 hour to 7 days
                    'spatial_clustering': True
                }
            },
            'aftershock_sequence': {
                'description': 'Decreasing activity after main event',
                'characteristics': {
                    'trend': 'decreasing',
                    'duration_range': (3600, 86400 * 365),  # 1 hour to 1 year
                    'follows_main_event': True
                }
            },
            'periodic_sequence': {
                'description': 'Regular periodic activity',
                'characteristics': {
                    'trend': 'periodic',
                    'period_range': (60, 86400),  # 1 minute to 1 day
                    'regularity_threshold': 0.8
                }
            },
            'escalating_unrest': {
                'description': 'Gradually escalating volcanic unrest',
                'characteristics': {
                    'trend': 'escalating',
                    'duration_range': (86400, 86400 * 180),  # 1 day to 6 months
                    'multi_parameter': True
                }
            }
        }
    
    def analyze_temporal_pattern(self, sequence: GeologicalEventSequence) -> Dict[str, Any]:
        """Analyze temporal patterns in event sequence"""
        events = sequence.events
        if len(events) < 2:
            
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
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
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
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


        
        # Extract timestamps and magnitudes/intensities
        timestamps = np.array([event.get('timestamp', 0) for event in events])
        magnitudes = np.array([event.get('magnitude', event.get('intensity', 1)) for event in events])
        
        # Sort by time
        sort_indices = np.argsort(timestamps)
        timestamps = timestamps[sort_indices]
        magnitudes = magnitudes[sort_indices]
        
        # Compute temporal characteristics
        duration = timestamps[-1] - timestamps[0]
        inter_event_times = np.diff(timestamps)
        mean_inter_event_time = np.mean(inter_event_times)
        
        # Analyze trends
        trend_analysis = self._analyze_trend(timestamps, magnitudes)
        clustering_analysis = self._analyze_clustering(timestamps, magnitudes)
        periodicity_analysis = self._analyze_periodicity(timestamps, magnitudes)
        
        # Match against known patterns
        pattern_match = self._match_pattern(
            duration, trend_analysis, clustering_analysis, periodicity_analysis
        )
        
        return {
            'pattern': pattern_match,
            'duration': duration,
            'event_count': len(events),
            'mean_inter_event_time': mean_inter_event_time,
            'trend': trend_analysis,
            'clustering': clustering_analysis,
            'periodicity': periodicity_analysis,
            'characteristics': {
                'temporal_density': len(events) / (duration + 1),
                'magnitude_trend': np.corrcoef(timestamps, magnitudes)[0, 1] if len(timestamps) > 1 else 0,
                'variability': np.std(inter_event_times) / (mean_inter_event_time + 1e-8)
            }
        }
    
    def _analyze_trend(self, timestamps: np.ndarray, magnitudes: np.ndarray) -> Dict[str, float]:
        """Analyze temporal trends in the sequence"""
        if len(timestamps) < 3:
            
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
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
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
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


        
        # Linear trend in event rate
        time_windows = np.linspace(timestamps[0], timestamps[-1], 10)
        event_rates = []
        
        for i in range(len(time_windows) - 1):
            window_start, window_end = time_windows[i], time_windows[i + 1]
            events_in_window = np.sum((timestamps >= window_start) & (timestamps < window_end))
            window_duration = window_end - window_start
            rate = events_in_window / (window_duration + 1e-8)
            event_rates.append(rate)
        
        # Trend analysis
        x = np.arange(len(event_rates))
        trend_coeff = np.corrcoef(x, event_rates)[0, 1] if len(event_rates) > 1 else 0
        
        if trend_coeff > 0.5:
            trend_type = 'increasing'
        elif trend_coeff < -0.5:
            trend_type = 'decreasing'
        else:
            trend_type = 'stable'
        
        return {
            'type': trend_type,
            'strength': abs(trend_coeff),
            'rate_change': (event_rates[-1] - event_rates[0]) / (event_rates[0] + 1e-8)
        }
    
    def _analyze_clustering(self, timestamps: np.ndarray, magnitudes: np.ndarray) -> Dict[str, Any]:
        """Analyze temporal clustering in the sequence"""
        if len(timestamps) < 3:
            return {'is_clustered': False, 'cluster_count': 0}
        
        # Use DBSCAN for temporal clustering
        timestamps_scaled = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(timestamps_scaled.reshape(-1, 1))
        
        cluster_labels = clustering.labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        return {
            'is_clustered': n_clusters > 1,
            'cluster_count': n_clusters,
            'noise_points': np.sum(cluster_labels == -1),
            'clustering_coefficient': n_clusters / len(timestamps) if len(timestamps) > 0 else 0
        }
    
    def _analyze_periodicity(self, timestamps: np.ndarray, magnitudes: np.ndarray) -> Dict[str, Any]:
        """Analyze periodicity in the sequence"""
        if len(timestamps) < 10:
            return {'is_periodic': False, 'period': 0, 'regularity': 0}
        
        # Compute inter-event times
        inter_event_times = np.diff(timestamps)
        
        # Look for dominant periods using autocorrelation
        if len(inter_event_times) > 5:
            autocorr = np.correlate(inter_event_times, inter_event_times, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            if len(autocorr) > 3:
                peak_indices = []
                for i in range(1, len(autocorr) - 1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peak_indices.append(i)
                
                if peak_indices:
                    # Most prominent period
                    max_peak_idx = peak_indices[np.argmax([autocorr[i] for i in peak_indices])]
                    period = inter_event_times[max_peak_idx] if max_peak_idx < len(inter_event_times) else 0
                    regularity = autocorr[max_peak_idx] / autocorr[0] if autocorr[0] > 0 else 0
                    
                    return {
                        'is_periodic': regularity > 0.3,
                        'period': period,
                        'regularity': regularity
                    }
        
        return {'is_periodic': False, 'period': 0, 'regularity': 0}
    
    def _match_pattern(self, duration: float, trend: Dict, clustering: Dict, 
                      periodicity: Dict) -> str:
        """Match sequence characteristics against known patterns"""
        # Check each pattern template
        for pattern_name, template in self.pattern_templates.items():
            characteristics = template['characteristics']
            
            # Duration check
            if 'duration_range' in characteristics:
                min_dur, max_dur = characteristics['duration_range']
                if not (min_dur <= duration <= max_dur):
                    continue
            
            # Trend check
            if 'trend' in characteristics:
                expected_trend = characteristics['trend']
                if trend['type'] != expected_trend:
                    continue
            
            # Clustering check
            if 'spatial_clustering' in characteristics:
                if not clustering['is_clustered']:
                    continue
            
            # Periodicity check
            if 'regularity_threshold' in characteristics:
                if periodicity['regularity'] < characteristics['regularity_threshold']:
                    continue
            
            return pattern_name
        
        return 'unknown_pattern'

class SpatialPatternAnalyzer:
    """Analyzer for spatial patterns in geological sequences"""
    
    def __init__(self):
        self.spatial_patterns = {
            'linear_migration': 'Events show linear spatial progression',
            'radial_expansion': 'Events expand radially from center',
            'clustered': 'Events are spatially clustered',
            'scattered': 'Events are randomly distributed',
            'fault_aligned': 'Events align with known fault structures'
        }
    
    def analyze_spatial_pattern(self, sequence: GeologicalEventSequence) -> Dict[str, Any]:
        """Analyze spatial patterns in event sequence"""
        events = sequence.events
        if len(events) < 3:
            
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
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
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
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


        
        # Extract locations
        locations = []
        for event in events:
            if 'location' in event:
                loc = event['location']
                if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    locations.append(loc[:3] if len(loc) >= 3 else loc + [0])
        
        if len(locations) < 3:
            
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
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
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
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
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
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


        
        locations = np.array(locations)
        
        # Compute spatial characteristics
        centroid = np.mean(locations, axis=0)
        distances_from_centroid = np.linalg.norm(locations - centroid, axis=1)
        
        # Analyze spatial distribution
        spatial_spread = np.std(distances_from_centroid)
        max_distance = np.max(distances_from_centroid)
        
        # Check for linear patterns
        linearity_score = self._compute_linearity(locations)
        
        # Check for radial patterns
        radiality_score = self._compute_radiality(locations, centroid)
        
        # Clustering analysis
        clustering_score = self._compute_clustering(locations)
        
        # Determine dominant pattern
        pattern = self._classify_spatial_pattern(
            linearity_score, radiality_score, clustering_score, spatial_spread
        )
        
        return {
            'pattern': pattern,
            'centroid': centroid.tolist(),
            'spatial_spread': spatial_spread,
            'max_distance': max_distance,
            'linearity_score': linearity_score,
            'radiality_score': radiality_score,
            'clustering_score': clustering_score,
            'characteristics': {
                'compactness': 1.0 / (1.0 + spatial_spread),
                'elongation': self._compute_elongation(locations),
                'directional_bias': self._compute_directional_bias(locations)
            }
        }
    
    def _compute_linearity(self, locations: np.ndarray) -> float:
        """Compute how linear the spatial distribution is"""
        if len(locations) < 3:
            return 0.0
        
        # Fit line using PCA
        centered_locations = locations - np.mean(locations, axis=0)
        cov_matrix = np.cov(centered_locations.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Linearity is ratio of largest to second largest eigenvalue
        eigenvalues = np.sort(eigenvalues)[::-1]
        if len(eigenvalues) >= 2 and eigenvalues[1] > 1e-8:
            linearity = eigenvalues[0] / eigenvalues[1]
            return min(1.0, linearity / 10.0)  # Normalize
        
        return 0.0
    
    def _compute_radiality(self, locations: np.ndarray, centroid: np.ndarray) -> float:
        """Compute how radial the spatial distribution is"""
        if len(locations) < 3:
            return 0.0
        
        distances = np.linalg.norm(locations - centroid, axis=1)
        timestamps = np.arange(len(locations))  # Assume chronological order
        
        # Check if distances increase with time (radial expansion)
        correlation = np.corrcoef(timestamps, distances)[0, 1] if len(timestamps) > 1 else 0
        
        return max(0.0, correlation)
    
    def _compute_clustering(self, locations: np.ndarray) -> float:
        """Compute spatial clustering score"""
        if len(locations) < 3:
            return 0.0
        
        # Use DBSCAN clustering
        clustering = DBSCAN(eps=0.1, min_samples=2).fit(locations)
        labels = clustering.labels_
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        # Clustering score based on cluster formation
        if len(locations) > 0:
            clustering_score = (len(locations) - n_noise) / len(locations)
            return clustering_score
        
        return 0.0
    
    def _compute_elongation(self, locations: np.ndarray) -> float:
        """Compute elongation of spatial distribution"""
        if len(locations) < 3:
            return 0.0
        
        centered_locations = locations - np.mean(locations, axis=0)
        cov_matrix = np.cov(centered_locations.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        if len(eigenvalues) >= 2 and eigenvalues[1] > 1e-8:
            return eigenvalues[0] / eigenvalues[1]
        
        return 1.0
    
    def _compute_directional_bias(self, locations: np.ndarray) -> float:
        """Compute directional bias in spatial distribution"""
        if len(locations) < 3:
            return 0.0
        
        # Compute direction vectors between consecutive events
        directions = []
        for i in range(len(locations) - 1):
            direction = locations[i + 1] - locations[i]
            if np.linalg.norm(direction) > 1e-8:
                directions.append(direction / np.linalg.norm(direction))
        
        if not directions:
            return 0.0
        
        directions = np.array(directions)
        
        # Compute consistency of directions
        mean_direction = np.mean(directions, axis=0)
        mean_direction = mean_direction / (np.linalg.norm(mean_direction) + 1e-8)
        
        # Compute alignment with mean direction
        alignments = [np.dot(direction, mean_direction) for direction in directions]
        directional_bias = np.mean(alignments)
        
        return max(0.0, directional_bias)
    
    def _classify_spatial_pattern(self, linearity: float, radiality: float, 
                                clustering: float, spread: float) -> str:
        """Classify the dominant spatial pattern"""
        scores = {
            'linear_migration': linearity,
            'radial_expansion': radiality,
            'clustered': clustering * (1.0 / (1.0 + spread)),
            'scattered': 1.0 - clustering
        }
        
        return max(scores.items(), key=lambda x: x[1])[0]

class CollectiveAnomalyDetector:
    """
    Collective anomaly detection for geological event sequences
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize analyzers
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.spatial_analyzer = SpatialPatternAnalyzer()
        
        # Neural network components
        self.sequence_encoder = None
        self.scaler = StandardScaler()
        
        # Pattern database
        self.normal_patterns = []
        self.anomalous_patterns = []
        
        self.is_fitted = False
        
        logger.info("Initialized CollectiveAnomalyDetector")
    
    def fit(self, sequences: List[GeologicalEventSequence], labels: Optional[List[int]] = None) -> None:
        """
        Fit the collective anomaly detector
        
        Args:
            sequences: Training sequences
            labels: Optional labels (1 for anomalous, 0 for normal)
        """
        logger.info(f"Fitting collective detector on {len(sequences)} sequences")
        
        # Extract features from sequences
        sequence_features = []
        for seq in sequences:
            features = self._extract_sequence_features(seq)
            sequence_features.append(features)
        
        sequence_features = np.array(sequence_features)
        
        # Normalize features
        sequence_features_normalized = self.scaler.fit_transform(sequence_features)
        
        # Initialize sequence encoder
        feature_dim = sequence_features.shape[1]
        hidden_dim = self.config.get('hidden_dim', 64)
        
        self.sequence_encoder = SequenceEncoder(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=self.config.get('num_layers', 2)
        ).to(self.device)
        
        # Prepare training data
        if labels is not None:
            # Supervised learning
            self._train_supervised(sequences, sequence_features_normalized, labels)
        else:
            # Unsupervised learning
            self._train_unsupervised(sequences, sequence_features_normalized)
        
        self.is_fitted = True
        logger.info("Successfully fitted collective anomaly detector")
    
    def _extract_sequence_features(self, sequence: GeologicalEventSequence) -> np.ndarray:
        """Extract numerical features from geological event sequence"""
        # Temporal pattern analysis
        temporal_analysis = self.temporal_analyzer.analyze_temporal_pattern(sequence)
        
        # Spatial pattern analysis
        spatial_analysis = self.spatial_analyzer.analyze_spatial_pattern(sequence)
        
        # Basic sequence statistics
        events = sequence.events
        event_count = len(events)
        duration = sequence.end_time - sequence.start_time
        
        # Event type distribution
        event_types = [event.get('type', 'unknown') for event in events]
        unique_types = list(set(event_types))
        type_counts = [event_types.count(t) for t in unique_types]
        type_diversity = len(unique_types) / max(1, event_count)
        
        # Magnitude/intensity statistics
        magnitudes = [event.get('magnitude', event.get('intensity', 0)) for event in events]
        mag_mean = np.mean(magnitudes) if magnitudes else 0
        mag_std = np.std(magnitudes) if magnitudes else 0
        mag_max = np.max(magnitudes) if magnitudes else 0
        
        # Compile features
        features = [
            # Basic statistics
            event_count,
            duration,
            event_count / (duration + 1) if duration > 0 else 0,  # event rate
            type_diversity,
            mag_mean,
            mag_std,
            mag_max,
            
            # Temporal characteristics
            temporal_analysis.get('duration', 0),
            temporal_analysis.get('mean_inter_event_time', 0),
            temporal_analysis['characteristics'].get('temporal_density', 0),
            temporal_analysis['characteristics'].get('magnitude_trend', 0),
            temporal_analysis['characteristics'].get('variability', 0),
            temporal_analysis['trend'].get('strength', 0),
            temporal_analysis['clustering'].get('cluster_count', 0),
            temporal_analysis['periodicity'].get('regularity', 0),
            
            # Spatial characteristics
            spatial_analysis.get('spatial_spread', 0),
            spatial_analysis.get('max_distance', 0),
            spatial_analysis.get('linearity_score', 0),
            spatial_analysis.get('radiality_score', 0),
            spatial_analysis.get('clustering_score', 0),
            spatial_analysis['characteristics'].get('compactness', 0),
            spatial_analysis['characteristics'].get('elongation', 0),
            spatial_analysis['characteristics'].get('directional_bias', 0),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _train_supervised(self, sequences: List[GeologicalEventSequence], 
                         features: np.ndarray, labels: List[int]) -> None:
        """Train with supervised learning using labeled sequences"""
        # Separate normal and anomalous patterns
        for i, label in enumerate(labels):
            if label == 0:  # Normal
                self.normal_patterns.append((sequences[i], features[i]))
            else:  # Anomalous
                self.anomalous_patterns.append((sequences[i], features[i]))
        
        logger.info(f"Training with {len(self.normal_patterns)} normal and "
                   f"{len(self.anomalous_patterns)} anomalous patterns")
    
    def _train_unsupervised(self, sequences: List[GeologicalEventSequence], 
                           features: np.ndarray) -> None:
        """Train with unsupervised learning"""
        # Use all sequences as normal patterns initially
        for i, seq in enumerate(sequences):
            self.normal_patterns.append((seq, features[i]))
        
        logger.info(f"Training unsupervised with {len(self.normal_patterns)} sequences")
    
    def detect_collective_anomalies(self, sequences: List[GeologicalEventSequence]) -> List[CollectiveAnomalyResult]:
        """
        Detect collective anomalies in new sequences
        
        Args:
            sequences: Sequences to analyze
            
        Returns:
            List of collective anomaly results
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detecting anomalies")
        
        logger.info(f"Detecting collective anomalies in {len(sequences)} sequences")
        
        results = []
        
        for i, sequence in enumerate(sequences):
            # Extract features
            features = self._extract_sequence_features(sequence)
            features_normalized = self.scaler.transform(features.reshape(1, -1))[0]
            
            # Compute anomaly score
            anomaly_score = self._compute_collective_anomaly_score(sequence, features_normalized)
            
            # Determine if anomalous
            threshold = self.config.get('collective_threshold', 0.7)
            is_anomaly = anomaly_score > threshold
            
            # Analyze patterns
            temporal_analysis = self.temporal_analyzer.analyze_temporal_pattern(sequence)
            spatial_analysis = self.spatial_analyzer.analyze_spatial_pattern(sequence)
            
            # Classify deviation type
            deviation_type = self._classify_deviation_type(
                sequence, temporal_analysis, spatial_analysis, anomaly_score
            )
            
            # Identify affected event types
            affected_types = self._identify_affected_event_types(sequence)
            
            # Generate geological interpretation
            geological_interpretation = self._generate_geological_interpretation(
                sequence, temporal_analysis, spatial_analysis, deviation_type
            )
            
            # Assess significance
            significance_level = self._assess_significance_level(
                sequence, anomaly_score, deviation_type
            )
            
            # Generate explanation
            explanation = self._generate_collective_explanation(
                sequence, temporal_analysis, spatial_analysis, deviation_type, significance_level
            )
            
            result = CollectiveAnomalyResult(
                sequence_id=f"seq_{i}",
                start_time=sequence.start_time,
                end_time=sequence.end_time,
                anomaly_score=anomaly_score,
                is_collective_anomaly=is_anomaly,
                sequence_pattern=temporal_analysis.get('pattern', 'unknown'),
                deviation_type=deviation_type,
                affected_event_types=affected_types,
                temporal_characteristics=temporal_analysis.get('characteristics', {}),
                spatial_characteristics=spatial_analysis.get('characteristics', {}),
                geological_interpretation=geological_interpretation,
                confidence=self._compute_collective_confidence(anomaly_score, temporal_analysis, spatial_analysis),
                significance_level=significance_level,
                explanation=explanation
            )
            
            results.append(result)
        
        logger.info(f"Detected {sum(r.is_collective_anomaly for r in results)} collective anomalies")
        return results
    
    def _compute_collective_anomaly_score(self, sequence: GeologicalEventSequence, 
                                        features: np.ndarray) -> float:
        """Compute collective anomaly score for a sequence"""
        if not self.normal_patterns:
            return 0.5  # Default score if no training data
        
        # Compare with normal patterns
        normal_features = np.array([pattern[1] for pattern in self.normal_patterns])
        
        # Compute distances to normal patterns
        distances = []
        for normal_feature in normal_features:
            distance = np.linalg.norm(features - normal_feature)
            distances.append(distance)
        
        # Anomaly score based on distance to nearest normal pattern
        min_distance = np.min(distances)
        mean_distance = np.mean(distances)
        
        # Normalize score
        anomaly_score = min_distance / (mean_distance + 1e-8)
        anomaly_score = min(1.0, anomaly_score)
        
        return anomaly_score
    
    def _classify_deviation_type(self, sequence: GeologicalEventSequence,
                               temporal_analysis: Dict, spatial_analysis: Dict,
                               anomaly_score: float) -> str:
        """Classify the type of deviation from normal patterns"""
        # Check temporal deviations
        temporal_pattern = temporal_analysis.get('pattern', 'unknown')
        spatial_pattern = spatial_analysis.get('pattern', 'unknown')
        
        if temporal_pattern == 'unknown_pattern':
            if spatial_pattern == 'unknown_pattern':
                return 'novel_spatiotemporal_pattern'
            else:
                return 'novel_temporal_pattern'
        elif spatial_pattern == 'unknown_pattern':
            return 'novel_spatial_pattern'
        
        # Check for extreme characteristics
        temporal_chars = temporal_analysis.get('characteristics', {})
        spatial_chars = spatial_analysis.get('characteristics', {})
        
        if temporal_chars.get('temporal_density', 0) > 10:  # Very high event rate
            return 'extreme_temporal_density'
        elif spatial_chars.get('compactness', 1) < 0.1:  # Very scattered
            return 'extreme_spatial_dispersion'
        elif temporal_chars.get('variability', 0) > 5:  # Very irregular
            return 'extreme_temporal_irregularity'
        
        return 'general_deviation'
    
    def _identify_affected_event_types(self, sequence: GeologicalEventSequence) -> List[str]:
        """Identify which event types are involved in the anomalous sequence"""
        event_types = [event.get('type', 'unknown') for event in sequence.events]
        unique_types = list(set(event_types))
        
        # Filter out common types and focus on significant ones
        significant_types = []
        for event_type in unique_types:
            count = event_types.count(event_type)
            if count >= 2 or event_type in ['earthquake', 'eruption', 'landslide']:
                significant_types.append(event_type)
        
        return significant_types
    
    def _generate_geological_interpretation(self, sequence: GeologicalEventSequence,
                                          temporal_analysis: Dict, spatial_analysis: Dict,
                                          deviation_type: str) -> str:
        """Generate geological interpretation of the anomalous sequence"""
        interpretations = []
        
        # Temporal interpretation
        temporal_pattern = temporal_analysis.get('pattern', 'unknown')
        if temporal_pattern == 'precursory_sequence':
            interpretations.append("Possible precursory activity to major geological event")
        elif temporal_pattern == 'escalating_unrest':
            interpretations.append("Escalating volcanic or tectonic unrest")
        elif temporal_pattern == 'swarm_sequence':
            interpretations.append("Earthquake or volcanic swarm activity")
        
        # Spatial interpretation
        spatial_pattern = spatial_analysis.get('pattern', 'unknown')
        if spatial_pattern == 'linear_migration':
            interpretations.append("Activity migrating along fault or magma conduit")
        elif spatial_pattern == 'radial_expansion':
            interpretations.append("Expanding zone of influence from central source")
        
        # Deviation type interpretation
        if deviation_type == 'novel_spatiotemporal_pattern':
            interpretations.append("Previously unobserved geological process")
        elif deviation_type == 'extreme_temporal_density':
            interpretations.append("Unusually intense geological activity")
        
        return "; ".join(interpretations) if interpretations else "Anomalous geological sequence"
    
    def _assess_significance_level(self, sequence: GeologicalEventSequence,
                                 anomaly_score: float, deviation_type: str) -> str:
        """Assess the significance level of the collective anomaly"""
        # High significance criteria
        if (anomaly_score > 0.9 or 
            deviation_type in ['novel_spatiotemporal_pattern', 'extreme_temporal_density'] or
            len(sequence.events) > 100):
            return 'high'
        
        # Medium significance criteria
        elif (anomaly_score > 0.7 or
              deviation_type in ['novel_temporal_pattern', 'novel_spatial_pattern'] or
              len(sequence.events) > 20):
            return 'medium'
        
        return 'low'
    
    def _generate_collective_explanation(self, sequence: GeologicalEventSequence,
                                       temporal_analysis: Dict, spatial_analysis: Dict,
                                       deviation_type: str, significance_level: str) -> str:
        """Generate explanation for collective anomaly"""
        explanations = []
        
        # Basic sequence description
        event_count = len(sequence.events)
        duration_hours = (sequence.end_time - sequence.start_time) / 3600
        
        explanations.append(f"Sequence of {event_count} events over {duration_hours:.1f} hours")
        
        # Pattern description
        temporal_pattern = temporal_analysis.get('pattern', 'unknown')
        spatial_pattern = spatial_analysis.get('pattern', 'unknown')
        
        if temporal_pattern != 'unknown':
            explanations.append(f"showing {temporal_pattern.replace('_', ' ')} temporal pattern")
        
        if spatial_pattern != 'unknown':
            explanations.append(f"with {spatial_pattern.replace('_', ' ')} spatial distribution")
        
        # Deviation explanation
        if deviation_type == 'novel_spatiotemporal_pattern':
            explanations.append("This pattern has not been observed before in the training data")
        elif deviation_type == 'extreme_temporal_density':
            explanations.append("Event rate is extremely high compared to normal patterns")
        
        # Significance
        if significance_level == 'high':
            explanations.append("This anomaly requires immediate attention")
        elif significance_level == 'medium':
            explanations.append("This anomaly warrants further investigation")
        
        return ". ".join(explanations)
    
    def _compute_collective_confidence(self, anomaly_score: float, 
                                     temporal_analysis: Dict, spatial_analysis: Dict) -> float:
        """Compute confidence in collective anomaly detection"""
        # Base confidence on anomaly score
        base_confidence = min(1.0, anomaly_score * 1.2)
        
        # Adjust based on pattern recognition confidence
        temporal_pattern = temporal_analysis.get('pattern', 'unknown')
        spatial_pattern = spatial_analysis.get('pattern', 'unknown')
        
        pattern_confidence = 1.0
        if temporal_pattern == 'unknown':
            pattern_confidence *= 0.8
        if spatial_pattern == 'unknown':
            pattern_confidence *= 0.8
        
        # Adjust based on data quality
        event_count = len(temporal_analysis.get('characteristics', {}).get('event_count', 0))
        data_quality_factor = min(1.0, event_count / 10.0)  # More events = higher confidence
        
        final_confidence = base_confidence * pattern_confidence * data_quality_factor
        return min(1.0, final_confidence)
    
    def update_pattern_database(self, new_sequences: List[GeologicalEventSequence],
                              labels: Optional[List[int]] = None) -> None:
        """Update the pattern database with new sequences"""
        logger.info(f"Updating pattern database with {len(new_sequences)} new sequences")
        
        for i, sequence in enumerate(new_sequences):
            features = self._extract_sequence_features(sequence)
            features_normalized = self.scaler.transform(features.reshape(1, -1))[0]
            
            if labels is not None and i < len(labels):
                if labels[i] == 0:  # Normal
                    self.normal_patterns.append((sequence, features_normalized))
                else:  # Anomalous
                    self.anomalous_patterns.append((sequence, features_normalized))
            else:
                # Default to normal if no label provided
                self.normal_patterns.append((sequence, features_normalized))
        
        logger.info(f"Pattern database now contains {len(self.normal_patterns)} normal and "
                   f"{len(self.anomalous_patterns)} anomalous patterns")