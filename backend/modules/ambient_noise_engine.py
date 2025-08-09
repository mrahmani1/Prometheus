"""
Advanced Ambient Noise Analysis Engine
Continuous monitoring, velocity change detection, Green's function extraction,
and structural health monitoring using ambient seismic noise
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from scipy import signal
from scipy.optimize import minimize_scalar

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.signal import hilbert, spectrogram, welch, correlate
    from scipy.fft import fft, ifft, fftfreq
    from scipy.interpolate import interp1d
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class NoiseConfig:
    """Configuration for ambient noise analysis"""
    sample_rate: float = 50.0
    window_length: float = 3600.0  # 1 hour windows
    overlap_ratio: float = 0.5
    frequency_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.1, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
    )
    velocity_bands: List[Tuple[float, float]] = field(
        default_factory=lambda: [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
    )
    correlation_length: float = 400.0  # seconds for cross-correlation
    min_coherence: float = 0.6
    velocity_change_threshold: float = 0.01  # 1% change threshold
    reference_window_days: int = 30
    monitoring_mode: str = "continuous"  # continuous, event-based, scheduled


@dataclass
class StationPair:
    """Station pair for cross-correlation"""
    station1: str
    station2: str
    distance: float  # km
    azimuth: float  # degrees
    coordinates1: Tuple[float, float]
    coordinates2: Tuple[float, float]


class VelocityChangeDetector(nn.Module):
    """Neural network for velocity change detection"""
    
    def __init__(self, n_frequency_bands: int, sequence_length: int = 24):
        super().__init__()
        self.n_frequency_bands = n_frequency_bands
        self.sequence_length = sequence_length
        
        # Feature extraction from cross-correlation functions
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_frequency_bands, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(100)
        )
        
        # Temporal modeling with LSTM
        self.temporal_model = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Velocity change prediction heads
        self.velocity_change_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_frequency_bands),
            nn.Tanh()  # Velocity changes typically small
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_frequency_bands),
            nn.Sigmoid()
        )
        
        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, correlation_sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for velocity change detection
        
        Args:
            correlation_sequences: (batch, sequence_length, frequency_bands, correlation_samples)
        """
        batch_size, seq_len, n_bands, n_samples = correlation_sequences.shape
        
        # Reshape for convolution
        x = correlation_sequences.view(batch_size * seq_len, n_bands, n_samples)
        
        # Feature extraction
        features = self.feature_extractor(x)  # (batch*seq, 128, 100)
        features = features.mean(dim=-1)  # (batch*seq, 128)
        
        # Reshape for temporal modeling
        features = features.view(batch_size, seq_len, 128)
        
        # Temporal modeling
        lstm_out, _ = self.temporal_model(features)  # (batch, seq, 256)
        
        # Use final hidden state for predictions
        final_hidden = lstm_out[:, -1, :]  # (batch, 256)
        
        # Predictions
        velocity_changes = self.velocity_change_head(final_hidden)
        confidences = self.confidence_head(final_hidden)
        anomaly_score = self.anomaly_head(final_hidden).squeeze(-1)
        
        return {
            'velocity_changes': velocity_changes,
            'confidences': confidences,
            'anomaly_score': anomaly_score
        }


class GreensFunction:
    """Green's function estimation and analysis"""
    
    def __init__(self, station_pair: StationPair, config: NoiseConfig):
        self.station_pair = station_pair
        self.config = config
        self.reference_functions = {}
        self.current_functions = {}
        
    def extract_greens_function(self, data1: np.ndarray, data2: np.ndarray,
                              frequency_band: Optional[Tuple[float, float]] = None) -> Dict:
        """Extract empirical Green's function from ambient noise cross-correlation"""
        
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available - using simplified correlation")
            correlation = np.correlate(data1, data2, mode='full')
            lag_times = np.arange(-len(data2)+1, len(data1)) / self.config.sample_rate
            return {'correlation': correlation, 'lag_times': lag_times}
        
        # Preprocessing
        if frequency_band:
            data1_filtered = self._bandpass_filter(data1, frequency_band)
            data2_filtered = self._bandpass_filter(data2, frequency_band)
        else:
            data1_filtered, data2_filtered = data1, data2
        
        # Spectral whitening
        data1_whitened = self._spectral_whitening(data1_filtered)
        data2_whitened = self._spectral_whitening(data2_filtered)
        
        # Time-domain normalization (one-bit normalization)
        data1_normalized = np.sign(data1_whitened)
        data2_normalized = np.sign(data2_whitened)
        
        # Cross-correlation
        correlation = correlate(data1_normalized, data2_normalized, mode='full')
        
        # Lag times
        n_samples = len(correlation)
        lag_times = np.arange(-n_samples//2, n_samples//2 + 1) / self.config.sample_rate
        
        if len(lag_times) != len(correlation):
            lag_times = lag_times[:len(correlation)]
        
        # Symmetrization (for reciprocity)
        symmetric_correlation = (correlation + correlation[::-1]) / 2
        
        # Quality metrics
        snr = self._calculate_snr(symmetric_correlation, lag_times)
        coherence = self._calculate_coherence(data1_normalized, data2_normalized)
        
        return {
            'correlation': symmetric_correlation,
            'lag_times': lag_times,
            'snr': snr,
            'coherence': coherence,
            'frequency_band': frequency_band,
            'distance': self.station_pair.distance,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _bandpass_filter(self, data: np.ndarray, 
                        frequency_band: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter"""
        low, high = frequency_band
        nyquist = self.config.sample_rate / 2
        
        if high >= nyquist:
            high = nyquist * 0.99
        if low <= 0:
            low = 0.01
        
        sos = signal.butter(4, [low/nyquist, high/nyquist], 
                           btype='band', output='sos')
        return signal.sosfilt(sos, data)
    
    def _spectral_whitening(self, data: np.ndarray) -> np.ndarray:
        """Apply spectral whitening in frequency domain"""
        data_fft = fft(data)
        amplitude_spectrum = np.abs(data_fft)
        
        # Avoid division by zero
        amplitude_spectrum[amplitude_spectrum == 0] = 1e-10
        
        # Normalize by amplitude spectrum
        whitened_fft = data_fft / amplitude_spectrum
        
        return np.real(ifft(whitened_fft))
    
    def _calculate_snr(self, correlation: np.ndarray, 
                      lag_times: np.ndarray) -> float:
        """Calculate signal-to-noise ratio of correlation function"""
        
        # Define signal window (around zero lag)
        signal_window = np.abs(lag_times) <= 10.0  # ±10 seconds
        
        # Define noise windows (far from zero lag)
        noise_window1 = lag_times <= -50.0
        noise_window2 = lag_times >= 50.0
        noise_window = noise_window1 | noise_window2
        
        if not np.any(signal_window) or not np.any(noise_window):
            return 0.0
        
        signal_power = np.mean(correlation[signal_window]**2)
        noise_power = np.mean(correlation[noise_window]**2)
        
        if noise_power == 0:
            return float('inf')
        
        return 10 * np.log10(signal_power / noise_power)
    
    def _calculate_coherence(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate coherence between two signals"""
        try:
            if SCIPY_AVAILABLE:
                freqs, coherence = signal.coherence(data1, data2, 
                                                  fs=self.config.sample_rate)
                return np.mean(coherence)
            else:
                # Simple correlation coefficient
                return abs(np.corrcoef(data1, data2)[0, 1])
        except:
            return 0.0
    
    def measure_velocity_change(self, reference_correlation: np.ndarray,
                              current_correlation: np.ndarray,
                              lag_times: np.ndarray,
                              method: str = 'doublet') -> Dict:
        """Measure velocity changes between reference and current correlations"""
        
        if method == 'doublet':
            return self._doublet_method(reference_correlation, current_correlation, lag_times)
        elif method == 'stretching':
            return self._stretching_method(reference_correlation, current_correlation, lag_times)
        elif method == 'dtw':
            return self._dtw_method(reference_correlation, current_correlation)
        else:
            warnings.warn(f"Unknown method {method}, using doublet")
            return self._doublet_method(reference_correlation, current_correlation, lag_times)
    
    def _doublet_method(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                       lag_times: np.ndarray) -> Dict:
        """Doublet method for velocity change measurement"""
        
        # Find the time window with highest energy
        energy = ref_corr**2 + cur_corr**2
        
        # Select positive and negative lag portions separately
        zero_idx = len(lag_times) // 2
        
        # Positive lags (causal part)
        pos_energy = energy[zero_idx:]
        pos_lag_times = lag_times[zero_idx:]
        
        if len(pos_energy) == 0:
            return {'velocity_change': 0.0, 'error': 1.0, 'correlation_coefficient': 0.0}
        
        # Find peak in positive lags
        peak_idx = np.argmax(pos_energy) + zero_idx
        
        # Define window around peak
        window_size = int(5.0 * self.config.sample_rate)  # 5 second window
        start_idx = max(0, peak_idx - window_size // 2)
        end_idx = min(len(ref_corr), peak_idx + window_size // 2)
        
        ref_window = ref_corr[start_idx:end_idx]
        cur_window = cur_corr[start_idx:end_idx]
        window_lag_times = lag_times[start_idx:end_idx]
        
        if len(ref_window) == 0:
            return {'velocity_change': 0.0, 'error': 1.0, 'correlation_coefficient': 0.0}
        
        # Calculate time derivatives
        dt = 1.0 / self.config.sample_rate
        ref_derivative = np.gradient(ref_window, dt)
        
        # Calculate velocity change using doublet formula
        numerator = np.sum((cur_window - ref_window) * ref_derivative)
        denominator = np.sum(ref_derivative**2)
        
        if denominator == 0:
            velocity_change = 0.0
            error = 1.0
        else:
            velocity_change = -numerator / denominator
            
            # Estimate error
            residual = cur_window - ref_window - velocity_change * ref_derivative
            error = np.std(residual) / np.max(np.abs(ref_window))
        
        # Correlation coefficient
        correlation_coeff = np.corrcoef(ref_window, cur_window)[0, 1]
        if np.isnan(correlation_coeff):
            correlation_coeff = 0.0
        
        return {
            'velocity_change': velocity_change,
            'error': error,
            'correlation_coefficient': correlation_coeff,
            'method': 'doublet',
            'window_center_time': window_lag_times[len(window_lag_times)//2]
        }
    
    def _stretching_method(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                          lag_times: np.ndarray) -> Dict:
        """Stretching method for velocity change measurement"""
        
        def stretch_function(epsilon):
            """Stretch the reference correlation by factor (1 + epsilon)"""
            stretched_times = lag_times * (1 + epsilon)
            
            # Interpolate to original time grid
            if SCIPY_AVAILABLE:
                interp_func = interp1d(stretched_times, ref_corr, 
                                     kind='linear', bounds_error=False, fill_value=0)
                stretched_ref = interp_func(lag_times)
            else:
                # Simple linear interpolation fallback
                stretched_ref = np.interp(lag_times, stretched_times, ref_corr)
            
            # Calculate correlation coefficient
            correlation_coeff = np.corrcoef(stretched_ref, cur_corr)[0, 1]
            return -correlation_coeff if not np.isnan(correlation_coeff) else -0.0
        
        # Optimize stretching factor
        try:
            result = minimize_scalar(stretch_function, bounds=(-0.1, 0.1), method='bounded')
            optimal_epsilon = result.x
            max_correlation = -result.fun
        except:
            optimal_epsilon = 0.0
            max_correlation = 0.0
        
        # Convert epsilon to velocity change (dv/v = -epsilon for surface waves)
        velocity_change = -optimal_epsilon
        
        # Estimate error based on curvature
        try:
            eps_test = np.linspace(optimal_epsilon - 0.01, optimal_epsilon + 0.01, 21)
            correlations = [stretch_function(eps) for eps in eps_test]
            
            # Fit parabola to estimate uncertainty
            coeffs = np.polyfit(eps_test, correlations, 2)
            if coeffs[0] > 0:  # Concave up (good fit)
                error = np.sqrt(2 / (coeffs[0] * len(ref_corr)))
            else:
                error = 0.01
        except:
            error = 0.01
        
        return {
            'velocity_change': velocity_change,
            'error': error,
            'correlation_coefficient': max_correlation,
            'method': 'stretching',
            'optimal_epsilon': optimal_epsilon
        }
    
    def _dtw_method(self, ref_corr: np.ndarray, cur_corr: np.ndarray) -> Dict:
        """Dynamic Time Warping method (simplified implementation)"""
        
        # Simplified DTW - would normally use proper DTW algorithm
        # For now, use cross-correlation based approach
        
        n = len(ref_corr)
        m = len(cur_corr)
        
        if n != m:
            # Resample to same length
            if SCIPY_AVAILABLE:
                from scipy.interpolate import interp1d
                f = interp1d(np.arange(m), cur_corr, kind='linear')
                cur_corr = f(np.linspace(0, m-1, n))
            else:
                cur_corr = np.interp(np.linspace(0, m-1, n), np.arange(m), cur_corr)
        
        # Simple cross-correlation to find optimal shift
        correlation = correlate(ref_corr, cur_corr, mode='full')
        lags = np.arange(-len(cur_corr)+1, len(ref_corr))
        
        # Find optimal lag
        optimal_lag_idx = np.argmax(correlation)
        optimal_lag = lags[optimal_lag_idx]
        
        # Convert to velocity change (simplified)
        dt = 1.0 / self.config.sample_rate
        time_shift = optimal_lag * dt
        
        # Assume characteristic travel time (distance/velocity)
        if self.station_pair.distance > 0:
            char_travel_time = self.station_pair.distance / 3.0  # Assume 3 km/s
            velocity_change = time_shift / char_travel_time
        else:
            velocity_change = 0.0
        
        max_correlation = correlation[optimal_lag_idx] / (np.linalg.norm(ref_corr) * np.linalg.norm(cur_corr))
        
        return {
            'velocity_change': velocity_change,
            'error': 0.01,  # Fixed error for simplified method
            'correlation_coefficient': max_correlation,
            'method': 'dtw_simplified',
            'time_shift': time_shift
        }


class AmbientNoiseMonitor:
    """Main ambient noise monitoring system"""
    
    def __init__(self, station_pairs: List[StationPair], 
                 config: Optional[NoiseConfig] = None):
        self.station_pairs = station_pairs
        self.config = config or NoiseConfig()
        
        # Green's function processors
        self.greens_functions = {
            f"{pair.station1}-{pair.station2}": GreensFunction(pair, self.config)
            for pair in station_pairs
        }
        
        # Reference correlations for velocity change monitoring
        self.reference_correlations = {}
        self.reference_established = {}
        
        # ML models
        self.ml_models = {}
        if TORCH_AVAILABLE:
            self._initialize_ml_models()
        
        # Monitoring state
        self.monitoring_active = False
        self.current_correlations = {}
        self.velocity_changes = {}
        self.anomaly_scores = {}
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        n_frequency_bands = len(self.config.frequency_bands)
        
        self.ml_models = {
            'velocity_detector': VelocityChangeDetector(
                n_frequency_bands=n_frequency_bands,
                sequence_length=24  # 24-hour sequences
            )
        }
    
    def process_noise_data(self, data_dict: Dict[str, np.ndarray], 
                          timestamp: datetime) -> Dict:
        """Process ambient noise data from multiple stations"""
        
        results = {
            'timestamp': timestamp.isoformat(),
            'correlations': {},
            'velocity_changes': {},
            'anomalies': {},
            'quality_metrics': {}
        }
        
        # Process each station pair
        for pair in self.station_pairs:
            pair_key = f"{pair.station1}-{pair.station2}"
            
            if pair.station1 not in data_dict or pair.station2 not in data_dict:
                continue
            
            data1 = data_dict[pair.station1]
            data2 = data_dict[pair.station2]
            
            # Process each frequency band
            pair_results = {}
            for i, freq_band in enumerate(self.config.frequency_bands):
                
                # Extract Green's function
                gf_processor = self.greens_functions[pair_key]
                gf_result = gf_processor.extract_greens_function(
                    data1, data2, freq_band
                )
                
                # Quality control
                if (gf_result['snr'] < 3.0 or 
                    gf_result['coherence'] < self.config.min_coherence):
                    continue
                
                # Store correlation
                band_key = f"band_{i}"
                pair_results[band_key] = gf_result
                
                # Velocity change analysis if reference exists
                if pair_key in self.reference_correlations and band_key in self.reference_correlations[pair_key]:
                    ref_corr = self.reference_correlations[pair_key][band_key]['correlation']
                    cur_corr = gf_result['correlation']
                    lag_times = gf_result['lag_times']
                    
                    # Measure velocity change
                    velocity_result = gf_processor.measure_velocity_change(
                        ref_corr, cur_corr, lag_times, method='stretching'
                    )
                    
                    pair_results[band_key]['velocity_change'] = velocity_result
                    
                    # Check for significant changes
                    if abs(velocity_result['velocity_change']) > self.config.velocity_change_threshold:
                        if pair_key not in results['anomalies']:
                            results['anomalies'][pair_key] = {}
                        results['anomalies'][pair_key][band_key] = velocity_result
            
            if pair_results:
                results['correlations'][pair_key] = pair_results
        
        # Update current correlations
        self.current_correlations[timestamp] = results['correlations']
        
        # ML-based anomaly detection
        if TORCH_AVAILABLE and 'velocity_detector' in self.ml_models:
            ml_anomalies = self._detect_ml_anomalies(results['correlations'])
            results['ml_anomalies'] = ml_anomalies
        
        return results
    
    def establish_reference(self, data_history: Dict[datetime, Dict[str, np.ndarray]], 
                          window_days: Optional[int] = None) -> Dict:
        """Establish reference Green's functions from historical data"""
        
        if window_days is None:
            window_days = self.config.reference_window_days
        
        # Get recent data
        cutoff_date = max(data_history.keys()) - timedelta(days=window_days)
        recent_data = {
            timestamp: data for timestamp, data in data_history.items()
            if timestamp >= cutoff_date
        }
        
        if len(recent_data) < 7:  # Need at least a week of data
            warnings.warn("Insufficient data for reference establishment")
            return {}
        
        reference_results = {}
        
        # Process each station pair
        for pair in self.station_pairs:
            pair_key = f"{pair.station1}-{pair.station2}"
            reference_results[pair_key] = {}
            
            # Collect correlations for each frequency band
            for i, freq_band in enumerate(self.config.frequency_bands):
                band_key = f"band_{i}"
                correlations = []
                snrs = []
                
                # Process all available data
                for timestamp, data_dict in recent_data.items():
                    if pair.station1 not in data_dict or pair.station2 not in data_dict:
                        continue
                    
                    gf_processor = self.greens_functions[pair_key]
                    gf_result = gf_processor.extract_greens_function(
                        data_dict[pair.station1], 
                        data_dict[pair.station2], 
                        freq_band
                    )
                    
                    if (gf_result['snr'] > 3.0 and 
                        gf_result['coherence'] > self.config.min_coherence):
                        correlations.append(gf_result['correlation'])
                        snrs.append(gf_result['snr'])
                
                if len(correlations) < 3:  # Need minimum correlations
                    continue
                
                # Stack correlations (weighted by SNR)
                correlations = np.array(correlations)
                snrs = np.array(snrs)
                weights = snrs / np.sum(snrs)
                
                # Weighted average
                reference_correlation = np.average(correlations, axis=0, weights=weights)
                
                # Store reference
                reference_results[pair_key][band_key] = {
                    'correlation': reference_correlation,
                    'lag_times': gf_result['lag_times'],  # Use last lag times (should be consistent)
                    'average_snr': np.mean(snrs),
                    'n_correlations': len(correlations),
                    'frequency_band': freq_band,
                    'establishment_date': datetime.now().isoformat()
                }
        
        # Store references
        self.reference_correlations = reference_results
        for pair_key in reference_results:
            self.reference_established[pair_key] = True
        
        return reference_results
    
    def _detect_ml_anomalies(self, current_correlations: Dict) -> Dict:
        """Detect anomalies using machine learning"""
        if not TORCH_AVAILABLE or 'velocity_detector' not in self.ml_models:
            return {}
        
        # This would require proper sequence data preparation
        # For now, return placeholder
        return {
            'ml_anomaly_detected': False,
            'anomaly_score': 0.0,
            'confidence': 0.0
        }
    
    def continuous_monitoring(self, data_stream_callback: Callable, 
                            alert_callback: Optional[Callable] = None) -> None:
        """Start continuous monitoring of ambient noise"""
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                # Get new data (this would be implemented based on data source)
                timestamp, data_dict = data_stream_callback()
                
                # Process data
                results = self.process_noise_data(data_dict, timestamp)
                
                # Check for alerts
                if alert_callback and results['anomalies']:
                    alert_callback(results)
                
                # Store results (in practice, would save to database)
                self._store_results(results)
                
            except KeyboardInterrupt:
                print("Monitoring stopped by user")
                break
            except Exception as e:
                warnings.warn(f"Error in monitoring: {e}")
                continue
        
        self.monitoring_active = False
    
    def _store_results(self, results: Dict) -> None:
        """Store monitoring results (placeholder for database storage)"""
        # In practice, this would store to a database
        pass
    
    def analyze_velocity_trends(self, time_window_days: int = 30) -> Dict:
        """Analyze velocity change trends over time"""
        
        if not self.current_correlations:
            return {'error': 'No correlation data available'}
        
        # Get timestamps within window
        cutoff_date = max(self.current_correlations.keys()) - timedelta(days=time_window_days)
        recent_correlations = {
            timestamp: data for timestamp, data in self.current_correlations.items()
            if timestamp >= cutoff_date
        }
        
        trend_analysis = {}
        
        for pair in self.station_pairs:
            pair_key = f"{pair.station1}-{pair.station2}"
            trend_analysis[pair_key] = {}
            
            for i, freq_band in enumerate(self.config.frequency_bands):
                band_key = f"band_{i}"
                
                # Collect velocity changes over time
                timestamps = []
                velocity_changes = []
                
                for timestamp, corr_data in recent_correlations.items():
                    if (pair_key in corr_data and 
                        band_key in corr_data[pair_key] and
                        'velocity_change' in corr_data[pair_key][band_key]):
                        
                        timestamps.append(timestamp)
                        velocity_changes.append(
                            corr_data[pair_key][band_key]['velocity_change']['velocity_change']
                        )
                
                if len(velocity_changes) < 3:
                    continue
                
                # Calculate trend statistics
                velocity_changes = np.array(velocity_changes)
                time_series = np.array([(t - timestamps[0]).total_seconds() / 86400 
                                      for t in timestamps])  # Days
                
                # Linear trend
                if len(time_series) > 1:
                    trend_slope = np.polyfit(time_series, velocity_changes, 1)[0]
                else:
                    trend_slope = 0.0
                
                trend_analysis[pair_key][band_key] = {
                    'mean_velocity_change': np.mean(velocity_changes),
                    'std_velocity_change': np.std(velocity_changes),
                    'trend_slope': trend_slope,  # %/day
                    'max_change': np.max(np.abs(velocity_changes)),
                    'n_measurements': len(velocity_changes),
                    'frequency_band': freq_band
                }
        
        return trend_analysis
    
    def generate_health_report(self) -> Dict:
        """Generate structural health monitoring report"""
        
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'station_pairs': len(self.station_pairs),
            'reference_established': sum(1 for established in self.reference_established.values() if established),
            'overall_health': 'unknown'
        }
        
        # Analyze recent velocity changes
        recent_trends = self.analyze_velocity_trends(time_window_days=7)
        
        if recent_trends:
            # Health assessment based on velocity changes
            max_changes = []
            for pair_data in recent_trends.values():
                for band_data in pair_data.values():
                    if 'max_change' in band_data:
                        max_changes.append(band_data['max_change'])
            
            if max_changes:
                max_overall_change = max(max_changes)
                
                if max_overall_change < 0.005:  # < 0.5%
                    health_status = 'excellent'
                elif max_overall_change < 0.01:  # < 1%
                    health_status = 'good'
                elif max_overall_change < 0.02:  # < 2%
                    health_status = 'fair'
                elif max_overall_change < 0.05:  # < 5%
                    health_status = 'poor'
                else:
                    health_status = 'critical'
                
                health_report['overall_health'] = health_status
                health_report['max_velocity_change'] = max_overall_change
            
            health_report['trend_analysis'] = recent_trends
        
        return health_report
    
    def export_monitoring_data(self, filename: str, time_window_days: int = 30) -> None:
        """Export monitoring data to file"""
        
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'config': {
                'sample_rate': self.config.sample_rate,
                'frequency_bands': self.config.frequency_bands,
                'velocity_bands': self.config.velocity_bands,
                'reference_window_days': self.config.reference_window_days
            },
            'station_pairs': [
                {
                    'station1': pair.station1,
                    'station2': pair.station2,
                    'distance': pair.distance,
                    'azimuth': pair.azimuth
                }
                for pair in self.station_pairs
            ],
            'reference_correlations': self.reference_correlations,
            'monitoring_results': {
                timestamp.isoformat(): data
                for timestamp, data in self.current_correlations.items()
                if timestamp >= cutoff_date
            },
            'health_report': self.generate_health_report()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to export data: {e}")


# Example usage and testing
def main():
    """Example usage of ambient noise monitoring"""
    
    # Create example station pairs
    station_pairs = [
        StationPair(
            station1='STA01', station2='STA02',
            distance=5.0, azimuth=45.0,
            coordinates1=(0.0, 0.0), coordinates2=(3.5, 3.5)
        ),
        StationPair(
            station1='STA01', station2='STA03',
            distance=8.0, azimuth=135.0,
            coordinates1=(0.0, 0.0), coordinates2=(-5.7, 5.7)
        ),
        StationPair(
            station1='STA02', station2='STA03',
            distance=10.0, azimuth=90.0,
            coordinates1=(3.5, 3.5), coordinates2=(-5.7, 5.7)
        )
    ]
    
    # Initialize monitoring system
    config = NoiseConfig(
        sample_rate=50.0,
        window_length=1800.0,  # 30 minutes for testing
        frequency_bands=[(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
        reference_window_days=7
    )
    
    monitor = AmbientNoiseMonitor(station_pairs, config)
    
    # Generate synthetic ambient noise data
    duration = 1800.0  # 30 minutes
    n_samples = int(duration * config.sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Simulate data for multiple days
    data_history = {}
    
    for day in range(10):  # 10 days of data
        timestamp = datetime.now() - timedelta(days=9-day)
        
        # Generate synthetic noise for each station
        station_data = {}
        
        for i, station in enumerate(['STA01', 'STA02', 'STA03']):
            # Different noise characteristics for each station
            noise_level = 1.0 + 0.1 * i
            
            # Add some coherent signal (simulating distant earthquake)
            coherent_signal = 0.1 * np.sin(2 * np.pi * 1.5 * time + i * np.pi/3) * \
                            np.exp(-(time - 900)**2 / 200000)
            
            # Random noise
            random_noise = noise_level * np.random.randn(n_samples)
            
            # Combine signals
            station_data[station] = coherent_signal + random_noise
        
        data_history[timestamp] = station_data
    
    print("Establishing reference Green's functions...")
    
    # Establish reference from first 7 days
    reference_data = {
        timestamp: data for timestamp, data in data_history.items()
        if (max(data_history.keys()) - timestamp).days >= 3
    }
    
    references = monitor.establish_reference(reference_data, window_days=7)
    print(f"Established references for {len(references)} station pairs")
    
    # Process recent data for velocity change detection
    print("\nProcessing recent data...")
    recent_results = []
    
    for timestamp, data_dict in list(data_history.items())[-3:]:  # Last 3 days
        results = monitor.process_noise_data(data_dict, timestamp)
        recent_results.append(results)
        
        print(f"Processed {timestamp.strftime('%Y-%m-%d')}:")
        print(f"  - Correlations: {len(results['correlations'])} pairs")
        print(f"  - Anomalies: {len(results['anomalies'])} detected")
    
    # Analyze trends
    print("\nAnalyzing velocity trends...")
    trends = monitor.analyze_velocity_trends(time_window_days=7)
    
    for pair_key, pair_trends in trends.items():
        print(f"\nStation pair {pair_key}:")
        for band_key, band_trends in pair_trends.items():
            freq_band = band_trends['frequency_band']
            mean_change = band_trends['mean_velocity_change']
            trend_slope = band_trends['trend_slope']
            print(f"  {freq_band[0]}-{freq_band[1]} Hz: "
                  f"mean Δv/v = {mean_change:.4f}, trend = {trend_slope:.6f}/day")
    
    # Generate health report
    print("\nGenerating health report...")
    health_report = monitor.generate_health_report()
    print(f"Overall health status: {health_report['overall_health']}")
    print(f"Maximum velocity change: {health_report.get('max_velocity_change', 'N/A')}")
    
    # Export results
    monitor.export_monitoring_data('ambient_noise_monitoring.json', time_window_days=10)
    print("\nResults exported to ambient_noise_monitoring.json")
    
    return monitor, health_report


if __name__ == "__main__":
    monitor, health_report = main()