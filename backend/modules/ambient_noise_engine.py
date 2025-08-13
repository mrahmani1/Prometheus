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
        """Comprehensive Dynamic Time Warping method with physics-based seismic correlation analysis"""
        
        try:
            # Initialize comprehensive DTW parameters
            n = len(ref_corr)
            m = len(cur_corr)
            dt = 1.0 / self.config.sample_rate
            
            # Pre-process signals with proper seismic filtering
            ref_processed, cur_processed = self._preprocess_for_dtw(ref_corr, cur_corr)
            
            # Multi-scale DTW analysis
            dtw_results = self._multi_scale_dtw_analysis(ref_processed, cur_processed)
            
            # Comprehensive distance matrix computation with seismic-aware metrics
            distance_matrix = self._compute_seismic_distance_matrix(ref_processed, cur_processed)
            
            # Dynamic programming solution with seismic constraints
            dtw_path, cumulative_distance = self._dtw_dynamic_programming(distance_matrix)
            
            # Extract velocity changes from DTW path with physics-based interpretation
            velocity_measurements = self._extract_velocity_from_dtw_path(
                dtw_path, ref_processed, cur_processed, dt
            )
            
            # Multi-frequency band analysis for robust velocity estimation
            frequency_band_results = self._frequency_band_dtw_analysis(
                ref_corr, cur_corr, dtw_path
            )
            
            # Uncertainty quantification using bootstrap resampling
            uncertainty_metrics = self._dtw_uncertainty_quantification(
                ref_processed, cur_processed, dtw_path, n_bootstrap=100
            )
            
            # Compute seismic wave travel time corrections
            travel_time_corrections = self._compute_travel_time_corrections(
                dtw_path, velocity_measurements
            )
            
            # Quality metrics and coherence analysis
            quality_metrics = self._compute_dtw_quality_metrics(
                ref_processed, cur_processed, dtw_path, distance_matrix
            )
            
            # Final velocity change with comprehensive error analysis
            final_velocity_change = self._integrate_velocity_measurements(
                velocity_measurements, frequency_band_results, uncertainty_metrics
            )
            
            # Seismic wave type identification and dispersion analysis
            wave_analysis = self._analyze_seismic_wave_characteristics(
                ref_processed, cur_processed, dtw_path, final_velocity_change
            )
            
            return {
                'velocity_change': final_velocity_change['velocity'],
                'error': final_velocity_change['error'],
                'correlation_coefficient': quality_metrics['correlation'],
                'method': 'dtw_comprehensive',
                'time_shift': final_velocity_change['time_shift'],
                'dtw_path': dtw_path,
                'cumulative_distance': cumulative_distance,
                'multi_scale_results': dtw_results,
                'frequency_band_analysis': frequency_band_results,
                'uncertainty_metrics': uncertainty_metrics,
                'travel_time_corrections': travel_time_corrections,
                'quality_metrics': quality_metrics,
                'wave_analysis': wave_analysis,
                'path_statistics': {
                    'path_length': len(dtw_path),
                    'average_warp': np.mean([abs(i - j) for i, j in dtw_path]),
                    'max_warp': max([abs(i - j) for i, j in dtw_path]),
                    'warp_variance': np.var([abs(i - j) for i, j in dtw_path])
                }
            }
            
        except Exception as e:
            # Fallback to robust cross-correlation if DTW fails
            return self._fallback_correlation_analysis(ref_corr, cur_corr, str(e))
    
    def _preprocess_for_dtw(self, ref_corr: np.ndarray, cur_corr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-process correlation functions for DTW analysis"""
        
        # Normalize signals
        ref_norm = (ref_corr - np.mean(ref_corr)) / np.std(ref_corr)
        cur_norm = (cur_corr - np.mean(cur_corr)) / np.std(cur_corr)
        
        # Apply seismic-specific filtering
        if SCIPY_AVAILABLE:
            from scipy.signal import butter, filtfilt, detrend
            
            # Detrend signals
            ref_norm = detrend(ref_norm)
            cur_norm = detrend(cur_norm)
            
            # Bandpass filter for seismic frequencies
            nyquist = self.config.sample_rate / 2.0
            low_freq = 0.1 / nyquist  # 0.1 Hz
            high_freq = min(10.0, nyquist * 0.9) / nyquist  # Up to 10 Hz or Nyquist
            
            b, a = butter(4, [low_freq, high_freq], btype='band')
            ref_filtered = filtfilt(b, a, ref_norm)
            cur_filtered = filtfilt(b, a, cur_norm)
            
            return ref_filtered, cur_filtered
        
        return ref_norm, cur_norm
    
    def _multi_scale_dtw_analysis(self, ref_corr: np.ndarray, cur_corr: np.ndarray) -> Dict:
        """Multi-scale DTW analysis for robust velocity measurements"""
        
        scales = [1, 2, 4, 8]  # Downsampling factors
        results = {}
        
        for scale in scales:
            if len(ref_corr) // scale < 10:  # Minimum length requirement
                continue
            
            # Downsample signals
            ref_downsampled = ref_corr[::scale]
            cur_downsampled = cur_corr[::scale]
            
            # Compute DTW at this scale
            distance_matrix = self._compute_seismic_distance_matrix(ref_downsampled, cur_downsampled)
            dtw_path, cumulative_distance = self._dtw_dynamic_programming(distance_matrix)
            
            results[f'scale_{scale}'] = {
                'path': dtw_path,
                'distance': cumulative_distance,
                'length': len(dtw_path)
            }
        
        return results
    
    def _compute_seismic_distance_matrix(self, ref_corr: np.ndarray, cur_corr: np.ndarray) -> np.ndarray:
        """Compute distance matrix with seismic-aware metrics"""
        
        n, m = len(ref_corr), len(cur_corr)
        distance_matrix = np.zeros((n, m))
        
        # Use multiple distance metrics appropriate for seismic correlations
        for i in range(n):
            for j in range(m):
                # Euclidean distance
                euclidean_dist = abs(ref_corr[i] - cur_corr[j])
                
                # Phase-sensitive distance for seismic signals
                if i > 0 and j > 0:
                    ref_gradient = ref_corr[i] - ref_corr[i-1]
                    cur_gradient = cur_corr[j] - cur_corr[j-1]
                    phase_dist = abs(ref_gradient - cur_gradient)
                else:
                    phase_dist = 0
                
                # Envelope-based distance for amplitude variations
                ref_envelope = abs(ref_corr[i])
                cur_envelope = abs(cur_corr[j])
                envelope_dist = abs(ref_envelope - cur_envelope)
                
                # Weighted combination for seismic signals
                distance_matrix[i, j] = (
                    0.6 * euclidean_dist +
                    0.3 * phase_dist +
                    0.1 * envelope_dist
                )
        
        return distance_matrix
    
    def _dtw_dynamic_programming(self, distance_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], float]:
        """Dynamic programming solution for DTW with seismic constraints"""
        
        n, m = distance_matrix.shape
        
        # Initialize cumulative cost matrix
        cumulative_cost = np.full((n, m), np.inf)
        cumulative_cost[0, 0] = distance_matrix[0, 0]
        
        # Fill first row and column
        for i in range(1, n):
            cumulative_cost[i, 0] = distance_matrix[i, 0] + cumulative_cost[i-1, 0]
        for j in range(1, m):
            cumulative_cost[0, j] = distance_matrix[0, j] + cumulative_cost[0, j-1]
        
        # Fill cumulative cost matrix with seismic constraints
        for i in range(1, n):
            for j in range(1, m):
                # Standard DTW moves with seismic-appropriate weighting
                cost_diagonal = cumulative_cost[i-1, j-1]
                cost_vertical = cumulative_cost[i-1, j] * 1.1  # Slightly penalize vertical moves
                cost_horizontal = cumulative_cost[i, j-1] * 1.1  # Slightly penalize horizontal moves
                
                # Seismic constraint: limit excessive warping
                warp_penalty = 0
                if abs(i - j) > min(n, m) * 0.1:  # 10% warp tolerance
                    warp_penalty = 0.5
                
                cumulative_cost[i, j] = distance_matrix[i, j] + min(
                    cost_diagonal,
                    cost_vertical + warp_penalty,
                    cost_horizontal + warp_penalty
                )
        
        # Backtrack to find optimal path
        path = []
        i, j = n - 1, m - 1
        
        while i > 0 or j > 0:
            path.append((i, j))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Choose the path with minimum cost
                costs = [
                    cumulative_cost[i-1, j-1],
                    cumulative_cost[i-1, j],
                    cumulative_cost[i, j-1]
                ]
                min_idx = np.argmin(costs)
                
                if min_idx == 0:
                    i -= 1
                    j -= 1
                elif min_idx == 1:
                    i -= 1
                else:
                    j -= 1
        
        path.append((0, 0))
        path.reverse()
        
        return path, cumulative_cost[n-1, m-1]
    
    def _extract_velocity_from_dtw_path(self, dtw_path: List[Tuple[int, int]], 
                                      ref_corr: np.ndarray, cur_corr: np.ndarray, 
                                      dt: float) -> Dict:
        """Extract velocity changes from DTW path with physics-based interpretation"""
        
        n_points = len(dtw_path)
        time_shifts = []
        local_velocities = []
        
        for k, (i, j) in enumerate(dtw_path):
            # Calculate local time shift
            expected_time = i * dt
            actual_time = j * dt
            local_time_shift = actual_time - expected_time
            time_shifts.append(local_time_shift)
            
            # Convert to velocity change using seismic wave theory
            if self.station_pair.distance > 0:
                # Use different wave velocities for different parts of correlation
                correlation_time = i * dt
                
                # Estimate appropriate reference velocity based on correlation lag
                if correlation_time < 10:  # Surface waves
                    reference_velocity = 3.5  # km/s
                elif correlation_time < 50:  # Body waves
                    reference_velocity = 6.0  # km/s
                else:  # Deep structure
                    reference_velocity = 8.0  # km/s
                
                # Compute relative velocity change
                travel_time = self.station_pair.distance / reference_velocity
                if travel_time > 0:
                    local_velocity_change = -local_time_shift / travel_time
                    local_velocities.append(local_velocity_change)
        
        # Statistics of velocity measurements
        if local_velocities:
            mean_velocity = np.mean(local_velocities)
            std_velocity = np.std(local_velocities)
            median_velocity = np.median(local_velocities)
            
            # Robust velocity estimate using median and MAD
            mad_velocity = np.median(np.abs(np.array(local_velocities) - median_velocity))
            robust_velocity = median_velocity
            
            return {
                'local_velocities': local_velocities,
                'mean_velocity': mean_velocity,
                'median_velocity': median_velocity,
                'std_velocity': std_velocity,
                'mad_velocity': mad_velocity * 1.4826,  # Convert MAD to std equivalent
                'robust_velocity': robust_velocity,
                'time_shifts': time_shifts,
                'n_measurements': len(local_velocities)
            }
        
        return {
            'local_velocities': [],
            'mean_velocity': 0.0,
            'median_velocity': 0.0,
            'std_velocity': 0.01,
            'mad_velocity': 0.01,
            'robust_velocity': 0.0,
            'time_shifts': time_shifts,
            'n_measurements': 0
        }
    
    def _frequency_band_dtw_analysis(self, ref_corr: np.ndarray, cur_corr: np.ndarray, 
                                   dtw_path: List[Tuple[int, int]]) -> Dict:
        """Multi-frequency band DTW analysis for robust velocity estimation"""
        
        results = {}
        
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy required for frequency band analysis'}
        
        from scipy.signal import butter, filtfilt
        
        # Define frequency bands for seismic analysis
        frequency_bands = [
            (0.1, 0.5, 'very_low'),
            (0.5, 1.0, 'low'),
            (1.0, 3.0, 'medium'),
            (3.0, 10.0, 'high')
        ]
        
        nyquist = self.config.sample_rate / 2.0
        
        for f_low, f_high, band_name in frequency_bands:
            if f_high >= nyquist:
                continue
            
            try:
                # Design bandpass filter
                b, a = butter(4, [f_low/nyquist, f_high/nyquist], btype='band')
                
                # Filter signals
                ref_filtered = filtfilt(b, a, ref_corr)
                cur_filtered = filtfilt(b, a, cur_corr)
                
                # Apply DTW path to filtered signals
                path_velocity = self._extract_velocity_from_dtw_path(
                    dtw_path, ref_filtered, cur_filtered, 1.0 / self.config.sample_rate
                )
                
                results[band_name] = {
                    'frequency_range': (f_low, f_high),
                    'velocity_change': path_velocity['robust_velocity'],
                    'uncertainty': path_velocity['mad_velocity'],
                    'correlation': self._compute_filtered_correlation(ref_filtered, cur_filtered, dtw_path),
                    'snr': self._compute_snr(ref_filtered, cur_filtered)
                }
                
            except Exception as e:
                results[band_name] = {'error': str(e)}
        
        return results
    
    def _dtw_uncertainty_quantification(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                                      dtw_path: List[Tuple[int, int]], n_bootstrap: int = 100) -> Dict:
        """Uncertainty quantification using bootstrap resampling"""
        
        bootstrap_velocities = []
        bootstrap_paths = []
        
        n_samples = min(len(ref_corr), len(cur_corr))
        
        for _ in range(n_bootstrap):
            # Bootstrap resampling with replacement
            indices = np.random.choice(n_samples, size=int(n_samples * 0.8), replace=True)
            indices = np.sort(indices)
            
            ref_bootstrap = ref_corr[indices]
            cur_bootstrap = cur_corr[indices]
            
            try:
                # Compute DTW for bootstrap sample
                distance_matrix = self._compute_seismic_distance_matrix(ref_bootstrap, cur_bootstrap)
                bootstrap_path, _ = self._dtw_dynamic_programming(distance_matrix)
                bootstrap_paths.append(bootstrap_path)
                
                # Extract velocity from bootstrap path
                velocity_result = self._extract_velocity_from_dtw_path(
                    bootstrap_path, ref_bootstrap, cur_bootstrap, 1.0 / self.config.sample_rate
                )
                
                if velocity_result['n_measurements'] > 0:
                    bootstrap_velocities.append(velocity_result['robust_velocity'])
                    
            except:
                continue
        
        # Calculate uncertainty statistics
        if bootstrap_velocities:
            bootstrap_velocities = np.array(bootstrap_velocities)
            
            return {
                'bootstrap_mean': np.mean(bootstrap_velocities),
                'bootstrap_std': np.std(bootstrap_velocities),
                'bootstrap_median': np.median(bootstrap_velocities),
                'confidence_interval_95': np.percentile(bootstrap_velocities, [2.5, 97.5]),
                'confidence_interval_68': np.percentile(bootstrap_velocities, [16, 84]),
                'n_successful_bootstrap': len(bootstrap_velocities),
                'bootstrap_range': (np.min(bootstrap_velocities), np.max(bootstrap_velocities))
            }
        
        return {
            'bootstrap_mean': 0.0,
            'bootstrap_std': 0.01,
            'bootstrap_median': 0.0,
            'confidence_interval_95': [-0.01, 0.01],
            'confidence_interval_68': [-0.005, 0.005],
            'n_successful_bootstrap': 0,
            'bootstrap_range': (0.0, 0.0)
        }
    
    def _compute_travel_time_corrections(self, dtw_path: List[Tuple[int, int]], 
                                       velocity_measurements: Dict) -> Dict:
        """Compute seismic wave travel time corrections"""
        
        dt = 1.0 / self.config.sample_rate
        
        # Analyze travel time variations along the DTW path
        early_path = dtw_path[:len(dtw_path)//3]
        middle_path = dtw_path[len(dtw_path)//3:2*len(dtw_path)//3]
        late_path = dtw_path[2*len(dtw_path)//3:]
        
        segments = {
            'early_arrivals': early_path,
            'body_waves': middle_path,
            'late_arrivals': late_path
        }
        
        corrections = {}
        
        for segment_name, segment_path in segments.items():
            time_shifts = []
            
            for i, j in segment_path:
                expected_time = i * dt
                actual_time = j * dt
                time_shift = actual_time - expected_time
                time_shifts.append(time_shift)
            
            if time_shifts:
                corrections[segment_name] = {
                    'mean_time_shift': np.mean(time_shifts),
                    'std_time_shift': np.std(time_shifts),
                    'max_time_shift': np.max(np.abs(time_shifts)),
                    'systematic_shift': np.median(time_shifts)
                }
        
        return corrections
    
    def _compute_dtw_quality_metrics(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                                   dtw_path: List[Tuple[int, int]], distance_matrix: np.ndarray) -> Dict:
        """Compute quality metrics for DTW alignment"""
        
        # Path-based correlation
        aligned_ref = []
        aligned_cur = []
        
        for i, j in dtw_path:
            if i < len(ref_corr) and j < len(cur_corr):
                aligned_ref.append(ref_corr[i])
                aligned_cur.append(cur_corr[j])
        
        if len(aligned_ref) > 1:
            correlation_coeff = np.corrcoef(aligned_ref, aligned_cur)[0, 1]
        else:
            correlation_coeff = 0.0
        
        # Path efficiency (deviation from diagonal)
        diagonal_length = min(len(ref_corr), len(cur_corr))
        path_length = len(dtw_path)
        path_efficiency = diagonal_length / path_length if path_length > 0 else 0
        
        # Warping penalty
        warping_penalty = np.mean([abs(i - j) for i, j in dtw_path])
        
        # Distance statistics
        path_distances = [distance_matrix[i, j] for i, j in dtw_path 
                         if i < distance_matrix.shape[0] and j < distance_matrix.shape[1]]
        
        return {
            'correlation': correlation_coeff if not np.isnan(correlation_coeff) else 0.0,
            'path_efficiency': path_efficiency,
            'warping_penalty': warping_penalty,
            'mean_path_distance': np.mean(path_distances) if path_distances else 0,
            'max_path_distance': np.max(path_distances) if path_distances else 0,
            'path_smoothness': self._compute_path_smoothness(dtw_path)
        }
    
    def _integrate_velocity_measurements(self, velocity_measurements: Dict, 
                                       frequency_results: Dict, uncertainty_metrics: Dict) -> Dict:
        """Integrate velocity measurements from multiple analyses"""
        
        # Primary velocity estimate from DTW path
        primary_velocity = velocity_measurements.get('robust_velocity', 0.0)
        primary_error = velocity_measurements.get('mad_velocity', 0.01)
        
        # Weighted average from frequency bands
        frequency_velocities = []
        frequency_weights = []
        
        for band_name, results in frequency_results.items():
            if 'velocity_change' in results and 'uncertainty' in results:
                velocity = results['velocity_change']
                uncertainty = results['uncertainty']
                
                if uncertainty > 0:
                    weight = 1.0 / (uncertainty ** 2)
                    frequency_velocities.append(velocity)
                    frequency_weights.append(weight)
        
        # Compute weighted average
        if frequency_velocities:
            frequency_weights = np.array(frequency_weights)
            frequency_velocities = np.array(frequency_velocities)
            
            weighted_velocity = np.sum(frequency_weights * frequency_velocities) / np.sum(frequency_weights)
            weighted_error = 1.0 / np.sqrt(np.sum(frequency_weights))
        else:
            weighted_velocity = primary_velocity
            weighted_error = primary_error
        
        # Final integrated estimate with bootstrap uncertainty
        bootstrap_error = uncertainty_metrics.get('bootstrap_std', primary_error)
        
        # Combine errors (systematic + random)
        total_error = np.sqrt(weighted_error**2 + bootstrap_error**2)
        
        # Conservative final estimate
        final_velocity = (primary_velocity + weighted_velocity) / 2.0
        
        # Time shift calculation
        if self.station_pair.distance > 0:
            reference_velocity = 3.5  # km/s for surface waves
            travel_time = self.station_pair.distance / reference_velocity
            time_shift = final_velocity * travel_time
        else:
            time_shift = 0.0
        
        return {
            'velocity': final_velocity,
            'error': total_error,
            'time_shift': time_shift,
            'primary_velocity': primary_velocity,
            'frequency_weighted_velocity': weighted_velocity,
            'bootstrap_uncertainty': bootstrap_error,
            'measurement_quality': min(1.0, 1.0 / (1.0 + total_error * 100))
        }
    
    def _analyze_seismic_wave_characteristics(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                                            dtw_path: List[Tuple[int, int]], velocity_result: Dict) -> Dict:
        """Analyze seismic wave characteristics from DTW alignment"""
        
        dt = 1.0 / self.config.sample_rate
        
        # Identify different wave phases from DTW path
        path_length = len(dtw_path)
        
        # Early arrivals (first 20% of path) - direct waves/surface waves
        early_phase = dtw_path[:int(path_length * 0.2)]
        
        # Middle phase (20-60% of path) - body waves
        body_phase = dtw_path[int(path_length * 0.2):int(path_length * 0.6)]
        
        # Late phase (60-100% of path) - scattered/multiple waves
        late_phase = dtw_path[int(path_length * 0.6):]
        
        wave_analysis = {}
        
        for phase_name, phase_path in [('early', early_phase), ('body', body_phase), ('late', late_phase)]:
            if not phase_path:
                continue
            
            # Compute phase-specific velocity changes
            phase_velocities = []
            for i, j in phase_path:
                if i < len(ref_corr) and j < len(cur_corr):
                    time_shift = (j - i) * dt
                    if self.station_pair.distance > 0:
                        # Phase-appropriate reference velocities
                        if phase_name == 'early':
                            ref_vel = 3.5  # Surface waves
                        elif phase_name == 'body':
                            ref_vel = 6.0  # Body waves
                        else:
                            ref_vel = 4.0  # Scattered waves
                        
                        travel_time = self.station_pair.distance / ref_vel
                        if travel_time > 0:
                            velocity_change = -time_shift / travel_time
                            phase_velocities.append(velocity_change)
            
            if phase_velocities:
                wave_analysis[f'{phase_name}_phase'] = {
                    'velocity_change': np.median(phase_velocities),
                    'velocity_std': np.std(phase_velocities),
                    'n_measurements': len(phase_velocities),
                    'dominant_period': self._estimate_dominant_period(phase_path, dt),
                    'coherence': self._compute_phase_coherence(ref_corr, cur_corr, phase_path)
                }
        
        return wave_analysis
    
    def _compute_filtered_correlation(self, ref_filtered: np.ndarray, cur_filtered: np.ndarray,
                                    dtw_path: List[Tuple[int, int]]) -> float:
        """Compute correlation coefficient for filtered signals along DTW path"""
        
        aligned_ref = []
        aligned_cur = []
        
        for i, j in dtw_path:
            if i < len(ref_filtered) and j < len(cur_filtered):
                aligned_ref.append(ref_filtered[i])
                aligned_cur.append(cur_filtered[j])
        
        if len(aligned_ref) > 1:
            return np.corrcoef(aligned_ref, aligned_cur)[0, 1]
        return 0.0
    
    def _compute_snr(self, ref_signal: np.ndarray, cur_signal: np.ndarray) -> float:
        """Compute signal-to-noise ratio"""
        
        # Use robust statistics for SNR estimation
        ref_power = np.median(ref_signal**2)
        cur_power = np.median(cur_signal**2)
        
        # Estimate noise as difference between signals
        if len(ref_signal) == len(cur_signal):
            noise = ref_signal - cur_signal
            noise_power = np.median(noise**2)
            
            if noise_power > 0:
                snr = 10 * np.log10((ref_power + cur_power) / (2 * noise_power))
                return max(0, snr)  # Clip negative SNR
        
        return 10.0  # Default SNR
    
    def _compute_path_smoothness(self, dtw_path: List[Tuple[int, int]]) -> float:
        """Compute smoothness of DTW path"""
        
        if len(dtw_path) < 3:
            return 1.0
        
        # Compute second differences of path
        second_diffs = []
        for k in range(2, len(dtw_path)):
            i_prev2, j_prev2 = dtw_path[k-2]
            i_prev1, j_prev1 = dtw_path[k-1]
            i_curr, j_curr = dtw_path[k]
            
            # Second difference in both dimensions
            second_diff_i = i_curr - 2*i_prev1 + i_prev2
            second_diff_j = j_curr - 2*j_prev1 + j_prev2
            
            second_diff_magnitude = np.sqrt(second_diff_i**2 + second_diff_j**2)
            second_diffs.append(second_diff_magnitude)
        
        # Smoothness as inverse of mean absolute second difference
        if second_diffs:
            mean_second_diff = np.mean(second_diffs)
            return 1.0 / (1.0 + mean_second_diff)
        
        return 1.0
    
    def _estimate_dominant_period(self, phase_path: List[Tuple[int, int]], dt: float) -> float:
        """Estimate dominant period of seismic phase"""
        
        if len(phase_path) < 10:
            return 1.0
        
        # Extract time series along path
        time_indices = [i for i, j in phase_path]
        
        # Estimate period from autocorrelation of path indices
        if len(time_indices) > 20:
            autocorr = np.correlate(time_indices, time_indices, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find first minimum after maximum (at lag 0)
            if len(autocorr) > 5:
                first_min_idx = np.argmin(autocorr[1:5]) + 1
                dominant_period = first_min_idx * dt * 2  # Factor of 2 for full period
                return max(0.1, dominant_period)
        
        return 1.0
    
    def _compute_phase_coherence(self, ref_corr: np.ndarray, cur_corr: np.ndarray,
                               phase_path: List[Tuple[int, int]]) -> float:
        """Compute coherence for a specific phase"""
        
        phase_ref = []
        phase_cur = []
        
        for i, j in phase_path:
            if i < len(ref_corr) and j < len(cur_corr):
                phase_ref.append(ref_corr[i])
                phase_cur.append(cur_corr[j])
        
        if len(phase_ref) > 5:
            correlation = np.corrcoef(phase_ref, phase_cur)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _fallback_correlation_analysis(self, ref_corr: np.ndarray, cur_corr: np.ndarray, error_msg: str) -> Dict:
        """Fallback to robust cross-correlation analysis if DTW fails"""
        
        try:
            # Ensure same length
            min_len = min(len(ref_corr), len(cur_corr))
            ref_corr = ref_corr[:min_len]
            cur_corr = cur_corr[:min_len]
            
            # Cross-correlation analysis
            correlation = np.correlate(ref_corr, cur_corr, mode='full')
            lags = np.arange(-len(cur_corr)+1, len(ref_corr))
            
            # Find optimal lag
            optimal_lag_idx = np.argmax(correlation)
            optimal_lag = lags[optimal_lag_idx]
            
            # Convert to velocity change
            dt = 1.0 / self.config.sample_rate
            time_shift = optimal_lag * dt
            
            if self.station_pair.distance > 0:
                travel_time = self.station_pair.distance / 3.5  # Surface wave velocity
                velocity_change = -time_shift / travel_time
            else:
                velocity_change = 0.0
            
            # Correlation coefficient
            max_correlation = correlation[optimal_lag_idx] / (np.linalg.norm(ref_corr) * np.linalg.norm(cur_corr))
            
            return {
                'velocity_change': velocity_change,
                'error': 0.02,  # Higher error for fallback method
                'correlation_coefficient': max_correlation,
                'method': 'dtw_fallback_correlation',
                'time_shift': time_shift,
                'dtw_error': error_msg,
                'fallback_used': True
            }
            
        except Exception as fallback_error:
            return {
                'velocity_change': 0.0,
                'error': 0.05,
                'correlation_coefficient': 0.0,
                'method': 'dtw_failed',
                'time_shift': 0.0,
                'dtw_error': error_msg,
                'fallback_error': str(fallback_error)
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
        """
        Comprehensive ML-based anomaly detection using neural networks
        with multi-modal analysis, uncertainty quantification, and temporal modeling
        """
        if not TORCH_AVAILABLE or 'velocity_detector' not in self.ml_models:
            return {
                'ml_anomaly_detected': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'error': 'PyTorch not available or models not initialized'
            }
        
        try:
            # Step 1: Prepare input sequences from correlation data
            sequence_data = self._prepare_correlation_sequences(current_correlations)
            
            if sequence_data is None or len(sequence_data) == 0:
                return {
                    'ml_anomaly_detected': False,
                    'anomaly_score': 0.0,
                    'confidence': 0.0,
                    'warning': 'Insufficient data for ML analysis'
                }
            
            # Step 2: Multi-model ensemble prediction
            ensemble_results = self._run_ensemble_prediction(sequence_data)
            
            # Step 3: Temporal pattern analysis
            temporal_features = self._extract_temporal_features(sequence_data)
            temporal_anomalies = self._detect_temporal_anomalies(temporal_features)
            
            # Step 4: Spectral anomaly detection
            spectral_features = self._extract_spectral_features(sequence_data)
            spectral_anomalies = self._detect_spectral_anomalies(spectral_features)
            
            # Step 5: Cross-correlation pattern analysis
            correlation_patterns = self._analyze_correlation_patterns(sequence_data)
            pattern_anomalies = self._detect_pattern_anomalies(correlation_patterns)
            
            # Step 6: Uncertainty quantification using Monte Carlo dropout
            uncertainty_metrics = self._quantify_prediction_uncertainty(sequence_data)
            
            # Step 7: Multi-scale anomaly detection
            multiscale_anomalies = self._detect_multiscale_anomalies(sequence_data)
            
            # Step 8: Fusion of all anomaly indicators
            fused_results = self._fuse_anomaly_indicators({
                'ensemble': ensemble_results,
                'temporal': temporal_anomalies,
                'spectral': spectral_anomalies,
                'pattern': pattern_anomalies,
                'multiscale': multiscale_anomalies,
                'uncertainty': uncertainty_metrics
            })
            
            # Step 9: Contextual anomaly scoring
            context_score = self._calculate_contextual_anomaly_score(
                fused_results, current_correlations
            )
            
            # Step 10: Generate comprehensive anomaly report
            final_score = fused_results['composite_score']
            final_confidence = fused_results['confidence']
            
            # Adaptive thresholding based on historical data
            adaptive_threshold = self._calculate_adaptive_threshold()
            is_anomaly = final_score > adaptive_threshold
            
            # Detailed anomaly characterization
            anomaly_characteristics = self._characterize_anomaly(fused_results) if is_anomaly else {}
            
            # Physical interpretation of detected anomalies
            physical_interpretation = self._interpret_anomaly_physically(
                fused_results, current_correlations
            ) if is_anomaly else {}
            
            return {
                'ml_anomaly_detected': is_anomaly,
                'anomaly_score': float(final_score),
                'confidence': float(final_confidence),
                'adaptive_threshold': float(adaptive_threshold),
                'anomaly_components': {
                    'ensemble_score': float(ensemble_results.get('score', 0.0)),
                    'temporal_score': float(temporal_anomalies.get('score', 0.0)),
                    'spectral_score': float(spectral_anomalies.get('score', 0.0)),
                    'pattern_score': float(pattern_anomalies.get('score', 0.0)),
                    'multiscale_score': float(multiscale_anomalies.get('score', 0.0))
                },
                'uncertainty_metrics': {
                    'epistemic_uncertainty': float(uncertainty_metrics.get('epistemic', 0.0)),
                    'aleatoric_uncertainty': float(uncertainty_metrics.get('aleatoric', 0.0)),
                    'total_uncertainty': float(uncertainty_metrics.get('total', 0.0))
                },
                'anomaly_characteristics': anomaly_characteristics,
                'physical_interpretation': physical_interpretation,
                'context_factors': {
                    'historical_context': context_score.get('historical', 0.0),
                    'spatial_context': context_score.get('spatial', 0.0),
                    'frequency_context': context_score.get('frequency', 0.0)
                },
                'model_diagnostics': {
                    'data_quality': self._assess_data_quality(sequence_data),
                    'model_performance': self._assess_model_performance(),
                    'processing_time': fused_results.get('processing_time', 0.0)
                }
            }
            
        except Exception as e:
            import traceback
            return {
                'ml_anomaly_detected': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'error': f'ML anomaly detection failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def _prepare_correlation_sequences(self, current_correlations: Dict) -> Optional[torch.Tensor]:
        """Prepare correlation data for ML model input"""
        if not current_correlations:
            return None
        
        try:
            # Collect correlation data for all station pairs and frequency bands
            sequences = []
            n_bands = len(self.config.frequency_bands)
            
            for pair_key, correlation_data in current_correlations.items():
                if 'frequency_bands' not in correlation_data:
                    continue
                
                # Extract correlation functions for each frequency band
                band_correlations = []
                for band_idx, (low_freq, high_freq) in enumerate(self.config.frequency_bands):
                    if band_idx < len(correlation_data['frequency_bands']):
                        corr_func = correlation_data['frequency_bands'][band_idx]['correlation']
                        # Standardize correlation function length
                        if len(corr_func) > 1000:
                            corr_func = corr_func[:1000]
                        elif len(corr_func) < 1000:
                            corr_func = np.pad(corr_func, (0, 1000 - len(corr_func)))
                        
                        band_correlations.append(corr_func)
                    else:
                        band_correlations.append(np.zeros(1000))
                
                if len(band_correlations) == n_bands:
                    sequences.append(np.array(band_correlations))
            
            if not sequences:
                return None
            
            # Convert to tensor (batch_size=1 for single time step, can be extended for sequences)
            sequence_array = np.array(sequences)
            sequence_tensor = torch.FloatTensor(sequence_array).unsqueeze(0)  # Add batch dimension
            
            return sequence_tensor
            
        except Exception as e:
            warnings.warn(f"Failed to prepare correlation sequences: {e}")
            return None
    
    def _run_ensemble_prediction(self, sequence_data: torch.Tensor) -> Dict:
        """Run ensemble prediction with multiple models and dropout sampling"""
        model = self.ml_models['velocity_detector']
        model.eval()
        
        start_time = datetime.now()
        ensemble_predictions = []
        
        # Monte Carlo sampling with dropout
        n_samples = 50
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout for uncertainty estimation
                model.train()  # Enables dropout
                prediction = model(sequence_data)
                ensemble_predictions.append(prediction['anomaly_score'].item())
                model.eval()  # Disable dropout for final prediction
        
        # Final prediction without dropout
        with torch.no_grad():
            final_prediction = model(sequence_data)
        
        # Ensemble statistics
        ensemble_mean = np.mean(ensemble_predictions)
        ensemble_std = np.std(ensemble_predictions)
        ensemble_confidence = 1.0 / (1.0 + ensemble_std)  # Higher std = lower confidence
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'score': ensemble_mean,
            'confidence': ensemble_confidence,
            'ensemble_std': ensemble_std,
            'final_prediction': final_prediction['anomaly_score'].item(),
            'velocity_changes': final_prediction['velocity_changes'].numpy(),
            'processing_time': processing_time
        }
    
    def _extract_temporal_features(self, sequence_data: torch.Tensor) -> Dict:
        """Extract temporal features from correlation sequences"""
        # Convert to numpy for feature extraction
        data = sequence_data.squeeze().numpy()  # Remove batch dimension
        
        temporal_features = {}
        
        for i, band_data in enumerate(data):
            # Statistical features
            temporal_features[f'band_{i}'] = {
                'mean': np.mean(band_data),
                'std': np.std(band_data),
                'skewness': signal.correlate(band_data, band_data, mode='full').max(),
                'kurtosis': np.mean((band_data - np.mean(band_data))**4) / (np.std(band_data)**4),
                'energy': np.sum(band_data**2),
                'zero_crossings': np.sum(np.diff(np.signbit(band_data)))
            }
        
        return temporal_features
    
    def _detect_temporal_anomalies(self, temporal_features: Dict) -> Dict:
        """Detect anomalies in temporal features using statistical methods"""
        anomaly_scores = []
        
        for band, features in temporal_features.items():
            # Z-score based anomaly detection
            feature_values = list(features.values())
            mean_features = np.mean(feature_values)
            std_features = np.std(feature_values)
            
            if std_features > 0:
                z_scores = np.abs((np.array(feature_values) - mean_features) / std_features)
                band_anomaly = np.max(z_scores)
            else:
                band_anomaly = 0.0
            
            anomaly_scores.append(band_anomaly)
        
        overall_score = np.mean(anomaly_scores)
        max_score = np.max(anomaly_scores)
        
        return {
            'score': min(overall_score / 3.0, 1.0),  # Normalize to [0, 1]
            'max_band_score': max_score,
            'band_scores': anomaly_scores,
            'threshold': 3.0  # 3-sigma threshold
        }
    
    def _extract_spectral_features(self, sequence_data: torch.Tensor) -> Dict:
        """Extract spectral features using FFT analysis"""
        data = sequence_data.squeeze().numpy()
        spectral_features = {}
        
        for i, band_data in enumerate(data):
            # FFT analysis
            fft_data = np.fft.fft(band_data)
            power_spectrum = np.abs(fft_data)**2
            freqs = np.fft.fftfreq(len(band_data))
            
            # Spectral features
            spectral_features[f'band_{i}'] = {
                'spectral_centroid': np.sum(freqs * power_spectrum) / np.sum(power_spectrum),
                'spectral_spread': np.sqrt(np.sum(((freqs - spectral_features.get(f'band_{i}', {}).get('spectral_centroid', 0))**2) * power_spectrum) / np.sum(power_spectrum)) if i in spectral_features else 0,
                'spectral_rolloff': freqs[np.where(np.cumsum(power_spectrum) >= 0.95 * np.sum(power_spectrum))[0][0]],
                'spectral_flatness': np.exp(np.mean(np.log(power_spectrum + 1e-10))) / np.mean(power_spectrum),
                'dominant_frequency': freqs[np.argmax(power_spectrum[:len(freqs)//2])]
            }
        
        return spectral_features
    
    def _detect_spectral_anomalies(self, spectral_features: Dict) -> Dict:
        """Detect anomalies in spectral features"""
        anomaly_scores = []
        
        for band, features in spectral_features.items():
            # Check for unusual spectral characteristics
            feature_anomalies = []
            
            # Flatness anomaly (too flat or too peaky)
            flatness = features.get('spectral_flatness', 0.5)
            flatness_anomaly = abs(flatness - 0.5) * 2  # Normalize to [0, 1]
            feature_anomalies.append(flatness_anomaly)
            
            # Centroid shift anomaly
            centroid = features.get('spectral_centroid', 0.0)
            centroid_anomaly = min(abs(centroid) * 10, 1.0)  # Scale and cap at 1
            feature_anomalies.append(centroid_anomaly)
            
            band_score = np.mean(feature_anomalies)
            anomaly_scores.append(band_score)
        
        return {
            'score': np.mean(anomaly_scores),
            'band_scores': anomaly_scores,
            'features': spectral_features
        }
    
    def _analyze_correlation_patterns(self, sequence_data: torch.Tensor) -> Dict:
        """Analyze cross-correlation patterns for anomalies"""
        data = sequence_data.squeeze().numpy()
        
        pattern_metrics = {}
        
        # Cross-correlation between frequency bands
        n_bands = data.shape[0]
        correlation_matrix = np.zeros((n_bands, n_bands))
        
        for i in range(n_bands):
            for j in range(i, n_bands):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    corr_coef = np.corrcoef(data[i], data[j])[0, 1]
                    correlation_matrix[i, j] = corr_coef
                    correlation_matrix[j, i] = corr_coef
        
        pattern_metrics = {
            'correlation_matrix': correlation_matrix,
            'matrix_trace': np.trace(correlation_matrix),
            'matrix_determinant': np.linalg.det(correlation_matrix + 1e-6 * np.eye(n_bands)),
            'max_off_diagonal': np.max(correlation_matrix - np.eye(n_bands)),
            'condition_number': np.linalg.cond(correlation_matrix + 1e-6 * np.eye(n_bands))
        }
        
        return pattern_metrics
    
    def _detect_pattern_anomalies(self, correlation_patterns: Dict) -> Dict:
        """Detect anomalies in correlation patterns"""
        # Check for unusual correlation structures
        matrix = correlation_patterns['correlation_matrix']
        
        # Anomaly indicators
        anomaly_indicators = []
        
        # Very high off-diagonal correlations (unusual coupling)
        max_off_diag = correlation_patterns['max_off_diagonal']
        high_correlation_anomaly = max(0, max_off_diag - 0.7) / 0.3  # Threshold at 0.7
        anomaly_indicators.append(high_correlation_anomaly)
        
        # Very low condition number (near singular matrix)
        cond_num = correlation_patterns['condition_number']
        if cond_num > 1e3:  # High condition number indicates near-singular matrix
            singular_anomaly = min((cond_num - 1e3) / 1e3, 1.0)
        else:
            singular_anomaly = 0.0
        anomaly_indicators.append(singular_anomaly)
        
        # Low determinant (indicating linear dependencies)
        det_val = abs(correlation_patterns['matrix_determinant'])
        if det_val < 0.1:
            det_anomaly = (0.1 - det_val) / 0.1
        else:
            det_anomaly = 0.0
        anomaly_indicators.append(det_anomaly)
        
        overall_score = np.mean(anomaly_indicators)
        
        return {
            'score': min(overall_score, 1.0),
            'high_correlation_anomaly': high_correlation_anomaly,
            'singular_anomaly': singular_anomaly,
            'determinant_anomaly': det_anomaly
        }
    
    def _quantify_prediction_uncertainty(self, sequence_data: torch.Tensor) -> Dict:
        """Quantify prediction uncertainty using Bayesian neural networks"""
        model = self.ml_models['velocity_detector']
        
        # Epistemic uncertainty (model uncertainty) via Monte Carlo dropout
        model.train()  # Enable dropout
        mc_predictions = []
        
        n_samples = 30
        for _ in range(n_samples):
            with torch.no_grad():
                pred = model(sequence_data)
                mc_predictions.append(pred['anomaly_score'].item())
        
        mc_predictions = np.array(mc_predictions)
        epistemic_uncertainty = np.std(mc_predictions)
        
        # Aleatoric uncertainty (data uncertainty) - approximated
        model.eval()
        with torch.no_grad():
            final_pred = model(sequence_data)
            # Use confidence head output as aleatoric uncertainty proxy
            confidences = final_pred['confidences'].numpy()
            aleatoric_uncertainty = 1.0 - np.mean(confidences)
        
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        return {
            'epistemic': epistemic_uncertainty,
            'aleatoric': aleatoric_uncertainty,
            'total': total_uncertainty,
            'prediction_mean': np.mean(mc_predictions),
            'prediction_std': np.std(mc_predictions)
        }
    
    def _detect_multiscale_anomalies(self, sequence_data: torch.Tensor) -> Dict:
        """Detect anomalies at multiple temporal and frequency scales"""
        data = sequence_data.squeeze().numpy()
        
        multiscale_scores = []
        
        for band_data in data:
            # Multi-resolution analysis using wavelets (simplified)
            scales = [2, 4, 8, 16]
            scale_anomalies = []
            
            for scale in scales:
                if len(band_data) >= scale:
                    # Downsample data
                    downsampled = band_data[::scale]
                    
                    # Calculate scale-specific features
                    scale_std = np.std(downsampled)
                    scale_mean = np.mean(downsampled)
                    scale_range = np.max(downsampled) - np.min(downsampled)
                    
                    # Normalize scale features
                    scale_features = np.array([scale_std, abs(scale_mean), scale_range])
                    scale_features = scale_features / (np.linalg.norm(scale_features) + 1e-10)
                    
                    # Scale anomaly score
                    scale_anomaly = np.linalg.norm(scale_features - 0.5)  # Distance from neutral
                    scale_anomalies.append(scale_anomaly)
            
            multiscale_scores.append(np.mean(scale_anomalies))
        
        return {
            'score': np.mean(multiscale_scores),
            'band_scores': multiscale_scores,
            'scales_analyzed': scales
        }
    
    def _fuse_anomaly_indicators(self, anomaly_results: Dict) -> Dict:
        """Fuse multiple anomaly indicators into composite score"""
        start_time = datetime.now()
        
        # Extract individual scores with weights
        weights = {
            'ensemble': 0.3,
            'temporal': 0.2,
            'spectral': 0.2,
            'pattern': 0.15,
            'multiscale': 0.15
        }
        
        scores = {}
        confidences = {}
        
        for component, result in anomaly_results.items():
            if component == 'uncertainty':
                continue
            
            score = result.get('score', 0.0)
            confidence = result.get('confidence', 0.5)
            
            scores[component] = score
            confidences[component] = confidence
        
        # Weighted composite score
        composite_score = sum(weights[comp] * scores[comp] for comp in weights.keys())
        
        # Confidence-weighted composite confidence
        total_weight = sum(weights[comp] * confidences.get(comp, 0.5) for comp in weights.keys())
        composite_confidence = total_weight / sum(weights.values())
        
        # Adjust score based on uncertainty
        uncertainty = anomaly_results['uncertainty']
        uncertainty_penalty = uncertainty['total'] * 0.1  # Small penalty for high uncertainty
        adjusted_score = max(0, composite_score - uncertainty_penalty)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'composite_score': adjusted_score,
            'confidence': composite_confidence,
            'component_scores': scores,
            'component_confidences': confidences,
            'uncertainty_adjustment': uncertainty_penalty,
            'processing_time': processing_time
        }
    
    def _calculate_contextual_anomaly_score(self, fused_results: Dict, 
                                          current_correlations: Dict) -> Dict:
        """Calculate contextual anomaly scores based on historical and spatial context"""
        context_scores = {}
        
        # Historical context (compare with reference period)
        if self.reference_correlations:
            historical_deviations = []
            for pair_key in current_correlations:
                if pair_key in self.reference_correlations:
                    current_corr = current_correlations[pair_key]
                    reference_corr = self.reference_correlations[pair_key]
                    
                    # Compare correlation strengths
                    if 'max_correlation' in current_corr and 'max_correlation' in reference_corr:
                        deviation = abs(current_corr['max_correlation'] - reference_corr['max_correlation'])
                        historical_deviations.append(deviation)
            
            context_scores['historical'] = np.mean(historical_deviations) if historical_deviations else 0.0
        else:
            context_scores['historical'] = 0.0
        
        # Spatial context (consistency across station pairs)
        pair_anomalies = []
        for pair_key, corr_data in current_correlations.items():
            pair_anomaly = corr_data.get('anomaly_score', 0.0)
            pair_anomalies.append(pair_anomaly)
        
        if len(pair_anomalies) > 1:
            spatial_consistency = 1.0 - np.std(pair_anomalies)  # High std = low consistency
            context_scores['spatial'] = max(0, spatial_consistency)
        else:
            context_scores['spatial'] = 0.5
        
        # Frequency context (consistency across frequency bands)
        frequency_anomalies = []
        for pair_key, corr_data in current_correlations.items():
            if 'frequency_bands' in corr_data:
                for band_data in corr_data['frequency_bands']:
                    band_anomaly = band_data.get('anomaly_score', 0.0)
                    frequency_anomalies.append(band_anomaly)
        
        if len(frequency_anomalies) > 1:
            frequency_consistency = 1.0 - np.std(frequency_anomalies)
            context_scores['frequency'] = max(0, frequency_consistency)
        else:
            context_scores['frequency'] = 0.5
        
        return context_scores
    
    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold based on historical data"""
        # Simplified adaptive thresholding - in practice, this would use historical anomaly scores
        base_threshold = 0.7
        
        # Adjust based on available historical data
        if hasattr(self, 'historical_anomaly_scores') and len(self.historical_anomaly_scores) > 10:
            historical_mean = np.mean(self.historical_anomaly_scores)
            historical_std = np.std(self.historical_anomaly_scores)
            adaptive_threshold = historical_mean + 2 * historical_std
        else:
            adaptive_threshold = base_threshold
        
        return min(max(adaptive_threshold, 0.5), 0.95)  # Clamp between 0.5 and 0.95
    
    def _characterize_anomaly(self, fused_results: Dict) -> Dict:
        """Characterize detected anomalies"""
        characteristics = {}
        
        # Primary anomaly source
        component_scores = fused_results['component_scores']
        primary_source = max(component_scores, key=component_scores.get)
        
        characteristics['primary_source'] = primary_source
        characteristics['primary_score'] = component_scores[primary_source]
        
        # Anomaly severity
        composite_score = fused_results['composite_score']
        if composite_score > 0.9:
            characteristics['severity'] = 'critical'
        elif composite_score > 0.8:
            characteristics['severity'] = 'high'
        elif composite_score > 0.7:
            characteristics['severity'] = 'moderate'
        else:
            characteristics['severity'] = 'low'
        
        # Anomaly type classification
        if primary_source == 'ensemble':
            characteristics['type'] = 'neural_network_detected'
        elif primary_source == 'temporal':
            characteristics['type'] = 'temporal_pattern_change'
        elif primary_source == 'spectral':
            characteristics['type'] = 'spectral_characteristic_change'
        elif primary_source == 'pattern':
            characteristics['type'] = 'cross_correlation_pattern_change'
        else:
            characteristics['type'] = 'multi_scale_structural_change'
        
        return characteristics
    
    def _interpret_anomaly_physically(self, fused_results: Dict, 
                                    current_correlations: Dict) -> Dict:
        """Provide physical interpretation of detected anomalies"""
        interpretation = {}
        
        primary_source = fused_results['component_scores']
        max_component = max(primary_source, key=primary_source.get)
        
        # Physical interpretations based on anomaly type
        if max_component == 'ensemble':
            interpretation['likely_cause'] = 'Complex structural change detected by neural network'
            interpretation['physical_meaning'] = 'Subsurface velocity structure modification'
            interpretation['recommended_action'] = 'Detailed geophysical investigation'
            
        elif max_component == 'temporal':
            interpretation['likely_cause'] = 'Temporal pattern disruption in seismic velocity'
            interpretation['physical_meaning'] = 'Possible stress accumulation or fluid migration'
            interpretation['recommended_action'] = 'Monitor temporal evolution and correlate with other geophysical signals'
            
        elif max_component == 'spectral':
            interpretation['likely_cause'] = 'Frequency-dependent velocity changes'
            interpretation['physical_meaning'] = 'Selective frequency attenuation or scattering changes'
            interpretation['recommended_action'] = 'Frequency-domain analysis and attenuation studies'
            
        elif max_component == 'pattern':
            interpretation['likely_cause'] = 'Cross-correlation pattern modification'
            interpretation['physical_meaning'] = 'Changes in wave propagation paths or scattering'
            interpretation['recommended_action'] = 'Spatial coherence analysis and array processing'
            
        else:  # multiscale
            interpretation['likely_cause'] = 'Multi-scale structural heterogeneity changes'
            interpretation['physical_meaning'] = 'Hierarchical medium property modifications'
            interpretation['recommended_action'] = 'Multi-scale modeling and tomographic imaging'
        
        # Add confidence in interpretation
        confidence = fused_results['confidence']
        if confidence > 0.8:
            interpretation['interpretation_confidence'] = 'high'
        elif confidence > 0.6:
            interpretation['interpretation_confidence'] = 'moderate'
        else:
            interpretation['interpretation_confidence'] = 'low'
        
        return interpretation
    
    def _assess_data_quality(self, sequence_data: torch.Tensor) -> Dict:
        """Assess quality of input data for ML analysis"""
        data = sequence_data.squeeze().numpy()
        
        quality_metrics = {}
        
        # Data completeness
        total_samples = data.size
        nan_count = np.isnan(data).sum()
        inf_count = np.isinf(data).sum()
        quality_metrics['completeness'] = 1.0 - (nan_count + inf_count) / total_samples
        
        # Signal-to-noise ratio estimation
        signal_power = np.var(data)
        noise_power = np.var(data - signal.medfilt(data.flatten(), 5).reshape(data.shape))
        if noise_power > 0:
            quality_metrics['snr_estimate'] = 10 * np.log10(signal_power / noise_power)
        else:
            quality_metrics['snr_estimate'] = float('inf')
        
        # Dynamic range
        data_range = np.max(data) - np.min(data)
        quality_metrics['dynamic_range'] = data_range
        
        # Overall quality score
        completeness_score = quality_metrics['completeness']
        snr_score = min(quality_metrics['snr_estimate'] / 20.0, 1.0)  # Normalize to [0,1]
        range_score = min(data_range / 1000.0, 1.0)  # Normalize assuming max range ~1000
        
        quality_metrics['overall_quality'] = (completeness_score + snr_score + range_score) / 3.0
        
        return quality_metrics
    
    def _assess_model_performance(self) -> Dict:
        """Assess current model performance metrics"""
        # In a real implementation, this would track model performance over time
        return {
            'model_confidence': 0.8,  # Placeholder
            'recent_accuracy': 0.85,  # Placeholder
            'calibration_score': 0.9,  # Placeholder
            'last_training_date': '2024-01-01',  # Placeholder
            'model_version': '1.0'  # Placeholder
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