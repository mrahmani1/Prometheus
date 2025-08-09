"""
Advanced Array Seismology with Machine Learning Enhancement
ML-enhanced beamforming, automatic array response optimization,
slowness vector estimation, and source discrimination
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from scipy import linalg
from scipy.optimize import minimize

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.fft import fft2, ifft2, fftshift
    from scipy.spatial.distance import pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class ArrayConfig:
    """Configuration for array processing"""
    sample_rate: float = 50.0
    frequency_range: Tuple[float, float] = (1.0, 10.0)
    slowness_range: Tuple[float, float] = (0.0, 0.3)  # s/km
    azimuth_range: Tuple[float, float] = (0.0, 360.0)  # degrees
    grid_points: int = 101
    beam_window: float = 10.0  # seconds
    overlap_ratio: float = 0.5
    coherence_threshold: float = 0.3
    ml_enhancement: bool = True
    adaptive_weights: bool = True


@dataclass
class ArrayGeometry:
    """Array station geometry"""
    station_coords: np.ndarray  # (n_stations, 2) - [x, y] in km
    station_names: List[str]
    reference_coord: Tuple[float, float] = (0.0, 0.0)
    elevation_coords: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.elevation_coords is None:
            self.elevation_coords = np.zeros(len(self.station_names))


class MLBeamformer(nn.Module):
    """Machine Learning enhanced beamformer"""
    
    def __init__(self, n_stations: int, frequency_bins: int):
        super().__init__()
        self.n_stations = n_stations
        self.frequency_bins = frequency_bins
        
        # Adaptive weight networks
        self.spatial_weights = nn.Sequential(
            nn.Linear(n_stations * 2, 128),  # x,y coordinates
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_stations),
            nn.Softmax(dim=-1)
        )
        
        self.frequency_weights = nn.Sequential(
            nn.Linear(frequency_bins, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, frequency_bins),
            nn.Sigmoid()
        )
        
        # Coherence enhancement network
        self.coherence_enhancer = nn.Sequential(
            nn.Conv1d(n_stations, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, n_stations, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Source discrimination network
        self.discriminator = nn.Sequential(
            nn.Linear(n_stations + 3, 256),  # +3 for slowness vector
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # Local, Regional, Teleseismic, Noise
        )
    
    def forward(self, data: torch.Tensor, geometry: torch.Tensor, 
                slowness_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ML beamformer
        
        Args:
            data: (batch, stations, time)
            geometry: (stations, 2) - station coordinates
            slowness_vector: (batch, 3) - [sx, sy, magnitude]
        """
        batch_size, n_stations, n_time = data.shape
        
        # Generate spatial weights
        geometry_flat = geometry.flatten().expand(batch_size, -1)
        spatial_w = self.spatial_weights(geometry_flat)
        
        # Generate frequency weights (simplified for time domain)
        freq_features = torch.mean(torch.abs(torch.fft.fft(data, dim=-1))[:, :, :64], dim=1)
        frequency_w = self.frequency_weights(freq_features)
        
        # Enhance coherence
        coherence_w = self.coherence_enhancer(data)
        
        # Apply weights
        enhanced_data = data * spatial_w.unsqueeze(-1) * coherence_w
        
        # Source discrimination
        beam_power = torch.mean(enhanced_data**2, dim=-1)  # (batch, stations)
        discriminator_input = torch.cat([beam_power, slowness_vector], dim=-1)
        source_type = self.discriminator(discriminator_input)
        
        return {
            'enhanced_data': enhanced_data,
            'spatial_weights': spatial_w,
            'frequency_weights': frequency_w,
            'coherence_weights': coherence_w,
            'source_classification': F.softmax(source_type, dim=-1)
        }


class SlownessEstimator(nn.Module):
    """Neural network for slowness vector estimation"""
    
    def __init__(self, n_stations: int):
        super().__init__()
        self.n_stations = n_stations
        
        # Feature extraction from cross-correlation delays
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_stations * (n_stations - 1) // 2, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Slowness prediction heads
        self.sx_head = nn.Linear(64, 1)
        self.sy_head = nn.Linear(64, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # sx, sy uncertainties
            nn.Softplus()
        )
    
    def forward(self, correlation_delays: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict slowness vector from correlation delays
        
        Args:
            correlation_delays: (batch, n_pairs) - time delays between station pairs
        """
        features = self.feature_extractor(correlation_delays)
        
        sx = self.sx_head(features).squeeze(-1)
        sy = self.sy_head(features).squeeze(-1)
        uncertainties = self.uncertainty_head(features)
        
        return {
            'slowness_x': sx,
            'slowness_y': sy,
            'uncertainties': uncertainties
        }


class ArrayProcessor:
    """Advanced array processing with ML enhancement"""
    
    def __init__(self, geometry: ArrayGeometry, config: Optional[ArrayConfig] = None):
        self.geometry = geometry
        self.config = config or ArrayConfig()
        self.n_stations = len(geometry.station_names)
        
        # ML models
        self.ml_models = {}
        if TORCH_AVAILABLE and self.config.ml_enhancement:
            self._initialize_ml_models()
        
        # Processing parameters
        self._setup_processing_grids()
        
        # Array response function cache
        self._arf_cache = {}
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        frequency_bins = int(self.config.sample_rate * self.config.beam_window / 2)
        
        self.ml_models = {
            'beamformer': MLBeamformer(self.n_stations, frequency_bins),
            'slowness_estimator': SlownessEstimator(self.n_stations)
        }
    
    def _setup_processing_grids(self):
        """Setup slowness and azimuth grids for processing"""
        # Slowness grid
        s_min, s_max = self.config.slowness_range
        self.slowness_grid = np.linspace(s_min, s_max, self.config.grid_points)
        
        # Azimuth grid
        az_min, az_max = self.config.azimuth_range
        self.azimuth_grid = np.linspace(az_min, az_max, self.config.grid_points)
        
        # Create 2D grid
        self.slowness_x_grid, self.slowness_y_grid = self._create_slowness_grid()
    
    def _create_slowness_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D slowness grid in Cartesian coordinates"""
        sx_grid = []
        sy_grid = []
        
        for s in self.slowness_grid:
            for az in self.azimuth_grid:
                az_rad = np.radians(az)
                sx = s * np.sin(az_rad)
                sy = s * np.cos(az_rad)
                sx_grid.append(sx)
                sy_grid.append(sy)
        
        return np.array(sx_grid), np.array(sy_grid)
    
    def calculate_array_response(self, frequency: float, 
                               slowness_x: float, slowness_y: float) -> complex:
        """Calculate array response function"""
        cache_key = (frequency, slowness_x, slowness_y)
        if cache_key in self._arf_cache:
            return self._arf_cache[cache_key]
        
        coords = self.geometry.station_coords
        omega = 2 * np.pi * frequency
        
        # Phase delays for each station
        phases = omega * (slowness_x * coords[:, 0] + slowness_y * coords[:, 1])
        
        # Array response (normalized)
        response = np.mean(np.exp(1j * phases))
        
        self._arf_cache[cache_key] = response
        return response
    
    def conventional_beamforming(self, data: np.ndarray, 
                               frequency_band: Optional[Tuple[float, float]] = None) -> Dict:
        """Conventional delay-and-sum beamforming"""
        if frequency_band is None:
            frequency_band = self.config.frequency_range
        
        # Filter data
        if SCIPY_AVAILABLE:
            filtered_data = self._bandpass_filter(data, frequency_band)
        else:
            filtered_data = data
        
        n_samples = filtered_data.shape[1]
        beam_power = np.zeros((len(self.slowness_x_grid),))
        
        coords = self.geometry.station_coords
        dt = 1.0 / self.config.sample_rate
        
        for i, (sx, sy) in enumerate(zip(self.slowness_x_grid, self.slowness_y_grid)):
            # Calculate time delays
            delays = sx * coords[:, 0] + sy * coords[:, 1]  # in seconds
            delay_samples = np.round(delays / dt).astype(int)
            
            # Apply delays and sum
            beam_trace = np.zeros(n_samples)
            valid_stations = 0
            
            for j, delay in enumerate(delay_samples):
                if 0 <= delay < n_samples:
                    shifted_trace = np.roll(filtered_data[j], -delay)
                    beam_trace += shifted_trace
                    valid_stations += 1
            
            if valid_stations > 0:
                beam_trace /= valid_stations
                beam_power[i] = np.mean(beam_trace**2)
        
        # Find maximum power
        max_idx = np.argmax(beam_power)
        best_sx = self.slowness_x_grid[max_idx]
        best_sy = self.slowness_y_grid[max_idx]
        best_slowness = np.sqrt(best_sx**2 + best_sy**2)
        best_azimuth = np.degrees(np.arctan2(best_sx, best_sy))
        if best_azimuth < 0:
            best_azimuth += 360
        
        return {
            'beam_power': beam_power.reshape(self.config.grid_points, self.config.grid_points),
            'best_slowness': best_slowness,
            'best_azimuth': best_azimuth,
            'slowness_x': best_sx,
            'slowness_y': best_sy,
            'max_power': beam_power[max_idx],
            'coherence': self._calculate_coherence(filtered_data, best_sx, best_sy)
        }
    
    def ml_enhanced_beamforming(self, data: np.ndarray) -> Dict:
        """ML-enhanced beamforming with adaptive weights"""
        if not TORCH_AVAILABLE or 'beamformer' not in self.ml_models:
            warnings.warn("ML models not available - falling back to conventional beamforming")
            return self.conventional_beamforming(data)
        
        model = self.ml_models['beamformer']
        model.eval()
        
        # Convert to torch tensors
        data_tensor = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
        geometry_tensor = torch.FloatTensor(self.geometry.station_coords)
        
        # Initial slowness estimate using conventional method
        conv_result = self.conventional_beamforming(data)
        slowness_vector = torch.FloatTensor([[
            conv_result['slowness_x'],
            conv_result['slowness_y'], 
            conv_result['best_slowness']
        ]])
        
        with torch.no_grad():
            ml_result = model(data_tensor, geometry_tensor, slowness_vector)
        
        # Enhanced processing
        enhanced_data = ml_result['enhanced_data'].numpy()[0]
        source_probs = ml_result['source_classification'].numpy()[0]
        
        # Re-run beamforming on enhanced data
        enhanced_beam = self.conventional_beamforming(enhanced_data)
        
        source_types = ['local', 'regional', 'teleseismic', 'noise']
        best_source_type = source_types[np.argmax(source_probs)]
        
        return {
            **enhanced_beam,
            'enhanced': True,
            'source_classification': {
                source_type: float(prob) for source_type, prob 
                in zip(source_types, source_probs)
            },
            'most_likely_source': best_source_type,
            'spatial_weights': ml_result['spatial_weights'].numpy()[0],
            'coherence_weights': ml_result['coherence_weights'].numpy()[0]
        }
    
    def estimate_slowness_ml(self, data: np.ndarray) -> Dict:
        """Estimate slowness vector using machine learning"""
        if not TORCH_AVAILABLE or 'slowness_estimator' not in self.ml_models:
            warnings.warn("ML slowness estimator not available")
            return {'slowness_x': 0.0, 'slowness_y': 0.0, 'uncertainties': [0.1, 0.1]}
        
        # Calculate cross-correlation delays
        correlation_delays = self._calculate_correlation_delays(data)
        
        model = self.ml_models['slowness_estimator']
        model.eval()
        
        with torch.no_grad():
            delays_tensor = torch.FloatTensor(correlation_delays).unsqueeze(0)
            prediction = model(delays_tensor)
        
        return {
            'slowness_x': float(prediction['slowness_x'].numpy()[0]),
            'slowness_y': float(prediction['slowness_y'].numpy()[0]),
            'uncertainties': prediction['uncertainties'].numpy()[0].tolist()
        }
    
    def _calculate_correlation_delays(self, data: np.ndarray) -> np.ndarray:
        """Calculate cross-correlation time delays between station pairs"""
        n_stations = data.shape[0]
        delays = []
        
        for i in range(n_stations):
            for j in range(i + 1, n_stations):
                # Cross-correlation
                correlation = np.correlate(data[i], data[j], mode='full')
                
                # Find peak
                peak_idx = np.argmax(correlation)
                delay = (peak_idx - len(data[j]) + 1) / self.config.sample_rate
                delays.append(delay)
        
        return np.array(delays)
    
    def adaptive_beamforming(self, data: np.ndarray) -> Dict:
        """Adaptive beamforming using minimum variance distortionless response (MVDR)"""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available - using conventional beamforming")
            return self.conventional_beamforming(data)
        
        # Filter data
        filtered_data = self._bandpass_filter(data, self.config.frequency_range)
        
        # Estimate covariance matrix
        cov_matrix = np.cov(filtered_data)
        
        # Regularization to handle singular matrices
        reg_param = 0.01 * np.trace(cov_matrix) / self.n_stations
        cov_matrix += reg_param * np.eye(self.n_stations)
        
        best_power = -np.inf
        best_result = None
        
        coords = self.geometry.station_coords
        
        for sx, sy in zip(self.slowness_x_grid[::10], self.slowness_y_grid[::10]):  # Subsample for efficiency
            # Steering vector
            omega = 2 * np.pi * np.mean(self.config.frequency_range)
            phases = omega * (sx * coords[:, 0] + sy * coords[:, 1])
            steering_vector = np.exp(1j * phases)
            
            try:
                # MVDR weights
                inv_cov = linalg.inv(cov_matrix)
                weights = inv_cov @ steering_vector
                weights = weights / (steering_vector.conj().T @ weights)
                
                # Beam power
                power = np.real(weights.conj().T @ cov_matrix @ weights)
                
                if power > best_power:
                    best_power = power
                    best_slowness = np.sqrt(sx**2 + sy**2)
                    best_azimuth = np.degrees(np.arctan2(sx, sy))
                    if best_azimuth < 0:
                        best_azimuth += 360
                    
                    best_result = {
                        'best_slowness': best_slowness,
                        'best_azimuth': best_azimuth,
                        'slowness_x': sx,
                        'slowness_y': sy,
                        'max_power': power,
                        'weights': weights,
                        'adaptive': True
                    }
            
            except linalg.LinAlgError:
                continue
        
        if best_result is None:
            warnings.warn("Adaptive beamforming failed - using conventional method")
            return self.conventional_beamforming(data)
        
        return best_result
    
    def frequency_wavenumber_analysis(self, data: np.ndarray) -> Dict:
        """Frequency-wavenumber (f-k) analysis"""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available - skipping f-k analysis")
            return {}
        
        # Apply temporal window
        windowed_data = data * np.hanning(data.shape[1])
        
        # Spatial FFT (simplified 1D approach)
        # In practice, would need proper 2D spatial gridding
        
        # Calculate average inter-station spacing
        coords = self.geometry.station_coords
        distances = pdist(coords)
        avg_spacing = np.mean(distances)
        
        # Frequency analysis
        frequencies = np.fft.fftfreq(data.shape[1], 1/self.config.sample_rate)
        freq_mask = (frequencies >= self.config.frequency_range[0]) & \
                   (frequencies <= self.config.frequency_range[1])
        
        fk_power = np.zeros((len(self.slowness_grid), np.sum(freq_mask)))
        
        for i, slowness in enumerate(self.slowness_grid):
            wavenumber = 2 * np.pi * slowness
            
            for j, freq in enumerate(frequencies[freq_mask]):
                if freq > 0:  # Skip DC component
                    # Calculate f-k power (simplified)
                    phase_shifts = np.exp(1j * wavenumber * 
                                        np.linalg.norm(coords, axis=1))
                    
                    # Apply phase shifts in frequency domain
                    fft_data = np.fft.fft(windowed_data, axis=1)
                    freq_idx = np.where(frequencies == freq)[0][0]
                    
                    if freq_idx < fft_data.shape[1]:
                        weighted_spectrum = fft_data[:, freq_idx] * phase_shifts
                        fk_power[i, j] = np.abs(np.mean(weighted_spectrum))**2
        
        # Find peak
        max_indices = np.unravel_index(np.argmax(fk_power), fk_power.shape)
        best_slowness = self.slowness_grid[max_indices[0]]
        best_frequency = frequencies[freq_mask][max_indices[1]]
        
        return {
            'fk_power': fk_power,
            'best_slowness': best_slowness,
            'best_frequency': best_frequency,
            'frequency_grid': frequencies[freq_mask],
            'slowness_grid': self.slowness_grid
        }
    
    def source_discrimination(self, beam_result: Dict) -> Dict:
        """Discriminate between different seismic source types"""
        features = {
            'slowness': beam_result.get('best_slowness', 0.0),
            'coherence': beam_result.get('coherence', 0.0),
            'power': beam_result.get('max_power', 0.0),
            'frequency_content': self._analyze_frequency_content(beam_result)
        }
        
        # Rule-based classification
        if features['slowness'] < 0.05:  # s/km
            source_type = 'local'
            confidence = 0.8
        elif features['slowness'] < 0.15:
            source_type = 'regional'
            confidence = 0.7
        elif features['slowness'] < 0.25:
            source_type = 'teleseismic'
            confidence = 0.6
        else:
            source_type = 'noise'
            confidence = 0.5
        
        # Adjust confidence based on coherence
        if features['coherence'] > 0.7:
            confidence = min(1.0, confidence + 0.2)
        elif features['coherence'] < 0.3:
            confidence = max(0.0, confidence - 0.3)
            source_type = 'noise'
        
        return {
            'source_type': source_type,
            'confidence': confidence,
            'features': features
        }
    
    def _analyze_frequency_content(self, beam_result: Dict) -> Dict:
        """Analyze frequency content of beam result"""
        # Placeholder for frequency analysis
        return {
            'dominant_frequency': 5.0,
            'bandwidth': 3.0,
            'spectral_ratio': 0.5
        }
    
    def _bandpass_filter(self, data: np.ndarray, 
                        frequency_band: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter to data"""
        if not SCIPY_AVAILABLE:
            warnings.warn("SciPy not available - skipping filtering")
            return data
        
        low, high = frequency_band
        nyquist = self.config.sample_rate / 2
        
        if high >= nyquist:
            high = nyquist * 0.99
        
        b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
        
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = filtfilt(b, a, data[i])
        
        return filtered_data
    
    def _calculate_coherence(self, data: np.ndarray, 
                           slowness_x: float, slowness_y: float) -> float:
        """Calculate beam coherence"""
        coords = self.geometry.station_coords
        dt = 1.0 / self.config.sample_rate
        
        # Calculate time delays
        delays = slowness_x * coords[:, 0] + slowness_y * coords[:, 1]
        delay_samples = np.round(delays / dt).astype(int)
        
        # Apply delays
        aligned_traces = []
        for i, delay in enumerate(delay_samples):
            if 0 <= delay < data.shape[1]:
                shifted_trace = np.roll(data[i], -delay)
                aligned_traces.append(shifted_trace)
        
        if len(aligned_traces) < 2:
            return 0.0
        
        aligned_traces = np.array(aligned_traces)
        
        # Calculate coherence as normalized cross-correlation
        mean_trace = np.mean(aligned_traces, axis=0)
        correlations = []
        
        for trace in aligned_traces:
            correlation = np.corrcoef(trace, mean_trace)[0, 1]
            if not np.isnan(correlation):
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.0
    
    def process_continuous_data(self, data_stream: np.ndarray, 
                              method: str = 'ml_enhanced') -> Dict:
        """Process continuous data stream with specified method"""
        window_size = int(self.config.beam_window * self.config.sample_rate)
        overlap = int(window_size * self.config.overlap_ratio)
        step_size = window_size - overlap
        
        results = []
        processing_methods = {
            'conventional': self.conventional_beamforming,
            'ml_enhanced': self.ml_enhanced_beamforming,
            'adaptive': self.adaptive_beamforming
        }
        
        if method not in processing_methods:
            method = 'conventional'
            warnings.warn(f"Unknown method, using {method}")
        
        process_func = processing_methods[method]
        
        for i in range(0, data_stream.shape[1] - window_size + 1, step_size):
            window_data = data_stream[:, i:i + window_size]
            
            # Process window
            beam_result = process_func(window_data)
            
            # Source discrimination
            source_result = self.source_discrimination(beam_result)
            
            # Add timing information
            result = {
                'start_time': i / self.config.sample_rate,
                'end_time': (i + window_size) / self.config.sample_rate,
                'beam_result': beam_result,
                'source_discrimination': source_result
            }
            
            results.append(result)
        
        return {
            'window_results': results,
            'processing_summary': self._summarize_results(results),
            'method_used': method
        }
    
    def _summarize_results(self, results: List[Dict]) -> Dict:
        """Summarize processing results"""
        if not results:
            return {}
        
        slownesses = [r['beam_result'].get('best_slowness', 0) for r in results]
        azimuths = [r['beam_result'].get('best_azimuth', 0) for r in results]
        coherences = [r['beam_result'].get('coherence', 0) for r in results]
        
        source_types = [r['source_discrimination']['source_type'] for r in results]
        source_counts = {stype: source_types.count(stype) for stype in set(source_types)}
        
        return {
            'total_windows': len(results),
            'average_slowness': np.mean(slownesses),
            'average_azimuth': np.mean(azimuths),
            'average_coherence': np.mean(coherences),
            'source_type_distribution': source_counts,
            'high_coherence_detections': sum(1 for c in coherences if c > 0.7)
        }
    
    def export_results(self, results: Dict, filename: str):
        """Export array processing results"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'array_geometry': {
                'station_names': self.geometry.station_names,
                'coordinates': self.geometry.station_coords.tolist(),
                'reference_coord': self.geometry.reference_coord
            },
            'config': {
                'sample_rate': self.config.sample_rate,
                'frequency_range': self.config.frequency_range,
                'slowness_range': self.config.slowness_range,
                'method_used': results.get('method_used', 'unknown')
            },
            'results': results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to export results: {e}")


# Example usage and testing
def main():
    """Example usage of array processor"""
    
    # Create example array geometry
    station_coords = np.array([
        [0.0, 0.0],     # Reference station
        [1.0, 0.0],     # East
        [0.0, 1.0],     # North
        [-1.0, 0.0],    # West
        [0.0, -1.0],    # South
        [0.5, 0.5],     # Northeast
        [-0.5, 0.5],    # Northwest
        [-0.5, -0.5],   # Southwest
        [0.5, -0.5]     # Southeast
    ])
    
    station_names = [f'STA{i:02d}' for i in range(len(station_coords))]
    
    geometry = ArrayGeometry(
        station_coords=station_coords,
        station_names=station_names
    )
    
    # Initialize processor
    config = ArrayConfig(
        sample_rate=50.0,
        frequency_range=(1.0, 8.0),
        beam_window=20.0,
        ml_enhancement=True
    )
    
    processor = ArrayProcessor(geometry, config)
    
    # Generate synthetic array data
    duration = 60.0  # seconds
    n_samples = int(duration * config.sample_rate)
    time = np.linspace(0, duration, n_samples)
    
    # Simulate plane wave arrival
    true_slowness = 0.1  # s/km
    true_azimuth = 45.0  # degrees
    true_frequency = 3.0  # Hz
    
    # Convert to slowness vector
    true_sx = true_slowness * np.sin(np.radians(true_azimuth))
    true_sy = true_slowness * np.cos(np.radians(true_azimuth))
    
    # Generate data for each station
    array_data = np.zeros((len(station_coords), n_samples))
    
    for i, coord in enumerate(station_coords):
        # Time delay for this station
        delay = true_sx * coord[0] + true_sy * coord[1]  # seconds
        
        # Phase shift
        phase = 2 * np.pi * true_frequency * (time - delay)
        
        # Signal with noise
        signal = np.sin(phase) * np.exp(-(time - 30.0)**2 / 100.0)  # Gaussian envelope
        noise = 0.2 * np.random.randn(n_samples)
        
        array_data[i] = signal + noise
    
    print(f"Processing array data...")
    print(f"True parameters: slowness={true_slowness:.3f} s/km, azimuth={true_azimuth:.1f}°")
    
    # Test different processing methods
    methods = ['conventional', 'adaptive']
    if TORCH_AVAILABLE:
        methods.append('ml_enhanced')
    
    for method in methods:
        print(f"\n--- {method.upper()} BEAMFORMING ---")
        
        results = processor.process_continuous_data(array_data, method=method)
        summary = results['processing_summary']
        
        print(f"Processed {summary['total_windows']} windows")
        print(f"Average slowness: {summary['average_slowness']:.4f} s/km")
        print(f"Average azimuth: {summary['average_azimuth']:.1f}°") 
        print(f"Average coherence: {summary['average_coherence']:.3f}")
        print(f"Source types: {summary['source_type_distribution']}")
        
        # Export results
        processor.export_results(results, f'array_results_{method}.json')
    
    # Test f-k analysis
    print("\n--- F-K ANALYSIS ---")
    fk_results = processor.frequency_wavenumber_analysis(array_data)
    if fk_results:
        print(f"F-K best slowness: {fk_results['best_slowness']:.4f} s/km")
        print(f"F-K best frequency: {fk_results['best_frequency']:.2f} Hz")
    
    # Test ML slowness estimation
    if TORCH_AVAILABLE:
        print("\n--- ML SLOWNESS ESTIMATION ---")
        ml_slowness = processor.estimate_slowness_ml(array_data)
        print(f"ML slowness: sx={ml_slowness['slowness_x']:.4f}, sy={ml_slowness['slowness_y']:.4f}")
        print(f"ML uncertainties: {ml_slowness['uncertainties']}")
    
    return processor, results


if __name__ == "__main__":
    processor, results = main()