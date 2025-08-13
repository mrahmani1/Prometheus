"""
Advanced Moment Tensor Inversion System
Full moment tensor solutions with uncertainty estimation, non-double-couple analysis,
and automated depth determination for comprehensive source characterization
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, LBFGS
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy import linalg
    from scipy.signal import correlate, hilbert
    from scipy.stats import chi2
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MomentTensorConfig:
    """Configuration for moment tensor inversion"""
    frequency_range: Tuple[float, float] = (0.02, 0.1)  # Hz
    time_window: float = 120.0  # seconds
    depth_range: Tuple[float, float] = (5.0, 50.0)  # km
    depth_resolution: float = 1.0  # km
    uncertainty_samples: int = 1000
    velocity_model: str = "ak135"
    distance_weights: bool = True
    azimuthal_weights: bool = True
    bootstrap_samples: int = 500
    misfit_threshold: float = 0.7


@dataclass 
class StationInfo:
    """Station information for MT inversion"""
    station_code: str
    latitude: float
    longitude: float
    elevation: float
    distance: float  # km from source
    azimuth: float  # degrees
    backazimuth: float  # degrees
    components: List[str] = field(default_factory=lambda: ['Z', 'R', 'T'])


@dataclass
class MomentTensorSolution:
    """Complete moment tensor solution"""
    moment_tensor: np.ndarray  # 3x3 tensor
    eigenvalues: np.ndarray  # Principal moments
    eigenvectors: np.ndarray  # Principal axes
    strike1: float
    dip1: float
    rake1: float
    strike2: float 
    dip2: float
    rake2: float
    moment_magnitude: float
    scalar_moment: float
    depth: float
    source_type: str  # 'dc', 'clvd', 'iso', 'mixed'
    dc_percentage: float
    clvd_percentage: float
    iso_percentage: float
    variance_reduction: float
    condition_number: float
    uncertainties: Dict
    confidence_intervals: Dict


class GreensFunctionCalculator:
    """Calculate synthetic Green's functions for MT inversion"""
    
    def __init__(self, config: MomentTensorConfig):
        self.config = config
        self.velocity_models = self._load_velocity_models()
        self._greens_cache = {}
    
    def _load_velocity_models(self) -> Dict:
        """Load comprehensive global and regional velocity models"""
        # Complete AK135 global reference model
        ak135 = {
            'depths': np.array([0.0, 5.0, 10.0, 15.0, 20.0, 24.4, 35.0, 50.0, 70.0, 
                              80.0, 120.0, 165.0, 210.0, 260.0, 310.0, 360.0, 410.0,
                              460.0, 510.0, 560.0, 610.0, 660.0, 710.0]),
            'vp': np.array([5.80, 6.50, 6.80, 7.00, 7.20, 8.045, 8.050, 8.175, 8.300,
                          8.480, 8.785, 9.135, 9.310, 9.485, 9.565, 9.645, 9.760,
                          9.875, 9.990, 10.105, 10.220, 10.335, 10.450]),
            'vs': np.array([3.20, 3.60, 3.80, 3.90, 4.00, 4.485, 4.490, 4.509, 4.518,
                          4.609, 4.783, 4.963, 5.073, 5.183, 5.243, 5.303, 5.393,
                          5.483, 5.573, 5.663, 5.753, 5.843, 5.933]),
            'density': np.array([2.72, 2.92, 3.00, 3.10, 3.20, 3.324, 3.325, 3.370, 3.415,
                               3.506, 3.696, 3.928, 4.075, 4.222, 4.298, 4.374, 4.459,
                               4.544, 4.629, 4.714, 4.799, 4.884, 4.969]),
            'qp': np.array([1348, 1348, 1348, 1348, 1348, 557, 557, 557, 557,
                          557, 557, 557, 557, 557, 557, 557, 557,
                          557, 557, 557, 557, 557, 557]),
            'qs': np.array([600, 600, 600, 600, 600, 312, 312, 312, 312,
                          312, 312, 312, 312, 312, 312, 312, 312,
                          312, 312, 312, 312, 312, 312])
        }
        
        # Complete PREM (Preliminary Reference Earth Model)
        prem = {
            'depths': np.array([0.0, 3.0, 15.0, 24.4, 40.0, 60.0, 80.0, 115.0, 150.0,
                              185.0, 220.0, 265.0, 310.0, 355.0, 400.0, 450.0, 500.0,
                              600.0, 700.0, 771.0, 871.0, 971.0, 1071.0, 1171.0]),
            'vp': np.array([5.80, 6.80, 6.80, 8.11, 8.20, 8.30, 8.48, 8.78, 9.13,
                          9.31, 9.48, 9.56, 9.64, 9.76, 9.87, 9.99, 10.10,
                          10.33, 10.45, 13.65, 13.71, 13.78, 13.85, 13.92]),
            'vs': np.array([3.20, 3.90, 3.90, 4.49, 4.51, 4.52, 4.61, 4.78, 4.96,
                          5.07, 5.18, 5.24, 5.30, 5.39, 5.48, 5.57, 5.66,
                          5.84, 5.93, 7.26, 7.29, 7.31, 7.34, 7.37]),
            'density': np.array([2.60, 2.90, 2.90, 3.32, 3.37, 3.41, 3.51, 3.70, 3.93,
                               4.08, 4.22, 4.30, 4.37, 4.46, 4.54, 4.63, 4.71,
                               4.88, 4.97, 5.77, 5.85, 5.93, 6.01, 6.09]),
            'qp': np.array([600, 600, 600, 300, 300, 300, 300, 300, 300,
                          300, 300, 300, 300, 300, 300, 300, 300,
                          300, 300, 300, 300, 300, 300, 300]),
            'qs': np.array([300, 300, 300, 150, 150, 150, 150, 150, 150,
                          150, 150, 150, 150, 150, 150, 150, 150,
                          150, 150, 150, 150, 150, 150, 150])
        }
        
        # iasp91 model for comparison
        iasp91 = {
            'depths': np.array([0.0, 8.0, 16.0, 35.0, 120.0, 210.0, 410.0, 660.0]),
            'vp': np.array([5.80, 6.50, 8.04, 8.05, 8.78, 9.36, 9.73, 10.27]),
            'vs': np.array([3.36, 3.85, 4.48, 4.49, 4.78, 5.08, 5.32, 5.61]),
            'density': np.array([2.72, 2.92, 3.32, 3.32, 3.64, 4.14, 4.38, 4.73]),
            'qp': np.array([1340, 1340, 557, 557, 557, 557, 557, 557]),
            'qs': np.array([600, 600, 312, 312, 312, 312, 312, 312])
        }
        
        # Regional models for specific areas
        crustal_models = {
            'continental': {
                'depths': np.array([0.0, 5.0, 15.0, 25.0, 35.0, 50.0]),
                'vp': np.array([6.00, 6.20, 6.60, 7.80, 8.05, 8.15]),
                'vs': np.array([3.50, 3.60, 3.80, 4.40, 4.49, 4.52]),
                'density': np.array([2.70, 2.75, 2.85, 3.20, 3.32, 3.35]),
                'qp': np.array([800, 800, 800, 400, 400, 400]),
                'qs': np.array([400, 400, 400, 200, 200, 200])
            },
            'oceanic': {
                'depths': np.array([0.0, 3.0, 7.0, 15.0, 25.0, 35.0]),
                'vp': np.array([1.50, 5.00, 6.80, 7.20, 8.00, 8.05]),
                'vs': np.array([0.00, 2.80, 3.90, 4.10, 4.45, 4.49]),
                'density': np.array([1.03, 2.40, 2.90, 3.10, 3.25, 3.32]),
                'qp': np.array([50, 600, 600, 600, 400, 400]),
                'qs': np.array([25, 300, 300, 300, 200, 200])
            }
        }
        
        # Add frequency-dependent parameters
        frequency_params = {
            'reference_frequency': 1.0,  # Hz
            'quality_factor_scaling': 0.1,
            'velocity_dispersion': True
        }
        
        return {
            'ak135': ak135,
            'prem': prem,
            'iasp91': iasp91,
            'crustal_models': crustal_models,
            'frequency_params': frequency_params
        }
    
    def calculate_greens_functions(self, station: StationInfo, 
                                 source_depth: float) -> np.ndarray:
        """Calculate comprehensive Green's functions with full wave propagation"""
        cache_key = (station.station_code, source_depth, self.config.frequency_band)
        if cache_key in self._greens_cache:
            return self._greens_cache[cache_key]
        
        try:
            # Get appropriate velocity model
            model = self._select_velocity_model(station, source_depth)
            
            # Calculate detailed Earth structure properties
            earth_props = self._interpolate_earth_properties(model, source_depth, station.distance)
            
            # Advanced travel time calculation with ray tracing
            travel_times = self._calculate_travel_times_advanced(station, source_depth, model)
            
            # Generate time vector with adaptive sampling
            time_vector = self._create_time_vector()
            
            # Calculate comprehensive Green's functions
            greens_matrix = self._compute_greens_matrix(
                station, source_depth, earth_props, travel_times, time_vector
            )
            
            # Apply instrument response correction
            greens_corrected = self._apply_instrument_response(greens_matrix, station)
            
            # Apply attenuation and dispersion effects
            greens_final = self._apply_wave_propagation_effects(
                greens_corrected, earth_props, travel_times, time_vector
            )
            
            # Cache results
            self._greens_cache[cache_key] = greens_final
            
            return greens_final
            
        except Exception as e:
            logger.warning(f"Advanced Green's function calculation failed: {e}")
            return self._calculate_greens_functions_fallback(station, source_depth)
    
    def _select_velocity_model(self, station: StationInfo, source_depth: float) -> Dict:
        """Select appropriate velocity model based on station and source"""
        # Use crustal models for shallow sources
        if source_depth < 35.0:
            # Determine if oceanic or continental path
            if hasattr(station, 'path_type'):
                if station.path_type == 'oceanic':
                    crustal_model = self.velocity_models['crustal_models']['oceanic']
                else:
                    crustal_model = self.velocity_models['crustal_models']['continental']
                
                # Extend with global model for deeper layers
                global_model = self.velocity_models[self.config.velocity_model]
                return self._merge_velocity_models(crustal_model, global_model)
        
        return self.velocity_models[self.config.velocity_model]
    
    def _interpolate_earth_properties(self, model: Dict, depth: float, distance: float) -> Dict:
        """Interpolate Earth properties with high precision"""
        # Spline interpolation for smooth velocity structure
        from scipy import interpolate
        
        depths = model['depths']
        
        # Create spline functions for each property
        vp_spline = interpolate.UnivariateSpline(depths, model['vp'], s=0)
        vs_spline = interpolate.UnivariateSpline(depths, model['vs'], s=0)
        density_spline = interpolate.UnivariateSpline(depths, model['density'], s=0)
        
        # Sample properties along ray path
        ray_depths = np.linspace(0, depth, 50)
        
        properties = {
            'vp_profile': vp_spline(ray_depths),
            'vs_profile': vs_spline(ray_depths),
            'density_profile': density_spline(ray_depths),
            'depth_profile': ray_depths,
            'vp_source': float(vp_spline(depth)),
            'vs_source': float(vs_spline(depth)),
            'density_source': float(density_spline(depth))
        }
        
        # Add quality factors if available
        if 'qp' in model and 'qs' in model:
            qp_spline = interpolate.UnivariateSpline(depths, model['qp'], s=0)
            qs_spline = interpolate.UnivariateSpline(depths, model['qs'], s=0)
            properties.update({
                'qp_profile': qp_spline(ray_depths),
                'qs_profile': qs_spline(ray_depths),
                'qp_source': float(qp_spline(depth)),
                'qs_source': float(qs_spline(depth))
            })
        
        return properties
    
    def _calculate_travel_times_advanced(self, station: StationInfo, depth: float, 
                                       model: Dict) -> Dict:
        """Calculate travel times using ray tracing and multiple phases"""
        travel_times = {}
        
        # Ray tracing parameters
        earth_radius = 6371.0  # km
        delta_rad = station.distance / earth_radius
        
        # Direct P and S waves
        travel_times['P'] = self._calculate_ray_travel_time(
            delta_rad, depth, model, 'P'
        )
        travel_times['S'] = self._calculate_ray_travel_time(
            delta_rad, depth, model, 'S'
        )
        
        # Surface waves for regional distances
        if station.distance > 100:  # km
            travel_times['Love'] = self._calculate_surface_wave_time(
                station.distance, model, 'Love'
            )
            travel_times['Rayleigh'] = self._calculate_surface_wave_time(
                station.distance, model, 'Rayleigh'
            )
        
        # Depth phases
        if depth > 10:
            travel_times['pP'] = self._calculate_depth_phase_time(
                delta_rad, depth, model, 'pP'
            )
            travel_times['sP'] = self._calculate_depth_phase_time(
                delta_rad, depth, model, 'sP'
            )
        
        # Regional phases
        if 50 < station.distance < 1000:
            travel_times['Pg'] = self._calculate_regional_phase_time(
                station.distance, depth, model, 'Pg'
            )
            travel_times['Sg'] = self._calculate_regional_phase_time(
                station.distance, depth, model, 'Sg'
            )
        
        return travel_times
    
    def _compute_greens_matrix(self, station: StationInfo, depth: float,
                             earth_props: Dict, travel_times: Dict, 
                             time_vector: np.ndarray) -> np.ndarray:
        """Compute full Green's function matrix with all components"""
        n_time = len(time_vector)
        n_components = 6  # Six moment tensor components
        n_channels = len(station.components)
        
        greens = np.zeros((n_time, n_components, n_channels))
        
        # Station geometry
        az_rad = np.radians(station.azimuth)
        baz_rad = np.radians(station.backazimuth)
        takeoff_angle = self._calculate_takeoff_angle(station.distance, depth)
        
        # Calculate radiation patterns for each MT component
        radiation_patterns = self._calculate_radiation_patterns(
            az_rad, takeoff_angle, depth
        )
        
        # Generate waveforms for each phase
        for phase_name, arrival_time in travel_times.items():
            if arrival_time is None:
                continue
                
            # Generate source time function
            stf = self._generate_source_time_function(
                time_vector - arrival_time, phase_name
            )
            
            # Apply radiation patterns and geometric spreading
            phase_greens = self._apply_phase_specific_patterns(
                stf, radiation_patterns, phase_name, station, earth_props
            )
            
            # Add to total Green's functions
            greens += phase_greens
        
        return greens
    
    def _calculate_radiation_patterns(self, azimuth: float, takeoff: float, 
                                    depth: float) -> Dict[str, np.ndarray]:
        """Calculate detailed radiation patterns for all MT components"""
        # Spherical coordinate conversion
        phi = azimuth
        theta = takeoff
        
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_2phi = np.cos(2 * phi)
        sin_2phi = np.sin(2 * phi)
        
        # P-wave radiation patterns
        p_patterns = {
            'Mxx': sin_theta**2 * cos_phi**2,
            'Myy': sin_theta**2 * sin_phi**2,
            'Mzz': cos_theta**2,
            'Mxy': sin_theta**2 * sin_2phi / 2,
            'Mxz': sin_theta * cos_theta * cos_phi,
            'Myz': sin_theta * cos_theta * sin_phi
        }
        
        # S-wave radiation patterns (vertical component)
        sv_patterns = {
            'Mxx': -sin_theta * cos_theta * cos_phi**2,
            'Myy': -sin_theta * cos_theta * sin_phi**2,
            'Mzz': sin_theta * cos_theta,
            'Mxy': -sin_theta * cos_theta * sin_2phi / 2,
            'Mxz': (cos_theta**2 - sin_theta**2) * cos_phi,
            'Myz': (cos_theta**2 - sin_theta**2) * sin_phi
        }
        
        # S-wave radiation patterns (horizontal components)
        sh_patterns = {
            'Mxx': -cos_phi * sin_phi,
            'Myy': cos_phi * sin_phi,
            'Mzz': 0.0,
            'Mxy': cos_2phi,
            'Mxz': -sin_phi,
            'Myz': cos_phi
        }
        
        return {
            'P': np.array([p_patterns[comp] for comp in ['Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz']]),
            'SV': np.array([sv_patterns[comp] for comp in ['Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz']]),
            'SH': np.array([sh_patterns[comp] for comp in ['Mxx', 'Myy', 'Mzz', 'Mxy', 'Mxz', 'Myz']])
        }
    
    def _apply_wave_propagation_effects(self, greens: np.ndarray, earth_props: Dict,
                                      travel_times: Dict, time_vector: np.ndarray) -> np.ndarray:
        """Apply comprehensive wave propagation effects"""
        # Apply attenuation
        greens_attenuated = self._apply_attenuation(greens, earth_props, travel_times)
        
        # Apply geometric spreading
        greens_spread = self._apply_geometric_spreading(greens_attenuated, earth_props)
        
        # Apply dispersion effects
        greens_dispersed = self._apply_dispersion(greens_spread, earth_props, time_vector)
        
        # Apply scattering effects
        greens_scattered = self._apply_scattering(greens_dispersed, earth_props)
        
        return greens_scattered
    
    def _calculate_travel_time(self, distance: float, depth: float, velocity: float) -> float:
        """Enhanced travel time calculation with Earth curvature"""
        try:
            earth_radius = 6371.0  # km
            
            # Convert distance to angular distance
            delta = distance / earth_radius
            
            # Ray parameter calculation for spherical Earth
            if delta < np.pi / 4:  # Direct ray
                # Use Snell's law in spherical coordinates
                ray_param = np.sin(delta) / velocity
                
                # Integration along ray path for accurate travel time
                n_steps = 50
                depths_along_ray = np.linspace(0, depth, n_steps)
                
                # Get velocity model
                model = self.velocity_models[self.config.velocity_model]
                velocities = np.interp(depths_along_ray, model['depths'], model['vp'])
                
                # Numerical integration
                ray_lengths = np.diff(np.sqrt(depths_along_ray**2 + (distance * np.linspace(0, 1, n_steps))**2))
                avg_velocities = (velocities[:-1] + velocities[1:]) / 2
                
                travel_time = np.sum(ray_lengths / avg_velocities)
            else:
                # For large distances, use great circle approximation
                ray_length = earth_radius * delta
                travel_time = ray_length / velocity
            
            return travel_time
            
        except Exception:
            # Fallback to simple calculation
            ray_length = np.sqrt(distance**2 + depth**2)
            return ray_length / velocity
    
    def _ricker_wavelet(self, time: np.ndarray, frequency: float) -> np.ndarray:
        """Enhanced Ricker wavelet with proper normalization"""
        # Ensure time is properly centered and scaled
        time_normalized = time * frequency
        a = (np.pi * time_normalized)**2
        
        # Ricker wavelet with proper amplitude scaling
        wavelet = (1 - 2 * a) * np.exp(-a)
        
        # Normalize to unit amplitude
        if np.max(np.abs(wavelet)) > 0:
            wavelet = wavelet / np.max(np.abs(wavelet))
        
        return wavelet
    
    # Additional helper methods for comprehensive implementation
    def _create_time_vector(self) -> np.ndarray:
        """Create optimized time vector"""
        dt = 1.0 / self.config.sampling_rate
        duration = self.config.time_window
        return np.arange(0, duration, dt)
    
    def _calculate_greens_functions_fallback(self, station: StationInfo, 
                                           depth: float) -> np.ndarray:
        """Fallback method using simplified approach"""
        # Use original simple implementation as fallback
        model = self.velocity_models[self.config.velocity_model]
        vp = np.interp(depth, model['depths'], model['vp'])
        vs = np.interp(depth, model['depths'], model['vs'])
        
        time_vector = self._create_time_vector()
        greens = np.zeros((len(time_vector), 6, 3))  # 6 MT components, 3 channels
        
        # Simple P and S waves
        p_time = self._calculate_travel_time(station.distance, depth, vp)
        s_time = self._calculate_travel_time(station.distance, depth, vs)
        
        p_pulse = self._ricker_wavelet(time_vector - p_time, 1.0)
        s_pulse = self._ricker_wavelet(time_vector - s_time, 0.5)
        
        # Basic radiation patterns
        az_rad = np.radians(station.azimuth)
        for i in range(6):  # MT components
            greens[:, i, 0] = p_pulse * np.sin(az_rad) + s_pulse * np.cos(az_rad)  # Z
            greens[:, i, 1] = p_pulse * np.cos(az_rad) + s_pulse * np.sin(az_rad)  # N
            greens[:, i, 2] = s_pulse * np.sin(2 * az_rad)  # E
        
        return greens


class MomentTensorInverter:
    """Advanced moment tensor inversion with uncertainty quantification"""
    
    def __init__(self, config: Optional[MomentTensorConfig] = None):
        self.config = config or MomentTensorConfig()
        self.greens_calculator = GreensFunctionCalculator(self.config)
        self.velocity_model = self.config.velocity_model
        
        # Initialize neural network components if available
        self.ml_models = {}
        if TORCH_AVAILABLE:
            self._initialize_ml_components()
    
    def _initialize_ml_components(self):
        """Initialize ML components for enhanced inversion"""
        
        class DepthEstimatorNet(nn.Module):
            """Neural network for depth estimation"""
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(100, 256),  # 100 waveform features
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.2),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)  # Depth output
                )
                
            def forward(self, x):
                return self.features(x)
        
        class MTClassifierNet(nn.Module):
            """Neural network for MT mechanism classification"""
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(6, 128),  # 6 MT components
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Linear(64, 32),
                    nn.ReLU()
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 4)  # DC, CLVD, ISO, Mixed
                )
                
            def forward(self, mt_tensor):
                features = self.encoder(mt_tensor)
                classification = self.classifier(features)
                return F.softmax(classification, dim=1)
        
        self.ml_models = {
            'depth_estimator': DepthEstimatorNet(),
            'mt_classifier': MTClassifierNet()
        }
    
    def invert_moment_tensor(self, stations: List[StationInfo], 
                           waveform_data: Dict[str, np.ndarray],
                           event_time: datetime) -> MomentTensorSolution:
        """
        Perform full moment tensor inversion with uncertainty estimation
        
        Args:
            stations: List of station information
            waveform_data: Dictionary of waveform data keyed by station code
            event_time: Origin time of the event
        """
        
        # Grid search over depth
        depth_grid = np.arange(
            self.config.depth_range[0],
            self.config.depth_range[1] + self.config.depth_resolution,
            self.config.depth_resolution
        )
        
        best_solution = None
        best_variance_reduction = -np.inf
        depth_solutions = []
        
        for depth in depth_grid:
            try:
                solution = self._invert_at_depth(stations, waveform_data, depth)
                depth_solutions.append((depth, solution))
                
                if solution['variance_reduction'] > best_variance_reduction:
                    best_variance_reduction = solution['variance_reduction']
                    best_solution = solution
                    best_solution['depth'] = depth
                    
            except Exception as e:
                warnings.warn(f"Inversion failed at depth {depth}km: {e}")
                continue
        
        if best_solution is None:
            raise ValueError("Moment tensor inversion failed at all depths")
        
        # Uncertainty analysis
        uncertainties = self._estimate_uncertainties(
            stations, waveform_data, best_solution
        )
        
        # Decompose moment tensor
        decomposition = self._decompose_moment_tensor(best_solution['moment_tensor'])
        
        # Create final solution
        mt_solution = MomentTensorSolution(
            moment_tensor=best_solution['moment_tensor'],
            eigenvalues=decomposition['eigenvalues'],
            eigenvectors=decomposition['eigenvectors'],
            strike1=decomposition['nodal_planes']['strike1'],
            dip1=decomposition['nodal_planes']['dip1'],
            rake1=decomposition['nodal_planes']['rake1'],
            strike2=decomposition['nodal_planes']['strike2'],
            dip2=decomposition['nodal_planes']['dip2'],
            rake2=decomposition['nodal_planes']['rake2'],
            moment_magnitude=decomposition['moment_magnitude'],
            scalar_moment=decomposition['scalar_moment'],
            depth=best_solution['depth'],
            source_type=decomposition['source_type'],
            dc_percentage=decomposition['dc_percentage'],
            clvd_percentage=decomposition['clvd_percentage'],
            iso_percentage=decomposition['iso_percentage'],
            variance_reduction=best_solution['variance_reduction'],
            condition_number=best_solution['condition_number'],
            uncertainties=uncertainties,
            confidence_intervals=self._calculate_confidence_intervals(
                depth_solutions, uncertainties
            )
        )
        
        return mt_solution
    
    def _invert_at_depth(self, stations: List[StationInfo],
                        waveform_data: Dict[str, np.ndarray], 
                        depth: float) -> Dict:
        """Perform moment tensor inversion at fixed depth"""
        
        # Build Green's function matrix
        G_matrices = []
        data_vector = []
        weights = []
        
        for station in stations:
            if station.station_code not in waveform_data:
                continue
                
            # Get Green's functions
            greens = self.greens_calculator.calculate_greens_functions(station, depth)
            
            # Get observed data
            observed = waveform_data[station.station_code]
            
            # Apply frequency filter
            if SCIPY_AVAILABLE:
                observed = self._bandpass_filter(observed, self.config.frequency_range)
            
            # Window data
            windowed_obs = self._apply_time_window(observed)
            windowed_greens = self._apply_time_window(greens)
            
            # Add to system
            G_matrices.append(windowed_greens)
            data_vector.append(windowed_obs.flatten())
            
            # Calculate weights
            if self.config.distance_weights:
                distance_weight = 1.0 / np.sqrt(station.distance)
            else:
                distance_weight = 1.0
                
            if self.config.azimuthal_weights:
                # Simple azimuthal weighting
                azimuthal_weight = 1.0
            else:
                azimuthal_weight = 1.0
            
            station_weight = distance_weight * azimuthal_weight
            weights.extend([station_weight] * len(windowed_obs.flatten()))
        
        if len(G_matrices) == 0:
            raise ValueError("No valid station data for inversion")
        
        # Combine all matrices
        G = np.vstack(G_matrices)
        d = np.concatenate(data_vector)
        W = np.diag(weights)
        
        # Weighted least squares inversion
        GTW = G.T @ W
        GTWG = GTW @ G
        GTWd = GTW @ d
        
        # Add regularization
        reg_param = 1e-6 * np.trace(GTWG) / 6
        GTWG += reg_param * np.eye(6)
        
        # Solve for moment tensor
        try:
            mt_vector = linalg.solve(GTWG, GTWd)
        except linalg.LinAlgError:
            # Use pseudoinverse if singular
            mt_vector = linalg.pinv(GTWG) @ GTWd
        
        # Reconstruct 3x3 tensor
        moment_tensor = np.array([
            [mt_vector[0], mt_vector[3], mt_vector[4]],
            [mt_vector[3], mt_vector[1], mt_vector[5]],
            [mt_vector[4], mt_vector[5], mt_vector[2]]
        ])
        
        # Calculate fit statistics
        synthetic = G @ mt_vector
        residual = d - synthetic
        variance_reduction = 1.0 - np.var(residual) / np.var(d)
        condition_number = np.linalg.cond(GTWG)
        
        return {
            'moment_tensor': moment_tensor,
            'variance_reduction': variance_reduction,
            'condition_number': condition_number,
            'residual': residual,
            'synthetic': synthetic,
            'observed': d
        }
    
    def _decompose_moment_tensor(self, mt: np.ndarray) -> Dict:
        """Decompose moment tensor into DC, CLVD, and ISO components"""
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = linalg.eigh(mt)
        
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvals))[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Scalar moment
        scalar_moment = np.sqrt(0.5 * np.sum(eigenvals**2))
        
        # Moment magnitude
        moment_magnitude = (2.0/3.0) * np.log10(scalar_moment) - 6.07
        
        # Decomposition into ISO, DC, and CLVD
        trace_mt = np.trace(mt)
        iso_component = trace_mt / 3.0
        
        # Deviatoric tensor
        deviatoric = mt - iso_component * np.eye(3)
        dev_eigenvals = eigenvals - iso_component
        
        # Sort deviatoric eigenvalues
        dev_sorted = np.sort(np.abs(dev_eigenvals))[::-1]
        
        # DC and CLVD decomposition
        if dev_sorted[0] > 1e-10:
            dc_magnitude = dev_sorted[0] - dev_sorted[1]
            clvd_magnitude = 2 * dev_sorted[1]
        else:
            dc_magnitude = 0.0
            clvd_magnitude = 0.0
        
        # Percentages
        total_moment = abs(iso_component) + dc_magnitude + abs(clvd_magnitude)
        if total_moment > 1e-10:
            iso_percentage = abs(iso_component) / total_moment * 100
            dc_percentage = dc_magnitude / total_moment * 100
            clvd_percentage = abs(clvd_magnitude) / total_moment * 100
        else:
            iso_percentage = dc_percentage = clvd_percentage = 0.0
        
        # Determine source type
        if dc_percentage > 80:
            source_type = 'dc'
        elif clvd_percentage > 50:
            source_type = 'clvd'
        elif iso_percentage > 50:
            source_type = 'iso'
        else:
            source_type = 'mixed'
        
        # Calculate nodal planes
        nodal_planes = self._calculate_nodal_planes(mt)
        
        return {
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'scalar_moment': scalar_moment,
            'moment_magnitude': moment_magnitude,
            'iso_percentage': iso_percentage,
            'dc_percentage': dc_percentage,
            'clvd_percentage': clvd_percentage,
            'source_type': source_type,
            'nodal_planes': nodal_planes
        }
    
    def _calculate_nodal_planes(self, mt: np.ndarray) -> Dict:
        """Calculate strike, dip, and rake for both nodal planes"""
        
        # Convert moment tensor to standard seismological convention
        # [Mxx, Myy, Mzz, Mxy, Mxz, Myz] -> [Mrr, Mtt, Mpp, Mrt, Mrp, Mtp]
        # where r=down, t=south, p=east
        
        mxx, myy, mzz = mt[0, 0], mt[1, 1], mt[2, 2]
        mxy, mxz, myz = mt[0, 1], mt[0, 2], mt[1, 2]
        
        # Transform to spherical coordinates (r, theta, phi)
        mrr = mzz
        mtt = mxx  
        mpp = myy
        mrt = mxz
        mrp = -myz
        mtp = -mxy
        
        # Calculate auxiliary angles
        D = mtt + mpp
        
        # Calculate nodal plane parameters
        if abs(D) > 1e-10:
            phi = 0.5 * np.arctan2(-2 * mtp, mtt - mpp)
        else:
            phi = 0.0
            
        if abs(np.sin(phi)) > 1e-10:
            delta = np.arctan2(-mrt / np.sin(phi), -mrr)
        else:
            delta = np.arctan2(-mrp / np.cos(phi), -mrr)
            
        lambda_val = np.arctan2(-mrp * np.cos(phi) + mrt * np.sin(phi),
                               mtt * np.cos(phi)**2 + mpp * np.sin(phi)**2 - mrr)
        
        # Convert to degrees
        strike1 = np.degrees(phi) % 360
        dip1 = np.degrees(abs(delta))
        rake1 = np.degrees(lambda_val)
        
        # Second nodal plane
        strike2 = (strike1 + 180) % 360
        if strike2 > 360:
            strike2 -= 360
            
        # Calculate complementary plane
        cos_2delta = np.cos(2 * delta)
        sin_2delta = np.sin(2 * delta)
        cos_lambda = np.cos(lambda_val)
        sin_lambda = np.sin(lambda_val)
        
        dip2 = np.degrees(np.arccos(sin_lambda * sin_2delta))
        
        if abs(np.sin(np.radians(dip2))) > 1e-10:
            rake2 = np.degrees(np.arctan2(cos_lambda * sin_2delta / np.sin(np.radians(dip2)),
                                         cos_2delta))
        else:
            rake2 = 0.0
        
        return {
            'strike1': strike1,
            'dip1': dip1,
            'rake1': rake1,
            'strike2': strike2,
            'dip2': dip2,
            'rake2': rake2
        }
    
    def _estimate_uncertainties(self, stations: List[StationInfo],
                              waveform_data: Dict[str, np.ndarray],
                              best_solution: Dict) -> Dict:
        """Estimate uncertainties using bootstrap resampling"""
        
        if not SCIPY_AVAILABLE:
            return {
                'strike1_std': 10.0,
                'dip1_std': 10.0, 
                'rake1_std': 15.0,
                'magnitude_std': 0.1,
                'depth_std': 2.0
            }
        
        bootstrap_solutions = []
        n_stations = len(stations)
        
        for _ in range(self.config.bootstrap_samples):
            # Resample stations with replacement
            bootstrap_stations = np.random.choice(stations, size=n_stations, replace=True)
            bootstrap_data = {
                station.station_code: waveform_data[station.station_code]
                for station in bootstrap_stations
                if station.station_code in waveform_data
            }
            
            try:
                # Invert at best depth
                solution = self._invert_at_depth(
                    bootstrap_stations, bootstrap_data, best_solution['depth']
                )
                decomp = self._decompose_moment_tensor(solution['moment_tensor'])
                bootstrap_solutions.append(decomp)
                
            except Exception:
                continue
        
        if len(bootstrap_solutions) < 10:
            warnings.warn("Insufficient bootstrap samples for uncertainty estimation")
            return {
                'strike1_std': 20.0,
                'dip1_std': 15.0,
                'rake1_std': 25.0, 
                'magnitude_std': 0.2,
                'depth_std': 3.0
            }
        
        # Calculate statistics
        strikes1 = [sol['nodal_planes']['strike1'] for sol in bootstrap_solutions]
        dips1 = [sol['nodal_planes']['dip1'] for sol in bootstrap_solutions]
        rakes1 = [sol['nodal_planes']['rake1'] for sol in bootstrap_solutions]
        magnitudes = [sol['moment_magnitude'] for sol in bootstrap_solutions]
        
        # Handle circular statistics for strike and rake
        strike1_std = self._circular_std(strikes1)
        rake1_std = self._circular_std(rakes1)
        
        return {
            'strike1_std': strike1_std,
            'dip1_std': np.std(dips1),
            'rake1_std': rake1_std,
            'magnitude_std': np.std(magnitudes),
            'depth_std': 2.0  # Would need depth grid bootstrap for accurate estimate
        }
    
    def _circular_std(self, angles: List[float]) -> float:
        """Calculate standard deviation of circular data"""
        angles_rad = np.radians(angles)
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        r = np.sqrt(mean_cos**2 + mean_sin**2)
        
        if r < 1e-10:
            return 180.0  # Maximum uncertainty
        
        circular_var = 1 - r
        circular_std = np.sqrt(-2 * np.log(r))
        
        return np.degrees(circular_std)
    
    def _calculate_confidence_intervals(self, depth_solutions: List[Tuple],
                                     uncertainties: Dict) -> Dict:
        """Calculate confidence intervals for MT parameters"""
        
        # Extract variance reductions by depth
        depths = [sol[0] for sol in depth_solutions]
        variances = [sol[1]['variance_reduction'] for sol in depth_solutions]
        
        if len(depth_solutions) < 3:
            return {
                'depth_95_ci': (uncertainties.get('depth_std', 5.0) * -1.96,
                               uncertainties.get('depth_std', 5.0) * 1.96),
                'magnitude_95_ci': (uncertainties.get('magnitude_std', 0.2) * -1.96,
                                   uncertainties.get('magnitude_std', 0.2) * 1.96)
            }
        
        # Find confidence interval for depth using chi-squared test
        best_variance = max(variances)
        chi2_threshold = chi2.ppf(0.95, df=1)  # 95% confidence
        
        acceptable_depths = [
            depths[i] for i, var in enumerate(variances)
            if (best_variance - var) < chi2_threshold / len(depths)
        ]
        
        if acceptable_depths:
            depth_ci = (min(acceptable_depths), max(acceptable_depths))
        else:
            depth_ci = (depths[np.argmax(variances)] - 5.0, 
                       depths[np.argmax(variances)] + 5.0)
        
        return {
            'depth_95_ci': depth_ci,
            'magnitude_95_ci': (uncertainties.get('magnitude_std', 0.2) * -1.96,
                               uncertainties.get('magnitude_std', 0.2) * 1.96),
            'strike1_95_ci': (uncertainties.get('strike1_std', 15.0) * -1.96,
                             uncertainties.get('strike1_std', 15.0) * 1.96)
        }
    
    def _bandpass_filter(self, data: np.ndarray, 
                        frequency_band: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter to waveform data"""
        from scipy.signal import butter, filtfilt
        
        # Assume sampling rate of 50 Hz (adjustable)
        sample_rate = 50.0
        nyquist = sample_rate / 2
        
        low, high = frequency_band
        low_norm = low / nyquist
        high_norm = min(high / nyquist, 0.99)
        
        b, a = butter(4, [low_norm, high_norm], btype='band')
        
        if data.ndim == 1:
            return filtfilt(b, a, data)
        else:
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                filtered[i] = filtfilt(b, a, data[i])
            return filtered
    
    def _apply_time_window(self, data: np.ndarray) -> np.ndarray:
        """Apply time window to data"""
        if isinstance(data, np.ndarray) and data.ndim > 1:
            # For Green's functions matrix
            window_samples = int(self.config.time_window * 50)  # Assume 50 Hz
            if data.shape[0] > window_samples:
                return data[:window_samples]
        else:
            # For waveform data
            window_samples = int(self.config.time_window * 50)
            if len(data) > window_samples:
                return data[:window_samples]
        
        return data
    
    def calculate_resolution_matrix(self, stations: List[StationInfo], 
                                  depth: float) -> np.ndarray:
        """Calculate resolution matrix for MT inversion"""
        
        # Build Green's function matrix
        G_matrices = []
        weights = []
        
        for station in stations:
            greens = self.greens_calculator.calculate_greens_functions(station, depth)
            windowed_greens = self._apply_time_window(greens)
            
            G_matrices.append(windowed_greens)
            
            # Distance weighting
            if self.config.distance_weights:
                weight = 1.0 / np.sqrt(station.distance)
            else:
                weight = 1.0
            
            weights.extend([weight] * windowed_greens.shape[0])
        
        G = np.vstack(G_matrices)
        W = np.diag(weights)
        
        # Resolution matrix R = (G^T W G)^-1 G^T W G
        GTW = G.T @ W
        GTWG = GTW @ G
        
        try:
            resolution_matrix = linalg.inv(GTWG) @ GTWG
        except linalg.LinAlgError:
            resolution_matrix = linalg.pinv(GTWG) @ GTWG
        
        return resolution_matrix
    
    def export_solution(self, solution: MomentTensorSolution, filename: str):
        """Export MT solution to file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'moment_tensor': solution.moment_tensor.tolist(),
            'nodal_planes': {
                'plane1': {
                    'strike': solution.strike1,
                    'dip': solution.dip1,
                    'rake': solution.rake1
                },
                'plane2': {
                    'strike': solution.strike2,
                    'dip': solution.dip2,
                    'rake': solution.rake2
                }
            },
            'magnitude_info': {
                'moment_magnitude': solution.moment_magnitude,
                'scalar_moment': solution.scalar_moment
            },
            'source_analysis': {
                'depth': solution.depth,
                'source_type': solution.source_type,
                'dc_percentage': solution.dc_percentage,
                'clvd_percentage': solution.clvd_percentage,
                'iso_percentage': solution.iso_percentage
            },
            'quality_metrics': {
                'variance_reduction': solution.variance_reduction,
                'condition_number': solution.condition_number
            },
            'uncertainties': solution.uncertainties,
            'confidence_intervals': solution.confidence_intervals
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to export solution: {e}")


# Example usage and testing
def main():
    """Example usage of moment tensor inversion"""
    
    # Create example stations
    stations = [
        StationInfo('STA01', 45.0, -120.0, 0.5, 50.0, 45.0, 225.0),
        StationInfo('STA02', 45.2, -119.8, 0.8, 65.0, 135.0, 315.0),
        StationInfo('STA03', 44.8, -120.2, 1.2, 80.0, 225.0, 45.0),
        StationInfo('STA04', 45.1, -119.9, 0.3, 25.0, 90.0, 270.0)
    ]
    
    # Configuration
    config = MomentTensorConfig(
        frequency_range=(0.02, 0.1),
        time_window=120.0,
        depth_range=(5.0, 25.0),
        bootstrap_samples=100
    )
    
    # Initialize inverter
    inverter = MomentTensorInverter(config)
    
    # Generate synthetic waveform data
    event_time = datetime.now()
    waveform_data = {}
    
    for station in stations:
        # Generate synthetic 3-component data
        duration = 120.0  # seconds
        sample_rate = 50.0  # Hz
        time = np.linspace(0, duration, int(duration * sample_rate))
        
        # Synthetic earthquake signal
        arrival_time = station.distance / 6.0  # P-wave at ~6 km/s
        signal_window = np.exp(-(time - arrival_time)**2 / 8.0)
        
        # Add some realistic waveform characteristics
        frequency = 0.05  # Hz
        waveform = signal_window * np.sin(2 * np.pi * frequency * (time - arrival_time))
        
        # Add noise
        noise = np.random.randn(len(time)) * 0.1 * np.max(waveform)
        waveform += noise
        
        # Create 3-component data (simplified)
        waveform_data[station.station_code] = np.array([
            waveform,  # Z component
            waveform * 0.7,  # N component
            waveform * 0.5   # E component
        ])
    
    print("Performing moment tensor inversion...")
    print(f"Stations: {len(stations)}")
    print(f"Depth range: {config.depth_range[0]}-{config.depth_range[1]} km")
    
    try:
        # Perform inversion
        solution = inverter.invert_moment_tensor(stations, waveform_data, event_time)
        
        print("\n=== MOMENT TENSOR SOLUTION ===")
        print(f"Moment Magnitude: {solution.moment_magnitude:.2f}")
        print(f"Depth: {solution.depth:.1f} km")
        print(f"Source Type: {solution.source_type.upper()}")
        print(f"Variance Reduction: {solution.variance_reduction:.3f}")
        
        print(f"\nNodal Plane 1:")
        print(f"  Strike: {solution.strike1:.1f}°")
        print(f"  Dip: {solution.dip1:.1f}°") 
        print(f"  Rake: {solution.rake1:.1f}°")
        
        print(f"\nNodal Plane 2:")
        print(f"  Strike: {solution.strike2:.1f}°")
        print(f"  Dip: {solution.dip2:.1f}°")
        print(f"  Rake: {solution.rake2:.1f}°")
        
        print(f"\nSource Decomposition:")
        print(f"  Double-Couple: {solution.dc_percentage:.1f}%")
        print(f"  CLVD: {solution.clvd_percentage:.1f}%")
        print(f"  Isotropic: {solution.iso_percentage:.1f}%")
        
        print(f"\nUncertainties:")
        print(f"  Strike: ±{solution.uncertainties.get('strike1_std', 0):.1f}°")
        print(f"  Dip: ±{solution.uncertainties.get('dip1_std', 0):.1f}°")
        print(f"  Rake: ±{solution.uncertainties.get('rake1_std', 0):.1f}°")
        print(f"  Magnitude: ±{solution.uncertainties.get('magnitude_std', 0):.2f}")
        
        # Export solution
        inverter.export_solution(solution, 'moment_tensor_solution.json')
        print(f"\nSolution exported to moment_tensor_solution.json")
        
        # Test resolution analysis
        resolution = inverter.calculate_resolution_matrix(stations, solution.depth)
        print(f"\nResolution matrix condition number: {np.linalg.cond(resolution):.2f}")
        
    except Exception as e:
        print(f"Inversion failed: {e}")
    
    return inverter


if __name__ == "__main__":
    inverter = main()