"""
Advanced Stress Field Mapping and Analysis System
Regional stress inversion, temporal stress changes, and comprehensive
stress field characterization for earthquake hazard assessment
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Callable
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
    from scipy.optimize import minimize, differential_evolution, least_squares
    from scipy import linalg
    from scipy.spatial import KDTree, distance
    from scipy.interpolate import griddata, RBFInterpolator
    from scipy.stats import gaussian_kde, chi2
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class StressFieldConfig:
    """Configuration for stress field analysis"""
    inversion_method: str = "michael"  # michael, gephart, hardebeck
    grid_spacing: float = 0.1  # degrees
    smoothing_radius: float = 50.0  # km
    minimum_events: int = 5
    temporal_window: float = 365.0  # days
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    stress_regime_classification: bool = True
    anisotropy_analysis: bool = True
    temporal_evolution: bool = True


@dataclass
class FocalMechanism:
    """Focal mechanism data structure"""
    event_id: str
    latitude: float
    longitude: float
    depth: float
    magnitude: float
    strike1: float
    dip1: float
    rake1: float
    strike2: float
    dip2: float
    rake2: float
    timestamp: datetime
    uncertainty: Optional[Dict] = None
    quality: Optional[str] = None


@dataclass
class StressTensor:
    """3D stress tensor representation"""
    sigma1: float  # Maximum principal stress
    sigma2: float  # Intermediate principal stress  
    sigma3: float  # Minimum principal stress
    azimuth1: float  # Azimuth of sigma1
    plunge1: float  # Plunge of sigma1
    azimuth2: float  # Azimuth of sigma2
    plunge2: float  # Plunge of sigma2
    azimuth3: float  # Azimuth of sigma3
    plunge3: float  # Plunge of sigma3
    stress_ratio: float  # (sigma2-sigma3)/(sigma1-sigma3)
    regime: str  # NF (normal), SS (strike-slip), TF (thrust)
    confidence: float


@dataclass
class StressField:
    """Regional stress field representation"""
    grid_points: np.ndarray  # (N, 2) lat/lon coordinates
    stress_tensors: List[StressTensor]
    interpolation_method: str
    resolution: float
    temporal_range: Tuple[datetime, datetime]
    quality_metrics: Dict
    uncertainties: Dict


class StressInversionEngine:
    """Core engine for stress field inversion"""
    
    def __init__(self, config: StressFieldConfig):
        self.config = config
        self.focal_mechanisms = []
        self.stress_field = None
        
        # Initialize ML components if available
        self.ml_models = {}
        if TORCH_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize ML models for enhanced stress analysis"""
        
        class StressRegimeClassifier(nn.Module):
            """Neural network for stress regime classification"""
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Linear(6, 128),  # strike, dip, rake for both planes
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3)  # Normal, Strike-slip, Thrust
                )
                
            def forward(self, focal_params):
                return F.softmax(self.features(focal_params), dim=1)
        
        class StressTensorPredictor(nn.Module):
            """Neural network for stress tensor prediction"""
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(6, 256),  # Focal mechanism parameters
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU()
                )
                
                # Stress orientation heads
                self.sigma1_head = nn.Linear(64, 2)  # azimuth, plunge
                self.sigma2_head = nn.Linear(64, 2)
                self.sigma3_head = nn.Linear(64, 2)
                self.ratio_head = nn.Sequential(
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, focal_params):
                features = self.encoder(focal_params)
                return {
                    'sigma1_orientation': self.sigma1_head(features),
                    'sigma2_orientation': self.sigma2_head(features),
                    'sigma3_orientation': self.sigma3_head(features),
                    'stress_ratio': self.ratio_head(features)
                }
        
        self.ml_models = {
            'regime_classifier': StressRegimeClassifier(),
            'stress_predictor': StressTensorPredictor()
        }
    
    def add_focal_mechanisms(self, focal_mechanisms: List[FocalMechanism]):
        """Add focal mechanism data for inversion"""
        self.focal_mechanisms.extend(focal_mechanisms)
        
        # Sort by time for temporal analysis
        self.focal_mechanisms.sort(key=lambda x: x.timestamp)
        
        print(f"Added {len(focal_mechanisms)} focal mechanisms")
        print(f"Total mechanisms: {len(self.focal_mechanisms)}")
    
    def invert_stress_field(self, region_bounds: Tuple[float, float, float, float],
                          method: Optional[str] = None) -> StressField:
        """
        Invert regional stress field from focal mechanisms
        
        Args:
            region_bounds: (min_lat, max_lat, min_lon, max_lon)
            method: Inversion method ('michael', 'gephart', 'hardebeck')
        """
        method = method or self.config.inversion_method
        
        # Filter focal mechanisms in region
        regional_mechanisms = self._filter_mechanisms_by_region(region_bounds)
        
        if len(regional_mechanisms) < self.config.minimum_events:
            raise ValueError(f"Insufficient focal mechanisms ({len(regional_mechanisms)}) for inversion")
        
        # Create spatial grid
        grid_points = self._create_spatial_grid(region_bounds)
        
        # Perform stress inversion at each grid point
        stress_tensors = []
        quality_metrics = {'successful_inversions': 0, 'failed_inversions': 0}
        
        for i, point in enumerate(grid_points):
            try:
                # Find nearby mechanisms
                nearby_mechanisms = self._find_nearby_mechanisms(
                    point, regional_mechanisms, self.config.smoothing_radius
                )
                
                if len(nearby_mechanisms) >= self.config.minimum_events:
                    # Perform stress inversion
                    if method == 'michael':
                        stress_tensor = self._michael_inversion(nearby_mechanisms)
                    elif method == 'gephart':
                        stress_tensor = self._gephart_inversion(nearby_mechanisms)
                    elif method == 'hardebeck':
                        stress_tensor = self._hardebeck_inversion(nearby_mechanisms)
                    else:
                        raise ValueError(f"Unknown inversion method: {method}")
                    
                    stress_tensors.append(stress_tensor)
                    quality_metrics['successful_inversions'] += 1
                else:
                    # Create placeholder stress tensor
                    stress_tensors.append(self._create_null_stress_tensor())
                    quality_metrics['failed_inversions'] += 1
                    
            except Exception as e:
                warnings.warn(f"Stress inversion failed at point {i}: {e}")
                stress_tensors.append(self._create_null_stress_tensor())
                quality_metrics['failed_inversions'] += 1
        
        # Calculate uncertainties
        uncertainties = self._estimate_stress_uncertainties(
            regional_mechanisms, grid_points, stress_tensors
        )
        
        # Create stress field object
        time_range = (
            min(fm.timestamp for fm in regional_mechanisms),
            max(fm.timestamp for fm in regional_mechanisms)
        )
        
        stress_field = StressField(
            grid_points=grid_points,
            stress_tensors=stress_tensors,
            interpolation_method=method,
            resolution=self.config.grid_spacing,
            temporal_range=time_range,
            quality_metrics=quality_metrics,
            uncertainties=uncertainties
        )
        
        self.stress_field = stress_field
        return stress_field
    
    def _michael_inversion(self, mechanisms: List[FocalMechanism]) -> StressTensor:
        """Michael (1984) stress inversion method"""
        
        # Convert focal mechanisms to slip vectors
        slip_data = []
        
        for fm in mechanisms:
            # Both nodal planes
            for plane_num in [1, 2]:
                if plane_num == 1:
                    strike, dip, rake = fm.strike1, fm.dip1, fm.rake1
                else:
                    strike, dip, rake = fm.strike2, fm.dip2, fm.rake2
                
                # Convert to Cartesian coordinates
                fault_normal = self._sdr_to_normal(strike, dip)
                slip_vector = self._sdr_to_slip(strike, dip, rake)
                
                slip_data.append((fault_normal, slip_vector))
        
        if len(slip_data) < 3:
            return self._create_null_stress_tensor()
        
        # Set up optimization problem
        def misfit_function(params):
            """Calculate misfit for given stress parameters"""
            sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi = params
            
            # Construct stress tensor
            stress_tensor = self._construct_stress_tensor(
                sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi
            )
            
            total_misfit = 0.0
            for fault_normal, slip_vector in slip_data:
                # Calculate shear stress on fault plane
                shear_stress = self._calculate_shear_stress(stress_tensor, fault_normal)
                
                # Misfit is angle between predicted and observed slip
                if np.linalg.norm(shear_stress) > 1e-10:
                    predicted_slip = shear_stress / np.linalg.norm(shear_stress)
                    misfit_angle = np.arccos(np.clip(np.dot(predicted_slip, slip_vector), -1, 1))
                    total_misfit += misfit_angle**2
                else:
                    total_misfit += np.pi**2  # Maximum misfit
            
            return total_misfit / len(slip_data)
        
        # Initial guess and bounds
        x0 = [0, 0, 90, 90, 0.5]  # sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi
        bounds = [(0, 360), (-90, 90), (0, 360), (-90, 90), (0, 1)]
        
        # Optimize
        try:
            if SCIPY_AVAILABLE:
                result = minimize(misfit_function, x0, bounds=bounds, method='L-BFGS-B')
                optimal_params = result.x
            else:
                optimal_params = x0  # Fallback
            
            # Construct final stress tensor
            sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi = optimal_params
            stress_tensor_matrix = self._construct_stress_tensor(
                sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi
            )
            
            # Extract principal stresses and orientations
            eigenvals, eigenvecs = linalg.eigh(stress_tensor_matrix)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Convert to geological convention
            orientations = []
            for i in range(3):
                trend, plunge = self._cartesian_to_geological(eigenvecs[:, i])
                orientations.append((trend, plunge))
            
            # Classify stress regime
            regime = self._classify_stress_regime(eigenvals[0], eigenvals[1], eigenvals[2])
            
            # Calculate stress ratio
            stress_ratio = (eigenvals[1] - eigenvals[2]) / (eigenvals[0] - eigenvals[2])
            if np.isnan(stress_ratio) or np.isinf(stress_ratio):
                stress_ratio = 0.5
            
            return StressTensor(
                sigma1=eigenvals[0],
                sigma2=eigenvals[1], 
                sigma3=eigenvals[2],
                azimuth1=orientations[0][0],
                plunge1=orientations[0][1],
                azimuth2=orientations[1][0],
                plunge2=orientations[1][1],
                azimuth3=orientations[2][0],
                plunge3=orientations[2][1],
                stress_ratio=stress_ratio,
                regime=regime,
                confidence=1.0 - result.fun if SCIPY_AVAILABLE else 0.5
            )
            
        except Exception as e:
            warnings.warn(f"Michael inversion failed: {e}")
            return self._create_null_stress_tensor()
    
    def _gephart_inversion(self, mechanisms: List[FocalMechanism]) -> StressTensor:
        """Gephart and Forsyth (1984) inversion method"""
        
        # Simplified implementation - similar to Michael but with different constraints
        return self._michael_inversion(mechanisms)  # Placeholder
    
    def _hardebeck_inversion(self, mechanisms: List[FocalMechanism]) -> StressTensor:
        """Hardebeck and Michael (2006) inversion method"""
        
        # Enhanced version with better uncertainty estimation
        base_result = self._michael_inversion(mechanisms)
        
        # Add uncertainty estimation through bootstrap
        if len(mechanisms) > 5:
            bootstrap_results = []
            n_bootstrap = min(100, self.config.bootstrap_samples // 10)
            
            for _ in range(n_bootstrap):
                # Resample mechanisms
                bootstrap_mechanisms = np.random.choice(mechanisms, size=len(mechanisms), replace=True)
                try:
                    bootstrap_result = self._michael_inversion(bootstrap_mechanisms.tolist())
                    bootstrap_results.append(bootstrap_result)
                except:
                    continue
            
            # Calculate confidence from bootstrap spread
            if bootstrap_results:
                azimuth_spread = np.std([result.azimuth1 for result in bootstrap_results])
                confidence = max(0.0, 1.0 - azimuth_spread / 180.0)
                base_result.confidence = confidence
        
        return base_result
    
    def analyze_temporal_evolution(self, time_windows: List[Tuple[datetime, datetime]]) -> Dict:
        """Analyze temporal evolution of stress field"""
        
        temporal_analysis = {
            'time_windows': [],
            'stress_evolution': [],
            'regime_changes': [],
            'significance_tests': []
        }
        
        previous_stress = None
        
        for i, (start_time, end_time) in enumerate(time_windows):
            # Filter mechanisms in time window
            windowed_mechanisms = [
                fm for fm in self.focal_mechanisms
                if start_time <= fm.timestamp <= end_time
            ]
            
            if len(windowed_mechanisms) < self.config.minimum_events:
                continue
            
            # Invert stress field for this time window
            try:
                # Use a representative point for temporal analysis
                center_point = np.array([0.0, 0.0])  # Would be region center
                stress_tensor = self._michael_inversion(windowed_mechanisms)
                
                temporal_analysis['time_windows'].append((start_time, end_time))
                temporal_analysis['stress_evolution'].append(stress_tensor)
                
                # Compare with previous stress state
                if previous_stress is not None:
                    regime_change = stress_tensor.regime != previous_stress.regime
                    angular_change = self._calculate_stress_rotation(previous_stress, stress_tensor)
                    
                    temporal_analysis['regime_changes'].append({
                        'time_window': i,
                        'regime_change': regime_change,
                        'angular_rotation': angular_change,
                        'significance': angular_change > 15.0  # Significant if >15 degrees
                    })
                
                previous_stress = stress_tensor
                
            except Exception as e:
                warnings.warn(f"Temporal analysis failed for window {i}: {e}")
                continue
        
        return temporal_analysis
    
    def calculate_stress_heterogeneity(self, stress_field: StressField) -> Dict:
        """Calculate spatial heterogeneity of stress field"""
        
        if not stress_field.stress_tensors:
            return {'heterogeneity_index': 0.0, 'coherence_length': 0.0}
        
        # Extract principal stress orientations
        azimuths = []
        plunges = []
        regimes = []
        
        for tensor in stress_field.stress_tensors:
            if tensor.confidence > 0.1:  # Only use reliable tensors
                azimuths.append(tensor.azimuth1)
                plunges.append(tensor.plunge1)
                regimes.append(tensor.regime)
        
        if len(azimuths) < 2:
            return {'heterogeneity_index': 0.0, 'coherence_length': 0.0}
        
        # Calculate angular differences
        angular_differences = []
        distances = []
        
        for i in range(len(azimuths)):
            for j in range(i + 1, len(azimuths)):
                # Angular difference
                az_diff = abs(azimuths[i] - azimuths[j])
                az_diff = min(az_diff, 360 - az_diff)  # Handle wraparound
                
                pl_diff = abs(plunges[i] - plunges[j])
                
                angular_diff = np.sqrt(az_diff**2 + pl_diff**2)
                angular_differences.append(angular_diff)
                
                # Spatial distance
                point_i = stress_field.grid_points[i]
                point_j = stress_field.grid_points[j]
                dist = self._haversine_distance(point_i[0], point_i[1], point_j[0], point_j[1])
                distances.append(dist)
        
        # Calculate heterogeneity index
        if angular_differences:
            heterogeneity_index = np.std(angular_differences) / 180.0  # Normalized
        else:
            heterogeneity_index = 0.0
        
        # Estimate coherence length
        if len(distances) > 0 and SCIPY_AVAILABLE:
            # Find distance at which angular differences exceed threshold
            threshold_angle = 30.0  # degrees
            coherent_distances = [
                d for d, ang in zip(distances, angular_differences)
                if ang < threshold_angle
            ]
            
            if coherent_distances:
                coherence_length = np.percentile(coherent_distances, 90)
            else:
                coherence_length = min(distances) if distances else 0.0
        else:
            coherence_length = 0.0
        
        return {
            'heterogeneity_index': heterogeneity_index,
            'coherence_length': coherence_length,
            'mean_angular_difference': np.mean(angular_differences) if angular_differences else 0.0,
            'regime_consistency': len(set(regimes)) / len(regimes) if regimes else 0.0
        }
    
    def detect_stress_anomalies(self, stress_field: StressField) -> List[Dict]:
        """Detect anomalous stress orientations or regimes"""
        
        anomalies = []
        
        if len(stress_field.stress_tensors) < 5:
            return anomalies
        
        # Extract stress parameters
        azimuths = np.array([t.azimuth1 for t in stress_field.stress_tensors if t.confidence > 0.1])
        plunges = np.array([t.plunge1 for t in stress_field.stress_tensors if t.confidence > 0.1])
        ratios = np.array([t.stress_ratio for t in stress_field.stress_tensors if t.confidence > 0.1])
        
        if len(azimuths) < 5:
            return anomalies
        
        # Statistical outlier detection
        if SCIPY_AVAILABLE:
            # Circular statistics for azimuths
            az_rad = np.radians(azimuths)
            mean_cos = np.mean(np.cos(az_rad))
            mean_sin = np.mean(np.sin(az_rad))
            circular_mean = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
            
            # Circular standard deviation
            r = np.sqrt(mean_cos**2 + mean_sin**2)
            circular_std = np.degrees(np.sqrt(-2 * np.log(r))) if r > 0 else 180.0
            
            # Find azimuth anomalies
            for i, (az, pl, ratio) in enumerate(zip(azimuths, plunges, ratios)):
                # Angular distance from mean
                angular_dist = min(abs(az - circular_mean), 360 - abs(az - circular_mean))
                
                if angular_dist > 2 * circular_std:  # 2-sigma outlier
                    point = stress_field.grid_points[i]
                    anomalies.append({
                        'type': 'orientation_anomaly',
                        'location': (point[0], point[1]),
                        'azimuth': az,
                        'deviation': angular_dist,
                        'significance': angular_dist / circular_std
                    })
            
            # Find stress ratio anomalies
            ratio_mean = np.mean(ratios)
            ratio_std = np.std(ratios)
            
            for i, ratio in enumerate(ratios):
                if abs(ratio - ratio_mean) > 2 * ratio_std:
                    point = stress_field.grid_points[i]
                    anomalies.append({
                        'type': 'ratio_anomaly',
                        'location': (point[0], point[1]),
                        'stress_ratio': ratio,
                        'deviation': abs(ratio - ratio_mean),
                        'significance': abs(ratio - ratio_mean) / ratio_std
                    })
        
        return anomalies
    
    def _filter_mechanisms_by_region(self, bounds: Tuple[float, float, float, float]) -> List[FocalMechanism]:
        """Filter focal mechanisms by geographic region"""
        min_lat, max_lat, min_lon, max_lon = bounds
        
        filtered = []
        for fm in self.focal_mechanisms:
            if (min_lat <= fm.latitude <= max_lat and 
                min_lon <= fm.longitude <= max_lon):
                filtered.append(fm)
        
        return filtered
    
    def _create_spatial_grid(self, bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Create spatial grid for stress field inversion"""
        min_lat, max_lat, min_lon, max_lon = bounds
        
        lat_points = np.arange(min_lat, max_lat + self.config.grid_spacing, self.config.grid_spacing)
        lon_points = np.arange(min_lon, max_lon + self.config.grid_spacing, self.config.grid_spacing)
        
        grid_points = []
        for lat in lat_points:
            for lon in lon_points:
                grid_points.append([lat, lon])
        
        return np.array(grid_points)
    
    def _find_nearby_mechanisms(self, point: np.ndarray, mechanisms: List[FocalMechanism], 
                              radius_km: float) -> List[FocalMechanism]:
        """Find focal mechanisms within radius of a point"""
        nearby = []
        
        for fm in mechanisms:
            distance = self._haversine_distance(point[0], point[1], fm.latitude, fm.longitude)
            if distance <= radius_km:
                nearby.append(fm)
        
        return nearby
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points"""
        R = 6371.0  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _sdr_to_normal(self, strike: float, dip: float) -> np.ndarray:
        """Convert strike/dip to fault normal vector"""
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Fault normal (upward)
        normal = np.array([
            -np.sin(dip_rad) * np.sin(strike_rad),
            np.sin(dip_rad) * np.cos(strike_rad),
            np.cos(dip_rad)
        ])
        
        return normal / np.linalg.norm(normal)
    
    def _sdr_to_slip(self, strike: float, dip: float, rake: float) -> np.ndarray:
        """Convert strike/dip/rake to slip vector"""
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        rake_rad = np.radians(rake)
        
        # Strike vector
        strike_vec = np.array([np.cos(strike_rad), np.sin(strike_rad), 0])
        
        # Dip vector
        dip_vec = np.array([
            -np.sin(strike_rad) * np.cos(dip_rad),
            np.cos(strike_rad) * np.cos(dip_rad),
            -np.sin(dip_rad)
        ])
        
        # Slip vector
        slip = np.cos(rake_rad) * strike_vec + np.sin(rake_rad) * dip_vec
        
        return slip / np.linalg.norm(slip)
    
    def _construct_stress_tensor(self, sigma1_trend: float, sigma1_plunge: float,
                               sigma3_trend: float, sigma3_plunge: float, phi: float) -> np.ndarray:
        """Construct 3D stress tensor from principal axes"""
        
        # Convert to Cartesian coordinates
        sigma1_vec = self._geological_to_cartesian(sigma1_trend, sigma1_plunge)
        sigma3_vec = self._geological_to_cartesian(sigma3_trend, sigma3_plunge)
        
        # Sigma2 is perpendicular to both sigma1 and sigma3
        sigma2_vec = np.cross(sigma1_vec, sigma3_vec)
        sigma2_vec = sigma2_vec / np.linalg.norm(sigma2_vec)
        
        # Principal stress magnitudes (normalized)
        sigma1_mag = 1.0
        sigma3_mag = 0.0
        sigma2_mag = phi  # Stress ratio parameter
        
        # Construct stress tensor
        stress_tensor = (sigma1_mag * np.outer(sigma1_vec, sigma1_vec) +
                        sigma2_mag * np.outer(sigma2_vec, sigma2_vec) +
                        sigma3_mag * np.outer(sigma3_vec, sigma3_vec))
        
        return stress_tensor
    
    def _geological_to_cartesian(self, trend: float, plunge: float) -> np.ndarray:
        """Convert geological trend/plunge to Cartesian coordinates"""
        trend_rad = np.radians(trend)
        plunge_rad = np.radians(plunge)
        
        x = np.cos(plunge_rad) * np.sin(trend_rad)
        y = np.cos(plunge_rad) * np.cos(trend_rad)
        z = np.sin(plunge_rad)
        
        return np.array([x, y, z])
    
    def _cartesian_to_geological(self, vector: np.ndarray) -> Tuple[float, float]:
        """Convert Cartesian vector to geological trend/plunge"""
        x, y, z = vector
        
        trend = np.degrees(np.arctan2(x, y)) % 360
        plunge = np.degrees(np.arcsin(z))
        
        return trend, plunge
    
    def _calculate_shear_stress(self, stress_tensor: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Calculate shear stress on a fault plane"""
        # Traction vector
        traction = stress_tensor @ normal
        
        # Normal stress component
        normal_stress = np.dot(traction, normal)
        
        # Shear stress vector
        shear_stress = traction - normal_stress * normal
        
        return shear_stress
    
    def _classify_stress_regime(self, sigma1: float, sigma2: float, sigma3: float) -> str:
        """Classify stress regime based on principal stresses"""
        # Stress ratios
        if abs(sigma1 - sigma3) < 1e-10:
            return 'isotropic'
        
        phi = (sigma2 - sigma3) / (sigma1 - sigma3)
        
        if phi < 0.33:
            return 'NF'  # Normal faulting
        elif phi < 0.67:
            return 'SS'  # Strike-slip
        else:
            return 'TF'  # Thrust faulting
    
    def _calculate_stress_rotation(self, stress1: StressTensor, stress2: StressTensor) -> float:
        """Calculate angular rotation between two stress states"""
        # Convert orientations to vectors
        vec1 = self._geological_to_cartesian(stress1.azimuth1, stress1.plunge1)
        vec2 = self._geological_to_cartesian(stress2.azimuth1, stress2.plunge1)
        
        # Calculate angle between vectors
        cos_angle = np.clip(np.dot(vec1, vec2), -1, 1)
        angle = np.degrees(np.arccos(abs(cos_angle)))
        
        return angle
    
    def _create_null_stress_tensor(self) -> StressTensor:
        """Create null stress tensor for failed inversions"""
        return StressTensor(
            sigma1=1.0, sigma2=0.5, sigma3=0.0,
            azimuth1=0.0, plunge1=0.0,
            azimuth2=90.0, plunge2=0.0,
            azimuth3=0.0, plunge3=90.0,
            stress_ratio=0.5,
            regime='unknown',
            confidence=0.0
        )
    
    def _estimate_stress_uncertainties(self, mechanisms: List[FocalMechanism],
                                     grid_points: np.ndarray,
                                     stress_tensors: List[StressTensor]) -> Dict:
        """Estimate uncertainties in stress field"""
        
        # Simple uncertainty estimation based on data density and quality
        uncertainties = {
            'azimuth_uncertainty': [],
            'plunge_uncertainty': [],
            'ratio_uncertainty': [],
            'regime_uncertainty': []
        }
        
        for i, point in enumerate(grid_points):
            nearby_mechanisms = self._find_nearby_mechanisms(
                point, mechanisms, self.config.smoothing_radius
            )
            
            # Uncertainty inversely proportional to data density
            n_mechanisms = len(nearby_mechanisms)
            base_uncertainty = max(5.0, 30.0 / max(n_mechanisms, 1))
            
            uncertainties['azimuth_uncertainty'].append(base_uncertainty)
            uncertainties['plunge_uncertainty'].append(base_uncertainty * 0.7)
            uncertainties['ratio_uncertainty'].append(0.1 * base_uncertainty / 10.0)
            uncertainties['regime_uncertainty'].append(min(0.5, base_uncertainty / 60.0))
        
        return uncertainties
    
    def export_stress_field(self, stress_field: StressField, filename: str):
        """Export stress field to file"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'inversion_method': self.config.inversion_method,
                'grid_spacing': self.config.grid_spacing,
                'smoothing_radius': self.config.smoothing_radius,
                'minimum_events': self.config.minimum_events
            },
            'temporal_range': [
                stress_field.temporal_range[0].isoformat(),
                stress_field.temporal_range[1].isoformat()
            ],
            'grid_points': stress_field.grid_points.tolist(),
            'stress_tensors': [],
            'quality_metrics': stress_field.quality_metrics,
            'uncertainties': stress_field.uncertainties
        }
        
        # Convert stress tensors to serializable format
        for i, tensor in enumerate(stress_field.stress_tensors):
            if tensor.confidence > 0:
                export_data['stress_tensors'].append({
                    'grid_index': i,
                    'location': stress_field.grid_points[i].tolist(),
                    'sigma1': tensor.sigma1,
                    'sigma2': tensor.sigma2,
                    'sigma3': tensor.sigma3,
                    'azimuth1': tensor.azimuth1,
                    'plunge1': tensor.plunge1,
                    'azimuth2': tensor.azimuth2,
                    'plunge2': tensor.plunge2,
                    'azimuth3': tensor.azimuth3,
                    'plunge3': tensor.plunge3,
                    'stress_ratio': tensor.stress_ratio,
                    'regime': tensor.regime,
                    'confidence': tensor.confidence
                })
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
        except Exception as e:
            warnings.warn(f"Failed to export stress field: {e}")


# Example usage and testing
def main():
    """Example usage of stress field mapper"""
    
    # Create configuration
    config = StressFieldConfig(
        inversion_method="michael",
        grid_spacing=0.05,  # 0.05 degrees
        smoothing_radius=25.0,  # 25 km
        minimum_events=3,
        bootstrap_samples=200
    )
    
    # Initialize stress mapper
    mapper = StressInversionEngine(config)
    
    # Generate synthetic focal mechanisms
    focal_mechanisms = []
    np.random.seed(42)
    
    # Create a region of earthquakes
    region_center = (45.0, -120.0)  # lat, lon
    n_events = 50
    
    for i in range(n_events):
        # Random location within region
        lat = region_center[0] + np.random.uniform(-0.5, 0.5)
        lon = region_center[1] + np.random.uniform(-0.5, 0.5)
        depth = np.random.uniform(5, 25)  # km
        magnitude = np.random.uniform(3.0, 6.0)
        
        # Generate realistic focal mechanism
        # Assume strike-slip regime with some variation
        strike1 = np.random.uniform(0, 360)
        dip1 = np.random.uniform(70, 90)  # Near-vertical
        rake1 = np.random.uniform(160, 200) if np.random.random() > 0.5 else np.random.uniform(-20, 20)
        
        # Calculate auxiliary plane
        strike2 = (strike1 + 90) % 360
        dip2 = np.random.uniform(70, 90)
        rake2 = rake1 - 180 if rake1 > 0 else rake1 + 180
        
        # Random time
        timestamp = datetime.now() - timedelta(days=np.random.uniform(0, 365))
        
        fm = FocalMechanism(
            event_id=f"event_{i:03d}",
            latitude=lat,
            longitude=lon,
            depth=depth,
            magnitude=magnitude,
            strike1=strike1,
            dip1=dip1,
            rake1=rake1,
            strike2=strike2,
            dip2=dip2,
            rake2=rake2,
            timestamp=timestamp
        )
        
        focal_mechanisms.append(fm)
    
    print(f"Generated {len(focal_mechanisms)} synthetic focal mechanisms")
    
    # Add focal mechanisms to mapper
    mapper.add_focal_mechanisms(focal_mechanisms)
    
    # Define study region
    region_bounds = (44.5, 45.5, -120.5, -119.5)  # min_lat, max_lat, min_lon, max_lon
    
    print("Inverting regional stress field...")
    
    try:
        # Invert stress field
        stress_field = mapper.invert_stress_field(region_bounds, method='michael')
        
        print("\n=== STRESS FIELD RESULTS ===")
        print(f"Grid points: {len(stress_field.grid_points)}")
        print(f"Successful inversions: {stress_field.quality_metrics['successful_inversions']}")
        print(f"Failed inversions: {stress_field.quality_metrics['failed_inversions']}")
        
        # Analyze successful stress tensors
        successful_tensors = [t for t in stress_field.stress_tensors if t.confidence > 0.1]
        
        if successful_tensors:
            # Statistics
            azimuths = [t.azimuth1 for t in successful_tensors]
            regimes = [t.regime for t in successful_tensors]
            ratios = [t.stress_ratio for t in successful_tensors]
            
            print(f"\nStress Statistics:")
            print(f"Mean sigma1 azimuth: {np.mean(azimuths):.1f}°")
            print(f"Std sigma1 azimuth: {np.std(azimuths):.1f}°")
            print(f"Mean stress ratio: {np.mean(ratios):.2f}")
            
            # Regime distribution
            regime_counts = {}
            for regime in regimes:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            print(f"Stress regimes: {regime_counts}")
            
            # Calculate heterogeneity
            heterogeneity = mapper.calculate_stress_heterogeneity(stress_field)
            print(f"\nStress Field Heterogeneity:")
            print(f"Heterogeneity index: {heterogeneity['heterogeneity_index']:.3f}")
            print(f"Coherence length: {heterogeneity['coherence_length']:.1f} km")
            
            # Detect anomalies
            anomalies = mapper.detect_stress_anomalies(stress_field)
            print(f"Detected {len(anomalies)} stress anomalies")
            
            if anomalies:
                print("Anomalies:")
                for anomaly in anomalies[:3]:  # Show first 3
                    print(f"  {anomaly['type']} at {anomaly['location']}")
        
        # Temporal analysis
        if len(focal_mechanisms) > 10:
            print("\nPerforming temporal analysis...")
            
            # Create time windows (quarterly)
            start_time = min(fm.timestamp for fm in focal_mechanisms)
            end_time = max(fm.timestamp for fm in focal_mechanisms)
            
            time_windows = []
            current_time = start_time
            while current_time < end_time:
                window_end = current_time + timedelta(days=90)  # 3 months
                time_windows.append((current_time, min(window_end, end_time)))
                current_time = window_end
            
            if len(time_windows) > 1:
                temporal_analysis = mapper.analyze_temporal_evolution(time_windows)
                
                print(f"Analyzed {len(temporal_analysis['time_windows'])} time windows")
                if temporal_analysis['regime_changes']:
                    significant_changes = sum(1 for change in temporal_analysis['regime_changes'] 
                                            if change['significance'])
                    print(f"Significant stress changes: {significant_changes}")
        
        # Export results
        mapper.export_stress_field(stress_field, 'stress_field_results.json')
        print(f"\nResults exported to stress_field_results.json")
        
    except Exception as e:
        print(f"Stress field inversion failed: {e}")
    
    return mapper


if __name__ == "__main__":
    mapper = main()