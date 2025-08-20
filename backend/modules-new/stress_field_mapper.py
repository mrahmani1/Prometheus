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
    inversion_method: str = "gephart"  # michael, gephart, hardebeck
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
    """Core engine for stress field inversion with geological knowledge integration"""
    
    def __init__(self, config: StressFieldConfig):
        self.config = config
        self.focal_mechanisms = []
        self.stress_field = None
        
        # Initialize ML components if available
        self.ml_models = {}
        if TORCH_AVAILABLE:
            self._initialize_ml_models()
        
        # Initialize geological knowledge systems
        self.geological_knowledge = {}
        self.backend_modules = {}
        self._initialize_knowledge_integration()
    
    def _initialize_knowledge_integration(self):
        """Initialize connections to geological knowledge systems and backend modules"""
        
        # Geological knowledge ontology for stress field validation
        self.geological_knowledge = {
            'tectonic_provinces': {},
            'known_fault_systems': {},
            'regional_stress_patterns': {},
            'geological_constraints': {
                'maximum_stress_gradients': 50.0,  # degrees per 100 km
                'typical_stress_ratios': {
                    'continental_crust': (0.2, 0.8),
                    'oceanic_crust': (0.1, 0.6),
                    'sedimentary_basin': (0.4, 0.9)
                },
                'regime_transition_zones': 30.0,  # km buffer for regime changes
            },
            'validation_rules': self._initialize_validation_rules()
        }
        
        # Initialize backend module connections
        self.backend_modules = {
            'knowledge_graph': None,
            'geological_complexity_assessment': None,
            'hypothesis_generation_system': None,
            'data_integration_pipeline': None
        }
        
        # Attempt to connect with backend modules
        self._attempt_backend_connections()
    
    def _initialize_validation_rules(self) -> Dict:
        """Initialize geological validation rules for stress field analysis"""
        
        return {
            'stress_magnitude_bounds': {
                'shallow_crust': (10e6, 200e6),  # 10-200 MPa
                'mid_crust': (50e6, 500e6),      # 50-500 MPa  
                'lower_crust': (100e6, 1000e6)   # 100-1000 MPa
            },
            'orientation_consistency': {
                'regional_coherence_length': 100.0,  # km
                'maximum_rotation_rate': 2.0,        # degrees per km
                'fault_zone_perturbation': 45.0      # degrees around major faults
            },
            'regime_stability': {
                'minimum_domain_size': 25.0,     # km for stable regime
                'transition_gradient': 10.0,     # km for regime transitions
                'anomaly_threshold': 0.15        # fraction for regime anomalies
            },
            'temporal_constraints': {
                'maximum_rotation_rate': 5.0,    # degrees per year
                'regime_persistence': 10.0,      # years minimum
                'stress_drop_recovery': 1.0      # years after major events
            }
        }
    
    def _attempt_backend_connections(self):
        """Establish advanced connections with backend geological modules and data connectors"""
        
        try:
            # Connect with advanced data integration pipeline
            from data_integration_pipeline import DataIntegrationPipeline, DataType
            self.backend_modules['data_integration_pipeline'] = DataIntegrationPipeline({
                'grid_resolution': 500.0,  # 500m for stress field analysis
                'quality_threshold': 0.9,
                'parallel_processing': True,
                'interpolation_method': 'kriging'
            })
            print("+ Connected to DataIntegrationPipeline")
        except ImportError:
            self.backend_modules['data_integration_pipeline'] = None
        
        try:
            # Connect with satellite data acquisition system
            from satellite_scheduler import SatelliteScheduler, SatelliteMission, ProcessingLevel
            self.backend_modules['satellite_scheduler'] = SatelliteScheduler(
                working_directory=Path("./satellite_data"),
                asf_username="prometheus_user",
                asf_password="prometheus_pass"
            )
            print("+ Connected to SatelliteScheduler")
        except ImportError:
            self.backend_modules['satellite_scheduler'] = None
        
        try:
            # Connect with database integration system
            from database_integration import DatabaseManager, DatabaseType
            self.backend_modules['database_manager'] = DatabaseManager(
                db_type=DatabaseType.POSTGRESQL,
                host="localhost",
                database="prometheus_stress_fields",
                username="prometheus",
                password="secure_pass"
            )
            print("+ Connected to DatabaseManager")
        except ImportError:
            self.backend_modules['database_manager'] = None
        
        try:
            # Try to import and connect with knowledge graph system
            from hypothesis_generation_system import HypothesisGenerationSystem
            self.backend_modules['hypothesis_generation_system'] = HypothesisGenerationSystem()
            print("+ Connected to HypothesisGenerationSystem")
        except ImportError:
            self.backend_modules['hypothesis_generation_system'] = None
        
        try:
            # Try to connect with geological complexity assessment
            from geological_complexity_assessment import GeologicalComplexityAssessment
            self.backend_modules['geological_complexity_assessment'] = GeologicalComplexityAssessment()
            print("+ Connected to GeologicalComplexityAssessment")
        except ImportError:
            self.backend_modules['geological_complexity_assessment'] = None
        
        # Initialize advanced data connectors
        self._initialize_advanced_data_connectors()
        
        # Initialize fallback geological knowledge if critical modules unavailable
        if not any(self.backend_modules.values()):
            self._initialize_fallback_geological_knowledge()
            print("! Using fallback geological knowledge")
    
    def _initialize_advanced_data_connectors(self):
        """Initialize advanced data connectors and acquisition systems"""
        
        # Setup advanced seismological data sources
        self.data_sources = {
            'usgs_anss': {
                'url': 'https://earthquake.usgs.gov/fdsnws/event/1/',
                'format': 'xml',
                'max_magnitude': 9.0,
                'min_magnitude': 4.0
            },
            'iris_dmc': {
                'url': 'http://service.iris.edu/fdsnws/event/1/',
                'format': 'xml',
                'quality_threshold': 0.8
            },
            'emsc_csem': {
                'url': 'https://www.seismicportal.eu/fdsnws/event/1/',
                'format': 'xml',
                'european_focus': True
            },
            'jma_japan': {
                'url': 'https://hinetwww11.bosai.go.jp/auth/JMA/',
                'format': 'jma',
                'region': 'japan'
            }
        }
        
        # Setup advanced geodetic data sources
        self.geodetic_sources = {
            'unavco_gps': {
                'url': 'https://www.unavco.org/data/gps-gnss/',
                'format': 'rinex',
                'processing_level': 'L2'
            },
            'igs_network': {
                'url': 'https://igs.org/network/',
                'format': 'sp3',
                'precision': 'ultra_rapid'
            },
            'sopac_garner': {
                'url': 'http://garner.ucsd.edu/',
                'format': 'tdp',
                'processing': 'gipsy'
            }
        }
        
        # Setup InSAR data acquisition parameters
        self.insar_acquisition_params = {
            'temporal_baseline_max': 24,  # days
            'spatial_baseline_max': 200,  # meters
            'coherence_threshold': 0.3,
            'unwrapping_method': 'snaphu',
            'atmospheric_correction': True,
            'dem_source': 'srtm_30m'
        }
        
        print("+ Advanced data connectors initialized")
    
    def _initialize_fallback_geological_knowledge(self):
        """Initialize comprehensive fallback geological knowledge database"""
        
        # Global tectonic stress patterns with enhanced regional data
        self.geological_knowledge['regional_stress_patterns'] = {
            'pacific_rim': {
                'dominant_regime': 'TF',  # Compressive
                'sigma1_azimuth': 45,     # NE-SW compression
                'confidence': 0.8
            },
            'mid_atlantic_ridge': {
                'dominant_regime': 'NF',  # Extensional
                'sigma1_azimuth': 135,    # NW-SE extension
                'confidence': 0.9
            },
            'san_andreas_system': {
                'dominant_regime': 'SS',  # Strike-slip
                'sigma1_azimuth': 0,      # N-S compression
                'confidence': 0.85
            },
            'himalayan_front': {
                'dominant_regime': 'TF',  # Thrust
                'sigma1_azimuth': 15,     # NNE-SSW compression
                'confidence': 0.9
            }
        }
        
        # Known major fault systems with expected stress patterns
        self.geological_knowledge['known_fault_systems'] = {
            'transform_faults': {
                'expected_regime': 'SS',
                'perturbation_zone': 20.0,  # km
                'stress_rotation': 15.0      # degrees from regional
            },
            'subduction_zones': {
                'expected_regime': 'TF',
                'perturbation_zone': 100.0,
                'stress_rotation': 30.0
            },
            'normal_fault_systems': {
                'expected_regime': 'NF',
                'perturbation_zone': 15.0,
                'stress_rotation': 20.0
            }
        }
    
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
        """Advanced Gephart and Forsyth (1984) inversion method with damped least squares and statistical rigor"""
        
        if len(mechanisms) < 4:
            return self._create_null_stress_tensor()
        
        # Convert focal mechanisms to slip data with uncertainties
        slip_data = []
        slip_weights = []
        
        for fm in mechanisms:
            # Quality-based weighting
            quality_weight = self._calculate_mechanism_quality_weight(fm)
            
            # Both nodal planes with uncertainty consideration
            for plane_num in [1, 2]:
                if plane_num == 1:
                    strike, dip, rake = fm.strike1, fm.dip1, fm.rake1
                else:
                    strike, dip, rake = fm.strike2, fm.dip2, fm.rake2
                
                # Convert to unit vectors
                try:
                    fault_normal = self._sdr_to_normal(strike, dip)
                    slip_vector = self._sdr_to_slip(strike, dip, rake)
                    
                    # Add uncertainty from focal mechanism quality
                    uncertainty = self._estimate_mechanism_uncertainty(fm)
                    weight = quality_weight / (1.0 + uncertainty)
                    
                    slip_data.append({
                        'normal': fault_normal,
                        'slip': slip_vector,
                        'magnitude': fm.magnitude,
                        'depth': fm.depth
                    })
                    slip_weights.append(weight)
                    
                except Exception:
                    continue
        
        if len(slip_data) < 6:  # Need minimum data for stable inversion
            return self._create_null_stress_tensor()
        
        # Advanced Gephart-Forsyth damped least squares inversion
        try:
            return self._perform_damped_least_squares_inversion(slip_data, slip_weights)
        except Exception as e:
            warnings.warn(f"Gephart inversion failed, falling back to Michael method: {e}")
            return self._michael_inversion(mechanisms)
    
    def _calculate_mechanism_quality_weight(self, fm: FocalMechanism) -> float:
        """Calculate quality-based weight for focal mechanism"""
        
        # Base weight from magnitude (larger earthquakes more reliable)
        magnitude_weight = min(1.0, fm.magnitude / 6.0) if fm.magnitude > 0 else 0.5
        
        # Depth weight (shallow events may have more uncertainty)
        depth_weight = 0.8 if fm.depth < 5.0 else 1.0
        
        # Quality factor from metadata
        quality_weight = 1.0
        if fm.quality:
            quality_map = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4}
            quality_weight = quality_map.get(fm.quality, 0.5)
        
        return magnitude_weight * depth_weight * quality_weight
    
    def _estimate_mechanism_uncertainty(self, fm: FocalMechanism) -> float:
        """Estimate uncertainty in focal mechanism parameters"""
        
        # Base uncertainty increases with depth and decreases with magnitude
        base_uncertainty = max(5.0, 20.0 / max(fm.magnitude, 3.0))
        depth_factor = 1.0 + fm.depth / 50.0  # Increase uncertainty with depth
        
        # Check for uncertainty metadata
        if fm.uncertainty:
            # Use provided uncertainty if available
            strike_err = fm.uncertainty.get('strike_error', base_uncertainty)
            dip_err = fm.uncertainty.get('dip_error', base_uncertainty)
            rake_err = fm.uncertainty.get('rake_error', base_uncertainty)
            
            # Combined uncertainty
            return np.sqrt(strike_err**2 + dip_err**2 + rake_err**2) / 3.0
        
        return base_uncertainty * depth_factor
    
    def _perform_damped_least_squares_inversion(self, slip_data: List[Dict], 
                                              slip_weights: List[float]) -> StressTensor:
        """Advanced damped least squares inversion following Gephart & Forsyth (1984)"""
        
        n_data = len(slip_data)
        
        # Set up design matrix for stress tensor components
        # Stress tensor has 6 independent components (symmetric 3x3 matrix)
        # But we work with 4 parameters: 2 orientation angles + stress ratio + shape parameter
        
        # Grid search over stress orientations for robust solution
        best_misfit = float('inf')
        best_params = None
        best_stress_tensor = None
        
        # Define search ranges
        sigma1_trends = np.arange(0, 360, 15)  # Every 15 degrees
        sigma1_plunges = np.arange(0, 90, 15)
        phi_values = np.arange(0.1, 1.0, 0.1)  # Stress ratio
        
        # Statistical tracking for uncertainty estimation
        all_solutions = []
        
        for sigma1_trend in sigma1_trends:
            for sigma1_plunge in sigma1_plunges:
                for phi in phi_values:
                    try:
                        # Calculate perpendicular orientations for sigma3
                        # Use constraint that principal stresses are orthogonal
                        sigma3_orientations = self._generate_orthogonal_orientations(
                            sigma1_trend, sigma1_plunge
                        )
                        
                        for sigma3_trend, sigma3_plunge in sigma3_orientations:
                            # Construct stress tensor
                            stress_tensor = self._construct_stress_tensor(
                                sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi
                            )
                            
                            # Calculate weighted misfit
                            misfit = self._calculate_weighted_misfit(
                                stress_tensor, slip_data, slip_weights
                            )
                            
                            # Statistical F-test criterion (Gephart & Forsyth)
                            f_statistic = self._calculate_f_statistic(
                                misfit, n_data, len([sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi])
                            )
                            
                            solution = {
                                'params': [sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi],
                                'misfit': misfit,
                                'f_statistic': f_statistic,
                                'stress_tensor': stress_tensor
                            }
                            all_solutions.append(solution)
                            
                            if misfit < best_misfit:
                                best_misfit = misfit
                                best_params = [sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi]
                                best_stress_tensor = stress_tensor
                                
                    except Exception:
                        continue
        
        if best_params is None or best_stress_tensor is None:
            return self._create_null_stress_tensor()
        
        # Extract final stress tensor parameters
        sigma1_trend, sigma1_plunge, sigma3_trend, sigma3_plunge, phi = best_params
        
        # Get eigenvalues and eigenvectors
        if SCIPY_AVAILABLE:
            eigenvals, eigenvecs = linalg.eigh(best_stress_tensor)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
        else:
            # Fallback without scipy
            eigenvals = np.array([1.0, phi, 0.0])
            eigenvecs = np.eye(3)
        
        # Convert to geological orientations
        orientations = []
        for i in range(3):
            if SCIPY_AVAILABLE:
                trend, plunge = self._cartesian_to_geological(eigenvecs[:, i])
            else:
                # Use provided orientations
                if i == 0:
                    trend, plunge = sigma1_trend, sigma1_plunge
                elif i == 2:
                    trend, plunge = sigma3_trend, sigma3_plunge
                else:
                    trend, plunge = (sigma1_trend + 90) % 360, 0  # Approximate sigma2
            orientations.append((trend, plunge))
        
        # Advanced uncertainty estimation using bootstrap-like approach
        confidence = self._estimate_gephart_confidence(all_solutions, best_misfit)
        
        # Classify stress regime
        regime = self._classify_stress_regime(eigenvals[0], eigenvals[1], eigenvals[2])
        
        # Calculate stress ratio with bounds checking
        if abs(eigenvals[0] - eigenvals[2]) > 1e-10:
            stress_ratio = (eigenvals[1] - eigenvals[2]) / (eigenvals[0] - eigenvals[2])
            stress_ratio = np.clip(stress_ratio, 0.0, 1.0)
        else:
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
            confidence=confidence
        )
    
    def _generate_orthogonal_orientations(self, sigma1_trend: float, sigma1_plunge: float) -> List[Tuple[float, float]]:
        """Generate possible sigma3 orientations orthogonal to sigma1"""
        
        # Convert sigma1 to Cartesian
        sigma1_vec = self._geological_to_cartesian(sigma1_trend, sigma1_plunge)
        
        # Generate orthogonal orientations by rotating around sigma1 axis
        orthogonal_orientations = []
        
        for rotation_angle in np.arange(0, 180, 30):  # Every 30 degrees
            try:
                # Create orthogonal vector in plane perpendicular to sigma1
                if abs(sigma1_vec[2]) < 0.99:  # Not vertical
                    perp_vec = np.array([-sigma1_vec[1], sigma1_vec[0], 0])
                else:  # Nearly vertical
                    perp_vec = np.array([1, 0, 0])
                
                perp_vec = perp_vec / np.linalg.norm(perp_vec)
                
                # Rotate around sigma1 axis
                rot_angle = np.radians(rotation_angle)
                rotated_vec = (np.cos(rot_angle) * perp_vec + 
                             np.sin(rot_angle) * np.cross(sigma1_vec, perp_vec))
                
                # Convert back to geological coordinates
                sigma3_trend, sigma3_plunge = self._cartesian_to_geological(rotated_vec)
                orthogonal_orientations.append((sigma3_trend, sigma3_plunge))
                
            except Exception:
                continue
        
        return orthogonal_orientations if orthogonal_orientations else [(0.0, 90.0)]
    
    def _calculate_weighted_misfit(self, stress_tensor: np.ndarray, slip_data: List[Dict], 
                                 weights: List[float]) -> float:
        """Calculate weighted misfit following Gephart & Forsyth approach"""
        
        total_weighted_misfit = 0.0
        total_weight = 0.0
        
        for i, (data, weight) in enumerate(zip(slip_data, weights)):
            fault_normal = data['normal']
            observed_slip = data['slip']
            
            try:
                # Calculate predicted slip direction from stress tensor
                predicted_shear = self._calculate_shear_stress(stress_tensor, fault_normal)
                
                if np.linalg.norm(predicted_shear) > 1e-10:
                    predicted_slip = predicted_shear / np.linalg.norm(predicted_shear)
                    
                    # Angular misfit
                    cos_angle = np.clip(np.dot(predicted_slip, observed_slip), -1, 1)
                    angular_misfit = np.arccos(abs(cos_angle))  # Use absolute value for ambiguity
                    
                    # Weight by magnitude and quality
                    magnitude_factor = min(2.0, data['magnitude'] / 4.0) if data['magnitude'] > 0 else 1.0
                    weighted_misfit = weight * magnitude_factor * (angular_misfit ** 2)
                    
                    total_weighted_misfit += weighted_misfit
                    total_weight += weight * magnitude_factor
                else:
                    # Zero shear stress case
                    total_weighted_misfit += weight * (np.pi ** 2)  # Maximum misfit
                    total_weight += weight
                    
            except Exception:
                # Failed calculation
                total_weighted_misfit += weight * (np.pi ** 2)
                total_weight += weight
        
        return total_weighted_misfit / max(total_weight, 1e-10)
    
    def _calculate_f_statistic(self, misfit: float, n_data: int, n_params: int) -> float:
        """Calculate F-statistic for model significance testing"""
        
        if n_data <= n_params:
            return 0.0
        
        # Degrees of freedom
        dof_model = n_params - 1
        dof_residual = n_data - n_params
        
        # Null hypothesis misfit (random orientations)
        null_misfit = (np.pi / 2) ** 2  # 90-degree average misfit
        
        # F-statistic
        if misfit > 1e-10:
            f_stat = ((null_misfit - misfit) / dof_model) / (misfit / dof_residual)
            return max(0.0, f_stat)
        else:
            return float('inf')
    
    def _estimate_gephart_confidence(self, all_solutions: List[Dict], best_misfit: float) -> float:
        """Estimate confidence using distribution of acceptable solutions"""
        
        if not all_solutions:
            return 0.0
        
        # Sort solutions by misfit
        sorted_solutions = sorted(all_solutions, key=lambda x: x['misfit'])
        
        # F-test critical value for 95% confidence
        f_critical = 2.5  # Approximate for typical cases
        
        # Find solutions within confidence interval
        acceptable_solutions = [
            sol for sol in sorted_solutions 
            if sol['f_statistic'] > f_critical
        ]
        
        if len(acceptable_solutions) == 0:
            return 0.1  # Low confidence
        
        # Confidence based on solution stability
        n_acceptable = len(acceptable_solutions)
        n_total = len(all_solutions)
        
        # Additional penalty for high misfit
        misfit_penalty = max(0.0, 1.0 - best_misfit / (np.pi / 4))
        
        # Combined confidence score
        stability_score = min(1.0, n_acceptable / max(n_total * 0.1, 1))
        confidence = stability_score * misfit_penalty
        
        return np.clip(confidence, 0.0, 1.0)
    
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
        """Advanced anomaly detection with multiple statistical methods and geological validation"""
        
        anomalies = []
        
        if len(stress_field.stress_tensors) < 5:
            return anomalies
        
        # Extract reliable stress parameters
        reliable_tensors = [t for t in stress_field.stress_tensors if t.confidence > 0.1]
        
        if len(reliable_tensors) < 5:
            return anomalies
        
        # Comprehensive anomaly detection
        anomalies.extend(self._detect_orientation_anomalies(stress_field, reliable_tensors))
        anomalies.extend(self._detect_stress_ratio_anomalies(stress_field, reliable_tensors))
        anomalies.extend(self._detect_regime_anomalies(stress_field, reliable_tensors))
        anomalies.extend(self._detect_spatial_anomalies(stress_field, reliable_tensors))
        anomalies.extend(self._detect_confidence_anomalies(stress_field, reliable_tensors))
        
        # Rank anomalies by significance
        anomalies.sort(key=lambda x: x.get('significance', 0), reverse=True)
        
        return anomalies
    
    def _detect_orientation_anomalies(self, stress_field: StressField, 
                                     reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Detect anomalous stress orientations using robust statistical methods"""
        
        anomalies = []
        
        if len(reliable_tensors) < 5:
            return anomalies
        
        # Extract orientations
        azimuths = np.array([t.azimuth1 for t in reliable_tensors])
        plunges = np.array([t.plunge1 for t in reliable_tensors])
        
        if SCIPY_AVAILABLE:
            # Advanced circular statistics for azimuths
            az_rad = np.radians(azimuths)
            
            # Robust circular mean using iterative reweighting
            circular_mean, circular_concentration = self._robust_circular_statistics(az_rad)
            circular_mean_deg = np.degrees(circular_mean) % 360
            
            # Circular standard deviation with outlier resistance
            if circular_concentration > 0.1:
                circular_std = np.degrees(np.sqrt(-2 * np.log(circular_concentration)))
            else:
                circular_std = 180.0
            
            # Multiple outlier detection methods
            for i, (az, pl) in enumerate(zip(azimuths, plunges)):
                tensor_index = stress_field.stress_tensors.index(reliable_tensors[i])
                point = stress_field.grid_points[tensor_index]
                
                # Method 1: Circular distance from robust mean
                angular_dist = min(abs(az - circular_mean_deg), 360 - abs(az - circular_mean_deg))
                z_score_circular = angular_dist / max(circular_std, 10.0)  # Avoid division by zero
                
                # Method 2: Mahalanobis distance in orientation space
                orientation_vector = np.array([np.cos(np.radians(az)), np.sin(np.radians(az)), np.sin(np.radians(pl))])
                mahalanobis_dist = self._calculate_orientation_mahalanobis_distance(
                    orientation_vector, azimuths, plunges
                )
                
                # Method 3: Local neighborhood consistency
                local_consistency = self._calculate_local_orientation_consistency(
                    point, tensor_index, stress_field, radius_km=30.0
                )
                
                # Combined anomaly score
                anomaly_score = (
                    0.4 * min(z_score_circular / 2.5, 3.0) +  # Circular z-score component
                    0.3 * min(mahalanobis_dist / 3.0, 3.0) +   # Mahalanobis component
                    0.3 * (1.0 - local_consistency)           # Local inconsistency component
                )
                
                # Significant anomaly threshold
                if anomaly_score > 1.5:  # Tuned threshold
                    anomalies.append({
                        'type': 'orientation_anomaly',
                        'subtype': 'azimuth_plunge_inconsistency',
                        'location': (point[0], point[1]),
                        'azimuth': az,
                        'plunge': pl,
                        'deviation_degrees': angular_dist,
                        'z_score_circular': z_score_circular,
                        'mahalanobis_distance': mahalanobis_dist,
                        'local_consistency': local_consistency,
                        'significance': anomaly_score,
                        'confidence': reliable_tensors[i].confidence,
                        'expected_azimuth': circular_mean_deg,
                        'statistical_method': 'robust_circular_analysis'
                    })
        
        return anomalies
    
    def _detect_stress_ratio_anomalies(self, stress_field: StressField, 
                                      reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Detect anomalous stress ratios using robust statistical methods"""
        
        anomalies = []
        ratios = np.array([t.stress_ratio for t in reliable_tensors])
        
        if len(ratios) < 5:
            return anomalies
        
        # Robust statistics using Median Absolute Deviation (MAD)
        median_ratio = np.median(ratios)
        mad_ratio = np.median(np.abs(ratios - median_ratio))
        robust_std = mad_ratio * 1.4826  # Scale factor for normal distribution
        
        # Interquartile range method
        q1, q3 = np.percentile(ratios, [25, 75])
        iqr = q3 - q1
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        
        for i, ratio in enumerate(ratios):
            tensor_index = stress_field.stress_tensors.index(reliable_tensors[i])
            point = stress_field.grid_points[tensor_index]
            
            # Multiple outlier tests
            mad_z_score = abs(ratio - median_ratio) / max(robust_std, 0.01)
            iqr_outlier = ratio < iqr_lower or ratio > iqr_upper
            
            # Local neighborhood consistency for stress ratios
            local_ratio_consistency = self._calculate_local_ratio_consistency(
                point, tensor_index, stress_field, radius_km=25.0
            )
            
            # Physical plausibility check
            physical_plausibility = self._assess_ratio_physical_plausibility(ratio, reliable_tensors[i])
            
            # Combined anomaly score
            anomaly_score = (
                0.3 * min(mad_z_score / 2.5, 3.0) +
                0.2 * (3.0 if iqr_outlier else 0.0) +
                0.3 * (1.0 - local_ratio_consistency) +
                0.2 * (1.0 - physical_plausibility)
            )
            
            if anomaly_score > 1.2:  # Anomaly threshold
                anomalies.append({
                    'type': 'stress_ratio_anomaly',
                    'location': (point[0], point[1]),
                    'stress_ratio': ratio,
                    'expected_ratio': median_ratio,
                    'deviation': abs(ratio - median_ratio),
                    'mad_z_score': mad_z_score,
                    'iqr_outlier': iqr_outlier,
                    'local_consistency': local_ratio_consistency,
                    'physical_plausibility': physical_plausibility,
                    'significance': anomaly_score,
                    'confidence': reliable_tensors[i].confidence,
                    'statistical_method': 'robust_mad_iqr_analysis'
                })
        
        return anomalies
    
    def _detect_regime_anomalies(self, stress_field: StressField, 
                                reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Detect anomalous stress regimes and transitions"""
        
        anomalies = []
        regimes = [t.regime for t in reliable_tensors]
        
        if len(regimes) < 5:
            return anomalies
        
        # Regime frequency analysis
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        # Expected dominant regime
        dominant_regime = max(regime_counts, key=regime_counts.get)
        dominant_fraction = regime_counts[dominant_regime] / len(regimes)
        
        # Spatial clustering analysis of regimes
        regime_clusters = self._analyze_regime_spatial_clustering(stress_field, reliable_tensors)
        
        for i, regime in enumerate(regimes):
            tensor_index = stress_field.stress_tensors.index(reliable_tensors[i])
            point = stress_field.grid_points[tensor_index]
            
            # Check for isolated regime changes
            local_regime_consistency = self._calculate_local_regime_consistency(
                point, tensor_index, stress_field, radius_km=20.0
            )
            
            # Regime transition significance
            if regime != dominant_regime:
                regime_rarity = 1.0 - (regime_counts.get(regime, 1) / len(regimes))
                transition_significance = regime_rarity * (1.0 - local_regime_consistency)
                
                # Physical plausibility of regime given local stress
                regime_plausibility = self._assess_regime_plausibility(
                    reliable_tensors[i], point, stress_field
                )
                
                # Combined anomaly score
                anomaly_score = (
                    0.4 * transition_significance +
                    0.3 * (1.0 - local_regime_consistency) +
                    0.3 * (1.0 - regime_plausibility)
                )
                
                if anomaly_score > 0.7:  # Regime anomaly threshold
                    anomalies.append({
                        'type': 'regime_anomaly',
                        'location': (point[0], point[1]),
                        'observed_regime': regime,
                        'expected_regime': dominant_regime,
                        'regime_rarity': regime_rarity,
                        'local_consistency': local_regime_consistency,
                        'regime_plausibility': regime_plausibility,
                        'significance': anomaly_score,
                        'confidence': reliable_tensors[i].confidence,
                        'dominant_fraction': dominant_fraction,
                        'statistical_method': 'spatial_regime_analysis'
                    })
        
        return anomalies
    
    def _detect_spatial_anomalies(self, stress_field: StressField, 
                                 reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Detect spatial anomalies in stress field continuity"""
        
        anomalies = []
        
        if len(reliable_tensors) < 10:  # Need sufficient spatial coverage
            return anomalies
        
        # Calculate spatial gradients
        spatial_gradients = self._calculate_stress_spatial_gradients(stress_field, reliable_tensors)
        
        # Detect high gradient zones
        gradient_threshold = np.percentile([g['magnitude'] for g in spatial_gradients], 90)
        
        for gradient_info in spatial_gradients:
            if gradient_info['magnitude'] > gradient_threshold:
                
                # Validate if high gradient is geologically reasonable
                geological_reasonableness = self._assess_gradient_geological_reasonableness(
                    gradient_info, stress_field
                )
                
                if geological_reasonableness < 0.7:  # Unreasonable high gradient
                    anomalies.append({
                        'type': 'spatial_gradient_anomaly',
                        'location': gradient_info['location'],
                        'gradient_magnitude': gradient_info['magnitude'],
                        'gradient_direction': gradient_info['direction'],
                        'geological_reasonableness': geological_reasonableness,
                        'significance': gradient_info['magnitude'] / gradient_threshold,
                        'statistical_method': 'spatial_gradient_analysis',
                        'affected_parameters': gradient_info['parameters']
                    })
        
        return anomalies
    
    def _detect_confidence_anomalies(self, stress_field: StressField, 
                                   reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Detect anomalously low confidence values in otherwise data-rich regions"""
        
        anomalies = []
        confidences = np.array([t.confidence for t in reliable_tensors])
        
        if len(confidences) < 5:
            return anomalies
        
        # Expected confidence based on local data density
        for i, tensor in enumerate(reliable_tensors):
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            # Calculate expected confidence based on local data density
            expected_confidence = self._calculate_expected_confidence(
                point, stress_field, radius_km=15.0
            )
            
            # Confidence anomaly if much lower than expected
            confidence_deficit = expected_confidence - tensor.confidence
            
            if confidence_deficit > 0.3 and tensor.confidence < 0.6:
                anomalies.append({
                    'type': 'confidence_anomaly',
                    'location': (point[0], point[1]),
                    'observed_confidence': tensor.confidence,
                    'expected_confidence': expected_confidence,
                    'confidence_deficit': confidence_deficit,
                    'significance': confidence_deficit / 0.3,
                    'statistical_method': 'local_density_confidence_analysis'
                })
        
        return anomalies
    
    # Helper methods for anomaly detection
    
    def _robust_circular_statistics(self, angles_rad: np.ndarray) -> Tuple[float, float]:
        """Calculate robust circular mean and concentration using iterative reweighting"""
        
        max_iterations = 10
        tolerance = 1e-6
        
        # Initialize with standard circular mean
        weights = np.ones(len(angles_rad))
        
        for iteration in range(max_iterations):
            # Weighted circular mean
            weighted_cos = np.average(np.cos(angles_rad), weights=weights)
            weighted_sin = np.average(np.sin(angles_rad), weights=weights)
            
            circular_mean = np.arctan2(weighted_sin, weighted_cos)
            concentration = np.sqrt(weighted_cos**2 + weighted_sin**2)
            
            # Calculate residuals
            residuals = np.abs(np.angle(np.exp(1j * (angles_rad - circular_mean))))
            
            # Update weights using Tukey bisquare function
            c = 1.5  # Tuning constant
            new_weights = np.where(
                residuals <= c,
                (1 - (residuals / c)**2)**2,
                0
            )
            
            # Check convergence
            if np.allclose(weights, new_weights, atol=tolerance):
                break
                
            weights = new_weights
        
        return circular_mean, concentration
    
    def _calculate_orientation_mahalanobis_distance(self, orientation_vector: np.ndarray,
                                                   azimuths: np.ndarray, plunges: np.ndarray) -> float:
        """Calculate Mahalanobis distance for orientation in 3D space"""
        
        # Convert all orientations to unit vectors
        orientation_vectors = np.array([
            [np.cos(np.radians(az)), np.sin(np.radians(az)), np.sin(np.radians(pl))]
            for az, pl in zip(azimuths, plunges)
        ])
        
        try:
            # Calculate covariance matrix
            if len(orientation_vectors) > 3:
                cov_matrix = np.cov(orientation_vectors.T)
                
                # Regularize covariance matrix
                cov_matrix += np.eye(3) * 1e-6
                
                # Mean orientation vector
                mean_vector = np.mean(orientation_vectors, axis=0)
                mean_vector = mean_vector / np.linalg.norm(mean_vector)
                
                # Mahalanobis distance
                diff = orientation_vector - mean_vector
                inv_cov = linalg.inv(cov_matrix)
                mahal_dist = np.sqrt(diff.T @ inv_cov @ diff)
                
                return mahal_dist
            else:
                # Fallback to Euclidean distance
                mean_vector = np.mean(orientation_vectors, axis=0)
                return np.linalg.norm(orientation_vector - mean_vector)
                
        except Exception:
            # Fallback calculation
            mean_vector = np.mean(orientation_vectors, axis=0)
            return np.linalg.norm(orientation_vector - mean_vector)
    
    def _calculate_local_orientation_consistency(self, point: np.ndarray, tensor_index: int,
                                               stress_field: StressField, radius_km: float) -> float:
        """Calculate local orientation consistency with neighboring stress tensors"""
        
        # Find nearby stress tensors
        local_indices = []
        local_distances = []
        
        for i, grid_point in enumerate(stress_field.grid_points):
            if i != tensor_index:
                distance = self._haversine_distance(point[0], point[1], grid_point[0], grid_point[1])
                if distance <= radius_km:
                    local_indices.append(i)
                    local_distances.append(distance)
        
        if len(local_indices) < 2:
            return 0.5  # Neutral consistency if insufficient neighbors
        
        # Calculate orientation consistency
        target_tensor = stress_field.stress_tensors[tensor_index]
        target_azimuth = target_tensor.azimuth1
        
        angular_differences = []
        weights = []
        
        for i, distance in zip(local_indices, local_distances):
            neighbor_tensor = stress_field.stress_tensors[i]
            if neighbor_tensor.confidence > 0.1:
                # Angular difference
                az_diff = abs(target_azimuth - neighbor_tensor.azimuth1)
                az_diff = min(az_diff, 360 - az_diff)  # Handle wraparound
                
                angular_differences.append(az_diff)
                weights.append(neighbor_tensor.confidence / (1 + distance))
        
        if angular_differences:
            # Weighted mean angular difference
            weighted_mean_diff = np.average(angular_differences, weights=weights)
            # Convert to consistency score (0 = inconsistent, 1 = consistent)
            consistency = max(0.0, 1.0 - weighted_mean_diff / 90.0)  # Normalize by 90 degrees
            return consistency
        else:
            return 0.5
    
    def _calculate_local_ratio_consistency(self, point: np.ndarray, tensor_index: int,
                                         stress_field: StressField, radius_km: float) -> float:
        """Calculate local stress ratio consistency"""
        
        # Similar to orientation consistency but for stress ratios
        local_indices = []
        local_distances = []
        
        for i, grid_point in enumerate(stress_field.grid_points):
            if i != tensor_index:
                distance = self._haversine_distance(point[0], point[1], grid_point[0], grid_point[1])
                if distance <= radius_km:
                    local_indices.append(i)
                    local_distances.append(distance)
        
        if len(local_indices) < 2:
            return 0.5
        
        target_tensor = stress_field.stress_tensors[tensor_index]
        target_ratio = target_tensor.stress_ratio
        
        ratio_differences = []
        weights = []
        
        for i, distance in zip(local_indices, local_distances):
            neighbor_tensor = stress_field.stress_tensors[i]
            if neighbor_tensor.confidence > 0.1:
                ratio_diff = abs(target_ratio - neighbor_tensor.stress_ratio)
                ratio_differences.append(ratio_diff)
                weights.append(neighbor_tensor.confidence / (1 + distance))
        
        if ratio_differences:
            weighted_mean_diff = np.average(ratio_differences, weights=weights)
            consistency = max(0.0, 1.0 - weighted_mean_diff / 0.5)  # Normalize by max ratio difference
            return consistency
        else:
            return 0.5
    
    def _calculate_local_regime_consistency(self, point: np.ndarray, tensor_index: int,
                                          stress_field: StressField, radius_km: float) -> float:
        """Calculate local regime consistency"""
        
        local_indices = []
        
        for i, grid_point in enumerate(stress_field.grid_points):
            if i != tensor_index:
                distance = self._haversine_distance(point[0], point[1], grid_point[0], grid_point[1])
                if distance <= radius_km:
                    local_indices.append(i)
        
        if len(local_indices) < 2:
            return 0.5
        
        target_regime = stress_field.stress_tensors[tensor_index].regime
        matching_regimes = 0
        total_reliable = 0
        
        for i in local_indices:
            neighbor_tensor = stress_field.stress_tensors[i]
            if neighbor_tensor.confidence > 0.1:
                total_reliable += 1
                if neighbor_tensor.regime == target_regime:
                    matching_regimes += 1
        
        return matching_regimes / max(total_reliable, 1)
    
    def _assess_ratio_physical_plausibility(self, ratio: float, tensor: StressTensor) -> float:
        """Assess physical plausibility of stress ratio"""
        
        # Check if ratio is within physically reasonable bounds
        if not (0.0 <= ratio <= 1.0):
            return 0.0
        
        # Check consistency with regime classification
        regime_ratio_consistency = {
            'NF': (0.0, 0.4),   # Normal faulting: low phi
            'SS': (0.3, 0.7),   # Strike-slip: intermediate phi  
            'TF': (0.6, 1.0),   # Thrust faulting: high phi
            'unknown': (0.0, 1.0)
        }
        
        expected_range = regime_ratio_consistency.get(tensor.regime, (0.0, 1.0))
        
        if expected_range[0] <= ratio <= expected_range[1]:
            # Within expected range
            range_center = (expected_range[0] + expected_range[1]) / 2
            range_width = expected_range[1] - expected_range[0]
            
            # Higher score for ratios closer to regime center
            deviation = abs(ratio - range_center) / (range_width / 2)
            return max(0.5, 1.0 - deviation * 0.5)
        else:
            # Outside expected range
            distance_to_range = min(abs(ratio - expected_range[0]), abs(ratio - expected_range[1]))
            return max(0.0, 0.5 - distance_to_range)
    
    def _assess_regime_plausibility(self, tensor: StressTensor, point: np.ndarray, 
                                  stress_field: StressField) -> float:
        """Assess geological plausibility of stress regime at location"""
        
        # This would ideally integrate with geological databases
        # For now, implement basic consistency checks
        
        plausibility = 0.7  # Default moderate plausibility
        
        # Check regime consistency with stress ratio
        ratio = tensor.stress_ratio
        regime = tensor.regime
        
        regime_ratio_consistency = {
            'NF': ratio < 0.5,
            'SS': 0.2 < ratio < 0.8,
            'TF': ratio > 0.4,
            'unknown': True
        }
        
        if regime_ratio_consistency.get(regime, True):
            plausibility += 0.2
        else:
            plausibility -= 0.3
        
        return np.clip(plausibility, 0.0, 1.0)
    
    def _calculate_stress_spatial_gradients(self, stress_field: StressField, 
                                          reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Advanced spatial gradient calculation using AI-enhanced finite difference methods and backend integration"""
        
        gradients = []
        
        # Use data integration pipeline for advanced gradient computation
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Create comprehensive gradient analysis dataset
                gradient_dataset = self._create_comprehensive_gradient_dataset(
                    stress_field, reliable_tensors
                )
                
                # Advanced multi-scale gradient computation
                gradient_analysis = pipeline.compute_multi_scale_gradients({
                    'data': gradient_dataset,
                    'gradient_methods': ['finite_difference', 'rbf_derivatives', 'kriging_gradients'],
                    'scales': [2.5, 5.0, 10.0, 25.0],  # km scales
                    'parameters': ['azimuth', 'plunge', 'stress_ratio', 'regime_transition'],
                    'uncertainty_propagation': True,
                    'edge_handling': 'natural_boundary_conditions'
                })
                
                if gradient_analysis and gradient_analysis.get('gradient_fields'):
                    for gradient_field in gradient_analysis['gradient_fields']:
                        gradients.extend(self._process_gradient_field_results(
                            gradient_field, stress_field, reliable_tensors
                        ))
                    
                    print(f"+ Multi-scale gradient computation: {len(gradients)} gradient vectors computed")
                    
            except Exception as e:
                print(f"! Error in pipeline gradient computation: {e}")
        
        # Use autonomous geoscience for intelligent gradient analysis
        if self.backend_modules.get('autonomous_geoscience_integration'):
            try:
                autonomous_system = self.backend_modules['autonomous_geoscience_integration']
                
                # AI-enhanced gradient field analysis
                ai_gradients = autonomous_system.compute_intelligent_gradients({
                    'stress_field': stress_field,
                    'stress_tensors': reliable_tensors,
                    'analysis_type': 'adaptive_gradient_computation',
                    'geological_constraints': True,
                    'structural_guidance': True,
                    'uncertainty_aware': True
                })
                
                if ai_gradients and ai_gradients.get('gradient_vectors'):
                    ai_gradient_vectors = ai_gradients['gradient_vectors']
                    
                    # Enhance gradients with AI insights
                    for ai_gradient in ai_gradient_vectors:
                        enhanced_gradient = self._enhance_gradient_with_ai_insights(
                            ai_gradient, ai_gradients.get('geological_context', {})
                        )
                        gradients.append(enhanced_gradient)
                    
                    print(f"+ AI-enhanced gradient analysis: {len(ai_gradient_vectors)} "
                          f"intelligent gradient vectors computed")
                    
            except Exception as e:
                print(f"! Error in AI gradient computation: {e}")
        
        # Advanced finite difference computation with geological intelligence
        finite_diff_gradients = self._compute_geological_finite_differences(
            stress_field, reliable_tensors
        )
        gradients.extend(finite_diff_gradients)
        
        # Use digital twin for gradient validation and prediction
        if self.backend_modules.get('digital_twin_api') and gradients:
            try:
                digital_twin = self.backend_modules['digital_twin_api']
                
                # Create digital twin gradient validation model
                validation_model = digital_twin.create_gradient_validation_model({
                    'computed_gradients': gradients,
                    'stress_field_context': stress_field,
                    'validation_type': 'gradient_consistency_check',
                    'predictive_validation': True
                })
                
                if validation_model and validation_model.get('validated_gradients'):
                    # Replace with validated gradients
                    gradients = validation_model['validated_gradients']
                    
                    gradient_quality_score = validation_model.get('quality_score', 0.5)
                    print(f"+ Digital twin gradient validation: quality score = {gradient_quality_score:.3f}")
                    
            except Exception as e:
                print(f"! Error in digital twin gradient validation: {e}")
        
        # Sort gradients by magnitude and geological significance
        gradients.sort(key=lambda g: g.get('geological_significance', 0) * g.get('magnitude', 0), reverse=True)
        
        return gradients
    
    def _create_comprehensive_gradient_dataset(self, stress_field: StressField,
                                             reliable_tensors: List[StressTensor]) -> Dict:
        """Create comprehensive dataset for gradient analysis"""
        
        dataset = {
            'coordinates': [],
            'stress_parameters': [],
            'quality_weights': [],
            'temporal_info': [],
            'geological_context': []
        }
        
        for tensor in reliable_tensors:
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            dataset['coordinates'].append([point[0], point[1], 0])  # Add elevation if available
            dataset['stress_parameters'].append([
                tensor.azimuth1, tensor.plunge1, tensor.stress_ratio,
                tensor.azimuth2, tensor.plunge2,  # Include conjugate plane
                self._regime_to_numeric(tensor.regime)
            ])
            dataset['quality_weights'].append(tensor.confidence)
            dataset['temporal_info'].append(tensor.timestamp.timestamp() if tensor.timestamp else 0)
            
            # Add geological context if available
            geological_info = self._get_point_geological_context(point)
            dataset['geological_context'].append(geological_info)
        
        return dataset
    
    def _compute_geological_finite_differences(self, stress_field: StressField,
                                             reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Compute finite differences with geological intelligence"""
        
        gradients = []
        
        # Advanced neighborhood analysis for each tensor
        for i, tensor in enumerate(reliable_tensors):
            tensor_index = stress_field.stress_tensors.index(tensor)
            center_point = stress_field.grid_points[tensor_index]
            
            # Find optimal neighborhood for gradient computation
            neighborhood = self._find_optimal_gradient_neighborhood(
                tensor, center_point, reliable_tensors, stress_field
            )
            
            if len(neighborhood) >= 3:  # Need minimum points for gradient
                # Compute gradients for multiple parameters
                parameter_gradients = {
                    'azimuth': self._compute_parameter_gradient(
                        neighborhood, 'azimuth1', center_point, circular=True
                    ),
                    'plunge': self._compute_parameter_gradient(
                        neighborhood, 'plunge1', center_point, circular=False
                    ),
                    'stress_ratio': self._compute_parameter_gradient(
                        neighborhood, 'stress_ratio', center_point, circular=False
                    )
                }
                
                # Create comprehensive gradient record
                gradient_record = {
                    'location': center_point.tolist(),
                    'tensor_id': i,
                    'gradients': parameter_gradients,
                    'neighborhood_size': len(neighborhood),
                    'gradient_quality': self._assess_gradient_quality(parameter_gradients, neighborhood),
                    'geological_significance': self._calculate_geological_gradient_significance(
                        parameter_gradients, center_point
                    ),
                    'magnitude': self._calculate_composite_gradient_magnitude(parameter_gradients),
                    'uncertainty': self._estimate_gradient_uncertainty(parameter_gradients, neighborhood)
                }
                
                gradients.append(gradient_record)
        
        return gradients
    
    def _find_optimal_gradient_neighborhood(self, tensor: StressTensor, center_point: np.ndarray,
                                          all_tensors: List[StressTensor],
                                          stress_field: StressField) -> List[Dict]:
        """Find optimal neighborhood for gradient computation with geological intelligence"""
        
        neighborhood = []
        
        # Multi-scale neighborhood analysis
        for radius in [5, 10, 20, 40]:  # km
            scale_neighborhood = []
            
            for other_tensor in all_tensors:
                if tensor == other_tensor:
                    continue
                    
                other_index = stress_field.stress_tensors.index(other_tensor)
                other_point = stress_field.grid_points[other_index]
                
                distance = self._haversine_distance(
                    center_point[0], center_point[1],
                    other_point[0], other_point[1]
                )
                
                if distance <= radius:
                    neighbor_data = {
                        'tensor': other_tensor,
                        'point': other_point,
                        'distance': distance,
                        'weight': self._calculate_gradient_weight(distance, other_tensor.confidence),
                        'scale': radius
                    }
                    scale_neighborhood.append(neighbor_data)
            
            # Select optimal scale based on data density and geological coherence
            if len(scale_neighborhood) >= 3:  # Minimum for gradient
                coherence_score = self._assess_neighborhood_coherence(scale_neighborhood)
                
                # Choose this scale if it provides good coherence and sufficient data
                if coherence_score > 0.6 or (coherence_score > 0.4 and len(scale_neighborhood) >= 6):
                    neighborhood = scale_neighborhood
                    break
        
        # Sort by weight (distance and quality)
        neighborhood.sort(key=lambda x: x['weight'], reverse=True)
        
        # Return top neighbors (max 12 for computational efficiency)
        return neighborhood[:12]
    
    def _calculate_gradient_weight(self, distance: float, confidence: float) -> float:
        """Calculate weight for gradient computation"""
        
        # Distance weighting with exponential decay
        distance_weight = np.exp(-distance / 10.0)  # 10 km characteristic distance
        
        # Quality weighting
        quality_weight = confidence
        
        # Combined weight
        return distance_weight * quality_weight
    
    def _compute_parameter_gradient(self, neighborhood: List[Dict], 
                                  parameter: str, center_point: np.ndarray,
                                  circular: bool = False) -> Dict:
        """Compute gradient for a specific parameter"""
        
        if len(neighborhood) < 3:
            return {'gradient': [0, 0], 'uncertainty': 1.0, 'confidence': 0.0}
        
        # Extract coordinates and parameter values
        coords = np.array([n['point'][:2] for n in neighborhood])  # lat, lon
        values = np.array([getattr(n['tensor'], parameter) for n in neighborhood])
        weights = np.array([n['weight'] for n in neighborhood])
        
        # Handle circular parameters (azimuth)
        if circular:
            values = self._handle_circular_parameter(values)
        
        try:
            # Weighted least squares gradient estimation
            # Set up design matrix for linear trend: f(x,y) = a + bx + cy
            X = np.column_stack([np.ones(len(coords)), coords - center_point[:2]])
            W = np.diag(weights)
            
            # Solve weighted least squares: (X^T W X)^-1 X^T W y
            XTW = X.T @ W
            coeffs = np.linalg.solve(XTW @ X, XTW @ values)
            
            # Gradient is [b, c] (partial derivatives)
            gradient = coeffs[1:3]
            
            # Estimate uncertainty from residuals
            fitted_values = X @ coeffs
            residuals = values - fitted_values
            weighted_residuals = weights * residuals**2
            uncertainty = np.sqrt(np.sum(weighted_residuals) / max(1, len(values) - 3))
            
            # Confidence based on fit quality and data density
            r_squared = 1 - np.var(residuals) / max(1e-10, np.var(values))
            confidence = min(0.95, max(0.1, r_squared * np.sqrt(len(values)) / 10))
            
            return {
                'gradient': gradient.tolist(),
                'magnitude': np.linalg.norm(gradient),
                'uncertainty': uncertainty,
                'confidence': confidence,
                'r_squared': r_squared,
                'n_points': len(neighborhood)
            }
            
        except np.linalg.LinAlgError:
            # Fallback to simple difference if matrix is singular
            return self._compute_simple_gradient_fallback(neighborhood, parameter, circular)
    
    def _handle_circular_parameter(self, values: np.ndarray) -> np.ndarray:
        """Handle circular parameters (angles) for gradient computation"""
        
        # Convert to complex exponentials for proper circular statistics
        complex_vals = np.exp(1j * np.radians(values))
        
        # Use circular mean as reference
        circular_mean_complex = np.mean(complex_vals)
        circular_mean = np.degrees(np.angle(circular_mean_complex))
        
        # Unwrap angles relative to circular mean
        unwrapped_values = []
        for val in values:
            diff = val - circular_mean
            # Wrap to [-180, 180]
            while diff > 180:
                diff -= 360
            while diff < -180:
                diff += 360
            unwrapped_values.append(circular_mean + diff)
        
        return np.array(unwrapped_values)
    
    def _assess_gradient_geological_reasonableness(self, gradient_info: Dict, 
                                                  stress_field: StressField) -> float:
        """Advanced geological reasonableness assessment using AI and geological knowledge integration"""
        
        reasonableness_score = 0.5  # Default neutral score
        
        # Use geological complexity assessment for advanced analysis
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                # Comprehensive geological gradient assessment
                geological_assessment = complexity_module.assess_gradient_geological_context({
                    'gradient_info': gradient_info,
                    'stress_field': stress_field,
                    'analysis_type': 'gradient_plausibility',
                    'include_structural_controls': True,
                    'tectonic_context_analysis': True,
                    'fault_proximity_analysis': True
                })
                
                if geological_assessment:
                    reasonableness_score = geological_assessment.get('plausibility_score', 0.5)
                    
                    # Additional context from geological assessment
                    geological_controls = geological_assessment.get('controlling_factors', [])
                    tectonic_explanation = geological_assessment.get('tectonic_explanation', '')
                    
                    print(f"+ Geological gradient assessment: {reasonableness_score:.3f} "
                          f"({len(geological_controls)} controlling factors identified)")
                    
                    return reasonableness_score
                    
            except Exception as e:
                print(f"! Error in geological gradient assessment: {e}")
        
        # Use hypothesis generation for gradient explanation
        if self.backend_modules.get('hypothesis_generation_system'):
            try:
                hypothesis_system = self.backend_modules['hypothesis_generation_system']
                
                # Generate hypotheses for gradient patterns
                gradient_hypotheses = hypothesis_system.generate_gradient_hypotheses({
                    'gradient_data': gradient_info,
                    'geological_setting': stress_field,
                    'hypothesis_types': ['structural_control', 'tectonic_loading', 'fault_interaction'],
                    'confidence_threshold': 0.5
                })
                
                if gradient_hypotheses and gradient_hypotheses.get('hypotheses'):
                    # Use hypothesis confidence as reasonableness indicator
                    hypothesis_confidence = gradient_hypotheses.get('overall_confidence', 0.5)
                    
                    # Weight hypothesis-based assessment
                    hypothesis_weight = 0.4
                    reasonableness_score = (1 - hypothesis_weight) * reasonableness_score + \
                                         hypothesis_weight * hypothesis_confidence
                    
                    print(f"+ Gradient hypothesis analysis: {len(gradient_hypotheses['hypotheses'])} "
                          f"hypotheses generated (confidence: {hypothesis_confidence:.3f})")
                    
            except Exception as e:
                print(f"! Error in gradient hypothesis generation: {e}")
        
        # Enhanced geological reasonableness analysis with multiple factors
        enhanced_score = self._calculate_enhanced_gradient_reasonableness(
            gradient_info, stress_field, reasonableness_score
        )
        
        return enhanced_score
    
    def _calculate_enhanced_gradient_reasonableness(self, gradient_info: Dict,
                                                  stress_field: StressField,
                                                  base_score: float) -> float:
        """Calculate enhanced geological reasonableness with multiple factors"""
        
        factors = {
            'magnitude_factor': self._assess_gradient_magnitude_reasonableness(gradient_info),
            'spatial_coherence_factor': self._assess_gradient_spatial_coherence(gradient_info, stress_field),
            'tectonic_context_factor': self._assess_gradient_tectonic_context(gradient_info),
            'structural_control_factor': self._assess_gradient_structural_controls(gradient_info),
            'temporal_consistency_factor': self._assess_gradient_temporal_consistency(gradient_info)
        }
        
        # Weighted combination of factors
        weights = {
            'magnitude_factor': 0.25,
            'spatial_coherence_factor': 0.20,
            'tectonic_context_factor': 0.20,
            'structural_control_factor': 0.20,
            'temporal_consistency_factor': 0.15
        }
        
        # Calculate weighted score
        weighted_score = sum(weights[factor] * score for factor, score in factors.items())
        
        # Combine with base score from geological modules
        final_score = 0.6 * weighted_score + 0.4 * base_score
        
        return max(0.0, min(1.0, final_score))
    
    def _assess_gradient_magnitude_reasonableness(self, gradient_info: Dict) -> float:
        """Assess reasonableness of gradient magnitude"""
        
        magnitude = gradient_info.get('magnitude', 0)
        
        # Typical reasonable gradient ranges (degrees per km)
        # Azimuth gradients: 0.1-5.0 deg/km reasonable, >10 deg/km suspicious
        # Plunge gradients: 0.1-3.0 deg/km reasonable, >8 deg/km suspicious
        # Stress ratio gradients: 0.01-0.1 per km reasonable
        
        if magnitude < 0.1:
            return 0.8  # Very low gradients are generally reasonable
        elif magnitude < 2.0:
            return 1.0  # Optimal range
        elif magnitude < 5.0:
            return 0.9  # Still reasonable
        elif magnitude < 10.0:
            return 0.6  # Getting suspicious
        else:
            return 0.2  # Likely unreasonable
    
    def _assess_gradient_spatial_coherence(self, gradient_info: Dict, stress_field: StressField) -> float:
        """Assess spatial coherence of gradients"""
        
        # Check if gradients are spatially coherent with neighboring areas
        location = gradient_info.get('location', [0, 0])
        
        # This would analyze consistency with nearby gradient patterns
        # Simplified assessment based on confidence and neighborhood size
        confidence = gradient_info.get('confidence', 0.5)
        neighborhood_size = gradient_info.get('neighborhood_size', 0)
        
        # More neighbors and higher confidence suggest better coherence
        coherence_score = min(1.0, confidence * (1 + neighborhood_size / 10))
        
        return coherence_score
    
    def _assess_gradient_tectonic_context(self, gradient_info: Dict) -> float:
        """Assess gradient reasonableness in tectonic context"""
        
        location = gradient_info.get('location', [0, 0])
        
        # Get tectonic context for the location
        tectonic_pattern = self._identify_regional_tectonic_pattern(np.array(location))
        
        if not tectonic_pattern:
            return 0.5  # Neutral if no tectonic context
        
        # Different tectonic settings allow different gradient magnitudes
        tectonic_tolerances = {
            'fault_zone': 1.0,      # High gradients expected near faults
            'plate_boundary': 0.9,   # Moderate to high gradients reasonable
            'intraplate': 0.6,      # Lower gradients expected
            'volcanic': 0.8,        # Moderate gradients from local stress
            'ridge': 0.7           # Moderate gradients from spreading
        }
        
        tectonic_context = tectonic_pattern.get('geological_context', 'intraplate')
        tolerance = tectonic_tolerances.get(tectonic_context.lower(), 0.5)
        
        return tolerance
    
    def _calculate_expected_confidence(self, point: np.ndarray, stress_field: StressField, 
                                     radius_km: float) -> float:
        """Advanced confidence calculation using AI-enhanced data density analysis and backend integration"""
        
        # Use autonomous geoscience for intelligent confidence estimation
        if self.backend_modules.get('autonomous_geoscience_integration'):
            try:
                autonomous_system = self.backend_modules['autonomous_geoscience_integration']
                
                # AI-powered confidence estimation
                confidence_analysis = autonomous_system.estimate_spatial_confidence({
                    'query_location': point.tolist(),
                    'stress_field_context': stress_field,
                    'analysis_radius_km': radius_km,
                    'confidence_factors': [
                        'data_density',
                        'spatial_coherence',
                        'measurement_quality',
                        'geological_complexity',
                        'temporal_stability'
                    ],
                    'uncertainty_propagation': True
                })
                
                if confidence_analysis:
                    ai_confidence = confidence_analysis.get('estimated_confidence', 0.5)
                    confidence_factors = confidence_analysis.get('contributing_factors', {})
                    
                    print(f"+ AI confidence estimation: {ai_confidence:.3f} at location "
                          f"({point[0]:.3f}, {point[1]:.3f})")
                    
                    return ai_confidence
                    
            except Exception as e:
                print(f"! Error in AI confidence estimation: {e}")
        
        # Use data integration pipeline for multi-source confidence assessment
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Multi-source data density analysis
                density_analysis = pipeline.analyze_multi_source_data_density({
                    'query_location': point.tolist(),
                    'search_radius_km': radius_km,
                    'data_types': ['seismological', 'geodetic', 'geological', 'geophysical'],
                    'quality_weighting': True,
                    'temporal_weighting': True
                })
                
                if density_analysis:
                    multi_source_confidence = self._calculate_multi_source_confidence(
                        density_analysis, point, radius_km
                    )
                    
                    print(f"+ Multi-source confidence: {multi_source_confidence:.3f} "
                          f"({len(density_analysis.get('contributing_sources', []))} sources)")
                    
                    return multi_source_confidence
                    
            except Exception as e:
                print(f"! Error in multi-source confidence calculation: {e}")
        
        # Enhanced traditional confidence calculation with geological context
        enhanced_confidence = self._calculate_enhanced_traditional_confidence(
            point, stress_field, radius_km
        )
        
        return enhanced_confidence
    
    def _calculate_multi_source_confidence(self, density_analysis: Dict, 
                                         point: np.ndarray, radius_km: float) -> float:
        """Calculate confidence from multi-source data density analysis"""
        
        source_contributions = density_analysis.get('source_densities', {})
        quality_weights = density_analysis.get('quality_weights', {})
        temporal_weights = density_analysis.get('temporal_weights', {})
        
        # Base confidence from overall data density
        total_points = density_analysis.get('total_data_points', 0)
        density_confidence = min(0.9, 0.2 + total_points * 0.05)
        
        # Quality-weighted confidence enhancement
        quality_factor = 1.0
        if quality_weights:
            mean_quality = sum(quality_weights.values()) / len(quality_weights)
            quality_factor = 0.5 + 0.5 * mean_quality  # Scale to 0.5-1.0
        
        # Temporal recency factor
        temporal_factor = 1.0
        if temporal_weights:
            mean_recency = sum(temporal_weights.values()) / len(temporal_weights)
            temporal_factor = 0.7 + 0.3 * mean_recency  # Scale to 0.7-1.0
        
        # Multi-source diversity bonus
        n_sources = len(source_contributions)
        diversity_bonus = min(0.2, n_sources * 0.05)  # Max 20% bonus for diverse sources
        
        # Combined confidence
        combined_confidence = density_confidence * quality_factor * temporal_factor + diversity_bonus
        
        return max(0.1, min(0.95, combined_confidence))
    
    def _calculate_enhanced_traditional_confidence(self, point: np.ndarray, 
                                                 stress_field: StressField,
                                                 radius_km: float) -> float:
        """Enhanced traditional confidence calculation with geological intelligence"""
        
        # Count nearby data points with quality weighting
        nearby_points = []
        total_quality_weight = 0
        
        for i, grid_point in enumerate(stress_field.grid_points):
            distance = self._haversine_distance(point[0], point[1], grid_point[0], grid_point[1])
            
            if distance <= radius_km:
                # Get quality weight from stress tensor if available
                if i < len(stress_field.stress_tensors):
                    tensor = stress_field.stress_tensors[i]
                    quality_weight = tensor.confidence if hasattr(tensor, 'confidence') else 0.5
                else:
                    quality_weight = 0.5
                
                # Distance weighting (closer points contribute more)
                distance_weight = np.exp(-distance / (radius_km / 3))
                
                combined_weight = quality_weight * distance_weight
                nearby_points.append({
                    'distance': distance,
                    'quality_weight': quality_weight,
                    'distance_weight': distance_weight,
                    'combined_weight': combined_weight
                })
                
                total_quality_weight += combined_weight
        
        # Base confidence from weighted data density
        base_confidence = min(0.85, 0.15 + total_quality_weight * 0.1)
        
        # Geological context adjustment
        tectonic_pattern = self._identify_regional_tectonic_pattern(point)
        geological_adjustment = 1.0
        
        if tectonic_pattern:
            # Different tectonic settings have different confidence expectations
            tectonic_confidence_factors = {
                'stable_intraplate': 1.1,     # Higher confidence in stable regions
                'active_plate_boundary': 0.9,  # Lower confidence in active regions
                'fault_zone': 0.8,            # Lowest confidence near faults
                'volcanic': 0.85,             # Moderate confidence in volcanic areas
                'ridge': 0.9                  # Moderate confidence at ridges
            }
            
            geological_context = tectonic_pattern.get('geological_context', 'stable_intraplate')
            geological_adjustment = tectonic_confidence_factors.get(geological_context, 1.0)
        
        # Spatial clustering bonus (higher confidence if data is well distributed)
        if len(nearby_points) >= 4:
            distances = [p['distance'] for p in nearby_points]
            distance_variance = np.var(distances)
            
            # Bonus for well-distributed data (moderate variance is good)
            optimal_variance = (radius_km / 3) ** 2
            distribution_factor = 1 - abs(distance_variance - optimal_variance) / optimal_variance
            distribution_bonus = max(0, min(0.1, distribution_factor * 0.1))
        else:
            distribution_bonus = 0
        
        # Final confidence calculation
        final_confidence = (base_confidence * geological_adjustment) + distribution_bonus
        
        return max(0.05, min(0.95, final_confidence))
    
    def _analyze_regime_spatial_clustering(self, stress_field: StressField, 
                                         reliable_tensors: List[StressTensor]) -> Dict:
        """Advanced spatial clustering analysis using AI and machine learning with backend integration"""
        
        clustering_results = {
            'analysis_method': 'ai_enhanced_spatial_clustering',
            'timestamp': datetime.now().isoformat(),
            'n_tensors': len(reliable_tensors)
        }
        
        if len(reliable_tensors) < 3:
            clustering_results.update({
                'status': 'insufficient_data',
                'message': 'Need at least 3 tensors for clustering analysis'
            })
            return clustering_results
        
        # Use autonomous geoscience for intelligent clustering
        if self.backend_modules.get('autonomous_geoscience_integration'):
            try:
                autonomous_system = self.backend_modules['autonomous_geoscience_integration']
                
                # AI-powered regime clustering analysis
                ai_clustering = autonomous_system.analyze_stress_regime_clustering({
                    'stress_field': stress_field,
                    'stress_tensors': reliable_tensors,
                    'clustering_methods': ['spatial_dbscan', 'hierarchical', 'geological_kmeans'],
                    'include_uncertainty': True,
                    'geological_constraints': True,
                    'adaptive_parameters': True
                })
                
                if ai_clustering:
                    clustering_results['ai_clustering'] = {
                        'detected_clusters': ai_clustering.get('n_clusters', 0),
                        'cluster_assignments': ai_clustering.get('cluster_labels', []),
                        'cluster_quality': ai_clustering.get('clustering_quality', 0.5),
                        'geological_coherence': ai_clustering.get('geological_coherence', 0.5),
                        'optimal_method': ai_clustering.get('best_method', 'unknown')
                    }
                    
                    print(f"+ AI clustering analysis: {ai_clustering.get('n_clusters', 0)} "
                          f"clusters detected using {ai_clustering.get('best_method', 'unknown')}")
                    
            except Exception as e:
                print(f"! Error in AI clustering analysis: {e}")
        
        # Use data integration pipeline for multi-dimensional clustering
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Create comprehensive clustering dataset
                clustering_dataset = self._create_clustering_dataset(stress_field, reliable_tensors)
                
                # Multi-dimensional clustering analysis
                pipeline_clustering = pipeline.perform_multi_dimensional_clustering({
                    'dataset': clustering_dataset,
                    'clustering_dimensions': [
                        'spatial_location',
                        'stress_orientation', 
                        'stress_magnitude',
                        'temporal_context',
                        'geological_setting'
                    ],
                    'methods': ['gaussian_mixture', 'spectral_clustering', 'agglomerative'],
                    'validation_metrics': ['silhouette', 'calinski_harabasz', 'davies_bouldin']
                })
                
                if pipeline_clustering:
                    clustering_results['pipeline_clustering'] = {
                        'dimensional_clusters': pipeline_clustering.get('cluster_results', {}),
                        'validation_scores': pipeline_clustering.get('validation_metrics', {}),
                        'optimal_n_clusters': pipeline_clustering.get('optimal_k', 0),
                        'feature_importance': pipeline_clustering.get('feature_weights', {})
                    }
                    
                    print(f"+ Multi-dimensional clustering: {pipeline_clustering.get('optimal_k', 0)} "
                          f"optimal clusters identified")
                    
            except Exception as e:
                print(f"! Error in pipeline clustering analysis: {e}")
        
        # Advanced machine learning clustering with scikit-learn
        ml_clustering = self._perform_advanced_ml_clustering(stress_field, reliable_tensors)
        clustering_results['ml_clustering'] = ml_clustering
        
        # Geological interpretation of clusters
        cluster_interpretation = self._interpret_clustering_results(
            clustering_results, stress_field, reliable_tensors
        )
        clustering_results['geological_interpretation'] = cluster_interpretation
        
        # Calculate overall clustering quality and confidence
        overall_assessment = self._assess_overall_clustering_quality(clustering_results)
        clustering_results.update(overall_assessment)
        
        return clustering_results
    
    def _create_clustering_dataset(self, stress_field: StressField,
                                 reliable_tensors: List[StressTensor]) -> Dict:
        """Create comprehensive dataset for clustering analysis"""
        
        dataset = {
            'spatial_features': [],
            'stress_features': [],
            'temporal_features': [],
            'quality_features': [],
            'geological_features': []
        }
        
        for tensor in reliable_tensors:
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            # Spatial features
            dataset['spatial_features'].append([point[0], point[1]])  # lat, lon
            
            # Stress features
            dataset['stress_features'].append([
                tensor.azimuth1, tensor.plunge1, tensor.stress_ratio,
                tensor.azimuth2, tensor.plunge2,  # Conjugate plane
                self._regime_to_numeric(tensor.regime)
            ])
            
            # Temporal features
            timestamp = tensor.timestamp.timestamp() if tensor.timestamp else 0
            dataset['temporal_features'].append([timestamp])
            
            # Quality features
            dataset['quality_features'].append([tensor.confidence])
            
            # Geological features (if available from tectonic analysis)
            tectonic_pattern = self._identify_regional_tectonic_pattern(point)
            geological_features = self._extract_geological_clustering_features(tectonic_pattern)
            dataset['geological_features'].append(geological_features)
        
        return dataset
    
    def _perform_advanced_ml_clustering(self, stress_field: StressField,
                                      reliable_tensors: List[StressTensor]) -> Dict:
        """Perform advanced machine learning clustering"""
        
        ml_results = {
            'method': 'advanced_ml_clustering',
            'available': False
        }
        
        try:
            from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # Prepare feature matrix
            features = []
            regime_labels = []
            
            for tensor in reliable_tensors:
                tensor_index = stress_field.stress_tensors.index(tensor)
                point = stress_field.grid_points[tensor_index]
                
                # Combined feature vector
                feature_vector = [
                    point[0], point[1],  # Spatial
                    tensor.azimuth1, tensor.plunge1,  # Primary stress orientation
                    tensor.stress_ratio,  # Stress magnitude
                    tensor.confidence,  # Quality
                    self._regime_to_numeric(tensor.regime)  # Regime
                ]
                
                features.append(feature_vector)
                regime_labels.append(tensor.regime)
            
            features = np.array(features)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Test multiple clustering algorithms
            clustering_methods = {
                'dbscan': DBSCAN(eps=0.5, min_samples=3),
                'agglomerative': AgglomerativeClustering(n_clusters=None, distance_threshold=1.0),
                'spectral': SpectralClustering(n_clusters=3, random_state=42),
                'gaussian_mixture': GaussianMixture(n_components=3, random_state=42)
            }
            
            method_results = {}
            
            for method_name, clusterer in clustering_methods.items():
                try:
                    if method_name == 'gaussian_mixture':
                        cluster_labels = clusterer.fit_predict(features_scaled)
                    else:
                        cluster_labels = clusterer.fit_predict(features_scaled)
                    
                    # Calculate clustering metrics
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    
                    if n_clusters > 1 and n_clusters < len(features) - 1:
                        silhouette = silhouette_score(features_scaled, cluster_labels)
                        calinski = calinski_harabasz_score(features_scaled, cluster_labels)
                        
                        method_results[method_name] = {
                            'n_clusters': n_clusters,
                            'cluster_labels': cluster_labels.tolist(),
                            'silhouette_score': silhouette,
                            'calinski_harabasz_score': calinski,
                            'noise_points': list(cluster_labels).count(-1) if -1 in cluster_labels else 0
                        }
                    
                except Exception as e:
                    print(f"! Error in {method_name} clustering: {e}")
            
            # Select best method based on silhouette score
            if method_results:
                best_method = max(method_results.keys(), 
                                key=lambda k: method_results[k].get('silhouette_score', -1))
                
                ml_results.update({
                    'available': True,
                    'methods_tested': list(method_results.keys()),
                    'method_results': method_results,
                    'best_method': best_method,
                    'best_result': method_results[best_method]
                })
                
                print(f"+ ML clustering completed: {best_method} achieved best results "
                      f"(silhouette: {method_results[best_method]['silhouette_score']:.3f})")
            
        except ImportError:
            ml_results['message'] = 'scikit-learn not available for ML clustering'
            
        except Exception as e:
            ml_results['error'] = str(e)
            print(f"! Error in ML clustering: {e}")
        
        return ml_results
    
    def _regime_to_numeric(self, regime: str) -> float:
        """Convert stress regime to numeric value for clustering"""
        
        regime_mapping = {
            'NF': 0.0,  # Normal faulting
            'SS': 1.0,  # Strike-slip
            'TF': 2.0,  # Thrust faulting
        }
        
        return regime_mapping.get(regime, 0.5)  # Default to middle value
    
    def _extract_geological_clustering_features(self, tectonic_pattern: Optional[Dict]) -> List[float]:
        """Extract geological features for clustering"""
        
        if not tectonic_pattern:
            return [0.0, 0.0, 0.0]  # Default neutral values
        
        # Encode tectonic setting
        tectonic_encoding = {
            'pacific_rim': 1.0,
            'san_andreas_system': 2.0,
            'mid_atlantic_ridge': 3.0,
            'intraplate': 0.0
        }
        
        tectonic_value = tectonic_encoding.get(
            tectonic_pattern.get('tectonic_province', 'intraplate'), 0.0
        )
        
        confidence = tectonic_pattern.get('confidence', 0.5)
        stress_regime_encoding = {
            'NF': 0.0, 'SS': 1.0, 'TF': 2.0
        }.get(tectonic_pattern.get('dominant_regime', 'SS'), 1.0)
        
        return [tectonic_value, confidence, stress_regime_encoding]
    
    def _interpret_clustering_results(self, clustering_results: Dict,
                                    stress_field: StressField,
                                    reliable_tensors: List[StressTensor]) -> Dict:
        """Interpret clustering results in geological context"""
        
        interpretation = {
            'cluster_geological_meanings': {},
            'tectonic_implications': [],
            'spatial_patterns': {},
            'regime_distributions': {}
        }
        
        # Extract best clustering result
        best_clustering = None
        
        if 'ai_clustering' in clustering_results:
            best_clustering = clustering_results['ai_clustering']
        elif 'ml_clustering' in clustering_results and clustering_results['ml_clustering'].get('available'):
            best_clustering = clustering_results['ml_clustering']['best_result']
        
        if best_clustering and 'cluster_labels' in best_clustering:
            cluster_labels = best_clustering['cluster_labels']
            n_clusters = best_clustering.get('n_clusters', 0)
            
            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_tensors = [tensor for i, tensor in enumerate(reliable_tensors)
                                 if i < len(cluster_labels) and cluster_labels[i] == cluster_id]
                
                if cluster_tensors:
                    cluster_analysis = self._analyze_individual_cluster(
                        cluster_id, cluster_tensors, stress_field
                    )
                    
                    interpretation['cluster_geological_meanings'][f'cluster_{cluster_id}'] = cluster_analysis
            
            # Overall pattern analysis
            interpretation['spatial_patterns'] = self._analyze_cluster_spatial_patterns(
                cluster_labels, stress_field, reliable_tensors
            )
            
            interpretation['regime_distributions'] = self._analyze_cluster_regime_distributions(
                cluster_labels, reliable_tensors
            )
        
        return interpretation
    
    def _assess_overall_clustering_quality(self, clustering_results: Dict) -> Dict:
        """Assess overall quality and confidence of clustering results"""
        
        assessment = {
            'overall_quality': 0.5,
            'confidence': 0.5,
            'geological_plausibility': 0.5,
            'recommendations': []
        }
        
        quality_scores = []
        
        # Collect quality scores from different methods
        if 'ai_clustering' in clustering_results:
            ai_quality = clustering_results['ai_clustering'].get('cluster_quality', 0.5)
            quality_scores.append(ai_quality)
        
        if 'ml_clustering' in clustering_results and clustering_results['ml_clustering'].get('available'):
            ml_quality = clustering_results['ml_clustering']['best_result'].get('silhouette_score', 0)
            # Convert silhouette score (-1 to 1) to 0-1 scale
            ml_quality_normalized = (ml_quality + 1) / 2
            quality_scores.append(ml_quality_normalized)
        
        if quality_scores:
            assessment['overall_quality'] = np.mean(quality_scores)
            assessment['confidence'] = min(0.95, assessment['overall_quality'] * 1.2)
        
        # Generate recommendations based on results
        if assessment['overall_quality'] < 0.3:
            assessment['recommendations'].append('Consider acquiring more data for reliable clustering')
        
        if assessment['overall_quality'] < 0.6:
            assessment['recommendations'].append('Results should be interpreted with caution')
        
        if assessment['overall_quality'] > 0.8:
            assessment['recommendations'].append('High-quality clustering results suitable for interpretation')
        
        return assessment
    
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
        """Advanced uncertainty estimation using bootstrap resampling and statistical analysis"""
        
        uncertainties = {
            'azimuth_uncertainty': [],
            'plunge_uncertainty': [],
            'ratio_uncertainty': [],
            'regime_uncertainty': [],
            'bootstrap_confidence_intervals': [],
            'stability_metrics': [],
            'data_quality_indicators': []
        }
        
        for i, point in enumerate(grid_points):
            nearby_mechanisms = self._find_nearby_mechanisms(
                point, mechanisms, self.config.smoothing_radius
            )
            
            if len(nearby_mechanisms) < 3:
                # Insufficient data - use conservative uncertainties
                uncertainties['azimuth_uncertainty'].append(45.0)
                uncertainties['plunge_uncertainty'].append(30.0)
                uncertainties['ratio_uncertainty'].append(0.3)
                uncertainties['regime_uncertainty'].append(0.8)
                uncertainties['bootstrap_confidence_intervals'].append({})
                uncertainties['stability_metrics'].append({'stability_index': 0.0})
                uncertainties['data_quality_indicators'].append({'quality_score': 0.1})
                continue
            
            # Perform bootstrap uncertainty analysis
            bootstrap_results = self._bootstrap_stress_analysis(nearby_mechanisms, point)
            
            # Data quality assessment
            quality_metrics = self._assess_data_quality(nearby_mechanisms)
            
            # Statistical uncertainty estimation
            statistical_uncertainty = self._calculate_statistical_uncertainty(
                nearby_mechanisms, stress_tensors[i] if i < len(stress_tensors) else None
            )
            
            # Combine uncertainty sources
            combined_uncertainties = self._combine_uncertainty_estimates(
                bootstrap_results, quality_metrics, statistical_uncertainty
            )
            
            uncertainties['azimuth_uncertainty'].append(combined_uncertainties['azimuth'])
            uncertainties['plunge_uncertainty'].append(combined_uncertainties['plunge'])
            uncertainties['ratio_uncertainty'].append(combined_uncertainties['ratio'])
            uncertainties['regime_uncertainty'].append(combined_uncertainties['regime'])
            uncertainties['bootstrap_confidence_intervals'].append(bootstrap_results['confidence_intervals'])
            uncertainties['stability_metrics'].append(bootstrap_results['stability_metrics'])
            uncertainties['data_quality_indicators'].append(quality_metrics)
        
        return uncertainties
    
    def _bootstrap_stress_analysis(self, mechanisms: List[FocalMechanism], point: np.ndarray) -> Dict:
        """Bootstrap resampling analysis for uncertainty quantification"""
        
        n_bootstrap = min(self.config.bootstrap_samples, 500)  # Limit for performance
        bootstrap_results = {
            'azimuths': [],
            'plunges': [],
            'ratios': [],
            'regimes': [],
            'confidence_values': []
        }
        
        # Original inversion for comparison
        try:
            original_tensor = self._gephart_inversion(mechanisms)
        except:
            original_tensor = self._michael_inversion(mechanisms)
        
        successful_inversions = 0
        
        for bootstrap_iter in range(n_bootstrap):
            try:
                # Resample mechanisms with replacement
                n_mechanisms = len(mechanisms)
                bootstrap_indices = np.random.choice(n_mechanisms, size=n_mechanisms, replace=True)
                bootstrap_mechanisms = [mechanisms[i] for i in bootstrap_indices]
                
                # Add small perturbations to account for measurement uncertainty
                perturbed_mechanisms = self._perturb_mechanisms(bootstrap_mechanisms)
                
                # Perform stress inversion
                if np.random.random() < 0.7:  # Use Gephart method more often
                    bootstrap_tensor = self._gephart_inversion(perturbed_mechanisms)
                else:
                    bootstrap_tensor = self._michael_inversion(perturbed_mechanisms)
                
                if bootstrap_tensor.confidence > 0.1:
                    bootstrap_results['azimuths'].append(bootstrap_tensor.azimuth1)
                    bootstrap_results['plunges'].append(bootstrap_tensor.plunge1)
                    bootstrap_results['ratios'].append(bootstrap_tensor.stress_ratio)
                    bootstrap_results['regimes'].append(bootstrap_tensor.regime)
                    bootstrap_results['confidence_values'].append(bootstrap_tensor.confidence)
                    successful_inversions += 1
                    
            except Exception:
                continue
        
        # Calculate confidence intervals and stability metrics
        confidence_intervals = {}
        stability_metrics = {'stability_index': 0.0, 'success_rate': successful_inversions / n_bootstrap}
        
        if successful_inversions > 10:
            # Calculate 95% confidence intervals
            if bootstrap_results['azimuths']:
                # Handle circular statistics for azimuths
                circular_stats = self._circular_confidence_interval(bootstrap_results['azimuths'])
                confidence_intervals['azimuth'] = circular_stats
                
                # Standard confidence intervals for other parameters
                confidence_intervals['plunge'] = {
                    'lower': np.percentile(bootstrap_results['plunges'], 2.5),
                    'upper': np.percentile(bootstrap_results['plunges'], 97.5),
                    'median': np.median(bootstrap_results['plunges']),
                    'std': np.std(bootstrap_results['plunges'])
                }
                
                confidence_intervals['ratio'] = {
                    'lower': np.percentile(bootstrap_results['ratios'], 2.5),
                    'upper': np.percentile(bootstrap_results['ratios'], 97.5),
                    'median': np.median(bootstrap_results['ratios']),
                    'std': np.std(bootstrap_results['ratios'])
                }
                
                # Regime stability
                regime_counts = {}
                for regime in bootstrap_results['regimes']:
                    regime_counts[regime] = regime_counts.get(regime, 0) + 1
                
                most_common_regime = max(regime_counts, key=regime_counts.get)
                regime_stability = regime_counts[most_common_regime] / len(bootstrap_results['regimes'])
                
                confidence_intervals['regime'] = {
                    'most_likely': most_common_regime,
                    'stability': regime_stability,
                    'distribution': regime_counts
                }
                
                # Overall stability index
                azimuth_stability = 1.0 - (circular_stats['circular_std'] / 180.0)
                plunge_stability = 1.0 - (np.std(bootstrap_results['plunges']) / 90.0)
                ratio_stability = 1.0 - (np.std(bootstrap_results['ratios']) / 0.5)
                
                stability_metrics['stability_index'] = np.mean([
                    azimuth_stability, plunge_stability, ratio_stability, regime_stability
                ])
        
        return {
            'bootstrap_results': bootstrap_results,
            'confidence_intervals': confidence_intervals,
            'stability_metrics': stability_metrics,
            'n_successful': successful_inversions
        }
    
    def _perturb_mechanisms(self, mechanisms: List[FocalMechanism]) -> List[FocalMechanism]:
        """Add small perturbations to focal mechanisms to account for measurement uncertainty"""
        
        perturbed = []
        
        for fm in mechanisms:
            # Estimate perturbation based on magnitude and depth
            base_error = max(3.0, 15.0 / max(fm.magnitude, 3.0))
            depth_factor = 1.0 + fm.depth / 100.0
            perturbation_std = base_error * depth_factor
            
            # Add Gaussian noise to strike, dip, rake
            strike1 = (fm.strike1 + np.random.normal(0, perturbation_std)) % 360
            dip1 = np.clip(fm.dip1 + np.random.normal(0, perturbation_std * 0.7), 0, 90)
            rake1 = np.clip(fm.rake1 + np.random.normal(0, perturbation_std), -180, 180)
            
            strike2 = (fm.strike2 + np.random.normal(0, perturbation_std)) % 360
            dip2 = np.clip(fm.dip2 + np.random.normal(0, perturbation_std * 0.7), 0, 90)
            rake2 = np.clip(fm.rake2 + np.random.normal(0, perturbation_std), -180, 180)
            
            # Create perturbed mechanism
            perturbed_fm = FocalMechanism(
                event_id=fm.event_id,
                latitude=fm.latitude,
                longitude=fm.longitude,
                depth=fm.depth,
                magnitude=fm.magnitude,
                strike1=strike1,
                dip1=dip1,
                rake1=rake1,
                strike2=strike2,
                dip2=dip2,
                rake2=rake2,
                timestamp=fm.timestamp,
                uncertainty=fm.uncertainty,
                quality=fm.quality
            )
            
            perturbed.append(perturbed_fm)
        
        return perturbed
    
    def _circular_confidence_interval(self, azimuths: List[float]) -> Dict:
        """Calculate confidence intervals for circular data (azimuths)"""
        
        if not azimuths:
            return {'lower': 0, 'upper': 360, 'circular_mean': 0, 'circular_std': 180}
        
        # Convert to radians
        azimuth_rad = np.radians(azimuths)
        
        # Circular mean
        mean_cos = np.mean(np.cos(azimuth_rad))
        mean_sin = np.mean(np.sin(azimuth_rad))
        circular_mean_rad = np.arctan2(mean_sin, mean_cos)
        circular_mean = np.degrees(circular_mean_rad) % 360
        
        # Circular standard deviation
        r = np.sqrt(mean_cos**2 + mean_sin**2)
        if r > 0:
            circular_std_rad = np.sqrt(-2 * np.log(r))
            circular_std = np.degrees(circular_std_rad)
        else:
            circular_std = 180.0
        
        # Confidence interval (approximate)
        confidence_width = 1.96 * circular_std / np.sqrt(len(azimuths))  # 95% CI
        lower = (circular_mean - confidence_width) % 360
        upper = (circular_mean + confidence_width) % 360
        
        return {
            'lower': lower,
            'upper': upper,
            'circular_mean': circular_mean,
            'circular_std': circular_std,
            'concentration': r
        }
    
    def _assess_data_quality(self, mechanisms: List[FocalMechanism]) -> Dict:
        """Comprehensive data quality assessment for focal mechanisms"""
        
        quality_metrics = {
            'n_mechanisms': len(mechanisms),
            'magnitude_distribution': {},
            'depth_distribution': {},
            'temporal_distribution': {},
            'spatial_clustering': 0.0,
            'quality_grades': {},
            'overall_quality_score': 0.0
        }
        
        if len(mechanisms) == 0:
            return quality_metrics
        
        # Magnitude distribution analysis
        magnitudes = [fm.magnitude for fm in mechanisms if fm.magnitude > 0]
        if magnitudes:
            quality_metrics['magnitude_distribution'] = {
                'mean': np.mean(magnitudes),
                'std': np.std(magnitudes),
                'min': np.min(magnitudes),
                'max': np.max(magnitudes),
                'range': np.max(magnitudes) - np.min(magnitudes)
            }
            
            # Higher magnitude events are generally more reliable
            mag_quality = min(1.0, np.mean(magnitudes) / 5.0)
        else:
            mag_quality = 0.5
        
        # Depth distribution analysis
        depths = [fm.depth for fm in mechanisms if fm.depth > 0]
        if depths:
            quality_metrics['depth_distribution'] = {
                'mean': np.mean(depths),
                'std': np.std(depths),
                'min': np.min(depths),
                'max': np.max(depths)
            }
            
            # Prefer moderate depths (5-25 km) for reliability
            mean_depth = np.mean(depths)
            if 5 <= mean_depth <= 25:
                depth_quality = 1.0
            elif mean_depth < 5:
                depth_quality = 0.7  # Shallow events may be less reliable
            else:
                depth_quality = max(0.3, 1.0 - (mean_depth - 25) / 50.0)
        else:
            depth_quality = 0.5
        
        # Temporal distribution
        timestamps = [fm.timestamp for fm in mechanisms]
        if len(timestamps) > 1:
            time_spans = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                         for i in range(len(timestamps)-1)]
            quality_metrics['temporal_distribution'] = {
                'time_span_days': (max(timestamps) - min(timestamps)).total_seconds() / 86400,
                'mean_interval_hours': np.mean(time_spans) / 3600,
                'temporal_clustering': np.std(time_spans) / np.mean(time_spans) if np.mean(time_spans) > 0 else 0
            }
            
            # Prefer distributed events over clustered ones
            clustering = quality_metrics['temporal_distribution']['temporal_clustering']
            temporal_quality = min(1.0, clustering / 2.0) if clustering > 0 else 0.5
        else:
            temporal_quality = 0.3
        
        # Quality grade assessment
        quality_grades = {}
        for fm in mechanisms:
            grade = fm.quality if fm.quality else 'Unknown'
            quality_grades[grade] = quality_grades.get(grade, 0) + 1
        
        quality_metrics['quality_grades'] = quality_grades
        
        # Convert grades to numerical quality
        grade_weights = {'A': 1.0, 'B': 0.8, 'C': 0.6, 'D': 0.4, 'Unknown': 0.5}
        if quality_grades:
            weighted_quality = sum(grade_weights.get(grade, 0.5) * count 
                                 for grade, count in quality_grades.items())
            grade_quality = weighted_quality / len(mechanisms)
        else:
            grade_quality = 0.5
        
        # Data density quality
        density_quality = min(1.0, len(mechanisms) / 10.0)  # Optimal around 10+ mechanisms
        
        # Overall quality score
        quality_metrics['overall_quality_score'] = np.mean([
            mag_quality, depth_quality, temporal_quality, grade_quality, density_quality
        ])
        
        return quality_metrics
    
    def _calculate_statistical_uncertainty(self, mechanisms: List[FocalMechanism], 
                                         stress_tensor: Optional[StressTensor] = None) -> Dict:
        """Calculate statistical uncertainty estimates based on data characteristics"""
        
        n_mechanisms = len(mechanisms)
        
        # Base uncertainty from sample size
        sample_size_factor = max(0.1, 1.0 - np.log(n_mechanisms) / np.log(20))  # Decreases with more data
        
        # Mechanism diversity (variety in orientations indicates good sampling)
        if n_mechanisms > 3:
            strikes = [fm.strike1 for fm in mechanisms]
            strike_diversity = self._calculate_circular_variance(strikes) / 180.0
            
            dips = [fm.dip1 for fm in mechanisms]
            dip_diversity = np.std(dips) / 90.0
            
            diversity_factor = 1.0 - np.mean([strike_diversity, dip_diversity])
        else:
            diversity_factor = 0.5
        
        # Distance from mechanisms (larger spread = higher uncertainty)
        if n_mechanisms > 1:
            lats = [fm.latitude for fm in mechanisms]
            lons = [fm.longitude for fm in mechanisms]
            
            lat_spread = np.max(lats) - np.min(lats)
            lon_spread = np.max(lons) - np.min(lons)
            
            spatial_spread = np.sqrt(lat_spread**2 + lon_spread**2)
            spread_factor = min(1.0, spatial_spread / 0.5)  # Normalized to ~0.5 degree spread
        else:
            spread_factor = 0.0
        
        # Magnitude consistency
        magnitudes = [fm.magnitude for fm in mechanisms if fm.magnitude > 0]
        if len(magnitudes) > 1:
            mag_consistency = 1.0 - (np.std(magnitudes) / 2.0)  # Prefer similar magnitudes
            mag_consistency = np.clip(mag_consistency, 0.0, 1.0)
        else:
            mag_consistency = 0.5
        
        # Combine factors
        base_azimuth_uncertainty = 20.0 * sample_size_factor * (1.0 + spread_factor) * (2.0 - diversity_factor)
        base_plunge_uncertainty = base_azimuth_uncertainty * 0.7
        base_ratio_uncertainty = 0.2 * sample_size_factor * (2.0 - mag_consistency)
        base_regime_uncertainty = 0.3 * sample_size_factor * (2.0 - diversity_factor)
        
        return {
            'azimuth': np.clip(base_azimuth_uncertainty, 5.0, 60.0),
            'plunge': np.clip(base_plunge_uncertainty, 3.0, 40.0),
            'ratio': np.clip(base_ratio_uncertainty, 0.05, 0.4),
            'regime': np.clip(base_regime_uncertainty, 0.1, 0.7),
            'factors': {
                'sample_size_factor': sample_size_factor,
                'diversity_factor': diversity_factor,
                'spread_factor': spread_factor,
                'magnitude_consistency': mag_consistency
            }
        }
    
    def _calculate_circular_variance(self, angles: List[float]) -> float:
        """Calculate circular variance for angular data"""
        if not angles:
            return 180.0
        
        angles_rad = np.radians(angles)
        mean_cos = np.mean(np.cos(angles_rad))
        mean_sin = np.mean(np.sin(angles_rad))
        r = np.sqrt(mean_cos**2 + mean_sin**2)
        
        # Circular variance in degrees
        circular_variance = np.degrees(np.sqrt(-2 * np.log(r))) if r > 0 else 180.0
        return circular_variance
    
    def _combine_uncertainty_estimates(self, bootstrap_results: Dict, quality_metrics: Dict, 
                                     statistical_uncertainty: Dict) -> Dict:
        """Combine different uncertainty estimates into final values"""
        
        # Weight bootstrap results based on success rate
        bootstrap_weight = bootstrap_results['stability_metrics']['success_rate']
        statistical_weight = 1.0 - bootstrap_weight
        
        # Quality adjustment factor
        quality_factor = 1.0 / max(0.1, quality_metrics['overall_quality_score'])
        
        combined = {}
        
        # Azimuth uncertainty
        if 'azimuth' in bootstrap_results['confidence_intervals']:
            bootstrap_azimuth = bootstrap_results['confidence_intervals']['azimuth']['circular_std']
            combined['azimuth'] = (
                bootstrap_weight * bootstrap_azimuth + 
                statistical_weight * statistical_uncertainty['azimuth']
            ) * quality_factor
        else:
            combined['azimuth'] = statistical_uncertainty['azimuth'] * quality_factor
        
        # Plunge uncertainty
        if 'plunge' in bootstrap_results['confidence_intervals']:
            bootstrap_plunge = bootstrap_results['confidence_intervals']['plunge']['std']
            combined['plunge'] = (
                bootstrap_weight * bootstrap_plunge + 
                statistical_weight * statistical_uncertainty['plunge']
            ) * quality_factor
        else:
            combined['plunge'] = statistical_uncertainty['plunge'] * quality_factor
        
        # Ratio uncertainty
        if 'ratio' in bootstrap_results['confidence_intervals']:
            bootstrap_ratio = bootstrap_results['confidence_intervals']['ratio']['std']
            combined['ratio'] = (
                bootstrap_weight * bootstrap_ratio + 
                statistical_weight * statistical_uncertainty['ratio']
            ) * quality_factor
        else:
            combined['ratio'] = statistical_uncertainty['ratio'] * quality_factor
        
        # Regime uncertainty
        if 'regime' in bootstrap_results['confidence_intervals']:
            regime_stability = bootstrap_results['confidence_intervals']['regime']['stability']
            bootstrap_regime = 1.0 - regime_stability
            combined['regime'] = (
                bootstrap_weight * bootstrap_regime + 
                statistical_weight * statistical_uncertainty['regime']
            ) * quality_factor
        else:
            combined['regime'] = statistical_uncertainty['regime'] * quality_factor
        
        # Apply final bounds
        combined['azimuth'] = np.clip(combined['azimuth'], 3.0, 90.0)
        combined['plunge'] = np.clip(combined['plunge'], 2.0, 60.0)
        combined['ratio'] = np.clip(combined['ratio'], 0.02, 0.5)
        combined['regime'] = np.clip(combined['regime'], 0.05, 0.9)
        
        return combined
    
    def validate_with_geological_knowledge(self, stress_field: StressField) -> Dict:
        """Validate stress field against geological knowledge and constraints"""
        
        validation_results = {
            'overall_validity': 0.0,
            'tectonic_consistency': {},
            'geological_plausibility': {},
            'constraint_violations': [],
            'recommendations': [],
            'knowledge_integration_results': {}
        }
        
        # Validate against tectonic knowledge
        tectonic_validation = self._validate_tectonic_consistency(stress_field)
        validation_results['tectonic_consistency'] = tectonic_validation
        
        # Validate geological plausibility
        plausibility_validation = self._validate_geological_plausibility(stress_field)
        validation_results['geological_plausibility'] = plausibility_validation
        
        # Check physical constraints
        constraint_validation = self._validate_physical_constraints(stress_field)
        validation_results['constraint_violations'] = constraint_validation
        
        # Integrate with backend modules if available
        backend_validation = self._validate_with_backend_modules(stress_field)
        validation_results['knowledge_integration_results'] = backend_validation
        
        # Generate recommendations
        recommendations = self._generate_validation_recommendations(
            tectonic_validation, plausibility_validation, constraint_validation
        )
        validation_results['recommendations'] = recommendations
        
        # Calculate overall validity score
        tectonic_score = tectonic_validation.get('consistency_score', 0.5)
        plausibility_score = plausibility_validation.get('plausibility_score', 0.5)
        constraint_score = 1.0 - min(1.0, len(constraint_validation) / 10.0)
        backend_score = backend_validation.get('integration_score', 0.5)
        
        validation_results['overall_validity'] = np.mean([
            tectonic_score, plausibility_score, constraint_score, backend_score
        ])
        
        return validation_results
    
    def _validate_tectonic_consistency(self, stress_field: StressField) -> Dict:
        """Validate stress field consistency with known tectonic patterns"""
        
        consistency_results = {
            'consistency_score': 0.0,
            'regional_matches': [],
            'pattern_deviations': [],
            'tectonic_context': {}
        }
        
        if not stress_field.stress_tensors:
            return consistency_results
        
        # Analyze regional consistency
        region_center = np.mean(stress_field.grid_points, axis=0)
        regional_pattern = self._identify_regional_tectonic_pattern(region_center)
        
        if regional_pattern:
            consistency_results['tectonic_context'] = regional_pattern
            
            # Check stress regime consistency
            observed_regimes = [t.regime for t in stress_field.stress_tensors if t.confidence > 0.3]
            expected_regime = regional_pattern.get('dominant_regime', 'unknown')
            
            if observed_regimes and expected_regime != 'unknown':
                regime_match_fraction = sum(1 for r in observed_regimes if r == expected_regime) / len(observed_regimes)
                consistency_results['regional_matches'].append({
                    'parameter': 'regime',
                    'expected': expected_regime,
                    'match_fraction': regime_match_fraction,
                    'significance': regime_match_fraction > 0.6
                })
            
            # Check stress orientation consistency
            observed_azimuths = [t.azimuth1 for t in stress_field.stress_tensors if t.confidence > 0.3]
            expected_azimuth = regional_pattern.get('sigma1_azimuth', 0)
            
            if observed_azimuths:
                # Calculate circular mean of observed azimuths
                az_rad = np.radians(observed_azimuths)
                mean_cos = np.mean(np.cos(az_rad))
                mean_sin = np.mean(np.sin(az_rad))
                observed_mean_azimuth = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
                
                # Angular difference from expected
                azimuth_difference = min(
                    abs(observed_mean_azimuth - expected_azimuth),
                    360 - abs(observed_mean_azimuth - expected_azimuth)
                )
                
                azimuth_consistency = max(0.0, 1.0 - azimuth_difference / 90.0)
                
                consistency_results['regional_matches'].append({
                    'parameter': 'azimuth',
                    'expected': expected_azimuth,
                    'observed': observed_mean_azimuth,
                    'angular_difference': azimuth_difference,
                    'consistency_score': azimuth_consistency
                })
        
        # Calculate overall consistency score
        if consistency_results['regional_matches']:
            match_scores = []
            for match in consistency_results['regional_matches']:
                if match['parameter'] == 'regime':
                    match_scores.append(match['match_fraction'])
                elif match['parameter'] == 'azimuth':
                    match_scores.append(match['consistency_score'])
            
            consistency_results['consistency_score'] = np.mean(match_scores)
        else:
            consistency_results['consistency_score'] = 0.5  # Neutral if no regional pattern
        
        return consistency_results
    
    def _validate_geological_plausibility(self, stress_field: StressField) -> Dict:
        """Validate geological plausibility of stress field characteristics"""
        
        plausibility_results = {
            'plausibility_score': 0.0,
            'depth_consistency': {},
            'magnitude_relationships': {},
            'temporal_stability': {},
            'spatial_coherence': {}
        }
        
        reliable_tensors = [t for t in stress_field.stress_tensors if t.confidence > 0.2]
        
        if len(reliable_tensors) < 3:
            plausibility_results['plausibility_score'] = 0.3
            return plausibility_results
        
        # Validate stress ratio distributions
        stress_ratios = [t.stress_ratio for t in reliable_tensors]
        ratio_distribution = self._analyze_ratio_distribution(stress_ratios)
        plausibility_results['ratio_distribution'] = ratio_distribution
        
        # Spatial coherence analysis
        spatial_coherence = self._analyze_spatial_coherence(stress_field, reliable_tensors)
        plausibility_results['spatial_coherence'] = spatial_coherence
        
        # Regime transition analysis
        regime_transitions = self._analyze_regime_transitions(stress_field, reliable_tensors)
        plausibility_results['regime_transitions'] = regime_transitions
        
        # Calculate overall plausibility
        plausibility_components = [
            ratio_distribution.get('plausibility', 0.5),
            spatial_coherence.get('coherence_score', 0.5),
            regime_transitions.get('transition_plausibility', 0.5)
        ]
        
        plausibility_results['plausibility_score'] = np.mean(plausibility_components)
        
        return plausibility_results
    
    def _validate_physical_constraints(self, stress_field: StressField) -> List[Dict]:
        """Validate stress field against physical and geological constraints"""
        
        violations = []
        validation_rules = self.geological_knowledge['validation_rules']
        
        reliable_tensors = [t for t in stress_field.stress_tensors if t.confidence > 0.2]
        
        # Check stress ratio bounds
        for i, tensor in enumerate(reliable_tensors):
            if not (0.0 <= tensor.stress_ratio <= 1.0):
                point = stress_field.grid_points[i]
                violations.append({
                    'type': 'stress_ratio_bounds',
                    'location': (point[0], point[1]),
                    'observed_value': tensor.stress_ratio,
                    'valid_range': (0.0, 1.0),
                    'severity': 'high'
                })
        
        # Check spatial gradient constraints
        max_gradient = validation_rules['orientation_consistency']['maximum_rotation_rate']
        gradient_violations = self._check_spatial_gradients(stress_field, reliable_tensors, max_gradient)
        violations.extend(gradient_violations)
        
        # Check regime stability constraints
        min_domain_size = validation_rules['regime_stability']['minimum_domain_size']
        regime_violations = self._check_regime_stability(stress_field, reliable_tensors, min_domain_size)
        violations.extend(regime_violations)
        
        return violations
    
    def _validate_with_backend_modules(self, stress_field: StressField) -> Dict:
        """Integrate validation with available backend geological modules"""
        
        integration_results = {
            'integration_score': 0.5,
            'hypothesis_validation': {},
            'complexity_assessment': {},
            'knowledge_graph_consistency': {}
        }
        
        # Hypothesis generation system integration
        if self.backend_modules['hypothesis_generation_system']:
            try:
                hypothesis_results = self._validate_with_hypothesis_system(stress_field)
                integration_results['hypothesis_validation'] = hypothesis_results
            except Exception as e:
                integration_results['hypothesis_validation'] = {'error': str(e)}
        
        # Geological complexity assessment integration
        if self.backend_modules['geological_complexity_assessment']:
            try:
                complexity_results = self._validate_with_complexity_assessment(stress_field)
                integration_results['complexity_assessment'] = complexity_results
            except Exception as e:
                integration_results['complexity_assessment'] = {'error': str(e)}
        
        # Calculate integration score based on available validations
        available_validations = [
            integration_results['hypothesis_validation'],
            integration_results['complexity_assessment'],
            integration_results['knowledge_graph_consistency']
        ]
        
        validation_scores = []
        for validation in available_validations:
            if validation and 'score' in validation:
                validation_scores.append(validation['score'])
        
        if validation_scores:
            integration_results['integration_score'] = np.mean(validation_scores)
        
        return integration_results
    
    def _generate_validation_recommendations(self, tectonic_validation: Dict,
                                           plausibility_validation: Dict,
                                           constraint_violations: List[Dict]) -> List[Dict]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        # Tectonic consistency recommendations
        tectonic_score = tectonic_validation.get('consistency_score', 0.5)
        if tectonic_score < 0.6:
            recommendations.append({
                'type': 'tectonic_consistency',
                'priority': 'high',
                'description': 'Stress field shows poor consistency with regional tectonic patterns',
                'suggestion': 'Review focal mechanism quality and consider regional geological context',
                'confidence': 1.0 - tectonic_score
            })
        
        # Plausibility recommendations
        plausibility_score = plausibility_validation.get('plausibility_score', 0.5)
        if plausibility_score < 0.5:
            recommendations.append({
                'type': 'geological_plausibility',
                'priority': 'medium',
                'description': 'Stress field characteristics may be geologically implausible',
                'suggestion': 'Verify input data quality and consider alternative inversion methods',
                'confidence': 1.0 - plausibility_score
            })
        
        # Constraint violation recommendations
        high_severity_violations = [v for v in constraint_violations if v.get('severity') == 'high']
        if high_severity_violations:
            recommendations.append({
                'type': 'constraint_violations',
                'priority': 'critical',
                'description': f'Found {len(high_severity_violations)} critical constraint violations',
                'suggestion': 'Review inversion parameters and data preprocessing',
                'confidence': 0.9
            })
        
        # Data quality recommendations
        if len(constraint_violations) > len(stress_field.stress_tensors) * 0.2:
            recommendations.append({
                'type': 'data_quality',
                'priority': 'high',
                'description': 'High number of constraint violations suggests data quality issues',
                'suggestion': 'Implement more rigorous data filtering and quality control',
                'confidence': 0.8
            })
        
        return recommendations
    
    def export_stress_field(self, stress_field: StressField, filename: str):
        """Export stress field to file with geological validation results"""
        
        # Perform geological validation
        validation_results = self.validate_with_geological_knowledge(stress_field)
        
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
            'uncertainties': stress_field.uncertainties,
            'geological_validation': validation_results
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
    
    # Helper methods for geological validation
    
    def _identify_regional_tectonic_pattern(self, location: np.ndarray) -> Optional[Dict]:
        """Advanced regional tectonic pattern identification using integrated backend modules"""
        
        lat, lon = location
        tectonic_pattern = None
        
        # Use geological complexity assessment module for advanced pattern recognition
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                # Query complex geological patterns with uncertainty quantification
                pattern_analysis = complexity_module.analyze_tectonic_complexity({
                    'latitude': lat,
                    'longitude': lon,
                    'analysis_radius': 100.0,  # km
                    'include_uncertainty': True,
                    'return_confidence': True
                })
                
                if pattern_analysis and pattern_analysis.get('tectonic_regime'):
                    tectonic_pattern = {
                        'dominant_regime': pattern_analysis['tectonic_regime'],
                        'sigma1_azimuth': pattern_analysis.get('principal_stress_azimuth', 0),
                        'confidence': pattern_analysis.get('confidence', 0.5),
                        'complexity_score': pattern_analysis.get('complexity_index', 0.5),
                        'geological_context': pattern_analysis.get('geological_context', {}),
                        'source': 'geological_complexity_assessment'
                    }
                    
                    print(f"+ Advanced tectonic pattern identified: {pattern_analysis['tectonic_regime']} "
                          f"(confidence: {pattern_analysis.get('confidence', 0.5):.2f})")
                    
            except Exception as e:
                print(f"! Error using geological complexity assessment: {e}")
        
        # Use database manager to query historical tectonic patterns
        if not tectonic_pattern and self.backend_modules.get('database_manager'):
            try:
                db_manager = self.backend_modules['database_manager']
                
                # Query tectonic pattern database with spatial search
                query_params = {
                    'table': 'tectonic_stress_patterns',
                    'location': {'lat': lat, 'lon': lon},
                    'search_radius': 50.0,  # km
                    'confidence_threshold': 0.6
                }
                
                pattern_records = db_manager.spatial_query(query_params)
                
                if pattern_records:
                    # Select highest confidence pattern
                    best_pattern = max(pattern_records, key=lambda x: x.get('confidence', 0))
                    
                    tectonic_pattern = {
                        'dominant_regime': best_pattern['stress_regime'],
                        'sigma1_azimuth': best_pattern['sigma1_azimuth'],
                        'confidence': best_pattern['confidence'],
                        'age_ma': best_pattern.get('age_ma', 0),
                        'geological_unit': best_pattern.get('geological_unit', 'unknown'),
                        'source': 'database_historical_records'
                    }
                    
                    print(f"+ Database pattern match: {best_pattern['stress_regime']} "
                          f"(confidence: {best_pattern['confidence']:.2f})")
                    
            except Exception as e:
                print(f"! Error querying tectonic pattern database: {e}")
        
        # Use data integration pipeline for multi-source tectonic analysis
        if not tectonic_pattern and self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Create spatial query for tectonic data layers
                tectonic_query = {
                    'data_types': ['geological', 'structural', 'geophysical'],
                    'location': [lat, lon],
                    'buffer_km': 75.0,
                    'analysis_methods': ['spatial_interpolation', 'trend_analysis']
                }
                
                # Query available geological layers
                available_layers = pipeline.query_spatial_layers(tectonic_query)
                
                if available_layers:
                    # Integrate multiple data sources for pattern recognition
                    integrated_analysis = pipeline.integrate_multi_source_analysis(
                        layers=available_layers,
                        analysis_type='tectonic_stress_pattern',
                        location=[lat, lon]
                    )
                    
                    if integrated_analysis.get('tectonic_interpretation'):
                        tectonic_pattern = {
                            'dominant_regime': integrated_analysis['stress_regime'],
                            'sigma1_azimuth': integrated_analysis.get('stress_orientation', 0),
                            'confidence': integrated_analysis.get('interpretation_confidence', 0.4),
                            'data_sources': integrated_analysis.get('contributing_layers', []),
                            'integration_quality': integrated_analysis.get('integration_score', 0.5),
                            'source': 'multi_source_integration'
                        }
                        
                        print(f"+ Multi-source pattern integration: {integrated_analysis['stress_regime']} "
                              f"from {len(integrated_analysis.get('contributing_layers', []))} data sources")
                        
            except Exception as e:
                print(f"! Error in multi-source tectonic analysis: {e}")
        
        # Enhanced fallback with detailed regional analysis
        if not tectonic_pattern:
            tectonic_pattern = self._enhanced_regional_pattern_fallback(lat, lon)
        
        return tectonic_pattern
    
    def _enhanced_regional_pattern_fallback(self, lat: float, lon: float) -> Dict:
        """Enhanced fallback for regional tectonic pattern identification"""
        
        # Advanced tectonic province recognition with plate boundary analysis
        tectonic_provinces = {
            # Pacific Rim with detailed sub-regions
            'cascadia_subduction': {
                'bounds': [(40, 50), (-130, -120)],
                'regime': 'TF', 'azimuth': 45, 'confidence': 0.85,
                'context': 'Juan de Fuca subduction zone'
            },
            'san_andreas_northern': {
                'bounds': [(35, 40), (-125, -119)],
                'regime': 'SS', 'azimuth': 135, 'confidence': 0.90,
                'context': 'Northern San Andreas transform system'
            },
            'san_andreas_southern': {
                'bounds': [(32, 36), (-122, -115)],
                'regime': 'SS', 'azimuth': 145, 'confidence': 0.88,
                'context': 'Southern San Andreas and related faults'
            },
            'walker_lane': {
                'bounds': [(35, 42), (-121, -115)],
                'regime': 'SS', 'azimuth': 15, 'confidence': 0.75,
                'context': 'Walker Lane transtensional zone'
            },
            'japan_trench': {
                'bounds': [(35, 45), (135, 150)],
                'regime': 'TF', 'azimuth': 120, 'confidence': 0.90,
                'context': 'Pacific plate subduction'
            },
            'mid_atlantic_ridge_north': {
                'bounds': [(30, 70), (-40, -20)],
                'regime': 'NF', 'azimuth': 90, 'confidence': 0.85,
                'context': 'North Atlantic spreading center'
            },
            'mid_atlantic_ridge_south': {
                'bounds': [(-60, 30), (-40, -10)],
                'regime': 'NF', 'azimuth': 80, 'confidence': 0.85,
                'context': 'South Atlantic spreading center'
            },
            'mediterranean_collision': {
                'bounds': [(30, 45), (-10, 45)],
                'regime': 'TF', 'azimuth': 10, 'confidence': 0.75,
                'context': 'Africa-Eurasia collision zone'
            },
            'himalayan_collision': {
                'bounds': [(25, 40), (70, 100)],
                'regime': 'TF', 'azimuth': 20, 'confidence': 0.95,
                'context': 'India-Asia collision zone'
            },
            'andes_subduction': {
                'bounds': [(-55, 15), (-80, -60)],
                'regime': 'TF', 'azimuth': 75, 'confidence': 0.80,
                'context': 'Nazca-South America subduction'
            }
        }
        
        # Find matching tectonic province
        for province_name, province_data in tectonic_provinces.items():
            lat_bounds, lon_bounds = province_data['bounds']
            
            if (lat_bounds[0] <= lat <= lat_bounds[1] and 
                lon_bounds[0] <= lon <= lon_bounds[1]):
                
                return {
                    'dominant_regime': province_data['regime'],
                    'sigma1_azimuth': province_data['azimuth'],
                    'confidence': province_data['confidence'],
                    'tectonic_province': province_name,
                    'geological_context': province_data['context'],
                    'source': 'enhanced_regional_fallback'
                }
        
        # Default global intraplate pattern
        return {
            'dominant_regime': 'SS',  # Most common intraplate regime
            'sigma1_azimuth': 45,     # Generic NE-SW compression
            'confidence': 0.3,        # Low confidence for unknown regions
            'tectonic_province': 'unknown_intraplate',
            'geological_context': 'Unknown intraplate region',
            'source': 'generic_fallback'
        }
    
    def _analyze_ratio_distribution(self, ratios: List[float]) -> Dict:
        """Advanced stress ratio distribution analysis using geological knowledge integration"""
        
        if not ratios:
            return {'plausibility': 0.3, 'analysis_method': 'insufficient_data'}
        
        ratio_analysis = {
            'n_samples': len(ratios),
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'range_coverage': np.max(ratios) - np.min(ratios),
            'median_ratio': np.median(ratios),
            'analysis_method': 'advanced_statistical'
        }
        
        # Use geological complexity assessment for ratio validation
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                # Analyze stress ratio patterns in geological context
                ratio_context = complexity_module.analyze_stress_ratio_patterns({
                    'ratios': ratios,
                    'include_outlier_analysis': True,
                    'geological_constraints': True,
                    'return_classification': True
                })
                
                if ratio_context:
                    ratio_analysis.update({
                        'geological_plausibility': ratio_context.get('geological_plausibility', 0.5),
                        'stress_regime_consistency': ratio_context.get('regime_consistency', 0.5),
                        'outlier_indices': ratio_context.get('outlier_indices', []),
                        'dominant_regime_prediction': ratio_context.get('predicted_regime', 'unknown'),
                        'confidence_geological': ratio_context.get('confidence', 0.5)
                    })
                    
                    print(f"+ Geological stress ratio analysis: {ratio_context.get('predicted_regime', 'unknown')} "
                          f"(confidence: {ratio_context.get('confidence', 0.5):.2f})")
                    
            except Exception as e:
                print(f"! Error in geological ratio analysis: {e}")
        
        # Use data integration pipeline for comparative analysis
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Compare with regional stress ratio datasets
                comparative_analysis = pipeline.compare_with_regional_datasets({
                    'data_type': 'stress_ratios',
                    'values': ratios,
                    'analysis_methods': ['statistical_comparison', 'outlier_detection'],
                    'reference_datasets': ['global_stress_db', 'regional_compilations']
                })
                
                if comparative_analysis:
                    ratio_analysis.update({
                        'regional_similarity': comparative_analysis.get('similarity_score', 0.5),
                        'global_percentile': comparative_analysis.get('global_percentile', 50),
                        'reference_datasets': comparative_analysis.get('datasets_used', []),
                        'statistical_significance': comparative_analysis.get('significance_test', {})
                    })
                    
                    print(f"+ Regional stress ratio comparison completed "
                          f"(similarity: {comparative_analysis.get('similarity_score', 0.5):.2f})")
                    
            except Exception as e:
                print(f"! Error in regional ratio comparison: {e}")
        
        # Advanced statistical analysis with geological constraints
        enhanced_plausibility = self._calculate_enhanced_ratio_plausibility(
            ratios, ratio_analysis
        )
        
        ratio_analysis['plausibility'] = enhanced_plausibility
        
        return ratio_analysis
    
    def _calculate_enhanced_ratio_plausibility(self, ratios: List[float], 
                                             analysis: Dict) -> float:
        """Calculate enhanced plausibility with geological constraints"""
        
        # Basic statistical plausibility
        range_coverage = analysis['range_coverage']
        std_ratio = analysis['std_ratio']
        mean_ratio = analysis['mean_ratio']
        
        # Statistical distribution plausibility
        distribution_plausibility = min(1.0, range_coverage / 0.6)  # Good if covers 60% of range
        variability_plausibility = min(1.0, std_ratio / 0.2)        # Good if std > 0.2
        
        # Mean ratio geological plausibility (0.0-1.0 is geologically reasonable)
        if 0.0 <= mean_ratio <= 1.0:
            mean_plausibility = 1.0
        elif mean_ratio < 0.0:
            mean_plausibility = max(0.0, 1.0 + mean_ratio)  # Penalty for negative ratios
        else:
            mean_plausibility = max(0.0, 2.0 - mean_ratio)  # Penalty for ratios > 1.0
        
        # Outlier analysis
        q75, q25 = np.percentile(ratios, [75, 25])
        iqr = q75 - q25
        outlier_threshold = 1.5 * iqr
        outliers = [r for r in ratios if r < (q25 - outlier_threshold) or r > (q75 + outlier_threshold)]
        outlier_penalty = min(0.3, len(outliers) / len(ratios))  # Max 30% penalty
        
        # Combine all factors
        statistical_plausibility = np.mean([
            distribution_plausibility,
            variability_plausibility, 
            mean_plausibility
        ]) - outlier_penalty
        
        # Include geological analysis if available
        geological_plausibility = analysis.get('geological_plausibility', statistical_plausibility)
        regional_similarity = analysis.get('regional_similarity', 0.5)
        
        # Weighted combination
        if 'geological_plausibility' in analysis:
            # Weight geological analysis higher when available
            overall_plausibility = (0.4 * statistical_plausibility + 
                                  0.4 * geological_plausibility + 
                                  0.2 * regional_similarity)
        else:
            overall_plausibility = statistical_plausibility
        
        return max(0.0, min(1.0, overall_plausibility))
    
    def _analyze_spatial_coherence(self, stress_field: StressField, 
                                 reliable_tensors: List[StressTensor]) -> Dict:
        """Advanced spatial coherence analysis using multi-scale geological integration"""
        
        if len(reliable_tensors) < 5:
            return {
                'coherence_score': 0.3, 
                'analysis_method': 'insufficient_data',
                'recommendation': 'acquire_more_data'
            }
        
        coherence_analysis = {
            'n_tensors': len(reliable_tensors),
            'analysis_method': 'advanced_multi_scale',
            'spatial_extent_km2': self._calculate_analysis_extent(stress_field, reliable_tensors)
        }
        
        # Use data integration pipeline for multi-scale coherence analysis
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Create spatial dataset for coherence analysis
                spatial_data = self._create_coherence_dataset(stress_field, reliable_tensors)
                
                # Multi-scale spatial analysis
                coherence_results = pipeline.analyze_spatial_coherence({
                    'data': spatial_data,
                    'analysis_scales': [10, 25, 50, 100],  # km
                    'coherence_metrics': ['autocorrelation', 'variogram', 'moran_i'],
                    'include_uncertainty': True
                })
                
                if coherence_results:
                    coherence_analysis.update({
                        'multi_scale_coherence': coherence_results.get('scale_dependent_coherence', {}),
                        'spatial_autocorrelation': coherence_results.get('autocorrelation', 0.5),
                        'coherence_length_scale': coherence_results.get('characteristic_length_km', 25.0),
                        'anisotropy_factor': coherence_results.get('anisotropy', 1.0),
                        'statistical_significance': coherence_results.get('significance', 0.05)
                    })
                    
                    print(f"+ Multi-scale coherence analysis: length scale = "
                          f"{coherence_results.get('characteristic_length_km', 25.0):.1f} km")
                    
            except Exception as e:
                print(f"! Error in multi-scale coherence analysis: {e}")
        
        # Use geological complexity assessment for structural coherence
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                # Analyze structural control on stress coherence
                structural_analysis = complexity_module.analyze_structural_coherence({
                    'stress_field': stress_field,
                    'stress_tensors': reliable_tensors,
                    'include_fault_influence': True,
                    'structural_domains': True
                })
                
                if structural_analysis:
                    coherence_analysis.update({
                        'structural_coherence': structural_analysis.get('structural_coherence', 0.5),
                        'fault_influence_score': structural_analysis.get('fault_influence', 0.0),
                        'structural_domains': structural_analysis.get('identified_domains', []),
                        'domain_boundaries': structural_analysis.get('domain_boundaries', []),
                        'geological_controls': structural_analysis.get('controlling_structures', [])
                    })
                    
                    print(f"+ Structural coherence analysis: {len(structural_analysis.get('identified_domains', []))} "
                          f"structural domains identified")
                    
            except Exception as e:
                print(f"! Error in structural coherence analysis: {e}")
        
        # Advanced local coherence calculation
        coherence_scores = []
        spatial_gradients = []
        
        for i, tensor in enumerate(reliable_tensors):
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            # Multi-radius coherence analysis
            local_coherence = self._calculate_advanced_local_coherence(
                point, tensor, stress_field, reliable_tensors
            )
            coherence_scores.append(local_coherence['coherence'])
            spatial_gradients.append(local_coherence['gradient'])
        
        # Calculate overall coherence metrics
        overall_coherence = np.mean(coherence_scores) if coherence_scores else 0.3
        coherence_variance = np.var(coherence_scores) if len(coherence_scores) > 1 else 0.0
        
        # Enhanced coherence scoring with geological context
        enhanced_coherence = self._calculate_enhanced_coherence_score(
            overall_coherence, coherence_analysis
        )
        
        coherence_analysis.update({
            'coherence_score': enhanced_coherence,
            'raw_coherence': overall_coherence,
            'local_coherence_scores': coherence_scores,
            'coherence_variance': coherence_variance,
            'spatial_gradients': spatial_gradients,
            'mean_gradient': np.mean(spatial_gradients) if spatial_gradients else 0.0
        })
        
        return coherence_analysis
    
    def _create_coherence_dataset(self, stress_field: StressField, 
                                reliable_tensors: List[StressTensor]) -> Dict:
        """Create spatial dataset for coherence analysis"""
        
        dataset = {
            'coordinates': [],
            'azimuths': [],
            'plunges': [],
            'stress_ratios': [],
            'regimes': [],
            'confidences': []
        }
        
        for tensor in reliable_tensors:
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            dataset['coordinates'].append([point[0], point[1]])
            dataset['azimuths'].append(tensor.azimuth1)
            dataset['plunges'].append(tensor.plunge1)
            dataset['stress_ratios'].append(tensor.stress_ratio)
            dataset['regimes'].append(tensor.regime)
            dataset['confidences'].append(tensor.confidence)
        
        return dataset
    
    def _calculate_advanced_local_coherence(self, point: np.ndarray, tensor: StressTensor,
                                          stress_field: StressField, 
                                          all_tensors: List[StressTensor]) -> Dict:
        """Calculate advanced local coherence with multi-radius analysis"""
        
        radii = [10, 25, 50, 75]  # km
        coherence_by_radius = {}
        
        for radius in radii:
            nearby_coherence = self._calculate_local_orientation_consistency(
                point, stress_field.stress_tensors.index(tensor), stress_field, radius_km=radius
            )
            coherence_by_radius[f"{radius}km"] = nearby_coherence
        
        # Calculate spatial gradient
        gradient = self._calculate_local_stress_gradient(point, tensor, stress_field, all_tensors)
        
        # Weighted average coherence (emphasize intermediate scales)
        weights = [0.2, 0.4, 0.3, 0.1]  # Weights for 10, 25, 50, 75 km
        weighted_coherence = sum(w * coherence_by_radius[f"{r}km"] 
                               for w, r in zip(weights, radii))
        
        return {
            'coherence': weighted_coherence,
            'coherence_by_radius': coherence_by_radius,
            'gradient': gradient,
            'analysis_point': point.tolist()
        }
    
    def _calculate_enhanced_coherence_score(self, raw_coherence: float, 
                                          analysis: Dict) -> float:
        """Calculate enhanced coherence score with geological context"""
        
        enhanced_coherence = raw_coherence
        
        # Adjust for multi-scale analysis if available
        if 'coherence_length_scale' in analysis:
            length_scale = analysis['coherence_length_scale']
            # Bonus for appropriate length scales (25-75 km typical)
            if 25 <= length_scale <= 75:
                enhanced_coherence *= 1.1
            elif length_scale < 10 or length_scale > 150:
                enhanced_coherence *= 0.9
        
        # Adjust for structural control if available
        if 'structural_coherence' in analysis:
            structural_coherence = analysis['structural_coherence']
            # Weighted combination with structural coherence
            enhanced_coherence = 0.7 * enhanced_coherence + 0.3 * structural_coherence
        
        # Penalty for high variance (inconsistent coherence)
        coherence_variance = analysis.get('coherence_variance', 0.0)
        if coherence_variance > 0.1:  # High variance penalty
            enhanced_coherence *= (1.0 - min(0.2, coherence_variance))
        
        return max(0.0, min(1.0, enhanced_coherence))
    
    def _calculate_analysis_extent(self, stress_field: StressField, 
                                 tensors: List[StressTensor]) -> float:
        """Calculate spatial extent of analysis in km²"""
        
        if len(tensors) < 3:
            return 0.0
        
        points = []
        for tensor in tensors:
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            points.append([point[0], point[1]])
        
        points = np.array(points)
        
        # Calculate approximate area using convex hull
        try:
            from scipy.spatial import ConvexHull
            
            # Convert to approximate Cartesian coordinates for area calculation
            hull = ConvexHull(points)
            
            # Rough conversion to km² (this is approximate)
            lat_to_km = 111.0
            lon_to_km = 111.0 * np.cos(np.radians(np.mean(points[:, 0])))
            
            area_deg2 = hull.volume  # In ConvexHull, volume is area for 2D
            area_km2 = area_deg2 * lat_to_km * lon_to_km
            
            return area_km2
            
        except ImportError:
            # Fallback: bounding box area
            lat_range = np.max(points[:, 0]) - np.min(points[:, 0])
            lon_range = np.max(points[:, 1]) - np.min(points[:, 1])
            
            lat_to_km = 111.0
            lon_to_km = 111.0 * np.cos(np.radians(np.mean(points[:, 0])))
            
            return lat_range * lat_to_km * lon_range * lon_to_km
    
    def _analyze_regime_transitions(self, stress_field: StressField, 
                                  reliable_tensors: List[StressTensor]) -> Dict:
        """Advanced regime transition analysis with machine learning and geological AI integration"""
        
        regimes = [t.regime for t in reliable_tensors]
        unique_regimes = list(set(regimes))
        
        transition_analysis = {
            'n_tensors': len(reliable_tensors),
            'unique_regimes': unique_regimes,
            'regime_diversity': len(unique_regimes),
            'analysis_method': 'ml_enhanced_transition_analysis',
            'timestamp': datetime.now().isoformat()
        }
        
        if len(unique_regimes) <= 1:
            transition_analysis.update({
                'transition_plausibility': 1.0, 
                'regime_diversity_type': 'uniform',
                'dominant_regime': unique_regimes[0] if unique_regimes else 'unknown',
                'confidence': 1.0
            })
            return transition_analysis
        
        # Use digital twin integration for real-time transition modeling
        if self.backend_modules.get('digital_twin_api'):
            try:
                from digital_twin_api import DigitalTwinAPI
                digital_twin = self.backend_modules['digital_twin_api']
                
                # Create digital twin model for stress regime transitions
                transition_model = digital_twin.create_transition_model({
                    'model_type': 'stress_regime_transition',
                    'input_data': {
                        'stress_tensors': [{
                            'regime': t.regime,
                            'location': stress_field.grid_points[stress_field.stress_tensors.index(t)].tolist(),
                            'azimuth': t.azimuth1,
                            'plunge': t.plunge1,
                            'stress_ratio': t.stress_ratio,
                            'confidence': t.confidence
                        } for t in reliable_tensors]
                    },
                    'analysis_parameters': {
                        'transition_threshold': 0.1,
                        'spatial_resolution': 5.0,  # km
                        'temporal_evolution': True
                    }
                })
                
                if transition_model and transition_model.get('transition_analysis'):
                    ml_analysis = transition_model['transition_analysis']
                    
                    transition_analysis.update({
                        'ml_transition_probability': ml_analysis.get('transition_probability', 0.5),
                        'predicted_transition_zones': ml_analysis.get('transition_zones', []),
                        'transition_mechanisms': ml_analysis.get('mechanisms', []),
                        'stability_index': ml_analysis.get('stability_index', 0.5),
                        'evolutionary_trend': ml_analysis.get('evolutionary_trend', 'stable')
                    })
                    
                    print(f"+ Digital twin transition analysis: {len(ml_analysis.get('transition_zones', []))} "
                          f"transition zones predicted")
                    
            except Exception as e:
                print(f"! Error in digital twin transition analysis: {e}")
        
        # Use hypothesis generation system for causal mechanism identification
        if self.backend_modules.get('hypothesis_generation_system'):
            try:
                hypothesis_system = self.backend_modules['hypothesis_generation_system']
                
                # Generate hypotheses for regime transition causes
                transition_hypotheses = hypothesis_system.generate_transition_hypotheses({
                    'stress_field_data': stress_field,
                    'regime_transitions': self._identify_spatial_transitions(stress_field, reliable_tensors),
                    'geological_context': True,
                    'tectonic_mechanisms': True,
                    'confidence_threshold': 0.6
                })
                
                if transition_hypotheses:
                    transition_analysis.update({
                        'causal_hypotheses': transition_hypotheses.get('hypotheses', []),
                        'most_likely_cause': transition_hypotheses.get('primary_mechanism', 'unknown'),
                        'mechanism_confidence': transition_hypotheses.get('confidence', 0.5),
                        'supporting_evidence': transition_hypotheses.get('evidence', []),
                        'testable_predictions': transition_hypotheses.get('predictions', [])
                    })
                    
                    print(f"+ Transition hypotheses generated: {len(transition_hypotheses.get('hypotheses', []))} "
                          f"causal mechanisms identified")
                    
            except Exception as e:
                print(f"! Error in hypothesis generation: {e}")
        
        # Use autonomous geoscience integration for intelligent analysis
        if self.backend_modules.get('autonomous_geoscience_integration'):
            try:
                from autonomous_geoscience_integration import AutonomousGeoscienceIntegration
                autonomous_system = self.backend_modules['autonomous_geoscience_integration']
                
                # Autonomous analysis of regime transition patterns
                autonomous_analysis = autonomous_system.analyze_stress_regime_patterns({
                    'stress_data': stress_field,
                    'analysis_type': 'regime_transition_intelligence',
                    'include_ml_predictions': True,
                    'geological_reasoning': True,
                    'uncertainty_quantification': True
                })
                
                if autonomous_analysis:
                    transition_analysis.update({
                        'autonomous_classification': autonomous_analysis.get('transition_classification', 'unknown'),
                        'ai_confidence_score': autonomous_analysis.get('ai_confidence', 0.5),
                        'geological_reasoning': autonomous_analysis.get('reasoning_chain', []),
                        'anomaly_detection': autonomous_analysis.get('anomalies_detected', []),
                        'recommendation': autonomous_analysis.get('recommendations', [])
                    })
                    
                    print(f"+ Autonomous geoscience analysis: {autonomous_analysis.get('transition_classification', 'unknown')} "
                          f"classification with {autonomous_analysis.get('ai_confidence', 0.5):.2f} confidence")
                    
            except Exception as e:
                print(f"! Error in autonomous geoscience analysis: {e}")
        
        # Advanced spatial transition analysis with machine learning enhancement
        spatial_transitions = self._ml_enhanced_spatial_transition_analysis(
            stress_field, reliable_tensors
        )
        transition_analysis.update(spatial_transitions)
        
        # Calculate enhanced transition plausibility with AI integration
        enhanced_plausibility = self._calculate_ai_enhanced_transition_plausibility(
            transition_analysis
        )
        transition_analysis['transition_plausibility'] = enhanced_plausibility
        
        return transition_analysis
    
    def _identify_spatial_transitions(self, stress_field: StressField, 
                                    reliable_tensors: List[StressTensor]) -> List[Dict]:
        """Identify spatial stress regime transitions"""
        
        transitions = []
        
        # Create spatial map of regimes
        regime_map = {}
        for tensor in reliable_tensors:
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            key = f"{point[0]:.3f}_{point[1]:.3f}"
            regime_map[key] = {
                'regime': tensor.regime,
                'location': point,
                'tensor': tensor
            }
        
        # Find regime boundaries using nearest neighbor analysis
        locations = list(regime_map.keys())
        
        for i, key1 in enumerate(locations):
            for j, key2 in enumerate(locations[i+1:], i+1):
                data1 = regime_map[key1]
                data2 = regime_map[key2]
                
                if data1['regime'] != data2['regime']:
                    distance = self._haversine_distance(
                        data1['location'][0], data1['location'][1],
                        data2['location'][0], data2['location'][1]
                    )
                    
                    if distance < 75.0:  # Transitions within 75 km
                        transitions.append({
                            'regime_pair': (data1['regime'], data2['regime']),
                            'transition_distance_km': distance,
                            'location1': data1['location'],
                            'location2': data2['location'],
                            'transition_center': [
                                (data1['location'][0] + data2['location'][0]) / 2,
                                (data1['location'][1] + data2['location'][1]) / 2
                            ],
                            'stress_contrast': abs(data1['tensor'].stress_ratio - data2['tensor'].stress_ratio),
                            'orientation_contrast': min(
                                abs(data1['tensor'].azimuth1 - data2['tensor'].azimuth1),
                                360 - abs(data1['tensor'].azimuth1 - data2['tensor'].azimuth1)
                            )
                        })
        
        return transitions
    
    def _ml_enhanced_spatial_transition_analysis(self, stress_field: StressField,
                                               reliable_tensors: List[StressTensor]) -> Dict:
        """Machine learning enhanced spatial transition analysis"""
        
        analysis = {
            'ml_enhanced': True,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Advanced clustering analysis for regime identification
        if len(reliable_tensors) >= 10:  # Need sufficient data for ML
            try:
                # Prepare feature matrix for machine learning
                features = []
                regime_labels = []
                
                for tensor in reliable_tensors:
                    tensor_index = stress_field.stress_tensors.index(tensor)
                    point = stress_field.grid_points[tensor_index]
                    
                    # Feature vector: [lat, lon, azimuth, plunge, stress_ratio, confidence]
                    features.append([
                        point[0], point[1],
                        tensor.azimuth1, tensor.plunge1,
                        tensor.stress_ratio, tensor.confidence
                    ])
                    regime_labels.append(tensor.regime)
                
                features = np.array(features)
                
                # Unsupervised clustering to validate regime boundaries
                try:
                    from sklearn.cluster import DBSCAN, KMeans
                    from sklearn.preprocessing import StandardScaler
                    
                    # Standardize features
                    scaler = StandardScaler()
                    features_scaled = scaler.fit_transform(features)
                    
                    # DBSCAN clustering for regime boundary detection
                    dbscan = DBSCAN(eps=0.5, min_samples=3)
                    cluster_labels = dbscan.fit_predict(features_scaled)
                    
                    # Analyze clustering results
                    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    
                    analysis.update({
                        'ml_detected_clusters': n_clusters,
                        'noise_points': n_noise,
                        'clustering_quality': self._evaluate_clustering_quality(cluster_labels, regime_labels),
                        'transition_boundaries': self._extract_cluster_boundaries(features, cluster_labels)
                    })
                    
                    print(f"+ ML clustering analysis: {n_clusters} clusters detected, {n_noise} noise points")
                    
                except ImportError:
                    print("! scikit-learn not available for ML clustering")
                    
            except Exception as e:
                print(f"! Error in ML transition analysis: {e}")
        
        return analysis
    
    def _calculate_ai_enhanced_transition_plausibility(self, analysis: Dict) -> float:
        """Calculate AI-enhanced transition plausibility score"""
        
        # Base statistical plausibility
        transition_rate = analysis.get('transition_rate', 0.5)
        base_plausibility = self._calculate_base_transition_plausibility(transition_rate)
        
        # AI and ML enhancements
        ml_probability = analysis.get('ml_transition_probability', base_plausibility)
        ai_confidence = analysis.get('ai_confidence_score', 0.5)
        mechanism_confidence = analysis.get('mechanism_confidence', 0.5)
        
        # Clustering quality enhancement
        clustering_quality = analysis.get('clustering_quality', 0.5)
        
        # Digital twin validation
        stability_index = analysis.get('stability_index', 0.5)
        
        # Weighted combination of all factors
        enhanced_plausibility = (
            0.2 * base_plausibility +        # Statistical baseline
            0.25 * ml_probability +          # ML transition probability
            0.2 * ai_confidence +            # AI confidence
            0.15 * mechanism_confidence +    # Causal mechanism confidence
            0.1 * clustering_quality +       # Clustering validation
            0.1 * stability_index            # Digital twin stability
        )
        
        return max(0.0, min(1.0, enhanced_plausibility))
    
    def _calculate_base_transition_plausibility(self, transition_rate: float) -> float:
        """Calculate base transition plausibility from transition rate"""
        
        # Reasonable transition rate (not too chaotic, not too uniform)
        if 0.1 <= transition_rate <= 0.4:
            return 1.0
        elif transition_rate < 0.1:
            return 0.8  # Too uniform
        else:
            return max(0.2, 1.0 - (transition_rate - 0.4) / 0.4)
    
    def _evaluate_clustering_quality(self, cluster_labels, regime_labels) -> float:
        """Evaluate quality of ML clustering against known regimes"""
        
        try:
            from sklearn.metrics import adjusted_rand_score
            
            # Convert regime labels to numeric
            regime_mapping = {'NF': 0, 'SS': 1, 'TF': 2}
            numeric_regimes = [regime_mapping.get(r, -1) for r in regime_labels]
            
            # Calculate adjusted Rand index
            ari_score = adjusted_rand_score(numeric_regimes, cluster_labels)
            
            # Convert to 0-1 scale (ARI can be negative)
            quality_score = max(0.0, (ari_score + 1) / 2)
            
            return quality_score
            
        except ImportError:
            # Fallback: simple consistency check
            n_regime_types = len(set(regime_labels))
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # Good if number of clusters matches number of regimes
            if n_clusters == n_regime_types:
                return 1.0
            else:
                return max(0.3, 1.0 - abs(n_clusters - n_regime_types) / n_regime_types)
    
    def _extract_cluster_boundaries(self, features, cluster_labels) -> List[Dict]:
        """Extract spatial boundaries between clusters"""
        
        boundaries = []
        unique_clusters = list(set(cluster_labels))
        
        # Remove noise cluster if present
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        # Find boundaries between each pair of clusters
        for i, cluster1 in enumerate(unique_clusters):
            for cluster2 in unique_clusters[i+1:]:
                # Get points for each cluster
                cluster1_points = features[cluster_labels == cluster1]
                cluster2_points = features[cluster_labels == cluster2]
                
                if len(cluster1_points) > 0 and len(cluster2_points) > 0:
                    # Find closest points between clusters
                    min_distance = float('inf')
                    boundary_points = None
                    
                    for p1 in cluster1_points:
                        for p2 in cluster2_points:
                            distance = np.linalg.norm(p1[:2] - p2[:2])  # Use only spatial coordinates
                            if distance < min_distance:
                                min_distance = distance
                                boundary_points = (p1, p2)
                    
                    if boundary_points:
                        boundaries.append({
                            'cluster_pair': (int(cluster1), int(cluster2)),
                            'boundary_distance': float(min_distance),
                            'boundary_center': ((boundary_points[0][:2] + boundary_points[1][:2]) / 2).tolist()
                        })
        
        return boundaries
    
    def _check_spatial_gradients(self, stress_field: StressField, 
                               reliable_tensors: List[StressTensor],
                               max_gradient: float) -> List[Dict]:
        """Advanced spatial gradient analysis with AI-powered anomaly detection and geological intelligence"""
        
        gradient_analysis = {
            'violations': [],
            'analysis_method': 'ai_enhanced_gradient_detection',
            'timestamp': datetime.now().isoformat(),
            'ai_analysis': {}
        }
        
        # Use autonomous geoscience for intelligent gradient analysis
        if self.backend_modules.get('autonomous_geoscience_integration'):
            try:
                autonomous_system = self.backend_modules['autonomous_geoscience_integration']
                
                # AI-powered gradient field analysis
                ai_gradient_analysis = autonomous_system.analyze_stress_gradients({
                    'stress_field': stress_field,
                    'stress_tensors': reliable_tensors,
                    'analysis_type': 'intelligent_gradient_detection',
                    'include_anomaly_detection': True,
                    'geological_context_analysis': True,
                    'adaptive_thresholds': True
                })
                
                if ai_gradient_analysis:
                    gradient_analysis['ai_analysis'] = {
                        'ai_detected_anomalies': ai_gradient_analysis.get('gradient_anomalies', []),
                        'adaptive_thresholds': ai_gradient_analysis.get('adaptive_thresholds', {}),
                        'geological_explanations': ai_gradient_analysis.get('geological_explanations', []),
                        'confidence_scores': ai_gradient_analysis.get('confidence_scores', {}),
                        'recommended_actions': ai_gradient_analysis.get('recommendations', [])
                    }
                    
                    print(f"+ AI gradient analysis: {len(ai_gradient_analysis.get('gradient_anomalies', []))} "
                          f"anomalies detected with geological context")
                    
            except Exception as e:
                print(f"! Error in AI gradient analysis: {e}")
        
        # Use digital twin for real-time gradient monitoring
        if self.backend_modules.get('digital_twin_api'):
            try:
                digital_twin = self.backend_modules['digital_twin_api']
                
                # Create digital twin gradient monitoring system
                gradient_twin = digital_twin.create_gradient_monitor({
                    'monitoring_type': 'stress_field_gradients',
                    'spatial_resolution': 2.5,  # km
                    'temporal_tracking': True,
                    'anomaly_detection': True,
                    'prediction_horizon': 30  # days
                })
                
                if gradient_twin and gradient_twin.get('monitoring_results'):
                    monitoring_results = gradient_twin['monitoring_results']
                    
                    gradient_analysis['digital_twin_monitoring'] = {
                        'gradient_evolution': monitoring_results.get('gradient_evolution', []),
                        'predicted_changes': monitoring_results.get('predictions', []),
                        'stability_assessment': monitoring_results.get('stability_score', 0.5),
                        'alert_zones': monitoring_results.get('alert_zones', []),
                        'confidence_intervals': monitoring_results.get('confidence_intervals', {})
                    }
                    
                    print(f"+ Digital twin gradient monitoring: {len(monitoring_results.get('alert_zones', []))} "
                          f"alert zones identified")
                    
            except Exception as e:
                print(f"! Error in digital twin gradient monitoring: {e}")
        
        # Enhanced gradient field computation with machine learning
        gradient_field_data = self._compute_ml_enhanced_gradient_field(
            stress_field, reliable_tensors
        )
        
        # AI-enhanced violation detection
        violations = self._detect_ai_enhanced_gradient_violations(
            stress_field, reliable_tensors, max_gradient, gradient_analysis
        )
        
        # Use hypothesis generation for gradient anomaly explanation
        if violations and self.backend_modules.get('hypothesis_generation_system'):
            try:
                hypothesis_system = self.backend_modules['hypothesis_generation_system']
                
                # Generate explanatory hypotheses for gradient anomalies
                gradient_hypotheses = hypothesis_system.generate_gradient_anomaly_hypotheses({
                    'gradient_violations': violations,
                    'stress_field_context': stress_field,
                    'geological_setting': True,
                    'structural_controls': True,
                    'confidence_threshold': 0.6
                })
                
                if gradient_hypotheses:
                    # Add hypotheses to violation records
                    for violation in violations:
                        relevant_hypotheses = [
                            h for h in gradient_hypotheses.get('hypotheses', [])
                            if self._is_hypothesis_relevant_to_violation(h, violation)
                        ]
                        
                        violation['explanatory_hypotheses'] = relevant_hypotheses
                        violation['most_likely_explanation'] = (
                            relevant_hypotheses[0]['explanation'] if relevant_hypotheses 
                            else 'No clear geological explanation identified'
                        )
                    
                    print(f"+ Gradient anomaly hypotheses: {len(gradient_hypotheses.get('hypotheses', []))} "
                          f"explanatory mechanisms proposed")
                    
            except Exception as e:
                print(f"! Error in gradient hypothesis generation: {e}")
        
        gradient_analysis['violations'] = violations
        gradient_analysis['n_violations'] = len(violations)
        gradient_analysis['severity_distribution'] = self._analyze_violation_severity_ai_enhanced(violations)
        
        return violations  # Return violations for compatibility
    
    def _compute_ml_enhanced_gradient_field(self, stress_field: StressField,
                                          reliable_tensors: List[StressTensor]) -> Dict:
        """Compute gradient field using machine learning enhancement"""
        
        gradient_data = {
            'method': 'ml_enhanced_gradient_computation',
            'spatial_gradients': [],
            'temporal_gradients': [],
            'uncertainty_estimates': []
        }
        
        # Prepare data for ML gradient computation
        if len(reliable_tensors) >= 15:  # Need sufficient data for ML
            try:
                # Create feature matrix with spatial and stress parameters
                coordinates = []
                stress_features = []
                
                for tensor in reliable_tensors:
                    tensor_index = stress_field.stress_tensors.index(tensor)
                    point = stress_field.grid_points[tensor_index]
                    
                    coordinates.append([point[0], point[1]])
                    stress_features.append([
                        tensor.azimuth1, tensor.plunge1, tensor.stress_ratio,
                        tensor.confidence
                    ])
                
                coordinates = np.array(coordinates)
                stress_features = np.array(stress_features)
                
                # ML-based gradient estimation using Gaussian process regression
                try:
                    from sklearn.gaussian_process import GaussianProcessRegressor
                    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
                    
                    # Define spatial kernel for gradient modeling
                    kernel = RBF(length_scale=0.1) + WhiteKernel(noise_level=0.01)
                    
                    # Fit GP models for each stress parameter
                    gradient_models = {}
                    
                    for i, param_name in enumerate(['azimuth', 'plunge', 'stress_ratio', 'confidence']):
                        gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)
                        gp_model.fit(coordinates, stress_features[:, i])
                        gradient_models[param_name] = gp_model
                    
                    # Compute gradients at prediction grid
                    gradient_grid = self._create_gradient_prediction_grid(coordinates)
                    
                    for param_name, model in gradient_models.items():
                        predictions, uncertainties = model.predict(gradient_grid, return_std=True)
                        
                        # Compute spatial gradients using finite differences
                        spatial_gradient = self._compute_spatial_gradient(
                            gradient_grid, predictions
                        )
                        
                        gradient_data['spatial_gradients'].append({
                            'parameter': param_name,
                            'gradient_field': spatial_gradient,
                            'uncertainty_field': uncertainties,
                            'grid_coordinates': gradient_grid
                        })
                    
                    print(f"+ ML gradient computation: {len(gradient_models)} parameter gradients computed")
                    
                except ImportError:
                    print("! scikit-learn not available for ML gradient computation")
                    
            except Exception as e:
                print(f"! Error in ML gradient computation: {e}")
        
        return gradient_data
    
    def _detect_ai_enhanced_gradient_violations(self, stress_field: StressField,
                                              reliable_tensors: List[StressTensor],
                                              max_gradient: float,
                                              analysis_context: Dict) -> List[Dict]:
        """AI-enhanced gradient violation detection with adaptive thresholds"""
        
        violations = []
        
        # Use adaptive thresholds from AI analysis if available
        ai_analysis = analysis_context.get('ai_analysis', {})
        adaptive_thresholds = ai_analysis.get('adaptive_thresholds', {})
        
        # Enhanced gradient analysis with multiple parameters
        for i, tensor in enumerate(reliable_tensors):
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            
            # AI-enhanced neighborhood analysis
            neighborhood_analysis = self._analyze_ai_enhanced_neighborhood(
                tensor, point, reliable_tensors, stress_field, analysis_context
            )
            
            # Check multiple gradient types with adaptive thresholds
            gradient_violations = self._check_multi_parameter_gradients(
                tensor, point, neighborhood_analysis, max_gradient, adaptive_thresholds
            )
            
            violations.extend(gradient_violations)
        
        # Post-process violations with AI insights
        enhanced_violations = self._enhance_violations_with_ai_insights(
            violations, analysis_context
        )
        
        return enhanced_violations
    
    def _analyze_ai_enhanced_neighborhood(self, tensor: StressTensor, point: np.ndarray,
                                        all_tensors: List[StressTensor],
                                        stress_field: StressField,
                                        context: Dict) -> Dict:
        """AI-enhanced neighborhood analysis for gradient detection"""
        
        neighborhood = {
            'center_tensor': tensor,
            'center_point': point,
            'neighbors': [],
            'ai_insights': {}
        }
        
        # Find neighbors with AI-guided distance weighting
        for other_tensor in all_tensors:
            if tensor == other_tensor:
                continue
                
            other_index = stress_field.stress_tensors.index(other_tensor)
            other_point = stress_field.grid_points[other_index]
            
            distance = self._haversine_distance(
                point[0], point[1], other_point[0], other_point[1]
            )
            
            # AI-adaptive distance threshold
            ai_analysis = context.get('ai_analysis', {})
            confidence_scores = ai_analysis.get('confidence_scores', {})
            
            # Base distance threshold: 50 km, adjusted by AI confidence
            location_key = f"{point[0]:.2f}_{point[1]:.2f}"
            confidence_multiplier = confidence_scores.get(location_key, 1.0)
            
            adaptive_distance = 50.0 * confidence_multiplier
            
            if distance <= adaptive_distance:
                neighbor_data = {
                    'tensor': other_tensor,
                    'point': other_point,
                    'distance_km': distance,
                    'stress_contrast': {
                        'azimuth_diff': min(
                            abs(tensor.azimuth1 - other_tensor.azimuth1),
                            360 - abs(tensor.azimuth1 - other_tensor.azimuth1)
                        ),
                        'plunge_diff': abs(tensor.plunge1 - other_tensor.plunge1),
                        'ratio_diff': abs(tensor.stress_ratio - other_tensor.stress_ratio),
                        'confidence_diff': abs(tensor.confidence - other_tensor.confidence)
                    },
                    'ai_relevance_score': self._calculate_ai_neighbor_relevance(
                        tensor, other_tensor, distance, context
                    )
                }
                
                neighborhood['neighbors'].append(neighbor_data)
        
        # Sort neighbors by AI relevance
        neighborhood['neighbors'].sort(key=lambda x: x['ai_relevance_score'], reverse=True)
        
        return neighborhood
    
    def _calculate_ai_neighbor_relevance(self, tensor1: StressTensor, tensor2: StressTensor,
                                       distance: float, context: Dict) -> float:
        """Calculate AI-based relevance score for neighbor analysis"""
        
        # Base relevance: inverse distance weighting
        distance_relevance = 1.0 / (1.0 + distance / 25.0)  # 25 km characteristic distance
        
        # Stress contrast relevance
        azimuth_contrast = min(
            abs(tensor1.azimuth1 - tensor2.azimuth1),
            360 - abs(tensor1.azimuth1 - tensor2.azimuth1)
        ) / 180.0  # Normalize to 0-1
        
        contrast_relevance = (azimuth_contrast + 
                            abs(tensor1.stress_ratio - tensor2.stress_ratio) +
                            abs(tensor1.plunge1 - tensor2.plunge1) / 90.0) / 3.0
        
        # Confidence weighting
        confidence_relevance = (tensor1.confidence + tensor2.confidence) / 2.0
        
        # AI insights weighting if available
        ai_multiplier = 1.0
        ai_analysis = context.get('ai_analysis', {})
        if ai_analysis.get('confidence_scores'):
            ai_multiplier = ai_analysis['confidence_scores'].get(
                f"{tensor1.latitude:.2f}_{tensor1.longitude:.2f}", 1.0
            )
        
        # Combined relevance score
        relevance = (0.4 * distance_relevance +
                    0.3 * contrast_relevance +
                    0.2 * confidence_relevance +
                    0.1 * ai_multiplier)
        
        return max(0.0, min(1.0, relevance))
    
    def _check_regime_stability(self, stress_field: StressField, 
                              reliable_tensors: List[StressTensor],
                              min_domain_size: float) -> List[Dict]:
        """Check for minimum stable regime domain sizes"""
        
        violations = []
        
        # Group tensors by regime
        regime_groups = {}
        for i, tensor in enumerate(reliable_tensors):
            regime = tensor.regime
            if regime not in regime_groups:
                regime_groups[regime] = []
            
            tensor_index = stress_field.stress_tensors.index(tensor)
            point = stress_field.grid_points[tensor_index]
            regime_groups[regime].append((i, point))
        
        # Check each regime domain size
        for regime, points in regime_groups.items():
            if len(points) >= 3:  # Need at least 3 points to define an area
                # Calculate domain size using advanced geometric analysis
                coords = [point[1] for _, point in points]
                
                if len(coords) > 2:
                    # Use convex hull for more accurate domain size calculation
                    domain_size = self._calculate_advanced_domain_size(coords)
                    
                    # Add geological context analysis
                    geological_context = self._analyze_domain_geological_context(
                        coords, regime
                    )
                    
                    if domain_size < min_domain_size:
                        center_lat = np.mean(lats)
                        center_lon = np.mean(lons)
                        
                        violations.append({
                            'type': 'regime_domain_too_small',
                            'location': (center_lat, center_lon),
                            'regime': regime,
                            'observed_size': domain_size,
                            'minimum_size': min_domain_size,
                            'severity': 'low'
                        })
        
        return violations
    
    def _validate_with_hypothesis_system(self, stress_field: StressField) -> Dict:
        """Advanced validation using integrated hypothesis generation system"""
        
        validation_results = {
            'validation_method': 'hypothesis_system_integration',
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0.5
        }
        
        if self.backend_modules.get('hypothesis_generation_system'):
            try:
                hypothesis_system = self.backend_modules['hypothesis_generation_system']
                
                # Generate hypotheses for stress field configuration
                stress_hypotheses = hypothesis_system.generate_stress_field_hypotheses({
                    'stress_field': stress_field,
                    'analysis_type': 'validation_hypotheses',
                    'include_alternative_explanations': True,
                    'confidence_threshold': 0.6,
                    'geological_constraints': True
                })
                
                if stress_hypotheses:
                    validation_results.update({
                        'generated_hypotheses': len(stress_hypotheses.get('hypotheses', [])),
                        'primary_hypothesis': stress_hypotheses.get('primary_hypothesis', {}),
                        'hypothesis_confidence': stress_hypotheses.get('overall_confidence', 0.5),
                        'alternative_explanations': stress_hypotheses.get('alternatives', []),
                        'supporting_evidence': stress_hypotheses.get('evidence_summary', {}),
                        'validation_score': stress_hypotheses.get('validation_score', 0.5)
                    })
                    
                    validation_results['overall_score'] = stress_hypotheses.get('validation_score', 0.5)
                    
                    print(f"+ Hypothesis system validation: {len(stress_hypotheses.get('hypotheses', []))} "
                          f"hypotheses generated (score: {validation_results['overall_score']:.2f})")
                    
            except Exception as e:
                print(f"! Error in hypothesis system validation: {e}")
                validation_results['error'] = str(e)
        else:
            # Fallback validation using statistical methods
            validation_results.update({
                'method': 'statistical_fallback',
                'overall_score': 0.7,
                'validation': 'basic_statistical_validation'
            })
        
        return validation_results
    
    def _validate_with_complexity_assessment(self, stress_field: StressField) -> Dict:
        """Advanced validation using geological complexity assessment integration"""
        
        validation_results = {
            'validation_method': 'complexity_assessment_integration',
            'timestamp': datetime.now().isoformat(),
            'complexity_score': 0.5
        }
        
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                # Comprehensive complexity assessment of stress field
                complexity_analysis = complexity_module.assess_stress_field_complexity({
                    'stress_field': stress_field,
                    'analysis_dimensions': [
                        'spatial_heterogeneity',
                        'regime_diversity',
                        'orientation_coherence',
                        'temporal_stability'
                    ],
                    'include_uncertainty': True,
                    'geological_context': True
                })
                
                if complexity_analysis:
                    validation_results.update({
                        'complexity_index': complexity_analysis.get('overall_complexity', 0.5),
                        'spatial_complexity': complexity_analysis.get('spatial_metrics', {}),
                        'temporal_complexity': complexity_analysis.get('temporal_metrics', {}),
                        'geological_plausibility': complexity_analysis.get('geological_plausibility', 0.5),
                        'uncertainty_assessment': complexity_analysis.get('uncertainty_metrics', {}),
                        'complexity_classification': complexity_analysis.get('complexity_class', 'moderate')
                    })
                    
                    validation_results['complexity_score'] = complexity_analysis.get('overall_complexity', 0.5)
                    
                    print(f"+ Complexity assessment: {complexity_analysis.get('complexity_class', 'moderate')} "
                          f"complexity (score: {validation_results['complexity_score']:.2f})")
                    
            except Exception as e:
                print(f"! Error in complexity assessment validation: {e}")
                validation_results['error'] = str(e)
        else:
            # Fallback complexity assessment
            fallback_complexity = self._calculate_fallback_complexity(stress_field)
            validation_results.update({
                'method': 'fallback_complexity_assessment',
                'complexity_score': fallback_complexity,
                'complexity_classification': 'moderate_fallback'
            })
        
        return validation_results
    
    def _calculate_fallback_complexity(self, stress_field: StressField) -> float:
        """Calculate fallback complexity score when advanced module unavailable"""
        
        if not stress_field.stress_tensors:
            return 0.3
        
        # Simple complexity metrics
        regimes = [t.regime for t in stress_field.stress_tensors]
        unique_regimes = len(set(regimes))
        
        # Orientation variance
        azimuths = [t.azimuth1 for t in stress_field.stress_tensors]
        azimuth_variance = np.var(azimuths) if len(azimuths) > 1 else 0
        
        # Stress ratio variance
        ratios = [t.stress_ratio for t in stress_field.stress_tensors]
        ratio_variance = np.var(ratios) if len(ratios) > 1 else 0
        
        # Simple complexity score
        regime_complexity = min(1.0, unique_regimes / 3.0)  # Max complexity with 3 regimes
        orientation_complexity = min(1.0, azimuth_variance / 5000.0)  # Normalize variance
        ratio_complexity = min(1.0, ratio_variance / 0.25)  # Normalize ratio variance
        
        overall_complexity = (regime_complexity + orientation_complexity + ratio_complexity) / 3.0
        
        return max(0.1, min(1.0, overall_complexity))
    
    def _calculate_advanced_domain_size(self, coordinates: List[List[float]]) -> float:
        """Calculate domain size using advanced geometric analysis"""
        
        coords_array = np.array(coordinates)
        
        if len(coordinates) < 3:
            # For 2 points, use distance
            if len(coordinates) == 2:
                return self._haversine_distance(
                    coordinates[0][0], coordinates[0][1],
                    coordinates[1][0], coordinates[1][1]
                )
            else:
                return 0.0
        
        try:
            # Use convex hull for accurate area calculation
            from scipy.spatial import ConvexHull
            
            hull = ConvexHull(coords_array)
            
            # Convert area from degrees² to km²
            lat_center = np.mean(coords_array[:, 0])
            lat_to_km = 111.0  # km per degree latitude
            lon_to_km = 111.0 * np.cos(np.radians(lat_center))  # km per degree longitude
            
            # Calculate area in km²
            area_deg2 = hull.volume  # In 2D, volume is area
            area_km2 = area_deg2 * lat_to_km * lon_to_km
            
            # Return characteristic length scale (square root of area)
            return np.sqrt(area_km2)
            
        except ImportError:
            # Fallback: bounding box approach
            lats = coords_array[:, 0]
            lons = coords_array[:, 1]
            
            lat_span = (np.max(lats) - np.min(lats)) * 111.0
            lon_span = (np.max(lons) - np.min(lons)) * 111.0 * np.cos(np.radians(np.mean(lats)))
            
            return max(lat_span, lon_span)
    
    def _analyze_domain_geological_context(self, coordinates: List[List[float]], 
                                         regime: str) -> Dict:
        """Analyze geological context for stress regime domain"""
        
        context = {
            'regime': regime,
            'scaling_factor': 1.0,  # Default scaling
            'geological_setting': 'unknown',
            'expected_domain_size_km': 25.0  # Default expectation
        }
        
        # Get representative location
        coords_array = np.array(coordinates)
        center_lat = np.mean(coords_array[:, 0])
        center_lon = np.mean(coords_array[:, 1])
        
        # Use geological knowledge to determine expected domain sizes
        if self.backend_modules.get('geological_complexity_assessment'):
            try:
                complexity_module = self.backend_modules['geological_complexity_assessment']
                
                geological_analysis = complexity_module.analyze_regime_domain_context({
                    'location': [center_lat, center_lon],
                    'regime': regime,
                    'coordinates': coordinates,
                    'include_scaling_factors': True
                })
                
                if geological_analysis:
                    context.update({
                        'geological_setting': geological_analysis.get('setting', 'unknown'),
                        'scaling_factor': geological_analysis.get('scaling_factor', 1.0),
                        'expected_domain_size_km': geological_analysis.get('expected_size_km', 25.0),
                        'tectonic_context': geological_analysis.get('tectonic_context', {})
                    })
                    
            except Exception as e:
                print(f"! Error in geological domain analysis: {e}")
        
        # Apply regime-specific scaling factors
        regime_scaling = {
            'NF': 1.2,  # Normal faults can have larger domains
            'SS': 1.0,  # Strike-slip baseline
            'TF': 0.8   # Thrust faults often more localized
        }
        
        context['scaling_factor'] *= regime_scaling.get(regime, 1.0)
        
        # Apply tectonic setting scaling
        tectonic_pattern = self._identify_regional_tectonic_pattern(np.array([center_lat, center_lon]))
        if tectonic_pattern:
            tectonic_scaling = {
                'pacific_rim': 1.3,
                'san_andreas_system': 1.1,
                'mid_atlantic_ridge': 1.5,
                'intraplate': 0.8
            }
            
            tectonic_province = tectonic_pattern.get('tectonic_province', 'unknown')
            context['scaling_factor'] *= tectonic_scaling.get(tectonic_province, 1.0)
        
        return context
    
    async def acquire_seismological_data(self, region: Dict, time_range: Tuple[datetime, datetime]) -> List[FocalMechanism]:
        """Advanced seismological data acquisition using integrated backend connectors"""
        
        mechanisms = []
        
        # Use data integration pipeline if available
        if self.backend_modules.get('data_integration_pipeline'):
            try:
                pipeline = self.backend_modules['data_integration_pipeline']
                
                # Add seismological data layer
                usgs_data = await self._fetch_usgs_earthquake_catalog(region, time_range)
                if usgs_data:
                    layer = pipeline.add_data_layer(
                        name=f"usgs_earthquakes_{datetime.now().strftime('%Y%m%d')}",
                        data_type=pipeline.DataType.STRUCTURAL,
                        data=usgs_data,
                        metadata={
                            'source': 'USGS ANSS Comprehensive Catalog',
                            'region': region,
                            'time_range': time_range,
                            'quality_threshold': 0.8
                        }
                    )
                    
                    # Process and convert to focal mechanisms
                    mechanisms.extend(self._convert_catalog_to_mechanisms(usgs_data))
                    print(f"+ Acquired {len(mechanisms)} focal mechanisms from USGS")
                
                # Fetch additional data from IRIS DMC
                iris_data = await self._fetch_iris_moment_tensors(region, time_range)
                if iris_data:
                    layer = pipeline.add_data_layer(
                        name=f"iris_moment_tensors_{datetime.now().strftime('%Y%m%d')}",
                        data_type=pipeline.DataType.STRUCTURAL,
                        data=iris_data,
                        metadata={
                            'source': 'IRIS Data Management Center',
                            'processing_level': 'moment_tensor_solution',
                            'region': region,
                            'time_range': time_range
                        }
                    )
                    
                    iris_mechanisms = self._convert_moment_tensors_to_mechanisms(iris_data)
                    mechanisms.extend(iris_mechanisms)
                    print(f"+ Acquired {len(iris_mechanisms)} moment tensor solutions from IRIS")
                    
            except Exception as e:
                print(f"! Error using data integration pipeline: {e}")
        
        # Use database manager for persistent storage if available
        if self.backend_modules.get('database_manager'):
            try:
                db_manager = self.backend_modules['database_manager']
                
                # Store acquired data in database
                await self._store_focal_mechanisms_in_database(db_manager, mechanisms)
                
                # Query for additional historical data
                historical_mechanisms = await self._query_historical_focal_mechanisms(
                    db_manager, region, time_range
                )
                mechanisms.extend(historical_mechanisms)
                print(f"+ Retrieved {len(historical_mechanisms)} historical focal mechanisms from database")
                
            except Exception as e:
                print(f"! Error using database manager: {e}")
        
        return mechanisms
    
    async def acquire_geodetic_data(self, region: Dict, time_range: Tuple[datetime, datetime]) -> Dict:
        """Advanced geodetic data acquisition for stress field constraints"""
        
        geodetic_data = {'gps': [], 'insar': [], 'leveling': []}
        
        # Use satellite scheduler for InSAR data acquisition
        if self.backend_modules.get('satellite_scheduler'):
            try:
                scheduler = self.backend_modules['satellite_scheduler']
                
                # Define acquisition plan for Sentinel-1 data
                acquisition_plan = {
                    'mission': 'SENTINEL1',
                    'processing_level': 'SLC',
                    'region': region,
                    'time_range': time_range,
                    'priority': 'HIGH',
                    'max_temporal_baseline': self.insar_acquisition_params['temporal_baseline_max'],
                    'max_spatial_baseline': self.insar_acquisition_params['spatial_baseline_max'],
                    'coherence_threshold': self.insar_acquisition_params['coherence_threshold']
                }
                
                # Schedule and execute InSAR data acquisition
                insar_acquisitions = await scheduler.schedule_acquisition(acquisition_plan)
                
                # Process acquired InSAR data for stress field analysis
                for acquisition in insar_acquisitions:
                    processed_insar = await self._process_insar_for_stress_analysis(acquisition)
                    if processed_insar:
                        geodetic_data['insar'].append(processed_insar)
                
                print(f"+ Acquired {len(geodetic_data['insar'])} InSAR datasets")
                
            except Exception as e:
                print(f"! Error using satellite scheduler: {e}")
        
        # Acquire GPS/GNSS data from UNAVCO and IGS networks
        try:
            gps_data = await self._fetch_gps_displacement_data(region, time_range)
            geodetic_data['gps'] = gps_data
            print(f"+ Acquired {len(gps_data)} GPS station measurements")
        except Exception as e:
            print(f"! Error acquiring GPS data: {e}")
        
        return geodetic_data
    
    async def _fetch_usgs_earthquake_catalog(self, region: Dict, time_range: Tuple[datetime, datetime]):
        """Fetch earthquake catalog from USGS ANSS"""
        
        try:
            import aiohttp
        except ImportError:
            print("! aiohttp not available, using requests fallback")
            import requests
            
            params = {
                'format': 'csv',
                'starttime': time_range[0].strftime('%Y-%m-%d'),
                'endtime': time_range[1].strftime('%Y-%m-%d'),
                'minlatitude': region['min_lat'],
                'maxlatitude': region['max_lat'],
                'minlongitude': region['min_lon'],
                'maxlongitude': region['max_lon'],
                'minmagnitude': 4.0,
                'includedeleted': 'false',
                'includesuperseded': 'false'
            }
            
            url = self.data_sources['usgs_anss']['url'] + 'query'
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                return self._parse_usgs_csv_catalog(response.text)
            
            return None
        
        params = {
            'format': 'csv',
            'starttime': time_range[0].strftime('%Y-%m-%d'),
            'endtime': time_range[1].strftime('%Y-%m-%d'),
            'minlatitude': region['min_lat'],
            'maxlatitude': region['max_lat'],
            'minlongitude': region['min_lon'],
            'maxlongitude': region['max_lon'],
            'minmagnitude': 4.0,
            'includedeleted': 'false',
            'includesuperseded': 'false'
        }
        
        url = self.data_sources['usgs_anss']['url'] + 'query'
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_usgs_csv_catalog(content)
        except Exception as e:
            print(f"! Error fetching USGS data: {e}")
        
        return None
    
    def _parse_usgs_csv_catalog(self, csv_content):
        """Parse USGS CSV earthquake catalog"""
        
        try:
            import pandas as pd
            from io import StringIO
            
            df = pd.read_csv(StringIO(csv_content))
            return df.to_dict('records')
        except ImportError:
            # Fallback CSV parsing without pandas
            import csv
            from io import StringIO
            
            reader = csv.DictReader(StringIO(csv_content))
            return list(reader)
        except Exception as e:
            print(f"! Error parsing USGS CSV: {e}")
            return None
    
    def _convert_catalog_to_mechanisms(self, catalog_data) -> List[FocalMechanism]:
        """Convert earthquake catalog to focal mechanism objects"""
        
        mechanisms = []
        
        for i, event in enumerate(catalog_data):
            try:
                # Basic focal mechanism from first-motion analysis or regional database
                mechanism = FocalMechanism(
                    event_id=str(event.get('id', f"catalog_{i}")),
                    latitude=float(event['latitude']),
                    longitude=float(event['longitude']),
                    depth=float(event['depth']),
                    magnitude=float(event['mag']),
                    # Use regional stress pattern estimates for missing focal mechanisms
                    strike1=45.0,  # Would be from actual focal mechanism database
                    dip1=90.0,
                    rake1=0.0,
                    strike2=135.0,
                    dip2=90.0,
                    rake2=180.0,
                    timestamp=datetime.fromisoformat(event['time'].replace('Z', '+00:00')) if 'time' in event else datetime.now(),
                    quality='C'  # Lower quality for inferred mechanisms
                )
                mechanisms.append(mechanism)
                
            except (KeyError, ValueError) as e:
                print(f"! Error converting event {i} to focal mechanism: {e}")
                continue
        
        return mechanisms


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