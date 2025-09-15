"""
Advanced Resurgent Caldera Monitoring System for Supereruption Analysis
Integrates 3D stress field mapping, seismic tomography, satellite data, and ML-based reservoir analysis
"""

import numpy as np
import pandas as pd
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import cKDTree, Delaunay
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Import existing Prometheus modules
try:
    from .supereruption_analysis import (
        SupereruptionPrecursorAnalyzer, CalderaSystem, 
        RealTimeMonitoringSystem, SupereruptionEvent
    )
    from .stress_field_mapper import StressFieldMapper
    from .satellite_scheduler import SatelliteScheduler, AcquisitionPlan
    from .data_integration_pipeline import DataIntegrationPipeline
    from .geological_modeler_3d import ImplicitSurfaceModeler
    from .high_resolution_visualization import HighResolutionVisualization
    from .caldera_system_classifier import CalderaSystemClassifier
    PROMETHEUS_MODULES_AVAILABLE = True
except ImportError:
    # Fallback for testing
    class SupereruptionPrecursorAnalyzer: pass
    class CalderaSystem: pass
    class RealTimeMonitoringSystem: pass
    class StressFieldMapper: pass
    class SatelliteScheduler: pass
    class DataIntegrationPipeline: pass
    class ImplicitSurfaceModeler: pass
    class HighResolutionVisualization: pass
    class CalderaSystemClassifier: pass
    PROMETHEUS_MODULES_AVAILABLE = False

warnings.filterwarnings('ignore')

class ResurgenceType(Enum):
    DOME_FORMATION = "dome_formation"
    BLOCK_UPLIFT = "block_uplift" 
    DISTRIBUTED_UPLIFT = "distributed_uplift"
    ASYMMETRIC_UPLIFT = "asymmetric_uplift"
    NO_RESURGENCE = "no_resurgence"

class MagmaReservoirState(Enum):
    SOLIDIFIED = "solidified"
    PARTIALLY_MOLTEN = "partially_molten"
    HIGHLY_MOLTEN = "highly_molten"
    CRITICAL_MELT_FRACTION = "critical_melt_fraction"
    VOLATILE_SATURATED = "volatile_saturated"

class RoofRockStability(Enum):
    STABLE = "stable"
    WEAKENING = "weakening"
    FRACTURED = "fractured"
    CRITICAL = "critical"
    IMMINENT_FAILURE = "imminent_failure"

@dataclass
class MagmaReservoir:
    """3D magma reservoir characteristics"""
    reservoir_id: str
    depth_km: float
    volume_km3: float
    temperature_c: float
    pressure_mpa: float
    melt_fraction: float
    volatile_content: Dict[str, float]  # H2O, CO2, SO2, etc.
    crystal_content: float
    viscosity_pas: float
    density_kg_m3: float
    roof_thickness_km: float
    floor_depth_km: float
    geometry_3d: Dict[str, np.ndarray]  # 3D mesh data
    stress_state: Dict[str, float]
    overpressure_mpa: float
    cap_rock_properties: Dict[str, float]

@dataclass
class SeismicTomography:
    """3D seismic tomography results"""
    velocity_model_3d: np.ndarray  # 3D P-wave velocity
    vs_model_3d: np.ndarray  # 3D S-wave velocity
    attenuation_3d: np.ndarray  # 3D Q factor
    grid_coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]  # X, Y, Z
    resolution_km: float
    uncertainty: np.ndarray
    melt_regions: List[Dict]  # Identified melt pockets
    velocity_anomalies: Dict[str, np.ndarray]
    anisotropy: Dict[str, np.ndarray]

@dataclass
class CalderaDeformation:
    """3D caldera deformation data"""
    uplift_3d: np.ndarray  # 3D displacement field
    horizontal_displacement: np.ndarray
    strain_tensor_3d: np.ndarray
    tilt_field: np.ndarray
    coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]
    temporal_evolution: List[np.ndarray]  # Time series
    deformation_source: Dict[str, float]  # Mogi/spheroid source
    resurgence_rate_mm_yr: float
    asymmetry_index: float

@dataclass
class StressField3D:
    """3D stress field analysis"""
    stress_tensor_3d: np.ndarray  # 6 components at each point
    principal_stresses: np.ndarray  # σ1, σ2, σ3
    stress_orientation: np.ndarray  # Principal stress directions
    shear_stress: np.ndarray
    differential_stress: np.ndarray
    failure_potential: np.ndarray  # Coulomb failure criterion
    coordinates: Tuple[np.ndarray, np.ndarray, np.ndarray]
    temporal_changes: List[np.ndarray]
    critical_zones: List[Dict]  # High failure potential areas

class MagmaReservoirML(nn.Module):
    """Neural network for magma reservoir state prediction"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple prediction heads
        self.melt_fraction_head = nn.Linear(prev_dim, 1)
        self.temperature_head = nn.Linear(prev_dim, 1)
        self.pressure_head = nn.Linear(prev_dim, 1)
        self.volatile_head = nn.Linear(prev_dim, 3)  # H2O, CO2, SO2
        self.stability_head = nn.Linear(prev_dim, len(RoofRockStability))
        self.eruption_potential_head = nn.Linear(prev_dim, 1)
        
        # Attention mechanism for interpretability
        self.attention = nn.MultiheadAttention(prev_dim, num_heads=8)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        # Apply attention
        features_att = features.unsqueeze(1)  # Add sequence dimension
        attended_features, attention_weights = self.attention(
            features_att, features_att, features_att
        )
        attended_features = attended_features.squeeze(1)
        
        return {
            'melt_fraction': torch.sigmoid(self.melt_fraction_head(attended_features)),
            'temperature': self.temperature_head(attended_features),
            'pressure': self.pressure_head(attended_features),
            'volatiles': torch.sigmoid(self.volatile_head(attended_features)),
            'roof_stability': torch.softmax(self.stability_head(attended_features), dim=1),
            'eruption_potential': torch.sigmoid(self.eruption_potential_head(attended_features)),
            'attention_weights': attention_weights,
            'features': attended_features
        }

class ResurgentCalderaMonitoringSystem:
    """Comprehensive resurgent caldera monitoring and analysis system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize subsystems
        self.supereruption_analyzer = SupereruptionPrecursorAnalyzer()
        self.stress_field_mapper = StressFieldMapper()
        self.satellite_scheduler = SatelliteScheduler()
        self.data_integration = DataIntegrationPipeline()
        self.surface_modeler = ImplicitSurfaceModeler()
        self.visualization = HighResolutionVisualization()
        
        # ML models
        self.reservoir_ml_model = None
        self.roof_stability_model = None
        self.deformation_predictor = None
        
        # 3D data storage
        self.magma_reservoirs: Dict[str, MagmaReservoir] = {}
        self.seismic_models: Dict[str, SeismicTomography] = {}
        self.deformation_fields: Dict[str, CalderaDeformation] = {}
        self.stress_fields: Dict[str, StressField3D] = {}
        
        # Monitoring data cache
        self.monitoring_cache = {}
        self.analysis_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'spatial_resolution_km': 0.5,
            'temporal_resolution_days': 1,
            'tomography_resolution_km': 1.0,
            'stress_field_resolution_km': 0.25,
            'monitoring_frequency_hours': 6,
            'alert_thresholds': {
                'melt_fraction': 0.4,
                'roof_pressure_ratio': 0.8,
                'deformation_rate_mm_day': 5.0,
                'seismic_energy_increase': 2.0,
                'volatile_saturation': 0.6
            },
            'ml_training': {
                'batch_size': 64,
                'learning_rate': 0.001,
                'epochs': 200,
                'validation_split': 0.2
            },
            '3d_visualization': {
                'mesh_resolution': 50,
                'transparency_levels': [0.1, 0.3, 0.6, 0.9],
                'color_schemes': {
                    'stress': 'viridis',
                    'temperature': 'hot',
                    'velocity': 'seismic',
                    'deformation': 'RdBu'
                }
            }
        }
    
    async def initialize_caldera_monitoring(self, caldera_id: str, 
                                          location: Tuple[float, float],
                                          bounds: Dict[str, float]) -> str:
        """Initialize comprehensive monitoring for a resurgent caldera"""
        
        self.logger.info(f"Initializing monitoring for caldera {caldera_id}")
        
        # Create monitoring grid
        monitoring_grid = self._create_3d_monitoring_grid(bounds)
        
        # Initialize satellite data acquisition
        satellite_plan = await self._setup_satellite_monitoring(caldera_id, location, bounds)
        
        # Setup seismic monitoring network
        seismic_network = self._design_seismic_network(location, bounds)
        
        # Initialize 3D models
        initial_models = await self._initialize_3d_models(caldera_id, bounds)
        
        monitoring_setup = {
            'caldera_id': caldera_id,
            'monitoring_grid': monitoring_grid,
            'satellite_plan': satellite_plan,
            'seismic_network': seismic_network,
            'initial_models': initial_models,
            'status': 'initialized',
            'timestamp': datetime.now()
        }
        
        self.monitoring_cache[caldera_id] = monitoring_setup
        
        return caldera_id
    
    def _create_3d_monitoring_grid(self, bounds: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Create 3D spatial grid for monitoring"""
        
        res = self.config['spatial_resolution_km']
        
        x = np.arange(bounds['west'], bounds['east'], res)
        y = np.arange(bounds['south'], bounds['north'], res)
        z = np.arange(-bounds.get('max_depth', 20), bounds.get('elevation', 5), res)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        return {
            'coordinates': (X, Y, Z),
            'x_range': x,
            'y_range': y, 
            'z_range': z,
            'resolution_km': res,
            'total_points': len(x) * len(y) * len(z)
        }
    
    async def _setup_satellite_monitoring(self, caldera_id: str, 
                                        location: Tuple[float, float],
                                        bounds: Dict[str, float]) -> Dict:
        """Setup automated satellite data acquisition"""
        
        from .satellite_scheduler import GeographicBounds, AcquisitionPlan, SatelliteMission
        
        # Create geographic bounds
        geo_bounds = GeographicBounds(
            north=bounds['north'],
            south=bounds['south'], 
            east=bounds['east'],
            west=bounds['west']
        )
        
        # Create acquisition plan
        plan = AcquisitionPlan(
            plan_id=f"resurgent_monitoring_{caldera_id}",
            name=f"Resurgent Caldera Monitoring - {caldera_id}",
            description="High-frequency InSAR monitoring for caldera resurgence",
            target_area=geo_bounds,
            priority="HIGH",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=365*2),
            missions=[SatelliteMission.SENTINEL1, SatelliteMission.ALOS2],
            temporal_baseline_max=12,  # days
            min_overlap=0.9,
            auto_download=True
        )
        
        plan_id = await self.satellite_scheduler.create_acquisition_plan(plan)
        
        return {
            'plan_id': plan_id,
            'coverage_area': geo_bounds,
            'missions': ['Sentinel-1', 'ALOS-2'],
            'repeat_cycle_days': 6,
            'expected_products_per_year': 120
        }
    
    def _design_seismic_network(self, location: Tuple[float, float], 
                               bounds: Dict[str, float]) -> Dict:
        """Design optimal seismic monitoring network"""
        
        # Create high-density network for resurgent calderas
        network_stations = []
        
        # Central cluster around caldera
        center_lon, center_lat = location
        radius_km = min((bounds['east'] - bounds['west']) * 111, 
                       (bounds['north'] - bounds['south']) * 111) / 2
        
        # Concentric rings of stations
        for ring in range(3):
            ring_radius = (ring + 1) * radius_km / 3
            n_stations = max(6, 6 * (ring + 1))
            
            for i in range(n_stations):
                angle = 2 * np.pi * i / n_stations
                station_lon = center_lon + ring_radius * np.cos(angle) / 111
                station_lat = center_lat + ring_radius * np.sin(angle) / 111
                
                network_stations.append({
                    'station_id': f"RCMS_{ring:02d}_{i:02d}",
                    'longitude': station_lon,
                    'latitude': station_lat,
                    'ring': ring,
                    'azimuth': np.degrees(angle),
                    'distance_km': ring_radius,
                    'instruments': ['broadband_seismometer', 'accelerometer', 'tiltmeter']
                })
        
        return {
            'stations': network_stations,
            'total_stations': len(network_stations),
            'network_geometry': 'concentric_rings',
            'detection_threshold_magnitude': 0.5,
            'location_accuracy_km': 0.1
        }
    
    async def _initialize_3d_models(self, caldera_id: str, bounds: Dict[str, float]) -> Dict:
        """Initialize 3D geological models"""
        
        # Create initial velocity model from global data
        velocity_model = self._create_initial_velocity_model(bounds)
        
        # Initialize stress field
        stress_field = self._initialize_stress_field(bounds)
        
        # Create magma reservoir template
        reservoir_model = self._initialize_reservoir_model(bounds)
        
        return {
            'velocity_model': velocity_model,
            'stress_field': stress_field,
            'reservoir_model': reservoir_model,
            'creation_time': datetime.now()
        }
    
    def _create_initial_velocity_model(self, bounds: Dict[str, float]) -> Dict:
        """Create initial 3D velocity model"""
        
        # Create grid
        grid = self._create_3d_monitoring_grid(bounds)
        X, Y, Z = grid['coordinates']
        
        # Simple layered model with low-velocity zone for magma chamber
        vp = np.ones_like(X) * 6.0  # km/s
        vs = vp / 1.73  # Poisson solid
        
        # Add low-velocity zone at depth (potential magma chamber)
        chamber_depth = -8.0  # km
        chamber_radius = 5.0  # km
        
        center_x = (bounds['east'] + bounds['west']) / 2
        center_y = (bounds['north'] + bounds['south']) / 2
        
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2 + (Z - chamber_depth)**2)
        magma_mask = distance < chamber_radius
        
        vp[magma_mask] = 3.5  # Reduced velocity in magma chamber
        vs[magma_mask] = 2.0
        
        return {
            'vp_3d': vp,
            'vs_3d': vs,
            'coordinates': grid['coordinates'],
            'resolution_km': grid['resolution_km']
        }
    
    def _initialize_stress_field(self, bounds: Dict[str, float]) -> Dict:
        """Initialize 3D stress field"""
        
        grid = self._create_3d_monitoring_grid(bounds)
        X, Y, Z = grid['coordinates']
        
        # Lithostatic stress
        density = 2700  # kg/m³
        g = 9.81  # m/s²
        
        sigma_v = density * g * np.abs(Z * 1000)  # Vertical stress (Pa)
        sigma_h = 0.7 * sigma_v  # Horizontal stress (typical ratio)
        
        # Create stress tensor (6 components: σxx, σyy, σzz, σxy, σxz, σyz)
        stress_tensor = np.zeros(X.shape + (6,))
        stress_tensor[..., 0] = sigma_h  # σxx
        stress_tensor[..., 1] = sigma_h  # σyy  
        stress_tensor[..., 2] = sigma_v  # σzz
        # Shear stresses initially zero
        
        return {
            'stress_tensor_3d': stress_tensor,
            'coordinates': grid['coordinates'],
            'units': 'Pa'
        }
    
    def _initialize_reservoir_model(self, bounds: Dict[str, float]) -> Dict:
        """Initialize magma reservoir model"""
        
        # Typical resurgent caldera parameters
        center_lon = (bounds['east'] + bounds['west']) / 2
        center_lat = (bounds['north'] + bounds['south']) / 2
        
        reservoir = MagmaReservoir(
            reservoir_id="main_chamber",
            depth_km=8.0,
            volume_km3=1000.0,
            temperature_c=750.0,
            pressure_mpa=200.0,
            melt_fraction=0.3,
            volatile_content={'H2O': 0.04, 'CO2': 0.002, 'SO2': 0.0001},
            crystal_content=0.7,
            viscosity_pas=1e8,
            density_kg_m3=2400.0,
            roof_thickness_km=2.0,
            floor_depth_km=15.0,
            geometry_3d={'center': [center_lon, center_lat, -8.0], 'radii': [5.0, 3.0, 2.0]},
            stress_state={'overpressure': 50e6},
            overpressure_mpa=50.0,
            cap_rock_properties={'strength_mpa': 100, 'permeability': 1e-15}
        )
        
        return {
            'main_reservoir': reservoir,
            'secondary_chambers': [],
            'conduit_system': {}
        }
    
    async def process_seismic_tomography(self, caldera_id: str, 
                                       seismic_data: Dict) -> SeismicTomography:
        """Process seismic data to create 3D tomography model"""
        
        self.logger.info(f"Processing seismic tomography for {caldera_id}")
        
        # Extract travel times and arrival data
        travel_times = seismic_data.get('travel_times', [])
        stations = seismic_data.get('stations', [])
        events = seismic_data.get('events', [])
        
        if not all([travel_times, stations, events]):
            raise ValueError("Insufficient seismic data for tomography")
        
        # Create inversion grid
        bounds = seismic_data.get('bounds', {})
        grid = self._create_3d_monitoring_grid(bounds)
        X, Y, Z = grid['coordinates']
        
        # Perform tomographic inversion (simplified)
        vp_model, vs_model = await self._invert_velocity_model(
            travel_times, stations, events, grid
        )
        
        # Calculate attenuation
        attenuation = self._calculate_attenuation(vp_model, vs_model)
        
        # Identify melt regions
        melt_regions = self._identify_melt_regions(vp_model, vs_model)
        
        # Calculate uncertainties
        uncertainty = self._calculate_tomography_uncertainty(
            travel_times, stations, events, grid
        )
        
        tomography = SeismicTomography(
            velocity_model_3d=vp_model,
            vs_model_3d=vs_model,
            attenuation_3d=attenuation,
            grid_coordinates=(X, Y, Z),
            resolution_km=grid['resolution_km'],
            uncertainty=uncertainty,
            melt_regions=melt_regions,
            velocity_anomalies=self._calculate_velocity_anomalies(vp_model),
            anisotropy=self._estimate_anisotropy(seismic_data)
        )
        
        # Store results
        self.seismic_models[caldera_id] = tomography
        
        return tomography
    
    async def _invert_velocity_model(self, travel_times: List, stations: List, 
                                   events: List, grid: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Perform 3D seismic tomography inversion"""
        
        # This is a simplified implementation
        # In practice, would use advanced tomography codes like FMTOMO, SIMUL2000, etc.
        
        X, Y, Z = grid['coordinates']
        
        # Initialize with 1D reference model
        vp_initial = 6.0 + 0.3 * np.abs(Z)  # Increasing velocity with depth
        vs_initial = vp_initial / 1.73
        
        # Apply smoothness constraints and data fitting
        # (This would be a complex iterative inversion in practice)
        
        # Add realistic velocity perturbations
        # Low velocity zones for potential melt
        for event in events:
            event_x, event_y, event_z = event['location']
            
            # Create anomaly around event
            distance = np.sqrt((X - event_x)**2 + (Y - event_y)**2 + (Z - event_z)**2)
            anomaly_mask = distance < 2.0  # 2 km radius
            
            # Reduce velocity in potential melt zones
            if event_z < -5:  # Deep events
                vp_initial[anomaly_mask] *= 0.8
                vs_initial[anomaly_mask] *= 0.7
        
        return vp_initial, vs_initial
    
    def _calculate_attenuation(self, vp: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """Calculate seismic attenuation (Q factor)"""
        
        # Empirical relationship: low velocity = high attenuation
        vp_mean = np.mean(vp)
        q_factor = 50 + 200 * (vp / vp_mean)  # Q ranges from 50-250
        
        return q_factor
    
    def _identify_melt_regions(self, vp: np.ndarray, vs: np.ndarray) -> List[Dict]:
        """Identify potential melt regions from velocity models"""
        
        # Calculate Vp/Vs ratio
        vp_vs_ratio = vp / (vs + 1e-8)
        
        # High Vp/Vs ratio indicates partial melt
        melt_threshold = 1.9
        melt_mask = vp_vs_ratio > melt_threshold
        
        # Also check for low absolute velocities
        low_vp_mask = vp < 4.0  # km/s
        low_vs_mask = vs < 2.3  # km/s
        
        combined_mask = melt_mask & (low_vp_mask | low_vs_mask)
        
        # Find connected regions
        from scipy.ndimage import label
        labeled_regions, num_regions = label(combined_mask)
        
        melt_regions = []
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            region_indices = np.where(region_mask)
            
            if len(region_indices[0]) > 10:  # Minimum size threshold
                center_idx = (
                    int(np.mean(region_indices[0])),
                    int(np.mean(region_indices[1])), 
                    int(np.mean(region_indices[2]))
                )
                
                melt_regions.append({
                    'region_id': f"melt_{i}",
                    'center_indices': center_idx,
                    'volume_points': len(region_indices[0]),
                    'mean_vp': float(np.mean(vp[region_mask])),
                    'mean_vs': float(np.mean(vs[region_mask])),
                    'mean_vp_vs_ratio': float(np.mean(vp_vs_ratio[region_mask]))
                })
        
        return melt_regions
    
    def _calculate_tomography_uncertainty(self, travel_times: List, stations: List,
                                        events: List, grid: Dict) -> np.ndarray:
        """Calculate uncertainty in tomography results"""
        
        X, Y, Z = grid['coordinates']
        
        # Resolution based on ray coverage
        # Areas with poor ray coverage have higher uncertainty
        
        # Initialize with high uncertainty
        uncertainty = np.ones_like(X) * 0.5  # km/s
        
        # Reduce uncertainty near stations and events
        for station in stations:
            station_x, station_y = station['longitude'], station['latitude']
            distance = np.sqrt((X - station_x)**2 + (Y - station_y)**2)
            
            # Uncertainty decreases with proximity to stations
            station_uncertainty = 0.1 + 0.4 * (distance / np.max(distance))
            uncertainty = np.minimum(uncertainty, station_uncertainty)
        
        return uncertainty
    
    def _calculate_velocity_anomalies(self, vp: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate velocity anomalies relative to reference model"""
        
        # 1D reference model
        depth_profile = np.mean(np.mean(vp, axis=0), axis=0)  # Average over x,y
        z_coords = np.arange(len(depth_profile))
        
        # Create 3D reference
        vp_reference = np.repeat(depth_profile[np.newaxis, np.newaxis, :], 
                                vp.shape[0], axis=0)
        vp_reference = np.repeat(vp_reference[:, np.newaxis, :], 
                                vp.shape[1], axis=1)
        
        # Calculate anomalies
        absolute_anomaly = vp - vp_reference
        relative_anomaly = (vp - vp_reference) / (vp_reference + 1e-8) * 100
        
        return {
            'absolute': absolute_anomaly,
            'relative_percent': relative_anomaly
        }
    
    def _estimate_anisotropy(self, seismic_data: Dict) -> Dict[str, np.ndarray]:
        """Estimate seismic anisotropy"""
        
        # Simplified anisotropy estimation
        # Would need splitting analysis for real implementation
        
        return {
            'fast_direction': np.zeros((10, 10, 10)),  # Placeholder
            'delay_time': np.zeros((10, 10, 10))
        }
    
    async def analyze_3d_stress_field(self, caldera_id: str, 
                                    deformation_data: Dict,
                                    seismic_data: Dict) -> StressField3D:
        """Analyze 3D stress field from deformation and seismic data"""
        
        self.logger.info(f"Analyzing 3D stress field for {caldera_id}")
        
        # Get monitoring grid
        if caldera_id not in self.monitoring_cache:
            raise ValueError(f"Caldera {caldera_id} not initialized")
        
        grid = self.monitoring_cache[caldera_id]['monitoring_grid']
        X, Y, Z = grid['coordinates']
        
        # Process deformation data
        deformation_field = await self._process_deformation_data(deformation_data, grid)
        
        # Invert for stress field
        stress_tensor_3d = await self._invert_stress_field(deformation_field, seismic_data, grid)
        
        # Calculate principal stresses
        principal_stresses, orientations = self._calculate_principal_stresses(stress_tensor_3d)
        
        # Calculate derived stress parameters
        shear_stress = self._calculate_shear_stress(stress_tensor_3d)
        differential_stress = principal_stresses[..., 0] - principal_stresses[..., 2]
        
        # Assess failure potential
        failure_potential = self._assess_failure_potential(stress_tensor_3d)
        
        # Identify critical zones
        critical_zones = self._identify_critical_stress_zones(failure_potential, X, Y, Z)
        
        stress_field = StressField3D(
            stress_tensor_3d=stress_tensor_3d,
            principal_stresses=principal_stresses,
            stress_orientation=orientations,
            shear_stress=shear_stress,
            differential_stress=differential_stress,
            failure_potential=failure_potential,
            coordinates=(X, Y, Z),
            temporal_changes=[],  # Initialize empty, will be populated over time
            critical_zones=critical_zones
        )
        
        # Store results
        self.stress_fields[caldera_id] = stress_field
        
        return stress_field
    
    async def _process_deformation_data(self, deformation_data: Dict, 
                                      grid: Dict) -> CalderaDeformation:
        """Process InSAR and GPS deformation data"""
        
        X, Y, Z = grid['coordinates']
        
        # Extract displacement data
        los_displacement = deformation_data.get('los_displacement', {})
        gps_displacement = deformation_data.get('gps_displacement', {})
        
        # Interpolate displacements to 3D grid
        if los_displacement:
            # InSAR line-of-sight displacement
            los_coords = np.array([[p['lon'], p['lat']] for p in los_displacement['points']])
            los_values = np.array([p['displacement_mm'] for p in los_displacement['points']])
            
            # Interpolate to surface grid
            from scipy.interpolate import griddata
            surface_displacement = griddata(
                los_coords, los_values, 
                (X[:, :, -1], Y[:, :, -1]), 
                method='cubic', fill_value=0
            )
            
            # Create 3D uplift field (assuming vertical motion)
            uplift_3d = np.zeros_like(X)
            uplift_3d[:, :, -1] = surface_displacement  # Surface
            
            # Decay with depth (simple model)
            for k in range(X.shape[2] - 2, -1, -1):
                depth_factor = np.exp(-0.2 * np.abs(Z[0, 0, k]))
                uplift_3d[:, :, k] = surface_displacement * depth_factor
        else:
            uplift_3d = np.zeros_like(X)
        
        # Calculate strain tensor from deformation gradient
        strain_tensor_3d = self._calculate_strain_tensor(uplift_3d, grid['resolution_km'])
        
        # Estimate deformation source parameters
        deformation_source = self._estimate_deformation_source(uplift_3d, X, Y, Z)
        
        # Calculate resurgence characteristics
        surface_uplift = uplift_3d[:, :, -1]
        resurgence_rate = np.max(surface_uplift) * 365.25  # mm/year
        asymmetry_index = self._calculate_asymmetry_index(surface_uplift)
        
        deformation = CalderaDeformation(
            uplift_3d=uplift_3d,
            horizontal_displacement=np.zeros_like(uplift_3d),  # Simplified
            strain_tensor_3d=strain_tensor_3d,
            tilt_field=self._calculate_tilt_field(surface_uplift),
            coordinates=(X, Y, Z),
            temporal_evolution=[],  # Initialize empty
            deformation_source=deformation_source,
            resurgence_rate_mm_yr=resurgence_rate,
            asymmetry_index=asymmetry_index
        )
        
        return deformation
    
    def _calculate_strain_tensor(self, displacement: np.ndarray, 
                                resolution_km: float) -> np.ndarray:
        """Calculate 3D strain tensor from displacement field"""
        
        # Calculate spatial derivatives
        du_dx = np.gradient(displacement, resolution_km * 1000, axis=0)
        du_dy = np.gradient(displacement, resolution_km * 1000, axis=1)
        du_dz = np.gradient(displacement, resolution_km * 1000, axis=2)
        
        # For now, assume only vertical displacement
        # In full implementation, would have all 3 displacement components
        
        strain_tensor = np.zeros(displacement.shape + (6,))
        
        # Strain components (εxx, εyy, εzz, γxy, γxz, γyz)
        strain_tensor[..., 2] = du_dz  # εzz (vertical strain)
        
        # Add some horizontal strain from vertical deformation
        strain_tensor[..., 0] = -0.3 * du_dz  # εxx (Poisson effect)
        strain_tensor[..., 1] = -0.3 * du_dz  # εyy (Poisson effect)
        
        return strain_tensor
    
    def _estimate_deformation_source(self, uplift_3d: np.ndarray,
                                   X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict:
        """Estimate deformation source parameters using Mogi model"""
        
        # Find center of maximum uplift
        surface_uplift = uplift_3d[:, :, -1]
        max_idx = np.unravel_index(np.argmax(surface_uplift), surface_uplift.shape)
        
        center_x = X[max_idx[0], max_idx[1], -1]
        center_y = Y[max_idx[0], max_idx[1], -1]
        
        # Estimate source depth from uplift pattern width
        max_uplift = np.max(surface_uplift)
        half_max_mask = surface_uplift > max_uplift / 2
        
        # Calculate effective radius
        n_points = np.sum(half_max_mask)
        effective_radius = np.sqrt(n_points / np.pi) * (X[1, 0, 0] - X[0, 0, 0])
        
        # Depth estimate (empirical relationship)
        estimated_depth = effective_radius * 0.8
        
        # Volume change estimate (Mogi model)
        volume_change = max_uplift * effective_radius**2 * np.pi / 1000  # m³
        
        return {
            'source_type': 'mogi_sphere',
            'center_x': float(center_x),
            'center_y': float(center_y),
            'depth_km': float(estimated_depth),
            'volume_change_m3': float(volume_change),
            'effective_radius_km': float(effective_radius)
        }
    
    def _calculate_asymmetry_index(self, surface_uplift: np.ndarray) -> float:
        """Calculate asymmetry index for resurgent dome"""
        
        # Find center of uplift
        max_idx = np.unravel_index(np.argmax(surface_uplift), surface_uplift.shape)
        center_i, center_j = max_idx
        
        # Calculate radial profiles in different directions
        profiles = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = int(20 * np.cos(angle))  # Sample 20 points radially
            dy = int(20 * np.sin(angle))
            
            profile_vals = []
            for r in range(1, 20):
                i = center_i + r * dx // 20
                j = center_j + r * dy // 20
                
                if 0 <= i < surface_uplift.shape[0] and 0 <= j < surface_uplift.shape[1]:
                    profile_vals.append(surface_uplift[i, j])
            
            if profile_vals:
                profiles.append(np.array(profile_vals))
        
        # Calculate asymmetry as variance of profile shapes
        if len(profiles) > 1:
            # Normalize profiles
            normalized_profiles = []
            for profile in profiles:
                if len(profile) > 5:
                    normalized = profile / (np.max(profile) + 1e-8)
                    normalized_profiles.append(normalized[:10])  # First 10 points
            
            if len(normalized_profiles) > 1:
                # Stack and calculate variance
                min_len = min(len(p) for p in normalized_profiles)
                profile_matrix = np.array([p[:min_len] for p in normalized_profiles])
                asymmetry = np.mean(np.var(profile_matrix, axis=0))
                return float(asymmetry)
        
        return 0.0
    
    def _calculate_tilt_field(self, surface_displacement: np.ndarray) -> np.ndarray:
        """Calculate tilt field from surface displacement"""
        
        # Calculate gradients
        grad_x = np.gradient(surface_displacement, axis=1)
        grad_y = np.gradient(surface_displacement, axis=0)
        
        # Tilt magnitude
        tilt_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return tilt_magnitude
    
    async def _invert_stress_field(self, deformation: CalderaDeformation,
                                 seismic_data: Dict, grid: Dict) -> np.ndarray:
        """Invert for 3D stress field from deformation and seismic data"""
        
        X, Y, Z = grid['coordinates']
        
        # Initialize stress tensor
        stress_tensor = np.zeros(X.shape + (6,))
        
        # Use deformation to constrain stress field
        strain_tensor = deformation.strain_tensor_3d
        
        # Elastic relationship: stress = C : strain
        # Using isotropic elasticity for simplicity
        young_modulus = 50e9  # Pa
        poisson_ratio = 0.25
        
        # Elastic constants
        lambda_lame = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        mu = young_modulus / (2 * (1 + poisson_ratio))
        
        # Convert strain to stress
        trace_strain = strain_tensor[..., 0] + strain_tensor[..., 1] + strain_tensor[..., 2]
        
        stress_tensor[..., 0] = lambda_lame * trace_strain + 2 * mu * strain_tensor[..., 0]  # σxx
        stress_tensor[..., 1] = lambda_lame * trace_strain + 2 * mu * strain_tensor[..., 1]  # σyy
        stress_tensor[..., 2] = lambda_lame * trace_strain + 2 * mu * strain_tensor[..., 2]  # σzz
        stress_tensor[..., 3] = mu * strain_tensor[..., 3]  # τxy
        stress_tensor[..., 4] = mu * strain_tensor[..., 4]  # τxz
        stress_tensor[..., 5] = mu * strain_tensor[..., 5]  # τyz
        
        # Add lithostatic stress
        density = 2700  # kg/m³
        g = 9.81
        
        lithostatic = density * g * np.abs(Z * 1000)
        stress_tensor[..., 2] += lithostatic  # Add to σzz
        
        # Add horizontal stress
        k0 = 0.7  # At-rest earth pressure coefficient
        stress_tensor[..., 0] += k0 * lithostatic  # σxx
        stress_tensor[..., 1] += k0 * lithostatic  # σyy
        
        return stress_tensor
    
    def _calculate_principal_stresses(self, stress_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate principal stresses and orientations"""
        
        # Extract stress tensor components
        sxx = stress_tensor[..., 0]
        syy = stress_tensor[..., 1]
        szz = stress_tensor[..., 2]
        sxy = stress_tensor[..., 3]
        sxz = stress_tensor[..., 4]
        syz = stress_tensor[..., 5]
        
        principal_stresses = np.zeros(stress_tensor.shape[:3] + (3,))
        orientations = np.zeros(stress_tensor.shape[:3] + (3, 3))
        
        # Calculate principal stresses for each point
        for i in range(stress_tensor.shape[0]):
            for j in range(stress_tensor.shape[1]):
                for k in range(stress_tensor.shape[2]):
                    # Form stress matrix
                    stress_matrix = np.array([
                        [sxx[i,j,k], sxy[i,j,k], sxz[i,j,k]],
                        [sxy[i,j,k], syy[i,j,k], syz[i,j,k]],
                        [sxz[i,j,k], syz[i,j,k], szz[i,j,k]]
                    ])
                    
                    # Eigenvalue decomposition
                    eigenvals, eigenvecs = np.linalg.eigh(stress_matrix)
                    
                    # Sort by magnitude (σ1 > σ2 > σ3)
                    sorted_idx = np.argsort(eigenvals)[::-1]
                    
                    principal_stresses[i,j,k] = eigenvals[sorted_idx]
                    orientations[i,j,k] = eigenvecs[:, sorted_idx].T
        
        return principal_stresses, orientations
    
    def _calculate_shear_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Calculate magnitude of shear stress"""
        
        # Shear stress components
        sxy = stress_tensor[..., 3]
        sxz = stress_tensor[..., 4]
        syz = stress_tensor[..., 5]
        
        # Magnitude
        shear_magnitude = np.sqrt(sxy**2 + sxz**2 + syz**2)
        
        return shear_magnitude
    
    def _assess_failure_potential(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Assess failure potential using Coulomb failure criterion"""
        
        # Calculate principal stresses
        principal_stresses, _ = self._calculate_principal_stresses(stress_tensor)
        
        sigma1 = principal_stresses[..., 0]  # Maximum principal stress
        sigma3 = principal_stresses[..., 2]  # Minimum principal stress
        
        # Coulomb failure criterion
        # τ = c + σn * tan(φ)
        # For rock: c = cohesion, φ = friction angle
        
        cohesion = 10e6  # Pa
        friction_angle = np.radians(35)  # degrees
        
        # Mohr-Coulomb failure stress
        failure_stress = 2 * cohesion * np.cos(friction_angle) / (1 - np.sin(friction_angle))
        failure_stress += sigma3 * (1 + np.sin(friction_angle)) / (1 - np.sin(friction_angle))
        
        # Failure potential (0 = stable, 1 = at failure)
        failure_potential = (sigma1 - sigma3) / failure_stress
        failure_potential = np.clip(failure_potential, 0, 2)  # Cap at 2 for visualization
        
        return failure_potential
    
    def _identify_critical_stress_zones(self, failure_potential: np.ndarray,
                                      X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> List[Dict]:
        """Identify zones with high failure potential"""
        
        # Threshold for critical stress
        critical_threshold = 0.8
        critical_mask = failure_potential > critical_threshold
        
        # Find connected regions
        from scipy.ndimage import label
        labeled_regions, num_regions = label(critical_mask)
        
        critical_zones = []
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            region_indices = np.where(region_mask)
            
            if len(region_indices[0]) > 5:  # Minimum size
                center_idx = (
                    int(np.mean(region_indices[0])),
                    int(np.mean(region_indices[1])),
                    int(np.mean(region_indices[2]))
                )
                
                center_coords = (
                    X[center_idx], Y[center_idx], Z[center_idx]
                )
                
                mean_failure_potential = np.mean(failure_potential[region_mask])
                max_failure_potential = np.max(failure_potential[region_mask])
                
                critical_zones.append({
                    'zone_id': f"critical_{i}",
                    'center_coordinates': center_coords,
                    'center_indices': center_idx,
                    'volume_points': len(region_indices[0]),
                    'mean_failure_potential': float(mean_failure_potential),
                    'max_failure_potential': float(max_failure_potential),
                    'risk_level': 'high' if max_failure_potential > 1.2 else 'moderate'
                })
        
        return critical_zones
    
    async def train_reservoir_ml_model(self, training_data: List[Dict]) -> Dict:
        """Train machine learning model for magma reservoir analysis"""
        
        self.logger.info("Training magma reservoir ML model")
        
        # Prepare training data
        X, y = self._prepare_ml_training_data(training_data)
        
        # Initialize model
        input_dim = X.shape[1]
        self.reservoir_ml_model = MagmaReservoirML(input_dim)
        
        # Training setup
        optimizer = torch.optim.Adam(self.reservoir_ml_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = {k: torch.FloatTensor(v) for k, v in y.items()}
        
        # Training loop
        epochs = self.config['ml_training']['epochs']
        losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.reservoir_ml_model(X_tensor)
            
            # Multi-task loss
            total_loss = 0
            
            # Melt fraction loss
            mf_loss = F.mse_loss(outputs['melt_fraction'].squeeze(), y_tensor['melt_fraction'])
            total_loss += mf_loss
            
            # Temperature loss
            temp_loss = F.mse_loss(outputs['temperature'].squeeze(), y_tensor['temperature'])
            total_loss += temp_loss * 0.01  # Scale to same order of magnitude
            
            # Pressure loss
            press_loss = F.mse_loss(outputs['pressure'].squeeze(), y_tensor['pressure'])
            total_loss += press_loss * 0.001
            
            # Eruption potential loss
            erupt_loss = F.binary_cross_entropy(outputs['eruption_potential'].squeeze(), 
                                              y_tensor['eruption_potential'])
            total_loss += erupt_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(total_loss.item())
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
        
        return {
            'status': 'trained',
            'epochs': epochs,
            'final_loss': losses[-1],
            'training_losses': losses
        }
    
    def _prepare_ml_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for ML model training"""
        
        features = []
        targets = {
            'melt_fraction': [],
            'temperature': [],
            'pressure': [],
            'eruption_potential': []
        }
        
        for sample in training_data:
            # Extract features
            feature_vector = []
            
            # Seismic features
            feature_vector.extend([
                sample.get('vp_anomaly', 0),
                sample.get('vs_anomaly', 0),
                sample.get('vp_vs_ratio', 1.73),
                sample.get('attenuation', 50)
            ])
            
            # Deformation features
            feature_vector.extend([
                sample.get('uplift_rate_mm_yr', 0),
                sample.get('horizontal_strain', 0),
                sample.get('volume_change', 0)
            ])
            
            # Gravity features
            feature_vector.extend([
                sample.get('gravity_anomaly', 0),
                sample.get('gravity_gradient', 0)
            ])
            
            # Gas features
            feature_vector.extend([
                sample.get('co2_flux', 0),
                sample.get('so2_flux', 0),
                sample.get('he3_he4_ratio', 0)
            ])
            
            # Thermal features
            feature_vector.extend([
                sample.get('thermal_anomaly', 0),
                sample.get('heat_flux', 0)
            ])
            
            features.append(feature_vector)
            
            # Extract targets
            targets['melt_fraction'].append(sample.get('melt_fraction', 0.3))
            targets['temperature'].append(sample.get('temperature_c', 750))
            targets['pressure'].append(sample.get('pressure_mpa', 200))
            targets['eruption_potential'].append(sample.get('eruption_potential', 0.1))
        
        X = np.array(features)
        y = {k: np.array(v) for k, v in targets.items()}
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y
    
    async def predict_reservoir_state(self, caldera_id: str, 
                                    current_data: Dict) -> Dict:
        """Predict current magma reservoir state using ML model"""
        
        if self.reservoir_ml_model is None:
            raise ValueError("ML model not trained")
        
        # Prepare input features
        feature_vector = []
        
        # Extract features from current data
        feature_vector.extend([
            current_data.get('vp_anomaly', 0),
            current_data.get('vs_anomaly', 0), 
            current_data.get('vp_vs_ratio', 1.73),
            current_data.get('attenuation', 50),
            current_data.get('uplift_rate_mm_yr', 0),
            current_data.get('horizontal_strain', 0),
            current_data.get('volume_change', 0),
            current_data.get('gravity_anomaly', 0),
            current_data.get('gravity_gradient', 0),
            current_data.get('co2_flux', 0),
            current_data.get('so2_flux', 0),
            current_data.get('he3_he4_ratio', 0),
            current_data.get('thermal_anomaly', 0),
            current_data.get('heat_flux', 0)
        ])
        
        # Convert to tensor
        X_tensor = torch.FloatTensor([feature_vector])
        
        # Predict
        with torch.no_grad():
            outputs = self.reservoir_ml_model(X_tensor)
            
            predictions = {
                'melt_fraction': float(outputs['melt_fraction'][0]),
                'temperature_c': float(outputs['temperature'][0]),
                'pressure_mpa': float(outputs['pressure'][0]),
                'volatile_content': {
                    'H2O': float(outputs['volatiles'][0, 0]),
                    'CO2': float(outputs['volatiles'][0, 1]),
                    'SO2': float(outputs['volatiles'][0, 2])
                },
                'roof_stability': {
                    'stable': float(outputs['roof_stability'][0, 0]),
                    'weakening': float(outputs['roof_stability'][0, 1]), 
                    'fractured': float(outputs['roof_stability'][0, 2]),
                    'critical': float(outputs['roof_stability'][0, 3]),
                    'imminent_failure': float(outputs['roof_stability'][0, 4])
                },
                'eruption_potential': float(outputs['eruption_potential'][0]),
                'confidence': float(torch.mean(outputs['attention_weights']).item())
            }
        
        # Interpret results
        predictions['state_interpretation'] = self._interpret_reservoir_state(predictions)
        predictions['risk_level'] = self._assess_eruption_risk_ml(predictions)
        predictions['recommendations'] = self._generate_ml_recommendations(predictions)
        
        return predictions
    
    def _interpret_reservoir_state(self, predictions: Dict) -> Dict:
        """Interpret ML model predictions"""
        
        melt_fraction = predictions['melt_fraction']
        temperature = predictions['temperature_c']
        pressure = predictions['pressure_mpa']
        eruption_potential = predictions['eruption_potential']
        
        # Determine reservoir state
        if melt_fraction < 0.2:
            reservoir_state = MagmaReservoirState.SOLIDIFIED
        elif melt_fraction < 0.4:
            reservoir_state = MagmaReservoirState.PARTIALLY_MOLTEN
        elif melt_fraction < 0.6:
            reservoir_state = MagmaReservoirState.HIGHLY_MOLTEN
        elif melt_fraction < 0.8:
            reservoir_state = MagmaReservoirState.CRITICAL_MELT_FRACTION
        else:
            reservoir_state = MagmaReservoirState.VOLATILE_SATURATED
        
        # Determine resurgence type
        h2o_content = predictions['volatile_content']['H2O']
        if eruption_potential > 0.7 and melt_fraction > 0.5:
            resurgence_type = ResurgenceType.DOME_FORMATION
        elif pressure > 300 and melt_fraction > 0.3:
            resurgence_type = ResurgenceType.BLOCK_UPLIFT
        elif melt_fraction > 0.2:
            resurgence_type = ResurgenceType.DISTRIBUTED_UPLIFT
        else:
            resurgence_type = ResurgenceType.NO_RESURGENCE
        
        return {
            'reservoir_state': reservoir_state.value,
            'resurgence_type': resurgence_type.value,
            'thermal_maturity': 'hot' if temperature > 800 else 'cooling',
            'pressure_state': 'overpressured' if pressure > 250 else 'normal',
            'volatile_saturation': 'saturated' if h2o_content > 0.05 else 'undersaturated'
        }
    
    def _assess_eruption_risk_ml(self, predictions: Dict) -> str:
        """Assess eruption risk from ML predictions"""
        
        eruption_potential = predictions['eruption_potential']
        melt_fraction = predictions['melt_fraction']
        roof_stability = predictions['roof_stability']
        
        # Risk factors
        risk_score = 0
        
        if eruption_potential > 0.8:
            risk_score += 3
        elif eruption_potential > 0.6:
            risk_score += 2
        elif eruption_potential > 0.4:
            risk_score += 1
        
        if melt_fraction > 0.6:
            risk_score += 2
        elif melt_fraction > 0.4:
            risk_score += 1
        
        if roof_stability['critical'] > 0.5 or roof_stability['imminent_failure'] > 0.3:
            risk_score += 3
        elif roof_stability['fractured'] > 0.5:
            risk_score += 2
        elif roof_stability['weakening'] > 0.5:
            risk_score += 1
        
        # Classify risk
        if risk_score >= 7:
            return 'extreme'
        elif risk_score >= 5:
            return 'high'
        elif risk_score >= 3:
            return 'moderate'
        else:
            return 'low'
    
    def _generate_ml_recommendations(self, predictions: Dict) -> List[str]:
        """Generate recommendations based on ML predictions"""
        
        recommendations = []
        
        eruption_potential = predictions['eruption_potential']
        melt_fraction = predictions['melt_fraction']
        roof_stability = predictions['roof_stability']
        risk_level = predictions['risk_level']
        
        # Risk-based recommendations
        if risk_level == 'extreme':
            recommendations.extend([
                "URGENT: Immediate evacuation planning required",
                "Deploy real-time monitoring systems",
                "Activate emergency response protocols",
                "Coordinate with civil protection authorities"
            ])
        elif risk_level == 'high':
            recommendations.extend([
                "Increase monitoring frequency to continuous",
                "Deploy additional seismic stations",
                "Install real-time GPS stations",
                "Conduct detailed hazard assessment"
            ])
        elif risk_level == 'moderate':
            recommendations.extend([
                "Enhance monitoring network",
                "Conduct regular hazard reviews",
                "Update contingency plans",
                "Monitor volatile emissions"
            ])
        
        # Specific recommendations
        if melt_fraction > 0.5:
            recommendations.append("Monitor magma chamber thermal evolution")
        
        if roof_stability['critical'] > 0.3:
            recommendations.append("Assess roof rock structural integrity")
        
        if predictions['volatile_content']['H2O'] > 0.04:
            recommendations.append("Monitor gas emissions for volatile saturation")
        
        return recommendations
    
    async def create_3d_visualization(self, caldera_id: str, 
                                    visualization_type: str = 'comprehensive') -> Dict:
        """Create 3D visualization of caldera system"""
        
        if caldera_id not in self.monitoring_cache:
            raise ValueError(f"Caldera {caldera_id} not found")
        
        self.logger.info(f"Creating 3D visualization for {caldera_id}")
        
        # Get data
        grid = self.monitoring_cache[caldera_id]['monitoring_grid']
        X, Y, Z = grid['coordinates']
        
        tomography = self.seismic_models.get(caldera_id)
        stress_field = self.stress_fields.get(caldera_id)
        reservoir = self.magma_reservoirs.get(caldera_id)
        
        if visualization_type == 'comprehensive':
            # Create comprehensive 3D visualization
            fig = self._create_comprehensive_3d_plot(
                X, Y, Z, tomography, stress_field, reservoir
            )
        elif visualization_type == 'stress_field':
            fig = self._create_stress_field_3d_plot(X, Y, Z, stress_field)
        elif visualization_type == 'tomography':
            fig = self._create_tomography_3d_plot(X, Y, Z, tomography)
        elif visualization_type == 'reservoir':
            fig = self._create_reservoir_3d_plot(X, Y, Z, reservoir)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")
        
        # Save visualization
        output_path = f"visualizations/{caldera_id}_{visualization_type}_3d.html"
        Path("visualizations").mkdir(exist_ok=True)
        fig.write_html(output_path)
        
        return {
            'caldera_id': caldera_id,
            'visualization_type': visualization_type,
            'output_path': output_path,
            'created_at': datetime.now().isoformat()
        }
    
    def _create_comprehensive_3d_plot(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                                    tomography: SeismicTomography,
                                    stress_field: StressField3D,
                                    reservoir: MagmaReservoir) -> go.Figure:
        """Create comprehensive 3D visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}],
                   [{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=['Velocity Anomalies', 'Stress Field', 
                           'Magma Chamber', 'Combined View']
        )
        
        # Subsample for visualization
        step = 3
        x_sub = X[::step, ::step, ::step]
        y_sub = Y[::step, ::step, ::step] 
        z_sub = Z[::step, ::step, ::step]
        
        if tomography:
            # Velocity anomalies
            vp_anomaly = tomography.velocity_anomalies['relative_percent'][::step, ::step, ::step]
            
            # Create isosurface for low velocity zones (potential melt)
            fig.add_trace(
                go.Isosurface(
                    x=x_sub.flatten(),
                    y=y_sub.flatten(), 
                    z=z_sub.flatten(),
                    value=vp_anomaly.flatten(),
                    isomin=-20,
                    isomax=-5,
                    colorscale='Reds',
                    name='Low Velocity Zones'
                ),
                row=1, col=1
            )
        
        if stress_field:
            # Stress field
            failure_pot = stress_field.failure_potential[::step, ::step, ::step]
            
            fig.add_trace(
                go.Isosurface(
                    x=x_sub.flatten(),
                    y=y_sub.flatten(),
                    z=z_sub.flatten(), 
                    value=failure_pot.flatten(),
                    isomin=0.5,
                    isomax=1.5,
                    colorscale='Plasma',
                    name='High Stress Zones'
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Comprehensive 3D Caldera Analysis",
            scene=dict(
                xaxis_title="Longitude",
                yaxis_title="Latitude", 
                zaxis_title="Depth (km)"
            ),
            showlegend=True
        )
        
        return fig
    
    async def generate_monitoring_report(self, caldera_id: str) -> Dict:
        """Generate comprehensive monitoring report"""
        
        if caldera_id not in self.monitoring_cache:
            raise ValueError(f"Caldera {caldera_id} not found")
        
        # Gather all analysis results
        tomography = self.seismic_models.get(caldera_id)
        stress_field = self.stress_fields.get(caldera_id)
        reservoir = self.magma_reservoirs.get(caldera_id)
        
        report = {
            'caldera_id': caldera_id,
            'report_timestamp': datetime.now().isoformat(),
            'monitoring_status': 'active',
            'analysis_summary': {},
            'risk_assessment': {},
            'recommendations': [],
            'data_quality': {},
            'trends': {},
            'visualizations': []
        }
        
        # Seismic tomography summary
        if tomography:
            report['analysis_summary']['tomography'] = {
                'velocity_model_resolution': f"{tomography.resolution_km} km",
                'melt_regions_identified': len(tomography.melt_regions),
                'velocity_anomaly_strength': float(np.std(tomography.velocity_anomalies['relative_percent'])),
                'model_uncertainty': float(np.mean(tomography.uncertainty))
            }
        
        # Stress field summary  
        if stress_field:
            report['analysis_summary']['stress_field'] = {
                'critical_zones': len(stress_field.critical_zones),
                'max_failure_potential': float(np.max(stress_field.failure_potential)),
                'mean_differential_stress': float(np.mean(stress_field.differential_stress)),
                'high_stress_volume_fraction': float(np.sum(stress_field.failure_potential > 0.8) / stress_field.failure_potential.size)
            }
        
        # Overall risk assessment
        risk_factors = []
        
        if tomography and len(tomography.melt_regions) > 2:
            risk_factors.append("Multiple melt regions detected")
        
        if stress_field and np.max(stress_field.failure_potential) > 1.0:
            risk_factors.append("High failure potential in critical zones")
        
        # Risk level
        if len(risk_factors) >= 3:
            risk_level = "high"
        elif len(risk_factors) >= 1:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        report['risk_assessment'] = {
            'overall_risk_level': risk_level,
            'risk_factors': risk_factors,
            'confidence': 0.8  # Placeholder
        }
        
        # Recommendations
        report['recommendations'] = self._generate_monitoring_recommendations(
            risk_level, tomography, stress_field
        )
        
        return report
    
    def _generate_monitoring_recommendations(self, risk_level: str,
                                           tomography: SeismicTomography,
                                           stress_field: StressField3D) -> List[str]:
        """Generate monitoring recommendations"""
        
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Implement continuous real-time monitoring",
                "Deploy additional seismic stations",
                "Increase GPS measurement frequency",
                "Monitor gas emissions continuously"
            ])
        elif risk_level == "moderate":
            recommendations.extend([
                "Enhance monitoring network density",
                "Conduct regular survey campaigns", 
                "Monitor key precursor signals"
            ])
        
        if tomography and len(tomography.melt_regions) > 0:
            recommendations.append("Focus monitoring on identified melt regions")
        
        if stress_field and len(stress_field.critical_zones) > 0:
            recommendations.append("Monitor stress evolution in critical zones")
        
        return recommendations

# Example usage and testing functions
async def example_resurgent_caldera_analysis():
    """Example of comprehensive resurgent caldera analysis"""
    
    # Initialize monitoring system
    monitor = ResurgentCalderaMonitoringSystem()
    
    # Example: Yellowstone Caldera
    caldera_bounds = {
        'north': 44.8,
        'south': 44.1,
        'east': -110.1,
        'west': -111.2,
        'max_depth': 15,
        'elevation': 2.5
    }
    
    # Initialize monitoring
    caldera_id = await monitor.initialize_caldera_monitoring(
        "yellowstone_test",
        (-110.65, 44.45),
        caldera_bounds
    )
    
    print(f"Initialized monitoring for {caldera_id}")
    
    # Simulate seismic data
    mock_seismic_data = {
        'bounds': caldera_bounds,
        'travel_times': [
            {'station': 'YNP01', 'event': 'E001', 'p_time': 5.2, 's_time': 9.1},
            {'station': 'YNP02', 'event': 'E001', 'p_time': 6.8, 's_time': 12.3}
        ],
        'stations': [
            {'id': 'YNP01', 'longitude': -110.5, 'latitude': 44.3},
            {'id': 'YNP02', 'longitude': -110.8, 'latitude': 44.6}
        ],
        'events': [
            {'id': 'E001', 'location': [-110.65, 44.45, -8.0], 'magnitude': 2.1}
        ]
    }
    
    # Process tomography
    tomography = await monitor.process_seismic_tomography(caldera_id, mock_seismic_data)
    print(f"Processed tomography: {len(tomography.melt_regions)} melt regions found")
    
    # Simulate deformation data
    mock_deformation = {
        'los_displacement': {
            'points': [
                {'lon': -110.6, 'lat': 44.4, 'displacement_mm': 15.2},
                {'lon': -110.7, 'lat': 44.5, 'displacement_mm': 8.7},
                {'lon': -110.8, 'lat': 44.3, 'displacement_mm': 3.1}
            ]
        }
    }
    
    # Analyze stress field
    stress_field = await monitor.analyze_3d_stress_field(
        caldera_id, mock_deformation, mock_seismic_data
    )
    print(f"Stress field analysis: {len(stress_field.critical_zones)} critical zones")
    
    # Create visualization
    viz_result = await monitor.create_3d_visualization(caldera_id, 'comprehensive')
    print(f"Created visualization: {viz_result['output_path']}")
    
    # Generate report
    report = await monitor.generate_monitoring_report(caldera_id)
    print(f"Generated report - Risk level: {report['risk_assessment']['overall_risk_level']}")
    
    print("Resurgent caldera analysis completed successfully!")

if __name__ == "__main__":
    asyncio.run(example_resurgent_caldera_analysis())