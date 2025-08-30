"""
Digital Twin Earth System
Real-time Earth system modeling with multi-scale coupling and data assimilation
"""

import numpy as np
import pandas as pd
import asyncio
import json
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import uuid
from pathlib import Path
import copy

# Scientific computing imports
try:
    import scipy.sparse as sp
    from scipy.integrate import solve_ivp, odeint
    from scipy.interpolate import RegularGridInterpolator, interp1d
    from scipy.ndimage import gaussian_filter, zoom
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Advanced numerical methods will be limited.")

# Geospatial and climate modeling
try:
    import xarray as xr
    import netCDF4 as nc
    import rasterio
    from rasterio.transform import from_bounds
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    warnings.warn("Geospatial libraries not available. Climate modeling features will be limited.")

# Machine learning for data assimilation
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    warnings.warn("ML libraries not available. Advanced data assimilation will be limited.")

# Parallel computing
try:
    import dask.array as da
    import dask.distributed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Distributed computing will be limited.")

warnings.filterwarnings('ignore')

class EarthSystemComponent(Enum):
    ATMOSPHERE = "atmosphere"
    HYDROSPHERE = "hydrosphere"
    LITHOSPHERE = "lithosphere"
    BIOSPHERE = "biosphere"
    CRYOSPHERE = "cryosphere"
    ANTHROPOSPHERE = "anthroposphere"

class ModelingScale(Enum):
    GLOBAL = "global"
    REGIONAL = "regional"
    LOCAL = "local"
    SITE_SPECIFIC = "site_specific"

class DataAssimilationMethod(Enum):
    KALMAN_FILTER = "kalman_filter"
    ENSEMBLE_KALMAN = "ensemble_kalman"
    PARTICLE_FILTER = "particle_filter"
    VARIATIONAL = "variational"
    HYBRID = "hybrid"

class UncertaintyType(Enum):
    ALEATORY = "aleatory"  # Natural variability
    EPISTEMIC = "epistemic"  # Knowledge uncertainty
    MODEL = "model"  # Model structure uncertainty
    PARAMETER = "parameter"  # Parameter uncertainty

@dataclass
class EarthSystemState:
    """Complete state of Earth system at given time"""
    timestamp: datetime
    components: Dict[EarthSystemComponent, np.ndarray]
    spatial_grid: Dict[str, np.ndarray]  # lat, lon, elevation grids
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'components': {k.value: v.tolist() for k, v in self.components.items()},
            'spatial_grid': {k: v.tolist() for k, v in self.spatial_grid.items()},
            'metadata': self.metadata,
            'uncertainty': {k: v.tolist() for k, v in self.uncertainty.items()} if self.uncertainty else None
        }

@dataclass
class ModelConfiguration:
    """Configuration for Earth system model"""
    spatial_resolution: float = 1.0  # degrees
    temporal_resolution: timedelta = timedelta(hours=1)
    domain_bounds: Tuple[float, float, float, float] = (-180, 180, -90, 90)  # lon_min, lon_max, lat_min, lat_max
    vertical_levels: int = 50
    enable_coupling: bool = True
    data_assimilation_method: DataAssimilationMethod = DataAssimilationMethod.ENSEMBLE_KALMAN
    uncertainty_quantification: bool = True

@dataclass
class ObservationData:
    """Observation data for assimilation"""
    observation_id: str
    timestamp: datetime
    location: Tuple[float, float, float]  # lat, lon, elevation
    variable_type: str
    value: float
    uncertainty: float
    instrument: str
    quality_flag: str = "GOOD"
    metadata: Dict[str, Any] = field(default_factory=dict)

class EarthSystemDynamics:
    """Core Earth system dynamics model"""
    
    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize spatial grids
        self._initialize_grids()
        
        # Physical constants
        self.constants = {
            'earth_radius': 6.371e6,  # meters
            'gravity': 9.81,  # m/s²
            'stefan_boltzmann': 5.67e-8,  # W/m²/K⁴
            'gas_constant': 287.0,  # J/kg/K for dry air
            'cp_air': 1004.0,  # J/kg/K specific heat of air
            'water_density': 1000.0,  # kg/m³
            'ice_density': 917.0,  # kg/m³
        }
        
        # Model state
        self.current_state = None
        self.state_history = []
    
    def _initialize_grids(self):
        """Initialize spatial grids"""
        lon_min, lon_max, lat_min, lat_max = self.config.domain_bounds
        
        # Create coordinate arrays
        lon_res = self.config.spatial_resolution
        lat_res = self.config.spatial_resolution
        
        self.longitudes = np.arange(lon_min, lon_max + lon_res, lon_res)
        self.latitudes = np.arange(lat_min, lat_max + lat_res, lat_res)
        self.pressure_levels = np.logspace(2, 5, self.config.vertical_levels)  # Pa
        
        # Create meshgrids
        self.lon_grid, self.lat_grid = np.meshgrid(self.longitudes, self.latitudes)
        
        # Grid dimensions
        self.nlat, self.nlon = len(self.latitudes), len(self.longitudes)
        self.nlev = self.config.vertical_levels
        
        self.logger.info(f"Initialized grids: {self.nlat}x{self.nlon}x{self.nlev}")
    
    def initialize_state(self, initial_conditions: Dict[str, Any] = None) -> EarthSystemState:
        """Initialize Earth system state"""
        components = {}
        
        # Atmosphere
        components[EarthSystemComponent.ATMOSPHERE] = self._initialize_atmosphere()
        
        # Hydrosphere
        components[EarthSystemComponent.HYDROSPHERE] = self._initialize_hydrosphere()
        
        # Lithosphere
        components[EarthSystemComponent.LITHOSPHERE] = self._initialize_lithosphere()
        
        # Biosphere
        components[EarthSystemComponent.BIOSPHERE] = self._initialize_biosphere()
        
        # Cryosphere
        components[EarthSystemComponent.CRYOSPHERE] = self._initialize_cryosphere()
        
        # Anthroposphere
        components[EarthSystemComponent.ANTHROPOSPHERE] = self._initialize_anthroposphere()
        
        # Spatial grid
        spatial_grid = {
            'longitude': self.longitudes,
            'latitude': self.latitudes,
            'pressure_levels': self.pressure_levels,
            'lon_grid': self.lon_grid,
            'lat_grid': self.lat_grid
        }
        
        # Initialize uncertainty if enabled
        uncertainty = None
        if self.config.uncertainty_quantification:
            uncertainty = self._initialize_uncertainty(components)
        
        self.current_state = EarthSystemState(
            timestamp=datetime.utcnow(),
            components=components,
            spatial_grid=spatial_grid,
            uncertainty=uncertainty,
            metadata={
                'model_version': '1.0',
                'initialization_method': 'climatology',
                'spatial_resolution': self.config.spatial_resolution,
                'vertical_levels': self.config.vertical_levels
            }
        )
        
        return self.current_state
    
    def _initialize_atmosphere(self) -> np.ndarray:
        """Initialize atmospheric state variables"""
        # Shape: (nlat, nlon, nlev, nvars)
        # Variables: temperature, humidity, u_wind, v_wind, pressure, density
        atm_state = np.zeros((self.nlat, self.nlon, self.nlev, 6))
        
        # Temperature profile (simple lapse rate)
        for k in range(self.nlev):
            altitude = -np.log(self.pressure_levels[k] / 101325) * 8400  # Scale height approximation
            temp_profile = 288.15 - 0.0065 * np.maximum(0, altitude)  # Standard atmosphere
            atm_state[:, :, k, 0] = temp_profile  # Temperature
        
        # Add realistic spatial variation
        lat_factor = np.sin(np.deg2rad(self.lat_grid))
        temp_variation = 40 * lat_factor  # Warmer at equator
        atm_state[:, :, :, 0] += temp_variation[:, :, np.newaxis]
        
        # Humidity (simplified)
        atm_state[:, :, :, 1] = 0.01 * np.exp(-self.pressure_levels / 20000)  # Decreasing with height
        
        # Wind components (simplified jet stream pattern)
        for k in range(self.nlev):
            if 20000 < self.pressure_levels[k] < 30000:  # Jet stream level
                jet_strength = 50 * np.exp(-((self.lat_grid - 45)**2) / 200)  # Gaussian jet
                atm_state[:, :, k, 2] = jet_strength  # u_wind
        
        # Pressure and density from hydrostatic equation
        atm_state[:, :, :, 4] = self.pressure_levels[np.newaxis, np.newaxis, :]
        atm_state[:, :, :, 5] = atm_state[:, :, :, 4] / (self.constants['gas_constant'] * atm_state[:, :, :, 0])
        
        return atm_state
    
    def _initialize_hydrosphere(self) -> np.ndarray:
        """Initialize hydrosphere state variables"""
        # Variables: sea_surface_temp, ocean_currents_u, ocean_currents_v, sea_level, salinity
        hydro_state = np.zeros((self.nlat, self.nlon, 5))
        
        # Sea surface temperature
        hydro_state[:, :, 0] = 273.15 + 25 * np.cos(np.deg2rad(self.lat_grid))  # Warmer at equator
        
        # Ocean currents (simplified)
        # Gulf Stream approximation
        gulf_stream_mask = (self.lon_grid > -80) & (self.lon_grid < -40) & \
                          (self.lat_grid > 25) & (self.lat_grid < 45)
        hydro_state[gulf_stream_mask, 2] = 1.0  # v_current northward
        
        # Sea level (reference)
        hydro_state[:, :, 3] = 0.0
        
        # Salinity
        hydro_state[:, :, 4] = 35.0  # PSU
        
        return hydro_state
    
    def _initialize_lithosphere(self) -> np.ndarray:
        """Initialize lithosphere state variables"""
        # Variables: temperature, stress_xx, stress_yy, stress_xy, displacement_z, heat_flow
        litho_state = np.zeros((self.nlat, self.nlon, 6))
        
        # Surface temperature (coupled to atmosphere)
        litho_state[:, :, 0] = 273.15 + 15 * np.cos(np.deg2rad(self.lat_grid))
        
        # Tectonic stress calculation using comprehensive plate boundary database
        # Major plate boundaries with different tectonic settings and stress regimes
        plate_boundary_zones = [
            # Transform boundaries (strike-slip stress regime)
            {
                'coords': ((-125, -110), (32, 42)),
                'name': 'San Andreas Transform',
                'type': 'transform',
                'stress_magnitude': 5e7,  # Pa
                'principal_stress_orientation': 45,  # degrees from north
                'stress_regime': 'strike_slip'
            },
            {
                'coords': ((-75, -65), (-55, -45)),
                'name': 'Falkland Transform',
                'type': 'transform', 
                'stress_magnitude': 3e7,
                'principal_stress_orientation': 90,
                'stress_regime': 'strike_slip'
            },
            
            # Convergent boundaries (compressive stress regime)
            {
                'coords': ((135, 145), (35, 45)),
                'name': 'Japan Subduction Zone',
                'type': 'convergent',
                'stress_magnitude': 8e7,
                'principal_stress_orientation': 135,  # NW-SE compression
                'stress_regime': 'thrust'
            },
            {
                'coords': ((-85, -65), (-20, 10)),
                'name': 'Andean Subduction Zone',
                'type': 'convergent',
                'stress_magnitude': 9e7,
                'principal_stress_orientation': 90,  # E-W compression
                'stress_regime': 'thrust'
            },
            {
                'coords': ((90, 110), (20, 35)),
                'name': 'Himalayan Collision Zone',
                'type': 'convergent',
                'stress_magnitude': 1.2e8,
                'principal_stress_orientation': 0,  # N-S compression
                'stress_regime': 'thrust'
            },
            
            # Divergent boundaries (extensional stress regime)
            {
                'coords': ((-35, -10), (-60, -40)),
                'name': 'Mid-Atlantic Ridge South',
                'type': 'divergent',
                'stress_magnitude': 2e7,
                'principal_stress_orientation': 90,  # E-W extension
                'stress_regime': 'normal'
            },
            {
                'coords': ((35, 45), (35, 45)),
                'name': 'East African Rift',
                'type': 'divergent',
                'stress_magnitude': 3e7,
                'principal_stress_orientation': 90,  # E-W extension
                'stress_regime': 'normal'
            },
            
            # Complex zones (mixed stress regimes)
            {
                'coords': ((20, 30), (35, 45)),
                'name': 'Mediterranean Collision Zone',
                'type': 'complex',
                'stress_magnitude': 6e7,
                'principal_stress_orientation': 45,
                'stress_regime': 'transpressive'
            },
            {
                'coords': ((165, 180), (-45, -35)),
                'name': 'Alpine Fault New Zealand',
                'type': 'transform',
                'stress_magnitude': 4e7,
                'principal_stress_orientation': 30,
                'stress_regime': 'transpressive'
            }
        ]
        
        # Initialize background stress field (lithostatic + regional tectonic stress)
        # Background compressive stress increases with depth (approximated at surface)
        background_stress = 1e6  # Pa, base stress level
        litho_state[:, :, 1] = background_stress  # sigma_xx
        litho_state[:, :, 2] = background_stress  # sigma_yy  
        litho_state[:, :, 3] = 0  # sigma_xy (shear stress)
        
        # Apply stress fields for each tectonic zone
        for zone in plate_boundary_zones:
            (lon_range, lat_range) = zone['coords']
            
            # Create mask for this tectonic zone
            mask = ((self.lon_grid >= lon_range[0]) & (self.lon_grid <= lon_range[1]) &
                   (self.lat_grid >= lat_range[0]) & (self.lat_grid <= lat_range[1]))
            
            if not np.any(mask):
                continue
                
            # Calculate stress tensor components based on tectonic setting
            stress_mag = zone['stress_magnitude']
            orientation = np.deg2rad(zone['principal_stress_orientation'])
            
            # Distance-based stress decay from plate boundary center
            center_lon = (lon_range[0] + lon_range[1]) / 2
            center_lat = (lat_range[0] + lat_range[1]) / 2
            
            # Great circle distance from boundary center using Haversine formula
            # Convert coordinates to radians
            lat1_rad = np.deg2rad(self.lat_grid)
            lon1_rad = np.deg2rad(self.lon_grid)
            lat2_rad = np.deg2rad(center_lat)
            lon2_rad = np.deg2rad(center_lon)
            
            # Haversine formula for great circle distance
            dlat = lat1_rad - lat2_rad
            dlon = lon1_rad - lon2_rad
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clamp to avoid numerical issues
            
            # Distance in degrees (for consistency with decay calculation)
            # Earth radius = 6371 km, so 1 degree ≈ 111 km
            distance = c * 180 / np.pi
            
            # Exponential decay with distance (characteristic length ~500 km)
            decay_factor = np.exp(-distance / 5.0)  # 5 degrees ≈ 500 km
            
            if zone['stress_regime'] == 'strike_slip':
                # Strike-slip: maximum shear stress at 45° to principal stress
                sigma_xx = stress_mag * 0.3 * np.cos(2 * orientation) * decay_factor
                sigma_yy = -stress_mag * 0.3 * np.cos(2 * orientation) * decay_factor
                sigma_xy = stress_mag * 0.7 * np.sin(2 * orientation) * decay_factor
                
            elif zone['stress_regime'] == 'thrust':
                # Thrust faulting: maximum compression horizontal
                sigma_xx = stress_mag * (0.8 * np.cos(orientation)**2 + 0.2) * decay_factor
                sigma_yy = stress_mag * (0.8 * np.sin(orientation)**2 + 0.2) * decay_factor
                sigma_xy = stress_mag * 0.6 * np.sin(2 * orientation) * decay_factor
                
            elif zone['stress_regime'] == 'normal':
                # Normal faulting: minimum compression horizontal
                sigma_xx = -stress_mag * 0.4 * np.cos(orientation)**2 * decay_factor
                sigma_yy = -stress_mag * 0.4 * np.sin(orientation)**2 * decay_factor
                sigma_xy = stress_mag * 0.3 * np.sin(2 * orientation) * decay_factor
                
            elif zone['stress_regime'] == 'transpressive':
                # Transpressive: combination of compression and shear
                sigma_xx = stress_mag * (0.5 * np.cos(orientation)**2 + 0.2) * decay_factor
                sigma_yy = stress_mag * (0.5 * np.sin(orientation)**2 + 0.2) * decay_factor
                sigma_xy = stress_mag * 0.8 * np.sin(2 * orientation) * decay_factor
                
            else:  # complex zones
                # Mixed stress regime with higher variability
                base_stress = stress_mag * 0.5 * decay_factor
                sigma_xx = base_stress * (1 + 0.3 * np.cos(2 * orientation))
                sigma_yy = base_stress * (1 - 0.3 * np.cos(2 * orientation))
                sigma_xy = base_stress * 0.6 * np.sin(2 * orientation)
            
            # Apply stress perturbations with some randomness
            stress_noise = 0.1  # 10% noise level
            litho_state[mask, 1] += sigma_xx[mask] * (1 + stress_noise * np.random.normal(0, 1, np.sum(mask)))
            litho_state[mask, 2] += sigma_yy[mask] * (1 + stress_noise * np.random.normal(0, 1, np.sum(mask)))  
            litho_state[mask, 3] += sigma_xy[mask] * (1 + stress_noise * np.random.normal(0, 1, np.sum(mask)))
        
        # Heat flow
        litho_state[:, :, 5] = 0.06 + 0.04 * np.random.random((self.nlat, self.nlon))  # W/m²
        
        return litho_state
    
    def _initialize_biosphere(self) -> np.ndarray:
        """Initialize biosphere state variables"""
        # Variables: biomass, carbon_flux, leaf_area_index, photosynthesis_rate, respiration_rate
        bio_state = np.zeros((self.nlat, self.nlon, 5))
        
        # Biomass distribution (higher in tropical regions)
        tropical_mask = np.abs(self.lat_grid) < 23.5
        temperate_mask = (np.abs(self.lat_grid) >= 23.5) & (np.abs(self.lat_grid) < 60)
        
        bio_state[tropical_mask, 0] = 400  # kg/m² biomass in tropics
        bio_state[temperate_mask, 0] = 200  # kg/m² biomass in temperate
        bio_state[:, :, 0] *= (1 + 0.3 * np.random.random((self.nlat, self.nlon)))  # Add variability
        
        # Leaf area index
        bio_state[:, :, 2] = np.maximum(0, 6 * np.exp(-np.abs(self.lat_grid) / 30))
        
        # Carbon fluxes (simplified)
        bio_state[:, :, 1] = -bio_state[:, :, 0] * 0.1  # Net carbon uptake
        bio_state[:, :, 3] = bio_state[:, :, 2] * 10    # Photosynthesis
        bio_state[:, :, 4] = bio_state[:, :, 0] * 0.05  # Respiration
        
        return bio_state
    
    def _initialize_cryosphere(self) -> np.ndarray:
        """Initialize cryosphere state variables"""
        # Variables: ice_thickness, snow_depth, ice_velocity_u, ice_velocity_v, ice_temperature
        cryo_state = np.zeros((self.nlat, self.nlon, 5))
        
        # Ice thickness (polar regions)
        arctic_mask = self.lat_grid > 70
        antarctic_mask = self.lat_grid < -70
        
        cryo_state[arctic_mask, 0] = 2.0  # meters sea ice
        cryo_state[antarctic_mask, 0] = 200.0  # meters ice sheet
        
        # Snow depth (seasonal, latitude dependent)
        winter_snow = np.maximum(0, 1.0 - np.abs(self.lat_grid) / 90) * 0.5  # meters
        cryo_state[:, :, 1] = winter_snow
        
        # Ice temperature
        cryo_state[:, :, 4] = np.minimum(273.15, 250 + 23 * np.cos(np.deg2rad(self.lat_grid)))
        
        return cryo_state
    
    def _initialize_anthroposphere(self) -> np.ndarray:
        """Initialize anthroposphere state variables"""
        # Variables: population_density, co2_emissions, land_use_change, urban_heat_island, energy_consumption
        anthro_state = np.zeros((self.nlat, self.nlon, 5))
        
        # Population density (simplified urban centers)
        urban_centers = [
            (-74, 40, 8e6),   # New York
            (139, 35, 9e6),   # Tokyo
            (2, 48, 2e6),     # Paris
            (-0.1, 51, 9e6),  # London
            (116, 39, 2e6),   # Beijing
        ]
        
        for lon, lat, pop in urban_centers:
            dist = np.sqrt((self.lon_grid - lon)**2 + (self.lat_grid - lat)**2)
            urban_influence = pop * np.exp(-dist**2 / 4)  # Gaussian distribution
            anthro_state[:, :, 0] += urban_influence
        
        # CO2 emissions (correlated with population)
        anthro_state[:, :, 1] = anthro_state[:, :, 0] * 1e-6  # kg/m²/s
        
        # Energy consumption
        anthro_state[:, :, 4] = anthro_state[:, :, 0] * 1e-9  # W/m²
        
        return anthro_state
    
    def _initialize_uncertainty(self, components: Dict[EarthSystemComponent, np.ndarray]) -> Dict[str, np.ndarray]:
        """Initialize uncertainty fields"""
        uncertainty = {}
        
        for component, data in components.items():
            # Initialize with 5-10% uncertainty
            base_uncertainty = 0.05 + 0.05 * np.random.random(data.shape)
            uncertainty[component.value] = np.abs(data) * base_uncertainty
        
        return uncertainty
    
    def step_forward(self, dt: float) -> EarthSystemState:
        """Advance Earth system one time step"""
        if self.current_state is None:
            raise ValueError("System not initialized. Call initialize_state() first.")
        
        # Extract current state
        current_components = self.current_state.components
        
        # Compute tendencies for each component
        tendencies = {}
        
        if self.config.enable_coupling:
            # Coupled system evolution
            tendencies[EarthSystemComponent.ATMOSPHERE] = self._atmosphere_tendency(current_components, dt)
            tendencies[EarthSystemComponent.HYDROSPHERE] = self._hydrosphere_tendency(current_components, dt)
            tendencies[EarthSystemComponent.LITHOSPHERE] = self._lithosphere_tendency(current_components, dt)
            tendencies[EarthSystemComponent.BIOSPHERE] = self._biosphere_tendency(current_components, dt)
            tendencies[EarthSystemComponent.CRYOSPHERE] = self._cryosphere_tendency(current_components, dt)
            tendencies[EarthSystemComponent.ANTHROPOSPHERE] = self._anthroposphere_tendency(current_components, dt)
        else:
            # Uncoupled evolution
            for component in current_components.keys():
                tendencies[component] = self._component_tendency(component, current_components[component], dt)
        
        # Update state
        new_components = {}
        for component, state in current_components.items():
            new_components[component] = state + tendencies[component] * dt
        
        # Update uncertainty if enabled
        new_uncertainty = None
        if self.config.uncertainty_quantification and self.current_state.uncertainty:
            new_uncertainty = self._propagate_uncertainty(self.current_state.uncertainty, tendencies, dt)
        
        # Create new state
        new_timestamp = self.current_state.timestamp + timedelta(seconds=dt)
        
        new_state = EarthSystemState(
            timestamp=new_timestamp,
            components=new_components,
            spatial_grid=self.current_state.spatial_grid,
            uncertainty=new_uncertainty,
            metadata=self.current_state.metadata
        )
        
        # Update current state and history
        self.state_history.append(self.current_state)
        self.current_state = new_state
        
        # Limit history size
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        return new_state
    
    def _atmosphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Compute atmospheric tendencies with coupling"""
        atm_state = components[EarthSystemComponent.ATMOSPHERE]
        tendency = np.zeros_like(atm_state)
        
        # Temperature evolution
        # Radiative heating/cooling
        solar_forcing = self._compute_solar_forcing()
        radiative_cooling = -2e-5 * (atm_state[:, :, :, 0] - 250)  # Simple cooling to space
        
        # Coupling with hydrosphere (SST influence)
        if EarthSystemComponent.HYDROSPHERE in components:
            sst = components[EarthSystemComponent.HYDROSPHERE][:, :, 0]
            surface_flux = 1e-4 * (sst[:, :, np.newaxis] - atm_state[:, :, -1, 0, np.newaxis])
            tendency[:, :, -1, 0] += surface_flux.squeeze()
        
        tendency[:, :, :, 0] = solar_forcing + radiative_cooling
        
        # Wind evolution (simplified)
        # Geostrophic adjustment and friction
        coriolis_param = 2 * 7.27e-5 * np.sin(np.deg2rad(self.lat_grid))  # Coriolis parameter
        
        for k in range(self.nlev):
            # Pressure gradient force (simplified)
            if k > 0:
                dp_dx = np.gradient(atm_state[:, :, k, 4], axis=1) / (self.constants['earth_radius'] * np.cos(np.deg2rad(self.lat_grid)))
                dp_dy = np.gradient(atm_state[:, :, k, 4], axis=0) / self.constants['earth_radius']
                
                density = atm_state[:, :, k, 5]
                
                # Geostrophic wind tendency
                tendency[:, :, k, 2] += -dp_dy / (density * coriolis_param + 1e-10)  # u-wind
                tendency[:, :, k, 3] += dp_dx / (density * coriolis_param + 1e-10)   # v-wind
                
                # Friction (simple)
                tendency[:, :, k, 2] -= 1e-5 * atm_state[:, :, k, 2]  # u-wind friction
                tendency[:, :, k, 3] -= 1e-5 * atm_state[:, :, k, 3]  # v-wind friction
        
        # Humidity evolution
        # Evaporation from ocean
        if EarthSystemComponent.HYDROSPHERE in components:
            sst = components[EarthSystemComponent.HYDROSPHERE][:, :, 0]
            evaporation_rate = 1e-8 * np.maximum(0, sst - 273.15)  # kg/kg/s
            tendency[:, :, -1, 1] += evaporation_rate
        
        # Precipitation (simple)
        condensation_rate = -1e-6 * np.maximum(0, atm_state[:, :, :, 1] - 0.02)
        tendency[:, :, :, 1] += condensation_rate
        
        return tendency
    
    def _hydrosphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Compute hydrosphere tendencies with coupling"""
        hydro_state = components[EarthSystemComponent.HYDROSPHERE]
        tendency = np.zeros_like(hydro_state)
        
        # Sea surface temperature evolution
        # Heat exchange with atmosphere
        if EarthSystemComponent.ATMOSPHERE in components:
            air_temp = components[EarthSystemComponent.ATMOSPHERE][:, :, -1, 0]  # Surface air temperature
            heat_flux = 5e-6 * (air_temp - hydro_state[:, :, 0])  # W/m²
            tendency[:, :, 0] = heat_flux / (self.constants['water_density'] * self.constants['cp_air'])
        
        # Ocean current evolution (simplified)
        # Wind stress coupling
        if EarthSystemComponent.ATMOSPHERE in components:
            wind_stress_x = 1e-3 * components[EarthSystemComponent.ATMOSPHERE][:, :, -1, 2]  # u-wind
            wind_stress_y = 1e-3 * components[EarthSystemComponent.ATMOSPHERE][:, :, -1, 3]  # v-wind
            
            tendency[:, :, 1] += wind_stress_x / self.constants['water_density']  # u-current
            tendency[:, :, 2] += wind_stress_y / self.constants['water_density']  # v-current
        
        # Friction
        tendency[:, :, 1] -= 1e-6 * hydro_state[:, :, 1]  # u-current friction
        tendency[:, :, 2] -= 1e-6 * hydro_state[:, :, 2]  # v-current friction
        
        return tendency
    
    def _lithosphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Compute comprehensive lithosphere tendencies with multi-physics coupling"""
        litho_state = components[EarthSystemComponent.LITHOSPHERE]
        tendency = np.zeros_like(litho_state)
        
        # Get coupling components
        atm_state = components[EarthSystemComponent.ATMOSPHERE]
        hydro_state = components[EarthSystemComponent.HYDROSPHERE]
        
        # === THERMAL EVOLUTION ===
        if SCIPY_AVAILABLE:
            # Surface temperature boundary condition from atmosphere
            surface_temp = atm_state[:, :, -1, 0]  # Surface air temperature
            ocean_temp = hydro_state[:, :, 0]      # Sea surface temperature
            
            # Use ocean temperature over water, air temperature over land
            # (simplified land-sea mask based on temperature difference)
            ocean_mask = np.abs(ocean_temp - surface_temp) < 5  # Within 5K suggests ocean
            boundary_temp = np.where(ocean_mask, ocean_temp, surface_temp)
            
            # Temperature diffusion with realistic parameters
            thermal_diffusivity = 1.2e-6  # m²/s for typical crustal rocks
            depth_scale = 30e3  # 30 km characteristic depth scale
            
            # Multi-scale diffusion: local + regional smoothing
            local_diffusion = gaussian_filter(litho_state[:, :, 0], sigma=0.5)
            regional_diffusion = gaussian_filter(litho_state[:, :, 0], sigma=2.0)
            
            # Surface heat flux coupling
            surface_flux = 0.1 * (boundary_temp - litho_state[:, :, 0])  # Surface coupling
            geothermal_gradient = 0.025e-3  # 25°C/km typical gradient
            deep_heat_source = geothermal_gradient * depth_scale / dt if dt > 0 else 0
            
            tendency[:, :, 0] = (
                thermal_diffusivity * (local_diffusion - litho_state[:, :, 0]) + 
                0.1 * thermal_diffusivity * (regional_diffusion - litho_state[:, :, 0]) +
                surface_flux + deep_heat_source
            )
        
        # === VISCOELASTIC STRESS EVOLUTION ===
        current_temp = litho_state[:, :, 0]
        
        # Temperature-dependent viscosity (Arrhenius law approximation)
        reference_temp = 1000  # K (typical upper mantle temperature)
        activation_energy = 500e3  # J/mol (typical for olivine)
        gas_constant = 8.314  # J/(mol⋅K)
        
        # Normalized viscosity (decreases exponentially with temperature)
        viscosity_factor = np.exp(activation_energy / (gas_constant * reference_temp) - 
                                activation_energy / (gas_constant * current_temp))
        viscosity_factor = np.clip(viscosity_factor, 0.01, 100)  # Reasonable bounds
        
        # Stress relaxation times based on temperature and rock type
        elastic_relaxation = 1e6 * viscosity_factor    # Elastic response (seconds)
        viscous_relaxation = 1e9 * viscosity_factor    # Viscous flow (seconds)
        
        # Maxwell viscoelastic model: σ̇ = -σ/τ_M + η⋅ε̇
        maxwell_time = elastic_relaxation * viscous_relaxation / (elastic_relaxation + viscous_relaxation)
        
        # Tectonic loading rates (background strain rates)
        plate_convergence_rate = 5e-15  # s⁻¹ (typical plate boundary)
        intraplate_rate = 1e-16         # s⁻¹ (stable interior)
        
        # Distance-based strain rate (higher near plate boundaries)
        # Reuse plate boundary information from initialization
        background_strain_rate = intraplate_rate
        
        # Comprehensive global plate boundary database with tectonic characteristics
        major_boundaries = [
            # Pacific Ring of Fire - Transform and Subduction Boundaries
            {
                'coords': ((-125, -110), (32, 42)),
                'name': 'San Andreas Transform System',
                'type': 'transform',
                'strain_rate_multiplier': 8.0,
                'boundary_width': 2.0  # degrees
            },
            {
                'coords': ((140, 150), (35, 45)),
                'name': 'Japan Trench Subduction Zone',
                'type': 'subduction',
                'strain_rate_multiplier': 12.0,
                'boundary_width': 3.0
            },
            {
                'coords': ((150, 165), (-45, -35)),
                'name': 'Kermadec-Tonga Subduction Zone',
                'type': 'subduction', 
                'strain_rate_multiplier': 10.0,
                'boundary_width': 2.5
            },
            {
                'coords': ((120, 135), (-15, 5)),
                'name': 'Java-Sumatra Subduction Zone',
                'type': 'subduction',
                'strain_rate_multiplier': 9.0,
                'boundary_width': 2.5
            },
            {
                'coords': ((-85, -65), (-20, 10)),
                'name': 'Andean Subduction Zone',
                'type': 'subduction',
                'strain_rate_multiplier': 11.0,
                'boundary_width': 4.0
            },
            {
                'coords': ((-165, -145), (50, 65)),
                'name': 'Aleutian Subduction Zone',
                'type': 'subduction',
                'strain_rate_multiplier': 7.0,
                'boundary_width': 3.0
            },
            
            # Mid-Ocean Ridge System - Divergent Boundaries
            {
                'coords': ((-35, -10), (-60, 60)),
                'name': 'Mid-Atlantic Ridge',
                'type': 'spreading',
                'strain_rate_multiplier': 3.0,
                'boundary_width': 1.5
            },
            {
                'coords': ((-120, -100), (-55, 45)),
                'name': 'East Pacific Rise',
                'type': 'spreading',
                'strain_rate_multiplier': 4.0,
                'boundary_width': 2.0
            },
            {
                'coords': ((60, 90), (-50, -20)),
                'name': 'Southwest Indian Ridge',
                'type': 'spreading',
                'strain_rate_multiplier': 2.5,
                'boundary_width': 1.0
            },
            {
                'coords': ((90, 140), (-35, -10)),
                'name': 'Southeast Indian Ridge',
                'type': 'spreading',
                'strain_rate_multiplier': 3.5,
                'boundary_width': 1.5
            },
            
            # Continental Collision and Orogenic Zones
            {
                'coords': ((70, 100), (25, 40)),
                'name': 'Himalayan-Tibetan Collision Zone',
                'type': 'collision',
                'strain_rate_multiplier': 15.0,
                'boundary_width': 8.0
            },
            {
                'coords': ((35, 50), (35, 45)),
                'name': 'Greater Caucasus',
                'type': 'collision',
                'strain_rate_multiplier': 6.0,
                'boundary_width': 2.0
            },
            {
                'coords': ((55, 75), (30, 40)),
                'name': 'Zagros Collision Zone',
                'type': 'collision',
                'strain_rate_multiplier': 7.0,
                'boundary_width': 3.0
            },
            {
                'coords': ((-10, 20), (35, 50)),
                'name': 'Alpine-Mediterranean System',
                'type': 'collision',
                'strain_rate_multiplier': 8.0,
                'boundary_width': 4.0
            },
            
            # Continental Rift Systems - Divergent Boundaries
            {
                'coords': ((30, 50), (-35, 15)),
                'name': 'East African Rift System',
                'type': 'continental_rift',
                'strain_rate_multiplier': 5.0,
                'boundary_width': 3.0
            },
            {
                'coords': ((-115, -105), (30, 40)),
                'name': 'Basin and Range Province',
                'type': 'continental_rift',
                'strain_rate_multiplier': 4.0,
                'boundary_width': 5.0
            },
            {
                'coords': ((10, 30), (45, 70)),
                'name': 'North Sea Rift System',
                'type': 'continental_rift',
                'strain_rate_multiplier': 2.0,
                'boundary_width': 2.0
            },
            
            # Major Transform Systems
            {
                'coords': ((-75, -65), (-55, -45)),
                'name': 'Falkland-Scotia Transform',
                'type': 'transform',
                'strain_rate_multiplier': 4.0,
                'boundary_width': 1.5
            },
            {
                'coords': ((165, 180), (-50, -40)),
                'name': 'Alpine Fault New Zealand',
                'type': 'transform',
                'strain_rate_multiplier': 6.0,
                'boundary_width': 1.0
            },
            {
                'coords': ((-180, -170), (50, 60)),
                'name': 'Queen Charlotte Transform',
                'type': 'transform',
                'strain_rate_multiplier': 5.0,
                'boundary_width': 1.5
            },
            
            # Back-arc and Marginal Sea Systems
            {
                'coords': ((120, 140), (20, 40)),
                'name': 'Philippine Sea Plate Boundaries',
                'type': 'complex',
                'strain_rate_multiplier': 8.0,
                'boundary_width': 3.0
            },
            {
                'coords': ((125, 145), (25, 45)),
                'name': 'Sea of Japan Back-arc',
                'type': 'back_arc',
                'strain_rate_multiplier': 4.0,
                'boundary_width': 2.0
            },
            
            # Intraplate Volcanic and Deformation Zones
            {
                'coords': ((-160, -150), (18, 25)),
                'name': 'Hawaiian Hotspot Chain',
                'type': 'hotspot',
                'strain_rate_multiplier': 2.0,
                'boundary_width': 1.0
            },
            {
                'coords': ((0, 20), (-25, -15)),
                'name': 'Cameroon Volcanic Line',
                'type': 'hotspot',
                'strain_rate_multiplier': 2.5,
                'boundary_width': 1.5
            }
        ]
        
        for boundary in major_boundaries:
            (lon_range, lat_range) = boundary['coords']
            boundary_type = boundary['type']
            strain_multiplier = boundary['strain_rate_multiplier']
            boundary_width = boundary['boundary_width']
            
            # Create boundary mask with buffer zone
            boundary_mask = ((self.lon_grid >= lon_range[0]) & (self.lon_grid <= lon_range[1]) &
                           (self.lat_grid >= lat_range[0]) & (self.lat_grid <= lat_range[1]))
            
            # Elevated strain rates near boundaries
            center_lon = (lon_range[0] + lon_range[1]) / 2
            center_lat = (lat_range[0] + lat_range[1]) / 2
            
            # Distance-based decay using Haversine formula
            lat1_rad = np.deg2rad(self.lat_grid)
            lon1_rad = np.deg2rad(self.lon_grid)
            lat2_rad = np.deg2rad(center_lat)
            lon2_rad = np.deg2rad(center_lon)
            
            dlat = lat1_rad - lat2_rad
            dlon = lon1_rad - lon2_rad
            a = (np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
            distance = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * 180 / np.pi
            
            # Boundary-type specific decay characteristics
            if boundary_type in ['collision', 'continental_rift']:
                # Wide deformation zones
                decay_scale = boundary_width * 2.0  # Broad influence
                base_strain_rate = plate_convergence_rate * strain_multiplier
            elif boundary_type in ['subduction', 'back_arc']:
                # Moderate width zones
                decay_scale = boundary_width * 1.5
                base_strain_rate = plate_convergence_rate * strain_multiplier
            elif boundary_type in ['transform']:
                # Narrow, focused zones
                decay_scale = boundary_width * 0.8
                base_strain_rate = plate_convergence_rate * strain_multiplier * 1.2  # Higher intensity
            elif boundary_type in ['spreading']:
                # Ridge-axis focused
                decay_scale = boundary_width * 1.0
                base_strain_rate = plate_convergence_rate * strain_multiplier * 0.6  # Lower magnitude
            elif boundary_type in ['hotspot']:
                # Localized volcanic zones
                decay_scale = boundary_width * 0.5
                base_strain_rate = intraplate_rate * strain_multiplier
            else:  # complex, back_arc
                decay_scale = boundary_width * 1.2
                base_strain_rate = plate_convergence_rate * strain_multiplier * 0.8
            
            # Apply distance-based strain rate enhancement
            proximity_factor = np.exp(-distance / decay_scale)
            background_strain_rate += base_strain_rate * proximity_factor
        
        # Comprehensive tectonic loading tensor calculation
        # Account for different stress regimes and tensor components
        shear_modulus = 30e9  # Pa (typical lithosphere shear modulus)
        bulk_modulus = 50e9   # Pa (typical bulk modulus)
        
        # Determine tectonic stress regime based on strain rate magnitude
        high_strain_threshold = 1e-14  # 1/s (active plate boundaries)
        moderate_strain_threshold = 1e-16  # 1/s (intraplate regions)
        
        # Initialize stress rate tensor components
        sigma_xx_rate = np.zeros_like(background_strain_rate)
        sigma_yy_rate = np.zeros_like(background_strain_rate)
        sigma_zz_rate = np.zeros_like(background_strain_rate)
        sigma_xy_rate = np.zeros_like(background_strain_rate)
        
        # High strain rate regions (extensional/compressional boundaries)
        high_strain_mask = background_strain_rate > high_strain_threshold
        sigma_xx_rate[high_strain_mask] = 2 * shear_modulus * background_strain_rate[high_strain_mask]
        sigma_yy_rate[high_strain_mask] = 2 * shear_modulus * background_strain_rate[high_strain_mask] * 0.7
        sigma_zz_rate[high_strain_mask] = bulk_modulus * background_strain_rate[high_strain_mask] * 0.5
        sigma_xy_rate[high_strain_mask] = shear_modulus * background_strain_rate[high_strain_mask] * 0.3
        
        # Moderate strain rate regions (transform boundaries)
        moderate_strain_mask = (background_strain_rate > moderate_strain_threshold) & (~high_strain_mask)
        sigma_xx_rate[moderate_strain_mask] = shear_modulus * background_strain_rate[moderate_strain_mask]
        sigma_yy_rate[moderate_strain_mask] = shear_modulus * background_strain_rate[moderate_strain_mask] * 0.8
        sigma_zz_rate[moderate_strain_mask] = bulk_modulus * background_strain_rate[moderate_strain_mask] * 0.3
        sigma_xy_rate[moderate_strain_mask] = shear_modulus * background_strain_rate[moderate_strain_mask] * 0.6
        
        # Low strain rate regions (intraplate)
        low_strain_mask = background_strain_rate <= moderate_strain_threshold
        sigma_xx_rate[low_strain_mask] = shear_modulus * background_strain_rate[low_strain_mask] * 0.5
        sigma_yy_rate[low_strain_mask] = shear_modulus * background_strain_rate[low_strain_mask] * 0.5
        sigma_zz_rate[low_strain_mask] = bulk_modulus * background_strain_rate[low_strain_mask] * 0.2
        sigma_xy_rate[low_strain_mask] = shear_modulus * background_strain_rate[low_strain_mask] * 0.1
        
        # Combined tectonic stress rate (using principal horizontal stress)
        tectonic_stress_rate = np.sqrt(sigma_xx_rate**2 + sigma_yy_rate**2 + 2*sigma_xy_rate**2)
        
        # Stress evolution: Maxwell viscoelastic + tectonic loading
        tendency[:, :, 1] = (-litho_state[:, :, 1] / maxwell_time +     # σxx relaxation
                           tectonic_stress_rate * (1 + 0.1 * np.random.randn(*litho_state[:, :, 1].shape)))
        tendency[:, :, 2] = (-litho_state[:, :, 2] / maxwell_time +     # σyy relaxation  
                           tectonic_stress_rate * (1 + 0.1 * np.random.randn(*litho_state[:, :, 2].shape)))
        tendency[:, :, 3] = (-litho_state[:, :, 3] / maxwell_time +     # σxy relaxation
                           0.5 * tectonic_stress_rate * (0.1 * np.random.randn(*litho_state[:, :, 3].shape)))
        
        # === VERTICAL DISPLACEMENT EVOLUTION ===
        # Elastic response to stress changes and thermal expansion
        bulk_modulus = 50e9     # Pa (typical crustal value)
        thermal_expansion = 3e-5  # /K
        
        # Comprehensive vertical displacement calculation
        
        # 1. Elastic stress-induced displacement with full tensor consideration
        sigma_xx = litho_state[:, :, 1]
        sigma_yy = litho_state[:, :, 2]
        sigma_zz = sigma_xx + sigma_yy  # Approximate vertical stress from horizontal components
        
        # Poisson effect and volumetric strain
        poisson_ratio = 0.25
        young_modulus = 2 * shear_modulus * (1 + poisson_ratio)
        
        # Vertical strain from horizontal stresses (Poisson effect)
        epsilon_zz_poisson = -poisson_ratio * (sigma_xx + sigma_yy) / young_modulus
        
        # Direct vertical compression
        epsilon_zz_direct = sigma_zz / young_modulus
        
        # Total elastic displacement rate
        lithosphere_thickness = 100e3  # 100 km typical lithosphere thickness
        elastic_displacement_rate = (epsilon_zz_poisson + epsilon_zz_direct) * lithosphere_thickness
        
        # 2. Enhanced thermal expansion with depth-dependent effects
        temp_change_rate = tendency[:, :, 0]
        
        # Depth-integrated thermal expansion (accounting for thermal gradient)
        surface_expansion = thermal_expansion * temp_change_rate * lithosphere_thickness
        
        # Thermal gradient effects (temperature decreases with depth)
        gradient_factor = 0.7  # Accounts for non-uniform temperature distribution
        thermal_displacement_rate = surface_expansion * gradient_factor
        
        # 3. Advanced isostatic adjustment with multiple timescales
        current_elevation = litho_state[:, :, 4]
        
        # Flexural isostasy (fast response, local loading)
        flexural_timescale = 1e3 * 365.25 * 24 * 3600  # 1,000 years
        flexural_rigidity = 1e23  # N⋅m (typical lithospheric flexural rigidity)
        local_load = np.abs(sigma_zz) * 1e-6  # Convert to MPa equivalent
        flexural_adjustment = -local_load / flexural_timescale * (1 - np.exp(-current_elevation**2 / 1e10))
        
        # Regional isostatic compensation (intermediate response)
        regional_timescale = 5e3 * 365.25 * 24 * 3600  # 5,000 years
        mantle_density = 3300  # kg/m³
        crustal_density = 2700  # kg/m³
        density_contrast = mantle_density - crustal_density
        
        # Airy isostatic equilibrium
        isostatic_equilibrium = -current_elevation * crustal_density / density_contrast
        regional_adjustment = (isostatic_equilibrium - current_elevation) / regional_timescale
        
        # Viscous mantle flow (slow response, large-scale)
        viscous_timescale = 2e4 * 365.25 * 24 * 3600  # 20,000 years
        mantle_viscosity = 1e21  # Pa⋅s (upper mantle viscosity)
        
        # Large-scale topographic relaxation
        topographic_wavelength = 500e3  # 500 km characteristic wavelength
        viscous_decay = np.exp(-np.abs(current_elevation) / topographic_wavelength)
        viscous_adjustment = -current_elevation / viscous_timescale * viscous_decay
        
        # 4. Tectonic vertical motion
        # Uplift/subsidence from horizontal tectonics
        horizontal_strain_rate = np.sqrt(sigma_xx_rate**2 + sigma_yy_rate**2) / shear_modulus
        
        # Transpression/transtension effects
        convergent_uplift_rate = horizontal_strain_rate * lithosphere_thickness * 0.1  # 10% coupling
        
        # Fault-related vertical motion
        fault_proximity_decay = 50e3  # 50 km decay distance
        fault_vertical_coupling = 0.05  # 5% horizontal-to-vertical coupling
        tectonic_vertical_rate = convergent_uplift_rate * fault_vertical_coupling
        
        # Combined vertical displacement rate
        total_vertical_rate = (elastic_displacement_rate + 
                             thermal_displacement_rate + 
                             flexural_adjustment + 
                             regional_adjustment + 
                             viscous_adjustment + 
                             tectonic_vertical_rate)
        
        tendency[:, :, 4] = total_vertical_rate
        
        # === HEAT FLOW EVOLUTION ===
        # Heat flow responds to temperature gradients and crustal processes
        surface_gradient = tendency[:, :, 0]  # Proxy for temperature gradient
        radiogenic_heat = 2e-6  # W/m² typical crustal heat production
        
        # Heat flow follows temperature changes with lag
        heat_flow_time_constant = 1e6 * 365.25 * 24 * 3600  # 1 million years
        
        tendency[:, :, 5] = (0.1 * surface_gradient +  # Response to thermal changes
                           radiogenic_heat / (litho_state[:, :, 5] + 0.01) - # Radiogenic production  
                           (litho_state[:, :, 5] - 0.06) / heat_flow_time_constant)  # Return to background
        
        return tendency
    
    def _biosphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Comprehensive biosphere tendencies with biogeochemical coupling"""
        bio_state = components[EarthSystemComponent.BIOSPHERE]
        tendency = np.zeros_like(bio_state)
        
        # Extract environmental drivers
        temperature = 288.15  # Default temperature (K)
        precipitation = 0.0   # Default precipitation (m/s)
        solar_radiation = 200.0  # Default solar radiation (W/m²)
        soil_moisture = 0.3   # Default soil moisture (fraction)
        
        if EarthSystemComponent.ATMOSPHERE in components:
            atm_state = components[EarthSystemComponent.ATMOSPHERE]
            temperature = atm_state[:, :, -1, 0]  # Surface temperature
            if atm_state.shape[-1] > 5:  # Check if precipitation exists
                precipitation = np.maximum(0, atm_state[:, :, -1, 5])  # Precipitation rate
        
        if EarthSystemComponent.LAND in components:
            land_state = components[EarthSystemComponent.LAND]
            if land_state.shape[-1] > 2:  # Check if soil moisture exists
                soil_moisture = np.clip(land_state[:, :, 2], 0.1, 0.8)  # Soil moisture
            if land_state.shape[-1] > 3:  # Check if solar radiation proxy
                solar_radiation = np.maximum(50, land_state[:, :, 3] * 300)  # Convert to W/m²
        
        # Advanced CO2 concentration calculation
        base_co2 = 400e-6  # Base atmospheric CO2
        co2_concentration = np.full_like(temperature, base_co2)
        
        if EarthSystemComponent.ANTHROPOSPHERE in components:
            anthro_state = components[EarthSystemComponent.ANTHROPOSPHERE]
            co2_emissions = anthro_state[:, :, 1] if anthro_state.shape[-1] > 1 else 0
            # Convert emissions to concentration perturbation
            co2_concentration = base_co2 + co2_emissions * 1e-9  # Simplified conversion
        
        # === 1. VEGETATION DYNAMICS ===
        # Multi-factor vegetation growth model
        
        # Temperature response with species-specific optima
        temp_celsius = temperature - 273.15
        
        # C3 plants (temperate): optimal 15-25°C
        c3_temp_factor = np.exp(-0.5 * ((temp_celsius - 20) / 15)**2)
        
        # C4 plants (tropical): optimal 25-35°C  
        c4_temp_factor = np.exp(-0.5 * ((temp_celsius - 30) / 12)**2)
        
        # Combine based on vegetation type (assume mixed)
        temp_factor = 0.6 * c3_temp_factor + 0.4 * c4_temp_factor
        
        # Water stress factor (Michaelis-Menten response)
        water_factor = (soil_moisture * precipitation * 1e6) / (0.2 + soil_moisture * precipitation * 1e6)
        water_factor = np.clip(water_factor, 0.1, 1.0)
        
        # Light response (saturating function)
        light_factor = solar_radiation / (100 + solar_radiation)
        
        # CO2 fertilization with diminishing returns
        co2_ref = 280e-6  # Pre-industrial CO2
        co2_factor = 1 + 0.3 * np.log(co2_concentration / co2_ref) * (1 - np.exp(-0.5 * co2_concentration / co2_ref))
        
        # Nutrient limitation (nitrogen cycle coupling)
        n_availability = np.clip(bio_state[:, :, 1] / 1000 + 0.1, 0.1, 1.0)  # Use carbon as N proxy
        nutrient_factor = n_availability / (0.3 + n_availability)
        
        # Combined growth rate with environmental coupling
        max_growth_rate = 2e-8  # Maximum growth rate (1/s)
        environmental_factor = temp_factor * water_factor * light_factor * co2_factor * nutrient_factor
        growth_rate = max_growth_rate * environmental_factor
        
        # Logistic growth with carrying capacity modulation
        carrying_capacity = 1500 * temp_factor * water_factor  # Dynamic carrying capacity
        biomass_density = bio_state[:, :, 0]
        tendency[:, :, 0] = growth_rate * biomass_density * (1 - biomass_density / carrying_capacity)
        
        # Natural mortality and disturbance
        mortality_rate = 5e-10 + 1e-9 * np.exp(-(temperature - 273.15) / 10)  # Temperature-dependent mortality
        disturbance_rate = 1e-10 * (1 + np.random.exponential(1, temperature.shape))  # Random disturbances
        tendency[:, :, 0] -= (mortality_rate + disturbance_rate) * biomass_density
        
        # === 2. CARBON CYCLE ===
        # Photosynthesis (carbon fixation)
        photosynthesis_efficiency = 0.02  # 2% of growth translates to carbon fixation
        carbon_fixation = tendency[:, :, 0] * photosynthesis_efficiency
        
        # Respiration (temperature-dependent)
        q10_respiration = 2.0  # Q10 factor for respiration
        base_respiration_rate = 1e-9  # Base respiration rate
        respiration_factor = q10_respiration ** ((temp_celsius - 15) / 10)
        respiration = base_respiration_rate * respiration_factor * bio_state[:, :, 1]
        
        # Decomposition and soil organic matter
        decomposition_rate = 5e-10 * np.exp(temp_celsius / 20) * soil_moisture
        decomposition = decomposition_rate * bio_state[:, :, 1]
        
        # Net carbon balance
        tendency[:, :, 1] = carbon_fixation - respiration - decomposition
        
        # === 3. NITROGEN CYCLE (if additional state variables exist) ===
        if bio_state.shape[-1] > 2:
            # Nitrogen fixation by legumes and microbes
            n_fixation_rate = 1e-11 * temp_factor * soil_moisture
            n_fixation = n_fixation_rate * (1000 - bio_state[:, :, 2])  # Asymptotic approach
            
            # Nitrogen mineralization from organic matter
            n_mineralization = 0.1 * decomposition
            
            # Nitrogen uptake by vegetation
            n_uptake = 0.05 * carbon_fixation  # Coupled to carbon fixation
            
            # Denitrification (anaerobic conditions)
            denitrification_rate = 2e-11 * (1 - soil_moisture) * np.exp(temp_celsius / 15)
            denitrification = denitrification_rate * bio_state[:, :, 2]
            
            # Net nitrogen balance
            tendency[:, :, 2] = n_fixation + n_mineralization - n_uptake - denitrification
        
        # === 4. ECOSYSTEM SERVICES (if additional variables exist) ===
        if bio_state.shape[-1] > 3:
            # Water regulation (transpiration)
            transpiration_rate = 1e-8 * biomass_density * temp_factor * (1 - soil_moisture)
            tendency[:, :, 3] = transpiration_rate
            
        # === 5. PHENOLOGICAL RESPONSES ===
        # Seasonal vegetation dynamics (simplified)
        day_of_year = (dt / (24 * 3600)) % 365
        seasonal_factor = 0.8 + 0.2 * np.cos(2 * np.pi * day_of_year / 365)
        tendency[:, :, 0] *= seasonal_factor
        
        return tendency
    
    def _cryosphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Advanced cryosphere tendencies with comprehensive ice dynamics"""
        cryo_state = components[EarthSystemComponent.CRYOSPHERE]
        tendency = np.zeros_like(cryo_state)
        
        # Environmental drivers
        air_temp = 273.15  # Default air temperature (K)
        surface_pressure = 101325  # Default surface pressure (Pa)
        wind_speed = 5.0   # Default wind speed (m/s)
        shortwave_radiation = 150.0  # Default solar radiation (W/m²)
        longwave_radiation = 300.0   # Default longwave radiation (W/m²)
        precipitation = 0.0  # Default precipitation (m/s)
        
        if EarthSystemComponent.ATMOSPHERE in components:
            atm_state = components[EarthSystemComponent.ATMOSPHERE]
            air_temp = atm_state[:, :, -1, 0]  # Surface air temperature
            if atm_state.shape[-1] > 1:
                surface_pressure = atm_state[:, :, -1, 1]
            if atm_state.shape[-1] > 2:
                wind_speed = np.sqrt(atm_state[:, :, -1, 2]**2 + atm_state[:, :, -1, 3]**2)
            if atm_state.shape[-1] > 5:
                precipitation = np.maximum(0, atm_state[:, :, -1, 5])
        
        # Extract current cryosphere state
        ice_thickness = cryo_state[:, :, 0]  # Ice thickness (m)
        snow_depth = cryo_state[:, :, 1] if cryo_state.shape[-1] > 1 else np.zeros_like(ice_thickness)
        
        # === 1. SNOW DYNAMICS ===
        freezing_point = 273.15
        
        # Snowfall (when T < 0°C and precipitation > 0)
        snow_fraction = 1.0 / (1 + np.exp(2 * (air_temp - freezing_point)))  # Sigmoid transition
        snowfall_rate = precipitation * snow_fraction  # m/s
        
        # Snow density evolution
        fresh_snow_density = 100  # kg/m³ (fresh snow)
        mature_snow_density = 400  # kg/m³ (mature snow)
        density_evolution_rate = 1e-7  # Densification rate
        
        # Snow metamorphism and densification
        snow_density = fresh_snow_density + (mature_snow_density - fresh_snow_density) * \
                      (1 - np.exp(-density_evolution_rate * dt))
        
        # Snow melt (degree-day model with energy balance)
        if snow_depth.max() > 0:
            # Energy balance components
            net_radiation = shortwave_radiation * (1 - 0.8) + longwave_radiation  # Albedo = 0.8 for snow
            sensible_heat = 10 * wind_speed * (air_temp - freezing_point)  # Simplified sensible heat
            latent_heat = 5 * wind_speed * np.maximum(0, air_temp - freezing_point)  # Latent heat
            
            # Total energy available for melting
            melt_energy = np.maximum(0, net_radiation + sensible_heat - latent_heat)  # W/m²
            
            # Convert energy to melt rate (latent heat of fusion = 334,000 J/kg)
            latent_heat_fusion = 334000  # J/kg
            melt_rate = melt_energy / (latent_heat_fusion * snow_density) * dt  # m/s equivalent
            
            # Apply temperature constraint
            temp_melt_factor = np.maximum(0, air_temp - freezing_point) / 10  # Temperature excess effect
            melt_rate *= temp_melt_factor
            
            # Update snow depth
            snow_change = snowfall_rate - melt_rate
            tendency[:, :, 1] = snow_change if cryo_state.shape[-1] > 1 else 0
        else:
            tendency[:, :, 1] = snowfall_rate if cryo_state.shape[-1] > 1 else 0
        
        # === 2. ICE DYNAMICS ===
        
        # Sea ice formation and melting
        seawater_freezing_point = freezing_point - 1.8  # Sea water freezing point
        
        # Ice formation (when air temp < freezing and no existing ice)
        ice_formation_rate = np.where(
            (air_temp < seawater_freezing_point) & (ice_thickness < 0.01),
            1e-6 * (seawater_freezing_point - air_temp),  # Formation rate proportional to cooling
            0.0
        )
        
        # Ice melting (comprehensive energy balance)
        if ice_thickness.max() > 0:
            # Surface energy balance
            ice_albedo = np.where(snow_depth > 0.01, 0.8, 0.5)  # Snow-covered vs bare ice
            absorbed_radiation = shortwave_radiation * (1 - ice_albedo)
            
            # Conductive heat flux through ice
            ice_thermal_conductivity = 2.0  # W/(m·K)
            ocean_temp = freezing_point + 2.0  # Assume ocean 2°C above freezing
            conductive_flux = ice_thermal_conductivity * (ocean_temp - air_temp) / \
                            np.maximum(ice_thickness, 0.01)  # Avoid division by zero
            
            # Total energy for melting
            total_melt_energy = absorbed_radiation + conductive_flux + sensible_heat
            
            # Convert to ice melt rate
            ice_density = 917  # kg/m³ (ice density)
            ice_melt_rate = np.maximum(0, total_melt_energy / (latent_heat_fusion * ice_density)) * dt
            
            # Apply temperature-dependent scaling
            temp_scale = np.maximum(0, (air_temp - seawater_freezing_point) / 5.0)
            ice_melt_rate *= (1 + temp_scale)
        else:
            ice_melt_rate = 0.0
        
        # Ice dynamics (growth from bottom, melting from top)
        bottom_growth_rate = np.where(
            (air_temp < seawater_freezing_point) & (ice_thickness > 0.01),
            2e-7 * (seawater_freezing_point - air_temp) / np.maximum(ice_thickness, 0.1),
            0.0
        )
        
        # Net ice thickness change
        net_ice_change = ice_formation_rate + bottom_growth_rate - ice_melt_rate
        tendency[:, :, 0] = net_ice_change
        
        # === 3. GLACIAL DYNAMICS (if additional variables exist) ===
        if cryo_state.shape[-1] > 2:
            # Glacier mass balance
            elevation_effect = np.maximum(0, (cryo_state[:, :, 2] - 1000) / 1000)  # Elevation dependency
            
            # Accumulation (snowfall at high elevation)
            accumulation_rate = snowfall_rate * (1 + 2 * elevation_effect)
            
            # Ablation (melting + sublimation)
            sublimation_rate = 1e-8 * wind_speed * np.maximum(0, air_temp - 263.15)  # Sublimation below -10°C
            ablation_rate = melt_rate + sublimation_rate
            
            # Net glacier change
            tendency[:, :, 2] = accumulation_rate - ablation_rate
        
        # === 4. PERMAFROST DYNAMICS (if additional variables exist) ===
        if cryo_state.shape[-1] > 3:
            # Active layer thickness based on temperature
            thawing_index = np.maximum(0, air_temp - freezing_point) * dt / (365.25 * 24 * 3600)
            active_layer_change = 1e-3 * np.sqrt(thawing_index)  # Stefan solution approximation
            
            # Permafrost stability
            mean_annual_temp = air_temp  # Simplified - would need annual averaging
            permafrost_threshold = freezing_point - 2.0  # -2°C threshold
            
            permafrost_change = np.where(
                mean_annual_temp > permafrost_threshold,
                -1e-8,  # Permafrost degradation
                1e-9    # Permafrost formation
            )
            
            tendency[:, :, 3] = active_layer_change + permafrost_change
        
        # === 5. ICE-ATMOSPHERE FEEDBACKS ===
        # Surface albedo feedback (affects radiation budget)
        total_ice_snow = ice_thickness + snow_depth
        surface_albedo = np.where(total_ice_snow > 0.01, 0.7, 0.3)
        
        # Modify energy balance based on albedo feedback
        albedo_feedback_factor = (surface_albedo - 0.3) / 0.4  # Normalized feedback
        tendency[:, :, 0] *= (1 - 0.2 * albedo_feedback_factor)  # Reduce melt with high albedo
        
        return tendency
    
    def _anthroposphere_tendency(self, components: Dict[EarthSystemComponent, np.ndarray], dt: float) -> np.ndarray:
        """Compute anthroposphere tendencies"""
        anthro_state = components[EarthSystemComponent.ANTHROPOSPHERE]
        tendency = np.zeros_like(anthro_state)
        
        # Population growth (simplified)
        growth_rate = 1e-9  # 1/s (very slow)
        tendency[:, :, 0] = growth_rate * anthro_state[:, :, 0]
        
        # CO2 emissions change
        tendency[:, :, 1] = 1e-12 * anthro_state[:, :, 0]  # Increase with population
        
        return tendency
    
    def _component_tendency(self, component: EarthSystemComponent, state: np.ndarray, dt: float) -> np.ndarray:
        """Compute tendency for uncoupled component"""
        tendency = np.zeros_like(state)
        
        # Simple decay/relaxation for uncoupled evolution
        relaxation_time = 3600.0  # 1 hour
        tendency = -state / relaxation_time
        
        return tendency
    
    def _compute_solar_forcing(self) -> np.ndarray:
        """Compute solar radiative forcing"""
        # Solar constant and earth-sun distance variation
        solar_constant = 1361  # W/m²
        
        # Seasonal variation (simplified)
        day_of_year = self.current_state.timestamp.timetuple().tm_yday
        solar_variation = 1 + 0.034 * np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Solar zenith angle
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365.25)
        hour_angle = 0  # Assume noon
        
        cos_zenith = (np.sin(np.deg2rad(self.lat_grid)) * np.sin(np.deg2rad(declination)) +
                     np.cos(np.deg2rad(self.lat_grid)) * np.cos(np.deg2rad(declination)) * np.cos(np.deg2rad(hour_angle)))
        
        # Solar heating (converted to temperature tendency)
        solar_heating = solar_constant * solar_variation * np.maximum(0, cos_zenith)
        
        # Convert to temperature tendency (simplified)
        heating_rate = solar_heating / (self.constants['cp_air'] * 1000)  # K/s (assuming 1000 kg/m² column)
        
        # Distribute vertically (more heating at surface)
        vertical_profile = np.exp(-np.arange(self.nlev) / 10)  # Exponential decay
        vertical_profile /= np.sum(vertical_profile)
        
        forcing = heating_rate[:, :, np.newaxis] * vertical_profile[np.newaxis, np.newaxis, :]
        
        return forcing
    
    def _propagate_uncertainty(self, current_uncertainty: Dict[str, np.ndarray], 
                             tendencies: Dict[EarthSystemComponent, np.ndarray], 
                             dt: float) -> Dict[str, np.ndarray]:
        """Propagate uncertainty forward in time"""
        new_uncertainty = {}
        
        for component_name, uncertainty in current_uncertainty.items():
            component = EarthSystemComponent(component_name)
            
            if component in tendencies:
                # Simple uncertainty propagation (could be made more sophisticated)
                tendency_uncertainty = np.abs(tendencies[component]) * dt * 0.1  # 10% of tendency magnitude
                new_uncertainty[component_name] = np.sqrt(uncertainty**2 + tendency_uncertainty**2)
            else:
                new_uncertainty[component_name] = uncertainty
        
        return new_uncertainty

class DataAssimilationSystem:
    """Advanced data assimilation for Digital Twin Earth with comprehensive filtering"""
    
    def __init__(self, method: DataAssimilationMethod = DataAssimilationMethod.ENSEMBLE_KALMAN):
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.observations = []
        self.ensemble_size = 100  # Increased for better statistics
        self.inflation_factor = 1.02  # Covariance inflation
        self.localization_radius = 500e3  # 500 km localization
        
        # Advanced observation operators with error modeling
        self.obs_operators = {
            'temperature': self._temperature_operator,
            'pressure': self._pressure_operator,
            'wind_speed': self._wind_speed_operator,
            'humidity': self._humidity_operator,
            'sea_surface_temperature': self._sst_operator,
            'geopotential_height': self._geopotential_operator,
            'precipitation': self._precipitation_operator,
            'soil_moisture': self._soil_moisture_operator,
            'snow_depth': self._snow_depth_operator,
            'ocean_temperature': self._ocean_temp_operator,
            'ocean_salinity': self._salinity_operator,
            'sea_ice_concentration': self._sea_ice_operator,
            'atmospheric_co2': self._co2_operator,
            'aerosol_optical_depth': self._aerosol_operator,
            'land_surface_temperature': self._lst_operator,
        }
        
        # Quality control parameters
        self.qc_params = {
            'temperature': {'min': 180.0, 'max': 330.0, 'std_threshold': 5.0},
            'pressure': {'min': 30000.0, 'max': 110000.0, 'std_threshold': 1000.0},
            'wind_speed': {'min': 0.0, 'max': 150.0, 'std_threshold': 10.0},
            'humidity': {'min': 0.0, 'max': 100.0, 'std_threshold': 20.0},
        }
        
        # Background error covariances
        self.background_errors = {
            'temperature': 2.0,      # K
            'pressure': 500.0,       # Pa
            'wind_speed': 3.0,       # m/s
            'humidity': 10.0,        # %
            'precipitation': 0.5,    # mm/hr
        }
    
    def add_observation(self, observation: ObservationData):
        """Add observation with comprehensive quality control"""
        # Quality control check
        if self._quality_control(observation):
            # Apply observation error inflation if needed
            observation.uncertainty = self._adjust_observation_error(observation)
            self.observations.append(observation)
            self.logger.debug(f"Added QC-passed observation: {observation.variable_type} at {observation.location}")
        else:
            self.logger.warning(f"Observation failed QC: {observation.variable_type} at {observation.location}")
    
    def _quality_control(self, obs: ObservationData) -> bool:
        """Comprehensive quality control for observations"""
        var_type = obs.variable_type
        if var_type not in self.qc_params:
            return True  # No QC parameters defined, assume good
        
        params = self.qc_params[var_type]
        value = obs.value
        
        # Range check
        if value < params['min'] or value > params['max']:
            return False
        
        # Gross error check (if we have background)
        if hasattr(obs, 'background_value'):
            if abs(value - obs.background_value) > params['std_threshold'] * obs.uncertainty:
                return False
        
        return True
    
    def _adjust_observation_error(self, obs: ObservationData) -> float:
        """Adjust observation error based on representativeness"""
        base_error = obs.uncertainty
        
        # Representativeness error for different observation types
        repr_errors = {
            'temperature': 0.5,      # K
            'pressure': 100.0,       # Pa
            'wind_speed': 1.0,       # m/s
            'humidity': 5.0,         # %
        }
        
        repr_error = repr_errors.get(obs.variable_type, 0.0)
        return np.sqrt(base_error**2 + repr_error**2)
    
    def assimilate_data(self, earth_state: EarthSystemState, 
                       observations: List[ObservationData]) -> EarthSystemState:
        """Assimilate observations into Earth system state with advanced methods"""
        if not observations:
            return earth_state
        
        # Filter observations by quality and temporal relevance
        valid_obs = self._filter_observations(observations)
        if not valid_obs:
            self.logger.warning("No valid observations after filtering")
            return earth_state
        
        if self.method == DataAssimilationMethod.ENSEMBLE_KALMAN:
            return self._ensemble_kalman_filter(earth_state, valid_obs)
        elif self.method == DataAssimilationMethod.VARIATIONAL:
            return self._variational_assimilation(earth_state, valid_obs)
        elif self.method == DataAssimilationMethod.HYBRID:
            return self._hybrid_assimilation(earth_state, valid_obs)
        else:
            self.logger.warning(f"Assimilation method {self.method} not implemented")
            return earth_state
    
    def _filter_observations(self, observations: List[ObservationData]) -> List[ObservationData]:
        """Filter observations by quality and relevance"""
        valid_obs = []
        current_time = time.time()
        
        for obs in observations:
            # Temporal filter (observations within last hour)
            if hasattr(obs, 'timestamp') and (current_time - obs.timestamp) > 3600:
                continue
                
            # Spatial filter (remove duplicates within small radius)
            is_duplicate = False
            for valid_obs_item in valid_obs:
                if (obs.variable_type == valid_obs_item.variable_type and
                    self._haversine_distance(obs.location, valid_obs_item.location) < 10e3):  # 10 km
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                valid_obs.append(obs)
        
        return valid_obs
    
    def _haversine_distance(self, loc1: tuple, loc2: tuple) -> float:
        """Calculate great circle distance between two points"""
        lat1, lon1 = np.radians(loc1)
        lat2, lon2 = np.radians(loc2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371000 * c  # Earth radius in meters
    
    def _ensemble_kalman_filter(self, earth_state: EarthSystemState, 
                               observations: List[ObservationData]) -> EarthSystemState:
        """Advanced Ensemble Kalman Filter with localization and inflation"""
        if not ML_AVAILABLE:
            self.logger.warning("ML not available, using simplified assimilation")
            return self._simple_assimilation(earth_state, observations)
        
        # Generate ensemble of perturbed states
        ensemble_states = self._generate_ensemble(earth_state)
        
        for obs in observations:
            # Forward operator application
            H_x = self._apply_observation_operator(ensemble_states, obs)
            
            # Innovation calculation
            innovations = obs.value - H_x
            
            # Observation error covariance
            R = obs.uncertainty**2
            
            # Background error covariance (ensemble-based)
            H_x_mean = np.mean(H_x)
            H_B_HT = np.var(H_x) * self.inflation_factor
            
            # Kalman gain with localization
            localization_weights = self._compute_localization_weights(obs, ensemble_states)
            K = (H_B_HT * localization_weights) / (H_B_HT + R)
            
            # State update for each ensemble member
            for i, state in enumerate(ensemble_states):
                # Perturbed observations for ensemble consistency
                perturbed_obs = obs.value + np.random.normal(0, obs.uncertainty)
                innovation_i = perturbed_obs - H_x[i]
                
                # Update state
                self._update_state_with_innovation(state, obs, K[i] * innovation_i)
        
        # Return ensemble mean as updated state
        return self._compute_ensemble_mean(ensemble_states)
    
    def _generate_ensemble(self, earth_state: EarthSystemState) -> List[EarthSystemState]:
        """Generate ensemble of perturbed states"""
        ensemble = []
        
        for i in range(self.ensemble_size):
            # Create perturbed state
            perturbed_state = copy.deepcopy(earth_state)
            
            # Add random perturbations based on background error
            for component in ['atmosphere', 'ocean', 'land']:
                if hasattr(perturbed_state, component):
                    component_data = getattr(perturbed_state, component)
                    if hasattr(component_data, 'temperature'):
                        noise = np.random.normal(0, self.background_errors['temperature'], 
                                               component_data.temperature.shape)
                        component_data.temperature += noise
                    
                    if hasattr(component_data, 'pressure'):
                        noise = np.random.normal(0, self.background_errors['pressure'], 
                                               component_data.pressure.shape)
                        component_data.pressure += noise
            
            ensemble.append(perturbed_state)
        
        return ensemble
    
    def _apply_observation_operator(self, ensemble_states: List[EarthSystemState], 
                                  obs: ObservationData) -> np.ndarray:
        """Apply observation operator to ensemble states"""
        H_x = np.zeros(len(ensemble_states))
        
        operator = self.obs_operators.get(obs.variable_type)
        if operator is None:
            self.logger.warning(f"No operator for {obs.variable_type}")
            return H_x
        
        for i, state in enumerate(ensemble_states):
            H_x[i] = operator(state, obs.location)
        
        return H_x
    
    def _compute_localization_weights(self, obs: ObservationData, 
                                    ensemble_states: List[EarthSystemState]) -> np.ndarray:
        """Compute localization weights using Gaspari-Cohn function"""
        weights = np.ones(len(ensemble_states))
        
        # Define localization radius based on observation type
        localization_radii = {
            'temperature': 500e3,  # 500 km for temperature
            'pressure': 1000e3,    # 1000 km for pressure
            'velocity': 200e3,     # 200 km for velocity
            'displacement': 100e3,  # 100 km for displacement
            'seismic': 150e3,      # 150 km for seismic data
            'gravity': 2000e3,     # 2000 km for gravity
            'magnetic': 1500e3,    # 1500 km for magnetic
            'default': 300e3       # Default 300 km
        }
        
        localization_radius = localization_radii.get(obs.variable_type, localization_radii['default'])
        
        # Compute distances from observation location to state grid points
        obs_lat, obs_lon = obs.location[0], obs.location[1]
        
        for i, state in enumerate(ensemble_states):
            if hasattr(state, 'grid_coordinates'):
                # Use actual grid coordinates if available
                state_lats = state.grid_coordinates.get('latitude', [obs_lat])
                state_lons = state.grid_coordinates.get('longitude', [obs_lon])
            else:
                # Use representative coordinates based on state properties
                state_lats = [obs_lat + np.random.normal(0, 0.1)]  # Small perturbation
                state_lons = [obs_lon + np.random.normal(0, 0.1)]
            
            # Compute minimum distance to any grid point in this ensemble member
            min_distance = float('inf')
            for state_lat, state_lon in zip(state_lats, state_lons):
                # Haversine distance calculation
                R = 6371e3  # Earth radius in meters
                lat1_rad = np.radians(obs_lat)
                lat2_rad = np.radians(state_lat)
                delta_lat = np.radians(state_lat - obs_lat)
                delta_lon = np.radians(state_lon - obs_lon)
                
                a = (np.sin(delta_lat/2)**2 + 
                     np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                distance = R * c
                
                min_distance = min(min_distance, distance)
            
            # Apply Gaspari-Cohn localization function
            normalized_distance = min_distance / localization_radius
            
            if normalized_distance >= 2.0:
                # No influence beyond 2 localization radii
                weights[i] = 0.0
            elif normalized_distance >= 1.0:
                # Smooth decay between 1 and 2 radii
                z = normalized_distance
                weights[i] = (1/12) * z**5 - (1/2) * z**4 + (5/8) * z**3 + (5/3) * z**2 - 5*z + 4 - (2/3) / z
            else:
                # Full weight with smooth transition within 1 radius
                z = normalized_distance
                weights[i] = 1 - (5/3) * z**2 + (5/8) * z**3 + (1/2) * z**4 - (1/4) * z**5
        
        # Ensure weights are positive and normalized
        weights = np.maximum(weights, 0.0)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        else:
            weights = np.ones(len(ensemble_states)) / len(ensemble_states)
        
        return weights
    
    def _update_state_with_innovation(self, state: EarthSystemState, 
                                    obs: ObservationData, innovation: float):
        """Update state components with innovation"""
        # Comprehensive state variable mapping and update system
        
        # Define update scaling factors based on variable type and uncertainty
        update_scalings = {
            'temperature': 0.15,
            'pressure': 0.12,
            'velocity': 0.08,
            'displacement': 0.25,
            'seismic': 0.10,
            'gravity': 0.05,
            'magnetic': 0.06,
            'density': 0.03,
            'stress': 0.20,
            'strain': 0.18,
            'flow': 0.14
        }
        
        # Get appropriate scaling factor
        update_scaling = update_scalings.get(obs.variable_type, 0.1)
        
        # Apply observation error-based adaptive scaling
        if hasattr(obs, 'error_variance') and obs.error_variance > 0:
            # Reduce update for high-uncertainty observations
            error_scaling = 1.0 / (1.0 + obs.error_variance / np.mean(list(update_scalings.values())))
            update_scaling *= error_scaling
        
        # Map innovation to specific state variables based on observation type
        if obs.variable_type == 'temperature':
            self._update_temperature_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'pressure':
            self._update_pressure_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'velocity':
            self._update_velocity_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'displacement':
            self._update_displacement_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'seismic':
            self._update_seismic_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'gravity':
            self._update_gravity_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'magnetic':
            self._update_magnetic_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'stress':
            self._update_stress_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'strain':
            self._update_strain_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'density':
            self._update_density_fields(state, innovation, update_scaling, obs.location)
        
        elif obs.variable_type == 'flow':
            self._update_flow_fields(state, innovation, update_scaling, obs.location)
        
        else:
            # Generic update for unknown variable types
            self._generic_state_update(state, innovation, update_scaling, obs)

    def _update_temperature_fields(self, state: EarthSystemState, innovation: float, 
                                 scaling: float, location: tuple):
        """Update temperature-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'atmosphere') and hasattr(state.atmosphere, 'temperature'):
            state.atmosphere.temperature += scaled_innovation
        
        if hasattr(state, 'ocean') and hasattr(state.ocean, 'temperature'):
            # Ocean temperature updates with depth-dependent scaling
            depth_scaling = 0.8  # Reduced update at depth
            state.ocean.temperature += scaled_innovation * depth_scaling
        
        if hasattr(state, 'land_surface') and hasattr(state.land_surface, 'temperature'):
            state.land_surface.temperature += scaled_innovation
    
    def _update_pressure_fields(self, state: EarthSystemState, innovation: float, 
                              scaling: float, location: tuple):
        """Update pressure-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'atmosphere') and hasattr(state.atmosphere, 'pressure'):
            state.atmosphere.pressure += scaled_innovation
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'pressure'):
            state.solid_earth.pressure += scaled_innovation
    
    def _update_velocity_fields(self, state: EarthSystemState, innovation: float, 
                              scaling: float, location: tuple):
        """Update velocity-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'velocity'):
            if hasattr(state.solid_earth.velocity, 'components'):
                # Update velocity components based on observation direction
                state.solid_earth.velocity.components[0] += scaled_innovation * 0.6  # x-component
                state.solid_earth.velocity.components[1] += scaled_innovation * 0.3  # y-component
                state.solid_earth.velocity.components[2] += scaled_innovation * 0.1  # z-component
        
        if hasattr(state, 'atmosphere') and hasattr(state.atmosphere, 'wind_velocity'):
            state.atmosphere.wind_velocity += scaled_innovation * 0.5
    
    def _update_displacement_fields(self, state: EarthSystemState, innovation: float, 
                                  scaling: float, location: tuple):
        """Update displacement-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'displacement'):
            if hasattr(state.solid_earth.displacement, 'components'):
                # Apply displacement innovation to all components
                state.solid_earth.displacement.components[0] += scaled_innovation * 0.7  # x
                state.solid_earth.displacement.components[1] += scaled_innovation * 0.7  # y
                state.solid_earth.displacement.components[2] += scaled_innovation * 0.6  # z (vertical)
    
    def _update_seismic_fields(self, state: EarthSystemState, innovation: float, 
                             scaling: float, location: tuple):
        """Update seismic-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth'):
            if hasattr(state.solid_earth, 'seismic_velocity'):
                state.solid_earth.seismic_velocity += scaled_innovation
            
            if hasattr(state.solid_earth, 'stress_tensor'):
                # Update stress tensor components
                stress_scaling = 0.3  # Conservative for stress updates
                state.solid_earth.stress_tensor += scaled_innovation * stress_scaling
    
    def _update_gravity_fields(self, state: EarthSystemState, innovation: float, 
                             scaling: float, location: tuple):
        """Update gravity-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'gravity_field'):
            state.solid_earth.gravity_field += scaled_innovation
    
    def _update_magnetic_fields(self, state: EarthSystemState, innovation: float, 
                              scaling: float, location: tuple):
        """Update magnetic-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'magnetic_field'):
            state.solid_earth.magnetic_field += scaled_innovation
    
    def _update_stress_fields(self, state: EarthSystemState, innovation: float, 
                            scaling: float, location: tuple):
        """Update stress-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'stress_tensor'):
            state.solid_earth.stress_tensor += scaled_innovation
    
    def _update_strain_fields(self, state: EarthSystemState, innovation: float, 
                            scaling: float, location: tuple):
        """Update strain-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'strain_tensor'):
            state.solid_earth.strain_tensor += scaled_innovation
    
    def _update_density_fields(self, state: EarthSystemState, innovation: float, 
                             scaling: float, location: tuple):
        """Update density-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'density'):
            state.solid_earth.density += scaled_innovation
        
        if hasattr(state, 'atmosphere') and hasattr(state.atmosphere, 'density'):
            state.atmosphere.density += scaled_innovation * 0.5
    
    def _update_flow_fields(self, state: EarthSystemState, innovation: float, 
                          scaling: float, location: tuple):
        """Update flow-related state fields"""
        scaled_innovation = innovation * scaling
        
        if hasattr(state, 'solid_earth') and hasattr(state.solid_earth, 'mantle_flow'):
            state.solid_earth.mantle_flow += scaled_innovation
        
        if hasattr(state, 'ocean') and hasattr(state.ocean, 'current_velocity'):
            state.ocean.current_velocity += scaled_innovation * 0.4
    
    def _generic_state_update(self, state: EarthSystemState, innovation: float, 
                            scaling: float, obs: ObservationData):
        """Generic update for unknown observation types"""
        scaled_innovation = innovation * scaling * 0.5  # Conservative scaling
        
        # Try to find any numerical field that might be related
        for component_name in ['atmosphere', 'solid_earth', 'ocean', 'land_surface']:
            if hasattr(state, component_name):
                component = getattr(state, component_name)
                # Look for any numerical attributes to update
                for attr_name in dir(component):
                    if not attr_name.startswith('_'):
                        attr = getattr(component, attr_name)
                        if isinstance(attr, (int, float)):
                            setattr(component, attr_name, attr + scaled_innovation * 0.1)
                        elif isinstance(attr, np.ndarray):
                            setattr(component, attr_name, attr + scaled_innovation * 0.1)
    
    def _compute_ensemble_mean(self, ensemble_states: List[EarthSystemState]) -> EarthSystemState:
        """Compute ensemble mean state"""
        if not ensemble_states:
            raise ValueError("Cannot compute ensemble mean: no states provided")
        
        if len(ensemble_states) == 1:
            return copy.deepcopy(ensemble_states[0])
        
        # Initialize mean state as deep copy of first member
        mean_state = copy.deepcopy(ensemble_states[0])
        n_members = len(ensemble_states)
        
        # Compute means for each Earth system component
        self._compute_component_mean(mean_state, ensemble_states, 'atmosphere', n_members)
        self._compute_component_mean(mean_state, ensemble_states, 'solid_earth', n_members)
        self._compute_component_mean(mean_state, ensemble_states, 'ocean', n_members)
        self._compute_component_mean(mean_state, ensemble_states, 'land_surface', n_members)
        self._compute_component_mean(mean_state, ensemble_states, 'biosphere', n_members)
        self._compute_component_mean(mean_state, ensemble_states, 'cryosphere', n_members)
        
        # Compute means for any additional components
        for attr_name in dir(mean_state):
            if (not attr_name.startswith('_') and 
                attr_name not in ['atmosphere', 'solid_earth', 'ocean', 'land_surface', 'biosphere', 'cryosphere'] and
                hasattr(mean_state, attr_name)):
                self._compute_component_mean(mean_state, ensemble_states, attr_name, n_members)
        
        return mean_state
    
    def _compute_component_mean(self, mean_state: EarthSystemState, 
                              ensemble_states: List[EarthSystemState], 
                              component_name: str, n_members: int):
        """Compute mean for a specific Earth system component"""
        if not hasattr(mean_state, component_name):
            return
        
        mean_component = getattr(mean_state, component_name)
        if mean_component is None:
            return
        
        # Collect all components from ensemble members
        components = []
        for state in ensemble_states:
            if hasattr(state, component_name):
                component = getattr(state, component_name)
                if component is not None:
                    components.append(component)
        
        if not components:
            return
        
        # Compute means for all numerical attributes in the component
        self._average_component_attributes(mean_component, components)
    
    def _average_component_attributes(self, mean_component, components):
        """Average all numerical attributes of a component across ensemble members"""
        # Get all attribute names from the mean component
        for attr_name in dir(mean_component):
            if attr_name.startswith('_'):
                continue
                
            if not hasattr(mean_component, attr_name):
                continue
            
            mean_attr = getattr(mean_component, attr_name)
            
            # Handle different types of attributes
            if isinstance(mean_attr, (int, float)):
                # Scalar numerical values
                values = []
                for component in components:
                    if hasattr(component, attr_name):
                        val = getattr(component, attr_name)
                        if isinstance(val, (int, float)):
                            values.append(val)
                
                if values:
                    setattr(mean_component, attr_name, np.mean(values))
            
            elif isinstance(mean_attr, np.ndarray):
                # Array values
                arrays = []
                for component in components:
                    if hasattr(component, attr_name):
                        arr = getattr(component, attr_name)
                        if isinstance(arr, np.ndarray) and arr.shape == mean_attr.shape:
                            arrays.append(arr)
                
                if arrays:
                    # Compute element-wise mean
                    mean_array = np.mean(np.stack(arrays, axis=0), axis=0)
                    setattr(mean_component, attr_name, mean_array)
            
            elif hasattr(mean_attr, '__dict__'):
                # Nested objects - recursively average their attributes
                nested_objects = []
                for component in components:
                    if hasattr(component, attr_name):
                        nested_obj = getattr(component, attr_name)
                        if hasattr(nested_obj, '__dict__'):
                            nested_objects.append(nested_obj)
                
                if nested_objects:
                    self._average_component_attributes(mean_attr, nested_objects)
            
            elif isinstance(mean_attr, dict):
                # Dictionary values
                self._average_dict_values(mean_attr, components, attr_name)
            
            elif isinstance(mean_attr, list):
                # List values (if they contain numerical data)
                self._average_list_values(mean_attr, components, attr_name)
    
    def _average_dict_values(self, mean_dict, components, attr_name):
        """Average dictionary values across ensemble members"""
        if not isinstance(mean_dict, dict):
            return
        
        for key in mean_dict.keys():
            values = []
            for component in components:
                if hasattr(component, attr_name):
                    comp_dict = getattr(component, attr_name)
                    if isinstance(comp_dict, dict) and key in comp_dict:
                        val = comp_dict[key]
                        if isinstance(val, (int, float)):
                            values.append(val)
                        elif isinstance(val, np.ndarray):
                            values.append(val)
            
            if values:
                if isinstance(values[0], (int, float)):
                    mean_dict[key] = np.mean(values)
                elif isinstance(values[0], np.ndarray):
                    # Ensure all arrays have same shape
                    if all(arr.shape == values[0].shape for arr in values):
                        mean_dict[key] = np.mean(np.stack(values, axis=0), axis=0)
    
    def _average_list_values(self, mean_list, components, attr_name):
        """Average list values across ensemble members (for numerical lists)"""
        if not isinstance(mean_list, list) or not mean_list:
            return
        
        # Only process lists of numerical values
        if not all(isinstance(item, (int, float, np.ndarray)) for item in mean_list):
            return
        
        # Collect corresponding lists from all components
        lists = []
        for component in components:
            if hasattr(component, attr_name):
                comp_list = getattr(component, attr_name)
                if isinstance(comp_list, list) and len(comp_list) == len(mean_list):
                    lists.append(comp_list)
        
        if not lists:
            return
        
        # Average each position in the list
        for i in range(len(mean_list)):
            values = []
            for comp_list in lists:
                if i < len(comp_list):
                    val = comp_list[i]
                    if isinstance(val, (int, float)):
                        values.append(val)
                    elif isinstance(val, np.ndarray):
                        values.append(val)
            
            if values:
                if isinstance(values[0], (int, float)):
                    mean_list[i] = np.mean(values)
                elif isinstance(values[0], np.ndarray):
                    if all(arr.shape == values[0].shape for arr in values):
                        mean_list[i] = np.mean(np.stack(values, axis=0), axis=0)
    
    def _simple_assimilation(self, earth_state: EarthSystemState, 
                           observations: List[ObservationData]) -> EarthSystemState:
        """Simplified assimilation when ML is not available"""
        updated_state = copy.deepcopy(earth_state)
        
        for obs in observations:
            # Simple nudging toward observations
            nudging_factor = 0.1
            
            if obs.variable_type == 'temperature' and hasattr(updated_state, 'atmosphere'):
                if hasattr(updated_state.atmosphere, 'temperature'):
                    # Simple spatial interpolation and nudging
                    operator_value = self._temperature_operator(updated_state, obs.location)
                    innovation = obs.value - operator_value
                    
                    # Apply nudging
                    if hasattr(updated_state.atmosphere.temperature, 'shape'):
                        update = innovation * nudging_factor
                        updated_state.atmosphere.temperature += update
        
        return updated_state
    
    def _pressure_operator(self, state: EarthSystemState, location: Tuple[float, float, float]) -> float:
        """Observation operator for pressure"""
        lat, lon, elevation = location
        
        # Find nearest grid point
        lat_idx = np.argmin(np.abs(state.spatial_grid['latitude'] - lat))
        lon_idx = np.argmin(np.abs(state.spatial_grid['longitude'] - lon))
        
        # Surface pressure (simplified)
        surface_pressure = state.components[EarthSystemComponent.ATMOSPHERE][lat_idx, lon_idx, -1, 4]
        
        # Adjust for elevation
        adjusted_pressure = surface_pressure * np.exp(-elevation / 8400)
        
        return adjusted_pressure
    
    def _wind_speed_operator(self, state: EarthSystemState, location: Tuple[float, float, float]) -> float:
        """Observation operator for wind speed"""
        lat, lon, elevation = location
        
        # Find nearest grid point and level
        lat_idx = np.argmin(np.abs(state.spatial_grid['latitude'] - lat))
        lon_idx = np.argmin(np.abs(state.spatial_grid['longitude'] - lon))
        
        elevation_pressure = 101325 * np.exp(-elevation / 8400)
        lev_idx = np.argmin(np.abs(state.spatial_grid['pressure_levels'] - elevation_pressure))
        
        # Extract wind components
        u_wind = state.components[EarthSystemComponent.ATMOSPHERE][lat_idx, lon_idx, lev_idx, 2]
        v_wind = state.components[EarthSystemComponent.ATMOSPHERE][lat_idx, lon_idx, lev_idx, 3]
        
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)
        
        return wind_speed
    
    def _humidity_operator(self, state: EarthSystemState, location: Tuple[float, float, float]) -> float:
        """Observation operator for humidity"""
        lat, lon, elevation = location
        
        lat_idx = np.argmin(np.abs(state.spatial_grid['latitude'] - lat))
        lon_idx = np.argmin(np.abs(state.spatial_grid['longitude'] - lon))
        
        elevation_pressure = 101325 * np.exp(-elevation / 8400)
        lev_idx = np.argmin(np.abs(state.spatial_grid['pressure_levels'] - elevation_pressure))
        
        humidity = state.components[EarthSystemComponent.ATMOSPHERE][lat_idx, lon_idx, lev_idx, 1]
        
        return humidity
    
    def _sst_operator(self, state: EarthSystemState, location: Tuple[float, float, float]) -> float:
        """Observation operator for sea surface temperature"""
        lat, lon, _ = location
        
        lat_idx = np.argmin(np.abs(state.spatial_grid['latitude'] - lat))
        lon_idx = np.argmin(np.abs(state.spatial_grid['longitude'] - lon))
        
        sst = state.components[EarthSystemComponent.HYDROSPHERE][lat_idx, lon_idx, 0]
        
        return sst

class DigitalTwinEarth:
    """Main Digital Twin Earth system"""
    
    def __init__(self, config: ModelConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.earth_dynamics = EarthSystemDynamics(config)
        self.data_assimilation = DataAssimilationSystem(config.data_assimilation_method)
        
        # State management
        self.current_time = datetime.utcnow()
        self.simulation_time = 0.0
        
        # Real-time data streams
        self.observation_queue = asyncio.Queue()
        self.prediction_cache = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'computation_time': [],
            'memory_usage': [],
            'assimilation_accuracy': [],
            'prediction_skill': []
        }
    
    async def initialize(self, initial_conditions: Dict[str, Any] = None):
        """Initialize Digital Twin Earth system"""
        self.logger.info("Initializing Digital Twin Earth system")
        
        # Initialize Earth system state
        initial_state = self.earth_dynamics.initialize_state(initial_conditions)
        
        # Start real-time processing tasks
        asyncio.create_task(self._process_observations())
        asyncio.create_task(self._update_predictions())
        
        self.logger.info("Digital Twin Earth system initialized successfully")
        
        return initial_state
    
    async def step_simulation(self, dt: Optional[float] = None) -> EarthSystemState:
        """Advance simulation one time step"""
        if dt is None:
            dt = self.config.temporal_resolution.total_seconds()
        
        start_time = time.time()
        
        # Advance Earth system dynamics
        new_state = self.earth_dynamics.step_forward(dt)
        
        # Assimilate any pending observations
        observations = await self._collect_recent_observations()
        if observations:
            new_state = self.data_assimilation.assimilate_data(new_state, observations)
        
        # Update simulation time
        self.simulation_time += dt
        self.current_time += timedelta(seconds=dt)
        
        # Record performance metrics
        computation_time = time.time() - start_time
        self.performance_metrics['computation_time'].append(computation_time)
        
        # Limit metrics history
        max_history = 1000
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > max_history:
                self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
        
        return new_state
    
    async def run_simulation(self, duration: timedelta, real_time: bool = False) -> List[EarthSystemState]:
        """Run simulation for specified duration"""
        self.logger.info(f"Starting simulation for {duration}")
        
        dt = self.config.temporal_resolution.total_seconds()
        total_steps = int(duration.total_seconds() / dt)
        
        states = []
        
        for step in range(total_steps):
            step_start = time.time()
            
            # Advance simulation
            state = await self.step_simulation(dt)
            states.append(state)
            
            # Real-time mode: wait to maintain real-time pace
            if real_time:
                elapsed = time.time() - step_start
                sleep_time = max(0, dt - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Progress logging
            if step % max(1, total_steps // 10) == 0:
                progress = (step + 1) / total_steps * 100
                self.logger.info(f"Simulation progress: {progress:.1f}%")
        
        self.logger.info(f"Simulation completed: {len(states)} states generated")
        
        return states
    
    async def add_observation(self, observation: ObservationData):
        """Add real-time observation"""
        await self.observation_queue.put(observation)
        self.logger.debug(f"Queued observation: {observation.variable_type} at {observation.location}")
    
    async def _collect_recent_observations(self) -> List[ObservationData]:
        """Collect recent observations for assimilation"""
        observations = []
        
        # Collect all observations in queue
        while not self.observation_queue.empty():
            try:
                obs = await asyncio.wait_for(self.observation_queue.get(), timeout=0.01)
                
                # Only use recent observations (within assimilation window)
                time_diff = abs((self.current_time - obs.timestamp).total_seconds())
                if time_diff <= 3600:  # 1 hour window
                    observations.append(obs)
                
            except asyncio.TimeoutError:
                break
        
        return observations
    
    async def _process_observations(self):
        """Background task to process incoming observations"""
        while True:
            try:
                # Process observation queue
                if not self.observation_queue.empty():
                    observations = await self._collect_recent_observations()
                    for obs in observations:
                        self.data_assimilation.add_observation(obs)
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error processing observations: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _update_predictions(self):
        """Background task to update prediction cache"""
        while True:
            try:
                # Update short-term predictions
                if self.earth_dynamics.current_state:
                    current_state = self.earth_dynamics.current_state
                    
                    # Generate predictions for next 24 hours
                    prediction_times = [1, 6, 12, 24]  # hours
                    
                    for hours in prediction_times:
                        prediction_key = f"{hours}h"
                        
                        # Simple prediction by advancing model
                        predicted_state = copy.deepcopy(current_state)
                        dt = 3600.0  # 1 hour steps
                        
                        for _ in range(hours):
                            # This would use a separate prediction model instance
                            # For now, just add some uncertainty
                            for component, data in predicted_state.components.items():
                                noise = np.random.normal(1.0, 0.01, data.shape)
                                predicted_state.components[component] *= noise
                        
                        self.prediction_cache[prediction_key] = predicted_state
                
                await asyncio.sleep(1800)  # Update every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Error updating predictions: {e}")
                await asyncio.sleep(3600)  # Wait longer on error
    
    def get_current_state(self) -> Optional[EarthSystemState]:
        """Get current Earth system state"""
        return self.earth_dynamics.current_state
    
    def get_prediction(self, forecast_hours: int) -> Optional[EarthSystemState]:
        """Get prediction for specified hours ahead"""
        prediction_key = f"{forecast_hours}h"
        return self.prediction_cache.get(prediction_key)
    
    def get_state_history(self) -> List[EarthSystemState]:
        """Get historical states"""
        return self.earth_dynamics.state_history
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        metrics = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'current': values[-1] if values else None
                }
        
        # Add system status
        metrics['system_status'] = {
            'simulation_time': self.simulation_time,
            'current_time': self.current_time.isoformat(),
            'queue_size': self.observation_queue.qsize(),
            'prediction_cache_size': len(self.prediction_cache)
        }
        
        return metrics
    
    def export_state(self, filepath: Path, format: str = 'netcdf') -> bool:
        """Export current state to file"""
        if not self.earth_dynamics.current_state:
            return False
        
        try:
            state = self.earth_dynamics.current_state
            
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
            
            elif format == 'netcdf' and GEOSPATIAL_AVAILABLE:
                # Create xarray dataset
                ds = xr.Dataset()
                
                for component, data in state.components.items():
                    ds[component.value] = xr.DataArray(
                        data,
                        dims=['latitude', 'longitude'] + [f'level_{i}' for i in range(data.ndim - 2)] + ['variable'],
                        coords={
                            'latitude': state.spatial_grid['latitude'],
                            'longitude': state.spatial_grid['longitude']
                        }
                    )
                
                ds.to_netcdf(filepath)
            
            else:
                # Fallback to numpy
                np.savez(filepath, **{
                    f"{comp.value}": data for comp, data in state.components.items()
                })
            
            self.logger.info(f"Exported state to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export state: {e}")
            return False

# Example usage and testing
async def example_digital_twin_earth():
    """Example of using Digital Twin Earth system"""
    
    print("=== Digital Twin Earth Example ===")
    
    # Configuration
    config = ModelConfiguration(
        spatial_resolution=2.0,  # 2 degrees
        temporal_resolution=timedelta(hours=1),
        domain_bounds=(-180, 180, -90, 90),
        vertical_levels=20,
        enable_coupling=True,
        uncertainty_quantification=True
    )
    
    # Create Digital Twin Earth
    digital_twin = DigitalTwinEarth(config)
    
    # Initialize system
    initial_state = await digital_twin.initialize()
    
    print(f"Initialized Digital Twin Earth:")
    print(f"  Spatial grid: {len(initial_state.spatial_grid['latitude'])}x{len(initial_state.spatial_grid['longitude'])}")
    print(f"  Vertical levels: {len(initial_state.spatial_grid['pressure_levels'])}")
    print(f"  Components: {list(initial_state.components.keys())}")
    
    # Add some synthetic observations
    observations = [
        ObservationData(
            observation_id="temp_001",
            timestamp=datetime.utcnow(),
            location=(40.0, -74.0, 0.0),  # New York
            variable_type="temperature",
            value=288.15,  # 15°C
            uncertainty=1.0,
            instrument="weather_station"
        ),
        ObservationData(
            observation_id="wind_001",
            timestamp=datetime.utcnow(),
            location=(51.5, -0.1, 10.0),  # London
            variable_type="wind_speed",
            value=5.0,  # m/s
            uncertainty=0.5,
            instrument="anemometer"
        ),
        ObservationData(
            observation_id="sst_001",
            timestamp=datetime.utcnow(),
            location=(35.0, -75.0, 0.0),  # Atlantic Ocean
            variable_type="sea_surface_temperature",
            value=293.15,  # 20°C
            uncertainty=0.5,
            instrument="buoy"
        )
    ]
    
    # Add observations to system
    for obs in observations:
        await digital_twin.add_observation(obs)
    
    print(f"Added {len(observations)} observations")
    
    # Run short simulation
    duration = timedelta(hours=6)
    print(f"Running simulation for {duration}...")
    
    states = await digital_twin.run_simulation(duration, real_time=False)
    
    print(f"Simulation completed: {len(states)} time steps")
    
    # Get final state
    final_state = digital_twin.get_current_state()
    
    if final_state:
        print(f"Final simulation time: {final_state.timestamp}")
        
        # Show some statistics
        for component, data in final_state.components.items():
            print(f"  {component.value}: shape {data.shape}, mean {np.mean(data):.3f}")
    
    # Get predictions
    prediction_6h = digital_twin.get_prediction(6)
    if prediction_6h:
        print(f"6-hour prediction available")
    
    # Performance metrics
    metrics = digital_twin.get_performance_metrics()
    print(f"Performance metrics:")
    for key, stats in metrics.items():
        if isinstance(stats, dict) and 'mean' in stats:
            print(f"  {key}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    # Export final state
    export_path = Path("digital_twin_final_state.npz")
    success = digital_twin.export_state(export_path)
    if success:
        print(f"Exported final state to {export_path}")
    
    print("Digital Twin Earth example completed!")

if __name__ == "__main__":
    print("=== Digital Twin Earth System ===")
    print(f"Available features:")
    print(f"  SciPy: {SCIPY_AVAILABLE}")
    print(f"  Geospatial: {GEOSPATIAL_AVAILABLE}")
    print(f"  Machine Learning: {ML_AVAILABLE}")
    print(f"  Dask: {DASK_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_earth())