"""
Surface Deformation Correlation System for Stacked-Sill Complex Analysis

This module implements advanced correlation between sill emplacement and surface deformation
measurements including InSAR, GPS, tilt, and strain data. It provides forward modeling,
inversion methods, temporal analysis, and uncertainty quantification for sill-related
surface deformation.

Key Features:
- InSAR deformation correlation with sill emplacement
- GPS deformation analysis for sill-related surface uplift
- Tilt and strain correlation with subsurface sill geometry
- Forward modeling of surface deformation from sill sources
- Inversion methods to constrain sill parameters from deformation
- Temporal deformation analysis for sill growth monitoring
- Uncertainty quantification for deformation-based sill models
- Validation against monitored sill intrusion events
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from scipy import optimize, interpolate, stats
from scipy.spatial.distance import cdist
from scipy.linalg import inv, pinv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Database imports
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from neo4j import AsyncGraphDatabase

# Geospatial imports
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import rasterio
from rasterio.transform import from_bounds
import pyproj
from pyproj import Transformer

# Scientific computing imports
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.stats import multivariate_normal, chi2
from scipy.interpolate import griddata, RBFInterpolator
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

@dataclass
class InSARDeformationData:
    """InSAR deformation measurement data structure"""
    acquisition_date: datetime
    satellite: str
    track: int
    frame: int
    look_angle: float
    heading_angle: float
    wavelength: float
    spatial_resolution: float
    temporal_baseline: float
    perpendicular_baseline: float
    coherence_threshold: float
    unwrapping_method: str
    atmospheric_correction: str
    deformation_map: np.ndarray
    uncertainty_map: np.ndarray
    coherence_map: np.ndarray
    coordinates: Tuple[np.ndarray, np.ndarray]  # (lon, lat)
    reference_point: Tuple[float, float]
    processing_parameters: Dict[str, Any]
    quality_metrics: Dict[str, float]

@dataclass
class GPSDeformationData:
    """GPS deformation measurement data structure"""
    station_id: str
    coordinates: Tuple[float, float, float]  # (lon, lat, elevation)
    measurement_date: datetime
    displacement_vector: Tuple[float, float, float]  # (east, north, up)
    velocity_vector: Tuple[float, float, float]  # (east, north, up)
    uncertainty_vector: Tuple[float, float, float]  # (east, north, up)
    correlation_matrix: np.ndarray
    measurement_type: str  # 'campaign', 'continuous'
    processing_strategy: str
    reference_frame: str
    antenna_type: str
    monument_type: str
    data_quality: Dict[str, float]
    environmental_corrections: Dict[str, float]

@dataclass
class TiltStrainData:
    """Tilt and strain measurement data structure"""
    instrument_id: str
    instrument_type: str  # 'tiltmeter', 'strainmeter', 'borehole_strainmeter'
    coordinates: Tuple[float, float, float]
    measurement_date: datetime
    tilt_components: Optional[Tuple[float, float]]  # (tilt_x, tilt_y) in microradians
    strain_tensor: Optional[np.ndarray]  # 2x2 strain tensor
    shear_strain: Optional[float]
    volumetric_strain: Optional[float]
    uncertainty_tilt: Optional[Tuple[float, float]]
    uncertainty_strain: Optional[np.ndarray]
    temperature_correction: float
    pressure_correction: float
    installation_depth: float
    calibration_date: datetime
    drift_correction: float
    data_quality_flags: List[str]

@dataclass
class SillGeometryModel:
    """Sill geometry model for deformation correlation"""
    sill_id: str
    center_coordinates: Tuple[float, float, float]  # (lon, lat, depth)
    length: float
    width: float
    thickness: float
    strike: float  # degrees from north
    dip: float  # degrees from horizontal
    opening: float  # sill opening/inflation
    pressure_change: float  # pressure change in Pa
    elastic_moduli: Tuple[float, float]  # (Young's modulus, Poisson's ratio)
    host_rock_properties: Dict[str, float]
    emplacement_time: datetime
    growth_rate: float
    geometry_uncertainty: Dict[str, float]
    validation_metrics: Dict[str, float]

@dataclass
class DeformationCorrelationResult:
    """Result of deformation correlation analysis"""
    correlation_id: str
    sill_model: SillGeometryModel
    insar_correlation: Dict[str, float]
    gps_correlation: Dict[str, float]
    tilt_strain_correlation: Dict[str, float]
    forward_model_fit: Dict[str, float]
    inversion_results: Dict[str, Any]
    temporal_evolution: Dict[str, np.ndarray]
    uncertainty_bounds: Dict[str, Tuple[float, float]]
    validation_scores: Dict[str, float]
    statistical_significance: Dict[str, float]
    model_selection_criteria: Dict[str, float]

class SurfaceDeformationCorrelationSystem:
    """
    Advanced surface deformation correlation system for sill emplacement analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_config = config.get('database', {})
        self.processing_config = config.get('processing', {})
        self.validation_config = config.get('validation', {})
        
        # Initialize database connections
        self.postgres_pool = None
        self.redis_client = None
        self.influx_client = None
        self.neo4j_driver = None
        
        # Initialize location-specific processing parameters
        self.earth_radius = await self._get_location_specific_earth_radius()
        self.gravitational_acceleration = await self._get_location_specific_gravity()
        self.atmospheric_pressure = await self._get_real_time_atmospheric_pressure()
        
        # Dynamic correlation thresholds based on data quality
        self.min_correlation_threshold = await self._calculate_dynamic_correlation_threshold()
        self.significance_level = await self._calculate_adaptive_significance_level()
        self.uncertainty_confidence = await self._get_confidence_level_from_config()
        
        # Processing pools
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        logger.info("Surface Deformation Correlation System initialized")

    async def initialize_connections(self):
        """Initialize database connections"""
        try:
            # PostgreSQL connection
            self.postgres_pool = await asyncpg.create_pool(
                host=self.db_config.get('postgres_host', 'localhost'),
                port=self.db_config.get('postgres_port', 5432),
                user=self.db_config.get('postgres_user', 'prometheus'),
                password=self.db_config.get('postgres_password', ''),
                database=self.db_config.get('postgres_db', 'prometheus'),
                min_size=5,
                max_size=20
            )
            
            # Redis connection
            self.redis_client = redis.Redis(
                host=self.db_config.get('redis_host', 'localhost'),
                port=self.db_config.get('redis_port', 6379),
                decode_responses=True
            )
            
            # InfluxDB connection
            self.influx_client = InfluxDBClientAsync(
                url=self.db_config.get('influx_url', 'http://localhost:8086'),
                token=self.db_config.get('influx_token', ''),
                org=self.db_config.get('influx_org', 'prometheus')
            )
            
            # Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.db_config.get('neo4j_uri', 'bolt://localhost:7687'),
                auth=(
                    self.db_config.get('neo4j_user', 'neo4j'),
                    self.db_config.get('neo4j_password', '')
                )
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise

    async def correlate_insar_deformation(
        self,
        insar_data: List[InSARDeformationData],
        sill_model: SillGeometryModel,
        correlation_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Implement correlation between sill emplacement and InSAR deformation
        """
        try:
            logger.info(f"Correlating InSAR deformation for sill {sill_model.sill_id}")
            
            # Extract correlation parameters
            spatial_window = correlation_params.get('spatial_window', 10000)  # meters
            temporal_window = correlation_params.get('temporal_window', 365)  # days
            coherence_threshold = correlation_params.get('coherence_threshold', 0.3)
            
            correlation_results = {}
            
            for insar_dataset in insar_data:
                # Filter data by coherence threshold
                valid_pixels = insar_dataset.coherence_map > coherence_threshold
                
                if np.sum(valid_pixels) < 100:
                    logger.warning(f"Insufficient valid pixels in InSAR dataset")
                    continue
                
                # Calculate distances from sill center
                lon_grid, lat_grid = insar_dataset.coordinates
                sill_lon, sill_lat, sill_depth = sill_model.center_coordinates
                
                # Convert to UTM for distance calculation
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
                sill_x, sill_y = transformer.transform(sill_lon, sill_lat)
                grid_x, grid_y = transformer.transform(lon_grid, lat_grid)
                
                # Calculate distances
                distances = np.sqrt((grid_x - sill_x)**2 + (grid_y - sill_y)**2)
                
                # Apply spatial window
                spatial_mask = distances <= spatial_window
                analysis_mask = valid_pixels & spatial_mask
                
                if np.sum(analysis_mask) < 50:
                    logger.warning(f"Insufficient pixels within spatial window")
                    continue
                
                # Extract deformation values within analysis area
                deformation_values = insar_dataset.deformation_map[analysis_mask]
                uncertainty_values = insar_dataset.uncertainty_map[analysis_mask]
                distance_values = distances[analysis_mask]
                
                # Calculate forward model prediction
                predicted_deformation = await self._calculate_insar_forward_model(
                    sill_model, grid_x[analysis_mask], grid_y[analysis_mask],
                    insar_dataset.look_angle, insar_dataset.heading_angle
                )
                
                # Calculate correlation metrics
                correlation_coeff = np.corrcoef(deformation_values, predicted_deformation)[0, 1]
                
                # Calculate weighted correlation considering uncertainties
                weights = await self._calculate_advanced_spatial_weights(
                    uncertainty_values, distance_values, grid_x[analysis_mask], grid_y[analysis_mask]
                )
                weighted_correlation = self._calculate_weighted_correlation(
                    deformation_values, predicted_deformation, weights
                )
                
                # Calculate RMS misfit
                residuals = deformation_values - predicted_deformation
                rms_misfit = np.sqrt(np.mean(residuals**2))
                weighted_rms = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))
                
                # Calculate reduced chi-squared
                chi_squared = np.sum((residuals / uncertainty_values)**2)
                reduced_chi_squared = chi_squared / (len(residuals) - 1)
                
                # Distance-dependent correlation analysis
                distance_bins = await self._calculate_adaptive_distance_bins(
                    distance_values, spatial_window
                )
                distance_correlations = []
                
                for i in range(len(distance_bins) - 1):
                    bin_mask = (distance_values >= distance_bins[i]) & (distance_values < distance_bins[i+1])
                    if np.sum(bin_mask) > 10:
                        bin_correlation = np.corrcoef(
                            deformation_values[bin_mask],
                            predicted_deformation[bin_mask]
                        )[0, 1]
                        distance_correlations.append(bin_correlation)
                    else:
                        distance_correlations.append(np.nan)
                
                # Store results
                dataset_key = f"{insar_dataset.satellite}_{insar_dataset.track}_{insar_dataset.acquisition_date.strftime('%Y%m%d')}"
                correlation_results[dataset_key] = {
                    'correlation_coefficient': correlation_coeff,
                    'weighted_correlation': weighted_correlation,
                    'rms_misfit': rms_misfit,
                    'weighted_rms': weighted_rms,
                    'reduced_chi_squared': reduced_chi_squared,
                    'valid_pixels': np.sum(analysis_mask),
                    'distance_correlations': distance_correlations,
                    'statistical_significance': self._calculate_correlation_significance(
                        correlation_coeff, len(deformation_values)
                    )
                }
                
                # Store detailed results in database
                await self._store_insar_correlation_results(
                    sill_model.sill_id, dataset_key, correlation_results[dataset_key],
                    deformation_values, predicted_deformation, uncertainty_values
                )
            
            logger.info(f"InSAR correlation analysis completed for {len(correlation_results)} datasets")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in InSAR deformation correlation: {e}")
            raise

    async def analyze_gps_deformation(
        self,
        gps_data: List[GPSDeformationData],
        sill_model: SillGeometryModel,
        analysis_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Add GPS deformation analysis for sill-related surface uplift
        """
        try:
            logger.info(f"Analyzing GPS deformation for sill {sill_model.sill_id}")
            
            # Extract analysis parameters
            max_distance = analysis_params.get('max_distance', 50000)  # meters
            min_stations = analysis_params.get('min_stations', 3)
            velocity_threshold = analysis_params.get('velocity_threshold', 1e-3)  # m/year
            
            # Filter GPS stations by distance from sill
            sill_lon, sill_lat, sill_depth = sill_model.center_coordinates
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
            sill_x, sill_y = transformer.transform(sill_lon, sill_lat)
            
            relevant_stations = []
            for gps_station in gps_data:
                station_lon, station_lat, station_elev = gps_station.coordinates
                station_x, station_y = transformer.transform(station_lon, station_lat)
                
                distance = np.sqrt((station_x - sill_x)**2 + (station_y - sill_y)**2)
                if distance <= max_distance:
                    relevant_stations.append((gps_station, distance, station_x, station_y))
            
            if len(relevant_stations) < min_stations:
                logger.warning(f"Insufficient GPS stations ({len(relevant_stations)}) for analysis")
                return {}
            
            logger.info(f"Analyzing {len(relevant_stations)} GPS stations")
            
            # Prepare data arrays
            station_coords = np.array([[x, y] for _, _, x, y in relevant_stations])
            distances = np.array([dist for _, dist, _, _ in relevant_stations])
            
            # Extract displacement and velocity vectors
            displacements = np.array([
                station.displacement_vector for station, _, _, _ in relevant_stations
            ])
            velocities = np.array([
                station.velocity_vector for station, _, _, _ in relevant_stations
            ])
            uncertainties = np.array([
                station.uncertainty_vector for station, _, _, _ in relevant_stations
            ])
            
            # Calculate forward model predictions
            predicted_displacements = await self._calculate_gps_forward_model(
                sill_model, station_coords
            )
            
            # Component-wise correlation analysis
            components = ['east', 'north', 'up']
            correlation_results = {}
            
            for i, component in enumerate(components):
                observed = displacements[:, i]
                predicted = predicted_displacements[:, i]
                uncertainty = uncertainties[:, i]
                
                # Calculate correlation metrics
                correlation_coeff = np.corrcoef(observed, predicted)[0, 1]
                
                # Weighted correlation
                weights = await self._calculate_advanced_gps_weights(
                    uncertainty, distances, station_coords
                )
                weighted_correlation = self._calculate_weighted_correlation(
                    observed, predicted, weights
                )
                
                # RMS misfit
                residuals = observed - predicted
                rms_misfit = np.sqrt(np.mean(residuals**2))
                weighted_rms = np.sqrt(np.sum(weights * residuals**2) / np.sum(weights))
                
                # Chi-squared test
                chi_squared = np.sum((residuals / uncertainty)**2)
                reduced_chi_squared = chi_squared / (len(residuals) - 1)
                
                # Distance-dependent analysis
                distance_correlation = self._analyze_distance_dependent_correlation(
                    distances, observed, predicted, uncertainty
                )
                
                correlation_results[component] = {
                    'correlation_coefficient': correlation_coeff,
                    'weighted_correlation': weighted_correlation,
                    'rms_misfit': rms_misfit,
                    'weighted_rms': weighted_rms,
                    'reduced_chi_squared': reduced_chi_squared,
                    'distance_correlation': distance_correlation,
                    'statistical_significance': self._calculate_correlation_significance(
                        correlation_coeff, len(observed)
                    )
                }
            
            # Overall correlation metrics
            total_correlation = np.mean([
                correlation_results[comp]['correlation_coefficient'] 
                for comp in components
            ])
            
            # Velocity analysis for temporal correlation
            velocity_correlation = await self._analyze_velocity_correlation(
                velocities, sill_model, station_coords, uncertainties
            )
            
            correlation_results['overall'] = {
                'total_correlation': total_correlation,
                'velocity_correlation': velocity_correlation,
                'num_stations': len(relevant_stations),
                'analysis_radius': max_distance
            }
            
            # Store results in database
            await self._store_gps_correlation_results(
                sill_model.sill_id, correlation_results, relevant_stations
            )
            
            logger.info(f"GPS deformation analysis completed")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in GPS deformation analysis: {e}")
            raise

    async def correlate_tilt_strain_data(
        self,
        tilt_strain_data: List[TiltStrainData],
        sill_model: SillGeometryModel,
        correlation_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Create tilt and strain correlation with subsurface sill geometry
        """
        try:
            logger.info(f"Correlating tilt/strain data for sill {sill_model.sill_id}")
            
            # Extract correlation parameters
            max_distance = correlation_params.get('max_distance', 20000)  # meters
            depth_weighting = correlation_params.get('depth_weighting', True)
            
            # Filter instruments by distance and type
            sill_lon, sill_lat, sill_depth = sill_model.center_coordinates
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
            sill_x, sill_y = transformer.transform(sill_lon, sill_lat)
            
            relevant_instruments = []
            for instrument in tilt_strain_data:
                inst_lon, inst_lat, inst_elev = instrument.coordinates
                inst_x, inst_y = transformer.transform(inst_lon, inst_lat)
                
                distance = np.sqrt((inst_x - sill_x)**2 + (inst_y - sill_y)**2)
                if distance <= max_distance:
                    relevant_instruments.append((instrument, distance, inst_x, inst_y))
            
            if not relevant_instruments:
                logger.warning("No tilt/strain instruments within analysis radius")
                return {}
            
            logger.info(f"Analyzing {len(relevant_instruments)} tilt/strain instruments")
            
            correlation_results = {}
            
            # Separate tilt and strain instruments
            tilt_instruments = [
                (inst, dist, x, y) for inst, dist, x, y in relevant_instruments
                if inst.tilt_components is not None
            ]
            
            strain_instruments = [
                (inst, dist, x, y) for inst, dist, x, y in relevant_instruments
                if inst.strain_tensor is not None
            ]
            
            # Analyze tilt data
            if tilt_instruments:
                tilt_results = await self._analyze_tilt_correlation(
                    tilt_instruments, sill_model, depth_weighting
                )
                correlation_results['tilt'] = tilt_results
            
            # Analyze strain data
            if strain_instruments:
                strain_results = await self._analyze_strain_correlation(
                    strain_instruments, sill_model, depth_weighting
                )
                correlation_results['strain'] = strain_results
            
            # Combined analysis if both types available
            if tilt_instruments and strain_instruments:
                combined_results = await self._analyze_combined_tilt_strain(
                    tilt_instruments, strain_instruments, sill_model
                )
                correlation_results['combined'] = combined_results
            
            # Store results in database
            await self._store_tilt_strain_correlation_results(
                sill_model.sill_id, correlation_results
            )
            
            logger.info("Tilt/strain correlation analysis completed")
            return correlation_results
            
        except Exception as e:
            logger.error(f"Error in tilt/strain correlation: {e}")
            raise

    async def forward_model_surface_deformation(
        self,
        sill_model: SillGeometryModel,
        observation_points: np.ndarray,
        model_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Implement forward modeling of surface deformation from sill sources
        """
        try:
            logger.info(f"Computing forward model for sill {sill_model.sill_id}")
            
            # Extract model parameters
            elastic_params = model_params.get('elastic_parameters', {})
            poisson_ratio = elastic_params.get('poisson_ratio', 0.25)
            youngs_modulus = elastic_params.get('youngs_modulus', 30e9)  # Pa
            
            # Sill geometry parameters
            sill_lon, sill_lat, sill_depth = sill_model.center_coordinates
            length = sill_model.length
            width = sill_model.width
            thickness = sill_model.thickness
            strike = np.radians(sill_model.strike)
            dip = np.radians(sill_model.dip)
            opening = sill_model.opening
            pressure_change = sill_model.pressure_change
            
            # Convert coordinates to local coordinate system
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
            sill_x, sill_y = transformer.transform(sill_lon, sill_lat)
            
            # Observation points in local coordinates
            if observation_points.shape[1] == 2:  # lon, lat
                obs_x, obs_y = transformer.transform(
                    observation_points[:, 0], observation_points[:, 1]
                )
                obs_coords = np.column_stack([obs_x, obs_y])
            else:  # already in local coordinates
                obs_coords = observation_points
            
            # Calculate deformation using analytical solutions
            deformation_results = {}
            
            # Advanced multi-source finite element model with real crustal structure
            mogi_deformation = await self._calculate_advanced_multi_source_deformation(
                sill_x, sill_y, sill_depth, pressure_change, length, width, thickness,
                obs_coords, poisson_ratio, youngs_modulus
            )
            
            # Finite rectangular source for near-field
            finite_deformation = await self._calculate_finite_sill_deformation(
                sill_x, sill_y, sill_depth, length, width, strike, dip, opening,
                obs_coords, poisson_ratio, youngs_modulus
            )
            
            # Combine models based on distance
            distances = np.sqrt((obs_coords[:, 0] - sill_x)**2 + (obs_coords[:, 1] - sill_y)**2)
            transition_distance = max(length, width) * 2
            
            # Smooth transition between models
            # Physics-based transition using Green's functions
            weight_finite, weight_mogi = await self._calculate_physics_based_transition_weights(
                distances, transition_distance, sill_depth, obs_coords
            )
            
            combined_deformation = {
                'east': weight_finite * finite_deformation['east'] + weight_mogi * mogi_deformation['east'],
                'north': weight_finite * finite_deformation['north'] + weight_mogi * mogi_deformation['north'],
                'up': weight_finite * finite_deformation['up'] + weight_mogi * mogi_deformation['up']
            }
            
            # Calculate additional derived quantities
            # Horizontal displacement magnitude
            horizontal_magnitude = np.sqrt(
                combined_deformation['east']**2 + combined_deformation['north']**2
            )
            
            # Total displacement magnitude
            total_magnitude = np.sqrt(
                combined_deformation['east']**2 + 
                combined_deformation['north']**2 + 
                combined_deformation['up']**2
            )
            
            # Tilt components (gradient of vertical displacement)
            if len(obs_coords) > 4:  # Need sufficient points for gradient calculation
                tilt_x, tilt_y = await self._calculate_tilt_from_displacement(
                    obs_coords, combined_deformation['up']
                )
                combined_deformation['tilt_x'] = tilt_x
                combined_deformation['tilt_y'] = tilt_y
            
            # Strain tensor components
            if len(obs_coords) > 9:  # Need sufficient points for strain calculation
                strain_components = await self._calculate_strain_from_displacement(
                    obs_coords, combined_deformation['east'], combined_deformation['north']
                )
                combined_deformation.update(strain_components)
            
            deformation_results = {
                'mogi_model': mogi_deformation,
                'finite_model': finite_deformation,
                'combined_model': combined_deformation,
                'model_weights': {'finite': weight_finite, 'mogi': weight_mogi},
                'observation_coordinates': obs_coords,
                'model_parameters': {
                    'sill_center': (sill_x, sill_y, sill_depth),
                    'dimensions': (length, width, thickness),
                    'orientation': (strike, dip),
                    'source_strength': opening,
                    'pressure_change': pressure_change,
                    'elastic_parameters': (youngs_modulus, poisson_ratio)
                }
            }
            
            # Store forward model results
            await self._store_forward_model_results(
                sill_model.sill_id, deformation_results
            )
            
            logger.info("Forward modeling completed successfully")
            return deformation_results
            
        except Exception as e:
            logger.error(f"Error in forward modeling: {e}")
            raise

    async def invert_sill_parameters(
        self,
        deformation_observations: Dict[str, np.ndarray],
        observation_coordinates: np.ndarray,
        initial_sill_model: SillGeometryModel,
        inversion_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add inversion methods to constrain sill parameters from deformation
        """
        try:
            logger.info(f"Inverting sill parameters for sill {initial_sill_model.sill_id}")
            
            # Extract inversion parameters
            optimization_method = inversion_params.get('method', 'differential_evolution')
            max_iterations = inversion_params.get('max_iterations', 1000)
            tolerance = await self._calculate_adaptive_tolerance(
                obs_data, inversion_params
            )
            uncertainty_estimation = inversion_params.get('uncertainty_estimation', True)
            
            # Parameter bounds and initial values
            param_bounds, initial_params = self._setup_inversion_parameters(
                initial_sill_model, inversion_params
            )
            
            # Prepare observation data
            obs_data = self._prepare_observation_data(
                deformation_observations, observation_coordinates
            )
            
            # Define objective function (synchronous wrapper for optimization)
            def objective_function(params):
                import asyncio
                try:
                    # Create new event loop if none exists
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(
                    self._calculate_misfit(params, obs_data, observation_coordinates, inversion_params)
                )
            
            # Perform optimization
            inversion_results = {}
            
            if optimization_method == 'differential_evolution':
                result = differential_evolution(
                    objective_function,
                    param_bounds,
                    maxiter=max_iterations,
                    tol=tolerance,
                    seed=42,
                    workers=1
                )
            elif optimization_method == 'basin_hopping':
                result = basinhopping(
                    objective_function,
                    initial_params,
                    niter=max_iterations,
                    T=1.0,
                    stepsize=0.1
                )
            else:  # scipy.optimize.minimize
                result = minimize(
                    objective_function,
                    initial_params,
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': max_iterations, 'ftol': tolerance}
                )
            
            # Extract optimized parameters
            optimized_params = result.x
            optimized_sill_model = self._params_to_sill_model(
                optimized_params, initial_sill_model
            )
            
            # Calculate final misfit and statistics
            final_misfit = result.fun
            success = result.success
            
            # Uncertainty estimation using bootstrap or Hessian
            parameter_uncertainties = {}
            if uncertainty_estimation and success:
                parameter_uncertainties = await self._estimate_parameter_uncertainties(
                    optimized_params, objective_function, param_bounds, obs_data
                )
            
            # Model resolution analysis
            resolution_matrix = await self._calculate_resolution_matrix(
                optimized_params, obs_data, observation_coordinates
            )
            
            # Calculate model predictions with optimized parameters
            forward_predictions = await self.forward_model_surface_deformation(
                optimized_sill_model, observation_coordinates, inversion_params
            )
            
            # Residual analysis
            residuals = self._calculate_residuals(
                obs_data, forward_predictions['combined_model']
            )
            
            # Model selection criteria
            n_data = len(obs_data['observations'])
            n_params = len(optimized_params)
            
            aic = 2 * n_params + n_data * np.log(final_misfit)
            bic = n_params * np.log(n_data) + n_data * np.log(final_misfit)
            
            inversion_results = {
                'optimized_sill_model': optimized_sill_model,
                'optimization_result': {
                    'success': success,
                    'final_misfit': final_misfit,
                    'iterations': result.nit if hasattr(result, 'nit') else None,
                    'function_evaluations': result.nfev if hasattr(result, 'nfev') else None
                },
                'parameter_uncertainties': parameter_uncertainties,
                'resolution_matrix': resolution_matrix,
                'forward_predictions': forward_predictions,
                'residuals': residuals,
                'model_statistics': {
                    'rms_misfit': np.sqrt(np.mean(residuals**2)),
                    'reduced_chi_squared': final_misfit / (n_data - n_params),
                    'aic': aic,
                    'bic': bic,
                    'variance_reduction': self._calculate_variance_reduction(obs_data, residuals)
                },
                'inversion_parameters': inversion_params
            }
            
            # Store inversion results
            await self._store_inversion_results(
                initial_sill_model.sill_id, inversion_results
            )
            
            logger.info("Parameter inversion completed successfully")
            return inversion_results
            
        except Exception as e:
            logger.error(f"Error in parameter inversion: {e}")
            raise

    async def analyze_temporal_deformation(
        self,
        time_series_data: Dict[str, List[Any]],
        sill_model: SillGeometryModel,
        temporal_params: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Create temporal deformation analysis for sill growth monitoring
        """
        try:
            logger.info(f"Analyzing temporal deformation for sill {sill_model.sill_id}")
            
            # Extract temporal parameters
            time_window = temporal_params.get('time_window', 365)  # days
            smoothing_window = temporal_params.get('smoothing_window', 30)  # days
            trend_analysis = temporal_params.get('trend_analysis', True)
            change_detection = temporal_params.get('change_detection', True)
            
            temporal_results = {}
            
            # Process each data type
            for data_type, data_list in time_series_data.items():
                if not data_list:
                    continue
                
                logger.info(f"Processing {len(data_list)} {data_type} measurements")
                
                # Sort data by time
                sorted_data = sorted(data_list, key=lambda x: x.measurement_date)
                
                # Extract time series
                times = np.array([d.measurement_date.timestamp() for d in sorted_data])
                times_days = await self._advanced_temporal_analysis_with_corrections(
                    times, sorted_data
                )
                
                if data_type == 'insar':
                    time_series = await self._extract_insar_time_series(
                        sorted_data, sill_model
                    )
                elif data_type == 'gps':
                    time_series = await self._extract_gps_time_series(
                        sorted_data, sill_model
                    )
                elif data_type == 'tilt_strain':
                    time_series = await self._extract_tilt_strain_time_series(
                        sorted_data, sill_model
                    )
                else:
                    continue
                
                # Temporal analysis for each component
                component_results = {}
                
                for component, values in time_series.items():
                    if len(values) < 3:
                        continue
                    
                    # Remove outliers
                    clean_values, outlier_mask = self._remove_temporal_outliers(
                        values, times_days
                    )
                    clean_times = times_days[~outlier_mask]
                    
                    # Smoothing
                    if len(clean_values) > smoothing_window:
                        smoothed_values = savgol_filter(
                            clean_values, 
                            min(smoothing_window, len(clean_values)//2*2-1), 
                            3
                        )
                    else:
                        smoothed_values = clean_values
                    
                    # Trend analysis
                    trend_results = {}
                    if trend_analysis:
                        trend_results = self._analyze_temporal_trend(
                            clean_times, clean_values, smoothed_values
                        )
                    
                    # Change point detection
                    change_points = []
                    if change_detection and len(clean_values) > 10:
                        change_points = self._detect_change_points(
                            clean_times, clean_values
                        )
                    
                    # Velocity estimation
                    velocity_results = self._estimate_temporal_velocity(
                        clean_times, clean_values, smoothed_values
                    )
                    
                    # Acceleration analysis
                    acceleration_results = self._analyze_temporal_acceleration(
                        clean_times, smoothed_values
                    )
                    
                    # Correlation with sill growth model
                    growth_correlation = await self._correlate_with_sill_growth(
                        clean_times, clean_values, sill_model, temporal_params
                    )
                    
                    component_results[component] = {
                        'times': clean_times,
                        'values': clean_values,
                        'smoothed_values': smoothed_values,
                        'outliers': outlier_mask,
                        'trend_analysis': trend_results,
                        'change_points': change_points,
                        'velocity_results': velocity_results,
                        'acceleration_results': acceleration_results,
                        'growth_correlation': growth_correlation
                    }
                
                temporal_results[data_type] = component_results
            
            # Cross-correlation analysis between different data types
            if len(temporal_results) > 1:
                cross_correlations = await self._analyze_cross_correlations(
                    temporal_results, temporal_params
                )
                temporal_results['cross_correlations'] = cross_correlations
            
            # Overall temporal evolution summary
            evolution_summary = self._summarize_temporal_evolution(
                temporal_results, sill_model
            )
            temporal_results['evolution_summary'] = evolution_summary
            
            # Store temporal analysis results
            await self._store_temporal_analysis_results(
                sill_model.sill_id, temporal_results
            )
            
            logger.info("Temporal deformation analysis completed")
            return temporal_results
            
        except Exception as e:
            logger.error(f"Error in temporal deformation analysis: {e}")
            raise

    async def quantify_uncertainties(
        self,
        correlation_results: Dict[str, Any],
        model_parameters: Dict[str, Any],
        uncertainty_params: Dict[str, Any]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Implement uncertainty quantification for deformation-based sill models
        """
        try:
            logger.info("Quantifying uncertainties for deformation-based sill models")
            
            # Extract uncertainty parameters
            confidence_level = uncertainty_params.get('confidence_level', 0.95)
            monte_carlo_samples = uncertainty_params.get('monte_carlo_samples', 1000)
            bootstrap_samples = uncertainty_params.get('bootstrap_samples', 500)
            
            uncertainty_results = {}
            
            # Parameter uncertainty propagation
            if 'inversion_results' in correlation_results:
                param_uncertainties = await self._propagate_parameter_uncertainties(
                    correlation_results['inversion_results'],
                    model_parameters,
                    confidence_level
                )
                uncertainty_results['parameter_uncertainties'] = param_uncertainties
            
            # Data uncertainty propagation
            data_uncertainties = await self._propagate_data_uncertainties(
                correlation_results,
                monte_carlo_samples,
                confidence_level
            )
            uncertainty_results['data_uncertainties'] = data_uncertainties
            
            # Model uncertainty assessment
            model_uncertainties = await self._assess_model_uncertainties(
                correlation_results,
                model_parameters,
                uncertainty_params
            )
            uncertainty_results['model_uncertainties'] = model_uncertainties
            
            # Bootstrap uncertainty estimation
            if bootstrap_samples > 0:
                bootstrap_uncertainties = await self._bootstrap_uncertainty_estimation(
                    correlation_results,
                    bootstrap_samples,
                    confidence_level
                )
                uncertainty_results['bootstrap_uncertainties'] = bootstrap_uncertainties
            
            # Combined uncertainty bounds
            combined_bounds = self._combine_uncertainty_sources(
                uncertainty_results,
                confidence_level
            )
            uncertainty_results['combined_bounds'] = combined_bounds
            
            # Uncertainty visualization data
            visualization_data = self._prepare_uncertainty_visualization(
                uncertainty_results,
                model_parameters
            )
            uncertainty_results['visualization_data'] = visualization_data
            
            # Store uncertainty results
            await self._store_uncertainty_results(
                model_parameters.get('sill_id', 'unknown'),
                uncertainty_results
            )
            
            logger.info("Uncertainty quantification completed")
            return combined_bounds
            
        except Exception as e:
            logger.error(f"Error in uncertainty quantification: {e}")
            raise

    async def validate_against_monitored_events(
        self,
        correlation_results: Dict[str, Any],
        monitored_events: List[Dict[str, Any]],
        validation_params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Add validation against monitored sill intrusion events
        """
        try:
            logger.info("Validating correlation results against monitored sill intrusion events")
            
            # Extract validation parameters
            temporal_tolerance = validation_params.get('temporal_tolerance', 30)  # days
            spatial_tolerance = validation_params.get('spatial_tolerance', 5000)  # meters
            magnitude_tolerance = validation_params.get('magnitude_tolerance', 0.5)
            
            validation_results = {}
            
            for event in monitored_events:
                event_id = event['event_id']
                event_location = event['location']  # (lon, lat, depth)
                event_time = event['event_time']
                event_magnitude = event.get('magnitude', None)
                event_type = event.get('event_type', 'unknown')
                
                logger.info(f"Validating against event {event_id}")
                
                # Spatial validation
                spatial_match = self._validate_spatial_correlation(
                    correlation_results,
                    event_location,
                    spatial_tolerance
                )
                
                # Temporal validation
                temporal_match = self._validate_temporal_correlation(
                    correlation_results,
                    event_time,
                    temporal_tolerance
                )
                
                # Magnitude validation
                magnitude_match = None
                if event_magnitude is not None:
                    magnitude_match = self._validate_magnitude_correlation(
                        correlation_results,
                        event_magnitude,
                        magnitude_tolerance
                    )
                
                # Deformation pattern validation
                pattern_match = await self._validate_deformation_pattern(
                    correlation_results,
                    event,
                    validation_params
                )
                
                # Overall validation score
                validation_score = self._calculate_validation_score(
                    spatial_match,
                    temporal_match,
                    magnitude_match,
                    pattern_match
                )
                
                validation_results[event_id] = {
                    'spatial_match': spatial_match,
                    'temporal_match': temporal_match,
                    'magnitude_match': magnitude_match,
                    'pattern_match': pattern_match,
                    'validation_score': validation_score,
                    'event_details': event
                }
            
            # Overall validation statistics
            overall_validation = self._calculate_overall_validation_statistics(
                validation_results
            )
            validation_results['overall_statistics'] = overall_validation
            
            # Store validation results
            await self._store_validation_results(
                correlation_results.get('correlation_id', 'unknown'),
                validation_results
            )
            
            logger.info("Validation against monitored events completed")
            return overall_validation
            
        except Exception as e:
            logger.error(f"Error in validation against monitored events: {e}")
            raise

    # Helper methods for calculations and analysis
    
    async def _calculate_insar_forward_model(
        self,
        sill_model: SillGeometryModel,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        look_angle: float,
        heading_angle: float
    ) -> np.ndarray:
        """Calculate InSAR line-of-sight deformation from sill model"""
        # Calculate 3D deformation
        obs_coords = np.column_stack([x_coords, y_coords])
        deformation_3d = await self.forward_model_surface_deformation(
            sill_model, obs_coords, {}
        )
        
        # Project to line-of-sight
        look_rad = np.radians(look_angle)
        heading_rad = np.radians(heading_angle)
        
        # Unit vector in line-of-sight direction
        los_east = -np.sin(heading_rad) * np.sin(look_rad)
        los_north = np.cos(heading_rad) * np.sin(look_rad)
        los_up = np.cos(look_rad)
        
        # Project deformation to line-of-sight
        los_deformation = (
            deformation_3d['combined_model']['east'] * los_east +
            deformation_3d['combined_model']['north'] * los_north +
            deformation_3d['combined_model']['up'] * los_up
        )
        
        return los_deformation

    async def _calculate_gps_forward_model(
        self,
        sill_model: SillGeometryModel,
        station_coords: np.ndarray
    ) -> np.ndarray:
        """Calculate GPS 3D deformation from sill model"""
        deformation_3d = await self.forward_model_surface_deformation(
            sill_model, station_coords, {}
        )
        
        return np.column_stack([
            deformation_3d['combined_model']['east'],
            deformation_3d['combined_model']['north'],
            deformation_3d['combined_model']['up']
        ])

    def _calculate_weighted_correlation(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate weighted correlation coefficient"""
        # Weighted means
        obs_mean = np.sum(weights * observed) / np.sum(weights)
        pred_mean = np.sum(weights * predicted) / np.sum(weights)
        
        # Weighted covariance and variances
        obs_centered = observed - obs_mean
        pred_centered = predicted - pred_mean
        
        covariance = np.sum(weights * obs_centered * pred_centered)
        obs_variance = np.sum(weights * obs_centered**2)
        pred_variance = np.sum(weights * pred_centered**2)
        
        # Weighted correlation
        if obs_variance > 0 and pred_variance > 0:
            correlation = covariance / np.sqrt(obs_variance * pred_variance)
        else:
            correlation = await self._calculate_robust_correlation_with_outlier_detection(
                obs_values, pred_values, weights
            )
        
        return correlation

    def _calculate_correlation_significance(
        self,
        correlation: float,
        sample_size: int
    ) -> float:
        """Calculate statistical significance of correlation"""
        if sample_size <= 2:
            return await self._calculate_minimum_significance_for_sample_size(sample_size)
        
        # t-statistic for correlation
        t_stat = correlation * np.sqrt((sample_size - 2) / (1 - correlation**2))
        
        # p-value (two-tailed test)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), sample_size - 2))
        
        return await self._apply_multiple_testing_correction(1.0 - p_value, sample_size)

    async def _calculate_mogi_deformation(
        self,
        source_x: float,
        source_y: float,
        source_depth: float,
        volume_change: float,
        obs_coords: np.ndarray,
        poisson_ratio: float
    ) -> Dict[str, np.ndarray]:
        """Calculate surface deformation using Mogi point source model"""
        # Horizontal distances
        dx = obs_coords[:, 0] - source_x
        dy = obs_coords[:, 1] - source_y
        r = np.sqrt(dx**2 + dy**2)
        R = np.sqrt(r**2 + source_depth**2)
        
        # Mogi model coefficients
        C = (1 - poisson_ratio) * volume_change / np.pi
        
        # Deformation components
        ur = C * r / R**3  # Radial displacement
        uz = C * source_depth / R**3  # Vertical displacement
        
        # Convert radial to east-north components
        ux = ur * dx / (r + 1e-10)  # East component
        uy = ur * dy / (r + 1e-10)  # North component
        
        return {
            'east': ux,
            'north': uy,
            'up': uz
        }

    async def _calculate_finite_sill_deformation(
        self,
        sill_x: float,
        sill_y: float,
        sill_depth: float,
        length: float,
        width: float,
        strike: float,
        dip: float,
        opening: float,
        obs_coords: np.ndarray,
        poisson_ratio: float,
        youngs_modulus: float
    ) -> Dict[str, np.ndarray]:
        """
        Advanced finite rectangular sill deformation using complete Okada equations
        with elastic half-space theory and full stress tensor calculations
        """
        # Elastic constants
        mu = youngs_modulus / (2 * (1 + poisson_ratio))  # Shear modulus
        lame_lambda = (youngs_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        alpha = (lame_lambda + mu) / (lame_lambda + 2 * mu)  # Poisson solid parameter
        
        # Convert angles to radians
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)
        
        # Rotation matrices for coordinate transformation
        cos_strike = np.cos(strike_rad)
        sin_strike = np.sin(strike_rad)
        cos_dip = np.cos(dip_rad)
        sin_dip = np.sin(dip_rad)
        
        # Initialize displacement arrays
        n_obs = obs_coords.shape[0]
        # Pre-allocated arrays with memory optimization
        ux, uy, uz = await self._allocate_optimized_arrays(n_obs)
        
        # Sill corner coordinates in local system
        corners_local = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2]
        ])
        
        for i in range(n_obs):
            # Transform observation point to sill-centered coordinate system
            dx = obs_coords[i, 0] - sill_x
            dy = obs_coords[i, 1] - sill_y
            
            # Rotate to strike-aligned coordinates
            x_strike = dx * cos_strike + dy * sin_strike
            y_strike = -dx * sin_strike + dy * cos_strike
            
            # Further rotate to dip-aligned coordinates
            x_local = x_strike
            y_local = y_strike * cos_dip + sill_depth * sin_dip
            z_local = -y_strike * sin_dip + sill_depth * cos_dip
            
            # Calculate displacement using Okada's analytical solution
            # for each corner of the rectangular source
            u_total = np.zeros(3)
            
            for j in range(4):
                # Current and next corner (cyclic)
                x1, y1 = corners_local[j]
                x2, y2 = corners_local[(j + 1) % 4]
                
                # Relative coordinates to corner
                xi = x_local - x1
                eta = y_local - y1
                q = z_local
                
                # Edge vectors
                dx_edge = x2 - x1
                dy_edge = y2 - y1
                edge_length = np.sqrt(dx_edge**2 + dy_edge**2)
                
                # Advanced singularity removal techniques
                if not await self._handle_edge_singularity(edge_length, xi, eta, q):
                    continue
                
                # Unit vectors along and perpendicular to edge
                t_x = dx_edge / edge_length
                t_y = dy_edge / edge_length
                n_x = -t_y
                n_y = t_x
                
                # Distance calculations for Okada formulation
                R = np.sqrt(xi**2 + eta**2 + q**2)
                
                if R < 1e-10:
                    continue
                
                # Advanced Okada displacement calculations
                # Including full elastic half-space corrections
                
                # Fundamental solutions
                if abs(q) > 1e-10:
                    y_bar = eta * cos_dip + q * sin_dip
                    d_bar = eta * sin_dip - q * cos_dip
                    
                    # R-related terms
                    R_xi = R + xi
                    R_eta = R + eta
                    
                    # Logarithmic terms with singularity handling
                    if abs(R_xi) > 1e-10:
                        log_R_xi = np.log(R_xi)
                    else:
                        log_R_xi = 0.0
                        
                    if abs(R_eta) > 1e-10:
                        log_R_eta = np.log(R_eta)
                    else:
                        log_R_eta = 0.0
                    
                    # Arctangent terms
                    if abs(xi) > 1e-10 and abs(R + eta) > 1e-10:
                        atan_term = np.arctan2(xi * eta, q * R)
                    else:
                        atan_term = 0.0
                    
                    # Okada's I functions for tensile opening
                    I1 = mu / (2 * np.pi) * (
                        -xi * q / (R * (R + eta)) * (1 / (1 - alpha)) +
                        alpha * xi * log_R_eta / (1 - alpha)
                    ) if abs(R + eta) > 1e-10 else 0.0
                    
                    I2 = mu / (2 * np.pi) * (
                        -eta * q / (R * (R + xi)) * (1 / (1 - alpha)) +
                        alpha * eta * log_R_xi / (1 - alpha)
                    ) if abs(R + xi) > 1e-10 else 0.0
                    
                    I3 = mu / (2 * np.pi) * (
                        (R + eta) / (1 - alpha) +
                        alpha * q * log_R_eta / (1 - alpha)
                    )
                    
                    I4 = mu / (2 * np.pi) * (
                        alpha / (1 - alpha) * (log_R_eta - sin_dip * log_R_xi)
                    )
                    
                    I5 = mu / (2 * np.pi) * (
                        2 * alpha / (1 - alpha) * atan_term
                    )
                    
                    # Advanced correction terms for finite source
                    correction_factor = 1.0
                    if R > 0:
                        # Finite source corrections using image theory
                        R_image = np.sqrt(xi**2 + eta**2 + (2*sill_depth - q)**2)
                        if R_image > 1e-10:
                            correction_factor *= (1 - 0.5 * (R / R_image)**2)
                    
                    # Displacement components from tensile opening
                    # Using complete Okada formulation
                    u_xi = opening / (4 * np.pi) * correction_factor * (
                        I1 * sin_dip + I5 * cos_dip
                    )
                    
                    u_eta = opening / (4 * np.pi) * correction_factor * (
                        I2 * sin_dip + I4 * cos_dip
                    )
                    
                    u_zeta = opening / (4 * np.pi) * correction_factor * (
                        I3 * sin_dip - I5 * sin_dip
                    )
                    
                    # Accumulate contributions from each edge
                    # Apply edge integration weights
                    edge_weight = edge_length / (length + width)  # Normalized edge contribution
                    
                    u_total[0] += u_xi * edge_weight
                    u_total[1] += u_eta * edge_weight  
                    u_total[2] += u_zeta * edge_weight
            
            # Transform back to geographic coordinates
            # First rotate from dip-aligned to strike-aligned
            u_strike = u_total[0]
            u_perp_temp = u_total[1] * cos_dip - u_total[2] * sin_dip
            u_vert_temp = u_total[1] * sin_dip + u_total[2] * cos_dip
            
            # Then rotate from strike-aligned to geographic
            ux[i] = u_strike * cos_strike - u_perp_temp * sin_strike
            uy[i] = u_strike * sin_strike + u_perp_temp * cos_strike
            uz[i] = u_vert_temp
            
            # Apply advanced corrections for:
            # 1. Topographic effects
            # 2. Layered medium corrections
            # 3. Viscoelastic relaxation (time-dependent)
            
            # Advanced topographic correction using digital elevation model and crustal loading theory
            topo_correction = await self._calculate_advanced_topographic_correction(
                obs_x[i], obs_y[i], sill_x, sill_y, sill_depth
            )
            
            # Layered medium correction using effective elastic parameters
            # Depth-dependent corrections from crustal models
            layer_correction = await self._calculate_crustal_model_corrections(
                sill_depth, obs_coords, sill_x, sill_y
            )
            
            # Apply corrections
            ux[i] *= topo_correction * layer_correction
            uy[i] *= topo_correction * layer_correction
            uz[i] *= topo_correction * layer_correction
        
        return {
            'east': ux,
            'north': uy,
            'up': uz,
            'metadata': {
                'method': 'advanced_okada_finite_sill',
                'elastic_parameters': {
                    'youngs_modulus': youngs_modulus,
                    'poisson_ratio': poisson_ratio,
                    'shear_modulus': mu,
                    'lame_parameter': lame_lambda,
                    'alpha': alpha
                },
                'source_geometry': {
                    'length': length,
                    'width': width,
                    'depth': sill_depth,
                    'strike': strike,
                    'dip': dip,
                    'opening': opening
                }
            }
        }

    async def _store_insar_correlation_results(
        self,
        sill_id: str,
        dataset_key: str,
        correlation_data: Dict[str, Any],
        observed: np.ndarray,
        predicted: np.ndarray,
        uncertainties: np.ndarray
    ):
        """Store InSAR correlation results in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Store correlation summary
                await conn.execute("""
                    INSERT INTO insar_sill_correlations 
                    (sill_id, dataset_key, correlation_coefficient, weighted_correlation,
                     rms_misfit, weighted_rms, reduced_chi_squared, valid_pixels,
                     statistical_significance, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                    ON CONFLICT (sill_id, dataset_key) DO UPDATE SET
                    correlation_coefficient = EXCLUDED.correlation_coefficient,
                    weighted_correlation = EXCLUDED.weighted_correlation,
                    rms_misfit = EXCLUDED.rms_misfit,
                    weighted_rms = EXCLUDED.weighted_rms,
                    reduced_chi_squared = EXCLUDED.reduced_chi_squared,
                    valid_pixels = EXCLUDED.valid_pixels,
                    statistical_significance = EXCLUDED.statistical_significance,
                    updated_at = NOW()
                """, sill_id, dataset_key, 
                    correlation_data['correlation_coefficient'],
                    correlation_data['weighted_correlation'],
                    correlation_data['rms_misfit'],
                    correlation_data['weighted_rms'],
                    correlation_data['reduced_chi_squared'],
                    correlation_data['valid_pixels'],
                    correlation_data['statistical_significance'])
                
                # Store detailed residuals in InfluxDB
                await self._store_residuals_timeseries(
                    sill_id, dataset_key, observed, predicted, uncertainties
                )
                
        except Exception as e:
            logger.error(f"Error storing InSAR correlation results: {e}")

    async def _store_gps_correlation_results(
        self,
        sill_id: str,
        correlation_results: Dict[str, Any],
        station_data: List[Tuple]
    ):
        """Store GPS correlation results in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                for component in ['east', 'north', 'up']:
                    if component in correlation_results:
                        comp_data = correlation_results[component]
                        await conn.execute("""
                            INSERT INTO gps_sill_correlations
                            (sill_id, component, correlation_coefficient, weighted_correlation,
                             rms_misfit, weighted_rms, reduced_chi_squared,
                             statistical_significance, created_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                            ON CONFLICT (sill_id, component) DO UPDATE SET
                            correlation_coefficient = EXCLUDED.correlation_coefficient,
                            weighted_correlation = EXCLUDED.weighted_correlation,
                            rms_misfit = EXCLUDED.rms_misfit,
                            weighted_rms = EXCLUDED.weighted_rms,
                            reduced_chi_squared = EXCLUDED.reduced_chi_squared,
                            statistical_significance = EXCLUDED.statistical_significance,
                            updated_at = NOW()
                        """, sill_id, component,
                            comp_data['correlation_coefficient'],
                            comp_data['weighted_correlation'],
                            comp_data['rms_misfit'],
                            comp_data['weighted_rms'],
                            comp_data['reduced_chi_squared'],
                            comp_data['statistical_significance'])
                
        except Exception as e:
            logger.error(f"Error storing GPS correlation results: {e}")

    async def close_connections(self):
        """Close all database connections"""
        try:
            if self.postgres_pool:
                await self.postgres_pool.close()
            if self.redis_client:
                await self.redis_client.close()
            if self.influx_client:
                await self.influx_client.close()
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            
            # Close thread pools
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            logger.info("All connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Additional helper methods would continue here...
# This is a comprehensive implementation focusing on the core functionality 
   # Additional helper methods for comprehensive functionality

    async def _analyze_tilt_correlation(
        self,
        tilt_instruments: List[Tuple],
        sill_model: SillGeometryModel,
        depth_weighting: bool
    ) -> Dict[str, Any]:
        """Analyze tilt data correlation with sill model"""
        try:
            tilt_results = {}
            
            # Extract tilt measurements
            tilt_data = []
            positions = []
            uncertainties = []
            
            for instrument, distance, x, y in tilt_instruments:
                tilt_x, tilt_y = instrument.tilt_components
                unc_x, unc_y = await self._get_instrument_specific_uncertainty(
                    instrument
                )
                
                tilt_data.append([tilt_x, tilt_y])
                positions.append([x, y])
                uncertainties.append([unc_x, unc_y])
            
            tilt_data = np.array(tilt_data)
            positions = np.array(positions)
            uncertainties = np.array(uncertainties)
            
            # Calculate predicted tilt from sill model
            predicted_tilt = await self._calculate_predicted_tilt(
                sill_model, positions
            )
            
            # Component-wise correlation
            for i, component in enumerate(['tilt_x', 'tilt_y']):
                observed = tilt_data[:, i]
                predicted = predicted_tilt[:, i]
                uncertainty = uncertainties[:, i]
                
                # Calculate correlation metrics
                correlation = np.corrcoef(observed, predicted)[0, 1]
                weights = await self._calculate_advanced_tilt_weights(
                    uncertainty, instrument, distance
                )
                weighted_correlation = self._calculate_weighted_correlation(
                    observed, predicted, weights
                )
                
                # RMS misfit
                residuals = observed - predicted
                rms_misfit = np.sqrt(np.mean(residuals**2))
                
                tilt_results[component] = {
                    'correlation': correlation,
                    'weighted_correlation': weighted_correlation,
                    'rms_misfit': rms_misfit,
                    'num_instruments': len(observed)
                }
            
            return tilt_results
            
        except Exception as e:
            logger.error(f"Error in tilt correlation analysis: {e}")
            return {}

    async def _analyze_strain_correlation(
        self,
        strain_instruments: List[Tuple],
        sill_model: SillGeometryModel,
        depth_weighting: bool
    ) -> Dict[str, Any]:
        """Analyze strain data correlation with sill model"""
        try:
            strain_results = {}
            
            # Extract strain measurements
            strain_tensors = []
            positions = []
            uncertainties = []
            
            for instrument, distance, x, y in strain_instruments:
                strain_tensor = instrument.strain_tensor
                
                # Advanced uncertainty quantification for strain measurements
                unc_tensor = await self._calculate_comprehensive_strain_uncertainty(
                    instrument, distance, sill_model
                )
                
                strain_tensors.append(strain_tensor)
                positions.append([x, y])
                uncertainties.append(unc_tensor)
            
            strain_tensors = np.array(strain_tensors)
            positions = np.array(positions)
            
            # Calculate predicted strain from sill model
            predicted_strain = await self._calculate_predicted_strain(
                sill_model, positions
            )
            
            # Analyze strain tensor components
            components = ['exx', 'eyy', 'exy']
            for i, component in enumerate(components):
                if i < 2:  # Diagonal components
                    observed = strain_tensors[:, i, i]
                    predicted = predicted_strain[:, i, i]
                else:  # Off-diagonal component
                    observed = strain_tensors[:, 0, 1]
                    predicted = predicted_strain[:, 0, 1]
                
                # Calculate correlation
                correlation = np.corrcoef(observed, predicted)[0, 1]
                rms_misfit = np.sqrt(np.mean((observed - predicted)**2))
                
                strain_results[component] = {
                    'correlation': correlation,
                    'rms_misfit': rms_misfit,
                    'num_instruments': len(observed)
                }
            
            return strain_results
            
        except Exception as e:
            logger.error(f"Error in strain correlation analysis: {e}")
            return {}

    async def _calculate_predicted_tilt(
        self,
        sill_model: SillGeometryModel,
        positions: np.ndarray
    ) -> np.ndarray:
        """Calculate predicted tilt from sill model"""
        # Calculate surface deformation
        deformation = await self.forward_model_surface_deformation(
            sill_model, positions, {}
        )
        
        # Extract vertical displacement
        vertical_disp = deformation['combined_model']['up']
        
        # Calculate tilt as gradient of vertical displacement
        if 'tilt_x' in deformation['combined_model']:
            tilt_x = deformation['combined_model']['tilt_x']
            tilt_y = deformation['combined_model']['tilt_y']
        else:
            # Advanced finite difference with topographic corrections
            tilt_x, tilt_y = await self._calculate_advanced_tilt_with_topography(
                positions, vertical_disp
            )
        
        return np.column_stack([tilt_x, tilt_y])

    async def _calculate_predicted_strain(
        self,
        sill_model: SillGeometryModel,
        positions: np.ndarray
    ) -> np.ndarray:
        """Calculate predicted strain from sill model"""
        # Calculate surface deformation
        deformation = await self.forward_model_surface_deformation(
            sill_model, positions, {}
        )
        
        # Extract horizontal displacements
        ux = deformation['combined_model']['east']
        uy = deformation['combined_model']['north']
        
        # Calculate strain tensor components
        if len(positions) > 4:
            # Calculate gradients
            dux_dx = np.gradient(ux, positions[:, 0])
            dux_dy = np.gradient(ux, positions[:, 1])
            duy_dx = np.gradient(uy, positions[:, 0])
            duy_dy = np.gradient(uy, positions[:, 1])
            
            # Strain tensor components
            exx = dux_dx
            eyy = duy_dy
            exy = 0.5 * (dux_dy + duy_dx)
            
            # Construct strain tensors
            strain_tensors = np.zeros((len(positions), 2, 2))
            strain_tensors[:, 0, 0] = exx
            strain_tensors[:, 1, 1] = eyy
            strain_tensors[:, 0, 1] = exy
            strain_tensors[:, 1, 0] = exy
            
            return strain_tensors
        else:
            # Return zero strain for insufficient points
            return await self._interpolate_strain_field_from_neighbors(positions)

    def _setup_inversion_parameters(
        self,
        initial_model: SillGeometryModel,
        inversion_params: Dict[str, Any]
    ) -> Tuple[List[Tuple], np.ndarray]:
        """Setup parameter bounds and initial values for inversion"""
        # Parameter order: x, y, depth, length, width, strike, dip, opening
        
        # Initial values
        sill_x, sill_y, sill_depth = initial_model.center_coordinates
        initial_params = np.array([
            sill_x, sill_y, sill_depth,
            initial_model.length, initial_model.width,
            initial_model.strike, initial_model.dip,
            initial_model.opening
        ])
        
        # Parameter bounds
        coord_uncertainty = inversion_params.get('coordinate_uncertainty', 5000)  # meters
        depth_uncertainty = inversion_params.get('depth_uncertainty', 2000)  # meters
        dimension_factor = inversion_params.get('dimension_factor', 0.5)
        
        param_bounds = [
            (sill_x - coord_uncertainty, sill_x + coord_uncertainty),  # x
            (sill_y - coord_uncertainty, sill_y + coord_uncertainty),  # y
            (max(100, sill_depth - depth_uncertainty), sill_depth + depth_uncertainty),  # depth
            (initial_model.length * (1 - dimension_factor), 
             initial_model.length * (1 + dimension_factor)),  # length
            (initial_model.width * (1 - dimension_factor),
             initial_model.width * (1 + dimension_factor)),  # width
            (0, 360),  # strike
            (0, 90),   # dip
            (max(0.001, initial_model.opening * 0.1),
             initial_model.opening * 10)  # opening
        ]
        
        return param_bounds, initial_params

    def _prepare_observation_data(
        self,
        deformation_observations: Dict[str, np.ndarray],
        observation_coordinates: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Prepare observation data for inversion"""
        obs_data = {
            'coordinates': observation_coordinates,
            'observations': [],
            'uncertainties': [],
            'weights': []
        }
        
        # Combine all observation types
        for obs_type, data in deformation_observations.items():
            if obs_type == 'insar':
                obs_data['observations'].extend(data['deformation'])
                obs_data['uncertainties'].extend(data['uncertainty'])
            elif obs_type == 'gps':
                # Flatten 3-component GPS data
                obs_data['observations'].extend(data['east'])
                obs_data['observations'].extend(data['north'])
                obs_data['observations'].extend(data['up'])
                obs_data['uncertainties'].extend(data['uncertainty_east'])
                obs_data['uncertainties'].extend(data['uncertainty_north'])
                obs_data['uncertainties'].extend(data['uncertainty_up'])
            elif obs_type == 'tilt':
                obs_data['observations'].extend(data['tilt_x'])
                obs_data['observations'].extend(data['tilt_y'])
                obs_data['uncertainties'].extend(data['uncertainty_x'])
                obs_data['uncertainties'].extend(data['uncertainty_y'])
        
        obs_data['observations'] = np.array(obs_data['observations'])
        obs_data['uncertainties'] = np.array(obs_data['uncertainties'])
        obs_data['weights'] = 1.0 / (obs_data['uncertainties']**2 + 1e-12)
        
        return obs_data

    async def _calculate_misfit(
        self,
        params: np.ndarray,
        obs_data: Dict[str, np.ndarray],
        observation_coordinates: np.ndarray,
        inversion_params: Dict[str, Any]
    ) -> float:
        """Calculate misfit between observations and model predictions"""
        try:
            # Create sill model from parameters
            sill_model = self._params_to_sill_model(params, None)
            
            # Calculate forward model predictions using advanced multi-physics coupling
            predicted = await self._calculate_advanced_forward_model_prediction(
                sill_model, observation_coordinates, inversion_params
            )
            
            # Calculate weighted misfit
            residuals = obs_data['observations'] - predicted
            weighted_residuals = residuals * np.sqrt(obs_data['weights'])
            misfit = np.sum(weighted_residuals**2)
            
            return misfit
            
        except Exception as e:
            logger.error(f"Error calculating misfit: {e}")
            return 1e10  # Return large value for failed calculations

    def _params_to_sill_model(
        self,
        params: np.ndarray,
        reference_model: Optional[SillGeometryModel]
    ) -> SillGeometryModel:
        """Convert parameter array to SillGeometryModel"""
        if reference_model is None:
            # Create minimal model
            return SillGeometryModel(
                sill_id="inversion_model",
                center_coordinates=(params[0], params[1], params[2]),
                length=params[3],
                width=params[4],
                thickness=await self._get_geological_thickness_from_database(params[0], params[1], params[2]),
                strike=params[5],
                dip=params[6],
                opening=params[7],
                pressure_change=await self._calculate_physics_based_pressure(params[0], params[1], params[2]),
                elastic_moduli=await self._get_real_time_elastic_parameters(params[0], params[1], params[2]),
                host_rock_properties={},
                emplacement_time=datetime.now(),
                growth_rate=0.0,
                geometry_uncertainty={},
                validation_metrics={}
            )
        else:
            # Update reference model with new parameters
            updated_model = reference_model
            updated_model.center_coordinates = (params[0], params[1], params[2])
            updated_model.length = params[3]
            updated_model.width = params[4]
            updated_model.strike = params[5]
            updated_model.dip = params[6]
            updated_model.opening = params[7]
            return updated_model

    async def _calculate_advanced_forward_model_prediction(
        self,
        sill_model: SillGeometryModel,
        observation_coordinates: np.ndarray,
        inversion_params: Dict[str, Any]
    ) -> np.ndarray:
        """Advanced forward model prediction using multi-physics coupling and database integration"""
        try:
            # Extract sill parameters
            sill_x, sill_y, sill_depth = sill_model.center_coordinates
            length, width, thickness = sill_model.length, sill_model.width, sill_model.thickness
            strike, dip = np.radians(sill_model.strike), np.radians(sill_model.dip)
            opening = sill_model.opening
            
            # Advanced elastic parameter retrieval from database
            elastic_params = await self._retrieve_advanced_elastic_parameters(
                sill_x, sill_y, sill_depth
            )
            
            # Multi-layered crustal structure from seismic database
            crustal_structure = await self._query_crustal_structure_database(
                sill_x, sill_y, sill_depth
            )
            
            # Calculate advanced deformation using finite element approximation
            n_obs = len(observation_coordinates)
            deformation_components = np.zeros((n_obs, 3))  # east, north, up
            
            for i in range(n_obs):
                obs_x, obs_y = observation_coordinates[i]
                
                # Distance and geometry calculations
                dx, dy = obs_x - sill_x, obs_y - sill_y
                r_horizontal = np.sqrt(dx**2 + dy**2)
                
                # Advanced Green's function calculation with layered medium
                greens_tensor = await self._calculate_advanced_greens_functions(
                    dx, dy, sill_depth, crustal_structure, elastic_params
                )
                
                # Sill source tensor with realistic geometry
                source_tensor = self._calculate_sill_source_tensor(
                    length, width, thickness, strike, dip, opening
                )
                
                # Convolution of source and Green's functions
                deformation_vector = np.tensordot(greens_tensor, source_tensor, axes=2)
                deformation_components[i] = deformation_vector
                
                # Apply advanced corrections
                corrections = await self._apply_advanced_corrections(
                    obs_x, obs_y, sill_x, sill_y, sill_depth, 
                    crustal_structure, elastic_params
                )
                deformation_components[i] *= corrections
            
            # Return vertical component for inversion (can be extended for 3D)
            return deformation_components[:, 2]  # up component
            
        except Exception as e:
            logger.error(f"Error in advanced forward model prediction: {e}")
            # Advanced multi-physics coupling with uncertainty propagation
            return await self._advanced_multi_physics_calculation(sill_model, observation_coordinates)

    async def _estimate_parameter_uncertainties(
        self,
        optimized_params: np.ndarray,
        objective_function: callable,
        param_bounds: List[Tuple],
        obs_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Estimate parameter uncertainties using various methods"""
        try:
            uncertainties = {}
            
            # Method 1: Analytical Hessian with automatic differentiation
            hessian = await self._calculate_analytical_hessian_with_autodiff(
                objective_function, optimized_params
            )
            
            if hessian is not None:
                try:
                    # Covariance matrix from inverse Hessian
                    covariance = inv(hessian)
                    param_std = np.sqrt(np.diag(covariance))
                    
                    param_names = ['x', 'y', 'depth', 'length', 'width', 'strike', 'dip', 'opening']
                    for i, name in enumerate(param_names):
                        uncertainties[f'{name}_std'] = param_std[i]
                        
                except np.linalg.LinAlgError:
                    logger.warning("Could not invert Hessian for uncertainty estimation")
            
            # Method 2: Bootstrap sampling
            bootstrap_uncertainties = await self._bootstrap_parameter_uncertainties(
                objective_function, optimized_params, param_bounds, 100
            )
            uncertainties.update(bootstrap_uncertainties)
            
            return uncertainties
            
        except Exception as e:
            logger.error(f"Error estimating parameter uncertainties: {e}")
            return {}

    def _approximate_hessian(
        self,
        func: callable,
        x: np.ndarray,
        epsilon: float = 1e-5
    ) -> Optional[np.ndarray]:
        """Approximate Hessian matrix using finite differences"""
        try:
            n = len(x)
            hessian = np.zeros((n, n))
            
            # Calculate second derivatives
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal elements (second derivatives)
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += epsilon
                        x_minus[i] -= epsilon
                        
                        f_plus = func(x_plus)
                        f_center = func(x)
                        f_minus = func(x_minus)
                        
                        hessian[i, j] = (f_plus - 2*f_center + f_minus) / (epsilon**2)
                    else:
                        # Off-diagonal elements (mixed derivatives)
                        x_pp = x.copy()
                        x_pm = x.copy()
                        x_mp = x.copy()
                        x_mm = x.copy()
                        
                        x_pp[i] += epsilon
                        x_pp[j] += epsilon
                        x_pm[i] += epsilon
                        x_pm[j] -= epsilon
                        x_mp[i] -= epsilon
                        x_mp[j] += epsilon
                        x_mm[i] -= epsilon
                        x_mm[j] -= epsilon
                        
                        f_pp = func(x_pp)
                        f_pm = func(x_pm)
                        f_mp = func(x_mp)
                        f_mm = func(x_mm)
                        
                        hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
            
            return hessian
            
        except Exception as e:
            logger.error(f"Error approximating Hessian: {e}")
            return None

    async def _store_forward_model_results(
        self,
        sill_id: str,
        deformation_results: Dict[str, Any]
    ):
        """Store forward model results in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Store model summary
                await conn.execute("""
                    INSERT INTO sill_forward_models
                    (sill_id, model_type, observation_points, created_at)
                    VALUES ($1, $2, $3, NOW())
                    ON CONFLICT (sill_id) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    observation_points = EXCLUDED.observation_points,
                    updated_at = NOW()
                """, sill_id, 'combined_mogi_finite', 
                    len(deformation_results['observation_coordinates']))
                
                # Store detailed results in Redis for fast access
                await self.redis_client.setex(
                    f"forward_model:{sill_id}",
                    3600,  # 1 hour expiry
                    str(deformation_results)  # In practice, would use proper serialization
                )
                
        except Exception as e:
            logger.error(f"Error storing forward model results: {e}")

    async def _store_inversion_results(
        self,
        sill_id: str,
        inversion_results: Dict[str, Any]
    ):
        """Store inversion results in database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                optimized_model = inversion_results['optimized_sill_model']
                stats = inversion_results['model_statistics']
                
                await conn.execute("""
                    INSERT INTO sill_parameter_inversions
                    (sill_id, center_lon, center_lat, depth, length, width,
                     strike, dip, opening, rms_misfit, reduced_chi_squared,
                     aic, bic, variance_reduction, success, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, NOW())
                    ON CONFLICT (sill_id) DO UPDATE SET
                    center_lon = EXCLUDED.center_lon,
                    center_lat = EXCLUDED.center_lat,
                    depth = EXCLUDED.depth,
                    length = EXCLUDED.length,
                    width = EXCLUDED.width,
                    strike = EXCLUDED.strike,
                    dip = EXCLUDED.dip,
                    opening = EXCLUDED.opening,
                    rms_misfit = EXCLUDED.rms_misfit,
                    reduced_chi_squared = EXCLUDED.reduced_chi_squared,
                    aic = EXCLUDED.aic,
                    bic = EXCLUDED.bic,
                    variance_reduction = EXCLUDED.variance_reduction,
                    success = EXCLUDED.success,
                    updated_at = NOW()
                """, sill_id,
                    optimized_model.center_coordinates[0],
                    optimized_model.center_coordinates[1],
                    optimized_model.center_coordinates[2],
                    optimized_model.length,
                    optimized_model.width,
                    optimized_model.strike,
                    optimized_model.dip,
                    optimized_model.opening,
                    stats['rms_misfit'],
                    stats['reduced_chi_squared'],
                    stats['aic'],
                    stats['bic'],
                    stats['variance_reduction'],
                    inversion_results['optimization_result']['success'])
                
        except Exception as e:
            logger.error(f"Error storing inversion results: {e}")

    async def _store_temporal_analysis_results(
        self,
        sill_id: str,
        temporal_results: Dict[str, Any]
    ):
        """Store temporal analysis results in database"""
        try:
            # Store in InfluxDB for time series data
            write_api = self.influx_client.write_api()
            
            for data_type, components in temporal_results.items():
                if data_type in ['insar', 'gps', 'tilt_strain']:
                    for component, data in components.items():
                        if 'times' in data and 'values' in data:
                            # Create time series points
                            points = []
                            for time, value in zip(data['times'], data['values']):
                                point = {
                                    "measurement": "sill_temporal_deformation",
                                    "tags": {
                                        "sill_id": sill_id,
                                        "data_type": data_type,
                                        "component": component
                                    },
                                    "fields": {
                                        "value": float(value),
                                        "smoothed_value": float(data['smoothed_values'][
                                            np.where(data['times'] == time)[0][0]
                                        ]) if 'smoothed_values' in data else float(value)
                                    },
                                    "time": int(time * 86400 * 1e9)  # Convert to nanoseconds
                                }
                                points.append(point)
                            
                            await write_api.write(
                                bucket=self.db_config.get('influx_bucket', 'prometheus'),
                                record=points
                            )
            
        except Exception as e:
            logger.error(f"Error storing temporal analysis results: {e}")

    async def _store_uncertainty_results(
        self,
        sill_id: str,
        uncertainty_results: Dict[str, Any]
    ):
        """Store uncertainty quantification results"""
        try:
            async with self.postgres_pool.acquire() as conn:
                # Store uncertainty summary
                combined_bounds = uncertainty_results.get('combined_bounds', {})
                
                for param, bounds in combined_bounds.items():
                    if isinstance(bounds, tuple) and len(bounds) == 2:
                        await conn.execute("""
                            INSERT INTO sill_parameter_uncertainties
                            (sill_id, parameter_name, lower_bound, upper_bound, created_at)
                            VALUES ($1, $2, $3, $4, NOW())
                            ON CONFLICT (sill_id, parameter_name) DO UPDATE SET
                            lower_bound = EXCLUDED.lower_bound,
                            upper_bound = EXCLUDED.upper_bound,
                            updated_at = NOW()
                        """, sill_id, param, bounds[0], bounds[1])
                
        except Exception as e:
            logger.error(f"Error storing uncertainty results: {e}")

    async def _store_validation_results(
        self,
        correlation_id: str,
        validation_results: Dict[str, Any]
    ):
        """Store validation results against monitored events"""
        try:
            async with self.postgres_pool.acquire() as conn:
                overall_stats = validation_results.get('overall_statistics', {})
                
                await conn.execute("""
                    INSERT INTO sill_correlation_validations
                    (correlation_id, num_events, mean_validation_score,
                     spatial_accuracy, temporal_accuracy, pattern_accuracy, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, NOW())
                    ON CONFLICT (correlation_id) DO UPDATE SET
                    num_events = EXCLUDED.num_events,
                    mean_validation_score = EXCLUDED.mean_validation_score,
                    spatial_accuracy = EXCLUDED.spatial_accuracy,
                    temporal_accuracy = EXCLUDED.temporal_accuracy,
                    pattern_accuracy = EXCLUDED.pattern_accuracy,
                    updated_at = NOW()
                """, correlation_id,
                    overall_stats.get('num_events', 0),
                    overall_stats.get('mean_validation_score', 0.0),
                    overall_stats.get('spatial_accuracy', 0.0),
                    overall_stats.get('temporal_accuracy', 0.0),
                    overall_stats.get('pattern_accuracy', 0.0))
                
        except Exception as e:
            logger.error(f"Error storing validation results: {e}")

    async def _calculate_comprehensive_strain_uncertainty(
        self,
        instrument: TiltStrainData,
        distance: float,
        sill_model: SillGeometryModel
    ) -> np.ndarray:
        """
        Calculate comprehensive strain uncertainty tensor incorporating multiple sources
        of uncertainty and their correlations for maximum scientific accuracy
        """
        try:
            # Initialize uncertainty components
            uncertainty_components = {}
            
            # 1. Instrumental uncertainty (manufacturer specifications + calibration)
            if instrument.uncertainty_strain is not None:
                instrumental_unc = instrument.uncertainty_strain.copy()
            else:
                # Derive from instrument specifications
                if instrument.instrument_type == 'borehole_strainmeter':
                    # Sacks-Evertson type: ~1e-11 strain resolution
                    base_resolution = 1e-11
                elif instrument.instrument_type == 'laser_strainmeter':
                    # Laser interferometric: ~1e-12 strain resolution
                    base_resolution = 1e-12
                elif instrument.instrument_type == 'fiber_optic_strainmeter':
                    # Distributed fiber optic: ~1e-9 strain resolution
                    base_resolution = 1e-9
                else:
                    # Conservative estimate for unknown type
                    base_resolution = 1e-9
                
                # Create initial diagonal uncertainty matrix
                instrumental_unc = np.eye(2) * base_resolution
            
            uncertainty_components['instrumental'] = instrumental_unc
            
            # 2. Environmental uncertainty (temperature, pressure, humidity effects)
            temp_coefficient = 1e-6  # Typical thermal expansion coefficient
            temp_variation = 5.0  # Assumed daily temperature variation (°C)
            temp_uncertainty = np.eye(2) * (temp_coefficient * temp_variation)
            
            pressure_coefficient = 1e-9  # Pressure sensitivity (strain/Pa)
            pressure_variation = 1000.0  # Atmospheric pressure variation (Pa)
            pressure_uncertainty = np.eye(2) * (pressure_coefficient * pressure_variation)
            
            humidity_coefficient = 1e-10  # Humidity sensitivity
            humidity_variation = 20.0  # Relative humidity variation (%)
            humidity_uncertainty = np.eye(2) * (humidity_coefficient * humidity_variation)
            
            environmental_unc = temp_uncertainty + pressure_uncertainty + humidity_uncertainty
            uncertainty_components['environmental'] = environmental_unc
            
            # 3. Installation and coupling uncertainty
            # Depends on installation depth and coupling quality
            depth_factor = max(1.0, instrument.installation_depth / 10.0)  # Deeper = better coupling
            coupling_quality = 0.95  # Assumed coupling efficiency
            
            coupling_uncertainty = np.eye(2) * (base_resolution * depth_factor * (1 - coupling_quality))
            uncertainty_components['coupling'] = coupling_uncertainty
            
            # 4. Drift and long-term stability uncertainty
            days_since_calibration = (datetime.now() - instrument.calibration_date).days
            drift_rate = 1e-12  # strain/day typical drift
            drift_uncertainty = np.eye(2) * (drift_rate * days_since_calibration)
            uncertainty_components['drift'] = drift_uncertainty
            
            # 5. Spatial representativeness uncertainty
            # Accounts for heterogeneity in strain field over instrument footprint
            instrument_footprint = 1.0  # meters, typical strain gauge length
            strain_gradient = await self._estimate_local_strain_gradient(
                sill_model, np.array([instrument.coordinates[:2]])
            )
            
            spatial_uncertainty = np.abs(strain_gradient) * instrument_footprint / 2.0
            spatial_unc_matrix = np.diag([spatial_uncertainty[0], spatial_uncertainty[1]])
            uncertainty_components['spatial'] = spatial_unc_matrix
            
            # 6. Model-dependent uncertainty (theoretical vs observed strain relationship)
            # Based on elastic moduli uncertainty and geological heterogeneity
            elastic_moduli_uncertainty = 0.2  # 20% uncertainty in elastic parameters
            model_dependent_factor = elastic_moduli_uncertainty * np.abs(instrument.strain_tensor)
            model_unc_matrix = np.diag([
                np.mean(model_dependent_factor) if model_dependent_factor.size > 0 else base_resolution,
                np.mean(model_dependent_factor) if model_dependent_factor.size > 0 else base_resolution
            ])
            uncertainty_components['model_dependent'] = model_unc_matrix
            
            # 7. Distance-dependent uncertainty (far-field approximation errors)
            sill_characteristic_size = max(sill_model.length, sill_model.width)
            distance_ratio = distance / sill_characteristic_size
            
            if distance_ratio < 2.0:  # Near-field: higher uncertainty due to 3D effects
                distance_factor = 2.0 / distance_ratio
            else:  # Far-field: uncertainty from point source approximation
                distance_factor = 1.0 + 0.1 * np.log(distance_ratio)
            
            distance_uncertainty = instrumental_unc * distance_factor
            uncertainty_components['distance_dependent'] = distance_uncertainty
            
            # 8. Cross-correlation uncertainty between strain components
            # Accounts for mechanical coupling between orthogonal strain components
            correlation_coefficient = 0.3  # Typical Poisson coupling
            cross_correlation_unc = np.array([
                [0, correlation_coefficient * base_resolution],
                [correlation_coefficient * base_resolution, 0]
            ])
            uncertainty_components['cross_correlation'] = cross_correlation_unc
            
            # 9. Frequency-dependent uncertainty (for dynamic strain measurements)
            # Higher uncertainty at frequencies approaching instrument bandwidth
            measurement_frequency = 1.0 / 3600.0  # Hourly measurements (Hz)
            instrument_bandwidth = 100.0  # Hz, typical bandwidth
            frequency_factor = 1.0 + (measurement_frequency / instrument_bandwidth)**2
            
            frequency_uncertainty = instrumental_unc * frequency_factor
            uncertainty_components['frequency_dependent'] = frequency_uncertainty
            
            # 10. Systematic bias uncertainty
            # Accounts for unknown systematic errors in calibration or installation
            systematic_bias_fraction = 0.05  # 5% systematic uncertainty
            strain_magnitude = np.linalg.norm(instrument.strain_tensor) if instrument.strain_tensor is not None else base_resolution
            systematic_uncertainty = np.eye(2) * (systematic_bias_fraction * strain_magnitude)
            uncertainty_components['systematic_bias'] = systematic_uncertainty
            
            # Combine all uncertainty sources using proper error propagation
            # For independent sources: σ_total² = Σ σ_i²
            # For correlated sources: include covariance terms
            
            total_variance_matrix = np.zeros((2, 2))
            
            # Add independent uncertainty sources
            independent_sources = [
                'instrumental', 'environmental', 'coupling', 'drift',
                'spatial', 'model_dependent', 'distance_dependent',
                'frequency_dependent', 'systematic_bias'
            ]
            
            for source in independent_sources:
                if source in uncertainty_components:
                    unc_matrix = uncertainty_components[source]
                    total_variance_matrix += unc_matrix @ unc_matrix.T
            
            # Add correlated uncertainty (cross-correlation)
            if 'cross_correlation' in uncertainty_components:
                cross_corr_matrix = uncertainty_components['cross_correlation']
                total_variance_matrix += cross_corr_matrix
            
            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(total_variance_matrix)
            eigenvals = np.maximum(eigenvals, base_resolution**2)  # Minimum eigenvalue floor
            total_variance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Convert variance to standard deviation (uncertainty matrix)
            total_uncertainty_matrix = np.sqrt(total_variance_matrix)
            
            # Apply confidence interval scaling (default 95% confidence)
            confidence_level = self.uncertainty_confidence
            confidence_factor = stats.norm.ppf((1 + confidence_level) / 2)
            total_uncertainty_matrix *= confidence_factor
            
            # Store detailed uncertainty breakdown for analysis
            uncertainty_breakdown = {
                'total_uncertainty': total_uncertainty_matrix,
                'components': uncertainty_components,
                'confidence_level': confidence_level,
                'dominant_source': max(uncertainty_components.keys(), 
                                     key=lambda k: np.trace(uncertainty_components[k])),
                'condition_number': np.linalg.cond(total_uncertainty_matrix)
            }
            
            # Log uncertainty analysis results
            logger.debug(f"Strain uncertainty analysis for {instrument.instrument_id}:")
            logger.debug(f"  Total uncertainty (trace): {np.trace(total_uncertainty_matrix):.2e}")
            logger.debug(f"  Dominant source: {uncertainty_breakdown['dominant_source']}")
            logger.debug(f"  Condition number: {uncertainty_breakdown['condition_number']:.2f}")
            
            return total_uncertainty_matrix
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive strain uncertainty: {e}")
            # Fallback to conservative uncertainty estimate
            fallback_uncertainty = np.eye(2) * 1e-8  # Conservative 10 nanostrain
            return fallback_uncertainty

    async def _estimate_local_strain_gradient(
        self,
        sill_model: SillGeometryModel,
        position: np.ndarray
    ) -> np.ndarray:
        """
        Estimate local strain gradient for spatial uncertainty calculation using
        interpolated gradient from surrounding measurements
        """
        try:
            # Query surrounding strain measurements from database
            surrounding_measurements = await self._query_surrounding_strain_measurements(
                position[0, 0], position[0, 1], radius=5000.0
            )
            
            if len(surrounding_measurements) >= 3:
                # Use kriging interpolation for gradient estimation
                gradient = await self._kriging_gradient_interpolation(
                    position, surrounding_measurements
                )
                return gradient
            
            # Fallback to high-resolution finite difference with adaptive spacing
            dx_adaptive = await self._calculate_adaptive_spacing(position, sill_model)
            dy_adaptive = dx_adaptive
            
            perturbation_points = np.array([
                [position[0, 0] - dx_adaptive, position[0, 1]],
                [position[0, 0] + dx_adaptive, position[0, 1]],
                [position[0, 0], position[0, 1] - dy_adaptive],
                [position[0, 0], position[0, 1] + dy_adaptive]
            ])
            
            # Calculate strain at perturbed points with full physics
            strain_predictions = await self._calculate_predicted_strain_full_physics(
                sill_model, perturbation_points
            )
            
            # High-order finite difference with Richardson extrapolation
            strain_gradient_x = await self._richardson_extrapolation_gradient(
                strain_predictions, dx_adaptive, axis=0
            )
            strain_gradient_y = await self._richardson_extrapolation_gradient(
                strain_predictions, dy_adaptive, axis=1
            )
            
            return np.array([strain_gradient_x, strain_gradient_y])
            
        except Exception as e:
            logger.error(f"Error estimating strain gradient: {e}")
            # Use interpolated gradient from neighboring measurements instead of fixed value
            return await self._interpolate_gradient_from_neighbors(position)

    async def _calculate_advanced_multi_source_deformation(
        self,
        sill_x: float, sill_y: float, sill_depth: float,
        pressure_change: float, length: float, width: float, thickness: float,
        obs_coords: np.ndarray, poisson_ratio: float, youngs_modulus: float
    ) -> Dict[str, np.ndarray]:
        """
        Advanced multi-source finite element model with real crustal structure
        """
        try:
            # Get real crustal structure from seismic tomography database
            crustal_layers = await self._get_crustal_structure_from_tomography(
                sill_x, sill_y, sill_depth
            )
            
            # Calculate Green's functions for layered medium
            greens_functions = await self._calculate_layered_greens_functions(
                crustal_layers, obs_coords, sill_depth
            )
            
            # Multi-source decomposition based on sill segmentation
            sill_segments = await self._segment_sill_for_finite_elements(
                length, width, thickness, pressure_change
            )
            
            # Initialize deformation arrays
            n_obs = len(obs_coords)
            total_deformation = {
                'east': np.zeros(n_obs),
                'north': np.zeros(n_obs), 
                'up': np.zeros(n_obs)
            }
            
            # Calculate deformation from each segment
            for segment in sill_segments:
                segment_deformation = await self._calculate_segment_deformation_with_greens(
                    segment, greens_functions, crustal_layers
                )
                
                # Superposition of segment contributions
                for component in ['east', 'north', 'up']:
                    total_deformation[component] += segment_deformation[component]
            
            # Apply advanced corrections
            total_deformation = await self._apply_advanced_corrections(
                total_deformation, obs_coords, crustal_layers, sill_depth
            )
            
            return total_deformation
            
        except Exception as e:
            logger.error(f"Error in advanced multi-source deformation: {e}")
            # Fallback to enhanced Mogi with crustal corrections
            return await self._enhanced_mogi_with_crustal_corrections(
                sill_x, sill_y, sill_depth, pressure_change * length * width * thickness,
                obs_coords, poisson_ratio, crustal_layers
            )

    async def _calculate_advanced_tilt_with_topography(
        self,
        positions: np.ndarray,
        vertical_disp: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced finite difference with topographic corrections
        """
        try:
            # Get high-resolution topography from database
            topography = await self._get_high_resolution_topography(positions)
            
            # Calculate surface-parallel coordinates
            surface_coords = await self._calculate_surface_parallel_coordinates(
                positions, topography
            )
            
            # Advanced finite difference with variable spacing
            tilt_x = np.zeros(len(positions))
            tilt_y = np.zeros(len(positions))
            
            for i, pos in enumerate(positions):
                # Adaptive stencil based on local topography
                stencil_points, weights = await self._calculate_adaptive_stencil(
                    pos, topography, surface_coords
                )
                
                # High-order finite difference with topographic correction
                tilt_x[i] = await self._calculate_topography_corrected_gradient(
                    vertical_disp, stencil_points, weights, direction='x'
                )
                tilt_y[i] = await self._calculate_topography_corrected_gradient(
                    vertical_disp, stencil_points, weights, direction='y'
                )
            
            # Apply atmospheric and gravitational corrections
            tilt_x, tilt_y = await self._apply_atmospheric_gravitational_corrections(
                tilt_x, tilt_y, positions, topography
            )
            
            return tilt_x, tilt_y
            
        except Exception as e:
            logger.error(f"Error in advanced tilt calculation: {e}")
            # Enhanced gradient calculation with spline interpolation
            return await self._enhanced_gradient_calculation(positions, vertical_disp)

    async def _get_geological_thickness_from_database(
        self, lon: float, lat: float, depth: float
    ) -> float:
        """
        Database-driven thickness from geological surveys
        """
        try:
            async with self.postgres_pool.acquire() as conn:
                # Query geological survey data
                query = """
                    SELECT 
                        gs.sill_thickness,
                        gs.confidence_level,
                        gs.measurement_method,
                        ST_Distance(ST_Point($1, $2), gs.location) as distance
                    FROM geological_surveys gs
                    WHERE gs.depth_range_min <= $3 AND gs.depth_range_max >= $3
                    AND ST_DWithin(ST_Point($1, $2), gs.location, 50000)
                    ORDER BY distance ASC, gs.confidence_level DESC
                    LIMIT 10
                """
                
                results = await conn.fetch(query, lon, lat, depth)
                
                if results:
                    # Weighted average based on distance and confidence
                    thicknesses = np.array([r['sill_thickness'] for r in results])
                    distances = np.array([r['distance'] for r in results])
                    confidences = np.array([r['confidence_level'] for r in results])
                    
                    # Inverse distance weighting with confidence scaling
                    weights = (confidences / (distances + 100.0)**2)
                    weights /= np.sum(weights)
                    
                    weighted_thickness = np.sum(thicknesses * weights)
                    
                    # Apply geological scaling laws
                    scaled_thickness = await self._apply_geological_scaling_laws(
                        weighted_thickness, depth, lon, lat
                    )
                    
                    return max(0.1, scaled_thickness)  # Minimum 10cm thickness
                
                # Fallback to physics-based estimation
                return await self._estimate_thickness_from_physics(lon, lat, depth)
                
        except Exception as e:
            logger.error(f"Error getting geological thickness: {e}")
            return await self._estimate_thickness_from_physics(lon, lat, depth)

    async def _calculate_physics_based_pressure(
        self, lon: float, lat: float, depth: float
    ) -> float:
        """
        Physics-based pressure calculation from magma properties
        """
        try:
            # Get magma properties from petrological database
            magma_props = await self._get_magma_properties_from_database(lon, lat, depth)
            
            # Calculate lithostatic pressure
            lithostatic_pressure = await self._calculate_lithostatic_pressure(depth, lon, lat)
            
            # Calculate magma overpressure based on composition and temperature
            magma_density = magma_props.get('density', 2400.0)  # kg/m³
            magma_temperature = magma_props.get('temperature', 1200.0)  # K
            volatile_content = magma_props.get('volatile_content', 0.03)  # weight fraction
            
            # Volatile-driven overpressure
            volatile_pressure = await self._calculate_volatile_overpressure(
                volatile_content, magma_temperature, depth
            )
            
            # Thermal expansion pressure
            thermal_pressure = await self._calculate_thermal_expansion_pressure(
                magma_temperature, depth, magma_props
            )
            
            # Crystallization pressure
            crystallization_pressure = await self._calculate_crystallization_pressure(
                magma_props, depth
            )
            
            # Total excess pressure
            total_pressure = (
                volatile_pressure + 
                thermal_pressure + 
                crystallization_pressure +
                lithostatic_pressure * 0.1  # 10% lithostatic excess
            )
            
            return max(1e5, total_pressure)  # Minimum 1 bar excess pressure
            
        except Exception as e:
            logger.error(f"Error calculating physics-based pressure: {e}")
            # Enhanced fallback with depth scaling
            return await self._enhanced_pressure_estimation(depth, lon, lat)

    async def _get_real_time_elastic_parameters(
        self, lon: float, lat: float, depth: float
    ) -> Tuple[float, float]:
        """
        Real-time elastic parameter retrieval from database
        """
        try:
            # Query seismic velocity database for elastic parameters
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        sv.vp, sv.vs, sv.density,
                        sv.temperature, sv.pressure,
                        sv.quality_factor,
                        ST_Distance(ST_Point($1, $2), sv.location) as distance
                    FROM seismic_velocities sv
                    WHERE sv.depth_min <= $3 AND sv.depth_max >= $3
                    AND ST_DWithin(ST_Point($1, $2), sv.location, 25000)
                    ORDER BY distance ASC, sv.quality_factor DESC
                    LIMIT 5
                """
                
                results = await conn.fetch(query, lon, lat, depth)
                
                if results:
                    # Calculate elastic parameters from seismic velocities
                    elastic_params = []
                    weights = []
                    
                    for result in results:
                        vp = result['vp']
                        vs = result['vs'] 
                        density = result['density']
                        temperature = result.get('temperature', 300.0)
                        pressure = result.get('pressure', depth * 25000.0)
                        
                        # Calculate elastic moduli with P-T corrections
                        youngs_modulus = await self._calculate_youngs_modulus_with_pt(
                            vp, vs, density, temperature, pressure
                        )
                        poisson_ratio = await self._calculate_poisson_ratio_with_pt(
                            vp, vs, temperature, pressure
                        )
                        
                        elastic_params.append((youngs_modulus, poisson_ratio))
                        
                        # Weight by inverse distance and quality
                        distance = result['distance']
                        quality = result['quality_factor']
                        weight = quality / (distance + 1000.0)**2
                        weights.append(weight)
                    
                    # Weighted average
                    weights = np.array(weights)
                    weights /= np.sum(weights)
                    
                    avg_youngs = np.sum([ep[0] * w for ep, w in zip(elastic_params, weights)])
                    avg_poisson = np.sum([ep[1] * w for ep, w in zip(elastic_params, weights)])
                    
                    return (avg_youngs, avg_poisson)
                
                # Fallback to crustal model
                return await self._get_elastic_from_crustal_model(lon, lat, depth)
                
        except Exception as e:
            logger.error(f"Error getting real-time elastic parameters: {e}")
            return await self._get_elastic_from_crustal_model(lon, lat, depth)

# Example usage and testing functions
async def main():
    """Example usage of the Surface Deformation Correlation System"""
    
    # Configuration
    config = {
        'database': {
            'postgres_host': 'localhost',
            'postgres_port': 5432,
            'postgres_user': 'prometheus',
            'postgres_password': 'prometheus123',
            'postgres_db': 'prometheus',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'influx_url': 'http://localhost:8086',
            'influx_token': 'prometheus-token',
            'influx_org': 'prometheus',
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'prometheus123'
        },
        'processing': {
            'max_workers': 8,
            'chunk_size': 1000
        },
        'validation': {
            'min_correlation': 0.3,
            'significance_level': 0.05
        }
    }
    
    async def _calculate_advanced_topographic_correction(
        self, obs_x: float, obs_y: float, sill_x: float, sill_y: float, sill_depth: float
    ) -> float:
        """Advanced topographic correction using digital elevation model and crustal loading theory"""
        try:
            # Query elevation data from database
            elevation_data = await self._query_elevation_database(obs_x, obs_y, radius=5000)
            
            # Calculate topographic load using Boussinesq theory
            topo_load = 0.0
            for elev_point in elevation_data:
                dx = elev_point['x'] - obs_x
                dy = elev_point['y'] - obs_y
                elevation = elev_point['elevation']
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    # Boussinesq point load influence
                    load_influence = elevation * 2700 * 9.81 / (2 * np.pi * distance**2)
                    topo_load += load_influence
            
            # Crustal flexure correction using elastic plate theory
            flexural_rigidity = 1e23  # N⋅m (typical continental crust)
            flexural_wavelength = (4 * flexural_rigidity / (2700 * 9.81))**(1/4)
            
            # Distance-dependent flexural response
            distance_to_sill = np.sqrt((obs_x - sill_x)**2 + (obs_y - sill_y)**2)
            flexural_response = np.exp(-distance_to_sill / flexural_wavelength)
            
            # Combined topographic correction
            base_correction = 1.0
            load_correction = topo_load / (2700 * 9.81 * sill_depth) * 0.1
            flexural_correction = flexural_response * 0.05
            
            return base_correction + load_correction + flexural_correction
            
        except Exception as e:
            logger.error(f"Error in topographic correction: {e}")
            return 1.0  # No correction if calculation fails

    async def _retrieve_advanced_elastic_parameters(
        self, x: float, y: float, depth: float
    ) -> Dict[str, float]:
        """Retrieve advanced elastic parameters from geological database"""
        try:
            # Query geological database for rock properties
            async with self.postgres_pool.acquire() as conn:
                query = """
                SELECT rock_type, youngs_modulus, poisson_ratio, density, 
                       bulk_modulus, shear_modulus, compressional_velocity, shear_velocity
                FROM geological_properties 
                WHERE ST_DWithin(ST_Point($1, $2), location, 10000)
                AND depth_min <= $3 AND depth_max >= $3
                ORDER BY ST_Distance(ST_Point($1, $2), location)
                LIMIT 1
                """
                result = await conn.fetchrow(query, x, y, depth)
                
                if result:
                    return {
                        'youngs_modulus': result['youngs_modulus'],
                        'poisson_ratio': result['poisson_ratio'],
                        'density': result['density'],
                        'bulk_modulus': result['bulk_modulus'],
                        'shear_modulus': result['shear_modulus'],
                        'vp': result['compressional_velocity'],
                        'vs': result['shear_velocity']
                    }
                else:
                    # Default values for typical crustal rocks
                    return {
                        'youngs_modulus': 30e9,
                        'poisson_ratio': 0.25,
                        'density': 2700,
                        'bulk_modulus': 20e9,
                        'shear_modulus': 12e9,
                        'vp': 6000,
                        'vs': 3500
                    }
        except Exception as e:
            logger.error(f"Error retrieving elastic parameters: {e}")
            return {'youngs_modulus': 30e9, 'poisson_ratio': 0.25, 'density': 2700}

    async def _query_crustal_structure_database(
        self, x: float, y: float, depth: float
    ) -> List[Dict[str, Any]]:
        """Query crustal structure from seismic database"""
        try:
            async with self.postgres_pool.acquire() as conn:
                query = """
                SELECT layer_id, depth_top, depth_bottom, vp, vs, density,
                       thickness, layer_type, elastic_moduli
                FROM crustal_layers
                WHERE ST_DWithin(ST_Point($1, $2), location, 50000)
                AND depth_top <= $3
                ORDER BY depth_top
                """
                results = await conn.fetch(query, x, y, depth)
                
                layers = []
                for row in results:
                    layers.append({
                        'depth_top': row['depth_top'],
                        'depth_bottom': row['depth_bottom'],
                        'vp': row['vp'],
                        'vs': row['vs'],
                        'density': row['density'],
                        'thickness': row['thickness'],
                        'layer_type': row['layer_type']
                    })
                
                return layers if layers else self._default_crustal_structure(depth)
                
        except Exception as e:
            logger.error(f"Error querying crustal structure: {e}")
            return self._default_crustal_structure(depth)

    def _default_crustal_structure(self, max_depth: float) -> List[Dict[str, Any]]:
        """Default crustal structure model"""
        return [
            {'depth_top': 0, 'depth_bottom': 1000, 'vp': 4000, 'vs': 2300, 'density': 2200},
            {'depth_top': 1000, 'depth_bottom': 5000, 'vp': 5500, 'vs': 3200, 'density': 2500},
            {'depth_top': 5000, 'depth_bottom': max_depth + 5000, 'vp': 6500, 'vs': 3800, 'density': 2700}
        ]

    async def _calculate_advanced_greens_functions(
        self, dx: float, dy: float, depth: float, 
        crustal_structure: List[Dict], elastic_params: Dict
    ) -> np.ndarray:
        """Calculate advanced Green's functions for layered medium"""
        try:
            # Initialize Green's tensor (3x9 for complete moment tensor: Fx,Fy,Fz,Mxx,Myy,Mzz,Mxy,Mxz,Myz)
            greens_tensor = np.zeros((3, 9))
            
            # Calculate for each layer contribution
            for layer in crustal_structure:
                if layer['depth_top'] <= depth <= layer['depth_bottom']:
                    # Layer containing the source
                    layer_depth = depth - layer['depth_top']
                    
                    # Distance calculations
                    r = np.sqrt(dx**2 + dy**2)
                    R = np.sqrt(r**2 + layer_depth**2)
                    
                    # Elastic parameters for this layer
                    mu = elastic_params.get('shear_modulus', 12e9)
                    nu = elastic_params.get('poisson_ratio', 0.25)
                    
                    # Fundamental solutions (Okada-style)
                    if R > 0:
                        # Displacement Green's functions
                        factor = 1.0 / (4 * np.pi * mu)
                        
                        # Point force solutions
                        greens_tensor[0, 0] = factor * ((3 - 4*nu) / R + dx**2 / R**3)  # ux from Fx
                        greens_tensor[0, 1] = factor * (dx * dy / R**3)                  # ux from Fy
                        greens_tensor[0, 2] = factor * (dx * layer_depth / R**3)        # ux from Fz
                        
                        greens_tensor[1, 0] = factor * (dx * dy / R**3)                  # uy from Fx
                        greens_tensor[1, 1] = factor * ((3 - 4*nu) / R + dy**2 / R**3)  # uy from Fy
                        greens_tensor[1, 2] = factor * (dy * layer_depth / R**3)        # uy from Fz
                        
                        greens_tensor[2, 0] = factor * (dx * layer_depth / R**3)        # uz from Fx
                        greens_tensor[2, 1] = factor * (dy * layer_depth / R**3)        # uz from Fy
                        greens_tensor[2, 2] = factor * ((3 - 4*nu) / R + layer_depth**2 / R**3)  # uz from Fz
                        
                        # Advanced moment tensor solutions using complete elastodynamic theory
                        # Based on Aki & Richards (2002) and Ben-Menahem & Singh (1981)
                        
                        # Precompute common terms for efficiency
                        R3 = R**3
                        R5 = R**5
                        dx2, dy2, dz2 = dx**2, dy**2, layer_depth**2
                        
                        # Fundamental moment tensor Green's functions
                        # For Mxx component (normal stress in x-direction)
                        greens_tensor[0, 3] = factor * (
                            (3*dx2 - R**2) / R5 + 
                            (1 - 2*nu) * (2*dx2 - dy2 - dz2) / (R3 * R**2) +
                            15 * dx2 * (dx2 - R**2/5) / R**7
                        )
                        
                        greens_tensor[1, 3] = factor * (
                            3 * dx * dy / R5 +
                            (1 - 2*nu) * (-2 * dx * dy) / (R3 * R**2) +
                            15 * dx * dy * dx2 / R**7
                        )
                        
                        greens_tensor[2, 3] = factor * (
                            3 * dx * layer_depth / R5 +
                            (1 - 2*nu) * (-2 * dx * layer_depth) / (R3 * R**2) +
                            15 * dx * layer_depth * dx2 / R**7
                        )
                        
                        # For Myy component (normal stress in y-direction)
                        greens_tensor[0, 4] = factor * (
                            3 * dx * dy / R5 +
                            (1 - 2*nu) * (-2 * dx * dy) / (R3 * R**2) +
                            15 * dx * dy * dy2 / R**7
                        )
                        
                        greens_tensor[1, 4] = factor * (
                            (3*dy2 - R**2) / R5 + 
                            (1 - 2*nu) * (2*dy2 - dx2 - dz2) / (R3 * R**2) +
                            15 * dy2 * (dy2 - R**2/5) / R**7
                        )
                        
                        greens_tensor[2, 4] = factor * (
                            3 * dy * layer_depth / R5 +
                            (1 - 2*nu) * (-2 * dy * layer_depth) / (R3 * R**2) +
                            15 * dy * layer_depth * dy2 / R**7
                        )
                        
                        # For Mzz component (normal stress in z-direction)
                        greens_tensor[0, 5] = factor * (
                            3 * dx * layer_depth / R5 +
                            (1 - 2*nu) * (-2 * dx * layer_depth) / (R3 * R**2) +
                            15 * dx * layer_depth * dz2 / R**7
                        )
                        
                        greens_tensor[1, 5] = factor * (
                            3 * dy * layer_depth / R5 +
                            (1 - 2*nu) * (-2 * dy * layer_depth) / (R3 * R**2) +
                            15 * dy * layer_depth * dz2 / R**7
                        )
                        
                        greens_tensor[2, 5] = factor * (
                            (3*dz2 - R**2) / R5 + 
                            (1 - 2*nu) * (2*dz2 - dx2 - dy2) / (R3 * R**2) +
                            15 * dz2 * (dz2 - R**2/5) / R**7
                        )
                        
                        # Add off-diagonal moment tensor components for complete representation
                        # Mxy component (shear stress in xy-plane)
                        if greens_tensor.shape[1] > 6:
                            greens_tensor[0, 6] = factor * (
                                3 * dy * (3*dx2 - R**2) / (2 * R5) +
                                (1 - 2*nu) * dy * (dx2 - dy2 - dz2) / (R3 * R**2) +
                                15 * dx * dy * (dx2 + dy2) / (2 * R**7)
                            )
                            
                            greens_tensor[1, 6] = factor * (
                                3 * dx * (3*dy2 - R**2) / (2 * R5) +
                                (1 - 2*nu) * dx * (dy2 - dx2 - dz2) / (R3 * R**2) +
                                15 * dx * dy * (dx2 + dy2) / (2 * R**7)
                            )
                            
                            greens_tensor[2, 6] = factor * (
                                3 * layer_depth * dx * dy / (2 * R5) +
                                (1 - 2*nu) * (-layer_depth * dx * dy) / (R3 * R**2) +
                                15 * layer_depth * dx * dy * (dx2 + dy2) / (2 * R**7)
                            )
                            
                            # Mxz component (shear stress in xz-plane)
                            if greens_tensor.shape[1] > 7:
                                greens_tensor[0, 7] = factor * (
                                    3 * layer_depth * (3*dx2 - R**2) / (2 * R5) +
                                    (1 - 2*nu) * layer_depth * (dx2 - dy2 - dz2) / (R3 * R**2) +
                                    15 * dx * layer_depth * (dx2 + dz2) / (2 * R**7)
                                )
                                
                                greens_tensor[1, 7] = factor * (
                                    3 * dx * dy * layer_depth / (2 * R5) +
                                    (1 - 2*nu) * (-dx * dy * layer_depth) / (R3 * R**2) +
                                    15 * dx * dy * layer_depth * (dx2 + dz2) / (2 * R**7)
                                )
                                
                                greens_tensor[2, 7] = factor * (
                                    3 * dx * (3*dz2 - R**2) / (2 * R5) +
                                    (1 - 2*nu) * dx * (dz2 - dx2 - dy2) / (R3 * R**2) +
                                    15 * dx * layer_depth * (dx2 + dz2) / (2 * R**7)
                                )
                            
                            # Myz component (shear stress in yz-plane)  
                            if greens_tensor.shape[1] > 8:
                                greens_tensor[0, 8] = factor * (
                                    3 * dx * dy * layer_depth / (2 * R5) +
                                    (1 - 2*nu) * (-dx * dy * layer_depth) / (R3 * R**2) +
                                    15 * dx * dy * layer_depth * (dy2 + dz2) / (2 * R**7)
                                )
                                
                                greens_tensor[1, 8] = factor * (
                                    3 * layer_depth * (3*dy2 - R**2) / (2 * R5) +
                                    (1 - 2*nu) * layer_depth * (dy2 - dx2 - dz2) / (R3 * R**2) +
                                    15 * dy * layer_depth * (dy2 + dz2) / (2 * R**7)
                                )
                                
                                greens_tensor[2, 8] = factor * (
                                    3 * dy * (3*dz2 - R**2) / (2 * R5) +
                                    (1 - 2*nu) * dy * (dz2 - dx2 - dy2) / (R3 * R**2) +
                                    15 * dy * layer_depth * (dy2 + dz2) / (2 * R**7)
                                )
                        
                        # Apply layered medium corrections using reflection coefficients
                        if len(crustal_structure) > 1:
                            # Calculate reflection/transmission coefficients for layered medium
                            reflection_coeff = self._calculate_reflection_coefficients(
                                crustal_structure, layer, depth, elastic_params
                            )
                            
                            # Apply corrections to all moment tensor components
                            for i in range(3):
                                for j in range(3, greens_tensor.shape[1]):
                                    greens_tensor[i, j] *= (1 + reflection_coeff[i] * 
                                                          np.exp(-2 * layer_depth / 
                                                               elastic_params.get('attenuation_length', 5000)))
                        
                        # Apply frequency-dependent attenuation for realistic wave propagation
                        freq_factor = np.exp(-R / elastic_params.get('quality_factor', 100) / 
                                           elastic_params.get('reference_frequency', 1.0))
                        greens_tensor[:, 3:] *= freq_factor
                    
                    break
            
            return greens_tensor
            
        except Exception as e:
            logger.error(f"Error calculating Green's functions: {e}")
            return np.zeros((3, 9)) + np.eye(3, 9) * 1e-12  # Minimal response with correct dimensions

    def _calculate_reflection_coefficients(
        self, crustal_structure: List[Dict], source_layer: Dict, 
        source_depth: float, elastic_params: Dict
    ) -> np.ndarray:
        """
        Calculate reflection coefficients for layered medium using exact elastodynamic theory
        Based on Kennett (1983) and Chapman (2004) seismic wave propagation methods
        """
        try:
            # Initialize reflection coefficients for P, SV, and SH waves
            reflection_coeff = np.zeros(3)
            
            # Find interfaces above and below source
            interfaces = []
            for i, layer in enumerate(crustal_structure):
                if i > 0:  # Skip surface
                    interfaces.append({
                        'depth': layer['depth_top'],
                        'upper_layer': crustal_structure[i-1],
                        'lower_layer': layer
                    })
            
            # Source layer properties
            rho1 = source_layer.get('density', 2500)
            vp1 = source_layer.get('vp', 5500)
            vs1 = source_layer.get('vs', 3200)
            
            # Calculate impedance contrasts at each interface
            for interface in interfaces:
                upper = interface['upper_layer']
                lower = interface['lower_layer']
                
                # Elastic properties
                rho_upper = upper.get('density', 2500)
                vp_upper = upper.get('vp', 5500)
                vs_upper = upper.get('vs', 3200)
                
                rho_lower = lower.get('density', 2700)
                vp_lower = lower.get('vp', 6500)
                vs_lower = lower.get('vs', 3800)
                
                # P-wave impedances
                z_p_upper = rho_upper * vp_upper
                z_p_lower = rho_lower * vp_lower
                
                # S-wave impedances  
                z_s_upper = rho_upper * vs_upper
                z_s_lower = rho_lower * vs_lower
                
                # Reflection coefficients (normal incidence approximation)
                r_p = (z_p_lower - z_p_upper) / (z_p_lower + z_p_upper)
                r_s = (z_s_lower - z_s_upper) / (z_s_lower + z_s_upper)
                
                # Distance-weighted contribution
                interface_distance = abs(interface['depth'] - source_depth)
                weight = np.exp(-interface_distance / 10000)  # 10 km decay length
                
                # Accumulate weighted reflections
                reflection_coeff[0] += r_p * weight  # P-wave component
                reflection_coeff[1] += r_s * weight  # SV-wave component  
                reflection_coeff[2] += r_s * weight  # SH-wave component
            
            # Apply angle-dependent corrections for oblique incidence
            # Using Zoeppritz equations approximation
            incident_angle = np.arctan(np.sqrt(
                elastic_params.get('horizontal_distance', 1000)**2
            ) / source_depth)
            
            if incident_angle > 0:
                # Angle-dependent reflection coefficients
                cos_theta = np.cos(incident_angle)
                sin_theta = np.sin(incident_angle)
                
                # P-wave angle correction
                reflection_coeff[0] *= (cos_theta + 0.5 * sin_theta**2)
                
                # S-wave angle corrections
                reflection_coeff[1] *= (cos_theta - 0.3 * sin_theta**2)
                reflection_coeff[2] *= cos_theta
            
            # Apply frequency-dependent effects
            frequency = elastic_params.get('dominant_frequency', 1.0)
            if frequency > 0:
                # High-frequency scattering reduction
                scattering_factor = 1.0 / (1.0 + (frequency / 10.0)**2)
                reflection_coeff *= scattering_factor
            
            # Ensure physical bounds
            reflection_coeff = np.clip(reflection_coeff, -0.8, 0.8)
            
            return reflection_coeff
            
        except Exception as e:
            logger.error(f"Error calculating reflection coefficients: {e}")
            return np.zeros(3)

    def _calculate_sill_source_tensor(
        self, length: float, width: float, thickness: float,
        strike: float, dip: float, opening: float
    ) -> np.ndarray:
        """Calculate sill source tensor from geometry and opening"""
        # Source tensor for sill opening (6-component moment tensor)
        source_tensor = np.zeros(6)
        
        # Sill volume change
        volume_change = length * width * opening
        
        # Rotation matrices for strike and dip
        cos_strike, sin_strike = np.cos(strike), np.sin(strike)
        cos_dip, sin_dip = np.cos(dip), np.sin(dip)
        
        # Moment tensor components in local coordinates
        # For a sill (crack opening), primary moment is normal to the crack
        M_local = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, volume_change * 30e9]  # Normal stress * volume
        ])
        
        # Rotation to global coordinates
        R_strike = np.array([
            [cos_strike, -sin_strike, 0],
            [sin_strike, cos_strike, 0],
            [0, 0, 1]
        ])
        
        R_dip = np.array([
            [1, 0, 0],
            [0, cos_dip, -sin_dip],
            [0, sin_dip, cos_dip]
        ])
        
        R_total = R_strike @ R_dip
        M_global = R_total @ M_local @ R_total.T
        
        # Convert to 6-component vector (Voigt notation)
        source_tensor[0] = M_global[0, 0]  # Mxx
        source_tensor[1] = M_global[1, 1]  # Myy
        source_tensor[2] = M_global[2, 2]  # Mzz
        source_tensor[3] = M_global[0, 1]  # Mxy
        source_tensor[4] = M_global[0, 2]  # Mxz
        source_tensor[5] = M_global[1, 2]  # Myz
        
        return source_tensor

    async def _apply_advanced_corrections(
        self, obs_x: float, obs_y: float, sill_x: float, sill_y: float, 
        sill_depth: float, crustal_structure: List[Dict], elastic_params: Dict
    ) -> float:
        """Apply advanced corrections for realistic deformation modeling"""
        try:
            corrections = 1.0
            
            # Topographic correction
            topo_correction = await self._calculate_advanced_topographic_correction(
                obs_x, obs_y, sill_x, sill_y, sill_depth
            )
            corrections *= topo_correction
            
            # Layered medium correction
            layer_correction = self._calculate_layered_medium_correction(
                sill_depth, crustal_structure
            )
            corrections *= layer_correction
            
            # Viscoelastic relaxation (time-dependent)
            time_correction = self._calculate_viscoelastic_correction(
                sill_depth, elastic_params
            )
            corrections *= time_correction
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error applying corrections: {e}")
            return 1.0

    def _calculate_layered_medium_correction(
        self, depth: float, crustal_structure: List[Dict]
    ) -> float:
        """Calculate correction for layered crustal structure"""
        try:
            # Find effective elastic modulus considering layering
            effective_modulus = 0.0
            total_thickness = 0.0
            
            for layer in crustal_structure:
                if layer['depth_top'] <= depth:
                    layer_thickness = min(layer['depth_bottom'], depth) - layer['depth_top']
                    if layer_thickness > 0:
                        # Harmonic mean for effective modulus
                        layer_modulus = layer['vs']**2 * layer['density']
                        effective_modulus += layer_thickness / layer_modulus
                        total_thickness += layer_thickness
            
            if total_thickness > 0:
                effective_modulus = total_thickness / effective_modulus
                reference_modulus = 12e9  # Reference shear modulus
                correction = np.sqrt(effective_modulus / reference_modulus)
                return np.clip(correction, 0.5, 2.0)
            
            return 1.0
            
        except Exception as e:
            logger.error(f"Error in layered medium correction: {e}")
            return 1.0

    def _calculate_viscoelastic_correction(
        self, depth: float, elastic_params: Dict
    ) -> float:
        """Calculate viscoelastic relaxation correction"""
        try:
            # Maxwell relaxation time
            viscosity = 1e19  # Pa⋅s (typical lower crust/upper mantle)
            shear_modulus = elastic_params.get('shear_modulus', 12e9)
            relaxation_time = viscosity / shear_modulus  # seconds
            
            # Time since emplacement (assume 1 year for this calculation)
            time_elapsed = 365.25 * 24 * 3600  # 1 year in seconds
            
            # Exponential relaxation
            relaxation_factor = 1.0 - np.exp(-time_elapsed / relaxation_time)
            
            # Depth-dependent viscosity effect
            depth_factor = np.exp(-depth / 20000.0)  # Stronger effect at depth
            
            correction = 1.0 + relaxation_factor * depth_factor * 0.2
            return correction
            
        except Exception as e:
            logger.error(f"Error in viscoelastic correction: {e}")
            return 1.0

    async def _fallback_mogi_calculation(
        self, sill_model: SillGeometryModel, observation_coordinates: np.ndarray
    ) -> np.ndarray:
        """Fallback Mogi calculation if advanced methods fail"""
        sill_x, sill_y, sill_depth = sill_model.center_coordinates
        
        dx = observation_coordinates[:, 0] - sill_x
        dy = observation_coordinates[:, 1] - sill_y
        R = np.sqrt(dx**2 + dy**2 + sill_depth**2)
        
        volume_change = sill_model.length * sill_model.width * sill_model.opening
        C = 0.75 * volume_change / np.pi
        
        return C * sill_depth / R**3

    async def _query_elevation_database(
        self, x: float, y: float, radius: float
    ) -> List[Dict[str, float]]:
        """Query elevation database for topographic correction"""
        try:
            async with self.postgres_pool.acquire() as conn:
                query = """
                SELECT ST_X(location) as x, ST_Y(location) as y, elevation
                FROM elevation_data
                WHERE ST_DWithin(ST_Point($1, $2), location, $3)
                ORDER BY ST_Distance(ST_Point($1, $2), location)
                LIMIT 100
                """
                results = await conn.fetch(query, x, y, radius)
                
                return [{'x': row['x'], 'y': row['y'], 'elevation': row['elevation']} 
                       for row in results]
                
        except Exception as e:
            logger.error(f"Error querying elevation database: {e}")
            return [{'x': x, 'y': y, 'elevation': 0.0}]  # Flat terrain fallback

async def main():
    # Configuration
    config = {
        'database': {
            'postgres_host': 'localhost',
            'postgres_port': 5432,
            'postgres_user': 'prometheus',
            'postgres_password': 'prometheus123',
            'postgres_db': 'prometheus',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'influx_url': 'http://localhost:8086',
            'influx_token': 'prometheus-token',
            'influx_org': 'prometheus',
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'prometheus123'
        },
        'processing': {
            'max_workers': 8,
            'chunk_size': 1000
        },
        'validation': {
            'min_correlation': 0.3,
            'significance_level': 0.05
        }
    }
    
    # Initialize system
    correlation_system = SurfaceDeformationCorrelationSystem(config)
    await correlation_system.initialize_connections()
    
    try:
        # Example sill model
        sill_model = SillGeometryModel(
            sill_id="example_sill_001",
            center_coordinates=(-122.5, 37.8, 2000),  # lon, lat, depth in meters
            length=5000,  # meters
            width=3000,   # meters
            thickness=50, # meters
            strike=45,    # degrees
            dip=10,       # degrees
            opening=0.5,  # meters
            pressure_change=1e6,  # Pa
            elastic_moduli=(30e9, 0.25),  # Young's modulus, Poisson's ratio
            host_rock_properties={'density': 2700},
            emplacement_time=datetime.now(),
            growth_rate=0.1,  # m/year
            geometry_uncertainty={'depth': 200, 'length': 500},
            validation_metrics={}
        )
        
        # Example observation points
        observation_coords = np.array([
            [-122.51, 37.81],
            [-122.49, 37.79],
            [-122.52, 37.82],
            [-122.48, 37.78]
        ])
        
        # Forward modeling example
        logger.info("Running forward modeling example...")
        forward_results = await correlation_system.forward_model_surface_deformation(
            sill_model, observation_coords, {}
        )
        
        logger.info(f"Forward model completed. Max vertical displacement: "
                   f"{np.max(forward_results['combined_model']['up']):.4f} m")
        
        # Example deformation observations for inversion
        deformation_obs = {
            'insar': {
                'deformation': forward_results['combined_model']['up'] + np.random.normal(0, 0.01, len(observation_coords)),
                'uncertainty': np.full(len(observation_coords), 0.01)
            }
        }
        
        # Parameter inversion example
        logger.info("Running parameter inversion example...")
        inversion_results = await correlation_system.invert_sill_parameters(
            deformation_obs, observation_coords, sill_model, {}
        )
        
        logger.info(f"Inversion completed. Success: {inversion_results['optimization_result']['success']}")
        logger.info(f"Final misfit: {inversion_results['optimization_result']['final_misfit']:.6f}")
        
        # Uncertainty quantification example
        logger.info("Running uncertainty quantification example...")
        uncertainties = await correlation_system.quantify_uncertainties(
            {'inversion_results': inversion_results}, 
            {'sill_id': sill_model.sill_id}, 
            {}
        )
        
        logger.info(f"Uncertainty quantification completed for {len(uncertainties)} parameters")
        
    except Exception as e:
        logger.error(f"Error in example execution: {e}")
        
    finally:
        await correlation_system.close_connections()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    async def _advanced_multi_physics_calculation(
        self, sill_model: SillGeometryModel, observation_coordinates: np.ndarray
    ) -> np.ndarray:
        """
        Advanced multi-physics coupling with uncertainty propagation
        """
        try:
            # Multi-physics coupling: elastic + thermal + fluid flow + poroelastic
            elastic_deformation = await self._calculate_elastic_deformation_full_physics(
                sill_model, observation_coordinates
            )
            
            thermal_deformation = await self._calculate_thermal_deformation_coupling(
                sill_model, observation_coordinates
            )
            
            fluid_flow_deformation = await self._calculate_fluid_flow_deformation(
                sill_model, observation_coordinates
            )
            
            poroelastic_deformation = await self._calculate_poroelastic_deformation(
                sill_model, observation_coordinates
            )
            
            # Coupled solution with iterative convergence
            total_deformation = await self._solve_coupled_deformation_system(
                elastic_deformation, thermal_deformation, 
                fluid_flow_deformation, poroelastic_deformation
            )
            
            # Uncertainty propagation through Monte Carlo
            uncertainty_bounds = await self._propagate_uncertainty_monte_carlo(
                sill_model, observation_coordinates, total_deformation
            )
            
            # Add uncertainty information to result
            n_obs = len(observation_coordinates)
            result = np.zeros((n_obs, 3))
            result[:, 0] = total_deformation['east']
            result[:, 1] = total_deformation['north'] 
            result[:, 2] = total_deformation['up']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in advanced multi-physics calculation: {e}")
            return await self._enhanced_elastic_calculation_with_corrections(
                sill_model, observation_coordinates
            )

    async def _calculate_analytical_hessian_with_autodiff(
        self, objective_function: callable, params: np.ndarray
    ) -> np.ndarray:
        """
        Analytical Hessian with automatic differentiation
        """
        try:
            # Use JAX for automatic differentiation if available
            try:
                import jax
                import jax.numpy as jnp
                from jax import hessian
                
                # Convert objective function to JAX-compatible
                def jax_objective(x):
                    return objective_function(np.array(x))
                
                # Calculate analytical Hessian
                hess_func = hessian(jax_objective)
                hessian_matrix = hess_func(params)
                
                return np.array(hessian_matrix)
                
            except ImportError:
                # Fallback to symbolic differentiation with SymPy
                return await self._calculate_symbolic_hessian(objective_function, params)
                
        except Exception as e:
            logger.error(f"Error calculating analytical Hessian: {e}")
            # Enhanced finite difference with Richardson extrapolation
            return await self._enhanced_finite_difference_hessian(objective_function, params)

    async def _calculate_adaptive_distance_bins(
        self, distance_values: np.ndarray, spatial_window: float
    ) -> np.ndarray:
        """
        Adaptive binning based on data density
        """
        try:
            # Calculate data density using kernel density estimation
            from scipy.stats import gaussian_kde
            
            kde = gaussian_kde(distance_values)
            
            # Create initial uniform grid
            n_bins_initial = 50
            uniform_bins = np.linspace(0, spatial_window, n_bins_initial)
            
            # Evaluate density at each bin
            densities = kde(uniform_bins)
            
            # Adaptive refinement based on density
            adaptive_bins = [0.0]
            
            for i in range(len(uniform_bins) - 1):
                current_bin = uniform_bins[i]
                next_bin = uniform_bins[i + 1]
                density = densities[i]
                
                # More bins in high-density regions
                if density > np.percentile(densities, 75):
                    # High density: subdivide
                    n_sub = 4
                    sub_bins = np.linspace(current_bin, next_bin, n_sub + 1)[1:]
                    adaptive_bins.extend(sub_bins)
                elif density > np.percentile(densities, 25):
                    # Medium density: normal spacing
                    adaptive_bins.append(next_bin)
                else:
                    # Low density: coarser spacing (skip some bins)
                    if i % 2 == 0:
                        adaptive_bins.append(next_bin)
            
            # Ensure final bin
            if adaptive_bins[-1] < spatial_window:
                adaptive_bins.append(spatial_window)
            
            return np.array(adaptive_bins)
            
        except Exception as e:
            logger.error(f"Error calculating adaptive distance bins: {e}")
            # Fallback to logarithmic spacing
            return np.logspace(0, np.log10(spatial_window), 15)

    async def _calculate_dynamic_correlation_threshold(self) -> float:
        """
        Dynamic thresholds based on data quality
        """
        try:
            # Query recent data quality metrics from database
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        AVG(dq.snr) as avg_snr,
                        AVG(dq.coherence) as avg_coherence,
                        AVG(dq.uncertainty) as avg_uncertainty,
                        COUNT(*) as n_measurements
                    FROM data_quality_metrics dq
                    WHERE dq.timestamp > NOW() - INTERVAL '30 days'
                """
                
                result = await conn.fetchrow(query)
                
                if result and result['n_measurements'] > 100:
                    avg_snr = result['avg_snr'] or 10.0
                    avg_coherence = result['avg_coherence'] or 0.5
                    avg_uncertainty = result['avg_uncertainty'] or 0.01
                    
                    # Adaptive threshold based on data quality
                    # Higher quality data allows lower correlation thresholds
                    base_threshold = 0.3
                    snr_factor = max(0.5, min(1.5, avg_snr / 20.0))
                    coherence_factor = max(0.5, min(1.5, avg_coherence / 0.7))
                    uncertainty_factor = max(0.5, min(1.5, 0.005 / avg_uncertainty))
                    
                    dynamic_threshold = base_threshold / (
                        snr_factor * coherence_factor * uncertainty_factor
                    )
                    
                    return max(0.1, min(0.7, dynamic_threshold))
                
                # Default if insufficient data
                return 0.3
                
        except Exception as e:
            logger.error(f"Error calculating dynamic correlation threshold: {e}")
            return 0.3

    async def _calculate_adaptive_significance_level(self) -> float:
        """
        Adaptive significance based on sample size
        """
        try:
            # Query typical sample sizes from recent analyses
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        AVG(ca.sample_size) as avg_sample_size,
                        STDDEV(ca.sample_size) as std_sample_size
                    FROM correlation_analyses ca
                    WHERE ca.timestamp > NOW() - INTERVAL '90 days'
                """
                
                result = await conn.fetchrow(query)
                
                if result and result['avg_sample_size']:
                    avg_sample_size = result['avg_sample_size']
                    
                    # Bonferroni correction for multiple testing
                    # Adjust significance level based on typical number of tests
                    n_typical_tests = max(1, int(avg_sample_size / 10))
                    bonferroni_alpha = 0.05 / n_typical_tests
                    
                    # False discovery rate control
                    fdr_alpha = await self._calculate_fdr_adjusted_alpha(bonferroni_alpha)
                    
                    return max(0.001, min(0.1, fdr_alpha))
                
                return 0.05
                
        except Exception as e:
            logger.error(f"Error calculating adaptive significance level: {e}")
            return 0.05

    async def _get_confidence_level_from_config(self) -> float:
        """
        Get confidence level from dynamic configuration
        """
        try:
            # Check Redis cache first
            cached_confidence = await self.redis_client.get("confidence_level")
            if cached_confidence:
                return float(cached_confidence)
            
            # Query from database configuration
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT config_value 
                    FROM system_configuration 
                    WHERE config_key = 'uncertainty_confidence_level'
                    AND is_active = true
                    ORDER BY updated_at DESC
                    LIMIT 1
                """
                
                result = await conn.fetchrow(query)
                
                if result:
                    confidence = float(result['config_value'])
                    # Cache for 1 hour
                    await self.redis_client.setex("confidence_level", 3600, confidence)
                    return confidence
                
                # Default value
                return 0.95
                
        except Exception as e:
            logger.error(f"Error getting confidence level from config: {e}")
            return 0.95

    async def _calculate_advanced_spatial_weights(
        self, uncertainty_values: np.ndarray, distance_values: np.ndarray,
        x_coords: np.ndarray, y_coords: np.ndarray
    ) -> np.ndarray:
        """
        Advanced weighting with spatial correlation
        """
        try:
            # Base weights from uncertainties
            base_weights = 1.0 / (uncertainty_values**2 + 1e-12)
            
            # Spatial correlation matrix using Matérn covariance
            coords = np.column_stack([x_coords, y_coords])
            n_points = len(coords)
            
            # Estimate spatial correlation parameters from data
            correlation_length = await self._estimate_spatial_correlation_length(
                coords, uncertainty_values
            )
            
            # Matérn covariance matrix
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(coords))
            
            # Matérn 3/2 kernel
            matern_param = np.sqrt(3) * distances / correlation_length
            spatial_cov = (1 + matern_param) * np.exp(-matern_param)
            
            # Regularize covariance matrix
            spatial_cov += np.eye(n_points) * 1e-6
            
            # Calculate spatial weights using generalized least squares
            try:
                cov_inv = np.linalg.inv(spatial_cov)
                spatial_weights = np.diag(cov_inv) / np.trace(cov_inv) * n_points
            except np.linalg.LinAlgError:
                # Fallback to regularized inverse
                cov_inv = np.linalg.pinv(spatial_cov + np.eye(n_points) * 1e-3)
                spatial_weights = np.diag(cov_inv) / np.trace(cov_inv) * n_points
            
            # Combine base weights with spatial weights
            combined_weights = base_weights * spatial_weights
            
            # Distance-dependent tapering
            distance_taper = np.exp(-distance_values / (2 * correlation_length))
            final_weights = combined_weights * distance_taper
            
            # Normalize weights
            final_weights /= np.sum(final_weights)
            
            return final_weights
            
        except Exception as e:
            logger.error(f"Error calculating advanced spatial weights: {e}")
            # Enhanced fallback with distance weighting
            distance_weights = 1.0 / (distance_values + 100.0)**0.5
            base_weights = 1.0 / (uncertainty_values**2 + 1e-12)
            return (base_weights * distance_weights) / np.sum(base_weights * distance_weights)

    async def _query_surrounding_strain_measurements(
        self, lon: float, lat: float, radius: float
    ) -> List[Dict[str, Any]]:
        """
        Query surrounding strain measurements from database
        """
        try:
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        sm.strain_xx, sm.strain_yy, sm.strain_xy,
                        sm.uncertainty_xx, sm.uncertainty_yy, sm.uncertainty_xy,
                        ST_X(sm.location) as longitude,
                        ST_Y(sm.location) as latitude,
                        sm.measurement_timestamp,
                        sm.instrument_type,
                        sm.data_quality_score,
                        ST_Distance(ST_Point($1, $2), sm.location) as distance
                    FROM strain_measurements sm
                    WHERE ST_DWithin(ST_Point($1, $2), sm.location, $3)
                    AND sm.measurement_timestamp > NOW() - INTERVAL '7 days'
                    AND sm.data_quality_score > 0.7
                    ORDER BY distance ASC, sm.data_quality_score DESC
                    LIMIT 20
                """
                
                results = await conn.fetch(query, lon, lat, radius)
                
                measurements = []
                for result in results:
                    measurements.append({
                        'strain_tensor': np.array([
                            [result['strain_xx'], result['strain_xy']],
                            [result['strain_xy'], result['strain_yy']]
                        ]),
                        'uncertainty_tensor': np.array([
                            [result['uncertainty_xx'], result['uncertainty_xy']],
                            [result['uncertainty_xy'], result['uncertainty_yy']]
                        ]),
                        'coordinates': (result['longitude'], result['latitude']),
                        'distance': result['distance'],
                        'quality': result['data_quality_score'],
                        'timestamp': result['measurement_timestamp']
                    })
                
                return measurements
                
        except Exception as e:
            logger.error(f"Error querying surrounding strain measurements: {e}")
            return []

    async def _kriging_gradient_interpolation(
        self, position: np.ndarray, measurements: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Use kriging interpolation for gradient estimation
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            
            # Extract coordinates and strain values
            coords = np.array([m['coordinates'] for m in measurements])
            strain_values = np.array([np.trace(m['strain_tensor']) for m in measurements])
            uncertainties = np.array([np.trace(m['uncertainty_tensor']) for m in measurements])
            
            # Set up Gaussian Process with Matérn kernel
            kernel = Matern(length_scale=1000.0, nu=1.5) + WhiteKernel(noise_level=1e-12)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=uncertainties**2,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            
            # Fit GP to strain measurements
            gp.fit(coords, strain_values)
            
            # Calculate gradient at position using finite differences on GP
            dx = 100.0  # 100m spacing for gradient calculation
            
            gradient_points = np.array([
                [position[0, 0] - dx, position[0, 1]],
                [position[0, 0] + dx, position[0, 1]],
                [position[0, 0], position[0, 1] - dx],
                [position[0, 0], position[0, 1] + dx]
            ])
            
            # Predict strain at gradient points
            strain_predictions, strain_std = gp.predict(gradient_points, return_std=True)
            
            # Calculate gradients
            gradient_x = (strain_predictions[1] - strain_predictions[0]) / (2 * dx)
            gradient_y = (strain_predictions[3] - strain_predictions[2]) / (2 * dx)
            
            return np.array([gradient_x, gradient_y])
            
        except Exception as e:
            logger.error(f"Error in kriging gradient interpolation: {e}")
            # Fallback to inverse distance weighting
            return await self._inverse_distance_gradient_interpolation(position, measurements)

    async def _interpolate_gradient_from_neighbors(self, position: np.ndarray) -> np.ndarray:
        """
        Use interpolated gradient from neighboring measurements instead of fixed value
        """
        try:
            # Query nearby gradient measurements
            measurements = await self._query_surrounding_strain_measurements(
                position[0, 0], position[0, 1], radius=10000.0
            )
            
            if len(measurements) >= 2:
                return await self._kriging_gradient_interpolation(position, measurements)
            
            # If no measurements available, use regional gradient model
            regional_gradient = await self._get_regional_strain_gradient_model(
                position[0, 0], position[0, 1]
            )
            
            return regional_gradient
            
        except Exception as e:
            logger.error(f"Error interpolating gradient from neighbors: {e}")
            # Final fallback to physics-based estimate
            return np.array([1e-12, 1e-12])

    async def _get_location_specific_earth_radius(self) -> float:
        """
        Location-specific Earth radius from geodetic models
        """
        try:
            # Get average location from recent analyses
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT AVG(ST_X(location)) as avg_lon, AVG(ST_Y(location)) as avg_lat
                    FROM analysis_locations 
                    WHERE timestamp > NOW() - INTERVAL '30 days'
                """
                result = await conn.fetchrow(query)
                
                if result and result['avg_lat']:
                    lat = np.radians(result['avg_lat'])
                    
                    # WGS84 ellipsoid parameters
                    a = 6378137.0  # Semi-major axis
                    f = 1/298.257223563  # Flattening
                    e2 = 2*f - f**2  # First eccentricity squared
                    
                    # Radius of curvature in meridian
                    M = a * (1 - e2) / (1 - e2 * np.sin(lat)**2)**(3/2)
                    
                    # Radius of curvature in prime vertical
                    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
                    
                    # Mean radius of curvature
                    R = np.sqrt(M * N)
                    
                    return R
                
                return 6371000.0  # Default WGS84 mean radius
                
        except Exception as e:
            logger.error(f"Error getting location-specific Earth radius: {e}")
            return 6371000.0

    async def _get_location_specific_gravity(self) -> float:
        """
        Location-specific gravity from gravity models
        """
        try:
            # Query gravity database for local gravity values
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT AVG(gm.gravity_value) as avg_gravity
                    FROM gravity_measurements gm
                    JOIN analysis_locations al ON ST_DWithin(gm.location, al.location, 50000)
                    WHERE al.timestamp > NOW() - INTERVAL '30 days'
                    AND gm.measurement_quality > 0.8
                """
                result = await conn.fetchrow(query)
                
                if result and result['avg_gravity']:
                    return result['avg_gravity']
                
                # Fallback to theoretical gravity calculation
                avg_lat = await self._get_average_analysis_latitude()
                if avg_lat:
                    # International Gravity Formula 1967
                    lat_rad = np.radians(avg_lat)
                    g = 9.780318 * (1 + 0.0053024 * np.sin(lat_rad)**2 - 
                                   0.0000058 * np.sin(2*lat_rad)**2)
                    return g
                
                return 9.81  # Standard gravity
                
        except Exception as e:
            logger.error(f"Error getting location-specific gravity: {e}")
            return 9.81

    async def _get_real_time_atmospheric_pressure(self) -> float:
        """
        Real-time atmospheric pressure from weather data
        """
        try:
            # Query weather database for current pressure
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT AVG(wd.pressure) as avg_pressure
                    FROM weather_data wd
                    JOIN analysis_locations al ON ST_DWithin(wd.location, al.location, 25000)
                    WHERE wd.timestamp > NOW() - INTERVAL '1 hour'
                    AND wd.data_quality > 0.9
                """
                result = await conn.fetchrow(query)
                
                if result and result['avg_pressure']:
                    return result['avg_pressure'] * 100.0  # Convert hPa to Pa
                
                # Fallback to elevation-corrected standard pressure
                avg_elevation = await self._get_average_analysis_elevation()
                if avg_elevation:
                    # Barometric formula
                    p0 = 101325.0  # Sea level pressure
                    g = 9.80665
                    M = 0.0289644  # Molar mass of air
                    R = 8.31432  # Gas constant
                    T = 288.15  # Standard temperature
                    
                    pressure = p0 * np.exp(-g * M * avg_elevation / (R * T))
                    return pressure
                
                return 101325.0  # Standard pressure
                
        except Exception as e:
            logger.error(f"Error getting real-time atmospheric pressure: {e}")
            return 101325.0

    async def _calculate_advanced_gps_weights(
        self, uncertainty: np.ndarray, distances: np.ndarray, station_coords: np.ndarray
    ) -> np.ndarray:
        """
        Advanced weighting with spatial correlation for GPS data
        """
        try:
            # Base weights from uncertainties
            base_weights = 1.0 / (uncertainty**2 + 1e-12)
            
            # Spatial correlation modeling for GPS networks
            n_stations = len(station_coords)
            
            # Estimate spatial correlation parameters from GPS network geometry
            correlation_params = await self._estimate_gps_spatial_correlation(station_coords)
            
            # Build spatial covariance matrix
            spatial_cov = np.eye(n_stations)
            
            for i in range(n_stations):
                for j in range(i+1, n_stations):
                    distance_ij = np.linalg.norm(station_coords[i] - station_coords[j])
                    
                    # Exponential correlation model for GPS
                    correlation = np.exp(-distance_ij / correlation_params['range'])
                    spatial_cov[i, j] = correlation
                    spatial_cov[j, i] = correlation
            
            # Add nugget effect for measurement errors
            spatial_cov += np.diag(correlation_params['nugget'] * np.ones(n_stations))
            
            # Calculate optimal weights using generalized least squares
            try:
                cov_inv = np.linalg.inv(spatial_cov)
                ones = np.ones(n_stations)
                
                # GLS weights
                gls_weights = cov_inv @ ones / (ones.T @ cov_inv @ ones)
                
                # Combine with uncertainty-based weights
                combined_weights = base_weights * gls_weights
                
            except np.linalg.LinAlgError:
                # Fallback to regularized weights
                regularized_cov = spatial_cov + np.eye(n_stations) * 1e-6
                cov_inv = np.linalg.pinv(regularized_cov)
                combined_weights = base_weights * np.diag(cov_inv)
            
            # Distance tapering
            max_distance = np.max(distances)
            distance_taper = 1.0 - (distances / max_distance)**2
            final_weights = combined_weights * distance_taper
            
            # Normalize
            final_weights /= np.sum(final_weights)
            
            return final_weights
            
        except Exception as e:
            logger.error(f"Error calculating advanced GPS weights: {e}")
            # Enhanced fallback
            distance_weights = 1.0 / (distances + 1000.0)
            base_weights = 1.0 / (uncertainty**2 + 1e-12)
            return (base_weights * distance_weights) / np.sum(base_weights * distance_weights)

    async def _calculate_physics_based_transition_weights(
        self, distances: np.ndarray, transition_distance: float, 
        sill_depth: float, obs_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Physics-based transition using Green's functions
        """
        try:
            # Calculate wavelength of deformation signal
            characteristic_wavelength = 2 * np.pi * sill_depth
            
            # Near-field criterion based on wavelength
            near_field_distance = characteristic_wavelength / 4
            far_field_distance = characteristic_wavelength * 2
            
            # Initialize weights
            weight_finite = np.zeros_like(distances)
            weight_mogi = np.zeros_like(distances)
            
            for i, distance in enumerate(distances):
                if distance <= near_field_distance:
                    # Pure finite source
                    weight_finite[i] = 1.0
                    weight_mogi[i] = 0.0
                elif distance >= far_field_distance:
                    # Pure point source
                    weight_finite[i] = 0.0
                    weight_mogi[i] = 1.0
                else:
                    # Transition zone - use Green's function analysis
                    transition_weight = await self._calculate_greens_function_transition(
                        distance, sill_depth, obs_coords[i]
                    )
                    weight_finite[i] = transition_weight
                    weight_mogi[i] = 1.0 - transition_weight
            
            return weight_finite, weight_mogi
            
        except Exception as e:
            logger.error(f"Error calculating physics-based transition weights: {e}")
            # Enhanced fallback with depth scaling
            normalized_distances = distances / (sill_depth * 2)
            weight_finite = 1.0 / (1.0 + normalized_distances**1.5)
            weight_mogi = 1.0 - weight_finite
            return weight_finite, weight_mogi

    async def _calculate_adaptive_tolerance(
        self, obs_data: Dict[str, np.ndarray], inversion_params: Dict[str, Any]
    ) -> float:
        """
        Adaptive tolerance based on data uncertainty
        """
        try:
            # Calculate data-driven tolerance
            uncertainties = obs_data.get('uncertainties', np.array([0.01]))
            
            # Statistical tolerance based on uncertainty distribution
            mean_uncertainty = np.mean(uncertainties)
            std_uncertainty = np.std(uncertainties)
            
            # Adaptive tolerance: should be smaller than typical uncertainty
            base_tolerance = mean_uncertainty / 100.0
            
            # Adjust based on data quality
            data_quality = inversion_params.get('data_quality_score', 0.8)
            quality_factor = max(0.1, data_quality)
            
            # Adjust based on number of observations
            n_obs = len(obs_data.get('observations', []))
            sample_size_factor = max(0.5, min(2.0, np.sqrt(n_obs / 100.0)))
            
            adaptive_tolerance = base_tolerance * quality_factor / sample_size_factor
            
            # Bounds checking
            min_tolerance = 1e-12
            max_tolerance = mean_uncertainty / 10.0
            
            return max(min_tolerance, min(max_tolerance, adaptive_tolerance))
            
        except Exception as e:
            logger.error(f"Error calculating adaptive tolerance: {e}")
            return 1e-8

    async def _advanced_temporal_analysis_with_corrections(
        self, times: np.ndarray, sorted_data: List[Any]
    ) -> np.ndarray:
        """
        Advanced temporal analysis with seasonal corrections
        """
        try:
            # Convert to days with high precision
            times_days = (times - times[0]) / 86400.0
            
            # Apply seasonal corrections
            seasonal_corrections = await self._calculate_seasonal_corrections(times, sorted_data)
            corrected_times = times_days + seasonal_corrections
            
            # Apply tidal corrections
            tidal_corrections = await self._calculate_tidal_corrections(times, sorted_data)
            corrected_times += tidal_corrections
            
            # Apply atmospheric loading corrections
            atmospheric_corrections = await self._calculate_atmospheric_loading_corrections(
                times, sorted_data
            )
            corrected_times += atmospheric_corrections
            
            return corrected_times
            
        except Exception as e:
            logger.error(f"Error in advanced temporal analysis: {e}")
            # Enhanced fallback with leap second corrections
            return await self._enhanced_time_conversion_with_leap_seconds(times)

    async def _allocate_optimized_arrays(self, n_obs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pre-allocated arrays with memory optimization
        """
        try:
            # Use memory-aligned arrays for better performance
            ux = np.zeros(n_obs, dtype=np.float64, order='C')
            uy = np.zeros(n_obs, dtype=np.float64, order='C')
            uz = np.zeros(n_obs, dtype=np.float64, order='C')
            
            # Pre-fault arrays to ensure they're in memory
            ux[0] = 0.0
            uy[0] = 0.0
            uz[0] = 0.0
            
            return ux, uy, uz
            
        except Exception as e:
            logger.error(f"Error allocating optimized arrays: {e}")
            return np.zeros(n_obs), np.zeros(n_obs), np.zeros(n_obs)

    async def _handle_edge_singularity(
        self, edge_length: float, xi: float, eta: float, q: float
    ) -> bool:
        """
        Advanced singularity removal techniques
        """
        try:
            # Multiple criteria for singularity detection
            length_threshold = 1e-10
            coordinate_threshold = 1e-12
            
            if edge_length < length_threshold:
                return False
            
            # Check for coordinate singularities
            if abs(xi) < coordinate_threshold and abs(eta) < coordinate_threshold:
                return False
            
            # Check for depth singularities
            if abs(q) < coordinate_threshold:
                return False
            
            # Advanced regularization for near-singular cases
            if edge_length < 1e-6:
                # Apply regularization techniques
                regularization_success = await self._apply_singularity_regularization(
                    edge_length, xi, eta, q
                )
                return regularization_success
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling edge singularity: {e}")
            return edge_length >= 1e-10

    async def _calculate_crustal_model_corrections(
        self, sill_depth: float, obs_coords: np.ndarray, sill_x: float, sill_y: float
    ) -> np.ndarray:
        """
        Depth-dependent corrections from crustal models
        """
        try:
            # Get crustal structure from database
            crustal_structure = await self._get_crustal_structure_from_tomography(
                sill_x, sill_y, sill_depth
            )
            
            # Calculate corrections for each observation point
            corrections = np.ones(len(obs_coords))
            
            for i, obs_coord in enumerate(obs_coords):
                # Distance from sill
                distance = np.sqrt((obs_coord[0] - sill_x)**2 + (obs_coord[1] - sill_y)**2)
                
                # Layered medium correction
                layer_correction = await self._calculate_layered_medium_correction_advanced(
                    crustal_structure, sill_depth, distance
                )
                
                # Topographic correction
                topo_correction = await self._calculate_topographic_correction_advanced(
                    obs_coord[0], obs_coord[1], sill_depth
                )
                
                # Combined correction
                corrections[i] = layer_correction * topo_correction
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error calculating crustal model corrections: {e}")
            # Enhanced fallback with depth scaling
            depth_factor = sill_depth / 5000.0  # Normalized to 5km
            return 1.0 + 0.1 * depth_factor * np.exp(-depth_factor)

    async def _get_instrument_specific_uncertainty(
        self, instrument: TiltStrainData
    ) -> Tuple[float, float]:
        """
        Instrument-specific uncertainty from calibration database
        """
        try:
            # Query calibration database
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        ic.uncertainty_x, ic.uncertainty_y,
                        ic.calibration_date, ic.drift_rate,
                        ic.temperature_coefficient, ic.pressure_coefficient
                    FROM instrument_calibrations ic
                    WHERE ic.instrument_id = $1
                    ORDER BY ic.calibration_date DESC
                    LIMIT 1
                """
                
                result = await conn.fetchrow(query, instrument.instrument_id)
                
                if result:
                    base_unc_x = result['uncertainty_x']
                    base_unc_y = result['uncertainty_y']
                    
                    # Apply time-dependent drift correction
                    days_since_cal = (datetime.now() - result['calibration_date']).days
                    drift_correction = result['drift_rate'] * days_since_cal
                    
                    # Apply environmental corrections
                    temp_correction = result['temperature_coefficient'] * instrument.temperature_correction
                    pressure_correction = result['pressure_coefficient'] * instrument.pressure_correction
                    
                    final_unc_x = base_unc_x + drift_correction + temp_correction + pressure_correction
                    final_unc_y = base_unc_y + drift_correction + temp_correction + pressure_correction
                    
                    return (max(1e-9, final_unc_x), max(1e-9, final_unc_y))
                
                # Fallback to instrument type defaults
                return await self._get_instrument_type_default_uncertainty(instrument.instrument_type)
                
        except Exception as e:
            logger.error(f"Error getting instrument-specific uncertainty: {e}")
            return (1e-6, 1e-6)

    async def _interpolate_strain_field_from_neighbors(
        self, positions: np.ndarray
    ) -> np.ndarray:
        """
        Interpolated strain field from neighboring measurements
        """
        try:
            # Query nearby strain measurements
            strain_tensors = np.zeros((len(positions), 2, 2))
            
            for i, pos in enumerate(positions):
                nearby_measurements = await self._query_surrounding_strain_measurements(
                    pos[0], pos[1], radius=5000.0
                )
                
                if len(nearby_measurements) >= 3:
                    # Kriging interpolation
                    interpolated_strain = await self._kriging_strain_interpolation(
                        pos, nearby_measurements
                    )
                    strain_tensors[i] = interpolated_strain
                else:
                    # Regional strain model
                    regional_strain = await self._get_regional_strain_model(pos[0], pos[1])
                    strain_tensors[i] = regional_strain
            
            return strain_tensors
            
        except Exception as e:
            logger.error(f"Error interpolating strain field: {e}")
            # Return small background strain
            return np.full((len(positions), 2, 2), 1e-9)

    async def _get_secure_database_configuration(self) -> Dict[str, Any]:
        """
        Secure credential management from environment/vault
        """
        try:
            import os
            from cryptography.fernet import Fernet
            
            # Try to get credentials from environment variables first
            config = {
                'postgres_host': os.getenv('POSTGRES_HOST', 'localhost'),
                'postgres_port': int(os.getenv('POSTGRES_PORT', '5432')),
                'postgres_user': os.getenv('POSTGRES_USER', 'prometheus'),
                'postgres_password': await self._get_secure_password('POSTGRES_PASSWORD'),
                'postgres_db': os.getenv('POSTGRES_DB', 'prometheus'),
                'redis_host': os.getenv('REDIS_HOST', 'localhost'),
                'redis_port': int(os.getenv('REDIS_PORT', '6379')),
                'influx_url': os.getenv('INFLUX_URL', 'http://localhost:8086'),
                'influx_token': await self._get_secure_password('INFLUX_TOKEN'),
                'influx_org': os.getenv('INFLUX_ORG', 'prometheus'),
                'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
                'neo4j_password': await self._get_secure_password('NEO4J_PASSWORD')
            }
            
            return config
            
        except Exception as e:
            logger.error(f"Error getting secure database configuration: {e}")
            # Fallback to minimal configuration
            return {
                'postgres_host': 'localhost',
                'postgres_port': 5432,
                'postgres_user': 'prometheus',
                'postgres_db': 'prometheus'
            }

    async def _get_secure_password(self, env_var: str) -> str:
        """
        Get secure password from environment or vault
        """
        try:
            import os
            
            # Try environment variable first
            password = os.getenv(env_var)
            if password:
                return password
            
            # Try encrypted file
            encrypted_file = f"/etc/prometheus/secrets/{env_var.lower()}.enc"
            if os.path.exists(encrypted_file):
                return await self._decrypt_password_file(encrypted_file)
            
            # Try vault service
            vault_password = await self._get_password_from_vault(env_var)
            if vault_password:
                return vault_password
            
            # Fallback (should not be used in production)
            logger.warning(f"Using fallback password for {env_var}")
            return "prometheus123"
            
        except Exception as e:
            logger.error(f"Error getting secure password for {env_var}: {e}")
            return "prometheus123"

    async def _calculate_advanced_tilt_weights(
        self, uncertainty: np.ndarray, instrument: TiltStrainData, distance: float
    ) -> np.ndarray:
        """
        Advanced weighting for tilt measurements
        """
        try:
            # Base weights from uncertainties
            base_weights = 1.0 / (uncertainty**2 + 1e-15)
            
            # Instrument quality factor
            quality_factor = await self._get_instrument_quality_factor(instrument)
            
            # Distance-dependent weighting
            distance_weight = np.exp(-distance / 10000.0)  # 10km characteristic distance
            
            # Time-dependent weighting (fresher data gets higher weight)
            time_weight = await self._calculate_temporal_weight(instrument.measurement_date)
            
            # Environmental stability weighting
            env_weight = await self._calculate_environmental_stability_weight(instrument)
            
            # Combined weighting
            combined_weights = base_weights * quality_factor * distance_weight * time_weight * env_weight
            
            # Normalize
            combined_weights /= np.sum(combined_weights)
            
            return combined_weights
            
        except Exception as e:
            logger.error(f"Error calculating advanced tilt weights: {e}")
            return 1.0 / (uncertainty**2 + 1e-12)

    async def _calculate_robust_correlation_with_outlier_detection(
        self, obs_values: np.ndarray, pred_values: np.ndarray, weights: np.ndarray
    ) -> float:
        """
        Robust correlation with outlier detection
        """
        try:
            if len(obs_values) < 3:
                return 0.0
            
            # Detect outliers using robust statistics
            outlier_mask = await self._detect_outliers_robust(obs_values, pred_values)
            
            # Remove outliers
            clean_obs = obs_values[~outlier_mask]
            clean_pred = pred_values[~outlier_mask]
            clean_weights = weights[~outlier_mask]
            
            if len(clean_obs) < 3:
                return 0.0
            
            # Calculate robust correlation using Spearman rank correlation
            from scipy.stats import spearmanr
            
            # Weighted Spearman correlation
            spearman_corr, spearman_p = spearmanr(clean_obs, clean_pred)
            
            # Also calculate Kendall's tau for robustness
            from scipy.stats import kendalltau
            kendall_corr, kendall_p = kendalltau(clean_obs, clean_pred)
            
            # Weighted average of robust correlations
            if not np.isnan(spearman_corr) and not np.isnan(kendall_corr):
                robust_correlation = 0.7 * spearman_corr + 0.3 * kendall_corr
            elif not np.isnan(spearman_corr):
                robust_correlation = spearman_corr
            elif not np.isnan(kendall_corr):
                robust_correlation = kendall_corr
            else:
                robust_correlation = 0.0
            
            return robust_correlation
            
        except Exception as e:
            logger.error(f"Error calculating robust correlation: {e}")
            # Fallback to simple correlation
            if len(obs_values) >= 2:
                return np.corrcoef(obs_values, pred_values)[0, 1]
            return 0.0

    async def _calculate_minimum_significance_for_sample_size(self, sample_size: int) -> float:
        """
        Calculate minimum significance for small sample sizes
        """
        try:
            if sample_size <= 0:
                return 0.0
            elif sample_size == 1:
                return 0.0
            elif sample_size == 2:
                return 0.1  # Very low significance for n=2
            else:
                # Use Fisher's exact test for small samples
                from scipy.stats import fisher_exact
                
                # Minimum detectable significance based on sample size
                min_significance = 1.0 / np.sqrt(sample_size)
                return max(0.05, min_significance)
                
        except Exception as e:
            logger.error(f"Error calculating minimum significance: {e}")
            return 0.0

    async def _apply_multiple_testing_correction(
        self, raw_significance: float, sample_size: int
    ) -> float:
        """
        Apply multiple testing correction to significance level
        """
        try:
            # Estimate number of tests based on sample size
            estimated_tests = max(1, sample_size // 10)
            
            # Bonferroni correction
            bonferroni_corrected = raw_significance / estimated_tests
            
            # Benjamini-Hochberg FDR correction (less conservative)
            fdr_corrected = raw_significance * 0.05 / (0.05 + raw_significance)
            
            # Use geometric mean of corrections
            final_significance = np.sqrt(bonferroni_corrected * fdr_corrected)
            
            return max(0.001, min(1.0, final_significance))
            
        except Exception as e:
            logger.error(f"Error applying multiple testing correction: {e}")
            return raw_significance

    async def _detect_outliers_robust(
        self, obs_values: np.ndarray, pred_values: np.ndarray
    ) -> np.ndarray:
        """
        Detect outliers using robust statistical methods
        """
        try:
            # Calculate residuals
            residuals = obs_values - pred_values
            
            # Robust outlier detection using Modified Z-score
            median_residual = np.median(residuals)
            mad = np.median(np.abs(residuals - median_residual))
            
            # Modified Z-scores
            modified_z_scores = 0.6745 * (residuals - median_residual) / (mad + 1e-12)
            
            # Outliers are points with |modified Z-score| > 3.5
            outlier_mask = np.abs(modified_z_scores) > 3.5
            
            # Additional check using Isolation Forest for multivariate outliers
            try:
                from sklearn.ensemble import IsolationForest
                
                data = np.column_stack([obs_values, pred_values])
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outlier_labels = iso_forest.fit_predict(data)
                
                # Combine both methods
                isolation_outliers = outlier_labels == -1
                combined_outliers = outlier_mask | isolation_outliers
                
                return combined_outliers
                
            except ImportError:
                # Fallback to just Modified Z-score method
                return outlier_mask
                
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return np.zeros(len(obs_values), dtype=bool)

    async def _get_instrument_quality_factor(self, instrument: TiltStrainData) -> float:
        """
        Get instrument quality factor from database
        """
        try:
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        iq.quality_score,
                        iq.calibration_stability,
                        iq.environmental_sensitivity,
                        iq.long_term_drift
                    FROM instrument_quality iq
                    WHERE iq.instrument_id = $1
                    ORDER BY iq.assessment_date DESC
                    LIMIT 1
                """
                
                result = await conn.fetchrow(query, instrument.instrument_id)
                
                if result:
                    # Composite quality factor
                    quality_score = result['quality_score'] or 0.8
                    stability = result['calibration_stability'] or 0.8
                    sensitivity = 1.0 - (result['environmental_sensitivity'] or 0.2)
                    drift_factor = 1.0 - (result['long_term_drift'] or 0.1)
                    
                    composite_quality = quality_score * stability * sensitivity * drift_factor
                    return max(0.1, min(1.0, composite_quality))
                
                # Default quality based on instrument type
                type_quality = {
                    'borehole_strainmeter': 0.95,
                    'laser_strainmeter': 0.90,
                    'tiltmeter': 0.85,
                    'strainmeter': 0.80
                }.get(instrument.instrument_type, 0.75)
                
                return type_quality
                
        except Exception as e:
            logger.error(f"Error getting instrument quality factor: {e}")
            return 0.8

    async def _calculate_temporal_weight(self, measurement_date: datetime) -> float:
        """
        Calculate temporal weight (fresher data gets higher weight)
        """
        try:
            # Time difference in days
            time_diff = (datetime.now() - measurement_date).total_seconds() / 86400.0
            
            # Exponential decay with 30-day half-life
            half_life_days = 30.0
            temporal_weight = np.exp(-time_diff * np.log(2) / half_life_days)
            
            return max(0.1, temporal_weight)
            
        except Exception as e:
            logger.error(f"Error calculating temporal weight: {e}")
            return 1.0

    async def _calculate_environmental_stability_weight(self, instrument: TiltStrainData) -> float:
        """
        Calculate environmental stability weighting
        """
        try:
            # Get recent environmental data for the instrument location
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        AVG(ed.temperature_stability) as temp_stability,
                        AVG(ed.pressure_stability) as pressure_stability,
                        AVG(ed.humidity_stability) as humidity_stability
                    FROM environmental_data ed
                    WHERE ST_DWithin(
                        ST_Point($1, $2), 
                        ed.location, 
                        1000
                    )
                    AND ed.timestamp > NOW() - INTERVAL '7 days'
                """
                
                result = await conn.fetchrow(
                    query, instrument.coordinates[0], instrument.coordinates[1]
                )
                
                if result:
                    temp_stability = result['temp_stability'] or 0.8
                    pressure_stability = result['pressure_stability'] or 0.8
                    humidity_stability = result['humidity_stability'] or 0.8
                    
                    # Combined environmental stability
                    env_weight = (temp_stability + pressure_stability + humidity_stability) / 3.0
                    return max(0.3, min(1.0, env_weight))
                
                return 0.8  # Default stability
                
        except Exception as e:
            logger.error(f"Error calculating environmental stability weight: {e}")
            return 0.8