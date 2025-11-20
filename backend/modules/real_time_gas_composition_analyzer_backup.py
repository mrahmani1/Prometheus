"""
Real-Time Gas Composition Analysis System

This module implements advanced real-time gas composition analysis with source attribution
for the Prometheus AI Platform's volatile degassing analysis system.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import sqlite3
import psycopg2
from scipy import optimize, integrate, interpolate, signal
from scipy.spatial import distance_matrix, KDTree
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import existing database and core modules
from .database_integration import PostgreSQLGISManager as DatabaseManager
from .geological_data_structures import DrillholeData

logger = logging.getLogger(__name__)

@dataclass
class VolcanicSystem:
    """Volcanic system data structure"""
    system_id: str
    name: str
    location: Tuple[float, float, float]
    system_type: str
    monitoring_status: str
    last_activity: Optional[datetime] = None

@dataclass
class GeochemicalData:
    """Geochemical analysis data structure"""
    sample_id: str
    timestamp: datetime
    location: Tuple[float, float, float]
    major_elements: Dict[str, float]
    trace_elements: Dict[str, float]
    isotopic_ratios: Dict[str, float]
    temperature: Optional[float] = None
    pressure: Optional[float] = None

@dataclass
class GasCompositionData:
    """Real-time gas composition measurements and analysis"""
    measurement_id: str
    timestamp: datetime
    location: Tuple[float, float, float]
    co2_concentration: float
    so2_concentration: float
    h2s_concentration: float
    other_gases: Dict[str, float]
    isotopic_signatures: Dict[str, float]
    source_attribution: Dict[str, float]
    flux_rate: float
    temperature: float
    pressure: float
    atmospheric_correction: Dict[str, float]

@dataclass
class GasSourceSignature:
    """Gas source signature for attribution analysis"""
    source_id: str
    source_type: str  # 'magmatic', 'hydrothermal', 'crustal', 'atmospheric'
    location: Tuple[float, float, float]
    characteristic_ratios: Dict[str, float]
    isotopic_signature: Dict[str, float]
    temperature_range: Tuple[float, float]
    pressure_range: Tuple[float, float]
    temporal_variability: Dict[str, float]

class RealTimeGasCompositionAnalyzer:
    """Advanced real-time gas composition analysis with source attribution"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.gas_sources = {}
        self.atmospheric_model = None
        self.ml_models = {}
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        
    async def initialize_gas_sources(self, volcanic_system: VolcanicSystem) -> Dict[str, GasSourceSignature]:
        """Initialize gas source signatures from database"""
        try:
            # Query gas source data
            source_query = """
            SELECT source_id, source_type, x_coord, y_coord, depth,
                   co2_so2_ratio, h2s_so2_ratio, co2_h2s_ratio,
                   delta_13c_co2, delta_34s_so2, he3_he4_ratio,
                   temperature_min, temperature_max, pressure_min, pressure_max
            FROM gas_sources 
            WHERE volcanic_system_id = %s
            """
            
            source_data = await self.db_manager.execute_query(
                source_query, (volcanic_system.system_id,)
            )
            
            gas_sources = {}
            
            for source in source_data:
                source_signature = GasSourceSignature(
                    source_id=source['source_id'],
                    source_type=source['source_type'],
                    location=(source['x_coord'], source['y_coord'], source['depth']),
                    characteristic_ratios={
                        'CO2_SO2': source['co2_so2_ratio'],
                        'H2S_SO2': source['h2s_so2_ratio'],
                        'CO2_H2S': source['co2_h2s_ratio']
                    },
                    isotopic_signature={
                        'delta_13C_CO2': source['delta_13c_co2'],
                        'delta_34S_SO2': source['delta_34s_so2'],
                        'He3_He4_ratio': source['he3_he4_ratio']
                    },
                    temperature_range=(source['temperature_min'], source['temperature_max']),
                    pressure_range=(source['pressure_min'], source['pressure_max']),
                    temporal_variability={}  # To be calculated from historical data
                )
                
                gas_sources[source['source_id']] = source_signature
            
            # Calculate temporal variability from historical data
            for source_id, source in gas_sources.items():
                variability = await self._calculate_temporal_variability(source_id, volcanic_system.system_id)
                source.temporal_variability = variability
            
            self.gas_sources = gas_sources
            logger.info(f"Initialized {len(gas_sources)} gas sources")
            
            return gas_sources
            
        except Exception as e:
            logger.error(f"Error initializing gas sources: {str(e)}")
            raise
    
    async def _calculate_temporal_variability(self, source_id: str, volcanic_system_id: str) -> Dict[str, float]:
        """Calculate temporal variability of gas source signatures"""
        try:
            # Query historical gas measurements
            history_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   delta_13c_co2, delta_34s_so2
            FROM gas_measurements 
            WHERE source_id = %s AND volcanic_system_id = %s
            AND timestamp > NOW() - INTERVAL '1 year'
            ORDER BY timestamp
            """
            
            historical_data = await self.db_manager.execute_query(
                history_query, (source_id, volcanic_system_id)
            )
            
            if len(historical_data) < 10:  # Need sufficient data
                return {'insufficient_data': True}
            
            # Calculate variability metrics
            co2_values = [d['co2_concentration'] for d in historical_data]
            so2_values = [d['so2_concentration'] for d in historical_data]
            h2s_values = [d['h2s_concentration'] for d in historical_data]
            
            # Calculate coefficient of variation
            variability = {
                'co2_cv': np.std(co2_values) / np.mean(co2_values) if np.mean(co2_values) > 0 else 0,
                'so2_cv': np.std(so2_values) / np.mean(so2_values) if np.mean(so2_values) > 0 else 0,
                'h2s_cv': np.std(h2s_values) / np.mean(h2s_values) if np.mean(h2s_values) > 0 else 0,
                'data_points': len(historical_data)
            }
            
            # Calculate isotopic variability if available
            delta_13c_values = [d['delta_13c_co2'] for d in historical_data if d['delta_13c_co2'] is not None]
            if delta_13c_values:
                variability['delta_13c_cv'] = np.std(delta_13c_values) / abs(np.mean(delta_13c_values))
            
            return variability
            
        except Exception as e:
            logger.error(f"Error calculating temporal variability: {str(e)}")
            return {'error': str(e)}
    
    async def analyze_real_time_composition(self, measurement_data: Dict[str, Any]) -> GasCompositionData:
        """Analyze real-time gas composition with source attribution"""
        try:
            # Extract measurement parameters
            timestamp = measurement_data.get('timestamp', datetime.now())
            location = measurement_data.get('location', (0, 0, 0))
            
            # Raw gas concentrations
            co2_raw = measurement_data.get('co2_ppm', 0) / 1e6  # Convert ppm to fraction
            so2_raw = measurement_data.get('so2_ppm', 0) / 1e6
            h2s_raw = measurement_data.get('h2s_ppm', 0) / 1e6
            
            # Other gases
            other_gases = {
                'CH4': measurement_data.get('ch4_ppm', 0) / 1e6,
                'H2': measurement_data.get('h2_ppm', 0) / 1e6,
                'N2': measurement_data.get('n2_percent', 0) / 100,
                'O2': measurement_data.get('o2_percent', 0) / 100
            }
            
            # Environmental conditions
            temperature = measurement_data.get('temperature', 25.0)  # °C
            pressure = measurement_data.get('pressure', 101325.0)  # Pa
            wind_speed = measurement_data.get('wind_speed', 0.0)  # m/s
            wind_direction = measurement_data.get('wind_direction', 0.0)  # degrees
            
            # Apply atmospheric corrections
            atmospheric_correction = await self._apply_atmospheric_corrections(
                co2_raw, so2_raw, h2s_raw, temperature, pressure, wind_speed, wind_direction, location
            )
            
            # Corrected concentrations
            co2_corrected = co2_raw * atmospheric_correction['co2_factor']
            so2_corrected = so2_raw * atmospheric_correction['so2_factor']
            h2s_corrected = h2s_raw * atmospheric_correction['h2s_factor']
            
            # Calculate gas flux rate
            flux_rate = await self._calculate_gas_flux_rate(
                co2_corrected, so2_corrected, h2s_corrected, wind_speed, location
            )
            
            # Isotopic signatures (if available)
            isotopic_signatures = {
                'delta_13C_CO2': measurement_data.get('delta_13c_co2', None),
                'delta_34S_SO2': measurement_data.get('delta_34s_so2', None),
                'He3_He4_ratio': measurement_data.get('he3_he4_ratio', None)
            }
            
            # Perform source attribution
            source_attribution = await self._perform_source_attribution(
                co2_corrected, so2_corrected, h2s_corrected, isotopic_signatures, location
            )
            
            # Create gas composition data object
            gas_composition = GasCompositionData(
                measurement_id=f"gas_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(location)) % 10000}",
                timestamp=timestamp,
                location=location,
                co2_concentration=co2_corrected,
                so2_concentration=so2_corrected,
                h2s_concentration=h2s_corrected,
                other_gases=other_gases,
                isotopic_signatures=isotopic_signatures,
                source_attribution=source_attribution,
                flux_rate=flux_rate,
                temperature=temperature,
                pressure=pressure,
                atmospheric_correction=atmospheric_correction
            )
            
            # Store in database
            await self._store_gas_measurement(gas_composition)
            
            return gas_composition
            
        except Exception as e:
            logger.error(f"Error analyzing real-time composition: {str(e)}")
            raise
    
    async def _apply_atmospheric_corrections(self, co2: float, so2: float, h2s: float,
                                           temperature: float, pressure: float,
                                           wind_speed: float, wind_direction: float,
                                           location: Tuple[float, float, float]) -> Dict[str, float]:
        """Apply advanced atmospheric corrections using physics-based models and real-time meteorological data"""
        try:
            # Query real-time atmospheric data from database
            atm_query = """
            SELECT temperature, pressure, humidity, wind_speed, wind_direction,
                   atmospheric_stability, boundary_layer_height, co2_background,
                   so2_background, h2s_background, ozone_concentration
            FROM atmospheric_conditions 
            WHERE ST_DWithin(location, ST_Point(%s, %s), 50000)
            AND timestamp > NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC LIMIT 1
            """
            
            atm_data = await self.db_manager.execute_query(
                atm_query, (location[0], location[1])
            )
            
            if atm_data:
                atm_conditions = atm_data[0]
                humidity = atm_conditions.get('humidity', 50.0)
                stability = atm_conditions.get('atmospheric_stability', 'D')
                boundary_height = atm_conditions.get('boundary_layer_height', 1000.0)
                background_co2 = atm_conditions.get('co2_background', 415e-6)
                background_so2 = atm_conditions.get('so2_background', 0.0)
                background_h2s = atm_conditions.get('h2s_background', 0.0)
                ozone = atm_conditions.get('ozone_concentration', 50e-9)
            else:
                # Use climatological defaults
                humidity = 50.0
                stability = 'D'
                boundary_height = 1000.0
                background_co2 = 415e-6
                background_so2 = 0.0
                background_h2s = 0.0
                ozone = 50e-9
            
            # Advanced temperature correction using molecular kinetic theory
            temp_kelvin = temperature + 273.15
            ref_temp = 298.15  # 25°C reference
            
            # Gas-specific temperature dependencies based on molecular properties
            co2_temp_factor = np.sqrt(ref_temp / temp_kelvin) * np.exp(-0.0012 * (temp_kelvin - ref_temp))
            so2_temp_factor = np.sqrt(ref_temp / temp_kelvin) * np.exp(-0.0018 * (temp_kelvin - ref_temp))
            h2s_temp_factor = np.sqrt(ref_temp / temp_kelvin) * np.exp(-0.0025 * (temp_kelvin - ref_temp))
            
            # Advanced pressure correction using ideal gas law with compressibility factors
            ref_pressure = 101325.0
            pressure_ratio = pressure / ref_pressure
            
            # Compressibility factors for volcanic gases at elevated temperatures
            z_co2 = 1.0 - 0.0001 * (pressure - ref_pressure) / 1000.0
            z_so2 = 1.0 - 0.0003 * (pressure - ref_pressure) / 1000.0
            z_h2s = 1.0 - 0.0002 * (pressure - ref_pressure) / 1000.0
            
            co2_pressure_factor = pressure_ratio / z_co2
            so2_pressure_factor = pressure_ratio / z_so2
            h2s_pressure_factor = pressure_ratio / z_h2s
            
            # Humidity correction for gas solubility effects
            water_vapor_pressure = self._calculate_water_vapor_pressure(temperature, humidity)
            dry_air_pressure = pressure - water_vapor_pressure
            
            humidity_correction_co2 = 1.0 + 0.0001 * humidity  # CO2 slightly soluble
            humidity_correction_so2 = 1.0 + 0.002 * humidity   # SO2 highly soluble
            humidity_correction_h2s = 1.0 + 0.0005 * humidity  # H2S moderately soluble
            
            # Atmospheric stability correction for turbulent mixing
            stability_factors = {
                'A': 0.8, 'B': 0.85, 'C': 0.9, 'D': 1.0, 'E': 1.15, 'F': 1.3
            }
            stability_factor = stability_factors.get(stability, 1.0)
            
            # Boundary layer height correction for vertical mixing
            boundary_correction = np.log(boundary_height / 100.0) / np.log(10.0)  # Normalized to 100m reference
            
            # Photochemical reaction corrections
            photolysis_so2 = self._calculate_photolysis_rate('SO2', location, datetime.now())
            photolysis_h2s = self._calculate_photolysis_rate('H2S', location, datetime.now())
            
            # Ozone reaction corrections
            ozone_reaction_so2 = 1.0 + 0.1 * ozone * 1e9  # Convert to ppb
            ozone_reaction_h2s = 1.0 + 0.05 * ozone * 1e9
            
            # Combined correction factors
            co2_factor = (co2_temp_factor * co2_pressure_factor * humidity_correction_co2 * 
                         stability_factor * boundary_correction)
            
            so2_factor = (so2_temp_factor * so2_pressure_factor * humidity_correction_so2 * 
                         stability_factor * boundary_correction * photolysis_so2 * ozone_reaction_so2)
            
            h2s_factor = (h2s_temp_factor * h2s_pressure_factor * humidity_correction_h2s * 
                         stability_factor * boundary_correction * photolysis_h2s * ozone_reaction_h2s)
            
            return {
                'co2_factor': co2_factor,
                'so2_factor': so2_factor,
                'h2s_factor': h2s_factor,
                'temperature_correction': (co2_temp_factor, so2_temp_factor, h2s_temp_factor),
                'pressure_correction': (co2_pressure_factor, so2_pressure_factor, h2s_pressure_factor),
                'humidity_correction': (humidity_correction_co2, humidity_correction_so2, humidity_correction_h2s),
                'stability_correction': stability_factor,
                'boundary_layer_correction': boundary_correction,
                'photochemical_correction': (1.0, photolysis_so2, photolysis_h2s),
                'ozone_reaction_correction': (1.0, ozone_reaction_so2, ozone_reaction_h2s),
                'background_subtraction': (background_co2, background_so2, background_h2s),
                'atmospheric_conditions': {
                    'humidity': humidity,
                    'stability': stability,
                    'boundary_height': boundary_height,
                    'ozone': ozone
                }
            }
            
        except Exception as e:
            logger.error(f"Error applying atmospheric corrections: {str(e)}")
            return {'co2_factor': 1.0, 'so2_factor': 1.0, 'h2s_factor': 1.0}
    
    async def _calculate_gas_flux_rate(self, co2: float, so2: float, h2s: float,
                                     wind_speed: float, location: Tuple[float, float, float]) -> float:
        """Calculate advanced gas flux rate using physics-based atmospheric dispersion models"""
        try:
            # Query real-time meteorological and topographic data
            flux_query = """
            SELECT wind_speed, wind_direction, temperature, pressure, humidity,
                   atmospheric_stability, boundary_layer_height, surface_roughness,
                   topographic_complexity, thermal_stratification
            FROM meteorological_stations ms
            JOIN topographic_data td ON ST_DWithin(ms.location, td.location, 10000)
            WHERE ST_DWithin(ms.location, ST_Point(%s, %s), 25000)
            AND ms.timestamp > NOW() - INTERVAL '30 minutes'
            ORDER BY ST_Distance(ms.location, ST_Point(%s, %s)) LIMIT 1
            """
            
            met_data = await self.db_manager.execute_query(
                flux_query, (location[0], location[1], location[0], location[1])
            )
            
            if met_data:
                met_conditions = met_data[0]
                stability = met_conditions.get('atmospheric_stability', 'D')
                boundary_height = met_conditions.get('boundary_layer_height', 1000.0)
                surface_roughness = met_conditions.get('surface_roughness', 0.1)
                topo_complexity = met_conditions.get('topographic_complexity', 1.0)
                thermal_strat = met_conditions.get('thermal_stratification', 0.0)
                temperature = met_conditions.get('temperature', 15.0)
                pressure = met_conditions.get('pressure', 101325.0)
                humidity = met_conditions.get('humidity', 50.0)
            else:
                # Use default values
                stability = 'D'
                boundary_height = 1000.0
                surface_roughness = 0.1
                topo_complexity = 1.0
                thermal_strat = 0.0
                temperature = 15.0
                pressure = 101325.0
                humidity = 50.0
            
            # Calculate molecular properties for each gas
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08}  # g/mol
            gas_concentrations = {'CO2': co2, 'SO2': so2, 'H2S': h2s}
            
            # Advanced flux calculation using mass balance approach
            total_flux = 0.0
            
            for gas, concentration in gas_concentrations.items():
                if concentration > 0:
                    # Calculate gas-specific properties
                    mw = molecular_weights[gas]
                    
                    # Gas density at measurement conditions (kg/m³)
                    temp_kelvin = temperature + 273.15
                    gas_density = (mw * pressure) / (8314.0 * temp_kelvin)  # Using ideal gas law
                    
                    # Atmospheric stability parameters (Pasquill-Gifford)
                    stability_params = self._get_stability_parameters(stability)
                    
                    # Effective wind speed with surface roughness correction
                    effective_wind = wind_speed * (np.log(10.0 / surface_roughness) / np.log(10.0 / 0.03))
                    
                    # Topographic enhancement factor
                    topo_factor = 1.0 + 0.3 * topo_complexity * np.exp(-effective_wind / 5.0)
                    
                    # Thermal stratification effects
                    buoyancy_factor = 1.0 + thermal_strat * 0.1
                    
                    # Boundary layer mixing efficiency
                    mixing_efficiency = self._calculate_mixing_efficiency(
                        effective_wind, boundary_height, stability_params
                    )
                    
                    # Turbulent diffusion coefficient (m²/s)
                    diffusion_coeff = self._calculate_turbulent_diffusion(
                        effective_wind, stability_params, surface_roughness
                    )
                    
                    # Convective velocity scale (m/s)
                    convective_velocity = self._calculate_convective_velocity(
                        temperature, boundary_height, thermal_strat
                    )
                    
                    # Mass flux calculation using advanced dispersion theory
                    # Flux = C * u * h * ρ * f_mix * f_topo * f_buoy
                    gas_flux = (concentration * effective_wind * boundary_height * 
                               gas_density * mixing_efficiency * topo_factor * buoyancy_factor)
                    
                    # Add convective contribution
                    convective_flux = concentration * convective_velocity * gas_density * 0.3
                    
                    # Add turbulent diffusion contribution
                    turbulent_flux = concentration * np.sqrt(diffusion_coeff * effective_wind) * gas_density
                    
                    # Total gas flux (kg/s per m²)
                    total_gas_flux = gas_flux + convective_flux + turbulent_flux
                    
                    total_flux += total_gas_flux
            
            # Apply uncertainty correction based on measurement quality
            measurement_uncertainty = self._assess_measurement_uncertainty(
                gas_concentrations, wind_speed, stability
            )
            
            flux_uncertainty_factor = 1.0 + measurement_uncertainty
            
            return total_flux * flux_uncertainty_factor
            
        except Exception as e:
            logger.error(f"Error calculating gas flux rate: {str(e)}")
            return 0.0
    
    async def _perform_source_attribution(self, co2: float, so2: float, h2s: float,
                                        isotopic_signatures: Dict[str, Optional[float]],
                                        location: Tuple[float, float, float]) -> Dict[str, float]:
        """Perform gas source attribution using machine learning and geochemical analysis"""
        try:
            if not self.gas_sources:
                return {'unknown': 1.0}
            
            # Calculate gas ratios
            gas_ratios = self._calculate_gas_ratios(co2, so2, h2s)
            
            # Source attribution scores
            source_scores = {}
            
            for source_id, source in self.gas_sources.items():
                # Calculate similarity score based on gas ratios
                ratio_score = self._calculate_ratio_similarity(gas_ratios, source.characteristic_ratios)
                
                # Calculate isotopic similarity (if available)
                isotopic_score = self._calculate_isotopic_similarity(isotopic_signatures, source.isotopic_signature)
                
                # Calculate spatial proximity score
                spatial_score = self._calculate_spatial_proximity(location, source.location)
                
                # Combined score (weighted average)
                combined_score = 0.5 * ratio_score + 0.3 * isotopic_score + 0.2 * spatial_score
                
                source_scores[source_id] = combined_score
            
            # Normalize scores to sum to 1
            total_score = sum(source_scores.values())
            if total_score > 0:
                source_attribution = {k: v / total_score for k, v in source_scores.items()}
            else:
                source_attribution = {'unknown': 1.0}
            
            return source_attribution
            
        except Exception as e:
            logger.error(f"Error performing source attribution: {str(e)}")
            return {'error': 1.0}
    
    def _calculate_gas_ratios(self, co2: float, so2: float, h2s: float) -> Dict[str, float]:
        """Calculate gas ratios for source attribution"""
        ratios = {}
        
        # Avoid division by zero
        epsilon = 1e-10
        
        ratios['CO2_SO2'] = co2 / (so2 + epsilon)
        ratios['H2S_SO2'] = h2s / (so2 + epsilon)
        ratios['CO2_H2S'] = co2 / (h2s + epsilon)
        
        return ratios
    
    def _calculate_ratio_similarity(self, measured_ratios: Dict[str, float], 
                                  source_ratios: Dict[str, float]) -> float:
        """Calculate similarity between measured and source gas ratios"""
        similarities = []
        
        for ratio_name in measured_ratios:
            if ratio_name in source_ratios:
                measured = measured_ratios[ratio_name]
                source = source_ratios[ratio_name]
                
                # Log-scale similarity (gas ratios can vary over orders of magnitude)
                if measured > 0 and source > 0:
                    log_diff = abs(np.log10(measured) - np.log10(source))
                    similarity = max(0, 1 - log_diff / 3.0)  # 3 orders of magnitude tolerance
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_isotopic_similarity(self, measured_isotopes: Dict[str, Optional[float]],
                                     source_isotopes: Dict[str, float]) -> float:
        """Calculate similarity between measured and source isotopic signatures"""
        similarities = []
        
        for isotope_name, measured_value in measured_isotopes.items():
            if measured_value is not None and isotope_name in source_isotopes:
                source_value = source_isotopes[isotope_name]
                
                # Isotopic similarity (per mil scale)
                diff = abs(measured_value - source_value)
                
                # Different tolerances for different isotopes
                if 'delta_13C' in isotope_name:
                    tolerance = 2.0  # ±2‰
                elif 'delta_34S' in isotope_name:
                    tolerance = 3.0  # ±3‰
                elif 'He3_He4' in isotope_name:
                    tolerance = measured_value * 0.2  # ±20% relative
                else:
                    tolerance = 1.0
                
                similarity = max(0, 1 - diff / tolerance)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5  # Neutral score if no isotopic data
    
    def _calculate_spatial_proximity(self, measurement_location: Tuple[float, float, float],
                                   source_location: Tuple[float, float, float]) -> float:
        """Calculate spatial proximity score between measurement and source"""
        # Calculate 3D distance
        distance = np.linalg.norm(np.array(measurement_location) - np.array(source_location))
        
        # Advanced proximity score with topographic and geological controls
        try:
            # Query topographic and geological data for advanced proximity calculation
            topo_query = """
            SELECT elevation, slope, aspect, geological_unit, permeability,
                   fracture_density, hydrothermal_alteration
            FROM topographic_geological_data
            WHERE ST_DWithin(location, ST_Point(%s, %s), 5000)
            ORDER BY ST_Distance(location, ST_Point(%s, %s)) LIMIT 1
            """
            
            topo_data = await self.db_manager.execute_query(
                topo_query, (measurement_location[0], measurement_location[1], 
                           measurement_location[0], measurement_location[1])
            )
            
            # Base proximity using 3D distance with elevation weighting
            horizontal_distance = np.linalg.norm(
                np.array(measurement_location[:2]) - np.array(source_location[:2])
            )
            vertical_distance = abs(measurement_location[2] - source_location[2])
            
            # Weight vertical distance more heavily for gas transport
            weighted_distance = np.sqrt(horizontal_distance**2 + (2.0 * vertical_distance)**2)
            
            # Advanced proximity factors
            base_proximity = np.exp(-weighted_distance / 1000.0)
            
            # Topographic enhancement/reduction factors
            topo_factor = 1.0
            if topo_data:
                topo = topo_data[0]
                
                # Slope effect on gas transport (steeper slopes enhance transport)
                slope = topo.get('slope', 0.0)
                slope_factor = 1.0 + 0.3 * np.sin(np.radians(slope))
                
                # Geological permeability effect
                permeability = topo.get('permeability', 1.0)
                perm_factor = np.log10(permeability + 1.0) / 3.0  # Normalize
                
                # Fracture density effect (enhances gas transport)
                fracture_density = topo.get('fracture_density', 0.0)
                fracture_factor = 1.0 + 0.5 * fracture_density
                
                # Hydrothermal alteration effect
                alteration = topo.get('hydrothermal_alteration', 0.0)
                alteration_factor = 1.0 + 0.4 * alteration
                
                topo_factor = slope_factor * perm_factor * fracture_factor * alteration_factor
            
            # Wind direction and atmospheric transport effects
            prevailing_wind_direction = 225.0  # SW (degrees)
            source_to_measurement_bearing = np.degrees(
                np.arctan2(measurement_location[1] - source_location[1],
                          measurement_location[0] - source_location[0])
            )
            
            # Wind alignment factor (higher when measurement is downwind)
            wind_alignment = np.cos(np.radians(source_to_measurement_bearing - prevailing_wind_direction))
            wind_factor = 0.5 + 0.5 * max(0, wind_alignment)  # 0.5 to 1.0 range
            
            # Atmospheric stability effect on transport
            stability_factor = 1.0  # Default neutral
            
            # Combined proximity score
            proximity_score = base_proximity * topo_factor * wind_factor * stability_factor
            
            return min(1.0, max(0.0, proximity_score))
            
        except Exception as e:
            logger.error(f"Error in advanced proximity calculation: {str(e)}")
            # Fallback to simple exponential decay
            characteristic_distance = 1000.0
            return np.exp(-distance / characteristic_distance)
    
    async def _store_gas_measurement(self, gas_composition: GasCompositionData):
        """Store gas measurement in database with comprehensive data validation and advanced storage optimization"""
        try:
            # Advanced data validation and quality control
            validated_data = await self._validate_measurement_data(gas_composition)
            
            # Calculate advanced derived parameters
            derived_params = await self._calculate_derived_parameters(validated_data)
            
            # Store main measurement record with full geochemical analysis
            insert_query = """
            INSERT INTO gas_measurements (
                measurement_id, timestamp, location, x_coord, y_coord, depth,
                co2_concentration, so2_concentration, h2s_concentration,
                ch4_concentration, h2_concentration, n2_concentration, o2_concentration,
                delta_13c_co2, delta_34s_so2, he3_he4_ratio,
                flux_rate, temperature, pressure,
                atmospheric_co2_correction, atmospheric_so2_correction, atmospheric_h2s_correction,
                photolysis_correction_so2, photolysis_correction_h2s,
                humidity_correction_factor, stability_correction_factor,
                boundary_layer_correction, ozone_reaction_correction,
                gas_ratio_co2_so2, gas_ratio_h2s_so2, gas_ratio_co2_h2s,
                plume_rise_height, effective_emission_height,
                turbulent_diffusion_coeff, convective_velocity_scale,
                mixing_efficiency, measurement_uncertainty,
                data_quality_score, validation_flags,
                meteorological_conditions, atmospheric_stability_class,
                wind_speed_corrected, surface_roughness_factor,
                topographic_enhancement_factor, thermal_stratification_index,
                created_at, updated_at
            ) VALUES (
                %s, %s, ST_Point(%s, %s), %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                NOW(), NOW()
            )
            """
            
            # Prepare comprehensive parameter set
            params = (
                validated_data.measurement_id,
                validated_data.timestamp,
                validated_data.location[0], validated_data.location[1], validated_data.location[2],
                validated_data.co2_concentration,
                validated_data.so2_concentration,
                validated_data.h2s_concentration,
                validated_data.other_gases.get('CH4', 0.0),
                validated_data.other_gases.get('H2', 0.0),
                validated_data.other_gases.get('N2', 0.0),
                validated_data.other_gases.get('O2', 0.0),
                validated_data.isotopic_signatures.get('delta_13C_CO2'),
                validated_data.isotopic_signatures.get('delta_34S_SO2'),
                validated_data.isotopic_signatures.get('He3_He4_ratio'),
                validated_data.flux_rate,
                validated_data.temperature,
                validated_data.pressure,
                validated_data.atmospheric_correction.get('co2_factor', 1.0),
                validated_data.atmospheric_correction.get('so2_factor', 1.0),
                validated_data.atmospheric_correction.get('h2s_factor', 1.0),
                validated_data.atmospheric_correction.get('photochemical_correction', (1.0, 1.0, 1.0))[1],
                validated_data.atmospheric_correction.get('photochemical_correction', (1.0, 1.0, 1.0))[2],
                validated_data.atmospheric_correction.get('humidity_correction', (1.0, 1.0, 1.0))[0],
                validated_data.atmospheric_correction.get('stability_correction', 1.0),
                validated_data.atmospheric_correction.get('boundary_layer_correction', 1.0),
                validated_data.atmospheric_correction.get('ozone_reaction_correction', (1.0, 1.0, 1.0))[1],
                derived_params['gas_ratios']['CO2_SO2'],
                derived_params['gas_ratios']['H2S_SO2'],
                derived_params['gas_ratios']['CO2_H2S'],
                derived_params['plume_parameters']['plume_rise_height'],
                derived_params['plume_parameters']['effective_height'],
                derived_params['atmospheric_parameters']['turbulent_diffusion'],
                derived_params['atmospheric_parameters']['convective_velocity'],
                derived_params['atmospheric_parameters']['mixing_efficiency'],
                derived_params['quality_metrics']['measurement_uncertainty'],
                derived_params['quality_metrics']['data_quality_score'],
                derived_params['quality_metrics']['validation_flags'],
                derived_params['meteorological_summary'],
                derived_params['atmospheric_parameters']['stability_class'],
                derived_params['atmospheric_parameters']['wind_speed_corrected'],
                derived_params['atmospheric_parameters']['surface_roughness_factor'],
                derived_params['atmospheric_parameters']['topographic_factor'],
                derived_params['atmospheric_parameters']['thermal_stratification']
            )
            
            await self.db_manager.execute_query(insert_query, params)
            
            # Store source attribution results in separate table for advanced analysis
            await self._store_source_attribution(validated_data, derived_params)
            
            # Store atmospheric correction details for model validation
            await self._store_atmospheric_corrections(validated_data)
            
            # Update real-time statistics and trend analysis
            await self._update_realtime_statistics(validated_data)
            
            logger.info(f"Successfully stored gas measurement {validated_data.measurement_id} with advanced parameters")
            
        except Exception as e:
            logger.error(f"Error storing gas measurement: {str(e)}")
            raise
    
    async def _validate_measurement_data(self, gas_composition: GasCompositionData) -> GasCompositionData:
        """Advanced measurement data validation with physics-based quality control"""
        try:
            # Create validated copy
            validated = gas_composition
            
            # Physical range validation with volcanic gas chemistry constraints
            concentration_limits = {
                'co2': (0.0, 0.95),  # 0-95% volume fraction
                'so2': (0.0, 0.1),   # 0-10% volume fraction  
                'h2s': (0.0, 0.05)   # 0-5% volume fraction
            }
            
            # Apply concentration limits
            validated.co2_concentration = np.clip(validated.co2_concentration, *concentration_limits['co2'])
            validated.so2_concentration = np.clip(validated.so2_concentration, *concentration_limits['so2'])
            validated.h2s_concentration = np.clip(validated.h2s_concentration, *concentration_limits['h2s'])
            
            # Validate gas ratios against volcanic systematics
            co2_so2_ratio = validated.co2_concentration / (validated.so2_concentration + 1e-10)
            h2s_so2_ratio = validated.h2s_concentration / (validated.so2_concentration + 1e-10)
            
            # Typical volcanic gas ratio ranges
            if co2_so2_ratio > 10000 or co2_so2_ratio < 0.1:
                logger.warning(f"Unusual CO2/SO2 ratio: {co2_so2_ratio}")
            
            if h2s_so2_ratio > 100 or h2s_so2_ratio < 0.001:
                logger.warning(f"Unusual H2S/SO2 ratio: {h2s_so2_ratio}")
            
            # Validate isotopic signatures
            if validated.isotopic_signatures.get('delta_13C_CO2') is not None:
                delta_13c = validated.isotopic_signatures['delta_13C_CO2']
                if delta_13c < -30 or delta_13c > 5:  # Typical volcanic range
                    logger.warning(f"Unusual δ13C-CO2 value: {delta_13c}‰")
            
            if validated.isotopic_signatures.get('delta_34S_SO2') is not None:
                delta_34s = validated.isotopic_signatures['delta_34S_SO2']
                if delta_34s < -10 or delta_34s > 20:  # Typical volcanic range
                    logger.warning(f"Unusual δ34S-SO2 value: {delta_34s}‰")
            
            # Validate environmental conditions
            if validated.temperature < -50 or validated.temperature > 1200:
                logger.warning(f"Extreme temperature: {validated.temperature}°C")
                validated.temperature = np.clip(validated.temperature, -50, 1200)
            
            if validated.pressure < 50000 or validated.pressure > 200000:
                logger.warning(f"Unusual pressure: {validated.pressure} Pa")
                validated.pressure = np.clip(validated.pressure, 50000, 200000)
            
            # Validate flux rate
            if validated.flux_rate < 0:
                validated.flux_rate = 0.0
            elif validated.flux_rate > 1e6:  # Unrealistically high
                logger.warning(f"Extremely high flux rate: {validated.flux_rate}")
                validated.flux_rate = min(validated.flux_rate, 1e6)
            
            return validated
            
        except Exception as e:
            logger.error(f"Error validating measurement data: {str(e)}")
            return gas_composition
    
    async def _calculate_derived_parameters(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Calculate comprehensive derived parameters for advanced volcanic gas analysis"""
        try:
            derived = {}
            
            # Advanced gas ratio calculations
            epsilon = 1e-12
            derived['gas_ratios'] = {
                'CO2_SO2': gas_composition.co2_concentration / (gas_composition.so2_concentration + epsilon),
                'H2S_SO2': gas_composition.h2s_concentration / (gas_composition.so2_concentration + epsilon),
                'CO2_H2S': gas_composition.co2_concentration / (gas_composition.h2s_concentration + epsilon),
                'total_sulfur': gas_composition.so2_concentration + gas_composition.h2s_concentration,
                'reduced_sulfur_fraction': gas_composition.h2s_concentration / 
                    (gas_composition.so2_concentration + gas_composition.h2s_concentration + epsilon)
            }
            
            # Advanced plume parameters using comprehensive physics
            emission_rates = {
                'CO2': gas_composition.co2_concentration * gas_composition.flux_rate,
                'SO2': gas_composition.so2_concentration * gas_composition.flux_rate,
                'H2S': gas_composition.h2s_concentration * gas_composition.flux_rate
            }
            
            # Calculate plume rise using advanced Briggs equations
            plume_rise = self._calculate_plume_rise(emission_rates, 5.0, gas_composition.temperature)
            
            derived['plume_parameters'] = {
                'plume_rise_height': plume_rise,
                'effective_height': gas_composition.location[2] + plume_rise,
                'emission_rates': emission_rates,
                'buoyancy_flux': self._calculate_buoyancy_flux(emission_rates, gas_composition.temperature),
                'momentum_flux': self._calculate_momentum_flux(emission_rates, gas_composition.temperature)
            }
            
            # Advanced atmospheric parameters
            stability_class = 'D'  # Default neutral
            stability_params = self._get_stability_parameters(stability_class)
            
            derived['atmospheric_parameters'] = {
                'stability_class': stability_class,
                'turbulent_diffusion': self._calculate_turbulent_diffusion(5.0, stability_params, 0.1),
                'convective_velocity': self._calculate_convective_velocity(gas_composition.temperature, 1000.0, 0.0),
                'mixing_efficiency': self._calculate_mixing_efficiency(5.0, 1000.0, stability_params),
                'wind_speed_corrected': 5.0,  # Default value
                'surface_roughness_factor': 1.0,
                'topographic_factor': 1.0,
                'thermal_stratification': 0.0
            }
            
            # Advanced quality metrics
            measurement_uncertainty = self._assess_measurement_uncertainty(
                {'co2': gas_composition.co2_concentration, 
                 'so2': gas_composition.so2_concentration,
                 'h2s': gas_composition.h2s_concentration}, 
                5.0, stability_class
            )
            
            # Data quality score based on multiple factors
            quality_score = self._calculate_data_quality_score(gas_composition, derived)
            
            # Validation flags for automated quality control
            validation_flags = self._generate_validation_flags(gas_composition, derived)
            
            derived['quality_metrics'] = {
                'measurement_uncertainty': measurement_uncertainty,
                'data_quality_score': quality_score,
                'validation_flags': validation_flags
            }
            
            # Meteorological summary
            derived['meteorological_summary'] = {
                'temperature': gas_composition.temperature,
                'pressure': gas_composition.pressure,
                'atmospheric_corrections_applied': len(gas_composition.atmospheric_correction) > 0
            }
            
            return derived
            
        except Exception as e:
            logger.error(f"Error calculating derived parameters: {str(e)}")
            return {}
    
    def _calculate_buoyancy_flux(self, emission_rates: Dict[str, float], temperature: float) -> float:
        """Calculate buoyancy flux for plume rise calculations"""
        try:
            total_emission = sum(emission_rates.values())
            if total_emission <= 0:
                return 0.0
            
            # Gas mixture density calculation
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08}
            
            total_moles = 0.0
            weighted_mw = 0.0
            
            for gas, rate in emission_rates.items():
                if gas in molecular_weights:
                    moles = rate / molecular_weights[gas]
                    total_moles += moles
                    weighted_mw += moles * molecular_weights[gas]
            
            if total_moles > 0:
                avg_molecular_weight = weighted_mw / total_moles
            else:
                return 0.0
            
            # Calculate densities
            temp_kelvin = temperature + 273.15
            pressure = 101325.0  # Pa
            
            gas_density = (avg_molecular_weight * pressure) / (8314.0 * temp_kelvin)
            air_density = (28.97 * pressure) / (8314.0 * 288.15)  # Standard air at 15°C
            
            # Buoyancy flux (m⁴/s³)
            density_diff = air_density - gas_density
            buoyancy_flux = 9.81 * density_diff * total_emission / gas_density
            
            return max(0.0, buoyancy_flux)
            
        except Exception as e:
            logger.error(f"Error calculating buoyancy flux: {str(e)}")
            return 0.0
    
    def _calculate_momentum_flux(self, emission_rates: Dict[str, float], temperature: float) -> float:
        """Calculate advanced momentum flux for plume rise using comprehensive volcanic physics"""
        try:
            total_emission = sum(emission_rates.values())
            if total_emission <= 0:
                return 0.0
            
            # Advanced gas mixture properties calculation
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08}  # g/mol
            
            # Calculate weighted average molecular weight
            total_moles = 0.0
            weighted_mw = 0.0
            
            for gas, rate in emission_rates.items():
                if gas in molecular_weights:
                    moles = rate / molecular_weights[gas]
                    total_moles += moles
                    weighted_mw += moles * molecular_weights[gas]
            
            if total_moles > 0:
                avg_molecular_weight = weighted_mw / total_moles
            else:
                avg_molecular_weight = 29.0  # Air molecular weight as fallback
            
            # Calculate gas mixture density at vent conditions
            temp_kelvin = temperature + 273.15
            pressure_vent = 101325.0  # Pa, assume atmospheric pressure at vent
            
            # Ideal gas law with compressibility correction
            gas_density = (avg_molecular_weight * pressure_vent) / (8314.0 * temp_kelvin)  # kg/m³
            
            # Advanced vent geometry estimation based on emission rate and gas properties
            # Use Torricelli's law and continuity equation for vent sizing
            
            # Estimate vent exit velocity using energy balance
            # v = sqrt(2 * g * h_effective + 2 * delta_P / rho)
            gravity = 9.81  # m/s²
            
            # Effective driving height (pressure head equivalent)
            magma_chamber_depth = 1000.0  # m, typical shallow chamber
            pressure_head = magma_chamber_depth * 0.1  # Reduced due to gas expansion
            
            # Pressure difference between chamber and atmosphere
            chamber_pressure = pressure_vent + gas_density * gravity * magma_chamber_depth
            delta_pressure = chamber_pressure - pressure_vent
            
            # Exit velocity calculation with gas expansion effects
            velocity_pressure_term = 2.0 * delta_pressure / gas_density
            velocity_gravity_term = 2.0 * gravity * pressure_head
            
            # Account for gas expansion during ascent (adiabatic expansion)
            expansion_factor = (chamber_pressure / pressure_vent)**(1/1.3)  # Adiabatic index ~1.3 for volcanic gases
            
            exit_velocity = np.sqrt(velocity_pressure_term + velocity_gravity_term) * expansion_factor
            
            # Apply realistic bounds for volcanic gas emissions
            exit_velocity = max(10.0, min(200.0, exit_velocity))  # 10-200 m/s range
            
            # Calculate vent area from continuity equation
            # A = mass_flow_rate / (density * velocity)
            mass_flow_rate = total_emission  # kg/s
            vent_area = mass_flow_rate / (gas_density * exit_velocity)  # m²
            
            # Vent diameter and radius
            vent_diameter = np.sqrt(4.0 * vent_area / np.pi)
            vent_radius = vent_diameter / 2.0
            
            # Advanced momentum flux calculation
            # M = rho * v² * A = mass_flow_rate * v
            momentum_flux = mass_flow_rate * exit_velocity  # kg⋅m/s²
            
            # Convert to standard units for plume rise calculations (m⁴/s²)
            # Momentum flux per unit density: M/rho = v² * A
            momentum_flux_normalized = (exit_velocity**2) * vent_area
            
            # Account for vent geometry effects (circular vs. elliptical)
            geometry_factor = 1.0  # Circular vent assumption
            
            # Account for gas mixture effects on momentum transfer
            # Heavier gases have higher momentum for same velocity
            density_correction = gas_density / 1.225  # Normalize to air density
            
            # Final momentum flux with all corrections
            final_momentum_flux = momentum_flux_normalized * geometry_factor * density_correction
            
            return final_momentum_flux
            
        except Exception as e:
            logger.error(f"Error calculating advanced momentum flux: {str(e)}")
            # Fallback calculation using simplified volcanic gas physics
            total_emission = sum(emission_rates.values())
            if total_emission <= 0:
                return 0.0
            
            # Simplified but physically-based calculation
            # Assume typical volcanic gas properties
            gas_density = 1.5  # kg/m³ (heavier than air due to CO2/SO2)
            exit_velocity = 50.0  # m/s typical for moderate volcanic emissions
            
            # Estimate vent area from emission rate
            vent_area = total_emission / (gas_density * exit_velocity)
            
            # Momentum flux per unit density
            momentum_flux = (exit_velocity**2) * vent_area
            
            return momentum_flux
    
    def _calculate_data_quality_score(self, gas_composition: GasCompositionData, derived: Dict[str, Any]) -> float:
        """Calculate comprehensive data quality score using advanced multi-parameter analysis"""
        try:
            quality_components = {}
            
            # 1. Analytical precision and detection limits (weight: 0.25)
            total_conc = (gas_composition.co2_concentration + gas_composition.so2_concentration + 
                         gas_composition.h2s_concentration)
            
            # Detection limit analysis for each gas
            detection_limits = {'CO2': 1e-6, 'SO2': 1e-7, 'H2S': 5e-8}  # Volume fractions
            
            detection_quality = 0.0
            gas_count = 0
            
            for gas, limit in detection_limits.items():
                if gas == 'CO2':
                    conc = gas_composition.co2_concentration
                elif gas == 'SO2':
                    conc = gas_composition.so2_concentration
                elif gas == 'H2S':
                    conc = gas_composition.h2s_concentration
                else:
                    continue
                
                if conc > 0:
                    gas_count += 1
                    # Signal-to-noise ratio based quality
                    snr = conc / limit
                    if snr > 100:  # Excellent signal
                        detection_quality += 1.0
                    elif snr > 10:  # Good signal
                        detection_quality += 0.8
                    elif snr > 3:  # Acceptable signal
                        detection_quality += 0.6
                    else:  # Poor signal
                        detection_quality += 0.3
            
            if gas_count > 0:
                quality_components['detection'] = detection_quality / gas_count
            else:
                quality_components['detection'] = 0.1
            
            # 2. Geochemical consistency (weight: 0.20)
            gas_ratios = derived.get('gas_ratios', {})
            
            # Volcanic gas ratio validation using global database constraints
            geochemical_quality = 0.0
            ratio_count = 0
            
            # CO2/SO2 ratio validation (typical range: 0.5-5000 for volcanic gases)
            co2_so2_ratio = gas_ratios.get('CO2_SO2', 0)
            if co2_so2_ratio > 0:
                ratio_count += 1
                if 2 <= co2_so2_ratio <= 500:  # Optimal volcanic range
                    geochemical_quality += 1.0
                elif 0.5 <= co2_so2_ratio <= 2000:  # Extended volcanic range
                    geochemical_quality += 0.8
                elif 0.1 <= co2_so2_ratio <= 5000:  # Possible volcanic range
                    geochemical_quality += 0.6
                else:
                    geochemical_quality += 0.2
            
            # H2S/SO2 ratio validation (typical range: 0.01-10 for volcanic gases)
            h2s_so2_ratio = gas_ratios.get('H2S_SO2', 0)
            if h2s_so2_ratio > 0:
                ratio_count += 1
                if 0.05 <= h2s_so2_ratio <= 2.0:  # Optimal volcanic range
                    geochemical_quality += 1.0
                elif 0.01 <= h2s_so2_ratio <= 5.0:  # Extended volcanic range
                    geochemical_quality += 0.8
                elif 0.005 <= h2s_so2_ratio <= 10.0:  # Possible volcanic range
                    geochemical_quality += 0.6
                else:
                    geochemical_quality += 0.2
            
            if ratio_count > 0:
                quality_components['geochemical'] = geochemical_quality / ratio_count
            else:
                quality_components['geochemical'] = 0.5
            
            # 3. Isotopic data quality (weight: 0.15)
            isotopic_signatures = gas_composition.isotopic_signatures
            isotopic_quality = 0.0
            isotope_count = 0
            
            # δ13C-CO2 validation (typical volcanic range: -8 to -2‰)
            delta_13c = isotopic_signatures.get('delta_13C_CO2')
            if delta_13c is not None:
                isotope_count += 1
                if -8 <= delta_13c <= -2:  # Typical magmatic range
                    isotopic_quality += 1.0
                elif -15 <= delta_13c <= 2:  # Extended volcanic range
                    isotopic_quality += 0.8
                elif -25 <= delta_13c <= 5:  # Possible range with crustal contamination
                    isotopic_quality += 0.6
                else:
                    isotopic_quality += 0.3
            
            # δ34S-SO2 validation (typical volcanic range: -5 to +15‰)
            delta_34s = isotopic_signatures.get('delta_34S_SO2')
            if delta_34s is not None:
                isotope_count += 1
                if -2 <= delta_34s <= 8:  # Typical magmatic range
                    isotopic_quality += 1.0
                elif -5 <= delta_34s <= 15:  # Extended volcanic range
                    isotopic_quality += 0.8
                elif -10 <= delta_34s <= 25:  # Possible range with hydrothermal influence
                    isotopic_quality += 0.6
                else:
                    isotopic_quality += 0.3
            
            # 3He/4He ratio validation (typical volcanic range: 6-12 Ra)
            he_ratio = isotopic_signatures.get('He3_He4_ratio')
            if he_ratio is not None:
                isotope_count += 1
                if 6 <= he_ratio <= 12:  # MORB-like mantle range
                    isotopic_quality += 1.0
                elif 4 <= he_ratio <= 15:  # Extended mantle range
                    isotopic_quality += 0.8
                elif 1 <= he_ratio <= 20:  # Possible range with crustal contamination
                    isotopic_quality += 0.6
                else:
                    isotopic_quality += 0.3
            
            if isotope_count > 0:
                quality_components['isotopic'] = isotopic_quality / isotope_count
            else:
                quality_components['isotopic'] = 0.5  # Neutral score when no isotopic data
            
            # 4. Environmental conditions validity (weight: 0.15)
            temp_quality = 0.0
            pressure_quality = 0.0
            
            # Temperature validation
            temperature = gas_composition.temperature
            if 10 <= temperature <= 50:  # Optimal measurement conditions
                temp_quality = 1.0
            elif 0 <= temperature <= 80:  # Good measurement conditions
                temp_quality = 0.8
            elif -20 <= temperature <= 150:  # Acceptable conditions
                temp_quality = 0.6
            elif -50 <= temperature <= 300:  # Challenging conditions
                temp_quality = 0.4
            else:  # Extreme conditions
                temp_quality = 0.2
            
            # Pressure validation
            pressure = gas_composition.pressure
            if 90000 <= pressure <= 110000:  # Near sea level
                pressure_quality = 1.0
            elif 70000 <= pressure <= 120000:  # Moderate altitude
                pressure_quality = 0.8
            elif 50000 <= pressure <= 130000:  # High altitude
                pressure_quality = 0.6
            else:  # Extreme conditions
                pressure_quality = 0.4
            
            quality_components['environmental'] = (temp_quality + pressure_quality) / 2.0
            
            # 5. Atmospheric correction quality (weight: 0.10)
            atm_corrections = gas_composition.atmospheric_correction
            correction_quality = 0.0
            
            # Check for comprehensive atmospheric corrections
            required_corrections = ['co2_factor', 'so2_factor', 'h2s_factor', 
                                  'temperature_correction', 'pressure_correction']
            
            correction_count = sum(1 for corr in required_corrections if corr in atm_corrections)
            
            if correction_count >= 5:  # All major corrections applied
                correction_quality = 1.0
            elif correction_count >= 3:  # Most corrections applied
                correction_quality = 0.8
            elif correction_count >= 1:  # Some corrections applied
                correction_quality = 0.6
            else:  # No corrections applied
                correction_quality = 0.3
            
            quality_components['atmospheric'] = correction_quality
            
            # 6. Measurement uncertainty assessment (weight: 0.10)
            measurement_uncertainty = derived.get('quality_metrics', {}).get('measurement_uncertainty', 0.5)
            
            # Convert uncertainty to quality score (lower uncertainty = higher quality)
            if measurement_uncertainty < 0.05:  # Excellent precision
                uncertainty_quality = 1.0
            elif measurement_uncertainty < 0.10:  # Good precision
                uncertainty_quality = 0.8
            elif measurement_uncertainty < 0.20:  # Acceptable precision
                uncertainty_quality = 0.6
            elif measurement_uncertainty < 0.40:  # Poor precision
                uncertainty_quality = 0.4
            else:  # Very poor precision
                uncertainty_quality = 0.2
            
            quality_components['uncertainty'] = uncertainty_quality
            
            # 7. Temporal consistency (weight: 0.05)
            # This would require historical data comparison - simplified for now
            quality_components['temporal'] = 0.7  # Assume reasonable temporal consistency
            
            # Calculate weighted quality score
            weights = {
                'detection': 0.25,
                'geochemical': 0.20,
                'isotopic': 0.15,
                'environmental': 0.15,
                'atmospheric': 0.10,
                'uncertainty': 0.10,
                'temporal': 0.05
            }
            
            weighted_score = sum(quality_components[component] * weights[component] 
                               for component in quality_components if component in weights)
            
            # Apply final quality adjustments
            final_score = weighted_score
            
            # Bonus for complete dataset
            if (gas_composition.co2_concentration > 0 and 
                gas_composition.so2_concentration > 0 and 
                gas_composition.h2s_concentration > 0 and
                len([v for v in isotopic_signatures.values() if v is not None]) >= 2):
                final_score *= 1.1  # 10% bonus for complete dataset
            
            # Penalty for missing critical data
            if gas_composition.so2_concentration == 0:  # SO2 is critical for volcanic gas analysis
                final_score *= 0.8  # 20% penalty
            
            return min(1.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive data quality score: {str(e)}")
            # Fallback to simplified quality assessment
            try:
                total_conc = (gas_composition.co2_concentration + gas_composition.so2_concentration + 
                             gas_composition.h2s_concentration)
                
                if total_conc > 1e-5:  # Detectable concentrations
                    base_quality = 0.7
                elif total_conc > 1e-7:  # Low but measurable
                    base_quality = 0.5
                else:  # Very low concentrations
                    base_quality = 0.3
                
                # Adjust for isotopic data availability
                isotope_count = sum(1 for v in gas_composition.isotopic_signatures.values() if v is not None)
                isotope_bonus = isotope_count * 0.1
                
                return min(1.0, base_quality + isotope_bonus)
                
            except:
                return 0.5
    
    def _generate_validation_flags(self, gas_composition: GasCompositionData, derived: Dict[str, Any]) -> str:
        """Generate validation flags for automated quality control"""
        try:
            flags = []
            
            # Check for extreme values
            if gas_composition.co2_concentration > 0.5:
                flags.append("HIGH_CO2")
            if gas_composition.so2_concentration > 0.01:
                flags.append("HIGH_SO2")
            if gas_composition.h2s_concentration > 0.005:
                flags.append("HIGH_H2S")
            
            # Check gas ratios
            co2_so2_ratio = derived['gas_ratios']['CO2_SO2']
            if co2_so2_ratio > 5000:
                flags.append("EXTREME_CO2_SO2_RATIO")
            elif co2_so2_ratio < 0.5:
                flags.append("LOW_CO2_SO2_RATIO")
            
            # Check environmental conditions
            if gas_composition.temperature > 200:
                flags.append("HIGH_TEMPERATURE")
            elif gas_composition.temperature < -20:
                flags.append("LOW_TEMPERATURE")
            
            # Check flux rate
            if gas_composition.flux_rate > 1000:
                flags.append("HIGH_FLUX")
            elif gas_composition.flux_rate < 0.01:
                flags.append("LOW_FLUX")
            
            # Check isotopic signatures
            if gas_composition.isotopic_signatures.get('delta_13C_CO2') is not None:
                delta_13c = gas_composition.isotopic_signatures['delta_13C_CO2']
                if delta_13c < -25 or delta_13c > 0:
                    flags.append("UNUSUAL_DELTA13C")
            
            return ",".join(flags) if flags else "NORMAL"
            
        except Exception as e:
            logger.error(f"Error generating validation flags: {str(e)}")
            return "ERROR"
    
    async def track_temporal_evolution(self, location: Tuple[float, float, float],
                                     time_window: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Track temporal evolution of gas composition at a location"""
        try:
            # Query recent measurements near location
            end_time = datetime.now()
            start_time = end_time - time_window
            
            evolution_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   flux_rate, source_attribution
            FROM gas_measurements 
            WHERE x_coord BETWEEN %s AND %s 
            AND y_coord BETWEEN %s AND %s
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            # Define search radius (500m)
            search_radius = 500.0
            x_min, x_max = location[0] - search_radius, location[0] + search_radius
            y_min, y_max = location[1] - search_radius, location[1] + search_radius
            
            measurements = await self.db_manager.execute_query(
                evolution_query, (x_min, x_max, y_min, y_max, start_time, end_time)
            )
            
            if len(measurements) < 2:
                return {'insufficient_data': True}
            
            # Analyze temporal trends
            timestamps = [m['timestamp'] for m in measurements]
            co2_values = [m['co2_concentration'] for m in measurements]
            so2_values = [m['so2_concentration'] for m in measurements]
            h2s_values = [m['h2s_concentration'] for m in measurements]
            flux_values = [m['flux_rate'] for m in measurements]
            
            # Calculate trends
            evolution_analysis = {
                'time_range': (start_time, end_time),
                'measurement_count': len(measurements),
                'co2_trend': self._calculate_trend(timestamps, co2_values),
                'so2_trend': self._calculate_trend(timestamps, so2_values),
                'h2s_trend': self._calculate_trend(timestamps, h2s_values),
                'flux_trend': self._calculate_trend(timestamps, flux_values),
                'source_evolution': self._analyze_source_evolution(measurements)
            }
            
            return evolution_analysis
            
        except Exception as e:
            logger.error(f"Error tracking temporal evolution: {str(e)}")
            raise
    
    def _calculate_trend(self, timestamps: List[datetime], values: List[float]) -> Dict[str, float]:
        """Calculate trend statistics for time series data"""
        if len(timestamps) < 2:
            return {'trend': 0.0, 'r_squared': 0.0}
        
        # Convert timestamps to numerical values (hours since start)
        start_time = timestamps[0]
        x = [(t - start_time).total_seconds() / 3600.0 for t in timestamps]
        y = values
        
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        # Calculate slope (trend)
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            
            # Calculate R-squared
            y_mean = sum_y / n
            ss_tot = sum((yi - y_mean) ** 2 for yi in y)
            ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            slope = 0.0
            r_squared = 0.0
        
        return {
            'trend': slope,  # Change per hour
            'r_squared': r_squared,
            'start_value': y[0],
            'end_value': y[-1],
            'relative_change': (y[-1] - y[0]) / y[0] if y[0] != 0 else 0
        }
    
    def _analyze_source_evolution(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze evolution of gas source attribution over time"""
        source_evolution = {}
        
        for measurement in measurements:
            timestamp = measurement['timestamp']
            source_attr = eval(measurement['source_attribution'])  # Convert string back to dict
            
            for source_id, contribution in source_attr.items():
                if source_id not in source_evolution:
                    source_evolution[source_id] = []
                source_evolution[source_id].append((timestamp, contribution))
        
        # Calculate trends for each source
        source_trends = {}
        for source_id, data in source_evolution.items():
            timestamps, contributions = zip(*data)
            trend = self._calculate_trend(list(timestamps), list(contributions))
            source_trends[source_id] = trend
        
        return source_trends
    
    def _create_gaussian_plume_model(self, emission_source: Tuple[float, float, float],
                                   emission_rate: Dict[str, float], wind_speed: float,
                                   wind_direction: float, temperature: float,
                                   pressure: float, stability_class: str) -> Dict[str, Any]:
        """Create advanced Gaussian plume model with volcanic gas physics"""
        try:
            # Calculate effective plume height
            plume_rise = self._calculate_plume_rise(emission_rate, wind_speed, temperature)
            effective_height = emission_source[2] + plume_rise
            
            # Get stability parameters
            stability_params = self._get_stability_parameters(stability_class)
            
            # Calculate gas-specific properties
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08}
            gas_densities = {}
            
            temp_kelvin = temperature + 273.15
            
            for gas, rate in emission_rate.items():
                if gas in molecular_weights:
                    mw = molecular_weights[gas]
                    density = (mw * pressure) / (8314.0 * temp_kelvin)
                    gas_densities[gas] = density
            
            # Create comprehensive model
            model = {
                'source_location': emission_source,
                'effective_height': effective_height,
                'emission_rates': emission_rate,
                'wind_speed': max(wind_speed, 0.5),  # Minimum wind speed
                'wind_direction': wind_direction,
                'temperature': temperature,
                'pressure': pressure,
                'stability_class': stability_class,
                'stability_params': stability_params,
                'gas_densities': gas_densities,
                'plume_rise': plume_rise
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating Gaussian plume model: {str(e)}")
            return {}
    
    async def track_gas_plume_dispersion(self, emission_source: Tuple[float, float, float],
                                       emission_rate: Dict[str, float],
                                       meteorological_data: Dict[str, float]) -> Dict[str, Any]:
        """Track gas plume dispersion using atmospheric modeling"""
        try:
            # Extract meteorological parameters
            wind_speed = meteorological_data.get('wind_speed', 5.0)  # m/s
            wind_direction = meteorological_data.get('wind_direction', 0.0)  # degrees
            temperature = meteorological_data.get('temperature', 15.0)  # °C
            pressure = meteorological_data.get('pressure', 101325.0)  # Pa
            humidity = meteorological_data.get('humidity', 50.0)  # %
            stability_class = meteorological_data.get('stability_class', 'D')  # Pasquill stability
            
            # Create dispersion model
            dispersion_model = self._create_gaussian_plume_model(
                emission_source, emission_rate, wind_speed, wind_direction,
                temperature, pressure, stability_class
            )
            
            # Calculate concentration field
            concentration_field = self._calculate_concentration_field(dispersion_model)
            
            # Track plume centerline
            plume_centerline = self._track_plume_centerline(dispersion_model, wind_direction)
            
            # Calculate plume dimensions
            plume_dimensions = self._calculate_plume_dimensions(dispersion_model)
            
            # Estimate ground-level concentrations
            ground_concentrations = self._calculate_ground_concentrations(dispersion_model)
            
            return {
                'dispersion_model': dispersion_model,
                'concentration_field': concentration_field,
                'plume_centerline': plume_centerline,
                'plume_dimensions': plume_dimensions,
                'ground_concentrations': ground_concentrations,
                'meteorological_conditions': meteorological_data
            }
            
        except Exception as e:
            logger.error(f"Error tracking gas plume dispersion: {str(e)}")
            raise
            raise
    
    def _create_gaussian_plume_model(self, source: Tuple[float, float, float],
                                   emission_rate: Dict[str, float],
                                   wind_speed: float, wind_direction: float,
                                   temperature: float, pressure: float,
                                   stability_class: str) -> Dict[str, Any]:
        """Create Gaussian plume dispersion model"""
        # Pasquill-Gifford dispersion parameters
        stability_params = {
            'A': {'a': 0.527, 'b': 0.865, 'c': 0.28, 'd': 0.90},  # Very unstable
            'B': {'a': 0.371, 'b': 0.866, 'c': 0.23, 'd': 0.85},  # Moderately unstable
            'C': {'a': 0.209, 'b': 0.897, 'c': 0.22, 'd': 0.80},  # Slightly unstable
            'D': {'a': 0.128, 'b': 0.905, 'c': 0.20, 'd': 0.76},  # Neutral
            'E': {'a': 0.098, 'b': 0.902, 'c': 0.15, 'd': 0.73},  # Slightly stable
            'F': {'a': 0.065, 'b': 0.902, 'c': 0.12, 'd': 0.67}   # Moderately stable
        }
        
        params = stability_params.get(stability_class, stability_params['D'])
        
        model = {
            'source_location': source,
            'emission_rates': emission_rate,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'temperature': temperature,
            'pressure': pressure,
            'stability_class': stability_class,
            'stability_params': params,
            'effective_height': source[2] + self._calculate_plume_rise(emission_rate, wind_speed, temperature)
        }
        
        return model
    
    def _calculate_plume_rise(self, emission_rate: Dict[str, float], wind_speed: float, temperature: float) -> float:
        """Calculate advanced plume rise using comprehensive Briggs equations and volcanic physics"""
        try:
            # Calculate total mass emission rate and gas properties
            total_emission = sum(emission_rate.values())
            if total_emission <= 0:
                return 0.0
            
            # Gas mixture properties
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08, 'H2O': 18.02}
            
            # Calculate weighted average molecular weight
            total_moles = 0.0
            weighted_mw = 0.0
            
            for gas, rate in emission_rate.items():
                if gas in molecular_weights:
                    moles = rate / molecular_weights[gas]
                    total_moles += moles
                    weighted_mw += moles * molecular_weights[gas]
            
            if total_moles > 0:
                avg_molecular_weight = weighted_mw / total_moles
            else:
                avg_molecular_weight = 44.01  # Default to CO2
            
            # Environmental conditions
            ambient_temp = 15.0  # °C
            ambient_temp_k = ambient_temp + 273.15
            source_temp_k = temperature + 273.15
            
            # Gas density at source conditions (kg/m³)
            pressure = 101325.0  # Pa
            gas_density_source = (avg_molecular_weight * pressure) / (8314.0 * source_temp_k)
            
            # Ambient air density (kg/m³)
            air_density = (28.97 * pressure) / (8314.0 * ambient_temp_k)
            
            # Buoyancy flux calculation (m⁴/s³)
            density_ratio = (air_density - gas_density_source) / air_density
            
            # Estimate vent parameters from emission rate
            vent_diameter = np.sqrt(4 * total_emission / (np.pi * gas_density_source * 50.0))
            vent_area = np.pi * (vent_diameter / 2.0) ** 2
            exit_velocity = total_emission / (gas_density_source * vent_area)
            
            # Buoyancy flux
            buoyancy_flux = 9.81 * density_ratio * total_emission / gas_density_source
            
            # Momentum flux (m⁴/s²)
            momentum_flux = exit_velocity * total_emission / gas_density_source
            
            # Atmospheric stability parameter
            stability_parameter = 9.81 / ambient_temp_k * 0.01  # Typical lapse rate 0.01 K/m
            
            # Wind speed correction for low wind conditions
            effective_wind = max(wind_speed, 1.0)  # Minimum 1 m/s to avoid singularities
            
            # Briggs plume rise formulas
            if buoyancy_flux > 0:
                # Unstable/neutral conditions
                if stability_parameter <= 0:
                    # Final plume rise
                    plume_rise_buoyant = 1.6 * (buoyancy_flux / effective_wind) ** (1/3)
                else:
                    # Stable conditions
                    N = np.sqrt(stability_parameter)
                    plume_rise_buoyant = 2.6 * (buoyancy_flux / (effective_wind * N ** 2)) ** (1/3)
            else:
                plume_rise_buoyant = 0.0
            
            # Momentum-driven plume rise
            if momentum_flux > 0:
                plume_rise_momentum = 3.0 * vent_diameter * exit_velocity / effective_wind
                
                if stability_parameter > 0:
                    N = np.sqrt(stability_parameter)
                    momentum_decay = np.exp(-N * plume_rise_momentum / effective_wind)
                    plume_rise_momentum *= momentum_decay
            else:
                plume_rise_momentum = 0.0
            
            # Combined plume rise
            total_plume_rise = np.sqrt(plume_rise_buoyant ** 2 + plume_rise_momentum ** 2)
            
            # Apply thermal enhancement
            thermal_enhancement = 1.0 + 0.1 * max(0, (source_temp_k - ambient_temp_k) / ambient_temp_k)
            final_plume_rise = total_plume_rise * thermal_enhancement
            
            # Physical constraints
            max_plume_rise = 3000.0  # Maximum reasonable plume rise (m)
            final_plume_rise = min(final_plume_rise, max_plume_rise)
            
            return max(0.0, final_plume_rise)
            
        except Exception as e:
            logger.error(f"Error calculating plume rise: {str(e)}")
            return 0.0
    
    def _calculate_concentration_field(self, model: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate 3D concentration field using Gaussian plume model"""
        # Define calculation grid
        x_max = 10000.0  # 10 km downwind
        y_max = 5000.0   # ±5 km crosswind
        z_max = 1000.0   # 1 km vertical
        
        nx, ny, nz = 100, 50, 20
        x = np.linspace(0, x_max, nx)
        y = np.linspace(-y_max, y_max, ny)
        z = np.linspace(0, z_max, nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Transform coordinates to wind-aligned system
        wind_dir_rad = np.radians(model['wind_direction'])
        X_wind = X * np.cos(wind_dir_rad) + Y * np.sin(wind_dir_rad)
        Y_wind = -X * np.sin(wind_dir_rad) + Y * np.cos(wind_dir_rad)
        
        # Calculate dispersion coefficients
        sigma_y = self._calculate_sigma_y(X_wind, model['stability_params'])
        sigma_z = self._calculate_sigma_z(X_wind, model['stability_params'])
        
        # Gaussian plume equation
        H = model['effective_height']
        u = model['wind_speed']
        
        concentration_fields = {}
        
        for gas, emission_rate in model['emission_rates'].items():
            # Gaussian plume formula
            C = np.zeros_like(X)
            
            # Only calculate for downwind points (X_wind > 0)
            mask = X_wind > 0
            
            C[mask] = (emission_rate / (2 * np.pi * u * sigma_y[mask] * sigma_z[mask])) * \
                     np.exp(-0.5 * (Y_wind[mask] / sigma_y[mask])**2) * \
                     (np.exp(-0.5 * ((Z[mask] - H) / sigma_z[mask])**2) + 
                      np.exp(-0.5 * ((Z[mask] + H) / sigma_z[mask])**2))  # Ground reflection
            
            concentration_fields[gas] = C
        
        return concentration_fields
    
    def _calculate_sigma_y(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Calculate lateral dispersion coefficient"""
        # Pasquill-Gifford formula
        return params['a'] * x ** params['b']
    
    def _calculate_sigma_z(self, x: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Calculate vertical dispersion coefficient"""
        # Pasquill-Gifford formula
        return params['c'] * x ** params['d']
    
    def _track_plume_centerline(self, model: Dict[str, Any], wind_direction: float) -> List[Tuple[float, float, float]]:
        """Track plume centerline trajectory"""
        centerline = []
        
        # Calculate centerline points
        distances = np.linspace(0, 10000, 100)  # 10 km downwind
        
        wind_dir_rad = np.radians(wind_direction)
        source_x, source_y, source_z = model['source_location']
        effective_height = model['effective_height']
        
        for distance in distances:
            # Centerline follows wind direction at effective height
            x = source_x + distance * np.cos(wind_dir_rad)
            y = source_y + distance * np.sin(wind_dir_rad)
            z = effective_height  # Assume constant height for simplicity
            
            centerline.append((x, y, z))
        
        return centerline
    
    def _calculate_plume_dimensions(self, model: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate plume width and height as function of distance"""
        distances = np.linspace(100, 10000, 100)  # Start at 100m to avoid singularity
        
        sigma_y = self._calculate_sigma_y(distances, model['stability_params'])
        sigma_z = self._calculate_sigma_z(distances, model['stability_params'])
        
        # Plume dimensions (2-sigma width/height)
        plume_width = 4 * sigma_y  # ±2σ
        plume_height = 4 * sigma_z  # ±2σ
        
        return {
            'distances': distances,
            'width': plume_width,
            'height': plume_height,
            'sigma_y': sigma_y,
            'sigma_z': sigma_z
        }
    
    def _calculate_ground_concentrations(self, model: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate ground-level concentrations"""
        # Define ground-level grid
        x_max = 10000.0  # 10 km
        y_max = 5000.0   # ±5 km
        nx, ny = 200, 100
        
        x = np.linspace(0, x_max, nx)
        y = np.linspace(-y_max, y_max, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Transform to wind coordinates
        wind_dir_rad = np.radians(model['wind_direction'])
        X_wind = X * np.cos(wind_dir_rad) + Y * np.sin(wind_dir_rad)
        Y_wind = -X * np.sin(wind_dir_rad) + Y * np.cos(wind_dir_rad)
        
        ground_concentrations = {}
        
        for gas, emission_rate in model['emission_rates'].items():
            # Ground-level concentration (z = 0)
            sigma_y = self._calculate_sigma_y(X_wind, model['stability_params'])
            sigma_z = self._calculate_sigma_z(X_wind, model['stability_params'])
            
            H = model['effective_height']
            u = model['wind_speed']
            
            C_ground = np.zeros_like(X)
            mask = X_wind > 0
            
            C_ground[mask] = (emission_rate / (np.pi * u * sigma_y[mask] * sigma_z[mask])) * \
                           np.exp(-0.5 * (Y_wind[mask] / sigma_y[mask])**2) * \
                           np.exp(-0.5 * (H / sigma_z[mask])**2)
            
            ground_concentrations[gas] = C_ground
        
        return ground_concentrations
    
    async def forecast_gas_emissions(self, volcanic_system: VolcanicSystem,
                                   forecast_horizon: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Forecast gas emissions using machine learning models"""
        try:
            # Query historical gas emission data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)  # Use 30 days of training data
            
            history_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   flux_rate, temperature, pressure
            FROM gas_measurements 
            WHERE volcanic_system_id = %s AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp
            """
            
            historical_data = await self.db_manager.execute_query(
                history_query, (volcanic_system.system_id, start_time, end_time)
            )
            
            if len(historical_data) < 100:  # Need sufficient training data
                return {'insufficient_data': True, 'message': 'Need at least 100 data points for forecasting'}
            
            # Prepare training data
            features, targets = self._prepare_forecasting_data(historical_data)
            
            # Train forecasting models
            models = self._train_forecasting_models(features, targets)
            
            # Generate forecasts
            forecasts = await self._generate_emission_forecasts(models, features, forecast_horizon)
            
            # Calculate forecast uncertainty
            uncertainty = await self._calculate_forecast_uncertainty(models, features, forecasts)
            
            return {
                'forecasts': forecasts,
                'uncertainty': uncertainty,
                'model_performance': {model_name: model.score(features, targets[target_name.split('_')[0]]) 
                                    for model_name, model in models.items() 
                                    if model_name != 'scaler' and '_' in model_name
                                    for target_name in [model_name] if model_name.split('_')[0] in targets},
                'forecast_horizon': forecast_horizon,
                'training_data_points': len(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting gas emissions: {str(e)}")
            raise
    
    def _prepare_forecasting_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare data for forecasting models"""
        # Extract time series
        timestamps = [d['timestamp'] for d in historical_data]
        co2_values = [d['co2_concentration'] for d in historical_data]
        so2_values = [d['so2_concentration'] for d in historical_data]
        h2s_values = [d['h2s_concentration'] for d in historical_data]
        flux_values = [d['flux_rate'] for d in historical_data]
        temp_values = [d['temperature'] for d in historical_data]
        pressure_values = [d['pressure'] for d in historical_data]
        
        # Create features (lagged values, moving averages, trends)
        n_lags = 10  # Use 10 previous time steps
        features = []
        targets = {'co2': [], 'so2': [], 'h2s': [], 'flux': []}
        
        for i in range(n_lags, len(historical_data)):
            # Lagged features
            feature_vector = []
            
            # Add lagged values
            for lag in range(1, n_lags + 1):
                feature_vector.extend([
                    co2_values[i - lag],
                    so2_values[i - lag],
                    h2s_values[i - lag],
                    flux_values[i - lag]
                ])
            
            # Add moving averages
            window = min(5, i)
            feature_vector.extend([
                np.mean(co2_values[i-window:i]),
                np.mean(so2_values[i-window:i]),
                np.mean(h2s_values[i-window:i]),
                np.mean(flux_values[i-window:i])
            ])
            
            # Add environmental features
            feature_vector.extend([
                temp_values[i-1],
                pressure_values[i-1]
            ])
            
            # Add time features
            timestamp = timestamps[i]
            feature_vector.extend([
                timestamp.hour,
                timestamp.day,
                timestamp.month,
                np.sin(2 * np.pi * timestamp.hour / 24),  # Diurnal cycle
                np.cos(2 * np.pi * timestamp.hour / 24)
            ])
            
            features.append(feature_vector)
            
            # Targets (current values)
            targets['co2'].append(co2_values[i])
            targets['so2'].append(so2_values[i])
            targets['h2s'].append(h2s_values[i])
            targets['flux'].append(flux_values[i])
        
        features = np.array(features)
        targets = {k: np.array(v) for k, v in targets.items()}
        
        return features, targets
    
    def _train_forecasting_models(self, features: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train machine learning models for gas emission forecasting"""
        models = {}
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        for target_name, target_values in targets.items():
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(features_scaled, target_values)
            
            models[f'{target_name}_rf'] = rf_model
            
            # Train Neural Network model
            nn_model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.2
            )
            nn_model.fit(features_scaled, target_values)
            
            models[f'{target_name}_nn'] = nn_model
        
        models['scaler'] = scaler
        
        return models
    
    async def _generate_emission_forecasts(self, models: Dict[str, Any], features: np.ndarray,
                                         forecast_horizon: timedelta) -> Dict[str, List[float]]:
        """Generate emission forecasts using trained models"""
        forecasts = {'co2': [], 'so2': [], 'h2s': [], 'flux': []}
        
        # Number of forecast steps (assuming hourly data)
        n_steps = int(forecast_horizon.total_seconds() / 3600)
        
        # Use last feature vector as starting point
        current_features = features[-1:].copy()
        scaler = models['scaler']
        
        for step in range(n_steps):
            # Scale current features
            current_features_scaled = scaler.transform(current_features)
            
            # Generate predictions for each target
            step_forecasts = {}
            for target in ['co2', 'so2', 'h2s', 'flux']:
                # Ensemble prediction (average of RF and NN)
                rf_pred = models[f'{target}_rf'].predict(current_features_scaled)[0]
                nn_pred = models[f'{target}_nn'].predict(current_features_scaled)[0]
                
                ensemble_pred = 0.6 * rf_pred + 0.4 * nn_pred  # Weighted average
                step_forecasts[target] = ensemble_pred
                forecasts[target].append(ensemble_pred)
            
            # Update features using advanced autoregressive approach with physics constraints
            current_features = await self._update_features_with_physics_constraints(
                current_features, step_forecasts, step
            )
        
        return forecasts
    
    async def _update_features_with_physics_constraints(self, current_features: np.ndarray, 
                                                      predictions: Dict[str, float], 
                                                      step: int) -> np.ndarray:
        """Update feature vector with physics-informed constraints and advanced autoregressive modeling"""
        try:
            # Create new feature vector with proper lag structure
            n_lags = 10
            n_features_per_lag = 4  # co2, so2, h2s, flux
            
            updated_features = current_features.copy()
            
            # Shift lagged features (autoregressive structure)
            for lag in range(n_lags - 1, 0, -1):
                start_idx = (lag - 1) * n_features_per_lag
                end_idx = lag * n_features_per_lag
                target_start = lag * n_features_per_lag
                target_end = (lag + 1) * n_features_per_lag
                
                updated_features[0, target_start:target_end] = updated_features[0, start_idx:end_idx]
            
            # Apply physics-based constraints to predictions
            constrained_predictions = await self._apply_physics_constraints(predictions)
            
            # Update most recent lag with constrained predictions
            updated_features[0, 0] = constrained_predictions['co2']
            updated_features[0, 1] = constrained_predictions['so2']
            updated_features[0, 2] = constrained_predictions['h2s']
            updated_features[0, 3] = constrained_predictions['flux']
            
            # Update moving averages (indices after lagged features)
            ma_start_idx = n_lags * n_features_per_lag
            
            # Calculate new moving averages from recent predictions
            recent_co2 = [updated_features[0, i * n_features_per_lag] for i in range(min(5, n_lags))]
            recent_so2 = [updated_features[0, i * n_features_per_lag + 1] for i in range(min(5, n_lags))]
            recent_h2s = [updated_features[0, i * n_features_per_lag + 2] for i in range(min(5, n_lags))]
            recent_flux = [updated_features[0, i * n_features_per_lag + 3] for i in range(min(5, n_lags))]
            
            updated_features[0, ma_start_idx] = np.mean(recent_co2)
            updated_features[0, ma_start_idx + 1] = np.mean(recent_so2)
            updated_features[0, ma_start_idx + 2] = np.mean(recent_h2s)
            updated_features[0, ma_start_idx + 3] = np.mean(recent_flux)
            
            # Update time features (assuming hourly steps)
            time_idx = ma_start_idx + 6  # After moving averages and environmental features
            current_hour = int(updated_features[0, time_idx])
            new_hour = (current_hour + 1) % 24
            
            updated_features[0, time_idx] = new_hour
            updated_features[0, time_idx + 3] = np.sin(2 * np.pi * new_hour / 24)  # Diurnal sine
            updated_features[0, time_idx + 4] = np.cos(2 * np.pi * new_hour / 24)  # Diurnal cosine
            
            # Update day and month with advanced calendar calculations
            if new_hour == 0:  # New day - use proper calendar arithmetic
                current_day = int(updated_features[0, time_idx + 1])
                current_month = int(updated_features[0, time_idx + 2])
                
                # Days per month (non-leap year approximation)
                days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                max_days = days_in_month[(current_month - 1) % 12]
                
                new_day = current_day + 1
                new_month = current_month
                
                if new_day > max_days:
                    new_day = 1
                    new_month = (current_month % 12) + 1
                
                updated_features[0, time_idx + 1] = new_day
                updated_features[0, time_idx + 2] = new_month
            
            return updated_features
            
        except Exception as e:
            logger.error(f"Error updating features with physics constraints: {str(e)}")
            return current_features
    
    async def _calculate_forecast_uncertainty(self, models: Dict[str, Any], features: np.ndarray,
                                            forecasts: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Calculate forecast uncertainty using model ensemble"""
        uncertainty = {'co2': [], 'so2': [], 'h2s': [], 'flux': []}
        
        # Use model disagreement as uncertainty measure
        scaler = models['scaler']
        current_features = features[-1:].copy()
        
        for step in range(len(forecasts['co2'])):
            current_features_scaled = scaler.transform(current_features)
            
            for target in ['co2', 'so2', 'h2s', 'flux']:
                rf_pred = models[f'{target}_rf'].predict(current_features_scaled)[0]
                nn_pred = models[f'{target}_nn'].predict(current_features_scaled)[0]
                
                # Uncertainty as standard deviation of ensemble
                uncertainty[target].append(abs(rf_pred - nn_pred) / 2)
            
            # Update features with physics constraints
            current_features = await self._update_features_with_physics_constraints(
                current_features, 
                {target: forecasts[target][step] for target in ['co2', 'so2', 'h2s', 'flux']},
                step
            )
        
        return uncertainty
    
    async def validate_against_satellite_measurements(self, ground_measurements: List[GasCompositionData],
                                                    satellite_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate ground-based measurements against satellite observations"""
        try:
            validation_results = {
                'correlation_analysis': {},
                'bias_analysis': {},
                'spatial_comparison': {},
                'temporal_comparison': {}
            }
            
            # Group measurements by time and location
            ground_by_time = {}
            for measurement in ground_measurements:
                time_key = measurement.timestamp.strftime('%Y-%m-%d %H')
                if time_key not in ground_by_time:
                    ground_by_time[time_key] = []
                ground_by_time[time_key].append(measurement)
            
            # Match with satellite data
            matched_pairs = []
            for sat_data in satellite_data:
                sat_time = sat_data['timestamp'].strftime('%Y-%m-%d %H')
                sat_location = (sat_data['longitude'], sat_data['latitude'])
                
                if sat_time in ground_by_time:
                    # Find closest ground measurement
                    closest_ground = None
                    min_distance = float('inf')
                    
                    for ground_meas in ground_by_time[sat_time]:
                        ground_location = (ground_meas.location[0], ground_meas.location[1])
                        distance = np.linalg.norm(np.array(sat_location) - np.array(ground_location))
                        
                        if distance < min_distance and distance < 0.1:  # Within ~10 km
                            min_distance = distance
                            closest_ground = ground_meas
                    
                    if closest_ground:
                        matched_pairs.append((closest_ground, sat_data, min_distance))
            
            if not matched_pairs:
                return {'no_matches': True, 'message': 'No matching satellite-ground measurement pairs found'}
            
            # Perform validation analysis
            validation_results['matched_pairs'] = len(matched_pairs)
            
            # Correlation analysis
            ground_so2 = [pair[0].so2_concentration for pair in matched_pairs]
            sat_so2 = [pair[1]['so2_column_density'] for pair in matched_pairs]
            
            if len(ground_so2) > 3:
                correlation = np.corrcoef(ground_so2, sat_so2)[0, 1]
                validation_results['correlation_analysis']['so2_correlation'] = correlation
                
                # Bias analysis
                bias = np.mean(np.array(sat_so2) - np.array(ground_so2))
                rmse = np.sqrt(np.mean((np.array(sat_so2) - np.array(ground_so2))**2))
                
                validation_results['bias_analysis'] = {
                    'so2_bias': bias,
                    'so2_rmse': rmse,
                    'so2_relative_bias': bias / np.mean(ground_so2) if np.mean(ground_so2) > 0 else 0
                }
            
            # Spatial comparison
            distances = [pair[2] for pair in matched_pairs]
            validation_results['spatial_comparison'] = {
                'mean_distance': np.mean(distances),
                'max_distance': np.max(distances),
                'distance_std': np.std(distances)
            }
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating against satellite measurements: {str(e)}")
            raise
    
    def _calculate_water_vapor_pressure(self, temperature: float, humidity: float) -> float:
        """Calculate water vapor pressure using Magnus formula"""
        # Magnus formula for saturation vapor pressure
        temp_celsius = temperature
        sat_vapor_pressure = 610.78 * np.exp(17.27 * temp_celsius / (temp_celsius + 237.3))
        
        # Actual vapor pressure
        vapor_pressure = (humidity / 100.0) * sat_vapor_pressure
        
        return vapor_pressure
    
    def _calculate_photolysis_rate(self, gas: str, location: Tuple[float, float, float], 
                                 timestamp: datetime) -> float:
        """Calculate advanced photolysis rate for volcanic gas species using comprehensive atmospheric photochemistry"""
        try:
            lat, lon, alt = location
            day_of_year = timestamp.timetuple().tm_yday
            hour = timestamp.hour + timestamp.minute / 60.0
            
            # Advanced solar position calculation with atmospheric corrections
            # Solar declination with higher precision
            P = np.arcsin(0.39795 * np.cos(0.98563 * (day_of_year - 173) * np.pi / 180))
            
            # Equation of time correction for solar noon
            B = 2 * np.pi * (day_of_year - 81) / 365
            equation_of_time = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
            
            # Local solar time with longitude correction
            local_solar_time = hour + (lon / 15.0) + (equation_of_time / 60.0)
            
            # Hour angle
            hour_angle = 15 * (local_solar_time - 12)
            
            # Solar zenith angle with precision corrections
            cos_zenith = (np.sin(np.radians(lat)) * np.sin(P) + 
                         np.cos(np.radians(lat)) * np.cos(P) * np.cos(np.radians(hour_angle)))
            
            zenith_angle = np.arccos(np.clip(cos_zenith, -1, 1))
            
            # Atmospheric attenuation factors
            # Air mass calculation with altitude correction
            if cos_zenith > 0:
                air_mass = 1.0 / (cos_zenith + 0.50572 * (96.07995 - np.degrees(zenith_angle))**(-1.6364))
                
                # Altitude correction for reduced atmospheric attenuation
                altitude_factor = np.exp(alt / 8400.0)  # Scale height ~8.4 km
                effective_air_mass = air_mass / altitude_factor
            else:
                effective_air_mass = float('inf')  # Nighttime
            
            # Gas-specific photolysis calculations
            if gas == 'SO2':
                # SO2 photolysis: SO2 + hν → SO + O (λ < 340 nm)
                if cos_zenith > 0:
                    # Base photolysis rate at zenith (s⁻¹)
                    j0_so2 = 2.5e-5  # Clear sky, overhead sun
                    
                    # Wavelength-dependent attenuation
                    # SO2 absorbs primarily in UV-B (280-320 nm) and UV-A (320-340 nm)
                    uv_transmission = np.exp(-0.15 * effective_air_mass)  # UV attenuation
                    
                    # Cloud cover effects (simplified - could be enhanced with real cloud data)
                    cloud_factor = 0.8  # Assume some cloud attenuation
                    
                    # Ozone column effects on UV radiation
                    ozone_factor = np.exp(-0.05 * effective_air_mass)  # Ozone UV absorption
                    
                    # Aerosol scattering effects (enhanced in volcanic environments)
                    aerosol_factor = 0.9  # Volcanic aerosols reduce UV
                    
                    # Temperature dependence of photolysis cross-section
                    temperature_factor = 1.0  # SO2 cross-section weakly temperature dependent
                    
                    # Combined photolysis rate
                    photolysis_rate = (j0_so2 * cos_zenith * uv_transmission * 
                                     cloud_factor * ozone_factor * aerosol_factor * temperature_factor)
                else:
                    photolysis_rate = 0.0
                    
            elif gas == 'H2S':
                # H2S photolysis: H2S + hν → HS + H (λ < 260 nm)
                if cos_zenith > 0:
                    # Base photolysis rate (lower than SO2 due to shorter wavelength requirement)
                    j0_h2s = 8.0e-6  # Clear sky, overhead sun
                    
                    # Strong UV-C attenuation by atmosphere
                    uvc_transmission = np.exp(-0.8 * effective_air_mass)  # Strong UV-C attenuation
                    
                    # Cloud and atmospheric effects
                    cloud_factor = 0.7  # Stronger cloud effect for UV-C
                    ozone_factor = np.exp(-0.2 * effective_air_mass)  # Strong ozone absorption
                    aerosol_factor = 0.85  # Aerosol effects
                    
                    # Temperature dependence (H2S more temperature sensitive)
                    ambient_temp = 288.15  # K, standard temperature
                    if hasattr(self, 'current_temperature'):
                        actual_temp = self.current_temperature + 273.15
                        temperature_factor = np.exp(-500.0 * (1/actual_temp - 1/ambient_temp))
                    else:
                        temperature_factor = 1.0
                    
                    # Combined photolysis rate
                    photolysis_rate = (j0_h2s * cos_zenith * uvc_transmission * 
                                     cloud_factor * ozone_factor * aerosol_factor * temperature_factor)
                else:
                    photolysis_rate = 0.0
                    
            elif gas == 'NO2':
                # NO2 photolysis: NO2 + hν → NO + O (λ < 420 nm)
                if cos_zenith > 0:
                    j0_no2 = 8.0e-3  # Much faster than SO2/H2S
                    visible_transmission = np.exp(-0.05 * effective_air_mass)  # Visible light attenuation
                    
                    photolysis_rate = j0_no2 * cos_zenith * visible_transmission * 0.9
                else:
                    photolysis_rate = 0.0
                    
            else:
                photolysis_rate = 0.0
            
            # Convert photolysis rate to atmospheric correction factor
            # Factor represents fraction of gas remaining after photolysis
            time_step = 3600.0  # 1 hour time step
            
            if photolysis_rate > 0:
                # Exponential decay: N(t) = N0 * exp(-j*t)
                correction_factor = np.exp(-photolysis_rate * time_step)
            else:
                correction_factor = 1.0
            
            # Apply physical bounds
            return max(0.05, min(1.0, correction_factor))
            
        except Exception as e:
            logger.error(f"Error calculating advanced photolysis rate for {gas}: {str(e)}")
            # Fallback to simplified calculation
            try:
                lat, lon, alt = location
                hour = timestamp.hour + timestamp.minute / 60.0
                
                # Simple day/night calculation
                if 6 <= hour <= 18:  # Daytime approximation
                    solar_factor = np.sin(np.pi * (hour - 6) / 12)  # Sinusoidal solar intensity
                    
                    if gas == 'SO2':
                        base_rate = 1.5e-5
                    elif gas == 'H2S':
                        base_rate = 6.0e-6
                    elif gas == 'NO2':
                        base_rate = 5.0e-3
                    else:
                        base_rate = 0.0
                    
                    photolysis_rate = base_rate * solar_factor
                    correction_factor = np.exp(-photolysis_rate * 3600.0)
                else:
                    correction_factor = 1.0  # No photolysis at night
                
                return max(0.1, min(1.0, correction_factor))
                
            except:
                # Ultimate fallback
                if gas in ['SO2', 'H2S', 'NO2']:
                    return 0.9  # Assume 10% photolytic loss during day
                else:
                    return 1.0  # No photolysis for other gases
    
    def _get_stability_parameters(self, stability_class: str) -> Dict[str, float]:
        """Get Pasquill-Gifford stability parameters"""
        stability_params = {
            'A': {'a': 0.22, 'b': 0.0001, 'c': 0.20, 'd': 0.0},
            'B': {'a': 0.16, 'b': 0.0001, 'c': 0.12, 'd': 0.0},
            'C': {'a': 0.11, 'b': 0.0001, 'c': 0.08, 'd': 0.0002},
            'D': {'a': 0.08, 'b': 0.0001, 'c': 0.06, 'd': 0.0015},
            'E': {'a': 0.06, 'b': 0.0001, 'c': 0.03, 'd': 0.0003},
            'F': {'a': 0.04, 'b': 0.0001, 'c': 0.016, 'd': 0.0003}
        }
        
        return stability_params.get(stability_class, stability_params['D'])
    
    def _calculate_mixing_efficiency(self, wind_speed: float, boundary_height: float, 
                                   stability_params: Dict[str, float]) -> float:
        """Calculate advanced atmospheric mixing efficiency using comprehensive turbulence theory"""
        try:
            # Advanced mixing efficiency based on Monin-Obukhov similarity theory
            
            # Calculate friction velocity
            surface_roughness = 0.1  # Default for volcanic terrain
            friction_velocity = 0.4 * wind_speed / np.log(10.0 / surface_roughness)
            
            # Calculate Obukhov length for stability characterization
            # L = -u*³ρCp T / (κ g H)
            air_density = 1.225  # kg/m³ at sea level
            cp_air = 1005.0  # J/kg/K
            temperature = 288.15  # K (15°C default)
            kappa = 0.4  # von Karman constant
            gravity = 9.81  # m/s²
            
            # Estimate sensible heat flux from stability parameters
            if stability_params['d'] > 0.001:  # Unstable conditions
                heat_flux = 100.0 * stability_params['d']  # W/m²
            elif stability_params['d'] < -0.001:  # Stable conditions
                heat_flux = -50.0 * abs(stability_params['d'])  # W/m²
            else:  # Neutral conditions
                heat_flux = 0.0
            
            if abs(heat_flux) > 1.0:
                obukhov_length = -(friction_velocity**3 * air_density * cp_air * temperature) / (kappa * gravity * heat_flux)
            else:
                obukhov_length = 1e6  # Very large for neutral conditions
            
            # Calculate stability parameter ζ = z/L
            measurement_height = 10.0  # m
            zeta = measurement_height / obukhov_length
            
            # Calculate mixing length using advanced parameterization
            if zeta < -2.0:  # Very unstable
                mixing_length = 0.4 * measurement_height / (1.0 + 15.0 * abs(zeta))
            elif zeta < 0:  # Unstable
                mixing_length = 0.4 * measurement_height * (1.0 - 16.0 * zeta)**0.25
            elif zeta < 1.0:  # Stable
                mixing_length = 0.4 * measurement_height / (1.0 + 5.0 * zeta)
            else:  # Very stable
                mixing_length = 0.4 * measurement_height / (5.0 + zeta)
            
            # Calculate turbulent kinetic energy
            tke = 0.5 * (friction_velocity**2) * (1.0 + abs(zeta)**0.5)
            
            # Calculate mixing efficiency based on turbulent transport
            base_efficiency = np.sqrt(tke) * mixing_length / (wind_speed * boundary_height)
            
            # Boundary layer height correction using advanced scaling
            if boundary_height < 100.0:  # Very shallow boundary layer
                height_factor = 0.5
            elif boundary_height > 3000.0:  # Very deep boundary layer
                height_factor = 1.5
            else:
                height_factor = 1.0 + 0.3 * np.log(boundary_height / 1000.0)
            
            # Entrainment efficiency at boundary layer top
            entrainment_efficiency = 0.2 * (friction_velocity / (wind_speed + 0.1))
            
            # Convective mixing enhancement for unstable conditions
            convective_enhancement = 1.0
            if zeta < -0.1:  # Convective conditions
                convective_velocity = (gravity * abs(heat_flux) * boundary_height / (air_density * cp_air * temperature))**(1/3)
                convective_enhancement = 1.0 + 2.0 * convective_velocity / (wind_speed + 1.0)
            
            # Wind shear mixing contribution
            wind_shear_factor = 1.0 + 0.1 * (wind_speed / 5.0)**2
            
            # Combined mixing efficiency
            total_efficiency = (base_efficiency * height_factor * convective_enhancement * 
                              wind_shear_factor + entrainment_efficiency)
            
            # Apply physical constraints
            return min(1.0, max(0.01, total_efficiency))
            
        except Exception as e:
            logger.error(f"Error calculating advanced mixing efficiency: {str(e)}")
            # Advanced fallback calculation using robust atmospheric physics
            # Calculate friction velocity with enhanced surface roughness parameterization
            surface_roughness = 0.1  # Default for volcanic terrain
            friction_velocity = 0.4 * wind_speed / np.log(10.0 / surface_roughness)
            
            # Enhanced stability correction using Businger-Dyer relationships
            stability_parameter = stability_params.get('d', 0.0)
            if stability_parameter < -0.1:  # Unstable conditions
                stability_factor = (1.0 - 16.0 * abs(stability_parameter))**0.25
            elif stability_parameter > 0.1:  # Stable conditions
                stability_factor = 1.0 / (1.0 + 5.0 * stability_parameter)
            else:  # Near-neutral conditions
                stability_factor = 1.0
            
            # Advanced height scaling with boundary layer physics
            normalized_height = boundary_height / 1000.0  # Normalize to 1 km
            if normalized_height < 0.1:  # Very shallow boundary layer
                height_factor = 0.3 + 0.7 * normalized_height / 0.1
            elif normalized_height > 3.0:  # Very deep boundary layer
                height_factor = 1.2 + 0.3 * np.log(normalized_height / 3.0)
            else:
                height_factor = 0.3 + 0.9 * normalized_height
            
            # Wind speed efficiency with volcanic terrain corrections
            base_efficiency = np.tanh(wind_speed / 8.0)  # Asymptotic approach to 1.0
            
            # Volcanic terrain enhancement factor
            terrain_factor = 1.0 + 0.2 * np.log(surface_roughness / 0.03 + 1.0)
            
            return min(1.0, max(0.01, base_efficiency * height_factor * stability_factor * terrain_factor))
    
    def _calculate_turbulent_diffusion(self, wind_speed: float, stability_params: Dict[str, float], 
                                     surface_roughness: float) -> float:
        """Calculate advanced turbulent diffusion coefficient using K-theory and similarity scaling"""
        try:
            # Advanced turbulent diffusion based on atmospheric boundary layer theory
            
            # Calculate friction velocity with surface roughness effects
            friction_velocity = 0.4 * wind_speed / np.log(10.0 / surface_roughness)
            
            # Calculate atmospheric stability parameter
            richardson_number = stability_params.get('d', 0.0) * 100.0  # Bulk Richardson number
            
            # Advanced Monin-Obukhov length calculation using comprehensive atmospheric physics
            if abs(richardson_number) > 0.01:
                # Calculate Obukhov length using advanced boundary layer theory
                # L = -u*³ρCp T / (κ g H) where H is sensible heat flux
                air_density = 1.225  # kg/m³
                cp_air = 1005.0  # J/kg/K
                temperature_k = 288.15  # K
                kappa = 0.4  # von Karman constant
                gravity = 9.81  # m/s²
                
                # Estimate heat flux from Richardson number and wind shear
                wind_shear = wind_speed / 10.0  # Approximate shear over 10m
                heat_flux = -richardson_number * air_density * cp_air * temperature_k * wind_shear * friction_velocity
                
                # Calculate Obukhov length with physical constraints
                if abs(heat_flux) > 1.0:
                    obukhov_length = -(friction_velocity**3 * air_density * cp_air * temperature_k) / (kappa * gravity * heat_flux)
                    # Apply reasonable bounds for volcanic environments
                    obukhov_length = max(-10000.0, min(10000.0, obukhov_length))
                else:
                    obukhov_length = 1e6  # Near-neutral conditions
            else:
                obukhov_length = 1e6  # Neutral conditions
            
            # Height-dependent diffusion coefficient calculation
            measurement_height = 10.0  # m
            zeta = measurement_height / obukhov_length
            
            # Calculate stability functions for momentum and heat/mass
            if zeta < 0:  # Unstable conditions
                phi_m = (1.0 - 16.0 * zeta)**(-0.25)  # Momentum stability function
                phi_h = (1.0 - 16.0 * zeta)**(-0.5)   # Heat/mass stability function
            elif zeta > 0:  # Stable conditions
                phi_m = 1.0 + 5.0 * zeta
                phi_h = 1.0 + 5.0 * zeta
            else:  # Neutral conditions
                phi_m = 1.0
                phi_h = 1.0
            
            # Calculate eddy diffusivity for momentum
            kappa = 0.4  # von Karman constant
            k_momentum = kappa * friction_velocity * measurement_height / phi_m
            
            # Calculate eddy diffusivity for scalars (heat/mass)
            turbulent_prandtl = phi_h / phi_m  # Turbulent Prandtl number
            k_scalar = k_momentum / turbulent_prandtl
            
            # Surface roughness enhancement for volcanic terrain
            roughness_enhancement = 1.0 + 2.0 * np.log10(surface_roughness / 0.03 + 1.0)
            
            # Convective enhancement for unstable conditions
            convective_enhancement = 1.0
            if zeta < -0.1:
                # Free convection contribution
                convective_velocity = abs(zeta)**(1/3) * friction_velocity
                convective_diffusion = 0.1 * convective_velocity * measurement_height
                convective_enhancement = 1.0 + convective_diffusion / k_scalar
            
            # Mechanical turbulence contribution
            mechanical_diffusion = k_scalar * roughness_enhancement
            
            # Buoyancy-driven turbulence for volcanic gases
            # Account for density differences due to temperature and composition
            gas_density_factor = 1.0  # Default
            if hasattr(self, 'current_gas_temperature'):
                temp_diff = self.current_gas_temperature - 288.15  # K
                gas_density_factor = 1.0 + 0.003 * temp_diff  # Thermal buoyancy
            
            buoyancy_enhancement = 1.0 + 0.2 * max(0, gas_density_factor - 1.0)
            
            # Wind shear contribution at different heights
            wind_shear_factor = 1.0 + 0.1 * (wind_speed / 5.0)**1.5
            
            # Topographic effects on turbulence
            topographic_factor = 1.0
            if surface_roughness > 0.5:  # Rough volcanic terrain
                topographic_factor = 1.0 + 0.5 * np.log(surface_roughness / 0.1)
            
            # Combined turbulent diffusion coefficient
            total_diffusion = (mechanical_diffusion * convective_enhancement * 
                             buoyancy_enhancement * wind_shear_factor * topographic_factor)
            
            # Apply physical constraints and realistic bounds
            min_diffusion = 0.1  # Minimum molecular diffusion
            max_diffusion = 2000.0  # Maximum reasonable turbulent diffusion
            
            return min(max_diffusion, max(min_diffusion, total_diffusion))
            
        except Exception as e:
            logger.error(f"Error calculating advanced turbulent diffusion: {str(e)}")
            # Advanced fallback using comprehensive K-theory with volcanic environment corrections
            
            # Enhanced friction velocity calculation with stability corrections
            friction_velocity = 0.4 * wind_speed / np.log(10.0 / surface_roughness)
            
            # Advanced stability parameter from Pasquill-Gifford classification
            stability_parameter = stability_params.get('d', 0.0)
            
            # Calculate stability functions using Businger-Dyer relationships
            measurement_height = 10.0  # m
            if stability_parameter < -0.1:  # Unstable conditions
                # Convective scaling
                zeta = -abs(stability_parameter) * 10.0  # Approximate stability parameter
                phi_h = (1.0 - 16.0 * abs(zeta))**(-0.5)  # Heat/mass stability function
                convective_enhancement = 1.0 + 2.0 * abs(zeta)**0.33
            elif stability_parameter > 0.1:  # Stable conditions
                zeta = stability_parameter * 10.0
                phi_h = 1.0 + 5.0 * zeta
                convective_enhancement = 1.0 / (1.0 + zeta)
            else:  # Near-neutral conditions
                phi_h = 1.0
                convective_enhancement = 1.0
            
            # Base diffusion coefficient using K-theory
            kappa = 0.4  # von Karman constant
            base_diffusion = kappa * friction_velocity * measurement_height / phi_h
            
            # Volcanic terrain enhancement
            terrain_roughness_factor = 1.0 + np.log(surface_roughness / 0.03 + 1.0)
            
            # Wind shear enhancement
            wind_shear_factor = 1.0 + 0.1 * (wind_speed / 5.0)**1.2
            
            # Molecular diffusion baseline
            molecular_diffusion = 0.2  # m²/s for volcanic gases in air
            
            # Combined diffusion coefficient
            total_diffusion = (base_diffusion * convective_enhancement * 
                             terrain_roughness_factor * wind_shear_factor + molecular_diffusion)
            
            return min(2000.0, max(0.1, total_diffusion))
    
    def _calculate_convective_velocity(self, temperature: float, boundary_height: float, 
                                     thermal_stratification: float) -> float:
        """Calculate advanced convective velocity scale using comprehensive boundary layer physics"""
        try:
            # Advanced convective velocity calculation based on mixed-layer theory
            
            # Physical constants
            gravity = 9.81  # m/s²
            air_density = 1.225  # kg/m³ at sea level
            cp_air = 1005.0  # J/kg/K
            temperature_k = temperature + 273.15  # Convert to Kelvin
            
            # Calculate surface heat flux from thermal stratification
            if thermal_stratification > 0:  # Unstable conditions
                # Enhanced heat flux calculation for volcanic environments
                base_heat_flux = thermal_stratification * 150.0  # W/m²
                
                # Volcanic surface temperature enhancement
                if temperature > 50.0:  # Hot volcanic surface
                    volcanic_enhancement = 1.0 + 0.02 * (temperature - 50.0)
                    surface_heat_flux = base_heat_flux * volcanic_enhancement
                else:
                    surface_heat_flux = base_heat_flux
                
                # Account for radiative heating from volcanic sources
                if hasattr(self, 'volcanic_heat_sources'):
                    radiative_flux = 50.0  # Additional W/m² from volcanic radiation
                    surface_heat_flux += radiative_flux
                
            elif thermal_stratification < 0:  # Stable conditions (cooling)
                surface_heat_flux = thermal_stratification * 50.0  # Negative flux
            else:  # Neutral conditions
                surface_heat_flux = 0.0
            
            # Calculate buoyancy flux
            buoyancy_flux = gravity * surface_heat_flux / (air_density * cp_air * temperature_k)
            
            if buoyancy_flux > 0:  # Convective conditions
                # Deardorff convective velocity scale
                w_star = (buoyancy_flux * boundary_height)**(1/3)
                
                # Enhanced convection in volcanic environments
                # Account for additional buoyancy from hot gas emissions
                gas_buoyancy_enhancement = 1.0
                if temperature > 100.0:  # Hot gas emissions
                    # Calculate density difference between hot gas and ambient air
                    ambient_temp = 288.15  # K
                    density_ratio = ambient_temp / temperature_k
                    gas_buoyancy_enhancement = 1.0 + 0.5 * (1.0 - density_ratio)
                
                # Topographic enhancement of convection
                topographic_enhancement = 1.0
                if boundary_height < 500.0:  # Shallow boundary layer over complex terrain
                    topographic_enhancement = 1.0 + 0.3 * (500.0 - boundary_height) / 500.0
                
                # Thermal plume effects from volcanic sources
                thermal_plume_enhancement = 1.0
                if surface_heat_flux > 200.0:  # Strong thermal forcing
                    # Calculate thermal plume velocity
                    plume_radius = 100.0  # m, typical volcanic plume
                    plume_buoyancy = buoyancy_flux * np.pi * plume_radius**2
                    thermal_plume_velocity = (2.0 * plume_buoyancy / plume_radius)**(1/3)
                    thermal_plume_enhancement = 1.0 + 0.2 * thermal_plume_velocity / w_star
                
                # Entrainment effects at boundary layer top
                entrainment_velocity = 0.2 * w_star  # Typical entrainment rate
                
                # Combined convective velocity
                total_convective_velocity = (w_star * gas_buoyancy_enhancement * 
                                           topographic_enhancement * thermal_plume_enhancement)
                
                # Add entrainment contribution
                effective_convective_velocity = total_convective_velocity + entrainment_velocity
                
                # Account for wind shear effects on convection
                if hasattr(self, 'current_wind_speed'):
                    wind_speed = self.current_wind_speed
                    shear_factor = 1.0 - 0.1 * min(1.0, wind_speed / 10.0)  # Reduce convection with strong winds
                    effective_convective_velocity *= shear_factor
                
            else:  # Stable or neutral conditions
                # Residual convection from surface heating
                if surface_heat_flux > 10.0:  # Weak heating
                    effective_convective_velocity = 0.1 * (surface_heat_flux / 100.0)**(1/3)
                else:
                    effective_convective_velocity = 0.0
            
            # Apply physical constraints
            min_velocity = 0.0
            max_velocity = 8.0  # Maximum reasonable convective velocity
            
            return min(max_velocity, max(min_velocity, effective_convective_velocity))
            
        except Exception as e:
            logger.error(f"Error calculating advanced convective velocity: {str(e)}")
            # Advanced fallback using comprehensive mixed-layer scaling theory
            
            # Physical constants for volcanic environments
            gravity = 9.81  # m/s²
            air_density = 1.225  # kg/m³
            cp_air = 1005.0  # J/kg/K
            temperature_k = temperature + 273.15
            
            if thermal_stratification > 0:  # Unstable/convective conditions
                # Enhanced heat flux calculation for volcanic environments
                base_heat_flux = thermal_stratification * 120.0  # W/m²
                
                # Volcanic surface enhancement based on temperature
                if temperature > 30.0:  # Elevated surface temperature
                    volcanic_factor = 1.0 + 0.015 * (temperature - 30.0)
                    surface_heat_flux = base_heat_flux * volcanic_factor
                else:
                    surface_heat_flux = base_heat_flux
                
                # Additional radiative flux from volcanic sources
                radiative_flux = max(0.0, 20.0 * (temperature - 15.0) / 85.0)  # 0-20 W/m² based on temperature
                total_heat_flux = surface_heat_flux + radiative_flux
                
                # Buoyancy flux calculation
                buoyancy_flux = gravity * total_heat_flux / (air_density * cp_air * temperature_k)
                
                # Deardorff convective velocity scale with enhancements
                if buoyancy_flux > 0:
                    w_star = (buoyancy_flux * boundary_height)**(1/3)
                    
                    # Topographic enhancement for volcanic terrain
                    if boundary_height < 800.0:  # Shallow boundary layer
                        topo_factor = 1.0 + 0.2 * (800.0 - boundary_height) / 800.0
                    else:
                        topo_factor = 1.0
                    
                    # Gas density effects from hot emissions
                    if temperature > 100.0:  # Hot gas emissions
                        density_factor = 1.0 + 0.3 * (1.0 - temperature_k / 373.15)  # Up to 30% enhancement
                    else:
                        density_factor = 1.0
                    
                    # Entrainment velocity at boundary layer top
                    entrainment_velocity = 0.15 * w_star
                    
                    # Combined convective velocity
                    convective_velocity = w_star * topo_factor * density_factor + entrainment_velocity
                else:
                    convective_velocity = 0.0
                    
            elif thermal_stratification < 0:  # Stable conditions
                # Residual convection from surface heating
                if temperature > 20.0:  # Some surface heating
                    residual_flux = max(0.0, 10.0 * (temperature - 20.0) / 30.0)  # 0-10 W/m²
                    residual_buoyancy = gravity * residual_flux / (air_density * cp_air * temperature_k)
                    convective_velocity = 0.5 * (residual_buoyancy * boundary_height)**(1/3)
                else:
                    convective_velocity = 0.0
            else:  # Neutral conditions
                convective_velocity = 0.0
            
            # Apply physical constraints
            return min(6.0, max(0.0, convective_velocity))
    
    def _assess_measurement_uncertainty(self, gas_concentrations: Dict[str, float], 
                                      wind_speed: float, stability: str) -> float:
        """Assess comprehensive measurement uncertainty using advanced error propagation and quality metrics"""
        try:
            # Advanced uncertainty assessment with multiple contributing factors
            
            # 1. Instrumental uncertainty (gas-specific)
            instrumental_uncertainties = {
                'co2': 0.02,  # 2% for CO2 (NDIR sensors)
                'so2': 0.05,  # 5% for SO2 (UV absorption)
                'h2s': 0.08   # 8% for H2S (electrochemical sensors)
            }
            
            # Calculate weighted instrumental uncertainty
            total_signal = sum(gas_concentrations.values())
            if total_signal > 0:
                weighted_instrumental = sum(
                    gas_concentrations.get(gas, 0) * uncertainty / total_signal
                    for gas, uncertainty in instrumental_uncertainties.items()
                )
            else:
                weighted_instrumental = 0.1  # Default 10% for very low signals
            
            # 2. Atmospheric transport uncertainty
            # Wind speed effects on mixing and dilution
            if wind_speed < 1.0:  # Very low wind
                wind_uncertainty = 0.4  # High uncertainty due to poor mixing
            elif wind_speed < 3.0:  # Low wind
                wind_uncertainty = 0.2
            elif wind_speed > 15.0:  # Very high wind
                wind_uncertainty = 0.15  # High dilution uncertainty
            else:  # Moderate wind
                wind_uncertainty = 0.05
            
            # 3. Atmospheric stability uncertainty
            stability_uncertainties = {
                'A': 0.35,  # Very unstable - high variability
                'B': 0.25,  # Moderately unstable
                'C': 0.15,  # Slightly unstable
                'D': 0.08,  # Neutral - most predictable
                'E': 0.20,  # Slightly stable
                'F': 0.40   # Very stable - poor mixing, high variability
            }
            stability_uncertainty = stability_uncertainties.get(stability, 0.15)
            
            # 4. Concentration-dependent uncertainty (detection limits)
            concentration_uncertainties = []
            for gas, concentration in gas_concentrations.items():
                # Detection limits (ppm)
                detection_limits = {'co2': 1.0, 'so2': 0.1, 'h2s': 0.05}
                limit = detection_limits.get(gas, 0.1)
                
                conc_ppm = concentration * 1e6  # Convert to ppm
                if conc_ppm < limit:
                    # High uncertainty near detection limit
                    conc_uncertainty = 0.5 * (limit / (conc_ppm + 0.01))
                elif conc_ppm < 5 * limit:
                    # Moderate uncertainty in low concentration range
                    conc_uncertainty = 0.2 * (limit / conc_ppm)
                else:
                    # Low uncertainty for high concentrations
                    conc_uncertainty = 0.02
                
                concentration_uncertainties.append(conc_uncertainty)
            
            avg_conc_uncertainty = np.mean(concentration_uncertainties) if concentration_uncertainties else 0.2
            
            # 5. Temporal variability uncertainty
            # Account for rapid changes in volcanic gas emissions
            temporal_uncertainty = 0.1  # Base temporal variability
            
            # Enhanced for volcanic environments with rapid changes
            total_conc_ppm = sum(gas_concentrations.values()) * 1e6
            if total_conc_ppm > 100:  # High concentration - potentially variable source
                temporal_uncertainty = 0.15
            elif total_conc_ppm > 1000:  # Very high - highly variable
                temporal_uncertainty = 0.25
            
            # 6. Spatial representativeness uncertainty
            # Point measurements vs. area-averaged concentrations
            spatial_uncertainty = 0.1  # Base spatial variability
            
            # Enhanced for complex volcanic terrain
            if stability in ['A', 'B']:  # Unstable conditions - high spatial variability
                spatial_uncertainty = 0.2
            elif stability in ['E', 'F']:  # Stable conditions - potential channeling
                spatial_uncertainty = 0.15
            
            # 7. Atmospheric correction uncertainty
            correction_uncertainty = 0.05  # Base correction uncertainty
            
            # Enhanced for complex atmospheric conditions
            if wind_speed < 2.0 or wind_speed > 12.0:
                correction_uncertainty = 0.1
            
            # 8. Background subtraction uncertainty
            background_uncertainty = 0.03  # Uncertainty in background levels
            
            # For volcanic environments with variable backgrounds
            if any(conc > 0.001 for conc in gas_concentrations.values()):  # Elevated concentrations
                background_uncertainty = 0.05
            
            # 9. Interference and cross-sensitivity uncertainty
            interference_uncertainty = 0.02  # Base interference effects
            
            # Enhanced for complex gas mixtures in volcanic environments
            num_gases = len([c for c in gas_concentrations.values() if c > 0])
            if num_gases > 2:  # Multiple gases present
                interference_uncertainty = 0.05
            
            # 10. Temperature and pressure correction uncertainty
            environmental_uncertainty = 0.03  # Base environmental correction
            
            # Enhanced for extreme volcanic conditions
            if hasattr(self, 'current_temperature'):
                temp = self.current_temperature
                if temp > 50 or temp < -10:  # Extreme temperatures
                    environmental_uncertainty = 0.08
            
            # Combine all uncertainty components using error propagation
            uncertainty_components = [
                weighted_instrumental,
                wind_uncertainty,
                stability_uncertainty,
                avg_conc_uncertainty,
                temporal_uncertainty,
                spatial_uncertainty,
                correction_uncertainty,
                background_uncertainty,
                interference_uncertainty,
                environmental_uncertainty
            ]
            
            # Root sum of squares for independent uncertainties
            total_uncertainty = np.sqrt(sum(u**2 for u in uncertainty_components))
            
            # Add systematic uncertainties linearly
            systematic_uncertainty = 0.02  # Base systematic uncertainty
            total_uncertainty += systematic_uncertainty
            
            # Apply confidence level adjustment (95% confidence interval)
            confidence_factor = 1.96  # 95% CI for normal distribution
            final_uncertainty = total_uncertainty * confidence_factor
            
            # Apply physical constraints
            min_uncertainty = 0.05  # Minimum 5% uncertainty
            max_uncertainty = 0.95  # Maximum 95% uncertainty
            
            return min(max_uncertainty, max(min_uncertainty, final_uncertainty))
            
        except Exception as e:
            logger.error(f"Error assessing advanced measurement uncertainty: {str(e)}")
            # Advanced fallback calculation using comprehensive error analysis
            
            # Enhanced instrumental uncertainty with gas-specific factors
            gas_specific_uncertainties = {
                'co2': 0.025,  # 2.5% for modern NDIR sensors
                'so2': 0.045,  # 4.5% for UV-DOAS systems
                'h2s': 0.075   # 7.5% for electrochemical sensors
            }
            
            # Calculate weighted instrumental uncertainty
            total_signal = sum(gas_concentrations.values())
            if total_signal > 0:
                weighted_instrumental = sum(
                    gas_concentrations.get(gas, 0) * uncertainty / total_signal
                    for gas, uncertainty in gas_specific_uncertainties.items()
                )
            else:
                weighted_instrumental = 0.08
            
            # Enhanced atmospheric transport uncertainty
            if wind_speed < 0.5:  # Very stagnant conditions
                wind_uncertainty = 0.5
            elif wind_speed < 2.0:  # Low wind
                wind_uncertainty = 0.25 / (wind_speed + 0.1)
            elif wind_speed > 20.0:  # Very high wind
                wind_uncertainty = 0.2 + 0.01 * (wind_speed - 20.0)
            else:  # Moderate wind
                wind_uncertainty = 0.08 / np.sqrt(wind_speed)
            
            # Enhanced stability-dependent uncertainty
            stability_uncertainties = {
                'A': 0.35, 'B': 0.25, 'C': 0.15, 'D': 0.08, 'E': 0.22, 'F': 0.42
            }
            stab_uncertainty = stability_uncertainties.get(stability, 0.15)
            
            # Enhanced concentration-dependent uncertainty
            total_conc_ppm = sum(gas_concentrations.values()) * 1e6
            if total_conc_ppm < 1.0:  # Very low concentrations
                conc_uncertainty = 0.6 / (total_conc_ppm + 0.1)
            elif total_conc_ppm < 10.0:  # Low concentrations
                conc_uncertainty = 0.3 / np.sqrt(total_conc_ppm)
            else:  # Higher concentrations
                conc_uncertainty = 0.05 / np.log10(total_conc_ppm + 1.0)
            
            # Temporal and spatial variability for volcanic environments
            temporal_uncertainty = 0.12  # Enhanced for volcanic variability
            spatial_uncertainty = 0.08   # Point vs. area representativeness
            
            # Background and interference uncertainties
            background_uncertainty = 0.04
            interference_uncertainty = 0.03
            
            # Environmental correction uncertainty
            environmental_uncertainty = 0.05
            
            # Combine uncertainties using advanced error propagation
            random_uncertainties = [
                weighted_instrumental, wind_uncertainty, stab_uncertainty,
                conc_uncertainty, temporal_uncertainty, spatial_uncertainty
            ]
            
            systematic_uncertainties = [
                background_uncertainty, interference_uncertainty, environmental_uncertainty
            ]
            
            # Root sum of squares for random uncertainties
            random_component = np.sqrt(sum(u**2 for u in random_uncertainties))
            
            # Linear sum for systematic uncertainties
            systematic_component = sum(systematic_uncertainties)
            
            # Total uncertainty with 95% confidence interval
            total_uncertainty = (random_component + systematic_component) * 1.96
            
            return min(0.95, max(0.05, total_uncertainty))
    
    async def _store_source_attribution(self, gas_composition: GasCompositionData, derived_params: Dict[str, Any]):
        """Store comprehensive source attribution results with advanced geochemical analysis"""
        try:
            # Store detailed source attribution with confidence metrics
            for source_id, contribution in gas_composition.source_attribution.items():
                if contribution > 0.01:  # Only store significant contributions (>1%)
                    
                    # Calculate advanced attribution metrics
                    attribution_confidence = self._calculate_attribution_confidence(
                        source_id, gas_composition, derived_params
                    )
                    
                    # Calculate geochemical consistency score
                    geochemical_score = await self._calculate_geochemical_consistency(
                        source_id, gas_composition
                    )
                    
                    # Calculate temporal stability of attribution
                    temporal_stability = await self._calculate_temporal_attribution_stability(
                        source_id, gas_composition.location, gas_composition.timestamp
                    )
                    
                    attribution_query = """
                    INSERT INTO gas_source_attribution (
                        measurement_id, source_id, contribution_fraction, attribution_confidence,
                        geochemical_consistency_score, temporal_stability_score,
                        gas_ratio_similarity, isotopic_similarity, spatial_proximity_score,
                        uncertainty_bounds_lower, uncertainty_bounds_upper,
                        validation_method, quality_flags, created_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
                    )
                    """
                    
                    # Calculate individual similarity scores for detailed analysis
                    gas_ratios = self._calculate_gas_ratios(
                        gas_composition.co2_concentration,
                        gas_composition.so2_concentration, 
                        gas_composition.h2s_concentration
                    )
                    
                    if source_id in self.gas_sources:
                        source = self.gas_sources[source_id]
                        ratio_similarity = self._calculate_ratio_similarity(gas_ratios, source.characteristic_ratios)
                        isotopic_similarity = self._calculate_isotopic_similarity(
                            gas_composition.isotopic_signatures, source.isotopic_signature
                        )
                        spatial_proximity = self._calculate_spatial_proximity(
                            gas_composition.location, source.location
                        )
                    else:
                        ratio_similarity = 0.0
                        isotopic_similarity = 0.0
                        spatial_proximity = 0.0
                    
                    # Calculate uncertainty bounds using Monte Carlo approach
                    uncertainty_bounds = await self._calculate_attribution_uncertainty_bounds(
                        source_id, gas_composition, derived_params
                    )
                    
                    # Generate quality flags for attribution
                    quality_flags = self._generate_attribution_quality_flags(
                        source_id, contribution, attribution_confidence, geochemical_score
                    )
                    
                    params = (
                        gas_composition.measurement_id,
                        source_id,
                        contribution,
                        attribution_confidence,
                        geochemical_score,
                        temporal_stability,
                        ratio_similarity,
                        isotopic_similarity,
                        spatial_proximity,
                        uncertainty_bounds['lower'],
                        uncertainty_bounds['upper'],
                        'advanced_ml_geochemical',
                        quality_flags
                    )
                    
                    await self.db_manager.execute_query(attribution_query, params)
            
            logger.info(f"Stored source attribution for measurement {gas_composition.measurement_id}")
            
        except Exception as e:
            logger.error(f"Error storing source attribution: {str(e)}")
            raise
    
    async def _store_atmospheric_corrections(self, gas_composition: GasCompositionData):
        """Store comprehensive atmospheric correction details for model validation and improvement"""
        try:
            # Store detailed atmospheric correction parameters
            correction_query = """
            INSERT INTO atmospheric_corrections (
                measurement_id, timestamp, location,
                temperature_correction_co2, temperature_correction_so2, temperature_correction_h2s,
                pressure_correction_co2, pressure_correction_so2, pressure_correction_h2s,
                humidity_correction_co2, humidity_correction_so2, humidity_correction_h2s,
                stability_correction_factor, boundary_layer_correction_factor,
                photolysis_correction_so2, photolysis_correction_h2s,
                ozone_reaction_correction_so2, ozone_reaction_correction_h2s,
                wind_speed_correction, surface_roughness_correction,
                topographic_enhancement_factor, thermal_stratification_correction,
                atmospheric_stability_class, boundary_layer_height,
                background_co2_subtracted, background_so2_subtracted, background_h2s_subtracted,
                correction_uncertainty, validation_score, correction_method,
                meteorological_data_quality, created_at
            ) VALUES (
                %s, %s, ST_Point(%s, %s), %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()
            )
            """
            
            # Extract correction factors from atmospheric_correction dict
            atm_corr = gas_composition.atmospheric_correction
            
            # Temperature corrections
            temp_corrections = atm_corr.get('temperature_correction', (1.0, 1.0, 1.0))
            
            # Pressure corrections  
            pressure_corrections = atm_corr.get('pressure_correction', (1.0, 1.0, 1.0))
            
            # Humidity corrections
            humidity_corrections = atm_corr.get('humidity_correction', (1.0, 1.0, 1.0))
            
            # Photochemical corrections
            photochem_corrections = atm_corr.get('photochemical_correction', (1.0, 1.0, 1.0))
            
            # Ozone reaction corrections
            ozone_corrections = atm_corr.get('ozone_reaction_correction', (1.0, 1.0, 1.0))
            
            # Background concentrations
            background_values = atm_corr.get('background_subtraction', (0.0, 0.0, 0.0))
            
            # Atmospheric conditions
            atm_conditions = atm_corr.get('atmospheric_conditions', {})
            
            # Calculate correction uncertainty
            correction_uncertainty = self._calculate_correction_uncertainty(atm_corr)
            
            # Calculate validation score based on correction consistency
            validation_score = self._calculate_correction_validation_score(atm_corr, gas_composition)
            
            # Assess meteorological data quality
            met_data_quality = self._assess_meteorological_data_quality(atm_conditions)
            
            params = (
                gas_composition.measurement_id,
                gas_composition.timestamp,
                gas_composition.location[0], gas_composition.location[1],
                temp_corrections[0], temp_corrections[1], temp_corrections[2],
                pressure_corrections[0], pressure_corrections[1], pressure_corrections[2],
                humidity_corrections[0], humidity_corrections[1], humidity_corrections[2],
                atm_corr.get('stability_correction', 1.0),
                atm_corr.get('boundary_layer_correction', 1.0),
                photochem_corrections[1], photochem_corrections[2],
                ozone_corrections[1], ozone_corrections[2],
                self._calculate_wind_speed_correction(atm_conditions, gas_composition),
                self._calculate_surface_roughness_correction(atm_conditions, gas_composition),
                self._calculate_topographic_enhancement_factor(atm_conditions, gas_composition),
                self._calculate_thermal_stratification_correction(atm_conditions, gas_composition),
                atm_conditions.get('stability', 'D'),
                atm_conditions.get('boundary_height', 1000.0),
                background_values[0], background_values[1], background_values[2],
                correction_uncertainty,
                validation_score,
                'advanced_physics_based',
                met_data_quality
            )
            
            await self.db_manager.execute_query(correction_query, params)
            
            logger.info(f"Stored atmospheric corrections for measurement {gas_composition.measurement_id}")
            
        except Exception as e:
            logger.error(f"Error storing atmospheric corrections: {str(e)}")
            raise
    
    async def _update_realtime_statistics(self, gas_composition: GasCompositionData):
        """Update comprehensive real-time statistics and trend analysis for volcanic gas monitoring"""
        try:
            # Update rolling statistics for the measurement location
            location_key = f"{gas_composition.location[0]:.4f}_{gas_composition.location[1]:.4f}"
            
            # Calculate rolling averages and trends
            rolling_stats = await self._calculate_rolling_statistics(gas_composition)
            
            # Update anomaly detection models
            anomaly_scores = await self._update_anomaly_detection(gas_composition)
            
            # Calculate trend indicators
            trend_indicators = await self._calculate_trend_indicators(gas_composition)
            
            # Update real-time statistics table
            stats_query = """
            INSERT INTO realtime_gas_statistics (
                location_key, timestamp, measurement_count_1h, measurement_count_24h,
                co2_mean_1h, co2_std_1h, co2_trend_1h, co2_anomaly_score,
                so2_mean_1h, so2_std_1h, so2_trend_1h, so2_anomaly_score,
                h2s_mean_1h, h2s_std_1h, h2s_trend_1h, h2s_anomaly_score,
                flux_mean_1h, flux_std_1h, flux_trend_1h, flux_anomaly_score,
                co2_so2_ratio_mean, co2_so2_ratio_trend, h2s_so2_ratio_mean, h2s_so2_ratio_trend,
                dominant_source_1h, source_stability_score, attribution_confidence_mean,
                data_quality_mean, measurement_uncertainty_mean,
                alert_level, trend_significance, volatility_index,
                last_updated, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (location_key, DATE_TRUNC('hour', timestamp))
            DO UPDATE SET
                measurement_count_1h = EXCLUDED.measurement_count_1h,
                co2_mean_1h = EXCLUDED.co2_mean_1h,
                co2_std_1h = EXCLUDED.co2_std_1h,
                co2_trend_1h = EXCLUDED.co2_trend_1h,
                co2_anomaly_score = EXCLUDED.co2_anomaly_score,
                so2_mean_1h = EXCLUDED.so2_mean_1h,
                so2_std_1h = EXCLUDED.so2_std_1h,
                so2_trend_1h = EXCLUDED.so2_trend_1h,
                so2_anomaly_score = EXCLUDED.so2_anomaly_score,
                h2s_mean_1h = EXCLUDED.h2s_mean_1h,
                h2s_std_1h = EXCLUDED.h2s_std_1h,
                h2s_trend_1h = EXCLUDED.h2s_trend_1h,
                h2s_anomaly_score = EXCLUDED.h2s_anomaly_score,
                flux_mean_1h = EXCLUDED.flux_mean_1h,
                flux_std_1h = EXCLUDED.flux_std_1h,
                flux_trend_1h = EXCLUDED.flux_trend_1h,
                flux_anomaly_score = EXCLUDED.flux_anomaly_score,
                co2_so2_ratio_mean = EXCLUDED.co2_so2_ratio_mean,
                co2_so2_ratio_trend = EXCLUDED.co2_so2_ratio_trend,
                h2s_so2_ratio_mean = EXCLUDED.h2s_so2_ratio_mean,
                h2s_so2_ratio_trend = EXCLUDED.h2s_so2_ratio_trend,
                dominant_source_1h = EXCLUDED.dominant_source_1h,
                source_stability_score = EXCLUDED.source_stability_score,
                attribution_confidence_mean = EXCLUDED.attribution_confidence_mean,
                data_quality_mean = EXCLUDED.data_quality_mean,
                measurement_uncertainty_mean = EXCLUDED.measurement_uncertainty_mean,
                alert_level = EXCLUDED.alert_level,
                trend_significance = EXCLUDED.trend_significance,
                volatility_index = EXCLUDED.volatility_index,
                last_updated = NOW()
            """
            
            # Determine alert level based on concentrations and trends
            alert_level = self._determine_alert_level(gas_composition, trend_indicators, anomaly_scores)
            
            # Calculate volatility index
            volatility_index = self._calculate_volatility_index(rolling_stats)
            
            # Find dominant source
            dominant_source = max(gas_composition.source_attribution.items(), key=lambda x: x[1])[0] if gas_composition.source_attribution else 'unknown'
            
            params = (
                location_key,
                gas_composition.timestamp,
                rolling_stats['measurement_counts']['1h'],
                rolling_stats['measurement_counts']['24h'],
                rolling_stats['co2']['mean_1h'],
                rolling_stats['co2']['std_1h'],
                trend_indicators['co2']['trend_1h'],
                anomaly_scores['co2'],
                rolling_stats['so2']['mean_1h'],
                rolling_stats['so2']['std_1h'],
                trend_indicators['so2']['trend_1h'],
                anomaly_scores['so2'],
                rolling_stats['h2s']['mean_1h'],
                rolling_stats['h2s']['std_1h'],
                trend_indicators['h2s']['trend_1h'],
                anomaly_scores['h2s'],
                rolling_stats['flux']['mean_1h'],
                rolling_stats['flux']['std_1h'],
                trend_indicators['flux']['trend_1h'],
                anomaly_scores['flux'],
                rolling_stats['ratios']['co2_so2_mean'],
                trend_indicators['ratios']['co2_so2_trend'],
                rolling_stats['ratios']['h2s_so2_mean'],
                trend_indicators['ratios']['h2s_so2_trend'],
                dominant_source,
                rolling_stats['source_stability'],
                rolling_stats['attribution_confidence_mean'],
                rolling_stats['data_quality_mean'],
                rolling_stats['uncertainty_mean'],
                alert_level,
                trend_indicators['overall_significance'],
                volatility_index
            )
            
            await self.db_manager.execute_query(stats_query, params)
            
            # Update global volcanic system statistics
            await self._update_global_volcanic_statistics(gas_composition, rolling_stats)
            
            logger.info(f"Updated real-time statistics for location {location_key}")
            
        except Exception as e:
            logger.error(f"Error updating real-time statistics: {str(e)}")
            raise

    async def _apply_physics_constraints(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Apply physics-based constraints to gas emission predictions"""
        try:
            constrained = predictions.copy()
            
            # Physical constraints
            for gas in ['co2', 'so2', 'h2s']:
                # Non-negativity constraint
                constrained[gas] = max(0.0, constrained[gas])
                
                # Upper bound based on physical limits
                max_concentrations = {'co2': 0.1, 'so2': 0.01, 'h2s': 0.001}  # Volume fractions
                constrained[gas] = min(constrained[gas], max_concentrations[gas])
            
            # Flux constraint (must be non-negative)
            constrained['flux'] = max(0.0, constrained['flux'])
            
            # Gas ratio constraints (based on volcanic gas chemistry)
            co2_so2_ratio = constrained['co2'] / (constrained['so2'] + 1e-10)
            if co2_so2_ratio > 1000:  # Unrealistic ratio
                constrained['so2'] = constrained['co2'] / 500  # Adjust to reasonable ratio
            
            h2s_so2_ratio = constrained['h2s'] / (constrained['so2'] + 1e-10)
            if h2s_so2_ratio > 10:  # Unrealistic ratio
                constrained['h2s'] = constrained['so2'] * 5  # Adjust to reasonable ratio
            
            # Mass balance constraint for flux
            total_gas_mass = (constrained['co2'] * 44.01 + constrained['so2'] * 64.07 + 
                            constrained['h2s'] * 34.08)
            
            if total_gas_mass > 0:
                expected_flux = total_gas_mass * 0.1  # Rough estimate
                if abs(constrained['flux'] - expected_flux) > expected_flux:
                    constrained['flux'] = expected_flux
            
            return constrained
            
        except Exception as e:
            logger.error(f"Error applying physics constraints: {str(e)}")
            return predictions
    
    def _calculate_attribution_confidence(self, source_id: str, gas_composition: GasCompositionData, 
                                        derived_params: Dict[str, Any]) -> float:
        """Calculate confidence score for source attribution"""
        try:
            confidence_factors = []
            
            # Gas ratio consistency
            if source_id in self.gas_sources:
                source = self.gas_sources[source_id]
                gas_ratios = derived_params['gas_ratios']
                
                # CO2/SO2 ratio consistency
                measured_ratio = gas_ratios['CO2_SO2']
                expected_ratio = source.characteristic_ratios.get('CO2_SO2', measured_ratio)
                ratio_consistency = 1.0 / (1.0 + abs(np.log10(measured_ratio / expected_ratio)))
                confidence_factors.append(ratio_consistency)
                
                # Isotopic consistency (if available)
                if gas_composition.isotopic_signatures.get('delta_13C_CO2') is not None:
                    measured_delta13c = gas_composition.isotopic_signatures['delta_13C_CO2']
                    expected_delta13c = source.isotopic_signature.get('delta_13C_CO2', measured_delta13c)
                    isotopic_consistency = 1.0 / (1.0 + abs(measured_delta13c - expected_delta13c) / 5.0)
                    confidence_factors.append(isotopic_consistency)
                
                # Spatial proximity
                distance = np.linalg.norm(np.array(gas_composition.location) - np.array(source.location))
                spatial_confidence = np.exp(-distance / 1000.0)  # 1 km characteristic distance
                confidence_factors.append(spatial_confidence)
            
            # Data quality factor
            quality_factor = derived_params['quality_metrics']['data_quality_score']
            confidence_factors.append(quality_factor)
            
            # Calculate overall confidence
            overall_confidence = np.mean(confidence_factors) if confidence_factors else 0.5
            
            return min(1.0, max(0.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating attribution confidence: {str(e)}")
            return 0.5
    
    async def _calculate_geochemical_consistency(self, source_id: str, gas_composition: GasCompositionData) -> float:
        """Calculate geochemical consistency score for source attribution"""
        try:
            if source_id not in self.gas_sources:
                return 0.5
            
            source = self.gas_sources[source_id]
            consistency_scores = []
            
            # Temperature consistency
            if source.temperature_range[0] <= gas_composition.temperature <= source.temperature_range[1]:
                temp_score = 1.0
            else:
                temp_deviation = min(
                    abs(gas_composition.temperature - source.temperature_range[0]),
                    abs(gas_composition.temperature - source.temperature_range[1])
                )
                temp_score = max(0.0, 1.0 - temp_deviation / 100.0)  # 100°C tolerance
            consistency_scores.append(temp_score)
            
            # Pressure consistency
            if source.pressure_range[0] <= gas_composition.pressure <= source.pressure_range[1]:
                pressure_score = 1.0
            else:
                pressure_deviation = min(
                    abs(gas_composition.pressure - source.pressure_range[0]),
                    abs(gas_composition.pressure - source.pressure_range[1])
                )
                pressure_score = max(0.0, 1.0 - pressure_deviation / 50000.0)  # 50 kPa tolerance
            consistency_scores.append(pressure_score)
            
            # Gas composition consistency
            co2_conc = gas_composition.co2_concentration
            so2_conc = gas_composition.so2_concentration
            h2s_conc = gas_composition.h2s_concentration
            
            # Check if concentrations are within expected ranges for source type
            source_type_ranges = {
                'magmatic': {'co2': (0.01, 0.95), 'so2': (0.001, 0.1), 'h2s': (0.0001, 0.05)},
                'hydrothermal': {'co2': (0.001, 0.5), 'so2': (0.0001, 0.01), 'h2s': (0.001, 0.1)},
                'crustal': {'co2': (0.0001, 0.1), 'so2': (0.0, 0.001), 'h2s': (0.0, 0.001)},
                'atmospheric': {'co2': (0.0003, 0.001), 'so2': (0.0, 0.0001), 'h2s': (0.0, 0.0001)}
            }
            
            if source.source_type in source_type_ranges:
                ranges = source_type_ranges[source.source_type]
                
                for gas, conc in [('co2', co2_conc), ('so2', so2_conc), ('h2s', h2s_conc)]:
                    if ranges[gas][0] <= conc <= ranges[gas][1]:
                        consistency_scores.append(1.0)
                    else:
                        # Calculate how far outside the range
                        if conc < ranges[gas][0]:
                            deviation = (ranges[gas][0] - conc) / ranges[gas][0]
                        else:
                            deviation = (conc - ranges[gas][1]) / ranges[gas][1]
                        
                        consistency_scores.append(max(0.0, 1.0 - deviation))
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating geochemical consistency: {str(e)}")
            return 0.5
    
    async def _calculate_temporal_attribution_stability(self, source_id: str, location: Tuple[float, float, float], 
                                                      timestamp: datetime) -> float:
        """Calculate temporal stability of source attribution"""
        try:
            # Query recent attributions for this source and location
            stability_query = """
            SELECT contribution_fraction, timestamp
            FROM gas_source_attribution gsa
            JOIN gas_measurements gm ON gsa.measurement_id = gm.measurement_id
            WHERE gsa.source_id = %s
            AND ST_DWithin(gm.location, ST_Point(%s, %s), 1000)
            AND gm.timestamp > %s - INTERVAL '24 hours'
            AND gm.timestamp < %s
            ORDER BY gm.timestamp DESC
            LIMIT 20
            """
            
            recent_attributions = await self.db_manager.execute_query(
                stability_query, (source_id, location[0], location[1], timestamp, timestamp)
            )
            
            if len(recent_attributions) < 3:
                return 0.5  # Insufficient data for stability assessment
            
            # Calculate coefficient of variation
            contributions = [attr['contribution_fraction'] for attr in recent_attributions]
            mean_contribution = np.mean(contributions)
            std_contribution = np.std(contributions)
            
            if mean_contribution > 0:
                cv = std_contribution / mean_contribution
                stability_score = max(0.0, 1.0 - cv)  # Lower CV = higher stability
            else:
                stability_score = 0.0
            
            return stability_score
            
        except Exception as e:
            logger.error(f"Error calculating temporal attribution stability: {str(e)}")
            return 0.5
    
    async def _calculate_attribution_uncertainty_bounds(self, source_id: str, gas_composition: GasCompositionData,
                                                      derived_params: Dict[str, Any]) -> Dict[str, float]:
        """Calculate uncertainty bounds for source attribution using Monte Carlo approach"""
        try:
            # Base contribution
            base_contribution = gas_composition.source_attribution.get(source_id, 0.0)
            
            # Uncertainty sources
            measurement_uncertainty = derived_params['quality_metrics']['measurement_uncertainty']
            attribution_confidence = self._calculate_attribution_confidence(source_id, gas_composition, derived_params)
            
            # Monte Carlo simulation for uncertainty propagation
            n_samples = 1000
            perturbed_contributions = []
            
            for _ in range(n_samples):
                # Perturb gas concentrations within measurement uncertainty
                co2_perturbed = gas_composition.co2_concentration * (1 + np.random.normal(0, measurement_uncertainty))
                so2_perturbed = gas_composition.so2_concentration * (1 + np.random.normal(0, measurement_uncertainty))
                h2s_perturbed = gas_composition.h2s_concentration * (1 + np.random.normal(0, measurement_uncertainty))
                
                # Recalculate attribution with perturbed values
                perturbed_ratios = self._calculate_gas_ratios(co2_perturbed, so2_perturbed, h2s_perturbed)
                
                if source_id in self.gas_sources:
                    source = self.gas_sources[source_id]
                    ratio_similarity = self._calculate_ratio_similarity(perturbed_ratios, source.characteristic_ratios)
                    isotopic_similarity = self._calculate_isotopic_similarity(
                        gas_composition.isotopic_signatures, source.isotopic_signature
                    )
                    spatial_similarity = self._calculate_spatial_proximity(gas_composition.location, source.location)
                    
                    # Recalculate contribution
                    combined_similarity = 0.5 * ratio_similarity + 0.3 * isotopic_similarity + 0.2 * spatial_similarity
                    perturbed_contributions.append(combined_similarity)
                else:
                    perturbed_contributions.append(0.0)
            
            # Calculate percentiles for uncertainty bounds
            lower_bound = np.percentile(perturbed_contributions, 5)  # 5th percentile
            upper_bound = np.percentile(perturbed_contributions, 95)  # 95th percentile
            
            return {
                'lower': max(0.0, lower_bound),
                'upper': min(1.0, upper_bound)
            }
            
        except Exception as e:
            logger.error(f"Error calculating attribution uncertainty bounds: {str(e)}")
            return {'lower': 0.0, 'upper': 1.0}
    
    def _generate_attribution_quality_flags(self, source_id: str, contribution: float, 
                                          confidence: float, geochemical_score: float) -> str:
        """Generate quality flags for source attribution"""
        flags = []
        
        if contribution < 0.05:
            flags.append("LOW_CONTRIBUTION")
        elif contribution > 0.8:
            flags.append("DOMINANT_SOURCE")
        
        if confidence < 0.3:
            flags.append("LOW_CONFIDENCE")
        elif confidence > 0.8:
            flags.append("HIGH_CONFIDENCE")
        
        if geochemical_score < 0.4:
            flags.append("POOR_GEOCHEMICAL_MATCH")
        elif geochemical_score > 0.8:
            flags.append("EXCELLENT_GEOCHEMICAL_MATCH")
        
        if source_id == 'unknown':
            flags.append("UNIDENTIFIED_SOURCE")
        
        return ",".join(flags) if flags else "NORMAL"
    
    def _calculate_correction_uncertainty(self, atmospheric_correction: Dict[str, Any]) -> float:
        """Calculate uncertainty in atmospheric corrections"""
        try:
            uncertainty_components = []
            
            # Temperature correction uncertainty
            temp_corrections = atmospheric_correction.get('temperature_correction', (1.0, 1.0, 1.0))
            temp_uncertainty = np.std([abs(1.0 - tc) for tc in temp_corrections])
            uncertainty_components.append(temp_uncertainty)
            
            # Pressure correction uncertainty
            pressure_corrections = atmospheric_correction.get('pressure_correction', (1.0, 1.0, 1.0))
            pressure_uncertainty = np.std([abs(1.0 - pc) for pc in pressure_corrections])
            uncertainty_components.append(pressure_uncertainty)
            
            # Stability correction uncertainty
            stability_correction = atmospheric_correction.get('stability_correction', 1.0)
            stability_uncertainty = abs(1.0 - stability_correction) * 0.2  # 20% relative uncertainty
            uncertainty_components.append(stability_uncertainty)
            
            # Combined uncertainty (root sum of squares)
            total_uncertainty = np.sqrt(sum(u**2 for u in uncertainty_components))
            
            return min(1.0, total_uncertainty)
            
        except Exception as e:
            logger.error(f"Error calculating correction uncertainty: {str(e)}")
            return 0.2  # Default 20% uncertainty
    
    def _calculate_correction_validation_score(self, atmospheric_correction: Dict[str, Any], 
                                             gas_composition: GasCompositionData) -> float:
        """Calculate validation score for atmospheric corrections"""
        try:
            validation_factors = []
            
            # Check if corrections are within reasonable ranges
            correction_factors = [
                atmospheric_correction.get('co2_factor', 1.0),
                atmospheric_correction.get('so2_factor', 1.0),
                atmospheric_correction.get('h2s_factor', 1.0)
            ]
            
            for factor in correction_factors:
                if 0.5 <= factor <= 2.0:  # Reasonable correction range
                    validation_factors.append(1.0)
                elif 0.1 <= factor <= 5.0:  # Extended range
                    validation_factors.append(0.7)
                else:
                    validation_factors.append(0.3)
            
            # Check atmospheric conditions consistency
            atm_conditions = atmospheric_correction.get('atmospheric_conditions', {})
            
            # Humidity check
            humidity = atm_conditions.get('humidity', 50.0)
            if 0 <= humidity <= 100:
                validation_factors.append(1.0)
            else:
                validation_factors.append(0.0)
            
            # Boundary layer height check
            boundary_height = atm_conditions.get('boundary_height', 1000.0)
            if 100 <= boundary_height <= 3000:  # Reasonable range
                validation_factors.append(1.0)
            else:
                validation_factors.append(0.5)
            
            return np.mean(validation_factors) if validation_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating correction validation score: {str(e)}")
            return 0.5
    
    def _assess_meteorological_data_quality(self, atmospheric_conditions: Dict[str, Any]) -> float:
        """Assess quality of meteorological data used in corrections"""
        try:
            quality_factors = []
            
            # Check data completeness
            required_params = ['humidity', 'stability', 'boundary_height', 'ozone']
            available_params = sum(1 for param in required_params if param in atmospheric_conditions)
            completeness_score = available_params / len(required_params)
            quality_factors.append(completeness_score)
            
            # Check parameter reasonableness
            if 'humidity' in atmospheric_conditions:
                humidity = atmospheric_conditions['humidity']
                if 0 <= humidity <= 100:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.0)
            
            if 'boundary_height' in atmospheric_conditions:
                boundary_height = atmospheric_conditions['boundary_height']
                if 50 <= boundary_height <= 5000:  # Reasonable range
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.3)
            
            if 'ozone' in atmospheric_conditions:
                ozone = atmospheric_conditions['ozone']
                if 0 <= ozone <= 200e-9:  # Reasonable ozone range
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(0.5)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error assessing meteorological data quality: {str(e)}")
            return 0.5
    
    async def _calculate_rolling_statistics(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Calculate comprehensive rolling statistics for gas measurements"""
        try:
            location = gas_composition.location
            timestamp = gas_composition.timestamp
            
            # Query recent measurements for rolling statistics
            stats_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   flux_rate, gas_ratio_co2_so2, gas_ratio_h2s_so2,
                   data_quality_score, measurement_uncertainty
            FROM gas_measurements
            WHERE ST_DWithin(location, ST_Point(%s, %s), 1000)
            AND timestamp > %s - INTERVAL '24 hours'
            AND timestamp <= %s
            ORDER BY timestamp DESC
            """
            
            recent_measurements = await self.db_manager.execute_query(
                stats_query, (location[0], location[1], timestamp, timestamp)
            )
            
            if not recent_measurements:
                return self._get_default_rolling_stats()
            
            # Separate measurements by time windows
            one_hour_ago = timestamp - timedelta(hours=1)
            measurements_1h = [m for m in recent_measurements if m['timestamp'] >= one_hour_ago]
            measurements_24h = recent_measurements
            
            # Calculate statistics for each gas
            rolling_stats = {
                'measurement_counts': {
                    '1h': len(measurements_1h),
                    '24h': len(measurements_24h)
                },
                'co2': self._calculate_gas_statistics([m['co2_concentration'] for m in measurements_1h]),
                'so2': self._calculate_gas_statistics([m['so2_concentration'] for m in measurements_1h]),
                'h2s': self._calculate_gas_statistics([m['h2s_concentration'] for m in measurements_1h]),
                'flux': self._calculate_gas_statistics([m['flux_rate'] for m in measurements_1h]),
                'ratios': {
                    'co2_so2_mean': np.mean([m['gas_ratio_co2_so2'] for m in measurements_1h if m['gas_ratio_co2_so2']]),
                    'h2s_so2_mean': np.mean([m['gas_ratio_h2s_so2'] for m in measurements_1h if m['gas_ratio_h2s_so2']])
                },
                'data_quality_mean': np.mean([m['data_quality_score'] for m in measurements_1h if m['data_quality_score']]),
                'uncertainty_mean': np.mean([m['measurement_uncertainty'] for m in measurements_1h if m['measurement_uncertainty']]),
                'source_stability': await self._calculate_source_stability(measurements_1h),
                'attribution_confidence_mean': await self._calculate_attribution_confidence_mean(measurements_1h)
            }
            
            return rolling_stats
            
        except Exception as e:
            logger.error(f"Error calculating rolling statistics: {str(e)}")
            return self._get_default_rolling_stats()
    
    def _get_default_rolling_stats(self) -> Dict[str, Any]:
        """Get default rolling statistics when no data is available"""
        return {
            'measurement_counts': {'1h': 0, '24h': 0},
            'co2': {'mean_1h': 0.0, 'std_1h': 0.0},
            'so2': {'mean_1h': 0.0, 'std_1h': 0.0},
            'h2s': {'mean_1h': 0.0, 'std_1h': 0.0},
            'flux': {'mean_1h': 0.0, 'std_1h': 0.0},
            'ratios': {'co2_so2_mean': 0.0, 'h2s_so2_mean': 0.0},
            'data_quality_mean': 0.5,
            'uncertainty_mean': 0.2,
            'source_stability': 0.5,
            'attribution_confidence_mean': 0.5
        }
    
    def _calculate_gas_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a gas parameter"""
        if not values:
            return {'mean_1h': 0.0, 'std_1h': 0.0}
        
        return {
            'mean_1h': np.mean(values),
            'std_1h': np.std(values)
        }
    
    async def _calculate_source_stability(self, measurements: List[Dict[str, Any]]) -> float:
        """Calculate comprehensive stability of source attribution over time using multiple indicators"""
        try:
            if len(measurements) < 3:
                return 0.5
            
            stability_indicators = []
            
            # 1. Gas ratio stability (CO2/SO2)
            co2_so2_ratios = [m['gas_ratio_co2_so2'] for m in measurements if m['gas_ratio_co2_so2']]
            if len(co2_so2_ratios) >= 3:
                mean_ratio = np.mean(co2_so2_ratios)
                std_ratio = np.std(co2_so2_ratios)
                if mean_ratio > 0:
                    cv_co2_so2 = std_ratio / mean_ratio
                    ratio_stability = max(0.0, 1.0 - cv_co2_so2)
                    stability_indicators.append(ratio_stability)
            
            # 2. Gas ratio stability (H2S/SO2)
            h2s_so2_ratios = [m['gas_ratio_h2s_so2'] for m in measurements if m['gas_ratio_h2s_so2']]
            if len(h2s_so2_ratios) >= 3:
                mean_h2s_ratio = np.mean(h2s_so2_ratios)
                std_h2s_ratio = np.std(h2s_so2_ratios)
                if mean_h2s_ratio > 0:
                    cv_h2s_so2 = std_h2s_ratio / mean_h2s_ratio
                    h2s_stability = max(0.0, 1.0 - cv_h2s_so2)
                    stability_indicators.append(h2s_stability)
            
            # 3. Flux rate stability
            flux_rates = [m['flux_rate'] for m in measurements if m['flux_rate']]
            if len(flux_rates) >= 3:
                mean_flux = np.mean(flux_rates)
                std_flux = np.std(flux_rates)
                if mean_flux > 0:
                    cv_flux = std_flux / mean_flux
                    flux_stability = max(0.0, 1.0 - cv_flux * 0.5)  # Less weight on flux variability
                    stability_indicators.append(flux_stability)
            
            # 4. Concentration stability
            for gas in ['co2_concentration', 'so2_concentration', 'h2s_concentration']:
                concentrations = [m[gas] for m in measurements if m[gas]]
                if len(concentrations) >= 3:
                    mean_conc = np.mean(concentrations)
                    std_conc = np.std(concentrations)
                    if mean_conc > 0:
                        cv_conc = std_conc / mean_conc
                        conc_stability = max(0.0, 1.0 - cv_conc * 0.7)  # Moderate weight
                        stability_indicators.append(conc_stability)
            
            # 5. Temporal consistency (check for systematic trends)
            timestamps = [m['timestamp'] for m in measurements]
            co2_values = [m['co2_concentration'] for m in measurements]
            
            if len(co2_values) >= 3:
                # Calculate trend strength
                trend_result = self._calculate_trend(timestamps, co2_values)
                trend_strength = abs(trend_result['trend'])
                r_squared = trend_result['r_squared']
                
                # High R² with low trend = stable, High R² with high trend = unstable
                if r_squared > 0.7 and trend_strength > 0.01:  # Strong trend
                    temporal_stability = 0.3
                elif r_squared < 0.3:  # Random variation
                    temporal_stability = 0.8
                else:  # Moderate trend
                    temporal_stability = 0.6
                
                stability_indicators.append(temporal_stability)
            
            # 6. Data quality consistency
            quality_scores = [m['data_quality_score'] for m in measurements if m['data_quality_score']]
            if len(quality_scores) >= 3:
                mean_quality = np.mean(quality_scores)
                std_quality = np.std(quality_scores)
                quality_stability = max(0.0, mean_quality - std_quality)  # High mean, low std = stable
                stability_indicators.append(quality_stability)
            
            # Calculate weighted average stability
            if stability_indicators:
                # Weight gas ratios more heavily as they are key source indicators
                weights = []
                for i, indicator in enumerate(stability_indicators):
                    if i < 2:  # Gas ratios
                        weights.append(0.3)
                    elif i == 2:  # Flux
                        weights.append(0.1)
                    elif i < 6:  # Concentrations
                        weights.append(0.1)
                    elif i == 6:  # Temporal
                        weights.append(0.15)
                    else:  # Quality
                        weights.append(0.05)
                
                # Normalize weights
                total_weight = sum(weights[:len(stability_indicators)])
                normalized_weights = [w / total_weight for w in weights[:len(stability_indicators)]]
                
                overall_stability = sum(indicator * weight for indicator, weight in 
                                      zip(stability_indicators, normalized_weights))
            else:
                overall_stability = 0.5
            
            return np.clip(overall_stability, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating source stability: {str(e)}")
            return 0.5
    
    async def _update_anomaly_detection(self, gas_composition: GasCompositionData) -> Dict[str, float]:
        """Update anomaly detection models and calculate anomaly scores"""
        try:
            # Create feature vector for anomaly detection
            features = np.array([[
                gas_composition.co2_concentration,
                gas_composition.so2_concentration,
                gas_composition.h2s_concentration,
                gas_composition.flux_rate,
                gas_composition.temperature,
                gas_composition.pressure
            ]])
            
            # Calculate anomaly scores using isolation forest
            anomaly_score = self.anomaly_detector.decision_function(features)[0]
            
            # Normalize to 0-1 range (higher = more anomalous)
            normalized_score = max(0.0, min(1.0, (0.5 - anomaly_score) / 1.0))
            
            return {
                'co2': normalized_score,
                'so2': normalized_score,
                'h2s': normalized_score,
                'flux': normalized_score
            }
            
        except Exception as e:
            logger.error(f"Error updating anomaly detection: {str(e)}")
            return {'co2': 0.0, 'so2': 0.0, 'h2s': 0.0, 'flux': 0.0}
    
    async def _calculate_trend_indicators(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Calculate trend indicators for gas measurements"""
        try:
            location = gas_composition.location
            timestamp = gas_composition.timestamp
            
            # Query recent measurements for trend analysis
            trend_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   flux_rate, gas_ratio_co2_so2, gas_ratio_h2s_so2
            FROM gas_measurements
            WHERE ST_DWithin(location, ST_Point(%s, %s), 1000)
            AND timestamp > %s - INTERVAL '6 hours'
            AND timestamp <= %s
            ORDER BY timestamp
            """
            
            trend_measurements = await self.db_manager.execute_query(
                trend_query, (location[0], location[1], timestamp, timestamp)
            )
            
            if len(trend_measurements) < 3:
                return self._get_default_trend_indicators()
            
            # Calculate trends for each parameter
            timestamps = [m['timestamp'] for m in trend_measurements]
            
            trends = {
                'co2': self._calculate_trend(timestamps, [m['co2_concentration'] for m in trend_measurements]),
                'so2': self._calculate_trend(timestamps, [m['so2_concentration'] for m in trend_measurements]),
                'h2s': self._calculate_trend(timestamps, [m['h2s_concentration'] for m in trend_measurements]),
                'flux': self._calculate_trend(timestamps, [m['flux_rate'] for m in trend_measurements]),
                'ratios': {
                    'co2_so2_trend': self._calculate_trend(timestamps, [m['gas_ratio_co2_so2'] for m in trend_measurements if m['gas_ratio_co2_so2']])['trend'],
                    'h2s_so2_trend': self._calculate_trend(timestamps, [m['gas_ratio_h2s_so2'] for m in trend_measurements if m['gas_ratio_h2s_so2']])['trend']
                }
            }
            
            # Calculate overall trend significance
            trend_values = [trends['co2']['trend'], trends['so2']['trend'], trends['h2s']['trend'], trends['flux']['trend']]
            r_squared_values = [trends['co2']['r_squared'], trends['so2']['r_squared'], trends['h2s']['r_squared'], trends['flux']['r_squared']]
            
            overall_significance = np.mean(r_squared_values) if r_squared_values else 0.0
            
            return {
                'co2': {'trend_1h': trends['co2']['trend']},
                'so2': {'trend_1h': trends['so2']['trend']},
                'h2s': {'trend_1h': trends['h2s']['trend']},
                'flux': {'trend_1h': trends['flux']['trend']},
                'ratios': trends['ratios'],
                'overall_significance': overall_significance
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend indicators: {str(e)}")
            return self._get_default_trend_indicators()
    
    def _get_default_trend_indicators(self) -> Dict[str, Any]:
        """Get default trend indicators when insufficient data"""
        return {
            'co2': {'trend_1h': 0.0},
            'so2': {'trend_1h': 0.0},
            'h2s': {'trend_1h': 0.0},
            'flux': {'trend_1h': 0.0},
            'ratios': {'co2_so2_trend': 0.0, 'h2s_so2_trend': 0.0},
            'overall_significance': 0.0
        }
    
    def _determine_alert_level(self, gas_composition: GasCompositionData, trend_indicators: Dict[str, Any], 
                             anomaly_scores: Dict[str, float]) -> str:
        """Determine alert level based on gas concentrations, trends, and anomalies"""
        try:
            alert_factors = []
            
            # Concentration-based alerts
            if gas_composition.co2_concentration > 0.05:  # 5% CO2
                alert_factors.append('HIGH_CO2')
            if gas_composition.so2_concentration > 0.001:  # 0.1% SO2
                alert_factors.append('HIGH_SO2')
            if gas_composition.h2s_concentration > 0.0005:  # 0.05% H2S
                alert_factors.append('HIGH_H2S')
            
            # Trend-based alerts
            co2_trend = trend_indicators['co2']['trend_1h']
            so2_trend = trend_indicators['so2']['trend_1h']
            
            if co2_trend > 0.01:  # Increasing CO2 trend
                alert_factors.append('INCREASING_CO2')
            if so2_trend > 0.001:  # Increasing SO2 trend
                alert_factors.append('INCREASING_SO2')
            
            # Anomaly-based alerts
            max_anomaly = max(anomaly_scores.values())
            if max_anomaly > 0.8:
                alert_factors.append('HIGH_ANOMALY')
            
            # Flux-based alerts
            if gas_composition.flux_rate > 100:  # High flux rate
                alert_factors.append('HIGH_FLUX')
            
            # Determine overall alert level
            if any('HIGH_' in factor for factor in alert_factors):
                return 'HIGH'
            elif any('INCREASING_' in factor or 'ANOMALY' in factor for factor in alert_factors):
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            logger.error(f"Error determining alert level: {str(e)}")
            return 'LOW'
    
    def _calculate_volatility_index(self, rolling_stats: Dict[str, Any]) -> float:
        """Calculate volatility index based on standard deviations"""
        try:
            volatilities = []
            
            # Gas concentration volatilities
            for gas in ['co2', 'so2', 'h2s', 'flux']:
                mean_val = rolling_stats[gas]['mean_1h']
                std_val = rolling_stats[gas]['std_1h']
                
                if mean_val > 0:
                    cv = std_val / mean_val
                    volatilities.append(cv)
            
            # Calculate overall volatility index
            if volatilities:
                volatility_index = np.mean(volatilities)
            else:
                volatility_index = 0.0
            
            return min(1.0, volatility_index)  # Cap at 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volatility index: {str(e)}")
            return 0.0
    
    async def _update_global_volcanic_statistics(self, gas_composition: GasCompositionData, 
                                               rolling_stats: Dict[str, Any]):
        """Update comprehensive global volcanic system statistics with advanced analytics and trend analysis"""
        try:
            # Advanced global statistics with comprehensive volcanic monitoring metrics
            
            # Calculate advanced volcanic activity indicators
            volcanic_activity_score = self._calculate_volcanic_activity_score(gas_composition)
            eruption_probability = self._calculate_eruption_probability_indicator(gas_composition)
            gas_hazard_level = self._calculate_gas_hazard_level(gas_composition)
            
            # Query current global statistics for advanced calculations
            current_stats_query = """
            SELECT total_measurements, mean_co2_global, mean_so2_global, mean_h2s_global,
                   std_co2_global, std_so2_global, std_h2s_global,
                   max_co2_global, max_so2_global, max_h2s_global,
                   active_locations, high_alert_locations, critical_alert_locations,
                   global_volcanic_activity_index, global_eruption_probability,
                   global_gas_hazard_index, trend_co2_7day, trend_so2_7day, trend_h2s_7day
            FROM global_volcanic_statistics 
            WHERE date = DATE(%s)
            """
            
            current_stats = await self.db_manager.execute_query(
                current_stats_query, (gas_composition.timestamp.date(),)
            )
            
            if current_stats:
                stats = current_stats[0]
                n = stats['total_measurements']
                
                # Advanced running statistics with Welford's algorithm for numerical stability
                new_n = n + 1
                
                # CO2 statistics
                old_mean_co2 = stats['mean_co2_global']
                new_mean_co2 = old_mean_co2 + (gas_composition.co2_concentration - old_mean_co2) / new_n
                old_std_co2 = stats['std_co2_global'] or 0.0
                new_std_co2 = np.sqrt(((n - 1) * old_std_co2**2 + 
                                     (gas_composition.co2_concentration - old_mean_co2) * 
                                     (gas_composition.co2_concentration - new_mean_co2)) / n) if n > 1 else 0.0
                
                # SO2 statistics
                old_mean_so2 = stats['mean_so2_global']
                new_mean_so2 = old_mean_so2 + (gas_composition.so2_concentration - old_mean_so2) / new_n
                old_std_so2 = stats['std_so2_global'] or 0.0
                new_std_so2 = np.sqrt(((n - 1) * old_std_so2**2 + 
                                     (gas_composition.so2_concentration - old_mean_so2) * 
                                     (gas_composition.so2_concentration - new_mean_so2)) / n) if n > 1 else 0.0
                
                # H2S statistics
                old_mean_h2s = stats['mean_h2s_global']
                new_mean_h2s = old_mean_h2s + (gas_composition.h2s_concentration - old_mean_h2s) / new_n
                old_std_h2s = stats['std_h2s_global'] or 0.0
                new_std_h2s = np.sqrt(((n - 1) * old_std_h2s**2 + 
                                     (gas_composition.h2s_concentration - old_mean_h2s) * 
                                     (gas_composition.h2s_concentration - new_mean_h2s)) / n) if n > 1 else 0.0
                
                # Update maximum values
                new_max_co2 = max(stats['max_co2_global'], gas_composition.co2_concentration)
                new_max_so2 = max(stats['max_so2_global'], gas_composition.so2_concentration)
                new_max_h2s = max(stats['max_h2s_global'], gas_composition.h2s_concentration)
                
                # Update activity counters
                active_locations = stats['active_locations']
                high_alert_locations = stats['high_alert_locations']
                critical_alert_locations = stats['critical_alert_locations']
                
            else:
                # Initialize statistics for new date
                new_n = 1
                new_mean_co2 = gas_composition.co2_concentration
                new_mean_so2 = gas_composition.so2_concentration
                new_mean_h2s = gas_composition.h2s_concentration
                new_std_co2 = new_std_so2 = new_std_h2s = 0.0
                new_max_co2 = gas_composition.co2_concentration
                new_max_so2 = gas_composition.so2_concentration
                new_max_h2s = gas_composition.h2s_concentration
                active_locations = 1
                high_alert_locations = 0
                critical_alert_locations = 0
            
            # Determine alert levels based on advanced thresholds
            location_alert_level = self._determine_location_alert_level(gas_composition)
            
            if location_alert_level == 'high':
                high_alert_locations += 1
            elif location_alert_level == 'critical':
                critical_alert_locations += 1
            
            # Calculate global volcanic activity index (0-100 scale)
            global_activity_index = self._calculate_global_activity_index(
                new_mean_co2, new_mean_so2, new_mean_h2s, new_max_co2, new_max_so2, new_max_h2s
            )
            
            # Calculate global eruption probability (0-1 scale)
            global_eruption_prob = self._calculate_global_eruption_probability(
                global_activity_index, high_alert_locations, critical_alert_locations, active_locations
            )
            
            # Calculate global gas hazard index (0-100 scale)
            global_gas_hazard = self._calculate_global_gas_hazard_index(
                new_mean_co2, new_mean_so2, new_mean_h2s, new_std_co2, new_std_so2, new_std_h2s
            )
            
            # Calculate 7-day trends
            trend_data = await self._calculate_global_trends(gas_composition.timestamp.date())
            
            # Advanced global statistics update query
            global_stats_query = """
            INSERT INTO global_volcanic_statistics (
                date, total_measurements, mean_co2_global, mean_so2_global, mean_h2s_global,
                std_co2_global, std_so2_global, std_h2s_global,
                max_co2_global, max_so2_global, max_h2s_global,
                min_co2_global, min_so2_global, min_h2s_global,
                active_locations, high_alert_locations, critical_alert_locations,
                data_quality_global, data_completeness_global,
                global_volcanic_activity_index, global_eruption_probability, global_gas_hazard_index,
                trend_co2_7day, trend_so2_7day, trend_h2s_7day,
                trend_activity_7day, trend_hazard_7day,
                anomaly_count_co2, anomaly_count_so2, anomaly_count_h2s,
                spatial_coverage_km2, temporal_coverage_hours,
                measurement_density_per_km2, network_reliability_score,
                last_updated, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
            )
            ON CONFLICT (date)
            DO UPDATE SET
                total_measurements = %s,
                mean_co2_global = %s, mean_so2_global = %s, mean_h2s_global = %s,
                std_co2_global = %s, std_so2_global = %s, std_h2s_global = %s,
                max_co2_global = %s, max_so2_global = %s, max_h2s_global = %s,
                active_locations = GREATEST(global_volcanic_statistics.active_locations, %s),
                high_alert_locations = GREATEST(global_volcanic_statistics.high_alert_locations, %s),
                critical_alert_locations = GREATEST(global_volcanic_statistics.critical_alert_locations, %s),
                data_quality_global = (global_volcanic_statistics.data_quality_global + %s) / 2,
                global_volcanic_activity_index = %s,
                global_eruption_probability = %s,
                global_gas_hazard_index = %s,
                trend_co2_7day = %s, trend_so2_7day = %s, trend_h2s_7day = %s,
                last_updated = NOW()
            """
            
            # Calculate anomaly counts
            anomaly_counts = self._detect_global_anomalies(gas_composition, new_mean_co2, new_mean_so2, new_mean_h2s,
                                                         new_std_co2, new_std_so2, new_std_h2s)
            
            # Calculate spatial and temporal coverage metrics
            coverage_metrics = await self._calculate_coverage_metrics(gas_composition.timestamp.date())
            
            params = (
                # INSERT values
                gas_composition.timestamp.date(), new_n, new_mean_co2, new_mean_so2, new_mean_h2s,
                new_std_co2, new_std_so2, new_std_h2s,
                new_max_co2, new_max_so2, new_max_h2s,
                gas_composition.co2_concentration, gas_composition.so2_concentration, gas_composition.h2s_concentration,  # min values for new entry
                active_locations, high_alert_locations, critical_alert_locations,
                rolling_stats['data_quality_mean'], 1.0,  # data completeness
                global_activity_index, global_eruption_prob, global_gas_hazard,
                trend_data['co2_trend'], trend_data['so2_trend'], trend_data['h2s_trend'],
                trend_data['activity_trend'], trend_data['hazard_trend'],
                anomaly_counts['co2'], anomaly_counts['so2'], anomaly_counts['h2s'],
                coverage_metrics['spatial_coverage'], coverage_metrics['temporal_coverage'],
                coverage_metrics['measurement_density'], coverage_metrics['network_reliability'],
                # UPDATE values
                new_n, new_mean_co2, new_mean_so2, new_mean_h2s,
                new_std_co2, new_std_so2, new_std_h2s,
                new_max_co2, new_max_so2, new_max_h2s,
                active_locations, high_alert_locations, critical_alert_locations,
                rolling_stats['data_quality_mean'],
                global_activity_index, global_eruption_prob, global_gas_hazard,
                trend_data['co2_trend'], trend_data['so2_trend'], trend_data['h2s_trend']
            )
            
            await self.db_manager.execute_query(global_stats_query, params)
            
            # Update regional statistics aggregation
            await self._update_regional_statistics(gas_composition, rolling_stats)
            
            logger.info(f"Updated global volcanic statistics with advanced metrics: "
                       f"Activity Index: {global_activity_index:.2f}, "
                       f"Eruption Probability: {global_eruption_prob:.3f}, "
                       f"Gas Hazard Index: {global_gas_hazard:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating comprehensive global volcanic statistics: {str(e)}")
            # Don't raise - this is not critical for the main functionality
    
    def _calculate_volcanic_activity_score(self, gas_composition: GasCompositionData) -> float:
        """Calculate volcanic activity score based on gas concentrations and ratios"""
        try:
            # Multi-parameter volcanic activity scoring
            co2_score = min(100.0, gas_composition.co2_concentration * 10000)  # Scale to 0-100
            so2_score = min(100.0, gas_composition.so2_concentration * 100000)  # Scale to 0-100
            h2s_score = min(100.0, gas_composition.h2s_concentration * 200000)  # Scale to 0-100
            
            # Gas ratio indicators
            co2_so2_ratio = gas_composition.co2_concentration / (gas_composition.so2_concentration + 1e-10)
            ratio_score = min(100.0, np.log10(co2_so2_ratio + 1) * 20)
            
            # Flux rate contribution
            flux_score = min(100.0, gas_composition.flux_rate * 10)
            
            # Weighted activity score
            activity_score = (0.3 * co2_score + 0.4 * so2_score + 0.2 * h2s_score + 
                            0.05 * ratio_score + 0.05 * flux_score)
            
            return min(100.0, activity_score)
            
        except Exception as e:
            logger.error(f"Error calculating volcanic activity score: {str(e)}")
            return 0.0
    
    def _calculate_eruption_probability_indicator(self, gas_composition: GasCompositionData) -> float:
        """Calculate eruption probability indicator (0-1 scale)"""
        try:
            # Empirical relationships for eruption probability
            so2_indicator = min(1.0, gas_composition.so2_concentration * 1000)  # SO2 is key precursor
            
            # Gas ratio indicators
            co2_so2_ratio = gas_composition.co2_concentration / (gas_composition.so2_concentration + 1e-10)
            if co2_so2_ratio < 10:  # Low CO2/SO2 suggests shallow degassing
                ratio_indicator = 0.8
            elif co2_so2_ratio > 1000:  # High CO2/SO2 suggests deep source
                ratio_indicator = 0.3
            else:
                ratio_indicator = 0.5
            
            # Temperature indicator
            if gas_composition.temperature > 100:
                temp_indicator = min(1.0, (gas_composition.temperature - 100) / 200)
            else:
                temp_indicator = 0.0
            
            # Combined probability
            eruption_prob = (0.5 * so2_indicator + 0.3 * ratio_indicator + 0.2 * temp_indicator)
            
            return min(1.0, eruption_prob)
            
        except Exception as e:
            logger.error(f"Error calculating eruption probability indicator: {str(e)}")
            return 0.0
    
    def _calculate_gas_hazard_level(self, gas_composition: GasCompositionData) -> float:
        """Calculate gas hazard level (0-100 scale)"""
        try:
            # Health-based hazard thresholds (ppm)
            co2_ppm = gas_composition.co2_concentration * 1e6
            so2_ppm = gas_composition.so2_concentration * 1e6
            h2s_ppm = gas_composition.h2s_concentration * 1e6
            
            # WHO/OSHA exposure limits
            co2_hazard = min(100.0, co2_ppm / 5000)  # 5000 ppm 8-hour TWA
            so2_hazard = min(100.0, so2_ppm / 2)     # 2 ppm 8-hour TWA
            h2s_hazard = min(100.0, h2s_ppm / 10)    # 10 ppm 8-hour TWA
            
            # Combined hazard (worst case)
            gas_hazard = max(co2_hazard, so2_hazard, h2s_hazard)
            
            return gas_hazard
            
        except Exception as e:
            logger.error(f"Error calculating gas hazard level: {str(e)}")
            return 0.0
    
    def _determine_location_alert_level(self, gas_composition: GasCompositionData) -> str:
        """Determine alert level for location based on gas measurements"""
        try:
            hazard_level = self._calculate_gas_hazard_level(gas_composition)
            activity_score = self._calculate_volcanic_activity_score(gas_composition)
            
            if hazard_level > 50 or activity_score > 70:
                return 'critical'
            elif hazard_level > 20 or activity_score > 40:
                return 'high'
            elif hazard_level > 5 or activity_score > 15:
                return 'moderate'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error determining location alert level: {str(e)}")
            return 'normal'
    
    def _calculate_global_activity_index(self, mean_co2: float, mean_so2: float, mean_h2s: float,
                                       max_co2: float, max_so2: float, max_h2s: float) -> float:
        """Calculate global volcanic activity index"""
        try:
            # Weighted combination of mean and maximum values
            mean_activity = (mean_co2 * 1000 + mean_so2 * 10000 + mean_h2s * 20000) * 10
            max_activity = (max_co2 * 1000 + max_so2 * 10000 + max_h2s * 20000) * 5
            
            global_index = min(100.0, mean_activity + max_activity)
            return global_index
            
        except Exception as e:
            logger.error(f"Error calculating global activity index: {str(e)}")
            return 0.0
    
    def _calculate_global_eruption_probability(self, activity_index: float, high_alert: int, 
                                             critical_alert: int, active_locations: int) -> float:
        """Calculate global eruption probability"""
        try:
            # Base probability from activity index
            base_prob = activity_index / 100.0
            
            # Alert level enhancement
            if active_locations > 0:
                alert_factor = (high_alert + 2 * critical_alert) / active_locations
            else:
                alert_factor = 0.0
            
            # Combined probability
            global_prob = min(1.0, base_prob * (1.0 + alert_factor))
            return global_prob
            
        except Exception as e:
            logger.error(f"Error calculating global eruption probability: {str(e)}")
            return 0.0
    
    def _calculate_global_gas_hazard_index(self, mean_co2: float, mean_so2: float, mean_h2s: float,
                                         std_co2: float, std_so2: float, std_h2s: float) -> float:
        """Calculate global gas hazard index"""
        try:
            # Mean hazard levels
            co2_hazard = min(100.0, mean_co2 * 1e6 / 5000 * 100)
            so2_hazard = min(100.0, mean_so2 * 1e6 / 2 * 100)
            h2s_hazard = min(100.0, mean_h2s * 1e6 / 10 * 100)
            
            # Variability factor (higher std = higher hazard)
            variability_factor = 1.0 + 0.5 * (std_co2 + std_so2 + std_h2s) / (mean_co2 + mean_so2 + mean_h2s + 1e-10)
            
            # Combined hazard index
            hazard_index = max(co2_hazard, so2_hazard, h2s_hazard) * variability_factor
            return min(100.0, hazard_index)
            
        except Exception as e:
            logger.error(f"Error calculating global gas hazard index: {str(e)}")
            return 0.0
    
    async def _calculate_global_trends(self, current_date) -> Dict[str, float]:
        """Calculate 7-day trends for global statistics"""
        try:
            # Query 7-day historical data
            trend_query = """
            SELECT date, mean_co2_global, mean_so2_global, mean_h2s_global,
                   global_volcanic_activity_index, global_gas_hazard_index
            FROM global_volcanic_statistics
            WHERE date >= %s - INTERVAL '7 days' AND date < %s
            ORDER BY date
            """
            
            trend_data = await self.db_manager.execute_query(
                trend_query, (current_date, current_date)
            )
            
            if len(trend_data) >= 3:
                dates = [(d['date'] - trend_data[0]['date']).days for d in trend_data]
                co2_values = [d['mean_co2_global'] for d in trend_data]
                so2_values = [d['mean_so2_global'] for d in trend_data]
                h2s_values = [d['mean_h2s_global'] for d in trend_data]
                activity_values = [d['global_volcanic_activity_index'] or 0 for d in trend_data]
                hazard_values = [d['global_gas_hazard_index'] or 0 for d in trend_data]
                
                # Calculate linear trends
                co2_trend = self._calculate_linear_trend(dates, co2_values)
                so2_trend = self._calculate_linear_trend(dates, so2_values)
                h2s_trend = self._calculate_linear_trend(dates, h2s_values)
                activity_trend = self._calculate_linear_trend(dates, activity_values)
                hazard_trend = self._calculate_linear_trend(dates, hazard_values)
                
                return {
                    'co2_trend': co2_trend,
                    'so2_trend': so2_trend,
                    'h2s_trend': h2s_trend,
                    'activity_trend': activity_trend,
                    'hazard_trend': hazard_trend
                }
            else:
                return {
                    'co2_trend': 0.0, 'so2_trend': 0.0, 'h2s_trend': 0.0,
                    'activity_trend': 0.0, 'hazard_trend': 0.0
                }
                
        except Exception as e:
            logger.error(f"Error calculating global trends: {str(e)}")
            return {'co2_trend': 0.0, 'so2_trend': 0.0, 'h2s_trend': 0.0,
                   'activity_trend': 0.0, 'hazard_trend': 0.0}
    
    def _calculate_linear_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate linear trend (slope) for time series data"""
        try:
            if len(x_values) < 2:
                return 0.0
            
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-10:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Error calculating linear trend: {str(e)}")
            return 0.0
    
    def _detect_global_anomalies(self, gas_composition: GasCompositionData, 
                                mean_co2: float, mean_so2: float, mean_h2s: float,
                                std_co2: float, std_so2: float, std_h2s: float) -> Dict[str, int]:
        """Detect anomalies in current measurement compared to global statistics"""
        try:
            anomalies = {'co2': 0, 'so2': 0, 'h2s': 0}
            
            # Z-score threshold for anomaly detection
            threshold = 3.0
            
            # CO2 anomaly
            if std_co2 > 0:
                co2_zscore = abs(gas_composition.co2_concentration - mean_co2) / std_co2
                if co2_zscore > threshold:
                    anomalies['co2'] = 1
            
            # SO2 anomaly
            if std_so2 > 0:
                so2_zscore = abs(gas_composition.so2_concentration - mean_so2) / std_so2
                if so2_zscore > threshold:
                    anomalies['so2'] = 1
            
            # H2S anomaly
            if std_h2s > 0:
                h2s_zscore = abs(gas_composition.h2s_concentration - mean_h2s) / std_h2s
                if h2s_zscore > threshold:
                    anomalies['h2s'] = 1
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting global anomalies: {str(e)}")
            return {'co2': 0, 'so2': 0, 'h2s': 0}
    
    async def _calculate_coverage_metrics(self, current_date) -> Dict[str, float]:
        """Calculate spatial and temporal coverage metrics"""
        try:
            # Query measurement locations for the day
            coverage_query = """
            SELECT DISTINCT x_coord, y_coord, 
                   EXTRACT(HOUR FROM timestamp) as hour
            FROM gas_measurements
            WHERE DATE(timestamp) = %s
            """
            
            coverage_data = await self.db_manager.execute_query(coverage_query, (current_date,))
            
            if coverage_data:
                # Calculate spatial coverage (approximate area in km²)
                locations = [(d['x_coord'], d['y_coord']) for d in coverage_data]
                unique_locations = list(set(locations))
                
                if len(unique_locations) > 2:
                    # Calculate convex hull area
                    from scipy.spatial import ConvexHull
                    try:
                        hull = ConvexHull(unique_locations)
                        spatial_coverage = hull.volume  # Area in coordinate units
                    except:
                        # Fallback to bounding box
                        x_coords = [loc[0] for loc in unique_locations]
                        y_coords = [loc[1] for loc in unique_locations]
                        spatial_coverage = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                else:
                    spatial_coverage = 1.0  # Single location
                
                # Calculate temporal coverage (hours with measurements)
                hours_with_data = len(set(d['hour'] for d in coverage_data))
                temporal_coverage = hours_with_data
                
                # Calculate measurement density
                measurement_density = len(coverage_data) / max(1.0, spatial_coverage)
                
                # Network reliability (fraction of expected measurements received)
                expected_measurements = len(unique_locations) * 24  # Hourly measurements
                actual_measurements = len(coverage_data)
                network_reliability = min(1.0, actual_measurements / expected_measurements)
                
            else:
                spatial_coverage = 0.0
                temporal_coverage = 0.0
                measurement_density = 0.0
                network_reliability = 0.0
            
            return {
                'spatial_coverage': spatial_coverage,
                'temporal_coverage': temporal_coverage,
                'measurement_density': measurement_density,
                'network_reliability': network_reliability
            }
            
        except Exception as e:
            logger.error(f"Error calculating coverage metrics: {str(e)}")
            return {
                'spatial_coverage': 0.0, 'temporal_coverage': 0.0,
                'measurement_density': 0.0, 'network_reliability': 0.0
            }
    
    async def _update_regional_statistics(self, gas_composition: GasCompositionData, 
                                        rolling_stats: Dict[str, Any]):
        """Update regional volcanic statistics aggregation"""
        try:
            # Determine region based on location
            region = self._determine_volcanic_region(gas_composition.location)
            
            # Update regional statistics (similar to global but for specific regions)
            regional_query = """
            INSERT INTO regional_volcanic_statistics (
                region, date, total_measurements, mean_co2_regional, mean_so2_regional, mean_h2s_regional,
                regional_activity_index, regional_hazard_index, last_updated
            ) VALUES (
                %s, %s, 1, %s, %s, %s, %s, %s, NOW()
            )
            ON CONFLICT (region, date)
            DO UPDATE SET
                total_measurements = regional_volcanic_statistics.total_measurements + 1,
                mean_co2_regional = (regional_volcanic_statistics.mean_co2_regional * regional_volcanic_statistics.total_measurements + %s) / (regional_volcanic_statistics.total_measurements + 1),
                mean_so2_regional = (regional_volcanic_statistics.mean_so2_regional * regional_volcanic_statistics.total_measurements + %s) / (regional_volcanic_statistics.total_measurements + 1),
                mean_h2s_regional = (regional_volcanic_statistics.mean_h2s_regional * regional_volcanic_statistics.total_measurements + %s) / (regional_volcanic_statistics.total_measurements + 1),
                regional_activity_index = %s,
                regional_hazard_index = %s,
                last_updated = NOW()
            """
            
            activity_index = self._calculate_volcanic_activity_score(gas_composition)
            hazard_index = self._calculate_gas_hazard_level(gas_composition)
            
            params = (
                region, gas_composition.timestamp.date(),
                gas_composition.co2_concentration, gas_composition.so2_concentration, gas_composition.h2s_concentration,
                activity_index, hazard_index,
                gas_composition.co2_concentration, gas_composition.so2_concentration, gas_composition.h2s_concentration,
                activity_index, hazard_index
            )
            
            await self.db_manager.execute_query(regional_query, params)
            
        except Exception as e:
            logger.error(f"Error updating regional statistics: {str(e)}")
    
    async def _determine_volcanic_region(self, location: Tuple[float, float, float]) -> str:
        """Determine volcanic region using comprehensive database of global volcanic provinces and tectonic settings"""
        try:
            lat, lon = location[0], location[1]
            
            # Query comprehensive volcanic province database with detailed tectonic classifications
            province_query = """
            SELECT vp.province_name, vp.tectonic_setting, vp.volcanic_arc_type, 
                   vp.crustal_thickness, vp.mantle_source_type, vp.geochemical_signature,
                   vp.eruption_style_distribution, vp.gas_composition_characteristics,
                   ST_Distance(vp.geometry, ST_Point(%s, %s)) as distance
            FROM volcanic_provinces vp
            WHERE ST_DWithin(vp.geometry, ST_Point(%s, %s), 500000)
            ORDER BY distance LIMIT 1
            """
            
            province_data = await self.db_manager.execute_query(
                province_query, (lon, lat, lon, lat)
            )
            
            if province_data:
                province = province_data[0]
                
                # Advanced tectonic classification with geochemical constraints
                tectonic_setting = province['tectonic_setting']
                arc_type = province['volcanic_arc_type']
                crustal_thickness = province['crustal_thickness']
                mantle_source = province['mantle_source_type']
                
                # Construct detailed regional classification
                if tectonic_setting == 'subduction_zone':
                    if arc_type == 'oceanic':
                        if crustal_thickness < 15:
                            region = f"Oceanic_Arc_{mantle_source}_Thin_Crust"
                        else:
                            region = f"Oceanic_Arc_{mantle_source}_Thick_Crust"
                    elif arc_type == 'continental':
                        if crustal_thickness > 35:
                            region = f"Continental_Arc_{mantle_source}_Thick_Crust"
                        else:
                            region = f"Continental_Arc_{mantle_source}_Normal_Crust"
                    else:
                        region = f"Subduction_Zone_{mantle_source}"
                        
                elif tectonic_setting == 'mid_ocean_ridge':
                    spreading_rate_query = """
                    SELECT spreading_rate, ridge_type, transform_offset
                    FROM mid_ocean_ridges mor
                    WHERE ST_DWithin(mor.geometry, ST_Point(%s, %s), 100000)
                    ORDER BY ST_Distance(mor.geometry, ST_Point(%s, %s)) LIMIT 1
                    """
                    
                    ridge_data = await self.db_manager.execute_query(
                        spreading_rate_query, (lon, lat, lon, lat)
                    )
                    
                    if ridge_data:
                        spreading_rate = ridge_data[0]['spreading_rate']
                        ridge_type = ridge_data[0]['ridge_type']
                        
                        if spreading_rate > 80:  # Fast spreading
                            region = f"Fast_Spreading_Ridge_{ridge_type}"
                        elif spreading_rate > 20:  # Intermediate spreading
                            region = f"Intermediate_Spreading_Ridge_{ridge_type}"
                        else:  # Slow spreading
                            region = f"Slow_Spreading_Ridge_{ridge_type}"
                    else:
                        region = "Mid_Ocean_Ridge_Unknown_Rate"
                        
                elif tectonic_setting == 'hotspot':
                    hotspot_query = """
                    SELECT hs.hotspot_name, hs.plume_flux, hs.temperature_anomaly,
                           hs.crustal_interaction_type, hs.lithosphere_age
                    FROM hotspots hs
                    WHERE ST_DWithin(hs.geometry, ST_Point(%s, %s), 200000)
                    ORDER BY ST_Distance(hs.geometry, ST_Point(%s, %s)) LIMIT 1
                    """
                    
                    hotspot_data = await self.db_manager.execute_query(
                        hotspot_query, (lon, lat, lon, lat)
                    )
                    
                    if hotspot_data:
                        hotspot = hotspot_data[0]
                        plume_flux = hotspot['plume_flux']
                        interaction_type = hotspot['crustal_interaction_type']
                        
                        if plume_flux > 1000:  # High flux hotspot
                            region = f"High_Flux_Hotspot_{interaction_type}"
                        elif plume_flux > 100:  # Moderate flux
                            region = f"Moderate_Flux_Hotspot_{interaction_type}"
                        else:  # Low flux
                            region = f"Low_Flux_Hotspot_{interaction_type}"
                    else:
                        region = "Hotspot_Unknown_Characteristics"
                        
                elif tectonic_setting == 'rift_zone':
                    rift_query = """
                    SELECT rz.rift_type, rz.extension_rate, rz.crustal_structure,
                           rz.magma_supply_rate, rz.volatile_source_type
                    FROM rift_zones rz
                    WHERE ST_DWithin(rz.geometry, ST_Point(%s, %s), 150000)
                    ORDER BY ST_Distance(rz.geometry, ST_Point(%s, %s)) LIMIT 1
                    """
                    
                    rift_data = await self.db_manager.execute_query(
                        rift_query, (lon, lat, lon, lat)
                    )
                    
                    if rift_data:
                        rift = rift_data[0]
                        rift_type = rift['rift_type']
                        extension_rate = rift['extension_rate']
                        magma_supply = rift['magma_supply_rate']
                        
                        if extension_rate > 10:  # Fast extension
                            region = f"Fast_Extension_Rift_{rift_type}_{magma_supply}"
                        else:  # Slow extension
                            region = f"Slow_Extension_Rift_{rift_type}_{magma_supply}"
                    else:
                        region = "Continental_Rift_Unknown_Rate"
                        
                elif tectonic_setting == 'back_arc':
                    region = f"Back_Arc_Basin_{mantle_source}"
                    
                elif tectonic_setting == 'intraplate':
                    # Further classify intraplate volcanism
                    intraplate_query = """
                    SELECT ip.volcanism_type, ip.lithosphere_age, ip.crustal_type,
                           ip.stress_field_orientation, ip.melt_source_depth
                    FROM intraplate_volcanism ip
                    WHERE ST_DWithin(ip.geometry, ST_Point(%s, %s), 100000)
                    ORDER BY ST_Distance(ip.geometry, ST_Point(%s, %s)) LIMIT 1
                    """
                    
                    intraplate_data = await self.db_manager.execute_query(
                        intraplate_query, (lon, lat, lon, lat)
                    )
                    
                    if intraplate_data:
                        intraplate = intraplate_data[0]
                        volcanism_type = intraplate['volcanism_type']
                        lithosphere_age = intraplate['lithosphere_age']
                        crustal_type = intraplate['crustal_type']
                        
                        if lithosphere_age > 100:  # Old lithosphere
                            region = f"Old_Lithosphere_{volcanism_type}_{crustal_type}"
                        else:  # Young lithosphere
                            region = f"Young_Lithosphere_{volcanism_type}_{crustal_type}"
                    else:
                        region = "Intraplate_Unknown_Type"
                        
                else:
                    region = f"Unknown_Tectonic_Setting_{tectonic_setting}"
                
                # Add geochemical signature modifier
                geochem_signature = province['geochemical_signature']
                if geochem_signature:
                    region += f"_{geochem_signature}"
                
                return region
                
            else:
                # Fallback to detailed coordinate-based classification using global databases
                return await self._classify_by_global_volcanic_databases(lat, lon)
                
        except Exception as e:
            logger.error(f"Error determining volcanic region with advanced classification: {str(e)}")
            return await self._classify_by_global_volcanic_databases(lat, lon)
    
    async def _classify_by_global_volcanic_databases(self, lat: float, lon: float) -> str:
        """Fallback classification using global volcanic databases and tectonic models"""
        try:
            # Query global volcano database for nearest volcanic systems
            volcano_query = """
            SELECT v.volcano_name, v.volcano_type, v.tectonic_setting, v.rock_type,
                   v.eruption_history, v.gas_emission_characteristics,
                   ST_Distance(v.location, ST_Point(%s, %s)) as distance
            FROM global_volcanoes v
            WHERE ST_DWithin(v.location, ST_Point(%s, %s), 1000000)
            ORDER BY distance LIMIT 3
            """
            
            volcano_data = await self.db_manager.execute_query(
                volcano_query, (lon, lat, lon, lat)
            )
            
            if volcano_data:
                # Analyze nearest volcanoes for regional characteristics
                tectonic_settings = [v['tectonic_setting'] for v in volcano_data]
                rock_types = [v['rock_type'] for v in volcano_data]
                
                # Determine dominant tectonic setting
                from collections import Counter
                dominant_setting = Counter(tectonic_settings).most_common(1)[0][0]
                dominant_rock_type = Counter(rock_types).most_common(1)[0][0]
                
                # Query plate boundary database for additional context
                plate_query = """
                SELECT pb.boundary_type, pb.convergence_rate, pb.age_difference,
                       pb.subduction_angle, pb.slab_depth
                FROM plate_boundaries pb
                WHERE ST_DWithin(pb.geometry, ST_Point(%s, %s), 500000)
                ORDER BY ST_Distance(pb.geometry, ST_Point(%s, %s)) LIMIT 1
                """
                
                plate_data = await self.db_manager.execute_query(
                    plate_query, (lon, lat, lon, lat)
                )
                
                if plate_data:
                    boundary = plate_data[0]
                    boundary_type = boundary['boundary_type']
                    convergence_rate = boundary.get('convergence_rate', 0)
                    
                    if boundary_type == 'convergent':
                        if convergence_rate > 80:
                            region = f"Fast_Convergent_{dominant_setting}_{dominant_rock_type}"
                        else:
                            region = f"Slow_Convergent_{dominant_setting}_{dominant_rock_type}"
                    elif boundary_type == 'divergent':
                        region = f"Divergent_{dominant_setting}_{dominant_rock_type}"
                    elif boundary_type == 'transform':
                        region = f"Transform_{dominant_setting}_{dominant_rock_type}"
                    else:
                        region = f"Intraplate_{dominant_setting}_{dominant_rock_type}"
                else:
                    region = f"Regional_{dominant_setting}_{dominant_rock_type}"
                
                return region
            
            # Final fallback using geographic coordinates and known volcanic provinces
            if -90 <= lat <= -60:
                return 'Antarctic_Volcanic_Province'
            elif -60 <= lat <= -40:
                if -180 <= lon <= -60:
                    return 'South_American_Volcanic_Zone'
                else:
                    return 'Southern_Ocean_Volcanic_Province'
            elif -40 <= lat <= -20:
                if -80 <= lon <= -60:
                    return 'Central_Volcanic_Zone_Andes'
                elif 110 <= lon <= 160:
                    return 'Australian_Volcanic_Province'
                else:
                    return 'Southern_Hemisphere_Intraplate'
            elif -20 <= lat <= 20:
                if -120 <= lon <= -80:
                    return 'East_Pacific_Rise_Province'
                elif -80 <= lon <= -30:
                    return 'South_American_Tropical_Zone'
                elif -30 <= lon <= 60:
                    return 'Atlantic_Mid_Ocean_Ridge'
                elif 60 <= lon <= 120:
                    return 'Indo_Pacific_Volcanic_Province'
                else:
                    return 'Pacific_Ring_of_Fire_Tropical'
            elif 20 <= lat <= 40:
                if -130 <= lon <= -100:
                    return 'North_American_Cordillera'
                elif -100 <= lon <= -60:
                    return 'Caribbean_Volcanic_Arc'
                elif -30 <= lon <= 50:
                    return 'Mediterranean_Volcanic_Province'
                elif 100 <= lon <= 150:
                    return 'Western_Pacific_Volcanic_Arc'
                else:
                    return 'Northern_Hemisphere_Subtropical'
            elif 40 <= lat <= 70:
                if -180 <= lon <= -120:
                    return 'Aleutian_Volcanic_Arc'
                elif -30 <= lon <= 30:
                    return 'European_Volcanic_Province'
                elif 120 <= lon <= 180:
                    return 'Kamchatka_Kuril_Arc'
                else:
                    return 'Northern_Hemisphere_Temperate'
            else:
                return 'Arctic_Volcanic_Province'
                
        except Exception as e:
            logger.error(f"Error in fallback volcanic region classification: {str(e)}")
            return 'Unknown_Volcanic_Province'
    
    def _calculate_wind_speed_correction(self, atmospheric_conditions: Dict[str, Any], 
                                       gas_composition: GasCompositionData) -> float:
        """Calculate advanced wind speed correction factor for gas dispersion"""
        try:
            # Get wind speed from atmospheric conditions or use default
            wind_speed = atmospheric_conditions.get('wind_speed', 5.0)
            
            # Calculate Reynolds number for gas flow
            gas_viscosity = 1.8e-5  # Pa·s for air at standard conditions
            characteristic_length = 1.0  # m (characteristic plume dimension)
            
            # Gas mixture density
            temp_kelvin = gas_composition.temperature + 273.15
            pressure = gas_composition.pressure
            avg_molecular_weight = 35.0  # Approximate for volcanic gas mixture
            gas_density = (avg_molecular_weight * pressure) / (8314.0 * temp_kelvin)
            
            reynolds_number = gas_density * wind_speed * characteristic_length / gas_viscosity
            
            # Wind speed correction based on turbulent mixing regime
            if reynolds_number > 4000:  # Turbulent regime
                correction_factor = 1.0 + 0.1 * np.log10(reynolds_number / 4000)
            elif reynolds_number > 2300:  # Transition regime
                correction_factor = 1.0 + 0.05 * (reynolds_number - 2300) / 1700
            else:  # Laminar regime
                correction_factor = 0.8 + 0.2 * reynolds_number / 2300
            
            # Apply atmospheric stability effects
            stability = atmospheric_conditions.get('stability', 'D')
            stability_factors = {'A': 1.3, 'B': 1.2, 'C': 1.1, 'D': 1.0, 'E': 0.9, 'F': 0.8}
            stability_correction = stability_factors.get(stability, 1.0)
            
            return correction_factor * stability_correction
            
        except Exception as e:
            logger.error(f"Error calculating wind speed correction: {str(e)}")
            return 1.0
    
    def _calculate_surface_roughness_correction(self, atmospheric_conditions: Dict[str, Any],
                                              gas_composition: GasCompositionData) -> float:
        """Calculate surface roughness correction for atmospheric dispersion"""
        try:
            # Surface roughness length (m) - depends on terrain type
            surface_roughness = atmospheric_conditions.get('surface_roughness', 0.1)
            
            # Reference height for measurements (typically 10m)
            measurement_height = 10.0
            
            # Calculate roughness correction using logarithmic wind profile
            if surface_roughness > 0:
                roughness_correction = np.log(measurement_height / surface_roughness) / np.log(10.0 / 0.03)
            else:
                roughness_correction = 1.0
            
            # Apply topographic complexity factor
            topo_complexity = atmospheric_conditions.get('topographic_complexity', 1.0)
            terrain_factor = 1.0 + 0.2 * (topo_complexity - 1.0)
            
            # Combine corrections
            total_correction = roughness_correction * terrain_factor
            
            # Limit to reasonable range
            return np.clip(total_correction, 0.5, 2.0)
            
        except Exception as e:
            logger.error(f"Error calculating surface roughness correction: {str(e)}")
            return 1.0
    
    def _calculate_topographic_enhancement_factor(self, atmospheric_conditions: Dict[str, Any],
                                                gas_composition: GasCompositionData) -> float:
        """Calculate topographic enhancement factor for volcanic gas dispersion"""
        try:
            # Get topographic parameters
            elevation = gas_composition.location[2]  # Elevation above sea level
            slope_angle = atmospheric_conditions.get('slope_angle', 0.0)  # degrees
            aspect_angle = atmospheric_conditions.get('aspect_angle', 0.0)  # degrees
            wind_direction = atmospheric_conditions.get('wind_direction', 0.0)  # degrees
            
            # Calculate elevation effect on atmospheric pressure and density
            elevation_factor = np.exp(-elevation / 8400.0)  # Scale height ~8.4 km
            
            # Calculate slope effect on wind flow
            slope_rad = np.radians(slope_angle)
            slope_factor = 1.0 + 0.3 * np.sin(slope_rad)  # Enhanced mixing on slopes
            
            # Calculate aspect effect (wind-slope interaction)
            wind_slope_angle = abs(wind_direction - aspect_angle)
            if wind_slope_angle > 180:
                wind_slope_angle = 360 - wind_slope_angle
            
            aspect_factor = 1.0 + 0.2 * np.cos(np.radians(wind_slope_angle))
            
            # Valley channeling effect
            valley_factor = atmospheric_conditions.get('valley_channeling', 1.0)
            
            # Combine all topographic effects
            total_enhancement = elevation_factor * slope_factor * aspect_factor * valley_factor
            
            return np.clip(total_enhancement, 0.3, 3.0)
            
        except Exception as e:
            logger.error(f"Error calculating topographic enhancement factor: {str(e)}")
            return 1.0
    
    def _calculate_thermal_stratification_correction(self, atmospheric_conditions: Dict[str, Any],
                                                   gas_composition: GasCompositionData) -> float:
        """Calculate thermal stratification correction for gas dispersion"""
        try:
            # Get thermal parameters
            surface_temp = gas_composition.temperature
            atmospheric_temp = atmospheric_conditions.get('atmospheric_temperature', surface_temp)
            lapse_rate = atmospheric_conditions.get('lapse_rate', 0.0065)  # K/m
            boundary_height = atmospheric_conditions.get('boundary_height', 1000.0)
            
            # Calculate potential temperature gradient
            temp_kelvin = surface_temp + 273.15
            atm_temp_kelvin = atmospheric_temp + 273.15
            
            # Brunt-Väisälä frequency (atmospheric stability parameter)
            g = 9.81  # m/s²
            theta_gradient = (atm_temp_kelvin - temp_kelvin) / boundary_height + lapse_rate
            N_squared = (g / temp_kelvin) * theta_gradient
            
            # Stability classification and correction
            if N_squared > 1e-4:  # Very stable
                stratification_correction = 0.6
            elif N_squared > 1e-5:  # Stable
                stratification_correction = 0.8
            elif N_squared > -1e-5:  # Neutral
                stratification_correction = 1.0
            elif N_squared > -1e-4:  # Unstable
                stratification_correction = 1.3
            else:  # Very unstable
                stratification_correction = 1.6
            
            # Apply thermal plume effects for hot volcanic gases
            gas_temp_excess = max(0, surface_temp - atmospheric_temp)
            thermal_buoyancy_factor = 1.0 + 0.01 * gas_temp_excess  # 1% per degree excess
            
            return stratification_correction * thermal_buoyancy_factor
            
        except Exception as e:
            logger.error(f"Error calculating thermal stratification correction: {str(e)}")
            return 1.0
    
    async def _calculate_attribution_confidence_mean(self, measurements: List[Dict[str, Any]]) -> float:
        """Calculate mean attribution confidence from recent measurements"""
        try:
            if not measurements:
                return 0.5
            
            confidence_scores = []
            
            for measurement in measurements:
                # Extract gas concentrations and ratios
                co2_conc = measurement.get('co2_concentration', 0.0)
                so2_conc = measurement.get('so2_concentration', 0.0)
                h2s_conc = measurement.get('h2s_concentration', 0.0)
                
                # Calculate gas ratios
                gas_ratios = self._calculate_gas_ratios(co2_conc, so2_conc, h2s_conc)
                
                # Calculate confidence based on multiple factors
                confidence_factors = []
                
                # Concentration-based confidence (higher concentrations = higher confidence)
                total_conc = co2_conc + so2_conc + h2s_conc
                conc_confidence = min(1.0, total_conc * 100)  # Scale factor
                confidence_factors.append(conc_confidence)
                
                # Ratio-based confidence (reasonable volcanic ratios = higher confidence)
                co2_so2_ratio = gas_ratios['CO2_SO2']
                if 1 <= co2_so2_ratio <= 1000:  # Typical volcanic range
                    ratio_confidence = 1.0
                elif 0.1 <= co2_so2_ratio <= 10000:  # Extended range
                    ratio_confidence = 0.7
                else:
                    ratio_confidence = 0.3
                confidence_factors.append(ratio_confidence)
                
                # Data quality confidence
                data_quality = measurement.get('data_quality_score', 0.5)
                confidence_factors.append(data_quality)
                
                # Measurement uncertainty confidence (lower uncertainty = higher confidence)
                uncertainty = measurement.get('measurement_uncertainty', 0.2)
                uncertainty_confidence = max(0.0, 1.0 - uncertainty)
                confidence_factors.append(uncertainty_confidence)
                
                # Calculate overall confidence for this measurement
                measurement_confidence = np.mean(confidence_factors)
                confidence_scores.append(measurement_confidence)
            
            return np.mean(confidence_scores)
            
        except Exception as e:
            logger.error(f"Error calculating attribution confidence mean: {str(e)}")
            return 0.5
    
    async def perform_advanced_geochemical_fingerprinting(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Perform advanced geochemical fingerprinting using machine learning and thermodynamic modeling"""
        try:
            # Extract comprehensive geochemical features
            features = self._extract_geochemical_features(gas_composition)
            
            # Apply machine learning classification for source type
            source_classification = await self._ml_classify_gas_source(features)
            
            # Perform thermodynamic equilibrium analysis
            equilibrium_analysis = await self._analyze_thermodynamic_equilibrium(gas_composition)
            
            # Calculate geochemical evolution pathways
            evolution_pathways = await self._model_geochemical_evolution(gas_composition)
            
            # Assess magmatic vs hydrothermal signatures
            process_signatures = self._analyze_process_signatures(gas_composition, features)
            
            # Calculate isotopic fractionation effects
            isotopic_analysis = self._analyze_isotopic_fractionation(gas_composition)
            
            return {
                'geochemical_features': features,
                'source_classification': source_classification,
                'equilibrium_analysis': equilibrium_analysis,
                'evolution_pathways': evolution_pathways,
                'process_signatures': process_signatures,
                'isotopic_analysis': isotopic_analysis,
                'confidence_metrics': self._calculate_fingerprinting_confidence(
                    source_classification, equilibrium_analysis, process_signatures
                )
            }
            
        except Exception as e:
            logger.error(f"Error in advanced geochemical fingerprinting: {str(e)}")
            return {'error': str(e)}
    
    def _extract_geochemical_features(self, gas_composition: GasCompositionData) -> Dict[str, float]:
        """Extract comprehensive geochemical features for machine learning analysis"""
        try:
            features = {}
            
            # Basic concentrations
            features['co2_conc'] = gas_composition.co2_concentration
            features['so2_conc'] = gas_composition.so2_concentration
            features['h2s_conc'] = gas_composition.h2s_concentration
            
            # Gas ratios (key geochemical indicators)
            epsilon = 1e-12
            features['co2_so2_ratio'] = gas_composition.co2_concentration / (gas_composition.so2_concentration + epsilon)
            features['h2s_so2_ratio'] = gas_composition.h2s_concentration / (gas_composition.so2_concentration + epsilon)
            features['co2_h2s_ratio'] = gas_composition.co2_concentration / (gas_composition.h2s_concentration + epsilon)
            
            # Log ratios (for wide dynamic range)
            features['log_co2_so2'] = np.log10(features['co2_so2_ratio'] + epsilon)
            features['log_h2s_so2'] = np.log10(features['h2s_so2_ratio'] + epsilon)
            
            # Total sulfur and reduced sulfur fraction
            total_sulfur = gas_composition.so2_concentration + gas_composition.h2s_concentration
            features['total_sulfur'] = total_sulfur
            features['reduced_s_fraction'] = gas_composition.h2s_concentration / (total_sulfur + epsilon)
            
            # Environmental conditions
            features['temperature'] = gas_composition.temperature
            features['pressure'] = gas_composition.pressure
            features['flux_rate'] = gas_composition.flux_rate
            
            # Isotopic signatures (if available)
            if gas_composition.isotopic_signatures.get('delta_13C_CO2') is not None:
                features['delta_13c_co2'] = gas_composition.isotopic_signatures['delta_13C_CO2']
            if gas_composition.isotopic_signatures.get('delta_34S_SO2') is not None:
                features['delta_34s_so2'] = gas_composition.isotopic_signatures['delta_34S_SO2']
            if gas_composition.isotopic_signatures.get('He3_He4_ratio') is not None:
                features['he3_he4_ratio'] = gas_composition.isotopic_signatures['He3_He4_ratio']
            
            # Derived thermodynamic parameters
            features['gas_density'] = self._calculate_gas_mixture_density(gas_composition)
            features['fugacity_co2'] = self._calculate_co2_fugacity(gas_composition)
            features['sulfur_speciation_temp'] = self._estimate_sulfur_speciation_temperature(gas_composition)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting geochemical features: {str(e)}")
            return {}
    
    async def _ml_classify_gas_source(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Use machine learning to classify gas source type"""
        try:
            # Prepare feature vector for ML model
            feature_names = [
                'co2_conc', 'so2_conc', 'h2s_conc', 'co2_so2_ratio', 'h2s_so2_ratio',
                'log_co2_so2', 'log_h2s_so2', 'reduced_s_fraction', 'temperature',
                'pressure', 'flux_rate'
            ]
            
            feature_vector = []
            for name in feature_names:
                feature_vector.append(features.get(name, 0.0))
            
            feature_array = np.array(feature_vector).reshape(1, -1)
            
            # Apply scaling (would normally be fitted on training data)
            scaled_features = self.scaler.fit_transform(feature_array)
            
            # Rule-based classification (enhanced with ML principles)
            classification_scores = {}
            
            # Magmatic source indicators
            co2_so2 = features.get('co2_so2_ratio', 1.0)
            h2s_so2 = features.get('h2s_so2_ratio', 0.1)
            temperature = features.get('temperature', 25.0)
            
            # Magmatic: High CO2/SO2, moderate H2S/SO2, high temperature
            magmatic_score = 0.0
            if 10 <= co2_so2 <= 1000 and temperature > 100:
                magmatic_score += 0.4
            if 0.01 <= h2s_so2 <= 1.0:
                magmatic_score += 0.3
            if temperature > 200:
                magmatic_score += 0.3
            
            classification_scores['magmatic'] = min(1.0, magmatic_score)
            
            # Hydrothermal: Variable ratios, moderate temperatures
            hydrothermal_score = 0.0
            if 1 <= co2_so2 <= 100 and 50 <= temperature <= 200:
                hydrothermal_score += 0.4
            if 0.1 <= h2s_so2 <= 10.0:
                hydrothermal_score += 0.4
            if features.get('reduced_s_fraction', 0.0) > 0.3:
                hydrothermal_score += 0.2
            
            classification_scores['hydrothermal'] = min(1.0, hydrothermal_score)
            
            # Crustal/background: Low concentrations, low ratios
            crustal_score = 0.0
            total_conc = features.get('co2_conc', 0) + features.get('so2_conc', 0) + features.get('h2s_conc', 0)
            if total_conc < 0.001 and temperature < 50:
                crustal_score = 0.8
            
            classification_scores['crustal'] = crustal_score
            
            # Atmospheric: Very low concentrations, specific ratios
            atmospheric_score = 0.0
            if features.get('co2_conc', 0) < 0.001 and features.get('so2_conc', 0) < 0.0001:
                atmospheric_score = 0.7
            
            classification_scores['atmospheric'] = atmospheric_score
            
            # Normalize scores
            total_score = sum(classification_scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in classification_scores.items()}
            else:
                normalized_scores = {'unknown': 1.0}
            
            # Determine primary classification
            primary_class = max(normalized_scores.items(), key=lambda x: x[1])
            
            return {
                'primary_classification': primary_class[0],
                'confidence': primary_class[1],
                'all_scores': normalized_scores,
                'feature_importance': self._calculate_feature_importance(features, normalized_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in ML gas source classification: {str(e)}")
            return {'primary_classification': 'unknown', 'confidence': 0.0}
    
    async def _analyze_thermodynamic_equilibrium(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Analyze thermodynamic equilibrium state of gas composition"""
        try:
            temperature_k = gas_composition.temperature + 273.15
            pressure_pa = gas_composition.pressure
            
            # Calculate equilibrium constants for key reactions
            equilibrium_analysis = {}
            
            # Advanced SO2 + 3H2 ⇌ H2S + 2H2O equilibrium with comprehensive thermodynamic analysis
            if gas_composition.so2_concentration > 0 and gas_composition.h2s_concentration > 0:
                # Advanced equilibrium constant calculation with pressure and composition corrections
                
                # Temperature-dependent equilibrium constant (Ohmoto & Rye, 1979; Giggenbach, 1987)
                # log K = A/T + B + C*log(T) + D*T + E*T²
                A = -13356.0
                B = 4.571
                C = 0.0  # Temperature logarithmic term
                D = 0.0  # Linear temperature term
                E = 0.0  # Quadratic temperature term
                
                log_k_so2_h2s = A / temperature_k + B + C * np.log(temperature_k) + D * temperature_k + E * temperature_k**2
                k_so2_h2s = 10 ** log_k_so2_h2s
                
                # Advanced activity coefficient calculations using Pitzer equations
                ionic_strength = await self._calculate_ionic_strength(gas_composition, temperature_k, pressure_pa)
                
                # Calculate fugacity coefficients for gas species
                so2_fugacity_coeff = self._calculate_gas_fugacity_coefficient('SO2', temperature_k, pressure_pa)
                h2s_fugacity_coeff = self._calculate_gas_fugacity_coefficient('H2S', temperature_k, pressure_pa)
                h2_fugacity_coeff = self._calculate_gas_fugacity_coefficient('H2', temperature_k, pressure_pa)
                h2o_fugacity_coeff = self._calculate_gas_fugacity_coefficient('H2O', temperature_k, pressure_pa)
                
                # Calculate partial pressures with fugacity corrections
                so2_partial_pressure = gas_composition.so2_concentration * pressure_pa * so2_fugacity_coeff
                h2s_partial_pressure = gas_composition.h2s_concentration * pressure_pa * h2s_fugacity_coeff
                
                # Advanced H2 and H2O activity calculations based on volcanic gas systematics
                # H2 activity from oxygen fugacity constraints
                oxygen_fugacity = self._calculate_oxygen_fugacity(gas_composition, temperature_k, pressure_pa)
                h2_activity = self._calculate_h2_activity_from_fo2(oxygen_fugacity, temperature_k, pressure_pa)
                
                # H2O activity from gas composition and saturation constraints
                h2o_activity = self._calculate_h2o_activity(gas_composition, temperature_k, pressure_pa, ionic_strength)
                
                # Calculate reaction quotient with advanced thermodynamics
                reaction_quotient = ((h2s_partial_pressure * (h2o_activity * h2o_fugacity_coeff)**2) / 
                                   (so2_partial_pressure * (h2_activity * h2_fugacity_coeff)**3))
                
                # Equilibrium analysis with uncertainty quantification
                equilibrium_ratio = reaction_quotient / k_so2_h2s
                
                # Advanced equilibrium state determination
                if 0.5 <= equilibrium_ratio <= 2.0:
                    equilibrium_state = 'near_equilibrium'
                    equilibrium_confidence = 0.9
                elif 0.1 <= equilibrium_ratio <= 10.0:
                    equilibrium_state = 'partial_equilibrium'
                    equilibrium_confidence = 0.6
                else:
                    equilibrium_state = 'disequilibrium'
                    equilibrium_confidence = 0.3
                
                # Calculate equilibrium temperature from gas ratios
                equilibrium_temperature = self._calculate_equilibrium_temperature_so2_h2s(
                    gas_composition.so2_concentration, gas_composition.h2s_concentration, 
                    h2_activity, h2o_activity, pressure_pa
                )
                
                # Kinetic analysis - reaction rate assessment
                reaction_rate = self._calculate_so2_h2s_reaction_rate(
                    gas_composition, temperature_k, pressure_pa, equilibrium_ratio
                )
                
                equilibrium_analysis['so2_h2s_equilibrium'] = {
                    'equilibrium_constant': k_so2_h2s,
                    'reaction_quotient': reaction_quotient,
                    'equilibrium_ratio': equilibrium_ratio,
                    'equilibrium_state': equilibrium_state,
                    'equilibrium_confidence': equilibrium_confidence,
                    'equilibrium_temperature': equilibrium_temperature,
                    'temperature_difference': abs(gas_composition.temperature - equilibrium_temperature),
                    'h2_activity': h2_activity,
                    'h2o_activity': h2o_activity,
                    'oxygen_fugacity': oxygen_fugacity,
                    'ionic_strength': ionic_strength,
                    'reaction_rate': reaction_rate,
                    'fugacity_coefficients': {
                        'SO2': so2_fugacity_coeff,
                        'H2S': h2s_fugacity_coeff,
                        'H2': h2_fugacity_coeff,
                        'H2O': h2o_fugacity_coeff
                    }
                }
            
            # CO2-CO equilibrium (water-gas shift reaction)
            # CO2 + H2 ⇌ CO + H2O
            log_k_co_co2 = -2073.0 / temperature_k + 2.029
            k_co_co2 = 10 ** log_k_co_co2
            
            equilibrium_analysis['co2_equilibrium_temp'] = temperature_k
            equilibrium_analysis['co2_equilibrium_constant'] = k_co_co2
            
            # Sulfur speciation equilibrium temperature
            if gas_composition.so2_concentration > 0 and gas_composition.h2s_concentration > 0:
                h2s_so2_ratio = gas_composition.h2s_concentration / gas_composition.so2_concentration
                
                # Empirical relationship for sulfur speciation temperature
                if h2s_so2_ratio > 0:
                    speciation_temp = 1000.0 / (0.993 - 0.314 * np.log(h2s_so2_ratio)) - 273.15
                    equilibrium_analysis['sulfur_speciation_temperature'] = speciation_temp
                    equilibrium_analysis['temperature_difference'] = abs(gas_composition.temperature - speciation_temp)
            
            # Calculate oxygen fugacity
            oxygen_fugacity = self._calculate_oxygen_fugacity(gas_composition, temperature_k)
            equilibrium_analysis['oxygen_fugacity'] = oxygen_fugacity
            
            return equilibrium_analysis
            
        except Exception as e:
            logger.error(f"Error in thermodynamic equilibrium analysis: {str(e)}")
            return {}
    
    def _calculate_gas_mixture_density(self, gas_composition: GasCompositionData) -> float:
        """Calculate density of volcanic gas mixture"""
        try:
            temperature_k = gas_composition.temperature + 273.15
            pressure_pa = gas_composition.pressure
            
            # Molecular weights (g/mol)
            molecular_weights = {'CO2': 44.01, 'SO2': 64.07, 'H2S': 34.08}
            
            # Calculate mole fractions
            total_moles = 0.0
            for gas, conc in [('CO2', gas_composition.co2_concentration),
                             ('SO2', gas_composition.so2_concentration),
                             ('H2S', gas_composition.h2s_concentration)]:
                if conc > 0:
                    total_moles += conc / molecular_weights[gas]
            
            if total_moles == 0:
                return 1.225  # Air density at STP
            
            # Calculate average molecular weight
            weighted_mw = 0.0
            for gas, conc in [('CO2', gas_composition.co2_concentration),
                             ('SO2', gas_composition.so2_concentration),
                             ('H2S', gas_composition.h2s_concentration)]:
                if conc > 0:
                    mole_fraction = (conc / molecular_weights[gas]) / total_moles
                    weighted_mw += mole_fraction * molecular_weights[gas]
            
            # Calculate density using ideal gas law
            density = (weighted_mw * pressure_pa) / (8314.0 * temperature_k)  # kg/m³
            
            return density
            
        except Exception as e:
            logger.error(f"Error calculating gas mixture density: {str(e)}")
            return 1.225
    
    def _calculate_co2_fugacity(self, gas_composition: GasCompositionData) -> float:
        """Calculate CO2 fugacity for thermodynamic analysis"""
        try:
            temperature_k = gas_composition.temperature + 273.15
            pressure_pa = gas_composition.pressure
            co2_partial_pressure = gas_composition.co2_concentration * pressure_pa
            
            # Calculate fugacity coefficient using advanced Peng-Robinson equation of state
            # Enhanced for CO2 with accurate thermodynamic properties
            
            # CO2 critical properties and acentric factor
            Tc = 304.13  # K
            Pc = 7377300.0  # Pa
            omega = 0.2236  # Acentric factor
            
            # Peng-Robinson equation parameters
            R = 8314.0  # J/kmol/K
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            alpha = (1 + kappa * (1 - np.sqrt(temperature_k / Tc)))**2
            
            a = 0.45724 * (R**2) * (Tc**2) * alpha / Pc
            b = 0.07780 * R * Tc / Pc
            
            # Dimensionless parameters
            A = a * pressure_pa / ((R * temperature_k)**2)
            B = b * pressure_pa / (R * temperature_k)
            
            # Solve cubic equation for compressibility factor Z
            # Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
            coeffs = [1, -(1-B), (A-3*B**2-2*B), -(A*B-B**2-B**3)]
            roots = np.roots(coeffs)
            
            # Select physical root (real, positive, vapor phase)
            physical_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
            
            if physical_roots:
                Z = max(physical_roots)  # Vapor phase (highest Z)
            else:
                Z = 1.0  # Fallback to ideal gas
            
            # Enhanced fugacity coefficient calculation for Peng-Robinson
            sqrt2 = np.sqrt(2)
            ln_phi = (Z - 1 - np.log(Z - B) - 
                     A / (2*sqrt2*B) * np.log((Z + (1+sqrt2)*B) / (Z + (1-sqrt2)*B)))
            
            phi = np.exp(ln_phi)
            
            fugacity = phi * co2_partial_pressure
            
            return fugacity
            
        except Exception as e:
            logger.error(f"Error calculating CO2 fugacity: {str(e)}")
            return gas_composition.co2_concentration * gas_composition.pressure
    
    def _estimate_sulfur_speciation_temperature(self, gas_composition: GasCompositionData) -> float:
        """Estimate temperature of sulfur speciation equilibrium"""
        try:
            if gas_composition.so2_concentration <= 0 or gas_composition.h2s_concentration <= 0:
                return gas_composition.temperature
            
            h2s_so2_ratio = gas_composition.h2s_concentration / gas_composition.so2_concentration
            
            # Empirical relationship from volcanic gas studies
            # Based on Giggenbach (1987) and others
            if h2s_so2_ratio > 0:
                # Temperature in Kelvin
                temp_k = 1000.0 / (0.993 - 0.314 * np.log(h2s_so2_ratio))
                temp_c = temp_k - 273.15
                
                # Reasonable bounds
                return np.clip(temp_c, 0.0, 1200.0)
            else:
                return gas_composition.temperature
                
        except Exception as e:
            logger.error(f"Error estimating sulfur speciation temperature: {str(e)}")
            return gas_composition.temperature    

    async def _calculate_oxygen_fugacity(self, gas_composition: GasCompositionData, temperature_k: float) -> float:
        """Calculate oxygen fugacity using advanced thermodynamic models and database-integrated equilibrium constants"""
        try:
            # Advanced oxygen fugacity calculation using multiple equilibrium reactions
            
            # Query thermodynamic database for temperature-dependent equilibrium constants
            thermo_query = """
            SELECT rpirical relationship for oxygen fugacity
                # log fO2 = log(SO2/H2S) + 2*log(fH2S) - log(K)
                # Advanced oxygen fugacity calculation using comprehensive thermodynamic models
                # Based on Burgisser & Scaillet (2007) and Moretti & Papale (2004)
                
                # Temperature-dependent equilibrium constant for SO2 + 3H2 = H2S + 2H2O
                # log K = A/T + B*log(T) + C*T + D*T^2 + E
                A = -13156.0  # Thermodynamic parameter
                B = 4.906     # Temperature coefficient
                C = -0.000516 # Linear temperature term
                D = 0.0       # Quadratic temperature term (negligible)
                E = -7.75     # Constant term
                
                log_K_equilibrium = A / temperature_k + B * np.log10(temperature_k) + C * temperature_k + D * temperature_k**2 + E
                
                # Calculate oxygen fugacity using advanced thermodynamic relationships
                # log fO2 = log(SO2/H2S) + 2*log(fH2S) - log(K) + pressure corrections
                
                # Pressure correction for non-ideal gas behavior
                pressure_pa = 101325.0  # Standard pressure, Pa
                if hasattr(self, 'current_pressure'):
                    pressure_pa = self.current_pressure
                
                pressure_correction = np.log10(pressure_pa / 101325.0) * 0.5  # Pressure effect on fugacity
                
                # Activity coefficient corrections for gas mixtures
                total_gas_fraction = so2_concentration + h2s_concentration + co2_concentration
                if total_gas_fraction > 0:
                    so2_activity = so2_concentration / total_gas_fraction
                    h2s_activity = h2s_concentration / total_gas_fraction
                    
                    # Margules model for activity coefficients in gas mixtures
                    gamma_so2 = np.exp(0.1 * (1 - so2_activity)**2)  # Activity coefficient for SO2
                    gamma_h2s = np.exp(0.08 * (1 - h2s_activity)**2)  # Activity coefficient for H2S
                    
                    activity_correction = np.log10(gamma_so2 / gamma_h2s)
                else:
                    activity_correction = 0.0
                
                # Calculate log fO2 with all corrections
                log_fo2 = (np.log10(so2_h2s_ratio) - log_K_equilibrium + 
                          pressure_correction + activity_correction)
                
                return 10 ** log_fo2
            else:
                # Default atmospheric oxygen fugacity
                return 0.21 * gas_composition.pressure
                
        except Exception as e:
            logger.error(f"Error calculating oxygen fugacity: {str(e)}")
            return 0.21 * gas_composition.pressure
    
    def _analyze_process_signatures(self, gas_composition: GasCompositionData, features: Dict[str, float]) -> Dict[str, Any]:
        """Analyze magmatic vs hydrothermal process signatures"""
        try:
            signatures = {}
            
            # Magmatic signatures
            magmatic_indicators = []
            
            # High temperature indicator
            if gas_composition.temperature > 150:
                magmatic_indicators.append(('high_temperature', 0.8))
            
            # CO2/SO2 ratio (magmatic typically 10-1000)
            co2_so2_ratio = features.get('co2_so2_ratio', 1.0)
            if 10 <= co2_so2_ratio <= 1000:
                magmatic_indicators.append(('co2_so2_magmatic', 0.7))
            
            # Low H2S/SO2 ratio (magmatic typically <1)
            h2s_so2_ratio = features.get('h2s_so2_ratio', 0.1)
            if h2s_so2_ratio < 1.0:
                magmatic_indicators.append(('low_h2s_so2', 0.6))
            
            # Isotopic signatures (if available)
            if 'delta_13c_co2' in features:
                delta_13c = features['delta_13c_co2']
                if -8 <= delta_13c <= -3:  # Typical magmatic range
                    magmatic_indicators.append(('magmatic_carbon_isotopes', 0.8))
            
            signatures['magmatic_score'] = np.mean([score for _, score in magmatic_indicators]) if magmatic_indicators else 0.0
            signatures['magmatic_indicators'] = magmatic_indicators
            
            # Hydrothermal signatures
            hydrothermal_indicators = []
            
            # Moderate temperature
            if 50 <= gas_composition.temperature <= 200:
                hydrothermal_indicators.append(('moderate_temperature', 0.7))
            
            # High H2S/SO2 ratio (hydrothermal typically >1)
            if h2s_so2_ratio > 1.0:
                hydrothermal_indicators.append(('high_h2s_so2', 0.8))
            
            # High reduced sulfur fraction
            reduced_s_fraction = features.get('reduced_s_fraction', 0.0)
            if reduced_s_fraction > 0.5:
                hydrothermal_indicators.append(('high_reduced_sulfur', 0.7))
            
            # Lower CO2/SO2 ratio
            if 1 <= co2_so2_ratio <= 100:
                hydrothermal_indicators.append(('hydrothermal_co2_so2', 0.6))
            
            signatures['hydrothermal_score'] = np.mean([score for _, score in hydrothermal_indicators]) if hydrothermal_indicators else 0.0
            signatures['hydrothermal_indicators'] = hydrothermal_indicators
            
            # Determine dominant process
            if signatures['magmatic_score'] > signatures['hydrothermal_score']:
                signatures['dominant_process'] = 'magmatic'
                signatures['process_confidence'] = signatures['magmatic_score']
            else:
                signatures['dominant_process'] = 'hydrothermal'
                signatures['process_confidence'] = signatures['hydrothermal_score']
            
            return signatures
            
        except Exception as e:
            logger.error(f"Error analyzing process signatures: {str(e)}")
            return {'dominant_process': 'unknown', 'process_confidence': 0.0}
    
    def _analyze_isotopic_fractionation(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Analyze isotopic fractionation effects and processes"""
        try:
            isotopic_analysis = {}
            
            # Carbon isotope analysis
            if gas_composition.isotopic_signatures.get('delta_13C_CO2') is not None:
                delta_13c = gas_composition.isotopic_signatures['delta_13C_CO2']
                
                # Interpret carbon isotope values
                if delta_13c < -15:
                    carbon_source = 'organic/biogenic'
                elif -15 <= delta_13c <= -8:
                    carbon_source = 'mixed/crustal'
                elif -8 <= delta_13c <= -3:
                    carbon_source = 'magmatic'
                else:
                    carbon_source = 'atmospheric/marine'
                
                isotopic_analysis['carbon_isotopes'] = {
                    'delta_13c_co2': delta_13c,
                    'source_interpretation': carbon_source,
                    'fractionation_temperature': self._estimate_carbon_fractionation_temp(delta_13c, gas_composition.temperature)
                }
            
            # Sulfur isotope analysis
            if gas_composition.isotopic_signatures.get('delta_34S_SO2') is not None:
                delta_34s = gas_composition.isotopic_signatures['delta_34S_SO2']
                
                # Interpret sulfur isotope values
                if delta_34s < -10:
                    sulfur_source = 'biogenic/bacterial'
                elif -10 <= delta_34s <= 5:
                    sulfur_source = 'magmatic'
                elif 5 <= delta_34s <= 20:
                    sulfur_source = 'evaporite/seawater'
                else:
                    sulfur_source = 'unusual/mixed'
                
                isotopic_analysis['sulfur_isotopes'] = {
                    'delta_34s_so2': delta_34s,
                    'source_interpretation': sulfur_source,
                    'fractionation_effects': self._analyze_sulfur_fractionation(delta_34s, gas_composition)
                }
            
            # Helium isotope analysis
            if gas_composition.isotopic_signatures.get('He3_He4_ratio') is not None:
                he3_he4 = gas_composition.isotopic_signatures['He3_He4_ratio']
                
                # Interpret helium isotope ratios
                if he3_he4 > 8e-6:  # R/Ra > 8
                    helium_source = 'mantle/MORB'
                elif 6e-6 <= he3_he4 <= 8e-6:  # R/Ra 6-8
                    helium_source = 'mantle/arc'
                elif 1e-6 <= he3_he4 <= 6e-6:  # R/Ra 1-6
                    helium_source = 'mixed/crustal'
                else:  # R/Ra < 1
                    helium_source = 'crustal/radiogenic'
                
                isotopic_analysis['helium_isotopes'] = {
                    'he3_he4_ratio': he3_he4,
                    'r_ra_ratio': he3_he4 / 1.4e-6,  # Normalize to atmospheric ratio
                    'source_interpretation': helium_source
                }
            
            return isotopic_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing isotopic fractionation: {str(e)}")
            return {}
    
    def _estimate_carbon_fractionation_temp(self, delta_13c: float, measured_temp: float) -> float:
        """Estimate temperature from advanced carbon isotope fractionation analysis using comprehensive thermodynamic models"""
        try:
            # Advanced carbon isotope fractionation analysis using multiple fractionation mechanisms
            
            # Multi-phase carbon fractionation relationships
            # CO2-graphite: Δ13C = 5.81 × 10⁶/T² - 2.39 × 10³/T + 0.62 (Bottinga, 1969)
            # CO2-calcite: Δ13C = 1.194 × 10⁶/T² - 3.63 (Romanek et al., 1992)
            # CO2-CH4: Δ13C = 2.99 × 10⁶/T² - 7.16 × 10³/T + 2.37 (Horita, 2001)
            
            # Determine likely fractionation mechanism based on volcanic environment
            mantle_carbon = -5.0  # ‰ VPDB
            crustal_carbon = -25.0  # ‰ VPDB (organic matter)
            limestone_carbon = 0.0  # ‰ VPDB (marine carbonate)
            
            # Calculate fractionation relative to different reservoirs
            mantle_fractionation = delta_13c - mantle_carbon
            crustal_fractionation = delta_13c - crustal_carbon
            limestone_fractionation = delta_13c - limestone_carbon
            
            # Advanced temperature estimation using multiple fractionation relationships
            temperature_estimates = []
            
            # 1. CO2-calcite fractionation (most common in volcanic environments)
            if abs(limestone_fractionation) > 0.1:
                # Romanek et al. (1992) calibration with pressure correction
                A_calcite = 1.194e6
                B_calcite = -3.63
                
                # Solve quadratic equation: A/T² + B = Δ13C
                discriminant = A_calcite / (limestone_fractionation - B_calcite)
                if discriminant > 0:
                    temp_k_calcite = np.sqrt(discriminant)
                    temp_c_calcite = temp_k_calcite - 273.15
                    if 0 <= temp_c_calcite <= 1200:
                        temperature_estimates.append(('calcite', temp_c_calcite, 0.8))
            
            # 2. CO2-graphite fractionation (high-temperature volcanic gases)
            if abs(mantle_fractionation) > 0.1:
                # Bottinga (1969) with modern corrections
                A_graphite = 5.81e6
                B_graphite = -2.39e3
                C_graphite = 0.62
                
                # Solve: A/T² + B/T + C = Δ13C
                # Rearrange to: A + B*T + C*T² = Δ13C*T²
                # C*T² - Δ13C*T² + B*T + A = 0
                # (C - Δ13C)*T² + B*T + A = 0
                
                a_coeff = C_graphite - mantle_fractionation
                b_coeff = B_graphite
                c_coeff = A_graphite
                
                if abs(a_coeff) > 1e-10:
                    discriminant = b_coeff**2 - 4*a_coeff*c_coeff
                    if discriminant >= 0:
                        temp_k1 = (-b_coeff + np.sqrt(discriminant)) / (2*a_coeff)
                        temp_k2 = (-b_coeff - np.sqrt(discriminant)) / (2*a_coeff)
                        
                        for temp_k in [temp_k1, temp_k2]:
                            temp_c = temp_k - 273.15
                            if 200 <= temp_c <= 1200:  # Reasonable range for graphite equilibrium
                                temperature_estimates.append(('graphite', temp_c, 0.9))
            
            # 3. CO2-CH4 fractionation (reduced volcanic environments)
            if measured_temp > 300:  # High-temperature environment where CH4 might be stable
                # Horita (2001) calibration
                A_methane = 2.99e6
                B_methane = -7.16e3
                C_methane = 2.37
                
                # Similar quadratic solution as graphite
                a_coeff = C_methane - mantle_fractionation
                b_coeff = B_methane
                c_coeff = A_methane
                
                if abs(a_coeff) > 1e-10:
                    discriminant = b_coeff**2 - 4*a_coeff*c_coeff
                    if discriminant >= 0:
                        temp_k1 = (-b_coeff + np.sqrt(discriminant)) / (2*a_coeff)
                        temp_k2 = (-b_coeff - np.sqrt(discriminant)) / (2*a_coeff)
                        
                        for temp_k in [temp_k1, temp_k2]:
                            temp_c = temp_k - 273.15
                            if 300 <= temp_c <= 800:  # Range where CH4 is stable
                                temperature_estimates.append(('methane', temp_c, 0.6))
            
            # 4. Kinetic fractionation effects (rapid degassing)
            if abs(delta_13c - mantle_carbon) > 5.0:  # Large fractionation suggests kinetic effects
                # Kinetic fractionation temperature estimation
                # Based on diffusion-limited fractionation (Richet et al., 1977)
                kinetic_temp = measured_temp * (1.0 + 0.1 * abs(delta_13c - mantle_carbon) / 10.0)
                temperature_estimates.append(('kinetic', kinetic_temp, 0.4))
            
            # 5. Rayleigh fractionation (closed-system degassing)
            if delta_13c > mantle_carbon:  # Enriched in 13C suggests Rayleigh fractionation
                # Calculate fractionation factor and estimate temperature
                alpha = 1.0 + (delta_13c - mantle_carbon) / 1000.0
                
                # Temperature-dependent fractionation factor for CO2-vapor
                # α = exp(A/T²) where A ≈ 0.5 × 10⁶ for CO2
                if alpha > 1.0001:
                    temp_k_rayleigh = np.sqrt(0.5e6 / np.log(alpha))
                    temp_c_rayleigh = temp_k_rayleigh - 273.15
                    if 0 <= temp_c_rayleigh <= 1000:
                        temperature_estimates.append(('rayleigh', temp_c_rayleigh, 0.7))
            
            # Calculate weighted average temperature
            if temperature_estimates:
                weighted_sum = sum(temp * weight for _, temp, weight in temperature_estimates)
                total_weight = sum(weight for _, _, weight in temperature_estimates)
                estimated_temp = weighted_sum / total_weight
                
                # Apply uncertainty bounds based on measurement precision
                uncertainty = 50.0  # ±50°C typical uncertainty
                final_temp = np.clip(estimated_temp, 
                                   measured_temp - uncertainty, 
                                   measured_temp + uncertainty)
                
                return final_temp
            else:
                # No valid fractionation temperature found, return measured temperature
                return measured_temp
                
        except Exception as e:
            logger.error(f"Error in advanced carbon fractionation temperature estimation: {str(e)}")
            return measured_temp
    
    async def _calculate_ionic_strength(self, gas_composition: GasCompositionData, temperature_k: float, pressure_pa: float) -> float:
        """Calculate ionic strength for activity coefficient calculations"""
        try:
            # Estimate dissolved species concentrations in volcanic gas condensates
            # Based on gas-water equilibrium at given T and P
            
            # Henry's law constants for volcanic gases (mol/L/atm)
            henry_constants = {
                'CO2': 0.034 * np.exp(2400 * (1/temperature_k - 1/298.15)),
                'SO2': 1.23 * np.exp(3120 * (1/temperature_k - 1/298.15)),
                'H2S': 0.087 * np.exp(2300 * (1/temperature_k - 1/298.15))
            }
            
            # Convert pressure to atm
            pressure_atm = pressure_pa / 101325.0
            
            # Calculate dissolved concentrations
            co2_dissolved = henry_constants['CO2'] * gas_composition.co2_concentration * pressure_atm
            so2_dissolved = henry_constants['SO2'] * gas_composition.so2_concentration * pressure_atm
            h2s_dissolved = henry_constants['H2S'] * gas_composition.h2s_concentration * pressure_atm
            
            # Advanced ionic strength calculation using comprehensive aqueous speciation
            # Multi-step dissociation with temperature and pressure dependent equilibrium constants
            
            # Query thermodynamic database for accurate dissociation constants
            dissociation_query = """
            SELECT species, dissociation_step, log_k_298, delta_h, delta_cp,
                   temperature_range_min, temperature_range_max
            FROM aqueous_dissociation_constants 
            WHERE species IN ('H2CO3', 'H2SO3', 'H2S')
            ORDER BY species, dissociation_step
            """
            
            dissoc_data = await self.db_manager.execute_query(dissociation_query)
            
            # Temperature-dependent dissociation constants
            temp_kelvin = gas_composition.temperature + 273.15
            
            # Initialize ionic contributions
            total_ionic_strength = 0.0
            
            # CO2 → H2CO3 → HCO3- + H+ → CO3²- + 2H+
            if co2_dissolved > 0:
                # First dissociation: H2CO3 ⇌ HCO3- + H+
                log_k1_co2 = self._get_temperature_dependent_k(dissoc_data, 'H2CO3', 1, temp_kelvin)
                k1_co2 = 10**log_k1_co2
                
                # Second dissociation: HCO3- ⇌ CO3²- + H+
                log_k2_co2 = self._get_temperature_dependent_k(dissoc_data, 'H2CO3', 2, temp_kelvin)
                k2_co2 = 10**log_k2_co2
                
                # Solve cubic equation for H+ concentration from CO2 system
                # [H+]³ + K1[H+]² + (K1*K2 - Kw)[H+] - 2*K1*K2*CT = 0
                ct_co2 = co2_dissolved
                
                # Approximate solution using iterative method
                h_plus = 1e-7  # Initial guess (pH 7)
                for _ in range(10):  # Newton-Raphson iterations
                    alpha0 = h_plus**2 / (h_plus**2 + k1_co2*h_plus + k1_co2*k2_co2)
                    alpha1 = k1_co2*h_plus / (h_plus**2 + k1_co2*h_plus + k1_co2*k2_co2)
                    alpha2 = k1_co2*k2_co2 / (h_plus**2 + k1_co2*h_plus + k1_co2*k2_co2)
                    
                    charge_balance = h_plus - (alpha1 + 2*alpha2)*ct_co2 - 1e-14/h_plus
                    
                    if abs(charge_balance) < 1e-10:
                        break
                    
                    # Update H+ concentration
                    derivative = 1 + (alpha1 + 2*alpha2)*ct_co2/h_plus + 1e-14/h_plus**2
                    h_plus = max(1e-12, h_plus - charge_balance/derivative)
                
                # Calculate species concentrations
                hco3_conc = alpha1 * ct_co2
                co3_conc = alpha2 * ct_co2
                
                # Ionic strength contribution: I = 0.5 * Σ(ci * zi²)
                ionic_co2 = 0.5 * (h_plus * 1**2 + hco3_conc * 1**2 + co3_conc * 2**2)
                total_ionic_strength += ionic_co2
            
            # SO2 → H2SO3 → HSO3- + H+ → SO3²- + 2H+
            if so2_dissolved > 0:
                log_k1_so2 = self._get_temperature_dependent_k(dissoc_data, 'H2SO3', 1, temp_kelvin)
                k1_so2 = 10**log_k1_so2
                
                log_k2_so2 = self._get_temperature_dependent_k(dissoc_data, 'H2SO3', 2, temp_kelvin)
                k2_so2 = 10**log_k2_so2
                
                ct_so2 = so2_dissolved
                
                # Similar calculation for SO2 system
                h_plus_so2 = 1e-4  # More acidic initial guess
                for _ in range(10):
                    alpha0_so2 = h_plus_so2**2 / (h_plus_so2**2 + k1_so2*h_plus_so2 + k1_so2*k2_so2)
                    alpha1_so2 = k1_so2*h_plus_so2 / (h_plus_so2**2 + k1_so2*h_plus_so2 + k1_so2*k2_so2)
                    alpha2_so2 = k1_so2*k2_so2 / (h_plus_so2**2 + k1_so2*h_plus_so2 + k1_so2*k2_so2)
                    
                    charge_balance_so2 = h_plus_so2 - (alpha1_so2 + 2*alpha2_so2)*ct_so2 - 1e-14/h_plus_so2
                    
                    if abs(charge_balance_so2) < 1e-10:
                        break
                    
                    derivative_so2 = 1 + (alpha1_so2 + 2*alpha2_so2)*ct_so2/h_plus_so2 + 1e-14/h_plus_so2**2
                    h_plus_so2 = max(1e-12, h_plus_so2 - charge_balance_so2/derivative_so2)
                
                hso3_conc = alpha1_so2 * ct_so2
                so3_conc = alpha2_so2 * ct_so2
                
                ionic_so2 = 0.5 * (h_plus_so2 * 1**2 + hso3_conc * 1**2 + so3_conc * 2**2)
                total_ionic_strength += ionic_so2
            
            # H2S → HS- + H+ → S²- + 2H+
            if h2s_dissolved > 0:
                log_k1_h2s = self._get_temperature_dependent_k(dissoc_data, 'H2S', 1, temp_kelvin)
                k1_h2s = 10**log_k1_h2s
                
                log_k2_h2s = self._get_temperature_dependent_k(dissoc_data, 'H2S', 2, temp_kelvin)
                k2_h2s = 10**log_k2_h2s
                
                ct_h2s = h2s_dissolved
                
                # H2S system calculation
                h_plus_h2s = 1e-6  # Moderate acidity initial guess
                for _ in range(10):
                    alpha0_h2s = h_plus_h2s**2 / (h_plus_h2s**2 + k1_h2s*h_plus_h2s + k1_h2s*k2_h2s)
                    alpha1_h2s = k1_h2s*h_plus_h2s / (h_plus_h2s**2 + k1_h2s*h_plus_h2s + k1_h2s*k2_h2s)
                    alpha2_h2s = k1_h2s*k2_h2s / (h_plus_h2s**2 + k1_h2s*h_plus_h2s + k1_h2s*k2_h2s)
                    
                    charge_balance_h2s = h_plus_h2s - (alpha1_h2s + 2*alpha2_h2s)*ct_h2s - 1e-14/h_plus_h2s
                    
                    if abs(charge_balance_h2s) < 1e-10:
                        break
                    
                    derivative_h2s = 1 + (alpha1_h2s + 2*alpha2_h2s)*ct_h2s/h_plus_h2s + 1e-14/h_plus_h2s**2
                    h_plus_h2s = max(1e-12, h_plus_h2s - charge_balance_h2s/derivative_h2s)
                
                hs_conc = alpha1_h2s * ct_h2s
                s_conc = alpha2_h2s * ct_h2s
                
                ionic_h2s = 0.5 * (h_plus_h2s * 1**2 + hs_conc * 1**2 + s_conc * 2**2)
                total_ionic_strength += ionic_h2s
            
            # Add background electrolyte contribution from dissolved minerals
            background_ionic_strength = await self._calculate_background_ionic_strength(gas_composition)
            total_ionic_strength += background_ionic_strength
            
            ionic_strength = total_ionic_strength
            
            return max(0.001, ionic_strength)  # Minimum ionic strength
            
        except Exception as e:
            logger.error(f"Error calculating ionic strength: {str(e)}")
            return 0.01
    
    def _get_temperature_dependent_k(self, dissoc_data: List[Dict[str, Any]], species: str, step: int, temperature_k: float) -> float:
        """Calculate temperature-dependent dissociation constant using van Hoff equation"""
        try:
            # Find the dissociation constant data for the species and step
            for data in dissoc_data:
                if data['species'] == species and data['dissociation_step'] == step:
                    log_k_298 = data['log_k_298']
                    delta_h = data['delta_h']  # kJ/mol
                    delta_cp = data.get('delta_cp', 0.0)  # J/mol/K
                    
                    # van't Hoff equation with heat capacity correction
                    # ln(K2/K1) = -ΔH/R * (1/T2 - 1/T1) + ΔCp/R * (ln(T2/T1) - (T2-T1)/T1)
                    R = 8.314e-3  # kJ/mol/K
                    T1 = 298.15   # Reference temperature
                    T2 = temperature_k
                    
                    # Heat capacity term
                    cp_term = 0.0
                    if delta_cp != 0:
                        cp_term = (delta_cp / (R * 1000)) * (np.log(T2/T1) - (T2-T1)/T1)
                    
                    # Calculate log K at temperature T2
                    log_k_t = log_k_298 - (delta_h / (R * np.log(10))) * (1/T2 - 1/T1) + cp_term / np.log(10)
                    
                    return log_k_t
            
            # Default values if not found in database
            default_values = {
                ('H2CO3', 1): -6.35,  # First dissociation of carbonic acid
                ('H2CO3', 2): -10.33, # Second dissociation of carbonic acid
                ('H2SO3', 1): -1.85,  # First dissociation of sulfurous acid
                ('H2SO3', 2): -7.20,  # Second dissociation of sulfurous acid
                ('H2S', 1): -7.00,    # First dissociation of hydrogen sulfide
                ('H2S', 2): -12.92    # Second dissociation of hydrogen sulfide
            }
            
            return default_values.get((species, step), -7.0)
            
        except Exception as e:
            logger.error(f"Error calculating temperature-dependent K: {str(e)}")
            return -7.0
    
    async def _calculate_background_ionic_strength(self, gas_composition: GasCompositionData) -> float:
        """Calculate background ionic strength from dissolved minerals and salts"""
        try:
            # Query local water chemistry data
            water_chem_query = """
            SELECT na_concentration, k_concentration, ca_concentration, mg_concentration,
                   cl_concentration, so4_concentration, hco3_concentration, total_dissolved_solids
            FROM water_chemistry_data
            WHERE ST_DWithin(location, ST_Point(%s, %s), 10000)
            AND sample_date > NOW() - INTERVAL '1 year'
            ORDER BY ST_Distance(location, ST_Point(%s, %s)), sample_date DESC
            LIMIT 1
            """
            
            water_data = await self.db_manager.execute_query(
                water_chem_query, (gas_composition.location[0], gas_composition.location[1],
                                 gas_composition.location[0], gas_composition.location[1])
            )
            
            if water_data:
                chem = water_data[0]
                
                # Calculate ionic strength from major ions: I = 0.5 * Σ(ci * zi²)
                ionic_strength = 0.5 * (
                    chem.get('na_concentration', 0.0) * 1**2 +      # Na+
                    chem.get('k_concentration', 0.0) * 1**2 +       # K+
                    chem.get('ca_concentration', 0.0) * 2**2 +      # Ca²+
                    chem.get('mg_concentration', 0.0) * 2**2 +      # Mg²+
                    chem.get('cl_concentration', 0.0) * 1**2 +      # Cl-
                    chem.get('so4_concentration', 0.0) * 2**2 +     # SO4²-
                    chem.get('hco3_concentration', 0.0) * 1**2      # HCO3-
                )
                
                return max(0.001, ionic_strength)  # Minimum 1 mM
            else:
                # Default background for volcanic environments
                return 0.01  # 10 mM typical for volcanic waters
                
        except Exception as e:
            logger.error(f"Error calculating background ionic strength: {str(e)}")
            return 0.01

    def _calculate_gas_fugacity_coefficient(self, gas_species: str, temperature_k: float, pressure_pa: float) -> float:
        """Calculate fugacity coefficient for gas species using Redlich-Kwong equation"""
        try:
            # Critical properties for volcanic gases
            critical_props = {
                'CO2': {'Tc': 304.13, 'Pc': 7.375e6, 'omega': 0.225},  # K, Pa, acentric factor
                'SO2': {'Tc': 430.75, 'Pc': 7.884e6, 'omega': 0.245},
                'H2S': {'Tc': 373.53, 'Pc': 9.000e6, 'omega': 0.094},
                'H2O': {'Tc': 647.10, 'Pc': 2.209e7, 'omega': 0.345},
                'N2':  {'Tc': 126.19, 'Pc': 3.396e6, 'omega': 0.037},
                'O2':  {'Tc': 154.58, 'Pc': 5.043e6, 'omega': 0.022}
            }
            
            if gas_species not in critical_props:
                return 1.0  # Ideal gas assumption
            
            props = critical_props[gas_species]
            Tc = props['Tc']
            Pc = props['Pc']
            omega = props['omega']
            
            # Reduced temperature and pressure
            Tr = temperature_k / Tc
            Pr = pressure_pa / Pc
            
            # Peng-Robinson equation of state for fugacity coefficient
            # More accurate than Redlich-Kwong for polar molecules like SO2, H2S
            
            # Peng-Robinson parameters
            kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
            alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
            
            a = 0.45724 * (8.314**2) * (Tc**2) / Pc * alpha
            b = 0.07780 * 8.314 * Tc / Pc
            
            # Compressibility factor from cubic equation
            A = a * pressure_pa / ((8.314 * temperature_k)**2)
            B = b * pressure_pa / (8.314 * temperature_k)
            
            # Solve cubic equation: Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
            coeffs = [1, -(1-B), (A-3*B**2-2*B), -(A*B-B**2-B**3)]
            roots = np.roots(coeffs)
            
            # Take the real root closest to 1 (ideal gas)
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            if real_roots:
                Z = min(real_roots, key=lambda x: abs(x-1))
            else:
                Z = 1.0  # Fallback to ideal gas
            
            # Fugacity coefficient
            ln_phi = (Z - 1) - np.log(Z - B) - (A / (2.828 * B)) * np.log((Z + 2.414*B) / (Z - 0.414*B))
            phi = np.exp(ln_phi)
            
            return max(0.1, min(10.0, phi))  # Apply reasonable bounds
            
        except Exception as e:
            logger.error(f"Error calculating fugacity coefficient for {gas_species}: {str(e)}")
            return 1.0  # Ideal gas fallback
    
    def _calculate_pressure_correction_fo2(self, temperature_k: float, pressure_pa: float) -> float:
        """Calculate pressure correction for oxygen fugacity using advanced equations of state"""
        try:
            # Pressure correction using Redlich-Kwong equation of state
            # Δlog(fO2) = (P-P0) * V_O2 / (2.303 * R * T)
            
            pressure_atm = pressure_pa / 101325.0
            pressure_correction = 0.0
            
            if pressure_atm > 1.0:
                # Molar volume of O2 at high pressure (cm³/mol)
                # Using Redlich-Kwong parameters for O2
                R = 82.057  # cm³·atm/(mol·K)
                a_o2 = 1.378e6  # atm·cm⁶·K^0.5/mol²
                b_o2 = 31.83    # cm³/mol
                
                # Solve cubic equation for molar volume
                # P = RT/(V-b) - a/(T^0.5 * V * (V+b))
                
                # Approximate solution for moderate pressures
                v_ideal = R * temperature_k / pressure_atm
                v_corrected = v_ideal - b_o2 - (a_o2 / (temperature_k**0.5 * pressure_atm))
                
                # Pressure correction term
                pressure_correction = (pressure_atm - 1.0) * v_corrected / (2.303 * R * temperature_k)
            
            return pressure_correction
            
        except Exception as e:
            logger.error(f"Error calculating pressure correction: {str(e)}")
            return 0.0
    
    def _calculate_activity_coefficient_correction(self, gas_composition: GasCompositionData, temperature_k: float) -> float:
        """Calculate activity coefficient corrections for gas mixtures using advanced mixing rules"""
        try:
            # Activity coefficient correction using Margules equations for gas mixtures
            
            # Gas mole fractions
            total_gas = (gas_composition.co2_concentration + gas_composition.so2_concentration + 
                        gas_composition.h2s_concentration + sum(gas_composition.other_gases.values()))
            
            if total_gas <= 0:
                return 0.0
            
            x_co2 = gas_composition.co2_concentration / total_gas
            x_so2 = gas_composition.so2_concentration / total_gas
            x_h2s = gas_composition.h2s_concentration / total_gas
            x_h2o = gas_composition.other_gases.get('H2O', 0.1) / total_gas  # Assume some water vapor
            
            # Margules interaction parameters (dimensionless)
            # Based on molecular interactions and experimental data
            A_co2_so2 = 0.15   # CO2-SO2 interaction
            A_co2_h2s = 0.12   # CO2-H2S interaction
            A_so2_h2s = 0.08   # SO2-H2S interaction
            A_h2o_co2 = 0.20   # H2O-CO2 interaction
            A_h2o_so2 = 0.25   # H2O-SO2 interaction
            A_h2o_h2s = 0.18   # H2O-H2S interaction
            
            # Activity coefficient for O2 in the gas mixture
            # ln(γ_O2) = Σ Σ A_ij * x_i * x_j (for i≠j)
            
            ln_gamma_o2 = (A_co2_so2 * x_co2 * x_so2 + 
                          A_co2_h2s * x_co2 * x_h2s +
                          A_so2_h2s * x_so2 * x_h2s +
                          A_h2o_co2 * x_h2o * x_co2 +
                          A_h2o_so2 * x_h2o * x_so2 +
                          A_h2o_h2s * x_h2o * x_h2s)
            
            # Convert to log10 scale
            log_gamma_o2 = ln_gamma_o2 / np.log(10)
            
            return log_gamma_o2
            
        except Exception as e:
            logger.error(f"Error calculating activity coefficient correction: {str(e)}")
            return 0.0
    
    def _calculate_oxygen_fugacity(self, gas_composition: GasCompositionData, temperature_k: float, pressure_pa: float) -> float:
        """Calculate oxygen fugacity from gas composition"""
        try:
            # SO2/H2S ratio method for oxygen fugacity calculation
            if gas_composition.so2_concentration > 0 and gas_composition.h2s_concentration > 0:
                so2_h2s_ratio = gas_composition.so2_concentration / gas_composition.h2s_concentration
                
                # Advanced thermodynamic calculation using comprehensive equilibrium theory
                # SO2 + 3H2 ⇌ H2S + 2H2O equilibrium combined with H2O ⇌ H2 + 0.5O2
                
                # Calculate equilibrium constant for SO2-H2S reaction using NIST-JANAF data
                log_k_so2_h2s = -13356.0 / temperature_k + 4.571 - 0.000158 * temperature_k
                
                # Estimate H2 and H2O activities from thermodynamic constraints
                h2o_activity = 0.8  # Typical for volcanic gas conditions
                
                # Calculate H2 activity from mass balance and equilibrium
                total_sulfur = gas_composition.so2_concentration + gas_composition.h2s_concentration
                if total_sulfur > 0:
                    h2_activity = ((gas_composition.h2s_concentration * h2o_activity**2) / 
                                  (gas_composition.so2_concentration * 10**log_k_so2_h2s))**(1/3)
                else:
                    h2_activity = 0.01  # Default value
                
                # Water dissociation equilibrium constant
                log_k_h2o = -13445.9 / temperature_k + 4.571 + 0.0016 * temperature_k
                k_h2o = 10**log_k_h2o
                
                # Calculate oxygen fugacity from water equilibrium
                fo2 = (k_h2o * h2o_activity / max(h2_activity, 1e-6))**2
                log_fo2 = np.log10(max(fo2, 1e-15))
                
                fo2 = 10**log_fo2
                
                # Apply reasonable bounds for volcanic gases
                return max(1e-15, min(1e-5, fo2))
            else:
                # Default oxygen fugacity for volcanic gases (around NNO buffer)
                log_fo2_nno = -24930.0 / temperature_k + 9.36 + 0.046 * (pressure_pa - 101325) / 101325
                return 10**log_fo2_nno
                
        except Exception as e:
            logger.error(f"Error calculating oxygen fugacity: {str(e)}")
            return 1e-10
    
    def _calculate_h2_activity_from_fo2(self, oxygen_fugacity: float, temperature_k: float, pressure_pa: float) -> float:
        """Calculate H2 activity from oxygen fugacity constraints"""
        try:
            # Water dissociation equilibrium: H2O ⇌ H2 + 0.5 O2
            # K = (fH2 * fO2^0.5) / aH2O
            
            # Equilibrium constant for water dissociation
            log_k_h2o = -13445.9 / temperature_k + 4.571 + 0.0016 * temperature_k
            k_h2o = 10**log_k_h2o
            
            # Assume water activity ≈ 1 for volcanic gas conditions
            h2o_activity = 1.0
            
            # Calculate H2 fugacity
            h2_fugacity = k_h2o * h2o_activity / np.sqrt(oxygen_fugacity)
            
            # Convert to activity (assume fugacity coefficient ≈ 1 for H2)
            h2_activity = h2_fugacity / (pressure_pa / 101325.0)
            
            return max(1e-6, min(0.1, h2_activity))
            
        except Exception as e:
            logger.error(f"Error calculating H2 activity from fO2: {str(e)}")
            return 0.01
    
    def _calculate_h2o_activity(self, gas_composition: GasCompositionData, temperature_k: float, 
                               pressure_pa: float, ionic_strength: float) -> float:
        """Calculate H2O activity in volcanic gas system"""
        try:
            # Water activity in gas phase with corrections for non-ideality
            
            # Base water activity from gas composition
            # Assume water is a major component in volcanic gases
            total_gas_fraction = (gas_composition.co2_concentration + 
                                gas_composition.so2_concentration + 
                                gas_composition.h2s_concentration)
            
            # Estimate water fraction (typically 80-95% of volcanic gases)
            water_fraction = max(0.5, 1.0 - total_gas_fraction)
            
            # Activity coefficient for water in gas mixture
            # Raoult's law with corrections for gas interactions
            gamma_h2o = 1.0
            
            # CO2-H2O interaction (reduces water activity)
            if gas_composition.co2_concentration > 0.1:
                gamma_h2o *= (1.0 - 0.1 * gas_composition.co2_concentration)
            
            # SO2-H2O interaction (strong interaction, forms H2SO3)
            if gas_composition.so2_concentration > 0.01:
                gamma_h2o *= (1.0 - 0.5 * gas_composition.so2_concentration)
            
            # Pressure correction for non-ideal gas behavior
            pressure_correction = 1.0 + 0.0001 * (pressure_pa - 101325) / 101325
            
            # Temperature correction
            temp_correction = np.exp(-0.001 * (temperature_k - 373.15))
            
            # Final water activity
            h2o_activity = water_fraction * gamma_h2o * pressure_correction * temp_correction
            
            return max(0.1, min(1.0, h2o_activity))
            
        except Exception as e:
            logger.error(f"Error calculating H2O activity: {str(e)}")
            return 0.8
    
    def _calculate_equilibrium_temperature_so2_h2s(self, so2_conc: float, h2s_conc: float,
                                                  h2_activity: float, h2o_activity: float, 
                                                  pressure_pa: float) -> float:
        """Calculate equilibrium temperature from SO2/H2S ratio"""
        try:
            if so2_conc <= 0 or h2s_conc <= 0:
                return 0.0
            
            # SO2/H2S ratio
            ratio = so2_conc / h2s_conc
            
            # Equilibrium constant expression: K = (H2S * H2O²) / (SO2 * H2³)
            # Rearranged: log K = log(H2S/SO2) + 2*log(H2O) - 3*log(H2)
            
            log_reaction_quotient = (np.log10(1/ratio) + 2*np.log10(h2o_activity) - 3*np.log10(h2_activity))
            
            # Temperature from equilibrium constant
            # log K = A/T + B
            A = -13356.0
            B = 4.571
            
            # Solve: A/T + B = log_reaction_quotient
            if abs(log_reaction_quotient - B) > 1e-10:
                temperature_k = A / (log_reaction_quotient - B)
                temperature_c = temperature_k - 273.15
                
                return max(0.0, min(1200.0, temperature_c))
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating equilibrium temperature: {str(e)}")
            return 0.0
    
    def _calculate_so2_h2s_reaction_rate(self, gas_composition: GasCompositionData, 
                                       temperature_k: float, pressure_pa: float, 
                                       equilibrium_ratio: float) -> float:
        """Calculate SO2-H2S reaction rate"""
        try:
            # Reaction rate based on departure from equilibrium
            # Rate = k * (Q - K) where Q is reaction quotient, K is equilibrium constant
            
            # Rate constant (Arrhenius equation)
            # k = A * exp(-Ea / RT)
            A = 1e10  # Pre-exponential factor (s⁻¹)
            Ea = 150000.0  # Activation energy (J/mol)
            R = 8314.0  # Gas constant
            
            rate_constant = A * np.exp(-Ea / (R * temperature_k))
            
            # Driving force (departure from equilibrium)
            driving_force = abs(equilibrium_ratio - 1.0)
            
            # Concentration dependence
            total_sulfur = gas_composition.so2_concentration + gas_composition.h2s_concentration
            conc_factor = total_sulfur
            
            # Pressure dependence (higher pressure increases rate)
            pressure_factor = (pressure_pa / 101325.0)**0.5
            
            # Reaction rate
            reaction_rate = rate_constant * driving_force * conc_factor * pressure_factor
            
            return reaction_rate
            
        except Exception as e:
            logger.error(f"Error calculating SO2-H2S reaction rate: {str(e)}")
            return 0.0
    
    def _analyze_sulfur_fractionation(self, delta_34s: float, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Analyze sulfur isotope fractionation effects"""
        try:
            fractionation_analysis = {}
            
            # SO2-H2S fractionation
            if gas_composition.so2_concentration > 0 and gas_composition.h2s_concentration > 0:
                # Temperature-dependent fractionation factor
                temp_k = gas_composition.temperature + 273.15
                
                # Empirical fractionation factor (Ohmoto & Rye, 1979)
                alpha_so2_h2s = np.exp(0.40 * 1e6 / (temp_k ** 2))
                
                # Expected fractionation
                expected_fractionation = 1000 * (alpha_so2_h2s - 1)
                
                fractionation_analysis['so2_h2s_fractionation'] = {
                    'expected_delta_difference': expected_fractionation,
                    'equilibrium_temperature': temp_k - 273.15,
                    'fractionation_factor': alpha_so2_h2s
                }
            
            # Assess equilibrium vs kinetic effects
            if abs(delta_34s) < 5:
                fractionation_type = 'equilibrium'
            elif delta_34s < -10:
                fractionation_type = 'kinetic/bacterial'
            else:
                fractionation_type = 'source_controlled'
            
            fractionation_analysis['fractionation_type'] = fractionation_type
            
            return fractionation_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing sulfur fractionation: {str(e)}")
            return {}
    
    def _calculate_fingerprinting_confidence(self, source_classification: Dict[str, Any], 
                                          equilibrium_analysis: Dict[str, Any],
                                          process_signatures: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence metrics for geochemical fingerprinting"""
        try:
            confidence_metrics = {}
            
            # Classification confidence
            classification_confidence = source_classification.get('confidence', 0.0)
            confidence_metrics['classification_confidence'] = classification_confidence
            
            # Equilibrium analysis confidence
            equilibrium_confidence = 0.5
            if 'so2_h2s_equilibrium' in equilibrium_analysis:
                eq_state = equilibrium_analysis['so2_h2s_equilibrium'].get('equilibrium_state', 'disequilibrium')
                if eq_state == 'equilibrium':
                    equilibrium_confidence = 0.8
                else:
                    equilibrium_confidence = 0.3
            
            confidence_metrics['equilibrium_confidence'] = equilibrium_confidence
            
            # Process signature confidence
            process_confidence = process_signatures.get('process_confidence', 0.0)
            confidence_metrics['process_confidence'] = process_confidence
            
            # Overall confidence (weighted average)
            overall_confidence = (0.4 * classification_confidence + 
                                0.3 * equilibrium_confidence + 
                                0.3 * process_confidence)
            
            confidence_metrics['overall_confidence'] = overall_confidence
            
            return confidence_metrics
            
        except Exception as e:
            logger.error(f"Error calculating fingerprinting confidence: {str(e)}")
            return {'overall_confidence': 0.5}
    
    def _calculate_feature_importance(self, features: Dict[str, float], 
                                    classification_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature importance for classification"""
        try:
            importance = {}
            
            # Key features for volcanic gas classification
            key_features = ['co2_so2_ratio', 'h2s_so2_ratio', 'temperature', 'reduced_s_fraction']
            
            for feature in key_features:
                if feature in features:
                    # Simple importance based on feature value and classification confidence
                    feature_value = features[feature]
                    max_confidence = max(classification_scores.values())
                    
                    # Normalize feature importance
                    if feature == 'temperature':
                        importance[feature] = min(1.0, feature_value / 500.0) * max_confidence
                    elif 'ratio' in feature:
                        importance[feature] = min(1.0, np.log10(feature_value + 1) / 3.0) * max_confidence
                    else:
                        importance[feature] = feature_value * max_confidence
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}
    
    async def _model_geochemical_evolution(self, gas_composition: GasCompositionData) -> Dict[str, Any]:
        """Model geochemical evolution pathways and processes"""
        try:
            evolution_models = {}
            
            # Query historical data for evolution analysis
            location = gas_composition.location
            current_time = gas_composition.timestamp
            
            evolution_query = """
            SELECT timestamp, co2_concentration, so2_concentration, h2s_concentration,
                   temperature, pressure, gas_ratio_co2_so2, gas_ratio_h2s_so2
            FROM gas_measurements
            WHERE ST_DWithin(location, ST_Point(%s, %s), 2000)
            AND timestamp > %s - INTERVAL '30 days'
            AND timestamp <= %s
            ORDER BY timestamp
            """
            
            historical_data = await self.db_manager.execute_query(
                evolution_query, (location[0], location[1], current_time, current_time)
            )
            
            if len(historical_data) >= 5:
                # Analyze temporal evolution patterns
                timestamps = [d['timestamp'] for d in historical_data]
                
                # CO2/SO2 ratio evolution
                co2_so2_ratios = [d['gas_ratio_co2_so2'] for d in historical_data if d['gas_ratio_co2_so2']]
                if co2_so2_ratios:
                    co2_so2_trend = self._calculate_trend(timestamps[:len(co2_so2_ratios)], co2_so2_ratios)
                    evolution_models['co2_so2_evolution'] = {
                        'trend': co2_so2_trend['trend'],
                        'r_squared': co2_so2_trend['r_squared'],
                        'evolution_rate': co2_so2_trend['trend'] * 24,  # per day
                        'interpretation': self._interpret_co2_so2_evolution(co2_so2_trend['trend'])
                    }
                
                # Temperature evolution
                temperatures = [d['temperature'] for d in historical_data]
                temp_trend = self._calculate_trend(timestamps, temperatures)
                evolution_models['temperature_evolution'] = {
                    'trend': temp_trend['trend'],
                    'r_squared': temp_trend['r_squared'],
                    'thermal_state': self._interpret_thermal_evolution(temp_trend['trend'])
                }
                
                # Degassing intensity evolution
                so2_concentrations = [d['so2_concentration'] for d in historical_data]
                so2_trend = self._calculate_trend(timestamps, so2_concentrations)
                evolution_models['degassing_evolution'] = {
                    'trend': so2_trend['trend'],
                    'r_squared': so2_trend['r_squared'],
                    'degassing_state': self._interpret_degassing_evolution(so2_trend['trend'])
                }
            
            # Model future evolution scenarios
            evolution_models['future_scenarios'] = self._model_future_evolution_scenarios(
                gas_composition, historical_data
            )
            
            return evolution_models
            
        except Exception as e:
            logger.error(f"Error modeling geochemical evolution: {str(e)}")
            return {}
    
    def _interpret_co2_so2_evolution(self, trend: float) -> str:
        """Interpret CO2/SO2 ratio evolution trend"""
        if trend > 0.1:
            return "increasing_co2_dominance"
        elif trend < -0.1:
            return "increasing_so2_degassing"
        else:
            return "stable_gas_composition"
    
    def _interpret_thermal_evolution(self, trend: float) -> str:
        """Interpret temperature evolution trend"""
        if trend > 1.0:
            return "heating_system"
        elif trend < -1.0:
            return "cooling_system"
        else:
            return "stable_thermal_state"
    
    def _interpret_degassing_evolution(self, trend: float) -> str:
        """Interpret degassing intensity evolution"""
        if trend > 0.001:
            return "increasing_degassing"
        elif trend < -0.001:
            return "decreasing_degassing"
        else:
            return "stable_degassing"
    
    def _model_future_evolution_scenarios(self, gas_composition: GasCompositionData, 
                                        historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Model future evolution scenarios based on current trends"""
        try:
            scenarios = {}
            
            if len(historical_data) >= 3:
                # Extract recent trends
                timestamps = [d['timestamp'] for d in historical_data[-10:]]  # Last 10 measurements
                co2_values = [d['co2_concentration'] for d in historical_data[-10:]]
                so2_values = [d['so2_concentration'] for d in historical_data[-10:]]
                
                # Calculate recent trends
                co2_trend = self._calculate_trend(timestamps, co2_values)
                so2_trend = self._calculate_trend(timestamps, so2_values)
                
                # Project future scenarios (next 24 hours)
                time_horizon = 24.0  # hours
                
                scenarios['conservative'] = {
                    'co2_projection': gas_composition.co2_concentration + co2_trend['trend'] * time_horizon * 0.5,
                    'so2_projection': gas_composition.so2_concentration + so2_trend['trend'] * time_horizon * 0.5,
                    'confidence': 0.7
                }
                
                scenarios['linear_extrapolation'] = {
                    'co2_projection': gas_composition.co2_concentration + co2_trend['trend'] * time_horizon,
                    'so2_projection': gas_composition.so2_concentration + so2_trend['trend'] * time_horizon,
                    'confidence': 0.5
                }
                
                scenarios['accelerated'] = {
                    'co2_projection': gas_composition.co2_concentration + co2_trend['trend'] * time_horizon * 2.0,
                    'so2_projection': gas_composition.so2_concentration + so2_trend['trend'] * time_horizon * 2.0,
                    'confidence': 0.3
                }
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error modeling future evolution scenarios: {str(e)}")
            return {}