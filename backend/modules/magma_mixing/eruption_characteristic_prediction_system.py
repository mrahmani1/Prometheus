"""
Advanced Eruption Characteristic Prediction System for Magma Mixing Analysis

This module implements comprehensive eruption characteristic prediction based on magma
mixing processes, including eruption intensity prediction, explosivity forecasting,
discharge rate prediction, eruption style prediction, temporal evolution modeling,
hazard assessment, uncertainty quantification, and validation against historical data.

Requirements: 4.4, 6.1, 6.2, 6.3
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d, griddata
from scipy.integrate import odeint, solve_ivp
from scipy.stats import norm, lognorm, weibull_min, gamma
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import asyncpg
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Import existing data models
from .data_models import MagmaComposition, ThermalProperties, RheologicalProperties

logger = logging.getLogger(__name__)

@dataclass
class EruptionIntensityPrediction:
    """Eruption intensity prediction based on mixing degree"""
    mixing_degree: float
    predicted_intensity: float  # Volcanic Explosivity Index (VEI) equivalent
    intensity_confidence: float
    intensity_bounds: Tuple[float, float]
    contributing_factors: Dict[str, float]
    prediction_timestamp: datetime
    
@dataclass
class ExplosivityForecast:
    """Explosivity forecasting from volatile content evolution"""
    volatile_content: Dict[str, float]
    predicted_explosivity: float  # 0-1 scale
    explosivity_index: int  # VEI scale 0-8
    gas_pressure_buildup: float
    fragmentation_threshold: float
    explosivity_confidence: float
    forecast_horizon: timedelta
    
@dataclass
class DischargeRatePrediction:
    """Magma discharge rate prediction from rheological changes"""
    viscosity: float
    yield_strength: float
    conduit_geometry: Dict[str, float]
    predicted_discharge_rate: float  # m3/s
    peak_discharge_rate: float
    sustained_discharge_rate: float
    discharge_duration: float
    rheological_controls: Dict[str, float]
    
@dataclass
class EruptionStylePrediction:
    """Eruption style prediction based on mixed magma properties"""
    magma_properties: Dict[str, float]
    predicted_style: str  # 'effusive', 'explosive', 'mixed', 'phreatomagmatic'
    style_probability: Dict[str, float]
    controlling_parameters: Dict[str, float]
    transition_thresholds: Dict[str, float]
    style_confidence: float@datacla
ss
class TemporalEruptionEvolution:
    """Temporal eruption evolution based on ongoing mixing"""
    evolution_timeline: List[Tuple[datetime, Dict[str, float]]]
    mixing_progression: List[float]
    intensity_evolution: List[float]
    style_transitions: List[Tuple[datetime, str, str]]  # time, from_style, to_style
    peak_activity_time: datetime
    eruption_duration: timedelta
    evolution_uncertainty: Dict[str, List[float]]
    
@dataclass
class MixingTriggeredHazardAssessment:
    """Hazard assessment for mixing-triggered eruptions"""
    hazard_zones: Dict[str, Dict[str, Any]]  # hazard_type -> zone_data
    population_exposure: Dict[str, int]
    infrastructure_risk: Dict[str, float]
    evacuation_recommendations: Dict[str, Any]
    hazard_probabilities: Dict[str, float]
    impact_scenarios: List[Dict[str, Any]]
    risk_mitigation_strategies: List[str]
    
@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification for eruption predictions"""
    aleatory_uncertainty: Dict[str, float]  # Natural variability
    epistemic_uncertainty: Dict[str, float]  # Knowledge uncertainty
    model_uncertainty: Dict[str, float]     # Model structure uncertainty
    parameter_uncertainty: Dict[str, float] # Parameter estimation uncertainty
    total_uncertainty: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    sensitivity_analysis: Dict[str, float]
    
@dataclass
class HistoricalValidationResult:
    """Validation results against historical mixing-related eruptions"""
    eruption_name: str
    eruption_date: datetime
    predicted_characteristics: Dict[str, float]
    observed_characteristics: Dict[str, float]
    prediction_accuracy: Dict[str, float]
    residuals: Dict[str, float]
    validation_metrics: Dict[str, float]
    lessons_learned: List[str]

@dataclass
class EruptionCharacteristicPredictionResult:
    """Complete eruption characteristic prediction result"""
    prediction_id: str
    chamber_id: str
    prediction_timestamp: datetime
    mixing_degree: float
    intensity_prediction: EruptionIntensityPrediction
    explosivity_forecast: ExplosivityForecast
    discharge_rate_prediction: DischargeRatePrediction
    style_prediction: EruptionStylePrediction
    temporal_evolution: TemporalEruptionEvolution
    hazard_assessment: MixingTriggeredHazardAssessment
    uncertainty_quantification: UncertaintyQuantification
    validation_score: float
    confidence_level: float


class EruptionCharacteristicPredictionSystem:
    """
    Advanced system for predicting eruption characteristics based on magma mixing processes.
    
    This system provides comprehensive modeling of:
    - Eruption intensity prediction based on mixing degree
    - Explosivity forecasting from volatile content evolution
    - Magma discharge rate prediction from rheological changes
    - Eruption style prediction based on mixed magma properties
    - Temporal eruption evolution based on ongoing mixing
    - Hazard assessment for mixing-triggered eruptions
    - Uncertainty quantification for eruption predictions
    - Validation against historical mixing-related eruptions
    """
    
    def __init__(self, database_config: Dict[str, Any]):
        """Initialize the eruption characteristic prediction system"""
        self.database_config = database_config
        self.db_pool = None
        self.executor = ThreadPoolExecutor(max_workers=12)
        
        # Initialize prediction models
        self.intensity_predictor = None
        self.explosivity_predictor = None
        self.discharge_rate_predictor = None
        self.style_predictor = None
        self.temporal_evolution_predictor = None
        
        # Initialize empirical relationships and databases
        self.empirical_relationships = self._load_empirical_relationships()
        self.historical_eruption_database = self._load_historical_eruption_database()
        self.hazard_models = self._load_hazard_models()
        self.uncertainty_models = self._load_uncertainty_models()
        
        # Initialize scalers for data preprocessing
        self.feature_scalers = {}
        self.target_scalers = {}
        
        logger.info("EruptionCharacteristicPredictionSystem initialized successfully")
    
    async def initialize_database_connection(self):
        """Initialize database connection pool"""
        try:
            self.db_pool = await asyncpg.create_pool(
                host=self.database_config['host'],
                port=self.database_config['port'],
                user=self.database_config['user'],
                password=self.database_config['password'],
                database=self.database_config['database'],
                min_size=5,
                max_size=20
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise    def
 _load_empirical_relationships(self) -> Dict[str, Any]:
        """Load empirical relationships for eruption characteristic prediction"""
        relationships = {
            'intensity_mixing_relationship': {
                # Relationship between mixing degree and eruption intensity
                'coefficients': [2.1, 1.8, -0.3, 0.05],  # Polynomial coefficients
                'mixing_threshold': 0.3,  # Minimum mixing for intensity increase
                'saturation_point': 0.85,  # Mixing degree where intensity saturates
                'volatility_factor': 1.2,  # Amplification factor for volatile-rich systems
                'temperature_correction': 0.001,  # Temperature dependence (per K)
                'pressure_correction': 0.0001   # Pressure dependence (per Pa)
            },
            'explosivity_volatile_relationship': {
                # Relationship between volatile content and explosivity
                'h2o_coefficient': 0.8,      # H2O contribution to explosivity
                'co2_coefficient': 0.3,      # CO2 contribution to explosivity
                'so2_coefficient': 0.5,      # SO2 contribution to explosivity
                'h2s_coefficient': 0.4,      # H2S contribution to explosivity
                'cl_coefficient': 0.2,       # Cl contribution to explosivity
                'f_coefficient': 0.15,       # F contribution to explosivity
                'synergistic_factor': 1.5,   # Multi-volatile synergy
                'pressure_threshold': 1e8,   # Critical pressure for explosive transition (Pa)
                'fragmentation_criterion': 0.75  # Critical gas volume fraction
            },
            'discharge_rate_rheology_relationship': {
                # Relationship between rheology and discharge rate
                'viscosity_exponent': -0.7,     # Viscosity dependence
                'yield_strength_factor': -0.5,  # Yield strength effect
                'conduit_diameter_exponent': 2.0, # Conduit geometry effect
                'pressure_gradient_factor': 1.0,  # Driving pressure effect
                'temperature_correction': 0.002,  # Temperature effect on flow
                'crystallinity_penalty': -1.2,   # Crystallinity effect on flow
                'gas_content_enhancement': 0.8   # Gas content effect on flow
            },
            'style_transition_thresholds': {
                # Thresholds for eruption style transitions
                'effusive_explosive_threshold': {
                    'gas_content': 0.03,        # Critical gas content (volume fraction)
                    'viscosity': 1e6,           # Critical viscosity (Pa·s)
                    'ascent_rate': 0.1,         # Critical ascent rate (m/s)
                    'fragmentation_pressure': 5e7  # Critical fragmentation pressure (Pa)
                },
                'explosive_phreatomagmatic_threshold': {
                    'water_interaction': 0.1,   # Critical water/magma mass ratio
                    'depth_threshold': 100,     # Critical depth for water interaction (m)
                    'cooling_rate': 1000,       # Critical cooling rate (K/s)
                    'steam_pressure': 1e7       # Critical steam pressure (Pa)
                },
                'mixed_style_conditions': {
                    'temporal_variability': 0.2, # Variability in eruption parameters
                    'conduit_complexity': 0.5,   # Conduit geometry complexity factor
                    'magma_supply_variability': 0.3  # Variability in magma supply
                }
            }
        }
        return relationships
    
    def _load_historical_eruption_database(self) -> Dict[str, Any]:
        """Load historical eruption database for validation"""
        historical_db = {
            'mount_vesuvius_1944': {
                'eruption_date': datetime(1944, 3, 18),
                'location': {'lat': 40.821, 'lon': 14.426},
                'mixing_evidence': 'disequilibrium_textures',
                'mixing_degree': 0.65,
                'observed_characteristics': {
                    'intensity': 3.2,  # VEI equivalent
                    'explosivity': 0.7,
                    'discharge_rate': 150.0,  # m3/s
                    'style': 'explosive',
                    'duration': 86400 * 12,  # seconds (12 days)
                    'peak_activity': 86400 * 3  # seconds (3 days)
                },
                'magma_properties': {
                    'viscosity': 1e5,
                    'gas_content': 0.045,
                    'temperature': 1373,
                    'crystallinity': 0.25,
                    'volatile_h2o': 0.035,
                    'volatile_co2': 0.008,
                    'volatile_so2': 0.002
                },
                'hazard_impacts': {
                    'fatalities': 26,
                    'evacuees': 12000,
                    'economic_loss': 5e6,  # USD (1944 values)
                    'affected_area': 150   # km2
                }
            },
            'santorini_minoan': {
                'eruption_date': datetime(-1613, 1, 1),  # 1613 BCE
                'location': {'lat': 36.404, 'lon': 25.396},
                'mixing_evidence': 'banded_pumice',
                'mixing_degree': 0.45,
                'observed_characteristics': {
                    'intensity': 6.8,  # VEI equivalent
                    'explosivity': 0.95,
                    'discharge_rate': 1e8,  # m3/s
                    'style': 'explosive',
                    'duration': 86400 * 30,  # seconds (30 days)
                    'peak_activity': 86400 * 5  # seconds (5 days)
                },
                'magma_properties': {
                    'viscosity': 1e7,
                    'gas_content': 0.08,
                    'temperature': 1123,
                    'crystallinity': 0.15,
                    'volatile_h2o': 0.06,
                    'volatile_co2': 0.015,
                    'volatile_so2': 0.005
                },
                'hazard_impacts': {
                    'fatalities': 'unknown',
                    'evacuees': 'unknown',
                    'economic_loss': 'catastrophic',
                    'affected_area': 50000   # km2
                }
            },
            'pinatubo_1991': {
                'eruption_date': datetime(1991, 6, 15),
                'location': {'lat': 15.13, 'lon': 120.35},
                'mixing_evidence': 'compositional_zoning',
                'mixing_degree': 0.55,
                'observed_characteristics': {
                    'intensity': 6.0,  # VEI
                    'explosivity': 0.9,
                    'discharge_rate': 5e7,  # m3/s
                    'style': 'explosive',
                    'duration': 86400 * 3,  # seconds (3 days)
                    'peak_activity': 86400 * 1  # seconds (1 day)
                },
                'magma_properties': {
                    'viscosity': 5e6,
                    'gas_content': 0.065,
                    'temperature': 1123,
                    'crystallinity': 0.3,
                    'volatile_h2o': 0.055,
                    'volatile_co2': 0.012,
                    'volatile_so2': 0.008
                },
                'hazard_impacts': {
                    'fatalities': 847,
                    'evacuees': 200000,
                    'economic_loss': 3e8,  # USD
                    'affected_area': 2000   # km2
                }
            },
            'krakatoa_1883': {
                'eruption_date': datetime(1883, 8, 27),
                'location': {'lat': -6.102, 'lon': 105.423},
                'mixing_evidence': 'magma_mingling',
                'mixing_degree': 0.75,
                'observed_characteristics': {
                    'intensity': 6.0,  # VEI
                    'explosivity': 0.98,
                    'discharge_rate': 1e8,  # m3/s
                    'style': 'explosive',
                    'duration': 86400 * 1,  # seconds (1 day)
                    'peak_activity': 3600 * 4  # seconds (4 hours)
                },
                'magma_properties': {
                    'viscosity': 2e6,
                    'gas_content': 0.09,
                    'temperature': 1273,
                    'crystallinity': 0.2,
                    'volatile_h2o': 0.07,
                    'volatile_co2': 0.018,
                    'volatile_so2': 0.012
                },
                'hazard_impacts': {
                    'fatalities': 36000,
                    'evacuees': 'unknown',
                    'economic_loss': 'catastrophic',
                    'affected_area': 300000   # km2 (including tsunami)
                }
            }
        }
        return historical_db   
 def _load_hazard_models(self) -> Dict[str, Any]:
        """Load hazard assessment models"""
        hazard_models = {
            'lava_flow_model': {
                'model_type': 'cellular_automata',
                'parameters': {
                    'flow_rate_factor': 0.8,
                    'cooling_rate': 0.001,  # K/s
                    'viscosity_threshold': 1e8,  # Pa·s
                    'slope_factor': 0.1,
                    'channel_factor': 1.5
                },
                'hazard_zones': {
                    'high_risk': {'distance': 5000, 'probability': 0.8},
                    'medium_risk': {'distance': 10000, 'probability': 0.5},
                    'low_risk': {'distance': 20000, 'probability': 0.2}
                }
            },
            'pyroclastic_flow_model': {
                'model_type': 'energy_cone',
                'parameters': {
                    'collapse_height': 1000,  # m
                    'mobility_coefficient': 0.2,
                    'energy_dissipation': 0.05,
                    'entrainment_factor': 0.1,
                    'temperature_decay': 0.001  # K/m
                },
                'hazard_zones': {
                    'high_risk': {'distance': 15000, 'probability': 0.9},
                    'medium_risk': {'distance': 25000, 'probability': 0.6},
                    'low_risk': {'distance': 40000, 'probability': 0.3}
                }
            },
            'ash_fall_model': {
                'model_type': 'advection_diffusion',
                'parameters': {
                    'column_height_factor': 1.2,
                    'wind_speed_factor': 0.8,
                    'particle_size_distribution': [0.1, 0.3, 0.4, 0.2],  # phi classes
                    'settling_velocity_factor': 1.0,
                    'atmospheric_diffusion': 100  # m2/s
                },
                'hazard_zones': {
                    'high_risk': {'thickness': 0.1, 'probability': 0.7},  # m
                    'medium_risk': {'thickness': 0.01, 'probability': 0.5},
                    'low_risk': {'thickness': 0.001, 'probability': 0.3}
                }
            },
            'gas_hazard_model': {
                'model_type': 'gaussian_plume',
                'parameters': {
                    'emission_rate_factor': 1.0,
                    'atmospheric_stability': 'neutral',
                    'mixing_height': 1000,  # m
                    'deposition_velocity': 0.01,  # m/s
                    'chemical_reaction_rate': 0.001  # s-1
                },
                'hazard_zones': {
                    'high_risk': {'concentration': 100, 'probability': 0.6},  # ppm
                    'medium_risk': {'concentration': 10, 'probability': 0.4},
                    'low_risk': {'concentration': 1, 'probability': 0.2}
                }
            },
            'lahar_model': {
                'model_type': 'flow_routing',
                'parameters': {
                    'water_magma_ratio': 0.5,
                    'debris_entrainment': 0.3,
                    'flow_velocity_factor': 0.8,
                    'channel_roughness': 0.05,
                    'bulking_factor': 2.0
                },
                'hazard_zones': {
                    'high_risk': {'distance': 30000, 'probability': 0.7},
                    'medium_risk': {'distance': 50000, 'probability': 0.4},
                    'low_risk': {'distance': 80000, 'probability': 0.2}
                }
            }
        }
        return hazard_models
    
    def _load_uncertainty_models(self) -> Dict[str, Any]:
        """Load uncertainty quantification models"""
        uncertainty_models = {
            'aleatory_uncertainty': {
                # Natural variability in volcanic processes
                'mixing_degree_variability': 0.15,
                'volatile_content_variability': 0.25,
                'temperature_variability': 50,  # K
                'pressure_variability': 1e7,   # Pa
                'viscosity_variability': 0.5,  # log10 units
                'discharge_rate_variability': 0.3
            },
            'epistemic_uncertainty': {
                # Knowledge uncertainty
                'model_parameter_uncertainty': 0.2,
                'empirical_relationship_uncertainty': 0.3,
                'database_completeness_uncertainty': 0.25,
                'measurement_uncertainty': 0.1,
                'temporal_extrapolation_uncertainty': 0.4
            },
            'model_uncertainty': {
                # Model structure uncertainty
                'intensity_model_uncertainty': 0.3,
                'explosivity_model_uncertainty': 0.25,
                'discharge_rate_model_uncertainty': 0.4,
                'style_prediction_uncertainty': 0.35,
                'temporal_evolution_uncertainty': 0.5
            },
            'correlation_structure': {
                # Correlations between uncertain parameters
                'intensity_explosivity_correlation': 0.7,
                'discharge_rate_intensity_correlation': 0.6,
                'volatile_explosivity_correlation': 0.8,
                'mixing_intensity_correlation': 0.5,
                'temperature_viscosity_correlation': -0.9
            }
        }
        return uncertainty_models
    
    async def predict_eruption_intensity(
        self, 
        mixing_degree: float,
        magma_properties: Dict[str, float],
        chamber_conditions: Dict[str, float]
    ) -> EruptionIntensityPrediction:
        """
        Predict eruption intensity based on mixing degree and magma properties
        
        Args:
            mixing_degree: Degree of magma mixing (0-1)
            magma_properties: Dictionary of magma properties
            chamber_conditions: Dictionary of chamber conditions
            
        Returns:
            EruptionIntensityPrediction object
        """
        try:
            # Extract key parameters
            volatile_content = magma_properties.get('volatile_content', 0.02)
            temperature = magma_properties.get('temperature', 1200)
            viscosity = magma_properties.get('viscosity', 1e5)
            crystallinity = magma_properties.get('crystallinity', 0.2)
            pressure = chamber_conditions.get('pressure', 1e8)
            
            # Apply empirical relationship for intensity prediction
            rel = self.empirical_relationships['intensity_mixing_relationship']
            
            # Base intensity from mixing degree (polynomial relationship)
            coeffs = rel['coefficients']
            base_intensity = (coeffs[0] * mixing_degree**3 + 
                            coeffs[1] * mixing_degree**2 + 
                            coeffs[2] * mixing_degree + 
                            coeffs[3])
            
            # Apply threshold and saturation effects
            if mixing_degree < rel['mixing_threshold']:
                base_intensity *= (mixing_degree / rel['mixing_threshold'])**2
            elif mixing_degree > rel['saturation_point']:
                saturation_factor = 1 + 0.1 * np.log(mixing_degree / rel['saturation_point'])
                base_intensity *= saturation_factor
            
            # Apply volatile content amplification
            volatile_factor = 1 + rel['volatility_factor'] * volatile_content
            
            # Apply temperature correction
            temp_correction = 1 + rel['temperature_correction'] * (temperature - 1200)
            
            # Apply pressure correction
            pressure_correction = 1 + rel['pressure_correction'] * (pressure - 1e8)
            
            # Calculate final intensity
            predicted_intensity = base_intensity * volatile_factor * temp_correction * pressure_correction
            
            # Ensure intensity is within reasonable bounds (0-8 VEI scale)
            predicted_intensity = np.clip(predicted_intensity, 0, 8)
            
            # Calculate contributing factors
            contributing_factors = {
                'mixing_degree_contribution': base_intensity / predicted_intensity,
                'volatile_contribution': (volatile_factor - 1) / (predicted_intensity / base_intensity - 1) if predicted_intensity != base_intensity else 0,
                'temperature_contribution': (temp_correction - 1) / (predicted_intensity / base_intensity - 1) if predicted_intensity != base_intensity else 0,
                'pressure_contribution': (pressure_correction - 1) / (predicted_intensity / base_intensity - 1) if predicted_intensity != base_intensity else 0,
                'crystallinity_penalty': -0.5 * crystallinity,
                'viscosity_effect': -0.1 * np.log10(viscosity / 1e5)
            }
            
            # Calculate uncertainty bounds
            uncertainty = self.uncertainty_models['model_uncertainty']['intensity_model_uncertainty']
            intensity_bounds = (
                max(0, predicted_intensity * (1 - uncertainty)),
                min(8, predicted_intensity * (1 + uncertainty))
            )
            
            # Calculate confidence based on data quality and model performance
            confidence = self._calculate_prediction_confidence(
                mixing_degree, magma_properties, 'intensity'
            )
            
            return EruptionIntensityPrediction(
                mixing_degree=mixing_degree,
                predicted_intensity=predicted_intensity,
                intensity_confidence=confidence,
                intensity_bounds=intensity_bounds,
                contributing_factors=contributing_factors,
                prediction_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in eruption intensity prediction: {e}")
            raise 
   async def forecast_explosivity(
        self,
        volatile_evolution: Dict[str, List[float]],
        pressure_evolution: List[float],
        temperature_evolution: List[float],
        time_steps: List[datetime]
    ) -> ExplosivityForecast:
        """
        Forecast explosivity from volatile content evolution
        
        Args:
            volatile_evolution: Dictionary of volatile species evolution over time
            pressure_evolution: Pressure evolution over time
            temperature_evolution: Temperature evolution over time
            time_steps: Time steps for evolution data
            
        Returns:
            ExplosivityForecast object
        """
        try:
            # Get current volatile content (latest values)
            current_volatiles = {species: values[-1] for species, values in volatile_evolution.items()}
            current_pressure = pressure_evolution[-1]
            current_temperature = temperature_evolution[-1]
            
            # Calculate explosivity based on volatile content
            rel = self.empirical_relationships['explosivity_volatile_relationship']
            
            # Individual volatile contributions
            h2o_contribution = rel['h2o_coefficient'] * current_volatiles.get('H2O', 0)
            co2_contribution = rel['co2_coefficient'] * current_volatiles.get('CO2', 0)
            so2_contribution = rel['so2_coefficient'] * current_volatiles.get('SO2', 0)
            h2s_contribution = rel['h2s_coefficient'] * current_volatiles.get('H2S', 0)
            cl_contribution = rel['cl_coefficient'] * current_volatiles.get('Cl', 0)
            f_contribution = rel['f_coefficient'] * current_volatiles.get('F', 0)
            
            # Base explosivity from individual contributions
            base_explosivity = (h2o_contribution + co2_contribution + so2_contribution + 
                              h2s_contribution + cl_contribution + f_contribution)
            
            # Apply synergistic effects for multi-volatile systems
            volatile_count = sum(1 for v in current_volatiles.values() if v > 0.001)
            if volatile_count > 1:
                synergy_factor = 1 + (volatile_count - 1) * 0.1 * rel['synergistic_factor']
                base_explosivity *= synergy_factor
            
            # Apply pressure effects
            if current_pressure > rel['pressure_threshold']:
                pressure_factor = 1 + 0.5 * np.log(current_pressure / rel['pressure_threshold'])
                base_explosivity *= pressure_factor
            
            # Calculate gas pressure buildup
            total_gas_content = sum(current_volatiles.values())
            gas_pressure_buildup = total_gas_content * current_pressure * 0.1  # Simplified calculation
            
            # Check fragmentation threshold
            fragmentation_threshold = rel['fragmentation_criterion']
            gas_volume_fraction = self._calculate_gas_volume_fraction(
                current_volatiles, current_pressure, current_temperature
            )
            
            # Apply fragmentation effects
            if gas_volume_fraction > fragmentation_threshold:
                fragmentation_factor = 1 + 2 * (gas_volume_fraction - fragmentation_threshold)
                base_explosivity *= fragmentation_factor
            
            # Normalize explosivity to 0-1 scale
            predicted_explosivity = np.tanh(base_explosivity)  # Sigmoid-like function
            
            # Convert to VEI scale (0-8)
            explosivity_index = int(np.round(predicted_explosivity * 8))
            
            # Calculate forecast horizon based on volatile evolution rate
            evolution_rates = []
            for species, values in volatile_evolution.items():
                if len(values) > 1:
                    rate = abs(values[-1] - values[-2]) / (time_steps[-1] - time_steps[-2]).total_seconds()
                    evolution_rates.append(rate)
            
            if evolution_rates:
                avg_evolution_rate = np.mean(evolution_rates)
                forecast_horizon = timedelta(hours=max(1, min(168, 24 / (avg_evolution_rate * 1000))))  # 1 hour to 1 week
            else:
                forecast_horizon = timedelta(hours=24)  # Default 24 hours
            
            # Calculate confidence
            confidence = self._calculate_prediction_confidence(
                predicted_explosivity, current_volatiles, 'explosivity'
            )
            
            return ExplosivityForecast(
                volatile_content=current_volatiles,
                predicted_explosivity=predicted_explosivity,
                explosivity_index=explosivity_index,
                gas_pressure_buildup=gas_pressure_buildup,
                fragmentation_threshold=fragmentation_threshold,
                explosivity_confidence=confidence,
                forecast_horizon=forecast_horizon
            )
            
        except Exception as e:
            logger.error(f"Error in explosivity forecasting: {e}")
            raise
    
    def _calculate_gas_volume_fraction(
        self, 
        volatile_content: Dict[str, float], 
        pressure: float, 
        temperature: float
    ) -> float:
        """Calculate gas volume fraction from volatile content and conditions"""
        try:
            # Simplified gas volume fraction calculation
            # In reality, this would use complex equation of state calculations
            
            total_moles_gas = 0
            for species, content in volatile_content.items():
                if species in ['H2O', 'CO2', 'SO2', 'H2S']:
                    # Convert mass fraction to moles (simplified)
                    molecular_weights = {'H2O': 18.015, 'CO2': 44.01, 'SO2': 64.066, 'H2S': 34.08}
                    mw = molecular_weights.get(species, 30.0)  # Default molecular weight
                    total_moles_gas += content / mw
            
            # Ideal gas law approximation (PV = nRT)
            R = 8.314  # J/mol/K
            if total_moles_gas > 0 and pressure > 0:
                gas_volume_fraction = (total_moles_gas * R * temperature) / (pressure * 1.0)  # Assuming 1 m3 total volume
                return min(gas_volume_fraction, 0.99)  # Cap at 99%
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating gas volume fraction: {e}")
            return 0.0    async
 def predict_discharge_rate(
        self,
        rheological_properties: RheologicalProperties,
        conduit_geometry: Dict[str, float],
        pressure_gradient: float,
        gas_content: float
    ) -> DischargeRatePrediction:
        """
        Predict magma discharge rate from rheological changes
        
        Args:
            rheological_properties: Rheological properties of magma
            conduit_geometry: Conduit geometry parameters
            pressure_gradient: Pressure gradient driving flow
            gas_content: Gas content in magma
            
        Returns:
            DischargeRatePrediction object
        """
        try:
            # Extract rheological parameters
            viscosity = rheological_properties.viscosity
            yield_strength = rheological_properties.yield_strength
            
            # Extract conduit parameters
            conduit_diameter = conduit_geometry.get('diameter', 100)  # m
            conduit_length = conduit_geometry.get('length', 5000)    # m
            conduit_roughness = conduit_geometry.get('roughness', 0.01)  # dimensionless
            
            # Apply empirical relationship for discharge rate
            rel = self.empirical_relationships['discharge_rate_rheology_relationship']
            
            # Base discharge rate calculation using modified Poiseuille flow
            # Q = (π * r^4 * ΔP) / (8 * η * L) for Newtonian flow
            # Modified for non-Newtonian behavior and gas effects
            
            radius = conduit_diameter / 2
            area = np.pi * radius**2
            
            # Viscosity correction for non-Newtonian behavior
            effective_viscosity = viscosity * (1 + yield_strength / (pressure_gradient * radius))
            
            # Base discharge rate
            base_discharge_rate = (np.pi * radius**4 * pressure_gradient) / (8 * effective_viscosity * conduit_length)
            
            # Apply empirical corrections
            viscosity_correction = (viscosity / 1e5) ** rel['viscosity_exponent']
            yield_strength_correction = (1 + yield_strength / 1e4) ** rel['yield_strength_factor']
            diameter_correction = (conduit_diameter / 100) ** rel['conduit_diameter_exponent']
            pressure_correction = (pressure_gradient / 1e4) ** rel['pressure_gradient_factor']
            
            # Gas content enhancement (gas reduces effective density and viscosity)
            gas_enhancement = 1 + rel['gas_content_enhancement'] * gas_content
            
            # Roughness penalty
            roughness_penalty = 1 - 0.2 * conduit_roughness
            
            # Calculate predicted discharge rate
            predicted_discharge_rate = (base_discharge_rate * 
                                      viscosity_correction * 
                                      yield_strength_correction * 
                                      diameter_correction * 
                                      pressure_correction * 
                                      gas_enhancement * 
                                      roughness_penalty)
            
            # Ensure positive discharge rate
            predicted_discharge_rate = max(0, predicted_discharge_rate)
            
            # Calculate peak and sustained discharge rates
            # Peak rate is typically 2-5 times higher than sustained rate
            peak_discharge_rate = predicted_discharge_rate * (2 + 3 * gas_content)
            sustained_discharge_rate = predicted_discharge_rate * 0.6
            
            # Estimate discharge duration based on chamber volume and discharge rate
            # This would typically require chamber volume information
            estimated_chamber_volume = conduit_geometry.get('chamber_volume', 1e8)  # m3
            discharge_duration = estimated_chamber_volume / predicted_discharge_rate if predicted_discharge_rate > 0 else 0
            
            # Calculate rheological controls
            rheological_controls = {
                'viscosity_control': abs(rel['viscosity_exponent']) * (np.log10(viscosity) - 5),
                'yield_strength_control': abs(rel['yield_strength_factor']) * (yield_strength / 1e4),
                'gas_content_control': rel['gas_content_enhancement'] * gas_content,
                'conduit_geometry_control': rel['conduit_diameter_exponent'] * np.log(conduit_diameter / 100),
                'pressure_control': rel['pressure_gradient_factor'] * np.log(pressure_gradient / 1e4)
            }
            
            return DischargeRatePrediction(
                viscosity=viscosity,
                yield_strength=yield_strength,
                conduit_geometry=conduit_geometry,
                predicted_discharge_rate=predicted_discharge_rate,
                peak_discharge_rate=peak_discharge_rate,
                sustained_discharge_rate=sustained_discharge_rate,
                discharge_duration=discharge_duration,
                rheological_controls=rheological_controls
            )
            
        except Exception as e:
            logger.error(f"Error in discharge rate prediction: {e}")
            raise
    
    async def predict_eruption_style(
        self,
        mixed_magma_properties: Dict[str, float],
        environmental_conditions: Dict[str, float]
    ) -> EruptionStylePrediction:
        """
        Predict eruption style based on mixed magma properties
        
        Args:
            mixed_magma_properties: Properties of mixed magma
            environmental_conditions: Environmental conditions
            
        Returns:
            EruptionStylePrediction object
        """
        try:
            # Extract key parameters
            gas_content = mixed_magma_properties.get('gas_content', 0.02)
            viscosity = mixed_magma_properties.get('viscosity', 1e5)
            temperature = mixed_magma_properties.get('temperature', 1200)
            ascent_rate = mixed_magma_properties.get('ascent_rate', 0.01)  # m/s
            crystallinity = mixed_magma_properties.get('crystallinity', 0.2)
            
            water_interaction = environmental_conditions.get('water_interaction', 0.0)
            depth_to_water = environmental_conditions.get('depth_to_water', 1000)  # m
            external_water = environmental_conditions.get('external_water', False)
            
            # Get style transition thresholds
            thresholds = self.empirical_relationships['style_transition_thresholds']
            
            # Calculate probabilities for different eruption styles
            style_probabilities = {}
            
            # Effusive probability
            effusive_prob = 1.0
            if gas_content > thresholds['effusive_explosive_threshold']['gas_content']:
                effusive_prob *= 0.1
            if viscosity > thresholds['effusive_explosive_threshold']['viscosity']:
                effusive_prob *= 0.2
            if ascent_rate > thresholds['effusive_explosive_threshold']['ascent_rate']:
                effusive_prob *= 0.3
            
            # Explosive probability
            explosive_prob = 0.0
            if gas_content > thresholds['effusive_explosive_threshold']['gas_content']:
                explosive_prob += 0.8 * (gas_content - thresholds['effusive_explosive_threshold']['gas_content']) / 0.05
            if viscosity > thresholds['effusive_explosive_threshold']['viscosity']:
                explosive_prob += 0.6 * np.log10(viscosity / thresholds['effusive_explosive_threshold']['viscosity'])
            if ascent_rate > thresholds['effusive_explosive_threshold']['ascent_rate']:
                explosive_prob += 0.4 * (ascent_rate - thresholds['effusive_explosive_threshold']['ascent_rate']) / 1.0
            
            explosive_prob = min(1.0, explosive_prob)
            
            # Phreatomagmatic probability
            phreatomagmatic_prob = 0.0
            if water_interaction > thresholds['explosive_phreatomagmatic_threshold']['water_interaction']:
                phreatomagmatic_prob += 0.9 * water_interaction
            if depth_to_water < thresholds['explosive_phreatomagmatic_threshold']['depth_threshold']:
                phreatomagmatic_prob += 0.5 * (1 - depth_to_water / thresholds['explosive_phreatomagmatic_threshold']['depth_threshold'])
            if external_water:
                phreatomagmatic_prob += 0.3
            
            phreatomagmatic_prob = min(1.0, phreatomagmatic_prob)
            
            # Mixed style probability
            mixed_conditions = thresholds['mixed_style_conditions']
            temporal_var = mixed_magma_properties.get('temporal_variability', 0.1)
            conduit_complexity = mixed_magma_properties.get('conduit_complexity', 0.2)
            supply_variability = mixed_magma_properties.get('supply_variability', 0.15)
            
            mixed_prob = 0.0
            if temporal_var > mixed_conditions['temporal_variability']:
                mixed_prob += 0.4 * temporal_var
            if conduit_complexity > mixed_conditions['conduit_complexity']:
                mixed_prob += 0.3 * conduit_complexity
            if supply_variability > mixed_conditions['magma_supply_variability']:
                mixed_prob += 0.3 * supply_variability
            
            mixed_prob = min(1.0, mixed_prob)
            
            # Normalize probabilities
            total_prob = effusive_prob + explosive_prob + phreatomagmatic_prob + mixed_prob
            if total_prob > 0:
                style_probabilities = {
                    'effusive': effusive_prob / total_prob,
                    'explosive': explosive_prob / total_prob,
                    'phreatomagmatic': phreatomagmatic_prob / total_prob,
                    'mixed': mixed_prob / total_prob
                }
            else:
                style_probabilities = {
                    'effusive': 0.25, 'explosive': 0.25, 
                    'phreatomagmatic': 0.25, 'mixed': 0.25
                }
            
            # Determine most likely style
            predicted_style = max(style_probabilities, key=style_probabilities.get)
            style_confidence = style_probabilities[predicted_style]
            
            # Calculate controlling parameters
            controlling_parameters = {
                'gas_content_control': gas_content / 0.1,  # Normalized
                'viscosity_control': np.log10(viscosity) / 8,  # Normalized
                'ascent_rate_control': ascent_rate / 1.0,  # Normalized
                'water_interaction_control': water_interaction,
                'crystallinity_control': crystallinity,
                'temperature_control': (temperature - 1000) / 500  # Normalized
            }
            
            # Calculate transition thresholds for this specific case
            transition_thresholds = {
                'effusive_to_explosive': {
                    'gas_content': thresholds['effusive_explosive_threshold']['gas_content'],
                    'viscosity': thresholds['effusive_explosive_threshold']['viscosity'],
                    'ascent_rate': thresholds['effusive_explosive_threshold']['ascent_rate']
                },
                'explosive_to_phreatomagmatic': {
                    'water_interaction': thresholds['explosive_phreatomagmatic_threshold']['water_interaction'],
                    'depth_threshold': thresholds['explosive_phreatomagmatic_threshold']['depth_threshold']
                }
            }
            
            return EruptionStylePrediction(
                magma_properties=mixed_magma_properties,
                predicted_style=predicted_style,
                style_probability=style_probabilities,
                controlling_parameters=controlling_parameters,
                transition_thresholds=transition_thresholds,
                style_confidence=style_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in eruption style prediction: {e}")
            raise    as
ync def model_temporal_eruption_evolution(
        self,
        mixing_progression: List[float],
        time_steps: List[datetime],
        initial_conditions: Dict[str, float]
    ) -> TemporalEruptionEvolution:
        """
        Model temporal eruption evolution based on ongoing mixing
        
        Args:
            mixing_progression: Evolution of mixing degree over time
            time_steps: Time steps for mixing progression
            initial_conditions: Initial magma and chamber conditions
            
        Returns:
            TemporalEruptionEvolution object
        """
        try:
            evolution_timeline = []
            intensity_evolution = []
            style_transitions = []
            
            current_style = 'effusive'  # Default initial style
            
            for i, (mixing_degree, timestamp) in enumerate(zip(mixing_progression, time_steps)):
                # Update magma properties based on mixing progression
                evolved_properties = self._evolve_magma_properties(
                    initial_conditions, mixing_degree, i / len(mixing_progression)
                )
                
                # Predict characteristics at this time step
                intensity_pred = await self.predict_eruption_intensity(
                    mixing_degree, evolved_properties, initial_conditions
                )
                
                style_pred = await self.predict_eruption_style(
                    evolved_properties, initial_conditions
                )
                
                # Record evolution data
                evolution_data = {
                    'mixing_degree': mixing_degree,
                    'intensity': intensity_pred.predicted_intensity,
                    'style': style_pred.predicted_style,
                    'gas_content': evolved_properties.get('gas_content', 0.02),
                    'viscosity': evolved_properties.get('viscosity', 1e5),
                    'temperature': evolved_properties.get('temperature', 1200)
                }
                
                evolution_timeline.append((timestamp, evolution_data))
                intensity_evolution.append(intensity_pred.predicted_intensity)
                
                # Check for style transitions
                if style_pred.predicted_style != current_style:
                    style_transitions.append((timestamp, current_style, style_pred.predicted_style))
                    current_style = style_pred.predicted_style
            
            # Find peak activity time
            peak_intensity_index = np.argmax(intensity_evolution)
            peak_activity_time = time_steps[peak_intensity_index]
            
            # Estimate eruption duration
            # Duration based on intensity evolution and mixing progression
            significant_activity_threshold = 0.5 * max(intensity_evolution)
            significant_indices = [i for i, intensity in enumerate(intensity_evolution) 
                                 if intensity > significant_activity_threshold]
            
            if significant_indices:
                start_time = time_steps[significant_indices[0]]
                end_time = time_steps[significant_indices[-1]]
                eruption_duration = end_time - start_time
            else:
                eruption_duration = time_steps[-1] - time_steps[0]
            
            # Calculate evolution uncertainty
            evolution_uncertainty = self._calculate_evolution_uncertainty(
                mixing_progression, intensity_evolution, time_steps
            )
            
            return TemporalEruptionEvolution(
                evolution_timeline=evolution_timeline,
                mixing_progression=mixing_progression,
                intensity_evolution=intensity_evolution,
                style_transitions=style_transitions,
                peak_activity_time=peak_activity_time,
                eruption_duration=eruption_duration,
                evolution_uncertainty=evolution_uncertainty
            )
            
        except Exception as e:
            logger.error(f"Error in temporal eruption evolution modeling: {e}")
            raise
    
    def _evolve_magma_properties(
        self, 
        initial_conditions: Dict[str, float], 
        mixing_degree: float, 
        time_fraction: float
    ) -> Dict[str, float]:
        """Evolve magma properties based on mixing progression and time"""
        evolved_properties = initial_conditions.copy()
        
        # Gas content evolution (typically increases with mixing)
        initial_gas = initial_conditions.get('gas_content', 0.02)
        evolved_properties['gas_content'] = initial_gas * (1 + 2 * mixing_degree * time_fraction)
        
        # Viscosity evolution (complex relationship with mixing and crystallization)
        initial_viscosity = initial_conditions.get('viscosity', 1e5)
        mixing_viscosity_factor = 1 + 0.5 * mixing_degree  # Mixing can increase viscosity
        crystallization_factor = 1 + 2 * time_fraction * initial_conditions.get('crystallinity', 0.2)
        evolved_properties['viscosity'] = initial_viscosity * mixing_viscosity_factor * crystallization_factor
        
        # Temperature evolution (cooling over time, heating from mixing)
        initial_temp = initial_conditions.get('temperature', 1200)
        cooling_rate = 10 * time_fraction  # K per time fraction
        mixing_heating = 50 * mixing_degree  # K from mixing
        evolved_properties['temperature'] = initial_temp - cooling_rate + mixing_heating
        
        # Crystallinity evolution
        initial_crystallinity = initial_conditions.get('crystallinity', 0.2)
        evolved_properties['crystallinity'] = min(0.8, initial_crystallinity + 0.3 * time_fraction)
        
        return evolved_properties
    
    def _calculate_evolution_uncertainty(
        self, 
        mixing_progression: List[float], 
        intensity_evolution: List[float], 
        time_steps: List[datetime]
    ) -> Dict[str, List[float]]:
        """Calculate uncertainty bounds for evolution parameters"""
        uncertainty = {}
        
        # Mixing degree uncertainty
        mixing_uncertainty = [0.1 + 0.05 * abs(m - 0.5) for m in mixing_progression]
        uncertainty['mixing_degree'] = mixing_uncertainty
        
        # Intensity uncertainty
        intensity_uncertainty = [0.2 + 0.1 * i / 8 for i in intensity_evolution]
        uncertainty['intensity'] = intensity_uncertainty
        
        # Temporal uncertainty (increases with time)
        time_uncertainty = [0.1 * (i / len(time_steps)) for i in range(len(time_steps))]
        uncertainty['temporal'] = time_uncertainty
        
        return uncertainty   
 async def assess_mixing_triggered_hazards(
        self,
        eruption_characteristics: Dict[str, Any],
        location: Dict[str, float],
        population_data: Dict[str, Any]
    ) -> MixingTriggeredHazardAssessment:
        """
        Assess hazards for mixing-triggered eruptions
        
        Args:
            eruption_characteristics: Predicted eruption characteristics
            location: Volcano location (lat, lon, elevation)
            population_data: Population and infrastructure data
            
        Returns:
            MixingTriggeredHazardAssessment object
        """
        try:
            hazard_zones = {}
            population_exposure = {}
            infrastructure_risk = {}
            impact_scenarios = []
            
            # Extract eruption parameters
            intensity = eruption_characteristics.get('intensity', 3.0)
            explosivity = eruption_characteristics.get('explosivity', 0.5)
            discharge_rate = eruption_characteristics.get('discharge_rate', 100)
            style = eruption_characteristics.get('style', 'explosive')
            
            # Assess different hazard types based on eruption characteristics
            
            # 1. Lava flow hazards (for effusive or mixed styles)
            if style in ['effusive', 'mixed']:
                lava_model = self.hazard_models['lava_flow_model']
                lava_zones = self._calculate_lava_flow_hazard_zones(
                    discharge_rate, location, lava_model
                )
                hazard_zones['lava_flows'] = lava_zones
                
                # Population exposure to lava flows
                population_exposure['lava_flows'] = self._calculate_population_exposure(
                    lava_zones, population_data
                )
            
            # 2. Pyroclastic flow hazards (for explosive styles)
            if style in ['explosive', 'mixed', 'phreatomagmatic']:
                pf_model = self.hazard_models['pyroclastic_flow_model']
                pf_zones = self._calculate_pyroclastic_flow_hazard_zones(
                    intensity, explosivity, location, pf_model
                )
                hazard_zones['pyroclastic_flows'] = pf_zones
                
                population_exposure['pyroclastic_flows'] = self._calculate_population_exposure(
                    pf_zones, population_data
                )
            
            # 3. Ash fall hazards
            ash_model = self.hazard_models['ash_fall_model']
            ash_zones = self._calculate_ash_fall_hazard_zones(
                intensity, explosivity, location, ash_model
            )
            hazard_zones['ash_fall'] = ash_zones
            
            population_exposure['ash_fall'] = self._calculate_population_exposure(
                ash_zones, population_data
            )
            
            # 4. Gas hazards
            gas_model = self.hazard_models['gas_hazard_model']
            gas_zones = self._calculate_gas_hazard_zones(
                eruption_characteristics.get('gas_emission_rate', 1000), location, gas_model
            )
            hazard_zones['volcanic_gases'] = gas_zones
            
            population_exposure['volcanic_gases'] = self._calculate_population_exposure(
                gas_zones, population_data
            )
            
            # 5. Lahar hazards (if water interaction)
            if eruption_characteristics.get('water_interaction', 0) > 0.1:
                lahar_model = self.hazard_models['lahar_model']
                lahar_zones = self._calculate_lahar_hazard_zones(
                    discharge_rate, location, lahar_model
                )
                hazard_zones['lahars'] = lahar_zones
                
                population_exposure['lahars'] = self._calculate_population_exposure(
                    lahar_zones, population_data
                )
            
            # Calculate infrastructure risk
            infrastructure_risk = self._calculate_infrastructure_risk(
                hazard_zones, population_data.get('infrastructure', {})
            )
            
            # Generate evacuation recommendations
            evacuation_recommendations = self._generate_evacuation_recommendations(
                hazard_zones, population_exposure, intensity
            )
            
            # Calculate hazard probabilities
            hazard_probabilities = self._calculate_hazard_probabilities(
                eruption_characteristics, hazard_zones
            )
            
            # Generate impact scenarios
            impact_scenarios = self._generate_impact_scenarios(
                hazard_zones, population_exposure, infrastructure_risk, intensity
            )
            
            # Develop risk mitigation strategies
            risk_mitigation_strategies = self._develop_risk_mitigation_strategies(
                hazard_zones, population_exposure, style
            )
            
            return MixingTriggeredHazardAssessment(
                hazard_zones=hazard_zones,
                population_exposure=population_exposure,
                infrastructure_risk=infrastructure_risk,
                evacuation_recommendations=evacuation_recommendations,
                hazard_probabilities=hazard_probabilities,
                impact_scenarios=impact_scenarios,
                risk_mitigation_strategies=risk_mitigation_strategies
            )
            
        except Exception as e:
            logger.error(f"Error in hazard assessment: {e}")
            raise
    
    def _calculate_lava_flow_hazard_zones(
        self, 
        discharge_rate: float, 
        location: Dict[str, float], 
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate lava flow hazard zones"""
        # Simplified lava flow hazard calculation
        # In reality, this would use detailed topographic modeling
        
        base_distance = 5000  # Base hazard distance in meters
        discharge_factor = (discharge_rate / 100) ** 0.5  # Discharge rate effect
        
        hazard_zones = {}
        for zone, params in model['hazard_zones'].items():
            distance = base_distance * discharge_factor * (params['distance'] / 5000)
            hazard_zones[zone] = {
                'distance': distance,
                'probability': params['probability'],
                'area': np.pi * distance**2,  # Simplified circular area
                'flow_thickness': max(1, 10 * discharge_rate / 1000)  # meters
            }
        
        return hazard_zones
    
    def _calculate_pyroclastic_flow_hazard_zones(
        self, 
        intensity: float, 
        explosivity: float, 
        location: Dict[str, float], 
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate pyroclastic flow hazard zones"""
        # Simplified pyroclastic flow hazard calculation
        
        base_distance = 15000  # Base hazard distance in meters
        intensity_factor = (intensity / 5) ** 1.2
        explosivity_factor = explosivity ** 0.8
        
        hazard_zones = {}
        for zone, params in model['hazard_zones'].items():
            distance = base_distance * intensity_factor * explosivity_factor * (params['distance'] / 15000)
            hazard_zones[zone] = {
                'distance': distance,
                'probability': params['probability'] * explosivity,
                'area': np.pi * distance**2,  # Simplified circular area
                'temperature': max(200, 800 * explosivity),  # Celsius
                'velocity': max(10, 50 * intensity_factor)   # m/s
            }
        
        return hazard_zones
    
    def _calculate_ash_fall_hazard_zones(
        self, 
        intensity: float, 
        explosivity: float, 
        location: Dict[str, float], 
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate ash fall hazard zones"""
        # Simplified ash fall hazard calculation
        # In reality, this would use atmospheric dispersion models
        
        column_height = 1000 * (2 ** intensity)  # Empirical column height (m)
        wind_speed = 10  # Assumed wind speed (m/s)
        wind_direction = 90  # Assumed wind direction (degrees)
        
        hazard_zones = {}
        for zone, params in model['hazard_zones'].items():
            # Elliptical dispersion pattern
            downwind_distance = column_height * 0.1 * (1 / params['thickness'])
            crosswind_distance = downwind_distance * 0.3
            
            hazard_zones[zone] = {
                'downwind_distance': downwind_distance,
                'crosswind_distance': crosswind_distance,
                'probability': params['probability'] * explosivity,
                'thickness': params['thickness'] * intensity,  # meters
                'area': np.pi * downwind_distance * crosswind_distance,
                'loading': params['thickness'] * 1500  # kg/m2 (assuming 1500 kg/m3 density)
            }
        
        return hazard_zones
    
    def _calculate_gas_hazard_zones(
        self, 
        emission_rate: float, 
        location: Dict[str, float], 
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate volcanic gas hazard zones"""
        # Simplified gas dispersion calculation
        
        wind_speed = 5  # m/s
        atmospheric_stability = 'neutral'
        
        hazard_zones = {}
        for zone, params in model['hazard_zones'].items():
            # Gaussian plume dispersion
            distance = (emission_rate / params['concentration']) ** 0.5 * 1000
            width = distance * 0.2  # Plume width
            
            hazard_zones[zone] = {
                'distance': distance,
                'width': width,
                'probability': params['probability'],
                'concentration': params['concentration'],  # ppm
                'area': distance * width,
                'exposure_time': 3600  # seconds (1 hour)
            }
        
        return hazard_zones
    
    def _calculate_lahar_hazard_zones(
        self, 
        discharge_rate: float, 
        location: Dict[str, float], 
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate lahar hazard zones"""
        # Simplified lahar hazard calculation
        
        water_availability = 0.5  # Assumed water availability factor
        debris_factor = 2.0  # Debris bulking factor
        
        hazard_zones = {}
        for zone, params in model['hazard_zones'].items():
            # Flow routing along valleys
            distance = params['distance'] * (discharge_rate / 100) ** 0.3 * water_availability
            
            hazard_zones[zone] = {
                'distance': distance,
                'probability': params['probability'] * water_availability,
                'area': distance * 500,  # Assuming 500m average valley width
                'velocity': max(5, 20 * (discharge_rate / 1000)),  # m/s
                'depth': max(1, 5 * (discharge_rate / 1000)),      # meters
                'debris_content': debris_factor
            }
        
        return hazard_zones    def
 _calculate_population_exposure(
        self, 
        hazard_zones: Dict[str, Any], 
        population_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """Calculate population exposure for hazard zones"""
        exposure = {}
        
        total_population = population_data.get('total_population', 100000)
        population_density = population_data.get('population_density', 100)  # people/km2
        
        for zone, zone_data in hazard_zones.items():
            area_km2 = zone_data.get('area', 0) / 1e6  # Convert m2 to km2
            exposed_population = int(area_km2 * population_density * zone_data.get('probability', 1.0))
            exposure[zone] = min(exposed_population, total_population)
        
        return exposure
    
    def _calculate_infrastructure_risk(
        self, 
        hazard_zones: Dict[str, Dict[str, Any]], 
        infrastructure_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate infrastructure risk"""
        risk = {}
        
        # Infrastructure types and their vulnerability
        infrastructure_types = {
            'buildings': {'vulnerability': 0.8, 'value_per_unit': 200000},
            'roads': {'vulnerability': 0.6, 'value_per_km': 1000000},
            'utilities': {'vulnerability': 0.9, 'value_per_unit': 500000},
            'airports': {'vulnerability': 0.7, 'value_per_unit': 50000000},
            'hospitals': {'vulnerability': 0.5, 'value_per_unit': 10000000}
        }
        
        for hazard_type, zones in hazard_zones.items():
            hazard_risk = 0
            for zone, zone_data in zones.items():
                zone_area_km2 = zone_data.get('area', 0) / 1e6
                zone_probability = zone_data.get('probability', 1.0)
                
                for infra_type, infra_params in infrastructure_types.items():
                    density = infrastructure_data.get(f'{infra_type}_density', 1.0)  # units per km2
                    units_at_risk = zone_area_km2 * density
                    vulnerability = infra_params['vulnerability']
                    value_per_unit = infra_params.get('value_per_unit', infra_params.get('value_per_km', 1000000))
                    
                    expected_loss = (units_at_risk * zone_probability * 
                                   vulnerability * value_per_unit)
                    hazard_risk += expected_loss
            
            risk[hazard_type] = hazard_risk
        
        return risk
    
    def _generate_evacuation_recommendations(
        self, 
        hazard_zones: Dict[str, Dict[str, Any]], 
        population_exposure: Dict[str, Dict[str, int]], 
        intensity: float
    ) -> Dict[str, Any]:
        """Generate evacuation recommendations"""
        recommendations = {}
        
        # Determine evacuation zones based on hazard severity
        high_risk_population = 0
        medium_risk_population = 0
        
        for hazard_type, zones in hazard_zones.items():
            for zone, zone_data in zones.items():
                if zone == 'high_risk' and zone_data.get('probability', 0) > 0.7:
                    high_risk_population += population_exposure.get(hazard_type, {}).get(zone, 0)
                elif zone in ['medium_risk', 'high_risk'] and zone_data.get('probability', 0) > 0.4:
                    medium_risk_population += population_exposure.get(hazard_type, {}).get(zone, 0)
        
        # Evacuation timeline based on intensity
        if intensity >= 5:
            evacuation_time = 24  # hours
            evacuation_radius = 20000  # meters
        elif intensity >= 3:
            evacuation_time = 48  # hours
            evacuation_radius = 10000  # meters
        else:
            evacuation_time = 72  # hours
            evacuation_radius = 5000  # meters
        
        recommendations = {
            'immediate_evacuation_population': high_risk_population,
            'precautionary_evacuation_population': medium_risk_population,
            'evacuation_radius': evacuation_radius,
            'evacuation_timeline': evacuation_time,
            'shelter_requirements': high_risk_population + medium_risk_population,
            'transportation_needs': (high_risk_population + medium_risk_population) * 0.3,  # 30% need assistance
            'special_needs_population': (high_risk_population + medium_risk_population) * 0.1  # 10% special needs
        }
        
        return recommendations
    
    def _calculate_hazard_probabilities(
        self, 
        eruption_characteristics: Dict[str, Any], 
        hazard_zones: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate probabilities for different hazards"""
        probabilities = {}
        
        style = eruption_characteristics.get('style', 'explosive')
        intensity = eruption_characteristics.get('intensity', 3.0)
        explosivity = eruption_characteristics.get('explosivity', 0.5)
        
        # Base probabilities by eruption style
        if style == 'effusive':
            probabilities = {
                'lava_flows': 0.9,
                'pyroclastic_flows': 0.1,
                'ash_fall': 0.3,
                'volcanic_gases': 0.7,
                'lahars': 0.2
            }
        elif style == 'explosive':
            probabilities = {
                'lava_flows': 0.2,
                'pyroclastic_flows': 0.8,
                'ash_fall': 0.95,
                'volcanic_gases': 0.9,
                'lahars': 0.4
            }
        elif style == 'phreatomagmatic':
            probabilities = {
                'lava_flows': 0.1,
                'pyroclastic_flows': 0.9,
                'ash_fall': 0.98,
                'volcanic_gases': 0.8,
                'lahars': 0.8
            }
        else:  # mixed
            probabilities = {
                'lava_flows': 0.6,
                'pyroclastic_flows': 0.6,
                'ash_fall': 0.8,
                'volcanic_gases': 0.8,
                'lahars': 0.5
            }
        
        # Adjust probabilities based on intensity and explosivity
        intensity_factor = min(2.0, intensity / 4.0)
        explosivity_factor = explosivity
        
        for hazard in probabilities:
            if hazard in ['pyroclastic_flows', 'ash_fall']:
                probabilities[hazard] *= explosivity_factor
            probabilities[hazard] *= intensity_factor
            probabilities[hazard] = min(1.0, probabilities[hazard])
        
        return probabilities
    
    def _generate_impact_scenarios(
        self, 
        hazard_zones: Dict[str, Dict[str, Any]], 
        population_exposure: Dict[str, Dict[str, int]], 
        infrastructure_risk: Dict[str, float], 
        intensity: float
    ) -> List[Dict[str, Any]]:
        """Generate impact scenarios"""
        scenarios = []
        
        # Low impact scenario
        low_scenario = {
            'scenario_name': 'Low Impact',
            'probability': 0.3,
            'fatalities': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.001,
            'injuries': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.01,
            'evacuees': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.1,
            'economic_loss': sum(infrastructure_risk.values()) * 0.1,
            'duration': '1-3 days',
            'recovery_time': '1-6 months'
        }
        scenarios.append(low_scenario)
        
        # Medium impact scenario
        medium_scenario = {
            'scenario_name': 'Medium Impact',
            'probability': 0.5,
            'fatalities': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.01,
            'injuries': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.05,
            'evacuees': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.5,
            'economic_loss': sum(infrastructure_risk.values()) * 0.5,
            'duration': '1-2 weeks',
            'recovery_time': '6 months - 2 years'
        }
        scenarios.append(medium_scenario)
        
        # High impact scenario
        high_scenario = {
            'scenario_name': 'High Impact',
            'probability': 0.2,
            'fatalities': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.05,
            'injuries': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.15,
            'evacuees': sum(sum(zones.values()) for zones in population_exposure.values()) * 0.8,
            'economic_loss': sum(infrastructure_risk.values()),
            'duration': '1-6 months',
            'recovery_time': '2-10 years'
        }
        scenarios.append(high_scenario)
        
        return scenarios
    
    def _develop_risk_mitigation_strategies(
        self, 
        hazard_zones: Dict[str, Dict[str, Any]], 
        population_exposure: Dict[str, Dict[str, int]], 
        style: str
    ) -> List[str]:
        """Develop risk mitigation strategies"""
        strategies = []
        
        # General strategies
        strategies.extend([
            "Establish early warning systems with real-time monitoring",
            "Develop and practice evacuation plans for high-risk areas",
            "Create emergency supply caches and shelter facilities",
            "Implement building codes resistant to volcanic hazards",
            "Establish communication systems for emergency coordination"
        ])
        
        # Style-specific strategies
        if style in ['explosive', 'phreatomagmatic']:
            strategies.extend([
                "Install ash fall protection systems for critical infrastructure",
                "Develop respiratory protection programs for residents",
                "Create pyroclastic flow barriers and deflection structures",
                "Establish aircraft ash avoidance procedures"
            ])
        
        if style in ['effusive', 'mixed']:
            strategies.extend([
                "Construct lava flow diversion barriers",
                "Develop water cooling systems for lava flows",
                "Create firebreaks around critical infrastructure",
                "Establish lava flow monitoring and prediction systems"
            ])
        
        if 'lahars' in hazard_zones:
            strategies.extend([
                "Install lahar detection and warning systems",
                "Construct lahar retention basins and channels",
                "Develop rapid drainage systems for crater lakes",
                "Create elevated evacuation routes in valleys"
            ])
        
        return strategies  
  async def quantify_prediction_uncertainty(
        self,
        prediction_results: Dict[str, Any],
        model_parameters: Dict[str, Any]
    ) -> UncertaintyQuantification:
        """
        Quantify uncertainty for eruption predictions
        
        Args:
            prediction_results: Results from eruption characteristic predictions
            model_parameters: Parameters used in prediction models
            
        Returns:
            UncertaintyQuantification object
        """
        try:
            # Get uncertainty models
            aleatory = self.uncertainty_models['aleatory_uncertainty']
            epistemic = self.uncertainty_models['epistemic_uncertainty']
            model_unc = self.uncertainty_models['model_uncertainty']
            correlations = self.uncertainty_models['correlation_structure']
            
            # Calculate aleatory uncertainty (natural variability)
            aleatory_uncertainty = {}
            for param, base_uncertainty in aleatory.items():
                # Scale uncertainty based on prediction magnitude
                param_value = prediction_results.get(param.replace('_variability', ''), 1.0)
                if isinstance(param_value, (int, float)):
                    aleatory_uncertainty[param] = base_uncertainty * (1 + 0.1 * abs(np.log10(param_value)))
                else:
                    aleatory_uncertainty[param] = base_uncertainty
            
            # Calculate epistemic uncertainty (knowledge uncertainty)
            epistemic_uncertainty = {}
            for param, base_uncertainty in epistemic.items():
                # Increase uncertainty for extrapolation beyond training data
                data_coverage = model_parameters.get('data_coverage', 0.8)
                epistemic_uncertainty[param] = base_uncertainty * (2 - data_coverage)
            
            # Calculate model uncertainty
            model_uncertainty = {}
            for param, base_uncertainty in model_unc.items():
                # Adjust based on model validation performance
                validation_score = model_parameters.get(f'{param}_validation_score', 0.7)
                model_uncertainty[param] = base_uncertainty * (1.5 - validation_score)
            
            # Calculate parameter uncertainty
            parameter_uncertainty = {}
            for param in ['intensity', 'explosivity', 'discharge_rate', 'style_probability']:
                if param in prediction_results:
                    # Combine different uncertainty sources
                    aleatory_comp = aleatory_uncertainty.get(f'{param}_variability', 0.1)
                    epistemic_comp = epistemic_uncertainty.get('model_parameter_uncertainty', 0.2)
                    model_comp = model_uncertainty.get(f'{param}_model_uncertainty', 0.3)
                    
                    # Root sum of squares for independent uncertainties
                    parameter_uncertainty[param] = np.sqrt(aleatory_comp**2 + epistemic_comp**2 + model_comp**2)
            
            # Calculate total uncertainty with correlations
            total_uncertainty = {}
            for param in parameter_uncertainty:
                base_uncertainty = parameter_uncertainty[param]
                
                # Apply correlation effects
                correlation_factor = 1.0
                for other_param in parameter_uncertainty:
                    if param != other_param:
                        corr_key = f'{param}_{other_param}_correlation'
                        correlation = correlations.get(corr_key, 0.0)
                        if correlation != 0:
                            correlation_factor += 0.1 * abs(correlation)
                
                total_uncertainty[param] = base_uncertainty * correlation_factor
            
            # Calculate confidence intervals (assuming normal distribution)
            confidence_intervals = {}
            for param, uncertainty in total_uncertainty.items():
                if param in prediction_results:
                    mean_value = prediction_results[param]
                    if isinstance(mean_value, (int, float)):
                        # 95% confidence interval (±1.96σ)
                        lower_bound = mean_value - 1.96 * uncertainty * mean_value
                        upper_bound = mean_value + 1.96 * uncertainty * mean_value
                        confidence_intervals[param] = (lower_bound, upper_bound)
            
            # Perform sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(
                prediction_results, model_parameters, total_uncertainty
            )
            
            return UncertaintyQuantification(
                aleatory_uncertainty=aleatory_uncertainty,
                epistemic_uncertainty=epistemic_uncertainty,
                model_uncertainty=model_uncertainty,
                parameter_uncertainty=parameter_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_intervals=confidence_intervals,
                sensitivity_analysis=sensitivity_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in uncertainty quantification: {e}")
            raise
    
    def _perform_sensitivity_analysis(
        self,
        prediction_results: Dict[str, Any],
        model_parameters: Dict[str, Any],
        uncertainties: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform sensitivity analysis for prediction parameters"""
        sensitivity = {}
        
        # Parameters to analyze
        key_parameters = ['mixing_degree', 'volatile_content', 'viscosity', 'temperature', 'pressure']
        
        for param in key_parameters:
            if param in model_parameters:
                base_value = model_parameters[param]
                uncertainty = uncertainties.get(param, 0.1)
                
                # Calculate sensitivity as relative change in output per relative change in input
                # This is a simplified approach; in practice, would use more sophisticated methods
                param_sensitivity = uncertainty * 0.5  # Simplified sensitivity coefficient
                sensitivity[param] = param_sensitivity
        
        return sensitivity
    
    async def validate_against_historical_eruptions(
        self,
        prediction_results: EruptionCharacteristicPredictionResult,
        historical_eruption_name: str
    ) -> HistoricalValidationResult:
        """
        Validate predictions against historical mixing-related eruptions
        
        Args:
            prediction_results: Current prediction results
            historical_eruption_name: Name of historical eruption for validation
            
        Returns:
            HistoricalValidationResult object
        """
        try:
            # Get historical eruption data
            if historical_eruption_name not in self.historical_eruption_database:
                raise ValueError(f"Historical eruption {historical_eruption_name} not found in database")
            
            historical_data = self.historical_eruption_database[historical_eruption_name]
            observed_characteristics = historical_data['observed_characteristics']
            
            # Extract predicted characteristics
            predicted_characteristics = {
                'intensity': prediction_results.intensity_prediction.predicted_intensity,
                'explosivity': prediction_results.explosivity_forecast.predicted_explosivity,
                'discharge_rate': prediction_results.discharge_rate_prediction.predicted_discharge_rate,
                'style': prediction_results.style_prediction.predicted_style
            }
            
            # Calculate prediction accuracy
            prediction_accuracy = {}
            residuals = {}
            
            for characteristic in ['intensity', 'explosivity', 'discharge_rate']:
                if characteristic in observed_characteristics and characteristic in predicted_characteristics:
                    observed = observed_characteristics[characteristic]
                    predicted = predicted_characteristics[characteristic]
                    
                    # Calculate relative error
                    if observed != 0:
                        relative_error = abs(predicted - observed) / abs(observed)
                        accuracy = max(0, 1 - relative_error)
                    else:
                        accuracy = 1.0 if predicted == 0 else 0.0
                    
                    prediction_accuracy[characteristic] = accuracy
                    residuals[characteristic] = predicted - observed
            
            # Style prediction accuracy (categorical)
            observed_style = observed_characteristics.get('style', 'unknown')
            predicted_style = predicted_characteristics.get('style', 'unknown')
            style_accuracy = 1.0 if observed_style == predicted_style else 0.0
            prediction_accuracy['style'] = style_accuracy
            residuals['style'] = 0 if observed_style == predicted_style else 1
            
            # Calculate overall validation metrics
            validation_metrics = {
                'overall_accuracy': np.mean(list(prediction_accuracy.values())),
                'rmse': np.sqrt(np.mean([r**2 for r in residuals.values() if isinstance(r, (int, float))])),
                'mae': np.mean([abs(r) for r in residuals.values() if isinstance(r, (int, float))]),
                'bias': np.mean([r for r in residuals.values() if isinstance(r, (int, float))])
            }
            
            # Generate lessons learned
            lessons_learned = self._generate_lessons_learned(
                prediction_accuracy, residuals, historical_data
            )
            
            return HistoricalValidationResult(
                eruption_name=historical_eruption_name,
                eruption_date=historical_data['eruption_date'],
                predicted_characteristics=predicted_characteristics,
                observed_characteristics=observed_characteristics,
                prediction_accuracy=prediction_accuracy,
                residuals=residuals,
                validation_metrics=validation_metrics,
                lessons_learned=lessons_learned
            )
            
        except Exception as e:
            logger.error(f"Error in historical validation: {e}")
            raise
    
    def _generate_lessons_learned(
        self,
        prediction_accuracy: Dict[str, float],
        residuals: Dict[str, float],
        historical_data: Dict[str, Any]
    ) -> List[str]:
        """Generate lessons learned from validation results"""
        lessons = []
        
        # Analyze prediction accuracy
        for characteristic, accuracy in prediction_accuracy.items():
            if accuracy < 0.7:
                lessons.append(f"Low accuracy in {characteristic} prediction ({accuracy:.2f}) - "
                             f"model may need improvement for this parameter")
        
        # Analyze residuals for systematic bias
        numeric_residuals = [r for r in residuals.values() if isinstance(r, (int, float))]
        if numeric_residuals:
            mean_residual = np.mean(numeric_residuals)
            if abs(mean_residual) > 0.1:
                bias_direction = "over-prediction" if mean_residual > 0 else "under-prediction"
                lessons.append(f"Systematic {bias_direction} detected - "
                             f"model calibration may need adjustment")
        
        # Analyze specific characteristics
        if 'intensity' in residuals and abs(residuals['intensity']) > 1.0:
            lessons.append("Large intensity prediction error - consider improving mixing degree "
                         "to intensity relationship")
        
        if 'explosivity' in residuals and abs(residuals['explosivity']) > 0.2:
            lessons.append("Explosivity prediction needs improvement - volatile content evolution "
                         "modeling may require refinement")
        
        # Historical context lessons
        mixing_evidence = historical_data.get('mixing_evidence', 'unknown')
        if mixing_evidence == 'disequilibrium_textures':
            lessons.append("Disequilibrium textures indicate rapid mixing - consider time-dependent "
                         "mixing effects in future predictions")
        elif mixing_evidence == 'banded_pumice':
            lessons.append("Banded pumice suggests incomplete mixing - improve mixing efficiency "
                         "calculations")
        
        return lessons
    
    def _calculate_prediction_confidence(
        self,
        primary_parameter: float,
        supporting_data: Dict[str, Any],
        prediction_type: str
    ) -> float:
        """Calculate confidence level for predictions"""
        base_confidence = 0.7  # Base confidence level
        
        # Adjust confidence based on data quality
        data_completeness = len(supporting_data) / 10  # Assume 10 ideal parameters
        data_quality_factor = min(1.0, data_completeness)
        
        # Adjust confidence based on parameter ranges
        if prediction_type == 'intensity':
            # Higher confidence for moderate intensities
            if 2 <= primary_parameter <= 5:
                range_factor = 1.0
            else:
                range_factor = 0.8
        elif prediction_type == 'explosivity':
            # Higher confidence for moderate explosivities
            if 0.3 <= primary_parameter <= 0.8:
                range_factor = 1.0
            else:
                range_factor = 0.8
        else:
            range_factor = 0.9
        
        # Adjust confidence based on model validation performance
        validation_factor = 0.85  # Based on historical validation results
        
        # Calculate final confidence
        confidence = base_confidence * data_quality_factor * range_factor * validation_factor
        return min(1.0, max(0.1, confidence))
    
    async def store_prediction_results(
        self,
        prediction_result: EruptionCharacteristicPredictionResult
    ) -> str:
        """Store prediction results in database"""
        try:
            async with self.db_pool.acquire() as conn:
                # Store main prediction result
                query = """
                INSERT INTO eruption_prediction_results (
                    prediction_id, chamber_id, timestamp, mixing_degree,
                    eruption_intensity_prediction, explosivity_forecast,
                    discharge_rate_prediction, eruption_style_prediction,
                    hazard_assessment, uncertainty_bounds, validation_metrics
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """
                
                await conn.execute(
                    query,
                    prediction_result.prediction_id,
                    prediction_result.chamber_id,
                    prediction_result.prediction_timestamp,
                    prediction_result.mixing_degree,
                    json.dumps({
                        'predicted_intensity': prediction_result.intensity_prediction.predicted_intensity,
                        'intensity_confidence': prediction_result.intensity_prediction.intensity_confidence,
                        'intensity_bounds': prediction_result.intensity_prediction.intensity_bounds,
                        'contributing_factors': prediction_result.intensity_prediction.contributing_factors
                    }),
                    json.dumps({
                        'predicted_explosivity': prediction_result.explosivity_forecast.predicted_explosivity,
                        'explosivity_index': prediction_result.explosivity_forecast.explosivity_index,
                        'gas_pressure_buildup': prediction_result.explosivity_forecast.gas_pressure_buildup,
                        'explosivity_confidence': prediction_result.explosivity_forecast.explosivity_confidence
                    }),
                    json.dumps({
                        'predicted_discharge_rate': prediction_result.discharge_rate_prediction.predicted_discharge_rate,
                        'peak_discharge_rate': prediction_result.discharge_rate_prediction.peak_discharge_rate,
                        'sustained_discharge_rate': prediction_result.discharge_rate_prediction.sustained_discharge_rate,
                        'rheological_controls': prediction_result.discharge_rate_prediction.rheological_controls
                    }),
                    json.dumps({
                        'predicted_style': prediction_result.style_prediction.predicted_style,
                        'style_probability': prediction_result.style_prediction.style_probability,
                        'controlling_parameters': prediction_result.style_prediction.controlling_parameters,
                        'style_confidence': prediction_result.style_prediction.style_confidence
                    }),
                    json.dumps({
                        'hazard_zones': prediction_result.hazard_assessment.hazard_zones,
                        'population_exposure': prediction_result.hazard_assessment.population_exposure,
                        'evacuation_recommendations': prediction_result.hazard_assessment.evacuation_recommendations
                    }),
                    json.dumps({
                        'total_uncertainty': prediction_result.uncertainty_quantification.total_uncertainty,
                        'confidence_intervals': prediction_result.uncertainty_quantification.confidence_intervals
                    }),
                    json.dumps({
                        'validation_score': prediction_result.validation_score,
                        'confidence_level': prediction_result.confidence_level
                    })
                )
                
                logger.info(f"Stored prediction results for {prediction_result.prediction_id}")
                return prediction_result.prediction_id
                
        except Exception as e:
            logger.error(f"Error storing prediction results: {e}")
            raise
    
    async def close(self):
        """Close database connections and cleanup resources"""
        if self.db_pool:
            await self.db_pool.close()
        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("EruptionCharacteristicPredictionSystem closed successfully")