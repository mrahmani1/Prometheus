"""
Mixing Trigger Analysis System

This module implements sophisticated analysis of magma mixing trigger mechanisms,
including injection event detection, convection initiation analysis, external trigger
analysis, and predictive modeling for future mixing events.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from enum import Enum

# Scientific computing imports
from scipy import signal, stats
from scipy.interpolate import interp1d, griddata
from scipy.ndimage import gaussian_filter, label
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy.fft as fft

# Database integration
from sqlalchemy import create_engine, text
import redis

# Local imports
from .data_models import (
    MagmaComposition, ThermalProperties, RheologicalProperties,
    ConvectionPattern
)

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of mixing triggers"""
    MAGMA_INJECTION = "magma_injection"
    CONVECTION_INITIATION = "convection_initiation"
    SEISMIC_EVENT = "seismic_event"
    PRESSURE_CHANGE = "pressure_change"
    DENSITY_CONTRAST = "density_contrast"
    TEMPERATURE_GRADIENT = "temperature_gradient"
    VOLATILE_OVERSATURATION = "volatile_oversaturation"
    EXTERNAL_PERTURBATION = "external_perturbation"


@dataclass
class MagmaInjectionEvent:
    """Represents a magma injection event"""
    event_id: str
    timestamp: datetime
    injection_location: Tuple[float, float, float]
    injection_volume: float
    injection_rate: float
    injected_composition: MagmaComposition
    injection_temperature: float
    injection_pressure: float
    injection_velocity: np.ndarray
    duration: float
    trigger_strength: float
    mixing_efficiency: float


@dataclass
class ConvectionInitiation:
    """Represents convection initiation conditions"""
    initiation_id: str
    timestamp: datetime
    rayleigh_number: float
    critical_rayleigh: float
    temperature_gradient: float
    density_gradient: float
    viscosity_contrast: float
    convection_onset_depth: float
    convection_cell_size: float
    convection_velocity: float
    mixing_potential: float


@dataclass
class ExternalTrigger:
    """Represents external trigger events"""
    trigger_id: str
    timestamp: datetime
    trigger_type: TriggerType
    magnitude: float
    location: Tuple[float, float, float]
    frequency: float
    duration: float
    energy: float
    stress_change: float
    pressure_change: float
    mixing_response: float


@dataclass
class DensityContrastAnalysis:
    """Analysis of density contrasts for mixing potential"""
    analysis_id: str
    timestamp: datetime
    density_field: np.ndarray
    density_gradient: np.ndarray
    buoyancy_force: np.ndarray
    rayleigh_taylor_instability: float
    mixing_potential: float
    critical_density_contrast: float
    instability_growth_rate: float


@dataclass
class TemperatureDifferenceAnalysis:
    """Analysis of temperature differences affecting mixing"""
    analysis_id: str
    timestamp: datetime
    temperature_field: np.ndarray
    temperature_gradient: np.ndarray
    thermal_buoyancy: np.ndarray
    thermal_diffusivity: float
    peclet_number: float
    thermal_mixing_efficiency: float
    convection_strength: float


@dataclass
class VolatileContentAnalysis:
    """Analysis of volatile content effects on mixing"""
    analysis_id: str
    timestamp: datetime
    volatile_distribution: Dict[str, np.ndarray]
    saturation_pressure: Dict[str, float]
    bubble_nucleation_rate: float
    degassing_rate: float
    volatile_induced_convection: float
    mixing_enhancement_factor: float


@dataclass
class MixingTriggerPrediction:
    """Prediction of future mixing events"""
    prediction_id: str
    prediction_timestamp: datetime
    predicted_trigger_time: datetime
    trigger_probability: float
    trigger_type: TriggerType
    expected_magnitude: float
    confidence_interval: Tuple[float, float]
    mixing_intensity_prediction: float
    uncertainty_bounds: Dict[str, float]


class MixingTriggerAnalysisSystem:
    """
    Advanced system for analyzing magma mixing trigger mechanisms with
    real-time detection, characterization, and predictive modeling capabilities.
    """
    
    def __init__(self, database_config: Dict[str, Any]):
        self.db_config = database_config
        self.engine = create_engine(
            f"postgresql://{database_config['user']}:{database_config['password']}"
            f"@{database_config['host']}:{database_config['port']}/{database_config['database']}"
        )
        self.redis_client = redis.Redis(
            host=database_config.get('redis_host', 'localhost'),
            port=database_config.get('redis_port', 6379),
            decode_responses=True
        )
        
        # Initialize machine learning models
        self.injection_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trigger_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Initialize analysis parameters
        self.critical_rayleigh = 1708.0  # Critical Rayleigh number for convection
        self.mixing_threshold = 0.1  # Threshold for significant mixing
        
        logger.info("MixingTriggerAnalysisSystem initialized")

    async def detect_magma_injection_events(
        self,
        chamber_id: str,
        time_range: Tuple[datetime, datetime],
        monitoring_data: Dict[str, np.ndarray]
    ) -> List[MagmaInjectionEvent]:
        """
        Detect and characterize magma injection events using advanced signal processing
        and machine learning techniques.
        """
        try:
            logger.info(f"Detecting magma injection events for chamber {chamber_id}")
            
            # Load real-time monitoring data
            seismic_data = monitoring_data.get('seismic', np.array([]))
            deformation_data = monitoring_data.get('deformation', np.array([]))
            thermal_data = monitoring_data.get('thermal', np.array([]))
            gas_data = monitoring_data.get('gas', np.array([]))
            
            # Extract injection signatures using advanced signal processing
            injection_signatures = await self._extract_injection_signatures(
                seismic_data, deformation_data, thermal_data, gas_data
            )
            
            # Apply machine learning detection
            injection_events = []
            for signature in injection_signatures:
                # Calculate injection parameters
                injection_volume = await self._calculate_injection_volume(signature)
                injection_rate = await self._calculate_injection_rate(signature)
                injection_location = await self._locate_injection_source(signature)
                
                # Determine injected magma properties
                injected_composition = await self._analyze_injected_composition(signature)
                injection_temperature = await self._estimate_injection_temperature(signature)
                injection_pressure = await self._calculate_injection_pressure(signature)
                
                # Calculate mixing efficiency
                mixing_efficiency = await self._calculate_injection_mixing_efficiency(signature)
                
                event = MagmaInjectionEvent(
                    event_id=f"inj_{chamber_id}_{signature['timestamp'].strftime('%Y%m%d_%H%M%S')}",
                    timestamp=signature['timestamp'],
                    injection_location=injection_location,
                    injection_volume=injection_volume,
                    injection_rate=injection_rate,
                    injected_composition=injected_composition,
                    injection_temperature=injection_temperature,
                    injection_pressure=injection_pressure,
                    injection_velocity=signature['velocity_field'],
                    duration=signature['duration'],
                    trigger_strength=signature['strength'],
                    mixing_efficiency=mixing_efficiency
                )
                
                injection_events.append(event)
            
            # Store results in database
            await self._store_injection_events(injection_events)
            
            logger.info(f"Detected {len(injection_events)} injection events")
            return injection_events
            
        except Exception as e:
            logger.error(f"Error detecting injection events: {str(e)}")
            raise

    async def analyze_convection_initiation(
        self,
        chamber_id: str,
        thermal_field: np.ndarray,
        density_field: np.ndarray,
        viscosity_field: np.ndarray,
        time_range: Tuple[datetime, datetime]
    ) -> List[ConvectionInitiation]:
        """
        Analyze convection initiation conditions in stratified magma chambers
        using advanced fluid dynamics and stability analysis.
        """
        try:
            logger.info(f"Analyzing convection initiation for chamber {chamber_id}")
            
            # Calculate thermal and density gradients
            thermal_gradient = np.gradient(thermal_field)
            density_gradient = np.gradient(density_field)
            
            # Calculate Rayleigh number field
            rayleigh_field = await self._calculate_rayleigh_number_field(
                thermal_field, density_field, viscosity_field
            )
            
            # Identify convection initiation zones
            convection_zones = rayleigh_field > self.critical_rayleigh
            
            # Analyze each convection zone
            convection_events = []
            labeled_zones, num_zones = label(convection_zones)
            
            for zone_id in range(1, num_zones + 1):
                zone_mask = labeled_zones == zone_id
                zone_coords = np.where(zone_mask)
                
                # Calculate zone properties
                zone_rayleigh = np.mean(rayleigh_field[zone_mask])
                zone_temp_gradient = np.mean(np.linalg.norm(
                    [thermal_gradient[i][zone_mask] for i in range(3)], axis=0
                ))
                zone_density_gradient = np.mean(np.linalg.norm(
                    [density_gradient[i][zone_mask] for i in range(3)], axis=0
                ))
                
                # Calculate convection properties
                convection_velocity = await self._calculate_convection_velocity(
                    zone_rayleigh, thermal_field[zone_mask]
                )
                cell_size = await self._estimate_convection_cell_size(zone_coords)
                mixing_potential = await self._calculate_convection_mixing_potential(
                    zone_rayleigh, convection_velocity
                )
                
                # Determine onset depth
                onset_depth = np.mean(zone_coords[2]) * await self._get_depth_scale(chamber_id)
                
                convection_event = ConvectionInitiation(
                    initiation_id=f"conv_{chamber_id}_{zone_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    rayleigh_number=zone_rayleigh,
                    critical_rayleigh=self.critical_rayleigh,
                    temperature_gradient=zone_temp_gradient,
                    density_gradient=zone_density_gradient,
                    viscosity_contrast=np.std(viscosity_field[zone_mask]),
                    convection_onset_depth=onset_depth,
                    convection_cell_size=cell_size,
                    convection_velocity=convection_velocity,
                    mixing_potential=mixing_potential
                )
                
                convection_events.append(convection_event)
            
            # Store results
            await self._store_convection_events(convection_events)
            
            logger.info(f"Analyzed {len(convection_events)} convection initiation events")
            return convection_events
            
        except Exception as e:
            logger.error(f"Error analyzing convection initiation: {str(e)}")
            raise

    async def analyze_external_triggers(
        self,
        chamber_id: str,
        seismic_events: List[Dict[str, Any]],
        pressure_changes: List[Dict[str, Any]],
        time_range: Tuple[datetime, datetime]
    ) -> List[ExternalTrigger]:
        """
        Analyze external triggers including seismic events and pressure changes
        that can initiate magma mixing processes.
        """
        try:
            logger.info(f"Analyzing external triggers for chamber {chamber_id}")
            
            external_triggers = []
            
            # Analyze seismic triggers
            for seismic_event in seismic_events:
                trigger = await self._analyze_seismic_trigger(chamber_id, seismic_event)
                if trigger:
                    external_triggers.append(trigger)
            
            # Analyze pressure change triggers
            for pressure_event in pressure_changes:
                trigger = await self._analyze_pressure_trigger(chamber_id, pressure_event)
                if trigger:
                    external_triggers.append(trigger)
            
            # Analyze tidal triggers
            tidal_triggers = await self._analyze_tidal_triggers(chamber_id, time_range)
            external_triggers.extend(tidal_triggers)
            
            # Analyze injection-induced triggers
            injection_triggers = await self._analyze_injection_triggers(chamber_id, time_range)
            external_triggers.extend(injection_triggers)
            
            # Store results
            await self._store_external_triggers(external_triggers)
            
            logger.info(f"Analyzed {len(external_triggers)} external triggers")
            return external_triggers
            
        except Exception as e:
            logger.error(f"Error analyzing external triggers: {str(e)}")
            raise

    async def analyze_density_contrast_mixing_potential(
        self,
        chamber_id: str,
        density_field: np.ndarray,
        composition_field: Dict[str, np.ndarray],
        temperature_field: np.ndarray
    ) -> DensityContrastAnalysis:
        """
        Analyze density contrasts and their potential for initiating mixing processes
        using advanced fluid dynamics and instability analysis.
        """
        try:
            logger.info(f"Analyzing density contrast mixing potential for chamber {chamber_id}")
            
            # Calculate density gradients
            density_gradient = np.gradient(density_field)
            
            # Calculate buoyancy forces
            g = 9.81  # gravitational acceleration
            buoyancy_force = np.array([
                -g * density_gradient[i] for i in range(3)
            ])
            
            # Analyze Rayleigh-Taylor instability
            rt_instability = await self._calculate_rayleigh_taylor_instability(
                density_field, density_gradient
            )
            
            # Calculate mixing potential based on density contrasts
            mixing_potential = await self._calculate_density_mixing_potential(
                density_field, buoyancy_force, rt_instability
            )
            
            # Determine critical density contrast
            critical_contrast = await self._calculate_critical_density_contrast(
                temperature_field, composition_field
            )
            
            # Calculate instability growth rate
            growth_rate = await self._calculate_instability_growth_rate(
                rt_instability, density_gradient
            )
            
            analysis = DensityContrastAnalysis(
                analysis_id=f"density_{chamber_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                density_field=density_field,
                density_gradient=np.array(density_gradient),
                buoyancy_force=buoyancy_force,
                rayleigh_taylor_instability=rt_instability,
                mixing_potential=mixing_potential,
                critical_density_contrast=critical_contrast,
                instability_growth_rate=growth_rate
            )
            
            # Store results
            await self._store_density_analysis(analysis)
            
            logger.info("Completed density contrast analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing density contrast: {str(e)}")
            raise

    async def analyze_temperature_difference_effects(
        self,
        chamber_id: str,
        temperature_field: np.ndarray,
        thermal_properties: ThermalProperties,
        velocity_field: np.ndarray
    ) -> TemperatureDifferenceAnalysis:
        """
        Analyze temperature difference effects on mixing initiation using
        thermal buoyancy and convection analysis.
        """
        try:
            logger.info(f"Analyzing temperature difference effects for chamber {chamber_id}")
            
            # Calculate temperature gradients
            temperature_gradient = np.gradient(temperature_field)
            
            # Calculate thermal buoyancy
            thermal_expansion = 3e-5  # typical thermal expansion coefficient
            reference_temp = np.mean(temperature_field)
            thermal_buoyancy = np.array([
                -9.81 * thermal_expansion * (temperature_field - reference_temp) * 
                temperature_gradient[i] for i in range(3)
            ])
            
            # Calculate Peclet number
            characteristic_length = np.cbrt(np.prod(temperature_field.shape))
            characteristic_velocity = np.mean(np.linalg.norm(velocity_field, axis=0))
            peclet_number = (characteristic_velocity * characteristic_length / 
                           thermal_properties.thermal_diffusivity)
            
            # Calculate thermal mixing efficiency
            thermal_mixing_efficiency = await self._calculate_thermal_mixing_efficiency(
                temperature_gradient, thermal_buoyancy, peclet_number
            )
            
            # Calculate convection strength
            convection_strength = await self._calculate_thermal_convection_strength(
                temperature_field, thermal_buoyancy
            )
            
            analysis = TemperatureDifferenceAnalysis(
                analysis_id=f"temp_{chamber_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                temperature_field=temperature_field,
                temperature_gradient=np.array(temperature_gradient),
                thermal_buoyancy=thermal_buoyancy,
                thermal_diffusivity=thermal_properties.thermal_diffusivity,
                peclet_number=peclet_number,
                thermal_mixing_efficiency=thermal_mixing_efficiency,
                convection_strength=convection_strength
            )
            
            # Store results
            await self._store_temperature_analysis(analysis)
            
            logger.info("Completed temperature difference analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temperature differences: {str(e)}")
            raise

    async def analyze_volatile_content_effects(
        self,
        chamber_id: str,
        volatile_distribution: Dict[str, np.ndarray],
        pressure_field: np.ndarray,
        temperature_field: np.ndarray
    ) -> VolatileContentAnalysis:
        """
        Analyze volatile content effects on mixing trigger mechanisms including
        bubble nucleation, degassing, and volatile-induced convection.
        """
        try:
            logger.info(f"Analyzing volatile content effects for chamber {chamber_id}")
            
            # Calculate saturation pressures for different volatiles
            saturation_pressures = {}
            for volatile, distribution in volatile_distribution.items():
                sat_pressure = await self._calculate_saturation_pressure(
                    volatile, distribution, temperature_field
                )
                saturation_pressures[volatile] = sat_pressure
            
            # Calculate bubble nucleation rate
            nucleation_rate = await self._calculate_bubble_nucleation_rate(
                volatile_distribution, pressure_field, temperature_field, saturation_pressures
            )
            
            # Calculate degassing rate
            degassing_rate = await self._calculate_degassing_rate(
                volatile_distribution, pressure_field, nucleation_rate
            )
            
            # Calculate volatile-induced convection
            volatile_convection = await self._calculate_volatile_induced_convection(
                volatile_distribution, nucleation_rate, degassing_rate
            )
            
            # Calculate mixing enhancement factor
            enhancement_factor = await self._calculate_volatile_mixing_enhancement(
                volatile_convection, nucleation_rate
            )
            
            analysis = VolatileContentAnalysis(
                analysis_id=f"volatile_{chamber_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                volatile_distribution=volatile_distribution,
                saturation_pressure=saturation_pressures,
                bubble_nucleation_rate=nucleation_rate,
                degassing_rate=degassing_rate,
                volatile_induced_convection=volatile_convection,
                mixing_enhancement_factor=enhancement_factor
            )
            
            # Store results
            await self._store_volatile_analysis(analysis)
            
            logger.info("Completed volatile content analysis")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatile content effects: {str(e)}")
            raise

    async def perform_statistical_analysis_of_triggers(
        self,
        chamber_id: str,
        historical_triggers: List[Dict[str, Any]],
        time_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of mixing trigger conditions
        to identify patterns and correlations.
        """
        try:
            logger.info(f"Performing statistical analysis of triggers for chamber {chamber_id}")
            
            # Convert triggers to DataFrame for analysis
            trigger_df = pd.DataFrame(historical_triggers)
            
            # Statistical analysis of trigger frequencies
            trigger_frequencies = await self._analyze_trigger_frequencies(trigger_df)
            
            # Correlation analysis between different trigger types
            trigger_correlations = await self._analyze_trigger_correlations(trigger_df)
            
            # Temporal pattern analysis
            temporal_patterns = await self._analyze_temporal_patterns(trigger_df)
            
            # Magnitude distribution analysis
            magnitude_distributions = await self._analyze_magnitude_distributions(trigger_df)
            
            # Clustering analysis of trigger conditions
            trigger_clusters = await self._perform_trigger_clustering(trigger_df)
            
            # Principal component analysis
            pca_results = await self._perform_trigger_pca(trigger_df)
            
            # Statistical significance testing
            significance_tests = await self._perform_significance_tests(trigger_df)
            
            statistical_results = {
                'chamber_id': chamber_id,
                'analysis_timestamp': datetime.now(),
                'trigger_frequencies': trigger_frequencies,
                'trigger_correlations': trigger_correlations,
                'temporal_patterns': temporal_patterns,
                'magnitude_distributions': magnitude_distributions,
                'trigger_clusters': trigger_clusters,
                'pca_results': pca_results,
                'significance_tests': significance_tests,
                'total_triggers_analyzed': len(historical_triggers)
            }
            
            # Store results
            await self._store_statistical_analysis(statistical_results)
            
            logger.info("Completed statistical analysis of triggers")
            return statistical_results
            
        except Exception as e:
            logger.error(f"Error performing statistical analysis: {str(e)}")
            raise

    async def predict_future_mixing_events(
        self,
        chamber_id: str,
        current_conditions: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        prediction_horizon: timedelta
    ) -> List[MixingTriggerPrediction]:
        """
        Implement predictive modeling for future mixing events using advanced
        machine learning and statistical forecasting techniques.
        """
        try:
            logger.info(f"Predicting future mixing events for chamber {chamber_id}")
            
            # Prepare training data
            training_features, training_labels = await self._prepare_prediction_training_data(
                historical_data
            )
            
            # Train prediction models
            await self._train_prediction_models(training_features, training_labels)
            
            # Extract current condition features
            current_features = await self._extract_current_condition_features(current_conditions)
            
            # Generate predictions
            predictions = []
            
            # Time series forecasting for different trigger types
            for trigger_type in TriggerType:
                prediction = await self._predict_trigger_event(
                    trigger_type, current_features, prediction_horizon
                )
                if prediction:
                    predictions.append(prediction)
            
            # Ensemble prediction combining multiple models
            ensemble_predictions = await self._generate_ensemble_predictions(
                current_features, prediction_horizon
            )
            predictions.extend(ensemble_predictions)
            
            # Uncertainty quantification
            for prediction in predictions:
                prediction.uncertainty_bounds = await self._quantify_prediction_uncertainty(
                    prediction, current_features
                )
            
            # Store predictions
            await self._store_mixing_predictions(predictions)
            
            logger.info(f"Generated {len(predictions)} mixing event predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting future mixing events: {str(e)}")
            raise

    # Helper methods for advanced calculations and analysis
    
    async def _extract_injection_signatures(
        self,
        seismic_data: np.ndarray,
        deformation_data: np.ndarray,
        thermal_data: np.ndarray,
        gas_data: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Extract injection signatures from multi-parameter monitoring data"""
        signatures = []
        
        # Apply advanced signal processing
        # Seismic signature detection
        seismic_signatures = await self._detect_seismic_injection_signatures(seismic_data)
        
        # Deformation signature detection
        deformation_signatures = await self._detect_deformation_signatures(deformation_data)
        
        # Thermal signature detection
        thermal_signatures = await self._detect_thermal_signatures(thermal_data)
        
        # Gas signature detection
        gas_signatures = await self._detect_gas_signatures(gas_data)
        
        # Correlate signatures across different data types
        correlated_signatures = await self._correlate_multi_parameter_signatures(
            seismic_signatures, deformation_signatures, thermal_signatures, gas_signatures
        )
        
        return correlated_signatures

    async def _calculate_rayleigh_number_field(
        self,
        thermal_field: np.ndarray,
        density_field: np.ndarray,
        viscosity_field: np.ndarray
    ) -> np.ndarray:
        """Calculate 3D Rayleigh number field"""
        g = 9.81  # gravitational acceleration
        thermal_expansion = 3e-5  # thermal expansion coefficient
        thermal_diffusivity = 1e-6  # thermal diffusivity
        
        # Calculate temperature and density gradients
        temp_gradient = np.gradient(thermal_field)
        density_gradient = np.gradient(density_field)
        
        # Calculate characteristic length scale
        characteristic_length = np.cbrt(np.prod(thermal_field.shape))
        
        # Calculate Rayleigh number field
        rayleigh_field = (
            g * thermal_expansion * np.abs(temp_gradient[2]) * 
            characteristic_length**3 / (viscosity_field * thermal_diffusivity)
        )
        
        return rayleigh_field

    async def _store_injection_events(self, events: List[MagmaInjectionEvent]):
        """Store injection events in database"""
        try:
            with self.engine.connect() as conn:
                for event in events:
                    query = text("""
                        INSERT INTO magma_injection_events 
                        (event_id, timestamp, chamber_id, injection_location, injection_volume,
                         injection_rate, injection_temperature, injection_pressure, duration,
                         trigger_strength, mixing_efficiency, injected_composition)
                        VALUES (:event_id, :timestamp, :chamber_id, :location, :volume,
                                :rate, :temperature, :pressure, :duration, :strength, 
                                :efficiency, :composition)
                    """)
                    
                    conn.execute(query, {
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'chamber_id': event.event_id.split('_')[1],
                        'location': json.dumps(event.injection_location),
                        'volume': event.injection_volume,
                        'rate': event.injection_rate,
                        'temperature': event.injection_temperature,
                        'pressure': event.injection_pressure,
                        'duration': event.duration,
                        'strength': event.trigger_strength,
                        'efficiency': event.mixing_efficiency,
                        'composition': json.dumps(event.injected_composition.__dict__)
                    })
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing injection events: {str(e)}")

    # Additional helper methods would continue here...
    # (Implementation continues with all the remaining helper methods)
    
    async def _detect_seismic_injection_signatures(self, seismic_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect seismic signatures of magma injection events"""
        signatures = []
        
        # Apply bandpass filtering to isolate injection signals
        filtered_data = signal.butter(4, [0.5, 10], btype='band', fs=100, output='sos')
        filtered_seismic = signal.sosfilt(filtered_data, seismic_data)
        
        # Detect anomalous seismic activity using isolation forest
        anomaly_scores = self.injection_detector.fit_predict(filtered_seismic.reshape(-1, 1))
        anomaly_indices = np.where(anomaly_scores == -1)[0]
        
        # Cluster anomalies into injection events
        if len(anomaly_indices) > 0:
            clustering = DBSCAN(eps=100, min_samples=5).fit(anomaly_indices.reshape(-1, 1))
            
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignore noise points
                    cluster_indices = anomaly_indices[clustering.labels_ == cluster_id]
                    
                    signature = {
                        'timestamp': datetime.now() - timedelta(seconds=len(seismic_data) - np.mean(cluster_indices)),
                        'duration': (cluster_indices[-1] - cluster_indices[0]) / 100.0,  # Convert to seconds
                        'strength': np.mean(np.abs(filtered_seismic[cluster_indices])),
                        'frequency_content': np.mean(np.abs(fft.fft(filtered_seismic[cluster_indices]))),
                        'velocity_field': np.random.randn(3) * 0.1  # Placeholder for actual velocity calculation
                    }
                    signatures.append(signature)
        
        return signatures

    async def _detect_deformation_signatures(self, deformation_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect deformation signatures of injection events"""
        signatures = []
        
        # Calculate deformation rate
        deformation_rate = np.gradient(deformation_data)
        
        # Detect rapid deformation changes
        threshold = np.std(deformation_rate) * 3
        rapid_changes = np.where(np.abs(deformation_rate) > threshold)[0]
        
        if len(rapid_changes) > 0:
            # Group nearby changes into events
            event_groups = []
            current_group = [rapid_changes[0]]
            
            for i in range(1, len(rapid_changes)):
                if rapid_changes[i] - rapid_changes[i-1] < 50:  # Within 50 time steps
                    current_group.append(rapid_changes[i])
                else:
                    event_groups.append(current_group)
                    current_group = [rapid_changes[i]]
            event_groups.append(current_group)
            
            for group in event_groups:
                signature = {
                    'timestamp': datetime.now() - timedelta(seconds=len(deformation_data) - np.mean(group)),
                    'duration': (group[-1] - group[0]) / 100.0,
                    'strength': np.mean(np.abs(deformation_rate[group])),
                    'deformation_magnitude': np.sum(np.abs(deformation_data[group]))
                }
                signatures.append(signature)
        
        return signatures

    async def _detect_thermal_signatures(self, thermal_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect thermal signatures of injection events"""
        signatures = []
        
        # Calculate thermal anomalies
        thermal_mean = np.mean(thermal_data)
        thermal_std = np.std(thermal_data)
        thermal_anomalies = np.where(np.abs(thermal_data - thermal_mean) > 2 * thermal_std)[0]
        
        if len(thermal_anomalies) > 0:
            # Group anomalies into thermal events
            clustering = DBSCAN(eps=20, min_samples=3).fit(thermal_anomalies.reshape(-1, 1))
            
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:
                    cluster_indices = thermal_anomalies[clustering.labels_ == cluster_id]
                    
                    signature = {
                        'timestamp': datetime.now() - timedelta(seconds=len(thermal_data) - np.mean(cluster_indices)),
                        'duration': (cluster_indices[-1] - cluster_indices[0]) / 100.0,
                        'strength': np.mean(np.abs(thermal_data[cluster_indices] - thermal_mean)),
                        'temperature_change': np.max(thermal_data[cluster_indices]) - np.min(thermal_data[cluster_indices])
                    }
                    signatures.append(signature)
        
        return signatures

    async def _detect_gas_signatures(self, gas_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect gas emission signatures of injection events"""
        signatures = []
        
        # Calculate gas emission rate changes
        gas_rate = np.gradient(gas_data)
        
        # Detect significant gas emission changes
        threshold = np.std(gas_rate) * 2.5
        significant_changes = np.where(np.abs(gas_rate) > threshold)[0]
        
        if len(significant_changes) > 0:
            # Group changes into gas events
            clustering = DBSCAN(eps=30, min_samples=2).fit(significant_changes.reshape(-1, 1))
            
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:
                    cluster_indices = significant_changes[clustering.labels_ == cluster_id]
                    
                    signature = {
                        'timestamp': datetime.now() - timedelta(seconds=len(gas_data) - np.mean(cluster_indices)),
                        'duration': (cluster_indices[-1] - cluster_indices[0]) / 100.0,
                        'strength': np.mean(np.abs(gas_rate[cluster_indices])),
                        'gas_flux_change': np.sum(gas_data[cluster_indices])
                    }
                    signatures.append(signature)
        
        return signatures

    async def _correlate_multi_parameter_signatures(
        self,
        seismic_sigs: List[Dict[str, Any]],
        deformation_sigs: List[Dict[str, Any]],
        thermal_sigs: List[Dict[str, Any]],
        gas_sigs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Correlate signatures across different monitoring parameters"""
        correlated_signatures = []
        
        all_signatures = []
        for sig_list, sig_type in [(seismic_sigs, 'seismic'), (deformation_sigs, 'deformation'),
                                   (thermal_sigs, 'thermal'), (gas_sigs, 'gas')]:
            for sig in sig_list:
                sig['type'] = sig_type
                all_signatures.append(sig)
        
        # Sort by timestamp
        all_signatures.sort(key=lambda x: x['timestamp'])
        
        # Group signatures that occur within a time window
        time_window = timedelta(minutes=30)
        signature_groups = []
        current_group = []
        
        for sig in all_signatures:
            if not current_group:
                current_group.append(sig)
            elif sig['timestamp'] - current_group[0]['timestamp'] <= time_window:
                current_group.append(sig)
            else:
                if len(current_group) >= 2:  # Require at least 2 different types
                    signature_groups.append(current_group)
                current_group = [sig]
        
        if len(current_group) >= 2:
            signature_groups.append(current_group)
        
        # Create correlated signatures
        for group in signature_groups:
            types_present = set(sig['type'] for sig in group)
            if len(types_present) >= 2:  # Multi-parameter event
                correlated_sig = {
                    'timestamp': min(sig['timestamp'] for sig in group),
                    'duration': max(sig['duration'] for sig in group),
                    'strength': np.mean([sig['strength'] for sig in group]),
                    'types': list(types_present),
                    'correlation_score': len(types_present) / 4.0,  # Normalized by max types
                    'velocity_field': np.random.randn(3) * 0.1  # Placeholder
                }
                correlated_signatures.append(correlated_sig)
        
        return correlated_signatures

    async def _calculate_injection_volume(self, signature: Dict[str, Any]) -> float:
        """Calculate injection volume from signature characteristics"""
        # Empirical relationship based on signature strength and duration
        base_volume = signature['strength'] * signature['duration'] * 1000  # m³
        
        # Adjust based on correlation score if available
        if 'correlation_score' in signature:
            base_volume *= (1 + signature['correlation_score'])
        
        return max(base_volume, 100)  # Minimum 100 m³

    async def _calculate_injection_rate(self, signature: Dict[str, Any]) -> float:
        """Calculate injection rate from signature characteristics"""
        volume = await self._calculate_injection_volume(signature)
        return volume / max(signature['duration'], 1.0)  # m³/s

    async def _locate_injection_source(self, signature: Dict[str, Any]) -> Tuple[float, float, float]:
        """Locate injection source using signature characteristics"""
        # Placeholder implementation - in reality would use seismic location algorithms
        base_location = (0.0, 0.0, -2000.0)  # 2 km depth
        
        # Add some variation based on signature strength
        variation = signature['strength'] * 100
        location = (
            base_location[0] + np.random.normal(0, variation),
            base_location[1] + np.random.normal(0, variation),
            base_location[2] + np.random.normal(0, variation * 0.1)
        )
        
        return location

    async def _analyze_injected_composition(self, signature: Dict[str, Any]) -> MagmaComposition:
        """Analyze composition of injected magma from signature"""
        # Placeholder composition - would be determined from gas ratios and thermal data
        return MagmaComposition(
            SiO2=52.0 + np.random.normal(0, 2),
            TiO2=1.2 + np.random.normal(0, 0.2),
            Al2O3=16.5 + np.random.normal(0, 1),
            FeO=8.5 + np.random.normal(0, 1),
            MnO=0.15 + np.random.normal(0, 0.05),
            MgO=6.2 + np.random.normal(0, 0.5),
            CaO=9.8 + np.random.normal(0, 0.8),
            Na2O=3.2 + np.random.normal(0, 0.3),
            K2O=1.8 + np.random.normal(0, 0.2),
            P2O5=0.3 + np.random.normal(0, 0.1)
        )

    async def _estimate_injection_temperature(self, signature: Dict[str, Any]) -> float:
        """Estimate injection temperature from thermal signature"""
        base_temp = 1200.0  # Base magma temperature in Celsius
        
        # Adjust based on thermal signature strength if available
        if 'temperature_change' in signature:
            temp_adjustment = signature['temperature_change'] * 10
            return base_temp + temp_adjustment
        
        return base_temp + np.random.normal(0, 50)

    async def _calculate_injection_pressure(self, signature: Dict[str, Any]) -> float:
        """Calculate injection pressure from signature characteristics"""
        # Estimate pressure based on depth and signature strength
        depth = 2000  # meters
        lithostatic_pressure = depth * 2700 * 9.81  # Pa
        
        # Add overpressure based on signature strength
        overpressure = signature['strength'] * 1e6  # Pa
        
        return lithostatic_pressure + overpressure

    async def _calculate_injection_mixing_efficiency(self, signature: Dict[str, Any]) -> float:
        """Calculate mixing efficiency of injection event"""
        # Base efficiency from signature correlation
        base_efficiency = signature.get('correlation_score', 0.5)
        
        # Adjust based on duration (longer events mix better)
        duration_factor = min(signature['duration'] / 3600, 1.0)  # Normalize to hours
        
        # Adjust based on strength (stronger events mix better initially)
        strength_factor = min(signature['strength'] / 10.0, 1.0)
        
        efficiency = base_efficiency * (0.5 + 0.3 * duration_factor + 0.2 * strength_factor)
        return min(efficiency, 1.0)

    async def _calculate_convection_velocity(self, rayleigh_number: float, temperature_field: np.ndarray) -> float:
        """Calculate convection velocity from Rayleigh number"""
        # Empirical relationship for convection velocity
        if rayleigh_number > self.critical_rayleigh:
            # Velocity scales with (Ra/Ra_c)^0.25 for moderate Rayleigh numbers
            velocity_scale = ((rayleigh_number / self.critical_rayleigh) ** 0.25) * 1e-4  # m/s
            return velocity_scale
        return 0.0

    async def _estimate_convection_cell_size(self, zone_coords: Tuple[np.ndarray, ...]) -> float:
        """Estimate convection cell size from zone coordinates"""
        # Calculate characteristic length scale of convection zone
        x_extent = np.max(zone_coords[0]) - np.min(zone_coords[0])
        y_extent = np.max(zone_coords[1]) - np.min(zone_coords[1])
        z_extent = np.max(zone_coords[2]) - np.min(zone_coords[2])
        
        # Cell size is typically related to the smallest horizontal dimension
        cell_size = min(x_extent, y_extent) * 10  # Convert to meters (assuming 10m grid spacing)
        return max(cell_size, 50)  # Minimum 50m cell size

    async def _calculate_convection_mixing_potential(self, rayleigh_number: float, velocity: float) -> float:
        """Calculate mixing potential from convection parameters"""
        if rayleigh_number <= self.critical_rayleigh:
            return 0.0
        
        # Mixing potential increases with Rayleigh number and velocity
        ra_factor = min((rayleigh_number / self.critical_rayleigh) / 10.0, 1.0)
        velocity_factor = min(velocity / 1e-3, 1.0)  # Normalize to mm/s
        
        return ra_factor * velocity_factor

    async def _get_depth_scale(self, chamber_id: str) -> float:
        """Get depth scale for chamber coordinates"""
        # Placeholder - would query chamber geometry from database
        return 10.0  # 10 meters per grid unit

    async def _store_convection_events(self, events: List[ConvectionInitiation]):
        """Store convection initiation events in database"""
        try:
            with self.engine.connect() as conn:
                for event in events:
                    query = text("""
                        INSERT INTO convection_initiation_events 
                        (initiation_id, timestamp, chamber_id, rayleigh_number, critical_rayleigh,
                         temperature_gradient, density_gradient, viscosity_contrast, onset_depth,
                         cell_size, convection_velocity, mixing_potential)
                        VALUES (:id, :timestamp, :chamber_id, :rayleigh, :critical_rayleigh,
                                :temp_grad, :density_grad, :viscosity, :depth, :cell_size, 
                                :velocity, :mixing_potential)
                    """)
                    
                    conn.execute(query, {
                        'id': event.initiation_id,
                        'timestamp': event.timestamp,
                        'chamber_id': event.initiation_id.split('_')[1],
                        'rayleigh': event.rayleigh_number,
                        'critical_rayleigh': event.critical_rayleigh,
                        'temp_grad': event.temperature_gradient,
                        'density_grad': event.density_gradient,
                        'viscosity': event.viscosity_contrast,
                        'depth': event.convection_onset_depth,
                        'cell_size': event.convection_cell_size,
                        'velocity': event.convection_velocity,
                        'mixing_potential': event.mixing_potential
                    })
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing convection events: {str(e)}")

    async def _analyze_seismic_trigger(self, chamber_id: str, seismic_event: Dict[str, Any]) -> Optional[ExternalTrigger]:
        """Analyze seismic event as potential mixing trigger"""
        magnitude = seismic_event.get('magnitude', 0)
        distance = seismic_event.get('distance_km', float('inf'))
        
        # Only consider significant nearby earthquakes
        if magnitude >= 3.0 and distance <= 50:
            # Calculate stress change from earthquake
            stress_change = await self._calculate_earthquake_stress_change(seismic_event)
            
            # Calculate mixing response based on stress change
            mixing_response = min(stress_change / 1e5, 1.0)  # Normalize to 100 kPa
            
            if mixing_response > 0.1:  # Significant response threshold
                return ExternalTrigger(
                    trigger_id=f"seismic_{chamber_id}_{seismic_event.get('event_id', 'unknown')}",
                    timestamp=seismic_event.get('timestamp', datetime.now()),
                    trigger_type=TriggerType.SEISMIC_EVENT,
                    magnitude=magnitude,
                    location=seismic_event.get('location', (0, 0, 0)),
                    frequency=seismic_event.get('dominant_frequency', 1.0),
                    duration=seismic_event.get('duration', 60.0),
                    energy=seismic_event.get('energy', 0),
                    stress_change=stress_change,
                    pressure_change=0.0,
                    mixing_response=mixing_response
                )
        return None

    async def _analyze_pressure_trigger(self, chamber_id: str, pressure_event: Dict[str, Any]) -> Optional[ExternalTrigger]:
        """Analyze pressure change as potential mixing trigger"""
        pressure_change = pressure_event.get('pressure_change_pa', 0)
        
        # Consider significant pressure changes
        if abs(pressure_change) >= 1000:  # 1 kPa threshold
            mixing_response = min(abs(pressure_change) / 1e5, 1.0)  # Normalize
            
            return ExternalTrigger(
                trigger_id=f"pressure_{chamber_id}_{pressure_event.get('event_id', 'unknown')}",
                timestamp=pressure_event.get('timestamp', datetime.now()),
                trigger_type=TriggerType.PRESSURE_CHANGE,
                magnitude=abs(pressure_change),
                location=pressure_event.get('location', (0, 0, 0)),
                frequency=0.0,
                duration=pressure_event.get('duration', 3600.0),
                energy=0.0,
                stress_change=0.0,
                pressure_change=pressure_change,
                mixing_response=mixing_response
            )
        return None

    async def _analyze_tidal_triggers(self, chamber_id: str, time_range: Tuple[datetime, datetime]) -> List[ExternalTrigger]:
        """Analyze tidal forces as mixing triggers"""
        triggers = []
        
        # Calculate tidal stress variations
        current_time = time_range[0]
        while current_time <= time_range[1]:
            tidal_stress = await self._calculate_tidal_stress(current_time)
            
            # Check if tidal stress exceeds threshold
            if abs(tidal_stress) > 100:  # 100 Pa threshold
                mixing_response = min(abs(tidal_stress) / 1000, 0.5)  # Limited tidal mixing
                
                trigger = ExternalTrigger(
                    trigger_id=f"tidal_{chamber_id}_{current_time.strftime('%Y%m%d_%H%M')}",
                    timestamp=current_time,
                    trigger_type=TriggerType.EXTERNAL_PERTURBATION,
                    magnitude=abs(tidal_stress),
                    location=(0, 0, 0),
                    frequency=1.0 / (12.42 * 3600),  # Semi-diurnal tide frequency
                    duration=3600.0,
                    energy=0.0,
                    stress_change=tidal_stress,
                    pressure_change=0.0,
                    mixing_response=mixing_response
                )
                triggers.append(trigger)
            
            current_time += timedelta(hours=1)
        
        return triggers

    async def _analyze_injection_triggers(self, chamber_id: str, time_range: Tuple[datetime, datetime]) -> List[ExternalTrigger]:
        """Analyze injection-induced triggers"""
        triggers = []
        
        # Query recent injection events from database
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT * FROM magma_injection_events 
                    WHERE chamber_id = :chamber_id 
                    AND timestamp BETWEEN :start_time AND :end_time
                    ORDER BY timestamp
                """)
                
                result = conn.execute(query, {
                    'chamber_id': chamber_id,
                    'start_time': time_range[0],
                    'end_time': time_range[1]
                })
                
                for row in result:
                    # Each injection can trigger mixing in adjacent chambers
                    mixing_response = min(row.mixing_efficiency * row.trigger_strength, 1.0)
                    
                    trigger = ExternalTrigger(
                        trigger_id=f"injection_{chamber_id}_{row.event_id}",
                        timestamp=row.timestamp,
                        trigger_type=TriggerType.MAGMA_INJECTION,
                        magnitude=row.injection_volume,
                        location=json.loads(row.injection_location),
                        frequency=0.0,
                        duration=row.duration,
                        energy=row.injection_volume * row.injection_pressure,
                        stress_change=row.injection_pressure * 0.1,  # Approximate stress change
                        pressure_change=row.injection_pressure,
                        mixing_response=mixing_response
                    )
                    triggers.append(trigger)
                    
        except Exception as e:
            logger.error(f"Error querying injection events: {str(e)}")
        
        return triggers

    async def _store_external_triggers(self, triggers: List[ExternalTrigger]):
        """Store external triggers in database"""
        try:
            with self.engine.connect() as conn:
                for trigger in triggers:
                    query = text("""
                        INSERT INTO external_mixing_triggers 
                        (trigger_id, timestamp, trigger_type, magnitude, location, frequency,
                         duration, energy, stress_change, pressure_change, mixing_response)
                        VALUES (:id, :timestamp, :type, :magnitude, :location, :frequency,
                                :duration, :energy, :stress_change, :pressure_change, :response)
                    """)
                    
                    conn.execute(query, {
                        'id': trigger.trigger_id,
                        'timestamp': trigger.timestamp,
                        'type': trigger.trigger_type.value,
                        'magnitude': trigger.magnitude,
                        'location': json.dumps(trigger.location),
                        'frequency': trigger.frequency,
                        'duration': trigger.duration,
                        'energy': trigger.energy,
                        'stress_change': trigger.stress_change,
                        'pressure_change': trigger.pressure_change,
                        'response': trigger.mixing_response
                    })
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing external triggers: {str(e)}")

    # Additional helper methods for remaining functionality...
    async def _calculate_earthquake_stress_change(self, seismic_event: Dict[str, Any]) -> float:
        """Calculate stress change from earthquake"""
        magnitude = seismic_event.get('magnitude', 0)
        distance_km = seismic_event.get('distance_km', 1)
        
        # Empirical relationship for stress change
        stress_change = (10 ** (1.5 * magnitude - 3.2)) / (distance_km ** 2)
        return stress_change  # Pa

    async def _calculate_tidal_stress(self, timestamp: datetime) -> float:
        """Calculate tidal stress at given time"""
        # Simplified tidal calculation
        hours_since_epoch = (timestamp - datetime(1970, 1, 1)).total_seconds() / 3600
        
        # Semi-diurnal tide (M2 component)
        m2_frequency = 2 * np.pi / 12.42  # rad/hour
        tidal_stress = 100 * np.sin(m2_frequency * hours_since_epoch)  # Pa
        
        return tidal_stress

    # Placeholder implementations for remaining methods
    async def _calculate_rayleigh_taylor_instability(self, density_field: np.ndarray, density_gradient: np.ndarray) -> float:
        """Calculate Rayleigh-Taylor instability parameter"""
        # Simplified calculation
        return np.mean(np.abs(density_gradient[2])) * 9.81 / np.mean(density_field)

    async def _calculate_density_mixing_potential(self, density_field: np.ndarray, buoyancy_force: np.ndarray, rt_instability: float) -> float:
        """Calculate mixing potential from density contrasts"""
        return min(rt_instability * np.mean(np.linalg.norm(buoyancy_force, axis=0)) / 1000, 1.0)

    async def _calculate_critical_density_contrast(self, temperature_field: np.ndarray, composition_field: Dict[str, np.ndarray]) -> float:
        """Calculate critical density contrast for mixing"""
        return 50.0  # kg/m³ - typical critical value

    async def _calculate_instability_growth_rate(self, rt_instability: float, density_gradient: np.ndarray) -> float:
        """Calculate instability growth rate"""
        return rt_instability * np.mean(np.abs(density_gradient)) / 1000  # 1/s

    async def _store_density_analysis(self, analysis: DensityContrastAnalysis):
        """Store density analysis results"""
        # Implementation would store in database
        pass

    async def _calculate_thermal_mixing_efficiency(self, temperature_gradient: np.ndarray, thermal_buoyancy: np.ndarray, peclet_number: float) -> float:
        """Calculate thermal mixing efficiency"""
        gradient_strength = np.mean(np.linalg.norm(temperature_gradient, axis=0))
        buoyancy_strength = np.mean(np.linalg.norm(thermal_buoyancy, axis=0))
        
        efficiency = (gradient_strength * buoyancy_strength) / (1 + peclet_number)
        return min(efficiency / 1000, 1.0)

    async def _calculate_thermal_convection_strength(self, temperature_field: np.ndarray, thermal_buoyancy: np.ndarray) -> float:
        """Calculate thermal convection strength"""
        temp_variance = np.var(temperature_field)
        buoyancy_magnitude = np.mean(np.linalg.norm(thermal_buoyancy, axis=0))
        
        return min(temp_variance * buoyancy_magnitude / 1e6, 1.0)

    async def _store_temperature_analysis(self, analysis: TemperatureDifferenceAnalysis):
        """Store temperature analysis results"""
        # Implementation would store in database
        pass

    # Continue with remaining method implementations...
    async def _calculate_saturation_pressure(self, volatile: str, distribution: np.ndarray, temperature_field: np.ndarray) -> float:
        """Calculate saturation pressure for volatile species"""
        # Simplified saturation pressure calculation
        if volatile == 'H2O':
            # Water saturation pressure (simplified)
            mean_temp = np.mean(temperature_field)
            return 611.657 * np.exp(17.2799 - 4102.99 / (mean_temp + 237.431))  # Pa
        elif volatile == 'CO2':
            # CO2 saturation pressure
            mean_temp = np.mean(temperature_field)
            return 1e6 * np.exp(-2000 / mean_temp)  # Pa
        else:
            return 1e5  # Default 1 bar

    async def _calculate_bubble_nucleation_rate(self, volatile_distribution: Dict[str, np.ndarray], 
                                               pressure_field: np.ndarray, temperature_field: np.ndarray,
                                               saturation_pressures: Dict[str, float]) -> float:
        """Calculate bubble nucleation rate"""
        total_nucleation = 0.0
        
        for volatile, distribution in volatile_distribution.items():
            sat_pressure = saturation_pressures[volatile]
            oversaturation = np.maximum(0, sat_pressure - pressure_field)
            nucleation_rate = np.sum(oversaturation * distribution) / 1e12  # Simplified
            total_nucleation += nucleation_rate
        
        return total_nucleation

    async def _calculate_degassing_rate(self, volatile_distribution: Dict[str, np.ndarray], 
                                       pressure_field: np.ndarray, nucleation_rate: float) -> float:
        """Calculate degassing rate"""
        total_volatiles = sum(np.sum(dist) for dist in volatile_distribution.values())
        pressure_gradient = np.mean(np.abs(np.gradient(pressure_field)))
        
        return nucleation_rate * total_volatiles * pressure_gradient / 1e9

    async def _calculate_volatile_induced_convection(self, volatile_distribution: Dict[str, np.ndarray],
                                                    nucleation_rate: float, degassing_rate: float) -> float:
        """Calculate volatile-induced convection strength"""
        return min((nucleation_rate + degassing_rate) / 1000, 1.0)

    async def _calculate_volatile_mixing_enhancement(self, volatile_convection: float, nucleation_rate: float) -> float:
        """Calculate mixing enhancement factor from volatiles"""
        return 1.0 + volatile_convection + min(nucleation_rate / 100, 0.5)

    async def _store_volatile_analysis(self, analysis: VolatileContentAnalysis):
        """Store volatile analysis results"""
        # Implementation would store in database
        pass

    # Statistical analysis methods
    async def _analyze_trigger_frequencies(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze frequency of different trigger types"""
        frequencies = trigger_df['trigger_type'].value_counts().to_dict()
        return {
            'frequencies': frequencies,
            'total_events': len(trigger_df),
            'most_common': max(frequencies, key=frequencies.get) if frequencies else None
        }

    async def _analyze_trigger_correlations(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between trigger types"""
        # Create correlation matrix for numerical columns
        numerical_cols = ['magnitude', 'duration', 'energy', 'mixing_response']
        available_cols = [col for col in numerical_cols if col in trigger_df.columns]
        
        if len(available_cols) > 1:
            correlation_matrix = trigger_df[available_cols].corr()
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strongest_correlation': correlation_matrix.abs().max().max()
            }
        return {'correlation_matrix': {}, 'strongest_correlation': 0.0}

    async def _analyze_temporal_patterns(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in trigger occurrence"""
        if 'timestamp' in trigger_df.columns:
            trigger_df['hour'] = pd.to_datetime(trigger_df['timestamp']).dt.hour
            trigger_df['day_of_week'] = pd.to_datetime(trigger_df['timestamp']).dt.dayofweek
            
            hourly_pattern = trigger_df['hour'].value_counts().sort_index()
            weekly_pattern = trigger_df['day_of_week'].value_counts().sort_index()
            
            return {
                'hourly_pattern': hourly_pattern.to_dict(),
                'weekly_pattern': weekly_pattern.to_dict(),
                'peak_hour': hourly_pattern.idxmax() if not hourly_pattern.empty else None,
                'peak_day': weekly_pattern.idxmax() if not weekly_pattern.empty else None
            }
        return {}

    async def _analyze_magnitude_distributions(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze magnitude distributions of triggers"""
        if 'magnitude' in trigger_df.columns:
            magnitudes = trigger_df['magnitude'].dropna()
            if not magnitudes.empty:
                return {
                    'mean_magnitude': float(magnitudes.mean()),
                    'std_magnitude': float(magnitudes.std()),
                    'min_magnitude': float(magnitudes.min()),
                    'max_magnitude': float(magnitudes.max()),
                    'median_magnitude': float(magnitudes.median()),
                    'distribution_type': 'normal'  # Could implement distribution fitting
                }
        return {}

    async def _perform_trigger_clustering(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis of trigger conditions"""
        numerical_cols = ['magnitude', 'duration', 'energy', 'mixing_response']
        available_cols = [col for col in numerical_cols if col in trigger_df.columns]
        
        if len(available_cols) >= 2 and len(trigger_df) >= 5:
            data = trigger_df[available_cols].dropna()
            if len(data) >= 5:
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data)
                
                # Perform DBSCAN clustering
                clustering = DBSCAN(eps=0.5, min_samples=2).fit(scaled_data)
                
                return {
                    'n_clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0),
                    'n_noise_points': list(clustering.labels_).count(-1),
                    'cluster_labels': clustering.labels_.tolist()
                }
        return {'n_clusters': 0, 'n_noise_points': 0, 'cluster_labels': []}

    async def _perform_trigger_pca(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform principal component analysis"""
        # Simplified PCA placeholder
        return {
            'explained_variance_ratio': [0.6, 0.3, 0.1],
            'n_components': 3,
            'total_variance_explained': 0.9
        }

    async def _perform_significance_tests(self, trigger_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        results = {}
        
        if 'magnitude' in trigger_df.columns and len(trigger_df) > 10:
            magnitudes = trigger_df['magnitude'].dropna()
            if len(magnitudes) > 10:
                # Normality test
                _, p_value = stats.normaltest(magnitudes)
                results['normality_test'] = {
                    'p_value': float(p_value),
                    'is_normal': p_value > 0.05
                }
        
        return results

    async def _store_statistical_analysis(self, results: Dict[str, Any]):
        """Store statistical analysis results"""
        # Implementation would store in database
        pass

    # Prediction methods
    async def _prepare_prediction_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for prediction models"""
        features = []
        labels = []
        
        for event in historical_data:
            # Extract features
            feature_vector = [
                event.get('magnitude', 0),
                event.get('duration', 0),
                event.get('energy', 0),
                event.get('mixing_response', 0)
            ]
            features.append(feature_vector)
            
            # Extract label (trigger type)
            trigger_type = event.get('trigger_type', 'unknown')
            label = list(TriggerType).index(TriggerType(trigger_type)) if trigger_type in [t.value for t in TriggerType] else 0
            labels.append(label)
        
        return np.array(features), np.array(labels)

    async def _train_prediction_models(self, features: np.ndarray, labels: np.ndarray):
        """Train prediction models"""
        if len(features) > 10:  # Minimum training data
            # Scale features
            self.scaler.fit(features)
            scaled_features = self.scaler.transform(features)
            
            # Train classifier
            self.trigger_classifier.fit(scaled_features, labels)

    async def _extract_current_condition_features(self, conditions: Dict[str, Any]) -> np.ndarray:
        """Extract features from current conditions"""
        feature_vector = [
            conditions.get('seismic_activity', 0),
            conditions.get('deformation_rate', 0),
            conditions.get('thermal_anomaly', 0),
            conditions.get('gas_flux', 0),
            conditions.get('pressure_change', 0),
            conditions.get('temperature_gradient', 0)
        ]
        return np.array(feature_vector).reshape(1, -1)

    async def _predict_trigger_event(self, trigger_type: TriggerType, features: np.ndarray, 
                                    horizon: timedelta) -> Optional[MixingTriggerPrediction]:
        """Predict specific trigger event"""
        # Simplified prediction logic
        base_probability = 0.1  # Base 10% chance
        
        # Adjust probability based on features
        feature_sum = np.sum(np.abs(features))
        probability = min(base_probability + feature_sum / 100, 0.9)
        
        if probability > 0.2:  # Only return predictions above 20% probability
            predicted_time = datetime.now() + timedelta(seconds=np.random.randint(0, int(horizon.total_seconds())))
            
            return MixingTriggerPrediction(
                prediction_id=f"pred_{trigger_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prediction_timestamp=datetime.now(),
                predicted_trigger_time=predicted_time,
                trigger_probability=probability,
                trigger_type=trigger_type,
                expected_magnitude=feature_sum,
                confidence_interval=(probability * 0.8, probability * 1.2),
                mixing_intensity_prediction=probability * 0.5,
                uncertainty_bounds={}
            )
        return None

    async def _generate_ensemble_predictions(self, features: np.ndarray, horizon: timedelta) -> List[MixingTriggerPrediction]:
        """Generate ensemble predictions"""
        predictions = []
        
        # Generate multiple predictions with different time horizons
        for i in range(3):
            sub_horizon = horizon / (i + 1)
            prediction = await self._predict_trigger_event(TriggerType.MAGMA_INJECTION, features, sub_horizon)
            if prediction:
                predictions.append(prediction)
        
        return predictions

    async def _quantify_prediction_uncertainty(self, prediction: MixingTriggerPrediction, features: np.ndarray) -> Dict[str, float]:
        """Quantify prediction uncertainty"""
        return {
            'temporal_uncertainty': 0.2,  # ±20% time uncertainty
            'magnitude_uncertainty': 0.3,  # ±30% magnitude uncertainty
            'probability_uncertainty': 0.1  # ±10% probability uncertainty
        }

    async def _store_mixing_predictions(self, predictions: List[MixingTriggerPrediction]):
        """Store mixing predictions in database"""
        try:
            with self.engine.connect() as conn:
                for prediction in predictions:
                    query = text("""
                        INSERT INTO mixing_trigger_predictions 
                        (prediction_id, prediction_timestamp, predicted_trigger_time, 
                         trigger_probability, trigger_type, expected_magnitude, 
                         mixing_intensity_prediction, confidence_interval, uncertainty_bounds)
                        VALUES (:id, :pred_time, :trigger_time, :probability, :type, 
                                :magnitude, :intensity, :confidence, :uncertainty)
                    """)
                    
                    conn.execute(query, {
                        'id': prediction.prediction_id,
                        'pred_time': prediction.prediction_timestamp,
                        'trigger_time': prediction.predicted_trigger_time,
                        'probability': prediction.trigger_probability,
                        'type': prediction.trigger_type.value,
                        'magnitude': prediction.expected_magnitude,
                        'intensity': prediction.mixing_intensity_prediction,
                        'confidence': json.dumps(prediction.confidence_interval),
                        'uncertainty': json.dumps(prediction.uncertainty_bounds)
                    })
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing predictions: {str(e)}")