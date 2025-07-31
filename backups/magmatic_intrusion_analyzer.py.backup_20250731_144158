#!/usr/bin/env python3
"""
Magmatic Intrusion Analysis Module
Provides comprehensive analysis of in-progress magmatic intrusions with geological model training
"""

import asyncio
import logging
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import httpx
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class IntrusionGeometry:
    """Geometric parameters of magmatic intrusion"""
    depth_range: Tuple[float, float]  # meters below surface
    volume_estimate: float  # cubic meters
    aspect_ratio: float  # length/width
    orientation: float  # degrees from north
    dip_angle: float  # degrees from horizontal
    uncertainty_bounds: Dict[str, Tuple[float, float]]

@dataclass
class IntrusionDynamics:
    """Dynamic parameters of magmatic intrusion"""
    intrusion_rate: float  # cubic meters per day
    pressure_change: float  # pascals
    temperature_profile: List[float]  # celsius at different depths
    volatile_content: Dict[str, float]  # H2O, CO2, SO2 percentages
    crystallization_rate: float  # fraction per day
    viscosity_estimate: float  # pascal-seconds

@dataclass
class IntrusionRiskAssessment:
    """Risk assessment for magmatic intrusion"""
    eruption_probability: Dict[str, float]  # timeframe -> probability
    hazard_zones: Dict[str, List[Tuple[float, float]]]  # hazard type -> coordinates
    population_at_risk: int
    infrastructure_impact: Dict[str, str]
    recommended_actions: List[str]
    alert_level: int  # 0-5 scale

@dataclass
class IntrusionAnalysisResult:
    """Complete intrusion analysis result"""
    analysis_id: str
    timestamp: datetime
    geometry: IntrusionGeometry
    dynamics: IntrusionDynamics
    risk_assessment: IntrusionRiskAssessment
    model_confidence: float
    data_quality_score: float
    next_analysis_time: datetime

class MagmaticIntrusionAnalyzer:
    """Comprehensive magmatic intrusion analysis system"""
    
    def __init__(self, service_registry: Dict[str, str]):
        self.service_registry = service_registry
        self.analysis_history = []
        self.model_cache = {}
        self.data_quality_threshold = 0.7
        
    async def analyze_intrusion(self, analysis_id: str) -> IntrusionAnalysisResult:
        """Perform comprehensive magmatic intrusion analysis"""
        try:
            logger.info(f"Starting intrusion analysis {analysis_id}")
            
            # Step 1: Collect and validate monitoring data
            monitoring_data = await self._collect_monitoring_data()
            data_quality = self._assess_data_quality(monitoring_data)
            
            if data_quality < self.data_quality_threshold:
                logger.warning(f"Data quality below threshold: {data_quality}")
            
            # Step 2: Train/update geological models
            model_results = await self._train_geological_models(monitoring_data)
            
            # Step 3: Analyze intrusion geometry
            geometry = await self._analyze_intrusion_geometry(monitoring_data, model_results)
            
            # Step 4: Analyze intrusion dynamics
            dynamics = await self._analyze_intrusion_dynamics(monitoring_data, model_results)
            
            # Step 5: Perform risk assessment
            risk_assessment = await self._perform_risk_assessment(geometry, dynamics, monitoring_data)
            
            # Step 6: Calculate model confidence
            model_confidence = self._calculate_model_confidence(model_results, data_quality)
            
            # Step 7: Determine next analysis time
            next_analysis_time = self._calculate_next_analysis_time(risk_assessment.alert_level)
            
            # Create analysis result
            result = IntrusionAnalysisResult(
                analysis_id=analysis_id,
                timestamp=datetime.utcnow(),
                geometry=geometry,
                dynamics=dynamics,
                risk_assessment=risk_assessment,
                model_confidence=model_confidence,
                data_quality_score=data_quality,
                next_analysis_time=next_analysis_time
            )
            
            # Store analysis in history
            self.analysis_history.append(result)
            
            logger.info(f"Intrusion analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Intrusion analysis {analysis_id} failed: {e}")
            raise
    
    async def _collect_monitoring_data(self) -> Dict[str, Any]:
        """Collect comprehensive monitoring data from all sources"""
        monitoring_data = {
            'seismic': {},
            'deformation': {},
            'thermal': {},
            'geochemical': {},
            'gravity': {},
            'magnetic': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            async with httpx.AsyncClient() as client:
                # Collect seismic data
                seismic_response = await client.get(
                    f"{self.service_registry['volcano-watchtower']}/data/seismic/comprehensive",
                    timeout=30.0
                )
                if seismic_response.status_code == 200:
                    monitoring_data['seismic'] = seismic_response.json()
                
                # Collect deformation data (GPS, InSAR, tiltmeters)
                deformation_response = await client.get(
                    f"{self.service_registry['data-ingestion']}/data/deformation/multi-source",
                    timeout=30.0
                )
                if deformation_response.status_code == 200:
                    monitoring_data['deformation'] = deformation_response.json()
                
                # Collect thermal data (satellite, ground-based)
                thermal_response = await client.get(
                    f"{self.service_registry['data-ingestion']}/data/thermal/comprehensive",
                    timeout=30.0
                )
                if thermal_response.status_code == 200:
                    monitoring_data['thermal'] = thermal_response.json()
                
                # Collect geochemical data
                geochem_response = await client.get(
                    f"{self.service_registry['data-ingestion']}/data/geochemical/recent",
                    timeout=30.0
                )
                if geochem_response.status_code == 200:
                    monitoring_data['geochemical'] = geochem_response.json()
                
                # Collect gravity data
                gravity_response = await client.get(
                    f"{self.service_registry['data-ingestion']}/data/gravity/recent",
                    timeout=30.0
                )
                if gravity_response.status_code == 200:
                    monitoring_data['gravity'] = gravity_response.json()
                
                # Collect magnetic data
                magnetic_response = await client.get(
                    f"{self.service_registry['data-ingestion']}/data/magnetic/recent",
                    timeout=30.0
                )
                if magnetic_response.status_code == 200:
                    monitoring_data['magnetic'] = magnetic_response.json()
                
        except Exception as e:
            logger.error(f"Failed to collect monitoring data: {e}")
            
        return monitoring_data
    
    def _assess_data_quality(self, monitoring_data: Dict[str, Any]) -> float:
        """Assess overall data quality for analysis"""
        quality_scores = []
        
        try:
            # Assess seismic data quality
            if 'seismic' in monitoring_data and monitoring_data['seismic']:
                seismic_quality = self._assess_seismic_data_quality(monitoring_data['seismic'])
                quality_scores.append(seismic_quality)
            
            # Assess deformation data quality
            if 'deformation' in monitoring_data and monitoring_data['deformation']:
                deformation_quality = self._assess_deformation_data_quality(monitoring_data['deformation'])
                quality_scores.append(deformation_quality)
            
            # Assess thermal data quality
            if 'thermal' in monitoring_data and monitoring_data['thermal']:
                thermal_quality = self._assess_thermal_data_quality(monitoring_data['thermal'])
                quality_scores.append(thermal_quality)
            
            # Calculate overall quality score
            if quality_scores:
                return np.mean(quality_scores)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return 0.0
    
    def _assess_seismic_data_quality(self, seismic_data: Dict[str, Any]) -> float:
        """Assess seismic data quality"""
        try:
            quality_factors = []
            
            # Check data completeness
            if 'events' in seismic_data:
                events = seismic_data['events']
                if len(events) > 10:  # Sufficient events for analysis
                    quality_factors.append(0.9)
                elif len(events) > 5:
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.3)
            
            # Check magnitude distribution
            if 'magnitude_distribution' in seismic_data:
                mag_dist = seismic_data['magnitude_distribution']
                if len(mag_dist) > 3:  # Good magnitude range
                    quality_factors.append(0.8)
                else:
                    quality_factors.append(0.5)
            
            # Check temporal coverage
            if 'temporal_coverage' in seismic_data:
                coverage = seismic_data['temporal_coverage']
                if coverage > 0.8:  # 80% temporal coverage
                    quality_factors.append(0.9)
                elif coverage > 0.5:
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.3)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Seismic data quality assessment failed: {e}")
            return 0.5
    
    def _assess_deformation_data_quality(self, deformation_data: Dict[str, Any]) -> float:
        """Assess deformation data quality"""
        try:
            quality_factors = []
            
            # Check spatial coverage
            if 'spatial_coverage' in deformation_data:
                coverage = deformation_data['spatial_coverage']
                if coverage > 0.7:
                    quality_factors.append(0.9)
                elif coverage > 0.4:
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.3)
            
            # Check measurement precision
            if 'measurement_precision' in deformation_data:
                precision = deformation_data['measurement_precision']
                if precision < 0.005:  # Sub-centimeter precision
                    quality_factors.append(0.9)
                elif precision < 0.01:
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.4)
            
            # Check temporal resolution
            if 'temporal_resolution' in deformation_data:
                resolution = deformation_data['temporal_resolution']
                if resolution < 3600:  # Hourly or better
                    quality_factors.append(0.9)
                elif resolution < 86400:  # Daily
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.4)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Deformation data quality assessment failed: {e}")
            return 0.5
    
    def _assess_thermal_data_quality(self, thermal_data: Dict[str, Any]) -> float:
        """Assess thermal data quality"""
        try:
            quality_factors = []
            
            # Check temperature range
            if 'temperature_range' in thermal_data:
                temp_range = thermal_data['temperature_range']
                if temp_range > 50:  # Good temperature contrast
                    quality_factors.append(0.8)
                elif temp_range > 20:
                    quality_factors.append(0.6)
                else:
                    quality_factors.append(0.3)
            
            # Check spatial resolution
            if 'spatial_resolution' in thermal_data:
                resolution = thermal_data['spatial_resolution']
                if resolution < 10:  # Sub-10m resolution
                    quality_factors.append(0.9)
                elif resolution < 30:
                    quality_factors.append(0.7)
                else:
                    quality_factors.append(0.4)
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Thermal data quality assessment failed: {e}")
            return 0.5
    
    async def _train_geological_models(self, monitoring_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train/update geological models with current data"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.service_registry['helios-core']}/model/train/intrusion-specific",
                    json={
                        'training_data': monitoring_data,
                        'model_type': 'magmatic_intrusion',
                        'physics_constraints': {
                            'elastic_deformation': True,
                            'thermal_diffusion': True,
                            'fluid_dynamics': True,
                            'mass_conservation': True
                        },
                        'uncertainty_quantification': True,
                        'ensemble_size': 10
                    },
                    timeout=300.0  # 5 minutes for model training
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Model training failed with status {response.status_code}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Geological model training failed: {e}")
            return {}
    
    async def _analyze_intrusion_geometry(self, monitoring_data: Dict[str, Any], model_results: Dict[str, Any]) -> IntrusionGeometry:
        """Analyze intrusion geometry using multi-source data"""
        try:
            # Extract deformation patterns
            deformation_data = monitoring_data.get('deformation', {})
            
            # Estimate depth range from seismic data
            seismic_data = monitoring_data.get('seismic', {})
            depth_range = self._estimate_depth_range(seismic_data)
            
            # Estimate volume from deformation modeling
            volume_estimate = self._estimate_intrusion_volume(deformation_data, model_results)
            
            # Estimate geometry parameters
            aspect_ratio = self._estimate_aspect_ratio(deformation_data)
            orientation = self._estimate_orientation(deformation_data)
            dip_angle = self._estimate_dip_angle(deformation_data, seismic_data)
            
            # Calculate uncertainty bounds
            uncertainty_bounds = self._calculate_geometry_uncertainties(
                depth_range, volume_estimate, aspect_ratio, orientation, dip_angle
            )
            
            return IntrusionGeometry(
                depth_range=depth_range,
                volume_estimate=volume_estimate,
                aspect_ratio=aspect_ratio,
                orientation=orientation,
                dip_angle=dip_angle,
                uncertainty_bounds=uncertainty_bounds
            )
            
        except Exception as e:
            logger.error(f"Intrusion geometry analysis failed: {e}")
            # Return default geometry with high uncertainty
            return IntrusionGeometry(
                depth_range=(1000.0, 5000.0),
                volume_estimate=1e6,
                aspect_ratio=2.0,
                orientation=0.0,
                dip_angle=45.0,
                uncertainty_bounds={
                    'depth_range': ((500.0, 10000.0), (500.0, 10000.0)),
                    'volume_estimate': (1e5, 1e7),
                    'aspect_ratio': (1.0, 5.0),
                    'orientation': (-180.0, 180.0),
                    'dip_angle': (0.0, 90.0)
                }
            )
    
    def _estimate_depth_range(self, seismic_data: Dict[str, Any]) -> Tuple[float, float]:
        """Estimate intrusion depth range from seismic data"""
        try:
            if 'events' in seismic_data and seismic_data['events']:
                depths = [event.get('depth', 3000) for event in seismic_data['events']]
                if depths:
                    min_depth = min(depths) - 500  # Add buffer
                    max_depth = max(depths) + 500
                    return (max(0, min_depth), max_depth)
            
            # Default depth range if no seismic data
            return (2000.0, 4000.0)
            
        except Exception as e:
            logger.error(f"Depth range estimation failed: {e}")
            return (2000.0, 4000.0)
    
    def _estimate_intrusion_volume(self, deformation_data: Dict[str, Any], model_results: Dict[str, Any]) -> float:
        """Estimate intrusion volume from deformation modeling"""
        try:
            # Use model results if available
            if 'volume_estimate' in model_results:
                return model_results['volume_estimate']
            
            # Estimate from deformation data
            if 'max_displacement' in deformation_data and 'affected_area' in deformation_data:
                max_disp = deformation_data['max_displacement']
                area = deformation_data['affected_area']
                
                # Simple volume estimate: displacement * area * scaling factor
                volume = max_disp * area * 1000  # Scaling factor for depth
                return max(1e5, min(1e8, volume))  # Reasonable bounds
            
            # Default volume estimate
            return 1e6
            
        except Exception as e:
            logger.error(f"Volume estimation failed: {e}")
            return 1e6
    
    def _estimate_aspect_ratio(self, deformation_data: Dict[str, Any]) -> float:
        """Estimate intrusion aspect ratio from deformation pattern"""
        try:
            if 'deformation_ellipse' in deformation_data:
                ellipse = deformation_data['deformation_ellipse']
                if 'major_axis' in ellipse and 'minor_axis' in ellipse:
                    return ellipse['major_axis'] / ellipse['minor_axis']
            
            # Default aspect ratio
            return 2.0
            
        except Exception as e:
            logger.error(f"Aspect ratio estimation failed: {e}")
            return 2.0
    
    def _estimate_orientation(self, deformation_data: Dict[str, Any]) -> float:
        """Estimate intrusion orientation from deformation pattern"""
        try:
            if 'deformation_ellipse' in deformation_data:
                ellipse = deformation_data['deformation_ellipse']
                if 'orientation' in ellipse:
                    return ellipse['orientation']
            
            # Default orientation
            return 0.0
            
        except Exception as e:
            logger.error(f"Orientation estimation failed: {e}")
            return 0.0
    
    def _estimate_dip_angle(self, deformation_data: Dict[str, Any], seismic_data: Dict[str, Any]) -> float:
        """Estimate intrusion dip angle"""
        try:
            # Use seismic focal mechanisms if available
            if 'focal_mechanisms' in seismic_data:
                mechanisms = seismic_data['focal_mechanisms']
                if mechanisms:
                    dips = [mech.get('dip', 45) for mech in mechanisms]
                    return np.mean(dips)
            
            # Use deformation asymmetry
            if 'asymmetry_factor' in deformation_data:
                asymmetry = deformation_data['asymmetry_factor']
                # Convert asymmetry to dip angle (simplified relationship)
                dip = 45 + (asymmetry - 0.5) * 60  # Range: -15 to 105 degrees
                return max(0, min(90, dip))
            
            # Default dip angle
            return 45.0
            
        except Exception as e:
            logger.error(f"Dip angle estimation failed: {e}")
            return 45.0
    
    def _calculate_geometry_uncertainties(self, depth_range: Tuple[float, float], 
                                        volume_estimate: float, aspect_ratio: float,
                                        orientation: float, dip_angle: float) -> Dict[str, Tuple[float, float]]:
        """Calculate uncertainty bounds for geometry parameters"""
        return {
            'depth_range': (
                (depth_range[0] * 0.7, depth_range[0] * 1.3),
                (depth_range[1] * 0.7, depth_range[1] * 1.3)
            ),
            'volume_estimate': (volume_estimate * 0.5, volume_estimate * 2.0),
            'aspect_ratio': (aspect_ratio * 0.7, aspect_ratio * 1.5),
            'orientation': (orientation - 30, orientation + 30),
            'dip_angle': (max(0, dip_angle - 20), min(90, dip_angle + 20))
        }
    
    async def _analyze_intrusion_dynamics(self, monitoring_data: Dict[str, Any], model_results: Dict[str, Any]) -> IntrusionDynamics:
        """Analyze intrusion dynamics and temporal evolution"""
        try:
            # Calculate intrusion rate from temporal deformation
            intrusion_rate = self._calculate_intrusion_rate(monitoring_data)
            
            # Estimate pressure change
            pressure_change = self._estimate_pressure_change(monitoring_data, model_results)
            
            # Analyze temperature profile
            temperature_profile = self._analyze_temperature_profile(monitoring_data)
            
            # Estimate volatile content
            volatile_content = self._estimate_volatile_content(monitoring_data)
            
            # Calculate crystallization rate
            crystallization_rate = self._calculate_crystallization_rate(monitoring_data)
            
            # Estimate viscosity
            viscosity_estimate = self._estimate_viscosity(temperature_profile, volatile_content)
            
            return IntrusionDynamics(
                intrusion_rate=intrusion_rate,
                pressure_change=pressure_change,
                temperature_profile=temperature_profile,
                volatile_content=volatile_content,
                crystallization_rate=crystallization_rate,
                viscosity_estimate=viscosity_estimate
            )
            
        except Exception as e:
            logger.error(f"Intrusion dynamics analysis failed: {e}")
            # Return default dynamics
            return IntrusionDynamics(
                intrusion_rate=1000.0,  # m³/day
                pressure_change=1e6,    # Pa
                temperature_profile=[1000, 950, 900, 850, 800],  # °C
                volatile_content={'H2O': 3.0, 'CO2': 0.5, 'SO2': 0.1},
                crystallization_rate=0.01,  # fraction/day
                viscosity_estimate=1e4      # Pa·s
            )
    
    def _calculate_intrusion_rate(self, monitoring_data: Dict[str, Any]) -> float:
        """Calculate intrusion rate from temporal data"""
        try:
            deformation_data = monitoring_data.get('deformation', {})
            if 'temporal_evolution' in deformation_data:
                evolution = deformation_data['temporal_evolution']
                if len(evolution) > 1:
                    # Calculate rate of volume change
                    times = [point['timestamp'] for point in evolution]
                    volumes = [point['estimated_volume'] for point in evolution]
                    
                    if len(times) > 1 and len(volumes) > 1:
                        # Linear regression to get rate
                        time_diffs = [(datetime.fromisoformat(t) - datetime.fromisoformat(times[0])).total_seconds() 
                                     for t in times]
                        slope, _, _, _, _ = stats.linregress(time_diffs, volumes)
                        return max(0, slope * 86400)  # Convert to m³/day
            
            # Default intrusion rate
            return 1000.0
            
        except Exception as e:
            logger.error(f"Intrusion rate calculation failed: {e}")
            return 1000.0
    
    def _estimate_pressure_change(self, monitoring_data: Dict[str, Any], model_results: Dict[str, Any]) -> float:
        """Estimate pressure change from deformation and seismic data"""
        try:
            # Use model results if available
            if 'pressure_estimate' in model_results:
                return model_results['pressure_estimate']
            
            # Estimate from deformation magnitude
            deformation_data = monitoring_data.get('deformation', {})
            if 'max_displacement' in deformation_data:
                max_disp = deformation_data['max_displacement']
                # Simplified relationship: pressure ~ displacement * elastic modulus / depth
                elastic_modulus = 30e9  # Pa (typical for rock)
                depth = 3000  # m (typical intrusion depth)
                pressure = (max_disp * elastic_modulus) / depth
                return pressure
            
            # Default pressure change
            return 1e6
            
        except Exception as e:
            logger.error(f"Pressure change estimation failed: {e}")
            return 1e6
    
    def _analyze_temperature_profile(self, monitoring_data: Dict[str, Any]) -> List[float]:
        """Analyze temperature profile of intrusion"""
        try:
            thermal_data = monitoring_data.get('thermal', {})
            if 'temperature_profile' in thermal_data:
                return thermal_data['temperature_profile']
            
            # Estimate temperature profile based on depth
            # Typical magma temperature: 1000-1200°C, cooling with depth
            depths = [1000, 2000, 3000, 4000, 5000]  # meters
            temperatures = []
            
            for depth in depths:
                # Temperature decreases with distance from intrusion center
                temp = 1100 - (depth - 2500) * 0.05  # °C
                temperatures.append(max(800, min(1200, temp)))
            
            return temperatures
            
        except Exception as e:
            logger.error(f"Temperature profile analysis failed: {e}")
            return [1000, 950, 900, 850, 800]
    
    def _estimate_volatile_content(self, monitoring_data: Dict[str, Any]) -> Dict[str, float]:
        """Estimate volatile content from geochemical data"""
        try:
            geochem_data = monitoring_data.get('geochemical', {})
            if 'volatile_analysis' in geochem_data:
                return geochem_data['volatile_analysis']
            
            # Default volatile content for typical magma
            return {
                'H2O': 3.0,   # weight percent
                'CO2': 0.5,   # weight percent
                'SO2': 0.1,   # weight percent
                'H2S': 0.05,  # weight percent
                'HCl': 0.02   # weight percent
            }
            
        except Exception as e:
            logger.error(f"Volatile content estimation failed: {e}")
            return {'H2O': 3.0, 'CO2': 0.5, 'SO2': 0.1}
    
    def _calculate_crystallization_rate(self, monitoring_data: Dict[str, Any]) -> float:
        """Calculate crystallization rate from thermal evolution"""
        try:
            thermal_data = monitoring_data.get('thermal', {})
            if 'cooling_rate' in thermal_data:
                cooling_rate = thermal_data['cooling_rate']  # °C/day
                # Crystallization rate proportional to cooling rate
                return min(0.1, max(0.001, cooling_rate / 100))  # fraction/day
            
            # Default crystallization rate
            return 0.01
            
        except Exception as e:
            logger.error(f"Crystallization rate calculation failed: {e}")
            return 0.01
    
    def _estimate_viscosity(self, temperature_profile: List[float], volatile_content: Dict[str, float]) -> float:
        """Estimate magma viscosity from temperature and volatile content"""
        try:
            # Use average temperature
            avg_temp = np.mean(temperature_profile)
            
            # Water content (main factor affecting viscosity)
            water_content = volatile_content.get('H2O', 3.0)
            
            # Simplified viscosity model (Giordano et al., 2008)
            # log10(viscosity) = A + B/T - C*log10(water_content)
            A = 4.5
            B = 8000
            C = 2.0
            
            log_viscosity = A + B / (avg_temp + 273.15) - C * np.log10(max(0.1, water_content))
            viscosity = 10 ** log_viscosity
            
            return max(1e2, min(1e8, viscosity))  # Reasonable bounds
            
        except Exception as e:
            logger.error(f"Viscosity estimation failed: {e}")
            return 1e4
    
    async def _perform_risk_assessment(self, geometry: IntrusionGeometry, 
                                     dynamics: IntrusionDynamics, 
                                     monitoring_data: Dict[str, Any]) -> IntrusionRiskAssessment:
        """Perform comprehensive risk assessment"""
        try:
            # Calculate eruption probabilities for different timeframes
            eruption_probability = self._calculate_eruption_probabilities(geometry, dynamics, monitoring_data)
            
            # Define hazard zones
            hazard_zones = self._define_hazard_zones(geometry, dynamics)
            
            # Estimate population at risk
            population_at_risk = self._estimate_population_at_risk(hazard_zones)
            
            # Assess infrastructure impact
            infrastructure_impact = self._assess_infrastructure_impact(hazard_zones)
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(eruption_probability, hazard_zones)
            
            # Determine alert level
            alert_level = self._determine_alert_level(eruption_probability, dynamics)
            
            return IntrusionRiskAssessment(
                eruption_probability=eruption_probability,
                hazard_zones=hazard_zones,
                population_at_risk=population_at_risk,
                infrastructure_impact=infrastructure_impact,
                recommended_actions=recommended_actions,
                alert_level=alert_level
            )
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            # Return conservative risk assessment
            return IntrusionRiskAssessment(
                eruption_probability={'24_hours': 0.05, '1_week': 0.15, '1_month': 0.30},
                hazard_zones={'pyroclastic_flow': [(0, 0), (1000, 1000)], 'ashfall': [(0, 0), (5000, 5000)]},
                population_at_risk=10000,
                infrastructure_impact={'roads': 'moderate', 'buildings': 'low', 'utilities': 'moderate'},
                recommended_actions=['increase_monitoring', 'prepare_evacuation_plans', 'notify_authorities'],
                alert_level=3
            )
    
    def _calculate_eruption_probabilities(self, geometry: IntrusionGeometry, 
                                        dynamics: IntrusionDynamics, 
                                        monitoring_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate eruption probabilities for different timeframes"""
        try:
            # Base probability factors
            depth_factor = 1.0 / (geometry.depth_range[0] / 1000)  # Shallower = higher probability
            volume_factor = min(2.0, geometry.volume_estimate / 1e6)  # Larger = higher probability
            rate_factor = min(2.0, dynamics.intrusion_rate / 1000)  # Faster = higher probability
            pressure_factor = min(2.0, dynamics.pressure_change / 1e6)  # Higher pressure = higher probability
            
            # Seismic activity factor
            seismic_data = monitoring_data.get('seismic', {})
            seismic_factor = 1.0
            if 'event_rate' in seismic_data:
                seismic_factor = min(2.0, seismic_data['event_rate'] / 10)  # Events per day
            
            # Combined probability factor
            combined_factor = (depth_factor * volume_factor * rate_factor * 
                             pressure_factor * seismic_factor) / 5
            
            # Base probabilities (conservative estimates)
            base_24h = 0.01
            base_1week = 0.05
            base_1month = 0.15
            
            return {
                '24_hours': min(0.5, base_24h * combined_factor),
                '1_week': min(0.7, base_1week * combined_factor),
                '1_month': min(0.8, base_1month * combined_factor),
                '6_months': min(0.9, base_1month * combined_factor * 2),
                '1_year': min(0.95, base_1month * combined_factor * 3)
            }
            
        except Exception as e:
            logger.error(f"Eruption probability calculation failed: {e}")
            return {'24_hours': 0.05, '1_week': 0.15, '1_month': 0.30}
    
    def _define_hazard_zones(self, geometry: IntrusionGeometry, dynamics: IntrusionDynamics) -> Dict[str, List[Tuple[float, float]]]:
        """Define hazard zones based on intrusion characteristics"""
        try:
            # Estimate hazard radii based on intrusion volume and dynamics
            volume = geometry.volume_estimate
            intrusion_rate = dynamics.intrusion_rate
            
            # Pyroclastic flow zone (high hazard)
            pf_radius = min(10000, max(1000, (volume / 1e6) ** 0.5 * 2000))
            
            # Ashfall zone (moderate hazard)
            ashfall_radius = min(50000, max(5000, (volume / 1e6) ** 0.5 * 10000))
            
            # Lahar zone (moderate hazard, depends on topography)
            lahar_radius = min(20000, max(2000, (volume / 1e6) ** 0.5 * 5000))
            
            # Gas hazard zone (low to moderate hazard)
            gas_radius = min(15000, max(1500, (volume / 1e6) ** 0.5 * 3000))
            
            # Generate circular zones (simplified - in reality would use topography)
            center = (0, 0)  # Relative to intrusion center
            
            return {
                'pyroclastic_flow': self._generate_circular_zone(center, pf_radius),
                'ashfall': self._generate_circular_zone(center, ashfall_radius),
                'lahar': self._generate_circular_zone(center, lahar_radius),
                'gas_hazard': self._generate_circular_zone(center, gas_radius)
            }
            
        except Exception as e:
            logger.error(f"Hazard zone definition failed: {e}")
            return {
                'pyroclastic_flow': [(0, 0), (2000, 2000)],
                'ashfall': [(0, 0), (10000, 10000)]
            }
    
    def _generate_circular_zone(self, center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        """Generate circular hazard zone coordinates"""
        angles = np.linspace(0, 2 * np.pi, 36)  # 36 points for circle
        x_center, y_center = center
        
        coordinates = []
        for angle in angles:
            x = x_center + radius * np.cos(angle)
            y = y_center + radius * np.sin(angle)
            coordinates.append((x, y))
        
        return coordinates
    
    def _estimate_population_at_risk(self, hazard_zones: Dict[str, List[Tuple[float, float]]]) -> int:
        """Estimate population at risk from hazard zones"""
        try:
            # Simplified population density estimation
            # In reality, would use actual population data
            
            max_radius = 0
            for zone_coords in hazard_zones.values():
                for x, y in zone_coords:
                    radius = (x**2 + y**2)**0.5
                    max_radius = max(max_radius, radius)
            
            # Assume population density decreases with distance from center
            # Urban: 1000 people/km², suburban: 100 people/km², rural: 10 people/km²
            area_km2 = (max_radius / 1000) ** 2 * np.pi
            
            if max_radius < 5000:  # Urban area
                population_density = 1000
            elif max_radius < 20000:  # Suburban area
                population_density = 100
            else:  # Rural area
                population_density = 10
            
            return int(area_km2 * population_density)
            
        except Exception as e:
            logger.error(f"Population estimation failed: {e}")
            return 10000
    
    def _assess_infrastructure_impact(self, hazard_zones: Dict[str, List[Tuple[float, float]]]) -> Dict[str, str]:
        """Assess potential infrastructure impact"""
        try:
            # Determine maximum hazard extent
            max_extent = 0
            for zone_coords in hazard_zones.values():
                for x, y in zone_coords:
                    extent = (x**2 + y**2)**0.5
                    max_extent = max(max_extent, extent)
            
            # Assess impact levels based on extent
            if max_extent < 2000:
                return {
                    'roads': 'low',
                    'buildings': 'low',
                    'utilities': 'low',
                    'airports': 'none',
                    'ports': 'none'
                }
            elif max_extent < 10000:
                return {
                    'roads': 'moderate',
                    'buildings': 'moderate',
                    'utilities': 'moderate',
                    'airports': 'low',
                    'ports': 'low'
                }
            else:
                return {
                    'roads': 'high',
                    'buildings': 'high',
                    'utilities': 'high',
                    'airports': 'moderate',
                    'ports': 'moderate'
                }
                
        except Exception as e:
            logger.error(f"Infrastructure impact assessment failed: {e}")
            return {'roads': 'moderate', 'buildings': 'moderate', 'utilities': 'moderate'}
    
    def _generate_recommended_actions(self, eruption_probability: Dict[str, float], 
                                    hazard_zones: Dict[str, List[Tuple[float, float]]]) -> List[str]:
        """Generate recommended actions based on risk assessment"""
        actions = []
        
        try:
            # Actions based on eruption probability
            prob_24h = eruption_probability.get('24_hours', 0)
            prob_1week = eruption_probability.get('1_week', 0)
            
            if prob_24h > 0.2:
                actions.extend([
                    'immediate_evacuation_recommended',
                    'close_hazard_zones',
                    'activate_emergency_response'
                ])
            elif prob_24h > 0.1:
                actions.extend([
                    'prepare_immediate_evacuation',
                    'restrict_access_to_hazard_zones',
                    'alert_emergency_services'
                ])
            elif prob_1week > 0.3:
                actions.extend([
                    'prepare_evacuation_plans',
                    'increase_monitoring_frequency',
                    'notify_local_authorities'
                ])
            else:
                actions.extend([
                    'continue_monitoring',
                    'review_evacuation_plans',
                    'inform_local_communities'
                ])
            
            # Actions based on hazard zones
            if 'pyroclastic_flow' in hazard_zones:
                actions.append('establish_pyroclastic_flow_exclusion_zone')
            
            if 'ashfall' in hazard_zones:
                actions.append('prepare_ashfall_protection_measures')
            
            # Always include these actions
            actions.extend([
                'maintain_communication_systems',
                'coordinate_with_scientific_community',
                'update_hazard_assessments_regularly'
            ])
            
        except Exception as e:
            logger.error(f"Recommended actions generation failed: {e}")
            actions = ['increase_monitoring', 'prepare_evacuation_plans', 'notify_authorities']
        
        return list(set(actions))  # Remove duplicates
    
    def _determine_alert_level(self, eruption_probability: Dict[str, float], dynamics: IntrusionDynamics) -> int:
        """Determine alert level (0-5 scale)"""
        try:
            prob_24h = eruption_probability.get('24_hours', 0)
            prob_1week = eruption_probability.get('1_week', 0)
            intrusion_rate = dynamics.intrusion_rate
            
            # Alert level based on probability and dynamics
            if prob_24h > 0.3 or intrusion_rate > 5000:
                return 5  # Critical
            elif prob_24h > 0.2 or prob_1week > 0.5:
                return 4  # High
            elif prob_24h > 0.1 or prob_1week > 0.3:
                return 3  # Elevated
            elif prob_24h > 0.05 or prob_1week > 0.15:
                return 2  # Moderate
            elif prob_24h > 0.01 or prob_1week > 0.05:
                return 1  # Low
            else:
                return 0  # Normal
                
        except Exception as e:
            logger.error(f"Alert level determination failed: {e}")
            return 2  # Default to moderate
    
    def _calculate_model_confidence(self, model_results: Dict[str, Any], data_quality: float) -> float:
        """Calculate overall model confidence"""
        try:
            confidence_factors = []
            
            # Data quality factor
            confidence_factors.append(data_quality)
            
            # Model performance factor
            if 'model_accuracy' in model_results:
                confidence_factors.append(model_results['model_accuracy'])
            
            # Physics constraints satisfaction
            if 'physics_constraints_satisfied' in model_results:
                if model_results['physics_constraints_satisfied']:
                    confidence_factors.append(0.9)
                else:
                    confidence_factors.append(0.5)
            
            # Ensemble agreement
            if 'ensemble_agreement' in model_results:
                confidence_factors.append(model_results['ensemble_agreement'])
            
            # Calculate overall confidence
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Model confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_next_analysis_time(self, alert_level: int) -> datetime:
        """Calculate when next analysis should be performed"""
        try:
            # Analysis frequency based on alert level
            if alert_level >= 4:
                hours = 1  # Every hour for critical/high alert
            elif alert_level == 3:
                hours = 4  # Every 4 hours for elevated alert
            elif alert_level == 2:
                hours = 12  # Every 12 hours for moderate alert
            elif alert_level == 1:
                hours = 24  # Daily for low alert
            else:
                hours = 72  # Every 3 days for normal conditions
            
            return datetime.utcnow() + timedelta(hours=hours)
            
        except Exception as e:
            logger.error(f"Next analysis time calculation failed: {e}")
            return datetime.utcnow() + timedelta(hours=24)

# Example usage and testing
if __name__ == "__main__":
    # Example service registry
    service_registry = {
        'helios-core': 'http://helios-core:8000',
        'volcano-watchtower': 'http://volcano-watchtower:8001',
        'data-ingestion': 'http://data-ingestion:8002'
    }
    
    # Create analyzer
    analyzer = MagmaticIntrusionAnalyzer(service_registry)
    
    # Example analysis (would be called asynchronously in production)
    print("Magmatic Intrusion Analyzer initialized successfully")
    print(f"Service registry: {service_registry}")
    print("Ready for intrusion analysis requests")