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
                
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


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
            
            
class AdvancedSeismicCNN:
    """Convolutional Neural Network for seismic event detection and classification"""
    
    def __init__(self, input_channels: int = 3, num_classes: int = 5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model(input_channels, num_classes)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def _build_model(self, input_channels: int, num_classes: int) -> nn.Module:
        """Build advanced CNN architecture with attention mechanisms"""
        
        class AttentionBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.attention = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Conv1d(channels, channels // 8, 1),
                    nn.ReLU(),
                    nn.Conv1d(channels // 8, channels, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                attention_weights = self.attention(x)
                return x * attention_weights
        
        class SeismicCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                
                # Feature extraction layers
                self.conv_blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        AttentionBlock(64),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(64, 128, kernel_size=5, padding=2),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        AttentionBlock(128),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(128, 256, kernel_size=3, padding=1),
                        nn.BatchNorm1d(256),
                        nn.ReLU(),
                        AttentionBlock(256),
                        nn.MaxPool1d(2)
                    ),
                    nn.Sequential(
                        nn.Conv1d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm1d(512),
                        nn.ReLU(),
                        AttentionBlock(512),
                        nn.AdaptiveAvgPool1d(1)
                    )
                ])
                
                # Classification head
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
                
            def forward(self, x):
                for block in self.conv_blocks:
                    x = block(x)
                
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return SeismicCNN(input_channels, num_classes).to(self.device)
    
    def detect_seismic_events(self, waveform_data: np.ndarray, 
                            sampling_rate: float = 100.0) -> Dict[str, Any]:
        """Detect and classify seismic events in waveform data"""
        
        # Preprocess data
        processed_data = self._preprocess_waveform(waveform_data, sampling_rate)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(processed_data).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Event classification mapping
        event_types = {
            0: 'background_noise',
            1: 'local_earthquake',
            2: 'regional_earthquake',
            3: 'volcanic_tremor',
            4: 'explosion'
        }
        
        return {
            'event_type': event_types[predicted_class.item()],
            'confidence': probabilities.max().item(),
            'all_probabilities': {
                event_types[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            },
            'magnitude_estimate': self._estimate_magnitude(waveform_data),
            'arrival_time': self._detect_p_wave_arrival(waveform_data, sampling_rate),
            'frequency_content': self._analyze_frequency_content(waveform_data, sampling_rate)
        }
    
    def _preprocess_waveform(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Advanced waveform preprocessing"""
        # Bandpass filter
        from scipy import signal
        nyquist = sampling_rate / 2
        low_freq = 1.0 / nyquist
        high_freq = 25.0 / nyquist
        b, a = signal.butter(4, [low_freq, high_freq], btype='band')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        
        # Normalize
        normalized_data = (filtered_data - np.mean(filtered_data, axis=-1, keepdims=True)) / \
                         (np.std(filtered_data, axis=-1, keepdims=True) + 1e-8)
        
        return normalized_data
    
    def _estimate_magnitude(self, waveform_data: np.ndarray) -> float:
        """Estimate earthquake magnitude from waveform amplitude"""
        max_amplitude = np.max(np.abs(waveform_data))
        
class AdvancedPhysicsSimulator:
    """Physics-Informed Neural Network for geological simulations"""
    
    def __init__(self, domain_size: Tuple[int, int, int] = (100, 100, 100)):
        self.domain_size = domain_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pinn_model = self._build_pinn_model()
        self.optimizer = torch.optim.Adam(self.pinn_model.parameters(), lr=1e-3)
        
    def solve_heat_equation_3d(self, thermal_conductivity: float = 2.5,
                              heat_capacity: float = 1000.0,
                              density: float = 2700.0) -> Dict[str, np.ndarray]:
        """Solve 3D heat equation with geological parameters"""
        
        # Generate spatial-temporal grid
        x = torch.linspace(0, 1, self.domain_size[0])
        y = torch.linspace(0, 1, self.domain_size[1])
        z = torch.linspace(0, 1, self.domain_size[2])
        t = torch.linspace(0, 1, 100)
        
        X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
        coords = coords.to(self.device)
        
        # Training loop with physics constraints
        for epoch in range(1000):
            self.optimizer.zero_grad()
            
            # Forward pass
            u_pred = self.pinn_model(coords)
            
            # Physics loss (heat equation)
            u_t = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 3]
            u_x = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 0]
            u_y = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 1]
            u_z = torch.autograd.grad(u_pred.sum(), coords, create_graph=True)[0][:, 2]
            
            u_xx = torch.autograd.grad(u_x.sum(), coords, create_graph=True)[0][:, 0]
            u_yy = torch.autograd.grad(u_y.sum(), coords, create_graph=True)[0][:, 1]
            u_zz = torch.autograd.grad(u_z.sum(), coords, create_graph=True)[0][:, 2]
            
            alpha = thermal_conductivity / (density * heat_capacity)
            physics_loss = torch.mean((u_t - alpha * (u_xx + u_yy + u_zz))**2)
            
            physics_loss.backward()
            self.optimizer.step()
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {physics_loss.item():.6f}")
        
        # Generate solution
        with torch.no_grad():
            solution = self.pinn_model(coords).cpu().numpy()
            
        return {
            'temperature_field': solution[:, 0].reshape(self.domain_size + (100,)),
            'heat_flux_x': solution[:, 1].reshape(self.domain_size + (100,)),
            'heat_flux_y': solution[:, 2].reshape(self.domain_size + (100,)),
            'heat_flux_z': solution[:, 3].reshape(self.domain_size + (100,)),
            'thermal_conductivity': thermal_conductivity,
            'convergence_history': {'final_loss': physics_loss.item()}
        }

        magnitude = np.log10(max_amplitude * 1e6) - 3.0
        return max(0.0, min(9.0, magnitude))
    
    def _detect_p_wave_arrival(self, data: np.ndarray, sampling_rate: float) -> float:
        """Detect P-wave arrival time using STA/LTA algorithm"""
        # Short-term average / Long-term average
        sta_window = int(0.5 * sampling_rate)  # 0.5 seconds
        lta_window = int(10.0 * sampling_rate)  # 10 seconds
        
        energy = np.square(data)
        sta = np.convolve(energy, np.ones(sta_window)/sta_window, mode='same')
        lta = np.convolve(energy, np.ones(lta_window)/lta_window, mode='same')
        
        sta_lta_ratio = sta / (lta + 1e-8)
        trigger_threshold = 3.0
        
        trigger_indices = np.where(sta_lta_ratio > trigger_threshold)[0]
        if len(trigger_indices) > 0:
            return trigger_indices[0] / sampling_rate
        return 0.0
    
    def _analyze_frequency_content(self, data: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        """Analyze frequency content of seismic signal"""
        from scipy import signal
        
        frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)
        
        # Frequency bands of interest
        bands = {
            'low_freq': (0.1, 1.0),
            'mid_freq': (1.0, 10.0),
            'high_freq': (10.0, 25.0)
        }
        
        band_powers = {}
        for band_name, (low, high) in bands.items():
            band_mask = (frequencies >= low) & (frequencies <= high)
            band_powers[band_name] = np.trapz(psd[band_mask], frequencies[band_mask])
        
        # Dominant frequency
        dominant_freq = frequencies[np.argmax(psd)]
        
        return {
            'dominant_frequency': dominant_freq,
            'band_powers': band_powers,
            'spectral_centroid': np.sum(frequencies * psd) / np.sum(psd),
            'spectral_bandwidth': np.sqrt(np.sum(((frequencies - dominant_freq) ** 2) * psd) / np.sum(psd))
        }

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
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


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
            
class AdvancedVolcanicMonitor:
    """Advanced volcanic monitoring system with multi-parameter analysis"""
    
    def __init__(self):
        self.alert_levels = {
            'green': {'level': 0, 'description': 'Normal background activity'},
            'yellow': {'level': 1, 'description': 'Elevated unrest'},
            'orange': {'level': 2, 'description': 'Heightened unrest'},
            'red': {'level': 3, 'description': 'Imminent eruption or in progress'}
        }
        
        self.monitoring_parameters = [
            'seismic_activity', 'ground_deformation', 'gas_emissions',
            'thermal_anomalies', 'acoustic_signals', 'hydrological_changes'
        ]
        
        self.ml_models = self._initialize_prediction_models()
    
    def _initialize_prediction_models(self) -> Dict[str, Any]:
        """Initialize machine learning models for eruption prediction"""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
        from sklearn.neural_network import MLPClassifier
        
        return {
            'eruption_classifier': RandomForestClassifier(n_estimators=200, random_state=42),
            'magnitude_predictor': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'time_to_eruption': MLPClassifier(hidden_layer_sizes=(100, 50, 25), random_state=42),
            'alert_level_classifier': RandomForestClassifier(n_estimators=100, random_state=42)
        }
    
    def comprehensive_volcanic_assessment(self, monitoring_data: Dict[str, Any],
                                        volcano_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive volcanic hazard assessment"""
        
        # Process individual monitoring parameters
        seismic_analysis = self._analyze_seismic_activity(monitoring_data.get('seismic', {}))
        deformation_analysis = self._analyze_ground_deformation(monitoring_data.get('deformation', {}))
        gas_analysis = self._analyze_gas_emissions(monitoring_data.get('gas', {}))
        thermal_analysis = self._analyze_thermal_data(monitoring_data.get('thermal', {}))
        
        # Integrate all parameters
        integrated_assessment = self._integrate_monitoring_parameters({
            'seismic': seismic_analysis,
            'deformation': deformation_analysis,
            'gas': gas_analysis,
            'thermal': thermal_analysis
        })
        
        # Calculate eruption probability
        eruption_probability = self._calculate_eruption_probability(integrated_assessment)
        
        # Determine alert level
        alert_level = self._determine_alert_level(integrated_assessment, eruption_probability)
        
        # Generate hazard maps
        hazard_maps = self._generate_hazard_maps(volcano_info, integrated_assessment)
        
        # Forecast timeline
        eruption_forecast = self._forecast_eruption_timeline(integrated_assessment)
        
        return {
            'volcano_id': volcano_info.get('volcano_id', 'unknown'),
            'assessment_timestamp': datetime.now().isoformat(),
            'current_alert_level': alert_level,
            'eruption_probability': eruption_probability,
            'parameter_analyses': {
                'seismic': seismic_analysis,
                'deformation': deformation_analysis,
                'gas': gas_analysis,
                'thermal': thermal_analysis
            },
            'integrated_assessment': integrated_assessment,
            'hazard_maps': hazard_maps,
            'eruption_forecast': eruption_forecast,
            'recommendations': self._generate_recommendations(alert_level, integrated_assessment)
        }
    
    def _analyze_seismic_activity(self, seismic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seismic activity patterns"""
        
        if not seismic_data:
            
class AdvancedSeismicProcessor:
    """Advanced seismic data processing with signal analysis and event detection"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        
    def process_seismic_stream(self, data: np.ndarray, 
                             station_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process continuous seismic data stream"""
        
        # Multi-component processing
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        processed_components = []
        
        for i, component_data in enumerate(data):
            # Apply preprocessing pipeline
            processed = self._preprocess_component(component_data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processed, component_data)
            
            # Detect events
            events = self._detect_events_in_component(processed)
            
            processed_components.append({
                'component_index': i,
                'component_name': ['Z', 'N', 'E'][i] if i < 3 else f'C{i}',
                'processed_data': processed,
                'quality_metrics': quality_metrics,
                'detected_events': events
            })
        
        return {
            'station_id': station_info.get('station_id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'sampling_rate': self.sampling_rate,
            'components': processed_components,
            'overall_quality': self._calculate_overall_quality(processed_components)
        }

        
        # Extract key metrics
        event_count = seismic_data.get('event_count_24h', 0)
        max_magnitude = seismic_data.get('max_magnitude_24h', 0)
        cumulative_energy = seismic_data.get('cumulative_energy', 0)
        b_value = seismic_data.get('b_value', 1.0)  # Gutenberg-Richter b-value
        
        # Analyze earthquake swarms
        swarm_analysis = self._analyze_earthquake_swarms(seismic_data.get('events', []))
        
        # Analyze frequency content
        frequency_analysis = self._analyze_seismic_frequencies(seismic_data.get('waveforms', []))
        
        # Calculate seismic risk score
        risk_factors = {
            'event_rate': min(1.0, event_count / 100),  # Normalize to 100 events/day
            'magnitude': min(1.0, max_magnitude / 5.0),  # Normalize to M5
            'energy': min(1.0, cumulative_energy / 1e12),  # Normalize to 10^12 J
            'b_value_anomaly': abs(1.0 - b_value),  # Deviation from normal b=1
            'swarm_activity': swarm_analysis['swarm_intensity'],
            'frequency_anomaly': frequency_analysis['anomaly_score']
        }
        
        seismic_risk = np.mean(list(risk_factors.values()))
        
        return {
            'status': 'analyzed',
            'risk_level': seismic_risk,
            'event_count_24h': event_count,
            'max_magnitude': max_magnitude,
            'cumulative_energy': cumulative_energy,
            'b_value': b_value,
            'swarm_analysis': swarm_analysis,
            'frequency_analysis': frequency_analysis,
            'risk_factors': risk_factors,
            'interpretation': self._interpret_seismic_activity(risk_factors)
        }
    
    def _analyze_ground_deformation(self, deformation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ground deformation patterns"""
        
        if not deformation_data:
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []

        
        # InSAR data analysis
        insar_data = deformation_data.get('insar', {})
        gps_data = deformation_data.get('gps', {})
        tilt_data = deformation_data.get('tilt', {})
        
        # Calculate deformation rates
        vertical_rate = insar_data.get('vertical_velocity_mm_year', 0)
        horizontal_rate = np.sqrt(
            gps_data.get('east_velocity_mm_year', 0)**2 + 
            gps_data.get('north_velocity_mm_year', 0)**2
        )
        
        # Analyze deformation patterns
        pattern_analysis = self._analyze_deformation_patterns(deformation_data)
        
        # Model magma intrusion
        intrusion_model = self._model_magma_intrusion(deformation_data)
        
        # Calculate deformation risk
        risk_factors = {
            'vertical_rate': min(1.0, abs(vertical_rate) / 100),  # Normalize to 100 mm/year
            'horizontal_rate': min(1.0, horizontal_rate / 50),    # Normalize to 50 mm/year
            'pattern_coherence': pattern_analysis['coherence'],
            'intrusion_probability': intrusion_model['probability'],
            'acceleration': pattern_analysis['acceleration_factor']
        }
        
        deformation_risk = np.mean(list(risk_factors.values()))
        
        return {
            'status': 'analyzed',
            'risk_level': deformation_risk,
            'vertical_velocity': vertical_rate,
            'horizontal_velocity': horizontal_rate,
            'pattern_analysis': pattern_analysis,
            'intrusion_model': intrusion_model,
            'risk_factors': risk_factors,
            'interpretation': self._interpret_deformation(risk_factors)
        }
    
    def _analyze_gas_emissions(self, gas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volcanic gas emissions"""
        
        if not gas_data:
            
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []

        
        # Key gas species
        so2_flux = gas_data.get('so2_flux_tons_day', 0)
        co2_flux = gas_data.get('co2_flux_tons_day', 0)
        h2s_ratio = gas_data.get('h2s_so2_ratio', 0)
        temperature = gas_data.get('fumarole_temperature_c', 0)
        
        # Calculate gas ratios
        co2_so2_ratio = co2_flux / (so2_flux + 1e-6)
        
        # Analyze temporal trends
        temporal_analysis = self._analyze_gas_trends(gas_data.get('time_series', []))
        
        # Calculate gas risk
        risk_factors = {
            'so2_flux': min(1.0, so2_flux / 1000),  # Normalize to 1000 tons/day
            'co2_flux': min(1.0, co2_flux / 5000),  # Normalize to 5000 tons/day
            'temperature': min(1.0, (temperature - 100) / 400),  # Normalize 100-500°C
            'ratio_anomaly': min(1.0, abs(co2_so2_ratio - 10) / 20),  # Typical ratio ~10
            'trend_significance': temporal_analysis['trend_significance']
        }
        
        gas_risk = np.mean(list(risk_factors.values()))
        
        return {
            'status': 'analyzed',
            'risk_level': gas_risk,
            'so2_flux': so2_flux,
            'co2_flux': co2_flux,
            'co2_so2_ratio': co2_so2_ratio,
            'fumarole_temperature': temperature,
            'temporal_analysis': temporal_analysis,
            'risk_factors': risk_factors,
            'interpretation': self._interpret_gas_emissions(risk_factors)
        }
    
    def _calculate_eruption_probability(self, integrated_assessment: Dict[str, Any]) -> Dict[str, float]:
        """Calculate eruption probability using integrated data"""
        
        # Weight different monitoring parameters
        weights = {
            'seismic': 0.3,
            'deformation': 0.25,
            'gas': 0.25,
            'thermal': 0.2
        }
        
        # Calculate weighted risk score
        total_risk = 0
        total_weight = 0
        
        for param, weight in weights.items():
            if param in integrated_assessment and integrated_assessment[param]['status'] == 'analyzed':
                total_risk += integrated_assessment[param]['risk_level'] * weight
                total_weight += weight
        
        if total_weight == 0:
            return {'24_hours': 0.0, '7_days': 0.0, '30_days': 0.0, '1_year': 0.0}
        
        base_probability = total_risk / total_weight
        
        # Time-dependent probabilities (simplified model)
        probabilities = {
            '24_hours': base_probability * 0.1,
            '7_days': base_probability * 0.3,
            '30_days': base_probability * 0.6,
            '1_year': min(1.0, base_probability * 1.2)
        }
        
        return probabilities

    
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