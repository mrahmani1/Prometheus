"""Integration module for sensor fusion with stream processing"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
import numpy as np
from collections import defaultdict, deque

from .sensor_fusion import MultiSensorDataFusion, FusionResult
from models.geological_data import (
    SeismicEvent, ThermalMeasurement, DeformationMeasurement,
    StrainMeasurement, AtmosphericMeasurement, GeologicalLocation,
    DataQuality
)

logger = structlog.get_logger()

class VolcanoWatchtowerFusion:
    """
    Integration class for multi-sensor data fusion in Volcano Watchtower
    real-time monitoring system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize fusion system
        fusion_config = {
            'temporal_window': config.get('fusion_temporal_window', 300),  # 5 minutes
            'spatial_threshold': config.get('fusion_spatial_threshold', 50.0),  # 50 km
            'outlier_threshold': config.get('fusion_outlier_threshold', 3.0)
        }
        
        self.fusion_system = MultiSensorDataFusion(fusion_config)
        
        # Data buffers for different sensor types
        self.sensor_buffers = {
            'seismic': deque(maxlen=config.get('seismic_buffer_size', 1000)),
            'thermal': deque(maxlen=config.get('thermal_buffer_size', 500)),
            'deformation': deque(maxlen=config.get('deformation_buffer_size', 200)),
            'strain': deque(maxlen=config.get('strain_buffer_size', 2000)),
            'atmospheric': deque(maxlen=config.get('atmospheric_buffer_size', 300))
        }
        
        # Fusion parameters
        self.fusion_interval = config.get('fusion_interval', 60)  # seconds
        self.min_sensors_for_fusion = config.get('min_sensors_for_fusion', 2)
        self.fusion_method = config.get('fusion_method', 'kalman')
        
        # Alert thresholds for fused data
        self.fusion_alert_thresholds = {
            'high_confidence_threshold': 0.8,
            'high_quality_threshold': 0.7,
            'significant_uncertainty_reduction': 0.3
        }
        
        # Statistics
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'alerts_generated': 0,
            'outliers_detected': 0
        }
        
        # Background task for periodic fusion
        self.fusion_task = None
        self.is_running = False
        
        logger.info("Volcano Watchtower fusion initialized", config=fusion_config)
    
    async def start_fusion_processing(self):
        """Start background fusion processing"""
        if self.is_running:
            logger.warning("Fusion processing already running")
            return
        
        self.is_running = True
        self.fusion_task = asyncio.create_task(self._fusion_processing_loop())
        
        logger.info("Started fusion processing", interval=self.fusion_interval)
    
    async def stop_fusion_processing(self):
        """Stop background fusion processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.fusion_task:
            self.fusion_task.cancel()
            try:
                await self.fusion_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped fusion processing")
    
    async def add_seismic_event(self, event_data: Dict[str, Any]):
        """Add seismic event to fusion buffer"""
        try:
            seismic_event = self._convert_to_seismic_event(event_data)
            if seismic_event:
                self.sensor_buffers['seismic'].append(seismic_event)
                
                logger.debug(
                    "Added seismic event to fusion buffer",
                    event_id=event_data.get('event_id'),
                    magnitude=event_data.get('magnitude'),
                    buffer_size=len(self.sensor_buffers['seismic'])
                )
        except Exception as e:
            logger.error("Error adding seismic event", error=str(e))
    
    async def add_thermal_measurement(self, measurement_data: Dict[str, Any]):
        """Add thermal measurement to fusion buffer"""
        try:
            thermal_measurement = self._convert_to_thermal_measurement(measurement_data)
            if thermal_measurement:
                self.sensor_buffers['thermal'].append(thermal_measurement)
                
                logger.debug(
                    "Added thermal measurement to fusion buffer",
                    temperature=measurement_data.get('temperature'),
                    buffer_size=len(self.sensor_buffers['thermal'])
                )
        except Exception as e:
            logger.error("Error adding thermal measurement", error=str(e))
    
    async def add_deformation_measurement(self, measurement_data: Dict[str, Any]):
        """Add deformation measurement to fusion buffer"""
        try:
            deformation_measurement = self._convert_to_deformation_measurement(measurement_data)
            if deformation_measurement:
                self.sensor_buffers['deformation'].append(deformation_measurement)
                
                logger.debug(
                    "Added deformation measurement to fusion buffer",
                    displacement=measurement_data.get('displacement_vector'),
                    buffer_size=len(self.sensor_buffers['deformation'])
                )
        except Exception as e:
            logger.error("Error adding deformation measurement", error=str(e))
    
    async def add_strain_measurement(self, measurement_data: Dict[str, Any]):
        """Add strain measurement to fusion buffer"""
        try:
            strain_measurement = self._convert_to_strain_measurement(measurement_data)
            if strain_measurement:
                self.sensor_buffers['strain'].append(strain_measurement)
                
                logger.debug(
                    "Added strain measurement to fusion buffer",
                    strain_rate=measurement_data.get('strain_rate'),
                    channel=measurement_data.get('channel_index'),
                    buffer_size=len(self.sensor_buffers['strain'])
                )
        except Exception as e:
            logger.error("Error adding strain measurement", error=str(e))
    
    async def add_atmospheric_measurement(self, measurement_data: Dict[str, Any]):
        """Add atmospheric measurement to fusion buffer"""
        try:
            atmospheric_measurement = self._convert_to_atmospheric_measurement(measurement_data)
            if atmospheric_measurement:
                self.sensor_buffers['atmospheric'].append(atmospheric_measurement)
                
                logger.debug(
                    "Added atmospheric measurement to fusion buffer",
                    pressure=measurement_data.get('pressure'),
                    temperature=measurement_data.get('temperature'),
                    buffer_size=len(self.sensor_buffers['atmospheric'])
                )
        except Exception as e:
            logger.error("Error adding atmospheric measurement", error=str(e))
    
    async def perform_immediate_fusion(
        self, 
        reference_location: Optional[GeologicalLocation] = None,
        reference_time: Optional[datetime] = None
    ) -> List[FusionResult]:
        """
        Perform immediate fusion of current buffer contents
        
        Args:
            reference_location: Reference location for spatial clustering
            reference_time: Reference time for temporal alignment
            
        Returns:
            List of fusion results
        """
        try:
            # Collect recent data from all buffers
            current_time = reference_time or datetime.utcnow()
            fusion_window_start = current_time - timedelta(seconds=self.fusion_system.config['temporal_window'])
            
            geological_data = []
            
            # Collect data from each sensor type
            for sensor_type, buffer in self.sensor_buffers.items():
                for data_item in buffer:
                    if data_item.timestamp >= fusion_window_start:
                        geological_data.append(data_item)
            
            if len(geological_data) < self.min_sensors_for_fusion:
                logger.debug(
                    "Insufficient data for fusion",
                    data_count=len(geological_data),
                    min_required=self.min_sensors_for_fusion
                )
                return []
            
            # Perform fusion
            self.fusion_stats['total_fusions'] += 1
            
            fusion_results = self.fusion_system.fuse_sensors(
                geological_data,
                fusion_method=self.fusion_method,
                reference_location=reference_location,
                reference_time=reference_time
            )
            
            if fusion_results:
                self.fusion_stats['successful_fusions'] += 1
                
                # Count outliers
                total_outliers = sum(len(result.outliers_rejected) for result in fusion_results)
                self.fusion_stats['outliers_detected'] += total_outliers
                
                # Check for alerts
                for result in fusion_results:
                    if await self._check_fusion_alerts(result):
                        self.fusion_stats['alerts_generated'] += 1
            
            logger.info(
                "Immediate fusion completed",
                input_data_count=len(geological_data),
                fusion_results_count=len(fusion_results),
                method=self.fusion_method
            )
            
            return fusion_results
            
        except Exception as e:
            logger.error("Error in immediate fusion", error=str(e))
            return []
    
    async def _fusion_processing_loop(self):
        """Background loop for periodic fusion processing"""
        while self.is_running:
            try:
                # Perform periodic fusion
                fusion_results = await self.perform_immediate_fusion()
                
                if fusion_results:
                    await self._process_fusion_results(fusion_results)
                
                # Wait for next fusion interval
                await asyncio.sleep(self.fusion_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in fusion processing loop", error=str(e))
                await asyncio.sleep(self.fusion_interval * 2)  # Wait longer on error
    
    async def _process_fusion_results(self, fusion_results: List[FusionResult]):
        """Process fusion results and generate alerts if needed"""
        try:
            for result in fusion_results:
                # Log fusion result
                logger.info(
                    "Fusion result processed",
                    contributing_sensors=result.contributing_sensors,
                    fusion_quality=result.fusion_quality,
                    confidence_level=result.confidence_level,
                    uncertainty_reduction=result.uncertainty_reduction,
                    outliers_count=len(result.outliers_rejected)
                )
                
                # Check for significant findings
                if result.confidence_level >= self.fusion_alert_thresholds['high_confidence_threshold']:
                    await self._generate_high_confidence_alert(result)
                
                if result.fusion_quality >= self.fusion_alert_thresholds['high_quality_threshold']:
                    await self._generate_high_quality_alert(result)
                
                if result.uncertainty_reduction >= self.fusion_alert_thresholds['significant_uncertainty_reduction']:
                    await self._generate_uncertainty_reduction_alert(result)
                
                # Process outliers
                if result.outliers_rejected:
                    await self._process_outliers(result.outliers_rejected)
                
        except Exception as e:
            logger.error("Error processing fusion results", error=str(e))
    
    async def _check_fusion_alerts(self, fusion_result: FusionResult) -> bool:
        """
        Check if fusion result should trigger alerts
        
        Args:
            fusion_result: Fusion result to check
            
        Returns:
            True if alert should be generated
        """
        try:
            # High confidence detection
            if fusion_result.confidence_level >= self.fusion_alert_thresholds['high_confidence_threshold']:
                return True
            
            # High quality fusion with multiple sensors
            if (fusion_result.fusion_quality >= self.fusion_alert_thresholds['high_quality_threshold'] and
                len(fusion_result.contributing_sensors) >= 3):
                return True
            
            # Significant uncertainty reduction
            if fusion_result.uncertainty_reduction >= self.fusion_alert_thresholds['significant_uncertainty_reduction']:
                return True
            
            # Multiple outliers detected (potential systematic issue)
            if len(fusion_result.outliers_rejected) >= 5:
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error checking fusion alerts", error=str(e))
            return False
    
    async def _generate_high_confidence_alert(self, fusion_result: FusionResult):
        """Generate alert for high confidence fusion result"""
        try:
            alert = {
                'type': 'high_confidence_fusion',
                'confidence_level': fusion_result.confidence_level,
                'contributing_sensors': fusion_result.contributing_sensors,
                'location': {
                    'latitude': fusion_result.fused_state.location.latitude,
                    'longitude': fusion_result.fused_state.location.longitude,
                    'elevation': fusion_result.fused_state.location.elevation
                },
                'timestamp': fusion_result.fused_state.timestamp.isoformat(),
                'fusion_quality': fusion_result.fusion_quality,
                'uncertainty_reduction': fusion_result.uncertainty_reduction,
                'severity': 'high' if fusion_result.confidence_level > 0.9 else 'medium'
            }
            
            logger.warning(
                "High confidence fusion detected",
                alert=alert
            )
            
            # TODO: Send alert to monitoring system
            
        except Exception as e:
            logger.error("Error generating high confidence alert", error=str(e))
    
    async def _generate_high_quality_alert(self, fusion_result: FusionResult):
        """Generate alert for high quality fusion result"""
        try:
            alert = {
                'type': 'high_quality_fusion',
                'fusion_quality': fusion_result.fusion_quality,
                'contributing_sensors': fusion_result.contributing_sensors,
                'sensor_count': len(fusion_result.contributing_sensors),
                'location': {
                    'latitude': fusion_result.fused_state.location.latitude,
                    'longitude': fusion_result.fused_state.location.longitude
                },
                'timestamp': fusion_result.fused_state.timestamp.isoformat(),
                'confidence_level': fusion_result.confidence_level,
                'severity': 'medium'
            }
            
            logger.info(
                "High quality fusion detected",
                alert=alert
            )
            
            # TODO: Send alert to monitoring system
            
        except Exception as e:
            logger.error("Error generating high quality alert", error=str(e))
    
    async def _generate_uncertainty_reduction_alert(self, fusion_result: FusionResult):
        """Generate alert for significant uncertainty reduction"""
        try:
            alert = {
                'type': 'significant_uncertainty_reduction',
                'uncertainty_reduction': fusion_result.uncertainty_reduction,
                'contributing_sensors': fusion_result.contributing_sensors,
                'location': {
                    'latitude': fusion_result.fused_state.location.latitude,
                    'longitude': fusion_result.fused_state.location.longitude
                },
                'timestamp': fusion_result.fused_state.timestamp.isoformat(),
                'confidence_level': fusion_result.confidence_level,
                'fusion_quality': fusion_result.fusion_quality,
                'severity': 'low'
            }
            
            logger.info(
                "Significant uncertainty reduction achieved",
                alert=alert
            )
            
            # TODO: Send alert to monitoring system
            
        except Exception as e:
            logger.error("Error generating uncertainty reduction alert", error=str(e))
    
    async def _process_outliers(self, outliers: List[Any]):
        """Process detected outliers"""
        try:
            for outlier in outliers:
                logger.warning(
                    "Outlier detected in sensor fusion",
                    sensor_type=outlier.sensor_type,
                    source=outlier.source,
                    timestamp=outlier.timestamp.isoformat(),
                    mahalanobis_distance=outlier.metadata.get('mahalanobis_distance'),
                    outlier_reason=outlier.metadata.get('outlier_reason')
                )
                
                # TODO: Flag sensor for quality check
                # TODO: Update sensor reliability metrics
            
        except Exception as e:
            logger.error("Error processing outliers", error=str(e))
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion processing statistics"""
        return {
            **self.fusion_stats,
            'buffer_sizes': {
                sensor_type: len(buffer) 
                for sensor_type, buffer in self.sensor_buffers.items()
            },
            'success_rate': (
                self.fusion_stats['successful_fusions'] / 
                max(self.fusion_stats['total_fusions'], 1)
            ),
            'is_running': self.is_running,
            'fusion_method': self.fusion_method,
            'fusion_interval': self.fusion_interval
        }
    
    def _convert_to_seismic_event(self, event_data: Dict[str, Any]) -> Optional[SeismicEvent]:
        """Convert event data to SeismicEvent object"""
        try:
            location_data = event_data.get('location', {})
            location = GeologicalLocation(
                latitude=location_data.get('latitude', 0.0),
                longitude=location_data.get('longitude', 0.0),
                elevation=location_data.get('elevation', 0.0),
                depth=location_data.get('depth')
            )
            
            return SeismicEvent(
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                location=location,
                magnitude=event_data.get('magnitude', 0.0),
                depth=event_data.get('depth', 0.0),
                event_type=event_data.get('event_type', 'unknown'),
                event_id=event_data.get('event_id', ''),
                source=event_data.get('source', ''),
                quality=DataQuality(event_data.get('quality', 'unknown'))
            )
            
        except Exception as e:
            logger.error("Error converting seismic event", error=str(e))
            return None
    
    def _convert_to_thermal_measurement(self, measurement_data: Dict[str, Any]) -> Optional[ThermalMeasurement]:
        """Convert measurement data to ThermalMeasurement object"""
        try:
            location_data = measurement_data.get('location', {})
            location = GeologicalLocation(
                latitude=location_data.get('latitude', 0.0),
                longitude=location_data.get('longitude', 0.0),
                elevation=location_data.get('elevation', 0.0)
            )
            
            return ThermalMeasurement(
                timestamp=datetime.fromisoformat(measurement_data['timestamp']),
                location=location,
                temperature=measurement_data.get('temperature', 0.0),
                measurement_method=measurement_data.get('measurement_method', 'unknown'),
                spatial_resolution=measurement_data.get('spatial_resolution', 1.0),
                uncertainty=measurement_data.get('uncertainty', 1.0),
                source=measurement_data.get('source', ''),
                quality=DataQuality(measurement_data.get('quality', 'unknown'))
            )
            
        except Exception as e:
            logger.error("Error converting thermal measurement", error=str(e))
            return None
    
    def _convert_to_deformation_measurement(self, measurement_data: Dict[str, Any]) -> Optional[DeformationMeasurement]:
        """Convert measurement data to DeformationMeasurement object"""
        try:
            location_data = measurement_data.get('location', {})
            location = GeologicalLocation(
                latitude=location_data.get('latitude', 0.0),
                longitude=location_data.get('longitude', 0.0),
                elevation=location_data.get('elevation', 0.0)
            )
            
            displacement_vector = measurement_data.get('displacement_vector', [0.0, 0.0, 0.0])
            if not isinstance(displacement_vector, np.ndarray):
                displacement_vector = np.array(displacement_vector)
            
            uncertainty = measurement_data.get('uncertainty', [0.001, 0.001, 0.001])
            if not isinstance(uncertainty, np.ndarray):
                uncertainty = np.array(uncertainty)
            
            return DeformationMeasurement(
                timestamp=datetime.fromisoformat(measurement_data['timestamp']),
                location=location,
                displacement_vector=displacement_vector,
                measurement_type=measurement_data.get('measurement_type', 'unknown'),
                uncertainty=uncertainty,
                reference_frame=measurement_data.get('reference_frame', 'ITRF2014'),
                source=measurement_data.get('source', ''),
                quality=DataQuality(measurement_data.get('quality', 'unknown'))
            )
            
        except Exception as e:
            logger.error("Error converting deformation measurement", error=str(e))
            return None
    
    def _convert_to_strain_measurement(self, measurement_data: Dict[str, Any]) -> Optional[StrainMeasurement]:
        """Convert measurement data to StrainMeasurement object"""
        try:
            location_data = measurement_data.get('location', {})
            location = GeologicalLocation(
                latitude=location_data.get('latitude', 0.0),
                longitude=location_data.get('longitude', 0.0),
                elevation=location_data.get('elevation', 0.0)
            )
            
            return StrainMeasurement(
                timestamp=datetime.fromisoformat(measurement_data['timestamp']),
                location=location,
                strain_rate=measurement_data.get('strain_rate', 0.0),
                strain_rate_std=measurement_data.get('strain_rate_std', 0.0),
                strain_rate_max=measurement_data.get('strain_rate_max', 0.0),
                measurement_type=measurement_data.get('measurement_type', 'das'),
                channel_index=measurement_data.get('channel_index', 0),
                gauge_length=measurement_data.get('gauge_length', 10.0),
                sampling_rate=measurement_data.get('sampling_rate', 1000.0),
                source=measurement_data.get('source', ''),
                quality=DataQuality(measurement_data.get('quality', 'unknown'))
            )
            
        except Exception as e:
            logger.error("Error converting strain measurement", error=str(e))
            return None
    
    def _convert_to_atmospheric_measurement(self, measurement_data: Dict[str, Any]) -> Optional[AtmosphericMeasurement]:
        """Convert measurement data to AtmosphericMeasurement object"""
        try:
            location_data = measurement_data.get('location', {})
            location = GeologicalLocation(
                latitude=location_data.get('latitude', 0.0),
                longitude=location_data.get('longitude', 0.0),
                elevation=location_data.get('elevation', 0.0)
            )
            
            return AtmosphericMeasurement(
                timestamp=datetime.fromisoformat(measurement_data['timestamp']),
                location=location,
                pressure=measurement_data.get('pressure'),
                temperature=measurement_data.get('temperature'),
                humidity=measurement_data.get('humidity'),
                wind_speed=measurement_data.get('wind_speed'),
                wind_direction=measurement_data.get('wind_direction'),
                gas_concentrations=measurement_data.get('gas_concentrations'),
                source=measurement_data.get('source', ''),
                quality=DataQuality(measurement_data.get('quality', 'unknown'))
            )
            
        except Exception as e:
            logger.error("Error converting atmospheric measurement", error=str(e))
            return None