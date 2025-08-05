"""
InSAR Integration Module

This module integrates the InSAR deformation analysis system with the existing
volcano watchtower components, providing seamless data flow and alert generation
based on deformation patterns.

Requirements addressed: 3.3, 1.2
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np

from .insar_analysis import (
    InSARDeformationAnalyzer, InSARData, DeformationResult,
    load_insar_data, create_synthetic_insar_time_series
)
from .alert_system import AlertLevelManager
from .risk_assessment import VolcanicRiskAssessmentEngine
from .magma_reservoir import MagmaReservoirEstimator

logger = logging.getLogger(__name__)


class InSARMonitoringService:
    """
    Service for continuous InSAR monitoring and integration with volcano watchtower.
    Processes InSAR data streams and generates alerts based on deformation patterns.
    """
    
    def __init__(self, analyzer: Optional[InSARDeformationAnalyzer] = None):
        self.analyzer = analyzer or InSARDeformationAnalyzer()
        self.alert_manager = AlertLevelManager()
        self.risk_engine = VolcanicRiskAssessmentEngine()
        self.magma_estimator = MagmaReservoirEstimator()
        
        # Monitoring configuration
        self.deformation_thresholds = {
            'minor': 0.02,      # 2 cm
            'moderate': 0.05,   # 5 cm
            'major': 0.10,      # 10 cm
            'extreme': 0.20     # 20 cm
        }
        
        self.velocity_thresholds = {
            'minor': 0.05,      # 5 cm/year
            'moderate': 0.15,   # 15 cm/year
            'major': 0.30,      # 30 cm/year
            'extreme': 0.60     # 60 cm/year
        }
        
        # Historical data storage
        self.deformation_history: List[DeformationResult] = []
        self.time_series_data: List[InSARData] = []
        
        logger.info("InSAR monitoring service initialized")
    
    async def process_new_insar_data(self, insar_data: InSARData, volcano_id: str) -> Dict:
        """
        Process new InSAR data and generate alerts if necessary.
        
        Args:
            insar_data: New InSAR interferogram data
            volcano_id: Identifier for the monitored volcano
            
        Returns:
            Processing results and any generated alerts
        """
        logger.info(f"Processing new InSAR data for volcano {volcano_id}")
        
        try:
            # Process the InSAR data
            result = self.analyzer.process_insar_data(insar_data)
            
            # Store in history
            self.deformation_history.append(result)
            self.time_series_data.append(insar_data)
            
            # Analyze deformation patterns and generate alerts
            alerts = await self._analyze_deformation_and_alert(result, volcano_id)
            
            # Update magma reservoir model with deformation data
            await self._update_magma_model(result, volcano_id)
            
            # Compile processing summary
            processing_summary = {
                'volcano_id': volcano_id,
                'processing_time': datetime.now(),
                'max_displacement_m': float(np.nanmax(np.abs(result.displacement_map))),
                'mean_uncertainty_m': float(np.nanmean(result.uncertainty_map)),
                'quality_score': result.quality_metrics['overall_quality_score'],
                'patterns_detected': len(result.deformation_patterns),
                'alerts_generated': len(alerts),
                'alert_levels': [alert['level'] for alert in alerts]
            }
            
            logger.info(f"InSAR processing completed for {volcano_id}: "
                       f"{processing_summary['max_displacement_m']:.3f}m max displacement, "
                       f"{len(alerts)} alerts generated")
            
            return {
                'deformation_result': result,
                'alerts': alerts,
                'processing_summary': processing_summary
            }
            
        except Exception as e:
            logger.error(f"Error processing InSAR data for {volcano_id}: {e}")
            raise
    
    async def process_time_series_update(self, volcano_id: str, max_acquisitions: int = 50) -> Dict:
        """
        Process accumulated time series data to estimate velocity and trends.
        
        Args:
            volcano_id: Volcano identifier
            max_acquisitions: Maximum number of acquisitions to use
            
        Returns:
            Time series analysis results
        """
        if len(self.time_series_data) < 3:
            logger.warning(f"Insufficient time series data for {volcano_id} "
                          f"({len(self.time_series_data)} acquisitions)")
            return {'status': 'insufficient_data'}
        
        logger.info(f"Processing time series for {volcano_id} "
                   f"with {len(self.time_series_data)} acquisitions")
        
        # Use most recent acquisitions
        recent_data = self.time_series_data[-max_acquisitions:]
        
        # Process time series
        result = self.analyzer.process_time_series(recent_data)
        
        # Analyze velocity patterns
        velocity_alerts = await self._analyze_velocity_patterns(result, volcano_id)
        
        # Detect acceleration/deceleration
        trend_analysis = self._analyze_deformation_trends(result)
        
        return {
            'velocity_result': result,
            'velocity_alerts': velocity_alerts,
            'trend_analysis': trend_analysis,
            'time_series_length': len(recent_data)
        }
    
    async def _analyze_deformation_and_alert(self, result: DeformationResult, volcano_id: str) -> List[Dict]:
        """Analyze deformation patterns and generate appropriate alerts"""
        alerts = []
        
        # Check maximum displacement
        max_displacement = np.nanmax(np.abs(result.displacement_map))
        displacement_alert_level = self._get_displacement_alert_level(max_displacement)
        
        if displacement_alert_level > 0:
            alerts.append({
                'type': 'displacement_threshold',
                'level': displacement_alert_level,
                'message': f"Maximum displacement of {max_displacement:.3f}m detected",
                'location': self._find_max_displacement_location(result.displacement_map),
                'confidence': result.quality_metrics['overall_quality_score']
            })
        
        # Analyze specific deformation patterns
        for pattern in result.deformation_patterns:
            pattern_alert = self._assess_pattern_significance(pattern, volcano_id)
            if pattern_alert:
                alerts.append(pattern_alert)
        
        # Check data quality issues
        if result.quality_metrics['overall_quality_score'] < 0.3:
            alerts.append({
                'type': 'data_quality_warning',
                'level': 1,
                'message': f"Low data quality detected (score: {result.quality_metrics['overall_quality_score']:.2f})",
                'confidence': result.quality_metrics['overall_quality_score']
            })
        
        # Generate formal alerts through alert manager
        for alert in alerts:
            if alert['level'] >= 2:  # Only send moderate or higher alerts
                # Create AlertLevel object
                from models.volcanic_data import AlertLevel
                alert_level = AlertLevel(
                    level=alert['level'],
                    description=alert['message'],
                    recommended_actions=[f"Monitor {alert['type']}"],
                    validity_period=(datetime.now(), datetime.now() + timedelta(hours=6)),
                    confidence=alert.get('confidence', 0.8)
                )
                await self.alert_manager.update_alert_level(volcano_id, alert_level)
        
        return alerts
    
    async def _analyze_velocity_patterns(self, result: DeformationResult, volcano_id: str) -> List[Dict]:
        """Analyze velocity patterns and generate velocity-based alerts"""
        alerts = []
        
        if result.velocity_map is None:
            return alerts
        
        # Check maximum velocity
        max_velocity = np.nanmax(np.abs(result.velocity_map))
        velocity_alert_level = self._get_velocity_alert_level(max_velocity)
        
        if velocity_alert_level > 0:
            alerts.append({
                'type': 'velocity_threshold',
                'level': velocity_alert_level,
                'message': f"Maximum velocity of {max_velocity:.3f}m/year detected",
                'location': self._find_max_velocity_location(result.velocity_map),
                'confidence': result.quality_metrics['overall_quality_score']
            })
        
        # Check for accelerating deformation
        acceleration_zones = self._detect_acceleration_zones(result.velocity_map)
        if len(acceleration_zones) > 0:
            alerts.append({
                'type': 'acceleration_detected',
                'level': 2,
                'message': f"Accelerating deformation detected in {len(acceleration_zones)} zones",
                'zones': acceleration_zones,
                'confidence': 0.8
            })
        
        return alerts
    
    def _analyze_deformation_trends(self, result: DeformationResult) -> Dict:
        """Analyze long-term deformation trends"""
        if len(self.deformation_history) < 5:
            return {'status': 'insufficient_history'}
        
        # Extract time series of maximum displacements
        max_displacements = []
        timestamps = []
        
        for hist_result in self.deformation_history[-20:]:  # Last 20 results
            max_displacements.append(np.nanmax(np.abs(hist_result.displacement_map)))
            timestamps.append(hist_result.processing_metadata['processing_date'])
        
        max_displacements = np.array(max_displacements)
        
        # Fit linear trend
        if len(max_displacements) >= 3:
            time_days = np.array([(t - timestamps[0]).days for t in timestamps])
            try:
                trend_slope = np.polyfit(time_days, max_displacements, 1)[0]
            except np.linalg.LinAlgError:
                # Handle numerical issues
                trend_slope = 0.0
            
            trend_analysis = {
                'trend_slope_m_per_day': float(trend_slope),
                'trend_slope_m_per_year': float(trend_slope * 365.25),
                'current_max_displacement': float(max_displacements[-1]),
                'displacement_change_30d': float(max_displacements[-1] - max_displacements[max(-30, -len(max_displacements))]),
                'trend_direction': 'increasing' if trend_slope > 0.001 else 'decreasing' if trend_slope < -0.001 else 'stable'
            }
        else:
            trend_analysis = {'status': 'insufficient_data_for_trend'}
        
        return trend_analysis
    
    async def _update_magma_model(self, result: DeformationResult, volcano_id: str):
        """Update magma reservoir model based on deformation patterns"""
        try:
            # Extract inflation/deflation patterns
            inflation_patterns = [p for p in result.deformation_patterns 
                                if 'inflation' in p.get('pattern_type', '')]
            deflation_patterns = [p for p in result.deformation_patterns 
                                if 'deflation' in p.get('pattern_type', '')]
            
            # Update magma reservoir parameters
            for pattern in inflation_patterns:
                if 'location' in pattern and 'magnitude_m' in pattern:
                    # In a real implementation, this would update the magma reservoir model
                    # For now, just log the update
                    logger.debug(f"Would update magma reservoir at {pattern['location']} with pressure change {pattern['magnitude_m'] * 1e6} Pa")
            
            for pattern in deflation_patterns:
                if 'location' in pattern and 'magnitude_m' in pattern:
                    # In a real implementation, this would update the magma reservoir model
                    # For now, just log the update
                    logger.debug(f"Would update magma reservoir at {pattern['location']} with pressure change {-abs(pattern['magnitude_m']) * 1e6} Pa")
            
            logger.debug(f"Updated magma model for {volcano_id} with {len(inflation_patterns + deflation_patterns)} patterns")
            
        except Exception as e:
            logger.warning(f"Could not update magma model for {volcano_id}: {e}")
    
    def _get_displacement_alert_level(self, displacement: float) -> int:
        """Determine alert level based on displacement magnitude"""
        if displacement >= self.deformation_thresholds['extreme']:
            return 4
        elif displacement >= self.deformation_thresholds['major']:
            return 3
        elif displacement >= self.deformation_thresholds['moderate']:
            return 2
        elif displacement >= self.deformation_thresholds['minor']:
            return 1
        else:
            return 0
    
    def _get_velocity_alert_level(self, velocity: float) -> int:
        """Determine alert level based on velocity magnitude"""
        if velocity >= self.velocity_thresholds['extreme']:
            return 4
        elif velocity >= self.velocity_thresholds['major']:
            return 3
        elif velocity >= self.velocity_thresholds['moderate']:
            return 2
        elif velocity >= self.velocity_thresholds['minor']:
            return 1
        else:
            return 0
    
    def _find_max_displacement_location(self, displacement_map: np.ndarray) -> Tuple[int, int]:
        """Find location of maximum displacement"""
        max_idx = np.nanargmax(np.abs(displacement_map))
        return np.unravel_index(max_idx, displacement_map.shape)
    
    def _find_max_velocity_location(self, velocity_map: np.ndarray) -> Tuple[int, int]:
        """Find location of maximum velocity"""
        max_idx = np.nanargmax(np.abs(velocity_map))
        return np.unravel_index(max_idx, velocity_map.shape)
    
    def _assess_pattern_significance(self, pattern: Dict, volcano_id: str) -> Optional[Dict]:
        """Assess significance of a deformation pattern and generate alert if needed"""
        pattern_type = pattern.get('pattern_type', '')
        confidence = pattern.get('confidence', 0.0)
        
        # Only alert on high-confidence patterns
        if confidence < 0.6:
            return None
        
        # Assess different pattern types
        if pattern_type in ['point_inflation', 'inflation']:
            magnitude = pattern.get('magnitude_m', 0)
            if magnitude > 0.03:  # 3 cm threshold
                return {
                    'type': 'inflation_pattern',
                    'level': 2 if magnitude > 0.05 else 1,
                    'message': f"Inflation pattern detected: {magnitude:.3f}m",
                    'pattern': pattern,
                    'confidence': confidence
                }
        
        elif pattern_type in ['point_deflation', 'deflation']:
            magnitude = abs(pattern.get('magnitude_m', 0))
            if magnitude > 0.03:  # 3 cm threshold
                return {
                    'type': 'deflation_pattern',
                    'level': 2 if magnitude > 0.05 else 1,
                    'message': f"Deflation pattern detected: {magnitude:.3f}m",
                    'pattern': pattern,
                    'confidence': confidence
                }
        
        elif pattern_type in ['dike_intrusion', 'sill_intrusion']:
            return {
                'type': 'intrusion_pattern',
                'level': 3,  # Intrusions are significant
                'message': f"{pattern_type.replace('_', ' ').title()} pattern detected",
                'pattern': pattern,
                'confidence': confidence
            }
        
        return None
    
    def _detect_acceleration_zones(self, velocity_map: np.ndarray) -> List[Dict]:
        """Detect zones of accelerating deformation"""
        # This is a simplified implementation
        # In practice, this would compare current velocity with historical velocity
        
        if len(self.deformation_history) < 2:
            return []
        
        # Compare with previous velocity map if available
        prev_result = self.deformation_history[-2]
        if prev_result.velocity_map is None:
            return []
        
        # Calculate velocity change
        velocity_change = velocity_map - prev_result.velocity_map
        
        # Find areas with significant acceleration (>50% increase)
        acceleration_threshold = 0.02  # 2 cm/year acceleration
        acceleration_mask = np.abs(velocity_change) > acceleration_threshold
        
        # Find connected components of acceleration
        from scipy import ndimage
        labeled_zones, num_zones = ndimage.label(acceleration_mask)
        
        zones = []
        for zone_id in range(1, num_zones + 1):
            zone_mask = labeled_zones == zone_id
            zone_area = np.sum(zone_mask)
            
            if zone_area > 10:  # Minimum 10 pixels
                zone_center = ndimage.center_of_mass(zone_mask)
                max_acceleration = np.max(np.abs(velocity_change[zone_mask]))
                
                zones.append({
                    'zone_id': zone_id,
                    'center': zone_center,
                    'area_pixels': int(zone_area),
                    'max_acceleration_m_per_year': float(max_acceleration)
                })
        
        return zones
    
    def get_monitoring_status(self, volcano_id: str) -> Dict:
        """Get current monitoring status and statistics"""
        if not self.deformation_history:
            return {'status': 'no_data'}
        
        latest_result = self.deformation_history[-1]
        
        status = {
            'volcano_id': volcano_id,
            'last_update': latest_result.processing_metadata.get('processing_date'),
            'total_acquisitions': len(self.time_series_data),
            'current_max_displacement_m': float(np.nanmax(np.abs(latest_result.displacement_map))),
            'current_quality_score': latest_result.quality_metrics['overall_quality_score'],
            'active_patterns': len(latest_result.deformation_patterns),
            'monitoring_duration_days': None
        }
        
        if len(self.time_series_data) > 1:
            first_date = self.time_series_data[0].acquisition_dates[0]
            last_date = self.time_series_data[-1].acquisition_dates[1]
            status['monitoring_duration_days'] = (last_date - first_date).days
        
        # Add velocity information if available
        if latest_result.velocity_map is not None:
            status['current_max_velocity_m_per_year'] = float(np.nanmax(np.abs(latest_result.velocity_map)))
        
        return status


# Example usage and testing functions
async def example_insar_monitoring():
    """Example of how to use the InSAR monitoring service"""
    
    # Initialize monitoring service
    monitoring_service = InSARMonitoringService()
    
    # Simulate processing multiple InSAR acquisitions
    volcano_id = "test_volcano_001"
    
    # Process individual acquisitions
    for i in range(5):
        # Load synthetic InSAR data
        metadata = {'acquisition_number': i, 'sensor': 'Sentinel-1'}
        insar_data = load_insar_data(f"synthetic_{i}.tif", metadata)
        
        # Process the data
        result = await monitoring_service.process_new_insar_data(insar_data, volcano_id)
        
        print(f"Processed acquisition {i}:")
        print(f"  Max displacement: {result['processing_summary']['max_displacement_m']:.3f}m")
        print(f"  Alerts generated: {result['processing_summary']['alerts_generated']}")
        print(f"  Quality score: {result['processing_summary']['quality_score']:.2f}")
        print()
    
    # Process time series
    time_series_result = await monitoring_service.process_time_series_update(volcano_id)
    print("Time series analysis:")
    print(f"  Time series length: {time_series_result.get('time_series_length', 'N/A')}")
    print(f"  Velocity alerts: {len(time_series_result.get('velocity_alerts', []))}")
    
    # Get monitoring status
    status = monitoring_service.get_monitoring_status(volcano_id)
    print("\nMonitoring status:")
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_insar_monitoring())