"""
Real-time Hazard Map Updates System

This module handles real-time updates to hazard maps based on incoming monitoring data,
providing continuous assessment and alerting capabilities.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .hazard_mapping import (
    HazardMappingEngine, MultiHazardAssessment, GeographicBounds,
    HazardType, RiskLevel, HazardZone
)

logger = logging.getLogger(__name__)


@dataclass
class MonitoringDataStream:
    """Real-time monitoring data stream"""
    stream_id: str
    data_type: str
    timestamp: datetime
    location: Dict[str, float]  # lat, lon, elevation
    value: float
    unit: str
    quality_flag: str
    sensor_id: str


@dataclass
class HazardAlert:
    """Hazard alert for emergency response"""
    alert_id: str
    timestamp: datetime
    hazard_type: HazardType
    risk_level: RiskLevel
    affected_area: GeographicBounds
    population_at_risk: int
    recommended_actions: List[str]
    confidence: float
    expires_at: datetime
    escalation_level: int


@dataclass
class UpdateTrigger:
    """Conditions that trigger hazard map updates"""
    trigger_id: str
    data_type: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    time_window: timedelta
    geographic_bounds: Optional[GeographicBounds] = None


class RealTimeHazardUpdater:
    """Manages real-time hazard map updates and alerting"""
    
    def __init__(self, hazard_engine: HazardMappingEngine):
        self.hazard_engine = hazard_engine
        self.active_assessments: Dict[str, MultiHazardAssessment] = {}
        self.monitoring_streams: Dict[str, List[MonitoringDataStream]] = {}
        self.update_triggers: List[UpdateTrigger] = []
        self.alert_callbacks: List[Callable[[HazardAlert], None]] = []
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.update_queue = asyncio.Queue()
        self.is_running = False
        
        # Initialize default update triggers
        self._initialize_default_triggers()
    
    def _initialize_default_triggers(self):
        """Initialize default triggers for hazard map updates"""
        self.update_triggers = [
            # Seismic triggers
            UpdateTrigger(
                trigger_id="earthquake_magnitude",
                data_type="seismic_event",
                threshold_value=4.0,
                comparison_operator=">=",
                time_window=timedelta(minutes=5)
            ),
            UpdateTrigger(
                trigger_id="earthquake_cluster",
                data_type="seismic_event_count",
                threshold_value=5,
                comparison_operator=">=",
                time_window=timedelta(hours=1)
            ),
            
            # Volcanic triggers
            UpdateTrigger(
                trigger_id="ground_deformation",
                data_type="insar_displacement",
                threshold_value=0.05,  # 5 cm
                comparison_operator=">=",
                time_window=timedelta(hours=6)
            ),
            UpdateTrigger(
                trigger_id="gas_emission_spike",
                data_type="so2_flux",
                threshold_value=1000,  # tons/day
                comparison_operator=">=",
                time_window=timedelta(hours=2)
            ),
            UpdateTrigger(
                trigger_id="thermal_anomaly",
                data_type="thermal_temperature",
                threshold_value=100,  # Celsius
                comparison_operator=">=",
                time_window=timedelta(minutes=30)
            ),
            
            # Environmental triggers
            UpdateTrigger(
                trigger_id="heavy_precipitation",
                data_type="precipitation_rate",
                threshold_value=50,  # mm/hour
                comparison_operator=">=",
                time_window=timedelta(hours=3)
            ),
            UpdateTrigger(
                trigger_id="slope_instability",
                data_type="slope_movement",
                threshold_value=0.01,  # m/day
                comparison_operator=">=",
                time_window=timedelta(hours=12)
            )
        ]
    
    async def start_monitoring(self):
        """Start real-time monitoring and update processing"""
        if self.is_running:
            logger.warning("Real-time monitoring already running")
            return
        
        self.is_running = True
        logger.info("Starting real-time hazard monitoring...")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_update_queue()),
            asyncio.create_task(self._periodic_assessment_refresh()),
            asyncio.create_task(self._monitor_data_streams())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in real-time monitoring: {e}")
        finally:
            self.is_running = False
    
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        logger.info("Stopping real-time hazard monitoring...")
    
    async def ingest_monitoring_data(self, data_stream: MonitoringDataStream):
        """Ingest new monitoring data and check for update triggers"""
        # Store data in appropriate stream
        if data_stream.data_type not in self.monitoring_streams:
            self.monitoring_streams[data_stream.data_type] = []
        
        self.monitoring_streams[data_stream.data_type].append(data_stream)
        
        # Keep only recent data (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.monitoring_streams[data_stream.data_type] = [
            stream for stream in self.monitoring_streams[data_stream.data_type]
            if stream.timestamp > cutoff_time
        ]
        
        # Check if this data triggers an update
        await self._check_update_triggers(data_stream)
    
    async def _check_update_triggers(self, new_data: MonitoringDataStream):
        """Check if new data triggers hazard map updates"""
        for trigger in self.update_triggers:
            if trigger.data_type != new_data.data_type:
                continue
            
            # Get recent data for this trigger
            recent_data = self._get_recent_data(
                trigger.data_type, trigger.time_window
            )
            
            if self._evaluate_trigger_condition(trigger, recent_data, new_data):
                logger.info(f"Trigger activated: {trigger.trigger_id}")
                
                # Determine affected area
                affected_bounds = trigger.geographic_bounds or self._estimate_affected_area(
                    new_data, trigger
                )
                
                # Queue update
                await self.update_queue.put({
                    'trigger_id': trigger.trigger_id,
                    'bounds': affected_bounds,
                    'priority': self._calculate_update_priority(trigger, new_data),
                    'timestamp': datetime.now()
                })
    
    def _get_recent_data(
        self, 
        data_type: str, 
        time_window: timedelta
    ) -> List[MonitoringDataStream]:
        """Get recent data for a specific type within time window"""
        if data_type not in self.monitoring_streams:
            return []
        
        cutoff_time = datetime.now() - time_window
        return [
            stream for stream in self.monitoring_streams[data_type]
            if stream.timestamp > cutoff_time
        ]
    
    def _evaluate_trigger_condition(
        self,
        trigger: UpdateTrigger,
        recent_data: List[MonitoringDataStream],
        new_data: MonitoringDataStream
    ) -> bool:
        """Evaluate if trigger condition is met"""
        if trigger.data_type.endswith('_count'):
            # Count-based trigger
            value = len(recent_data)
        else:
            # Value-based trigger
            value = new_data.value
        
        # Evaluate condition
        if trigger.comparison_operator == '>=':
            return value >= trigger.threshold_value
        elif trigger.comparison_operator == '>':
            return value > trigger.threshold_value
        elif trigger.comparison_operator == '<=':
            return value <= trigger.threshold_value
        elif trigger.comparison_operator == '<':
            return value < trigger.threshold_value
        elif trigger.comparison_operator == '==':
            return value == trigger.threshold_value
        elif trigger.comparison_operator == '!=':
            return value != trigger.threshold_value
        
        return False
    
    def _estimate_affected_area(
        self,
        data_stream: MonitoringDataStream,
        trigger: UpdateTrigger
    ) -> GeographicBounds:
        """Estimate affected geographic area based on data and trigger type"""
        center_lat = data_stream.location['lat']
        center_lon = data_stream.location['lon']
        
        # Estimate radius based on trigger type and severity
        radius_map = {
            'seismic_event': 0.5,  # degrees
            'insar_displacement': 0.2,
            'so2_flux': 0.3,
            'thermal_temperature': 0.1,
            'precipitation_rate': 0.4,
            'slope_movement': 0.1
        }
        
        radius = radius_map.get(trigger.data_type, 0.2)
        
        # Adjust radius based on severity
        if hasattr(trigger, 'threshold_value') and data_stream.value > trigger.threshold_value:
            severity_factor = min(data_stream.value / trigger.threshold_value, 3.0)
            radius *= severity_factor
        
        return GeographicBounds(
            min_lat=center_lat - radius,
            max_lat=center_lat + radius,
            min_lon=center_lon - radius,
            max_lon=center_lon + radius,
            resolution=0.01
        )
    
    def _calculate_update_priority(
        self,
        trigger: UpdateTrigger,
        data_stream: MonitoringDataStream
    ) -> int:
        """Calculate priority for hazard map update (1-10, 10 = highest)"""
        base_priority = {
            'seismic_event': 8,
            'insar_displacement': 6,
            'so2_flux': 7,
            'thermal_temperature': 5,
            'precipitation_rate': 4,
            'slope_movement': 5
        }.get(trigger.data_type, 3)
        
        # Adjust based on severity
        if hasattr(trigger, 'threshold_value') and data_stream.value > trigger.threshold_value:
            severity_factor = min(data_stream.value / trigger.threshold_value, 2.0)
            base_priority = min(int(base_priority * severity_factor), 10)
        
        return base_priority
    
    async def _process_update_queue(self):
        """Process queued hazard map updates"""
        while self.is_running:
            try:
                # Wait for update request with timeout
                update_request = await asyncio.wait_for(
                    self.update_queue.get(), timeout=1.0
                )
                
                logger.info(f"Processing hazard update: {update_request['trigger_id']}")
                
                # Perform hazard assessment update
                await self._perform_hazard_update(update_request)
                
            except asyncio.TimeoutError:
                # No updates in queue, continue monitoring
                continue
            except Exception as e:
                logger.error(f"Error processing update queue: {e}")
                await asyncio.sleep(1)
    
    async def _perform_hazard_update(self, update_request: Dict[str, Any]):
        """Perform actual hazard map update"""
        try:
            bounds = update_request['bounds']
            
            # Prepare monitoring data for assessment
            monitoring_data = self._prepare_monitoring_data_for_bounds(bounds)
            
            # Perform multi-hazard assessment
            assessment = await self.hazard_engine.assess_multi_hazard_risk(
                geographic_bounds=bounds,
                monitoring_data=monitoring_data,
                time_horizon=timedelta(days=7)  # Short-term assessment for real-time
            )
            
            # Store updated assessment
            self.active_assessments[assessment.assessment_id] = assessment
            
            # Check for alert conditions
            await self._check_alert_conditions(assessment, update_request)
            
            logger.info(f"Hazard assessment updated: {assessment.assessment_id}")
            
        except Exception as e:
            logger.error(f"Error performing hazard update: {e}")
    
    def _prepare_monitoring_data_for_bounds(
        self, 
        bounds: GeographicBounds
    ) -> Dict[str, Any]:
        """Prepare monitoring data for hazard assessment within geographic bounds"""
        monitoring_data = {}
        
        # Filter data by geographic bounds
        for data_type, streams in self.monitoring_streams.items():
            filtered_streams = []
            for stream in streams:
                lat = stream.location['lat']
                lon = stream.location['lon']
                
                if (bounds.min_lat <= lat <= bounds.max_lat and
                    bounds.min_lon <= lon <= bounds.max_lon):
                    filtered_streams.append(stream)
            
            if filtered_streams:
                monitoring_data[data_type] = filtered_streams
        
        # Convert to format expected by hazard engine
        processed_data = {}
        
        # Process seismic data
        if 'seismic_event' in monitoring_data:
            seismic_events = [
                {
                    'magnitude': stream.value,
                    'timestamp': stream.timestamp,
                    'location': stream.location
                }
                for stream in monitoring_data['seismic_event']
            ]
            processed_data['seismic_events'] = seismic_events
            processed_data['recent_earthquakes'] = seismic_events
        
        # Process deformation data
        if 'insar_displacement' in monitoring_data:
            deformation_data = {
                'uplift_rate': max(
                    stream.value for stream in monitoring_data['insar_displacement']
                ),
                'measurements': [
                    {
                        'displacement': stream.value,
                        'timestamp': stream.timestamp,
                        'location': stream.location
                    }
                    for stream in monitoring_data['insar_displacement']
                ]
            }
            processed_data['ground_deformation'] = deformation_data
        
        # Process gas emission data
        if 'so2_flux' in monitoring_data:
            gas_data = {
                'so2_flux': max(
                    stream.value for stream in monitoring_data['so2_flux']
                ),
                'measurements': [
                    {
                        'flux': stream.value,
                        'timestamp': stream.timestamp,
                        'location': stream.location
                    }
                    for stream in monitoring_data['so2_flux']
                ]
            }
            processed_data['gas_emissions'] = gas_data
        
        # Process thermal data
        if 'thermal_temperature' in monitoring_data:
            thermal_data = {
                'max_temperature': max(
                    stream.value for stream in monitoring_data['thermal_temperature']
                ),
                'measurements': [
                    {
                        'temperature': stream.value,
                        'timestamp': stream.timestamp,
                        'location': stream.location
                    }
                    for stream in monitoring_data['thermal_temperature']
                ]
            }
            processed_data['thermal_data'] = thermal_data
        
        # Process precipitation data
        if 'precipitation_rate' in monitoring_data:
            precip_data = max(
                stream.value for stream in monitoring_data['precipitation_rate']
            )
            processed_data['precipitation'] = precip_data
        
        # Process slope movement data
        if 'slope_movement' in monitoring_data:
            slope_data = {
                'max_movement': max(
                    stream.value for stream in monitoring_data['slope_movement']
                ),
                'stability_factor': 1.0 - min(
                    stream.value for stream in monitoring_data['slope_movement']
                )
            }
            processed_data['slope_stability'] = slope_data['stability_factor']
        
        return processed_data
    
    async def _check_alert_conditions(
        self,
        assessment: MultiHazardAssessment,
        update_request: Dict[str, Any]
    ):
        """Check if assessment warrants issuing alerts"""
        # Define alert thresholds
        alert_thresholds = {
            RiskLevel.MODERATE: 0.3,
            RiskLevel.HIGH: 0.5,
            RiskLevel.VERY_HIGH: 0.7,
            RiskLevel.EXTREME: 0.9
        }
        
        # Check combined risk level
        max_risk = np.max(assessment.combined_risk_map)
        
        for risk_level, threshold in alert_thresholds.items():
            if max_risk >= threshold:
                await self._issue_hazard_alert(assessment, risk_level, update_request)
                break
    
    async def _issue_hazard_alert(
        self,
        assessment: MultiHazardAssessment,
        risk_level: RiskLevel,
        update_request: Dict[str, Any]
    ):
        """Issue hazard alert based on assessment"""
        # Determine primary hazard type
        primary_hazard = max(
            assessment.individual_hazards.items(),
            key=lambda x: x[1].probability
        )[0]
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            primary_hazard, risk_level
        )
        
        # Calculate alert expiration
        expires_at = datetime.now() + timedelta(hours=24)
        
        # Create alert
        alert = HazardAlert(
            alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            hazard_type=primary_hazard,
            risk_level=risk_level,
            affected_area=assessment.geographic_bounds,
            population_at_risk=assessment.total_population_at_risk,
            recommended_actions=recommended_actions,
            confidence=1.0 - np.mean(assessment.uncertainty_map),
            expires_at=expires_at,
            escalation_level=self._calculate_escalation_level(risk_level)
        )
        
        # Send alert to all registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Hazard alert issued: {alert.alert_id} - {risk_level.name}")
    
    def _generate_recommended_actions(
        self,
        hazard_type: HazardType,
        risk_level: RiskLevel
    ) -> List[str]:
        """Generate recommended actions based on hazard type and risk level"""
        base_actions = {
            HazardType.VOLCANIC_ERUPTION: [
                "Monitor volcanic activity closely",
                "Prepare evacuation plans",
                "Check emergency supplies"
            ],
            HazardType.SEISMIC_EARTHQUAKE: [
                "Secure heavy objects",
                "Review earthquake safety procedures",
                "Check building structural integrity"
            ],
            HazardType.LANDSLIDE: [
                "Avoid steep slopes",
                "Monitor drainage systems",
                "Prepare alternative routes"
            ],
            HazardType.TSUNAMI: [
                "Move to higher ground",
                "Stay away from coastlines",
                "Monitor official warnings"
            ]
        }
        
        actions = base_actions.get(hazard_type, ["Monitor conditions closely"])
        
        # Add risk-level specific actions
        if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
            actions.extend([
                "Consider evacuation",
                "Contact emergency services",
                "Activate emergency response plans"
            ])
        
        if risk_level == RiskLevel.EXTREME:
            actions.extend([
                "Immediate evacuation recommended",
                "Seek emergency shelter",
                "Follow official emergency instructions"
            ])
        
        return actions
    
    def _calculate_escalation_level(self, risk_level: RiskLevel) -> int:
        """Calculate escalation level for emergency response"""
        escalation_map = {
            RiskLevel.VERY_LOW: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.VERY_HIGH: 4,
            RiskLevel.EXTREME: 5
        }
        return escalation_map.get(risk_level, 2)
    
    async def _periodic_assessment_refresh(self):
        """Periodically refresh hazard assessments"""
        while self.is_running:
            try:
                # Refresh every 30 minutes
                await asyncio.sleep(1800)
                
                if not self.active_assessments:
                    continue
                
                logger.info("Performing periodic hazard assessment refresh...")
                
                # Refresh each active assessment
                for assessment_id, assessment in list(self.active_assessments.items()):
                    try:
                        # Check if assessment is still relevant (not too old)
                        age = datetime.now() - assessment.timestamp
                        if age > timedelta(hours=6):
                            # Remove old assessment
                            del self.active_assessments[assessment_id]
                            continue
                        
                        # Refresh assessment
                        monitoring_data = self._prepare_monitoring_data_for_bounds(
                            assessment.geographic_bounds
                        )
                        
                        updated_assessment = await self.hazard_engine.assess_multi_hazard_risk(
                            geographic_bounds=assessment.geographic_bounds,
                            monitoring_data=monitoring_data,
                            time_horizon=timedelta(days=7)
                        )
                        
                        self.active_assessments[updated_assessment.assessment_id] = updated_assessment
                        
                        # Remove old assessment
                        if updated_assessment.assessment_id != assessment_id:
                            del self.active_assessments[assessment_id]
                        
                    except Exception as e:
                        logger.error(f"Error refreshing assessment {assessment_id}: {e}")
                
                logger.info("Periodic assessment refresh completed")
                
            except Exception as e:
                logger.error(f"Error in periodic refresh: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitor_data_streams(self):
        """Monitor data stream health and quality"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                
                # Check for stale data streams
                for data_type, streams in self.monitoring_streams.items():
                    if not streams:
                        continue
                    
                    latest_data = max(streams, key=lambda x: x.timestamp)
                    age = current_time - latest_data.timestamp
                    
                    if age > timedelta(hours=2):
                        logger.warning(f"Stale data stream detected: {data_type} (age: {age})")
                    
                    # Check data quality
                    poor_quality_count = sum(
                        1 for stream in streams[-10:]  # Last 10 readings
                        if stream.quality_flag in ['questionable', 'bad']
                    )
                    
                    if poor_quality_count > 5:
                        logger.warning(f"Poor data quality in stream: {data_type}")
                
            except Exception as e:
                logger.error(f"Error monitoring data streams: {e}")
    
    def register_alert_callback(self, callback: Callable[[HazardAlert], None]):
        """Register callback function for hazard alerts"""
        self.alert_callbacks.append(callback)
    
    def get_active_assessments(self) -> Dict[str, MultiHazardAssessment]:
        """Get all active hazard assessments"""
        return self.active_assessments.copy()
    
    def get_monitoring_data_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of current monitoring data"""
        summary = {}
        
        for data_type, streams in self.monitoring_streams.items():
            if not streams:
                continue
            
            latest_stream = max(streams, key=lambda x: x.timestamp)
            summary[data_type] = {
                'count': len(streams),
                'latest_timestamp': latest_stream.timestamp,
                'latest_value': latest_stream.value,
                'data_quality': self._assess_stream_quality(streams)
            }
        
        return summary
    
    def _assess_stream_quality(self, streams: List[MonitoringDataStream]) -> str:
        """Assess overall quality of a data stream"""
        if not streams:
            return 'no_data'
        
        recent_streams = streams[-10:]  # Last 10 readings
        good_quality_count = sum(
            1 for stream in recent_streams
            if stream.quality_flag == 'good'
        )
        
        quality_ratio = good_quality_count / len(recent_streams)
        
        if quality_ratio >= 0.8:
            return 'good'
        elif quality_ratio >= 0.6:
            return 'fair'
        else:
            return 'poor'