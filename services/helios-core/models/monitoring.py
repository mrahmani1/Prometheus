"""
Real-Time Monitoring Data Models

Data structures for real-time volcanic and seismic monitoring, including
sensor readings, alert systems, and risk assessments.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from enum import Enum
import numpy as np
import json
import uuid
from collections import deque
from .geological import GeologicalLocation


class SensorType(str, Enum):
    """Types of monitoring sensors"""
    SEISMOMETER = "seismometer"
    GPS = "gps"
    TILTMETER = "tiltmeter"
    STRAINMETER = "strainmeter"
    GAS_SENSOR = "gas_sensor"
    THERMAL_CAMERA = "thermal_camera"
    INFRASOUND = "infrasound"
    MAGNETOMETER = "magnetometer"
    GRAVIMETER = "gravimeter"
    FIBER_OPTIC_DAS = "fiber_optic_das"
    FIBER_OPTIC_DTS = "fiber_optic_dts"
    HYDROPHONE = "hydrophone"
    WEATHER_STATION = "weather_station"
    WEBCAM = "webcam"


class QualityFlag(str, Enum):
    """Data quality flags for sensor readings"""
    GOOD = "good"
    QUESTIONABLE = "questionable"
    BAD = "bad"
    MISSING = "missing"
    CALIBRATION_ERROR = "calibration_error"
    INSTRUMENT_MALFUNCTION = "instrument_malfunction"
    COMMUNICATION_ERROR = "communication_error"
    OUT_OF_RANGE = "out_of_range"
    SPIKE_DETECTED = "spike_detected"
    DRIFT_DETECTED = "drift_detected"


class AlertLevelEnum(int, Enum):
    """Volcanic alert levels (0-5 scale)"""
    GREEN = 0      # Normal background activity
    YELLOW = 1     # Elevated unrest
    ORANGE = 2     # Heightened unrest
    RED = 3        # Eruption imminent or in progress
    CRITICAL = 4   # Major eruption in progress
    EXTREME = 5    # Catastrophic eruption


class NotificationPriority(str, Enum):
    """Priority levels for notifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Channels for sending notifications"""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    SLACK = "slack"
    PHONE_CALL = "phone_call"
    EMERGENCY_BROADCAST = "emergency_broadcast"


@dataclass
class SensorReading:
    """
    Represents a single sensor reading with quality control metadata.
    
    Attributes:
        sensor_id: Unique identifier for the sensor
        timestamp: Reading timestamp (UTC)
        location: Sensor location
        measurement_type: Type of measurement (e.g., 'velocity', 'displacement', 'temperature')
        value: Measured value
        unit: Unit of measurement
        quality_flag: Data quality assessment
        calibration_date: Last calibration date
        raw_value: Original uncorrected value
        correction_applied: Description of corrections applied
        sampling_rate: Sensor sampling rate in Hz
        instrument_response: Instrument response characteristics
        metadata: Additional sensor-specific metadata
    """
    sensor_id: str
    timestamp: datetime
    location: GeologicalLocation
    measurement_type: str
    value: Union[float, np.ndarray]
    unit: str
    quality_flag: QualityFlag = QualityFlag.GOOD
    calibration_date: Optional[datetime] = None
    raw_value: Optional[Union[float, np.ndarray]] = None
    correction_applied: Optional[str] = None
    sampling_rate: Optional[float] = None
    instrument_response: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sensor reading parameters"""
        if self.sampling_rate is not None and self.sampling_rate <= 0:
            raise ValueError(f"Sampling rate must be positive, got {self.sampling_rate}")
        
        # Check for reasonable timestamp
        now = datetime.utcnow()
        if self.timestamp > now + timedelta(minutes=5):
            raise ValueError("Sensor reading timestamp cannot be more than 5 minutes in the future")
        
        # Validate numeric values
        if isinstance(self.value, (int, float)):
            if not np.isfinite(self.value):
                self.quality_flag = QualityFlag.BAD
                self.metadata["validation_error"] = "Non-finite value detected"
        elif isinstance(self.value, np.ndarray):
            if not np.all(np.isfinite(self.value)):
                self.quality_flag = QualityFlag.BAD
                self.metadata["validation_error"] = "Non-finite values in array"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "location": self.location.to_dict(),
            "measurement_type": self.measurement_type,
            "value": self.value.tolist() if isinstance(self.value, np.ndarray) else self.value,
            "unit": self.unit,
            "quality_flag": self.quality_flag.value,
            "calibration_date": self.calibration_date.isoformat() if self.calibration_date else None,
            "raw_value": self.raw_value.tolist() if isinstance(self.raw_value, np.ndarray) else self.raw_value,
            "correction_applied": self.correction_applied,
            "sampling_rate": self.sampling_rate,
            "instrument_response": self.instrument_response,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SensorReading':
        """Create from dictionary"""
        value = data["value"]
        if isinstance(value, list):
            value = np.array(value)
        
        raw_value = data.get("raw_value")
        if isinstance(raw_value, list):
            raw_value = np.array(raw_value)
        
        return cls(
            sensor_id=data["sensor_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=GeologicalLocation.from_dict(data["location"]),
            measurement_type=data["measurement_type"],
            value=value,
            unit=data["unit"],
            quality_flag=QualityFlag(data.get("quality_flag", "good")),
            calibration_date=datetime.fromisoformat(data["calibration_date"]) if data.get("calibration_date") else None,
            raw_value=raw_value,
            correction_applied=data.get("correction_applied"),
            sampling_rate=data.get("sampling_rate"),
            instrument_response=data.get("instrument_response"),
            metadata=data.get("metadata", {})
        )


@dataclass
class AlertLevel:
    """
    Represents a volcanic alert level with associated metadata.
    
    Attributes:
        level: Alert level (0-5 scale)
        description: Human-readable description
        recommended_actions: List of recommended actions
        validity_period: Time period for which alert is valid
        confidence: Confidence in alert level (0-1)
        contributing_factors: Factors that led to this alert level
        hazard_zones: Geographic areas affected by different hazard types
        evacuation_zones: Areas requiring evacuation at this alert level
        restrictions: Activity restrictions in place
    """
    level: AlertLevelEnum
    description: str
    recommended_actions: List[str]
    validity_period: Tuple[datetime, datetime]
    confidence: float
    contributing_factors: List[str] = field(default_factory=list)
    hazard_zones: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    evacuation_zones: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate alert level parameters"""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        start_time, end_time = self.validity_period
        if start_time >= end_time:
            raise ValueError("Alert validity start time must be before end time")
        
        # Check if validity period is reasonable
        duration = end_time - start_time
        if duration > timedelta(days=30):
            import warnings
            warnings.warn("Alert validity period exceeds 30 days")
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if alert is valid at given timestamp"""
        start_time, end_time = self.validity_period
        return start_time <= timestamp <= end_time
    
    def time_until_expiry(self, current_time: Optional[datetime] = None) -> timedelta:
        """Calculate time until alert expires"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        _, end_time = self.validity_period
        return end_time - current_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        start_time, end_time = self.validity_period
        return {
            "level": self.level.value,
            "description": self.description,
            "recommended_actions": self.recommended_actions,
            "validity_period": [start_time.isoformat(), end_time.isoformat()],
            "confidence": self.confidence,
            "contributing_factors": self.contributing_factors,
            "hazard_zones": self.hazard_zones,
            "evacuation_zones": self.evacuation_zones,
            "restrictions": self.restrictions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertLevel':
        """Create from dictionary"""
        validity_period = tuple(datetime.fromisoformat(t) for t in data["validity_period"])
        
        return cls(
            level=AlertLevelEnum(data["level"]),
            description=data["description"],
            recommended_actions=data["recommended_actions"],
            validity_period=validity_period,
            confidence=data["confidence"],
            contributing_factors=data.get("contributing_factors", []),
            hazard_zones=data.get("hazard_zones", {}),
            evacuation_zones=data.get("evacuation_zones", []),
            restrictions=data.get("restrictions", [])
        )


@dataclass
class VolcanicRiskAssessment:
    """
    Comprehensive volcanic risk assessment with probabilistic forecasting.
    
    Attributes:
        volcano_id: Unique identifier for the volcano
        timestamp: Assessment timestamp (UTC)
        alert_level: Current alert level
        eruption_probability: Probability of eruption in different timeframes
        hazard_zones: Areas affected by different volcanic hazards
        contributing_factors: Factors contributing to current risk level
        model_ensemble_agreement: Agreement between different models (0-1)
        uncertainty_quantification: Uncertainty estimates for key parameters
        scenario_probabilities: Probabilities for different eruption scenarios
        impact_assessment: Potential impacts on population and infrastructure
        monitoring_recommendations: Recommended monitoring enhancements
        assessment_id: Unique identifier for this assessment
    """
    volcano_id: str
    timestamp: datetime
    alert_level: AlertLevel
    eruption_probability: Dict[str, float]  # timeframe -> probability
    hazard_zones: Dict[str, np.ndarray]     # hazard type -> affected coordinates
    contributing_factors: List[str]
    model_ensemble_agreement: float
    uncertainty_quantification: Dict[str, float] = field(default_factory=dict)
    scenario_probabilities: Dict[str, float] = field(default_factory=dict)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    monitoring_recommendations: List[str] = field(default_factory=list)
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate volcanic risk assessment parameters"""
        if not 0 <= self.model_ensemble_agreement <= 1:
            raise ValueError(f"Model ensemble agreement must be between 0 and 1, got {self.model_ensemble_agreement}")
        
        # Validate eruption probabilities
        for timeframe, probability in self.eruption_probability.items():
            if not 0 <= probability <= 1:
                raise ValueError(f"Eruption probability for {timeframe} must be between 0 and 1, got {probability}")
        
        # Validate scenario probabilities
        for scenario, probability in self.scenario_probabilities.items():
            if not 0 <= probability <= 1:
                raise ValueError(f"Scenario probability for {scenario} must be between 0 and 1, got {probability}")
        
        # Check if scenario probabilities sum to reasonable value
        if self.scenario_probabilities and abs(sum(self.scenario_probabilities.values()) - 1.0) > 0.1:
            import warnings
            warnings.warn("Scenario probabilities do not sum to 1.0")
    
    def get_max_eruption_probability(self) -> Tuple[str, float]:
        """Get the timeframe with maximum eruption probability"""
        if not self.eruption_probability:
            return "unknown", 0.0
        
        max_timeframe = max(self.eruption_probability.items(), key=lambda x: x[1])
        return max_timeframe
    
    def get_most_likely_scenario(self) -> Tuple[str, float]:
        """Get the most likely eruption scenario"""
        if not self.scenario_probabilities:
            return "unknown", 0.0
        
        most_likely = max(self.scenario_probabilities.items(), key=lambda x: x[1])
        return most_likely
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "volcano_id": self.volcano_id,
            "timestamp": self.timestamp.isoformat(),
            "alert_level": self.alert_level.to_dict(),
            "eruption_probability": self.eruption_probability,
            "hazard_zones": {k: v.tolist() for k, v in self.hazard_zones.items()},
            "contributing_factors": self.contributing_factors,
            "model_ensemble_agreement": self.model_ensemble_agreement,
            "uncertainty_quantification": self.uncertainty_quantification,
            "scenario_probabilities": self.scenario_probabilities,
            "impact_assessment": self.impact_assessment,
            "monitoring_recommendations": self.monitoring_recommendations,
            "assessment_id": self.assessment_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolcanicRiskAssessment':
        """Create from dictionary"""
        hazard_zones = {k: np.array(v) for k, v in data["hazard_zones"].items()}
        
        return cls(
            volcano_id=data["volcano_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            alert_level=AlertLevel.from_dict(data["alert_level"]),
            eruption_probability=data["eruption_probability"],
            hazard_zones=hazard_zones,
            contributing_factors=data["contributing_factors"],
            model_ensemble_agreement=data["model_ensemble_agreement"],
            uncertainty_quantification=data.get("uncertainty_quantification", {}),
            scenario_probabilities=data.get("scenario_probabilities", {}),
            impact_assessment=data.get("impact_assessment", {}),
            monitoring_recommendations=data.get("monitoring_recommendations", []),
            assessment_id=data.get("assessment_id", str(uuid.uuid4()))
        )


@dataclass
class TimeSeriesBuffer:
    """
    Optimized circular buffer for streaming time-series data.
    
    Attributes:
        sensor_id: Identifier for the sensor
        max_size: Maximum number of readings to store
        readings: Circular buffer of sensor readings
        current_index: Current position in the buffer
        is_full: Whether the buffer is full
        sampling_rate: Expected sampling rate in Hz
        last_update: Timestamp of last update
    """
    sensor_id: str
    max_size: int
    readings: deque = field(default_factory=deque)
    current_index: int = 0
    is_full: bool = False
    sampling_rate: Optional[float] = None
    last_update: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize the circular buffer"""
        if self.max_size <= 0:
            raise ValueError(f"Buffer size must be positive, got {self.max_size}")
        
        # Initialize deque with maxlen for automatic circular behavior
        self.readings = deque(maxlen=self.max_size)
    
    def add_reading(self, reading: SensorReading) -> None:
        """Add a new sensor reading to the buffer"""
        if reading.sensor_id != self.sensor_id:
            raise ValueError(f"Reading sensor_id {reading.sensor_id} does not match buffer sensor_id {self.sensor_id}")
        
        self.readings.append(reading)
        self.last_update = reading.timestamp
        
        # Update sampling rate estimate
        if len(self.readings) >= 2:
            time_diff = (self.readings[-1].timestamp - self.readings[-2].timestamp).total_seconds()
            if time_diff > 0:
                current_rate = 1.0 / time_diff
                if self.sampling_rate is None:
                    self.sampling_rate = current_rate
                else:
                    # Exponential moving average
                    self.sampling_rate = 0.9 * self.sampling_rate + 0.1 * current_rate
    
    def get_recent_readings(self, count: int) -> List[SensorReading]:
        """Get the most recent N readings"""
        if count <= 0:
            return []
        
        return list(self.readings)[-count:]
    
    def get_readings_in_timerange(self, start_time: datetime, end_time: datetime) -> List[SensorReading]:
        """Get readings within a specific time range"""
        return [
            reading for reading in self.readings
            if start_time <= reading.timestamp <= end_time
        ]
    
    def get_quality_statistics(self) -> Dict[str, int]:
        """Get statistics on data quality flags"""
        quality_counts = {}
        for reading in self.readings:
            flag = reading.quality_flag.value
            quality_counts[flag] = quality_counts.get(flag, 0) + 1
        
        return quality_counts
    
    def detect_data_gaps(self, max_gap_seconds: float = 300) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in the time series data"""
        gaps = []
        
        if len(self.readings) < 2:
            return gaps
        
        readings_list = list(self.readings)
        for i in range(1, len(readings_list)):
            time_diff = (readings_list[i].timestamp - readings_list[i-1].timestamp).total_seconds()
            if time_diff > max_gap_seconds:
                gaps.append((readings_list[i-1].timestamp, readings_list[i].timestamp))
        
        return gaps
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get information about the buffer state"""
        return {
            "sensor_id": self.sensor_id,
            "current_size": len(self.readings),
            "max_size": self.max_size,
            "is_full": len(self.readings) == self.max_size,
            "sampling_rate": self.sampling_rate,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "quality_statistics": self.get_quality_statistics()
        }


@dataclass
class StreamingDataProcessor:
    """
    Processes streaming sensor data with real-time quality control.
    
    Attributes:
        sensor_buffers: Dictionary of sensor buffers by sensor_id
        quality_thresholds: Thresholds for quality flag assignment
        spike_detection_threshold: Standard deviations for spike detection
        drift_detection_window: Window size for drift detection
        missing_data_timeout: Timeout for missing data detection (seconds)
    """
    sensor_buffers: Dict[str, TimeSeriesBuffer] = field(default_factory=dict)
    quality_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    spike_detection_threshold: float = 3.0
    drift_detection_window: int = 100
    missing_data_timeout: float = 600.0  # 10 minutes
    
    def register_sensor(self, sensor_id: str, buffer_size: int = 1000, 
                       quality_thresholds: Optional[Dict[str, float]] = None) -> None:
        """Register a new sensor for monitoring"""
        self.sensor_buffers[sensor_id] = TimeSeriesBuffer(
            sensor_id=sensor_id,
            max_size=buffer_size
        )
        
        if quality_thresholds:
            self.quality_thresholds[sensor_id] = quality_thresholds
    
    def process_reading(self, reading: SensorReading) -> SensorReading:
        """Process a sensor reading with quality control"""
        # Create buffer if it doesn't exist
        if reading.sensor_id not in self.sensor_buffers:
            self.register_sensor(reading.sensor_id)
        
        buffer = self.sensor_buffers[reading.sensor_id]
        
        # Apply quality control checks
        reading = self._apply_quality_control(reading, buffer)
        
        # Add to buffer
        buffer.add_reading(reading)
        
        return reading
    
    def _apply_quality_control(self, reading: SensorReading, buffer: TimeSeriesBuffer) -> SensorReading:
        """Apply quality control checks to a sensor reading"""
        # Check for spikes
        if len(buffer.readings) >= 10:
            recent_values = [r.value for r in buffer.get_recent_readings(10) 
                           if isinstance(r.value, (int, float)) and r.quality_flag == QualityFlag.GOOD]
            
            if len(recent_values) >= 5 and isinstance(reading.value, (int, float)):
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                
                if std_val > 0 and abs(reading.value - mean_val) > self.spike_detection_threshold * std_val:
                    reading.quality_flag = QualityFlag.SPIKE_DETECTED
                    reading.metadata["spike_detection"] = {
                        "deviation": abs(reading.value - mean_val) / std_val,
                        "threshold": self.spike_detection_threshold
                    }
        
        # Check for drift
        if len(buffer.readings) >= self.drift_detection_window:
            recent_readings = buffer.get_recent_readings(self.drift_detection_window)
            values = [r.value for r in recent_readings 
                     if isinstance(r.value, (int, float)) and r.quality_flag == QualityFlag.GOOD]
            
            if len(values) >= self.drift_detection_window // 2:
                # Simple linear trend detection
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                # Check if slope is significant
                if abs(slope) > 0.1 * np.std(values):  # Configurable threshold
                    reading.metadata["drift_detected"] = {
                        "slope": slope,
                        "window_size": len(values)
                    }
        
        # Check sensor-specific thresholds
        if reading.sensor_id in self.quality_thresholds:
            thresholds = self.quality_thresholds[reading.sensor_id]
            
            if isinstance(reading.value, (int, float)):
                if "min_value" in thresholds and reading.value < thresholds["min_value"]:
                    reading.quality_flag = QualityFlag.OUT_OF_RANGE
                    reading.metadata["threshold_violation"] = "below_minimum"
                
                if "max_value" in thresholds and reading.value > thresholds["max_value"]:
                    reading.quality_flag = QualityFlag.OUT_OF_RANGE
                    reading.metadata["threshold_violation"] = "above_maximum"
        
        return reading
    
    def detect_missing_sensors(self, current_time: Optional[datetime] = None) -> List[str]:
        """Detect sensors that haven't reported data recently"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        missing_sensors = []
        
        for sensor_id, buffer in self.sensor_buffers.items():
            if buffer.last_update is None:
                missing_sensors.append(sensor_id)
            else:
                time_since_update = (current_time - buffer.last_update).total_seconds()
                if time_since_update > self.missing_data_timeout:
                    missing_sensors.append(sensor_id)
        
        return missing_sensors
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health statistics"""
        total_sensors = len(self.sensor_buffers)
        active_sensors = len([b for b in self.sensor_buffers.values() if b.last_update is not None])
        missing_sensors = self.detect_missing_sensors()
        
        quality_summary = {}
        for buffer in self.sensor_buffers.values():
            quality_stats = buffer.get_quality_statistics()
            for flag, count in quality_stats.items():
                quality_summary[flag] = quality_summary.get(flag, 0) + count
        
        return {
            "total_sensors": total_sensors,
            "active_sensors": active_sensors,
            "missing_sensors": len(missing_sensors),
            "missing_sensor_ids": missing_sensors,
            "quality_summary": quality_summary,
            "timestamp": datetime.utcnow().isoformat()
        }


@dataclass
class NotificationRule:
    """
    Defines rules for when and how to send notifications.
    
    Attributes:
        rule_id: Unique identifier for the rule
        name: Human-readable name for the rule
        conditions: Conditions that trigger the notification
        priority: Notification priority level
        channels: Channels to use for notification
        recipients: List of recipients
        cooldown_period: Minimum time between notifications (seconds)
        escalation_rules: Rules for escalating notifications
        is_active: Whether the rule is currently active
        last_triggered: Timestamp of last trigger
    """
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    priority: NotificationPriority
    channels: List[NotificationChannel]
    recipients: List[str]
    cooldown_period: float = 300.0  # 5 minutes default
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    last_triggered: Optional[datetime] = None
    
    def should_trigger(self, assessment: VolcanicRiskAssessment, current_time: Optional[datetime] = None) -> bool:
        """Check if notification should be triggered based on conditions"""
        if not self.is_active:
            return False
        
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Check cooldown period
        if self.last_triggered is not None:
            time_since_last = (current_time - self.last_triggered).total_seconds()
            if time_since_last < self.cooldown_period:
                return False
        
        # Check conditions
        return self._evaluate_conditions(assessment)
    
    def _evaluate_conditions(self, assessment: VolcanicRiskAssessment) -> bool:
        """Evaluate notification conditions against assessment"""
        for condition_type, condition_value in self.conditions.items():
            if condition_type == "alert_level_min":
                if assessment.alert_level.level.value < condition_value:
                    return False
            
            elif condition_type == "eruption_probability_min":
                max_prob = max(assessment.eruption_probability.values()) if assessment.eruption_probability else 0
                if max_prob < condition_value:
                    return False
            
            elif condition_type == "model_agreement_min":
                if assessment.model_ensemble_agreement < condition_value:
                    return False
            
            elif condition_type == "contributing_factors":
                required_factors = set(condition_value)
                actual_factors = set(assessment.contributing_factors)
                if not required_factors.intersection(actual_factors):
                    return False
        
        return True
    
    def mark_triggered(self, current_time: Optional[datetime] = None) -> None:
        """Mark the rule as triggered"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        self.last_triggered = current_time


@dataclass
class Notification:
    """
    Represents a notification to be sent.
    
    Attributes:
        notification_id: Unique identifier
        rule_id: ID of the rule that triggered this notification
        timestamp: When the notification was created
        priority: Notification priority
        channels: Channels to send through
        recipients: List of recipients
        subject: Notification subject/title
        message: Notification message body
        assessment: Associated volcanic risk assessment
        status: Current status of the notification
        delivery_attempts: Number of delivery attempts
        delivered_channels: Channels successfully delivered to
        failed_channels: Channels that failed delivery
    """
    notification_id: str
    rule_id: str
    timestamp: datetime
    priority: NotificationPriority
    channels: List[NotificationChannel]
    recipients: List[str]
    subject: str
    message: str
    assessment: VolcanicRiskAssessment
    status: str = "pending"
    delivery_attempts: int = 0
    delivered_channels: List[NotificationChannel] = field(default_factory=list)
    failed_channels: List[NotificationChannel] = field(default_factory=list)
    
    def mark_delivered(self, channel: NotificationChannel) -> None:
        """Mark notification as delivered for a specific channel"""
        if channel not in self.delivered_channels:
            self.delivered_channels.append(channel)
        
        if channel in self.failed_channels:
            self.failed_channels.remove(channel)
        
        # Update overall status
        if len(self.delivered_channels) == len(self.channels):
            self.status = "delivered"
        elif len(self.delivered_channels) > 0:
            self.status = "partially_delivered"
    
    def mark_failed(self, channel: NotificationChannel, error: str) -> None:
        """Mark notification as failed for a specific channel"""
        if channel not in self.failed_channels:
            self.failed_channels.append(channel)
        
        # Update overall status
        if len(self.failed_channels) == len(self.channels):
            self.status = "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "notification_id": self.notification_id,
            "rule_id": self.rule_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "recipients": self.recipients,
            "subject": self.subject,
            "message": self.message,
            "assessment": self.assessment.to_dict(),
            "status": self.status,
            "delivery_attempts": self.delivery_attempts,
            "delivered_channels": [c.value for c in self.delivered_channels],
            "failed_channels": [c.value for c in self.failed_channels]
        }


@dataclass
class AlertEscalationManager:
    """
    Manages alert escalation based on time and conditions.
    
    Attributes:
        escalation_rules: Rules for escalating alerts
        active_escalations: Currently active escalation processes
        notification_rules: Notification rules by priority level
        max_escalation_level: Maximum escalation level
    """
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    active_escalations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notification_rules: Dict[NotificationPriority, List[NotificationRule]] = field(default_factory=dict)
    max_escalation_level: int = 5
    
    def start_escalation(self, assessment: VolcanicRiskAssessment) -> None:
        """Start escalation process for a volcanic risk assessment"""
        escalation_id = f"{assessment.volcano_id}_{assessment.assessment_id}"
        
        self.active_escalations[escalation_id] = {
            "assessment": assessment,
            "start_time": datetime.utcnow(),
            "current_level": 0,
            "last_escalation": datetime.utcnow(),
            "notifications_sent": []
        }
    
    def check_escalations(self, current_time: Optional[datetime] = None) -> List[Notification]:
        """Check for escalations that need to be triggered"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        notifications = []
        
        for escalation_id, escalation_data in self.active_escalations.items():
            assessment = escalation_data["assessment"]
            current_level = escalation_data["current_level"]
            last_escalation = escalation_data["last_escalation"]
            
            # Check if escalation should occur
            for rule in self.escalation_rules:
                if (rule["level"] == current_level + 1 and
                    (current_time - last_escalation).total_seconds() >= rule["delay_seconds"]):
                    
                    # Escalate
                    escalation_data["current_level"] += 1
                    escalation_data["last_escalation"] = current_time
                    
                    # Generate notifications for new level
                    priority = NotificationPriority(rule["priority"])
                    if priority in self.notification_rules:
                        for notification_rule in self.notification_rules[priority]:
                            if notification_rule.should_trigger(assessment, current_time):
                                notification = self._create_escalation_notification(
                                    notification_rule, assessment, escalation_data["current_level"]
                                )
                                notifications.append(notification)
                                notification_rule.mark_triggered(current_time)
                    
                    break
        
        return notifications
    
    def _create_escalation_notification(self, rule: NotificationRule, 
                                      assessment: VolcanicRiskAssessment, 
                                      escalation_level: int) -> Notification:
        """Create an escalation notification"""
        subject = f"ESCALATED ALERT Level {escalation_level}: {assessment.volcano_id}"
        
        message = f"""
        ESCALATED VOLCANIC ALERT - Level {escalation_level}
        
        Volcano: {assessment.volcano_id}
        Current Alert Level: {assessment.alert_level.level.name} ({assessment.alert_level.level.value})
        Assessment Time: {assessment.timestamp.isoformat()}
        
        Eruption Probabilities:
        {chr(10).join([f"  {timeframe}: {prob:.1%}" for timeframe, prob in assessment.eruption_probability.items()])}
        
        Contributing Factors:
        {chr(10).join([f"  - {factor}" for factor in assessment.contributing_factors])}
        
        Model Agreement: {assessment.model_ensemble_agreement:.1%}
        
        Recommended Actions:
        {chr(10).join([f"  - {action}" for action in assessment.alert_level.recommended_actions])}
        
        This is an escalated alert requiring immediate attention.
        """
        
        return Notification(
            notification_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            timestamp=datetime.utcnow(),
            priority=rule.priority,
            channels=rule.channels,
            recipients=rule.recipients,
            subject=subject,
            message=message,
            assessment=assessment
        )
    
    def stop_escalation(self, volcano_id: str, assessment_id: str) -> None:
        """Stop escalation process"""
        escalation_id = f"{volcano_id}_{assessment_id}"
        if escalation_id in self.active_escalations:
            del self.active_escalations[escalation_id]
    
    def get_active_escalations(self) -> Dict[str, Dict[str, Any]]:
        """Get information about active escalations"""
        return {
            escalation_id: {
                "volcano_id": data["assessment"].volcano_id,
                "start_time": data["start_time"].isoformat(),
                "current_level": data["current_level"],
                "last_escalation": data["last_escalation"].isoformat(),
                "alert_level": data["assessment"].alert_level.level.value
            }
            for escalation_id, data in self.active_escalations.items()
        }