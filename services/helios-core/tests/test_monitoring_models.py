"""
Tests for Real-Time Monitoring Data Models

Comprehensive tests for sensor readings, alert systems, risk assessments,
and streaming data processing.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
from models.geological import GeologicalLocation
from models.monitoring import (
    AlertEscalationManager,
    AlertLevel,
    AlertLevelEnum,
    Notification,
    NotificationChannel,
    NotificationPriority,
    NotificationRule,
    QualityFlag,
    SensorReading,
    StreamingDataProcessor,
    TimeSeriesBuffer,
    VolcanicRiskAssessment,
)
from models.monitoring import numpy as np
from models.monitoring import torch.nn as nn
from models.monitoring import torch.nn.functional as F

class TestSensorReading:
    """Test SensorReading class"""

    def test_sensor_reading_creation(self):
        """Test basic sensor reading creation"""
        location = GeologicalLocation(latitude=45.0, longitude=-122.0, elevation=1000.0)

        reading = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow(),
            location=location,
            measurement_type="velocity",
            value=0.001,
            unit="m/s",
            quality_flag=QualityFlag.GOOD,
            sampling_rate=100.0,
        )

        assert reading.sensor_id == "SEIS001"
        assert reading.measurement_type == "velocity"
        assert reading.value == 0.001
        assert reading.unit == "m/s"
        assert reading.quality_flag == QualityFlag.GOOD
        assert reading.sampling_rate == 100.0

    def test_sensor_reading_validation(self):
        """Test sensor reading validation"""
        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Test invalid sampling rate
        with pytest.raises(ValueError, match="Sampling rate must be positive"):
            SensorReading(
                sensor_id="SEIS001",
                timestamp=datetime.utcnow(),
                location=location,
                measurement_type="velocity",
                value=0.001,
                unit="m/s",
                sampling_rate=-1.0,
            )

        # Test future timestamp
        future_time = datetime.utcnow() + timedelta(minutes=10)
        with pytest.raises(
            ValueError, match="cannot be more than 5 minutes in the future"
        ):
            SensorReading(
                sensor_id="SEIS001",
                timestamp=future_time,
                location=location,
                measurement_type="velocity",
                value=0.001,
                unit="m/s",
            )

    def test_sensor_reading_non_finite_values(self):
        """Test handling of non-finite values"""
        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Test NaN value
        reading = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow(),
            location=location,
            measurement_type="velocity",
            value=float("nan"),
            unit="m/s",
        )

        assert reading.quality_flag == QualityFlag.BAD
        assert "validation_error" in reading.metadata

    def test_sensor_reading_serialization(self):
        """Test sensor reading serialization/deserialization"""
        location = GeologicalLocation(45.0, -122.0, 1000.0)

        original = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow(),
            location=location,
            measurement_type="velocity",
            value=np.array([0.001, 0.002, 0.003]),
            unit="m/s",
            quality_flag=QualityFlag.GOOD,
            sampling_rate=100.0,
            metadata={"test": "value"},
        )

        # Test to_dict
        data = original.to_dict()
        assert data["sensor_id"] == "SEIS001"
        assert data["measurement_type"] == "velocity"
        assert data["value"] == [0.001, 0.002, 0.003]
        assert data["quality_flag"] == "good"

        # Test from_dict
        reconstructed = SensorReading.from_dict(data)
        assert reconstructed.sensor_id == original.sensor_id
        assert reconstructed.measurement_type == original.measurement_type
        assert np.array_equal(reconstructed.value, original.value)
        assert reconstructed.quality_flag == original.quality_flag

class TestAlertLevel:
    """Test AlertLevel class"""

    def test_alert_level_creation(self):
        """Test basic alert level creation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Heightened volcanic unrest",
            recommended_actions=["Monitor closely", "Restrict access"],
            validity_period=(start_time, end_time),
            confidence=0.85,
            contributing_factors=["Increased seismicity", "Ground deformation"],
        )

        assert alert.level == AlertLevelEnum.ORANGE
        assert alert.confidence == 0.85
        assert len(alert.recommended_actions) == 2
        assert len(alert.contributing_factors) == 2

    def test_alert_level_validation(self):
        """Test alert level validation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            AlertLevel(
                level=AlertLevelEnum.ORANGE,
                description="Test",
                recommended_actions=["Test"],
                validity_period=(start_time, end_time),
                confidence=1.5,
            )

        # Test invalid validity period
        with pytest.raises(ValueError, match="start time must be before end time"):
            AlertLevel(
                level=AlertLevelEnum.ORANGE,
                description="Test",
                recommended_actions=["Test"],
                validity_period=(end_time, start_time),
                confidence=0.8,
            )

    def test_alert_level_validity_check(self):
        """Test alert level validity checking"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.8,
        )

        # Test valid time
        test_time = start_time + timedelta(hours=12)
        assert alert.is_valid_at(test_time)

        # Test invalid time
        test_time = start_time - timedelta(hours=1)
        assert not alert.is_valid_at(test_time)

    def test_alert_level_serialization(self):
        """Test alert level serialization"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        original = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test alert",
            recommended_actions=["Action 1", "Action 2"],
            validity_period=(start_time, end_time),
            confidence=0.85,
            hazard_zones={"pyroclastic_flow": [(45.0, -122.0), (45.1, -122.1)]},
        )

        # Test serialization
        data = original.to_dict()
        reconstructed = AlertLevel.from_dict(data)

        assert reconstructed.level == original.level
        assert reconstructed.description == original.description
        assert reconstructed.confidence == original.confidence
        assert reconstructed.hazard_zones == original.hazard_zones

class TestVolcanicRiskAssessment:
    """Test VolcanicRiskAssessment class"""

    def test_volcanic_risk_assessment_creation(self):
        """Test basic volcanic risk assessment creation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Heightened unrest",
            recommended_actions=["Monitor closely"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        assessment = VolcanicRiskAssessment(
            volcano_id="VOLCANO001",
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24h": 0.1, "1week": 0.3, "1month": 0.5},
            hazard_zones={"lava_flow": np.array([[45.0, -122.0], [45.1, -122.1]])},
            contributing_factors=["Seismic swarm", "Ground uplift"],
            model_ensemble_agreement=0.75,
        )

        assert assessment.volcano_id == "VOLCANO001"
        assert assessment.eruption_probability["24h"] == 0.1
        assert assessment.model_ensemble_agreement == 0.75
        assert len(assessment.contributing_factors) == 2

    def test_volcanic_risk_assessment_validation(self):
        """Test volcanic risk assessment validation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        # Test invalid model ensemble agreement
        with pytest.raises(
            ValueError, match="Model ensemble agreement must be between 0 and 1"
        ):
            VolcanicRiskAssessment(
                volcano_id="VOLCANO001",
                timestamp=datetime.utcnow(),
                alert_level=alert_level,
                eruption_probability={"24h": 0.1},
                hazard_zones={},
                contributing_factors=[],
                model_ensemble_agreement=1.5,
            )

        # Test invalid eruption probability
        with pytest.raises(
            ValueError, match="Eruption probability.*must be between 0 and 1"
        ):
            VolcanicRiskAssessment(
                volcano_id="VOLCANO001",
                timestamp=datetime.utcnow(),
                alert_level=alert_level,
                eruption_probability={"24h": 1.5},
                hazard_zones={},
                contributing_factors=[],
                model_ensemble_agreement=0.75,
            )

    def test_volcanic_risk_assessment_methods(self):
        """Test volcanic risk assessment utility methods"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        assessment = VolcanicRiskAssessment(
            volcano_id="VOLCANO001",
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24h": 0.1, "1week": 0.3, "1month": 0.5},
            hazard_zones={},
            contributing_factors=[],
            model_ensemble_agreement=0.75,
            scenario_probabilities={"explosive": 0.6, "effusive": 0.4},
        )

        # Test max eruption probability
        timeframe, prob = assessment.get_max_eruption_probability()
        assert timeframe == "1month"
        assert prob == 0.5

        # Test most likely scenario
        scenario, prob = assessment.get_most_likely_scenario()
        assert scenario == "explosive"
        assert prob == 0.6

class TestTimeSeriesBuffer:
    """Test TimeSeriesBuffer class"""

    def test_time_series_buffer_creation(self):
        """Test basic time series buffer creation"""
        buffer = TimeSeriesBuffer(sensor_id="SEIS001", max_size=100)

        assert buffer.sensor_id == "SEIS001"
        assert buffer.max_size == 100
        assert len(buffer.readings) == 0
        assert not buffer.is_full

    def test_time_series_buffer_add_readings(self):
        """Test adding readings to buffer"""
        buffer = TimeSeriesBuffer(sensor_id="SEIS001", max_size=3)

        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Add readings
        for i in range(5):
            reading = SensorReading(
                sensor_id="SEIS001",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                location=location,
                measurement_type="velocity",
                value=0.001 * i,
                unit="m/s",
            )
            buffer.add_reading(reading)

        # Buffer should only contain last 3 readings due to maxlen
        assert len(buffer.readings) == 3
        assert buffer.readings[-1].value == 0.004  # Last reading

    def test_time_series_buffer_quality_statistics(self):
        """Test quality statistics calculation"""
        buffer = TimeSeriesBuffer(sensor_id="SEIS001", max_size=10)

        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Add readings with different quality flags
        qualities = [
            QualityFlag.GOOD,
            QualityFlag.GOOD,
            QualityFlag.QUESTIONABLE,
            QualityFlag.BAD,
        ]

        for i, quality in enumerate(qualities):
            reading = SensorReading(
                sensor_id="SEIS001",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                location=location,
                measurement_type="velocity",
                value=0.001,
                unit="m/s",
                quality_flag=quality,
            )
            buffer.add_reading(reading)

        stats = buffer.get_quality_statistics()
        assert stats["good"] == 2
        assert stats["questionable"] == 1
        assert stats["bad"] == 1

    def test_time_series_buffer_data_gaps(self):
        """Test data gap detection"""
        buffer = TimeSeriesBuffer(sensor_id="SEIS001", max_size=10)

        location = GeologicalLocation(45.0, -122.0, 1000.0)
        # Use a fixed past time to avoid future timestamp validation issues
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Add readings with a gap
        timestamps = [
            base_time,
            base_time + timedelta(seconds=1),
            base_time + timedelta(seconds=600),  # 10 minute gap
            base_time + timedelta(seconds=601),
        ]

        for timestamp in timestamps:
            reading = SensorReading(
                sensor_id="SEIS001",
                timestamp=timestamp,
                location=location,
                measurement_type="velocity",
                value=0.001,
                unit="m/s",
            )
            buffer.add_reading(reading)

        gaps = buffer.detect_data_gaps(max_gap_seconds=300)  # 5 minute threshold
        assert len(gaps) == 1
        assert gaps[0][0] == base_time + timedelta(seconds=1)
        assert gaps[0][1] == base_time + timedelta(seconds=600)

class TestStreamingDataProcessor:
    """Test StreamingDataProcessor class"""

    def test_streaming_data_processor_creation(self):
        """Test basic streaming data processor creation"""
        processor = StreamingDataProcessor()

        assert len(processor.sensor_buffers) == 0
        assert processor.spike_detection_threshold == 3.0
        assert processor.missing_data_timeout == 600.0

    def test_streaming_data_processor_register_sensor(self):
        """Test sensor registration"""
        processor = StreamingDataProcessor()

        processor.register_sensor("SEIS001", buffer_size=100)

        assert "SEIS001" in processor.sensor_buffers
        assert processor.sensor_buffers["SEIS001"].max_size == 100

    def test_streaming_data_processor_quality_control(self):
        """Test quality control processing"""
        processor = StreamingDataProcessor()
        processor.register_sensor(
            "SEIS001", quality_thresholds={"min_value": -1.0, "max_value": 1.0}
        )

        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Test normal reading
        reading = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow(),
            location=location,
            measurement_type="velocity",
            value=0.5,
            unit="m/s",
        )

        processed = processor.process_reading(reading)
        assert processed.quality_flag == QualityFlag.GOOD

        # Test out-of-range reading
        reading_out_of_range = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow(),
            location=location,
            measurement_type="velocity",
            value=2.0,  # Above max_value
            unit="m/s",
        )

        processed = processor.process_reading(reading_out_of_range)
        assert processed.quality_flag == QualityFlag.OUT_OF_RANGE

    def test_streaming_data_processor_spike_detection(self):
        """Test spike detection"""
        processor = StreamingDataProcessor(spike_detection_threshold=2.0)
        processor.register_sensor("SEIS001")

        location = GeologicalLocation(45.0, -122.0, 1000.0)

        # Add normal readings
        for i in range(15):
            reading = SensorReading(
                sensor_id="SEIS001",
                timestamp=datetime.utcnow() + timedelta(seconds=i),
                location=location,
                measurement_type="velocity",
                value=0.001,  # Consistent value
                unit="m/s",
            )
            processor.process_reading(reading)

        # Add spike
        spike_reading = SensorReading(
            sensor_id="SEIS001",
            timestamp=datetime.utcnow() + timedelta(seconds=16),
            location=location,
            measurement_type="velocity",
            value=0.1,  # Much larger value
            unit="m/s",
        )

        processed = processor.process_reading(spike_reading)
        assert processed.quality_flag == QualityFlag.SPIKE_DETECTED
        assert "spike_detection" in processed.metadata

class TestNotificationSystem:
    """Test notification system components"""

    def test_notification_rule_creation(self):
        """Test notification rule creation"""
        rule = NotificationRule(
            rule_id="RULE001",
            name="High Alert Rule",
            conditions={"alert_level_min": 3, "eruption_probability_min": 0.5},
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
            recipients=["admin@volcano.org", "+1234567890"],
            cooldown_period=300.0,
        )

        assert rule.rule_id == "RULE001"
        assert rule.priority == NotificationPriority.HIGH
        assert len(rule.channels) == 2
        assert len(rule.recipients) == 2

    def test_notification_rule_evaluation(self):
        """Test notification rule condition evaluation"""
        rule = NotificationRule(
            rule_id="RULE001",
            name="Test Rule",
            conditions={"alert_level_min": 2, "eruption_probability_min": 0.3},
            priority=NotificationPriority.MEDIUM,
            channels=[NotificationChannel.EMAIL],
            recipients=["test@example.com"],
        )

        # Create test assessment
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,  # Level 2
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        assessment = VolcanicRiskAssessment(
            volcano_id="VOLCANO001",
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24h": 0.4},  # Above threshold
            hazard_zones={},
            contributing_factors=[],
            model_ensemble_agreement=0.75,
        )

        # Should trigger
        assert rule.should_trigger(assessment)

        # Test with lower probability
        assessment.eruption_probability = {"24h": 0.2}  # Below threshold
        assert not rule.should_trigger(assessment)

    def test_notification_creation(self):
        """Test notification creation"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        assessment = VolcanicRiskAssessment(
            volcano_id="VOLCANO001",
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24h": 0.4},
            hazard_zones={},
            contributing_factors=[],
            model_ensemble_agreement=0.75,
        )

        notification = Notification(
            notification_id="NOTIF001",
            rule_id="RULE001",
            timestamp=datetime.utcnow(),
            priority=NotificationPriority.HIGH,
            channels=[NotificationChannel.EMAIL],
            recipients=["test@example.com"],
            subject="Volcanic Alert",
            message="Test message",
            assessment=assessment,
        )

        assert notification.status == "pending"
        assert len(notification.delivered_channels) == 0

        # Test marking as delivered
        notification.mark_delivered(NotificationChannel.EMAIL)
        assert notification.status == "delivered"
        assert NotificationChannel.EMAIL in notification.delivered_channels

class TestAlertEscalationManager:
    """Test AlertEscalationManager class"""

    def test_escalation_manager_creation(self):
        """Test escalation manager creation"""
        manager = AlertEscalationManager()

        assert len(manager.escalation_rules) == 0
        assert len(manager.active_escalations) == 0
        assert manager.max_escalation_level == 5

    def test_escalation_start_and_stop(self):
        """Test starting and stopping escalations"""
        manager = AlertEscalationManager()

        # Create test assessment
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=24)

        alert_level = AlertLevel(
            level=AlertLevelEnum.ORANGE,
            description="Test",
            recommended_actions=["Test"],
            validity_period=(start_time, end_time),
            confidence=0.85,
        )

        assessment = VolcanicRiskAssessment(
            volcano_id="VOLCANO001",
            timestamp=datetime.utcnow(),
            alert_level=alert_level,
            eruption_probability={"24h": 0.4},
            hazard_zones={},
            contributing_factors=[],
            model_ensemble_agreement=0.75,
        )

        # Start escalation
        manager.start_escalation(assessment)
        assert len(manager.active_escalations) == 1

        # Stop escalation
        manager.stop_escalation("VOLCANO001", assessment.assessment_id)
        assert len(manager.active_escalations) == 0

if __name__ == "__main__":
    pytest.main([__file__])
