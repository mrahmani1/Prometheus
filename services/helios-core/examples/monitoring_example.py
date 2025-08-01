"""
Real-Time Monitoring Data Models Example

Demonstrates the usage of real-time monitoring data models for volcanic
monitoring and alert systems.
"""

import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch.nn as nn
from models.geological import GeologicalLocation
from models.monitoring import (
    AlertEscalationManager,
    AlertLevel,
    AlertLevelEnum,
    NotificationChannel,
    NotificationPriority,
    NotificationRule,
    SensorReading,
    StreamingDataProcessor,
    VolcanicRiskAssessment,
)
from models.monitoring import numpy as np
from models.monitoring import pandas as pd
from models.monitoring import torch.nn as nn

def main():
    """Demonstrate real-time monitoring data models"""

    print("=== Prometheus AI Geodynamic Platform ===")
    print("Real-Time Monitoring Data Models Example\n")

    # 1. Create a volcanic monitoring location
    volcano_location = GeologicalLocation(
        latitude=45.3311,  # Mount Hood, Oregon
        longitude=-121.7113,
        elevation=3429.0,  # meters
        coordinate_system="WGS84",
    )

    print("Monitoring Location: Mount Hood")
    print(f"Coordinates: {volcano_location.latitude}°N, {volcano_location.longitude}°W")
    print(f"Elevation: {volcano_location.elevation}m\n")

    # 2. Create streaming data processor for real-time monitoring
    processor = StreamingDataProcessor()

    # Register sensors with quality thresholds
    processor.register_sensor(
        "SEIS_MH01",
        buffer_size=1000,
        quality_thresholds={"min_value": -0.1, "max_value": 0.1},
    )
    processor.register_sensor(
        "GPS_MH01",
        buffer_size=500,
        quality_thresholds={"min_value": -0.01, "max_value": 0.01},
    )

    print("Registered sensors:")
    print("- SEIS_MH01: Seismometer with velocity thresholds ±0.1 m/s")
    print("- GPS_MH01: GPS station with displacement thresholds ±0.01 m\n")

    # 3. Simulate incoming sensor data
    base_time = datetime.utcnow()

    # Normal seismic readings
    for i in range(10):
        reading = SensorReading(
            sensor_id="SEIS_MH01",
            timestamp=base_time + timedelta(seconds=i),
            location=volcano_location,
            measurement_type="velocity",
            value=np.random.normal(0, 0.001),  # Normal background noise
            unit="m/s",
            sampling_rate=100.0,
        )
        processed_reading = processor.process_reading(reading)

        if i == 0:
            print("Sample seismic reading:")
            print(f"  Value: {processed_reading.value:.6f} {processed_reading.unit}")
            print(f"  Quality: {processed_reading.quality_flag.value}")

    # Simulate a spike (potential earthquake)
    spike_reading = SensorReading(
        sensor_id="SEIS_MH01",
        timestamp=base_time + timedelta(seconds=15),
        location=volcano_location,
        measurement_type="velocity",
        value=0.05,  # Large spike
        unit="m/s",
        sampling_rate=100.0,
    )
    processed_spike = processor.process_reading(spike_reading)

    print("\nSpike detected:")
    print(f"  Value: {processed_spike.value:.6f} {processed_spike.unit}")
    print(f"  Quality: {processed_spike.quality_flag.value}")
    if "spike_detection" in processed_spike.metadata:
        deviation = processed_spike.metadata["spike_detection"]["deviation"]
        print(f"  Deviation: {deviation:.1f} standard deviations")

    # 4. Create alert level
    alert_start = datetime.utcnow()
    alert_end = alert_start + timedelta(hours=24)

    alert_level = AlertLevel(
        level=AlertLevelEnum.YELLOW,
        description="Elevated volcanic unrest detected",
        recommended_actions=[
            "Increase monitoring frequency",
            "Notify local authorities",
            "Prepare evacuation plans",
        ],
        validity_period=(alert_start, alert_end),
        confidence=0.75,
        contributing_factors=[
            "Increased seismic activity",
            "Ground deformation detected",
        ],
    )

    print(f"\nAlert Level: {alert_level.level.name} ({alert_level.level.value})")
    print(f"Description: {alert_level.description}")
    print(f"Confidence: {alert_level.confidence:.1%}")
    print(f"Valid until: {alert_end.strftime('%Y-%m-%d %H:%M UTC')}")

    # 5. Create volcanic risk assessment
    assessment = VolcanicRiskAssessment(
        volcano_id="MOUNT_HOOD",
        timestamp=datetime.utcnow(),
        alert_level=alert_level,
        eruption_probability={"24h": 0.05, "1week": 0.15, "1month": 0.30},
        hazard_zones={
            "pyroclastic_flow": np.array([[45.33, -121.71], [45.35, -121.69]]),
            "lahar": np.array([[45.30, -121.75], [45.40, -121.65]]),
        },
        contributing_factors=[
            "Seismic swarm detected",
            "GPS stations show uplift",
            "Increased gas emissions",
        ],
        model_ensemble_agreement=0.82,
        scenario_probabilities={
            "explosive": 0.3,
            "effusive": 0.5,
            "phreatomagmatic": 0.2,
        },
    )

    print("\nVolcanic Risk Assessment:")
    print(f"Volcano: {assessment.volcano_id}")
    print(f"Model Agreement: {assessment.model_ensemble_agreement:.1%}")

    print("\nEruption Probabilities:")
    for timeframe, prob in assessment.eruption_probability.items():
        print(f"  {timeframe}: {prob:.1%}")

    most_likely_scenario, scenario_prob = assessment.get_most_likely_scenario()
    print(f"\nMost Likely Scenario: {most_likely_scenario} ({scenario_prob:.1%})")

    # 6. Set up notification system
    notification_rule = NotificationRule(
        rule_id="HIGH_ALERT_RULE",
        name="High Alert Notification",
        conditions={
            "alert_level_min": 2,  # Orange or higher
            "eruption_probability_min": 0.2,
            "model_agreement_min": 0.7,
        },
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.EMAIL, NotificationChannel.SMS],
        recipients=["volcano.observatory@usgs.gov", "+1-555-VOLCANO"],
        cooldown_period=1800.0,  # 30 minutes
    )

    should_notify = notification_rule.should_trigger(assessment)
    print("\nNotification Rule Evaluation:")
    print(f"Rule: {notification_rule.name}")
    print(f"Should trigger: {should_notify}")

    if should_notify:
        print(f"Priority: {notification_rule.priority.value}")
        print(f"Channels: {[c.value for c in notification_rule.channels]}")

    # 7. Demonstrate escalation management
    escalation_manager = AlertEscalationManager()

    # Add escalation rules
    escalation_manager.escalation_rules = [
        {"level": 1, "delay_seconds": 3600, "priority": "medium"},  # 1 hour
        {"level": 2, "delay_seconds": 7200, "priority": "high"},  # 2 hours
        {"level": 3, "delay_seconds": 14400, "priority": "urgent"},  # 4 hours
    ]

    # Start escalation process
    escalation_manager.start_escalation(assessment)

    active_escalations = escalation_manager.get_active_escalations()
    print("\nEscalation Management:")
    print(f"Active escalations: {len(active_escalations)}")

    for escalation_id, data in active_escalations.items():
        print(f"  {escalation_id}:")
        print(f"    Current level: {data['current_level']}")
        print(f"    Alert level: {data['alert_level']}")

    # 8. Show system health
    system_health = processor.get_system_health()
    print("\nSystem Health:")
    print(f"Total sensors: {system_health['total_sensors']}")
    print(f"Active sensors: {system_health['active_sensors']}")
    print(f"Missing sensors: {system_health['missing_sensors']}")

    if system_health["quality_summary"]:
        print("Data quality summary:")
        for quality, count in system_health["quality_summary"].items():
            print(f"  {quality}: {count} readings")

    # 9. Demonstrate time series buffer
    buffer = processor.sensor_buffers["SEIS_MH01"]
    buffer_info = buffer.get_buffer_info()

    print("\nTime Series Buffer (SEIS_MH01):")
    print(f"Current size: {buffer_info['current_size']}/{buffer_info['max_size']}")
    print(f"Sampling rate: {buffer_info['sampling_rate']:.1f} Hz")
    print(f"Last update: {buffer_info['last_update']}")

    # Check for data gaps
    gaps = buffer.detect_data_gaps(max_gap_seconds=5)
    print(f"Data gaps detected: {len(gaps)}")

    print("\n=== Monitoring System Ready ===")
    print("Real-time volcanic monitoring system is operational.")
    print("All data models are functioning correctly.")

if __name__ == "__main__":
    main()
