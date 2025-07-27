#!/usr/bin/env python3
"""
Demonstration of multi-sensor data fusion algorithms for volcanic monitoring

This script shows how to use the sensor fusion algorithms to combine data from
multiple sensor types (seismic, thermal, deformation, strain) for improved
volcanic monitoring and analysis.
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.sensor_fusion import MultiSensorDataFusion
from models.geological_data import (
    SeismicEvent, ThermalMeasurement, DeformationMeasurement,
    StrainMeasurement, GeologicalLocation, DataQuality
)

def create_sample_volcanic_scenario():
    """
    Create a sample volcanic monitoring scenario with multiple sensor types
    simulating volcanic unrest at Mount St. Helens
    """
    print("Creating sample volcanic monitoring scenario...")
    
    # Mount St. Helens location
    volcano_location = GeologicalLocation(
        latitude=46.1978,
        longitude=-122.1956,
        elevation=2549.0
    )
    
    base_time = datetime.now()
    geological_data = []
    
    # Simulate seismic swarm (increasing magnitude over time)
    print("  - Adding seismic events...")
    for i in range(8):
        event_time = base_time + timedelta(minutes=i*15)
        
        # Add some spatial variation around the volcano
        event_location = GeologicalLocation(
            latitude=volcano_location.latitude + np.random.normal(0, 0.01),
            longitude=volcano_location.longitude + np.random.normal(0, 0.01),
            elevation=volcano_location.elevation + np.random.normal(0, 200)
        )
        
        # Increasing magnitude trend with some noise
        magnitude = 2.5 + i*0.3 + np.random.normal(0, 0.2)
        
        seismic_event = SeismicEvent(
            timestamp=event_time,
            location=event_location,
            magnitude=magnitude,
            depth=5.0 + np.random.normal(0, 2.0),
            event_type='volcanic',
            quality=DataQuality.GOOD if i % 2 == 0 else DataQuality.EXCELLENT,
            source=f'seismic_station_{i%3 + 1}'
        )
        
        geological_data.append(seismic_event)
    
    # Simulate thermal anomalies (increasing temperature)
    print("  - Adding thermal measurements...")
    for i in range(5):
        measurement_time = base_time + timedelta(minutes=i*25)
        
        thermal_location = GeologicalLocation(
            latitude=volcano_location.latitude + np.random.normal(0, 0.005),
            longitude=volcano_location.longitude + np.random.normal(0, 0.005),
            elevation=volcano_location.elevation
        )
        
        # Increasing temperature trend
        temperature = 300.0 + i*15 + np.random.normal(0, 5)  # Kelvin
        
        thermal_measurement = ThermalMeasurement(
            timestamp=measurement_time,
            location=thermal_location,
            temperature=temperature,
            measurement_method='satellite_thermal',
            spatial_resolution=30.0,
            uncertainty=3.0,
            quality=DataQuality.GOOD,
            source='thermal_satellite_1'
        )
        
        geological_data.append(thermal_measurement)
    
    # Simulate ground deformation (inflation pattern)
    print("  - Adding deformation measurements...")
    for i in range(6):
        measurement_time = base_time + timedelta(minutes=i*20)
        
        deformation_location = GeologicalLocation(
            latitude=volcano_location.latitude + np.random.normal(0, 0.008),
            longitude=volcano_location.longitude + np.random.normal(0, 0.008),
            elevation=volcano_location.elevation
        )
        
        # Inflation pattern (outward and upward movement)
        displacement = np.array([
            0.002 + i*0.003 + np.random.normal(0, 0.001),  # East
            0.001 + i*0.002 + np.random.normal(0, 0.001),  # North
            0.005 + i*0.004 + np.random.normal(0, 0.001)   # Up
        ])
        
        deformation_measurement = DeformationMeasurement(
            timestamp=measurement_time,
            location=deformation_location,
            displacement_vector=displacement,
            measurement_type='InSAR',
            uncertainty=np.array([0.001, 0.001, 0.002]),
            quality=DataQuality.EXCELLENT,
            source='insar_satellite_1'
        )
        
        geological_data.append(deformation_measurement)
    
    # Simulate strain measurements (increasing strain rates)
    print("  - Adding strain measurements...")
    for i in range(10):
        measurement_time = base_time + timedelta(minutes=i*12)
        
        strain_location = GeologicalLocation(
            latitude=volcano_location.latitude + np.random.normal(0, 0.01),
            longitude=volcano_location.longitude + np.random.normal(0, 0.01),
            elevation=volcano_location.elevation
        )
        
        # Increasing strain rate
        strain_rate = 1e-7 + i*2e-7 + np.random.normal(0, 5e-8)
        
        strain_measurement = StrainMeasurement(
            timestamp=measurement_time,
            location=strain_location,
            strain_rate=strain_rate,
            strain_rate_std=strain_rate * 0.1,
            strain_rate_max=strain_rate * 1.5,
            measurement_type='das',
            channel_index=i % 4,
            quality=DataQuality.GOOD,
            source=f'fiber_optic_array_channel_{i%4}'
        )
        
        geological_data.append(strain_measurement)
    
    print(f"Created {len(geological_data)} sensor measurements")
    return geological_data, volcano_location

def demonstrate_kalman_fusion(geological_data, reference_location):
    """Demonstrate Kalman filter-based sensor fusion"""
    print("\n" + "="*60)
    print("KALMAN FILTER SENSOR FUSION DEMONSTRATION")
    print("="*60)
    
    # Configure fusion system
    fusion_config = {
        'temporal_window': 3600,  # 1 hour window
        'spatial_threshold': 25.0,  # 25 km threshold
        'outlier_threshold': 2.5   # Moderate outlier threshold
    }
    
    fusion_system = MultiSensorDataFusion(fusion_config)
    
    # Perform Kalman fusion
    print("Performing Kalman filter fusion...")
    fusion_results = fusion_system.fuse_sensors(
        geological_data,
        fusion_method='kalman',
        reference_location=reference_location,
        reference_time=datetime.now()
    )
    
    print(f"Generated {len(fusion_results)} fusion results")
    
    # Analyze results
    for i, result in enumerate(fusion_results):
        print(f"\nFusion Result {i+1}:")
        print(f"  Contributing sensors: {result.contributing_sensors}")
        print(f"  Fusion quality: {result.fusion_quality:.3f}")
        print(f"  Confidence level: {result.confidence_level:.3f}")
        print(f"  Uncertainty reduction: {result.uncertainty_reduction:.3f}")
        print(f"  Outliers rejected: {len(result.outliers_rejected)}")
        print(f"  Temporal alignment offset: {result.temporal_alignment_offset:.1f} seconds")
        
        # Location information
        location = result.fused_state.location
        print(f"  Fused location: {location.latitude:.4f}°N, {location.longitude:.4f}°W")
        print(f"  Elevation: {location.elevation:.0f} m")
        
        # State vector information
        state_size = len(result.fused_state.state_vector)
        print(f"  State vector size: {state_size}")
        print(f"  State uncertainty (trace): {np.trace(result.fused_state.covariance_matrix):.6f}")
        
        # Alert assessment
        if result.confidence_level > 0.8:
            print("  🚨 HIGH CONFIDENCE DETECTION - Potential volcanic unrest!")
        elif result.confidence_level > 0.6:
            print("  ⚠️  MODERATE CONFIDENCE - Monitor closely")
        else:
            print("  ℹ️  LOW CONFIDENCE - Normal background activity")
    
    return fusion_results

def demonstrate_weighted_fusion(geological_data, reference_location):
    """Demonstrate weighted average sensor fusion"""
    print("\n" + "="*60)
    print("WEIGHTED AVERAGE SENSOR FUSION DEMONSTRATION")
    print("="*60)
    
    # Configure fusion system
    fusion_config = {
        'temporal_window': 3600,  # 1 hour window
        'spatial_threshold': 25.0,  # 25 km threshold
        'outlier_threshold': 2.5   # Moderate outlier threshold
    }
    
    fusion_system = MultiSensorDataFusion(fusion_config)
    
    # Perform weighted fusion
    print("Performing weighted average fusion...")
    fusion_results = fusion_system.fuse_sensors(
        geological_data,
        fusion_method='weighted',
        reference_location=reference_location
    )
    
    print(f"Generated {len(fusion_results)} fusion result")
    
    # Analyze results
    for i, result in enumerate(fusion_results):
        print(f"\nWeighted Fusion Result {i+1}:")
        print(f"  Contributing sensors: {result.contributing_sensors}")
        print(f"  Fusion quality: {result.fusion_quality:.3f}")
        print(f"  Confidence level: {result.confidence_level:.3f}")
        print(f"  Uncertainty reduction: {result.uncertainty_reduction:.3f}")
        
        # Location information
        location = result.fused_state.location
        print(f"  Fused location: {location.latitude:.4f}°N, {location.longitude:.4f}°W")
        print(f"  Elevation: {location.elevation:.0f} m")
        
        # State vector information
        state_size = len(result.fused_state.state_vector)
        print(f"  State vector size: {state_size}")
        
        # Comparison with Kalman results
        print(f"  Note: Weighted fusion provides simpler but faster processing")
    
    return fusion_results

def analyze_sensor_contributions(geological_data):
    """Analyze the contribution of different sensor types"""
    print("\n" + "="*60)
    print("SENSOR CONTRIBUTION ANALYSIS")
    print("="*60)
    
    # Count sensor types
    sensor_counts = {}
    quality_distribution = {}
    temporal_span = {}
    
    for data in geological_data:
        sensor_type = type(data).__name__
        
        # Count by type
        sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1
        
        # Quality distribution
        if sensor_type not in quality_distribution:
            quality_distribution[sensor_type] = {}
        quality = data.quality.value
        quality_distribution[sensor_type][quality] = quality_distribution[sensor_type].get(quality, 0) + 1
        
        # Temporal span
        if sensor_type not in temporal_span:
            temporal_span[sensor_type] = {'min': data.timestamp, 'max': data.timestamp}
        else:
            temporal_span[sensor_type]['min'] = min(temporal_span[sensor_type]['min'], data.timestamp)
            temporal_span[sensor_type]['max'] = max(temporal_span[sensor_type]['max'], data.timestamp)
    
    # Print analysis
    print("Sensor Type Distribution:")
    for sensor_type, count in sensor_counts.items():
        print(f"  {sensor_type}: {count} measurements")
    
    print("\nData Quality Distribution:")
    for sensor_type, qualities in quality_distribution.items():
        print(f"  {sensor_type}:")
        for quality, count in qualities.items():
            print(f"    {quality}: {count}")
    
    print("\nTemporal Coverage:")
    for sensor_type, span in temporal_span.items():
        duration = (span['max'] - span['min']).total_seconds() / 60  # minutes
        print(f"  {sensor_type}: {duration:.1f} minutes")

def main():
    """Main demonstration function"""
    print("Multi-Sensor Data Fusion Demonstration")
    print("Prometheus AI Geodynamic Platform - Volcano Watchtower")
    print("="*60)
    
    try:
        # Create sample data
        geological_data, reference_location = create_sample_volcanic_scenario()
        
        # Analyze sensor contributions
        analyze_sensor_contributions(geological_data)
        
        # Demonstrate Kalman fusion
        kalman_results = demonstrate_kalman_fusion(geological_data, reference_location)
        
        # Demonstrate weighted fusion
        weighted_results = demonstrate_weighted_fusion(geological_data, reference_location)
        
        # Summary comparison
        print("\n" + "="*60)
        print("FUSION METHOD COMPARISON")
        print("="*60)
        
        print(f"Kalman Filter Results:")
        print(f"  Number of clusters: {len(kalman_results)}")
        if kalman_results:
            avg_quality = np.mean([r.fusion_quality for r in kalman_results])
            avg_confidence = np.mean([r.confidence_level for r in kalman_results])
            total_outliers = sum(len(r.outliers_rejected) for r in kalman_results)
            print(f"  Average fusion quality: {avg_quality:.3f}")
            print(f"  Average confidence: {avg_confidence:.3f}")
            print(f"  Total outliers detected: {total_outliers}")
        
        print(f"\nWeighted Average Results:")
        print(f"  Number of results: {len(weighted_results)}")
        if weighted_results:
            result = weighted_results[0]
            print(f"  Fusion quality: {result.fusion_quality:.3f}")
            print(f"  Confidence level: {result.confidence_level:.3f}")
        
        print("\nRecommendations:")
        print("- Use Kalman filtering for real-time monitoring with outlier detection")
        print("- Use weighted averaging for fast processing of high-quality data")
        print("- Combine both methods for comprehensive volcanic monitoring")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())