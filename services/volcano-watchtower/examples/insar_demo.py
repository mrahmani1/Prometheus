#!/usr/bin/env python3
"""
InSAR Deformation Analysis Demo

This script demonstrates the InSAR deformation analysis capabilities
of the Volcano Watchtower system, including deep learning-based
atmospheric correction, noise reduction, and pattern recognition.

Requirements addressed: 3.3, 1.2
"""

import sys
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.insar_analysis import (
    InSARDeformationAnalyzer, InSARData, 
    load_insar_data, create_synthetic_insar_time_series
)
from core.insar_integration import InSARMonitoringService


def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_deformation_result(result, title: str = "Deformation Analysis Results"):
    """Print formatted deformation analysis results"""
    print_separator(title)
    
    print(f"Processing Date: {result.processing_metadata['processing_date']}")
    print(f"Wavelength: {result.processing_metadata['wavelength']} m")
    print(f"Device Used: {result.processing_metadata['device_used']}")
    
    print("\nDisplacement Statistics:")
    print(f"  Max Displacement: {np.nanmax(np.abs(result.displacement_map)):.3f} m")
    print(f"  Min Displacement: {np.nanmin(result.displacement_map):.3f} m")
    print(f"  Mean Displacement: {np.nanmean(result.displacement_map):.3f} m")
    print(f"  RMS Displacement: {np.sqrt(np.nanmean(result.displacement_map**2)):.3f} m")
    
    print("\nUncertainty Statistics:")
    print(f"  Mean Uncertainty: {np.nanmean(result.uncertainty_map):.3f} m")
    print(f"  Max Uncertainty: {np.nanmax(result.uncertainty_map):.3f} m")
    
    print("\nQuality Metrics:")
    for key, value in result.quality_metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nDeformation Patterns Detected: {len(result.deformation_patterns)}")
    for i, pattern in enumerate(result.deformation_patterns):
        print(f"  Pattern {i+1}:")
        print(f"    Type: {pattern.get('pattern_type', 'unknown')}")
        print(f"    Confidence: {pattern.get('confidence', 0.0):.3f}")
        if 'magnitude_m' in pattern:
            print(f"    Magnitude: {pattern['magnitude_m']:.3f} m")
        if 'location' in pattern:
            print(f"    Location: ({pattern['location'][0]:.1f}, {pattern['location'][1]:.1f})")


def demonstrate_single_interferogram():
    """Demonstrate processing of a single InSAR interferogram"""
    print_separator("Single Interferogram Processing Demo")
    
    # Initialize analyzer
    analyzer = InSARDeformationAnalyzer()
    
    # Create synthetic InSAR data
    print("Creating synthetic InSAR data...")
    metadata = {
        'sensor': 'Sentinel-1',
        'track': 'T064',
        'frame': 'F135',
        'demo': True
    }
    insar_data = load_insar_data("demo_interferogram.tif", metadata)
    
    print(f"Data shape: {insar_data.interferogram.shape}")
    print(f"Wavelength: {insar_data.wavelength} m")
    print(f"Acquisition dates: {insar_data.acquisition_dates[0]} to {insar_data.acquisition_dates[1]}")
    
    # Process the data
    print("\nProcessing InSAR data...")
    result = analyzer.process_insar_data(insar_data)
    
    # Display results
    print_deformation_result(result)


def demonstrate_time_series_analysis():
    """Demonstrate time series analysis with velocity estimation"""
    print_separator("Time Series Analysis Demo")
    
    # Initialize analyzer
    analyzer = InSARDeformationAnalyzer()
    
    # Create synthetic time series
    num_acquisitions = 8
    print(f"Creating synthetic time series with {num_acquisitions} acquisitions...")
    time_series = create_synthetic_insar_time_series(num_acquisitions)
    
    print("Acquisition dates:")
    for i, data in enumerate(time_series):
        print(f"  {i+1}: {data.acquisition_dates[0]} to {data.acquisition_dates[1]}")
    
    # Process time series
    print("\nProcessing time series...")
    result = analyzer.process_time_series(time_series)
    
    # Display results
    print_deformation_result(result, "Time Series Analysis Results")
    
    if result.velocity_map is not None:
        print("\nVelocity Statistics:")
        print(f"  Max Velocity: {np.nanmax(np.abs(result.velocity_map)):.3f} m/year")
        print(f"  Min Velocity: {np.nanmin(result.velocity_map):.3f} m/year")
        print(f"  Mean Velocity: {np.nanmean(result.velocity_map):.3f} m/year")
        print(f"  RMS Velocity: {np.sqrt(np.nanmean(result.velocity_map**2)):.3f} m/year")


async def demonstrate_monitoring_service():
    """Demonstrate the integrated monitoring service"""
    print_separator("Monitoring Service Demo")
    
    # Initialize monitoring service
    monitoring_service = InSARMonitoringService()
    volcano_id = "demo_volcano_001"
    
    print(f"Initializing monitoring for volcano: {volcano_id}")
    
    # Process several acquisitions
    num_acquisitions = 5
    print(f"\nProcessing {num_acquisitions} InSAR acquisitions...")
    
    for i in range(num_acquisitions):
        print(f"\nProcessing acquisition {i+1}/{num_acquisitions}...")
        
        # Create synthetic data
        metadata = {'acquisition_number': i, 'sensor': 'Sentinel-1'}
        insar_data = load_insar_data(f"monitoring_{i}.tif", metadata)
        
        # Add some temporal evolution to the deformation
        base_date = datetime(2024, 1, 1)
        days_offset = i * 12
        insar_data.acquisition_dates = (
            base_date + timedelta(days=days_offset),
            base_date + timedelta(days=days_offset + 12)
        )
        
        # Process through monitoring service
        result = await monitoring_service.process_new_insar_data(insar_data, volcano_id)
        
        # Display processing summary
        summary = result['processing_summary']
        print(f"  Max displacement: {summary['max_displacement_m']:.3f} m")
        print(f"  Quality score: {summary['quality_score']:.2f}")
        print(f"  Patterns detected: {summary['patterns_detected']}")
        print(f"  Alerts generated: {summary['alerts_generated']}")
        
        if result['alerts']:
            print("  Alerts:")
            for alert in result['alerts']:
                print(f"    - {alert['type']}: Level {alert['level']} - {alert['message']}")
    
    # Process time series
    print("\nProcessing accumulated time series...")
    time_series_result = await monitoring_service.process_time_series_update(volcano_id)
    
    if time_series_result.get('status') != 'insufficient_data':
        print("Time series analysis completed:")
        print(f"  Time series length: {time_series_result.get('time_series_length', 'N/A')}")
        print(f"  Velocity alerts: {len(time_series_result.get('velocity_alerts', []))}")
        
        if time_series_result.get('velocity_alerts'):
            print("  Velocity alerts:")
            for alert in time_series_result['velocity_alerts']:
                print(f"    - {alert['type']}: Level {alert['level']} - {alert['message']}")
    
    # Get monitoring status
    print("\nFinal monitoring status:")
    status = monitoring_service.get_monitoring_status(volcano_id)
    for key, value in status.items():
        if key != 'volcano_id':
            print(f"  {key.replace('_', ' ').title()}: {value}")


def demonstrate_pattern_recognition():
    """Demonstrate deformation pattern recognition capabilities"""
    print_separator("Pattern Recognition Demo")
    
    # Initialize analyzer
    analyzer = InSARDeformationAnalyzer()
    
    # Create synthetic data with known patterns
    print("Creating synthetic data with known deformation patterns...")
    
    shape = (256, 256)
    y, x = np.ogrid[:shape[0], :shape[1]]
    
    # Create base interferogram
    interferogram = np.random.randn(*shape) * 0.05
    coherence = np.random.uniform(0.4, 0.9, shape)
    amplitude = np.random.uniform(1000, 5000, shape)
    
    # Add inflation pattern (Gaussian)
    center_y, center_x = 100, 100
    r1 = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    inflation_pattern = 0.08 * np.exp(-r1**2 / (2 * 25**2))
    interferogram += inflation_pattern * 4 * np.pi / 0.056
    
    # Add deflation pattern
    center_y, center_x = 180, 150
    r2 = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    deflation_pattern = -0.06 * np.exp(-r2**2 / (2 * 20**2))
    interferogram += deflation_pattern * 4 * np.pi / 0.056
    
    # Create InSAR data structure
    coordinates = np.zeros((*shape, 2))
    coordinates[:, :, 0] = np.linspace(45.0, 45.1, shape[0])[:, np.newaxis]
    coordinates[:, :, 1] = np.linspace(-122.0, -121.9, shape[1])[np.newaxis, :]
    
    insar_data = InSARData(
        interferogram=interferogram,
        coherence=coherence,
        amplitude=amplitude,
        coordinates=coordinates,
        acquisition_dates=(datetime(2024, 1, 1), datetime(2024, 1, 13)),
        wavelength=0.056,
        incidence_angle=np.full(shape, 23.0),
        heading_angle=-167.0,
        metadata={'demo': 'pattern_recognition'}
    )
    
    print("Expected patterns:")
    print(f"  - Inflation at ({100}, {100}) with 8 cm magnitude")
    print(f"  - Deflation at ({180}, {150}) with 6 cm magnitude")
    
    # Process the data
    print("\nProcessing data for pattern recognition...")
    result = analyzer.process_insar_data(insar_data)
    
    # Display results
    print_deformation_result(result, "Pattern Recognition Results")


def main():
    """Main demo function"""
    print("InSAR Deformation Analysis System Demo")
    print("=====================================")
    print("This demo showcases the capabilities of the InSAR analysis system")
    print("including deep learning-based processing, pattern recognition,")
    print("and integrated monitoring services.")
    
    try:
        # Run demonstrations
        demonstrate_single_interferogram()
        demonstrate_time_series_analysis()
        demonstrate_pattern_recognition()
        
        # Run async monitoring demo
        print("\nRunning monitoring service demo...")
        asyncio.run(demonstrate_monitoring_service())
        
        print_separator("Demo Complete")
        print("All demonstrations completed successfully!")
        print("\nKey capabilities demonstrated:")
        print("✓ Deep learning-based atmospheric correction")
        print("✓ Noise reduction using residual networks")
        print("✓ Deformation pattern recognition and classification")
        print("✓ Ground displacement measurement with uncertainty")
        print("✓ Time series analysis and velocity estimation")
        print("✓ Integrated monitoring service with alerting")
        print("✓ Real-time processing pipeline")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())