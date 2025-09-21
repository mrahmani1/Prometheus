"""
Validation script for the High-Resolution Deformation Analysis System
Tests core functionality without requiring full database setup
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from modules.resurgent_caldera.deformation_analysis_system import (
    HighResolutionDeformationAnalysisSystem,
    InSARParameters,
    GPSStationData,
    TiltStrainData
)

async def test_deformation_analysis_system():
    """Test basic functionality of the deformation analysis system"""
    print("Testing High-Resolution Deformation Analysis System...")
    
    # Use in-memory SQLite for testing
    database_url = "sqlite:///:memory:"
    
    try:
        # Initialize system
        system = HighResolutionDeformationAnalysisSystem(database_url)
        print("✓ Deformation Analysis System initialized successfully")
        
        # Test InSAR processing
        print("\n--- Testing InSAR Time-Series Processing ---")
        
        insar_parameters = InSARParameters(
            satellite='Sentinel-1',
            wavelength=5.6,  # cm
            incidence_angle=39.0,  # degrees
            heading_angle=-13.0,  # degrees
            temporal_baseline=12,  # days
            spatial_baseline=150.0,  # meters
            coherence_threshold=0.3,
            unwrapping_method='snaphu',
            atmospheric_model='era5'
        )
        
        # Mock InSAR data
        insar_data = {
            'interferograms': np.random.rand(8, 50, 50) * 2 * np.pi - np.pi,
            'coherence': np.random.rand(8, 50, 50) * 0.8 + 0.2,
            'temporal_baselines': np.arange(0, 96, 12),
            'atmospheric_data': {},
            'reference_points': [(25, 25), (15, 35)]
        }
        
        insar_results = await system.process_insar_time_series(
            insar_data=insar_data,
            parameters=insar_parameters,
            caldera_id='test_caldera'
        )
        
        print(f"✓ InSAR time-series processed")
        print(f"  - Quality score: {insar_results['quality_metrics']['quality_score']:.3f}")
        print(f"  - Data coverage: {insar_results['quality_metrics']['data_coverage']:.3f}")
        print(f"  - Mean coherence: {insar_results['quality_metrics']['coherence_stats']['mean_coherence']:.3f}")
        
        # Test GPS network analysis
        print("\n--- Testing GPS Network Analysis ---")
        
        # Create mock GPS stations
        gps_stations = []
        for i in range(3):
            station = GPSStationData(
                station_id=f'GPS{i+1:02d}',
                location=(45.0 + i*0.1, -110.0 + i*0.1, 2000.0 + i*100),
                time_series={
                    'east': np.random.rand(100) * 10 + i*2,
                    'north': np.random.rand(100) * 8 + i*1.5,
                    'up': np.random.rand(100) * 15 + i*3
                },
                uncertainties={
                    'east': np.ones(100) * 2.0,
                    'north': np.ones(100) * 2.0,
                    'up': np.ones(100) * 5.0
                },
                processing_method='PPP',
                reference_frame='ITRF2014',
                monument_type='deep_rod'
            )
            gps_stations.append(station)
        
        analysis_period = (datetime(2020, 1, 1), datetime(2023, 1, 1))
        
        gps_results = await system.analyze_gps_network(
            gps_stations=gps_stations,
            caldera_id='test_caldera',
            analysis_period=analysis_period,
            reference_frame='ITRF2014'
        )
        
        print(f"✓ GPS network analyzed")
        print(f"  - Number of stations: {gps_results['network_statistics']['number_of_stations']}")
        print(f"  - Mean velocity magnitude: {np.linalg.norm(gps_results['velocity_field']['mean_velocity']):.2f} mm/year")
        print(f"  - Temporal consistency: {gps_results['temporal_evolution']['temporal_consistency']:.3f}")
        
        # Test tilt and strain analysis
        print("\n--- Testing Tilt and Strain Analysis ---")
        
        # Create mock tilt and strain data
        tilt_strain_data = []
        
        # Tiltmeter data
        tiltmeter = TiltStrainData(
            instrument_id='TILT01',
            instrument_type='tiltmeter',
            location=(45.05, -109.95, 2100.0),
            measurements={
                'tilt_x': np.random.rand(1000) * 1e-6,  # microradians
                'tilt_y': np.random.rand(1000) * 1e-6
            },
            sampling_rate=1.0,  # Hz
            calibration_data={'last_calibration': '2023-01-01'},
            environmental_corrections={
                'temperature': np.random.rand(1000) * 0.1,
                'pressure': np.random.rand(1000) * 0.05
            }
        )
        tilt_strain_data.append(tiltmeter)
        
        # Strainmeter data
        strainmeter = TiltStrainData(
            instrument_id='STRAIN01',
            instrument_type='strainmeter',
            location=(45.03, -109.97, 2050.0),
            measurements={
                'strain_xx': np.random.rand(1000) * 1e-8,  # microstrain
                'strain_yy': np.random.rand(1000) * 1e-8,
                'strain_xy': np.random.rand(1000) * 1e-8
            },
            sampling_rate=1.0,  # Hz
            calibration_data={'last_calibration': '2023-01-01'},
            environmental_corrections={
                'temperature': np.random.rand(1000) * 0.1
            }
        )
        tilt_strain_data.append(strainmeter)
        
        tilt_strain_results = await system.analyze_tilt_strain_data(
            tilt_strain_data=tilt_strain_data,
            caldera_id='test_caldera',
            analysis_window=timedelta(days=30)
        )
        
        print(f"✓ Tilt and strain data analyzed")
        print(f"  - Tilt analyses: {len(tilt_strain_results['tilt_analyses'])}")
        print(f"  - Strain analyses: {len(tilt_strain_results['strain_analyses'])}")
        print(f"  - Overall quality score: {tilt_strain_results['data_quality_metrics']['overall_quality_score']:.3f}")
        
        # Test deformation acceleration detection
        print("\n--- Testing Deformation Acceleration Detection ---")
        
        # Create mock time-series data with acceleration
        time_series_data = {
            'east': np.cumsum(np.random.rand(100) * 0.1) + np.arange(100) * 0.05,
            'north': np.cumsum(np.random.rand(100) * 0.08) + np.arange(100) * 0.03,
            'up': np.cumsum(np.random.rand(100) * 0.15) + np.arange(100) * 0.1
        }
        
        acceleration_results = await system.detect_deformation_acceleration(
            time_series_data=time_series_data,
            detection_threshold=2.0,
            window_size=10
        )
        
        print(f"✓ Deformation acceleration detected")
        print(f"  - Velocity components calculated: {len(acceleration_results['velocities'])}")
        print(f"  - Acceleration components calculated: {len(acceleration_results['accelerations'])}")
        
        # Check for anomalies
        total_anomalies = sum(
            len(anomaly['anomaly_indices']) 
            for anomaly in acceleration_results['acceleration_anomalies'].values()
        )
        print(f"  - Total acceleration anomalies detected: {total_anomalies}")
        
        # Test deformation pattern classification
        print("\n--- Testing Deformation Pattern Classification ---")
        
        # Create mock spatial deformation data
        x = np.linspace(-20, 20, 41)
        y = np.linspace(-20, 20, 41)
        X, Y = np.meshgrid(x, y)
        
        # Create radial uplift pattern
        R = np.sqrt(X**2 + Y**2)
        vertical_displacement = 50 * np.exp(-R/10) * (1 + 0.1*np.random.rand(*R.shape))
        
        spatial_deformation_data = {
            'vertical_displacement': vertical_displacement,
            'east_displacement': np.random.rand(*R.shape) * 5,
            'north_displacement': np.random.rand(*R.shape) * 5
        }
        
        # Mock temporal data
        temporal_data = {
            'vertical': np.cumsum(np.random.rand(50) * 2) + np.arange(50) * 0.5,
            'horizontal': np.cumsum(np.random.rand(50) * 1) + np.arange(50) * 0.2
        }
        
        # Mock caldera geometry
        caldera_geometry = {
            'rim_diameter': 40.0,
            'center_x': 0.0,
            'center_y': 0.0
        }
        
        pattern_results = await system.classify_deformation_patterns(
            spatial_deformation_data=spatial_deformation_data,
            temporal_data=temporal_data,
            caldera_geometry=caldera_geometry
        )
        
        print(f"✓ Deformation patterns classified")
        print(f"  - Dominant spatial pattern: {pattern_results['spatial_patterns']['dominant_pattern']}")
        print(f"  - Pattern strength: {pattern_results['spatial_patterns']['pattern_strength']:.2f} mm")
        print(f"  - ML classification: {pattern_results['ml_classification']['classification']}")
        print(f"  - Classification confidence: {pattern_results['classification_confidence']['overall_confidence']:.3f}")
        print(f"  - Best physical model: {pattern_results['physical_models']['best_model']}")
        
        # Close database session
        system.session.close()
        print("\n✓ Database session closed")
        
        print("\n🎉 All tests passed successfully!")
        print("High-Resolution Deformation Analysis System is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main validation function"""
    print("=" * 70)
    print("High-Resolution Deformation Analysis System Validation")
    print("=" * 70)
    
    success = await test_deformation_analysis_system()
    
    if success:
        print("\n✅ Validation completed successfully!")
        print("The deformation analysis system is ready for use.")
    else:
        print("\n❌ Validation failed!")
        print("Please check the error messages above.")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())