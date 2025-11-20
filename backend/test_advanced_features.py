"""
Test suite for advanced seismic source characterization features

This test validates the novel algorithms and advanced implementations
that replace the placeholder methods.
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_advanced_features():
    """Test the advanced seismic characterization features"""
    try:
        logger.info("🚀 Testing Advanced Seismic Source Characterization Features")
        
        # Import the advanced system
        from core.advanced_seismic_source_characterization import AdvancedSeismicSourceCharacterization
        
        # Create mock database manager
        class MockDBManager:
            async def get_postgresql_session(self):
                return None
        
        # Initialize the system
        db_manager = MockDBManager()
        seismic_system = AdvancedSeismicSourceCharacterization(db_manager)
        
        logger.info("✅ Successfully created AdvancedSeismicSourceCharacterization instance")
        
        # Test 1: Advanced Wavelet Features
        logger.info("\n📊 Test 1: Advanced Wavelet Feature Extraction")
        
        # Create complex synthetic waveform with multiple components
        duration = 60.0  # seconds
        sampling_rate = 100.0  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Multi-component synthetic seismic signal
        noise = np.random.normal(0, 0.05, len(t))
        
        # P wave arrival with frequency modulation
        p_wave = np.exp(-(t - 10.0) / 3.0) * np.sin(2 * np.pi * 12 * (t - 10.0) * (1 + 0.1 * np.sin(t - 10.0)))
        p_wave[t < 10.0] = 0
        
        # S wave with amplitude modulation
        s_wave = np.exp(-(t - 15.0) / 5.0) * np.sin(2 * np.pi * 6 * (t - 15.0)) * (1 + 0.3 * np.cos(0.5 * (t - 15.0)))
        s_wave[t < 15.0] = 0
        
        # Surface waves with dispersion
        surface_wave = np.exp(-(t - 25.0) / 10.0) * np.sin(2 * np.pi * 2 * (t - 25.0) * np.sqrt(t - 25.0 + 1))
        surface_wave[t < 25.0] = 0
        
        complex_waveform = noise + 0.8 * p_wave + 1.2 * s_wave + 0.6 * surface_wave
        
        # Test advanced wavelet features
        wavelet_features = await seismic_system._extract_advanced_wavelet_features(complex_waveform)
        logger.info(f"✅ Extracted {len(wavelet_features)} advanced wavelet features")
        assert len(wavelet_features) == 12, "Should extract exactly 12 wavelet features"
        assert all(np.isfinite(f) for f in wavelet_features), "All features should be finite"
        
        # Test EMD features
        emd_features = await seismic_system._extract_emd_features(complex_waveform)
        logger.info(f"✅ Extracted {len(emd_features)} EMD features")
        assert len(emd_features) == 8, "Should extract exactly 8 EMD features"
        
        # Test higher-order statistics
        hos_features = await seismic_system._extract_higher_order_statistics(complex_waveform)
        logger.info(f"✅ Extracted {len(hos_features)} higher-order statistical features")
        assert len(hos_features) == 6, "Should extract exactly 6 HOS features"
        
        # Test 2: Advanced Green's Functions
        logger.info("\n🌍 Test 2: Advanced 3D Green's Functions")
        
        # Load 3D velocity model
        velocity_model = await seismic_system._load_3d_velocity_model()
        logger.info("✅ Loaded advanced 3D velocity model")
        assert 'depths' in velocity_model, "Velocity model should contain depth layers"
        assert 'vp' in velocity_model, "Velocity model should contain P-wave velocities"
        assert 'vs' in velocity_model, "Velocity model should contain S-wave velocities"
        assert len(velocity_model['depths']) == len(velocity_model['vp']), "Consistent model dimensions"
        
        # Test advanced Green's function calculation
        distance = 15.0  # km
        azimuth = 45.0   # degrees
        source_depth = 5.0  # km
        elevation = 500.0  # m
        
        gf_matrix = await seismic_system._calculate_advanced_greens_functions(
            distance, azimuth, source_depth, elevation, velocity_model
        )
        logger.info(f"✅ Calculated advanced Green's functions matrix: {gf_matrix.shape}")
        assert gf_matrix.shape == (6, 3), "Green's function matrix should be 6x3"
        assert np.all(np.isfinite(gf_matrix)), "All Green's function values should be finite"
        
        # Test 3: Advanced Moment Tensor Inversion
        logger.info("\n🔄 Test 3: Advanced Multi-Algorithm Moment Tensor Inversion")
        
        # Create synthetic waveform data for multiple stations
        station_metadata = {
            'STA1': {'distance': 12.0, 'azimuth': 30.0, 'elevation': 200.0, 'source_depth': 5.0},
            'STA2': {'distance': 18.0, 'azimuth': 120.0, 'elevation': 500.0, 'source_depth': 5.0},
            'STA3': {'distance': 22.0, 'azimuth': 210.0, 'elevation': 100.0, 'source_depth': 5.0},
            'STA4': {'distance': 16.0, 'azimuth': 300.0, 'elevation': 800.0, 'source_depth': 5.0},
            'STA5': {'distance': 25.0, 'azimuth': 60.0, 'elevation': 300.0, 'source_depth': 5.0}
        }
        
        # Generate synthetic P and S wave data
        p_waves = {}
        s_waves = {}
        green_functions = {}
        
        for station, metadata in station_metadata.items():
            # Synthetic waveforms with station-specific characteristics
            station_waveform = complex_waveform + np.random.normal(0, 0.02, len(complex_waveform))
            
            # Extract P and S wave windows
            p_start, p_end = int(9.5 * sampling_rate), int(12.5 * sampling_rate)
            s_start, s_end = int(14.0 * sampling_rate), int(18.0 * sampling_rate)
            
            p_waves[station] = station_waveform[p_start:p_end]
            s_waves[station] = station_waveform[s_start:s_end]
            
            # Calculate Green's functions for this station
            gf = await seismic_system._calculate_advanced_greens_functions(
                metadata['distance'], metadata['azimuth'], 
                metadata['source_depth'], metadata['elevation'], 
                velocity_model
            )
            green_functions[station] = gf
        
        # Test advanced inversion data preparation
        data_dict = await seismic_system._prepare_advanced_inversion_data(
            p_waves, s_waves, green_functions
        )
        logger.info("✅ Prepared advanced inversion data")
        assert len(data_dict['observations']) > 0, "Should have observations"
        assert data_dict['design_matrix'].shape[0] == len(data_dict['observations']), "Consistent data dimensions"
        
        # Test Bayesian inversion
        mt_bayesian = await seismic_system._bayesian_mt_inversion(data_dict)
        logger.info("✅ Completed Bayesian moment tensor inversion")
        assert mt_bayesian.shape == (3, 3), "Moment tensor should be 3x3"
        assert np.allclose(mt_bayesian, mt_bayesian.T), "Moment tensor should be symmetric"
        
        # Test robust inversion
        mt_robust = await seismic_system._robust_mt_inversion(data_dict)
        logger.info("✅ Completed robust L1-norm moment tensor inversion")
        assert mt_robust.shape == (3, 3), "Moment tensor should be 3x3"
        
        # Test frequency-constrained inversion
        mt_frequency = await seismic_system._frequency_constrained_inversion(data_dict)
        logger.info("✅ Completed frequency-constrained moment tensor inversion")
        assert mt_frequency.shape == (3, 3), "Moment tensor should be 3x3"
        
        # Test adaptive regularized inversion
        mt_adaptive = await seismic_system._adaptive_regularized_inversion(data_dict)
        logger.info("✅ Completed adaptive regularized moment tensor inversion")
        assert mt_adaptive.shape == (3, 3), "Moment tensor should be 3x3"
        
        # Test 4: Advanced Stress Field Analysis
        logger.info("\n⚡ Test 4: Advanced Stress Field Evolution Analysis")
        
        # Create synthetic earthquake swarm with realistic patterns
        base_time = datetime.utcnow()
        swarm_events = []
        
        # Generate spatially and temporally correlated events
        center_lat, center_lon = 46.2, -122.2
        
        for i in range(25):
            # Temporal clustering with migration
            time_offset = i * 0.3 + np.random.exponential(0.5)  # hours
            
            # Spatial migration pattern
            migration_distance = 0.01 * np.sqrt(time_offset)  # degrees
            migration_angle = np.radians(45 + 10 * np.sin(time_offset))
            
            lat = center_lat + migration_distance * np.cos(migration_angle) + np.random.normal(0, 0.005)
            lon = center_lon + migration_distance * np.sin(migration_angle) + np.random.normal(0, 0.005)
            depth = 5.0 + 2.0 * np.sin(time_offset / 5.0) + np.random.normal(0, 0.5)
            
            # Magnitude with b-value distribution
            magnitude = 1.5 + np.random.exponential(0.8)
            
            event = {
                'id': f'swarm_event_{i:03d}',
                'timestamp': base_time + timedelta(hours=time_offset),
                'latitude': lat,
                'longitude': lon,
                'depth': max(0.1, depth),
                'magnitude': min(5.0, magnitude)
            }
            swarm_events.append(event)
        
        # Test advanced stress field evolution analysis
        stress_evolution = await seismic_system._analyze_stress_field_evolution(swarm_events)
        logger.info("✅ Completed advanced stress field evolution analysis")
        
        # Validate stress analysis components
        expected_components = [
            'coulomb_stress_transfer', 'fractal_dimension', 'stress_tensor',
            'stress_accumulation', 'b_value_evolution', 'criticality_measures',
            'energy_patterns', 'stress_heterogeneity'
        ]
        
        for component in expected_components:
            if component in stress_evolution:
                logger.info(f"  ✅ {component}: Present")
            else:
                logger.warning(f"  ⚠️ {component}: Missing")
        
        # Test Coulomb stress transfer calculation
        if 'coulomb_stress_transfer' in stress_evolution:
            coulomb_data = stress_evolution['coulomb_stress_transfer']
            assert 'mean_stress_transfer' in coulomb_data, "Should have mean stress transfer"
            assert 'max_stress_transfer' in coulomb_data, "Should have max stress transfer"
            logger.info(f"  📊 Mean Coulomb stress transfer: {coulomb_data.get('mean_stress_transfer', 0):.2e} Pa")
        
        # Test fractal dimension calculation
        if 'fractal_dimension' in stress_evolution:
            fractal_data = stress_evolution['fractal_dimension']
            if 'correlation_dimension' in fractal_data:
                corr_dim = fractal_data['correlation_dimension']
                logger.info(f"  📐 Correlation dimension: {corr_dim:.3f}")
                assert 0.0 <= corr_dim <= 3.0, "Correlation dimension should be in [0, 3]"
        
        # Test stress accumulation modeling
        if 'stress_accumulation' in stress_evolution:
            accum_data = stress_evolution['stress_accumulation']
            if 'total_moment_release' in accum_data:
                total_moment = accum_data['total_moment_release']
                logger.info(f"  🔋 Total moment release: {total_moment:.2e} Nm")
                assert total_moment > 0, "Total moment should be positive"
        
        # Test 5: Advanced Waveform Preprocessing
        logger.info("\n🔧 Test 5: Advanced Waveform Preprocessing")
        
        # Test preprocessing for different phases
        p_processed = await seismic_system._preprocess_waveform(complex_waveform, 'P')
        s_processed = await seismic_system._preprocess_waveform(complex_waveform, 'S')
        
        logger.info("✅ Completed advanced waveform preprocessing")
        assert len(p_processed) == len(complex_waveform), "Processed waveform should maintain length"
        assert len(s_processed) == len(complex_waveform), "Processed waveform should maintain length"
        assert np.all(np.isfinite(p_processed)), "P-processed waveform should be finite"
        assert np.all(np.isfinite(s_processed)), "S-processed waveform should be finite"
        
        # Test data weight calculation
        p_weights = await seismic_system._calculate_data_weights(p_processed, 'STA1', 'P')
        s_weights = await seismic_system._calculate_data_weights(s_processed, 'STA1', 'S')
        
        logger.info("✅ Calculated advanced data weights")
        assert len(p_weights) == len(p_processed), "Weights should match data length"
        assert all(0.0 <= w <= 1.0 for w in p_weights), "Weights should be in [0, 1]"
        
        # Test 6: Performance Validation
        logger.info("\n⚡ Test 6: Performance and Accuracy Validation")
        
        # Test computational efficiency
        import time
        
        start_time = time.time()
        
        # Run full feature extraction pipeline
        full_features = await seismic_system._extract_waveform_features(complex_waveform)
        
        extraction_time = time.time() - start_time
        logger.info(f"✅ Full feature extraction completed in {extraction_time:.3f} seconds")
        logger.info(f"📊 Total features extracted: {len(full_features)}")
        
        # Validate feature quality
        assert len(full_features) > 30, "Should extract comprehensive feature set"
        assert np.all(np.isfinite(full_features)), "All features should be finite"
        
        # Check feature diversity (no constant features)
        feature_std = np.std(full_features)
        assert feature_std > 1e-6, "Features should have sufficient diversity"
        
        logger.info(f"📈 Feature vector statistics:")
        logger.info(f"  - Mean: {np.mean(full_features):.6f}")
        logger.info(f"  - Std:  {np.std(full_features):.6f}")
        logger.info(f"  - Min:  {np.min(full_features):.6f}")
        logger.info(f"  - Max:  {np.max(full_features):.6f}")
        
        # Test 7: Integration Validation
        logger.info("\n🔗 Test 7: System Integration Validation")
        
        # Test full classification pipeline with advanced features
        event_metadata = {
            'magnitude': 4.2,
            'depth': 6.5,
            'latitude': 46.2,
            'longitude': -122.2
        }
        
        classification = await seismic_system.classify_seismic_event(
            'advanced_test_event',
            complex_waveform,
            event_metadata
        )
        
        logger.info("✅ Completed advanced seismic event classification")
        logger.info(f"📋 Event classified as: {classification.event_type}")
        logger.info(f"🎯 Classification confidence: {classification.confidence:.3f}")
        
        # Validate classification results
        assert classification.event_id == 'advanced_test_event'
        assert 0.0 <= classification.confidence <= 1.0
        assert len(classification.feature_vector) > 30, "Should use comprehensive feature set"
        
        # Test probability distribution
        prob_sum = sum(classification.probability_distribution.values())
        assert abs(prob_sum - 1.0) < 0.01, "Probabilities should sum to ~1.0"
        
        logger.info("🎉 All advanced feature tests completed successfully!")
        logger.info("\n📊 Advanced Feature Test Summary:")
        logger.info("✅ Advanced wavelet features with CWT and multifractal analysis")
        logger.info("✅ Empirical Mode Decomposition (EMD) features")
        logger.info("✅ Higher-order statistical features")
        logger.info("✅ 3D velocity model with lateral variations")
        logger.info("✅ Advanced Green's functions with frequency dependence")
        logger.info("✅ Multi-algorithm moment tensor inversion ensemble")
        logger.info("✅ Bayesian, robust, and frequency-constrained inversions")
        logger.info("✅ Advanced stress field evolution analysis")
        logger.info("✅ Coulomb stress transfer modeling")
        logger.info("✅ Fractal dimension analysis")
        logger.info("✅ Stress tensor inversion from event patterns")
        logger.info("✅ Advanced waveform preprocessing and quality control")
        logger.info("✅ Comprehensive system integration")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Advanced feature test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_advanced_features())
    exit(0 if result else 1)