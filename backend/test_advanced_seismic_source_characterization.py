"""
Test suite for Advanced Seismic Source Characterization System

This test suite validates the functionality of the advanced seismic analysis
capabilities including moment tensor inversion, ML classification, swarm analysis,
and stress field characterization.
"""

import pytest
import numpy as np
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import logging

# Import the system under test
from core.advanced_seismic_source_characterization import (
    AdvancedSeismicSourceCharacterization,
    MomentTensorSolution,
    SeismicEventClassification,
    EarthquakeSwarm,
    StressFieldCharacterization,
    SeismicSourceVisualization
)
from core.database_integration import DatabaseManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAdvancedSeismicSourceCharacterization:
    """Test class for Advanced Seismic Source Characterization"""
    
    @pytest.fixture
    async def mock_db_manager(self):
        """Create mock database manager"""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.get_postgresql_session = AsyncMock()
        return db_manager
    
    @pytest.fixture
    async def seismic_system(self, mock_db_manager):
        """Create seismic characterization system instance"""
        system = AdvancedSeismicSourceCharacterization(mock_db_manager)
        await system.initialize_ml_models()
        return system
    
    @pytest.fixture
    def sample_waveform(self):
        """Generate sample seismic waveform"""
        # Create synthetic waveform with P and S arrivals
        duration = 30.0  # seconds
        sampling_rate = 100.0  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Background noise
        noise = np.random.normal(0, 0.1, len(t))
        
        # P wave arrival at 5 seconds
        p_arrival = 5.0
        p_mask = t >= p_arrival
        p_wave = np.exp(-(t - p_arrival) / 2.0) * np.sin(2 * np.pi * 10 * (t - p_arrival))
        p_wave[~p_mask] = 0
        
        # S wave arrival at 8 seconds
        s_arrival = 8.0
        s_mask = t >= s_arrival
        s_wave = np.exp(-(t - s_arrival) / 4.0) * np.sin(2 * np.pi * 5 * (t - s_arrival))
        s_wave[~s_mask] = 0
        
        # Combine components
        waveform = noise + 0.5 * p_wave + 0.8 * s_wave
        
        return waveform
    
    @pytest.fixture
    def sample_station_metadata(self):
        """Generate sample station metadata"""
        return {
            'STA1': {
                'distance': 10.0,  # km
                'azimuth': 45.0,   # degrees
                'response': {'gain': 1.0}
            },
            'STA2': {
                'distance': 15.0,
                'azimuth': 135.0,
                'response': {'gain': 1.2}
            },
            'STA3': {
                'distance': 20.0,
                'azimuth': 225.0,
                'response': {'gain': 0.8}
            },
            'STA4': {
                'distance': 12.0,
                'azimuth': 315.0,
                'response': {'gain': 1.1}
            }
        }
    
    @pytest.fixture
    def sample_events(self):
        """Generate sample seismic events for swarm analysis"""
        base_time = datetime.utcnow()
        events = []
        
        # Create a cluster of events (swarm)
        for i in range(20):
            event = {
                'id': f'event_{i:03d}',
                'timestamp': base_time + timedelta(hours=i * 0.5),
                'latitude': 46.2 + np.random.normal(0, 0.01),
                'longitude': -122.2 + np.random.normal(0, 0.01),
                'depth': 5.0 + np.random.normal(0, 1.0),
                'magnitude': 2.0 + np.random.exponential(0.5)
            }
            events.append(event)
        
        return events
    
    @pytest.fixture
    def sample_focal_mechanisms(self):
        """Generate sample focal mechanisms"""
        mechanisms = []
        
        for i in range(10):
            mechanism = {
                'id': f'fm_{i:03d}',
                'latitude': 46.2 + np.random.normal(0, 0.02),
                'longitude': -122.2 + np.random.normal(0, 0.02),
                'depth': 5.0 + np.random.normal(0, 2.0),
                'strike': np.random.uniform(0, 360),
                'dip': np.random.uniform(30, 90),
                'rake': np.random.uniform(-180, 180),
                'magnitude': 3.0 + np.random.exponential(0.3),
                'timestamp': datetime.utcnow() - timedelta(days=np.random.randint(0, 30))
            }
            mechanisms.append(mechanism)
        
        return mechanisms

    async def test_waveform_feature_extraction(self, seismic_system, sample_waveform):
        """Test waveform feature extraction"""
        logger.info("Testing waveform feature extraction")
        
        features = await seismic_system._extract_waveform_features(sample_waveform)
        
        # Verify feature vector properties
        assert isinstance(features, np.ndarray)
        assert len(features) > 10  # Should have multiple features
        assert np.all(np.isfinite(features))  # No NaN or infinite values
        
        # Check specific feature ranges
        assert features[0] > 0  # Peak amplitude should be positive
        assert features[1] > 0  # Mean amplitude should be positive
        
        logger.info(f"Extracted {len(features)} features from waveform")

    async def test_seismic_event_classification(self, seismic_system, sample_waveform):
        """Test seismic event classification"""
        logger.info("Testing seismic event classification")
        
        event_metadata = {
            'magnitude': 3.5,
            'depth': 8.0,
            'latitude': 46.2,
            'longitude': -122.2
        }
        
        classification = await seismic_system.classify_seismic_event(
            'test_event_001',
            sample_waveform,
            event_metadata
        )
        
        # Verify classification results
        assert isinstance(classification, SeismicEventClassification)
        assert classification.event_id == 'test_event_001'
        assert classification.event_type in ['vt', 'lp', 'vlp', 'hybrid', 'explosion', 'tremor', 'unknown']
        assert 0.0 <= classification.confidence <= 1.0
        assert isinstance(classification.probability_distribution, dict)
        assert len(classification.feature_vector) > 0
        
        logger.info(f"Event classified as {classification.event_type} with confidence {classification.confidence:.3f}")

    async def test_moment_tensor_inversion(self, seismic_system, sample_waveform, sample_station_metadata):
        """Test moment tensor inversion"""
        logger.info("Testing moment tensor inversion")
        
        # Create waveform data for multiple stations
        waveform_data = {}
        for station in sample_station_metadata.keys():
            # Add some station-specific variation
            station_waveform = sample_waveform + np.random.normal(0, 0.05, len(sample_waveform))
            waveform_data[station] = {'waveform': station_waveform}
        
        solution = await seismic_system.invert_moment_tensor(
            'test_event_mt',
            waveform_data,
            sample_station_metadata
        )
        
        # Verify moment tensor solution
        assert isinstance(solution, MomentTensorSolution)
        assert -180.0 <= solution.strike <= 360.0
        assert 0.0 <= solution.dip <= 90.0
        assert -180.0 <= solution.rake <= 180.0
        assert 0.0 <= solution.dc_percentage <= 100.0
        assert 0.0 <= solution.clvd_percentage <= 100.0
        assert 0.0 <= solution.iso_percentage <= 100.0
        assert abs(solution.dc_percentage + solution.clvd_percentage + solution.iso_percentage - 100.0) < 1.0
        
        # Verify uncertainty bounds
        assert isinstance(solution.uncertainty_bounds, dict)
        assert 'mxx' in solution.uncertainty_bounds
        
        # Verify quality metrics
        assert isinstance(solution.quality_metrics, dict)
        assert 'variance_reduction' in solution.quality_metrics
        
        logger.info(f"Moment tensor: Strike={solution.strike:.1f}°, Dip={solution.dip:.1f}°, Rake={solution.rake:.1f}°")
        logger.info(f"DC={solution.dc_percentage:.1f}%, CLVD={solution.clvd_percentage:.1f}%, ISO={solution.iso_percentage:.1f}%")

    async def test_earthquake_swarm_analysis(self, seismic_system, sample_events):
        """Test earthquake swarm analysis"""
        logger.info("Testing earthquake swarm analysis")
        
        swarms = await seismic_system.analyze_earthquake_swarm(
            sample_events,
            time_window=timedelta(days=2),
            spatial_threshold=10.0
        )
        
        # Verify swarm analysis results
        assert isinstance(swarms, list)
        
        if len(swarms) > 0:
            swarm = swarms[0]
            assert isinstance(swarm, EarthquakeSwarm)
            assert len(swarm.events) >= 3  # Minimum events for a swarm
            assert swarm.start_time <= swarm.end_time
            assert len(swarm.centroid_location) == 3  # lat, lon, depth
            assert isinstance(swarm.spatial_extent, dict)
            assert isinstance(swarm.temporal_evolution, dict)
            assert 0.0 <= swarm.significance_score <= 1.0
            
            logger.info(f"Identified {len(swarms)} swarms with {len(swarm.events)} events in first swarm")
            logger.info(f"Swarm significance score: {swarm.significance_score:.3f}")
        else:
            logger.info("No swarms identified in sample events")

    async def test_stress_field_characterization(self, seismic_system, sample_focal_mechanisms):
        """Test stress field characterization"""
        logger.info("Testing stress field characterization")
        
        stress_field = await seismic_system.characterize_stress_field(
            sample_focal_mechanisms,
            'test_region'
        )
        
        # Verify stress field characterization
        assert isinstance(stress_field, StressFieldCharacterization)
        assert stress_field.region_id == 'test_region'
        assert isinstance(stress_field.principal_stress_axes, dict)
        assert 'sigma1' in stress_field.principal_stress_axes
        assert 'sigma2' in stress_field.principal_stress_axes
        assert 'sigma3' in stress_field.principal_stress_axes
        assert 0.0 <= stress_field.stress_ratio <= 1.0
        assert stress_field.stress_regime in ['normal', 'strike_slip', 'thrust', 'oblique', 'unknown']
        assert 0.0 <= stress_field.spatial_coherence <= 1.0
        assert 0.0 <= stress_field.temporal_stability <= 1.0
        
        # Verify principal stress axes
        for axis_name in ['sigma1', 'sigma2', 'sigma3']:
            axis = stress_field.principal_stress_axes[axis_name]
            assert 0.0 <= axis['trend'] <= 360.0
            assert 0.0 <= axis['plunge'] <= 90.0
        
        logger.info(f"Stress regime: {stress_field.stress_regime}")
        logger.info(f"Stress ratio: {stress_field.stress_ratio:.3f}")
        logger.info(f"Spatial coherence: {stress_field.spatial_coherence:.3f}")

    async def test_3d_visualization_creation(self, seismic_system, sample_events, sample_focal_mechanisms):
        """Test 3D seismic source visualization creation"""
        logger.info("Testing 3D visualization creation")
        
        # Create stress field for visualization
        stress_field = await seismic_system.characterize_stress_field(
            sample_focal_mechanisms,
            'test_region'
        )
        
        visualization = await seismic_system.create_3d_visualization(
            sample_events,
            sample_focal_mechanisms,
            stress_field
        )
        
        # Verify visualization data
        assert isinstance(visualization, SeismicSourceVisualization)
        assert isinstance(visualization.event_locations, np.ndarray)
        assert visualization.event_locations.shape[0] == len(sample_events)
        assert visualization.event_locations.shape[1] == 3  # lat, lon, depth
        
        assert isinstance(visualization.focal_mechanisms, list)
        assert len(visualization.focal_mechanisms) == len(sample_focal_mechanisms)
        
        assert isinstance(visualization.uncertainty_ellipsoids, list)
        assert len(visualization.uncertainty_ellipsoids) == len(sample_events)
        
        assert isinstance(visualization.stress_field_vectors, np.ndarray)
        if len(visualization.stress_field_vectors) > 0:
            assert visualization.stress_field_vectors.shape[1] == 3  # 3D vectors
        
        assert isinstance(visualization.temporal_evolution, dict)
        assert isinstance(visualization.visualization_metadata, dict)
        
        logger.info(f"Created visualization with {len(sample_events)} events and {len(sample_focal_mechanisms)} focal mechanisms")

    async def test_real_time_event_processing(self, seismic_system, sample_waveform):
        """Test real-time seismic event processing"""
        logger.info("Testing real-time event processing")
        
        event_data = {
            'id': 'realtime_001',
            'timestamp': datetime.utcnow(),
            'latitude': 46.2,
            'longitude': -122.2,
            'depth': 5.0,
            'magnitude': 3.2
        }
        
        results = await seismic_system.process_real_time_event(
            event_data,
            sample_waveform
        )
        
        # Verify processing results
        assert isinstance(results, dict)
        assert results['event_id'] == 'realtime_001'
        assert 'processing_timestamp' in results
        assert 'classification' in results
        assert 'alerts' in results
        
        if results['classification']:
            classification = results['classification']
            assert 'event_type' in classification
            assert 'confidence' in classification
            assert 'probability_distribution' in classification
        
        logger.info(f"Real-time processing completed for event {results['event_id']}")
        if results['classification']:
            logger.info(f"Classification: {results['classification']['event_type']} (confidence: {results['classification']['confidence']:.3f})")

    async def test_validation_against_known_mechanisms(self, seismic_system):
        """Test validation against known seismic source mechanisms"""
        logger.info("Testing validation against known mechanisms")
        
        # Create sample moment tensor solution
        solution = MomentTensorSolution(
            mxx=1.0, myy=-0.5, mzz=-0.5,
            mxy=0.2, mxz=0.1, myz=0.3,
            magnitude=3.5,
            strike=45.0, dip=60.0, rake=90.0,
            dc_percentage=85.0, clvd_percentage=10.0, iso_percentage=5.0,
            uncertainty_bounds={},
            quality_metrics={},
            station_coverage={}
        )
        
        # Create reference solutions
        reference_solutions = [
            {
                'id': 'ref_001',
                'type': 'normal_fault',
                'strike': 50.0,
                'dip': 65.0,
                'rake': 85.0,
                'dc_percentage': 90.0
            },
            {
                'id': 'ref_002',
                'type': 'strike_slip',
                'strike': 120.0,
                'dip': 80.0,
                'rake': 10.0,
                'dc_percentage': 95.0
            }
        ]
        
        validation_metrics = await seismic_system.validate_against_known_mechanisms(
            solution,
            reference_solutions
        )
        
        # Verify validation results
        assert isinstance(validation_metrics, dict)
        assert 'similarity_scores' in validation_metrics
        assert 'mechanism_matches' in validation_metrics
        assert 'confidence_score' in validation_metrics
        assert 'validation_quality' in validation_metrics
        
        assert isinstance(validation_metrics['similarity_scores'], list)
        assert len(validation_metrics['similarity_scores']) == len(reference_solutions)
        assert 0.0 <= validation_metrics['confidence_score'] <= 1.0
        assert validation_metrics['validation_quality'] in ['excellent', 'good', 'fair', 'poor', 'error']
        
        logger.info(f"Validation confidence score: {validation_metrics['confidence_score']:.3f}")
        logger.info(f"Validation quality: {validation_metrics['validation_quality']}")

    async def test_wave_arrival_detection(self, seismic_system, sample_waveform):
        """Test P and S wave arrival detection"""
        logger.info("Testing wave arrival detection")
        
        p_arrival, s_arrival = await seismic_system._detect_wave_arrivals(sample_waveform)
        
        # Verify arrival detection
        if p_arrival is not None:
            assert isinstance(p_arrival, (int, np.integer))
            assert 0 <= p_arrival < len(sample_waveform)
            logger.info(f"P wave arrival detected at sample {p_arrival} ({p_arrival/100.0:.2f} seconds)")
        
        if s_arrival is not None:
            assert isinstance(s_arrival, (int, np.integer))
            assert 0 <= s_arrival < len(sample_waveform)
            logger.info(f"S wave arrival detected at sample {s_arrival} ({s_arrival/100.0:.2f} seconds)")
        
        # S arrival should be after P arrival if both detected
        if p_arrival is not None and s_arrival is not None:
            assert s_arrival > p_arrival

    async def test_spectral_analysis(self, seismic_system, sample_waveform):
        """Test spectral analysis of waveforms"""
        logger.info("Testing spectral analysis")
        
        spectral_properties = await seismic_system._calculate_spectral_properties(sample_waveform)
        
        # Verify spectral properties
        assert isinstance(spectral_properties, dict)
        assert 'dominant_frequency' in spectral_properties
        assert 'spectral_centroid' in spectral_properties
        assert 'spectral_spread' in spectral_properties
        assert 'spectral_entropy' in spectral_properties
        
        # Check frequency band energy ratios
        for band in ['vt', 'lp', 'vlp', 'hybrid', 'explosion', 'tremor']:
            assert f'{band}_energy_ratio' in spectral_properties
            assert 0.0 <= spectral_properties[f'{band}_energy_ratio'] <= 1.0
        
        # Verify reasonable values
        assert spectral_properties['dominant_frequency'] > 0
        assert spectral_properties['spectral_centroid'] > 0
        assert spectral_properties['spectral_entropy'] >= 0
        
        logger.info(f"Dominant frequency: {spectral_properties['dominant_frequency']:.2f} Hz")
        logger.info(f"Spectral centroid: {spectral_properties['spectral_centroid']:.2f} Hz")

    async def test_temporal_analysis(self, seismic_system, sample_waveform):
        """Test temporal analysis of waveforms"""
        logger.info("Testing temporal analysis")
        
        temporal_features = await seismic_system._calculate_temporal_features(sample_waveform)
        
        # Verify temporal features
        assert isinstance(temporal_features, dict)
        assert 'envelope_peak' in temporal_features
        assert 'envelope_mean' in temporal_features
        assert 'attack_time' in temporal_features
        assert 'decay_time' in temporal_features
        assert 'temporal_centroid' in temporal_features
        
        # Verify reasonable values
        assert temporal_features['envelope_peak'] > 0
        assert temporal_features['envelope_mean'] > 0
        assert temporal_features['attack_time'] >= 0
        assert temporal_features['decay_time'] >= 0
        assert temporal_features['temporal_centroid'] >= 0
        
        logger.info(f"Attack time: {temporal_features['attack_time']:.2f} seconds")
        logger.info(f"Decay time: {temporal_features['decay_time']:.2f} seconds")

    async def test_error_handling(self, seismic_system):
        """Test error handling in various scenarios"""
        logger.info("Testing error handling")
        
        # Test with empty waveform
        empty_waveform = np.array([])
        classification = await seismic_system.classify_seismic_event(
            'empty_test',
            empty_waveform,
            {}
        )
        assert classification.event_type == "unknown"
        assert classification.confidence == 0.0
        
        # Test with insufficient stations for moment tensor
        insufficient_waveform_data = {'STA1': {'waveform': np.random.randn(1000)}}
        insufficient_station_metadata = {'STA1': {'distance': 10.0, 'azimuth': 0.0}}
        
        try:
            await seismic_system.invert_moment_tensor(
                'insufficient_test',
                insufficient_waveform_data,
                insufficient_station_metadata
            )
            assert False, "Should have raised ValueError for insufficient stations"
        except ValueError as e:
            assert "Minimum 4 stations required" in str(e)
        
        # Test with empty events for swarm analysis
        empty_swarms = await seismic_system.analyze_earthquake_swarm([])
        assert len(empty_swarms) == 0
        
        logger.info("Error handling tests completed successfully")

# Integration test function
async def run_integration_test():
    """Run integration test of the seismic characterization system"""
    logger.info("Starting integration test of Advanced Seismic Source Characterization")
    
    # Create test instance
    test_instance = TestAdvancedSeismicSourceCharacterization()
    
    # Create fixtures
    mock_db = await test_instance.mock_db_manager()
    system = await test_instance.seismic_system(mock_db)
    waveform = test_instance.sample_waveform()
    events = test_instance.sample_events()
    focal_mechanisms = test_instance.sample_focal_mechanisms()
    
    try:
        # Run core functionality tests
        await test_instance.test_waveform_feature_extraction(system, waveform)
        await test_instance.test_seismic_event_classification(system, waveform)
        await test_instance.test_earthquake_swarm_analysis(system, events)
        await test_instance.test_stress_field_characterization(system, focal_mechanisms)
        await test_instance.test_3d_visualization_creation(system, events, focal_mechanisms)
        await test_instance.test_real_time_event_processing(system, waveform)
        
        logger.info("✅ All integration tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the integration test
    result = asyncio.run(run_integration_test())
    exit(0 if result else 1)