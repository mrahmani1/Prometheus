"""
Simple test for Advanced Seismic Source Characterization System
"""

import sys
import os
import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_seismic_system():
    """Test the seismic characterization system"""
    try:
        # Import the system
        from core.advanced_seismic_source_characterization import AdvancedSeismicSourceCharacterization
        
        # Create a mock database manager
        class MockDBManager:
            async def get_postgresql_session(self):
                return None
        
        # Initialize the system
        db_manager = MockDBManager()
        seismic_system = AdvancedSeismicSourceCharacterization(db_manager)
        
        logger.info("✅ Successfully created AdvancedSeismicSourceCharacterization instance")
        
        # Test waveform feature extraction
        # Create synthetic waveform
        duration = 30.0  # seconds
        sampling_rate = 100.0  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Background noise + P wave + S wave
        noise = np.random.normal(0, 0.1, len(t))
        p_wave = np.exp(-(t - 5.0) / 2.0) * np.sin(2 * np.pi * 10 * (t - 5.0))
        p_wave[t < 5.0] = 0
        s_wave = np.exp(-(t - 8.0) / 4.0) * np.sin(2 * np.pi * 5 * (t - 8.0))
        s_wave[t < 8.0] = 0
        
        waveform = noise + 0.5 * p_wave + 0.8 * s_wave
        
        # Test feature extraction
        features = await seismic_system._extract_waveform_features(waveform)
        logger.info(f"✅ Extracted {len(features)} features from synthetic waveform")
        
        # Test event classification
        event_metadata = {
            'magnitude': 3.5,
            'depth': 8.0,
            'latitude': 46.2,
            'longitude': -122.2
        }
        
        classification = await seismic_system.classify_seismic_event(
            'test_event_001',
            waveform,
            event_metadata
        )
        
        logger.info(f"✅ Event classified as {classification.event_type} with confidence {classification.confidence:.3f}")
        
        # Test earthquake swarm analysis
        base_time = datetime.utcnow()
        sample_events = []
        
        for i in range(15):
            event = {
                'id': f'event_{i:03d}',
                'timestamp': base_time + timedelta(hours=i * 0.5),
                'latitude': 46.2 + np.random.normal(0, 0.01),
                'longitude': -122.2 + np.random.normal(0, 0.01),
                'depth': 5.0 + np.random.normal(0, 1.0),
                'magnitude': 2.0 + np.random.exponential(0.5)
            }
            sample_events.append(event)
        
        swarms = await seismic_system.analyze_earthquake_swarm(
            sample_events,
            time_window=timedelta(days=2),
            spatial_threshold=10.0
        )
        
        logger.info(f"✅ Identified {len(swarms)} earthquake swarms from {len(sample_events)} events")
        
        # Test focal mechanism analysis
        sample_focal_mechanisms = []
        for i in range(8):
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
            sample_focal_mechanisms.append(mechanism)
        
        stress_field = await seismic_system.characterize_stress_field(
            sample_focal_mechanisms,
            'test_region'
        )
        
        logger.info(f"✅ Characterized stress field: {stress_field.stress_regime} regime")
        logger.info(f"   Stress ratio: {stress_field.stress_ratio:.3f}")
        logger.info(f"   Spatial coherence: {stress_field.spatial_coherence:.3f}")
        
        # Test 3D visualization creation
        visualization = await seismic_system.create_3d_visualization(
            sample_events,
            sample_focal_mechanisms,
            stress_field
        )
        
        logger.info(f"✅ Created 3D visualization with {len(visualization.event_locations)} events")
        
        # Test real-time processing
        real_time_event = {
            'id': 'realtime_001',
            'timestamp': datetime.utcnow(),
            'latitude': 46.2,
            'longitude': -122.2,
            'depth': 5.0,
            'magnitude': 3.2
        }
        
        results = await seismic_system.process_real_time_event(
            real_time_event,
            waveform
        )
        
        logger.info(f"✅ Real-time processing completed for event {results['event_id']}")
        
        logger.info("🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_seismic_system())
    exit(0 if result else 1)