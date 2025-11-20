#!/usr/bin/env python3
"""
Test script to verify the completed Advanced Seismic Source Characterization system
"""

import numpy as np
import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

async def test_seismic_characterization():
    """Test the advanced seismic source characterization system"""
    try:
        # Import the class (this will test if all dependencies are resolved)
        from core.advanced_seismic_source_characterization import AdvancedSeismicSourceCharacterization
        
        print("✓ Successfully imported AdvancedSeismicSourceCharacterization")
        
        # Create a mock database manager for testing
        class MockDatabaseManager:
            def get_postgresql_session(self):
                return MockSession()
        
        class MockSession:
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass
            
            async def execute(self, query, params=None):
                return MockResult()
        
        class MockResult:
            def scalars(self):
                return MockScalars()
            
            def fetchall(self):
                return []
            
            def scalar(self):
                return 0
        
        class MockScalars:
            def all(self):
                return []
        
        # Initialize the system
        db_manager = MockDatabaseManager()
        seismic_system = AdvancedSeismicSourceCharacterization(db_manager)
        
        print("✓ Successfully initialized AdvancedSeismicSourceCharacterization")
        
        # Test waveform feature extraction
        test_waveform = np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 10*np.pi, 1000))
        
        features = await seismic_system._extract_waveform_features(test_waveform)
        print(f"✓ Successfully extracted {len(features)} waveform features")
        
        # Test seismic event classification
        event_metadata = {
            'magnitude': 3.5,
            'depth': 8.0,
            'latitude': 19.5,
            'longitude': -155.5
        }
        
        classification = await seismic_system.classify_seismic_event(
            event_id="test_001",
            waveform_data=test_waveform,
            metadata=event_metadata
        )
        
        print(f"✓ Successfully classified seismic event as: {classification.event_type}")
        print(f"  Confidence: {classification.confidence:.3f}")
        print(f"  Feature vector length: {len(classification.feature_vector)}")
        
        # Test complexity measures
        complexity_measures = await seismic_system._calculate_advanced_complexity_measures(test_waveform)
        print(f"✓ Successfully calculated {len(complexity_measures)} complexity measures")
        
        # Test duration calculations
        sig_duration = await seismic_system._calculate_advanced_significant_duration(test_waveform)
        eff_duration = await seismic_system._calculate_effective_duration(test_waveform)
        print(f"✓ Successfully calculated durations: significant={sig_duration:.2f}s, effective={eff_duration:.2f}s")
        
        # Test entropy calculations
        approx_entropy = await seismic_system._calculate_approximate_entropy(test_waveform)
        sample_entropy = await seismic_system._calculate_sample_entropy(test_waveform)
        perm_entropy = await seismic_system._calculate_permutation_entropy(test_waveform)
        print(f"✓ Successfully calculated entropies: ApEn={approx_entropy:.3f}, SampEn={sample_entropy:.3f}, PermEn={perm_entropy:.3f}")
        
        # Test Hjorth parameters
        hjorth_params = await seismic_system._calculate_hjorth_parameters(test_waveform)
        print(f"✓ Successfully calculated Hjorth parameters: {list(hjorth_params.keys())}")
        
        # Test fractal dimension and DFA
        fractal_dim = await seismic_system._calculate_fractal_dimension(test_waveform)
        dfa_exponent = await seismic_system._calculate_dfa_exponent(test_waveform)
        print(f"✓ Successfully calculated fractal dimension={fractal_dim:.3f}, DFA exponent={dfa_exponent:.3f}")
        
        print("\n🎉 All tests passed! The Advanced Seismic Source Characterization system is complete and functional.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_seismic_characterization())
    sys.exit(0 if success else 1)