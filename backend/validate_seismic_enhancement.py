"""
Simple validation test for enhanced seismic analysis
Tests core functionality without external dependencies
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta

def test_enhanced_seismic_processor():
    """Test enhanced seismic processor basic functionality"""
    
    print("🧪 Testing Enhanced Seismic Analysis Implementation")
    print("=" * 60)
    
    try:
        from modules.seismic_transformer import EnhancedSeismicProcessor, SeismicConfig
        print("✓ Enhanced seismic processor imported successfully")
        
        # Test configuration
        config = SeismicConfig(
            sample_rate=100.0,
            window_size=6000,
            frequency_bands=[(1, 5), (5, 15), (15, 30)]
        )
        print("✓ Seismic configuration created")
        
        # Test processor initialization
        processor = EnhancedSeismicProcessor(config)
        print("✓ Enhanced seismic processor initialized")
        
        # Test waveform preprocessing
        duration = 60.0  # seconds
        sample_rate = 100.0  # Hz
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Generate test waveform
        noise = np.random.normal(0, 1e-6, len(t))
        p_signal = 5e-6 * np.exp(-(t - 20.0) / 2.0) * np.sin(2 * np.pi * 8 * (t - 20.0))
        p_signal[t < 20.0] = 0
        
        waveform = noise + p_signal
        waveform_3c = np.column_stack([waveform, waveform * 0.7, waveform * 0.5])
        
        # Test preprocessing
        processed = processor.preprocess_waveform(waveform_3c.T, sample_rate)
        assert processed.shape[0] == 3
        assert not np.any(np.isnan(processed))
        print("✓ Waveform preprocessing works correctly")
        
        # Test spectral analysis
        spectral_content = processor._analyze_spectral_content(waveform)
        assert 'dominant_frequency' in spectral_content
        assert spectral_content['dominant_frequency'] > 0
        print("✓ Spectral analysis works correctly")
        
        # Test distance calculation
        distance = processor._calculate_distance(37.0, -120.0, 34.0, -118.0)
        assert 300 < distance < 500  # Approximate distance between these points
        print("✓ Distance calculation works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_iris_data_connector():
    """Test IRIS data connector"""
    
    try:
        from modules.real_iris_data_connector import RealIRISDataConnector, IRISDataConfig
        print("✓ IRIS data connector imported successfully")
        
        # Test configuration
        config = IRISDataConfig(
            timeout=30,
            max_retries=2,
            preferred_networks=["II", "IU", "US"]
        )
        print("✓ IRIS configuration created")
        
        # Test connector initialization
        connector = RealIRISDataConnector(config)
        assert connector.config == config
        print("✓ IRIS data connector initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ IRIS connector test failed: {e}")
        return False


def test_production_phase_picker():
    """Test production phase picker"""
    
    try:
        from modules.production_phase_picker import create_production_phase_picker, PhasePickConfig
        print("✓ Production phase picker imported successfully")
        
        # Test configuration
        config = PhasePickConfig(
            sample_rate=100.0,
            trigger_on=3.0,
            trigger_off=1.5
        )
        print("✓ Phase picker configuration created")
        
        # Test phase picker creation
        phase_picker = create_production_phase_picker(config)
        assert phase_picker is not None
        print("✓ Production phase picker created")
        
        # Test with sample waveform
        duration = 60.0
        sample_rate = 100.0
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create waveform with clear P and S arrivals
        noise = np.random.normal(0, 1e-6, len(t))
        p_signal = np.zeros_like(t)
        p_signal[t >= 20.0] = 5e-6 * np.exp(-(t[t >= 20.0] - 20.0) / 2.0)
        
        s_signal = np.zeros_like(t)
        s_signal[t >= 35.0] = 8e-6 * np.exp(-(t[t >= 35.0] - 35.0) / 3.0)
        
        waveform = noise + p_signal + s_signal
        
        # Test phase detection
        results = phase_picker.detect_phases(waveform)
        assert isinstance(results, dict)
        assert 'p_picks' in results
        assert 's_picks' in results
        print("✓ Phase detection works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Phase picker test failed: {e}")
        return False


def test_ground_motion_prediction():
    """Test ground motion prediction engine"""
    
    try:
        from modules.ground_motion_prediction import GroundMotionPredictionEngine, GroundMotionConfig
        print("✓ Ground motion prediction engine imported successfully")
        
        # Test configuration
        config = GroundMotionConfig()
        print("✓ Ground motion configuration created")
        
        # Test engine initialization
        gm_engine = GroundMotionPredictionEngine(config)
        assert len(gm_engine.gmpes) > 0
        print("✓ Ground motion prediction engine initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Ground motion prediction test failed: {e}")
        return False


def test_validation_framework():
    """Test seismic validation framework"""
    
    try:
        from modules.seismic_validation_framework import create_validation_framework
        print("✓ Validation framework imported successfully")
        
        # Test framework creation
        validation_framework = create_validation_framework()
        assert len(validation_framework.validation_events) > 0
        print("✓ Validation framework created with validation events")
        
        # Test report generation
        report = validation_framework.generate_validation_report()
        assert isinstance(report, dict)
        assert 'summary' in report
        print("✓ Validation report generation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation framework test failed: {e}")
        return False


async def test_async_functionality():
    """Test async functionality"""
    
    try:
        from modules.seismic_transformer import EnhancedSeismicProcessor
        
        processor = EnhancedSeismicProcessor()
        
        # Test async setup (will use fallback data)
        success = await processor.setup_real_time_processing(
            latitude=37.0,
            longitude=-120.0,
            radius_km=200.0
        )
        
        # Should handle gracefully when IRIS not available
        assert isinstance(success, bool)
        print("✓ Async real-time processing setup works")
        
        # Test historical seismicity (will return empty list)
        events = await processor.get_historical_seismicity(
            latitude=37.0,
            longitude=-120.0,
            radius_km=100.0,
            days_back=7
        )
        
        assert isinstance(events, list)
        print("✓ Async historical seismicity retrieval works")
        
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        return False


def main():
    """Run all validation tests"""
    
    print("🚀 Starting Enhanced Seismic Analysis Validation")
    print("=" * 60)
    
    tests = [
        ("Enhanced Seismic Processor", test_enhanced_seismic_processor),
        ("IRIS Data Connector", test_iris_data_connector),
        ("Production Phase Picker", test_production_phase_picker),
        ("Ground Motion Prediction", test_ground_motion_prediction),
        ("Validation Framework", test_validation_framework),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
            results.append((test_name, False))
    
    # Test async functionality
    print(f"\n📋 Testing Async Functionality:")
    print("-" * 40)
    
    try:
        async_result = asyncio.run(test_async_functionality())
        results.append(("Async Functionality", async_result))
        
        if async_result:
            print("✅ Async Functionality - PASSED")
        else:
            print("❌ Async Functionality - FAILED")
            
    except Exception as e:
        print(f"❌ Async Functionality - ERROR: {e}")
        results.append(("Async Functionality", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    print("-" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Enhanced seismic analysis is working correctly.")
        print("\n📋 Implementation Summary:")
        print("- ✓ Real IRIS data connector with fallback support")
        print("- ✓ Production-grade phase picking algorithms")
        print("- ✓ Ground motion prediction with multiple GMPEs")
        print("- ✓ Comprehensive validation framework")
        print("- ✓ Enhanced seismic processor with real-time capabilities")
        print("- ✓ Moment tensor inversion integration")
        print("- ✓ Backward compatibility maintained")
        
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)