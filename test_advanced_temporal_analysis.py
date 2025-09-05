#!/usr/bin/env python3
"""
Test Advanced Temporal Analysis Implementations
Comprehensive validation of the advanced temporal analysis methods
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_geological_data():
    """Generate synthetic geological time series data for testing"""
    np.random.seed(42)
    
    # Generate time series with different geological characteristics
    n_points = 500
    time = np.linspace(0, 10, n_points)
    
    # Seismic activity simulation
    seismic_base = 2.0
    seismic_trend = 0.1 * time
    seismic_seasonal = 0.5 * np.sin(2 * np.pi * time)  # Annual cycle
    seismic_events = np.random.exponential(0.2, n_points) * (np.random.random(n_points) > 0.95)
    seismic_noise = np.random.normal(0, 0.1, n_points)
    
    seismic_activity = seismic_base + seismic_trend + seismic_seasonal + seismic_events + seismic_noise
    
    # Thermal activity simulation  
    thermal_base = 150.0
    thermal_trend = -0.5 * time  # Cooling trend
    thermal_cyclical = 10 * np.sin(0.5 * np.pi * time)  # Long-term cycle
    thermal_noise = np.random.normal(0, 2, n_points)
    
    thermal_activity = thermal_base + thermal_trend + thermal_cyclical + thermal_noise
    
    # Pressure variations simulation
    pressure_base = 1013.25
    pressure_trend = 0.01 * time
    pressure_weather = 5 * np.sin(4 * np.pi * time) + 3 * np.cos(6 * np.pi * time)  # Weather patterns
    pressure_geological = 2 * np.sin(0.2 * np.pi * time)  # Geological processes
    pressure_noise = np.random.normal(0, 1, n_points)
    
    pressure_variations = pressure_base + pressure_trend + pressure_weather + pressure_geological + pressure_noise
    
    # Deformation measurements
    deformation_base = 0.0
    deformation_trend = 0.02 * time**1.5  # Accelerating deformation
    deformation_events = np.zeros(n_points)
    
    # Add some step changes (fault movements)
    event_times = [150, 300, 450]
    for event_time in event_times:
        if event_time < n_points:
            deformation_events[event_time:] += np.random.uniform(0.5, 2.0)
    
    deformation_noise = np.random.normal(0, 0.05, n_points)
    deformation_measurements = deformation_base + deformation_trend + deformation_events + deformation_noise
    
    return {
        'time': time,
        'seismic_activity': seismic_activity,
        'thermal_activity': thermal_activity, 
        'pressure_variations': pressure_variations,
        'deformation_measurements': deformation_measurements
    }

def test_hmm_model():
    """Test the advanced HMM model implementation"""
    logger.info("Testing Advanced HMM Model Implementation...")
    
    try:
        # Import the module
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data
        data = generate_synthetic_geological_data()
        test_values = data['seismic_activity']
        
        # Create integrator instance
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test HMM model
        result = integrator._build_hmm_model(test_values)
        
        # Validate results
        assert isinstance(result, dict), "HMM result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"HMM failed: {result.get('message', 'Unknown error')}"
        
        if 'model' in result and result['model'] != 'fallback_kmeans':
            assert 'n_states' in result, "HMM result should include number of states"
            assert 'state_sequence' in result, "HMM result should include state sequence"
            assert 'transition_matrix' in result, "HMM result should include transition matrix"
            assert 'state_interpretations' in result, "HMM result should include state interpretations"
            
            logger.info(f"HMM Model: {result['n_states']} states detected")
            logger.info(f"State interpretations: {list(result['state_interpretations'].keys())}")
        else:
            logger.info("HMM using fallback implementation")
        
        logger.info("✓ HMM Model test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ HMM Model test failed: {e}")
        return False

def test_temporal_trends():
    """Test the temporal trends analysis implementation"""
    logger.info("Testing Temporal Trends Analysis...")
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data with clear trend
        data = generate_synthetic_geological_data()
        test_values = data['deformation_measurements']  # Has clear trend
        
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test trends analysis
        result = integrator._analyze_temporal_trends(test_values)
        
        # Validate results
        assert isinstance(result, dict), "Trends result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"Trends analysis failed: {result.get('message', 'Unknown error')}"
        
        assert 'trend_analysis' in result, "Result should include trend analysis"
        assert 'geological_interpretation' in result, "Result should include geological interpretation"
        
        trend_analysis = result['trend_analysis']
        assert 'linear' in trend_analysis, "Should include linear trend analysis"
        assert 'robust' in trend_analysis, "Should include robust trend analysis"
        assert 'mann_kendall' in trend_analysis, "Should include Mann-Kendall test"
        
        # Check if linear trend is detected (deformation data has clear trend)
        if 'linear' in trend_analysis and 'slope' in trend_analysis['linear']:
            slope = trend_analysis['linear']['slope']
            logger.info(f"Detected linear trend slope: {slope:.6f}")
        
        logger.info("✓ Temporal Trends Analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Temporal Trends Analysis test failed: {e}")
        return False

def test_seasonal_patterns():
    """Test the seasonal patterns analysis implementation"""
    logger.info("Testing Seasonal Patterns Analysis...")
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data with seasonal pattern
        data = generate_synthetic_geological_data()
        test_values = data['seismic_activity']  # Has seasonal component
        
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test seasonal analysis
        result = integrator._analyze_seasonal_patterns(test_values)
        
        # Validate results
        assert isinstance(result, dict), "Seasonal result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"Seasonal analysis failed: {result.get('message', 'Unknown error')}"
        
        assert 'seasonal_analysis' in result, "Result should include seasonal analysis"
        
        seasonal_analysis = result['seasonal_analysis']
        assert 'fourier_analysis' in seasonal_analysis, "Should include Fourier analysis"
        assert 'autocorrelation' in seasonal_analysis, "Should include autocorrelation analysis"
        
        # Check for detected seasonal periods
        if 'autocorrelation' in seasonal_analysis and 'seasonal_periods' in seasonal_analysis['autocorrelation']:
            periods = seasonal_analysis['autocorrelation']['seasonal_periods']
            if periods:
                logger.info(f"Detected seasonal periods: {[p['period'] for p in periods[:3]]}")
        
        logger.info("✓ Seasonal Patterns Analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Seasonal Patterns Analysis test failed: {e}")
        return False

def test_cyclical_patterns():
    """Test the cyclical patterns analysis implementation"""
    logger.info("Testing Cyclical Patterns Analysis...")
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data with cyclical pattern
        data = generate_synthetic_geological_data()
        test_values = data['thermal_activity']  # Has cyclical component
        
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test cyclical analysis
        result = integrator._analyze_cyclical_patterns(test_values)
        
        # Validate results
        assert isinstance(result, dict), "Cyclical result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"Cyclical analysis failed: {result.get('message', 'Unknown error')}"
        
        assert 'cyclical_analysis' in result, "Result should include cyclical analysis"
        
        cyclical_analysis = result['cyclical_analysis']
        assert 'harmonic_analysis' in cyclical_analysis, "Should include harmonic analysis"
        assert 'cycle_detection' in cyclical_analysis, "Should include cycle detection"
        
        # Check for detected cycles
        if 'harmonic_analysis' in cyclical_analysis and 'harmonics' in cyclical_analysis['harmonic_analysis']:
            harmonics = cyclical_analysis['harmonic_analysis']['harmonics']
            if harmonics:
                dominant_periods = [h['period'] for h in harmonics[:3]]
                logger.info(f"Detected dominant periods: {dominant_periods}")
        
        logger.info("✓ Cyclical Patterns Analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cyclical Patterns Analysis test failed: {e}")
        return False

def test_temporal_causality():
    """Test the temporal causality analysis implementation"""
    logger.info("Testing Temporal Causality Analysis...")
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data with causal relationships
        data = generate_synthetic_geological_data()
        
        # Create causal relationship: seismic -> deformation
        time_series = {
            'seismic_activity': data['seismic_activity'],
            'deformation_measurements': data['deformation_measurements'],
            'thermal_activity': data['thermal_activity']
        }
        
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test causality analysis
        result = integrator._analyze_temporal_causality(time_series)
        
        # Validate results
        assert isinstance(result, dict), "Causality result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"Causality analysis failed: {result.get('message', 'Unknown error')}"
        
        assert 'causality_analysis' in result, "Result should include causality analysis"
        
        causality_analysis = result['causality_analysis']
        assert 'granger_causality' in causality_analysis, "Should include Granger causality"
        assert 'cross_correlation' in causality_analysis, "Should include cross-correlation"
        
        # Check for detected causal relationships
        if 'granger_causality' in causality_analysis:
            granger_results = causality_analysis['granger_causality']
            significant_relationships = [rel for rel, res in granger_results.items() 
                                       if isinstance(res, dict) and res.get('overall_significant', False)]
            logger.info(f"Detected significant causal relationships: {len(significant_relationships)}")
        
        logger.info("✓ Temporal Causality Analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Temporal Causality Analysis test failed: {e}")
        return False

def test_temporal_correlations():
    """Test the temporal correlations analysis implementation"""
    logger.info("Testing Temporal Correlations Analysis...")
    
    try:
        from enkf_knowledge_integration import AdvancedEnKFKnowledgeIntegrator
        
        # Generate test data with correlations
        data = generate_synthetic_geological_data()
        
        time_series = {
            'seismic_activity': data['seismic_activity'],
            'thermal_activity': data['thermal_activity'],
            'pressure_variations': data['pressure_variations']
        }
        
        integrator = AdvancedEnKFKnowledgeIntegrator({})
        
        # Test correlation analysis
        result = integrator._analyze_temporal_correlations(time_series)
        
        # Validate results
        assert isinstance(result, dict), "Correlation result should be a dictionary"
        assert 'status' not in result or result['status'] != 'error', f"Correlation analysis failed: {result.get('message', 'Unknown error')}"
        
        assert 'correlation_analysis' in result, "Result should include correlation analysis"
        
        correlation_analysis = result['correlation_analysis']
        assert 'pearson_correlation' in correlation_analysis, "Should include Pearson correlation"
        assert 'spearman_correlation' in correlation_analysis, "Should include Spearman correlation"
        assert 'cross_correlation_detailed' in correlation_analysis, "Should include detailed cross-correlation"
        
        # Check correlation matrix
        if 'pearson_correlation' in correlation_analysis and 'matrix' in correlation_analysis['pearson_correlation']:
            corr_matrix = correlation_analysis['pearson_correlation']['matrix']
            logger.info(f"Correlation matrix shape: {np.array(corr_matrix).shape}")
        
        logger.info("✓ Temporal Correlations Analysis test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Temporal Correlations Analysis test failed: {e}")
        return False

def run_all_tests():
    """Run all temporal analysis tests"""
    logger.info("Starting Advanced Temporal Analysis Tests")
    logger.info("=" * 50)
    
    tests = [
        ("HMM Model", test_hmm_model),
        ("Temporal Trends", test_temporal_trends),
        ("Seasonal Patterns", test_seasonal_patterns),
        ("Cyclical Patterns", test_cyclical_patterns),
        ("Temporal Causality", test_temporal_causality),
        ("Temporal Correlations", test_temporal_correlations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Advanced temporal analysis implementations are working correctly.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Please review the implementations.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)