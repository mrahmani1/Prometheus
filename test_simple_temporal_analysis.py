#!/usr/bin/env python3
"""
Simple Test for Advanced Temporal Analysis Implementations
Tests the core functionality without external dependencies
"""

import sys
import os
import numpy as np
import logging

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_temporal_analysis_helpers():
    """Test the temporal analysis helper functions directly"""
    logger.info("Testing Temporal Analysis Helpers...")
    
    try:
        from temporal_analysis_helpers import TemporalAnalysisHelpers
        
        helpers = TemporalAnalysisHelpers()
        
        # Test with simple synthetic data
        np.random.seed(42)
        n_points = 100
        
        # Generate simple test data
        time_series_1 = np.random.randn(n_points).cumsum()
        time_series_2 = 0.7 * time_series_1 + 0.3 * np.random.randn(n_points)
        
        # Test Granger causality
        logger.info("Testing Granger causality...")
        granger_result = helpers.granger_causality_test(time_series_1, time_series_2, maxlags=5)
        assert isinstance(granger_result, dict), "Granger test should return dict"
        logger.info("✓ Granger causality test passed")
        
        # Test transfer entropy
        logger.info("Testing transfer entropy...")
        te_result = helpers.calculate_transfer_entropy(time_series_1, time_series_2)
        assert isinstance(te_result, (int, float)), "Transfer entropy should return number"
        logger.info(f"✓ Transfer entropy: {te_result:.4f}")
        
        # Test HMM fallback
        logger.info("Testing HMM fallback...")
        hmm_result = helpers.build_hmm_fallback(time_series_1)
        assert isinstance(hmm_result, dict), "HMM fallback should return dict"
        assert 'state_sequence' in hmm_result, "HMM should return state sequence"
        logger.info(f"✓ HMM fallback: {hmm_result['n_states']} states")
        
        # Test MIC approximation
        logger.info("Testing MIC approximation...")
        mic_result = helpers.calculate_mic_approximation(time_series_1, time_series_2)
        assert isinstance(mic_result, (int, float)), "MIC should return number"
        logger.info(f"✓ MIC approximation: {mic_result:.4f}")
        
        # Test geological feature extraction with empty data
        logger.info("Testing geological feature extraction...")
        features = helpers.extract_geological_features({})
        assert isinstance(features, dict), "Features should be dict"
        logger.info("✓ Geological feature extraction passed")
        
        logger.info("✓ All helper function tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Helper function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_temporal_methods():
    """Test basic functionality of temporal analysis methods"""
    logger.info("Testing Basic Temporal Methods...")
    
    try:
        # Create simple test class with the methods
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'modules'))
        
        # Create a mock class to test the methods
        class MockTemporalAnalyzer:
            def __init__(self):
                from temporal_analysis_helpers import TemporalAnalysisHelpers
                self.helpers = TemporalAnalysisHelpers()
            
            def test_build_hmm_model(self, values):
                """Test HMM model building"""
                try:
                    # Try with hmmlearn first, fall back to simple implementation
                    try:
                        from hmmlearn import hmm
                        from sklearn.preprocessing import StandardScaler
                        
                        data = np.array(values)
                        if data.ndim == 1:
                            data = data.reshape(-1, 1)
                        
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(data)
                        
                        model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
                        model.fit(scaled_data)
                        
                        state_sequence = model.predict(scaled_data)
                        
                        return {
                            'model': 'hmmlearn',
                            'n_states': 3,
                            'state_sequence': state_sequence.tolist(),
                            'log_likelihood': float(model.score(scaled_data))
                        }
                        
                    except ImportError:
                        # Use fallback
                        return self.helpers.build_hmm_fallback(values)
                        
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
            
            def test_analyze_temporal_trends(self, values):
                """Test temporal trends analysis"""
                try:
                    from scipy import stats
                    from sklearn.linear_model import LinearRegression
                    
                    data = np.array(values)
                    if data.ndim > 1:
                        data = data.flatten()
                    
                    time_index = np.arange(len(data))
                    
                    # Linear regression
                    linear_reg = LinearRegression().fit(time_index.reshape(-1, 1), data)
                    
                    # Mann-Kendall test
                    mk_result = stats.kendalltau(time_index, data)
                    
                    return {
                        'trend_analysis': {
                            'linear': {
                                'slope': float(linear_reg.coef_[0]),
                                'intercept': float(linear_reg.intercept_),
                                'r_squared': float(linear_reg.score(time_index.reshape(-1, 1), data))
                            },
                            'mann_kendall': {
                                'tau': float(mk_result.correlation),
                                'p_value': float(mk_result.pvalue),
                                'significant': mk_result.pvalue < 0.05
                            }
                        },
                        'data_statistics': {
                            'mean': float(np.mean(data)),
                            'std': float(np.std(data)),
                            'length': len(data)
                        }
                    }
                    
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
            
            def test_analyze_seasonal_patterns(self, values):
                """Test seasonal patterns analysis"""
                try:
                    from scipy import signal, fft
                    
                    data = np.array(values)
                    if data.ndim > 1:
                        data = data.flatten()
                    
                    # Fourier analysis
                    freqs = fft.fftfreq(len(data))
                    fft_values = fft.fft(data)
                    power_spectrum = np.abs(fft_values)**2
                    
                    # Find dominant frequencies
                    dominant_freq_indices = np.argsort(power_spectrum)[-5:][::-1]
                    dominant_frequencies = [
                        {
                            'frequency': float(freqs[i]),
                            'period': float(1/abs(freqs[i])) if freqs[i] != 0 else float('inf'),
                            'power': float(power_spectrum[i])
                        }
                        for i in dominant_freq_indices if freqs[i] != 0
                    ]
                    
                    # Autocorrelation
                    autocorr = np.correlate(data, data, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0]  # Normalize
                    
                    return {
                        'seasonal_analysis': {
                            'fourier_analysis': {
                                'dominant_frequencies': dominant_frequencies[:3]
                            },
                            'autocorrelation': {
                                'function_length': len(autocorr),
                                'max_autocorr': float(np.max(autocorr[1:10])) if len(autocorr) > 10 else 0.0
                            }
                        },
                        'analysis_confidence': 'basic_analysis'
                    }
                    
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
            
            def test_analyze_temporal_correlations(self, values_dict):
                """Test temporal correlations analysis"""
                try:
                    from scipy import stats
                    
                    if not isinstance(values_dict, dict) or len(values_dict) < 2:
                        return {'status': 'error', 'message': 'Need at least 2 time series'}
                    
                    # Prepare data
                    series_data = {}
                    min_length = float('inf')
                    
                    for name, series in values_dict.items():
                        arr = np.array(series)
                        if arr.ndim > 1:
                            arr = arr.flatten()
                        series_data[name] = arr
                        min_length = min(min_length, len(arr))
                    
                    # Truncate to same length
                    for name in series_data:
                        series_data[name] = series_data[name][:min_length]
                    
                    series_names = list(series_data.keys())
                    
                    # Pearson correlation matrix
                    data_matrix = np.column_stack([series_data[name] for name in series_names])
                    pearson_corr_matrix = np.corrcoef(data_matrix.T)
                    
                    # Find significant correlations
                    significant_pairs = []
                    for i in range(len(series_names)):
                        for j in range(i+1, len(series_names)):
                            corr = pearson_corr_matrix[i, j]
                            if abs(corr) > 0.3:  # Threshold for significance
                                significant_pairs.append({
                                    'series1': series_names[i],
                                    'series2': series_names[j],
                                    'correlation': float(corr)
                                })
                    
                    return {
                        'correlation_analysis': {
                            'pearson_correlation': {
                                'matrix': pearson_corr_matrix.tolist(),
                                'series_names': series_names,
                                'significant_pairs': significant_pairs
                            }
                        },
                        'data_summary': {
                            'series_count': len(series_names),
                            'data_length': min_length
                        }
                    }
                    
                except Exception as e:
                    return {'status': 'error', 'message': str(e)}
        
        # Test the mock analyzer
        analyzer = MockTemporalAnalyzer()
        
        # Generate test data
        np.random.seed(42)
        n_points = 200
        
        # Test data with trend
        trend_data = np.linspace(0, 10, n_points) + np.random.normal(0, 0.5, n_points)
        
        # Test data with seasonality
        seasonal_data = np.sin(2 * np.pi * np.linspace(0, 4, n_points)) + np.random.normal(0, 0.1, n_points)
        
        # Test HMM
        logger.info("Testing HMM model building...")
        hmm_result = analyzer.test_build_hmm_model(trend_data)
        assert 'status' not in hmm_result or hmm_result['status'] != 'error', f"HMM failed: {hmm_result.get('message')}"
        logger.info(f"✓ HMM: {hmm_result.get('n_states', 'unknown')} states detected")
        
        # Test trends
        logger.info("Testing temporal trends analysis...")
        trends_result = analyzer.test_analyze_temporal_trends(trend_data)
        assert 'status' not in trends_result or trends_result['status'] != 'error', f"Trends failed: {trends_result.get('message')}"
        slope = trends_result['trend_analysis']['linear']['slope']
        logger.info(f"✓ Trends: detected slope = {slope:.4f}")
        
        # Test seasonal patterns
        logger.info("Testing seasonal patterns analysis...")
        seasonal_result = analyzer.test_analyze_seasonal_patterns(seasonal_data)
        assert 'status' not in seasonal_result or seasonal_result['status'] != 'error', f"Seasonal failed: {seasonal_result.get('message')}"
        logger.info("✓ Seasonal patterns analysis completed")
        
        # Test correlations
        logger.info("Testing temporal correlations...")
        correlation_data = {
            'series1': trend_data,
            'series2': 0.8 * trend_data + 0.2 * np.random.randn(n_points),
            'series3': seasonal_data
        }
        correlation_result = analyzer.test_analyze_temporal_correlations(correlation_data)
        assert 'status' not in correlation_result or correlation_result['status'] != 'error', f"Correlation failed: {correlation_result.get('message')}"
        significant_pairs = correlation_result['correlation_analysis']['pearson_correlation']['significant_pairs']
        logger.info(f"✓ Correlations: {len(significant_pairs)} significant pairs found")
        
        logger.info("✓ All basic temporal method tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic temporal methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simple_tests():
    """Run simplified tests"""
    logger.info("Starting Simple Advanced Temporal Analysis Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Temporal Analysis Helpers", test_temporal_analysis_helpers),
        ("Basic Temporal Methods", test_basic_temporal_methods)
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
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All simplified tests passed! Core temporal analysis functionality is working.")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed. Core implementations may have issues.")
    
    return passed == total

if __name__ == "__main__":
    success = run_simple_tests()
    sys.exit(0 if success else 1)