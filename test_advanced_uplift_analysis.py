#!/usr/bin/env python3
"""
Test script for advanced uplift-subsidence cycle analysis
Validates the enhanced statistical methods and real data integration
"""

import asyncio
import numpy as np
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.modules.resurgent_caldera.uplift_subsidence_cycle_analysis import UpliftSubsidenceCycleAnalyzer

async def test_advanced_trend_analysis():
    """Test the advanced trend significance assessment"""
    print("Testing Advanced Trend Significance Assessment...")
    
    # Create analyzer instance with mock database URL
    analyzer = UpliftSubsidenceCycleAnalyzer("sqlite:///:memory:")
    
    # Generate synthetic test data with known characteristics
    n_points = 200
    time = np.linspace(0, 10, n_points)  # 10 years
    
    # Create realistic deformation signal
    trend = 0.5 * time  # 0.5 mm/year uplift
    seasonal = 3 * np.sin(2 * np.pi * time)  # 3mm seasonal amplitude
    noise = np.random.normal(0, 2, n_points)  # 2mm noise
    
    deformation = trend + seasonal + noise
    
    # Prepare test data
    trend_analysis = {
        'mean_trend_slope': 0.5,
        'detrended': deformation - trend,
        'trend_slope': np.full(n_points, 0.5),
        'trend': trend
    }
    
    processed_data = {
        'time_series': deformation.tolist(),
        'location': {'latitude': 44.0, 'longitude': -110.5}  # Yellowstone
    }
    
    # Test trend significance assessment
    try:
        significance_results = await analyzer._assess_trend_significance(trend_analysis, processed_data)
        
        print("✓ Trend Significance Results:")
        for key, value in significance_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Verify key results
        assert 'trend_slope' in significance_results
        assert 'mann_kendall_p' in significance_results
        assert 'is_significant' in significance_results
        assert isinstance(significance_results['is_significant'], bool)
        
        print("✓ Trend significance assessment passed!")
        
    except Exception as e:
        print(f"✗ Trend significance assessment failed: {e}")
        return False
    
    return True

async def test_advanced_uncertainty_calculation():
    """Test the advanced uncertainty quantification"""
    print("\nTesting Advanced Uncertainty Quantification...")
    
    analyzer = UpliftSubsidenceCycleAnalyzer("sqlite:///:memory:")
    
    # Generate test data
    n_points = 150
    trend = np.linspace(0, 5, n_points)
    detrended = np.random.normal(0, 1, n_points)
    
    trend_analysis = {
        'trend': trend,
        'detrended': detrended,
        'mean_trend_slope': 5.0 / n_points
    }
    
    processed_data = {
        'location': {'latitude': 37.7, 'longitude': -119.0}  # Long Valley
    }
    
    try:
        uncertainty_results = await analyzer._calculate_trend_uncertainties(trend_analysis, processed_data)
        
        print("✓ Uncertainty Quantification Results:")
        for key, value in uncertainty_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Verify key results
        assert 'trend_uncertainty' in uncertainty_results
        assert 'slope_uncertainty' in uncertainty_results
        assert 'cycle_detection_uncertainty' in uncertainty_results
        assert all(isinstance(v, (int, float)) for v in uncertainty_results.values())
        
        print("✓ Uncertainty quantification passed!")
        
    except Exception as e:
        print(f"✗ Uncertainty quantification failed: {e}")
        return False
    
    return True

async def test_real_data_integration():
    """Test real data fetching and integration"""
    print("\nTesting Real Data Integration...")
    
    analyzer = UpliftSubsidenceCycleAnalyzer("sqlite:///:memory:")
    
    processed_data = {
        'location': {'latitude': 44.0, 'longitude': -110.5}  # Yellowstone
    }
    
    try:
        real_data = await analyzer._fetch_real_deformation_data(processed_data)
        
        print("✓ Real Data Integration Results:")
        print(f"  Data sources found: {len(real_data)}")
        for key, value in real_data.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} data points")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Should return at least mock data
        assert isinstance(real_data, dict)
        assert len(real_data) > 0
        
        print("✓ Real data integration passed!")
        
    except Exception as e:
        print(f"✗ Real data integration failed: {e}")
        return False
    
    return True

async def main():
    """Run all tests"""
    print("=" * 60)
    print("Advanced Uplift-Subsidence Cycle Analysis Tests")
    print("=" * 60)
    
    tests = [
        test_advanced_trend_analysis,
        test_advanced_uncertainty_calculation,
        test_real_data_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)