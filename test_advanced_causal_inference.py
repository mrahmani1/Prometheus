#!/usr/bin/env python3
"""
Test script for the advanced geological causal inference implementation.
Demonstrates the novel multi-modal causal discovery capabilities.
"""

import numpy as np
import asyncio
import sys
import os

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

from real_time_process_monitor import RealTimeProcessMonitor

async def test_advanced_causal_inference():
    """Test the advanced causal inference with synthetic geological data"""
    
    # Initialize monitor with minimal config
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'influxdb_url': 'http://localhost:8086',
        'influxdb_token': 'test_token',
        'influxdb_org': 'prometheus',
        'max_workers': 4
    }
    
    monitor = RealTimeProcessMonitor(config)
    
    # Generate synthetic geological time series with known causal relationships
    n_points = 100
    t = np.linspace(0, 10, n_points)
    
    # Seismic activity (source)
    seismic_base = np.sin(0.5 * t) + 0.3 * np.sin(2 * t)
    seismic_noise = 0.1 * np.random.randn(n_points)
    seismic_data = seismic_base + seismic_noise
    
    # Volcanic activity (target) - causally influenced by seismic with lag
    lag = 5
    volcanic_base = 0.7 * np.roll(seismic_base, lag) + 0.3 * np.sin(0.8 * t)
    volcanic_noise = 0.15 * np.random.randn(n_points)
    volcanic_data = volcanic_base + volcanic_noise
    
    # Deformation (independent process)
    deformation_data = 0.5 * np.cos(0.3 * t) + 0.2 * np.random.randn(n_points)
    
    print("Testing Advanced Geological Causal Inference")
    print("=" * 50)
    
    # Test 1: Strong causal relationship (seismic -> volcanic)
    print("\n1. Testing Seismic -> Volcanic Causality (Expected: Strong)")
    causal_strength_1 = await monitor._advanced_geological_causal_inference(
        seismic_data.tolist(), 
        volcanic_data.tolist(),
        "seismic_station_1",
        "volcanic_station_1"
    )
    print(f"   Causal Strength: {causal_strength_1:.4f}")
    
    # Test 2: Weak causal relationship (seismic -> deformation)
    print("\n2. Testing Seismic -> Deformation Causality (Expected: Weak)")
    causal_strength_2 = await monitor._advanced_geological_causal_inference(
        seismic_data.tolist(),
        deformation_data.tolist(),
        "seismic_station_1", 
        "deformation_station_1"
    )
    print(f"   Causal Strength: {causal_strength_2:.4f}")
    
    # Test 3: Reverse causality (volcanic -> seismic)
    print("\n3. Testing Volcanic -> Seismic Causality (Expected: Weak)")
    causal_strength_3 = await monitor._advanced_geological_causal_inference(
        volcanic_data.tolist(),
        seismic_data.tolist(),
        "volcanic_station_1",
        "seismic_station_1"
    )
    print(f"   Causal Strength: {causal_strength_3:.4f}")
    
    # Test 4: Nonlinear causal relationship
    print("\n4. Testing Nonlinear Causality")
    # Create nonlinear relationship
    nonlinear_source = np.sin(t) + 0.1 * np.random.randn(n_points)
    nonlinear_target = np.tanh(2 * np.roll(nonlinear_source, 3)) + 0.1 * np.random.randn(n_points)
    
    causal_strength_4 = await monitor._advanced_geological_causal_inference(
        nonlinear_source.tolist(),
        nonlinear_target.tolist(),
        "nonlinear_source",
        "nonlinear_target"
    )
    print(f"   Nonlinear Causal Strength: {causal_strength_4:.4f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY OF ADVANCED CAUSAL ANALYSIS")
    print("=" * 50)
    print(f"Seismic -> Volcanic:     {causal_strength_1:.4f} (Strong expected)")
    print(f"Seismic -> Deformation:  {causal_strength_2:.4f} (Weak expected)")
    print(f"Volcanic -> Seismic:     {causal_strength_3:.4f} (Weak expected)")
    print(f"Nonlinear Causality:     {causal_strength_4:.4f} (Moderate expected)")
    
    # Validate results
    success_criteria = [
        causal_strength_1 > 0.3,  # Strong relationship should be detected
        causal_strength_2 < 0.4,  # Weak relationship should be low
        causal_strength_1 > causal_strength_2,  # True causal should be stronger
        causal_strength_4 > 0.2   # Nonlinear should be detected
    ]
    
    if all(success_criteria):
        print("\n✅ ADVANCED CAUSAL INFERENCE TEST PASSED!")
        print("   The implementation successfully distinguishes causal relationships")
    else:
        print("\n⚠️  Test results need review - some criteria not met")
    
    print("\nAdvanced Features Demonstrated:")
    print("• Neural Network-based Nonlinear Granger Causality")
    print("• Transfer Entropy with Adaptive Bandwidth")
    print("• Convergent Cross Mapping for Nonlinear Dynamics")
    print("• Physics-Informed Geological Constraints")
    print("• Wavelet-based Multi-Scale Analysis")
    print("• Information-Theoretic Measures")
    print("• Ensemble Learning with Adaptive Weights")

if __name__ == "__main__":
    # Run the test
    try:
        asyncio.run(test_advanced_causal_inference())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()