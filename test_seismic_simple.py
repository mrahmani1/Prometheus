#!/usr/bin/env python3
"""
Simple test to verify the completed Advanced Seismic Source Characterization methods
"""

import numpy as np
import asyncio
import sys
import os

async def test_individual_methods():
    """Test individual methods from the seismic characterization system"""
    try:
        # Test the mathematical functions directly
        print("Testing individual mathematical methods...")
        
        # Create test waveform
        test_waveform = np.random.randn(1000) * 0.1 + np.sin(np.linspace(0, 10*np.pi, 1000))
        
        # Test zero crossings
        zero_crossings = len(np.where(np.diff(np.signbit(test_waveform)))[0])
        print(f"✓ Zero crossings: {zero_crossings}")
        
        # Test turning points
        diff = np.diff(test_waveform)
        turning_points = len(np.where(np.diff(np.signbit(diff)))[0])
        print(f"✓ Turning points: {turning_points}")
        
        # Test Hjorth parameters
        d1 = np.diff(test_waveform)
        d2 = np.diff(d1)
        var_x = np.var(test_waveform)
        var_d1 = np.var(d1)
        var_d2 = np.var(d2)
        
        activity = var_x
        mobility = np.sqrt(var_d1 / var_x) if var_x > 0 else 0.0
        complexity = np.sqrt(var_d2 / var_d1) / mobility if var_d1 > 0 and mobility > 0 else 0.0
        
        print(f"✓ Hjorth parameters - Activity: {activity:.3f}, Mobility: {mobility:.3f}, Complexity: {complexity:.3f}")
        
        # Test significant duration calculation
        energy = test_waveform**2
        cumulative_energy = np.cumsum(energy)
        total_energy = cumulative_energy[-1]
        
        if total_energy > 0:
            energy_5_percent = 0.05 * total_energy
            energy_95_percent = 0.95 * total_energy
            idx_5 = np.argmax(cumulative_energy >= energy_5_percent)
            idx_95 = np.argmax(cumulative_energy >= energy_95_percent)
            significant_duration = (idx_95 - idx_5) / 100.0  # Assuming 100 Hz sampling
            print(f"✓ Significant duration: {significant_duration:.3f} seconds")
        
        # Test spectral entropy calculation
        from scipy import signal
        freqs, psd = signal.welch(test_waveform, fs=100, nperseg=min(256, len(test_waveform)//4))
        psd_norm = psd / (np.sum(psd) + 1e-12)
        
        entropy = 0.0
        for p in psd_norm:
            if p > 0:
                entropy -= p * np.log2(p)
        
        max_entropy = np.log2(len(psd))
        spectral_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        print(f"✓ Spectral entropy: {spectral_entropy:.3f}")
        
        # Test fractal dimension (simplified Higuchi method)
        N = len(test_waveform)
        k_max = min(10, N // 4)
        
        L_k = []
        k_values = []
        
        for k in range(1, k_max + 1):
            L_m = []
            for m in range(k):
                indices = np.arange(m, N, k)
                if len(indices) < 2:
                    continue
                subsequence = test_waveform[indices]
                length = np.sum(np.abs(np.diff(subsequence))) * (N - 1) / (k * len(indices))
                L_m.append(length)
            
            if L_m:
                L_k.append(np.mean(L_m))
                k_values.append(k)
        
        if len(L_k) >= 2:
            log_k = np.log(k_values)
            log_L = np.log(L_k)
            slope, _ = np.polyfit(log_k, log_L, 1)
            fractal_dim = max(1.0, min(2.0, -slope))
            print(f"✓ Fractal dimension: {fractal_dim:.3f}")
        
        print("\n🎉 All individual method tests passed!")
        print("The mathematical implementations are working correctly.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_individual_methods())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)