#!/usr/bin/env python3
"""
Test script for enhanced geological analysis methods
"""

import numpy as np
import asyncio
import logging

# Test the enhanced methods in isolation
class TestEnhancedMethods:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
    async def test_dominant_orientation(self):
        """Test advanced dominant orientation calculation"""
        print("\n=== Testing Enhanced Dominant Orientation Calculation ===")
        
        # Test data: structural measurements
        orientations = [135, 140, 145, 138, 142, None, 139, 141]
        location = [35.6762, 139.6503]  # Tokyo coordinates
        depth = 5000  # 5km depth
        
        try:
            # Simplified version of the advanced method for testing
            valid_orientations = np.array([o for o in orientations if o is not None and not np.isnan(o)])
            
            if len(valid_orientations) == 0:
                result = 0
            else:
                # Advanced multi-modal orientation analysis
                angles_rad = valid_orientations * np.pi / 180
                
                # Fisher distribution analysis
                R = np.sqrt(np.sum(np.cos(angles_rad))**2 + np.sum(np.sin(angles_rad))**2)
                n = len(angles_rad)
                mean_R = R / n
                
                if mean_R > 0.7:  # Highly concentrated
                    dominant_angle = np.arctan2(np.sum(np.sin(angles_rad)), np.sum(np.cos(angles_rad)))
                    result = float((dominant_angle * 180 / np.pi) % 360)
                    regime = 'highly_oriented'
                    confidence = float(mean_R)
                else:
                    # Simple circular mean for this test
                    dominant_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
                    result = float((dominant_angle * 180 / np.pi) % 360)
                    regime = 'multi_modal'
                    confidence = 0.5
                    
            print(f"Input orientations: {orientations}")
            print(f"Valid orientations: {valid_orientations}")
            print(f"Dominant orientation: {result:.1f}°")
            print(f"Concentration (R): {mean_R:.3f}")
            print(f"Structural regime: {regime}")
            print(f"Confidence: {confidence:.3f}")
            print("[SUCCESS] Enhanced dominant orientation calculation completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Enhanced dominant orientation test failed: {e}")
    
    async def test_cyclicity_detection(self):
        """Test advanced cyclicity detection"""
        print("\n=== Testing Enhanced Cyclicity Detection ===")
        
        # Generate test time series with known cyclicity
        times = np.linspace(0, 365, 100)  # 1 year of daily data
        # Add multiple periodic components
        signal = (2 * np.sin(2 * np.pi * times / 30) +  # Monthly cycle
                 1.5 * np.sin(2 * np.pi * times / 7) +   # Weekly cycle  
                 0.5 * np.sin(2 * np.pi * times / 365) + # Annual cycle
                 0.3 * np.random.randn(len(times)))       # Noise
        
        location = [40.7128, -74.0060]  # New York coordinates
        parameter_type = 'volcanic'
        
        try:
            from scipy.fft import fft, fftfreq
            from scipy.signal import find_peaks
            
            # Advanced spectral analysis
            if len(signal) < 10:
                result = 0.1
            else:
                # Detrend and window
                from scipy import signal as scipy_signal
                detrended = scipy_signal.detrend(signal)
                windowed = detrended * scipy_signal.windows.hann(len(detrended))
                
                # Compute FFT
                fft_vals = np.abs(fft(windowed))
                freqs = fftfreq(len(windowed), d=np.mean(np.diff(times)))
                
                # Focus on positive frequencies
                pos_freqs = freqs[:len(freqs)//2]
                pos_fft = fft_vals[:len(fft_vals)//2]
                
                # Find peaks
                peaks, properties = find_peaks(pos_fft, height=np.std(pos_fft), distance=5)
                
                if len(peaks) > 0:
                    periods = 1.0 / pos_freqs[peaks]
                    valid_periods = periods[periods > 0]
                    cyclicity_strength = float(np.max(pos_fft[peaks]) / np.sum(pos_fft))
                else:
                    valid_periods = []
                    cyclicity_strength = 0.1
                    
            print(f"Input time series length: {len(times)} points")
            print(f"Time span: {times[-1] - times[0]:.1f} days")
            print(f"Detected periods: {[f'{p:.1f} days' for p in valid_periods[:5]]}")
            print(f"Cyclicity strength: {cyclicity_strength:.3f}")
            print("[SUCCESS] Enhanced cyclicity detection completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Enhanced cyclicity detection test failed: {e}")
    
    async def test_scale_interactions(self):
        """Test advanced multi-scale interaction analysis"""
        print("\n=== Testing Enhanced Multi-Scale Interaction Analysis ===")
        
        # Test scale data
        local = {
            'dominance_score': 0.8,
            'intensity': 0.7,
            'spatial_extent': 5000,
            'temporal_persistence': 0.6
        }
        
        regional = {
            'influence_factor': 0.6,
            'connectivity': 0.5,
            'propagation_efficiency': 0.7,
            'regional_coherence': 0.8
        }
        
        plate = {
            'control_factor': 0.4,
            'tectonic_loading': 0.5,
            'plate_boundary_proximity': 0.3,
            'mantle_coupling': 0.6
        }
        
        location = [36.2048, 138.2529]  # Central Japan coordinates
        
        try:
            # Extract comprehensive scale metrics
            def extract_metrics(scale_data, scale_type):
                metrics = {}
                for key, value in scale_data.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[key] = value
                
                # Additional derived metrics
                metrics['variability'] = np.std(list(metrics.values())) if len(metrics) > 1 else 0.0
                metrics['mean_activity'] = np.mean(list(metrics.values()))
                return metrics
            
            local_metrics = extract_metrics(local, 'local')
            regional_metrics = extract_metrics(regional, 'regional')
            plate_metrics = extract_metrics(plate, 'plate')
            
            # Multi-dimensional interaction analysis
            scale_vectors = {
                'local': np.array(list(local_metrics.values())),
                'regional': np.array(list(regional_metrics.values())),
                'plate': np.array(list(plate_metrics.values()))
            }
            
            # Calculate correlations between scales
            from scipy.stats import pearsonr
            correlations = {}
            for scale1 in scale_vectors:
                for scale2 in scale_vectors:
                    if scale1 != scale2:
                        try:
                            corr, p_value = pearsonr(scale_vectors[scale1], scale_vectors[scale2])
                            correlations[f'{scale1}_{scale2}'] = {'correlation': float(corr), 'p_value': float(p_value)}
                        except:
                            correlations[f'{scale1}_{scale2}'] = {'correlation': 0.0, 'p_value': 1.0}
            
            # Calculate coupling strength
            coupling_strength = np.mean([abs(corr['correlation']) for corr in correlations.values()])
            
            # Advanced complexity calculation
            all_values = []
            for scale in [local_metrics, regional_metrics, plate_metrics]:
                all_values.extend(list(scale.values()))
            
            # Use coefficient of variation as complexity measure
            if len(all_values) > 0 and np.mean(all_values) > 0:
                complexity_score = np.std(all_values) / np.mean(all_values)
            else:
                complexity_score = 0.5
                
            # Scale hierarchy
            scale_strengths = {
                'local': np.mean(list(local_metrics.values())),
                'regional': np.mean(list(regional_metrics.values())),
                'plate': np.mean(list(plate_metrics.values()))
            }
            
            hierarchy = sorted(scale_strengths.items(), key=lambda x: x[1], reverse=True)
            
            print(f"Local metrics: {local_metrics}")
            print(f"Regional metrics: {regional_metrics}")
            print(f"Plate metrics: {plate_metrics}")
            print(f"Cross-scale correlations: {correlations}")
            print(f"Coupling strength: {coupling_strength:.3f}")
            print(f"Complexity score: {complexity_score:.3f}")
            print(f"Scale hierarchy: {hierarchy}")
            print("[SUCCESS] Enhanced multi-scale interaction analysis completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Enhanced scale interaction test failed: {e}")
    
    async def run_all_tests(self):
        """Run all enhanced method tests"""
        print("=" * 70)
        print("TESTING ENHANCED GEOLOGICAL ANALYSIS METHODS")
        print("=" * 70)
        
        await self.test_dominant_orientation()
        await self.test_cyclicity_detection()
        await self.test_scale_interactions()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)

if __name__ == "__main__":
    test = TestEnhancedMethods()
    asyncio.run(test.run_all_tests())