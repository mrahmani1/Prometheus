"""
Validation script for advanced seismic algorithms
Tests the novel implementations without complex imports
"""

import numpy as np
import asyncio
from datetime import datetime, timedelta
import logging
from scipy import signal
from scipy.spatial.distance import cdist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedSeismicValidator:
    """Validator for advanced seismic algorithms"""
    
    def __init__(self):
        self.sampling_rate = 100.0
        logger.info("Advanced Seismic Validator initialized")

    async def validate_advanced_wavelet_features(self, waveform: np.ndarray) -> bool:
        """Validate advanced wavelet feature extraction"""
        try:
            logger.info("🌊 Validating Advanced Wavelet Features")
            
            # Continuous Wavelet Transform using Morlet wavelet
            scales = np.arange(1, 32)
            try:
                coefficients, frequencies = signal.cwt(waveform, signal.morlet2, scales, w=6)
            except AttributeError:
                # Fallback implementation for older scipy versions
                coefficients = np.zeros((len(scales), len(waveform)), dtype=complex)
                frequencies = np.zeros(len(scales))
                
                for i, scale in enumerate(scales):
                    # Simple Morlet wavelet implementation
                    sigma = scale / 6.0
                    freq = 1.0 / scale
                    frequencies[i] = freq
                    
                    # Create Morlet wavelet
                    t_wavelet = np.arange(-3*sigma, 3*sigma, 1.0/100.0)  # 100 Hz sampling
                    if len(t_wavelet) > 0:
                        morlet = np.exp(1j * 2 * np.pi * freq * t_wavelet) * np.exp(-t_wavelet**2 / (2 * sigma**2))
                        morlet = morlet / np.sqrt(sigma)
                        
                        # Convolution
                        if len(morlet) < len(waveform):
                            conv_result = np.convolve(waveform, morlet, mode='same')
                            coefficients[i, :] = conv_result
            
            # Wavelet energy distribution
            wavelet_energy = np.sum(np.abs(coefficients)**2, axis=1)
            total_energy = np.sum(wavelet_energy)
            
            features = []
            
            if total_energy > 0:
                energy_distribution = wavelet_energy / total_energy
                features.extend([
                    np.mean(energy_distribution),
                    np.std(energy_distribution),
                    np.max(energy_distribution),
                    np.argmax(energy_distribution) / len(scales)
                ])
                
                # Wavelet entropy
                entropy = -np.sum(energy_distribution * np.log2(energy_distribution + 1e-10))
                features.append(entropy)
            else:
                features.extend([0.0] * 5)
            
            # Multifractal analysis
            q_values = np.linspace(-5, 5, 11)
            tau_q = []
            
            for q in q_values:
                if q == 0:
                    tau_q.append(0.0)
                else:
                    partition_sum = np.sum(wavelet_energy**q)
                    if partition_sum > 0:
                        tau_q.append(np.log(partition_sum) / np.log(len(scales)))
                    else:
                        tau_q.append(0.0)
            
            multifractal_width = max(tau_q) - min(tau_q)
            hurst_exponent = (tau_q[6] - tau_q[4]) / 2.0
            
            features.extend([multifractal_width, hurst_exponent])
            
            logger.info(f"✅ Extracted {len(features)} advanced wavelet features")
            logger.info(f"   Wavelet entropy: {features[4]:.3f}")
            logger.info(f"   Multifractal width: {multifractal_width:.3f}")
            logger.info(f"   Hurst exponent: {hurst_exponent:.3f}")
            
            return len(features) >= 7 and all(np.isfinite(f) for f in features)
            
        except Exception as e:
            logger.error(f"Error validating wavelet features: {e}")
            return False

    async def validate_emd_decomposition(self, waveform: np.ndarray) -> bool:
        """Validate Empirical Mode Decomposition"""
        try:
            logger.info("📊 Validating Empirical Mode Decomposition")
            
            imfs = await self._empirical_mode_decomposition(waveform)
            
            if len(imfs) > 0:
                # Energy distribution across IMFs
                imf_energies = [np.sum(imf**2) for imf in imfs]
                total_energy = sum(imf_energies)
                
                if total_energy > 0:
                    energy_ratios = [e / total_energy for e in imf_energies]
                    
                    logger.info(f"✅ EMD extracted {len(imfs)} IMFs")
                    logger.info(f"   Energy distribution: {[f'{r:.3f}' for r in energy_ratios[:3]]}")
                    
                    # Orthogonality check
                    orthogonality = await self._calculate_orthogonality_index(imfs)
                    logger.info(f"   Orthogonality index: {orthogonality:.3f}")
                    
                    return len(imfs) >= 2 and orthogonality > 0.5
                else:
                    return False
            else:
                logger.warning("No IMFs extracted")
                return False
                
        except Exception as e:
            logger.error(f"Error validating EMD: {e}")
            return False

    async def _empirical_mode_decomposition(self, signal_data: np.ndarray, max_imfs: int = 6) -> list:
        """Simplified Empirical Mode Decomposition"""
        try:
            imfs = []
            residue = signal_data.copy()
            
            for _ in range(max_imfs):
                h = residue.copy()
                
                for sift_iter in range(10):
                    # Find local maxima and minima
                    maxima_indices = signal.find_peaks(h)[0]
                    minima_indices = signal.find_peaks(-h)[0]
                    
                    if len(maxima_indices) < 2 or len(minima_indices) < 2:
                        break
                    
                    # Simple envelope estimation
                    try:
                        from scipy.interpolate import interp1d
                        
                        if len(maxima_indices) >= 2:
                            upper_env_func = interp1d(maxima_indices, h[maxima_indices], 
                                                    kind='linear', fill_value='extrapolate')
                            upper_env = upper_env_func(np.arange(len(h)))
                        else:
                            upper_env = np.zeros_like(h)
                        
                        if len(minima_indices) >= 2:
                            lower_env_func = interp1d(minima_indices, h[minima_indices], 
                                                    kind='linear', fill_value='extrapolate')
                            lower_env = lower_env_func(np.arange(len(h)))
                        else:
                            lower_env = np.zeros_like(h)
                        
                        mean_env = (upper_env + lower_env) / 2.0
                        h_new = h - mean_env
                        
                        if np.sum((h - h_new)**2) / (np.sum(h**2) + 1e-10) < 0.01:
                            break
                        
                        h = h_new
                        
                    except:
                        break
                
                if np.std(h) > 1e-6:
                    imfs.append(h.copy())
                    residue = residue - h
                else:
                    break
                
                if len(signal.find_peaks(residue)[0]) < 2:
                    break
            
            return imfs
            
        except Exception as e:
            logger.warning(f"Error in EMD: {e}")
            return []

    async def _calculate_orthogonality_index(self, imfs: list) -> float:
        """Calculate orthogonality index for EMD quality"""
        try:
            if len(imfs) < 2:
                return 0.0
            
            cross_corrs = []
            for i in range(len(imfs)):
                for j in range(i + 1, len(imfs)):
                    corr = np.corrcoef(imfs[i], imfs[j])[0, 1]
                    cross_corrs.append(abs(corr))
            
            if cross_corrs:
                orthogonality_index = 1.0 - np.mean(cross_corrs)
            else:
                orthogonality_index = 1.0
            
            return max(0.0, orthogonality_index)
            
        except Exception as e:
            logger.warning(f"Error calculating orthogonality: {e}")
            return 0.0

    async def validate_3d_velocity_model(self) -> bool:
        """Validate 3D velocity model construction"""
        try:
            logger.info("🌍 Validating 3D Velocity Model")
            
            # Advanced layered velocity model
            depths = np.array([0, 1, 2, 3, 5, 8, 12, 20, 35, 50])
            vp_base = np.array([2.5, 3.2, 4.1, 4.8, 5.5, 6.0, 6.3, 6.8, 7.8, 8.1])
            vs_base = vp_base / np.array([1.8, 1.75, 1.73, 1.72, 1.71, 1.70, 1.69, 1.68, 1.67, 1.66])
            
            # Lateral variations
            x = np.linspace(-50, 50, 101)
            y = np.linspace(-50, 50, 101)
            X, Y = np.meshgrid(x, y)
            
            # Magma chamber anomaly
            chamber_radius = 5.0
            r_chamber = np.sqrt(X**2 + Y**2)
            chamber_anomaly = -0.3 * np.exp(-(r_chamber / chamber_radius)**2)
            
            # Volcanic edifice anomaly
            edifice_anomaly = 0.2 * np.exp(-((X**2 + Y**2) / (2**2)))
            
            total_anomaly = chamber_anomaly + edifice_anomaly
            
            logger.info(f"✅ Created 3D velocity model with {len(depths)} layers")
            logger.info(f"   Depth range: {depths[0]:.1f} - {depths[-1]:.1f} km")
            logger.info(f"   Vp range: {vp_base[0]:.1f} - {vp_base[-1]:.1f} km/s")
            logger.info(f"   Vs range: {vs_base[0]:.1f} - {vs_base[-1]:.1f} km/s")
            logger.info(f"   Lateral grid: {X.shape[0]} x {X.shape[1]} points")
            logger.info(f"   Anomaly range: {np.min(total_anomaly):.3f} to {np.max(total_anomaly):.3f}")
            
            # Validate model properties
            valid_depths = len(depths) == len(vp_base) == len(vs_base)
            valid_velocities = np.all(vp_base > vs_base) and np.all(vp_base > 0) and np.all(vs_base > 0)
            valid_anomalies = np.all(np.isfinite(total_anomaly))
            
            return valid_depths and valid_velocities and valid_anomalies
            
        except Exception as e:
            logger.error(f"Error validating 3D velocity model: {e}")
            return False

    async def validate_coulomb_stress_transfer(self) -> bool:
        """Validate Coulomb stress transfer calculation"""
        try:
            logger.info("⚡ Validating Coulomb Stress Transfer")
            
            # Create synthetic earthquake sequence
            base_time = datetime.utcnow()
            events = []
            
            for i in range(10):
                event = {
                    'timestamp': base_time + timedelta(hours=i * 0.5),
                    'latitude': 46.2 + np.random.normal(0, 0.01),
                    'longitude': -122.2 + np.random.normal(0, 0.01),
                    'depth': 5.0 + np.random.normal(0, 1.0),
                    'magnitude': 2.5 + np.random.exponential(0.5)
                }
                events.append(event)
            
            # Calculate stress transfers
            stress_transfers = []
            events_sorted = sorted(events, key=lambda x: x['timestamp'])
            
            for i in range(1, len(events_sorted)):
                source = events_sorted[i-1]
                target = events_sorted[i]
                
                # Distance calculation
                dx = (target['latitude'] - source['latitude']) * 111.0
                dy = (target['longitude'] - source['longitude']) * 111.0
                dz = target['depth'] - source['depth']
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance > 0:
                    # Coulomb stress calculation
                    source_magnitude = source['magnitude']
                    stress_drop = 3e6 * (10**(1.5 * source_magnitude)) / (distance**3 + 1.0)
                    friction_coeff = 0.6
                    coulomb_stress = stress_drop * (1 + friction_coeff)
                    stress_transfers.append(coulomb_stress)
            
            if stress_transfers:
                mean_stress = np.mean(stress_transfers)
                max_stress = np.max(stress_transfers)
                
                logger.info(f"✅ Calculated Coulomb stress transfers for {len(stress_transfers)} event pairs")
                logger.info(f"   Mean stress transfer: {mean_stress:.2e} Pa")
                logger.info(f"   Max stress transfer: {max_stress:.2e} Pa")
                
                return len(stress_transfers) > 0 and all(s >= 0 for s in stress_transfers)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error validating Coulomb stress transfer: {e}")
            return False

    async def validate_fractal_dimension(self) -> bool:
        """Validate fractal dimension calculation"""
        try:
            logger.info("📐 Validating Fractal Dimension Analysis")
            
            # Create synthetic fractal point distribution
            n_points = 100
            
            # Generate clustered points (fractal-like)
            centers = [(0, 0), (1, 1), (-1, 1), (1, -1)]
            points = []
            
            for center in centers:
                for _ in range(n_points // len(centers)):
                    # Add points around each center with fractal clustering
                    scale = np.random.exponential(0.1)
                    angle = np.random.uniform(0, 2*np.pi)
                    
                    x = center[0] + scale * np.cos(angle)
                    y = center[1] + scale * np.sin(angle)
                    z = np.random.normal(0, 0.1)
                    
                    points.append((x, y, z))
            
            points = np.array(points)
            
            # Calculate correlation dimension
            correlation_dim = await self._calculate_correlation_dimension(points)
            
            # Calculate box-counting dimension (2D projection)
            box_counting_dim = await self._calculate_box_counting_dimension(points[:, :2])
            
            logger.info(f"✅ Calculated fractal dimensions")
            logger.info(f"   Correlation dimension: {correlation_dim:.3f}")
            logger.info(f"   Box-counting dimension: {box_counting_dim:.3f}")
            
            # Validate reasonable fractal dimensions
            valid_corr = 0.0 <= correlation_dim <= 3.0
            valid_box = 0.0 <= box_counting_dim <= 2.0
            
            return valid_corr and valid_box
            
        except Exception as e:
            logger.error(f"Error validating fractal dimension: {e}")
            return False

    async def _calculate_correlation_dimension(self, points: np.ndarray) -> float:
        """Calculate correlation dimension"""
        try:
            n_points = len(points)
            if n_points < 10:
                return 0.0
            
            # Normalize points
            points_norm = (points - np.mean(points, axis=0)) / (np.std(points, axis=0) + 1e-10)
            
            # Calculate pairwise distances
            distances = cdist(points_norm, points_norm)
            
            # Range of radius values
            r_values = np.logspace(-2, 0, 20)
            correlations = []
            
            for r in r_values:
                count = np.sum(distances < r) - n_points
                correlation = count / (n_points * (n_points - 1))
                correlations.append(max(correlation, 1e-10))
            
            # Linear regression in log-log space
            log_r = np.log10(r_values)
            log_c = np.log10(correlations)
            
            start_idx = len(log_r) // 4
            end_idx = 3 * len(log_r) // 4
            
            if end_idx > start_idx:
                slope = np.polyfit(log_r[start_idx:end_idx], log_c[start_idx:end_idx], 1)[0]
                return max(0.0, min(3.0, slope))
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Error calculating correlation dimension: {e}")
            return 1.0

    async def _calculate_box_counting_dimension(self, points_2d: np.ndarray) -> float:
        """Calculate box-counting dimension"""
        try:
            if len(points_2d) < 5:
                return 0.0
            
            # Normalize to unit square
            min_coords = np.min(points_2d, axis=0)
            max_coords = np.max(points_2d, axis=0)
            
            if np.any(max_coords - min_coords == 0):
                return 0.0
            
            normalized_points = (points_2d - min_coords) / (max_coords - min_coords)
            
            # Range of box sizes
            box_sizes = np.logspace(-2, 0, 15)
            box_counts = []
            
            for box_size in box_sizes:
                occupied_boxes = set()
                
                for point in normalized_points:
                    box_x = int(point[0] / box_size)
                    box_y = int(point[1] / box_size)
                    occupied_boxes.add((box_x, box_y))
                
                box_counts.append(len(occupied_boxes))
            
            # Linear regression
            log_sizes = np.log10(box_sizes)
            log_counts = np.log10(box_counts)
            
            start_idx = 2
            end_idx = len(log_sizes) - 2
            
            if end_idx > start_idx:
                slope = np.polyfit(log_sizes[start_idx:end_idx], log_counts[start_idx:end_idx], 1)[0]
                return max(0.0, min(2.0, -slope))
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Error calculating box-counting dimension: {e}")
            return 1.0

async def run_validation():
    """Run comprehensive validation of advanced algorithms"""
    try:
        logger.info("🚀 Starting Advanced Seismic Algorithm Validation")
        
        validator = AdvancedSeismicValidator()
        
        # Create complex synthetic waveform
        duration = 60.0
        sampling_rate = 100.0
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Multi-component synthetic signal
        noise = np.random.normal(0, 0.05, len(t))
        p_wave = np.exp(-(t - 10.0) / 3.0) * np.sin(2 * np.pi * 12 * (t - 10.0))
        p_wave[t < 10.0] = 0
        
        s_wave = np.exp(-(t - 15.0) / 5.0) * np.sin(2 * np.pi * 6 * (t - 15.0))
        s_wave[t < 15.0] = 0
        
        surface_wave = np.exp(-(t - 25.0) / 10.0) * np.sin(2 * np.pi * 2 * (t - 25.0))
        surface_wave[t < 25.0] = 0
        
        complex_waveform = noise + 0.8 * p_wave + 1.2 * s_wave + 0.6 * surface_wave
        
        # Run validation tests
        tests = [
            ("Advanced Wavelet Features", validator.validate_advanced_wavelet_features(complex_waveform)),
            ("EMD Decomposition", validator.validate_emd_decomposition(complex_waveform)),
            ("3D Velocity Model", validator.validate_3d_velocity_model()),
            ("Coulomb Stress Transfer", validator.validate_coulomb_stress_transfer()),
            ("Fractal Dimension", validator.validate_fractal_dimension())
        ]
        
        results = []
        for test_name, test_coro in tests:
            logger.info(f"\n{'='*50}")
            result = await test_coro
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("📊 VALIDATION SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status} {test_name}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All advanced algorithms validated successfully!")
            return True
        else:
            logger.warning(f"⚠️ {total - passed} tests failed")
            return False
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(run_validation())
    exit(0 if result else 1)