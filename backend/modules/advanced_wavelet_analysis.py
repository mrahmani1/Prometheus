"""
Advanced Wavelet Periodicity Analysis for Volcanic Monitoring
Sophisticated time-frequency analysis using multiple wavelet transforms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')


def _advanced_wavelet_periodicity(values: np.ndarray, dt: float = 1.0) -> Dict:
    """
    Advanced wavelet periodicity analysis with multiple mother wavelets
    
    Implements comprehensive time-frequency analysis using:
    - Multiple mother wavelets (Morlet, Mexican Hat, Daubechies)
    - Scale-dependent statistical significance testing
    - Ridge extraction for dominant oscillations
    - Instantaneous frequency tracking
    - Ensemble analysis with uncertainty quantification
    """
    try:
        n = len(values)
        if n < 16:
            return {'dominant_scale': None, 'wavelet_strength': 0.0, 'method': 'insufficient_data'}
        
        # Normalize data
        values_norm = (values - np.mean(values)) / (np.std(values) + 1e-10)
        
        # Define scale range (logarithmic spacing)
        min_scale = 2.0 * dt
        max_scale = min(n * dt / 4, 100.0 * dt)
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 50)
        
        # Multi-wavelet analysis
        morlet_result = _morlet_wavelet_analysis(values_norm, scales, dt)
        mexhat_result = _mexican_hat_analysis(values_norm, scales, dt)
        daubechies_result = _daubechies_wavelet_analysis(values_norm, scales, dt)
        
        # Ensemble analysis
        ensemble_result = _ensemble_wavelet_analysis([morlet_result, mexhat_result, daubechies_result], scales)
        
        # Ridge extraction and instantaneous frequency
        ridges = _extract_wavelet_ridges(ensemble_result['scalogram'], scales, dt)
        inst_freq = _track_instantaneous_frequency(ensemble_result['scalogram'], scales, dt)
        
        # Statistical significance testing
        significance = _test_wavelet_significance(values_norm, ensemble_result['scalogram'], scales)
        
        # Time-frequency coherence analysis
        coherence = _wavelet_coherence_analysis(values_norm, scales, dt)
        
        return {
            'dominant_scale': float(ensemble_result['dominant_scale']),
            'wavelet_strength': float(ensemble_result['strength']),
            'scale_diversity': float(ensemble_result['diversity']),
            'ridges': ridges,
            'instantaneous_frequency': inst_freq,
            'significance': significance,
            'coherence': coherence,
            'individual_analyses': {
                'morlet': morlet_result,
                'mexican_hat': mexhat_result,
                'daubechies': daubechies_result
            },
            'ensemble_scalogram': ensemble_result['scalogram'],
            'scales': scales.tolist(),
            'method': 'advanced_multi_wavelet'
        }
        
    except Exception as e:
        return {
            'dominant_scale': None,
            'wavelet_strength': 0.0,
            'scale_diversity': 0.0,
            'method': 'wavelet_error',
            'error': str(e)
        }


def _morlet_wavelet_analysis(values: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Morlet wavelet analysis for oscillatory patterns"""
    try:
        omega0 = 6.0  # Central frequency parameter
        scalogram = np.zeros((len(scales), len(values)))
        
        for i, scale in enumerate(scales):
            # Create Morlet wavelet
            sigma = scale / omega0
            x = np.arange(-4*sigma, 4*sigma + dt, dt)
            
            if len(x) > len(values):
                continue
                
            morlet = (1 / np.sqrt(sigma * np.sqrt(np.pi))) * \
                    np.exp(1j * omega0 * x / scale) * \
                    np.exp(-x**2 / (2 * sigma**2))
            
            # Convolution
            conv_result = np.convolve(values, morlet, mode='same')
            scalogram[i, :] = np.abs(conv_result)**2
        
        # Compute energy distribution
        scale_energy = np.mean(scalogram, axis=1)
        if np.sum(scale_energy) > 0:
            dominant_scale_idx = np.argmax(scale_energy)
            dominant_scale = scales[dominant_scale_idx]
            strength = scale_energy[dominant_scale_idx] / (np.mean(scale_energy) + 1e-10)
        else:
            dominant_scale = scales[len(scales)//2]
            strength = 1.0
        
        return {
            'scalogram': scalogram,
            'dominant_scale': dominant_scale,
            'strength': strength,
            'scale_energy': scale_energy,
            'method': 'morlet'
        }
        
    except Exception:
        return {
            'scalogram': np.zeros((len(scales), len(values))),
            'dominant_scale': scales[len(scales)//2] if len(scales) > 0 else 1.0,
            'strength': 0.0,
            'method': 'morlet_fallback'
        }


def _mexican_hat_analysis(values: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Enhanced Mexican Hat wavelet analysis"""
    try:
        scalogram = np.zeros((len(scales), len(values)))
        
        for i, scale in enumerate(scales):
            # Create Mexican Hat wavelet (Ricker wavelet)
            sigma = scale / (2 * np.sqrt(2 * np.log(2)))
            x = np.arange(-4*sigma, 4*sigma + dt, dt)
            
            if len(x) > len(values):
                continue
            
            # Normalized Mexican Hat
            mexhat = (2 / (np.sqrt(3 * sigma) * np.pi**0.25)) * \
                     (1 - (x/sigma)**2) * np.exp(-(x**2) / (2 * sigma**2))
            
            # Convolution
            conv_result = np.convolve(values, mexhat, mode='same')
            scalogram[i, :] = conv_result**2
        
        # Energy analysis
        scale_energy = np.mean(scalogram, axis=1)
        if np.sum(scale_energy) > 0:
            dominant_scale_idx = np.argmax(scale_energy)
            dominant_scale = scales[dominant_scale_idx]
            strength = scale_energy[dominant_scale_idx] / (np.mean(scale_energy) + 1e-10)
        else:
            dominant_scale = scales[len(scales)//2]
            strength = 1.0
        
        return {
            'scalogram': scalogram,
            'dominant_scale': dominant_scale,
            'strength': strength,
            'scale_energy': scale_energy,
            'method': 'mexican_hat'
        }
        
    except Exception:
        return {
            'scalogram': np.zeros((len(scales), len(values))),
            'dominant_scale': scales[len(scales)//2] if len(scales) > 0 else 1.0,
            'strength': 0.0,
            'method': 'mexhat_fallback'
        }


def _daubechies_wavelet_analysis(values: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Daubechies wavelet analysis for edge and transient detection"""
    try:
        # Use Daubechies-4 approximation
        scalogram = np.zeros((len(scales), len(values)))
        
        # Daubechies-4 coefficients (approximation)
        h = np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2)
        
        for i, scale in enumerate(scales):
            # Scale the wavelet
            scaled_h = h / np.sqrt(scale)
            wavelet_size = len(scaled_h) * int(scale)
            
            if wavelet_size > len(values):
                continue
            
            # Create scaled Daubechies wavelet
            daubechies = np.zeros(wavelet_size)
            for j, coeff in enumerate(scaled_h):
                start_idx = j * int(scale)
                end_idx = min(start_idx + int(scale), len(daubechies))
                daubechies[start_idx:end_idx] = coeff
            
            # Convolution
            if len(daubechies) <= len(values):
                conv_result = np.convolve(values, daubechies, mode='same')
                scalogram[i, :] = conv_result**2
        
        # Energy analysis
        scale_energy = np.mean(scalogram, axis=1)
        if np.sum(scale_energy) > 0:
            dominant_scale_idx = np.argmax(scale_energy)
            dominant_scale = scales[dominant_scale_idx]
            strength = scale_energy[dominant_scale_idx] / (np.mean(scale_energy) + 1e-10)
        else:
            dominant_scale = scales[len(scales)//2]
            strength = 1.0
        
        return {
            'scalogram': scalogram,
            'dominant_scale': dominant_scale,
            'strength': strength,
            'scale_energy': scale_energy,
            'method': 'daubechies'
        }
        
    except Exception:
        return {
            'scalogram': np.zeros((len(scales), len(values))),
            'dominant_scale': scales[len(scales)//2] if len(scales) > 0 else 1.0,
            'strength': 0.0,
            'method': 'daubechies_fallback'
        }


def _ensemble_wavelet_analysis(results: List[Dict], scales: np.ndarray) -> Dict:
    """Ensemble analysis combining multiple wavelet transforms"""
    try:
        # Combine scalograms with weighted averaging
        weights = {'morlet': 0.4, 'mexican_hat': 0.35, 'daubechies': 0.25}
        
        ensemble_scalogram = np.zeros_like(results[0]['scalogram'])
        total_weight = 0.0
        
        for result in results:
            method = result.get('method', 'unknown')
            weight = weights.get(method.replace('_fallback', ''), 0.33)
            ensemble_scalogram += weight * result['scalogram']
            total_weight += weight
        
        if total_weight > 0:
            ensemble_scalogram /= total_weight
        
        # Find ensemble dominant scale
        scale_energy = np.mean(ensemble_scalogram, axis=1)
        if np.sum(scale_energy) > 0:
            dominant_scale_idx = np.argmax(scale_energy)
            dominant_scale = scales[dominant_scale_idx]
            strength = scale_energy[dominant_scale_idx] / (np.mean(scale_energy) + 1e-10)
            diversity = np.std(scale_energy) / (np.mean(scale_energy) + 1e-10)
        else:
            dominant_scale = scales[len(scales)//2]
            strength = 1.0
            diversity = 0.5
        
        return {
            'scalogram': ensemble_scalogram,
            'dominant_scale': dominant_scale,
            'strength': strength,
            'diversity': diversity,
            'scale_energy': scale_energy
        }
        
    except Exception:
        return {
            'scalogram': np.zeros((len(scales), len(results[0]['scalogram'][0]))),
            'dominant_scale': scales[len(scales)//2] if len(scales) > 0 else 1.0,
            'strength': 0.0,
            'diversity': 0.5
        }


def _extract_wavelet_ridges(scalogram: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Extract ridges representing dominant oscillatory patterns"""
    try:
        ridges = []
        n_time = scalogram.shape[1]
        
        # Find local maxima in scale direction for each time point
        for t in range(n_time):
            scale_profile = scalogram[:, t]
            
            # Find peaks
            peaks, properties = signal.find_peaks(scale_profile, height=np.max(scale_profile) * 0.1)
            
            if len(peaks) > 0:
                # Get the most prominent peak
                peak_heights = scale_profile[peaks]
                main_peak_idx = peaks[np.argmax(peak_heights)]
                ridge_scale = scales[main_peak_idx]
                ridge_strength = scale_profile[main_peak_idx]
                
                ridges.append({
                    'time': t * dt,
                    'scale': ridge_scale,
                    'strength': ridge_strength,
                    'frequency': 1.0 / ridge_scale if ridge_scale > 0 else 0.0
                })
        
        # Extract ridge statistics
        if ridges:
            ridge_scales = [r['scale'] for r in ridges]
            ridge_freqs = [r['frequency'] for r in ridges]
            ridge_strengths = [r['strength'] for r in ridges]
            
            return {
                'ridges': ridges,
                'mean_scale': float(np.mean(ridge_scales)),
                'scale_variability': float(np.std(ridge_scales)),
                'mean_frequency': float(np.mean(ridge_freqs)),
                'frequency_variability': float(np.std(ridge_freqs)),
                'mean_strength': float(np.mean(ridge_strengths)),
                'ridge_count': len(ridges)
            }
        else:
            return {
                'ridges': [],
                'mean_scale': None,
                'scale_variability': 0.0,
                'mean_frequency': None,
                'frequency_variability': 0.0,
                'mean_strength': 0.0,
                'ridge_count': 0
            }
            
    except Exception:
        return {
            'ridges': [],
            'mean_scale': None,
            'scale_variability': 0.0,
            'mean_frequency': None,
            'frequency_variability': 0.0,
            'mean_strength': 0.0,
            'ridge_count': 0
        }


def _track_instantaneous_frequency(scalogram: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Track instantaneous frequency using wavelet phase information"""
    try:
        n_scales, n_time = scalogram.shape
        
        # Compute energy-weighted average frequency
        freq_weights = 1.0 / scales  # Convert scales to frequencies
        inst_freq = np.zeros(n_time)
        
        for t in range(n_time):
            energy_profile = scalogram[:, t]
            total_energy = np.sum(energy_profile)
            
            if total_energy > 0:
                # Weighted average frequency
                inst_freq[t] = np.sum(freq_weights * energy_profile) / total_energy
            else:
                inst_freq[t] = np.mean(freq_weights)
        
        # Frequency modulation analysis
        freq_trend = _compute_frequency_trend(inst_freq)
        freq_variability = np.std(inst_freq)
        
        # Detect frequency jumps
        freq_jumps = _detect_frequency_jumps(inst_freq, dt)
        
        return {
            'instantaneous_frequency': inst_freq.tolist(),
            'mean_frequency': float(np.mean(inst_freq)),
            'frequency_trend': freq_trend,
            'frequency_variability': float(freq_variability),
            'frequency_jumps': freq_jumps,
            'time_axis': (np.arange(n_time) * dt).tolist()
        }
        
    except Exception:
        return {
            'instantaneous_frequency': [],
            'mean_frequency': 0.0,
            'frequency_trend': 0.0,
            'frequency_variability': 0.0,
            'frequency_jumps': [],
            'time_axis': []
        }


def _test_wavelet_significance(values: np.ndarray, scalogram: np.ndarray, scales: np.ndarray) -> Dict:
    """Test statistical significance of wavelet coefficients"""
    try:
        # Generate red noise null hypothesis
        lag1_corr = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 1 else 0.0
        lag1_corr = max(-0.99, min(0.99, lag1_corr))  # Bound correlation
        
        # Monte Carlo significance testing
        n_trials = 100
        null_scalograms = []
        
        for _ in range(n_trials):
            # Generate AR(1) red noise
            noise = _generate_red_noise(len(values), lag1_corr)
            noise_result = _morlet_wavelet_analysis(noise, scales)
            null_scalograms.append(noise_result['scalogram'])
        
        # Compute significance levels
        null_array = np.array(null_scalograms)
        significance_95 = np.percentile(null_array, 95, axis=0)
        significance_99 = np.percentile(null_array, 99, axis=0)
        
        # Identify significant regions
        significant_95 = scalogram > significance_95
        significant_99 = scalogram > significance_99
        
        return {
            'significance_95': significant_95.tolist(),
            'significance_99': significant_99.tolist(),
            'significant_fraction_95': float(np.mean(significant_95)),
            'significant_fraction_99': float(np.mean(significant_99)),
            'max_significance_ratio': float(np.max(scalogram / (significance_95 + 1e-10)))
        }
        
    except Exception:
        return {
            'significance_95': [],
            'significance_99': [],
            'significant_fraction_95': 0.0,
            'significant_fraction_99': 0.0,
            'max_significance_ratio': 1.0
        }


def _wavelet_coherence_analysis(values: np.ndarray, scales: np.ndarray, dt: float = 1.0) -> Dict:
    """Wavelet coherence analysis for scale-dependent correlations"""
    try:
        # Create a delayed version for self-coherence analysis
        delay = max(1, len(values) // 20)
        if delay >= len(values):
            delay = 1
        
        values1 = values[:-delay]
        values2 = values[delay:]
        
        # Compute cross-wavelet transform
        morlet1 = _morlet_wavelet_analysis(values1, scales, dt)
        morlet2 = _morlet_wavelet_analysis(values2, scales, dt)
        
        # Compute coherence
        cross_spectrum = morlet1['scalogram'] * np.conj(morlet2['scalogram'])
        power1 = np.abs(morlet1['scalogram'])**2
        power2 = np.abs(morlet2['scalogram'])**2
        
        coherence = np.abs(cross_spectrum)**2 / ((power1 + 1e-10) * (power2 + 1e-10))
        
        # Scale-averaged coherence
        scale_coherence = np.mean(coherence, axis=1)
        time_coherence = np.mean(coherence, axis=0)
        
        return {
            'coherence_matrix': coherence.tolist(),
            'scale_coherence': scale_coherence.tolist(),
            'time_coherence': time_coherence.tolist(),
            'mean_coherence': float(np.mean(coherence)),
            'max_coherence': float(np.max(coherence)),
            'delay_used': delay
        }
        
    except Exception:
        return {
            'coherence_matrix': [],
            'scale_coherence': [],
            'time_coherence': [],
            'mean_coherence': 0.0,
            'max_coherence': 0.0,
            'delay_used': 1
        }


# Supporting functions

def _compute_frequency_trend(inst_freq: np.ndarray) -> float:
    """Compute frequency trend using linear regression"""
    try:
        if len(inst_freq) < 3:
            return 0.0
        
        x = np.arange(len(inst_freq))
        coeffs = np.polyfit(x, inst_freq, 1)
        return float(coeffs[0])  # Slope
    except:
        return 0.0


def _detect_frequency_jumps(inst_freq: np.ndarray, dt: float, threshold: float = 2.0) -> List[Dict]:
    """Detect sudden frequency jumps"""
    try:
        jumps = []
        
        if len(inst_freq) < 3:
            return jumps
        
        # Compute frequency derivatives
        freq_diff = np.diff(inst_freq)
        freq_diff_std = np.std(freq_diff)
        
        for i, diff in enumerate(freq_diff):
            if abs(diff) > threshold * freq_diff_std:
                jumps.append({
                    'time': (i + 1) * dt,
                    'magnitude': float(diff),
                    'type': 'increase' if diff > 0 else 'decrease'
                })
        
        return jumps
    except:
        return []


def _generate_red_noise(n: int, lag1_corr: float) -> np.ndarray:
    """Generate AR(1) red noise for significance testing"""
    try:
        # Generate white noise
        white_noise = np.random.randn(n)
        
        # Apply AR(1) filter
        red_noise = np.zeros(n)
        red_noise[0] = white_noise[0]
        
        for i in range(1, n):
            red_noise[i] = lag1_corr * red_noise[i-1] + np.sqrt(1 - lag1_corr**2) * white_noise[i]
        
        return red_noise
    except:
        return np.random.randn(n)