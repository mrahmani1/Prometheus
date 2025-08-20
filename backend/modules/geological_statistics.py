"""
Geological Statistics Module
Advanced statistical methods for geological data analysis including:
- Circular/spherical statistics for orientation data
- Directional data analysis
- Fisher distributions and hypothesis testing
- Multi-scale stability analysis
- Uncertainty quantification
"""

import numpy as np
from scipy import stats
from scipy.special import i0
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings


class CircularStatistics:
    """Advanced circular statistics for geological orientation analysis"""
    
    @staticmethod
    def compute_advanced_circular_statistics(angles_rad: np.ndarray) -> Dict:
        """Compute comprehensive circular statistics"""
        n = len(angles_rad)
        
        # Basic circular moments
        cos_angles = np.cos(angles_rad)
        sin_angles = np.sin(angles_rad)
        
        # First moment (mean direction)
        C1 = np.mean(cos_angles)
        S1 = np.mean(sin_angles)
        R1 = np.sqrt(C1**2 + S1**2)  # Mean resultant length
        
        # Mean direction
        mean_angle = np.arctan2(S1, C1)
        
        # Second moment for dispersion
        C2 = np.mean(np.cos(2 * angles_rad))
        S2 = np.mean(np.sin(2 * angles_rad))
        R2 = np.sqrt(C2**2 + S2**2)
        
        # Circular variance and standard deviation
        circular_variance = 1 - R1
        circular_std = np.sqrt(-2 * np.log(R1)) if R1 > 0 else np.pi
        
        # Von Mises concentration parameter estimation
        if R1 < 0.53:
            kappa = 2 * R1 + R1**3 + (5 * R1**5) / 6
        elif R1 < 0.85:
            kappa = -0.4 + 1.39 * R1 + 0.43 / (1 - R1)
        else:
            kappa = 1 / (R1**3 - 4 * R1**2 + 3 * R1) if R1 < 1.0 else 1000.0
        
        # Skewness and kurtosis for circular data
        skewness = (S2 * np.sin(mean_angle) - C2 * np.cos(mean_angle)) / (R1**1.5) if R1 > 0 else 0
        kurtosis = (C2 * np.cos(2 * mean_angle) + S2 * np.sin(2 * mean_angle) - R1**2) / (circular_variance**2) if circular_variance > 0 else 0
        
        return {
            'mean_direction_rad': float(mean_angle),
            'mean_direction_deg': float(np.degrees(mean_angle)),
            'mean_resultant_length': float(R1),
            'circular_variance': float(circular_variance),
            'circular_std_rad': float(circular_std),
            'circular_std_deg': float(np.degrees(circular_std)),
            'von_mises_kappa': float(kappa),
            'second_moment_length': float(R2),
            'circular_skewness': float(skewness),
            'circular_kurtosis': float(kurtosis),
            'sample_size': n
        }
    
    @staticmethod
    def fit_von_mises_distribution(angles_rad: np.ndarray) -> Dict:
        """Fit Von Mises distribution to circular data"""
        stats = CircularStatistics.compute_advanced_circular_statistics(angles_rad)
        mu = stats['mean_direction_rad']
        kappa = stats['von_mises_kappa']
        
        # Calculate log-likelihood
        try:
            log_likelihood = len(angles_rad) * (kappa * np.cos(angles_rad - mu).sum() / len(angles_rad) - np.log(2 * np.pi * i0(kappa)))
        except:
            log_likelihood = -np.inf
        
        # AIC and BIC
        aic = -2 * log_likelihood + 2 * 2  # 2 parameters
        bic = -2 * log_likelihood + np.log(len(angles_rad)) * 2
        
        return {
            'mu': float(mu),
            'kappa': float(kappa),
            'log_likelihood': float(log_likelihood),
            'aic': float(aic),
            'bic': float(bic)
        }
    
    @staticmethod
    def fit_watson_distribution(angles_rad: np.ndarray) -> Dict:
        """Fit Watson distribution (for girdle-type distributions)"""
        # Convert to 3D unit vectors
        vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad), np.zeros(len(angles_rad))])
        
        # Calculate orientation matrix
        orientation_matrix = vectors.T @ vectors / len(vectors)
        eigenvals, eigenvecs = np.linalg.eigh(orientation_matrix)
        
        # Watson concentration parameter (simplified)
        kappa = (eigenvals[-1] - eigenvals[0]) * len(angles_rad)
        
        return {
            'kappa': float(kappa),
            'principal_direction': eigenvecs[:, -1].tolist(),
            'eigenvalues': eigenvals.tolist()
        }
    
    @staticmethod
    def fit_fisher_distribution(angles_rad: np.ndarray) -> Dict:
        """Fit Fisher distribution for 3D directional data"""
        # Convert to 3D unit vectors on sphere
        vectors = np.column_stack([
            np.cos(angles_rad),
            np.sin(angles_rad), 
            np.zeros(len(angles_rad))
        ])
        
        # Mean direction
        mean_vector = np.mean(vectors, axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)
        
        # Resultant length
        R = np.linalg.norm(np.mean(vectors, axis=0))
        
        # Fisher concentration parameter
        n = len(vectors)
        if R < 1.0:
            kappa = (n - 1) / (n - R)
        else:
            kappa = np.inf
        
        return {
            'mean_direction': mean_vector.tolist(),
            'kappa': float(kappa),
            'resultant_length': float(R)
        }
    
    @staticmethod
    def fit_wrapped_normal_distribution(angles_rad: np.ndarray) -> Dict:
        """Fit wrapped normal distribution"""
        # Method of moments estimation
        C = np.mean(np.cos(angles_rad))
        S = np.mean(np.sin(angles_rad))
        
        mu = np.arctan2(S, C)
        R = np.sqrt(C**2 + S**2)
        
        # Estimate concentration parameter rho
        if R > 0:
            rho = np.sqrt(-2 * np.log(R))
        else:
            rho = np.pi
        
        return {
            'mu': float(mu),
            'rho': float(rho),
            'resultant_length': float(R)
        }


class HypothesisTesting:
    """Hypothesis testing for circular and directional data"""
    
    @staticmethod
    def rayleigh_test(angles_rad: np.ndarray) -> Tuple[float, float]:
        """Rayleigh test for uniformity"""
        n = len(angles_rad)
        C = np.sum(np.cos(angles_rad))
        S = np.sum(np.sin(angles_rad))
        R = np.sqrt(C**2 + S**2)
        
        # Test statistic
        R_bar = R / n
        test_stat = n * R_bar**2
        
        # P-value approximation
        p_value = np.exp(-test_stat)
        
        return test_stat, p_value
    
    @staticmethod
    def kuiper_test(angles_rad: np.ndarray) -> Tuple[float, float]:
        """Kuiper test for circular uniformity"""
        n = len(angles_rad)
        angles_sorted = np.sort(angles_rad)
        
        # Convert to [0, 2π]
        angles_uniform = (angles_sorted + 2*np.pi) % (2*np.pi)
        angles_uniform = np.sort(angles_uniform)
        
        # Empirical distribution function
        i = np.arange(1, n + 1)
        
        # Calculate D+ and D-
        D_plus = np.max((i / n) - (angles_uniform / (2 * np.pi)))
        D_minus = np.max((angles_uniform / (2 * np.pi)) - ((i - 1) / n))
        
        # Kuiper statistic
        V = D_plus + D_minus
        
        # Critical value approximation
        V_crit = (1.747 / np.sqrt(n)) * np.sqrt(1 + 0.155/n + 0.24/n**2)
        p_value = 2 * np.exp(-2 * n * V**2) if V > V_crit else 1.0
        
        return V, p_value
    
    @staticmethod
    def watson_u2_test(angles_rad: np.ndarray) -> Tuple[float, float]:
        """Watson U-squared test for uniformity"""
        n = len(angles_rad)
        angles_sorted = np.sort((angles_rad + 2*np.pi) % (2*np.pi))
        
        # Calculate U-squared statistic
        U2 = 0
        for j in range(n):
            U2 += (angles_sorted[j] / (2*np.pi) - (2*j - 1)/(2*n))**2
        U2 = U2 + 1/(12*n)
        
        # Approximate p-value
        if U2 < 0.052:
            p_value = 1 - np.exp(-1.386294 * ((U2 - 0.052) / 0.052)**2)
        else:
            p_value = 2 * np.exp(-2 * U2)
        
        return U2, max(0.0, min(1.0, p_value))
    
    @staticmethod
    def hermans_rasson_test(angles_rad: np.ndarray) -> Tuple[float, float]:
        """Hermans-Rasson test for symmetry"""
        n = len(angles_rad)
        if n < 8:
            return 0.0, 1.0
        
        sorted_angles = np.sort(angles_rad)
        mean_angle = np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))
        
        reflected_angles = 2 * mean_angle - sorted_angles
        
        ks_statistic = np.max(np.abs(
            np.searchsorted(sorted_angles, reflected_angles) / n - 
            np.arange(1, n+1) / n
        ))
        
        p_value = 2 * np.exp(-2 * n * ks_statistic**2) if ks_statistic > 0 else 1.0
        
        return float(ks_statistic), float(min(1.0, p_value))
    
    @staticmethod
    def test_specific_mean_direction(angles_rad: np.ndarray, test_angle: float = 0.0) -> Dict:
        """Test for specific mean direction"""
        stats = CircularStatistics.compute_advanced_circular_statistics(angles_rad)
        
        mean_dir = stats['mean_direction_rad']
        angular_diff = abs(np.arctan2(np.sin(mean_dir - test_angle), np.cos(mean_dir - test_angle)))
        
        n = len(angles_rad)
        kappa = stats['von_mises_kappa']
        
        if kappa > 0 and n > 5:
            ci_half_width = 1.96 / np.sqrt(n * kappa)
        else:
            ci_half_width = np.pi / 2
        
        is_significant = angular_diff > ci_half_width
        
        return {
            'test_angle_rad': float(test_angle),
            'mean_direction_rad': float(mean_dir),
            'angular_difference_rad': float(angular_diff),
            'is_significantly_different': bool(is_significant),
            'confidence_interval_half_width': float(ci_half_width),
            'p_value_approx': float(2 * (1 - stats.norm.cdf(angular_diff / (ci_half_width / 1.96))) if ci_half_width > 0 else 0.5)
        }


class OrientationAnalysis:
    """Advanced orientation data analysis for geological structures"""
    
    def __init__(self):
        self.circular_stats = CircularStatistics()
        self.hypothesis_tests = HypothesisTesting()
    
    def analyze_circular_dispersion(self, angles: List[float]) -> Dict:
        """Advanced directional statistics with multiple circular distribution models"""
        if len(angles) < 2:
            return {
                'dispersion_degrees': 0.0,
                'insufficient_data': True,
                'analysis_quality': 'insufficient'
            }
        
        angles_array = np.array(angles)
        n = len(angles_array)
        
        # Convert to radians for calculations
        angles_rad = angles_array * np.pi / 180.0
        
        # Advanced circular statistics computation
        circular_stats = self.circular_stats.compute_advanced_circular_statistics(angles_rad)
        
        # Multiple distribution fitting
        distribution_analysis = self._fit_circular_distributions(angles_rad)
        
        # Hypothesis testing for uniformity and specific distributions
        hypothesis_tests = self._perform_circular_hypothesis_tests(angles_rad)
        
        # Confidence intervals and bootstrap analysis
        confidence_analysis = self._calculate_circular_confidence_intervals(angles_rad)
        
        # Harmonic analysis for periodic components
        harmonic_analysis = self._perform_circular_harmonic_analysis(angles_rad)
        
        # Outlier detection in circular data
        outlier_analysis = self._detect_circular_outliers(angles_rad)
        
        # Multi-modal analysis
        modality_analysis = self._analyze_circular_modality(angles_rad)
        
        # Dispersion measures collection
        dispersion_measures = self._calculate_multiple_dispersion_measures(circular_stats)
        
        return {
            'dispersion_degrees': float(dispersion_measures['primary_dispersion']),
            'circular_statistics': circular_stats,
            'distribution_analysis': distribution_analysis,
            'hypothesis_tests': hypothesis_tests,
            'confidence_analysis': confidence_analysis,
            'harmonic_analysis': harmonic_analysis,
            'outlier_analysis': outlier_analysis,
            'modality_analysis': modality_analysis,
            'dispersion_measures': dispersion_measures,
            'sample_size': n,
            'analysis_quality': self._assess_circular_analysis_quality(n, circular_stats)
        }
    
    def _fit_circular_distributions(self, angles_rad: np.ndarray) -> Dict:
        """Fit multiple circular distributions and select best model"""
        distributions = {}
        
        # Von Mises distribution
        vm_params = self.circular_stats.fit_von_mises_distribution(angles_rad)
        distributions['von_mises'] = vm_params
        
        # Watson distribution (for girdle distributions)
        watson_params = self.circular_stats.fit_watson_distribution(angles_rad)
        distributions['watson'] = watson_params
        
        # Fisher distribution (3D directional data)
        fisher_params = self.circular_stats.fit_fisher_distribution(angles_rad)
        distributions['fisher'] = fisher_params
        
        # Wrapped normal distribution
        wrapped_normal_params = self.circular_stats.fit_wrapped_normal_distribution(angles_rad)
        distributions['wrapped_normal'] = wrapped_normal_params
        
        # Model selection using AIC/BIC
        model_selection = self._select_best_circular_model(angles_rad, distributions)
        
        return {
            'fitted_distributions': distributions,
            'best_model': model_selection['best_model'],
            'model_selection_criteria': model_selection,
            'goodness_of_fit': self._assess_circular_goodness_of_fit(angles_rad, distributions)
        }
    
    def _perform_circular_hypothesis_tests(self, angles_rad: np.ndarray) -> Dict:
        """Perform comprehensive circular hypothesis tests"""
        # Rayleigh test for uniformity
        rayleigh_stat, rayleigh_p = self.hypothesis_tests.rayleigh_test(angles_rad)
        
        # Kuiper test for uniformity (more powerful than Rayleigh)
        kuiper_stat, kuiper_p = self.hypothesis_tests.kuiper_test(angles_rad)
        
        # Watson U-squared test
        watson_stat, watson_p = self.hypothesis_tests.watson_u2_test(angles_rad)
        
        # Hermans-Rasson test for symmetry
        hr_stat, hr_p = self.hypothesis_tests.hermans_rasson_test(angles_rad)
        
        # Test for specific mean direction
        mean_direction_test = self.hypothesis_tests.test_specific_mean_direction(angles_rad)
        
        return {
            'rayleigh_test': {'statistic': float(rayleigh_stat), 'p_value': float(rayleigh_p)},
            'kuiper_test': {'statistic': float(kuiper_stat), 'p_value': float(kuiper_p)},
            'watson_u2_test': {'statistic': float(watson_stat), 'p_value': float(watson_p)},
            'hermans_rasson_test': {'statistic': float(hr_stat), 'p_value': float(hr_p)},
            'mean_direction_test': mean_direction_test,
            'overall_uniformity_p': float(min(rayleigh_p, kuiper_p, watson_p)),
            'test_interpretations': self._interpret_circular_tests(rayleigh_p, kuiper_p, watson_p)
        }
    
    def _calculate_circular_confidence_intervals(self, angles_rad: np.ndarray, alpha: float = 0.05) -> Dict:
        """Calculate confidence intervals for circular parameters"""
        n = len(angles_rad)
        
        # Bootstrap confidence intervals
        n_bootstrap = 2000
        bootstrap_means = []
        bootstrap_concentrations = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(angles_rad, size=n, replace=True)
            stats = self.circular_stats.compute_advanced_circular_statistics(bootstrap_sample)
            bootstrap_means.append(stats['mean_direction_rad'])
            bootstrap_concentrations.append(stats['von_mises_kappa'])
        
        # Handle circular nature of mean direction
        bootstrap_means = np.array(bootstrap_means)
        mean_ci = self._circular_confidence_interval(bootstrap_means, alpha)
        
        # Concentration parameter CI
        bootstrap_concentrations = np.array(bootstrap_concentrations)
        concentration_ci = [
            float(np.percentile(bootstrap_concentrations, 100 * alpha / 2)),
            float(np.percentile(bootstrap_concentrations, 100 * (1 - alpha / 2)))
        ]
        
        return {
            'mean_direction_ci': mean_ci,
            'concentration_ci': concentration_ci,
            'confidence_level': float(1 - alpha),
            'bootstrap_samples': n_bootstrap,
            'ci_method': 'bootstrap_percentile'
        }
    
    def _perform_circular_harmonic_analysis(self, angles_rad: np.ndarray) -> Dict:
        """Perform harmonic analysis to detect periodic components"""
        n = len(angles_rad)
        
        # Calculate harmonic coefficients up to order 5
        harmonics = {}
        for k in range(1, 6):
            a_k = np.mean(np.cos(k * angles_rad))
            b_k = np.mean(np.sin(k * angles_rad))
            R_k = np.sqrt(a_k**2 + b_k**2)
            phi_k = np.arctan2(b_k, a_k)
            
            harmonics[f'harmonic_{k}'] = {
                'amplitude': float(R_k),
                'phase': float(phi_k),
                'a_coefficient': float(a_k),
                'b_coefficient': float(b_k)
            }
        
        # Identify dominant harmonics
        dominant_harmonics = sorted(
            [(k, v['amplitude']) for k, v in harmonics.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        return {
            'harmonic_coefficients': harmonics,
            'dominant_harmonics': dominant_harmonics,
            'total_harmonic_content': float(sum(h['amplitude'] for h in harmonics.values())),
            'harmonic_complexity': len([h for h in harmonics.values() if h['amplitude'] > 0.1])
        }
    
    def _detect_circular_outliers(self, angles_rad: np.ndarray) -> Dict:
        """Detect outliers in circular data"""
        n = len(angles_rad)
        
        if n < 5:
            return {'outlier_indices': [], 'outlier_count': 0, 'insufficient_data': True}
        
        # Calculate circular distances from mean direction
        stats = self.circular_stats.compute_advanced_circular_statistics(angles_rad)
        mean_dir = stats['mean_direction_rad']
        
        # Circular distances
        circular_distances = np.abs(np.arctan2(np.sin(angles_rad - mean_dir), np.cos(angles_rad - mean_dir)))
        
        # Outlier detection using modified z-score for circular data
        median_distance = np.median(circular_distances)
        mad = np.median(np.abs(circular_distances - median_distance))
        
        if mad == 0:
            return {'outlier_indices': [], 'outlier_count': 0, 'no_variability': True}
        
        modified_z_scores = 0.6745 * (circular_distances - median_distance) / mad
        outlier_threshold = 3.5  # Conservative threshold
        
        outlier_indices = np.where(np.abs(modified_z_scores) > outlier_threshold)[0].tolist()
        
        return {
            'outlier_indices': outlier_indices,
            'outlier_count': len(outlier_indices),
            'outlier_proportion': float(len(outlier_indices) / n),
            'circular_distances': circular_distances.tolist(),
            'modified_z_scores': modified_z_scores.tolist(),
            'outlier_threshold': float(outlier_threshold)
        }
    
    # Additional helper methods with simplified implementations
    def _analyze_circular_modality(self, angles_rad: np.ndarray) -> Dict:
        """Analyze modality in circular data"""
        n = len(angles_rad)
        
        if n < 10:
            return {'modality': 'insufficient_data', 'n_modes': 0}
        
        # Simple mode detection using histogram
        bins = 36  # 10-degree bins
        hist, bin_edges = np.histogram(angles_rad, bins=bins, range=(0, 2*np.pi))
        
        # Find local maxima as modes
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=max(hist) * 0.3)
        
        modes = []
        for peak in peaks:
            angle_rad = bin_edges[peak] + (bin_edges[1] - bin_edges[0]) / 2
            modes.append({
                'angle_rad': float(angle_rad),
                'angle_deg': float(np.degrees(angle_rad)),
                'count': int(hist[peak])
            })
        
        return {
            'n_modes': len(modes),
            'modes': modes,
            'modality_classification': self._classify_circular_modality(len(modes))
        }
    
    def _calculate_multiple_dispersion_measures(self, circular_stats: Dict) -> Dict:
        """Calculate multiple dispersion measures"""
        R = circular_stats['mean_resultant_length']
        circular_var = circular_stats['circular_variance']
        
        # Primary dispersion (angular deviation)
        primary_dispersion = np.degrees(np.sqrt(2 * circular_var))
        
        # Alternative dispersion measures
        mean_deviation = np.degrees(np.sqrt(2 * (1 - R)))
        circular_range = 2 * np.degrees(np.arccos(R)) if R > 0 else 180.0
        
        # Concentration-based measures
        if circular_stats['von_mises_kappa'] > 0:
            dispersion_vm = np.degrees(np.sqrt(1 / circular_stats['von_mises_kappa']))
        else:
            dispersion_vm = 180.0
        
        return {
            'primary_dispersion': float(primary_dispersion),
            'mean_angular_deviation': float(mean_deviation),
            'circular_range': float(circular_range),
            'von_mises_dispersion': float(dispersion_vm),
            'relative_dispersion': float(circular_var),
            'dispersion_quality': 'low' if primary_dispersion > 60 else 'moderate' if primary_dispersion > 30 else 'high'
        }
    
    def _assess_circular_analysis_quality(self, n: int, circular_stats: Dict) -> str:
        """Assess quality of circular analysis"""
        R = circular_stats['mean_resultant_length']
        
        if n < 5:
            return 'insufficient'
        elif n < 15:
            return 'limited'
        elif n >= 30 and R > 0.7:
            return 'excellent'
        elif n >= 20 and R > 0.5:
            return 'good'
        elif n >= 10 and R > 0.3:
            return 'moderate'
        else:
            return 'poor'
    
    def _select_best_circular_model(self, angles_rad: np.ndarray, distributions: Dict) -> Dict:
        """Select best circular distribution model using AIC/BIC"""
        model_criteria = {}
        
        for name, params in distributions.items():
            if 'aic' in params:
                model_criteria[name] = params['aic']
            elif 'log_likelihood' in params:
                k = 2  # Assume 2 parameters
                model_criteria[name] = -2 * params['log_likelihood'] + 2 * k
            else:
                model_criteria[name] = np.inf
        
        best_model = min(model_criteria.keys(), key=lambda k: model_criteria[k])
        return {
            'best_model': best_model,
            'aic_values': model_criteria,
            'model_weights': self._calculate_model_weights(model_criteria)
        }
    
    def _calculate_model_weights(self, aic_values: Dict) -> Dict:
        """Calculate Akaike model weights"""
        min_aic = min(aic_values.values())
        delta_aic = {k: v - min_aic for k, v in aic_values.items()}
        
        weights = {}
        total_weight = sum(np.exp(-0.5 * delta) for delta in delta_aic.values())
        
        for k, delta in delta_aic.items():
            weights[k] = np.exp(-0.5 * delta) / total_weight
            
        return weights
    
    def _assess_circular_goodness_of_fit(self, angles_rad: np.ndarray, distributions: Dict) -> Dict:
        """Assess goodness of fit for circular distributions"""
        gof_results = {}
        
        for name, params in distributions.items():
            if name == 'von_mises' and 'mu' in params and 'kappa' in params:
                stats = self.circular_stats.compute_advanced_circular_statistics(angles_rad)
                gof_score = stats['mean_resultant_length']
                
                gof_results[name] = {
                    'goodness_score': float(gof_score),
                    'interpretation': 'excellent' if gof_score > 0.8 else 'good' if gof_score > 0.6 else 'moderate'
                }
            else:
                gof_results[name] = {'goodness_score': 0.5, 'interpretation': 'not_assessed'}
        
        return gof_results
    
    def _interpret_circular_tests(self, rayleigh_p: float, kuiper_p: float, watson_p: float) -> Dict:
        """Interpret results of circular hypothesis tests"""
        alpha = 0.05
        
        interpretations = {
            'rayleigh_interpretation': 'uniform' if rayleigh_p > alpha else 'non_uniform',
            'kuiper_interpretation': 'uniform' if kuiper_p > alpha else 'non_uniform',
            'watson_interpretation': 'uniform' if watson_p > alpha else 'non_uniform',
            'overall_conclusion': 'uniform' if all(p > alpha for p in [rayleigh_p, kuiper_p, watson_p]) else 'non_uniform',
            'strongest_evidence': min([('rayleigh', rayleigh_p), ('kuiper', kuiper_p), ('watson', watson_p)], key=lambda x: x[1])[0]
        }
        
        return interpretations
    
    def _circular_confidence_interval(self, angles: np.ndarray, alpha: float) -> List[float]:
        """Calculate circular confidence interval"""
        stats = self.circular_stats.compute_advanced_circular_statistics(angles)
        mean_dir = stats['mean_direction_rad']
        concentration = stats['von_mises_kappa']
        
        if concentration > 0:
            margin = 1.96 / np.sqrt(len(angles) * concentration)
        else:
            margin = np.pi / 2
        
        return [float((mean_dir - margin) % (2 * np.pi)), float((mean_dir + margin) % (2 * np.pi))]
    
    def _classify_circular_modality(self, n_modes: int) -> str:
        """Classify circular modality"""
        if n_modes <= 1:
            return 'unimodal'
        elif n_modes == 2:
            return 'bimodal'
        elif n_modes <= 4:
            return 'multimodal'
        else:
            return 'complex'


class StabilityAnalysis:
    """Multi-scale stability analysis for geological orientations"""
    
    def __init__(self):
        self.orientation_analyzer = OrientationAnalysis()
    
    def assess_orientation_stability(self, points: np.ndarray, feature_type: str = 'unknown') -> Dict:
        """Advanced multi-scale stability analysis"""
        n_points = len(points)
        if n_points < 6:
            return self._handle_insufficient_stability_data(n_points)
        
        # Multi-scale hierarchical analysis
        hierarchical_analysis = self._perform_hierarchical_stability_analysis(points)
        
        # Statistical stability using resampling
        resampling_stability = self._calculate_resampling_stability(points)
        
        # Spatial coherence analysis
        spatial_coherence = self._analyze_spatial_coherence_stability(points)
        
        # Overall stability score
        stability_components = {
            'hierarchical': hierarchical_analysis.get('stability_score', 0.0),
            'resampling': resampling_stability.get('bootstrap_stability', 0.0),
            'spatial': spatial_coherence.get('coherence_score', 0.0)
        }
        
        # Weighted stability fusion
        weights = {'hierarchical': 0.4, 'resampling': 0.4, 'spatial': 0.2}
        overall_stability = sum(
            stability_components[component] * weights[component]
            for component in stability_components.keys()
        )
        
        return {
            'overall_stability': float(overall_stability),
            'stability_grade': self._grade_stability(overall_stability),
            'hierarchical_analysis': hierarchical_analysis,
            'resampling_stability': resampling_stability,
            'spatial_coherence': spatial_coherence,
            'stability_components': stability_components
        }
    
    def _handle_insufficient_stability_data(self, n_points: int) -> Dict:
        """Handle insufficient data for stability analysis"""
        return {
            'overall_stability': 0.0,
            'stability_grade': {'grade': 'insufficient_data', 'confidence': 'none'},
            'data_points': n_points,
            'recommendation': 'Collect at least 6 data points for stability analysis',
            'insufficient_data': True
        }
    
    def _perform_hierarchical_stability_analysis(self, points: np.ndarray) -> Dict:
        """Perform hierarchical multi-scale stability analysis"""
        n_points = len(points)
        
        # Define hierarchical scales
        scales = []
        for i in range(2, min(6, n_points // 3 + 1)):
            scale_size = n_points // i
            if scale_size >= 3:
                scales.append(scale_size)
        
        if not scales:
            return {'stability_score': 0.0, 'insufficient_scales': True}
        
        # For simplified implementation, use basic variance analysis
        # In a full implementation, this would calculate orientations at each scale
        point_variance = np.var(points, axis=0)
        total_variance = np.sum(point_variance)
        
        # Stability inversely related to variance
        stability_score = 1.0 / (1.0 + total_variance * 0.001)
        
        return {
            'stability_score': float(stability_score),
            'scales_analyzed': scales,
            'point_variance': point_variance.tolist(),
            'hierarchical_consistency': float(stability_score)
        }
    
    def _calculate_resampling_stability(self, points: np.ndarray) -> Dict:
        """Calculate stability using bootstrap resampling"""
        n_points = len(points)
        n_bootstrap = 100  # Reduced for efficiency
        
        bootstrap_stabilities = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_points, n_points, replace=True)
            bootstrap_points = points[bootstrap_indices]
            
            # Calculate some measure of consistency (simplified)
            point_variance = np.var(bootstrap_points, axis=0)
            stability = 1.0 / (1.0 + np.sum(point_variance) * 0.001)
            bootstrap_stabilities.append(stability)
        
        bootstrap_stabilities = np.array(bootstrap_stabilities)
        
        return {
            'bootstrap_stability': float(np.mean(bootstrap_stabilities)),
            'stability_std': float(np.std(bootstrap_stabilities)),
            'stability_ci': [float(np.percentile(bootstrap_stabilities, 2.5)),
                           float(np.percentile(bootstrap_stabilities, 97.5))],
            'bootstrap_samples': n_bootstrap
        }
    
    def _analyze_spatial_coherence_stability(self, points: np.ndarray) -> Dict:
        """Analyze spatial coherence for stability assessment"""
        n_points = len(points)
        
        if n_points < 4:
            return {'coherence_score': 0.0, 'insufficient_data': True}
        
        # Calculate spatial coherence using nearest neighbor distances
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)  # k=2 to exclude self
        nn_distances = distances[:, 1]  # Nearest neighbor distances
        
        # Coherence inversely related to distance variability
        distance_cv = np.std(nn_distances) / np.mean(nn_distances) if np.mean(nn_distances) > 0 else 1.0
        coherence_score = 1.0 / (1.0 + distance_cv)
        
        return {
            'coherence_score': float(coherence_score),
            'mean_nn_distance': float(np.mean(nn_distances)),
            'distance_variability': float(distance_cv),
            'spatial_distribution': 'clustered' if distance_cv < 0.5 else 'dispersed'
        }
    
    def _grade_stability(self, stability_score: float) -> Dict:
        """Grade stability level"""
        if stability_score > 0.8:
            grade = 'excellent'
            confidence = 'high'
        elif stability_score > 0.6:
            grade = 'good'
            confidence = 'moderate'
        elif stability_score > 0.4:
            grade = 'moderate'
            confidence = 'moderate'
        elif stability_score > 0.2:
            grade = 'poor'
            confidence = 'low'
        else:
            grade = 'very_poor'
            confidence = 'very_low'
        
        return {
            'grade': grade,
            'confidence': confidence,
            'score': float(stability_score)
        }


# Utility functions
def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data distribution"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    n = len(data)
    skew = np.mean(((data - mean) / std) ** 3)
    return skew * np.sqrt(n * (n - 1)) / (n - 2) if n > 2 else skew

def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data distribution"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 3.0  # Normal distribution kurtosis
    n = len(data)
    kurt = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    return kurt