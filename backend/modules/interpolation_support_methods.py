"""
Advanced interpolation support methods for physics-informed geological interpolation.
This module contains comprehensive support methods for the enhanced interpolation system.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from scipy import linalg, optimize, stats
from scipy.spatial.distance import cdist


class InterpolationSupportMethods:
    """Support methods for advanced interpolation techniques."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def combine_interpolation_ensemble(self, ensemble_results: Dict[str, Dict], 
                                     method_weights: Dict[str, float], 
                                     grid: np.ndarray, 
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligently combine multiple interpolation results using weighted ensemble."""
        if not ensemble_results:
            return {'values': np.full(len(grid), np.nan), 'quality': 0.0, 'method': 'empty_ensemble'}
        
        # Normalize weights
        total_weight = sum(method_weights.values())
        if total_weight <= 0:
            # Equal weighting if all weights are zero
            normalized_weights = {method: 1.0/len(ensemble_results) for method in ensemble_results.keys()}
        else:
            normalized_weights = {method: weight/total_weight for method, weight in method_weights.items()}
        
        # Adaptive ensemble combination
        combination_method = constraints.get('ensemble_method', 'weighted_average')
        
        if combination_method == 'stacking':
            return self._stacking_combination(ensemble_results, normalized_weights, grid, constraints)
        elif combination_method == 'dynamic_selection':
            return self._dynamic_selection_combination(ensemble_results, normalized_weights, grid, constraints)
        else:
            return self._weighted_average_combination(ensemble_results, normalized_weights, grid, constraints)
    
    def _weighted_average_combination(self, ensemble_results: Dict, weights: Dict, 
                                    grid: np.ndarray, constraints: Dict) -> Dict[str, Any]:
        """Standard weighted average combination."""
        combined_values = np.zeros(len(grid))
        combined_quality = 0.0
        combined_physics = 0.0
        
        for method, result in ensemble_results.items():
            weight = weights.get(method, 0.0)
            combined_values += weight * result['values']
            combined_quality += weight * result['quality']
            combined_physics += weight * result.get('physics_compliance', 1.0)
        
        return {
            'values': combined_values,
            'quality': combined_quality,
            'physics_compliance': combined_physics,
            'method': 'weighted_ensemble',
            'component_methods': list(ensemble_results.keys()),
            'weights': weights
        }
    
    def _stacking_combination(self, ensemble_results: Dict, weights: Dict, 
                            grid: np.ndarray, constraints: Dict) -> Dict[str, Any]:
        """Meta-learning stacking combination."""
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score
            
            # Prepare stacking features (predictions from base methods)
            n_methods = len(ensemble_results)
            n_grid = len(grid)
            
            if n_methods < 2:
                return self._weighted_average_combination(ensemble_results, weights, grid, constraints)
            
            # Create feature matrix
            X_stack = np.zeros((n_grid, n_methods))
            method_names = list(ensemble_results.keys())
            
            for i, method in enumerate(method_names):
                X_stack[:, i] = ensemble_results[method]['values']
            
            # Use cross-validation scores as target weights for meta-learner
            cv_scores = np.array([result['quality'] for result in ensemble_results.values()])
            meta_weights = cv_scores / np.sum(cv_scores)
            
            # Simple linear combination with physics-informed constraints
            combined_values = X_stack @ meta_weights
            
            # Quality is weighted average of component qualities
            combined_quality = np.sum([weights[method] * result['quality'] 
                                     for method, result in ensemble_results.items()])
            
            combined_physics = np.sum([weights[method] * result.get('physics_compliance', 1.0) 
                                     for method, result in ensemble_results.items()])
            
            return {
                'values': combined_values,
                'quality': combined_quality,
                'physics_compliance': combined_physics,
                'method': 'stacking_ensemble',
                'meta_weights': meta_weights,
                'component_methods': method_names
            }
            
        except ImportError:
            self.logger.warning("Sklearn not available, falling back to weighted average")
            return self._weighted_average_combination(ensemble_results, weights, grid, constraints)
        except Exception as e:
            self.logger.warning(f"Stacking combination failed: {str(e)}")
            return self._weighted_average_combination(ensemble_results, weights, grid, constraints)
    
    def _dynamic_selection_combination(self, ensemble_results: Dict, weights: Dict, 
                                     grid: np.ndarray, constraints: Dict) -> Dict[str, Any]:
        """Dynamic selection based on local performance."""
        try:
            # For each grid point, select best performing method based on local criteria
            n_grid = len(grid)
            combined_values = np.zeros(n_grid)
            selected_methods = []
            
            for i in range(n_grid):
                # Local selection criteria (could be enhanced with more sophisticated metrics)
                best_method = None
                best_score = -np.inf
                
                for method, result in ensemble_results.items():
                    # Combine quality, physics compliance, and weight
                    local_score = (result['quality'] * weights.get(method, 0.0) * 
                                 result.get('physics_compliance', 1.0))
                    
                    if local_score > best_score:
                        best_score = local_score
                        best_method = method
                
                if best_method:
                    combined_values[i] = ensemble_results[best_method]['values'][i]
                    selected_methods.append(best_method)
                else:
                    combined_values[i] = np.nan
                    selected_methods.append('none')
            
            # Overall quality based on selected methods
            method_counts = {}
            for method in selected_methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            combined_quality = np.mean([ensemble_results[method]['quality'] 
                                      for method in set(selected_methods) if method in ensemble_results])
            
            combined_physics = np.mean([ensemble_results[method].get('physics_compliance', 1.0) 
                                      for method in set(selected_methods) if method in ensemble_results])
            
            return {
                'values': combined_values,
                'quality': combined_quality if not np.isnan(combined_quality) else 0.0,
                'physics_compliance': combined_physics if not np.isnan(combined_physics) else 0.0,
                'method': 'dynamic_selection_ensemble',
                'selected_methods': selected_methods,
                'method_usage': method_counts
            }
            
        except Exception as e:
            self.logger.warning(f"Dynamic selection failed: {str(e)}")
            return self._weighted_average_combination(ensemble_results, weights, grid, constraints)
    
    def apply_physics_constraints(self, result: Dict[str, Any], timestamps: np.ndarray, 
                                values: np.ndarray, grid: np.ndarray, 
                                constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physics-informed post-processing constraints."""
        interp_values = result['values'].copy()
        
        # Energy conservation constraint
        if constraints.get('conserve_energy', False):
            interp_values = self._apply_energy_conservation(interp_values, values, constraints)
        
        # Monotonicity constraint
        if constraints.get('enforce_monotonicity', False):
            interp_values = self._enforce_monotonicity_constraint(interp_values, constraints)
        
        # Boundary condition constraints
        if 'boundary_conditions' in constraints:
            interp_values = self._apply_boundary_constraints(interp_values, grid, constraints)
        
        # Smoothness constraint
        if constraints.get('enforce_smoothness', False):
            interp_values = self._apply_smoothness_constraint(interp_values, grid, constraints)
        
        # Value range constraint
        if 'value_bounds' in constraints:
            bounds = constraints['value_bounds']
            interp_values = np.clip(interp_values, bounds[0], bounds[1])
        
        # Update physics compliance score
        physics_compliance = self.assess_physics_compliance(interp_values, grid, constraints)
        
        result['values'] = interp_values
        result['physics_compliance'] = physics_compliance
        
        return result
    
    def quantify_interpolation_uncertainty(self, ensemble_results: Dict, weights: Dict, 
                                         final_values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Quantify interpolation uncertainty using ensemble statistics."""
        if len(ensemble_results) < 2:
            return np.full(len(grid), 0.1)  # Default uncertainty
        
        # Collect all predictions
        predictions = np.array([result['values'] for result in ensemble_results.values()])
        
        # Weighted ensemble variance
        weighted_mean = np.zeros(len(grid))
        total_weight = sum(weights.values())
        
        for i, (method, result) in enumerate(ensemble_results.items()):
            weight = weights.get(method, 0.0) / total_weight
            weighted_mean += weight * predictions[i]
        
        # Compute weighted variance
        weighted_variance = np.zeros(len(grid))
        for i, (method, result) in enumerate(ensemble_results.items()):
            weight = weights.get(method, 0.0) / total_weight
            weighted_variance += weight * (predictions[i] - weighted_mean)**2
        
        # Add model uncertainty from individual methods
        method_uncertainties = []
        for method, result in ensemble_results.items():
            if 'variance' in result:
                method_uncertainties.append(result['variance'])
            elif 'uncertainty' in result:
                method_uncertainties.append(result['uncertainty'])
        
        if method_uncertainties:
            avg_method_uncertainty = np.mean(method_uncertainties, axis=0)
            total_uncertainty = np.sqrt(weighted_variance + avg_method_uncertainty)
        else:
            total_uncertainty = np.sqrt(weighted_variance)
        
        # Ensure minimum uncertainty
        total_uncertainty = np.maximum(total_uncertainty, 1e-6)
        
        return total_uncertainty
    
    def assess_physics_compliance(self, values: np.ndarray, grid: np.ndarray, 
                                constraints: Dict[str, Any]) -> float:
        """Assess compliance with physics-based constraints."""
        compliance_scores = []
        
        # Smoothness compliance (geological processes are generally smooth)
        smoothness_score = self._assess_smoothness_compliance(values, constraints)
        compliance_scores.append(smoothness_score)
        
        # Energy conservation compliance
        if constraints.get('conserve_energy', False):
            energy_score = self._assess_energy_conservation_compliance(values, constraints)
            compliance_scores.append(energy_score)
        
        # Causality compliance (no unphysical correlations)
        causality_score = self._assess_causality_compliance(values, grid, constraints)
        compliance_scores.append(causality_score)
        
        # Scale invariance compliance
        scale_score = self._assess_scale_compliance(values, grid, constraints)
        compliance_scores.append(scale_score)
        
        # Geological realism compliance
        realism_score = self._assess_geological_realism(values, grid, constraints)
        compliance_scores.append(realism_score)
        
        # Weighted average of compliance scores
        if compliance_scores:
            return float(np.mean(compliance_scores))
        else:
            return 1.0
    
    def fallback_interpolation(self, timestamps: np.ndarray, values: np.ndarray, 
                             grid: np.ndarray) -> Dict[str, Any]:
        """Robust fallback interpolation method."""
        try:
            if len(timestamps) < 2:
                return {
                    'values': np.full(len(grid), np.mean(values) if len(values) > 0 else 0.0),
                    'quality': 0.1,
                    'method': 'constant_fallback',
                    'uncertainty': np.full(len(grid), 1.0),
                    'physics_compliance': 0.5
                }
            
            # Use simple linear interpolation with extrapolation
            interp_values = np.interp(grid, timestamps, values)
            
            # Assess quality based on data coverage
            grid_range = np.ptp(grid)
            data_range = np.ptp(timestamps)
            coverage_ratio = min(1.0, data_range / (grid_range + 1e-10))
            quality = 0.3 + 0.4 * coverage_ratio
            
            # Estimate uncertainty based on local data density
            uncertainty = self._estimate_fallback_uncertainty(timestamps, values, grid)
            
            return {
                'values': interp_values,
                'quality': quality,
                'method': 'linear_fallback',
                'uncertainty': uncertainty,
                'physics_compliance': 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Fallback interpolation failed: {str(e)}")
            return {
                'values': np.full(len(grid), 0.0),
                'quality': 0.0,
                'method': 'zero_fallback',
                'uncertainty': np.full(len(grid), 1.0),
                'physics_compliance': 0.0
            }
    
    # Private methods for constraint enforcement and assessment
    
    def _apply_energy_conservation(self, values: np.ndarray, original_values: np.ndarray, 
                                 constraints: Dict) -> np.ndarray:
        """Apply energy conservation constraint."""
        if len(original_values) == 0:
            return values
        
        original_energy = np.sum(original_values**2)
        current_energy = np.sum(values**2)
        
        if current_energy > 1e-12:
            scaling_factor = np.sqrt(original_energy / current_energy)
            conservation_strength = constraints.get('energy_conservation_strength', 0.5)
            corrected_values = values * (1 - conservation_strength + conservation_strength * scaling_factor)
            return corrected_values
        
        return values
    
    def _enforce_monotonicity_constraint(self, values: np.ndarray, constraints: Dict) -> np.ndarray:
        """Enforce monotonicity constraint."""
        monotonicity_type = constraints.get('monotonicity_type', 'increasing')
        
        if monotonicity_type == 'increasing':
            return np.maximum.accumulate(values)
        elif monotonicity_type == 'decreasing':
            return np.minimum.accumulate(values[::-1])[::-1]
        else:
            return values
    
    def _apply_boundary_constraints(self, values: np.ndarray, grid: np.ndarray, 
                                  constraints: Dict) -> np.ndarray:
        """Apply boundary condition constraints."""
        boundary_conditions = constraints['boundary_conditions']
        corrected_values = values.copy()
        
        if 'left_boundary' in boundary_conditions:
            corrected_values[0] = boundary_conditions['left_boundary']
        
        if 'right_boundary' in boundary_conditions:
            corrected_values[-1] = boundary_conditions['right_boundary']
        
        # Smooth transition from boundaries
        if 'smooth_boundaries' in boundary_conditions and boundary_conditions['smooth_boundaries']:
            transition_length = min(len(values) // 10, 5)
            if transition_length > 0:
                # Left boundary smoothing
                if 'left_boundary' in boundary_conditions:
                    for i in range(1, min(transition_length + 1, len(values))):
                        alpha = i / transition_length
                        corrected_values[i] = (1 - alpha) * boundary_conditions['left_boundary'] + alpha * values[i]
                
                # Right boundary smoothing
                if 'right_boundary' in boundary_conditions:
                    for i in range(max(0, len(values) - transition_length - 1), len(values) - 1):
                        alpha = (len(values) - 1 - i) / transition_length
                        corrected_values[i] = (1 - alpha) * boundary_conditions['right_boundary'] + alpha * values[i]
        
        return corrected_values
    
    def _apply_smoothness_constraint(self, values: np.ndarray, grid: np.ndarray, 
                                   constraints: Dict) -> np.ndarray:
        """Apply smoothness constraint using regularization."""
        if len(values) < 3:
            return values
        
        smoothness_strength = constraints.get('smoothness_strength', 0.1)
        
        # Second-order smoothness penalty
        n = len(values)
        L = np.zeros((n-2, n))
        for i in range(n-2):
            L[i, i:i+3] = [1, -2, 1]
        
        # Regularized solution
        try:
            from scipy.linalg import solve
            A = np.eye(n) + smoothness_strength * L.T @ L
            corrected_values = solve(A, values)
            return corrected_values
        except:
            return values
    
    def _assess_smoothness_compliance(self, values: np.ndarray, constraints: Dict) -> float:
        """Assess smoothness compliance."""
        if len(values) < 3:
            return 1.0
        
        # Measure second derivative magnitude
        second_derivatives = np.abs(np.diff(values, 2))
        smoothness_threshold = constraints.get('smoothness_threshold', np.std(values) * 0.1)
        
        smoothness_violations = np.sum(second_derivatives > smoothness_threshold)
        compliance = max(0.0, 1.0 - smoothness_violations / len(second_derivatives))
        
        return compliance
    
    def _assess_energy_conservation_compliance(self, values: np.ndarray, constraints: Dict) -> float:
        """Assess energy conservation compliance."""
        expected_energy = constraints.get('expected_energy', np.sum(values**2))
        current_energy = np.sum(values**2)
        
        if expected_energy > 1e-12:
            energy_ratio = current_energy / expected_energy
            compliance = 1.0 - abs(1.0 - energy_ratio)
            return max(0.0, compliance)
        
        return 1.0
    
    def _assess_causality_compliance(self, values: np.ndarray, grid: np.ndarray, constraints: Dict) -> float:
        """Assess causality compliance (no unphysical temporal correlations)."""
        if len(values) < 4:
            return 1.0
        
        # Check for non-causal oscillations or correlations
        autocorr = np.correlate(values - np.mean(values), values - np.mean(values), mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
        
        # Penalize high-frequency oscillations that might be unphysical
        max_allowed_correlation_lag = constraints.get('max_correlation_lag', len(values) // 4)
        if len(autocorr) > max_allowed_correlation_lag:
            distant_correlations = np.abs(autocorr[max_allowed_correlation_lag:])
            causality_violation = np.mean(distant_correlations)
            compliance = max(0.0, 1.0 - causality_violation * 2)
        else:
            compliance = 1.0
        
        return compliance
    
    def _assess_scale_compliance(self, values: np.ndarray, grid: np.ndarray, constraints: Dict) -> float:
        """Assess scale invariance compliance."""
        if len(values) < 4:
            return 1.0
        
        # Check if interpolation respects natural scaling
        expected_scale = constraints.get('characteristic_scale', np.ptp(grid) / 10)
        
        # Measure characteristic variation scale
        gradients = np.abs(np.diff(values))
        if len(gradients) > 0:
            variation_scale = np.median(np.diff(grid)) / (np.median(gradients) + 1e-12)
            scale_ratio = variation_scale / expected_scale
            compliance = 1.0 / (1.0 + abs(np.log(scale_ratio + 1e-12)))
        else:
            compliance = 1.0
        
        return compliance
    
    def _assess_geological_realism(self, values: np.ndarray, grid: np.ndarray, constraints: Dict) -> float:
        """Assess geological realism of interpolated values."""
        realism_score = 1.0
        
        # Check for geological value ranges
        if 'geological_bounds' in constraints:
            bounds = constraints['geological_bounds']
            out_of_bounds = np.sum((values < bounds[0]) | (values > bounds[1]))
            realism_score *= max(0.0, 1.0 - out_of_bounds / len(values))
        
        # Check for geological trend consistency
        if 'expected_trend' in constraints:
            expected_trend = constraints['expected_trend']
            if expected_trend == 'increasing':
                trend_compliance = np.sum(np.diff(values) >= 0) / (len(values) - 1) if len(values) > 1 else 1.0
            elif expected_trend == 'decreasing':
                trend_compliance = np.sum(np.diff(values) <= 0) / (len(values) - 1) if len(values) > 1 else 1.0
            else:
                trend_compliance = 1.0
            
            realism_score *= trend_compliance
        
        return realism_score
    
    def _estimate_fallback_uncertainty(self, timestamps: np.ndarray, values: np.ndarray, 
                                     grid: np.ndarray) -> np.ndarray:
        """Estimate uncertainty for fallback interpolation."""
        uncertainty = np.ones(len(grid))
        
        if len(timestamps) < 2:
            return uncertainty
        
        # Distance-based uncertainty
        for i, t in enumerate(grid):
            min_distance = np.min(np.abs(timestamps - t))
            max_distance = np.ptp(timestamps)
            
            if max_distance > 0:
                distance_factor = min_distance / max_distance
                uncertainty[i] = 0.1 + 0.9 * distance_factor
        
        return uncertainty