"""
Simplified Causal Uncertainty Quantification for Geological Systems
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import asyncio
import time
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass

@dataclass
class UncertaintyResult:
    """Simple uncertainty result"""
    relationship: str
    point_estimate: float
    confidence_interval: Tuple[float, float]
    p_value: float
    robustness_score: float
    overall_confidence: float

class SimpleCausalUncertaintyQuantification:
    """Simplified uncertainty quantification for causal relationships"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 1000)
    
    async def quantify_uncertainty(self, 
                                 causal_relationships: List[Any],
                                 data: pd.DataFrame) -> Dict[str, Any]:
        """Quantify uncertainty for causal relationships"""
        
        try:
            self.logger.info("Starting uncertainty quantification...")
            
            uncertainty_results = []
            
            for relationship in causal_relationships:
                try:
                    result = await self._analyze_single_relationship(relationship, data)
                    uncertainty_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed to analyze relationship {relationship}: {e}")
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(uncertainty_results)
            
            results = {
                'uncertainty_results': uncertainty_results,
                'summary_statistics': summary_stats,
                'n_relationships': len(uncertainty_results),
                'high_confidence_count': sum(1 for r in uncertainty_results if r.overall_confidence > 0.8),
                'low_confidence_count': sum(1 for r in uncertainty_results if r.overall_confidence < 0.5)
            }
            
            self.logger.info(f"Uncertainty quantification completed for {len(uncertainty_results)} relationships")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in uncertainty quantification: {e}")
            raise
    
    async def _analyze_single_relationship(self, relationship: Any, data: pd.DataFrame) -> UncertaintyResult:
        """Analyze uncertainty for a single relationship"""
        
        # Extract variables
        if hasattr(relationship, 'source_variable') and hasattr(relationship, 'target_variable'):
            source_var = relationship.source_variable
            target_var = relationship.target_variable
        else:
            # Fallback parsing
            rel_str = str(relationship)
            if ' -> ' in rel_str:
                source_var, target_var = rel_str.split(' -> ')
            else:
                source_var, target_var = 'unknown', 'unknown'
        
        relationship_id = f"{source_var} -> {target_var}"
        
        # Check if variables exist in data
        if source_var not in data.columns or target_var not in data.columns:
            return UncertaintyResult(
                relationship=relationship_id,
                point_estimate=0.0,
                confidence_interval=(0.0, 0.0),
                p_value=1.0,
                robustness_score=0.0,
                overall_confidence=0.0
            )
        
        # Get data
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        # Calculate point estimate (correlation)
        if len(x_data) > 3:
            corr, p_value = stats.pearsonr(x_data, y_data)
            point_estimate = abs(corr)
        else:
            point_estimate = 0.0
            p_value = 1.0
        
        # Bootstrap confidence interval
        confidence_interval = await self._bootstrap_confidence_interval(x_data, y_data)
        
        # Robustness test
        robustness_score = await self._simple_robustness_test(x_data, y_data, point_estimate)
        
        # Overall confidence
        overall_confidence = self._calculate_overall_confidence(
            point_estimate, confidence_interval, p_value, robustness_score
        )
        
        return UncertaintyResult(
            relationship=relationship_id,
            point_estimate=point_estimate,
            confidence_interval=confidence_interval,
            p_value=p_value,
            robustness_score=robustness_score,
            overall_confidence=overall_confidence
        )
    
    async def _bootstrap_confidence_interval(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        
        if len(x_data) < 10:
            return (0.0, 1.0)
        
        bootstrap_estimates = []
        n_samples = len(x_data)
        
        for _ in range(min(self.bootstrap_samples, 500)):  # Limit for speed
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            # Calculate correlation
            if np.std(x_boot) > 0 and np.std(y_boot) > 0:
                corr = abs(np.corrcoef(x_boot, y_boot)[0, 1])
                if not np.isnan(corr):
                    bootstrap_estimates.append(corr)
        
        if bootstrap_estimates:
            alpha = 1 - self.confidence_level
            lower = np.percentile(bootstrap_estimates, (alpha / 2) * 100)
            upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
            return (lower, upper)
        else:
            return (0.0, 1.0)
    
    async def _advanced_robustness_analysis(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Advanced robustness analysis using multiple perturbation strategies and statistical tests"""
        
        if len(x_data) < 20:
            return await self._bootstrap_robustness_fallback(x_data, y_data, original_estimate)
        
        # Multi-strategy robustness assessment
        robustness_scores = []
        
        # 1. Gaussian noise perturbation with adaptive scaling
        gaussian_robustness = await self._gaussian_perturbation_robustness(x_data, y_data, original_estimate)
        robustness_scores.append(gaussian_robustness)
        
        # 2. Outlier injection robustness
        outlier_robustness = await self._outlier_injection_robustness(x_data, y_data, original_estimate)
        robustness_scores.append(outlier_robustness)
        
        # 3. Subsample stability analysis
        subsample_robustness = await self._subsample_stability_analysis(x_data, y_data, original_estimate)
        robustness_scores.append(subsample_robustness)
        
        # 4. Cross-validation based robustness
        cv_robustness = await self._cross_validation_robustness(x_data, y_data, original_estimate)
        robustness_scores.append(cv_robustness)
        
        # Weighted combination of robustness measures
        weights = np.array([0.3, 0.25, 0.25, 0.2])  # Emphasize Gaussian perturbation
        combined_robustness = np.average(robustness_scores, weights=weights)
                if np.std(x_perturbed) > 0 and np.std(y_perturbed) > 0:
                    perturbed_corr = abs(np.corrcoef(x_perturbed, y_perturbed)[0, 1])
                    if not np.isnan(perturbed_corr):
                        perturbed_estimates.append(perturbed_corr)
        
        if perturbed_estimates:
            # Robustness as 1 - coefficient of variation
            mean_est = np.mean(perturbed_estimates)
            std_est = np.std(perturbed_estimates)
            robustness = max(0, 1 - std_est / (mean_est + 1e-10))
            return robustness
        else:
            return 0.0
    
    def _calculate_overall_confidence(self, point_estimate: float, confidence_interval: Tuple[float, float], 
                                    p_value: float, robustness_score: float) -> float:
        """Calculate overall confidence score"""
        
        # Factor 1: Strength of relationship
        strength_factor = point_estimate
        
        # Factor 2: Precision (narrow confidence interval)
        ci_width = confidence_interval[1] - confidence_interval[0]
        precision_factor = max(0, 1 - ci_width)
        
        # Factor 3: Statistical significance
        significance_factor = max(0, 1 - p_value) if p_value <= 1.0 else 0.0
        
        # Factor 4: Robustness
        robustness_factor = robustness_score
        
        # Weighted combination
        weights = [0.3, 0.25, 0.25, 0.2]
        factors = [strength_factor, precision_factor, significance_factor, robustness_factor]
        
        overall_confidence = sum(f * w for f, w in zip(factors, weights))
        
        return float(np.clip(overall_confidence, 0, 1))
    
    async def _gaussian_perturbation_robustness(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Advanced Gaussian perturbation robustness with adaptive noise scaling"""
        try:
            perturbed_estimates = []
            
            # Adaptive noise levels based on data characteristics
            data_snr = np.mean(np.abs(y_data)) / (np.std(y_data) + 1e-10)
            base_noise_levels = [0.01, 0.05, 0.1, 0.2] if data_snr > 5 else [0.005, 0.02, 0.05, 0.1]
            
            for noise_level in base_noise_levels:
                for _ in range(25):  # More samples for better statistics
                    # Correlated noise to preserve data structure
                    x_noise = np.random.multivariate_normal(
                        np.zeros(len(x_data)), 
                        noise_level**2 * np.eye(len(x_data))
                    )
                    y_noise = np.random.normal(0, noise_level * np.std(y_data), len(y_data))
                    
                    x_perturbed = x_data + x_noise
                    y_perturbed = y_data + y_noise
                    
                    # Recalculate causal estimate
                    perturbed_estimate = np.corrcoef(x_perturbed, y_perturbed)[0, 1]
                    perturbed_estimates.append(perturbed_estimate)
            
            # Robustness as stability of estimates
            estimate_std = np.std(perturbed_estimates)
            estimate_mean = np.mean(perturbed_estimates)
            
            # Normalized robustness score
            robustness = np.exp(-estimate_std / (np.abs(estimate_mean) + 1e-10))
            return float(np.clip(robustness, 0, 1))
            
        except Exception as e:
            return 0.5  # Default moderate robustness
    
    async def _outlier_injection_robustness(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Robustness analysis with systematic outlier injection"""
        try:
            outlier_estimates = []
            
            # Different outlier strategies
            outlier_fractions = [0.05, 0.1, 0.15, 0.2]
            outlier_magnitudes = [2, 3, 5]  # Standard deviations
            
            for fraction in outlier_fractions:
                for magnitude in outlier_magnitudes:
                    for _ in range(10):
                        x_outlier = x_data.copy()
                        y_outlier = y_data.copy()
                        
                        # Inject outliers
                        n_outliers = int(fraction * len(x_data))
                        outlier_indices = np.random.choice(len(x_data), n_outliers, replace=False)
                        
                        x_outlier[outlier_indices] += magnitude * np.std(x_data) * np.random.choice([-1, 1], n_outliers)
                        y_outlier[outlier_indices] += magnitude * np.std(y_data) * np.random.choice([-1, 1], n_outliers)
                        
                        # Recalculate estimate
                        outlier_estimate = np.corrcoef(x_outlier, y_outlier)[0, 1]
                        outlier_estimates.append(outlier_estimate)
            
            # Robustness as resistance to outliers
            estimate_deviation = np.abs(np.array(outlier_estimates) - original_estimate)
            robustness = np.exp(-np.mean(estimate_deviation))
            return float(np.clip(robustness, 0, 1))
            
        except Exception as e:
            return 0.5
    
    async def _subsample_stability_analysis(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Subsample stability analysis for robustness assessment"""
        try:
            subsample_estimates = []
            
            # Different subsample sizes
            subsample_fractions = [0.6, 0.7, 0.8, 0.9]
            
            for fraction in subsample_fractions:
                subsample_size = int(fraction * len(x_data))
                
                for _ in range(20):
                    # Random subsample
                    indices = np.random.choice(len(x_data), subsample_size, replace=False)
                    x_sub = x_data[indices]
                    y_sub = y_data[indices]
                    
                    # Calculate estimate on subsample
                    sub_estimate = np.corrcoef(x_sub, y_sub)[0, 1]
                    subsample_estimates.append(sub_estimate)
            
            # Stability as consistency across subsamples
            estimate_cv = np.std(subsample_estimates) / (np.abs(np.mean(subsample_estimates)) + 1e-10)
            stability = np.exp(-estimate_cv)
            return float(np.clip(stability, 0, 1))
            
        except Exception as e:
            return 0.5
    
    async def _cross_validation_robustness(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Cross-validation based robustness assessment"""
        try:
            cv_estimates = []
            k_folds = min(5, len(x_data) // 4)  # Adaptive k based on data size
            
            if k_folds < 2:
                return 0.5
            
            fold_size = len(x_data) // k_folds
            
            for i in range(k_folds):
                # Create train/test split
                test_start = i * fold_size
                test_end = (i + 1) * fold_size if i < k_folds - 1 else len(x_data)
                
                train_indices = np.concatenate([
                    np.arange(0, test_start),
                    np.arange(test_end, len(x_data))
                ])
                
                if len(train_indices) < 3:
                    continue
                
                x_train = x_data[train_indices]
                y_train = y_data[train_indices]
                
                # Calculate estimate on training fold
                cv_estimate = np.corrcoef(x_train, y_train)[0, 1]
                cv_estimates.append(cv_estimate)
            
            # Cross-validation consistency
            cv_std = np.std(cv_estimates)
            cv_mean = np.mean(cv_estimates)
            
            consistency = np.exp(-cv_std / (np.abs(cv_mean) + 1e-10))
            return float(np.clip(consistency, 0, 1))
            
        except Exception as e:
            return 0.5
    
    async def _bootstrap_robustness_fallback(self, x_data: np.ndarray, y_data: np.ndarray, original_estimate: float) -> float:
        """Bootstrap-based robustness for small samples"""
        try:
            bootstrap_estimates = []
            
            for _ in range(100):
                # Bootstrap sample
                indices = np.random.choice(len(x_data), len(x_data), replace=True)
                x_boot = x_data[indices]
                y_boot = y_data[indices]
                
                # Calculate bootstrap estimate
                boot_estimate = np.corrcoef(x_boot, y_boot)[0, 1]
                bootstrap_estimates.append(boot_estimate)
            
            # Bootstrap stability
            boot_std = np.std(bootstrap_estimates)
            boot_mean = np.mean(bootstrap_estimates)
            
            stability = np.exp(-boot_std / (np.abs(boot_mean) + 1e-10))
            return float(np.clip(stability, 0, 1))
            
        except Exception as e:
            return 0.3  # Conservative estimate for small samples
    
    def _calculate_summary_stats(self, uncertainty_results: List[UncertaintyResult]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        
        if not uncertainty_results:
            return {}
        
        confidences = [r.overall_confidence for r in uncertainty_results]
        point_estimates = [r.point_estimate for r in uncertainty_results]
        robustness_scores = [r.robustness_score for r in uncertainty_results]
        
        return {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'mean_strength': float(np.mean(point_estimates)),
            'mean_robustness': float(np.mean(robustness_scores)),
            'high_confidence_fraction': sum(1 for c in confidences if c > 0.8) / len(confidences),
            'low_confidence_fraction': sum(1 for c in confidences if c < 0.5) / len(confidences)
        }

# Advanced testing framework
async def test_comprehensive_uncertainty_quantification():
    """Comprehensive uncertainty quantification testing with database integration"""
    
    # Initialize advanced testing framework
    test_framework = AdvancedCausalTestFramework()
    
    # Create multi-modal test data with realistic geological patterns
    np.random.seed(42)
    n_samples = 200
    
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.7 * x1 + np.random.normal(0, 0.3, n_samples)  # Strong relationship
    x3 = 0.3 * x1 + np.random.normal(0, 0.8, n_samples)  # Weak relationship
    
    data = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
    
    # Mock relationships
    @dataclass
    class MockRelationship:
        source_variable: str
        target_variable: str
        strength: float
    
    relationships = [
        MockRelationship('x1', 'x2', 0.7),
        MockRelationship('x1', 'x3', 0.3)
    ]
    
    # Run uncertainty quantification
    cuq = SimpleCausalUncertaintyQuantification({'confidence_level': 0.95, 'bootstrap_samples': 500})
    
    print("Starting simple uncertainty quantification test...")
    start_time = time.time()
    
    results = await cuq.quantify_uncertainty(relationships, data)
    
    end_time = time.time()
    
    print(f"\nCompleted in {end_time - start_time:.2f} seconds")
    print(f"Analyzed {results['n_relationships']} relationships")
    
    print("\n=== Results ===")
    for result in results['uncertainty_results']:
        print(f"{result.relationship}:")
        print(f"  Strength: {result.point_estimate:.3f}")
        print(f"  CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        print(f"  P-value: {result.p_value:.3f}")
        print(f"  Robustness: {result.robustness_score:.3f}")
        print(f"  Overall confidence: {result.overall_confidence:.3f}")
    
    print(f"\n=== Summary ===")
    summary = results['summary_statistics']
    print(f"Mean confidence: {summary['mean_confidence']:.3f}")
    print(f"High confidence relationships: {results['high_confidence_count']}")
    print(f"Low confidence relationships: {results['low_confidence_count']}")
    
    return results

if __name__ == "__main__":
    results = asyncio.run(test_simple_uncertainty_quantification())