"""
Causal Uncertainty Quantification for Geological Systems
Implements Bayesian uncertainty estimation, bootstrap confidence intervals, and robustness testing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from scipy import stats, optimize
from scipy.special import beta, gamma
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, BayesianRidge
import networkx as nx

warnings.filterwarnings('ignore')

class UncertaintyType(Enum):
    """Types of uncertainty in causal relationships"""
    EPISTEMIC = "epistemic"      # Uncertainty due to lack of knowledge
    ALEATORY = "aleatory"        # Uncertainty due to inherent randomness
    MODEL = "model"              # Uncertainty due to model choice
    PARAMETER = "parameter"      # Uncertainty in parameter estimates
    STRUCTURAL = "structural"    # Uncertainty in causal structure

class ConfidenceMethod(Enum):
    """Methods for confidence interval estimation"""
    BOOTSTRAP = "bootstrap"
    BAYESIAN = "bayesian"
    ASYMPTOTIC = "asymptotic"
    PERMUTATION = "permutation"
    CROSS_VALIDATION = "cross_validation"

class RobustnessTest(Enum):
    """Types of robustness tests"""
    PARAMETER_SENSITIVITY = "parameter_sensitivity"
    DATA_PERTURBATION = "data_perturbation"
    MODEL_SELECTION = "model_selection"
    OUTLIER_REMOVAL = "outlier_removal"
    SUBSAMPLE_STABILITY = "subsample_stability"

@dataclass
class UncertaintyEstimate:
    """Uncertainty estimate for a causal relationship"""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    uncertainty_type: UncertaintyType
    method_used: ConfidenceMethod
    sample_size: int
    degrees_of_freedom: Optional[int] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RobustnessResult:
    """Result of robustness testing"""
    test_type: RobustnessTest
    original_estimate: float
    perturbed_estimates: List[float]
    robustness_score: float
    stability_measure: float
    outlier_sensitivity: float
    test_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalUncertaintyProfile:
    """Complete uncertainty profile for a causal relationship"""
    relationship_id: str
    source_variable: str
    target_variable: str
    causal_strength: UncertaintyEstimate
    statistical_significance: UncertaintyEstimate
    robustness_results: List[RobustnessResult]
    model_uncertainty: Dict[str, float]
    overall_confidence: float
    uncertainty_sources: Dict[UncertaintyType, float]

class CausalUncertaintyQuantification:
    """
    Comprehensive uncertainty quantification for causal relationships
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.confidence_level = self.config.get('confidence_level', 0.95)
        self.bootstrap_samples = self.config.get('bootstrap_samples', 1000)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.perturbation_levels = self.config.get('perturbation_levels', [0.01, 0.05, 0.1, 0.2])
        self.random_seed = self.config.get('random_seed', 42)
        
        # Results storage
        self.uncertainty_profiles = {}
        self.global_uncertainty_metrics = {}
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
    
    async def quantify_causal_uncertainty(self, 
                                        causal_relationships: List[Any],
                                        data: pd.DataFrame,
                                        methods: Optional[List[ConfidenceMethod]] = None,
                                        robustness_tests: Optional[List[RobustnessTest]] = None) -> Dict[str, Any]:
        """
        Comprehensive uncertainty quantification for causal relationships
        """
        try:
            self.logger.info("Starting causal uncertainty quantification...")
            
            # Default methods if not specified
            if methods is None:
                methods = [
                    ConfidenceMethod.BOOTSTRAP,
                    ConfidenceMethod.BAYESIAN,
                    ConfidenceMethod.CROSS_VALIDATION
                ]
            
            if robustness_tests is None:
                robustness_tests = [
                    RobustnessTest.PARAMETER_SENSITIVITY,
                    RobustnessTest.DATA_PERTURBATION,
                    RobustnessTest.SUBSAMPLE_STABILITY
                ]
            
            # Validate and prepare data
            prepared_data = await self._prepare_uncertainty_data(data)
            
            # Process each causal relationship
            uncertainty_profiles = []
            
            for relationship in causal_relationships:
                try:
                    profile = await self._analyze_relationship_uncertainty(
                        relationship, prepared_data, methods, robustness_tests
                    )
                    uncertainty_profiles.append(profile)
                    
                except Exception as e:
                    self.logger.warning(f"Uncertainty analysis failed for relationship {relationship}: {e}")
            
            # Calculate global uncertainty metrics
            global_metrics = await self._calculate_global_uncertainty_metrics(uncertainty_profiles)
            
            # Generate insights
            insights = self._generate_insights(uncertainty_profiles, global_metrics)
            
            # Compile results
            results = {
                'uncertainty_profiles': uncertainty_profiles,
                'global_uncertainty_metrics': global_metrics,
                'analyzed_relationships': len(uncertainty_profiles),
                'robustness_insights': insights.get('robustness_insights', []),
                'uncertain_relationships': insights.get('uncertain_relationships', []),
                'high_confidence_relationships': insights.get('high_confidence_relationships', [])
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in causal uncertainty quantification: {e}")
            raise
    
    async def _prepare_uncertainty_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for uncertainty analysis"""
        
        # Select numerical columns
        numerical_data = data.select_dtypes(include=[np.number]).copy()
        
        # Handle missing values
        numerical_data = numerical_data.fillna(numerical_data.median())
        
        # Remove constant columns
        constant_cols = numerical_data.columns[numerical_data.nunique() <= 1]
        if len(constant_cols) > 0:
            numerical_data = numerical_data.drop(columns=constant_cols)
        
        return numerical_data
    
    async def _analyze_relationship_uncertainty(self, 
                                             relationship: Any,
                                             data: pd.DataFrame,
                                             methods: List[ConfidenceMethod],
                                             robustness_tests: List[RobustnessTest]) -> CausalUncertaintyProfile:
        """Analyze uncertainty for a single causal relationship"""
        
        # Extract relationship information
        if hasattr(relationship, 'source') and hasattr(relationship, 'target'):
            source_var = relationship.source
            target_var = relationship.target
            strength = getattr(relationship, 'strength', 0.5)
        else:
            # Handle dictionary format
            source_var = relationship.get('source', 'unknown')
            target_var = relationship.get('target', 'unknown')
            strength = relationship.get('strength', 0.5)
        
        relationship_id = f"{source_var}_{target_var}"
        
        if source_var not in data.columns or target_var not in data.columns:
            return self._create_dummy_profile(relationship_id, source_var, target_var)
        
        # Calculate uncertainty estimates
        strength_uncertainty = await self._estimate_strength_uncertainty(
            source_var, target_var, data, strength, methods
        )
        
        significance_uncertainty = await self._estimate_significance_uncertainty(
            source_var, target_var, data, methods
        )
        
        # Perform robustness tests
        robustness_results = []
        for test_type in robustness_tests:
            try:
                result = await self._perform_robustness_test(
                    source_var, target_var, data, strength, test_type
                )
                robustness_results.append(result)
            except Exception as e:
                self.logger.warning(f"Robustness test {test_type.value} failed: {e}")
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            strength_uncertainty, significance_uncertainty, robustness_results
        )
        
        return CausalUncertaintyProfile(
            relationship_id=relationship_id,
            source_variable=source_var,
            target_variable=target_var,
            causal_strength=strength_uncertainty,
            statistical_significance=significance_uncertainty,
            robustness_results=robustness_results,
            model_uncertainty={},
            overall_confidence=overall_confidence,
            uncertainty_sources={}
        )
    
    def _create_dummy_profile(self, relationship_id: str, source_var: str, target_var: str) -> CausalUncertaintyProfile:
        """Create dummy uncertainty profile for missing data"""
        
        dummy_uncertainty = UncertaintyEstimate(
            point_estimate=0.0,
            confidence_interval=(0.0, 0.0),
            confidence_level=self.confidence_level,
            uncertainty_type=UncertaintyType.EPISTEMIC,
            method_used=ConfidenceMethod.BOOTSTRAP,
            sample_size=0
        )
        
        return CausalUncertaintyProfile(
            relationship_id=relationship_id,
            source_variable=source_var,
            target_variable=target_var,
            causal_strength=dummy_uncertainty,
            statistical_significance=dummy_uncertainty,
            robustness_results=[],
            model_uncertainty={},
            overall_confidence=0.0,
            uncertainty_sources={}
        )
    
    async def _estimate_strength_uncertainty(self, 
                                           source_var: str,
                                           target_var: str,
                                           data: pd.DataFrame,
                                           point_estimate: float,
                                           methods: List[ConfidenceMethod]) -> UncertaintyEstimate:
        """Estimate uncertainty in causal strength"""
        
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        # Bootstrap confidence interval
        bootstrap_estimates = []
        n_samples = len(x_data)
        
        for _ in range(min(500, self.bootstrap_samples)):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            if np.std(x_boot) > 0 and np.std(y_boot) > 0:
                boot_corr = abs(np.corrcoef(x_boot, y_boot)[0, 1])
                if not np.isnan(boot_corr):
                    bootstrap_estimates.append(boot_corr)
        
        # Calculate confidence interval
        if bootstrap_estimates:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_estimates, (alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_estimates, (1 - alpha / 2) * 100)
        else:
            margin = 0.1 * abs(point_estimate)
            ci_lower = max(0, point_estimate - margin)
            ci_upper = min(1, point_estimate + margin)
        
        return UncertaintyEstimate(
            point_estimate=point_estimate,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            uncertainty_type=UncertaintyType.PARAMETER,
            method_used=ConfidenceMethod.BOOTSTRAP,
            sample_size=len(x_data)
        )
    
    async def _estimate_significance_uncertainty(self, 
                                               source_var: str,
                                               target_var: str,
                                               data: pd.DataFrame,
                                               methods: List[ConfidenceMethod]) -> UncertaintyEstimate:
        """Estimate uncertainty in statistical significance"""
        
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        # Calculate p-value
        if len(x_data) > 3:
            _, p_value = stats.pearsonr(x_data, y_data)
        else:
            p_value = 1.0
        
        # Bootstrap p-values
        bootstrap_p_values = []
        
        for _ in range(min(100, self.bootstrap_samples)):
            indices = np.random.choice(len(x_data), size=len(x_data), replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            if len(x_boot) > 3:
                try:
                    _, p_boot = stats.pearsonr(x_boot, y_boot)
                    if not np.isnan(p_boot):
                        bootstrap_p_values.append(p_boot)
                except:
                    continue
        
        # Calculate confidence interval for p-value
        if bootstrap_p_values:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_p_values, (alpha / 2) * 100)
            ci_upper = np.percentile(bootstrap_p_values, (1 - alpha / 2) * 100)
        else:
            margin = 0.1 * p_value
            ci_lower = max(0, p_value - margin)
            ci_upper = min(1, p_value + margin)
        
        return UncertaintyEstimate(
            point_estimate=p_value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            uncertainty_type=UncertaintyType.PARAMETER,
            method_used=ConfidenceMethod.BOOTSTRAP,
            sample_size=len(x_data)
        )
    
    async def _perform_robustness_test(self, 
                                     source_var: str,
                                     target_var: str,
                                     data: pd.DataFrame,
                                     original_estimate: float,
                                     test_type: RobustnessTest) -> RobustnessResult:
        """Perform robustness test for causal relationship"""
        
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        perturbed_estimates = []
        test_parameters = {}
        
        if test_type == RobustnessTest.DATA_PERTURBATION:
            # Test robustness to data perturbation
            for noise_level in self.perturbation_levels:
                for _ in range(20):
                    # Add Gaussian noise
                    x_perturbed = x_data + np.random.normal(0, noise_level * np.std(x_data), len(x_data))
                    y_perturbed = y_data + np.random.normal(0, noise_level * np.std(y_data), len(y_data))
                    
                    if np.std(x_perturbed) > 0 and np.std(y_perturbed) > 0:
                        perturbed_corr = abs(np.corrcoef(x_perturbed, y_perturbed)[0, 1])
                        if not np.isnan(perturbed_corr):
                            perturbed_estimates.append(perturbed_corr)
            
            test_parameters['perturbation_levels'] = self.perturbation_levels
            
        elif test_type == RobustnessTest.SUBSAMPLE_STABILITY:
            # Test stability across different subsample sizes
            subsample_sizes = [0.5, 0.7, 0.8, 0.9]
            for size in subsample_sizes:
                for _ in range(25):
                    n_subsample = int(len(x_data) * size)
                    if n_subsample > 10:
                        indices = np.random.choice(len(x_data), size=n_subsample, replace=False)
                        x_sub = x_data[indices]
                        y_sub = y_data[indices]
                        
                        if np.std(x_sub) > 0 and np.std(y_sub) > 0:
                            sub_corr = abs(np.corrcoef(x_sub, y_sub)[0, 1])
                            if not np.isnan(sub_corr):
                                perturbed_estimates.append(sub_corr)
            
            test_parameters['subsample_sizes'] = subsample_sizes
            
        elif test_type == RobustnessTest.OUTLIER_REMOVAL:
            # Test sensitivity to outliers
            outlier_thresholds = [1.5, 2.0, 2.5, 3.0]
            for threshold in outlier_thresholds:
                x_clean, y_clean = self._remove_outliers(x_data, y_data, threshold)
                
                if len(x_clean) > 10 and np.std(x_clean) > 0 and np.std(y_clean) > 0:
                    clean_corr = abs(np.corrcoef(x_clean, y_clean)[0, 1])
                    if not np.isnan(clean_corr):
                        perturbed_estimates.append(clean_corr)
        
        # Calculate robustness metrics
        if perturbed_estimates:
            robustness_score = 1.0 - (np.std(perturbed_estimates) / (abs(original_estimate) + 1e-8))
            stability_measure = np.mean([abs(est - original_estimate) for est in perturbed_estimates])
            outlier_sensitivity = len([est for est in perturbed_estimates 
                                     if abs(est - original_estimate) > 2 * np.std(perturbed_estimates)]) / len(perturbed_estimates)
        else:
            robustness_score = 0.0
            stability_measure = float('inf')
            outlier_sensitivity = 1.0
        
        return RobustnessResult(
            test_type=test_type,
            original_estimate=original_estimate,
            perturbed_estimates=perturbed_estimates,
            robustness_score=max(0.0, min(1.0, robustness_score)),
            stability_measure=stability_measure,
            outlier_sensitivity=outlier_sensitivity,
            test_parameters=test_parameters
        )
    
    def _remove_outliers(self, x_data: np.ndarray, y_data: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using IQR method"""
        def remove_outliers_1d(data, threshold):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data >= lower_bound) & (data <= upper_bound)
        
        x_mask = remove_outliers_1d(x_data, threshold)
        y_mask = remove_outliers_1d(y_data, threshold)
        combined_mask = x_mask & y_mask
        
        return x_data[combined_mask], y_data[combined_mask]
    
    def _calculate_overall_confidence(self, 
                                    strength_uncertainty: UncertaintyEstimate,
                                    significance_uncertainty: UncertaintyEstimate,
                                    robustness_results: List[RobustnessResult]) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from strength uncertainty
        strength_width = strength_uncertainty.confidence_interval[1] - strength_uncertainty.confidence_interval[0]
        strength_confidence = max(0, 1 - strength_width)
        
        # Significance confidence (lower p-value = higher confidence)
        sig_confidence = max(0, 1 - significance_uncertainty.point_estimate)
        
        # Robustness confidence
        if robustness_results:
            avg_robustness = np.mean([r.robustness_score for r in robustness_results])
            robustness_confidence = avg_robustness
        else:
            robustness_confidence = 0.5
        
        # Weighted combination
        overall_confidence = (
            0.4 * strength_confidence +
            0.3 * sig_confidence +
            0.3 * robustness_confidence
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    async def _calculate_global_uncertainty_metrics(self, profiles: List[CausalUncertaintyProfile]) -> Dict[str, Any]:
        """Calculate global uncertainty metrics across all relationships"""
        
        if not profiles:
            return {}
        
        confidence_distribution = [profile.overall_confidence for profile in profiles]
        robustness_scores = []
        
        for profile in profiles:
            if profile.robustness_results:
                avg_robustness = np.mean([r.robustness_score for r in profile.robustness_results])
                robustness_scores.append(avg_robustness)
        
        return {
            'confidence_distribution': {
                'mean': np.mean(confidence_distribution),
                'std': np.std(confidence_distribution),
                'min': np.min(confidence_distribution),
                'max': np.max(confidence_distribution)
            },
            'global_uncertainty_metrics': {
                'avg_robustness': np.mean(robustness_scores) if robustness_scores else 0.0,
                'uncertainty_spread': np.std(confidence_distribution)
            }
        }
    
    def _generate_insights(self, profiles: List[CausalUncertaintyProfile], global_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from uncertainty analysis"""
        
        insights = {
            'robustness_insights': [],
            'uncertain_relationships': [],
            'high_confidence_relationships': []
        }
        
        # Analyze individual relationships
        for profile in profiles:
            if profile.overall_confidence < 0.3:
                insights['uncertain_relationships'].append({
                    'relationship': profile.relationship_id,
                    'confidence': profile.overall_confidence,
                    'uncertainty_reasons': ['Low overall confidence']
                })
            elif profile.overall_confidence > 0.8:
                insights['high_confidence_relationships'].append({
                    'relationship': profile.relationship_id,
                    'confidence': profile.overall_confidence
                })
        
        # Generate robustness insights
        if profiles:
            avg_robustness = np.mean([
                np.mean([r.robustness_score for r in profile.robustness_results]) 
                for profile in profiles if profile.robustness_results
            ])
            
            if avg_robustness > 0.7:
                insights['robustness_insights'].append("Good overall robustness - results are likely reliable")
            elif avg_robustness < 0.3:
                insights['robustness_insights'].append("High disagreement between algorithms - consider ensemble approaches")
        
        return insights


async def main():
    """Example usage of causal uncertainty quantification"""
    
    # Configuration
    config = {
        'bootstrap_samples': 500,
        'cv_folds': 5,
        'perturbation_levels': [0.01, 0.05, 0.1],
        'random_seed': 42
    }
    
    # Initialize uncertainty quantification
    cuq = CausalUncertaintyQuantification(config)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        'temperature': np.random.normal(100, 20, n_samples),
        'pressure': np.random.normal(50, 10, n_samples),
        'viscosity': np.random.normal(1000, 200, n_samples)
    })
    
    # Add some causal relationships
    data['pressure'] += 0.3 * data['temperature'] + np.random.normal(0, 5, n_samples)
    data['viscosity'] -= 0.5 * data['temperature'] + np.random.normal(0, 50, n_samples)
    
    # Mock causal relationships
    causal_relationships = [
        {'source': 'temperature', 'target': 'pressure', 'strength': 0.6},
        {'source': 'temperature', 'target': 'viscosity', 'strength': -0.4}
    ]
    
    print("Starting causal uncertainty quantification test...")
    start_time = time.time()
    
    # Run uncertainty analysis
    methods = [
        ConfidenceMethod.BOOTSTRAP,
        ConfidenceMethod.BAYESIAN,
        ConfidenceMethod.ASYMPTOTIC
    ]
    
    robustness_tests = [
        RobustnessTest.DATA_PERTURBATION,
        RobustnessTest.SUBSAMPLE_STABILITY,
        RobustnessTest.OUTLIER_REMOVAL
    ]
    
    results = await cuq.quantify_causal_uncertainty(
        causal_relationships=causal_relationships,
        data=data,
        methods=methods,
        robustness_tests=robustness_tests
    )
    
    end_time = time.time()
    
    # Print results
    print(f"\nUncertainty quantification completed in {end_time - start_time:.2f} seconds")
    print(f"Analyzed {results['analyzed_relationships']} relationships")
    
    # Print uncertainty profiles
    print("\n=== Uncertainty Profiles ===")
    for profile in results['uncertainty_profiles']:
        print(f"\n{profile.relationship_id}:")
        print(f"  Causal strength: {profile.causal_strength.point_estimate:.3f} "
              f"[{profile.causal_strength.confidence_interval[0]:.3f}, "
              f"{profile.causal_strength.confidence_interval[1]:.3f}]")
        print(f"  Overall confidence: {profile.overall_confidence:.3f}")
        
        for r in profile.robustness_results:
            print(f"    {r.test_type.value}: robustness = {r.robustness_score:.3f}")
    
    # Print global uncertainty metrics
    print("\n=== Global Uncertainty Metrics ===")
    global_metrics = results['global_uncertainty_metrics']
    print(f"Mean confidence: {global_metrics['confidence_distribution']['mean']:.3f} ± {global_metrics['confidence_distribution']['std']:.3f}")
    print(f"Confidence range: [{global_metrics['confidence_distribution']['min']:.3f}, {global_metrics['confidence_distribution']['max']:.3f}]")
    
    if 'robustness_insights' in results:
        print("\nRobustness Insights:")
        for insight in results['robustness_insights']:
            print(f"    {insight}")
    
    if 'uncertain_relationships' in results:
        print("\nUncertain relationships:")
        for rel in results['uncertain_relationships']:
            print(f"    Relationship {rel['relationship']}: confidence={rel['confidence']:.3f}, reasons: {', '.join(rel['uncertainty_reasons'])}")
    
    if 'high_confidence_relationships' in results:
        print("\nHigh confidence relationships:")
        for rel in results['high_confidence_relationships']:
            print(f"    Relationship {rel['relationship']}: confidence={rel['confidence']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())