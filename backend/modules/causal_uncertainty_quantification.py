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
from sklearn.model_selection import cross_val_score, KFold, BootstrapSample
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
        
        Args:
            causal_relationships: List of discovered causal relationships
            data: Original data used for causal discovery
            methods: List of uncertainty quantification methods to use
            robustness_tests: List of robustness tests to perform
            
        Returns:
            Dictionary containing uncertainty analysis results
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
                    
                    relationship_id = f"{relationship.source_variable}_{relationship.target_variable}"
                    self.uncertainty_profiles[relationship_id] = profile
                    
                except Exception as e:
                    self.logger.warning(f"Uncertainty analysis failed for relationship {relationship}: {e}")
            
            # Global uncertainty analysis
            global_metrics = await self._analyze_global_uncertainty(uncertainty_profiles, prepared_data)
            
            # Model uncertainty analysis
            model_uncertainty = await self._analyze_model_uncertainty(causal_relationships, prepared_data)
            
            # Uncertainty propagation analysis
            propagation_analysis = await self._analyze_uncertainty_propagation(uncertainty_profiles)
            
            # Generate uncertainty insights
            insights = await self._generate_uncertainty_insights(
                uncertainty_profiles, global_metrics, model_uncertainty
            )
            
            # Compile final results
            results = {
                'uncertainty_profiles': uncertainty_profiles,
                'global_uncertainty_metrics': global_metrics,
                'model_uncertainty': model_uncertainty,
                'uncertainty_propagation': propagation_analysis,
                'uncertainty_insights': insights,
                'analysis_metadata': {
                    'n_relationships_analyzed': len(uncertainty_profiles),
                    'methods_used': [m.value for m in methods],
                    'robustness_tests_used': [r.value for r in robustness_tests],
                    'confidence_level': self.confidence_level,
                    'bootstrap_samples': self.bootstrap_samples,
                    'analysis_timestamp': time.time()
                }
            }
            
            self.global_uncertainty_metrics = global_metrics
            
            self.logger.info(f"Uncertainty quantification completed for {len(uncertainty_profiles)} relationships")
            
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
        
        # Ensure sufficient sample size
        if len(numerical_data) < 30:
            self.logger.warning(f"Small sample size ({len(numerical_data)}) may affect uncertainty estimates")
        
        return numerical_data
    
    async def _analyze_relationship_uncertainty(self, 
                                             relationship: Any,
                                             data: pd.DataFrame,
                                             methods: List[ConfidenceMethod],
                                             robustness_tests: List[RobustnessTest]) -> CausalUncertaintyProfile:
        """Analyze uncertainty for a single causal relationship"""
        
        relationship_id = f"{relationship.source_variable}_{relationship.target_variable}"
        
        # Extract relationship data
        if hasattr(relationship, 'source_variable') and hasattr(relationship, 'target_variable'):
            source_var = relationship.source_variable
            target_var = relationship.target_variable
        else:
            # Fallback for different relationship formats
            source_var = str(relationship).split(' -> ')[0] if ' -> ' in str(relationship) else 'unknown'
            target_var = str(relationship).split(' -> ')[1] if ' -> ' in str(relationship) else 'unknown'
        
        if source_var not in data.columns or target_var not in data.columns:
            # Create dummy uncertainty profile for missing variables
            return self._create_dummy_uncertainty_profile(relationship_id, source_var, target_var)
        
        # Get causal strength estimate
        if hasattr(relationship, 'strength'):
            causal_strength_point = relationship.strength
        elif hasattr(relationship, 'lag_strength'):
            causal_strength_point = relationship.lag_strength
        else:
            # Calculate simple correlation as fallback
            causal_strength_point = abs(data[source_var].corr(data[target_var]))
        
        # Uncertainty estimation for causal strength
        strength_uncertainty = await self._estimate_causal_strength_uncertainty(
            source_var, target_var, data, causal_strength_point, methods
        )
        
        # Uncertainty estimation for statistical significance
        significance_uncertainty = await self._estimate_significance_uncertainty(
            source_var, target_var, data, methods
        )
        
        # Robustness testing
        robustness_results = []
        for test_type in robustness_tests:
            try:
                robustness_result = await self._perform_robustness_test(
                    source_var, target_var, data, causal_strength_point, test_type
                )
                robustness_results.append(robustness_result)
            except Exception as e:
                self.logger.warning(f"Robustness test {test_type.value} failed: {e}")
        
        # Model uncertainty
        model_uncertainty = await self._estimate_model_uncertainty(source_var, target_var, data)
        
        # Overall confidence calculation
        overall_confidence = self._calculate_overall_confidence(
            strength_uncertainty, significance_uncertainty, robustness_results
        )
        
        # Uncertainty source decomposition
        uncertainty_sources = self._decompose_uncertainty_sources(
            strength_uncertainty, significance_uncertainty, robustness_results, model_uncertainty
        )
        
        return CausalUncertaintyProfile(
            relationship_id=relationship_id,
            source_variable=source_var,
            target_variable=target_var,
            causal_strength=strength_uncertainty,
            statistical_significance=significance_uncertainty,
            robustness_results=robustness_results,
            model_uncertainty=model_uncertainty,
            overall_confidence=overall_confidence,
            uncertainty_sources=uncertainty_sources
        )
    
    def _create_dummy_uncertainty_profile(self, relationship_id: str, source_var: str, target_var: str) -> CausalUncertaintyProfile:
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
    
    async def _estimate_causal_strength_uncertainty(self, 
                                                  source_var: str,
                                                  target_var: str,
                                                  data: pd.DataFrame,
                                                  point_estimate: float,
                                                  methods: List[ConfidenceMethod]) -> UncertaintyEstimate:
        """Estimate uncertainty in causal strength"""
        
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        # Combine results from multiple methods
        confidence_intervals = []
        method_weights = []
        
        for method in methods:
            try:
                if method == ConfidenceMethod.BOOTSTRAP:
                    ci = await self._bootstrap_confidence_interval(x_data, y_data, point_estimate)
                    confidence_intervals.append(ci)
                    method_weights.append(0.4)
                
                elif method == ConfidenceMethod.BAYESIAN:
                    ci = await self._bayesian_confidence_interval(x_data, y_data, point_estimate)
                    confidence_intervals.append(ci)
                    method_weights.append(0.3)
                
                elif method == ConfidenceMethod.ASYMPTOTIC:
                    ci = await self._asymptotic_confidence_interval(x_data, y_data, point_estimate)
                    confidence_intervals.append(ci)
                    method_weights.append(0.2)
                
                elif method == ConfidenceMethod.CROSS_VALIDATION:
                    ci = await self._cv_confidence_interval(x_data, y_data, point_estimate)
                    confidence_intervals.append(ci)
                    method_weights.append(0.1)
            
            except Exception as e:
                self.logger.warning(f"Confidence interval method {method.value} failed: {e}")
        
        # Weighted average of confidence intervals
        if confidence_intervals:
            total_weight = sum(method_weights[:len(confidence_intervals)])
            weighted_lower = sum(ci[0] * w for ci, w in zip(confidence_intervals, method_weights[:len(confidence_intervals)])) / total_weight
            weighted_upper = sum(ci[1] * w for ci, w in zip(confidence_intervals, method_weights[:len(confidence_intervals)])) / total_weight
            
            final_ci = (weighted_lower, weighted_upper)
        else:
            # Fallback confidence interval
            margin = 0.1 * abs(point_estimate)
            final_ci = (max(0, point_estimate - margin), min(1, point_estimate + margin))
        
        return UncertaintyEstimate(
            point_estimate=point_estimate,
            confidence_interval=final_ci,
            confidence_level=self.confidence_level,
            uncertainty_type=UncertaintyType.PARAMETER,
            method_used=ConfidenceMethod.BOOTSTRAP,  # Primary method
            sample_size=len(x_data),
            additional_metrics={
                'methods_used': [m.value for m in methods],
                'individual_intervals': confidence_intervals
            }
        )
    
    async def _bootstrap_confidence_interval(self, 
                                           x_data: np.ndarray,
                                           y_data: np.ndarray,
                                           point_estimate: float) -> Tuple[float, float]:
        """Bootstrap confidence interval for causal strength"""
        
        n_samples = len(x_data)
        bootstrap_estimates = []
        
        for _ in range(self.bootstrap_samples):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            x_boot = x_data[indices]
            y_boot = y_data[indices]
            
            # Calculate correlation for bootstrap sample
            if np.std(x_boot) > 0 and np.std(y_boot) > 0:
                boot_estimate = abs(np.corrcoef(x_boot, y_boot)[0, 1])
                if not np.isnan(boot_estimate):
                    bootstrap_estimates.append(boot_estimate)
        
        if bootstrap_estimates:
            # Calculate confidence interval
            alpha = 1 - self.confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
            ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
            
            return (ci_lower, ci_upper)
        else:
            # Fallback
            margin = 0.1 * abs(point_estimate)
            return (max(0, point_estimate - margin), min(1, point_estimate + margin))
    
    async def _bayesian_confidence_interval(self, 
                                          x_data: np.ndarray,
                                          y_data: np.ndarray,
                                          point_estimate: float) -> Tuple[float, float]:
        """Bayesian confidence interval using conjugate priors"""
        
        try:
            # Use Bayesian linear regression
            from sklearn.linear_model import BayesianRidge
            
            X = x_data.reshape(-1, 1)
            y = y_data
            
            # Fit Bayesian ridge regression
            bayesian_reg = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
            bayesian_reg.fit(X, y)
            
            # Predict with uncertainty
            y_pred, y_std = bayesian_reg.predict(X, return_std=True)
            
            # Calculate R² and its uncertainty
            r2 = r2_score(y, y_pred)
            
            # Approximate confidence interval for correlation
            # Using Fisher transformation
            if abs(r2) < 0.99:
                z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                se_r = 1 / np.sqrt(len(x_data) - 3)  # Standard error of correlation
                
                # Fisher z-transform
                z_r = 0.5 * np.log((1 + np.sqrt(abs(r2))) / (1 - np.sqrt(abs(r2))))
                z_lower = z_r - z_score * se_r
                z_upper = z_r + z_score * se_r
                
                # Transform back
                r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                
                return (abs(r_lower), abs(r_upper))
            else:
                margin = 0.05
                return (max(0, point_estimate - margin), min(1, point_estimate + margin))
        
        except Exception as e:
            self.logger.warning(f"Bayesian confidence interval calculation failed: {e}")
            margin = 0.1 * abs(point_estimate)
            return (max(0, point_estimate - margin), min(1, point_estimate + margin))
    
    async def _asymptotic_confidence_interval(self, 
                                            x_data: np.ndarray,
                                            y_data: np.ndarray,
                                            point_estimate: float) -> Tuple[float, float]:
        """Asymptotic confidence interval for correlation"""
        
        try:
            n = len(x_data)
            
            if n < 4:
                margin = 0.2 * abs(point_estimate)
                return (max(0, point_estimate - margin), min(1, point_estimate + margin))
            
            # Fisher z-transformation
            if abs(point_estimate) < 0.99:
                z_r = 0.5 * np.log((1 + point_estimate) / (1 - point_estimate))
                se_z = 1 / np.sqrt(n - 3)
                
                z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                
                z_lower = z_r - z_score * se_z
                z_upper = z_r + z_score * se_z
                
                # Transform back to correlation scale
                r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                
                return (abs(r_lower), abs(r_upper))
            else:
                margin = 0.05
                return (max(0, point_estimate - margin), min(1, point_estimate + margin))
        
        except Exception as e:
            self.logger.warning(f"Asymptotic confidence interval calculation failed: {e}")
            margin = 0.1 * abs(point_estimate)
            return (max(0, point_estimate - margin), min(1, point_estimate + margin))
    
    async def _cv_confidence_interval(self, 
                                    x_data: np.ndarray,
                                    y_data: np.ndarray,
                                    point_estimate: float) -> Tuple[float, float]:
        """Cross-validation based confidence interval"""
        
        try:
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_seed)
            cv_estimates = []
            
            for train_idx, test_idx in kf.split(x_data):
                x_train, x_test = x_data[train_idx], x_data[test_idx]
                y_train, y_test = y_data[train_idx], y_data[test_idx]
                
                if len(x_train) > 1 and len(x_test) > 1:
                    # Fit simple linear regression
                    reg = LinearRegression()
                    reg.fit(x_train.reshape(-1, 1), y_train)
                    
                    # Predict on test set
                    y_pred = reg.predict(x_test.reshape(-1, 1))
                    
                    # Calculate correlation between predicted and actual
                    if np.std(y_pred) > 0 and np.std(y_test) > 0:
                        cv_corr = abs(np.corrcoef(y_pred, y_test)[0, 1])
                        if not np.isnan(cv_corr):
                            cv_estimates.append(cv_corr)
            
            if cv_estimates:
                # Calculate confidence interval from CV estimates
                cv_mean = np.mean(cv_estimates)
                cv_std = np.std(cv_estimates)
                
                z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
                margin = z_score * cv_std / np.sqrt(len(cv_estimates))
                
                return (max(0, cv_mean - margin), min(1, cv_mean + margin))
            else:
                margin = 0.15 * abs(point_estimate)
                return (max(0, point_estimate - margin), min(1, point_estimate + margin))
        
        except Exception as e:
            self.logger.warning(f"CV confidence interval calculation failed: {e}")
            margin = 0.1 * abs(point_estimate)
            return (max(0, point_estimate - margin), min(1, point_estimate + margin)) 
   async def _estimate_significance_uncertainty(self, 
                                               source_var: str,
                                               target_var: str,
                                               data: pd.DataFrame,
                                               methods: List[ConfidenceMethod]) -> UncertaintyEstimate:
        """Estimate uncertainty in statistical significance"""
        
        x_data = data[source_var].values
        y_data = data[target_var].values
        
        # Calculate p-value for correlation
        if len(x_data) > 3:
            corr, p_value = stats.pearsonr(x_data, y_data)
        else:
            p_value = 1.0
        
        # Bootstrap p-values
        bootstrap_p_values = []
        
        for _ in range(min(100, self.bootstrap_samples)):  # Fewer samples for p-values
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
            # Fallback confidence interval
            margin = 0.1 * p_value
            ci_lower = max(0, p_value - margin)
            ci_upper = min(1, p_value + margin)
        
        return UncertaintyEstimate(
            point_estimate=p_value,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=self.confidence_level,
            uncertainty_type=UncertaintyType.PARAMETER,
            method_used=ConfidenceMethod.BOOTSTRAP,
            sample_size=len(x_data),
            additional_metrics={'bootstrap_p_values': bootstrap_p_values[:10]}  # Store first 10
        )
    
    async def _perform_robustness_test(self, 
                                     source_var: str,
                                     target_var: str,
                                     data: pd.DataFrame,
                                     original_estimate: float,
                                     test_type: RobustnessTest) -> RobustnessResult:
        """Perform robustness test for causal relationship"""
        
        x_data = data[source_var].values
 on())ificatiinty_quantrtacausal_uncen(test_.rus = asyncio
    resultin__":ma_ == "__
if __name_ts
 resul
    return")
    ght}{insi  - int(f"    pr      ts']:
  ighinsess_'robustninsights[n ht i  for insig     s:")
 insightobustness   print("R  hts']:
    stness_insig'robuif insights[
    
    ")easons'])}ainty_rcertrel['unin('.jos: {', soneaint(f"    R    pr    f}")
    ce']:.3onfidenl['cce={re}: confidenship']tionrela {rel['t(f" in        pr  ships']:
  relationuncertain_nsights[' rel in i
        foronships:")tirtain relat("Unce       prin
 :hips']lationscertain_reinsights['unif   
    
  ")3f}ce']:.enconfidce={rel['onfiden c]}:tionship'ela"  {rel['r  print(f          :
hips']tionsela_rceonfidengh_cts['hisighel in inor r        f
")ps:ationshielidence r"High conf     print(
   s']:shiprelatione_encigh_confidts['hsigh in   
    ifsights']
 intainty_['uncerltsesuights = r")
    insghts ===nsitainty I\n=== Uncerprint("    ghts
t insi   # Prin
 ]")
    ']:.3f}st['max{conf_dimin']:.3f}, ist['nf_d range: [{co"Confidence    print(f   .3f}")
 ['std']:onf_dist {c.3f} ±mean']:ist['nf_dence: {coMean confid print(f"       ion']
ributfidence_distics']['connty_metral_uncertaiults['glob= resf_dist     concs']:
    inty_metriertancs['global_u in resultbution'distrience_f 'confid==")
    iy Metrics =rtaintcelobal Un\n=== Gint("ics
    pral metrt glob   # Prin")
    
 tness:.3f}obusvg_ress: {aobustnage rf"  Averrint(  p
          ults])bustness_rese.rofilror r in p_score foobustness[r.r np.mean(ess =obustn       avg_rults:
     ss_resustnerobprofile.  if     
         :.3f}")
 confidencell_.overalerofinfidence: {p co"  Overallint(f
        pr.3f}]")nterval[1]:ence_igth.confidl_strenile.causa   f"{prof           ]:.3f}, "
terval[0nfidence_in.col_strength.causarofile      f"[{p        } "
.3ftimate:espoint_l_strength.saofile.cauth: {prausal strengint(f"  C      prid}:")
  nship_relatioofile.pr"\n{    print(f  es']:
  profiluncertainty_s['sultofile in re    for prles ===")
tainty Profi\n=== Uncer  print("  
profilesrtainty int unce
    # Pr)
    s"ationship} relzed']ips_analyrelationsh]['n_etadata'lysis_mnaesults['analyzed {r print(f"Ads")
   .2f} secon start_time:me -{end_tid in pleteon comntificatitainty qua"\nUncer    print(flts
 resu
    # Print
    time()e = time.nd_tim e       
   )
   ]
 OVAL
     UTLIER_REMnessTest.Ostobu           RABILITY,
 AMPLE_ST.SUBStnessTest    Robus      RBATION,
  ATA_PERTUssTest.Dustne         Robtests=[
   obustness_
        r   ],    OTIC
 ASYMPTceMethod.    Confiden     
   YESIAN,hod.BAnfidenceMet   Co       AP,
  hod.BOOTSTRceMetnfiden     Co    =[
   ethods    m
    a,ta=dat  da      ships,
=relationonshipsticausal_rela
        y(ertaintsal_unccauuantify_wait cuq.qesults = a r
    
   e().timime = time   start_t")
 est...tion ty quantificantertai uncng causalti("Stars
    printnalysi a uncertainty    # Run   
n(config)
 ntificatiotyQuasalUncertain= Cau cuq    
      }
 ed': 42
 random_se        '],
0.05, 0.1ls': [0.01, ion_leveurbat    'pert  
   5,cv_folds':
        ' 500,es':tstrap_sampl      'boo
  vel': 0.95,idence_leconf        'onfig = {
   cation
 icty quantif uncertainlizeitia
    # In ]
     0.1)
   'x4', 'x2',ionship(usalRelatkCa    Moc.3),
    'x3', 0nship('x1', elatiousalR    MockCa.7),
    'x2', 0('x1', ationshipckCausalRel      Mo = [
  ipsshon relati    
   th: float
streng    r
    le: striabt_va  targe str
      e:ce_variabl        sourship:
lRelation MockCausas
    classas   @dataclhips
 elationssal rk cau Create moc #     })
    
 x4
     'x4':     ,
3': x3        'x': x2,
'x2       ,
  x1':'x1        e({
 pd.DataFram   data =    
 ependent
# Indmples)  sa 1, n_om.normal(0,p.rand x4 = nship
   ionat relles)  # Weak0.8, n_sampl(0, m.norma+ np.rando = 0.3 * x1 
    x3ionshipong relat# Strsamples)  (0, 0.5, n_andom.normal + np.rx1 x2 = 0.7 * amples)
    n_s1,ormal(0,  np.random.n  x1 =e
  tructurusal scawn s with knolete variab Crea   
    #
 les = 200 n_samp  42)
 m.seed(    np.randonships
latioreth known  data wieticthe syn   # Creat   
 """
 ic datanthet with syantification quintyal uncertat caus"Tes    ""):
cation(fity_quantiertainal_unccausnc def test_testing
asy usage and pleamExts

# sighn inuret
        r
              )        "
  kely robustlts are lisu- rems ithen algorment betwed agree     "Goo               .append(
nsights']on_i_selecti['model   insights           .1:
  rtainty < 0g_unce   elif al            )
       
      es"approachemble  enssider con -gorithmst between aleemenigh disagr      "H             d(
 .appen_insights']ctionodel_sele['mights     ins          ty > 0.3:
 inalg_uncerta    if    
               
  y']_uncertaint['algorithmuncertaintymodel_ty = lg_uncertain      a
      nty:el_uncertai' in mod_uncertaintyhm'algoritf         its
insigh selection odel  # M
                     )
       "
  tesnty estimaaincert reliable ued for morereascould be inc size ple"Overall sam            (
        '].appendndationsommeple_size_rec['sam   insights     
         100:'] <e_size['mean_sampllysis sample_ana    if  
                   )
       
        "sizesample  smalls with sionshiprelatize']} le_s['small_sampalysis_ansampledata for {ing more ollectder c    f"Consi           nd(
     appens'].mendatiosize_recomle_sampsights[' in             
  0:_size'] > small_samples['nalysisample_a      if            
       nalysis']
ample_size_arics['sbal_metysis = gloe_analsampl     
       s:bal_metricglolysis' in e_size_anaamplf 's i       endations
recomm size Sample        # 
        
)rbations"ive to pertue sensitnships may belatiosome rbustness - Moderate rod(".appents']nsighustness_is['rob   insight                    else:
           ")
  uesality isser data qu consid detected -tness"Low robusend(ights'].apps_insbustneshts['ro      insig             < 0.4:
  nessan_robustlif me   e             ps")
tionshiacross relas nes robustl highalppend("Overghts'].a_insi'robustnessinsights[             :
        > 0.7ustness mean_rob          if
                      scores)
ss_ustneall_rob = np.mean(ssobustnean_r      me
          s:ores_scesrobustn    if all_
            
        s_score)tnessult.robuss_reobustnesappend(rscores.ess_l_robustn    al            s:
    ltsuobustness_reofile.result in prrobustness_r    for   
          s:ofilety_prin uncertainr profile           fo  ]
 = [resscotness_ all_robus           les:
fitainty_prof uncer     its
   ess insigh# Robustn      
    )
           }          _reasons
 aintyrtce: un_reasons'intyerta   'unc               e,
  ncnfideall_cofile.overpro: nfidence'         'co    ",
       e}abl_variofile.targetble} -> {pre_variae.sourcprofil"{hip': flations 're                   append({
ips'].nshiotain_relats['uncer   insight        
                ss")
     tne"low robusappend(sons.nty_reaertai         unc             
   < 0.5:g_robustness    if av               results])
 ss_obustnen profile.rre for r iustness_scoean([r.robss = np.mneustavg_rob                  esults:
  ustness_rprofile.robf        i  
       ssobustnek rhec# C                          
  ze")
     sample simall"s.append(reasonsnty_   uncertai                50:
  e_size <mpl.sa_strengthusalcaile.rof   if p             ize
k sample s  # Chec            
            )
      rval"dence inteide confiappend("wy_reasons.intncerta      u         0.3:
     h > i_widtf c          i     val[0]
 intere_dencth.confiausal_strengofile.crval[1] - prdence_inteh.confiusal_strengte.cailrofth = p      ci_wid       h
   l widtintervaidence conf  # Check             
             []
      sons =ertainty_rea        unc         0.5:
e <encll_confidra.overofile if p          ofiles:
 ty_pr uncertainofile in pror      f
  ionshipsertain relatnc udentify
        # I)
             }       te
    timaoint_estrength.pile.causal_sth': prof   'streng                 onfidence,
_callvere.oofil': prnfidence   'co            
     le}",rget_variabprofile.tale} -> {e_variabourcprofile.sp': f"{shi   'relation            nd({
     ips'].appeshtionfidence_relahigh_conts['insigh            
    ce > 0.8:fidenll_conerae.ovrofil      if p  iles:
    y_profncertaintile in u  for prof   ips
    relationshenceonfidigh c# Identify h  
        }
      
        : []ghts'on_insielecti'model_s       
     ions': [],mendatsize_recom  'sample_          s': [],
ss_insightnerobust    '
        ],onships': [ain_relaticertun           's': [],
 ionshipdence_relath_confi 'hig       {
      =  insights        
   """
   salysiertainty anhts from uncsiginenerate     """G
     Any]:tr,t[sny]) -> Dicct[str, Aty: Diertain_uncdel         mo                           
        Any], Dict[str,trics:obal_megl                                       le],
    rofiaintyPausalUncertst[Crofiles: Litainty_p uncer                                   , 
       ights(selftainty_insrate_uncerc def _gene
    asynlysis
    napagation_aturn pro   re     
     
      }       
      nties)taiuncere_cascaden(ascades': l   'n_c          )),
       ncertaintiesx(cascade_uoat(np.ma: fltainty'erscade_unc     'max_ca            ies)),
   uncertaintcade_casan(np.mey': float(_uncertaintn_cascade     'mea          = {
     certainty'] ade_unalysis['cascann_ropagatio     p         ies:
  uncertaintscade_f ca       i         
 e
        continu                         
  t:   excep                    ainty)
 e_uncertnd(cascadainties.appertscade_unce ca                            
       uncertaintyth_ 1 - pacertainty =  cascade_un                                         
                             inty)
certage_un*= (1 - ednty th_uncertai     pa                                 nty']
  ertaih[i+1]]['uncth[i]][pat= G[partainty  edge_unce                                
       :1)) - len(pathi in range(       for                              1.0
certainty = path_un                                  th
  ty along pa uncertainulativeate cum# Calcul                                scade)
     nodes (caAt least 32:  # en(path) > f l          i                     n paths:
   for path i                       
                              
 f=3))arget, cutofce, turhs(G, somple_patall_si(nx.s = list      path                      y:
  tr                     t:
 rgeource != ta sif            
        odes(): in G.nget    for tar        ):
     G.nodes(e inurcfor so        
               ]
 = [nties taiscade_uncerca          tainty
  uncercascade d calculate  paths an      # Find   
      
               }
       1 else 0nodes() >mber_of_)) if G.nu - 1nodes()G.number_of_es() * (mber_of_nod(G.nu/ s() of_edge: G.number_y'nty_densit'uncertai       ,
         =True)]))dges(data in G.e _, data _, forty']ncertain[data['uax((np.my': float_uncertaint      'max    ])),
      data=True)edges( in G._, _, datar y'] focertaintdata['unnp.mean([float(certainty':    'avg_un          (),
   _of_edges.numberges': G  'n_ed      
        ),s(ode_n G.number_of': 'n_nodes           = {
    y'] aintwork_uncert['netisysion_anal    propagat        s() > 0:
mber_of_node.nuif G    
    ertiesork prop netwyzealAn        #        
nty)
 y=uncertaintuncertairget, , ta(source G.add_edge   
                   nce
 ideoverall_conffile.pro-  1 y =nttai  uncer          le
riabget_vaofile.taret = pr     targ
       ce_variablesourle.rce = profi  sou    s:
      ilenty_profertaiuncin e  profilor
        f
        ).DiGraph(    G = nx
    tworkainty necertild un        # Bu    
lysis
    naopagation_aurn pr         ret   es) < 2:
inty_profiluncerta len(if      
  
          }      {}
 ation':plificnty_amcertai     'un
       ainty': {},rte_uncescad   'ca        y': {},
 certaintk_unortw        'ne = {
    nalysisn_a propagatio
           ""
    s"networkausal s through cagateoptainty preruncze how """Analy       tr, Any]:
 ]) -> Dict[sofilertaintyPrsalUncees: List[Cauilinty_prof  uncerta                                          self, 
 ropagation(y_pntze_uncertainaly_ac def yn    
    astainty
del_uncer moeturn    r           

 y'] = 0.0uncertaintm_lgorithinty['artadel_unce   mo
         lse:
        ecertainty)unt(algorithm_'] = floaertaintyhm_unc'algoritainty[el_uncert         mod
   ) + 1e-10)(values.meanlues) / (np = np.std(vancertainty algorithm_u        s())
   lts.valueesulgorithm_rt(alis   values = :
         ts) > 1thm_resulen(algori        if linty
thm uncertaulate algori     # Calc       
 s
   _resultthm] = algoriparison'omorithm_clgertainty['ael_unc       mod
         = 0
rithm] algots[thm_resul    algori     ")
       e}m} failed: {orithm {alggorithning(f"Aler.warself.logg              
  e:tion as xcepept Exc         e      
     ships
     mi_relation] =[algorithmtsesullgorithm_r         a     
       += 1onshipsrelati        mi_                    0.1:
    scores) > mi_.max(if np                          X, y)
  ession(al_info_regrres = mutuco      mi_s                     [col]
   y = data                          cols]
data[other_     X =              
          s:other_col  if             
          != col]olumns if c  data.cr c in fo[cols = r_c othe                    ns:
   n data.colum   for col i               s = 0
  onship  mi_relati          
        gression_info_re mutualortmption iure_seleclearn.feat    from sk            y
    discover-based ationrmal info     # Mutu          ion':
     atnformmutual_i == 'orithm elif alg        
                  
     ipsionshatreg_relgorithm] = ts[alhm_resulitlgor     a          
      += 1ionships  reg_relat                     
         0.3:X, y) > g.score(re         if                y)
     it(X,).fegression(nearRg = Li        re                   [col]
  data      y =                      her_cols]
a[ot   X = dat                         
ols:her_c      if ot             
     c != col]mns if  data.colu inr c = [c foher_cols ot                       columns:
ol in data.r c   fo       
          = 0tionships relag_      re            
  iscoveryd dn-basessio Regre          #
          egression': == 'rgorithm alifel         
                s
       correlationtrong_rithm] = s[algosultslgorithm_re    a              olumns)
  ata.cm() - len(d.5).sum().sur_matrix > 0cors = (ionrrelatg_cotron     s            
   .abs()orr()x = data.corr_matri c           ry
        veed discoion-basorrelatSimple c  #              ion':
     elat'corr=  = algorithm       if  
            try:      ithms:
 algor in rithmfor algo    
    
        ts = {}hm_resulit    algorion']
    atrmfoual_in, 'mutegression'ation', 'relms = ['corrorith        algmplified)
ms (sialgorithvery l disco causare different    # Compa     
    
   
        }': {}tyuncertaintural_truc       's
     y': {},sensitiviteter_     'param     },
  rison': {mpalgorithm_co  'a       {
    ertainty =model_unc          
   ""
   selection"l modeo due tainty yze uncertalAn"    ""ny]:
    str, A) -> Dict[DataFramepd.:       data                              ,
   s: List[Any]iprelationshal_  caus                                     , 
tainty(selfl_unceranalyze_modeync def _ 
    astrics
   global_me   return   
                 }

  ize < 30)e_sizes if spl samr size inum(1 fosize': s_sample_     'small0),
        size >= 5es if sample_size inm(1 for siz su_size':uate_sample'adeq        s)),
    le_size.min(sampize': int(npmple_s  'min_sa       ,
   izes))ple_sp.mean(samze': float(nsiean_sample_  'm      '] = {
    analysissize_le_amp['sobal_metrics    gl
    profiles]ainty_rtle in unceprofir fosize sample__strength.ofile.causals = [przesi    sample_y
    uacle size adeqmp       # Sa
        
       })
      e < 0.3orscores if scrobustness_ score in um(1 fors': sationshiple_relgi        'fra       ,
 core > 0.7)if sess_scores obustn score in r forips': sum(1relationsh'robust_                _scores)),
robustness.std(': float(np        'std     res)),
   ss_scoustnemean(rob': float(np.       'mean   {
      bution'] = _distritnessics['robusl_metr       globa  
   res:sconess_robust      if     
  re)
    ess_scorobustnesult.robustness_rpend(ss_scores.apbustne    ro        lts:
    ss_resu.robustneprofile in ness_resultrobustr    fo         rofiles:
_prtaintyfile in uncerofor p     = []
    _scores robustness     s
  silystness ana# Robu        
     }
                 lues))
  at(np.std(va': flo    'std            ),
    mean(values)np.ean': float(   'm                ] = {
 _type.value'][uncpetainty_by_tycer_metrics['un    global     s:
       lueif va            tems():
pe.iby_tyertainty_uncues in alnc_type, vfor u     = {}
    type']rtainty_by_ce_metrics['un    global      
    value)
  nd(unc_pe_type].ap_type[uncrtainty_byunce          
      ):s(urces.itemtainty_sorofile.uncer pvalue innc_type, uor unc_          frofiles:
  inty_prtae in unce for profil          
    Type}
 tycertainut in Unfor [] = {ut: pe tainty_by_ty      unceralysis
  source anertainty     # Unc     
    }
      
     dences))edian(confiloat(np.m'median': f            ),
ences)(confidt(np.max'max': floa            
nfidences)),np.min(coat(  'min': flo          dences)),
td(confi float(np.s    'std':   es)),
     n(confidenc.meafloat(npn':    'mea       
  tion'] = {_distribuceens['confidobal_metric  glles]
      y_profiintcertae in unprofil for denceerall_confi.ov[profiles =  confidence
       onbuti distrionfidencel cOveral   #       
 = {}
      trics   global_me    
      
    turn {}re  
          iles:ty_profin uncertanotif         
        
ips"""ationshall relcs across tainty metribal uncer glolyzeAna""":
        [str, Any]-> Dicte) ram: pd.DataF     data                             ],
      intyProfilesalUncerta List[Cauprofiles:ainty_    uncert                                  f, 
  tainty(selere_global_unclyzana def _
    asyncurces
    y_sortainturn unce      ret     
  1)
   ined * 0.l_explaota.1 - t 00,TORY] = max(e.ALEAaintyTypes[Uncertrcrtainty_souunce   s())
     s.valuenty_sourcem(uncertaied = suainpl   total_ex  idual)
    (restaintyncerleatory u A     # 
   nc
       emic_ust= epiC] ISTEMIype.EPrtaintyT[Unceourcesinty_s     uncerta
   ple sizes with sam Decrease0)  #size / 1000.le_(0, 1 - sampc = maxmic_un  episteize
      mple_ssaainty.gth_uncertensize = strple_am   sed)
     ze relatsie y (samplc uncertaintmi   # Episte    
   
      L] = 0.0pe.STRUCTURAaintyTycert_sources[Uncertainty   un
           else:   
   ilityg_stab] = avSTRUCTURALintyType.ncertaurces[Usocertainty_          un  easures)
bility_mta= np.mean(sy littabi      avg_s     sults]
 tness_re robus inultesasure for r_met.stability = [resul_measuresability   st       lts:
  suobustness_re     if r   
sts)ness te robustfromtainty ( uncer# Structural 
        nc
       l_uEL] = modeintyType.MODs[Uncertainty_sourceerta   unc
     .0)rtainty', 0'model_unceainty.get(_uncertunc = modell_      modeertainty
  unc# Model 
              _width
  th_cingETER] = streARAMpe.PTyncertaintyrces[Uainty_souuncert     rval[0]
   dence_inteainty.confirtrength_uncestal[1] - tervinfidence_tainty.conuncerh_gtstreni_width = ength_c        strtervals)
dence in confiainty (fromr uncertaramete  # P  
    
        {}s = rcetainty_sou      uncer   
       
"""rcesferent sou difty intoain uncertDecompose"""        
pe, float]:tyTytainncer -> Dict[U float])y: Dict[str,taintdel_uncer         mo                      lt],
      tnessResu List[Robuslts:sustness_re      robu                               e,
stimatrtaintyE: Uncentytaicerce_unifican        sign                         
    yEstimate,ertaintrtainty: Uncgth_unce       stren                           , 
   selfty_sources(se_uncertaindef _decompo   
    ce)
 nfidencorall_at(overn floetu  r    
      
    s)/ sum(weights)) ors, weightctfidence_fa zip(con in for cf, wm(cf * w = sufidenceall_conver    o]
    e_factors)en(confidenc.15][:l.25, 0[0.3, 0.3, 0 =    weightserage
     hted av    # Weig    
       nfidence)
 e_cosampl.append(ce_factorsdennfico     es
   sampl+ t 100onfidence a0)  # Full cize / 100., sample_s min(1.0idence =ple_conf  samize
      nty.sample_sgth_uncertaiize = stren sample_s     equacy
  e adle siz: Sampor 4   # Fact
            s)
 nesobustg_rpend(av_factors.apconfidence           res)
 ustness_scomean(robness = np.robust  avg_       
   lts]ess_resubustnlt in rofor resu_score nessobust.r [resultores =ness_sc    robust        s:
esultustness_rob if r      ness
 ust3: Rob  # Factor 
      )
        onfidenceance_cnd(significctors.appefidence_fa   con 0.0
     lse.0 e= 1value <value) if p_0, 1 - p_nce = max(ce_confidean   signific
     stimatent_eainty.poincerticance_uignifvalue = sp_e
        significanccal Statisti2: # Factor      
          nce)
 th_confidestrengors.append(dence_factfi   conidth)
     ength_ci_w(0, 1 - strdence = max_configth      stren
  val[0]nteronfidence_iinty.ch_uncerta- strengterval[1] inte_confidencty.th_uncertainth = strengci_wid   strength_
     e interval)f confidencce (width oconfidenrength actor 1: St      # F  
       ors = []
 _factnfidence co         
    "
  nship""usal relatioe in caconfidencall overCalculate       """  
 -> float:esult])essRt[Robustnults: Lis_res robustness                                  
 e,Estimatertaintytainty: Uncercance_uncfi signi                             ,
      ateaintyEstimertainty: Unccert strength_un                               self, 
    idence(nferall_coovalculate_def _c
    
    _resultsturn model    re     
    e
   scorainty_uncert model_inty'] =ertadel_unc['model_results     mo
          
  0.0s) > 1 elseen(scoreores) if l(sce = np.stdy_scoraintrtel_unce        mods()]
ts.valuel_resulin modesult r reore'] foan_sclt['meresus = [core       s
 delsss moiance acroty as varl uncertainculate modeCal   #      
       1.0}
 ore': 'std_sc 0.0, ':score = {'mean_del_name]esults[mo  model_r            : {e}")
  e} failedamel_noddel {mning(f"Mologger.war      self.        :
  ception as ept Ex  exce         }
         
        cv_scores))std(float(np.td_score':  's                  res)),
 an(cv_sco(np.meloate': foran_scme          '
          me] = {nats[model_sulreodel_  m            'r2')
  g=corin sdata, cv=3,ta, y_el, x_da(modore_val_scres = cross cv_sco           score
    ation lid# Cross-va       
            try:    ms():
     teels.iin model e, modor model_nam       f
        
         }m_seed)
.randotate=selfrandom_srs=50, r(n_estimatoressotReg RandomForesforest':om_nd        'ra0),
    1.dge(alpha='ridge': Ri    
        ession(),Regrr': Linearea   'lin
          = {odels  mdels
      nt moreffe di     # Test    
    = {}
    _results       model
        
 aluesarget_var].vta[t y_data = da      )
 ape(-1, 1es.reshlu.vaar]ce_vourdata[sata =        x_d 
 
       ""choice"e to model ainty ducert"Estimate un  ""      
t]:float[str, Dic-> me) ataFra pd.D    data:                                   var: str,
   target_                                      ,
tr: ssource_var                                     
   self, rtainty(nceodel_ue_m _estimat   async def
    
 mask]bined_om y_data[cbined_mask],x_data[com     return       
   _mask
  k & y= x_masined_mask   comb   ariable
   her vers in eitre not outlis that a only pointep Ke
        #      reshold)
  , th(y_datas_singleve_outliermomask = re
        y_ld) threshongle(x_data,iers_si remove_outlmask =x_       
        
 und)er_bouppdata <= und) & (ower_bo(data >= lrn         retu 
              
 old * IQR Q3 + threshnd =pper_bou           u* IQR
 threshold Q1 -  = ower_bound      l   
      
          - Q1R = Q3   IQ       a, 75)
  e(datercentilnp.p     Q3 = 
       25)data, .percentile(     Q1 = np    old):
   ata, threshngle(dtliers_si_ou  def remove    ariables
  R for both vIQCalculate    # 
           od"""
  R meth using IQ outliers""Remove   "]:
     arrayarray, np.nde[np.ndTupl -> ld: float)shorray, thredanp.na: y_datarray, np.ndlf, x_data: (see_outliersdef _remov
    
           )meters
 st_paras=teteraramest_p         tety,
   iviensit_stliernsitivity=ououtlier_se      e,
      ty_measursure=stabili_meaability      st),
      s_scoreobustnes0.0, rscore=max(bustness_ro           es,
 ed_estimats=perturbd_estimate  perturbe          _estimate,
e=originalmatal_esti      origin,
      =test_type test_type   (
        nessResult Robusturn    ret  
       y = 1.0
   vitier_sensiti   outl        = 1.0
 y_measure   stabilit        .0
  ss_score = 0stne robu     
          else:
    tes)stimarbed_eertulen(p) > 0.2]) / _estimateiginalt - or abs(esates ifrbed_estim in pertuest for estty = len([sensitiviier_      outl     mates])
 urbed_esti in pertstr e foe)atginal_estim(est - orian([abs= np.mesure ty_meabilista            10))
mates) + 1e-estibed_erturean(p / (np.mstimates)bed_e.std(pertur = 1.0 - (nporeness_scobust     r       
d_estimates: perturbe        ifmetrics
ess robustnlculate         # Ca     

    methodsethods'] =relation_ms['corameterst_parte      
               e
       continu             pt:
     exce      )
        (corr)ppend(abs_estimates.arbed    pertu               rr):
     snan(coif not np.i                        
               y_data)
  data,au(x_ts.kendallt_ = staorr,   c                    endall':
  'k method == elif                ta)
     y_dax_data,.spearmanr(atsrr, _ = stco                 
        'spearman':ethod ==  elif m           )
       data_data, y_arsonr(x stats.pecorr, _ =                        arson':
d == 'pemetho    if                try:
                ethods:
 hod in mr met fo                     
]
  'kendall'earman', ', 'sp = ['pearsonthods   me        ds
 ation methoalculion cary correlat  # V         mplified)
 ers (siparametlysis o anavity tTest sensiti     # 
       SITIVITY:METER_SENessTest.PARAtn == Robustypetest_   elif    
     s
     holdes outlier_thr =olds']ier_thresheters['outl_paramtest                   
   
  r)orpend(clean_cestimates.ap perturbed_                
       _corr):leannp.isnan(cnot         if       
      ean)[0, 1]) y_clan,oef(x_cle(np.corrccorr = abs clean_           
        ) > 0:cleantd(y_ np.sn) > 0 and(x_cleand np.std0 a) > 1n(x_clean   if le                 
        old)
     thresha,_data, y_date_outliers(xf._remov= sel, y_clean   x_clean     
         esablom both varis frere outli# Remov                sholds:
er_thren outlireshold i   for th           
          liers
tip  # IQR mul.5, 3.0], 2.0, 2= [1.5olds shthreier_outl          s
  ierto outltivity Test sensi   #          VAL:
TLIER_REMOsTest.OU Robustnesst_type ==   elif te  
           sizes
e_ubsamples'] = ssample_sizeters['subramest_pa t             
     r)
     b_cornd(suappeimates.erturbed_est        p                
        n(sub_corr):np.isnat      if no                      , 1])
 )[0b, y_suborrcoef(x_su.cbs(nporr = a sub_c                     > 0:
      std(y_sub) 0 and np.(x_sub) > np.std         if                      
                ices]
  = y_data[indub y_s                  
      ndices]_data[i_sub = x           x            
 lse)Fae=le, replac_subsamp=n, size(len(x_data)icecho np.random.  indices =                      10:
subsample >  if n_               size)
    subsample_a) * x_datnt(len(sample = iub_s          n       e
    sizmples pere subsa# Multipl  ge(25): _ in ran  for             _sizes:
 ubsample sze inubsample_sior s           f   
 
         0.8, 0.9]0.7, 0.5, zes = [_sisamplesub            es
amplubst sfereny across difitabil # Test st          ITY:
 PLE_STABILsTest.SUBSAM= Robustnes_type =estelif t         
     s
  ion_levelturbatself.pervels'] = bation_leertur'pameters[arest_p    t    
                
_corr)nd(perturbedtimates.apperturbed_es     pe                    rr):
   rbed_co(pertuot np.isnanf n        i               , 1])
 rturbed)[0urbed, y_peert(x_pcoef(np.corrbsd_corr = arbe      pertu                 ed) > 0:
 d(y_perturbp.st and n) > 0erturbedd(x_pstif np.                   stimate
 perturbed ealculate     # C             
                    ))
   y_datadata), len((y_l * np.stdse_leveal(0, noindom.norma + np.rad = y_dat y_perturbe           
        ))_data, len(xdata)td(x_evel * np.snoise_lrmal(0, .random.no + npdata x_bed =pertur         x_          ian noise
 uss# Add Ga                 r level
   ions perbatertu# Multiple p20):  in range( _ for            
    on_levels:batiturself.perin evel se_lnoi     for 
       bationerturta ptness to dat robusTes#       :
      ION_PERTURBATest.DATAustnessTRobpe ==  if test_ty  
       }
      meters = {test_para
        imates = []rturbed_est  pe
      s
        valuer].get_vaa[taratta = d   y_da    