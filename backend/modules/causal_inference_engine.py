"""
Causal Inference Engine for Geological Relationship Discovery
Advanced causal discovery algorithms (PC, FCI, GES) for geological data analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, permutations
from scipy import stats
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import networkx as nx

# Causal discovery imports with fallbacks
try:
    import causal_learn as cl
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.search.ConstraintBased.FCI import fci
    from causal_learn.search.ScoreBased.GES import ges
    from causal_learn.utils.cit import CIT
    from causal_learn.utils.PCUtils import SkeletonDiscovery
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("causal-learn not available. Using custom implementations.")

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import PC as PgmpyPC, HillClimbSearch, BayesianEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available. Some Bayesian network features will be limited.")

warnings.filterwarnings('ignore')

class CausalAlgorithm(Enum):
    PC = "pc"  # Peter-Clark algorithm
    FCI = "fci"  # Fast Causal Inference
    GES = "ges"  # Greedy Equivalence Search
    LINGAM = "lingam"  # Linear Non-Gaussian Acyclic Model
    NOTEARS = "notears"  # Non-combinatorial Optimization via Trace Exponential

class GeologicalCausalRelationType(Enum):
    TEMPORAL = "temporal"  # Temporal causation (A happens before B)
    SPATIAL = "spatial"  # Spatial causation (A influences nearby B)
    PROCESS = "process"  # Process causation (geological process A causes B)
    STRUCTURAL = "structural"  # Structural causation (structure A influences B)
    COMPOSITIONAL = "compositional"  # Chemical/mineralogical causation

@dataclass
class CausalEdge:
    """Represents a causal relationship between two geological variables"""
    source: str
    target: str
    causal_type: GeologicalCausalRelationType
    strength: float  # 0-1 confidence in causal relationship
    p_value: float
    effect_size: float
    confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    domain_validation: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CausalGraph:
    """Causal graph representation with geological domain knowledge"""
    nodes: List[str]
    edges: List[CausalEdge]
    adjacency_matrix: np.ndarray
    algorithm_used: CausalAlgorithm
    confidence_threshold: float
    domain_constraints: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)

class GeologicalDomainValidator:
    """Validates causal relationships using geological domain knowledge"""
    
    def __init__(self):
        self.domain_rules = self._initialize_domain_rules()
        self.temporal_constraints = self._initialize_temporal_constraints()
        self.process_hierarchies = self._initialize_process_hierarchies()
    
    def _initialize_domain_rules(self) -> Dict[str, List[str]]:
        """Initialize geological domain knowledge rules"""
        return {
            # Temporal ordering rules
            'temporal_precedence': [
                'magma_intrusion -> metamorphism',
                'deformation -> folding',
                'weathering -> erosion',
                'sedimentation -> diagenesis',
                'uplift -> erosion',
                'fault_formation -> earthquake_activity'
            ],
            
            # Process causation rules
            'process_causation': [
                'temperature_gradient -> convection',
                'pressure_increase -> metamorphism',
                'fluid_circulation -> hydrothermal_alteration',
                'tectonic_stress -> fault_activation',
                'magma_differentiation -> mineral_crystallization'
            ],
            
            # Spatial causation rules
            'spatial_causation': [
                'fault_proximity -> seismic_activity',
                'volcanic_center -> geothermal_anomaly',
                'ore_deposit -> geochemical_anomaly',
                'fracture_network -> fluid_flow'
            ],
            
            # Compositional causation rules
            'compositional_causation': [
                'silica_content -> melt_viscosity',
                'water_content -> melting_temperature',
                'iron_content -> magnetic_susceptibility',
                'carbonate_content -> dissolution_rate'
            ]
        }
    
    def _initialize_temporal_constraints(self) -> Dict[str, Dict[str, float]]:
        """Initialize temporal constraints for geological processes"""
        return {
            'magmatic_processes': {'timescale_min': 1e3, 'timescale_max': 1e7},  # years
            'metamorphic_processes': {'timescale_min': 1e5, 'timescale_max': 1e8},
            'sedimentary_processes': {'timescale_min': 1e2, 'timescale_max': 1e6},
            'tectonic_processes': {'timescale_min': 1e4, 'timescale_max': 1e9},
            'surface_processes': {'timescale_min': 1e0, 'timescale_max': 1e6}
        }
    
    def _initialize_process_hierarchies(self) -> Dict[str, List[str]]:
        """Initialize hierarchical process relationships"""
        return {
            'tectonic_system': [
                'plate_motion', 'stress_field', 'fault_activity', 'earthquake_occurrence'
            ],
            'magmatic_system': [
                'heat_source', 'melting', 'magma_transport', 'crystallization', 'eruption'
            ],
            'hydrologic_system': [
                'precipitation', 'infiltration', 'groundwater_flow', 'spring_discharge'
            ],
            'geochemical_system': [
                'source_composition', 'transport_mechanism', 'reaction_kinetics', 'product_formation'
            ]
        }
    
    def validate_causal_relationship(self, edge: CausalEdge, 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """Validate causal relationship using geological domain knowledge"""
        validation_results = {
            'domain_plausible': False,
            'temporal_consistent': False,
            'process_logical': False,
            'spatial_reasonable': False,
            'strength_adequate': False,
            'overall_score': 0.0,
            'warnings': [],
            'recommendations': []
        }
        
        # Check domain plausibility
        validation_results['domain_plausible'] = self._check_domain_plausibility(edge)
        
        # Check temporal consistency
        validation_results['temporal_consistent'] = self._check_temporal_consistency(edge, data)
        
        # Check process logic
        validation_results['process_logical'] = self._check_process_logic(edge)
        
        # Check spatial reasonableness
        validation_results['spatial_reasonable'] = self._check_spatial_reasonableness(edge, data)
        
        # Check statistical strength
        validation_results['strength_adequate'] = edge.strength > 0.3 and edge.p_value < 0.05
        
        # Calculate overall validation score
        scores = [
            validation_results['domain_plausible'],
            validation_results['temporal_consistent'],
            validation_results['process_logical'],
            validation_results['spatial_reasonable'],
            validation_results['strength_adequate']
        ]
        validation_results['overall_score'] = sum(scores) / len(scores)
        
        # Generate warnings and recommendations
        validation_results['warnings'] = self._generate_warnings(edge, validation_results)
        validation_results['recommendations'] = self._generate_recommendations(edge, validation_results)
        
        return validation_results
    
    def _check_domain_plausibility(self, edge: CausalEdge) -> bool:
        """Check if causal relationship is plausible from domain perspective"""
        relationship_key = f"{edge.source} -> {edge.target}"
        
        for rule_type, rules in self.domain_rules.items():
            for rule in rules:
                if relationship_key in rule or any(term in rule for term in [edge.source, edge.target]):
                    return True
        
        return False
    
    def _check_temporal_consistency(self, edge: CausalEdge, data: pd.DataFrame) -> bool:
        """Check temporal consistency of causal relationship"""
        if edge.causal_type != GeologicalCausalRelationType.TEMPORAL:
            return True  # Non-temporal relationships don't need temporal validation
        
        # Check if source precedes target temporally (simplified check)
        if 'timestamp' in data.columns or 'time' in data.columns:
            time_col = 'timestamp' if 'timestamp' in data.columns else 'time'
            
            # Calculate lag correlation to check temporal precedence
            try:
                source_series = data[edge.source]
                target_series = data[edge.target]
                
                # Simple lag analysis
                for lag in range(1, min(10, len(data) // 4)):
                    if len(source_series) > lag:
                        lagged_corr = np.corrcoef(
                            source_series[:-lag], target_series[lag:]
                        )[0, 1]
                        if abs(lagged_corr) > abs(np.corrcoef(source_series, target_series)[0, 1]):
                            return True
            except:
                pass
        
        return True  # Default to true if temporal analysis not possible
    
    def _check_process_logic(self, edge: CausalEdge) -> bool:
        """Check if causal relationship follows geological process logic"""
        for system, processes in self.process_hierarchies.items():
            source_in_system = any(proc in edge.source.lower() for proc in processes)
            target_in_system = any(proc in edge.target.lower() for proc in processes)
            
            if source_in_system and target_in_system:
                # Check if source comes before target in process hierarchy
                source_idx = next((i for i, proc in enumerate(processes) 
                                 if proc in edge.source.lower()), -1)
                target_idx = next((i for i, proc in enumerate(processes) 
                                 if proc in edge.target.lower()), -1)
                
                if source_idx >= 0 and target_idx >= 0:
                    return source_idx < target_idx
        
        return True  # Default to true if no specific process logic found
    
    def _check_spatial_reasonableness(self, edge: CausalEdge, data: pd.DataFrame) -> bool:
        """Check spatial reasonableness of causal relationship"""
        if edge.causal_type != GeologicalCausalRelationType.SPATIAL:
            return True
        
        # Check if spatial coordinates are available
        spatial_cols = [col for col in data.columns if any(coord in col.lower() 
                       for coord in ['x', 'y', 'z', 'lat', 'lon', 'depth', 'elevation'])]
        
        if len(spatial_cols) >= 2:
            # Simple spatial correlation check
            try:
                # This is a simplified check - in practice, you'd want more sophisticated spatial analysis
                return True  # Placeholder for more complex spatial validation
            except:
                pass
        
        return True
    
    def _generate_warnings(self, edge: CausalEdge, validation_results: Dict[str, Any]) -> List[str]:
        """Generate warnings based on validation results"""
        warnings = []
        
        if not validation_results['domain_plausible']:
            warnings.append(f"Causal relationship {edge.source} -> {edge.target} may not be geologically plausible")
        
        if not validation_results['temporal_consistent']:
            warnings.append(f"Temporal ordering of {edge.source} -> {edge.target} may be inconsistent")
        
        if edge.strength < 0.2:
            warnings.append(f"Causal strength ({edge.strength:.3f}) is relatively weak")
        
        if edge.p_value > 0.05:
            warnings.append(f"Statistical significance (p={edge.p_value:.3f}) is questionable")
        
        return warnings
    
    def _generate_recommendations(self, edge: CausalEdge, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving causal inference"""
        recommendations = []
        
        if validation_results['overall_score'] < 0.6:
            recommendations.append("Consider additional domain expertise consultation")
            recommendations.append("Collect more targeted observational data")
        
        if not validation_results['temporal_consistent']:
            recommendations.append("Include temporal lag analysis in causal discovery")
            recommendations.append("Consider time-series causal methods")
        
        if len(edge.confounders) == 0:
            recommendations.append("Investigate potential confounding variables")
        
        return recommendations

class CausalDiscoveryEngine:
    """Advanced causal discovery engine for geological data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.domain_validator = GeologicalDomainValidator()
        self.discovered_graphs = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for causal discovery"""
        return {
            'algorithms': [CausalAlgorithm.PC, CausalAlgorithm.GES, CausalAlgorithm.FCI],
            'confidence_threshold': 0.05,
            'independence_test': 'fisherz',  # fisherz, chi_square, gsq
            'score_function': 'bic',  # bic, aic, bdeu
            'max_conditioning_set_size': 3,
            'bootstrap_samples': 100,
            'domain_validation': True,
            'parallel_processing': True,
            'cache_results': True,
            'intervention_analysis': True
        }
    
    async def discover_causal_structure(self, data: pd.DataFrame, 
                                      variable_types: Dict[str, str] = None,
                                      prior_knowledge: Dict[str, Any] = None) -> Dict[CausalAlgorithm, CausalGraph]:
        """Discover causal structure using multiple algorithms"""
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Run multiple causal discovery algorithms
        results = {}
        
        for algorithm in self.config['algorithms']:
            self.logger.info(f"Running {algorithm.value} algorithm...")
            
            try:
                if algorithm == CausalAlgorithm.PC:
                    causal_graph = await self._run_pc_algorithm(processed_data, prior_knowledge)
                elif algorithm == CausalAlgorithm.FCI:
                    causal_graph = await self._run_fci_algorithm(processed_data, prior_knowledge)
                elif algorithm == CausalAlgorithm.GES:
                    causal_graph = await self._run_ges_algorithm(processed_data, prior_knowledge)
                else:
                    self.logger.warning(f"Algorithm {algorithm.value} not implemented")
                    continue
                
                # Validate discovered relationships
                if self.config['domain_validation']:
                    causal_graph = self._validate_causal_graph(causal_graph, data)
                
                results[algorithm] = causal_graph
                
            except Exception as e:
                self.logger.error(f"Error running {algorithm.value}: {str(e)}")
                continue
        
        # Store results
        self.discovered_graphs = results
        
        return results
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess data for causal discovery"""
        # Handle missing values
        processed_data = data.fillna(data.mean())
        
        # Standardize continuous variables
        scaler = StandardScaler()
        
        # Identify continuous columns
        continuous_cols = processed_data.select_dtypes(include=[np.number]).columns
        
        if len(continuous_cols) > 0:
            processed_data[continuous_cols] = scaler.fit_transform(processed_data[continuous_cols])
        
        return processed_data.values
    
    async def _run_pc_algorithm(self, data: np.ndarray, 
                               prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Peter-Clark (PC) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            cg = pc(
                data, 
                alpha=self.config['confidence_threshold'],
                indep_test=self.config['independence_test'],
                stable=True,
                uc_rule=1,
                uc_priority=2
            )
            
            adjacency_matrix = cg.G.graph
            
        else:
            # Custom PC implementation
            adjacency_matrix = await self._custom_pc_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.PC
        )
        
        return causal_graph
    
    async def _run_fci_algorithm(self, data: np.ndarray, 
                                prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Fast Causal Inference (FCI) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            G, edges = fci(
                data, 
                independence_test_method=self.config['independence_test'],
                alpha=self.config['confidence_threshold'],
                depth=-1,
                max_path_length=-1
            )
            
            adjacency_matrix = G.graph
            
        else:
            # Fallback to PC algorithm
            adjacency_matrix = await self._custom_pc_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.FCI
        )
        
        return causal_graph
    
    async def _run_ges_algorithm(self, data: np.ndarray, 
                                prior_knowledge: Dict[str, Any] = None) -> CausalGraph:
        """Run Greedy Equivalence Search (GES) algorithm"""
        
        if CAUSAL_LEARN_AVAILABLE:
            # Use causal-learn implementation
            Record = ges(
                data,
                score_func=self.config['score_function'],
                maxP=self.config['max_conditioning_set_size'],
                parameters=None
            )
            
            adjacency_matrix = Record['G'].graph
            
        else:
            # Custom GES implementation
            adjacency_matrix = await self._custom_ges_algorithm(data)
        
        # Convert to causal graph
        causal_graph = self._adjacency_to_causal_graph(
            adjacency_matrix, data, CausalAlgorithm.GES
        )
        
        return causal_graph
    
    async def _custom_pc_algorithm(self, data: np.ndarray) -> np.ndarray:
        """Custom implementation of PC algorithm"""
        n_vars = data.shape[1]
        adjacency = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase 1: Edge removal based on conditional independence
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency[i, j] == 0:
                    continue
                
                # Test independence conditioning on subsets of neighbors
                neighbors = [k for k in range(n_vars) if k != i and k != j and adjacency[i, k] == 1]
                
                for cond_set_size in range(min(len(neighbors), self.config['max_conditioning_set_size']) + 1):
                    if cond_set_size == 0:
                        # Test marginal independence
                        p_value = self._independence_test(data[:, i], data[:, j])
                    else:
                        # Test conditional independence
                        for cond_set in combinations(neighbors, cond_set_size):
                            cond_data = data[:, list(cond_set)]
                            p_value = self._conditional_independence_test(
                                data[:, i], data[:, j], cond_data
                            )
                            
                            if p_value > self.config['confidence_threshold']:
                                adjacency[i, j] = 0
                                adjacency[j, i] = 0
                                break
                    
                    if adjacency[i, j] == 0:
                        break
        
        return adjacency
    
    async def _custom_ges_algorithm(self, data: np.ndarray) -> np.ndarray:
        """Custom implementation of GES algorithm"""
        n_vars = data.shape[1]
        
        # Start with empty graph
        adjacency = np.zeros((n_vars, n_vars))
        
        # Forward phase: Add edges that improve score
        improved = True
        while improved:
            improved = False
            best_score_improvement = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adjacency[i, j] == 0:
                        # Try adding edge i -> j
                        test_adjacency = adjacency.copy()
                        test_adjacency[i, j] = 1
                        
                        if self._is_acyclic(test_adjacency):
                            score_improvement = self._score_improvement(data, adjacency, test_adjacency)
                            
                            if score_improvement > best_score_improvement:
                                best_score_improvement = score_improvement
                                best_edge = (i, j)
            
            if best_edge is not None:
                adjacency[best_edge[0], best_edge[1]] = 1
                improved = True
        
        # Backward phase: Remove edges that improve score
        improved = True
        while improved:
            improved = False
            best_score_improvement = 0
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency[i, j] == 1:
                        # Try removing edge i -> j
                        test_adjacency = adjacency.copy()
                        test_adjacency[i, j] = 0
                        
                        score_improvement = self._score_improvement(data, adjacency, test_adjacency)
                        
                        if score_improvement > best_score_improvement:
                            best_score_improvement = score_improvement
                            best_edge = (i, j)
            
            if best_edge is not None:
                adjacency[best_edge[0], best_edge[1]] = 0
                improved = True
        
        return adjacency
    
    def _independence_test(self, x: np.ndarray, y: np.ndarray) -> float:
        """Test marginal independence between two variables"""
        if self.config['independence_test'] == 'fisherz':
            # Fisher's z-test for correlation
            corr, p_value = stats.pearsonr(x, y)
            return p_value
        elif self.config['independence_test'] == 'chi_square':
            # Chi-square test for independence
            _, p_value = stats.chi2_contingency(np.histogram2d(x, y)[0])[:2]
            return p_value
        else:
            # Default to Fisher's z-test
            corr, p_value = stats.pearsonr(x, y)
            return p_value
    
    def _conditional_independence_test(self, x: np.ndarray, y: np.ndarray, 
                                     z: np.ndarray) -> float:
        """Test conditional independence X ⊥ Y | Z"""
        # Partial correlation test
        try:
            # Regress X on Z
            reg_x = LinearRegression().fit(z, x)
            residual_x = x - reg_x.predict(z)
            
            # Regress Y on Z
            reg_y = LinearRegression().fit(z, y)
            residual_y = y - reg_y.predict(z)
            
            # Test independence of residuals
            corr, p_value = stats.pearsonr(residual_x, residual_y)
            return p_value
            
        except Exception:
            # Fallback to marginal independence test
            return self._independence_test(x, y)
    
    def _is_acyclic(self, adjacency: np.ndarray) -> bool:
        """Check if directed graph is acyclic"""
        n_vars = adjacency.shape[0]
        
        # Topological sort using Kahn's algorithm
        in_degree = np.sum(adjacency, axis=0)
        queue = [i for i in range(n_vars) if in_degree[i] == 0]
        processed = 0
        
        while queue:
            node = queue.pop(0)
            processed += 1
            
            for neighbor in range(n_vars):
                if adjacency[node, neighbor] == 1:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return processed == n_vars
    
    def _score_improvement(self, data: np.ndarray, old_graph: np.ndarray, 
                          new_graph: np.ndarray) -> float:
        """Calculate BIC score improvement"""
        old_score = self._calculate_bic_score(data, old_graph)
        new_score = self._calculate_bic_score(data, new_graph)
        return new_score - old_score
    
    def _calculate_bic_score(self, data: np.ndarray, adjacency: np.ndarray) -> float:
        """Calculate BIC score for graph structure"""
        n_samples, n_vars = data.shape
        
        total_score = 0
        
        for j in range(n_vars):
            # Find parents of variable j
            parents = [i for i in range(n_vars) if adjacency[i, j] == 1]
            
            if len(parents) == 0:
                # No parents - just use sample variance
                variance = np.var(data[:, j])
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * n_samples
                n_params = 1
            else:
                # Linear regression with parents
                try:
                    reg = LinearRegression().fit(data[:, parents], data[:, j])
                    predictions = reg.predict(data[:, parents])
                    mse = mean_squared_error(data[:, j], predictions)
                    
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * mse) - 0.5 * n_samples
                    n_params = len(parents) + 1  # coefficients + intercept
                except Exception:
                    # Fallback to no parents case
                    variance = np.var(data[:, j])
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * n_samples
                    n_params = 1
            
            # BIC = log-likelihood - 0.5 * k * log(n)
            bic_contribution = log_likelihood - 0.5 * n_params * np.log(n_samples)
            total_score += bic_contribution
        
        return total_score
    
    def _adjacency_to_causal_graph(self, adjacency: np.ndarray, 
                                  original_data: np.ndarray,
                                  algorithm: CausalAlgorithm) -> CausalGraph:
        """Convert adjacency matrix to CausalGraph object"""
        n_vars = adjacency.shape[0]
        nodes = [f"var_{i}" for i in range(n_vars)]
        edges = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency[i, j] == 1:
                    # Calculate edge statistics
                    source_data = original_data[:, i]
                    target_data = original_data[:, j]
                    
                    # Calculate correlation and p-value
                    corr, p_value = stats.pearsonr(source_data, target_data)
                    
                    # Estimate causal effect size using linear regression
                    try:
                        reg = LinearRegression().fit(source_data.reshape(-1, 1), target_data)
                        effect_size = reg.coef_[0]
                        strength = min(abs(corr), 1.0)
                    except Exception:
                        effect_size = corr
                        strength = abs(corr)
                    
                    # Determine causal relationship type (simplified heuristic)
                    causal_type = self._infer_causal_type(nodes[i], nodes[j], original_data)
                    
                    edge = CausalEdge(
                        source=nodes[i],
                        target=nodes[j],
                        causal_type=causal_type,
                        strength=strength,
                        p_value=p_value,
                        effect_size=effect_size
                    )
                    
                    edges.append(edge)
        
        return CausalGraph(
            nodes=nodes,
            edges=edges,
            adjacency_matrix=adjacency,
            algorithm_used=algorithm,
            confidence_threshold=self.config['confidence_threshold']
        )
    
    def _infer_causal_type(self, source: str, target: str, data: np.ndarray) -> GeologicalCausalRelationType:
        """Infer causal relationship type from variable names (simplified heuristic)"""
        source_lower = source.lower()
        target_lower = target.lower()
        
        # Temporal indicators
        temporal_keywords = ['time', 'age', 'temporal', 'sequence', 'before', 'after']
        if any(keyword in source_lower or keyword in target_lower for keyword in temporal_keywords):
            return GeologicalCausalRelationType.TEMPORAL
        
        # Spatial indicators
        spatial_keywords = ['spatial', 'distance', 'proximity', 'location', 'coordinate']
        if any(keyword in source_lower or keyword in target_lower for keyword in spatial_keywords):
            return GeologicalCausalRelationType.SPATIAL
        
        # Process indicators
        process_keywords = ['process', 'rate', 'flux', 'flow', 'transport', 'reaction']
        if any(keyword in source_lower or keyword in target_lower for keyword in process_keywords):
            return GeologicalCausalRelationType.PROCESS
        
        # Structural indicators
        structural_keywords = ['structure', 'fault', 'fold', 'fracture', 'deformation']
        if any(keyword in source_lower or keyword in target_lower for keyword in structural_keywords):
            return GeologicalCausalRelationType.STRUCTURAL
        
        # Compositional indicators
        compositional_keywords = ['composition', 'chemical', 'mineral', 'element', 'oxide']
        if any(keyword in source_lower or keyword in target_lower for keyword in compositional_keywords):
            return GeologicalCausalRelationType.COMPOSITIONAL
        
        # Default to process causation
        return GeologicalCausalRelationType.PROCESS
    
    def _validate_causal_graph(self, causal_graph: CausalGraph, 
                              original_data: pd.DataFrame) -> CausalGraph:
        """Validate causal graph using domain knowledge"""
        validated_edges = []
        
        for edge in causal_graph.edges:
            validation_results = self.domain_validator.validate_causal_relationship(
                edge, original_data
            )
            
            # Update edge with validation results
            edge.domain_validation = validation_results
            
            # Only keep edges that pass validation threshold
            if validation_results['overall_score'] >= 0.5:
                validated_edges.append(edge)
        
        # Update causal graph with validated edges
        causal_graph.edges = validated_edges
        
        # Update adjacency matrix
        validated_adjacency = np.zeros_like(causal_graph.adjacency_matrix)
        node_to_idx = {node: i for i, node in enumerate(causal_graph.nodes)}
        
        for edge in validated_edges:
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            validated_adjacency[i, j] = 1
        
        causal_graph.adjacency_matrix = validated_adjacency
        
        return causal_graph
    
    async def perform_intervention_analysis(self, causal_graph: CausalGraph,
                                          data: pd.DataFrame,
                                          intervention_variable: str,
                                          intervention_values: List[float]) -> Dict[str, Any]:
        """Perform intervention analysis to understand causal effects"""
        
        intervention_results = {
            'intervention_variable': intervention_variable,
            'intervention_values': intervention_values,
            'causal_effects': {},
            'downstream_effects': {},
            'confidence_intervals': {}
        }
        
        # Find all variables causally downstream from intervention variable
        downstream_vars = self._find_downstream_variables(causal_graph, intervention_variable)
        
        for target_var in downstream_vars:
            causal_effects = []
            
            # Estimate causal effect for each intervention value
            for intervention_value in intervention_values:
                effect = await self._estimate_causal_effect(
                    causal_graph, data, intervention_variable, target_var, intervention_value
                )
                causal_effects.append(effect)
            
            intervention_results['causal_effects'][target_var] = causal_effects
            
            # Calculate confidence intervals using bootstrap
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                causal_graph, data, intervention_variable, target_var
            )
            intervention_results['confidence_intervals'][target_var] = (ci_lower, ci_upper)
        
        return intervention_results
    
    def _find_downstream_variables(self, causal_graph: CausalGraph, 
                                  source_var: str) -> List[str]:
        """Find all variables causally downstream from source variable"""
        downstream = set()
        
        # Use graph traversal to find all reachable nodes
        def dfs_downstream(current_var):
            for edge in causal_graph.edges:
                if edge.source == current_var and edge.target not in downstream:
                    downstream.add(edge.target)
                    dfs_downstream(edge.target)
        
        dfs_downstream(source_var)
        return list(downstream)
    
    async def _estimate_causal_effect(self, causal_graph: CausalGraph,
                                    data: pd.DataFrame,
                                    intervention_var: str,
                                    target_var: str,
                                    intervention_value: float) -> float:
        """Estimate causal effect of intervention"""
        
        # Find path from intervention to target
        causal_path = self._find_causal_path(causal_graph, intervention_var, target_var)
        
        if not causal_path:
            return 0.0
        
        # Calculate total causal effect along path
        total_effect = 1.0
        
        for i in range(len(causal_path) - 1):
            source = causal_path[i]
            target = causal_path[i + 1]
            
            # Find edge effect size
            edge_effect = 0.0
            for edge in causal_graph.edges:
                if edge.source == source and edge.target == target:
                    edge_effect = edge.effect_size
                    break
            
            total_effect *= edge_effect
        
        # Calculate expected change in target variable
        if intervention_var in data.columns and target_var in data.columns:
            baseline_mean = data[target_var].mean()
            intervention_change = intervention_value - data[intervention_var].mean()
            expected_effect = total_effect * intervention_change
            
            return expected_effect
        
        return total_effect
    
    def _find_causal_path(self, causal_graph: CausalGraph, 
                         source: str, target: str) -> List[str]:
        """Find causal path from source to target variable"""
        
        # Build adjacency dictionary
        adjacency_dict = {}
        for edge in causal_graph.edges:
            if edge.source not in adjacency_dict:
                adjacency_dict[edge.source] = []
            adjacency_dict[edge.source].append(edge.target)
        
        # BFS to find shortest path
        from collections import deque
        
        queue = deque([(source, [source])])
        visited = set()
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in adjacency_dict:
                for neighbor in adjacency_dict[current]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return []  # No path found
    
    def _bootstrap_confidence_interval(self, causal_graph: CausalGraph,
                                     data: pd.DataFrame,
                                     intervention_var: str,
                                     target_var: str,
                                     alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for causal effect"""
        
        bootstrap_effects = []
        n_bootstrap = 100
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(len(data), len(data), replace=True)
            bootstrap_data = data.iloc[bootstrap_indices]
            
            # Estimate effect on bootstrap sample
            try:
                effect = asyncio.run(self._estimate_causal_effect(
                    causal_graph, bootstrap_data, intervention_var, target_var, 0.0
                ))
                bootstrap_effects.append(effect)
            except:
                continue
        
        if len(bootstrap_effects) == 0:
            return 0.0, 0.0
        
        # Calculate confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_effects, lower_percentile)
        ci_upper = np.percentile(bootstrap_effects, upper_percentile)
        
        return ci_lower, ci_upper
    
    def export_causal_graph(self, causal_graph: CausalGraph, 
                           format: str = 'networkx') -> Union[nx.DiGraph, Dict[str, Any]]:
        """Export causal graph in specified format"""
        
        if format == 'networkx':
            # Export as NetworkX DiGraph
            G = nx.DiGraph()
            
            # Add nodes
            G.add_nodes_from(causal_graph.nodes)
            
            # Add edges with attributes
            for edge in causal_graph.edges:
                G.add_edge(
                    edge.source,
                    edge.target,
                    strength=edge.strength,
                    p_value=edge.p_value,
                    effect_size=edge.effect_size,
                    causal_type=edge.causal_type.value,
                    domain_validation=edge.domain_validation
                )
            
            return G
            
        elif format == 'json':
            # Export as JSON
            graph_dict = {
                'nodes': causal_graph.nodes,
                'edges': [
                    {
                        'source': edge.source,
                        'target': edge.target,
                        'causal_type': edge.causal_type.value,
                        'strength': edge.strength,
                        'p_value': edge.p_value,
                        'effect_size': edge.effect_size,
                        'confounders': edge.confounders,
                        'mediators': edge.mediators,
                        'domain_validation': edge.domain_validation
                    }
                    for edge in causal_graph.edges
                ],
                'algorithm_used': causal_graph.algorithm_used.value,
                'confidence_threshold': causal_graph.confidence_threshold,
                'adjacency_matrix': causal_graph.adjacency_matrix.tolist()
            }
            
            return graph_dict
        
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Example usage and testing
async def example_causal_discovery():
    """Example of geological causal discovery"""
    
    # Create synthetic geological dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic geological variables
    data = {
        'temperature': np.random.normal(800, 100, n_samples),  # °C
        'pressure': np.random.normal(5, 1, n_samples),  # GPa
        'silica_content': np.random.normal(65, 10, n_samples),  # wt%
        'water_content': np.random.normal(3, 1, n_samples),  # wt%
        'melt_viscosity': np.zeros(n_samples),
        'eruption_intensity': np.zeros(n_samples)
    }
    
    # Create causal relationships
    for i in range(n_samples):
        # Viscosity depends on temperature, silica, and water content
        data['melt_viscosity'][i] = (
            -0.01 * data['temperature'][i] +
            0.1 * data['silica_content'][i] +
            -0.5 * data['water_content'][i] +
            np.random.normal(0, 0.5)
        )
        
        # Eruption intensity depends on pressure and viscosity
        data['eruption_intensity'][i] = (
            0.3 * data['pressure'][i] +
            -0.2 * data['melt_viscosity'][i] +
            np.random.normal(0, 0.3)
        )
    
    df = pd.DataFrame(data)
    
    print("Created synthetic geological dataset with causal relationships")
    print(f"Dataset shape: {df.shape}")
    
    # Initialize causal discovery engine
    config = {
        'algorithms': [CausalAlgorithm.PC, CausalAlgorithm.GES],
        'confidence_threshold': 0.05,
        'domain_validation': True
    }
    
    engine = CausalDiscoveryEngine(config)
    
    # Discover causal structure
    causal_graphs = await engine.discover_causal_structure(df)
    
    print(f"\nDiscovered {len(causal_graphs)} causal graphs")
    
    for algorithm, graph in causal_graphs.items():
        print(f"\n{algorithm.value.upper()} Algorithm Results:")
        print(f"  - Nodes: {len(graph.nodes)}")
        print(f"  - Edges: {len(graph.edges)}")
        
        # Show strongest causal relationships
        strong_edges = [edge for edge in graph.edges if edge.strength > 0.3]
        print(f"  - Strong causal relationships ({len(strong_edges)}):")
        
        for edge in strong_edges:
            print(f"    {edge.source} -> {edge.target}")
            print(f"      Strength: {edge.strength:.3f}, p-value: {edge.p_value:.3f}")
            print(f"      Type: {edge.causal_type.value}")
            if edge.domain_validation:
                print(f"      Domain validation score: {edge.domain_validation.get('overall_score', 0):.3f}")
    
    # Perform intervention analysis
    if causal_graphs:
        first_graph = list(causal_graphs.values())[0]
        
        print("\nPerforming intervention analysis...")
        intervention_results = await engine.perform_intervention_analysis(
            first_graph, df, 'temperature', [700, 800, 900, 1000]
        )
        
        print("Intervention Analysis Results:")
        for target_var, effects in intervention_results['causal_effects'].items():
            print(f"  Effect on {target_var}: {effects}")

if __name__ == "__main__":
    print(f"Causal-learn available: {CAUSAL_LEARN_AVAILABLE}")
    print(f"pgmpy available: {PGMPY_AVAILABLE}")
    
    # Run example
    asyncio.run(example_causal_discovery())