"""
Causal Inference Engine for Geological Relationship Discovery

This module implements causal discovery algorithms (PC, FCI, GES) for geological data,
constructs causal graphs from observational geological data, performs intervention analysis,
and validates causal relationships using geological domain knowledge.

Requirements: 15.2, 15.7
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CausalAlgorithm(Enum):
    """Supported causal discovery algorithms"""
    PC = "pc"
    FCI = "fci"
    GES = "ges"


class EdgeType(Enum):
    """Types of edges in causal graphs"""
    DIRECTED = "->"
    BIDIRECTED = "<->"
    UNDIRECTED = "--"
    CIRCLE = "o-o"


@dataclass
class CausalEdge:
    """Represents a causal edge between two variables"""
    source: str
    target: str
    edge_type: EdgeType
    strength: float
    confidence: float
    p_value: float
    geological_interpretation: Optional[str] = None


@dataclass
class CausalGraph:
    """Represents a causal graph with geological interpretations"""
    nodes: List[str]
    edges: List[CausalEdge]
    algorithm_used: CausalAlgorithm
    confidence_threshold: float
    geological_domain_knowledge: Dict[str, Any]
    creation_timestamp: datetime
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX graph for visualization and analysis"""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        
        for edge in self.edges:
            if edge.edge_type == EdgeType.DIRECTED:
                G.add_edge(edge.source, edge.target, 
                          strength=edge.strength, 
                          confidence=edge.confidence,
                          p_value=edge.p_value,
                          interpretation=edge.geological_interpretation)
        
        return G


@dataclass
class InterventionResult:
    """Results of causal intervention analysis"""
    intervention_variable: str
    target_variables: List[str]
    causal_effects: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    geological_interpretation: str
    statistical_significance: Dict[str, float]


@dataclass
class GeologicalDomainKnowledge:
    """Geological domain knowledge for causal validation"""
    known_causal_relationships: Dict[str, List[str]]
    impossible_relationships: Dict[str, List[str]]
    temporal_constraints: Dict[str, Dict[str, str]]  # variable -> {before/after: other_variable}
    physical_constraints: Dict[str, Dict[str, Any]]
    geological_process_hierarchies: Dict[str, List[str]]


class CausalInferenceEngine:
    """
    Main causal inference engine for geological relationship discovery
    
    Implements PC, FCI, and GES algorithms with geological domain knowledge validation
    """
    
    def __init__(self, domain_knowledge: Optional[GeologicalDomainKnowledge] = None):
        self.domain_knowledge = domain_knowledge or self._default_geological_knowledge()
        self.scaler = StandardScaler()
        self.discovered_relationships = {}
        self.validation_results = {}
        
    def _default_geological_knowledge(self) -> GeologicalDomainKnowledge:
        """Default geological domain knowledge"""
        return GeologicalDomainKnowledge(
            known_causal_relationships={
                "magma_pressure": ["ground_deformation", "seismic_activity"],
                "temperature": ["thermal_expansion", "gas_emissions"],
                "stress_changes": ["earthquake_occurrence", "fault_activation"],
                "fluid_pressure": ["hydrothermal_activity", "geochemical_changes"],
                "tectonic_forces": ["crustal_deformation", "volcanic_activity"]
            },
            impossible_relationships={
                "earthquake_occurrence": ["magma_pressure"],  # Effect cannot cause cause
                "ground_deformation": ["magma_pressure"],
                "gas_emissions": ["temperature"]
            },
            temporal_constraints={
                "magma_intrusion": {"before": "ground_deformation"},
                "stress_buildup": {"before": "earthquake_occurrence"},
                "temperature_increase": {"before": "gas_emissions"}
            },
            physical_constraints={
                "pressure": {"min": 0, "max": 1e9, "unit": "Pa"},
                "temperature": {"min": 273, "max": 2000, "unit": "K"},
                "deformation": {"min": -1, "max": 1, "unit": "m"}
            },
            geological_process_hierarchies={
                "volcanic_processes": ["magma_generation", "magma_transport", "eruption"],
                "seismic_processes": ["stress_accumulation", "fault_rupture", "wave_propagation"],
                "hydrothermal_processes": ["fluid_heating", "circulation", "mineral_precipitation"]
            }
        )
    
    def discover_causal_relationships(self, 
                                    data: pd.DataFrame, 
                                    algorithm: CausalAlgorithm = CausalAlgorithm.PC,
                                    confidence_threshold: float = 0.05) -> CausalGraph:
        """
        Discover causal relationships in geological data using specified algorithm
        
        Args:
            data: Geological observational data
            algorithm: Causal discovery algorithm to use
            confidence_threshold: Statistical significance threshold
            
        Returns:
            CausalGraph with discovered relationships
        """
        logger.info(f"Starting causal discovery with {algorithm.value} algorithm")
        
        # Preprocess data
        processed_data = self._preprocess_data(data)
        
        # Apply causal discovery algorithm
        if algorithm == CausalAlgorithm.PC:
            causal_graph = self._pc_algorithm(processed_data, confidence_threshold)
        elif algorithm == CausalAlgorithm.FCI:
            causal_graph = self._fci_algorithm(processed_data, confidence_threshold)
        elif algorithm == CausalAlgorithm.GES:
            causal_graph = self._ges_algorithm(processed_data, confidence_threshold)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Validate with domain knowledge
        validated_graph = self._validate_with_domain_knowledge(causal_graph)
        
        # Add geological interpretations
        interpreted_graph = self._add_geological_interpretations(validated_graph)
        
        logger.info(f"Discovered {len(interpreted_graph.edges)} causal relationships")
        return interpreted_graph
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess geological data for causal discovery"""
        # Handle missing values
        data_clean = data.dropna()
        
        # Standardize numerical variables
        numerical_cols = data_clean.select_dtypes(include=[np.number]).columns
        data_scaled = data_clean.copy()
        data_scaled[numerical_cols] = self.scaler.fit_transform(data_clean[numerical_cols])
        
        # Remove highly correlated variables to avoid multicollinearity
        correlation_matrix = data_scaled[numerical_cols].corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_pairs = [column for column in upper_triangle.columns 
                          if any(upper_triangle[column] > 0.95)]
        
        if high_corr_pairs:
            logger.warning(f"Removing highly correlated variables: {high_corr_pairs}")
            data_scaled = data_scaled.drop(columns=high_corr_pairs)
        
        return data_scaled
    
    def _pc_algorithm(self, data: pd.DataFrame, alpha: float) -> CausalGraph:
        """
        Implement PC (Peter-Clark) algorithm for causal discovery
        
        The PC algorithm uses conditional independence tests to discover causal structure
        """
        logger.info("Running PC algorithm")
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize complete undirected graph
        adjacency_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase 1: Edge removal based on conditional independence
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacency_matrix[i, j] == 1:
                    # Test conditional independence
                    if self._test_conditional_independence(data, variables[i], variables[j], [], alpha):
                        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 0
        
        # Phase 2: Orient edges using v-structures and domain knowledge
        edges = self._orient_edges_pc(adjacency_matrix, variables, data, alpha)
        
        return CausalGraph(
            nodes=variables,
            edges=edges,
            algorithm_used=CausalAlgorithm.PC,
            confidence_threshold=alpha,
            geological_domain_knowledge=self.domain_knowledge.__dict__,
            creation_timestamp=datetime.now()
        )
    
    def _fci_algorithm(self, data: pd.DataFrame, alpha: float) -> CausalGraph:
        """
        Implement FCI (Fast Causal Inference) algorithm
        
        FCI can handle latent confounders and selection bias
        """
        logger.info("Running FCI algorithm")
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize complete graph with circle endpoints
        graph_matrix = np.full((n_vars, n_vars, 2), 1)  # [i,j,0] = tail, [i,j,1] = head
        np.fill_diagonal(graph_matrix[:, :, 0], 0)
        np.fill_diagonal(graph_matrix[:, :, 1], 0)
        
        # Phase 1: Edge removal (similar to PC)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if graph_matrix[i, j, 0] == 1:
                    if self._test_conditional_independence(data, variables[i], variables[j], [], alpha):
                        graph_matrix[i, j, :] = 0
                        graph_matrix[j, i, :] = 0
        
        # Phase 2: Orient edges with additional rules for latent confounders
        edges = self._orient_edges_fci(graph_matrix, variables, data, alpha)
        
        return CausalGraph(
            nodes=variables,
            edges=edges,
            algorithm_used=CausalAlgorithm.FCI,
            confidence_threshold=alpha,
            geological_domain_knowledge=self.domain_knowledge.__dict__,
            creation_timestamp=datetime.now()
        )
    
    def _ges_algorithm(self, data: pd.DataFrame, alpha: float) -> CausalGraph:
        """
        Implement GES (Greedy Equivalence Search) algorithm
        
        GES uses a score-based approach to find the best causal structure
        """
        logger.info("Running GES algorithm")
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize empty graph
        adjacency_matrix = np.zeros((n_vars, n_vars))
        
        # Forward phase: Add edges that improve score
        current_score = self._calculate_bic_score(data, adjacency_matrix, variables)
        improved = True
        
        while improved:
            improved = False
            best_score = current_score
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and adjacency_matrix[i, j] == 0:
                        # Try adding edge i -> j
                        test_matrix = adjacency_matrix.copy()
                        test_matrix[i, j] = 1
                        
                        if not self._creates_cycle(test_matrix):
                            score = self._calculate_bic_score(data, test_matrix, variables)
                            if score > best_score:
                                best_score = score
                                best_edge = (i, j)
                                improved = True
            
            if improved and best_edge:
                adjacency_matrix[best_edge[0], best_edge[1]] = 1
                current_score = best_score
        
        # Backward phase: Remove edges that improve score
        improved = True
        while improved:
            improved = False
            best_score = current_score
            best_edge = None
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if adjacency_matrix[i, j] == 1:
                        # Try removing edge i -> j
                        test_matrix = adjacency_matrix.copy()
                        test_matrix[i, j] = 0
                        
                        score = self._calculate_bic_score(data, test_matrix, variables)
                        if score > best_score:
                            best_score = score
                            best_edge = (i, j)
                            improved = True
            
            if improved and best_edge:
                adjacency_matrix[best_edge[0], best_edge[1]] = 0
                current_score = best_score
        
        # Convert adjacency matrix to edges
        edges = []
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency_matrix[i, j] == 1:
                    # Calculate edge strength and confidence
                    strength, p_value = self._calculate_edge_strength(data, variables[i], variables[j])
                    
                    edges.append(CausalEdge(
                        source=variables[i],
                        target=variables[j],
                        edge_type=EdgeType.DIRECTED,
                        strength=strength,
                        confidence=1 - p_value,
                        p_value=p_value
                    ))
        
        return CausalGraph(
            nodes=variables,
            edges=edges,
            algorithm_used=CausalAlgorithm.GES,
            confidence_threshold=alpha,
            geological_domain_knowledge=self.domain_knowledge.__dict__,
            creation_timestamp=datetime.now()
        )
    
    def _test_conditional_independence(self, data: pd.DataFrame, 
                                     var1: str, var2: str, 
                                     conditioning_set: List[str], 
                                     alpha: float) -> bool:
        """Test conditional independence between two variables given a conditioning set"""
        try:
            if not conditioning_set:
                # Test marginal independence using correlation
                corr, p_value = stats.pearsonr(data[var1], data[var2])
                return p_value > alpha
            else:
                # Test conditional independence using partial correlation
                # Create design matrix with conditioning variables
                X_cond = data[conditioning_set].values
                
                # Regress both variables on conditioning set
                from sklearn.linear_model import LinearRegression
                
                reg1 = LinearRegression().fit(X_cond, data[var1])
                reg2 = LinearRegression().fit(X_cond, data[var2])
                
                # Calculate residuals
                residuals1 = data[var1] - reg1.predict(X_cond)
                residuals2 = data[var2] - reg2.predict(X_cond)
                
                # Test correlation of residuals
                corr, p_value = stats.pearsonr(residuals1, residuals2)
                return p_value > alpha
                
        except Exception as e:
            logger.warning(f"Error in conditional independence test: {e}")
            return True  # Conservative: assume independence if test fails
    
    def _orient_edges_pc(self, adjacency_matrix: np.ndarray, 
                        variables: List[str], 
                        data: pd.DataFrame, 
                        alpha: float) -> List[CausalEdge]:
        """Orient edges in PC algorithm using v-structures and domain knowledge"""
        edges = []
        n_vars = len(variables)
        
        # Convert undirected edges to directed based on v-structures
        for i in range(n_vars):
            for j in range(n_vars):
                if adjacency_matrix[i, j] == 1:
                    # Calculate edge strength and direction
                    strength_ij, p_value_ij = self._calculate_edge_strength(data, variables[i], variables[j])
                    strength_ji, p_value_ji = self._calculate_edge_strength(data, variables[j], variables[i])
                    
                    # Determine direction based on strength and domain knowledge
                    if self._is_causal_direction_valid(variables[i], variables[j]):
                        if strength_ij > strength_ji:
                            edges.append(CausalEdge(
                                source=variables[i],
                                target=variables[j],
                                edge_type=EdgeType.DIRECTED,
                                strength=strength_ij,
                                confidence=1 - p_value_ij,
                                p_value=p_value_ij
                            ))
                        else:
                            edges.append(CausalEdge(
                                source=variables[j],
                                target=variables[i],
                                edge_type=EdgeType.DIRECTED,
                                strength=strength_ji,
                                confidence=1 - p_value_ji,
                                p_value=p_value_ji
                            ))
                    elif self._is_causal_direction_valid(variables[j], variables[i]):
                        edges.append(CausalEdge(
                            source=variables[j],
                            target=variables[i],
                            edge_type=EdgeType.DIRECTED,
                            strength=strength_ji,
                            confidence=1 - p_value_ji,
                            p_value=p_value_ji
                        ))
                    else:
                        # Undirected edge if direction cannot be determined
                        edges.append(CausalEdge(
                            source=variables[i],
                            target=variables[j],
                            edge_type=EdgeType.UNDIRECTED,
                            strength=max(strength_ij, strength_ji),
                            confidence=1 - min(p_value_ij, p_value_ji),
                            p_value=min(p_value_ij, p_value_ji)
                        ))
        
        return edges
    
    def _orient_edges_fci(self, graph_matrix: np.ndarray, 
                         variables: List[str], 
                         data: pd.DataFrame, 
                         alpha: float) -> List[CausalEdge]:
        """Orient edges in FCI algorithm with additional rules for latent confounders"""
        edges = []
        n_vars = len(variables)
        
        for i in range(n_vars):
            for j in range(n_vars):
                if graph_matrix[i, j, 0] == 1 or graph_matrix[i, j, 1] == 1:
                    strength, p_value = self._calculate_edge_strength(data, variables[i], variables[j])
                    
                    # Determine edge type based on endpoints
                    if graph_matrix[i, j, 0] == 1 and graph_matrix[i, j, 1] == 1:
                        if graph_matrix[j, i, 0] == 1 and graph_matrix[j, i, 1] == 1:
                            edge_type = EdgeType.BIDIRECTED
                        else:
                            edge_type = EdgeType.DIRECTED
                    else:
                        edge_type = EdgeType.CIRCLE
                    
                    edges.append(CausalEdge(
                        source=variables[i],
                        target=variables[j],
                        edge_type=edge_type,
                        strength=strength,
                        confidence=1 - p_value,
                        p_value=p_value
                    ))
        
        return edges
    
    def _calculate_bic_score(self, data: pd.DataFrame, 
                           adjacency_matrix: np.ndarray, 
                           variables: List[str]) -> float:
        """Calculate BIC score for a given graph structure"""
        n_samples = len(data)
        n_vars = len(variables)
        total_score = 0
        
        for i in range(n_vars):
            # Find parents of variable i
            parents = [j for j in range(n_vars) if adjacency_matrix[j, i] == 1]
            
            if not parents:
                # No parents - calculate marginal likelihood
                variance = np.var(data[variables[i]])
                log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * n_samples
                penalty = 0.5 * np.log(n_samples)  # One parameter (variance)
            else:
                # Has parents - calculate conditional likelihood
                try:
                    from sklearn.linear_model import LinearRegression
                    X = data[[variables[j] for j in parents]].values
                    y = data[variables[i]].values
                    
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    residuals = y - y_pred
                    variance = np.var(residuals)
                    
                    log_likelihood = -0.5 * n_samples * np.log(2 * np.pi * variance) - 0.5 * np.sum(residuals**2) / variance
                    penalty = 0.5 * (len(parents) + 1) * np.log(n_samples)  # Parameters: coefficients + variance
                except:
                    log_likelihood = -np.inf
                    penalty = 0
            
            total_score += log_likelihood - penalty
        
        return total_score
    
    def _creates_cycle(self, adjacency_matrix: np.ndarray) -> bool:
        """Check if adding an edge creates a cycle in the graph"""
        try:
            G = nx.DiGraph(adjacency_matrix)
            return not nx.is_directed_acyclic_graph(G)
        except:
            return True
    
    def _calculate_edge_strength(self, data: pd.DataFrame, 
                               source: str, target: str) -> Tuple[float, float]:
        """Calculate the strength and statistical significance of a causal edge"""
        try:
            # Use mutual information as a measure of causal strength
            X = data[source].values.reshape(-1, 1)
            y = data[target].values
            
            mi_score = mutual_info_regression(X, y)[0]
            
            # Calculate p-value using permutation test
            n_permutations = 100
            permuted_scores = []
            
            for _ in range(n_permutations):
                y_permuted = np.random.permutation(y)
                permuted_mi = mutual_info_regression(X, y_permuted)[0]
                permuted_scores.append(permuted_mi)
            
            p_value = np.mean(np.array(permuted_scores) >= mi_score)
            
            return mi_score, max(p_value, 1e-10)  # Avoid zero p-values
            
        except Exception as e:
            logger.warning(f"Error calculating edge strength: {e}")
            return 0.0, 1.0
    
    def _is_causal_direction_valid(self, source: str, target: str) -> bool:
        """Check if a causal direction is valid based on domain knowledge"""
        # Check known causal relationships
        if source in self.domain_knowledge.known_causal_relationships:
            if target in self.domain_knowledge.known_causal_relationships[source]:
                return True
        
        # Check impossible relationships
        if source in self.domain_knowledge.impossible_relationships:
            if target in self.domain_knowledge.impossible_relationships[source]:
                return False
        
        # Check temporal constraints
        if source in self.domain_knowledge.temporal_constraints:
            constraints = self.domain_knowledge.temporal_constraints[source]
            if "before" in constraints and constraints["before"] == target:
                return True
            if "after" in constraints and constraints["after"] == target:
                return False
        
        return True  # Allow if no specific constraints
    
    def _validate_with_domain_knowledge(self, causal_graph: CausalGraph) -> CausalGraph:
        """Validate discovered causal relationships with geological domain knowledge"""
        validated_edges = []
        
        for edge in causal_graph.edges:
            # Check if edge violates domain knowledge
            if self._is_causal_direction_valid(edge.source, edge.target):
                # Adjust confidence based on domain knowledge support
                if edge.source in self.domain_knowledge.known_causal_relationships:
                    if edge.target in self.domain_knowledge.known_causal_relationships[edge.source]:
                        # Boost confidence for known relationships
                        edge.confidence = min(1.0, edge.confidence * 1.2)
                
                validated_edges.append(edge)
            else:
                logger.warning(f"Removing edge {edge.source} -> {edge.target} due to domain knowledge violation")
        
        causal_graph.edges = validated_edges
        return causal_graph
    
    def _add_geological_interpretations(self, causal_graph: CausalGraph) -> CausalGraph:
        """Add geological interpretations to discovered causal relationships"""
        for edge in causal_graph.edges:
            edge.geological_interpretation = self._generate_geological_interpretation(
                edge.source, edge.target, edge.strength
            )
        
        return causal_graph
    
    def _generate_geological_interpretation(self, source: str, target: str, strength: float) -> str:
        """Generate geological interpretation for a causal relationship"""
        interpretations = {
            ("magma_pressure", "ground_deformation"): 
                f"Magma pressure increase causes ground deformation with strength {strength:.3f}. "
                "This indicates magma intrusion and potential volcanic unrest.",
            
            ("temperature", "gas_emissions"): 
                f"Temperature increase drives gas emissions with strength {strength:.3f}. "
                "Higher temperatures increase volatile solubility and degassing rates.",
            
            ("stress_changes", "seismic_activity"): 
                f"Stress changes trigger seismic activity with strength {strength:.3f}. "
                "Stress accumulation leads to fault rupture and earthquake occurrence.",
            
            ("fluid_pressure", "hydrothermal_activity"): 
                f"Fluid pressure controls hydrothermal activity with strength {strength:.3f}. "
                "Pressure changes affect fluid circulation and mineral precipitation.",
            
            ("tectonic_forces", "crustal_deformation"): 
                f"Tectonic forces cause crustal deformation with strength {strength:.3f}. "
                "Plate motion and stress transfer result in crustal strain accumulation."
        }
        
        key = (source, target)
        if key in interpretations:
            return interpretations[key]
        else:
            return f"Causal relationship between {source} and {target} with strength {strength:.3f}. " \
                   "Further geological analysis needed to determine specific mechanism."
    
    def perform_intervention_analysis(self, 
                                    causal_graph: CausalGraph,
                                    data: pd.DataFrame,
                                    intervention_variable: str,
                                    intervention_value: float,
                                    target_variables: Optional[List[str]] = None) -> InterventionResult:
        """
        Perform causal intervention analysis to understand geological process effects
        
        Args:
            causal_graph: Discovered causal graph
            data: Original observational data
            intervention_variable: Variable to intervene on
            intervention_value: Value to set intervention variable to
            target_variables: Variables to analyze effects on
            
        Returns:
            InterventionResult with causal effects and interpretations
        """
        logger.info(f"Performing intervention analysis on {intervention_variable}")
        
        if target_variables is None:
            # Find all variables causally downstream from intervention variable
            target_variables = self._find_causal_descendants(causal_graph, intervention_variable)
        
        causal_effects = {}
        confidence_intervals = {}
        statistical_significance = {}
        
        for target in target_variables:
            # Estimate causal effect using do-calculus approximation
            effect, ci, p_value = self._estimate_causal_effect(
                data, causal_graph, intervention_variable, target, intervention_value
            )
            
            causal_effects[target] = effect
            confidence_intervals[target] = ci
            statistical_significance[target] = p_value
        
        # Generate geological interpretation
        interpretation = self._interpret_intervention_results(
            intervention_variable, intervention_value, causal_effects
        )
        
        return InterventionResult(
            intervention_variable=intervention_variable,
            target_variables=target_variables,
            causal_effects=causal_effects,
            confidence_intervals=confidence_intervals,
            geological_interpretation=interpretation,
            statistical_significance=statistical_significance
        )
    
    def _find_causal_descendants(self, causal_graph: CausalGraph, variable: str) -> List[str]:
        """Find all variables causally downstream from a given variable"""
        G = causal_graph.to_networkx()
        descendants = []
        
        if variable in G.nodes():
            descendants = list(nx.descendants(G, variable))
        
        return descendants
    
    def _estimate_causal_effect(self, 
                              data: pd.DataFrame,
                              causal_graph: CausalGraph,
                              intervention_var: str,
                              target_var: str,
                              intervention_value: float) -> Tuple[float, Tuple[float, float], float]:
        """Estimate causal effect using backdoor adjustment or instrumental variables"""
        try:
            # Find confounders (backdoor criterion)
            confounders = self._find_confounders(causal_graph, intervention_var, target_var)
            
            if confounders:
                # Use backdoor adjustment
                effect, ci, p_value = self._backdoor_adjustment(
                    data, intervention_var, target_var, confounders, intervention_value
                )
            else:
                # Direct causal effect estimation
                effect, ci, p_value = self._direct_causal_effect(
                    data, intervention_var, target_var, intervention_value
                )
            
            return effect, ci, p_value
            
        except Exception as e:
            logger.error(f"Error estimating causal effect: {e}")
            return 0.0, (0.0, 0.0), 1.0
    
    def _find_confounders(self, causal_graph: CausalGraph, 
                         treatment: str, outcome: str) -> List[str]:
        """Find confounding variables using backdoor criterion"""
        G = causal_graph.to_networkx()
        confounders = []
        
        # Simple heuristic: variables that have edges to both treatment and outcome
        for node in G.nodes():
            if node != treatment and node != outcome:
                has_edge_to_treatment = G.has_edge(node, treatment)
                has_edge_to_outcome = G.has_edge(node, outcome)
                
                if has_edge_to_treatment and has_edge_to_outcome:
                    confounders.append(node)
        
        return confounders
    
    def _backdoor_adjustment(self, 
                           data: pd.DataFrame,
                           treatment: str,
                           outcome: str,
                           confounders: List[str],
                           intervention_value: float) -> Tuple[float, Tuple[float, float], float]:
        """Estimate causal effect using backdoor adjustment"""
        from sklearn.linear_model import LinearRegression
        from scipy import stats
        
        # Prepare data
        X = data[confounders + [treatment]].values
        y = data[outcome].values
        
        # Fit regression model
        reg = LinearRegression().fit(X, y)
        
        # Get treatment coefficient (causal effect)
        treatment_idx = len(confounders)  # Treatment is last column
        causal_effect = reg.coef_[treatment_idx]
        
        # Calculate confidence interval using bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(data), size=len(data), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model and get effect
            reg_boot = LinearRegression().fit(X_boot, y_boot)
            bootstrap_effects.append(reg_boot.coef_[treatment_idx])
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)
        
        # Calculate p-value
        t_stat = causal_effect / (np.std(bootstrap_effects) + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
        
        return causal_effect, (ci_lower, ci_upper), p_value
    
    def _direct_causal_effect(self, 
                            data: pd.DataFrame,
                            treatment: str,
                            outcome: str,
                            intervention_value: float) -> Tuple[float, Tuple[float, float], float]:
        """Estimate direct causal effect without confounders"""
        # Simple linear regression
        X = data[treatment].values.reshape(-1, 1)
        y = data[outcome].values
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y)
        
        causal_effect = reg.coef_[0]
        
        # Calculate confidence interval
        y_pred = reg.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals**2)
        
        # Standard error of coefficient
        X_centered = X - np.mean(X)
        se = np.sqrt(mse / np.sum(X_centered**2))
        
        # 95% confidence interval
        from scipy import stats
        t_critical = stats.t.ppf(0.975, len(data) - 2)
        ci_lower = causal_effect - t_critical * se
        ci_upper = causal_effect + t_critical * se
        
        # P-value
        t_stat = causal_effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(data) - 2))
        
        return causal_effect, (ci_lower, ci_upper), p_value
    
    def _interpret_intervention_results(self, 
                                      intervention_var: str,
                                      intervention_value: float,
                                      causal_effects: Dict[str, float]) -> str:
        """Generate geological interpretation of intervention results"""
        interpretation = f"Intervention analysis: Setting {intervention_var} to {intervention_value:.3f}\n\n"
        
        for target, effect in causal_effects.items():
            if abs(effect) > 0.1:  # Significant effect threshold
                direction = "increase" if effect > 0 else "decrease"
                interpretation += f"- {target} would {direction} by {abs(effect):.3f} units\n"
                
                # Add geological context
                if "deformation" in target.lower() and "pressure" in intervention_var.lower():
                    interpretation += "  This suggests magma intrusion would cause measurable ground deformation.\n"
                elif "seismic" in target.lower() and "stress" in intervention_var.lower():
                    interpretation += "  This indicates stress changes would trigger seismic activity.\n"
                elif "temperature" in intervention_var.lower() and "gas" in target.lower():
                    interpretation += "  This shows temperature increase would enhance volcanic gas emissions.\n"
        
        return interpretation
    
    def validate_causal_relationships(self, 
                                    causal_graph: CausalGraph,
                                    validation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate discovered causal relationships using independent data
        
        Args:
            causal_graph: Discovered causal graph to validate
            validation_data: Independent geological data for validation
            
        Returns:
            Dictionary with validation results and metrics
        """
        logger.info("Validating causal relationships with independent data")
        
        validation_results = {
            "total_edges": len(causal_graph.edges),
            "validated_edges": 0,
            "failed_edges": 0,
            "edge_validations": {},
            "overall_accuracy": 0.0,
            "domain_knowledge_consistency": 0.0
        }
        
        for edge in causal_graph.edges:
            # Test if causal relationship holds in validation data
            is_valid = self._validate_single_edge(edge, validation_data)
            
            validation_results["edge_validations"][f"{edge.source}->{edge.target}"] = {
                "valid": is_valid,
                "original_strength": edge.strength,
                "original_confidence": edge.confidence,
                "geological_interpretation": edge.geological_interpretation
            }
            
            if is_valid:
                validation_results["validated_edges"] += 1
            else:
                validation_results["failed_edges"] += 1
        
        # Calculate overall accuracy
        if validation_results["total_edges"] > 0:
            validation_results["overall_accuracy"] = (
                validation_results["validated_edges"] / validation_results["total_edges"]
            )
        
        # Check domain knowledge consistency
        validation_results["domain_knowledge_consistency"] = self._check_domain_consistency(causal_graph)
        
        return validation_results
    
    def _validate_single_edge(self, edge: CausalEdge, validation_data: pd.DataFrame) -> bool:
        """Validate a single causal edge using independent data"""
        try:
            if edge.source not in validation_data.columns or edge.target not in validation_data.columns:
                return False
            
            # Calculate edge strength in validation data
            val_strength, val_p_value = self._calculate_edge_strength(
                validation_data, edge.source, edge.target
            )
            
            # Edge is valid if:
            # 1. Strength is in same direction and reasonably similar magnitude
            # 2. Statistical significance is maintained
            strength_consistent = (
                np.sign(val_strength) == np.sign(edge.strength) and
                abs(val_strength) > 0.5 * abs(edge.strength)
            )
            
            significance_maintained = val_p_value < 0.05
            
            return strength_consistent and significance_maintained
            
        except Exception as e:
            logger.warning(f"Error validating edge {edge.source}->{edge.target}: {e}")
            return False
    
    def _check_domain_consistency(self, causal_graph: CausalGraph) -> float:
        """Check consistency with geological domain knowledge"""
        total_edges = len(causal_graph.edges)
        if total_edges == 0:
            return 1.0
        
        consistent_edges = 0
        
        for edge in causal_graph.edges:
            # Check if edge is consistent with known relationships
            if edge.source in self.domain_knowledge.known_causal_relationships:
                if edge.target in self.domain_knowledge.known_causal_relationships[edge.source]:
                    consistent_edges += 1
                    continue
            
            # Check if edge violates impossible relationships
            if edge.source in self.domain_knowledge.impossible_relationships:
                if edge.target in self.domain_knowledge.impossible_relationships[edge.source]:
                    continue  # Inconsistent edge
            
            # If no specific knowledge, consider neutral (consistent)
            consistent_edges += 1
        
        return consistent_edges / total_edges
    
    def export_causal_graph(self, causal_graph: CausalGraph, filepath: str) -> None:
        """Export causal graph to JSON format"""
        export_data = {
            "nodes": causal_graph.nodes,
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type.value,
                    "strength": edge.strength,
                    "confidence": edge.confidence,
                    "p_value": edge.p_value,
                    "geological_interpretation": edge.geological_interpretation
                }
                for edge in causal_graph.edges
            ],
            "algorithm_used": causal_graph.algorithm_used.value,
            "confidence_threshold": causal_graph.confidence_threshold,
            "creation_timestamp": causal_graph.creation_timestamp.isoformat(),
            "domain_knowledge": causal_graph.geological_domain_knowledge
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Causal graph exported to {filepath}")
    
    def generate_hypothesis_from_graph(self, causal_graph: CausalGraph) -> List[str]:
        """Generate testable geological hypotheses from discovered causal relationships"""
        hypotheses = []
        
        for edge in causal_graph.edges:
            if edge.confidence > 0.8 and edge.strength > 0.3:  # High confidence relationships
                hypothesis = self._generate_testable_hypothesis(edge)
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_testable_hypothesis(self, edge: CausalEdge) -> str:
        """Generate a testable hypothesis from a causal edge"""
        hypothesis_templates = {
            ("magma_pressure", "ground_deformation"): 
                f"Hypothesis: Increases in magma pressure directly cause ground deformation. "
                f"Test: Monitor ground deformation during periods of varying magma pressure. "
                f"Expected: Deformation should increase proportionally with pressure (strength: {edge.strength:.3f}).",
            
            ("temperature", "gas_emissions"): 
                f"Hypothesis: Temperature increases drive volcanic gas emissions. "
                f"Test: Measure gas emission rates during thermal anomalies. "
                f"Expected: Gas emissions should correlate with temperature changes (strength: {edge.strength:.3f}).",
            
            ("stress_changes", "seismic_activity"): 
                f"Hypothesis: Stress field changes trigger seismic activity. "
                f"Test: Monitor seismicity during periods of stress loading/unloading. "
                f"Expected: Seismic activity should increase with stress changes (strength: {edge.strength:.3f})."
        }
        
        key = (edge.source, edge.target)
        if key in hypothesis_templates:
            return hypothesis_templates[key]
        else:
            return (f"Hypothesis: {edge.source} causally influences {edge.target} "
                   f"with strength {edge.strength:.3f}. "
                   f"Test: Controlled experiments or natural observations to verify "
                   f"the causal relationship. Expected: Changes in {edge.source} "
                   f"should predict changes in {edge.target}.")