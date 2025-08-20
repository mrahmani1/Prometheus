"""
Geological Causal Discovery Integration
Advanced causal discovery algorithms specifically designed for geological process relationships
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import matplotlib.pyplot as plt
import seaborn as sns

# Causal discovery libraries with fallbacks
try:
    from causal_learn.search.ConstraintBased.PC import pc
    from causal_learn.search.ScoreBased.GES import ges
    from causal_learn.search.ConstraintBased.FCI import fci
    from causal_learn.utils.cit import CIT
    from causal_learn.utils.PCUtils import SkeletonDiscovery
    CAUSAL_LEARN_AVAILABLE = True
except ImportError:
    CAUSAL_LEARN_AVAILABLE = False
    warnings.warn("Causal-learn not available. Advanced causal discovery will use fallback methods.")

try:
    import pgmpy
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    warnings.warn("pgmpy not available. Bayesian network features will be limited.")

try:
    from scipy import stats
    from scipy.spatial.distance import pdist, squareform
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Statistical tests will be limited.")

# Import local modules
try:
    from .causal_inference_engine import (
        CausalInferenceEngine,
        CausalGraph,
        CausalRelationship,
        InterventionEffect
    )
    from .geological_gnn import (
        GeologicalGraphDataset, GeologicalNode, GeologicalEdge
    )
    from .hidden_relationship_discovery import (
        HiddenRelationshipDiscovery,
        RelationshipType
    )
except ImportError as e:
    warnings.warn(f"Could not import local modules: {e}")

warnings.filterwarnings('ignore')

class GeologicalProcessType(Enum):
    TECTONIC = "tectonic"
    VOLCANIC = "volcanic"
    SEDIMENTARY = "sedimentary"
    METAMORPHIC = "metamorphic"
    EROSIONAL = "erosional"
    HYDROTHERMAL = "hydrothermal"
    DIAGENETIC = "diagenetic"
    MAGMATIC = "magmatic"
    STRUCTURAL = "structural"
    GEOCHEMICAL = "geochemical"

class CausalityStrength(Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class TemporalScale(Enum):
    SECONDS = "seconds"
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    MONTHS = "months"
    YEARS = "years"
    DECADES = "decades"
    CENTURIES = "centuries"
    MILLENNIA = "millennia"
    MILLIONS_OF_YEARS = "millions_of_years"

@dataclass
class GeologicalCausalConfig:
    """Configuration for geological causal discovery"""
    significance_level: float = 0.05
    max_conditioning_set_size: int = 3
    independence_test: str = 'fisherz'  # 'fisherz', 'kci', 'gsq'
    score_function: str = 'bic'  # 'bic', 'aic', 'bdeu'
    max_parents: int = 5
    temporal_resolution: TemporalScale = TemporalScale.YEARS
    spatial_resolution: float = 1000.0  # meters
    process_types: List[GeologicalProcessType] = field(default_factory=lambda: list(GeologicalProcessType))
    bootstrap_samples: int = 100
    confidence_threshold: float = 0.8
    causal_strength_threshold: float = 0.6
    use_domain_knowledge: bool = True
    enable_temporal_ordering: bool = True
    enable_spatial_constraints: bool = True

class GeologicalCausalRelationship:
    """Represents a causal relationship between geological processes"""
    
    def __init__(self, 
                 cause: str, 
                 effect: str,
                 strength: float,
                 confidence: float,
                 process_type: GeologicalProcessType,
                 temporal_delay: Optional[float] = None,
                 spatial_distance: Optional[float] = None,
                 mechanism: Optional[str] = None,
                 evidence: Optional[Dict[str, Any]] = None):
        
        self.cause = cause
        self.effect = effect
        self.strength = strength
        self.confidence = confidence
        self.process_type = process_type
        self.temporal_delay = temporal_delay
        self.spatial_distance = spatial_distance
        self.mechanism = mechanism
        self.evidence = evidence or {}
        
        # Classify strength
        if strength >= 0.8:
            self.strength_category = CausalityStrength.VERY_STRONG
        elif strength >= 0.6:
            self.strength_category = CausalityStrength.STRONG
        elif strength >= 0.4:
            self.strength_category = CausalityStrength.MODERATE
        else:
            self.strength_category = CausalityStrength.WEAK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'cause': self.cause,
            'effect': self.effect,
            'strength': self.strength,
            'confidence': self.confidence,
            'process_type': self.process_type.value,
            'strength_category': self.strength_category.value,
            'temporal_delay': self.temporal_delay,
            'spatial_distance': self.spatial_distance,
            'mechanism': self.mechanism,
            'evidence': self.evidence
        }

class GeologicalCausalDiscovery:
    """
    Advanced causal discovery system for geological processes
    """
    
    def __init__(self, config: GeologicalCausalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Causal discovery components
        self.causal_engine = None
        if CAUSAL_LEARN_AVAILABLE:
            self._initialize_causal_engine()
        
        # Domain knowledge constraints
        self.temporal_constraints = self._build_temporal_constraints()
        self.spatial_constraints = self._build_spatial_constraints()
        self.process_constraints = self._build_process_constraints()
        
        # Discovery results
        self.discovered_relationships = []
        self.causal_graph = nx.DiGraph()
        self.confidence_scores = {}
        self.intervention_effects = {}
        
        # Caches
        self.data_cache = {}
        self.results_cache = {}
        
    def _initialize_causal_engine(self):
        """Initialize causal inference engine"""
        try:
            self.causal_engine = CausalInferenceEngine({
                'significance_level': self.config.significance_level,
                'bootstrap_samples': self.config.bootstrap_samples
            })
        except Exception as e:
            self.logger.warning(f"Could not initialize causal engine: {e}")
    
    def _build_temporal_constraints(self) -> Dict[str, List[str]]:
        """Build temporal constraints for geological processes"""
        constraints = {
            # Processes that must precede others
            'precedes': {
                'magma_intrusion': ['contact_metamorphism', 'hydrothermal_alteration'],
                'tectonic_uplift': ['erosion', 'exposure'],
                'weathering': ['transportation', 'deposition'],
                'burial': ['compaction', 'diagenesis'],
                'heating': ['metamorphism', 'recrystallization'],
                'deformation': ['folding', 'faulting', 'fracturing'],
                'volcanic_eruption': ['ash_deposition', 'lava_cooling'],
                'sea_level_change': ['transgression', 'regression'],
                'glaciation': ['glacial_erosion', 'moraine_deposition']
            },
            
            # Processes that cannot occur simultaneously
            'mutually_exclusive': {
                'freezing_weathering': ['chemical_weathering_high_temp'],
                'deep_burial': ['surface_exposure'],
                'active_volcanism': ['glaciation_at_same_location']
            }
        }
        
        return constraints
    
    def _build_spatial_constraints(self) -> Dict[str, Dict[str, float]]:
        """Build spatial constraints for geological processes"""
        constraints = {
            # Maximum distances over which processes can causally influence each other
            'max_influence_distance': {
                'contact_metamorphism': 1000,      # 1 km from intrusion
                'hydrothermal_alteration': 5000,   # 5 km from source
                'regional_metamorphism': 100000,   # 100 km regional scale
                'volcanic_ashfall': 1000000,       # 1000 km for large eruptions
                'seismic_effects': 10000,          # 10 km for earthquakes
                'groundwater_flow': 50000,         # 50 km for regional flow
                'erosion_headward': 10000,         # 10 km for headward erosion
                'glacial_transport': 100000,       # 100 km for glacial transport
                'tectonic_stress': 500000          # 500 km for plate-scale stress
            },
            
            # Minimum distances for certain processes
            'min_separation_distance': {
                'independent_intrusions': 1000,
                'separate_fault_systems': 5000,
                'distinct_mineral_deposits': 500
            }
        }
        
        return constraints
    
    def _build_process_constraints(self) -> Dict[str, List[str]]:
        """Build constraints based on geological process compatibility"""
        constraints = {
            # Processes that commonly occur together
            'co_occurring': {
                'magma_intrusion': ['contact_metamorphism', 'hydrothermal_alteration'],
                'faulting': ['fracturing', 'hydrothermal_circulation'],
                'weathering': ['erosion', 'transportation'],
                'burial': ['compaction', 'diagenesis', 'pressure_dissolution'],
                'metamorphism': ['recrystallization', 'mineral_reactions'],
                'volcanism': ['magma_degassing', 'thermal_effects']
            },
            
            # Processes that enable others
            'enables': {
                'fracturing': ['fluid_circulation', 'mineralization'],
                'heating': ['chemical_reactions', 'phase_transitions'],
                'pressure': ['compaction', 'metamorphic_reactions'],
                'fluid_circulation': ['alteration', 'transport', 'deposition'],
                'tectonic_activity': ['deformation', 'metamorphism', 'volcanism'],
                'surface_exposure': ['weathering', 'erosion']
            }
        }
        
        return constraints
    
    async def discover_geological_causality(self, 
                                          geological_data: Dict[str, Any],
                                          time_series_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Discover causal relationships in geological processes
        
        Args:
            geological_data: Dictionary containing geological measurements and processes
            time_series_data: Optional time series data for temporal causal analysis
            
        Returns:
            Dictionary containing discovered causal relationships and analysis
        """
        
        try:
            self.logger.info("Starting geological causal discovery...")
            
            # Prepare data for causal analysis
            prepared_data = await self._prepare_causal_data(geological_data, time_series_data)
            
            # Apply multiple causal discovery algorithms
            causal_results = {}
            
            # Constraint-based methods (PC, FCI)
            if CAUSAL_LEARN_AVAILABLE:
                pc_results = await self._apply_pc_algorithm(prepared_data)
                causal_results['pc'] = pc_results
                
                fci_results = await self._apply_fci_algorithm(prepared_data)
                causal_results['fci'] = fci_results
            
            # Score-based methods (GES)
            if CAUSAL_LEARN_AVAILABLE:
                ges_results = await self._apply_ges_algorithm(prepared_data)
                causal_results['ges'] = ges_results
            
            # Domain-specific causal discovery
            domain_results = await self._apply_domain_specific_discovery(prepared_data)
            causal_results['domain_specific'] = domain_results
            
            # Temporal causal discovery
            if time_series_data is not None:
                temporal_results = await self._apply_temporal_causal_discovery(
                    prepared_data, time_series_data
                )
                causal_results['temporal'] = temporal_results
            
            # Ensemble and integration of results
            integrated_results = await self._integrate_causal_results(causal_results)
            
            # Validate discovered relationships
            validated_results = await self._validate_causal_relationships(integrated_results)
            
            # Generate intervention recommendations
            intervention_results = await self._generate_intervention_recommendations(validated_results)
            
            # Compile final results
            final_results = {
                'discovered_relationships': validated_results['relationships'],
                'causal_graph': validated_results['causal_graph'],
                'confidence_scores': validated_results['confidence_scores'],
                'algorithm_results': causal_results,
                'intervention_recommendations': intervention_results,
                'validation_metrics': validated_results['validation_metrics'],
                'discovery_statistics': self._calculate_discovery_statistics()
            }
            
            # Update internal state
            self.discovered_relationships = validated_results['relationships']
            self.causal_graph = validated_results['causal_graph']
            self.confidence_scores = validated_results['confidence_scores']
            
            self.logger.info(f"Causal discovery completed. Found {len(self.discovered_relationships)} relationships.")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in geological causal discovery: {e}")
            raise
    
    async def _prepare_causal_data(self, 
                                 geological_data: Dict[str, Any],
                                 time_series_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare data for causal analysis"""
        
        # Extract variables from geological data
        variables = []
        data_points = []
        
        # Process measurement data
        if 'measurements' in geological_data:
            for location, measurements in geological_data['measurements'].items():
                data_point = {'location': location}
                
                # Add spatial coordinates
                if 'coordinates' in measurements:
                    coords = measurements['coordinates']
                    data_point.update({
                        'x': coords[0],
                        'y': coords[1],
                        'z': coords[2] if len(coords) > 2 else 0
                    })
                
                # Add measurement values
                for var_name, value in measurements.items():
                    if isinstance(value, (int, float)) and var_name != 'coordinates':
                        data_point[var_name] = value
                
                data_points.append(data_point)
        
        # Process process indicators
        if 'processes' in geological_data:
            for i, data_point in enumerate(data_points):
                if i < len(geological_data['processes']):
                    process_info = geological_data['processes'][i]
                    
                    for process_name, intensity in process_info.items():
                        if isinstance(intensity, (int, float)):
                            data_point[f'process_{process_name}'] = intensity
        
        # Create DataFrame
        if data_points:
            df = pd.DataFrame(data_points)
            
            # Clean and normalize data
            df = df.dropna(thresh=len(df.columns) * 0.5)  # Remove rows with too many NaN values
            df = df.fillna(df.mean(numeric_only=True))    # Fill remaining NaN values
            
            # Normalize numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            
            return df
        else:
            # Create dummy data if no valid data points
            return pd.DataFrame({
                'var1': np.random.randn(100),
                'var2': np.random.randn(100),
                'var3': np.random.randn(100)
            })
    
    async def _apply_pc_algorithm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply PC algorithm for causal discovery"""
        
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return self._fallback_constraint_based_discovery(data)
            
            # Convert to numpy array
            data_array = data.select_dtypes(include=[np.number]).values
            variable_names = list(data.select_dtypes(include=[np.number]).columns)
            
            if data_array.shape[1] < 2:
                return {'edges': [], 'variable_names': variable_names, 'method': 'pc'}
            
            # Apply PC algorithm
            cg = pc(
                data_array,
                alpha=self.config.significance_level,
                indep_test=self.config.independence_test,
                stable=True,
                uc_rule=1,
                uc_priority=2
            )
            
            # Extract edges
            edges = []
            if hasattr(cg, 'G') and hasattr(cg.G, 'graph'):
                adj_matrix = cg.G.graph
                for i in range(len(variable_names)):
                    for j in range(len(variable_names)):
                        if i != j and adj_matrix[i, j] != 0:
                            # Determine edge direction
                            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                                # Directed edge i -> j
                                edges.append({
                                    'source': variable_names[i],
                                    'target': variable_names[j],
                                    'type': 'directed',
                                    'strength': 1.0,
                                    'method': 'pc'
                                })
                            elif adj_matrix[i, j] == 1 and adj_matrix[j, i] == 1:
                                # Undirected edge
                                if i < j:  # Avoid duplicates
                                    edges.append({
                                        'source': variable_names[i],
                                        'target': variable_names[j],
                                        'type': 'undirected',
                                        'strength': 1.0,
                                        'method': 'pc'
                                    })
            
            return {
                'edges': edges,
                'variable_names': variable_names,
                'method': 'pc',
                'algorithm_params': {
                    'alpha': self.config.significance_level,
                    'indep_test': self.config.independence_test
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Error in PC algorithm: {e}")
            return self._fallback_constraint_based_discovery(data)
    
    async def _apply_fci_algorithm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply FCI algorithm for causal discovery with latent confounders"""
        
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return self._fallback_constraint_based_discovery(data)
            
            # Convert to numpy array
            data_array = data.select_dtypes(include=[np.number]).values
            variable_names = list(data.select_dtypes(include=[np.number]).columns)
            
            if data_array.shape[1] < 2:
                return {'edges': [], 'variable_names': variable_names, 'method': 'fci'}
            
            # Apply FCI algorithm
            G, edges = fci(
                data_array,
                alpha=self.config.significance_level,
                indep_test=self.config.independence_test,
                depth=self.config.max_conditioning_set_size,
                max_path_length=-1
            )
            
            # Extract edges
            edge_list = []
            if hasattr(G, 'graph'):
                adj_matrix = G.graph
                for i in range(len(variable_names)):
                    for j in range(len(variable_names)):
                        if i != j and adj_matrix[i, j] != 0:
                            edge_type = 'undirected'
                            if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                                edge_type = 'directed'
                            elif adj_matrix[i, j] == 2:
                                edge_type = 'circle'  # Circle endpoint
                            
                            edge_list.append({
                                'source': variable_names[i],
                                'target': variable_names[j],
                                'type': edge_type,
                                'strength': 1.0,
                                'method': 'fci'
                            })
            
            return {
                'edges': edge_list,
                'variable_names': variable_names,
                'method': 'fci',
                'algorithm_params': {
                    'alpha': self.config.significance_level,
                    'depth': self.config.max_conditioning_set_size
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Error in FCI algorithm: {e}")
            return self._fallback_constraint_based_discovery(data)
    
    async def _apply_ges_algorithm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply GES algorithm for score-based causal discovery"""
        
        try:
            if not CAUSAL_LEARN_AVAILABLE:
                return self._fallback_score_based_discovery(data)
            
            # Convert to numpy array
            data_array = data.select_dtypes(include=[np.number]).values
            variable_names = list(data.select_dtypes(include=[np.number]).columns)
            
            if data_array.shape[1] < 2:
                return {'edges': [], 'variable_names': variable_names, 'method': 'ges'}
            
            # Apply GES algorithm
            Record = ges(data_array, score_func=self.config.score_function)
            
            # Extract edges from the result
            edges = []
            if hasattr(Record, 'G') and hasattr(Record.G, 'graph'):
                adj_matrix = Record.G.graph
                for i in range(len(variable_names)):
                    for j in range(len(variable_names)):
                        if adj_matrix[i, j] == 1:
                            # Directed edge i -> j
                            edges.append({
                                'source': variable_names[i],
                                'target': variable_names[j],
                                'type': 'directed',
                                'strength': 1.0,
                                'method': 'ges',
                                'score': getattr(Record, 'score', 0.0)
                            })
            
            return {
                'edges': edges,
                'variable_names': variable_names,
                'method': 'ges',
                'algorithm_params': {
                    'score_func': self.config.score_function
                },
                'final_score': getattr(Record, 'score', 0.0)
            }
            
        except Exception as e:
            self.logger.warning(f"Error in GES algorithm: {e}")
            return self._fallback_score_based_discovery(data)
    
    def _fallback_constraint_based_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback constraint-based causal discovery using correlation"""
        
        edges = []
        variable_names = list(data.select_dtypes(include=[np.number]).columns)
        
        if len(variable_names) < 2:
            return {'edges': edges, 'variable_names': variable_names, 'method': 'fallback_correlation'}
        
        # Calculate correlation matrix
        corr_matrix = data[variable_names].corr()
        
        # Create edges based on strong correlations
        threshold = 0.5
        for i, var1 in enumerate(variable_names):
            for j, var2 in enumerate(variable_names[i+1:], i+1):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > threshold and not np.isnan(corr_value):
                    edges.append({
                        'source': var1,
                        'target': var2,
                        'type': 'undirected',
                        'strength': corr_value,
                        'method': 'fallback_correlation'
                    })
        
        return {
            'edges': edges,
            'variable_names': variable_names,
            'method': 'fallback_correlation'
        }
    
    def _fallback_score_based_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback score-based causal discovery using regression"""
        
        edges = []
        variable_names = list(data.select_dtypes(include=[np.number]).columns)
        
        if len(variable_names) < 2:
            return {'edges': edges, 'variable_names': variable_names, 'method': 'fallback_regression'}
        
        # Use regression to identify potential causal relationships
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        
        threshold = 0.3  # R² threshold for significant relationships
        
        for target_var in variable_names:
            predictor_vars = [v for v in variable_names if v != target_var]
            
            if len(predictor_vars) == 0:
                continue
            
            X = data[predictor_vars].fillna(0)
            y = data[target_var].fillna(0)
            
            # Fit random forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create edges for important features
            for i, predictor in enumerate(predictor_vars):
                if importances[i] > threshold / len(predictor_vars):  # Adjust threshold
                    edges.append({
                        'source': predictor,
                        'target': target_var,
                        'type': 'directed',
                        'strength': importances[i],
                        'method': 'fallback_regression'
                    })
        
        return {
            'edges': edges,
            'variable_names': variable_names,
            'method': 'fallback_regression'
        }
    
    async def _apply_domain_specific_discovery(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply domain-specific geological causal discovery"""
        
        edges = []
        variable_names = list(data.select_dtypes(include=[np.number]).columns)
        
        # Apply geological domain knowledge
        for var1 in variable_names:
            for var2 in variable_names:
                if var1 != var2:
                    # Check if this relationship is geologically plausible
                    causal_strength = self._assess_geological_causality(var1, var2, data)
                    
                    if causal_strength > self.config.causal_strength_threshold:
                        edges.append({
                            'source': var1,
                            'target': var2,
                            'type': 'directed',
                            'strength': causal_strength,
                            'method': 'domain_specific',
                            'geological_reasoning': self._get_geological_reasoning(var1, var2)
                        })
        
        return {
            'edges': edges,
            'variable_names': variable_names,
            'method': 'domain_specific',
            'domain_constraints_applied': True
        }
    
    def _assess_geological_causality(self, cause: str, effect: str, data: pd.DataFrame) -> float:
        """Assess the geological plausibility of a causal relationship"""
        
        # Base score from correlation
        if cause in data.columns and effect in data.columns:
            correlation = data[cause].corr(data[effect])
            base_score = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            base_score = 0.0
        
        # Domain knowledge adjustments
        domain_score = self._get_domain_knowledge_score(cause, effect)
        
        # Temporal plausibility
        temporal_score = self._get_temporal_plausibility_score(cause, effect)
        
        # Spatial plausibility
        spatial_score = self._get_spatial_plausibility_score(cause, effect)
        
        # Combined score
        final_score = (base_score * 0.4 + 
                      domain_score * 0.3 + 
                      temporal_score * 0.2 + 
                      spatial_score * 0.1)
        
        return min(final_score, 1.0)
    
    def _get_domain_knowledge_score(self, cause: str, effect: str) -> float:
        """Get domain knowledge score for causal relationship"""
        
        # Check known causal relationships in geology
        known_relationships = {
            ('temperature', 'metamorphism'): 0.9,
            ('pressure', 'metamorphism'): 0.9,
            ('magma_intrusion', 'contact_metamorphism'): 0.95,
            ('weathering', 'erosion'): 0.8,
            ('tectonic_stress', 'faulting'): 0.85,
            ('burial', 'compaction'): 0.9,
            ('fluid_circulation', 'alteration'): 0.8,
            ('volcanism', 'ash_deposition'): 0.9,
            ('glaciation', 'glacial_erosion'): 0.85,
            ('sea_level_change', 'transgression'): 0.8
        }
        
        # Exact match
        if (cause, effect) in known_relationships:
            return known_relationships[(cause, effect)]
        
        # Partial matches (contains keywords)
        for (known_cause, known_effect), score in known_relationships.items():
            if known_cause in cause.lower() and known_effect in effect.lower():
                return score * 0.7  # Reduced confidence for partial match
        
        # Check process constraints
        if self.config.use_domain_knowledge:
            if 'precedes' in self.temporal_constraints:
                for precursor, consequences in self.temporal_constraints['precedes'].items():
                    if precursor in cause.lower():
                        for consequence in consequences:
                            if consequence in effect.lower():
                                return 0.8
        
        return 0.1  # Default low score for unknown relationships
    
    def _get_temporal_plausibility_score(self, cause: str, effect: str) -> float:
        """Get temporal plausibility score"""
        
        # Check temporal constraints
        if not self.config.enable_temporal_ordering:
            return 0.5  # Neutral score if temporal ordering is disabled
        
        # Check if the cause-effect relationship violates temporal constraints
        if 'mutually_exclusive' in self.temporal_constraints:
            for exclusive_group in self.temporal_constraints['mutually_exclusive'].values():
                if any(proc in cause.lower() for proc in exclusive_group) and \
                   any(proc in effect.lower() for proc in exclusive_group):
                    return 0.0  # Mutually exclusive processes
        
        # Check if cause precedes effect
        if 'precedes' in self.temporal_constraints:
            for precursor, consequences in self.temporal_constraints['precedes'].items():
                if precursor in cause.lower():
                    if any(cons in effect.lower() for cons in consequences):
                        return 1.0  # Perfect temporal ordering
                    else:
                        return 0.3  # Cause identified but effect doesn't match
                elif any(cons in cause.lower() for cons in consequences):
                    if precursor in effect.lower():
                        return 0.1  # Reversed temporal ordering (unlikely)
        
        return 0.5  # Neutral score for unknown temporal relationship
    
    def _get_spatial_plausibility_score(self, cause: str, effect: str) -> float:
        """Get spatial plausibility score"""
        
        if not self.config.enable_spatial_constraints:
            return 0.5  # Neutral score if spatial constraints are disabled
        
        # Check spatial constraints
        max_distances = self.spatial_constraints.get('max_influence_distance', {})
        
        for process, max_dist in max_distances.items():
            if process in cause.lower():
                # If we have distance information, check if it's within limits
                # For now, return high score if process is known to have spatial influence
                if any(effect_proc in effect.lower() for effect_proc in [
                    'alteration', 'metamorphism', 'mineralization', 'deformation'
                ]):
                    return 0.8
        
        return 0.5  # Neutral score for unknown spatial relationship
    
    def _get_geological_reasoning(self, cause: str, effect: str) -> str:
        """Get geological reasoning for the causal relationship"""
        
        reasoning_templates = {
            ('temperature', 'metamorphism'): "Elevated temperature drives metamorphic reactions",
            ('pressure', 'metamorphism'): "Increased pressure causes structural and mineralogical changes",
            ('magma_intrusion', 'contact_metamorphism'): "Heat from magma intrusion causes local metamorphism",
            ('weathering', 'erosion'): "Chemical and physical weathering weakens rocks, enabling erosion",
            ('tectonic_stress', 'faulting'): "Tectonic stress exceeds rock strength, causing brittle failure",
            ('burial', 'compaction'): "Increased overburden pressure leads to sediment compaction",
            ('fluid_circulation', 'alteration'): "Hydrothermal fluids cause chemical alteration of host rocks"
        }
        
        # Check for exact or partial matches
        for (known_cause, known_effect), reasoning in reasoning_templates.items():
            if known_cause in cause.lower() and known_effect in effect.lower():
                return reasoning
        
        return f"Potential causal relationship between {cause} and {effect} based on geological analysis"
    
    async def _apply_temporal_causal_discovery(self, 
                                             data: pd.DataFrame,
                                             time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply temporal causal discovery methods"""
        
        edges = []
        
        # Granger causality analysis
        if 'timestamp' in time_series_data.columns:
            # Prepare time series data
            ts_data = time_series_data.set_index('timestamp').sort_index()
            
            # Apply Granger causality test for each pair of variables
            numeric_vars = ts_data.select_dtypes(include=[np.number]).columns
            
            for var1 in numeric_vars:
                for var2 in numeric_vars:
                    if var1 != var2:
                        granger_score = self._calculate_granger_causality(
                            ts_data[var1], ts_data[var2]
                        )
                        
                        if granger_score > self.config.confidence_threshold:
                            edges.append({
                                'source': var1,
                                'target': var2,
                                'type': 'temporal',
                                'strength': granger_score,
                                'method': 'granger_causality',
                                'temporal_lag': self._estimate_temporal_lag(
                                    ts_data[var1], ts_data[var2]
                                )
                            })
        
        return {
            'edges': edges,
            'method': 'temporal',
            'time_series_length': len(time_series_data) if time_series_data is not None else 0
        }
    
    def _calculate_granger_causality(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Granger causality between two time series"""
        
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Prepare data (remove NaN values)
            combined_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(combined_data) < 10:  # Need sufficient data points
                return 0.0
            
            # Perform Granger causality test
            max_lag = min(5, len(combined_data) // 4)  # Reasonable lag limit
            
            try:
                gc_result = grangercausalitytests(
                    combined_data[['y', 'x']], 
                    maxlag=max_lag, 
                    verbose=False
                )
                
                # Extract p-value from the test
                p_values = []
                for lag in range(1, max_lag + 1):
                    if lag in gc_result:
                        p_val = gc_result[lag][0]['ssr_ftest'][1]  # F-test p-value
                        p_values.append(p_val)
                
                if p_values:
                    # Return 1 - minimum p-value as strength score
                    return 1.0 - min(p_values)
                else:
                    return 0.0
                    
            except Exception:
                # Fallback to correlation-based measure
                correlation = combined_data['x'].corr(combined_data['y'])
                return abs(correlation) if not np.isnan(correlation) else 0.0
                
        except ImportError:
            # Fallback if statsmodels is not available
            try:
                correlation = x.corr(y)
                return abs(correlation) if not np.isnan(correlation) else 0.0
            except:
                return 0.0
    
    def _estimate_temporal_lag(self, cause_series: pd.Series, effect_series: pd.Series) -> Optional[float]:
        """Estimate temporal lag between cause and effect"""
        
        try:
            # Cross-correlation analysis
            from scipy.signal import correlate
            
            # Remove NaN values and align series
            combined_data = pd.DataFrame({'cause': cause_series, 'effect': effect_series}).dropna()
            
            if len(combined_data) < 10:
                return None
            
            cause_vals = combined_data['cause'].values
            effect_vals = combined_data['effect'].values
            
            # Calculate cross-correlation
            correlation = correlate(effect_vals, cause_vals, mode='full')
            
            # Find lag with maximum correlation
            max_corr_idx = np.argmax(np.abs(correlation))
            lag = max_corr_idx - len(cause_vals) + 1
            
            # Convert to time units (assuming regular sampling)
            if hasattr(cause_series.index, 'freq') and cause_series.index.freq:
                time_lag = lag * cause_series.index.freq
                return float(time_lag.total_seconds())
            else:
                return float(lag)  # Return lag in index units
                
        except Exception:
            return None
    
    async def _integrate_causal_results(self, causal_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Integrate results from multiple causal discovery algorithms"""
        
        # Collect all edges from different algorithms
        all_edges = []
        edge_votes = {}  # Track how many algorithms found each edge
        
        for algorithm, results in causal_results.items():
            edges = results.get('edges', [])
            
            for edge in edges:
                source = edge['source']
                target = edge['target']
                edge_key = (source, target)
                
                # Add to vote counting
                if edge_key not in edge_votes:
                    edge_votes[edge_key] = {
                        'votes': 0,
                        'total_strength': 0.0,
                        'algorithms': [],
                        'edge_data': edge.copy()
                    }
                
                edge_votes[edge_key]['votes'] += 1
                edge_votes[edge_key]['total_strength'] += edge['strength']
                edge_votes[edge_key]['algorithms'].append(algorithm)
        
        # Create integrated edges based on consensus
        integrated_edges = []
        min_votes = max(1, len(causal_results) // 2)  # Require majority consensus
        
        for edge_key, vote_data in edge_votes.items():
            if vote_data['votes'] >= min_votes:
                # Calculate consensus strength
                consensus_strength = vote_data['total_strength'] / vote_data['votes']
                confidence = vote_data['votes'] / len(causal_results)
                
                edge = vote_data['edge_data'].copy()
                edge.update({
                    'strength': consensus_strength,
                    'confidence': confidence,
                    'consensus_votes': vote_data['votes'],
                    'supporting_algorithms': vote_data['algorithms']
                })
                
                integrated_edges.append(edge)
        
        # Build integrated causal graph
        causal_graph = nx.DiGraph()
        
        for edge in integrated_edges:
            causal_graph.add_edge(
                edge['source'],
                edge['target'],
                **{k: v for k, v in edge.items() if k not in ['source', 'target']}
            )
        
        return {
            'edges': integrated_edges,
            'causal_graph': causal_graph,
            'integration_method': 'consensus_voting',
            'algorithm_count': len(causal_results),
            'min_votes_required': min_votes
        }
    
    async def _validate_causal_relationships(self, integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate discovered causal relationships"""
        
        edges = integrated_results.get('edges', [])
        validated_relationships = []
        confidence_scores = {}
        validation_metrics = {}
        
        # Validate each relationship
        for edge in edges:
            validation_result = await self._validate_single_relationship(edge)
            
            if validation_result['is_valid']:
                # Convert to GeologicalCausalRelationship
                relationship = GeologicalCausalRelationship(
                    cause=edge['source'],
                    effect=edge['target'],
                    strength=edge['strength'],
                    confidence=validation_result['confidence'],
                    process_type=self._infer_process_type(edge['source'], edge['target']),
                    temporal_delay=edge.get('temporal_lag'),
                    mechanism=edge.get('geological_reasoning'),
                    evidence=validation_result['evidence']
                )
                
                validated_relationships.append(relationship)
                confidence_scores[f"{edge['source']}->{edge['target']}"] = validation_result['confidence']
        
        # Build validated causal graph
        validated_graph = nx.DiGraph()
        for rel in validated_relationships:
            validated_graph.add_edge(
                rel.cause,
                rel.effect,
                strength=rel.strength,
                confidence=rel.confidence,
                process_type=rel.process_type.value,
                mechanism=rel.mechanism
            )
        
        # Calculate validation metrics
        validation_metrics = {
            'total_discovered': len(edges),
            'total_validated': len(validated_relationships),
            'validation_rate': len(validated_relationships) / len(edges) if edges else 0,
            'average_confidence': np.mean(list(confidence_scores.values())) if confidence_scores else 0,
            'high_confidence_relationships': sum(1 for c in confidence_scores.values() if c > 0.8),
            'process_type_distribution': self._calculate_process_type_distribution(validated_relationships)
        }
        
        return {
            'relationships': validated_relationships,
            'causal_graph': validated_graph,
            'confidence_scores': confidence_scores,
            'validation_metrics': validation_metrics
        }
    
    async def _validate_single_relationship(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single causal relationship"""
        
        validation_score = 0.0
        evidence = {}
        
        # Statistical validation
        statistical_score = edge.get('strength', 0.0)
        validation_score += statistical_score * 0.4
        evidence['statistical_strength'] = statistical_score
        
        # Domain knowledge validation
        domain_score = self._get_domain_knowledge_score(edge['source'], edge['target'])
        validation_score += domain_score * 0.3
        evidence['domain_knowledge_score'] = domain_score
        
        # Temporal validation
        temporal_score = self._get_temporal_plausibility_score(edge['source'], edge['target'])
        validation_score += temporal_score * 0.2
        evidence['temporal_plausibility'] = temporal_score
        
        # Consensus validation
        consensus_score = edge.get('confidence', 0.0)
        validation_score += consensus_score * 0.1
        evidence['algorithm_consensus'] = consensus_score
        
        # Overall confidence
        confidence = min(validation_score, 1.0)
        is_valid = confidence > self.config.confidence_threshold
        
        return {
            'is_valid': is_valid,
            'confidence': confidence,
            'validation_score': validation_score,
            'evidence': evidence
        }
    
    def _infer_process_type(self, cause: str, effect: str) -> GeologicalProcessType:
        """Infer geological process type from cause and effect variables"""
        
        # Process type keywords
        process_keywords = {
            GeologicalProcessType.TECTONIC: ['fault', 'stress', 'deformation', 'plate', 'tectonic'],
            GeologicalProcessType.VOLCANIC: ['magma', 'lava', 'volcanic', 'eruption', 'ash'],
            GeologicalProcessType.METAMORPHIC: ['metamorphism', 'temperature', 'pressure', 'recrystallization'],
            GeologicalProcessType.HYDROTHERMAL: ['fluid', 'hydrothermal', 'alteration', 'circulation'],
            GeologicalProcessType.SEDIMENTARY: ['sediment', 'deposition', 'burial', 'compaction'],
            GeologicalProcessType.EROSIONAL: ['erosion', 'weathering', 'transportation'],
            GeologicalProcessType.STRUCTURAL: ['folding', 'faulting', 'fracturing', 'jointing']
        }
        
        # Check for matches in cause or effect
        combined_text = f"{cause} {effect}".lower()
        
        for process_type, keywords in process_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return process_type
        
        # Default to tectonic if no match found
        return GeologicalProcessType.TECTONIC
    
    def _calculate_process_type_distribution(self, relationships: List[GeologicalCausalRelationship]) -> Dict[str, int]:
        """Calculate distribution of process types"""
        distribution = {}
        
        for relationship in relationships:
            process_type = relationship.process_type.value
            distribution[process_type] = distribution.get(process_type, 0) + 1
        
        return distribution
    
    async def _generate_intervention_recommendations(self, validated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intervention recommendations based on discovered causal relationships"""
        
        relationships = validated_results['relationships']
        causal_graph = validated_results['causal_graph']
        
        recommendations = {
            'monitoring_priorities': [],
            'intervention_targets': [],
            'predictive_indicators': [],
            'risk_mitigation': []
        }
        
        # Identify high-impact variables (many outgoing edges)
        out_degrees = dict(causal_graph.out_degree())
        high_impact_vars = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for var, degree in high_impact_vars:
            if degree > 0:
                recommendations['monitoring_priorities'].append({
                    'variable': var,
                    'impact_score': degree,
                    'downstream_effects': list(causal_graph.successors(var)),
                    'recommendation': f"Monitor {var} closely as it influences {degree} other processes"
                })
        
        # Identify intervention targets (variables with high centrality)
        centrality_scores = nx.betweenness_centrality(causal_graph.to_undirected())
        intervention_targets = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for var, centrality in intervention_targets:
            if centrality > 0:
                recommendations['intervention_targets'].append({
                    'variable': var,
                    'centrality_score': centrality,
                    'recommendation': f"Consider {var} for intervention as it has high network centrality"
                })
        
        # Identify predictive indicators (variables with strong causal relationships)
        strong_relationships = [rel for rel in relationships if rel.strength > 0.8]
        
        for rel in strong_relationships[:5]:  # Top 5 strongest relationships
            recommendations['predictive_indicators'].append({
                'cause': rel.cause,
                'effect': rel.effect,
                'strength': rel.strength,
                'recommendation': f"Use {rel.cause} as a predictive indicator for {rel.effect}"
            })
        
        return recommendations
    
    def _calculate_discovery_statistics(self) -> Dict[str, Any]:
        """Calculate overall discovery statistics"""
        return {
            'total_relationships_discovered': len(self.discovered_relationships),
            'causal_graph_nodes': self.causal_graph.number_of_nodes(),
            'causal_graph_edges': self.causal_graph.number_of_edges(),
            'average_confidence': np.mean([rel.confidence for rel in self.discovered_relationships]) if self.discovered_relationships else 0,
            'process_type_counts': self._calculate_process_type_distribution(self.discovered_relationships),
            'strength_distribution': {
                'weak': sum(1 for rel in self.discovered_relationships if rel.strength_category == CausalityStrength.WEAK),
                'moderate': sum(1 for rel in self.discovered_relationships if rel.strength_category == CausalityStrength.MODERATE),
                'strong': sum(1 for rel in self.discovered_relationships if rel.strength_category == CausalityStrength.STRONG),
                'very_strong': sum(1 for rel in self.discovered_relationships if rel.strength_category == CausalityStrength.VERY_STRONG)
            },
            'temporal_relationships': sum(1 for rel in self.discovered_relationships if rel.temporal_delay is not None),
            'config': {
                'significance_level': self.config.significance_level,
                'confidence_threshold': self.config.confidence_threshold,
                'causal_strength_threshold': self.config.causal_strength_threshold
            }
        }

# Example usage and testing
async def example_geological_causal_discovery():
    """Example of geological causal discovery"""
    
    # Configuration
    config = GeologicalCausalConfig(
        significance_level=0.05,
        confidence_threshold=0.7,
        causal_strength_threshold=0.6,
        process_types=[
            GeologicalProcessType.TECTONIC,
            GeologicalProcessType.VOLCANIC,
            GeologicalProcessType.METAMORPHIC
        ]
    )
    
    # Create discovery system
    causal_discovery = GeologicalCausalDiscovery(config)
    
    # Example geological data
    geological_data = {
        'measurements': {
            'location_1': {
                'coordinates': [100, 200, 0],
                'temperature': 450,
                'pressure': 2.5,
                'strain_rate': 1e-14,
                'magnetic_intensity': 45000,
                'seismic_velocity': 6200
            },
            'location_2': {
                'coordinates': [150, 250, 100],
                'temperature': 800,
                'pressure': 1.8,
                'strain_rate': 5e-13,
                'magnetic_intensity': 42000,
                'seismic_velocity': 5800
            },
            'location_3': {
                'coordinates': [200, 300, 50],
                'temperature': 350,
                'pressure': 3.2,
                'strain_rate': 8e-15,
                'magnetic_intensity': 47000,
                'seismic_velocity': 6500
            }
        },
        'processes': [
            {'metamorphism': 0.7, 'deformation': 0.5, 'fluid_circulation': 0.3},
            {'volcanism': 0.9, 'metamorphism': 0.4, 'degassing': 0.8},
            {'burial': 0.6, 'compaction': 0.7, 'diagenesis': 0.4}
        ]
    }
    
    # Create time series data
    time_series_data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
        'temperature': 400 + 50 * np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 10, 100),
        'pressure': 2.0 + 0.5 * np.cos(np.linspace(0, 3*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'metamorphism': np.random.beta(2, 3, 100),
        'deformation': np.random.gamma(2, 0.3, 100)
    })
    
    # Discover causal relationships
    print("Discovering causal relationships in geological processes...")
    results = await causal_discovery.discover_geological_causality(
        geological_data, time_series_data
    )
    
    # Display results
    discovered_rels = results['discovered_relationships']
    print(f"Discovered {len(discovered_rels)} causal relationships")
    
    for rel in discovered_rels[:5]:  # Show first 5
        print(f"  {rel.cause} -> {rel.effect}: {rel.process_type.value} "
              f"(strength: {rel.strength:.3f}, confidence: {rel.confidence:.3f})")
    
    # Show statistics
    stats = results['discovery_statistics']
    print(f"Graph nodes: {stats['causal_graph_nodes']}")
    print(f"Graph edges: {stats['causal_graph_edges']}")
    print(f"Average confidence: {stats['average_confidence']:.3f}")
    
    # Show intervention recommendations
    interventions = results['intervention_recommendations']
    print(f"Monitoring priorities: {len(interventions['monitoring_priorities'])}")
    print(f"Intervention targets: {len(interventions['intervention_targets'])}")

if __name__ == "__main__":
    print(f"Geological Causal Discovery System")
    print(f"Causal-learn available: {CAUSAL_LEARN_AVAILABLE}")
    print(f"pgmpy available: {PGMPY_AVAILABLE}")
    print(f"SciPy available: {SCIPY_AVAILABLE}")
    
    # Run example
    asyncio.run(example_geological_causal_discovery())