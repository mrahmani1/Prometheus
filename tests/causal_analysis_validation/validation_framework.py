"""
Causal Analysis Validation Framework

Comprehensive validation framework for causal discovery algorithms,
statistical validation, benchmark testing, and accuracy assessment.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables"""
    source: str
    target: str
    strength: float
    confidence: float
    p_value: float
    method: str
    lag: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalGraph:
    """Represents a complete causal graph"""
    nodes: List[str]
    edges: List[CausalRelationship]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph"""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        
        for edge in self.edges:
            G.add_edge(
                edge.source, 
                edge.target, 
                weight=edge.strength,
                confidence=edge.confidence,
                p_value=edge.p_value,
                method=edge.method,
                lag=edge.lag
            )
        
        return G
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix representation"""
        n = len(self.nodes)
        adj_matrix = np.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        for edge in self.edges:
            i = node_to_idx[edge.source]
            j = node_to_idx[edge.target]
            adj_matrix[i, j] = edge.strength
        
        return adj_matrix


@dataclass
class ValidationResult:
    """Result of causal validation test"""
    test_name: str
    success: bool
    score: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None


@dataclass
class BenchmarkDataset:
    """Benchmark dataset for causal validation"""
    name: str
    data: pd.DataFrame
    true_graph: CausalGraph
    description: str
    source: str
    variables: List[str]
    sample_size: int
    time_series: bool = False
    domain: str = "general"


class CausalValidationFramework:
    """
    Comprehensive framework for validating causal discovery algorithms
    """
    
    def __init__(self, output_dir: str = "causal_validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.benchmark_datasets: Dict[str, BenchmarkDataset] = {}
        self.validation_results: List[ValidationResult] = []
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        # Load benchmark datasets
        self._load_benchmark_datasets()
        
        # Statistical thresholds
        self.thresholds = {
            'precision': 0.7,
            'recall': 0.6,
            'f1_score': 0.65,
            'structural_hamming_distance': 0.3,
            'causal_strength_correlation': 0.5,
            'statistical_significance': 0.05
        }
    
    def _load_benchmark_datasets(self):
        """Load benchmark datasets for validation"""
        
        # Geological benchmark: Volcanic eruption causality
        volcanic_data = self._create_volcanic_benchmark()
        self.benchmark_datasets['volcanic_eruption'] = volcanic_data
        
        # Seismic benchmark: Earthquake triggering
        seismic_data = self._create_seismic_benchmark()
        self.benchmark_datasets['seismic_triggering'] = seismic_data
        
        # Geochemical benchmark: Magma evolution
        geochem_data = self._create_geochemical_benchmark()
        self.benchmark_datasets['magma_evolution'] = geochem_data
        
        # Multi-physics benchmark: Coupled processes
        multiphysics_data = self._create_multiphysics_benchmark()
        self.benchmark_datasets['multiphysics_coupling'] = multiphysics_data
        
        # Synthetic linear benchmark
        linear_data = self._create_synthetic_linear_benchmark()
        self.benchmark_datasets['synthetic_linear'] = linear_data
        
        # Synthetic nonlinear benchmark
        nonlinear_data = self._create_synthetic_nonlinear_benchmark()
        self.benchmark_datasets['synthetic_nonlinear'] = nonlinear_data
    
    def _create_volcanic_benchmark(self) -> BenchmarkDataset:
        """Create volcanic eruption causality benchmark"""
        np.random.seed(42)
        n_samples = 1000
        
        # Variables: seismic activity, deformation, gas emissions, thermal anomalies, eruption
        variables = ['seismic_activity', 'deformation', 'gas_emissions', 'thermal_anomalies', 'eruption_intensity']
        
        # Generate data with known causal structure
        # seismic_activity -> deformation -> eruption_intensity
        # gas_emissions -> thermal_anomalies -> eruption_intensity
        # seismic_activity -> gas_emissions (indirect)
        
        seismic = np.random.normal(0, 1, n_samples)
        deformation = 0.8 * seismic + np.random.normal(0, 0.5, n_samples)
        gas_emissions = 0.6 * seismic + np.random.normal(0, 0.7, n_samples)
        thermal = 0.7 * gas_emissions + np.random.normal(0, 0.4, n_samples)
        eruption = 0.5 * deformation + 0.4 * thermal + np.random.normal(0, 0.3, n_samples)
        
        data = pd.DataFrame({
            'seismic_activity': seismic,
            'deformation': deformation,
            'gas_emissions': gas_emissions,
            'thermal_anomalies': thermal,
            'eruption_intensity': eruption
        })
        
        # True causal graph
        true_edges = [
            CausalRelationship('seismic_activity', 'deformation', 0.8, 0.95, 0.001, 'ground_truth'),
            CausalRelationship('seismic_activity', 'gas_emissions', 0.6, 0.90, 0.005, 'ground_truth'),
            CausalRelationship('gas_emissions', 'thermal_anomalies', 0.7, 0.92, 0.002, 'ground_truth'),
            CausalRelationship('deformation', 'eruption_intensity', 0.5, 0.88, 0.003, 'ground_truth'),
            CausalRelationship('thermal_anomalies', 'eruption_intensity', 0.4, 0.85, 0.008, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'volcanic', 'type': 'benchmark'}
        )
        
        return BenchmarkDataset(
            name='volcanic_eruption',
            data=data,
            true_graph=true_graph,
            description='Volcanic eruption causality with seismic, deformation, gas, and thermal precursors',
            source='synthetic_geological',
            variables=variables,
            sample_size=n_samples,
            time_series=False,
            domain='volcanic'
        )
    
    def _create_seismic_benchmark(self) -> BenchmarkDataset:
        """Create seismic triggering benchmark"""
        np.random.seed(123)
        n_samples = 800
        
        variables = ['stress_field', 'pore_pressure', 'fault_strength', 'seismic_magnitude', 'aftershock_rate']
        
        # Known causal structure for earthquake triggering
        stress = np.random.normal(0, 1, n_samples)
        pore_pressure = 0.7 * stress + np.random.normal(0, 0.6, n_samples)
        fault_strength = -0.5 * pore_pressure + np.random.normal(0, 0.8, n_samples)
        magnitude = 0.6 * stress - 0.4 * fault_strength + np.random.normal(0, 0.4, n_samples)
        aftershocks = 0.8 * magnitude + np.random.normal(0, 0.3, n_samples)
        
        data = pd.DataFrame({
            'stress_field': stress,
            'pore_pressure': pore_pressure,
            'fault_strength': fault_strength,
            'seismic_magnitude': magnitude,
            'aftershock_rate': aftershocks
        })
        
        true_edges = [
            CausalRelationship('stress_field', 'pore_pressure', 0.7, 0.93, 0.001, 'ground_truth'),
            CausalRelationship('pore_pressure', 'fault_strength', -0.5, 0.87, 0.004, 'ground_truth'),
            CausalRelationship('stress_field', 'seismic_magnitude', 0.6, 0.89, 0.002, 'ground_truth'),
            CausalRelationship('fault_strength', 'seismic_magnitude', -0.4, 0.82, 0.007, 'ground_truth'),
            CausalRelationship('seismic_magnitude', 'aftershock_rate', 0.8, 0.94, 0.001, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'seismic', 'type': 'benchmark'}
        )
        
        return BenchmarkDataset(
            name='seismic_triggering',
            data=data,
            true_graph=true_graph,
            description='Seismic triggering causality with stress, pressure, and fault mechanics',
            source='synthetic_geological',
            variables=variables,
            sample_size=n_samples,
            domain='seismic'
        )
    
    def _create_geochemical_benchmark(self) -> BenchmarkDataset:
        """Create geochemical evolution benchmark"""
        np.random.seed(456)
        n_samples = 600
        
        variables = ['temperature', 'pressure', 'SiO2_content', 'crystal_fraction', 'viscosity']
        
        # Magma evolution causality
        temperature = np.random.normal(1200, 100, n_samples)  # Celsius
        pressure = np.random.normal(2, 0.5, n_samples)  # GPa
        SiO2 = 45 + 0.02 * temperature + 5 * pressure + np.random.normal(0, 2, n_samples)
        crystals = np.maximum(0, (1300 - temperature) / 100 + np.random.normal(0, 0.1, n_samples))
        viscosity = np.exp(0.1 * SiO2 + 2 * crystals - 0.005 * temperature) + np.random.normal(0, 0.5, n_samples)
        
        data = pd.DataFrame({
            'temperature': temperature,
            'pressure': pressure,
            'SiO2_content': SiO2,
            'crystal_fraction': crystals,
            'viscosity': viscosity
        })
        
        true_edges = [
            CausalRelationship('temperature', 'SiO2_content', 0.02, 0.75, 0.02, 'ground_truth'),
            CausalRelationship('pressure', 'SiO2_content', 5.0, 0.85, 0.005, 'ground_truth'),
            CausalRelationship('temperature', 'crystal_fraction', -0.01, 0.88, 0.003, 'ground_truth'),
            CausalRelationship('SiO2_content', 'viscosity', 0.1, 0.92, 0.001, 'ground_truth'),
            CausalRelationship('crystal_fraction', 'viscosity', 2.0, 0.90, 0.002, 'ground_truth'),
            CausalRelationship('temperature', 'viscosity', -0.005, 0.78, 0.015, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'geochemical', 'type': 'benchmark'}
        )
        
        return BenchmarkDataset(
            name='magma_evolution',
            data=data,
            true_graph=true_graph,
            description='Magma evolution causality with temperature, pressure, and composition',
            source='synthetic_geological',
            variables=variables,
            sample_size=n_samples,
            domain='geochemical'
        )
    
    def _create_multiphysics_benchmark(self) -> BenchmarkDataset:
        """Create multi-physics coupling benchmark"""
        np.random.seed(789)
        n_samples = 1200
        
        variables = ['thermal_gradient', 'fluid_flow', 'mechanical_stress', 'chemical_reaction', 'electromagnetic_field']
        
        # Complex multi-physics interactions
        thermal = np.random.normal(0, 1, n_samples)
        fluid = 0.6 * thermal + np.random.normal(0, 0.5, n_samples)
        stress = 0.4 * thermal + 0.3 * fluid + np.random.normal(0, 0.6, n_samples)
        chemical = 0.5 * thermal + 0.7 * fluid + np.random.normal(0, 0.4, n_samples)
        electromagnetic = 0.3 * stress + 0.4 * chemical + np.random.normal(0, 0.5, n_samples)
        
        data = pd.DataFrame({
            'thermal_gradient': thermal,
            'fluid_flow': fluid,
            'mechanical_stress': stress,
            'chemical_reaction': chemical,
            'electromagnetic_field': electromagnetic
        })
        
        true_edges = [
            CausalRelationship('thermal_gradient', 'fluid_flow', 0.6, 0.91, 0.002, 'ground_truth'),
            CausalRelationship('thermal_gradient', 'mechanical_stress', 0.4, 0.83, 0.008, 'ground_truth'),
            CausalRelationship('fluid_flow', 'mechanical_stress', 0.3, 0.79, 0.012, 'ground_truth'),
            CausalRelationship('thermal_gradient', 'chemical_reaction', 0.5, 0.87, 0.004, 'ground_truth'),
            CausalRelationship('fluid_flow', 'chemical_reaction', 0.7, 0.93, 0.001, 'ground_truth'),
            CausalRelationship('mechanical_stress', 'electromagnetic_field', 0.3, 0.76, 0.018, 'ground_truth'),
            CausalRelationship('chemical_reaction', 'electromagnetic_field', 0.4, 0.84, 0.006, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'multiphysics', 'type': 'benchmark'}
        )
        
        return BenchmarkDataset(
            name='multiphysics_coupling',
            data=data,
            true_graph=true_graph,
            description='Multi-physics coupling with thermal, fluid, mechanical, chemical, and electromagnetic processes',
            source='synthetic_multiphysics',
            variables=variables,
            sample_size=n_samples,
            domain='multiphysics'
        )
    
    def _create_synthetic_linear_benchmark(self) -> BenchmarkDataset:
        """Create synthetic linear causal benchmark"""
        np.random.seed(101)
        n_samples = 500
        
        variables = ['X1', 'X2', 'X3', 'X4', 'X5']
        
        # Linear causal chain: X1 -> X2 -> X3 -> X4 -> X5
        X1 = np.random.normal(0, 1, n_samples)
        X2 = 0.8 * X1 + np.random.normal(0, 0.5, n_samples)
        X3 = 0.7 * X2 + np.random.normal(0, 0.5, n_samples)
        X4 = 0.6 * X3 + np.random.normal(0, 0.5, n_samples)
        X5 = 0.5 * X4 + np.random.normal(0, 0.5, n_samples)
        
        data = pd.DataFrame({
            'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5
        })
        
        true_edges = [
            CausalRelationship('X1', 'X2', 0.8, 0.95, 0.001, 'ground_truth'),
            CausalRelationship('X2', 'X3', 0.7, 0.93, 0.002, 'ground_truth'),
            CausalRelationship('X3', 'X4', 0.6, 0.90, 0.003, 'ground_truth'),
            CausalRelationship('X4', 'X5', 0.5, 0.87, 0.005, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'synthetic', 'type': 'linear'}
        )
        
        return BenchmarkDataset(
            name='synthetic_linear',
            data=data,
            true_graph=true_graph,
            description='Synthetic linear causal chain for algorithm validation',
            source='synthetic',
            variables=variables,
            sample_size=n_samples,
            domain='synthetic'
        )
    
    def _create_synthetic_nonlinear_benchmark(self) -> BenchmarkDataset:
        """Create synthetic nonlinear causal benchmark"""
        np.random.seed(202)
        n_samples = 800
        
        variables = ['A', 'B', 'C', 'D', 'E']
        
        # Nonlinear relationships
        A = np.random.normal(0, 1, n_samples)
        B = np.sin(A) + np.random.normal(0, 0.3, n_samples)
        C = A**2 + 0.5 * B + np.random.normal(0, 0.4, n_samples)
        D = np.exp(0.2 * B) + np.random.normal(0, 0.5, n_samples)
        E = np.log(np.abs(C) + 1) + 0.3 * D + np.random.normal(0, 0.3, n_samples)
        
        data = pd.DataFrame({
            'A': A, 'B': B, 'C': C, 'D': D, 'E': E
        })
        
        true_edges = [
            CausalRelationship('A', 'B', 0.85, 0.92, 0.002, 'ground_truth'),
            CausalRelationship('A', 'C', 0.75, 0.88, 0.004, 'ground_truth'),
            CausalRelationship('B', 'C', 0.5, 0.82, 0.008, 'ground_truth'),
            CausalRelationship('B', 'D', 0.70, 0.86, 0.005, 'ground_truth'),
            CausalRelationship('C', 'E', 0.65, 0.84, 0.006, 'ground_truth'),
            CausalRelationship('D', 'E', 0.3, 0.76, 0.015, 'ground_truth')
        ]
        
        true_graph = CausalGraph(
            nodes=variables,
            edges=true_edges,
            metadata={'domain': 'synthetic', 'type': 'nonlinear'}
        )
        
        return BenchmarkDataset(
            name='synthetic_nonlinear',
            data=data,
            true_graph=true_graph,
            description='Synthetic nonlinear causal relationships for algorithm validation',
            source='synthetic',
            variables=variables,
            sample_size=n_samples,
            domain='synthetic'
        )
    
    async def validate_causal_discovery_algorithm(
        self, 
        algorithm_func: callable, 
        algorithm_name: str,
        benchmark_names: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """Validate a causal discovery algorithm against benchmarks"""
        
        if benchmark_names is None:
            benchmark_names = list(self.benchmark_datasets.keys())
        
        results = {}
        
        for benchmark_name in benchmark_names:
            if benchmark_name not in self.benchmark_datasets:
                logger.warning(f"Benchmark {benchmark_name} not found")
                continue
            
            logger.info(f"Validating {algorithm_name} on {benchmark_name}")
            
            benchmark = self.benchmark_datasets[benchmark_name]
            
            try:
                start_time = datetime.now()
                
                # Run algorithm
                discovered_graph = await self._run_algorithm_async(
                    algorithm_func, 
                    benchmark.data
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds() * 1000
                
                # Validate results
                validation_result = self._validate_discovered_graph(
                    discovered_graph,
                    benchmark.true_graph,
                    f"{algorithm_name}_{benchmark_name}"
                )
                
                validation_result.execution_time_ms = execution_time
                results[benchmark_name] = validation_result
                
            except Exception as e:
                logger.error(f"Algorithm validation failed for {benchmark_name}: {e}")
                results[benchmark_name] = ValidationResult(
                    test_name=f"{algorithm_name}_{benchmark_name}",
                    success=False,
                    score=0.0,
                    details={},
                    error_message=str(e)
                )
        
        return results
    
    async def _run_algorithm_async(self, algorithm_func: callable, data: pd.DataFrame) -> CausalGraph:
        """Run causal discovery algorithm asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run algorithm in thread pool to avoid blocking
        result = await loop.run_in_executor(
            self.executor,
            algorithm_func,
            data
        )
        
        return result
    
    def _validate_discovered_graph(
        self, 
        discovered: CausalGraph, 
        true_graph: CausalGraph,
        test_name: str
    ) -> ValidationResult:
        """Validate discovered causal graph against ground truth"""
        
        details = {}
        
        try:
            # Convert to adjacency matrices for comparison
            true_adj = true_graph.get_adjacency_matrix()
            discovered_adj = discovered.get_adjacency_matrix()
            
            # Ensure same dimensions
            if true_adj.shape != discovered_adj.shape:
                # Pad or truncate to match
                max_size = max(true_adj.shape[0], discovered_adj.shape[0])
                true_adj_padded = np.zeros((max_size, max_size))
                discovered_adj_padded = np.zeros((max_size, max_size))
                
                true_adj_padded[:true_adj.shape[0], :true_adj.shape[1]] = true_adj
                discovered_adj_padded[:discovered_adj.shape[0], :discovered_adj.shape[1]] = discovered_adj
                
                true_adj = true_adj_padded
                discovered_adj = discovered_adj_padded
            
            # Binary adjacency matrices for structural comparison
            true_binary = (true_adj != 0).astype(int)
            discovered_binary = (discovered_adj != 0).astype(int)
            
            # Calculate structural metrics
            precision = self._calculate_precision(discovered_binary, true_binary)
            recall = self._calculate_recall(discovered_binary, true_binary)
            f1 = self._calculate_f1_score(precision, recall)
            
            details['precision'] = precision
            details['recall'] = recall
            details['f1_score'] = f1
            
            # Structural Hamming Distance
            shd = self._calculate_structural_hamming_distance(discovered_binary, true_binary)
            details['structural_hamming_distance'] = shd
            
            # Causal strength correlation (for non-zero edges)
            strength_correlation = self._calculate_strength_correlation(discovered_adj, true_adj)
            details['causal_strength_correlation'] = strength_correlation
            
            # Edge-wise analysis
            edge_analysis = self._analyze_edge_accuracy(discovered, true_graph)
            details['edge_analysis'] = edge_analysis
            
            # Statistical significance validation
            sig_validation = self._validate_statistical_significance(discovered)
            details['statistical_validation'] = sig_validation
            
            # Calculate overall score
            score_components = [
                precision * 0.25,
                recall * 0.25,
                f1 * 0.20,
                max(0, 1 - shd) * 0.15,  # Lower SHD is better
                strength_correlation * 0.15
            ]
            
            overall_score = sum(score_components)
            
            # Success criteria
            success = (
                precision >= self.thresholds['precision'] and
                recall >= self.thresholds['recall'] and
                f1 >= self.thresholds['f1_score'] and
                shd <= self.thresholds['structural_hamming_distance'] and
                strength_correlation >= self.thresholds['causal_strength_correlation']
            )
            
            return ValidationResult(
                test_name=test_name,
                success=success,
                score=overall_score,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                success=False,
                score=0.0,
                details=details,
                error_message=str(e)
            )
    
    def _calculate_precision(self, discovered: np.ndarray, true: np.ndarray) -> float:
        """Calculate precision for causal edge discovery"""
        discovered_flat = discovered.flatten()
        true_flat = true.flatten()
        
        if np.sum(discovered_flat) == 0:
            return 0.0
        
        return precision_score(true_flat, discovered_flat, zero_division=0)
    
    def _calculate_recall(self, discovered: np.ndarray, true: np.ndarray) -> float:
        """Calculate recall for causal edge discovery"""
        discovered_flat = discovered.flatten()
        true_flat = true.flatten()
        
        if np.sum(true_flat) == 0:
            return 1.0 if np.sum(discovered_flat) == 0 else 0.0
        
        return recall_score(true_flat, discovered_flat, zero_division=0)
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_structural_hamming_distance(self, discovered: np.ndarray, true: np.ndarray) -> float:
        """Calculate Structural Hamming Distance (normalized)"""
        n = discovered.shape[0]
        max_edges = n * (n - 1)  # Maximum possible edges in DAG
        
        if max_edges == 0:
            return 0.0
        
        # Count differences
        differences = np.sum(discovered != true)
        
        # Normalize by maximum possible differences
        return differences / max_edges
    
    def _calculate_strength_correlation(self, discovered: np.ndarray, true: np.ndarray) -> float:
        """Calculate correlation between causal strengths"""
        # Only consider edges that exist in both graphs
        mask = (true != 0) & (discovered != 0)
        
        if np.sum(mask) < 2:
            return 0.0
        
        true_strengths = true[mask]
        discovered_strengths = discovered[mask]
        
        try:
            correlation, _ = stats.pearsonr(true_strengths, discovered_strengths)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _analyze_edge_accuracy(self, discovered: CausalGraph, true_graph: CausalGraph) -> Dict[str, Any]:
        """Analyze accuracy of individual edges"""
        true_edges = {(e.source, e.target): e for e in true_graph.edges}
        discovered_edges = {(e.source, e.target): e for e in discovered.edges}
        
        true_positives = []
        false_positives = []
        false_negatives = []
        
        # Check discovered edges
        for edge_key, edge in discovered_edges.items():
            if edge_key in true_edges:
                true_edge = true_edges[edge_key]
                strength_error = abs(edge.strength - true_edge.strength)
                true_positives.append({
                    'edge': edge_key,
                    'discovered_strength': edge.strength,
                    'true_strength': true_edge.strength,
                    'strength_error': strength_error,
                    'confidence': edge.confidence
                })
            else:
                false_positives.append({
                    'edge': edge_key,
                    'strength': edge.strength,
                    'confidence': edge.confidence
                })
        
        # Check missed edges
        for edge_key, edge in true_edges.items():
            if edge_key not in discovered_edges:
                false_negatives.append({
                    'edge': edge_key,
                    'true_strength': edge.strength
                })
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'tp_count': len(true_positives),
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }
    
    def _validate_statistical_significance(self, discovered: CausalGraph) -> Dict[str, Any]:
        """Validate statistical significance of discovered relationships"""
        significant_edges = 0
        total_edges = len(discovered.edges)
        
        p_values = []
        confidence_scores = []
        
        for edge in discovered.edges:
            p_values.append(edge.p_value)
            confidence_scores.append(edge.confidence)
            
            if edge.p_value < self.thresholds['statistical_significance']:
                significant_edges += 1
        
        return {
            'total_edges': total_edges,
            'significant_edges': significant_edges,
            'significance_rate': significant_edges / total_edges if total_edges > 0 else 0,
            'mean_p_value': np.mean(p_values) if p_values else 1.0,
            'mean_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'min_p_value': np.min(p_values) if p_values else 1.0,
            'max_confidence': np.max(confidence_scores) if confidence_scores else 0.0
        }
    
    async def run_cross_validation_test(
        self, 
        algorithm_func: callable, 
        algorithm_name: str,
        benchmark_name: str,
        k_folds: int = 5
    ) -> ValidationResult:
        """Run k-fold cross-validation on a benchmark dataset"""
        
        if benchmark_name not in self.benchmark_datasets:
            return ValidationResult(
                test_name=f"{algorithm_name}_cv_{benchmark_name}",
                success=False,
                score=0.0,
                details={},
                error_message=f"Benchmark {benchmark_name} not found"
            )
        
        benchmark = self.benchmark_datasets[benchmark_name]
        data = benchmark.data
        
        try:
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_results = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data)):
                logger.info(f"Running cross-validation fold {fold_idx + 1}/{k_folds}")
                
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Train on training data
                discovered_graph = await self._run_algorithm_async(algorithm_func, train_data)
                
                # Validate structure on test data (simplified validation)
                fold_score = self._evaluate_fold_performance(discovered_graph, benchmark.true_graph)
                fold_results.append(fold_score)
            
            # Calculate cross-validation statistics
            cv_mean = np.mean(fold_results)
            cv_std = np.std(fold_results)
            cv_min = np.min(fold_results)
            cv_max = np.max(fold_results)
            
            details = {
                'k_folds': k_folds,
                'fold_scores': fold_results,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_min': cv_min,
                'cv_max': cv_max,
                'cv_stability': 1 - (cv_std / cv_mean) if cv_mean > 0 else 0
            }
            
            success = cv_mean >= 0.7 and cv_std <= 0.2  # Stability criteria
            
            return ValidationResult(
                test_name=f"{algorithm_name}_cv_{benchmark_name}",
                success=success,
                score=cv_mean,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{algorithm_name}_cv_{benchmark_name}",
                success=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def _evaluate_fold_performance(self, discovered: CausalGraph, true_graph: CausalGraph) -> float:
        """Evaluate performance for a single cross-validation fold"""
        try:
            true_adj = true_graph.get_adjacency_matrix()
            discovered_adj = discovered.get_adjacency_matrix()
            
            # Ensure same dimensions
            if true_adj.shape != discovered_adj.shape:
                max_size = max(true_adj.shape[0], discovered_adj.shape[0])
                true_adj_padded = np.zeros((max_size, max_size))
                discovered_adj_padded = np.zeros((max_size, max_size))
                
                true_adj_padded[:true_adj.shape[0], :true_adj.shape[1]] = true_adj
                discovered_adj_padded[:discovered_adj.shape[0], :discovered_adj.shape[1]] = discovered_adj
                
                true_adj = true_adj_padded
                discovered_adj = discovered_adj_padded
            
            # Calculate F1 score as fold performance metric
            true_binary = (true_adj != 0).astype(int)
            discovered_binary = (discovered_adj != 0).astype(int)
            
            precision = self._calculate_precision(discovered_binary, true_binary)
            recall = self._calculate_recall(discovered_binary, true_binary)
            f1 = self._calculate_f1_score(precision, recall)
            
            return f1
            
        except Exception:
            return 0.0
    
    async def run_robustness_test(
        self, 
        algorithm_func: callable, 
        algorithm_name: str,
        benchmark_name: str,
        noise_levels: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
    ) -> ValidationResult:
        """Test algorithm robustness to noise"""
        
        if benchmark_name not in self.benchmark_datasets:
            return ValidationResult(
                test_name=f"{algorithm_name}_robustness_{benchmark_name}",
                success=False,
                score=0.0,
                details={},
                error_message=f"Benchmark {benchmark_name} not found"
            )
        
        benchmark = self.benchmark_datasets[benchmark_name]
        original_data = benchmark.data
        
        try:
            noise_results = []
            
            for noise_level in noise_levels:
                logger.info(f"Testing robustness with noise level {noise_level}")
                
                # Add noise to data
                noisy_data = original_data.copy()
                for col in noisy_data.columns:
                    noise = np.random.normal(0, noise_level * noisy_data[col].std(), len(noisy_data))
                    noisy_data[col] += noise
                
                # Run algorithm on noisy data
                discovered_graph = await self._run_algorithm_async(algorithm_func, noisy_data)
                
                # Evaluate performance
                validation_result = self._validate_discovered_graph(
                    discovered_graph,
                    benchmark.true_graph,
                    f"{algorithm_name}_noise_{noise_level}"
                )
                
                noise_results.append({
                    'noise_level': noise_level,
                    'score': validation_result.score,
                    'precision': validation_result.details.get('precision', 0),
                    'recall': validation_result.details.get('recall', 0),
                    'f1_score': validation_result.details.get('f1_score', 0)
                })
            
            # Calculate robustness metrics
            scores = [r['score'] for r in noise_results]
            score_degradation = scores[0] - scores[-1] if len(scores) > 1 else 0
            mean_score = np.mean(scores)
            score_stability = 1 - (np.std(scores) / mean_score) if mean_score > 0 else 0
            
            details = {
                'noise_levels': noise_levels,
                'noise_results': noise_results,
                'mean_score': mean_score,
                'score_degradation': score_degradation,
                'score_stability': score_stability,
                'robustness_score': max(0, 1 - score_degradation)
            }
            
            success = score_degradation <= 0.3 and score_stability >= 0.7
            
            return ValidationResult(
                test_name=f"{algorithm_name}_robustness_{benchmark_name}",
                success=success,
                score=details['robustness_score'],
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                test_name=f"{algorithm_name}_robustness_{benchmark_name}",
                success=False,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        
        # Calculate aggregate metrics
        scores = [r.score for r in results if r.score is not None]
        mean_score = np.mean(scores) if scores else 0.0
        std_score = np.std(scores) if scores else 0.0
        
        # Execution time statistics
        execution_times = [r.execution_time_ms for r in results if r.execution_time_ms is not None]
        mean_execution_time = np.mean(execution_times) if execution_times else 0.0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
                'mean_score': mean_score,
                'std_score': std_score,
                'mean_execution_time_ms': mean_execution_time
            },
            'test_results': [
                {
                    'test_name': r.test_name,
                    'success': r.success,
                    'score': r.score,
                    'execution_time_ms': r.execution_time_ms,
                    'error_message': r.error_message,
                    'details': r.details
                }
                for r in results
            ],
            'benchmarks_used': list(self.benchmark_datasets.keys()),
            'validation_thresholds': self.thresholds,
            'recommendations': []
        }
        
        # Generate recommendations
        recommendations = []
        
        if report['summary']['success_rate'] < 0.8:
            recommendations.append("Overall success rate is below 80% - review algorithm implementation")
        
        if mean_score < 0.7:
            recommendations.append("Mean validation score is below 0.7 - consider algorithm improvements")
        
        if std_score > 0.3:
            recommendations.append("High score variance indicates inconsistent performance across benchmarks")
        
        failed_tests = [r for r in results if not r.success]
        if failed_tests:
            failed_benchmarks = [r.test_name for r in failed_tests]
            recommendations.append(f"Failed tests: {', '.join(failed_benchmarks)}")
        
        report['recommendations'] = recommendations
        
        return report
    
    async def run_comprehensive_validation(
        self, 
        algorithm_func: callable, 
        algorithm_name: str
    ) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        
        logger.info(f"Starting comprehensive validation for {algorithm_name}")
        
        all_results = []
        
        # Basic benchmark validation
        benchmark_results = await self.validate_causal_discovery_algorithm(
            algorithm_func, 
            algorithm_name
        )
        all_results.extend(benchmark_results.values())
        
        # Cross-validation tests
        for benchmark_name in ['synthetic_linear', 'volcanic_eruption']:
            cv_result = await self.run_cross_validation_test(
                algorithm_func, 
                algorithm_name, 
                benchmark_name
            )
            all_results.append(cv_result)
        
        # Robustness tests
        for benchmark_name in ['synthetic_linear', 'seismic_triggering']:
            robustness_result = await self.run_robustness_test(
                algorithm_func, 
                algorithm_name, 
                benchmark_name
            )
            all_results.append(robustness_result)
        
        # Generate comprehensive report
        report = self.generate_validation_report(all_results)
        
        # Save report
        report_file = self.output_dir / f"{algorithm_name}_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive validation completed for {algorithm_name}")
        logger.info(f"Success rate: {report['summary']['success_rate']:.2%}")
        logger.info(f"Mean score: {report['summary']['mean_score']:.3f}")
        
        return report


# Example causal discovery algorithms for testing
def example_pc_algorithm(data: pd.DataFrame) -> CausalGraph:
    """Example PC algorithm implementation"""
    variables = list(data.columns)
    edges = []
    
    # Simplified PC algorithm simulation
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i != j:
                # Calculate correlation as proxy for causal strength
                corr = data[var1].corr(data[var2])
                if abs(corr) > 0.3:  # Threshold for edge
                    edges.append(CausalRelationship(
                        source=var1,
                        target=var2,
                        strength=abs(corr),
                        confidence=0.8,
                        p_value=0.01,
                        method='PC'
                    ))
    
    return CausalGraph(nodes=variables, edges=edges)


def example_ges_algorithm(data: pd.DataFrame) -> CausalGraph:
    """Example GES algorithm implementation"""
    variables = list(data.columns)
    edges = []
    
    # Simplified GES algorithm simulation
    correlation_matrix = data.corr()
    
    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            if i < j:  # Avoid duplicates
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > 0.4:  # Higher threshold for GES
                    edges.append(CausalRelationship(
                        source=var1,
                        target=var2,
                        strength=abs(corr),
                        confidence=0.85,
                        p_value=0.005,
                        method='GES'
                    ))
    
    return CausalGraph(nodes=variables, edges=edges)


# Test execution
async def main():
    """Main test execution"""
    framework = CausalValidationFramework()
    
    # Test PC algorithm
    pc_report = await framework.run_comprehensive_validation(
        example_pc_algorithm,
        'PC_Algorithm'
    )
    
    # Test GES algorithm
    ges_report = await framework.run_comprehensive_validation(
        example_ges_algorithm,
        'GES_Algorithm'
    )
    
    print("Validation completed!")
    print(f"PC Algorithm Success Rate: {pc_report['summary']['success_rate']:.2%}")
    print(f"GES Algorithm Success Rate: {ges_report['summary']['success_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())