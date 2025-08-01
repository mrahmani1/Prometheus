"""
Tests for the Causal Inference Engine

Comprehensive test suite for geological causal relationship discovery,
intervention analysis, and hypothesis generation.
"""

import json
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from core.causal_inference_engine import (
    CausalAlgorithm,
    CausalEdge,
    CausalGraph,
    CausalInferenceEngine,
    EdgeType,
    GeologicalDomainKnowledge,
    InterventionResult,
)
from core.causal_inference_engine import numpy as np
from core.causal_inference_engine import pandas as pd
from core.causal_inference_engine import torch.nn.functional as F

class TestCausalInferenceEngine:
    """Test suite for CausalInferenceEngine"""

    @pytest.fixture
    def sample_geological_data(self):
        """Create sample geological data for testing"""
        np.random.seed(42)
        n_samples = 200

        # Create synthetic geological data with known causal relationships
        # magma_pressure -> ground_deformation
        # temperature -> gas_emissions
        # stress_changes -> seismic_activity

        magma_pressure = np.random.normal(100, 20, n_samples)
        temperature = np.random.normal(800, 100, n_samples)
        stress_changes = np.random.normal(0, 1, n_samples)

        # Create causal relationships with noise
        ground_deformation = 0.5 * magma_pressure + np.random.normal(0, 5, n_samples)
        gas_emissions = 0.3 * temperature + np.random.normal(0, 10, n_samples)
        seismic_activity = 0.7 * stress_changes + np.random.normal(0, 0.2, n_samples)

        # Add some confounding variables
        geological_age = np.random.uniform(1e6, 1e9, n_samples)
        depth = np.random.uniform(1000, 10000, n_samples)

        data = pd.DataFrame(
            {
                "magma_pressure": magma_pressure,
                "temperature": temperature,
                "stress_changes": stress_changes,
                "ground_deformation": ground_deformation,
                "gas_emissions": gas_emissions,
                "seismic_activity": seismic_activity,
                "geological_age": geological_age,
                "depth": depth,
            }
        )

        return data

    @pytest.fixture
    def domain_knowledge(self):
        """Create sample geological domain knowledge"""
        return GeologicalDomainKnowledge(
            known_causal_relationships={
                "magma_pressure": ["ground_deformation"],
                "temperature": ["gas_emissions"],
                "stress_changes": ["seismic_activity"],
            },
            impossible_relationships={
                "ground_deformation": ["magma_pressure"],
                "gas_emissions": ["temperature"],
                "seismic_activity": ["stress_changes"],
            },
            temporal_constraints={
                "magma_pressure": {"before": "ground_deformation"},
                "temperature": {"before": "gas_emissions"},
            },
            physical_constraints={
                "pressure": {"min": 0, "max": 1e9, "unit": "Pa"},
                "temperature": {"min": 273, "max": 2000, "unit": "K"},
            },
            geological_process_hierarchies={
                "volcanic_processes": [
                    "magma_generation",
                    "magma_transport",
                    "eruption",
                ]
            },
        )

    @pytest.fixture
    def causal_engine(self, domain_knowledge):
        """Create CausalInferenceEngine instance"""
        return CausalInferenceEngine(domain_knowledge=domain_knowledge)

    def test_initialization(self, causal_engine):
        """Test engine initialization"""
        assert causal_engine is not None
        assert causal_engine.domain_knowledge is not None
        assert causal_engine.scaler is not None
        assert isinstance(causal_engine.discovered_relationships, dict)
        assert isinstance(causal_engine.validation_results, dict)

    def test_data_preprocessing(self, causal_engine, sample_geological_data):
        """Test data preprocessing functionality"""
        processed_data = causal_engine._preprocess_data(sample_geological_data)

        # Check that data is properly scaled
        assert processed_data is not None
        assert len(processed_data) <= len(
            sample_geological_data
        )  # May remove some rows

        # Check that numerical columns are standardized (approximately)
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            mean_val = processed_data[col].mean()
            std_val = processed_data[col].std()
            assert abs(mean_val) < 0.1  # Should be close to 0
            assert abs(std_val - 1.0) < 0.1  # Should be close to 1

    def test_pc_algorithm(self, causal_engine, sample_geological_data):
        """Test PC algorithm for causal discovery"""
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.05,
        )

        assert isinstance(causal_graph, CausalGraph)
        assert causal_graph.algorithm_used == CausalAlgorithm.PC
        assert len(causal_graph.nodes) > 0
        assert len(causal_graph.edges) >= 0  # May find no significant relationships
        assert causal_graph.confidence_threshold == 0.05
        assert isinstance(causal_graph.creation_timestamp, datetime)

        # Check that known relationships are discovered (with some tolerance)
        edge_pairs = [(edge.source, edge.target) for edge in causal_graph.edges]

        # Should find at least some of the known causal relationships
        known_pairs = [
            ("magma_pressure", "ground_deformation"),
            ("temperature", "gas_emissions"),
            ("stress_changes", "seismic_activity"),
        ]

        found_relationships = sum(1 for pair in known_pairs if pair in edge_pairs)
        assert found_relationships >= 1  # Should find at least one known relationship

    def test_fci_algorithm(self, causal_engine, sample_geological_data):
        """Test FCI algorithm for causal discovery"""
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.FCI,
            confidence_threshold=0.05,
        )

        assert isinstance(causal_graph, CausalGraph)
        assert causal_graph.algorithm_used == CausalAlgorithm.FCI
        assert len(causal_graph.nodes) > 0
        assert isinstance(causal_graph.creation_timestamp, datetime)

    def test_ges_algorithm(self, causal_engine, sample_geological_data):
        """Test GES algorithm for causal discovery"""
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.GES,
            confidence_threshold=0.05,
        )

        assert isinstance(causal_graph, CausalGraph)
        assert causal_graph.algorithm_used == CausalAlgorithm.GES
        assert len(causal_graph.nodes) > 0
        assert isinstance(causal_graph.creation_timestamp, datetime)

    def test_conditional_independence_test(self, causal_engine, sample_geological_data):
        """Test conditional independence testing"""
        # Test marginal independence
        is_independent = causal_engine._test_conditional_independence(
            sample_geological_data, "magma_pressure", "geological_age", [], 0.05
        )
        assert isinstance(is_independent, bool)

        # Test conditional independence
        is_independent_cond = causal_engine._test_conditional_independence(
            sample_geological_data,
            "magma_pressure",
            "ground_deformation",
            ["depth"],
            0.05,
        )
        assert isinstance(is_independent_cond, bool)

    def test_edge_strength_calculation(self, causal_engine, sample_geological_data):
        """Test edge strength calculation"""
        strength, p_value = causal_engine._calculate_edge_strength(
            sample_geological_data, "magma_pressure", "ground_deformation"
        )

        assert isinstance(strength, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        assert strength >= 0  # Mutual information is non-negative

        # Known causal relationship should have higher strength
        strength_causal, _ = causal_engine._calculate_edge_strength(
            sample_geological_data, "magma_pressure", "ground_deformation"
        )
        strength_random, _ = causal_engine._calculate_edge_strength(
            sample_geological_data, "magma_pressure", "geological_age"
        )

        assert strength_causal >= strength_random

    def test_domain_knowledge_validation(self, causal_engine):
        """Test domain knowledge validation"""
        # Test valid causal direction
        is_valid = causal_engine._is_causal_direction_valid(
            "magma_pressure", "ground_deformation"
        )
        assert is_valid  is True

        # Test invalid causal direction
        is_invalid = causal_engine._is_causal_direction_valid(
            "ground_deformation", "magma_pressure"
        )
        assert is_invalid  is False

        # Test unknown relationship (should be allowed)
        is_unknown = causal_engine._is_causal_direction_valid(
            "unknown_var1", "unknown_var2"
        )
        assert is_unknown  is True

    def test_intervention_analysis(self, causal_engine, sample_geological_data):
        """Test causal intervention analysis"""
        # First discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.1,  # More lenient for testing
        )

        # Perform intervention analysis
        if len(causal_graph.edges) > 0:
            # Find a variable that has outgoing edges
            intervention_var = None
            for edge in causal_graph.edges:
                if edge.edge_type == EdgeType.DIRECTED:
                    intervention_var = edge.source
                    break

            if intervention_var:
                intervention_result = causal_engine.perform_intervention_analysis(
                    causal_graph=causal_graph,
                    data=sample_geological_data,
                    intervention_variable=intervention_var,
                    intervention_value=1.0,
                )

                assert isinstance(intervention_result, InterventionResult)
                assert intervention_result.intervention_variable == intervention_var
                assert isinstance(intervention_result.causal_effects, dict)
                assert isinstance(intervention_result.confidence_intervals, dict)
                assert isinstance(intervention_result.geological_interpretation, str)
                assert len(intervention_result.geological_interpretation) > 0

    def test_causal_graph_validation(self, causal_engine, sample_geological_data):
        """Test causal graph validation with independent data"""
        # Discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.1,
        )

        # Create validation data (similar structure but different samples)
        np.random.seed(123)  # Different seed for validation data
        validation_data = sample_geological_data.sample(n=100, replace=True)

        # Validate relationships
        validation_results = causal_engine.validate_causal_relationships(
            causal_graph=causal_graph, validation_data=validation_data
        )

        assert isinstance(validation_results, dict)
        assert "total_edges" in validation_results
        assert "validated_edges" in validation_results
        assert "failed_edges" in validation_results
        assert "overall_accuracy" in validation_results
        assert "domain_knowledge_consistency" in validation_results

        assert validation_results["total_edges"] == len(causal_graph.edges)
        assert 0 <= validation_results["overall_accuracy"] <= 1
        assert 0 <= validation_results["domain_knowledge_consistency"] <= 1

    def test_hypothesis_generation(self, causal_engine, sample_geological_data):
        """Test hypothesis generation from causal graph"""
        # Discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.1,
        )

        # Generate hypotheses
        hypotheses = causal_engine.generate_hypothesis_from_graph(causal_graph)

        assert isinstance(hypotheses, list)
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, str)
            assert len(hypothesis) > 0
            assert "Hypothesis:" in hypothesis
            assert "Test:" in hypothesis or "Expected:" in hypothesis

    def test_causal_graph_export(self, causal_engine, sample_geological_data):
        """Test causal graph export functionality"""
        # Discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.1,
        )

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            causal_engine.export_causal_graph(causal_graph, f.name)

            # Read back and validate
            with open(f.name, "r") as read_f:
                exported_data = json.load(read_f)

                assert "nodes" in exported_data
                assert "edges" in exported_data
                assert "algorithm_used" in exported_data
                assert "confidence_threshold" in exported_data
                assert "creation_timestamp" in exported_data

                assert exported_data["algorithm_used"] == CausalAlgorithm.PC.value
                assert exported_data["confidence_threshold"] == 0.1
                assert len(exported_data["nodes"]) == len(causal_graph.nodes)
                assert len(exported_data["edges"]) == len(causal_graph.edges)

    def test_networkx_conversion(self, causal_engine, sample_geological_data):
        """Test conversion to NetworkX graph"""
        # Discover causal relationships
        causal_graph = causal_engine.discover_causal_relationships(
            data=sample_geological_data,
            algorithm=CausalAlgorithm.PC,
            confidence_threshold=0.1,
        )

        # Convert to NetworkX
        nx_graph = causal_graph.to_networkx()

        assert nx_graph is not None
        assert len(nx_graph.nodes()) == len(causal_graph.nodes)

        # Count directed edges
        directed_edges = [
            edge for edge in causal_graph.edges if edge.edge_type == EdgeType.DIRECTED
        ]
        assert len(nx_graph.edges()) == len(directed_edges)

    def test_error_handling(self, causal_engine):
        """Test error handling in various scenarios"""
        # Test with empty data
        empty_data = pd.DataFrame()

        with pytest.raises(Exception):
            causal_engine.discover_causal_relationships(
                data=empty_data, algorithm=CausalAlgorithm.PC
            )

        # Test with insufficient data
        small_data = pd.DataFrame({"var1": [1, 2], "var2": [3, 4]})

        # Should handle gracefully (may return empty graph)
        result = causal_engine.discover_causal_relationships(
            data=small_data, algorithm=CausalAlgorithm.PC
        )
        assert isinstance(result, CausalGraph)

    def test_bic_score_calculation(self, causal_engine, sample_geological_data):
        """Test BIC score calculation for GES algorithm"""
        n_vars = 3
        variables = ["var1", "var2", "var3"]
        test_data = sample_geological_data[
            ["magma_pressure", "temperature", "stress_changes"]
        ].copy()
        test_data.columns = variables

        # Test empty graph
        empty_adjacency = np.zeros((n_vars, n_vars))
        score_empty = causal_engine._calculate_bic_score(
            test_data, empty_adjacency, variables
        )
        assert isinstance(score_empty, float)

        # Test graph with one edge
        one_edge_adjacency = np.zeros((n_vars, n_vars))
        one_edge_adjacency[0, 1] = 1
        score_one_edge = causal_engine._calculate_bic_score(
            test_data, one_edge_adjacency, variables
        )
        assert isinstance(score_one_edge, float)

    def test_cycle_detection(self, causal_engine):
        """Test cycle detection in graphs"""
        # Test acyclic graph
        acyclic_matrix = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        assert not causal_engine._creates_cycle(acyclic_matrix)

        # Test cyclic graph
        cyclic_matrix = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        assert causal_engine._creates_cycle(cyclic_matrix)

    def test_geological_interpretation_generation(self, causal_engine):
        """Test geological interpretation generation"""
        # Test known relationship
        interpretation = causal_engine._generate_geological_interpretation(
            "magma_pressure", "ground_deformation", 0.75
        )
        assert isinstance(interpretation, str)
        assert len(interpretation) > 0
        assert "magma pressure" in interpretation.lower()
        assert "ground deformation" in interpretation.lower()

        # Test unknown relationship
        unknown_interpretation = causal_engine._generate_geological_interpretation(
            "unknown_var1", "unknown_var2", 0.5
        )
        assert isinstance(unknown_interpretation, str)
        assert len(unknown_interpretation) > 0
        assert "unknown_var1" in unknown_interpretation
        assert "unknown_var2" in unknown_interpretation

    def test_causal_descendants_finding(self, causal_engine):
        """Test finding causal descendants in graph"""
        # Create a simple causal graph
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED, 0.5, 0.8, 0.01),
            CausalEdge("B", "C", EdgeType.DIRECTED, 0.6, 0.9, 0.005),
            CausalEdge("A", "D", EdgeType.DIRECTED, 0.4, 0.7, 0.02),
        ]

        causal_graph = CausalGraph(
            nodes=["A", "B", "C", "D"],
            edges=edges,
            algorithm_used=CausalAlgorithm.PC,
            confidence_threshold=0.05,
            geological_domain_knowledge={},
            creation_timestamp=datetime.now(),
        )

        # Find descendants of A
        descendants = causal_engine._find_causal_descendants(causal_graph, "A")
        assert "B" in descendants
        assert "C" in descendants  # Indirect descendant
        assert "D" in descendants

        # Find descendants of B
        descendants_b = causal_engine._find_causal_descendants(causal_graph, "B")
        assert "C" in descendants_b
        assert "A" not in descendants_b
        assert "D" not in descendants_b

class TestCausalGraphDataStructures:
    """Test causal graph data structures"""

    def test_causal_edge_creation(self):
        """Test CausalEdge creation and properties"""
        edge = CausalEdge(
            source="magma_pressure",
            target="ground_deformation",
            edge_type=EdgeType.DIRECTED,
            strength=0.75,
            confidence=0.95,
            p_value=0.001,
            geological_interpretation="Magma pressure causes ground deformation",
        )

        assert edge.source == "magma_pressure"
        assert edge.target == "ground_deformation"
        assert edge.edge_type == EdgeType.DIRECTED
        assert edge.strength == 0.75
        assert edge.confidence == 0.95
        assert edge.p_value == 0.001
        assert "Magma pressure" in edge.geological_interpretation

    def test_causal_graph_creation(self):
        """Test CausalGraph creation and properties"""
        edges = [
            CausalEdge("A", "B", EdgeType.DIRECTED, 0.5, 0.8, 0.01),
            CausalEdge("B", "C", EdgeType.DIRECTED, 0.6, 0.9, 0.005),
        ]

        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges=edges,
            algorithm_used=CausalAlgorithm.PC,
            confidence_threshold=0.05,
            geological_domain_knowledge={"test": "data"},
            creation_timestamp=datetime.now(),
        )

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.algorithm_used == CausalAlgorithm.PC
        assert graph.confidence_threshold == 0.05
        assert isinstance(graph.creation_timestamp, datetime)

    def test_intervention_result_creation(self):
        """Test InterventionResult creation and properties"""
        result = InterventionResult(
            intervention_variable="magma_pressure",
            target_variables=["ground_deformation", "seismic_activity"],
            causal_effects={"ground_deformation": 0.5, "seismic_activity": 0.2},
            confidence_intervals={
                "ground_deformation": (0.3, 0.7),
                "seismic_activity": (0.1, 0.3),
            },
            geological_interpretation="Intervention analysis results",
            statistical_significance={
                "ground_deformation": 0.01,
                "seismic_activity": 0.05,
            },
        )

        assert result.intervention_variable == "magma_pressure"
        assert len(result.target_variables) == 2
        assert len(result.causal_effects) == 2
        assert len(result.confidence_intervals) == 2
        assert len(result.statistical_significance) == 2
        assert isinstance(result.geological_interpretation, str)

class TestGeologicalDomainKnowledge:
    """Test geological domain knowledge functionality"""

    def test_domain_knowledge_creation(self):
        """Test GeologicalDomainKnowledge creation"""
        knowledge = GeologicalDomainKnowledge(
            known_causal_relationships={"A": ["B", "C"]},
            impossible_relationships={"B": ["A"]},
            temporal_constraints={"A": {"before": "B"}},
            physical_constraints={"pressure": {"min": 0, "max": 1e9}},
            geological_process_hierarchies={"volcanic": ["magma", "eruption"]},
        )

        assert "A" in knowledge.known_causal_relationships
        assert "B" in knowledge.known_causal_relationships["A"]
        assert "B" in knowledge.impossible_relationships
        assert "A" in knowledge.temporal_constraints
        assert "pressure" in knowledge.physical_constraints
        assert "volcanic" in knowledge.geological_process_hierarchies

if __name__ == "__main__":
    pytest.main([__file__])
