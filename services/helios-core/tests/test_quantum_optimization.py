"""
Test suite for quantum optimization algorithms for geological problems
"""

import os
import sys

import numpy as np
import pytest

# Add the parent directory to the path to import the core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_optimization import (
    GeologicalQuantumProblem,
    QuantumAnnealingOptimizer,
    QuantumClassicalHybrid,
    QuantumGeologicalOptimizationManager,
    QuantumMachineLearning,
    QuantumOptimizationResult,
    VariationalQuantumEigensolver,
    create_example_geological_problems,
)

class TestGeologicalQuantumProblem:
    """Test the GeologicalQuantumProblem data structure"""

    def test_problem_creation(self):
        """Test creating a geological quantum problem"""
        data = np.random.normal(0, 1, 10)
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=data,
            constraints={"max_value": 5.0},
            objective_function="minimize_energy",
            quantum_encoding="amplitude",
        )

        assert problem.problem_type == "parameter_optimization"
        assert len(problem.geological_data) == 10
        assert problem.constraints["max_value"] == 5.0
        assert problem.quantum_encoding == "amplitude"
        assert problem.classical_preprocessing is True

class TestQuantumAnnealingOptimizer:
    """Test quantum annealing optimizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = QuantumAnnealingOptimizer(num_qubits=6, annealing_time=10.0)
        self.test_data = np.array([1.0, -0.5, 2.0, 0.8, -1.2, 1.5])

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.num_qubits == 6
        assert self.optimizer.annealing_time == 10.0

    def test_geological_to_qubo_parameter_optimization(self):
        """Test QUBO conversion for parameter optimization"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="minimize_parameters",
            quantum_encoding="amplitude",
        )

        qubo = self.optimizer._geological_to_qubo(problem)

        assert qubo.shape == (6, 6)
        assert np.allclose(
            np.diag(qubo), -self.test_data
        )  # Diagonal should be negative data

    def test_geological_to_qubo_system_analysis(self):
        """Test QUBO conversion for system analysis"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="analyze_system",
            quantum_encoding="angle",
        )

        qubo = self.optimizer._geological_to_qubo(problem)

        assert qubo.shape == (6, 6)
        assert np.all(np.diag(qubo) == 0)  # Diagonal should be zero for system analysis

    def test_evaluate_qubo(self):
        """Test QUBO evaluation"""
        qubo = np.array([[1, 0.5], [0.5, -1]])
        solution = np.array([1, 0])

        energy = self.optimizer._evaluate_qubo(solution, qubo)
        expected_energy = 1 * 1 + 0 * (-1) + 2 * 0.5 * 1 * 0  # x1^2 + x2^2 + 2*x1*x2

        assert energy == expected_energy

    def test_simulate_quantum_annealing(self):
        """Test classical simulation of quantum annealing"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="minimize_energy",
            quantum_encoding="amplitude",
        )

        qubo = self.optimizer._geological_to_qubo(problem)
        result = self.optimizer._simulate_quantum_annealing(qubo, problem)

        assert "parameters" in result
        assert "energy" in result
        assert "history" in result
        assert "confidence" in result
        assert len(result["parameters"]) == 6
        assert len(result["history"]) > 0
        assert 0 <= result["confidence"] <= 1

    def test_optimize_parameter_optimization(self):
        """Test full optimization for parameter optimization problem"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="minimize_energy",
            quantum_encoding="amplitude",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "quantum_annealing"
        assert len(result.optimal_parameters) == 6
        assert result.execution_time > 0
        assert 0 <= result.confidence_score <= 1
        assert "optimization_type" in result.geological_interpretation

    def test_optimize_system_analysis(self):
        """Test full optimization for system analysis problem"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="analyze_connectivity",
            quantum_encoding="angle",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "quantum_annealing"
        assert "connectivity_strength" in result.geological_interpretation

class TestVariationalQuantumEigensolver:
    """Test variational quantum eigensolver"""

    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = VariationalQuantumEigensolver(num_qubits=4, max_iterations=50)
        self.test_data = np.array([0.5, -1.0, 1.5, 0.8])

    def test_optimizer_initialization(self):
        """Test VQE optimizer initialization"""
        assert self.optimizer.num_qubits == 4
        assert self.optimizer.max_iterations == 50

    def test_evaluate_geological_energy(self):
        """Test geological energy evaluation"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="find_ground_state",
            quantum_encoding="amplitude",
        )

        parameters = np.random.uniform(0, 2 * np.pi, 8)
        hamiltonian = self.test_data  # Use data as simple Hamiltonian

        energy = self.optimizer._evaluate_geological_energy(
            parameters, hamiltonian, problem
        )

        assert isinstance(energy, (float, np.floating))

    def test_simulate_vqe(self):
        """Test VQE simulation"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="find_ground_state",
            quantum_encoding="amplitude",
        )

        hamiltonian = self.test_data
        result = self.optimizer._simulate_vqe(hamiltonian, problem)

        assert "parameters" in result
        assert "energy" in result
        assert "history" in result
        assert "confidence" in result
        assert len(result["parameters"]) == 8  # 2 * num_qubits
        assert len(result["history"]) > 0

    def test_optimize_system_analysis(self):
        """Test VQE optimization for system analysis"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="analyze_ground_state",
            quantum_encoding="amplitude",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "variational_quantum_eigensolver"
        assert "stability" in result.geological_interpretation

    def test_optimize_parameter_optimization(self):
        """Test VQE optimization for parameter optimization"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="optimize_parameters",
            quantum_encoding="angle",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert "dominant_parameters" in result.geological_interpretation

class TestQuantumMachineLearning:
    """Test quantum machine learning optimizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = QuantumMachineLearning(num_qubits=6, num_layers=2)
        self.test_data = np.array([1.0, -0.5, 2.0, 0.8, -1.2, 1.5, 0.3, -0.8])

    def test_optimizer_initialization(self):
        """Test QML optimizer initialization"""
        assert self.optimizer.num_qubits == 6
        assert self.optimizer.num_layers == 2

    def test_compute_quantum_kernel(self):
        """Test quantum kernel computation"""
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([1.5, 2.5])

        kernel_matrix = self.optimizer._compute_quantum_kernel(X1, X2)

        assert kernel_matrix.shape == (3, 2)
        assert np.all(kernel_matrix >= 0)  # Kernel values should be non-negative
        assert np.all(kernel_matrix <= 1)  # Normalized kernel values

    def test_quantum_feature_transform(self):
        """Test quantum feature transformation"""
        X = self.test_data[:4].reshape(-1, 1)
        X_quantum = self.optimizer._quantum_feature_transform(X)

        assert X_quantum.shape[0] == X.shape[0]
        assert X_quantum.shape[1] > 0
        assert np.all(X_quantum >= 0)  # Probabilities should be non-negative

    def test_quantum_pattern_recognition(self):
        """Test quantum pattern recognition"""
        problem = GeologicalQuantumProblem(
            problem_type="pattern_recognition",
            geological_data=self.test_data,
            constraints={},
            objective_function="classify_patterns",
            quantum_encoding="amplitude",
        )

        feature_map = self.optimizer._create_geological_feature_map(problem)
        result = self.optimizer._quantum_pattern_recognition(feature_map, problem)

        assert "accuracy" in result
        assert "predictions" in result
        assert "true_labels" in result
        assert 0 <= result["accuracy"] <= 1

    def test_quantum_regression(self):
        """Test quantum regression"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="predict_parameters",
            quantum_encoding="angle",
        )

        feature_map = self.optimizer._create_geological_feature_map(problem)
        result = self.optimizer._quantum_regression(feature_map, problem)

        assert "accuracy" in result
        assert "predictions" in result
        assert "true_values" in result
        assert result["accuracy"] >= 0  # R² can be negative

    def test_optimize_pattern_recognition(self):
        """Test QML optimization for pattern recognition"""
        problem = GeologicalQuantumProblem(
            problem_type="pattern_recognition",
            geological_data=self.test_data,
            constraints={},
            objective_function="classify_geological_patterns",
            quantum_encoding="amplitude",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "quantum_machine_learning"
        assert "pattern_distribution" in result.geological_interpretation

class TestQuantumClassicalHybrid:
    """Test quantum-classical hybrid optimizer"""

    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = QuantumClassicalHybrid(
            num_qubits=4, classical_optimizer="adam"
        )
        self.test_data = np.array([1.0, -0.5, 2.0, 0.8, -1.2, 1.5, 0.3, -0.8])

    def test_optimizer_initialization(self):
        """Test hybrid optimizer initialization"""
        assert self.optimizer.num_qubits == 4
        assert self.optimizer.classical_optimizer == "adam"

    def test_decompose_problem(self):
        """Test problem decomposition into quantum and classical components"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="hybrid_optimization",
            quantum_encoding="amplitude",
        )

        quantum_sub, classical_sub = self.optimizer._decompose_problem(problem)

        assert "data" in quantum_sub
        assert "data" in classical_sub
        assert len(quantum_sub["data"]) <= self.optimizer.num_qubits
        assert len(classical_sub["data"]) == len(self.test_data) - len(
            quantum_sub["data"]
        )

    def test_evaluate_quantum_energy(self):
        """Test quantum energy evaluation with classical context"""
        quantum_sub = {"data": np.array([1.0, -0.5, 2.0])}
        params = np.array([0.5, 1.0, 1.5])
        context = 0.1

        energy = self.optimizer._evaluate_quantum_energy(params, quantum_sub, context)

        assert isinstance(energy, (float, np.floating))

    def test_evaluate_classical_objective(self):
        """Test classical objective evaluation with quantum context"""
        classical_sub = {"data": np.array([0.8, -1.2, 1.5])}
        params = np.array([0.9, -1.0, 1.4])
        context = 0.1

        objective = self.optimizer._evaluate_classical_objective(
            params, classical_sub, context
        )

        assert isinstance(objective, (float, np.floating))
        assert objective >= 0  # Quadratic objective should be non-negative

    def test_compute_classical_gradient(self):
        """Test classical gradient computation"""
        classical_sub = {"data": np.array([0.8, -1.2, 1.5])}
        params = np.array([0.9, -1.0, 1.4])
        context = 0.1

        gradient = self.optimizer._compute_classical_gradient(
            params, classical_sub, context
        )

        assert len(gradient) == len(params)
        assert isinstance(gradient, np.ndarray)

    def test_optimize_hybrid(self):
        """Test hybrid optimization"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="hybrid_optimization",
            quantum_encoding="amplitude",
        )

        result = self.optimizer.optimize(problem)

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "quantum_classical_hybrid"
        assert "quantum_contribution" in result.geological_interpretation
        assert "classical_contribution" in result.geological_interpretation
        assert "dominant_component" in result.geological_interpretation

class TestQuantumGeologicalOptimizationManager:
    """Test the optimization manager"""

    def setup_method(self):
        """Set up test fixtures"""
        self.manager = QuantumGeologicalOptimizationManager()
        self.test_data = np.array([1.0, -0.5, 2.0, 0.8, -1.2, 1.5])

    def test_manager_initialization(self):
        """Test manager initialization"""
        assert "quantum_annealing" in self.manager.optimizers
        assert "vqe" in self.manager.optimizers
        assert "qml" in self.manager.optimizers
        assert "hybrid" in self.manager.optimizers

    def test_select_optimal_algorithm(self):
        """Test automatic algorithm selection"""
        # Small parameter optimization problem
        problem1 = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="optimize",
            quantum_encoding="amplitude",
        )

        algo1 = self.manager._select_optimal_algorithm(problem1)
        assert algo1 == "quantum_annealing"

        # System analysis problem
        problem2 = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="analyze",
            quantum_encoding="angle",
        )

        algo2 = self.manager._select_optimal_algorithm(problem2)
        assert algo2 == "vqe"

        # Pattern recognition problem
        problem3 = GeologicalQuantumProblem(
            problem_type="pattern_recognition",
            geological_data=self.test_data,
            constraints={},
            objective_function="classify",
            quantum_encoding="basis",
        )

        algo3 = self.manager._select_optimal_algorithm(problem3)
        assert algo3 == "qml"

    def test_optimize_geological_problem_auto(self):
        """Test optimization with automatic algorithm selection"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data,
            constraints={},
            objective_function="optimize",
            quantum_encoding="amplitude",
        )

        result = self.manager.optimize_geological_problem(problem, algorithm="auto")

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "quantum_annealing"

    def test_optimize_geological_problem_specific(self):
        """Test optimization with specific algorithm"""
        problem = GeologicalQuantumProblem(
            problem_type="system_analysis",
            geological_data=self.test_data,
            constraints={},
            objective_function="analyze",
            quantum_encoding="angle",
        )

        result = self.manager.optimize_geological_problem(problem, algorithm="vqe")

        assert isinstance(result, QuantumOptimizationResult)
        assert result.algorithm_type == "variational_quantum_eigensolver"

    def test_benchmark_quantum_algorithms(self):
        """Test benchmarking all algorithms"""
        problem = GeologicalQuantumProblem(
            problem_type="parameter_optimization",
            geological_data=self.test_data[:4],  # Smaller data for faster testing
            constraints={},
            objective_function="optimize",
            quantum_encoding="amplitude",
        )

        results = self.manager.benchmark_quantum_algorithms(problem)

        assert isinstance(results, dict)
        assert len(results) > 0  # At least some algorithms should succeed

        for algo_name, result in results.items():
            assert isinstance(result, QuantumOptimizationResult)
            assert result.algorithm_type in [
                "quantum_annealing",
                "variational_quantum_eigensolver",
                "quantum_machine_learning",
                "quantum_classical_hybrid",
            ]

    def test_generate_quantum_optimization_report(self):
        """Test report generation"""
        # Create mock results
        mock_result1 = QuantumOptimizationResult(
            optimal_parameters=np.array([1, 0, 1]),
            optimal_energy=-2.5,
            convergence_history=[0, -1, -2, -2.5],
            quantum_advantage_metric=0.7,
            execution_time=1.5,
            algorithm_type="quantum_annealing",
            geological_interpretation={"test": "interpretation"},
            confidence_score=0.8,
        )

        mock_result2 = QuantumOptimizationResult(
            optimal_parameters=np.array([0.5, 1.2, -0.3]),
            optimal_energy=-1.8,
            convergence_history=[0, -0.5, -1.2, -1.8],
            quantum_advantage_metric=0.6,
            execution_time=2.1,
            algorithm_type="vqe",
            geological_interpretation={"test": "interpretation2"},
            confidence_score=0.75,
        )

        results = {"quantum_annealing": mock_result1, "vqe": mock_result2}

        report = self.manager.generate_quantum_optimization_report(results)

        assert "timestamp" in report
        assert "algorithms_compared" in report
        assert "performance_comparison" in report
        assert "recommendations" in report
        assert "quantum_advantage_analysis" in report

        assert len(report["algorithms_compared"]) == 2
        assert "quantum_annealing" in report["performance_comparison"]
        assert "vqe" in report["performance_comparison"]

        assert (
            report["quantum_advantage_analysis"]["quantum_advantage_achieved"] is True
        )

class TestExampleProblems:
    """Test example problem creation and execution"""

    def test_create_example_geological_problems(self):
        """Test creation of example problems"""
        problems = create_example_geological_problems()

        assert len(problems) == 3
        assert problems[0].problem_type == "parameter_optimization"
        assert problems[1].problem_type == "system_analysis"
        assert problems[2].problem_type == "pattern_recognition"

        for problem in problems:
            assert isinstance(problem, GeologicalQuantumProblem)
            assert len(problem.geological_data) > 0
            assert problem.objective_function is not None
            assert problem.quantum_encoding in ["amplitude", "angle", "basis"]

@pytest.mark.asyncio
async def test_run_quantum_optimization_example():
    """Test the complete example workflow"""
    # Import the example function
    from core.quantum_optimization import run_quantum_optimization_example

import numpy as np
import torch.nn as nn

# Run the example (with reduced problem sizes for faster testing)
results = await run_quantum_optimization_example()

assert isinstance(results, dict)
assert len(results) == 3

    for problem_key, result in results.items():
        assert isinstance(result, QuantumOptimizationResult)
        assert result.execution_time > 0
        assert result.algorithm_type in [
            "quantum_annealing",
            "variational_quantum_eigensolver",
            "quantum_machine_learning",
            "quantum_classical_hybrid",
        ]

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
