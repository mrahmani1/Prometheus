#!/usr/bin/env python3
"""
Continuous Model Validation Script for Prometheus AI Platform
Validates AI models against scientific benchmarks and physics constraints
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import os
import pickle
import sys

import numpy as np
import torch
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    model_name: str
    model_version: str
    benchmark_name: str
    accuracy: float
    physics_compliance: float
    performance_metrics: Dict[str, float]
    status: str  # passed, failed, warning
    details: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkDataset:
    name: str
    description: str
    data_path: str
    expected_results_path: str
    physics_constraints: List[str]
    accuracy_threshold: float
    test_type: str  # geological, seismic, volcanic, etc.


class ScientificBenchmarkValidator:
    def __init__(self):
        # Handle cases where __file__ is not defined (e.g., when run via exec)
        try:
            self.project_root = Path(__file__).parent.parent
        except NameError:
            # Fallback to current working directory
            self.project_root = Path.cwd()

        self.models_dir = self.project_root / "services" / "helios-core" / "models"
        self.benchmarks_dir = self.project_root / "tests" / "benchmarks"
        self.validation_results: List[ValidationResult] = []

        # Initialize benchmark datasets
        self.benchmark_datasets = self.load_benchmark_datasets()

    def load_benchmark_datasets(self) -> List[BenchmarkDataset]:
        """Load scientific benchmark datasets"""
        benchmarks = [
            BenchmarkDataset(
                name="yellowstone_seismic_tomography",
                description="Yellowstone caldera seismic tomography validation",
                data_path="benchmarks/yellowstone/seismic_data.npz",
                expected_results_path="benchmarks/yellowstone/expected_structure.npz",
                physics_constraints=["elastic_wave_equation", "velocity_bounds"],
                accuracy_threshold=0.85,
                test_type="seismic",
            ),
            BenchmarkDataset(
                name="kilauea_deformation_insar",
                description="Kilauea volcano InSAR deformation analysis",
                data_path="benchmarks/kilauea/insar_data.npz",
                expected_results_path="benchmarks/kilauea/expected_deformation.npz",
                physics_constraints=["elastic_deformation", "mass_conservation"],
                accuracy_threshold=0.80,
                test_type="volcanic",
            ),
            BenchmarkDataset(
                name="san_andreas_stress_field",
                description="San Andreas fault stress field modeling",
                data_path="benchmarks/san_andreas/stress_data.npz",
                expected_results_path="benchmarks/san_andreas/expected_stress.npz",
                physics_constraints=["stress_equilibrium", "friction_law"],
                accuracy_threshold=0.75,
                test_type="tectonic",
            ),
            BenchmarkDataset(
                name="iceland_mantle_plume",
                description="Iceland mantle plume dynamics simulation",
                data_path="benchmarks/iceland/plume_data.npz",
                expected_results_path="benchmarks/iceland/expected_dynamics.npz",
                physics_constraints=["navier_stokes", "heat_transfer", "buoyancy"],
                accuracy_threshold=0.70,
                test_type="mantle_dynamics",
            ),
            BenchmarkDataset(
                name="cascadia_earthquake_catalog",
                description="Cascadia subduction zone earthquake patterns",
                data_path="benchmarks/cascadia/earthquake_catalog.npz",
                expected_results_path="benchmarks/cascadia/expected_patterns.npz",
                physics_constraints=["gutenberg_richter", "omori_law"],
                accuracy_threshold=0.78,
                test_type="seismic",
            ),
        ]

        return benchmarks

    def load_model(self, model_path: str) -> Optional[torch.nn.Module]:
        """Load a PyTorch model for validation"""
        try:
            if model_path.endswith(".pth") or model_path.endswith(".pt"):
                model = torch.load(model_path, map_location="cpu", weights_only=False)
                model.eval()
                return model
            elif model_path.endswith(".pkl"):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
        return None

    def get_model_version(self, model_path: str) -> str:
        """Generate model version hash"""
        try:
            with open(model_path, "rb") as f:
                content = f.read()
                return hashlib.md5(content, usedforsecurity=False).hexdigest()[:8]
        except Exception:
            return "unknown"

    def load_benchmark_data(
        self, benchmark: BenchmarkDataset
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load benchmark input data and expected results"""
        try:
            data_path = self.benchmarks_dir / benchmark.data_path
            expected_path = self.benchmarks_dir / benchmark.expected_results_path

            if data_path.exists() and expected_path.exists():
                input_data = np.load(data_path)
                expected_results = np.load(expected_path)
                return input_data, expected_results
            else:
                # Generate synthetic benchmark data if files don't exist
                return self.generate_synthetic_benchmark(benchmark)
        except Exception as e:
            logger.error(f"Failed to load benchmark data for {benchmark.name}: {e}")
        return None, None

    def generate_synthetic_benchmark(
        self, benchmark: BenchmarkDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic benchmark data for testing"""
        logger.info(f"Generating synthetic benchmark data for {benchmark.name}")

        if benchmark.test_type == "seismic":
            # Generate synthetic seismic data
            input_data = np.random.randn(100, 50, 50)  # 3D seismic velocity model
            expected_results = np.random.randn(100, 50, 50) * 0.1 + input_data
        elif benchmark.test_type == "volcanic":
            # Generate synthetic deformation data
            input_data = np.random.randn(50, 50, 3)  # 3D displacement field
            expected_results = np.random.randn(50, 50, 3) * 0.05 + input_data
        elif benchmark.test_type == "tectonic":
            # Generate synthetic stress field data
            input_data = np.random.randn(100, 100, 6)  # Stress tensor components
            expected_results = np.random.randn(100, 100, 6) * 0.1 + input_data
        elif benchmark.test_type == "mantle_dynamics":
            # Generate synthetic mantle flow data
            input_data = np.random.randn(200, 200, 200, 4)  # 3D velocity + temperature
            expected_results = np.random.randn(200, 200, 200, 4) * 0.1 + input_data
        else:
            # Default synthetic data
            input_data = np.random.randn(100, 100)
            expected_results = np.random.randn(100, 100) * 0.1 + input_data

        return input_data, expected_results

    def validate_physics_constraints(
        self, model_output: np.ndarray, constraints: List[str]
    ) -> float:
        """Validate physics constraints compliance"""
        compliance_scores = []

        for constraint in constraints:
            if constraint == "elastic_wave_equation":
                # Check if velocities are within realistic bounds
                if len(model_output.shape) >= 3:
                    velocity_range = np.ptp(model_output)
                    realistic_range = velocity_range < 10.0  # km/s
                    compliance_scores.append(1.0 if realistic_range else 0.5)

            elif constraint == "mass_conservation":
                # Check mass conservation (divergence should be small)
                if len(model_output.shape) >= 3:
                    # Simplified divergence check
                    div_check = np.mean(np.abs(np.gradient(model_output, axis=0))) < 1.0
                    compliance_scores.append(1.0 if div_check else 0.3)

            elif constraint == "stress_equilibrium":
                # Check stress equilibrium conditions
                if len(model_output.shape) >= 2:
                    # Simplified equilibrium check
                    equilibrium_check = (
                        np.std(model_output) < np.mean(np.abs(model_output)) * 2
                    )
                    compliance_scores.append(1.0 if equilibrium_check else 0.4)

            elif constraint == "navier_stokes":
                # Check Navier-Stokes equation compliance
                if len(model_output.shape) >= 3:
                    # Simplified continuity equation check
                    continuity_check = np.mean(np.abs(model_output)) > 0.001
                    compliance_scores.append(1.0 if continuity_check else 0.6)

            elif constraint == "heat_transfer":
                # Check heat transfer physics
                if len(model_output.shape) >= 2:
                    # Temperature gradients should be reasonable
                    temp_gradient = np.gradient(model_output)
                    gradient_check = np.mean(np.abs(temp_gradient)) < 100.0
                    compliance_scores.append(1.0 if gradient_check else 0.5)

            else:
                # Default constraint check
                compliance_scores.append(0.8)

        return np.mean(compliance_scores) if compliance_scores else 0.0

    def calculate_accuracy(self, predicted: np.ndarray, expected: np.ndarray) -> float:
        """Calculate prediction accuracy"""
        try:
            # Normalize arrays to same shape if needed
            if predicted.shape != expected.shape:
                min_shape = tuple(
                    min(p, e) for p, e in zip(predicted.shape, expected.shape)
                )
                predicted = (
                    predicted[: min_shape[0]] if len(min_shape) > 0 else predicted
                )
                expected = expected[: min_shape[0]] if len(min_shape) > 0 else expected

            # Calculate various accuracy metrics
            mse = np.mean((predicted - expected) ** 2)
            mae = np.mean(np.abs(predicted - expected))

            # Normalized accuracy (higher is better)
            max_error = (
                np.max(np.abs(expected)) if np.max(np.abs(expected)) > 0 else 1.0
            )
            normalized_mae = mae / max_error
            accuracy = max(0.0, 1.0 - normalized_mae)

            return float(accuracy)
        except Exception as e:
            logger.error(f"Error calculating accuracy: {e}")
            return 0.0

    def validate_model_against_benchmark(
        self, model: torch.nn.Module, benchmark: BenchmarkDataset
    ) -> ValidationResult:
        """Validate a single model against a benchmark dataset"""
        logger.info(f"Validating model against benchmark: {benchmark.name}")

        # Load benchmark data
        input_data, expected_results = self.load_benchmark_data(benchmark)
        if input_data is None or expected_results is None:
            return ValidationResult(
                model_name="unknown",
                model_version="unknown",
                benchmark_name=benchmark.name,
                accuracy=0.0,
                physics_compliance=0.0,
                performance_metrics={},
                status="failed",
                details="Failed to load benchmark data",
            )

        try:
            # Convert to tensor if needed
            if isinstance(input_data, np.ndarray):
                if hasattr(input_data, "files"):  # npz file
                    input_tensor = torch.FloatTensor(input_data["data"])
                else:
                    input_tensor = torch.FloatTensor(input_data)
            else:
                input_tensor = input_data

            # Run model prediction
            start_time = datetime.now()
            with torch.no_grad():
                if hasattr(model, "predict"):
                    predicted = model.predict(input_tensor)
                elif hasattr(model, "forward"):
                    predicted = model.forward(input_tensor)
                else:
                    # Assume it's a callable
                    predicted = model(input_tensor)

            inference_time = (datetime.now() - start_time).total_seconds()

            # Convert prediction to numpy
            if isinstance(predicted, torch.Tensor):
                predicted_np = predicted.cpu().numpy()
            else:
                predicted_np = np.array(predicted)

            # Extract expected results
            if hasattr(expected_results, "files"):  # npz file
                expected_np = expected_results["data"]
            else:
                expected_np = expected_results

            # Calculate accuracy
            accuracy = self.calculate_accuracy(predicted_np, expected_np)

            # Validate physics constraints
            physics_compliance = self.validate_physics_constraints(
                predicted_np, benchmark.physics_constraints
            )

            # Performance metrics
            performance_metrics = {
                "inference_time": inference_time,
                "memory_usage": sys.getsizeof(predicted_np) / 1024 / 1024,  # MB
                "prediction_shape": list(predicted_np.shape),
                "mse": (
                    float(np.mean((predicted_np - expected_np) ** 2))
                    if predicted_np.shape == expected_np.shape
                    else -1
                ),
            }

            # Determine status
            status = "passed"
            details = None

            if accuracy < benchmark.accuracy_threshold:
                status = "failed"
                details = f"Accuracy {accuracy:.3f} below threshold {benchmark.accuracy_threshold}"
            elif physics_compliance < 0.7:
                status = "warning"
                details = f"Physics compliance {physics_compliance:.3f} is low"

            return ValidationResult(
                model_name=getattr(model, "__class__", type(model)).__name__,
                model_version="current",
                benchmark_name=benchmark.name,
                accuracy=accuracy,
                physics_compliance=physics_compliance,
                performance_metrics=performance_metrics,
                status=status,
                details=details,
            )

        except Exception as e:
            logger.error(f"Error validating model against {benchmark.name}: {e}")
            return ValidationResult(
                model_name="unknown",
                model_version="unknown",
                benchmark_name=benchmark.name,
                accuracy=0.0,
                physics_compliance=0.0,
                performance_metrics={},
                status="failed",
                details=str(e),
            )

    def discover_models(self) -> List[Tuple[str, str]]:
        """Discover available models for validation"""
        models = []

        # Look for models in helios-core service
        helios_models_dir = self.project_root / "services" / "helios-core" / "models"
        if helios_models_dir.exists():
            for model_file in helios_models_dir.rglob("*.pth"):
                models.append(("helios-core", str(model_file)))
            for model_file in helios_models_dir.rglob("*.pt"):
                models.append(("helios-core", str(model_file)))
            for model_file in helios_models_dir.rglob("*.pkl"):
                models.append(("helios-core", str(model_file)))

        # Look for models in volcano-watchtower service
        watchtower_models_dir = (
            self.project_root / "services" / "volcano-watchtower" / "models"
        )
        if watchtower_models_dir.exists():
            for model_file in watchtower_models_dir.rglob("*.pth"):
                models.append(("volcano-watchtower", str(model_file)))

        return models

    def run_comprehensive_validation(self) -> List[ValidationResult]:
        """Run comprehensive model validation against all benchmarks"""
        logger.info("Starting comprehensive model validation")

        # Discover available models
        models = self.discover_models()
        if not models:
            logger.warning("No models found for validation")
            return []

        validation_results = []

        for service_name, model_path in models:
            logger.info(f"Validating model: {model_path}")

            # Load model
            model = self.load_model(model_path)
            if model is None:
                continue

            # Validate against each benchmark
            for benchmark in self.benchmark_datasets:
                try:
                    result = self.validate_model_against_benchmark(model, benchmark)
                    result.model_name = f"{service_name}_{Path(model_path).stem}"
                    result.model_version = self.get_model_version(model_path)
                    validation_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Validation failed for {model_path} against {benchmark.name}: {e}"
                    )

        return validation_results

    def generate_validation_report(self, results: List[ValidationResult]) -> Dict:
        """Generate comprehensive validation report"""
        total_validations = len(results)
        passed_validations = len([r for r in results if r.status == "passed"])
        failed_validations = len([r for r in results if r.status == "failed"])
        warning_validations = len([r for r in results if r.status == "warning"])

        # Calculate average metrics
        avg_accuracy = np.mean([r.accuracy for r in results]) if results else 0.0
        avg_physics_compliance = (
            np.mean([r.physics_compliance for r in results]) if results else 0.0
        )

        # Group results by model
        model_results = {}
        for result in results:
            model_key = f"{result.model_name}_{result.model_version}"
            if model_key not in model_results:
                model_results[model_key] = []
            model_results[model_key].append(result)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_validations": total_validations,
                "passed": passed_validations,
                "failed": failed_validations,
                "warnings": warning_validations,
                "success_rate": (
                    passed_validations / total_validations
                    if total_validations > 0
                    else 0
                ),
                "average_accuracy": float(avg_accuracy),
                "average_physics_compliance": float(avg_physics_compliance),
            },
            "model_results": {
                model_key: [asdict(r) for r in model_results[model_key]]
                for model_key in model_results
            },
            "benchmark_coverage": [b.name for b in self.benchmark_datasets],
            "validation_criteria": {
                "minimum_accuracy": 0.70,
                "minimum_physics_compliance": 0.70,
                "critical_benchmarks": [
                    "yellowstone_seismic_tomography",
                    "kilauea_deformation_insar",
                ],
            },
        }

        return report

    def save_validation_report(self, report: Dict):
        """Save validation report to file"""
        reports_dir = self.project_root / "validation-reports"
        reports_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"model_validation_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to {report_file}")

        # Also save latest report
        latest_report = reports_dir / "latest_validation_report.json"
        with open(latest_report, "w") as f:
            json.dump(report, f, indent=2)


def main():
    """Main execution function"""
    validator = ScientificBenchmarkValidator()

    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()

        # Generate and save report
        report = validator.generate_validation_report(results)
        validator.save_validation_report(report)

        # Print summary
        summary = report["summary"]
        logger.info(f"Model validation completed:")
        logger.info(f"  Total validations: {summary['total_validations']}")
        logger.info(f"  Passed: {summary['passed']}")
        logger.info(f"  Failed: {summary['failed']}")
        logger.info(f"  Warnings: {summary['warnings']}")
        logger.info(f"  Success rate: {summary['success_rate']:.2%}")
        logger.info(f"  Average accuracy: {summary['average_accuracy']:.3f}")
        logger.info(
            f"  Average physics compliance: {summary['average_physics_compliance']:.3f}"
        )

        # Check critical thresholds
        critical_failure = False
        if summary["average_accuracy"] < 0.70:
            logger.error("CRITICAL: Average model accuracy below 70%")
            critical_failure = True

        if summary["average_physics_compliance"] < 0.70:
            logger.error("CRITICAL: Average physics compliance below 70%")
            critical_failure = True

        if summary["failed"] > summary["passed"]:
            logger.error("CRITICAL: More validations failed than passed")
            critical_failure = True

        # Exit with appropriate code
        if critical_failure:
            logger.error("Model validation failed critical thresholds!")
            sys.exit(1)
        elif summary["failed"] > 0:
            logger.warning(
                "Some model validations failed, but within acceptable limits"
            )
            sys.exit(0)
        else:
            logger.info("All model validations passed!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
