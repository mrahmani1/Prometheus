"""
Edge AI Model Optimization Pipeline

This module provides comprehensive model optimization capabilities for deploying
AI models to resource-constrained edge devices. It includes model compression,
quantization, pruning, and automated optimization pipeline generation.

Requirements addressed: 10.1, 10.5
"""

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraints:
    """Hardware constraints for edge deployment"""

    max_memory_mb: int
    max_inference_time_ms: float
    cpu_cores: int
    has_gpu: bool = False
    gpu_memory_mb: Optional[int] = None
    target_platform: str = "cpu"  # cpu, cuda, mobile, embedded


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""

    compression_ratio: float = 0.5
    quantization_bits: int = 8
    pruning_sparsity: float = 0.3
    enable_quantization: bool = True
    enable_pruning: bool = True
    enable_knowledge_distillation: bool = True
    validation_threshold: float = 0.95  # Minimum accuracy retention


@dataclass
class OptimizationResult:
    """Results from model optimization"""

    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    original_inference_time_ms: float
    optimized_inference_time_ms: float
    speedup_factor: float
    accuracy_retention: float
    memory_usage_mb: float
    meets_constraints: bool
    optimization_steps: List[str]


class ModelCompressor:
    """Implements various model compression techniques"""

    def __init__(self):
        self.compression_methods = {
            "weight_sharing": self._weight_sharing_compression,
            "low_rank_approximation": self._low_rank_approximation,
            "huffman_coding": self._huffman_coding_compression,
        }

    def compress_model(
        self,
        model: nn.Module,
        method: str = "weight_sharing",
        compression_ratio: float = 0.5,
    ) -> nn.Module:
        """
        Compress model using specified method

        Args:
            model: PyTorch model to compress
            method: Compression method to use
            compression_ratio: Target compression ratio

        Returns:
            Compressed model
        """
        if method not in self.compression_methods:
            raise ValueError(f"Unknown compression method: {method}")

        logger.info(f"Compressing model using {method} with ratio {compression_ratio}")
        return self.compression_methods[method](model, compression_ratio)

    def _weight_sharing_compression(self, model: nn.Module, ratio: float) -> nn.Module:
        """Implement weight sharing compression"""
        compressed_model = copy.deepcopy(model)

        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Cluster weights and share values
                weights = module.weight.data.flatten()
                n_clusters = int(len(weights) * ratio)

                # Simple k-means clustering for weight sharing
                centroids = self._kmeans_clustering(weights, n_clusters)

                # Replace weights with nearest centroids
                centroids_tensor = torch.tensor(
                    centroids, device=weights.device, dtype=weights.dtype
                )
                distances = torch.abs(
                    weights.unsqueeze(1) - centroids_tensor.unsqueeze(0)
                )
                closest_indices = torch.argmin(distances, dim=1)
                weights[:] = centroids_tensor[closest_indices]

                module.weight.data = weights.reshape(module.weight.shape)

        return compressed_model

    def _low_rank_approximation(self, model: nn.Module, ratio: float) -> nn.Module:
        """Implement low-rank approximation compression"""
        compressed_model = copy.deepcopy(model)

        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # SVD decomposition for low-rank approximation
                U, S, V = torch.svd(module.weight.data)
                rank = int(module.weight.size(0) * ratio)

                # Create low-rank approximation
                module.weight.data = torch.mm(U[:, :rank] * S[:rank], V[:, :rank].t())

        return compressed_model

    def _huffman_coding_compression(self, model: nn.Module, ratio: float) -> nn.Module:
        """Implement Huffman coding compression (simplified)"""
        # This is a simplified version - in practice, you'd implement full Huffman coding
        compressed_model = copy.deepcopy(model)

        for name, module in compressed_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Quantize weights to reduce entropy
                weights = module.weight.data
                quantized_weights = torch.round(weights * (1 / ratio)) * ratio
                module.weight.data = quantized_weights

        return compressed_model

    def _kmeans_clustering(self, data: torch.Tensor, n_clusters: int) -> List[float]:
        """Simple k-means clustering for weight sharing"""
        data_np = data.cpu().numpy()

        # Initialize centroids
        centroids = np.random.choice(data_np, n_clusters, replace=False)

        for _ in range(10):  # Simple iteration limit
            # Assign points to clusters
            distances = np.abs(data_np[:, np.newaxis] - centroids)
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = []
            for i in range(n_clusters):
                cluster_points = data_np[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids.append(np.mean(cluster_points))
                else:
                    new_centroids.append(centroids[i])

            centroids = np.array(new_centroids)

        return centroids.tolist()


class ModelQuantizer:
    """Implements model quantization for edge deployment"""

    def __init__(self):
        self.quantization_schemes = {
            "dynamic": self._dynamic_quantization,
            "static": self._static_quantization,
            "qat": self._quantization_aware_training,
        }

    def quantize_model(
        self,
        model: nn.Module,
        scheme: str = "dynamic",
        bits: int = 8,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Quantize model using specified scheme

        Args:
            model: PyTorch model to quantize
            scheme: Quantization scheme ('dynamic', 'static', 'qat')
            bits: Number of bits for quantization
            calibration_data: Calibration data for static quantization

        Returns:
            Quantized model
        """
        if scheme not in self.quantization_schemes:
            raise ValueError(f"Unknown quantization scheme: {scheme}")

        logger.info(f"Quantizing model using {scheme} quantization with {bits} bits")
        return self.quantization_schemes[scheme](model, bits, calibration_data)

    def _dynamic_quantization(
        self, model: nn.Module, bits: int, calibration_data: Optional[torch.Tensor]
    ) -> nn.Module:
        """Implement dynamic quantization"""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8 if bits == 8 else torch.qint32,
        )
        return quantized_model

    def _static_quantization(
        self, model: nn.Module, bits: int, calibration_data: Optional[torch.Tensor]
    ) -> nn.Module:
        """Implement static quantization"""
        model.eval()

        # Set quantization config
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        # Prepare model for quantization
        prepared_model = torch.quantization.prepare(model)

        # Calibrate with sample data if provided
        if calibration_data is not None:
            with torch.no_grad():
                prepared_model(calibration_data)

        # Convert to quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model

    def _quantization_aware_training(
        self, model: nn.Module, bits: int, calibration_data: Optional[torch.Tensor]
    ) -> nn.Module:
        """Implement quantization-aware training (simplified)"""
        # This is a simplified version - full QAT requires training loop
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        prepared_model = torch.quantization.prepare_qat(model)

        # In practice, you would train the model here
        # For now, we'll just prepare it
        prepared_model.eval()
        quantized_model = torch.quantization.convert(prepared_model)
        return quantized_model


class ModelPruner:
    """Implements model pruning techniques"""

    def __init__(self):
        self.pruning_methods = {
            "magnitude": self._magnitude_pruning,
            "structured": self._structured_pruning,
            "gradual": self._gradual_pruning,
        }

    def prune_model(
        self,
        model: nn.Module,
        method: str = "magnitude",
        sparsity: float = 0.3,
        structured: bool = False,
    ) -> nn.Module:
        """
        Prune model using specified method

        Args:
            model: PyTorch model to prune
            method: Pruning method to use
            sparsity: Target sparsity level (0.0 to 1.0)
            structured: Whether to use structured pruning

        Returns:
            Pruned model
        """
        if method not in self.pruning_methods:
            raise ValueError(f"Unknown pruning method: {method}")

        logger.info(f"Pruning model using {method} method with sparsity {sparsity}")
        return self.pruning_methods[method](model, sparsity, structured)

    def _magnitude_pruning(
        self, model: nn.Module, sparsity: float, structured: bool
    ) -> nn.Module:
        """Implement magnitude-based pruning"""
        pruned_model = copy.deepcopy(model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if structured:
                    prune.ln_structured(
                        module, name="weight", amount=sparsity, n=2, dim=0
                    )
                else:
                    prune.l1_unstructured(module, name="weight", amount=sparsity)

        return pruned_model

    def _structured_pruning(
        self, model: nn.Module, sparsity: float, structured: bool
    ) -> nn.Module:
        """Implement structured pruning"""
        pruned_model = copy.deepcopy(model)

        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Remove entire channels/neurons
                prune.ln_structured(module, name="weight", amount=sparsity, n=2, dim=0)

        return pruned_model

    def _gradual_pruning(
        self, model: nn.Module, sparsity: float, structured: bool
    ) -> nn.Module:
        """Implement gradual pruning"""
        # This would typically be done during training
        # For now, we'll apply magnitude pruning
        return self._magnitude_pruning(model, sparsity, structured)


class PerformanceValidator:
    """Validates optimized model performance"""

    def __init__(self):
        self.metrics = {}

    def validate_model(
        self,
        original_model: nn.Module,
        optimized_model: nn.Module,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        hardware_constraints: HardwareConstraints,
    ) -> OptimizationResult:
        """
        Validate optimized model against original

        Args:
            original_model: Original unoptimized model
            optimized_model: Optimized model
            test_data: Test dataset
            test_labels: Test labels
            hardware_constraints: Target hardware constraints

        Returns:
            Optimization results
        """
        logger.info("Validating optimized model performance")

        # Measure model sizes
        original_size = self._get_model_size(original_model)
        optimized_size = self._get_model_size(optimized_model)

        # Measure inference times
        original_time = self._measure_inference_time(original_model, test_data)
        optimized_time = self._measure_inference_time(optimized_model, test_data)

        # Measure accuracy
        original_accuracy = self._measure_accuracy(
            original_model, test_data, test_labels
        )
        optimized_accuracy = self._measure_accuracy(
            optimized_model, test_data, test_labels
        )

        # Measure memory usage
        memory_usage = self._measure_memory_usage(optimized_model, test_data)

        # Check if constraints are met
        meets_constraints = self._check_constraints(
            optimized_size, optimized_time, memory_usage, hardware_constraints
        )

        return OptimizationResult(
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            compression_ratio=(
                original_size / optimized_size if optimized_size > 0 else 1.0
            ),
            original_inference_time_ms=original_time,
            optimized_inference_time_ms=optimized_time,
            speedup_factor=(
                original_time / optimized_time if optimized_time > 0 else 1.0
            ),
            accuracy_retention=(
                optimized_accuracy / original_accuracy if original_accuracy > 0 else 1.0
            ),
            memory_usage_mb=memory_usage,
            meets_constraints=meets_constraints,
            optimization_steps=["compression", "quantization", "pruning"],
        )

    def _get_model_size(self, model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    def _measure_inference_time(
        self, model: nn.Module, test_data: torch.Tensor
    ) -> float:
        """Measure average inference time in milliseconds"""
        model.eval()
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(test_data[:1])

            # Measure
            for _ in range(100):
                start_time = time.time()
                _ = model(test_data[:1])
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        return np.mean(times)

    def _measure_accuracy(
        self, model: nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor
    ) -> float:
        """Measure model accuracy"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            outputs = model(test_data)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Classification
                _, predicted = torch.max(outputs.data, 1)
                total += test_labels.size(0)
                correct += (predicted == test_labels).sum().item()
            else:
                # Regression - use MSE as accuracy metric
                mse = torch.mean((outputs - test_labels) ** 2)
                return 1.0 / (1.0 + mse.item())  # Convert MSE to accuracy-like metric

        return correct / total if total > 0 else 0.0

    def _measure_memory_usage(self, model: nn.Module, test_data: torch.Tensor) -> float:
        """Measure memory usage during inference"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        model.eval()
        with torch.no_grad():
            _ = model(test_data)

        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        return final_memory - initial_memory

    def _check_constraints(
        self,
        model_size: float,
        inference_time: float,
        memory_usage: float,
        constraints: HardwareConstraints,
    ) -> bool:
        """Check if optimized model meets hardware constraints"""
        size_ok = model_size <= constraints.max_memory_mb
        time_ok = inference_time <= constraints.max_inference_time_ms
        memory_ok = memory_usage <= constraints.max_memory_mb

        return size_ok and time_ok and memory_ok


class EdgeAIOptimizer:
    """Main edge AI optimization pipeline"""

    def __init__(self):
        self.compressor = ModelCompressor()
        self.quantizer = ModelQuantizer()
        self.pruner = ModelPruner()
        self.validator = PerformanceValidator()

    def optimize_for_edge(
        self,
        model: nn.Module,
        hardware_constraints: HardwareConstraints,
        optimization_config: OptimizationConfig,
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        calibration_data: Optional[torch.Tensor] = None,
    ) -> Tuple[nn.Module, OptimizationResult]:
        """
        Complete optimization pipeline for edge deployment

        Args:
            model: Original PyTorch model
            hardware_constraints: Target hardware constraints
            optimization_config: Optimization configuration
            test_data: Test dataset for validation
            test_labels: Test labels for validation
            calibration_data: Optional calibration data for quantization

        Returns:
            Tuple of (optimized_model, optimization_results)
        """
        logger.info("Starting edge AI optimization pipeline")

        optimized_model = copy.deepcopy(model)
        optimization_steps = []

        # Step 1: Model Compression
        if optimization_config.compression_ratio < 1.0:
            logger.info("Applying model compression")
            optimized_model = self.compressor.compress_model(
                optimized_model,
                method="weight_sharing",
                compression_ratio=optimization_config.compression_ratio,
            )
            optimization_steps.append("compression")

        # Step 2: Quantization
        if optimization_config.enable_quantization:
            logger.info("Applying model quantization")
            optimized_model = self.quantizer.quantize_model(
                optimized_model,
                scheme="dynamic",
                bits=optimization_config.quantization_bits,
                calibration_data=calibration_data,
            )
            optimization_steps.append("quantization")

        # Step 3: Pruning
        if optimization_config.enable_pruning:
            logger.info("Applying model pruning")
            optimized_model = self.pruner.prune_model(
                optimized_model,
                method="magnitude",
                sparsity=optimization_config.pruning_sparsity,
            )
            optimization_steps.append("pruning")

        # Step 4: Validation
        logger.info("Validating optimized model")
        results = self.validator.validate_model(
            model, optimized_model, test_data, test_labels, hardware_constraints
        )
        results.optimization_steps = optimization_steps

        # Check if accuracy threshold is met
        if results.accuracy_retention < optimization_config.validation_threshold:
            logger.warning(
                f"Accuracy retention {results.accuracy_retention:.3f} below threshold {optimization_config.validation_threshold}"
            )

        logger.info(
            f"Optimization complete. Compression ratio: {results.compression_ratio:.2f}x, "
            f"Speedup: {results.speedup_factor:.2f}x, "
            f"Accuracy retention: {results.accuracy_retention:.3f}"
        )

        return optimized_model, results

    def generate_edge_model_variants(
        self,
        model: nn.Module,
        hardware_profiles: List[HardwareConstraints],
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
    ) -> Dict[str, Tuple[nn.Module, OptimizationResult]]:
        """
        Generate multiple optimized model variants for different hardware profiles

        Args:
            model: Original model
            hardware_profiles: List of different hardware constraint profiles
            test_data: Test dataset
            test_labels: Test labels

        Returns:
            Dictionary mapping profile names to (optimized_model, results)
        """
        logger.info(f"Generating {len(hardware_profiles)} edge model variants")

        variants = {}

        for i, constraints in enumerate(hardware_profiles):
            profile_name = f"profile_{i}_{constraints.target_platform}"

            # Adjust optimization config based on constraints
            config = self._create_optimization_config_for_constraints(constraints)

            try:
                optimized_model, results = self.optimize_for_edge(
                    model, constraints, config, test_data, test_labels
                )
                variants[profile_name] = (optimized_model, results)

                logger.info(
                    f"Generated variant {profile_name}: "
                    f"{results.compression_ratio:.2f}x compression, "
                    f"{results.speedup_factor:.2f}x speedup"
                )

            except Exception as e:
                logger.error(f"Failed to generate variant {profile_name}: {e}")

        return variants

    def _create_optimization_config_for_constraints(
        self, constraints: HardwareConstraints
    ) -> OptimizationConfig:
        """Create optimization config tailored to hardware constraints"""

        # Aggressive optimization for very constrained devices
        if constraints.max_memory_mb < 100:
            return OptimizationConfig(
                compression_ratio=0.3,
                quantization_bits=8,
                pruning_sparsity=0.5,
                enable_quantization=True,
                enable_pruning=True,
                validation_threshold=0.85,
            )

        # Moderate optimization for mid-range devices
        elif constraints.max_memory_mb < 500:
            return OptimizationConfig(
                compression_ratio=0.5,
                quantization_bits=8,
                pruning_sparsity=0.3,
                enable_quantization=True,
                enable_pruning=True,
                validation_threshold=0.90,
            )

        # Light optimization for higher-end edge devices
        else:
            return OptimizationConfig(
                compression_ratio=0.7,
                quantization_bits=16,
                pruning_sparsity=0.2,
                enable_quantization=True,
                enable_pruning=False,
                validation_threshold=0.95,
            )

    def save_optimized_model(
        self,
        model: nn.Module,
        results: OptimizationResult,
        save_path: Path,
        metadata: Optional[Dict] = None,
    ):
        """Save optimized model with metadata"""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_path / "optimized_model.pth"
        torch.save(model.state_dict(), model_path)

        # Save optimization results
        results_dict = {
            "original_size_mb": results.original_size_mb,
            "optimized_size_mb": results.optimized_size_mb,
            "compression_ratio": results.compression_ratio,
            "original_inference_time_ms": results.original_inference_time_ms,
            "optimized_inference_time_ms": results.optimized_inference_time_ms,
            "speedup_factor": results.speedup_factor,
            "accuracy_retention": results.accuracy_retention,
            "memory_usage_mb": results.memory_usage_mb,
            "meets_constraints": results.meets_constraints,
            "optimization_steps": results.optimization_steps,
        }

        if metadata:
            results_dict.update(metadata)

        results_path = save_path / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Saved optimized model to {save_path}")


# Example usage and factory functions
def create_mobile_constraints() -> HardwareConstraints:
    """Create typical mobile device constraints"""
    return HardwareConstraints(
        max_memory_mb=200,
        max_inference_time_ms=100,
        cpu_cores=4,
        has_gpu=False,
        target_platform="mobile",
    )


def create_embedded_constraints() -> HardwareConstraints:
    """Create typical embedded device constraints"""
    return HardwareConstraints(
        max_memory_mb=50,
        max_inference_time_ms=500,
        cpu_cores=1,
        has_gpu=False,
        target_platform="embedded",
    )


def create_edge_server_constraints() -> HardwareConstraints:
    """Create typical edge server constraints"""
    return HardwareConstraints(
        max_memory_mb=1000,
        max_inference_time_ms=50,
        cpu_cores=8,
        has_gpu=True,
        gpu_memory_mb=4000,
        target_platform="edge_server",
    )
