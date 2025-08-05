"""
Model Complexity Management System

Implements automatic complexity assessment for geological simulations including:
- Automatic complexity assessment for geological simulations
- Adaptive model selection based on data availability
- Model hierarchy from simplified to full complexity
- Computational cost estimation and optimization

This module provides intelligent model selection and complexity management
to balance accuracy with computational efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog
import time
import psutil
import json
from abc import ABC, abstractmethod

from models.ai_models import GeologicalProcess, PhysicsConstraint
from .pinn_engine import PINNNetwork

logger = structlog.get_logger()


class ComplexityLevel(str, Enum):
    """Model complexity levels"""
    MINIMAL = "minimal"
    SIMPLIFIED = "simplified"
    MODERATE = "moderate"
    ADVANCED = "advanced"
    FULL = "full"


class DataAvailability(str, Enum):
    """Data availability levels"""
    SPARSE = "sparse"          # < 10% of ideal data
    LIMITED = "limited"        # 10-30% of ideal data
    MODERATE = "moderate"      # 30-70% of ideal data
    GOOD = "good"             # 70-90% of ideal data
    EXCELLENT = "excellent"    # > 90% of ideal data


class ComputationalResource(str, Enum):
    """Available computational resources"""
    MINIMAL = "minimal"        # Single CPU, < 4GB RAM
    LIMITED = "limited"        # Multi-core CPU, 4-16GB RAM
    MODERATE = "moderate"      # GPU available, 16-64GB RAM
    HIGH = "high"             # Multiple GPUs, 64-256GB RAM
    EXTREME = "extreme"       # TPU/cluster, > 256GB RAM


@dataclass
class ComplexityMetrics:
    """Metrics for assessing model complexity"""
    parameter_count: int
    computational_flops: float
    memory_requirement: float  # in GB
    training_time_estimate: float  # in hours
    inference_time_estimate: float  # in seconds
    accuracy_score: float
    physics_constraint_count: int
    coupling_terms: int    

    def complexity_score(self) -> float:
        """Compute overall complexity score"""
        # Normalize and weight different factors
        param_score = np.log10(max(1, self.parameter_count)) / 10
        flop_score = np.log10(max(1, self.computational_flops)) / 15
        memory_score = self.memory_requirement / 100
        time_score = np.log10(max(1, self.training_time_estimate)) / 5
        
        return param_score + flop_score + memory_score + time_score


@dataclass
class DataAssessment:
    """Assessment of available data for model selection"""
    total_data_points: int
    data_quality_score: float  # 0-1
    temporal_coverage: float   # fraction of desired time range
    spatial_coverage: float    # fraction of desired spatial domain
    data_types_available: List[str]
    missing_data_fraction: float
    noise_level: float
    
    def availability_level(self) -> DataAvailability:
        """Determine data availability level"""
        coverage_score = (self.temporal_coverage + self.spatial_coverage) / 2
        quality_score = self.data_quality_score * (1 - self.missing_data_fraction)
        overall_score = coverage_score * quality_score
        
        if overall_score < 0.1:
            return DataAvailability.SPARSE
        elif overall_score < 0.3:
            return DataAvailability.LIMITED
        elif overall_score < 0.7:
            return DataAvailability.MODERATE
        elif overall_score < 0.9:
            return DataAvailability.GOOD
        else:
            return DataAvailability.EXCELLENT


@dataclass
class ResourceAssessment:
    """Assessment of available computational resources"""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    storage_gb: float
    network_bandwidth_mbps: float
    
    def resource_level(self) -> ComputationalResource:
        """Determine computational resource level"""
        if self.gpu_count == 0 and self.memory_gb < 4:
            return ComputationalResource.MINIMAL
        elif self.gpu_count == 0 and self.memory_gb < 16:
            return ComputationalResource.LIMITED
        elif self.gpu_count <= 1 and self.memory_gb < 64:
            return ComputationalResource.MODERATE
        elif self.gpu_count <= 4 and self.memory_gb < 256:
            return ComputationalResource.HIGH
        else:
            return ComputationalResource.EXTREME


@dataclass
class ModelConfiguration:
    """Configuration for a specific model complexity level"""
    complexity_level: ComplexityLevel
    network_layers: List[int]
    physics_constraints: List[str]
    coupling_terms: List[str]
    time_stepping_method: str
    spatial_resolution: float
    temporal_resolution: float
    batch_size: int
    training_epochs: int
    convergence_tolerance: float    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'complexity_level': self.complexity_level.value,
            'network_layers': self.network_layers,
            'physics_constraints': self.physics_constraints,
            'coupling_terms': self.coupling_terms,
            'time_stepping_method': self.time_stepping_method,
            'spatial_resolution': self.spatial_resolution,
            'temporal_resolution': self.temporal_resolution,
            'batch_size': self.batch_size,
            'training_epochs': self.training_epochs,
            'convergence_tolerance': self.convergence_tolerance
        }


class ModelComplexityAssessor:
    """Assesses the complexity of geological models"""
    
    def __init__(self):
        self.complexity_cache = {}
        
    def assess_model_complexity(self, model: nn.Module, 
                              input_shape: Tuple[int, ...],
                              physics_constraints: List[PhysicsConstraint] = None) -> ComplexityMetrics:
        """Assess the complexity of a given model"""
        
        # Count parameters
        parameter_count = sum(p.numel() for p in model.parameters())
        
        # Estimate computational FLOPs
        computational_flops = self._estimate_flops(model, input_shape)
        
        # Estimate memory requirement
        memory_requirement = self._estimate_memory(model, input_shape)
        
        # Estimate training time
        training_time_estimate = self._estimate_training_time(
            parameter_count, computational_flops
        )
        
        # Estimate inference time
        inference_time_estimate = self._estimate_inference_time(
            computational_flops
        )
        
        # Assess accuracy (placeholder - would need validation data)
        accuracy_score = 0.85  # Default estimate
        
        # Count physics constraints
        physics_constraint_count = len(physics_constraints) if physics_constraints else 0
        
        # Count coupling terms (simplified)
        coupling_terms = self._count_coupling_terms(model)
        
        return ComplexityMetrics(
            parameter_count=parameter_count,
            computational_flops=computational_flops,
            memory_requirement=memory_requirement,
            training_time_estimate=training_time_estimate,
            inference_time_estimate=inference_time_estimate,
            accuracy_score=accuracy_score,
            physics_constraint_count=physics_constraint_count,
            coupling_terms=coupling_terms
        )    

    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Estimate FLOPs for model inference"""
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                output_elements = np.prod(input_shape) // (module.stride[0] * module.stride[1])
                total_flops += kernel_flops * module.in_channels * module.out_channels * output_elements
        
        return float(total_flops)
    
    def _estimate_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Estimate memory requirement in GB"""
        # Parameter memory
        param_memory = sum(p.numel() * 4 for p in model.parameters()) / (1024**3)  # 4 bytes per float32
        
        # Activation memory (rough estimate)
        activation_memory = np.prod(input_shape) * 4 / (1024**3)
        
        # Gradient memory (same as parameters)
        gradient_memory = param_memory
        
        # Add some overhead
        total_memory = (param_memory + activation_memory + gradient_memory) * 1.5
        
        return total_memory
    
    def _estimate_training_time(self, parameter_count: int, flops: float) -> float:
        """Estimate training time in hours"""
        # Very rough estimate based on parameter count and FLOPs
        base_time = parameter_count / 1e6  # 1 hour per million parameters
        flop_time = flops / 1e12  # Additional time based on FLOPs
        
        return base_time + flop_time
    
    def _estimate_inference_time(self, flops: float) -> float:
        """Estimate inference time in seconds"""
        # Assume 1 TFLOP/s processing speed
        return flops / 1e12
    
    def _count_coupling_terms(self, model: nn.Module) -> int:
        """Count coupling terms in the model (simplified)"""
        # This is a placeholder - would need model-specific logic
        return 0


class DataAvailabilityAssessor:
    """Assesses data availability for model selection"""
    
    def assess_data_availability(self, 
                               data: Dict[str, torch.Tensor],
                               desired_coverage: Dict[str, Any] = None) -> DataAssessment:
        """Assess the availability and quality of data"""
        
        total_data_points = sum(tensor.shape[0] for tensor in data.values())
        
        # Assess data quality (simplified)
        data_quality_score = self._assess_data_quality(data)
        
        # Assess temporal coverage
        temporal_coverage = self._assess_temporal_coverage(data, desired_coverage)
        
        # Assess spatial coverage
        spatial_coverage = self._assess_spatial_coverage(data, desired_coverage)
        
        # Get available data types
        data_types_available = list(data.keys())
        
        # Assess missing data
        missing_data_fraction = self._assess_missing_data(data)
        
        # Assess noise level
        noise_level = self._assess_noise_level(data)
        
        return DataAssessment(
            total_data_points=total_data_points,
            data_quality_score=data_quality_score,
            temporal_coverage=temporal_coverage,
            spatial_coverage=spatial_coverage,
            data_types_available=data_types_available,
            missing_data_fraction=missing_data_fraction,
            noise_level=noise_level
        ) 
   
    def _assess_data_quality(self, data: Dict[str, torch.Tensor]) -> float:
        """Assess overall data quality"""
        quality_scores = []
        
        for key, tensor in data.items():
            # Check for NaN/inf values
            nan_fraction = torch.isnan(tensor).float().mean().item()
            inf_fraction = torch.isinf(tensor).float().mean().item()
            
            # Check for reasonable value ranges
            value_range_score = 1.0  # Placeholder
            
            # Combine scores
            quality_score = (1 - nan_fraction - inf_fraction) * value_range_score
            quality_scores.append(quality_score)
        
        return np.mean(quality_scores) if quality_scores else 0.0
    
    def _assess_temporal_coverage(self, data: Dict[str, torch.Tensor], 
                                desired_coverage: Dict[str, Any] = None) -> float:
        """Assess temporal coverage of data"""
        # Simplified - assume first dimension is time
        if not data:
            return 0.0
        
        max_time_points = max(tensor.shape[0] for tensor in data.values())
        desired_time_points = 1000  # Default desired coverage
        
        if desired_coverage and 'temporal_points' in desired_coverage:
            desired_time_points = desired_coverage['temporal_points']
        
        return min(1.0, max_time_points / desired_time_points)
    
    def _assess_spatial_coverage(self, data: Dict[str, torch.Tensor],
                               desired_coverage: Dict[str, Any] = None) -> float:
        """Assess spatial coverage of data"""
        # Simplified assessment
        if not data:
            return 0.0
        
        # Assume spatial dimensions are after time dimension
        spatial_points = 1
        for tensor in data.values():
            if len(tensor.shape) > 1:
                spatial_points = max(spatial_points, np.prod(tensor.shape[1:]))
        
        desired_spatial_points = 10000  # Default
        if desired_coverage and 'spatial_points' in desired_coverage:
            desired_spatial_points = desired_coverage['spatial_points']
        
        return min(1.0, spatial_points / desired_spatial_points)
    
    def _assess_missing_data(self, data: Dict[str, torch.Tensor]) -> float:
        """Assess fraction of missing data"""
        total_elements = 0
        missing_elements = 0
        
        for tensor in data.values():
            total_elements += tensor.numel()
            missing_elements += torch.isnan(tensor).sum().item()
        
        return missing_elements / total_elements if total_elements > 0 else 1.0
    
    def _assess_noise_level(self, data: Dict[str, torch.Tensor]) -> float:
        """Assess noise level in data"""
        noise_levels = []
        
        for tensor in data.values():
            if tensor.numel() > 1:
                # Simple noise estimate using variance
                variance = torch.var(tensor).item()
                mean_abs = torch.abs(tensor).mean().item()
                
                if mean_abs > 0:
                    noise_level = np.sqrt(variance) / mean_abs
                    noise_levels.append(min(1.0, noise_level))
        
        return np.mean(noise_levels) if noise_levels else 0.0


class ResourceAssessor:
    """Assesses available computational resources"""
    
    def assess_resources(self) -> ResourceAssessment:
        """Assess current computational resources"""
        
        # CPU information
        cpu_cores = psutil.cpu_count(logical=True)
        
        # Memory information
        memory_info = psutil.virtual_memory()
        memory_gb = memory_info.total / (1024**3)
        
        # GPU information (simplified - would need proper GPU detection)
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_memory_gb = 0.0
        
        if gpu_count > 0:
            for i in range(gpu_count):
                gpu_memory_gb += torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        # Storage information
        disk_info = psutil.disk_usage('/')
        storage_gb = disk_info.total / (1024**3)
        
        # Network bandwidth (placeholder)
        network_bandwidth_mbps = 1000.0  # Default assumption
        
        return ResourceAssessment(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            storage_gb=storage_gb,
            network_bandwidth_mbps=network_bandwidth_mbps
        )


class ModelHierarchy:
    """Manages hierarchy of models from simplified to full complexity"""
    
    def __init__(self):
        self.model_configs = self._initialize_model_hierarchy()
        
    def _initialize_model_hierarchy(self) -> Dict[ComplexityLevel, ModelConfiguration]:
        """Initialize the model hierarchy with different complexity levels"""
        
        configs = {}
        
        # Minimal complexity
        configs[ComplexityLevel.MINIMAL] = ModelConfiguration(
            complexity_level=ComplexityLevel.MINIMAL,
            network_layers=[32, 16],
            physics_constraints=['mass_conservation'],
            coupling_terms=[],
            time_stepping_method='explicit_euler',
            spatial_resolution=1000.0,  # meters
            temporal_resolution=3600.0,  # seconds
            batch_size=32,
            training_epochs=50,
            convergence_tolerance=1e-3
        )
        
        # Simplified complexity
        configs[ComplexityLevel.SIMPLIFIED] = ModelConfiguration(
            complexity_level=ComplexityLevel.SIMPLIFIED,
            network_layers=[64, 32, 16],
            physics_constraints=['mass_conservation', 'energy_conservation'],
            coupling_terms=['thermal_mechanical'],
            time_stepping_method='implicit_euler',
            spatial_resolution=500.0,
            temporal_resolution=1800.0,
            batch_size=64,
            training_epochs=100,
            convergence_tolerance=1e-4
        )
        
        # Moderate complexity
        configs[ComplexityLevel.MODERATE] = ModelConfiguration(
            complexity_level=ComplexityLevel.MODERATE,
            network_layers=[128, 64, 32, 16],
            physics_constraints=['mass_conservation', 'energy_conservation', 'momentum_conservation'],
            coupling_terms=['thermal_mechanical', 'chemical_thermal'],
            time_stepping_method='runge_kutta_4',
            spatial_resolution=250.0,
            temporal_resolution=900.0,
            batch_size=128,
            training_epochs=200,
            convergence_tolerance=1e-5
        )
        
        # Advanced complexity
        configs[ComplexityLevel.ADVANCED] = ModelConfiguration(
            complexity_level=ComplexityLevel.ADVANCED,
            network_layers=[256, 128, 64, 32, 16],
            physics_constraints=['mass_conservation', 'energy_conservation', 'momentum_conservation', 'navier_stokes'],
            coupling_terms=['thermal_mechanical', 'chemical_thermal', 'electromagnetic_thermal'],
            time_stepping_method='adaptive_rk45',
            spatial_resolution=100.0,
            temporal_resolution=300.0,
            batch_size=256,
            training_epochs=500,
            convergence_tolerance=1e-6
        )
        
        # Full complexity
        configs[ComplexityLevel.FULL] = ModelConfiguration(
            complexity_level=ComplexityLevel.FULL,
            network_layers=[512, 256, 128, 64, 32, 16],
            physics_constraints=['mass_conservation', 'energy_conservation', 'momentum_conservation', 
                               'navier_stokes', 'maxwell_equations', 'reaction_diffusion'],
            coupling_terms=['thermal_mechanical', 'chemical_thermal', 'electromagnetic_thermal',
                          'fluid_mechanical', 'chemical_mechanical'],
            time_stepping_method='adaptive_rk45',
            spatial_resolution=50.0,
            temporal_resolution=60.0,
            batch_size=512,
            training_epochs=1000,
            convergence_tolerance=1e-8
        )
        
        return configs
    
    def get_config(self, complexity_level: ComplexityLevel) -> ModelConfiguration:
        """Get configuration for specified complexity level"""
        return self.model_configs[complexity_level]
    
    def get_all_configs(self) -> Dict[ComplexityLevel, ModelConfiguration]:
        """Get all model configurations"""
        return self.model_configs.copy()


class ComputationalCostEstimator:
    """Estimates and optimizes computational costs"""
    
    def __init__(self):
        self.cost_history = []
        
    def estimate_cost(self, 
                     config: ModelConfiguration,
                     data_size: int,
                     resource_assessment: ResourceAssessment) -> Dict[str, float]:
        """Estimate computational cost for a given configuration"""
        
        # Estimate training cost
        training_cost = self._estimate_training_cost(config, data_size, resource_assessment)
        
        # Estimate inference cost
        inference_cost = self._estimate_inference_cost(config, resource_assessment)
        
        # Estimate memory cost
        memory_cost = self._estimate_memory_cost(config, data_size)
        
        # Estimate storage cost
        storage_cost = self._estimate_storage_cost(config, data_size)
        
        # Total cost (normalized)
        total_cost = training_cost + inference_cost + memory_cost + storage_cost
        
        cost_breakdown = {
            'training_cost': training_cost,
            'inference_cost': inference_cost,
            'memory_cost': memory_cost,
            'storage_cost': storage_cost,
            'total_cost': total_cost
        }
        
        self.cost_history.append(cost_breakdown)
        
        return cost_breakdown
    
    def _estimate_training_cost(self, 
                              config: ModelConfiguration,
                              data_size: int,
                              resources: ResourceAssessment) -> float:
        """Estimate training computational cost"""
        
        # Base cost from network complexity
        network_complexity = sum(config.network_layers)
        base_cost = network_complexity * config.training_epochs * data_size / 1e6
        
        # Adjust for physics constraints
        physics_multiplier = 1 + 0.2 * len(config.physics_constraints)
        
        # Adjust for coupling terms
        coupling_multiplier = 1 + 0.3 * len(config.coupling_terms)
        
        # Adjust for available resources
        if resources.gpu_count > 0:
            resource_multiplier = 1.0 / (1 + resources.gpu_count * 0.5)
        else:
            resource_multiplier = 2.0  # CPU-only penalty
        
        total_cost = base_cost * physics_multiplier * coupling_multiplier * resource_multiplier
        
        return total_cost
    
    def _estimate_inference_cost(self, 
                               config: ModelConfiguration,
                               resources: ResourceAssessment) -> float:
        """Estimate inference computational cost"""
        
        network_complexity = sum(config.network_layers)
        base_cost = network_complexity / 1e3
        
        # Adjust for physics constraints (inference overhead)
        physics_multiplier = 1 + 0.1 * len(config.physics_constraints)
        
        # Adjust for resources
        if resources.gpu_count > 0:
            resource_multiplier = 0.5
        else:
            resource_multiplier = 1.0
        
        return base_cost * physics_multiplier * resource_multiplier
    
    def _estimate_memory_cost(self, 
                            config: ModelConfiguration,
                            data_size: int) -> float:
        """Estimate memory cost"""
        
        # Model parameters memory
        param_memory = sum(config.network_layers) * 4 / (1024**3)  # GB
        
        # Data memory
        data_memory = data_size * 4 / (1024**3)  # GB
        
        # Batch memory
        batch_memory = config.batch_size * sum(config.network_layers) * 4 / (1024**3)
        
        total_memory = param_memory + data_memory + batch_memory
        
        # Normalize to cost score
        return total_memory / 10.0  # Assume 10GB baseline
    
    def _estimate_storage_cost(self, 
                             config: ModelConfiguration,
                             data_size: int) -> float:
        """Estimate storage cost"""
        
        # Model storage
        model_storage = sum(config.network_layers) * 4 / (1024**3)  # GB
        
        # Data storage
        data_storage = data_size * 4 / (1024**3)  # GB
        
        # Checkpoint storage
        checkpoint_storage = model_storage * config.training_epochs / 10  # Save every 10 epochs
        
        total_storage = model_storage + data_storage + checkpoint_storage
        
        # Normalize to cost score
        return total_storage / 100.0  # Assume 100GB baseline
    
    def optimize_configuration(self, 
                             configs: List[ModelConfiguration],
                             data_assessment: DataAssessment,
                             resource_assessment: ResourceAssessment,
                             target_accuracy: float = 0.8) -> ModelConfiguration:
        """Optimize configuration based on cost-accuracy tradeoff"""
        
        best_config = None
        best_score = float('inf')
        
        for config in configs:
            # Estimate cost
            cost = self.estimate_cost(config, data_assessment.total_data_points, resource_assessment)
            
            # Estimate accuracy (simplified)
            estimated_accuracy = self._estimate_accuracy(config, data_assessment)
            
            # Compute cost-accuracy score
            if estimated_accuracy >= target_accuracy:
                score = cost['total_cost']
            else:
                # Penalty for not meeting accuracy target
                accuracy_penalty = (target_accuracy - estimated_accuracy) * 10
                score = cost['total_cost'] + accuracy_penalty
            
            if score < best_score:
                best_score = score
                best_config = config
        
        return best_config
    
    def _estimate_accuracy(self, config: ModelConfiguration, data_assessment: DataAssessment) -> float:
        """Estimate model accuracy based on configuration and data"""
        
        # Base accuracy from network complexity
        network_score = min(1.0, sum(config.network_layers) / 1000)
        
        # Bonus from physics constraints
        physics_score = min(0.2, len(config.physics_constraints) * 0.05)
        
        # Bonus from coupling terms
        coupling_score = min(0.1, len(config.coupling_terms) * 0.02)
        
        # Data availability impact
        data_score = data_assessment.data_quality_score * 0.3
        
        # Training epochs impact
        training_score = min(0.2, config.training_epochs / 1000 * 0.2)
        
        total_accuracy = network_score + physics_score + coupling_score + data_score + training_score
        
        return min(1.0, total_accuracy)


class ModelComplexityManager:
    """
    Main class for managing model complexity in geological simulations
    
    Provides automatic complexity assessment, adaptive model selection,
    and computational cost optimization.
    """
    
    def __init__(self):
        self.complexity_assessor = ModelComplexityAssessor()
        self.data_assessor = DataAvailabilityAssessor()
        self.resource_assessor = ResourceAssessor()
        self.model_hierarchy = ModelHierarchy()
        self.cost_estimator = ComputationalCostEstimator()
        
        # Cache for assessments
        self.assessment_cache = {}
        
        logger.info("Model Complexity Manager initialized")
    
    def assess_simulation_complexity(self, 
                                   geological_process: GeologicalProcess,
                                   data: Dict[str, torch.Tensor],
                                   target_accuracy: float = 0.8) -> Dict[str, Any]:
        """
        Comprehensive complexity assessment for a geological simulation
        
        Args:
            geological_process: The geological process to simulate
            data: Available data for the simulation
            target_accuracy: Target accuracy requirement
            
        Returns:
            Dictionary containing complexity assessment results
        """
        
        logger.info(f"Assessing complexity for {geological_process}")
        
        # Assess data availability
        data_assessment = self.data_assessor.assess_data_availability(data)
        
        # Assess computational resources
        resource_assessment = self.resource_assessor.assess_resources()
        
        # Get all model configurations
        all_configs = self.model_hierarchy.get_all_configs()
        
        # Assess complexity for each configuration
        complexity_assessments = {}
        for level, config in all_configs.items():
            # Create a dummy model for assessment
            dummy_model = self._create_dummy_model(config)
            input_shape = (data_assessment.total_data_points, 10)  # Simplified
            
            complexity_metrics = self.complexity_assessor.assess_model_complexity(
                dummy_model, input_shape
            )
            
            complexity_assessments[level] = complexity_metrics
        
        # Select optimal configuration
        optimal_config = self.select_optimal_model(
            data_assessment, resource_assessment, target_accuracy
        )
        
        # Estimate costs for optimal configuration
        cost_breakdown = self.cost_estimator.estimate_cost(
            optimal_config, data_assessment.total_data_points, resource_assessment
        )
        
        assessment_result = {
            'data_assessment': data_assessment,
            'resource_assessment': resource_assessment,
            'complexity_assessments': complexity_assessments,
            'optimal_configuration': optimal_config,
            'cost_breakdown': cost_breakdown,
            'recommendations': self._generate_recommendations(
                data_assessment, resource_assessment, optimal_config
            )
        }
        
        # Cache the assessment
        cache_key = f"{geological_process}_{hash(str(data.keys()))}"
        self.assessment_cache[cache_key] = assessment_result
        
        return assessment_result
    
    def select_optimal_model(self, 
                           data_assessment: DataAssessment,
                           resource_assessment: ResourceAssessment,
                           target_accuracy: float = 0.8) -> ModelConfiguration:
        """
        Select optimal model configuration based on data and resources
        
        Args:
            data_assessment: Assessment of available data
            resource_assessment: Assessment of computational resources
            target_accuracy: Target accuracy requirement
            
        Returns:
            Optimal model configuration
        """
        
        # Get data and resource levels
        data_level = data_assessment.availability_level()
        resource_level = resource_assessment.resource_level()
        
        # Decision matrix for model selection
        selection_matrix = {
            (DataAvailability.SPARSE, ComputationalResource.MINIMAL): ComplexityLevel.MINIMAL,
            (DataAvailability.SPARSE, ComputationalResource.LIMITED): ComplexityLevel.MINIMAL,
            (DataAvailability.SPARSE, ComputationalResource.MODERATE): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.SPARSE, ComputationalResource.HIGH): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.SPARSE, ComputationalResource.EXTREME): ComplexityLevel.MODERATE,
            
            (DataAvailability.LIMITED, ComputationalResource.MINIMAL): ComplexityLevel.MINIMAL,
            (DataAvailability.LIMITED, ComputationalResource.LIMITED): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.LIMITED, ComputationalResource.MODERATE): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.LIMITED, ComputationalResource.HIGH): ComplexityLevel.MODERATE,
            (DataAvailability.LIMITED, ComputationalResource.EXTREME): ComplexityLevel.MODERATE,
            
            (DataAvailability.MODERATE, ComputationalResource.MINIMAL): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.MODERATE, ComputationalResource.LIMITED): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.MODERATE, ComputationalResource.MODERATE): ComplexityLevel.MODERATE,
            (DataAvailability.MODERATE, ComputationalResource.HIGH): ComplexityLevel.ADVANCED,
            (DataAvailability.MODERATE, ComputationalResource.EXTREME): ComplexityLevel.ADVANCED,
            
            (DataAvailability.GOOD, ComputationalResource.MINIMAL): ComplexityLevel.SIMPLIFIED,
            (DataAvailability.GOOD, ComputationalResource.LIMITED): ComplexityLevel.MODERATE,
            (DataAvailability.GOOD, ComputationalResource.MODERATE): ComplexityLevel.ADVANCED,
            (DataAvailability.GOOD, ComputationalResource.HIGH): ComplexityLevel.ADVANCED,
            (DataAvailability.GOOD, ComputationalResource.EXTREME): ComplexityLevel.FULL,
            
            (DataAvailability.EXCELLENT, ComputationalResource.MINIMAL): ComplexityLevel.MODERATE,
            (DataAvailability.EXCELLENT, ComputationalResource.LIMITED): ComplexityLevel.MODERATE,
            (DataAvailability.EXCELLENT, ComputationalResource.MODERATE): ComplexityLevel.ADVANCED,
            (DataAvailability.EXCELLENT, ComputationalResource.HIGH): ComplexityLevel.FULL,
            (DataAvailability.EXCELLENT, ComputationalResource.EXTREME): ComplexityLevel.FULL,
        }
        
        # Get recommended complexity level
        recommended_level = selection_matrix.get(
            (data_level, resource_level), 
            ComplexityLevel.MODERATE
        )
        
        # Get the configuration
        optimal_config = self.model_hierarchy.get_config(recommended_level)
        
        # Fine-tune based on cost optimization
        all_configs = list(self.model_hierarchy.get_all_configs().values())
        optimized_config = self.cost_estimator.optimize_configuration(
            all_configs, data_assessment, resource_assessment, target_accuracy
        )
        
        logger.info(f"Selected model complexity: {optimized_config.complexity_level}")
        
        return optimized_config
    
    def _create_dummy_model(self, config: ModelConfiguration) -> nn.Module:
        """Create a dummy model for complexity assessment"""
        layers = []
        
        for i, layer_size in enumerate(config.network_layers):
            if i == 0:
                layers.append(nn.Linear(10, layer_size))  # Assume 10 input features
            else:
                layers.append(nn.Linear(config.network_layers[i-1], layer_size))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(config.network_layers[-1], 1))
        
        return nn.Sequential(*layers)
    
    def _generate_recommendations(self, 
                                data_assessment: DataAssessment,
                                resource_assessment: ResourceAssessment,
                                optimal_config: ModelConfiguration) -> List[str]:
        """Generate recommendations for improving simulation setup"""
        
        recommendations = []
        
        # Data-related recommendations
        if data_assessment.availability_level() == DataAvailability.SPARSE:
            recommendations.append("Consider collecting more data or using transfer learning")
        
        if data_assessment.missing_data_fraction > 0.2:
            recommendations.append("High missing data fraction - consider data imputation methods")
        
        if data_assessment.noise_level > 0.3:
            recommendations.append("High noise level detected - consider data preprocessing")
        
        # Resource-related recommendations
        if resource_assessment.resource_level() == ComputationalResource.MINIMAL:
            recommendations.append("Limited computational resources - consider cloud computing")
        
        if resource_assessment.gpu_count == 0:
            recommendations.append("No GPU detected - consider GPU acceleration for better performance")
        
        # Configuration-related recommendations
        if optimal_config.complexity_level == ComplexityLevel.MINIMAL:
            recommendations.append("Using minimal complexity - results may have limited accuracy")
        
        if optimal_config.complexity_level == ComplexityLevel.FULL:
            recommendations.append("Using full complexity - ensure adequate computational resources")
        
        return recommendations 
   
    def get_complexity_hierarchy(self) -> Dict[ComplexityLevel, Dict[str, Any]]:
        """Get the complete model complexity hierarchy"""
        
        hierarchy = {}
        
        for level, config in self.model_hierarchy.get_all_configs().items():
            # Create dummy model for metrics
            dummy_model = self._create_dummy_model(config)
            input_shape = (1000, 10)  # Standard shape for comparison
            
            complexity_metrics = self.complexity_assessor.assess_model_complexity(
                dummy_model, input_shape
            )
            
            hierarchy[level] = {
                'configuration': config.to_dict(),
                'metrics': {
                    'parameter_count': complexity_metrics.parameter_count,
                    'computational_flops': complexity_metrics.computational_flops,
                    'memory_requirement': complexity_metrics.memory_requirement,
                    'training_time_estimate': complexity_metrics.training_time_estimate,
                    'inference_time_estimate': complexity_metrics.inference_time_estimate,
                    'complexity_score': complexity_metrics.complexity_score()
                }
            }
        
        return hierarchy
    
    def save_assessment(self, assessment: Dict[str, Any], filepath: str):
        """Save complexity assessment to file"""
        
        # Convert non-serializable objects to dictionaries
        serializable_assessment = self._make_serializable(assessment)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_assessment, f, indent=2)
        
        logger.info(f"Assessment saved to {filepath}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (DataAssessment, ResourceAssessment, ModelConfiguration, ComplexityMetrics)):
            return obj.__dict__
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj