"""
Automated Edge Model Generation Service

This module provides automated generation of edge-optimized models from full Helios models.
It integrates with the existing Helios architecture to automatically create optimized variants
for different edge deployment scenarios.

Requirements addressed: 10.1, 10.5
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from .edge_ai_optimizer import (
    EdgeAIOptimizer, HardwareConstraints, OptimizationConfig, OptimizationResult,
    create_mobile_constraints, create_embedded_constraints, create_edge_server_constraints
)
from .helios_engine import HeliosCore
from .pinn_engine import PINNEngine
from .seismic_cnn_processor import SeismicCNNProcessor

logger = logging.getLogger(__name__)


@dataclass
class EdgeDeploymentProfile:
    """Profile for edge deployment scenarios"""
    name: str
    description: str
    hardware_constraints: HardwareConstraints
    optimization_config: OptimizationConfig
    target_use_cases: List[str]
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class EdgeModelMetadata:
    """Metadata for edge-optimized models"""
    model_id: str
    source_model_id: str
    deployment_profile: str
    creation_timestamp: datetime
    optimization_results: OptimizationResult
    model_hash: str
    validation_status: str
    deployment_ready: bool


class EdgeModelRegistry:
    """Registry for managing edge-optimized models"""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = storage_path / "edge_model_registry.json"
        self.models: Dict[str, EdgeModelMetadata] = {}
        self._load_registry()
    
    def register_model(self, metadata: EdgeModelMetadata):
        """Register a new edge model"""
        self.models[metadata.model_id] = metadata
        self._save_registry()
        logger.info(f"Registered edge model: {metadata.model_id}")
    
    def get_model(self, model_id: str) -> Optional[EdgeModelMetadata]:
        """Get model metadata by ID"""
        return self.models.get(model_id)
    
    def list_models_by_profile(self, profile_name: str) -> List[EdgeModelMetadata]:
        """List all models for a specific deployment profile"""
        return [model for model in self.models.values() 
                if model.deployment_profile == profile_name]
    
    def get_best_model_for_constraints(self, constraints: HardwareConstraints) -> Optional[EdgeModelMetadata]:
        """Find the best model that meets given constraints"""
        compatible_models = []
        
        for model in self.models.values():
            if (model.deployment_ready and 
                model.optimization_results.meets_constraints and
                model.optimization_results.optimized_size_mb <= constraints.max_memory_mb and
                model.optimization_results.optimized_inference_time_ms <= constraints.max_inference_time_ms):
                compatible_models.append(model)
        
        if not compatible_models:
            return None
        
        # Return model with best accuracy retention
        return max(compatible_models, 
                  key=lambda m: m.optimization_results.accuracy_retention)
    
    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)
                
                for model_id, model_data in data.items():
                    # Reconstruct objects from JSON
                    opt_results = OptimizationResult(**model_data['optimization_results'])
                    model_data['optimization_results'] = opt_results
                    model_data['creation_timestamp'] = datetime.fromisoformat(
                        model_data['creation_timestamp']
                    )
                    
                    self.models[model_id] = EdgeModelMetadata(**model_data)
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {}
            for model_id, metadata in self.models.items():
                model_dict = asdict(metadata)
                model_dict['creation_timestamp'] = metadata.creation_timestamp.isoformat()
                model_dict['optimization_results'] = asdict(metadata.optimization_results)
                data[model_id] = model_dict
            
            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")


class AutomatedEdgeGenerator:
    """Automated edge model generation service"""
    
    def __init__(self, helios_core: HeliosCore, storage_path: Path):
        self.helios_core = helios_core
        self.optimizer = EdgeAIOptimizer()
        self.registry = EdgeModelRegistry(storage_path)
        self.deployment_profiles = self._create_default_profiles()
        
        # Test data for validation (would be loaded from actual datasets)
        self.test_data = self._generate_test_data()
        self.test_labels = self._generate_test_labels()
    
    def _create_default_profiles(self) -> Dict[str, EdgeDeploymentProfile]:
        """Create default deployment profiles"""
        profiles = {}
        
        # Mobile/Tablet Profile
        profiles['mobile'] = EdgeDeploymentProfile(
            name='mobile',
            description='Smartphones and tablets with limited resources',
            hardware_constraints=create_mobile_constraints(),
            optimization_config=OptimizationConfig(
                compression_ratio=0.3,
                quantization_bits=8,
                pruning_sparsity=0.4,
                enable_quantization=True,
                enable_pruning=True,
                validation_threshold=0.85
            ),
            target_use_cases=['field_monitoring', 'quick_analysis', 'emergency_alerts'],
            priority=1
        )
        
        # Embedded Systems Profile
        profiles['embedded'] = EdgeDeploymentProfile(
            name='embedded',
            description='Embedded systems and IoT devices',
            hardware_constraints=create_embedded_constraints(),
            optimization_config=OptimizationConfig(
                compression_ratio=0.2,
                quantization_bits=8,
                pruning_sparsity=0.6,
                enable_quantization=True,
                enable_pruning=True,
                validation_threshold=0.80
            ),
            target_use_cases=['sensor_nodes', 'remote_monitoring', 'autonomous_systems'],
            priority=2
        )
        
        # Edge Server Profile
        profiles['edge_server'] = EdgeDeploymentProfile(
            name='edge_server',
            description='Edge servers with moderate computational resources',
            hardware_constraints=create_edge_server_constraints(),
            optimization_config=OptimizationConfig(
                compression_ratio=0.6,
                quantization_bits=16,
                pruning_sparsity=0.2,
                enable_quantization=True,
                enable_pruning=False,
                validation_threshold=0.92
            ),
            target_use_cases=['regional_processing', 'real_time_analysis', 'local_inference'],
            priority=1
        )
        
        # Field Station Profile
        profiles['field_station'] = EdgeDeploymentProfile(
            name='field_station',
            description='Field research stations with ruggedized hardware',
            hardware_constraints=HardwareConstraints(
                max_memory_mb=300,
                max_inference_time_ms=200,
                cpu_cores=4,
                has_gpu=False,
                target_platform="field_station"
            ),
            optimization_config=OptimizationConfig(
                compression_ratio=0.4,
                quantization_bits=8,
                pruning_sparsity=0.3,
                enable_quantization=True,
                enable_pruning=True,
                validation_threshold=0.88
            ),
            target_use_cases=['volcano_monitoring', 'seismic_detection', 'offline_analysis'],
            priority=1
        )
        
        return profiles
    
    async def generate_all_edge_models(self, source_model: nn.Module, 
                                     source_model_id: str) -> Dict[str, EdgeModelMetadata]:
        """
        Generate edge models for all deployment profiles
        
        Args:
            source_model: Original full-scale model
            source_model_id: Identifier for the source model
            
        Returns:
            Dictionary mapping profile names to generated model metadata
        """
        logger.info(f"Generating edge models for all profiles from source: {source_model_id}")
        
        generated_models = {}
        
        for profile_name, profile in self.deployment_profiles.items():
            try:
                logger.info(f"Generating model for profile: {profile_name}")
                
                # Generate optimized model
                optimized_model, results = self.optimizer.optimize_for_edge(
                    source_model,
                    profile.hardware_constraints,
                    profile.optimization_config,
                    self.test_data,
                    self.test_labels
                )
                
                # Create model metadata
                model_id = self._generate_model_id(source_model_id, profile_name)
                model_hash = self._calculate_model_hash(optimized_model)
                
                metadata = EdgeModelMetadata(
                    model_id=model_id,
                    source_model_id=source_model_id,
                    deployment_profile=profile_name,
                    creation_timestamp=datetime.now(),
                    optimization_results=results,
                    model_hash=model_hash,
                    validation_status="validated" if results.meets_constraints else "failed",
                    deployment_ready=results.meets_constraints and 
                                   results.accuracy_retention >= profile.optimization_config.validation_threshold
                )
                
                # Save model and register
                model_path = self.registry.storage_path / model_id
                self.optimizer.save_optimized_model(
                    optimized_model, 
                    results, 
                    model_path,
                    {
                        'profile': profile_name,
                        'source_model_id': source_model_id,
                        'target_use_cases': profile.target_use_cases
                    }
                )
                
                self.registry.register_model(metadata)
                generated_models[profile_name] = metadata
                
                logger.info(f"Successfully generated {profile_name} model: "
                           f"{results.compression_ratio:.2f}x compression, "
                           f"{results.accuracy_retention:.3f} accuracy retention")
                
            except Exception as e:
                logger.error(f"Failed to generate model for profile {profile_name}: {e}")
        
        return generated_models
    
    async def generate_custom_edge_model(self, source_model: nn.Module,
                                       source_model_id: str,
                                       custom_constraints: HardwareConstraints,
                                       custom_config: OptimizationConfig,
                                       profile_name: str = "custom") -> EdgeModelMetadata:
        """
        Generate edge model with custom constraints and configuration
        
        Args:
            source_model: Original model
            source_model_id: Source model identifier
            custom_constraints: Custom hardware constraints
            custom_config: Custom optimization configuration
            profile_name: Name for the custom profile
            
        Returns:
            Generated model metadata
        """
        logger.info(f"Generating custom edge model: {profile_name}")
        
        # Generate optimized model
        optimized_model, results = self.optimizer.optimize_for_edge(
            source_model,
            custom_constraints,
            custom_config,
            self.test_data,
            self.test_labels
        )
        
        # Create metadata
        model_id = self._generate_model_id(source_model_id, profile_name)
        model_hash = self._calculate_model_hash(optimized_model)
        
        metadata = EdgeModelMetadata(
            model_id=model_id,
            source_model_id=source_model_id,
            deployment_profile=profile_name,
            creation_timestamp=datetime.now(),
            optimization_results=results,
            model_hash=model_hash,
            validation_status="validated" if results.meets_constraints else "failed",
            deployment_ready=results.meets_constraints and 
                           results.accuracy_retention >= custom_config.validation_threshold
        )
        
        # Save and register
        model_path = self.registry.storage_path / model_id
        self.optimizer.save_optimized_model(
            optimized_model,
            results,
            model_path,
            {
                'profile': profile_name,
                'source_model_id': source_model_id,
                'custom_constraints': asdict(custom_constraints),
                'custom_config': asdict(custom_config)
            }
        )
        
        self.registry.register_model(metadata)
        
        logger.info(f"Generated custom edge model: {model_id}")
        return metadata
    
    def get_recommended_model(self, use_case: str, 
                            constraints: Optional[HardwareConstraints] = None) -> Optional[EdgeModelMetadata]:
        """
        Get recommended edge model for specific use case and constraints
        
        Args:
            use_case: Target use case
            constraints: Optional hardware constraints
            
        Returns:
            Recommended model metadata or None
        """
        # Find profiles that support the use case
        compatible_profiles = []
        for profile in self.deployment_profiles.values():
            if use_case in profile.target_use_cases:
                compatible_profiles.append(profile.name)
        
        if not compatible_profiles:
            logger.warning(f"No profiles found for use case: {use_case}")
            return None
        
        # Get models for compatible profiles
        candidate_models = []
        for profile_name in compatible_profiles:
            models = self.registry.list_models_by_profile(profile_name)
            candidate_models.extend([m for m in models if m.deployment_ready])
        
        if not candidate_models:
            logger.warning(f"No deployment-ready models found for use case: {use_case}")
            return None
        
        # Filter by constraints if provided
        if constraints:
            candidate_models = [
                m for m in candidate_models
                if (m.optimization_results.optimized_size_mb <= constraints.max_memory_mb and
                    m.optimization_results.optimized_inference_time_ms <= constraints.max_inference_time_ms)
            ]
        
        if not candidate_models:
            logger.warning(f"No models meet the specified constraints for use case: {use_case}")
            return None
        
        # Return model with best accuracy retention
        recommended = max(candidate_models, 
                         key=lambda m: m.optimization_results.accuracy_retention)
        
        logger.info(f"Recommended model {recommended.model_id} for use case {use_case}")
        return recommended
    
    def update_deployment_profile(self, profile_name: str, 
                                profile: EdgeDeploymentProfile):
        """Update or add a deployment profile"""
        self.deployment_profiles[profile_name] = profile
        logger.info(f"Updated deployment profile: {profile_name}")
    
    def get_optimization_report(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed optimization report for a model"""
        metadata = self.registry.get_model(model_id)
        if not metadata:
            return None
        
        results = metadata.optimization_results
        
        return {
            'model_id': model_id,
            'source_model_id': metadata.source_model_id,
            'deployment_profile': metadata.deployment_profile,
            'creation_date': metadata.creation_timestamp.isoformat(),
            'optimization_summary': {
                'compression_ratio': f"{results.compression_ratio:.2f}x",
                'size_reduction': f"{results.original_size_mb:.1f}MB → {results.optimized_size_mb:.1f}MB",
                'speedup': f"{results.speedup_factor:.2f}x",
                'inference_time': f"{results.original_inference_time_ms:.1f}ms → {results.optimized_inference_time_ms:.1f}ms",
                'accuracy_retention': f"{results.accuracy_retention:.1%}",
                'memory_usage': f"{results.memory_usage_mb:.1f}MB"
            },
            'optimization_steps': results.optimization_steps,
            'meets_constraints': results.meets_constraints,
            'deployment_ready': metadata.deployment_ready,
            'validation_status': metadata.validation_status
        }
    
    def _generate_model_id(self, source_model_id: str, profile_name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{source_model_id}_edge_{profile_name}_{timestamp}"
    
    def _calculate_model_hash(self, model: nn.Module) -> str:
        """Calculate hash of model parameters"""
        model_str = ""
        for param in model.parameters():
            model_str += str(param.data.cpu().numpy().tobytes())
        
        return hashlib.md5(model_str.encode()).hexdigest()
    
    def _generate_test_data(self) -> torch.Tensor:
        """Generate synthetic test data for validation"""
        # This would be replaced with actual geological test data
        return torch.randn(100, 10)  # 100 samples, 10 features
    
    def _generate_test_labels(self) -> torch.Tensor:
        """Generate synthetic test labels for validation"""
        # This would be replaced with actual geological test labels
        return torch.randint(0, 3, (100,))  # 100 labels, 3 classes


# Integration with existing Helios models
class HeliosEdgeModelFactory:
    """Factory for creating edge models from Helios components"""
    
    def __init__(self, helios_core: HeliosCore, storage_path: Path):
        self.helios_core = helios_core
        self.edge_generator = AutomatedEdgeGenerator(helios_core, storage_path)
    
    async def create_edge_pinn_models(self) -> Dict[str, EdgeModelMetadata]:
        """Create edge-optimized PINN models"""
        logger.info("Creating edge-optimized PINN models")
        
        # Get PINN model from Helios core
        pinn_model = self.helios_core.pinn_engine.model
        
        return await self.edge_generator.generate_all_edge_models(
            pinn_model, "helios_pinn_v1"
        )
    
    async def create_edge_seismic_cnn_models(self) -> Dict[str, EdgeModelMetadata]:
        """Create edge-optimized seismic CNN models"""
        logger.info("Creating edge-optimized seismic CNN models")
        
        # Create a seismic CNN processor and get its model
        seismic_processor = SeismicCNNProcessor()
        seismic_model = seismic_processor.model
        
        return await self.edge_generator.generate_all_edge_models(
            seismic_model, "helios_seismic_cnn_v1"
        )
    
    async def create_edge_gan_models(self) -> Dict[str, EdgeModelMetadata]:
        """Create edge-optimized GAN models"""
        logger.info("Creating edge-optimized GAN models")
        
        # Get GAN generator from Helios core
        gan_generator = self.helios_core.gan_generator
        
        return await self.edge_generator.generate_all_edge_models(
            gan_generator, "helios_gan_generator_v1"
        )
    
    def get_edge_model_for_deployment(self, use_case: str, 
                                    hardware_constraints: HardwareConstraints) -> Optional[Tuple[nn.Module, EdgeModelMetadata]]:
        """
        Get ready-to-deploy edge model for specific use case and hardware
        
        Args:
            use_case: Target use case
            hardware_constraints: Hardware constraints
            
        Returns:
            Tuple of (model, metadata) or None
        """
        metadata = self.edge_generator.get_recommended_model(use_case, hardware_constraints)
        if not metadata:
            return None
        
        # Load the model
        model_path = self.edge_generator.registry.storage_path / metadata.model_id / "optimized_model.pth"
        
        try:
            # This is simplified - in practice you'd need to reconstruct the model architecture
            model = torch.load(model_path, map_location='cpu')
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load edge model {metadata.model_id}: {e}")
            return None