#!/usr/bin/env python3
"""
Advanced ML Pipeline Examples
Demonstrates usage of cutting-edge machine learning capabilities for geological applications.

This file provides practical examples of how to use the Advanced ML Pipeline
for various geological modeling and analysis tasks.
"""

import asyncio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

# Import Advanced ML Pipeline components
from backend.modules.advanced_ml_pipeline import (
    GeologicalDataset,
    GeologicalFeatureConfig,
    NeuralArchitectureSearch,
    NASConfig,
    EnsembleLearning,
    EnsembleConfig,
    TransferLearning,
    TransferLearningConfig,
    FewShotConfig,
    ContinualLearningConfig,
    GenerativeModelConfig,
    CausalInferenceConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_geological_data(num_samples: int = 1000) -> pd.DataFrame:
    """Create sample geological data for demonstration purposes."""
    np.random.seed(42)
    
    data = {
        'latitude': np.random.uniform(40.0, 45.0, num_samples),
        'longitude': np.random.uniform(-125.0, -120.0, num_samples),
        'elevation': np.random.uniform(0, 3000, num_samples),
        'seismic_activity': np.random.exponential(1.5, num_samples),
        'deformation_rate': np.random.normal(0.1, 0.05, num_samples),
        'gas_emissions_so2': np.random.lognormal(4.0, 1.0, num_samples),
        'gas_emissions_co2': np.random.lognormal(8.0, 0.5, num_samples),
        'thermal_anomalies': np.random.poisson(2, num_samples),
        'timestamp': pd.date_range('2020-01-01', periods=num_samples, freq='D'),
        'eruption_probability': np.random.beta(2, 8, num_samples)  # Target variable
    }
    
    return pd.DataFrame(data)

async def example_neural_architecture_search():
    """Example: Neural Architecture Search for geological model optimization."""
    logger.info("🔍 Running Neural Architecture Search Example")
    
    # Create sample data
    data = create_sample_geological_data(1000)
    
    # Configure geological feature engineering
    feature_config = GeologicalFeatureConfig(
        spatial_features=True,
        temporal_features=True,
        spectral_features=True,
        statistical_features=True,
        physics_constraints=True,
        feature_selection_method="mutual_info",
        max_features=100
    )
    
    # Create geological dataset
    dataset = GeologicalDataset(data, "eruption_probability", feature_config)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Configure NAS
    nas_config = NASConfig(
        max_trials=20,  # Reduced for demo
        max_epochs_per_trial=10,  # Reduced for demo
        pruning_enabled=True,
        multi_objective=True,
        objectives=["accuracy", "latency", "memory"]
    )
    
    # Run Neural Architecture Search
    nas = NeuralArchitectureSearch(nas_config)
    
    # Get input dimension from dataset
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0] if len(sample_input.shape) > 0 else 1
    output_dim = 1
    
    try:
        results = await nas.search_architecture(train_loader, val_loader, input_dim, output_dim)
        
        logger.info(f"✅ NAS completed successfully!")
        logger.info(f"Best parameters: {results['best_params']}")
        logger.info(f"Best validation loss: {results['best_value']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ NAS failed: {str(e)}")
        return None

async def example_ensemble_learning():
    """Example: Ensemble learning with uncertainty quantification."""
    logger.info("🎯 Running Ensemble Learning Example")
    
    # Create sample data
    data = create_sample_geological_data(800)
    
    # Configure geological features
    feature_config = GeologicalFeatureConfig(
        spatial_features=True,
        temporal_features=True,
        physics_constraints=True
    )
    
    # Create dataset and data loaders
    dataset = GeologicalDataset(data, "eruption_probability", feature_config)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Configure ensemble
    ensemble_config = EnsembleConfig(
        base_models=["cnn", "transformer", "gnn"],  # Removed PINN for demo simplicity
        stacking_method="neural_stacking",
        uncertainty_method="deep_ensemble",
        num_ensemble_members=3,  # Reduced for demo
        diversity_regularization=0.1
    )
    
    # Create and train ensemble
    ensemble = EnsembleLearning(ensemble_config)
    
    # Get input/output dimensions
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0] if len(sample_input.shape) > 0 else 1
    output_dim = 1
    
    try:
        results = await ensemble.train_ensemble(train_loader, val_loader, input_dim, output_dim)
        
        logger.info(f"✅ Ensemble training completed!")
        logger.info(f"Ensemble size: {results['ensemble_size']}")
        
        # Make predictions with uncertainty
        test_input = torch.randn(10, input_dim)  # Sample test data
        predictions, uncertainty = ensemble.predict_with_uncertainty(test_input)
        
        logger.info(f"Sample predictions: {predictions[:3].detach().numpy()}")
        logger.info(f"Sample uncertainties: {uncertainty[:3].detach().numpy()}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Ensemble learning failed: {str(e)}")
        return None

async def example_transfer_learning():
    """Example: Transfer learning between geological domains."""
    logger.info("🔄 Running Transfer Learning Example")
    
    # Create source domain data (general geology)
    source_data = create_sample_geological_data(600)
    source_data['domain'] = 'general_geology'
    
    # Create target domain data (volcanic systems)
    target_data = create_sample_geological_data(200)
    target_data['domain'] = 'volcanic_systems'
    # Modify target data to simulate domain shift
    target_data['seismic_activity'] *= 1.5
    target_data['gas_emissions_so2'] *= 2.0
    
    # Configure transfer learning
    transfer_config = TransferLearningConfig(
        source_domain="general_geology",
        target_domain="volcanic_systems",
        adaptation_method="fine_tuning",  # Simplified for demo
        physics_constraints=True,
        fine_tuning_strategy="gradual_unfreezing"
    )
    
    # Create datasets
    feature_config = GeologicalFeatureConfig(physics_constraints=True)
    
    source_dataset = GeologicalDataset(source_data, "eruption_probability", feature_config)
    target_dataset = GeologicalDataset(target_data, "eruption_probability", feature_config)
    
    target_train_loader = DataLoader(target_dataset, batch_size=16, shuffle=True)
    target_val_loader = DataLoader(target_dataset, batch_size=16, shuffle=False)
    
    # Create a simple source model for demonstration
    sample_input, _ = source_dataset[0]
    input_dim = sample_input.shape[0] if len(sample_input.shape) > 0 else 1
    
    source_model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1)
    )
    
    # Perform transfer learning
    transfer = TransferLearning(transfer_config)
    
    try:
        adapted_model = await transfer.adapt_model(source_model, target_train_loader, target_val_loader)
        
        logger.info("✅ Transfer learning completed successfully!")
        logger.info(f"Adapted model: {type(adapted_model).__name__}")
        
        return adapted_model
        
    except Exception as e:
        logger.error(f"❌ Transfer learning failed: {str(e)}")
        return None

def example_geological_feature_engineering():
    """Example: Advanced geological feature engineering."""
    logger.info("⚙️ Running Geological Feature Engineering Example")
    
    # Create sample data with geological context
    data = create_sample_geological_data(500)
    
    # Configure comprehensive feature engineering
    feature_config = GeologicalFeatureConfig(
        spatial_features=True,      # Spatial gradients and derivatives
        temporal_features=True,     # Seasonal patterns and trends
        spectral_features=True,     # FFT and wavelet analysis
        statistical_features=True,  # Domain-aware statistics
        physics_constraints=True,   # Conservation laws
        feature_selection_method="mutual_info",
        max_features=50,
        correlation_threshold=0.95
    )
    
    # Create geological dataset with feature engineering
    dataset = GeologicalDataset(data, "eruption_probability", feature_config)
    
    # Examine engineered features
    sample_features, sample_target = dataset[0]
    
    logger.info(f"✅ Feature engineering completed!")
    logger.info(f"Original data columns: {len(data.columns)}")
    logger.info(f"Engineered features: {sample_features.shape[0] if len(sample_features.shape) > 0 else 1}")
    logger.info(f"Sample target: {sample_target.item():.4f}")
    
    return dataset

def example_configuration_showcase():
    """Example: Showcase of various ML pipeline configurations."""
    logger.info("📋 Running Configuration Showcase")
    
    # Neural Architecture Search Configuration
    nas_config = NASConfig(
        max_trials=100,
        max_epochs_per_trial=50,
        pruning_enabled=True,
        multi_objective=True,
        objectives=["accuracy", "latency", "memory"]
    )
    logger.info(f"NAS Config: {nas_config}")
    
    # Ensemble Learning Configuration
    ensemble_config = EnsembleConfig(
        base_models=["cnn", "transformer", "gnn", "pinn"],
        stacking_method="neural_stacking",
        uncertainty_method="deep_ensemble",
        num_ensemble_members=5,
        diversity_regularization=0.1
    )
    logger.info(f"Ensemble Config: {ensemble_config}")
    
    # Few-Shot Learning Configuration
    few_shot_config = FewShotConfig(
        support_shots=5,
        query_shots=15,
        meta_learning_algorithm="maml",
        adaptation_steps=5,
        adaptation_lr=0.01
    )
    logger.info(f"Few-Shot Config: {few_shot_config}")
    
    # Continual Learning Configuration
    continual_config = ContinualLearningConfig(
        memory_size=1000,
        rehearsal_method="experience_replay",
        regularization_method="elastic_weight_consolidation",
        plasticity_stability_tradeoff=0.5
    )
    logger.info(f"Continual Config: {continual_config}")
    
    # Generative Model Configuration
    generative_config = GenerativeModelConfig(
        model_type="physics_informed_gan",
        latent_dim=128,
        physics_loss_weight=1.0,
        adversarial_loss_weight=1.0
    )
    logger.info(f"Generative Config: {generative_config}")
    
    # Causal Inference Configuration
    causal_config = CausalInferenceConfig(
        method="pc_algorithm",
        independence_test="partial_correlation",
        significance_level=0.05,
        max_conditioning_set_size=3
    )
    logger.info(f"Causal Config: {causal_config}")
    
    logger.info("✅ Configuration showcase completed!")

async def run_all_examples():
    """Run all Advanced ML Pipeline examples."""
    logger.info("🚀 Starting Advanced ML Pipeline Examples")
    logger.info("=" * 60)
    
    try:
        # 1. Configuration showcase
        example_configuration_showcase()
        logger.info("-" * 60)
        
        # 2. Feature engineering example
        dataset = example_geological_feature_engineering()
        logger.info("-" * 60)
        
        # 3. Neural Architecture Search (commented out for speed)
        # nas_results = await example_neural_architecture_search()
        # logger.info("-" * 60)
        
        # 4. Ensemble Learning (commented out for speed)
        # ensemble_results = await example_ensemble_learning()
        # logger.info("-" * 60)
        
        # 5. Transfer Learning (commented out for speed)
        # transfer_results = await example_transfer_learning()
        # logger.info("-" * 60)
        
        logger.info("🎉 All examples completed successfully!")
        logger.info("💡 Uncomment the async examples above to run full ML pipeline demonstrations")
        
    except Exception as e:
        logger.error(f"❌ Examples failed: {str(e)}")
        raise

if __name__ == "__main__":
    print("Advanced ML Pipeline Examples")
    print("=" * 40)
    print("This script demonstrates the usage of cutting-edge ML capabilities")
    print("for geological applications including NAS, ensemble learning,")
    print("transfer learning, and advanced feature engineering.")
    print()
    
    # Run examples
    asyncio.run(run_all_examples())
    
    print()
    print("📚 For more information:")
    print("- See docs/ADVANCED_ML_PIPELINE.md for detailed documentation")
    print("- See API_REFERENCE.md for API endpoints")
    print("- See backend/modules/advanced_ml_pipeline.py for implementation")