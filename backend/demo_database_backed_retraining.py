"""
Demonstration of Database-Backed Model Retraining

This script demonstrates the enhanced continuous learning framework that uses
the database to store and retrieve training data for intelligent model retraining.
"""

import asyncio
import numpy as np
import logging
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.advanced_ml_volcanic_infrastructure import (
    AdvancedMLVolcanicInfrastructure,
    VolcanicDataset,
    MLModelConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseBackedRetrainingDemo:
    """Demonstration of database-backed model retraining capabilities"""
    
    def __init__(self):
        self.ml_infrastructure = None
        self.temp_dir = None
        
    async def setup_demo_environment(self):
        """Setup demonstration environment with temporary database"""
        logger.info("Setting up demonstration environment...")
        
        # Create temporary directory for models and data
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Initialize ML infrastructure with SQLite for demo
        # In production, this would use PostgreSQL
        db_url = f"sqlite:///{self.temp_dir}/demo.db"
        
        try:
            self.ml_infrastructure = AdvancedMLVolcanicInfrastructure(db_url=db_url)
            logger.info("✓ ML Infrastructure initialized with temporary database")
        except Exception as e:
            logger.error(f"✗ Failed to initialize ML infrastructure: {e}")
            raise
    
    def cleanup_demo_environment(self):
        """Cleanup demonstration environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("✓ Cleaned up temporary directory")
    
    def create_synthetic_volcanic_data(self, n_samples: int, noise_level: float = 0.1, 
                                     data_shift: float = 0.0) -> VolcanicDataset:
        """Create synthetic volcanic monitoring data"""
        np.random.seed(42)  # For reproducibility
        
        # Generate features representing volcanic monitoring parameters
        features = np.random.rand(n_samples, 12)
        
        # Add data shift to simulate distribution drift
        if data_shift > 0:
            features += np.random.normal(data_shift, data_shift * 0.1, features.shape)
        
        # Create complex eruption probability based on multiple factors
        eruption_prob = (
            0.25 * np.sin(features[:, 0] * np.pi) +  # Seismic activity (periodic)
            0.20 * features[:, 1] ** 2 +  # Deformation rate (quadratic)
            0.15 * np.exp(-features[:, 2] * 2) +  # Gas composition (exponential decay)
            0.15 * features[:, 3] * features[:, 4] +  # Temperature-pressure interaction
            0.10 * np.tanh(features[:, 5] * 3) +  # Ground tilt (saturating)
            0.10 * features[:, 6] * np.sin(features[:, 7] * 2 * np.pi) +  # Periodic coupling
            0.05 * np.random.normal(0, noise_level, n_samples)  # Noise
        )
        
        # Normalize to [0, 1] range
        eruption_prob = np.clip(eruption_prob, 0, 1).reshape(-1, 1)
        
        feature_names = [
            'seismic_amplitude', 'deformation_rate', 'so2_flux', 'temperature', 'pressure',
            'ground_tilt', 'earthquake_count', 'spectral_ratio', 'correlation_dim', 
            'thermal_anomaly', 'gas_ratio', 'energy_release'
        ]
        
        return VolcanicDataset(
            features=features,
            targets=eruption_prob,
            feature_names=feature_names,
            target_names=['eruption_probability'],
            metadata={
                'dataset_type': 'synthetic_volcanic',
                'n_samples': n_samples,
                'noise_level': noise_level,
                'data_shift': data_shift,
                'generation_timestamp': np.datetime64('now').isoformat()
            }
        )
    
    async def demonstrate_initial_training(self):
        """Demonstrate initial model training with data storage"""
        logger.info("\n" + "="*60)
        logger.info("INITIAL MODEL TRAINING WITH DATABASE STORAGE")
        logger.info("="*60)
        
        # Create initial training dataset
        initial_data = self.create_synthetic_volcanic_data(n_samples=1000, noise_level=0.08)
        
        # Split into train/validation
        train_size = int(0.8 * len(initial_data.features))
        
        train_data = VolcanicDataset(
            features=initial_data.features[:train_size],
            targets=initial_data.targets[:train_size],
            feature_names=initial_data.feature_names,
            target_names=initial_data.target_names,
            metadata=initial_data.metadata
        )
        
        val_data = VolcanicDataset(
            features=initial_data.features[train_size:],
            targets=initial_data.targets[train_size:],
            feature_names=initial_data.feature_names,
            target_names=initial_data.target_names,
            metadata=initial_data.metadata
        )
        
        logger.info(f"Created initial dataset: {len(train_data.features)} training, {len(val_data.features)} validation samples")
        
        # Train ensemble model
        ensemble_config = {
            'base_models': ['random_forest', 'gradient_boosting'],
            'model_configs': {
                'random_forest': {'n_estimators': 50, 'max_depth': 8, 'random_state': 42},
                'gradient_boosting': {'n_estimators': 50, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42}
            }
        }
        
        logger.info("Training initial ensemble model...")
        training_result = await self.ml_infrastructure.train_ensemble_model(
            train_data, val_data, ensemble_config
        )
        
        logger.info(f"✓ Initial training completed in {training_result.training_time:.2f}s")
        logger.info(f"  Model ID: {training_result.model_id}")
        logger.info(f"  Validation MSE: {training_result.validation_metrics['validation_mse']:.6f}")
        logger.info(f"  Validation R²: {training_result.validation_metrics.get('validation_r2', 0):.4f}")
        
        # Verify training data was stored
        training_data_list = await self.ml_infrastructure.model_versioning.list_training_data()
        logger.info(f"✓ Training data stored in database: {len(training_data_list)} datasets")
        
        return training_result.model_id, train_data, val_data
    
    async def demonstrate_performance_monitoring(self, model_id: str, original_val_data: VolcanicDataset):
        """Demonstrate performance monitoring with new data"""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE MONITORING WITH NEW DATA")
        logger.info("="*60)
        
        # Create new data with distribution shift to simulate real-world drift
        new_data = self.create_synthetic_volcanic_data(
            n_samples=200, 
            noise_level=0.12,  # Increased noise
            data_shift=0.1     # Distribution shift
        )
        
        logger.info(f"Generated new monitoring data: {len(new_data.features)} samples with distribution shift")
        
        # Monitor performance on new data
        performance_drift = await self.ml_infrastructure.continuous_learning.monitor_model_performance(
            model_id, new_data
        )
        
        logger.info("Performance drift analysis:")
        logger.info(f"  Current MSE: {performance_drift['current_mse']:.6f}")
        logger.info(f"  Baseline MSE: {performance_drift['baseline_mse']:.6f}")
        logger.info(f"  MSE drift: {performance_drift['mse_drift']:.2%}")
        logger.info(f"  Current R²: {performance_drift['current_r2']:.4f}")
        logger.info(f"  Baseline R²: {performance_drift['baseline_r2']:.4f}")
        logger.info(f"  R² drift: {performance_drift['r2_drift']:.2%}")
        logger.info(f"  Needs retraining: {performance_drift['needs_retraining']}")
        
        return new_data, performance_drift
    
    async def demonstrate_database_backed_retraining(self, model_id: str, new_data: VolcanicDataset):
        """Demonstrate database-backed model retraining"""
        logger.info("\n" + "="*60)
        logger.info("DATABASE-BACKED MODEL RETRAINING")
        logger.info("="*60)
        
        # Get model metadata
        metadata = await self.ml_infrastructure.model_versioning.get_model_metadata(model_id)
        logger.info(f"Original model: {metadata['model_name']} (ID: {model_id})")
        
        # Demonstrate full retraining with database query
        logger.info("Initiating full retraining with database-stored training data...")
        
        retrained_model_id = await self.ml_infrastructure.continuous_learning.trigger_retraining(
            model_id, new_data, 'full_retrain'
        )
        
        logger.info(f"✓ Retraining completed. New model ID: {retrained_model_id}")
        
        # Get performance metrics for comparison
        retrained_metadata = await self.ml_infrastructure.model_versioning.get_model_metadata(retrained_model_id)
        
        logger.info("Performance comparison:")
        logger.info(f"  Original model validation MSE: {metadata['performance_metrics'].get('validation_mse', 0):.6f}")
        logger.info(f"  Retrained model validation MSE: {retrained_metadata['performance_metrics'].get('validation_mse', 0):.6f}")
        
        if 'train_mse' in retrained_metadata['performance_metrics']:
            logger.info(f"  Retrained model train MSE: {retrained_metadata['performance_metrics']['train_mse']:.6f}")
        if 'test_mse' in retrained_metadata['performance_metrics']:
            logger.info(f"  Retrained model test MSE: {retrained_metadata['performance_metrics']['test_mse']:.6f}")
        
        # Show data combination details
        retrained_training_type = retrained_metadata['performance_metrics'].get('training_type', 'unknown')
        data_combination = retrained_metadata['performance_metrics'].get('data_combination', 'unknown')
        total_samples = retrained_metadata['performance_metrics'].get('total_training_samples', 0)
        
        logger.info(f"  Training type: {retrained_training_type}")
        logger.info(f"  Data combination: {data_combination}")
        logger.info(f"  Total training samples used: {total_samples}")
        
        return retrained_model_id
    
    async def demonstrate_training_data_management(self):
        """Demonstrate training data management capabilities"""
        logger.info("\n" + "="*60)
        logger.info("TRAINING DATA MANAGEMENT")
        logger.info("="*60)
        
        # List all stored training data
        training_data_list = await self.ml_infrastructure.model_versioning.list_training_data()
        
        logger.info(f"Total training datasets stored: {len(training_data_list)}")
        
        for i, data_info in enumerate(training_data_list, 1):
            logger.info(f"  Dataset {i}:")
            logger.info(f"    Hash: {data_info['data_hash'][:16]}...")
            logger.info(f"    Model ID: {data_info['model_id']}")
            logger.info(f"    Size: {data_info['data_size']} samples")
            logger.info(f"    Features: {len(data_info['feature_names'])}")
            logger.info(f"    Created: {data_info['created_at']}")
            logger.info(f"    Compressed: {data_info['is_compressed']}")
            
            # Show metadata if available
            if 'metadata' in data_info and data_info['metadata']:
                metadata = data_info['metadata']
                if 'dataset_type' in metadata:
                    logger.info(f"    Type: {metadata['dataset_type']}")
                if 'noise_level' in metadata:
                    logger.info(f"    Noise level: {metadata['noise_level']}")
                if 'data_shift' in metadata:
                    logger.info(f"    Data shift: {metadata['data_shift']}")
    
    async def demonstrate_fallback_behavior(self):
        """Demonstrate fallback behavior when no original training data is available"""
        logger.info("\n" + "="*60)
        logger.info("FALLBACK BEHAVIOR DEMONSTRATION")
        logger.info("="*60)
        
        # Create a new dataset for training a model without storing training data
        fallback_data = self.create_synthetic_volcanic_data(n_samples=300, noise_level=0.1)
        
        # Split data
        train_size = int(0.8 * len(fallback_data.features))
        
        train_data = VolcanicDataset(
            features=fallback_data.features[:train_size],
            targets=fallback_data.targets[:train_size],
            feature_names=fallback_data.feature_names,
            target_names=fallback_data.target_names,
            metadata={'test_fallback': True}
        )
        
        val_data = VolcanicDataset(
            features=fallback_data.features[train_size:],
            targets=fallback_data.targets[train_size:],
            feature_names=fallback_data.feature_names,
            target_names=fallback_data.target_names,
            metadata={'test_fallback': True}
        )
        
        # Train model
        ensemble_config = {
            'base_models': ['random_forest'],
            'model_configs': {
                'random_forest': {'n_estimators': 30, 'max_depth': 6, 'random_state': 42}
            }
        }
        
        logger.info("Training model for fallback demonstration...")
        training_result = await self.ml_infrastructure.train_ensemble_model(
            train_data, val_data, ensemble_config
        )
        
        fallback_model_id = training_result.model_id
        logger.info(f"✓ Fallback test model trained: {fallback_model_id}")
        
        # Manually remove training data to simulate missing data scenario
        logger.info("Simulating missing training data scenario...")
        
        # Create new data for retraining
        new_fallback_data = self.create_synthetic_volcanic_data(n_samples=100, noise_level=0.15)
        
        # Mock the retrieve_training_data method to return None
        original_retrieve = self.ml_infrastructure.model_versioning.retrieve_training_data
        
        async def mock_retrieve_none(model_id):
            logger.info(f"Simulated: No training data found for model {model_id}")
            return None
        
        self.ml_infrastructure.model_versioning.retrieve_training_data = mock_retrieve_none
        
        try:
            # Attempt retraining - should fallback to new data only
            logger.info("Attempting retraining with missing original data...")
            
            fallback_retrained_id = await self.ml_infrastructure.continuous_learning.trigger_retraining(
                fallback_model_id, new_fallback_data, 'full_retrain'
            )
            
            logger.info(f"✓ Fallback retraining completed: {fallback_retrained_id}")
            
            # Check the metadata to confirm fallback behavior
            fallback_metadata = await self.ml_infrastructure.model_versioning.get_model_metadata(fallback_retrained_id)
            
            data_combination = fallback_metadata['performance_metrics'].get('data_combination', 'unknown')
            fallback_training = fallback_metadata['performance_metrics'].get('fallback_training', False)
            
            logger.info(f"  Data combination strategy: {data_combination}")
            logger.info(f"  Fallback training used: {fallback_training}")
            
            if data_combination == 'new_only':
                logger.info("✓ Fallback behavior working correctly - used new data only")
            else:
                logger.warning("⚠ Unexpected data combination strategy")
                
        finally:
            # Restore original method
            self.ml_infrastructure.model_versioning.retrieve_training_data = original_retrieve
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of database-backed retraining"""
        logger.info("🔄 DATABASE-BACKED MODEL RETRAINING DEMONSTRATION 🔄")
        logger.info("=" * 80)
        
        try:
            # Setup environment
            await self.setup_demo_environment()
            
            # Run demonstrations
            model_id, train_data, val_data = await self.demonstrate_initial_training()
            
            new_data, performance_drift = await self.demonstrate_performance_monitoring(model_id, val_data)
            
            if performance_drift['needs_retraining']:
                retrained_model_id = await self.demonstrate_database_backed_retraining(model_id, new_data)
            else:
                logger.info("Performance drift not significant enough to trigger retraining")
                # Force retraining for demonstration
                logger.info("Forcing retraining for demonstration purposes...")
                retrained_model_id = await self.demonstrate_database_backed_retraining(model_id, new_data)
            
            await self.demonstrate_training_data_management()
            
            await self.demonstrate_fallback_behavior()
            
            logger.info("\n" + "="*80)
            logger.info("🎉 DATABASE-BACKED RETRAINING DEMONSTRATION COMPLETED! 🎉")
            logger.info("="*80)
            logger.info("\nKey Features Demonstrated:")
            logger.info("✓ Automatic training data storage with compression")
            logger.info("✓ Database-backed retrieval of original training data")
            logger.info("✓ Intelligent combination of original and new training data")
            logger.info("✓ Performance monitoring and drift detection")
            logger.info("✓ Graceful fallback when original data is unavailable")
            logger.info("✓ Comprehensive training data management")
            logger.info("✓ Model versioning with performance tracking")
            
            logger.info("\nThe enhanced continuous learning framework is ready for production use!")
            
        except Exception as e:
            logger.error(f"✗ Demo failed: {e}")
            raise
        finally:
            self.cleanup_demo_environment()

async def main():
    """Main demonstration function"""
    demo = DatabaseBackedRetrainingDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main())