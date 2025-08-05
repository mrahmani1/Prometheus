"""
Active Learning Engine Demonstration

This script demonstrates the capabilities of the active learning engine
for geological data selection and model improvement.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from active_learning_engine import (
    ActiveLearningEngine,
    ActiveLearningConfig,
    GeologicalSample,
    create_sample_geological_data_consistent
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_uncertainty_sampling():
    """Demonstrate uncertainty-based sampling strategy"""
    print("\n" + "="*60)
    print("UNCERTAINTY-BASED SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure uncertainty-based active learning
    config = ActiveLearningConfig(
        strategy="uncertainty",
        batch_size=15,
        uncertainty_threshold=0.1
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using uncertainty-based strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    # Analyze sample type distribution
    selected_types = [sample.sample_type for sample in result.selected_samples]
    type_counts = {}
    for sample_type in selected_types:
        type_counts[sample_type] = type_counts.get(sample_type, 0) + 1
    
    print(f"\nSelected Sample Types Distribution:")
    for sample_type, count in type_counts.items():
        print(f"  {sample_type}: {count} samples")
    
    return result


def demonstrate_diversity_sampling():
    """Demonstrate diversity-based sampling strategy"""
    print("\n" + "="*60)
    print("DIVERSITY-BASED SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure diversity-based active learning
    config = ActiveLearningConfig(
        strategy="diversity",
        batch_size=15,
        diversity_weight=0.8
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using diversity-based strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    # Analyze geographic distribution
    locations = np.array([sample.location[:2] for sample in result.selected_samples])
    print(f"\nGeographic Distribution:")
    print(f"  Latitude range: {np.min(locations[:, 0]):.2f} to {np.max(locations[:, 0]):.2f}")
    print(f"  Longitude range: {np.min(locations[:, 1]):.2f} to {np.max(locations[:, 1]):.2f}")
    print(f"  Geographic spread: {np.std(locations.flatten()):.2f}")
    
    return result


def demonstrate_hybrid_sampling():
    """Demonstrate hybrid sampling strategy"""
    print("\n" + "="*60)
    print("HYBRID SAMPLING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset...")
    all_samples = create_sample_geological_data_consistent(500)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Configure hybrid active learning
    config = ActiveLearningConfig(
        strategy="hybrid",
        batch_size=15,
        diversity_weight=0.4,  # Balance between uncertainty and diversity
        uncertainty_threshold=0.05
    )
    
    engine = ActiveLearningEngine(config)
    
    # Create a simple model for demonstration
    import torch
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = DemoModel()
    
    # Select optimal samples
    logger.info("Selecting samples using hybrid strategy...")
    result = engine.select_optimal_samples(
        unlabeled_pool=unlabeled_pool,
        labeled_data=labeled_data,
        model=model
    )
    
    # Display results
    print(f"\nSelected {len(result.selected_samples)} samples")
    print(f"Strategy used: {result.strategy_used}")
    print("\nEvaluation Metrics:")
    for metric, value in result.evaluation_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nGeological Coverage:")
    for coverage_type, value in result.geological_coverage.items():
        print(f"  {coverage_type}: {value:.4f}")
    
    return result


def demonstrate_strategy_comparison():
    """Compare different active learning strategies"""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset for comparison...")
    all_samples = create_sample_geological_data_consistent(1000)
    labeled_data = all_samples[:100]
    unlabeled_pool = all_samples[100:]
    
    # Create model
    import torch
    import torch.nn as nn
    
    class ComparisonModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = ComparisonModel()
    
    # Test different strategies
    strategies = ['uncertainty', 'diversity', 'hybrid']
    results = {}
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy...")
        
        config = ActiveLearningConfig(
            strategy=strategy,
            batch_size=25,
            diversity_weight=0.3
        )
        
        engine = ActiveLearningEngine(config)
        
        result = engine.select_optimal_samples(
            unlabeled_pool=unlabeled_pool,
            labeled_data=labeled_data,
            model=model
        )
        
        results[strategy] = result
    
    # Compare results
    print("\nSTRATEGY COMPARISON RESULTS:")
    print("-" * 40)
    
    metrics_to_compare = [
        'sample_type_diversity',
        'diversity_score',
        'mean_uncertainty',
        'geological_information_gain'
    ]
    
    for metric in metrics_to_compare:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for strategy, result in results.items():
            value = result.evaluation_metrics.get(metric, 0.0)
            print(f"  {strategy:12}: {value:.4f}")
    
    # Coverage comparison
    print(f"\nGeological Coverage Comparison:")
    coverage_metrics = ['sample_type_coverage', 'geographic_coverage', 'depth_coverage']
    
    for metric in coverage_metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for strategy, result in results.items():
            value = result.geological_coverage.get(metric, 0.0)
            if metric == 'geographic_coverage':
                print(f"  {strategy:12}: {value:.2f}")
            else:
                print(f"  {strategy:12}: {value:.4f}")
    
    # Recommend best strategy
    print(f"\nSTRATEGY RECOMMENDATIONS:")
    print("-" * 30)
    
    recommendations = engine.get_strategy_recommendations(unlabeled_pool, labeled_data)
    for key, value in recommendations.items():
        print(f"{key}: {value}")
    
    return results


def demonstrate_iterative_learning():
    """Demonstrate iterative active learning process"""
    print("\n" + "="*60)
    print("ITERATIVE ACTIVE LEARNING DEMONSTRATION")
    print("="*60)
    
    # Create synthetic geological dataset
    logger.info("Creating synthetic geological dataset for iterative learning...")
    all_samples = create_sample_geological_data_consistent(1000)
    labeled_data = all_samples[:50]
    unlabeled_pool = all_samples[50:]
    
    # Create model
    import torch
    import torch.nn as nn
    
    class IterativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = IterativeModel()
    
    # Configure active learning
    config = ActiveLearningConfig(
        strategy="hybrid",
        batch_size=20,
        diversity_weight=0.3
    )
    
    engine = ActiveLearningEngine(config)
    
    # Simulate iterative learning process
    iterations = 5
    iteration_results = []
    
    print(f"Starting iterative learning with {len(labeled_data)} labeled samples")
    print(f"Unlabeled pool size: {len(unlabeled_pool)}")
    
    for iteration in range(iterations):
        logger.info(f"Iteration {iteration + 1}/{iterations}")
        
        # Select optimal samples
        result = engine.select_optimal_samples(
            unlabeled_pool=unlabeled_pool,
            labeled_data=labeled_data,
            model=model
        )
        
        # Move selected samples to labeled data
        selected_samples = result.selected_samples
        labeled_data.extend(selected_samples)
        
        # Remove selected samples from unlabeled pool
        selected_ids = {sample.id for sample in selected_samples}
        unlabeled_pool = [sample for sample in unlabeled_pool 
                         if sample.id not in selected_ids]
        
        # Store iteration results
        iteration_results.append({
            'iteration': iteration + 1,
            'labeled_count': len(labeled_data),
            'unlabeled_count': len(unlabeled_pool),
            'selected_count': len(selected_samples),
            'diversity_score': result.evaluation_metrics.get('diversity_score', 0.0),
            'sample_type_diversity': result.evaluation_metrics.get('sample_type_diversity', 0.0),
            'geological_coverage': result.geological_coverage.get('sample_type_coverage', 0.0)
        })
        
        print(f"  Iteration {iteration + 1}: Selected {len(selected_samples)} samples")
        print(f"    Labeled data: {len(labeled_data)} samples")
        print(f"    Unlabeled pool: {len(unlabeled_pool)} samples")
        print(f"    Diversity score: {result.evaluation_metrics.get('diversity_score', 0.0):.4f}")
    
    # Display final results
    print(f"\nITERATIVE LEARNING SUMMARY:")
    print("-" * 30)
    print(f"Initial labeled samples: 50")
    print(f"Final labeled samples: {len(labeled_data)}")
    print(f"Total samples selected: {len(labeled_data) - 50}")
    print(f"Remaining unlabeled samples: {len(unlabeled_pool)}")
    
    # Show progression
    print(f"\nLearning Progression:")
    for result in iteration_results:
        print(f"  Iteration {result['iteration']}: "
              f"Labeled={result['labeled_count']}, "
              f"Diversity={result['diversity_score']:.3f}, "
              f"Coverage={result['geological_coverage']:.3f}")
    
    return iteration_results


def main():
    """Main demonstration function"""
    print("ACTIVE LEARNING ENGINE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration showcases the capabilities of the active learning engine")
    print("for optimal geological data selection and model improvement.")
    
    try:
        # Run demonstrations
        uncertainty_result = demonstrate_uncertainty_sampling()
        diversity_result = demonstrate_diversity_sampling()
        hybrid_result = demonstrate_hybrid_sampling()
        comparison_results = demonstrate_strategy_comparison()
        iterative_results = demonstrate_iterative_learning()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*80)
        print("All active learning strategies have been demonstrated.")
        print("The engine is ready for integration with geological modeling workflows.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        print(f"\nError during demonstration: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nActive learning engine demonstration completed successfully!")
    else:
        print("\nActive learning engine demonstration failed!")
        sys.exit(1)