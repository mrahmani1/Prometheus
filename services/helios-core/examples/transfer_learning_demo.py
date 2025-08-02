"""
Transfer Learning Demo

This example demonstrates the transfer learning capabilities of the Helios AI engine,
including domain adaptation, few-shot learning, and model fine-tuning for different
geological settings.

Requirements addressed: 5.1, 5.2
"""

import logging
import signal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import transfer learning components
from core.domain_adaptation_utils import (
    AdaptationStrategySelector,
    DomainDataPreprocessor,
    GeologicalDomainRegistry,
)
from core.transfer_learning_engine import AdaptationConfig, TransferLearningEngine
from torch.utils.data import DataLoader, TensorDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeologicalDataSimulator:
    """Simulates geological data for different domains"""

    def __init__(self):
        self.domain_characteristics = {
            "yellowstone_usa": {
                "base_frequency": 2.0,
                "noise_level": 0.05,
                "amplitude_scale": 1.0,
                "seasonal_component": True,
            },
            "vesuvius_italy": {
                "base_frequency": 1.5,
                "noise_level": 0.10,
                "amplitude_scale": 0.8,
                "seasonal_component": False,
            },
            "kilauea_hawaii": {
                "base_frequency": 3.0,
                "noise_level": 0.02,
                "amplitude_scale": 1.2,
                "seasonal_component": True,
            },
            "kamchatka_russia": {
                "base_frequency": 1.0,
                "noise_level": 0.30,
                "amplitude_scale": 0.6,
                "seasonal_component": False,
            },
        }

    def generate_domain_data(
        self, domain_id: str, num_samples: int = 1000, num_features: int = 10
    ) -> tuple:
        """Generate synthetic geological data for a specific domain"""
        if domain_id not in self.domain_characteristics:
            raise ValueError(f"Unknown domain: {domain_id}")

        char = self.domain_characteristics[domain_id]

        # Generate time series
        t = np.linspace(0, 10, num_samples)

        # Base geological signal
        base_signal = char["amplitude_scale"] * np.sin(char["base_frequency"] * t)

        # Add seasonal component if applicable
        if char["seasonal_component"]:
            seasonal = 0.3 * np.sin(0.5 * t)
            base_signal += seasonal

        # Add noise
        noise = char["noise_level"] * np.random.randn(num_samples)
        signal = base_signal + noise

        # Create multi-dimensional features
        features = np.zeros((num_samples, num_features))
        for i in range(num_features):
            # Each feature is a variation of the base signal
            phase_shift = i * np.pi / num_features
            features[:, i] = signal * np.cos(phase_shift) + 0.1 * np.random.randn(
                num_samples
            )

        # Create targets (simplified geological response)
        targets = np.sum(
            features[:, :3], axis=1, keepdims=True
        ) + 0.1 * np.random.randn(num_samples, 1)

        return torch.FloatTensor(features), torch.FloatTensor(targets)

    def create_dataloader(
        self, domain_id: str, num_samples: int = 1000, batch_size: int = 32
    ) -> DataLoader:
        """Create DataLoader for a specific domain"""
        features, targets = self.generate_domain_data(domain_id, num_samples)
        dataset = TensorDataset(features, targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def demonstrate_domain_adaptation():
    """Demonstrate domain adaptation between different geological settings"""
    logger.info("=== Domain Adaptation Demo ===")

    # Initialize components
    engine = TransferLearningEngine()
    registry = GeologicalDomainRegistry()
    data_simulator = GeologicalDataSimulator()

    # Create source model (pre-trained on Yellowstone data)
    source_model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
    )

    # Simulate pre-training on source domain
    logger.info("Pre-training source model on Yellowstone data...")
    source_data = data_simulator.create_dataloader("yellowstone_usa", 2000, 32)

    # Simple training loop for source model
    optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    source_model.train()
    for epoch in range(50):
        epoch_loss = 0.0
        for batch_data, batch_targets in source_data:
            optimizer.zero_grad()
            predictions = source_model(batch_data)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            logger.info(
                f"Source training epoch {epoch}, Loss: {epoch_loss/len(source_data):.6f}"
            )

    # Now adapt to different target domains
    target_domains = ["vesuvius_italy", "kilauea_hawaii", "kamchatka_russia"]

    adaptation_results = {}

    for target_domain_id in target_domains:
        logger.info(f"\n--- Adapting to {target_domain_id} ---")

        # Get target domain
        target_domain = registry.get_domain(target_domain_id)

        # Generate target domain data
        target_data = data_simulator.create_dataloader(target_domain_id, 500, 32)

        # Create adaptation configuration
        config = AdaptationConfig(
            source_domain=None,
            target_domain=target_domain,
            adaptation_method="fine_tuning",
            learning_rate=1e-4,
            num_epochs=30,
            batch_size=32,
            regularization_strength=0.01,
        )

        # Perform adaptation
        result = engine.adapt_to_domain(
            source_model, target_domain, target_data, config
        )
        adaptation_results[target_domain_id] = result

        # Log results
        logger.info(f"Adaptation completed for {target_domain.region_name}")
        logger.info(f"Final loss: {result.adaptation_metrics['final_loss']:.6f}")
        logger.info(
            f"Loss reduction: {result.adaptation_metrics['loss_reduction']:.4f}"
        )
        logger.info(f"Confidence score: {result.confidence_score:.4f}")
        logger.info(
            f"MSE improvement: {result.validation_scores.get('mse_improvement', 0):.4f}"
        )

    return adaptation_results


def demonstrate_few_shot_learning():
    """Demonstrate few-shot learning for data-sparse regions"""
    logger.info("\n=== Few-Shot Learning Demo ===")

    # Initialize components
    engine = TransferLearningEngine()
    registry = GeologicalDomainRegistry()
    data_simulator = GeologicalDataSimulator()

    # Create pre-trained model
    pretrained_model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
    )

    # Get sparse data domain (Kamchatka)
    sparse_domain = registry.get_domain("kamchatka_russia")

    # Create very small dataset (few-shot scenario)
    few_shot_data = data_simulator.create_dataloader(
        "kamchatka_russia", 50, 10
    )  # Only 50 samples

    # Configuration for few-shot learning
    config = AdaptationConfig(
        source_domain=None,
        target_domain=sparse_domain,
        adaptation_method="few_shot",
        learning_rate=1e-3,
        num_epochs=20,
        few_shot_samples=10,
        batch_size=10,
    )

    # Perform few-shot adaptation
    logger.info(f"Performing few-shot adaptation for {sparse_domain.region_name}")
    logger.info(f"Using only {config.few_shot_samples} samples per task")

    result = engine.few_shot_adapt(
        pretrained_model, sparse_domain, few_shot_data, config
    )

    # Log results
    logger.info("Few-shot adaptation completed")
    logger.info(f"Confidence score: {result.confidence_score:.4f}")
    logger.info(
        f"Physics consistency: {result.validation_scores.get('physics_consistency', 0):.4f}"
    )
    logger.info(
        f"Geological plausibility: {result.validation_scores.get('geological_plausibility', 0):.4f}"
    )

    return result


def demonstrate_strategy_selection():
    """Demonstrate automatic strategy selection based on domain characteristics"""
    logger.info("\n=== Strategy Selection Demo ===")

    registry = GeologicalDomainRegistry()
    selector = AdaptationStrategySelector()

    # Test strategy selection for different domains
    test_domains = ["yellowstone_usa", "vesuvius_italy", "kamchatka_russia"]

    for domain_id in test_domains:
        domain = registry.get_domain(domain_id)

        logger.info(f"\n--- Strategy Selection for {domain.region_name} ---")
        logger.info(f"Data sparsity level: {domain.data_sparsity_level:.2f}")
        logger.info(f"Geological setting: {domain.geological_setting}")

        # Get strategy recommendations
        recommendations = selector.get_strategy_recommendations(domain)

        # Select best strategy
        best_strategy = selector.select_strategy(domain)

        logger.info(f"Recommended strategy: {best_strategy}")
        logger.info("Strategy scores:")
        for strategy, details in recommendations.items():
            logger.info(
                f"  {strategy}: {details['suitability_score']:.3f} "
                f"(complexity: {details['complexity']})"
            )


def demonstrate_domain_similarity():
    """Demonstrate domain similarity analysis"""
    logger.info("\n=== Domain Similarity Demo ===")

    registry = GeologicalDomainRegistry()

    # Find similar domains for each registered domain
    all_domains = registry.list_domains()

    for domain_id in all_domains[:3]:  # Test first 3 domains
        domain = registry.get_domain(domain_id)
        similar_domains = registry.get_similar_domains(domain_id, top_k=2)

        logger.info(f"\n--- Similar domains to {domain.region_name} ---")
        for similar_id, similarity_score in similar_domains:
            similar_domain = registry.get_domain(similar_id)
            logger.info(f"  {similar_domain.region_name}: {similarity_score:.3f}")
            logger.info(f"    Setting: {similar_domain.geological_setting}")
            logger.info(f"    Context: {similar_domain.tectonic_context}")


def demonstrate_data_preprocessing():
    """Demonstrate domain data preprocessing and visualization"""
    logger.info("\n=== Data Preprocessing Demo ===")

    preprocessor = DomainDataPreprocessor()
    data_simulator = GeologicalDataSimulator()

    # Generate data for two different domains
    source_features, _ = data_simulator.generate_domain_data("yellowstone_usa", 200)
    target_features, _ = data_simulator.generate_domain_data("vesuvius_italy", 200)

    # Preprocess for adaptation
    source_processed, target_processed = preprocessor.preprocess_for_adaptation(
        source_features.numpy(), target_features.numpy(), "vesuvius_italy"
    )

    logger.info(f"Original source data shape: {source_features.shape}")
    logger.info(f"Original target data shape: {target_features.shape}")
    logger.info(f"Processed source data shape: {source_processed.shape}")
    logger.info(f"Processed target data shape: {target_processed.shape}")

    # Extract domain-invariant features
    invariant_features = preprocessor.extract_domain_invariant_features(
        source_processed, "yellowstone_usa"
    )

    logger.info(f"Domain-invariant features shape: {invariant_features.shape}")

    # Visualize domain shift (if matplotlib is available)
    try:
        preprocessor.visualize_domain_shift(
            source_processed[:100],
            target_processed[:100],
            ("Yellowstone", "Vesuvius"),
            save_path="domain_shift_visualization.png",
        )
        logger.info(
            "Domain shift visualization saved to domain_shift_visualization.png"
        )
    except Exception as e:
        logger.warning(f"Could not create visualization: {e}")


def demonstrate_model_validation():
    """Demonstrate comprehensive model validation"""
    logger.info("\n=== Model Validation Demo ===")

    engine = TransferLearningEngine()
    registry = GeologicalDomainRegistry()
    data_simulator = GeologicalDataSimulator()

    # Create original and adapted models
    original_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    adapted_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    # Get validation domain and data
    validation_domain = registry.get_domain("vesuvius_italy")
    validation_data = data_simulator.create_dataloader("vesuvius_italy", 300, 32)

    # Perform validation
    validation_scores = engine.validator.validate_adapted_model(
        adapted_model, validation_data, original_model, validation_domain
    )

    logger.info("Validation Results:")
    for metric, score in validation_scores.items():
        logger.info(f"  {metric}: {score:.4f}")


def create_adaptation_report(adaptation_results):
    """Create a comprehensive adaptation report"""
    logger.info("\n=== Adaptation Report ===")

    report = {
        "total_adaptations": len(adaptation_results),
        "successful_adaptations": 0,
        "average_confidence": 0.0,
        "best_adaptation": None,
        "adaptation_details": {},
    }

    total_confidence = 0.0
    best_confidence = 0.0

    for domain_id, result in adaptation_results.items():
        confidence = result.confidence_score
        total_confidence += confidence

        if confidence > best_confidence:
            best_confidence = confidence
            report["best_adaptation"] = domain_id

        if confidence > 0.5:  # Threshold for successful adaptation
            report["successful_adaptations"] += 1

        report["adaptation_details"][domain_id] = {
            "confidence_score": confidence,
            "final_loss": result.adaptation_metrics.get("final_loss", 0),
            "loss_reduction": result.adaptation_metrics.get("loss_reduction", 0),
            "mse_improvement": result.validation_scores.get("mse_improvement", 0),
        }

    report["average_confidence"] = total_confidence / len(adaptation_results)

    # Print report
    logger.info(f"Total adaptations performed: {report['total_adaptations']}")
    logger.info(f"Successful adaptations: {report['successful_adaptations']}")
    logger.info(
        f"Success rate: {report['successful_adaptations']/report['total_adaptations']*100:.1f}%"
    )
    logger.info(f"Average confidence: {report['average_confidence']:.3f}")
    logger.info(f"Best adaptation: {report['best_adaptation']}")

    return report


def main():
    """Run all transfer learning demonstrations"""
    logger.info("Starting Transfer Learning Demonstration")
    logger.info("=" * 50)

    try:
        # Run demonstrations
        adaptation_results = demonstrate_domain_adaptation()
        few_shot_result = demonstrate_few_shot_learning()
        demonstrate_strategy_selection()
        demonstrate_domain_similarity()
        demonstrate_data_preprocessing()
        demonstrate_model_validation()

        # Create final report
        report = create_adaptation_report(adaptation_results)

        logger.info("\n" + "=" * 50)
        logger.info("Transfer Learning Demonstration Completed Successfully!")

        return {
            "adaptation_results": adaptation_results,
            "few_shot_result": few_shot_result,
            "report": report,
        }

    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    results = main()
