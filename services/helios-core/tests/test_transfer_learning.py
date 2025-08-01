"""
Tests for Transfer Learning Engine

This module contains comprehensive tests for the transfer learning functionality,
including domain adaptation, few-shot learning, and model validation.

Requirements addressed: 5.1, 5.2
"""

from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from core.domain_adaptation_utils import (
    AdaptationStrategySelector,
    DataCharacteristics,
    DomainDataPreprocessor,
    GeologicalDomainRegistry,
    GeologicalSetting,
    TectonicContext,
)
from core.transfer_learning_engine import (
    AdaptationConfig,
    AdaptationResult,
    DomainAdversarialAdapter,
    FewShotLearner,
    FineTuningAdapter,
    GeologicalDomain,
    ModelPerformanceValidator,
    TransferLearningEngine,
)
from torch.utils.data import DataLoader, TensorDataset

class SimpleTestModel(nn.Module):
    """Simple neural network for testing"""

    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.layers(x)

@pytest.fixture
def sample_model():
    """Create a sample model for testing"""
    return SimpleTestModel()

@pytest.fixture
def sample_domain():
    """Create a sample geological domain for testing"""
    return GeologicalDomain(
        domain_id="test_domain",
        region_name="Test Volcano",
        geological_setting="stratovolcano",
        tectonic_context="convergent_margin",
        data_characteristics={
            "data_quality": 0.8,
            "completeness": 0.9,
            "noise_level": 0.1,
        },
        available_data_types=["seismic", "deformation", "thermal"],
        data_sparsity_level=0.7,
    )

@pytest.fixture
def sample_data():
    """Create sample training data"""
    num_samples = 100
    num_features = 10

    data = torch.randn(num_samples, num_features)
    targets = torch.randn(num_samples, 1)

    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    return dataloader

@pytest.fixture
def adaptation_config(sample_domain):
    """Create sample adaptation configuration"""
    return AdaptationConfig(
        source_domain=None,
        target_domain=sample_domain,
        adaptation_method="fine_tuning",
        learning_rate=1e-3,
        num_epochs=10,
        batch_size=16,
        regularization_strength=0.01,
        few_shot_samples=5,
        validation_split=0.2,
    )

class TestFineTuningAdapter:
    """Test fine-tuning adaptation strategy"""

    def test_fine_tuning_adapter_initialization(self):
        """Test fine-tuning adapter initialization"""
        adapter = FineTuningAdapter()
        assert adapter.name == "fine_tuning"

    def test_adapt_model(self, sample_model, sample_data, adaptation_config):
        """Test model adaptation using fine-tuning"""
        adapter = FineTuningAdapter()

        # Perform adaptation
        result = adapter.adapt_model(sample_model, sample_data, adaptation_config)

        # Verify result structure
        assert isinstance(result, AdaptationResult)
        assert result.adapted_model is not None
        assert isinstance(result.adaptation_metrics, dict)
        assert isinstance(result.convergence_history, list)
        assert len(result.convergence_history) == adaptation_config.num_epochs

        # Verify metrics
        assert "final_loss" in result.adaptation_metrics
        assert "best_loss" in result.adaptation_metrics
        assert "loss_reduction" in result.adaptation_metrics
        assert result.confidence_score >= 0.0 and result.confidence_score <= 1.0

    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        adapter = FineTuningAdapter()

        metrics = {"loss_reduction": 0.5, "convergence_stability": 0.1}

        confidence = adapter._calculate_confidence(metrics)
        assert 0.0 <= confidence <= 1.0

class TestDomainAdversarialAdapter:
    """Test domain adversarial adaptation strategy"""

    def test_domain_adversarial_adapter_initialization(self):
        """Test domain adversarial adapter initialization"""
        adapter = DomainAdversarialAdapter()
        assert adapter.name == "domain_adversarial"

    def test_adapt_model(self, sample_model, sample_data, adaptation_config):
        """Test model adaptation using domain adversarial training"""
        adapter = DomainAdversarialAdapter()

        # Perform adaptation
        result = adapter.adapt_model(sample_model, sample_data, adaptation_config)

        # Verify result structure
        assert isinstance(result, AdaptationResult)
        assert result.adapted_model is not None
        assert isinstance(result.adaptation_metrics, dict)
        assert "domain_confusion" in result.adaptation_metrics

    def test_create_domain_classifier(self, sample_model):
        """Test domain classifier creation"""
        adapter = DomainAdversarialAdapter()

        domain_classifier = adapter._create_domain_classifier(sample_model)

        assert isinstance(domain_classifier, nn.Module)
        # Test forward pass
        test_input = torch.randn(5, 128)
        output = domain_classifier(test_input)
        assert output.shape == (5, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output

class TestFewShotLearner:
    """Test few-shot learning functionality"""

    def test_few_shot_learner_initialization(self):
        """Test few-shot learner initialization"""
        learner = FewShotLearner()
        assert learner.support_set_size == 10
        assert learner.query_set_size == 5

    def test_meta_learn(self, sample_model, sample_data, adaptation_config):
        """Test meta-learning for few-shot adaptation"""
        learner = FewShotLearner()

        few_shot_tasks = [sample_data]

        meta_model = learner.meta_learn(sample_model, few_shot_tasks, adaptation_config)

        assert isinstance(meta_model, nn.Module)
        # Verify model structure is preserved
        assert len(list(meta_model.parameters())) == len(
            list(sample_model.parameters())
        )

class TestModelPerformanceValidator:
    """Test model performance validation"""

    def test_validator_initialization(self):
        """Test validator initialization"""
        validator = ModelPerformanceValidator()
        expected_metrics = [
            "mse",
            "r2",
            "physics_consistency",
            "geological_plausibility",
        ]
        assert validator.validation_metrics == expected_metrics

    def test_validate_adapted_model(self, sample_model, sample_data, sample_domain):
        """Test adapted model validation"""
        validator = ModelPerformanceValidator()

        # Create a slightly different model as "adapted" model
        adapted_model = SimpleTestModel()

        validation_scores = validator.validate_adapted_model(
            adapted_model, sample_data, sample_model, sample_domain
        )

        # Verify all expected metrics are present
        expected_keys = [
            "adapted_mse",
            "original_mse",
            "adapted_r2",
            "original_r2",
            "mse_improvement",
            "r2_improvement",
            "physics_consistency",
            "geological_plausibility",
        ]

        for key in expected_keys:
            assert key in validation_scores

        # Verify metric ranges
        assert validation_scores["physics_consistency"] >= 0.0
        assert validation_scores["geological_plausibility"] >= 0.0

    def test_physics_consistency_check(self, sample_domain):
        """Test physics consistency checking"""
        validator = ModelPerformanceValidator()

        predictions = np.random.randn(100, 1)
        consistency_score = validator._check_physics_consistency(
            predictions, sample_domain
        )

        assert 0.0 <= consistency_score <= 1.0

    def test_geological_plausibility_check(self, sample_domain):
        """Test geological plausibility checking"""
        validator = ModelPerformanceValidator()

        predictions = np.random.randn(100, 1)
        plausibility_score = validator._check_geological_plausibility(
            predictions, sample_domain
        )

        assert 0.0 <= plausibility_score <= 1.0

class TestTransferLearningEngine:
    """Test main transfer learning engine"""

    def test_engine_initialization(self):
        """Test transfer learning engine initialization"""
        engine = TransferLearningEngine()

        assert "fine_tuning" in engine.adaptation_strategies
        assert "domain_adversarial" in engine.adaptation_strategies
        assert isinstance(engine.few_shot_learner, FewShotLearner)
        assert isinstance(engine.validator, ModelPerformanceValidator)
        assert isinstance(engine.adaptation_history, list)

    def test_adapt_to_domain(
        self, sample_model, sample_domain, sample_data, adaptation_config
    ):
        """Test domain adaptation"""
        engine = TransferLearningEngine()

        result = engine.adapt_to_domain(
            sample_model, sample_domain, sample_data, adaptation_config
        )

        assert isinstance(result, AdaptationResult)
        assert result.adapted_model is not None
        assert isinstance(result.validation_scores, dict)

        # Verify adaptation history is updated
        assert len(engine.adaptation_history) == 1
        assert engine.adaptation_history[0]["target_domain"] == sample_domain.domain_id

    def test_few_shot_adapt(
        self, sample_model, sample_domain, sample_data, adaptation_config
    ):
        """Test few-shot adaptation"""
        engine = TransferLearningEngine()

        result = engine.few_shot_adapt(
            sample_model, sample_domain, sample_data, adaptation_config
        )

        assert isinstance(result, AdaptationResult)
        assert result.adapted_model is not None
        assert "few_shot_samples" in result.adaptation_metrics
        assert result.confidence_score <= 1.0  # Few-shot typically has lower confidence

    def test_get_adaptation_recommendations(self, sample_domain):
        """Test adaptation recommendations"""
        engine = TransferLearningEngine()

        recommendations = engine.get_adaptation_recommendations(sample_domain)

        expected_keys = [
            "recommended_strategy",
            "estimated_samples_needed",
            "expected_performance_gain",
            "adaptation_complexity",
        ]

        for key in expected_keys:
            assert key in recommendations

    def test_save_adaptation_history(
        self, sample_model, sample_domain, sample_data, adaptation_config, tmp_path
    ):
        """Test saving adaptation history"""
        engine = TransferLearningEngine()

        # Perform an adaptation to create history
        engine.adapt_to_domain(
            sample_model, sample_domain, sample_data, adaptation_config
        )

        # Save history
        history_file = tmp_path / "adaptation_history.json"
        engine.save_adaptation_history(str(history_file))

        assert history_file.exists()

        # Verify file content
        import json

        with open(history_file, "r") as f:
            history_data = json.load(f)

        assert len(history_data) == 1
        assert history_data[0]["target_domain"] == sample_domain.domain_id

class TestGeologicalDomainRegistry:
    """Test geological domain registry"""

    def test_registry_initialization(self):
        """Test domain registry initialization"""
        registry = GeologicalDomainRegistry()

        # Should have some pre-initialized domains
        assert len(registry.domains) > 0
        assert "yellowstone_usa" in registry.domains
        assert "vesuvius_italy" in registry.domains

    def test_register_domain(self):
        """Test domain registration"""
        registry = GeologicalDomainRegistry()

        data_char = DataCharacteristics(
            temporal_resolution="minutes",
            spatial_resolution=100.0,
            data_quality=0.8,
            completeness=0.9,
            noise_level=0.1,
            sensor_types=["seismometer"],
            measurement_frequency=1.0,
            historical_coverage=10,
        )

        registry.register_domain(
            domain_id="test_volcano",
            region_name="Test Volcano",
            geological_setting=GeologicalSetting.STRATOVOLCANO,
            tectonic_context=TectonicContext.CONVERGENT_MARGIN,
            data_characteristics=data_char,
            available_data_types=["seismic"],
            data_sparsity_level=0.5,
        )

        assert "test_volcano" in registry.domains
        domain = registry.get_domain("test_volcano")
        assert domain.region_name == "Test Volcano"

    def test_get_similar_domains(self):
        """Test finding similar domains"""
        registry = GeologicalDomainRegistry()

        similar_domains = registry.get_similar_domains("yellowstone_usa", top_k=2)

        assert len(similar_domains) <= 2
        for domain_id, similarity_score in similar_domains:
            assert isinstance(domain_id, str)
            assert 0.0 <= similarity_score <= 1.0

    def test_domain_similarity_calculation(self):
        """Test domain similarity calculation"""
        registry = GeologicalDomainRegistry()

        domain1 = registry.get_domain("yellowstone_usa")
        domain2 = registry.get_domain("vesuvius_italy")

        similarity = registry._calculate_domain_similarity(domain1, domain2)
        assert 0.0 <= similarity <= 1.0

    def test_export_domains(self, tmp_path):
        """Test domain registry export"""
        registry = GeologicalDomainRegistry()

        export_file = tmp_path / "domains_export.json"
        registry.export_domains(str(export_file))

        assert export_file.exists()

        # Verify export content
        import json

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

        with open(export_file, "r") as f:
            export_data = json.load(f)

        assert len(export_data) == len(registry.domains)
        assert "yellowstone_usa" in export_data

class TestDomainDataPreprocessor:
    """Test domain data preprocessing"""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = DomainDataPreprocessor()

        assert isinstance(preprocessor.scalers, dict)
        assert isinstance(preprocessor.feature_extractors, dict)

    def test_preprocess_for_adaptation(self):
        """Test data preprocessing for adaptation"""
        preprocessor = DomainDataPreprocessor()

        source_data = np.random.randn(100, 10)
        target_data = np.random.randn(80, 10)

        source_processed, target_processed = preprocessor.preprocess_for_adaptation(
            source_data, target_data, "test_domain"
        )

        assert (
            source_processed.shape[1] == target_processed.shape[1]
        )  # Same number of features
        assert (
            source_processed.shape[0] == source_data.shape[0]
        )  # Same number of samples
        assert target_processed.shape[0] == target_data.shape[0]

    def test_extract_domain_invariant_features(self):
        """Test domain-invariant feature extraction"""
        preprocessor = DomainDataPreprocessor()

        data = np.random.randn(100, 20)
        invariant_features = preprocessor.extract_domain_invariant_features(
            data, "test_domain"
        )

        assert invariant_features.shape[0] == data.shape[0]  # Same number of samples
        assert invariant_features.shape[1] <= data.shape[1]  # Reduced features (PCA)

    @patch("matplotlib.pyplot.show")
    def test_visualize_domain_shift(self, mock_show):
        """Test domain shift visualization"""
        preprocessor = DomainDataPreprocessor()

        source_data = np.random.randn(50, 10)
        target_data = np.random.randn(50, 10) + 1  # Shifted distribution

        # Should not raise an exception
        preprocessor.visualize_domain_shift(
            source_data, target_data, ("Source", "Target")
        )

        mock_show.assert_called_once()

class TestAdaptationStrategySelector:
    """Test adaptation strategy selection"""

    def test_selector_initialization(self):
        """Test strategy selector initialization"""
        selector = AdaptationStrategySelector()

        assert isinstance(selector.strategy_rules, dict)
        assert "fine_tuning" in selector.strategy_rules
        assert "domain_adversarial" in selector.strategy_rules
        assert "few_shot" in selector.strategy_rules

    def test_select_strategy(self, sample_domain):
        """Test strategy selection"""
        selector = AdaptationStrategySelector()

        strategy = selector.select_strategy(sample_domain)

        assert strategy in ["fine_tuning", "domain_adversarial", "few_shot"]

    def test_get_strategy_recommendations(self, sample_domain):
        """Test strategy recommendations"""
        selector = AdaptationStrategySelector()

        recommendations = selector.get_strategy_recommendations(sample_domain)

        assert isinstance(recommendations, dict)
        for strategy in ["fine_tuning", "domain_adversarial", "few_shot"]:
            assert strategy in recommendations
            assert "suitability_score" in recommendations[strategy]
            assert "complexity" in recommendations[strategy]
            assert "estimated_performance" in recommendations[strategy]

    def test_calculate_strategy_score(self, sample_domain):
        """Test strategy score calculation"""
        selector = AdaptationStrategySelector()

        score = selector._calculate_strategy_score("fine_tuning", sample_domain)

        assert 0.0 <= score <= 1.0

# Integration tests
class TestTransferLearningIntegration:
    """Integration tests for transfer learning system"""

    def test_end_to_end_adaptation_workflow(self, sample_model, sample_data):
        """Test complete adaptation workflow"""
        # Initialize components
        engine = TransferLearningEngine()
        registry = GeologicalDomainRegistry()

        # Get a domain
        domain = registry.get_domain("yellowstone_usa")

        # Create configuration
        config = AdaptationConfig(
            source_domain=None,
            target_domain=domain,
            adaptation_method="fine_tuning",
            learning_rate=1e-3,
            num_epochs=5,  # Reduced for testing
            batch_size=16,
        )

        # Perform adaptation
        result = engine.adapt_to_domain(sample_model, domain, sample_data, config)

        # Verify complete workflow
        assert isinstance(result, AdaptationResult)
        assert result.adapted_model is not None
        assert len(result.validation_scores) > 0
        assert len(engine.adaptation_history) == 1

    def test_few_shot_learning_workflow(self, sample_model):
        """Test few-shot learning workflow"""
        engine = TransferLearningEngine()
        registry = GeologicalDomainRegistry()

        # Get a sparse data domain
        domain = registry.get_domain("kamchatka_russia")

        # Create small dataset for few-shot learning
        small_data = torch.randn(20, 10)  # Very small dataset
        small_targets = torch.randn(20, 1)
        dataset = TensorDataset(small_data, small_targets)
        dataloader = DataLoader(dataset, batch_size=5)

        config = AdaptationConfig(
            source_domain=None,
            target_domain=domain,
            adaptation_method="few_shot",
            few_shot_samples=5,
            num_epochs=3,
        )

        result = engine.few_shot_adapt(sample_model, domain, dataloader, config)

        assert isinstance(result, AdaptationResult)
        assert result.confidence_score <= 0.8  # Lower confidence for few-shot

if __name__ == "__main__":
    pytest.main([__file__])
