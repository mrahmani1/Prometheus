"""
Tests for Active Learning Engine

This module contains comprehensive tests for the active learning engine,
including unit tests for each strategy and integration tests.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# Add the core module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))

from active_learning_engine import (
    ActiveLearningConfig,
    ActiveLearningEngine,
    ActiveLearningEvaluator,
    DiversityBasedSampling,
    GeologicalSample,
    QueryByCommittee,
    UncertaintyBasedSampling,
    create_sample_geological_data,
    import,
)
from active_learning_engine import numpy as np
from active_learning_engine import pandas as pd
from active_learning_engine import torch.nn as nn

class TestGeologicalSample:
    """Test GeologicalSample data structure"""

    def test_geological_sample_creation(self):
        """Test creating a geological sample"""
        features = np.random.normal(0, 1, 10)
        location = (45.0, -120.0, 1000.0)

        sample = GeologicalSample(
            id="test_sample",
            features=features,
            location=location,
            sample_type="seismic",
            metadata={"test": True},
        )

        assert sample.id == "test_sample"
        assert np.array_equal(sample.features, features)
        assert sample.location == location
        assert sample.sample_type == "seismic"
        assert sample.metadata["test"] is True

class TestActiveLearningConfig:
    """Test ActiveLearningConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = ActiveLearningConfig()

        assert config.strategy == "uncertainty"
        assert config.batch_size == 10
        assert config.uncertainty_threshold == 0.1
        assert config.diversity_weight == 0.3
        assert config.committee_size == 5

    def test_custom_config(self):
        """Test custom configuration"""
        config = ActiveLearningConfig(
            strategy="diversity", batch_size=20, diversity_weight=0.5
        )

        assert config.strategy == "diversity"
        assert config.batch_size == 20
        assert config.diversity_weight == 0.5

class SimpleTestModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_size=10, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TestUncertaintyBasedSampling:
    """Test uncertainty-based sampling strategy"""

    def setup_method(self):
        """Set up test data"""
        self.samples = create_sample_geological_data(100)
        self.model = SimpleTestModel()
        self.strategy = UncertaintyBasedSampling()

    def test_select_samples(self):
        """Test sample selection"""
        labeled_data = self.samples[:20]
        unlabeled_pool = self.samples[20:]

        selected_indices = self.strategy.select_samples(
            unlabeled_pool, labeled_data, self.model, batch_size=10
        )

        assert len(selected_indices) == 10
        assert all(isinstance(idx, int) for idx in selected_indices)
        assert all(0 <= idx < len(unlabeled_pool) for idx in selected_indices)

    def test_compute_scores(self):
        """Test uncertainty score computation"""
        scores = self.strategy.compute_scores(self.samples[:10], self.model)

        assert len(scores) == 10
        assert all(isinstance(score, (int, float, np.number)) for score in scores)
        assert all(score >= 0 for score in scores)

    def test_physics_constraints(self):
        """Test physics constraint checking"""
        # Create a seismic sample with known constraints
        seismic_sample = GeologicalSample(
            id="seismic_test",
            features=np.random.normal(0, 1, 10),
            location=(45.0, -120.0, 1000.0),
            sample_type="seismic",
        )

        # Test with valid velocity
        valid_prediction = np.array([5.0])  # Valid seismic velocity
        violation = self.strategy._check_physics_constraints(
            seismic_sample, valid_prediction
        )
        assert violation == 0.0

        # Test with invalid velocity
        invalid_prediction = np.array([10.0])  # Invalid seismic velocity
        violation = self.strategy._check_physics_constraints(
            seismic_sample, invalid_prediction
        )
        assert violation > 0.0

class TestQueryByCommittee:
    """Test query-by-committee strategy"""

    def setup_method(self):
        """Set up test data"""
        self.samples = create_sample_geological_data(100)
        self.committee_models = [SimpleTestModel() for _ in range(3)]
        self.strategy = QueryByCommittee(self.committee_models)

    def test_select_samples(self):
        """Test committee-based sample selection"""
        labeled_data = self.samples[:20]
        unlabeled_pool = self.samples[20:]

        selected_indices = self.strategy.select_samples(
            unlabeled_pool, labeled_data, self.committee_models[0], batch_size=10
        )

        assert len(selected_indices) == 10
        assert all(isinstance(idx, int) for idx in selected_indices)

    def test_compute_scores(self):
        """Test committee disagreement computation"""
        scores = self.strategy.compute_scores(
            self.samples[:10], self.committee_models[0]
        )

        assert len(scores) == 10
        assert all(isinstance(score, (int, float, np.number)) for score in scores)

    def test_geological_disagreement(self):
        """Test geological domain-specific disagreement"""
        seismic_sample = GeologicalSample(
            id="seismic_test",
            features=np.random.normal(0, 1, 10),
            location=(45.0, -120.0, 1000.0),
            sample_type="seismic",
        )

        # Create mock predictions with disagreement
        predictions = np.array(
            [
                [1.0, 2.0, 3.0],  # Model 1 predictions
                [1.5, 2.5, 3.5],  # Model 2 predictions
                [0.5, 1.5, 2.5],  # Model 3 predictions
            ]
        )

        disagreement = self.strategy._geological_disagreement(
            seismic_sample, predictions
        )
        assert disagreement > 0.0

class TestDiversityBasedSampling:
    """Test diversity-based sampling strategy"""

    def setup_method(self):
        """Set up test data"""
        self.samples = create_sample_geological_data(100)
        self.model = SimpleTestModel()
        self.strategy = DiversityBasedSampling()

    def test_select_samples(self):
        """Test diversity-based sample selection"""
        labeled_data = self.samples[:20]
        unlabeled_pool = self.samples[20:]

        selected_indices = self.strategy.select_samples(
            unlabeled_pool, labeled_data, self.model, batch_size=10
        )

        assert len(selected_indices) == 10
        assert all(isinstance(idx, int) for idx in selected_indices)

    def test_kmeans_selection(self):
        """Test K-means based selection"""
        selected_indices = self.strategy._kmeans_selection(self.samples, batch_size=5)

        assert len(selected_indices) == 5
        assert len(set(selected_indices)) == 5  # All unique indices

    def test_geological_coverage_analysis(self):
        """Test geological coverage analysis"""
        coverage = self.strategy._analyze_geological_coverage(self.samples[:50])

        assert "sample_types" in coverage
        assert "depth_ranges" in coverage
        assert "geographic_regions" in coverage
        assert isinstance(coverage["sample_types"], dict)

    def test_coverage_improvement_calculation(self):
        """Test coverage improvement calculation"""
        coverage = self.strategy._analyze_geological_coverage(self.samples[:20])

        # Test with a new sample type
        new_sample = GeologicalSample(
            id="new_sample",
            features=np.random.normal(0, 1, 10),
            location=(60.0, -100.0, 2000.0),
            sample_type="new_type",
        )

        improvement = self.strategy._calculate_coverage_improvement(
            new_sample, coverage
        )
        assert improvement > 0.0

class TestActiveLearningEvaluator:
    """Test active learning evaluator"""

    def setup_method(self):
        """Set up test data"""
        self.samples = create_sample_geological_data(100)
        self.model = SimpleTestModel()
        self.evaluator = ActiveLearningEvaluator()

    def test_evaluate_selection(self):
        """Test evaluation of sample selection"""
        selected_samples = self.samples[:10]
        unlabeled_pool = self.samples[10:50]
        labeled_data = self.samples[50:]

        metrics = self.evaluator.evaluate_selection(
            selected_samples, unlabeled_pool, labeled_data, self.model, "uncertainty"
        )

        assert isinstance(metrics, dict)
        assert "sample_type_diversity" in metrics
        assert "diversity_score" in metrics
        assert "mean_uncertainty" in metrics

    def test_geological_coverage_evaluation(self):
        """Test geological coverage evaluation"""
        selected_samples = self.samples[:10]
        labeled_data = self.samples[10:30]

        metrics = self.evaluator._evaluate_geological_coverage(
            selected_samples, labeled_data
        )

        assert "sample_type_diversity" in metrics
        assert "geographic_spread" in metrics
        assert "depth_range_coverage" in metrics

    def test_diversity_evaluation(self):
        """Test diversity evaluation"""
        selected_samples = self.samples[:10]
        unlabeled_pool = self.samples[10:]

        metrics = self.evaluator._evaluate_diversity(selected_samples, unlabeled_pool)

        assert "diversity_score" in metrics
        assert "intra_diversity" in metrics

    def test_evaluation_summary(self):
        """Test evaluation summary generation"""
        # Add some mock evaluation history
        self.evaluator.metrics_history = [
            {
                "timestamp": np.datetime64("now"),
                "strategy": "uncertainty",
                "metrics": {"diversity_score": 0.8, "sample_type_diversity": 0.6},
                "sample_count": 10,
            },
            {
                "timestamp": np.datetime64("now"),
                "strategy": "diversity",
                "metrics": {"diversity_score": 0.9, "sample_type_diversity": 0.7},
                "sample_count": 10,
            },
        ]

        summary = self.evaluator.get_evaluation_summary()

        assert "total_evaluations" in summary
        assert "strategies_used" in summary
        assert "average_metrics" in summary
        assert "best_performing_strategy" in summary
        assert summary["total_evaluations"] == 2

class TestActiveLearningEngine:
    """Test main active learning engine"""

    def setup_method(self):
        """Set up test data"""
        self.samples = create_sample_geological_data(200)
        self.labeled_data = self.samples[:50]
        self.unlabeled_pool = self.samples[50:]
        self.model = SimpleTestModel()

        self.config = ActiveLearningConfig(strategy="uncertainty", batch_size=10)
        self.engine = ActiveLearningEngine(self.config)

    def test_engine_initialization(self):
        """Test engine initialization"""
        assert self.engine.config == self.config
        assert "uncertainty" in self.engine.strategies
        assert "diversity" in self.engine.strategies
        assert isinstance(self.engine.evaluator, ActiveLearningEvaluator)

    def test_select_optimal_samples_uncertainty(self):
        """Test optimal sample selection with uncertainty strategy"""
        result = self.engine.select_optimal_samples(
            self.unlabeled_pool, self.labeled_data, self.model
        )

        assert len(result.selected_samples) == 10
        assert result.strategy_used == "uncertainty"
        assert isinstance(result.evaluation_metrics, dict)
        assert isinstance(result.geological_coverage, dict)
        assert len(result.selection_scores) == 10

    def test_select_optimal_samples_diversity(self):
        """Test optimal sample selection with diversity strategy"""
        config = ActiveLearningConfig(strategy="diversity", batch_size=10)
        engine = ActiveLearningEngine(config)

        result = engine.select_optimal_samples(
            self.unlabeled_pool, self.labeled_data, self.model
        )

        assert len(result.selected_samples) == 10
        assert result.strategy_used == "diversity"

    def test_select_optimal_samples_hybrid(self):
        """Test optimal sample selection with hybrid strategy"""
        config = ActiveLearningConfig(
            strategy="hybrid", batch_size=10, diversity_weight=0.4
        )
        engine = ActiveLearningEngine(config)

        result = engine.select_optimal_samples(
            self.unlabeled_pool, self.labeled_data, self.model
        )

        assert len(result.selected_samples) == 10
        assert result.strategy_used == "hybrid"

    def test_select_optimal_samples_committee(self):
        """Test optimal sample selection with committee strategy"""
        committee_models = [SimpleTestModel() for _ in range(3)]
        config = ActiveLearningConfig(strategy="committee", batch_size=10)
        engine = ActiveLearningEngine(config)

        result = engine.select_optimal_samples(
            self.unlabeled_pool, self.labeled_data, self.model, committee_models
        )

        assert len(result.selected_samples) == 10
        assert result.strategy_used == "committee"

    def test_geological_coverage_calculation(self):
        """Test geological coverage calculation"""
        selected_samples = self.samples[:10]
        labeled_data = self.samples[10:30]

        coverage = self.engine._calculate_geological_coverage(
            selected_samples, labeled_data
        )

        assert "sample_type_coverage" in coverage
        assert "geographic_coverage" in coverage
        assert "depth_coverage" in coverage
        assert all(
            0 <= v <= 1
            for v in coverage.values()
            if v != coverage["geographic_coverage"]
        )

    def test_strategy_recommendations(self):
        """Test strategy recommendations"""
        recommendations = self.engine.get_strategy_recommendations(
            self.unlabeled_pool, self.labeled_data
        )

        assert "primary" in recommendations
        assert "reason" in recommendations
        assert recommendations["primary"] in ["uncertainty", "diversity", "hybrid"]

    def test_update_strategy_config(self):
        """Test updating strategy configuration"""
        new_config = ActiveLearningConfig(strategy="diversity", batch_size=20)
        self.engine.update_strategy_config(new_config)

        assert self.engine.config.strategy == "diversity"
        assert self.engine.config.batch_size == 20

    def test_invalid_strategy(self):
        """Test handling of invalid strategy"""
        config = ActiveLearningConfig(strategy="invalid_strategy")
        engine = ActiveLearningEngine(config)

        with pytest.raises(ValueError, match="Unknown strategy"):
            engine.select_optimal_samples(
                self.unlabeled_pool, self.labeled_data, self.model
            )

    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_export_evaluation_report(self, mock_json_dump, mock_open):
        """Test exporting evaluation report"""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        self.engine.export_evaluation_report("test_report.json")

        mock_open.assert_called_once_with("test_report.json", "w")
        mock_json_dump.assert_called_once()

class TestIntegration:
    """Integration tests for the complete active learning system"""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end active learning workflow"""
        # Create test data
        all_samples = create_sample_geological_data(500)
        labeled_data = all_samples[:100]
        unlabeled_pool = all_samples[100:]

        # Create model
        model = SimpleTestModel()

        # Test multiple strategies
        strategies = ["uncertainty", "diversity", "hybrid"]

        for strategy_name in strategies:
            config = ActiveLearningConfig(
                strategy=strategy_name, batch_size=20, diversity_weight=0.3
            )

            engine = ActiveLearningEngine(config)

            # Select samples
            result = engine.select_optimal_samples(unlabeled_pool, labeled_data, model)

            # Verify results
            assert len(result.selected_samples) == 20
            assert result.strategy_used == strategy_name
            assert isinstance(result.evaluation_metrics, dict)
            assert isinstance(result.geological_coverage, dict)

            # Verify evaluation metrics are reasonable
            assert "sample_type_diversity" in result.evaluation_metrics
            assert "diversity_score" in result.evaluation_metrics

            # Verify geological coverage
            assert "sample_type_coverage" in result.geological_coverage
            assert "geographic_coverage" in result.geological_coverage

    def test_iterative_learning(self):
        """Test iterative active learning process"""
        # Create initial dataset
        all_samples = create_sample_geological_data(1000)
        labeled_data = all_samples[:50]
        unlabeled_pool = all_samples[50:]

        model = SimpleTestModel()
        config = ActiveLearningConfig(strategy="hybrid", batch_size=10)
        engine = ActiveLearningEngine(config)

        # Simulate multiple iterations
        for iteration in range(5):
            result = engine.select_optimal_samples(unlabeled_pool, labeled_data, model)

            # Move selected samples to labeled data
            selected_samples = result.selected_samples
            labeled_data.extend(selected_samples)

            # Remove selected samples from unlabeled pool
            selected_ids = {sample.id for sample in selected_samples}
            unlabeled_pool = [
                sample for sample in unlabeled_pool if sample.id not in selected_ids
            ]

            # Verify progress
            assert len(labeled_data) == 50 + (iteration + 1) * 10
            assert len(result.selected_samples) == 10

        # Verify final state
        assert len(labeled_data) == 100
        assert len(unlabeled_pool) == 900

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
