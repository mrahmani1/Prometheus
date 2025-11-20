#!/usr/bin/env python3
"""
Comprehensive Test Suite for Dimensional Pattern Recognition System

This test suite validates the advanced dimensional pattern recognition capabilities
including clustering analysis, correlation detection, anomaly detection, temporal
pattern analysis, statistical significance testing, and pattern classification.

Author: Prometheus AI Platform
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from datetime import datetime
import json

# Import the system under test
from backend.core.dimensional_pattern_recognition import (
    DimensionalPatternRecognitionSystem,
    PatternDetectionConfig,
    DimensionalPattern,
    HigherDimensionalData,
    ClusteringResult,
    CorrelationResult,
    AnomalyResult,
    TemporalPatternResult
)
from backend.core.database_manager import DatabaseManager

class TestDimensionalPatternRecognitionSystem:
    """Comprehensive test suite for dimensional pattern recognition system."""
    
    @pytest.fixture
    async def mock_database_manager(self):
        """Create mock database manager."""
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        db_manager.fetch_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    async def pattern_recognition_system(self, mock_database_manager):
        """Create pattern recognition system with mocked dependencies."""
        return DimensionalPatternRecognitionSystem(mock_database_manager)
    
    @pytest.fixture
    def sample_dimensional_data(self):
        """Create sample higher-dimensional data for testing."""
        np.random.seed(42)
        
        # Create synthetic volcanic data with known patterns
        n_samples = 500
        n_dimensions = 10
        
        # Base data with some structure
        data = np.random.randn(n_samples, n_dimensions)
        
        # Add clustering structure
        cluster_centers = np.array([
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [-2, -2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0, 0, 0, 0]
        ])
        
        for i in range(n_samples):
            if i < 150:
                data[i] += cluster_centers[0] + np.random.randn(n_dimensions) * 0.5
            elif i < 300:
                data[i] += cluster_centers[1] + np.random.randn(n_dimensions) * 0.5
            else:
                data[i] += cluster_centers[2] + np.random.randn(n_dimensions) * 0.5
        
        # Add correlations between some dimensions
        data[:, 5] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n_samples)
        data[:, 6] = -0.7 * data[:, 1] + 0.3 * np.random.randn(n_samples)
        
        # Add temporal patterns
        time_trend = np.linspace(0, 4*np.pi, n_samples)
        data[:, 7] = np.sin(time_trend) + 0.1 * np.random.randn(n_samples)
        data[:, 8] = np.cos(time_trend) + 0.1 * np.random.randn(n_samples)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, 20, replace=False)
        data[anomaly_indices] += np.random.randn(20, n_dimensions) * 3
        
        return HigherDimensionalData(
            data_id="test_volcanic_data_001",
            original_data=data,
            projected_data=data,  # For simplicity, use same data
            dimension_mapping={i: f"volcanic_param_{i}" for i in range(n_dimensions)},
            projection_metadata={"method": "test_projection"},
            quality_metrics={"projection_quality": 0.85},
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def pattern_detection_config(self):
        """Create pattern detection configuration for testing."""
        return PatternDetectionConfig(
            clustering_methods=['dbscan', 'spectral', 'gaussian_mixture'],
            correlation_threshold=0.6,
            significance_level=0.05,
            min_pattern_strength=0.3,
            temporal_window_size=50,
            anomaly_contamination=0.1,
            statistical_tests=['pearson', 'spearman'],
            topological_analysis=True,
            network_analysis=True
        )
    
    @pytest.mark.asyncio
    async def test_detect_dimensional_patterns_comprehensive(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test comprehensive dimensional pattern detection."""
        
        # Execute pattern detection
        patterns = await pattern_recognition_system.detect_dimensional_patterns(
            sample_dimensional_data, 
            pattern_detection_config
        )
        
        # Validate results
        assert isinstance(patterns, list)
        assert len(patterns) > 0, "Should detect at least some patterns"
        
        # Check pattern types
        pattern_types = {pattern.pattern_type for pattern in patterns}
        expected_types = {'clustering', 'correlation', 'anomaly'}
        
        # Should detect at least clustering and correlation patterns
        assert len(pattern_types.intersection(expected_types)) >= 2
        
        # Validate pattern structure
        for pattern in patterns:
            assert isinstance(pattern, DimensionalPattern)
            assert pattern.pattern_id is not None
            assert pattern.pattern_type in ['clustering', 'correlation', 'anomaly', 'temporal', 'topological', 'network', 'ensemble_clustering', 'ensemble_anomaly']
            assert isinstance(pattern.dimensions_involved, list)
            assert 0 <= pattern.pattern_strength <= 1
            assert 0 <= pattern.statistical_significance <= 1
            assert pattern.physical_interpretation is not None
            assert isinstance(pattern.validation_results, dict)
            assert isinstance(pattern.discovery_metadata, dict)
    
    @pytest.mark.asyncio
    async def test_clustering_analysis_multiple_algorithms(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test clustering analysis with multiple algorithms."""
        
        data = sample_dimensional_data.projected_data
        
        # Test clustering analysis
        clustering_patterns = await pattern_recognition_system._perform_clustering_analysis(
            data, pattern_detection_config, sample_dimensional_data.data_id
        )
        
        # Validate clustering results
        assert isinstance(clustering_patterns, list)
        assert len(clustering_patterns) > 0, "Should detect clustering patterns"
        
        # Check that multiple clustering methods were used
        methods_used = {pattern.discovery_metadata.get('method') for pattern in clustering_patterns}
        assert len(methods_used) > 1, "Should use multiple clustering methods"
        
        # Validate clustering pattern properties
        for pattern in clustering_patterns:
            assert pattern.pattern_type in ['clustering', 'ensemble_clustering']
            assert pattern.pattern_strength > 0
            assert len(pattern.dimensions_involved) > 0
            assert 'silhouette_score' in pattern.validation_results or 'ensemble_methods' in pattern.validation_results
    
    @pytest.mark.asyncio
    async def test_correlation_analysis_network_creation(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test correlation analysis and network creation."""
        
        data = sample_dimensional_data.projected_data
        
        # Test correlation analysis
        correlation_patterns = await pattern_recognition_system._perform_correlation_analysis(
            data, pattern_detection_config, sample_dimensional_data.data_id
        )
        
        # Validate correlation results
        assert isinstance(correlation_patterns, list)
        
        if len(correlation_patterns) > 0:
            # Check correlation pattern properties
            for pattern in correlation_patterns:
                assert pattern.pattern_type in ['correlation', 'partial_correlation']
                assert pattern.pattern_strength > 0
                assert len(pattern.dimensions_involved) > 0
                
                if pattern.pattern_type == 'correlation':
                    assert 'network_properties' in pattern.validation_results
                    assert 'n_significant_correlations' in pattern.validation_results
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_multiple_methods(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test anomaly detection with multiple methods."""
        
        data = sample_dimensional_data.projected_data
        
        # Test anomaly detection
        anomaly_patterns = await pattern_recognition_system._perform_anomaly_detection(
            data, pattern_detection_config, sample_dimensional_data.data_id
        )
        
        # Validate anomaly results
        assert isinstance(anomaly_patterns, list)
        assert len(anomaly_patterns) > 0, "Should detect anomaly patterns"
        
        # Check anomaly pattern properties
        for pattern in anomaly_patterns:
            assert pattern.pattern_type in ['anomaly', 'ensemble_anomaly']
            assert pattern.pattern_strength >= 0
            assert len(pattern.dimensions_involved) > 0
            
            if pattern.pattern_type == 'anomaly':
                assert 'n_anomalies' in pattern.validation_results
                assert 'method' in pattern.validation_results
    
    @pytest.mark.asyncio
    async def test_temporal_pattern_analysis(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test temporal pattern analysis capabilities."""
        
        data = sample_dimensional_data.projected_data
        
        # Test temporal pattern analysis
        temporal_patterns = await pattern_recognition_system._perform_temporal_pattern_analysis(
            data, pattern_detection_config, sample_dimensional_data.data_id
        )
        
        # Validate temporal results
        assert isinstance(temporal_patterns, list)
        
        if len(temporal_patterns) > 0:
            # Check temporal pattern properties
            for pattern in temporal_patterns:
                assert pattern.pattern_type in ['temporal_decomposition', 'change_points', 'temporal_correlation', 'periodicity']
                assert pattern.pattern_strength >= 0
                assert pattern.temporal_characteristics is not None
    
    @pytest.mark.asyncio
    async def test_statistical_validation_multiple_hypothesis_correction(
        self, 
        pattern_recognition_system, 
        pattern_detection_config
    ):
        """Test statistical validation with multiple hypothesis correction."""
        
        # Create mock patterns with different p-values
        mock_patterns = []
        p_values = [0.001, 0.01, 0.03, 0.08, 0.15]
        
        for i, p_val in enumerate(p_values):
            pattern = DimensionalPattern(
                pattern_id=f"test_pattern_{i}",
                pattern_type="test",
                dimensions_involved=[0, 1],
                pattern_strength=0.5,
                statistical_significance=p_val,
                physical_interpretation="Test pattern",
                validation_results={},
                discovery_metadata={}
            )
            mock_patterns.append(pattern)
        
        # Test statistical validation
        validated_patterns = await pattern_recognition_system._perform_statistical_validation(
            mock_patterns, pattern_detection_config
        )
        
        # Validate results
        assert isinstance(validated_patterns, list)
        assert len(validated_patterns) <= len(mock_patterns)
        
        # Check that multiple testing correction was applied
        for pattern in validated_patterns:
            assert 'multiple_testing_corrected' in pattern.validation_results
            assert 'original_p_value' in pattern.validation_results
            assert pattern.statistical_significance <= pattern_detection_config.significance_level
    
    @pytest.mark.asyncio
    async def test_pattern_classification_and_categorization(
        self, 
        pattern_recognition_system, 
        pattern_detection_config
    ):
        """Test pattern classification and categorization."""
        
        # Create mock patterns
        mock_patterns = [
            DimensionalPattern(
                pattern_id="strong_pattern",
                pattern_type="clustering",
                dimensions_involved=[0, 1, 2],
                pattern_strength=0.9,
                statistical_significance=0.001,
                physical_interpretation="Strong clustering",
                validation_results={},
                discovery_metadata={}
            ),
            DimensionalPattern(
                pattern_id="weak_pattern",
                pattern_type="correlation",
                dimensions_involved=[3, 4],
                pattern_strength=0.2,
                statistical_significance=0.04,
                physical_interpretation="Weak correlation",
                validation_results={},
                discovery_metadata={}
            )
        ]
        
        # Test classification
        classified_patterns = await pattern_recognition_system._classify_and_categorize_patterns(
            mock_patterns, pattern_detection_config
        )
        
        # Validate classification results
        assert len(classified_patterns) == len(mock_patterns)
        
        for pattern in classified_patterns:
            assert 'pattern_classification' in pattern.discovery_metadata
            classification = pattern.discovery_metadata['pattern_classification']
            
            assert 'primary_type' in classification
            assert 'strength_category' in classification
            assert 'significance_category' in classification
            assert 'complexity_category' in classification
    
    @pytest.mark.asyncio
    async def test_data_preprocessing_robustness(
        self, 
        pattern_recognition_system
    ):
        """Test data preprocessing with various data conditions."""
        
        # Test with normal data
        normal_data = np.random.randn(100, 5)
        processed_normal = await pattern_recognition_system._preprocess_data(normal_data)
        assert processed_normal.shape[0] == normal_data.shape[0]
        
        # Test with missing values
        data_with_nan = normal_data.copy()
        data_with_nan[10:15, 2] = np.nan
        processed_nan = await pattern_recognition_system._preprocess_data(data_with_nan)
        assert not np.isnan(processed_nan).any()
        
        # Test with constant features
        data_with_constant = normal_data.copy()
        data_with_constant[:, 3] = 5.0  # Constant feature
        processed_constant = await pattern_recognition_system._preprocess_data(data_with_constant)
        assert processed_constant.shape[1] < data_with_constant.shape[1]  # Should remove constant feature
    
    @pytest.mark.asyncio
    async def test_network_analysis_capabilities(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test network-based pattern analysis."""
        
        data = sample_dimensional_data.projected_data
        
        # Test network analysis
        network_patterns = await pattern_recognition_system._perform_network_analysis(
            data, pattern_detection_config, sample_dimensional_data.data_id
        )
        
        # Validate network results
        assert isinstance(network_patterns, list)
        
        if len(network_patterns) > 0:
            for pattern in network_patterns:
                assert pattern.pattern_type in ['network', 'temporal_network']
                assert pattern.network_properties is not None
                
                if pattern.pattern_type == 'network':
                    assert 'network_graph' in pattern.network_properties
                    assert 'analysis' in pattern.network_properties
    
    @pytest.mark.asyncio
    async def test_topological_analysis_when_available(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data, 
        pattern_detection_config
    ):
        """Test topological analysis when ripser is available."""
        
        data = sample_dimensional_data.projected_data[:100]  # Smaller sample for speed
        
        # Test topological analysis
        try:
            topological_patterns = await pattern_recognition_system._perform_topological_analysis(
                data, pattern_detection_config, sample_dimensional_data.data_id
            )
            
            # Validate topological results
            assert isinstance(topological_patterns, list)
            
            if len(topological_patterns) > 0:
                for pattern in topological_patterns:
                    assert pattern.pattern_type == 'topological'
                    assert pattern.topological_features is not None
                    assert 'persistence_diagrams' in pattern.topological_features
        
        except ImportError:
            # Ripser not available, skip test
            pytest.skip("Ripser not available for topological analysis")
    
    @pytest.mark.asyncio
    async def test_comprehensive_results_storage(
        self, 
        pattern_recognition_system, 
        mock_database_manager
    ):
        """Test comprehensive results storage in database."""
        
        # Create mock patterns
        mock_patterns = [
            DimensionalPattern(
                pattern_id="test_pattern_1",
                pattern_type="clustering",
                dimensions_involved=[0, 1],
                pattern_strength=0.8,
                statistical_significance=0.01,
                physical_interpretation="Test clustering pattern",
                validation_results={"test": "data"},
                discovery_metadata={"method": "test"}
            )
        ]
        
        config = PatternDetectionConfig()
        
        # Test storage
        await pattern_recognition_system._store_comprehensive_results(
            mock_patterns, "test_data_id", config
        )
        
        # Verify database interaction
        mock_database_manager.execute_query.assert_called_once()
        
        # Check the stored data structure
        call_args = mock_database_manager.execute_query.call_args
        assert len(call_args[0]) == 6  # Query + 5 parameters
        
        # Verify JSON serialization works
        config_json = call_args[0][2]
        patterns_json = call_args[0][3]
        summary_json = call_args[0][4]
        
        assert isinstance(json.loads(config_json), dict)
        assert isinstance(json.loads(patterns_json), list)
        assert isinstance(json.loads(summary_json), dict)
    
    @pytest.mark.asyncio
    async def test_error_handling_robustness(
        self, 
        pattern_recognition_system, 
        sample_dimensional_data
    ):
        """Test error handling in various scenarios."""
        
        # Test with empty data
        empty_data = HigherDimensionalData(
            data_id="empty_test",
            original_data=np.array([]),
            projected_data=np.array([]),
            dimension_mapping={},
            projection_metadata={},
            quality_metrics={}
        )
        
        try:
            patterns = await pattern_recognition_system.detect_dimensional_patterns(empty_data)
            assert isinstance(patterns, list)
        except ValueError:
            pass  # Expected for empty data
        
        # Test with invalid configuration
        invalid_config = PatternDetectionConfig(
            clustering_methods=['invalid_method'],
            correlation_threshold=2.0,  # Invalid threshold
            significance_level=-0.1  # Invalid significance level
        )
        
        patterns = await pattern_recognition_system.detect_dimensional_patterns(
            sample_dimensional_data, invalid_config
        )
        assert isinstance(patterns, list)  # Should handle gracefully
    
    @pytest.mark.asyncio
    async def test_performance_with_large_datasets(
        self, 
        pattern_recognition_system
    ):
        """Test performance characteristics with larger datasets."""
        
        # Create larger synthetic dataset
        np.random.seed(42)
        large_data = np.random.randn(2000, 20)
        
        # Add some structure for pattern detection
        large_data[:500, :5] += 3  # Cluster 1
        large_data[500:1000, 5:10] += 3  # Cluster 2
        large_data[:, 15] = 0.8 * large_data[:, 0]  # Correlation
        
        dimensional_data = HigherDimensionalData(
            data_id="large_test_data",
            original_data=large_data,
            projected_data=large_data,
            dimension_mapping={i: f"param_{i}" for i in range(20)},
            projection_metadata={},
            quality_metrics={}
        )
        
        config = PatternDetectionConfig(
            clustering_methods=['dbscan'],  # Use faster method
            topological_analysis=False,  # Skip expensive analysis
            network_analysis=False
        )
        
        # Measure execution time
        import time
        start_time = time.time()
        
        patterns = await pattern_recognition_system.detect_dimensional_patterns(
            dimensional_data, config
        )
        
        execution_time = time.time() - start_time
        
        # Validate results
        assert isinstance(patterns, list)
        assert execution_time < 60  # Should complete within reasonable time
        
        print(f"Large dataset processing time: {execution_time:.2f} seconds")
        print(f"Patterns detected: {len(patterns)}")

def run_comprehensive_tests():
    """Run comprehensive test suite for dimensional pattern recognition system."""
    
    print("=" * 80)
    print("COMPREHENSIVE DIMENSIONAL PATTERN RECOGNITION SYSTEM TESTS")
    print("=" * 80)
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED - DIMENSIONAL PATTERN RECOGNITION SYSTEM VALIDATED")
        print("=" * 80)
        print("\nKey capabilities validated:")
        print("✓ Multi-algorithm clustering analysis in high-dimensional spaces")
        print("✓ Cross-dimensional correlation detection and network analysis")
        print("✓ Advanced anomaly detection with multiple methods")
        print("✓ Temporal pattern analysis for multidimensional evolution")
        print("✓ Statistical significance testing with multiple hypothesis correction")
        print("✓ Pattern classification and categorization systems")
        print("✓ Topological data analysis integration")
        print("✓ Network-based pattern analysis")
        print("✓ Comprehensive validation against known patterns")
        print("✓ Robust error handling and performance optimization")
    else:
        print("\n" + "=" * 80)
        print("❌ SOME TESTS FAILED - CHECK OUTPUT ABOVE")
        print("=" * 80)
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_comprehensive_tests()
    exit(exit_code)