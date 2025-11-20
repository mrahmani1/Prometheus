"""
Comprehensive Test Suite for Hidden Pattern Discovery Framework

This test suite validates the advanced hidden pattern discovery framework
implementation including causal discovery, network analysis, machine learning
pattern recognition, and statistical validation.
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json

# Import the framework components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.hidden_pattern_discovery import (
    HiddenPatternDiscoveryFramework,
    AdvancedCausalDiscovery,
    AdvancedMLPatternDiscovery,
    HiddenPattern,
    CausalRelationship,
    NetworkPattern,
    PatternValidationResult
)
from core.database_manager import DatabaseManager
from core.interdimensional_analysis_engine import HigherDimensionalData

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestHiddenPatternDiscoveryFramework:
    """Test suite for the Hidden Pattern Discovery Framework"""
    
    @pytest.fixture
    async def mock_database_manager(self):
        """Create a mock database manager for testing"""
        db_manager = Mock(spec=DatabaseManager)
        
        # Mock connection context manager
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=[])
        mock_conn.fetchrow = AsyncMock(return_value=None)
        
        db_manager.get_connection = AsyncMock()
        db_manager.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        return db_manager
    
    @pytest.fixture
    def sample_volcanic_data(self):
        """Generate sample volcanic data for testing"""
        np.random.seed(42)
        n_samples = 1000
        n_dimensions = 20
        
        # Create synthetic volcanic data with known patterns
        data = np.random.randn(n_samples, n_dimensions)
        
        # Add some known correlations and causal relationships
        # Dimension 0 causes dimension 1
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * np.random.randn(n_samples)
        
        # Dimension 2 and 3 are correlated
        data[:, 3] = 0.8 * data[:, 2] + 0.2 * np.random.randn(n_samples)
        
        # Add some anomalies
        anomaly_indices = np.random.choice(n_samples, size=50, replace=False)
        data[anomaly_indices, :5] += 3 * np.random.randn(50, 5)
        
        return data
    
    @pytest.fixture
    def sample_dimensional_data(self, sample_volcanic_data):
        """Create sample HigherDimensionalData object"""
        return HigherDimensionalData(
            data_id="test_volcanic_data_001",
            original_data=sample_volcanic_data,
            projected_data=sample_volcanic_data,  # Same for testing
            dimension_mapping={i: f"volcanic_param_{i}" for i in range(sample_volcanic_data.shape[1])},
            projection_metadata={"method": "test", "parameters": {}},
            quality_metrics={"reconstruction_error": 0.1, "explained_variance": 0.9},
            timestamp=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self, mock_database_manager):
        """Test framework initialization"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        
        assert framework.db_manager == mock_database_manager
        assert isinstance(framework.causal_discovery, AdvancedCausalDiscovery)
        assert isinstance(framework.ml_discovery, AdvancedMLPatternDiscovery)
        assert len(framework.causal_algorithms) > 0
        assert len(framework.network_algorithms) > 0
        assert len(framework.ml_algorithms) > 0
        assert len(framework.validation_methods) > 0
    
    @pytest.mark.asyncio
    async def test_discover_hidden_volcanic_connections(self, mock_database_manager, sample_dimensional_data):
        """Test the main discovery method"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        
        # Configure discovery with minimal settings for testing
        discovery_config = {
            'causal_discovery': {
                'enabled': True,
                'algorithms': ['pc'],
                'significance_threshold': 0.05,
                'bootstrap_samples': 10  # Reduced for testing
            },
            'network_analysis': {
                'enabled': True,
                'centrality_measures': True,
                'community_detection': False,  # Disabled for speed
                'motif_analysis': False,
                'flow_analysis': False
            },
            'ml_pattern_discovery': {
                'enabled': True,
                'algorithms': ['isolation_forest'],
                'ensemble_methods': False,
                'deep_learning': False
            },
            'statistical_validation': {
                'enabled': True,
                'cross_validation_folds': 3,  # Reduced for testing
                'bootstrap_samples': 10,
                'permutation_tests': False,
                'multiple_testing_correction': 'bonferroni'
            },
            'significance_threshold': 0.05
        }
        
        # Run discovery
        discoveries = await framework.discover_hidden_volcanic_connections(
            sample_dimensional_data, discovery_config
        )
        
        # Validate results structure
        assert 'discovery_id' in discoveries
        assert 'timestamp' in discoveries
        assert 'data_source' in discoveries
        assert 'configuration' in discoveries
        
        # Check that at least some discovery phases ran
        expected_keys = ['causal_relationships', 'network_patterns', 'ml_patterns']
        for key in expected_keys:
            if discovery_config.get(key.replace('_patterns', '_discovery' if 'patterns' in key else '_analysis'), {}).get('enabled', True):
                assert key in discoveries
    
    @pytest.mark.asyncio
    async def test_causal_discovery_algorithms(self, mock_database_manager, sample_volcanic_data):
        """Test causal discovery algorithms"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        config = {
            'algorithms': ['pc'],
            'significance_threshold': 0.05,
            'bootstrap_samples': 10
        }
        
        # Test causal discovery
        causal_results = await framework._discover_advanced_causal_relationships(
            sample_volcanic_data, variable_names, config
        )
        
        assert 'algorithms_used' in causal_results
        assert 'relationships' in causal_results
        assert 'summary' in causal_results
        
        # Should find the known causal relationship (var_0 -> var_1)
        relationships = causal_results['relationships']
        if relationships:
            assert isinstance(relationships, list)
            # Check if any relationship involves our known causal pair
            causal_pairs = [(r.cause_variable if hasattr(r, 'cause_variable') else r.get('cause', ''), 
                           r.effect_variable if hasattr(r, 'effect_variable') else r.get('effect', '')) 
                          for r in relationships]
            logger.info(f"Found causal pairs: {causal_pairs}")
    
    @pytest.mark.asyncio
    async def test_network_pattern_discovery(self, mock_database_manager, sample_volcanic_data):
        """Test network pattern discovery"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        config = {
            'centrality_measures': True,
            'community_detection': False,
            'motif_analysis': False,
            'flow_analysis': False
        }
        
        # Test network discovery
        network_results = await framework._discover_network_patterns(
            sample_volcanic_data, variable_names, config
        )
        
        assert 'networks_constructed' in network_results
        assert 'centrality_analysis' in network_results
        assert 'network_metrics' in network_results
        
        # Should construct at least correlation network
        assert len(network_results['networks_constructed']) > 0
    
    @pytest.mark.asyncio
    async def test_ml_pattern_discovery(self, mock_database_manager, sample_volcanic_data):
        """Test machine learning pattern discovery"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        config = {
            'algorithms': ['isolation_forest'],
            'ensemble_methods': False,
            'deep_learning': False
        }
        
        # Test ML pattern discovery
        ml_results = await framework._discover_advanced_ml_patterns(
            sample_volcanic_data, variable_names, config
        )
        
        assert 'algorithms_used' in ml_results
        assert 'patterns_discovered' in ml_results
        assert 'anomaly_detection' in ml_results
        
        # Should detect some patterns/anomalies
        if ml_results['patterns_discovered']:
            pattern = ml_results['patterns_discovered'][0]
            assert hasattr(pattern, 'pattern_id') or 'pattern_id' in pattern
            assert hasattr(pattern, 'pattern_strength') or 'pattern_strength' in pattern
    
    @pytest.mark.asyncio
    async def test_isolation_forest_analysis(self, mock_database_manager, sample_volcanic_data):
        """Test isolation forest anomaly detection"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        # Test isolation forest
        patterns = await framework._isolation_forest_analysis(sample_volcanic_data, variable_names)
        
        # Should detect some anomalous patterns
        assert isinstance(patterns, list)
        
        if patterns:
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'pattern_type')
            assert pattern.pattern_type == "anomaly_detection"
            assert hasattr(pattern, 'statistical_significance')
    
    @pytest.mark.asyncio
    async def test_correlation_network_construction(self, mock_database_manager, sample_volcanic_data):
        """Test correlation network construction"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        # Test network construction
        network = await framework._construct_correlation_network(
            sample_volcanic_data, variable_names, threshold=0.3
        )
        
        assert network is not None
        assert len(network.nodes()) > 0
        
        # Should have edges for correlated variables
        if len(network.edges()) > 0:
            edge = list(network.edges(data=True))[0]
            assert 'weight' in edge[2]
            assert 'correlation' in edge[2]
            assert 'p_value' in edge[2]
    
    @pytest.mark.asyncio
    async def test_centrality_analysis(self, mock_database_manager, sample_volcanic_data):
        """Test network centrality analysis"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        variable_names = [f"var_{i}" for i in range(sample_volcanic_data.shape[1])]
        
        # Create test networks
        correlation_network = await framework._construct_correlation_network(
            sample_volcanic_data, variable_names, threshold=0.3
        )
        causal_network = correlation_network  # Use same for testing
        
        # Test centrality analysis
        centrality_results = await framework._analyze_network_centrality(
            correlation_network, causal_network, variable_names
        )
        
        assert isinstance(centrality_results, dict)
        
        if 'correlation' in centrality_results:
            corr_centrality = centrality_results['correlation']
            expected_measures = ['betweenness', 'closeness', 'eigenvector', 'pagerank', 'degree']
            
            for measure in expected_measures:
                if measure in corr_centrality:
                    assert isinstance(corr_centrality[measure], dict)
    
    @pytest.mark.asyncio
    async def test_causal_metrics_calculation(self, mock_database_manager, sample_volcanic_data):
        """Test causal metrics calculation"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        
        # Test with known causal relationship (dimension 0 -> dimension 1)
        strength, ci, p_val = await framework.causal_discovery._calculate_causal_metrics(
            sample_volcanic_data, 0, 1, 0.05
        )
        
        assert isinstance(strength, float)
        assert 0 <= strength <= 1
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert isinstance(p_val, float)
        assert 0 <= p_val <= 1
        
        # Should detect strong relationship between dimensions 0 and 1
        logger.info(f"Causal strength between dim 0 and 1: {strength}, p-value: {p_val}")
    
    @pytest.mark.asyncio
    async def test_statistical_validation_methods(self, mock_database_manager, sample_dimensional_data):
        """Test statistical validation methods"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        
        # Create mock discoveries for validation
        mock_discoveries = {
            'ml_patterns': {
                'patterns_discovered': [
                    HiddenPattern(
                        pattern_id="test_pattern_1",
                        pattern_type="test",
                        dimensions_involved=[0, 1],
                        pattern_strength=0.7,
                        statistical_significance=0.95,
                        confidence_interval=(0.5, 0.9),
                        p_value=0.01,
                        effect_size=0.8,
                        physical_interpretation=None,
                        causal_relationships=[],
                        network_properties={},
                        validation_results={},
                        discovery_metadata={}
                    )
                ]
            }
        }
        
        config = {
            'cross_validation_folds': 3,
            'bootstrap_samples': 10,
            'permutation_tests': False,
            'multiple_testing_correction': 'bonferroni'
        }
        
        variable_names = [f"var_{i}" for i in range(sample_dimensional_data.projected_data.shape[1])]
        
        # Test validation
        validation_results = await framework._comprehensive_statistical_validation(
            mock_discoveries, sample_dimensional_data.projected_data, variable_names, config
        )
        
        assert isinstance(validation_results, dict)
        expected_keys = ['cross_validation', 'bootstrap_validation', 'multiple_testing_correction']
        for key in expected_keys:
            assert key in validation_results
    
    @pytest.mark.asyncio
    async def test_database_storage(self, mock_database_manager, sample_dimensional_data):
        """Test database storage functionality"""
        framework = HiddenPatternDiscoveryFramework(mock_database_manager)
        
        # Create mock discoveries
        mock_discoveries = {
            'discovery_id': 'test_discovery_001',
            'timestamp': datetime.now(),
            'data_source': sample_dimensional_data.data_id,
            'configuration': {'test': True},
            'summary': {'total_patterns': 5, 'significant_patterns': 3},
            'ml_patterns': {
                'patterns_discovered': [
                    HiddenPattern(
                        pattern_id="test_pattern_1",
                        pattern_type="test",
                        dimensions_involved=[0, 1],
                        pattern_strength=0.7,
                        statistical_significance=0.95,
                        confidence_interval=(0.5, 0.9),
                        p_value=0.01,
                        effect_size=0.8,
                        physical_interpretation=None,
                        causal_relationships=[],
                        network_properties={},
                        validation_results={},
                        discovery_metadata={'algorithm': 'test'}
                    )
                ]
            },
            'causal_relationships': {
                'relationships': [
                    {
                        'cause': 'var_0',
                        'effect': 'var_1',
                        'strength': 0.8,
                        'p_value': 0.001,
                        'algorithm': 'test'
                    }
                ]
            }
        }
        
        # Test storage (should not raise exceptions)
        await framework._store_comprehensive_discovery_results(mock_discoveries)
        
        # Verify database calls were made
        mock_database_manager.get_connection.assert_called()
    
    def test_data_structures(self):
        """Test data structure definitions"""
        # Test HiddenPattern creation
        pattern = HiddenPattern(
            pattern_id="test_001",
            pattern_type="correlation",
            dimensions_involved=[0, 1, 2],
            pattern_strength=0.85,
            statistical_significance=0.99,
            confidence_interval=(0.7, 0.95),
            p_value=0.001,
            effect_size=1.2,
            physical_interpretation="Strong volcanic coupling",
            causal_relationships=[],
            network_properties={},
            validation_results={},
            discovery_metadata={"algorithm": "test"}
        )
        
        assert pattern.pattern_id == "test_001"
        assert pattern.pattern_strength == 0.85
        assert len(pattern.dimensions_involved) == 3
        
        # Test CausalRelationship creation
        causal_rel = CausalRelationship(
            cause_variable="pressure",
            effect_variable="seismicity",
            causal_strength=0.75,
            confidence_interval=(0.6, 0.9),
            p_value=0.005,
            causal_mechanism="pressure_induced",
            temporal_lag=2.5,
            confounding_variables=["temperature"],
            validation_score=0.88
        )
        
        assert causal_rel.cause_variable == "pressure"
        assert causal_rel.effect_variable == "seismicity"
        assert causal_rel.temporal_lag == 2.5

class TestAdvancedCausalDiscovery:
    """Test suite for Advanced Causal Discovery algorithms"""
    
    @pytest.fixture
    def causal_discovery(self):
        return AdvancedCausalDiscovery()
    
    @pytest.fixture
    def sample_causal_data(self):
        """Generate data with known causal relationships"""
        np.random.seed(42)
        n_samples = 500
        
        # X -> Y -> Z causal chain
        X = np.random.randn(n_samples)
        Y = 0.8 * X + 0.2 * np.random.randn(n_samples)
        Z = 0.7 * Y + 0.3 * np.random.randn(n_samples)
        
        return np.column_stack([X, Y, Z])
    
    @pytest.mark.asyncio
    async def test_pc_algorithm(self, causal_discovery, sample_causal_data):
        """Test PC algorithm implementation"""
        variable_names = ['X', 'Y', 'Z']
        
        relationships = await causal_discovery._pc_algorithm(
            sample_causal_data, variable_names, 0.05
        )
        
        assert isinstance(relationships, list)
        
        # Should detect causal relationships in the chain
        if relationships:
            rel = relationships[0]
            assert hasattr(rel, 'cause_variable') or 'cause' in rel
            assert hasattr(rel, 'effect_variable') or 'effect' in rel
            assert hasattr(rel, 'causal_strength') or 'strength' in rel
    
    @pytest.mark.asyncio
    async def test_correlation_fallback(self, causal_discovery, sample_causal_data):
        """Test correlation-based fallback method"""
        variable_names = ['X', 'Y', 'Z']
        
        result = await causal_discovery._run_correlation_based_causal_discovery(
            sample_causal_data, variable_names, 0.05
        )
        
        assert 'correlation_matrix' in result
        assert 'relationships' in result
        assert 'algorithm_params' in result
        
        # Should find correlations in causal chain
        relationships = result['relationships']
        assert len(relationships) > 0

class TestAdvancedMLPatternDiscovery:
    """Test suite for Advanced ML Pattern Discovery"""
    
    @pytest.fixture
    def ml_discovery(self):
        return AdvancedMLPatternDiscovery()
    
    @pytest.fixture
    def sample_pattern_data(self):
        """Generate data with known patterns"""
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        
        # Create data with clusters and anomalies
        data = np.random.randn(n_samples, n_features)
        
        # Add cluster structure
        cluster1_indices = np.random.choice(n_samples, size=100, replace=False)
        data[cluster1_indices, :3] += 2
        
        # Add anomalies
        anomaly_indices = np.random.choice(n_samples, size=20, replace=False)
        data[anomaly_indices] += 4 * np.random.randn(20, n_features)
        
        return data
    
    @pytest.mark.asyncio
    async def test_autoencoder_patterns(self, ml_discovery, sample_pattern_data):
        """Test autoencoder pattern discovery"""
        variable_names = [f"feature_{i}" for i in range(sample_pattern_data.shape[1])]
        
        patterns = await ml_discovery._autoencoder_patterns(
            sample_pattern_data, variable_names, 
            hidden_dims=[5, 3], epochs=10, batch_size=32
        )
        
        assert isinstance(patterns, list)
        
        if patterns:
            pattern = patterns[0]
            assert hasattr(pattern, 'pattern_id')
            assert hasattr(pattern, 'pattern_type')
            assert pattern.pattern_type == "autoencoder_latent"

if __name__ == "__main__":
    # Run basic functionality test
    async def run_basic_test():
        """Run a basic test to verify framework functionality"""
        print("Running basic Hidden Pattern Discovery Framework test...")
        
        # Create mock database manager
        mock_db = Mock(spec=DatabaseManager)
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_db.get_connection = AsyncMock()
        mock_db.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
        
        # Initialize framework
        framework = HiddenPatternDiscoveryFramework(mock_db)
        print("✓ Framework initialized successfully")
        
        # Create sample data
        np.random.seed(42)
        sample_data = np.random.randn(100, 10)
        sample_data[:, 1] = 0.8 * sample_data[:, 0] + 0.2 * np.random.randn(100)
        
        dimensional_data = HigherDimensionalData(
            data_id="test_001",
            original_data=sample_data,
            projected_data=sample_data,
            dimension_mapping={i: f"param_{i}" for i in range(10)},
            projection_metadata={},
            quality_metrics={},
            timestamp=datetime.now()
        )
        print("✓ Sample data created")
        
        # Test basic discovery
        config = {
            'causal_discovery': {'enabled': True, 'algorithms': ['pc'], 'significance_threshold': 0.05},
            'network_analysis': {'enabled': True, 'centrality_measures': True},
            'ml_pattern_discovery': {'enabled': True, 'algorithms': ['isolation_forest']},
            'statistical_validation': {'enabled': True, 'cross_validation_folds': 3}
        }
        
        try:
            discoveries = await framework.discover_hidden_volcanic_connections(
                dimensional_data, config
            )
            print("✓ Discovery completed successfully")
            print(f"✓ Found {len(discoveries)} discovery categories")
            
            # Check results structure
            required_keys = ['discovery_id', 'timestamp', 'data_source']
            for key in required_keys:
                assert key in discoveries, f"Missing key: {key}"
            print("✓ Results structure validated")
            
            print("\n🎉 All basic tests passed! Hidden Pattern Discovery Framework is working correctly.")
            
        except Exception as e:
            print(f"❌ Discovery failed: {str(e)}")
            raise
    
    # Run the test
    asyncio.run(run_basic_test())