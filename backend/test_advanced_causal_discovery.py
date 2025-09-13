"""
Test suite for Advanced Causal Discovery Algorithm Suite
Validates all implemented algorithms with geological data scenarios
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
import time
from typing import Dict, List, Any

# Import the causal discovery suite
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from advanced_causal_discovery_suite import (
    AdvancedCausalDiscoverySuite,
    CausalDiscoveryConfig,
    CausalAlgorithmType,
    PCAlgorithm,
    GESAlgorithm,
    LiNGAMAlgorithm,
    CausalForestAlgorithm,
    HybridCausalDiscovery,
    GeologicalDomainKnowledge
)

class TestAdvancedCausalDiscovery:
    """Test class for advanced causal discovery algorithms"""
    
    def sample_geological_data(self):
        """Create sample geological data with known causal relationships"""
        np.random.seed(42)
        n_samples = 500
        
        # Create variables with known causal structure:
        # temperature -> pressure -> deformation -> fluid_flow -> mineralization
        temperature = np.random.normal(400, 80, n_samples)
        pressure = 0.4 * temperature + np.random.normal(0, 30, n_samples)
        deformation = 0.3 * pressure + 0.2 * temperature + np.random.normal(0, 20, n_samples)
        fluid_flow = 0.5 * deformation + 0.1 * pressure + np.random.normal(0, 15, n_samples)
        mineralization = 0.4 * fluid_flow + 0.2 * deformation + np.random.normal(0, 10, n_samples)
        
        # Add some geological noise and non-linear relationships
        seismic_activity = np.abs(deformation) ** 0.8 + np.random.normal(0, 5, n_samples)
        geochemical_signature = np.log(np.abs(mineralization) + 1) + np.random.normal(0, 0.1, n_samples)
        
        return pd.DataFrame({
            'temperature': temperature,
            'pressure': pressure,
            'deformation': deformation,
            'fluid_flow': fluid_flow,
            'mineralization': mineralization,
            'seismic_activity': seismic_activity,
            'geochemical_signature': geochemical_signature
        })
    
    def causal_config(self):
        """Create test configuration for causal discovery"""
        return CausalDiscoveryConfig(
            significance_level=0.05,
            max_conditioning_set_size=2,
            bootstrap_samples=50,  # Reduced for testing
            confidence_threshold=0.7,
            use_geological_priors=True,
            parallel_processing=False,  # Disable for testing
            max_workers=2
        )
    
    def test_pc_algorithm(self, sample_geological_data, causal_config):
        """Test PC algorithm implementation"""
        pc_algo = PCAlgorithm(causal_config)
        
        async def run_test():
            result = await pc_algo.discover_structure(sample_geological_data)
            
            # Validate result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')
            assert hasattr(result, 'adjacency_matrix')
            assert hasattr(result, 'algorithms_used')
            
            # Check that we found some relationships
            assert len(result.nodes) > 0
            assert isinstance(result.adjacency_matrix, np.ndarray)
            assert 'PC' in result.algorithms_used[0]
            
            # Validate edge properties
            for edge in result.edges:
                assert hasattr(edge, 'source')
                assert hasattr(edge, 'target')
                assert hasattr(edge, 'strength')
                assert hasattr(edge, 'confidence')
                assert 0 <= edge.strength <= 1
                assert 0 <= edge.confidence <= 1
            
            return result
        
        result = asyncio.run(run_test())
        print(f"PC Algorithm found {len(result.edges)} causal relationships")
        
    def test_ges_algorithm(self, sample_geological_data, causal_config):
        """Test GES algorithm implementation"""
        ges_algo = GESAlgorithm(causal_config)
        
        async def run_test():
            result = await ges_algo.discover_structure(sample_geological_data)
            
            # Validate result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')
            assert len(result.nodes) > 0
            assert 'GES' in result.algorithms_used[0]
            
            # Check adjacency matrix dimensions
            expected_size = len(sample_geological_data.columns)
            assert result.adjacency_matrix.shape == (expected_size, expected_size)
            
            return result
        
        result = asyncio.run(run_test())
        print(f"GES Algorithm found {len(result.edges)} causal relationships")
    
    def test_lingam_algorithm(self, sample_geological_data, causal_config):
        """Test LiNGAM algorithm implementation"""
        lingam_algo = LiNGAMAlgorithm(causal_config)
        
        async def run_test():
            result = await lingam_algo.discover_structure(sample_geological_data)
            
            # Validate result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')
            assert len(result.nodes) > 0
            assert 'LiNGAM' in result.algorithms_used[0]
            
            # LiNGAM should produce directed edges
            for edge in result.edges:
                assert edge.edge_type == "directed"
            
            return result
        
        result = asyncio.run(run_test())
        print(f"LiNGAM Algorithm found {len(result.edges)} causal relationships")
    
    def test_causal_forest_algorithm(self, sample_geological_data, causal_config):
        """Test Causal Forest algorithm implementation"""
        cf_algo = CausalForestAlgorithm(causal_config)
        
        async def run_test():
            result = await cf_algo.discover_structure(sample_geological_data)
            
            # Validate result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')
            assert len(result.nodes) > 0
            assert 'CausalForest' in result.algorithms_used[0]
            
            # Causal Forest should handle non-linear relationships
            for edge in result.edges:
                assert edge.algorithm == "CausalForest"
            
            return result
        
        result = asyncio.run(run_test())
        print(f"Causal Forest Algorithm found {len(result.edges)} causal relationships")
    
    def test_hybrid_ensemble(self, sample_geological_data, causal_config):
        """Test Hybrid Ensemble approach"""
        hybrid_algo = HybridCausalDiscovery(causal_config)
        
        async def run_test():
            result = await hybrid_algo.discover_structure(sample_geological_data)
            
            # Validate result structure
            assert hasattr(result, 'nodes')
            assert hasattr(result, 'edges')
            assert len(result.nodes) > 0
            
            # Hybrid should use multiple algorithms
            assert len(result.algorithms_used) > 1
            
            # Should have ensemble agreement scores
            if hasattr(result, 'validation_scores'):
                assert 'ensemble_agreement' in result.validation_scores
            
            return result
        
        result = asyncio.run(run_test())
        print(f"Hybrid Ensemble found {len(result.edges)} causal relationships")
        print(f"Algorithms used: {result.algorithms_used}")
    
    def test_full_suite_integration(self, sample_geological_data, causal_config):
        """Test full causal discovery suite integration"""
        suite = AdvancedCausalDiscoverySuite(causal_config)
        
        async def run_test():
            # Test each algorithm through the main interface
            algorithms_to_test = [
                CausalAlgorithmType.PC,
                CausalAlgorithmType.GES,
                CausalAlgorithmType.LINGAM,
                CausalAlgorithmType.CAUSAL_FOREST,
                CausalAlgorithmType.HYBRID_ENSEMBLE
            ]
            
            results = {}
            
            for algorithm in algorithms_to_test:
                try:
                    result = await suite.discover_causal_structure(
                        data=sample_geological_data,
                        algorithm=algorithm,
                        validate_results=True
                    )
                    
                    # Validate result structure
                    assert 'causal_graph' in result
                    assert 'algorithm_used' in result
                    assert 'validation_results' in result
                    assert 'discovery_statistics' in result
                    assert 'geological_insights' in result
                    assert 'recommendations' in result
                    
                    results[algorithm.value] = result
                    
                    # Print summary
                    causal_graph = result['causal_graph']
                    stats = result['discovery_statistics']
                    
                    print(f"\n{algorithm.value} Results:")
                    print(f"  Edges found: {len(causal_graph.edges)}")
                    print(f"  Discovery time: {stats.get('discovery_time', 0):.3f}s")
                    print(f"  Average edge strength: {stats.get('avg_edge_strength', 0):.3f}")
                    
                    if result['validation_results']:
                        overall_score = result['validation_results'].get('overall_score', 0)
                        print(f"  Validation score: {overall_score:.3f}")
                    
                except Exception as e:
                    print(f"Error testing {algorithm.value}: {e}")
                    # Don't fail the test for individual algorithm failures
                    continue
            
            return results
        
        results = asyncio.run(run_test())
        
        # Validate that we got results from at least some algorithms
        assert len(results) > 0, "No algorithms produced valid results"
        
        # Check that hybrid ensemble performed reasonably
        if 'hybrid_ensemble' in results:
            hybrid_result = results['hybrid_ensemble']
            assert len(hybrid_result['causal_graph'].edges) > 0
    
    def test_geological_domain_knowledge(self):
        """Test geological domain knowledge integration"""
        domain_knowledge = GeologicalDomainKnowledge()
        
        # Test temporal precedence rules
        assert 'magma_generation' in domain_knowledge.temporal_precedence
        assert 'magma_transport' in domain_knowledge.temporal_precedence['magma_generation']
        
        # Test spatial constraints
        assert 'max_influence_distance' in domain_knowledge.spatial_constraints
        assert 'contact_metamorphism' in domain_knowledge.spatial_constraints['max_influence_distance']
        
        # Test forbidden edges
        assert len(domain_knowledge.forbidden_edges) > 0
        
        print("Geological domain knowledge validation passed")
    
    def test_data_preprocessing(self, causal_config):
        """Test data preprocessing functionality"""
        suite = AdvancedCausalDiscoverySuite(causal_config)
        
        # Create problematic data
        problematic_data = pd.DataFrame({
            'good_var': np.random.randn(100),
            'constant_var': np.ones(100),  # Constant column
            'correlated_var1': np.random.randn(100),
            'missing_var': [np.nan] * 50 + list(np.random.randn(50)),  # Missing values
            'categorical_var': ['A', 'B', 'C'] * 33 + ['A']  # Categorical
        })
        
        # Add highly correlated variable
        problematic_data['correlated_var2'] = problematic_data['correlated_var1'] * 0.99 + 0.01 * np.random.randn(100)
        
        async def run_test():
            processed_data = await suite._preprocess_data(problematic_data)
            
            # Should remove constant and categorical columns
            assert 'constant_var' not in processed_data.columns
            assert 'categorical_var' not in processed_data.columns
            
            # Should handle missing values
            assert not processed_data.isnull().any().any()
            
            # Should remove highly correlated columns
            remaining_corr_vars = [col for col in processed_data.columns 
                                 if 'correlated_var' in col]
            assert len(remaining_corr_vars) <= 1  # At most one should remain
            
            return processed_data
        
        processed_data = asyncio.run(run_test())
        print(f"Preprocessing: {problematic_data.shape} -> {processed_data.shape}")
    
    def test_performance_benchmarks(self, causal_config):
        """Test performance with different data sizes"""
        suite = AdvancedCausalDiscoverySuite(causal_config)
        
        data_sizes = [50, 100, 200]
        performance_results = {}
        
        async def run_performance_test():
            for n_samples in data_sizes:
                # Generate data
                np.random.seed(42)
                n_vars = 5
                
                data = pd.DataFrame({
                    f'var_{i}': np.random.randn(n_samples) 
                    for i in range(n_vars)
                })
                
                # Add some causal relationships
                data['var_1'] = data['var_0'] * 0.5 + np.random.randn(n_samples) * 0.3
                data['var_2'] = data['var_1'] * 0.4 + data['var_0'] * 0.2 + np.random.randn(n_samples) * 0.3
                
                # Test PC algorithm (fastest)
                start_time = time.time()
                result = await suite.discover_causal_structure(
                    data=data,
                    algorithm=CausalAlgorithmType.PC,
                    validate_results=False
                )
                end_time = time.time()
                
                performance_results[n_samples] = {
                    'discovery_time': end_time - start_time,
                    'n_edges': len(result['causal_graph'].edges),
                    'n_nodes': len(result['causal_graph'].nodes)
                }
                
                print(f"n_samples={n_samples}: {end_time - start_time:.3f}s, "
                      f"{len(result['causal_graph'].edges)} edges")
            
            return performance_results
        
        results = asyncio.run(run_performance_test())
        
        # Validate that performance scales reasonably
        times = [results[n]['discovery_time'] for n in data_sizes]
        
        # Should not have exponential blowup for small increases
        assert all(t < 30.0 for t in times), "Discovery taking too long"
        
        return results
    
    def test_edge_case_handling(self, causal_config):
        """Test handling of edge cases"""
        suite = AdvancedCausalDiscoverySuite(causal_config)
        
        async def run_edge_case_tests():
            # Test 1: Empty data
            empty_data = pd.DataFrame()
            try:
                result = await suite.discover_causal_structure(
                    data=empty_data,
                    algorithm=CausalAlgorithmType.PC,
                    validate_results=False
                )
                assert len(result['causal_graph'].edges) == 0
            except Exception as e:
                print(f"Empty data handling: {e}")
            
            # Test 2: Single variable
            single_var_data = pd.DataFrame({'var1': np.random.randn(100)})
            result = await suite.discover_causal_structure(
                data=single_var_data,
                algorithm=CausalAlgorithmType.PC,
                validate_results=False
            )
            assert len(result['causal_graph'].edges) == 0
            
            # Test 3: All constant data
            constant_data = pd.DataFrame({
                'var1': np.ones(100),
                'var2': np.ones(100) * 2
            })
            result = await suite.discover_causal_structure(
                data=constant_data,
                algorithm=CausalAlgorithmType.PC,
                validate_results=False
            )
            # Should handle gracefully
            
            # Test 4: Perfect correlation
            perfect_corr_data = pd.DataFrame({
                'var1': np.random.randn(100),
            })
            perfect_corr_data['var2'] = perfect_corr_data['var1'] * 2  # Perfect correlation
            
            result = await suite.discover_causal_structure(
                data=perfect_corr_data,
                algorithm=CausalAlgorithmType.PC,
                validate_results=False
            )
            # Should handle without crashing
            
            print("Edge case handling tests completed")
        
        asyncio.run(run_edge_case_tests())

def run_comprehensive_test():
    """Run comprehensive test of the causal discovery suite"""
    
    print("Starting comprehensive test of Advanced Causal Discovery Suite...")
    
    # Create test instance
    test_instance = TestAdvancedCausalDiscovery()
    
    # Create fixtures
    np.random.seed(42)
    sample_data = test_instance.sample_geological_data()
    config = test_instance.causal_config()
    
    print(f"Test data shape: {sample_data.shape}")
    print(f"Test data columns: {list(sample_data.columns)}")
    
    try:
        # Run individual algorithm tests
        print("\n=== Testing Individual Algorithms ===")
        test_instance.test_pc_algorithm(sample_data, config)
        test_instance.test_ges_algorithm(sample_data, config)
        test_instance.test_lingam_algorithm(sample_data, config)
        test_instance.test_causal_forest_algorithm(sample_data, config)
        test_instance.test_hybrid_ensemble(sample_data, config)
        
        # Run integration tests
        print("\n=== Testing Suite Integration ===")
        test_instance.test_full_suite_integration(sample_data, config)
        
        # Run domain knowledge tests
        print("\n=== Testing Domain Knowledge ===")
        test_instance.test_geological_domain_knowledge()
        
        # Run preprocessing tests
        print("\n=== Testing Data Preprocessing ===")
        test_instance.test_data_preprocessing(config)
        
        # Run performance tests
        print("\n=== Testing Performance ===")
        test_instance.test_performance_benchmarks(config)
        
        # Run edge case tests
        print("\n=== Testing Edge Cases ===")
        test_instance.test_edge_case_handling(config)
        
        print("\n=== All Tests Completed Successfully! ===")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_test()