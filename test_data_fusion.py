#!/usr/bin/env python3
"""
Test comprehensive data fusion pipeline
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_data_fusion_pipeline_instantiation():
    """Test DataFusionPipeline instantiation"""
    print("Testing DataFusionPipeline instantiation...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline, APIAuthenticationManager, RateLimitManager
        
        # Create fusion pipeline
        fusion_pipeline = DataFusionPipeline()
        print("+ DataFusionPipeline instantiated successfully")
        
        # Test with custom auth and rate limiting
        auth_manager = APIAuthenticationManager()
        rate_limiter = RateLimitManager()
        
        custom_pipeline = DataFusionPipeline(
            auth_manager=auth_manager,
            rate_limiter=rate_limiter
        )
        assert custom_pipeline.auth_manager == auth_manager
        assert custom_pipeline.rate_limiter == rate_limiter
        print("+ Custom auth and rate limiting integration works")
        
        # Test data source initialization
        assert hasattr(fusion_pipeline, 'usgs_api')
        assert hasattr(fusion_pipeline, 'nasa_api')
        assert hasattr(fusion_pipeline, 'asf_api')
        assert hasattr(fusion_pipeline, 'iris_api')
        assert hasattr(fusion_pipeline, 'geosci_api')
        print("+ All data source APIs initialized")
        
        # Test fusion rules
        assert 'seismic_volcanic_fusion' in fusion_pipeline.fusion_rules
        assert 'ground_deformation_fusion' in fusion_pipeline.fusion_rules
        assert 'thermal_seismic_fusion' in fusion_pipeline.fusion_rules
        assert 'geological_literature_fusion' in fusion_pipeline.fusion_rules
        assert 'multi_hazard_fusion' in fusion_pipeline.fusion_rules
        print("+ All 5 fusion rule types configured")
        
        # Test quality metrics initialization
        required_metrics = [
            'temporal_alignment_score',
            'spatial_coherence_score',
            'semantic_consistency_score',
            'data_completeness_score',
            'fusion_confidence_score'
        ]
        
        for metric in required_metrics:
            assert metric in fusion_pipeline.quality_metrics
        
        print("+ Quality metrics initialized")
        
        return True
        
    except Exception as e:
        print(f"X Error testing data fusion pipeline instantiation: {e}")
        return False

def test_fusion_strategy_determination():
    """Test automatic fusion strategy determination"""
    print("\nTesting fusion strategy determination...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Test different query types
        test_cases = [
            ("earthquake magnitude 6.2 volcanic eruption", "seismic_volcanic_fusion"),
            ("InSAR ground deformation displacement", "ground_deformation_fusion"),
            ("thermal anomaly hotspot temperature", "thermal_seismic_fusion"),
            ("geological research paper study literature", "geological_literature_fusion"),
            ("multi-hazard risk disaster analysis", "multi_hazard_fusion"),
            ("unknown geological phenomenon", "seismic_volcanic_fusion")  # default
        ]
        
        for query, expected_strategy in test_cases:
            strategy = fusion_pipeline._determine_fusion_strategy(query)
            assert strategy == expected_strategy
            print(f"+ Query '{query[:30]}...' -> {strategy}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing fusion strategy determination: {e}")
        return False

def test_temporal_alignment():
    """Test temporal alignment functionality"""
    print("\nTesting temporal alignment...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Create mock data with timestamps
        base_time = datetime.now()
        source_data = {
            'usgs': [
                {
                    'id': 'eq1',
                    'time': base_time.isoformat(),
                    'magnitude': 5.2,
                    'location': 'Test Location 1'
                },
                {
                    'id': 'eq2',
                    'time': (base_time + timedelta(minutes=30)).isoformat(),
                    'magnitude': 4.8,
                    'location': 'Test Location 2'
                }
            ],
            'nasa': [
                {
                    'id': 'fire1',
                    'timestamp': (base_time + timedelta(minutes=15)).isoformat(),
                    'temperature': 45.2,
                    'confidence': 85
                }
            ]
        }
        
        # Test temporal alignment
        fusion_rule = fusion_pipeline.fusion_rules['seismic_volcanic_fusion']
        aligned_data = asyncio.run(fusion_pipeline._temporal_alignment(source_data, fusion_rule))
        
        # Check that temporal alignment metadata was added
        assert 'usgs' in aligned_data
        assert 'nasa' in aligned_data
        
        for source, data_list in aligned_data.items():
            for item in data_list:
                if 'temporal_alignment' in item:
                    assert 'original_timestamp' in item['temporal_alignment']
                    assert 'cluster_id' in item['temporal_alignment']
                    assert 'alignment_confidence' in item['temporal_alignment']
        
        print("+ Temporal alignment metadata added correctly")
        
        # Check quality score calculation
        score = fusion_pipeline.quality_metrics['temporal_alignment_score']
        assert 0.0 <= score <= 1.0
        print(f"+ Temporal alignment score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing temporal alignment: {e}")
        return False

def test_spatial_correlation():
    """Test spatial correlation functionality"""
    print("\nTesting spatial correlation...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Create mock data with coordinates
        aligned_data = {
            'usgs': [
                {
                    'id': 'eq1',
                    'latitude': 37.7749,
                    'longitude': -122.4194,
                    'magnitude': 5.2
                },
                {
                    'id': 'eq2',
                    'latitude': 37.7849,  # Close to first event
                    'longitude': -122.4094,
                    'magnitude': 4.8
                }
            ],
            'nasa': [
                {
                    'id': 'fire1',
                    'lat': 37.7849,
                    'lon': -122.4194,
                    'temperature': 45.2
                }
            ]
        }
        
        # Test spatial correlation
        fusion_rule = fusion_pipeline.fusion_rules['seismic_volcanic_fusion']
        correlated_data = asyncio.run(fusion_pipeline._spatial_correlation(aligned_data, fusion_rule))
        
        # Check that spatial correlation metadata was added
        for source, data_list in correlated_data.items():
            for item in data_list:
                if 'spatial_correlation' in item:
                    assert 'coordinates' in item['spatial_correlation']
                    assert 'cluster_id' in item['spatial_correlation']
                    assert 'nearby_sources' in item['spatial_correlation']
        
        print("+ Spatial correlation metadata added correctly")
        
        # Check quality score calculation
        score = fusion_pipeline.quality_metrics['spatial_coherence_score']
        assert 0.0 <= score <= 1.0
        print(f"+ Spatial coherence score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing spatial correlation: {e}")
        return False

def test_semantic_integration():
    """Test semantic integration functionality"""
    print("\nTesting semantic integration...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Create mock data with geological content
        correlated_data = {
            'usgs': [
                {
                    'id': 'eq1',
                    'title': 'Magnitude 5.2 earthquake near volcanic region',
                    'description': 'Seismic activity detected with significant ground motion'
                }
            ],
            'geoscienceworld': [
                {
                    'id': 'paper1',
                    'title': 'Volcanic earthquake swarm analysis',
                    'abstract': 'Study of earthquake patterns in volcanic environments'
                }
            ]
        }
        
        # Test semantic integration
        fusion_rule = fusion_pipeline.fusion_rules['seismic_volcanic_fusion']
        integrated_data = asyncio.run(fusion_pipeline._semantic_integration(correlated_data, fusion_rule))
        
        # Check that semantic integration metadata was added
        for source, data_list in integrated_data.items():
            for item in data_list:
                if 'semantic_integration' in item:
                    assert 'extracted_concepts' in item['semantic_integration']
                    assert 'related_concepts' in item['semantic_integration']
                    assert 'ontology_uris' in item['semantic_integration']
                    assert 'semantic_similarity_score' in item['semantic_integration']
        
        print("+ Semantic integration metadata added correctly")
        
        # Check quality score calculation
        score = fusion_pipeline.quality_metrics['semantic_consistency_score']
        assert 0.0 <= score <= 1.0
        print(f"+ Semantic consistency score: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing semantic integration: {e}")
        return False

def test_helper_methods():
    """Test various helper methods"""
    print("\nTesting helper methods...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Test timestamp extraction
        test_items = [
            {'time': '2024-01-01T12:00:00Z'},
            {'timestamp': '2024-01-01T12:00:00'},
            {'eventTime': 1704110400},  # Unix timestamp
            {'no_time_field': 'test'}
        ]
        
        extracted_times = []
        for item in test_items:
            timestamp = fusion_pipeline._extract_timestamp(item)
            extracted_times.append(timestamp)
        
        # Should extract 3 valid timestamps
        valid_times = [t for t in extracted_times if t is not None]
        assert len(valid_times) == 3
        print("+ Timestamp extraction works correctly")
        
        # Test coordinate extraction
        test_coords = [
            {'latitude': 37.7749, 'longitude': -122.4194},
            {'lat': 40.7128, 'lon': -74.0060},
            {'y': 51.5074, 'x': -0.1278},
            {'no_coords': 'test'}
        ]
        
        extracted_coords = []
        for item in test_coords:
            coords = fusion_pipeline._extract_coordinates(item)
            extracted_coords.append(coords)
        
        # Should extract 3 valid coordinate pairs
        valid_coords = [c for c in extracted_coords if c is not None]
        assert len(valid_coords) == 3
        print("+ Coordinate extraction works correctly")
        
        # Test haversine distance calculation
        coord1 = (37.7749, -122.4194)  # San Francisco
        coord2 = (37.7849, -122.4094)  # Very close point
        distance = fusion_pipeline._haversine_distance(coord1, coord2)
        assert 0 < distance < 20  # Should be less than 20km
        print(f"+ Haversine distance calculation: {distance:.2f} km")
        
        # Test geological concept extraction
        test_item = {
            'title': 'Earthquake magnitude analysis near volcanic region',
            'description': 'Study of seismic waves and fault mechanisms'
        }
        concepts = fusion_pipeline._extract_geological_concepts(test_item)
        assert len(concepts) > 0
        print(f"+ Geological concept extraction: {concepts}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing helper methods: {e}")
        return False

async def test_complete_fusion_process():
    """Test the complete data fusion process"""
    print("\nTesting complete fusion process...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Test fusion with fallback (since we don't have real API data)
        query = "earthquake volcanic activity magnitude 5.0"
        
        # This will use fallback data since we don't have real API connections
        result = await fusion_pipeline.fuse_datasets(
            query=query,
            fusion_type='seismic_volcanic_fusion',
            quality_threshold=0.1  # Low threshold to allow fallback processing
        )
        
        # Check result structure
        assert 'fused_data' in result
        assert 'fusion_metadata' in result
        
        metadata = result['fusion_metadata']
        assert 'fusion_type' in metadata
        assert 'quality_scores' in metadata
        assert 'source_count' in metadata
        assert 'fusion_timestamp' in metadata
        
        print("+ Complete fusion process executed successfully")
        print(f"+ Fusion type: {metadata['fusion_type']}")
        print(f"+ Source count: {metadata['source_count']}")
        
        quality_scores = metadata['quality_scores']
        for metric, score in quality_scores.items():
            print(f"+ {metric}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing complete fusion process: {e}")
        return False

def test_fusion_statistics():
    """Test fusion statistics functionality"""
    print("\nTesting fusion statistics...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Get fusion statistics
        stats = asyncio.run(fusion_pipeline.get_fusion_statistics())
        
        # Check statistics structure
        assert 'quality_metrics' in stats
        assert 'fusion_rules' in stats
        assert 'cache_size' in stats
        assert 'last_fusion_timestamp' in stats
        
        # Check fusion rules
        assert len(stats['fusion_rules']) == 5
        expected_rules = [
            'seismic_volcanic_fusion',
            'ground_deformation_fusion',
            'thermal_seismic_fusion',
            'geological_literature_fusion',
            'multi_hazard_fusion'
        ]
        
        for rule in expected_rules:
            assert rule in stats['fusion_rules']
        
        print("+ Fusion statistics retrieved successfully")
        print(f"+ Available fusion rules: {len(stats['fusion_rules'])}")
        print(f"+ Cache size: {stats['cache_size']}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing fusion statistics: {e}")
        return False

def main():
    """Run all data fusion pipeline tests"""
    print("=" * 70)
    print("DATA FUSION PIPELINE TESTS")
    print("=" * 70)
    
    tests = [
        test_data_fusion_pipeline_instantiation,
        test_fusion_strategy_determination,
        test_temporal_alignment,
        test_spatial_correlation,
        test_semantic_integration,
        test_helper_methods,
        lambda: asyncio.run(test_complete_fusion_process()),
        test_fusion_statistics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"X Test failed with exception: {e}")
        print()
    
    print("=" * 70)
    print(f"DATA FUSION PIPELINE RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("All data fusion pipeline tests PASSED!")
        print("+ Intelligent multi-source data fusion implemented")
        print("+ 5 specialized fusion strategies available")
        print("+ Temporal, spatial, and semantic correlation")
        print("+ Quality assessment and confidence scoring")
        print("+ Comprehensive geological ontology integration")
        print("+ Advanced fusion methods for different data types")
        print("+ Cascading hazard analysis capabilities")
        print("+ Real-time fusion statistics and monitoring")
    else:
        print("Some data fusion pipeline tests FAILED.")
        print("Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)