#!/usr/bin/env python3
"""
Basic test for scientific_knowledge_graph.py without heavy imports
Tests class instantiation and basic methods without CUDA dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_api_classes():
    """Test instantiation of API classes"""
    print("Testing API class instantiation...")
    
    try:
        # Import the module
        from scientific_knowledge_graph import (
            USGSDataAPI, 
            GeoScienceWorldAPI, 
            NASAEarthDataAPI,
            ASFHyp3API,
            EarthScopeIRISAPI
        )
        
        # Test basic instantiation
        usgs_api = USGSDataAPI()
        print("✓ USGSDataAPI instantiated successfully")
        
        geosci_api = GeoScienceWorldAPI()
        print("✓ GeoScienceWorldAPI instantiated successfully")
        
        nasa_api = NASAEarthDataAPI()
        print("✓ NASAEarthDataAPI instantiated successfully")
        
        asf_api = ASFHyp3API()
        print("✓ ASFHyp3API instantiated successfully")
        
        iris_api = EarthScopeIRISAPI()
        print("✓ EarthScopeIRISAPI instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error instantiating API classes: {e}")
        return False

def test_ontology_mappings():
    """Test ontology mapping methods"""
    print("\nTesting ontology mappings...")
    
    try:
        from scientific_knowledge_graph import USGSDataAPI
        
        api = USGSDataAPI()
        
        # Test sync version (fallback)
        import asyncio
        
        async def test_async():
            ontology = await api.get_ontology_mapping()
            return ontology
        
        # Run async test
        ontology = asyncio.run(test_async())
        
        if isinstance(ontology, dict) and len(ontology) > 0:
            print(f"✓ Ontology mapping returned {len(ontology)} entries")
            print(f"✓ Sample entries: {list(ontology.items())[:3]}")
            return True
        else:
            print("✗ Ontology mapping returned invalid data")
            return False
            
    except Exception as e:
        print(f"✗ Error testing ontology mappings: {e}")
        return False

def test_fallback_queries():
    """Test fallback query methods"""
    print("\nTesting fallback query methods...")
    
    try:
        from scientific_knowledge_graph import USGSDataAPI, NASAEarthDataAPI
        
        # Test USGS fallback
        usgs_api = USGSDataAPI()
        usgs_results = usgs_api._fallback_query("earthquake", limit=5)
        
        if isinstance(usgs_results, list) and len(usgs_results) > 0:
            print(f"✓ USGS fallback query returned {len(usgs_results)} results")
            if 'source' in usgs_results[0]:
                print(f"✓ USGS result source: {usgs_results[0]['source']}")
        else:
            print("✗ USGS fallback query failed")
            return False
            
        # Test NASA fallback
        nasa_api = NASAEarthDataAPI()
        nasa_results = nasa_api._fallback_query("satellite imagery", limit=3)
        
        if isinstance(nasa_results, list) and len(nasa_results) > 0:
            print(f"✓ NASA fallback query returned {len(nasa_results)} results")
            if 'source' in nasa_results[0]:
                print(f"✓ NASA result source: {nasa_results[0]['source']}")
        else:
            print("✗ NASA fallback query failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error testing fallback queries: {e}")
        return False

def test_helper_methods():
    """Test helper methods in API classes"""
    print("\nTesting helper methods...")
    
    try:
        from scientific_knowledge_graph import ASFHyp3API, EarthScopeIRISAPI
        
        # Test ASF helper methods
        asf_api = ASFHyp3API()
        platform = asf_api._get_platform_for_instrument("Sentinel-1A")
        resolution = asf_api._get_spatial_resolution("MODIS")
        
        print(f"✓ ASF platform mapping: Sentinel-1A -> {platform}")
        print(f"✓ ASF resolution mapping: MODIS -> {resolution}")
        
        # Test IRIS helper methods
        iris_api = EarthScopeIRISAPI()
        channel = iris_api._get_channel_code("broadband", 0)
        sample_rate = iris_api._get_sample_rate("broadband")
        
        print(f"✓ IRIS channel mapping: broadband -> {channel}")
        print(f"✓ IRIS sample rate: broadband -> {sample_rate}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing helper methods: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("TESTING SCIENTIFIC KNOWLEDGE GRAPH MODULE")
    print("=" * 60)
    
    tests = [
        test_api_classes,
        test_ontology_mappings,
        test_fallback_queries,
        test_helper_methods
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)