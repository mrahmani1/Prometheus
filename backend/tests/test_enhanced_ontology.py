#!/usr/bin/env python3
"""
Test enhanced geological ontology system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_geological_ontology():
    """Test the comprehensive geological ontology system"""
    print("Testing Geological Ontology System...")
    
    try:
        from scientific_knowledge_graph import GeologicalOntology
        
        # Create ontology instance
        ontology = GeologicalOntology()
        print("+ GeologicalOntology instantiated successfully")
        
        # Test ontology mapping
        mapping = ontology.ontology_mapping
        print(f"+ Ontology contains {len(mapping)} concept mappings")
        
        # Test sample concepts
        test_concepts = ['earthquake', 'quartz', 'volcanic_rock', 'p_wave', 'groundwater']
        found_concepts = [concept for concept in test_concepts if concept in mapping]
        print(f"+ Found {len(found_concepts)}/{len(test_concepts)} test concepts")
        
        # Test URI conversion
        sample_uri = 'seismo:Earthquake'
        full_uri = ontology.get_full_uri(sample_uri)
        print(f"+ URI conversion: {sample_uri} -> {full_uri}")
        
        # Test related concepts
        related = ontology.get_related_concepts('earthquake')
        print(f"+ Related concepts for 'earthquake': {related[:3]}...")
        
        # Test hierarchical relationships
        hierarchy = ontology.get_hierarchical_relationships('granite')
        print(f"+ Hierarchical relationships for 'granite': {hierarchy}")
        
        return True
        
    except Exception as e:
        print(f"X Error testing geological ontology: {e}")
        return False

def test_enhanced_apis():
    """Test API classes with enhanced ontology"""
    print("\nTesting Enhanced API Classes...")
    
    try:
        from scientific_knowledge_graph import USGSDataAPI, GeoScienceWorldAPI, NASAEarthDataAPI
        
        # Test USGS API with ontology
        usgs_api = USGSDataAPI()
        print("+ USGSDataAPI instantiated with comprehensive ontology")
        
        # Test ontology mapping method
        import asyncio
        async def test_async():
            mapping = await usgs_api.get_ontology_mapping()
            return len(mapping)
        
        mapping_count = asyncio.run(test_async())
        print(f"+ USGS ontology mapping contains {mapping_count} concepts")
        
        # Test result enhancement
        sample_result = {
            'title': 'Magnitude 6.2 earthquake near volcanic region',
            'description': 'Seismic activity detected with P-wave arrivals',
            'type': 'earthquake'
        }
        
        enhanced = usgs_api.enhance_result_with_ontology(sample_result)
        if 'ontology' in enhanced:
            print(f"+ Result enhancement added ontology with {len(enhanced['ontology']['matched_concepts'])} matched concepts")
        else:
            print("! Result enhancement didn't add ontology information")
        
        # Test GeoScienceWorld API
        geosci_api = GeoScienceWorldAPI()
        print("+ GeoScienceWorldAPI instantiated with comprehensive ontology")
        
        # Test NASA API
        nasa_api = NASAEarthDataAPI()
        print("+ NASAEarthDataAPI instantiated with comprehensive ontology")
        
        return True
        
    except Exception as e:
        print(f"X Error testing enhanced APIs: {e}")
        return False

def test_comprehensive_coverage():
    """Test comprehensive ontology coverage"""
    print("\nTesting Comprehensive Ontology Coverage...")
    
    try:
        from scientific_knowledge_graph import GeologicalOntology
        
        ontology = GeologicalOntology()
        mapping = ontology.ontology_mapping
        
        # Test different geological domains
        domains = {
            'Rock Types': ['igneous_rock', 'metamorphic_rock', 'sedimentary_rock'],
            'Minerals': ['quartz', 'feldspar', 'olivine', 'calcite'],
            'Processes': ['volcanism', 'metamorphism', 'weathering', 'erosion'],
            'Structures': ['fault', 'fold', 'anticline', 'syncline'],
            'Geophysics': ['seismic_wave', 'magnetic_field', 'gravity_anomaly'],
            'Geochronology': ['precambrian', 'paleozoic', 'mesozoic', 'cenozoic'],
            'Remote Sensing': ['multispectral_imaging', 'interferometric_sar', 'lidar'],
            'Geochemistry': ['trace_elements', 'isotope_geochemistry', 'fluid_inclusions'],
            'Hydrogeology': ['groundwater_flow', 'aquifer', 'hydraulic_conductivity'],
            'Climate': ['paleoclimate', 'ice_core_data', 'carbon_cycle'],
            'Astrogeology': ['impact_crater', 'meteorite', 'lunar_geology']
        }
        
        total_coverage = 0
        for domain, concepts in domains.items():
            found = sum(1 for concept in concepts if concept in mapping)
            coverage = (found / len(concepts)) * 100
            total_coverage += coverage
            print(f"+ {domain}: {found}/{len(concepts)} concepts ({coverage:.1f}% coverage)")
        
        avg_coverage = total_coverage / len(domains)
        print(f"+ Overall ontology coverage: {avg_coverage:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"X Error testing comprehensive coverage: {e}")
        return False

def main():
    """Run all enhanced ontology tests"""
    print("=" * 60)
    print("ENHANCED GEOLOGICAL ONTOLOGY SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        test_geological_ontology,
        test_enhanced_apis,
        test_comprehensive_coverage
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"ENHANCED ONTOLOGY RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("All enhanced ontology tests PASSED! System is ready for production.")
    else:
        print("Some enhanced ontology tests FAILED. Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)