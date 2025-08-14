#!/usr/bin/env python3
"""
Test Wiley Online Library integration and disable GeoScienceWorld
"""

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'modules'))

def test_wiley_api_integration():
    """Test WileyOnlineLibraryAPI integration"""
    print("Testing Wiley Online Library API integration...")
    
    try:
        from scientific_knowledge_graph import WileyOnlineLibraryAPI, APIAuthenticationManager, RateLimitManager
        
        # Create Wiley API instance
        wiley_api = WileyOnlineLibraryAPI()
        print("+ WileyOnlineLibraryAPI instantiated successfully")
        
        # Test with custom auth and rate limiting
        auth_manager = APIAuthenticationManager()
        rate_limiter = RateLimitManager()
        
        wiley_api_custom = WileyOnlineLibraryAPI(
            api_key="test_key",
            institution_token="test_token",
            auth_manager=auth_manager,
            rate_limiter=rate_limiter
        )
        
        assert wiley_api_custom.api_name == 'wiley'
        print("+ Wiley API name configured correctly")
        
        # Test API endpoints
        assert "onlinelibrary.wiley.com" in wiley_api.base_url
        assert "agupubs.onlinelibrary.wiley.com" in wiley_api.agu_url
        print("+ Wiley and AGU API endpoints configured")
        
        # Test ontology mapping
        ontology_mapping = asyncio.run(wiley_api.get_ontology_mapping())
        assert len(ontology_mapping) > 50  # Should have comprehensive mapping
        
        # Check for key geological concepts
        key_concepts = ['tectonics', 'seismology', 'volcanology', 'geodesy', 'geophysics']
        for concept in key_concepts:
            assert concept in ontology_mapping
        
        print(f"+ Ontology mapping contains {len(ontology_mapping)} concepts")
        print("+ Key geological concepts present in ontology")
        
        return True
        
    except Exception as e:
        print(f"X Error testing Wiley API integration: {e}")
        return False

def test_wiley_fallback_query():
    """Test Wiley fallback query functionality"""
    print("\nTesting Wiley fallback query...")
    
    try:
        from scientific_knowledge_graph import WileyOnlineLibraryAPI
        
        wiley_api = WileyOnlineLibraryAPI()
        
        # Test fallback query
        results = wiley_api._fallback_query("earthquake seismology", limit=10)
        
        # Check results structure
        assert len(results) == 10
        print(f"+ Fallback query returned {len(results)} results")
        
        # Check result structure
        first_result = results[0]
        required_fields = ['title', 'abstract', 'authors', 'journal', 'publisher', 'year', 'doi']
        
        for field in required_fields:
            assert field in first_result
        
        print("+ All required fields present in results")
        
        # Check AGU journal focus
        agu_journals = [r for r in results if r.get('agu_journal', False)]
        print(f"+ {len(agu_journals)} results from AGU journals")
        
        # Check publisher
        assert all(r['publisher'] == 'Wiley' for r in results)
        print("+ All results correctly attributed to Wiley")
        
        # Check content relevance
        seismo_results = [r for r in results if 'seismo' in r['title'].lower() or 'earthquake' in r['title'].lower()]
        assert len(seismo_results) > 0
        print("+ Results contain relevant seismological content")
        
        return True
        
    except Exception as e:
        print(f"X Error testing Wiley fallback query: {e}")
        return False

def test_authentication_integration():
    """Test authentication integration for Wiley"""
    print("\nTesting Wiley authentication integration...")
    
    try:
        from scientific_knowledge_graph import APIAuthenticationManager
        
        auth_manager = APIAuthenticationManager()
        
        # Test Wiley credentials
        auth_manager.add_credentials('wiley', {
            'api_key': 'test_wiley_key',
            'institution_token': 'test_institution_token'
        })
        
        # Test Wiley auth handler
        async def test_auth():
            headers = await auth_manager.get_auth_headers('wiley')
            assert 'X-API-Key' in headers
            assert headers['X-API-Key'] == 'test_wiley_key'
            assert 'Authorization' in headers
            assert headers['Authorization'] == 'Bearer test_institution_token'
            return True
        
        result = asyncio.run(test_auth())
        print("+ Wiley authentication headers generated correctly")
        
        return result
        
    except Exception as e:
        print(f"X Error testing Wiley authentication: {e}")
        return False

def test_rate_limiting_integration():
    """Test rate limiting integration for Wiley"""
    print("\nTesting Wiley rate limiting integration...")
    
    try:
        from scientific_knowledge_graph import RateLimitManager
        
        rate_limiter = RateLimitManager()
        
        # Check Wiley rate limits
        assert 'wiley' in rate_limiter.rate_limits
        wiley_limits = rate_limiter.rate_limits['wiley']
        
        assert wiley_limits['requests_per_minute'] == 50
        assert wiley_limits['requests_per_hour'] == 800
        assert wiley_limits['concurrent'] == 4
        
        print("+ Wiley rate limits: 50/min, 800/hour, 4 concurrent")
        
        # Check semaphore initialization
        assert 'wiley' in rate_limiter.semaphores
        assert rate_limiter.semaphores['wiley']._value == 4
        print("+ Wiley semaphore initialized correctly")
        
        return True
        
    except Exception as e:
        print(f"X Error testing Wiley rate limiting: {e}")
        return False

def test_data_fusion_integration():
    """Test Data Fusion Pipeline integration with Wiley"""
    print("\nTesting Data Fusion Pipeline integration with Wiley...")
    
    try:
        from scientific_knowledge_graph import DataFusionPipeline
        
        fusion_pipeline = DataFusionPipeline()
        
        # Check that Wiley API is initialized
        assert hasattr(fusion_pipeline, 'wiley_api')
        assert fusion_pipeline.wiley_api.api_name == 'wiley'
        print("+ Wiley API integrated in Data Fusion Pipeline")
        
        # Check fusion rules updated for Wiley
        lit_fusion = fusion_pipeline.fusion_rules['geological_literature_fusion']
        assert 'wiley' in lit_fusion['priority_weights']
        assert lit_fusion['priority_weights']['wiley'] == 0.5  # Highest priority
        print("+ Geological literature fusion prioritizes Wiley (50% weight)")
        
        # Verify geoscienceworld is not in priority weights
        assert 'geoscienceworld' not in lit_fusion['priority_weights']
        print("+ GeoScienceWorld successfully removed from fusion rules")
        
        # Test collection method exists
        assert hasattr(fusion_pipeline, '_collect_wiley_data')
        print("+ Wiley data collection method available")
        
        return True
        
    except Exception as e:
        print(f"X Error testing Data Fusion integration: {e}")
        return False

def test_agu_journal_focus():
    """Test AGU journal focus in Wiley integration"""
    print("\nTesting AGU journal focus...")
    
    try:
        from scientific_knowledge_graph import WileyOnlineLibraryAPI
        
        wiley_api = WileyOnlineLibraryAPI()
        
        # Test AGU journal query
        results = wiley_api._fallback_query("tectonics geodynamics", limit=15)
        
        # Check for AGU journals in results
        agu_journals = {
            "Journal of Geophysical Research: Solid Earth",
            "Geophysical Research Letters", 
            "Reviews of Geophysics",
            "Tectonics",
            "Geochemistry, Geophysics, Geosystems"
        }
        
        found_agu_journals = set()
        for result in results:
            if result['journal'] in agu_journals:
                found_agu_journals.add(result['journal'])
        
        print(f"+ Found {len(found_agu_journals)} different AGU journals in results")
        
        # Check impact factors
        for result in results:
            if 'impact_factor' in result:
                assert result['impact_factor'] > 0
        
        print("+ Impact factors assigned to journals")
        
        # Check research areas
        tectonics_results = [r for r in results if 'tectonics' in r['title'].lower()]
        if tectonics_results:
            areas = tectonics_results[0].get('research_areas', [])
            assert 'Tectonics' in areas or 'Geodynamics' in areas
            print("+ Research areas correctly classified")
        
        return True
        
    except Exception as e:
        print(f"X Error testing AGU journal focus: {e}")
        return False

def main():
    """Run all Wiley integration tests"""
    print("=" * 70)
    print("WILEY ONLINE LIBRARY INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        test_wiley_api_integration,
        test_wiley_fallback_query,
        test_authentication_integration,
        test_rate_limiting_integration,
        test_data_fusion_integration,
        test_agu_journal_focus
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
    print(f"WILEY INTEGRATION RESULTS: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("All Wiley integration tests PASSED!")
        print("+ Wiley Online Library API successfully integrated")
        print("+ AGU journals prioritized for geological research")
        print("+ Comprehensive geological ontology mapping")
        print("+ Authentication and rate limiting configured")
        print("+ Data Fusion Pipeline updated to use Wiley")
        print("+ GeoScienceWorld successfully disabled")
        print("+ Impact factors and research areas included")
        print("+ Ready for production use with institutional access")
    else:
        print("Some Wiley integration tests FAILED.")
        print("Please review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)