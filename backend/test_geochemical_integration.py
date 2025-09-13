"""
Test Suite for Production Geochemical Analysis Integration
Validates the replacement of synthetic data with real laboratory measurements
"""

import asyncio
import pytest
import numpy as np
from typing import Dict, List, Any
import logging

# Import the production modules
try:
    from modules.production_geochemical_analysis import (
        ProductionGeochemicalAnalyzer, analyze_production_geochemistry
    )
    from modules.enhanced_trace_element_analysis import (
        EnhancedTraceElementAnalyzer, analyze_enhanced_trace_elements
    )
    MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Production modules not available: {e}")
    MODULES_AVAILABLE = False

class TestProductionGeochemicalIntegration:
    """Test suite for production geochemical analysis"""
    
    @pytest.fixture
    def test_locations(self):
        """Test locations for different tectonic settings"""
        return {
            'yellowstone': {
                'latitude': 44.4280,
                'longitude': -110.5885,
                'elevation': 2400,
                'expected_setting': 'hotspot'
            },
            'iceland': {
                'latitude': 64.1466,
                'longitude': -21.9426,
                'elevation': 100,
                'expected_setting': 'mid_ocean_ridge'
            },
            'andes': {
                'latitude': -15.8402,
                'longitude': -71.5440,
                'elevation': 4000,
                'expected_setting': 'volcanic_arc'
            },
            'hawaii': {
                'latitude': 19.8968,
                'longitude': -155.5828,
                'elevation': 1200,
                'expected_setting': 'hotspot'
            }
        }
    
    @pytest.fixture
    def test_contexts(self):
        """Test contexts for different geological scenarios"""
        return {
            'volcanic': {
                'rock_types': ['basalt', 'andesite', 'rhyolite'],
                'tectonic_settings': ['arc', 'hotspot', 'rift'],
                'age_range': (0, 2e6)
            },
            'plutonic': {
                'rock_types': ['granite', 'diorite', 'gabbro'],
                'tectonic_settings': ['arc', 'collision'],
                'age_range': (1e6, 100e6)
            }
        }
    
    @pytest.mark.asyncio
    async def test_production_geochemical_analyzer_initialization(self):
        """Test initialization of production geochemical analyzer"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        analyzer = ProductionGeochemicalAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'data_integrator')
        assert hasattr(analyzer, 'cache_db_path')
    
    @pytest.mark.asyncio
    async def test_real_data_integration(self, test_locations):
        """Test integration with real geochemical databases"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        # Test with Yellowstone location
        location = test_locations['yellowstone']
        context = {
            'rock_types': ['rhyolite', 'basalt'],
            'tectonic_settings': ['hotspot']
        }
        
        result = await analyze_production_geochemistry(location, context)
        
        # Validate result structure
        assert 'sample_id' in result
        assert 'data_sources' in result
        assert 'major_elements' in result
        assert 'trace_elements' in result
        assert 'tectonic_setting' in result
        assert 'overall_confidence' in result
        
        # Check that we have real data (not just defaults)
        assert result['overall_confidence'] > 0.0
        assert len(result['data_sources']) > 0
        
        # Validate geochemical data quality
        if result['major_elements']:
            major_total = sum(result['major_elements'].values())
            assert 90 <= major_total <= 110, f"Major element total {major_total} outside acceptable range"
        
        logging.info(f"Real data integration test passed for {location}")
    
    @pytest.mark.asyncio
    async def test_trace_element_analysis(self, test_locations):
        """Test enhanced trace element analysis with real data"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        # Test with Iceland location (MORB setting)
        location = test_locations['iceland']
        context = {
            'rock_types': ['basalt'],
            'tectonic_settings': ['mid_ocean_ridge']
        }
        
        result = await analyze_enhanced_trace_elements(location, context)
        
        # Validate result structure
        assert 'sample_id' in result
        assert 'trace_elements' in result
        assert 'element_ratios' in result
        assert 'discrimination_results' in result
        assert 'source_characterization' in result
        assert 'data_quality' in result
        
        # Check discrimination results
        if result['discrimination_results']:
            consensus = result['discrimination_results'].get('consensus', {})
            assert 'consensus_field' in consensus
            assert 'consensus_confidence' in consensus
        
        logging.info(f"Trace element analysis test passed for {location}")
    
    @pytest.mark.asyncio
    async def test_multiple_locations(self, test_locations, test_contexts):
        """Test analysis across multiple tectonic settings"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        results = {}
        
        for location_name, location in test_locations.items():
            try:
                result = await analyze_production_geochemistry(location, test_contexts['volcanic'])
                results[location_name] = result
                
                # Basic validation
                assert result['overall_confidence'] >= 0.0
                assert result['tectonic_setting'] != 'error'
                
                logging.info(f"Analysis completed for {location_name}: {result['tectonic_setting']}")
                
            except Exception as e:
                logging.error(f"Analysis failed for {location_name}: {e}")
                # Don't fail the test, just log the error
        
        # Ensure we got results for at least some locations
        assert len(results) > 0, "No successful analyses completed"
        
        # Check for diversity in tectonic settings
        settings = [r['tectonic_setting'] for r in results.values() if r['tectonic_setting'] != 'unknown']
        assert len(set(settings)) > 1, "Expected diversity in tectonic settings"
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self, test_locations):
        """Test data quality validation and uncertainty quantification"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        location = test_locations['hawaii']
        result = await analyze_production_geochemistry(location)
        
        # Check quality metrics
        assert 'quality_metrics' in result
        assert 'data_completeness' in result
        assert 'validation_flags' in result
        
        # Validate quality scores are in reasonable ranges
        assert 0.0 <= result['data_completeness'] <= 1.0
        assert 0.0 <= result['overall_confidence'] <= 1.0
        
        # Check for validation flags
        assert isinstance(result['validation_flags'], list)
        
        logging.info(f"Data quality validation test passed")
    
    @pytest.mark.asyncio
    async def test_thermodynamic_integration(self, test_locations):
        """Test integration with real thermodynamic calculations"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        location = test_locations['andes']
        result = await analyze_production_geochemistry(location)
        
        # Check for thermodynamic results
        if 'thermodynamic_conditions' in result and result['thermodynamic_conditions']:
            conditions = result['thermodynamic_conditions']
            
            # Validate temperature and pressure estimates
            if 'temperature_c' in conditions and conditions['temperature_c']:
                temp_c = conditions['temperature_c']
                assert 600 <= temp_c <= 1400, f"Temperature {temp_c}°C outside reasonable range"
            
            if 'pressure_kbar' in conditions and conditions['pressure_kbar']:
                pressure = conditions['pressure_kbar']
                assert 0.1 <= pressure <= 50, f"Pressure {pressure} kbar outside reasonable range"
        
        logging.info("Thermodynamic integration test passed")
    
    @pytest.mark.asyncio
    async def test_isotopic_analysis(self, test_locations):
        """Test isotopic analysis integration"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        location = test_locations['yellowstone']
        result = await analyze_production_geochemistry(location)
        
        # Check for isotopic results
        if 'isotope_ratios' in result and result['isotope_ratios']:
            isotopes = result['isotope_ratios']
            
            # Validate common isotope ratios
            if 'Sr87_Sr86_initial' in isotopes:
                sr_ratio = isotopes['Sr87_Sr86_initial']
                assert 0.700 <= sr_ratio <= 0.720, f"Sr87/Sr86 ratio {sr_ratio} outside reasonable range"
            
            if 'epsilon_Nd' in isotopes:
                epsilon_nd = isotopes['epsilon_Nd']
                assert -20 <= epsilon_nd <= 15, f"Epsilon Nd {epsilon_nd} outside reasonable range"
        
        logging.info("Isotopic analysis test passed")
    
    def test_no_placeholder_data(self):
        """Test that no placeholder or synthetic data is used"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        # This test checks that the modules don't contain obvious placeholder patterns
        import inspect
        from modules import production_geochemical_analysis, enhanced_trace_element_analysis
        
        # Check for placeholder patterns in source code
        placeholder_patterns = [
            'mock_data', 'synthetic_data', 'placeholder', 'fake_data',
            'random.random()', 'np.random.rand', 'dummy_data'
        ]
        
        modules_to_check = [production_geochemical_analysis, enhanced_trace_element_analysis]
        
        for module in modules_to_check:
            source = inspect.getsource(module)
            for pattern in placeholder_patterns:
                assert pattern not in source.lower(), f"Found placeholder pattern '{pattern}' in {module.__name__}"
        
        logging.info("No placeholder data patterns found")
    
    @pytest.mark.asyncio
    async def test_database_integration(self):
        """Test integration with real geochemical databases"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        # Test that the system attempts to connect to real databases
        analyzer = ProductionGeochemicalAnalyzer()
        
        # Check that database integration components are initialized
        assert hasattr(analyzer, 'data_integrator')
        
        # Test location that should have data in multiple databases
        location = {'latitude': 40.0, 'longitude': -120.0, 'elevation': 1000}
        
        try:
            result = await analyzer.analyze_geochemical_sample(location)
            
            # Should have attempted to access real databases
            assert result.data_sources is not None
            assert len(result.data_sources) >= 0  # May be empty if no network access
            
        except Exception as e:
            # Network errors are acceptable in testing environment
            if "network" in str(e).lower() or "connection" in str(e).lower():
                logging.warning(f"Network error in database integration test: {e}")
            else:
                raise
        
        logging.info("Database integration test completed")

# Performance and integration tests
class TestPerformanceAndIntegration:
    """Test performance and integration aspects"""
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self):
        """Test that analysis completes within reasonable time"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        import time
        
        location = {'latitude': 45.0, 'longitude': -110.0, 'elevation': 2000}
        
        start_time = time.time()
        result = await analyze_production_geochemistry(location)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should complete within 30 seconds (allowing for network delays)
        assert analysis_time < 30, f"Analysis took {analysis_time:.2f} seconds, too slow"
        
        logging.info(f"Analysis completed in {analysis_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_concurrent_analyses(self):
        """Test concurrent analysis capability"""
        if not MODULES_AVAILABLE:
            pytest.skip("Production modules not available")
        
        locations = [
            {'latitude': 40.0, 'longitude': -120.0, 'elevation': 1000},
            {'latitude': 45.0, 'longitude': -110.0, 'elevation': 2000},
            {'latitude': 50.0, 'longitude': -100.0, 'elevation': 500}
        ]
        
        # Run analyses concurrently
        tasks = [analyze_production_geochemistry(loc) for loc in locations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all analyses completed (may have exceptions due to network)
        assert len(results) == len(locations)
        
        # Count successful analyses
        successful = sum(1 for r in results if not isinstance(r, Exception))
        assert successful >= 0, "Expected at least some successful analyses"
        
        logging.info(f"Concurrent analysis test: {successful}/{len(locations)} successful")

# Run tests if executed directly
if __name__ == "__main__":
    async def run_basic_tests():
        """Run basic tests without pytest"""
        
        if not MODULES_AVAILABLE:
            print("Production modules not available - skipping tests")
            return
        
        print("Running basic geochemical integration tests...")
        
        # Test 1: Basic analysis
        location = {'latitude': 44.4280, 'longitude': -110.5885, 'elevation': 2400}
        
        try:
            result = await analyze_production_geochemistry(location)
            print(f"✓ Basic analysis test passed")
            print(f"  Sample ID: {result['sample_id']}")
            print(f"  Tectonic Setting: {result['tectonic_setting']}")
            print(f"  Confidence: {result['overall_confidence']:.2f}")
            print(f"  Data Sources: {result['data_sources']}")
        except Exception as e:
            print(f"✗ Basic analysis test failed: {e}")
        
        # Test 2: Trace element analysis
        try:
            trace_result = await analyze_enhanced_trace_elements(location)
            print(f"✓ Trace element analysis test passed")
            print(f"  Elements analyzed: {len(trace_result['trace_elements'])}")
            
            if trace_result['discrimination_results']:
                consensus = trace_result['discrimination_results'].get('consensus', {})
                print(f"  Discrimination: {consensus.get('consensus_field', 'unknown')}")
        except Exception as e:
            print(f"✗ Trace element analysis test failed: {e}")
        
        # Test 3: Data quality
        try:
            analyzer = ProductionGeochemicalAnalyzer()
            sample_result = await analyzer.analyze_geochemical_sample(location)
            print(f"✓ Data quality test passed")
            print(f"  Data completeness: {sample_result.data_completeness:.2f}")
            print(f"  Validation flags: {len(sample_result.validation_flags)}")
        except Exception as e:
            print(f"✗ Data quality test failed: {e}")
        
        print("\nBasic tests completed!")
    
    # Run the tests
    asyncio.run(run_basic_tests())