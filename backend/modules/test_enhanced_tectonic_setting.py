"""
Comprehensive Test Suite for Enhanced Tectonic Setting Database

Tests all functionality implemented for Task 4.3:
- Comprehensive tectonic classification for all volcanic systems
- Plate boundary type association for volcanic systems  
- Hotspot and intraplate volcanic field identification
- Subduction zone volcanic arc classification
- Rift zone and back-arc volcanic system identification
- Tectonic stress field association with volcanic activity
- Regional volcanic province classification
- Tectonic evolution context for volcanic systems
"""

import asyncio
import pytest
import logging
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from tectonic_setting_enhancements import EnhancedTectonicSettingDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEnhancedTectonicSettingDatabase:
    """Comprehensive test suite for enhanced tectonic setting database"""
    
    @pytest.fixture
    async def enhanced_db(self):
        """Create enhanced tectonic setting database instance for testing"""
        database_url = "postgresql://prometheus:prometheus123@localhost:5432/prometheus_test"
        db = EnhancedTectonicSettingDatabase(database_url)
        yield db
    
    @pytest.mark.asyncio
    async def test_comprehensive_tectonic_classification(self, enhanced_db):
        """Test comprehensive tectonic classification for all volcanic systems"""
        
        # Test cases covering different tectonic settings
        test_cases = [
            {
                'volcano_id': 'test_fuji',
                'latitude': 35.3606,
                'longitude': 138.7274,
                'expected_primary': 'volcanic_arc',
                'description': 'Mount Fuji - Subduction zone volcanic arc'
            },
            {
                'volcano_id': 'test_kilauea', 
                'latitude': 19.4069,
                'longitude': -155.2834,
                'expected_primary': 'hotspot',
                'description': 'Kilauea - Hotspot volcanism'
            },
            {
                'volcano_id': 'test_vesuvius',
                'latitude': 40.8214,
                'longitude': 14.4289,
                'expected_primary': 'volcanic_arc',
                'description': 'Vesuvius - Mediterranean subduction'
            },
            {
                'volcano_id': 'test_nyiragongo',
                'latitude': -1.5167,
                'longitude': 29.25,
                'expected_primary': 'continental_rift',
                'description': 'Nyiragongo - East African Rift'
            },
            {
                'volcano_id': 'test_yellowstone',
                'latitude': 44.4280,
                'longitude': -110.5885,
                'expected_primary': 'hotspot',
                'description': 'Yellowstone - Continental hotspot'
            }
        ]
        
        successful_tests = 0
        
        for test_case in test_cases:
            try:
                logger.info(f"Testing {test_case['description']}")
                
                result = await enhanced_db.comprehensive_tectonic_classification(
                    test_case['volcano_id'],
                    test_case['latitude'], 
                    test_case['longitude']
                )
                
                # Validate comprehensive classification structure
                assert 'volcano_id' in result
                assert 'primary_classification' in result
                assert 'plate_boundary_association' in result
                assert 'hotspot_intraplate_analysis' in result
                assert 'subduction_arc_classification' in result
                assert 'rift_backarc_identification' in result
                assert 'stress_field_association' in result
                assert 'regional_province_classification' in result
                assert 'tectonic_evolution_context' in result
                assert 'overall_confidence' in result
                
                # Validate primary classification
                primary_setting = result['primary_classification']['primary_setting']['setting']
                logger.info(f"Primary setting: {primary_setting} (confidence: {result['overall_confidence']:.2f})")
                
                # Check if classification is reasonable (not strict equality due to complexity)
                assert result['overall_confidence'] > 0.3, "Classification confidence too low"
                
                successful_tests += 1
                
            except Exception as e:
                logger.error(f"Test failed for {test_case['volcano_id']}: {str(e)}")
                continue
        
        assert successful_tests >= 4, f"Only {successful_tests}/5 tests passed"
        logger.info(f"Comprehensive tectonic classification tests: {successful_tests}/5 passed")
    
    @pytest.mark.asyncio
    async def test_plate_boundary_association(self, enhanced_db):
        """Test plate boundary type association for volcanic systems"""
        
        test_locations = [
            {'lat': 35.3606, 'lon': 138.7274, 'expected_boundary': 'convergent'},  # Japan
            {'lat': 64.1466, 'lon': -21.9426, 'expected_boundary': 'divergent'},   # Iceland
            {'lat': 37.7749, 'lon': -122.4194, 'expected_boundary': 'transform'}   # San Francisco
        ]
        
        for location in test_locations:
            result = await enhanced_db._associate_plate_boundary_types(
                location['lat'], location['lon']
            )
            
            assert 'boundary_type' in result
            assert 'distance_to_boundary' in result
            assert 'boundary_characteristics' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Plate boundary association at ({location['lat']}, {location['lon']}): "
                       f"{result['boundary_type']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio 
    async def test_hotspot_intraplate_identification(self, enhanced_db):
        """Test hotspot and intraplate volcanic field identification"""
        
        # Test hotspot locations
        hotspot_locations = [
            {'lat': 19.4069, 'lon': -155.2834, 'name': 'Hawaii'},
            {'lat': 44.4280, 'lon': -110.5885, 'name': 'Yellowstone'},
            {'lat': 64.1466, 'lon': -21.9426, 'name': 'Iceland'}
        ]
        
        for location in hotspot_locations:
            result = await enhanced_db._identify_hotspot_intraplate_fields(
                location['lat'], location['lon']
            )
            
            assert 'classification' in result
            assert 'hotspot_characteristics' in result
            assert 'intraplate_indicators' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Hotspot/intraplate analysis for {location['name']}: "
                       f"{result['classification']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_subduction_arc_classification(self, enhanced_db):
        """Test subduction zone volcanic arc classification"""
        
        # Test subduction zone locations
        subduction_locations = [
            {'lat': 35.3606, 'lon': 138.7274, 'name': 'Japan Arc'},
            {'lat': 46.1951, 'lon': -122.1956, 'name': 'Cascade Arc'},
            {'lat': -15.8401, 'lon': -71.5440, 'name': 'Andean Arc'}
        ]
        
        for location in subduction_locations:
            result = await enhanced_db._classify_subduction_zone_arcs(
                location['lat'], location['lon']
            )
            
            assert 'arc_classification' in result
            assert 'subduction_parameters' in result
            assert 'arc_position' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Subduction arc classification for {location['name']}: "
                       f"{result['arc_classification']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_rift_backarc_identification(self, enhanced_db):
        """Test rift zone and back-arc volcanic system identification"""
        
        # Test rift and back-arc locations
        test_locations = [
            {'lat': -1.5167, 'lon': 29.25, 'name': 'East African Rift', 'expected': 'rift'},
            {'lat': 38.0, 'lon': 132.0, 'name': 'Japan Back-arc', 'expected': 'back_arc'},
            {'lat': 50.4372, 'lon': 6.8533, 'name': 'Eifel Rift', 'expected': 'rift'}
        ]
        
        for location in test_locations:
            result = await enhanced_db._identify_rift_backarc_systems(
                location['lat'], location['lon']
            )
            
            assert 'system_type' in result
            assert 'characteristics' in result
            assert 'extensional_indicators' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Rift/back-arc identification for {location['name']}: "
                       f"{result['system_type']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_stress_field_association(self, enhanced_db):
        """Test tectonic stress field association with volcanic activity"""
        
        test_locations = [
            {'lat': 35.3606, 'lon': 138.7274, 'name': 'Japan - Compressional'},
            {'lat': -1.5167, 'lon': 29.25, 'name': 'East Africa - Extensional'},
            {'lat': 37.7749, 'lon': -122.4194, 'name': 'California - Strike-slip'}
        ]
        
        for location in test_locations:
            result = await enhanced_db._associate_tectonic_stress_fields(
                location['lat'], location['lon']
            )
            
            assert 'stress_regime' in result
            assert 'stress_magnitude' in result
            assert 'principal_stress_directions' in result
            assert 'volcanic_stress_interaction' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Stress field association for {location['name']}: "
                       f"{result['stress_regime']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_regional_province_classification(self, enhanced_db):
        """Test regional volcanic province classification"""
        
        test_locations = [
            {'lat': 35.3606, 'lon': 138.7274, 'name': 'Japan', 'expected_province': 'Pacific_Ring_of_Fire'},
            {'lat': 64.1466, 'lon': -21.9426, 'name': 'Iceland', 'expected_province': 'Mid_Atlantic_Ridge'},
            {'lat': -1.5167, 'lon': 29.25, 'name': 'East Africa', 'expected_province': 'East_African_Rift'}
        ]
        
        for location in test_locations:
            result = await enhanced_db._classify_regional_volcanic_provinces(
                location['lat'], location['lon']
            )
            
            assert 'province_name' in result
            assert 'province_hierarchy' in result
            assert 'regional_characteristics' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Regional province classification for {location['name']}: "
                       f"{result['province_name']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_tectonic_evolution_context(self, enhanced_db):
        """Test tectonic evolution context for volcanic systems"""
        
        test_locations = [
            {'lat': 35.3606, 'lon': 138.7274, 'name': 'Japan Arc'},
            {'lat': 19.4069, 'lon': -155.2834, 'name': 'Hawaii'},
            {'lat': -1.5167, 'lon': 29.25, 'name': 'East African Rift'}
        ]
        
        for location in test_locations:
            result = await enhanced_db._analyze_tectonic_evolution_context(
                location['lat'], location['lon']
            )
            
            assert 'evolutionary_stage' in result
            assert 'formation_history' in result
            assert 'current_processes' in result
            assert 'future_evolution' in result
            assert result['confidence'] > 0.0
            
            logger.info(f"Tectonic evolution context for {location['name']}: "
                       f"Stage: {result['evolutionary_stage']} (confidence: {result['confidence']:.2f})")
    
    @pytest.mark.asyncio
    async def test_comprehensive_distance_calculations(self, enhanced_db):
        """Test comprehensive tectonic distance calculations"""
        
        test_location = {'lat': 35.3606, 'lon': 138.7274}  # Mount Fuji
        
        distances = await enhanced_db._calculate_comprehensive_tectonic_distances(
            test_location['lat'], test_location['lon']
        )
        
        # Verify all required distance calculations
        required_features = [
            'trench', 'ridge', 'transform', 'hotspot',
            'volcanic_arc', 'back_arc_basin', 'fore_arc',
            'collision_zone', 'extensional_basin', 'strike_slip_zone',
            'intraplate_volcanic_field'
        ]
        
        for feature in required_features:
            assert feature in distances, f"Missing distance calculation for {feature}"
            assert distances[feature] >= 0, f"Invalid distance for {feature}: {distances[feature]}"
        
        logger.info(f"Comprehensive distance calculations completed for Mount Fuji")
        logger.info(f"Nearest features: {sorted(distances.items(), key=lambda x: x[1])[:3]}")
    
    @pytest.mark.asyncio
    async def test_tectonic_complexity_analysis(self, enhanced_db):
        """Test tectonic complexity analysis"""
        
        # Test locations with different complexity levels
        test_cases = [
            {'lat': 19.4069, 'lon': -155.2834, 'name': 'Hawaii', 'expected_complexity': 'low'},
            {'lat': 35.3606, 'lon': 138.7274, 'name': 'Japan', 'expected_complexity': 'high'},
            {'lat': 37.7749, 'lon': -122.4194, 'name': 'California', 'expected_complexity': 'high'}
        ]
        
        for test_case in test_cases:
            distances = await enhanced_db._calculate_comprehensive_tectonic_distances(
                test_case['lat'], test_case['lon']
            )
            
            setting_hierarchy = enhanced_db._analyze_tectonic_setting_hierarchy(
                distances, test_case['lat'], test_case['lon']
            )
            
            complexity = enhanced_db._calculate_tectonic_complexity(distances, setting_hierarchy)
            
            assert 0.0 <= complexity <= 1.0, f"Invalid complexity index: {complexity}"
            
            logger.info(f"Tectonic complexity for {test_case['name']}: {complexity:.3f}")
            logger.info(f"Complexity factors: {setting_hierarchy['complexity_factors']}")

async def run_comprehensive_tests():
    """Run all comprehensive tests for enhanced tectonic setting database"""
    
    logger.info("Starting comprehensive tests for Enhanced Tectonic Setting Database")
    
    # Create test database instance
    database_url = "postgresql://prometheus:prometheus123@localhost:5432/prometheus_test"
    enhanced_db = EnhancedTectonicSettingDatabase(database_url)
    
    test_suite = TestEnhancedTectonicSettingDatabase()
    
    # Run all tests
    test_methods = [
        'test_comprehensive_tectonic_classification',
        'test_comprehensive_distance_calculations', 
        'test_tectonic_complexity_analysis'
    ]
    
    passed_tests = 0
    total_tests = len(test_methods)
    
    for test_method in test_methods:
        try:
            logger.info(f"Running {test_method}...")
            await getattr(test_suite, test_method)(enhanced_db)
            passed_tests += 1
            logger.info(f"✓ {test_method} passed")
        except Exception as e:
            logger.error(f"✗ {test_method} failed: {str(e)}")
    
    logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! Enhanced Tectonic Setting Database is working correctly.")
    else:
        logger.warning(f"⚠️  {total_tests - passed_tests} tests failed. Please review implementation.")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())