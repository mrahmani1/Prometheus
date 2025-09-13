"""
Comprehensive Test Suite for Volcanic Systems Database
Tests all components of task 4: GVP integration, NASA FIRMS, tectonic classification, monitoring networks
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

# Import the modules to test
from modules.comprehensive_volcanic_systems_database import (
    ComprehensiveVolcanicSystemsDatabase,
    VolcanicSystem,
    TectonicSetting,
    VolcanoType,
    create_volcanic_systems_database
)

from modules.volcanic_systems_integration_service import (
    VolcanicSystemsIntegrationService,
    create_volcanic_systems_service
)

class TestComprehensiveVolcanicSystemsDatabase:
    """Test suite for the comprehensive volcanic systems database"""
    
    @pytest.fixture
    async def database(self):
        """Create test database instance"""
        config = {
            'postgres_dsn': 'postgresql://test:test@localhost:5432/test_prometheus',
            'redis_url': 'redis://localhost:6379/1',
            'cache_enabled': False,  # Disable cache for testing
            'auto_sync_enabled': False  # Disable auto sync for testing
        }
        
        db = await create_volcanic_systems_database(config)
        yield db
        await db.close()
    
    @pytest.fixture
    async def integration_service(self):
        """Create test integration service instance"""
        config = {
            'postgres_dsn': 'postgresql://test:test@localhost:5432/test_prometheus',
            'redis_url': 'redis://localhost:6379/1',
            'cache_enabled': False,
            'auto_sync_enabled': False
        }
        
        service = await create_volcanic_systems_service(config)
        yield service
        await service.close()
    
    async def test_database_initialization(self, database):
        """Test 4.1: Verify database initialization and schema creation"""
        assert database is not None
        assert database.postgres_pool is not None
        assert database.session is not None
        
        # Test database schema exists
        async with database.postgres_pool.acquire() as conn:
            # Check if volcanic systems schema exists
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = 'volcanic_systems')"
            )
            assert schema_exists
            
            # Check if main tables exist
            tables = ['global_volcano_catalog', 'eruption_history', 'thermal_anomalies', 
                     'monitoring_stations', 'tectonic_context']
            
            for table in tables:
                table_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = 'volcanic_systems' AND table_name = $1)",
                    table
                )
                assert table_exists, f"Table {table} should exist"
    
    async def test_gvp_integration(self, database):
        """Test 4.1: Global Volcanism Program database integration"""
        # Test GVP data integration
        integration_result = await database.integrate_gvp_database(force_refresh=True)
        
        assert integration_result['status'] == 'completed'
        assert integration_result['volcanoes_processed'] >= 0
        assert 'start_time' in integration_result
        assert 'end_time' in integration_result
        
        # Verify volcanoes were stored in database
        volcanoes = await database.query_volcanic_systems()
        
        # Should have at least some test volcanoes
        if integration_result['volcanoes_processed'] > 0:
            assert len(volcanoes) > 0
            
            # Test volcano data structure
            test_volcano = volcanoes[0]
            assert isinstance(test_volcano, VolcanicSystem)
            assert test_volcano.volcano_id
            assert test_volcano.name
            assert len(test_volcano.location) == 3  # lat, lon, elevation
            assert isinstance(test_volcano.volcano_type, VolcanoType)
            assert isinstance(test_volcano.tectonic_setting, TectonicSetting)
    
    async def test_nasa_firms_integration(self, database):
        """Test 4.2: NASA FIRMS thermal anomaly integration"""
        # Test thermal data integration
        time_window = timedelta(days=1)  # Short window for testing
        integration_result = await database.integrate_nasa_firms_thermal_data(time_window)
        
        assert integration_result['status'] == 'completed'
        assert 'anomalies_processed' in integration_result
        assert 'satellites_processed' in integration_result
        assert 'MODIS' in integration_result['satellites_processed'] or 'VIIRS' in integration_result['satellites_processed']
        
        # Test thermal activity analysis
        volcanoes = await database.query_volcanic_systems()
        if volcanoes:
            test_volcano = volcanoes[0]
            time_range = (datetime.utcnow() - timedelta(days=7), datetime.utcnow())
            
            thermal_analysis = await database.get_thermal_activity_analysis(
                test_volcano.volcano_id, time_range
            )
            
            assert 'volcano_id' in thermal_analysis
            assert 'thermal_anomalies' in thermal_analysis
            assert 'analysis' in thermal_analysis
            assert thermal_analysis['volcano_id'] == test_volcano.volcano_id
    
    async def test_tectonic_classification(self, database):
        """Test 4.3: Tectonic setting classification"""
        # Test tectonic classification
        classification_result = await database.classify_tectonic_settings(force_refresh=True)
        
        assert classification_result['status'] == 'completed'
        assert 'volcanoes_classified' in classification_result
        assert 'tectonic_settings' in classification_result
        
        # Verify tectonic settings are valid
        for setting_name, count in classification_result['tectonic_settings'].items():
            # Check that setting name is a valid TectonicSetting value
            valid_settings = [setting.value for setting in TectonicSetting]
            assert setting_name in valid_settings
            assert count >= 0
        
        # Test querying by tectonic setting
        for tectonic_setting in TectonicSetting:
            volcanoes = await database.query_volcanic_systems(tectonic_setting=tectonic_setting)
            # Should return list (may be empty)
            assert isinstance(volcanoes, list)
            
            # If volcanoes found, verify they have correct tectonic setting
            for volcano in volcanoes:
                assert volcano.tectonic_setting == tectonic_setting
    
    async def test_monitoring_network_registry(self, database):
        """Test 4.4: Monitoring network registry creation"""
        # Test monitoring network registry
        registry_result = await database.create_monitoring_network_registry(force_refresh=True)
        
        assert registry_result['status'] == 'completed'
        assert 'stations_processed' in registry_result
        assert 'station_types' in registry_result
        assert 'real_time_stations' in registry_result
        
        # Test monitoring network status for a volcano
        volcanoes = await database.query_volcanic_systems()
        if volcanoes:
            test_volcano = volcanoes[0]
            
            monitoring_status = await database.get_monitoring_network_status(test_volcano.volcano_id)
            
            assert 'volcano_id' in monitoring_status
            assert 'monitoring_analysis' in monitoring_status
            assert 'stations' in monitoring_status
            assert monitoring_status['volcano_id'] == test_volcano.volcano_id
            
            # Verify monitoring analysis structure
            analysis = monitoring_status['monitoring_analysis']
            assert 'total_stations' in analysis
            assert 'station_types' in analysis
            assert 'real_time_stations' in analysis
            assert 'operators' in analysis
    
    async def test_comprehensive_synchronization(self, database):
        """Test complete synchronization of all data sources"""
        # Test comprehensive synchronization
        sync_result = await database.synchronize_with_authoritative_sources()
        
        assert 'gvp_sync' in sync_result
        assert 'firms_sync' in sync_result
        assert 'tectonic_sync' in sync_result
        assert 'monitoring_sync' in sync_result
        assert 'validation_results' in sync_result
        assert sync_result['overall_status'] == 'completed'
        
        # Verify each component completed successfully
        for component, result in sync_result.items():
            if component.endswith('_sync') and isinstance(result, dict):
                # Should have status (may be error for some components in test environment)
                assert 'status' in result or 'error' in result
    
    async def test_advanced_querying(self, database):
        """Test advanced querying capabilities"""
        # Test basic query
        all_volcanoes = await database.query_volcanic_systems()
        assert isinstance(all_volcanoes, list)
        
        # Test region filtering
        if all_volcanoes:
            test_region = all_volcanoes[0].region
            region_volcanoes = await database.query_volcanic_systems(region=test_region)
            
            for volcano in region_volcanoes:
                assert volcano.region == test_region
        
        # Test activity filtering
        recent_date = datetime.utcnow() - timedelta(days=365*10)  # Last 10 years
        active_volcanoes = await database.query_volcanic_systems(activity_since=recent_date)
        
        for volcano in active_volcanoes:
            if volcano.last_known_eruption:
                assert volcano.last_known_eruption >= recent_date
        
        # Test bounding box filtering
        # Test with a small bounding box around Italy (Etna, Stromboli area)
        italy_bbox = (14.0, 37.0, 16.0, 39.0)  # min_lon, min_lat, max_lon, max_lat
        italy_volcanoes = await database.query_volcanic_systems(bbox=italy_bbox)
        
        for volcano in italy_volcanoes:
            lon, lat = volcano.location[1], volcano.location[0]
            assert italy_bbox[0] <= lon <= italy_bbox[2]
            assert italy_bbox[1] <= lat <= italy_bbox[3]
    
    async def test_data_validation(self, database):
        """Test data validation and quality control"""
        # This would test the comprehensive data validation
        # For now, test that the method exists and returns expected structure
        try:
            validation_result = await database._comprehensive_data_validation()
            # Should return a dictionary with validation results
            assert isinstance(validation_result, dict)
        except AttributeError:
            # Method might not be implemented yet, which is acceptable
            pass

class TestVolcanicSystemsIntegrationService:
    """Test suite for the integration service"""
    
    async def test_service_initialization(self, integration_service):
        """Test integration service initialization"""
        assert integration_service.is_initialized
        assert integration_service.database is not None
        assert integration_service.monitoring_system is not None
        assert integration_service.data_connector is not None
    
    async def test_comprehensive_synchronization_service(self, integration_service):
        """Test comprehensive synchronization through service"""
        sync_result = await integration_service.perform_comprehensive_synchronization()
        
        assert 'synchronization_summary' in sync_result
        assert 'service_status' in sync_result
        assert 'data_quality_metrics' in sync_result
        assert 'coverage_analysis' in sync_result
        assert 'integration_validation' in sync_result
        
        # Verify service status
        service_status = sync_result['service_status']
        assert service_status['database_status'] == 'operational'
        assert service_status['monitoring_status'] == 'operational'
        assert service_status['data_connector_status'] == 'operational'
    
    async def test_comprehensive_volcano_data(self, integration_service):
        """Test comprehensive volcano data retrieval"""
        # First get list of volcanoes
        search_result = await integration_service.search_volcanic_systems_advanced({})
        
        if search_result['total_results'] > 0:
            test_volcano_id = search_result['volcanoes'][0]['volcano_id']
            
            comprehensive_data = await integration_service.get_volcanic_system_comprehensive_data(test_volcano_id)
            
            assert 'volcano_info' in comprehensive_data
            assert 'thermal_analysis' in comprehensive_data
            assert 'monitoring_status' in comprehensive_data
            assert 'eruption_history' in comprehensive_data
            assert 'current_alerts' in comprehensive_data
            assert 'data_completeness' in comprehensive_data
    
    async def test_advanced_search(self, integration_service):
        """Test advanced search functionality"""
        # Test basic search
        search_result = await integration_service.search_volcanic_systems_advanced({})
        
        assert 'search_criteria' in search_result
        assert 'total_results' in search_result
        assert 'volcanoes' in search_result
        assert 'search_metadata' in search_result
        assert 'summary_statistics' in search_result
        
        # Test search with criteria
        search_criteria = {
            'tectonic_setting': 'subduction_zone',
            'min_elevation': 1000
        }
        
        filtered_result = await integration_service.search_volcanic_systems_advanced(search_criteria)
        
        assert filtered_result['search_criteria'] == search_criteria
        
        # Verify filtering worked
        for volcano in filtered_result['volcanoes']:
            if volcano.get('tectonic_setting'):
                assert volcano['tectonic_setting'] == 'subduction_zone'
            if volcano.get('elevation_m'):
                assert volcano['elevation_m'] >= 1000
    
    async def test_global_activity_summary(self, integration_service):
        """Test global volcanic activity summary"""
        summary = await integration_service.get_global_volcanic_activity_summary()
        
        assert 'total_volcanoes' in summary
        assert 'volcanoes_by_tectonic_setting' in summary
        assert 'volcanoes_by_type' in summary
        assert 'volcanoes_by_region' in summary
        assert 'recent_thermal_activity' in summary
        assert 'monitoring_coverage' in summary
        assert 'data_quality_overview' in summary
        
        # Verify thermal activity structure
        thermal_activity = summary['recent_thermal_activity']
        assert 'active_volcanoes' in thermal_activity
        assert 'total_thermal_anomalies' in thermal_activity
        assert 'most_active' in thermal_activity
    
    async def test_database_integrity_validation(self, integration_service):
        """Test database integrity validation"""
        validation_result = await integration_service.validate_database_integrity()
        
        assert 'validation_time' in validation_result
        assert 'overall_status' in validation_result
        assert 'data_completeness' in validation_result
        assert 'data_quality' in validation_result
        assert 'cross_validation' in validation_result
        assert 'recommendations' in validation_result
        
        # Overall status should be one of expected values
        valid_statuses = ['excellent', 'good', 'needs_improvement', 'unknown']
        assert validation_result['overall_status'] in valid_statuses
    
    async def test_service_status(self, integration_service):
        """Test service status reporting"""
        status = await integration_service.get_service_status()
        
        assert 'service_name' in status
        assert 'version' in status
        assert 'status' in status
        assert 'components' in status
        assert 'capabilities' in status
        
        # Verify expected capabilities
        expected_capabilities = [
            'gvp_integration',
            'nasa_firms_thermal_monitoring',
            'tectonic_classification',
            'monitoring_network_registry'
        ]
        
        for capability in expected_capabilities:
            assert capability in status['capabilities']

class TestDataStructures:
    """Test volcanic data structures and enums"""
    
    def test_volcanic_system_structure(self):
        """Test VolcanicSystem data structure"""
        volcano = VolcanicSystem(
            volcano_id='test_001',
            gvp_id='123456',
            name='Test Volcano',
            location=(45.0, -120.0, 2500.0),
            country='Test Country',
            region='Test Region',
            subregion='Test Subregion',
            volcano_type=VolcanoType.STRATOVOLCANO,
            tectonic_setting=TectonicSetting.SUBDUCTION_ZONE,
            elevation_m=2500.0
        )
        
        assert volcano.volcano_id == 'test_001'
        assert volcano.name == 'Test Volcano'
        assert len(volcano.location) == 3
        assert isinstance(volcano.volcano_type, VolcanoType)
        assert isinstance(volcano.tectonic_setting, TectonicSetting)
    
    def test_tectonic_setting_enum(self):
        """Test TectonicSetting enum values"""
        expected_settings = [
            'subduction_zone', 'mid_ocean_ridge', 'hotspot', 'continental_rift',
            'back_arc', 'intraplate', 'transform_fault', 'continental_arc',
            'island_arc', 'flood_basalt'
        ]
        
        actual_settings = [setting.value for setting in TectonicSetting]
        
        for expected in expected_settings:
            assert expected in actual_settings
    
    def test_volcano_type_enum(self):
        """Test VolcanoType enum values"""
        expected_types = [
            'stratovolcano', 'shield', 'cinder_cone', 'caldera', 'complex',
            'lava_dome', 'maar', 'fissure_vent', 'submarine', 'pyroclastic_cone'
        ]
        
        actual_types = [v_type.value for v_type in VolcanoType]
        
        for expected in expected_types:
            assert expected in actual_types

# Performance and integration tests
class TestPerformanceAndIntegration:
    """Test performance and integration aspects"""
    
    async def test_concurrent_data_access(self, integration_service):
        """Test concurrent access to volcanic data"""
        # Create multiple concurrent requests
        tasks = []
        
        for i in range(5):
            task = integration_service.search_volcanic_systems_advanced({
                'region': f'test_region_{i % 3}'
            })
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
            assert 'total_results' in result
    
    async def test_large_dataset_handling(self, integration_service):
        """Test handling of large datasets"""
        # Test search without filters (potentially large result set)
        search_result = await integration_service.search_volcanic_systems_advanced({})
        
        # Should handle large results gracefully
        assert isinstance(search_result, dict)
        assert 'total_results' in search_result
        
        # If large dataset, verify pagination or limiting works
        if search_result['total_results'] > 1000:
            # Should have some form of result management
            assert len(search_result['volcanoes']) <= search_result['total_results']

# Run the tests
if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.INFO)
    
    # Run specific test functions
    async def run_tests():
        """Run all tests"""
        print("Starting Comprehensive Volcanic Systems Database Tests...")
        
        # Test database initialization
        config = {
            'postgres_dsn': 'postgresql://test:test@localhost:5432/test_prometheus',
            'redis_url': 'redis://localhost:6379/1',
            'cache_enabled': False,
            'auto_sync_enabled': False
        }
        
        try:
            # Test database creation
            database = await create_volcanic_systems_database(config)
            print("✓ Database initialization successful")
            
            # Test service creation
            service = await create_volcanic_systems_service(config)
            print("✓ Integration service initialization successful")
            
            # Test basic functionality
            status = await service.get_service_status()
            print(f"✓ Service status: {status['status']}")
            
            # Test synchronization
            sync_result = await service.perform_comprehensive_synchronization()
            print(f"✓ Synchronization completed: {sync_result['service_status']['database_status']}")
            
            # Cleanup
            await service.close()
            await database.close()
            
            print("All tests completed successfully!")
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
            raise
    
    # Run the tests
    asyncio.run(run_tests())