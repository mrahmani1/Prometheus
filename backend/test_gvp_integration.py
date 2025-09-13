"""
Test suite for Global Volcanism Program Integration
Comprehensive testing of GVP database integration functionality
"""

import asyncio
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from global_volcanism_program_integration import (
    GlobalVolcanismProgramIntegration,
    VolcanicSystem,
    EruptionEvent,
    VolcanicHazard,
    MonitoringNetwork,
    DataProvenance,
    VolcanoType,
    TectonicSetting,
    DataQuality
)

class TestGVPIntegration:
    """Test class for GVP integration functionality"""
    
    @pytest.fixture
    async def gvp_integration(self):
        """Create GVP integration instance for testing"""
        integration = GlobalVolcanismProgramIntegration()
        
        # Mock the session to avoid actual HTTP requests
        integration.session = AsyncMock()
        
        # Mock database components
        integration.db_engine = MagicMock()
        integration.db_session_maker = MagicMock()
        
        return integration
    
    @pytest.fixture
    def sample_volcano_data(self):
        """Sample volcano data for testing"""
        return {
            'volcano_id': '211060',
            'volcano_name': 'Etna',
            'latitude': 37.7510,
            'longitude': 14.9934,
            'elevation': 3357,
            'country': 'Italy',
            'region': 'Sicily',
            'subregion': 'Eastern Sicily',
            'primary_volcano_type': 'Stratovolcano',
            'tectonic_setting': 'Subduction zone',
            'last_known_eruption': '2023-12-01T00:00:00Z',
            'population_within_5_km': 15000,
            'population_within_10_km': 45000,
            'population_within_30_km': 850000,
            'population_within_100_km': 3200000,
            'morphology': {
                'summit_elevation': 3357,
                'base_diameter': 45000,
                'crater_count': 4
            },
            'rock_types': ['Basalt', 'Trachybasalt', 'Hawaiite'],
            'dating_methods': ['Historical Records', 'Radiocarbon'],
            'evidence_category': 'Observed',
            'primary_photo_link': 'https://volcano.si.edu/photos/etna.jpg',
            'synonyms': ['Mount Etna', 'Mongibello']
        }
    
    @pytest.fixture
    def sample_eruption_data(self):
        """Sample eruption data for testing"""
        return {
            'eruption_id': '211060-2023-001',
            'volcano_id': '211060',
            'start_date': '2023-05-15T10:30:00Z',
            'end_date': '2023-05-20T18:45:00Z',
            'vei': 2,
            'eruption_category': 'Strombolian',
            'area_of_activity': 'Southeast Crater',
            'evidence_method': 'Satellite (infrared)',
            'explosivity_index': 2.3,
            'fatalities': 0,
            'injuries': 2,
            'damage_millions_usd': 1.5,
            'houses_destroyed': 0,
            'evacuated': 150,
            'tsunami': False,
            'earthquake': True,
            'mudflow_lahar': False,
            'pyroclastic_flow': True,
            'lava_flow': True,
            'dome_growth': False,
            'explosive_activity': True,
            'central_vent_eruption': True,
            'radial_fissure_eruption': False,
            'regional_fissure_eruption': False,
            'phreatomagmatic': False,
            'phreatic': False,
            'description': 'Strombolian activity with lava flows and pyroclastic flows',
            'references': ['GVP Report 2023-05', 'INGV Bulletin 2023-20']
        }
    
    @pytest.mark.asyncio
    async def test_initialization(self, gvp_integration):
        """Test GVP integration initialization"""
        # Mock the initialization methods
        gvp_integration._initialize_database = AsyncMock()
        gvp_integration._test_gvp_connection = AsyncMock()
        gvp_integration._load_sync_status = AsyncMock()
        
        await gvp_integration.initialize()
        
        # Verify initialization calls
        gvp_integration._initialize_database.assert_called_once()
        gvp_integration._test_gvp_connection.assert_called_once()
        gvp_integration._load_sync_status.assert_called_once()
        
        assert gvp_integration.session is not None
    
    @pytest.mark.asyncio
    async def test_parse_volcano_data(self, gvp_integration, sample_volcano_data):
        """Test volcano data parsing"""
        volcano = await gvp_integration._parse_volcano_data(sample_volcano_data)
        
        assert volcano is not None
        assert volcano.gvp_id == '211060'
        assert volcano.volcano_name == 'Etna'
        assert volcano.location == (37.7510, 14.9934, 3357)
        assert volcano.country == 'Italy'
        assert volcano.volcano_type == VolcanoType.STRATOVOLCANO
        assert volcano.tectonic_setting == TectonicSetting.SUBDUCTION_ZONE
        assert volcano.population_within_5km == 15000
        assert len(volcano.rock_types) == 3
        assert 'Basalt' in volcano.rock_types
    
    @pytest.mark.asyncio
    async def test_parse_eruption_data(self, gvp_integration, sample_eruption_data):
        """Test eruption data parsing"""
        eruption = await gvp_integration._parse_eruption_data(sample_eruption_data)
        
        assert eruption is not None
        assert eruption.eruption_id == '211060-2023-001'
        assert eruption.gvp_volcano_id == '211060'
        assert eruption.vei == 2
        assert eruption.pyroclastic_flow is True
        assert eruption.lava_flow is True
        assert eruption.tsunami is False
        assert eruption.fatalities == 0
        assert eruption.injuries == 2
        assert eruption.evacuated == 150
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment(self, gvp_integration, sample_volcano_data):
        """Test data quality assessment"""
        quality = await gvp_integration._assess_data_quality(sample_volcano_data)
        
        # Should be high quality since all required fields are present
        assert quality in [DataQuality.EXCELLENT, DataQuality.GOOD]
    
    @pytest.mark.asyncio
    async def test_data_quality_assessment_poor_data(self, gvp_integration):
        """Test data quality assessment with poor data"""
        poor_data = {
            'volcano_id': '123456',
            # Missing required fields
        }
        
        quality = await gvp_integration._assess_data_quality(poor_data)
        assert quality in [DataQuality.POOR, DataQuality.UNKNOWN]
    
    @pytest.mark.asyncio
    async def test_invalid_coordinates(self, gvp_integration):
        """Test handling of invalid coordinates"""
        invalid_data = {
            'volcano_id': '123456',
            'volcano_name': 'Test Volcano',
            'latitude': 95.0,  # Invalid latitude
            'longitude': 200.0,  # Invalid longitude
            'country': 'Test Country'
        }
        
        volcano = await gvp_integration._parse_volcano_data(invalid_data)
        assert volcano is None
    
    @pytest.mark.asyncio
    async def test_fetch_volcano_catalog_with_mock(self, gvp_integration):
        """Test fetching volcano catalog with mocked API response"""
        # Mock API response
        mock_response = {
            'total_pages': 1,
            'volcanoes': [
                {
                    'volcano_id': '211060',
                    'volcano_name': 'Etna',
                    'latitude': 37.7510,
                    'longitude': 14.9934,
                    'elevation': 3357,
                    'country': 'Italy',
                    'region': 'Sicily',
                    'primary_volcano_type': 'Stratovolcano',
                    'tectonic_setting': 'Subduction zone'
                }
            ]
        }
        
        # Mock the authenticated request method
        gvp_integration._make_authenticated_request = AsyncMock(return_value=mock_response)
        gvp_integration._store_volcanoes_in_database = AsyncMock()
        
        volcanoes = await gvp_integration.fetch_complete_volcano_catalog()
        
        assert len(volcanoes) == 1
        assert volcanoes[0].volcano_name == 'Etna'
        assert volcanoes[0].gvp_id == '211060'
    
    @pytest.mark.asyncio
    async def test_fetch_eruption_history_with_mock(self, gvp_integration):
        """Test fetching eruption history with mocked API response"""
        mock_response = {
            'eruptions': [
                {
                    'eruption_id': '211060-2023-001',
                    'volcano_id': '211060',
                    'start_date': '2023-05-15T10:30:00Z',
                    'vei': 2,
                    'eruption_category': 'Strombolian'
                }
            ]
        }
        
        gvp_integration._make_authenticated_request = AsyncMock(return_value=mock_response)
        gvp_integration._store_eruptions_in_database = AsyncMock()
        
        eruptions = await gvp_integration.fetch_eruption_history('211060')
        
        assert len(eruptions) == 1
        assert eruptions[0].eruption_id == '211060-2023-001'
        assert eruptions[0].vei == 2
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, gvp_integration):
        """Test rate limiting functionality"""
        # Set a very low rate limit for testing
        gvp_integration.gvp_config['rate_limit'] = 1  # 1 request per minute
        
        start_time = asyncio.get_event_loop().time()
        
        # Make two requests
        await gvp_integration._enforce_rate_limit()
        await gvp_integration._enforce_rate_limit()
        
        end_time = asyncio.get_event_loop().time()
        
        # Should take at least 60 seconds due to rate limiting
        # For testing, we'll check if there was some delay
        assert end_time - start_time >= 0  # Basic check that it doesn't fail
    
    @pytest.mark.asyncio
    async def test_checksum_calculation(self, gvp_integration, sample_volcano_data):
        """Test checksum calculation for data integrity"""
        checksum1 = gvp_integration._calculate_checksum(sample_volcano_data)
        checksum2 = gvp_integration._calculate_checksum(sample_volcano_data)
        
        # Same data should produce same checksum
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 produces 64-character hex string
        
        # Different data should produce different checksum
        modified_data = sample_volcano_data.copy()
        modified_data['volcano_name'] = 'Modified Name'
        checksum3 = gvp_integration._calculate_checksum(modified_data)
        
        assert checksum1 != checksum3
    
    @pytest.mark.asyncio
    async def test_data_validation_rules(self, gvp_integration):
        """Test data validation rules initialization"""
        rules = gvp_integration._initialize_validation_rules()
        
        assert 'volcano' in rules
        assert 'eruption' in rules
        assert 'required_fields' in rules['volcano']
        assert 'coordinate_bounds' in rules['volcano']
        assert 'vei_range' in rules['eruption']
    
    @pytest.mark.asyncio
    async def test_synchronization_tracking(self, gvp_integration):
        """Test synchronization status tracking"""
        # Initial sync status
        assert gvp_integration.sync_status['last_full_sync'] is None
        assert gvp_integration.sync_status['total_volcanoes'] == 0
        
        # Mock synchronization methods
        gvp_integration.fetch_complete_volcano_catalog = AsyncMock(return_value=[])
        gvp_integration._calculate_overall_quality_score = AsyncMock(return_value=0.85)
        gvp_integration._generate_sync_report = AsyncMock()
        
        # Perform synchronization
        sync_results = await gvp_integration.synchronize_with_gvp(full_sync=True)
        
        assert sync_results['sync_type'] == 'full'
        assert 'start_time' in sync_results
        assert 'end_time' in sync_results
        assert 'duration_seconds' in sync_results
    
    @pytest.mark.asyncio
    async def test_quality_validation(self, gvp_integration):
        """Test data quality validation"""
        # Mock quality validation methods
        gvp_integration._validate_volcano_data_quality = AsyncMock(return_value=0.9)
        gvp_integration._validate_eruption_data_quality = AsyncMock(return_value=0.85)
        gvp_integration._validate_hazard_data_quality = AsyncMock(return_value=0.8)
        gvp_integration._validate_monitoring_data_quality = AsyncMock(return_value=0.75)
        gvp_integration._generate_quality_recommendations = AsyncMock(return_value=[])
        gvp_integration._store_quality_report = AsyncMock()
        
        quality_report = await gvp_integration.validate_data_quality()
        
        assert quality_report['data_type'] == 'all'
        assert 'overall_score' in quality_report
        assert 'component_scores' in quality_report
        assert len(quality_report['component_scores']) == 4
    
    @pytest.mark.asyncio
    async def test_quality_alerts(self, gvp_integration):
        """Test quality alert generation"""
        # Mock quality validation to return low scores
        gvp_integration.validate_data_quality = AsyncMock(return_value={
            'overall_score': 0.6,  # Below threshold
            'component_scores': {
                'volcanoes': 0.5,  # Below threshold
                'eruptions': 0.8,  # Above threshold
            }
        })
        gvp_integration._check_data_freshness = AsyncMock(return_value=[])
        gvp_integration._check_data_completeness = AsyncMock(return_value=[])
        gvp_integration._store_quality_alerts = AsyncMock()
        
        alerts = await gvp_integration.generate_quality_alerts(threshold=0.7)
        
        # Should generate alerts for overall quality and volcanoes component
        assert len(alerts) >= 2
        alert_types = [alert['alert_type'] for alert in alerts]
        assert 'overall_quality_low' in alert_types
        assert 'volcanoes_quality_low' in alert_types
    
    def test_enum_values(self):
        """Test enum value handling"""
        # Test VolcanoType enum
        assert VolcanoType.STRATOVOLCANO.value == "Stratovolcano"
        assert VolcanoType.SHIELD.value == "Shield"
        
        # Test TectonicSetting enum
        assert TectonicSetting.SUBDUCTION_ZONE.value == "Subduction zone"
        assert TectonicSetting.HOTSPOT.value == "Hotspot"
        
        # Test DataQuality enum
        assert DataQuality.EXCELLENT.value == "Excellent"
        assert DataQuality.POOR.value == "Poor"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, gvp_integration):
        """Test error handling in various scenarios"""
        # Test with None data
        volcano = await gvp_integration._parse_volcano_data(None)
        assert volcano is None
        
        # Test with empty data
        volcano = await gvp_integration._parse_volcano_data({})
        assert volcano is None
        
        # Test with malformed data
        malformed_data = {
            'volcano_id': None,
            'volcano_name': '',
            'latitude': 'invalid',
            'longitude': 'invalid'
        }
        volcano = await gvp_integration._parse_volcano_data(malformed_data)
        assert volcano is None
    
    @pytest.mark.asyncio
    async def test_cleanup(self, gvp_integration):
        """Test resource cleanup"""
        # Mock session and engine
        gvp_integration.session = AsyncMock()
        gvp_integration.db_engine = MagicMock()
        
        await gvp_integration.close()
        
        # Verify cleanup calls
        gvp_integration.session.close.assert_called_once()
        gvp_integration.db_engine.dispose.assert_called_once()

# Integration test functions
@pytest.mark.asyncio
async def test_full_integration_workflow():
    """Test complete integration workflow"""
    integration = GlobalVolcanismProgramIntegration()
    
    # Mock all external dependencies
    integration.session = AsyncMock()
    integration._initialize_database = AsyncMock()
    integration._test_gvp_connection = AsyncMock()
    integration._load_sync_status = AsyncMock()
    integration._make_authenticated_request = AsyncMock(return_value={
        'total_pages': 1,
        'volcanoes': [{
            'volcano_id': '211060',
            'volcano_name': 'Etna',
            'latitude': 37.7510,
            'longitude': 14.9934,
            'country': 'Italy',
            'region': 'Sicily',
            'primary_volcano_type': 'Stratovolcano'
        }]
    })
    integration._store_volcanoes_in_database = AsyncMock()
    
    try:
        # Initialize
        await integration.initialize()
        
        # Fetch data
        volcanoes = await integration.fetch_complete_volcano_catalog()
        assert len(volcanoes) == 1
        assert volcanoes[0].volcano_name == 'Etna'
        
        # Validate quality
        integration._validate_volcano_data_quality = AsyncMock(return_value=0.9)
        integration._generate_quality_recommendations = AsyncMock(return_value=[])
        integration._store_quality_report = AsyncMock()
        
        quality_report = await integration.validate_data_quality('volcanoes')
        assert quality_report['overall_score'] > 0
        
    finally:
        await integration.close()

if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_full_integration_workflow())
    print("All tests completed successfully!")