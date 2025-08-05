"""
Test suite for PostgreSQL with PostGIS Manager

Tests advanced spatial indexing, complex spatial-temporal queries,
materialized views, and stored procedures for geological calculations.
"""

import pytest
import asyncio
import asyncpg
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch
import json

from core.database.postgis_manager import (
    PostGISManager,
    GeologicalLocation,
    SeismicEvent,
    DeformationMeasurement,
    SpatialTemporalQuery
)

@pytest.fixture
async def db_manager():
    """Create PostGIS manager for testing"""
    # Use test database connection string
    connection_string = "postgresql://prometheus:password@localhost:5432/prometheus_test"
    manager = PostGISManager(connection_string)
    
    # Mock the connection pool for unit tests
    manager.pool = AsyncMock()
    
    yield manager
    
    if hasattr(manager, 'pool') and manager.pool:
        await manager.close()

@pytest.fixture
def sample_location():
    """Sample geological location for testing"""
    return GeologicalLocation(
        id=None,
        name="Test Volcano",
        latitude=46.1912,
        longitude=-122.1944,
        elevation=2549.0,
        depth=None,
        coordinate_system="WGS84"
    )

@pytest.fixture
def sample_seismic_event():
    """Sample seismic event for testing"""
    return SeismicEvent(
        id=None,
        event_time=datetime.now(),
        location_id=1,
        magnitude=3.2,
        depth=5000.0,
        event_type="volcanic",
        confidence=0.95,
        metadata={"source": "USGS", "quality": "A"}
    )

@pytest.fixture
def sample_deformation():
    """Sample deformation measurement for testing"""
    return DeformationMeasurement(
        id=None,
        measurement_time=datetime.now(),
        location_id=1,
        displacement_x=0.001,
        displacement_y=0.002,
        displacement_z=0.003,
        uncertainty_x=0.0001,
        uncertainty_y=0.0001,
        uncertainty_z=0.0001,
        measurement_type="GPS",
        reference_frame="ITRF2014"
    )

class TestPostGISManager:
    """Test cases for PostGIS manager functionality"""
    
    @pytest.mark.asyncio
    async def test_initialize_connection_pool(self, db_manager):
        """Test database connection pool initialization"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            await db_manager.initialize()
            
            mock_create_pool.assert_called_once()
            assert db_manager.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_create_geological_location(self, db_manager, sample_location):
        """Test creating geological location with spatial indexing"""
        # Mock database responses
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 123
        mock_conn.execute.return_value = None
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        location_id = await db_manager.create_geological_location(sample_location)
        
        assert location_id == 123
        assert mock_conn.fetchval.called
        assert mock_conn.execute.called  # UTM coordinate update
    
    @pytest.mark.asyncio
    async def test_insert_seismic_event(self, db_manager, sample_seismic_event):
        """Test inserting seismic event with spatial-temporal indexing"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 456
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        event_id = await db_manager.insert_seismic_event(sample_seismic_event)
        
        assert event_id == 456
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_insert_deformation_measurement(self, db_manager, sample_deformation):
        """Test inserting deformation measurement"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 789
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        measurement_id = await db_manager.insert_deformation_measurement(sample_deformation)
        
        assert measurement_id == 789
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_events_in_spacetime(self, db_manager):
        """Test spatial-temporal event search"""
        query_params = SpatialTemporalQuery(
            center_lat=46.1912,
            center_lon=-122.1944,
            spatial_radius=50000,
            time_center=datetime.now(),
            time_window=timedelta(days=30),
            event_types=["volcanic", "earthquake"]
        )
        
        # Mock database response
        mock_conn = AsyncMock()
        mock_events = [
            {
                'event_id': 1,
                'event_type': 'volcanic',
                'event_time': datetime.now(),
                'distance_from_center': 25000.0,
                'magnitude': 3.2,
                'depth': 5000.0,
                'location_name': 'Test Location'
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), event) for event in mock_events]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        events = await db_manager.find_events_in_spacetime(query_params)
        
        assert len(events) == 1
        assert events[0]['event_type'] == 'volcanic'
        assert events[0]['magnitude'] == 3.2
    
    @pytest.mark.asyncio
    async def test_calculate_geological_distance(self, db_manager):
        """Test geological distance calculation"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 15000.5
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        distance = await db_manager.calculate_geological_distance(1, 2, 'euclidean')
        
        assert distance == 15000.5
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_seismic_moment(self, db_manager):
        """Test seismic moment calculation"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1.26e15  # Example seismic moment
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        moment = await db_manager.calculate_seismic_moment(5.0, 'Mw')
        
        assert moment == 1.26e15
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_estimate_vei(self, db_manager):
        """Test Volcanic Explosivity Index estimation"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 4
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        vei = await db_manager.estimate_vei(1e8, 20000.0)
        
        assert vei == 4
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_calculate_hazard_probability(self, db_manager):
        """Test geological hazard probability calculation"""
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 0.15
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        probability = await db_manager.calculate_hazard_probability(
            46.1912, -122.1944, 'volcanic', timedelta(days=365)
        )
        
        assert probability == 0.15
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_seismic_activity_summary(self, db_manager):
        """Test seismic activity summary from materialized view"""
        mock_conn = AsyncMock()
        mock_summary = [
            {
                'activity_date': datetime.now().date(),
                'event_type': 'volcanic',
                'event_count': 5,
                'avg_magnitude': 3.2,
                'max_magnitude': 4.1,
                'min_magnitude': 2.8,
                'magnitude_stddev': 0.5,
                'avg_depth': 5500.0,
                'centroid_wkt': 'POINT(-122.1944 46.1912)',
                'envelope_wkt': 'POLYGON((-122.2 46.18, -122.19 46.18, -122.19 46.2, -122.2 46.2, -122.2 46.18))'
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), summary) for summary in mock_summary]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        summary = await db_manager.get_seismic_activity_summary(days_back=30)
        
        assert len(summary) == 1
        assert summary[0]['event_type'] == 'volcanic'
        assert summary[0]['event_count'] == 5
    
    @pytest.mark.asyncio
    async def test_get_deformation_analysis(self, db_manager):
        """Test deformation analysis from materialized view"""
        mock_conn = AsyncMock()
        mock_analysis = [
            {
                'location_id': 1,
                'location_name': 'Test Location',
                'location_wkt': 'POINT(-122.1944 46.1912)',
                'analysis_month': datetime.now().replace(day=1),
                'measurement_type': 'GPS',
                'measurement_count': 30,
                'avg_total_displacement': 0.005,
                'max_total_displacement': 0.012,
                'avg_displacement_x': 0.001,
                'avg_displacement_y': 0.002,
                'avg_displacement_z': 0.003,
                'stddev_displacement_x': 0.0005,
                'stddev_displacement_y': 0.0008,
                'stddev_displacement_z': 0.001
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), analysis) for analysis in mock_analysis]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        analysis = await db_manager.get_deformation_analysis(location_id=1, months_back=12)
        
        assert len(analysis) == 1
        assert analysis[0]['location_id'] == 1
        assert analysis[0]['measurement_type'] == 'GPS'
    
    @pytest.mark.asyncio
    async def test_get_thermal_anomalies(self, db_manager):
        """Test thermal anomaly detection from materialized view"""
        mock_conn = AsyncMock()
        mock_anomalies = [
            {
                'location_id': 1,
                'location_name': 'Hot Springs',
                'location_wkt': 'POINT(-122.1944 46.1912)',
                'analysis_week': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
                'measurement_method': 'satellite',
                'measurement_count': 7,
                'avg_temperature': 85.5,
                'max_temperature': 92.1,
                'min_temperature': 78.3,
                'temperature_stddev': 4.2,
                'anomaly_level': 'HIGH_ANOMALY'
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), anomaly) for anomaly in mock_anomalies]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        anomalies = await db_manager.get_thermal_anomalies(anomaly_level='HIGH_ANOMALY')
        
        assert len(anomalies) == 1
        assert anomalies[0]['anomaly_level'] == 'HIGH_ANOMALY'
        assert anomalies[0]['avg_temperature'] == 85.5
    
    @pytest.mark.asyncio
    async def test_get_geological_correlation(self, db_manager):
        """Test multi-parameter geological correlation"""
        mock_conn = AsyncMock()
        mock_correlation = [
            {
                'location_id': 1,
                'location_name': 'Monitoring Station',
                'location_wkt': 'POINT(-122.1944 46.1912)',
                'elevation': 2549.0,
                'depth': None,
                'analysis_month': datetime.now().replace(day=1),
                'seismic_event_count': 15,
                'avg_seismic_magnitude': 3.1,
                'deformation_measurement_count': 30,
                'avg_total_deformation': 0.008,
                'thermal_measurement_count': 7,
                'avg_temperature': 45.2,
                'geochemical_sample_count': 3
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), corr) for corr in mock_correlation]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        correlation = await db_manager.get_geological_correlation(location_id=1)
        
        assert len(correlation) == 1
        assert correlation[0]['seismic_event_count'] == 15
        assert correlation[0]['avg_seismic_magnitude'] == 3.1
    
    @pytest.mark.asyncio
    async def test_get_spatiotemporal_seismic_analysis(self, db_manager):
        """Test complex spatial-temporal seismic analysis"""
        mock_conn = AsyncMock()
        mock_analysis = [
            {
                'id': 1,
                'event_time': datetime.now(),
                'magnitude': 3.2,
                'depth': 5000.0,
                'event_type': 'volcanic',
                'location_name': 'Test Location',
                'location_wkt': 'POINT(-122.1944 46.1912)',
                'spatial_cluster_id': 1,
                'temporal_cluster_start': 1,
                'distance_to_previous': 2500.0,
                'hours_since_previous': 2.5,
                'magnitude_diff': 0.3
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), analysis) for analysis in mock_analysis]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        analysis = await db_manager.get_spatiotemporal_seismic_analysis(days_back=365)
        
        assert len(analysis) == 1
        assert analysis[0]['spatial_cluster_id'] == 1
        assert analysis[0]['distance_to_previous'] == 2500.0
    
    @pytest.mark.asyncio
    async def test_get_deformation_velocity_analysis(self, db_manager):
        """Test deformation velocity analysis"""
        mock_conn = AsyncMock()
        mock_velocity = [
            {
                'location_id': 1,
                'location_name': 'GPS Station',
                'location_wkt': 'POINT(-122.1944 46.1912)',
                'time1': datetime.now() - timedelta(days=1),
                'time2': datetime.now(),
                'days_between': 1.0,
                'velocity_x_mm_day': 0.5,
                'velocity_y_mm_day': 1.2,
                'velocity_z_mm_day': 0.8,
                'total_velocity_mm_day': 1.6
            }
        ]
        mock_conn.fetch.return_value = [type('Row', (), vel) for vel in mock_velocity]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        velocity = await db_manager.get_deformation_velocity_analysis(location_id=1, days_back=365)
        
        assert len(velocity) == 1
        assert velocity[0]['total_velocity_mm_day'] == 1.6
        assert velocity[0]['days_between'] == 1.0
    
    @pytest.mark.asyncio
    async def test_refresh_materialized_views(self, db_manager):
        """Test materialized view refresh"""
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        await db_manager.refresh_materialized_views()
        
        mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_spatial_query(self, db_manager):
        """Test custom spatial query execution"""
        mock_conn = AsyncMock()
        mock_results = [
            {'id': 1, 'name': 'Location 1'},
            {'id': 2, 'name': 'Location 2'}
        ]
        mock_conn.fetch.return_value = [type('Row', (), result) for result in mock_results]
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        query = "SELECT id, name FROM geological.locations WHERE ST_DWithin(geom, $1, $2)"
        params = ["POINT(-122.1944 46.1912)", 10000]
        
        results = await db_manager.execute_spatial_query(query, params)
        
        assert len(results) == 2
        assert results[0]['name'] == 'Location 1'
    
    @pytest.mark.asyncio
    async def test_get_database_statistics(self, db_manager):
        """Test database statistics retrieval"""
        mock_conn = AsyncMock()
        
        # Mock table counts
        mock_conn.fetchval.side_effect = [100, 500, 200, 150, 75]  # Table row counts
        
        # Mock index statistics
        mock_index_stats = [
            {
                'schemaname': 'geological',
                'tablename': 'seismic_events',
                'indexname': 'idx_seismic_events_time',
                'idx_tup_read': 1000,
                'idx_tup_fetch': 950
            }
        ]
        
        # Mock materialized view stats
        mock_mv_stats = [
            {
                'schemaname': 'geological',
                'matviewname': 'mv_seismic_activity_summary',
                'ispopulated': True
            }
        ]
        
        mock_conn.fetch.side_effect = [
            [type('Row', (), stat) for stat in mock_index_stats],
            [type('Row', (), stat) for stat in mock_mv_stats]
        ]
        
        db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        stats = await db_manager.get_database_statistics()
        
        assert 'geological.locations_count' in stats
        assert 'top_indexes' in stats
        assert 'materialized_views' in stats
        assert len(stats['top_indexes']) == 1
        assert len(stats['materialized_views']) == 1

class TestSpatialTemporalQueries:
    """Test cases for complex spatial-temporal queries"""
    
    def test_spatial_temporal_query_creation(self):
        """Test spatial-temporal query parameter creation"""
        query = SpatialTemporalQuery(
            center_lat=46.1912,
            center_lon=-122.1944,
            spatial_radius=50000,
            time_center=datetime.now(),
            time_window=timedelta(days=30),
            event_types=["volcanic", "earthquake"]
        )
        
        assert query.center_lat == 46.1912
        assert query.center_lon == -122.1944
        assert query.spatial_radius == 50000
        assert query.event_types == ["volcanic", "earthquake"]
    
    def test_spatial_temporal_query_defaults(self):
        """Test spatial-temporal query with default parameters"""
        query = SpatialTemporalQuery(
            center_lat=46.1912,
            center_lon=-122.1944,
            spatial_radius=50000,
            time_center=datetime.now(),
            time_window=timedelta(days=30)
        )
        
        assert query.event_types is None

class TestGeologicalDataModels:
    """Test cases for geological data models"""
    
    def test_geological_location_creation(self):
        """Test geological location data model"""
        location = GeologicalLocation(
            id=1,
            name="Test Volcano",
            latitude=46.1912,
            longitude=-122.1944,
            elevation=2549.0,
            depth=None,
            coordinate_system="WGS84"
        )
        
        assert location.name == "Test Volcano"
        assert location.latitude == 46.1912
        assert location.coordinate_system == "WGS84"
    
    def test_seismic_event_creation(self):
        """Test seismic event data model"""
        event = SeismicEvent(
            id=1,
            event_time=datetime.now(),
            location_id=1,
            magnitude=3.2,
            depth=5000.0,
            event_type="volcanic",
            confidence=0.95,
            metadata={"source": "USGS"}
        )
        
        assert event.magnitude == 3.2
        assert event.event_type == "volcanic"
        assert event.confidence == 0.95
        assert event.metadata["source"] == "USGS"
    
    def test_deformation_measurement_creation(self):
        """Test deformation measurement data model"""
        measurement = DeformationMeasurement(
            id=1,
            measurement_time=datetime.now(),
            location_id=1,
            displacement_x=0.001,
            displacement_y=0.002,
            displacement_z=0.003,
            measurement_type="GPS"
        )
        
        assert measurement.displacement_x == 0.001
        assert measurement.measurement_type == "GPS"
        assert measurement.reference_frame == "ITRF2014"  # Default value

# Integration test (requires actual database)
@pytest.mark.integration
class TestPostGISIntegration:
    """Integration tests requiring actual PostgreSQL database"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self):
        """Test complete workflow with real database"""
        # This test would require a real PostgreSQL database with PostGIS
        # and would test the complete workflow from location creation
        # to complex spatial-temporal analysis
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])