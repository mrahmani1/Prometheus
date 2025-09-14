"""
Test suite for GIS-enabled PostgreSQL operations
Comprehensive testing of spatial indexing, queries, transformations, and analysis
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

# Import the modules to test
from backend.core.gis_postgresql_operations import (
    GISPostgreSQLOperations,
    VolcanicSystem,
    SeismicEvent,
    DeformationMeasurement,
    MonitoringStation,
    GeologicalUnit,
    TectonicFeature,
    CoordinateSystem,
    SpatialIndexType,
    ExportFormat,
    SpatialQuery,
    SpatialAnalysisResult,
    SpatialUtils
)
from backend.core.database_integration import EnhancedDatabaseIntegrator


class TestGISPostgreSQLOperations:
    """Test suite for GIS PostgreSQL operations"""
    
    @pytest.fixture
    async def mock_database_integrator(self):
        """Create mock database integrator"""
        mock_db = Mock(spec=EnhancedDatabaseIntegrator)
        mock_session = AsyncMock()
        
        # Mock session context manager
        mock_db.get_postgresql_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_db.get_postgresql_session.return_value.__aexit__ = AsyncMock(return_value=None)
        
        return mock_db, mock_session
    
    @pytest.fixture
    async def gis_operations(self, mock_database_integrator):
        """Create GIS operations instance with mock database"""
        mock_db, mock_session = mock_database_integrator
        gis_ops = GISPostgreSQLOperations(mock_db)
        return gis_ops, mock_session
    
    @pytest.mark.asyncio
    async def test_initialize_spatial_extensions(self, gis_operations):
        """Test spatial extensions initialization"""
        gis_ops, mock_session = gis_operations
        
        # Mock successful initialization
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        
        result = await gis_ops.initialize_spatial_extensions()
        
        assert result is True
        
        # Verify PostGIS extensions were created
        expected_extensions = [
            "CREATE EXTENSION IF NOT EXISTS postgis",
            "CREATE EXTENSION IF NOT EXISTS postgis_topology",
            "CREATE EXTENSION IF NOT EXISTS postgis_raster",
            "CREATE EXTENSION IF NOT EXISTS postgis_sfcgal",
            "CREATE EXTENSION IF NOT EXISTS address_standardizer",
            "CREATE EXTENSION IF NOT EXISTS fuzzystrmatch",
            "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"
        ]
        
        # Check that extensions were created
        assert mock_session.execute.call_count >= len(expected_extensions)
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_spatial_indexes(self, gis_operations):
        """Test spatial index creation"""
        gis_ops, mock_session = gis_operations
        
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        
        # Test GIST index creation
        result = await gis_ops.create_spatial_indexes(
            table_name="volcanic_systems",
            geometry_columns=["location", "hazard_zone"],
            index_type=SpatialIndexType.GIST
        )
        
        assert result is True
        assert gis_ops.spatial_metrics['indexes_created'] == 2
        
        # Verify index creation calls
        assert mock_session.execute.call_count >= 4  # Drop + Create for each column
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_volcanic_systems_by_region(self, gis_operations):
        """Test querying volcanic systems by region"""
        gis_ops, mock_session = gis_operations
        
        # Mock query result
        mock_volcanic_system = Mock()
        mock_volcanic_system.id = "test-volcano-id"
        mock_volcanic_system.gvp_id = "123456"
        mock_volcanic_system.name = "Test Volcano"
        mock_volcanic_system.elevation = 2500.0
        mock_volcanic_system.volcano_type = "Stratovolcano"
        mock_volcanic_system.tectonic_setting = "Subduction zone"
        mock_volcanic_system.hazard_level = 3
        mock_volcanic_system.monitoring_status = "monitored"
        mock_volcanic_system.last_eruption = datetime(2020, 1, 1)
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [mock_volcanic_system]
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.scalar = Mock(side_effect=[-122.1944, 46.1912])  # Mock coordinates
        
        # Test region query
        region_wkt = "POLYGON((-123 45, -121 45, -121 47, -123 47, -123 45))"
        
        result = await gis_ops.query_volcanic_systems_by_region(
            region_geometry=region_wkt,
            tectonic_settings=["Subduction zone"],
            hazard_levels=[3, 4, 5]
        )
        
        assert len(result) == 1
        assert result[0]['name'] == "Test Volcano"
        assert result[0]['gvp_id'] == "123456"
        assert result[0]['tectonic_setting'] == "Subduction zone"
        assert gis_ops.spatial_metrics['queries_executed'] == 1
    
    @pytest.mark.asyncio
    async def test_query_seismic_events_spatial_temporal(self, gis_operations):
        """Test spatial-temporal seismic event queries"""
        gis_ops, mock_session = gis_operations
        
        # Mock seismic event
        mock_event = Mock()
        mock_event.id = "test-event-id"
        mock_event.origin_time = datetime(2024, 1, 15, 10, 30, 0)
        mock_event.magnitude = 4.2
        mock_event.magnitude_type = "ML"
        mock_event.depth = 8.5
        mock_event.event_type = "earthquake"
        mock_event.event_class = "VT"
        mock_event.location_uncertainty = 1.2
        mock_event.station_count = 12
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [mock_event]
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.scalar = Mock(side_effect=[-122.1944, 46.1912, -122.1944, 46.1912, 8.5])
        
        # Test spatial-temporal query
        spatial_bounds = {
            'min_lon': -123.0,
            'max_lon': -121.0,
            'min_lat': 45.0,
            'max_lat': 47.0
        }
        
        time_range = (
            datetime(2024, 1, 1),
            datetime(2024, 1, 31)
        )
        
        result = await gis_ops.query_seismic_events_spatial_temporal(
            spatial_bounds=spatial_bounds,
            time_range=time_range,
            magnitude_range=(3.0, 6.0),
            depth_range=(0.0, 20.0),
            event_types=["earthquake"]
        )
        
        assert len(result) == 1
        assert result[0]['magnitude'] == 4.2
        assert result[0]['event_type'] == "earthquake"
        assert gis_ops.spatial_metrics['queries_executed'] == 1
    
    @pytest.mark.asyncio
    async def test_query_monitoring_network_coverage(self, gis_operations):
        """Test monitoring network coverage analysis"""
        gis_ops, mock_session = gis_operations
        
        # Mock volcanic system
        mock_volcano = Mock()
        mock_volcano.id = "test-volcano-id"
        mock_volcano.name = "Test Volcano"
        
        # Mock monitoring stations
        mock_station = Mock()
        mock_station.id = "test-station-id"
        mock_station.station_code = "TEST01"
        mock_station.station_name = "Test Station 1"
        mock_station.station_type = "seismic"
        mock_station.operational_status = "operational"
        mock_station.real_time_availability = True
        
        # Mock query results
        volcano_result = Mock()
        volcano_result.scalar_one_or_none.return_value = mock_volcano
        
        stations_result = Mock()
        stations_result.scalars.return_value.all.return_value = [mock_station]
        
        coverage_result = Mock()
        coverage_result.scalar.return_value = '{"type":"Polygon","coordinates":[[[-122.2,-46.1],[-122.1,-46.1],[-122.1,-46.2],[-122.2,-46.2],[-122.2,-46.1]]]}'
        
        mock_session.execute = AsyncMock(side_effect=[volcano_result, stations_result, coverage_result])
        mock_session.scalar = Mock(side_effect=["POINT(-122.1944 46.1912)", -122.1944, 46.1912, 2500.0])
        
        result = await gis_ops.query_monitoring_network_coverage(
            volcanic_system_id="test-volcano-id",
            station_types=["seismic"],
            operational_only=True
        )
        
        assert result['volcano_name'] == "Test Volcano"
        assert result['station_count'] == 1
        assert len(result['stations']) == 1
        assert result['stations'][0]['code'] == "TEST01"
        assert result['coverage_area'] is not None
    
    @pytest.mark.asyncio
    async def test_transform_coordinates(self, gis_operations):
        """Test coordinate system transformations"""
        gis_ops, mock_session = gis_operations
        
        # Mock transformation result
        mock_result = Mock()
        mock_result.scalar.return_value = "POINT(500000 5000000)"
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        geometries = ["POINT(-122.1944 46.1912)"]
        
        result = await gis_ops.transform_coordinates(
            geometries=geometries,
            source_crs=4326,
            target_crs=32610  # UTM Zone 10N
        )
        
        assert len(result) == 1
        assert result[0] == "POINT(500000 5000000)"
        assert gis_ops.spatial_metrics['transformations_performed'] == 1
    
    @pytest.mark.asyncio
    async def test_get_optimal_utm_zone(self, gis_operations):
        """Test optimal UTM zone calculation"""
        gis_ops, mock_session = gis_operations
        
        # Test various locations
        test_cases = [
            (-122.1944, 46.1912, 32610),  # Washington State -> UTM 10N
            (-74.0060, 40.7128, 32618),   # New York -> UTM 18N
            (2.3522, 48.8566, 32631),     # Paris -> UTM 31N
            (-58.3816, -34.6037, 32721),  # Buenos Aires -> UTM 21S
        ]
        
        for lon, lat, expected_utm in test_cases:
            result = await gis_ops.get_optimal_utm_zone(lon, lat)
            assert result == expected_utm
    
    @pytest.mark.asyncio
    async def test_validate_coordinate_system(self, gis_operations):
        """Test coordinate system validation"""
        gis_ops, mock_session = gis_operations
        
        # Mock valid SRID
        mock_result = Mock()
        mock_srs_info = Mock()
        mock_srs_info.srid = 4326
        mock_srs_info.auth_name = "EPSG"
        mock_srs_info.auth_srid = 4326
        mock_srs_info.srtext = "GEOGCS[\"WGS 84\"...]"
        mock_srs_info.proj4text = "+proj=longlat +datum=WGS84 +no_defs"
        
        mock_result.fetchone.return_value = mock_srs_info
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await gis_ops.validate_coordinate_system(4326)
        
        assert result['valid'] is True
        assert result['srid'] == 4326
        assert result['authority'] == "EPSG"
    
    @pytest.mark.asyncio
    async def test_validate_geometry_quality(self, gis_operations):
        """Test geometry quality validation"""
        gis_ops, mock_session = gis_operations
        
        # Mock validation results
        mock_results = [
            Mock(scalar=Mock(return_value=5)),    # invalid_count
            Mock(scalar=Mock(return_value=2)),    # empty_count
            Mock(scalar=Mock(return_value=1)),    # null_count
            Mock(fetchall=Mock(return_value=[Mock(srid=4326, count=100)])),  # srid_distribution
            Mock(fetchall=Mock(return_value=[Mock(geom_type='ST_Point', count=100)])),  # type_distribution
            Mock(fetchone=Mock(return_value=Mock(min_x=-180, min_y=-90, max_x=180, max_y=90))),  # bbox
            Mock(scalar=Mock(return_value=108))   # total_count
        ]
        
        mock_session.execute = AsyncMock(side_effect=mock_results)
        
        result = await gis_ops.validate_geometry_quality("test_table", "geometry")
        
        assert result['invalid_geometries'] == 5
        assert result['empty_geometries'] == 2
        assert result['null_geometries'] == 1
        assert result['total_records'] == 108
        assert result['quality_score'] == 92.59  # (100/108) * 100
        assert len(result['srid_distribution']) == 1
        assert len(result['geometry_types']) == 1
    
    @pytest.mark.asyncio
    async def test_repair_invalid_geometries(self, gis_operations):
        """Test geometry repair functionality"""
        gis_ops, mock_session = gis_operations
        
        # Mock repair process
        invalid_ids_result = Mock()
        invalid_ids_result.fetchall.return_value = [("id1",), ("id2",), ("id3",)]
        
        still_invalid_result = Mock()
        still_invalid_result.scalar.return_value = 1  # One geometry couldn't be repaired
        
        mock_session.execute = AsyncMock(side_effect=[invalid_ids_result, AsyncMock(), still_invalid_result])
        mock_session.commit = AsyncMock()
        
        result = await gis_ops.repair_invalid_geometries("test_table", "geometry", "id")
        
        assert result['invalid_count'] == 3
        assert result['repaired_count'] == 2
        assert result['still_invalid_count'] == 1
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_perform_spatial_clustering(self, gis_operations):
        """Test spatial clustering analysis"""
        gis_ops, mock_session = gis_operations
        
        # Mock clustering results
        mock_clusters = [
            Mock(
                cluster_id=0,
                point_count=15,
                cluster_center="POINT(-122.1944 46.1912)",
                cluster_hull="POLYGON((-122.2 46.1, -122.1 46.1, -122.1 46.2, -122.2 46.2, -122.2 46.1))",
                cluster_area=0.01
            ),
            Mock(
                cluster_id=1,
                point_count=8,
                cluster_center="POINT(-122.3 46.3)",
                cluster_hull="POLYGON((-122.35 46.25, -122.25 46.25, -122.25 46.35, -122.35 46.35, -122.35 46.25))",
                cluster_area=0.005
            )
        ]
        
        mock_result = Mock()
        mock_result.fetchall.return_value = mock_clusters
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        result = await gis_ops.perform_spatial_clustering(
            table_name="seismic_events",
            geometry_column="epicenter",
            cluster_distance=5000.0,  # 5km
            min_points=3
        )
        
        assert result['cluster_count'] == 2
        assert result['clustering_parameters']['distance_threshold'] == 5000.0
        assert result['clustering_parameters']['minimum_points'] == 3
        assert len(result['clusters']) == 2
        assert result['statistics']['mean_cluster_size'] == 11.5
        assert result['statistics']['total_clustered_points'] == 23
    
    @pytest.mark.asyncio
    async def test_calculate_spatial_statistics(self, gis_operations):
        """Test spatial statistics calculation"""
        gis_ops, mock_session = gis_operations
        
        # Mock statistics results
        basic_stats = Mock(
            feature_count=100,
            geometric_center="POINT(-122.1944 46.1912)",
            total_area=1000.0,
            total_perimeter=500.0,
            min_x=-123.0,
            min_y=45.0,
            max_x=-121.0,
            max_y=47.0
        )
        
        feature_stats = Mock(
            mean_area=10.0,
            stddev_area=2.5,
            min_area=1.0,
            max_area=50.0,
            mean_perimeter=15.0,
            stddev_perimeter=3.0
        )
        
        distribution_stats = Mock(
            mean_density=10.0,
            stddev_density=3.0,
            min_density=5.0,
            max_density=20.0
        )
        
        mock_session.execute = AsyncMock(side_effect=[
            Mock(fetchone=Mock(return_value=basic_stats)),
            Mock(fetchone=Mock(return_value=feature_stats)),
            Mock(fetchone=Mock(return_value=distribution_stats))
        ])
        
        result = await gis_ops.calculate_spatial_statistics(
            table_name="geological_units",
            geometry_column="geometry",
            analysis_type="comprehensive"
        )
        
        assert result['basic_statistics']['feature_count'] == 100
        assert result['basic_statistics']['total_area'] == 1000.0
        assert result['feature_statistics']['area']['mean'] == 10.0
        assert result['spatial_distribution']['mean_density'] == 10.0
    
    @pytest.mark.asyncio
    async def test_export_geospatial_data_geojson(self, gis_operations):
        """Test GeoJSON export functionality"""
        gis_ops, mock_session = gis_operations
        
        # Mock export data
        mock_row = Mock()
        mock_row._mapping = {
            'id': 'test-id',
            'name': 'Test Feature',
            'geometry_geojson': '{"type":"Point","coordinates":[-122.1944,46.1912]}',
            'elevation': 2500.0
        }
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row]
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result = await gis_ops.export_geospatial_data(
                table_name="volcanic_systems",
                export_format=ExportFormat.GEOJSON,
                output_path=output_path,
                geometry_column="location"
            )
            
            assert result['format'] == 'geojson'
            assert result['record_count'] == 1
            assert result['coordinate_system'] == 4326
            assert gis_ops.spatial_metrics['exports_completed'] == 1
            
            # Verify file was created and contains valid GeoJSON
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                geojson_data = json.load(f)
            
            assert geojson_data['type'] == 'FeatureCollection'
            assert len(geojson_data['features']) == 1
            assert geojson_data['features'][0]['geometry']['type'] == 'Point'
            
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    @pytest.mark.asyncio
    async def test_export_geospatial_data_kml(self, gis_operations):
        """Test KML export functionality"""
        gis_ops, mock_session = gis_operations
        
        # Mock export data
        mock_row = Mock()
        mock_row._mapping = {
            'id': 'test-id',
            'name': 'Test Volcano',
            'geometry_wkt': 'POINT(-122.1944 46.1912)',
            'elevation': 2500.0
        }
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row]
        
        mock_session.execute = AsyncMock(return_value=mock_result)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.kml', delete=False) as tmp_file:
            output_path = tmp_file.name
        
        try:
            result = await gis_ops.export_geospatial_data(
                table_name="volcanic_systems",
                export_format=ExportFormat.KML,
                output_path=output_path,
                geometry_column="location"
            )
            
            assert result['format'] == 'kml'
            assert result['record_count'] == 1
            assert gis_ops.spatial_metrics['exports_completed'] == 1
            
            # Verify file was created and contains valid KML
            assert os.path.exists(output_path)
            
            with open(output_path, 'r') as f:
                kml_content = f.read()
            
            assert '<?xml version="1.0" encoding="UTF-8"?>' in kml_content
            assert '<kml xmlns="http://www.opengis.net/kml/2.2">' in kml_content
            assert '<Placemark>' in kml_content
            assert '<coordinates>-122.1944,46.1912,0</coordinates>' in kml_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_get_spatial_metrics(self, gis_operations):
        """Test spatial metrics retrieval"""
        gis_ops, mock_session = gis_operations
        
        # Set some metrics
        gis_ops.spatial_metrics['queries_executed'] = 10
        gis_ops.spatial_metrics['transformations_performed'] = 5
        gis_ops.spatial_metrics['indexes_created'] = 3
        gis_ops.spatial_metrics['exports_completed'] = 2
        
        metrics = gis_ops.get_spatial_metrics()
        
        assert metrics['queries_executed'] == 10
        assert metrics['transformations_performed'] == 5
        assert metrics['indexes_created'] == 3
        assert metrics['exports_completed'] == 2


class TestSpatialUtils:
    """Test suite for spatial utility functions"""
    
    def test_create_point_from_coordinates(self):
        """Test point creation from coordinates"""
        # Test 2D point
        point_2d = SpatialUtils.create_point_from_coordinates(-122.1944, 46.1912)
        assert point_2d == "POINT(-122.1944 46.1912)"
        
        # Test 3D point
        point_3d = SpatialUtils.create_point_from_coordinates(-122.1944, 46.1912, 2500.0)
        assert point_3d == "POINT Z (-122.1944 46.1912 2500.0)"
    
    def test_create_polygon_from_bbox(self):
        """Test polygon creation from bounding box"""
        polygon = SpatialUtils.create_polygon_from_bbox(-123.0, 45.0, -121.0, 47.0)
        expected = "POLYGON((-123.0 45.0, -121.0 45.0, \n                           -121.0 47.0, -123.0 47.0, -123.0 45.0))"
        assert polygon == expected
    
    def test_validate_wkt_geometry(self):
        """Test WKT geometry validation"""
        # Valid geometries
        assert SpatialUtils.validate_wkt_geometry("POINT(-122.1944 46.1912)") is True
        assert SpatialUtils.validate_wkt_geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))") is True
        
        # Invalid geometries
        assert SpatialUtils.validate_wkt_geometry("INVALID WKT") is False
        assert SpatialUtils.validate_wkt_geometry("POINT(invalid coordinates)") is False
    
    def test_convert_geojson_to_wkt(self):
        """Test GeoJSON to WKT conversion"""
        geojson_point = {
            "type": "Point",
            "coordinates": [-122.1944, 46.1912]
        }
        
        wkt = SpatialUtils.convert_geojson_to_wkt(geojson_point)
        assert "POINT" in wkt
        assert "-122.1944" in wkt
        assert "46.1912" in wkt
        
        # Test invalid GeoJSON
        with pytest.raises(ValueError):
            SpatialUtils.convert_geojson_to_wkt({"invalid": "geojson"})
    
    def test_calculate_distance_meters(self):
        """Test distance calculation between points"""
        # Test known distance (approximately)
        point1 = (-122.1944, 46.1912)  # Mount St. Helens
        point2 = (-122.3, 46.2)        # Nearby point
        
        distance = SpatialUtils.calculate_distance_meters(point1, point2)
        
        # Should be approximately 11-12 km
        assert 10000 < distance < 15000
        
        # Test same point (should be 0)
        distance_same = SpatialUtils.calculate_distance_meters(point1, point1)
        assert distance_same == 0.0


class TestSQLAlchemyModels:
    """Test suite for SQLAlchemy model definitions"""
    
    def test_volcanic_system_model(self):
        """Test VolcanicSystem model structure"""
        # Test model has required attributes
        assert hasattr(VolcanicSystem, 'id')
        assert hasattr(VolcanicSystem, 'gvp_id')
        assert hasattr(VolcanicSystem, 'name')
        assert hasattr(VolcanicSystem, 'location')
        assert hasattr(VolcanicSystem, 'elevation')
        assert hasattr(VolcanicSystem, 'volcano_type')
        assert hasattr(VolcanicSystem, 'tectonic_setting')
        assert hasattr(VolcanicSystem, 'hazard_zone')
        assert hasattr(VolcanicSystem, 'monitoring_network')
        
        # Test relationships
        assert hasattr(VolcanicSystem, 'seismic_events')
        assert hasattr(VolcanicSystem, 'deformation_measurements')
        assert hasattr(VolcanicSystem, 'monitoring_stations')
    
    def test_seismic_event_model(self):
        """Test SeismicEvent model structure"""
        assert hasattr(SeismicEvent, 'id')
        assert hasattr(SeismicEvent, 'volcanic_system_id')
        assert hasattr(SeismicEvent, 'origin_time')
        assert hasattr(SeismicEvent, 'hypocenter')
        assert hasattr(SeismicEvent, 'epicenter')
        assert hasattr(SeismicEvent, 'magnitude')
        assert hasattr(SeismicEvent, 'depth')
        assert hasattr(SeismicEvent, 'event_type')
        assert hasattr(SeismicEvent, 'waveform_data')
        assert hasattr(SeismicEvent, 'moment_tensor')
    
    def test_deformation_measurement_model(self):
        """Test DeformationMeasurement model structure"""
        assert hasattr(DeformationMeasurement, 'id')
        assert hasattr(DeformationMeasurement, 'volcanic_system_id')
        assert hasattr(DeformationMeasurement, 'acquisition_time')
        assert hasattr(DeformationMeasurement, 'measurement_point')
        assert hasattr(DeformationMeasurement, 'measurement_area')
        assert hasattr(DeformationMeasurement, 'displacement_east')
        assert hasattr(DeformationMeasurement, 'displacement_north')
        assert hasattr(DeformationMeasurement, 'displacement_up')
        assert hasattr(DeformationMeasurement, 'line_of_sight_displacement')
        assert hasattr(DeformationMeasurement, 'coherence')
        assert hasattr(DeformationMeasurement, 'measurement_type')
        assert hasattr(DeformationMeasurement, 'satellite_platform')
    
    def test_monitoring_station_model(self):
        """Test MonitoringStation model structure"""
        assert hasattr(MonitoringStation, 'id')
        assert hasattr(MonitoringStation, 'volcanic_system_id')
        assert hasattr(MonitoringStation, 'station_code')
        assert hasattr(MonitoringStation, 'network_code')
        assert hasattr(MonitoringStation, 'station_name')
        assert hasattr(MonitoringStation, 'location')
        assert hasattr(MonitoringStation, 'station_type')
        assert hasattr(MonitoringStation, 'instrument_type')
        assert hasattr(MonitoringStation, 'operational_status')
        assert hasattr(MonitoringStation, 'real_time_availability')
    
    def test_geological_unit_model(self):
        """Test GeologicalUnit model structure"""
        assert hasattr(GeologicalUnit, 'id')
        assert hasattr(GeologicalUnit, 'unit_name')
        assert hasattr(GeologicalUnit, 'formation_name')
        assert hasattr(GeologicalUnit, 'geometry')
        assert hasattr(GeologicalUnit, 'representative_point')
        assert hasattr(GeologicalUnit, 'age_min')
        assert hasattr(GeologicalUnit, 'age_max')
        assert hasattr(GeologicalUnit, 'primary_lithology')
        assert hasattr(GeologicalUnit, 'geochemical_composition')
        assert hasattr(GeologicalUnit, 'structural_features')
    
    def test_tectonic_feature_model(self):
        """Test TectonicFeature model structure"""
        assert hasattr(TectonicFeature, 'id')
        assert hasattr(TectonicFeature, 'feature_name')
        assert hasattr(TectonicFeature, 'feature_type')
        assert hasattr(TectonicFeature, 'geometry')
        assert hasattr(TectonicFeature, 'zone_geometry')
        assert hasattr(TectonicFeature, 'movement_sense')
        assert hasattr(TectonicFeature, 'dip_direction')
        assert hasattr(TectonicFeature, 'dip_angle')
        assert hasattr(TectonicFeature, 'strike')
        assert hasattr(TectonicFeature, 'activity_status')


class TestEnumDefinitions:
    """Test suite for enum definitions"""
    
    def test_coordinate_system_enum(self):
        """Test CoordinateSystem enum values"""
        assert CoordinateSystem.WGS84.value == 4326
        assert CoordinateSystem.WEB_MERCATOR.value == 3857
        assert CoordinateSystem.UTM_NORTH.value == 32600
        assert CoordinateSystem.UTM_SOUTH.value == 32700
    
    def test_spatial_index_type_enum(self):
        """Test SpatialIndexType enum values"""
        assert SpatialIndexType.GIST.value == "GIST"
        assert SpatialIndexType.SPGIST.value == "SP-GIST"
        assert SpatialIndexType.BRIN.value == "BRIN"
        assert SpatialIndexType.BTREE.value == "BTREE"
    
    def test_export_format_enum(self):
        """Test ExportFormat enum values"""
        assert ExportFormat.GEOJSON.value == "geojson"
        assert ExportFormat.KML.value == "kml"
        assert ExportFormat.SHAPEFILE.value == "shapefile"
        assert ExportFormat.GPKG.value == "gpkg"
        assert ExportFormat.WKT.value == "wkt"
        assert ExportFormat.WKB.value == "wkb"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])