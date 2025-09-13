"""
Comprehensive tests for the data transformation and standardization pipeline
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os

from backend.core.data_transformation import (
    DataTransformationPipeline,
    CoordinateTransformer,
    TemporalSynchronizer,
    UnitConverter,
    MetadataHarmonizer,
    InterpolationEngine,
    AggregationEngine,
    ExportManager
)
from backend.core.data_transformation.schemas import (
    StandardizedDataFormat,
    DataType,
    CoordinateSystem,
    TemporalResolution,
    TransformationConfig,
    Point3D,
    QualityMetrics
)
from backend.core.data_transformation.export_manager import ExportConfig
from backend.core.data_transformation.aggregation_engine import AggregationConfig


class TestDataTransformationPipeline:
    """Test suite for the complete data transformation pipeline"""
    
    @pytest.fixture
    def sample_raw_data(self) -> List[Dict[str, Any]]:
        """Create sample raw geological data"""
        return [
            {
                'id': 'seismic_001',
                'timestamp': '2024-01-15T10:30:00Z',
                'latitude': 40.7128,
                'longitude': -74.0060,
                'elevation': 100.0,
                'amplitude': 0.005,
                'frequency': 2.5,
                'magnitude': 3.2,
                'network': 'US',
                'station': 'NYC01'
            },
            {
                'id': 'seismic_002',
                'timestamp': '2024-01-15T10:31:00Z',
                'latitude': 40.7130,
                'longitude': -74.0058,
                'elevation': 98.0,
                'amplitude': 0.008,
                'frequency': 3.1,
                'magnitude': 3.5,
                'network': 'US',
                'station': 'NYC02'
            },
            {
                'id': 'seismic_003',
                'timestamp': '2024-01-15T10:32:00Z',
                'latitude': 40.7125,
                'longitude': -74.0062,
                'elevation': 102.0,
                'amplitude': 0.003,
                'frequency': 1.8,
                'magnitude': 2.9,
                'network': 'US',
                'station': 'NYC03'
            }
        ]
    
    @pytest.fixture
    def pipeline(self) -> DataTransformationPipeline:
        """Create pipeline instance"""
        config = TransformationConfig(
            target_coordinate_system=CoordinateSystem.WGS84,
            target_temporal_resolution=TemporalResolution.MINUTE,
            target_timezone="UTC",
            interpolation_method="linear",
            aggregation_method="mean",
            fill_gaps=True,
            max_gap_duration=3600.0
        )
        return DataTransformationPipeline(config)
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_processing(self, pipeline, sample_raw_data):
        """Test complete pipeline processing"""
        result = await pipeline.process_data(
            sample_raw_data,
            source_name="IRIS",
            data_type=DataType.SEISMIC
        )
        
        assert result.success
        assert result.processed_count == len(sample_raw_data)
        assert len(result.processed_data) == len(sample_raw_data)
        assert result.processing_time_seconds > 0
        assert len(result.transformations_applied) > 0
        
        # Check that data is properly standardized
        for data in result.processed_data:
            assert isinstance(data, StandardizedDataFormat)
            assert data.data_type == DataType.SEISMIC
            assert data.source == "IRIS"
            assert data.timestamp is not None
            assert data.location is not None
            assert len(data.values) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_with_empty_data(self, pipeline):
        """Test pipeline with empty data"""
        result = await pipeline.process_data(
            [],
            source_name="TEST",
            data_type=DataType.SEISMIC
        )
        
        assert not result.success
        assert result.processed_count == 0
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, pipeline):
        """Test pipeline error handling with malformed data"""
        malformed_data = [
            {'invalid': 'data'},
            {'no_timestamp': True},
            {'timestamp': 'invalid_format', 'latitude': 'not_a_number'}
        ]
        
        result = await pipeline.process_data(
            malformed_data,
            source_name="TEST",
            data_type=DataType.SEISMIC
        )
        
        # Pipeline should handle errors gracefully
        assert len(result.warnings) > 0 or len(result.errors) > 0


class TestCoordinateTransformer:
    """Test suite for coordinate transformation"""
    
    @pytest.fixture
    def transformer(self) -> CoordinateTransformer:
        return CoordinateTransformer()
    
    def test_wgs84_to_utm_transformation(self, transformer):
        """Test WGS84 to UTM coordinate transformation"""
        # New York coordinates
        coordinates = np.array([[40.7128, -74.0060, 100.0]])
        
        result = transformer.transform_coordinates(
            coordinates,
            CoordinateSystem.WGS84,
            CoordinateSystem.UTM_NORTH
        )
        
        assert result.transformed_coordinates.shape == coordinates.shape
        assert result.transformation_accuracy > 0.9
        assert result.source_crs == "EPSG:4326"
        assert "326" in result.target_crs  # UTM zone
    
    def test_utm_zone_determination(self, transformer):
        """Test UTM zone determination"""
        # Test various locations
        test_cases = [
            (-74.0060, 40.7128),  # New York
            (2.3522, 48.8566),    # Paris
            (139.6917, 35.6895),  # Tokyo
        ]
        
        for lon, lat in test_cases:
            utm_zone = transformer.get_utm_zone(lon, lat)
            assert utm_zone.startswith("EPSG:326")
            assert len(utm_zone) >= 9  # EPSG:32601 minimum
    
    def test_coordinate_uncertainty_propagation(self, transformer):
        """Test uncertainty propagation through coordinate transformation"""
        coordinates = np.array([[40.7128, -74.0060]])
        uncertainty = np.array([[0.001, 0.001]])  # 1mm uncertainty
        
        result = transformer.transform_coordinates(
            coordinates,
            CoordinateSystem.WGS84,
            CoordinateSystem.UTM_NORTH,
            uncertainty=uncertainty
        )
        
        assert result.uncertainty_propagated is not None
        assert result.uncertainty_propagated.shape == uncertainty.shape
    
    def test_batch_coordinate_transformation(self, transformer):
        """Test batch coordinate transformation"""
        data_list = []
        for i in range(5):
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060 + i*0.001, 'y': 40.7128 + i*0.001, 'z': 100.0},
                coordinate_system=CoordinateSystem.WGS84,
                timestamp=datetime.now(),
                values={'amplitude': 0.005},
                units={'amplitude': 'm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        transformed_data = transformer.batch_transform_coordinates(
            data_list, CoordinateSystem.UTM_NORTH
        )
        
        assert len(transformed_data) == len(data_list)
        for data in transformed_data:
            assert data.coordinate_system == CoordinateSystem.UTM_NORTH


class TestTemporalSynchronizer:
    """Test suite for temporal synchronization"""
    
    @pytest.fixture
    def synchronizer(self) -> TemporalSynchronizer:
        return TemporalSynchronizer()
    
    def test_timezone_standardization(self, synchronizer):
        """Test timezone standardization"""
        # Test various timezone conversions
        test_time = datetime(2024, 1, 15, 10, 30, 0)
        
        # Convert from EST to UTC
        utc_time = synchronizer.standardize_timezone(
            test_time, source_timezone="US/Eastern", target_timezone="UTC"
        )
        
        assert utc_time.tzinfo is not None
        assert utc_time != test_time  # Should be different due to timezone
    
    def test_time_grid_creation(self, synchronizer):
        """Test time grid creation"""
        start_time = datetime(2024, 1, 15, 10, 0, 0)
        end_time = datetime(2024, 1, 15, 11, 0, 0)
        
        time_grid = synchronizer.create_time_grid(
            start_time, end_time, TemporalResolution.MINUTE
        )
        
        assert len(time_grid) == 61  # 0 to 60 minutes inclusive
        assert time_grid[0] == start_time
        assert time_grid[-1] <= end_time
    
    def test_temporal_synchronization(self, synchronizer):
        """Test temporal data synchronization"""
        # Create test data with irregular timestamps
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(5):
            # Irregular intervals: 0, 37, 83, 142, 195 seconds
            offset_seconds = i * 37 + i * i * 3
            timestamp = base_time + timedelta(seconds=offset_seconds)
            
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
                timestamp=timestamp,
                values={'amplitude': 0.005 + i * 0.001},
                units={'amplitude': 'm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        result = synchronizer.synchronize_data_formats(
            data_list, TemporalResolution.MINUTE
        )
        
        assert result.synchronization_accuracy > 0
        assert len(result.synchronized_data) >= len(data_list)


class TestUnitConverter:
    """Test suite for unit conversion"""
    
    @pytest.fixture
    def converter(self) -> UnitConverter:
        return UnitConverter()
    
    def test_basic_unit_conversion(self, converter):
        """Test basic unit conversions"""
        # Length conversions
        assert converter.convert_value(1000, "m", "km") == 1.0
        assert converter.convert_value(1, "km", "m") == 1000.0
        
        # Temperature conversions
        celsius_value = converter.convert_value(273.15, "K", "C")
        assert abs(celsius_value - 0.0) < 0.01
        
        # Pressure conversions
        assert converter.convert_value(1, "bar", "Pa") == 100000.0
    
    def test_unit_normalization(self, converter):
        """Test unit name normalization"""
        assert converter.normalize_unit_name("Meters") == "meters"
        assert converter.normalize_unit_name("  km/h  ") == "km/h"
        assert converter.normalize_unit_name("degrees") == "degrees"
    
    def test_data_format_unit_conversion(self, converter):
        """Test unit conversion in standardized data format"""
        data = StandardizedDataFormat(
            data_id="test_001",
            data_type=DataType.SEISMIC,
            location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
            timestamp=datetime.now(),
            values={'velocity': 1000.0, 'displacement': 0.5},
            units={'velocity': 'mm/s', 'displacement': 'km'},
            source='TEST'
        )
        
        standardized = converter.standardize_data_units(data)
        
        # Check that units were converted to standard geological units
        assert standardized.units['velocity'] == 'm/s'
        assert standardized.units['displacement'] == 'm'
        assert standardized.values['velocity'] == 0.001  # 1000 mm/s = 0.001 m/s
        assert standardized.values['displacement'] == 500.0  # 0.5 km = 500 m
    
    def test_batch_unit_standardization(self, converter):
        """Test batch unit standardization"""
        data_list = []
        for i in range(3):
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
                timestamp=datetime.now(),
                values={'amplitude': float(i + 1)},
                units={'amplitude': 'mm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        standardized_data = converter.batch_standardize_units(data_list)
        
        assert len(standardized_data) == len(data_list)
        for data in standardized_data:
            assert data.units['amplitude'] == 'm/s'


class TestMetadataHarmonizer:
    """Test suite for metadata harmonization"""
    
    @pytest.fixture
    def harmonizer(self) -> MetadataHarmonizer:
        return MetadataHarmonizer()
    
    def test_iris_metadata_harmonization(self, harmonizer):
        """Test IRIS metadata harmonization"""
        iris_metadata = {
            'network': 'US',
            'station': 'NYC01',
            'channel': 'BHZ',
            'starttime': '2024-01-15T10:30:00Z',
            'sampling_rate': 40.0,
            'npts': 4000
        }
        
        result = harmonizer.harmonize_metadata(iris_metadata, "IRIS")
        
        assert result.quality_score > 0.5
        assert 'network_code' in result.harmonized_metadata
        assert 'station_code' in result.harmonized_metadata
        assert result.harmonized_metadata['network_code'] == 'US'
        assert result.harmonized_metadata['station_code'] == 'NYC01'
    
    def test_usgs_metadata_harmonization(self, harmonizer):
        """Test USGS metadata harmonization"""
        usgs_metadata = {
            'eventid': 'us7000abcd',
            'time': '2024-01-15T10:30:00.000Z',
            'latitude': 40.7128,
            'longitude': -74.0060,
            'depth': 10.0,
            'mag': 3.2,
            'magType': 'ml'
        }
        
        result = harmonizer.harmonize_metadata(usgs_metadata, "USGS")
        
        assert result.quality_score > 0.8
        assert 'event_id' in result.harmonized_metadata
        assert 'magnitude' in result.harmonized_metadata
        assert result.harmonized_metadata['event_id'] == 'us7000abcd'
        assert result.harmonized_metadata['magnitude'] == 3.2
    
    def test_vocabulary_standardization(self, harmonizer):
        """Test controlled vocabulary standardization"""
        metadata = {
            'volcano_type': 'stratovolcano',
            'tectonic_setting': 'subduction zone',
            'rock_type': 'basalt'
        }
        
        standardized = harmonizer._apply_vocabulary_standardization(metadata, [])
        
        assert standardized['volcano_type'] == 'stratovolcano'
        assert standardized['tectonic_setting'] == 'subduction_zone'
        assert standardized['rock_type'] == 'basalt'


class TestInterpolationEngine:
    """Test suite for data interpolation"""
    
    @pytest.fixture
    def interpolation_engine(self) -> InterpolationEngine:
        return InterpolationEngine()
    
    def test_gap_detection(self, interpolation_engine):
        """Test temporal gap detection"""
        # Create data with gaps
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        # Regular data for first 3 points (1 minute intervals)
        for i in range(3):
            timestamp = base_time + timedelta(minutes=i)
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
                timestamp=timestamp,
                values={'amplitude': 0.005},
                units={'amplitude': 'm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        # Add point after 10-minute gap
        gap_timestamp = base_time + timedelta(minutes=13)
        gap_data = StandardizedDataFormat(
            data_id="test_gap",
            data_type=DataType.SEISMIC,
            location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
            timestamp=gap_timestamp,
            values={'amplitude': 0.008},
            units={'amplitude': 'm/s'},
            source='TEST'
        )
        data_list.append(gap_data)
        
        gaps = interpolation_engine.detect_gaps(
            data_list, TemporalResolution.MINUTE, gap_threshold_factor=2.0
        )
        
        assert len(gaps) == 1
        assert gaps[0].duration_seconds == 600.0  # 10 minutes
    
    def test_linear_interpolation(self, interpolation_engine):
        """Test linear interpolation for gap filling"""
        # Create data with a gap
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        # Point before gap
        data_before = StandardizedDataFormat(
            data_id="before",
            data_type=DataType.SEISMIC,
            location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
            timestamp=base_time,
            values={'amplitude': 0.005},
            units={'amplitude': 'm/s'},
            source='TEST'
        )
        data_list.append(data_before)
        
        # Point after gap (5 minutes later)
        data_after = StandardizedDataFormat(
            data_id="after",
            data_type=DataType.SEISMIC,
            location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
            timestamp=base_time + timedelta(minutes=5),
            values={'amplitude': 0.015},
            units={'amplitude': 'm/s'},
            source='TEST'
        )
        data_list.append(data_after)
        
        result = interpolation_engine.interpolate_temporal_gaps(
            data_list, method='linear', max_gap_duration=600.0
        )
        
        assert result.gaps_filled > 0
        assert result.interpolated_points > 0
        assert len(result.interpolated_data) > len(data_list)
        
        # Check interpolated values are between original values
        interpolated_amplitudes = [
            d.values['amplitude'] for d in result.interpolated_data 
            if 'temporally_interpolated' in d.quality_flags
        ]
        
        for amp in interpolated_amplitudes:
            assert 0.005 <= amp <= 0.015


class TestAggregationEngine:
    """Test suite for data aggregation"""
    
    @pytest.fixture
    def aggregation_engine(self) -> AggregationEngine:
        return AggregationEngine()
    
    def test_temporal_aggregation(self, aggregation_engine):
        """Test temporal data aggregation"""
        # Create high-resolution data (1-second intervals)
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(120):  # 2 minutes of data
            timestamp = base_time + timedelta(seconds=i)
            amplitude = 0.005 + 0.001 * np.sin(i * 0.1)  # Sinusoidal variation
            
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
                timestamp=timestamp,
                values={'amplitude': amplitude},
                units={'amplitude': 'm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        # Aggregate to 1-minute resolution
        config = AggregationConfig(
            target_resolution=TemporalResolution.MINUTE,
            aggregation_method='mean'
        )
        
        result = aggregation_engine.aggregate_temporal_data(data_list, config)
        
        assert result.aggregated_count < result.original_count
        assert result.aggregated_count == 2  # Should have 2 minute bins
        assert result.quality_preservation > 0.5
    
    def test_multi_resolution_dataset(self, aggregation_engine):
        """Test multi-resolution dataset creation"""
        # Create sample data
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(3600):  # 1 hour of second-resolution data
            timestamp = base_time + timedelta(seconds=i)
            data = StandardizedDataFormat(
                data_id=f"test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060, 'y': 40.7128, 'z': 100.0},
                timestamp=timestamp,
                values={'amplitude': 0.005},
                units={'amplitude': 'm/s'},
                source='TEST'
            )
            data_list.append(data)
        
        resolutions = [TemporalResolution.MINUTE, TemporalResolution.HOUR]
        multi_res_data = aggregation_engine.create_multi_resolution_dataset(
            data_list, resolutions
        )
        
        assert len(multi_res_data) == 2
        assert TemporalResolution.MINUTE in multi_res_data
        assert TemporalResolution.HOUR in multi_res_data
        
        # Minute resolution should have ~60 points, hour resolution should have 1
        assert len(multi_res_data[TemporalResolution.MINUTE]) == 60
        assert len(multi_res_data[TemporalResolution.HOUR]) == 1


class TestExportManager:
    """Test suite for data export"""
    
    @pytest.fixture
    def export_manager(self) -> ExportManager:
        return ExportManager()
    
    @pytest.fixture
    def sample_standardized_data(self) -> List[StandardizedDataFormat]:
        """Create sample standardized data for export testing"""
        data_list = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(5):
            data = StandardizedDataFormat(
                data_id=f"export_test_{i}",
                data_type=DataType.SEISMIC,
                location={'x': -74.0060 + i*0.001, 'y': 40.7128 + i*0.001, 'z': 100.0},
                timestamp=base_time + timedelta(minutes=i),
                values={'amplitude': 0.005 + i*0.001, 'frequency': 2.5 + i*0.1},
                units={'amplitude': 'm/s', 'frequency': 'Hz'},
                source='TEST',
                quality_metrics=QualityMetrics(
                    completeness=0.95,
                    accuracy=0.90,
                    consistency=0.85,
                    timeliness=0.98,
                    overall_score=0.92
                )
            )
            data_list.append(data)
        
        return data_list
    
    def test_json_export(self, export_manager, sample_standardized_data):
        """Test JSON export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.json")
            
            config = ExportConfig(
                format='json',
                output_path=output_path,
                include_metadata=True,
                include_quality_metrics=True
            )
            
            result = export_manager.export_data(sample_standardized_data, config)
            
            assert result.success
            assert len(result.output_files) == 1
            assert os.path.exists(result.output_files[0])
            assert result.records_exported == len(sample_standardized_data)
            assert result.file_size_bytes > 0
    
    def test_csv_export(self, export_manager, sample_standardized_data):
        """Test CSV export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.csv")
            
            config = ExportConfig(
                format='csv',
                output_path=output_path,
                include_uncertainty=True,
                precision=4
            )
            
            result = export_manager.export_data(sample_standardized_data, config)
            
            assert result.success
            assert len(result.output_files) == 1
            assert os.path.exists(result.output_files[0])
    
    def test_geojson_export(self, export_manager, sample_standardized_data):
        """Test GeoJSON export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_export.geojson")
            
            config = ExportConfig(
                format='geojson',
                output_path=output_path,
                coordinate_format='decimal'
            )
            
            result = export_manager.export_data(sample_standardized_data, config)
            
            assert result.success
            assert len(result.output_files) == 1
            assert os.path.exists(result.output_files[0])
    
    def test_batch_export(self, export_manager, sample_standardized_data):
        """Test batch export functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split data into groups
            data_groups = {
                'group1': sample_standardized_data[:2],
                'group2': sample_standardized_data[2:]
            }
            
            base_config = ExportConfig(
                format='json',
                output_path=os.path.join(temp_dir, "batch_export"),
                include_metadata=True
            )
            
            results = export_manager.batch_export(data_groups, base_config)
            
            assert len(results) == 2
            assert 'group1' in results
            assert 'group2' in results
            assert all(result.success for result in results.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])