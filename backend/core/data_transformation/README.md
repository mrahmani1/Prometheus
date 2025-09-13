# Data Transformation and Standardization Pipeline

This module provides a comprehensive data transformation and standardization pipeline for geological data from multiple sources. It systematically converts raw geological observations into standardized formats while preserving data quality and scientific accuracy.

## Overview

The Data Transformation Pipeline addresses the requirements for task 1.3 "Create Data Transformation and Standardization Pipeline" by implementing:

- **Standardized geological data formats and schemas**
- **Coordinate system transformation and projection handling**
- **Temporal synchronization and time zone standardization**
- **Unit conversion and measurement standardization**
- **Metadata harmonization across different data sources**
- **Data interpolation and gap-filling algorithms**
- **Data aggregation and downsampling for multi-resolution analysis**
- **Export capabilities for standardized geological data formats**

## Architecture

```
DataTransformationPipeline
├── CoordinateTransformer     # Spatial coordinate transformations
├── TemporalSynchronizer      # Time synchronization and alignment
├── UnitConverter            # Unit standardization and conversion
├── MetadataHarmonizer       # Metadata standardization across sources
├── InterpolationEngine      # Gap filling and data interpolation
├── AggregationEngine        # Multi-resolution data aggregation
└── ExportManager           # Multi-format data export
```

## Key Components

### 1. Standardized Data Schema

The `StandardizedDataFormat` class provides a unified structure for all geological data:

```python
@dataclass
class StandardizedDataFormat:
    data_id: str
    data_type: DataType
    location: Dict[str, float]
    coordinate_system: CoordinateSystem
    timestamp: datetime
    temporal_resolution: TemporalResolution
    values: Dict[str, Union[float, List[float], np.ndarray]]
    units: Dict[str, str]
    value_uncertainty: Optional[Dict[str, float]]
    source: str
    quality_metrics: Optional[QualityMetrics]
    processing_history: List[ProcessingHistory]
    metadata: Dict[str, Any]
```

### 2. Coordinate System Transformation

Handles transformations between different coordinate reference systems:

- **WGS84 ↔ UTM** conversions with automatic zone detection
- **Uncertainty propagation** through coordinate transformations
- **Batch processing** for large datasets
- **Accuracy estimation** for transformation quality assessment

```python
transformer = CoordinateTransformer()
result = transformer.transform_coordinates(
    coordinates, 
    source_crs=CoordinateSystem.WGS84,
    target_crs=CoordinateSystem.UTM_NORTH
)
```

### 3. Temporal Synchronization

Synchronizes data to common temporal grids:

- **Timezone standardization** to UTC or specified timezone
- **Regular time grid creation** for various resolutions
- **Temporal interpolation** for irregular observations
- **Gap detection and analysis**

```python
synchronizer = TemporalSynchronizer()
result = synchronizer.synchronize_data_formats(
    data_list,
    target_resolution=TemporalResolution.MINUTE,
    target_timezone="UTC"
)
```

### 4. Unit Conversion and Standardization

Converts measurements to standard geological units:

- **Comprehensive unit registry** covering geological measurements
- **Automatic unit inference** based on field names and data types
- **Uncertainty propagation** through unit conversions
- **Data type-specific standards** (seismic, volcanic, geochemical, etc.)

```python
converter = UnitConverter()
standardized_data = converter.standardize_data_units(data)
```

### 5. Metadata Harmonization

Standardizes metadata across different data sources:

- **Source-specific mappings** for IRIS, USGS, NASA, Copernicus, etc.
- **Controlled vocabulary standardization**
- **Field transformation and validation**
- **Quality scoring for harmonization success**

```python
harmonizer = MetadataHarmonizer()
result = harmonizer.harmonize_metadata(
    source_metadata, 
    source_name="IRIS",
    data_type=DataType.SEISMIC
)
```

### 6. Data Interpolation and Gap Filling

Fills temporal gaps using various interpolation methods:

- **Linear, cubic, and spline interpolation**
- **Gaussian Process interpolation** with uncertainty quantification
- **Kriging and RBF methods**
- **Gap detection and quality assessment**

```python
interpolator = InterpolationEngine()
result = interpolator.interpolate_temporal_gaps(
    data_list,
    method='gaussian_process',
    max_gap_duration=3600.0
)
```

### 7. Multi-Resolution Aggregation

Creates multi-resolution datasets for different analysis scales:

- **Temporal aggregation** to various resolutions (second to year)
- **Multiple aggregation methods** (mean, median, RMS, percentiles)
- **Quality-weighted aggregation**
- **Statistical preservation** and uncertainty propagation

```python
aggregator = AggregationEngine()
config = AggregationConfig(
    target_resolution=TemporalResolution.HOUR,
    aggregation_method='mean',
    weight_by_quality=True
)
result = aggregator.aggregate_temporal_data(data_list, config)
```

### 8. Multi-Format Export

Exports standardized data to various formats:

- **Scientific formats**: NetCDF, HDF5
- **Geospatial formats**: GeoJSON, Shapefile
- **General formats**: JSON, CSV, Excel, Parquet
- **Metadata preservation** and compression support

```python
exporter = ExportManager()
config = ExportConfig(
    format='netcdf',
    output_path='geological_data.nc',
    include_metadata=True,
    compression='gzip'
)
result = exporter.export_data(data_list, config)
```

## Supported Data Types

The pipeline supports various geological data types:

- **SEISMIC**: Earthquake waveforms, seismic events, ground motion
- **VOLCANIC**: Deformation, gas emissions, thermal anomalies, eruption data
- **GEOCHEMICAL**: Rock compositions, isotope ratios, mineral analyses
- **DEFORMATION**: GPS, MT-SAR, tilt, strain measurements
- **THERMAL**: Temperature, heat flow, thermal imagery
- **MAGNETIC**: Magnetic field measurements, aeromagnetic surveys
- **GRAVITY**: Gravity measurements, gravity anomalies

## Data Sources Integration

Pre-configured mappings for major geological data sources:

- **IRIS/EarthScope**: Seismic networks and earthquake catalogs
- **USGS**: Earthquake events, volcano monitoring, geological surveys
- **NASA EarthData**: Satellite observations and remote sensing
- **Copernicus**: Sentinel satellite data
- **ASF-Hyp3**: MT-SAR interferometry processing
- **EarthChem/GEOROC**: Geochemical databases
- **Global Volcanism Program**: Volcano catalogs and eruption data

## Usage Examples

### Basic Pipeline Processing

```python
import asyncio
from backend.core.data_transformation import (
    DataTransformationPipeline,
    TransformationConfig,
    DataType
)

async def process_seismic_data():
    # Configure pipeline
    config = TransformationConfig(
        target_coordinate_system=CoordinateSystem.WGS84,
        target_temporal_resolution=TemporalResolution.MINUTE,
        fill_gaps=True,
        interpolation_method='linear'
    )
    
    pipeline = DataTransformationPipeline(config)
    
    # Process raw IRIS data
    result = await pipeline.process_data(
        raw_iris_data,
        source_name="IRIS",
        data_type=DataType.SEISMIC
    )
    
    print(f"Processed {result.processed_count} records")
    print(f"Quality improvement: {result.quality_improvement:.2%}")
    
    return result.processed_data

# Run the pipeline
processed_data = asyncio.run(process_seismic_data())
```

### Multi-Source Data Integration

```python
async def integrate_multiple_sources():
    pipeline = DataTransformationPipeline()
    
    # Process different data sources
    sources = [
        (iris_data, "IRIS", DataType.SEISMIC),
        (usgs_data, "USGS", DataType.SEISMIC),
        (mtsar_data, "ASF_Hyp3", DataType.MTSAR),
        (geochem_data, "EarthChem", DataType.GEOCHEMICAL)
    ]
    
    all_processed_data = []
    
    for raw_data, source, data_type in sources:
        result = await pipeline.process_data(raw_data, source, data_type)
        if result.success:
            all_processed_data.extend(result.processed_data)
    
    return all_processed_data
```

### Export to Multiple Formats

```python
from backend.core.data_transformation import ExportManager, ExportConfig

def export_processed_data(processed_data):
    exporter = ExportManager()
    
    # Export to different formats
    formats = ['netcdf', 'geojson', 'csv', 'hdf5']
    
    for format_name in formats:
        config = ExportConfig(
            format=format_name,
            output_path=f'geological_data.{format_name}',
            include_metadata=True,
            include_uncertainty=True
        )
        
        result = exporter.export_data(processed_data, config)
        print(f"Exported to {format_name}: {result.file_size_bytes} bytes")
```

## Quality Assurance

The pipeline includes comprehensive quality assurance:

- **Data validation** at each processing step
- **Quality metrics** tracking (completeness, accuracy, consistency)
- **Processing history** preservation for full traceability
- **Error handling** with graceful degradation
- **Uncertainty quantification** and propagation

## Performance Optimization

- **Batch processing** for large datasets
- **Parallel processing** support for independent operations
- **Memory-efficient** streaming for large files
- **Caching** of coordinate transformations and metadata mappings
- **Compression** support for storage optimization

## Testing

Comprehensive test suite covering:

- **Unit tests** for individual components
- **Integration tests** for complete pipeline
- **Performance tests** for large datasets
- **Accuracy validation** against known geological datasets

Run tests with:
```bash
pytest backend/tests/test_data_transformation_pipeline.py -v
```

## Configuration

The pipeline is highly configurable through `TransformationConfig`:

```python
config = TransformationConfig(
    target_coordinate_system=CoordinateSystem.WGS84,
    target_temporal_resolution=TemporalResolution.MINUTE,
    target_timezone="UTC",
    interpolation_method="gaussian_process",
    aggregation_method="weighted_mean",
    quality_threshold=0.7,
    fill_gaps=True,
    max_gap_duration=3600.0,
    export_format="netcdf",
    include_uncertainty=True,
    preserve_raw_data=True
)
```

## Error Handling

Robust error handling ensures pipeline reliability:

- **Graceful degradation** when transformations fail
- **Detailed error reporting** with context
- **Partial processing** success for batch operations
- **Quality flagging** for problematic data
- **Recovery mechanisms** for common failure modes

## Future Enhancements

Planned improvements include:

- **Machine learning-based** gap filling and quality assessment
- **Real-time streaming** data processing
- **Distributed processing** for very large datasets
- **Advanced uncertainty quantification** methods
- **Integration with cloud storage** systems
- **Interactive visualization** of processing results

## Dependencies

Core dependencies:
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `scipy`: Scientific computing
- `pyproj`: Coordinate transformations
- `netCDF4`: NetCDF file handling
- `h5py`: HDF5 file handling
- `scikit-learn`: Machine learning algorithms
- `pydantic`: Data validation

## License

This module is part of the Prometheus AI Platform and follows the project's licensing terms.

## Contributing

Contributions are welcome! Please see the main project's contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## Support

For questions or issues:

1. Check the comprehensive test suite for usage examples
2. Review the example scripts in `backend/examples/`
3. Consult the main project documentation
4. Open an issue in the project repository