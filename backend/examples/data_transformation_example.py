"""
Comprehensive example demonstrating the data transformation and standardization pipeline

This example shows how to:
1. Process raw geological data from multiple sources
2. Apply coordinate transformations
3. Synchronize temporal data
4. Standardize units and metadata
5. Fill gaps with interpolation
6. Aggregate data to multiple resolutions
7. Export to various formats
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.core.data_transformation import (
    DataTransformationPipeline,
    TransformationConfig,
    ExportConfig,
    AggregationConfig
)
from backend.core.data_transformation.schemas import (
    DataType,
    CoordinateSystem,
    TemporalResolution
)


def create_sample_iris_data() -> List[Dict[str, Any]]:
    """Create sample IRIS seismic data"""
    logger.info("Creating sample IRIS seismic data...")
    
    data = []
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    # Simulate seismic stations around Mount St. Helens
    stations = [
        {'code': 'MSH01', 'lat': 46.1912, 'lon': -122.1944, 'elev': 1200},
        {'code': 'MSH02', 'lat': 46.2000, 'lon': -122.2000, 'elev': 1150},
        {'code': 'MSH03', 'lat': 46.1800, 'lon': -122.1800, 'elev': 1300},
    ]
    
    for i in range(100):  # 100 time points
        timestamp = base_time + timedelta(seconds=i * 30)  # 30-second intervals
        
        for j, station in enumerate(stations):
            # Simulate seismic activity with some noise
            base_amplitude = 0.001 + 0.0005 * np.sin(i * 0.1)
            noise = np.random.normal(0, 0.0001)
            amplitude = base_amplitude + noise
            
            frequency = 2.0 + 0.5 * np.sin(i * 0.05) + np.random.normal(0, 0.1)
            
            record = {
                'id': f"iris_{station['code']}_{i:04d}",
                'network': 'UW',
                'station': station['code'],
                'channel': 'BHZ',
                'location': '00',
                'starttime': timestamp.isoformat() + 'Z',
                'latitude': station['lat'],
                'longitude': station['lon'],
                'elevation': station['elev'],
                'sampling_rate': 40.0,
                'amplitude': amplitude,
                'frequency': frequency,
                'npts': 1200,
                'delta': 0.025,
                'calib': 1.0
            }
            data.append(record)
    
    logger.info(f"Created {len(data)} IRIS seismic records")
    return data


def create_sample_usgs_data() -> List[Dict[str, Any]]:
    """Create sample USGS earthquake data"""
    logger.info("Creating sample USGS earthquake data...")
    
    data = []
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    # Simulate earthquake events in the Cascadia region
    for i in range(20):  # 20 earthquake events
        timestamp = base_time + timedelta(hours=i * 2)  # Every 2 hours
        
        # Random location in Cascadia region
        lat = 45.0 + np.random.uniform(-2, 2)
        lon = -122.0 + np.random.uniform(-3, 3)
        depth = np.random.uniform(5, 50)  # 5-50 km depth
        magnitude = 2.0 + np.random.exponential(1.0)  # Exponential magnitude distribution
        
        record = {
            'eventid': f"uw{60000000 + i:08d}",
            'time': timestamp.isoformat() + '.000Z',
            'latitude': lat,
            'longitude': lon,
            'depth': depth,
            'mag': magnitude,
            'magType': 'ml',
            'place': f"Cascadia Region Event {i+1}",
            'type': 'earthquake',
            'status': 'reviewed',
            'updated': (timestamp + timedelta(minutes=30)).isoformat() + '.000Z',
            'gap': np.random.uniform(50, 150),
            'dmin': np.random.uniform(0.1, 2.0),
            'rms': np.random.uniform(0.1, 0.5),
            'net': 'uw',
            'nst': int(np.random.uniform(10, 50))
        }
        data.append(record)
    
    logger.info(f"Created {len(data)} USGS earthquake records")
    return data


def create_sample_mtsar_data() -> List[Dict[str, Any]]:
    """Create sample MT-SAR deformation data"""
    logger.info("Creating sample MT-SAR deformation data...")
    
    data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # Simulate MT-SAR observations around Mount Rainier
    center_lat, center_lon = 46.8523, -121.7603
    
    for i in range(50):  # 50 SAR acquisitions over time
        timestamp = base_time + timedelta(days=i * 12)  # Every 12 days (Sentinel-1 cycle)
        
        # Create grid of pixels around the volcano
        for x_offset in np.linspace(-0.1, 0.1, 10):  # 10x10 pixel grid
            for y_offset in np.linspace(-0.1, 0.1, 10):
                lat = center_lat + y_offset
                lon = center_lon + x_offset
                
                # Simulate deformation pattern (inflation near center)
                distance_from_center = np.sqrt(x_offset**2 + y_offset**2)
                
                # Seasonal deformation + volcanic inflation
                seasonal = 0.005 * np.sin(2 * np.pi * i / 30.4)  # Annual cycle
                volcanic = 0.02 * np.exp(-distance_from_center * 50) * (i / 50)  # Gradual inflation
                noise = np.random.normal(0, 0.002)
                
                displacement = seasonal + volcanic + noise
                coherence = 0.8 - distance_from_center * 2 + np.random.normal(0, 0.1)
                coherence = max(0.1, min(1.0, coherence))
                
                record = {
                    'id': f"mtsar_rainier_{i:03d}_{int(lat*10000):06d}_{int(abs(lon)*10000):06d}",
                    'acquisition_date': timestamp.isoformat() + 'Z',
                    'satellite_platform': 'Sentinel-1A',
                    'orbit_direction': 'ascending' if i % 2 == 0 else 'descending',
                    'latitude': lat,
                    'longitude': lon,
                    'displacement': displacement,  # meters, line-of-sight
                    'coherence': coherence,
                    'phase': np.random.uniform(-np.pi, np.pi),
                    'temporal_baseline': 12,  # days
                    'perpendicular_baseline': np.random.uniform(50, 200),  # meters
                    'looks': '5x1',
                    'pixel_spacing': 20.0,  # meters
                    'processing_level': 'L2',
                    'dem_source': 'SRTM'
                }
                data.append(record)
    
    logger.info(f"Created {len(data)} MT-SAR deformation records")
    return data


def create_sample_geochemical_data() -> List[Dict[str, Any]]:
    """Create sample geochemical data"""
    logger.info("Creating sample geochemical data...")
    
    data = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # Simulate rock samples from Cascade volcanoes
    volcanoes = [
        {'name': 'Mount St. Helens', 'lat': 46.1912, 'lon': -122.1944},
        {'name': 'Mount Rainier', 'lat': 46.8523, 'lon': -121.7603},
        {'name': 'Mount Baker', 'lat': 48.7767, 'lon': -121.8144},
    ]
    
    rock_types = ['basalt', 'andesite', 'dacite']
    
    for i in range(60):  # 60 samples
        volcano = volcanoes[i % len(volcanoes)]
        rock_type = rock_types[i % len(rock_types)]
        
        # Sample collection date
        timestamp = base_time + timedelta(days=i * 5)
        
        # Simulate typical volcanic rock compositions
        if rock_type == 'basalt':
            sio2 = np.random.normal(50, 2)
            mgo = np.random.normal(8, 1)
            cao = np.random.normal(10, 1)
        elif rock_type == 'andesite':
            sio2 = np.random.normal(58, 3)
            mgo = np.random.normal(4, 1)
            cao = np.random.normal(7, 1)
        else:  # dacite
            sio2 = np.random.normal(65, 2)
            mgo = np.random.normal(2, 0.5)
            cao = np.random.normal(4, 0.5)
        
        record = {
            'sample_id': f"casc_{volcano['name'].replace(' ', '').lower()}_{i:03d}",
            'material': 'volcanic rock',
            'rock_class': rock_type,
            'latitude': volcano['lat'] + np.random.uniform(-0.01, 0.01),
            'longitude': volcano['lon'] + np.random.uniform(-0.01, 0.01),
            'elevation': np.random.uniform(1000, 3000),
            'collection_date': timestamp.isoformat() + 'Z',
            'age': np.random.uniform(0.001, 2.0),  # Ma
            'SiO2': sio2,  # wt%
            'MgO': mgo,    # wt%
            'CaO': cao,    # wt%
            'Al2O3': np.random.normal(16, 2),  # wt%
            'FeO': np.random.normal(8, 2),     # wt%
            'Na2O': np.random.normal(3, 0.5),  # wt%
            'K2O': np.random.normal(2, 0.5),   # wt%
            'method': 'XRF',
            'laboratory': 'University of Washington',
            'tectonic_setting': 'subduction zone'
        }
        data.append(record)
    
    logger.info(f"Created {len(data)} geochemical records")
    return data


async def demonstrate_pipeline_processing():
    """Demonstrate complete pipeline processing with multiple data sources"""
    logger.info("Starting comprehensive data transformation pipeline demonstration")
    
    # Create pipeline with custom configuration
    config = TransformationConfig(
        target_coordinate_system=CoordinateSystem.WGS84,
        target_temporal_resolution=TemporalResolution.HOUR,
        target_timezone="UTC",
        interpolation_method="linear",
        aggregation_method="mean",
        fill_gaps=True,
        max_gap_duration=7200.0,  # 2 hours
        export_format="netcdf",
        include_uncertainty=True,
        preserve_raw_data=True
    )
    
    pipeline = DataTransformationPipeline(config)
    
    # Process different data types
    data_sources = [
        (create_sample_iris_data(), "IRIS", DataType.SEISMIC),
        (create_sample_usgs_data(), "USGS", DataType.SEISMIC),
        (create_sample_mtsar_data(), "ASF_Hyp3", DataType.MTSAR),
        (create_sample_geochemical_data(), "EarthChem", DataType.GEOCHEMICAL)
    ]
    
    all_results = {}
    
    for raw_data, source_name, data_type in data_sources:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {source_name} {data_type.value} data")
        logger.info(f"{'='*60}")
        
        # Process data through pipeline
        result = await pipeline.process_data(
            raw_data, source_name, data_type, config
        )
        
        # Log results
        logger.info(f"Processing Results for {source_name}:")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Original records: {result.original_count}")
        logger.info(f"  Processed records: {result.processed_count}")
        logger.info(f"  Processing time: {result.processing_time_seconds:.2f} seconds")
        logger.info(f"  Transformations applied: {', '.join(result.transformations_applied)}")
        logger.info(f"  Quality improvement: {result.quality_improvement:.2%}")
        
        if result.warnings:
            logger.warning(f"  Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:  # Show first 3 warnings
                logger.warning(f"    - {warning}")
        
        if result.errors:
            logger.error(f"  Errors: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                logger.error(f"    - {error}")
        
        all_results[f"{source_name}_{data_type.value}"] = result
    
    return all_results


async def demonstrate_export_capabilities(processed_results):
    """Demonstrate various export formats"""
    logger.info(f"\n{'='*60}")
    logger.info("Demonstrating Export Capabilities")
    logger.info(f"{'='*60}")
    
    from backend.core.data_transformation import ExportManager
    
    export_manager = ExportManager()
    
    # Export formats to demonstrate
    export_formats = ['json', 'csv', 'geojson', 'netcdf', 'hdf5']
    
    for source_data_type, result in processed_results.items():
        if not result.success or not result.processed_data:
            continue
        
        logger.info(f"\nExporting {source_data_type} data:")
        
        for format_name in export_formats:
            try:
                export_config = ExportConfig(
                    format=format_name,
                    output_path=f"output/{source_data_type}_export",
                    include_metadata=True,
                    include_processing_history=True,
                    include_quality_metrics=True,
                    include_uncertainty=True,
                    compression='gzip' if format_name in ['netcdf', 'hdf5'] else None,
                    precision=6
                )
                
                export_result = export_manager.export_data(
                    result.processed_data[:10],  # Export first 10 records for demo
                    export_config
                )
                
                if export_result.success:
                    logger.info(f"  ✓ {format_name.upper()}: {export_result.file_size_bytes} bytes")
                else:
                    logger.error(f"  ✗ {format_name.upper()}: {export_result.errors}")
                    
            except Exception as e:
                logger.error(f"  ✗ {format_name.upper()}: Export failed - {e}")


async def demonstrate_aggregation_capabilities(processed_results):
    """Demonstrate multi-resolution aggregation"""
    logger.info(f"\n{'='*60}")
    logger.info("Demonstrating Multi-Resolution Aggregation")
    logger.info(f"{'='*60}")
    
    from backend.core.data_transformation import AggregationEngine, AggregationConfig
    
    aggregation_engine = AggregationEngine()
    
    # Test with seismic data (highest temporal resolution)
    seismic_results = [r for k, r in processed_results.items() 
                      if 'SEISMIC' in k and r.success and r.processed_data]
    
    if seismic_results:
        seismic_data = seismic_results[0].processed_data
        logger.info(f"Aggregating {len(seismic_data)} seismic records")
        
        # Create multi-resolution dataset
        resolutions = [
            TemporalResolution.MINUTE,
            TemporalResolution.HOUR,
            TemporalResolution.DAY
        ]
        
        multi_res_data = aggregation_engine.create_multi_resolution_dataset(
            seismic_data, resolutions
        )
        
        logger.info("Multi-resolution aggregation results:")
        for resolution, data in multi_res_data.items():
            logger.info(f"  {resolution.value}: {len(data)} records")
            
            # Show sample aggregated values
            if data:
                sample = data[0]
                logger.info(f"    Sample values: {list(sample.values.keys())}")
                logger.info(f"    Quality flags: {sample.quality_flags}")


async def demonstrate_coordinate_transformations():
    """Demonstrate coordinate system transformations"""
    logger.info(f"\n{'='*60}")
    logger.info("Demonstrating Coordinate Transformations")
    logger.info(f"{'='*60}")
    
    from backend.core.data_transformation import CoordinateTransformer
    from backend.core.data_transformation.schemas import StandardizedDataFormat
    
    transformer = CoordinateTransformer()
    
    # Test locations around the world
    test_locations = [
        {"name": "Mount St. Helens", "lat": 46.1912, "lon": -122.1944},
        {"name": "Mount Fuji", "lat": 35.3606, "lon": 138.7274},
        {"name": "Mount Vesuvius", "lat": 40.8214, "lon": 14.4289},
        {"name": "Kilauea", "lat": 19.4069, "lon": -155.2834},
    ]
    
    for location in test_locations:
        logger.info(f"\nTransforming coordinates for {location['name']}:")
        
        # Create sample data
        data = StandardizedDataFormat(
            data_id=f"coord_test_{location['name'].replace(' ', '_').lower()}",
            data_type=DataType.VOLCANIC,
            location={'x': location['lon'], 'y': location['lat'], 'z': 0.0},
            coordinate_system=CoordinateSystem.WGS84,
            timestamp=datetime.now(),
            values={'temperature': 25.0},
            units={'temperature': 'C'},
            source='TEST'
        )
        
        # Transform to UTM
        transformed = transformer.transform_data_format(data, CoordinateSystem.UTM_NORTH)
        
        logger.info(f"  Original (WGS84): {location['lat']:.4f}, {location['lon']:.4f}")
        logger.info(f"  UTM: {transformed.location['x']:.2f}, {transformed.location['y']:.2f}")
        logger.info(f"  Coordinate system: {transformed.coordinate_system.value}")


async def main():
    """Main demonstration function"""
    logger.info("Geological Data Transformation Pipeline Demonstration")
    logger.info("=" * 80)
    
    try:
        # Create output directory
        import os
        os.makedirs("output", exist_ok=True)
        
        # 1. Demonstrate complete pipeline processing
        processed_results = await demonstrate_pipeline_processing()
        
        # 2. Demonstrate coordinate transformations
        await demonstrate_coordinate_transformations()
        
        # 3. Demonstrate aggregation capabilities
        await demonstrate_aggregation_capabilities(processed_results)
        
        # 4. Demonstrate export capabilities
        await demonstrate_export_capabilities(processed_results)
        
        logger.info(f"\n{'='*80}")
        logger.info("Demonstration completed successfully!")
        logger.info("Check the 'output' directory for exported files.")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())