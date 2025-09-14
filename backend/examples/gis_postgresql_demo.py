"""
GIS PostgreSQL Operations Demo
Comprehensive demonstration of spatial indexing, queries, transformations, and analysis
for geological datasets using PostGIS and the Prometheus AI Platform
"""

import asyncio
import logging
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Import GIS operations modules
from backend.core.gis_database_integration import (
    GISDatabaseIntegrator,
    GISQueryRequest,
    GISQueryResponse
)
from backend.core.gis_postgresql_operations import (
    GISPostgreSQLOperations,
    CoordinateSystem,
    SpatialIndexType,
    ExportFormat,
    SpatialUtils
)
from backend.core.database_integration import EnhancedDatabaseIntegrator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GISPostgreSQLDemo:
    """Comprehensive demo of GIS PostgreSQL operations"""
    
    def __init__(self):
        self.gis_integrator = None
        self.demo_data = {}
        
    async def initialize(self):
        """Initialize the GIS database integrator"""
        logger.info("Initializing GIS PostgreSQL Demo")
        
        # Initialize GIS database integrator
        self.gis_integrator = GISDatabaseIntegrator()
        
        if not await self.gis_integrator.initialize():
            raise Exception("Failed to initialize GIS database integrator")
        
        logger.info("GIS PostgreSQL Demo initialized successfully")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all GIS capabilities"""
        logger.info("Starting comprehensive GIS PostgreSQL demonstration")
        
        try:
            # 1. Spatial Indexing Demo
            await self.demo_spatial_indexing()
            
            # 2. Volcanic Systems Spatial Queries Demo
            await self.demo_volcanic_systems_queries()
            
            # 3. Seismic Events Spatial-Temporal Queries Demo
            await self.demo_seismic_events_queries()
            
            # 4. Deformation Data Analysis Demo
            await self.demo_deformation_analysis()
            
            # 5. Monitoring Network Coverage Demo
            await self.demo_monitoring_network_analysis()
            
            # 6. Coordinate System Transformations Demo
            await self.demo_coordinate_transformations()
            
            # 7. Spatial Clustering Analysis Demo
            await self.demo_spatial_clustering()
            
            # 8. Geological Context Queries Demo
            await self.demo_geological_context_queries()
            
            # 9. Data Quality Validation Demo
            await self.demo_data_quality_validation()
            
            # 10. Geospatial Data Export Demo
            await self.demo_geospatial_export()
            
            # 11. Performance Analysis Demo
            await self.demo_performance_analysis()
            
            logger.info("Comprehensive GIS PostgreSQL demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def demo_spatial_indexing(self):
        """Demonstrate spatial indexing capabilities"""
        logger.info("=== Spatial Indexing Demo ===")
        
        gis_ops = self.gis_integrator.gis_ops
        
        # Create spatial indexes for volcanic systems
        logger.info("Creating GIST spatial indexes for volcanic systems...")
        result = await gis_ops.create_spatial_indexes(
            table_name="volcanic_systems",
            geometry_columns=["location", "hazard_zone"],
            index_type=SpatialIndexType.GIST
        )
        logger.info(f"Spatial indexes created: {result}")
        
        # Optimize spatial queries
        logger.info("Optimizing spatial queries...")
        optimization_results = await gis_ops.optimize_spatial_queries("volcanic_systems")
        logger.info(f"Optimization results: {json.dumps(optimization_results, indent=2, default=str)}")
        
        # Get spatial metrics
        spatial_metrics = gis_ops.get_spatial_metrics()
        logger.info(f"Spatial metrics: {spatial_metrics}")
    
    async def demo_volcanic_systems_queries(self):
        """Demonstrate volcanic systems spatial queries"""
        logger.info("=== Volcanic Systems Spatial Queries Demo ===")
        
        # Query volcanic systems in the Pacific Ring of Fire
        logger.info("Querying volcanic systems in the Pacific Ring of Fire...")
        
        # Define Pacific Ring of Fire region (simplified polygon)
        ring_of_fire_region = {
            "type": "Polygon",
            "coordinates": [[
                [-180, -60], [180, -60], [180, 70], [-180, 70], [-180, -60]
            ]]
        }
        
        request = GISQueryRequest(
            query_type="volcanic_systems",
            spatial_filters={
                "region": ring_of_fire_region,
                "buffer_distance": 50000  # 50km buffer
            },
            attribute_filters={
                "tectonic_settings": ["Subduction zone", "Mid-ocean ridge"],
                "hazard_levels": [3, 4, 5]
            },
            coordinate_system=4326,
            output_format="geojson"
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        logger.info(f"Found {response.feature_count} volcanic systems")
        logger.info(f"Query processing time: {response.processing_time:.3f} seconds")
        logger.info(f"Spatial extent: {response.spatial_extent}")
        
        # Store results for later use
        self.demo_data['volcanic_systems'] = response.data
        
        # Query volcanic systems by specific coordinates
        logger.info("Querying volcanic systems near Mount St. Helens...")
        
        # Create point buffer around Mount St. Helens
        mt_st_helens_point = SpatialUtils.create_point_from_coordinates(-122.1944, 46.1912)
        
        request = GISQueryRequest(
            query_type="volcanic_systems",
            spatial_filters={
                "region": mt_st_helens_point,
                "buffer_distance": 100000  # 100km buffer
            },
            attribute_filters={
                "volcano_types": ["Stratovolcano", "Shield"]
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        logger.info(f"Found {response.feature_count} volcanic systems near Mount St. Helens")
    
    async def demo_seismic_events_queries(self):
        """Demonstrate seismic events spatial-temporal queries"""
        logger.info("=== Seismic Events Spatial-Temporal Queries Demo ===")
        
        # Query recent seismic events in Cascadia region
        logger.info("Querying recent seismic events in Cascadia region...")
        
        cascadia_bounds = {
            'min_lon': -130.0,
            'max_lon': -115.0,
            'min_lat': 40.0,
            'max_lat': 55.0
        }
        
        # Last 30 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        
        request = GISQueryRequest(
            query_type="seismic_events",
            spatial_filters={
                "bounds": cascadia_bounds
            },
            temporal_filters={
                "time_range": (start_time, end_time)
            },
            attribute_filters={
                "magnitude_range": (2.0, 8.0),
                "depth_range": (0.0, 50.0),
                "event_types": ["earthquake", "volcanic"]
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        logger.info(f"Found {response.feature_count} seismic events in the last 30 days")
        logger.info(f"Query processing time: {response.processing_time:.3f} seconds")
        
        # Store results for clustering analysis
        self.demo_data['seismic_events'] = response.data
        
        # Query high-magnitude events globally
        logger.info("Querying high-magnitude seismic events globally...")
        
        request = GISQueryRequest(
            query_type="seismic_events",
            temporal_filters={
                "time_range": (datetime(2020, 1, 1), datetime.now())
            },
            attribute_filters={
                "magnitude_range": (6.0, 9.0),
                "event_types": ["earthquake"]
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        logger.info(f"Found {response.feature_count} high-magnitude events since 2020")
    
    async def demo_deformation_analysis(self):
        """Demonstrate deformation data analysis with MT-SAR integration"""
        logger.info("=== Deformation Data Analysis Demo ===")
        
        # Query deformation data for Yellowstone
        logger.info("Querying deformation data for Yellowstone caldera...")
        
        # Yellowstone approximate location
        yellowstone_id = "yellowstone-caldera-id"  # Would be actual UUID in real system
        
        request = GISQueryRequest(
            query_type="deformation_data",
            spatial_filters={
                "volcanic_system_id": yellowstone_id
            },
            temporal_filters={
                "time_range": (datetime(2023, 1, 1), datetime.now())
            },
            attribute_filters={
                "measurement_types": ["InSAR", "GPS"]
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        logger.info(f"Found {response.feature_count} deformation measurements")
        
        # Analyze deformation patterns
        if response.data:
            logger.info("Analyzing deformation patterns...")
            
            # Calculate statistics for different measurement types
            insar_data = [d for d in response.data if d.get('measurement_type') == 'InSAR']
            gps_data = [d for d in response.data if d.get('measurement_type') == 'GPS']
            
            logger.info(f"InSAR measurements: {len(insar_data)}")
            logger.info(f"GPS measurements: {len(gps_data)}")
            
            # Calculate average displacements
            if insar_data:
                avg_los_displacement = sum(
                    d.get('displacement', {}).get('line_of_sight', 0) 
                    for d in insar_data
                ) / len(insar_data)
                logger.info(f"Average line-of-sight displacement (InSAR): {avg_los_displacement:.3f} mm")
            
            if gps_data:
                avg_vertical_displacement = sum(
                    d.get('displacement', {}).get('up', 0) 
                    for d in gps_data
                ) / len(gps_data)
                logger.info(f"Average vertical displacement (GPS): {avg_vertical_displacement:.3f} mm")
    
    async def demo_monitoring_network_analysis(self):
        """Demonstrate monitoring network coverage analysis"""
        logger.info("=== Monitoring Network Coverage Demo ===")
        
        # Analyze monitoring network for Mount Rainier
        logger.info("Analyzing monitoring network for Mount Rainier...")
        
        mt_rainier_id = "mount-rainier-id"  # Would be actual UUID in real system
        
        request = GISQueryRequest(
            query_type="monitoring_network",
            spatial_filters={
                "volcanic_system_id": mt_rainier_id
            },
            attribute_filters={
                "station_types": ["seismic", "GPS", "tilt", "gas"],
                "operational_only": True
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        
        if isinstance(response.data, dict):
            network_data = response.data
            logger.info(f"Monitoring network analysis for {network_data.get('volcano_name', 'Unknown')}:")
            logger.info(f"  Total stations: {network_data.get('station_count', 0)}")
            logger.info(f"  Station types: {network_data.get('station_types', [])}")
            
            # Analyze station distribution
            stations = network_data.get('stations', [])
            if stations:
                station_type_counts = {}
                real_time_count = 0
                
                for station in stations:
                    station_type = station.get('type', 'unknown')
                    station_type_counts[station_type] = station_type_counts.get(station_type, 0) + 1
                    
                    if station.get('real_time_availability'):
                        real_time_count += 1
                
                logger.info(f"  Station type distribution: {station_type_counts}")
                logger.info(f"  Real-time stations: {real_time_count}/{len(stations)}")
                
                # Coverage area analysis
                coverage_area = network_data.get('coverage_area')
                if coverage_area:
                    logger.info("  Network coverage area calculated successfully")
    
    async def demo_coordinate_transformations(self):
        """Demonstrate coordinate system transformations"""
        logger.info("=== Coordinate System Transformations Demo ===")
        
        gis_ops = self.gis_integrator.gis_ops
        
        # Test coordinate transformations for various locations
        test_locations = [
            ("Mount St. Helens", -122.1944, 46.1912),
            ("Kilauea", -155.2920, 19.4069),
            ("Mount Fuji", 138.7274, 35.3606),
            ("Etna", 14.9955, 37.7510)
        ]
        
        for name, lon, lat in test_locations:
            logger.info(f"Transforming coordinates for {name}...")
            
            # Create WGS84 point
            wgs84_point = SpatialUtils.create_point_from_coordinates(lon, lat)
            logger.info(f"  WGS84: {wgs84_point}")
            
            # Get optimal UTM zone
            utm_zone = await gis_ops.get_optimal_utm_zone(lon, lat)
            logger.info(f"  Optimal UTM zone: {utm_zone}")
            
            # Transform to UTM
            utm_points = await gis_ops.transform_coordinates(
                geometries=[wgs84_point],
                source_crs=CoordinateSystem.WGS84.value,
                target_crs=utm_zone
            )
            
            if utm_points:
                logger.info(f"  UTM coordinates: {utm_points[0]}")
            
            # Transform to Web Mercator
            web_mercator_points = await gis_ops.transform_coordinates(
                geometries=[wgs84_point],
                source_crs=CoordinateSystem.WGS84.value,
                target_crs=CoordinateSystem.WEB_MERCATOR.value
            )
            
            if web_mercator_points:
                logger.info(f"  Web Mercator: {web_mercator_points[0]}")
        
        # Validate coordinate systems
        logger.info("Validating coordinate systems...")
        
        test_srids = [4326, 3857, 32610, 32633]
        for srid in test_srids:
            validation_result = await gis_ops.validate_coordinate_system(srid)
            logger.info(f"  SRID {srid}: {'Valid' if validation_result['valid'] else 'Invalid'}")
    
    async def demo_spatial_clustering(self):
        """Demonstrate spatial clustering analysis"""
        logger.info("=== Spatial Clustering Analysis Demo ===")
        
        gis_ops = self.gis_integrator.gis_ops
        
        # Perform clustering analysis on seismic events
        if 'seismic_events' in self.demo_data and self.demo_data['seismic_events']:
            logger.info("Performing spatial clustering on seismic events...")
            
            clustering_result = await gis_ops.perform_spatial_clustering(
                table_name="seismic_events",
                geometry_column="epicenter",
                cluster_distance=10000.0,  # 10km
                min_points=5
            )
            
            if 'error' not in clustering_result:
                logger.info(f"Clustering analysis results:")
                logger.info(f"  Number of clusters: {clustering_result['cluster_count']}")
                logger.info(f"  Clustering parameters: {clustering_result['clustering_parameters']}")
                
                if 'statistics' in clustering_result:
                    stats = clustering_result['statistics']
                    logger.info(f"  Mean cluster size: {stats['mean_cluster_size']:.1f}")
                    logger.info(f"  Total clustered points: {stats['total_clustered_points']}")
                
                # Analyze largest clusters
                clusters = clustering_result.get('clusters', [])
                if clusters:
                    largest_clusters = sorted(clusters, key=lambda x: x['point_count'], reverse=True)[:3]
                    logger.info("  Largest clusters:")
                    for i, cluster in enumerate(largest_clusters, 1):
                        logger.info(f"    {i}. Cluster {cluster['cluster_id']}: {cluster['point_count']} events")
            else:
                logger.warning(f"Clustering analysis failed: {clustering_result['error']}")
        else:
            logger.info("No seismic events data available for clustering analysis")
    
    async def demo_geological_context_queries(self):
        """Demonstrate geological context queries"""
        logger.info("=== Geological Context Queries Demo ===")
        
        # Query geological context for Yellowstone region
        logger.info("Querying geological context for Yellowstone region...")
        
        yellowstone_region = {
            "type": "Polygon",
            "coordinates": [[
                [-111.5, 44.0], [-110.0, 44.0], [-110.0, 45.0], [-111.5, 45.0], [-111.5, 44.0]
            ]]
        }
        
        request = GISQueryRequest(
            query_type="geological_context",
            spatial_filters={
                "region": yellowstone_region
            },
            attribute_filters={
                "age_range": (0.0, 2.1),  # Quaternary period
                "lithologies": ["volcanic", "sedimentary", "igneous"]
            }
        )
        
        response = await self.gis_integrator.execute_gis_query(request)
        
        if response.data and len(response.data) > 0:
            context_data = response.data[0]  # First (and likely only) result
            
            geological_units = context_data.get('geological_units', [])
            tectonic_features = context_data.get('tectonic_features', [])
            
            logger.info(f"Geological context analysis:")
            logger.info(f"  Geological units: {len(geological_units)}")
            logger.info(f"  Tectonic features: {len(tectonic_features)}")
            
            # Analyze geological units
            if geological_units:
                lithology_counts = {}
                age_ranges = []
                
                for unit in geological_units:
                    lithology = unit.get('primary_lithology', 'unknown')
                    lithology_counts[lithology] = lithology_counts.get(lithology, 0) + 1
                    
                    age_range = unit.get('age_range', [])
                    if len(age_range) == 2:
                        age_ranges.append(age_range)
                
                logger.info(f"  Lithology distribution: {lithology_counts}")
                
                if age_ranges:
                    min_age = min(age[0] for age in age_ranges)
                    max_age = max(age[1] for age in age_ranges)
                    logger.info(f"  Age range: {min_age:.1f} - {max_age:.1f} Ma")
            
            # Analyze tectonic features
            if tectonic_features:
                feature_types = {}
                active_features = 0
                
                for feature in tectonic_features:
                    feature_type = feature.get('feature_type', 'unknown')
                    feature_types[feature_type] = feature_types.get(feature_type, 0) + 1
                    
                    if feature.get('activity_status') == 'active':
                        active_features += 1
                
                logger.info(f"  Tectonic feature types: {feature_types}")
                logger.info(f"  Active features: {active_features}/{len(tectonic_features)}")
    
    async def demo_data_quality_validation(self):
        """Demonstrate data quality validation"""
        logger.info("=== Data Quality Validation Demo ===")
        
        gis_ops = self.gis_integrator.gis_ops
        
        # Validate geometry quality for different tables
        tables_to_validate = [
            ("volcanic_systems", "location"),
            ("seismic_events", "epicenter"),
            ("geological_units", "geometry")
        ]
        
        for table_name, geometry_column in tables_to_validate:
            logger.info(f"Validating geometry quality for {table_name}...")
            
            validation_result = await gis_ops.validate_geometry_quality(
                table_name=table_name,
                geometry_column=geometry_column
            )
            
            if 'error' not in validation_result:
                logger.info(f"  Quality validation results:")
                logger.info(f"    Total records: {validation_result.get('total_records', 0)}")
                logger.info(f"    Invalid geometries: {validation_result.get('invalid_geometries', 0)}")
                logger.info(f"    Empty geometries: {validation_result.get('empty_geometries', 0)}")
                logger.info(f"    Null geometries: {validation_result.get('null_geometries', 0)}")
                logger.info(f"    Quality score: {validation_result.get('quality_score', 0):.1f}%")
                
                # Show SRID distribution
                srid_dist = validation_result.get('srid_distribution', [])
                if srid_dist:
                    logger.info(f"    SRID distribution: {srid_dist}")
                
                # Show geometry types
                geom_types = validation_result.get('geometry_types', [])
                if geom_types:
                    logger.info(f"    Geometry types: {geom_types}")
                
                # Repair invalid geometries if any
                invalid_count = validation_result.get('invalid_geometries', 0)
                if invalid_count > 0:
                    logger.info(f"    Attempting to repair {invalid_count} invalid geometries...")
                    
                    repair_result = await gis_ops.repair_invalid_geometries(
                        table_name=table_name,
                        geometry_column=geometry_column
                    )
                    
                    if 'error' not in repair_result:
                        logger.info(f"    Repair results:")
                        logger.info(f"      Repaired: {repair_result.get('repaired_count', 0)}")
                        logger.info(f"      Still invalid: {repair_result.get('still_invalid_count', 0)}")
            else:
                logger.warning(f"  Validation failed: {validation_result['error']}")
    
    async def demo_geospatial_export(self):
        """Demonstrate geospatial data export capabilities"""
        logger.info("=== Geospatial Data Export Demo ===")
        
        # Create temporary directory for exports
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Export volcanic systems as GeoJSON
            logger.info("Exporting volcanic systems as GeoJSON...")
            
            request = GISQueryRequest(
                query_type="volcanic_systems",
                attribute_filters={
                    "hazard_levels": [3, 4, 5]
                },
                coordinate_system=4326
            )
            
            geojson_path = temp_path / "volcanic_systems.geojson"
            export_result = await self.gis_integrator.export_spatial_data(
                query_request=request,
                export_format=ExportFormat.GEOJSON,
                output_path=str(geojson_path)
            )
            
            if 'error' not in export_result:
                logger.info(f"  GeoJSON export successful:")
                logger.info(f"    File: {geojson_path}")
                logger.info(f"    Records: {export_result.get('record_count', 0)}")
                logger.info(f"    Format: {export_result.get('format', 'unknown')}")
                
                # Verify file exists and has content
                if geojson_path.exists():
                    file_size = geojson_path.stat().st_size
                    logger.info(f"    File size: {file_size} bytes")
            else:
                logger.warning(f"  GeoJSON export failed: {export_result['error']}")
            
            # Export seismic events as KML
            logger.info("Exporting seismic events as KML...")
            
            request = GISQueryRequest(
                query_type="seismic_events",
                temporal_filters={
                    "time_range": (datetime.now() - timedelta(days=7), datetime.now())
                },
                attribute_filters={
                    "magnitude_range": (3.0, 8.0)
                }
            )
            
            kml_path = temp_path / "seismic_events.kml"
            export_result = await self.gis_integrator.export_spatial_data(
                query_request=request,
                export_format=ExportFormat.KML,
                output_path=str(kml_path)
            )
            
            if 'error' not in export_result:
                logger.info(f"  KML export successful:")
                logger.info(f"    File: {kml_path}")
                logger.info(f"    Records: {export_result.get('record_count', 0)}")
                
                if kml_path.exists():
                    file_size = kml_path.stat().st_size
                    logger.info(f"    File size: {file_size} bytes")
            else:
                logger.warning(f"  KML export failed: {export_result['error']}")
    
    async def demo_performance_analysis(self):
        """Demonstrate performance analysis and metrics"""
        logger.info("=== Performance Analysis Demo ===")
        
        # Get comprehensive performance metrics
        performance_metrics = self.gis_integrator.get_performance_metrics()
        
        logger.info("Performance metrics summary:")
        
        # Integration metrics
        integration_metrics = performance_metrics.get('integration_metrics', {})
        logger.info(f"  Integration operations:")
        logger.info(f"    Spatial queries: {integration_metrics.get('spatial_queries', 0)}")
        logger.info(f"    Temporal queries: {integration_metrics.get('temporal_queries', 0)}")
        logger.info(f"    Hybrid queries: {integration_metrics.get('hybrid_queries', 0)}")
        logger.info(f"    Data exports: {integration_metrics.get('data_exports', 0)}")
        logger.info(f"    Coordinate transformations: {integration_metrics.get('coordinate_transformations', 0)}")
        
        # Cache statistics
        cache_stats = performance_metrics.get('cache_statistics', {})
        logger.info(f"  Cache performance:")
        logger.info(f"    Cache size: {cache_stats.get('cache_size', 0)} entries")
        logger.info(f"    Cache hit rate: {cache_stats.get('cache_hit_rate', 0):.1f}%")
        
        # Database metrics
        db_metrics = performance_metrics.get('database_metrics', {})
        logger.info(f"  Database performance:")
        logger.info(f"    Query count: {db_metrics.get('query_count', 0)}")
        logger.info(f"    Error count: {db_metrics.get('error_count', 0)}")
        logger.info(f"    Average query time: {db_metrics.get('average_query_time', 0):.3f}s")
        
        # Spatial metrics
        spatial_metrics = performance_metrics.get('spatial_metrics', {})
        logger.info(f"  Spatial operations:")
        logger.info(f"    Queries executed: {spatial_metrics.get('queries_executed', 0)}")
        logger.info(f"    Transformations performed: {spatial_metrics.get('transformations_performed', 0)}")
        logger.info(f"    Indexes created: {spatial_metrics.get('indexes_created', 0)}")
        logger.info(f"    Exports completed: {spatial_metrics.get('exports_completed', 0)}")
        
        # Analyze query performance
        logger.info("Analyzing spatial query performance...")
        
        gis_ops = self.gis_integrator.gis_ops
        
        sample_queries = [
            "SELECT COUNT(*) FROM volcanic_systems WHERE ST_DWithin(location, ST_MakePoint(-122, 46), 100000)",
            "SELECT * FROM seismic_events WHERE magnitude > 4.0 AND ST_Intersects(epicenter, ST_MakeEnvelope(-125, 45, -120, 50, 4326))",
            "SELECT ST_Area(ST_Union(geometry)) FROM geological_units WHERE primary_lithology = 'volcanic'"
        ]
        
        performance_analysis = await gis_ops.analyze_query_performance(
            table_name="volcanic_systems",
            sample_queries=sample_queries
        )
        
        if 'error' not in performance_analysis:
            logger.info("Query performance analysis:")
            for query_key, analysis in performance_analysis.items():
                if isinstance(analysis, dict) and 'execution_time_seconds' in analysis:
                    logger.info(f"  {query_key}: {analysis['execution_time_seconds']:.3f}s")
    
    async def cleanup(self):
        """Clean up demo resources"""
        logger.info("Cleaning up demo resources...")
        
        if self.gis_integrator:
            await self.gis_integrator.cleanup()
        
        logger.info("Demo cleanup completed")


async def main():
    """Main demo function"""
    demo = GISPostgreSQLDemo()
    
    try:
        # Initialize demo
        await demo.initialize()
        
        # Run comprehensive demonstration
        await demo.run_comprehensive_demo()
        
        logger.info("GIS PostgreSQL demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    
    finally:
        # Clean up
        await demo.cleanup()


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())