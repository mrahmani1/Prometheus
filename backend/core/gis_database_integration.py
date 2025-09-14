"""
GIS Database Integration Module
Integrates GIS-enabled PostgreSQL operations with the existing database infrastructure
Provides unified interface for spatial and non-spatial geological data operations
"""

import asyncio
import logging
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Import existing database components
from backend.core.database_integration import EnhancedDatabaseIntegrator, DatabaseMetrics, QueryResult
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

# Import advanced database integration
from backend.modules.advanced_database_integration import AdvancedDatabaseIntegrator, GeologicalDataPoint


@dataclass
class GISQueryRequest:
    """Request structure for GIS-enabled queries"""
    query_type: str
    spatial_filters: Optional[Dict[str, Any]] = None
    temporal_filters: Optional[Dict[str, Any]] = None
    attribute_filters: Optional[Dict[str, Any]] = None
    coordinate_system: int = 4326
    output_format: str = 'geojson'
    include_metadata: bool = True


@dataclass
class GISQueryResponse:
    """Response structure for GIS-enabled queries"""
    data: Any
    spatial_extent: Optional[Dict[str, float]]
    feature_count: int
    coordinate_system: int
    processing_time: float
    metadata: Dict[str, Any]
    cache_info: Optional[Dict[str, Any]] = None


class GISDatabaseIntegrator:
    """
    Unified GIS-enabled database integrator
    Combines spatial and non-spatial operations for comprehensive geological data management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".GISDatabaseIntegrator")
        
        # Initialize database components
        self.enhanced_db = EnhancedDatabaseIntegrator(config)
        self.gis_ops = GISPostgreSQLOperations(self.enhanced_db)
        
        # Advanced database integrator for compatibility
        self.advanced_db = AdvancedDatabaseIntegrator(config)
        
        # Performance tracking
        self.performance_metrics = {
            'spatial_queries': 0,
            'temporal_queries': 0,
            'hybrid_queries': 0,
            'data_exports': 0,
            'coordinate_transformations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Query cache for expensive spatial operations
        self.spatial_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        self.logger.info("GIS Database Integrator initialized")
    
    async def initialize(self) -> bool:
        """Initialize all database components"""
        try:
            self.logger.info("Initializing GIS database integration")
            
            # Initialize enhanced database integrator
            if not await self.enhanced_db.initialize():
                raise Exception("Failed to initialize enhanced database integrator")
            
            # Initialize spatial extensions
            if not await self.gis_ops.initialize_spatial_extensions():
                raise Exception("Failed to initialize spatial extensions")
            
            # Initialize advanced database integrator
            if hasattr(self.advanced_db, 'initialize'):
                await self.advanced_db.initialize()
            
            # Create spatial indexes for performance
            await self._create_default_spatial_indexes()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.logger.info("GIS database integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize GIS database integration: {e}")
            return False
    
    async def _create_default_spatial_indexes(self):
        """Create default spatial indexes for optimal performance"""
        try:
            # Volcanic systems indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="volcanic_systems",
                geometry_columns=["location", "hazard_zone", "monitoring_network"],
                index_type=SpatialIndexType.GIST
            )
            
            # Seismic events indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="seismic_events",
                geometry_columns=["hypocenter", "epicenter"],
                index_type=SpatialIndexType.GIST
            )
            
            # Deformation measurements indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="deformation_measurements",
                geometry_columns=["measurement_point", "measurement_area"],
                index_type=SpatialIndexType.GIST
            )
            
            # Monitoring stations indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="monitoring_stations",
                geometry_columns=["location"],
                index_type=SpatialIndexType.GIST
            )
            
            # Geological units indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="geological_units",
                geometry_columns=["geometry", "representative_point"],
                index_type=SpatialIndexType.GIST
            )
            
            # Tectonic features indexes
            await self.gis_ops.create_spatial_indexes(
                table_name="tectonic_features",
                geometry_columns=["geometry", "zone_geometry"],
                index_type=SpatialIndexType.GIST
            )
            
            self.logger.info("Default spatial indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create default spatial indexes: {e}")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())
        
        # Performance monitoring task
        asyncio.create_task(self._performance_monitoring_task())
    
    async def _cache_cleanup_task(self):
        """Background task to clean up expired cache entries"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                current_time = datetime.now()
                expired_keys = []
                
                for key, (data, timestamp) in self.spatial_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.spatial_cache[key]
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                self.logger.error(f"Cache cleanup task error: {e}")
    
    async def _performance_monitoring_task(self):
        """Background task to monitor and log performance metrics"""
        while True:
            try:
                await asyncio.sleep(1800)  # Run every 30 minutes
                
                # Log performance metrics
                self.logger.info(f"Performance metrics: {self.performance_metrics}")
                
                # Get database metrics
                db_metrics = self.enhanced_db.get_database_metrics()
                spatial_metrics = self.gis_ops.get_spatial_metrics()
                
                # Log combined metrics
                combined_metrics = {
                    'database': db_metrics,
                    'spatial': spatial_metrics,
                    'integration': self.performance_metrics
                }
                
                self.logger.debug(f"Combined metrics: {combined_metrics}")
                
            except Exception as e:
                self.logger.error(f"Performance monitoring task error: {e}")
    
    # Unified Query Interface
    async def execute_gis_query(self, request: GISQueryRequest) -> GISQueryResponse:
        """Execute unified GIS query with spatial, temporal, and attribute filters"""
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self.performance_metrics['cache_hits'] += 1
                return cached_result
            
            self.performance_metrics['cache_misses'] += 1
            
            # Execute query based on type
            if request.query_type == "volcanic_systems":
                data = await self._query_volcanic_systems(request)
                self.performance_metrics['spatial_queries'] += 1
                
            elif request.query_type == "seismic_events":
                data = await self._query_seismic_events(request)
                self.performance_metrics['hybrid_queries'] += 1
                
            elif request.query_type == "deformation_data":
                data = await self._query_deformation_data(request)
                self.performance_metrics['temporal_queries'] += 1
                
            elif request.query_type == "monitoring_network":
                data = await self._query_monitoring_network(request)
                self.performance_metrics['spatial_queries'] += 1
                
            elif request.query_type == "geological_context":
                data = await self._query_geological_context(request)
                self.performance_metrics['spatial_queries'] += 1
                
            else:
                raise ValueError(f"Unknown query type: {request.query_type}")
            
            # Calculate spatial extent
            spatial_extent = self._calculate_spatial_extent(data)
            
            # Create response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = GISQueryResponse(
                data=data,
                spatial_extent=spatial_extent,
                feature_count=len(data) if isinstance(data, list) else 1,
                coordinate_system=request.coordinate_system,
                processing_time=processing_time,
                metadata={
                    'query_type': request.query_type,
                    'filters_applied': {
                        'spatial': bool(request.spatial_filters),
                        'temporal': bool(request.temporal_filters),
                        'attribute': bool(request.attribute_filters)
                    },
                    'output_format': request.output_format
                }
            )
            
            # Cache result
            self._cache_result(cache_key, response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"GIS query execution failed: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return GISQueryResponse(
                data=None,
                spatial_extent=None,
                feature_count=0,
                coordinate_system=request.coordinate_system,
                processing_time=processing_time,
                metadata={'error': str(e)}
            )
    
    async def _query_volcanic_systems(self, request: GISQueryRequest) -> List[Dict[str, Any]]:
        """Query volcanic systems with spatial and attribute filters"""
        spatial_filters = request.spatial_filters or {}
        attribute_filters = request.attribute_filters or {}
        
        # Extract spatial parameters
        region_geometry = spatial_filters.get('region')
        buffer_distance = spatial_filters.get('buffer_distance')
        
        # Extract attribute parameters
        tectonic_settings = attribute_filters.get('tectonic_settings')
        hazard_levels = attribute_filters.get('hazard_levels')
        volcano_types = attribute_filters.get('volcano_types')
        
        # Execute spatial query
        if region_geometry:
            results = await self.gis_ops.query_volcanic_systems_by_region(
                region_geometry=region_geometry,
                buffer_distance=buffer_distance,
                tectonic_settings=tectonic_settings,
                hazard_levels=hazard_levels
            )
        else:
            # Fallback to attribute-only query
            results = await self._query_volcanic_systems_by_attributes(attribute_filters)
        
        # Apply additional filtering
        if volcano_types:
            results = [r for r in results if r.get('volcano_type') in volcano_types]
        
        return results
    
    async def _query_seismic_events(self, request: GISQueryRequest) -> List[Dict[str, Any]]:
        """Query seismic events with spatial-temporal filters"""
        spatial_filters = request.spatial_filters or {}
        temporal_filters = request.temporal_filters or {}
        attribute_filters = request.attribute_filters or {}
        
        # Extract parameters
        spatial_bounds = spatial_filters.get('bounds')
        time_range = temporal_filters.get('time_range')
        magnitude_range = attribute_filters.get('magnitude_range')
        depth_range = attribute_filters.get('depth_range')
        event_types = attribute_filters.get('event_types')
        
        if spatial_bounds and time_range:
            # Spatial-temporal query
            results = await self.gis_ops.query_seismic_events_spatial_temporal(
                spatial_bounds=spatial_bounds,
                time_range=time_range,
                magnitude_range=magnitude_range,
                depth_range=depth_range,
                event_types=event_types
            )
        else:
            # Fallback to basic query
            results = await self._query_seismic_events_basic(request)
        
        return results
    
    async def _query_deformation_data(self, request: GISQueryRequest) -> List[Dict[str, Any]]:
        """Query deformation measurements with temporal and spatial filters"""
        # Implementation for deformation data queries
        # This would integrate with MT-SAR processing and GPS data
        
        spatial_filters = request.spatial_filters or {}
        temporal_filters = request.temporal_filters or {}
        attribute_filters = request.attribute_filters or {}
        
        # Extract parameters
        volcanic_system_id = spatial_filters.get('volcanic_system_id')
        time_range = temporal_filters.get('time_range')
        measurement_types = attribute_filters.get('measurement_types', ['InSAR', 'GPS'])
        
        async with self.enhanced_db.get_postgresql_session() as session:
            # Build query for deformation measurements
            query = """
            SELECT 
                dm.id,
                dm.acquisition_time,
                ST_AsGeoJSON(dm.measurement_point) as location,
                dm.displacement_east,
                dm.displacement_north,
                dm.displacement_up,
                dm.line_of_sight_displacement,
                dm.coherence,
                dm.measurement_type,
                dm.satellite_platform,
                vs.name as volcano_name
            FROM deformation_measurements dm
            JOIN volcanic_systems vs ON dm.volcanic_system_id = vs.id
            WHERE 1=1
            """
            
            params = {}
            
            if volcanic_system_id:
                query += " AND dm.volcanic_system_id = :volcanic_system_id"
                params['volcanic_system_id'] = volcanic_system_id
            
            if time_range:
                query += " AND dm.acquisition_time BETWEEN :start_time AND :end_time"
                params['start_time'] = time_range[0]
                params['end_time'] = time_range[1]
            
            if measurement_types:
                query += " AND dm.measurement_type = ANY(:measurement_types)"
                params['measurement_types'] = measurement_types
            
            query += " ORDER BY dm.acquisition_time DESC"
            
            result = await session.execute(query, params)
            rows = result.fetchall()
            
            # Convert to dictionary format
            deformation_data = []
            for row in rows:
                data_dict = {
                    'id': str(row.id),
                    'acquisition_time': row.acquisition_time.isoformat(),
                    'location': json.loads(row.location),
                    'displacement': {
                        'east': row.displacement_east,
                        'north': row.displacement_north,
                        'up': row.displacement_up,
                        'line_of_sight': row.line_of_sight_displacement
                    },
                    'coherence': row.coherence,
                    'measurement_type': row.measurement_type,
                    'satellite_platform': row.satellite_platform,
                    'volcano_name': row.volcano_name
                }
                deformation_data.append(data_dict)
            
            return deformation_data
    
    async def _query_monitoring_network(self, request: GISQueryRequest) -> Dict[str, Any]:
        """Query monitoring network coverage and capabilities"""
        spatial_filters = request.spatial_filters or {}
        attribute_filters = request.attribute_filters or {}
        
        volcanic_system_id = spatial_filters.get('volcanic_system_id')
        station_types = attribute_filters.get('station_types')
        operational_only = attribute_filters.get('operational_only', True)
        
        if volcanic_system_id:
            # Query specific volcanic system
            result = await self.gis_ops.query_monitoring_network_coverage(
                volcanic_system_id=volcanic_system_id,
                station_types=station_types,
                operational_only=operational_only
            )
        else:
            # Query regional monitoring networks
            result = await self._query_regional_monitoring_networks(request)
        
        return result
    
    async def _query_geological_context(self, request: GISQueryRequest) -> List[Dict[str, Any]]:
        """Query geological context including units and tectonic features"""
        spatial_filters = request.spatial_filters or {}
        attribute_filters = request.attribute_filters or {}
        
        # Extract parameters
        region_geometry = spatial_filters.get('region')
        age_range = attribute_filters.get('age_range')
        lithologies = attribute_filters.get('lithologies')
        
        async with self.enhanced_db.get_postgresql_session() as session:
            # Query geological units
            units_query = """
            SELECT 
                gu.id,
                gu.unit_name,
                gu.formation_name,
                ST_AsGeoJSON(gu.geometry) as geometry,
                gu.age_min,
                gu.age_max,
                gu.primary_lithology,
                gu.geochemical_composition
            FROM geological_units gu
            WHERE 1=1
            """
            
            params = {}
            
            if region_geometry:
                if isinstance(region_geometry, dict):
                    geom_condition = "ST_Intersects(gu.geometry, ST_GeomFromGeoJSON(:region_geom))"
                    params['region_geom'] = json.dumps(region_geometry)
                else:
                    geom_condition = "ST_Intersects(gu.geometry, ST_GeomFromText(:region_geom, 4326))"
                    params['region_geom'] = region_geometry
                
                units_query += f" AND {geom_condition}"
            
            if age_range:
                units_query += " AND gu.age_min <= :max_age AND gu.age_max >= :min_age"
                params['min_age'] = age_range[0]
                params['max_age'] = age_range[1]
            
            if lithologies:
                units_query += " AND gu.primary_lithology = ANY(:lithologies)"
                params['lithologies'] = lithologies
            
            result = await session.execute(units_query, params)
            units_rows = result.fetchall()
            
            # Query tectonic features
            features_query = """
            SELECT 
                tf.id,
                tf.feature_name,
                tf.feature_type,
                ST_AsGeoJSON(tf.geometry) as geometry,
                tf.movement_sense,
                tf.activity_status,
                tf.slip_rate
            FROM tectonic_features tf
            WHERE 1=1
            """
            
            if region_geometry:
                if isinstance(region_geometry, dict):
                    geom_condition = "ST_Intersects(tf.geometry, ST_GeomFromGeoJSON(:region_geom))"
                else:
                    geom_condition = "ST_Intersects(tf.geometry, ST_GeomFromText(:region_geom, 4326))"
                
                features_query += f" AND {geom_condition}"
            
            result = await session.execute(features_query, params)
            features_rows = result.fetchall()
            
            # Combine results
            geological_context = {
                'geological_units': [
                    {
                        'id': str(row.id),
                        'unit_name': row.unit_name,
                        'formation_name': row.formation_name,
                        'geometry': json.loads(row.geometry),
                        'age_range': [row.age_min, row.age_max],
                        'primary_lithology': row.primary_lithology,
                        'geochemical_composition': row.geochemical_composition
                    }
                    for row in units_rows
                ],
                'tectonic_features': [
                    {
                        'id': str(row.id),
                        'feature_name': row.feature_name,
                        'feature_type': row.feature_type,
                        'geometry': json.loads(row.geometry),
                        'movement_sense': row.movement_sense,
                        'activity_status': row.activity_status,
                        'slip_rate': row.slip_rate
                    }
                    for row in features_rows
                ]
            }
            
            return [geological_context]  # Return as list for consistency
    
    # Data Export and Transformation
    async def export_spatial_data(self,
                                query_request: GISQueryRequest,
                                export_format: ExportFormat,
                                output_path: str,
                                coordinate_system: Optional[int] = None) -> Dict[str, Any]:
        """Export spatial data in various formats"""
        try:
            # Execute query to get data
            query_response = await self.execute_gis_query(query_request)
            
            if not query_response.data:
                return {'error': 'No data to export'}
            
            # Determine table name based on query type
            table_mapping = {
                'volcanic_systems': 'volcanic_systems',
                'seismic_events': 'seismic_events',
                'deformation_data': 'deformation_measurements',
                'monitoring_network': 'monitoring_stations',
                'geological_context': 'geological_units'
            }
            
            table_name = table_mapping.get(query_request.query_type)
            if not table_name:
                return {'error': f'Export not supported for query type: {query_request.query_type}'}
            
            # Build where clause from filters
            where_clause = self._build_where_clause(query_request)
            
            # Export data
            export_result = await self.gis_ops.export_geospatial_data(
                table_name=table_name,
                export_format=export_format,
                output_path=output_path,
                where_clause=where_clause,
                coordinate_system=coordinate_system
            )
            
            self.performance_metrics['data_exports'] += 1
            return export_result
            
        except Exception as e:
            self.logger.error(f"Failed to export spatial data: {e}")
            return {'error': str(e)}
    
    async def transform_coordinate_systems(self,
                                         data: List[Dict[str, Any]],
                                         source_crs: int,
                                         target_crs: int) -> List[Dict[str, Any]]:
        """Transform coordinate systems for spatial data"""
        try:
            # Extract geometries
            geometries = []
            for item in data:
                if 'geometry' in item:
                    if isinstance(item['geometry'], dict):
                        # GeoJSON geometry
                        wkt_geom = SpatialUtils.convert_geojson_to_wkt(item['geometry'])
                        geometries.append(wkt_geom)
                    else:
                        # WKT geometry
                        geometries.append(item['geometry'])
            
            if not geometries:
                return data
            
            # Transform coordinates
            transformed_geometries = await self.gis_ops.transform_coordinates(
                geometries=geometries,
                source_crs=source_crs,
                target_crs=target_crs
            )
            
            # Update data with transformed geometries
            transformed_data = []
            for i, item in enumerate(data):
                if i < len(transformed_geometries):
                    updated_item = item.copy()
                    # Convert back to GeoJSON if needed
                    if isinstance(item.get('geometry'), dict):
                        # Convert WKT back to GeoJSON (simplified)
                        updated_item['geometry'] = {
                            'type': 'Point',  # Simplified - would need proper conversion
                            'coordinates': []  # Would extract from WKT
                        }
                    else:
                        updated_item['geometry'] = transformed_geometries[i]
                    
                    transformed_data.append(updated_item)
                else:
                    transformed_data.append(item)
            
            self.performance_metrics['coordinate_transformations'] += 1
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Failed to transform coordinate systems: {e}")
            return data
    
    # Utility Methods
    def _generate_cache_key(self, request: GISQueryRequest) -> str:
        """Generate cache key for query request"""
        key_data = {
            'query_type': request.query_type,
            'spatial_filters': request.spatial_filters,
            'temporal_filters': request.temporal_filters,
            'attribute_filters': request.attribute_filters,
            'coordinate_system': request.coordinate_system
        }
        return f"gis_query_{hash(json.dumps(key_data, sort_keys=True))}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[GISQueryResponse]:
        """Get cached query result"""
        if cache_key in self.spatial_cache:
            result, timestamp = self.spatial_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                del self.spatial_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, response: GISQueryResponse):
        """Cache query result"""
        self.spatial_cache[cache_key] = (response, datetime.now())
    
    def _calculate_spatial_extent(self, data: Any) -> Optional[Dict[str, float]]:
        """Calculate spatial extent of query results"""
        if not data or not isinstance(data, list):
            return None
        
        try:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            
            for item in data:
                if 'location' in item and isinstance(item['location'], dict):
                    coords = item['location'].get('coordinates', [])
                    if len(coords) >= 2:
                        lon, lat = coords[0], coords[1]
                        min_x = min(min_x, lon)
                        max_x = max(max_x, lon)
                        min_y = min(min_y, lat)
                        max_y = max(max_y, lat)
            
            if min_x != float('inf'):
                return {
                    'min_x': min_x,
                    'min_y': min_y,
                    'max_x': max_x,
                    'max_y': max_y
                }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate spatial extent: {e}")
        
        return None
    
    def _build_where_clause(self, request: GISQueryRequest) -> Optional[str]:
        """Build SQL WHERE clause from query request filters"""
        conditions = []
        
        # Add attribute filters
        if request.attribute_filters:
            for key, value in request.attribute_filters.items():
                if isinstance(value, list):
                    conditions.append(f"{key} = ANY(ARRAY{value})")
                elif isinstance(value, tuple) and len(value) == 2:
                    conditions.append(f"{key} BETWEEN {value[0]} AND {value[1]}")
                else:
                    conditions.append(f"{key} = '{value}'")
        
        # Add temporal filters
        if request.temporal_filters:
            time_range = request.temporal_filters.get('time_range')
            if time_range:
                time_column = self._get_time_column(request.query_type)
                if time_column:
                    conditions.append(f"{time_column} BETWEEN '{time_range[0]}' AND '{time_range[1]}'")
        
        return " AND ".join(conditions) if conditions else None
    
    def _get_time_column(self, query_type: str) -> Optional[str]:
        """Get appropriate time column for query type"""
        time_columns = {
            'seismic_events': 'origin_time',
            'deformation_data': 'acquisition_time',
            'volcanic_systems': 'last_eruption'
        }
        return time_columns.get(query_type)
    
    # Performance and Monitoring
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            'integration_metrics': self.performance_metrics,
            'database_metrics': self.enhanced_db.get_database_metrics(),
            'spatial_metrics': self.gis_ops.get_spatial_metrics(),
            'cache_statistics': {
                'cache_size': len(self.spatial_cache),
                'cache_hit_rate': (
                    self.performance_metrics['cache_hits'] / 
                    max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                ) * 100
            }
        }
    
    async def cleanup(self):
        """Clean up all resources"""
        self.logger.info("Cleaning up GIS database integration")
        
        # Clear caches
        self.spatial_cache.clear()
        
        # Cleanup components
        await self.enhanced_db.cleanup()
        await self.gis_ops.cleanup()
        
        if hasattr(self.advanced_db, 'cleanup'):
            await self.advanced_db.cleanup()
        
        self.logger.info("GIS database integration cleanup complete")