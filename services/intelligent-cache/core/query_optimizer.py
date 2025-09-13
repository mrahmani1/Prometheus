"""
Advanced Query Optimization and Result Caching System

This module implements intelligent query optimization with geological domain awareness,
result caching, query plan optimization, and automated performance tuning.

Features:
- Intelligent query analysis and optimization
- Geological domain-aware query patterns
- Result caching with semantic understanding
- Query plan optimization and rewriting
- Automated index recommendations
- Performance monitoring and tuning
- Spatial and temporal query optimization
- Multi-database query coordination
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .cache_manager import DataType, IntelligentCacheManager


class QueryType(Enum):
    """Types of geological queries"""
    SPATIAL_RANGE = "spatial_range"
    TEMPORAL_SERIES = "temporal_series"
    SEISMIC_ANALYSIS = "seismic_analysis"
    VOLCANIC_MONITORING = "volcanic_monitoring"
    GEOLOGICAL_MODEL = "geological_model"
    RESOURCE_ASSESSMENT = "resource_assessment"
    REAL_TIME_STREAM = "real_time_stream"


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    REDIS = "redis"


@dataclass
class QueryPattern:
    """Query pattern for optimization"""
    pattern_id: str
    query_type: QueryType
    database_type: DatabaseType
    frequency: int
    avg_execution_time: float
    spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    geological_context: Dict[str, Any] = field(default_factory=dict)
    optimization_hints: List[str] = field(default_factory=list)


@dataclass
class QueryResult:
    """Query result with caching metadata"""
    query_hash: str
    result_data: Any
    execution_time: float
    cache_key: str
    geological_context: Dict[str, Any]
    spatial_coverage: Optional[Tuple[float, float, float, float]] = None
    temporal_coverage: Optional[Tuple[datetime, datetime]] = None
    result_size: int = 0
    confidence_score: float = 1.0


class GeologicalQueryAnalyzer:
    """Analyzes queries for geological domain optimization"""
    
    def __init__(self):
        self.volcanic_keywords = {
            'eruption', 'magma', 'lava', 'volcanic', 'caldera', 'crater',
            'ash', 'pyroclastic', 'tephra', 'fumarole', 'hydrothermal'
        }
        
        self.seismic_keywords = {
            'earthquake', 'seismic', 'magnitude', 'epicenter', 'hypocenter',
            'p-wave', 's-wave', 'tremor', 'swarm', 'foreshock', 'aftershock'
        }
        
        self.geological_keywords = {
            'rock', 'mineral', 'formation', 'stratum', 'lithology', 'facies',
            'structure', 'fault', 'fold', 'joint', 'fracture', 'ore'
        }
        
        self.spatial_patterns = [
            r'ST_Within|ST_Intersects|ST_DWithin|ST_Contains',
            r'latitude|longitude|lat|lon|x|y|z|elevation|depth',
            r'bbox|bounding.*box|spatial.*range|geographic.*extent'
        ]
        
        self.temporal_patterns = [
            r'timestamp|datetime|time|date',
            r'between.*and|since|until|before|after',
            r'interval|duration|period|range'
        ]
    
    def analyze_query(self, query: str, parameters: Dict[str, Any] = None) -> QueryPattern:
        """Analyze query to determine type and optimization opportunities"""
        query_lower = query.lower()
        parameters = parameters or {}
        
        # Determine query type
        query_type = self._determine_query_type(query_lower)
        
        # Determine database type
        database_type = self._determine_database_type(query_lower)
        
        # Extract spatial bounds
        spatial_bounds = self._extract_spatial_bounds(query_lower, parameters)
        
        # Extract temporal range
        temporal_range = self._extract_temporal_range(query_lower, parameters)
        
        # Extract geological context
        geological_context = self._extract_geological_context(query_lower, parameters)
        
        # Generate optimization hints
        optimization_hints = self._generate_optimization_hints(
            query_lower, query_type, spatial_bounds, temporal_range
        )
        
        pattern = QueryPattern(
            pattern_id=str(uuid4()),
            query_type=query_type,
            database_type=database_type,
            frequency=1,
            avg_execution_time=0.0,
            spatial_bounds=spatial_bounds,
            temporal_range=temporal_range,
            geological_context=geological_context,
            optimization_hints=optimization_hints
        )
        
        return pattern
    
    def _determine_query_type(self, query: str) -> QueryType:
        """Determine the type of geological query"""
        if any(keyword in query for keyword in self.volcanic_keywords):
            return QueryType.VOLCANIC_MONITORING
        elif any(keyword in query for keyword in self.seismic_keywords):
            return QueryType.SEISMIC_ANALYSIS
        elif 'st_' in query or any(pattern in query for pattern in ['spatial', 'geographic', 'geometry']):
            return QueryType.SPATIAL_RANGE
        elif any(pattern in query for pattern in ['time', 'date', 'temporal']):
            return QueryType.TEMPORAL_SERIES
        elif any(keyword in query for keyword in self.geological_keywords):
            return QueryType.GEOLOGICAL_MODEL
        elif 'resource' in query or 'mineral' in query or 'ore' in query:
            return QueryType.RESOURCE_ASSESSMENT
        else:
            return QueryType.SPATIAL_RANGE  # Default
    
    def _determine_database_type(self, query: str) -> DatabaseType:
        """Determine target database type from query"""
        if any(func in query for func in ['st_', 'geometry', 'geography']):
            return DatabaseType.POSTGRESQL
        elif any(func in query for func in ['select', 'from', 'where', 'group by']):
            return DatabaseType.POSTGRESQL
        elif any(func in query for func in ['match', 'create', 'merge', 'cypher']):
            return DatabaseType.NEO4J
        elif 'get' in query or 'set' in query or 'hget' in query:
            return DatabaseType.REDIS
        else:
            return DatabaseType.INFLUXDB
    
    def _extract_spatial_bounds(self, query: str, parameters: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
        """Extract spatial bounds from query"""
        # Look for bounding box parameters
        if 'bbox' in parameters:
            bbox = parameters['bbox']
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return tuple(bbox)
        
        # Look for lat/lon ranges in parameters
        lat_min = parameters.get('lat_min') or parameters.get('latitude_min')
        lat_max = parameters.get('lat_max') or parameters.get('latitude_max')
        lon_min = parameters.get('lon_min') or parameters.get('longitude_min')
        lon_max = parameters.get('lon_max') or parameters.get('longitude_max')
        
        if all(v is not None for v in [lat_min, lat_max, lon_min, lon_max]):
            return (float(lon_min), float(lat_min), float(lon_max), float(lat_max))
        
        return None
    
    def _extract_temporal_range(self, query: str, parameters: Dict[str, Any]) -> Optional[Tuple[datetime, datetime]]:
        """Extract temporal range from query"""
        start_time = parameters.get('start_time') or parameters.get('time_start')
        end_time = parameters.get('end_time') or parameters.get('time_end')
        
        if start_time and end_time:
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            
            return (start_time, end_time)
        
        return None
    
    def _extract_geological_context(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geological context from query"""
        context = {}
        
        # Extract volcano ID
        volcano_id = parameters.get('volcano_id')
        if volcano_id:
            context['volcano_id'] = volcano_id
        
        # Extract geological features
        if any(keyword in query for keyword in self.volcanic_keywords):
            context['domain'] = 'volcanic'
        elif any(keyword in query for keyword in self.seismic_keywords):
            context['domain'] = 'seismic'
        elif any(keyword in query for keyword in self.geological_keywords):
            context['domain'] = 'geological'
        
        # Extract data resolution
        if 'high_resolution' in query or parameters.get('resolution') == 'high':
            context['resolution'] = 'high'
        elif 'low_resolution' in query or parameters.get('resolution') == 'low':
            context['resolution'] = 'low'
        
        return context
    
    def _generate_optimization_hints(self, query: str, query_type: QueryType,
                                   spatial_bounds: Optional[Tuple[float, float, float, float]],
                                   temporal_range: Optional[Tuple[datetime, datetime]]) -> List[str]:
        """Generate optimization hints for query"""
        hints = []
        
        # Spatial optimization hints
        if spatial_bounds:
            hints.append("use_spatial_index")
            if query_type == QueryType.VOLCANIC_MONITORING:
                hints.append("volcanic_spatial_partitioning")
        
        # Temporal optimization hints
        if temporal_range:
            hints.append("use_temporal_index")
            time_span = temporal_range[1] - temporal_range[0]
            if time_span.days > 365:
                hints.append("long_term_partitioning")
            elif time_span.total_seconds() < 3600:
                hints.append("real_time_optimization")
        
        # Query type specific hints
        if query_type == QueryType.SEISMIC_ANALYSIS:
            hints.extend(["seismic_waveform_compression", "parallel_processing"])
        elif query_type == QueryType.VOLCANIC_MONITORING:
            hints.extend(["real_time_caching", "alert_prioritization"])
        elif query_type == QueryType.GEOLOGICAL_MODEL:
            hints.extend(["model_caching", "3d_spatial_index"])
        
        # General optimization hints
        if 'group by' in query:
            hints.append("aggregation_optimization")
        if 'order by' in query:
            hints.append("sorting_optimization")
        if 'join' in query:
            hints.append("join_optimization")
        
        return hints


class QueryOptimizer:
    """Advanced query optimizer with geological domain awareness"""
    
    def __init__(self, cache_manager: IntelligentCacheManager):
        self.cache_manager = cache_manager
        self.query_analyzer = GeologicalQueryAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Query pattern tracking
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.execution_history: List[Tuple[str, float, datetime]] = []
        
        # Optimization rules
        self.optimization_rules = {
            QueryType.SPATIAL_RANGE: self._optimize_spatial_query,
            QueryType.TEMPORAL_SERIES: self._optimize_temporal_query,
            QueryType.SEISMIC_ANALYSIS: self._optimize_seismic_query,
            QueryType.VOLCANIC_MONITORING: self._optimize_volcanic_query,
            QueryType.GEOLOGICAL_MODEL: self._optimize_geological_query,
            QueryType.RESOURCE_ASSESSMENT: self._optimize_resource_query
        }
        
        # Index recommendations
        self.recommended_indexes: Dict[str, List[str]] = defaultdict(list)
    
    async def optimize_and_execute(self, query: str, parameters: Dict[str, Any] = None,
                                 database_session: Optional[AsyncSession] = None) -> QueryResult:
        """Optimize query and execute with intelligent caching"""
        start_time = time.time()
        
        try:
            # Analyze query
            pattern = self.query_analyzer.analyze_query(query, parameters)
            
            # Generate cache key
            cache_key = self._generate_cache_key(query, parameters)
            
            # Check cache first
            cached_result = await self.cache_manager.get(cache_key)
            if cached_result:
                self.logger.info(f"Cache hit for query: {cache_key[:50]}...")
                
                execution_time = time.time() - start_time
                return QueryResult(
                    query_hash=cache_key,
                    result_data=cached_result,
                    execution_time=execution_time,
                    cache_key=cache_key,
                    geological_context=pattern.geological_context,
                    spatial_coverage=pattern.spatial_bounds,
                    temporal_coverage=pattern.temporal_range
                )
            
            # Optimize query
            optimized_query, optimized_params = await self._optimize_query(
                query, parameters, pattern
            )
            
            # Execute optimized query
            result_data = await self._execute_query(
                optimized_query, optimized_params, database_session, pattern
            )
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = QueryResult(
                query_hash=cache_key,
                result_data=result_data,
                execution_time=execution_time,
                cache_key=cache_key,
                geological_context=pattern.geological_context,
                spatial_coverage=pattern.spatial_bounds,
                temporal_coverage=pattern.temporal_range,
                result_size=self._calculate_result_size(result_data)
            )
            
            # Cache result if beneficial
            if await self._should_cache_result(result, pattern):
                await self._cache_result(result, pattern)
            
            # Update query patterns
            await self._update_query_pattern(pattern, execution_time)
            
            # Record execution history
            self.execution_history.append((cache_key, execution_time, datetime.now()))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query optimization/execution error: {e}")
            raise
    
    async def _optimize_query(self, query: str, parameters: Dict[str, Any],
                            pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Apply optimization rules based on query pattern"""
        if pattern.query_type in self.optimization_rules:
            optimizer_func = self.optimization_rules[pattern.query_type]
            return await optimizer_func(query, parameters, pattern)
        else:
            return query, parameters
    
    async def _optimize_spatial_query(self, query: str, parameters: Dict[str, Any],
                                    pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize spatial queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Add spatial index hints
        if 'use_spatial_index' in pattern.optimization_hints:
            if 'ST_' in query and 'USING GIST' not in query:
                # Suggest spatial index usage
                self.recommended_indexes['spatial'].append(
                    "CREATE INDEX IF NOT EXISTS idx_spatial_geom ON table_name USING GIST (geom);"
                )
        
        # Optimize bounding box queries
        if pattern.spatial_bounds:
            bbox = pattern.spatial_bounds
            bbox_condition = f"""
            AND ST_Intersects(geom, ST_MakeEnvelope({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}, 4326))
            """
            
            if 'WHERE' in optimized_query and 'ST_MakeEnvelope' not in optimized_query:
                optimized_query = optimized_query.replace('WHERE', f'WHERE {bbox_condition.strip()} AND')
        
        # Add spatial clustering hints for volcanic data
        if pattern.geological_context.get('domain') == 'volcanic':
            if 'volcanic_spatial_partitioning' in pattern.optimization_hints:
                # Add volcano-specific spatial optimization
                optimized_params['_spatial_partition'] = 'volcanic_regions'
        
        return optimized_query, optimized_params
    
    async def _optimize_temporal_query(self, query: str, parameters: Dict[str, Any],
                                     pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize temporal queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Add temporal index hints
        if 'use_temporal_index' in pattern.optimization_hints:
            self.recommended_indexes['temporal'].append(
                "CREATE INDEX IF NOT EXISTS idx_temporal_timestamp ON table_name (timestamp);"
            )
        
        # Optimize for real-time queries
        if 'real_time_optimization' in pattern.optimization_hints:
            # Add LIMIT for real-time queries to prevent large result sets
            if 'LIMIT' not in optimized_query:
                optimized_query += " LIMIT 10000"
            
            # Add recent data optimization
            if pattern.temporal_range:
                start_time, end_time = pattern.temporal_range
                if (end_time - start_time).total_seconds() < 3600:  # Less than 1 hour
                    optimized_params['_real_time_cache'] = True
        
        # Partition optimization for long-term queries
        if 'long_term_partitioning' in pattern.optimization_hints:
            optimized_params['_use_partitioning'] = True
        
        return optimized_query, optimized_params
    
    async def _optimize_seismic_query(self, query: str, parameters: Dict[str, Any],
                                    pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize seismic analysis queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Add seismic-specific optimizations
        if 'seismic_waveform_compression' in pattern.optimization_hints:
            optimized_params['_compression'] = 'seismic_wavelet'
        
        if 'parallel_processing' in pattern.optimization_hints:
            optimized_params['_parallel'] = True
            optimized_params['_worker_count'] = 4
        
        # Optimize magnitude range queries
        if 'magnitude' in query.lower():
            # Add magnitude index recommendation
            self.recommended_indexes['seismic'].append(
                "CREATE INDEX IF NOT EXISTS idx_seismic_magnitude ON seismic_events (magnitude);"
            )
        
        # Optimize depth queries
        if 'depth' in query.lower():
            self.recommended_indexes['seismic'].append(
                "CREATE INDEX IF NOT EXISTS idx_seismic_depth ON seismic_events (depth);"
            )
        
        return optimized_query, optimized_params
    
    async def _optimize_volcanic_query(self, query: str, parameters: Dict[str, Any],
                                     pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize volcanic monitoring queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Real-time caching for volcanic data
        if 'real_time_caching' in pattern.optimization_hints:
            optimized_params['_cache_ttl'] = 300  # 5 minutes for real-time data
        
        # Alert prioritization
        if 'alert_prioritization' in pattern.optimization_hints:
            # Prioritize high alert level data
            if 'alert_level' in query.lower():
                if 'ORDER BY' not in optimized_query:
                    optimized_query += " ORDER BY alert_level DESC, timestamp DESC"
        
        # Volcano-specific indexes
        volcano_id = pattern.geological_context.get('volcano_id')
        if volcano_id:
            self.recommended_indexes['volcanic'].append(
                f"CREATE INDEX IF NOT EXISTS idx_volcanic_{volcano_id} ON volcanic_data (volcano_id, timestamp);"
            )
        
        return optimized_query, optimized_params
    
    async def _optimize_geological_query(self, query: str, parameters: Dict[str, Any],
                                       pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize geological model queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Model caching optimization
        if 'model_caching' in pattern.optimization_hints:
            optimized_params['_model_cache'] = True
            optimized_params['_cache_ttl'] = 3600  # 1 hour for geological models
        
        # 3D spatial index optimization
        if '3d_spatial_index' in pattern.optimization_hints:
            self.recommended_indexes['geological'].append(
                "CREATE INDEX IF NOT EXISTS idx_geological_3d ON geological_models USING GIST (geom_3d);"
            )
        
        # Lithology optimization
        if 'lithology' in query.lower():
            self.recommended_indexes['geological'].append(
                "CREATE INDEX IF NOT EXISTS idx_geological_lithology ON geological_units (lithology);"
            )
        
        return optimized_query, optimized_params
    
    async def _optimize_resource_query(self, query: str, parameters: Dict[str, Any],
                                     pattern: QueryPattern) -> Tuple[str, Dict[str, Any]]:
        """Optimize resource assessment queries"""
        optimized_query = query
        optimized_params = parameters.copy()
        
        # Resource-specific optimizations
        if 'resource' in query.lower() or 'mineral' in query.lower():
            # Add resource type index
            self.recommended_indexes['resource'].append(
                "CREATE INDEX IF NOT EXISTS idx_resource_type ON resources (resource_type);"
            )
            
            # Add grade index for ore deposits
            if 'grade' in query.lower():
                self.recommended_indexes['resource'].append(
                    "CREATE INDEX IF NOT EXISTS idx_resource_grade ON resources (grade);"
                )
        
        # Economic optimization
        if 'economic' in query.lower() or 'value' in query.lower():
            optimized_params['_economic_cache'] = True
        
        return optimized_query, optimized_params
    
    async def _execute_query(self, query: str, parameters: Dict[str, Any],
                           database_session: Optional[AsyncSession],
                           pattern: QueryPattern) -> Any:
        """Execute optimized query"""
        if database_session:
            # Execute SQL query
            result = await database_session.execute(text(query), parameters)
            return result.fetchall()
        else:
            # For demonstration, return mock data based on query type
            return self._generate_mock_result(pattern)
    
    def _generate_mock_result(self, pattern: QueryPattern) -> Any:
        """Generate mock result for testing"""
        if pattern.query_type == QueryType.SEISMIC_ANALYSIS:
            return [
                {"event_id": f"eq_{i}", "magnitude": 3.0 + i * 0.1, "depth": 10.0 + i}
                for i in range(100)
            ]
        elif pattern.query_type == QueryType.VOLCANIC_MONITORING:
            return [
                {"volcano_id": "test_volcano", "alert_level": 2, "timestamp": datetime.now()}
            ]
        else:
            return {"result": "mock_data", "count": 100}
    
    def _generate_cache_key(self, query: str, parameters: Dict[str, Any] = None) -> str:
        """Generate unique cache key for query"""
        query_normalized = ' '.join(query.split()).lower()
        params_str = json.dumps(parameters or {}, sort_keys=True, default=str)
        
        combined = f"{query_normalized}|{params_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _calculate_result_size(self, result_data: Any) -> int:
        """Calculate size of result data"""
        if isinstance(result_data, list):
            return len(result_data)
        elif isinstance(result_data, dict):
            return len(str(result_data))
        else:
            return len(str(result_data))
    
    async def _should_cache_result(self, result: QueryResult, pattern: QueryPattern) -> bool:
        """Determine if result should be cached"""
        # Cache if execution time is significant
        if result.execution_time > 1.0:  # More than 1 second
            return True
        
        # Cache if result is large
        if result.result_size > 1000:
            return True
        
        # Cache real-time data with short TTL
        if pattern.query_type == QueryType.REAL_TIME_STREAM:
            return True
        
        # Cache geological models (expensive to compute)
        if pattern.query_type == QueryType.GEOLOGICAL_MODEL:
            return True
        
        return False
    
    async def _cache_result(self, result: QueryResult, pattern: QueryPattern):
        """Cache query result with appropriate TTL"""
        # Determine TTL based on query type
        ttl_mapping = {
            QueryType.REAL_TIME_STREAM: timedelta(minutes=5),
            QueryType.VOLCANIC_MONITORING: timedelta(minutes=15),
            QueryType.SEISMIC_ANALYSIS: timedelta(hours=1),
            QueryType.GEOLOGICAL_MODEL: timedelta(hours=24),
            QueryType.RESOURCE_ASSESSMENT: timedelta(hours=12),
            QueryType.SPATIAL_RANGE: timedelta(hours=6),
            QueryType.TEMPORAL_SERIES: timedelta(hours=2)
        }
        
        ttl = ttl_mapping.get(pattern.query_type, timedelta(hours=1))
        
        # Determine data type for cache
        data_type_mapping = {
            QueryType.SEISMIC_ANALYSIS: DataType.SEISMIC_WAVEFORM,
            QueryType.VOLCANIC_MONITORING: DataType.VOLCANIC_STATE,
            QueryType.GEOLOGICAL_MODEL: DataType.GEOLOGICAL_MODEL,
            QueryType.REAL_TIME_STREAM: DataType.REAL_TIME_STREAM
        }
        
        data_type = data_type_mapping.get(pattern.query_type, DataType.SIMULATION_RESULT)
        
        # Cache the result
        await self.cache_manager.set(
            key=result.cache_key,
            data=result.result_data,
            data_type=data_type,
            geological_context=result.geological_context,
            ttl=ttl
        )
    
    async def _update_query_pattern(self, pattern: QueryPattern, execution_time: float):
        """Update query pattern statistics"""
        pattern_key = f"{pattern.query_type.value}_{pattern.database_type.value}"
        
        if pattern_key in self.query_patterns:
            existing_pattern = self.query_patterns[pattern_key]
            existing_pattern.frequency += 1
            existing_pattern.avg_execution_time = (
                existing_pattern.avg_execution_time * 0.9 + execution_time * 0.1
            )
        else:
            pattern.frequency = 1
            pattern.avg_execution_time = execution_time
            self.query_patterns[pattern_key] = pattern
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get query optimization recommendations"""
        recommendations = {
            "indexes": dict(self.recommended_indexes),
            "slow_queries": [],
            "frequent_patterns": [],
            "cache_efficiency": {}
        }
        
        # Identify slow queries
        for pattern_key, pattern in self.query_patterns.items():
            if pattern.avg_execution_time > 5.0:  # Slower than 5 seconds
                recommendations["slow_queries"].append({
                    "pattern": pattern_key,
                    "avg_time": pattern.avg_execution_time,
                    "frequency": pattern.frequency,
                    "hints": pattern.optimization_hints
                })
        
        # Identify frequent patterns
        frequent_patterns = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1].frequency,
            reverse=True
        )[:10]
        
        recommendations["frequent_patterns"] = [
            {
                "pattern": pattern_key,
                "frequency": pattern.frequency,
                "avg_time": pattern.avg_execution_time
            }
            for pattern_key, pattern in frequent_patterns
        ]
        
        # Cache efficiency metrics
        if hasattr(self.cache_manager, 'get_metrics'):
            cache_metrics = self.cache_manager.get_metrics()
            recommendations["cache_efficiency"] = {
                "hit_rate": cache_metrics.hit_rate,
                "miss_rate": cache_metrics.miss_rate,
                "compression_ratio": cache_metrics.compression_ratio
            }
        
        return recommendations
    
    async def apply_optimization_recommendations(self, database_session: AsyncSession):
        """Apply recommended optimizations to database"""
        try:
            # Apply index recommendations
            for category, indexes in self.recommended_indexes.items():
                for index_sql in indexes:
                    try:
                        await database_session.execute(text(index_sql))
                        self.logger.info(f"Applied index: {index_sql}")
                    except Exception as e:
                        self.logger.warning(f"Failed to apply index {index_sql}: {e}")
            
            await database_session.commit()
            
            # Clear applied recommendations
            self.recommended_indexes.clear()
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization recommendations: {e}")
            await database_session.rollback()


# Query optimization configuration
class QueryOptimizerConfig(BaseModel):
    """Query optimizer configuration"""
    enable_caching: bool = Field(default=True, description="Enable result caching")
    cache_ttl_hours: int = Field(default=1, description="Default cache TTL in hours")
    slow_query_threshold: float = Field(default=5.0, description="Slow query threshold in seconds")
    auto_index_creation: bool = Field(default=False, description="Automatically create recommended indexes")
    geological_optimization: bool = Field(default=True, description="Enable geological domain optimization")
    parallel_execution: bool = Field(default=True, description="Enable parallel query execution")


async def create_query_optimizer(cache_manager: IntelligentCacheManager,
                               config: QueryOptimizerConfig) -> QueryOptimizer:
    """Factory function to create query optimizer"""
    optimizer = QueryOptimizer(cache_manager)
    return optimizer


# Example usage and testing
if __name__ == "__main__":
    async def test_query_optimizer():
        """Test the query optimizer"""
        from .cache_manager import create_cache_manager, CacheConfig
        
        # Create cache manager
        cache_config = CacheConfig()
        cache_manager = await create_cache_manager(cache_config)
        
        # Create query optimizer
        optimizer_config = QueryOptimizerConfig()
        optimizer = await create_query_optimizer(cache_manager, optimizer_config)
        
        # Test spatial query
        spatial_query = """
        SELECT volcano_id, name, ST_AsText(location) as location
        FROM volcanoes 
        WHERE ST_DWithin(location, ST_Point(-155.0, 19.5), 50000)
        AND alert_level >= 2
        ORDER BY alert_level DESC
        """
        
        spatial_params = {
            "lat_min": 19.0,
            "lat_max": 20.0,
            "lon_min": -156.0,
            "lon_max": -154.0,
            "volcano_id": "kilauea_001"
        }
        
        # Execute optimized query
        result = await optimizer.optimize_and_execute(spatial_query, spatial_params)
        print(f"Spatial query result: {len(result.result_data) if isinstance(result.result_data, list) else 'N/A'} records")
        print(f"Execution time: {result.execution_time:.3f} seconds")
        
        # Test seismic query
        seismic_query = """
        SELECT event_id, magnitude, depth, timestamp
        FROM seismic_events
        WHERE magnitude >= 4.0
        AND timestamp BETWEEN %s AND %s
        ORDER BY magnitude DESC
        """
        
        seismic_params = {
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z"
        }
        
        result = await optimizer.optimize_and_execute(seismic_query, seismic_params)
        print(f"Seismic query result: {len(result.result_data) if isinstance(result.result_data, list) else 'N/A'} records")
        print(f"Execution time: {result.execution_time:.3f} seconds")
        
        # Get optimization recommendations
        recommendations = optimizer.get_optimization_recommendations()
        print(f"Optimization recommendations: {recommendations}")
    
    # Run test
    asyncio.run(test_query_optimizer())