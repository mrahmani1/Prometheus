"""
Cross-Database Query Optimization System
Advanced query optimization, routing, caching, and performance monitoring
for multi-database geological analysis platform
"""

import asyncio
import logging
import json
import time
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import zlib

# Database clients and types
from backend.core.unified_database_manager import DatabaseType, UnifiedDatabaseManager, QueryResult
from backend.core.database_config_manager import DatabaseConfig, get_config_manager

# Performance monitoring
import psutil
import asyncpg
import redis.asyncio as redis

# Query parsing and optimization
import sqlparse
from sqlparse import sql, tokens
import re

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of database queries"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    GRAPH = "graph"
    TIME_SERIES = "time_series"
    CROSS_DATABASE = "cross_database"

class OptimizationStrategy(Enum):
    """Query optimization strategies"""
    INDEX_HINT = "index_hint"
    PARTITION_PRUNING = "partition_pruning"
    PARALLEL_EXECUTION = "parallel_execution"
    RESULT_CACHING = "result_caching"
    QUERY_REWRITE = "query_rewrite"
    CROSS_DB_JOIN = "cross_db_join"
    MATERIALIZED_VIEW = "materialized_view"
    BATCH_PROCESSING = "batch_processing"

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    execution_time: float
    rows_processed: int
    memory_usage: float
    cpu_usage: float
    cache_hit: bool
    optimization_applied: List[OptimizationStrategy]
    database_type: DatabaseType
    query_hash: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class QueryPlan:
    """Optimized query execution plan"""
    original_query: str
    optimized_queries: Dict[DatabaseType, str]
    execution_order: List[Tuple[DatabaseType, str]]
    join_strategy: Optional[str]
    caching_strategy: Optional[str]
    estimated_cost: float
    parallelizable: bool
    requires_aggregation: bool

@dataclass
class CacheEntry:
    """Query result cache entry"""
    result: Any
    metadata: Dict[str, Any]
    created_at: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    ttl: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    size_bytes: int = 0

class QueryRouter:
    """Intelligent query routing based on data type and performance"""
    
    def __init__(self):
        self.routing_rules: Dict[str, DatabaseType] = {}
        self.performance_history: Dict[DatabaseType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.load_balancing_weights: Dict[DatabaseType, float] = {}
        
        # Initialize default routing rules
        self._initialize_routing_rules()
    
    def _initialize_routing_rules(self):
        """Initialize default data type to database routing rules"""
        self.routing_rules = {
            # Spatial data patterns
            'ST_': DatabaseType.POSTGRESQL,
            'geometry': DatabaseType.POSTGRESQL,
            'location': DatabaseType.POSTGRESQL,
            'coordinates': DatabaseType.POSTGRESQL,
            'spatial': DatabaseType.POSTGRESQL,
            
            # Time-series patterns
            'time_series': DatabaseType.INFLUXDB,
            'measurement': DatabaseType.INFLUXDB,
            'sensor_data': DatabaseType.INFLUXDB,
            'real_time': DatabaseType.INFLUXDB,
            'monitoring': DatabaseType.INFLUXDB,
            
            # Graph patterns
            'MATCH': DatabaseType.NEO4J,
            'RETURN': DatabaseType.NEO4J,
            'relationship': DatabaseType.NEO4J,
            'causal': DatabaseType.NEO4J,
            'network': DatabaseType.NEO4J,
            
            # Cache patterns
            'cache': DatabaseType.REDIS,
            'session': DatabaseType.REDIS,
            'temporary': DatabaseType.REDIS,
            'quick_lookup': DatabaseType.REDIS,
        }
        
        # Initialize load balancing weights
        for db_type in DatabaseType:
            self.load_balancing_weights[db_type] = 1.0
    
    def route_query(self, query: str, query_type: QueryType, 
                   preferred_db: Optional[DatabaseType] = None) -> DatabaseType:
        """Route query to optimal database based on content and performance"""
        
        # Use preferred database if specified and available
        if preferred_db:
            return preferred_db
        
        # Analyze query content for routing hints
        query_lower = query.lower()
        
        # Check for explicit routing patterns
        for pattern, db_type in self.routing_rules.items():
            if pattern.lower() in query_lower:
                # Consider load balancing
                if self._is_database_overloaded(db_type):
                    alternative = self._find_alternative_database(db_type, query_type)
                    if alternative:
                        return alternative
                return db_type
        
        # Default routing based on query type
        return self._get_default_database_for_query_type(query_type)
    
    def _is_database_overloaded(self, db_type: DatabaseType) -> bool:
        """Check if database is currently overloaded"""
        recent_metrics = list(self.performance_history[db_type])[-10:]  # Last 10 queries
        
        if len(recent_metrics) < 5:
            return False
        
        avg_response_time = statistics.mean([m.execution_time for m in recent_metrics])
        return avg_response_time > 5.0  # 5 second threshold
    
    def _find_alternative_database(self, overloaded_db: DatabaseType, 
                                 query_type: QueryType) -> Optional[DatabaseType]:
        """Find alternative database for overloaded primary choice"""
        alternatives = {
            DatabaseType.POSTGRESQL: [DatabaseType.TIMESCALEDB],
            DatabaseType.INFLUXDB: [DatabaseType.TIMESCALEDB, DatabaseType.POSTGRESQL],
            DatabaseType.NEO4J: [DatabaseType.POSTGRESQL],  # For simple graph queries
            DatabaseType.REDIS: [DatabaseType.POSTGRESQL],  # For caching fallback
        }
        
        for alt_db in alternatives.get(overloaded_db, []):
            if not self._is_database_overloaded(alt_db):
                return alt_db
        
        return None
    
    def _get_default_database_for_query_type(self, query_type: QueryType) -> DatabaseType:
        """Get default database for query type"""
        defaults = {
            QueryType.SPATIAL: DatabaseType.POSTGRESQL,
            QueryType.TEMPORAL: DatabaseType.INFLUXDB,
            QueryType.TIME_SERIES: DatabaseType.INFLUXDB,
            QueryType.GRAPH: DatabaseType.NEO4J,
            QueryType.AGGREGATE: DatabaseType.POSTGRESQL,
            QueryType.CROSS_DATABASE: DatabaseType.POSTGRESQL,  # Coordinator
        }
        
        return defaults.get(query_type, DatabaseType.POSTGRESQL)
    
    def update_performance_metrics(self, db_type: DatabaseType, metrics: QueryMetrics):
        """Update performance history for load balancing"""
        self.performance_history[db_type].append(metrics)
        
        # Update load balancing weights based on recent performance
        recent_metrics = list(self.performance_history[db_type])[-20:]
        if len(recent_metrics) >= 10:
            avg_time = statistics.mean([m.execution_time for m in recent_metrics])
            # Lower weight for slower databases
            self.load_balancing_weights[db_type] = max(0.1, 1.0 / (1.0 + avg_time))

class QueryCache:
    """Advanced query result caching with intelligent eviction"""
    
    def __init__(self, max_size_mb: int = 1024, redis_client: Optional[redis.Redis] = None):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()
        self.redis_client = redis_client
        self.lock = asyncio.Lock()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_cache_key(self, query: str, params: Optional[Dict] = None, 
                          db_type: DatabaseType = None) -> str:
        """Generate cache key for query"""
        key_data = {
            'query': query.strip().lower(),
            'params': params or {},
            'db_type': db_type.value if db_type else None
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def get(self, query: str, params: Optional[Dict] = None, 
                 db_type: DatabaseType = None) -> Optional[QueryResult]:
        """Get cached query result"""
        cache_key = self._generate_cache_key(query, params, db_type)
        
        async with self.lock:
            # Check local cache first
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                # Check TTL
                if datetime.utcnow() - entry.created_at > entry.ttl:
                    await self._evict_entry(cache_key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = datetime.utcnow()
                self.access_order.append(cache_key)
                self.hits += 1
                
                # Reconstruct QueryResult
                return QueryResult(
                    data=entry.result,
                    metadata=entry.metadata,
                    execution_time=0.001,  # Cache hit time
                    row_count=entry.metadata.get('row_count', 0),
                    cached=True,
                    database_type=db_type or DatabaseType.POSTGRESQL
                )
        
        # Check Redis cache if available
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"query_cache:{cache_key}")
                if cached_data:
                    entry_data = pickle.loads(zlib.decompress(cached_data))
                    self.hits += 1
                    
                    return QueryResult(
                        data=entry_data['result'],
                        metadata=entry_data['metadata'],
                        execution_time=0.002,  # Redis cache hit time
                        row_count=entry_data['metadata'].get('row_count', 0),
                        cached=True,
                        database_type=db_type or DatabaseType.POSTGRESQL
                    )
            except Exception as e:
                logger.warning(f"Redis cache lookup failed: {e}")
        
        self.misses += 1
        return None
    
    async def put(self, query: str, result: QueryResult, params: Optional[Dict] = None,
                 ttl: Optional[timedelta] = None):
        """Cache query result"""
        cache_key = self._generate_cache_key(query, params, result.database_type)
        
        # Serialize result for size calculation
        serialized_result = pickle.dumps(result.data)
        entry_size = len(serialized_result)
        
        # Skip caching if result is too large
        if entry_size > self.max_size_bytes * 0.1:  # Don't cache items > 10% of total cache
            return
        
        async with self.lock:
            # Ensure we have space
            await self._ensure_space(entry_size)
            
            # Create cache entry
            entry = CacheEntry(
                result=result.data,
                metadata=result.metadata,
                created_at=datetime.utcnow(),
                ttl=ttl or timedelta(minutes=15),
                size_bytes=entry_size
            )
            
            self.cache[cache_key] = entry
            self.current_size_bytes += entry_size
            self.access_order.append(cache_key)
        
        # Also cache in Redis if available
        if self.redis_client:
            try:
                entry_data = {
                    'result': result.data,
                    'metadata': result.metadata,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                compressed_data = zlib.compress(pickle.dumps(entry_data))
                ttl_seconds = int((ttl or timedelta(minutes=15)).total_seconds())
                
                await self.redis_client.setex(
                    f"query_cache:{cache_key}",
                    ttl_seconds,
                    compressed_data
                )
            except Exception as e:
                logger.warning(f"Redis cache storage failed: {e}")
    
    async def _ensure_space(self, required_bytes: int):
        """Ensure cache has enough space using LRU eviction"""
        while (self.current_size_bytes + required_bytes > self.max_size_bytes and 
               self.access_order):
            
            # Find least recently used entry
            oldest_key = None
            while self.access_order and not oldest_key:
                candidate = self.access_order.popleft()
                if candidate in self.cache:
                    oldest_key = candidate
            
            if oldest_key:
                await self._evict_entry(oldest_key)
    
    async def _evict_entry(self, cache_key: str):
        """Evict entry from cache"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[cache_key]
            self.evictions += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'entry_count': len(self.cache)
        }

class CrossDatabaseJoinExecutor:
    """Execute joins across different database types"""
    
    def __init__(self, db_manager: UnifiedDatabaseManager):
        self.db_manager = db_manager
        self.temp_table_counter = 0
        self.join_strategies = {
            'nested_loop': self._nested_loop_join,
            'hash_join': self._hash_join,
            'merge_join': self._merge_join,
            'broadcast_join': self._broadcast_join
        }
    
    async def execute_cross_db_join(self, left_query: str, left_db: DatabaseType,
                                  right_query: str, right_db: DatabaseType,
                                  join_condition: str, join_type: str = 'INNER') -> QueryResult:
        """Execute join between queries from different databases"""
        
        # Execute both queries in parallel
        left_task = asyncio.create_task(
            self.db_manager.execute_query(left_db, left_query)
        )
        right_task = asyncio.create_task(
            self.db_manager.execute_query(right_db, right_query)
        )
        
        left_result, right_result = await asyncio.gather(left_task, right_task)
        
        # Choose optimal join strategy based on data size
        strategy = self._choose_join_strategy(left_result, right_result)
        
        # Execute join
        join_result = await self.join_strategies[strategy](
            left_result.data, right_result.data, join_condition, join_type
        )
        
        return QueryResult(
            data=join_result,
            metadata={
                'left_db': left_db.value,
                'right_db': right_db.value,
                'join_strategy': strategy,
                'join_type': join_type
            },
            execution_time=left_result.execution_time + right_result.execution_time,
            row_count=len(join_result) if isinstance(join_result, list) else 1,
            database_type=DatabaseType.POSTGRESQL  # Coordinator database
        )
    
    def _choose_join_strategy(self, left_result: QueryResult, 
                            right_result: QueryResult) -> str:
        """Choose optimal join strategy based on data characteristics"""
        left_size = left_result.row_count
        right_size = right_result.row_count
        
        # Small datasets: nested loop
        if left_size < 1000 and right_size < 1000:
            return 'nested_loop'
        
        # One small, one large: broadcast join
        if min(left_size, right_size) < 10000 and max(left_size, right_size) > 100000:
            return 'broadcast_join'
        
        # Large datasets: hash join
        if left_size > 10000 or right_size > 10000:
            return 'hash_join'
        
        # Default: merge join
        return 'merge_join'
    
    async def _nested_loop_join(self, left_data: List[Dict], right_data: List[Dict],
                              join_condition: str, join_type: str) -> List[Dict]:
        """Nested loop join implementation"""
        result = []
        
        for left_row in left_data:
            for right_row in right_data:
                if self._evaluate_join_condition(left_row, right_row, join_condition):
                    joined_row = {**left_row, **right_row}
                    result.append(joined_row)
        
        return result
    
    async def _hash_join(self, left_data: List[Dict], right_data: List[Dict],
                        join_condition: str, join_type: str) -> List[Dict]:
        """Hash join implementation"""
        # Build hash table from smaller dataset
        if len(left_data) <= len(right_data):
            build_data, probe_data = left_data, right_data
        else:
            build_data, probe_data = right_data, left_data
        
        # Extract join key from condition (simplified)
        join_key = self._extract_join_key(join_condition)
        
        # Build hash table
        hash_table = defaultdict(list)
        for row in build_data:
            key_value = row.get(join_key)
            if key_value is not None:
                hash_table[key_value].append(row)
        
        # Probe and join
        result = []
        for probe_row in probe_data:
            key_value = probe_row.get(join_key)
            if key_value in hash_table:
                for build_row in hash_table[key_value]:
                    joined_row = {**probe_row, **build_row}
                    result.append(joined_row)
        
        return result
    
    async def _merge_join(self, left_data: List[Dict], right_data: List[Dict],
                         join_condition: str, join_type: str) -> List[Dict]:
        """Merge join implementation (requires sorted data)"""
        join_key = self._extract_join_key(join_condition)
        
        # Sort both datasets by join key
        left_sorted = sorted(left_data, key=lambda x: x.get(join_key, ''))
        right_sorted = sorted(right_data, key=lambda x: x.get(join_key, ''))
        
        result = []
        left_idx = right_idx = 0
        
        while left_idx < len(left_sorted) and right_idx < len(right_sorted):
            left_key = left_sorted[left_idx].get(join_key)
            right_key = right_sorted[right_idx].get(join_key)
            
            if left_key == right_key:
                # Found match
                joined_row = {**left_sorted[left_idx], **right_sorted[right_idx]}
                result.append(joined_row)
                left_idx += 1
                right_idx += 1
            elif left_key < right_key:
                left_idx += 1
            else:
                right_idx += 1
        
        return result
    
    async def _broadcast_join(self, left_data: List[Dict], right_data: List[Dict],
                            join_condition: str, join_type: str) -> List[Dict]:
        """Broadcast join (broadcast smaller dataset)"""
        # Use hash join strategy but optimize for broadcasting
        return await self._hash_join(left_data, right_data, join_condition, join_type)
    
    def _extract_join_key(self, join_condition: str) -> str:
        """Extract join key from condition (simplified parser)"""
        # Simple pattern matching for "table1.key = table2.key"
        pattern = r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)'
        match = re.search(pattern, join_condition)
        
        if match:
            return match.group(2)  # Return first key
        
        # Fallback: look for common key names
        common_keys = ['id', 'key', 'timestamp', 'location_id', 'volcano_id']
        for key in common_keys:
            if key in join_condition:
                return key
        
        return 'id'  # Default fallback
    
    def _evaluate_join_condition(self, left_row: Dict, right_row: Dict, 
                               condition: str) -> bool:
        """Evaluate join condition (simplified)"""
        # For now, implement simple equality join
        join_key = self._extract_join_key(condition)
        return left_row.get(join_key) == right_row.get(join_key)

class QueryPerformanceMonitor:
    """Monitor and analyze query performance across databases"""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[QueryMetrics]] = defaultdict(list)
        self.slow_query_threshold = 5.0  # seconds
        self.alert_callbacks: List[callable] = []
        self.monitoring_enabled = True
        
        # Performance analytics
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.database_load: Dict[DatabaseType, float] = defaultdict(float)
        
    def record_query_metrics(self, query: str, metrics: QueryMetrics):
        """Record query performance metrics"""
        if not self.monitoring_enabled:
            return
        
        query_hash = hashlib.md5(query.encode()).hexdigest()
        self.metrics_history[query_hash].append(metrics)
        
        # Update query patterns
        self.query_patterns[self._classify_query(query)] += 1
        
        # Update database load
        self.database_load[metrics.database_type] = (
            self.database_load[metrics.database_type] * 0.9 + 
            metrics.execution_time * 0.1
        )
        
        # Check for slow queries
        if metrics.execution_time > self.slow_query_threshold:
            asyncio.create_task(self._handle_slow_query(query, metrics))
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for pattern analysis"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            if 'join' in query_lower:
                return 'select_join'
            elif 'group by' in query_lower:
                return 'select_aggregate'
            elif 'st_' in query_lower:
                return 'select_spatial'
            else:
                return 'select_simple'
        elif query_lower.startswith('insert'):
            return 'insert'
        elif query_lower.startswith('update'):
            return 'update'
        elif query_lower.startswith('delete'):
            return 'delete'
        elif query_lower.startswith('match'):
            return 'cypher_match'
        elif 'from(' in query_lower:
            return 'flux_query'
        else:
            return 'other'
    
    async def _handle_slow_query(self, query: str, metrics: QueryMetrics):
        """Handle slow query detection"""
        logger.warning(f"Slow query detected: {metrics.execution_time:.2f}s on {metrics.database_type.value}")
        
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(query, metrics)
                else:
                    callback(query, metrics)
            except Exception as e:
                logger.error(f"Error in slow query callback: {e}")
    
    def get_performance_report(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Generate performance analysis report"""
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.utcnow() - time_window
        
        # Collect metrics within time window
        recent_metrics = []
        for query_metrics_list in self.metrics_history.values():
            for metrics in query_metrics_list:
                if metrics.timestamp >= cutoff_time:
                    recent_metrics.append(metrics)
        
        if not recent_metrics:
            return {'message': 'No metrics available for the specified time window'}
        
        # Calculate statistics
        execution_times = [m.execution_time for m in recent_metrics]
        memory_usage = [m.memory_usage for m in recent_metrics if m.memory_usage > 0]
        
        # Database-specific statistics
        db_stats = {}
        for db_type in DatabaseType:
            db_metrics = [m for m in recent_metrics if m.database_type == db_type]
            if db_metrics:
                db_stats[db_type.value] = {
                    'query_count': len(db_metrics),
                    'avg_execution_time': statistics.mean([m.execution_time for m in db_metrics]),
                    'max_execution_time': max([m.execution_time for m in db_metrics]),
                    'cache_hit_rate': sum([1 for m in db_metrics if m.cache_hit]) / len(db_metrics)
                }
        
        return {
            'time_window': str(time_window),
            'total_queries': len(recent_metrics),
            'avg_execution_time': statistics.mean(execution_times),
            'median_execution_time': statistics.median(execution_times),
            'max_execution_time': max(execution_times),
            'min_execution_time': min(execution_times),
            'slow_queries': len([t for t in execution_times if t > self.slow_query_threshold]),
            'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0,
            'database_statistics': db_stats,
            'query_patterns': dict(self.query_patterns),
            'database_load': {k.value: v for k, v in self.database_load.items()}
        }
    
    def register_alert_callback(self, callback: callable):
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)

class CrossDatabaseQueryOptimizer:
    """
    Comprehensive Cross-Database Query Optimization System
    Handles intelligent query routing, caching, cross-database joins,
    performance monitoring, and automated optimization
    """
    
    def __init__(self, db_manager: UnifiedDatabaseManager):
        self.db_manager = db_manager
        self.query_router = QueryRouter()
        self.query_cache = QueryCache()
        self.join_executor = CrossDatabaseJoinExecutor(db_manager)
        self.performance_monitor = QueryPerformanceMonitor()
        
        # Query optimization rules
        self.optimization_rules: Dict[DatabaseType, List[callable]] = {
            DatabaseType.POSTGRESQL: [
                self._optimize_postgresql_spatial,
                self._optimize_postgresql_temporal,
                self._optimize_postgresql_joins
            ],
            DatabaseType.INFLUXDB: [
                self._optimize_influxdb_time_range,
                self._optimize_influxdb_aggregation,
                self._optimize_influxdb_retention
            ],
            DatabaseType.NEO4J: [
                self._optimize_neo4j_traversal,
                self._optimize_neo4j_indexes,
                self._optimize_neo4j_patterns
            ],
            DatabaseType.REDIS: [
                self._optimize_redis_pipeline,
                self._optimize_redis_lua_scripts
            ]
        }
        
        # Materialized view management
        self.materialized_views: Dict[str, Dict[str, Any]] = {}
        
        # Distributed query execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info("Cross-Database Query Optimizer initialized")
    
    async def execute_optimized_query(self, query: str, params: Optional[Dict] = None,
                                    query_type: Optional[QueryType] = None,
                                    preferred_db: Optional[DatabaseType] = None,
                                    use_cache: bool = True) -> QueryResult:
        """Execute query with full optimization pipeline"""
        
        start_time = time.time()
        
        # Detect query type if not provided
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        # Check cache first
        if use_cache:
            cached_result = await self.query_cache.get(query, params, preferred_db)
            if cached_result:
                return cached_result
        
        # Route query to optimal database
        target_db = self.query_router.route_query(query, query_type, preferred_db)
        
        # Apply database-specific optimizations
        optimized_query = await self._apply_optimizations(query, target_db, params)
        
        # Execute query
        try:
            result = await self.db_manager.execute_query(target_db, optimized_query, params, use_cache=False)
            
            # Record performance metrics
            execution_time = time.time() - start_time
            metrics = QueryMetrics(
                execution_time=execution_time,
                rows_processed=result.row_count,
                memory_usage=self._get_memory_usage(),
                cpu_usage=self._get_cpu_usage(),
                cache_hit=False,
                optimization_applied=self._get_applied_optimizations(optimized_query, query),
                database_type=target_db,
                query_hash=hashlib.md5(query.encode()).hexdigest()
            )
            
            self.performance_monitor.record_query_metrics(query, metrics)
            self.query_router.update_performance_metrics(target_db, metrics)
            
            # Cache successful results
            if use_cache and result.data is not None:
                await self.query_cache.put(query, result, params)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def execute_cross_database_query(self, query_plan: QueryPlan) -> QueryResult:
        """Execute complex query spanning multiple databases"""
        
        if len(query_plan.execution_order) == 1:
            # Single database query
            db_type, query = query_plan.execution_order[0]
            return await self.execute_optimized_query(query, preferred_db=db_type)
        
        # Multi-database query execution
        results = []
        
        if query_plan.parallelizable:
            # Execute queries in parallel
            tasks = []
            for db_type, query in query_plan.execution_order:
                task = asyncio.create_task(
                    self.execute_optimized_query(query, preferred_db=db_type)
                )
                tasks.append((db_type, task))
            
            for db_type, task in tasks:
                result = await task
                results.append((db_type, result))
        else:
            # Execute queries sequentially
            for db_type, query in query_plan.execution_order:
                result = await self.execute_optimized_query(query, preferred_db=db_type)
                results.append((db_type, result))
        
        # Combine results if needed
        if query_plan.requires_aggregation:
            return await self._aggregate_results(results, query_plan)
        else:
            # Return the last result (assuming it's the final output)
            return results[-1][1] if results else QueryResult(
                data=None, metadata={}, execution_time=0, row_count=0
            )
    
    async def create_materialized_view(self, view_name: str, query: str, 
                                     refresh_interval: timedelta = timedelta(hours=1)) -> bool:
        """Create materialized view for frequently accessed queries"""
        try:
            # Execute initial query to populate view
            result = await self.execute_optimized_query(query)
            
            # Store materialized view metadata
            self.materialized_views[view_name] = {
                'query': query,
                'last_refresh': datetime.utcnow(),
                'refresh_interval': refresh_interval,
                'row_count': result.row_count,
                'data': result.data
            }
            
            # Schedule periodic refresh
            asyncio.create_task(self._schedule_view_refresh(view_name))
            
            logger.info(f"Created materialized view: {view_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create materialized view {view_name}: {e}")
            return False
    
    async def _schedule_view_refresh(self, view_name: str):
        """Schedule periodic refresh of materialized view"""
        while view_name in self.materialized_views:
            view_info = self.materialized_views[view_name]
            
            # Wait for refresh interval
            await asyncio.sleep(view_info['refresh_interval'].total_seconds())
            
            try:
                # Refresh view data
                result = await self.execute_optimized_query(view_info['query'])
                view_info['data'] = result.data
                view_info['last_refresh'] = datetime.utcnow()
                view_info['row_count'] = result.row_count
                
                logger.info(f"Refreshed materialized view: {view_name}")
                
            except Exception as e:
                logger.error(f"Failed to refresh materialized view {view_name}: {e}")
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from SQL/query content"""
        query_lower = query.lower().strip()
        
        # Spatial queries
        if any(spatial_func in query_lower for spatial_func in ['st_', 'geometry', 'spatial']):
            return QueryType.SPATIAL
        
        # Temporal queries
        if any(temporal_keyword in query_lower for temporal_keyword in ['timestamp', 'time_series', 'range(']):
            return QueryType.TEMPORAL
        
        # Graph queries (Cypher)
        if query_lower.startswith('match') or 'relationship' in query_lower:
            return QueryType.GRAPH
        
        # Time series (InfluxDB)
        if 'from(' in query_lower or 'measurement' in query_lower:
            return QueryType.TIME_SERIES
        
        # Aggregation queries
        if any(agg_func in query_lower for agg_func in ['count(', 'sum(', 'avg(', 'max(', 'min(', 'group by']):
            return QueryType.AGGREGATE
        
        # Cross-database indicators
        if len([db.value for db in DatabaseType if db.value in query_lower]) > 1:
            return QueryType.CROSS_DATABASE
        
        # Default to SELECT
        return QueryType.SELECT
    
    async def _apply_optimizations(self, query: str, db_type: DatabaseType, 
                                 params: Optional[Dict] = None) -> str:
        """Apply database-specific optimizations"""
        optimized_query = query
        
        # Apply optimization rules for the database type
        for optimization_rule in self.optimization_rules.get(db_type, []):
            try:
                optimized_query = optimization_rule(optimized_query, params)
            except Exception as e:
                logger.warning(f"Optimization rule failed: {e}")
        
        return optimized_query
    
    # PostgreSQL optimization rules
    def _optimize_postgresql_spatial(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize PostgreSQL spatial queries"""
        optimized = query
        
        # Add spatial index hints
        if 'ST_' in query and 'location' in query:
            # Add index hint for spatial operations
            optimized = f"/*+ USE_INDEX(geological_observations idx_geo_obs_location) */ {optimized}"
        
        # Optimize spatial joins
        if 'ST_Intersects' in query or 'ST_Within' in query:
            # Add spatial join optimization
            optimized = optimized.replace(
                'ST_Intersects(',
                'ST_Intersects(/*+ SPATIAL_JOIN */ '
            )
        
        return optimized
    
    def _optimize_postgresql_temporal(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize PostgreSQL temporal queries"""
        optimized = query
        
        # Add temporal index hints
        if 'timestamp' in query.lower():
            optimized = f"/*+ USE_INDEX(geological_observations idx_geo_obs_timestamp) */ {optimized}"
        
        # Optimize time range queries
        if 'BETWEEN' in query and 'timestamp' in query.lower():
            # Add partition pruning hint
            optimized = f"/*+ PARTITION_PRUNING */ {optimized}"
        
        return optimized
    
    def _optimize_postgresql_joins(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize PostgreSQL join queries"""
        optimized = query
        
        # Add join hints for large tables
        if 'JOIN' in query.upper() and 'geological_observations' in query:
            # Suggest hash join for large tables
            optimized = f"/*+ HASH_JOIN */ {optimized}"
        
        return optimized
    
    # InfluxDB optimization rules
    def _optimize_influxdb_time_range(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize InfluxDB time range queries"""
        optimized = query
        
        # Ensure time range is specified
        if 'range(' not in query and 'from(' in query:
            # Add default time range
            optimized = optimized.replace(
                'from(',
                'from(\n  |> range(start: -1h)\n  |> from('
            )
        
        return optimized
    
    def _optimize_influxdb_aggregation(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize InfluxDB aggregation queries"""
        optimized = query
        
        # Add aggregation window optimization
        if 'aggregateWindow(' in query:
            # Ensure proper window alignment
            if 'offset:' not in query:
                optimized = optimized.replace(
                    'aggregateWindow(',
                    'aggregateWindow(offset: 0s, '
                )
        
        return optimized
    
    def _optimize_influxdb_retention(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize InfluxDB retention policy queries"""
        optimized = query
        
        # Add retention policy hints
        if 'from(' in query and 'bucket:' in query:
            # Add retention policy optimization
            optimized = optimized.replace(
                'bucket:',
                '/*+ RETENTION_POLICY */ bucket:'
            )
        
        return optimized
    
    # Neo4j optimization rules
    def _optimize_neo4j_traversal(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize Neo4j graph traversal queries"""
        optimized = query
        
        # Add traversal depth limits
        if 'MATCH' in query and '*' in query and '..' not in query:
            # Add depth limit to prevent runaway queries
            optimized = optimized.replace('*', '*1..5')
        
        return optimized
    
    def _optimize_neo4j_indexes(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize Neo4j index usage"""
        optimized = query
        
        # Add index hints for node lookups
        if 'MATCH (p:GeologicalProcess' in query:
            optimized = optimized.replace(
                'MATCH (p:GeologicalProcess',
                'MATCH (p:GeologicalProcess USING INDEX p:GeologicalProcess(id)'
            )
        
        return optimized
    
    def _optimize_neo4j_patterns(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize Neo4j pattern matching"""
        optimized = query
        
        # Optimize relationship patterns
        if '-->' in query:
            # Add relationship type hints
            optimized = optimized.replace('-->', '-[:CAUSES]->')
        
        return optimized
    
    # Redis optimization rules
    def _optimize_redis_pipeline(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize Redis pipeline operations"""
        # Redis queries are typically simple commands
        return query
    
    def _optimize_redis_lua_scripts(self, query: str, params: Optional[Dict] = None) -> str:
        """Optimize Redis Lua script execution"""
        # For complex Redis operations, suggest Lua scripts
        return query
    
    def _get_applied_optimizations(self, optimized_query: str, original_query: str) -> List[OptimizationStrategy]:
        """Determine which optimizations were applied"""
        applied = []
        
        if '/*+' in optimized_query:
            applied.append(OptimizationStrategy.INDEX_HINT)
        
        if 'PARTITION_PRUNING' in optimized_query:
            applied.append(OptimizationStrategy.PARTITION_PRUNING)
        
        if 'HASH_JOIN' in optimized_query:
            applied.append(OptimizationStrategy.PARALLEL_EXECUTION)
        
        if optimized_query != original_query:
            applied.append(OptimizationStrategy.QUERY_REWRITE)
        
        return applied
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    async def _aggregate_results(self, results: List[Tuple[DatabaseType, QueryResult]], 
                               query_plan: QueryPlan) -> QueryResult:
        """Aggregate results from multiple database queries"""
        
        if not results:
            return QueryResult(data=None, metadata={}, execution_time=0, row_count=0)
        
        # Simple aggregation: combine all data
        combined_data = []
        total_execution_time = 0
        total_rows = 0
        
        for db_type, result in results:
            if result.data:
                if isinstance(result.data, list):
                    combined_data.extend(result.data)
                else:
                    combined_data.append(result.data)
            
            total_execution_time += result.execution_time
            total_rows += result.row_count
        
        return QueryResult(
            data=combined_data,
            metadata={
                'aggregated_from': [db_type.value for db_type, _ in results],
                'aggregation_strategy': 'simple_combine'
            },
            execution_time=total_execution_time,
            row_count=total_rows,
            database_type=DatabaseType.POSTGRESQL  # Coordinator
        )
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization performance report"""
        
        # Get cache statistics
        cache_stats = self.query_cache.get_statistics()
        
        # Get performance statistics
        perf_report = self.performance_monitor.get_performance_report()
        
        # Get routing statistics
        routing_stats = {
            'routing_rules_count': len(self.query_router.routing_rules),
            'load_balancing_weights': {
                db.value: weight for db, weight in self.query_router.load_balancing_weights.items()
            }
        }
        
        # Get materialized view statistics
        mv_stats = {
            'materialized_views_count': len(self.materialized_views),
            'views': {
                name: {
                    'last_refresh': info['last_refresh'].isoformat(),
                    'row_count': info['row_count']
                }
                for name, info in self.materialized_views.items()
            }
        }
        
        return {
            'cache_statistics': cache_stats,
            'performance_report': perf_report,
            'routing_statistics': routing_stats,
            'materialized_view_statistics': mv_stats,
            'optimization_rules_count': sum(len(rules) for rules in self.optimization_rules.values()),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        # Clear materialized views
        self.materialized_views.clear()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Cross-Database Query Optimizer cleanup complete")

# Global optimizer instance
_query_optimizer: Optional[CrossDatabaseQueryOptimizer] = None

async def get_query_optimizer() -> CrossDatabaseQueryOptimizer:
    """Get global query optimizer instance"""
    global _query_optimizer
    
    if _query_optimizer is None:
        from backend.core.unified_database_manager import get_database_manager
        db_manager = await get_database_manager()
        _query_optimizer = CrossDatabaseQueryOptimizer(db_manager)
    
    return _query_optimizer

async def cleanup_query_optimizer():
    """Cleanup global query optimizer"""
    global _query_optimizer
    
    if _query_optimizer:
        await _query_optimizer.cleanup()
        _query_optimizer = None