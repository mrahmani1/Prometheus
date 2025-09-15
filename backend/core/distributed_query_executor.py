"""
Distributed Query Execution System
Advanced distributed processing for large-scale geological datasets
with intelligent partitioning, parallel execution, and result aggregation
"""

import asyncio
import logging
import json
import time
import hashlib
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle
import numpy as np
import pandas as pd

# Database and optimization
from backend.core.unified_database_manager import DatabaseType, UnifiedDatabaseManager, QueryResult
from backend.core.cross_database_query_optimizer import CrossDatabaseQueryOptimizer, QueryType

# Geospatial and temporal processing
try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, box
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False

logger = logging.getLogger(__name__)

class PartitionStrategy(Enum):
    """Data partitioning strategies"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    HASH = "hash"
    RANGE = "range"
    ROUND_ROBIN = "round_robin"
    ADAPTIVE = "adaptive"

class ExecutionMode(Enum):
    """Query execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    STREAMING = "streaming"

@dataclass
class PartitionInfo:
    """Information about a data partition"""
    partition_id: str
    partition_strategy: PartitionStrategy
    partition_key: Any
    estimated_size: int
    database_type: DatabaseType
    query_fragment: str
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionPlan:
    """Distributed execution plan"""
    plan_id: str
    original_query: str
    partitions: List[PartitionInfo]
    execution_mode: ExecutionMode
    aggregation_strategy: str
    estimated_total_cost: float
    parallelism_degree: int
    resource_requirements: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ExecutionResult:
    """Result from distributed execution"""
    partition_id: str
    data: Any
    execution_time: float
    rows_processed: int
    memory_used: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class SpatialPartitioner:
    """Spatial data partitioning for geographical datasets"""
    
    def __init__(self):
        self.grid_cache: Dict[str, List[Polygon]] = {}
    
    def create_spatial_partitions(self, bounds: Tuple[float, float, float, float], 
                                num_partitions: int) -> List[Polygon]:
        """Create spatial grid partitions"""
        
        if not GEOSPATIAL_AVAILABLE:
            raise ImportError("Geospatial libraries not available for spatial partitioning")
        
        min_x, min_y, max_x, max_y = bounds
        
        # Calculate grid dimensions
        aspect_ratio = (max_x - min_x) / (max_y - min_y)
        
        if aspect_ratio > 1:
            cols = int(math.ceil(math.sqrt(num_partitions * aspect_ratio)))
            rows = int(math.ceil(num_partitions / cols))
        else:
            rows = int(math.ceil(math.sqrt(num_partitions / aspect_ratio)))
            cols = int(math.ceil(num_partitions / rows))
        
        # Create grid cells
        x_step = (max_x - min_x) / cols
        y_step = (max_y - min_y) / rows
        
        partitions = []
        for i in range(rows):
            for j in range(cols):
                if len(partitions) >= num_partitions:
                    break
                
                x1 = min_x + j * x_step
                y1 = min_y + i * y_step
                x2 = min_x + (j + 1) * x_step
                y2 = min_y + (i + 1) * y_step
                
                partition = box(x1, y1, x2, y2)
                partitions.append(partition)
        
        return partitions
    
    def partition_by_administrative_boundaries(self, region: str) -> List[Polygon]:
        """Partition by administrative boundaries (countries, states, etc.)"""
        # This would integrate with administrative boundary datasets
        # For now, return a simple implementation
        
        # Example boundaries for major volcanic regions
        volcanic_regions = {
            'pacific_ring_of_fire': [
                box(-180, -60, -120, 60),  # Pacific West
                box(-120, -60, -60, 60),   # Americas
                box(120, -60, 180, 60),    # Pacific East
            ],
            'mediterranean': [
                box(-10, 30, 50, 50),      # Mediterranean Basin
            ],
            'mid_atlantic_ridge': [
                box(-40, -60, -10, 70),    # Mid-Atlantic
            ]
        }
        
        return volcanic_regions.get(region, [box(-180, -90, 180, 90)])

class TemporalPartitioner:
    """Temporal data partitioning for time-series datasets"""
    
    def create_temporal_partitions(self, start_time: datetime, end_time: datetime,
                                 partition_size: timedelta) -> List[Tuple[datetime, datetime]]:
        """Create temporal partitions"""
        
        partitions = []
        current_time = start_time
        
        while current_time < end_time:
            partition_end = min(current_time + partition_size, end_time)
            partitions.append((current_time, partition_end))
            current_time = partition_end
        
        return partitions
    
    def create_adaptive_temporal_partitions(self, start_time: datetime, end_time: datetime,
                                          data_density: Dict[datetime, int],
                                          target_partition_size: int) -> List[Tuple[datetime, datetime]]:
        """Create adaptive temporal partitions based on data density"""
        
        partitions = []
        current_time = start_time
        current_size = 0
        
        sorted_times = sorted(data_density.keys())
        
        for timestamp in sorted_times:
            if timestamp < start_time:
                continue
            if timestamp > end_time:
                break
            
            current_size += data_density[timestamp]
            
            if current_size >= target_partition_size:
                partitions.append((current_time, timestamp))
                current_time = timestamp
                current_size = 0
        
        # Add final partition if needed
        if current_time < end_time:
            partitions.append((current_time, end_time))
        
        return partitions

class QueryPartitioner:
    """Intelligent query partitioning system"""
    
    def __init__(self):
        self.spatial_partitioner = SpatialPartitioner()
        self.temporal_partitioner = TemporalPartitioner()
        
        # Partitioning heuristics
        self.size_thresholds = {
            'small': 10000,      # rows
            'medium': 100000,
            'large': 1000000,
            'xlarge': 10000000
        }
    
    async def create_execution_plan(self, query: str, query_type: QueryType,
                                  estimated_data_size: int,
                                  available_resources: Dict[str, Any]) -> ExecutionPlan:
        """Create optimized execution plan for query"""
        
        plan_id = f"plan_{int(time.time())}_{hashlib.md5(query.encode()).hexdigest()[:8]}"
        
        # Determine partitioning strategy
        partition_strategy = self._choose_partition_strategy(query, query_type, estimated_data_size)
        
        # Create partitions
        partitions = await self._create_partitions(query, partition_strategy, estimated_data_size)
        
        # Determine execution mode
        execution_mode = self._choose_execution_mode(partitions, available_resources)
        
        # Calculate parallelism degree
        parallelism_degree = self._calculate_parallelism(partitions, available_resources)
        
        # Estimate costs
        estimated_cost = self._estimate_execution_cost(partitions, execution_mode)
        
        return ExecutionPlan(
            plan_id=plan_id,
            original_query=query,
            partitions=partitions,
            execution_mode=execution_mode,
            aggregation_strategy=self._choose_aggregation_strategy(query_type),
            estimated_total_cost=estimated_cost,
            parallelism_degree=parallelism_degree,
            resource_requirements=self._calculate_resource_requirements(partitions)
        )
    
    def _choose_partition_strategy(self, query: str, query_type: QueryType, 
                                 data_size: int) -> PartitionStrategy:
        """Choose optimal partitioning strategy"""
        
        query_lower = query.lower()
        
        # Spatial queries benefit from spatial partitioning
        if query_type == QueryType.SPATIAL or any(spatial_keyword in query_lower 
                                                for spatial_keyword in ['st_', 'geometry', 'location']):
            return PartitionStrategy.SPATIAL
        
        # Temporal queries benefit from temporal partitioning
        if query_type == QueryType.TEMPORAL or any(temporal_keyword in query_lower 
                                                 for temporal_keyword in ['timestamp', 'time', 'date']):
            return PartitionStrategy.TEMPORAL
        
        # Large datasets benefit from hash partitioning
        if data_size > self.size_thresholds['large']:
            return PartitionStrategy.HASH
        
        # Medium datasets can use range partitioning
        if data_size > self.size_thresholds['medium']:
            return PartitionStrategy.RANGE
        
        # Small datasets don't need partitioning
        return PartitionStrategy.ROUND_ROBIN
    
    async def _create_partitions(self, query: str, strategy: PartitionStrategy,
                               data_size: int) -> List[PartitionInfo]:
        """Create query partitions based on strategy"""
        
        partitions = []
        
        if strategy == PartitionStrategy.SPATIAL:
            partitions = await self._create_spatial_partitions(query, data_size)
        elif strategy == PartitionStrategy.TEMPORAL:
            partitions = await self._create_temporal_partitions(query, data_size)
        elif strategy == PartitionStrategy.HASH:
            partitions = await self._create_hash_partitions(query, data_size)
        elif strategy == PartitionStrategy.RANGE:
            partitions = await self._create_range_partitions(query, data_size)
        else:
            # Single partition for small queries
            partitions = [PartitionInfo(
                partition_id="single_partition",
                partition_strategy=strategy,
                partition_key="all",
                estimated_size=data_size,
                database_type=DatabaseType.POSTGRESQL,
                query_fragment=query
            )]
        
        return partitions
    
    async def _create_spatial_partitions(self, query: str, data_size: int) -> List[PartitionInfo]:
        """Create spatial partitions for geographical queries"""
        
        # Determine number of partitions based on data size
        num_partitions = min(max(data_size // self.size_thresholds['medium'], 1), 16)
        
        # Extract spatial bounds from query (simplified)
        bounds = self._extract_spatial_bounds(query)
        
        # Create spatial grid
        spatial_partitions = self.spatial_partitioner.create_spatial_partitions(bounds, num_partitions)
        
        partitions = []
        for i, spatial_partition in enumerate(spatial_partitions):
            # Create spatial filter for this partition
            minx, miny, maxx, maxy = spatial_partition.bounds
            
            spatial_filter = f"""
            ST_Intersects(location, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326))
            """
            
            # Modify query to include spatial filter
            partition_query = self._add_spatial_filter(query, spatial_filter)
            
            partition = PartitionInfo(
                partition_id=f"spatial_partition_{i}",
                partition_strategy=PartitionStrategy.SPATIAL,
                partition_key=f"bounds_{minx}_{miny}_{maxx}_{maxy}",
                estimated_size=data_size // num_partitions,
                database_type=DatabaseType.POSTGRESQL,
                query_fragment=partition_query,
                metadata={'bounds': (minx, miny, maxx, maxy)}
            )
            
            partitions.append(partition)
        
        return partitions
    
    async def _create_temporal_partitions(self, query: str, data_size: int) -> List[PartitionInfo]:
        """Create temporal partitions for time-series queries"""
        
        # Extract temporal range from query
        start_time, end_time = self._extract_temporal_range(query)
        
        # Determine partition size based on data volume
        if data_size > self.size_thresholds['xlarge']:
            partition_duration = timedelta(hours=1)
        elif data_size > self.size_thresholds['large']:
            partition_duration = timedelta(hours=6)
        elif data_size > self.size_thresholds['medium']:
            partition_duration = timedelta(days=1)
        else:
            partition_duration = timedelta(days=7)
        
        # Create temporal partitions
        temporal_partitions = self.temporal_partitioner.create_temporal_partitions(
            start_time, end_time, partition_duration
        )
        
        partitions = []
        for i, (part_start, part_end) in enumerate(temporal_partitions):
            # Create temporal filter
            temporal_filter = f"""
            timestamp >= '{part_start.isoformat()}' AND timestamp < '{part_end.isoformat()}'
            """
            
            # Modify query to include temporal filter
            partition_query = self._add_temporal_filter(query, temporal_filter)
            
            partition = PartitionInfo(
                partition_id=f"temporal_partition_{i}",
                partition_strategy=PartitionStrategy.TEMPORAL,
                partition_key=f"time_{part_start.isoformat()}_{part_end.isoformat()}",
                estimated_size=data_size // len(temporal_partitions),
                database_type=DatabaseType.INFLUXDB if 'time_series' in query.lower() else DatabaseType.POSTGRESQL,
                query_fragment=partition_query,
                metadata={'start_time': part_start, 'end_time': part_end}
            )
            
            partitions.append(partition)
        
        return partitions
    
    async def _create_hash_partitions(self, query: str, data_size: int) -> List[PartitionInfo]:
        """Create hash-based partitions"""
        
        num_partitions = min(max(data_size // self.size_thresholds['medium'], 2), 32)
        
        # Find suitable hash key (simplified)
        hash_key = self._find_hash_key(query)
        
        partitions = []
        for i in range(num_partitions):
            # Create hash filter
            hash_filter = f"ABS(HASHTEXT({hash_key})) % {num_partitions} = {i}"
            
            # Modify query to include hash filter
            partition_query = self._add_hash_filter(query, hash_filter)
            
            partition = PartitionInfo(
                partition_id=f"hash_partition_{i}",
                partition_strategy=PartitionStrategy.HASH,
                partition_key=f"hash_{i}_{num_partitions}",
                estimated_size=data_size // num_partitions,
                database_type=DatabaseType.POSTGRESQL,
                query_fragment=partition_query,
                metadata={'hash_key': hash_key, 'partition_number': i, 'total_partitions': num_partitions}
            )
            
            partitions.append(partition)
        
        return partitions
    
    async def _create_range_partitions(self, query: str, data_size: int) -> List[PartitionInfo]:
        """Create range-based partitions"""
        
        num_partitions = min(max(data_size // self.size_thresholds['medium'], 2), 16)
        
        # Find suitable range key
        range_key = self._find_range_key(query)
        
        # This would typically query the database to find actual value ranges
        # For now, use a simplified approach
        partitions = []
        for i in range(num_partitions):
            range_filter = f"{range_key} >= {i * 1000} AND {range_key} < {(i + 1) * 1000}"
            
            partition_query = self._add_range_filter(query, range_filter)
            
            partition = PartitionInfo(
                partition_id=f"range_partition_{i}",
                partition_strategy=PartitionStrategy.RANGE,
                partition_key=f"range_{i}",
                estimated_size=data_size // num_partitions,
                database_type=DatabaseType.POSTGRESQL,
                query_fragment=partition_query,
                metadata={'range_key': range_key, 'range_start': i * 1000, 'range_end': (i + 1) * 1000}
            )
            
            partitions.append(partition)
        
        return partitions
    
    def _extract_spatial_bounds(self, query: str) -> Tuple[float, float, float, float]:
        """Extract spatial bounds from query (simplified)"""
        # Default to global bounds - in practice, this would parse the query
        return (-180.0, -90.0, 180.0, 90.0)
    
    def _extract_temporal_range(self, query: str) -> Tuple[datetime, datetime]:
        """Extract temporal range from query (simplified)"""
        # Default to last 24 hours - in practice, this would parse the query
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=1)
        return (start_time, end_time)
    
    def _find_hash_key(self, query: str) -> str:
        """Find suitable column for hash partitioning"""
        # Look for common ID columns
        common_keys = ['id', 'volcano_id', 'station_id', 'observation_id']
        
        query_lower = query.lower()
        for key in common_keys:
            if key in query_lower:
                return key
        
        return 'id'  # Default
    
    def _find_range_key(self, query: str) -> str:
        """Find suitable column for range partitioning"""
        # Look for numeric columns
        common_keys = ['id', 'magnitude', 'depth', 'elevation']
        
        query_lower = query.lower()
        for key in common_keys:
            if key in query_lower:
                return key
        
        return 'id'  # Default
    
    def _add_spatial_filter(self, query: str, spatial_filter: str) -> str:
        """Add spatial filter to query"""
        if 'WHERE' in query.upper():
            return query.replace('WHERE', f'WHERE {spatial_filter} AND')
        else:
            return f"{query} WHERE {spatial_filter}"
    
    def _add_temporal_filter(self, query: str, temporal_filter: str) -> str:
        """Add temporal filter to query"""
        if 'WHERE' in query.upper():
            return query.replace('WHERE', f'WHERE {temporal_filter} AND')
        else:
            return f"{query} WHERE {temporal_filter}"
    
    def _add_hash_filter(self, query: str, hash_filter: str) -> str:
        """Add hash filter to query"""
        if 'WHERE' in query.upper():
            return query.replace('WHERE', f'WHERE {hash_filter} AND')
        else:
            return f"{query} WHERE {hash_filter}"
    
    def _add_range_filter(self, query: str, range_filter: str) -> str:
        """Add range filter to query"""
        if 'WHERE' in query.upper():
            return query.replace('WHERE', f'WHERE {range_filter} AND')
        else:
            return f"{query} WHERE {range_filter}"
    
    def _choose_execution_mode(self, partitions: List[PartitionInfo], 
                             resources: Dict[str, Any]) -> ExecutionMode:
        """Choose optimal execution mode"""
        
        num_partitions = len(partitions)
        available_cores = resources.get('cpu_cores', mp.cpu_count())
        
        if num_partitions == 1:
            return ExecutionMode.SEQUENTIAL
        elif num_partitions <= available_cores:
            return ExecutionMode.PARALLEL
        else:
            return ExecutionMode.DISTRIBUTED
    
    def _calculate_parallelism(self, partitions: List[PartitionInfo], 
                             resources: Dict[str, Any]) -> int:
        """Calculate optimal parallelism degree"""
        
        available_cores = resources.get('cpu_cores', mp.cpu_count())
        available_memory_gb = resources.get('memory_gb', 8)
        
        # Consider memory constraints
        memory_per_partition = 0.5  # GB per partition (estimate)
        max_parallel_by_memory = int(available_memory_gb / memory_per_partition)
        
        # Consider CPU constraints
        max_parallel_by_cpu = available_cores
        
        # Consider number of partitions
        num_partitions = len(partitions)
        
        return min(num_partitions, max_parallel_by_cpu, max_parallel_by_memory)
    
    def _estimate_execution_cost(self, partitions: List[PartitionInfo], 
                               execution_mode: ExecutionMode) -> float:
        """Estimate execution cost"""
        
        total_size = sum(p.estimated_size for p in partitions)
        
        # Base cost per row (simplified)
        base_cost_per_row = 0.001  # seconds
        
        if execution_mode == ExecutionMode.SEQUENTIAL:
            return total_size * base_cost_per_row
        elif execution_mode == ExecutionMode.PARALLEL:
            # Parallel execution with some overhead
            max_partition_size = max(p.estimated_size for p in partitions) if partitions else 0
            return max_partition_size * base_cost_per_row * 1.2  # 20% overhead
        else:
            # Distributed execution with more overhead
            max_partition_size = max(p.estimated_size for p in partitions) if partitions else 0
            return max_partition_size * base_cost_per_row * 1.5  # 50% overhead
    
    def _choose_aggregation_strategy(self, query_type: QueryType) -> str:
        """Choose aggregation strategy based on query type"""
        
        if query_type == QueryType.AGGREGATE:
            return "sum_aggregation"
        elif query_type == QueryType.SPATIAL:
            return "spatial_union"
        elif query_type == QueryType.TEMPORAL:
            return "temporal_merge"
        else:
            return "simple_concatenation"
    
    def _calculate_resource_requirements(self, partitions: List[PartitionInfo]) -> Dict[str, Any]:
        """Calculate resource requirements for execution"""
        
        total_size = sum(p.estimated_size for p in partitions)
        
        # Estimate memory requirements (simplified)
        memory_per_row = 0.001  # MB per row
        estimated_memory_mb = total_size * memory_per_row
        
        return {
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_cpu_cores': len(partitions),
            'estimated_disk_io_mb': total_size * 0.01,  # Rough estimate
            'network_bandwidth_mbps': 100  # For distributed execution
        }

class DistributedQueryExecutor:
    """
    Distributed Query Execution System
    Handles parallel and distributed execution of large-scale geological queries
    """
    
    def __init__(self, db_manager: UnifiedDatabaseManager, 
                 query_optimizer: CrossDatabaseQueryOptimizer):
        
        self.db_manager = db_manager
        self.query_optimizer = query_optimizer
        self.query_partitioner = QueryPartitioner()
        
        # Execution resources
        self.thread_executor = ThreadPoolExecutor(max_workers=mp.cpu_count())
        self.process_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Result aggregation
        self.aggregation_strategies = {
            'simple_concatenation': self._simple_concatenation,
            'sum_aggregation': self._sum_aggregation,
            'spatial_union': self._spatial_union,
            'temporal_merge': self._temporal_merge
        }
        
        # Execution monitoring
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info("Distributed Query Executor initialized")
    
    async def execute_distributed_query(self, query: str, query_type: QueryType = None,
                                      estimated_data_size: int = None,
                                      execution_options: Dict[str, Any] = None) -> QueryResult:
        """Execute query using distributed processing"""
        
        start_time = time.time()
        
        # Detect query type if not provided
        if query_type is None:
            query_type = self._detect_query_type(query)
        
        # Estimate data size if not provided
        if estimated_data_size is None:
            estimated_data_size = await self._estimate_data_size(query, query_type)
        
        # Get available resources
        available_resources = self._get_available_resources()
        
        # Create execution plan
        execution_plan = await self.query_partitioner.create_execution_plan(
            query, query_type, estimated_data_size, available_resources
        )
        
        # Register active execution
        self.active_executions[execution_plan.plan_id] = execution_plan
        
        try:
            # Execute based on execution mode
            if execution_plan.execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(execution_plan)
            elif execution_plan.execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(execution_plan)
            else:
                result = await self._execute_distributed(execution_plan)
            
            # Record execution history
            execution_time = time.time() - start_time
            self._record_execution_history(execution_plan, execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_execution_history(execution_plan, execution_time, False, str(e))
            raise
            
        finally:
            # Cleanup
            if execution_plan.plan_id in self.active_executions:
                del self.active_executions[execution_plan.plan_id]
    
    async def _execute_sequential(self, plan: ExecutionPlan) -> QueryResult:
        """Execute partitions sequentially"""
        
        results = []
        total_execution_time = 0
        total_rows = 0
        
        for partition in plan.partitions:
            try:
                result = await self.query_optimizer.execute_optimized_query(
                    partition.query_fragment,
                    preferred_db=partition.database_type
                )
                
                execution_result = ExecutionResult(
                    partition_id=partition.partition_id,
                    data=result.data,
                    execution_time=result.execution_time,
                    rows_processed=result.row_count,
                    memory_used=0.0  # Would be measured in practice
                )
                
                results.append(execution_result)
                total_execution_time += result.execution_time
                total_rows += result.row_count
                
            except Exception as e:
                logger.error(f"Partition {partition.partition_id} failed: {e}")
                
                error_result = ExecutionResult(
                    partition_id=partition.partition_id,
                    data=None,
                    execution_time=0,
                    rows_processed=0,
                    memory_used=0,
                    error=str(e)
                )
                results.append(error_result)
        
        # Aggregate results
        aggregated_data = await self._aggregate_results(results, plan.aggregation_strategy)
        
        return QueryResult(
            data=aggregated_data,
            metadata={
                'execution_plan_id': plan.plan_id,
                'execution_mode': plan.execution_mode.value,
                'partitions_executed': len(results),
                'partitions_successful': len([r for r in results if r.error is None])
            },
            execution_time=total_execution_time,
            row_count=total_rows,
            database_type=DatabaseType.POSTGRESQL  # Coordinator
        )
    
    async def _execute_parallel(self, plan: ExecutionPlan) -> QueryResult:
        """Execute partitions in parallel"""
        
        # Create tasks for parallel execution
        tasks = []
        for partition in plan.partitions:
            task = asyncio.create_task(
                self._execute_partition(partition)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        execution_results = []
        total_rows = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = ExecutionResult(
                    partition_id=plan.partitions[i].partition_id,
                    data=None,
                    execution_time=0,
                    rows_processed=0,
                    memory_used=0,
                    error=str(result)
                )
                execution_results.append(error_result)
            else:
                execution_results.append(result)
                total_rows += result.rows_processed
        
        # Aggregate results
        aggregated_data = await self._aggregate_results(execution_results, plan.aggregation_strategy)
        
        # Calculate total execution time (max of parallel executions)
        successful_results = [r for r in execution_results if r.error is None]
        max_execution_time = max([r.execution_time for r in successful_results]) if successful_results else 0
        
        return QueryResult(
            data=aggregated_data,
            metadata={
                'execution_plan_id': plan.plan_id,
                'execution_mode': plan.execution_mode.value,
                'partitions_executed': len(execution_results),
                'partitions_successful': len(successful_results),
                'parallelism_degree': plan.parallelism_degree
            },
            execution_time=max_execution_time,
            row_count=total_rows,
            database_type=DatabaseType.POSTGRESQL
        )
    
    async def _execute_distributed(self, plan: ExecutionPlan) -> QueryResult:
        """Execute partitions in distributed manner"""
        
        # For now, use process-based parallelism as a proxy for distributed execution
        # In a real implementation, this would distribute across multiple nodes
        
        loop = asyncio.get_event_loop()
        
        # Submit tasks to process executor
        futures = []
        for partition in plan.partitions:
            future = loop.run_in_executor(
                self.process_executor,
                self._execute_partition_in_process,
                partition
            )
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in as_completed(futures):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                logger.error(f"Distributed partition execution failed: {e}")
                error_result = ExecutionResult(
                    partition_id="unknown",
                    data=None,
                    execution_time=0,
                    rows_processed=0,
                    memory_used=0,
                    error=str(e)
                )
                results.append(error_result)
        
        # Aggregate results
        aggregated_data = await self._aggregate_results(results, plan.aggregation_strategy)
        
        # Calculate metrics
        successful_results = [r for r in results if r.error is None]
        total_rows = sum(r.rows_processed for r in successful_results)
        max_execution_time = max([r.execution_time for r in successful_results]) if successful_results else 0
        
        return QueryResult(
            data=aggregated_data,
            metadata={
                'execution_plan_id': plan.plan_id,
                'execution_mode': plan.execution_mode.value,
                'partitions_executed': len(results),
                'partitions_successful': len(successful_results),
                'distributed_nodes': plan.parallelism_degree
            },
            execution_time=max_execution_time,
            row_count=total_rows,
            database_type=DatabaseType.POSTGRESQL
        )
    
    async def _execute_partition(self, partition: PartitionInfo) -> ExecutionResult:
        """Execute a single partition"""
        
        start_time = time.time()
        
        try:
            result = await self.query_optimizer.execute_optimized_query(
                partition.query_fragment,
                preferred_db=partition.database_type
            )
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                partition_id=partition.partition_id,
                data=result.data,
                execution_time=execution_time,
                rows_processed=result.row_count,
                memory_used=0.0,  # Would be measured in practice
                metadata=partition.metadata
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                partition_id=partition.partition_id,
                data=None,
                execution_time=execution_time,
                rows_processed=0,
                memory_used=0.0,
                error=str(e),
                metadata=partition.metadata
            )
    
    def _execute_partition_in_process(self, partition: PartitionInfo) -> ExecutionResult:
        """Execute partition in separate process (for distributed simulation)"""
        
        # This is a simplified version for process-based execution
        # In practice, this would involve serializing the partition info
        # and executing it in a separate process or node
        
        start_time = time.time()
        
        try:
            # Simulate query execution
            import random
            execution_time = random.uniform(0.1, 2.0)
            rows_processed = random.randint(100, 10000)
            
            # Simulate some data
            data = [{'id': i, 'value': random.random()} for i in range(min(rows_processed, 100))]
            
            return ExecutionResult(
                partition_id=partition.partition_id,
                data=data,
                execution_time=execution_time,
                rows_processed=rows_processed,
                memory_used=random.uniform(10, 100),
                metadata=partition.metadata
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                partition_id=partition.partition_id,
                data=None,
                execution_time=execution_time,
                rows_processed=0,
                memory_used=0.0,
                error=str(e),
                metadata=partition.metadata
            )
    
    async def _aggregate_results(self, results: List[ExecutionResult], 
                               strategy: str) -> Any:
        """Aggregate results from multiple partitions"""
        
        # Filter successful results
        successful_results = [r for r in results if r.error is None and r.data is not None]
        
        if not successful_results:
            return None
        
        # Apply aggregation strategy
        aggregation_func = self.aggregation_strategies.get(strategy, self._simple_concatenation)
        return await aggregation_func(successful_results)
    
    async def _simple_concatenation(self, results: List[ExecutionResult]) -> List[Any]:
        """Simple concatenation of results"""
        
        aggregated = []
        for result in results:
            if isinstance(result.data, list):
                aggregated.extend(result.data)
            else:
                aggregated.append(result.data)
        
        return aggregated
    
    async def _sum_aggregation(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Sum aggregation for numerical results"""
        
        totals = defaultdict(float)
        counts = defaultdict(int)
        
        for result in results:
            if isinstance(result.data, dict):
                for key, value in result.data.items():
                    if isinstance(value, (int, float)):
                        totals[key] += value
                        counts[key] += 1
            elif isinstance(result.data, list):
                for item in result.data:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if isinstance(value, (int, float)):
                                totals[key] += value
                                counts[key] += 1
        
        return dict(totals)
    
    async def _spatial_union(self, results: List[ExecutionResult]) -> Any:
        """Spatial union aggregation for geographical results"""
        
        if not GEOSPATIAL_AVAILABLE:
            return await self._simple_concatenation(results)
        
        # This would perform actual spatial union operations
        # For now, return simple concatenation
        return await self._simple_concatenation(results)
    
    async def _temporal_merge(self, results: List[ExecutionResult]) -> List[Any]:
        """Temporal merge aggregation for time-series results"""
        
        all_data = []
        
        for result in results:
            if isinstance(result.data, list):
                all_data.extend(result.data)
            else:
                all_data.append(result.data)
        
        # Sort by timestamp if available
        try:
            all_data.sort(key=lambda x: x.get('timestamp', x.get('time', 0)))
        except (AttributeError, TypeError):
            pass  # Data doesn't have sortable timestamps
        
        return all_data
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect query type from query content"""
        
        query_lower = query.lower()
        
        if any(spatial_keyword in query_lower for spatial_keyword in ['st_', 'geometry', 'spatial']):
            return QueryType.SPATIAL
        elif any(temporal_keyword in query_lower for temporal_keyword in ['timestamp', 'time_series']):
            return QueryType.TEMPORAL
        elif any(agg_keyword in query_lower for agg_keyword in ['sum(', 'count(', 'avg(', 'group by']):
            return QueryType.AGGREGATE
        else:
            return QueryType.SELECT
    
    async def _estimate_data_size(self, query: str, query_type: QueryType) -> int:
        """Estimate data size for query"""
        
        # This would typically run EXPLAIN or similar to get row count estimates
        # For now, return a reasonable default based on query type
        
        if query_type == QueryType.SPATIAL:
            return 100000  # Spatial queries often involve many points
        elif query_type == QueryType.TEMPORAL:
            return 500000  # Time-series data can be large
        elif query_type == QueryType.AGGREGATE:
            return 1000000  # Aggregations typically process lots of data
        else:
            return 10000  # Default estimate
    
    def _get_available_resources(self) -> Dict[str, Any]:
        """Get available system resources"""
        
        try:
            import psutil
            
            return {
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / (1024**3),
                'disk_space_gb': psutil.disk_usage('/').free / (1024**3)
            }
        except ImportError:
            return {
                'cpu_cores': mp.cpu_count(),
                'memory_gb': 8,  # Default assumption
                'disk_space_gb': 100
            }
    
    def _record_execution_history(self, plan: ExecutionPlan, execution_time: float,
                                success: bool, error_message: str = None):
        """Record execution history for analysis"""
        
        history_entry = {
            'plan_id': plan.plan_id,
            'execution_time': execution_time,
            'success': success,
            'error_message': error_message,
            'execution_mode': plan.execution_mode.value,
            'num_partitions': len(plan.partitions),
            'parallelism_degree': plan.parallelism_degree,
            'estimated_cost': plan.estimated_total_cost,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.execution_history.append(history_entry)
        
        # Keep only recent history
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics"""
        
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        successful_executions = [h for h in self.execution_history if h['success']]
        failed_executions = [h for h in self.execution_history if not h['success']]
        
        execution_times = [h['execution_time'] for h in successful_executions]
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': len(successful_executions),
            'failed_executions': len(failed_executions),
            'success_rate': len(successful_executions) / len(self.execution_history),
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
            'median_execution_time': statistics.median(execution_times) if execution_times else 0,
            'max_execution_time': max(execution_times) if execution_times else 0,
            'execution_modes': {
                mode.value: len([h for h in self.execution_history if h['execution_mode'] == mode.value])
                for mode in ExecutionMode
            },
            'active_executions': len(self.active_executions)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Distributed Query Executor cleanup complete")

# Global distributed executor instance
_distributed_executor: Optional[DistributedQueryExecutor] = None

async def get_distributed_executor() -> DistributedQueryExecutor:
    """Get global distributed executor instance"""
    global _distributed_executor
    
    if _distributed_executor is None:
        from backend.core.unified_database_manager import get_database_manager
        from backend.core.cross_database_query_optimizer import get_query_optimizer
        
        db_manager = await get_database_manager()
        query_optimizer = await get_query_optimizer()
        _distributed_executor = DistributedQueryExecutor(db_manager, query_optimizer)
    
    return _distributed_executor

async def cleanup_distributed_executor():
    """Cleanup global distributed executor"""
    global _distributed_executor
    
    if _distributed_executor:
        await _distributed_executor.cleanup()
        _distributed_executor = None