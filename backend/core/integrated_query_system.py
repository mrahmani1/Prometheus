"""
Integrated Cross-Database Query System
Unified interface combining query optimization, caching, auditing, and distributed execution
for production-grade geological data analysis
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading

# Core components
from backend.core.unified_database_manager import (
    UnifiedDatabaseManager, DatabaseType, QueryResult, get_database_manager
)
from backend.core.cross_database_query_optimizer import (
    CrossDatabaseQueryOptimizer, QueryType, get_query_optimizer
)
from backend.core.query_audit_logger import (
    QueryAuditLogger, get_audit_logger
)
from backend.core.distributed_query_executor import (
    DistributedQueryExecutor, get_distributed_executor
)
from backend.core.database_config_manager import (
    DatabaseConfigManager, get_config_manager
)

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    DISTRIBUTED = "distributed"

class ExecutionStrategy(Enum):
    """Query execution strategies"""
    DIRECT = "direct"
    OPTIMIZED = "optimized"
    CACHED = "cached"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"

@dataclass
class QueryRequest:
    """Comprehensive query request"""
    query: str
    query_type: Optional[QueryType] = None
    preferred_database: Optional[DatabaseType] = None
    execution_strategy: Optional[ExecutionStrategy] = None
    use_cache: bool = True
    enable_optimization: bool = True
    enable_distributed: bool = True
    timeout_seconds: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QueryResponse:
    """Comprehensive query response"""
    request_id: str
    query_result: QueryResult
    execution_strategy_used: ExecutionStrategy
    optimization_applied: List[str]
    cache_hit: bool
    execution_time: float
    audit_log_id: str
    performance_metrics: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class QueryComplexityAnalyzer:
    """Analyze query complexity to determine execution strategy"""
    
    def __init__(self):
        self.complexity_rules = {
            # Simple queries
            'simple_select': r'^\s*select\s+[\w\s,*]+\s+from\s+\w+\s*(?:where\s+[\w\s=<>\'\"]+)?\s*(?:limit\s+\d+)?\s*;?\s*$',
            'simple_insert': r'^\s*insert\s+into\s+\w+\s*\([^)]+\)\s*values\s*\([^)]+\)\s*;?\s*$',
            
            # Moderate complexity
            'joins': r'\s+(?:inner\s+|left\s+|right\s+|full\s+)?join\s+',
            'subqueries': r'\(\s*select\s+',
            'aggregations': r'\s+(?:group\s+by|having|order\s+by)\s+',
            
            # Complex queries
            'multiple_joins': r'(?:\s+(?:inner\s+|left\s+|right\s+|full\s+)?join\s+.*?){2,}',
            'window_functions': r'\s+(?:over\s*\(|row_number\(\)|rank\(\)|dense_rank\(\))',
            'recursive_cte': r'\s+with\s+recursive\s+',
            
            # Distributed indicators
            'cross_database': r'(?:postgresql|influxdb|neo4j|redis)',
            'large_datasets': r'(?:geological_observations|time_series_data|volcanic_systems)',
            'spatial_operations': r'\s+st_\w+\s*\(',
            'temporal_analysis': r'(?:time_series|temporal|range\s*\()'
        }
    
    def analyze_complexity(self, query: str) -> Tuple[QueryComplexity, Dict[str, Any]]:
        """Analyze query complexity and return complexity level with details"""
        
        import re
        
        query_lower = query.lower().strip()
        complexity_indicators = {}
        
        # Check for complexity patterns
        for pattern_name, pattern in self.complexity_rules.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE | re.MULTILINE)
            if matches:
                complexity_indicators[pattern_name] = len(matches)
        
        # Determine complexity level
        complexity_score = 0
        
        # Simple indicators (score: 1 each)
        simple_patterns = ['simple_select', 'simple_insert']
        complexity_score += sum(complexity_indicators.get(p, 0) for p in simple_patterns)
        
        # Moderate indicators (score: 3 each)
        moderate_patterns = ['joins', 'subqueries', 'aggregations']
        complexity_score += sum(complexity_indicators.get(p, 0) * 3 for p in moderate_patterns)
        
        # Complex indicators (score: 5 each)
        complex_patterns = ['multiple_joins', 'window_functions', 'recursive_cte']
        complexity_score += sum(complexity_indicators.get(p, 0) * 5 for p in complex_patterns)
        
        # Distributed indicators (score: 10 each)
        distributed_patterns = ['cross_database', 'large_datasets', 'spatial_operations', 'temporal_analysis']
        complexity_score += sum(complexity_indicators.get(p, 0) * 10 for p in distributed_patterns)
        
        # Determine complexity level
        if complexity_score >= 20:
            complexity = QueryComplexity.DISTRIBUTED
        elif complexity_score >= 10:
            complexity = QueryComplexity.COMPLEX
        elif complexity_score >= 3:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        analysis_details = {
            'complexity_score': complexity_score,
            'indicators': complexity_indicators,
            'query_length': len(query),
            'estimated_tables': len(re.findall(r'\bfrom\s+(\w+)', query_lower)) + 
                              len(re.findall(r'\bjoin\s+(\w+)', query_lower))
        }
        
        return complexity, analysis_details

class ExecutionStrategySelector:
    """Select optimal execution strategy based on query characteristics"""
    
    def __init__(self):
        self.strategy_rules = {
            QueryComplexity.SIMPLE: ExecutionStrategy.DIRECT,
            QueryComplexity.MODERATE: ExecutionStrategy.OPTIMIZED,
            QueryComplexity.COMPLEX: ExecutionStrategy.OPTIMIZED,
            QueryComplexity.DISTRIBUTED: ExecutionStrategy.DISTRIBUTED
        }
        
        # Performance thresholds
        self.cache_hit_threshold = 0.8  # Use cached strategy if cache hit rate > 80%
        self.distributed_size_threshold = 1000000  # Use distributed for > 1M rows
    
    def select_strategy(self, complexity: QueryComplexity, 
                       query_request: QueryRequest,
                       system_metrics: Dict[str, Any]) -> ExecutionStrategy:
        """Select optimal execution strategy"""
        
        # Check if user specified a strategy
        if query_request.execution_strategy:
            return query_request.execution_strategy
        
        # Check cache performance
        cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
        if cache_hit_rate > self.cache_hit_threshold and query_request.use_cache:
            return ExecutionStrategy.CACHED
        
        # Check for distributed execution needs
        estimated_data_size = system_metrics.get('estimated_data_size', 0)
        if (estimated_data_size > self.distributed_size_threshold and 
            query_request.enable_distributed):
            return ExecutionStrategy.DISTRIBUTED
        
        # Check system load
        system_load = system_metrics.get('system_load', 0.5)
        if system_load > 0.8:
            # High system load - prefer cached or simple strategies
            if query_request.use_cache:
                return ExecutionStrategy.CACHED
            else:
                return ExecutionStrategy.DIRECT
        
        # Default based on complexity
        base_strategy = self.strategy_rules.get(complexity, ExecutionStrategy.OPTIMIZED)
        
        # Consider hybrid approach for complex queries
        if complexity == QueryComplexity.COMPLEX and query_request.enable_optimization:
            return ExecutionStrategy.HYBRID
        
        return base_strategy

class IntegratedQuerySystem:
    """
    Integrated Cross-Database Query System
    Unified interface for all query processing capabilities with intelligent
    routing, optimization, caching, auditing, and distributed execution
    """
    
    def __init__(self):
        # Core components (initialized lazily)
        self.db_manager: Optional[UnifiedDatabaseManager] = None
        self.query_optimizer: Optional[CrossDatabaseQueryOptimizer] = None
        self.audit_logger: Optional[QueryAuditLogger] = None
        self.distributed_executor: Optional[DistributedQueryExecutor] = None
        self.config_manager: Optional[DatabaseConfigManager] = None
        
        # Analysis components
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.strategy_selector = ExecutionStrategySelector()
        
        # System state
        self.system_metrics: Dict[str, Any] = {}
        self.active_queries: Dict[str, QueryRequest] = {}
        self.query_history: List[Dict[str, Any]] = []
        
        # Performance monitoring
        self.performance_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'cache_hits': 0,
            'optimization_applied': 0,
            'distributed_executions': 0
        }
        
        # Background tasks
        self.metrics_update_task = None
        self.cleanup_task = None
        
        logger.info("Integrated Query System initialized")
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing Integrated Query System components")
            
            # Initialize core components
            self.db_manager = await get_database_manager()
            self.query_optimizer = await get_query_optimizer()
            self.audit_logger = get_audit_logger()
            self.distributed_executor = await get_distributed_executor()
            self.config_manager = get_config_manager()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update initial system metrics
            await self._update_system_metrics()
            
            logger.info("Integrated Query System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Integrated Query System: {e}")
            return False
    
    async def execute_query(self, query_request: QueryRequest) -> QueryResponse:
        """Execute query using integrated optimization pipeline"""
        
        request_id = f"req_{int(time.time())}_{len(self.active_queries)}"
        start_time = time.time()
        
        # Register active query
        self.active_queries[request_id] = query_request
        
        try:
            # Analyze query complexity
            complexity, complexity_details = self.complexity_analyzer.analyze_complexity(
                query_request.query
            )
            
            # Select execution strategy
            execution_strategy = self.strategy_selector.select_strategy(
                complexity, query_request, self.system_metrics
            )
            
            # Execute based on strategy
            if execution_strategy == ExecutionStrategy.DIRECT:
                result = await self._execute_direct(query_request)
            elif execution_strategy == ExecutionStrategy.OPTIMIZED:
                result = await self._execute_optimized(query_request)
            elif execution_strategy == ExecutionStrategy.CACHED:
                result = await self._execute_cached(query_request)
            elif execution_strategy == ExecutionStrategy.DISTRIBUTED:
                result = await self._execute_distributed(query_request)
            else:  # HYBRID
                result = await self._execute_hybrid(query_request)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Log audit entry
            audit_log_id = await self.audit_logger.log_query_execution(
                query_request.query, result, 
                self._create_query_metrics(result, execution_time),
                query_request.context
            )
            
            # Update performance statistics
            self._update_performance_stats(True, result.cached, execution_strategy)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                query_request, complexity, execution_strategy, result
            )
            
            # Create response
            response = QueryResponse(
                request_id=request_id,
                query_result=result,
                execution_strategy_used=execution_strategy,
                optimization_applied=getattr(result.metadata, 'optimizations', []),
                cache_hit=result.cached,
                execution_time=execution_time,
                audit_log_id=audit_log_id,
                performance_metrics={
                    'complexity': complexity.value,
                    'complexity_score': complexity_details['complexity_score'],
                    'estimated_tables': complexity_details['estimated_tables']
                },
                recommendations=recommendations
            )
            
            # Record query history
            self._record_query_history(query_request, response, True)
            
            return response
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            audit_log_id = await self.audit_logger.log_query_error(
                query_request.query, e, 
                query_request.preferred_database or DatabaseType.POSTGRESQL,
                query_request.context
            )
            
            # Update performance statistics
            self._update_performance_stats(False, False, ExecutionStrategy.DIRECT)
            
            # Create error response
            error_response = QueryResponse(
                request_id=request_id,
                query_result=QueryResult(
                    data=None, metadata={'error': str(e)}, 
                    execution_time=execution_time, row_count=0
                ),
                execution_strategy_used=ExecutionStrategy.DIRECT,
                optimization_applied=[],
                cache_hit=False,
                execution_time=execution_time,
                audit_log_id=audit_log_id,
                performance_metrics={},
                warnings=[f"Query execution failed: {str(e)}"]
            )
            
            # Record query history
            self._record_query_history(query_request, error_response, False)
            
            raise
            
        finally:
            # Cleanup active query
            if request_id in self.active_queries:
                del self.active_queries[request_id]
    
    async def _execute_direct(self, query_request: QueryRequest) -> QueryResult:
        """Execute query directly without optimization"""
        
        target_db = query_request.preferred_database or DatabaseType.POSTGRESQL
        
        return await self.db_manager.execute_query(
            target_db, query_request.query, 
            query_request.context.get('parameters'),
            use_cache=query_request.use_cache
        )
    
    async def _execute_optimized(self, query_request: QueryRequest) -> QueryResult:
        """Execute query with optimization"""
        
        return await self.query_optimizer.execute_optimized_query(
            query_request.query,
            params=query_request.context.get('parameters'),
            query_type=query_request.query_type,
            preferred_db=query_request.preferred_database,
            use_cache=query_request.use_cache
        )
    
    async def _execute_cached(self, query_request: QueryRequest) -> QueryResult:
        """Execute query with aggressive caching"""
        
        # Force cache usage
        return await self.query_optimizer.execute_optimized_query(
            query_request.query,
            params=query_request.context.get('parameters'),
            query_type=query_request.query_type,
            preferred_db=query_request.preferred_database,
            use_cache=True
        )
    
    async def _execute_distributed(self, query_request: QueryRequest) -> QueryResult:
        """Execute query using distributed processing"""
        
        return await self.distributed_executor.execute_distributed_query(
            query_request.query,
            query_type=query_request.query_type,
            estimated_data_size=self.system_metrics.get('estimated_data_size'),
            execution_options=query_request.context
        )
    
    async def _execute_hybrid(self, query_request: QueryRequest) -> QueryResult:
        """Execute query using hybrid approach"""
        
        # Try optimized execution first
        try:
            result = await self._execute_optimized(query_request)
            
            # If execution time is too high, consider distributed approach
            if result.execution_time > 10.0:  # 10 second threshold
                logger.info("Switching to distributed execution due to slow performance")
                return await self._execute_distributed(query_request)
            
            return result
            
        except Exception as e:
            # Fallback to direct execution
            logger.warning(f"Optimized execution failed, falling back to direct: {e}")
            return await self._execute_direct(query_request)
    
    def _create_query_metrics(self, result: QueryResult, execution_time: float):
        """Create query metrics for audit logging"""
        from backend.core.cross_database_query_optimizer import QueryMetrics, OptimizationStrategy
        
        return QueryMetrics(
            execution_time=execution_time,
            rows_processed=result.row_count,
            memory_usage=0.0,  # Would be measured in practice
            cpu_usage=0.0,     # Would be measured in practice
            cache_hit=result.cached,
            optimization_applied=[],  # Would be populated from result metadata
            database_type=result.database_type,
            query_hash="",  # Would be calculated
            timestamp=datetime.utcnow()
        )
    
    def _update_performance_stats(self, success: bool, cache_hit: bool, 
                                strategy: ExecutionStrategy):
        """Update performance statistics"""
        
        self.performance_stats['total_queries'] += 1
        
        if success:
            self.performance_stats['successful_queries'] += 1
        else:
            self.performance_stats['failed_queries'] += 1
        
        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        
        if strategy in [ExecutionStrategy.OPTIMIZED, ExecutionStrategy.HYBRID]:
            self.performance_stats['optimization_applied'] += 1
        
        if strategy == ExecutionStrategy.DISTRIBUTED:
            self.performance_stats['distributed_executions'] += 1
    
    def _generate_recommendations(self, query_request: QueryRequest, 
                                complexity: QueryComplexity,
                                strategy: ExecutionStrategy,
                                result: QueryResult) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if result.execution_time > 5.0:
            recommendations.append("Consider adding database indexes for better performance")
            
            if strategy != ExecutionStrategy.DISTRIBUTED:
                recommendations.append("Large query detected - consider distributed execution")
        
        # Complexity-based recommendations
        if complexity == QueryComplexity.COMPLEX and not query_request.enable_optimization:
            recommendations.append("Enable query optimization for complex queries")
        
        # Caching recommendations
        if not result.cached and result.execution_time > 1.0:
            recommendations.append("Consider enabling caching for frequently executed queries")
        
        # Database-specific recommendations
        if 'ST_' in query_request.query and result.database_type != DatabaseType.POSTGRESQL:
            recommendations.append("Use PostgreSQL with PostGIS for optimal spatial query performance")
        
        if 'time_series' in query_request.query.lower() and result.database_type != DatabaseType.INFLUXDB:
            recommendations.append("Consider InfluxDB for time-series data queries")
        
        return recommendations
    
    def _record_query_history(self, query_request: QueryRequest, 
                            response: QueryResponse, success: bool):
        """Record query execution in history"""
        
        history_entry = {
            'request_id': response.request_id,
            'timestamp': datetime.utcnow().isoformat(),
            'query_hash': hash(query_request.query),
            'execution_strategy': response.execution_strategy_used.value,
            'execution_time': response.execution_time,
            'success': success,
            'cache_hit': response.cache_hit,
            'row_count': response.query_result.row_count,
            'database_type': response.query_result.database_type.value if hasattr(response.query_result, 'database_type') else 'unknown'
        }
        
        self.query_history.append(history_entry)
        
        # Keep only recent history
        if len(self.query_history) > 10000:
            self.query_history = self.query_history[-10000:]
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Metrics update task
        self.metrics_update_task = asyncio.create_task(self._periodic_metrics_update())
        
        # Cleanup task
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start audit logger background tasks
        await self.audit_logger.start_background_tasks()
        
        logger.info("Background tasks started")
    
    async def _periodic_metrics_update(self):
        """Periodically update system metrics"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                await self._update_system_metrics()
                
            except Exception as e:
                logger.error(f"Metrics update failed: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Clean old query history
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                self.query_history = [
                    h for h in self.query_history 
                    if datetime.fromisoformat(h['timestamp']) > cutoff_time
                ]
                
                logger.info(f"Cleaned query history, {len(self.query_history)} entries remaining")
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
    
    async def _update_system_metrics(self):
        """Update system performance metrics"""
        
        try:
            # Get cache statistics
            if self.query_optimizer:
                cache_stats = self.query_optimizer.query_cache.get_statistics()
                self.system_metrics['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
                self.system_metrics['cache_size_mb'] = cache_stats.get('current_size_mb', 0.0)
            
            # Get system load (simplified)
            try:
                import psutil
                self.system_metrics['system_load'] = psutil.cpu_percent(interval=1) / 100.0
                self.system_metrics['memory_usage'] = psutil.virtual_memory().percent / 100.0
            except ImportError:
                self.system_metrics['system_load'] = 0.5  # Default assumption
                self.system_metrics['memory_usage'] = 0.5
            
            # Calculate query performance metrics
            if self.query_history:
                recent_queries = [
                    h for h in self.query_history 
                    if datetime.fromisoformat(h['timestamp']) > datetime.utcnow() - timedelta(hours=1)
                ]
                
                if recent_queries:
                    avg_execution_time = sum(h['execution_time'] for h in recent_queries) / len(recent_queries)
                    success_rate = sum(1 for h in recent_queries if h['success']) / len(recent_queries)
                    
                    self.system_metrics['avg_execution_time'] = avg_execution_time
                    self.system_metrics['success_rate'] = success_rate
            
            # Estimate typical data size (would be more sophisticated in practice)
            self.system_metrics['estimated_data_size'] = 100000  # Default estimate
            
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get component status
        component_status = {
            'database_manager': self.db_manager is not None,
            'query_optimizer': self.query_optimizer is not None,
            'audit_logger': self.audit_logger is not None,
            'distributed_executor': self.distributed_executor is not None,
            'config_manager': self.config_manager is not None
        }
        
        # Get performance statistics
        total_queries = self.performance_stats['total_queries']
        success_rate = (self.performance_stats['successful_queries'] / total_queries 
                       if total_queries > 0 else 0)
        cache_hit_rate = (self.performance_stats['cache_hits'] / total_queries 
                         if total_queries > 0 else 0)
        
        return {
            'system_status': 'operational' if all(component_status.values()) else 'degraded',
            'components': component_status,
            'performance_statistics': {
                **self.performance_stats,
                'success_rate': success_rate,
                'cache_hit_rate': cache_hit_rate
            },
            'system_metrics': self.system_metrics,
            'active_queries': len(self.active_queries),
            'query_history_size': len(self.query_history),
            'uptime': time.time() - (self.system_metrics.get('start_time', time.time()))
        }
    
    async def get_performance_report(self, time_window: timedelta = None) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        if time_window is None:
            time_window = timedelta(hours=24)
        
        # Get audit report
        audit_report = await self.audit_logger.get_audit_report(
            start_time=datetime.utcnow() - time_window
        )
        
        # Get optimization report
        optimization_report = await self.query_optimizer.get_optimization_report()
        
        # Get distributed execution statistics
        distributed_stats = self.distributed_executor.get_execution_statistics()
        
        # Combine reports
        return {
            'report_period': str(time_window),
            'generated_at': datetime.utcnow().isoformat(),
            'audit_report': audit_report,
            'optimization_report': optimization_report,
            'distributed_execution_statistics': distributed_stats,
            'system_status': await self.get_system_status()
        }
    
    async def optimize_system_configuration(self) -> Dict[str, Any]:
        """Analyze performance and suggest system optimizations"""
        
        recommendations = []
        
        # Analyze cache performance
        cache_hit_rate = self.system_metrics.get('cache_hit_rate', 0.0)
        if cache_hit_rate < 0.5:
            recommendations.append({
                'type': 'cache_optimization',
                'description': 'Low cache hit rate detected',
                'suggestion': 'Increase cache size or adjust TTL settings',
                'priority': 'high'
            })
        
        # Analyze query performance
        avg_execution_time = self.system_metrics.get('avg_execution_time', 0.0)
        if avg_execution_time > 2.0:
            recommendations.append({
                'type': 'query_performance',
                'description': 'High average query execution time',
                'suggestion': 'Enable distributed execution for large queries',
                'priority': 'medium'
            })
        
        # Analyze system resources
        system_load = self.system_metrics.get('system_load', 0.0)
        if system_load > 0.8:
            recommendations.append({
                'type': 'resource_optimization',
                'description': 'High system load detected',
                'suggestion': 'Consider scaling up resources or optimizing queries',
                'priority': 'high'
            })
        
        # Analyze error rates
        success_rate = self.system_metrics.get('success_rate', 1.0)
        if success_rate < 0.95:
            recommendations.append({
                'type': 'reliability',
                'description': 'Low query success rate',
                'suggestion': 'Review error logs and improve error handling',
                'priority': 'critical'
            })
        
        return {
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'system_health_score': self._calculate_health_score(),
            'recommendations': recommendations,
            'current_metrics': self.system_metrics
        }
    
    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        
        score = 100.0
        
        # Deduct for low cache hit rate
        cache_hit_rate = self.system_metrics.get('cache_hit_rate', 0.0)
        score -= max(0, (0.7 - cache_hit_rate) * 50)  # Deduct up to 50 points
        
        # Deduct for high execution times
        avg_execution_time = self.system_metrics.get('avg_execution_time', 0.0)
        if avg_execution_time > 1.0:
            score -= min(30, (avg_execution_time - 1.0) * 10)  # Deduct up to 30 points
        
        # Deduct for low success rate
        success_rate = self.system_metrics.get('success_rate', 1.0)
        score -= (1.0 - success_rate) * 100  # Deduct proportionally
        
        # Deduct for high system load
        system_load = self.system_metrics.get('system_load', 0.0)
        if system_load > 0.7:
            score -= (system_load - 0.7) * 100  # Deduct up to 30 points
        
        return max(0.0, min(100.0, score))
    
    async def cleanup(self):
        """Cleanup all system components"""
        
        logger.info("Starting Integrated Query System cleanup")
        
        # Cancel background tasks
        if self.metrics_update_task:
            self.metrics_update_task.cancel()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Cleanup components
        if self.audit_logger:
            await self.audit_logger.cleanup()
        
        if self.query_optimizer:
            await self.query_optimizer.cleanup()
        
        if self.distributed_executor:
            await self.distributed_executor.cleanup()
        
        if self.db_manager:
            await self.db_manager.cleanup()
        
        logger.info("Integrated Query System cleanup complete")

# Global integrated query system instance
_integrated_query_system: Optional[IntegratedQuerySystem] = None

async def get_integrated_query_system() -> IntegratedQuerySystem:
    """Get global integrated query system instance"""
    global _integrated_query_system
    
    if _integrated_query_system is None:
        _integrated_query_system = IntegratedQuerySystem()
        await _integrated_query_system.initialize()
    
    return _integrated_query_system

async def cleanup_integrated_query_system():
    """Cleanup global integrated query system"""
    global _integrated_query_system
    
    if _integrated_query_system:
        await _integrated_query_system.cleanup()
        _integrated_query_system = None

# Convenience functions for common operations
async def execute_geological_query(query: str, **kwargs) -> QueryResponse:
    """Execute geological query with automatic optimization"""
    
    system = await get_integrated_query_system()
    
    query_request = QueryRequest(
        query=query,
        query_type=kwargs.get('query_type'),
        preferred_database=kwargs.get('preferred_database'),
        execution_strategy=kwargs.get('execution_strategy'),
        use_cache=kwargs.get('use_cache', True),
        enable_optimization=kwargs.get('enable_optimization', True),
        enable_distributed=kwargs.get('enable_distributed', True),
        context=kwargs.get('context', {}),
        metadata=kwargs.get('metadata', {})
    )
    
    return await system.execute_query(query_request)

async def get_query_performance_report() -> Dict[str, Any]:
    """Get comprehensive query performance report"""
    
    system = await get_integrated_query_system()
    return await system.get_performance_report()

async def optimize_query_system() -> Dict[str, Any]:
    """Analyze and optimize query system configuration"""
    
    system = await get_integrated_query_system()
    return await system.optimize_system_configuration()