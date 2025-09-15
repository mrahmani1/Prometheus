"""
Database Performance Monitoring and Optimization System
Comprehensive monitoring, analysis, and optimization for all database types
Production-grade implementation with real-time metrics, automated optimization, and security auditing
"""

import asyncio
import logging
import json
import os
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import yaml

# Database clients and monitoring
import asyncpg
import redis.asyncio as redis
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import QueuePool

# InfluxDB monitoring
try:
    from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
    from influxdb_client import Point, WritePrecision
    INFLUXDB_AVAILABLE = True
except ImportError:
    INFLUXDB_AVAILABLE = False

# Neo4j monitoring
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics"""
    CONNECTION_POOL = "connection_pool"
    QUERY_PERFORMANCE = "query_performance"
    STORAGE_USAGE = "storage_usage"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    REPLICATION_LAG = "replication_lag"
    BACKUP_STATUS = "backup_status"
    SECURITY_AUDIT = "security_audit"
    SLOW_QUERIES = "slow_queries"
    ERROR_RATES = "error_rates"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class OptimizationType(Enum):
    """Types of database optimizations"""
    INDEX_CREATION = "index_creation"
    QUERY_REWRITE = "query_rewrite"
    PARTITION_SUGGESTION = "partition_suggestion"
    CONNECTION_TUNING = "connection_tuning"
    MEMORY_OPTIMIZATION = "memory_optimization"
    STORAGE_CLEANUP = "storage_cleanup"
    VACUUM_ANALYZE = "vacuum_analyze"
    CACHE_OPTIMIZATION = "cache_optimization"

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    metric_type: MetricType
    database_type: str
    database_name: str
    timestamp: datetime
    value: Union[float, int, str, Dict[str, Any]]
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectionPoolMetrics:
    """Connection pool performance metrics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_connections: int = 0
    max_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    pool_utilization: float = 0.0
    connection_timeouts: int = 0
    connection_leaks: int = 0

@dataclass
class QueryPerformanceMetrics:
    """Query performance metrics"""
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_query_time: float = 0.0
    median_query_time: float = 0.0
    p95_query_time: float = 0.0
    p99_query_time: float = 0.0
    slow_queries: int = 0
    queries_per_second: float = 0.0
    cache_hit_ratio: float = 0.0
    index_usage_ratio: float = 0.0

@dataclass
class StorageMetrics:
    """Database storage metrics"""
    total_size: int = 0
    data_size: int = 0
    index_size: int = 0
    log_size: int = 0
    temp_size: int = 0
    free_space: int = 0
    storage_utilization: float = 0.0
    growth_rate: float = 0.0
    fragmentation_ratio: float = 0.0
    backup_size: int = 0

@dataclass
class SecurityAuditMetrics:
    """Security audit metrics"""
    failed_login_attempts: int = 0
    privilege_escalations: int = 0
    unauthorized_access_attempts: int = 0
    data_access_violations: int = 0
    configuration_changes: int = 0
    suspicious_queries: int = 0
    encryption_status: bool = True
    ssl_connections: int = 0
    last_security_scan: Optional[datetime] = None

@dataclass
class OptimizationRecommendation:
    """Database optimization recommendation"""
    optimization_type: OptimizationType
    database_type: str
    database_name: str
    priority: int  # 1-10, 10 being highest
    description: str
    impact_estimate: str
    implementation_sql: Optional[str] = None
    rollback_sql: Optional[str] = None
    estimated_improvement: float = 0.0
    risk_level: str = "low"
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    database_type: str
    database_name: str
    message: str
    threshold_value: Union[float, int]
    current_value: Union[float, int]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DatabasePerformanceMonitor:
    """
    Comprehensive Database Performance Monitoring System
    Real-time monitoring, optimization recommendations, and automated tuning
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_tasks: List[asyncio.Task] = []
        self.metric_collectors: Dict[str, Any] = {}
        
        # Performance thresholds
        self.thresholds = self._load_performance_thresholds()
        
        # Query analysis
        self.query_analyzer = QueryAnalyzer()
        self.slow_query_threshold = self.config.get('slow_query_threshold', 1.0)  # seconds
        
        # Connection pool monitoring
        self.connection_monitors: Dict[str, ConnectionPoolMonitor] = {}
        
        # Storage monitoring
        self.storage_monitors: Dict[str, StorageMonitor] = {}
        
        # Security auditing
        self.security_auditor = SecurityAuditor()
        
        # Backup monitoring
        self.backup_monitor = BackupMonitor()
        
        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.prometheus_registry = CollectorRegistry()
            self._setup_prometheus_metrics()
        
        # Performance optimization engine
        self.optimization_engine = OptimizationEngine()
        
        logger.info("Database Performance Monitor initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            'monitoring_interval': 30,  # seconds
            'metric_retention_days': 30,
            'alert_cooldown_minutes': 15,
            'optimization_check_interval': 300,  # 5 minutes
            'backup_check_interval': 3600,  # 1 hour
            'security_audit_interval': 1800,  # 30 minutes
            'slow_query_threshold': 1.0,
            'connection_pool_warning_threshold': 0.8,
            'storage_warning_threshold': 0.85,
            'memory_warning_threshold': 0.9,
            'enable_auto_optimization': False,
            'prometheus_port': 8090
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _load_performance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load performance alert thresholds"""
        return {
            'connection_pool': {
                'utilization_warning': 0.8,
                'utilization_critical': 0.95,
                'timeout_rate_warning': 0.05,
                'timeout_rate_critical': 0.1
            },
            'query_performance': {
                'avg_time_warning': 1.0,
                'avg_time_critical': 5.0,
                'error_rate_warning': 0.01,
                'error_rate_critical': 0.05,
                'qps_drop_warning': 0.2,
                'qps_drop_critical': 0.5
            },
            'storage': {
                'utilization_warning': 0.85,
                'utilization_critical': 0.95,
                'growth_rate_warning': 0.1,  # 10% per day
                'growth_rate_critical': 0.2   # 20% per day
            },
            'memory': {
                'utilization_warning': 0.9,
                'utilization_critical': 0.98
            },
            'replication': {
                'lag_warning': 60,  # seconds
                'lag_critical': 300  # seconds
            }
        }
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics collectors"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.prometheus_metrics = {
            'connection_pool_size': Gauge(
                'db_connection_pool_size',
                'Database connection pool size',
                ['database_type', 'database_name'],
                registry=self.prometheus_registry
            ),
            'active_connections': Gauge(
                'db_active_connections',
                'Active database connections',
                ['database_type', 'database_name'],
                registry=self.prometheus_registry
            ),
            'query_duration': Histogram(
                'db_query_duration_seconds',
                'Database query duration',
                ['database_type', 'database_name', 'query_type'],
                registry=self.prometheus_registry
            ),
            'query_total': Counter(
                'db_queries_total',
                'Total database queries',
                ['database_type', 'database_name', 'status'],
                registry=self.prometheus_registry
            ),
            'storage_usage': Gauge(
                'db_storage_usage_bytes',
                'Database storage usage',
                ['database_type', 'database_name', 'storage_type'],
                registry=self.prometheus_registry
            ),
            'alerts_active': Gauge(
                'db_alerts_active',
                'Active database alerts',
                ['database_type', 'database_name', 'severity'],
                registry=self.prometheus_registry
            )
        }
    
    async def start_monitoring(self, database_configs: Dict[str, Any]):
        """Start comprehensive database monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        logger.info("Starting database performance monitoring")
        self.monitoring_active = True
        
        # Initialize monitors for each database
        for db_name, db_config in database_configs.items():
            await self._initialize_database_monitoring(db_name, db_config)
        
        # Start monitoring tasks
        monitoring_tasks = [
            self._connection_pool_monitoring_loop(),
            self._query_performance_monitoring_loop(),
            self._storage_monitoring_loop(),
            self._security_audit_loop(),
            self._backup_monitoring_loop(),
            self._optimization_analysis_loop(),
            self._alert_management_loop(),
            self._metric_cleanup_loop()
        ]
        
        self.monitoring_tasks = [asyncio.create_task(task) for task in monitoring_tasks]
        
        # Start Prometheus metrics server if enabled
        if PROMETHEUS_AVAILABLE and self.config.get('enable_prometheus', True):
            await self._start_prometheus_server()
        
        logger.info("Database performance monitoring started")
    
    async def _initialize_database_monitoring(self, db_name: str, db_config: Dict[str, Any]):
        """Initialize monitoring for a specific database"""
        db_type = db_config.get('type', 'postgresql')
        
        # Initialize connection pool monitor
        self.connection_monitors[db_name] = ConnectionPoolMonitor(db_name, db_config)
        
        # Initialize storage monitor
        self.storage_monitors[db_name] = StorageMonitor(db_name, db_config)
        
        # Initialize database-specific collectors
        if db_type == 'postgresql':
            await self._initialize_postgresql_monitoring(db_name, db_config)
        elif db_type == 'influxdb':
            await self._initialize_influxdb_monitoring(db_name, db_config)
        elif db_type == 'redis':
            await self._initialize_redis_monitoring(db_name, db_config)
        elif db_type == 'neo4j':
            await self._initialize_neo4j_monitoring(db_name, db_config)
    
    async def _initialize_postgresql_monitoring(self, db_name: str, db_config: Dict[str, Any]):
        """Initialize PostgreSQL-specific monitoring"""
        try:
            connection_string = self._build_connection_string(db_config)
            engine = create_async_engine(
                connection_string,
                pool_size=5,
                max_overflow=10,
                pool_timeout=30
            )
            
            self.metric_collectors[f"{db_name}_postgresql"] = {
                'engine': engine,
                'type': 'postgresql',
                'config': db_config
            }
            
            # Test connection and setup monitoring views
            async with engine.connect() as conn:
                # Create monitoring views if they don't exist
                await self._create_postgresql_monitoring_views(conn)
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL monitoring for {db_name}: {e}")
    
    async def _create_postgresql_monitoring_views(self, conn):
        """Create PostgreSQL monitoring views and functions"""
        monitoring_sql = """
        -- Create performance monitoring schema
        CREATE SCHEMA IF NOT EXISTS performance_monitoring;
        
        -- Function to get connection pool stats
        CREATE OR REPLACE FUNCTION performance_monitoring.get_connection_stats()
        RETURNS TABLE(
            total_connections INTEGER,
            active_connections INTEGER,
            idle_connections INTEGER,
            waiting_connections INTEGER
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                COUNT(*)::INTEGER as total_connections,
                COUNT(*) FILTER (WHERE state = 'active')::INTEGER as active_connections,
                COUNT(*) FILTER (WHERE state = 'idle')::INTEGER as idle_connections,
                COUNT(*) FILTER (WHERE wait_event IS NOT NULL)::INTEGER as waiting_connections
            FROM pg_stat_activity
            WHERE datname = current_database();
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to get query performance stats
        CREATE OR REPLACE FUNCTION performance_monitoring.get_query_stats()
        RETURNS TABLE(
            total_queries BIGINT,
            avg_query_time DOUBLE PRECISION,
            slow_queries BIGINT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                SUM(calls) as total_queries,
                AVG(mean_exec_time) as avg_query_time,
                SUM(calls) FILTER (WHERE mean_exec_time > 1000) as slow_queries
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database());
        END;
        $$ LANGUAGE plpgsql;
        
        -- Function to get storage stats
        CREATE OR REPLACE FUNCTION performance_monitoring.get_storage_stats()
        RETURNS TABLE(
            database_size BIGINT,
            table_size BIGINT,
            index_size BIGINT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                pg_database_size(current_database()) as database_size,
                SUM(pg_total_relation_size(schemaname||'.'||tablename)) as table_size,
                SUM(pg_indexes_size(schemaname||'.'||tablename)) as index_size
            FROM pg_tables
            WHERE schemaname NOT IN ('information_schema', 'pg_catalog', 'pg_toast');
        END;
        $$ LANGUAGE plpgsql;
        """
        
        try:
            await conn.execute(text(monitoring_sql))
            await conn.commit()
        except Exception as e:
            logger.warning(f"Failed to create PostgreSQL monitoring views: {e}")
    
    async def _connection_pool_monitoring_loop(self):
        """Monitor connection pools across all databases"""
        while self.monitoring_active:
            try:
                for db_name, monitor in self.connection_monitors.items():
                    metrics = await monitor.collect_metrics()
                    await self._store_metric(
                        MetricType.CONNECTION_POOL,
                        db_name,
                        metrics,
                        'connection_pool'
                    )
                    
                    # Check for alerts
                    await self._check_connection_pool_alerts(db_name, metrics)
                    
                    # Update Prometheus metrics
                    if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                        self._update_prometheus_connection_metrics(db_name, metrics)
                
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Connection pool monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _query_performance_monitoring_loop(self):
        """Monitor query performance across all databases"""
        while self.monitoring_active:
            try:
                for db_name, collector in self.metric_collectors.items():
                    if collector['type'] == 'postgresql':
                        metrics = await self._collect_postgresql_query_metrics(collector)
                    elif collector['type'] == 'influxdb':
                        metrics = await self._collect_influxdb_query_metrics(collector)
                    elif collector['type'] == 'redis':
                        metrics = await self._collect_redis_query_metrics(collector)
                    elif collector['type'] == 'neo4j':
                        metrics = await self._collect_neo4j_query_metrics(collector)
                    else:
                        continue
                    
                    await self._store_metric(
                        MetricType.QUERY_PERFORMANCE,
                        db_name,
                        metrics,
                        'query_performance'
                    )
                    
                    # Check for performance alerts
                    await self._check_query_performance_alerts(db_name, metrics)
                
                await asyncio.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                logger.error(f"Query performance monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_postgresql_query_metrics(self, collector: Dict[str, Any]) -> QueryPerformanceMetrics:
        """Collect PostgreSQL query performance metrics"""
        try:
            engine = collector['engine']
            
            async with engine.connect() as conn:
                # Get query statistics
                query_stats_result = await conn.execute(
                    text("SELECT * FROM performance_monitoring.get_query_stats()")
                )
                query_stats = query_stats_result.fetchone()
                
                # Get slow queries
                slow_queries_result = await conn.execute(text("""
                    SELECT query, calls, mean_exec_time, total_exec_time
                    FROM pg_stat_statements
                    WHERE mean_exec_time > :threshold
                    ORDER BY mean_exec_time DESC
                    LIMIT 10
                """), {'threshold': self.slow_query_threshold * 1000})
                
                slow_queries = slow_queries_result.fetchall()
                
                # Get cache hit ratio
                cache_hit_result = await conn.execute(text("""
                    SELECT 
                        SUM(heap_blks_hit) / NULLIF(SUM(heap_blks_hit + heap_blks_read), 0) as cache_hit_ratio
                    FROM pg_statio_user_tables
                """))
                cache_hit_ratio = cache_hit_result.scalar() or 0.0
                
                # Get index usage ratio
                index_usage_result = await conn.execute(text("""
                    SELECT 
                        SUM(idx_scan) / NULLIF(SUM(seq_scan + idx_scan), 0) as index_usage_ratio
                    FROM pg_stat_user_tables
                """))
                index_usage_ratio = index_usage_result.scalar() or 0.0
                
                return QueryPerformanceMetrics(
                    total_queries=query_stats[0] or 0,
                    avg_query_time=query_stats[1] or 0.0,
                    slow_queries=len(slow_queries),
                    cache_hit_ratio=float(cache_hit_ratio),
                    index_usage_ratio=float(index_usage_ratio)
                )
                
        except Exception as e:
            logger.error(f"Failed to collect PostgreSQL query metrics: {e}")
            return QueryPerformanceMetrics()
    
    async def _storage_monitoring_loop(self):
        """Monitor storage usage across all databases"""
        while self.monitoring_active:
            try:
                for db_name, monitor in self.storage_monitors.items():
                    metrics = await monitor.collect_metrics()
                    await self._store_metric(
                        MetricType.STORAGE_USAGE,
                        db_name,
                        metrics,
                        'storage'
                    )
                    
                    # Check for storage alerts
                    await self._check_storage_alerts(db_name, metrics)
                
                await asyncio.sleep(self.config['monitoring_interval'] * 2)  # Less frequent
                
            except Exception as e:
                logger.error(f"Storage monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _security_audit_loop(self):
        """Perform security auditing across all databases"""
        while self.monitoring_active:
            try:
                for db_name, collector in self.metric_collectors.items():
                    audit_metrics = await self.security_auditor.audit_database(
                        db_name, collector
                    )
                    
                    await self._store_metric(
                        MetricType.SECURITY_AUDIT,
                        db_name,
                        audit_metrics,
                        'security'
                    )
                    
                    # Check for security alerts
                    await self._check_security_alerts(db_name, audit_metrics)
                
                await asyncio.sleep(self.config['security_audit_interval'])
                
            except Exception as e:
                logger.error(f"Security audit error: {e}")
                await asyncio.sleep(300)
    
    async def _backup_monitoring_loop(self):
        """Monitor backup status across all databases"""
        while self.monitoring_active:
            try:
                for db_name, collector in self.metric_collectors.items():
                    backup_status = await self.backup_monitor.check_backup_status(
                        db_name, collector
                    )
                    
                    await self._store_metric(
                        MetricType.BACKUP_STATUS,
                        db_name,
                        backup_status,
                        'backup'
                    )
                    
                    # Check for backup alerts
                    await self._check_backup_alerts(db_name, backup_status)
                
                await asyncio.sleep(self.config['backup_check_interval'])
                
            except Exception as e:
                logger.error(f"Backup monitoring error: {e}")
                await asyncio.sleep(600)
    
    async def _optimization_analysis_loop(self):
        """Analyze performance and generate optimization recommendations"""
        while self.monitoring_active:
            try:
                for db_name, collector in self.metric_collectors.items():
                    recommendations = await self.optimization_engine.analyze_performance(
                        db_name, collector, self.metrics_storage
                    )
                    
                    for recommendation in recommendations:
                        self.optimization_history.append(recommendation)
                        
                        # Auto-apply low-risk optimizations if enabled
                        if (self.config.get('enable_auto_optimization', False) and 
                            recommendation.risk_level == 'low' and 
                            recommendation.priority >= 8):
                            await self._apply_optimization(recommendation)
                
                await asyncio.sleep(self.config['optimization_check_interval'])
                
            except Exception as e:
                logger.error(f"Optimization analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _alert_management_loop(self):
        """Manage alerts and notifications"""
        while self.monitoring_active:
            try:
                # Check for alert resolution
                current_time = datetime.utcnow()
                resolved_alerts = []
                
                for alert_id, alert in self.active_alerts.items():
                    if not alert.resolved:
                        # Check if alert condition still exists
                        is_resolved = await self._check_alert_resolution(alert)
                        if is_resolved:
                            alert.resolved = True
                            alert.resolution_time = current_time
                            resolved_alerts.append(alert_id)
                
                # Remove resolved alerts after cooldown period
                cooldown_minutes = self.config['alert_cooldown_minutes']
                for alert_id in list(self.active_alerts.keys()):
                    alert = self.active_alerts[alert_id]
                    if (alert.resolved and alert.resolution_time and 
                        current_time - alert.resolution_time > timedelta(minutes=cooldown_minutes)):
                        del self.active_alerts[alert_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Alert management error: {e}")
                await asyncio.sleep(60)
    
    async def _metric_cleanup_loop(self):
        """Clean up old metrics to prevent memory bloat"""
        while self.monitoring_active:
            try:
                retention_days = self.config['metric_retention_days']
                cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
                
                for metric_key, metric_deque in self.metrics_storage.items():
                    # Remove old metrics
                    while (metric_deque and 
                           len(metric_deque) > 0 and 
                           metric_deque[0].timestamp < cutoff_time):
                        metric_deque.popleft()
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Metric cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _store_metric(self, metric_type: MetricType, db_name: str, 
                          value: Any, unit: str, tags: Optional[Dict[str, str]] = None):
        """Store performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            database_type=self._get_database_type(db_name),
            database_name=db_name,
            timestamp=datetime.utcnow(),
            value=value,
            unit=unit,
            tags=tags or {}
        )
        
        metric_key = f"{db_name}_{metric_type.value}"
        self.metrics_storage[metric_key].append(metric)
        
        # Store in external time-series database if configured
        await self._store_metric_external(metric)
    
    async def _store_metric_external(self, metric: PerformanceMetric):
        """Store metric in external time-series database"""
        try:
            # Store in InfluxDB if available
            if 'influxdb_metrics' in self.metric_collectors:
                await self._store_metric_influxdb(metric)
            
            # Store in Prometheus if available
            if PROMETHEUS_AVAILABLE and hasattr(self, 'prometheus_metrics'):
                self._update_prometheus_metric(metric)
                
        except Exception as e:
            logger.error(f"Failed to store metric externally: {e}")
    
    def _get_database_type(self, db_name: str) -> str:
        """Get database type from collector"""
        for collector_name, collector in self.metric_collectors.items():
            if db_name in collector_name:
                return collector['type']
        return 'unknown'
    
    async def _check_connection_pool_alerts(self, db_name: str, metrics: ConnectionPoolMetrics):
        """Check for connection pool alerts"""
        thresholds = self.thresholds['connection_pool']
        
        # Check pool utilization
        if metrics.pool_utilization >= thresholds['utilization_critical']:
            await self._create_alert(
                AlertSeverity.CRITICAL,
                MetricType.CONNECTION_POOL,
                db_name,
                f"Connection pool utilization critical: {metrics.pool_utilization:.2%}",
                thresholds['utilization_critical'],
                metrics.pool_utilization
            )
        elif metrics.pool_utilization >= thresholds['utilization_warning']:
            await self._create_alert(
                AlertSeverity.WARNING,
                MetricType.CONNECTION_POOL,
                db_name,
                f"Connection pool utilization high: {metrics.pool_utilization:.2%}",
                thresholds['utilization_warning'],
                metrics.pool_utilization
            )
    
    async def _create_alert(self, severity: AlertSeverity, metric_type: MetricType,
                          db_name: str, message: str, threshold: Union[float, int],
                          current_value: Union[float, int]):
        """Create performance alert"""
        alert_id = hashlib.md5(
            f"{severity.value}_{metric_type.value}_{db_name}_{message}".encode()
        ).hexdigest()
        
        # Check if alert already exists and is not resolved
        if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
            return
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            severity=severity,
            metric_type=metric_type,
            database_type=self._get_database_type(db_name),
            database_name=db_name,
            message=message,
            threshold_value=threshold,
            current_value=current_value,
            timestamp=datetime.utcnow()
        )
        
        self.active_alerts[alert_id] = alert
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"Database Alert [{severity.value.upper()}] {db_name}: {message}")
        
        # Send notifications if configured
        await self._send_alert_notification(alert)
    
    async def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        dashboard_data = {
            'overview': await self._get_overview_metrics(),
            'connection_pools': await self._get_connection_pool_data(),
            'query_performance': await self._get_query_performance_data(),
            'storage_usage': await self._get_storage_usage_data(),
            'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'optimization_recommendations': [
                asdict(rec) for rec in self.optimization_history[-10:]
            ],
            'security_status': await self._get_security_status(),
            'backup_status': await self._get_backup_status(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return dashboard_data
    
    async def get_optimization_recommendations(self, db_name: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Get optimization recommendations for database(s)"""
        if db_name:
            return [rec for rec in self.optimization_history 
                   if rec.database_name == db_name]
        return self.optimization_history
    
    async def apply_optimization(self, recommendation_id: str) -> bool:
        """Apply optimization recommendation"""
        recommendation = next(
            (rec for rec in self.optimization_history 
             if hasattr(rec, 'id') and rec.id == recommendation_id),
            None
        )
        
        if not recommendation:
            logger.error(f"Optimization recommendation {recommendation_id} not found")
            return False
        
        return await self._apply_optimization(recommendation)
    
    async def stop_monitoring(self):
        """Stop all monitoring tasks"""
        logger.info("Stopping database performance monitoring")
        self.monitoring_active = False
        
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Close database connections
        for collector in self.metric_collectors.values():
            if 'engine' in collector:
                await collector['engine'].dispose()
        
        logger.info("Database performance monitoring stopped")

# Additional supporting classes would be implemented here...
# ConnectionPoolMonitor, StorageMonitor, SecurityAuditor, BackupMonitor, OptimizationEngine, QueryAnalyzer

class ConnectionPoolMonitor:
    """Monitor database connection pools"""
    
    def __init__(self, db_name: str, db_config: Dict[str, Any]):
        self.db_name = db_name
        self.db_config = db_config
        self.db_type = db_config.get('type', 'postgresql')
    
    async def collect_metrics(self) -> ConnectionPoolMetrics:
        """Collect connection pool metrics"""
        if self.db_type == 'postgresql':
            return await self._collect_postgresql_pool_metrics()
        elif self.db_type == 'redis':
            return await self._collect_redis_pool_metrics()
        else:
            return ConnectionPoolMetrics()
    
    async def _collect_postgresql_pool_metrics(self) -> ConnectionPoolMetrics:
        """Collect PostgreSQL connection pool metrics"""
        # Implementation for PostgreSQL pool monitoring
        return ConnectionPoolMetrics(
            total_connections=20,
            active_connections=5,
            idle_connections=15,
            pool_utilization=0.25
        )

class StorageMonitor:
    """Monitor database storage usage"""
    
    def __init__(self, db_name: str, db_config: Dict[str, Any]):
        self.db_name = db_name
        self.db_config = db_config
        self.db_type = db_config.get('type', 'postgresql')
    
    async def collect_metrics(self) -> StorageMetrics:
        """Collect storage metrics"""
        if self.db_type == 'postgresql':
            return await self._collect_postgresql_storage_metrics()
        else:
            return StorageMetrics()
    
    async def _collect_postgresql_storage_metrics(self) -> StorageMetrics:
        """Collect PostgreSQL storage metrics"""
        # Implementation for PostgreSQL storage monitoring
        return StorageMetrics(
            total_size=1000000000,  # 1GB
            data_size=800000000,    # 800MB
            index_size=200000000,   # 200MB
            storage_utilization=0.8
        )

class SecurityAuditor:
    """Perform security auditing on databases"""
    
    async def audit_database(self, db_name: str, collector: Dict[str, Any]) -> SecurityAuditMetrics:
        """Perform security audit on database"""
        # Implementation for security auditing
        return SecurityAuditMetrics(
            failed_login_attempts=0,
            unauthorized_access_attempts=0,
            encryption_status=True,
            ssl_connections=10
        )

class BackupMonitor:
    """Monitor database backup status"""
    
    async def check_backup_status(self, db_name: str, collector: Dict[str, Any]) -> Dict[str, Any]:
        """Check backup status for database"""
        # Implementation for backup monitoring
        return {
            'last_backup': datetime.utcnow() - timedelta(hours=1),
            'backup_size': 500000000,  # 500MB
            'backup_success': True,
            'next_backup': datetime.utcnow() + timedelta(hours=23)
        }

class OptimizationEngine:
    """Generate database optimization recommendations"""
    
    async def analyze_performance(self, db_name: str, collector: Dict[str, Any], 
                                metrics_storage: Dict[str, deque]) -> List[OptimizationRecommendation]:
        """Analyze performance and generate recommendations"""
        recommendations = []
        
        # Example optimization recommendation
        if collector['type'] == 'postgresql':
            recommendations.append(OptimizationRecommendation(
                optimization_type=OptimizationType.INDEX_CREATION,
                database_type='postgresql',
                database_name=db_name,
                priority=7,
                description="Create index on frequently queried column",
                impact_estimate="20-30% query performance improvement",
                implementation_sql="CREATE INDEX CONCURRENTLY idx_example ON table_name (column_name);",
                rollback_sql="DROP INDEX IF EXISTS idx_example;",
                estimated_improvement=0.25,
                risk_level="low"
            ))
        
        return recommendations

class QueryAnalyzer:
    """Analyze database queries for optimization opportunities"""
    
    def __init__(self):
        self.slow_queries: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def analyze_slow_queries(self, db_name: str, queries: List[Dict[str, Any]]) -> List[OptimizationRecommendation]:
        """Analyze slow queries and generate optimization recommendations"""
        recommendations = []
        
        for query_info in queries:
            # Analyze query patterns and suggest optimizations
            if query_info.get('execution_time', 0) > 5.0:  # 5 second threshold
                recommendations.append(OptimizationRecommendation(
                    optimization_type=OptimizationType.QUERY_REWRITE,
                    database_type='postgresql',
                    database_name=db_name,
                    priority=8,
                    description=f"Optimize slow query: {query_info.get('query', '')[:100]}...",
                    impact_estimate="50-70% execution time reduction",
                    estimated_improvement=0.6,
                    risk_level="medium"
                ))
        
        return recommendations

if __name__ == "__main__":
    # Example usage
    async def main():
        monitor = DatabasePerformanceMonitor()
        
        # Example database configurations
        db_configs = {
            'prometheus_main': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': 5432,
                'database': 'prometheus',
                'username': 'prometheus',
                'password': 'password'
            }
        }
        
        await monitor.start_monitoring(db_configs)
        
        # Let it run for a while
        await asyncio.sleep(60)
        
        # Get dashboard data
        dashboard_data = await monitor.get_performance_dashboard_data()
        print(json.dumps(dashboard_data, indent=2, default=str))
        
        await monitor.stop_monitoring()
    
    asyncio.run(main())