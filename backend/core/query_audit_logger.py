"""
Query Audit Logging and Performance Analytics System
Comprehensive logging, monitoring, and analytics for database query performance
"""

import asyncio
import logging
import json
import time
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import os
from pathlib import Path
import gzip
import pickle

# Database and monitoring
from backend.core.unified_database_manager import DatabaseType, QueryResult
from backend.core.cross_database_query_optimizer import QueryMetrics, OptimizationStrategy

# Structured logging
import structlog

logger = structlog.get_logger(__name__)

class AuditLevel(Enum):
    """Audit logging levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class QueryCategory(Enum):
    """Query categorization for analytics"""
    READ_ONLY = "read_only"
    WRITE_OPERATION = "write_operation"
    ADMINISTRATIVE = "administrative"
    ANALYTICAL = "analytical"
    REAL_TIME = "real_time"
    BATCH_PROCESSING = "batch_processing"

@dataclass
class QueryAuditEntry:
    """Complete audit entry for a database query"""
    query_id: str
    timestamp: datetime
    database_type: DatabaseType
    query_text: str
    query_hash: str
    query_category: QueryCategory
    execution_time: float
    rows_affected: int
    memory_usage: float
    cpu_usage: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    application_name: Optional[str] = None
    client_ip: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    optimization_applied: List[OptimizationStrategy] = field(default_factory=list)
    cache_hit: bool = False
    error_message: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert configuration and data"""
    alert_id: str
    alert_type: str
    threshold_value: float
    current_value: float
    database_type: DatabaseType
    query_pattern: Optional[str] = None
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    severity: AuditLevel = AuditLevel.WARNING
    description: str = ""
    suggested_actions: List[str] = field(default_factory=list)

@dataclass
class QueryPattern:
    """Identified query pattern for analytics"""
    pattern_id: str
    pattern_signature: str
    query_template: str
    occurrence_count: int
    avg_execution_time: float
    total_execution_time: float
    databases_used: Set[DatabaseType]
    first_seen: datetime
    last_seen: datetime
    optimization_opportunities: List[str] = field(default_factory=list)

class QueryAuditLogger:
    """
    Comprehensive Query Audit Logging System
    Handles structured logging, performance tracking, and compliance auditing
    """
    
    def __init__(self, log_directory: str = "logs/query_audit", 
                 max_log_size_mb: int = 100, retention_days: int = 90):
        
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.retention_days = retention_days
        
        # Audit storage
        self.audit_entries: deque = deque(maxlen=10000)  # In-memory buffer
        self.current_log_file = None
        self.current_log_size = 0
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.slow_query_threshold = 5.0  # seconds
        self.memory_threshold_mb = 500
        self.cpu_threshold_percent = 80
        
        # Alert system
        self.alert_callbacks: List[callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        
        # Pattern analysis
        self.query_patterns: Dict[str, QueryPattern] = {}
        self.pattern_analysis_enabled = True
        
        # Compliance and security
        self.sensitive_data_patterns = [
            r'password', r'secret', r'token', r'key', r'credential'
        ]
        
        # Background tasks
        self.cleanup_task = None
        self.analysis_task = None
        
        # Initialize structured logger
        self._setup_structured_logging()
        
        logger.info("Query Audit Logger initialized", 
                   log_directory=str(self.log_directory),
                   retention_days=retention_days)
    
    def _setup_structured_logging(self):
        """Setup structured logging configuration"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    async def log_query_execution(self, query: str, result: QueryResult, 
                                metrics: QueryMetrics, context: Dict[str, Any] = None) -> str:
        """Log query execution with full audit trail"""
        
        query_id = self._generate_query_id(query, metrics.timestamp)
        
        # Categorize query
        query_category = self._categorize_query(query)
        
        # Create audit entry
        audit_entry = QueryAuditEntry(
            query_id=query_id,
            timestamp=metrics.timestamp,
            database_type=metrics.database_type,
            query_text=self._sanitize_query(query),
            query_hash=metrics.query_hash,
            query_category=query_category,
            execution_time=metrics.execution_time,
            rows_affected=metrics.rows_processed,
            memory_usage=metrics.memory_usage,
            cpu_usage=metrics.cpu_usage,
            optimization_applied=metrics.optimization_applied,
            cache_hit=metrics.cache_hit,
            user_id=context.get('user_id') if context else None,
            session_id=context.get('session_id') if context else None,
            application_name=context.get('application_name', 'prometheus-ai') if context else 'prometheus-ai',
            client_ip=context.get('client_ip') if context else None,
            parameters=context.get('parameters') if context else None,
            metadata=context.get('metadata', {}) if context else {}
        )
        
        # Add to buffer
        self.audit_entries.append(audit_entry)
        
        # Write to persistent storage
        await self._write_audit_entry(audit_entry)
        
        # Update performance tracking
        self._update_performance_metrics(audit_entry)
        
        # Check for alerts
        await self._check_performance_alerts(audit_entry)
        
        # Update query patterns
        if self.pattern_analysis_enabled:
            self._update_query_patterns(audit_entry)
        
        # Log structured entry
        logger.info("Query executed",
                   query_id=query_id,
                   database=metrics.database_type.value,
                   execution_time=metrics.execution_time,
                   rows_processed=metrics.rows_processed,
                   cache_hit=metrics.cache_hit,
                   optimizations=len(metrics.optimization_applied))
        
        return query_id
    
    async def log_query_error(self, query: str, error: Exception, 
                            database_type: DatabaseType, context: Dict[str, Any] = None) -> str:
        """Log query execution error"""
        
        query_id = self._generate_query_id(query, datetime.utcnow())
        
        audit_entry = QueryAuditEntry(
            query_id=query_id,
            timestamp=datetime.utcnow(),
            database_type=database_type,
            query_text=self._sanitize_query(query),
            query_hash=hashlib.md5(query.encode()).hexdigest(),
            query_category=self._categorize_query(query),
            execution_time=0.0,
            rows_affected=0,
            memory_usage=0.0,
            cpu_usage=0.0,
            error_message=str(error),
            user_id=context.get('user_id') if context else None,
            session_id=context.get('session_id') if context else None,
            application_name=context.get('application_name', 'prometheus-ai') if context else 'prometheus-ai',
            client_ip=context.get('client_ip') if context else None,
            metadata=context.get('metadata', {}) if context else {}
        )
        
        self.audit_entries.append(audit_entry)
        await self._write_audit_entry(audit_entry)
        
        # Log error
        logger.error("Query execution failed",
                    query_id=query_id,
                    database=database_type.value,
                    error=str(error),
                    query_hash=audit_entry.query_hash)
        
        return query_id
    
    def _generate_query_id(self, query: str, timestamp: datetime) -> str:
        """Generate unique query ID"""
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        return f"q_{timestamp_str}_{query_hash}"
    
    def _categorize_query(self, query: str) -> QueryCategory:
        """Categorize query for analytics"""
        query_lower = query.lower().strip()
        
        # Administrative queries
        if any(admin_keyword in query_lower for admin_keyword in 
               ['create', 'drop', 'alter', 'grant', 'revoke', 'vacuum', 'analyze']):
            return QueryCategory.ADMINISTRATIVE
        
        # Write operations
        if any(write_keyword in query_lower for write_keyword in 
               ['insert', 'update', 'delete', 'merge', 'upsert']):
            return QueryCategory.WRITE_OPERATION
        
        # Real-time queries (based on patterns)
        if any(realtime_pattern in query_lower for realtime_pattern in 
               ['real_time', 'streaming', 'live', 'current', 'now()']):
            return QueryCategory.REAL_TIME
        
        # Analytical queries
        if any(analytical_pattern in query_lower for analytical_pattern in 
               ['group by', 'having', 'window', 'partition', 'aggregate', 'sum(', 'avg(', 'count(']):
            return QueryCategory.ANALYTICAL
        
        # Batch processing
        if any(batch_pattern in query_lower for batch_pattern in 
               ['batch', 'bulk', 'import', 'export', 'etl']):
            return QueryCategory.BATCH_PROCESSING
        
        # Default to read-only
        return QueryCategory.READ_ONLY
    
    def _sanitize_query(self, query: str) -> str:
        """Sanitize query text for logging (remove sensitive data)"""
        sanitized = query
        
        # Replace potential sensitive data patterns
        import re
        for pattern in self.sensitive_data_patterns:
            # Replace values after sensitive keywords
            sanitized = re.sub(
                rf"{pattern}\s*=\s*['\"]([^'\"]*)['\"]",
                rf"{pattern} = '***'",
                sanitized,
                flags=re.IGNORECASE
            )
        
        return sanitized
    
    async def _write_audit_entry(self, entry: QueryAuditEntry):
        """Write audit entry to persistent storage"""
        try:
            # Rotate log file if needed
            if self._should_rotate_log():
                await self._rotate_log_file()
            
            # Prepare log entry
            log_data = {
                'query_id': entry.query_id,
                'timestamp': entry.timestamp.isoformat(),
                'database_type': entry.database_type.value,
                'query_hash': entry.query_hash,
                'query_category': entry.query_category.value,
                'execution_time': entry.execution_time,
                'rows_affected': entry.rows_affected,
                'memory_usage': entry.memory_usage,
                'cpu_usage': entry.cpu_usage,
                'cache_hit': entry.cache_hit,
                'optimization_applied': [opt.value for opt in entry.optimization_applied],
                'user_id': entry.user_id,
                'session_id': entry.session_id,
                'application_name': entry.application_name,
                'client_ip': entry.client_ip,
                'error_message': entry.error_message,
                'warning_messages': entry.warning_messages,
                'metadata': entry.metadata
            }
            
            # Write to current log file
            if not self.current_log_file:
                await self._create_new_log_file()
            
            log_line = json.dumps(log_data) + '\n'
            
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)
            
            self.current_log_size += len(log_line.encode('utf-8'))
            
        except Exception as e:
            logger.error("Failed to write audit entry", error=str(e))
    
    def _should_rotate_log(self) -> bool:
        """Check if log file should be rotated"""
        return (self.current_log_file and 
                self.current_log_size >= self.max_log_size_bytes)
    
    async def _rotate_log_file(self):
        """Rotate current log file"""
        if self.current_log_file:
            # Compress old log file
            compressed_file = f"{self.current_log_file}.gz"
            
            with open(self.current_log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            os.remove(self.current_log_file)
            
            logger.info("Log file rotated", 
                       old_file=str(self.current_log_file),
                       compressed_file=str(compressed_file))
        
        await self._create_new_log_file()
    
    async def _create_new_log_file(self):
        """Create new log file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_directory / f"query_audit_{timestamp}.jsonl"
        self.current_log_size = 0
        
        logger.info("Created new log file", file=str(self.current_log_file))
    
    def _update_performance_metrics(self, entry: QueryAuditEntry):
        """Update performance metrics for monitoring"""
        db_key = entry.database_type.value
        
        # Track execution times
        self.performance_metrics[f"{db_key}_execution_time"].append(entry.execution_time)
        
        # Track memory usage
        if entry.memory_usage > 0:
            self.performance_metrics[f"{db_key}_memory_usage"].append(entry.memory_usage)
        
        # Track CPU usage
        if entry.cpu_usage > 0:
            self.performance_metrics[f"{db_key}_cpu_usage"].append(entry.cpu_usage)
        
        # Keep only recent metrics (last 1000 entries per metric)
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > 1000:
                self.performance_metrics[key] = self.performance_metrics[key][-1000:]
    
    async def _check_performance_alerts(self, entry: QueryAuditEntry):
        """Check for performance alerts and trigger notifications"""
        alerts = []
        
        # Slow query alert
        if entry.execution_time > self.slow_query_threshold:
            alert = PerformanceAlert(
                alert_id=f"slow_query_{entry.query_id}",
                alert_type="slow_query",
                threshold_value=self.slow_query_threshold,
                current_value=entry.execution_time,
                database_type=entry.database_type,
                query_pattern=entry.query_hash,
                severity=AuditLevel.WARNING if entry.execution_time < 10 else AuditLevel.ERROR,
                description=f"Query execution time ({entry.execution_time:.2f}s) exceeds threshold ({self.slow_query_threshold}s)",
                suggested_actions=[
                    "Review query execution plan",
                    "Check for missing indexes",
                    "Consider query optimization",
                    "Verify database performance"
                ]
            )
            alerts.append(alert)
        
        # High memory usage alert
        if entry.memory_usage > self.memory_threshold_mb:
            alert = PerformanceAlert(
                alert_id=f"high_memory_{entry.query_id}",
                alert_type="high_memory_usage",
                threshold_value=self.memory_threshold_mb,
                current_value=entry.memory_usage,
                database_type=entry.database_type,
                severity=AuditLevel.WARNING,
                description=f"Query memory usage ({entry.memory_usage:.1f}MB) exceeds threshold ({self.memory_threshold_mb}MB)",
                suggested_actions=[
                    "Optimize query to reduce memory usage",
                    "Consider result set pagination",
                    "Review data processing logic"
                ]
            )
            alerts.append(alert)
        
        # High CPU usage alert
        if entry.cpu_usage > self.cpu_threshold_percent:
            alert = PerformanceAlert(
                alert_id=f"high_cpu_{entry.query_id}",
                alert_type="high_cpu_usage",
                threshold_value=self.cpu_threshold_percent,
                current_value=entry.cpu_usage,
                database_type=entry.database_type,
                severity=AuditLevel.WARNING,
                description=f"Query CPU usage ({entry.cpu_usage:.1f}%) exceeds threshold ({self.cpu_threshold_percent}%)",
                suggested_actions=[
                    "Optimize query complexity",
                    "Consider parallel processing",
                    "Review system resources"
                ]
            )
            alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            self.alert_history.append(alert)
            await self._trigger_alert(alert)
    
    async def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger performance alert notifications"""
        logger.warning("Performance alert triggered",
                      alert_id=alert.alert_id,
                      alert_type=alert.alert_type,
                      database=alert.database_type.value,
                      threshold=alert.threshold_value,
                      current_value=alert.current_value,
                      severity=alert.severity.value)
        
        # Notify registered callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error("Alert callback failed", error=str(e))
    
    def _update_query_patterns(self, entry: QueryAuditEntry):
        """Update query pattern analysis"""
        try:
            # Generate pattern signature (simplified)
            pattern_signature = self._generate_pattern_signature(entry.query_text)
            
            if pattern_signature in self.query_patterns:
                # Update existing pattern
                pattern = self.query_patterns[pattern_signature]
                pattern.occurrence_count += 1
                pattern.total_execution_time += entry.execution_time
                pattern.avg_execution_time = pattern.total_execution_time / pattern.occurrence_count
                pattern.databases_used.add(entry.database_type)
                pattern.last_seen = entry.timestamp
                
                # Check for optimization opportunities
                if entry.execution_time > pattern.avg_execution_time * 2:
                    if "performance_degradation" not in pattern.optimization_opportunities:
                        pattern.optimization_opportunities.append("performance_degradation")
                
            else:
                # Create new pattern
                pattern = QueryPattern(
                    pattern_id=f"pattern_{len(self.query_patterns)}",
                    pattern_signature=pattern_signature,
                    query_template=self._create_query_template(entry.query_text),
                    occurrence_count=1,
                    avg_execution_time=entry.execution_time,
                    total_execution_time=entry.execution_time,
                    databases_used={entry.database_type},
                    first_seen=entry.timestamp,
                    last_seen=entry.timestamp
                )
                
                self.query_patterns[pattern_signature] = pattern
                
        except Exception as e:
            logger.warning("Pattern analysis failed", error=str(e))
    
    def _generate_pattern_signature(self, query: str) -> str:
        """Generate pattern signature for query grouping"""
        # Normalize query for pattern matching
        normalized = query.lower().strip()
        
        # Replace literals with placeholders
        import re
        
        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        
        # Replace numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        # Replace IN clauses with placeholders
        normalized = re.sub(r'in\s*\([^)]+\)', 'in (?)', normalized)
        
        # Generate hash of normalized query
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _create_query_template(self, query: str) -> str:
        """Create query template for pattern analysis"""
        # Similar to pattern signature but more readable
        template = query
        
        import re
        
        # Replace string literals
        template = re.sub(r"'[^']*'", "'<string>'", template)
        
        # Replace numeric literals
        template = re.sub(r'\b\d+\b', '<number>', template)
        
        return template
    
    async def get_audit_report(self, start_time: datetime = None, 
                             end_time: datetime = None,
                             database_type: DatabaseType = None) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=24)
        
        if end_time is None:
            end_time = datetime.utcnow()
        
        # Filter audit entries
        filtered_entries = [
            entry for entry in self.audit_entries
            if start_time <= entry.timestamp <= end_time and
            (database_type is None or entry.database_type == database_type)
        ]
        
        if not filtered_entries:
            return {'message': 'No audit entries found for the specified criteria'}
        
        # Calculate statistics
        total_queries = len(filtered_entries)
        successful_queries = len([e for e in filtered_entries if e.error_message is None])
        failed_queries = total_queries - successful_queries
        
        execution_times = [e.execution_time for e in filtered_entries if e.execution_time > 0]
        
        # Database statistics
        db_stats = {}
        for db_type in DatabaseType:
            db_entries = [e for e in filtered_entries if e.database_type == db_type]
            if db_entries:
                db_execution_times = [e.execution_time for e in db_entries if e.execution_time > 0]
                db_stats[db_type.value] = {
                    'query_count': len(db_entries),
                    'success_rate': len([e for e in db_entries if e.error_message is None]) / len(db_entries),
                    'avg_execution_time': statistics.mean(db_execution_times) if db_execution_times else 0,
                    'max_execution_time': max(db_execution_times) if db_execution_times else 0,
                    'cache_hit_rate': len([e for e in db_entries if e.cache_hit]) / len(db_entries)
                }
        
        # Query category statistics
        category_stats = {}
        for category in QueryCategory:
            category_entries = [e for e in filtered_entries if e.query_category == category]
            if category_entries:
                category_stats[category.value] = len(category_entries)
        
        # Top slow queries
        slow_queries = sorted(
            [e for e in filtered_entries if e.execution_time > self.slow_query_threshold],
            key=lambda x: x.execution_time,
            reverse=True
        )[:10]
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if start_time <= alert.triggered_at <= end_time
        ]
        
        return {
            'report_period': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'summary': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'failed_queries': failed_queries,
                'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
                'avg_execution_time': statistics.mean(execution_times) if execution_times else 0,
                'median_execution_time': statistics.median(execution_times) if execution_times else 0,
                'max_execution_time': max(execution_times) if execution_times else 0
            },
            'database_statistics': db_stats,
            'query_category_statistics': category_stats,
            'slow_queries': [
                {
                    'query_id': q.query_id,
                    'execution_time': q.execution_time,
                    'database_type': q.database_type.value,
                    'timestamp': q.timestamp.isoformat(),
                    'query_hash': q.query_hash
                }
                for q in slow_queries
            ],
            'recent_alerts': [
                {
                    'alert_id': a.alert_id,
                    'alert_type': a.alert_type,
                    'severity': a.severity.value,
                    'triggered_at': a.triggered_at.isoformat(),
                    'database_type': a.database_type.value,
                    'description': a.description
                }
                for a in recent_alerts
            ],
            'query_patterns': {
                pattern_id: {
                    'occurrence_count': pattern.occurrence_count,
                    'avg_execution_time': pattern.avg_execution_time,
                    'databases_used': [db.value for db in pattern.databases_used],
                    'optimization_opportunities': pattern.optimization_opportunities
                }
                for pattern_id, pattern in list(self.query_patterns.items())[:20]  # Top 20 patterns
            }
        }
    
    def register_alert_callback(self, callback: callable):
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_background_tasks(self):
        """Start background maintenance tasks"""
        # Cleanup old log files
        self.cleanup_task = asyncio.create_task(self._cleanup_old_logs())
        
        # Pattern analysis task
        self.analysis_task = asyncio.create_task(self._periodic_pattern_analysis())
        
        logger.info("Background tasks started")
    
    async def _cleanup_old_logs(self):
        """Cleanup old log files based on retention policy"""
        while True:
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                
                for log_file in self.log_directory.glob("query_audit_*.jsonl*"):
                    file_stat = log_file.stat()
                    file_date = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        logger.info("Deleted old log file", file=str(log_file))
                
                # Sleep for 24 hours
                await asyncio.sleep(24 * 3600)
                
            except Exception as e:
                logger.error("Log cleanup failed", error=str(e))
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    async def _periodic_pattern_analysis(self):
        """Perform periodic pattern analysis and optimization suggestions"""
        while True:
            try:
                # Analyze patterns every hour
                await asyncio.sleep(3600)
                
                # Find patterns with optimization opportunities
                for pattern in self.query_patterns.values():
                    if pattern.occurrence_count > 10:  # Frequent patterns
                        if pattern.avg_execution_time > 2.0:  # Slow patterns
                            if "frequent_slow_pattern" not in pattern.optimization_opportunities:
                                pattern.optimization_opportunities.append("frequent_slow_pattern")
                                
                                logger.info("Optimization opportunity identified",
                                          pattern_id=pattern.pattern_id,
                                          occurrence_count=pattern.occurrence_count,
                                          avg_execution_time=pattern.avg_execution_time)
                
            except Exception as e:
                logger.error("Pattern analysis failed", error=str(e))
    
    async def cleanup(self):
        """Cleanup resources"""
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        if self.analysis_task:
            self.analysis_task.cancel()
        
        # Final log rotation if needed
        if self.current_log_file and self.current_log_size > 0:
            await self._rotate_log_file()
        
        logger.info("Query Audit Logger cleanup complete")

# Global audit logger instance
_audit_logger: Optional[QueryAuditLogger] = None

def get_audit_logger() -> QueryAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = QueryAuditLogger()
    
    return _audit_logger

async def cleanup_audit_logger():
    """Cleanup global audit logger"""
    global _audit_logger
    
    if _audit_logger:
        await _audit_logger.cleanup()
        _audit_logger = None