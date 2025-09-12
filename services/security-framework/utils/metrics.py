"""
Security Metrics Collection and Monitoring
Implements comprehensive security metrics with geological context awareness
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import aioredis
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class SecurityMetric:
    """Security metric with geological context"""
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    geological_context: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThreatMetrics:
    """Threat detection and response metrics"""
    threats_detected: int = 0
    threats_mitigated: int = 0
    false_positives: int = 0
    mean_detection_time: float = 0.0
    mean_response_time: float = 0.0
    threat_types: Dict[str, int] = field(default_factory=dict)
    geological_threat_distribution: Dict[str, int] = field(default_factory=dict)

@dataclass
class AccessMetrics:
    """Access control and authorization metrics"""
    total_requests: int = 0
    successful_authentications: int = 0
    failed_authentications: int = 0
    access_granted: int = 0
    access_denied: int = 0
    mfa_challenges: int = 0
    mfa_successes: int = 0
    geological_access_patterns: Dict[str, int] = field(default_factory=dict)

@dataclass
class ComplianceMetrics:
    """Compliance and audit metrics"""
    compliance_score: float = 0.0
    policy_violations: int = 0
    audit_events: int = 0
    data_classification_compliance: Dict[str, float] = field(default_factory=dict)
    geological_compliance_scores: Dict[str, float] = field(default_factory=dict)
    retention_compliance: float = 0.0

class SecurityMetrics:
    """
    Comprehensive security metrics collection and monitoring
    Implements real-time metrics with geological domain awareness
    """
    
    def __init__(self):
        self.redis_client = None
        
        # Metrics storage
        self.threat_metrics = ThreatMetrics()
        self.access_metrics = AccessMetrics()
        self.compliance_metrics = ComplianceMetrics()
        
        # Time-series data (last 24 hours)
        self.metrics_history = defaultdict(lambda: deque(maxlen=1440))  # 1 minute intervals
        
        # Performance tracking
        self.request_times = deque(maxlen=1000)
        self.error_rates = defaultdict(lambda: deque(maxlen=100))
        
        # Geological context metrics
        self.geological_metrics = {
            'volcanology': defaultdict(int),
            'seismology': defaultdict(int),
            'geochemistry': defaultdict(int),
            'structural_geology': defaultdict(int),
            'petrology': defaultdict(int),
            'mineralogy': defaultdict(int)
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'failed_auth_rate': 0.1,  # 10% failure rate
            'threat_detection_rate': 0.05,  # 5% threat rate
            'policy_violation_rate': 0.02,  # 2% violation rate
            'response_time_p95': 2.0,  # 2 seconds
            'error_rate': 0.01  # 1% error rate
        }
        
    async def initialize(self):
        """Initialize metrics collection system"""
        logger.info("Initializing Security Metrics System")
        
        # Initialize Redis for metrics storage
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/6",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Start background metrics collection
        asyncio.create_task(self._periodic_metrics_collection())
        asyncio.create_task(self._metrics_aggregation())
        
        logger.info("Security Metrics System initialized successfully")
        
    async def record_request(self, request, response, processing_time: float):
        """Record request metrics with geological context"""
        
        # Record processing time
        self.request_times.append(processing_time)
        
        # Extract geological context
        geological_context = request.headers.get('X-Geological-Context')
        geological_domain = None
        
        if geological_context:
            try:
                context_data = json.loads(geological_context)
                geological_domain = context_data.get('primary_domain')
            except:
                pass
                
        # Record by status code
        status_code = response.status_code
        if status_code >= 400:
            self.error_rates[status_code].append(time.time())
            
        # Record geological metrics
        if geological_domain and geological_domain in self.geological_metrics:
            self.geological_metrics[geological_domain]['requests'] += 1
            self.geological_metrics[geological_domain]['total_time'] += processing_time
            
        # Store in Redis for real-time monitoring
        await self._store_request_metric(request, response, processing_time, geological_domain)
        
    async def record_authentication_event(
        self, 
        event_type: str, 
        success: bool,
        user_id: str = None,
        geological_context: str = None
    ):
        """Record authentication metrics"""
        
        if event_type == 'login':
            if success:
                self.access_metrics.successful_authentications += 1
            else:
                self.access_metrics.failed_authentications += 1
                
        elif event_type == 'mfa_challenge':
            self.access_metrics.mfa_challenges += 1
            
        elif event_type == 'mfa_verification':
            if success:
                self.access_metrics.mfa_successes += 1
                
        # Record geological context
        if geological_context and geological_context in self.geological_metrics:
            auth_key = f'auth_{event_type}_{"success" if success else "failure"}'
            self.geological_metrics[geological_context][auth_key] += 1
            
        # Store in time series
        metric = SecurityMetric(
            metric_name=f'authentication_{event_type}',
            value=1.0,
            unit='count',
            timestamp=datetime.utcnow(),
            geological_context=geological_context,
            tags={'success': str(success), 'user_id': user_id or 'unknown'}
        )
        
        await self._store_metric(metric)
        
    async def record_access_event(
        self, 
        event_type: str, 
        success: bool,
        resource_id: str = None,
        access_level: str = None,
        geological_context: str = None
    ):
        """Record access control metrics"""
        
        if event_type == 'access_request':
            if success:
                self.access_metrics.access_granted += 1
            else:
                self.access_metrics.access_denied += 1
                
        # Record geological access patterns
        if geological_context and geological_context in self.geological_metrics:
            access_key = f'access_{event_type}_{"granted" if success else "denied"}'
            self.geological_metrics[geological_context][access_key] += 1
            
            if access_level:
                level_key = f'access_level_{access_level}'
                self.geological_metrics[geological_context][level_key] += 1
                
        # Store metric
        metric = SecurityMetric(
            metric_name=f'access_{event_type}',
            value=1.0,
            unit='count',
            timestamp=datetime.utcnow(),
            geological_context=geological_context,
            tags={
                'success': str(success), 
                'resource_id': resource_id or 'unknown',
                'access_level': access_level or 'unknown'
            }
        )
        
        await self._store_metric(metric)
        
    async def record_threat_event(
        self, 
        threat_type: str, 
        threat_level: str,
        detection_time: float = None,
        mitigation_time: float = None,
        geological_context: str = None
    ):
        """Record threat detection and response metrics"""
        
        self.threat_metrics.threats_detected += 1
        
        if threat_type in self.threat_metrics.threat_types:
            self.threat_metrics.threat_types[threat_type] += 1
        else:
            self.threat_metrics.threat_types[threat_type] = 1
            
        # Record geological threat distribution
        if geological_context:
            if geological_context in self.threat_metrics.geological_threat_distribution:
                self.threat_metrics.geological_threat_distribution[geological_context] += 1
            else:
                self.threat_metrics.geological_threat_distribution[geological_context] = 1
                
        # Update timing metrics
        if detection_time:
            current_mean = self.threat_metrics.mean_detection_time
            count = self.threat_metrics.threats_detected
            self.threat_metrics.mean_detection_time = (
                (current_mean * (count - 1) + detection_time) / count
            )
            
        if mitigation_time:
            self.threat_metrics.threats_mitigated += 1
            current_mean = self.threat_metrics.mean_response_time
            count = self.threat_metrics.threats_mitigated
            self.threat_metrics.mean_response_time = (
                (current_mean * (count - 1) + mitigation_time) / count
            )
            
        # Store metric
        metric = SecurityMetric(
            metric_name='threat_detection',
            value=1.0,
            unit='count',
            timestamp=datetime.utcnow(),
            geological_context=geological_context,
            tags={
                'threat_type': threat_type,
                'threat_level': threat_level
            },
            metadata={
                'detection_time': detection_time,
                'mitigation_time': mitigation_time
            }
        )
        
        await self._store_metric(metric)
        
    async def record_compliance_event(
        self, 
        framework: str, 
        event_type: str,
        compliance_score: float = None,
        geological_context: str = None
    ):
        """Record compliance metrics"""
        
        if event_type == 'violation':
            self.compliance_metrics.policy_violations += 1
            
        if compliance_score is not None:
            if framework in self.compliance_metrics.data_classification_compliance:
                # Update running average
                current_score = self.compliance_metrics.data_classification_compliance[framework]
                self.compliance_metrics.data_classification_compliance[framework] = (
                    (current_score + compliance_score) / 2
                )
            else:
                self.compliance_metrics.data_classification_compliance[framework] = compliance_score
                
        # Record geological compliance
        if geological_context and compliance_score is not None:
            if geological_context in self.compliance_metrics.geological_compliance_scores:
                current_score = self.compliance_metrics.geological_compliance_scores[geological_context]
                self.compliance_metrics.geological_compliance_scores[geological_context] = (
                    (current_score + compliance_score) / 2
                )
            else:
                self.compliance_metrics.geological_compliance_scores[geological_context] = compliance_score
                
        # Store metric
        metric = SecurityMetric(
            metric_name=f'compliance_{event_type}',
            value=compliance_score or 1.0,
            unit='score' if compliance_score else 'count',
            timestamp=datetime.utcnow(),
            geological_context=geological_context,
            tags={'framework': framework}
        )
        
        await self._store_metric(metric)
        
    async def get_current_metrics(self) -> Dict:
        """Get current security metrics summary"""
        
        # Calculate derived metrics
        total_auth_attempts = (
            self.access_metrics.successful_authentications + 
            self.access_metrics.failed_authentications
        )
        
        auth_failure_rate = (
            self.access_metrics.failed_authentications / max(total_auth_attempts, 1)
        )
        
        mfa_success_rate = (
            self.access_metrics.mfa_successes / max(self.access_metrics.mfa_challenges, 1)
        )
        
        # Calculate performance metrics
        avg_response_time = sum(self.request_times) / max(len(self.request_times), 1)
        p95_response_time = self._calculate_percentile(list(self.request_times), 95)
        
        # Calculate error rates
        current_time = time.time()
        recent_errors = sum(
            len([t for t in times if current_time - t < 300])  # Last 5 minutes
            for times in self.error_rates.values()
        )
        
        total_recent_requests = len([t for t in self.request_times if current_time - t < 300])
        error_rate = recent_errors / max(total_recent_requests, 1)
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'threat_metrics': {
                'threats_detected': self.threat_metrics.threats_detected,
                'threats_mitigated': self.threat_metrics.threats_mitigated,
                'false_positives': self.threat_metrics.false_positives,
                'mean_detection_time': self.threat_metrics.mean_detection_time,
                'mean_response_time': self.threat_metrics.mean_response_time,
                'threat_types': dict(self.threat_metrics.threat_types),
                'geological_distribution': dict(self.threat_metrics.geological_threat_distribution)
            },
            'access_metrics': {
                'total_auth_attempts': total_auth_attempts,
                'successful_authentications': self.access_metrics.successful_authentications,
                'failed_authentications': self.access_metrics.failed_authentications,
                'auth_failure_rate': auth_failure_rate,
                'access_granted': self.access_metrics.access_granted,
                'access_denied': self.access_metrics.access_denied,
                'mfa_challenges': self.access_metrics.mfa_challenges,
                'mfa_successes': self.access_metrics.mfa_successes,
                'mfa_success_rate': mfa_success_rate
            },
            'performance_metrics': {
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'error_rate': error_rate,
                'total_requests': len(self.request_times)
            },
            'compliance_metrics': {
                'compliance_score': self.compliance_metrics.compliance_score,
                'policy_violations': self.compliance_metrics.policy_violations,
                'audit_events': self.compliance_metrics.audit_events,
                'framework_compliance': dict(self.compliance_metrics.data_classification_compliance),
                'geological_compliance': dict(self.compliance_metrics.geological_compliance_scores)
            },
            'geological_metrics': {
                domain: dict(metrics) for domain, metrics in self.geological_metrics.items()
            },
            'alerts': await self._check_alert_thresholds({
                'auth_failure_rate': auth_failure_rate,
                'error_rate': error_rate,
                'p95_response_time': p95_response_time
            })
        }
        
    async def _store_metric(self, metric: SecurityMetric):
        """Store metric in Redis time series"""
        
        metric_key = f"security_metric:{metric.metric_name}"
        
        metric_data = {
            'value': metric.value,
            'unit': metric.unit,
            'timestamp': metric.timestamp.isoformat(),
            'geological_context': metric.geological_context,
            'tags': metric.tags,
            'metadata': metric.metadata
        }
        
        # Store in Redis with expiration
        await self.redis_client.zadd(
            metric_key,
            {json.dumps(metric_data): metric.timestamp.timestamp()}
        )
        
        # Keep only last 24 hours of data
        cutoff_time = (datetime.utcnow() - timedelta(hours=24)).timestamp()
        await self.redis_client.zremrangebyscore(metric_key, 0, cutoff_time)
        
    async def _store_request_metric(
        self, 
        request, 
        response, 
        processing_time: float,
        geological_domain: str = None
    ):
        """Store request-specific metrics"""
        
        request_data = {
            'method': request.method,
            'path': str(request.url.path),
            'status_code': response.status_code,
            'processing_time': processing_time,
            'geological_domain': geological_domain,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis_client.lpush('request_metrics', json.dumps(request_data))
        await self.redis_client.ltrim('request_metrics', 0, 9999)  # Keep last 10k requests
        
    def _calculate_percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
            
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
        
    async def _check_alert_thresholds(self, current_metrics: Dict) -> List[Dict]:
        """Check if any metrics exceed alert thresholds"""
        
        alerts = []
        
        for metric_name, threshold in self.alert_thresholds.items():
            current_value = current_metrics.get(metric_name, 0)
            
            if current_value > threshold:
                alerts.append({
                    'metric': metric_name,
                    'current_value': current_value,
                    'threshold': threshold,
                    'severity': 'high' if current_value > threshold * 2 else 'medium',
                    'timestamp': datetime.utcnow().isoformat()
                })
                
        return alerts
        
    async def _periodic_metrics_collection(self):
        """Periodic collection of system metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Collect system-level metrics
                await self._collect_system_metrics()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
                
    async def _collect_system_metrics(self):
        """Collect system-level security metrics"""
        
        # This would collect system metrics like:
        # - Active sessions
        # - Resource utilization
        # - Database connection health
        # - Cache hit rates
        # etc.
        
        pass
        
    async def _metrics_aggregation(self):
        """Aggregate metrics for reporting"""
        while True:
            try:
                await asyncio.sleep(300)  # Aggregate every 5 minutes
                
                # Perform metrics aggregation
                await self._aggregate_geological_metrics()
                await self._calculate_compliance_scores()
                
            except Exception as e:
                logger.error(f"Metrics aggregation error: {str(e)}")
                
    async def _aggregate_geological_metrics(self):
        """Aggregate metrics by geological domain"""
        
        for domain, metrics in self.geological_metrics.items():
            if metrics['requests'] > 0:
                avg_response_time = metrics['total_time'] / metrics['requests']
                
                # Store aggregated metric
                metric = SecurityMetric(
                    metric_name=f'geological_avg_response_time',
                    value=avg_response_time,
                    unit='seconds',
                    timestamp=datetime.utcnow(),
                    geological_context=domain
                )
                
                await self._store_metric(metric)
                
    async def _calculate_compliance_scores(self):
        """Calculate overall compliance scores"""
        
        if self.compliance_metrics.data_classification_compliance:
            scores = list(self.compliance_metrics.data_classification_compliance.values())
            self.compliance_metrics.compliance_score = sum(scores) / len(scores)
            
        # Store compliance score metric
        metric = SecurityMetric(
            metric_name='overall_compliance_score',
            value=self.compliance_metrics.compliance_score,
            unit='score',
            timestamp=datetime.utcnow()
        )
        
        await self._store_metric(metric)