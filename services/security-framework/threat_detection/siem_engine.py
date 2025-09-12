"""
Advanced Security Information and Event Management (SIEM) Engine
Provides comprehensive threat detection, correlation, and response capabilities
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import pandas as pd
import hashlib
import uuid
from collections import defaultdict, deque
import aioredis
import asyncpg
from elasticsearch import AsyncElasticsearch
import aiokafka
from cryptography.fernet import Fernet

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    NETWORK_ACTIVITY = "network_activity"
    SYSTEM_ACTIVITY = "system_activity"
    GEOLOGICAL_WORKFLOW = "geological_workflow"
    API_ACTIVITY = "api_activity"
    DATABASE_ACTIVITY = "database_activity"

class IncidentStatus(Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    CLOSED = "closed"

@dataclass
class SecurityEvent:
    id: str
    timestamp: datetime
    event_type: EventType
    source_ip: str
    user_id: Optional[str]
    resource: str
    action: str
    outcome: str
    severity: int
    raw_data: Dict[str, Any]
    geological_context: Optional[Dict[str, Any]] = None
    threat_indicators: List[str] = None
    correlation_id: Optional[str] = None

@dataclass
class ThreatIndicator:
    id: str
    type: str
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_seen: datetime
    geological_relevance: bool = False
    context: Dict[str, Any] = None

@dataclass
class SecurityIncident:
    id: str
    title: str
    description: str
    threat_level: ThreatLevel
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    assigned_to: Optional[str]
    events: List[SecurityEvent]
    indicators: List[ThreatIndicator]
    geological_impact: Optional[Dict[str, Any]]
    response_actions: List[Dict[str, Any]]
    timeline: List[Dict[str, Any]]

class BehavioralAnalytics:
    """Advanced behavioral analytics for geological workflow anomaly detection"""
    
    def __init__(self):
        self.user_profiles = {}
        self.geological_patterns = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        
    async def build_user_profile(self, user_id: str, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Build comprehensive user behavioral profile"""
        profile = {
            'user_id': user_id,
            'typical_hours': self._analyze_access_patterns(events),
            'common_resources': self._analyze_resource_access(events),
            'geological_workflows': self._analyze_geological_patterns(events),
            'network_patterns': self._analyze_network_behavior(events),
            'api_usage_patterns': self._analyze_api_usage(events),
            'risk_score': self._calculate_risk_score(events)
        }
        
        self.user_profiles[user_id] = profile
        return profile
    
    def _analyze_access_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze temporal access patterns"""
        hours = [event.timestamp.hour for event in events]
        days = [event.timestamp.weekday() for event in events]
        
        return {
            'typical_hours': np.histogram(hours, bins=24)[0].tolist(),
            'typical_days': np.histogram(days, bins=7)[0].tolist(),
            'peak_hour': max(set(hours), key=hours.count) if hours else 9,
            'activity_variance': np.var(hours) if hours else 0
        }
    
    def _analyze_resource_access(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze resource access patterns"""
        resources = [event.resource for event in events]
        resource_counts = defaultdict(int)
        
        for resource in resources:
            resource_counts[resource] += 1
            
        return {
            'common_resources': dict(sorted(resource_counts.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]),
            'resource_diversity': len(set(resources)),
            'access_frequency': len(resources) / max(1, len(set(resources)))
        }
    
    def _analyze_geological_patterns(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze geological workflow patterns"""
        geological_events = [e for e in events if e.geological_context]
        
        if not geological_events:
            return {'has_geological_activity': False}
            
        workflows = []
        data_types = []
        analysis_types = []
        
        for event in geological_events:
            ctx = event.geological_context
            if ctx:
                workflows.append(ctx.get('workflow_type', 'unknown'))
                data_types.append(ctx.get('data_type', 'unknown'))
                analysis_types.append(ctx.get('analysis_type', 'unknown'))
        
        return {
            'has_geological_activity': True,
            'common_workflows': list(set(workflows)),
            'preferred_data_types': list(set(data_types)),
            'analysis_complexity': len(set(analysis_types)),
            'geological_expertise_level': self._assess_expertise_level(workflows, analysis_types)
        }
    
    def _analyze_network_behavior(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze network behavior patterns"""
        ips = [event.source_ip for event in events]
        ip_counts = defaultdict(int)
        
        for ip in ips:
            ip_counts[ip] += 1
            
        return {
            'common_ips': dict(sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'ip_diversity': len(set(ips)),
            'mobility_score': len(set(ips)) / max(1, len(ips))
        }
    
    def _analyze_api_usage(self, events: List[SecurityEvent]) -> Dict[str, Any]:
        """Analyze API usage patterns"""
        api_events = [e for e in events if e.event_type == EventType.API_ACTIVITY]
        
        if not api_events:
            return {'has_api_activity': False}
            
        endpoints = [event.resource for event in api_events]
        methods = [event.action for event in api_events]
        
        return {
            'has_api_activity': True,
            'common_endpoints': list(set(endpoints)),
            'preferred_methods': list(set(methods)),
            'api_usage_frequency': len(api_events) / max(1, len(events))
        }
    
    def _calculate_risk_score(self, events: List[SecurityEvent]) -> float:
        """Calculate user risk score based on behavior"""
        if not events:
            return 0.5
            
        # Factors contributing to risk score
        failed_auths = len([e for e in events if e.event_type == EventType.AUTHENTICATION and e.outcome == 'failure'])
        privilege_escalations = len([e for e in events if 'privilege' in e.action.lower()])
        unusual_hours = len([e for e in events if e.timestamp.hour < 6 or e.timestamp.hour > 22])
        high_severity_events = len([e for e in events if e.severity >= 7])
        
        total_events = len(events)
        
        risk_factors = {
            'failed_auth_rate': failed_auths / total_events,
            'privilege_escalation_rate': privilege_escalations / total_events,
            'unusual_hours_rate': unusual_hours / total_events,
            'high_severity_rate': high_severity_events / total_events
        }
        
        # Weighted risk calculation
        weights = {'failed_auth_rate': 0.3, 'privilege_escalation_rate': 0.4, 
                  'unusual_hours_rate': 0.2, 'high_severity_rate': 0.1}
        
        risk_score = sum(risk_factors[factor] * weights[factor] for factor in weights)
        return min(1.0, max(0.0, risk_score))
    
    def _assess_expertise_level(self, workflows: List[str], analysis_types: List[str]) -> str:
        """Assess geological expertise level based on workflow complexity"""
        advanced_workflows = ['3d_modeling', 'seismic_inversion', 'mantle_convection', 'pinn_modeling']
        advanced_analyses = ['physics_informed', 'multi_physics', 'uncertainty_quantification']
        
        advanced_workflow_count = sum(1 for w in workflows if w in advanced_workflows)
        advanced_analysis_count = sum(1 for a in analysis_types if a in advanced_analyses)
        
        total_advanced = advanced_workflow_count + advanced_analysis_count
        
        if total_advanced >= 5:
            return 'expert'
        elif total_advanced >= 3:
            return 'advanced'
        elif total_advanced >= 1:
            return 'intermediate'
        else:
            return 'beginner'
    
    async def detect_anomalies(self, user_id: str, recent_events: List[SecurityEvent]) -> List[Dict[str, Any]]:
        """Detect behavioral anomalies using machine learning"""
        if user_id not in self.user_profiles:
            return []
            
        profile = self.user_profiles[user_id]
        anomalies = []
        
        # Temporal anomalies
        current_hours = [event.timestamp.hour for event in recent_events]
        if current_hours:
            typical_hours = profile['typical_hours']['typical_hours']
            for hour in current_hours:
                if typical_hours[hour] < np.mean(typical_hours) * 0.1:  # Very unusual hour
                    anomalies.append({
                        'type': 'temporal_anomaly',
                        'description': f'Access at unusual hour: {hour}:00',
                        'severity': 'medium',
                        'confidence': 0.8
                    })
        
        # Resource access anomalies
        current_resources = [event.resource for event in recent_events]
        common_resources = set(profile['common_resources']['common_resources'].keys())
        
        for resource in current_resources:
            if resource not in common_resources:
                anomalies.append({
                    'type': 'resource_anomaly',
                    'description': f'Access to unusual resource: {resource}',
                    'severity': 'medium',
                    'confidence': 0.7
                })
        
        # Geological workflow anomalies
        if profile['geological_workflows']['has_geological_activity']:
            geological_events = [e for e in recent_events if e.geological_context]
            for event in geological_events:
                ctx = event.geological_context
                if ctx and ctx.get('workflow_type') not in profile['geological_workflows']['common_workflows']:
                    anomalies.append({
                        'type': 'geological_workflow_anomaly',
                        'description': f'Unusual geological workflow: {ctx.get("workflow_type")}',
                        'severity': 'high',
                        'confidence': 0.9
                    })
        
        return anomalies

class ThreatIntelligence:
    """Threat intelligence integration and analysis"""
    
    def __init__(self):
        self.indicators = {}
        self.feeds = {}
        self.geological_threats = {}
        
    async def load_threat_feeds(self) -> None:
        """Load threat intelligence feeds"""
        # Simulated threat intelligence feeds
        self.feeds = {
            'malicious_ips': await self._load_malicious_ips(),
            'suspicious_domains': await self._load_suspicious_domains(),
            'geological_attack_patterns': await self._load_geological_threats(),
            'api_abuse_patterns': await self._load_api_abuse_patterns()
        }
    
    async def _load_malicious_ips(self) -> List[str]:
        """Load known malicious IP addresses"""
        # In production, this would connect to threat intelligence feeds
        return [
            '192.168.1.100',  # Example malicious IP
            '10.0.0.50',      # Example suspicious IP
            '172.16.0.25'     # Example compromised IP
        ]
    
    async def _load_suspicious_domains(self) -> List[str]:
        """Load suspicious domains"""
        return [
            'malicious-geological-data.com',
            'fake-seismic-api.net',
            'geological-phishing.org'
        ]
    
    async def _load_geological_threats(self) -> Dict[str, Any]:
        """Load geological-specific threat patterns"""
        return {
            'data_exfiltration_patterns': [
                'bulk_seismic_download',
                'comprehensive_geological_survey_access',
                'proprietary_model_extraction'
            ],
            'insider_threat_indicators': [
                'after_hours_model_access',
                'unusual_data_export_volumes',
                'unauthorized_geological_database_queries'
            ],
            'api_abuse_patterns': [
                'rapid_geological_api_calls',
                'systematic_data_harvesting',
                'unauthorized_model_inference_requests'
            ]
        }
    
    async def _load_api_abuse_patterns(self) -> List[Dict[str, Any]]:
        """Load API abuse patterns"""
        return [
            {
                'pattern': 'rapid_requests',
                'threshold': 100,  # requests per minute
                'window': 60,      # seconds
                'severity': 'high'
            },
            {
                'pattern': 'systematic_enumeration',
                'indicators': ['sequential_ids', 'parameter_fuzzing'],
                'severity': 'medium'
            }
        ]
    
    async def check_indicators(self, event: SecurityEvent) -> List[ThreatIndicator]:
        """Check event against threat intelligence indicators"""
        indicators = []
        
        # Check IP reputation
        if event.source_ip in self.feeds.get('malicious_ips', []):
            indicators.append(ThreatIndicator(
                id=str(uuid.uuid4()),
                type='malicious_ip',
                value=event.source_ip,
                confidence=0.9,
                source='threat_intelligence',
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                geological_relevance=bool(event.geological_context)
            ))
        
        # Check for geological-specific threats
        if event.geological_context:
            geological_threats = self.feeds.get('geological_attack_patterns', {})
            
            workflow_type = event.geological_context.get('workflow_type', '')
            if workflow_type in geological_threats.get('data_exfiltration_patterns', []):
                indicators.append(ThreatIndicator(
                    id=str(uuid.uuid4()),
                    type='geological_data_exfiltration',
                    value=workflow_type,
                    confidence=0.8,
                    source='geological_threat_intelligence',
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    geological_relevance=True,
                    context={'workflow': workflow_type, 'risk': 'data_exfiltration'}
                ))
        
        return indicators

class SIEMEngine:
    """Main SIEM engine for threat detection and correlation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.behavioral_analytics = BehavioralAnalytics()
        self.threat_intelligence = ThreatIntelligence()
        self.event_buffer = deque(maxlen=10000)
        self.correlation_rules = []
        self.incidents = {}
        self.redis_client = None
        self.elasticsearch_client = None
        self.db_pool = None
        
    async def initialize(self) -> None:
        """Initialize SIEM engine components"""
        # Initialize Redis for caching
        self.redis_client = await aioredis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379')
        )
        
        # Initialize Elasticsearch for log storage and search
        self.elasticsearch_client = AsyncElasticsearch([
            self.config.get('elasticsearch_url', 'http://localhost:9200')
        ])
        
        # Initialize PostgreSQL for structured data
        self.db_pool = await asyncpg.create_pool(
            self.config.get('database_url', 'postgresql://user:pass@localhost/siem')
        )
        
        # Load threat intelligence
        await self.threat_intelligence.load_threat_feeds()
        
        # Load correlation rules
        await self.load_correlation_rules()
        
        logging.info("SIEM Engine initialized successfully")
    
    async def load_correlation_rules(self) -> None:
        """Load event correlation rules"""
        self.correlation_rules = [
            {
                'name': 'multiple_failed_logins',
                'description': 'Multiple failed login attempts from same IP',
                'conditions': {
                    'event_type': EventType.AUTHENTICATION,
                    'outcome': 'failure',
                    'time_window': 300,  # 5 minutes
                    'threshold': 5
                },
                'severity': ThreatLevel.HIGH
            },
            {
                'name': 'geological_data_exfiltration',
                'description': 'Suspicious geological data access patterns',
                'conditions': {
                    'geological_context': True,
                    'data_volume_threshold': 1000000,  # 1MB
                    'time_window': 3600,  # 1 hour
                    'unusual_hours': True
                },
                'severity': ThreatLevel.CRITICAL
            },
            {
                'name': 'privilege_escalation_attempt',
                'description': 'Potential privilege escalation',
                'conditions': {
                    'event_type': EventType.AUTHORIZATION,
                    'outcome': 'failure',
                    'action_contains': ['admin', 'root', 'sudo'],
                    'time_window': 600  # 10 minutes
                },
                'severity': ThreatLevel.HIGH
            }
        ]
    
    async def process_event(self, event: SecurityEvent) -> Optional[SecurityIncident]:
        """Process security event and detect threats"""
        # Store event
        await self.store_event(event)
        
        # Add to buffer for correlation
        self.event_buffer.append(event)
        
        # Check threat intelligence
        threat_indicators = await self.threat_intelligence.check_indicators(event)
        event.threat_indicators = [indicator.type for indicator in threat_indicators]
        
        # Perform behavioral analysis
        if event.user_id:
            anomalies = await self.behavioral_analytics.detect_anomalies(
                event.user_id, [event]
            )
            
            if anomalies:
                logging.warning(f"Behavioral anomalies detected for user {event.user_id}: {anomalies}")
        
        # Check correlation rules
        incident = await self.check_correlation_rules(event)
        
        if incident:
            await self.create_incident(incident)
            
        return incident
    
    async def store_event(self, event: SecurityEvent) -> None:
        """Store security event in Elasticsearch and database"""
        # Store in Elasticsearch for full-text search
        await self.elasticsearch_client.index(
            index=f"security-events-{datetime.now().strftime('%Y-%m')}",
            body=asdict(event)
        )
        
        # Store structured data in PostgreSQL
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_events 
                (id, timestamp, event_type, source_ip, user_id, resource, action, outcome, severity, raw_data)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, event.id, event.timestamp, event.event_type.value, event.source_ip,
                event.user_id, event.resource, event.action, event.outcome, 
                event.severity, json.dumps(event.raw_data))
    
    async def check_correlation_rules(self, current_event: SecurityEvent) -> Optional[SecurityIncident]:
        """Check if current event triggers any correlation rules"""
        for rule in self.correlation_rules:
            if await self.evaluate_rule(rule, current_event):
                return await self.create_incident_from_rule(rule, current_event)
        
        return None
    
    async def evaluate_rule(self, rule: Dict[str, Any], event: SecurityEvent) -> bool:
        """Evaluate if a correlation rule is triggered"""
        conditions = rule['conditions']
        
        # Check event type
        if 'event_type' in conditions and event.event_type != conditions['event_type']:
            return False
        
        # Check outcome
        if 'outcome' in conditions and event.outcome != conditions['outcome']:
            return False
        
        # Check geological context
        if conditions.get('geological_context') and not event.geological_context:
            return False
        
        # Check time-based conditions
        if 'time_window' in conditions and 'threshold' in conditions:
            time_window = timedelta(seconds=conditions['time_window'])
            cutoff_time = event.timestamp - time_window
            
            matching_events = [
                e for e in self.event_buffer 
                if e.timestamp >= cutoff_time and self.event_matches_conditions(e, conditions)
            ]
            
            if len(matching_events) < conditions['threshold']:
                return False
        
        return True
    
    def event_matches_conditions(self, event: SecurityEvent, conditions: Dict[str, Any]) -> bool:
        """Check if event matches rule conditions"""
        if 'event_type' in conditions and event.event_type != conditions['event_type']:
            return False
            
        if 'outcome' in conditions and event.outcome != conditions['outcome']:
            return False
            
        if 'action_contains' in conditions:
            action_keywords = conditions['action_contains']
            if not any(keyword.lower() in event.action.lower() for keyword in action_keywords):
                return False
        
        return True
    
    async def create_incident_from_rule(self, rule: Dict[str, Any], trigger_event: SecurityEvent) -> SecurityIncident:
        """Create security incident from triggered rule"""
        incident_id = str(uuid.uuid4())
        
        # Gather related events
        time_window = timedelta(seconds=rule['conditions'].get('time_window', 3600))
        cutoff_time = trigger_event.timestamp - time_window
        
        related_events = [
            e for e in self.event_buffer 
            if e.timestamp >= cutoff_time and self.event_matches_conditions(e, rule['conditions'])
        ]
        
        incident = SecurityIncident(
            id=incident_id,
            title=rule['name'].replace('_', ' ').title(),
            description=rule['description'],
            threat_level=rule['severity'],
            status=IncidentStatus.OPEN,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            assigned_to=None,
            events=related_events,
            indicators=[],
            geological_impact=self.assess_geological_impact(related_events),
            response_actions=[],
            timeline=[{
                'timestamp': datetime.utcnow(),
                'action': 'incident_created',
                'description': f'Incident created from rule: {rule["name"]}',
                'user': 'system'
            }]
        )
        
        return incident
    
    def assess_geological_impact(self, events: List[SecurityEvent]) -> Optional[Dict[str, Any]]:
        """Assess potential impact on geological operations"""
        geological_events = [e for e in events if e.geological_context]
        
        if not geological_events:
            return None
        
        workflows_affected = set()
        data_types_affected = set()
        
        for event in geological_events:
            ctx = event.geological_context
            if ctx:
                workflows_affected.add(ctx.get('workflow_type', 'unknown'))
                data_types_affected.add(ctx.get('data_type', 'unknown'))
        
        return {
            'workflows_affected': list(workflows_affected),
            'data_types_affected': list(data_types_affected),
            'potential_data_compromise': len(geological_events) > 5,
            'research_impact_level': 'high' if len(workflows_affected) > 3 else 'medium'
        }
    
    async def create_incident(self, incident: SecurityIncident) -> None:
        """Create and store security incident"""
        self.incidents[incident.id] = incident
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO security_incidents 
                (id, title, description, threat_level, status, created_at, updated_at, geological_impact)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, incident.id, incident.title, incident.description, 
                incident.threat_level.value, incident.status.value,
                incident.created_at, incident.updated_at, 
                json.dumps(incident.geological_impact))
        
        # Trigger automated response
        await self.trigger_automated_response(incident)
        
        logging.critical(f"Security incident created: {incident.title} (ID: {incident.id})")
    
    async def trigger_automated_response(self, incident: SecurityIncident) -> None:
        """Trigger automated incident response actions"""
        if incident.threat_level == ThreatLevel.CRITICAL:
            # Immediate containment actions
            await self.execute_containment_actions(incident)
        
        # Send notifications
        await self.send_incident_notifications(incident)
        
        # Update incident timeline
        incident.timeline.append({
            'timestamp': datetime.utcnow(),
            'action': 'automated_response_triggered',
            'description': 'Automated response actions initiated',
            'user': 'system'
        })
    
    async def execute_containment_actions(self, incident: SecurityIncident) -> None:
        """Execute automated containment actions"""
        actions = []
        
        # Block suspicious IPs
        suspicious_ips = set()
        for event in incident.events:
            if event.threat_indicators and 'malicious_ip' in event.threat_indicators:
                suspicious_ips.add(event.source_ip)
        
        for ip in suspicious_ips:
            await self.block_ip_address(ip)
            actions.append(f"Blocked IP address: {ip}")
        
        # Disable compromised user accounts
        compromised_users = set()
        for event in incident.events:
            if event.user_id and event.severity >= 8:
                compromised_users.add(event.user_id)
        
        for user_id in compromised_users:
            await self.disable_user_account(user_id)
            actions.append(f"Disabled user account: {user_id}")
        
        incident.response_actions.extend([{
            'timestamp': datetime.utcnow(),
            'action': action,
            'type': 'automated_containment'
        } for action in actions])
    
    async def block_ip_address(self, ip_address: str) -> None:
        """Block IP address at network level"""
        # In production, this would integrate with firewall/WAF
        await self.redis_client.sadd('blocked_ips', ip_address)
        logging.warning(f"IP address blocked: {ip_address}")
    
    async def disable_user_account(self, user_id: str) -> None:
        """Disable user account"""
        # In production, this would integrate with identity provider
        await self.redis_client.sadd('disabled_users', user_id)
        logging.warning(f"User account disabled: {user_id}")
    
    async def send_incident_notifications(self, incident: SecurityIncident) -> None:
        """Send incident notifications to security team"""
        notification = {
            'incident_id': incident.id,
            'title': incident.title,
            'threat_level': incident.threat_level.value,
            'description': incident.description,
            'timestamp': incident.created_at.isoformat(),
            'geological_impact': incident.geological_impact
        }
        
        # Send to notification service (Kafka, email, Slack, etc.)
        # This is a placeholder for actual notification implementation
        logging.info(f"Incident notification sent: {json.dumps(notification)}")
    
    async def get_incident_dashboard_data(self) -> Dict[str, Any]:
        """Get data for security incident dashboard"""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        async with self.db_pool.acquire() as conn:
            # Recent incidents
            recent_incidents = await conn.fetch("""
                SELECT threat_level, COUNT(*) as count
                FROM security_incidents 
                WHERE created_at >= $1
                GROUP BY threat_level
            """, last_24h)
            
            # Incident trends
            incident_trends = await conn.fetch("""
                SELECT DATE(created_at) as date, COUNT(*) as count
                FROM security_incidents 
                WHERE created_at >= $1
                GROUP BY DATE(created_at)
                ORDER BY date
            """, last_7d)
            
            # Top threat indicators
            top_indicators = await conn.fetch("""
                SELECT event_type, COUNT(*) as count
                FROM security_events 
                WHERE timestamp >= $1 AND severity >= 7
                GROUP BY event_type
                ORDER BY count DESC
                LIMIT 10
            """, last_24h)
        
        return {
            'recent_incidents': {row['threat_level']: row['count'] for row in recent_incidents},
            'incident_trends': [{'date': row['date'].isoformat(), 'count': row['count']} 
                              for row in incident_trends],
            'top_threat_indicators': [{'type': row['event_type'], 'count': row['count']} 
                                    for row in top_indicators],
            'total_incidents': len(self.incidents),
            'active_incidents': len([i for i in self.incidents.values() 
                                   if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]])
        }

# Database schema for SIEM
SIEM_SCHEMA = """
CREATE TABLE IF NOT EXISTS security_events (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    source_ip INET NOT NULL,
    user_id VARCHAR(255),
    resource VARCHAR(500) NOT NULL,
    action VARCHAR(255) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    severity INTEGER NOT NULL,
    raw_data JSONB,
    geological_context JSONB,
    threat_indicators TEXT[],
    correlation_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS security_incidents (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    threat_level VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    assigned_to VARCHAR(255),
    geological_impact JSONB,
    response_actions JSONB,
    timeline JSONB
);

CREATE TABLE IF NOT EXISTS threat_indicators (
    id UUID PRIMARY KEY,
    type VARCHAR(100) NOT NULL,
    value VARCHAR(500) NOT NULL,
    confidence FLOAT NOT NULL,
    source VARCHAR(100) NOT NULL,
    first_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen TIMESTAMP WITH TIME ZONE NOT NULL,
    geological_relevance BOOLEAN DEFAULT FALSE,
    context JSONB
);

CREATE INDEX IF NOT EXISTS idx_security_events_timestamp ON security_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX IF NOT EXISTS idx_security_events_source_ip ON security_events(source_ip);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_incidents_threat_level ON security_incidents(threat_level);
CREATE INDEX IF NOT EXISTS idx_security_incidents_status ON security_incidents(status);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_type ON threat_indicators(type);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_value ON threat_indicators(value);
"""