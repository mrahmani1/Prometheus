"""
Security Audit Logger with Geological Context Awareness
Implements comprehensive audit logging for zero-trust security framework
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import aioredis
import asyncpg
from dataclasses import asdict

from models.security_models import (
    SecurityEvent, SecurityEventType, GeologicalContext, 
    ThreatAlert, ThreatLevel, generate_security_event_id
)

logger = logging.getLogger(__name__)

class SecurityAuditLogger:
    """
    Comprehensive security audit logger with geological domain awareness
    Implements tamper-proof logging and real-time security event processing
    """
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.log_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        
        # Event severity mapping
        self.event_severity = {
            SecurityEventType.AUTHENTICATION_SUCCESS: 'info',
            SecurityEventType.AUTHENTICATION_FAILURE: 'warning',
            SecurityEventType.AUTHORIZATION_GRANTED: 'info',
            SecurityEventType.AUTHORIZATION_DENIED: 'warning',
            SecurityEventType.SUSPICIOUS_ACTIVITY: 'high',
            SecurityEventType.POLICY_VIOLATION: 'high',
            SecurityEventType.DATA_ACCESS: 'info',
            SecurityEventType.SYSTEM_BREACH: 'critical',
            SecurityEventType.GEOLOGICAL_ANOMALY: 'medium',
            SecurityEventType.EMERGENCY_ACCESS: 'high'
        }
        
        # Geological context enrichment
        self.geological_enrichment = {
            'volcanology': {
                'risk_multiplier': 1.5,  # Higher risk due to public safety
                'retention_period': 2555,  # 7 years in days
                'notification_required': True
            },
            'seismology': {
                'risk_multiplier': 1.4,
                'retention_period': 2555,
                'notification_required': True
            },
            'geochemistry': {
                'risk_multiplier': 1.1,
                'retention_period': 1825,  # 5 years
                'notification_required': False
            },
            'structural_geology': {
                'risk_multiplier': 1.2,
                'retention_period': 1825,
                'notification_required': False
            }
        }
        
    async def initialize(self):
        """Initialize audit logger with database connections"""
        logger.info("Initializing Security Audit Logger")
        
        # Initialize Redis for real-time event streaming
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/5",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Initialize PostgreSQL for persistent audit storage
        self.db_pool = await asyncpg.create_pool(
            "postgresql://prometheus:geological_security@localhost:5432/prometheus_security",
            min_size=5,
            max_size=20
        )
        
        # Create audit tables
        await self._create_audit_tables()
        
        # Start background flush task
        asyncio.create_task(self._periodic_flush())
        
        logger.info("Security Audit Logger initialized successfully")
        
    async def _create_audit_tables(self):
        """Create audit logging database tables"""
        async with self.db_pool.acquire() as conn:
            # Security events audit table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    event_id VARCHAR(100) PRIMARY KEY,
                    event_type VARCHAR(50) NOT NULL,
                    user_id VARCHAR(50),
                    resource_id VARCHAR(100),
                    source_ip INET,
                    user_agent TEXT,
                    geological_context JSONB,
                    event_data JSONB NOT NULL,
                    risk_score FLOAT DEFAULT 0.0,
                    severity VARCHAR(20) NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    correlation_id VARCHAR(100),
                    session_id VARCHAR(100),
                    geological_domain VARCHAR(50),
                    data_classification VARCHAR(50),
                    retention_until DATE,
                    hash_chain VARCHAR(128),
                    previous_hash VARCHAR(128)
                )
            """)
            
            # Create index for efficient querying
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_audit_timestamp 
                ON security_audit_log(timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_audit_user 
                ON security_audit_log(user_id, timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_security_audit_geological 
                ON security_audit_log(geological_domain, timestamp DESC)
            """)
            
            # Threat alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS threat_alerts (
                    alert_id VARCHAR(100) PRIMARY KEY,
                    threat_type VARCHAR(50) NOT NULL,
                    threat_level INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    affected_resources JSONB,
                    geological_context JSONB,
                    indicators JSONB,
                    mitigation_actions JSONB,
                    source_ip INET,
                    user_id VARCHAR(50),
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT false,
                    resolved_at TIMESTAMP,
                    resolved_by VARCHAR(50),
                    resolution_notes TEXT,
                    escalated BOOLEAN DEFAULT false,
                    escalated_to VARCHAR(100)
                )
            """)
            
            # Audit log integrity table for tamper detection
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_integrity (
                    batch_id VARCHAR(100) PRIMARY KEY,
                    start_event_id VARCHAR(100) NOT NULL,
                    end_event_id VARCHAR(100) NOT NULL,
                    event_count INTEGER NOT NULL,
                    batch_hash VARCHAR(128) NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    verified_at TIMESTAMP,
                    integrity_status VARCHAR(20) DEFAULT 'pending'
                )
            """)
            
    async def log_authentication_success(
        self, 
        user_id: str, 
        geological_context: Optional[GeologicalContext] = None,
        additional_data: Optional[Dict] = None
    ):
        """Log successful authentication event"""
        
        event_data = {
            'authentication_method': additional_data.get('method') if additional_data else 'unknown',
            'mfa_used': additional_data.get('mfa_used', False) if additional_data else False,
            'device_trusted': additional_data.get('device_trusted', False) if additional_data else False
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            user_id=user_id,
            geological_context=geological_context,
            event_data=event_data,
            success=True
        )
        
    async def log_authentication_failure(
        self, 
        username: str, 
        failure_reason: str,
        geological_context: Optional[GeologicalContext] = None,
        source_ip: Optional[str] = None
    ):
        """Log failed authentication event"""
        
        event_data = {
            'username': username,
            'failure_reason': failure_reason,
            'brute_force_indicator': await self._check_brute_force_pattern(username, source_ip)
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            geological_context=geological_context,
            event_data=event_data,
            success=False,
            source_ip=source_ip
        )
        
    async def log_access_granted(
        self, 
        user_id: str, 
        resource_id: str, 
        access_level: str,
        geological_context: Optional[GeologicalContext] = None
    ):
        """Log successful resource access"""
        
        event_data = {
            'resource_id': resource_id,
            'access_level': access_level,
            'authorization_method': 'dynamic_permission'
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION_GRANTED,
            user_id=user_id,
            resource_id=resource_id,
            geological_context=geological_context,
            event_data=event_data,
            success=True
        )
        
    async def log_access_denied(
        self, 
        user_id: str, 
        resource_id: str, 
        access_level: str,
        geological_context: Optional[GeologicalContext] = None,
        denial_reason: str = "Insufficient permissions"
    ):
        """Log denied resource access"""
        
        event_data = {
            'resource_id': resource_id,
            'access_level': access_level,
            'denial_reason': denial_reason,
            'risk_indicators': await self._analyze_access_risk(user_id, resource_id)
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION_DENIED,
            user_id=user_id,
            resource_id=resource_id,
            geological_context=geological_context,
            event_data=event_data,
            success=False
        )
        
    async def log_resource_accessed(
        self, 
        user_id: str, 
        resource_id: str, 
        access_level: str,
        geological_context: Optional[GeologicalContext] = None,
        data_volume: Optional[int] = None
    ):
        """Log actual resource access/usage"""
        
        event_data = {
            'resource_id': resource_id,
            'access_level': access_level,
            'data_volume_bytes': data_volume,
            'access_duration': None,  # Would be calculated from session data
            'geological_workflow': geological_context.intended_workflow if geological_context else None
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            resource_id=resource_id,
            geological_context=geological_context,
            event_data=event_data,
            success=True
        )
        
    async def log_policy_violation(
        self, 
        policy_id: str, 
        violation_type: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None,
        violation_details: Optional[Dict] = None
    ):
        """Log security policy violation"""
        
        event_data = {
            'policy_id': policy_id,
            'violation_type': violation_type,
            'violation_details': violation_details or {},
            'automatic_mitigation': await self._determine_automatic_mitigation(violation_type)
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.POLICY_VIOLATION,
            user_id=user_id,
            resource_id=resource_id,
            geological_context=geological_context,
            event_data=event_data,
            success=False
        )
        
        # Create threat alert for high-severity violations
        if violation_type in ['data_exfiltration', 'privilege_escalation', 'unauthorized_access']:
            await self._create_threat_alert(
                threat_type=violation_type,
                threat_level=ThreatLevel.HIGH,
                description=f"Security policy violation: {violation_type}",
                user_id=user_id,
                resource_id=resource_id,
                geological_context=geological_context
            )
            
    async def log_suspicious_activity(
        self, 
        activity_type: str, 
        description: str,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None,
        indicators: Optional[List[str]] = None
    ):
        """Log suspicious security activity"""
        
        event_data = {
            'activity_type': activity_type,
            'description': description,
            'indicators': indicators or [],
            'confidence_score': await self._calculate_threat_confidence(activity_type, indicators)
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=user_id,
            geological_context=geological_context,
            event_data=event_data,
            success=False,
            source_ip=source_ip
        )
        
    async def log_mfa_challenge_initiated(
        self, 
        user_id: str, 
        method_id: str, 
        challenge_id: str,
        geological_context: Optional[GeologicalContext] = None
    ):
        """Log MFA challenge initiation"""
        
        event_data = {
            'mfa_method': method_id,
            'challenge_id': challenge_id,
            'geological_workflow_validation': geological_context.intended_workflow if geological_context else None
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,  # Sub-type of auth
            user_id=user_id,
            geological_context=geological_context,
            event_data=event_data,
            success=True
        )
        
    async def log_mfa_verification(
        self, 
        user_id: str, 
        method_id: str, 
        challenge_id: str, 
        success: bool
    ):
        """Log MFA verification result"""
        
        event_data = {
            'mfa_method': method_id,
            'challenge_id': challenge_id,
            'verification_result': 'success' if success else 'failure'
        }
        
        event_type = SecurityEventType.AUTHENTICATION_SUCCESS if success else SecurityEventType.AUTHENTICATION_FAILURE
        
        await self._log_security_event(
            event_type=event_type,
            user_id=user_id,
            event_data=event_data,
            success=success
        )
        
    async def log_device_validation(
        self, 
        device_id: str, 
        trust_assessment: Dict,
        geological_context: Optional[GeologicalContext] = None
    ):
        """Log device trust validation"""
        
        event_data = {
            'device_id': device_id,
            'trust_score': trust_assessment.get('trust_score', 0.0),
            'risk_factors': trust_assessment.get('risk_factors', []),
            'geological_equipment': geological_context.equipment_context if geological_context else []
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.AUTHORIZATION_GRANTED,
            geological_context=geological_context,
            event_data=event_data,
            success=trust_assessment.get('trust_score', 0.0) >= 0.7
        )
        
    async def log_security_violation(self, violation_data: Dict):
        """Log high-severity security violation"""
        
        await self._log_security_event(
            event_type=SecurityEventType.POLICY_VIOLATION,
            user_id=violation_data.get('user_id'),
            geological_context=violation_data.get('geological_context'),
            event_data=violation_data,
            success=False,
            source_ip=violation_data.get('source_ip')
        )
        
        # Create immediate threat alert
        await self._create_threat_alert(
            threat_type='security_violation',
            threat_level=ThreatLevel.HIGH,
            description=violation_data.get('violation_reason', 'Security violation detected'),
            user_id=violation_data.get('user_id'),
            source_ip=violation_data.get('source_ip')
        )
        
    async def log_mfa_error(self, challenge_id: str, error_message: str):
        """Log MFA system error"""
        
        event_data = {
            'challenge_id': challenge_id,
            'error_type': 'mfa_system_error',
            'error_message': error_message
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.SYSTEM_BREACH,  # System errors are potential security issues
            event_data=event_data,
            success=False,
            error_message=error_message
        )
        
    async def log_authentication_error(self, username: str, error_message: str):
        """Log authentication system error"""
        
        event_data = {
            'username': username,
            'error_type': 'authentication_system_error',
            'error_message': error_message
        }
        
        await self._log_security_event(
            event_type=SecurityEventType.SYSTEM_BREACH,
            event_data=event_data,
            success=False,
            error_message=error_message
        )
        
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None,
        event_data: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Internal method to log security events with geological enrichment"""
        
        event_id = generate_security_event_id()
        timestamp = datetime.utcnow()
        
        # Calculate risk score
        risk_score = await self._calculate_event_risk_score(
            event_type, geological_context, event_data, success
        )
        
        # Determine geological domain
        geological_domain = None
        if geological_context and geological_context.primary_domain:
            geological_domain = geological_context.primary_domain.value
            
        # Create security event
        security_event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            resource_id=resource_id,
            source_ip=source_ip,
            user_agent=user_agent,
            geological_context=geological_context,
            event_data=event_data or {},
            risk_score=risk_score,
            success=success,
            error_message=error_message,
            timestamp=timestamp
        )
        
        # Add to buffer for batch processing
        self.log_buffer.append(security_event)
        
        # Immediate flush for high-risk events
        if risk_score >= 0.8 or event_type in [SecurityEventType.SYSTEM_BREACH, SecurityEventType.POLICY_VIOLATION]:
            await self._flush_log_buffer()
            
        # Stream to Redis for real-time monitoring
        await self._stream_event_to_redis(security_event)
        
    async def _calculate_event_risk_score(
        self,
        event_type: SecurityEventType,
        geological_context: Optional[GeologicalContext],
        event_data: Optional[Dict],
        success: bool
    ) -> float:
        """Calculate risk score for security event"""
        
        base_risk = {
            SecurityEventType.AUTHENTICATION_SUCCESS: 0.1,
            SecurityEventType.AUTHENTICATION_FAILURE: 0.6,
            SecurityEventType.AUTHORIZATION_GRANTED: 0.2,
            SecurityEventType.AUTHORIZATION_DENIED: 0.5,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.8,
            SecurityEventType.POLICY_VIOLATION: 0.9,
            SecurityEventType.DATA_ACCESS: 0.3,
            SecurityEventType.SYSTEM_BREACH: 1.0,
            SecurityEventType.GEOLOGICAL_ANOMALY: 0.4,
            SecurityEventType.EMERGENCY_ACCESS: 0.7
        }.get(event_type, 0.5)
        
        # Adjust for success/failure
        if not success:
            base_risk *= 1.5
            
        # Geological context adjustments
        if geological_context and geological_context.primary_domain:
            domain_name = geological_context.primary_domain.value
            enrichment = self.geological_enrichment.get(domain_name, {})
            risk_multiplier = enrichment.get('risk_multiplier', 1.0)
            base_risk *= risk_multiplier
            
        # Event-specific risk factors
        if event_data:
            if event_data.get('brute_force_indicator'):
                base_risk += 0.3
            if event_data.get('privilege_escalation'):
                base_risk += 0.4
            if event_data.get('data_exfiltration'):
                base_risk += 0.5
                
        return min(base_risk, 1.0)
        
    async def _stream_event_to_redis(self, event: SecurityEvent):
        """Stream security event to Redis for real-time monitoring"""
        
        event_dict = asdict(event)
        
        # Convert datetime to ISO format
        event_dict['timestamp'] = event.timestamp.isoformat()
        
        # Convert geological context
        if event.geological_context:
            event_dict['geological_context'] = asdict(event.geological_context)
            
        # Stream to Redis
        await self.redis_client.xadd(
            'security_events_stream',
            event_dict,
            maxlen=10000  # Keep last 10k events in stream
        )
        
        # Also publish to specific channels for real-time alerts
        if event.risk_score >= 0.8:
            await self.redis_client.publish('high_risk_security_events', json.dumps(event_dict))
            
    async def _flush_log_buffer(self):
        """Flush log buffer to database"""
        
        if not self.log_buffer:
            return
            
        try:
            async with self.db_pool.acquire() as conn:
                # Calculate hash chain for tamper detection
                previous_hash = await self._get_last_hash_chain()
                
                for i, event in enumerate(self.log_buffer):
                    # Calculate retention period
                    retention_until = await self._calculate_retention_period(event)
                    
                    # Calculate hash chain
                    event_hash = await self._calculate_event_hash(event, previous_hash)
                    
                    # Insert event
                    await conn.execute("""
                        INSERT INTO security_audit_log 
                        (event_id, event_type, user_id, resource_id, source_ip, user_agent,
                         geological_context, event_data, risk_score, severity, success,
                         error_message, timestamp, geological_domain, retention_until,
                         hash_chain, previous_hash)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                    """,
                        event.event_id,
                        event.event_type.value,
                        event.user_id,
                        event.resource_id,
                        event.source_ip,
                        event.user_agent,
                        json.dumps(asdict(event.geological_context)) if event.geological_context else None,
                        json.dumps(event.event_data),
                        event.risk_score,
                        self.event_severity.get(event.event_type, 'info'),
                        event.success,
                        event.error_message,
                        event.timestamp,
                        event.geological_context.primary_domain.value if event.geological_context and event.geological_context.primary_domain else None,
                        retention_until,
                        event_hash,
                        previous_hash
                    )
                    
                    previous_hash = event_hash
                    
            # Clear buffer
            self.log_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush audit log buffer: {str(e)}")
            
    async def _periodic_flush(self):
        """Periodic flush of log buffer"""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                if len(self.log_buffer) >= self.buffer_size:
                    await self._flush_log_buffer()
            except Exception as e:
                logger.error(f"Periodic flush error: {str(e)}")
                
    async def _create_threat_alert(
        self,
        threat_type: str,
        threat_level: ThreatLevel,
        description: str,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None
    ):
        """Create threat alert for high-severity events"""
        
        alert_id = f"threat_{datetime.utcnow().timestamp()}_{secrets.token_urlsafe(8)}"
        
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO threat_alerts 
                (alert_id, threat_type, threat_level, description, affected_resources,
                 geological_context, source_ip, user_id, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                alert_id,
                threat_type,
                threat_level.value,
                description,
                json.dumps([resource_id] if resource_id else []),
                json.dumps(asdict(geological_context)) if geological_context else None,
                source_ip,
                user_id,
                datetime.utcnow()
            )
            
        # Publish alert to Redis for immediate notification
        alert_data = {
            'alert_id': alert_id,
            'threat_type': threat_type,
            'threat_level': threat_level.value,
            'description': description,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.redis_client.publish('threat_alerts', json.dumps(alert_data))