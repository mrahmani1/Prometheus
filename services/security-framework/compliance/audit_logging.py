"""
Tamper-Proof Audit Logging System for Prometheus AI Platform

This module implements comprehensive audit logging with cryptographic integrity,
immutable storage, and automated compliance reporting for regulatory requirements.
"""

import asyncio
import json
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import aioredis
import asyncpg
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os

# Configure structured logging
logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Audit event type classification"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_ADMINISTRATION = "system_administration"
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    AI_MODEL_OPERATION = "ai_model_operation"
    GEOLOGICAL_DATA_PROCESSING = "geological_data_processing"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "gdpr"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    NIST = "nist"

@dataclass
class AuditEvent:
    """Comprehensive audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: str
    user_agent: Optional[str]
    resource: str
    action: str
    outcome: str  # success, failure, error
    details: Dict[str, Any]
    geological_context: Optional[Dict[str, Any]]  # Geological-specific context
    compliance_tags: List[ComplianceFramework]
    risk_score: int  # 1-10 scale
    data_classification: str
    retention_period: int  # days

@dataclass
class AuditChainBlock:
    """Blockchain-style audit log block for tamper detection"""
    block_id: str
    timestamp: datetime
    previous_hash: str
    events: List[str]  # Event IDs in this block
    merkle_root: str
    block_hash: str
    digital_signature: str

@dataclass
class ComplianceReport:
    """Automated compliance report generation"""
    report_id: str
    framework: ComplianceFramework
    report_period_start: datetime
    report_period_end: datetime
    total_events: int
    events_by_type: Dict[str, int]
    security_incidents: int
    compliance_violations: int
    risk_assessment: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime

class TamperProofAuditLogger:
    """
    Comprehensive tamper-proof audit logging system with cryptographic integrity,
    blockchain-style chaining, and automated compliance reporting for geological
    research platform.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, 
                 signing_key: rsa.RSAPrivateKey, encryption_key: bytes):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.signing_key = signing_key
        self.encryption_key = encryption_key
        self.current_block_events = []
        self.block_size = 100  # Events per block
        
    async def log_audit_event(self, event: AuditEvent) -> bool:
        """
        Log audit event with cryptographic integrity and tamper detection
        
        Args:
            event: Audit event to log
            
        Returns:
            bool: Success status of logging operation
        """
        try:
            # Encrypt sensitive event details
            encrypted_details = await self._encrypt_event_details(event.details)
            
            # Calculate event hash for integrity
            event_hash = self._calculate_event_hash(event)
            
            # Store event in database with encryption
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO audit_events 
                    (event_id, timestamp, event_type, severity, user_id, session_id,
                     source_ip, user_agent, resource, action, outcome, encrypted_details,
                     geological_context, compliance_tags, risk_score, data_classification,
                     retention_period, event_hash, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                """,
                event.event_id, event.timestamp, event.event_type.value, event.severity.value,
                event.user_id, event.session_id, event.source_ip, event.user_agent,
                event.resource, event.action, event.outcome, encrypted_details,
                json.dumps(event.geological_context) if event.geological_context else None,
                [f.value for f in event.compliance_tags], event.risk_score,
                event.data_classification, event.retention_period, event_hash,
                datetime.utcnow()
                )
            
            # Add to current block for blockchain-style chaining
            self.current_block_events.append(event.event_id)
            
            # Create new block if current block is full
            if len(self.current_block_events) >= self.block_size:
                await self._create_audit_block()
            
            # Cache high-risk events for real-time monitoring
            if event.risk_score >= 8:
                await self.redis_client.lpush(
                    "high_risk_audit_events",
                    json.dumps({
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "event_type": event.event_type.value,
                        "risk_score": event.risk_score,
                        "user_id": event.user_id,
                        "resource": event.resource,
                        "action": event.action
                    })
                )
                await self.redis_client.ltrim("high_risk_audit_events", 0, 999)  # Keep last 1000
            
            # Update compliance metrics
            await self._update_compliance_metrics(event)
            
            logger.info(f"Audit event logged: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
            return False
    
    async def _encrypt_event_details(self, details: Dict[str, Any]) -> bytes:
        """Encrypt sensitive event details using AES encryption"""
        # Convert details to JSON string
        details_json = json.dumps(details, default=str)
        
        # Generate random IV
        iv = os.urandom(16)
        
        # Create cipher
        cipher = Cipher(algorithms.AES(self.encryption_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(details_json.encode())
        
        # Encrypt data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted_data
    
    def _pad_data(self, data: bytes) -> bytes:
        """Apply PKCS7 padding to data"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Calculate SHA-256 hash of event for integrity verification"""
        event_string = f"{event.event_id}{event.timestamp.isoformat()}{event.event_type.value}{event.user_id}{event.resource}{event.action}{event.outcome}"
        return hashlib.sha256(event_string.encode()).hexdigest()
    
    async def _create_audit_block(self) -> AuditChainBlock:
        """Create blockchain-style audit block with cryptographic chaining"""
        if not self.current_block_events:
            return None
        
        # Get previous block hash
        async with self.db_pool.acquire() as conn:
            previous_block = await conn.fetchrow("""
                SELECT block_hash FROM audit_blocks 
                ORDER BY timestamp DESC LIMIT 1
            """)
            previous_hash = previous_block['block_hash'] if previous_block else "0" * 64
        
        # Calculate Merkle root of events
        merkle_root = self._calculate_merkle_root(self.current_block_events)
        
        # Create block
        block = AuditChainBlock(
            block_id=f"block_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            previous_hash=previous_hash,
            events=self.current_block_events.copy(),
            merkle_root=merkle_root,
            block_hash="",  # Will be calculated
            digital_signature=""  # Will be calculated
        )
        
        # Calculate block hash
        block_data = f"{block.block_id}{block.timestamp.isoformat()}{block.previous_hash}{block.merkle_root}"
        block.block_hash = hashlib.sha256(block_data.encode()).hexdigest()
        
        # Create digital signature
        block.digital_signature = self._sign_block(block.block_hash)
        
        # Store block in database
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO audit_blocks 
                (block_id, timestamp, previous_hash, events, merkle_root, 
                 block_hash, digital_signature, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            block.block_id, block.timestamp, block.previous_hash, block.events,
            block.merkle_root, block.block_hash, block.digital_signature,
            datetime.utcnow()
            )
        
        # Clear current block events
        self.current_block_events = []
        
        logger.info(f"Audit block created: {block.block_id}")
        return block
    
    def _calculate_merkle_root(self, event_ids: List[str]) -> str:
        """Calculate Merkle root hash of event IDs"""
        if not event_ids:
            return hashlib.sha256(b"").hexdigest()
        
        # Create leaf hashes
        hashes = [hashlib.sha256(event_id.encode()).hexdigest() for event_id in event_ids]
        
        # Build Merkle tree
        while len(hashes) > 1:
            new_hashes = []
            for i in range(0, len(hashes), 2):
                if i + 1 < len(hashes):
                    combined = hashes[i] + hashes[i + 1]
                else:
                    combined = hashes[i] + hashes[i]  # Duplicate if odd number
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0]
    
    def _sign_block(self, block_hash: str) -> str:
        """Create digital signature for audit block"""
        signature = self.signing_key.sign(
            block_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    async def verify_audit_integrity(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Verify integrity of audit logs within specified date range
        
        Args:
            start_date: Start of verification period
            end_date: End of verification period
            
        Returns:
            Dict containing integrity verification results
        """
        verification_results = {
            "verification_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events_verified": 0,
            "total_blocks_verified": 0,
            "integrity_violations": [],
            "chain_integrity": True,
            "signature_validity": True,
            "overall_status": "valid"
        }
        
        try:
            async with self.db_pool.acquire() as conn:
                # Verify individual event hashes
                events = await conn.fetch("""
                    SELECT event_id, timestamp, event_type, user_id, resource, 
                           action, outcome, event_hash
                    FROM audit_events 
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp
                """, start_date, end_date)
                
                for event_row in events:
                    # Recalculate event hash
                    event_string = f"{event_row['event_id']}{event_row['timestamp'].isoformat()}{event_row['event_type']}{event_row['user_id']}{event_row['resource']}{event_row['action']}{event_row['outcome']}"
                    calculated_hash = hashlib.sha256(event_string.encode()).hexdigest()
                    
                    if calculated_hash != event_row['event_hash']:
                        verification_results["integrity_violations"].append({
                            "type": "event_hash_mismatch",
                            "event_id": event_row['event_id'],
                            "expected_hash": event_row['event_hash'],
                            "calculated_hash": calculated_hash
                        })
                
                verification_results["total_events_verified"] = len(events)
                
                # Verify block chain integrity
                blocks = await conn.fetch("""
                    SELECT block_id, timestamp, previous_hash, events, merkle_root,
                           block_hash, digital_signature
                    FROM audit_blocks 
                    WHERE timestamp BETWEEN $1 AND $2
                    ORDER BY timestamp
                """, start_date, end_date)
                
                previous_hash = None
                for i, block_row in enumerate(blocks):
                    # Verify previous hash linkage
                    if i > 0 and block_row['previous_hash'] != previous_hash:
                        verification_results["integrity_violations"].append({
                            "type": "chain_break",
                            "block_id": block_row['block_id'],
                            "expected_previous_hash": previous_hash,
                            "actual_previous_hash": block_row['previous_hash']
                        })
                        verification_results["chain_integrity"] = False
                    
                    # Verify block hash
                    block_data = f"{block_row['block_id']}{block_row['timestamp'].isoformat()}{block_row['previous_hash']}{block_row['merkle_root']}"
                    calculated_block_hash = hashlib.sha256(block_data.encode()).hexdigest()
                    
                    if calculated_block_hash != block_row['block_hash']:
                        verification_results["integrity_violations"].append({
                            "type": "block_hash_mismatch",
                            "block_id": block_row['block_id'],
                            "expected_hash": block_row['block_hash'],
                            "calculated_hash": calculated_block_hash
                        })
                    
                    # Verify Merkle root
                    calculated_merkle = self._calculate_merkle_root(block_row['events'])
                    if calculated_merkle != block_row['merkle_root']:
                        verification_results["integrity_violations"].append({
                            "type": "merkle_root_mismatch",
                            "block_id": block_row['block_id'],
                            "expected_merkle": block_row['merkle_root'],
                            "calculated_merkle": calculated_merkle
                        })
                    
                    previous_hash = block_row['block_hash']
                
                verification_results["total_blocks_verified"] = len(blocks)
                
                # Determine overall status
                if verification_results["integrity_violations"]:
                    verification_results["overall_status"] = "compromised"
                elif not verification_results["chain_integrity"]:
                    verification_results["overall_status"] = "chain_broken"
                
        except Exception as e:
            logger.error(f"Audit integrity verification failed: {str(e)}")
            verification_results["overall_status"] = "verification_failed"
            verification_results["error"] = str(e)
        
        return verification_results
    
    async def generate_compliance_report(self, framework: ComplianceFramework,
                                       start_date: datetime, end_date: datetime) -> ComplianceReport:
        """
        Generate automated compliance report for specified framework
        
        Args:
            framework: Compliance framework to report on
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            ComplianceReport: Generated compliance report
        """
        report_id = f"compliance_{framework.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        async with self.db_pool.acquire() as conn:
            # Get total events for period
            total_events = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND $3 = ANY(compliance_tags)
            """, start_date, end_date, framework.value)
            
            # Get events by type
            events_by_type = await conn.fetch("""
                SELECT event_type, COUNT(*) as count
                FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND $3 = ANY(compliance_tags)
                GROUP BY event_type
            """, start_date, end_date, framework.value)
            
            events_by_type_dict = {row['event_type']: row['count'] for row in events_by_type}
            
            # Count security incidents
            security_incidents = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND event_type = 'security_event'
                AND severity IN ('error', 'critical')
                AND $3 = ANY(compliance_tags)
            """, start_date, end_date, framework.value)
            
            # Count compliance violations
            compliance_violations = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND event_type = 'compliance_event'
                AND outcome = 'failure'
                AND $3 = ANY(compliance_tags)
            """, start_date, end_date, framework.value)
        
        # Generate risk assessment
        risk_assessment = await self._generate_risk_assessment(framework, start_date, end_date)
        
        # Generate recommendations
        recommendations = await self._generate_compliance_recommendations(
            framework, events_by_type_dict, security_incidents, compliance_violations
        )
        
        report = ComplianceReport(
            report_id=report_id,
            framework=framework,
            report_period_start=start_date,
            report_period_end=end_date,
            total_events=total_events or 0,
            events_by_type=events_by_type_dict,
            security_incidents=security_incidents or 0,
            compliance_violations=compliance_violations or 0,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            generated_at=datetime.utcnow()
        )
        
        # Store report
        await self._store_compliance_report(report)
        
        return report
    
    async def _generate_risk_assessment(self, framework: ComplianceFramework,
                                      start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate risk assessment for compliance report"""
        async with self.db_pool.acquire() as conn:
            # Calculate average risk score
            avg_risk = await conn.fetchval("""
                SELECT AVG(risk_score) FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND $3 = ANY(compliance_tags)
            """, start_date, end_date, framework.value)
            
            # Count high-risk events
            high_risk_events = await conn.fetchval("""
                SELECT COUNT(*) FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND risk_score >= 8
                AND $3 = ANY(compliance_tags)
            """, start_date, end_date, framework.value)
            
            # Identify top risk sources
            risk_sources = await conn.fetch("""
                SELECT resource, AVG(risk_score) as avg_risk, COUNT(*) as event_count
                FROM audit_events 
                WHERE timestamp BETWEEN $1 AND $2
                AND $3 = ANY(compliance_tags)
                GROUP BY resource
                ORDER BY avg_risk DESC
                LIMIT 10
            """, start_date, end_date, framework.value)
        
        return {
            "average_risk_score": float(avg_risk) if avg_risk else 0.0,
            "high_risk_events": high_risk_events or 0,
            "risk_trend": "stable",  # Could be calculated from historical data
            "top_risk_sources": [
                {
                    "resource": row['resource'],
                    "average_risk": float(row['avg_risk']),
                    "event_count": row['event_count']
                }
                for row in risk_sources
            ]
        }
    
    async def _generate_compliance_recommendations(self, framework: ComplianceFramework,
                                                 events_by_type: Dict[str, int],
                                                 security_incidents: int,
                                                 compliance_violations: int) -> List[str]:
        """Generate compliance recommendations based on audit data"""
        recommendations = []
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            if events_by_type.get("data_access", 0) > 1000:
                recommendations.append("Consider implementing additional access controls for personal data")
            if compliance_violations > 0:
                recommendations.append("Review and strengthen GDPR compliance procedures")
        
        elif framework == ComplianceFramework.SOC2:
            if security_incidents > 5:
                recommendations.append("Enhance security monitoring and incident response procedures")
            if events_by_type.get("system_administration", 0) > 500:
                recommendations.append("Implement additional controls for privileged access management")
        
        elif framework == ComplianceFramework.ISO27001:
            if compliance_violations > 2:
                recommendations.append("Conduct additional security awareness training")
            recommendations.append("Review and update information security risk assessment")
        
        # General recommendations
        if security_incidents > 10:
            recommendations.append("Consider implementing advanced threat detection capabilities")
        
        if not recommendations:
            recommendations.append("Compliance posture appears satisfactory - continue current practices")
        
        return recommendations

# Database schema for audit logging
AUDIT_LOGGING_SCHEMA_SQL = """
-- Audit Events Table
CREATE TABLE IF NOT EXISTS audit_events (
    event_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    source_ip INET NOT NULL,
    user_agent TEXT,
    resource VARCHAR(500) NOT NULL,
    action VARCHAR(100) NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    encrypted_details BYTEA,
    geological_context JSONB,
    compliance_tags TEXT[],
    risk_score INTEGER NOT NULL,
    data_classification VARCHAR(50) NOT NULL,
    retention_period INTEGER NOT NULL,
    event_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit Blocks Table (Blockchain-style)
CREATE TABLE IF NOT EXISTS audit_blocks (
    block_id VARCHAR(255) PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    previous_hash VARCHAR(64) NOT NULL,
    events TEXT[] NOT NULL,
    merkle_root VARCHAR(64) NOT NULL,
    block_hash VARCHAR(64) NOT NULL,
    digital_signature TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance Reports Table
CREATE TABLE IF NOT EXISTS compliance_reports (
    report_id VARCHAR(255) PRIMARY KEY,
    framework VARCHAR(50) NOT NULL,
    report_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    report_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    total_events INTEGER NOT NULL,
    events_by_type JSONB NOT NULL,
    security_incidents INTEGER NOT NULL,
    compliance_violations INTEGER NOT NULL,
    risk_assessment JSONB NOT NULL,
    recommendations TEXT[],
    generated_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance and compliance queries
CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_events_user ON audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_events_compliance ON audit_events USING GIN(compliance_tags);
CREATE INDEX IF NOT EXISTS idx_audit_events_risk ON audit_events(risk_score);
CREATE INDEX IF NOT EXISTS idx_audit_blocks_timestamp ON audit_blocks(timestamp);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_framework ON compliance_reports(framework);
CREATE INDEX IF NOT EXISTS idx_compliance_reports_period ON compliance_reports(report_period_start, report_period_end);

-- Partitioning for large audit tables (optional, for high-volume environments)
-- CREATE TABLE audit_events_y2024m01 PARTITION OF audit_events 
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
"""