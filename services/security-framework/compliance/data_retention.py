"""
Data Retention and Deletion Policy Engine for Prometheus AI Platform

This module implements comprehensive data retention policies, automated deletion
workflows, and compliance with various regulatory requirements for geological
research data lifecycle management.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
import aioredis
import asyncpg
import hashlib
from cryptography.fernet import Fernet

# Configure structured logging
logger = logging.getLogger(__name__)

class RetentionTrigger(Enum):
    """Data retention trigger types"""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    LEGAL_HOLD = "legal_hold"
    USER_REQUEST = "user_request"
    REGULATORY_REQUIREMENT = "regulatory_requirement"

class DataCategory(Enum):
    """Data category classification for retention"""
    PERSONAL_DATA = "personal_data"
    RESEARCH_DATA = "research_data"
    SYSTEM_LOGS = "system_logs"
    AUDIT_LOGS = "audit_logs"
    BACKUP_DATA = "backup_data"
    TEMPORARY_DATA = "temporary_data"
    GEOLOGICAL_MEASUREMENTS = "geological_measurements"
    AI_MODEL_DATA = "ai_model_data"

class DeletionMethod(Enum):
    """Data deletion methods"""
    SOFT_DELETE = "soft_delete"
    HARD_DELETE = "hard_delete"
    CRYPTOGRAPHIC_ERASURE = "cryptographic_erasure"
    PHYSICAL_DESTRUCTION = "physical_destruction"
    ANONYMIZATION = "anonymization"

class RetentionStatus(Enum):
    """Retention policy status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    ON_HOLD = "on_hold"
    DELETED = "deleted"
    ARCHIVED = "archived"

@dataclass
class RetentionPolicy:
    """Data retention policy definition"""
    policy_id: str
    policy_name: str
    data_category: DataCategory
    retention_period_days: int
    trigger_type: RetentionTrigger
    deletion_method: DeletionMethod
    legal_basis: str
    regulatory_framework: List[str]  # GDPR, SOC2, etc.
    exceptions: List[str]
    approval_required: bool
    notification_before_deletion: int  # days
    responsible_party: str
    created_date: datetime
    last_updated: datetime
    active: bool

@dataclass
class DataRecord:
    """Individual data record for retention tracking"""
    record_id: str
    data_category: DataCategory
    creation_date: datetime
    last_accessed: datetime
    retention_policy_id: str
    expiration_date: datetime
    legal_hold: bool
    legal_hold_reason: Optional[str]
    deletion_scheduled: Optional[datetime]
    deletion_method: Optional[DeletionMethod]
    status: RetentionStatus
    metadata: Dict[str, Any]
    geological_context: Optional[Dict[str, Any]]

@dataclass
class DeletionJob:
    """Scheduled deletion job"""
    job_id: str
    policy_id: str
    scheduled_date: datetime
    records_to_delete: List[str]
    deletion_method: DeletionMethod
    approval_status: str  # pending, approved, rejected
    approved_by: Optional[str]
    approval_date: Optional[datetime]
    execution_status: str  # scheduled, running, completed, failed
    execution_date: Optional[datetime]
    deletion_certificate: Optional[str]
    verification_hash: Optional[str]

@dataclass
class LegalHold:
    """Legal hold on data records"""
    hold_id: str
    hold_name: str
    reason: str
    initiated_by: str
    initiation_date: datetime
    affected_records: List[str]
    affected_categories: List[DataCategory]
    expected_end_date: Optional[datetime]
    actual_end_date: Optional[datetime]
    status: str  # active, released, expired

class DataRetentionEngine:
    """
    Comprehensive data retention and deletion engine with automated policy
    enforcement, legal hold management, and compliance reporting for geological
    research platform.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, encryption_key: bytes):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.cipher_suite = Fernet(encryption_key)
        self.retention_policies = {}
        self.active_legal_holds = {}
        
    async def initialize_retention_policies(self):
        """Initialize data retention policies for geological research platform"""
        policies = [
            # Personal Data Retention (GDPR Compliance)
            RetentionPolicy(
                policy_id="GDPR_PERSONAL_001",
                policy_name="Researcher Personal Data",
                data_category=DataCategory.PERSONAL_DATA,
                retention_period_days=2555,  # 7 years
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                legal_basis="GDPR Article 5(1)(e) - storage limitation",
                regulatory_framework=["GDPR"],
                exceptions=["Active research participation", "Legal proceedings"],
                approval_required=True,
                notification_before_deletion=30,
                responsible_party="Data Protection Officer",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            ),
            
            # Research Data Retention
            RetentionPolicy(
                policy_id="RESEARCH_DATA_001",
                policy_name="Geological Research Data",
                data_category=DataCategory.RESEARCH_DATA,
                retention_period_days=3650,  # 10 years (scientific standard)
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.ANONYMIZATION,
                legal_basis="Scientific research value and reproducibility",
                regulatory_framework=["ISO27001", "Research Ethics"],
                exceptions=["Ongoing research projects", "Publication requirements"],
                approval_required=True,
                notification_before_deletion=90,
                responsible_party="Research Data Manager",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            ),
            
            # System Logs Retention
            RetentionPolicy(
                policy_id="SYSTEM_LOGS_001",
                policy_name="System and Application Logs",
                data_category=DataCategory.SYSTEM_LOGS,
                retention_period_days=365,  # 1 year
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.HARD_DELETE,
                legal_basis="Operational necessity and storage optimization",
                regulatory_framework=["SOC2"],
                exceptions=["Security incidents", "Audit requirements"],
                approval_required=False,
                notification_before_deletion=7,
                responsible_party="System Administrator",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            ),
            
            # Audit Logs Retention (Extended for Compliance)
            RetentionPolicy(
                policy_id="AUDIT_LOGS_001",
                policy_name="Security and Compliance Audit Logs",
                data_category=DataCategory.AUDIT_LOGS,
                retention_period_days=2555,  # 7 years (compliance requirement)
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.CRYPTOGRAPHIC_ERASURE,
                legal_basis="Regulatory compliance and audit requirements",
                regulatory_framework=["SOC2", "ISO27001", "GDPR"],
                exceptions=["Active investigations", "Legal proceedings"],
                approval_required=True,
                notification_before_deletion=60,
                responsible_party="Compliance Officer",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            ),
            
            # Geological Measurements (Long-term Scientific Value)
            RetentionPolicy(
                policy_id="GEO_MEASUREMENTS_001",
                policy_name="Seismic and Volcanic Monitoring Data",
                data_category=DataCategory.GEOLOGICAL_MEASUREMENTS,
                retention_period_days=7300,  # 20 years (long-term geological studies)
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.ANONYMIZATION,
                legal_basis="Long-term scientific research and climate studies",
                regulatory_framework=["Research Ethics", "Environmental Regulations"],
                exceptions=["Ongoing monitoring", "Historical significance"],
                approval_required=True,
                notification_before_deletion=180,
                responsible_party="Chief Geologist",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            ),
            
            # AI Model Data
            RetentionPolicy(
                policy_id="AI_MODEL_001",
                policy_name="AI Model Training and Inference Data",
                data_category=DataCategory.AI_MODEL_DATA,
                retention_period_days=1825,  # 5 years
                trigger_type=RetentionTrigger.TIME_BASED,
                deletion_method=DeletionMethod.HARD_DELETE,
                legal_basis="Model lifecycle management and improvement",
                regulatory_framework=["AI Ethics", "Data Protection"],
                exceptions=["Production models", "Benchmark datasets"],
                approval_required=False,
                notification_before_deletion=30,
                responsible_party="ML Engineering Lead",
                created_date=datetime(2024, 1, 1),
                last_updated=datetime(2024, 1, 1),
                active=True
            )
        ]
        
        for policy in policies:
            self.retention_policies[policy.policy_id] = policy
            await self._store_retention_policy(policy)
    
    async def register_data_record(self, record: DataRecord) -> bool:
        """
        Register a new data record for retention tracking
        
        Args:
            record: Data record to register
            
        Returns:
            bool: Success status
        """
        try:
            # Get applicable retention policy
            policy = self.retention_policies.get(record.retention_policy_id)
            if not policy:
                logger.error(f"No retention policy found for {record.retention_policy_id}")
                return False
            
            # Calculate expiration date
            record.expiration_date = record.creation_date + timedelta(days=policy.retention_period_days)
            
            # Encrypt sensitive metadata
            encrypted_metadata = self.cipher_suite.encrypt(
                json.dumps(record.metadata, default=str).encode()
            )
            
            # Store record in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO data_retention_records 
                    (record_id, data_category, creation_date, last_accessed, 
                     retention_policy_id, expiration_date, legal_hold, legal_hold_reason,
                     deletion_scheduled, deletion_method, status, encrypted_metadata,
                     geological_context, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                record.record_id, record.data_category.value, record.creation_date,
                record.last_accessed, record.retention_policy_id, record.expiration_date,
                record.legal_hold, record.legal_hold_reason, record.deletion_scheduled,
                record.deletion_method.value if record.deletion_method else None,
                record.status.value, encrypted_metadata,
                json.dumps(record.geological_context) if record.geological_context else None,
                datetime.utcnow()
                )
            
            # Schedule deletion job if policy allows automatic deletion
            if not policy.approval_required and not record.legal_hold:
                await self._schedule_deletion_job(record, policy)
            
            logger.info(f"Data record registered: {record.record_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register data record: {str(e)}")
            return False
    
    async def apply_legal_hold(self, hold: LegalHold) -> bool:
        """
        Apply legal hold to prevent data deletion
        
        Args:
            hold: Legal hold details
            
        Returns:
            bool: Success status
        """
        try:
            # Store legal hold
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO legal_holds 
                    (hold_id, hold_name, reason, initiated_by, initiation_date,
                     affected_records, affected_categories, expected_end_date, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                hold.hold_id, hold.hold_name, hold.reason, hold.initiated_by,
                hold.initiation_date, hold.affected_records,
                [cat.value for cat in hold.affected_categories],
                hold.expected_end_date, hold.status
                )
                
                # Update affected records
                if hold.affected_records:
                    await conn.execute("""
                        UPDATE data_retention_records 
                        SET legal_hold = true, legal_hold_reason = $1
                        WHERE record_id = ANY($2)
                    """, hold.reason, hold.affected_records)
                
                # Update records by category
                if hold.affected_categories:
                    await conn.execute("""
                        UPDATE data_retention_records 
                        SET legal_hold = true, legal_hold_reason = $1
                        WHERE data_category = ANY($2)
                    """, hold.reason, [cat.value for cat in hold.affected_categories])
                
                # Cancel scheduled deletions for affected records
                await conn.execute("""
                    UPDATE deletion_jobs 
                    SET execution_status = 'cancelled'
                    WHERE records_to_delete && $1
                    AND execution_status = 'scheduled'
                """, hold.affected_records)
            
            self.active_legal_holds[hold.hold_id] = hold
            
            logger.info(f"Legal hold applied: {hold.hold_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply legal hold: {str(e)}")
            return False
    
    async def release_legal_hold(self, hold_id: str, released_by: str) -> bool:
        """
        Release legal hold and resume normal retention processing
        
        Args:
            hold_id: Legal hold identifier
            released_by: Person releasing the hold
            
        Returns:
            bool: Success status
        """
        try:
            hold = self.active_legal_holds.get(hold_id)
            if not hold:
                logger.error(f"Legal hold not found: {hold_id}")
                return False
            
            async with self.db_pool.acquire() as conn:
                # Update legal hold status
                await conn.execute("""
                    UPDATE legal_holds 
                    SET status = 'released', actual_end_date = $1
                    WHERE hold_id = $2
                """, datetime.utcnow(), hold_id)
                
                # Remove legal hold from affected records
                if hold.affected_records:
                    await conn.execute("""
                        UPDATE data_retention_records 
                        SET legal_hold = false, legal_hold_reason = null
                        WHERE record_id = ANY($1)
                    """, hold.affected_records)
                
                # Remove legal hold from records by category
                if hold.affected_categories:
                    await conn.execute("""
                        UPDATE data_retention_records 
                        SET legal_hold = false, legal_hold_reason = null
                        WHERE data_category = ANY($1)
                        AND legal_hold_reason = $2
                    """, [cat.value for cat in hold.affected_categories], hold.reason)
                
                # Reschedule deletions for expired records
                expired_records = await conn.fetch("""
                    SELECT record_id, retention_policy_id, expiration_date
                    FROM data_retention_records 
                    WHERE (record_id = ANY($1) OR data_category = ANY($2))
                    AND expiration_date < NOW()
                    AND legal_hold = false
                    AND status = 'active'
                """, hold.affected_records or [], 
                [cat.value for cat in hold.affected_categories] or [])
                
                for record_row in expired_records:
                    await self._schedule_deletion_job_for_record(record_row)
            
            # Remove from active holds
            del self.active_legal_holds[hold_id]
            
            logger.info(f"Legal hold released: {hold_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to release legal hold: {str(e)}")
            return False
    
    async def execute_deletion_job(self, job_id: str) -> Dict[str, Any]:
        """
        Execute scheduled deletion job
        
        Args:
            job_id: Deletion job identifier
            
        Returns:
            Dict containing execution results
        """
        execution_results = {
            "job_id": job_id,
            "execution_start": datetime.utcnow(),
            "records_processed": 0,
            "records_deleted": 0,
            "records_failed": 0,
            "errors": [],
            "deletion_certificate": None,
            "verification_hash": None
        }
        
        try:
            async with self.db_pool.acquire() as conn:
                # Get deletion job details
                job_row = await conn.fetchrow("""
                    SELECT * FROM deletion_jobs WHERE job_id = $1
                """, job_id)
                
                if not job_row:
                    raise ValueError(f"Deletion job not found: {job_id}")
                
                job = DeletionJob(
                    job_id=job_row['job_id'],
                    policy_id=job_row['policy_id'],
                    scheduled_date=job_row['scheduled_date'],
                    records_to_delete=job_row['records_to_delete'],
                    deletion_method=DeletionMethod(job_row['deletion_method']),
                    approval_status=job_row['approval_status'],
                    approved_by=job_row['approved_by'],
                    approval_date=job_row['approval_date'],
                    execution_status=job_row['execution_status'],
                    execution_date=job_row['execution_date'],
                    deletion_certificate=job_row['deletion_certificate'],
                    verification_hash=job_row['verification_hash']
                )
                
                # Verify job is approved and ready for execution
                if job.approval_status != 'approved':
                    raise ValueError(f"Deletion job not approved: {job_id}")
                
                # Update job status to running
                await conn.execute("""
                    UPDATE deletion_jobs 
                    SET execution_status = 'running', execution_date = $1
                    WHERE job_id = $2
                """, datetime.utcnow(), job_id)
                
                # Process each record for deletion
                for record_id in job.records_to_delete:
                    try:
                        execution_results["records_processed"] += 1
                        
                        # Verify record is still eligible for deletion
                        record_row = await conn.fetchrow("""
                            SELECT * FROM data_retention_records 
                            WHERE record_id = $1 AND legal_hold = false
                        """, record_id)
                        
                        if not record_row:
                            execution_results["errors"].append(f"Record {record_id} not eligible for deletion")
                            execution_results["records_failed"] += 1
                            continue
                        
                        # Execute deletion based on method
                        deletion_success = await self._execute_record_deletion(
                            record_id, job.deletion_method, record_row
                        )
                        
                        if deletion_success:
                            # Update record status
                            await conn.execute("""
                                UPDATE data_retention_records 
                                SET status = 'deleted', deletion_scheduled = $1
                                WHERE record_id = $2
                            """, datetime.utcnow(), record_id)
                            
                            execution_results["records_deleted"] += 1
                        else:
                            execution_results["records_failed"] += 1
                            
                    except Exception as e:
                        execution_results["errors"].append(f"Failed to delete record {record_id}: {str(e)}")
                        execution_results["records_failed"] += 1
                
                # Generate deletion certificate
                certificate_data = {
                    "job_id": job_id,
                    "execution_date": execution_results["execution_start"].isoformat(),
                    "records_deleted": execution_results["records_deleted"],
                    "deletion_method": job.deletion_method.value,
                    "policy_id": job.policy_id
                }
                
                certificate_hash = hashlib.sha256(
                    json.dumps(certificate_data, sort_keys=True).encode()
                ).hexdigest()
                
                execution_results["deletion_certificate"] = certificate_data
                execution_results["verification_hash"] = certificate_hash
                
                # Update job with completion status
                final_status = "completed" if execution_results["records_failed"] == 0 else "partial_failure"
                await conn.execute("""
                    UPDATE deletion_jobs 
                    SET execution_status = $1, deletion_certificate = $2, verification_hash = $3
                    WHERE job_id = $4
                """, final_status, json.dumps(certificate_data), certificate_hash, job_id)
            
            execution_results["execution_end"] = datetime.utcnow()
            execution_results["duration_seconds"] = (
                execution_results["execution_end"] - execution_results["execution_start"]
            ).total_seconds()
            
            logger.info(f"Deletion job completed: {job_id} - {execution_results['records_deleted']} records deleted")
            
        except Exception as e:
            execution_results["errors"].append(f"Job execution failed: {str(e)}")
            logger.error(f"Deletion job failed: {job_id} - {str(e)}")
            
            # Update job status to failed
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE deletion_jobs 
                    SET execution_status = 'failed'
                    WHERE job_id = $1
                """, job_id)
        
        return execution_results
    
    async def _execute_record_deletion(self, record_id: str, method: DeletionMethod, 
                                     record_data: Dict[str, Any]) -> bool:
        """Execute actual data deletion based on specified method"""
        try:
            if method == DeletionMethod.SOFT_DELETE:
                # Mark as deleted but keep data
                return True
                
            elif method == DeletionMethod.HARD_DELETE:
                # Actually delete the data from storage systems
                await self._hard_delete_record_data(record_id, record_data)
                return True
                
            elif method == DeletionMethod.CRYPTOGRAPHIC_ERASURE:
                # Delete encryption keys to make data unrecoverable
                await self._cryptographic_erasure(record_id, record_data)
                return True
                
            elif method == DeletionMethod.ANONYMIZATION:
                # Remove personally identifiable information
                await self._anonymize_record_data(record_id, record_data)
                return True
                
            elif method == DeletionMethod.PHYSICAL_DESTRUCTION:
                # Schedule physical destruction of storage media
                await self._schedule_physical_destruction(record_id, record_data)
                return True
                
            else:
                logger.error(f"Unknown deletion method: {method}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to execute deletion for record {record_id}: {str(e)}")
            return False
    
    async def generate_retention_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive data retention compliance report
        
        Args:
            start_date: Report period start
            end_date: Report period end
            
        Returns:
            Dict containing retention report
        """
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "report_type": "Data Retention Compliance"
            },
            "retention_summary": {},
            "deletion_summary": {},
            "legal_holds": {},
            "policy_compliance": {},
            "upcoming_expirations": [],
            "recommendations": []
        }
        
        async with self.db_pool.acquire() as conn:
            # Retention summary by category
            retention_stats = await conn.fetch("""
                SELECT 
                    data_category,
                    COUNT(*) as total_records,
                    COUNT(*) FILTER (WHERE status = 'active') as active_records,
                    COUNT(*) FILTER (WHERE status = 'expired') as expired_records,
                    COUNT(*) FILTER (WHERE legal_hold = true) as on_hold_records
                FROM data_retention_records 
                WHERE creation_date BETWEEN $1 AND $2
                GROUP BY data_category
            """, start_date, end_date)
            
            report["retention_summary"] = {
                row['data_category']: {
                    "total_records": row['total_records'],
                    "active_records": row['active_records'],
                    "expired_records": row['expired_records'],
                    "on_hold_records": row['on_hold_records']
                }
                for row in retention_stats
            }
            
            # Deletion summary
            deletion_stats = await conn.fetch("""
                SELECT 
                    deletion_method,
                    COUNT(*) as jobs_executed,
                    SUM(array_length(records_to_delete, 1)) as total_records_deleted
                FROM deletion_jobs 
                WHERE execution_date BETWEEN $1 AND $2
                AND execution_status = 'completed'
                GROUP BY deletion_method
            """, start_date, end_date)
            
            report["deletion_summary"] = {
                row['deletion_method']: {
                    "jobs_executed": row['jobs_executed'],
                    "records_deleted": row['total_records_deleted'] or 0
                }
                for row in deletion_stats
            }
            
            # Legal holds summary
            legal_hold_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_holds,
                    COUNT(*) FILTER (WHERE status = 'active') as active_holds,
                    COUNT(*) FILTER (WHERE status = 'released') as released_holds
                FROM legal_holds 
                WHERE initiation_date BETWEEN $1 AND $2
            """, start_date, end_date)
            
            report["legal_holds"] = dict(legal_hold_stats) if legal_hold_stats else {}
            
            # Upcoming expirations (next 30 days)
            upcoming_expirations = await conn.fetch("""
                SELECT 
                    data_category,
                    COUNT(*) as expiring_records,
                    retention_policy_id
                FROM data_retention_records 
                WHERE expiration_date BETWEEN NOW() AND NOW() + INTERVAL '30 days'
                AND status = 'active'
                AND legal_hold = false
                GROUP BY data_category, retention_policy_id
                ORDER BY COUNT(*) DESC
            """)
            
            report["upcoming_expirations"] = [
                {
                    "data_category": row['data_category'],
                    "expiring_records": row['expiring_records'],
                    "policy_id": row['retention_policy_id']
                }
                for row in upcoming_expirations
            ]
        
        # Policy compliance analysis
        for policy_id, policy in self.retention_policies.items():
            compliance_score = await self._calculate_policy_compliance(policy, start_date, end_date)
            report["policy_compliance"][policy_id] = compliance_score
        
        # Generate recommendations
        report["recommendations"] = await self._generate_retention_recommendations(report)
        
        return report

# Database schema for data retention
DATA_RETENTION_SCHEMA_SQL = """
-- Data Retention Policies
CREATE TABLE IF NOT EXISTS data_retention_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    policy_name VARCHAR(255) NOT NULL,
    data_category VARCHAR(50) NOT NULL,
    retention_period_days INTEGER NOT NULL,
    trigger_type VARCHAR(50) NOT NULL,
    deletion_method VARCHAR(50) NOT NULL,
    legal_basis TEXT NOT NULL,
    regulatory_framework TEXT[],
    exceptions TEXT[],
    approval_required BOOLEAN DEFAULT false,
    notification_before_deletion INTEGER DEFAULT 0,
    responsible_party VARCHAR(255) NOT NULL,
    created_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    active BOOLEAN DEFAULT true
);

-- Data Retention Records
CREATE TABLE IF NOT EXISTS data_retention_records (
    record_id VARCHAR(255) PRIMARY KEY,
    data_category VARCHAR(50) NOT NULL,
    creation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    last_accessed TIMESTAMP WITH TIME ZONE NOT NULL,
    retention_policy_id VARCHAR(255) NOT NULL REFERENCES data_retention_policies(policy_id),
    expiration_date TIMESTAMP WITH TIME ZONE NOT NULL,
    legal_hold BOOLEAN DEFAULT false,
    legal_hold_reason TEXT,
    deletion_scheduled TIMESTAMP WITH TIME ZONE,
    deletion_method VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    encrypted_metadata BYTEA,
    geological_context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Legal Holds
CREATE TABLE IF NOT EXISTS legal_holds (
    hold_id VARCHAR(255) PRIMARY KEY,
    hold_name VARCHAR(255) NOT NULL,
    reason TEXT NOT NULL,
    initiated_by VARCHAR(255) NOT NULL,
    initiation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    affected_records TEXT[],
    affected_categories TEXT[],
    expected_end_date TIMESTAMP WITH TIME ZONE,
    actual_end_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Deletion Jobs
CREATE TABLE IF NOT EXISTS deletion_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    policy_id VARCHAR(255) NOT NULL REFERENCES data_retention_policies(policy_id),
    scheduled_date TIMESTAMP WITH TIME ZONE NOT NULL,
    records_to_delete TEXT[] NOT NULL,
    deletion_method VARCHAR(50) NOT NULL,
    approval_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    approved_by VARCHAR(255),
    approval_date TIMESTAMP WITH TIME ZONE,
    execution_status VARCHAR(20) NOT NULL DEFAULT 'scheduled',
    execution_date TIMESTAMP WITH TIME ZONE,
    deletion_certificate JSONB,
    verification_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_retention_records_category ON data_retention_records(data_category);
CREATE INDEX IF NOT EXISTS idx_retention_records_expiration ON data_retention_records(expiration_date);
CREATE INDEX IF NOT EXISTS idx_retention_records_status ON data_retention_records(status);
CREATE INDEX IF NOT EXISTS idx_retention_records_legal_hold ON data_retention_records(legal_hold);
CREATE INDEX IF NOT EXISTS idx_legal_holds_status ON legal_holds(status);
CREATE INDEX IF NOT EXISTS idx_deletion_jobs_scheduled ON deletion_jobs(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_deletion_jobs_status ON deletion_jobs(execution_status);
"""