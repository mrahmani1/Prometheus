"""
GDPR Compliance Framework for Prometheus AI Platform

This module implements comprehensive GDPR compliance including data subject rights,
consent management, privacy by design, and automated compliance monitoring.
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import aioredis
import asyncpg
from pydantic import BaseModel, Field

# Configure structured logging
logger = logging.getLogger(__name__)

class DataSubjectRights(Enum):
    """GDPR Data Subject Rights enumeration"""
    ACCESS = "access"  # Article 15
    RECTIFICATION = "rectification"  # Article 16
    ERASURE = "erasure"  # Article 17 (Right to be forgotten)
    RESTRICT_PROCESSING = "restrict_processing"  # Article 18
    DATA_PORTABILITY = "data_portability"  # Article 20
    OBJECT = "object"  # Article 21
    WITHDRAW_CONSENT = "withdraw_consent"  # Article 7(3)

class ConsentStatus(Enum):
    """Consent status tracking"""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"

class ProcessingLawfulBasis(Enum):
    """GDPR Article 6 lawful basis for processing"""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"

@dataclass
class PersonalDataCategory:
    """Classification of personal data categories"""
    category_id: str
    name: str
    description: str
    sensitivity_level: str  # low, medium, high, special
    retention_period: int  # days
    lawful_basis: ProcessingLawfulBasis
    special_category: bool = False  # Article 9 special categories

@dataclass
class ConsentRecord:
    """Consent management record"""
    consent_id: str
    data_subject_id: str
    purpose: str
    lawful_basis: ProcessingLawfulBasis
    consent_text: str
    timestamp: datetime
    status: ConsentStatus
    expiry_date: Optional[datetime]
    withdrawal_date: Optional[datetime]
    ip_address: str
    user_agent: str
    consent_method: str  # web_form, api, email, etc.

@dataclass
class DataSubjectRequest:
    """Data subject rights request tracking"""
    request_id: str
    data_subject_id: str
    request_type: DataSubjectRights
    request_date: datetime
    status: str  # pending, in_progress, completed, rejected
    completion_date: Optional[datetime]
    response_data: Optional[Dict[str, Any]]
    verification_method: str
    notes: str

class GDPRComplianceEngine:
    """
    Comprehensive GDPR compliance engine with automated data subject rights,
    consent management, and privacy by design implementation.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis, encryption_key: bytes):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.cipher_suite = Fernet(encryption_key)
        self.personal_data_categories = self._initialize_data_categories()
        
    def _initialize_data_categories(self) -> Dict[str, PersonalDataCategory]:
        """Initialize personal data categories for geological research platform"""
        categories = {
            "user_profile": PersonalDataCategory(
                category_id="user_profile",
                name="User Profile Data",
                description="Basic user identification and contact information",
                sensitivity_level="medium",
                retention_period=2555,  # 7 years
                lawful_basis=ProcessingLawfulBasis.CONTRACT
            ),
            "research_data": PersonalDataCategory(
                category_id="research_data",
                name="Research Collaboration Data",
                description="Scientific research collaboration and publication data",
                sensitivity_level="low",
                retention_period=3650,  # 10 years for scientific records
                lawful_basis=ProcessingLawfulBasis.LEGITIMATE_INTERESTS
            ),
            "location_data": PersonalDataCategory(
                category_id="location_data",
                name="Geological Field Location Data",
                description="GPS coordinates and field work locations",
                sensitivity_level="high",
                retention_period=1095,  # 3 years
                lawful_basis=ProcessingLawfulBasis.CONSENT
            ),
            "biometric_data": PersonalDataCategory(
                category_id="biometric_data",
                name="Biometric Authentication Data",
                description="Fingerprint and facial recognition for secure access",
                sensitivity_level="special",
                retention_period=365,  # 1 year
                lawful_basis=ProcessingLawfulBasis.CONSENT,
                special_category=True
            )
        }
        return categories
    
    async def record_consent(self, consent_record: ConsentRecord) -> bool:
        """
        Record user consent with cryptographic proof and audit trail
        
        Args:
            consent_record: Detailed consent information
            
        Returns:
            bool: Success status of consent recording
        """
        try:
            # Encrypt sensitive consent data
            encrypted_consent = self.cipher_suite.encrypt(
                json.dumps(asdict(consent_record)).encode()
            )
            
            # Store in database with audit trail
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO gdpr_consent_records 
                    (consent_id, data_subject_id, purpose, lawful_basis, 
                     consent_text, timestamp, status, expiry_date, 
                     ip_address, user_agent, consent_method, encrypted_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                consent_record.consent_id,
                consent_record.data_subject_id,
                consent_record.purpose,
                consent_record.lawful_basis.value,
                consent_record.consent_text,
                consent_record.timestamp,
                consent_record.status.value,
                consent_record.expiry_date,
                consent_record.ip_address,
                consent_record.user_agent,
                consent_record.consent_method,
                encrypted_consent
                )
            
            # Cache consent status for quick lookup
            await self.redis_client.setex(
                f"consent:{consent_record.data_subject_id}:{consent_record.purpose}",
                86400,  # 24 hours
                consent_record.status.value
            )
            
            logger.info(f"Consent recorded for subject {consent_record.data_subject_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record consent: {str(e)}")
            return False
    
    async def process_data_subject_request(self, request: DataSubjectRequest) -> Dict[str, Any]:
        """
        Process data subject rights requests with automated workflows
        
        Args:
            request: Data subject rights request
            
        Returns:
            Dict containing request processing results
        """
        try:
            # Verify data subject identity
            if not await self._verify_data_subject_identity(request.data_subject_id):
                return {
                    "status": "rejected",
                    "reason": "Identity verification failed",
                    "request_id": request.request_id
                }
            
            # Process based on request type
            if request.request_type == DataSubjectRights.ACCESS:
                response_data = await self._process_access_request(request.data_subject_id)
            elif request.request_type == DataSubjectRights.ERASURE:
                response_data = await self._process_erasure_request(request.data_subject_id)
            elif request.request_type == DataSubjectRights.RECTIFICATION:
                response_data = await self._process_rectification_request(request)
            elif request.request_type == DataSubjectRights.DATA_PORTABILITY:
                response_data = await self._process_portability_request(request.data_subject_id)
            elif request.request_type == DataSubjectRights.RESTRICT_PROCESSING:
                response_data = await self._process_restriction_request(request.data_subject_id)
            elif request.request_type == DataSubjectRights.OBJECT:
                response_data = await self._process_objection_request(request.data_subject_id)
            elif request.request_type == DataSubjectRights.WITHDRAW_CONSENT:
                response_data = await self._process_consent_withdrawal(request.data_subject_id)
            else:
                return {
                    "status": "rejected",
                    "reason": "Unsupported request type",
                    "request_id": request.request_id
                }
            
            # Update request status
            request.status = "completed"
            request.completion_date = datetime.utcnow()
            request.response_data = response_data
            
            # Store request record
            await self._store_request_record(request)
            
            return {
                "status": "completed",
                "request_id": request.request_id,
                "completion_date": request.completion_date.isoformat(),
                "response_data": response_data
            }
            
        except Exception as e:
            logger.error(f"Failed to process data subject request: {str(e)}")
            return {
                "status": "error",
                "reason": str(e),
                "request_id": request.request_id
            }
    
    async def _process_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Process Article 15 access request - provide all personal data"""
        personal_data = {}
        
        async with self.db_pool.acquire() as conn:
            # Collect user profile data
            user_data = await conn.fetchrow("""
                SELECT user_id, email, name, created_at, last_login,
                       research_affiliations, geological_specializations
                FROM users WHERE user_id = $1
            """, data_subject_id)
            
            if user_data:
                personal_data["profile"] = dict(user_data)
            
            # Collect research data
            research_data = await conn.fetch("""
                SELECT project_id, project_name, role, participation_date,
                       data_contributions, publication_references
                FROM research_participation WHERE user_id = $1
            """, data_subject_id)
            
            personal_data["research_activities"] = [dict(row) for row in research_data]
            
            # Collect location data
            location_data = await conn.fetch("""
                SELECT location_id, latitude, longitude, timestamp,
                       field_site_name, geological_context
                FROM field_locations WHERE recorded_by = $1
            """, data_subject_id)
            
            personal_data["location_data"] = [dict(row) for row in location_data]
            
            # Collect consent records
            consent_data = await conn.fetch("""
                SELECT consent_id, purpose, timestamp, status, expiry_date
                FROM gdpr_consent_records WHERE data_subject_id = $1
            """, data_subject_id)
            
            personal_data["consent_records"] = [dict(row) for row in consent_data]
        
        return {
            "data_categories": list(personal_data.keys()),
            "personal_data": personal_data,
            "processing_purposes": await self._get_processing_purposes(data_subject_id),
            "data_retention_periods": self._get_retention_periods(),
            "third_party_recipients": await self._get_third_party_recipients(data_subject_id)
        }
    
    async def _process_erasure_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Process Article 17 erasure request - right to be forgotten"""
        deleted_records = {}
        
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                # Check for legal obligations preventing deletion
                legal_holds = await conn.fetch("""
                    SELECT table_name, reason, retention_end_date
                    FROM legal_hold_records 
                    WHERE data_subject_id = $1 AND retention_end_date > NOW()
                """, data_subject_id)
                
                if legal_holds:
                    return {
                        "status": "partial_deletion",
                        "reason": "Legal obligations prevent complete deletion",
                        "legal_holds": [dict(row) for row in legal_holds]
                    }
                
                # Anonymize user profile (keep for research integrity)
                await conn.execute("""
                    UPDATE users SET 
                        email = 'anonymized_' || user_id || '@deleted.local',
                        name = 'Anonymized User',
                        phone = NULL,
                        address = NULL,
                        date_of_birth = NULL
                    WHERE user_id = $1
                """, data_subject_id)
                deleted_records["user_profile"] = "anonymized"
                
                # Delete location data (if consent withdrawn)
                location_count = await conn.fetchval("""
                    DELETE FROM field_locations 
                    WHERE recorded_by = $1 
                    RETURNING COUNT(*)
                """, data_subject_id)
                deleted_records["location_records"] = location_count
                
                # Delete biometric data
                biometric_count = await conn.fetchval("""
                    DELETE FROM biometric_auth_data 
                    WHERE user_id = $1 
                    RETURNING COUNT(*)
                """, data_subject_id)
                deleted_records["biometric_records"] = biometric_count
                
                # Mark consent as withdrawn
                await conn.execute("""
                    UPDATE gdpr_consent_records 
                    SET status = 'withdrawn', withdrawal_date = NOW()
                    WHERE data_subject_id = $1
                """, data_subject_id)
        
        # Clear cached data
        await self.redis_client.delete(f"user_profile:{data_subject_id}")
        
        return {
            "status": "completed",
            "deleted_records": deleted_records,
            "anonymized_data": ["research_contributions", "scientific_publications"]
        }
    
    async def check_consent_validity(self, data_subject_id: str, purpose: str) -> bool:
        """
        Check if valid consent exists for specific processing purpose
        
        Args:
            data_subject_id: Unique identifier for data subject
            purpose: Processing purpose to check consent for
            
        Returns:
            bool: True if valid consent exists
        """
        try:
            # Check cache first
            cached_consent = await self.redis_client.get(
                f"consent:{data_subject_id}:{purpose}"
            )
            
            if cached_consent:
                return cached_consent.decode() == ConsentStatus.GIVEN.value
            
            # Check database
            async with self.db_pool.acquire() as conn:
                consent_record = await conn.fetchrow("""
                    SELECT status, expiry_date 
                    FROM gdpr_consent_records 
                    WHERE data_subject_id = $1 AND purpose = $2 
                    ORDER BY timestamp DESC LIMIT 1
                """, data_subject_id, purpose)
                
                if not consent_record:
                    return False
                
                # Check if consent is valid and not expired
                if consent_record['status'] != ConsentStatus.GIVEN.value:
                    return False
                
                if (consent_record['expiry_date'] and 
                    consent_record['expiry_date'] < datetime.utcnow()):
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to check consent validity: {str(e)}")
            return False
    
    async def generate_privacy_impact_assessment(self, processing_activity: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate automated Privacy Impact Assessment (PIA) for new processing activities
        
        Args:
            processing_activity: Details of the new processing activity
            
        Returns:
            Dict containing PIA results and recommendations
        """
        pia_results = {
            "assessment_id": f"pia_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "processing_activity": processing_activity,
            "risk_assessment": {},
            "mitigation_measures": [],
            "compliance_status": "pending_review"
        }
        
        # Assess data protection risks
        risk_factors = []
        
        # Check for special category data
        if processing_activity.get("special_categories", False):
            risk_factors.append({
                "risk": "Special category data processing",
                "severity": "high",
                "article": "Article 9 GDPR",
                "mitigation": "Explicit consent or legal exception required"
            })
        
        # Check for automated decision making
        if processing_activity.get("automated_decisions", False):
            risk_factors.append({
                "risk": "Automated decision making",
                "severity": "medium",
                "article": "Article 22 GDPR",
                "mitigation": "Human intervention and explanation rights required"
            })
        
        # Check for international transfers
        if processing_activity.get("international_transfers", False):
            risk_factors.append({
                "risk": "International data transfers",
                "severity": "medium",
                "article": "Chapter V GDPR",
                "mitigation": "Adequacy decision or appropriate safeguards required"
            })
        
        pia_results["risk_assessment"] = {
            "risk_factors": risk_factors,
            "overall_risk_level": self._calculate_overall_risk(risk_factors),
            "dpo_consultation_required": len(risk_factors) > 2
        }
        
        return pia_results
    
    def _calculate_overall_risk(self, risk_factors: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on individual risk factors"""
        if not risk_factors:
            return "low"
        
        high_risks = sum(1 for risk in risk_factors if risk["severity"] == "high")
        medium_risks = sum(1 for risk in risk_factors if risk["severity"] == "medium")
        
        if high_risks > 0:
            return "high"
        elif medium_risks > 2:
            return "high"
        elif medium_risks > 0:
            return "medium"
        else:
            return "low"
    
    async def monitor_compliance_metrics(self) -> Dict[str, Any]:
        """
        Monitor GDPR compliance metrics and generate automated reports
        
        Returns:
            Dict containing compliance metrics and status
        """
        metrics = {}
        
        async with self.db_pool.acquire() as conn:
            # Data subject requests metrics
            request_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(*) FILTER (WHERE status = 'completed') as completed_requests,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending_requests,
                    AVG(EXTRACT(EPOCH FROM (completion_date - request_date))/86400) as avg_response_days
                FROM gdpr_data_subject_requests 
                WHERE request_date >= NOW() - INTERVAL '30 days'
            """)
            
            metrics["data_subject_requests"] = dict(request_metrics)
            
            # Consent metrics
            consent_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_consents,
                    COUNT(*) FILTER (WHERE status = 'given') as active_consents,
                    COUNT(*) FILTER (WHERE status = 'withdrawn') as withdrawn_consents,
                    COUNT(*) FILTER (WHERE expiry_date < NOW()) as expired_consents
                FROM gdpr_consent_records
            """)
            
            metrics["consent_management"] = dict(consent_metrics)
            
            # Data breach incidents
            breach_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_breaches,
                    COUNT(*) FILTER (WHERE severity = 'high') as high_severity_breaches,
                    COUNT(*) FILTER (WHERE reported_to_authority = true) as reported_breaches
                FROM data_breach_incidents 
                WHERE incident_date >= NOW() - INTERVAL '12 months'
            """)
            
            metrics["data_breaches"] = dict(breach_metrics)
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(metrics)
        metrics["compliance_score"] = compliance_score
        
        return metrics

    def _calculate_compliance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall GDPR compliance score based on metrics"""
        score = 100.0
        
        # Deduct points for delayed responses
        if metrics["data_subject_requests"]["avg_response_days"] > 30:
            score -= 20
        elif metrics["data_subject_requests"]["avg_response_days"] > 15:
            score -= 10
        
        # Deduct points for pending requests
        pending_ratio = (metrics["data_subject_requests"]["pending_requests"] / 
                        max(metrics["data_subject_requests"]["total_requests"], 1))
        if pending_ratio > 0.2:
            score -= 15
        
        # Deduct points for data breaches
        if metrics["data_breaches"]["high_severity_breaches"] > 0:
            score -= 25
        
        return max(score, 0.0)

# Database schema initialization
GDPR_SCHEMA_SQL = """
-- GDPR Consent Records
CREATE TABLE IF NOT EXISTS gdpr_consent_records (
    consent_id VARCHAR(255) PRIMARY KEY,
    data_subject_id VARCHAR(255) NOT NULL,
    purpose VARCHAR(500) NOT NULL,
    lawful_basis VARCHAR(50) NOT NULL,
    consent_text TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) NOT NULL,
    expiry_date TIMESTAMP WITH TIME ZONE,
    withdrawal_date TIMESTAMP WITH TIME ZONE,
    ip_address INET,
    user_agent TEXT,
    consent_method VARCHAR(50),
    encrypted_data BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data Subject Rights Requests
CREATE TABLE IF NOT EXISTS gdpr_data_subject_requests (
    request_id VARCHAR(255) PRIMARY KEY,
    data_subject_id VARCHAR(255) NOT NULL,
    request_type VARCHAR(50) NOT NULL,
    request_date TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    completion_date TIMESTAMP WITH TIME ZONE,
    verification_method VARCHAR(100),
    notes TEXT,
    response_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Legal Hold Records
CREATE TABLE IF NOT EXISTS legal_hold_records (
    hold_id VARCHAR(255) PRIMARY KEY,
    data_subject_id VARCHAR(255) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    reason TEXT NOT NULL,
    retention_end_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data Breach Incidents
CREATE TABLE IF NOT EXISTS data_breach_incidents (
    incident_id VARCHAR(255) PRIMARY KEY,
    incident_date TIMESTAMP WITH TIME ZONE NOT NULL,
    severity VARCHAR(20) NOT NULL,
    affected_records INTEGER,
    data_categories TEXT[],
    breach_description TEXT,
    containment_measures TEXT,
    reported_to_authority BOOLEAN DEFAULT FALSE,
    notification_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_consent_subject_purpose ON gdpr_consent_records(data_subject_id, purpose);
CREATE INDEX IF NOT EXISTS idx_consent_status ON gdpr_consent_records(status);
CREATE INDEX IF NOT EXISTS idx_requests_subject ON gdpr_data_subject_requests(data_subject_id);
CREATE INDEX IF NOT EXISTS idx_requests_status ON gdpr_data_subject_requests(status);
"""