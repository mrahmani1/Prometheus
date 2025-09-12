"""
Automated Compliance Monitoring and Reporting System for Prometheus AI Platform

This module implements real-time compliance monitoring, automated policy enforcement,
regulatory change management, and comprehensive compliance dashboards.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
import aioredis
import asyncpg
from collections import defaultdict
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd

# Configure structured logging
logger = logging.getLogger(__name__)

class ComplianceStatus(Enum):
    """Compliance status levels"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    AT_RISK = "at_risk"
    UNKNOWN = "unknown"

class PolicyType(Enum):
    """Types of compliance policies"""
    DATA_PROTECTION = "data_protection"
    ACCESS_CONTROL = "access_control"
    RETENTION = "retention"
    ENCRYPTION = "encryption"
    AUDIT_LOGGING = "audit_logging"
    INCIDENT_RESPONSE = "incident_response"
    BUSINESS_CONTINUITY = "business_continuity"
    VENDOR_MANAGEMENT = "vendor_management"

class MonitoringFrequency(Enum):
    """Monitoring frequency options"""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CompliancePolicy:
    """Compliance policy definition"""
    policy_id: str
    policy_name: str
    policy_type: PolicyType
    framework: str  # GDPR, SOC2, ISO27001, etc.
    description: str
    requirements: List[str]
    monitoring_rules: List[Dict[str, Any]]
    automated_checks: List[str]
    remediation_actions: List[str]
    responsible_party: str
    review_frequency: MonitoringFrequency
    last_review_date: Optional[datetime]
    next_review_date: datetime
    active: bool

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    policy_id: str
    detection_time: datetime
    violation_type: str
    severity: AlertSeverity
    description: str
    affected_resources: List[str]
    evidence: Dict[str, Any]
    risk_score: int
    auto_remediated: bool
    remediation_actions_taken: List[str]
    status: str  # open, investigating, resolved, false_positive
    assigned_to: Optional[str]
    resolution_time: Optional[datetime]
    root_cause: Optional[str]

@dataclass
class ComplianceMetric:
    """Compliance metric tracking"""
    metric_id: str
    metric_name: str
    framework: str
    measurement_time: datetime
    value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    trend: str  # improving, stable, declining
    context: Dict[str, Any]

@dataclass
class RegulatoryChange:
    """Regulatory change tracking"""
    change_id: str
    regulation_name: str
    change_type: str  # new_requirement, modification, clarification
    effective_date: datetime
    description: str
    impact_assessment: Dict[str, Any]
    affected_policies: List[str]
    implementation_plan: List[str]
    implementation_status: str
    responsible_party: str
    deadline: datetime

class ComplianceMonitoringEngine:
    """
    Comprehensive compliance monitoring engine with real-time policy enforcement,
    automated violation detection, and regulatory change management for geological
    research platform.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.policies = {}
        self.monitoring_tasks = {}
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.alert_handlers = {}
        
    async def initialize_compliance_policies(self):
        """Initialize compliance policies for geological research platform"""
        policies = [
            # GDPR Data Protection Policies
            CompliancePolicy(
                policy_id="GDPR_001",
                policy_name="Personal Data Processing Consent",
                policy_type=PolicyType.DATA_PROTECTION,
                framework="GDPR",
                description="Ensure valid consent exists for all personal data processing activities",
                requirements=[
                    "Explicit consent required for special category data",
                    "Consent must be freely given, specific, informed, and unambiguous",
                    "Consent can be withdrawn at any time"
                ],
                monitoring_rules=[
                    {
                        "rule": "check_consent_validity",
                        "frequency": "real_time",
                        "parameters": {"max_age_days": 365}
                    }
                ],
                automated_checks=["consent_expiry_check", "consent_withdrawal_processing"],
                remediation_actions=["suspend_processing", "request_renewed_consent"],
                responsible_party="Data Protection Officer",
                review_frequency=MonitoringFrequency.MONTHLY,
                last_review_date=datetime(2024, 1, 1),
                next_review_date=datetime(2024, 2, 1),
                active=True
            ),
            
            # SOC2 Access Control Policies
            CompliancePolicy(
                policy_id="SOC2_001",
                policy_name="Privileged Access Management",
                policy_type=PolicyType.ACCESS_CONTROL,
                framework="SOC2",
                description="Control and monitor privileged access to geological research systems",
                requirements=[
                    "Multi-factor authentication for all privileged accounts",
                    "Regular access reviews and certifications",
                    "Principle of least privilege enforcement"
                ],
                monitoring_rules=[
                    {
                        "rule": "check_privileged_access",
                        "frequency": "daily",
                        "parameters": {"max_inactive_days": 30}
                    }
                ],
                automated_checks=["mfa_enforcement", "access_certification", "privilege_escalation_detection"],
                remediation_actions=["disable_account", "require_mfa", "escalate_to_security"],
                responsible_party="Security Administrator",
                review_frequency=MonitoringFrequency.WEEKLY,
                last_review_date=datetime(2024, 1, 1),
                next_review_date=datetime(2024, 1, 8),
                active=True
            ),
            
            # ISO27001 Information Security Policies
            CompliancePolicy(
                policy_id="ISO27001_001",
                policy_name="Information Classification and Handling",
                policy_type=PolicyType.DATA_PROTECTION,
                framework="ISO27001",
                description="Ensure proper classification and handling of geological research information",
                requirements=[
                    "All information assets must be classified",
                    "Handling procedures must match classification level",
                    "Regular classification reviews required"
                ],
                monitoring_rules=[
                    {
                        "rule": "check_data_classification",
                        "frequency": "daily",
                        "parameters": {"unclassified_threshold": 5}
                    }
                ],
                automated_checks=["classification_compliance", "handling_procedure_adherence"],
                remediation_actions=["classify_data", "apply_protection_controls"],
                responsible_party="Information Security Manager",
                review_frequency=MonitoringFrequency.MONTHLY,
                last_review_date=datetime(2024, 1, 1),
                next_review_date=datetime(2024, 2, 1),
                active=True
            )
        ]
        
        for policy in policies:
            self.policies[policy.policy_id] = policy
            await self._store_policy(policy)
    
    async def start_compliance_monitoring(self):
        """Start real-time compliance monitoring tasks"""
        for policy_id, policy in self.policies.items():
            if policy.active:
                for rule in policy.monitoring_rules:
                    task_name = f"{policy_id}_{rule['rule']}"
                    if rule['frequency'] == 'real_time':
                        self.monitoring_tasks[task_name] = asyncio.create_task(
                            self._real_time_monitoring(policy, rule)
                        )
                    elif rule['frequency'] == 'hourly':
                        self.monitoring_tasks[task_name] = asyncio.create_task(
                            self._scheduled_monitoring(policy, rule, 3600)
                        )
                    elif rule['frequency'] == 'daily':
                        self.monitoring_tasks[task_name] = asyncio.create_task(
                            self._scheduled_monitoring(policy, rule, 86400)
                        )
        
        logger.info(f"Started {len(self.monitoring_tasks)} compliance monitoring tasks")
    
    async def _real_time_monitoring(self, policy: CompliancePolicy, rule: Dict[str, Any]):
        """Real-time compliance monitoring using Redis streams"""
        stream_key = f"compliance_events:{policy.framework.lower()}"
        
        while True:
            try:
                # Listen for compliance-related events
                events = await self.redis_client.xread({stream_key: '$'}, block=1000)
                
                for stream, messages in events:
                    for message_id, fields in messages:
                        event_data = {k.decode(): v.decode() for k, v in fields.items()}
                        
                        # Check if event violates policy
                        violation = await self._check_policy_violation(policy, rule, event_data)
                        if violation:
                            await self._handle_compliance_violation(violation)
                
            except Exception as e:
                logger.error(f"Error in real-time monitoring for {policy.policy_id}: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _scheduled_monitoring(self, policy: CompliancePolicy, rule: Dict[str, Any], interval: int):
        """Scheduled compliance monitoring"""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Execute scheduled compliance check
                violations = await self._execute_scheduled_check(policy, rule)
                
                for violation in violations:
                    await self._handle_compliance_violation(violation)
                
            except Exception as e:
                logger.error(f"Error in scheduled monitoring for {policy.policy_id}: {str(e)}")
    
    async def _check_policy_violation(self, policy: CompliancePolicy, rule: Dict[str, Any], 
                                    event_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check if an event violates a compliance policy"""
        
        if rule['rule'] == 'check_consent_validity':
            return await self._check_consent_violation(policy, rule, event_data)
        elif rule['rule'] == 'check_privileged_access':
            return await self._check_access_violation(policy, rule, event_data)
        elif rule['rule'] == 'check_data_classification':
            return await self._check_classification_violation(policy, rule, event_data)
        
        return None
    
    async def _check_consent_violation(self, policy: CompliancePolicy, rule: Dict[str, Any], 
                                     event_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check for GDPR consent violations"""
        if event_data.get('event_type') != 'data_access':
            return None
        
        user_id = event_data.get('user_id')
        data_type = event_data.get('data_type')
        
        if not user_id or not data_type:
            return None
        
        # Check if valid consent exists
        async with self.db_pool.acquire() as conn:
            consent_record = await conn.fetchrow("""
                SELECT status, expiry_date 
                FROM gdpr_consent_records 
                WHERE data_subject_id = $1 AND purpose = $2 
                ORDER BY timestamp DESC LIMIT 1
            """, user_id, data_type)
            
            if not consent_record or consent_record['status'] != 'given':
                return ComplianceViolation(
                    violation_id=f"GDPR_CONSENT_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    policy_id=policy.policy_id,
                    detection_time=datetime.utcnow(),
                    violation_type="missing_consent",
                    severity=AlertSeverity.HIGH,
                    description=f"Data access without valid consent for user {user_id}",
                    affected_resources=[event_data.get('resource', 'unknown')],
                    evidence=event_data,
                    risk_score=8,
                    auto_remediated=False,
                    remediation_actions_taken=[],
                    status="open",
                    assigned_to=policy.responsible_party,
                    resolution_time=None,
                    root_cause=None
                )
        
        return None
    
    async def _handle_compliance_violation(self, violation: ComplianceViolation):
        """Handle detected compliance violation"""
        try:
            # Store violation in database
            await self._store_violation(violation)
            
            # Attempt automated remediation
            if violation.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                remediated = await self._attempt_auto_remediation(violation)
                if remediated:
                    violation.auto_remediated = True
                    violation.status = "auto_resolved"
            
            # Send alerts
            await self._send_compliance_alert(violation)
            
            # Update compliance metrics
            await self._update_compliance_metrics(violation)
            
            logger.warning(f"Compliance violation detected: {violation.violation_id}")
            
        except Exception as e:
            logger.error(f"Failed to handle compliance violation: {str(e)}")
    
    async def _attempt_auto_remediation(self, violation: ComplianceViolation) -> bool:
        """Attempt automated remediation of compliance violation"""
        policy = self.policies.get(violation.policy_id)
        if not policy:
            return False
        
        remediation_success = False
        
        for action in policy.remediation_actions:
            try:
                if action == "suspend_processing":
                    # Suspend data processing for affected resources
                    await self._suspend_data_processing(violation.affected_resources)
                    violation.remediation_actions_taken.append("Data processing suspended")
                    remediation_success = True
                    
                elif action == "disable_account":
                    # Disable user account if access violation
                    if violation.violation_type == "unauthorized_access":
                        user_id = violation.evidence.get('user_id')
                        if user_id:
                            await self._disable_user_account(user_id)
                            violation.remediation_actions_taken.append(f"Account {user_id} disabled")
                            remediation_success = True
                
                elif action == "classify_data":
                    # Apply default classification to unclassified data
                    for resource in violation.affected_resources:
                        await self._apply_default_classification(resource)
                        violation.remediation_actions_taken.append(f"Default classification applied to {resource}")
                        remediation_success = True
                
            except Exception as e:
                logger.error(f"Auto-remediation action '{action}' failed: {str(e)}")
        
        return remediation_success
    
    async def generate_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate comprehensive compliance dashboard"""
        dashboard = {
            "dashboard_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "reporting_period": "last_30_days",
                "frameworks_monitored": list(set(p.framework for p in self.policies.values()))
            },
            "overall_compliance_score": 0.0,
            "framework_scores": {},
            "active_violations": {},
            "compliance_trends": {},
            "policy_effectiveness": {},
            "upcoming_deadlines": [],
            "recommendations": []
        }
        
        # Calculate overall compliance score
        total_policies = len([p for p in self.policies.values() if p.active])
        compliant_policies = 0
        
        async with self.db_pool.acquire() as conn:
            # Get violation counts by framework
            framework_violations = await conn.fetch("""
                SELECT p.framework, COUNT(v.violation_id) as violation_count
                FROM compliance_policies p
                LEFT JOIN compliance_violations v ON p.policy_id = v.policy_id
                WHERE v.detection_time >= NOW() - INTERVAL '30 days'
                AND v.status IN ('open', 'investigating')
                GROUP BY p.framework
            """)
            
            framework_violation_dict = {row['framework']: row['violation_count'] for row in framework_violations}
            
            # Calculate framework scores
            for framework in dashboard["dashboard_metadata"]["frameworks_monitored"]:
                framework_policies = [p for p in self.policies.values() if p.framework == framework and p.active]
                violations = framework_violation_dict.get(framework, 0)
                
                if framework_policies:
                    # Simple scoring: 100% - (violations / policies * 10)
                    score = max(0, 100 - (violations / len(framework_policies) * 10))
                    dashboard["framework_scores"][framework] = round(score, 2)
                    
                    if score >= 80:
                        compliant_policies += len(framework_policies)
            
            # Overall compliance score
            dashboard["overall_compliance_score"] = round(
                (compliant_policies / total_policies * 100) if total_policies > 0 else 0, 2
            )
            
            # Active violations by severity
            violation_counts = await conn.fetch("""
                SELECT severity, COUNT(*) as count
                FROM compliance_violations 
                WHERE status IN ('open', 'investigating')
                AND detection_time >= NOW() - INTERVAL '30 days'
                GROUP BY severity
            """)
            
            dashboard["active_violations"] = {
                row['severity']: row['count'] for row in violation_counts
            }
            
            # Compliance trends (last 6 months)
            trend_data = await conn.fetch("""
                SELECT 
                    DATE_TRUNC('month', detection_time) as month,
                    COUNT(*) as violations
                FROM compliance_violations 
                WHERE detection_time >= NOW() - INTERVAL '6 months'
                GROUP BY DATE_TRUNC('month', detection_time)
                ORDER BY month
            """)
            
            dashboard["compliance_trends"] = {
                row['month'].strftime('%Y-%m'): row['violations'] for row in trend_data
            }
        
        # Policy effectiveness analysis
        for policy_id, policy in self.policies.items():
            if policy.active:
                effectiveness = await self._calculate_policy_effectiveness(policy)
                dashboard["policy_effectiveness"][policy_id] = effectiveness
        
        # Upcoming compliance deadlines
        dashboard["upcoming_deadlines"] = await self._get_upcoming_deadlines()
        
        # Generate recommendations
        dashboard["recommendations"] = await self._generate_compliance_recommendations(dashboard)
        
        return dashboard
    
    async def _calculate_policy_effectiveness(self, policy: CompliancePolicy) -> Dict[str, Any]:
        """Calculate effectiveness metrics for a compliance policy"""
        async with self.db_pool.acquire() as conn:
            # Get violation statistics for this policy
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(*) FILTER (WHERE auto_remediated = true) as auto_remediated,
                    AVG(EXTRACT(EPOCH FROM (resolution_time - detection_time))/3600) as avg_resolution_hours
                FROM compliance_violations 
                WHERE policy_id = $1 
                AND detection_time >= NOW() - INTERVAL '30 days'
            """, policy.policy_id)
        
        effectiveness_score = 100.0
        
        # Reduce score based on violations
        if stats['total_violations'] > 0:
            effectiveness_score -= min(stats['total_violations'] * 5, 50)
        
        # Increase score for auto-remediation
        if stats['auto_remediated'] > 0:
            auto_remediation_rate = stats['auto_remediated'] / max(stats['total_violations'], 1)
            effectiveness_score += auto_remediation_rate * 20
        
        return {
            "effectiveness_score": max(0, min(100, effectiveness_score)),
            "total_violations": stats['total_violations'] or 0,
            "auto_remediation_rate": (stats['auto_remediated'] / max(stats['total_violations'], 1)) * 100 if stats['total_violations'] else 0,
            "avg_resolution_hours": float(stats['avg_resolution_hours']) if stats['avg_resolution_hours'] else 0
        }
    
    async def track_regulatory_change(self, change: RegulatoryChange):
        """Track and manage regulatory changes"""
        try:
            # Store regulatory change
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO regulatory_changes 
                    (change_id, regulation_name, change_type, effective_date, description,
                     impact_assessment, affected_policies, implementation_plan, 
                     implementation_status, responsible_party, deadline)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                change.change_id, change.regulation_name, change.change_type,
                change.effective_date, change.description, json.dumps(change.impact_assessment),
                change.affected_policies, change.implementation_plan,
                change.implementation_status, change.responsible_party, change.deadline
                )
            
            # Update affected policies
            for policy_id in change.affected_policies:
                if policy_id in self.policies:
                    policy = self.policies[policy_id]
                    policy.next_review_date = min(policy.next_review_date, change.effective_date)
                    await self._store_policy(policy)
            
            # Create implementation tasks
            await self._create_implementation_tasks(change)
            
            logger.info(f"Regulatory change tracked: {change.change_id}")
            
        except Exception as e:
            logger.error(f"Failed to track regulatory change: {str(e)}")

# Database schema for compliance monitoring
COMPLIANCE_MONITORING_SCHEMA_SQL = """
-- Compliance Policies
CREATE TABLE IF NOT EXISTS compliance_policies (
    policy_id VARCHAR(255) PRIMARY KEY,
    policy_name VARCHAR(255) NOT NULL,
    policy_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    requirements TEXT[],
    monitoring_rules JSONB,
    automated_checks TEXT[],
    remediation_actions TEXT[],
    responsible_party VARCHAR(255) NOT NULL,
    review_frequency VARCHAR(20) NOT NULL,
    last_review_date TIMESTAMP WITH TIME ZONE,
    next_review_date TIMESTAMP WITH TIME ZONE NOT NULL,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance Violations
CREATE TABLE IF NOT EXISTS compliance_violations (
    violation_id VARCHAR(255) PRIMARY KEY,
    policy_id VARCHAR(255) NOT NULL REFERENCES compliance_policies(policy_id),
    detection_time TIMESTAMP WITH TIME ZONE NOT NULL,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT NOT NULL,
    affected_resources TEXT[],
    evidence JSONB,
    risk_score INTEGER NOT NULL,
    auto_remediated BOOLEAN DEFAULT false,
    remediation_actions_taken TEXT[],
    status VARCHAR(50) NOT NULL DEFAULT 'open',
    assigned_to VARCHAR(255),
    resolution_time TIMESTAMP WITH TIME ZONE,
    root_cause TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance Metrics
CREATE TABLE IF NOT EXISTS compliance_metrics (
    metric_id VARCHAR(255) PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    target_value DECIMAL(10,2) NOT NULL,
    threshold_warning DECIMAL(10,2),
    threshold_critical DECIMAL(10,2),
    unit VARCHAR(50),
    trend VARCHAR(20),
    context JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Regulatory Changes
CREATE TABLE IF NOT EXISTS regulatory_changes (
    change_id VARCHAR(255) PRIMARY KEY,
    regulation_name VARCHAR(255) NOT NULL,
    change_type VARCHAR(50) NOT NULL,
    effective_date TIMESTAMP WITH TIME ZONE NOT NULL,
    description TEXT NOT NULL,
    impact_assessment JSONB,
    affected_policies TEXT[],
    implementation_plan TEXT[],
    implementation_status VARCHAR(50) NOT NULL,
    responsible_party VARCHAR(255) NOT NULL,
    deadline TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_policies_framework ON compliance_policies(framework);
CREATE INDEX IF NOT EXISTS idx_policies_active ON compliance_policies(active);
CREATE INDEX IF NOT EXISTS idx_violations_policy ON compliance_violations(policy_id);
CREATE INDEX IF NOT EXISTS idx_violations_status ON compliance_violations(status);
CREATE INDEX IF NOT EXISTS idx_violations_detection_time ON compliance_violations(detection_time);
CREATE INDEX IF NOT EXISTS idx_violations_severity ON compliance_violations(severity);
CREATE INDEX IF NOT EXISTS idx_metrics_framework ON compliance_metrics(framework);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON compliance_metrics(measurement_time);
CREATE INDEX IF NOT EXISTS idx_regulatory_changes_effective ON regulatory_changes(effective_date);
"""