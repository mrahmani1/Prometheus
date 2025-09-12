"""
SOC 2 Type II Compliance Framework for Prometheus AI Platform

This module implements comprehensive SOC 2 Type II compliance controls across
all five trust service criteria: Security, Availability, Processing Integrity,
Confidentiality, and Privacy.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import hashlib
import aioredis
import asyncpg
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding

# Configure structured logging
logger = logging.getLogger(__name__)

class SOC2Criteria(Enum):
    """SOC 2 Trust Service Criteria"""
    SECURITY = "security"
    AVAILABILITY = "availability"
    PROCESSING_INTEGRITY = "processing_integrity"
    CONFIDENTIALITY = "confidentiality"
    PRIVACY = "privacy"

class ControlType(Enum):
    """SOC 2 Control Types"""
    PREVENTIVE = "preventive"
    DETECTIVE = "detective"
    CORRECTIVE = "corrective"

class ControlFrequency(Enum):
    """Control testing frequency"""
    CONTINUOUS = "continuous"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

@dataclass
class SOC2Control:
    """SOC 2 Control definition and tracking"""
    control_id: str
    control_name: str
    criteria: SOC2Criteria
    control_type: ControlType
    description: str
    control_objective: str
    testing_frequency: ControlFrequency
    automated: bool
    responsible_party: str
    implementation_date: datetime
    last_tested: Optional[datetime]
    test_results: List[Dict[str, Any]]
    exceptions: List[Dict[str, Any]]
    remediation_actions: List[Dict[str, Any]]

@dataclass
class ControlTestResult:
    """Control testing result record"""
    test_id: str
    control_id: str
    test_date: datetime
    tester: str
    test_method: str
    test_result: str  # effective, ineffective, not_applicable
    evidence: List[str]
    exceptions_noted: List[str]
    management_response: Optional[str]
    remediation_timeline: Optional[datetime]

@dataclass
class SecurityIncident:
    """Security incident tracking for SOC 2"""
    incident_id: str
    incident_date: datetime
    severity: str  # critical, high, medium, low
    category: str
    description: str
    affected_systems: List[str]
    data_involved: bool
    customer_impact: bool
    detection_method: str
    response_actions: List[str]
    resolution_date: Optional[datetime]
    lessons_learned: str
    preventive_measures: List[str]

class SOC2ComplianceEngine:
    """
    Comprehensive SOC 2 Type II compliance engine with automated control testing,
    continuous monitoring, and evidence collection for geological research platform.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.controls = self._initialize_soc2_controls()
        
    def _initialize_soc2_controls(self) -> Dict[str, SOC2Control]:
        """Initialize SOC 2 controls specific to geological research platform"""
        controls = {}
        
        # Security Controls (CC6.0 series)
        controls["CC6.1"] = SOC2Control(
            control_id="CC6.1",
            control_name="Logical Access Security",
            criteria=SOC2Criteria.SECURITY,
            control_type=ControlType.PREVENTIVE,
            description="Logical access to geological data and systems is restricted through authentication and authorization procedures",
            control_objective="Ensure only authorized users can access geological research data and AI models",
            testing_frequency=ControlFrequency.CONTINUOUS,
            automated=True,
            responsible_party="Security Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        controls["CC6.2"] = SOC2Control(
            control_id="CC6.2",
            control_name="User Access Provisioning",
            criteria=SOC2Criteria.SECURITY,
            control_type=ControlType.PREVENTIVE,
            description="User access to geological systems is provisioned based on job responsibilities and approved by management",
            control_objective="Ensure appropriate access levels for geological researchers and system administrators",
            testing_frequency=ControlFrequency.MONTHLY,
            automated=False,
            responsible_party="IT Administration",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        controls["CC6.3"] = SOC2Control(
            control_id="CC6.3",
            control_name="User Access Modification and Termination",
            criteria=SOC2Criteria.SECURITY,
            control_type=ControlType.PREVENTIVE,
            description="User access modifications and terminations are processed timely and completely",
            control_objective="Ensure timely removal of access for departed researchers and role changes",
            testing_frequency=ControlFrequency.WEEKLY,
            automated=True,
            responsible_party="HR and IT Teams",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        # Availability Controls (A1.0 series)
        controls["A1.1"] = SOC2Control(
            control_id="A1.1",
            control_name="System Availability Monitoring",
            criteria=SOC2Criteria.AVAILABILITY,
            control_type=ControlType.DETECTIVE,
            description="System availability is monitored continuously with automated alerting for geological monitoring systems",
            control_objective="Ensure 99.9% uptime for critical volcanic monitoring and earthquake detection systems",
            testing_frequency=ControlFrequency.CONTINUOUS,
            automated=True,
            responsible_party="DevOps Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        controls["A1.2"] = SOC2Control(
            control_id="A1.2",
            control_name="Capacity Management",
            criteria=SOC2Criteria.AVAILABILITY,
            control_type=ControlType.PREVENTIVE,
            description="System capacity is monitored and managed to handle geological data processing loads",
            control_objective="Ensure adequate capacity for real-time seismic data processing and AI model inference",
            testing_frequency=ControlFrequency.DAILY,
            automated=True,
            responsible_party="Infrastructure Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        # Processing Integrity Controls (PI1.0 series)
        controls["PI1.1"] = SOC2Control(
            control_id="PI1.1",
            control_name="Data Processing Accuracy",
            criteria=SOC2Criteria.PROCESSING_INTEGRITY,
            control_type=ControlType.DETECTIVE,
            description="Geological data processing accuracy is validated through automated checks and scientific validation",
            control_objective="Ensure accurate processing of seismic, volcanic, and geological survey data",
            testing_frequency=ControlFrequency.CONTINUOUS,
            automated=True,
            responsible_party="Data Science Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        controls["PI1.2"] = SOC2Control(
            control_id="PI1.2",
            control_name="AI Model Validation",
            criteria=SOC2Criteria.PROCESSING_INTEGRITY,
            control_type=ControlType.DETECTIVE,
            description="AI model outputs are validated against known geological phenomena and peer-reviewed research",
            control_objective="Ensure AI predictions for volcanic eruptions and seismic events meet scientific accuracy standards",
            testing_frequency=ControlFrequency.WEEKLY,
            automated=True,
            responsible_party="ML Engineering Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        # Confidentiality Controls (C1.0 series)
        controls["C1.1"] = SOC2Control(
            control_id="C1.1",
            control_name="Data Encryption",
            criteria=SOC2Criteria.CONFIDENTIALITY,
            control_type=ControlType.PREVENTIVE,
            description="Sensitive geological research data is encrypted in transit and at rest",
            control_objective="Protect proprietary geological models and research data from unauthorized disclosure",
            testing_frequency=ControlFrequency.CONTINUOUS,
            automated=True,
            responsible_party="Security Team",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        # Privacy Controls (P1.0 series)
        controls["P1.1"] = SOC2Control(
            control_id="P1.1",
            control_name="Personal Data Protection",
            criteria=SOC2Criteria.PRIVACY,
            control_type=ControlType.PREVENTIVE,
            description="Personal data of researchers and field personnel is protected according to privacy regulations",
            control_objective="Ensure GDPR compliance for researcher personal data and field location information",
            testing_frequency=ControlFrequency.MONTHLY,
            automated=False,
            responsible_party="Privacy Officer",
            implementation_date=datetime(2024, 1, 1),
            last_tested=None,
            test_results=[],
            exceptions=[],
            remediation_actions=[]
        )
        
        return controls
    
    async def execute_control_test(self, control_id: str, tester: str) -> ControlTestResult:
        """
        Execute automated or manual control test and record results
        
        Args:
            control_id: SOC 2 control identifier
            tester: Person or system executing the test
            
        Returns:
            ControlTestResult: Test execution results
        """
        control = self.controls.get(control_id)
        if not control:
            raise ValueError(f"Control {control_id} not found")
        
        test_result = ControlTestResult(
            test_id=f"test_{control_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            control_id=control_id,
            test_date=datetime.utcnow(),
            tester=tester,
            test_method="automated" if control.automated else "manual",
            test_result="effective",  # Will be updated based on actual test
            evidence=[],
            exceptions_noted=[],
            management_response=None,
            remediation_timeline=None
        )
        
        try:
            # Execute control-specific tests
            if control_id == "CC6.1":
                test_result = await self._test_logical_access_security(test_result)
            elif control_id == "CC6.2":
                test_result = await self._test_user_access_provisioning(test_result)
            elif control_id == "CC6.3":
                test_result = await self._test_access_modification_termination(test_result)
            elif control_id == "A1.1":
                test_result = await self._test_availability_monitoring(test_result)
            elif control_id == "A1.2":
                test_result = await self._test_capacity_management(test_result)
            elif control_id == "PI1.1":
                test_result = await self._test_data_processing_accuracy(test_result)
            elif control_id == "PI1.2":
                test_result = await self._test_ai_model_validation(test_result)
            elif control_id == "C1.1":
                test_result = await self._test_data_encryption(test_result)
            elif control_id == "P1.1":
                test_result = await self._test_personal_data_protection(test_result)
            
            # Store test results
            await self._store_control_test_result(test_result)
            
            # Update control last tested date
            control.last_tested = test_result.test_date
            control.test_results.append(asdict(test_result))
            
            logger.info(f"Control test completed: {control_id} - {test_result.test_result}")
            
        except Exception as e:
            test_result.test_result = "ineffective"
            test_result.exceptions_noted.append(f"Test execution error: {str(e)}")
            logger.error(f"Control test failed: {control_id} - {str(e)}")
        
        return test_result
    
    async def _test_logical_access_security(self, test_result: ControlTestResult) -> ControlTestResult:
        """Test CC6.1 - Logical Access Security"""
        async with self.db_pool.acquire() as conn:
            # Test 1: Verify all active users have valid authentication
            invalid_auth_users = await conn.fetch("""
                SELECT user_id, last_login, account_status 
                FROM users 
                WHERE account_status = 'active' 
                AND (password_hash IS NULL OR password_hash = '')
            """)
            
            if invalid_auth_users:
                test_result.exceptions_noted.append(
                    f"Found {len(invalid_auth_users)} active users without valid authentication"
                )
                test_result.test_result = "ineffective"
            
            # Test 2: Verify MFA is enabled for privileged accounts
            privileged_no_mfa = await conn.fetch("""
                SELECT u.user_id, u.email, r.role_name
                FROM users u
                JOIN user_roles ur ON u.user_id = ur.user_id
                JOIN roles r ON ur.role_id = r.role_id
                WHERE r.role_name IN ('admin', 'system_admin', 'security_admin')
                AND u.mfa_enabled = false
            """)
            
            if privileged_no_mfa:
                test_result.exceptions_noted.append(
                    f"Found {len(privileged_no_mfa)} privileged accounts without MFA"
                )
                test_result.test_result = "ineffective"
            
            # Test 3: Verify session timeout configuration
            session_config = await conn.fetchrow("""
                SELECT config_value 
                FROM system_config 
                WHERE config_key = 'session_timeout_minutes'
            """)
            
            if not session_config or int(session_config['config_value']) > 480:  # 8 hours max
                test_result.exceptions_noted.append("Session timeout exceeds security policy")
                test_result.test_result = "ineffective"
            
            test_result.evidence.append("Authentication configuration review completed")
            test_result.evidence.append("MFA enforcement verification completed")
            test_result.evidence.append("Session management validation completed")
        
        return test_result
    
    async def _test_availability_monitoring(self, test_result: ControlTestResult) -> ControlTestResult:
        """Test A1.1 - System Availability Monitoring"""
        # Test monitoring system health
        monitoring_metrics = await self.redis_client.hgetall("system_availability_metrics")
        
        if not monitoring_metrics:
            test_result.exceptions_noted.append("No availability metrics found in monitoring system")
            test_result.test_result = "ineffective"
            return test_result
        
        # Check uptime metrics
        uptime_percentage = float(monitoring_metrics.get(b'uptime_percentage', 0))
        if uptime_percentage < 99.9:
            test_result.exceptions_noted.append(f"System uptime {uptime_percentage}% below SLA requirement of 99.9%")
            test_result.test_result = "ineffective"
        
        # Check alert system functionality
        async with self.db_pool.acquire() as conn:
            recent_alerts = await conn.fetch("""
                SELECT alert_type, alert_time, resolution_time
                FROM system_alerts 
                WHERE alert_time >= NOW() - INTERVAL '24 hours'
                AND alert_type = 'availability'
            """)
            
            # Verify alerts are being generated and resolved
            unresolved_alerts = [alert for alert in recent_alerts if not alert['resolution_time']]
            if len(unresolved_alerts) > 5:
                test_result.exceptions_noted.append(f"Found {len(unresolved_alerts)} unresolved availability alerts")
                test_result.test_result = "ineffective"
        
        test_result.evidence.append("Availability metrics validation completed")
        test_result.evidence.append("Alert system functionality verified")
        
        return test_result
    
    async def _test_data_processing_accuracy(self, test_result: ControlTestResult) -> ControlTestResult:
        """Test PI1.1 - Data Processing Accuracy"""
        async with self.db_pool.acquire() as conn:
            # Test data validation checksums
            checksum_failures = await conn.fetch("""
                SELECT data_batch_id, processing_date, validation_errors
                FROM data_processing_logs 
                WHERE processing_date >= NOW() - INTERVAL '7 days'
                AND checksum_valid = false
            """)
            
            if checksum_failures:
                test_result.exceptions_noted.append(
                    f"Found {len(checksum_failures)} data batches with checksum validation failures"
                )
                test_result.test_result = "ineffective"
            
            # Test geological data accuracy against known benchmarks
            accuracy_metrics = await conn.fetchrow("""
                SELECT 
                    AVG(seismic_accuracy) as avg_seismic_accuracy,
                    AVG(volcanic_prediction_accuracy) as avg_volcanic_accuracy,
                    COUNT(*) as total_validations
                FROM geological_validation_results 
                WHERE validation_date >= NOW() - INTERVAL '30 days'
            """)
            
            if (accuracy_metrics['avg_seismic_accuracy'] < 0.95 or 
                accuracy_metrics['avg_volcanic_accuracy'] < 0.85):
                test_result.exceptions_noted.append("Geological data processing accuracy below acceptable thresholds")
                test_result.test_result = "ineffective"
            
            test_result.evidence.append("Data checksum validation completed")
            test_result.evidence.append("Geological accuracy benchmarking completed")
        
        return test_result
    
    async def generate_soc2_report(self, report_period_start: datetime, report_period_end: datetime) -> Dict[str, Any]:
        """
        Generate comprehensive SOC 2 Type II compliance report
        
        Args:
            report_period_start: Start of reporting period
            report_period_end: End of reporting period
            
        Returns:
            Dict containing complete SOC 2 report
        """
        report = {
            "report_metadata": {
                "report_type": "SOC 2 Type II",
                "report_period": {
                    "start": report_period_start.isoformat(),
                    "end": report_period_end.isoformat()
                },
                "entity_name": "Prometheus AI Platform",
                "service_description": "AI-powered geological monitoring and research platform",
                "generation_date": datetime.utcnow().isoformat()
            },
            "trust_service_criteria": {},
            "control_testing_results": {},
            "exceptions_and_deficiencies": [],
            "management_responses": [],
            "complementary_user_entity_controls": []
        }
        
        # Analyze each trust service criteria
        for criteria in SOC2Criteria:
            criteria_controls = [c for c in self.controls.values() if c.criteria == criteria]
            criteria_results = await self._analyze_criteria_effectiveness(
                criteria_controls, report_period_start, report_period_end
            )
            report["trust_service_criteria"][criteria.value] = criteria_results
        
        # Compile control testing results
        for control_id, control in self.controls.items():
            control_results = await self._compile_control_results(
                control, report_period_start, report_period_end
            )
            report["control_testing_results"][control_id] = control_results
        
        # Identify exceptions and deficiencies
        exceptions = await self._identify_control_exceptions(report_period_start, report_period_end)
        report["exceptions_and_deficiencies"] = exceptions
        
        # Generate executive summary
        report["executive_summary"] = await self._generate_executive_summary(report)
        
        return report
    
    async def _analyze_criteria_effectiveness(self, controls: List[SOC2Control], 
                                           start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze effectiveness of controls for a specific trust service criteria"""
        total_controls = len(controls)
        effective_controls = 0
        
        for control in controls:
            # Check if control was tested during the period
            period_tests = [
                test for test in control.test_results 
                if start_date <= datetime.fromisoformat(test['test_date']) <= end_date
            ]
            
            if period_tests:
                # Control is effective if majority of tests passed
                effective_tests = [test for test in period_tests if test['test_result'] == 'effective']
                if len(effective_tests) / len(period_tests) >= 0.8:
                    effective_controls += 1
        
        effectiveness_percentage = (effective_controls / total_controls * 100) if total_controls > 0 else 0
        
        return {
            "total_controls": total_controls,
            "effective_controls": effective_controls,
            "effectiveness_percentage": effectiveness_percentage,
            "criteria_opinion": "effective" if effectiveness_percentage >= 80 else "ineffective"
        }
    
    async def monitor_continuous_controls(self):
        """
        Continuous monitoring of automated SOC 2 controls
        This method should be run as a background task
        """
        while True:
            try:
                # Test all continuous controls
                continuous_controls = [
                    control_id for control_id, control in self.controls.items()
                    if control.testing_frequency == ControlFrequency.CONTINUOUS
                ]
                
                for control_id in continuous_controls:
                    await self.execute_control_test(control_id, "automated_system")
                
                # Wait 5 minutes before next cycle
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in continuous control monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def record_security_incident(self, incident: SecurityIncident) -> bool:
        """
        Record security incident for SOC 2 reporting
        
        Args:
            incident: Security incident details
            
        Returns:
            bool: Success status
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO soc2_security_incidents 
                    (incident_id, incident_date, severity, category, description,
                     affected_systems, data_involved, customer_impact, detection_method,
                     response_actions, resolution_date, lessons_learned, preventive_measures)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                incident.incident_id, incident.incident_date, incident.severity,
                incident.category, incident.description, incident.affected_systems,
                incident.data_involved, incident.customer_impact, incident.detection_method,
                incident.response_actions, incident.resolution_date,
                incident.lessons_learned, incident.preventive_measures
                )
            
            # Update control effectiveness if incident indicates control failure
            await self._assess_incident_control_impact(incident)
            
            logger.info(f"Security incident recorded: {incident.incident_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record security incident: {str(e)}")
            return False

# Database schema for SOC 2 compliance
SOC2_SCHEMA_SQL = """
-- SOC 2 Control Test Results
CREATE TABLE IF NOT EXISTS soc2_control_test_results (
    test_id VARCHAR(255) PRIMARY KEY,
    control_id VARCHAR(50) NOT NULL,
    test_date TIMESTAMP WITH TIME ZONE NOT NULL,
    tester VARCHAR(255) NOT NULL,
    test_method VARCHAR(50) NOT NULL,
    test_result VARCHAR(20) NOT NULL,
    evidence TEXT[],
    exceptions_noted TEXT[],
    management_response TEXT,
    remediation_timeline TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- SOC 2 Security Incidents
CREATE TABLE IF NOT EXISTS soc2_security_incidents (
    incident_id VARCHAR(255) PRIMARY KEY,
    incident_date TIMESTAMP WITH TIME ZONE NOT NULL,
    severity VARCHAR(20) NOT NULL,
    category VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    affected_systems TEXT[],
    data_involved BOOLEAN NOT NULL,
    customer_impact BOOLEAN NOT NULL,
    detection_method VARCHAR(100),
    response_actions TEXT[],
    resolution_date TIMESTAMP WITH TIME ZONE,
    lessons_learned TEXT,
    preventive_measures TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System Availability Metrics
CREATE TABLE IF NOT EXISTS system_availability_metrics (
    metric_id SERIAL PRIMARY KEY,
    measurement_time TIMESTAMP WITH TIME ZONE NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    availability_percentage DECIMAL(5,2) NOT NULL,
    downtime_minutes INTEGER DEFAULT 0,
    incident_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Data Processing Validation Logs
CREATE TABLE IF NOT EXISTS data_processing_logs (
    log_id SERIAL PRIMARY KEY,
    data_batch_id VARCHAR(255) NOT NULL,
    processing_date TIMESTAMP WITH TIME ZONE NOT NULL,
    checksum_valid BOOLEAN NOT NULL,
    validation_errors TEXT[],
    processing_duration_seconds INTEGER,
    records_processed INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Geological Validation Results
CREATE TABLE IF NOT EXISTS geological_validation_results (
    validation_id SERIAL PRIMARY KEY,
    validation_date TIMESTAMP WITH TIME ZONE NOT NULL,
    validation_type VARCHAR(100) NOT NULL,
    seismic_accuracy DECIMAL(4,3),
    volcanic_prediction_accuracy DECIMAL(4,3),
    benchmark_dataset VARCHAR(255),
    validation_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_control_tests_date ON soc2_control_test_results(test_date);
CREATE INDEX IF NOT EXISTS idx_control_tests_control ON soc2_control_test_results(control_id);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON soc2_security_incidents(incident_date);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON soc2_security_incidents(severity);
"""