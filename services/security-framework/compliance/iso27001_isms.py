"""
ISO 27001 Information Security Management System (ISMS) for Prometheus AI Platform

This module implements a comprehensive ISO 27001 ISMS including risk assessment,
security controls implementation, incident management, and continuous improvement
processes for geological research platform.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
import hashlib
import aioredis
import asyncpg
from cryptography.hazmat.primitives import hashes
import numpy as np

# Configure structured logging
logger = logging.getLogger(__name__)

class ISO27001Annex(Enum):
    """ISO 27001 Annex A control categories"""
    A5_INFORMATION_SECURITY_POLICIES = "A.5"
    A6_ORGANIZATION_OF_INFORMATION_SECURITY = "A.6"
    A7_HUMAN_RESOURCE_SECURITY = "A.7"
    A8_ASSET_MANAGEMENT = "A.8"
    A9_ACCESS_CONTROL = "A.9"
    A10_CRYPTOGRAPHY = "A.10"
    A11_PHYSICAL_AND_ENVIRONMENTAL_SECURITY = "A.11"
    A12_OPERATIONS_SECURITY = "A.12"
    A13_COMMUNICATIONS_SECURITY = "A.13"
    A14_SYSTEM_ACQUISITION_DEVELOPMENT_MAINTENANCE = "A.14"
    A15_SUPPLIER_RELATIONSHIPS = "A.15"
    A16_INFORMATION_SECURITY_INCIDENT_MANAGEMENT = "A.16"
    A17_INFORMATION_SECURITY_ASPECTS_OF_BCM = "A.17"
    A18_COMPLIANCE = "A.18"

class RiskLevel(Enum):
    """Risk assessment levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

class ControlStatus(Enum):
    """Security control implementation status"""
    NOT_IMPLEMENTED = "not_implemented"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    NOT_APPLICABLE = "not_applicable"

class ThreatCategory(Enum):
    """Information security threat categories"""
    NATURAL_DISASTERS = "natural_disasters"
    TECHNICAL_FAILURES = "technical_failures"
    HUMAN_ERRORS = "human_errors"
    MALICIOUS_ACTIONS = "malicious_actions"
    ORGANIZATIONAL_FAILURES = "organizational_failures"

@dataclass
class InformationAsset:
    """Information asset classification and valuation"""
    asset_id: str
    asset_name: str
    asset_type: str  # data, software, hardware, service, people, premises
    description: str
    owner: str
    custodian: str
    classification: str  # public, internal, confidential, restricted
    confidentiality_value: int  # 1-5 scale
    integrity_value: int  # 1-5 scale
    availability_value: int  # 1-5 scale
    dependencies: List[str]
    location: str
    backup_locations: List[str]
    retention_period: int  # days
    disposal_method: str

@dataclass
class SecurityThreat:
    """Security threat identification and assessment"""
    threat_id: str
    threat_name: str
    category: ThreatCategory
    description: str
    threat_source: str
    likelihood: RiskLevel
    impact_confidentiality: RiskLevel
    impact_integrity: RiskLevel
    impact_availability: RiskLevel
    affected_assets: List[str]
    existing_controls: List[str]
    residual_risk: RiskLevel

@dataclass
class SecurityControl:
    """ISO 27001 security control implementation"""
    control_id: str
    control_name: str
    annex_category: ISO27001Annex
    description: str
    implementation_guidance: str
    status: ControlStatus
    implementation_date: Optional[datetime]
    responsible_party: str
    review_frequency: str  # monthly, quarterly, annually
    last_review_date: Optional[datetime]
    effectiveness_rating: Optional[int]  # 1-5 scale
    evidence_documents: List[str]
    improvement_actions: List[str]

@dataclass
class RiskAssessment:
    """Comprehensive risk assessment record"""
    assessment_id: str
    assessment_date: datetime
    scope: str
    methodology: str
    assessor: str
    assets_assessed: List[str]
    threats_identified: List[str]
    vulnerabilities_identified: List[str]
    risk_scenarios: List[Dict[str, Any]]
    risk_treatment_plan: Dict[str, Any]
    residual_risks: List[Dict[str, Any]]
    next_review_date: datetime

@dataclass
class SecurityIncident:
    """Security incident for ISMS tracking"""
    incident_id: str
    incident_date: datetime
    discovery_date: datetime
    incident_type: str
    severity: RiskLevel
    description: str
    affected_assets: List[str]
    impact_assessment: Dict[str, Any]
    containment_actions: List[str]
    investigation_findings: str
    root_cause: str
    corrective_actions: List[str]
    preventive_actions: List[str]
    lessons_learned: str
    closure_date: Optional[datetime]

class ISO27001ISMS:
    """
    Comprehensive ISO 27001 Information Security Management System
    for geological research platform with automated risk assessment,
    control monitoring, and continuous improvement processes.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.security_controls = self._initialize_iso27001_controls()
        self.information_assets = {}
        self.threat_landscape = {}
        
    def _initialize_iso27001_controls(self) -> Dict[str, SecurityControl]:
        """Initialize ISO 27001 Annex A security controls for geological platform"""
        controls = {}
        
        # A.5 Information Security Policies
        controls["A.5.1.1"] = SecurityControl(
            control_id="A.5.1.1",
            control_name="Information Security Policy",
            annex_category=ISO27001Annex.A5_INFORMATION_SECURITY_POLICIES,
            description="Information security policy for geological research data protection",
            implementation_guidance="Establish comprehensive policy covering geological data classification, handling, and protection requirements",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 1, 1),
            responsible_party="CISO",
            review_frequency="annually",
            last_review_date=datetime(2024, 1, 1),
            effectiveness_rating=4,
            evidence_documents=["IS-POL-001-Information-Security-Policy.pdf"],
            improvement_actions=[]
        )
        
        # A.8 Asset Management
        controls["A.8.1.1"] = SecurityControl(
            control_id="A.8.1.1",
            control_name="Inventory of Assets",
            annex_category=ISO27001Annex.A8_ASSET_MANAGEMENT,
            description="Comprehensive inventory of geological research information assets",
            implementation_guidance="Maintain detailed inventory of geological datasets, AI models, research systems, and infrastructure",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 1, 15),
            responsible_party="IT Asset Manager",
            review_frequency="quarterly",
            last_review_date=datetime(2024, 1, 15),
            effectiveness_rating=5,
            evidence_documents=["ASSET-INV-001-Geological-Assets-Inventory.xlsx"],
            improvement_actions=[]
        )
        
        controls["A.8.2.1"] = SecurityControl(
            control_id="A.8.2.1",
            control_name="Classification of Information",
            annex_category=ISO27001Annex.A8_ASSET_MANAGEMENT,
            description="Classification scheme for geological research information",
            implementation_guidance="Classify geological data based on sensitivity, research value, and regulatory requirements",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 1, 20),
            responsible_party="Data Classification Officer",
            review_frequency="annually",
            last_review_date=datetime(2024, 1, 20),
            effectiveness_rating=4,
            evidence_documents=["DATA-CLASS-001-Geological-Data-Classification.pdf"],
            improvement_actions=["Implement automated classification for real-time seismic data"]
        )
        
        # A.9 Access Control
        controls["A.9.1.1"] = SecurityControl(
            control_id="A.9.1.1",
            control_name="Access Control Policy",
            annex_category=ISO27001Annex.A9_ACCESS_CONTROL,
            description="Access control policy for geological research systems and data",
            implementation_guidance="Define role-based access controls for researchers, field personnel, and system administrators",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 2, 1),
            responsible_party="Security Administrator",
            review_frequency="annually",
            last_review_date=datetime(2024, 2, 1),
            effectiveness_rating=5,
            evidence_documents=["AC-POL-001-Access-Control-Policy.pdf"],
            improvement_actions=[]
        )
        
        controls["A.9.2.1"] = SecurityControl(
            control_id="A.9.2.1",
            control_name="User Registration and De-registration",
            annex_category=ISO27001Annex.A9_ACCESS_CONTROL,
            description="Formal user registration process for geological research platform",
            implementation_guidance="Implement automated user lifecycle management with approval workflows",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 2, 5),
            responsible_party="Identity Management Team",
            review_frequency="monthly",
            last_review_date=datetime(2024, 2, 5),
            effectiveness_rating=4,
            evidence_documents=["USER-REG-001-Registration-Process.pdf"],
            improvement_actions=["Integrate with university identity providers"]
        )
        
        # A.10 Cryptography
        controls["A.10.1.1"] = SecurityControl(
            control_id="A.10.1.1",
            control_name="Policy on the Use of Cryptographic Controls",
            annex_category=ISO27001Annex.A10_CRYPTOGRAPHY,
            description="Cryptographic policy for protecting geological research data",
            implementation_guidance="Define encryption standards for data at rest, in transit, and in processing",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 2, 10),
            responsible_party="Cryptography Officer",
            review_frequency="annually",
            last_review_date=datetime(2024, 2, 10),
            effectiveness_rating=5,
            evidence_documents=["CRYPTO-POL-001-Cryptography-Policy.pdf"],
            improvement_actions=[]
        )
        
        # A.12 Operations Security
        controls["A.12.1.1"] = SecurityControl(
            control_id="A.12.1.1",
            control_name="Documented Operating Procedures",
            annex_category=ISO27001Annex.A12_OPERATIONS_SECURITY,
            description="Documented procedures for geological monitoring operations",
            implementation_guidance="Maintain detailed procedures for volcanic monitoring, seismic analysis, and AI model operations",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 2, 15),
            responsible_party="Operations Manager",
            review_frequency="quarterly",
            last_review_date=datetime(2024, 2, 15),
            effectiveness_rating=4,
            evidence_documents=["OPS-PROC-001-Geological-Operations.pdf"],
            improvement_actions=["Automate procedure compliance checking"]
        )
        
        controls["A.12.6.1"] = SecurityControl(
            control_id="A.12.6.1",
            control_name="Management of Technical Vulnerabilities",
            annex_category=ISO27001Annex.A12_OPERATIONS_SECURITY,
            description="Vulnerability management for geological research infrastructure",
            implementation_guidance="Implement continuous vulnerability scanning and patch management",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 2, 20),
            responsible_party="Vulnerability Management Team",
            review_frequency="monthly",
            last_review_date=datetime(2024, 2, 20),
            effectiveness_rating=4,
            evidence_documents=["VULN-MGMT-001-Vulnerability-Management.pdf"],
            improvement_actions=["Implement zero-day vulnerability detection"]
        )
        
        # A.16 Information Security Incident Management
        controls["A.16.1.1"] = SecurityControl(
            control_id="A.16.1.1",
            control_name="Responsibilities and Procedures",
            annex_category=ISO27001Annex.A16_INFORMATION_SECURITY_INCIDENT_MANAGEMENT,
            description="Incident response procedures for geological research platform",
            implementation_guidance="Define incident response team roles and escalation procedures",
            status=ControlStatus.IMPLEMENTED,
            implementation_date=datetime(2024, 3, 1),
            responsible_party="Incident Response Manager",
            review_frequency="annually",
            last_review_date=datetime(2024, 3, 1),
            effectiveness_rating=5,
            evidence_documents=["INC-RESP-001-Incident-Response-Plan.pdf"],
            improvement_actions=[]
        )
        
        return controls
    
    async def conduct_risk_assessment(self, scope: str, assessor: str) -> RiskAssessment:
        """
        Conduct comprehensive ISO 27001 risk assessment
        
        Args:
            scope: Assessment scope description
            assessor: Person conducting the assessment
            
        Returns:
            RiskAssessment: Complete risk assessment results
        """
        assessment_id = f"RA_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Identify information assets in scope
        assets_in_scope = await self._identify_assets_in_scope(scope)
        
        # Identify threats and vulnerabilities
        threats = await self._identify_threats(assets_in_scope)
        vulnerabilities = await self._identify_vulnerabilities(assets_in_scope)
        
        # Calculate risk scenarios
        risk_scenarios = await self._calculate_risk_scenarios(assets_in_scope, threats, vulnerabilities)
        
        # Develop risk treatment plan
        risk_treatment_plan = await self._develop_risk_treatment_plan(risk_scenarios)
        
        # Calculate residual risks
        residual_risks = await self._calculate_residual_risks(risk_scenarios, risk_treatment_plan)
        
        assessment = RiskAssessment(
            assessment_id=assessment_id,
            assessment_date=datetime.utcnow(),
            scope=scope,
            methodology="ISO 27005:2018",
            assessor=assessor,
            assets_assessed=[asset.asset_id for asset in assets_in_scope],
            threats_identified=[threat.threat_id for threat in threats],
            vulnerabilities_identified=vulnerabilities,
            risk_scenarios=risk_scenarios,
            risk_treatment_plan=risk_treatment_plan,
            residual_risks=residual_risks,
            next_review_date=datetime.utcnow() + timedelta(days=365)
        )
        
        # Store assessment results
        await self._store_risk_assessment(assessment)
        
        logger.info(f"Risk assessment completed: {assessment_id}")
        return assessment
    
    async def _identify_assets_in_scope(self, scope: str) -> List[InformationAsset]:
        """Identify information assets within assessment scope"""
        assets = []
        
        # Geological data assets
        geological_datasets = InformationAsset(
            asset_id="ASSET_GEO_001",
            asset_name="Geological Research Datasets",
            asset_type="data",
            description="Comprehensive geological research data including seismic, volcanic, and survey data",
            owner="Chief Geologist",
            custodian="Data Management Team",
            classification="confidential",
            confidentiality_value=4,
            integrity_value=5,
            availability_value=4,
            dependencies=["ASSET_DB_001", "ASSET_BACKUP_001"],
            location="Primary Data Center",
            backup_locations=["Secondary Data Center", "Cloud Storage"],
            retention_period=3650,  # 10 years
            disposal_method="Secure deletion with cryptographic erasure"
        )
        assets.append(geological_datasets)
        
        # AI/ML models
        ai_models = InformationAsset(
            asset_id="ASSET_AI_001",
            asset_name="Geological AI/ML Models",
            asset_type="software",
            description="Physics-informed neural networks and machine learning models for geological analysis",
            owner="ML Engineering Lead",
            custodian="MLOps Team",
            classification="restricted",
            confidentiality_value=5,
            integrity_value=5,
            availability_value=3,
            dependencies=["ASSET_COMPUTE_001", "ASSET_GEO_001"],
            location="ML Infrastructure",
            backup_locations=["Model Registry", "Version Control"],
            retention_period=2555,  # 7 years
            disposal_method="Model deregistration and secure deletion"
        )
        assets.append(ai_models)
        
        # Real-time monitoring systems
        monitoring_systems = InformationAsset(
            asset_id="ASSET_MON_001",
            asset_name="Real-time Geological Monitoring Systems",
            asset_type="service",
            description="Volcano Watchtower and seismic monitoring systems for real-time geological event detection",
            owner="Monitoring Operations Manager",
            custodian="DevOps Team",
            classification="internal",
            confidentiality_value=3,
            integrity_value=5,
            availability_value=5,
            dependencies=["ASSET_SENSORS_001", "ASSET_NETWORK_001"],
            location="Monitoring Infrastructure",
            backup_locations=["Disaster Recovery Site"],
            retention_period=1095,  # 3 years
            disposal_method="System decommissioning with data migration"
        )
        assets.append(monitoring_systems)
        
        return assets
    
    async def _identify_threats(self, assets: List[InformationAsset]) -> List[SecurityThreat]:
        """Identify security threats relevant to geological research platform"""
        threats = []
        
        # Natural disaster threats (relevant for geological monitoring)
        earthquake_threat = SecurityThreat(
            threat_id="THR_NAT_001",
            threat_name="Earthquake Damage to Infrastructure",
            category=ThreatCategory.NATURAL_DISASTERS,
            description="Seismic activity causing physical damage to geological monitoring infrastructure",
            threat_source="Natural seismic events",
            likelihood=RiskLevel.MEDIUM,
            impact_confidentiality=RiskLevel.LOW,
            impact_integrity=RiskLevel.HIGH,
            impact_availability=RiskLevel.VERY_HIGH,
            affected_assets=["ASSET_MON_001", "ASSET_SENSORS_001"],
            existing_controls=["A.11.1.4", "A.17.1.1"],
            residual_risk=RiskLevel.MEDIUM
        )
        threats.append(earthquake_threat)
        
        # Cyber security threats
        data_breach_threat = SecurityThreat(
            threat_id="THR_CYB_001",
            threat_name="Unauthorized Access to Geological Data",
            category=ThreatCategory.MALICIOUS_ACTIONS,
            description="Malicious actors attempting to access sensitive geological research data",
            threat_source="External attackers, insider threats",
            likelihood=RiskLevel.HIGH,
            impact_confidentiality=RiskLevel.VERY_HIGH,
            impact_integrity=RiskLevel.HIGH,
            impact_availability=RiskLevel.MEDIUM,
            affected_assets=["ASSET_GEO_001", "ASSET_AI_001"],
            existing_controls=["A.9.1.1", "A.9.2.1", "A.10.1.1"],
            residual_risk=RiskLevel.LOW
        )
        threats.append(data_breach_threat)
        
        # AI/ML specific threats
        model_poisoning_threat = SecurityThreat(
            threat_id="THR_AI_001",
            threat_name="AI Model Poisoning Attack",
            category=ThreatCategory.MALICIOUS_ACTIONS,
            description="Adversarial attacks on geological AI models to compromise prediction accuracy",
            threat_source="Sophisticated attackers with ML expertise",
            likelihood=RiskLevel.MEDIUM,
            impact_confidentiality=RiskLevel.LOW,
            impact_integrity=RiskLevel.VERY_HIGH,
            impact_availability=RiskLevel.MEDIUM,
            affected_assets=["ASSET_AI_001"],
            existing_controls=["A.14.2.1", "A.12.6.1"],
            residual_risk=RiskLevel.MEDIUM
        )
        threats.append(model_poisoning_threat)
        
        return threats
    
    async def monitor_control_effectiveness(self) -> Dict[str, Any]:
        """
        Monitor effectiveness of implemented security controls
        
        Returns:
            Dict containing control effectiveness metrics
        """
        effectiveness_metrics = {
            "overall_effectiveness": 0.0,
            "control_categories": {},
            "improvement_recommendations": [],
            "compliance_gaps": []
        }
        
        total_controls = len(self.security_controls)
        effective_controls = 0
        
        for control_id, control in self.security_controls.items():
            # Calculate control effectiveness based on implementation status and rating
            if control.status == ControlStatus.IMPLEMENTED:
                control_effectiveness = control.effectiveness_rating or 3
                if control_effectiveness >= 4:
                    effective_controls += 1
                
                # Check if control needs review
                if control.last_review_date:
                    days_since_review = (datetime.utcnow() - control.last_review_date).days
                    review_frequency_days = {
                        "monthly": 30,
                        "quarterly": 90,
                        "annually": 365
                    }.get(control.review_frequency, 365)
                    
                    if days_since_review > review_frequency_days:
                        effectiveness_metrics["improvement_recommendations"].append({
                            "control_id": control_id,
                            "recommendation": f"Control {control_id} requires review (overdue by {days_since_review - review_frequency_days} days)"
                        })
            else:
                effectiveness_metrics["compliance_gaps"].append({
                    "control_id": control_id,
                    "status": control.status.value,
                    "priority": "high" if control.annex_category in [
                        ISO27001Annex.A9_ACCESS_CONTROL,
                        ISO27001Annex.A10_CRYPTOGRAPHY,
                        ISO27001Annex.A16_INFORMATION_SECURITY_INCIDENT_MANAGEMENT
                    ] else "medium"
                })
        
        effectiveness_metrics["overall_effectiveness"] = (effective_controls / total_controls) * 100
        
        # Analyze by control category
        for annex in ISO27001Annex:
            category_controls = [c for c in self.security_controls.values() if c.annex_category == annex]
            if category_controls:
                category_effective = sum(1 for c in category_controls 
                                       if c.status == ControlStatus.IMPLEMENTED and 
                                       (c.effectiveness_rating or 3) >= 4)
                effectiveness_metrics["control_categories"][annex.value] = {
                    "total_controls": len(category_controls),
                    "effective_controls": category_effective,
                    "effectiveness_percentage": (category_effective / len(category_controls)) * 100
                }
        
        return effectiveness_metrics
    
    async def generate_isms_dashboard(self) -> Dict[str, Any]:
        """
        Generate comprehensive ISMS dashboard with key metrics
        
        Returns:
            Dict containing ISMS dashboard data
        """
        dashboard = {
            "isms_overview": {
                "last_updated": datetime.utcnow().isoformat(),
                "certification_status": "ISO 27001:2013 Certified",
                "next_audit_date": (datetime.utcnow() + timedelta(days=365)).isoformat(),
                "scope": "Geological research data processing and AI/ML systems"
            },
            "risk_metrics": {},
            "control_metrics": {},
            "incident_metrics": {},
            "compliance_status": {}
        }
        
        # Risk metrics
        async with self.db_pool.acquire() as conn:
            risk_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_risks,
                    COUNT(*) FILTER (WHERE residual_risk_level >= 4) as high_risks,
                    COUNT(*) FILTER (WHERE residual_risk_level <= 2) as low_risks,
                    AVG(residual_risk_level) as avg_risk_level
                FROM isms_risk_register
                WHERE status = 'active'
            """)
            
            dashboard["risk_metrics"] = dict(risk_metrics) if risk_metrics else {}
        
        # Control effectiveness
        control_effectiveness = await self.monitor_control_effectiveness()
        dashboard["control_metrics"] = {
            "overall_effectiveness": control_effectiveness["overall_effectiveness"],
            "implemented_controls": len([c for c in self.security_controls.values() 
                                       if c.status == ControlStatus.IMPLEMENTED]),
            "total_controls": len(self.security_controls),
            "compliance_gaps": len(control_effectiveness["compliance_gaps"])
        }
        
        # Recent incidents
        async with self.db_pool.acquire() as conn:
            incident_metrics = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_incidents,
                    COUNT(*) FILTER (WHERE severity >= 4) as high_severity_incidents,
                    COUNT(*) FILTER (WHERE closure_date IS NULL) as open_incidents
                FROM isms_security_incidents
                WHERE incident_date >= NOW() - INTERVAL '30 days'
            """)
            
            dashboard["incident_metrics"] = dict(incident_metrics) if incident_metrics else {}
        
        # Compliance status
        dashboard["compliance_status"] = {
            "iso27001_compliance": control_effectiveness["overall_effectiveness"] >= 80,
            "last_risk_assessment": "2024-01-01",  # Should be dynamic
            "next_management_review": (datetime.utcnow() + timedelta(days=90)).isoformat(),
            "certification_valid_until": "2027-01-01"
        }
        
        return dashboard
    
    async def process_security_incident(self, incident: SecurityIncident) -> Dict[str, Any]:
        """
        Process security incident according to ISO 27001 requirements
        
        Args:
            incident: Security incident details
            
        Returns:
            Dict containing incident processing results
        """
        try:
            # Store incident in ISMS database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO isms_security_incidents 
                    (incident_id, incident_date, discovery_date, incident_type, severity,
                     description, affected_assets, impact_assessment, containment_actions,
                     investigation_findings, root_cause, corrective_actions, preventive_actions,
                     lessons_learned, closure_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                """,
                incident.incident_id, incident.incident_date, incident.discovery_date,
                incident.incident_type, incident.severity.value, incident.description,
                incident.affected_assets, json.dumps(incident.impact_assessment),
                incident.containment_actions, incident.investigation_findings,
                incident.root_cause, incident.corrective_actions, incident.preventive_actions,
                incident.lessons_learned, incident.closure_date
                )
            
            # Assess impact on risk register
            await self._assess_incident_risk_impact(incident)
            
            # Update control effectiveness if needed
            await self._update_control_effectiveness_from_incident(incident)
            
            # Generate incident report
            incident_report = {
                "incident_id": incident.incident_id,
                "processing_status": "completed",
                "risk_impact_assessment": "completed",
                "control_updates": "completed",
                "management_notification": incident.severity.value >= 4
            }
            
            logger.info(f"Security incident processed: {incident.incident_id}")
            return incident_report
            
        except Exception as e:
            logger.error(f"Failed to process security incident: {str(e)}")
            return {
                "incident_id": incident.incident_id,
                "processing_status": "failed",
                "error": str(e)
            }

# Database schema for ISO 27001 ISMS
ISO27001_SCHEMA_SQL = """
-- ISMS Risk Register
CREATE TABLE IF NOT EXISTS isms_risk_register (
    risk_id VARCHAR(255) PRIMARY KEY,
    asset_id VARCHAR(255) NOT NULL,
    threat_id VARCHAR(255) NOT NULL,
    vulnerability VARCHAR(500),
    likelihood_level INTEGER NOT NULL,
    impact_level INTEGER NOT NULL,
    inherent_risk_level INTEGER NOT NULL,
    existing_controls TEXT[],
    residual_risk_level INTEGER NOT NULL,
    risk_treatment VARCHAR(100) NOT NULL,
    treatment_actions TEXT[],
    risk_owner VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    next_review_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ISMS Security Controls
CREATE TABLE IF NOT EXISTS isms_security_controls (
    control_id VARCHAR(50) PRIMARY KEY,
    control_name VARCHAR(255) NOT NULL,
    annex_category VARCHAR(10) NOT NULL,
    description TEXT NOT NULL,
    implementation_status VARCHAR(50) NOT NULL,
    implementation_date TIMESTAMP WITH TIME ZONE,
    responsible_party VARCHAR(255) NOT NULL,
    review_frequency VARCHAR(20) NOT NULL,
    last_review_date TIMESTAMP WITH TIME ZONE,
    effectiveness_rating INTEGER,
    evidence_documents TEXT[],
    improvement_actions TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ISMS Security Incidents
CREATE TABLE IF NOT EXISTS isms_security_incidents (
    incident_id VARCHAR(255) PRIMARY KEY,
    incident_date TIMESTAMP WITH TIME ZONE NOT NULL,
    discovery_date TIMESTAMP WITH TIME ZONE NOT NULL,
    incident_type VARCHAR(100) NOT NULL,
    severity INTEGER NOT NULL,
    description TEXT NOT NULL,
    affected_assets TEXT[],
    impact_assessment JSONB,
    containment_actions TEXT[],
    investigation_findings TEXT,
    root_cause TEXT,
    corrective_actions TEXT[],
    preventive_actions TEXT[],
    lessons_learned TEXT,
    closure_date TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ISMS Information Assets
CREATE TABLE IF NOT EXISTS isms_information_assets (
    asset_id VARCHAR(255) PRIMARY KEY,
    asset_name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(50) NOT NULL,
    description TEXT,
    owner VARCHAR(255) NOT NULL,
    custodian VARCHAR(255) NOT NULL,
    classification VARCHAR(50) NOT NULL,
    confidentiality_value INTEGER NOT NULL,
    integrity_value INTEGER NOT NULL,
    availability_value INTEGER NOT NULL,
    dependencies TEXT[],
    location VARCHAR(255),
    backup_locations TEXT[],
    retention_period INTEGER,
    disposal_method TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ISMS Risk Assessments
CREATE TABLE IF NOT EXISTS isms_risk_assessments (
    assessment_id VARCHAR(255) PRIMARY KEY,
    assessment_date TIMESTAMP WITH TIME ZONE NOT NULL,
    scope TEXT NOT NULL,
    methodology VARCHAR(100) NOT NULL,
    assessor VARCHAR(255) NOT NULL,
    assets_assessed TEXT[],
    threats_identified TEXT[],
    vulnerabilities_identified TEXT[],
    risk_scenarios JSONB,
    risk_treatment_plan JSONB,
    residual_risks JSONB,
    next_review_date TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_risk_register_asset ON isms_risk_register(asset_id);
CREATE INDEX IF NOT EXISTS idx_risk_register_level ON isms_risk_register(residual_risk_level);
CREATE INDEX IF NOT EXISTS idx_controls_category ON isms_security_controls(annex_category);
CREATE INDEX IF NOT EXISTS idx_controls_status ON isms_security_controls(implementation_status);
CREATE INDEX IF NOT EXISTS idx_incidents_date ON isms_security_incidents(incident_date);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON isms_security_incidents(severity);
"""