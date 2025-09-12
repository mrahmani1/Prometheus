"""
Security Models for Zero-Trust Architecture with Geological Domain Expertise
Comprehensive data models for advanced security framework
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

class ThreatLevel(Enum):
    """Threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class GeologicalDomain(Enum):
    """Geological domain specializations"""
    VOLCANOLOGY = "volcanology"
    SEISMOLOGY = "seismology"
    GEOCHEMISTRY = "geochemistry"
    STRUCTURAL_GEOLOGY = "structural_geology"
    PETROLOGY = "petrology"
    MINERALOGY = "mineralogy"
    GEOPHYSICS = "geophysics"
    HYDROGEOLOGY = "hydrogeology"
    TECTONICS = "tectonics"
    FIELD_GEOLOGY = "field_geology"

class SecurityEventType(Enum):
    """Security event types"""
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_GRANTED = "authz_granted"
    AUTHORIZATION_DENIED = "authz_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    POLICY_VIOLATION = "policy_violation"
    DATA_ACCESS = "data_access"
    SYSTEM_BREACH = "system_breach"
    GEOLOGICAL_ANOMALY = "geological_anomaly"
    EMERGENCY_ACCESS = "emergency_access"

@dataclass
class GeologicalContext:
    """Geological context for security decisions"""
    primary_domain: Optional[GeologicalDomain] = None
    secondary_domains: List[GeologicalDomain] = field(default_factory=list)
    experience_level: int = 1  # 1-5 scale
    certifications: List[str] = field(default_factory=list)
    current_project: Optional[str] = None
    intended_workflow: Optional[str] = None
    data_classification_need: Optional[str] = None
    collaboration_context: Optional[str] = None
    equipment_context: List[str] = field(default_factory=list)
    location: Optional[str] = None
    location_risk_score: float = 0.5
    access_time: Optional[datetime] = None
    device_fingerprint: Optional[str] = None
    device_trusted: bool = False
    risk_score: float = 0.0
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GeologicalContext':
        """Create GeologicalContext from dictionary"""
        return cls(**data)
        
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class UserProfile:
    """Enhanced user profile with geological expertise"""
    user_id: str
    username: str
    email: str
    full_name: str
    geological_roles: List[str] = field(default_factory=list)
    geological_domains: List[str] = field(default_factory=list)
    clearance_level: Optional[int] = None
    clearance_levels: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    institution_affiliation: Optional[str] = None
    research_projects: List[str] = field(default_factory=list)
    equipment_authorizations: List[str] = field(default_factory=list)
    collaboration_level: str = "individual"
    mfa_enabled: bool = False
    behavioral_profile: Dict = field(default_factory=dict)
    last_login: Optional[datetime] = None
    account_status: str = "active"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class GeologicalRole:
    """Geological role with domain-specific permissions"""
    role_id: str
    name: str
    description: str
    geological_domains: List[GeologicalDomain]
    data_access_levels: List[str]
    computational_permissions: List[str]
    temporal_access: Dict[str, str]
    spatial_boundaries: Optional[Dict] = None
    equipment_access: List[str] = field(default_factory=list)
    collaboration_level: str = "individual"
    minimum_experience: int = 1
    required_certifications: List[str] = field(default_factory=list)

@dataclass
class GeologicalPermission:
    """Geological-specific permission with context awareness"""
    permission_id: str
    name: str
    description: str
    geological_domain: GeologicalDomain
    resource_types: List[str]
    access_levels: List[str]
    conditions: Dict = field(default_factory=dict)
    temporal_restrictions: Optional[Dict] = None
    spatial_restrictions: Optional[Dict] = None
    workflow_requirements: List[str] = field(default_factory=list)
    collaboration_requirements: Optional[str] = None

@dataclass
class SecurityContext:
    """Comprehensive security context for requests"""
    user_id: str
    roles: List[str]
    permissions: Dict
    device_trust: Dict
    data_classification: Dict
    request_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    geological_context: Optional[GeologicalContext] = None
    risk_score: float = 0.0
    threat_indicators: List[str] = field(default_factory=list)
    policy_violations: List[str] = field(default_factory=list)

@dataclass
class AuthenticationResult:
    """Result of authentication attempt"""
    success: bool
    user_profile: Optional[UserProfile] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    requires_mfa: bool = False
    mfa_methods: List[str] = field(default_factory=list)
    geological_permissions: Dict = field(default_factory=dict)
    risk_score: float = 0.0
    error: Optional[str] = None
    expires_at: Optional[datetime] = None

@dataclass
class MFAChallenge:
    """Multi-factor authentication challenge"""
    challenge_id: str
    method_id: str
    challenge_type: str
    challenge_data: Dict
    expected_response: Dict
    expires_at: datetime
    geological_context_required: bool = False
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['expires_at'] = self.expires_at.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MFAChallenge':
        """Create from JSON string"""
        data = json.loads(json_str)
        # Convert ISO format back to datetime
        data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)
    
    @classmethod
    def from_db_record(cls, record: Dict) -> 'MFAChallenge':
        """Create from database record"""
        return cls(
            challenge_id=record['challenge_id'],
            method_id=record['method_id'],
            challenge_type=record.get('challenge_type', 'unknown'),
            challenge_data=record['challenge_data'],
            expected_response=record.get('expected_response', {}),
            expires_at=record['expires_at'],
            geological_context_required=record.get('geological_context_required', False),
            user_id=record.get('user_id'),
            created_at=record.get('created_at', datetime.utcnow())
        )

@dataclass
class MFAResponse:
    """Multi-factor authentication response"""
    challenge_id: str
    method_id: str
    response_data: Dict
    totp_code: Optional[str] = None
    biometric_data: Optional[bytes] = None
    hardware_token_response: Optional[str] = None
    geological_context: Optional[Dict] = None
    device_fingerprint: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class BiometricData:
    """Biometric authentication data"""
    biometric_type: str  # 'face', 'fingerprint', 'voice', 'iris'
    template_data: bytes
    quality_score: float
    liveness_score: Optional[float] = None
    confidence_score: Optional[float] = None
    capture_device: Optional[str] = None
    capture_timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class HardwareTokenData:
    """Hardware token authentication data"""
    token_type: str  # 'yubikey', 'fido2', 'rsa_securid'
    token_id: str
    challenge: str
    response: str
    signature: Optional[str] = None
    certificate: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class NetworkSegment:
    """Network segment configuration"""
    segment_id: str
    name: str
    description: str
    ip_ranges: List[str]
    security_zone: str
    allowed_protocols: List[str]
    allowed_ports: List[int]
    encryption_required: bool = True
    monitoring_level: str = "standard"
    geological_data_types: List[str] = field(default_factory=list)
    access_policies: List[str] = field(default_factory=list)

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    policy_type: str
    rules: List[Dict]
    geological_context_rules: List[Dict] = field(default_factory=list)
    enforcement_level: str = "strict"
    exceptions: List[Dict] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ThreatAlert:
    """Security threat alert"""
    alert_id: str
    threat_type: str
    threat_level: ThreatLevel
    description: str
    affected_resources: List[str]
    geological_context: Optional[GeologicalContext] = None
    indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None

@dataclass
class SecurityEvent:
    """Security event for audit and monitoring"""
    event_id: str
    event_type: SecurityEventType
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    geological_context: Optional[GeologicalContext] = None
    event_data: Dict = field(default_factory=dict)
    risk_score: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None

@dataclass
class AccessRequest:
    """Access request for geological resources"""
    request_id: str
    user_id: str
    resource_id: str
    requested_access_level: str
    geological_context: GeologicalContext
    justification: str
    requested_at: datetime = field(default_factory=datetime.utcnow)
    urgency_level: str = "normal"
    project_context: Optional[str] = None
    collaboration_context: Optional[str] = None
    expected_duration: Optional[timedelta] = None

@dataclass
class AccessDecision:
    """Access decision result"""
    request_id: str
    approved: bool
    reason: str
    risk_score: float
    permission: Optional['DynamicPermission'] = None
    conditions: Dict = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    reviewed_at: datetime = field(default_factory=datetime.utcnow)
    ai_confidence: Optional[float] = None

@dataclass
class ResourceClassification:
    """Geological resource classification"""
    resource_id: str
    classification_level: str
    geological_domains: List[GeologicalDomain]
    data_sensitivity: int
    regulatory_constraints: List[str] = field(default_factory=list)
    access_requirements: List[str] = field(default_factory=list)
    temporal_restrictions: Optional[Dict] = None
    spatial_restrictions: Optional[Dict] = None
    collaboration_restrictions: Optional[str] = None

@dataclass
class GeologicalDataClassification:
    """Classification of geological data with security implications"""
    classification_id: str
    data_type: str
    geological_domain: GeologicalDomain
    sensitivity_level: int  # 1-5 scale
    classification_level: str  # 'public', 'internal', 'restricted', 'confidential', 'top_secret'
    access_requirements: List[str]
    handling_requirements: List[str]
    retention_period: Optional[timedelta] = None
    disposal_requirements: List[str] = field(default_factory=list)
    regulatory_frameworks: List[str] = field(default_factory=list)

@dataclass
class DeviceTrust:
    """Device trust assessment"""
    device_id: str
    device_fingerprint: str
    trust_score: float
    device_type: str
    operating_system: str
    security_features: List[str] = field(default_factory=list)
    compliance_status: Dict = field(default_factory=dict)
    last_assessment: datetime = field(default_factory=datetime.utcnow)
    risk_factors: List[str] = field(default_factory=list)
    geological_equipment_integration: bool = False

@dataclass
class BehavioralPattern:
    """User behavioral pattern for anomaly detection"""
    user_id: str
    pattern_type: str
    pattern_data: Dict
    geological_workflows: List[str] = field(default_factory=list)
    access_patterns: Dict = field(default_factory=dict)
    temporal_patterns: Dict = field(default_factory=dict)
    spatial_patterns: Dict = field(default_factory=dict)
    anomaly_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CertificateInfo:
    """Certificate information for automated management"""
    certificate_id: str
    common_name: str
    subject_alternative_names: List[str]
    issuer: str
    serial_number: str
    not_before: datetime
    not_after: datetime
    key_algorithm: str
    key_size: int
    signature_algorithm: str
    geological_service_context: Optional[str] = None
    auto_renewal_enabled: bool = True
    renewal_threshold_days: int = 30

@dataclass
class PolicyViolation:
    """Security policy violation"""
    violation_id: str
    policy_id: str
    user_id: Optional[str] = None
    resource_id: Optional[str] = None
    violation_type: str
    description: str
    severity: str
    geological_context: Optional[GeologicalContext] = None
    evidence: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolution_actions: List[str] = field(default_factory=list)

@dataclass
class SecurityMetrics:
    """Security metrics for monitoring and reporting"""
    metric_id: str
    metric_type: str
    metric_name: str
    value: Union[int, float, str]
    unit: Optional[str] = None
    geological_context: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_breached: bool = False

# Utility functions for model operations

def generate_correlation_id() -> str:
    """Generate unique correlation ID for request tracking"""
    return f"corr_{uuid.uuid4().hex[:16]}"

def generate_security_event_id() -> str:
    """Generate unique security event ID"""
    return f"sec_event_{uuid.uuid4().hex[:16]}"

def generate_threat_alert_id() -> str:
    """Generate unique threat alert ID"""
    return f"threat_{uuid.uuid4().hex[:16]}"

def calculate_risk_score(factors: Dict[str, float]) -> float:
    """Calculate overall risk score from individual factors"""
    if not factors:
        return 0.0
    
    # Weighted average with higher weights for critical factors
    weights = {
        'authentication_risk': 0.25,
        'authorization_risk': 0.20,
        'behavioral_anomaly': 0.15,
        'device_trust': 0.15,
        'network_risk': 0.10,
        'geological_context_risk': 0.10,
        'temporal_risk': 0.05
    }
    
    weighted_sum = sum(factors.get(factor, 0.0) * weight for factor, weight in weights.items())
    total_weight = sum(weights.values())
    
    return min(weighted_sum / total_weight, 1.0)

def classify_geological_data_sensitivity(
    data_type: str, 
    geological_domain: GeologicalDomain,
    content_indicators: List[str]
) -> int:
    """Classify geological data sensitivity level (1-5)"""
    
    base_sensitivity = {
        GeologicalDomain.VOLCANOLOGY: 4,  # High due to hazard implications
        GeologicalDomain.SEISMOLOGY: 4,   # High due to earthquake monitoring
        GeologicalDomain.GEOCHEMISTRY: 2, # Medium for general analysis
        GeologicalDomain.STRUCTURAL_GEOLOGY: 3, # Medium-high for resource implications
        GeologicalDomain.PETROLOGY: 2,    # Medium for academic research
        GeologicalDomain.MINERALOGY: 2,   # Medium for academic research
        GeologicalDomain.GEOPHYSICS: 3,   # Medium-high for subsurface info
        GeologicalDomain.HYDROGEOLOGY: 3, # Medium-high for water resources
        GeologicalDomain.TECTONICS: 3,    # Medium-high for regional stability
        GeologicalDomain.FIELD_GEOLOGY: 2 # Medium for observational data
    }.get(geological_domain, 2)
    
    # Adjust based on content indicators
    sensitivity_modifiers = {
        'real_time': +1,
        'proprietary': +2,
        'classified_location': +2,
        'commercial_value': +1,
        'hazard_assessment': +1,
        'emergency_response': +1,
        'public_safety': +2,
        'national_security': +3
    }
    
    for indicator in content_indicators:
        base_sensitivity += sensitivity_modifiers.get(indicator, 0)
    
    return min(max(base_sensitivity, 1), 5)

def validate_geological_context(context: GeologicalContext) -> List[str]:
    """Validate geological context and return any issues"""
    issues = []
    
    if not context.primary_domain:
        issues.append("Primary geological domain not specified")
    
    if context.experience_level < 1 or context.experience_level > 5:
        issues.append("Experience level must be between 1 and 5")
    
    if context.risk_score < 0.0 or context.risk_score > 1.0:
        issues.append("Risk score must be between 0.0 and 1.0")
    
    if context.location_risk_score < 0.0 or context.location_risk_score > 1.0:
        issues.append("Location risk score must be between 0.0 and 1.0")
    
    return issues