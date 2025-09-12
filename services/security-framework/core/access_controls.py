"""
Dynamic Least-Privilege Access Controls with Geological Context-Aware Permissions
Implements AI-powered dynamic permission management and geological workflow validation
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aioredis
import asyncpg
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from models.security_models import (
    GeologicalPermission, AccessRequest, AccessDecision, 
    GeologicalContext, UserProfile, ResourceClassification
)
from utils.encryption import AdvancedEncryption
from utils.audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)

class AccessLevel(Enum):
    """Access levels for geological resources"""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4

class GeologicalResourceType(Enum):
    """Types of geological resources"""
    SEISMIC_DATA = "seismic_data"
    VOLCANIC_MONITORING = "volcanic_monitoring"
    GEOCHEMICAL_ANALYSIS = "geochemical_analysis"
    STRUCTURAL_MODELS = "structural_models"
    FIELD_OBSERVATIONS = "field_observations"
    COMPUTATIONAL_RESOURCES = "computational_resources"
    REAL_TIME_STREAMS = "real_time_streams"
    RESEARCH_DATASETS = "research_datasets"
    PROPRIETARY_ALGORITHMS = "proprietary_algorithms"
    EQUIPMENT_CONTROL = "equipment_control"

@dataclass
class GeologicalResource:
    """Geological resource with context-aware access controls"""
    resource_id: str
    name: str
    resource_type: GeologicalResourceType
    classification_level: str  # 'public', 'internal', 'restricted', 'confidential', 'top_secret'
    geological_domains: List[str]
    data_sensitivity: int  # 1-5 scale
    temporal_restrictions: Optional[Dict] = None
    spatial_restrictions: Optional[Dict] = None
    workflow_requirements: List[str] = field(default_factory=list)
    equipment_dependencies: List[str] = field(default_factory=list)
    collaboration_requirements: Optional[str] = None
    regulatory_constraints: List[str] = field(default_factory=list)

@dataclass
class DynamicPermission:
    """Dynamic permission with geological context and time-based validity"""
    permission_id: str
    user_id: str
    resource_id: str
    access_level: AccessLevel
    geological_context: GeologicalContext
    granted_at: datetime
    expires_at: datetime
    conditions: Dict
    usage_count: int = 0
    max_usage: Optional[int] = None
    location_restrictions: Optional[List[str]] = None
    device_restrictions: Optional[List[str]] = None
    workflow_validation_required: bool = False

class DynamicAccessController:
    """
    Dynamic Least-Privilege Access Controller with geological context awareness
    Implements AI-powered permission management and real-time access decisions
    """
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.encryption = AdvancedEncryption()
        self.audit_logger = SecurityAuditLogger()
        
        # AI models for access decisions
        self.access_decision_model = None
        self.risk_assessment_model = None
        self.behavioral_analysis_model = None
        
        # Geological resources registry
        self.geological_resources = {}
        
        # Dynamic permissions cache
        self.active_permissions = {}
        self.permission_history = {}
        
        # Access patterns for ML training
        self.access_patterns = {}
        
    async def initialize(self):
        """Initialize dynamic access controller with geological awareness"""
        logger.info("Initializing Dynamic Access Controller")
        
        # Initialize Redis for real-time permissions
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/3",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Initialize PostgreSQL for persistent data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://prometheus:geological_security@localhost:5432/prometheus_security",
            min_size=5,
            max_size=20
        )
        
        # Initialize geological resources
        await self._initialize_geological_resources()
        
        # Create database tables
        await self._create_access_control_tables()
        
        # Initialize AI models
        await self._initialize_ai_models()
        
        # Load existing permissions
        await self._load_active_permissions()
        
        logger.info("Dynamic Access Controller initialized successfully")
        
    async def _initialize_geological_resources(self):
        """Initialize geological resources with detailed access controls"""
        
        self.geological_resources = {
            'real_time_seismic_feed': GeologicalResource(
                resource_id='real_time_seismic_feed',
                name='Real-time Seismic Data Feed',
                resource_type=GeologicalResourceType.REAL_TIME_STREAMS,
                classification_level='restricted',
                geological_domains=['seismology', 'earthquake_monitoring'],
                data_sensitivity=4,
                temporal_restrictions={'emergency_priority': True},
                workflow_requirements=['seismic_analysis_certified'],
                equipment_dependencies=['seismic_network_access'],
                regulatory_constraints=['data_sharing_agreement']
            ),
            
            'volcanic_monitoring_system': GeologicalResource(
                resource_id='volcanic_monitoring_system',
                name='Volcanic Monitoring and Alert System',
                resource_type=GeologicalResourceType.VOLCANIC_MONITORING,
                classification_level='confidential',
                geological_domains=['volcanology', 'hazard_assessment'],
                data_sensitivity=5,
                temporal_restrictions={'24x7_monitoring': True},
                spatial_restrictions={'volcanic_regions': ['cascade_range', 'yellowstone', 'alaska']},
                workflow_requirements=['volcanic_monitoring_certified', 'emergency_response_trained'],
                equipment_dependencies=['volcanic_sensor_network', 'satellite_data_access'],
                collaboration_requirements='multi_agency_coordination'
            ),
            
            'geochemical_database': GeologicalResource(
                resource_id='geochemical_database',
                name='Comprehensive Geochemical Analysis Database',
                resource_type=GeologicalResourceType.GEOCHEMICAL_ANALYSIS,
                classification_level='internal',
                geological_domains=['geochemistry', 'petrology', 'mineralogy'],
                data_sensitivity=3,
                workflow_requirements=['geochemical_analysis_certified'],
                equipment_dependencies=['laboratory_access', 'analytical_instruments'],
                regulatory_constraints=['sample_provenance_verification']
            ),
            
            'structural_modeling_suite': GeologicalResource(
                resource_id='structural_modeling_suite',
                name='3D Structural Geology Modeling Suite',
                resource_type=GeologicalResourceType.STRUCTURAL_MODELS,
                classification_level='restricted',
                geological_domains=['structural_geology', '3d_modeling', 'tectonics'],
                data_sensitivity=4,
                workflow_requirements=['structural_modeling_certified', 'advanced_geology_degree'],
                equipment_dependencies=['high_performance_computing', 'specialized_software'],
                collaboration_requirements='research_institution_affiliation'
            ),
            
            'field_data_collection': GeologicalResource(
                resource_id='field_data_collection',
                name='Field Data Collection and Management System',
                resource_type=GeologicalResourceType.FIELD_OBSERVATIONS,
                classification_level='internal',
                geological_domains=['field_geology', 'data_collection'],
                data_sensitivity=2,
                spatial_restrictions={'authorized_field_sites': True},
                workflow_requirements=['field_safety_certified', 'data_collection_trained'],
                equipment_dependencies=['mobile_devices', 'gps_equipment', 'sampling_tools']
            ),
            
            'hpc_computational_cluster': GeologicalResource(
                resource_id='hpc_computational_cluster',
                name='High-Performance Computing Cluster',
                resource_type=GeologicalResourceType.COMPUTATIONAL_RESOURCES,
                classification_level='restricted',
                geological_domains=['computational_geology', 'numerical_modeling'],
                data_sensitivity=3,
                temporal_restrictions={'resource_scheduling': True},
                workflow_requirements=['hpc_certified', 'computational_methods_trained'],
                equipment_dependencies=['cluster_access', 'job_scheduler'],
                regulatory_constraints=['computational_resource_allocation']
            ),
            
            'proprietary_algorithms': GeologicalResource(
                resource_id='proprietary_algorithms',
                name='Proprietary Geological Analysis Algorithms',
                resource_type=GeologicalResourceType.PROPRIETARY_ALGORITHMS,
                classification_level='top_secret',
                geological_domains=['algorithm_development', 'intellectual_property'],
                data_sensitivity=5,
                workflow_requirements=['senior_researcher_status', 'ip_agreement_signed'],
                collaboration_requirements='internal_only',
                regulatory_constraints=['intellectual_property_protection', 'non_disclosure_agreement']
            ),
            
            'equipment_control_system': GeologicalResource(
                resource_id='equipment_control_system',
                name='Geological Equipment Control System',
                resource_type=GeologicalResourceType.EQUIPMENT_CONTROL,
                classification_level='confidential',
                geological_domains=['equipment_operation', 'remote_sensing'],
                data_sensitivity=4,
                temporal_restrictions={'operational_hours': True},
                spatial_restrictions={'equipment_locations': True},
                workflow_requirements=['equipment_certified', 'safety_trained'],
                equipment_dependencies=['control_systems', 'communication_links'],
                regulatory_constraints=['equipment_operation_permits']
            )
        }
        
    async def _create_access_control_tables(self):
        """Create access control database tables"""
        async with self.db_pool.acquire() as conn:
            # Geological resources table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS geological_resources (
                    resource_id VARCHAR(100) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    resource_type VARCHAR(50) NOT NULL,
                    classification_level VARCHAR(50) NOT NULL,
                    geological_domains JSONB NOT NULL,
                    data_sensitivity INTEGER NOT NULL,
                    temporal_restrictions JSONB,
                    spatial_restrictions JSONB,
                    workflow_requirements JSONB,
                    equipment_dependencies JSONB,
                    collaboration_requirements VARCHAR(100),
                    regulatory_constraints JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Dynamic permissions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dynamic_permissions (
                    permission_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    resource_id VARCHAR(100) NOT NULL,
                    access_level INTEGER NOT NULL,
                    geological_context JSONB NOT NULL,
                    granted_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    conditions JSONB,
                    usage_count INTEGER DEFAULT 0,
                    max_usage INTEGER,
                    location_restrictions JSONB,
                    device_restrictions JSONB,
                    workflow_validation_required BOOLEAN DEFAULT false,
                    active BOOLEAN DEFAULT true,
                    granted_by VARCHAR(50),
                    revoked_at TIMESTAMP,
                    revoked_by VARCHAR(50),
                    revocation_reason TEXT
                )
            """)
            
            # Access requests table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS access_requests (
                    request_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    resource_id VARCHAR(100) NOT NULL,
                    requested_access_level INTEGER NOT NULL,
                    geological_context JSONB NOT NULL,
                    justification TEXT,
                    requested_at TIMESTAMP DEFAULT NOW(),
                    reviewed_at TIMESTAMP,
                    reviewed_by VARCHAR(50),
                    decision VARCHAR(20), -- 'approved', 'denied', 'pending'
                    decision_reason TEXT,
                    ai_risk_score FLOAT,
                    ai_recommendation VARCHAR(20),
                    conditions_applied JSONB
                )
            """)
            
            # Access audit log table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS access_audit_log (
                    audit_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    resource_id VARCHAR(100) NOT NULL,
                    action VARCHAR(50) NOT NULL, -- 'access_granted', 'access_denied', 'resource_accessed', 'permission_revoked'
                    access_level INTEGER,
                    geological_context JSONB,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    source_ip INET,
                    user_agent TEXT,
                    success BOOLEAN,
                    risk_score FLOAT,
                    additional_data JSONB
                )
            """)
            
    async def _initialize_ai_models(self):
        """Initialize AI models for access decision making"""
        
        # Access decision model (would be trained on historical data)
        self.access_decision_model = {
            'model_type': 'random_forest',
            'features': [
                'user_role_score', 'geological_expertise', 'resource_sensitivity',
                'time_of_access', 'location_risk', 'device_trust_score',
                'workflow_compliance', 'historical_behavior', 'collaboration_level'
            ],
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'trained': False
        }
        
        # Risk assessment model
        self.risk_assessment_model = {
            'model_type': 'gradient_boosting',
            'risk_factors': [
                'data_sensitivity', 'user_clearance_level', 'access_pattern_anomaly',
                'geographic_location', 'time_anomaly', 'device_anomaly',
                'workflow_deviation', 'collaboration_risk'
            ],
            'thresholds': {
                'low_risk': 0.3,
                'medium_risk': 0.6,
                'high_risk': 0.8
            }
        }
        
        # Behavioral analysis model
        self.behavioral_analysis_model = {
            'model_type': 'isolation_forest',
            'behavioral_features': [
                'access_frequency', 'access_duration', 'resource_diversity',
                'time_patterns', 'location_patterns', 'workflow_patterns'
            ],
            'anomaly_threshold': -0.1
        }
        
    async def _load_active_permissions(self):
        """Load active permissions from database"""
        async with self.db_pool.acquire() as conn:
            permissions = await conn.fetch("""
                SELECT * FROM dynamic_permissions 
                WHERE active = true AND expires_at > NOW()
            """)
            
            for perm in permissions:
                permission = DynamicPermission(
                    permission_id=perm['permission_id'],
                    user_id=perm['user_id'],
                    resource_id=perm['resource_id'],
                    access_level=AccessLevel(perm['access_level']),
                    geological_context=GeologicalContext.from_dict(perm['geological_context']),
                    granted_at=perm['granted_at'],
                    expires_at=perm['expires_at'],
                    conditions=perm['conditions'] or {},
                    usage_count=perm['usage_count'],
                    max_usage=perm['max_usage'],
                    location_restrictions=perm['location_restrictions'],
                    device_restrictions=perm['device_restrictions'],
                    workflow_validation_required=perm['workflow_validation_required']
                )
                
                self.active_permissions[perm['permission_id']] = permission
                
    async def request_access(
        self, 
        user_id: str, 
        resource_id: str, 
        requested_access_level: AccessLevel,
        geological_context: GeologicalContext,
        justification: str = ""
    ) -> AccessDecision:
        """
        Request access to geological resource with AI-powered decision making
        Returns access decision with dynamic permissions if approved
        """
        try:
            request_id = f"access_req_{datetime.utcnow().timestamp()}_{user_id}_{resource_id}"
            
            # Validate resource exists
            if resource_id not in self.geological_resources:
                return AccessDecision(
                    request_id=request_id,
                    approved=False,
                    reason="Resource not found",
                    risk_score=1.0
                )
                
            resource = self.geological_resources[resource_id]
            
            # Get user profile and permissions
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                return AccessDecision(
                    request_id=request_id,
                    approved=False,
                    reason="User profile not found",
                    risk_score=1.0
                )
                
            # AI-powered risk assessment
            risk_score = await self._assess_access_risk(
                user_profile, resource, requested_access_level, geological_context
            )
            
            # AI-powered access decision
            ai_decision = await self._make_ai_access_decision(
                user_profile, resource, requested_access_level, geological_context, risk_score
            )
            
            # Apply business rules and geological constraints
            business_decision = await self._apply_business_rules(
                user_profile, resource, requested_access_level, geological_context
            )
            
            # Combine AI and business decisions
            final_decision = ai_decision and business_decision['approved']
            
            # Store access request
            await self._store_access_request(
                request_id, user_id, resource_id, requested_access_level,
                geological_context, justification, final_decision, risk_score,
                ai_decision, business_decision
            )
            
            if final_decision:
                # Generate dynamic permission
                permission = await self._generate_dynamic_permission(
                    user_id, resource_id, requested_access_level, 
                    geological_context, business_decision.get('conditions', {})
                )
                
                # Store permission
                await self._store_dynamic_permission(permission)
                
                # Cache permission
                self.active_permissions[permission.permission_id] = permission
                
                await self.audit_logger.log_access_granted(
                    user_id, resource_id, requested_access_level, geological_context
                )
                
                return AccessDecision(
                    request_id=request_id,
                    approved=True,
                    reason="Access granted based on AI analysis and business rules",
                    risk_score=risk_score,
                    permission=permission,
                    conditions=business_decision.get('conditions', {}),
                    expires_at=permission.expires_at
                )
            else:
                await self.audit_logger.log_access_denied(
                    user_id, resource_id, requested_access_level, geological_context,
                    business_decision.get('reason', 'AI decision negative')
                )
                
                return AccessDecision(
                    request_id=request_id,
                    approved=False,
                    reason=business_decision.get('reason', 'Access denied by AI analysis'),
                    risk_score=risk_score
                )
                
        except Exception as e:
            logger.error(f"Access request error: {str(e)}")
            return AccessDecision(
                request_id=request_id,
                approved=False,
                reason=f"System error: {str(e)}",
                risk_score=1.0
            )
            
    async def validate_access(
        self, 
        user_id: str, 
        resource_id: str, 
        access_level: AccessLevel,
        geological_context: GeologicalContext
    ) -> bool:
        """
        Validate user access to resource with real-time permission checking
        Returns True if access is allowed, False otherwise
        """
        try:
            # Find applicable permissions
            user_permissions = [
                perm for perm in self.active_permissions.values()
                if perm.user_id == user_id and perm.resource_id == resource_id
                and perm.access_level.value >= access_level.value
                and perm.expires_at > datetime.utcnow()
            ]
            
            if not user_permissions:
                await self.audit_logger.log_access_denied(
                    user_id, resource_id, access_level, geological_context, "No valid permissions"
                )
                return False
                
            # Check each permission
            for permission in user_permissions:
                if await self._validate_permission_conditions(permission, geological_context):
                    # Update usage count
                    await self._increment_permission_usage(permission.permission_id)
                    
                    await self.audit_logger.log_resource_accessed(
                        user_id, resource_id, access_level, geological_context
                    )
                    return True
                    
            await self.audit_logger.log_access_denied(
                user_id, resource_id, access_level, geological_context, "Permission conditions not met"
            )
            return False
            
        except Exception as e:
            logger.error(f"Access validation error: {str(e)}")
            return False
            
    async def _assess_access_risk(
        self, 
        user_profile: UserProfile, 
        resource: GeologicalResource,
        access_level: AccessLevel,
        geological_context: GeologicalContext
    ) -> float:
        """AI-powered risk assessment for access request"""
        
        risk_factors = {
            'data_sensitivity_mismatch': 0.0,
            'user_clearance_insufficient': 0.0,
            'unusual_access_pattern': 0.0,
            'geographic_anomaly': 0.0,
            'time_anomaly': 0.0,
            'device_trust_low': 0.0,
            'workflow_deviation': 0.0,
            'collaboration_risk': 0.0
        }
        
        # Data sensitivity vs user clearance
        user_clearance = user_profile.clearance_level or 1
        if resource.data_sensitivity > user_clearance:
            risk_factors['user_clearance_insufficient'] = 0.8
            
        # Access pattern analysis
        historical_access = await self._get_user_access_history(user_profile.user_id)
        if not self._is_typical_access_pattern(historical_access, resource.resource_type):
            risk_factors['unusual_access_pattern'] = 0.4
            
        # Geographic risk
        if geological_context.location and not self._is_authorized_location(
            geological_context.location, resource.spatial_restrictions
        ):
            risk_factors['geographic_anomaly'] = 0.6
            
        # Time-based risk
        current_time = datetime.utcnow()
        if not self._is_authorized_time(current_time, resource.temporal_restrictions):
            risk_factors['time_anomaly'] = 0.3
            
        # Device trust
        if geological_context.device_trust_score < 0.7:
            risk_factors['device_trust_low'] = 0.5
            
        # Workflow validation
        if not self._validate_workflow_requirements(
            user_profile.geological_roles, resource.workflow_requirements
        ):
            risk_factors['workflow_deviation'] = 0.7
            
        # Calculate overall risk score
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        return min(risk_score, 1.0)
        
    async def _make_ai_access_decision(
        self, 
        user_profile: UserProfile, 
        resource: GeologicalResource,
        access_level: AccessLevel,
        geological_context: GeologicalContext,
        risk_score: float
    ) -> bool:
        """AI-powered access decision making"""
        
        # Feature extraction for ML model
        features = np.array([[
            self._calculate_user_role_score(user_profile.geological_roles),
            self._calculate_geological_expertise(user_profile, resource.geological_domains),
            resource.data_sensitivity,
            datetime.utcnow().hour,  # Time of access
            geological_context.location_risk_score if geological_context.location else 0.5,
            geological_context.device_trust_score,
            self._calculate_workflow_compliance(user_profile, resource),
            self._calculate_historical_behavior_score(user_profile.user_id),
            self._calculate_collaboration_score(resource.collaboration_requirements)
        ]])
        
        # Use AI model for decision (placeholder - would use trained model)
        if not self.access_decision_model['trained']:
            # Rule-based decision until model is trained
            if risk_score < 0.3:
                return True
            elif risk_score < 0.6:
                return self._apply_additional_validation(user_profile, resource)
            else:
                return False
        else:
            # Use trained ML model
            prediction = self.access_decision_model['model'].predict(features)[0]
            confidence = self.access_decision_model['model'].predict_proba(features)[0].max()
            
            return prediction == 1 and confidence > 0.7
            
    async def _apply_business_rules(
        self, 
        user_profile: UserProfile, 
        resource: GeologicalResource,
        access_level: AccessLevel,
        geological_context: GeologicalContext
    ) -> Dict:
        """Apply business rules and geological constraints"""
        
        conditions = {}
        
        # Check classification level clearance
        user_clearance_levels = user_profile.clearance_levels or ['public']
        if resource.classification_level not in user_clearance_levels:
            return {
                'approved': False,
                'reason': f'Insufficient clearance level. Required: {resource.classification_level}'
            }
            
        # Check geological domain expertise
        user_domains = set(user_profile.geological_domains or [])
        resource_domains = set(resource.geological_domains)
        if not user_domains.intersection(resource_domains):
            return {
                'approved': False,
                'reason': 'Insufficient geological domain expertise'
            }
            
        # Check workflow requirements
        user_certifications = set(user_profile.certifications or [])
        required_workflows = set(resource.workflow_requirements)
        if not required_workflows.issubset(user_certifications):
            missing_certs = required_workflows - user_certifications
            return {
                'approved': False,
                'reason': f'Missing required certifications: {list(missing_certs)}'
            }
            
        # Apply temporal restrictions
        if resource.temporal_restrictions:
            time_conditions = self._apply_temporal_restrictions(resource.temporal_restrictions)
            conditions.update(time_conditions)
            
        # Apply spatial restrictions
        if resource.spatial_restrictions:
            spatial_conditions = self._apply_spatial_restrictions(
                resource.spatial_restrictions, geological_context.location
            )
            if not spatial_conditions['allowed']:
                return {
                    'approved': False,
                    'reason': 'Geographic access restrictions not met'
                }
            conditions.update(spatial_conditions.get('conditions', {}))
            
        # Apply collaboration requirements
        if resource.collaboration_requirements:
            collab_conditions = self._apply_collaboration_requirements(
                resource.collaboration_requirements, user_profile
            )
            if not collab_conditions['allowed']:
                return {
                    'approved': False,
                    'reason': collab_conditions['reason']
                }
            conditions.update(collab_conditions.get('conditions', {}))
            
        return {
            'approved': True,
            'reason': 'Business rules validation passed',
            'conditions': conditions
        }
        
    async def _generate_dynamic_permission(
        self, 
        user_id: str, 
        resource_id: str, 
        access_level: AccessLevel,
        geological_context: GeologicalContext,
        conditions: Dict
    ) -> DynamicPermission:
        """Generate dynamic permission with geological context"""
        
        permission_id = f"perm_{datetime.utcnow().timestamp()}_{user_id}_{resource_id}"
        
        # Calculate expiration based on resource sensitivity and access level
        resource = self.geological_resources[resource_id]
        
        if resource.data_sensitivity >= 4:
            # High sensitivity - shorter duration
            duration_hours = 4
        elif resource.data_sensitivity >= 3:
            # Medium sensitivity - medium duration
            duration_hours = 8
        else:
            # Lower sensitivity - longer duration
            duration_hours = 24
            
        # Adjust for access level
        if access_level == AccessLevel.ADMIN:
            duration_hours = min(duration_hours, 2)  # Admin access is always short-lived
        elif access_level == AccessLevel.EXECUTE:
            duration_hours = min(duration_hours, 4)
            
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours)
        
        return DynamicPermission(
            permission_id=permission_id,
            user_id=user_id,
            resource_id=resource_id,
            access_level=access_level,
            geological_context=geological_context,
            granted_at=datetime.utcnow(),
            expires_at=expires_at,
            conditions=conditions,
            usage_count=0,
            max_usage=conditions.get('max_usage'),
            location_restrictions=conditions.get('location_restrictions'),
            device_restrictions=conditions.get('device_restrictions'),
            workflow_validation_required=conditions.get('workflow_validation_required', False)
        )
        
    async def health_check(self) -> Dict:
        """Health check for access controller"""
        try:
            # Check Redis connection
            redis_healthy = await self.redis_client.ping()
            
            # Check database connection
            async with self.db_pool.acquire() as conn:
                db_healthy = await conn.fetchval("SELECT 1") == 1
                
            # Check AI models status
            ai_models_healthy = all([
                self.access_decision_model is not None,
                self.risk_assessment_model is not None,
                self.behavioral_analysis_model is not None
            ])
            
            return {
                "healthy": redis_healthy and db_healthy and ai_models_healthy,
                "components": {
                    "redis": redis_healthy,
                    "database": db_healthy,
                    "ai_models": ai_models_healthy,
                    "geological_resources": len(self.geological_resources),
                    "active_permissions": len(self.active_permissions)
                }
            }
        except Exception as e:
            logger.error(f"Access controller health check failed: {str(e)}")
            return {"healthy": False, "error": str(e)}