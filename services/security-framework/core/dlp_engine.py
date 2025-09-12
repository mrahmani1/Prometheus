"""
Data Loss Prevention (DLP) Engine for Prometheus AI Platform

This module implements comprehensive DLP controls including:
- Real-time data monitoring and classification
- Policy enforcement for data access and sharing
- Geological data sensitivity detection
- Automated data masking and redaction
- Compliance monitoring and reporting
- Incident detection and response
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import uuid

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Import from encryption engine
from .encryption_engine import DataClassification, GeologicalDataField, DataClassifier


class DLPAction(Enum):
    """DLP enforcement actions"""
    ALLOW = "allow"
    BLOCK = "block"
    MASK = "mask"
    REDACT = "redact"
    ENCRYPT = "encrypt"
    AUDIT = "audit"
    ALERT = "alert"
    QUARANTINE = "quarantine"


class DLPViolationType(Enum):
    """Types of DLP violations"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    POLICY_VIOLATION = "policy_violation"
    CLASSIFICATION_MISMATCH = "classification_mismatch"
    GEOLOGICAL_SENSITIVITY = "geological_sensitivity"
    PII_EXPOSURE = "pii_exposure"
    COMPLIANCE_VIOLATION = "compliance_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


class GeologicalDataSensitivity(Enum):
    """Geological data sensitivity levels"""
    PUBLIC_RESEARCH = "public_research"
    ACADEMIC_USE = "academic_use"
    COMMERCIAL_SENSITIVE = "commercial_sensitive"
    NATIONAL_SECURITY = "national_security"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    DISASTER_PREPAREDNESS = "disaster_preparedness"


@dataclass
class DLPPolicy:
    """Data Loss Prevention policy definition"""
    policy_id: str
    name: str
    description: str
    data_types: List[str]
    classifications: List[DataClassification]
    geological_sensitivities: List[GeologicalDataSensitivity]
    conditions: Dict[str, Any]
    actions: List[DLPAction]
    enabled: bool = True
    priority: int = 100
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DLPViolation:
    """DLP violation record"""
    violation_id: str
    violation_type: DLPViolationType
    policy_id: str
    user_id: str
    resource_id: str
    data_classification: DataClassification
    geological_sensitivity: GeologicalDataSensitivity
    violation_details: Dict[str, Any]
    actions_taken: List[DLPAction]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "medium"
    resolved: bool = False


@dataclass
class DataAccessRequest:
    """Data access request for DLP evaluation"""
    request_id: str
    user_id: str
    user_roles: List[str]
    resource_id: str
    data_type: str
    operation: str  # read, write, delete, export, share
    destination: Optional[str] = None
    purpose: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeologicalDataContext:
    """Geological data context for DLP decisions"""
    data_source: str
    geological_domain: str  # seismic, volcanic, mineral, etc.
    temporal_scope: str  # real-time, historical, predictive
    spatial_scope: str  # local, regional, global
    research_purpose: str
    commercial_value: str  # none, low, medium, high, critical
    public_safety_impact: str  # none, low, medium, high, critical
    national_security_relevance: str  # none, low, medium, high, critical


class GeologicalSensitivityAnalyzer:
    """Analyzer for geological data sensitivity"""
    
    def __init__(self):
        self.sensitivity_rules = self._load_sensitivity_rules()
        self.critical_locations = self._load_critical_locations()
        self.sensitive_parameters = self._load_sensitive_parameters()
    
    def _load_sensitivity_rules(self) -> Dict[str, Any]:
        """Load geological sensitivity rules"""
        return {
            'earthquake_prediction': {
                'base_sensitivity': GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                'critical_regions': ['san_andreas', 'yellowstone', 'cascadia'],
                'sensitive_parameters': ['magnitude_prediction', 'timing_forecast', 'probability_maps']
            },
            'volcanic_monitoring': {
                'base_sensitivity': GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                'critical_volcanoes': ['yellowstone', 'mount_st_helens', 'kilauea', 'long_valley'],
                'sensitive_parameters': ['eruption_probability', 'magma_chamber_pressure', 'gas_emissions']
            },
            'nuclear_facility_geology': {
                'base_sensitivity': GeologicalDataSensitivity.NATIONAL_SECURITY,
                'buffer_distance_km': 50,
                'sensitive_parameters': ['fault_activity', 'ground_stability', 'seismic_hazard']
            },
            'critical_infrastructure': {
                'base_sensitivity': GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE,
                'infrastructure_types': ['power_plants', 'dams', 'bridges', 'tunnels', 'airports'],
                'sensitive_parameters': ['foundation_stability', 'liquefaction_risk', 'landslide_hazard']
            },
            'mineral_exploration': {
                'base_sensitivity': GeologicalDataSensitivity.COMMERCIAL_SENSITIVE,
                'valuable_commodities': ['rare_earth_elements', 'lithium', 'gold', 'platinum', 'uranium'],
                'sensitive_parameters': ['ore_grade', 'reserve_estimates', 'exploration_targets']
            },
            'oil_gas_resources': {
                'base_sensitivity': GeologicalDataSensitivity.COMMERCIAL_SENSITIVE,
                'strategic_importance': True,
                'sensitive_parameters': ['reservoir_properties', 'production_forecasts', 'exploration_data']
            },
            'groundwater_resources': {
                'base_sensitivity': GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                'critical_aquifers': ['ogallala', 'california_central_valley', 'floridan'],
                'sensitive_parameters': ['contamination_levels', 'depletion_rates', 'quality_assessments']
            }
        }
    
    def _load_critical_locations(self) -> Dict[str, Any]:
        """Load critical geological locations"""
        return {
            'yellowstone_caldera': {
                'coordinates': {'lat': 44.428, 'lon': -110.588},
                'radius_km': 100,
                'sensitivity': GeologicalDataSensitivity.NATIONAL_SECURITY,
                'monitoring_level': 'continuous'
            },
            'san_andreas_fault': {
                'coordinates': [
                    {'lat': 36.0, 'lon': -120.0},
                    {'lat': 34.0, 'lon': -118.0}
                ],
                'buffer_km': 25,
                'sensitivity': GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                'monitoring_level': 'high'
            },
            'cascadia_subduction_zone': {
                'coordinates': [
                    {'lat': 40.0, 'lon': -125.0},
                    {'lat': 49.0, 'lon': -127.0}
                ],
                'buffer_km': 200,
                'sensitivity': GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                'monitoring_level': 'continuous'
            }
        }
    
    def _load_sensitive_parameters(self) -> Set[str]:
        """Load sensitive geological parameters"""
        return {
            # Seismic parameters
            'earthquake_magnitude_prediction',
            'fault_rupture_probability',
            'ground_motion_forecast',
            'liquefaction_susceptibility',
            
            # Volcanic parameters
            'eruption_probability',
            'volcanic_explosivity_index',
            'pyroclastic_flow_modeling',
            'ash_dispersion_forecast',
            'magma_chamber_dynamics',
            
            # Resource parameters
            'ore_grade_distribution',
            'reserve_calculations',
            'economic_viability_assessment',
            'exploration_target_ranking',
            
            # Infrastructure parameters
            'foundation_bearing_capacity',
            'slope_stability_analysis',
            'seismic_hazard_assessment',
            'ground_subsidence_rate',
            
            # Environmental parameters
            'contamination_plume_modeling',
            'groundwater_flow_direction',
            'aquifer_vulnerability_assessment',
            'environmental_impact_prediction'
        }
    
    async def analyze_sensitivity(
        self,
        data: Dict[str, Any],
        context: GeologicalDataContext
    ) -> GeologicalDataSensitivity:
        """Analyze geological data sensitivity"""
        
        base_sensitivity = GeologicalDataSensitivity.ACADEMIC_USE
        
        # Check domain-specific sensitivity rules
        domain_rules = self.sensitivity_rules.get(context.geological_domain, {})
        if domain_rules:
            base_sensitivity = domain_rules.get('base_sensitivity', base_sensitivity)
        
        # Check for sensitive parameters
        for field_name in data.keys():
            if field_name.lower() in self.sensitive_parameters:
                if context.public_safety_impact in ['high', 'critical']:
                    base_sensitivity = max(base_sensitivity, GeologicalDataSensitivity.DISASTER_PREPAREDNESS)
                elif context.national_security_relevance in ['high', 'critical']:
                    base_sensitivity = max(base_sensitivity, GeologicalDataSensitivity.NATIONAL_SECURITY)
                elif context.commercial_value in ['high', 'critical']:
                    base_sensitivity = max(base_sensitivity, GeologicalDataSensitivity.COMMERCIAL_SENSITIVE)
        
        # Check spatial context
        if 'location' in data or 'coordinates' in data:
            location_sensitivity = await self._analyze_location_sensitivity(data)
            base_sensitivity = max(base_sensitivity, location_sensitivity)
        
        # Check temporal context
        if context.temporal_scope == 'predictive' and context.public_safety_impact == 'critical':
            base_sensitivity = max(base_sensitivity, GeologicalDataSensitivity.DISASTER_PREPAREDNESS)
        
        return base_sensitivity
    
    async def _analyze_location_sensitivity(self, data: Dict[str, Any]) -> GeologicalDataSensitivity:
        """Analyze location-based sensitivity"""
        # Extract coordinates from data
        coordinates = self._extract_coordinates(data)
        if not coordinates:
            return GeologicalDataSensitivity.ACADEMIC_USE
        
        max_sensitivity = GeologicalDataSensitivity.ACADEMIC_USE
        
        # Check against critical locations
        for location_name, location_info in self.critical_locations.items():
            if self._is_within_critical_area(coordinates, location_info):
                location_sensitivity = location_info['sensitivity']
                max_sensitivity = max(max_sensitivity, location_sensitivity)
        
        return max_sensitivity
    
    def _extract_coordinates(self, data: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Extract coordinates from data"""
        # Try various coordinate field names
        coord_fields = [
            ('latitude', 'longitude'),
            ('lat', 'lon'),
            ('y', 'x'),
            ('location_lat', 'location_lon')
        ]
        
        for lat_field, lon_field in coord_fields:
            if lat_field in data and lon_field in data:
                try:
                    lat = float(data[lat_field])
                    lon = float(data[lon_field])
                    return (lat, lon)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _is_within_critical_area(
        self,
        coordinates: Tuple[float, float],
        location_info: Dict[str, Any]
    ) -> bool:
        """Check if coordinates are within critical area"""
        lat, lon = coordinates
        
        if 'coordinates' in location_info:
            if isinstance(location_info['coordinates'], dict):
                # Point location with radius
                center_lat = location_info['coordinates']['lat']
                center_lon = location_info['coordinates']['lon']
                radius_km = location_info.get('radius_km', 10)
                
                # Simple distance calculation (approximate)
                distance_km = self._calculate_distance(lat, lon, center_lat, center_lon)
                return distance_km <= radius_km
                
            elif isinstance(location_info['coordinates'], list):
                # Line or polygon - simplified check
                buffer_km = location_info.get('buffer_km', 10)
                for coord in location_info['coordinates']:
                    distance_km = self._calculate_distance(lat, lon, coord['lat'], coord['lon'])
                    if distance_km <= buffer_km:
                        return True
        
        return False
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate approximate distance between two points in km"""
        # Simplified distance calculation using Haversine formula
        import math
        
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c


class DLPPolicyEngine:
    """DLP policy evaluation and enforcement engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policies: Dict[str, DLPPolicy] = {}
        self.violations: List[DLPViolation] = []
        self.redis_client = None
        self.sensitivity_analyzer = GeologicalSensitivityAnalyzer()
        self.data_classifier = DataClassifier(config)
    
    async def initialize(self) -> bool:
        """Initialize DLP policy engine"""
        try:
            # Initialize Redis for policy and violation storage
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Load default policies
            await self._load_default_policies()
            
            logging.info("DLP policy engine initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"DLP policy engine initialization failed: {e}")
            return False
    
    async def _load_default_policies(self):
        """Load default DLP policies"""
        
        # Policy 1: Prevent unauthorized access to classified geological data
        policy1 = DLPPolicy(
            policy_id="geological_classification_access",
            name="Geological Data Classification Access Control",
            description="Prevent unauthorized access to classified geological data",
            data_types=["seismic_data", "volcanic_monitoring", "resource_assessment"],
            classifications=[DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
            geological_sensitivities=[
                GeologicalDataSensitivity.NATIONAL_SECURITY,
                GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE,
                GeologicalDataSensitivity.DISASTER_PREPAREDNESS
            ],
            conditions={
                "require_clearance": True,
                "max_concurrent_access": 5,
                "time_restrictions": {"start": "06:00", "end": "22:00"}
            },
            actions=[DLPAction.BLOCK, DLPAction.AUDIT, DLPAction.ALERT],
            priority=10
        )
        
        # Policy 2: Monitor data exfiltration attempts
        policy2 = DLPPolicy(
            policy_id="data_exfiltration_monitoring",
            name="Geological Data Exfiltration Monitoring",
            description="Monitor and prevent unauthorized data exfiltration",
            data_types=["all"],
            classifications=[DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
            geological_sensitivities=[
                GeologicalDataSensitivity.COMMERCIAL_SENSITIVE,
                GeologicalDataSensitivity.NATIONAL_SECURITY
            ],
            conditions={
                "max_export_size_mb": 100,
                "max_exports_per_hour": 10,
                "allowed_destinations": ["internal", "approved_partners"]
            },
            actions=[DLPAction.BLOCK, DLPAction.QUARANTINE, DLPAction.ALERT],
            priority=5
        )
        
        # Policy 3: PII protection in geological datasets
        policy3 = DLPPolicy(
            policy_id="pii_protection_geological",
            name="PII Protection in Geological Data",
            description="Protect personally identifiable information in geological datasets",
            data_types=["field_surveys", "research_data", "collaboration_data"],
            classifications=[DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            geological_sensitivities=[GeologicalDataSensitivity.ACADEMIC_USE],
            conditions={
                "auto_detect_pii": True,
                "mask_pii_fields": True,
                "require_consent": True
            },
            actions=[DLPAction.MASK, DLPAction.REDACT, DLPAction.AUDIT],
            priority=20
        )
        
        # Policy 4: Critical infrastructure protection
        policy4 = DLPPolicy(
            policy_id="critical_infrastructure_protection",
            name="Critical Infrastructure Geological Data Protection",
            description="Protect geological data related to critical infrastructure",
            data_types=["infrastructure_assessment", "hazard_analysis", "risk_modeling"],
            classifications=[DataClassification.RESTRICTED, DataClassification.TOP_SECRET],
            geological_sensitivities=[
                GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE,
                GeologicalDataSensitivity.NATIONAL_SECURITY
            ],
            conditions={
                "require_multi_factor_auth": True,
                "restrict_to_authorized_personnel": True,
                "no_external_sharing": True
            },
            actions=[DLPAction.BLOCK, DLPAction.ENCRYPT, DLPAction.AUDIT, DLPAction.ALERT],
            priority=1
        )
        
        # Store policies
        policies = [policy1, policy2, policy3, policy4]
        for policy in policies:
            self.policies[policy.policy_id] = policy
            await self._store_policy(policy)
    
    async def evaluate_access_request(
        self,
        request: DataAccessRequest,
        data: Dict[str, Any],
        context: GeologicalDataContext
    ) -> Tuple[bool, List[DLPAction], Optional[DLPViolation]]:
        """Evaluate data access request against DLP policies"""
        
        # Classify the data
        classified_fields = await self.data_classifier.classify_data(data, request.data_type)
        
        # Analyze geological sensitivity
        geological_sensitivity = await self.sensitivity_analyzer.analyze_sensitivity(data, context)
        
        # Find applicable policies
        applicable_policies = await self._find_applicable_policies(
            request, classified_fields, geological_sensitivity
        )
        
        # Evaluate each policy
        violations = []
        actions_to_take = []
        allow_access = True
        
        for policy in applicable_policies:
            policy_result = await self._evaluate_policy(
                policy, request, data, classified_fields, geological_sensitivity, context
            )
            
            if policy_result['violation']:
                violations.append(policy_result['violation'])
                actions_to_take.extend(policy_result['actions'])
                
                # Check if any action blocks access
                if DLPAction.BLOCK in policy_result['actions']:
                    allow_access = False
        
        # Create consolidated violation if any violations occurred
        violation = None
        if violations:
            violation = await self._create_consolidated_violation(violations, request)
        
        # Remove duplicates from actions
        unique_actions = list(set(actions_to_take))
        
        return allow_access, unique_actions, violation
    
    async def _find_applicable_policies(
        self,
        request: DataAccessRequest,
        classified_fields: Dict[str, GeologicalDataField],
        geological_sensitivity: GeologicalDataSensitivity
    ) -> List[DLPPolicy]:
        """Find policies applicable to the request"""
        
        applicable_policies = []
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            # Check data type match
            if policy.data_types != ["all"] and request.data_type not in policy.data_types:
                continue
            
            # Check classification match
            data_classifications = [field.classification for field in classified_fields.values()]
            if not any(cls in policy.classifications for cls in data_classifications):
                continue
            
            # Check geological sensitivity match
            if geological_sensitivity not in policy.geological_sensitivities:
                continue
            
            applicable_policies.append(policy)
        
        # Sort by priority (lower number = higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        return applicable_policies
    
    async def _evaluate_policy(
        self,
        policy: DLPPolicy,
        request: DataAccessRequest,
        data: Dict[str, Any],
        classified_fields: Dict[str, GeologicalDataField],
        geological_sensitivity: GeologicalDataSensitivity,
        context: GeologicalDataContext
    ) -> Dict[str, Any]:
        """Evaluate a specific policy against the request"""
        
        violation_details = {}
        actions = []
        
        # Check policy conditions
        conditions = policy.conditions
        
        # Check clearance requirements
        if conditions.get('require_clearance', False):
            if not await self._check_user_clearance(request.user_id, geological_sensitivity):
                violation_details['clearance_violation'] = True
                actions.extend(policy.actions)
        
        # Check concurrent access limits
        if 'max_concurrent_access' in conditions:
            current_access_count = await self._get_concurrent_access_count(request.resource_id)
            if current_access_count >= conditions['max_concurrent_access']:
                violation_details['concurrent_access_violation'] = True
                actions.extend(policy.actions)
        
        # Check time restrictions
        if 'time_restrictions' in conditions:
            if not await self._check_time_restrictions(conditions['time_restrictions']):
                violation_details['time_restriction_violation'] = True
                actions.extend(policy.actions)
        
        # Check export restrictions
        if request.operation == 'export':
            if 'max_export_size_mb' in conditions:
                data_size_mb = await self._estimate_data_size(data)
                if data_size_mb > conditions['max_export_size_mb']:
                    violation_details['export_size_violation'] = True
                    actions.extend(policy.actions)
            
            if 'allowed_destinations' in conditions:
                if request.destination not in conditions['allowed_destinations']:
                    violation_details['destination_violation'] = True
                    actions.extend(policy.actions)
        
        # Check PII protection
        if conditions.get('auto_detect_pii', False):
            pii_fields = [name for name, field in classified_fields.items() if field.pii_detected]
            if pii_fields:
                violation_details['pii_detected'] = pii_fields
                if conditions.get('mask_pii_fields', False):
                    actions.append(DLPAction.MASK)
                else:
                    actions.extend(policy.actions)
        
        # Create violation if any conditions were violated
        violation = None
        if violation_details:
            violation = DLPViolation(
                violation_id=str(uuid.uuid4()),
                violation_type=DLPViolationType.POLICY_VIOLATION,
                policy_id=policy.policy_id,
                user_id=request.user_id,
                resource_id=request.resource_id,
                data_classification=max([field.classification for field in classified_fields.values()]),
                geological_sensitivity=geological_sensitivity,
                violation_details=violation_details,
                actions_taken=actions,
                severity=self._calculate_violation_severity(violation_details, geological_sensitivity)
            )
        
        return {
            'violation': violation,
            'actions': actions,
            'policy_id': policy.policy_id
        }
    
    async def _check_user_clearance(self, user_id: str, sensitivity: GeologicalDataSensitivity) -> bool:
        """Check if user has required clearance level"""
        # This would integrate with user management system
        # For now, return a simplified check
        user_clearance = await self.redis_client.get(f"user_clearance:{user_id}")
        if not user_clearance:
            return False
        
        clearance_levels = {
            GeologicalDataSensitivity.PUBLIC_RESEARCH: 0,
            GeologicalDataSensitivity.ACADEMIC_USE: 1,
            GeologicalDataSensitivity.COMMERCIAL_SENSITIVE: 2,
            GeologicalDataSensitivity.DISASTER_PREPAREDNESS: 3,
            GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE: 4,
            GeologicalDataSensitivity.NATIONAL_SECURITY: 5
        }
        
        required_level = clearance_levels.get(sensitivity, 0)
        user_level = clearance_levels.get(GeologicalDataSensitivity(user_clearance), 0)
        
        return user_level >= required_level
    
    async def _get_concurrent_access_count(self, resource_id: str) -> int:
        """Get current concurrent access count for resource"""
        count = await self.redis_client.get(f"concurrent_access:{resource_id}")
        return int(count) if count else 0
    
    async def _check_time_restrictions(self, time_restrictions: Dict[str, str]) -> bool:
        """Check if current time is within allowed hours"""
        from datetime import datetime
        
        current_time = datetime.now().time()
        start_time = datetime.strptime(time_restrictions['start'], '%H:%M').time()
        end_time = datetime.strptime(time_restrictions['end'], '%H:%M').time()
        
        if start_time <= end_time:
            return start_time <= current_time <= end_time
        else:  # Crosses midnight
            return current_time >= start_time or current_time <= end_time
    
    async def _estimate_data_size(self, data: Dict[str, Any]) -> float:
        """Estimate data size in MB"""
        # Simple estimation based on JSON serialization
        json_str = json.dumps(data)
        size_bytes = len(json_str.encode('utf-8'))
        return size_bytes / (1024 * 1024)  # Convert to MB
    
    def _calculate_violation_severity(
        self,
        violation_details: Dict[str, Any],
        sensitivity: GeologicalDataSensitivity
    ) -> str:
        """Calculate violation severity"""
        
        base_severity = 1
        
        # Increase severity based on sensitivity
        if sensitivity == GeologicalDataSensitivity.NATIONAL_SECURITY:
            base_severity += 4
        elif sensitivity == GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE:
            base_severity += 3
        elif sensitivity == GeologicalDataSensitivity.DISASTER_PREPAREDNESS:
            base_severity += 2
        elif sensitivity == GeologicalDataSensitivity.COMMERCIAL_SENSITIVE:
            base_severity += 1
        
        # Increase severity based on violation types
        if violation_details.get('clearance_violation'):
            base_severity += 2
        if violation_details.get('destination_violation'):
            base_severity += 2
        if violation_details.get('pii_detected'):
            base_severity += 1
        
        # Map to severity levels
        if base_severity >= 6:
            return "critical"
        elif base_severity >= 4:
            return "high"
        elif base_severity >= 2:
            return "medium"
        else:
            return "low"
    
    async def _create_consolidated_violation(
        self,
        violations: List[DLPViolation],
        request: DataAccessRequest
    ) -> DLPViolation:
        """Create consolidated violation from multiple policy violations"""
        
        # Use the highest severity violation as base
        primary_violation = max(violations, key=lambda v: v.severity)
        
        # Consolidate details
        consolidated_details = {}
        all_actions = []
        
        for violation in violations:
            consolidated_details.update(violation.violation_details)
            all_actions.extend(violation.actions_taken)
        
        # Create consolidated violation
        consolidated_violation = DLPViolation(
            violation_id=str(uuid.uuid4()),
            violation_type=DLPViolationType.POLICY_VIOLATION,
            policy_id="consolidated",
            user_id=request.user_id,
            resource_id=request.resource_id,
            data_classification=primary_violation.data_classification,
            geological_sensitivity=primary_violation.geological_sensitivity,
            violation_details=consolidated_details,
            actions_taken=list(set(all_actions)),
            severity=primary_violation.severity
        )
        
        # Store violation
        await self._store_violation(consolidated_violation)
        
        return consolidated_violation
    
    async def _store_policy(self, policy: DLPPolicy):
        """Store DLP policy"""
        policy_data = {
            'name': policy.name,
            'description': policy.description,
            'data_types': policy.data_types,
            'classifications': [c.value for c in policy.classifications],
            'geological_sensitivities': [s.value for s in policy.geological_sensitivities],
            'conditions': policy.conditions,
            'actions': [a.value for a in policy.actions],
            'enabled': policy.enabled,
            'priority': policy.priority,
            'created_at': policy.created_at.isoformat(),
            'updated_at': policy.updated_at.isoformat()
        }
        
        await self.redis_client.set(
            f"dlp_policy:{policy.policy_id}",
            json.dumps(policy_data)
        )
    
    async def _store_violation(self, violation: DLPViolation):
        """Store DLP violation"""
        violation_data = {
            'violation_type': violation.violation_type.value,
            'policy_id': violation.policy_id,
            'user_id': violation.user_id,
            'resource_id': violation.resource_id,
            'data_classification': violation.data_classification.value,
            'geological_sensitivity': violation.geological_sensitivity.value,
            'violation_details': violation.violation_details,
            'actions_taken': [a.value for a in violation.actions_taken],
            'timestamp': violation.timestamp.isoformat(),
            'severity': violation.severity,
            'resolved': violation.resolved
        }
        
        await self.redis_client.set(
            f"dlp_violation:{violation.violation_id}",
            json.dumps(violation_data)
        )
        
        # Add to violations list for reporting
        await self.redis_client.lpush(
            "dlp_violations_list",
            violation.violation_id
        )


class DataMaskingEngine:
    """Data masking and redaction engine"""
    
    def __init__(self):
        self.masking_rules = self._load_masking_rules()
    
    def _load_masking_rules(self) -> Dict[str, Any]:
        """Load data masking rules"""
        return {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': '***@***.***'
            },
            'phone': {
                'pattern': r'\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b',
                'replacement': '***-***-****'
            },
            'coordinates': {
                'pattern': r'-?\d+\.\d+',
                'replacement': lambda m: str(round(float(m.group()), 1))  # Reduce precision
            },
            'precise_location': {
                'fields': ['latitude', 'longitude', 'elevation'],
                'precision_reduction': 2  # Reduce to 2 decimal places
            },
            'sensitive_measurements': {
                'fields': ['magnitude', 'pressure', 'temperature', 'concentration'],
                'noise_factor': 0.05  # Add 5% random noise
            }
        }
    
    async def mask_data(
        self,
        data: Dict[str, Any],
        classified_fields: Dict[str, GeologicalDataField],
        masking_level: str = "standard"
    ) -> Dict[str, Any]:
        """Mask sensitive data based on classification"""
        
        masked_data = data.copy()
        
        for field_name, field_info in classified_fields.items():
            if field_name not in masked_data:
                continue
            
            field_value = masked_data[field_name]
            
            # Apply masking based on classification and PII detection
            if field_info.pii_detected:
                masked_data[field_name] = await self._mask_pii(field_value)
            elif field_info.classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                masked_data[field_name] = await self._mask_sensitive_geological_data(
                    field_name, field_value, masking_level
                )
            elif field_info.geological_sensitivity == "high":
                masked_data[field_name] = await self._reduce_precision(field_name, field_value)
        
        return masked_data
    
    async def _mask_pii(self, value: Any) -> str:
        """Mask personally identifiable information"""
        if not isinstance(value, str):
            value = str(value)
        
        # Apply PII masking rules
        for rule_name, rule in self.masking_rules.items():
            if rule_name in ['email', 'phone'] and 'pattern' in rule:
                value = re.sub(rule['pattern'], rule['replacement'], value)
        
        return value
    
    async def _mask_sensitive_geological_data(
        self,
        field_name: str,
        value: Any,
        masking_level: str
    ) -> Any:
        """Mask sensitive geological data"""
        
        field_lower = field_name.lower()
        
        # Location data masking
        if any(keyword in field_lower for keyword in ['location', 'coordinate', 'lat', 'lon']):
            if isinstance(value, (int, float)):
                if masking_level == "high":
                    return round(value, 0)  # Round to nearest integer
                else:
                    return round(value, 1)  # Reduce precision
        
        # Measurement data masking
        if any(keyword in field_lower for keyword in ['magnitude', 'pressure', 'temperature', 'grade']):
            if isinstance(value, (int, float)):
                # Add random noise
                noise_factor = 0.1 if masking_level == "high" else 0.05
                noise = np.random.normal(0, abs(value) * noise_factor)
                return round(value + noise, 2)
        
        # Probability and forecast masking
        if any(keyword in field_lower for keyword in ['probability', 'forecast', 'prediction']):
            if isinstance(value, (int, float)):
                # Reduce precision for probabilities
                if 0 <= value <= 1:
                    return round(value, 1)  # Round to 1 decimal place
                else:
                    return round(value, 0)  # Round to integer
        
        # Default: return original value
        return value
    
    async def _reduce_precision(self, field_name: str, value: Any) -> Any:
        """Reduce precision of numerical data"""
        
        if isinstance(value, float):
            # Reduce precision based on field type
            if any(keyword in field_name.lower() for keyword in ['coordinate', 'lat', 'lon']):
                return round(value, 2)  # 2 decimal places for coordinates
            else:
                return round(value, 1)  # 1 decimal place for other measurements
        
        return value


if __name__ == "__main__":
    # Example usage and testing
    async def test_dlp_engine():
        """Test the DLP engine functionality"""
        
        # Initialize DLP engine
        config = {
            'redis_url': 'redis://localhost:6379'
        }
        
        dlp_engine = DLPPolicyEngine(config)
        await dlp_engine.initialize()
        
        # Test data access request
        test_data = {
            'seismic_magnitude': 7.2,
            'latitude': 46.852,
            'longitude': -121.760,
            'eruption_probability': 0.85,
            'researcher_email': 'scientist@university.edu',
            'magma_chamber_pressure': 150.5,
            'alert_level': 'orange'
        }
        
        context = GeologicalDataContext(
            data_source="usgs_monitoring",
            geological_domain="volcanic_monitoring",
            temporal_scope="real-time",
            spatial_scope="regional",
            research_purpose="eruption_forecasting",
            commercial_value="low",
            public_safety_impact="critical",
            national_security_relevance="medium"
        )
        
        request = DataAccessRequest(
            request_id=str(uuid.uuid4()),
            user_id="researcher_001",
            user_roles=["volcanologist", "researcher"],
            resource_id="mount_st_helens_data",
            data_type="volcanic_monitoring",
            operation="read",
            purpose="research"
        )
        
        # Evaluate access request
        allow_access, actions, violation = await dlp_engine.evaluate_access_request(
            request, test_data, context
        )
        
        print(f"Access allowed: {allow_access}")
        print(f"Actions to take: {[action.value for action in actions]}")
        if violation:
            print(f"Violation detected: {violation.violation_type.value}")
            print(f"Severity: {violation.severity}")
            print(f"Details: {violation.violation_details}")
        
        # Test data masking
        masking_engine = DataMaskingEngine()
        classifier = DataClassifier(config)
        
        classified_fields = await classifier.classify_data(test_data, "volcanic_monitoring")
        masked_data = await masking_engine.mask_data(test_data, classified_fields)
        
        print(f"\nOriginal data: {test_data}")
        print(f"Masked data: {masked_data}")
    
    # Run test
    asyncio.run(test_dlp_engine())