"""
Geological Device Trust Manager with Advanced Security Validation
Implements device trust assessment with geological data sensitivity validation
"""

import asyncio
import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import aioredis
import asyncpg
import ssl
import socket
from cryptography import x509
from cryptography.hazmat.primitives import hashes
import psutil
import platform

from models.security_models import DeviceTrust, GeologicalContext, ThreatAlert
from utils.encryption import AdvancedEncryption
from utils.audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)

@dataclass
class DeviceFingerprint:
    """Comprehensive device fingerprint for geological equipment"""
    device_id: str
    hardware_fingerprint: str
    software_fingerprint: str
    network_fingerprint: str
    geological_equipment_signature: Optional[str] = None
    security_features: Dict = None
    compliance_status: Dict = None
    trust_indicators: List[str] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.security_features is None:
            self.security_features = {}
        if self.compliance_status is None:
            self.compliance_status = {}
        if self.trust_indicators is None:
            self.trust_indicators = []
        if self.risk_factors is None:
            self.risk_factors = []

@dataclass
class GeologicalEquipmentProfile:
    """Profile for geological equipment with security requirements"""
    equipment_id: str
    equipment_type: str  # 'seismometer', 'spectrometer', 'drone', 'field_tablet'
    manufacturer: str
    model: str
    firmware_version: str
    security_certification: Optional[str] = None
    data_sensitivity_handling: int = 1  # 1-5 scale
    encryption_capabilities: List[str] = None
    authentication_methods: List[str] = None
    network_protocols: List[str] = None
    geological_domains: List[str] = None
    field_deployment_capable: bool = False
    remote_access_enabled: bool = False
    
    def __post_init__(self):
        if self.encryption_capabilities is None:
            self.encryption_capabilities = []
        if self.authentication_methods is None:
            self.authentication_methods = []
        if self.network_protocols is None:
            self.network_protocols = []
        if self.geological_domains is None:
            self.geological_domains = []

class GeologicalDeviceTrustManager:
    """
    Advanced Device Trust Manager with geological equipment awareness
    Implements comprehensive device validation and trust scoring
    """
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.encryption = AdvancedEncryption()
        self.audit_logger = SecurityAuditLogger()
        
        # Device trust models
        self.trust_calculation_model = None
        self.anomaly_detection_model = None
        
        # Geological equipment profiles
        self.equipment_profiles = {}
        
        # Device trust cache
        self.device_trust_cache = {}
        
        # Security compliance requirements
        self.compliance_requirements = {
            'geological_field_equipment': {
                'encryption': ['AES-256', 'ChaCha20'],
                'authentication': ['certificate', 'biometric', 'multi_factor'],
                'network_security': ['TLS1.3', 'VPN', 'certificate_pinning'],
                'data_protection': ['encrypted_storage', 'secure_boot', 'tamper_detection'],
                'update_mechanism': ['signed_updates', 'secure_channel'],
                'audit_logging': ['comprehensive', 'tamper_proof']
            },
            'laboratory_equipment': {
                'encryption': ['AES-256'],
                'authentication': ['certificate', 'password'],
                'network_security': ['TLS1.2', 'network_isolation'],
                'data_protection': ['encrypted_storage', 'access_control'],
                'update_mechanism': ['signed_updates'],
                'audit_logging': ['standard']
            },
            'mobile_devices': {
                'encryption': ['device_encryption', 'app_encryption'],
                'authentication': ['biometric', 'pin', 'pattern'],
                'network_security': ['VPN', 'certificate_validation'],
                'data_protection': ['app_sandboxing', 'remote_wipe'],
                'update_mechanism': ['automatic_updates', 'app_store_validation'],
                'audit_logging': ['application_level']
            },
            'workstations': {
                'encryption': ['full_disk_encryption', 'file_encryption'],
                'authentication': ['multi_factor', 'certificate', 'biometric'],
                'network_security': ['firewall', 'endpoint_protection'],
                'data_protection': ['access_control', 'dlp', 'backup_encryption'],
                'update_mechanism': ['managed_updates', 'patch_management'],
                'audit_logging': ['comprehensive', 'centralized']
            }
        }
        
    async def initialize(self):
        """Initialize device trust manager with geological equipment awareness"""
        logger.info("Initializing Geological Device Trust Manager")
        
        # Initialize Redis for device state caching
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/4",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Initialize PostgreSQL for device data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://prometheus:geological_security@localhost:5432/prometheus_security",
            min_size=5,
            max_size=20
        )
        
        # Initialize geological equipment profiles
        await self._initialize_equipment_profiles()
        
        # Create database tables
        await self._create_device_trust_tables()
        
        # Initialize trust calculation models
        await self._initialize_trust_models()
        
        # Load existing device trust data
        await self._load_device_trust_cache()
        
        logger.info("Geological Device Trust Manager initialized successfully")
        
    async def _initialize_equipment_profiles(self):
        """Initialize geological equipment profiles with security requirements"""
        
        self.equipment_profiles = {
            'broadband_seismometer': GeologicalEquipmentProfile(
                equipment_id='broadband_seismometer',
                equipment_type='seismometer',
                manufacturer='Nanometrics',
                model='Trillium_Compact',
                firmware_version='2.1.0',
                security_certification='FIPS_140_2_Level_2',
                data_sensitivity_handling=4,
                encryption_capabilities=['AES-256-GCM', 'ChaCha20-Poly1305'],
                authentication_methods=['X.509_certificate', 'HMAC_authentication'],
                network_protocols=['HTTPS', 'MQTT_TLS', 'SSH'],
                geological_domains=['seismology', 'earthquake_monitoring'],
                field_deployment_capable=True,
                remote_access_enabled=True
            ),
            
            'portable_xrf_spectrometer': GeologicalEquipmentProfile(
                equipment_id='portable_xrf_spectrometer',
                equipment_type='spectrometer',
                manufacturer='Olympus',
                model='Vanta_Element',
                firmware_version='3.2.1',
                security_certification='CE_Marking',
                data_sensitivity_handling=3,
                encryption_capabilities=['AES-128', 'Bluetooth_LE_Security'],
                authentication_methods=['PIN', 'biometric_fingerprint'],
                network_protocols=['Bluetooth_LE', 'WiFi_WPA3', 'USB_encrypted'],
                geological_domains=['geochemistry', 'mineralogy', 'field_analysis'],
                field_deployment_capable=True,
                remote_access_enabled=False
            ),
            
            'geological_drone_system': GeologicalEquipmentProfile(
                equipment_id='geological_drone_system',
                equipment_type='drone',
                manufacturer='DJI',
                model='Matrice_300_RTK_Geological',
                firmware_version='1.5.2',
                security_certification='Part_107_Compliant',
                data_sensitivity_handling=4,
                encryption_capabilities=['AES-256', 'Secure_Video_Transmission'],
                authentication_methods=['controller_pairing', 'geofencing', 'return_to_home'],
                network_protocols=['OcuSync_3.0_Encrypted', 'HTTPS', 'RTMP_Secure'],
                geological_domains=['remote_sensing', 'structural_geology', 'volcanic_monitoring'],
                field_deployment_capable=True,
                remote_access_enabled=True
            ),
            
            'field_tablet_rugged': GeologicalEquipmentProfile(
                equipment_id='field_tablet_rugged',
                equipment_type='field_tablet',
                manufacturer='Getac',
                model='F110_Geological',
                firmware_version='Android_11_Geological',
                security_certification='MIL_STD_810H',
                data_sensitivity_handling=3,
                encryption_capabilities=['Android_FDE', 'App_Level_Encryption'],
                authentication_methods=['biometric', 'smart_card', 'PIN'],
                network_protocols=['4G_LTE', 'WiFi_WPA3', 'Bluetooth_5.0', 'GPS_Encrypted'],
                geological_domains=['field_geology', 'data_collection', 'sample_management'],
                field_deployment_capable=True,
                remote_access_enabled=True
            ),
            
            'laboratory_mass_spectrometer': GeologicalEquipmentProfile(
                equipment_id='laboratory_mass_spectrometer',
                equipment_type='mass_spectrometer',
                manufacturer='Thermo_Fisher',
                model='Neptune_Plus_MC_ICP_MS',
                firmware_version='4.1.3',
                security_certification='ISO_17025_Compliant',
                data_sensitivity_handling=4,
                encryption_capabilities=['AES-256', 'Database_Encryption'],
                authentication_methods=['user_authentication', 'role_based_access'],
                network_protocols=['Ethernet_Secure', 'HTTPS', 'SFTP'],
                geological_domains=['geochemistry', 'isotope_analysis', 'geochronology'],
                field_deployment_capable=False,
                remote_access_enabled=True
            ),
            
            'workstation_geological_modeling': GeologicalEquipmentProfile(
                equipment_id='workstation_geological_modeling',
                equipment_type='workstation',
                manufacturer='Dell',
                model='Precision_7000_Geological',
                firmware_version='BIOS_2.15.0',
                security_certification='Common_Criteria_EAL4',
                data_sensitivity_handling=5,
                encryption_capabilities=['BitLocker', 'TPM_2.0', 'Hardware_Security_Module'],
                authentication_methods=['smart_card', 'biometric', 'multi_factor'],
                network_protocols=['Ethernet_802.1X', 'VPN_IPSec', 'TLS_1.3'],
                geological_domains=['3d_modeling', 'numerical_simulation', 'data_analysis'],
                field_deployment_capable=False,
                remote_access_enabled=True
            )
        }
        
    async def _create_device_trust_tables(self):
        """Create device trust database tables"""
        async with self.db_pool.acquire() as conn:
            # Device registry table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS device_registry (
                    device_id VARCHAR(100) PRIMARY KEY,
                    device_fingerprint TEXT NOT NULL,
                    device_type VARCHAR(50) NOT NULL,
                    equipment_profile_id VARCHAR(100),
                    owner_user_id VARCHAR(50),
                    geological_context JSONB,
                    security_features JSONB,
                    compliance_status JSONB,
                    trust_score FLOAT DEFAULT 0.0,
                    risk_factors JSONB,
                    last_assessment TIMESTAMP DEFAULT NOW(),
                    registered_at TIMESTAMP DEFAULT NOW(),
                    active BOOLEAN DEFAULT true
                )
            """)
            
            # Device trust assessments table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS device_trust_assessments (
                    assessment_id VARCHAR(100) PRIMARY KEY,
                    device_id VARCHAR(100) NOT NULL,
                    assessment_type VARCHAR(50) NOT NULL,
                    trust_score FLOAT NOT NULL,
                    risk_factors JSONB,
                    security_compliance JSONB,
                    geological_compliance JSONB,
                    assessment_data JSONB,
                    assessed_at TIMESTAMP DEFAULT NOW(),
                    assessor_id VARCHAR(50),
                    valid_until TIMESTAMP
                )
            """)
            
            # Device security events table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS device_security_events (
                    event_id VARCHAR(100) PRIMARY KEY,
                    device_id VARCHAR(100) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    description TEXT,
                    event_data JSONB,
                    geological_context JSONB,
                    detected_at TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT false,
                    resolved_at TIMESTAMP,
                    resolution_actions JSONB
                )
            """)
            
    async def _initialize_trust_models(self):
        """Initialize device trust calculation models"""
        
        # Trust calculation model
        self.trust_calculation_model = {
            'base_trust_factors': {
                'device_authentication': 0.20,
                'encryption_capabilities': 0.15,
                'security_certifications': 0.15,
                'compliance_status': 0.15,
                'geological_equipment_validation': 0.10,
                'network_security': 0.10,
                'update_status': 0.10,
                'behavioral_analysis': 0.05
            },
            'risk_penalty_factors': {
                'known_vulnerabilities': -0.30,
                'outdated_firmware': -0.20,
                'failed_compliance': -0.25,
                'suspicious_behavior': -0.15,
                'unauthorized_modifications': -0.40,
                'network_anomalies': -0.10
            },
            'geological_bonus_factors': {
                'certified_geological_equipment': +0.10,
                'field_hardened': +0.05,
                'tamper_resistant': +0.08,
                'environmental_certified': +0.03
            }
        }
        
        # Anomaly detection model (placeholder for ML model)
        self.anomaly_detection_model = {
            'behavioral_baselines': {},
            'anomaly_thresholds': {
                'network_pattern_deviation': 0.7,
                'usage_pattern_deviation': 0.6,
                'location_pattern_deviation': 0.8,
                'geological_workflow_deviation': 0.9
            }
        }
        
    async def _load_device_trust_cache(self):
        """Load device trust data into cache"""
        async with self.db_pool.acquire() as conn:
            devices = await conn.fetch("""
                SELECT device_id, trust_score, risk_factors, last_assessment
                FROM device_registry 
                WHERE active = true
            """)
            
            for device in devices:
                self.device_trust_cache[device['device_id']] = {
                    'trust_score': device['trust_score'],
                    'risk_factors': device['risk_factors'] or [],
                    'last_assessment': device['last_assessment'],
                    'cache_timestamp': datetime.utcnow()
                }
                
    async def validate_device(
        self, 
        source_ip: str, 
        user_agent: str, 
        device_id: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None
    ) -> Dict:
        """
        Comprehensive device validation with geological equipment awareness
        Returns device trust assessment and validation results
        """
        try:
            # Generate device fingerprint
            device_fingerprint = await self._generate_device_fingerprint(
                source_ip, user_agent, device_id
            )
            
            # Check if device is registered
            device_record = await self._get_device_record(device_fingerprint.device_id)
            
            if not device_record:
                # New device - perform initial assessment
                trust_assessment = await self._perform_initial_device_assessment(
                    device_fingerprint, geological_context
                )
                
                # Register device
                await self._register_device(device_fingerprint, trust_assessment, geological_context)
            else:
                # Existing device - validate and update trust
                trust_assessment = await self._perform_device_trust_update(
                    device_record, device_fingerprint, geological_context
                )
                
            # Check for security anomalies
            anomalies = await self._detect_device_anomalies(
                device_fingerprint, geological_context
            )
            
            if anomalies:
                trust_assessment['trust_score'] *= 0.7  # Reduce trust for anomalies
                trust_assessment['risk_factors'].extend(anomalies)
                
            # Validate geological equipment compliance
            geological_compliance = await self._validate_geological_equipment_compliance(
                device_fingerprint, geological_context
            )
            
            # Update cache
            self.device_trust_cache[device_fingerprint.device_id] = {
                'trust_score': trust_assessment['trust_score'],
                'risk_factors': trust_assessment['risk_factors'],
                'last_assessment': datetime.utcnow(),
                'cache_timestamp': datetime.utcnow()
            }
            
            # Log device validation
            await self.audit_logger.log_device_validation(
                device_fingerprint.device_id, trust_assessment, geological_context
            )
            
            return {
                'device_id': device_fingerprint.device_id,
                'trusted': trust_assessment['trust_score'] >= 0.7,
                'trust_score': trust_assessment['trust_score'],
                'risk_factors': trust_assessment['risk_factors'],
                'geological_compliance': geological_compliance,
                'security_features': trust_assessment.get('security_features', {}),
                'recommendations': trust_assessment.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Device validation error: {str(e)}")
            return {
                'device_id': 'unknown',
                'trusted': False,
                'trust_score': 0.0,
                'risk_factors': ['validation_error'],
                'error': str(e)
            }
            
    async def _generate_device_fingerprint(
        self, 
        source_ip: str, 
        user_agent: str, 
        device_id: Optional[str]
    ) -> DeviceFingerprint:
        """Generate comprehensive device fingerprint"""
        
        # Hardware fingerprint (simplified for demonstration)
        hardware_info = {
            'ip_address': source_ip,
            'user_agent': user_agent,
            'provided_device_id': device_id
        }
        
        hardware_fingerprint = hashlib.sha256(
            json.dumps(hardware_info, sort_keys=True).encode()
        ).hexdigest()
        
        # Software fingerprint
        software_info = {
            'user_agent_parsed': self._parse_user_agent(user_agent),
            'capabilities': self._detect_capabilities(user_agent)
        }
        
        software_fingerprint = hashlib.sha256(
            json.dumps(software_info, sort_keys=True).encode()
        ).hexdigest()
        
        # Network fingerprint
        network_info = {
            'source_ip': source_ip,
            'ip_geolocation': await self._get_ip_geolocation(source_ip),
            'network_characteristics': await self._analyze_network_characteristics(source_ip)
        }
        
        network_fingerprint = hashlib.sha256(
            json.dumps(network_info, sort_keys=True).encode()
        ).hexdigest()
        
        # Generate unique device ID
        if not device_id:
            device_id = f"dev_{hardware_fingerprint[:16]}"
            
        return DeviceFingerprint(
            device_id=device_id,
            hardware_fingerprint=hardware_fingerprint,
            software_fingerprint=software_fingerprint,
            network_fingerprint=network_fingerprint
        )
        
    async def _perform_initial_device_assessment(
        self, 
        fingerprint: DeviceFingerprint, 
        geological_context: Optional[GeologicalContext]
    ) -> Dict:
        """Perform initial device trust assessment"""
        
        trust_score = 0.5  # Start with neutral trust
        risk_factors = []
        security_features = {}
        recommendations = []
        
        # Analyze device characteristics
        device_analysis = await self._analyze_device_characteristics(fingerprint)
        
        # Check against known geological equipment
        equipment_match = await self._match_geological_equipment(fingerprint, geological_context)
        
        if equipment_match:
            # Known geological equipment - higher base trust
            trust_score += 0.2
            security_features.update(equipment_match.get('security_features', {}))
        else:
            # Unknown device - additional scrutiny
            risk_factors.append('unknown_device_type')
            recommendations.append('Device registration and certification required')
            
        # Security feature assessment
        security_assessment = await self._assess_security_features(fingerprint)
        trust_score += security_assessment['trust_bonus']
        risk_factors.extend(security_assessment['risk_factors'])
        security_features.update(security_assessment['features'])
        
        # Network security assessment
        network_assessment = await self._assess_network_security(fingerprint)
        trust_score += network_assessment['trust_bonus']
        risk_factors.extend(network_assessment['risk_factors'])
        
        # Ensure trust score is within bounds
        trust_score = max(0.0, min(1.0, trust_score))
        
        return {
            'trust_score': trust_score,
            'risk_factors': risk_factors,
            'security_features': security_features,
            'recommendations': recommendations,
            'equipment_match': equipment_match
        }
        
    async def _validate_geological_equipment_compliance(
        self, 
        fingerprint: DeviceFingerprint, 
        geological_context: Optional[GeologicalContext]
    ) -> Dict:
        """Validate geological equipment compliance requirements"""
        
        compliance_results = {
            'compliant': True,
            'compliance_level': 'basic',
            'violations': [],
            'recommendations': []
        }
        
        if not geological_context:
            return compliance_results
            
        # Determine required compliance level based on geological context
        required_compliance = self._determine_required_compliance(geological_context)
        
        # Check equipment profile compliance
        equipment_profile = await self._get_equipment_profile(fingerprint)
        
        if equipment_profile:
            compliance_check = await self._check_equipment_compliance(
                equipment_profile, required_compliance
            )
            compliance_results.update(compliance_check)
        else:
            compliance_results['violations'].append('No registered equipment profile')
            compliance_results['compliant'] = False
            
        return compliance_results
        
    async def health_check(self) -> Dict:
        """Health check for device trust manager"""
        try:
            # Check Redis connection
            redis_healthy = await self.redis_client.ping()
            
            # Check database connection
            async with self.db_pool.acquire() as conn:
                db_healthy = await conn.fetchval("SELECT 1") == 1
                
            # Check trust models
            models_healthy = all([
                self.trust_calculation_model is not None,
                self.anomaly_detection_model is not None
            ])
            
            return {
                "healthy": redis_healthy and db_healthy and models_healthy,
                "components": {
                    "redis": redis_healthy,
                    "database": db_healthy,
                    "trust_models": models_healthy,
                    "equipment_profiles": len(self.equipment_profiles),
                    "cached_devices": len(self.device_trust_cache)
                }
            }
        except Exception as e:
            logger.error(f"Device trust manager health check failed: {str(e)}")
            return {"healthy": False, "error": str(e)}