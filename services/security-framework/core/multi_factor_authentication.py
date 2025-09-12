"""
Advanced Multi-Factor Authentication with Geological Workflow Validation
Implements hardware tokens, biometrics, and geological context-aware MFA
"""

import asyncio
import logging
import secrets
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pyotp
import aioredis
import asyncpg
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest
import face_recognition

from models.security_models import (
    MFAChallenge, MFAResponse, BiometricData, 
    GeologicalWorkflowContext, HardwareTokenData
)
from utils.encryption import AdvancedEncryption
from utils.audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)

@dataclass
class MFAMethod:
    """Advanced MFA method with geological context awareness"""
    method_id: str
    method_type: str  # 'totp', 'hardware_token', 'biometric', 'geological_workflow'
    name: str
    description: str
    security_level: int  # 1-5, higher is more secure
    geological_context_required: bool
    hardware_requirements: List[str]
    backup_methods: List[str]
    
@dataclass
class GeologicalWorkflowChallenge:
    """Geological workflow-based authentication challenge"""
    challenge_id: str
    workflow_type: str  # 'seismic_analysis', 'volcanic_monitoring', 'sample_classification'
    challenge_data: Dict
    expected_response: Dict
    difficulty_level: int
    time_limit: int  # seconds
    geological_domain: str

class AdvancedMFAManager:
    """
    Advanced Multi-Factor Authentication with geological domain expertise
    Implements hardware tokens, biometrics, and geological workflow validation
    """
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.encryption = AdvancedEncryption()
        self.audit_logger = SecurityAuditLogger()
        
        # MFA methods configuration
        self.mfa_methods = self._initialize_mfa_methods()
        
        # Biometric models
        self.face_recognition_model = None
        self.fingerprint_matcher = None
        self.voice_recognition_model = None
        
        # Geological workflow challenges
        self.workflow_challenges = {}
        
        # Hardware token integration
        self.hardware_token_providers = {
            'yubikey': {'api_key': 'yubikey_api_key', 'client_id': 'yubikey_client'},
            'rsa_securid': {'server_url': 'https://rsa-server.com', 'realm': 'geological'},
            'fido2': {'relying_party': 'prometheus-ai.com', 'origin': 'https://prometheus-ai.com'}
        }
        
    async def initialize(self):
        """Initialize MFA manager with geological domain awareness"""
        logger.info("Initializing Advanced MFA Manager")
        
        # Initialize Redis for MFA session management
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/1",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Initialize PostgreSQL for MFA data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://prometheus:geological_security@localhost:5432/prometheus_security",
            min_size=5,
            max_size=20
        )
        
        # Initialize biometric models
        await self._initialize_biometric_models()
        
        # Initialize geological workflow challenges
        await self._initialize_geological_challenges()
        
        # Create MFA tables
        await self._create_mfa_tables()
        
        logger.info("Advanced MFA Manager initialized successfully")
        
    def _initialize_mfa_methods(self) -> Dict[str, MFAMethod]:
        """Initialize comprehensive MFA methods with geological context"""
        return {
            'totp_standard': MFAMethod(
                method_id='totp_std',
                method_type='totp',
                name='Time-based OTP (Standard)',
                description='Standard TOTP with 6-digit codes',
                security_level=2,
                geological_context_required=False,
                hardware_requirements=[],
                backup_methods=['backup_codes']
            ),
            'totp_geological': MFAMethod(
                method_id='totp_geo',
                method_type='totp',
                name='Geological Context TOTP',
                description='TOTP with geological workflow validation',
                security_level=4,
                geological_context_required=True,
                hardware_requirements=[],
                backup_methods=['geological_workflow', 'backup_codes']
            ),
            'yubikey_otp': MFAMethod(
                method_id='yubikey_otp',
                method_type='hardware_token',
                name='YubiKey OTP',
                description='YubiKey hardware token authentication',
                security_level=4,
                geological_context_required=False,
                hardware_requirements=['yubikey'],
                backup_methods=['totp_geological', 'backup_codes']
            ),
            'fido2_webauthn': MFAMethod(
                method_id='fido2_webauthn',
                method_type='hardware_token',
                name='FIDO2/WebAuthn',
                description='FIDO2 hardware security key',
                security_level=5,
                geological_context_required=False,
                hardware_requirements=['fido2_key'],
                backup_methods=['yubikey_otp', 'totp_geological']
            ),
            'biometric_face': MFAMethod(
                method_id='bio_face',
                method_type='biometric',
                name='Facial Recognition',
                description='AI-powered facial recognition with liveness detection',
                security_level=4,
                geological_context_required=False,
                hardware_requirements=['camera'],
                backup_methods=['biometric_fingerprint', 'totp_geological']
            ),
            'biometric_fingerprint': MFAMethod(
                method_id='bio_fingerprint',
                method_type='biometric',
                name='Fingerprint Recognition',
                description='Fingerprint biometric authentication',
                security_level=4,
                geological_context_required=False,
                hardware_requirements=['fingerprint_scanner'],
                backup_methods=['biometric_face', 'totp_geological']
            ),
            'geological_workflow': MFAMethod(
                method_id='geo_workflow',
                method_type='geological_workflow',
                name='Geological Workflow Challenge',
                description='Domain-specific geological knowledge verification',
                security_level=5,
                geological_context_required=True,
                hardware_requirements=[],
                backup_methods=['totp_geological', 'hardware_token']
            ),
            'adaptive_mfa': MFAMethod(
                method_id='adaptive_mfa',
                method_type='adaptive',
                name='AI-Adaptive MFA',
                description='AI-powered adaptive authentication based on risk assessment',
                security_level=5,
                geological_context_required=True,
                hardware_requirements=[],
                backup_methods=['geological_workflow', 'fido2_webauthn']
            )
        }
        
    async def _initialize_biometric_models(self):
        """Initialize biometric recognition models"""
        # Face recognition model (using face_recognition library)
        self.face_recognition_model = {
            'tolerance': 0.6,
            'model': 'hog',  # or 'cnn' for better accuracy
            'num_jitters': 1
        }
        
        # Fingerprint matching (placeholder for actual implementation)
        self.fingerprint_matcher = {
            'algorithm': 'minutiae_based',
            'threshold': 0.8
        }
        
        # Voice recognition (placeholder for actual implementation)
        self.voice_recognition_model = {
            'model_type': 'speaker_verification',
            'threshold': 0.85
        }
        
    async def _initialize_geological_challenges(self):
        """Initialize geological workflow challenges for authentication"""
        self.workflow_challenges = {
            'seismic_phase_identification': {
                'name': 'Seismic Phase Identification',
                'description': 'Identify P and S wave arrivals in seismogram',
                'difficulty_levels': {
                    1: {'noise_level': 0.1, 'time_limit': 60},
                    2: {'noise_level': 0.3, 'time_limit': 45},
                    3: {'noise_level': 0.5, 'time_limit': 30}
                },
                'geological_domain': 'seismology'
            },
            'volcanic_gas_classification': {
                'name': 'Volcanic Gas Classification',
                'description': 'Classify volcanic gas composition from spectra',
                'difficulty_levels': {
                    1: {'species_count': 3, 'time_limit': 90},
                    2: {'species_count': 5, 'time_limit': 60},
                    3: {'species_count': 8, 'time_limit': 45}
                },
                'geological_domain': 'volcanology'
            },
            'mineral_identification': {
                'name': 'Mineral Identification',
                'description': 'Identify minerals from optical properties',
                'difficulty_levels': {
                    1: {'mineral_count': 3, 'time_limit': 120},
                    2: {'mineral_count': 5, 'time_limit': 90},
                    3: {'mineral_count': 8, 'time_limit': 60}
                },
                'geological_domain': 'mineralogy'
            },
            'structural_geology_analysis': {
                'name': 'Structural Analysis',
                'description': 'Analyze geological structures and stress fields',
                'difficulty_levels': {
                    1: {'structure_complexity': 'simple', 'time_limit': 180},
                    2: {'structure_complexity': 'moderate', 'time_limit': 120},
                    3: {'structure_complexity': 'complex', 'time_limit': 90}
                },
                'geological_domain': 'structural_geology'
            }
        }
        
    async def _create_mfa_tables(self):
        """Create MFA-related database tables"""
        async with self.db_pool.acquire() as conn:
            # User MFA methods table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_mfa_methods (
                    user_id VARCHAR(50) NOT NULL,
                    method_id VARCHAR(50) NOT NULL,
                    method_data JSONB NOT NULL,
                    is_primary BOOLEAN DEFAULT false,
                    is_backup BOOLEAN DEFAULT false,
                    created_at TIMESTAMP DEFAULT NOW(),
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    PRIMARY KEY (user_id, method_id)
                )
            """)
            
            # MFA challenges table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS mfa_challenges (
                    challenge_id VARCHAR(100) PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    method_id VARCHAR(50) NOT NULL,
                    challenge_data JSONB NOT NULL,
                    expected_response JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    completed BOOLEAN DEFAULT false,
                    success BOOLEAN DEFAULT false
                )
            """)
            
            # Biometric templates table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS biometric_templates (
                    user_id VARCHAR(50) NOT NULL,
                    biometric_type VARCHAR(50) NOT NULL,
                    template_data BYTEA NOT NULL,
                    quality_score FLOAT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (user_id, biometric_type)
                )
            """)
            
    async def initiate_mfa_challenge(
        self, 
        user_id: str, 
        method_id: str,
        geological_context: Optional[Dict] = None
    ) -> MFAChallenge:
        """
        Initiate MFA challenge with geological context awareness
        Returns appropriate challenge based on method and context
        """
        try:
            method = self.mfa_methods.get(method_id)
            if not method:
                raise ValueError(f"Unknown MFA method: {method_id}")
                
            challenge_id = f"mfa_{user_id}_{method_id}_{secrets.token_urlsafe(16)}"
            
            # Generate challenge based on method type
            if method.method_type == 'totp':
                challenge = await self._generate_totp_challenge(
                    user_id, method, geological_context
                )
            elif method.method_type == 'hardware_token':
                challenge = await self._generate_hardware_token_challenge(
                    user_id, method, geological_context
                )
            elif method.method_type == 'biometric':
                challenge = await self._generate_biometric_challenge(
                    user_id, method, geological_context
                )
            elif method.method_type == 'geological_workflow':
                challenge = await self._generate_geological_workflow_challenge(
                    user_id, method, geological_context
                )
            elif method.method_type == 'adaptive':
                challenge = await self._generate_adaptive_challenge(
                    user_id, method, geological_context
                )
            else:
                raise ValueError(f"Unsupported MFA method type: {method.method_type}")
                
            # Store challenge in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO mfa_challenges 
                    (challenge_id, user_id, method_id, challenge_data, expected_response, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, 
                    challenge_id, user_id, method_id,
                    challenge.challenge_data, challenge.expected_response,
                    datetime.utcnow() + timedelta(minutes=5)
                )
                
            # Store in Redis for quick access
            await self.redis_client.setex(
                f"mfa_challenge:{challenge_id}",
                300,  # 5 minutes
                challenge.to_json()
            )
            
            await self.audit_logger.log_mfa_challenge_initiated(
                user_id, method_id, challenge_id, geological_context
            )
            
            return challenge
            
        except Exception as e:
            logger.error(f"MFA challenge initiation failed: {str(e)}")
            raise
            
    async def _generate_totp_challenge(
        self, 
        user_id: str, 
        method: MFAMethod,
        geological_context: Optional[Dict]
    ) -> MFAChallenge:
        """Generate TOTP challenge with optional geological context"""
        
        # Get user's TOTP secret
        async with self.db_pool.acquire() as conn:
            method_data = await conn.fetchval("""
                SELECT method_data FROM user_mfa_methods 
                WHERE user_id = $1 AND method_id = $2
            """, user_id, method.method_id)
            
        if not method_data:
            raise ValueError("TOTP not configured for user")
            
        totp_secret = method_data.get('secret')
        totp = pyotp.TOTP(totp_secret)
        
        challenge_data = {
            'type': 'totp',
            'qr_code': None,  # Only for setup
            'backup_codes_available': True,
            'time_window': 30,
            'digits': 6
        }
        
        # Add geological context validation if required
        if method.geological_context_required and geological_context:
            geological_challenge = await self._generate_geological_context_validation(
                geological_context
            )
            challenge_data['geological_validation'] = geological_challenge
            
        expected_response = {
            'totp_code': totp.now(),
            'geological_context': geological_context if method.geological_context_required else None
        }
        
        return MFAChallenge(
            challenge_id=f"totp_{secrets.token_urlsafe(16)}",
            method_id=method.method_id,
            challenge_type='totp',
            challenge_data=challenge_data,
            expected_response=expected_response,
            expires_at=datetime.utcnow() + timedelta(minutes=5),
            geological_context_required=method.geological_context_required
        )
        
    async def _generate_hardware_token_challenge(
        self, 
        user_id: str, 
        method: MFAMethod,
        geological_context: Optional[Dict]
    ) -> MFAChallenge:
        """Generate hardware token challenge (YubiKey, FIDO2, etc.)"""
        
        if method.method_id == 'yubikey_otp':
            challenge_data = {
                'type': 'yubikey_otp',
                'instructions': 'Touch your YubiKey to generate OTP',
                'expected_length': 44,
                'validation_server': 'https://api.yubico.com/wsapi/2.0/verify'
            }
            
        elif method.method_id == 'fido2_webauthn':
            # Generate FIDO2 challenge
            challenge_bytes = secrets.token_bytes(32)
            challenge_data = {
                'type': 'fido2_webauthn',
                'challenge': base64.b64encode(challenge_bytes).decode(),
                'rp_id': 'prometheus-ai.com',
                'user_verification': 'required',
                'timeout': 60000
            }
            
        else:
            raise ValueError(f"Unsupported hardware token method: {method.method_id}")
            
        return MFAChallenge(
            challenge_id=f"hw_{secrets.token_urlsafe(16)}",
            method_id=method.method_id,
            challenge_type='hardware_token',
            challenge_data=challenge_data,
            expected_response={},  # Will be validated against hardware token service
            expires_at=datetime.utcnow() + timedelta(minutes=2),
            geological_context_required=method.geological_context_required
        )
        
    async def _generate_biometric_challenge(
        self, 
        user_id: str, 
        method: MFAMethod,
        geological_context: Optional[Dict]
    ) -> MFAChallenge:
        """Generate biometric authentication challenge"""
        
        if method.method_id == 'bio_face':
            challenge_data = {
                'type': 'facial_recognition',
                'instructions': 'Look directly at the camera for facial recognition',
                'liveness_detection': True,
                'quality_threshold': 0.8,
                'max_attempts': 3
            }
            
        elif method.method_id == 'bio_fingerprint':
            challenge_data = {
                'type': 'fingerprint',
                'instructions': 'Place your finger on the scanner',
                'quality_threshold': 0.7,
                'max_attempts': 3
            }
            
        else:
            raise ValueError(f"Unsupported biometric method: {method.method_id}")
            
        return MFAChallenge(
            challenge_id=f"bio_{secrets.token_urlsafe(16)}",
            method_id=method.method_id,
            challenge_type='biometric',
            challenge_data=challenge_data,
            expected_response={},  # Will be validated against stored biometric template
            expires_at=datetime.utcnow() + timedelta(minutes=2),
            geological_context_required=method.geological_context_required
        )
        
    async def _generate_geological_workflow_challenge(
        self, 
        user_id: str, 
        method: MFAMethod,
        geological_context: Optional[Dict]
    ) -> MFAChallenge:
        """Generate geological workflow-based authentication challenge"""
        
        # Determine user's geological expertise from context
        user_domain = geological_context.get('primary_domain', 'general') if geological_context else 'general'
        
        # Select appropriate challenge based on user's domain
        available_challenges = [
            challenge_type for challenge_type, challenge_info in self.workflow_challenges.items()
            if challenge_info['geological_domain'] == user_domain or user_domain == 'general'
        ]
        
        if not available_challenges:
            available_challenges = list(self.workflow_challenges.keys())
            
        challenge_type = secrets.choice(available_challenges)
        challenge_info = self.workflow_challenges[challenge_type]
        
        # Determine difficulty based on user's experience level
        user_experience = geological_context.get('experience_level', 1) if geological_context else 1
        difficulty = min(user_experience, 3)
        
        # Generate specific challenge
        if challenge_type == 'seismic_phase_identification':
            challenge_data = await self._generate_seismic_challenge(difficulty)
        elif challenge_type == 'volcanic_gas_classification':
            challenge_data = await self._generate_gas_classification_challenge(difficulty)
        elif challenge_type == 'mineral_identification':
            challenge_data = await self._generate_mineral_identification_challenge(difficulty)
        elif challenge_type == 'structural_geology_analysis':
            challenge_data = await self._generate_structural_analysis_challenge(difficulty)
        else:
            raise ValueError(f"Unknown geological challenge type: {challenge_type}")
            
        challenge_data.update({
            'type': 'geological_workflow',
            'challenge_name': challenge_info['name'],
            'description': challenge_info['description'],
            'difficulty_level': difficulty,
            'geological_domain': challenge_info['geological_domain'],
            'time_limit': challenge_info['difficulty_levels'][difficulty]['time_limit']
        })
        
        return MFAChallenge(
            challenge_id=f"geo_{secrets.token_urlsafe(16)}",
            method_id=method.method_id,
            challenge_type='geological_workflow',
            challenge_data=challenge_data,
            expected_response=challenge_data.get('expected_response', {}),
            expires_at=datetime.utcnow() + timedelta(
                seconds=challenge_info['difficulty_levels'][difficulty]['time_limit'] + 30
            ),
            geological_context_required=True
        )
        
    async def _generate_seismic_challenge(self, difficulty: int) -> Dict:
        """Generate seismic phase identification challenge"""
        # This would generate synthetic seismogram data with known P and S arrivals
        # For demonstration, we'll create a simplified version
        
        p_arrival_time = secrets.randbelow(50) + 10  # P wave arrival between 10-60 seconds
        s_arrival_time = p_arrival_time + secrets.randbelow(30) + 15  # S wave 15-45 seconds later
        
        noise_level = self.workflow_challenges['seismic_phase_identification']['difficulty_levels'][difficulty]['noise_level']
        
        challenge_data = {
            'seismogram_data': f"synthetic_seismogram_{secrets.token_urlsafe(8)}",
            'sampling_rate': 100,  # Hz
            'duration': 120,  # seconds
            'noise_level': noise_level,
            'instructions': 'Identify the P and S wave arrival times in the seismogram',
            'expected_response': {
                'p_arrival': p_arrival_time,
                's_arrival': s_arrival_time,
                'tolerance': 2.0  # seconds
            }
        }
        
        return challenge_data
        
    async def _generate_gas_classification_challenge(self, difficulty: int) -> Dict:
        """Generate volcanic gas classification challenge"""
        
        gas_species = ['SO2', 'CO2', 'H2S', 'HCl', 'HF', 'H2O', 'CO', 'CH4']
        species_count = self.workflow_challenges['volcanic_gas_classification']['difficulty_levels'][difficulty]['species_count']
        
        selected_species = secrets.SystemRandom().sample(gas_species, species_count)
        concentrations = {species: secrets.randbelow(1000) + 10 for species in selected_species}
        
        challenge_data = {
            'spectra_data': f"synthetic_spectra_{secrets.token_urlsafe(8)}",
            'wavelength_range': [200, 400],  # nm
            'resolution': 0.1,  # nm
            'instructions': f'Identify the {species_count} gas species and their approximate concentrations',
            'expected_response': {
                'species': selected_species,
                'concentrations': concentrations,
                'tolerance': 0.1  # 10% tolerance
            }
        }
        
        return challenge_data
        
    async def _generate_mineral_identification_challenge(self, difficulty: int) -> Dict:
        """Generate mineral identification challenge"""
        
        minerals = ['quartz', 'feldspar', 'mica', 'pyroxene', 'olivine', 'amphibole', 'calcite', 'pyrite']
        mineral_count = self.workflow_challenges['mineral_identification']['difficulty_levels'][difficulty]['mineral_count']
        
        selected_minerals = secrets.SystemRandom().sample(minerals, mineral_count)
        
        challenge_data = {
            'thin_section_images': [f"mineral_image_{i}_{secrets.token_urlsafe(6)}" for i in range(mineral_count)],
            'optical_properties': {
                mineral: {
                    'birefringence': secrets.choice(['low', 'moderate', 'high']),
                    'pleochroism': secrets.choice(['none', 'weak', 'strong']),
                    'extinction': secrets.choice(['parallel', 'oblique', 'symmetric'])
                } for mineral in selected_minerals
            },
            'instructions': f'Identify the {mineral_count} minerals based on their optical properties',
            'expected_response': {
                'minerals': selected_minerals
            }
        }
        
        return challenge_data
        
    async def _generate_structural_analysis_challenge(self, difficulty: int) -> Dict:
        """Generate structural geology analysis challenge"""
        
        structure_types = ['normal_fault', 'reverse_fault', 'strike_slip_fault', 'fold', 'joint_set']
        complexity = self.workflow_challenges['structural_geology_analysis']['difficulty_levels'][difficulty]['structure_complexity']
        
        if complexity == 'simple':
            structures = [secrets.choice(structure_types)]
        elif complexity == 'moderate':
            structures = secrets.SystemRandom().sample(structure_types, 2)
        else:  # complex
            structures = secrets.SystemRandom().sample(structure_types, 3)
            
        challenge_data = {
            'geological_map': f"structural_map_{secrets.token_urlsafe(8)}",
            'cross_section': f"cross_section_{secrets.token_urlsafe(8)}",
            'stereonet_data': f"stereonet_{secrets.token_urlsafe(8)}",
            'instructions': f'Analyze the geological structures and determine stress field orientation',
            'expected_response': {
                'structures': structures,
                'stress_field': {
                    'sigma1_trend': secrets.randbelow(360),
                    'sigma1_plunge': secrets.randbelow(90),
                    'stress_regime': secrets.choice(['extensional', 'compressional', 'strike_slip'])
                }
            }
        }
        
        return challenge_data
        
    async def verify_mfa_response(
        self, 
        challenge_id: str, 
        response: MFAResponse
    ) -> bool:
        """
        Verify MFA response with geological context validation
        Returns True if authentication successful
        """
        try:
            # Retrieve challenge from Redis or database
            challenge_data = await self.redis_client.get(f"mfa_challenge:{challenge_id}")
            
            if not challenge_data:
                # Try database as fallback
                async with self.db_pool.acquire() as conn:
                    challenge_record = await conn.fetchrow("""
                        SELECT * FROM mfa_challenges 
                        WHERE challenge_id = $1 AND expires_at > NOW() AND NOT completed
                    """, challenge_id)
                    
                if not challenge_record:
                    logger.warning(f"MFA challenge not found or expired: {challenge_id}")
                    return False
                    
                challenge = MFAChallenge.from_db_record(challenge_record)
            else:
                challenge = MFAChallenge.from_json(challenge_data)
                
            # Increment attempt count
            await self._increment_challenge_attempts(challenge_id)
            
            # Verify response based on challenge type
            method = self.mfa_methods[challenge.method_id]
            
            if method.method_type == 'totp':
                success = await self._verify_totp_response(challenge, response)
            elif method.method_type == 'hardware_token':
                success = await self._verify_hardware_token_response(challenge, response)
            elif method.method_type == 'biometric':
                success = await self._verify_biometric_response(challenge, response)
            elif method.method_type == 'geological_workflow':
                success = await self._verify_geological_workflow_response(challenge, response)
            elif method.method_type == 'adaptive':
                success = await self._verify_adaptive_response(challenge, response)
            else:
                logger.error(f"Unknown MFA method type: {method.method_type}")
                return False
                
            # Mark challenge as completed
            await self._complete_challenge(challenge_id, success)
            
            # Update method usage statistics
            if success:
                await self._update_method_usage(challenge.user_id, challenge.method_id)
                
            await self.audit_logger.log_mfa_verification(
                challenge.user_id, challenge.method_id, challenge_id, success
            )
            
            return success
            
        except Exception as e:
            logger.error(f"MFA verification error: {str(e)}")
            await self.audit_logger.log_mfa_error(challenge_id, str(e))
            return False
            
    async def _verify_totp_response(self, challenge: MFAChallenge, response: MFAResponse) -> bool:
        """Verify TOTP response with optional geological context"""
        
        # Get user's TOTP secret
        async with self.db_pool.acquire() as conn:
            method_data = await conn.fetchval("""
                SELECT method_data FROM user_mfa_methods 
                WHERE user_id = $1 AND method_id = $2
            """, challenge.user_id, challenge.method_id)
            
        if not method_data:
            return False
            
        totp_secret = method_data.get('secret')
        totp = pyotp.TOTP(totp_secret)
        
        # Verify TOTP code (allow 1 time window before/after for clock skew)
        totp_valid = totp.verify(response.totp_code, valid_window=1)
        
        if not totp_valid:
            return False
            
        # Verify geological context if required
        if challenge.geological_context_required:
            geological_valid = await self._verify_geological_context(
                challenge.challenge_data.get('geological_validation'),
                response.geological_context
            )
            return geological_valid
            
        return True
        
    async def _verify_geological_context(self, expected_context: Dict, provided_context: Dict) -> bool:
        """Verify geological context validation"""
        if not expected_context or not provided_context:
            return False
            
        # Implement geological context validation logic
        # This would check geological knowledge, workflow patterns, etc.
        
        return True  # Simplified for demonstration
        
    async def health_check(self) -> Dict:
        """Health check for MFA manager"""
        try:
            # Check Redis connection
            redis_healthy = await self.redis_client.ping()
            
            # Check database connection
            async with self.db_pool.acquire() as conn:
                db_healthy = await conn.fetchval("SELECT 1") == 1
                
            return {
                "healthy": redis_healthy and db_healthy,
                "components": {
                    "redis": redis_healthy,
                    "database": db_healthy,
                    "mfa_methods": len(self.mfa_methods),
                    "geological_challenges": len(self.workflow_challenges),
                    "hardware_token_providers": len(self.hardware_token_providers)
                }
            }
        except Exception as e:
            logger.error(f"MFA manager health check failed: {str(e)}")
            return {"healthy": False, "error": str(e)}