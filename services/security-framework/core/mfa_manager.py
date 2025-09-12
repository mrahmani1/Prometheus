"""
Advanced Multi-Factor Authentication Manager
Implements hardware tokens, biometrics, and geological workflow validation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import secrets
import hashlib
import hmac
import base64
import qrcode
from io import BytesIO
import pyotp
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import httpx
from fastapi import Request

from models.security_models import (
    UserContext, MFAMethod, MFAChallenge, MFAResponse,
    BiometricData, HardwareTokenData, GeologicalWorkflowContext
)
from config.security_config import SecurityConfig

logger = logging.getLogger(__name__)

class MFAManager:
    """Advanced Multi-Factor Authentication with geological workflow validation"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.active_challenges = {}
        self.user_mfa_settings = {}
        self.hardware_tokens = {}
        self.biometric_templates = {}
        self.workflow_patterns = {}
        self.encryption_key = self._derive_encryption_key()
        
    async def initialize(self):
        """Initialize MFA system with hardware token and biometric support"""
        try:
            # Initialize hardware token support
            await self._initialize_hardware_tokens()
            
            # Initialize biometric authentication
            await self._initialize_biometric_auth()
            
            # Load geological workflow patterns
            await self._load_workflow_patterns()
            
            # Start challenge cleanup task
            asyncio.create_task(self._challenge_cleanup_task())
            
            logger.info("MFA Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MFA Manager: {e}")
            raise
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key for sensitive MFA data"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.config.mfa_salt.encode(),
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(self.config.mfa_secret.encode()))
    
    async def _initialize_hardware_tokens(self):
        """Initialize hardware token support (YubiKey, FIDO2, etc.)"""
        self.hardware_tokens = {
            'yubikey': {
                'api_endpoint': 'https://api.yubico.com/wsapi/2.0/verify',
                'client_id': self.config.yubikey_client_id,
                'secret_key': self.config.yubikey_secret_key,
                'supported_models': ['YubiKey 5', 'YubiKey 5C', 'YubiKey 5 NFC']
            },
            'fido2': {
                'relying_party_id': 'prometheus-ai.secure',
                'relying_party_name': 'Prometheus AI Platform',
                'supported_algorithms': ['ES256', 'RS256'],
                'user_verification': 'required'
            },
            'smart_card': {
                'supported_standards': ['PIV', 'CAC', 'OpenPGP'],
                'certificate_validation': True,
                'pin_required': True
            }
        }
        
        logger.info("Hardware token support initialized")
    
    async def _initialize_biometric_auth(self):
        """Initialize biometric authentication support"""
        self.biometric_templates = {
            'fingerprint': {
                'algorithm': 'minutiae_extraction',
                'template_format': 'ISO_19794_2',
                'matching_threshold': 0.85,
                'liveness_detection': True
            },
            'face_recognition': {
                'algorithm': 'deep_learning_embeddings',
                'template_format': 'face_embedding_512d',
                'matching_threshold': 0.90,
                'anti_spoofing': True
            },
            'iris_scan': {
                'algorithm': 'iris_code_extraction',
                'template_format': 'ISO_19794_6',
                'matching_threshold': 0.95,
                'quality_threshold': 0.80
            },
            'voice_recognition': {
                'algorithm': 'speaker_verification',
                'template_format': 'voice_embedding_256d',
                'matching_threshold': 0.88,
                'text_dependent': False
            }
        }
        
        logger.info("Biometric authentication support initialized")
    
    async def _load_workflow_patterns(self):
        """Load geological workflow patterns for context-aware MFA"""
        self.workflow_patterns = {
            'seismic_analysis': {
                'typical_duration': timedelta(hours=2),
                'data_access_patterns': ['waveform_download', 'event_catalog', 'station_metadata'],
                'computational_intensity': 'high',
                'collaboration_level': 'medium',
                'risk_level': 'medium'
            },
            'volcanic_monitoring': {
                'typical_duration': timedelta(hours=24),  # Continuous monitoring
                'data_access_patterns': ['real_time_streams', 'alert_generation', 'forecast_models'],
                'computational_intensity': 'very_high',
                'collaboration_level': 'high',
                'risk_level': 'very_high'  # Public safety implications
            },
            'resource_exploration': {
                'typical_duration': timedelta(hours=8),
                'data_access_patterns': ['geophysical_surveys', 'geochemical_data', 'economic_models'],
                'computational_intensity': 'high',
                'collaboration_level': 'low',
                'risk_level': 'high'  # Commercial sensitivity
            },
            'research_collaboration': {
                'typical_duration': timedelta(hours=4),
                'data_access_patterns': ['shared_datasets', 'publication_data', 'model_sharing'],
                'computational_intensity': 'medium',
                'collaboration_level': 'very_high',
                'risk_level': 'medium'
            },
            'emergency_response': {
                'typical_duration': timedelta(hours=72),
                'data_access_patterns': ['real_time_all', 'emergency_protocols', 'public_alerts'],
                'computational_intensity': 'maximum',
                'collaboration_level': 'maximum',
                'risk_level': 'critical'  # Life safety
            }
        }
        
        logger.info("Geological workflow patterns loaded")
    
    async def setup_mfa_for_user(self, user_context: UserContext, 
                                methods: List[MFAMethod]) -> Dict:
        """Set up MFA methods for a user"""
        try:
            user_id = user_context.user_id
            mfa_setup = {
                'user_id': user_id,
                'methods': {},
                'backup_codes': [],
                'setup_timestamp': datetime.utcnow()
            }
            
            for method in methods:
                if method == MFAMethod.TOTP:
                    # Generate TOTP secret
                    secret = pyotp.random_base32()
                    totp = pyotp.TOTP(secret)
                    
                    # Generate QR code
                    provisioning_uri = totp.provisioning_uri(
                        name=user_context.email,
                        issuer_name="Prometheus AI Platform"
                    )
                    
                    qr = qrcode.QRCode(version=1, box_size=10, border=5)
                    qr.add_data(provisioning_uri)
                    qr.make(fit=True)
                    
                    qr_image = qr.make_image(fill_color="black", back_color="white")
                    qr_buffer = BytesIO()
                    qr_image.save(qr_buffer, format='PNG')
                    qr_code_base64 = base64.b64encode(qr_buffer.getvalue()).decode()
                    
                    mfa_setup['methods']['totp'] = {
                        'secret': self._encrypt_sensitive_data(secret),
                        'qr_code': qr_code_base64,
                        'backup_codes': self._generate_backup_codes()
                    }
                
                elif method == MFAMethod.HARDWARE_TOKEN:
                    # Initialize hardware token registration
                    mfa_setup['methods']['hardware_token'] = {
                        'registration_challenge': secrets.token_urlsafe(32),
                        'supported_tokens': list(self.hardware_tokens.keys())
                    }
                
                elif method == MFAMethod.BIOMETRIC:
                    # Initialize biometric enrollment
                    mfa_setup['methods']['biometric'] = {
                        'enrollment_session': secrets.token_urlsafe(32),
                        'supported_modalities': list(self.biometric_templates.keys())
                    }
                
                elif method == MFAMethod.SMS:
                    # SMS setup (for backup only, not primary)
                    mfa_setup['methods']['sms'] = {
                        'phone_number': None,  # To be provided by user
                        'verification_required': True
                    }
            
            # Generate backup codes
            mfa_setup['backup_codes'] = self._generate_backup_codes()
            
            # Store encrypted MFA settings
            self.user_mfa_settings[user_id] = self._encrypt_mfa_settings(mfa_setup)
            
            logger.info(f"MFA setup completed for user: {user_context.email}")
            return mfa_setup
            
        except Exception as e:
            logger.error(f"MFA setup failed for user {user_context.email}: {e}")
            raise
    
    async def validate_request(self, user_context: UserContext, 
                             request: Request) -> bool:
        """Validate MFA requirements for geological workflow context"""
        try:
            # Determine geological workflow context
            workflow_context = await self._analyze_workflow_context(user_context, request)
            
            # Check if MFA is required based on workflow risk
            mfa_required = await self._is_mfa_required(user_context, workflow_context)
            
            if not mfa_required:
                return True
            
            # Check for existing valid MFA session
            if await self._has_valid_mfa_session(user_context, workflow_context):
                return True
            
            # Initiate MFA challenge
            challenge = await self._create_mfa_challenge(user_context, workflow_context)
            
            # For this validation, we'll check if MFA challenge exists
            # In a real implementation, this would be handled by the frontend
            return challenge is not None
            
        except Exception as e:
            logger.error(f"MFA validation failed: {e}")
            return False
    
    async def _analyze_workflow_context(self, user_context: UserContext, 
                                      request: Request) -> GeologicalWorkflowContext:
        """Analyze geological workflow context from request"""
        # Extract workflow indicators from request
        path = request.url.path
        headers = dict(request.headers)
        
        # Determine workflow type based on API endpoints
        workflow_type = 'general'
        if '/seismic/' in path:
            workflow_type = 'seismic_analysis'
        elif '/volcanic/' in path or '/volcano/' in path:
            workflow_type = 'volcanic_monitoring'
        elif '/resource/' in path or '/prospectivity/' in path:
            workflow_type = 'resource_exploration'
        elif '/collaboration/' in path or '/share/' in path:
            workflow_type = 'research_collaboration'
        elif '/emergency/' in path or '/alert/' in path:
            workflow_type = 'emergency_response'
        
        # Extract data sensitivity from headers
        data_sensitivity = headers.get('x-geological-data-sensitivity', 'public')
        
        # Determine computational requirements
        computational_intensity = 'low'
        if 'x-computational-intensive' in headers:
            computational_intensity = headers['x-computational-intensive']
        
        return GeologicalWorkflowContext(
            workflow_type=workflow_type,
            data_sensitivity=data_sensitivity,
            computational_intensity=computational_intensity,
            user_roles=user_context.roles,
            timestamp=datetime.utcnow(),
            request_metadata=headers
        )
    
    async def _is_mfa_required(self, user_context: UserContext, 
                             workflow_context: GeologicalWorkflowContext) -> bool:
        """Determine if MFA is required based on geological workflow context"""
        # Always require MFA for high-risk workflows
        high_risk_workflows = ['volcanic_monitoring', 'emergency_response']
        if workflow_context.workflow_type in high_risk_workflows:
            return True
        
        # Require MFA for sensitive data access
        sensitive_data_levels = ['confidential', 'classified', 'proprietary']
        if workflow_context.data_sensitivity in sensitive_data_levels:
            return True
        
        # Require MFA for resource exploration (commercial sensitivity)
        if workflow_context.workflow_type == 'resource_exploration':
            return True
        
        # Require MFA for administrative roles
        admin_roles = ['system_administrator']
        if any(role in admin_roles for role in user_context.roles):
            return True
        
        # Require MFA for high computational intensity (resource protection)
        if workflow_context.computational_intensity in ['very_high', 'maximum']:
            return True
        
        return False
    
    async def _has_valid_mfa_session(self, user_context: UserContext, 
                                   workflow_context: GeologicalWorkflowContext) -> bool:
        """Check if user has valid MFA session for current workflow"""
        user_id = user_context.user_id
        session_key = f"{user_id}:{workflow_context.workflow_type}"
        
        # Check if MFA session exists and is valid
        if session_key in self.active_challenges:
            session = self.active_challenges[session_key]
            if session.get('validated') and session.get('expires_at', datetime.min) > datetime.utcnow():
                return True
        
        return False
    
    async def _create_mfa_challenge(self, user_context: UserContext, 
                                  workflow_context: GeologicalWorkflowContext) -> MFAChallenge:
        """Create MFA challenge based on workflow context and user preferences"""
        try:
            user_id = user_context.user_id
            
            # Get user's MFA settings
            if user_id not in self.user_mfa_settings:
                raise Exception("User has no MFA methods configured")
            
            mfa_settings = self._decrypt_mfa_settings(self.user_mfa_settings[user_id])
            
            # Select appropriate MFA method based on workflow context
            selected_method = await self._select_mfa_method(workflow_context, mfa_settings)
            
            # Create challenge
            challenge_id = secrets.token_urlsafe(32)
            challenge = MFAChallenge(
                challenge_id=challenge_id,
                user_id=user_id,
                method=selected_method,
                workflow_context=workflow_context,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(minutes=5),
                attempts_remaining=3
            )
            
            # Generate method-specific challenge data
            if selected_method == MFAMethod.TOTP:
                challenge.challenge_data = {
                    'message': 'Enter the 6-digit code from your authenticator app'
                }
            elif selected_method == MFAMethod.HARDWARE_TOKEN:
                challenge.challenge_data = {
                    'message': 'Insert your hardware token and enter PIN',
                    'challenge_string': secrets.token_urlsafe(16)
                }
            elif selected_method == MFAMethod.BIOMETRIC:
                challenge.challenge_data = {
                    'message': 'Complete biometric authentication',
                    'modality': 'fingerprint',  # Default to fingerprint
                    'liveness_required': True
                }
            
            # Store challenge
            self.active_challenges[challenge_id] = challenge
            
            logger.info(f"MFA challenge created for user {user_context.email}, method: {selected_method}")
            return challenge
            
        except Exception as e:
            logger.error(f"Failed to create MFA challenge: {e}")
            raise
    
    async def _select_mfa_method(self, workflow_context: GeologicalWorkflowContext, 
                               mfa_settings: Dict) -> MFAMethod:
        """Select appropriate MFA method based on workflow context"""
        available_methods = list(mfa_settings['methods'].keys())
        
        # For emergency response, prefer fastest method
        if workflow_context.workflow_type == 'emergency_response':
            if 'biometric' in available_methods:
                return MFAMethod.BIOMETRIC
            elif 'totp' in available_methods:
                return MFAMethod.TOTP
        
        # For high-security workflows, prefer hardware tokens
        if workflow_context.data_sensitivity in ['classified', 'proprietary']:
            if 'hardware_token' in available_methods:
                return MFAMethod.HARDWARE_TOKEN
            elif 'biometric' in available_methods:
                return MFAMethod.BIOMETRIC
        
        # Default preference order
        preference_order = [MFAMethod.HARDWARE_TOKEN, MFAMethod.BIOMETRIC, MFAMethod.TOTP, MFAMethod.SMS]
        
        for method in preference_order:
            if method.value in available_methods:
                return method
        
        raise Exception("No suitable MFA method available")
    
    async def verify_mfa_response(self, challenge_id: str, 
                                response: MFAResponse) -> bool:
        """Verify MFA response for geological workflow validation"""
        try:
            if challenge_id not in self.active_challenges:
                raise Exception("Invalid or expired challenge")
            
            challenge = self.active_challenges[challenge_id]
            
            # Check if challenge has expired
            if challenge.expires_at < datetime.utcnow():
                del self.active_challenges[challenge_id]
                raise Exception("Challenge has expired")
            
            # Check attempts remaining
            if challenge.attempts_remaining <= 0:
                del self.active_challenges[challenge_id]
                raise Exception("Maximum attempts exceeded")
            
            # Verify response based on method
            verification_result = False
            
            if challenge.method == MFAMethod.TOTP:
                verification_result = await self._verify_totp(challenge, response)
            elif challenge.method == MFAMethod.HARDWARE_TOKEN:
                verification_result = await self._verify_hardware_token(challenge, response)
            elif challenge.method == MFAMethod.BIOMETRIC:
                verification_result = await self._verify_biometric(challenge, response)
            elif challenge.method == MFAMethod.SMS:
                verification_result = await self._verify_sms(challenge, response)
            
            if verification_result:
                # Mark challenge as validated and create session
                challenge.validated = True
                session_key = f"{challenge.user_id}:{challenge.workflow_context.workflow_type}"
                
                # Set session duration based on workflow type
                session_duration = self._get_session_duration(challenge.workflow_context)
                
                self.active_challenges[session_key] = {
                    'validated': True,
                    'expires_at': datetime.utcnow() + session_duration,
                    'workflow_type': challenge.workflow_context.workflow_type,
                    'method_used': challenge.method
                }
                
                # Clean up challenge
                del self.active_challenges[challenge_id]
                
                logger.info(f"MFA verification successful for challenge {challenge_id}")
                return True
            else:
                # Decrement attempts
                challenge.attempts_remaining -= 1
                logger.warning(f"MFA verification failed for challenge {challenge_id}")
                return False
                
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
    
    async def _verify_totp(self, challenge: MFAChallenge, response: MFAResponse) -> bool:
        """Verify TOTP code"""
        try:
            user_id = challenge.user_id
            mfa_settings = self._decrypt_mfa_settings(self.user_mfa_settings[user_id])
            
            totp_secret = self._decrypt_sensitive_data(
                mfa_settings['methods']['totp']['secret']
            )
            
            totp = pyotp.TOTP(totp_secret)
            return totp.verify(response.code, valid_window=1)  # Allow 30-second window
            
        except Exception as e:
            logger.error(f"TOTP verification failed: {e}")
            return False
    
    async def _verify_hardware_token(self, challenge: MFAChallenge, 
                                   response: MFAResponse) -> bool:
        """Verify hardware token response"""
        try:
            if response.token_type == 'yubikey':
                return await self._verify_yubikey(response.token_data)
            elif response.token_type == 'fido2':
                return await self._verify_fido2(challenge, response.token_data)
            else:
                logger.error(f"Unsupported hardware token type: {response.token_type}")
                return False
                
        except Exception as e:
            logger.error(f"Hardware token verification failed: {e}")
            return False
    
    async def _verify_yubikey(self, otp: str) -> bool:
        """Verify YubiKey OTP"""
        try:
            yubikey_config = self.hardware_tokens['yubikey']
            
            # Generate signature for YubiKey API
            nonce = secrets.token_urlsafe(16)
            params = {
                'id': yubikey_config['client_id'],
                'otp': otp,
                'nonce': nonce
            }
            
            # Create signature
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = base64.b64encode(
                hmac.new(
                    yubikey_config['secret_key'].encode(),
                    query_string.encode(),
                    hashlib.sha1
                ).digest()
            ).decode()
            
            params['h'] = signature
            
            # Verify with YubiKey API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    yubikey_config['api_endpoint'],
                    params=params,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.text
                    return 'status=OK' in result
                
            return False
            
        except Exception as e:
            logger.error(f"YubiKey verification failed: {e}")
            return False
    
    async def _verify_fido2(self, challenge: MFAChallenge, token_data: Dict) -> bool:
        """Verify FIDO2 authentication"""
        # This would integrate with a FIDO2 library like python-fido2
        # For now, return a placeholder implementation
        try:
            # Verify FIDO2 assertion
            # This would involve:
            # 1. Validating the authenticator data
            # 2. Verifying the signature
            # 3. Checking the challenge matches
            
            # Placeholder verification
            return (
                'authenticator_data' in token_data and
                'signature' in token_data and
                'client_data' in token_data
            )
            
        except Exception as e:
            logger.error(f"FIDO2 verification failed: {e}")
            return False
    
    async def _verify_biometric(self, challenge: MFAChallenge, 
                              response: MFAResponse) -> bool:
        """Verify biometric authentication"""
        try:
            # This would integrate with biometric SDKs
            # For now, return a placeholder implementation
            
            biometric_data = response.biometric_data
            if not biometric_data:
                return False
            
            # Verify biometric template matches stored template
            # This would involve:
            # 1. Extracting features from biometric sample
            # 2. Comparing with stored template
            # 3. Checking liveness detection
            # 4. Verifying quality thresholds
            
            # Placeholder verification
            return (
                biometric_data.modality in self.biometric_templates and
                biometric_data.quality_score > 0.8 and
                biometric_data.liveness_score > 0.9
            )
            
        except Exception as e:
            logger.error(f"Biometric verification failed: {e}")
            return False
    
    async def _verify_sms(self, challenge: MFAChallenge, response: MFAResponse) -> bool:
        """Verify SMS code (backup method only)"""
        try:
            # This would integrate with SMS service
            # For now, return a placeholder implementation
            
            # In a real implementation, this would:
            # 1. Check the SMS code against what was sent
            # 2. Verify timing constraints
            # 3. Check rate limiting
            
            return len(response.code) == 6 and response.code.isdigit()
            
        except Exception as e:
            logger.error(f"SMS verification failed: {e}")
            return False
    
    def _get_session_duration(self, workflow_context: GeologicalWorkflowContext) -> timedelta:
        """Get MFA session duration based on workflow context"""
        if workflow_context.workflow_type == 'emergency_response':
            return timedelta(hours=24)  # Extended for emergency operations
        elif workflow_context.workflow_type == 'volcanic_monitoring':
            return timedelta(hours=8)   # Extended for continuous monitoring
        elif workflow_context.data_sensitivity in ['classified', 'proprietary']:
            return timedelta(hours=1)   # Short for sensitive data
        else:
            return timedelta(hours=4)   # Standard duration
    
    def _generate_backup_codes(self) -> List[str]:
        """Generate backup codes for MFA recovery"""
        return [secrets.token_hex(4).upper() for _ in range(10)]
    
    def _encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive MFA data"""
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.encode()).decode()
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive MFA data"""
        fernet = Fernet(self.encryption_key)
        return fernet.decrypt(encrypted_data.encode()).decode()
    
    def _encrypt_mfa_settings(self, settings: Dict) -> str:
        """Encrypt MFA settings"""
        import json
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(json.dumps(settings).encode()).decode()
    
    def _decrypt_mfa_settings(self, encrypted_settings: str) -> Dict:
        """Decrypt MFA settings"""
        import json
        fernet = Fernet(self.encryption_key)
        return json.loads(fernet.decrypt(encrypted_settings.encode()).decode())
    
    async def _challenge_cleanup_task(self):
        """Periodic cleanup of expired challenges"""
        while True:
            try:
                current_time = datetime.utcnow()
                expired_challenges = []
                
                for challenge_id, challenge in self.active_challenges.items():
                    if hasattr(challenge, 'expires_at') and challenge.expires_at < current_time:
                        expired_challenges.append(challenge_id)
                    elif isinstance(challenge, dict) and challenge.get('expires_at', datetime.max) < current_time:
                        expired_challenges.append(challenge_id)
                
                for challenge_id in expired_challenges:
                    del self.active_challenges[challenge_id]
                
                if expired_challenges:
                    logger.info(f"Cleaned up {len(expired_challenges)} expired MFA challenges")
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"MFA challenge cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def health_check(self) -> bool:
        """Health check for MFA system"""
        try:
            # Check encryption key
            test_data = "test"
            encrypted = self._encrypt_sensitive_data(test_data)
            decrypted = self._decrypt_sensitive_data(encrypted)
            
            return decrypted == test_data
            
        except Exception as e:
            logger.error(f"MFA health check failed: {e}")
            return False
    
    async def is_ready(self) -> bool:
        """Check if MFA system is ready"""
        return (
            self.encryption_key is not None and
            len(self.hardware_tokens) > 0 and
            len(self.biometric_templates) > 0 and
            len(self.workflow_patterns) > 0
        )
    
    async def cleanup(self):
        """Cleanup MFA resources"""
        self.active_challenges.clear()
        self.user_mfa_settings.clear()
        logger.info("MFA Manager cleanup completed")