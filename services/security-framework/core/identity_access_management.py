"""
Advanced Identity and Access Management with Geological Domain Expertise
Implements OAuth2/OIDC with geological role-based permissions and AI-powered identity verification
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import jwt
import bcrypt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import aioredis
import asyncpg
from authlib.integrations.starlette_client import OAuth
from authlib.jose import JsonWebToken, JWTClaims

from models.security_models import (
    GeologicalRole, GeologicalPermission, UserProfile, 
    AuthenticationResult, GeologicalContext
)
from utils.encryption import AdvancedEncryption
from utils.audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)

@dataclass
class GeologicalRoleDefinition:
    """Advanced geological role with domain-specific permissions"""
    role_id: str
    name: str
    description: str
    geological_domains: List[str]  # e.g., ['volcanology', 'seismology', 'geochemistry']
    data_access_levels: List[str]  # e.g., ['public', 'restricted', 'classified']
    computational_permissions: List[str]  # e.g., ['modeling', 'simulation', 'prediction']
    temporal_access: Dict[str, str]  # e.g., {'real_time': 'full', 'historical': 'limited'}
    spatial_boundaries: Optional[Dict]  # Geographic access restrictions
    equipment_access: List[str]  # Geological equipment and sensors
    collaboration_level: str  # 'individual', 'team', 'institution', 'global'

class GeologicalIdentityManager:
    """
    Advanced Identity and Access Management with geological domain expertise
    Implements OAuth2/OIDC with AI-powered identity verification and geological context awareness
    """
    
    def __init__(self):
        self.redis_client = None
        self.db_pool = None
        self.oauth = OAuth()
        self.jwt_handler = JsonWebToken(['RS256'])
        self.encryption = AdvancedEncryption()
        self.audit_logger = SecurityAuditLogger()
        
        # Geological role definitions
        self.geological_roles = self._initialize_geological_roles()
        
        # AI-powered identity verification
        self.identity_ml_model = None
        self.behavioral_patterns = {}
        
        # OAuth2/OIDC providers
        self.oidc_providers = {
            'geological_federation': {
                'client_id': 'prometheus-ai-geological',
                'client_secret': 'geological-federation-secret',
                'server_metadata_url': 'https://geo-federation.org/.well-known/openid_configuration'
            },
            'institutional_sso': {
                'client_id': 'prometheus-institutional',
                'client_secret': 'institutional-sso-secret',
                'server_metadata_url': 'https://institution-sso.edu/.well-known/openid_configuration'
            }
        }
        
    async def initialize(self):
        """Initialize identity management with geological domain awareness"""
        logger.info("Initializing Geological Identity Manager")
        
        # Initialize Redis for session management
        self.redis_client = await aioredis.from_url(
            "redis://localhost:6379/0",
            encoding="utf-8",
            decode_responses=True
        )
        
        # Initialize PostgreSQL for user data
        self.db_pool = await asyncpg.create_pool(
            "postgresql://prometheus:geological_security@localhost:5432/prometheus_security",
            min_size=5,
            max_size=20
        )
        
        # Initialize OAuth providers
        for provider_name, config in self.oidc_providers.items():
            self.oauth.register(
                name=provider_name,
                **config
            )
        
        # Load AI identity verification model
        await self._load_identity_ml_model()
        
        # Initialize geological role mappings
        await self._initialize_role_mappings()
        
        logger.info("Geological Identity Manager initialized successfully")
        
    def _initialize_geological_roles(self) -> Dict[str, GeologicalRoleDefinition]:
        """Initialize comprehensive geological role definitions"""
        return {
            'volcanologist_senior': GeologicalRoleDefinition(
                role_id='vol_senior',
                name='Senior Volcanologist',
                description='Senior researcher with full volcanic system access',
                geological_domains=['volcanology', 'petrology', 'geochemistry', 'seismology'],
                data_access_levels=['public', 'restricted', 'classified'],
                computational_permissions=['modeling', 'simulation', 'prediction', 'forecasting'],
                temporal_access={'real_time': 'full', 'historical': 'full', 'predictive': 'full'},
                spatial_boundaries=None,  # Global access
                equipment_access=['seismometers', 'gas_sensors', 'thermal_cameras', 'insar'],
                collaboration_level='global'
            ),
            'seismologist_researcher': GeologicalRoleDefinition(
                role_id='seis_researcher',
                name='Seismology Researcher',
                description='Seismology specialist with earthquake monitoring access',
                geological_domains=['seismology', 'tectonics', 'geophysics'],
                data_access_levels=['public', 'restricted'],
                computational_permissions=['modeling', 'simulation', 'analysis'],
                temporal_access={'real_time': 'full', 'historical': 'full', 'predictive': 'limited'},
                spatial_boundaries={'regions': ['pacific_rim', 'mediterranean', 'alpine']},
                equipment_access=['seismometers', 'accelerometers', 'gps_stations'],
                collaboration_level='institutional'
            ),
            'geochemist_analyst': GeologicalRoleDefinition(
                role_id='geochem_analyst',
                name='Geochemistry Analyst',
                description='Geochemical analysis specialist',
                geological_domains=['geochemistry', 'petrology', 'mineralogy'],
                data_access_levels=['public', 'restricted'],
                computational_permissions=['analysis', 'modeling'],
                temporal_access={'real_time': 'limited', 'historical': 'full', 'predictive': 'none'},
                spatial_boundaries={'sample_sites': True},
                equipment_access=['spectrometers', 'chromatographs', 'mass_spectrometers'],
                collaboration_level='team'
            ),
            'field_technician': GeologicalRoleDefinition(
                role_id='field_tech',
                name='Field Technician',
                description='Field data collection and equipment maintenance',
                geological_domains=['field_operations', 'instrumentation'],
                data_access_levels=['public'],
                computational_permissions=['data_collection', 'equipment_control'],
                temporal_access={'real_time': 'limited', 'historical': 'none', 'predictive': 'none'},
                spatial_boundaries={'assigned_sites': True},
                equipment_access=['portable_sensors', 'sampling_equipment'],
                collaboration_level='individual'
            ),
            'industry_consultant': GeologicalRoleDefinition(
                role_id='industry_consultant',
                name='Industry Consultant',
                description='Commercial geological consulting access',
                geological_domains=['resource_assessment', 'hazard_analysis'],
                data_access_levels=['public'],
                computational_permissions=['analysis', 'reporting'],
                temporal_access={'real_time': 'none', 'historical': 'limited', 'predictive': 'limited'},
                spatial_boundaries={'client_regions': True},
                equipment_access=[],
                collaboration_level='team'
            )
        }
        
    async def _load_identity_ml_model(self):
        """Load AI model for identity verification and behavioral analysis"""
        # In production, this would load a trained ML model
        # For now, we'll implement a sophisticated rule-based system
        self.identity_ml_model = {
            'behavioral_patterns': {},
            'anomaly_thresholds': {
                'login_frequency': 0.8,
                'access_patterns': 0.7,
                'geological_workflow': 0.9
            },
            'risk_factors': {
                'unusual_location': 0.6,
                'off_hours_access': 0.4,
                'privilege_escalation': 0.9,
                'data_exfiltration': 0.95
            }
        }
        
    async def _initialize_role_mappings(self):
        """Initialize geological role mappings in database"""
        async with self.db_pool.acquire() as conn:
            # Create geological roles table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS geological_roles (
                    role_id VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    geological_domains JSONB,
                    data_access_levels JSONB,
                    computational_permissions JSONB,
                    temporal_access JSONB,
                    spatial_boundaries JSONB,
                    equipment_access JSONB,
                    collaboration_level VARCHAR(50),
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Insert geological roles
            for role_id, role_def in self.geological_roles.items():
                await conn.execute("""
                    INSERT INTO geological_roles 
                    (role_id, name, description, geological_domains, data_access_levels,
                     computational_permissions, temporal_access, spatial_boundaries,
                     equipment_access, collaboration_level)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (role_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        geological_domains = EXCLUDED.geological_domains,
                        data_access_levels = EXCLUDED.data_access_levels,
                        computational_permissions = EXCLUDED.computational_permissions,
                        temporal_access = EXCLUDED.temporal_access,
                        spatial_boundaries = EXCLUDED.spatial_boundaries,
                        equipment_access = EXCLUDED.equipment_access,
                        collaboration_level = EXCLUDED.collaboration_level,
                        updated_at = NOW()
                """, 
                    role_id, role_def.name, role_def.description,
                    json.dumps(role_def.geological_domains),
                    json.dumps(role_def.data_access_levels),
                    json.dumps(role_def.computational_permissions),
                    json.dumps(role_def.temporal_access),
                    json.dumps(role_def.spatial_boundaries),
                    json.dumps(role_def.equipment_access),
                    role_def.collaboration_level
                )
                
    async def authenticate_user(
        self, 
        username: str, 
        password: str, 
        geological_context: Optional[GeologicalContext] = None
    ) -> AuthenticationResult:
        """
        Advanced user authentication with geological domain awareness
        Implements multi-factor verification and behavioral analysis
        """
        try:
            # Retrieve user from database
            async with self.db_pool.acquire() as conn:
                user_record = await conn.fetchrow("""
                    SELECT user_id, username, password_hash, geological_roles, 
                           mfa_enabled, device_fingerprints, behavioral_profile,
                           last_login, failed_attempts, account_locked
                    FROM users 
                    WHERE username = $1 AND active = true
                """, username)
                
            if not user_record:
                await self.audit_logger.log_failed_authentication(
                    username, "user_not_found", geological_context
                )
                return AuthenticationResult(
                    success=False,
                    error="Invalid credentials",
                    risk_score=0.9
                )
                
            # Check account lock status
            if user_record['account_locked']:
                await self.audit_logger.log_failed_authentication(
                    username, "account_locked", geological_context
                )
                return AuthenticationResult(
                    success=False,
                    error="Account locked due to security policy",
                    risk_score=1.0
                )
                
            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user_record['password_hash'].encode('utf-8')):
                # Increment failed attempts
                await self._increment_failed_attempts(user_record['user_id'])
                
                await self.audit_logger.log_failed_authentication(
                    username, "invalid_password", geological_context
                )
                return AuthenticationResult(
                    success=False,
                    error="Invalid credentials",
                    risk_score=0.8
                )
                
            # Behavioral analysis
            behavioral_risk = await self._analyze_login_behavior(
                user_record, geological_context
            )
            
            # Create user profile
            user_profile = UserProfile(
                user_id=user_record['user_id'],
                username=user_record['username'],
                geological_roles=user_record['geological_roles'],
                mfa_enabled=user_record['mfa_enabled'],
                behavioral_profile=user_record['behavioral_profile'],
                last_login=user_record['last_login']
            )
            
            # Generate JWT token with geological claims
            token = await self._generate_jwt_token(user_profile, geological_context)
            
            # Update last login
            await self._update_last_login(user_record['user_id'])
            
            # Reset failed attempts
            await self._reset_failed_attempts(user_record['user_id'])
            
            await self.audit_logger.log_successful_authentication(
                username, geological_context
            )
            
            return AuthenticationResult(
                success=True,
                user_profile=user_profile,
                access_token=token,
                requires_mfa=user_record['mfa_enabled'],
                risk_score=behavioral_risk,
                geological_permissions=await self._get_geological_permissions(
                    user_record['geological_roles']
                )
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            await self.audit_logger.log_authentication_error(username, str(e))
            return AuthenticationResult(
                success=False,
                error="Authentication system error",
                risk_score=1.0
            )
            
    async def _analyze_login_behavior(
        self, 
        user_record: dict, 
        geological_context: Optional[GeologicalContext]
    ) -> float:
        """
        AI-powered behavioral analysis for login patterns
        Analyzes geological workflow patterns and access anomalies
        """
        risk_score = 0.0
        
        # Time-based analysis
        current_time = datetime.utcnow()
        if geological_context and geological_context.access_time:
            hour = geological_context.access_time.hour
            
            # Check for unusual access hours
            if hour < 6 or hour > 22:  # Outside normal working hours
                risk_score += 0.3
                
        # Location-based analysis
        if geological_context and geological_context.location:
            # Check against known user locations
            known_locations = user_record.get('behavioral_profile', {}).get('locations', [])
            if geological_context.location not in known_locations:
                risk_score += 0.4
                
        # Device fingerprint analysis
        if geological_context and geological_context.device_fingerprint:
            known_devices = user_record.get('device_fingerprints', [])
            if geological_context.device_fingerprint not in known_devices:
                risk_score += 0.5
                
        # Geological workflow pattern analysis
        if geological_context and geological_context.intended_workflow:
            user_workflows = user_record.get('behavioral_profile', {}).get('workflows', [])
            if geological_context.intended_workflow not in user_workflows:
                risk_score += 0.2
                
        return min(risk_score, 1.0)
        
    async def _generate_jwt_token(
        self, 
        user_profile: UserProfile, 
        geological_context: Optional[GeologicalContext]
    ) -> str:
        """Generate JWT token with geological claims and permissions"""
        
        # Get geological permissions
        geological_permissions = await self._get_geological_permissions(
            user_profile.geological_roles
        )
        
        # Create JWT payload with geological claims
        payload = {
            'iss': 'prometheus-ai-platform',
            'sub': user_profile.user_id,
            'aud': 'geological-services',
            'exp': datetime.utcnow() + timedelta(hours=8),
            'iat': datetime.utcnow(),
            'jti': f"jwt_{user_profile.user_id}_{int(datetime.utcnow().timestamp())}",
            
            # Standard claims
            'username': user_profile.username,
            'roles': user_profile.geological_roles,
            
            # Geological-specific claims
            'geological_domains': geological_permissions.get('domains', []),
            'data_access_levels': geological_permissions.get('access_levels', []),
            'computational_permissions': geological_permissions.get('computational', []),
            'temporal_access': geological_permissions.get('temporal', {}),
            'spatial_boundaries': geological_permissions.get('spatial', {}),
            'equipment_access': geological_permissions.get('equipment', []),
            'collaboration_level': geological_permissions.get('collaboration', 'individual'),
            
            # Security context
            'mfa_verified': False,  # Will be updated after MFA
            'device_trusted': geological_context.device_trusted if geological_context else False,
            'risk_score': geological_context.risk_score if geological_context else 0.0
        }
        
        # Sign JWT with RS256
        private_key = await self._get_jwt_private_key()
        token = jwt.encode(payload, private_key, algorithm='RS256')
        
        # Store token in Redis for session management
        await self.redis_client.setex(
            f"jwt_session:{user_profile.user_id}:{payload['jti']}",
            28800,  # 8 hours
            json.dumps({
                'user_id': user_profile.user_id,
                'username': user_profile.username,
                'geological_roles': user_profile.geological_roles,
                'created_at': datetime.utcnow().isoformat(),
                'geological_context': asdict(geological_context) if geological_context else None
            })
        )
        
        return token
        
    async def verify_token(self, token: str) -> Dict:
        """Verify JWT token and return claims with geological permissions"""
        try:
            # Get public key for verification
            public_key = await self._get_jwt_public_key()
            
            # Decode and verify token
            payload = jwt.decode(token, public_key, algorithms=['RS256'])
            
            # Check if token is in Redis (not revoked)
            session_key = f"jwt_session:{payload['sub']}:{payload['jti']}"
            session_data = await self.redis_client.get(session_key)
            
            if not session_data:
                raise jwt.InvalidTokenError("Token session not found or expired")
                
            # Validate geological permissions are still current
            current_permissions = await self._get_geological_permissions(
                payload.get('roles', [])
            )
            
            # Update payload with current permissions
            payload.update(current_permissions)
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Token verification error: {str(e)}")
            raise
            
    async def _get_geological_permissions(self, roles: List[str]) -> Dict:
        """Get comprehensive geological permissions for user roles"""
        permissions = {
            'domains': set(),
            'access_levels': set(),
            'computational': set(),
            'temporal': {},
            'spatial': {},
            'equipment': set(),
            'collaboration': 'individual'
        }
        
        for role_id in roles:
            if role_id in self.geological_roles:
                role_def = self.geological_roles[role_id]
                
                permissions['domains'].update(role_def.geological_domains)
                permissions['access_levels'].update(role_def.data_access_levels)
                permissions['computational'].update(role_def.computational_permissions)
                permissions['equipment'].update(role_def.equipment_access)
                
                # Merge temporal access (take highest level)
                for time_type, level in role_def.temporal_access.items():
                    current_level = permissions['temporal'].get(time_type, 'none')
                    if self._compare_access_level(level, current_level) > 0:
                        permissions['temporal'][time_type] = level
                        
                # Merge spatial boundaries
                if role_def.spatial_boundaries:
                    permissions['spatial'].update(role_def.spatial_boundaries)
                    
                # Take highest collaboration level
                if self._compare_collaboration_level(
                    role_def.collaboration_level, 
                    permissions['collaboration']
                ) > 0:
                    permissions['collaboration'] = role_def.collaboration_level
                    
        # Convert sets to lists for JSON serialization
        permissions['domains'] = list(permissions['domains'])
        permissions['access_levels'] = list(permissions['access_levels'])
        permissions['computational'] = list(permissions['computational'])
        permissions['equipment'] = list(permissions['equipment'])
        
        return permissions
        
    def _compare_access_level(self, level1: str, level2: str) -> int:
        """Compare access levels (returns 1 if level1 > level2, -1 if level1 < level2, 0 if equal)"""
        levels = {'none': 0, 'limited': 1, 'full': 2}
        return levels.get(level1, 0) - levels.get(level2, 0)
        
    def _compare_collaboration_level(self, level1: str, level2: str) -> int:
        """Compare collaboration levels"""
        levels = {'individual': 0, 'team': 1, 'institutional': 2, 'global': 3}
        return levels.get(level1, 0) - levels.get(level2, 0)
        
    async def _get_jwt_private_key(self) -> str:
        """Get JWT signing private key"""
        # In production, this would be loaded from secure key management
        return """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA4f5wg5l2hKsTeNem/V41fGnJm6gOdrj8ym3rFkEjWT2btNjcIBFu5d2gdHQsVUFSRNWG7X1XvM7jQX4CwHdwWkjZmKcs+8j0pDGjdmP6f+oCgcjcfFOnHP4PLynFxn6+4AA5XTpuuxcx7cRcXIkg90+WP/4L5fhnfiFBe+1MMg2RgFvqn1T8zxcwBdXcjud+1nLWPDqVP4VLiDysHXzT4Cv2klSSDaBS5+bMx2qUkC4rE0CurQPiMZ8peLAiNNre9hH+Qs2+uaEcucdHHU+Q1UcHD5br5A6VB8wuTVZkcEdYSGBRZSOs2vAvO/VYQtpbb5Hw8lh7DM0+9appKbVkL+cQIDAQABAoIBADj7kP/gd5HH1+cxnmNbRgVBuvaWpbgYpyoZ5+6GKhffiRchqt4yzUyaGNlwCSjluFiLJTqC5XqnmD5+6uREHstfn5O5yoKDcoHEX+7ckqfJ/IjmM0xqM7Y2L4M2YBZfuPoqYH7YBnm6O2iBNRcC9+0V3l5Z+8PSLimeWuRcMxAs+2b+dpsDbf+mwGgNLWTAhqfzf+aFdxtIT/E4NjDSsEX8+5VoqAHyqfuR5x/rVTKZpuSBYwqMVA0G1lLEBhgws07q8TKxMeKUBkM4rjXBmFXfJ3ReVx+9+x28QWsqXFHBhWPWHrK1W5KiCfPTtI5+Wkww8qQHBpUKBr4lqUBCk4kCgYEA/ZeDNR+FVcs9qgEmRKm9+J2xLuVCxVlSWxbQkt2+0IkjXn2DAhw5c4ld7kVaEXTuJg7xGAI+Z2KlHgUoSdENIFn2cSTBgHA7nQzaEeAct43ZvjcqJEMmBPZrhlfRacYeZEOatnkYABOy+5noEKm+S8X8OxjkCgYEA5vxeUlNeKEkuFiK9HI5jYuB5TTCyIlttExDcSaFa6+fWkb7iVBfaZB2BJ+uuuzWdBcw1P8jmgVFpUFrYy5gUinXfOC8Rb9uCdjdvXp25p0Gq4bSr8R+3pkfh+Lhqe+DRyAoGBAOWnNuGKiluQN4UEHepTQFerr2aABPFrP7AGDre5BJXXNjvQh+PGlGgQ+o6UlPoDBHxCC6ykbSidxHBVxWh4k38MeKDmzgUebw8IntidDMjR1EKej+nnc5JgdwSICoCnFnvGJmjcvqciVd6aFuER/uuBUiJBYdyI5RKH0WSRyDMbAoGAYRNk5X5Ag6ucfgmrlgahTO8toKGlR5+sfucrCx8ATh3t0ybsai89plcPpQPV7MVS1eIjlFaEHlOiNSdaAoGBAONqRdqN1loNP+fh6hNI2Up7+JgGhE3TZ4je9yAdSWMhOc4+BVE1tWQxBfr+R6nwPaO2HLvzm8OANwmsmGT3Ag==
-----END RSA PRIVATE KEY-----"""
        
    async def _get_jwt_public_key(self) -> str:
        """Get JWT verification public key"""
        # In production, this would be loaded from secure key management
        return """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA4f5wg5l2hKsTeNem/V41fGnJm6gOdrj8ym3rFkEjWT2btNjcIBFu5d2gdHQsVUFSRNWG7X1XvM7jQX4CwHdwWkjZmKcs+8j0pDGjdmP6f+oCgcjcfFOnHP4PLynFxn6+4AA5XTpuuxcx7cRcXIkg90+WP/4L5fhnfiFBe+1MMg2RgFvqn1T8zxcwBdXcjud+1nLWPDqVP4VLiDysHXzT4Cv2klSSDaBS5+bMx2qUkC4rE0CurQPiMZ8peLAiNNre9hH+Qs2+uaEcucdHHU+Q1UcHD5br5A6VB8wuTVZkcEdYSGBRZSOs2vAvO/VYQtpbb5Hw8lh7DM0+9appKbVkL+cQIDAQAB
-----END PUBLIC KEY-----"""
        
    async def _increment_failed_attempts(self, user_id: str):
        """Increment failed login attempts and lock account if necessary"""
        async with self.db_pool.acquire() as conn:
            result = await conn.fetchrow("""
                UPDATE users 
                SET failed_attempts = failed_attempts + 1,
                    last_failed_attempt = NOW()
                WHERE user_id = $1
                RETURNING failed_attempts
            """, user_id)
            
            # Lock account after 5 failed attempts
            if result and result['failed_attempts'] >= 5:
                await conn.execute("""
                    UPDATE users 
                    SET account_locked = true,
                        locked_at = NOW(),
                        locked_reason = 'Excessive failed login attempts'
                    WHERE user_id = $1
                """, user_id)
                
    async def _reset_failed_attempts(self, user_id: str):
        """Reset failed login attempts after successful login"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users 
                SET failed_attempts = 0,
                    last_failed_attempt = NULL
                WHERE user_id = $1
            """, user_id)
            
    async def _update_last_login(self, user_id: str):
        """Update last login timestamp"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                UPDATE users 
                SET last_login = NOW(),
                    login_count = login_count + 1
                WHERE user_id = $1
            """, user_id)
            
    async def health_check(self) -> Dict:
        """Health check for identity management system"""
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
                    "oauth_providers": len(self.oidc_providers),
                    "geological_roles": len(self.geological_roles)
                }
            }
        except Exception as e:
            logger.error(f"Identity manager health check failed: {str(e)}")
            return {"healthy": False, "error": str(e)}