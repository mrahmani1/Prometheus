"""
Advanced Identity and Access Management with OAuth2/OIDC and Geological Role-Based Permissions
Implements sophisticated IAM with geological domain expertise
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import jwt
import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secrets
import base64

from models.security_models import (
    UserContext, GeologicalRole, GeologicalPermission, 
    OAuthProvider, OIDCClaims, SecurityContext
)
from config.security_config import SecurityConfig

logger = logging.getLogger(__name__)

class IdentityAccessManager:
    """Advanced IAM with geological domain expertise and OAuth2/OIDC integration"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.oauth_providers = {}
        self.geological_roles = {}
        self.permission_cache = {}
        self.token_blacklist = set()
        self.active_sessions = {}
        
    async def initialize(self):
        """Initialize IAM system with OAuth providers and geological roles"""
        try:
            # Initialize OAuth2/OIDC providers
            await self._initialize_oauth_providers()
            
            # Load geological role definitions
            await self._load_geological_roles()
            
            # Initialize permission cache
            await self._initialize_permission_cache()
            
            # Start token cleanup task
            asyncio.create_task(self._token_cleanup_task())
            
            logger.info("Identity Access Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize IAM: {e}")
            raise
    
    async def _initialize_oauth_providers(self):
        """Initialize OAuth2/OIDC providers for geological research institutions"""
        providers = {
            'usgs': OAuthProvider(
                name='USGS',
                client_id=self.config.oauth_usgs_client_id,
                client_secret=self.config.oauth_usgs_client_secret,
                authorization_url='https://auth.usgs.gov/oauth2/authorize',
                token_url='https://auth.usgs.gov/oauth2/token',
                userinfo_url='https://auth.usgs.gov/oauth2/userinfo',
                scopes=['geological_data', 'seismic_access', 'volcanic_monitoring']
            ),
            'iris': OAuthProvider(
                name='IRIS',
                client_id=self.config.oauth_iris_client_id,
                client_secret=self.config.oauth_iris_client_secret,
                authorization_url='https://service.iris.edu/oauth2/authorize',
                token_url='https://service.iris.edu/oauth2/token',
                userinfo_url='https://service.iris.edu/oauth2/userinfo',
                scopes=['seismic_data', 'waveform_access', 'metadata_read']
            ),
            'earthscope': OAuthProvider(
                name='EarthScope',
                client_id=self.config.oauth_earthscope_client_id,
                client_secret=self.config.oauth_earthscope_client_secret,
                authorization_url='https://auth.earthscope.org/oauth2/authorize',
                token_url='https://auth.earthscope.org/oauth2/token',
                userinfo_url='https://auth.earthscope.org/oauth2/userinfo',
                scopes=['facility_access', 'instrument_data', 'research_collaboration']
            ),
            'nasa': OAuthProvider(
                name='NASA EarthData',
                client_id=self.config.oauth_nasa_client_id,
                client_secret=self.config.oauth_nasa_client_secret,
                authorization_url='https://urs.earthdata.nasa.gov/oauth/authorize',
                token_url='https://urs.earthdata.nasa.gov/oauth/token',
                userinfo_url='https://urs.earthdata.nasa.gov/api/users/user',
                scopes=['satellite_data', 'earth_observation', 'climate_data']
            )
        }
        
        self.oauth_providers = providers
        logger.info(f"Initialized {len(providers)} OAuth providers")
    
    async def _load_geological_roles(self):
        """Load geological role definitions with domain-specific permissions"""
        roles = {
            'seismologist': GeologicalRole(
                name='seismologist',
                description='Seismic data analysis and earthquake research',
                permissions={
                    GeologicalPermission.SEISMIC_DATA_READ,
                    GeologicalPermission.SEISMIC_DATA_ANALYZE,
                    GeologicalPermission.WAVEFORM_ACCESS,
                    GeologicalPermission.EVENT_DETECTION,
                    GeologicalPermission.MAGNITUDE_CALCULATION,
                    GeologicalPermission.FOCAL_MECHANISM_ANALYSIS
                },
                data_sensitivity_levels={'public', 'restricted', 'confidential'},
                temporal_access_patterns=['real_time', 'historical', 'continuous'],
                spatial_access_regions=['global', 'regional', 'local']
            ),
            'volcanologist': GeologicalRole(
                name='volcanologist',
                description='Volcanic monitoring and eruption forecasting',
                permissions={
                    GeologicalPermission.VOLCANIC_DATA_READ,
                    GeologicalPermission.VOLCANIC_DATA_ANALYZE,
                    GeologicalPermission.ERUPTION_FORECASTING,
                    GeologicalPermission.DEFORMATION_ANALYSIS,
                    GeologicalPermission.GAS_EMISSION_MONITORING,
                    GeologicalPermission.THERMAL_ANALYSIS,
                    GeologicalPermission.ALERT_GENERATION
                },
                data_sensitivity_levels={'public', 'restricted', 'confidential', 'classified'},
                temporal_access_patterns=['real_time', 'near_real_time', 'historical'],
                spatial_access_regions=['volcanic_regions', 'caldera_systems', 'global']
            ),
            'geodynamicist': GeologicalRole(
                name='geodynamicist',
                description='Tectonic processes and long-term geological evolution',
                permissions={
                    GeologicalPermission.TECTONIC_DATA_READ,
                    GeologicalPermission.TECTONIC_DATA_ANALYZE,
                    GeologicalPermission.PLATE_MOTION_ANALYSIS,
                    GeologicalPermission.STRESS_FIELD_MODELING,
                    GeologicalPermission.MANTLE_CONVECTION_ACCESS,
                    GeologicalPermission.GEOLOGICAL_TIME_ANALYSIS
                },
                data_sensitivity_levels={'public', 'restricted'},
                temporal_access_patterns=['geological_time', 'long_term', 'historical'],
                spatial_access_regions=['global', 'continental', 'regional']
            ),
            'resource_geologist': GeologicalRole(
                name='resource_geologist',
                description='Mineral and energy resource assessment',
                permissions={
                    GeologicalPermission.RESOURCE_DATA_READ,
                    GeologicalPermission.RESOURCE_DATA_ANALYZE,
                    GeologicalPermission.PROSPECTIVITY_MAPPING,
                    GeologicalPermission.GEOCHEMICAL_ANALYSIS,
                    GeologicalPermission.ECONOMIC_ASSESSMENT,
                    GeologicalPermission.EXPLORATION_PLANNING
                },
                data_sensitivity_levels={'public', 'restricted', 'confidential', 'proprietary'},
                temporal_access_patterns=['current', 'historical', 'predictive'],
                spatial_access_regions=['exploration_areas', 'mining_districts', 'regional']
            ),
            'geophysicist': GeologicalRole(
                name='geophysicist',
                description='Geophysical data analysis and subsurface imaging',
                permissions={
                    GeologicalPermission.GEOPHYSICAL_DATA_READ,
                    GeologicalPermission.GEOPHYSICAL_DATA_ANALYZE,
                    GeologicalPermission.GRAVITY_ANALYSIS,
                    GeologicalPermission.MAGNETIC_ANALYSIS,
                    GeologicalPermission.ELECTROMAGNETIC_ANALYSIS,
                    GeologicalPermission.SUBSURFACE_IMAGING,
                    GeologicalPermission.TOMOGRAPHY_ACCESS
                },
                data_sensitivity_levels={'public', 'restricted', 'confidential'},
                temporal_access_patterns=['real_time', 'historical', 'continuous'],
                spatial_access_regions=['global', 'regional', 'local', 'subsurface']
            ),
            'system_administrator': GeologicalRole(
                name='system_administrator',
                description='System administration with geological context awareness',
                permissions={
                    GeologicalPermission.SYSTEM_ADMIN,
                    GeologicalPermission.USER_MANAGEMENT,
                    GeologicalPermission.SECURITY_MANAGEMENT,
                    GeologicalPermission.DATA_MANAGEMENT,
                    GeologicalPermission.MONITORING_ACCESS,
                    GeologicalPermission.CONFIGURATION_MANAGEMENT
                },
                data_sensitivity_levels={'public', 'restricted', 'confidential', 'classified', 'proprietary'},
                temporal_access_patterns=['all'],
                spatial_access_regions=['all']
            )
        }
        
        self.geological_roles = roles
        logger.info(f"Loaded {len(roles)} geological roles")
    
    async def _initialize_permission_cache(self):
        """Initialize permission cache for fast access control decisions"""
        self.permission_cache = {
            'role_permissions': {},
            'user_permissions': {},
            'context_permissions': {},
            'cache_timestamps': {}
        }
        
        # Pre-populate role permissions
        for role_name, role in self.geological_roles.items():
            self.permission_cache['role_permissions'][role_name] = role.permissions
        
        logger.info("Permission cache initialized")
    
    async def authenticate_oauth(self, provider: str, authorization_code: str, 
                               redirect_uri: str) -> UserContext:
        """Authenticate user via OAuth2/OIDC provider"""
        try:
            if provider not in self.oauth_providers:
                raise ValueError(f"Unknown OAuth provider: {provider}")
            
            oauth_provider = self.oauth_providers[provider]
            
            # Exchange authorization code for access token
            token_data = await self._exchange_code_for_token(
                oauth_provider, authorization_code, redirect_uri
            )
            
            # Get user information from provider
            user_info = await self._get_user_info(oauth_provider, token_data['access_token'])
            
            # Create OIDC claims
            oidc_claims = OIDCClaims(
                sub=user_info.get('sub'),
                email=user_info.get('email'),
                name=user_info.get('name'),
                preferred_username=user_info.get('preferred_username'),
                given_name=user_info.get('given_name'),
                family_name=user_info.get('family_name'),
                institution=user_info.get('institution'),
                department=user_info.get('department'),
                research_interests=user_info.get('research_interests', [])
            )
            
            # Determine geological roles based on provider and user info
            geological_roles = await self._determine_geological_roles(provider, user_info)
            
            # Create user context
            user_context = UserContext(
                user_id=oidc_claims.sub,
                email=oidc_claims.email,
                name=oidc_claims.name,
                provider=provider,
                roles=geological_roles,
                geological_permissions=await self._get_user_permissions(geological_roles),
                oidc_claims=oidc_claims,
                access_token=token_data['access_token'],
                refresh_token=token_data.get('refresh_token'),
                token_expires_at=datetime.utcnow() + timedelta(seconds=token_data['expires_in'])
            )
            
            # Store active session
            session_id = secrets.token_urlsafe(32)
            self.active_sessions[session_id] = user_context
            
            logger.info(f"User authenticated via {provider}: {user_context.email}")
            return user_context
            
        except Exception as e:
            logger.error(f"OAuth authentication failed for {provider}: {e}")
            raise
    
    async def _exchange_code_for_token(self, provider: OAuthProvider, 
                                     authorization_code: str, redirect_uri: str) -> Dict:
        """Exchange authorization code for access token"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider.token_url,
                data={
                    'grant_type': 'authorization_code',
                    'code': authorization_code,
                    'redirect_uri': redirect_uri,
                    'client_id': provider.client_id,
                    'client_secret': provider.client_secret
                },
                headers={'Accept': 'application/json'}
            )
            
            if response.status_code != 200:
                raise Exception(f"Token exchange failed: {response.text}")
            
            return response.json()
    
    async def _get_user_info(self, provider: OAuthProvider, access_token: str) -> Dict:
        """Get user information from OAuth provider"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                provider.userinfo_url,
                headers={'Authorization': f'Bearer {access_token}'}
            )
            
            if response.status_code != 200:
                raise Exception(f"User info request failed: {response.text}")
            
            return response.json()
    
    async def _determine_geological_roles(self, provider: str, user_info: Dict) -> List[str]:
        """Determine geological roles based on provider and user information"""
        roles = []
        
        # Role mapping based on provider and user attributes
        if provider == 'usgs':
            if 'seismology' in user_info.get('department', '').lower():
                roles.append('seismologist')
            if 'volcano' in user_info.get('department', '').lower():
                roles.append('volcanologist')
            if 'geophysics' in user_info.get('department', '').lower():
                roles.append('geophysicist')
        
        elif provider == 'iris':
            roles.append('seismologist')  # IRIS users are typically seismologists
            if 'geophysics' in user_info.get('research_interests', []):
                roles.append('geophysicist')
        
        elif provider == 'earthscope':
            # EarthScope users often have multiple geological interests
            research_interests = user_info.get('research_interests', [])
            if 'seismology' in research_interests:
                roles.append('seismologist')
            if 'volcanology' in research_interests:
                roles.append('volcanologist')
            if 'geodynamics' in research_interests:
                roles.append('geodynamicist')
        
        elif provider == 'nasa':
            roles.append('geophysicist')  # NASA EarthData users typically work with geophysical data
        
        # Default role if no specific roles determined
        if not roles:
            roles.append('geophysicist')
        
        return roles
    
    async def _get_user_permissions(self, roles: List[str]) -> Set[GeologicalPermission]:
        """Get combined permissions for user roles"""
        permissions = set()
        
        for role_name in roles:
            if role_name in self.geological_roles:
                role_permissions = self.geological_roles[role_name].permissions
                permissions.update(role_permissions)
        
        return permissions
    
    async def validate_token(self, token: str) -> UserContext:
        """Validate JWT token and return user context"""
        try:
            # Check if token is blacklisted
            if token in self.token_blacklist:
                raise Exception("Token is blacklisted")
            
            # Decode and validate JWT token
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Extract user context from token
            user_id = payload.get('sub')
            if not user_id:
                raise Exception("Invalid token: missing subject")
            
            # Find active session
            user_context = None
            for session_id, context in self.active_sessions.items():
                if context.user_id == user_id:
                    user_context = context
                    break
            
            if not user_context:
                raise Exception("No active session found")
            
            # Check token expiration
            if user_context.token_expires_at < datetime.utcnow():
                # Try to refresh token
                user_context = await self._refresh_token(user_context)
            
            return user_context
            
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.InvalidTokenError as e:
            raise Exception(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise
    
    async def _refresh_token(self, user_context: UserContext) -> UserContext:
        """Refresh expired access token"""
        if not user_context.refresh_token:
            raise Exception("No refresh token available")
        
        try:
            provider = self.oauth_providers[user_context.provider]
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    provider.token_url,
                    data={
                        'grant_type': 'refresh_token',
                        'refresh_token': user_context.refresh_token,
                        'client_id': provider.client_id,
                        'client_secret': provider.client_secret
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"Token refresh failed: {response.text}")
                
                token_data = response.json()
                
                # Update user context with new tokens
                user_context.access_token = token_data['access_token']
                user_context.refresh_token = token_data.get('refresh_token', user_context.refresh_token)
                user_context.token_expires_at = datetime.utcnow() + timedelta(
                    seconds=token_data['expires_in']
                )
                
                logger.info(f"Token refreshed for user: {user_context.email}")
                return user_context
                
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise
    
    async def revoke_token(self, token: str):
        """Revoke JWT token by adding to blacklist"""
        self.token_blacklist.add(token)
        logger.info("Token revoked and added to blacklist")
    
    async def _token_cleanup_task(self):
        """Periodic cleanup of expired tokens and sessions"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired sessions
                expired_sessions = []
                for session_id, context in self.active_sessions.items():
                    if context.token_expires_at < current_time:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                # Sleep for 5 minutes before next cleanup
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Token cleanup task error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    async def health_check(self) -> bool:
        """Health check for IAM system"""
        try:
            # Check OAuth provider connectivity
            for provider_name, provider in self.oauth_providers.items():
                async with httpx.AsyncClient(timeout=5.0) as client:
                    try:
                        response = await client.get(provider.authorization_url.split('?')[0])
                        if response.status_code >= 500:
                            logger.warning(f"OAuth provider {provider_name} health check failed")
                            return False
                    except Exception:
                        logger.warning(f"OAuth provider {provider_name} unreachable")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"IAM health check failed: {e}")
            return False
    
    async def is_ready(self) -> bool:
        """Check if IAM system is ready"""
        return (
            len(self.oauth_providers) > 0 and
            len(self.geological_roles) > 0 and
            self.permission_cache is not None
        )
    
    async def cleanup(self):
        """Cleanup IAM resources"""
        self.active_sessions.clear()
        self.token_blacklist.clear()
        self.permission_cache.clear()
        logger.info("IAM cleanup completed")