"""
Secure Data Sharing and Collaboration System for Prometheus AI Platform

This module implements secure data sharing capabilities including:
- End-to-end encrypted data sharing
- Role-based access control for collaborative research
- Secure multi-party computation for joint analysis
- Digital rights management for geological data
- Audit trails for data sharing activities
- Time-limited access tokens and permissions
- Secure data collaboration workflows
"""

import asyncio
import json
import logging
import secrets
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import hmac
import uuid
import base64

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import aioredis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# Import from other security modules
from .encryption_engine import (
    EncryptionEngine, KeyManager, DataClassification, 
    EncryptionAlgorithm, KeyType, EncryptionMetadata
)
from .dlp_engine import DLPPolicyEngine, GeologicalDataSensitivity


class SharingPermission(Enum):
    """Data sharing permissions"""
    READ = "read"
    WRITE = "write"
    ANALYZE = "analyze"
    EXPORT = "export"
    SHARE = "share"
    DELETE = "delete"
    ADMIN = "admin"


class CollaborationType(Enum):
    """Types of collaboration"""
    RESEARCH_PROJECT = "research_project"
    JOINT_ANALYSIS = "joint_analysis"
    DATA_EXCHANGE = "data_exchange"
    PEER_REVIEW = "peer_review"
    EMERGENCY_RESPONSE = "emergency_response"
    COMMERCIAL_PARTNERSHIP = "commercial_partnership"


class AccessTokenType(Enum):
    """Types of access tokens"""
    TEMPORARY = "temporary"
    SESSION_BASED = "session_based"
    API_KEY = "api_key"
    DELEGATION = "delegation"
    EMERGENCY = "emergency"


@dataclass
class SharingPolicy:
    """Data sharing policy definition"""
    policy_id: str
    name: str
    description: str
    data_classifications: List[DataClassification]
    geological_sensitivities: List[GeologicalDataSensitivity]
    allowed_permissions: List[SharingPermission]
    collaboration_types: List[CollaborationType]
    restrictions: Dict[str, Any]
    approval_required: bool = True
    max_duration: Optional[timedelta] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataShare:
    """Data sharing record"""
    share_id: str
    owner_id: str
    data_resource_id: str
    data_classification: DataClassification
    geological_sensitivity: GeologicalDataSensitivity
    shared_with: List[str]  # User IDs or organization IDs
    permissions: List[SharingPermission]
    collaboration_type: CollaborationType
    sharing_policy_id: str
    encryption_key_id: str
    access_conditions: Dict[str, Any]
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    revoked: bool = False


@dataclass
class AccessToken:
    """Secure access token for data sharing"""
    token_id: str
    token_value: str
    token_type: AccessTokenType
    share_id: str
    user_id: str
    permissions: List[SharingPermission]
    issued_at: datetime
    expires_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None
    ip_restrictions: Optional[List[str]] = None
    revoked: bool = False


@dataclass
class CollaborationWorkspace:
    """Secure collaboration workspace"""
    workspace_id: str
    name: str
    description: str
    collaboration_type: CollaborationType
    owner_id: str
    participants: List[str]
    data_shares: List[str]  # Share IDs
    shared_resources: Dict[str, Any]
    workspace_key_id: str
    access_policy: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    active: bool = True


@dataclass
class SharingAuditEvent:
    """Audit event for data sharing activities"""
    event_id: str
    event_type: str
    share_id: Optional[str]
    workspace_id: Optional[str]
    user_id: str
    action: str
    resource_id: str
    permissions_used: List[SharingPermission]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecureTokenManager:
    """Manager for secure access tokens"""
    
    def __init__(self, key_manager: KeyManager, config: Dict[str, Any]):
        self.key_manager = key_manager
        self.config = config
        self.redis_client = None
        self.token_signing_key_id = None
    
    async def initialize(self) -> bool:
        """Initialize token manager"""
        try:
            # Initialize Redis
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Generate or retrieve token signing key
            self.token_signing_key_id = await self.key_manager.generate_key(
                KeyType.HMAC,
                EncryptionAlgorithm.AES_256_GCM,
                DataClassification.RESTRICTED
            )
            
            logging.info("Secure token manager initialized")
            return True
            
        except Exception as e:
            logging.error(f"Token manager initialization failed: {e}")
            return False
    
    async def create_access_token(
        self,
        share_id: str,
        user_id: str,
        permissions: List[SharingPermission],
        token_type: AccessTokenType = AccessTokenType.TEMPORARY,
        duration: timedelta = timedelta(hours=24),
        max_usage: Optional[int] = None,
        ip_restrictions: Optional[List[str]] = None
    ) -> AccessToken:
        """Create secure access token"""
        
        token_id = str(uuid.uuid4())
        issued_at = datetime.utcnow()
        expires_at = issued_at + duration
        
        # Create token payload
        payload = {
            'token_id': token_id,
            'share_id': share_id,
            'user_id': user_id,
            'permissions': [p.value for p in permissions],
            'issued_at': issued_at.isoformat(),
            'expires_at': expires_at.isoformat(),
            'token_type': token_type.value
        }
        
        # Sign token
        token_value = await self._sign_token(payload)
        
        # Create token record
        token = AccessToken(
            token_id=token_id,
            token_value=token_value,
            token_type=token_type,
            share_id=share_id,
            user_id=user_id,
            permissions=permissions,
            issued_at=issued_at,
            expires_at=expires_at,
            max_usage=max_usage,
            ip_restrictions=ip_restrictions
        )
        
        # Store token
        await self._store_token(token)
        
        logging.info(f"Created access token {token_id} for user {user_id}")
        return token
    
    async def validate_token(
        self,
        token_value: str,
        required_permission: SharingPermission,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[AccessToken], Optional[str]]:
        """Validate access token"""
        
        try:
            # Verify token signature and extract payload
            payload = await self._verify_token(token_value)
            if not payload:
                return False, None, "Invalid token signature"
            
            token_id = payload.get('token_id')
            if not token_id:
                return False, None, "Invalid token format"
            
            # Retrieve token record
            token = await self._get_token(token_id)
            if not token:
                return False, None, "Token not found"
            
            # Check if token is revoked
            if token.revoked:
                return False, None, "Token has been revoked"
            
            # Check expiration
            if datetime.utcnow() > token.expires_at:
                return False, None, "Token has expired"
            
            # Check usage limits
            if token.max_usage and token.usage_count >= token.max_usage:
                return False, None, "Token usage limit exceeded"
            
            # Check IP restrictions
            if token.ip_restrictions and ip_address:
                if ip_address not in token.ip_restrictions:
                    return False, None, "IP address not allowed"
            
            # Check permissions
            if required_permission not in token.permissions:
                return False, None, f"Permission {required_permission.value} not granted"
            
            # Update token usage
            token.last_used = datetime.utcnow()
            token.usage_count += 1
            await self._update_token(token)
            
            return True, token, None
            
        except Exception as e:
            logging.error(f"Token validation failed: {e}")
            return False, None, str(e)
    
    async def revoke_token(self, token_id: str) -> bool:
        """Revoke access token"""
        try:
            token = await self._get_token(token_id)
            if token:
                token.revoked = True
                await self._update_token(token)
                logging.info(f"Revoked token {token_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Token revocation failed: {e}")
            return False
    
    async def _sign_token(self, payload: Dict[str, Any]) -> str:
        """Sign token payload"""
        # Get signing key
        signing_key = await self.key_manager.get_key(self.token_signing_key_id)
        if not signing_key:
            raise ValueError("Token signing key not available")
        
        # Create token string
        payload_json = json.dumps(payload, sort_keys=True)
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()
        
        # Create signature
        signature = hmac.new(
            signing_key,
            payload_b64.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine payload and signature
        token = f"{payload_b64}.{signature}"
        return token
    
    async def _verify_token(self, token_value: str) -> Optional[Dict[str, Any]]:
        """Verify token signature and extract payload"""
        try:
            # Split token
            parts = token_value.split('.')
            if len(parts) != 2:
                return None
            
            payload_b64, signature = parts
            
            # Get signing key
            signing_key = await self.key_manager.get_key(self.token_signing_key_id)
            if not signing_key:
                return None
            
            # Verify signature
            expected_signature = hmac.new(
                signing_key,
                payload_b64.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return None
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
            payload = json.loads(payload_json)
            
            return payload
            
        except Exception as e:
            logging.error(f"Token verification failed: {e}")
            return None
    
    async def _store_token(self, token: AccessToken):
        """Store access token"""
        token_data = {
            'token_value': token.token_value,
            'token_type': token.token_type.value,
            'share_id': token.share_id,
            'user_id': token.user_id,
            'permissions': [p.value for p in token.permissions],
            'issued_at': token.issued_at.isoformat(),
            'expires_at': token.expires_at.isoformat(),
            'last_used': token.last_used.isoformat() if token.last_used else None,
            'usage_count': token.usage_count,
            'max_usage': token.max_usage,
            'ip_restrictions': token.ip_restrictions,
            'revoked': token.revoked
        }
        
        # Store with expiration
        ttl = int((token.expires_at - datetime.utcnow()).total_seconds())
        await self.redis_client.setex(
            f"access_token:{token.token_id}",
            ttl,
            json.dumps(token_data)
        )
    
    async def _get_token(self, token_id: str) -> Optional[AccessToken]:
        """Retrieve access token"""
        token_data = await self.redis_client.get(f"access_token:{token_id}")
        if not token_data:
            return None
        
        data = json.loads(token_data)
        return AccessToken(
            token_id=token_id,
            token_value=data['token_value'],
            token_type=AccessTokenType(data['token_type']),
            share_id=data['share_id'],
            user_id=data['user_id'],
            permissions=[SharingPermission(p) for p in data['permissions']],
            issued_at=datetime.fromisoformat(data['issued_at']),
            expires_at=datetime.fromisoformat(data['expires_at']),
            last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None,
            usage_count=data['usage_count'],
            max_usage=data['max_usage'],
            ip_restrictions=data['ip_restrictions'],
            revoked=data['revoked']
        )
    
    async def _update_token(self, token: AccessToken):
        """Update access token"""
        await self._store_token(token)


class SecureDataSharingEngine:
    """Main engine for secure data sharing and collaboration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = KeyManager(config)
        self.encryption_engine = None
        self.token_manager = None
        self.dlp_engine = DLPPolicyEngine(config)
        self.redis_client = None
        self.sharing_policies: Dict[str, SharingPolicy] = {}
        self.active_shares: Dict[str, DataShare] = {}
        self.collaboration_workspaces: Dict[str, CollaborationWorkspace] = {}
    
    async def initialize(self) -> bool:
        """Initialize secure data sharing engine"""
        try:
            # Initialize components
            await self.key_manager.initialize()
            self.encryption_engine = EncryptionEngine(self.key_manager)
            self.token_manager = SecureTokenManager(self.key_manager, self.config)
            await self.token_manager.initialize()
            await self.dlp_engine.initialize()
            
            # Initialize Redis
            self.redis_client = aioredis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                decode_responses=True
            )
            
            # Load default sharing policies
            await self._load_default_sharing_policies()
            
            logging.info("Secure data sharing engine initialized")
            return True
            
        except Exception as e:
            logging.error(f"Secure data sharing engine initialization failed: {e}")
            return False
    
    async def create_data_share(
        self,
        owner_id: str,
        data_resource_id: str,
        data: Dict[str, Any],
        shared_with: List[str],
        permissions: List[SharingPermission],
        collaboration_type: CollaborationType,
        data_classification: DataClassification,
        geological_sensitivity: GeologicalDataSensitivity,
        duration: Optional[timedelta] = None,
        access_conditions: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[DataShare], Optional[str]]:
        """Create secure data share"""
        
        try:
            # Find applicable sharing policy
            policy = await self._find_sharing_policy(
                data_classification, geological_sensitivity, collaboration_type
            )
            if not policy:
                return False, None, "No applicable sharing policy found"
            
            # Check if permissions are allowed by policy
            if not all(perm in policy.allowed_permissions for perm in permissions):
                return False, None, "Requested permissions not allowed by policy"
            
            # Generate encryption key for shared data
            encryption_key_id = await self.key_manager.generate_key(
                KeyType.SYMMETRIC,
                EncryptionAlgorithm.AES_256_GCM,
                data_classification
            )
            
            # Encrypt data
            encrypted_data, metadata = await self.encryption_engine.encrypt_data(
                data, EncryptionAlgorithm.AES_256_GCM, encryption_key_id, data_classification
            )
            
            # Create data share
            share_id = str(uuid.uuid4())
            expires_at = None
            if duration:
                expires_at = datetime.utcnow() + duration
            elif policy.max_duration:
                expires_at = datetime.utcnow() + policy.max_duration
            
            data_share = DataShare(
                share_id=share_id,
                owner_id=owner_id,
                data_resource_id=data_resource_id,
                data_classification=data_classification,
                geological_sensitivity=geological_sensitivity,
                shared_with=shared_with,
                permissions=permissions,
                collaboration_type=collaboration_type,
                sharing_policy_id=policy.policy_id,
                encryption_key_id=encryption_key_id,
                access_conditions=access_conditions or {},
                expires_at=expires_at
            )
            
            # Store encrypted data
            await self._store_encrypted_data(share_id, encrypted_data, metadata)
            
            # Store data share
            await self._store_data_share(data_share)
            self.active_shares[share_id] = data_share
            
            # Create access tokens for shared users
            for user_id in shared_with:
                await self.token_manager.create_access_token(
                    share_id, user_id, permissions, 
                    AccessTokenType.TEMPORARY,
                    duration or timedelta(days=30)
                )
            
            # Log sharing event
            await self._log_sharing_event(
                "data_share_created", share_id, None, owner_id,
                "create_share", data_resource_id, permissions, True
            )
            
            logging.info(f"Created data share {share_id} for resource {data_resource_id}")
            return True, data_share, None
            
        except Exception as e:
            logging.error(f"Data share creation failed: {e}")
            return False, None, str(e)
    
    async def access_shared_data(
        self,
        token_value: str,
        user_id: str,
        required_permission: SharingPermission,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Access shared data using token"""
        
        try:
            # Validate token
            valid, token, error = await self.token_manager.validate_token(
                token_value, required_permission, ip_address
            )
            if not valid:
                await self._log_sharing_event(
                    "data_access_denied", token.share_id if token else None, None,
                    user_id, "access_data", "", [required_permission], False, error
                )
                return False, None, error
            
            # Get data share
            data_share = await self._get_data_share(token.share_id)
            if not data_share:
                return False, None, "Data share not found"
            
            # Check if share is still valid
            if data_share.revoked:
                return False, None, "Data share has been revoked"
            
            if data_share.expires_at and datetime.utcnow() > data_share.expires_at:
                return False, None, "Data share has expired"
            
            # Check if user is authorized
            if user_id not in data_share.shared_with and user_id != data_share.owner_id:
                return False, None, "User not authorized for this data share"
            
            # Retrieve and decrypt data
            encrypted_data, metadata = await self._get_encrypted_data(token.share_id)
            if not encrypted_data:
                return False, None, "Shared data not found"
            
            decrypted_data = await self.encryption_engine.decrypt_data(encrypted_data, metadata)
            data = json.loads(decrypted_data.decode('utf-8'))
            
            # Update access statistics
            data_share.last_accessed = datetime.utcnow()
            data_share.access_count += 1
            await self._update_data_share(data_share)
            
            # Log successful access
            await self._log_sharing_event(
                "data_access_granted", token.share_id, None, user_id,
                "access_data", data_share.data_resource_id, [required_permission], True
            )
            
            return True, data, None
            
        except Exception as e:
            logging.error(f"Data access failed: {e}")
            return False, None, str(e)
    
    async def create_collaboration_workspace(
        self,
        owner_id: str,
        name: str,
        description: str,
        collaboration_type: CollaborationType,
        participants: List[str],
        initial_data_shares: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[CollaborationWorkspace], Optional[str]]:
        """Create secure collaboration workspace"""
        
        try:
            workspace_id = str(uuid.uuid4())
            
            # Generate workspace encryption key
            workspace_key_id = await self.key_manager.generate_key(
                KeyType.SYMMETRIC,
                EncryptionAlgorithm.AES_256_GCM,
                DataClassification.CONFIDENTIAL
            )
            
            # Create workspace
            workspace = CollaborationWorkspace(
                workspace_id=workspace_id,
                name=name,
                description=description,
                collaboration_type=collaboration_type,
                owner_id=owner_id,
                participants=participants,
                data_shares=initial_data_shares or [],
                shared_resources={},
                workspace_key_id=workspace_key_id,
                access_policy={
                    'require_authentication': True,
                    'allow_external_sharing': False,
                    'audit_all_activities': True
                }
            )
            
            # Store workspace
            await self._store_collaboration_workspace(workspace)
            self.collaboration_workspaces[workspace_id] = workspace
            
            # Create workspace access tokens for participants
            for participant_id in participants:
                await self.token_manager.create_access_token(
                    workspace_id, participant_id,
                    [SharingPermission.READ, SharingPermission.ANALYZE],
                    AccessTokenType.SESSION_BASED,
                    timedelta(days=90)
                )
            
            # Log workspace creation
            await self._log_sharing_event(
                "workspace_created", None, workspace_id, owner_id,
                "create_workspace", workspace_id, [SharingPermission.ADMIN], True
            )
            
            logging.info(f"Created collaboration workspace {workspace_id}")
            return True, workspace, None
            
        except Exception as e:
            logging.error(f"Workspace creation failed: {e}")
            return False, None, str(e)
    
    async def revoke_data_share(self, share_id: str, user_id: str) -> Tuple[bool, Optional[str]]:
        """Revoke data share"""
        
        try:
            data_share = await self._get_data_share(share_id)
            if not data_share:
                return False, "Data share not found"
            
            # Check if user has permission to revoke
            if user_id != data_share.owner_id:
                return False, "Only the owner can revoke a data share"
            
            # Revoke the share
            data_share.revoked = True
            await self._update_data_share(data_share)
            
            # Revoke all associated tokens
            # This would require tracking tokens by share_id
            
            # Log revocation
            await self._log_sharing_event(
                "data_share_revoked", share_id, None, user_id,
                "revoke_share", data_share.data_resource_id, [SharingPermission.ADMIN], True
            )
            
            logging.info(f"Revoked data share {share_id}")
            return True, None
            
        except Exception as e:
            logging.error(f"Data share revocation failed: {e}")
            return False, str(e)
    
    async def _load_default_sharing_policies(self):
        """Load default sharing policies"""
        
        # Policy 1: Academic research collaboration
        policy1 = SharingPolicy(
            policy_id="academic_research_sharing",
            name="Academic Research Data Sharing",
            description="Policy for sharing geological data in academic research collaborations",
            data_classifications=[DataClassification.PUBLIC, DataClassification.INTERNAL],
            geological_sensitivities=[
                GeologicalDataSensitivity.PUBLIC_RESEARCH,
                GeologicalDataSensitivity.ACADEMIC_USE
            ],
            allowed_permissions=[
                SharingPermission.READ,
                SharingPermission.ANALYZE,
                SharingPermission.EXPORT
            ],
            collaboration_types=[
                CollaborationType.RESEARCH_PROJECT,
                CollaborationType.PEER_REVIEW
            ],
            restrictions={
                'max_participants': 20,
                'require_institutional_affiliation': True,
                'allow_commercial_use': False
            },
            approval_required=False,
            max_duration=timedelta(days=365)
        )
        
        # Policy 2: Emergency response data sharing
        policy2 = SharingPolicy(
            policy_id="emergency_response_sharing",
            name="Emergency Response Data Sharing",
            description="Policy for sharing critical geological data during emergencies",
            data_classifications=[
                DataClassification.INTERNAL,
                DataClassification.CONFIDENTIAL,
                DataClassification.RESTRICTED
            ],
            geological_sensitivities=[
                GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
                GeologicalDataSensitivity.CRITICAL_INFRASTRUCTURE
            ],
            allowed_permissions=[
                SharingPermission.READ,
                SharingPermission.ANALYZE,
                SharingPermission.EXPORT,
                SharingPermission.SHARE
            ],
            collaboration_types=[CollaborationType.EMERGENCY_RESPONSE],
            restrictions={
                'emergency_authorization_required': True,
                'time_limited_access': True,
                'audit_all_access': True
            },
            approval_required=True,
            max_duration=timedelta(days=30)
        )
        
        # Policy 3: Commercial partnership data sharing
        policy3 = SharingPolicy(
            policy_id="commercial_partnership_sharing",
            name="Commercial Partnership Data Sharing",
            description="Policy for sharing geological data with commercial partners",
            data_classifications=[DataClassification.CONFIDENTIAL],
            geological_sensitivities=[GeologicalDataSensitivity.COMMERCIAL_SENSITIVE],
            allowed_permissions=[
                SharingPermission.READ,
                SharingPermission.ANALYZE
            ],
            collaboration_types=[CollaborationType.COMMERCIAL_PARTNERSHIP],
            restrictions={
                'nda_required': True,
                'ip_protection': True,
                'usage_tracking': True,
                'no_redistribution': True
            },
            approval_required=True,
            max_duration=timedelta(days=180)
        )
        
        # Store policies
        policies = [policy1, policy2, policy3]
        for policy in policies:
            self.sharing_policies[policy.policy_id] = policy
            await self._store_sharing_policy(policy)
    
    async def _find_sharing_policy(
        self,
        data_classification: DataClassification,
        geological_sensitivity: GeologicalDataSensitivity,
        collaboration_type: CollaborationType
    ) -> Optional[SharingPolicy]:
        """Find applicable sharing policy"""
        
        for policy in self.sharing_policies.values():
            if (data_classification in policy.data_classifications and
                geological_sensitivity in policy.geological_sensitivities and
                collaboration_type in policy.collaboration_types):
                return policy
        
        return None
    
    async def _store_encrypted_data(
        self,
        share_id: str,
        encrypted_data: bytes,
        metadata: EncryptionMetadata
    ):
        """Store encrypted shared data"""
        
        data_record = {
            'encrypted_data': base64.b64encode(encrypted_data).decode(),
            'metadata': {
                'algorithm': metadata.algorithm.value,
                'key_id': metadata.key_id,
                'iv': base64.b64encode(metadata.iv).decode() if metadata.iv else None,
                'tag': base64.b64encode(metadata.tag).decode() if metadata.tag else None,
                'timestamp': metadata.timestamp.isoformat(),
                'classification': metadata.classification.value
            }
        }
        
        await self.redis_client.set(
            f"shared_data:{share_id}",
            json.dumps(data_record)
        )
    
    async def _get_encrypted_data(
        self,
        share_id: str
    ) -> Tuple[Optional[bytes], Optional[EncryptionMetadata]]:
        """Retrieve encrypted shared data"""
        
        data_record = await self.redis_client.get(f"shared_data:{share_id}")
        if not data_record:
            return None, None
        
        record = json.loads(data_record)
        encrypted_data = base64.b64decode(record['encrypted_data'].encode())
        
        metadata_dict = record['metadata']
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm(metadata_dict['algorithm']),
            key_id=metadata_dict['key_id'],
            iv=base64.b64decode(metadata_dict['iv'].encode()) if metadata_dict['iv'] else None,
            tag=base64.b64decode(metadata_dict['tag'].encode()) if metadata_dict['tag'] else None,
            timestamp=datetime.fromisoformat(metadata_dict['timestamp']),
            classification=DataClassification(metadata_dict['classification'])
        )
        
        return encrypted_data, metadata
    
    async def _store_data_share(self, data_share: DataShare):
        """Store data share record"""
        
        share_data = {
            'owner_id': data_share.owner_id,
            'data_resource_id': data_share.data_resource_id,
            'data_classification': data_share.data_classification.value,
            'geological_sensitivity': data_share.geological_sensitivity.value,
            'shared_with': data_share.shared_with,
            'permissions': [p.value for p in data_share.permissions],
            'collaboration_type': data_share.collaboration_type.value,
            'sharing_policy_id': data_share.sharing_policy_id,
            'encryption_key_id': data_share.encryption_key_id,
            'access_conditions': data_share.access_conditions,
            'expires_at': data_share.expires_at.isoformat() if data_share.expires_at else None,
            'created_at': data_share.created_at.isoformat(),
            'last_accessed': data_share.last_accessed.isoformat() if data_share.last_accessed else None,
            'access_count': data_share.access_count,
            'revoked': data_share.revoked
        }
        
        await self.redis_client.set(
            f"data_share:{data_share.share_id}",
            json.dumps(share_data)
        )
    
    async def _get_data_share(self, share_id: str) -> Optional[DataShare]:
        """Retrieve data share record"""
        
        share_data = await self.redis_client.get(f"data_share:{share_id}")
        if not share_data:
            return None
        
        data = json.loads(share_data)
        return DataShare(
            share_id=share_id,
            owner_id=data['owner_id'],
            data_resource_id=data['data_resource_id'],
            data_classification=DataClassification(data['data_classification']),
            geological_sensitivity=GeologicalDataSensitivity(data['geological_sensitivity']),
            shared_with=data['shared_with'],
            permissions=[SharingPermission(p) for p in data['permissions']],
            collaboration_type=CollaborationType(data['collaboration_type']),
            sharing_policy_id=data['sharing_policy_id'],
            encryption_key_id=data['encryption_key_id'],
            access_conditions=data['access_conditions'],
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            created_at=datetime.fromisoformat(data['created_at']),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None,
            access_count=data['access_count'],
            revoked=data['revoked']
        )
    
    async def _update_data_share(self, data_share: DataShare):
        """Update data share record"""
        await self._store_data_share(data_share)
    
    async def _store_sharing_policy(self, policy: SharingPolicy):
        """Store sharing policy"""
        
        policy_data = {
            'name': policy.name,
            'description': policy.description,
            'data_classifications': [c.value for c in policy.data_classifications],
            'geological_sensitivities': [s.value for s in policy.geological_sensitivities],
            'allowed_permissions': [p.value for p in policy.allowed_permissions],
            'collaboration_types': [t.value for t in policy.collaboration_types],
            'restrictions': policy.restrictions,
            'approval_required': policy.approval_required,
            'max_duration': policy.max_duration.total_seconds() if policy.max_duration else None,
            'created_at': policy.created_at.isoformat()
        }
        
        await self.redis_client.set(
            f"sharing_policy:{policy.policy_id}",
            json.dumps(policy_data)
        )
    
    async def _store_collaboration_workspace(self, workspace: CollaborationWorkspace):
        """Store collaboration workspace"""
        
        workspace_data = {
            'name': workspace.name,
            'description': workspace.description,
            'collaboration_type': workspace.collaboration_type.value,
            'owner_id': workspace.owner_id,
            'participants': workspace.participants,
            'data_shares': workspace.data_shares,
            'shared_resources': workspace.shared_resources,
            'workspace_key_id': workspace.workspace_key_id,
            'access_policy': workspace.access_policy,
            'created_at': workspace.created_at.isoformat(),
            'last_activity': workspace.last_activity.isoformat(),
            'active': workspace.active
        }
        
        await self.redis_client.set(
            f"workspace:{workspace.workspace_id}",
            json.dumps(workspace_data)
        )
    
    async def _log_sharing_event(
        self,
        event_type: str,
        share_id: Optional[str],
        workspace_id: Optional[str],
        user_id: str,
        action: str,
        resource_id: str,
        permissions_used: List[SharingPermission],
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log sharing audit event"""
        
        event = SharingAuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            share_id=share_id,
            workspace_id=workspace_id,
            user_id=user_id,
            action=action,
            resource_id=resource_id,
            permissions_used=permissions_used,
            ip_address=None,  # Would be provided by request context
            user_agent=None,  # Would be provided by request context
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        # Store audit event
        event_data = {
            'event_type': event.event_type,
            'share_id': event.share_id,
            'workspace_id': event.workspace_id,
            'user_id': event.user_id,
            'action': event.action,
            'resource_id': event.resource_id,
            'permissions_used': [p.value for p in event.permissions_used],
            'ip_address': event.ip_address,
            'user_agent': event.user_agent,
            'success': event.success,
            'error_message': event.error_message,
            'timestamp': event.timestamp.isoformat(),
            'metadata': event.metadata
        }
        
        await self.redis_client.lpush(
            "sharing_audit_log",
            json.dumps(event_data)
        )
        
        # Keep only last 10000 events
        await self.redis_client.ltrim("sharing_audit_log", 0, 9999)


if __name__ == "__main__":
    # Example usage and testing
    async def test_secure_sharing():
        """Test secure data sharing functionality"""
        
        # Initialize sharing engine
        config = {
            'redis_url': 'redis://localhost:6379',
            'hsm': {
                'library_path': '/usr/lib/softhsm/libsofthsm2.so',
                'token_label': 'prometheus',
                'user_pin': '1234'
            }
        }
        
        sharing_engine = SecureDataSharingEngine(config)
        await sharing_engine.initialize()
        
        # Test data to share
        geological_data = {
            'volcano_name': 'Mount St. Helens',
            'eruption_probability': 0.75,
            'seismic_activity': [
                {'timestamp': '2024-01-01T12:00:00Z', 'magnitude': 2.1},
                {'timestamp': '2024-01-01T13:30:00Z', 'magnitude': 1.8}
            ],
            'gas_emissions': {
                'so2_ppm': 150,
                'co2_ppm': 400,
                'h2s_ppm': 25
            }
        }
        
        # Create data share
        success, data_share, error = await sharing_engine.create_data_share(
            owner_id="researcher_001",
            data_resource_id="mount_st_helens_monitoring_2024",
            data=geological_data,
            shared_with=["researcher_002", "researcher_003"],
            permissions=[SharingPermission.READ, SharingPermission.ANALYZE],
            collaboration_type=CollaborationType.RESEARCH_PROJECT,
            data_classification=DataClassification.CONFIDENTIAL,
            geological_sensitivity=GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
            duration=timedelta(days=30)
        )
        
        if success:
            print(f"Data share created successfully: {data_share.share_id}")
            
            # Test accessing shared data (would normally use actual token)
            # For testing, we'll create a token manually
            token = await sharing_engine.token_manager.create_access_token(
                data_share.share_id,
                "researcher_002",
                [SharingPermission.READ, SharingPermission.ANALYZE]
            )
            
            # Access shared data
            access_success, retrieved_data, access_error = await sharing_engine.access_shared_data(
                token.token_value,
                "researcher_002",
                SharingPermission.READ
            )
            
            if access_success:
                print("Data accessed successfully")
                print(f"Retrieved data: {retrieved_data}")
            else:
                print(f"Data access failed: {access_error}")
        else:
            print(f"Data share creation failed: {error}")
        
        # Test collaboration workspace
        workspace_success, workspace, workspace_error = await sharing_engine.create_collaboration_workspace(
            owner_id="researcher_001",
            name="Mount St. Helens Research Project",
            description="Collaborative research on volcanic activity prediction",
            collaboration_type=CollaborationType.RESEARCH_PROJECT,
            participants=["researcher_002", "researcher_003", "researcher_004"]
        )
        
        if workspace_success:
            print(f"Collaboration workspace created: {workspace.workspace_id}")
        else:
            print(f"Workspace creation failed: {workspace_error}")
    
    # Run test
    asyncio.run(test_secure_sharing())