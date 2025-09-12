"""
Main Security Framework Service for Prometheus AI Platform

This service integrates all advanced data protection and encryption capabilities:
- Advanced encryption engine with HSM integration
- Data Loss Prevention (DLP) controls
- Secure data sharing and collaboration
- Encrypted backup and archival systems
- Secure multi-party computation
- Comprehensive security monitoring and compliance
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add the core modules to the path
sys.path.append(str(Path(__file__).parent / "core"))

# Import security framework components
from core.encryption_engine import (
    EncryptionEngine, KeyManager, DataClassifier, HomomorphicEncryption,
    DataClassification, EncryptionAlgorithm, GeologicalDataField
)
from core.dlp_engine import (
    DLPPolicyEngine, DataMaskingEngine, GeologicalSensitivityAnalyzer,
    DLPAction, DLPViolationType, GeologicalDataSensitivity,
    DataAccessRequest, GeologicalDataContext
)
from core.secure_sharing import (
    SecureDataSharingEngine, SecureTokenManager,
    SharingPermission, CollaborationType, AccessTokenType
)
from core.encrypted_backup import (
    EncryptedBackupEngine, CloudStorageManager,
    BackupTier, BackupType, StorageProvider
)
from core.secure_multiparty import (
    SecureMultiPartyEngine, MPCProtocol, ComputationType,
    MPCParticipant, ParticipantRole, DifferentialPrivacyParams
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Prometheus Security Framework",
    description="Advanced data protection and encryption service for geological data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global components
key_manager: Optional[KeyManager] = None
encryption_engine: Optional[EncryptionEngine] = None
dlp_engine: Optional[DLPPolicyEngine] = None
sharing_engine: Optional[SecureDataSharingEngine] = None
backup_engine: Optional[EncryptedBackupEngine] = None
mpc_engine: Optional[SecureMultiPartyEngine] = None
homomorphic_engine: Optional[HomomorphicEncryption] = None


# Pydantic models for API
class EncryptionRequest(BaseModel):
    data: Dict[str, Any]
    algorithm: str = "aes_256_gcm"
    classification: str = "internal"
    key_id: Optional[str] = None


class DecryptionRequest(BaseModel):
    encrypted_data: str  # Base64 encoded
    metadata: Dict[str, Any]


class DataClassificationRequest(BaseModel):
    data: Dict[str, Any]
    data_type: str


class DataAccessRequestModel(BaseModel):
    user_id: str
    user_roles: List[str]
    resource_id: str
    data_type: str
    operation: str
    destination: Optional[str] = None
    purpose: Optional[str] = None


class GeologicalDataContextModel(BaseModel):
    data_source: str
    geological_domain: str
    temporal_scope: str
    spatial_scope: str
    research_purpose: str
    commercial_value: str = "low"
    public_safety_impact: str = "low"
    national_security_relevance: str = "low"


class DataSharingRequest(BaseModel):
    owner_id: str
    data_resource_id: str
    data: Dict[str, Any]
    shared_with: List[str]
    permissions: List[str]
    collaboration_type: str
    data_classification: str
    geological_sensitivity: str
    duration_hours: Optional[int] = None


class BackupRequest(BaseModel):
    source_path: str
    data_classification: str
    geological_data_type: str
    backup_type: str = "full"
    policy_id: Optional[str] = None


class RestoreRequest(BaseModel):
    backup_id: str
    restore_path: str
    user_id: str
    point_in_time: Optional[str] = None


class MPCSessionRequest(BaseModel):
    name: str
    description: str
    protocol: str
    computation_type: str
    coordinator_id: str
    participant_ids: List[str]
    data_requirements: Dict[str, Any]
    privacy_parameters: Optional[Dict[str, Any]] = None
    security_parameters: Optional[Dict[str, Any]] = None


class MPCComputationRequest(BaseModel):
    session_id: str
    participant_data: Dict[str, Any]


class MPCParticipantModel(BaseModel):
    participant_id: str
    organization: str
    role: str
    public_key: str
    endpoint: str
    capabilities: List[str]
    trust_level: float
    data_classification_clearance: List[str]
    geological_domains: List[str]


# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Validate JWT token and return user info"""
    # This would integrate with your authentication system
    # For now, return a mock user
    return {"user_id": "test_user", "roles": ["researcher"]}


@app.on_event("startup")
async def startup_event():
    """Initialize security framework components"""
    global key_manager, encryption_engine, dlp_engine, sharing_engine
    global backup_engine, mpc_engine, homomorphic_engine
    
    try:
        # Load configuration
        config = {
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
            'hsm': {
                'library_path': os.getenv('HSM_LIBRARY_PATH', '/usr/lib/softhsm/libsofthsm2.so'),
                'token_label': os.getenv('HSM_TOKEN_LABEL', 'prometheus'),
                'user_pin': os.getenv('HSM_USER_PIN', '1234')
            },
            'aws': {
                'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'region': os.getenv('AWS_REGION', 'us-west-2'),
                'backup_bucket': os.getenv('AWS_BACKUP_BUCKET', 'prometheus-backups')
            },
            'azure': {
                'account_url': os.getenv('AZURE_STORAGE_ACCOUNT_URL'),
                'account_key': os.getenv('AZURE_STORAGE_ACCOUNT_KEY'),
                'backup_container': os.getenv('AZURE_BACKUP_CONTAINER', 'prometheus-backups')
            },
            'gcp': {
                'project_id': os.getenv('GCP_PROJECT_ID'),
                'credentials_path': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
                'backup_bucket': os.getenv('GCP_BACKUP_BUCKET', 'prometheus-backups')
            },
            'local_backup_dir': os.getenv('LOCAL_BACKUP_DIR', '/var/backups/prometheus')
        }
        
        # Initialize key manager
        key_manager = KeyManager(config)
        await key_manager.initialize()
        logger.info("Key manager initialized")
        
        # Initialize encryption engine
        encryption_engine = EncryptionEngine(key_manager)
        logger.info("Encryption engine initialized")
        
        # Initialize homomorphic encryption
        homomorphic_engine = HomomorphicEncryption()
        await homomorphic_engine.initialize()
        logger.info("Homomorphic encryption initialized")
        
        # Initialize DLP engine
        dlp_engine = DLPPolicyEngine(config)
        await dlp_engine.initialize()
        logger.info("DLP engine initialized")
        
        # Initialize secure sharing engine
        sharing_engine = SecureDataSharingEngine(config)
        await sharing_engine.initialize()
        logger.info("Secure sharing engine initialized")
        
        # Initialize backup engine
        backup_engine = EncryptedBackupEngine(config)
        await backup_engine.initialize()
        logger.info("Backup engine initialized")
        
        # Initialize MPC engine
        mpc_engine = SecureMultiPartyEngine(config)
        await mpc_engine.initialize()
        logger.info("MPC engine initialized")
        
        logger.info("Security framework startup completed successfully")
        
    except Exception as e:
        logger.error(f"Security framework startup failed: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "key_manager": key_manager is not None,
            "encryption_engine": encryption_engine is not None,
            "dlp_engine": dlp_engine is not None,
            "sharing_engine": sharing_engine is not None,
            "backup_engine": backup_engine is not None,
            "mpc_engine": mpc_engine is not None,
            "homomorphic_engine": homomorphic_engine is not None
        }
    }


# Encryption endpoints
@app.post("/api/v1/encrypt")
async def encrypt_data(
    request: EncryptionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Encrypt geological data"""
    try:
        algorithm = EncryptionAlgorithm(request.algorithm)
        classification = DataClassification(request.classification)
        
        encrypted_data, metadata = await encryption_engine.encrypt_data(
            request.data,
            algorithm,
            request.key_id,
            classification
        )
        
        return {
            "success": True,
            "encrypted_data": encrypted_data.hex(),
            "metadata": {
                "algorithm": metadata.algorithm.value,
                "key_id": metadata.key_id,
                "iv": metadata.iv.hex() if metadata.iv else None,
                "tag": metadata.tag.hex() if metadata.tag else None,
                "timestamp": metadata.timestamp.isoformat(),
                "classification": metadata.classification.value
            }
        }
        
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/decrypt")
async def decrypt_data(
    request: DecryptionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Decrypt geological data"""
    try:
        # Reconstruct metadata
        from core.encryption_engine import EncryptionMetadata
        
        metadata = EncryptionMetadata(
            algorithm=EncryptionAlgorithm(request.metadata["algorithm"]),
            key_id=request.metadata["key_id"],
            iv=bytes.fromhex(request.metadata["iv"]) if request.metadata.get("iv") else None,
            tag=bytes.fromhex(request.metadata["tag"]) if request.metadata.get("tag") else None,
            timestamp=datetime.fromisoformat(request.metadata["timestamp"]),
            classification=DataClassification(request.metadata["classification"])
        )
        
        encrypted_data = bytes.fromhex(request.encrypted_data)
        decrypted_data = await encryption_engine.decrypt_data(encrypted_data, metadata)
        
        # Try to parse as JSON
        try:
            result = json.loads(decrypted_data.decode('utf-8'))
        except:
            result = decrypted_data.decode('utf-8')
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data classification endpoints
@app.post("/api/v1/classify")
async def classify_data(
    request: DataClassificationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Classify geological data"""
    try:
        classifier = DataClassifier({})
        classified_fields = await classifier.classify_data(request.data, request.data_type)
        
        result = {}
        for field_name, field_info in classified_fields.items():
            result[field_name] = {
                "classification": field_info.classification.value,
                "encryption_required": field_info.encryption_required,
                "pii_detected": field_info.pii_detected,
                "geological_sensitivity": field_info.geological_sensitivity
            }
        
        return {
            "success": True,
            "classified_fields": result
        }
        
    except Exception as e:
        logger.error(f"Data classification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DLP endpoints
@app.post("/api/v1/dlp/evaluate")
async def evaluate_data_access(
    access_request: DataAccessRequestModel,
    context: GeologicalDataContextModel,
    data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """Evaluate data access request against DLP policies"""
    try:
        # Convert to internal models
        request = DataAccessRequest(
            request_id=str(datetime.utcnow().timestamp()),
            user_id=access_request.user_id,
            user_roles=access_request.user_roles,
            resource_id=access_request.resource_id,
            data_type=access_request.data_type,
            operation=access_request.operation,
            destination=access_request.destination,
            purpose=access_request.purpose
        )
        
        geological_context = GeologicalDataContext(
            data_source=context.data_source,
            geological_domain=context.geological_domain,
            temporal_scope=context.temporal_scope,
            spatial_scope=context.spatial_scope,
            research_purpose=context.research_purpose,
            commercial_value=context.commercial_value,
            public_safety_impact=context.public_safety_impact,
            national_security_relevance=context.national_security_relevance
        )
        
        # Evaluate access request
        allow_access, actions, violation = await dlp_engine.evaluate_access_request(
            request, data, geological_context
        )
        
        result = {
            "success": True,
            "allow_access": allow_access,
            "actions": [action.value for action in actions]
        }
        
        if violation:
            result["violation"] = {
                "violation_id": violation.violation_id,
                "violation_type": violation.violation_type.value,
                "severity": violation.severity,
                "details": violation.violation_details
            }
        
        return result
        
    except Exception as e:
        logger.error(f"DLP evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Secure sharing endpoints
@app.post("/api/v1/sharing/create")
async def create_data_share(
    request: DataSharingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create secure data share"""
    try:
        # Convert string enums to proper types
        permissions = [SharingPermission(p) for p in request.permissions]
        collaboration_type = CollaborationType(request.collaboration_type)
        data_classification = DataClassification(request.data_classification)
        geological_sensitivity = GeologicalDataSensitivity(request.geological_sensitivity)
        
        duration = None
        if request.duration_hours:
            duration = timedelta(hours=request.duration_hours)
        
        success, data_share, error = await sharing_engine.create_data_share(
            owner_id=request.owner_id,
            data_resource_id=request.data_resource_id,
            data=request.data,
            shared_with=request.shared_with,
            permissions=permissions,
            collaboration_type=collaboration_type,
            data_classification=data_classification,
            geological_sensitivity=geological_sensitivity,
            duration=duration
        )
        
        if success:
            return {
                "success": True,
                "share_id": data_share.share_id,
                "expires_at": data_share.expires_at.isoformat() if data_share.expires_at else None
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"Data sharing creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sharing/access")
async def access_shared_data(
    token: str,
    permission: str,
    current_user: dict = Depends(get_current_user)
):
    """Access shared data using token"""
    try:
        required_permission = SharingPermission(permission)
        
        success, data, error = await sharing_engine.access_shared_data(
            token_value=token,
            user_id=current_user["user_id"],
            required_permission=required_permission
        )
        
        if success:
            return {
                "success": True,
                "data": data
            }
        else:
            raise HTTPException(status_code=403, detail=error)
            
    except Exception as e:
        logger.error(f"Data access failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backup endpoints
@app.post("/api/v1/backup/create")
async def create_backup(
    request: BackupRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create encrypted backup"""
    try:
        data_classification = DataClassification(request.data_classification)
        backup_type = BackupType(request.backup_type)
        
        success, job_id, error = await backup_engine.create_backup(
            source_path=request.source_path,
            data_classification=data_classification,
            geological_data_type=request.geological_data_type,
            backup_type=backup_type,
            policy_id=request.policy_id
        )
        
        if success:
            return {
                "success": True,
                "job_id": job_id
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/backup/restore")
async def restore_backup(
    request: RestoreRequest,
    current_user: dict = Depends(get_current_user)
):
    """Restore encrypted backup"""
    try:
        point_in_time = None
        if request.point_in_time:
            point_in_time = datetime.fromisoformat(request.point_in_time)
        
        success, request_id, error = await backup_engine.restore_backup(
            backup_id=request.backup_id,
            restore_path=request.restore_path,
            user_id=request.user_id,
            point_in_time=point_in_time
        )
        
        if success:
            return {
                "success": True,
                "request_id": request_id
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"Backup restore failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# MPC endpoints
@app.post("/api/v1/mpc/register")
async def register_mpc_participant(
    participant: MPCParticipantModel,
    current_user: dict = Depends(get_current_user)
):
    """Register MPC participant"""
    try:
        # Convert to internal model
        mpc_participant = MPCParticipant(
            participant_id=participant.participant_id,
            organization=participant.organization,
            role=ParticipantRole(participant.role),
            public_key=participant.public_key,
            endpoint=participant.endpoint,
            capabilities=[MPCProtocol(c) for c in participant.capabilities],
            trust_level=participant.trust_level,
            data_classification_clearance=[DataClassification(c) for c in participant.data_classification_clearance],
            geological_domains=participant.geological_domains
        )
        
        success, error = await mpc_engine.register_participant(mpc_participant)
        
        if success:
            return {
                "success": True,
                "participant_id": participant.participant_id
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"MPC participant registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/mpc/session/create")
async def create_mpc_session(
    request: MPCSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create MPC session"""
    try:
        protocol = MPCProtocol(request.protocol)
        computation_type = ComputationType(request.computation_type)
        
        success, session_id, error = await mpc_engine.create_mpc_session(
            name=request.name,
            description=request.description,
            protocol=protocol,
            computation_type=computation_type,
            coordinator_id=request.coordinator_id,
            participant_ids=request.participant_ids,
            data_requirements=request.data_requirements,
            privacy_parameters=request.privacy_parameters,
            security_parameters=request.security_parameters
        )
        
        if success:
            return {
                "success": True,
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"MPC session creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/mpc/compute")
async def execute_mpc_computation(
    request: MPCComputationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Execute MPC computation"""
    try:
        success, result, error = await mpc_engine.execute_secure_computation(
            session_id=request.session_id,
            participant_data=request.participant_data
        )
        
        if success:
            return {
                "success": True,
                "result": result
            }
        else:
            raise HTTPException(status_code=400, detail=error)
            
    except Exception as e:
        logger.error(f"MPC computation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Homomorphic encryption endpoints
@app.post("/api/v1/homomorphic/encrypt")
async def encrypt_homomorphic(
    data: List[float],
    current_user: dict = Depends(get_current_user)
):
    """Encrypt data for homomorphic operations"""
    try:
        encrypted_data = await homomorphic_engine.encrypt_vector(data)
        
        if encrypted_data:
            return {
                "success": True,
                "encrypted_data": encrypted_data.hex()
            }
        else:
            raise HTTPException(status_code=500, detail="Homomorphic encryption failed")
            
    except Exception as e:
        logger.error(f"Homomorphic encryption failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/homomorphic/add")
async def add_homomorphic(
    encrypted_a: str,
    encrypted_b: str,
    current_user: dict = Depends(get_current_user)
):
    """Add two homomorphically encrypted vectors"""
    try:
        data_a = bytes.fromhex(encrypted_a)
        data_b = bytes.fromhex(encrypted_b)
        
        result = await homomorphic_engine.add_encrypted_vectors(data_a, data_b)
        
        if result:
            return {
                "success": True,
                "result": result.hex()
            }
        else:
            raise HTTPException(status_code=500, detail="Homomorphic addition failed")
            
    except Exception as e:
        logger.error(f"Homomorphic addition failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )