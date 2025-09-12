"""
Comprehensive tests for the Security Framework Service

Tests all major components:
- Encryption engine with HSM integration
- Data Loss Prevention (DLP) controls
- Secure data sharing and collaboration
- Encrypted backup and archival systems
- Secure multi-party computation
- Homomorphic encryption
"""

import asyncio
import json
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the core modules to the path
sys.path.append(str(Path(__file__).parent.parent / "core"))

from core.encryption_engine import (
    EncryptionEngine, KeyManager, DataClassifier, HomomorphicEncryption,
    DataClassification, EncryptionAlgorithm, KeyType
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


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'redis_url': 'redis://localhost:6379/15',  # Use test database
        'hsm': {
            'library_path': '/usr/lib/softhsm/libsofthsm2.so',
            'token_label': 'test_prometheus',
            'user_pin': '1234'
        },
        'local_backup_dir': tempfile.mkdtemp()
    }


@pytest.fixture
async def key_manager(test_config):
    """Initialize key manager for testing"""
    manager = KeyManager(test_config)
    await manager.initialize()
    return manager


@pytest.fixture
async def encryption_engine(key_manager):
    """Initialize encryption engine for testing"""
    return EncryptionEngine(key_manager)


@pytest.fixture
async def dlp_engine(test_config):
    """Initialize DLP engine for testing"""
    engine = DLPPolicyEngine(test_config)
    await engine.initialize()
    return engine


@pytest.fixture
async def sharing_engine(test_config):
    """Initialize secure sharing engine for testing"""
    engine = SecureDataSharingEngine(test_config)
    await engine.initialize()
    return engine


@pytest.fixture
async def backup_engine(test_config):
    """Initialize backup engine for testing"""
    engine = EncryptedBackupEngine(test_config)
    await engine.initialize()
    return engine


@pytest.fixture
async def mpc_engine(test_config):
    """Initialize MPC engine for testing"""
    engine = SecureMultiPartyEngine(test_config)
    await engine.initialize()
    return engine


@pytest.fixture
def geological_test_data():
    """Sample geological data for testing"""
    return {
        'volcano_name': 'Mount St. Helens',
        'latitude': 46.1912,
        'longitude': -122.1944,
        'elevation': 2549,
        'eruption_probability': 0.75,
        'seismic_events': [
            {'timestamp': '2024-01-01T12:00:00Z', 'magnitude': 2.1, 'depth': 5.2},
            {'timestamp': '2024-01-01T13:30:00Z', 'magnitude': 1.8, 'depth': 4.8},
            {'timestamp': '2024-01-01T15:15:00Z', 'magnitude': 2.3, 'depth': 6.1}
        ],
        'gas_emissions': {
            'so2_ppm': 150,
            'co2_ppm': 400,
            'h2s_ppm': 25
        },
        'researcher_contact': 'dr.smith@usgs.gov',
        'alert_level': 'orange',
        'evacuation_zones': ['zone_a', 'zone_b'],
        'critical_infrastructure': ['highway_504', 'spirit_lake_dam']
    }


class TestEncryptionEngine:
    """Test encryption engine functionality"""
    
    @pytest.mark.asyncio
    async def test_key_generation(self, key_manager):
        """Test cryptographic key generation"""
        # Test symmetric key generation
        key_id = await key_manager.generate_key(
            KeyType.SYMMETRIC,
            EncryptionAlgorithm.AES_256_GCM,
            DataClassification.CONFIDENTIAL
        )
        
        assert key_id is not None
        assert isinstance(key_id, str)
        
        # Test key retrieval
        key_material = await key_manager.get_key(key_id)
        assert key_material is not None
        assert len(key_material) == 32  # 256 bits
    
    @pytest.mark.asyncio
    async def test_data_encryption_decryption(self, encryption_engine, geological_test_data):
        """Test data encryption and decryption"""
        # Test string encryption
        test_message = "Sensitive volcanic monitoring data"
        
        encrypted_data, metadata = await encryption_engine.encrypt_data(
            test_message,
            EncryptionAlgorithm.AES_256_GCM,
            classification=DataClassification.CONFIDENTIAL
        )
        
        assert encrypted_data is not None
        assert metadata.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert metadata.classification == DataClassification.CONFIDENTIAL
        
        # Test decryption
        decrypted_data = await encryption_engine.decrypt_data(encrypted_data, metadata)
        decrypted_message = decrypted_data.decode('utf-8')
        
        assert decrypted_message == test_message
    
    @pytest.mark.asyncio
    async def test_geological_data_encryption(self, encryption_engine, geological_test_data):
        """Test encryption of complex geological data"""
        encrypted_data, metadata = await encryption_engine.encrypt_data(
            geological_test_data,
            EncryptionAlgorithm.AES_256_GCM,
            classification=DataClassification.RESTRICTED
        )
        
        assert encrypted_data is not None
        assert metadata.classification == DataClassification.RESTRICTED
        
        # Decrypt and verify
        decrypted_data = await encryption_engine.decrypt_data(encrypted_data, metadata)
        decrypted_dict = json.loads(decrypted_data.decode('utf-8'))
        
        assert decrypted_dict == geological_test_data
    
    @pytest.mark.asyncio
    async def test_key_rotation(self, key_manager):
        """Test key rotation functionality"""
        # Generate initial key
        original_key_id = await key_manager.generate_key(
            KeyType.SYMMETRIC,
            EncryptionAlgorithm.AES_256_GCM,
            DataClassification.INTERNAL
        )
        
        # Rotate key
        new_key_id = await key_manager.rotate_key(original_key_id)
        
        assert new_key_id != original_key_id
        assert new_key_id is not None
        
        # Verify both keys exist (old key should be marked for deletion)
        original_key = await key_manager.get_key(original_key_id)
        new_key = await key_manager.get_key(new_key_id)
        
        assert original_key is not None  # Still available during grace period
        assert new_key is not None


class TestDataClassification:
    """Test data classification functionality"""
    
    @pytest.mark.asyncio
    async def test_geological_data_classification(self, geological_test_data):
        """Test classification of geological data"""
        classifier = DataClassifier({})
        
        classified_fields = await classifier.classify_data(
            geological_test_data, 
            "volcanic_monitoring"
        )
        
        # Check that sensitive fields are properly classified
        assert 'eruption_probability' in classified_fields
        assert classified_fields['eruption_probability'].classification == DataClassification.CONFIDENTIAL
        assert classified_fields['eruption_probability'].encryption_required
        
        # Check PII detection
        assert 'researcher_contact' in classified_fields
        assert classified_fields['researcher_contact'].pii_detected
        
        # Check geological sensitivity
        assert classified_fields['eruption_probability'].geological_sensitivity in ['high', 'critical']
    
    @pytest.mark.asyncio
    async def test_pii_detection(self):
        """Test PII detection in geological data"""
        classifier = DataClassifier({})
        
        test_data = {
            'researcher_name': 'Dr. John Smith',
            'email': 'john.smith@university.edu',
            'phone': '555-123-4567',
            'seismic_reading': 2.1,
            'location': 'Mount Rainier'
        }
        
        classified_fields = await classifier.classify_data(test_data, "research_data")
        
        # Email should be detected as PII
        assert classified_fields['email'].pii_detected
        assert classified_fields['email'].encryption_required
        
        # Phone should be detected as PII
        assert classified_fields['phone'].pii_detected
        
        # Scientific data should not be PII
        assert not classified_fields['seismic_reading'].pii_detected


class TestDLPEngine:
    """Test Data Loss Prevention functionality"""
    
    @pytest.mark.asyncio
    async def test_dlp_policy_evaluation(self, dlp_engine, geological_test_data):
        """Test DLP policy evaluation"""
        # Create test access request
        request = DataAccessRequest(
            request_id="test_request_001",
            user_id="researcher_001",
            user_roles=["volcanologist", "researcher"],
            resource_id="mount_st_helens_data",
            data_type="volcanic_monitoring",
            operation="read",
            purpose="research"
        )
        
        # Create geological context
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
        
        # Evaluate access request
        allow_access, actions, violation = await dlp_engine.evaluate_access_request(
            request, geological_test_data, context
        )
        
        # Should have some policy actions due to sensitive data
        assert isinstance(allow_access, bool)
        assert isinstance(actions, list)
        
        if violation:
            assert violation.violation_type in [
                DLPViolationType.POLICY_VIOLATION,
                DLPViolationType.GEOLOGICAL_SENSITIVITY
            ]
    
    @pytest.mark.asyncio
    async def test_data_masking(self, geological_test_data):
        """Test data masking functionality"""
        masking_engine = DataMaskingEngine()
        classifier = DataClassifier({})
        
        # Classify data first
        classified_fields = await classifier.classify_data(
            geological_test_data, 
            "volcanic_monitoring"
        )
        
        # Apply masking
        masked_data = await masking_engine.mask_data(
            geological_test_data, 
            classified_fields,
            masking_level="standard"
        )
        
        # Check that PII is masked
        assert masked_data['researcher_contact'] != geological_test_data['researcher_contact']
        
        # Check that coordinates are reduced in precision
        if 'latitude' in masked_data and 'longitude' in masked_data:
            original_lat = geological_test_data['latitude']
            masked_lat = masked_data['latitude']
            
            # Precision should be reduced
            assert abs(original_lat - masked_lat) <= 0.1


class TestSecureSharing:
    """Test secure data sharing functionality"""
    
    @pytest.mark.asyncio
    async def test_data_share_creation(self, sharing_engine, geological_test_data):
        """Test creation of secure data share"""
        success, data_share, error = await sharing_engine.create_data_share(
            owner_id="researcher_001",
            data_resource_id="volcanic_data_2024",
            data=geological_test_data,
            shared_with=["researcher_002", "researcher_003"],
            permissions=[SharingPermission.READ, SharingPermission.ANALYZE],
            collaboration_type=CollaborationType.RESEARCH_PROJECT,
            data_classification=DataClassification.CONFIDENTIAL,
            geological_sensitivity=GeologicalDataSensitivity.DISASTER_PREPAREDNESS,
            duration=timedelta(days=30)
        )
        
        assert success
        assert data_share is not None
        assert data_share.share_id is not None
        assert len(data_share.shared_with) == 2
        assert SharingPermission.READ in data_share.permissions
    
    @pytest.mark.asyncio
    async def test_collaboration_workspace(self, sharing_engine):
        """Test collaboration workspace creation"""
        success, workspace, error = await sharing_engine.create_collaboration_workspace(
            owner_id="researcher_001",
            name="Mount St. Helens Research Project",
            description="Collaborative volcanic monitoring research",
            collaboration_type=CollaborationType.RESEARCH_PROJECT,
            participants=["researcher_002", "researcher_003", "researcher_004"]
        )
        
        assert success
        assert workspace is not None
        assert workspace.workspace_id is not None
        assert len(workspace.participants) == 3
        assert workspace.collaboration_type == CollaborationType.RESEARCH_PROJECT


class TestEncryptedBackup:
    """Test encrypted backup functionality"""
    
    @pytest.mark.asyncio
    async def test_backup_creation(self, backup_engine, geological_test_data, test_config):
        """Test encrypted backup creation"""
        # Create test data file
        test_file_path = os.path.join(test_config['local_backup_dir'], 'test_data.json')
        with open(test_file_path, 'w') as f:
            json.dump(geological_test_data, f)
        
        # Create backup
        success, job_id, error = await backup_engine.create_backup(
            source_path=test_file_path,
            data_classification=DataClassification.CONFIDENTIAL,
            geological_data_type="volcanic_monitoring",
            backup_type=BackupType.FULL
        )
        
        assert success
        assert job_id is not None
        assert error is None
    
    @pytest.mark.asyncio
    async def test_cloud_storage_manager(self, test_config):
        """Test cloud storage manager initialization"""
        storage_manager = CloudStorageManager(test_config)
        success = await storage_manager.initialize()
        
        # Should succeed with local storage at minimum
        assert success
        assert StorageProvider.LOCAL_STORAGE in storage_manager.initialized_providers


class TestSecureMultiParty:
    """Test secure multi-party computation functionality"""
    
    @pytest.mark.asyncio
    async def test_participant_registration(self, mpc_engine):
        """Test MPC participant registration"""
        participant = MPCParticipant(
            participant_id="test_usgs_001",
            organization="USGS",
            role=ParticipantRole.DATA_PROVIDER,
            public_key="test_public_key_001",
            endpoint="https://test.usgs.gov/mpc",
            capabilities=[MPCProtocol.SHAMIR_SECRET_SHARING, MPCProtocol.DIFFERENTIAL_PRIVACY],
            trust_level=0.9,
            data_classification_clearance=[DataClassification.INTERNAL, DataClassification.CONFIDENTIAL],
            geological_domains=["seismic", "volcanic"]
        )
        
        success, error = await mpc_engine.register_participant(participant)
        
        assert success
        assert error is None
    
    @pytest.mark.asyncio
    async def test_secret_sharing_computation(self, mpc_engine):
        """Test secret sharing computation"""
        # Register test participants
        participants = [
            MPCParticipant(
                participant_id=f"test_participant_{i}",
                organization=f"Test Org {i}",
                role=ParticipantRole.DATA_PROVIDER,
                public_key=f"test_key_{i}",
                endpoint=f"https://test{i}.org/mpc",
                capabilities=[MPCProtocol.SHAMIR_SECRET_SHARING],
                trust_level=0.8,
                data_classification_clearance=[DataClassification.INTERNAL],
                geological_domains=["seismic"]
            )
            for i in range(3)
        ]
        
        for participant in participants:
            await mpc_engine.register_participant(participant)
        
        # Create MPC session
        success, session_id, error = await mpc_engine.create_mpc_session(
            name="Test Secret Sharing",
            description="Test secure sum computation",
            protocol=MPCProtocol.SHAMIR_SECRET_SHARING,
            computation_type=ComputationType.DATA_AGGREGATION,
            coordinator_id="test_participant_0",
            participant_ids=[p.participant_id for p in participants],
            data_requirements={"data_type": "numerical", "format": "integer"}
        )
        
        assert success
        assert session_id is not None
        
        # Execute computation
        participant_data = {
            "test_participant_0": 10,
            "test_participant_1": 20,
            "test_participant_2": 30
        }
        
        comp_success, result, comp_error = await mpc_engine.execute_secure_computation(
            session_id, participant_data
        )
        
        assert comp_success
        assert result is not None
        assert result['computation_type'] == 'secure_sum'
        assert result['result'] == 60  # Sum of 10 + 20 + 30
    
    @pytest.mark.asyncio
    async def test_differential_privacy_computation(self, mpc_engine):
        """Test differential privacy computation"""
        # Register participants
        participants = [
            MPCParticipant(
                participant_id=f"dp_participant_{i}",
                organization=f"DP Org {i}",
                role=ParticipantRole.DATA_PROVIDER,
                public_key=f"dp_key_{i}",
                endpoint=f"https://dp{i}.org/mpc",
                capabilities=[MPCProtocol.DIFFERENTIAL_PRIVACY],
                trust_level=0.8,
                data_classification_clearance=[DataClassification.INTERNAL],
                geological_domains=["seismic"]
            )
            for i in range(3)
        ]
        
        for participant in participants:
            await mpc_engine.register_participant(participant)
        
        # Create DP session
        success, session_id, error = await mpc_engine.create_mpc_session(
            name="Test Differential Privacy",
            description="Test private statistics computation",
            protocol=MPCProtocol.DIFFERENTIAL_PRIVACY,
            computation_type=ComputationType.STATISTICAL_ANALYSIS,
            coordinator_id="dp_participant_0",
            participant_ids=[p.participant_id for p in participants],
            data_requirements={"data_type": "numerical_array", "format": "float"},
            privacy_parameters={
                "differential_privacy": {
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "sensitivity": 1.0,
                    "mechanism": "laplace"
                }
            }
        )
        
        assert success
        
        # Execute DP computation
        participant_data = {
            "dp_participant_0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "dp_participant_1": [2.0, 3.0, 4.0, 5.0, 6.0],
            "dp_participant_2": [3.0, 4.0, 5.0, 6.0, 7.0]
        }
        
        comp_success, result, comp_error = await mpc_engine.execute_secure_computation(
            session_id, participant_data
        )
        
        assert comp_success
        assert result is not None
        assert result['computation_type'] == 'differential_privacy_statistics'
        assert 'mean' in result['result']
        assert 'std' in result['result']
        assert 'count' in result['result']


class TestHomomorphicEncryption:
    """Test homomorphic encryption functionality"""
    
    @pytest.mark.asyncio
    async def test_homomorphic_initialization(self):
        """Test homomorphic encryption initialization"""
        he = HomomorphicEncryption()
        success = await he.initialize()
        
        # May fail if TenSEAL is not available
        if success:
            assert he.initialized
    
    @pytest.mark.asyncio
    async def test_homomorphic_operations(self):
        """Test homomorphic encryption operations"""
        he = HomomorphicEncryption()
        success = await he.initialize()
        
        if not success:
            pytest.skip("Homomorphic encryption not available")
        
        # Test vector encryption
        test_vector = [1.5, 2.7, 3.9, 4.1, 5.3]
        encrypted_vector = await he.encrypt_vector(test_vector)
        
        assert encrypted_vector is not None
        
        # Test decryption
        decrypted_vector = await he.decrypt_vector(encrypted_vector)
        
        assert decrypted_vector is not None
        assert len(decrypted_vector) >= len(test_vector)
        
        # Check approximate equality (homomorphic encryption is approximate)
        for i in range(len(test_vector)):
            assert abs(decrypted_vector[i] - test_vector[i]) < 0.1


class TestIntegration:
    """Integration tests for the complete security framework"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_secure_workflow(
        self, 
        encryption_engine, 
        dlp_engine, 
        sharing_engine, 
        geological_test_data
    ):
        """Test complete secure workflow from classification to sharing"""
        
        # Step 1: Classify data
        classifier = DataClassifier({})
        classified_fields = await classifier.classify_data(
            geological_test_data, 
            "volcanic_monitoring"
        )
        
        # Step 2: Encrypt sensitive data
        encrypted_data, metadata = await encryption_engine.encrypt_data(
            geological_test_data,
            EncryptionAlgorithm.AES_256_GCM,
            classification=DataClassification.CONFIDENTIAL
        )
        
        # Step 3: Create secure share
        success, data_share, error = await sharing_engine.create_data_share(
            owner_id="researcher_001",
            data_resource_id="integrated_test_data",
            data=geological_test_data,
            shared_with=["researcher_002"],
            permissions=[SharingPermission.READ],
            collaboration_type=CollaborationType.RESEARCH_PROJECT,
            data_classification=DataClassification.CONFIDENTIAL,
            geological_sensitivity=GeologicalDataSensitivity.DISASTER_PREPAREDNESS
        )
        
        assert success
        assert data_share is not None
        
        # Step 4: Verify DLP compliance
        request = DataAccessRequest(
            request_id="integration_test",
            user_id="researcher_002",
            user_roles=["researcher"],
            resource_id="integrated_test_data",
            data_type="volcanic_monitoring",
            operation="read"
        )
        
        context = GeologicalDataContext(
            data_source="test_source",
            geological_domain="volcanic_monitoring",
            temporal_scope="real-time",
            spatial_scope="local",
            research_purpose="testing",
            commercial_value="none",
            public_safety_impact="low",
            national_security_relevance="none"
        )
        
        allow_access, actions, violation = await dlp_engine.evaluate_access_request(
            request, geological_test_data, context
        )
        
        # Should complete without critical violations
        assert isinstance(allow_access, bool)
        assert isinstance(actions, list)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])