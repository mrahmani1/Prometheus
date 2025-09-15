"""
Comprehensive Test Suite for Database Security and Compliance Framework
Tests all security components including encryption, RBAC, audit logging, and compliance
"""

import pytest
import asyncio
import os
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

# Import the security framework components
from backend.core.database_security_compliance_framework import (
    DatabaseSecurityComplianceFramework,
    DatabaseEncryptionManager,
    RBACManager,
    AuditLogger,
    VulnerabilityScanner,
    ComplianceManager,
    DataMaskingManager,
    SecurityIncidentManager,
    EncryptionConfig,
    EncryptionType,
    RBACRole,
    SecurityUser,
    AccessLevel,
    ComplianceStandard,
    SecurityIncident,
    SecurityIncidentType,
    ThreatLevel,
    AuditLogEntry
)

from backend.core.database_manager import DatabaseManager, DatabaseConfig, DatabaseType

class TestDatabaseEncryptionManager:
    """Test encryption functionality"""
    
    @pytest.fixture
    def encryption_config(self):
        return EncryptionConfig(
            encryption_type=EncryptionType.AES_256_GCM,
            key_rotation_days=30
        )
    
    @pytest.fixture
    def encryption_manager(self, encryption_config):
        return DatabaseEncryptionManager(encryption_config)
    
    def test_generate_encryption_key(self, encryption_manager):
        """Test encryption key generation"""
        key = encryption_manager.generate_encryption_key("test_db")
        assert key is not None
        assert len(key) == 44  # Fernet key length
        assert "test_db" in encryption_manager.encryption_keys
    
    def test_encrypt_decrypt_data(self, encryption_manager):
        """Test data encryption and decryption"""
        test_data = b"sensitive geological data"
        database_name = "test_db"
        
        # Encrypt data
        encrypted_data = encryption_manager.encrypt_data(test_data, database_name)
        assert encrypted_data != test_data
        assert len(encrypted_data) > len(test_data)
        
        # Decrypt data
        decrypted_data = encryption_manager.decrypt_data(encrypted_data, database_name)
        assert decrypted_data == test_data
    
    def test_aes_gcm_encryption(self, encryption_manager):
        """Test AES-256-GCM encryption"""
        test_data = b"test data for AES-GCM"
        key = encryption_manager.generate_encryption_key("test_db")
        
        encrypted = encryption_manager._encrypt_aes_gcm(test_data, key)
        decrypted = encryption_manager._decrypt_aes_gcm(encrypted, key)
        
        assert decrypted == test_data
    
    def test_chacha20_encryption(self, encryption_manager):
        """Test ChaCha20-Poly1305 encryption"""
        encryption_manager.config.encryption_type = EncryptionType.CHACHA20_POLY1305
        test_data = b"test data for ChaCha20"
        key = encryption_manager.generate_encryption_key("test_db")
        
        encrypted = encryption_manager._encrypt_chacha20(test_data, key)
        decrypted = encryption_manager._decrypt_chacha20(encrypted, key)
        
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_key_rotation(self, encryption_manager):
        """Test encryption key rotation"""
        database_name = "test_db"
        
        # Generate initial key
        old_key = encryption_manager.generate_encryption_key(database_name)
        
        # Force key rotation by setting past rotation time
        encryption_manager.key_rotation_schedule[database_name] = datetime.utcnow() - timedelta(days=1)
        
        # Rotate keys
        results = await encryption_manager.rotate_keys()
        
        assert results[database_name] is True
        new_key = encryption_manager.encryption_keys[database_name]
        assert new_key != old_key

class TestRBACManager:
    """Test Role-Based Access Control"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def rbac_manager(self, mock_db_manager):
        return RBACManager(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_initialize_rbac_schema(self, rbac_manager, mock_db_manager):
        """Test RBAC schema initialization"""
        await rbac_manager.initialize_rbac_schema()
        
        # Verify schema creation query was called
        mock_db_manager.execute_query.assert_called_once()
        call_args = mock_db_manager.execute_query.call_args
        assert 'CREATE SCHEMA IF NOT EXISTS security' in call_args[0][1]
    
    def test_create_role(self, rbac_manager):
        """Test role creation"""
        role = RBACRole(
            role_id="test_role",
            role_name="Test Role",
            access_level=AccessLevel.READ_ONLY,
            database_permissions={"postgresql": ["SELECT"]},
            table_permissions={}
        )
        
        result = rbac_manager.create_role(role)
        assert result is True
        assert "test_role" in rbac_manager.roles
    
    def test_hash_verify_password(self, rbac_manager):
        """Test password hashing and verification"""
        password = "secure_password_123"
        
        password_hash, salt = rbac_manager.hash_password(password)
        assert password_hash is not None
        assert salt is not None
        
        # Verify correct password
        assert rbac_manager.verify_password(password, password_hash, salt) is True
        
        # Verify incorrect password
        assert rbac_manager.verify_password("wrong_password", password_hash, salt) is False
    
    def test_create_user(self, rbac_manager):
        """Test user creation"""
        password_hash, salt = rbac_manager.hash_password("test_password")
        
        user = SecurityUser(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["readonly"],
            password_hash=password_hash,
            salt=salt
        )
        
        result = rbac_manager.create_user(user)
        assert result is True
        assert "test_user" in rbac_manager.users
    
    @pytest.mark.asyncio
    async def test_authenticate_user(self, rbac_manager):
        """Test user authentication"""
        # Create test user
        password = "test_password"
        password_hash, salt = rbac_manager.hash_password(password)
        
        user = SecurityUser(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["readonly"],
            password_hash=password_hash,
            salt=salt
        )
        
        rbac_manager.create_user(user)
        
        # Test successful authentication
        session_token = await rbac_manager.authenticate_user("testuser", password, "127.0.0.1")
        assert session_token is not None
        assert session_token in rbac_manager.active_sessions
        
        # Test failed authentication
        failed_token = await rbac_manager.authenticate_user("testuser", "wrong_password", "127.0.0.1")
        assert failed_token is None
    
    def test_check_permission(self, rbac_manager):
        """Test permission checking"""
        # Create role and user
        role = RBACRole(
            role_id="test_role",
            role_name="Test Role",
            access_level=AccessLevel.READ_WRITE,
            database_permissions={"postgresql": ["SELECT", "INSERT"]},
            table_permissions={}
        )
        rbac_manager.create_role(role)
        
        # Create session
        session_token = "test_session"
        rbac_manager.active_sessions[session_token] = {
            'user_id': 'test_user',
            'username': 'testuser',
            'roles': ['test_role'],
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(hours=8)
        }
        
        # Test permission check
        assert rbac_manager.check_permission(session_token, "postgresql", "SELECT") is True
        assert rbac_manager.check_permission(session_token, "postgresql", "DELETE") is False

class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def mock_encryption_manager(self):
        encryption_manager = Mock(spec=DatabaseEncryptionManager)
        encryption_manager.encrypt_data = Mock(return_value=b"encrypted_data")
        return encryption_manager
    
    @pytest.fixture
    def audit_logger(self, mock_db_manager, mock_encryption_manager):
        return AuditLogger(mock_db_manager, mock_encryption_manager)
    
    @pytest.mark.asyncio
    async def test_start_audit_processing(self, audit_logger):
        """Test audit processing startup"""
        await audit_logger.start_audit_processing()
        assert audit_logger.processing_task is not None
        
        # Cleanup
        audit_logger.processing_task.cancel()
    
    @pytest.mark.asyncio
    async def test_log_database_operation(self, audit_logger):
        """Test database operation logging"""
        await audit_logger.log_database_operation(
            user_id="test_user",
            username="testuser",
            database_name="postgresql",
            operation="SELECT",
            table_name="users",
            query="SELECT * FROM users",
            source_ip="127.0.0.1"
        )
        
        # Verify audit entry was queued
        assert audit_logger.audit_queue.qsize() > 0
    
    def test_calculate_risk_score(self, audit_logger):
        """Test risk score calculation"""
        # High risk operation
        high_risk_score = audit_logger._calculate_risk_score("DROP", "users", "DROP TABLE users")
        assert high_risk_score > 0.7
        
        # Low risk operation
        low_risk_score = audit_logger._calculate_risk_score("SELECT", "public_data", "SELECT * FROM public_data")
        assert low_risk_score < 0.5
        
        # Suspicious query
        suspicious_score = audit_logger._calculate_risk_score("SELECT", "users", "SELECT * FROM users WHERE 1=1")
        assert suspicious_score > 0.5

class TestVulnerabilityScanner:
    """Test vulnerability scanning"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def vulnerability_scanner(self, mock_db_manager):
        return VulnerabilityScanner(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_scan_vulnerabilities(self, vulnerability_scanner, mock_db_manager):
        """Test vulnerability scanning"""
        # Mock database responses
        mock_db_manager.execute_query.return_value = [("testuser", "hash", "salt", datetime.utcnow())]
        
        results = await vulnerability_scanner.scan_vulnerabilities("test_db")
        
        assert "scan_id" in results
        assert "database_name" in results
        assert "vulnerabilities" in results
        assert "risk_score" in results
        assert results["database_name"] == "test_db"
    
    @pytest.mark.asyncio
    async def test_check_weak_passwords(self, vulnerability_scanner, mock_db_manager):
        """Test weak password detection"""
        # Mock old password
        old_date = datetime.utcnow() - timedelta(days=100)
        mock_db_manager.execute_query.return_value = [("testuser", "hash", "salt", old_date)]
        
        result = await vulnerability_scanner._check_weak_passwords("test_db")
        
        assert "vulnerable" in result
        assert "details" in result
        assert "remediation" in result
    
    @pytest.mark.asyncio
    async def test_check_excessive_privileges(self, vulnerability_scanner, mock_db_manager):
        """Test excessive privilege detection"""
        # Mock admin users
        mock_db_manager.execute_query.return_value = [
            ("admin1", "Admin", "admin", "{}"),
            ("admin2", "Admin", "admin", "{}"),
            ("admin3", "Admin", "admin", "{}"),
            ("admin4", "Admin", "admin", "{}")
        ]
        
        result = await vulnerability_scanner._check_excessive_privileges("test_db")
        
        assert "vulnerable" in result
        assert result["vulnerable"] is True  # More than 3 admin users
    
    def test_calculate_vulnerability_risk_score(self, vulnerability_scanner):
        """Test vulnerability risk score calculation"""
        vulnerabilities = [
            {"severity": "CRITICAL"},
            {"severity": "HIGH"},
            {"severity": "MEDIUM"}
        ]
        
        risk_score = vulnerability_scanner._calculate_vulnerability_risk_score(vulnerabilities)
        assert 0.0 <= risk_score <= 1.0
        assert risk_score > 0.5  # Should be high due to critical vulnerability

class TestComplianceManager:
    """Test compliance monitoring and reporting"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def mock_audit_logger(self):
        return Mock(spec=AuditLogger)
    
    @pytest.fixture
    def compliance_manager(self, mock_db_manager, mock_audit_logger):
        return ComplianceManager(mock_db_manager, mock_audit_logger)
    
    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, compliance_manager, mock_db_manager):
        """Test compliance report generation"""
        # Mock database responses
        mock_db_manager.execute_query.return_value = [(5,)]  # Mock role count
        
        report = await compliance_manager.generate_compliance_report("test_db", ComplianceStandard.SOC2)
        
        assert report.report_id is not None
        assert report.standard == ComplianceStandard.SOC2
        assert report.database_name == "test_db"
        assert report.overall_score >= 0.0
        assert report.certification_status in ["compliant", "partially_compliant", "non_compliant"]
    
    @pytest.mark.asyncio
    async def test_check_access_controls(self, compliance_manager, mock_db_manager):
        """Test access control compliance check"""
        # Mock successful RBAC implementation
        mock_db_manager.execute_query.side_effect = [(5,), (10,)]  # roles and users with roles
        
        result = await compliance_manager._check_access_controls("test_db")
        
        assert result["compliant"] is True
        assert "RBAC system" in result["description"]
    
    @pytest.mark.asyncio
    async def test_check_monitoring_logging(self, compliance_manager, mock_db_manager):
        """Test monitoring and logging compliance check"""
        # Mock active audit logging
        mock_db_manager.execute_query.return_value = [(100,)]  # Recent log entries
        
        result = await compliance_manager._check_monitoring_logging("test_db")
        
        assert result["compliant"] is True
        assert "100 entries" in result["description"]

class TestDataMaskingManager:
    """Test data masking and anonymization"""
    
    @pytest.fixture
    def mock_encryption_manager(self):
        return Mock(spec=DatabaseEncryptionManager)
    
    @pytest.fixture
    def data_masking_manager(self, mock_encryption_manager):
        return DataMaskingManager(mock_encryption_manager)
    
    def test_mask_email(self, data_masking_manager):
        """Test email masking"""
        email = "user@example.com"
        masked = data_masking_manager.mask_email(email)
        
        assert masked != email
        assert "@example.com" in masked
        assert masked.startswith("u")
        assert masked.endswith("r@example.com")
    
    def test_mask_phone(self, data_masking_manager):
        """Test phone number masking"""
        phone = "555-123-4567"
        masked = data_masking_manager.mask_phone(phone)
        
        assert masked != phone
        assert masked.endswith("4567")
        assert "***-***-" in masked
    
    def test_mask_ssn(self, data_masking_manager):
        """Test SSN masking"""
        ssn = "123-45-6789"
        masked = data_masking_manager.mask_ssn(ssn)
        
        assert masked != ssn
        assert masked.endswith("6789")
        assert masked.startswith("***-**-")
    
    def test_mask_credit_card(self, data_masking_manager):
        """Test credit card masking"""
        cc = "1234-5678-9012-3456"
        masked = data_masking_manager.mask_credit_card(cc)
        
        assert masked != cc
        assert masked.endswith("3456")
        assert "****-****-****-" in masked
    
    def test_mask_name(self, data_masking_manager):
        """Test name masking"""
        name = "John Doe"
        masked = data_masking_manager.mask_name(name)
        
        assert masked != name
        assert masked.startswith("J")
        assert "D" in masked
        assert "***" in masked
    
    @pytest.mark.asyncio
    async def test_apply_masking(self, data_masking_manager):
        """Test applying masking rules to data"""
        test_data = [
            {
                "id": 1,
                "email": "user@example.com",
                "phone": "555-123-4567",
                "name": "John Doe"
            }
        ]
        
        masked_data = await data_masking_manager.apply_masking("test_db", "users", test_data)
        
        assert len(masked_data) == 1
        assert masked_data[0]["id"] == 1  # ID not masked
        assert masked_data[0]["email"] != test_data[0]["email"]
        assert masked_data[0]["phone"] != test_data[0]["phone"]

class TestSecurityIncidentManager:
    """Test security incident response"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def mock_audit_logger(self):
        audit_logger = Mock(spec=AuditLogger)
        audit_logger.log_database_operation = AsyncMock()
        return audit_logger
    
    @pytest.fixture
    def incident_manager(self, mock_db_manager, mock_audit_logger):
        return SecurityIncidentManager(mock_db_manager, mock_audit_logger)
    
    @pytest.mark.asyncio
    async def test_create_incident(self, incident_manager, mock_audit_logger):
        """Test security incident creation"""
        incident = SecurityIncident(
            incident_id="test_incident",
            incident_type=SecurityIncidentType.UNAUTHORIZED_ACCESS,
            severity=ThreatLevel.HIGH,
            database_name="test_db",
            detected_at=datetime.utcnow(),
            description="Test security incident",
            affected_users=["testuser"],
            affected_data=["users"],
            source_ip="192.168.1.100",
            attack_vector="SQL injection",
            indicators_of_compromise=["suspicious query"],
            response_actions=["block IP"],
            status="open",
            assigned_to="security_team"
        )
        
        incident_id = await incident_manager.create_incident(incident)
        
        assert incident_id == "test_incident"
        assert incident_id in incident_manager.active_incidents
        mock_audit_logger.log_database_operation.assert_called_once()

class TestDatabaseSecurityComplianceFramework:
    """Test the main security framework"""
    
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock(spec=DatabaseManager)
        db_manager.execute_query = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def security_framework(self, mock_db_manager):
        return DatabaseSecurityComplianceFramework(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_initialize_security_framework(self, security_framework):
        """Test security framework initialization"""
        with patch.object(security_framework.rbac_manager, 'initialize_rbac_schema', new_callable=AsyncMock):
            with patch.object(security_framework.audit_logger, 'start_audit_processing', new_callable=AsyncMock):
                with patch.object(security_framework.security_auditor, 'start_monitoring', new_callable=AsyncMock):
                    results = await security_framework.initialize_security_framework()
                    
                    assert results['rbac_schema'] is True
                    assert results['audit_logging'] is True
                    assert results['security_monitoring'] is True
                    assert results['default_roles'] is True
    
    @pytest.mark.asyncio
    async def test_perform_security_assessment(self, security_framework):
        """Test comprehensive security assessment"""
        # Mock vulnerability scan results
        vuln_results = {
            'scan_id': 'test_scan',
            'vulnerabilities': [],
            'risk_score': 0.2
        }
        
        # Mock compliance report
        compliance_report = Mock()
        compliance_report.overall_score = 0.9
        
        with patch.object(security_framework.vulnerability_scanner, 'scan_vulnerabilities', return_value=vuln_results):
            with patch.object(security_framework.compliance_manager, 'generate_compliance_report', return_value=compliance_report):
                assessment = await security_framework.perform_security_assessment("test_db")
                
                assert "assessment_id" in assessment
                assert "vulnerability_scan" in assessment
                assert "compliance_reports" in assessment
                assert "overall_security_score" in assessment
                assert assessment["database_name"] == "test_db"
    
    @pytest.mark.asyncio
    async def test_enable_encryption(self, security_framework):
        """Test encryption enablement"""
        result = await security_framework.enable_encryption("test_db")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_setup_mutual_tls(self, security_framework):
        """Test mutual TLS setup"""
        # Create temporary certificate files
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as cert_file:
            cert_file.write("-----BEGIN CERTIFICATE-----\ntest_cert\n-----END CERTIFICATE-----")
            cert_path = cert_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as key_file:
            key_file.write("-----BEGIN PRIVATE KEY-----\ntest_key\n-----END PRIVATE KEY-----")
            key_path = key_file.name
        
        try:
            result = await security_framework.setup_mutual_tls("test_db", cert_path, key_path)
            assert result is True
        finally:
            os.unlink(cert_path)
            os.unlink(key_path)

# Integration tests
class TestSecurityFrameworkIntegration:
    """Integration tests for the complete security framework"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_security_workflow(self):
        """Test complete security workflow"""
        # This would test the entire security framework workflow
        # from initialization through incident response
        pass
    
    @pytest.mark.asyncio
    async def test_compliance_audit_workflow(self):
        """Test complete compliance audit workflow"""
        # This would test the full compliance audit process
        pass
    
    @pytest.mark.asyncio
    async def test_incident_response_workflow(self):
        """Test complete incident response workflow"""
        # This would test the full incident response process
        pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])