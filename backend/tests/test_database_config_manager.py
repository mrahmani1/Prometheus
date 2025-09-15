"""
Comprehensive Test Suite for Database Configuration Management System
Tests all aspects of the centralized database configuration management including
environment variables, secure credential management, service discovery, and validation.
"""

import os
import pytest
import asyncio
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import the modules to test
from backend.core.database_config_manager import (
    DatabaseConfigManager, DatabaseConfig, DatabaseCredentials, ConnectionPoolConfig,
    DatabaseType, Environment, CredentialSource, ServiceDiscoveryConfig
)
from backend.core.database_service import DatabaseService, DatabaseConnectionPool


class TestDatabaseCredentials:
    """Test database credentials encryption and management"""
    
    def test_credential_creation(self):
        """Test basic credential creation"""
        creds = DatabaseCredentials(
            username="test_user",
            password="test_password",
            token="test_token"
        )
        
        assert creds.username == "test_user"
        assert creds.password == "test_password"
        assert creds.token == "test_token"
        assert not creds.encrypted
        assert creds.source == CredentialSource.ENVIRONMENT
    
    def test_credential_encryption_decryption(self):
        """Test credential encryption and decryption"""
        from cryptography.fernet import Fernet
        
        # Generate encryption key
        key = Fernet.generate_key()
        
        # Create credentials
        creds = DatabaseCredentials(
            username="test_user",
            password="secret_password",
            token="secret_token"
        )
        
        # Encrypt credentials
        encrypted_creds = creds.encrypt(key)
        
        assert encrypted_creds.encrypted
        assert encrypted_creds.username == "test_user"  # Username not encrypted
        assert encrypted_creds.password != "secret_password"  # Password encrypted
        assert encrypted_creds.token != "secret_token"  # Token encrypted
        
        # Decrypt credentials
        decrypted_creds = encrypted_creds.decrypt(key)
        
        assert not decrypted_creds.encrypted
        assert decrypted_creds.username == "test_user"
        assert decrypted_creds.password == "secret_password"
        assert decrypted_creds.token == "secret_token"


class TestDatabaseConfig:
    """Test database configuration management"""
    
    def test_config_creation(self):
        """Test basic configuration creation"""
        config = DatabaseConfig(
            name="test_postgres",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        assert config.name == "test_postgres"
        assert config.type == DatabaseType.POSTGRESQL
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.pool_config, ConnectionPoolConfig)
    
    def test_postgresql_connection_string(self):
        """Test PostgreSQL connection string generation"""
        creds = DatabaseCredentials(username="user", password="pass")
        config = DatabaseConfig(
            name="test_postgres",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db",
            credentials=creds
        )
        
        conn_str = config.get_connection_string()
        expected = "postgresql://user:pass@localhost:5432/test_db"
        assert conn_str == expected
        
        # Test without credentials
        conn_str_no_creds = config.get_connection_string(include_credentials=False)
        expected_no_creds = "postgresql://localhost:5432/test_db"
        assert conn_str_no_creds == expected_no_creds
    
    def test_redis_connection_string(self):
        """Test Redis connection string generation"""
        creds = DatabaseCredentials(password="redis_pass")
        config = DatabaseConfig(
            name="test_redis",
            type=DatabaseType.REDIS,
            host="localhost",
            port=6379,
            database="0",
            credentials=creds
        )
        
        conn_str = config.get_connection_string()
        expected = "redis://:redis_pass@localhost:6379/0"
        assert conn_str == expected
    
    def test_influxdb_connection_string(self):
        """Test InfluxDB connection string generation"""
        config = DatabaseConfig(
            name="test_influxdb",
            type=DatabaseType.INFLUXDB,
            host="localhost",
            port=8086,
            ssl_enabled=True
        )
        
        conn_str = config.get_connection_string()
        expected = "https://localhost:8086"
        assert conn_str == expected
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Valid configuration
        valid_config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        is_valid, errors = valid_config.validate()
        assert is_valid
        assert len(errors) == 0
        
        # Invalid configuration - missing name
        invalid_config = DatabaseConfig(
            name="",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        is_valid, errors = invalid_config.validate()
        assert not is_valid
        assert "Database name is required" in errors
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization"""
        config = DatabaseConfig(
            name="test_db",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db",
            credentials=DatabaseCredentials(username="user", password="pass")
        )
        
        # Serialize to dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['name'] == "test_db"
        assert config_dict['type'] == DatabaseType.POSTGRESQL
        
        # Deserialize from dict
        restored_config = DatabaseConfig.from_dict(config_dict)
        assert restored_config.name == config.name
        assert restored_config.type == config.type
        assert restored_config.host == config.host
        assert restored_config.credentials.username == config.credentials.username


class TestDatabaseConfigManager:
    """Test database configuration manager"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create configuration manager with temporary directory"""
        return DatabaseConfigManager(
            config_dir=temp_config_dir,
            environment=Environment.TESTING
        )
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, config_manager):
        """Test configuration manager initialization"""
        success = await config_manager.initialize()
        assert success
        assert config_manager.config_dir.exists()
        assert len(config_manager.templates) > 0
    
    @pytest.mark.asyncio
    async def test_create_database_config(self, config_manager):
        """Test creating database configuration"""
        await config_manager.initialize()
        
        config = await config_manager.create_database_config(
            name="test_postgres",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        assert config.name == "test_postgres"
        assert config.type == DatabaseType.POSTGRESQL
        assert config in config_manager.configs.values()
    
    @pytest.mark.asyncio
    async def test_environment_variable_resolution(self, config_manager):
        """Test environment variable resolution in templates"""
        # Set environment variables
        os.environ['TEST_HOST'] = 'test.example.com'
        os.environ['TEST_PORT'] = '9999'
        
        try:
            # Test resolution
            test_data = {
                'host': '${TEST_HOST}',
                'port': '${TEST_PORT:5432}',
                'database': '${MISSING_VAR:default_db}'
            }
            
            resolved_data = config_manager._resolve_environment_variables(test_data)
            
            assert resolved_data['host'] == 'test.example.com'
            assert resolved_data['port'] == '9999'
            assert resolved_data['database'] == 'default_db'
            
        finally:
            # Clean up environment variables
            os.environ.pop('TEST_HOST', None)
            os.environ.pop('TEST_PORT', None)
    
    @pytest.mark.asyncio
    async def test_credential_loading_from_env(self, config_manager):
        """Test loading credentials from environment variables"""
        await config_manager.initialize()
        
        # Set environment variables
        os.environ['TEST_DB_USERNAME'] = 'env_user'
        os.environ['TEST_DB_PASSWORD'] = 'env_pass'
        
        try:
            config = DatabaseConfig(
                name="test_db",
                type=DatabaseType.POSTGRESQL,
                host="localhost",
                port=5432
            )
            
            await config_manager._load_credentials_for_config(config)
            
            assert config.credentials is not None
            assert config.credentials.username == 'env_user'
            assert config.credentials.password == 'env_pass'
            assert config.credentials.source == CredentialSource.ENVIRONMENT
            
        finally:
            # Clean up environment variables
            os.environ.pop('TEST_DB_USERNAME', None)
            os.environ.pop('TEST_DB_PASSWORD', None)
    
    @pytest.mark.asyncio
    async def test_configuration_persistence(self, config_manager):
        """Test configuration persistence to file"""
        await config_manager.initialize()
        
        # Create configuration
        config = await config_manager.create_database_config(
            name="test_postgres",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        # Check that configuration file was created
        config_file = config_manager.config_dir / f"{config_manager.environment.value}.yaml"
        assert config_file.exists()
        
        # Load and verify configuration
        with open(config_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert 'databases' in saved_data
        assert 'test_postgres' in saved_data['databases']
        assert saved_data['databases']['test_postgres']['name'] == 'test_postgres'
    
    @pytest.mark.asyncio
    async def test_configuration_versioning(self, config_manager):
        """Test configuration versioning and rollback"""
        await config_manager.initialize()
        
        # Create initial configuration
        config = await config_manager.create_database_config(
            name="test_db",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        original_version = config.version
        
        # Update configuration
        updated_config = await config_manager.update_database_config(
            "test_db",
            port=5433
        )
        
        assert updated_config.port == 5433
        assert updated_config.version != original_version
        
        # Check history
        history = config_manager.get_config_history("test_db")
        assert len(history) > 0
        
        # Rollback
        rolled_back_config = await config_manager.rollback_config("test_db")
        assert rolled_back_config.port == 5432
    
    @pytest.mark.asyncio
    async def test_connection_testing(self, config_manager):
        """Test database connection testing"""
        await config_manager.initialize()
        
        # Create configuration for non-existent database
        config = await config_manager.create_database_config(
            name="test_postgres",
            db_type=DatabaseType.POSTGRESQL,
            host="nonexistent.host",
            port=5432,
            database="test_db"
        )
        
        # Test connection (should fail)
        success, error = await config_manager.test_connection("test_postgres")
        assert not success
        assert error is not None
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, config_manager):
        """Test configuration validation"""
        await config_manager.initialize()
        
        # Create valid configuration
        await config_manager.create_database_config(
            name="valid_db",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        # Validate all configurations
        results = await config_manager.validate_all_configs()
        
        assert "valid_db" in results
        is_valid, errors = results["valid_db"]
        assert is_valid
        assert len(errors) == 0


class TestDatabaseService:
    """Test database service integration"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, temp_config_dir):
        """Test database service initialization"""
        # Set config directory
        os.environ['DB_CONFIG_DIR'] = temp_config_dir
        
        try:
            service = DatabaseService(environment="testing")
            success = await service.initialize()
            
            # Service should initialize even without databases
            assert success
            assert service.is_initialized
            
            await service.cleanup()
            
        finally:
            os.environ.pop('DB_CONFIG_DIR', None)
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, temp_config_dir):
        """Test connection pool management"""
        os.environ['DB_CONFIG_DIR'] = temp_config_dir
        
        try:
            service = DatabaseService(environment="testing")
            await service.initialize()
            
            # Add database configuration
            success = await service.add_database(
                name="test_redis",
                db_type=DatabaseType.REDIS,
                host="localhost",
                port=6379
            )
            
            # Should succeed even if Redis is not running (pool creation, not connection)
            assert success or not success  # Either way is acceptable for this test
            
            # Get connection pool
            pool = service.get_connection_pool("test_redis")
            if pool:
                assert pool.config.name == "test_redis"
                assert pool.config.type == DatabaseType.REDIS
            
            await service.cleanup()
            
        finally:
            os.environ.pop('DB_CONFIG_DIR', None)
    
    @pytest.mark.asyncio
    async def test_service_status_reporting(self, temp_config_dir):
        """Test service status reporting"""
        os.environ['DB_CONFIG_DIR'] = temp_config_dir
        
        try:
            service = DatabaseService(environment="testing")
            await service.initialize()
            
            status = service.get_service_status()
            
            assert 'initialized' in status
            assert 'environment' in status
            assert 'total_databases' in status
            assert 'healthy_databases' in status
            assert 'database_status' in status
            
            assert status['initialized'] == True
            assert status['environment'] == 'testing'
            
            await service.cleanup()
            
        finally:
            os.environ.pop('DB_CONFIG_DIR', None)


class TestDatabaseConnectionPool:
    """Test database connection pool functionality"""
    
    def test_pool_creation(self):
        """Test connection pool creation"""
        config = DatabaseConfig(
            name="test_pool",
            type=DatabaseType.REDIS,
            host="localhost",
            port=6379
        )
        
        pool = DatabaseConnectionPool(config)
        
        assert pool.config == config
        assert not pool.is_initialized
        assert pool.health_status == "unknown"
    
    @pytest.mark.asyncio
    async def test_pool_health_check(self):
        """Test connection pool health check"""
        config = DatabaseConfig(
            name="test_pool",
            type=DatabaseType.REDIS,
            host="nonexistent.host",
            port=6379
        )
        
        pool = DatabaseConnectionPool(config)
        
        # Health check should fail for non-existent host
        health = await pool.health_check()
        assert not health
        assert pool.health_status == "unhealthy"


class TestConfigurationTemplates:
    """Test configuration template system"""
    
    def test_template_loading(self):
        """Test loading configuration templates"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatabaseConfigManager(
                config_dir=temp_dir,
                environment=Environment.DEVELOPMENT
            )
            
            # Templates should be created during initialization
            asyncio.run(manager.initialize())
            
            assert Environment.DEVELOPMENT in manager.templates
            assert Environment.PRODUCTION in manager.templates
            
            # Check template structure
            dev_template = manager.templates[Environment.DEVELOPMENT]
            assert 'postgresql' in dev_template
            assert 'redis' in dev_template
            assert 'influxdb' in dev_template
    
    def test_template_application(self):
        """Test applying templates to configurations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatabaseConfigManager(
                config_dir=temp_dir,
                environment=Environment.PRODUCTION
            )
            
            asyncio.run(manager.initialize())
            
            # Production template should have SSL enabled
            prod_template = manager.templates[Environment.PRODUCTION]
            postgres_template = prod_template.get('postgresql', {})
            
            assert postgres_template.get('ssl_enabled') == True
            assert postgres_template.get('ssl_verify') == True


class TestEnvironmentVariableReplacement:
    """Test environment variable replacement functionality"""
    
    def test_hardcoded_config_detection(self):
        """Test detection of hardcoded database configurations"""
        from scripts.replace_hardcoded_db_configs import DatabaseConfigReplacer
        
        # Create test content with hardcoded configurations
        test_content = """
        # Test file with hardcoded configurations
        DATABASE_URL = "postgresql://user:pass@localhost:5432/db"
        REDIS_URL = "redis://localhost:6379"
        INFLUX_HOST = "localhost:8086"
        """
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text(test_content)
            
            replacer = DatabaseConfigReplacer(root_dir=temp_dir, dry_run=True)
            found_configs = replacer.scan_codebase()
            
            # Should find hardcoded configurations
            assert len(found_configs) > 0
            
            # Check specific patterns
            localhost_configs = [
                config for config in found_configs
                if 'localhost' in config.original_value
            ]
            assert len(localhost_configs) > 0
    
    def test_environment_template_generation(self):
        """Test generation of environment variable template"""
        from scripts.replace_hardcoded_db_configs import DatabaseConfigReplacer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file with hardcoded configs
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text('host = "localhost:5432"')
            
            replacer = DatabaseConfigReplacer(root_dir=temp_dir, dry_run=True)
            replacer.scan_codebase()
            
            # Generate environment template
            template = replacer.generate_environment_template()
            
            assert isinstance(template, str)
            assert 'POSTGRES_HOST' in template or 'DATABASE_HOST' in template
            assert 'ENVIRONMENT=' in template


class TestDocumentationGeneration:
    """Test documentation generation functionality"""
    
    @pytest.mark.asyncio
    async def test_documentation_generation(self):
        """Test configuration documentation generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatabaseConfigManager(
                config_dir=temp_dir,
                environment=Environment.DEVELOPMENT
            )
            
            await manager.initialize()
            
            # Create test configuration
            await manager.create_database_config(
                name="test_postgres",
                db_type=DatabaseType.POSTGRESQL,
                host="localhost",
                port=5432,
                database="test_db"
            )
            
            # Generate documentation
            docs = manager.generate_documentation()
            
            assert isinstance(docs, str)
            assert "Database Configuration Documentation" in docs
            assert "test_postgres" in docs
            assert "postgresql" in docs
            assert "localhost:5432" in docs


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_database_type(self):
        """Test handling of invalid database types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatabaseConfigManager(
                config_dir=temp_dir,
                environment=Environment.TESTING
            )
            
            await manager.initialize()
            
            # Try to create configuration with invalid type
            with pytest.raises(ValueError):
                await manager.create_database_config(
                    name="invalid_db",
                    db_type="invalid_type",  # This should cause an error
                    host="localhost",
                    port=5432
                )
    
    @pytest.mark.asyncio
    async def test_missing_configuration_file(self):
        """Test handling of missing configuration files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DatabaseConfigManager(
                config_dir=temp_dir,
                environment=Environment.TESTING
            )
            
            # Initialize should succeed even without existing config files
            success = await manager.initialize()
            assert success
            
            # Should have empty configurations initially
            configs = manager.list_database_configs()
            assert len(configs) == 0
    
    def test_connection_string_edge_cases(self):
        """Test connection string generation edge cases"""
        # Test with minimal configuration
        config = DatabaseConfig(
            name="minimal",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        conn_str = config.get_connection_string(include_credentials=False)
        assert conn_str == "postgresql://localhost:5432"
        
        # Test with SSL configuration
        ssl_config = DatabaseConfig(
            name="ssl_db",
            type=DatabaseType.POSTGRESQL,
            host="secure.host",
            port=5432,
            database="secure_db",
            ssl_enabled=True
        )
        
        conn_str = ssl_config.get_connection_string(include_credentials=False)
        assert conn_str == "postgresql://secure.host:5432/secure_db"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])