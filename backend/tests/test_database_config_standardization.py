"""
Comprehensive Test Suite for Database Configuration Standardization

Tests all aspects of the standardized database configuration management system
including templates, versioning, rollback, discovery, and validation.
"""

import pytest
import asyncio
import tempfile
import shutil
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import os

# Import the modules to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database_config_manager import (
    DatabaseConfigManager,
    DatabaseConfig,
    DatabaseCredentials,
    ConnectionPoolConfig,
    DatabaseType,
    Environment,
    CredentialSource
)
from core.database_discovery_service import (
    DatabaseDiscoveryService,
    DiscoveredService
)


class TestDatabaseConfigManager:
    """Test suite for DatabaseConfigManager"""
    
    @pytest.fixture
    async def temp_config_dir(self):
        """Create temporary configuration directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def config_manager(self, temp_config_dir):
        """Create DatabaseConfigManager instance for testing"""
        manager = DatabaseConfigManager(
            config_dir=temp_config_dir,
            environment=Environment.DEVELOPMENT
        )
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization(self, config_manager):
        """Test configuration manager initialization"""
        assert config_manager is not None
        assert config_manager.environment == Environment.DEVELOPMENT
        assert config_manager.config_dir.exists()
        assert len(config_manager.templates) > 0
    
    @pytest.mark.asyncio
    async def test_create_database_config(self, config_manager):
        """Test creating new database configuration"""
        config = await config_manager.create_database_config(
            name="test_postgres",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        assert config is not None
        assert config.name == "test_postgres"
        assert config.type == DatabaseType.POSTGRESQL
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.version == "1.0.0"
        
        # Verify it's stored in manager
        assert "test_postgres" in config_manager.configs
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, config_manager):
        """Test configuration validation"""
        # Valid configuration
        config = await config_manager.create_database_config(
            name="valid_config",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        is_valid, errors = config.validate()
        assert is_valid
        assert len(errors) == 0
        
        # Invalid configuration (missing required fields)
        invalid_config = DatabaseConfig(
            name="",  # Empty name
            type=DatabaseType.POSTGRESQL,
            host="",  # Empty host
            port=-1   # Invalid port
        )
        
        is_valid, errors = invalid_config.validate()
        assert not is_valid
        assert len(errors) > 0
        assert any("name is required" in error.lower() for error in errors)
        assert any("host is required" in error.lower() for error in errors)
        assert any("port" in error.lower() for error in errors)
    
    @pytest.mark.asyncio
    async def test_configuration_versioning(self, config_manager):
        """Test configuration versioning system"""
        # Create initial configuration
        config = await config_manager.create_database_config(
            name="versioned_config",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        initial_version = config.version
        assert initial_version == "1.0.0"
        
        # Update configuration
        updated_config = await config_manager.update_config(
            "versioned_config",
            {"port": 5433}
        )
        
        assert updated_config.version != initial_version
        assert updated_config.port == 5433
        
        # Check history
        history = config_manager.get_config_history("versioned_config")
        assert len(history) >= 1
    
    @pytest.mark.asyncio
    async def test_configuration_rollback(self, config_manager):
        """Test configuration rollback functionality"""
        # Create and update configuration multiple times
        config = await config_manager.create_database_config(
            name="rollback_test",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        original_port = config.port
        original_version = config.version
        
        # First update
        await config_manager.update_config("rollback_test", {"port": 5433})
        
        # Second update
        await config_manager.update_config("rollback_test", {"port": 5434})
        
        current_config = config_manager.configs["rollback_test"]
        assert current_config.port == 5434
        
        # Rollback to previous version
        rolled_back = await config_manager.rollback_config("rollback_test")
        
        assert rolled_back is not None
        assert rolled_back.port != 5434  # Should be different from current
    
    @pytest.mark.asyncio
    async def test_configuration_export_import(self, config_manager):
        """Test configuration export and import"""
        # Create configuration
        original_config = await config_manager.create_database_config(
            name="export_test",
            db_type=DatabaseType.REDIS,
            host="redis.example.com",
            port=6379
        )
        
        # Export configuration
        exported_yaml = await config_manager.export_configuration("export_test", "yaml")
        exported_json = await config_manager.export_configuration("export_test", "json")
        
        assert exported_yaml is not None
        assert exported_json is not None
        assert "redis.example.com" in exported_yaml
        assert "redis.example.com" in exported_json
        
        # Import configuration with new name
        success = await config_manager.import_configuration(
            "imported_test",
            exported_yaml,
            "yaml",
            validate=False  # Skip validation for test
        )
        
        assert success
        assert "imported_test" in config_manager.configs
        
        imported_config = config_manager.configs["imported_test"]
        assert imported_config.host == original_config.host
        assert imported_config.port == original_config.port
        assert imported_config.type == original_config.type
    
    @pytest.mark.asyncio
    async def test_configuration_templates(self, config_manager):
        """Test configuration template system"""
        # Test template creation
        template_data = {
            "postgresql": {
                "host": "${POSTGRES_HOST:localhost}",
                "port": "${POSTGRES_PORT:5432}",
                "ssl_enabled": True,
                "pool_config": {
                    "min_connections": 5,
                    "max_connections": 25
                }
            }
        }
        
        success = await config_manager.create_configuration_template(
            Environment.TESTING,
            template_data
        )
        
        assert success
        assert Environment.TESTING in config_manager.templates
        assert config_manager.templates[Environment.TESTING] == template_data
    
    @pytest.mark.asyncio
    async def test_environment_variable_resolution(self, config_manager):
        """Test environment variable resolution in templates"""
        # Set test environment variables
        os.environ['TEST_DB_HOST'] = 'test.example.com'
        os.environ['TEST_DB_PORT'] = '5433'
        
        try:
            # Test resolution with environment variables
            resolved = config_manager._resolve_environment_variables({
                "host": "${TEST_DB_HOST}",
                "port": "${TEST_DB_PORT}",
                "database": "${MISSING_VAR:default_db}"
            })
            
            assert resolved["host"] == "test.example.com"
            assert resolved["port"] == "5433"
            assert resolved["database"] == "default_db"  # Default value
            
        finally:
            # Clean up environment variables
            os.environ.pop('TEST_DB_HOST', None)
            os.environ.pop('TEST_DB_PORT', None)
    
    @pytest.mark.asyncio
    async def test_connection_string_generation(self, config_manager):
        """Test database connection string generation"""
        # PostgreSQL connection string
        pg_config = DatabaseConfig(
            name="test_pg",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb",
            credentials=DatabaseCredentials(
                username="testuser",
                password="testpass"
            )
        )
        
        conn_str = pg_config.get_connection_string()
        assert "postgresql://testuser:testpass@localhost:5432/testdb" == conn_str
        
        # Redis connection string
        redis_config = DatabaseConfig(
            name="test_redis",
            type=DatabaseType.REDIS,
            host="localhost",
            port=6379,
            database="0",
            credentials=DatabaseCredentials(password="redispass")
        )
        
        redis_conn_str = redis_config.get_connection_string()
        assert "redis://:redispass@localhost:6379/0" == redis_conn_str
    
    @pytest.mark.asyncio
    async def test_configuration_diff(self, config_manager):
        """Test configuration version comparison"""
        # Create configuration and update it
        config = await config_manager.create_database_config(
            name="diff_test",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        version1 = config.version
        
        # Update configuration
        updated = await config_manager.update_config(
            "diff_test",
            {"port": 5433, "ssl_enabled": True}
        )
        
        version2 = updated.version
        
        # Get diff
        diff = await config_manager.get_configuration_diff("diff_test", version1, version2)
        
        assert diff is not None
        assert diff["version1"] == version1
        assert diff["version2"] == version2
        assert "changes" in diff
        assert "port" in diff["changes"]
        assert "ssl_enabled" in diff["changes"]
    
    @pytest.mark.asyncio
    async def test_version_cleanup(self, config_manager):
        """Test old version cleanup"""
        # Create configuration and update it multiple times
        config = await config_manager.create_database_config(
            name="cleanup_test",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432
        )
        
        # Create multiple versions
        for i in range(15):
            await config_manager.update_config("cleanup_test", {"port": 5432 + i})
        
        # Check history length
        history_before = config_manager.get_config_history("cleanup_test")
        assert len(history_before) > 10
        
        # Cleanup old versions
        removed_count = await config_manager.cleanup_old_versions("cleanup_test", keep_versions=5)
        
        assert removed_count > 0
        
        # Check history length after cleanup
        history_after = config_manager.get_config_history("cleanup_test")
        assert len(history_after) <= 5


class TestDatabaseDiscoveryService:
    """Test suite for DatabaseDiscoveryService"""
    
    @pytest.fixture
    async def discovery_service(self, config_manager):
        """Create DatabaseDiscoveryService instance for testing"""
        discovery_config = {
            'scan_interval': 60,
            'network_ranges': ['127.0.0.1/32'],
            'network_scan': False,  # Disable for testing
            'docker_discovery': False,
            'kubernetes_discovery': False,
            'consul_discovery': False
        }
        
        service = DatabaseDiscoveryService(config_manager, discovery_config)
        await service.initialize()
        return service
    
    @pytest.mark.asyncio
    async def test_discovery_service_initialization(self, discovery_service):
        """Test discovery service initialization"""
        assert discovery_service is not None
        assert discovery_service.config_manager is not None
        assert not discovery_service.is_running
    
    @pytest.mark.asyncio
    async def test_database_type_identification(self, discovery_service):
        """Test database type identification methods"""
        # Test port identification
        assert discovery_service._identify_database_type_by_port(5432) == DatabaseType.POSTGRESQL
        assert discovery_service._identify_database_type_by_port(6379) == DatabaseType.REDIS
        assert discovery_service._identify_database_type_by_port(8086) == DatabaseType.INFLUXDB
        
        # Test image identification
        assert discovery_service._identify_database_type_by_image(['postgres:13']) == DatabaseType.POSTGRESQL
        assert discovery_service._identify_database_type_by_image(['redis:alpine']) == DatabaseType.REDIS
        assert discovery_service._identify_database_type_by_image(['influxdb:latest']) == DatabaseType.INFLUXDB
        
        # Test name identification
        assert discovery_service._identify_database_type_by_name('postgres-service') == DatabaseType.POSTGRESQL
        assert discovery_service._identify_database_type_by_name('redis-cache') == DatabaseType.REDIS
        assert discovery_service._identify_database_type_by_name('influxdb-metrics') == DatabaseType.INFLUXDB
    
    @pytest.mark.asyncio
    async def test_discovered_service_creation(self, discovery_service):
        """Test DiscoveredService creation and properties"""
        service = DiscoveredService(
            name="test_service",
            type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb",
            discovery_method="test"
        )
        
        assert service.name == "test_service"
        assert service.type == DatabaseType.POSTGRESQL
        assert service.host == "localhost"
        assert service.port == 5432
        assert service.database == "testdb"
        assert service.discovery_method == "test"
        assert service.discovered_at is not None
    
    @pytest.mark.asyncio
    async def test_discovery_status(self, discovery_service):
        """Test discovery service status reporting"""
        status = await discovery_service.get_discovery_status()
        
        assert isinstance(status, dict)
        assert 'is_running' in status
        assert 'discovery_methods' in status
        assert 'scan_interval' in status
        assert 'clients_available' in status
        
        assert status['is_running'] == False
        assert status['scan_interval'] == 60


class TestConfigurationIntegration:
    """Integration tests for the complete configuration system"""
    
    @pytest.fixture
    async def integrated_system(self, temp_config_dir):
        """Create integrated configuration system for testing"""
        config_manager = DatabaseConfigManager(
            config_dir=temp_config_dir,
            environment=Environment.DEVELOPMENT
        )
        await config_manager.initialize()
        
        discovery_service = DatabaseDiscoveryService(
            config_manager,
            {'network_scan': False, 'docker_discovery': False, 
             'kubernetes_discovery': False, 'consul_discovery': False}
        )
        await discovery_service.initialize()
        
        return config_manager, discovery_service
    
    @pytest.mark.asyncio
    async def test_end_to_end_configuration_lifecycle(self, integrated_system):
        """Test complete configuration lifecycle"""
        config_manager, discovery_service = integrated_system
        
        # 1. Create configuration
        config = await config_manager.create_database_config(
            name="lifecycle_test",
            db_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="testdb"
        )
        
        assert config is not None
        initial_version = config.version
        
        # 2. Validate configuration
        is_valid, errors = config.validate()
        assert is_valid
        
        # 3. Update configuration
        updated = await config_manager.update_config(
            "lifecycle_test",
            {"port": 5433, "ssl_enabled": True}
        )
        
        assert updated.version != initial_version
        assert updated.port == 5433
        assert updated.ssl_enabled == True
        
        # 4. Export configuration
        exported = await config_manager.export_configuration("lifecycle_test", "yaml")
        assert exported is not None
        assert "5433" in exported
        
        # 5. Rollback configuration
        rolled_back = await config_manager.rollback_config("lifecycle_test")
        assert rolled_back is not None
        
        # 6. Check history
        history = config_manager.get_config_history("lifecycle_test")
        assert len(history) >= 1
        
        # 7. Cleanup old versions
        removed = await config_manager.cleanup_old_versions("lifecycle_test", keep_versions=2)
        assert removed >= 0
    
    @pytest.mark.asyncio
    async def test_multi_environment_configuration(self, temp_config_dir):
        """Test configuration across multiple environments"""
        environments = [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]
        managers = {}
        
        # Create managers for each environment
        for env in environments:
            manager = DatabaseConfigManager(
                config_dir=temp_config_dir,
                environment=env
            )
            await manager.initialize()
            managers[env] = manager
        
        # Create similar configurations in each environment
        for env, manager in managers.items():
            config = await manager.create_database_config(
                name="multi_env_test",
                db_type=DatabaseType.POSTGRESQL,
                host=f"{env.value}.example.com",
                port=5432,
                ssl_enabled=(env != Environment.DEVELOPMENT)
            )
            
            assert config.host == f"{env.value}.example.com"
            assert config.ssl_enabled == (env != Environment.DEVELOPMENT)
        
        # Verify configurations are environment-specific
        dev_config = managers[Environment.DEVELOPMENT].configs["multi_env_test"]
        prod_config = managers[Environment.PRODUCTION].configs["multi_env_test"]
        
        assert dev_config.host != prod_config.host
        assert dev_config.ssl_enabled != prod_config.ssl_enabled


@pytest.mark.asyncio
async def test_configuration_performance():
    """Test configuration system performance with large numbers of configurations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = DatabaseConfigManager(
            config_dir=temp_dir,
            environment=Environment.DEVELOPMENT
        )
        await config_manager.initialize()
        
        # Create many configurations
        start_time = datetime.utcnow()
        
        for i in range(100):
            await config_manager.create_database_config(
                name=f"perf_test_{i}",
                db_type=DatabaseType.POSTGRESQL,
                host="localhost",
                port=5432 + i
            )
        
        creation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Test validation performance
        start_time = datetime.utcnow()
        results = await config_manager.validate_all_configurations()
        validation_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Performance assertions
        assert creation_time < 30.0  # Should create 100 configs in under 30 seconds
        assert validation_time < 10.0  # Should validate 100 configs in under 10 seconds
        assert len(results) == 100
        assert all(is_valid for is_valid, _ in results.values())


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--asyncio-mode=auto'])