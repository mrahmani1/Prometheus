"""
Database Configuration Management System
Centralized configuration management with environment-based settings,
secure credential handling, and dynamic configuration updates
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import hashlib
from datetime import datetime
import asyncio

# Encryption for sensitive data
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    REDIS = "redis"
    NEO4J = "neo4j"
    TIMESCALEDB = "timescaledb"

@dataclass
class DatabaseCredentials:
    """Secure database credentials"""
    username: str
    password: str
    api_key: Optional[str] = None
    token: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class ConnectionPoolConfig:
    """Database connection pool configuration"""
    min_size: int = 5
    max_size: int = 20
    max_overflow: int = 30
    timeout: int = 30
    recycle_time: int = 3600
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 60

@dataclass
class SecurityConfig:
    """Database security configuration"""
    ssl_enabled: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    ssl_ca_path: Optional[str] = None
    encryption_enabled: bool = True
    audit_logging: bool = True
    access_control_enabled: bool = True
    allowed_hosts: List[str] = field(default_factory=list)
    
@dataclass
class PerformanceConfig:
    """Database performance configuration"""
    query_timeout: int = 30
    batch_size: int = 1000
    cache_enabled: bool = True
    cache_ttl: int = 900  # 15 minutes
    compression_enabled: bool = True
    read_preference: str = "primary"
    write_concern: str = "majority"
    
@dataclass
class DatabaseConfig:
    """Complete database configuration"""
    db_type: DatabaseType
    environment: Environment
    host: str
    port: int
    database: str
    credentials: DatabaseCredentials
    pool_config: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    additional_params: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_connection_string(self) -> str:
        """Generate connection string for the database"""
        if self.db_type == DatabaseType.POSTGRESQL:
            return self._get_postgresql_connection_string()
        elif self.db_type == DatabaseType.INFLUXDB:
            return self._get_influxdb_connection_string()
        elif self.db_type == DatabaseType.REDIS:
            return self._get_redis_connection_string()
        elif self.db_type == DatabaseType.NEO4J:
            return self._get_neo4j_connection_string()
        elif self.db_type == DatabaseType.TIMESCALEDB:
            return self._get_postgresql_connection_string()  # Same as PostgreSQL
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _get_postgresql_connection_string(self) -> str:
        """Generate PostgreSQL connection string"""
        ssl_mode = "require" if self.security_config.ssl_enabled else "disable"
        return (f"postgresql://{self.credentials.username}:{self.credentials.password}@"
                f"{self.host}:{self.port}/{self.database}?sslmode={ssl_mode}")
    
    def _get_influxdb_connection_string(self) -> str:
        """Generate InfluxDB connection string"""
        protocol = "https" if self.security_config.ssl_enabled else "http"
        return f"{protocol}://{self.host}:{self.port}"
    
    def _get_redis_connection_string(self) -> str:
        """Generate Redis connection string"""
        protocol = "rediss" if self.security_config.ssl_enabled else "redis"
        auth_part = f":{self.credentials.password}@" if self.credentials.password else ""
        db_part = f"/{self.additional_params.get('db', 0)}"
        return f"{protocol}://{auth_part}{self.host}:{self.port}{db_part}"
    
    def _get_neo4j_connection_string(self) -> str:
        """Generate Neo4j connection string"""
        protocol = "bolt+s" if self.security_config.ssl_enabled else "bolt"
        return f"{protocol}://{self.host}:{self.port}"

class DatabaseConfigManager:
    """
    Database Configuration Management System
    Handles configuration loading, validation, encryption, and environment-specific settings
    """
    
    def __init__(self, config_dir: Optional[str] = None, encryption_key: Optional[str] = None):
        self.config_dir = Path(config_dir or "backend/config")
        self.encryption_key = encryption_key or os.getenv("DB_ENCRYPTION_KEY")
        self.cipher_suite = None
        
        # Configuration storage
        self.configs: Dict[str, DatabaseConfig] = {}
        self.environment_configs: Dict[Environment, Dict[DatabaseType, DatabaseConfig]] = {}
        
        # Configuration validation rules
        self.validation_rules = self._load_validation_rules()
        
        # Configuration change callbacks
        self.change_callbacks: List[callable] = []
        
        # Initialize encryption if available
        if ENCRYPTION_AVAILABLE and self.encryption_key:
            self._initialize_encryption()
        
        logger.info("Database Configuration Manager initialized")
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive data"""
        try:
            # Derive key from password
            password = self.encryption_key.encode()
            salt = b'prometheus_db_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self.cipher_suite = Fernet(key)
            logger.info("Database encryption initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self.cipher_suite = None
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load configuration validation rules"""
        return {
            'postgresql': {
                'required_fields': ['host', 'port', 'database', 'username', 'password'],
                'port_range': (1, 65535),
                'default_port': 5432
            },
            'influxdb': {
                'required_fields': ['host', 'port', 'token'],
                'port_range': (1, 65535),
                'default_port': 8086
            },
            'redis': {
                'required_fields': ['host', 'port'],
                'port_range': (1, 65535),
                'default_port': 6379
            },
            'neo4j': {
                'required_fields': ['host', 'port', 'username', 'password'],
                'port_range': (1, 65535),
                'default_port': 7687
            },
            'timescaledb': {
                'required_fields': ['host', 'port', 'database', 'username', 'password'],
                'port_range': (1, 65535),
                'default_port': 5432
            }
        }
    
    async def load_configurations(self, environment: Environment) -> Dict[DatabaseType, DatabaseConfig]:
        """Load database configurations for specific environment"""
        try:
            config_file = self.config_dir / f"database_{environment.value}.yaml"
            
            if not config_file.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return self._load_from_environment_variables(environment)
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            configs = {}
            for db_type_str, db_config in config_data.get('databases', {}).items():
                try:
                    db_type = DatabaseType(db_type_str)
                    config = self._parse_database_config(db_type, environment, db_config)
                    
                    # Validate configuration
                    if self._validate_config(config):
                        configs[db_type] = config
                        logger.info(f"Loaded configuration for {db_type.value}")
                    else:
                        logger.error(f"Invalid configuration for {db_type.value}")
                        
                except ValueError as e:
                    logger.error(f"Invalid database type {db_type_str}: {e}")
                except Exception as e:
                    logger.error(f"Error parsing config for {db_type_str}: {e}")
            
            # Store in cache
            self.environment_configs[environment] = configs
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            return self._load_from_environment_variables(environment)
    
    def _load_from_environment_variables(self, environment: Environment) -> Dict[DatabaseType, DatabaseConfig]:
        """Load configurations from environment variables as fallback"""
        configs = {}
        
        for db_type in DatabaseType:
            try:
                config = self._create_config_from_env(db_type, environment)
                if config and self._validate_config(config):
                    configs[db_type] = config
                    logger.info(f"Loaded {db_type.value} config from environment variables")
                    
            except Exception as e:
                logger.warning(f"Failed to load {db_type.value} from environment: {e}")
        
        return configs
    
    def _create_config_from_env(self, db_type: DatabaseType, environment: Environment) -> Optional[DatabaseConfig]:
        """Create configuration from environment variables"""
        env_prefix = f"{db_type.value.upper()}_{environment.value.upper()}"
        
        # Check if required environment variables exist
        host = os.getenv(f"{env_prefix}_HOST") or os.getenv(f"{db_type.value.upper()}_HOST")
        if not host:
            return None
        
        # Get default port
        default_port = self.validation_rules[db_type.value]['default_port']
        port = int(os.getenv(f"{env_prefix}_PORT") or os.getenv(f"{db_type.value.upper()}_PORT") or default_port)
        
        # Database name
        database = os.getenv(f"{env_prefix}_DATABASE") or os.getenv(f"{db_type.value.upper()}_DATABASE") or "prometheus"
        
        # Credentials
        username = os.getenv(f"{env_prefix}_USERNAME") or os.getenv(f"{db_type.value.upper()}_USERNAME") or "prometheus"
        password = os.getenv(f"{env_prefix}_PASSWORD") or os.getenv(f"{db_type.value.upper()}_PASSWORD") or ""
        api_key = os.getenv(f"{env_prefix}_API_KEY") or os.getenv(f"{db_type.value.upper()}_API_KEY")
        token = os.getenv(f"{env_prefix}_TOKEN") or os.getenv(f"{db_type.value.upper()}_TOKEN")
        
        credentials = DatabaseCredentials(
            username=username,
            password=password,
            api_key=api_key,
            token=token
        )
        
        # Pool configuration
        pool_config = ConnectionPoolConfig(
            min_size=int(os.getenv(f"{env_prefix}_POOL_MIN_SIZE", "5")),
            max_size=int(os.getenv(f"{env_prefix}_POOL_MAX_SIZE", "20")),
            max_overflow=int(os.getenv(f"{env_prefix}_POOL_MAX_OVERFLOW", "30")),
            timeout=int(os.getenv(f"{env_prefix}_TIMEOUT", "30"))
        )
        
        # Security configuration
        security_config = SecurityConfig(
            ssl_enabled=os.getenv(f"{env_prefix}_SSL_ENABLED", "true").lower() == "true",
            encryption_enabled=os.getenv(f"{env_prefix}_ENCRYPTION_ENABLED", "true").lower() == "true",
            audit_logging=os.getenv(f"{env_prefix}_AUDIT_LOGGING", "true").lower() == "true"
        )
        
        # Performance configuration
        performance_config = PerformanceConfig(
            query_timeout=int(os.getenv(f"{env_prefix}_QUERY_TIMEOUT", "30")),
            batch_size=int(os.getenv(f"{env_prefix}_BATCH_SIZE", "1000")),
            cache_enabled=os.getenv(f"{env_prefix}_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv(f"{env_prefix}_CACHE_TTL", "900"))
        )
        
        # Additional parameters
        additional_params = {}
        if db_type == DatabaseType.INFLUXDB:
            additional_params['org'] = os.getenv(f"{env_prefix}_ORG", "prometheus")
            additional_params['bucket'] = os.getenv(f"{env_prefix}_BUCKET", "geological_data")
        elif db_type == DatabaseType.REDIS:
            additional_params['db'] = int(os.getenv(f"{env_prefix}_DB", "0"))
        
        return DatabaseConfig(
            db_type=db_type,
            environment=environment,
            host=host,
            port=port,
            database=database,
            credentials=credentials,
            pool_config=pool_config,
            security_config=security_config,
            performance_config=performance_config,
            additional_params=additional_params
        )
    
    def _parse_database_config(self, db_type: DatabaseType, environment: Environment, 
                             config_data: Dict[str, Any]) -> DatabaseConfig:
        """Parse database configuration from YAML data"""
        
        # Parse credentials
        creds_data = config_data.get('credentials', {})
        credentials = DatabaseCredentials(
            username=creds_data.get('username', 'prometheus'),
            password=self._decrypt_if_encrypted(creds_data.get('password', '')),
            api_key=self._decrypt_if_encrypted(creds_data.get('api_key')),
            token=self._decrypt_if_encrypted(creds_data.get('token'))
        )
        
        # Parse pool configuration
        pool_data = config_data.get('pool', {})
        pool_config = ConnectionPoolConfig(
            min_size=pool_data.get('min_size', 5),
            max_size=pool_data.get('max_size', 20),
            max_overflow=pool_data.get('max_overflow', 30),
            timeout=pool_data.get('timeout', 30),
            recycle_time=pool_data.get('recycle_time', 3600),
            retry_attempts=pool_data.get('retry_attempts', 3),
            retry_delay=pool_data.get('retry_delay', 1.0),
            health_check_interval=pool_data.get('health_check_interval', 60)
        )
        
        # Parse security configuration
        security_data = config_data.get('security', {})
        security_config = SecurityConfig(
            ssl_enabled=security_data.get('ssl_enabled', True),
            ssl_cert_path=security_data.get('ssl_cert_path'),
            ssl_key_path=security_data.get('ssl_key_path'),
            ssl_ca_path=security_data.get('ssl_ca_path'),
            encryption_enabled=security_data.get('encryption_enabled', True),
            audit_logging=security_data.get('audit_logging', True),
            access_control_enabled=security_data.get('access_control_enabled', True),
            allowed_hosts=security_data.get('allowed_hosts', [])
        )
        
        # Parse performance configuration
        perf_data = config_data.get('performance', {})
        performance_config = PerformanceConfig(
            query_timeout=perf_data.get('query_timeout', 30),
            batch_size=perf_data.get('batch_size', 1000),
            cache_enabled=perf_data.get('cache_enabled', True),
            cache_ttl=perf_data.get('cache_ttl', 900),
            compression_enabled=perf_data.get('compression_enabled', True),
            read_preference=perf_data.get('read_preference', 'primary'),
            write_concern=perf_data.get('write_concern', 'majority')
        )
        
        return DatabaseConfig(
            db_type=db_type,
            environment=environment,
            host=config_data['host'],
            port=config_data['port'],
            database=config_data.get('database', 'prometheus'),
            credentials=credentials,
            pool_config=pool_config,
            security_config=security_config,
            performance_config=performance_config,
            additional_params=config_data.get('additional_params', {})
        )
    
    def _decrypt_if_encrypted(self, value: Optional[str]) -> Optional[str]:
        """Decrypt value if it's encrypted"""
        if not value or not self.cipher_suite:
            return value
        
        try:
            # Check if value is encrypted (starts with encrypted prefix)
            if value.startswith('encrypted:'):
                encrypted_data = value[10:]  # Remove 'encrypted:' prefix
                decrypted = self.cipher_suite.decrypt(encrypted_data.encode())
                return decrypted.decode()
            else:
                return value
                
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return value
    
    def _encrypt_sensitive_data(self, value: str) -> str:
        """Encrypt sensitive data"""
        if not self.cipher_suite:
            return value
        
        try:
            encrypted = self.cipher_suite.encrypt(value.encode())
            return f"encrypted:{encrypted.decode()}"
            
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value
    
    def _validate_config(self, config: DatabaseConfig) -> bool:
        """Validate database configuration"""
        try:
            db_type_str = config.db_type.value
            rules = self.validation_rules.get(db_type_str, {})
            
            # Check required fields
            required_fields = rules.get('required_fields', [])
            for field in required_fields:
                if field in ['username', 'password', 'token', 'api_key']:
                    # Check credentials
                    creds_dict = config.credentials.to_dict()
                    if field not in creds_dict or not creds_dict[field]:
                        logger.error(f"Missing required credential field: {field}")
                        return False
                elif field == 'host' and not config.host:
                    logger.error("Missing required field: host")
                    return False
                elif field == 'port' and not config.port:
                    logger.error("Missing required field: port")
                    return False
                elif field == 'database' and not config.database:
                    logger.error("Missing required field: database")
                    return False
            
            # Validate port range
            port_range = rules.get('port_range', (1, 65535))
            if not (port_range[0] <= config.port <= port_range[1]):
                logger.error(f"Port {config.port} out of valid range {port_range}")
                return False
            
            # Validate pool configuration
            if config.pool_config.min_size > config.pool_config.max_size:
                logger.error("Pool min_size cannot be greater than max_size")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    async def save_configuration(self, environment: Environment, 
                               configs: Dict[DatabaseType, DatabaseConfig]) -> bool:
        """Save database configurations to file"""
        try:
            config_file = self.config_dir / f"database_{environment.value}.yaml"
            
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare configuration data
            config_data = {
                'environment': environment.value,
                'created_at': datetime.utcnow().isoformat(),
                'databases': {}
            }
            
            for db_type, config in configs.items():
                # Encrypt sensitive data before saving
                config_dict = self._config_to_dict(config, encrypt_sensitive=True)
                config_data['databases'][db_type.value] = config_dict
            
            # Write to file
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration for {environment.value} environment")
            
            # Notify callbacks
            await self._notify_config_change(environment, configs)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _config_to_dict(self, config: DatabaseConfig, encrypt_sensitive: bool = False) -> Dict[str, Any]:
        """Convert DatabaseConfig to dictionary"""
        config_dict = {
            'host': config.host,
            'port': config.port,
            'database': config.database,
            'credentials': {
                'username': config.credentials.username,
                'password': self._encrypt_sensitive_data(config.credentials.password) if encrypt_sensitive else config.credentials.password
            },
            'pool': asdict(config.pool_config),
            'security': asdict(config.security_config),
            'performance': asdict(config.performance_config),
            'additional_params': config.additional_params
        }
        
        # Add optional credential fields
        if config.credentials.api_key:
            config_dict['credentials']['api_key'] = (
                self._encrypt_sensitive_data(config.credentials.api_key) if encrypt_sensitive 
                else config.credentials.api_key
            )
        
        if config.credentials.token:
            config_dict['credentials']['token'] = (
                self._encrypt_sensitive_data(config.credentials.token) if encrypt_sensitive 
                else config.credentials.token
            )
        
        return config_dict
    
    def register_change_callback(self, callback: callable):
        """Register callback for configuration changes"""
        self.change_callbacks.append(callback)
    
    async def _notify_config_change(self, environment: Environment, 
                                  configs: Dict[DatabaseType, DatabaseConfig]):
        """Notify registered callbacks of configuration changes"""
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(environment, configs)
                else:
                    callback(environment, configs)
            except Exception as e:
                logger.error(f"Error in config change callback: {e}")
    
    def get_config(self, db_type: DatabaseType, environment: Environment) -> Optional[DatabaseConfig]:
        """Get specific database configuration"""
        env_configs = self.environment_configs.get(environment, {})
        return env_configs.get(db_type)
    
    def list_environments(self) -> List[Environment]:
        """List available environments"""
        return list(self.environment_configs.keys())
    
    def list_database_types(self, environment: Environment) -> List[DatabaseType]:
        """List configured database types for environment"""
        env_configs = self.environment_configs.get(environment, {})
        return list(env_configs.keys())
    
    async def test_connection(self, config: DatabaseConfig) -> bool:
        """Test database connection with given configuration"""
        try:
            # Import the unified database manager to test connection
            from backend.core.unified_database_manager import DatabaseConnection
            
            connection = DatabaseConnection(config)
            success = await connection.connect()
            
            if success:
                await connection.disconnect()
            
            return success
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def generate_sample_config(self, environment: Environment) -> Dict[str, Any]:
        """Generate sample configuration file"""
        sample_config = {
            'environment': environment.value,
            'databases': {}
        }
        
        for db_type in DatabaseType:
            rules = self.validation_rules[db_type.value]
            
            sample_config['databases'][db_type.value] = {
                'host': 'localhost',
                'port': rules['default_port'],
                'database': 'prometheus' if db_type != DatabaseType.REDIS else None,
                'credentials': {
                    'username': 'prometheus',
                    'password': 'your_password_here'
                },
                'pool': {
                    'min_size': 5,
                    'max_size': 20,
                    'max_overflow': 30,
                    'timeout': 30
                },
                'security': {
                    'ssl_enabled': True,
                    'encryption_enabled': True,
                    'audit_logging': True
                },
                'performance': {
                    'query_timeout': 30,
                    'batch_size': 1000,
                    'cache_enabled': True,
                    'cache_ttl': 900
                }
            }
            
            # Add database-specific fields
            if db_type == DatabaseType.INFLUXDB:
                sample_config['databases'][db_type.value]['credentials']['token'] = 'your_influxdb_token'
                sample_config['databases'][db_type.value]['additional_params'] = {
                    'org': 'prometheus',
                    'bucket': 'geological_data'
                }
            elif db_type == DatabaseType.REDIS:
                sample_config['databases'][db_type.value]['additional_params'] = {
                    'db': 0
                }
        
        return sample_config

# Global configuration manager instance
_config_manager: Optional[DatabaseConfigManager] = None

def get_config_manager() -> DatabaseConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = DatabaseConfigManager()
    
    return _config_manager