"""
Centralized Database Configuration Management System
Supports all database types with environment-based configuration, secure credential management,
and automatic service discovery with production-grade features.
"""

import os
import json
import yaml
import logging
import asyncio
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Kubernetes client for secrets management
try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False

# HashiCorp Vault client
try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

# Consul client for service discovery
try:
    import consul
    CONSUL_AVAILABLE = True
except ImportError:
    CONSUL_AVAILABLE = False


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    REDIS = "redis"
    TIMESCALEDB = "timescaledb"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"
    KAFKA = "kafka"


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class CredentialSource(Enum):
    """Credential source types"""
    ENVIRONMENT = "environment"
    KUBERNETES_SECRET = "kubernetes_secret"
    VAULT = "vault"
    FILE = "file"


@dataclass
class DatabaseCredentials:
    """Database credentials with encryption support"""
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    certificate: Optional[str] = None
    private_key: Optional[str] = None
    ca_certificate: Optional[str] = None
    encrypted: bool = False
    source: CredentialSource = CredentialSource.ENVIRONMENT
    
    def encrypt(self, encryption_key: bytes) -> 'DatabaseCredentials':
        """Encrypt sensitive credential data"""
        if self.encrypted:
            return self
        
        fernet = Fernet(encryption_key)
        
        encrypted_creds = DatabaseCredentials(
            username=self.username,
            password=fernet.encrypt(self.password.encode()).decode() if self.password else None,
            token=fernet.encrypt(self.token.encode()).decode() if self.token else None,
            certificate=fernet.encrypt(self.certificate.encode()).decode() if self.certificate else None,
            private_key=fernet.encrypt(self.private_key.encode()).decode() if self.private_key else None,
            ca_certificate=fernet.encrypt(self.ca_certificate.encode()).decode() if self.ca_certificate else None,
            encrypted=True,
            source=self.source
        )
        
        return encrypted_creds
    
    def decrypt(self, encryption_key: bytes) -> 'DatabaseCredentials':
        """Decrypt sensitive credential data"""
        if not self.encrypted:
            return self
        
        fernet = Fernet(encryption_key)
        
        decrypted_creds = DatabaseCredentials(
            username=self.username,
            password=fernet.decrypt(self.password.encode()).decode() if self.password else None,
            token=fernet.decrypt(self.token.encode()).decode() if self.token else None,
            certificate=fernet.decrypt(self.certificate.encode()).decode() if self.certificate else None,
            private_key=fernet.decrypt(self.private_key.encode()).decode() if self.private_key else None,
            ca_certificate=fernet.decrypt(self.ca_certificate.encode()).decode() if self.ca_certificate else None,
            encrypted=False,
            source=self.source
        )
        
        return decrypted_creds


@dataclass
class ConnectionPoolConfig:
    """Database connection pool configuration"""
    min_connections: int = 5
    max_connections: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 300


@dataclass
class DatabaseConfig:
    """Comprehensive database configuration"""
    name: str
    type: DatabaseType
    host: str
    port: int
    database: Optional[str] = None
    schema: Optional[str] = None
    credentials: Optional[DatabaseCredentials] = None
    pool_config: Optional[ConnectionPoolConfig] = None
    ssl_enabled: bool = False
    ssl_verify: bool = True
    ssl_ca_path: Optional[str] = None
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    timeout: int = 30
    read_timeout: int = 60
    write_timeout: int = 60
    environment: Environment = Environment.DEVELOPMENT
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    
    def __post_init__(self):
        """Initialize default values"""
        if self.pool_config is None:
            self.pool_config = ConnectionPoolConfig()
        if self.credentials is None:
            self.credentials = DatabaseCredentials()
    
    def get_connection_string(self, include_credentials: bool = True) -> str:
        """Generate database connection string"""
        if self.type == DatabaseType.POSTGRESQL:
            base_url = f"postgresql://"
            if include_credentials and self.credentials:
                if self.credentials.username and self.credentials.password:
                    base_url += f"{self.credentials.username}:{self.credentials.password}@"
            base_url += f"{self.host}:{self.port}"
            if self.database:
                base_url += f"/{self.database}"
            return base_url
        
        elif self.type == DatabaseType.INFLUXDB:
            protocol = "https" if self.ssl_enabled else "http"
            return f"{protocol}://{self.host}:{self.port}"
        
        elif self.type == DatabaseType.NEO4J:
            protocol = "bolt+s" if self.ssl_enabled else "bolt"
            return f"{protocol}://{self.host}:{self.port}"
        
        elif self.type == DatabaseType.REDIS:
            protocol = "rediss" if self.ssl_enabled else "redis"
            base_url = f"{protocol}://"
            if include_credentials and self.credentials and self.credentials.password:
                base_url += f":{self.credentials.password}@"
            base_url += f"{self.host}:{self.port}"
            if self.database:
                base_url += f"/{self.database}"
            return base_url
        
        elif self.type == DatabaseType.MONGODB:
            protocol = "mongodb+srv" if self.ssl_enabled else "mongodb"
            base_url = f"{protocol}://"
            if include_credentials and self.credentials:
                if self.credentials.username and self.credentials.password:
                    base_url += f"{self.credentials.username}:{self.credentials.password}@"
            base_url += f"{self.host}:{self.port}"
            if self.database:
                base_url += f"/{self.database}"
            return base_url
        
        else:
            raise ValueError(f"Unsupported database type: {self.type}")
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate database configuration"""
        errors = []
        
        # Basic validation
        if not self.name:
            errors.append("Database name is required")
        
        if not self.host:
            errors.append("Database host is required")
        
        if not isinstance(self.port, int) or self.port <= 0:
            errors.append("Valid port number is required")
        
        # Type-specific validation
        if self.type in [DatabaseType.POSTGRESQL, DatabaseType.MONGODB]:
            if not self.database:
                errors.append(f"{self.type.value} requires database name")
        
        if self.type == DatabaseType.INFLUXDB:
            if not self.credentials or not self.credentials.token:
                errors.append("InfluxDB requires authentication token")
        
        # SSL validation
        if self.ssl_enabled:
            if self.ssl_verify and not self.ssl_ca_path:
                errors.append("SSL verification requires CA certificate path")
        
        # Pool configuration validation
        if self.pool_config:
            if self.pool_config.min_connections > self.pool_config.max_connections:
                errors.append("Minimum connections cannot exceed maximum connections")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatabaseConfig':
        """Create from dictionary representation"""
        # Handle nested objects
        if 'credentials' in data and data['credentials']:
            data['credentials'] = DatabaseCredentials(**data['credentials'])
        
        if 'pool_config' in data and data['pool_config']:
            data['pool_config'] = ConnectionPoolConfig(**data['pool_config'])
        
        # Handle enums
        if 'type' in data:
            data['type'] = DatabaseType(data['type'])
        
        if 'environment' in data:
            data['environment'] = Environment(data['environment'])
        
        # Handle datetime fields
        for field_name in ['created_at', 'updated_at']:
            if field_name in data and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)


@dataclass
class ServiceDiscoveryConfig:
    """Service discovery configuration"""
    enabled: bool = False
    consul_host: str = "localhost"
    consul_port: int = 8500
    consul_token: Optional[str] = None
    service_name_prefix: str = "prometheus-db"
    health_check_interval: int = 30
    deregister_critical_after: int = 300


class DatabaseConfigManager:
    """
    Centralized Database Configuration Management System
    
    Features:
    - Environment-based configuration
    - Secure credential management (Kubernetes Secrets, Vault)
    - Automatic service discovery
    - Configuration versioning and rollback
    - Connection string validation
    - Multi-environment support
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 environment: Environment = Environment.DEVELOPMENT,
                 encryption_key: Optional[bytes] = None):
        self.config_dir = Path(config_dir or os.getenv('DB_CONFIG_DIR', './config/database'))
        self.environment = environment
        self.logger = logging.getLogger(__name__ + ".DatabaseConfigManager")
        
        # Encryption for sensitive data
        self.encryption_key = encryption_key or self._generate_encryption_key()
        
        # Configuration storage
        self.configs: Dict[str, DatabaseConfig] = {}
        self.config_history: Dict[str, List[DatabaseConfig]] = {}
        
        # Service discovery
        self.service_discovery = ServiceDiscoveryConfig()
        self.consul_client = None
        
        # Credential sources
        self.k8s_client = None
        self.vault_client = None
        
        # Configuration templates
        self.templates: Dict[Environment, Dict[str, Any]] = {}
        
        self.logger.info(f"Database Config Manager initialized for {environment.value} environment")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key from environment or create new one"""
        key_env = os.getenv('DB_ENCRYPTION_KEY')
        if key_env:
            return base64.urlsafe_b64decode(key_env.encode())
        
        # Generate new key from password
        password = os.getenv('DB_ENCRYPTION_PASSWORD', 'prometheus-default-key').encode()
        salt = os.getenv('DB_ENCRYPTION_SALT', 'prometheus-salt').encode()
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        return base64.urlsafe_b64encode(kdf.derive(password))
    
    async def initialize(self) -> bool:
        """Initialize configuration manager"""
        try:
            self.logger.info("Initializing Database Configuration Manager")
            
            # Create configuration directory
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize credential sources
            await self._initialize_credential_sources()
            
            # Initialize service discovery
            await self._initialize_service_discovery()
            
            # Load configuration templates
            await self._load_configuration_templates()
            
            # Load existing configurations
            await self._load_configurations()
            
            # Auto-discover services if enabled
            if self.service_discovery.enabled:
                await self._discover_database_services()
            
            self.logger.info("Database Configuration Manager initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration manager initialization failed: {e}")
            return False
    
    async def _initialize_credential_sources(self):
        """Initialize credential management sources"""
        # Initialize Kubernetes client
        if KUBERNETES_AVAILABLE:
            try:
                # Try in-cluster config first, then local config
                try:
                    k8s_config.load_incluster_config()
                except k8s_config.ConfigException:
                    k8s_config.load_kube_config()
                
                self.k8s_client = client.CoreV1Api()
                self.logger.info("Kubernetes client initialized")
                
            except Exception as e:
                self.logger.warning(f"Kubernetes client initialization failed: {e}")
        
        # Initialize Vault client
        if VAULT_AVAILABLE:
            vault_url = os.getenv('VAULT_URL')
            vault_token = os.getenv('VAULT_TOKEN')
            
            if vault_url and vault_token:
                try:
                    self.vault_client = hvac.Client(url=vault_url, token=vault_token)
                    if self.vault_client.is_authenticated():
                        self.logger.info("Vault client initialized")
                    else:
                        self.logger.warning("Vault authentication failed")
                        self.vault_client = None
                        
                except Exception as e:
                    self.logger.warning(f"Vault client initialization failed: {e}")
    
    async def _initialize_service_discovery(self):
        """Initialize service discovery with Consul"""
        if not CONSUL_AVAILABLE:
            return
        
        consul_host = os.getenv('CONSUL_HOST', self.service_discovery.consul_host)
        consul_port = int(os.getenv('CONSUL_PORT', str(self.service_discovery.consul_port)))
        consul_token = os.getenv('CONSUL_TOKEN', self.service_discovery.consul_token)
        
        try:
            self.consul_client = consul.Consul(
                host=consul_host,
                port=consul_port,
                token=consul_token
            )
            
            # Test connection
            self.consul_client.agent.self()
            self.service_discovery.enabled = True
            
            self.logger.info("Consul service discovery initialized")
            
        except Exception as e:
            self.logger.warning(f"Consul initialization failed: {e}")
            self.service_discovery.enabled = False
    
    async def _load_configuration_templates(self):
        """Load configuration templates for different environments"""
        templates_dir = self.config_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        
        # Default templates for each environment
        default_templates = {
            Environment.DEVELOPMENT: {
                "postgresql": {
                    "host": "localhost",
                    "port": 5432,
                    "ssl_enabled": False,
                    "pool_config": {
                        "min_connections": 2,
                        "max_connections": 10
                    }
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "ssl_enabled": False
                },
                "influxdb": {
                    "host": "localhost",
                    "port": 8086,
                    "ssl_enabled": False
                }
            },
            Environment.PRODUCTION: {
                "postgresql": {
                    "host": "${POSTGRES_HOST}",
                    "port": "${POSTGRES_PORT:5432}",
                    "ssl_enabled": True,
                    "ssl_verify": True,
                    "pool_config": {
                        "min_connections": 10,
                        "max_connections": 50,
                        "max_overflow": 100
                    }
                },
                "redis": {
                    "host": "${REDIS_HOST}",
                    "port": "${REDIS_PORT:6379}",
                    "ssl_enabled": True
                },
                "influxdb": {
                    "host": "${INFLUXDB_HOST}",
                    "port": "${INFLUXDB_PORT:8086}",
                    "ssl_enabled": True
                }
            }
        }
        
        # Load or create templates
        for env, template_data in default_templates.items():
            template_file = templates_dir / f"{env.value}.yaml"
            
            if template_file.exists():
                with open(template_file, 'r') as f:
                    self.templates[env] = yaml.safe_load(f)
            else:
                self.templates[env] = template_data
                with open(template_file, 'w') as f:
                    yaml.dump(template_data, f, default_flow_style=False)
        
        self.logger.info(f"Loaded configuration templates for {len(self.templates)} environments")
    
    async def _load_configurations(self):
        """Load existing database configurations"""
        config_file = self.config_dir / f"{self.environment.value}.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                for name, data in config_data.get('databases', {}).items():
                    config = DatabaseConfig.from_dict(data)
                    self.configs[name] = config
                
                self.logger.info(f"Loaded {len(self.configs)} database configurations")
                
            except Exception as e:
                self.logger.error(f"Failed to load configurations: {e}")
    
    async def _discover_database_services(self):
        """Discover database services via Consul"""
        if not self.consul_client:
            return
        
        try:
            # Get all services with database prefix
            services = self.consul_client.catalog.services()[1]
            db_services = {name: tags for name, tags in services.items() 
                          if name.startswith(self.service_discovery.service_name_prefix)}
            
            for service_name, tags in db_services.items():
                # Get service details
                service_info = self.consul_client.catalog.service(service_name)[1]
                
                if service_info:
                    service = service_info[0]
                    
                    # Determine database type from tags
                    db_type = None
                    for tag in tags:
                        if tag.startswith('type:'):
                            db_type = DatabaseType(tag.split(':', 1)[1])
                            break
                    
                    if db_type:
                        # Create configuration from discovered service
                        config = DatabaseConfig(
                            name=service_name,
                            type=db_type,
                            host=service['ServiceAddress'] or service['Address'],
                            port=service['ServicePort'],
                            environment=self.environment,
                            tags={'discovered': 'true', 'consul_service': service_name}
                        )
                        
                        # Load credentials from appropriate source
                        await self._load_credentials_for_config(config)
                        
                        self.configs[service_name] = config
                        
                        self.logger.info(f"Discovered database service: {service_name}")
            
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}")
    
    async def create_database_config(self, 
                                   name: str,
                                   db_type: DatabaseType,
                                   host: str,
                                   port: int,
                                   database: Optional[str] = None,
                                   **kwargs) -> DatabaseConfig:
        """Create new database configuration"""
        # Apply template defaults
        template_data = self.templates.get(self.environment, {}).get(db_type.value, {})
        
        # Merge template with provided values
        config_data = {
            'name': name,
            'type': db_type,
            'host': host,
            'port': port,
            'database': database,
            'environment': self.environment,
            **template_data,
            **kwargs
        }
        
        # Resolve environment variables in template
        config_data = self._resolve_environment_variables(config_data)
        
        # Create configuration
        config = DatabaseConfig.from_dict(config_data)
        
        # Load credentials
        await self._load_credentials_for_config(config)
        
        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            raise ValueError(f"Invalid configuration: {', '.join(errors)}")
        
        # Store configuration
        self.configs[name] = config
        
        # Save configuration history
        if name not in self.config_history:
            self.config_history[name] = []
        self.config_history[name].append(config)
        
        # Persist to file
        await self._save_configurations()
        
        # Register with service discovery
        if self.service_discovery.enabled:
            await self._register_service(config)
        
        self.logger.info(f"Created database configuration: {name}")
        return config
    
    def _resolve_environment_variables(self, data: Any) -> Any:
        """Resolve environment variables in configuration data"""
        if isinstance(data, dict):
            return {key: self._resolve_environment_variables(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._resolve_environment_variables(item) for item in data]
        elif isinstance(data, str) and data.startswith('${') and data.endswith('}'):
            # Parse environment variable with optional default
            env_expr = data[2:-1]  # Remove ${ and }
            
            if ':' in env_expr:
                env_name, default_value = env_expr.split(':', 1)
                return os.getenv(env_name, default_value)
            else:
                return os.getenv(env_expr, data)
        else:
            return data
    
    async def _load_credentials_for_config(self, config: DatabaseConfig):
        """Load credentials for database configuration"""
        # Try different credential sources in order of preference
        
        # 1. Environment variables
        credentials = await self._load_credentials_from_env(config)
        if credentials:
            config.credentials = credentials
            return
        
        # 2. Kubernetes secrets
        if self.k8s_client:
            credentials = await self._load_credentials_from_k8s(config)
            if credentials:
                config.credentials = credentials
                return
        
        # 3. Vault
        if self.vault_client:
            credentials = await self._load_credentials_from_vault(config)
            if credentials:
                config.credentials = credentials
                return
        
        self.logger.warning(f"No credentials found for database: {config.name}")
    
    async def _load_credentials_from_env(self, config: DatabaseConfig) -> Optional[DatabaseCredentials]:
        """Load credentials from environment variables"""
        prefix = f"{config.name.upper().replace('-', '_')}_"
        
        username = os.getenv(f"{prefix}USERNAME") or os.getenv(f"{config.type.value.upper()}_USERNAME")
        password = os.getenv(f"{prefix}PASSWORD") or os.getenv(f"{config.type.value.upper()}_PASSWORD")
        token = os.getenv(f"{prefix}TOKEN") or os.getenv(f"{config.type.value.upper()}_TOKEN")
        
        if username or password or token:
            return DatabaseCredentials(
                username=username,
                password=password,
                token=token,
                source=CredentialSource.ENVIRONMENT
            )
        
        return None
    
    async def _load_credentials_from_k8s(self, config: DatabaseConfig) -> Optional[DatabaseCredentials]:
        """Load credentials from Kubernetes secrets"""
        try:
            namespace = os.getenv('KUBERNETES_NAMESPACE', 'default')
            secret_name = f"{config.name}-credentials"
            
            secret = self.k8s_client.read_namespaced_secret(
                name=secret_name,
                namespace=namespace
            )
            
            data = secret.data
            credentials = DatabaseCredentials(
                username=base64.b64decode(data.get('username', '')).decode() if 'username' in data else None,
                password=base64.b64decode(data.get('password', '')).decode() if 'password' in data else None,
                token=base64.b64decode(data.get('token', '')).decode() if 'token' in data else None,
                source=CredentialSource.KUBERNETES_SECRET
            )
            
            return credentials
            
        except ApiException as e:
            if e.status != 404:  # Ignore not found errors
                self.logger.warning(f"Failed to load K8s secret for {config.name}: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to load K8s credentials for {config.name}: {e}")
        
        return None
    
    async def _load_credentials_from_vault(self, config: DatabaseConfig) -> Optional[DatabaseCredentials]:
        """Load credentials from HashiCorp Vault"""
        try:
            secret_path = f"database/{config.environment.value}/{config.name}"
            
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=secret_path
            )
            
            data = response['data']['data']
            credentials = DatabaseCredentials(
                username=data.get('username'),
                password=data.get('password'),
                token=data.get('token'),
                source=CredentialSource.VAULT
            )
            
            return credentials
            
        except Exception as e:
            self.logger.warning(f"Failed to load Vault credentials for {config.name}: {e}")
        
        return None
    
    async def _save_configurations(self):
        """Save configurations to file"""
        config_file = self.config_dir / f"{self.environment.value}.yaml"
        
        # Prepare data for serialization
        config_data = {
            'environment': self.environment.value,
            'updated_at': datetime.utcnow().isoformat(),
            'databases': {}
        }
        
        for name, config in self.configs.items():
            # Encrypt sensitive data before saving
            config_dict = config.to_dict()
            if config.credentials:
                encrypted_creds = config.credentials.encrypt(self.encryption_key)
                config_dict['credentials'] = asdict(encrypted_creds)
            
            config_data['databases'][name] = config_dict
        
        # Write to file
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        self.logger.debug(f"Saved {len(self.configs)} configurations to {config_file}")
    
    async def _register_service(self, config: DatabaseConfig):
        """Register database service with Consul"""
        if not self.consul_client:
            return
        
        try:
            service_id = f"{self.service_discovery.service_name_prefix}-{config.name}"
            
            # Create health check
            health_check = consul.Check.tcp(
                host=config.host,
                port=config.port,
                interval=f"{self.service_discovery.health_check_interval}s",
                deregister=f"{self.service_discovery.deregister_critical_after}s"
            )
            
            # Register service
            self.consul_client.agent.service.register(
                name=service_id,
                service_id=service_id,
                address=config.host,
                port=config.port,
                tags=[
                    f"type:{config.type.value}",
                    f"environment:{config.environment.value}",
                    f"version:{config.version}"
                ],
                check=health_check
            )
            
            self.logger.info(f"Registered service with Consul: {service_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to register service {config.name}: {e}")
    
    def get_database_config(self, name: str) -> Optional[DatabaseConfig]:
        """Get database configuration by name"""
        config = self.configs.get(name)
        if config and config.credentials and config.credentials.encrypted:
            # Decrypt credentials for use
            config.credentials = config.credentials.decrypt(self.encryption_key)
        return config
    
    def list_database_configs(self) -> List[DatabaseConfig]:
        """List all database configurations"""
        return list(self.configs.values())
    
    def get_configs_by_type(self, db_type: DatabaseType) -> List[DatabaseConfig]:
        """Get configurations by database type"""
        return [config for config in self.configs.values() if config.type == db_type]
    
    def get_configs_by_environment(self, environment: Environment) -> List[DatabaseConfig]:
        """Get configurations by environment"""
        return [config for config in self.configs.values() if config.environment == environment]
    
    async def update_database_config(self, name: str, **updates) -> Optional[DatabaseConfig]:
        """Update database configuration"""
        if name not in self.configs:
            return None
        
        config = self.configs[name]
        
        # Store current version in history
        if name not in self.config_history:
            self.config_history[name] = []
        self.config_history[name].append(DatabaseConfig.from_dict(config.to_dict()))
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        config.updated_at = datetime.utcnow()
        config.version = self._increment_version(config.version)
        
        # Validate updated configuration
        is_valid, errors = config.validate()
        if not is_valid:
            # Rollback to previous version
            if self.config_history[name]:
                self.configs[name] = self.config_history[name][-1]
            raise ValueError(f"Invalid configuration update: {', '.join(errors)}")
        
        # Save changes
        await self._save_configurations()
        
        self.logger.info(f"Updated database configuration: {name}")
        return config
    
    def _increment_version(self, version: str) -> str:
        """Increment semantic version"""
        try:
            parts = version.split('.')
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except (IndexError, ValueError):
            return "1.0.1"
    
    async def rollback_config(self, name: str, version: Optional[str] = None) -> Optional[DatabaseConfig]:
        """Rollback configuration to previous version with comprehensive validation"""
        if name not in self.config_history or not self.config_history[name]:
            self.logger.warning(f"No configuration history found for {name}")
            return None
        
        if version:
            # Find specific version
            for config in reversed(self.config_history[name]):
                if config.version == version:
                    # Validate rollback target
                    is_valid, errors = config.validate()
                    if not is_valid:
                        self.logger.error(f"Cannot rollback to invalid configuration version {version}: {errors}")
                        return None
                    
                    # Test connection before rollback
                    test_success, test_error = await self._test_config_connection(config)
                    if not test_success:
                        self.logger.error(f"Connection test failed for rollback version {version}: {test_error}")
                        return None
                    
                    # Store current config in history before rollback
                    current_config = self.configs[name]
                    self.config_history[name].append(DatabaseConfig.from_dict(current_config.to_dict()))
                    
                    # Perform rollback
                    self.configs[name] = config
                    await self._save_configurations()
                    
                    # Update service discovery
                    if self.service_discovery.enabled:
                        await self._register_service(config)
                    
                    self.logger.info(f"Successfully rolled back {name} to version {version}")
                    return config
            
            self.logger.error(f"Version {version} not found in history for {name}")
            return None
        else:
            # Rollback to previous version
            if len(self.config_history[name]) < 1:
                self.logger.warning(f"No previous version available for {name}")
                return None
            
            previous_config = self.config_history[name][-1]
            
            # Validate rollback target
            is_valid, errors = previous_config.validate()
            if not is_valid:
                self.logger.error(f"Cannot rollback to invalid previous configuration: {errors}")
                return None
            
            # Test connection before rollback
            test_success, test_error = await self._test_config_connection(previous_config)
            if not test_success:
                self.logger.error(f"Connection test failed for previous version: {test_error}")
                return None
            
            # Store current config in history before rollback
            current_config = self.configs[name]
            self.config_history[name].append(DatabaseConfig.from_dict(current_config.to_dict()))
            
            # Perform rollback
            self.configs[name] = previous_config
            await self._save_configurations()
            
            # Update service discovery
            if self.service_discovery.enabled:
                await self._register_service(previous_config)
            
            self.logger.info(f"Successfully rolled back {name} to previous version {previous_config.version}")
            return previous_config
    
    def get_config_history(self, name: str) -> List[DatabaseConfig]:
        """Get configuration history for a database"""
        return self.config_history.get(name, [])
    
    async def _test_config_connection(self, config: DatabaseConfig) -> Tuple[bool, Optional[str]]:
        """Test database connection for a configuration"""
        try:
            if config.type == DatabaseType.POSTGRESQL:
                import asyncpg
                conn_str = config.get_connection_string()
                conn = await asyncpg.connect(conn_str, timeout=config.timeout)
                await conn.execute('SELECT 1')
                await conn.close()
                return True, None
                
            elif config.type == DatabaseType.REDIS:
                import aioredis
                conn_str = config.get_connection_string()
                redis = aioredis.from_url(conn_str, socket_timeout=config.timeout)
                await redis.ping()
                await redis.close()
                return True, None
                
            elif config.type == DatabaseType.INFLUXDB:
                import aiohttp
                url = config.get_connection_string()
                headers = {}
                if config.credentials and config.credentials.token:
                    headers['Authorization'] = f'Token {config.credentials.token}'
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                    async with session.get(f"{url}/ping", headers=headers) as response:
                        return response.status == 204, None
                        
            elif config.type == DatabaseType.NEO4J:
                from neo4j import AsyncGraphDatabase
                conn_str = config.get_connection_string()
                auth = None
                if config.credentials:
                    auth = (config.credentials.username, config.credentials.password)
                
                driver = AsyncGraphDatabase.driver(conn_str, auth=auth)
                async with driver.session() as session:
                    await session.run("RETURN 1")
                await driver.close()
                return True, None
                
            elif config.type == DatabaseType.MONGODB:
                from motor.motor_asyncio import AsyncIOMotorClient
                conn_str = config.get_connection_string()
                client = AsyncIOMotorClient(conn_str, serverSelectionTimeoutMS=config.timeout * 1000)
                await client.admin.command('ping')
                client.close()
                return True, None
                
            else:
                return False, f"Connection test not implemented for {config.type.value}"
                
        except Exception as e:
            return False, str(e)
    
    async def create_configuration_template(self, 
                                          environment: Environment,
                                          template_data: Dict[str, Any]) -> bool:
        """Create or update configuration template for environment"""
        try:
            templates_dir = self.config_dir / "templates"
            templates_dir.mkdir(exist_ok=True)
            
            template_file = templates_dir / f"{environment.value}.yaml"
            
            # Backup existing template
            if template_file.exists():
                backup_file = templates_dir / f"{environment.value}.yaml.backup.{int(datetime.utcnow().timestamp())}"
                template_file.rename(backup_file)
                self.logger.info(f"Backed up existing template to {backup_file}")
            
            # Save new template
            with open(template_file, 'w') as f:
                yaml.dump(template_data, f, default_flow_style=False, indent=2)
            
            # Update in-memory templates
            self.templates[environment] = template_data
            
            self.logger.info(f"Created configuration template for {environment.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration template: {e}")
            return False
    
    async def validate_all_configurations(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all database configurations"""
        validation_results = {}
        
        for name, config in self.configs.items():
            is_valid, errors = config.validate()
            validation_results[name] = (is_valid, errors)
            
            if not is_valid:
                self.logger.warning(f"Configuration {name} validation failed: {errors}")
        
        return validation_results
    
    async def test_all_connections(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        """Test connections for all database configurations"""
        connection_results = {}
        
        for name, config in self.configs.items():
            success, error = await self._test_config_connection(config)
            connection_results[name] = (success, error)
            
            if not success:
                self.logger.warning(f"Connection test failed for {name}: {error}")
        
        return connection_results
    
    async def export_configuration(self, name: str, format: str = 'yaml') -> Optional[str]:
        """Export database configuration in specified format"""
        if name not in self.configs:
            return None
        
        config = self.configs[name]
        config_dict = config.to_dict()
        
        # Remove sensitive data for export
        if 'credentials' in config_dict and config_dict['credentials']:
            config_dict['credentials'] = {
                'source': config_dict['credentials'].get('source', 'environment'),
                'encrypted': config_dict['credentials'].get('encrypted', False)
            }
        
        if format.lower() == 'json':
            return json.dumps(config_dict, indent=2, default=str)
        elif format.lower() == 'yaml':
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_configuration(self, 
                                 name: str, 
                                 config_data: str, 
                                 format: str = 'yaml',
                                 validate: bool = True) -> bool:
        """Import database configuration from string data"""
        try:
            if format.lower() == 'json':
                data = json.loads(config_data)
            elif format.lower() == 'yaml':
                data = yaml.safe_load(config_data)
            else:
                raise ValueError(f"Unsupported import format: {format}")
            
            # Create configuration from imported data
            config = DatabaseConfig.from_dict(data)
            config.name = name  # Ensure name matches
            config.updated_at = datetime.utcnow()
            
            # Load credentials for imported config
            await self._load_credentials_for_config(config)
            
            if validate:
                # Validate configuration
                is_valid, errors = config.validate()
                if not is_valid:
                    self.logger.error(f"Imported configuration validation failed: {errors}")
                    return False
                
                # Test connection
                test_success, test_error = await self._test_config_connection(config)
                if not test_success:
                    self.logger.error(f"Imported configuration connection test failed: {test_error}")
                    return False
            
            # Store configuration
            if name in self.configs:
                # Save current version to history
                if name not in self.config_history:
                    self.config_history[name] = []
                self.config_history[name].append(self.configs[name])
            
            self.configs[name] = config
            await self._save_configurations()
            
            self.logger.info(f"Successfully imported configuration for {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    async def get_configuration_diff(self, name: str, version1: str, version2: str) -> Optional[Dict[str, Any]]:
        """Get differences between two configuration versions"""
        if name not in self.config_history:
            return None
        
        config1 = None
        config2 = None
        
        # Find configurations by version
        for config in self.config_history[name]:
            if config.version == version1:
                config1 = config
            if config.version == version2:
                config2 = config
        
        # Check current config as well
        current_config = self.configs.get(name)
        if current_config:
            if current_config.version == version1:
                config1 = current_config
            if current_config.version == version2:
                config2 = current_config
        
        if not config1 or not config2:
            return None
        
        # Generate diff
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        diff = {
            'version1': version1,
            'version2': version2,
            'changes': {}
        }
        
        # Compare configurations
        all_keys = set(dict1.keys()) | set(dict2.keys())
        for key in all_keys:
            val1 = dict1.get(key)
            val2 = dict2.get(key)
            
            if val1 != val2:
                diff['changes'][key] = {
                    'from': val1,
                    'to': val2
                }
        
        return diff
    
    async def cleanup_old_versions(self, name: str, keep_versions: int = 10) -> int:
        """Clean up old configuration versions, keeping only the most recent ones"""
        if name not in self.config_history:
            return 0
        
        history = self.config_history[name]
        if len(history) <= keep_versions:
            return 0
        
        # Sort by creation date and keep only the most recent
        history.sort(key=lambda x: x.created_at, reverse=True)
        removed_count = len(history) - keep_versions
        self.config_history[name] = history[:keep_versions]
        
        # Save updated history
        await self._save_configurations()
        
        self.logger.info(f"Cleaned up {removed_count} old versions for {name}")
        return removed_count
    
    async def test_connection(self, name: str) -> Tuple[bool, Optional[str]]:
        """Test database connection"""
        config = self.get_database_config(name)
        if not config:
            return False, f"Configuration not found: {name}"
        
        try:
            if config.type == DatabaseType.POSTGRESQL:
                import asyncpg
                conn = await asyncpg.connect(
                    host=config.host,
                    port=config.port,
                    database=config.database,
                    user=config.credentials.username if config.credentials else None,
                    password=config.credentials.password if config.credentials else None,
                    timeout=config.timeout
                )
                await conn.execute("SELECT 1")
                await conn.close()
                
            elif config.type == DatabaseType.REDIS:
                import redis.asyncio as redis
                client = redis.Redis(
                    host=config.host,
                    port=config.port,
                    password=config.credentials.password if config.credentials else None,
                    socket_timeout=config.timeout
                )
                await client.ping()
                await client.close()
                
            elif config.type == DatabaseType.INFLUXDB:
                from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
                client = InfluxDBClientAsync(
                    url=config.get_connection_string(include_credentials=False),
                    token=config.credentials.token if config.credentials else None,
                    timeout=config.timeout * 1000
                )
                health = await client.health()
                await client.close()
                
                if health.status != 'pass':
                    return False, f"InfluxDB health check failed: {health.message}"
            
            else:
                return False, f"Connection testing not implemented for {config.type.value}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    async def validate_all_configs(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate all database configurations"""
        results = {}
        
        for name, config in self.configs.items():
            is_valid, errors = config.validate()
            results[name] = (is_valid, errors)
        
        return results
    
    def generate_documentation(self) -> str:
        """Generate configuration documentation"""
        doc = f"""# Database Configuration Documentation

Environment: {self.environment.value}
Generated: {datetime.utcnow().isoformat()}

## Configured Databases

"""
        
        for name, config in self.configs.items():
            doc += f"""### {name}

- **Type**: {config.type.value}
- **Host**: {config.host}
- **Port**: {config.port}
- **Database**: {config.database or 'N/A'}
- **SSL Enabled**: {config.ssl_enabled}
- **Connection String**: `{config.get_connection_string(include_credentials=False)}`
- **Pool Config**: Min: {config.pool_config.min_connections}, Max: {config.pool_config.max_connections}
- **Version**: {config.version}
- **Created**: {config.created_at.isoformat()}
- **Updated**: {config.updated_at.isoformat()}

"""
        
        doc += """## Environment Variables

The following environment variables are used for configuration:

"""
        
        for name, config in self.configs.items():
            prefix = name.upper().replace('-', '_')
            doc += f"""### {name}

- `{prefix}_USERNAME` - Database username
- `{prefix}_PASSWORD` - Database password
- `{prefix}_TOKEN` - Authentication token (for InfluxDB, etc.)

"""
        
        return doc
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Database Configuration Manager")
        
        # Deregister services from Consul
        if self.consul_client and self.service_discovery.enabled:
            for name in self.configs.keys():
                try:
                    service_id = f"{self.service_discovery.service_name_prefix}-{name}"
                    self.consul_client.agent.service.deregister(service_id)
                except Exception as e:
                    self.logger.warning(f"Failed to deregister service {name}: {e}")
        
        # Close Vault client
        if self.vault_client:
            self.vault_client.adapter.close()
        
        self.logger.info("Database Configuration Manager cleanup complete")


# Utility functions for easy configuration management
async def create_config_manager(environment: str = None) -> DatabaseConfigManager:
    """Create and initialize database configuration manager"""
    env = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
    
    manager = DatabaseConfigManager(environment=env)
    await manager.initialize()
    
    return manager


def get_database_url(db_name: str, environment: str = None) -> Optional[str]:
    """Get database connection URL for a specific database"""
    env = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
    
    # This is a synchronous helper - in practice you'd cache the manager
    # or use async context managers
    config_file = Path(f"./config/database/{env.value}.yaml")
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        db_config = config_data.get('databases', {}).get(db_name)
        if db_config:
            config = DatabaseConfig.from_dict(db_config)
            return config.get_connection_string()
    
    return None


# Configuration validation decorator
def validate_db_config(config_name: str):
    """Decorator to validate database configuration before function execution"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get config manager from context or create new one
            manager = kwargs.get('config_manager') or await create_config_manager()
            
            # Test connection
            success, error = await manager.test_connection(config_name)
            if not success:
                raise ConnectionError(f"Database {config_name} connection failed: {error}")
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator