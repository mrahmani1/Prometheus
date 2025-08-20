"""
Digital Twin Earth Configuration Management System
Advanced configuration management for Digital Twin Earth with dynamic updates and validation
"""

import json
import yaml
import asyncio
import logging
import warnings
from typing import Dict, List, Any, Optional, Union, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from pathlib import Path
import threading
import hashlib
import copy
import os
from contextlib import asynccontextmanager

# Configuration validation imports
try:
    import jsonschema
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn("jsonschema not available. Configuration validation will be limited.")

# Environment management imports
try:
    from pydantic import BaseModel, Field, validator
    from pydantic.dataclasses import dataclass as pydantic_dataclass
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    warnings.warn("pydantic not available. Advanced validation features will be limited.")

# Configuration encryption imports
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    warnings.warn("cryptography not available. Configuration encryption will be disabled.")

warnings.filterwarnings('ignore')

class ConfigurationFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"

class ConfigurationScope(Enum):
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    USER = "user"
    SESSION = "session"

class ValidationType(Enum):
    SCHEMA = "schema"
    CUSTOM = "custom"
    RANGE = "range"
    ENUM = "enum"

@dataclass
class ConfigurationMetadata:
    """Metadata for configuration entries"""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    encrypted: bool = False
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
class ConfigurationEntry:
    """Individual configuration entry with metadata and validation"""
    
    def __init__(self, key: str, value: Any, metadata: ConfigurationMetadata = None):
        self.key = key
        self.value = value
        self.metadata = metadata or ConfigurationMetadata()
        self._original_value = copy.deepcopy(value)
        self._change_listeners = []
        self._lock = threading.RLock()
    
    def get_value(self) -> Any:
        """Get the configuration value"""
        with self._lock:
            return self.value
    
    def set_value(self, value: Any, validate: bool = True) -> bool:
        """Set the configuration value with optional validation"""
        with self._lock:
            if validate and not self._validate_value(value):
                return False
            
            old_value = self.value
            self.value = value
            self.metadata.updated_at = datetime.utcnow()
            
            # Notify change listeners
            for listener in self._change_listeners:
                try:
                    listener(self.key, old_value, value)
                except Exception as e:
                    logging.error(f"Error in configuration change listener: {e}")
            
            return True
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Add a listener for value changes"""
        self._change_listeners.append(listener)
    
    def remove_change_listener(self, listener: Callable[[str, Any, Any], None]):
        """Remove a change listener"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
    
    def _validate_value(self, value: Any) -> bool:
        """Validate configuration value against rules"""
        try:
            rules = self.metadata.validation_rules
            
            # Type validation
            if 'type' in rules:
                expected_type = rules['type']
                if not isinstance(value, expected_type):
                    return False
            
            # Range validation for numeric values
            if 'min' in rules or 'max' in rules:
                if 'min' in rules and value < rules['min']:
                    return False
                if 'max' in rules and value > rules['max']:
                    return False
            
            # Enum validation
            if 'enum' in rules:
                if value not in rules['enum']:
                    return False
            
            # Custom validation function
            if 'validator' in rules:
                validator_func = rules['validator']
                if callable(validator_func) and not validator_func(value):
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating configuration value: {e}")
            return False
    
    def reset_to_original(self):
        """Reset to original value"""
        self.set_value(copy.deepcopy(self._original_value), validate=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'key': self.key,
            'value': self.value,
            'metadata': asdict(self.metadata)
        }

class DigitalTwinConfigurationManager:
    """Advanced configuration management system for Digital Twin Earth"""
    
    def __init__(self, config_dir: str = "config", encryption_key: str = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Configuration storage
        self.configurations = {}  # scope -> environment -> key -> ConfigurationEntry
        self.schemas = {}  # Configuration schemas for validation
        self.watchers = {}  # File system watchers for hot reloading
        
        # Security
        self.encryption_key = encryption_key
        self.cipher = None
        if ENCRYPTION_AVAILABLE and encryption_key:
            self.cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        
        # Event handling
        self.change_listeners = {}  # scope -> list of listeners
        self.validation_functions = {}  # key -> validation function
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Hot reloading
        self.hot_reload_enabled = True
        self.reload_tasks = {}
        
        # Version control
        self.configuration_history = {}  # key -> list of historical values
        self.max_history_entries = 100
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configurations
        asyncio.create_task(self._initialize_default_configurations())
    
    async def _initialize_default_configurations(self):
        """Initialize default Digital Twin configurations"""
        try:
            # Earth System Model configurations
            await self.set_configuration(
                "earth_system.atmosphere.resolution",
                {"horizontal": 50, "vertical": 100, "temporal": 3600},
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Atmosphere model resolution settings",
                    tags=["earth_system", "atmosphere", "resolution"],
                    validation_rules={
                        'type': dict,
                        'validator': lambda x: all(k in x for k in ['horizontal', 'vertical', 'temporal'])
                    }
                )
            )
            
            await self.set_configuration(
                "earth_system.hydrosphere.ocean_layers",
                50,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Number of ocean depth layers",
                    tags=["earth_system", "hydrosphere", "ocean"],
                    validation_rules={'type': int, 'min': 10, 'max': 200}
                )
            )
            
            await self.set_configuration(
                "earth_system.lithosphere.plate_resolution",
                1.0,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Tectonic plate model resolution in degrees",
                    tags=["earth_system", "lithosphere", "plates"],
                    validation_rules={'type': float, 'min': 0.1, 'max': 10.0}
                )
            )
            
            # AI System configurations
            await self.set_configuration(
                "ai_system.autonomous_discovery.hypothesis_generation_rate",
                10,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Number of hypotheses to generate per iteration",
                    tags=["ai_system", "discovery", "hypotheses"],
                    validation_rules={'type': int, 'min': 1, 'max': 100}
                )
            )
            
            await self.set_configuration(
                "ai_system.model_training.batch_size",
                32,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Training batch size for AI models",
                    tags=["ai_system", "training", "batch"],
                    validation_rules={'type': int, 'min': 1, 'max': 1024}
                )
            )
            
            # Performance configurations
            await self.set_configuration(
                "performance.optimization.enable_gpu_acceleration",
                True,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Enable GPU acceleration for computations",
                    tags=["performance", "gpu", "acceleration"],
                    validation_rules={'type': bool}
                )
            )
            
            await self.set_configuration(
                "performance.caching.memory_limit_gb",
                8.0,
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Memory limit for caching in GB",
                    tags=["performance", "cache", "memory"],
                    validation_rules={'type': float, 'min': 1.0, 'max': 128.0}
                )
            )
            
            # Visualization configurations
            await self.set_configuration(
                "visualization.high_resolution.max_resolution",
                [4096, 4096],
                ConfigurationScope.GLOBAL,
                metadata=ConfigurationMetadata(
                    description="Maximum visualization resolution [width, height]",
                    tags=["visualization", "resolution", "graphics"],
                    validation_rules={
                        'type': list,
                        'validator': lambda x: len(x) == 2 and all(isinstance(i, int) and 512 <= i <= 8192 for i in x)
                    }
                )
            )
            
            self.logger.info("Default Digital Twin configurations initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing default configurations: {e}")
    
    async def get_configuration(self, key: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL, 
                               environment: str = "default", default: Any = None) -> Any:
        """Get configuration value"""
        try:
            with self._lock:
                if scope not in self.configurations:
                    return default
                
                if environment not in self.configurations[scope]:
                    return default
                
                if key not in self.configurations[scope][environment]:
                    return default
                
                entry = self.configurations[scope][environment][key]
                return entry.get_value()
                
        except Exception as e:
            self.logger.error(f"Error getting configuration {key}: {e}")
            return default
    
    async def set_configuration(self, key: str, value: Any, scope: ConfigurationScope = ConfigurationScope.GLOBAL,
                               environment: str = "default", metadata: ConfigurationMetadata = None,
                               persist: bool = True) -> bool:
        """Set configuration value"""
        try:
            with self._lock:
                # Initialize scope and environment if needed
                if scope not in self.configurations:
                    self.configurations[scope] = {}
                
                if environment not in self.configurations[scope]:
                    self.configurations[scope][environment] = {}
                
                # Create or update configuration entry
                if key in self.configurations[scope][environment]:
                    entry = self.configurations[scope][environment][key]
                    old_value = entry.get_value()
                    success = entry.set_value(value)
                    
                    # Add to history
                    self._add_to_history(key, old_value)
                else:
                    entry = ConfigurationEntry(key, value, metadata)
                    self.configurations[scope][environment][key] = entry
                    success = True
                
                # Persist to file if requested
                if persist and success:
                    await self._persist_configuration(key, scope, environment)
                
                # Notify listeners
                if success:
                    await self._notify_change_listeners(scope, key, value)
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error setting configuration {key}: {e}")
            return False
    
    async def delete_configuration(self, key: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL,
                                  environment: str = "default") -> bool:
        """Delete configuration entry"""
        try:
            with self._lock:
                if (scope in self.configurations and 
                    environment in self.configurations[scope] and 
                    key in self.configurations[scope][environment]):
                    
                    old_value = self.configurations[scope][environment][key].get_value()
                    del self.configurations[scope][environment][key]
                    
                    # Add to history
                    self._add_to_history(key, old_value)
                    
                    # Remove persisted file
                    await self._remove_persisted_configuration(key, scope, environment)
                    
                    # Notify listeners
                    await self._notify_change_listeners(scope, key, None)
                    
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting configuration {key}: {e}")
            return False
    
    async def get_all_configurations(self, scope: ConfigurationScope = None, 
                                    environment: str = None) -> Dict[str, Any]:
        """Get all configurations for scope/environment"""
        try:
            result = {}
            
            with self._lock:
                scopes_to_check = [scope] if scope else self.configurations.keys()
                
                for current_scope in scopes_to_check:
                    if current_scope not in self.configurations:
                        continue
                    
                    environments_to_check = [environment] if environment else self.configurations[current_scope].keys()
                    
                    for current_env in environments_to_check:
                        if current_env not in self.configurations[current_scope]:
                            continue
                        
                        scope_key = f"{current_scope.value}" if scope is None else ""
                        env_key = f"{current_env}" if environment is None else ""
                        prefix = f"{scope_key}.{env_key}." if scope_key or env_key else ""
                        
                        for key, entry in self.configurations[current_scope][current_env].items():
                            result[f"{prefix}{key}"] = entry.get_value()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting all configurations: {e}")
            return {}
    
    async def load_configuration_file(self, file_path: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL,
                                     environment: str = "default", format: ConfigurationFormat = None) -> bool:
        """Load configurations from file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"Configuration file not found: {file_path}")
                return False
            
            # Determine format from extension if not specified
            if format is None:
                ext = file_path.suffix.lower()
                if ext == '.json':
                    format = ConfigurationFormat.JSON
                elif ext in ['.yml', '.yaml']:
                    format = ConfigurationFormat.YAML
                elif ext == '.toml':
                    format = ConfigurationFormat.TOML
                elif ext == '.ini':
                    format = ConfigurationFormat.INI
                else:
                    format = ConfigurationFormat.JSON
            
            # Read and parse file
            content = file_path.read_text()
            
            if format == ConfigurationFormat.JSON:
                data = json.loads(content)
            elif format == ConfigurationFormat.YAML:
                import yaml
                data = yaml.safe_load(content)
            else:
                self.logger.error(f"Unsupported configuration format: {format}")
                return False
            
            # Load configurations
            return await self._load_configuration_data(data, scope, environment)
            
        except Exception as e:
            self.logger.error(f"Error loading configuration file {file_path}: {e}")
            return False
    
    async def save_configuration_file(self, file_path: str, scope: ConfigurationScope = ConfigurationScope.GLOBAL,
                                     environment: str = "default", format: ConfigurationFormat = None) -> bool:
        """Save configurations to file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine format from extension if not specified
            if format is None:
                ext = file_path.suffix.lower()
                if ext == '.json':
                    format = ConfigurationFormat.JSON
                elif ext in ['.yml', '.yaml']:
                    format = ConfigurationFormat.YAML
                else:
                    format = ConfigurationFormat.JSON
            
            # Get configurations for scope/environment
            configs = await self.get_all_configurations(scope, environment)
            
            # Convert to proper format and save
            if format == ConfigurationFormat.JSON:
                content = json.dumps(configs, indent=2, default=str)
            elif format == ConfigurationFormat.YAML:
                import yaml
                content = yaml.dump(configs, default_flow_style=False)
            else:
                self.logger.error(f"Unsupported configuration format: {format}")
                return False
            
            file_path.write_text(content)
            self.logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration file {file_path}: {e}")
            return False
    
    async def _load_configuration_data(self, data: Dict[str, Any], scope: ConfigurationScope, environment: str) -> bool:
        """Load configuration data from dictionary"""
        try:
            for key, value in data.items():
                if isinstance(value, dict) and 'value' in value and 'metadata' in value:
                    # Structured format with metadata
                    metadata = ConfigurationMetadata(**value['metadata'])
                    await self.set_configuration(key, value['value'], scope, environment, metadata)
                else:
                    # Simple key-value format
                    await self.set_configuration(key, value, scope, environment)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration data: {e}")
            return False
    
    async def _persist_configuration(self, key: str, scope: ConfigurationScope, environment: str):
        """Persist single configuration to file"""
        try:
            # Create file path
            file_path = self.config_dir / f"{scope.value}_{environment}_{key.replace('.', '_')}.json"
            
            entry = self.configurations[scope][environment][key]
            data = entry.to_dict()
            
            # Encrypt if needed
            if entry.metadata.encrypted and self.cipher:
                data['value'] = self.cipher.encrypt(json.dumps(data['value']).encode()).decode()
                data['encrypted'] = True
            
            file_path.write_text(json.dumps(data, indent=2, default=str))
            
        except Exception as e:
            self.logger.error(f"Error persisting configuration {key}: {e}")
    
    async def _remove_persisted_configuration(self, key: str, scope: ConfigurationScope, environment: str):
        """Remove persisted configuration file"""
        try:
            file_path = self.config_dir / f"{scope.value}_{environment}_{key.replace('.', '_')}.json"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            self.logger.error(f"Error removing persisted configuration {key}: {e}")
    
    def _add_to_history(self, key: str, value: Any):
        """Add configuration value to history"""
        try:
            if key not in self.configuration_history:
                self.configuration_history[key] = []
            
            self.configuration_history[key].append({
                'value': copy.deepcopy(value),
                'timestamp': datetime.utcnow().isoformat()
            })
            
            # Limit history size
            if len(self.configuration_history[key]) > self.max_history_entries:
                self.configuration_history[key].pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding to configuration history: {e}")
    
    async def get_configuration_history(self, key: str) -> List[Dict[str, Any]]:
        """Get configuration change history"""
        return self.configuration_history.get(key, [])
    
    async def rollback_configuration(self, key: str, steps: int = 1,
                                    scope: ConfigurationScope = ConfigurationScope.GLOBAL,
                                    environment: str = "default") -> bool:
        """Rollback configuration to previous value"""
        try:
            history = await self.get_configuration_history(key)
            
            if len(history) < steps + 1:
                self.logger.error(f"Not enough history to rollback {steps} steps for {key}")
                return False
            
            # Get value from history (excluding current value)
            rollback_entry = history[-(steps + 1)]
            rollback_value = rollback_entry['value']
            
            # Set the rolled back value
            return await self.set_configuration(key, rollback_value, scope, environment, persist=True)
            
        except Exception as e:
            self.logger.error(f"Error rolling back configuration {key}: {e}")
            return False
    
    def add_change_listener(self, scope: ConfigurationScope, listener: Callable[[str, Any], None]):
        """Add configuration change listener"""
        if scope not in self.change_listeners:
            self.change_listeners[scope] = []
        
        self.change_listeners[scope].append(listener)
    
    def remove_change_listener(self, scope: ConfigurationScope, listener: Callable[[str, Any], None]):
        """Remove configuration change listener"""
        if scope in self.change_listeners and listener in self.change_listeners[scope]:
            self.change_listeners[scope].remove(listener)
    
    async def _notify_change_listeners(self, scope: ConfigurationScope, key: str, value: Any):
        """Notify configuration change listeners"""
        try:
            if scope in self.change_listeners:
                for listener in self.change_listeners[scope]:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(key, value)
                        else:
                            listener(key, value)
                    except Exception as e:
                        self.logger.error(f"Error in configuration change listener: {e}")
        except Exception as e:
            self.logger.error(f"Error notifying change listeners: {e}")
    
    async def validate_all_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations and return errors"""
        errors = {}
        
        try:
            with self._lock:
                for scope in self.configurations:
                    for environment in self.configurations[scope]:
                        for key, entry in self.configurations[scope][environment].items():
                            if not entry._validate_value(entry.get_value()):
                                if key not in errors:
                                    errors[key] = []
                                errors[key].append(f"Validation failed for {scope.value}.{environment}.{key}")
        except Exception as e:
            self.logger.error(f"Error validating configurations: {e}")
            errors['validation_error'] = [str(e)]
        
        return errors
    
    @asynccontextmanager
    async def configuration_transaction(self):
        """Context manager for atomic configuration updates"""
        backup = copy.deepcopy(self.configurations)
        try:
            yield
        except Exception as e:
            # Rollback on error
            self.configurations = backup
            raise e
    
    async def export_configurations(self, file_path: str, include_metadata: bool = True) -> bool:
        """Export all configurations to a single file"""
        try:
            export_data = {}
            
            with self._lock:
                for scope in self.configurations:
                    scope_data = {}
                    for environment in self.configurations[scope]:
                        env_data = {}
                        for key, entry in self.configurations[scope][environment].items():
                            if include_metadata:
                                env_data[key] = entry.to_dict()
                            else:
                                env_data[key] = entry.get_value()
                        
                        if env_data:
                            scope_data[environment] = env_data
                    
                    if scope_data:
                        export_data[scope.value] = scope_data
            
            # Save to file
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(json.dumps(export_data, indent=2, default=str))
            
            self.logger.info(f"Configurations exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configurations: {e}")
            return False
    
    async def import_configurations(self, file_path: str, merge: bool = True) -> bool:
        """Import configurations from file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"Import file not found: {file_path}")
                return False
            
            import_data = json.loads(file_path.read_text())
            
            if not merge:
                # Clear existing configurations
                with self._lock:
                    self.configurations.clear()
            
            # Load imported data
            for scope_name, scope_data in import_data.items():
                try:
                    scope = ConfigurationScope(scope_name)
                except ValueError:
                    self.logger.warning(f"Unknown configuration scope: {scope_name}")
                    continue
                
                for environment, env_data in scope_data.items():
                    for key, value in env_data.items():
                        if isinstance(value, dict) and 'value' in value:
                            # Structured format
                            metadata = ConfigurationMetadata(**value.get('metadata', {}))
                            await self.set_configuration(key, value['value'], scope, environment, metadata)
                        else:
                            # Simple format
                            await self.set_configuration(key, value, scope, environment)
            
            self.logger.info(f"Configurations imported from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing configurations: {e}")
            return False
    
    async def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        try:
            summary = {
                'total_configurations': 0,
                'scopes': {},
                'environments': set(),
                'encrypted_count': 0,
                'last_updated': None
            }
            
            latest_update = None
            
            with self._lock:
                for scope in self.configurations:
                    scope_count = 0
                    for environment in self.configurations[scope]:
                        summary['environments'].add(environment)
                        for key, entry in self.configurations[scope][environment].items():
                            summary['total_configurations'] += 1
                            scope_count += 1
                            
                            if entry.metadata.encrypted:
                                summary['encrypted_count'] += 1
                            
                            if latest_update is None or entry.metadata.updated_at > latest_update:
                                latest_update = entry.metadata.updated_at
                    
                    summary['scopes'][scope.value] = scope_count
            
            summary['environments'] = list(summary['environments'])
            summary['last_updated'] = latest_update.isoformat() if latest_update else None
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating configuration summary: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources and save configurations"""
        try:
            # Cancel any running reload tasks
            for task in self.reload_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Save all configurations
            await self.export_configurations(str(self.config_dir / "backup_configurations.json"))
            
            self.logger.info("Configuration manager cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during configuration manager cleanup: {e}")

# Example usage and testing
async def example_digital_twin_configuration():
    """Example of Digital Twin configuration management"""
    
    print("=== Digital Twin Configuration Management Example ===")
    
    # Initialize configuration manager
    config_manager = DigitalTwinConfigurationManager()
    
    try:
        # Wait for initialization
        await asyncio.sleep(1)
        
        print("\n1. Getting default configurations:")
        default_configs = await config_manager.get_all_configurations()
        for key, value in list(default_configs.items())[:5]:  # Show first 5
            print(f"  {key}: {value}")
        
        print(f"\nTotal configurations loaded: {len(default_configs)}")
        
        # Example: Set custom configuration
        print("\n2. Setting custom configuration:")
        success = await config_manager.set_configuration(
            "custom.experiment.parameter",
            {"learning_rate": 0.001, "batch_size": 64},
            ConfigurationScope.USER,
            environment="development",
            metadata=ConfigurationMetadata(
                description="Custom ML experiment parameters",
                tags=["experiment", "ml", "custom"]
            )
        )
        print(f"Custom configuration set: {success}")
        
        # Example: Get specific configuration
        print("\n3. Getting specific configurations:")
        gpu_config = await config_manager.get_configuration("performance.optimization.enable_gpu_acceleration")
        print(f"GPU acceleration enabled: {gpu_config}")
        
        custom_config = await config_manager.get_configuration(
            "custom.experiment.parameter", 
            ConfigurationScope.USER, 
            "development"
        )
        print(f"Custom experiment config: {custom_config}")
        
        # Example: Configuration history
        print("\n4. Configuration history:")
        await config_manager.set_configuration("test.value", 100)
        await config_manager.set_configuration("test.value", 200)
        await config_manager.set_configuration("test.value", 300)
        
        history = await config_manager.get_configuration_history("test.value")
        print(f"History entries: {len(history)}")
        
        # Rollback test
        rollback_success = await config_manager.rollback_configuration("test.value", steps=1)
        current_value = await config_manager.get_configuration("test.value")
        print(f"Rollback successful: {rollback_success}, Current value: {current_value}")
        
        # Example: Export/Import
        print("\n5. Export/Import test:")
        export_path = "test_config_export.json"
        export_success = await config_manager.export_configurations(export_path)
        print(f"Export successful: {export_success}")
        
        # Example: Configuration summary
        print("\n6. Configuration summary:")
        summary = await config_manager.get_configuration_summary()
        print(f"Total configurations: {summary['total_configurations']}")
        print(f"Scopes: {list(summary['scopes'].keys())}")
        print(f"Environments: {summary['environments']}")
        
        # Example: Validation
        print("\n7. Configuration validation:")
        errors = await config_manager.validate_all_configurations()
        if errors:
            print(f"Validation errors found: {len(errors)}")
        else:
            print("All configurations are valid")
        
        print("\nConfiguration management example completed successfully!")
        
    finally:
        await config_manager.cleanup()

if __name__ == "__main__":
    print("=== Digital Twin Configuration Management System ===")
    print(f"Features available:")
    print(f"  JSON Schema Validation: {JSONSCHEMA_AVAILABLE}")
    print(f"  Pydantic Validation: {PYDANTIC_AVAILABLE}")
    print(f"  Configuration Encryption: {ENCRYPTION_AVAILABLE}")
    
    # Run example
    asyncio.run(example_digital_twin_configuration())