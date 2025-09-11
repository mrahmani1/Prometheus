"""
Core Module Integration Framework
Unified module loader and dependency management system for 100+ geological modules
"""

import asyncio
import importlib
import inspect
import logging
import sys
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import threading
import weakref

# Import existing modules
from backend.modules.advanced_error_handling_validation import AdvancedErrorHandler, ErrorSeverity, ErrorCategory, RecoveryStrategy
from backend.modules.security_framework import SecurityFramework, SecurityLevel, ThreatLevel
from backend.modules.advanced_database_integration import AdvancedDatabaseIntegrator


class ModuleStatus(Enum):
    """Module lifecycle status"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"


class ModulePriority(Enum):
    """Module loading priority levels"""
    CRITICAL = 1    # Core infrastructure modules
    HIGH = 2        # Essential geological modules
    MEDIUM = 3      # Standard processing modules
    LOW = 4         # Optional enhancement modules


@dataclass
class ModuleDependency:
    """Module dependency specification"""
    name: str
    version: Optional[str] = None
    optional: bool = False
    initialization_order: int = 0


@dataclass
class ModuleMetadata:
    """Comprehensive module metadata"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[ModuleDependency] = field(default_factory=list)
    priority: ModulePriority = ModulePriority.MEDIUM
    geological_domain: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    configuration_schema: Optional[Dict] = None


@dataclass
class ModuleInstance:
    """Module instance tracking"""
    metadata: ModuleMetadata
    module_object: Any
    status: ModuleStatus
    load_time: datetime
    initialization_time: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    health_status: str = "unknown"
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class GeologicalModuleInterface(ABC):
    """Base interface for all geological modules"""
    
    @abstractmethod
    def get_metadata(self) -> ModuleMetadata:
        """Return module metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the module with configuration"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Clean up module resources"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of module capabilities"""
        return []
    
    def get_geological_domain(self) -> Optional[str]:
        """Return geological domain specialization"""
        return None


class ModuleRegistry:
    """Central registry for all geological modules"""
    
    def __init__(self):
        self.modules: Dict[str, ModuleInstance] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependencies: Dict[str, Set[str]] = {}
        self.loading_order: List[str] = []
        self.logger = logging.getLogger(__name__ + ".ModuleRegistry")
        self._lock = threading.RLock()
    
    def register_module(self, name: str, instance: ModuleInstance) -> bool:
        """Register a module instance"""
        with self._lock:
            if name in self.modules:
                self.logger.warning(f"Module {name} already registered, updating")
            
            self.modules[name] = instance
            self._update_dependency_graph(name, instance.metadata.dependencies)
            self.logger.info(f"Registered module: {name}")
            return True
    
    def get_module(self, name: str) -> Optional[ModuleInstance]:
        """Get module instance by name"""
        with self._lock:
            return self.modules.get(name)
    
    def get_modules_by_domain(self, domain: str) -> List[ModuleInstance]:
        """Get all modules for a geological domain"""
        with self._lock:
            return [
                instance for instance in self.modules.values()
                if instance.metadata.geological_domain == domain
            ]
    
    def get_modules_by_capability(self, capability: str) -> List[ModuleInstance]:
        """Get all modules with a specific capability"""
        with self._lock:
            return [
                instance for instance in self.modules.values()
                if capability in instance.metadata.capabilities
            ]
    
    def _update_dependency_graph(self, module_name: str, dependencies: List[ModuleDependency]):
        """Update dependency graph for module"""
        self.dependency_graph[module_name] = set()
        
        for dep in dependencies:
            self.dependency_graph[module_name].add(dep.name)
            
            # Update reverse dependencies
            if dep.name not in self.reverse_dependencies:
                self.reverse_dependencies[dep.name] = set()
            self.reverse_dependencies[dep.name].add(module_name)
    
    def get_loading_order(self) -> List[str]:
        """Calculate optimal module loading order based on dependencies"""
        if self.loading_order:
            return self.loading_order
        
        # Topological sort with priority consideration
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(module_name: str):
            if module_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {module_name}")
            if module_name in visited:
                return
            
            temp_visited.add(module_name)
            
            # Visit dependencies first
            for dep_name in self.dependency_graph.get(module_name, set()):
                if dep_name in self.modules:
                    visit(dep_name)
            
            temp_visited.remove(module_name)
            visited.add(module_name)
            order.append(module_name)
        
        # Sort modules by priority first
        modules_by_priority = sorted(
            self.modules.keys(),
            key=lambda name: self.modules[name].metadata.priority.value
        )
        
        for module_name in modules_by_priority:
            if module_name not in visited:
                visit(module_name)
        
        self.loading_order = order
        return order


class CoreModuleIntegrationFramework:
    """
    Core Module Integration Framework
    Manages loading, initialization, and coordination of 100+ geological modules
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".CoreModuleIntegrationFramework")
        
        # Core components
        self.error_handler = AdvancedErrorHandler(self.config.get('error_handling', {}))
        self.security_framework = SecurityFramework(self.config.get('security', {}))
        self.database_integrator = AdvancedDatabaseIntegrator(self.config.get('database', {}))
        
        # Module management
        self.registry = ModuleRegistry()
        self.module_path = Path(self.config.get('module_path', 'backend/modules'))
        
        # Communication system
        self.event_bus = asyncio.Queue()
        self.message_handlers: Dict[str, List[Callable]] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'total_modules': 0,
            'loaded_modules': 0,
            'failed_modules': 0,
            'average_load_time': 0.0,
            'total_memory_usage': 0.0
        }
        
        # Health monitoring
        self.health_check_interval = self.config.get('health_check_interval', 300)  # 5 minutes
        self.health_check_task: Optional[asyncio.Task] = None
        
        self.logger.info("Core Module Integration Framework initialized")
    
    async def initialize(self) -> bool:
        """Initialize the core framework"""
        try:
            self.logger.info("Initializing Core Module Integration Framework")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Discover and load modules
            await self.discover_modules()
            
            # Load modules in dependency order
            await self.load_all_modules()
            
            # Start health monitoring
            await self._start_health_monitoring()
            
            # Start event processing
            await self._start_event_processing()
            
            self.logger.info("Core Module Integration Framework initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Framework initialization failed: {e}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core framework components"""
        # Initialize error handler
        if hasattr(self.error_handler, 'initialize'):
            await self.error_handler.initialize()
        
        # Initialize security framework
        if hasattr(self.security_framework, 'initialize'):
            await self.security_framework.initialize()
        
        # Initialize database integrator
        if hasattr(self.database_integrator, 'initialize'):
            await self.database_integrator.initialize()
    
    async def discover_modules(self) -> List[str]:
        """Discover all available modules in the module directory"""
        discovered_modules = []
        
        try:
            if not self.module_path.exists():
                self.logger.warning(f"Module path does not exist: {self.module_path}")
                return discovered_modules
            
            for module_file in self.module_path.glob("*.py"):
                if module_file.name.startswith("__"):
                    continue
                
                module_name = module_file.stem
                try:
                    # Import module to check if it's a valid geological module
                    module_spec = importlib.util.spec_from_file_location(
                        f"backend.modules.{module_name}", module_file
                    )
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    
                    # Check if module has geological module classes
                    geological_classes = self._find_geological_classes(module)
                    
                    if geological_classes:
                        discovered_modules.append(module_name)
                        self.logger.debug(f"Discovered geological module: {module_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to discover module {module_name}: {e}")
            
            self.logger.info(f"Discovered {len(discovered_modules)} geological modules")
            return discovered_modules
            
        except Exception as e:
            self.logger.error(f"Module discovery failed: {e}")
            return discovered_modules
    
    def _find_geological_classes(self, module) -> List[Type]:
        """Find geological module classes in a module"""
        geological_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if class implements GeologicalModuleInterface or has geological methods
            if (issubclass(obj, GeologicalModuleInterface) or 
                self._has_geological_methods(obj)):
                geological_classes.append(obj)
        
        return geological_classes
    
    def _has_geological_methods(self, cls: Type) -> bool:
        """Check if class has geological processing methods"""
        geological_indicators = [
            'geological', 'seismic', 'volcanic', 'tectonic', 'geophysical',
            'process', 'analyze', 'model', 'simulate', 'predict'
        ]
        
        methods = [method for method in dir(cls) if not method.startswith('_')]
        
        for method in methods:
            method_name = method.lower()
            if any(indicator in method_name for indicator in geological_indicators):
                return True
        
        return False
    
    @AdvancedErrorHandler().error_handler(
        severity=ErrorSeverity.HIGH,
        category=ErrorCategory.INTEGRATION_FAILURE,
        recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION
    )
    async def load_module(self, module_name: str) -> bool:
        """Load a specific geological module"""
        try:
            self.logger.info(f"Loading module: {module_name}")
            
            # Import the module
            module_path = f"backend.modules.{module_name}"
            module = importlib.import_module(module_path)
            
            # Find geological classes in the module
            geological_classes = self._find_geological_classes(module)
            
            if not geological_classes:
                self.logger.warning(f"No geological classes found in module: {module_name}")
                return False
            
            # Create module instances
            for cls in geological_classes:
                try:
                    instance = cls()
                    
                    # Get metadata
                    if hasattr(instance, 'get_metadata'):
                        metadata = instance.get_metadata()
                    else:
                        metadata = self._create_default_metadata(module_name, cls)
                    
                    # Create module instance record
                    module_instance = ModuleInstance(
                        metadata=metadata,
                        module_object=instance,
                        status=ModuleStatus.LOADED,
                        load_time=datetime.now()
                    )
                    
                    # Register module
                    self.registry.register_module(f"{module_name}.{cls.__name__}", module_instance)
                    
                    self.logger.info(f"Successfully loaded module class: {module_name}.{cls.__name__}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to instantiate class {cls.__name__} in module {module_name}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load module {module_name}: {e}")
            return False
    
    def _create_default_metadata(self, module_name: str, cls: Type) -> ModuleMetadata:
        """Create default metadata for modules without explicit metadata"""
        return ModuleMetadata(
            name=f"{module_name}.{cls.__name__}",
            version="1.0.0",
            description=f"Geological module: {cls.__name__}",
            author="Prometheus AI Platform",
            geological_domain=self._infer_geological_domain(module_name),
            capabilities=self._infer_capabilities(cls)
        )
    
    def _infer_geological_domain(self, module_name: str) -> Optional[str]:
        """Infer geological domain from module name"""
        domain_mapping = {
            'seismic': 'seismology',
            'volcanic': 'volcanology',
            'tectonic': 'tectonics',
            'magma': 'magmatism',
            'hydrothermal': 'hydrothermal_systems',
            'mineral': 'mineralogy',
            'geochemical': 'geochemistry',
            'stress': 'structural_geology',
            'mantle': 'geodynamics',
            'plate': 'plate_tectonics'
        }
        
        module_lower = module_name.lower()
        for keyword, domain in domain_mapping.items():
            if keyword in module_lower:
                return domain
        
        return 'general_geology'
    
    def _infer_capabilities(self, cls: Type) -> List[str]:
        """Infer capabilities from class methods"""
        capabilities = []
        methods = [method for method in dir(cls) if not method.startswith('_')]
        
        capability_mapping = {
            'analyze': 'analysis',
            'process': 'data_processing',
            'model': 'modeling',
            'simulate': 'simulation',
            'predict': 'prediction',
            'classify': 'classification',
            'detect': 'detection',
            'monitor': 'monitoring',
            'visualize': 'visualization'
        }
        
        for method in methods:
            method_lower = method.lower()
            for keyword, capability in capability_mapping.items():
                if keyword in method_lower and capability not in capabilities:
                    capabilities.append(capability)
        
        return capabilities
    
    async def load_all_modules(self) -> Dict[str, bool]:
        """Load all discovered modules in dependency order"""
        results = {}
        
        try:
            # Get discovered modules
            discovered_modules = await self.discover_modules()
            
            # Load modules in batches by priority
            priority_batches = {
                ModulePriority.CRITICAL: [],
                ModulePriority.HIGH: [],
                ModulePriority.MEDIUM: [],
                ModulePriority.LOW: []
            }
            
            # First pass: load all modules to determine priorities
            for module_name in discovered_modules:
                success = await self.load_module(module_name)
                results[module_name] = success
            
            # Initialize modules in dependency order
            loading_order = self.registry.get_loading_order()
            
            for module_key in loading_order:
                module_instance = self.registry.get_module(module_key)
                if module_instance and module_instance.status == ModuleStatus.LOADED:
                    await self._initialize_module(module_instance)
            
            # Update performance metrics
            self.performance_metrics['total_modules'] = len(discovered_modules)
            self.performance_metrics['loaded_modules'] = sum(1 for r in results.values() if r)
            self.performance_metrics['failed_modules'] = sum(1 for r in results.values() if not r)
            
            self.logger.info(f"Module loading complete. Success: {self.performance_metrics['loaded_modules']}, "
                           f"Failed: {self.performance_metrics['failed_modules']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to load all modules: {e}")
            return results
    
    async def _initialize_module(self, module_instance: ModuleInstance) -> bool:
        """Initialize a loaded module"""
        try:
            module_instance.status = ModuleStatus.INITIALIZING
            
            if hasattr(module_instance.module_object, 'initialize'):
                config = self.config.get('modules', {}).get(module_instance.metadata.name, {})
                success = await module_instance.module_object.initialize(config)
                
                if success:
                    module_instance.status = ModuleStatus.READY
                    module_instance.initialization_time = datetime.now()
                    self.logger.info(f"Module initialized: {module_instance.metadata.name}")
                    return True
                else:
                    module_instance.status = ModuleStatus.ERROR
                    module_instance.last_error = "Initialization failed"
                    return False
            else:
                # Module doesn't have initialize method, consider it ready
                module_instance.status = ModuleStatus.READY
                module_instance.initialization_time = datetime.now()
                return True
                
        except Exception as e:
            module_instance.status = ModuleStatus.ERROR
            module_instance.error_count += 1
            module_instance.last_error = str(e)
            self.logger.error(f"Failed to initialize module {module_instance.metadata.name}: {e}")
            return False
    
    async def _start_health_monitoring(self):
        """Start periodic health monitoring of all modules"""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.health_check_interval)
                    await self._perform_health_checks()
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
        
        self.health_check_task = asyncio.create_task(health_monitor())
    
    async def _perform_health_checks(self):
        """Perform health checks on all modules"""
        for module_name, module_instance in self.registry.modules.items():
            try:
                if hasattr(module_instance.module_object, 'health_check'):
                    health_status = await module_instance.module_object.health_check()
                    module_instance.health_status = health_status.get('status', 'unknown')
                    
                    if health_status.get('status') == 'error':
                        self.logger.warning(f"Module health check failed: {module_name}")
                        
            except Exception as e:
                module_instance.health_status = 'error'
                module_instance.error_count += 1
                module_instance.last_error = str(e)
                self.logger.error(f"Health check failed for module {module_name}: {e}")
    
    async def _start_event_processing(self):
        """Start event processing system"""
        async def event_processor():
            while True:
                try:
                    event = await self.event_bus.get()
                    await self._process_event(event)
                except Exception as e:
                    self.logger.error(f"Event processing error: {e}")
        
        asyncio.create_task(event_processor())
    
    async def _process_event(self, event: Dict[str, Any]):
        """Process inter-module communication events"""
        event_type = event.get('type')
        handlers = self.message_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"Event handler error: {e}")
    
    def register_message_handler(self, event_type: str, handler: Callable):
        """Register a message handler for inter-module communication"""
        if event_type not in self.message_handlers:
            self.message_handlers[event_type] = []
        self.message_handlers[event_type].append(handler)
    
    async def send_message(self, event_type: str, data: Dict[str, Any]):
        """Send a message through the event bus"""
        event = {
            'type': event_type,
            'timestamp': datetime.now(),
            'data': data,
            'id': str(uuid.uuid4())
        }
        await self.event_bus.put(event)
    
    def get_module_by_name(self, name: str) -> Optional[Any]:
        """Get module instance by name"""
        module_instance = self.registry.get_module(name)
        return module_instance.module_object if module_instance else None
    
    def get_modules_by_capability(self, capability: str) -> List[Any]:
        """Get all modules with a specific capability"""
        module_instances = self.registry.get_modules_by_capability(capability)
        return [instance.module_object for instance in module_instances]
    
    def get_framework_status(self) -> Dict[str, Any]:
        """Get comprehensive framework status"""
        return {
            'total_modules': len(self.registry.modules),
            'ready_modules': len([m for m in self.registry.modules.values() if m.status == ModuleStatus.READY]),
            'error_modules': len([m for m in self.registry.modules.values() if m.status == ModuleStatus.ERROR]),
            'performance_metrics': self.performance_metrics,
            'health_status': 'healthy' if self.health_check_task and not self.health_check_task.done() else 'degraded'
        }
    
    async def shutdown(self):
        """Gracefully shutdown the framework"""
        self.logger.info("Shutting down Core Module Integration Framework")
        
        # Cancel health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
        
        # Cleanup all modules
        for module_name, module_instance in self.registry.modules.items():
            try:
                if hasattr(module_instance.module_object, 'cleanup'):
                    await module_instance.module_object.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up module {module_name}: {e}")
        
        self.logger.info("Framework shutdown complete")


# Global framework instance
_framework_instance: Optional[CoreModuleIntegrationFramework] = None


def get_framework() -> CoreModuleIntegrationFramework:
    """Get the global framework instance"""
    global _framework_instance
    if _framework_instance is None:
        _framework_instance = CoreModuleIntegrationFramework()
    return _framework_instance


async def initialize_framework(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize the global framework instance"""
    global _framework_instance
    _framework_instance = CoreModuleIntegrationFramework(config)
    return await _framework_instance.initialize()