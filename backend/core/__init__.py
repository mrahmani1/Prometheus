"""
Core Integration Framework for Prometheus AI Platform
Unified module loader, security, and database integration
"""

from .module_loader import (
    CoreModuleIntegrationFramework,
    GeologicalModuleInterface,
    ModuleRegistry,
    ModuleMetadata,
    ModulePriority,
    ModuleStatus,
    get_framework,
    initialize_framework
)

from .security_integration import (
    EnhancedSecurityFramework,
    UserProfile,
    SecurityPolicy,
    AuthenticationMethod,
    AccessLevel,
    ThreatDetectionSystem
)

from .database_integration import (
    EnhancedDatabaseIntegrator,
    DatabaseMetrics,
    QueryResult,
    get_database_integrator,
    initialize_database_integrator
)

__all__ = [
    # Module Integration
    'CoreModuleIntegrationFramework',
    'GeologicalModuleInterface',
    'ModuleRegistry',
    'ModuleMetadata',
    'ModulePriority',
    'ModuleStatus',
    'get_framework',
    'initialize_framework',
    
    # Security Integration
    'EnhancedSecurityFramework',
    'UserProfile',
    'SecurityPolicy',
    'AuthenticationMethod',
    'AccessLevel',
    'ThreatDetectionSystem',
    
    # Database Integration
    'EnhancedDatabaseIntegrator',
    'DatabaseMetrics',
    'QueryResult',
    'get_database_integrator',
    'initialize_database_integrator'
]