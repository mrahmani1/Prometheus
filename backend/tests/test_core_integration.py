"""
Integration Tests for Core Module Integration Framework
Tests module loading, dependency management, and inter-module communication
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import the framework components
from backend.core.module_loader import (
    CoreModuleIntegrationFramework,
    GeologicalModuleInterface,
    ModuleMetadata,
    ModulePriority,
    ModuleStatus,
    ModuleDependency,
    get_framework,
    initialize_framework
)


class MockGeologicalModule(GeologicalModuleInterface):
    """Mock geological module for testing"""
    
    def __init__(self, name: str = "test_module"):
        self.name = name
        self.initialized = False
        self.health_status = "healthy"
    
    def get_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            name=self.name,
            version="1.0.0",
            description="Test geological module",
            author="Test Author",
            geological_domain="test_domain",
            capabilities=["analysis", "modeling"]
        )
    
    async def initialize(self, config: dict) -> bool:
        self.initialized = True
        return True
    
    async def health_check(self) -> dict:
        return {"status": self.health_status}
    
    async def cleanup(self) -> bool:
        return True
    
    def analyze_seismic_data(self, data):
        """Mock geological analysis method"""
        return {"result": "analyzed"}


class MockVolcanicModule(GeologicalModuleInterface):
    """Mock volcanic module with dependencies"""
    
    def get_metadata(self) -> ModuleMetadata:
        return ModuleMetadata(
            name="volcanic_module",
            version="1.0.0",
            description="Test volcanic module",
            author="Test Author",
            dependencies=[
                ModuleDependency(name="test_module", optional=False)
            ],
            priority=ModulePriority.HIGH,
            geological_domain="volcanology",
            capabilities=["monitoring", "prediction"]
        )
    
    async def initialize(self, config: dict) -> bool:
        return True
    
    async def health_check(self) -> dict:
        return {"status": "healthy"}
    
    async def cleanup(self) -> bool:
        return True


@pytest.fixture
async def framework():
    """Create a test framework instance"""
    config = {
        'module_path': 'backend/modules',
        'health_check_interval': 1,  # Short interval for testing
        'error_handling': {},
        'security': {},
        'database': {}
    }
    
    framework = CoreModuleIntegrationFramework(config)
    
    # Mock the core components to avoid actual initialization
    framework.error_handler = Mock()
    framework.security_framework = Mock()
    framework.database_integrator = Mock()
    
    yield framework
    
    # Cleanup
    await framework.shutdown()


@pytest.fixture
def temp_module_dir():
    """Create temporary module directory for testing"""
    temp_dir = tempfile.mkdtemp()
    module_path = Path(temp_dir) / "modules"
    module_path.mkdir()
    
    # Create a test module file
    test_module_content = '''
"""Test geological module"""

class TestGeologicalProcessor:
    def analyze_geological_data(self, data):
        return {"processed": True}
    
    def get_metadata(self):
        from backend.core.module_loader import ModuleMetadata
        return ModuleMetadata(
            name="test_geological_processor",
            version="1.0.0",
            description="Test processor",
            author="Test"
        )
    
    async def initialize(self, config):
        return True
    
    async def health_check(self):
        return {"status": "healthy"}
    
    async def cleanup(self):
        return True
'''
    
    (module_path / "test_geological_processor.py").write_text(test_module_content)
    
    yield str(module_path)
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestModuleRegistry:
    """Test the module registry functionality"""
    
    def test_module_registration(self, framework):
        """Test module registration"""
        mock_module = MockGeologicalModule("test_module")
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        # Register module
        success = framework.registry.register_module("test_module", instance)
        assert success
        
        # Retrieve module
        retrieved = framework.registry.get_module("test_module")
        assert retrieved is not None
        assert retrieved.metadata.name == "test_module"
    
    def test_dependency_graph(self, framework):
        """Test dependency graph construction"""
        # Register base module
        base_module = MockGeologicalModule("base_module")
        base_metadata = base_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        base_instance = ModuleInstance(
            metadata=base_metadata,
            module_object=base_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("base_module", base_instance)
        
        # Register dependent module
        dependent_module = MockVolcanicModule()
        dependent_metadata = dependent_module.get_metadata()
        
        dependent_instance = ModuleInstance(
            metadata=dependent_metadata,
            module_object=dependent_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("volcanic_module", dependent_instance)
        
        # Check dependency graph
        assert "base_module" in framework.registry.dependency_graph.get("volcanic_module", set())
    
    def test_loading_order(self, framework):
        """Test module loading order calculation"""
        # Register modules with dependencies
        base_module = MockGeologicalModule("base_module")
        base_metadata = base_module.get_metadata()
        base_metadata.priority = ModulePriority.CRITICAL
        
        dependent_module = MockVolcanicModule()
        dependent_metadata = dependent_module.get_metadata()
        dependent_metadata.dependencies = [ModuleDependency(name="base_module")]
        
        from backend.core.module_loader import ModuleInstance
        
        base_instance = ModuleInstance(
            metadata=base_metadata,
            module_object=base_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        dependent_instance = ModuleInstance(
            metadata=dependent_metadata,
            module_object=dependent_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("base_module", base_instance)
        framework.registry.register_module("volcanic_module", dependent_instance)
        
        # Get loading order
        order = framework.registry.get_loading_order()
        
        # Base module should come before dependent module
        base_index = order.index("base_module")
        dependent_index = order.index("volcanic_module")
        assert base_index < dependent_index


class TestModuleLoading:
    """Test module loading functionality"""
    
    @pytest.mark.asyncio
    async def test_module_discovery(self, framework, temp_module_dir):
        """Test module discovery"""
        framework.module_path = Path(temp_module_dir)
        
        discovered = await framework.discover_modules()
        assert "test_geological_processor" in discovered
    
    @pytest.mark.asyncio
    async def test_module_loading(self, framework):
        """Test individual module loading"""
        # Mock the importlib to return our test module
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.TestGeologicalProcessor = MockGeologicalModule
            mock_import.return_value = mock_module
            
            success = await framework.load_module("test_module")
            assert success
            
            # Check if module was registered
            loaded_module = framework.registry.get_module("test_module.MockGeologicalModule")
            assert loaded_module is not None
    
    @pytest.mark.asyncio
    async def test_module_initialization(self, framework):
        """Test module initialization"""
        mock_module = MockGeologicalModule("test_init_module")
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        # Initialize module
        success = await framework._initialize_module(instance)
        assert success
        assert mock_module.initialized
        assert instance.status == ModuleStatus.READY


class TestInterModuleCommunication:
    """Test inter-module communication system"""
    
    @pytest.mark.asyncio
    async def test_message_handling(self, framework):
        """Test message registration and handling"""
        received_events = []
        
        async def test_handler(event):
            received_events.append(event)
        
        # Register handler
        framework.register_message_handler("test_event", test_handler)
        
        # Send message
        await framework.send_message("test_event", {"data": "test"})
        
        # Process events (simulate event loop)
        await asyncio.sleep(0.1)
        
        # Check if event was received
        assert len(received_events) > 0
        assert received_events[0]["type"] == "test_event"
        assert received_events[0]["data"]["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_module_retrieval_by_capability(self, framework):
        """Test retrieving modules by capability"""
        mock_module = MockGeologicalModule("capability_test")
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.READY,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("capability_test", instance)
        
        # Get modules by capability
        analysis_modules = framework.get_modules_by_capability("analysis")
        assert len(analysis_modules) > 0
        assert mock_module in analysis_modules


class TestHealthMonitoring:
    """Test health monitoring functionality"""
    
    @pytest.mark.asyncio
    async def test_health_checks(self, framework):
        """Test periodic health checks"""
        mock_module = MockGeologicalModule("health_test")
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.READY,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("health_test", instance)
        
        # Perform health check
        await framework._perform_health_checks()
        
        # Check health status was updated
        assert instance.health_status == "healthy"
    
    @pytest.mark.asyncio
    async def test_unhealthy_module_detection(self, framework):
        """Test detection of unhealthy modules"""
        mock_module = MockGeologicalModule("unhealthy_test")
        mock_module.health_status = "error"
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.READY,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("unhealthy_test", instance)
        
        # Perform health check
        await framework._perform_health_checks()
        
        # Check error was detected
        assert instance.health_status == "error"


class TestFrameworkIntegration:
    """Test overall framework integration"""
    
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test complete framework initialization"""
        config = {
            'module_path': 'backend/modules',
            'health_check_interval': 1
        }
        
        with patch('backend.core.module_loader.CoreModuleIntegrationFramework._initialize_core_components') as mock_init:
            mock_init.return_value = None
            
            framework = CoreModuleIntegrationFramework(config)
            
            # Mock core components
            framework.error_handler = Mock()
            framework.security_framework = Mock()
            framework.database_integrator = Mock()
            
            # Mock module discovery and loading
            with patch.object(framework, 'discover_modules', return_value=[]):
                with patch.object(framework, 'load_all_modules', return_value={}):
                    success = await framework.initialize()
                    assert success
    
    def test_framework_status(self, framework):
        """Test framework status reporting"""
        # Add some mock modules
        mock_module = MockGeologicalModule("status_test")
        metadata = mock_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=mock_module,
            status=ModuleStatus.READY,
            load_time=datetime.now()
        )
        
        framework.registry.register_module("status_test", instance)
        
        # Get status
        status = framework.get_framework_status()
        
        assert status['total_modules'] == 1
        assert status['ready_modules'] == 1
        assert status['error_modules'] == 0
        assert 'performance_metrics' in status
    
    @pytest.mark.asyncio
    async def test_global_framework_instance(self):
        """Test global framework instance management"""
        # Test getting framework instance
        framework1 = get_framework()
        framework2 = get_framework()
        
        # Should be the same instance
        assert framework1 is framework2
        
        # Test initialization
        with patch.object(CoreModuleIntegrationFramework, 'initialize', return_value=True):
            success = await initialize_framework({'test': True})
            assert success


class TestErrorHandling:
    """Test error handling in module operations"""
    
    @pytest.mark.asyncio
    async def test_module_loading_error_recovery(self, framework):
        """Test error recovery during module loading"""
        # Mock import to raise an exception
        with patch('importlib.import_module', side_effect=ImportError("Module not found")):
            success = await framework.load_module("nonexistent_module")
            assert not success
    
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self, framework):
        """Test handling of module initialization errors"""
        class FailingModule(GeologicalModuleInterface):
            def get_metadata(self):
                return ModuleMetadata(
                    name="failing_module",
                    version="1.0.0",
                    description="Failing module",
                    author="Test"
                )
            
            async def initialize(self, config):
                raise Exception("Initialization failed")
            
            async def health_check(self):
                return {"status": "error"}
            
            async def cleanup(self):
                return True
        
        failing_module = FailingModule()
        metadata = failing_module.get_metadata()
        
        from backend.core.module_loader import ModuleInstance
        instance = ModuleInstance(
            metadata=metadata,
            module_object=failing_module,
            status=ModuleStatus.LOADED,
            load_time=datetime.now()
        )
        
        # Try to initialize failing module
        success = await framework._initialize_module(instance)
        assert not success
        assert instance.status == ModuleStatus.ERROR
        assert instance.error_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])