"""
Test suite for the hierarchical error management system

Tests all components of the error management system including:
    - Error classification
- Recovery strategies
- Graceful degradation
- Error analysis and alerting
"""

from datetime import datetime, timedelta

import pytest
from core.error_management import (
    AlertSystem,
    DataQualityRecoveryStrategy,
    ErrorAnalyzer,
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
    ErrorInfo,
    ErrorSeverity,
    GracefulDegradationManager,
    ModelConvergenceRecoveryStrategy,
    NetworkConnectivityRecoveryStrategy,
    PhysicsViolationRecoveryStrategy,
    PrometheusErrorHandler,
    SecurityRecoveryStrategy,
    ServiceDegradationLevel,
    SystemResourceRecoveryStrategy,
    create_error_context,
    get_system_health,
    handle_error,
    import,
)
from core.error_management import torch.nn as nn
from core.error_management import torch.nn.functional as F

class TestErrorClassifier:
    """Test error classification functionality"""

    def setup_method(self):
        self.classifier = ErrorClassifier()

    def test_data_quality_error_classification(self):
        """Test classification of data quality errors"""
        error = ValueError("Invalid data format in sensor reading")
        context = create_error_context("test_service", "data_processing")

        error_info = self.classifier.classify(error, context)

        assert error_info.category == ErrorCategory.DATA_QUALITY
        assert error_info.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.LOW]
        assert error_info.message == "Invalid data format in sensor reading"
        assert error_info.context == context

    def test_model_convergence_error_classification(self):
        """Test classification of model convergence errors"""
        error = Exception("Model training diverged after 100 epochs")

        error_info = self.classifier.classify(error)

        assert error_info.category == ErrorCategory.MODEL_CONVERGENCE
        assert error_info.severity == ErrorSeverity.HIGH

    def test_physics_violation_error_classification(self):
        """Test classification of physics constraint violations"""
        error = Exception("Physics constraint violated: energy conservation failed")

        error_info = self.classifier.classify(error)

        assert error_info.category == ErrorCategory.PHYSICS_VIOLATION
        assert error_info.severity == ErrorSeverity.CRITICAL

    def test_system_resource_error_classification(self):
        """Test classification of system resource errors"""
        error = MemoryError("Out of memory during model training")

        error_info = self.classifier.classify(error)

        assert error_info.category == ErrorCategory.SYSTEM_RESOURCE
        assert error_info.severity == ErrorSeverity.CRITICAL

    def test_network_connectivity_error_classification(self):
        """Test classification of network connectivity errors"""
        error = ConnectionError("Connection failed to external API")

        error_info = self.classifier.classify(error)

        assert error_info.category == ErrorCategory.NETWORK_CONNECTIVITY
        assert error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]

    def test_security_error_classification(self):
        """Test classification of security errors"""
        error = Exception("Authentication failed: invalid token")

        error_info = self.classifier.classify(error)

        assert error_info.category == ErrorCategory.SECURITY
        assert error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM]

class TestRecoveryStrategies:
    """Test recovery strategy implementations"""

    @pytest.mark.asyncio
    async def test_data_quality_recovery_strategy(self):
        """Test data quality recovery strategy"""
        strategy = DataQualityRecoveryStrategy()

        # Create test error info
        error_info = ErrorInfo(
            error_id="test_error_1",
            category=ErrorCategory.DATA_QUALITY,
            severity=ErrorSeverity.MEDIUM,
            message="Invalid sensor data",
            context=ErrorContext(
                service_name="test_service",
                operation="data_processing",
                timestamp=datetime.now(),
                data={"temperature": None, "pressure": 1013.25},
            ),
        )

        # Test if strategy can handle the error
        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        # Test recovery
        result = await strategy.recover(error_info)
        assert result is not None
        assert "timestamp" in result
        assert "location" in result

    @pytest.mark.asyncio
    async def test_model_convergence_recovery_strategy(self):
        """Test model convergence recovery strategy"""
        strategy = ModelConvergenceRecoveryStrategy()

        error_info = ErrorInfo(
            error_id="test_error_2",
            category=ErrorCategory.MODEL_CONVERGENCE,
            severity=ErrorSeverity.HIGH,
            message="Model training failed to converge",
            context=ErrorContext(
                service_name="helios_core",
                operation="model_training",
                timestamp=datetime.now(),
                model_complexity="standard",
            ),
        )

        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        result = await strategy.recover(error_info)
        assert result is not None
        assert "model_type" in result
        assert result["model_type"] in ["simplified", "ensemble_fallback"]

    @pytest.mark.asyncio
    async def test_physics_violation_recovery_strategy(self):
        """Test physics violation recovery strategy"""
        strategy = PhysicsViolationRecoveryStrategy()

        error_info = ErrorInfo(
            error_id="test_error_3",
            category=ErrorCategory.PHYSICS_VIOLATION,
            severity=ErrorSeverity.CRITICAL,
            message="Energy conservation law violated",
            context=ErrorContext(
                service_name="helios_core",
                operation="physics_simulation",
                timestamp=datetime.now(),
                data={"energy": -100, "temperature": 300},
            ),
        )

        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        result = await strategy.recover(error_info)
        assert result is not None
        # Energy should be corrected to positive value
        if "energy" in result:
            assert result["energy"] >= 0

    @pytest.mark.asyncio
    async def test_system_resource_recovery_strategy(self):
        """Test system resource recovery strategy"""
        strategy = SystemResourceRecoveryStrategy()

        error_info = ErrorInfo(
            error_id="test_error_4",
            category=ErrorCategory.SYSTEM_RESOURCE,
            severity=ErrorSeverity.HIGH,
            message="Memory exhausted during computation",
            context=ErrorContext(
                service_name="helios_core",
                operation="large_simulation",
                timestamp=datetime.now(),
            ),
        )

        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        result = await strategy.recover(error_info)
        assert result is not None
        assert "result_type" in result
        assert result["result_type"] == "reduced_resource_result"

    @pytest.mark.asyncio
    async def test_network_connectivity_recovery_strategy(self):
        """Test network connectivity recovery strategy"""
        strategy = NetworkConnectivityRecoveryStrategy()

        error_info = ErrorInfo(
            error_id="test_error_5",
            category=ErrorCategory.NETWORK_CONNECTIVITY,
            severity=ErrorSeverity.MEDIUM,
            message="Connection timeout to external service",
            context=ErrorContext(
                service_name="data_ingestion",
                operation="external_api_call",
                timestamp=datetime.now(),
            ),
        )

        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        result = await strategy.recover(error_info)
        assert result is not None
        assert "data_source" in result

    @pytest.mark.asyncio
    async def test_security_recovery_strategy(self):
        """Test security recovery strategy"""
        strategy = SecurityRecoveryStrategy()

        error_info = ErrorInfo(
            error_id="test_error_6",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.MEDIUM,
            message="Token expired",
            context=ErrorContext(
                service_name="api_gateway",
                operation="authentication",
                timestamp=datetime.now(),
                user_id="test_user",
            ),
        )

        can_handle = await strategy.can_handle(error_info)
        assert can_handle is True

        result = await strategy.recover(error_info)
        assert result is not None
        assert "auth_status" in result

class TestErrorAnalyzer:
    """Test error analysis functionality"""

    def setup_method(self):
        self.analyzer = ErrorAnalyzer()

    def test_error_pattern_analysis(self):
        """Test error pattern analysis"""
        # Add sample errors
        for i in range(10):
            error_info = ErrorInfo(
                error_id=f"error_{i}",
                category=ErrorCategory.DATA_QUALITY,
                severity=ErrorSeverity.MEDIUM,
                message=f"Test error {i}",
                timestamp=datetime.now() - timedelta(hours=i),
            )
            self.analyzer.add_error(error_info)

        # Add some critical errors
        for i in range(3):
            error_info = ErrorInfo(
                error_id=f"critical_error_{i}",
                category=ErrorCategory.PHYSICS_VIOLATION,
                severity=ErrorSeverity.CRITICAL,
                message=f"Critical test error {i}",
                timestamp=datetime.now() - timedelta(hours=i),
            )
            self.analyzer.add_error(error_info)

        analysis = self.analyzer.analyze_error_patterns()

        assert analysis["total_errors"] == 13
        assert analysis["critical_error_count"] == 3
        assert "data_quality" in analysis["category_distribution"]
        assert "physics_violation" in analysis["category_distribution"]
        assert analysis["category_distribution"]["data_quality"] == 10
        assert analysis["category_distribution"]["physics_violation"] == 3
        assert len(analysis["recommendations"]) > 0

    def test_recommendation_generation(self):
        """Test recommendation generation based on error patterns"""
        # Add many data quality errors
        for i in range(15):
            error_info = ErrorInfo(
                error_id=f"dq_error_{i}",
                category=ErrorCategory.DATA_QUALITY,
                severity=ErrorSeverity.MEDIUM,
                message=f"Data quality error {i}",
                timestamp=datetime.now(),
            )
            self.analyzer.add_error(error_info)

        analysis = self.analyzer.analyze_error_patterns()
        recommendations = analysis["recommendations"]

        # Should recommend data validation improvements
        assert any("data validation" in rec.lower() for rec in recommendations)

class TestGracefulDegradationManager:
    """Test graceful degradation functionality"""

    def setup_method(self):
        self.degradation_manager = GracefulDegradationManager()
        self.error_analyzer = ErrorAnalyzer()

    def test_degradation_assessment(self):
        """Test degradation level assessment"""
        # Add critical errors to trigger degradation
        for i in range(6):
            error_info = ErrorInfo(
                error_id=f"critical_{i}",
                category=ErrorCategory.SYSTEM_RESOURCE,
                severity=ErrorSeverity.CRITICAL,
                message=f"Critical error {i}",
                timestamp=datetime.now(),
            )
            self.error_analyzer.add_error(error_info)

        recommended_level = self.degradation_manager.assess_degradation_need(
            self.error_analyzer
        )

        assert recommended_level == ServiceDegradationLevel.EMERGENCY_MODE

    def test_service_configuration_for_degradation_levels(self):
        """Test service configuration for different degradation levels"""
        # Test full service configuration
        config = self.degradation_manager._get_service_configuration(
            ServiceDegradationLevel.FULL_SERVICE
        )
        assert "all" in config["available_features"]
        assert config["model_complexity"] == "full"

        # Test emergency mode configuration
        config = self.degradation_manager._get_service_configuration(
            ServiceDegradationLevel.EMERGENCY_MODE
        )
        assert config["available_features"] == ["emergency_alerts"]
        assert config["model_complexity"] == "precomputed"

    def test_degradation_application(self):
        """Test application of service degradation"""
        self.degradation_manager.current_degradation_level

        config = self.degradation_manager.apply_degradation(
            ServiceDegradationLevel.REDUCED_PRECISION
        )

        assert (
            self.degradation_manager.current_degradation_level == ServiceDegradationLevel.REDUCED_PRECISION
        )
        assert len(self.degradation_manager.degradation_history) == 1
        assert config["model_complexity"] == "reduced"

class TestAlertSystem:
    """Test alert system functionality"""

    def setup_method(self):
        self.alert_system = AlertSystem()

    @pytest.mark.asyncio
    async def test_individual_error_alert(self):
        """Test sending alerts for individual errors"""
        error_info = ErrorInfo(
            error_id="test_alert_error",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            message="Security breach detected",
            context=ErrorContext(
                service_name="api_gateway",
                operation="authentication",
                timestamp=datetime.now(),
            ),
        )

        await self.alert_system.send_alert(error_info)

        assert len(self.alert_system.alert_history) == 1
        alert = self.alert_system.alert_history[0]
        assert alert["error_id"] == "test_alert_error"
        assert alert["category"] == "security"
        assert alert["severity"] == "high"

    @pytest.mark.asyncio
    async def test_critical_system_alert(self):
        """Test sending critical system alerts"""
        await self.alert_system.send_critical_alert("System failure detected")

        assert len(self.alert_system.alert_history) == 1
        alert = self.alert_system.alert_history[0]
        assert alert["alert_type"] == "critical_system"
        assert alert["requires_immediate_attention"] is True

    @pytest.mark.asyncio
    async def test_pattern_alert(self):
        """Test sending pattern-based alerts"""
        analysis = {"critical_error_count": 5, "error_rate_24h": 25.0}

        await self.alert_system.send_pattern_alert("High error rate detected", analysis)

        assert len(self.alert_system.alert_history) == 1
        alert = self.alert_system.alert_history[0]
        assert alert["alert_type"] == "error_pattern"
        assert alert["analysis"]["critical_error_count"] == 5

class TestPrometheusErrorHandler:
    """Test main error handler functionality"""

    def setup_method(self):
        self.error_handler = PrometheusErrorHandler()

    @pytest.mark.asyncio
    async def test_complete_error_handling_flow(self):
        """Test complete error handling flow"""
        # Create a data quality error
        error = ValueError("Invalid sensor data format")
        context = create_error_context(
            service_name="volcano_watchtower",
            operation="sensor_data_processing",
            data={"temperature": None, "pressure": 1013.25},
        )

        result = await self.error_handler.handle_error(error, context)

        # Verify error was handled and result returned
        assert result is not None

        # Verify error was added to analyzer
        assert len(self.error_handler.error_analyzer.error_history) == 1

        # Verify error was classified correctly
        error_info = self.error_handler.error_analyzer.error_history[0]
        assert error_info.category == ErrorCategory.DATA_QUALITY
        assert error_info.resolved is True
        assert error_info.recovery_time is not None

    @pytest.mark.asyncio
    async def test_error_handling_with_degradation(self):
        """Test error handling that triggers service degradation"""
        # Add multiple critical errors to trigger degradation
        for i in range(6):
            error = Exception(f"Critical system failure {i}")
            context = create_error_context("helios_core", f"operation_{i}")

            try:
                await self.error_handler.handle_error(error, context)
            except Exception:
                pass  # Some errors might not be recoverable

        # Check if degradation was applied
        health_status = self.error_handler.get_system_health_status()
        assert health_status["system_status"] in ["critical", "degraded"]

    def test_system_health_status(self):
        """Test system health status reporting"""
        health_status = self.error_handler.get_system_health_status()

        assert "degradation_level" in health_status
        assert "error_analysis" in health_status
        assert "service_configuration" in health_status
        assert "timestamp" in health_status
        assert "system_status" in health_status

        # Initially should be healthy
        assert health_status["system_status"] == "healthy"

class TestConvenienceFunctions:
    """Test convenience functions for easy integration"""

    @pytest.mark.asyncio
    async def test_global_handle_error_function(self):
        """Test global handle_error convenience function"""
        error = ValueError("Test error for global handler")
        context = create_error_context("test_service", "test_operation")

        result = await handle_error(error, context)
        assert result is not None

    def test_get_system_health_function(self):
        """Test global get_system_health convenience function"""
        health = get_system_health()

        assert isinstance(health, dict)
        assert "system_status" in health
        assert "degradation_level" in health

    def test_create_error_context_function(self):
        """Test create_error_context convenience function"""
        context = create_error_context(
            service_name="test_service",
            operation="test_operation",
            user_id="test_user",
            session_id="test_session",
        )

        assert context.service_name == "test_service"
        assert context.operation == "test_operation"
        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert isinstance(context.timestamp, datetime)

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""

    @pytest.mark.asyncio
    async def test_volcanic_monitoring_error_scenario(self):
        """Test error handling in volcanic monitoring scenario"""
        error_handler = PrometheusErrorHandler()

        # Simulate sensor data corruption
        error = ValueError("Seismic sensor data corrupted - invalid waveform format")
        context = create_error_context(
            service_name="volcano_watchtower",
            operation="seismic_data_processing",
            data={
                "sensor_id": "VS001",
                "waveform_data": None,
                "timestamp": datetime.now(),
                "location": {"lat": 45.3311, "lon": -121.7113},  # Mount Hood
            },
        )

        result = await error_handler.handle_error(error, context)

        # Should provide fallback data
        assert result is not None
        assert "quality_flag" in result
        assert result["quality_flag"] == "estimated"

    @pytest.mark.asyncio
    async def test_model_training_failure_scenario(self):
        """Test error handling in model training failure scenario"""
        error_handler = PrometheusErrorHandler()

        # Simulate PINN training convergence failure
        error = Exception("PINN model training diverged - gradient explosion detected")
        context = create_error_context(
            service_name="helios_core",
            operation="pinn_training",
            model_complexity="full",
            data={
                "model_type": "physics_informed_neural_network",
                "physics_constraints": ["navier_stokes", "heat_transfer"],
                "training_data_size": 100000,
            },
        )

        result = await error_handler.handle_error(error, context)

        # Should fall back to simplified model
        assert result is not None
        assert result["model_type"] in ["simplified", "ensemble_fallback"]
        assert result["confidence"] < 1.0

    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenario(self):
        """Test error handling in resource exhaustion scenario"""
        error_handler = PrometheusErrorHandler()

        # Simulate memory exhaustion during large simulation
        error = MemoryError(
            "Cannot allocate memory for geological simulation - 64GB required"
        )
        context = create_error_context(
            service_name="helios_core",
            operation="large_scale_simulation",
            data={
                "simulation_type": "mantle_convection",
                "grid_resolution": "ultra_high",
                "time_span": "100_million_years",
            },
        )

        result = await error_handler.handle_error(error, context)

        # Should optimize resource usage and retry
        assert result is not None
        assert result["resource_usage"] == "optimized"

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
