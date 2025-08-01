"""
Integration module for hierarchical error management system

This module provides decorators and utilities to integrate the error management
system with existing Prometheus AI services.

Requirements: 12.3, 12.4
"""

import asyncio
import functools
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional

try:
    from .error_management import (
        PrometheusErrorHandler,
        create_error_context,
        prometheus_error_handler,
    )
except ImportError:
    from error_management import (
        PrometheusErrorHandler,
        create_error_context,
    )
    from error_management import pandas as pd
    from error_management import (
        prometheus_error_handler,
        rt,
    )
    from error_management import torch.nn as nn
    from error_management import torch.nn.functional as F

logger = logging.getLogger(__name__)

def with_error_handling(
    service_name: str,
    operation: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_result: Any = None,
):
    """
    Decorator to add hierarchical error handling to functions

    Args:
        service_name: Name of the service (e.g., 'helios_core', 'volcano_watchtower')
        operation: Name of the operation (defaults to function name)
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retry attempts in seconds
        fallback_result: Result to return if all recovery attempts fail
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            op_name = operation or func.__name__

            for attempt in range(max_retries + 1):
                try:
                    # Create error context
                    context = create_error_context(
                        service_name=service_name,
                        operation=op_name,
                        retry_count=attempt,
                        max_retries=max_retries,
                        additional_context={
                            "function_args": str(args)[:200],  # Truncate for logging
                            "function_kwargs": str(kwargs)[:200],
                        },
                    )

                    # Execute the function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)

                    # Log successful execution after retries
                    if attempt > 0:
                        logger.info(
                            f"Function {op_name} succeeded after {attempt} retries"
                        )

                    return result

                except Exception as error:
                    logger.error(f"Error in {op_name} (attempt {attempt + 1}): {error}")

                    # Use error management system for recovery
                    try:
                        recovery_result = await prometheus_error_handler.handle_error(
                            error, context
                        )
                        logger.info(f"Error recovery successful for {op_name}")
                        return recovery_result
                    except Exception as recovery_error:
                        logger.error(
                            f"Error recovery failed for {op_name}: {recovery_error}"
                        )

                        # If not the last attempt, wait and retry
                        if attempt < max_retries:
                            await asyncio.sleep(
                                retry_delay * (2**attempt)
                            )  # Exponential backoff
                            continue
                        else:
                            # All attempts failed, return fallback or re-raise
                            if fallback_result is not None:
                                logger.warning(
                                    f"Returning fallback result for {op_name}"
                                )
                                return fallback_result
                            else:
                                logger.critical(
                                    f"All recovery attempts failed for {op_name}"
                                )
                                raise recovery_error

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # For synchronous functions, run the async wrapper in an event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def with_geological_error_handling(operation: Optional[str] = None):
    """
    Specialized decorator for geological modeling operations

    Provides domain-specific error handling for geological simulations
    """

    def decorator(func: Callable) -> Callable:
        @with_error_handling(
            service_name="helios_core",
            operation=operation,
            max_retries=2,
            retry_delay=2.0,
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Add geological-specific context
            context_data = {
                "domain": "geological_modeling",
                "physics_constraints_active": True,
            }

            # Extract geological parameters if available
            if "geological_params" in kwargs:
                context_data["geological_params"] = str(kwargs["geological_params"])[
                    :100
                ]

            # Update context in kwargs for the error handler
            if "error_context_data" not in kwargs:
                kwargs["error_context_data"] = context_data
            else:
                kwargs["error_context_data"].update(context_data)

            return await func(*args, **kwargs)

        return wrapper

    return decorator

def with_realtime_monitoring_error_handling(operation: Optional[str] = None):
    """
    Specialized decorator for real-time monitoring operations

    Provides domain-specific error handling for volcanic monitoring
    """

    def decorator(func: Callable) -> Callable:
        @with_error_handling(
            service_name="volcano_watchtower",
            operation=operation,
            max_retries=1,  # Fast retry for real-time operations
            retry_delay=0.5,
            fallback_result={"status": "degraded", "data": "cached"},
        )
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Add monitoring-specific context
            context_data = {
                "domain": "realtime_monitoring",
                "time_critical": True,
                "sensor_data_processing": True,
            }

            # Extract sensor information if available
            if "sensor_id" in kwargs:
                context_data["sensor_id"] = kwargs["sensor_id"]
            if "data_timestamp" in kwargs:
                context_data["data_timestamp"] = str(kwargs["data_timestamp"])

            kwargs["error_context_data"] = kwargs.get("error_context_data", {})
            kwargs["error_context_data"].update(context_data)

            return await func(*args, **kwargs)

        return wrapper

    return decorator

class ErrorHandlingMixin:
    """
    Mixin class to add error handling capabilities to existing classes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_handler = prometheus_error_handler
        self.service_name = getattr(self, "SERVICE_NAME", self.__class__.__name__)

    async def handle_operation_error(
        self,
        error: Exception,
        operation: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Handle errors for class operations"""
        context = create_error_context(
            service_name=self.service_name,
            operation=operation,
            additional_context=context_data or {},
        )

        return await self.error_handler.handle_error(error, context)

    def get_error_health_status(self) -> Dict[str, Any]:
        """Get error management health status"""
        return self.error_handler.get_system_health_status()

class GeologicalModelErrorHandler(ErrorHandlingMixin):
    """
    Specialized error handler for geological modeling operations
    """

    SERVICE_NAME = "helios_core"

    async def handle_physics_constraint_violation(
        self, error: Exception, model_params: Dict[str, Any], constraint_type: str
    ) -> Any:
        """Handle physics constraint violations in geological models"""
        context_data = {
            "constraint_type": constraint_type,
            "model_parameters": str(model_params)[:200],
            "physics_domain": "geological_modeling",
        }

        return await self.handle_operation_error(
            error, f"physics_constraint_{constraint_type}", context_data
        )

    async def handle_convergence_failure(
        self, error: Exception, model_type: str, training_params: Dict[str, Any]
    ) -> Any:
        """Handle model convergence failures"""
        context_data = {
            "model_type": model_type,
            "training_parameters": str(training_params)[:200],
            "convergence_issue": True,
        }

        return await self.handle_operation_error(
            error, f"convergence_failure_{model_type}", context_data
        )

class VolcanicMonitoringErrorHandler(ErrorHandlingMixin):
    """
    Specialized error handler for volcanic monitoring operations
    """

    SERVICE_NAME = "volcano_watchtower"

    async def handle_sensor_data_error(
        self, error: Exception, sensor_id: str, data_type: str, timestamp: datetime
    ) -> Any:
        """Handle sensor data errors in volcanic monitoring"""
        context_data = {
            "sensor_id": sensor_id,
            "data_type": data_type,
            "timestamp": timestamp.isoformat(),
            "monitoring_domain": "volcanic_sensors",
        }

        return await self.handle_operation_error(
            error, f"sensor_data_{data_type}", context_data
        )

    async def handle_alert_system_error(
        self, error: Exception, alert_level: str, volcano_id: str
    ) -> Any:
        """Handle alert system errors"""
        context_data = {
            "alert_level": alert_level,
            "volcano_id": volcano_id,
            "critical_system": True,
        }

        return await self.handle_operation_error(
            error, f"alert_system_{alert_level}", context_data
        )

# Example usage and integration patterns
class ExampleGeologicalModel(GeologicalModelErrorHandler):
    """Example geological model with integrated error handling"""

    @with_geological_error_handling("pinn_simulation")
    async def run_pinn_simulation(
        self, geological_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run PINN simulation with error handling"""
        # Simulate potential physics constraint violation
        if geological_params.get("temperature", 0) < 0:
            raise Exception(
                "Physics constraint violated: temperature below absolute zero"
            )

        # Simulate model computation
        result = {
            "simulation_type": "pinn",
            "parameters": geological_params,
            "result": "simulation_complete",
            "timestamp": datetime.now(),
        }

        return result

    @with_geological_error_handling("mantle_convection")
    async def simulate_mantle_convection(
        self, grid_resolution: str, time_span: str
    ) -> Dict[str, Any]:
        """Simulate mantle convection with error handling"""
        # Simulate potential memory error for high resolution
        if grid_resolution == "ultra_high":
            raise MemoryError(
                "Insufficient memory for ultra-high resolution simulation"
            )

        return {
            "simulation_type": "mantle_convection",
            "grid_resolution": grid_resolution,
            "time_span": time_span,
            "result": "convection_patterns_computed",
        }

class ExampleVolcanicMonitor(VolcanicMonitoringErrorHandler):
    """Example volcanic monitor with integrated error handling"""

    @with_realtime_monitoring_error_handling("seismic_processing")
    async def process_seismic_data(
        self, sensor_id: str, waveform_data: Any
    ) -> Dict[str, Any]:
        """Process seismic data with error handling"""
        # Simulate data corruption error
        if waveform_data is None:
            raise ValueError(f"Corrupted waveform data from sensor {sensor_id}")

        return {
            "sensor_id": sensor_id,
            "processing_result": "seismic_events_detected",
            "event_count": 3,
            "timestamp": datetime.now(),
        }

    @with_realtime_monitoring_error_handling("alert_generation")
    async def generate_volcanic_alert(
        self, volcano_id: str, risk_level: float
    ) -> Dict[str, Any]:
        """Generate volcanic alert with error handling"""
        # Simulate alert system failure
        if risk_level > 0.9:
            raise Exception("Alert system overload - too many high-risk alerts")

        return {
            "volcano_id": volcano_id,
            "alert_level": "moderate" if risk_level > 0.5 else "low",
            "risk_assessment": risk_level,
            "timestamp": datetime.now(),
        }

# Utility functions for error handling integration
def setup_service_error_handling(service_name: str) -> PrometheusErrorHandler:
    """Set up error handling for a specific service"""
    logger.info(f"Setting up error handling for service: {service_name}")
    return prometheus_error_handler

async def test_error_handling_integration():
    """Test the error handling integration"""
    logger.info("Testing error handling integration...")

    # Test geological model error handling
    geo_model = ExampleGeologicalModel()

    try:
        # This should trigger physics constraint violation
        result = await geo_model.run_pinn_simulation({"temperature": -100})
        logger.info(f"Geological simulation result: {result}")
    except Exception as e:
        logger.error(f"Geological simulation failed: {e}")

    # Test volcanic monitoring error handling
    volcano_monitor = ExampleVolcanicMonitor()

    try:
        # This should trigger data corruption error
        result = await volcano_monitor.process_seismic_data("VS001", None)
        logger.info(f"Seismic processing result: {result}")
    except Exception as e:
        logger.error(f"Seismic processing failed: {e}")

    # Check system health
    health = prometheus_error_handler.get_system_health_status()
    logger.info(f"System health status: {health['system_status']}")

if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_error_handling_integration())
