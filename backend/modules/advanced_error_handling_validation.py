"""
Advanced Error Handling and Validation System
Comprehensive error handling, validation, and recovery mechanisms for all advanced implementations
"""

import asyncio
import logging
import traceback
import functools
import inspect
import sys
import time
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Type, TypeVar
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import pickle
import os
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager
import warnings
from abc import ABC, abstractmethod

# Advanced validation imports
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator, ValidationError
from marshmallow import Schema, fields, ValidationError as MarshmallowValidationError

# Monitoring and alerting
try:
    import sentry_sdk
    from sentry_sdk import capture_exception, capture_message
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    warnings.warn("Sentry SDK not available. Advanced error tracking will be limited.")

# Circuit breaker pattern
try:
    import pybreaker
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False
    warnings.warn("PyBreaker not available. Circuit breaker pattern will be limited.")

# Retry mechanisms
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    warnings.warn("Tenacity not available. Advanced retry mechanisms will be limited.")

# Schema validation
try:
    from cerberus import Validator
    CERBERUS_AVAILABLE = True
except ImportError:
    CERBERUS_AVAILABLE = False
    warnings.warn("Cerberus not available. Advanced schema validation will be limited.")

warnings.filterwarnings('ignore')

# Type variables for generic error handling
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

class ErrorSeverity(Enum):
    """Error severity levels for categorization and handling priority."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for systematic classification."""
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATABASE_OPERATION = "database_operation"
    NETWORK_COMMUNICATION = "network_communication"
    SYSTEM_RESOURCE = "system_resource"
    GEOLOGICAL_CONSTRAINT = "geological_constraint"
    INTEGRATION_FAILURE = "integration_failure"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_VIOLATION = "security_violation"

class RecoveryStrategy(Enum):
    """Recovery strategies for different types of errors."""
    RETRY_IMMEDIATELY = "retry_immediately"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_TO_DEFAULT = "fallback_to_default"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class ErrorContext:
    """Comprehensive error context for detailed error tracking."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    module_name: str
    function_name: str
    line_number: int
    error_message: str
    stack_trace: str
    input_parameters: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_strategy: RecoveryStrategy
    recovery_attempts: int = 0
    resolution_status: str = "pending"
    geological_context: Optional[Dict[str, Any]] = None
    performance_impact: Optional[Dict[str, float]] = None

@dataclass
class ValidationResult:
    """Result of validation operations."""
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    corrected_data: Optional[Any] = None

class AdvancedErrorHandler:
    """
    Advanced error handling system with geological domain awareness.
    
    Provides comprehensive error tracking, classification, recovery mechanisms,
    and integration with monitoring systems for production-ready error management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".AdvancedErrorHandler")
        
        # Error tracking and storage
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, int] = {}
        self.recovery_statistics: Dict[RecoveryStrategy, Dict[str, int]] = {}
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, Any] = {}
        
        # Performance monitoring
        self.error_metrics = {
            'total_errors': 0,
            'errors_by_category': {},
            'errors_by_severity': {},
            'recovery_success_rate': 0.0,
            'average_resolution_time': 0.0
        }
        
        # Initialize components
        self._initialize_circuit_breakers()
        self._initialize_sentry_integration()
        self._setup_error_handlers()
        
        self.logger.info("Advanced Error Handler initialized")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for different system components."""
        if not CIRCUIT_BREAKER_AVAILABLE:
            return
        
        # Circuit breaker configurations for different components
        breaker_configs = {
            'database_operations': {
                'failure_threshold': 5,
                'recovery_timeout': 30,
                'expected_exception': Exception
            },
            'model_inference': {
                'failure_threshold': 10,
                'recovery_timeout': 60,
                'expected_exception': Exception
            },
            'external_apis': {
                'failure_threshold': 3,
                'recovery_timeout': 120,
                'expected_exception': Exception
            },
            'geological_validation': {
                'failure_threshold': 8,
                'recovery_timeout': 45,
                'expected_exception': Exception
            }
        }
        
        for component, config in breaker_configs.items():
            self.circuit_breakers[component] = pybreaker.CircuitBreaker(
                fail_max=config['failure_threshold'],
                reset_timeout=config['recovery_timeout'],
                exclude=[config['expected_exception']]
            )
    
    def _initialize_sentry_integration(self):
        """Initialize Sentry for error tracking and monitoring."""
        if not SENTRY_AVAILABLE:
            return
        
        sentry_dsn = self.config.get('sentry_dsn', os.getenv('SENTRY_DSN'))
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                traces_sample_rate=0.1,
                environment=self.config.get('environment', 'development'),
                release=self.config.get('version', '1.0.0')
            )
            self.logger.info("Sentry error tracking initialized")
    
    def _setup_error_handlers(self):
        """Setup global error handlers for uncaught exceptions."""
        # Set up uncaught exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            error_context = ErrorContext(
                error_id=self._generate_error_id(),
                timestamp=datetime.now(),
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM_RESOURCE,
                module_name="global",
                function_name="uncaught_exception",
                line_number=0,
                error_message=str(exc_value),
                stack_trace=traceback.format_exception(exc_type, exc_value, exc_traceback),
                input_parameters={},
                system_state=self._capture_system_state(),
                recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
            )
            
            self._log_error(error_context)
            
        sys.excepthook = handle_exception
    
    def error_handler(self, 
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.INTEGRATION_FAILURE,
                     recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY_WITH_BACKOFF,
                     max_retries: int = 3,
                     fallback_value: Any = None):
        """
        Decorator for comprehensive error handling with automatic recovery.
        
        This decorator provides:
        - Automatic error classification and logging
        - Recovery strategy execution
        - Performance impact monitoring
        - Geological domain validation
        - Circuit breaker integration
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self._execute_with_error_handling(
                    func, args, kwargs, severity, category, recovery_strategy, 
                    max_retries, fallback_value, is_async=True
                )
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self._execute_with_error_handling(
                    func, args, kwargs, severity, category, recovery_strategy,
                    max_retries, fallback_value, is_async=False
                ))
            
            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    async def _execute_with_error_handling(self, func: Callable, args: tuple, kwargs: dict,
                                         severity: ErrorSeverity, category: ErrorCategory,
                                         recovery_strategy: RecoveryStrategy, max_retries: int,
                                         fallback_value: Any, is_async: bool) -> Any:
        """Execute function with comprehensive error handling."""
        start_time = time.time()
        error_context = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check circuit breaker
                component_name = self._get_component_name(func)
                if component_name in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[component_name]
                    
                    if is_async:
                        result = await circuit_breaker.call_async(func, *args, **kwargs)
                    else:
                        result = circuit_breaker.call(func, *args, **kwargs)
                else:
                    # Execute function
                    if is_async:
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                
                # Validate result if geological domain function
                if self._is_geological_function(func):
                    validation_result = await self._validate_geological_result(result, func.__name__)
                    if not validation_result.is_valid:
                        raise GeologicalValidationError(f"Result validation failed: {validation_result.validation_errors}")
                
                # Success - log performance metrics
                execution_time = time.time() - start_time
                await self._log_successful_execution(func, execution_time, attempt)
                
                return result
                
            except Exception as e:
                # Create error context
                error_context = ErrorContext(
                    error_id=self._generate_error_id(),
                    timestamp=datetime.now(),
                    severity=severity,
                    category=category,
                    module_name=func.__module__ if hasattr(func, '__module__') else 'unknown',
                    function_name=func.__name__,
                    line_number=self._get_line_number(),
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    input_parameters=self._sanitize_parameters(args, kwargs),
                    system_state=self._capture_system_state(),
                    recovery_strategy=recovery_strategy,
                    recovery_attempts=attempt,
                    geological_context=self._extract_geological_context(args, kwargs) if self._is_geological_function(func) else None,
                    performance_impact={'execution_time': time.time() - start_time, 'attempt': attempt}
                )
                
                # Log error
                self._log_error(error_context)
                
                # Attempt recovery if not last attempt
                if attempt < max_retries:
                    recovery_successful = await self._attempt_recovery(error_context, e)
                    if recovery_successful:
                        continue
                    
                    # Wait before retry if using backoff strategy
                    if recovery_strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                        wait_time = min(2 ** attempt, 60)  # Exponential backoff capped at 60 seconds
                        await asyncio.sleep(wait_time)
                else:
                    # Final attempt failed - execute final recovery strategy
                    return await self._execute_final_recovery_strategy(
                        error_context, e, fallback_value
                    )
        
        # Should not reach here, but return fallback as safety
        return fallback_value
    
    async def _attempt_recovery(self, error_context: ErrorContext, exception: Exception) -> bool:
        """Attempt recovery based on the specified recovery strategy."""
        try:
            strategy = error_context.recovery_strategy
            
            if strategy == RecoveryStrategy.RETRY_IMMEDIATELY:
                return True  # Simply retry
                
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return True  # Retry with backoff handled by caller
                
            elif strategy == RecoveryStrategy.FALLBACK_TO_DEFAULT:
                # Attempt to reset to default state
                await self._reset_to_default_state(error_context)
                return True
                
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Enable degraded mode
                await self._enable_degraded_mode(error_context)
                return True
                
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                # Circuit breaker handled automatically
                return False
                
            else:
                return False
                
        except Exception as recovery_exception:
            self.logger.error(f"Recovery attempt failed: {recovery_exception}")
            return False
    
    async def _execute_final_recovery_strategy(self, error_context: ErrorContext, 
                                             exception: Exception, fallback_value: Any) -> Any:
        """Execute final recovery strategy when all retries have failed."""
        try:
            strategy = error_context.recovery_strategy
            
            if strategy == RecoveryStrategy.FAIL_FAST:
                raise exception
                
            elif strategy == RecoveryStrategy.FALLBACK_TO_DEFAULT:
                return fallback_value
                
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Return degraded functionality result
                return await self._get_degraded_result(error_context, fallback_value)
                
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                # Log for manual intervention and return fallback
                await self._request_manual_intervention(error_context)
                return fallback_value
                
            else:
                return fallback_value
                
        except Exception as final_recovery_exception:
            self.logger.critical(f"Final recovery strategy failed: {final_recovery_exception}")
            return fallback_value
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking."""
        import uuid
        return f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _get_component_name(self, func: Callable) -> str:
        """Get component name for circuit breaker mapping."""
        if hasattr(func, '__module__'):
            module_name = func.__module__
            if 'database' in module_name or 'db' in module_name:
                return 'database_operations'
            elif 'model' in module_name and ('inference' in module_name or 'predict' in module_name):
                return 'model_inference'
            elif 'external' in module_name or 'api' in module_name or 'connector' in module_name:
                return 'external_apis'
            elif 'geological' in module_name or 'geo' in module_name:
                return 'geological_validation'
        
        return 'default'
    
    def _is_geological_function(self, func: Callable) -> bool:
        """Check if function is geological domain-related."""
        if hasattr(func, '__module__'):
            module_name = func.__module__.lower()
            return any(keyword in module_name for keyword in ['geological', 'geo', 'volcano', 'seismic', 'tectonic'])
        
        function_name = func.__name__.lower()
        return any(keyword in function_name for keyword in ['geological', 'geo', 'volcano', 'seismic', 'validate'])
    
    async def _validate_geological_result(self, result: Any, function_name: str) -> ValidationResult:
        """Validate geological domain results."""
        validation_errors = []
        validation_warnings = []
        
        try:
            # Validate numerical results
            if isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    validation_errors.append(f"Result contains NaN or infinite values: {result}")
                
                # Check reasonable geological ranges based on function type
                if 'temperature' in function_name.lower() and (result < -273 or result > 2000):
                    validation_warnings.append(f"Temperature value outside typical geological range: {result}")
                elif 'pressure' in function_name.lower() and result < 0:
                    validation_errors.append(f"Negative pressure value: {result}")
                elif 'depth' in function_name.lower() and result < 0:
                    validation_errors.append(f"Negative depth value: {result}")
            
            # Validate array results
            elif isinstance(result, np.ndarray):
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    validation_errors.append("Result array contains NaN or infinite values")
                
                if result.size == 0:
                    validation_warnings.append("Result array is empty")
            
            # Validate dictionary results
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                        validation_errors.append(f"Dictionary key '{key}' contains invalid value: {value}")
            
            return ValidationResult(
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                validation_metadata={'function_name': function_name, 'result_type': type(result).__name__}
            )
            
        except Exception as validation_exception:
            return ValidationResult(
                is_valid=False,
                validation_errors=[f"Validation process failed: {validation_exception}"],
                validation_warnings=[],
                validation_metadata={'validation_error': str(validation_exception)}
            )
    
    def _get_line_number(self) -> int:
        """Get current line number for error tracking."""
        try:
            return inspect.currentframe().f_back.f_back.f_lineno
        except:
            return 0
    
    def _sanitize_parameters(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Sanitize parameters for logging (remove sensitive data)."""
        try:
            sanitized = {}
            
            # Sanitize args
            for i, arg in enumerate(args):
                if isinstance(arg, (str, int, float, bool)):
                    sanitized[f'arg_{i}'] = arg
                elif isinstance(arg, (list, tuple)) and len(arg) < 10:
                    sanitized[f'arg_{i}'] = str(arg)[:200]  # Truncate long sequences
                else:
                    sanitized[f'arg_{i}'] = f"<{type(arg).__name__}>"
            
            # Sanitize kwargs
            for key, value in kwargs.items():
                # Skip sensitive keys
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret']):
                    sanitized[key] = '<REDACTED>'
                elif isinstance(value, (str, int, float, bool)):
                    sanitized[key] = value
                else:
                    sanitized[key] = f"<{type(value).__name__}>"
            
            return sanitized
            
        except Exception:
            return {'sanitization_error': 'Failed to sanitize parameters'}
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context."""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'timestamp': datetime.now().isoformat(),
                'process_id': os.getpid(),
                'thread_count': threading.active_count()
            }
        except Exception:
            return {'system_state_error': 'Failed to capture system state'}
    
    def _extract_geological_context(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract geological context from function parameters."""
        try:
            geological_context = {}
            
            # Look for common geological parameters
            geological_keys = ['latitude', 'longitude', 'depth', 'magnitude', 'temperature', 
                             'pressure', 'time', 'location', 'volcano_name', 'station_id']
            
            for key in geological_keys:
                if key in kwargs:
                    geological_context[key] = kwargs[key]
            
            # Look for geological data structures in args
            for i, arg in enumerate(args):
                if hasattr(arg, '__dict__') and any(attr in str(arg.__dict__).lower() for attr in geological_keys):
                    geological_context[f'geological_object_{i}'] = str(type(arg).__name__)
            
            return geological_context
            
        except Exception:
            return {'geological_context_error': 'Failed to extract geological context'}
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with comprehensive details."""
        try:
            # Update error statistics
            self.error_metrics['total_errors'] += 1
            
            category_count = self.error_metrics['errors_by_category'].get(error_context.category.value, 0)
            self.error_metrics['errors_by_category'][error_context.category.value] = category_count + 1
            
            severity_count = self.error_metrics['errors_by_severity'].get(error_context.severity.value, 0)
            self.error_metrics['errors_by_severity'][error_context.severity.value] = severity_count + 1
            
            # Store error history
            self.error_history.append(error_context)
            
            # Keep only last 1000 errors to manage memory
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
            
            # Log to standard logger
            log_level = self._get_log_level(error_context.severity)
            self.logger.log(log_level, f"Error {error_context.error_id}: {error_context.error_message}")
            
            # Log to Sentry if available
            if SENTRY_AVAILABLE:
                sentry_sdk.set_context("error_context", {
                    'error_id': error_context.error_id,
                    'severity': error_context.severity.value,
                    'category': error_context.category.value,
                    'recovery_strategy': error_context.recovery_strategy.value,
                    'geological_context': error_context.geological_context
                })
                capture_message(f"Advanced Error Handler: {error_context.error_message}")
            
            # Update error patterns
            pattern_key = f"{error_context.category.value}:{error_context.function_name}"
            self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
            
        except Exception as logging_exception:
            # Fallback logging
            print(f"Error logging failed: {logging_exception}")
            print(f"Original error: {error_context.error_message}")
    
    async def _log_successful_execution(self, func: Callable, execution_time: float, attempts: int):
        """Log successful execution for performance monitoring."""
        try:
            if attempts > 0:
                # Recovery was successful
                recovery_strategy = getattr(func, '_recovery_strategy', RecoveryStrategy.RETRY_WITH_BACKOFF)
                
                if recovery_strategy not in self.recovery_statistics:
                    self.recovery_statistics[recovery_strategy] = {'successes': 0, 'failures': 0}
                
                self.recovery_statistics[recovery_strategy]['successes'] += 1
                
                # Update recovery success rate
                total_recoveries = sum(stats['successes'] + stats['failures'] 
                                     for stats in self.recovery_statistics.values())
                total_successes = sum(stats['successes'] for stats in self.recovery_statistics.values())
                
                self.error_metrics['recovery_success_rate'] = total_successes / total_recoveries if total_recoveries > 0 else 0.0
            
        except Exception:
            pass  # Don't let success logging interfere with operation
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get appropriate log level for error severity."""
        level_mapping = {
            ErrorSeverity.LOW: logging.WARNING,
            ErrorSeverity.MEDIUM: logging.ERROR,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }
        return level_mapping.get(severity, logging.ERROR)
    
    async def _reset_to_default_state(self, error_context: ErrorContext):
        """Reset system component to default state."""
        try:
            # Implementation would depend on the specific component
            # This is a placeholder for component-specific reset logic
            self.logger.info(f"Resetting component to default state for error {error_context.error_id}")
            
        except Exception as reset_exception:
            self.logger.error(f"Failed to reset to default state: {reset_exception}")
    
    async def _enable_degraded_mode(self, error_context: ErrorContext):
        """Enable degraded mode operation."""
        try:
            # Implementation would enable limited functionality
            # This is a placeholder for degraded mode logic
            self.logger.warning(f"Enabling degraded mode for error {error_context.error_id}")
            
        except Exception as degraded_exception:
            self.logger.error(f"Failed to enable degraded mode: {degraded_exception}")
    
    async def _get_degraded_result(self, error_context: ErrorContext, fallback_value: Any) -> Any:
        """Get result in degraded mode."""
        try:
            # Return simplified or cached result
            return fallback_value
            
        except Exception:
            return None
    
    async def _request_manual_intervention(self, error_context: ErrorContext):
        """Request manual intervention for critical errors."""
        try:
            # This would integrate with alerting systems
            self.logger.critical(f"MANUAL INTERVENTION REQUIRED - Error {error_context.error_id}: {error_context.error_message}")
            
            # Could integrate with:
            # - PagerDuty for on-call alerts
            # - Slack/Teams for team notifications
            # - Email alerts for administrators
            # - Ticketing systems for tracking
            
        except Exception as alert_exception:
            self.logger.error(f"Failed to send manual intervention alert: {alert_exception}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            recent_errors = [err for err in self.error_history 
                           if err.timestamp > datetime.now() - timedelta(hours=24)]
            
            return {
                'total_errors': self.error_metrics['total_errors'],
                'errors_last_24h': len(recent_errors),
                'errors_by_category': self.error_metrics['errors_by_category'],
                'errors_by_severity': self.error_metrics['errors_by_severity'],
                'recovery_success_rate': self.error_metrics['recovery_success_rate'],
                'error_patterns': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
                'recovery_statistics': self.recovery_statistics,
                'circuit_breaker_status': {name: breaker.current_state for name, breaker in self.circuit_breakers.items()}
            }
            
        except Exception as stats_exception:
            return {'error': f"Failed to generate statistics: {stats_exception}"}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        try:
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'error_handler_active': True,
                'recent_error_rate': 0.0,
                'circuit_breakers': {},
                'system_resources': self._capture_system_state()
            }
            
            # Check recent error rate
            recent_errors = [err for err in self.error_history 
                           if err.timestamp > datetime.now() - timedelta(minutes=5)]
            health_status['recent_error_rate'] = len(recent_errors) / 5.0  # errors per minute
            
            # Check circuit breakers
            for name, breaker in self.circuit_breakers.items():
                health_status['circuit_breakers'][name] = {
                    'state': breaker.current_state,
                    'failure_count': breaker.fail_counter,
                    'last_failure': getattr(breaker, 'last_failure', None)
                }
            
            # Determine overall health
            if health_status['recent_error_rate'] > 10:  # More than 10 errors per minute
                health_status['status'] = 'degraded'
            
            critical_errors = [err for err in recent_errors if err.severity == ErrorSeverity.CRITICAL]
            if critical_errors:
                health_status['status'] = 'critical'
            
            return health_status
            
        except Exception as health_exception:
            return {
                'status': 'error',
                'timestamp': datetime.now().isoformat(),
                'error': str(health_exception)
            }

class GeologicalValidationError(Exception):
    """Custom exception for geological domain validation errors."""
    pass

class AdvancedValidator:
    """
    Advanced validation system with geological domain knowledge.
    
    Provides comprehensive data validation, schema validation, and domain-specific
    validation rules for geological AI systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__ + ".AdvancedValidator")
        
        # Validation rules and schemas
        self.validation_schemas: Dict[str, Any] = {}
        self.geological_constraints: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
        
        # Initialize validation components
        self._initialize_geological_constraints()
        self._initialize_validation_schemas()
        
        self.logger.info("Advanced Validator initialized")
    
    def _initialize_geological_constraints(self):
        """Initialize geological domain constraints."""
        self.geological_constraints = {
            'temperature': {
                'min': -273.15,  # Absolute zero
                'max': 6000.0,   # Core temperature
                'typical_surface': {'min': -100, 'max': 60},
                'typical_mantle': {'min': 1000, 'max': 4000}
            },
            'pressure': {
                'min': 0.0,
                'max': 400.0,   # Core pressure in GPa
                'typical_surface': {'min': 0.8, 'max': 1.2},  # Atmospheric
                'typical_crustal': {'min': 1.0, 'max': 10.0}
            },
            'depth': {
                'min': -11.0,   # Below sea level
                'max': 6371.0,  # Earth's radius
                'typical_crustal': {'min': 0, 'max': 70},
                'typical_mantle': {'min': 70, 'max': 2890}
            },
            'magnitude': {
                'min': -2.0,
                'max': 10.0,
                'typical_range': {'min': 0, 'max': 9}
            },
            'coordinates': {
                'latitude': {'min': -90.0, 'max': 90.0},
                'longitude': {'min': -180.0, 'max': 180.0}
            }
        }
    
    def _initialize_validation_schemas(self):
        """Initialize validation schemas for different data types."""
        if CERBERUS_AVAILABLE:
            # Seismic event schema
            self.validation_schemas['seismic_event'] = {
                'latitude': {'type': 'float', 'min': -90, 'max': 90, 'required': True},
                'longitude': {'type': 'float', 'min': -180, 'max': 180, 'required': True},
                'depth': {'type': 'float', 'min': 0, 'max': 1000, 'required': True},
                'magnitude': {'type': 'float', 'min': -2, 'max': 10, 'required': True},
                'timestamp': {'type': 'datetime', 'required': True},
                'location': {'type': 'string', 'maxlength': 200}
            }
            
            # Volcanic monitoring data schema
            self.validation_schemas['volcanic_data'] = {
                'volcano_name': {'type': 'string', 'maxlength': 100, 'required': True},
                'latitude': {'type': 'float', 'min': -90, 'max': 90, 'required': True},
                'longitude': {'type': 'float', 'min': -180, 'max': 180, 'required': True},
                'elevation': {'type': 'float', 'min': -500, 'max': 9000},
                'temperature': {'type': 'float', 'min': -100, 'max': 2000},
                'gas_emissions': {'type': 'dict'},
                'deformation': {'type': 'dict'},
                'alert_level': {'type': 'integer', 'min': 0, 'max': 5}
            }
    
    def validate_data(self, data: Any, schema_name: str) -> ValidationResult:
        """Validate data against specified schema."""
        try:
            if not CERBERUS_AVAILABLE or schema_name not in self.validation_schemas:
                return ValidationResult(
                    is_valid=True,
                    validation_warnings=[f"Schema validation not available for {schema_name}"]
                )
            
            validator = Validator(self.validation_schemas[schema_name])
            is_valid = validator.validate(data)
            
            return ValidationResult(
                is_valid=is_valid,
                validation_errors=list(validator.errors.values()) if not is_valid else [],
                validation_warnings=[],
                validation_metadata={'schema': schema_name, 'validator': 'cerberus'}
            )
            
        except Exception as validation_exception:
            return ValidationResult(
                is_valid=False,
                validation_errors=[f"Validation failed: {validation_exception}"],
                validation_warnings=[],
                validation_metadata={'error': str(validation_exception)}
            )
    
    def validate_geological_constraints(self, data: Dict[str, Any], data_type: str) -> ValidationResult:
        """Validate data against geological domain constraints."""
        try:
            validation_errors = []
            validation_warnings = []
            
            for field_name, value in data.items():
                constraint_key = self._map_field_to_constraint(field_name)
                
                if constraint_key in self.geological_constraints:
                    constraint = self.geological_constraints[constraint_key]
                    
                    # Check absolute bounds
                    if isinstance(value, (int, float)):
                        if value < constraint['min']:
                            validation_errors.append(f"{field_name} value {value} below minimum {constraint['min']}")
                        elif value > constraint['max']:
                            validation_errors.append(f"{field_name} value {value} above maximum {constraint['max']}")
                        
                        # Check typical ranges (warnings only)
                        typical_key = self._get_typical_range_key(data_type)
                        if typical_key in constraint:
                            typical_range = constraint[typical_key]
                            if value < typical_range['min'] or value > typical_range['max']:
                                validation_warnings.append(
                                    f"{field_name} value {value} outside typical range "
                                    f"[{typical_range['min']}, {typical_range['max']}]"
                                )
            
            return ValidationResult(
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                validation_metadata={'data_type': data_type, 'constraints_checked': len(data)}
            )
            
        except Exception as validation_exception:
            return ValidationResult(
                is_valid=False,
                validation_errors=[f"Geological constraint validation failed: {validation_exception}"],
                validation_warnings=[],
                validation_metadata={'error': str(validation_exception)}
            )
    
    def _map_field_to_constraint(self, field_name: str) -> str:
        """Map field name to constraint key."""
        field_mapping = {
            'temp': 'temperature',
            'temperature': 'temperature',
            'press': 'pressure',
            'pressure': 'pressure',
            'depth': 'depth',
            'elevation': 'depth',
            'mag': 'magnitude',
            'magnitude': 'magnitude',
            'lat': 'coordinates',
            'latitude': 'coordinates',
            'lon': 'coordinates',
            'longitude': 'coordinates'
        }
        
        return field_mapping.get(field_name.lower(), field_name.lower())
    
    def _get_typical_range_key(self, data_type: str) -> str:
        """Get typical range key based on data type."""
        if 'surface' in data_type.lower():
            return 'typical_surface'
        elif 'crustal' in data_type.lower():
            return 'typical_crustal'
        elif 'mantle' in data_type.lower():
            return 'typical_mantle'
        else:
            return 'typical_range'
    
    def add_custom_validator(self, name: str, validator_func: Callable) -> bool:
        """Add custom validation function."""
        try:
            self.custom_validators[name] = validator_func
            self.logger.info(f"Added custom validator: {name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom validator {name}: {e}")
            return False
    
    async def validate_model_output(self, output: Any, model_type: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate ML model output with domain-specific rules."""
        try:
            validation_errors = []
            validation_warnings = []
            
            # Check for basic data integrity
            if output is None:
                validation_errors.append("Model output is None")
                return ValidationResult(is_valid=False, validation_errors=validation_errors)
            
            # Validate numerical outputs
            if isinstance(output, (int, float)):
                if np.isnan(output) or np.isinf(output):
                    validation_errors.append(f"Model output contains invalid numerical value: {output}")
                
                # Model-specific validation
                if model_type == 'eruption_probability':
                    if not (0 <= output <= 1):
                        validation_errors.append(f"Eruption probability must be between 0 and 1, got {output}")
                elif model_type == 'magnitude_prediction':
                    if not (-2 <= output <= 10):
                        validation_warnings.append(f"Magnitude prediction outside typical range: {output}")
            
            # Validate array outputs
            elif isinstance(output, np.ndarray):
                if output.size == 0:
                    validation_warnings.append("Model output array is empty")
                elif np.any(np.isnan(output)) or np.any(np.isinf(output)):
                    validation_errors.append("Model output array contains NaN or infinite values")
            
            # Validate dictionary outputs
            elif isinstance(output, dict):
                required_keys = context.get('required_output_keys', [])
                for key in required_keys:
                    if key not in output:
                        validation_errors.append(f"Required output key missing: {key}")
            
            return ValidationResult(
                is_valid=len(validation_errors) == 0,
                validation_errors=validation_errors,
                validation_warnings=validation_warnings,
                validation_metadata={
                    'model_type': model_type,
                    'output_type': type(output).__name__,
                    'context': context
                }
            )
            
        except Exception as validation_exception:
            return ValidationResult(
                is_valid=False,
                validation_errors=[f"Model output validation failed: {validation_exception}"],
                validation_warnings=[],
                validation_metadata={'error': str(validation_exception)}
            )

# Factory function for creating the advanced error handling system
def create_advanced_error_handling_system(config: Dict[str, Any] = None) -> Tuple[AdvancedErrorHandler, AdvancedValidator]:
    """
    Factory function to create comprehensive error handling and validation system.
    
    Returns:
        Tuple of (AdvancedErrorHandler, AdvancedValidator) for complete error management
    """
    
    error_handler = AdvancedErrorHandler(config)
    validator = AdvancedValidator(config)
    
    return error_handler, validator

# Context managers for error handling
@asynccontextmanager
async def error_handling_context(error_handler: AdvancedErrorHandler,
                                category: ErrorCategory = ErrorCategory.INTEGRATION_FAILURE,
                                severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Async context manager for error handling."""
    try:
        yield error_handler
    except Exception as e:
        error_context = ErrorContext(
            error_id=error_handler._generate_error_id(),
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            module_name=__name__,
            function_name='error_handling_context',
            line_number=0,
            error_message=str(e),
            stack_trace=traceback.format_exc(),
            input_parameters={},
            system_state=error_handler._capture_system_state(),
            recovery_strategy=RecoveryStrategy.FAIL_FAST
        )
        error_handler._log_error(error_context)
        raise

@contextmanager
def validation_context(validator: AdvancedValidator, schema_name: str):
    """Context manager for validation operations."""
    try:
        yield validator
    except ValidationError as ve:
        validator.logger.error(f"Validation failed for schema {schema_name}: {ve}")
        raise
    except Exception as e:
        validator.logger.error(f"Validation context error: {e}")
        raise

if __name__ == "__main__":
    # Example usage and testing
    async def test_error_handling():
        """Test the advanced error handling system."""
        
        # Create error handling system
        error_handler, validator = create_advanced_error_handling_system({
            'environment': 'development',
            'version': '1.0.0'
        })
        
        # Test error handler decorator
        @error_handler.error_handler(
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.GEOLOGICAL_CONSTRAINT,
            recovery_strategy=RecoveryStrategy.RETRY_WITH_BACKOFF,
            max_retries=2,
            fallback_value=0.0
        )
        async def test_geological_function(temperature: float, depth: float):
            """Test function that might fail."""
            if temperature > 1000 and depth < 10:
                raise GeologicalValidationError("Invalid geological conditions")
            
            return temperature * depth * 0.1
        
        # Test successful execution
        result1 = await test_geological_function(800, 15)
        print(f"Successful result: {result1}")
        
        # Test error handling and recovery
        result2 = await test_geological_function(1200, 5)  # Should trigger error and recovery
        print(f"Recovered result: {result2}")
        
        # Test validator
        geological_data = {
            'latitude': 45.0,
            'longitude': -122.0,
            'depth': 15.0,
            'magnitude': 6.5,
            'temperature': 850.0
        }
        
        validation_result = validator.validate_geological_constraints(geological_data, 'surface')
        print(f"Validation result: {validation_result}")
        
        # Get error statistics
        stats = error_handler.get_error_statistics()
        print(f"Error statistics: {stats}")
        
        # Health check
        health = await error_handler.health_check()
        print(f"System health: {health}")
    
    # Run test
    if __name__ == "__main__":
        asyncio.run(test_error_handling())