"""
Advanced Circuit Breaker Implementation for Prometheus AI Platform
Implements circuit breaker and bulkhead patterns for service resilience
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import logging
import json
from datetime import datetime, timedelta
import aiohttp
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    """Advanced circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: tuple = (Exception,)
    success_threshold: int = 3
    timeout: float = 30.0
    
    # Advanced features
    sliding_window_size: int = 100
    minimum_throughput: int = 10
    error_rate_threshold: float = 0.5
    slow_call_duration_threshold: float = 10.0
    slow_call_rate_threshold: float = 0.5
    
    # Bulkhead configuration
    max_concurrent_calls: int = 100
    max_wait_duration: float = 5.0

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics and statistics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    slow_calls: int = 0
    rejected_calls: int = 0
    state_transitions: List[Dict[str, Any]] = field(default_factory=list)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None

class AdvancedCircuitBreaker:
    """
    Advanced Circuit Breaker with sliding window, bulkhead pattern,
    and comprehensive monitoring capabilities
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig, redis_client: Optional[redis.Redis] = None):
        self.name = name
        self.config = config
        self.redis_client = redis_client
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
        # Sliding window for advanced metrics
        self.call_history = []
        self.concurrent_calls = 0
        self.waiting_calls = 0
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        
        # Prometheus metrics
        self.calls_total = Counter(
            'circuit_breaker_calls_total',
            'Total number of calls through circuit breaker',
            ['name', 'result']
        )
        self.call_duration = Histogram(
            'circuit_breaker_call_duration_seconds',
            'Duration of calls through circuit breaker',
            ['name']
        )
        self.state_gauge = Gauge(
            'circuit_breaker_state',
            'Current state of circuit breaker (0=closed, 1=open, 2=half_open)',
            ['name']
        )
        self.concurrent_calls_gauge = Gauge(
            'circuit_breaker_concurrent_calls',
            'Number of concurrent calls',
            ['name']
        )
        
        # Semaphore for bulkhead pattern
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        
        logger.info(f"Initialized circuit breaker '{name}' with config: {config}")

    async def __call__(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        return await self.call(func, *args, **kwargs)

    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with advanced circuit breaker protection
        Implements bulkhead pattern and comprehensive monitoring
        """
        start_time = time.time()
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.metrics.rejected_calls += 1
                self.calls_total.labels(name=self.name, result='rejected').inc()
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is open")
        
        # Bulkhead pattern - limit concurrent calls
        if self.concurrent_calls >= self.config.max_concurrent_calls:
            self.metrics.rejected_calls += 1
            self.calls_total.labels(name=self.name, result='rejected').inc()
            raise BulkheadException(f"Too many concurrent calls to '{self.name}'")
        
        # Acquire semaphore with timeout
        try:
            await asyncio.wait_for(
                self.semaphore.acquire(),
                timeout=self.config.max_wait_duration
            )
        except asyncio.TimeoutError:
            self.metrics.rejected_calls += 1
            self.calls_total.labels(name=self.name, result='rejected').inc()
            raise BulkheadException(f"Timeout waiting for semaphore in '{self.name}'")
        
        self.concurrent_calls += 1
        self.concurrent_calls_gauge.labels(name=self.name).set(self.concurrent_calls)
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            
            # Record successful call
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except self.config.expected_exception as e:
            # Record failure
            duration = time.time() - start_time
            await self._record_failure(duration, e)
            raise
            
        except asyncio.TimeoutError:
            # Record timeout as failure
            duration = time.time() - start_time
            await self._record_failure(duration, TimeoutError("Function call timed out"))
            raise TimeoutError(f"Function call to '{self.name}' timed out after {self.config.timeout}s")
            
        finally:
            self.concurrent_calls -= 1
            self.concurrent_calls_gauge.labels(name=self.name).set(self.concurrent_calls)
            self.semaphore.release()
            
            # Record call duration
            duration = time.time() - start_time
            self.call_duration.labels(name=self.name).observe(duration)

    async def _record_success(self, duration: float):
        """Record successful call and update circuit breaker state"""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = datetime.now()
        
        # Check if it's a slow call
        if duration > self.config.slow_call_duration_threshold:
            self.metrics.slow_calls += 1
        
        # Add to sliding window
        self._add_to_sliding_window(True, duration)
        
        # Update Prometheus metrics
        self.calls_total.labels(name=self.name, result='success').inc()
        
        # State transitions
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        
        # Persist state if Redis is available
        await self._persist_state()
        
        logger.debug(f"Circuit breaker '{self.name}': Recorded success (duration: {duration:.2f}s)")

    async def _record_failure(self, duration: float, exception: Exception):
        """Record failed call and update circuit breaker state"""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = datetime.now()
        
        # Add to sliding window
        self._add_to_sliding_window(False, duration)
        
        # Update Prometheus metrics
        self.calls_total.labels(name=self.name, result='failure').inc()
        
        # State transitions
        if self.state in [CircuitState.CLOSED, CircuitState.HALF_OPEN]:
            self.failure_count += 1
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open()
        
        # Persist state if Redis is available
        await self._persist_state()
        
        logger.warning(f"Circuit breaker '{self.name}': Recorded failure - {exception}")

    def _add_to_sliding_window(self, success: bool, duration: float):
        """Add call result to sliding window for advanced metrics"""
        call_record = {
            'timestamp': time.time(),
            'success': success,
            'duration': duration,
            'slow': duration > self.config.slow_call_duration_threshold
        }
        
        self.call_history.append(call_record)
        
        # Maintain sliding window size
        if len(self.call_history) > self.config.sliding_window_size:
            self.call_history.pop(0)

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on advanced metrics"""
        # Simple threshold check
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Advanced sliding window analysis
        if len(self.call_history) < self.config.minimum_throughput:
            return False
        
        recent_calls = [call for call in self.call_history 
                       if time.time() - call['timestamp'] <= 60]  # Last minute
        
        if len(recent_calls) < self.config.minimum_throughput:
            return False
        
        # Calculate error rate
        failed_calls = sum(1 for call in recent_calls if not call['success'])
        error_rate = failed_calls / len(recent_calls)
        
        # Calculate slow call rate
        slow_calls = sum(1 for call in recent_calls if call['slow'])
        slow_call_rate = slow_calls / len(recent_calls)
        
        return (error_rate >= self.config.error_rate_threshold or 
                slow_call_rate >= self.config.slow_call_rate_threshold)

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open state"""
        if self.next_attempt_time is None:
            return True
        return time.time() >= self.next_attempt_time

    def _transition_to_open(self):
        """Transition circuit breaker to open state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.next_attempt_time = time.time() + self.config.recovery_timeout
        
        self._record_state_transition(old_state, self.state)
        self.state_gauge.labels(name=self.name).set(1)
        
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")

    def _transition_to_half_open(self):
        """Transition circuit breaker to half-open state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        
        self._record_state_transition(old_state, self.state)
        self.state_gauge.labels(name=self.name).set(2)
        
        logger.info(f"Circuit breaker '{self.name}' transitioned to half-open")

    def _transition_to_closed(self):
        """Transition circuit breaker to closed state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        
        self._record_state_transition(old_state, self.state)
        self.state_gauge.labels(name=self.name).set(0)
        
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")

    def _record_state_transition(self, old_state: CircuitState, new_state: CircuitState):
        """Record state transition for monitoring"""
        transition = {
            'timestamp': datetime.now().isoformat(),
            'from_state': old_state.value,
            'to_state': new_state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count
        }
        self.metrics.state_transitions.append(transition)
        
        # Keep only last 100 transitions
        if len(self.metrics.state_transitions) > 100:
            self.metrics.state_transitions.pop(0)

    async def _persist_state(self):
        """Persist circuit breaker state to Redis for cluster coordination"""
        if not self.redis_client:
            return
        
        try:
            state_data = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                'next_attempt_time': self.next_attempt_time,
                'metrics': {
                    'total_calls': self.metrics.total_calls,
                    'successful_calls': self.metrics.successful_calls,
                    'failed_calls': self.metrics.failed_calls,
                    'rejected_calls': self.metrics.rejected_calls
                }
            }
            
            await self.redis_client.setex(
                f"circuit_breaker:{self.name}",
                300,  # 5 minutes TTL
                json.dumps(state_data)
            )
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker state: {e}")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics"""
        recent_calls = [call for call in self.call_history 
                       if time.time() - call['timestamp'] <= 300]  # Last 5 minutes
        
        if recent_calls:
            success_rate = sum(1 for call in recent_calls if call['success']) / len(recent_calls)
            avg_duration = sum(call['duration'] for call in recent_calls) / len(recent_calls)
            slow_call_rate = sum(1 for call in recent_calls if call['slow']) / len(recent_calls)
        else:
            success_rate = 0.0
            avg_duration = 0.0
            slow_call_rate = 0.0
        
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'concurrent_calls': self.concurrent_calls,
            'metrics': {
                'total_calls': self.metrics.total_calls,
                'successful_calls': self.metrics.successful_calls,
                'failed_calls': self.metrics.failed_calls,
                'slow_calls': self.metrics.slow_calls,
                'rejected_calls': self.metrics.rejected_calls,
                'success_rate': success_rate,
                'average_duration': avg_duration,
                'slow_call_rate': slow_call_rate
            },
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'timeout': self.config.timeout,
                'max_concurrent_calls': self.config.max_concurrent_calls
            },
            'last_failure_time': self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            'last_success_time': self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            'state_transitions': self.metrics.state_transitions[-10:]  # Last 10 transitions
        }

class CircuitBreakerManager:
    """
    Centralized management of circuit breakers across the service mesh
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.redis_client = redis_client
        
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> AdvancedCircuitBreaker:
        """Get or create a circuit breaker"""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig()
            self.circuit_breakers[name] = AdvancedCircuitBreaker(name, config, self.redis_client)
        
        return self.circuit_breakers[name]
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all circuit breakers"""
        metrics = {}
        for name, cb in self.circuit_breakers.items():
            metrics[name] = await cb.get_metrics()
        return metrics
    
    async def reset_circuit_breaker(self, name: str) -> bool:
        """Manually reset a circuit breaker"""
        if name in self.circuit_breakers:
            cb = self.circuit_breakers[name]
            cb._transition_to_closed()
            await cb._persist_state()
            return True
        return False

# Custom Exceptions
class CircuitBreakerOpenException(Exception):
    """Raised when circuit breaker is open"""
    pass

class BulkheadException(Exception):
    """Raised when bulkhead limits are exceeded"""
    pass

# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to apply circuit breaker to async functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = CircuitBreakerManager()
            cb = manager.get_circuit_breaker(name, config)
            return await cb.call(func, *args, **kwargs)
        return wrapper
    return decorator

# Example usage and testing
async def example_usage():
    """Example of how to use the advanced circuit breaker"""
    
    # Create circuit breaker manager
    redis_client = await redis.from_url("redis://localhost:6379")
    manager = CircuitBreakerManager(redis_client)
    
    # Configure circuit breaker for a service
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30,
        timeout=10.0,
        max_concurrent_calls=50,
        error_rate_threshold=0.3,
        slow_call_duration_threshold=5.0
    )
    
    cb = manager.get_circuit_breaker("helios-core-service", config)
    
    # Example service call
    async def call_helios_service():
        async with aiohttp.ClientSession() as session:
            async with session.get("http://helios-core:8000/health") as response:
                return await response.json()
    
    try:
        result = await cb.call(call_helios_service)
        print(f"Service call successful: {result}")
    except CircuitBreakerOpenException:
        print("Circuit breaker is open - service unavailable")
    except BulkheadException:
        print("Too many concurrent calls - try again later")
    
    # Get metrics
    metrics = await cb.get_metrics()
    print(f"Circuit breaker metrics: {json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(example_usage())