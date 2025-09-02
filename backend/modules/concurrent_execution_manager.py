"""
Advanced Concurrent Execution Management Framework for Prometheus AI
Sophisticated task scheduling, resource allocation, and execution monitoring system
"""

import asyncio
import threading
import multiprocessing as mp
import concurrent.futures
import queue
import time
import uuid
import logging
import psutil
import signal
import resource
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle
import sys
import traceback
import os
from contextlib import contextmanager
import weakref
import gc
import json
from pathlib import Path
import warnings

# Third-party imports with fallbacks
try:
    import torch
    import torch.multiprocessing as torch_mp
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Some GPU-accelerated features will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available. Some numerical features will be limited.")

try:
    import dask
    from dask.distributed import Client, as_completed as dask_as_completed
    from dask.delayed import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

warnings.filterwarnings('ignore')

class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskState(Enum):
    """Task execution states"""
    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()

class ExecutionMode(Enum):
    """Execution modes for tasks"""
    SEQUENTIAL = "sequential"
    THREAD_POOL = "thread_pool"
    PROCESS_POOL = "process_pool"
    ASYNC_COROUTINE = "async_coroutine"
    DISTRIBUTED = "distributed"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"

class ResourceType(Enum):
    """System resource types"""
    CPU_CORES = "cpu_cores"
    MEMORY = "memory"
    GPU_MEMORY = "gpu_memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"

@dataclass
class ResourceRequirement:
    """Resource requirements for task execution"""
    cpu_cores: float = 1.0  # Number of CPU cores
    memory_mb: float = 512.0  # Memory in MB
    gpu_memory_mb: float = 0.0  # GPU memory in MB
    disk_io_mb_s: float = 0.0  # Disk I/O in MB/s
    network_io_mb_s: float = 0.0  # Network I/O in MB/s
    execution_time_estimate: Optional[float] = None  # Estimated time in seconds
    max_execution_time: Optional[float] = None  # Maximum allowed time
    
    def __post_init__(self):
        """Validate resource requirements"""
        if self.cpu_cores < 0:
            raise ValueError("CPU cores must be non-negative")
        if self.memory_mb < 0:
            raise ValueError("Memory must be non-negative")
        if self.gpu_memory_mb < 0:
            raise ValueError("GPU memory must be non-negative")

@dataclass
class TaskMetrics:
    """Metrics collected during task execution"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0
    error_count: int = 0
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'execution_time': self.execution_time,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_memory_usage_mb': self.gpu_memory_usage_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'io_read_mb': self.io_read_mb,
            'io_write_mb': self.io_write_mb,
            'error_count': self.error_count,
            'retry_count': self.retry_count
        }

@dataclass
class Task:
    """Executable task with comprehensive metadata"""
    task_id: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.THREAD_POOL
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    dependencies: Set[str] = field(default_factory=set)
    retry_config: Dict[str, Any] = field(default_factory=lambda: {'max_retries': 3, 'backoff_factor': 2.0})
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime fields
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Optional[Exception] = None
    metrics: TaskMetrics = field(default_factory=TaskMetrics)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    worker_id: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if not callable(self.function):
            raise ValueError("Task function must be callable")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("Task ID must be a non-empty string")

class ResourceMonitor:
    """System resource monitoring and allocation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.update_interval = 1.0  # seconds
        self.history_size = 1000
        self._monitoring = False
        self._monitor_thread = None
        self._resource_history = []
        self._lock = threading.Lock()
        
        # Resource limits
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.logical_cores = psutil.cpu_count(logical=True)
        self.total_memory = psutil.virtual_memory().total / (1024**2)  # MB
        self.available_cores = self.logical_cores
        self.available_memory = self.total_memory * 0.8  # Reserve 20%
        
        # GPU information
        self.gpu_count = 0
        self.gpu_memory = 0
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            if self.gpu_count > 0:
                self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
    
    def start_monitoring(self):
        """Start resource monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources continuously"""
        while self._monitoring:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
                network_io = psutil.net_io_counters()
                
                # GPU metrics
                gpu_memory_used = 0
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                
                resource_snapshot = {
                    'timestamp': datetime.now(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': (memory.total - memory.available) / (1024**2),
                    'memory_available_mb': memory.available / (1024**2),
                    'disk_read_mb': disk_io.read_bytes / (1024**2) if disk_io else 0,
                    'disk_write_mb': disk_io.write_bytes / (1024**2) if disk_io else 0,
                    'network_sent_mb': network_io.bytes_sent / (1024**2) if network_io else 0,
                    'network_recv_mb': network_io.bytes_recv / (1024**2) if network_io else 0,
                    'gpu_memory_used_mb': gpu_memory_used
                }
                
                with self._lock:
                    self._resource_history.append(resource_snapshot)
                    if len(self._resource_history) > self.history_size:
                        self._resource_history.pop(0)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def can_allocate_resources(self, requirements: ResourceRequirement) -> bool:
        """Check if resources can be allocated for requirements"""
        try:
            # CPU check
            if requirements.cpu_cores > self.available_cores:
                return False
            
            # Memory check
            current_memory = psutil.virtual_memory()
            available_memory_mb = current_memory.available / (1024**2)
            if requirements.memory_mb > available_memory_mb * 0.9:  # Leave 10% buffer
                return False
            
            # GPU memory check
            if requirements.gpu_memory_mb > 0:
                if not TORCH_AVAILABLE or not torch.cuda.is_available():
                    return False
                available_gpu_memory = (torch.cuda.get_device_properties(0).total_memory - 
                                      torch.cuda.memory_allocated()) / (1024**2)
                if requirements.gpu_memory_mb > available_gpu_memory * 0.9:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Resource allocation check failed: {e}")
            return False
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            gpu_usage = 0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_usage = (torch.cuda.memory_allocated() / 
                           torch.cuda.get_device_properties(0).total_memory) * 100
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'gpu_percent': gpu_usage,
                'available_cores': self.available_cores,
                'available_memory_mb': memory.available / (1024**2)
            }
        except Exception as e:
            self.logger.error(f"Failed to get current usage: {e}")
            return {}
    
    def get_resource_history(self, duration_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get resource usage history"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with self._lock:
            return [snapshot for snapshot in self._resource_history 
                   if snapshot['timestamp'] >= cutoff_time]

class TaskScheduler:
    """Intelligent task scheduling and prioritization"""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.logger = logging.getLogger(__name__)
        
        # Task queues by priority
        self.task_queues = {
            priority: queue.PriorityQueue() for priority in TaskPriority
        }
        
        # Task dependency tracking
        self.dependency_graph = {}  # task_id -> set of dependency task_ids
        self.dependents_graph = {}  # task_id -> set of dependent task_ids
        self.completed_tasks = set()
        
        # Scheduling state
        self._lock = threading.Lock()
        self._scheduler_thread = None
        self._scheduling = False
        
        # Statistics
        self.stats = {
            'total_scheduled': 0,
            'total_completed': 0,
            'total_failed': 0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0
        }
    
    def start_scheduling(self):
        """Start the task scheduler"""
        if self._scheduling:
            return
        
        self._scheduling = True
        self._scheduler_thread = threading.Thread(target=self._schedule_tasks, daemon=True)
        self._scheduler_thread.start()
        self.logger.info("Task scheduler started")
    
    def stop_scheduling(self):
        """Stop the task scheduler"""
        self._scheduling = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)
        self.logger.info("Task scheduler stopped")
    
    def submit_task(self, task: Task) -> bool:
        """Submit a task for scheduling"""
        try:
            with self._lock:
                # Update dependency tracking
                if task.dependencies:
                    self.dependency_graph[task.task_id] = task.dependencies.copy()
                    for dep_id in task.dependencies:
                        if dep_id not in self.dependents_graph:
                            self.dependents_graph[dep_id] = set()
                        self.dependents_graph[dep_id].add(task.task_id)
                
                # Queue task based on priority and readiness
                if self._are_dependencies_satisfied(task):
                    priority_value = (task.priority.value, task.created_at.timestamp())
                    self.task_queues[task.priority].put((priority_value, task))
                    task.state = TaskState.QUEUED
                    self.stats['total_scheduled'] += 1
                    self.logger.debug(f"Task {task.task_id} queued with priority {task.priority}")
                    return True
                else:
                    task.state = TaskState.PENDING
                    self.logger.debug(f"Task {task.task_id} waiting for dependencies")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def get_next_ready_task(self) -> Optional[Task]:
        """Get the next task ready for execution"""
        with self._lock:
            # Check all priority queues in order
            for priority in TaskPriority:
                if not self.task_queues[priority].empty():
                    try:
                        _, task = self.task_queues[priority].get_nowait()
                        
                        # Double-check resources are available
                        if self.resource_monitor.can_allocate_resources(task.resource_requirements):
                            task.state = TaskState.RUNNING
                            task.started_at = datetime.now()
                            return task
                        else:
                            # Put back in queue if resources not available
                            priority_value = (priority.value, task.created_at.timestamp())
                            self.task_queues[priority].put((priority_value, task))
                            
                    except queue.Empty:
                        continue
            
            return None
    
    def mark_task_completed(self, task: Task, success: bool):
        """Mark a task as completed and update dependencies"""
        with self._lock:
            task.completed_at = datetime.now()
            if task.started_at:
                task.metrics.execution_time = (task.completed_at - task.started_at).total_seconds()
            
            if success:
                task.state = TaskState.COMPLETED
                self.completed_tasks.add(task.task_id)
                self.stats['total_completed'] += 1
            else:
                task.state = TaskState.FAILED
                self.stats['total_failed'] += 1
            
            # Check for newly ready dependent tasks
            if task.task_id in self.dependents_graph:
                for dependent_id in self.dependents_graph[task.task_id]:
                    self._check_and_queue_dependent(dependent_id)
    
    def _are_dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied"""
        if not task.dependencies:
            return True
        return task.dependencies.issubset(self.completed_tasks)
    
    def _check_and_queue_dependent(self, task_id: str):
        """Check if a dependent task is ready to queue"""
        # This would need to be implemented with a task registry
        # For now, we'll assume tasks are managed externally
        pass
    
    def _schedule_tasks(self):
        """Main scheduling loop"""
        while self._scheduling:
            try:
                # Scheduling logic runs continuously
                # Actual task execution is handled by workers
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"Scheduling error: {e}")
                time.sleep(1.0)
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        status = {}
        with self._lock:
            for priority in TaskPriority:
                status[priority.name] = self.task_queues[priority].qsize()
        
        status['statistics'] = self.stats.copy()
        return status

class ExecutionWorker(ABC):
    """Abstract base class for task execution workers"""
    
    def __init__(self, worker_id: str, execution_mode: ExecutionMode):
        self.worker_id = worker_id
        self.execution_mode = execution_mode
        self.logger = logging.getLogger(__name__)
        self.is_busy = False
        self.current_task = None
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'created_at': datetime.now()
        }
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a task and return the result"""
        pass
    
    @abstractmethod
    def can_execute(self, task: Task) -> bool:
        """Check if this worker can execute the given task"""
        pass
    
    def update_stats(self, execution_time: float, success: bool):
        """Update worker statistics"""
        if success:
            self.stats['tasks_completed'] += 1
        else:
            self.stats['tasks_failed'] += 1
        self.stats['total_execution_time'] += execution_time

class ThreadPoolWorker(ExecutionWorker):
    """Thread pool-based task execution worker"""
    
    def __init__(self, worker_id: str, max_workers: int = None):
        super().__init__(worker_id, ExecutionMode.THREAD_POOL)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix=f"Worker-{worker_id}")
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using thread pool"""
        loop = asyncio.get_event_loop()
        
        try:
            self.is_busy = True
            self.current_task = task
            
            # Execute in thread pool
            start_time = time.time()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_function,
                task.function,
                task.args,
                task.kwargs
            )
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.update_stats(execution_time, False)
            raise e
        finally:
            self.is_busy = False
            self.current_task = None
    
    def can_execute(self, task: Task) -> bool:
        """Check if this worker can execute the task"""
        return (task.execution_mode == ExecutionMode.THREAD_POOL and 
                not self.is_busy)
    
    def _execute_function(self, func: Callable, args: Tuple, kwargs: Dict[str, Any]) -> Any:
        """Execute function with error handling"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Function execution failed: {e}")
            raise

class ProcessPoolWorker(ExecutionWorker):
    """Process pool-based task execution worker"""
    
    def __init__(self, worker_id: str, max_workers: int = None):
        super().__init__(worker_id, ExecutionMode.PROCESS_POOL)
        self.max_workers = max_workers or min(61, (os.cpu_count() or 1) + 4)
        
        # Use spawn method for better isolation
        ctx = mp.get_context('spawn')
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx)
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using process pool"""
        loop = asyncio.get_event_loop()
        
        try:
            self.is_busy = True
            self.current_task = task
            
            # Serialize task for inter-process communication
            serialized_task = self._serialize_task(task)
            
            # Execute in process pool
            start_time = time.time()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_serialized_task,
                serialized_task
            )
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.update_stats(execution_time, False)
            raise e
        finally:
            self.is_busy = False
            self.current_task = None
    
    def can_execute(self, task: Task) -> bool:
        """Check if this worker can execute the task"""
        return (task.execution_mode == ExecutionMode.PROCESS_POOL and 
                not self.is_busy)
    
    def _serialize_task(self, task: Task) -> bytes:
        """Serialize task for inter-process communication"""
        try:
            task_data = {
                'function': task.function,
                'args': task.args,
                'kwargs': task.kwargs,
                'task_id': task.task_id
            }
            return pickle.dumps(task_data)
        except Exception as e:
            self.logger.error(f"Task serialization failed: {e}")
            raise
    
    @staticmethod
    def _execute_serialized_task(serialized_task: bytes) -> Any:
        """Execute serialized task in separate process"""
        try:
            task_data = pickle.loads(serialized_task)
            func = task_data['function']
            args = task_data['args']
            kwargs = task_data['kwargs']
            
            return func(*args, **kwargs)
            
        except Exception as e:
            logging.error(f"Serialized task execution failed: {e}")
            raise

class AsyncWorker(ExecutionWorker):
    """Async coroutine-based task execution worker"""
    
    def __init__(self, worker_id: str):
        super().__init__(worker_id, ExecutionMode.ASYNC_COROUTINE)
        self.semaphore = asyncio.Semaphore(100)  # Limit concurrent coroutines
    
    async def execute_task(self, task: Task) -> Any:
        """Execute async task"""
        async with self.semaphore:
            try:
                self.is_busy = True
                self.current_task = task
                
                start_time = time.time()
                
                # Check if function is coroutine
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(*task.args, **task.kwargs)
                else:
                    # Wrap sync function in coroutine
                    result = await asyncio.to_thread(task.function, *task.args, **task.kwargs)
                
                execution_time = time.time() - start_time
                self.update_stats(execution_time, True)
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time if 'start_time' in locals() else 0
                self.update_stats(execution_time, False)
                raise e
            finally:
                self.is_busy = False
                self.current_task = None
    
    def can_execute(self, task: Task) -> bool:
        """Check if this worker can execute the task"""
        return (task.execution_mode == ExecutionMode.ASYNC_COROUTINE and 
                not self.is_busy)

class DistributedWorker(ExecutionWorker):
    """Distributed computing task execution worker"""
    
    def __init__(self, worker_id: str):
        super().__init__(worker_id, ExecutionMode.DISTRIBUTED)
        self.dask_client = None
        self.ray_initialized = False
        
        # Initialize distributed backends
        self._setup_dask()
        self._setup_ray()
    
    def _setup_dask(self):
        """Setup Dask distributed client"""
        if DASK_AVAILABLE:
            try:
                self.dask_client = Client('localhost:8786', timeout=5)
                self.logger.info("Dask distributed client initialized")
            except Exception as e:
                self.logger.warning(f"Dask setup failed: {e}")
    
    def _setup_ray(self):
        """Setup Ray distributed computing"""
        if RAY_AVAILABLE:
            try:
                ray.init(address='auto', ignore_reinit_error=True)
                self.ray_initialized = True
                self.logger.info("Ray distributed computing initialized")
            except Exception as e:
                self.logger.warning(f"Ray setup failed: {e}")
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using distributed computing"""
        try:
            self.is_busy = True
            self.current_task = task
            
            start_time = time.time()
            
            # Try Dask first, then Ray, then fallback
            if self.dask_client and 'dask' in task.metadata.get('preferred_backend', ['dask']):
                result = await self._execute_with_dask(task)
            elif self.ray_initialized and 'ray' in task.metadata.get('preferred_backend', ['ray']):
                result = await self._execute_with_ray(task)
            else:
                # Fallback to local execution
                result = task.function(*task.args, **task.kwargs)
            
            execution_time = time.time() - start_time
            self.update_stats(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time if 'start_time' in locals() else 0
            self.update_stats(execution_time, False)
            raise e
        finally:
            self.is_busy = False
            self.current_task = None
    
    async def _execute_with_dask(self, task: Task) -> Any:
        """Execute task with Dask"""
        if not self.dask_client:
            raise RuntimeError("Dask client not available")
        
        # Create delayed computation
        delayed_func = delayed(task.function)
        computation = delayed_func(*task.args, **task.kwargs)
        
        # Execute and get result
        future = self.dask_client.compute(computation)
        return await asyncio.wrap_future(future)
    
    async def _execute_with_ray(self, task: Task) -> Any:
        """Execute task with Ray"""
        if not self.ray_initialized:
            raise RuntimeError("Ray not available")
        
        # Create remote function
        @ray.remote
        def remote_function(func, args, kwargs):
            return func(*args, **kwargs)
        
        # Execute remotely
        future = remote_function.remote(task.function, task.args, task.kwargs)
        return ray.get(future)
    
    def can_execute(self, task: Task) -> bool:
        """Check if this worker can execute the task"""
        return (task.execution_mode == ExecutionMode.DISTRIBUTED and 
                not self.is_busy and 
                (self.dask_client is not None or self.ray_initialized))

class ConcurrentExecutionManager:
    """Main concurrent execution management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.resource_monitor = ResourceMonitor()
        self.task_scheduler = TaskScheduler(self.resource_monitor)
        
        # Task registry
        self.tasks = {}  # task_id -> Task
        self.task_results = {}  # task_id -> result
        
        # Worker management
        self.workers = []
        self.worker_assignments = {}  # task_id -> worker_id
        
        # Execution control
        self._running = False
        self._execution_loop = None
        self._shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'throughput_per_second': 0.0,
            'resource_utilization': {}
        }
        
        # Initialize workers
        self._initialize_workers()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_thread_workers': min(32, (os.cpu_count() or 1) + 4),
            'max_process_workers': min(61, (os.cpu_count() or 1) + 4),
            'max_async_workers': 100,
            'enable_distributed': True,
            'enable_gpu': True,
            'task_timeout_default': 3600,  # 1 hour
            'max_retries_default': 3,
            'resource_monitoring': True,
            'performance_profiling': True
        }
    
    def _initialize_workers(self):
        """Initialize execution workers"""
        # Thread pool workers
        for i in range(min(4, self.config['max_thread_workers'] // 8)):
            worker = ThreadPoolWorker(f"thread-{i}", self.config['max_thread_workers'] // 4)
            self.workers.append(worker)
        
        # Process pool workers
        for i in range(min(2, self.config['max_process_workers'] // 16)):
            worker = ProcessPoolWorker(f"process-{i}", self.config['max_process_workers'] // 2)
            self.workers.append(worker)
        
        # Async workers
        for i in range(min(4, self.config['max_async_workers'] // 25)):
            worker = AsyncWorker(f"async-{i}")
            self.workers.append(worker)
        
        # Distributed worker
        if self.config['enable_distributed']:
            distributed_worker = DistributedWorker("distributed-0")
            self.workers.append(distributed_worker)
        
        self.logger.info(f"Initialized {len(self.workers)} execution workers")
    
    async def start(self):
        """Start the concurrent execution manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring and scheduling
        self.resource_monitor.start_monitoring()
        self.task_scheduler.start_scheduling()
        
        # Start execution loop
        self._execution_loop = asyncio.create_task(self._execution_loop_async())
        
        self.logger.info("Concurrent execution manager started")
    
    async def stop(self):
        """Stop the concurrent execution manager"""
        if not self._running:
            return
        
        self._running = False
        self._shutdown_event.set()
        
        # Stop components
        self.task_scheduler.stop_scheduling()
        self.resource_monitor.stop_monitoring()
        
        # Wait for execution loop to complete
        if self._execution_loop:
            await self._execution_loop
        
        # Cleanup workers
        for worker in self.workers:
            if hasattr(worker, 'executor'):
                worker.executor.shutdown(wait=True, cancel_futures=True)
        
        self.logger.info("Concurrent execution manager stopped")
    
    async def submit_task(self, 
                         function: Callable,
                         args: Tuple = (),
                         kwargs: Dict[str, Any] = None,
                         task_id: str = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         execution_mode: ExecutionMode = ExecutionMode.THREAD_POOL,
                         resource_requirements: ResourceRequirement = None,
                         dependencies: Set[str] = None,
                         retry_config: Dict[str, Any] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a task for execution"""
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Create task
        task = Task(
            task_id=task_id,
            function=function,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority,
            execution_mode=execution_mode,
            resource_requirements=resource_requirements or ResourceRequirement(),
            dependencies=dependencies or set(),
            retry_config=retry_config or {'max_retries': self.config['max_retries_default']},
            metadata=metadata or {}
        )
        
        # Register task
        self.tasks[task_id] = task
        self.metrics['total_tasks'] += 1
        
        # Submit to scheduler
        success = self.task_scheduler.submit_task(task)
        
        if success:
            self.logger.info(f"Task {task_id} submitted successfully")
        else:
            self.logger.error(f"Failed to submit task {task_id}")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        return {
            'task_id': task.task_id,
            'state': task.state.name,
            'priority': task.priority.name,
            'execution_mode': task.execution_mode.value,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'worker_id': task.worker_id,
            'metrics': task.metrics.to_dict(),
            'error': str(task.error) if task.error else None
        }
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of a completed task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.tasks[task_id]
        
        # Wait for completion if still running
        start_time = time.time()
        while task.state in [TaskState.PENDING, TaskState.QUEUED, TaskState.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            await asyncio.sleep(0.1)
        
        if task.state == TaskState.COMPLETED:
            return task.result
        elif task.state == TaskState.FAILED:
            raise task.error or RuntimeError(f"Task {task_id} failed")
        else:
            raise RuntimeError(f"Task {task_id} in unexpected state: {task.state}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.state in [TaskState.PENDING, TaskState.QUEUED]:
            task.state = TaskState.CANCELLED
            self.logger.info(f"Task {task_id} cancelled")
            return True
        elif task.state == TaskState.RUNNING:
            # This would require more sophisticated cancellation logic
            task.state = TaskState.CANCELLED
            self.logger.info(f"Running task {task_id} marked for cancellation")
            return True
        
        return False
    
    async def _execution_loop_async(self):
        """Main execution loop"""
        while self._running and not self._shutdown_event.is_set():
            try:
                # Get next ready task
                task = self.task_scheduler.get_next_ready_task()
                
                if task is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Find available worker
                worker = self._find_available_worker(task)
                if worker is None:
                    # Put task back in queue
                    task.state = TaskState.QUEUED
                    priority_value = (task.priority.value, task.created_at.timestamp())
                    self.task_scheduler.task_queues[task.priority].put((priority_value, task))
                    await asyncio.sleep(0.1)
                    continue
                
                # Execute task
                asyncio.create_task(self._execute_task_with_worker(task, worker))
                
            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                await asyncio.sleep(1.0)
    
    def _find_available_worker(self, task: Task) -> Optional[ExecutionWorker]:
        """Find an available worker for the task"""
        for worker in self.workers:
            if worker.can_execute(task):
                return worker
        return None
    
    async def _execute_task_with_worker(self, task: Task, worker: ExecutionWorker):
        """Execute a task with the specified worker"""
        try:
            # Record worker assignment
            self.worker_assignments[task.task_id] = worker.worker_id
            task.worker_id = worker.worker_id
            
            # Execute task with timeout
            timeout = task.resource_requirements.max_execution_time or self.config['task_timeout_default']
            
            result = await asyncio.wait_for(
                worker.execute_task(task),
                timeout=timeout
            )
            
            # Store result and mark completed
            task.result = result
            self.task_results[task.task_id] = result
            
            self.task_scheduler.mark_task_completed(task, True)
            self.metrics['completed_tasks'] += 1
            
            self.logger.info(f"Task {task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            task.error = TimeoutError(f"Task exceeded timeout of {timeout} seconds")
            task.state = TaskState.TIMEOUT
            self.task_scheduler.mark_task_completed(task, False)
            self.metrics['failed_tasks'] += 1
            self.logger.error(f"Task {task.task_id} timed out")
            
        except Exception as e:
            task.error = e
            
            # Retry logic
            if task.metrics.retry_count < task.retry_config.get('max_retries', 0):
                task.metrics.retry_count += 1
                backoff_factor = task.retry_config.get('backoff_factor', 2.0)
                delay = backoff_factor ** task.metrics.retry_count
                
                self.logger.warning(f"Task {task.task_id} failed, retrying in {delay}s (attempt {task.metrics.retry_count})")
                
                # Schedule retry
                asyncio.create_task(self._retry_task(task, delay))
            else:
                self.task_scheduler.mark_task_completed(task, False)
                self.metrics['failed_tasks'] += 1
                self.logger.error(f"Task {task.task_id} failed permanently: {e}")
    
    async def _retry_task(self, task: Task, delay: float):
        """Retry a failed task after delay"""
        await asyncio.sleep(delay)
        
        # Reset task state and resubmit
        task.state = TaskState.PENDING
        task.started_at = None
        task.completed_at = None
        task.worker_id = None
        
        self.task_scheduler.submit_task(task)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        resource_usage = self.resource_monitor.get_current_usage()
        queue_status = self.task_scheduler.get_queue_status()
        
        worker_status = {}
        for worker in self.workers:
            worker_status[worker.worker_id] = {
                'execution_mode': worker.execution_mode.value,
                'is_busy': worker.is_busy,
                'current_task': worker.current_task.task_id if worker.current_task else None,
                'stats': worker.stats.copy()
            }
        
        return {
            'running': self._running,
            'total_workers': len(self.workers),
            'busy_workers': sum(1 for w in self.workers if w.is_busy),
            'resource_usage': resource_usage,
            'queue_status': queue_status,
            'worker_status': worker_status,
            'system_metrics': self.metrics.copy(),
            'tasks_by_state': {
                state.name: sum(1 for task in self.tasks.values() if task.state == state)
                for state in TaskState
            }
        }
    
    def export_metrics(self, filepath: str = None) -> Dict[str, Any]:
        """Export comprehensive metrics"""
        if filepath is None:
            filepath = f"execution_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'resource_history': self.resource_monitor.get_resource_history(60),  # Last hour
            'task_details': {
                task_id: {
                    'state': task.state.name,
                    'metrics': task.metrics.to_dict(),
                    'execution_mode': task.execution_mode.value,
                    'priority': task.priority.name
                }
                for task_id, task in self.tasks.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {filepath}")
        return metrics_data

# Example usage and testing functions
async def example_concurrent_execution():
    """Example of using the concurrent execution framework"""
    
    print("=== Concurrent Execution Manager Example ===")
    
    # Initialize manager
    manager = ConcurrentExecutionManager({
        'max_thread_workers': 8,
        'max_process_workers': 4,
        'enable_distributed': True,
        'performance_profiling': True
    })
    
    await manager.start()
    
    # Example functions to execute
    def cpu_intensive_task(n: int, task_name: str = "unnamed") -> dict:
        """CPU intensive computation"""
        result = sum(i * i for i in range(n))
        return {
            'task_name': task_name,
            'result': result,
            'computed_range': n
        }
    
    def io_intensive_task(delay: float, data: dict = None) -> dict:
        """I/O intensive simulation"""
        time.sleep(delay)
        return {
            'delay': delay,
            'data': data or {},
            'completed_at': datetime.now().isoformat()
        }
    
    async def async_task(iterations: int, prefix: str = "async") -> dict:
        """Async task example"""
        results = []
        for i in range(iterations):
            await asyncio.sleep(0.01)  # Simulate async work
            results.append(f"{prefix}-{i}")
        return {'results': results, 'count': len(results)}
    
    print("1. Submitting various types of tasks...")
    
    # Submit different types of tasks
    task_ids = []
    
    # CPU-intensive tasks
    for i in range(3):
        task_id = await manager.submit_task(
            function=cpu_intensive_task,
            args=(10000 + i * 1000,),
            kwargs={'task_name': f'cpu_task_{i}'},
            priority=TaskPriority.HIGH,
            execution_mode=ExecutionMode.PROCESS_POOL,
            resource_requirements=ResourceRequirement(cpu_cores=1.0, memory_mb=100)
        )
        task_ids.append(task_id)
    
    # I/O intensive tasks
    for i in range(4):
        task_id = await manager.submit_task(
            function=io_intensive_task,
            args=(0.5,),
            kwargs={'data': {'task_index': i, 'type': 'io_intensive'}},
            priority=TaskPriority.NORMAL,
            execution_mode=ExecutionMode.THREAD_POOL
        )
        task_ids.append(task_id)
    
    # Async tasks
    for i in range(2):
        task_id = await manager.submit_task(
            function=async_task,
            args=(5,),
            kwargs={'prefix': f'async_task_{i}'},
            priority=TaskPriority.NORMAL,
            execution_mode=ExecutionMode.ASYNC_COROUTINE
        )
        task_ids.append(task_id)
    
    print(f"Submitted {len(task_ids)} tasks")
    
    # Monitor progress
    print("2. Monitoring task execution...")
    completed_tasks = 0
    start_time = time.time()
    
    while completed_tasks < len(task_ids):
        status = manager.get_system_status()
        completed_count = status['tasks_by_state'].get('COMPLETED', 0)
        failed_count = status['tasks_by_state'].get('FAILED', 0)
        running_count = status['tasks_by_state'].get('RUNNING', 0)
        
        print(f"  Progress: {completed_count + failed_count}/{len(task_ids)} tasks completed, {running_count} running")
        print(f"  Resource usage: CPU {status['resource_usage'].get('cpu_percent', 0):.1f}%, Memory {status['resource_usage'].get('memory_percent', 0):.1f}%")
        
        await asyncio.sleep(1.0)
        completed_tasks = completed_count + failed_count
    
    execution_time = time.time() - start_time
    
    # Collect results
    print("3. Collecting results...")
    results = {}
    for task_id in task_ids:
        try:
            result = await manager.get_task_result(task_id)
            status = await manager.get_task_status(task_id)
            results[task_id] = {
                'result': result,
                'execution_time': status['metrics']['execution_time'],
                'worker_id': status['worker_id']
            }
        except Exception as e:
            results[task_id] = {'error': str(e)}
    
    # Display summary
    print("4. Execution Summary:")
    print(f"   Total execution time: {execution_time:.2f} seconds")
    print(f"   Successful tasks: {sum(1 for r in results.values() if 'result' in r)}")
    print(f"   Failed tasks: {sum(1 for r in results.values() if 'error' in r)}")
    
    # Show worker utilization
    final_status = manager.get_system_status()
    print("5. Worker Statistics:")
    for worker_id, worker_info in final_status['worker_status'].items():
        stats = worker_info['stats']
        print(f"   {worker_id} ({worker_info['execution_mode']}): "
              f"{stats['tasks_completed']} completed, "
              f"{stats['tasks_failed']} failed, "
              f"{stats['total_execution_time']:.2f}s total")
    
    # Export metrics
    print("6. Exporting metrics...")
    metrics = manager.export_metrics()
    print(f"   Metrics exported with {len(metrics['task_details'])} task details")
    
    # Cleanup
    await manager.stop()
    print("=== Concurrent Execution Example Completed ===")

if __name__ == "__main__":
    print("=== Advanced Concurrent Execution Manager ===")
    
    # System information
    print(f"System: {psutil.cpu_count(logical=False)} physical cores, {psutil.cpu_count()} logical cores")
    print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"CUDA available: {TORCH_AVAILABLE and torch.cuda.is_available()}")
    print(f"Dask available: {DASK_AVAILABLE}")
    print(f"Ray available: {RAY_AVAILABLE}")
    
    # Run comprehensive example
    asyncio.run(example_concurrent_execution())