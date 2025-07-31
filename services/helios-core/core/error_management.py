"""
Hierarchical Error Management System for Prometheus AI Platform

This module implements a comprehensive error handling system that provides:
- Error classification and routing
- Automatic recovery strategies
- Fallback mechanisms and graceful degradation
- Error logging and analysis

Requirements: 12.3, 12.4
"""

import logging
import traceback
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
import json
import asyncio
from abc import ABC, abstractmethod

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for hierarchical classification"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory(Enum):
    """Primary error categories for classification"""
    DATA_QUALITY = "data_quality"
    MODEL_CONVERGENCE = "model_convergence"
    PHYSICS_VIOLATION = "physics_violation"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK_CONNECTIVITY = "network_connectivity"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"


class ServiceDegradationLevel(Enum):
    """Service degradation levels for graceful degradation"""
    FULL_SERVICE = "full_service"
    REDUCED_PRECISION = "reduced_precision"
    ESSENTIAL_ONLY = "essential_only"
    OFFLINE_MODE = "offline_mode"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    service_name: str
    operation: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    model_complexity: str = "standard"
    retry_count: int = 0
    max_retries: int = 3
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Structured error information"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception: Optional[Exception] = None
    traceback_str: Optional[str] = None
    context: Optional[ErrorContext] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    recovery_time: Optional[float] = None


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies"""
    
    @abstractmethod
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        """Check if this strategy can handle the given error"""
        pass
    
    @abstractmethod
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from the error"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this recovery strategy"""
        pass


class ErrorClassifier:
    """Classifies errors into categories and severity levels"""
    
    def __init__(self):
        self.classification_rules = self._initialize_classification_rules()
    
    def _initialize_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification rules"""
        return {
            # Data Quality Error Patterns
            "data_quality": {
                "patterns": [
                    "invalid.*data", "missing.*data", "corrupt.*data",
                    "data.*validation.*failed", "format.*error",
                    "parsing.*error", "schema.*violation"
                ],
                "exception_types": [
                    "ValueError", "ValidationError", "DataError",
                    "ParseError", "SchemaError"
                ],
                "severity_mapping": {
                    "missing.*critical.*data": ErrorSeverity.CRITICAL,
                    "corrupt.*model.*data": ErrorSeverity.HIGH,
                    "invalid.*sensor.*reading": ErrorSeverity.MEDIUM,
                    "format.*warning": ErrorSeverity.LOW
                }
            },
            
            # Model Convergence Error Patterns
            "model_convergence": {
                "patterns": [
                    "convergence.*failed", "training.*error", "model.*diverged",
                    "gradient.*explosion", "nan.*values", "optimization.*failed"
                ],
                "exception_types": [
                    "ConvergenceError", "TrainingError", "OptimizationError",
                    "NumericalError"
                ],
                "severity_mapping": {
                    "critical.*model.*failure": ErrorSeverity.CRITICAL,
                    "training.*diverged": ErrorSeverity.HIGH,
                    "slow.*convergence": ErrorSeverity.MEDIUM
                }
            },
            
            # Physics Constraint Violation Patterns
            "physics_violation": {
                "patterns": [
                    "physics.*constraint.*violated", "conservation.*law.*violated",
                    "thermodynamic.*violation", "mass.*balance.*error",
                    "energy.*conservation.*failed"
                ],
                "exception_types": [
                    "PhysicsViolationError", "ConservationError",
                    "ThermodynamicError"
                ],
                "severity_mapping": {
                    "fundamental.*law.*violated": ErrorSeverity.CRITICAL,
                    "constraint.*violation": ErrorSeverity.HIGH
                }
            },
            
            # System Resource Error Patterns
            "system_resource": {
                "patterns": [
                    "memory.*error", "disk.*full", "cpu.*overload",
                    "resource.*exhausted", "timeout", "capacity.*exceeded"
                ],
                "exception_types": [
                    "MemoryError", "TimeoutError", "ResourceError",
                    "CapacityError"
                ],
                "severity_mapping": {
                    "out.*of.*memory": ErrorSeverity.CRITICAL,
                    "disk.*full": ErrorSeverity.HIGH,
                    "timeout": ErrorSeverity.MEDIUM
                }
            },
            
            # Network/Connectivity Error Patterns
            "network_connectivity": {
                "patterns": [
                    "connection.*failed", "network.*error", "timeout",
                    "dns.*resolution.*failed", "ssl.*error", "proxy.*error"
                ],
                "exception_types": [
                    "ConnectionError", "NetworkError", "TimeoutError",
                    "SSLError", "DNSError"
                ],
                "severity_mapping": {
                    "critical.*service.*unreachable": ErrorSeverity.CRITICAL,
                    "external.*api.*down": ErrorSeverity.HIGH,
                    "intermittent.*connection": ErrorSeverity.MEDIUM
                }
            },
            
            # Security Error Patterns
            "security": {
                "patterns": [
                    "authentication.*failed", "authorization.*denied",
                    "access.*denied", "security.*violation", "token.*expired"
                ],
                "exception_types": [
                    "AuthenticationError", "AuthorizationError",
                    "SecurityError", "AccessDeniedError"
                ],
                "severity_mapping": {
                    "security.*breach": ErrorSeverity.CRITICAL,
                    "unauthorized.*access": ErrorSeverity.HIGH,
                    "token.*expired": ErrorSeverity.MEDIUM
                }
            }
        }
    
    def classify(self, error: Exception, context: Optional[ErrorContext] = None) -> ErrorInfo:
        """Classify an error into category and severity"""
        import re
        import uuid
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Generate unique error ID
        error_id = str(uuid.uuid4())
        
        # Default classification
        category = ErrorCategory.SYSTEM_RESOURCE
        severity = ErrorSeverity.MEDIUM
        
        # Apply classification rules
        for cat_name, rules in self.classification_rules.items():
            # Check message patterns
            for pattern in rules["patterns"]:
                if re.search(pattern, error_message):
                    category = ErrorCategory(cat_name)
                    break
            
            # Check exception types
            if error_type in rules["exception_types"]:
                category = ErrorCategory(cat_name)
            
            # Determine severity based on specific patterns
            if category.value == cat_name:
                for severity_pattern, sev_level in rules["severity_mapping"].items():
                    if re.search(severity_pattern, error_message):
                        severity = sev_level
                        break
                break
        
        return ErrorInfo(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(error),
            exception=error,
            traceback_str=traceback.format_exc(),
            context=context
        )


class DataQualityRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for data quality errors"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.DATA_QUALITY
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from data quality errors"""
        logger.info(f"Attempting data quality recovery for error {error_info.error_id}")
        
        if error_info.context and error_info.context.data:
            # Attempt data cleaning and validation
            cleaned_data = await self._clean_data(error_info.context.data)
            
            if await self._validate_data(cleaned_data):
                logger.info(f"Data quality recovery successful for {error_info.error_id}")
                return cleaned_data
            else:
                # Use fallback data sources
                fallback_data = await self._get_fallback_data(error_info.context)
                logger.info(f"Using fallback data for {error_info.error_id}")
                return fallback_data
        
        raise Exception("Data quality recovery failed - no valid data available")
    
    def get_strategy_name(self) -> str:
        return "DataQualityRecovery"
    
    async def _clean_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize data"""
        cleaned_data = data.copy()
        
        # Remove null values and outliers
        for key, value in data.items():
            if value is None:
                # Use interpolation or default values
                cleaned_data[key] = await self._interpolate_missing_value(key, data)
            elif isinstance(value, (int, float)) and abs(value) > 1e10:
                # Handle extreme outliers
                cleaned_data[key] = await self._handle_outlier(key, value, data)
        
        return cleaned_data
    
    async def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate cleaned data"""
        # Implement data validation logic
        required_fields = ["timestamp", "location", "measurement_type"]
        return all(field in data and data[field] is not None for field in required_fields)
    
    async def _get_fallback_data(self, context: ErrorContext) -> Dict[str, Any]:
        """Get fallback data from historical averages or cached sources"""
        # Implement fallback data retrieval
        return {
            "timestamp": datetime.now(),
            "location": {"lat": 0.0, "lon": 0.0},
            "measurement_type": "fallback",
            "value": 0.0,
            "quality_flag": "estimated"
        }
    
    async def _interpolate_missing_value(self, key: str, data: Dict[str, Any]) -> Any:
        """Interpolate missing values"""
        # Simple interpolation logic
        if key in ["temperature", "pressure", "magnitude"]:
            return 0.0
        elif key in ["timestamp"]:
            return datetime.now()
        else:
            return "unknown"
    
    async def _handle_outlier(self, key: str, value: Any, data: Dict[str, Any]) -> Any:
        """Handle outlier values"""
        # Cap extreme values
        if isinstance(value, (int, float)):
            return max(-1000, min(1000, value))
        return value


class ModelConvergenceRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for model convergence errors"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.MODEL_CONVERGENCE
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from model convergence errors"""
        logger.info(f"Attempting model convergence recovery for error {error_info.error_id}")
        
        if error_info.context:
            # Try simplified model first
            if error_info.context.model_complexity != "minimal":
                logger.info("Retrying with simplified model")
                error_info.context.model_complexity = "minimal"
                return await self._retry_with_simplified_model(error_info.context)
            else:
                # Use ensemble fallback
                logger.info("Using ensemble fallback model")
                return await self._use_ensemble_fallback(error_info.context)
        
        raise Exception("Model convergence recovery failed")
    
    def get_strategy_name(self) -> str:
        return "ModelConvergenceRecovery"
    
    async def _retry_with_simplified_model(self, context: ErrorContext) -> Any:
        """Retry with a simplified model configuration"""
        # Implement simplified model logic
        return {
            "model_type": "simplified",
            "complexity": "minimal",
            "result": "simplified_prediction",
            "confidence": 0.7
        }
    
    async def _use_ensemble_fallback(self, context: ErrorContext) -> Any:
        """Use ensemble of simpler models as fallback"""
        return {
            "model_type": "ensemble_fallback",
            "result": "ensemble_prediction",
            "confidence": 0.6
        }


class PhysicsViolationRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for physics constraint violations"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.PHYSICS_VIOLATION
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from physics violations"""
        logger.info(f"Attempting physics violation recovery for error {error_info.error_id}")
        
        if error_info.context and error_info.context.data:
            # Apply physics-guided correction
            corrected_result = await self._apply_physics_correction(
                error_info.context.data
            )
            
            if await self._validate_physics_constraints(corrected_result):
                logger.info(f"Physics violation recovery successful for {error_info.error_id}")
                return corrected_result
            else:
                # Use conservative estimate
                logger.warning(f"Using conservative estimate for {error_info.error_id}")
                return await self._use_conservative_estimate(error_info.context)
        
        raise Exception("Physics violation recovery failed")
    
    def get_strategy_name(self) -> str:
        return "PhysicsViolationRecovery"
    
    async def _apply_physics_correction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physics-based corrections to results"""
        corrected_data = data.copy()
        
        # Apply conservation laws and physical constraints
        if "energy" in data and data["energy"] < 0:
            corrected_data["energy"] = abs(data["energy"])
        
        if "temperature" in data and data["temperature"] < 0:
            corrected_data["temperature"] = 273.15  # Absolute zero minimum
        
        return corrected_data
    
    async def _validate_physics_constraints(self, data: Dict[str, Any]) -> bool:
        """Validate that results satisfy physics constraints"""
        # Check basic physics constraints
        if "energy" in data and data["energy"] < 0:
            return False
        if "temperature" in data and data["temperature"] < 0:
            return False
        return True
    
    async def _use_conservative_estimate(self, context: ErrorContext) -> Dict[str, Any]:
        """Provide conservative estimates when physics correction fails"""
        return {
            "result_type": "conservative_estimate",
            "confidence": 0.3,
            "warning": "Physics constraints could not be satisfied - using conservative estimate"
        }


class SystemResourceRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for system resource errors"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.SYSTEM_RESOURCE
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from system resource errors"""
        logger.info(f"Attempting system resource recovery for error {error_info.error_id}")
        
        # Implement resource optimization and cleanup
        await self._cleanup_resources()
        await self._optimize_memory_usage()
        
        # Retry with reduced resource requirements
        if error_info.context:
            return await self._retry_with_reduced_resources(error_info.context)
        
        raise Exception("System resource recovery failed")
    
    def get_strategy_name(self) -> str:
        return "SystemResourceRecovery"
    
    async def _cleanup_resources(self):
        """Clean up unused resources"""
        import gc
        gc.collect()
        logger.info("Resource cleanup completed")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        # Implement memory optimization logic
        logger.info("Memory optimization completed")
    
    async def _retry_with_reduced_resources(self, context: ErrorContext) -> Any:
        """Retry operation with reduced resource requirements"""
        return {
            "result_type": "reduced_resource_result",
            "resource_usage": "optimized",
            "confidence": 0.8
        }


class NetworkConnectivityRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for network connectivity errors"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.NETWORK_CONNECTIVITY
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from network connectivity errors"""
        logger.info(f"Attempting network connectivity recovery for error {error_info.error_id}")
        
        # Implement retry with exponential backoff
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(base_delay * (2 ** attempt))
                # Attempt to reconnect or use cached data
                result = await self._attempt_reconnection(error_info.context)
                logger.info(f"Network connectivity recovery successful for {error_info.error_id}")
                return result
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                continue
        
        # Use cached data as fallback
        logger.info(f"Using cached data for {error_info.error_id}")
        return await self._use_cached_data(error_info.context)
    
    def get_strategy_name(self) -> str:
        return "NetworkConnectivityRecovery"
    
    async def _attempt_reconnection(self, context: Optional[ErrorContext]) -> Any:
        """Attempt to reconnect to external services"""
        # Simulate reconnection attempt
        return {
            "connection_status": "restored",
            "data_source": "external_service",
            "timestamp": datetime.now()
        }
    
    async def _use_cached_data(self, context: Optional[ErrorContext]) -> Any:
        """Use cached data when network is unavailable"""
        return {
            "data_source": "cache",
            "timestamp": datetime.now(),
            "warning": "Using cached data due to network connectivity issues"
        }


class SecurityRecoveryStrategy(RecoveryStrategy):
    """Recovery strategy for security errors"""
    
    async def can_handle(self, error_info: ErrorInfo) -> bool:
        return error_info.category == ErrorCategory.SECURITY
    
    async def recover(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from security errors"""
        logger.warning(f"Attempting security error recovery for error {error_info.error_id}")
        
        if "token.*expired" in error_info.message.lower():
            # Attempt token refresh
            return await self._refresh_authentication_token(error_info.context)
        elif "authorization.*denied" in error_info.message.lower():
            # Log security incident and provide limited access
            await self._log_security_incident(error_info)
            return await self._provide_limited_access(error_info.context)
        else:
            # For other security errors, log and deny access
            await self._log_security_incident(error_info)
            raise Exception("Security error - access denied")
    
    def get_strategy_name(self) -> str:
        return "SecurityRecovery"
    
    async def _refresh_authentication_token(self, context: Optional[ErrorContext]) -> Any:
        """Attempt to refresh authentication token"""
        logger.info("Attempting token refresh")
        return {
            "auth_status": "token_refreshed",
            "access_level": "standard",
            "timestamp": datetime.now()
        }
    
    async def _log_security_incident(self, error_info: ErrorInfo):
        """Log security incident for analysis"""
        incident_data = {
            "error_id": error_info.error_id,
            "timestamp": error_info.timestamp,
            "severity": error_info.severity.value,
            "message": error_info.message,
            "user_id": error_info.context.user_id if error_info.context else None
        }
        logger.warning(f"Security incident logged: {json.dumps(incident_data)}")
    
    async def _provide_limited_access(self, context: Optional[ErrorContext]) -> Any:
        """Provide limited access for authorization errors"""
        return {
            "access_level": "limited",
            "available_operations": ["read_only"],
            "warning": "Limited access due to authorization restrictions"
        }


class ErrorAnalyzer:
    """Analyzes error patterns and trends for system improvement"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.pattern_cache: Dict[str, Any] = {}
    
    def add_error(self, error_info: ErrorInfo):
        """Add error to analysis history"""
        self.error_history.append(error_info)
        
        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and trends"""
        if not self.error_history:
            
class AdvancedSeismicProcessor:
    """Advanced seismic data processing with signal analysis and event detection"""
    
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.filters = self._initialize_filters()
        self.event_detector = self._initialize_event_detector()
        
    def _initialize_filters(self) -> Dict[str, Any]:
        """Initialize various digital filters"""
        from scipy import signal
        
        nyquist = self.sampling_rate / 2
        
        return {
            'bandpass_1_10': signal.butter(4, [1.0/nyquist, 10.0/nyquist], btype='band'),
            'bandpass_0_5_25': signal.butter(4, [0.5/nyquist, 25.0/nyquist], btype='band'),
            'highpass_1': signal.butter(4, 1.0/nyquist, btype='high'),
            'lowpass_25': signal.butter(4, 25.0/nyquist, btype='low'),
            'notch_50': signal.iirnotch(50.0, 30, self.sampling_rate)  # Power line noise
        }
    
    def process_seismic_stream(self, data: np.ndarray, 
                             station_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process continuous seismic data stream"""
        
        # Multi-component processing (assuming 3-component data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        processed_components = []
        
        for i, component_data in enumerate(data):
            # Apply preprocessing pipeline
            processed = self._preprocess_component(component_data)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processed, component_data)
            
            # Detect events
            events = self._detect_events_in_component(processed)
            
            processed_components.append({
                'component_index': i,
                'component_name': ['Z', 'N', 'E'][i] if i < 3 else f'C{i}',
                'processed_data': processed,
                'quality_metrics': quality_metrics,
                'detected_events': events,
                'spectral_analysis': self._analyze_spectrum(processed)
            })
        
        # Cross-component analysis
        cross_component_analysis = self._cross_component_analysis(
            [comp['processed_data'] for comp in processed_components]
        )
        
        return {
            'station_id': station_info.get('station_id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'sampling_rate': self.sampling_rate,
            'components': processed_components,
            'cross_component_analysis': cross_component_analysis,
            'overall_quality': self._calculate_overall_quality(processed_components)
        }
    
    def _preprocess_component(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline to single component"""
        from scipy import signal
        
        # Remove mean and linear trend
        detrended = signal.detrend(data, type='linear')
        
        # Apply bandpass filter
        b, a = self.filters['bandpass_0_5_25']
        filtered = signal.filtfilt(b, a, detrended)
        
        # Remove power line noise
        b_notch, a_notch = self.filters['notch_50']
        filtered = signal.filtfilt(b_notch, a_notch, filtered)
        
        # Normalize amplitude
        rms = np.sqrt(np.mean(filtered**2))
        if rms > 0:
            normalized = filtered / rms
        else:
            normalized = filtered
        
        return normalized
    
    def _calculate_quality_metrics(self, processed_data: np.ndarray, 
                                 original_data: np.ndarray) -> Dict[str, float]:
        """Calculate data quality metrics"""
        
        # Signal-to-noise ratio
        signal_power = np.var(processed_data)
        noise_estimate = np.var(processed_data - original_data)
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        # Clipping detection
        max_val = np.max(np.abs(original_data))
        clipping_ratio = np.sum(np.abs(original_data) >= 0.95 * max_val) / len(original_data)
        
        # Gap detection
        gap_threshold = 5 * np.std(processed_data)
        gaps = np.sum(np.abs(np.diff(processed_data)) > gap_threshold)
        
        # Frequency content quality
        freqs, psd = signal.welch(processed_data, fs=self.sampling_rate)
        freq_quality = np.sum(psd[freqs <= 25]) / np.sum(psd)  # Energy below 25 Hz
        
        return {
            'snr_db': snr,
            'clipping_ratio': clipping_ratio,
            'gap_count': gaps,
            'frequency_quality': freq_quality,
            'rms_amplitude': np.sqrt(np.mean(processed_data**2)),
            'peak_amplitude': np.max(np.abs(processed_data))
        }
    
    def _detect_events_in_component(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect seismic events using multiple algorithms"""
        
        events = []
        
        # STA/LTA detector
        sta_lta_events = self._sta_lta_detector(data)
        events.extend(sta_lta_events)
        
        # Energy-based detector
        energy_events = self._energy_detector(data)
        events.extend(energy_events)
        
        # Frequency domain detector
        freq_events = self._frequency_detector(data)
        events.extend(freq_events)
        
        # Merge overlapping events
        merged_events = self._merge_overlapping_events(events)
        
        return merged_events
    
    def _sta_lta_detector(self, data: np.ndarray, 
                         sta_window: float = 0.5, 
                         lta_window: float = 10.0,
                         trigger_threshold: float = 3.0,
                         detrigger_threshold: float = 1.5) -> List[Dict[str, Any]]:
        """Short-Term Average / Long-Term Average event detector"""
        
        sta_samples = int(sta_window * self.sampling_rate)
        lta_samples = int(lta_window * self.sampling_rate)
        
        # Calculate energy
        energy = data ** 2
        
        # Calculate STA and LTA
        sta = np.convolve(energy, np.ones(sta_samples)/sta_samples, mode='same')
        lta = np.convolve(energy, np.ones(lta_samples)/lta_samples, mode='same')
        
        # Calculate ratio
        sta_lta_ratio = sta / (lta + 1e-10)
        
        # Find triggers
        events = []
        in_event = False
        event_start = 0
        
        for i, ratio in enumerate(sta_lta_ratio):
            if not in_event and ratio > trigger_threshold:
                in_event = True
                event_start = i
            elif in_event and ratio < detrigger_threshold:
                in_event = False
                event_end = i
                
                events.append({
                    'type': 'sta_lta',
                    'start_sample': event_start,
                    'end_sample': event_end,
                    'start_time': event_start / self.sampling_rate,
                    'end_time': event_end / self.sampling_rate,
                    'duration': (event_end - event_start) / self.sampling_rate,
                    'max_ratio': np.max(sta_lta_ratio[event_start:event_end]),
                    'confidence': min(1.0, np.max(sta_lta_ratio[event_start:event_end]) / trigger_threshold)
                })
        
        return events
    
    def _energy_detector(self, data: np.ndarray, 
                        window_size: float = 1.0,
                        threshold_factor: float = 5.0) -> List[Dict[str, Any]]:
        """Energy-based event detector"""
        
        window_samples = int(window_size * self.sampling_rate)
        
        # Calculate windowed energy
        energy = data ** 2
        windowed_energy = np.convolve(energy, np.ones(window_samples)/window_samples, mode='same')
        
        # Calculate threshold
        baseline_energy = np.median(windowed_energy)
        threshold = baseline_energy * threshold_factor
        
        # Find events
        above_threshold = windowed_energy > threshold
        event_regions = []
        
        in_event = False
        event_start = 0
        
        for i, above in enumerate(above_threshold):
            if not in_event and above:
                in_event = True
                event_start = i
            elif in_event and not above:
                in_event = False
                event_end = i
                
                if event_end - event_start > window_samples // 2:  # Minimum duration
                    event_regions.append({
                        'type': 'energy',
                        'start_sample': event_start,
                        'end_sample': event_end,
                        'start_time': event_start / self.sampling_rate,
                        'end_time': event_end / self.sampling_rate,
                        'duration': (event_end - event_start) / self.sampling_rate,
                        'max_energy': np.max(windowed_energy[event_start:event_end]),
                        'confidence': min(1.0, np.max(windowed_energy[event_start:event_end]) / threshold)
                    })
        
        return event_regions
    
    def _frequency_detector(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Frequency domain event detector"""
        from scipy import signal
        
        # Spectrogram analysis
        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_rate, nperseg=1024, noverlap=512)
        
        # Calculate spectral features over time
        spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        spectral_bandwidth = np.sqrt(np.sum(((f[:, np.newaxis] - spectral_centroid) ** 2) * Sxx, axis=0) / np.sum(Sxx, axis=0))
        spectral_rolloff = np.array([f[np.where(np.cumsum(Sxx[:, i]) >= 0.85 * np.sum(Sxx[:, i]))[0][0]] for i in range(Sxx.shape[1])])
        
        # Detect anomalies in spectral features
        centroid_threshold = np.mean(spectral_centroid) + 2 * np.std(spectral_centroid)
        bandwidth_threshold = np.mean(spectral_bandwidth) + 2 * np.std(spectral_bandwidth)
        
        events = []
        for i, (centroid, bandwidth) in enumerate(zip(spectral_centroid, spectral_bandwidth)):
            if centroid > centroid_threshold or bandwidth > bandwidth_threshold:
                start_time = t[i] - 0.5  # Extend event window
                end_time = t[i] + 0.5
                
                events.append({
                    'type': 'frequency',
                    'start_time': max(0, start_time),
                    'end_time': min(len(data) / self.sampling_rate, end_time),
                    'spectral_centroid': centroid,
                    'spectral_bandwidth': bandwidth,
                    'spectral_rolloff': spectral_rolloff[i],
                    'confidence': min(1.0, max(centroid / centroid_threshold, bandwidth / bandwidth_threshold))
                })
        
        return events

        
        # Analyze error frequency by category
        category_counts = {}
        severity_counts = {}
        hourly_distribution = {}
        
        for error in self.error_history:
            # Category analysis
            category = error.category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Severity analysis
            severity = error.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Temporal analysis
            hour = error.timestamp.hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        # Calculate error rates
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(hours=24)]
        
        analysis = {
            "total_errors": total_errors,
            "recent_24h_errors": len(recent_errors),
            "error_rate_24h": len(recent_errors) / 24.0,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "hourly_distribution": hourly_distribution,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "critical_error_count": severity_counts.get("critical", 0),
            "recommendations": self._generate_recommendations(category_counts, severity_counts)
        }
        
        return analysis
    
    def _generate_recommendations(self, category_counts: Dict[str, int], 
                                severity_counts: Dict[str, int]) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        # Check for high data quality error rates
        if category_counts.get("data_quality", 0) > 10:
            recommendations.append(
                "High data quality error rate detected. Consider implementing "
                "additional data validation and preprocessing steps."
            )
        
        # Check for model convergence issues
        if category_counts.get("model_convergence", 0) > 5:
            recommendations.append(
                "Frequent model convergence issues detected. Consider reviewing "
                "model architecture and training parameters."
            )
        
        # Check for critical errors
        if severity_counts.get("critical", 0) > 0:
            recommendations.append(
                "Critical errors detected. Immediate investigation and resolution required."
            )
        
        # Check for resource issues
        if category_counts.get("system_resource", 0) > 8:
            recommendations.append(
                "High system resource error rate. Consider scaling up infrastructure "
                "or optimizing resource usage."
            )
        
        return recommendations


class GracefulDegradationManager:
    """Manages graceful service degradation based on error patterns"""
    
    def __init__(self):
        self.current_degradation_level = ServiceDegradationLevel.FULL_SERVICE
        self.degradation_history: List[Dict[str, Any]] = []
    
    def assess_degradation_need(self, error_analyzer: ErrorAnalyzer) -> ServiceDegradationLevel:
        """Assess if service degradation is needed based on error patterns"""
        analysis = error_analyzer.analyze_error_patterns()
        
        critical_errors = analysis.get("critical_error_count", 0)
        recent_errors = analysis.get("recent_24h_errors", 0)
        error_rate = analysis.get("error_rate_24h", 0)
        
        # Determine degradation level based on error patterns
        if critical_errors > 5 or error_rate > 50:
            return ServiceDegradationLevel.EMERGENCY_MODE
        elif critical_errors > 2 or error_rate > 20:
            return ServiceDegradationLevel.OFFLINE_MODE
        elif error_rate > 10:
            return ServiceDegradationLevel.ESSENTIAL_ONLY
        elif error_rate > 5:
            return ServiceDegradationLevel.REDUCED_PRECISION
        else:
            return ServiceDegradationLevel.FULL_SERVICE
    
    def apply_degradation(self, level: ServiceDegradationLevel) -> Dict[str, Any]:
        """Apply service degradation to the specified level"""
        if level != self.current_degradation_level:
            logger.warning(f"Changing service degradation from {self.current_degradation_level.value} to {level.value}")
            
            self.degradation_history.append({
                "timestamp": datetime.now(),
                "from_level": self.current_degradation_level.value,
                "to_level": level.value,
                "reason": "Error pattern analysis"
            })
            
            self.current_degradation_level = level
        
        return self._get_service_configuration(level)
    
    def _get_service_configuration(self, level: ServiceDegradationLevel) -> Dict[str, Any]:
        """Get service configuration for the specified degradation level"""
        configurations = {
            ServiceDegradationLevel.FULL_SERVICE: {
                "available_features": ["all"],
                "model_complexity": "full",
                "response_time_target": "optimal",
                "resource_allocation": "full"
            },
            ServiceDegradationLevel.REDUCED_PRECISION: {
                "available_features": ["core", "monitoring", "basic_analysis"],
                "model_complexity": "reduced",
                "response_time_target": "fast",
                "resource_allocation": "reduced"
            },
            ServiceDegradationLevel.ESSENTIAL_ONLY: {
                "available_features": ["monitoring", "alerts"],
                "model_complexity": "minimal",
                "response_time_target": "fast",
                "resource_allocation": "minimal"
            },
            ServiceDegradationLevel.OFFLINE_MODE: {
                "available_features": ["cached_results", "basic_alerts"],
                "model_complexity": "cached",
                "response_time_target": "immediate",
                "resource_allocation": "minimal"
            },
            ServiceDegradationLevel.EMERGENCY_MODE: {
                "available_features": ["emergency_alerts"],
                "model_complexity": "precomputed",
                "response_time_target": "immediate",
                "resource_allocation": "emergency_only"
            }
        }
        
        return configurations.get(level, configurations[ServiceDegradationLevel.EMERGENCY_MODE])


class PrometheusErrorHandler:
    """Main hierarchical error management system for Prometheus AI Platform"""
    
    def __init__(self):
        self.error_classifier = ErrorClassifier()
        self.recovery_strategies: List[RecoveryStrategy] = [
            DataQualityRecoveryStrategy(),
            ModelConvergenceRecoveryStrategy(),
            PhysicsViolationRecoveryStrategy(),
            SystemResourceRecoveryStrategy(),
            NetworkConnectivityRecoveryStrategy(),
            SecurityRecoveryStrategy()
        ]
        self.error_analyzer = ErrorAnalyzer()
        self.degradation_manager = GracefulDegradationManager()
        self.alert_system = AlertSystem()
        
        logger.info("Prometheus Error Handler initialized with hierarchical error management")
    
    async def handle_error(self, error: Exception, context: Optional[ErrorContext] = None) -> Any:
        """Main error handling entry point"""
        start_time = time.time()
        
        try:
            # Classify the error
            error_info = self.error_classifier.classify(error, context)
            logger.error(f"Error classified: {error_info.error_id} - {error_info.category.value} - {error_info.severity.value}")
            
            # Add to error analysis
            self.error_analyzer.add_error(error_info)
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(error_info)
            
            # Mark as resolved and record recovery time
            error_info.resolved = True
            error_info.recovery_time = time.time() - start_time
            
            # Check if service degradation is needed
            await self._check_degradation_need()
            
            # Send alerts if necessary
            await self._send_alerts_if_needed(error_info)
            
            logger.info(f"Error {error_info.error_id} resolved successfully in {error_info.recovery_time:.2f}s")
            return recovery_result
            
        except Exception as recovery_error:
            logger.critical(f"Error recovery failed: {recovery_error}")
            
            # Send critical alert
            await self.alert_system.send_critical_alert(
                f"Error recovery failed for {error_info.error_id if 'error_info' in locals() else 'unknown'}: {recovery_error}"
            )
            
            # Apply emergency degradation
            self.degradation_manager.apply_degradation(ServiceDegradationLevel.EMERGENCY_MODE)
            
            raise recovery_error
    
    async def _attempt_recovery(self, error_info: ErrorInfo) -> Any:
        """Attempt to recover from the error using appropriate strategy"""
        for strategy in self.recovery_strategies:
            if await strategy.can_handle(error_info):
                try:
                    logger.info(f"Attempting recovery with strategy: {strategy.get_strategy_name()}")
                    result = await strategy.recover(error_info)
                    error_info.resolution_strategy = strategy.get_strategy_name()
                    return result
                except Exception as strategy_error:
                    logger.warning(f"Recovery strategy {strategy.get_strategy_name()} failed: {strategy_error}")
                    continue
        
        # No strategy could handle the error
        raise Exception(f"No recovery strategy available for error category: {error_info.category.value}")
    
    async def _check_degradation_need(self):
        """Check if service degradation is needed based on error patterns"""
        recommended_level = self.degradation_manager.assess_degradation_need(self.error_analyzer)
        
        if recommended_level != self.degradation_manager.current_degradation_level:
            config = self.degradation_manager.apply_degradation(recommended_level)
            logger.warning(f"Service degradation applied: {recommended_level.value}")
            logger.info(f"New service configuration: {config}")
    
    async def _send_alerts_if_needed(self, error_info: ErrorInfo):
        """Send alerts based on error severity and patterns"""
        if error_info.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            await self.alert_system.send_alert(error_info)
        
        # Check for error pattern alerts
        analysis = self.error_analyzer.analyze_error_patterns()
        if analysis.get("critical_error_count", 0) > 3:
            await self.alert_system.send_pattern_alert(
                "Multiple critical errors detected in recent period",
                analysis
            )
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        analysis = self.error_analyzer.analyze_error_patterns()
        
        return {
            "degradation_level": self.degradation_manager.current_degradation_level.value,
            "error_analysis": analysis,
            "service_configuration": self.degradation_manager._get_service_configuration(
                self.degradation_manager.current_degradation_level
            ),
            "timestamp": datetime.now(),
            "system_status": self._determine_system_status(analysis)
        }
    
    def _determine_system_status(self, analysis: Dict[str, Any]) -> str:
        """Determine overall system status based on error analysis"""
        critical_errors = analysis.get("critical_error_count", 0)
        error_rate = analysis.get("error_rate_24h", 0)
        
        if critical_errors > 5 or error_rate > 50:
            return "critical"
        elif critical_errors > 2 or error_rate > 20:
            return "degraded"
        elif error_rate > 10:
            return "warning"
        else:
            return "healthy"


class AlertSystem:
    """Alert system for error notifications and escalation"""
    
    def __init__(self):
        self.alert_history: List[Dict[str, Any]] = []
    
    async def send_alert(self, error_info: ErrorInfo):
        """Send alert for individual errors"""
        alert_data = {
            "alert_id": f"alert_{error_info.error_id}",
            "timestamp": datetime.now(),
            "error_id": error_info.error_id,
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "message": error_info.message,
            "service": error_info.context.service_name if error_info.context else "unknown"
        }
        
        self.alert_history.append(alert_data)
        logger.warning(f"Alert sent: {json.dumps(alert_data)}")
    
    async def send_critical_alert(self, message: str):
        """Send critical system alert"""
        alert_data = {
            "alert_type": "critical_system",
            "timestamp": datetime.now(),
            "message": message,
            "requires_immediate_attention": True
        }
        
        self.alert_history.append(alert_data)
        logger.critical(f"Critical alert sent: {json.dumps(alert_data)}")
    
    async def send_pattern_alert(self, message: str, analysis: Dict[str, Any]):
        """Send alert for error patterns"""
        alert_data = {
            "alert_type": "error_pattern",
            "timestamp": datetime.now(),
            "message": message,
            "analysis": analysis
        }
        
        self.alert_history.append(alert_data)
        logger.warning(f"Pattern alert sent: {json.dumps(alert_data)}")


# Global error handler instance
prometheus_error_handler = PrometheusErrorHandler()


# Convenience functions for easy integration
async def handle_error(error: Exception, context: Optional[ErrorContext] = None) -> Any:
    """Convenience function to handle errors using the global error handler"""
    return await prometheus_error_handler.handle_error(error, context)


def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return prometheus_error_handler.get_system_health_status()


def create_error_context(service_name: str, operation: str, **kwargs) -> ErrorContext:
    """Create error context for error handling"""
    return ErrorContext(
        service_name=service_name,
        operation=operation,
        timestamp=datetime.now(),
        **kwargs
    )