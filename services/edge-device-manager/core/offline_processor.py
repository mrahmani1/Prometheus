"""
Offline data processing and analysis capabilities for edge devices
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pickle
from dataclasses import dataclass, asdict

from models.device import DeviceMetrics, SyncStatus
from models.config import EdgeDeviceConfig

logger = logging.getLogger(__name__)


@dataclass
class OfflineAnalysisResult:
    """Result of offline data analysis"""
    timestamp: datetime
    analysis_type: str
    input_data_hash: str
    results: Dict[str, Any]
    confidence_score: float
    alerts_generated: List[Dict[str, Any]]
    processing_time_ms: float


@dataclass
class LocalDataRecord:
    """Local data storage record"""
    record_id: str
    timestamp: datetime
    data_type: str
    data_payload: bytes
    metadata: Dict[str, Any]
    processed: bool = False
    sync_status: str = "pending"


class OfflineDataProcessor:
    """Handles local data processing and analysis when offline"""
    
    def __init__(self, config: EdgeDeviceConfig, storage_path: str = "./offline_data"):
        self.config = config
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Local database for data storage
        self.db_path = self.storage_path / "offline_data.db"
        self.db_connection: Optional[sqlite3.Connection] = None
        
        # Cached models and thresholds
        self.cached_models: Dict[str, Any] = {}
        self.alert_thresholds: Dict[str, Dict[str, float]] = {}
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "alerts_generated": 0,
            "processing_errors": 0,
            "last_processing_time": None
        }
        
        self._initialize_database()
        self._load_cached_models()
        self._load_alert_thresholds()
    
    def _initialize_database(self):
        """Initialize local SQLite database for offline storage"""
        try:
            self.db_connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create tables for local data storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE,
                    timestamp REAL,
                    data_type TEXT,
                    data_payload BLOB,
                    metadata TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    sync_status TEXT DEFAULT 'pending',
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    analysis_type TEXT,
                    input_data_hash TEXT,
                    results TEXT,
                    confidence_score REAL,
                    alerts_generated TEXT,
                    processing_time_ms REAL,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    data_source TEXT,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    created_at REAL DEFAULT (julianday('now'))
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(data_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis_results(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alert_timestamp ON alert_history(timestamp)")
            
            self.db_connection.commit()
            logger.info("Offline database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize offline database: {e}")
            raise
    
    def _load_cached_models(self):
        """Load pre-trained models for offline processing"""
        try:
            models_path = self.storage_path / "cached_models"
            models_path.mkdir(exist_ok=True)
            
            # Load simplified geological models
            model_files = {
                "seismic_classifier": "seismic_event_classifier.pkl",
                "deformation_analyzer": "deformation_pattern_analyzer.pkl",
                "thermal_monitor": "thermal_anomaly_detector.pkl",
                "gas_analyzer": "gas_composition_analyzer.pkl"
            }
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    try:
                        with open(model_path, 'rb') as f:
                            self.cached_models[model_name] = pickle.load(f)
                        logger.info(f"Loaded cached model: {model_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load model {model_name}: {e}")
                else:
                    # Create placeholder simplified models
                    self.cached_models[model_name] = self._create_simplified_model(model_name)
                    logger.info(f"Created simplified model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load cached models: {e}")
    
    def _load_alert_thresholds(self):
        """Load pre-computed alert thresholds"""
        try:
            thresholds_path = self.storage_path / "alert_thresholds.json"
            
            if thresholds_path.exists():
                with open(thresholds_path, 'r') as f:
                    self.alert_thresholds = json.load(f)
            else:
                # Default thresholds for geological monitoring
                self.alert_thresholds = {
                    "seismic": {
                        "magnitude_warning": 3.0,
                        "magnitude_alert": 4.5,
                        "frequency_threshold": 10,  # events per hour
                        "depth_shallow": 10.0  # km
                    },
                    "deformation": {
                        "displacement_warning": 0.01,  # meters
                        "displacement_alert": 0.05,
                        "velocity_warning": 0.001,  # m/day
                        "velocity_alert": 0.01
                    },
                    "thermal": {
                        "temperature_warning": 50.0,  # Celsius above baseline
                        "temperature_alert": 100.0,
                        "gradient_warning": 10.0,  # C/day
                        "gradient_alert": 25.0
                    },
                    "gas": {
                        "co2_warning": 1000.0,  # ppm
                        "co2_alert": 5000.0,
                        "so2_warning": 100.0,
                        "so2_alert": 500.0,
                        "h2s_warning": 50.0,
                        "h2s_alert": 200.0
                    }
                }
                
                # Save default thresholds
                with open(thresholds_path, 'w') as f:
                    json.dump(self.alert_thresholds, f, indent=2)
            
            logger.info("Alert thresholds loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load alert thresholds: {e}")
    
    def _create_simplified_model(self, model_name: str) -> Dict[str, Any]:
        """Create simplified rule-based models for offline processing"""
        if model_name == "seismic_classifier":
            return {
                "type": "rule_based",
                "rules": {
                    "earthquake": {"min_magnitude": 2.0, "max_depth": 50.0},
                    "volcanic": {"min_magnitude": 1.0, "max_depth": 15.0, "frequency_pattern": "swarm"},
                    "explosion": {"min_magnitude": 1.5, "max_depth": 5.0, "duration": "short"},
                    "noise": {"max_magnitude": 1.0}
                }
            }
        elif model_name == "deformation_analyzer":
            return {
                "type": "threshold_based",
                "thresholds": {
                    "uplift": 0.01,  # meters
                    "subsidence": -0.01,
                    "lateral": 0.005,
                    "velocity": 0.001  # m/day
                }
            }
        elif model_name == "thermal_monitor":
            return {
                "type": "anomaly_detection",
                "baseline_temp": 25.0,  # Celsius
                "anomaly_threshold": 3.0,  # standard deviations
                "trend_window": 24  # hours
            }
        elif model_name == "gas_analyzer":
            return {
                "type": "composition_analysis",
                "normal_ranges": {
                    "CO2": (300, 500),  # ppm
                    "SO2": (0, 10),
                    "H2S": (0, 5),
                    "CH4": (1.8, 2.2)
                }
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            re
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []
 config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps
        for step_name, step_config in processing_config.items():
            try:
                if step_name == 'filter':
                    data = self._apply_filter(data, step_config)
                elif step_name == 'normalize':
                    data = self._normalize_data(data, step_config)
                elif step_name == 'detect_anomalies':
                    anomalies = self._detect_anomalies(data, step_config)
                    results['anomalies'] = anomalies
                
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'success',
                    'output_size': len(data) if isinstance(data, list) else 'N/A'
                })
                
            except Exception as e:
                results['processing_steps'].append({
                    'step': step_name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results
    
    def _apply_filter(self, data: Any, config: Dict[str, Any]) -> Any:
        """Apply filtering to data"""
        # Implement filtering logic
        return data
    
    def _normalize_data(self, data: Any, config: Dict[str, Any]) -> Any:
        """Normalize data"""
        # Implement normalization logic
        return data
    
    def _detect_anomalies(self, data: Any, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        # Implement anomaly detection
        return []


    
    async def store_sensor_data(self, data_type: str, data_payload: bytes, metadata: Dict[str, Any]) -> str:
        """Store sensor data locally for offline processing"""
        try:
            record_id = f"{data_type}_{datetime.utcnow().timestamp()}_{hash(data_payload) % 10000}"
            timestamp = datetime.utcnow().timestamp()
            
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO sensor_data (record_id, timestamp, data_type, data_payload, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (record_id, timestamp, data_type, data_payload, json.dumps(metadata)))
            
            self.db_connection.commit()
            
            # Add to processing queue
            await self.processing_queue.put({
                "record_id": record_id,
                "data_type": data_type,
                "timestamp": timestamp
            })
            
            logger.debug(f"Stored sensor data: {record_id}")
            return record_id
            
        except Exception as e:
            logger.error(f"Failed to store sensor data: {e}")
            raise
    
    async def process_offline_data(self) -> List[OfflineAnalysisResult]:
        """Process queued data using offline models"""
        results = []
        
        try:
            # Process items from queue
            processed_count = 0
            while not self.processing_queue.empty() and processed_count < 100:  # Batch processing
                try:
                    queue_item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                    result = await self._process_single_record(queue_item)
                    if result:
                        results.append(result)
                        await self._store_analysis_result(result)
                    processed_count += 1
                    
                except asyncio.TimeoutError:
                    break
                except Exception as e:
                    logger.error(f"Error processing queue item: {e}")
                    self.processing_stats["processing_errors"] += 1
            
            self.processing_stats["total_processed"] += processed_count
            self.processing_stats["last_processing_time"] = datetime.utcnow()
            
            logger.info(f"Processed {processed_count} offline data records")
            
        except Exception as e:
            logger.error(f"Error in offline data processing: {e}")
        
        return results
    
    async def _process_single_record(self, queue_item: Dict[str, Any]) -> Optional[OfflineAnalysisResult]:
        """Process a single data record"""
        try:
            start_time = datetime.utcnow()
            record_id = queue_item["record_id"]
            data_type = queue_item["data_type"]
            
            # Retrieve data from database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                SELECT data_payload, metadata FROM sensor_data 
                WHERE record_id = ? AND processed = FALSE
            """, (record_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            data_payload, metadata_json = row
            metadata = json.loads(metadata_json)
            
            # Process based on data type
            analysis_result = None
            if data_type == "seismic":
                analysis_result = await self._process_seismic_data(data_payload, metadata)
            elif data_type == "deformation":
                analysis_result = await self._process_deformation_data(data_payload, metadata)
            elif data_type == "thermal":
                analysis_result = await self._process_thermal_data(data_payload, metadata)
            elif data_type == "gas":
                analysis_result = await self._process_gas_data(data_payload, metadata)
            else:
                logger.warning(f"Unknown data type for offline processing: {data_type}")
                return None
            
            # Mark as processed
            cursor.execute("""
                UPDATE sensor_data SET processed = TRUE WHERE record_id = ?
            """, (record_id,))
            self.db_connection.commit()
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create result object
            result = OfflineAnalysisResult(
                timestamp=datetime.utcnow(),
                analysis_type=data_type,
                input_data_hash=str(hash(data_payload)),
                results=analysis_result["results"],
                confidence_score=analysis_result["confidence"],
                alerts_generated=analysis_result["alerts"],
                processing_time_ms=processing_time
            )
            
            # Generate alerts if needed
            if result.alerts_generated:
                await self._handle_offline_alerts(result.alerts_generated)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing record {queue_item.get('record_id', 'unknown')}: {e}")
            return None
    
    async def _process_seismic_data(self, data_payload: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process seismic data using offline models"""
        try:
            # Deserialize seismic data
            seismic_data = pickle.loads(data_payload)
            model = self.cached_models.get("seismic_classifier", {})
            
            # Extract features (simplified)
            magnitude = seismic_data.get("magnitude", 0.0)
            depth = seismic_data.get("depth", 0.0)
            duration = seismic_data.get("duration", 0.0)
            frequency_content = seismic_data.get("frequency_content", {})
            
            # Rule-based classification
            event_type = "unknown"
            confidence = 0.5
            
            if model.get("type") == "rule_based":
                rules = model.get("rules", {})
                
                if magnitude >= rules["earthquake"]["min_magnitude"] and depth <= rules["earthquake"]["max_depth"]:
                    event_type = "earthquake"
                    confidence = min(0.9, 0.5 + magnitude * 0.1)
                elif magnitude >= rules["volcanic"]["min_magnitude"] and depth <= rules["volcanic"]["max_depth"]:
                    event_type = "volcanic"
                    confidence = 0.7
                elif magnitude >= rules["explosion"]["min_magnitude"] and depth <= rules["explosion"]["max_depth"]:
                    event_type = "explosion"
                    confidence = 0.6
                else:
                    event_type = "noise"
                    confidence = 0.3
            
            # Check for alerts
            alerts = []
            thresholds = self.alert_thresholds.get("seismic", {})
            
            if magnitude >= thresholds.get("magnitude_alert", 4.5):
                alerts.append({
                    "type": "seismic_alert",
                    "severity": "high",
                    "message": f"High magnitude seismic event detected: M{magnitude:.1f}",
                    "data": {"magnitude": magnitude, "depth": depth, "event_type": event_type}
                })
            elif magnitude >= thresholds.get("magnitude_warning", 3.0):
                alerts.append({
                    "type": "seismic_warning",
                    "severity": "medium",
                    "message": f"Moderate seismic event detected: M{magnitude:.1f}",
                    "data": {"magnitude": magnitude, "depth": depth, "event_type": event_type}
                })
            
            return {
                "results": {
                    "event_type": event_type,
                    "magnitude": magnitude,
                    "depth": depth,
                    "duration": duration,
                    "location": metadata.get("location", {}),
                    "frequency_analysis": frequency_content
                },
                "confidence": confidence,
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Error processing seismic data: {e}")
            return {"results": {}, "confidence": 0.0, "alerts": []}
    
    async def _process_deformation_data(self, data_payload: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process deformation data using offline models"""
        try:
            deformation_data = pickle.loads(data_payload)
            model = self.cached_models.get("deformation_analyzer", {})
            
            # Extract displacement vectors
            displacement = deformation_data.get("displacement_vector", [0, 0, 0])
            velocity = deformation_data.get("velocity", [0, 0, 0])
            
            # Calculate magnitudes
            displacement_magnitude = np.linalg.norm(displacement)
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Threshold-based analysis
            alerts = []
            thresholds = self.alert_thresholds.get("deformation", {})
            
            if displacement_magnitude >= thresholds.get("displacement_alert", 0.05):
                alerts.append({
                    "type": "deformation_alert",
                    "severity": "high",
                    "message": f"Significant ground deformation detected: {displacement_magnitude:.3f}m",
                    "data": {"displacement": displacement, "velocity": velocity}
                })
            elif displacement_magnitude >= thresholds.get("displacement_warning", 0.01):
                alerts.append({
                    "type": "deformation_warning",
                    "severity": "medium",
                    "message": f"Ground deformation detected: {displacement_magnitude:.3f}m",
                    "data": {"displacement": displacement, "velocity": velocity}
                })
            
            # Determine deformation pattern
            pattern = "stable"
            confidence = 0.6
            
            if displacement_magnitude > 0.001:
                if displacement[2] > 0:  # Vertical component positive
                    pattern = "uplift"
                elif displacement[2] < 0:
                    pattern = "subsidence"
                else:
                    pattern = "lateral"
                confidence = min(0.9, 0.5 + displacement_magnitude * 10)
            
            return {
                "results": {
                    "pattern": pattern,
                    "displacement_magnitude": displacement_magnitude,
                    "velocity_magnitude": velocity_magnitude,
                    "displacement_vector": displacement,
                    "velocity_vector": velocity,
                    "location": metadata.get("location", {})
                },
                "confidence": confidence,
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Error processing deformation data: {e}")
            return {"results": {}, "confidence": 0.0, "alerts": []}
    
    async def _process_thermal_data(self, data_payload: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process thermal data using offline models"""
        try:
            thermal_data = pickle.loads(data_payload)
            model = self.cached_models.get("thermal_monitor", {})
            
            temperature = thermal_data.get("temperature", 0.0)
            baseline_temp = model.get("baseline_temp", 25.0)
            anomaly_threshold = model.get("anomaly_threshold", 3.0)
            
            # Calculate temperature anomaly
            temp_anomaly = temperature - baseline_temp
            
            # Simple anomaly detection
            alerts = []
            thresholds = self.alert_thresholds.get("thermal", {})
            
            if temp_anomaly >= thresholds.get("temperature_alert", 100.0):
                alerts.append({
                    "type": "thermal_alert",
                    "severity": "high",
                    "message": f"High temperature anomaly detected: +{temp_anomaly:.1f}°C",
                    "data": {"temperature": temperature, "anomaly": temp_anomaly}
                })
            elif temp_anomaly >= thresholds.get("temperature_warning", 50.0):
                alerts.append({
                    "type": "thermal_warning",
                    "severity": "medium",
                    "message": f"Temperature anomaly detected: +{temp_anomaly:.1f}°C",
                    "data": {"temperature": temperature, "anomaly": temp_anomaly}
                })
            
            # Determine thermal status
            status = "normal"
            confidence = 0.7
            
            if abs(temp_anomaly) > anomaly_threshold:
                status = "anomalous"
                confidence = min(0.95, 0.6 + abs(temp_anomaly) / 100.0)
            
            return {
                "results": {
                    "status": status,
                    "temperature": temperature,
                    "baseline_temperature": baseline_temp,
                    "temperature_anomaly": temp_anomaly,
                    "location": metadata.get("location", {})
                },
                "confidence": confidence,
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Error processing thermal data: {e}")
            return {"results": {}, "confidence": 0.0, "alerts": []}
    
    async def _process_gas_data(self, data_payload: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process gas composition data using offline models"""
        try:
            gas_data = pickle.loads(data_payload)
            model = self.cached_models.get("gas_analyzer", {})
            
            composition = gas_data.get("composition", {})
            normal_ranges = model.get("normal_ranges", {})
            
            # Analyze gas composition
            anomalies = []
            alerts = []
            thresholds = self.alert_thresholds.get("gas", {})
            
            for gas, concentration in composition.items():
                if gas in normal_ranges:
                    min_normal, max_normal = normal_ranges[gas]
                    if concentration < min_normal or concentration > max_normal:
                        anomalies.append({
                            "gas": gas,
                            "concentration": concentration,
                            "normal_range": normal_ranges[gas],
                            "deviation": "low" if concentration < min_normal else "high"
                        })
                
                # Check alert thresholds
                alert_threshold = thresholds.get(f"{gas.lower()}_alert")
                warning_threshold = thresholds.get(f"{gas.lower()}_warning")
                
                if alert_threshold and concentration >= alert_threshold:
                    alerts.append({
                        "type": "gas_alert",
                        "severity": "high",
                        "message": f"High {gas} concentration detected: {concentration} ppm",
                        "data": {"gas": gas, "concentration": concentration}
                    })
                elif warning_threshold and concentration >= warning_threshold:
                    alerts.append({
                        "type": "gas_warning",
                        "severity": "medium",
                        "message": f"Elevated {gas} concentration: {concentration} ppm",
                        "data": {"gas": gas, "concentration": concentration}
                    })
            
            # Overall assessment
            status = "normal" if not anomalies else "anomalous"
            confidence = 0.8 if not anomalies else max(0.4, 0.8 - len(anomalies) * 0.1)
            
            return {
                "results": {
                    "status": status,
                    "composition": composition,
                    "anomalies": anomalies,
                    "location": metadata.get("location", {})
                },
                "confidence": confidence,
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Error processing gas data: {e}")
            return {"results": {}, "confidence": 0.0, "alerts": []}
    
    async def _handle_offline_alerts(self, alerts: List[Dict[str, Any]]):
        """Handle alerts generated during offline processing"""
        try:
            cursor = self.db_connection.cursor()
            
            for alert in alerts:
                cursor.execute("""
                    INSERT INTO alert_history (timestamp, alert_type, severity, message, data_source)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().timestamp(),
                    alert["type"],
                    alert["severity"],
                    alert["message"],
                    json.dumps(alert.get("data", {}))
                ))
                
                # Log alert
                logger.warning(f"OFFLINE ALERT [{alert['severity'].upper()}]: {alert['message']}")
            
            self.db_connection.commit()
            self.processing_stats["alerts_generated"] += len(alerts)
            
        except Exception as e:
            logger.error(f"Error handling offline alerts: {e}")
    
    async def _store_analysis_result(self, result: OfflineAnalysisResult):
        """Store analysis result in local database"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO analysis_results 
                (timestamp, analysis_type, input_data_hash, results, confidence_score, alerts_generated, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                result.timestamp.timestamp(),
                result.analysis_type,
                result.input_data_hash,
                json.dumps(result.results),
                result.confidence_score,
                json.dumps(result.alerts_generated),
                result.processing_time_ms
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error storing analysis result: {e}")
    
    async def get_offline_alerts(self, since: Optional[datetime] = None, acknowledged: bool = False) -> List[Dict[str, Any]]:
        """Retrieve offline alerts from local storage"""
        try:
            cursor = self.db_connection.cursor()
            
            query = "SELECT * FROM alert_history WHERE acknowledged = ?"
            params = [acknowledged]
            
            if since:
                query += " AND timestamp >= ?"
                params.append(since.timestamp())
            
            query += " ORDER BY timestamp DESC LIMIT 100"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                alerts.append({
                    "id": row[0],
                    "timestamp": datetime.fromtimestamp(row[1]),
                    "alert_type": row[2],
                    "severity": row[3],
                    "message": row[4],
                    "data_source": json.loads(row[5]) if row[5] else {},
                    "acknowledged": bool(row[6]),
                    "created_at": datetime.fromtimestamp(row[7])
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error retrieving offline alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: int):
        """Acknowledge an offline alert"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute("UPDATE alert_history SET acknowledged = TRUE WHERE id = ?", (alert_id,))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get offline processing statistics"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get data counts
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            total_data_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sensor_data WHERE processed = TRUE")
            processed_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM analysis_results")
            analysis_results = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM alert_history WHERE acknowledged = FALSE")
            unacknowledged_alerts = cursor.fetchone()[0]
            
            # Get storage usage
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size_bytes = cursor.fetchone()[0]
            
            return {
                "total_data_records": total_data_records,
                "processed_records": processed_records,
                "pending_records": total_data_records - processed_records,
                "analysis_results": analysis_results,
                "unacknowledged_alerts": unacknowledged_alerts,
                "database_size_bytes": db_size_bytes,
                "processing_stats": self.processing_stats.copy(),
                "cached_models": list(self.cached_models.keys()),
                "queue_size": self.processing_queue.qsize()
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {e}")
            return {}
    
    async def cleanup_old_data(self, retention_days: int = 7):
        """Clean up old data based on retention policy"""
        try:
            cutoff_timestamp = (datetime.utcnow() - timedelta(days=retention_days)).timestamp()
            cursor = self.db_connection.cursor()
            
            # Clean up old sensor data
            cursor.execute("DELETE FROM sensor_data WHERE timestamp < ? AND processed = TRUE", (cutoff_timestamp,))
            deleted_sensor = cursor.rowcount
            
            # Clean up old analysis results
            cursor.execute("DELETE FROM analysis_results WHERE timestamp < ?", (cutoff_timestamp,))
            deleted_analysis = cursor.rowcount
            
            # Clean up old acknowledged alerts
            cursor.execute("DELETE FROM alert_history WHERE timestamp < ? AND acknowledged = TRUE", (cutoff_timestamp,))
            deleted_alerts = cursor.rowcount
            
            self.db_connection.commit()
            
            logger.info(f"Cleaned up old data: {deleted_sensor} sensor records, {deleted_analysis} analysis results, {deleted_alerts} alerts")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    async def shutdown(self):
        """Shutdown offline processor and cleanup resources"""
        try:
            if self.db_connection:
                self.db_connection.close()
            logger.info("Offline processor shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during offline processor shutdown: {e}")