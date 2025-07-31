"""Service registry for service discovery using Consul"""

import asyncio
import logging
from typing import Dict, List, Optional

import consul.aio
import structlog

logger = structlog.get_logger()


class ServiceRegistry:
    """Service registry for managing service discovery with Consul"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_client: Optional[consul.aio.Consul] = None
        self.registered_services: Dict[str, str] = {}
        
    async def initialize(self):
        """Initialize the Consul client"""
        try:
            self.consul_client = consul.aio.Consul(
                host=self.consul_host,
                port=self.consul_port
            )
            logger.info("Service registry initialized", 
                       consul_host=self.consul_host, 
                       consul_port=self.consul_port)
        except Exception as e:
            logger.error("Failed to initialize service registry", error=str(e))
            raise
    
    async def register_service(
        self,
        name: str,
        service_id: str,
        address: str,
        port: int,
        health_check_url: str,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, str]] = None
    ):
        """Register a service with Consul"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            health_check = consul.Check.http(
                url=f"http://{address}:{port}{health_check_url}",
                interval="10s",
                timeout="5s",
                deregister="30s"
            )
            
            await self.consul_client.agent.service.register(
                name=name,
                service_id=service_id,
                address=address,
                port=port,
                tags=tags or [],
                meta=meta or {},
                check=health_check
            )
            
            self.registered_services[service_id] = name
            logger.info("Service registered successfully",
                       service_name=name,
                       service_id=service_id,
                       address=address,
                       port=port)
            
        except Exception as e:
            logger.error("Failed to register service",
                        service_name=name,
                        service_id=service_id,
                        error=str(e))
            raise
    
    async def deregister_service(self, service_id: str):
        """Deregister a service from Consul"""
        if not self.consul_client:
            return
        
        try:
            await self.consul_client.agent.service.deregister(service_id)
            if service_id in self.registered_services:
                service_name = self.registered_services.pop(service_id)
                logger.info("Service deregistered successfully",
                           service_name=service_name,
                           service_id=service_id)
        except Exception as e:
            logger.error("Failed to deregister service",
                        service_id=service_id,
                        error=str(e))
    
    async def discover_service(self, service_name: str) -> Optional[Dict]:
        """Discover a service by name"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, services = await self.consul_client.health.service(
                service_name, 
                passing=True
            )
            
            if services:
                service = services[0]
                return {
                    "name": service["Service"]["Service"],
                    "id": service["Service"]["ID"],
                    "address": service["Service"]["Address"],
                    "port": service["Service"]["Port"],
                    "tags": service["Service"]["Tags"],
                    "meta": service["Service"]["Meta"]
                }
            return None
            
        except Exception as e:
            logger.error("Failed to discover service",
                        service_name=service_name,
                        error=str(e))
            return None
    
    async def discover_all_services(self) -> List[Dict]:
        """Discover all healthy services"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, services = await self.consul_client.catalog.services()
            discovered_services = []
            
            for service_name in services.keys():
                service_info = await self.discover_service(service_name)
                if service_info:
                    discovered_services.append(service_info)
            
            return discovered_services
            
        except Exception as e:
            logger.error("Failed to discover all services", error=str(e))
            return []
    
    async def get_service_health(self, service_name: str) -> Dict:
        """Get health status of a service"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        try:
            _, checks = await self.consul_client.health.service(service_name)
            
            if not checks:
                
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
            resample_config = config['resample']
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
        
        return {'method': 'none', 'anomaly_count': 0}

            
            health_status = "passing"
            check_details = []
            
            for check in checks:
                for health_check in check.get("Checks", []):
                    check_details.append({
                        "name": health_check["Name"],
                        "status": health_check["Status"],
                        "output": health_check["Output"]
                    })
                    
                    if health_check["Status"] != "passing":
                        health_status = "failing"
            
            return {
                "status": health_status,
                "checks": check_details
            }
            
        except Exception as e:
            logger.error("Failed to get service health",
                        service_name=service_name,
                        error=str(e))
            
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
            resample_config = config['resample']
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
        
        return {'method': 'none', 'anomaly_count': 0}

    
    async def watch_service(self, service_name: str, callback):
        """Watch for changes in a service"""
        if not self.consul_client:
            raise RuntimeError("Service registry not initialized")
        
        index = None
        while True:
            try:
                index, services = await self.consul_client.health.service(
                    service_name,
                    index=index,
                    wait="30s"
                )
                
                await callback(service_name, services)
                
            except Exception as e:
                logger.error("Error watching service",
                            service_name=service_name,
                            error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def cleanup(self):
        """Cleanup and deregister all services"""
        for service_id in list(self.registered_services.keys()):
            await self.deregister_service(service_id)
        
        if self.consul_client:
            await self.consul_client.close()