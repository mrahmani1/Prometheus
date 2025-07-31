"""
API routes for the Graph Database Service
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from core.database import GraphDatabaseManager
from core.graph_algorithms import GeologicalGraphAlgorithms
from core.advanced_analytics import AdvancedGraphAnalytics
from core.causal_discovery import CausalDiscoveryEngine
from models.geological_models import (
    GeologicalProcess, GeologicalRelationship, GraphQuery,
    NetworkAnalysisResult, CausalRelationship, GraphVisualizationConfig
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get database manager
async def get_db_manager() -> GraphDatabaseManager:
    """Get database manager instance"""
    from main import db_manager
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_manager


@router.post("/processes", response_model=Dict[str, str])
async def create_geological_process(
    process: GeologicalProcess,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create a new geological process"""
    try:
        process_id = await db.create_geological_process(process)
        
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

    except Exception as e:
        logger.error(f"Failed to create geological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes/{process_id}")
async def get_geological_process(
    process_id: str,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get a geological process by ID"""
    try:
        process = await db.get_process_by_id(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Process not found")
        return process
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get geological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes")
async def search_geological_processes(
    search_term: Optional[str] = Query(None, description="Search term for process names/descriptions"),
    process_type: Optional[str] = Query(None, description="Filter by process type"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Search geological processes"""
    try:
        if search_term:
            results = await db.search_processes(search_term, process_type, limit)
        else:
            # Get all processes with optional type filter
            query = """
            MATCH (p:GeologicalProcess)
            """ + (f"WHERE p.process_type = '{process_type}'" if process_type else "") + """
            RETURN p
            ORDER BY p.name
            LIMIT $limit
            """
            results = await db.execute_query(query, {"limit": limit})
            results = [{"process": record["p"], "relevance_score": 1.0} for record in results]
        
        return {"processes": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Failed to search geological processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships", response_model=Dict[str, str])
async def create_geological_relationship(
    relationship: GeologicalRelationship,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create a relationship between geological processes"""
    try:
        relationship_id = await db.create_relationship(relationship)
        
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

    except Exception as e:
        logger.error(f"Failed to create geological relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-chains/{process_id}")
async def find_causal_chains(
    process_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum chain depth"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find causal chains starting from a specific process"""
    try:
        causal_chains = await db.find_causal_chains(process_id, max_depth)
        return {
            "start_process_id": process_id,
            "causal_chains": [chain.dict() for chain in causal_chains],
            "count": len(causal_chains)
        }
    except Exception as e:
        logger.error(f"Failed to find causal chains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network-analysis")
async def analyze_process_network(
    process_ids: List[str],
    include_communities: bool = Query(True, description="Include community detection"),
    include_centrality: bool = Query(True, description="Include centrality measures"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze network properties of geological processes"""
    try:
        # Basic network analysis
        network_result = await db.analyze_process_network(process_ids)
        
        # Enhanced analysis with algorithms
        algorithms = GeologicalGraphAlgorithms(db)
        
        enhanced_result = network_result.dict()
        
        if include_communities:
            try:
                communities = await algorithms.detect_communities(process_ids)
                enhanced_result["communities"] = communities
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                enhanced_result["communities"] = {}
        
        if include_centrality:
            try:
                centrality_measures = await algorithms.calculate_centrality_measures(process_ids)
                enhanced_result["enhanced_centrality_measures"] = centrality_measures
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {e}")
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Failed to analyze process network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shortest-paths")
async def find_shortest_paths(
    source_id: str = Query(..., description="Source process ID"),
    target_id: str = Query(..., description="Target process ID"),
    relationship_types: Optional[List[str]] = Query(None, description="Filter by relationship types"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find shortest paths between two geological processes"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        paths = await algorithms.find_shortest_paths(source_id, target_id, relationship_types)
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "paths": paths,
            "count": len(paths)
        }
    except Exception as e:
        logger.error(f"Failed to find shortest paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critical-processes")
async def find_critical_processes(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    criteria: str = Query("centrality", description="Criticality criteria (centrality, connectivity)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find critical processes in the geological network"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        critical_processes = await algorithms.find_critical_processes(process_ids, criteria)
        
        return {
            "criteria": criteria,
            "critical_processes": critical_processes,
            "count": len(critical_processes)
        }
    except Exception as e:
        logger.error(f"Failed to find critical processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cascade-analysis/{trigger_process_id}")
async def analyze_cascade_effects(
    trigger_process_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum cascade depth"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze potential cascade effects from a trigger process"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        cascade_analysis = await algorithms.analyze_cascade_effects(trigger_process_id, max_depth)
        
        return cascade_analysis
    except Exception as e:
        logger.error(f"Failed to analyze cascade effects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualization/config")
async def create_visualization_config(
    config: GraphVisualizationConfig,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create visualization configuration for graph data"""
    try:
        # Store visualization config in database
        query = """
        CREATE (v:VisualizationConfig {
            visualization_id: $visualization_id,
            layout_algorithm: $layout_algorithm,
            node_size_property: $node_size_property,
            node_color_property: $node_color_property,
            edge_width_property: $edge_width_property,
            show_node_labels: $show_node_labels,
            show_edge_labels: $show_edge_labels,
            min_edge_strength: $min_edge_strength,
            export_format: $export_format,
            include_metadata: $include_metadata,
            created_at: datetime()
        })
        RETURN v.visualization_id as visualization_id
        """
        
        parameters = config.dict()
        result = await db.execute_write_transaction(query, parameters)
        
        
class AdvancedGeologicalDatabase:
    """Advanced database operations for geological data with spatial indexing and time-series optimization"""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.postgres_params = connection_params.get('postgres', {})
        self.influx_params = connection_params.get('influxdb', {})
        self.neo4j_params = connection_params.get('neo4j', {})
        self.redis_params = connection_params.get('redis', {})
        
        self.connections = {}
        self._initialize_connections()
        
    def _initialize_connections(self):
        """Initialize database connections with connection pooling"""
        import psycopg2.pool
        import influxdb_client
        from neo4j import GraphDatabase
        import redis
        
        # PostgreSQL connection pool
        if self.postgres_params:
            self.connections['postgres'] = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, **self.postgres_params
            )
        
        # InfluxDB client
        if self.influx_params:
            self.connections['influxdb'] = influxdb_client.InfluxDBClient(
                url=self.influx_params['url'],
                token=self.influx_params['token'],
                org=self.influx_params['org']
            )
        
        # Neo4j driver
        if self.neo4j_params:
            self.connections['neo4j'] = GraphDatabase.driver(
                self.neo4j_params['uri'],
                auth=(self.neo4j_params['user'], self.neo4j_params['password'])
            )
        
        # Redis connection
        if self.redis_params:
            self.connections['redis'] = redis.Redis(**self.redis_params)
    
    def store_seismic_data(self, seismic_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store seismic data with spatial and temporal indexing"""
        
        stored_count = 0
        errors = []
        
        # Get PostgreSQL connection
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                # Create spatial index if not exists
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_seismic_location 
                    ON seismic_events USING GIST (location);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_time 
                    ON seismic_events (event_time);
                    
                    CREATE INDEX IF NOT EXISTS idx_seismic_magnitude 
                    ON seismic_events (magnitude);
                """)
                
                # Prepare batch insert
                insert_query = """
                    INSERT INTO seismic_events 
                    (event_id, event_time, location, magnitude, depth, event_type, 
                     waveform_data, frequency_content, uncertainty_metrics)
                    VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO UPDATE SET
                        magnitude = EXCLUDED.magnitude,
                        depth = EXCLUDED.depth,
                        event_type = EXCLUDED.event_type,
                        waveform_data = EXCLUDED.waveform_data,
                        frequency_content = EXCLUDED.frequency_content,
                        uncertainty_metrics = EXCLUDED.uncertainty_metrics;
                """
                
                # Process events in batches
                batch_size = 1000
                for i in range(0, len(seismic_events), batch_size):
                    batch = seismic_events[i:i + batch_size]
                    batch_data = []
                    
                    for event in batch:
                        try:
                            # Convert location to WKT format
                            location_wkt = f"POINT({event['longitude']} {event['latitude']})"
                            
                            batch_data.append((
                                event['event_id'],
                                event['event_time'],
                                location_wkt,
                                event['magnitude'],
                                event['depth'],
                                event['event_type'],
                                json.dumps(event.get('waveform_data', {})),
                                json.dumps(event.get('frequency_content', {})),
                                json.dumps(event.get('uncertainty_metrics', {}))
                            ))
                            
                        except Exception as e:
                            errors.append(f"Event {event.get('event_id', 'unknown')}: {str(e)}")
                    
                    # Execute batch insert
                    cursor.executemany(insert_query, batch_data)
                    stored_count += len(batch_data)
                
                postgres_conn.commit()
                
        except Exception as e:
            postgres_conn.rollback()
            errors.append(f"Database error: {str(e)}")
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
        
        # Store time-series data in InfluxDB
        if 'influxdb' in self.connections:
            self._store_timeseries_data(seismic_events)
        
        return {
            'stored_count': stored_count,
            'total_events': len(seismic_events),
            'errors': errors,
            'success_rate': stored_count / len(seismic_events) if seismic_events else 0
        }
    
    def _store_timeseries_data(self, seismic_events: List[Dict[str, Any]]):
        """Store time-series data in InfluxDB"""
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        write_api = self.connections['influxdb'].write_api(write_options=SYNCHRONOUS)
        
        points = []
        for event in seismic_events:
            # Create point for magnitude time series
            point = {
                "measurement": "seismic_magnitude",
                "tags": {
                    "event_type": event.get('event_type', 'unknown'),
                    "region": event.get('region', 'unknown'),
                    "station": event.get('station', 'unknown')
                },
                "fields": {
                    "magnitude": float(event.get('magnitude', 0)),
                    "depth": float(event.get('depth', 0)),
                    "latitude": float(event.get('latitude', 0)),
                    "longitude": float(event.get('longitude', 0))
                },
                "time": event.get('event_time')
            }
            points.append(point)
            
            # Store waveform data if available
            if 'waveform_data' in event:
                waveform_point = {
                    "measurement": "seismic_waveform",
                    "tags": {
                        "event_id": event.get('event_id'),
                        "component": "combined"
                    },
                    "fields": {
                        "amplitude": float(np.max(np.abs(event['waveform_data']))),
                        "rms": float(np.sqrt(np.mean(np.square(event['waveform_data'])))),
                        "peak_frequency": float(event.get('frequency_content', {}).get('dominant_frequency', 0))
                    },
                    "time": event.get('event_time')
                }
                points.append(waveform_point)
        
        try:
            write_api.write(bucket="geological_data", record=points)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
    
    def query_spatial_events(self, bbox: Tuple[float, float, float, float],
                           time_range: Tuple[str, str],
                           magnitude_range: Tuple[float, float] = (0, 10)) -> List[Dict[str, Any]]:
        """Query seismic events within spatial and temporal bounds"""
        
        min_lon, min_lat, max_lon, max_lat = bbox
        start_time, end_time = time_range
        min_mag, max_mag = magnitude_range
        
        postgres_conn = self.connections['postgres'].getconn()
        
        try:
            with postgres_conn.cursor() as cursor:
                query = """
                    SELECT event_id, event_time, ST_X(location) as longitude, 
                           ST_Y(location) as latitude, magnitude, depth, event_type,
                           waveform_data, frequency_content, uncertainty_metrics
                    FROM seismic_events
                    WHERE location && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
                      AND event_time BETWEEN %s AND %s
                      AND magnitude BETWEEN %s AND %s
                    ORDER BY event_time DESC
                    LIMIT 10000;
                """
                
                cursor.execute(query, (min_lon, min_lat, max_lon, max_lat, 
                                     start_time, end_time, min_mag, max_mag))
                
                results = []
                for row in cursor.fetchall():
                    event = {
                        'event_id': row[0],
                        'event_time': row[1],
                        'longitude': row[2],
                        'latitude': row[3],
                        'magnitude': row[4],
                        'depth': row[5],
                        'event_type': row[6],
                        'waveform_data': json.loads(row[7]) if row[7] else {},
                        'frequency_content': json.loads(row[8]) if row[8] else {},
                        'uncertainty_metrics': json.loads(row[9]) if row[9] else {}
                    }
                    results.append(event)
                
                return results
                
        except Exception as e:
            logger.error(f"Spatial query error: {e}")
            return []
        
        finally:
            self.connections['postgres'].putconn(postgres_conn)
    
    def create_geological_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create geological process relationships in Neo4j graph database"""
        
        if 'neo4j' not in self.connections:
            return {'error': 'Neo4j connection not available'}
        
        created_count = 0
        errors = []
        
        with self.connections['neo4j'].session() as session:
            try:
                # Create nodes and relationships
                for rel in relationships:
                    try:
                        # Create or update source node
                        session.run("""
                            MERGE (source:GeologicalProcess {id: $source_id})
                            SET source.name = $source_name,
                                source.type = $source_type,
                                source.properties = $source_props
                        """, 
                        source_id=rel['source_id'],
                        source_name=rel['source_name'],
                        source_type=rel['source_type'],
                        source_props=json.dumps(rel.get('source_properties', {})))
                        
                        # Create or update target node
                        session.run("""
                            MERGE (target:GeologicalProcess {id: $target_id})
                            SET target.name = $target_name,
                                target.type = $target_type,
                                target.properties = $target_props
                        """,
                        target_id=rel['target_id'],
                        target_name=rel['target_name'],
                        target_type=rel['target_type'],
                        target_props=json.dumps(rel.get('target_properties', {})))
                        
                        # Create relationship
                        session.run("""
                            MATCH (source:GeologicalProcess {id: $source_id})
                            MATCH (target:GeologicalProcess {id: $target_id})
                            MERGE (source)-[r:INFLUENCES {type: $rel_type}]->(target)
                            SET r.strength = $strength,
                                r.confidence = $confidence,
                                r.temporal_lag = $temporal_lag,
                                r.properties = $rel_props
                        """,
                        source_id=rel['source_id'],
                        target_id=rel['target_id'],
                        rel_type=rel['relationship_type'],
                        strength=rel.get('strength', 1.0),
                        confidence=rel.get('confidence', 0.5),
                        temporal_lag=rel.get('temporal_lag', 0),
                        rel_props=json.dumps(rel.get('relationship_properties', {})))
                        
                        created_count += 1
                        
                    except Exception as e:
                        errors.append(f"Relationship {rel.get('source_id', 'unknown')} -> {rel.get('target_id', 'unknown')}: {str(e)}")
                
            except Exception as e:
                errors.append(f"Neo4j session error: {str(e)}")
        
        return {
            'created_count': created_count,
            'total_relationships': len(relationships),
            'errors': errors,
            'success_rate': created_count / len(relationships) if relationships else 0
        }
    
    def cache_analysis_results(self, analysis_id: str, results: Dict[str, Any], 
                             expiration_hours: int = 24) -> bool:
        """Cache analysis results in Redis with expiration"""
        
        if 'redis' not in self.connections:
            return False
        
        try:
            redis_client = self.connections['redis']
            
            # Store results with expiration
            cache_key = f"analysis:{analysis_id}"
            redis_client.setex(
                cache_key,
                expiration_hours * 3600,
                json.dumps(results, default=str)
            )
            
            # Store metadata
            metadata_key = f"analysis_meta:{analysis_id}"
            metadata = {
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=expiration_hours)).isoformat(),
                'size_bytes': len(json.dumps(results, default=str))
            }
            redis_client.setex(
                metadata_key,
                expiration_hours * 3600,
                json.dumps(metadata)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache error: {e}")
            return False
    
    def get_cached_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached analysis results"""
        
        if 'redis' not in self.connections:
            return None
        
        try:
            redis_client = self.connections['redis']
            cache_key = f"analysis:{analysis_id}"
            
            cached_data = redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Redis retrieval error: {e}")
            return None

    except Exception as e:
        logger.error(f"Failed to create visualization config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/{visualization_id}/data")
async def get_visualization_data(
    visualization_id: str,
    process_ids: List[str] = Query(..., description="Process IDs to visualize"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get graph data formatted for visualization"""
    try:
        # Get visualization config
        config_query = """
        MATCH (v:VisualizationConfig {visualization_id: $visualization_id})
        RETURN v
        """
        config_results = await db.execute_query(config_query, {"visualization_id": visualization_id})
        
        if not config_results:
            raise HTTPException(status_code=404, detail="Visualization config not found")
        
        config = config_results[0]["v"]
        
        # Get network data
        network_query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        AND r.strength >= $min_edge_strength
        RETURN p, collect(DISTINCT {
            relationship: r,
            connected_process: connected
        }) as connections
        """
        
        parameters = {
            "process_ids": process_ids,
            "min_edge_strength": config.get("min_edge_strength", 0.1)
        }
        
        network_results = await db.execute_query(network_query, parameters)
        
        # Format data for visualization
        nodes = []
        edges = []
        edge_set = set()  # To avoid duplicate edges
        
        for record in network_results:
            process = record["p"]
            connections = record["connections"]
            
            # Add node
            node_data = {
                "id": process["process_id"],
                "label": process["name"] if config.get("show_node_labels", True) else "",
                "type": process["process_type"],
                "size": process.get(config.get("node_size_property", "magnitude"), 10),
                "color": process["process_type"],  # Color by type by default
                "metadata": {
                    "description": process.get("description", ""),
                    "location_id": process.get("location_id", ""),
                    "depth_range": process.get("depth_range", []),
                    "temperature_range": process.get("temperature_range", [])
                } if config.get("include_metadata", True) else {}
            }
            nodes.append(node_data)
            
            # Add edges
            for conn in connections:
                if conn["relationship"] and conn["connected_process"]:
                    rel = conn["relationship"]
                    target = conn["connected_process"]
                    
                    edge_key = tuple(sorted([process["process_id"], target["process_id"]]))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        
                        edge_data = {
                            "source": process["process_id"],
                            "target": target["process_id"],
                            "label": rel["relationship_type"] if config.get("show_edge_labels", False) else "",
                            "weight": rel["strength"],
                            "width": rel.get(config.get("edge_width_property", "strength"), 1),
                            "type": rel["relationship_type"],
                            "metadata": {
                                "confidence": rel.get("confidence", 1.0),
                                "mechanism": rel.get("mechanism", ""),
                                "time_delay": rel.get("time_delay", None)
                            } if config.get("include_metadata", True) else {}
                        }
                        edges.append(edge_data)
        
        visualization_data = {
            "visualization_id": visualization_id,
            "config": config,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "generated_at": datetime.now().isoformat()
        }
        
        return visualization_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def execute_graph_query(
    query: GraphQuery,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Execute a complex graph query"""
    try:
        if query.query_type == "causal_chain":
            if not query.start_processes:
                raise HTTPException(status_code=400, detail="Start processes required for causal chain query")
            
            results = []
            for start_process in query.start_processes:
                chains = await db.find_causal_chains(start_process, query.max_depth or 5)
                results.extend([chain.dict() for chain in chains])
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": results,
                "count": len(results)
            }
        
        elif query.query_type == "network_analysis":
            if not query.start_processes:
                raise HTTPException(status_code=400, detail="Process IDs required for network analysis")
            
            network_result = await db.analyze_process_network(query.start_processes)
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": network_result.dict(),
                "count": 1
            }
        
        elif query.query_type == "pattern_search":
            # Custom pattern search implementation
            cypher_query = await _build_pattern_search_query(query)
            results = await db.execute_query(cypher_query, query.dict())
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": results,
                "count": len(results)
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported query type: {query.query_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute graph query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _build_pattern_search_query(query: GraphQuery) -> str:
    """Build Cypher query for pattern search"""
    
    # Base query structure
    cypher_parts = ["MATCH (p:GeologicalProcess)"]
    
    # Add filters
    where_conditions = []
    
    if query.process_types:
        process_types = "', '".join(query.process_types)
        where_conditions.append(f"p.process_type IN ['{process_types}']")
    
    if query.spatial_bounds:
        # Assuming spatial bounds are provided as lat/lon bounds
        bounds = query.spatial_bounds
        if all(key in bounds for key in ["min_lat", "max_lat", "min_lon", "max_lon"]):
            where_conditions.append(
                f"p.latitude >= {bounds['min_lat']} AND p.latitude <= {bounds['max_lat']} "
                f"AND p.longitude >= {bounds['min_lon']} AND p.longitude <= {bounds['max_lon']}"
            )
    
    # Add relationship patterns if specified
    if query.relationship_types:
        rel_types = "|".join(query.relationship_types)
        cypher_parts.append(f"OPTIONAL MATCH (p)-[r:{rel_types}]-(connected:GeologicalProcess)")
        
        if query.min_strength:
            where_conditions.append(f"r.strength >= {query.min_strength}")
        
        if query.min_confidence:
            where_conditions.append(f"r.confidence >= {query.min_confidence}")
    
    # Add WHERE clause if conditions exist
    if where_conditions:
        cypher_parts.append("WHERE " + " AND ".join(where_conditions))
    
    # Add RETURN clause
    if query.relationship_types:
        cypher_parts.append("RETURN p, collect(DISTINCT {relationship: r, connected: connected}) as relationships")
    else:
        cypher_parts.append("RETURN p")
    
    # Add LIMIT
    cypher_parts.append(f"LIMIT {query.max_results}")
    
    return " ".join(cypher_parts)


@router.get("/statistics")
async def get_graph_statistics(
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get overall graph database statistics"""
    try:
        stats_query = """
        MATCH (p:GeologicalProcess)
        OPTIONAL MATCH (p)-[r:INFLUENCES]-()
        RETURN 
            count(DISTINCT p) as total_processes,
            count(DISTINCT r) as total_relationships,
            collect(DISTINCT p.process_type) as process_types,
            avg(r.strength) as avg_relationship_strength,
            avg(r.confidence) as avg_relationship_confidence
        """
        
        results = await db.execute_query(stats_query)
        
        if results:
            stats = results[0]
            return {
                "total_processes": stats["total_processes"],
                "total_relationships": stats["total_relationships"],
                "process_types": stats["process_types"],
                "average_relationship_strength": stats["avg_relationship_strength"],
                "average_relationship_confidence": stats["avg_relationship_confidence"],
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "total_processes": 0,
                "total_relationships": 0,
                "process_types": [],
                "average_relationship_strength": 0.0,
                "average_relationship_confidence": 0.0,
                "generated_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/temporal-evolution")
async def analyze_temporal_evolution(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    time_window_days: int = Query(365, description="Time window in days"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze temporal evolution of the geological network"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        temporal_analysis = await analytics.analyze_temporal_evolution(process_ids, time_window_days)
        return temporal_analysis
    except Exception as e:
        logger.error(f"Failed to analyze temporal evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/anomaly-detection")
async def detect_anomalous_patterns(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    threshold_std_dev: float = Query(2.0, description="Standard deviation threshold for anomaly detection"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Detect anomalous patterns in the geological network"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        anomalies = await analytics.detect_anomalous_patterns(process_ids, threshold_std_dev)
        return anomalies
    except Exception as e:
        logger.error(f"Failed to detect anomalous patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/cascade-prediction/{trigger_process_id}")
async def predict_cascade_probability(
    trigger_process_id: str,
    target_process_ids: List[str] = Query(..., description="Target process IDs"),
    time_horizon_days: int = Query(30, description="Time horizon for prediction in days"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Predict probability of cascade effects reaching target processes"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        predictions = await analytics.predict_cascade_probability(
            trigger_process_id, target_process_ids, time_horizon_days
        )
        return predictions
    except Exception as e:
        logger.error(f"Failed to predict cascade probability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/resilience-analysis")
async def analyze_network_resilience(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    failure_scenarios: Optional[List[str]] = Query(None, description="Specific failure scenarios to test"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze network resilience to process failures"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        resilience = await analytics.analyze_network_resilience(process_ids, failure_scenarios)
        return resilience
    except Exception as e:
        logger.error(f"Failed to analyze network resilience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/process-clustering")
async def identify_process_clusters(
    process_ids: List[str] = Query(..., description="Process IDs to cluster"),
    clustering_method: str = Query("spatial_temporal", description="Clustering method (spatial_temporal, process_type, strength_based)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Identify clusters of related geological processes"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        clusters = await analytics.identify_process_clusters(process_ids, clustering_method)
        return clusters
    except Exception as e:
        logger.error(f"Failed to identify process clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced-analytics/comprehensive-report")
async def generate_comprehensive_report(
    process_ids: List[str],
    include_predictions: bool = Query(True, description="Include cascade predictions"),
    include_anomalies: bool = Query(True, description="Include anomaly detection"),
    include_resilience: bool = Query(True, description="Include resilience analysis"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Generate comprehensive network analysis report"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        report = await analytics.generate_network_report(
            process_ids, include_predictions, include_anomalies, include_resilience
        )
        return report
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-discovery/hidden-relationships")
async def discover_hidden_causality(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    confidence_threshold: float = Query(0.7, description="Confidence threshold for discovery"),
    strength_threshold: float = Query(0.5, description="Strength threshold for discovery"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Discover hidden causal relationships using advanced graph analysis"""
    try:
        causal_engine = CausalDiscoveryEngine(db)
        hidden_relationships = await causal_engine.discover_hidden_causality(
            process_ids, confidence_threshold, strength_threshold
        )
        return hidden_relationships
    except Exception as e:
        logger.error(f"Failed to discover hidden causality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal-discovery/validate-relationships")
async def validate_causal_relationships(
    proposed_relationships: List[Dict[str, Any]],
    validation_method: str = Query("statistical", description="Validation method (statistical, domain_knowledge)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Validate proposed causal relationships"""
    try:
        causal_engine = CausalDiscoveryEngine(db)
        validation_results = await causal_engine.validate_causal_relationships(
            proposed_relationships, validation_method
        )
        return validation_results
    except Exception as e:
        logger.error(f"Failed to validate causal relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))