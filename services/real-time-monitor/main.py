#!/usr/bin/env python3
"""
Real-Time Geological Process Monitor Service

This service provides real-time monitoring of geological processes with causal context,
streaming data visualization, alert generation, and anomaly detection capabilities.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import redis
import asyncpg
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessType(str, Enum):
    SEISMIC = "seismic"
    VOLCANIC = "volcanic"
    DEFORMATION = "deformation"
    GEOCHEMICAL = "geochemical"
    THERMAL = "thermal"

class AlertLevel(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(str, Enum):
    THRESHOLD = "threshold"
    ANOMALY = "anomaly"
    CAUSAL_CHANGE = "causal_change"
    PREDICTIVE = "predictive"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class GeologicalProcess:
    id: str
    name: str
    type: ProcessType
    location: Dict[str, float]  # lat, lon, depth
    current_value: float
    unit: str
    trend: str
    alert_level: AlertLevel
    causal_strength: float
    last_update: datetime
    thresholds: Dict[str, float]
    quality_score: float = 1.0

@dataclass
class CausalRelationship:
    source_process: str
    target_process: str
    strength: float
    confidence: float
    lag: float  # in seconds
    relationship_type: str
    last_updated: datetime
    statistical_significance: float

@dataclass
class ProcessAlert:
    id: str
    process_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime
    causal_context: List[str]
    acknowledged: bool = False
    metadata: Dict[str, Any] = None

@dataclass
class StreamingDataPoint:
    timestamp: datetime
    process_id: str
    value: float
    quality: float
    causal_influence: float
    metadata: Dict[str, Any] = None

@dataclass
class AnomalyDetection:
    process_id: str
    anomaly_score: float
    threshold: float
    causal_pathway: List[str]
    confidence: float
    detection_method: str
    timestamp: datetime

class CausalAnalysisEngine:
    """Advanced causal analysis engine for geological processes"""
    
    def __init__(self):
        self.causal_models = {}
        self.temporal_lags = {}
        self.confidence_thresholds = {
            'weak': 0.3,
            'moderate': 0.6,
            'strong': 0.8
        }
    
    async def analyze_causal_relationships(self, 
                                        data: Dict[str, List[StreamingDataPoint]]) -> List[CausalRelationship]:
        """Analyze causal relationships between geological processes"""
        relationships = []
        
        # Convert data to time series format
        time_series = self._prepare_time_series(data)
        
        if len(time_series) < 2:
            return relationships
        
        # Perform pairwise causal analysis
        process_ids = list(time_series.keys())
        for i, source_id in enumerate(process_ids):
            for j, target_id in enumerate(process_ids):
                if i != j:
                    relationship = await self._compute_causal_relationship(
                        source_id, target_id, time_series
                    )
                    if relationship and relationship.confidence > self.confidence_thresholds['weak']:
                        relationships.append(relationship)
        
        return relationships
    
    def _prepare_time_series(self, data: Dict[str, List[StreamingDataPoint]]) -> Dict[str, pd.Series]:
        """Convert streaming data to pandas time series"""
        time_series = {}
        
        for process_id, points in data.items():
            if len(points) < 10:  # Need minimum data points
                continue
                
            df = pd.DataFrame([
                {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'quality': point.quality
                }
                for point in points
            ])
            
            # Filter by quality and sort by timestamp
            df = df[df['quality'] > 0.7].sort_values('timestamp')
            
            if len(df) >= 10:
                df.set_index('timestamp', inplace=True)
                time_series[process_id] = df['value']
        
        return time_series
    
    async def _compute_causal_relationship(self, 
                                         source_id: str, 
                                         target_id: str,
                                         time_series: Dict[str, pd.Series]) -> Optional[CausalRelationship]:
        """Compute causal relationship between two processes using Granger causality"""
        try:
            source_series = time_series[source_id]
            target_series = time_series[target_id]
            
            # Align time series
            aligned_data = pd.concat([source_series, target_series], axis=1, join='inner')
            aligned_data.columns = ['source', 'target']
            
            if len(aligned_data) < 20:
                return None
            
            # Compute Granger causality
            from statsmodels.tsa.stattools import grangercausalitytests
            
            # Test different lag orders
            max_lag = min(10, len(aligned_data) // 4)
            best_lag = 1
            best_p_value = 1.0
            
            for lag in range(1, max_lag + 1):
                try:
                    result = grangercausalitytests(aligned_data[['target', 'source']], 
                                                 maxlag=lag, verbose=False)
                    p_value = result[lag][0]['ssr_ftest'][1]
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_lag = lag
                except:
                    continue
            
            # Calculate causal strength (1 - p_value)
            causal_strength = max(0, 1 - best_p_value)
            confidence = 1 - best_p_value if best_p_value < 0.05 else 0
            
            # Calculate temporal lag in seconds (assuming regular intervals)
            time_diff = aligned_data.index[1] - aligned_data.index[0]
            lag_seconds = best_lag * time_diff.total_seconds()
            
            return CausalRelationship(
                source_process=source_id,
                target_process=target_id,
                strength=causal_strength,
                confidence=confidence,
                lag=lag_seconds,
                relationship_type="granger_causal",
                last_updated=datetime.now(),
                statistical_significance=best_p_value
            )
            
        except Exception as e:
            logger.error(f"Error computing causal relationship: {e}")
            return None

class AnomalyDetector:
    """Advanced anomaly detection for geological processes"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.baseline_data = {}
        
    async def detect_anomalies(self, 
                             data: Dict[str, List[StreamingDataPoint]],
                             causal_relationships: List[CausalRelationship]) -> List[AnomalyDetection]:
        """Detect anomalies in geological processes with causal context"""
        anomalies = []
        
        for process_id, points in data.items():
            if len(points) < 50:  # Need sufficient data for anomaly detection
                continue
                
            anomaly = await self._detect_process_anomaly(process_id, points, causal_relationships)
            if anomaly:
                anomalies.append(anomaly)
        
        return anomalies
    
    async def _detect_process_anomaly(self, 
                                    process_id: str,
                                    points: List[StreamingDataPoint],
                                    causal_relationships: List[CausalRelationship]) -> Optional[AnomalyDetection]:
        """Detect anomalies for a specific process"""
        try:
            # Prepare feature matrix
            features = self._extract_features(points)
            
            if process_id not in self.models:
                # Initialize and train isolation forest
                self.models[process_id] = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                self.scalers[process_id] = StandardScaler()
                
                # Use historical data for training (first 80% of points)
                train_size = int(len(features) * 0.8)
                train_features = features[:train_size]
                
                scaled_features = self.scalers[process_id].fit_transform(train_features)
                self.models[process_id].fit(scaled_features)
                
                # Set baseline threshold
                baseline_scores = self.models[process_id].decision_function(scaled_features)
                self.baseline_data[process_id] = {
                    'threshold': np.percentile(baseline_scores, 5),  # 5th percentile as threshold
                    'mean_score': np.mean(baseline_scores),
                    'std_score': np.std(baseline_scores)
                }
            
            # Detect anomalies in recent data
            recent_features = features[-10:]  # Last 10 data points
            scaled_recent = self.scalers[process_id].transform(recent_features)
            anomaly_scores = self.models[process_id].decision_function(scaled_recent)
            
            # Check if any recent points are anomalous
            threshold = self.baseline_data[process_id]['threshold']
            anomalous_indices = np.where(anomaly_scores < threshold)[0]
            
            if len(anomalous_indices) > 0:
                # Find causal pathway for this process
                causal_pathway = self._find_causal_pathway(process_id, causal_relationships)
                
                # Calculate confidence based on how far below threshold
                worst_score = np.min(anomaly_scores)
                confidence = min(1.0, abs(worst_score - threshold) / abs(threshold))
                
                return AnomalyDetection(
                    process_id=process_id,
                    anomaly_score=float(worst_score),
                    threshold=float(threshold),
                    causal_pathway=causal_pathway,
                    confidence=confidence,
                    detection_method="isolation_forest",
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Error detecting anomaly for {process_id}: {e}")
        
        return None
    
    def _extract_features(self, points: List[StreamingDataPoint]) -> np.ndarray:
        """Extract features from streaming data points"""
        values = [p.value for p in points]
        qualities = [p.quality for p in points]
        influences = [p.causal_influence for p in points]
        
        # Calculate rolling statistics
        window_size = min(10, len(values))
        features = []
        
        for i in range(window_size - 1, len(values)):
            window_values = values[i - window_size + 1:i + 1]
            window_qualities = qualities[i - window_size + 1:i + 1]
            window_influences = influences[i - window_size + 1:i + 1]
            
            feature_vector = [
                values[i],  # Current value
                np.mean(window_values),  # Mean
                np.std(window_values),   # Standard deviation
                np.max(window_values) - np.min(window_values),  # Range
                np.mean(window_qualities),  # Average quality
                np.mean(window_influences),  # Average causal influence
                len([v for j, v in enumerate(window_values[1:], 1) 
                     if v > window_values[j-1]]) / (window_size - 1),  # Trend ratio
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _find_causal_pathway(self, 
                           process_id: str, 
                           causal_relationships: List[CausalRelationship]) -> List[str]:
        """Find causal pathway leading to this process"""
        pathway = []
        
        # Find direct causal influences
        direct_influences = [
            rel.source_process for rel in causal_relationships
            if rel.target_process == process_id and rel.confidence > 0.5
        ]
        
        # Sort by causal strength
        influence_strengths = {
            rel.source_process: rel.strength for rel in causal_relationships
            if rel.target_process == process_id and rel.confidence > 0.5
        }
        
        sorted_influences = sorted(direct_influences, 
                                 key=lambda x: influence_strengths.get(x, 0), 
                                 reverse=True)
        
        pathway.extend(sorted_influences[:3])  # Top 3 influences
        pathway.append(process_id)
        
        return pathway

class AlertManager:
    """Manages alert generation and processing"""
    
    def __init__(self):
        self.alert_history = []
        self.alert_rules = {}
        self.predictive_models = {}
    
    async def generate_alerts(self, 
                            processes: Dict[str, GeologicalProcess],
                            anomalies: List[AnomalyDetection],
                            causal_relationships: List[CausalRelationship]) -> List[ProcessAlert]:
        """Generate alerts based on current system state"""
        alerts = []
        
        # Threshold-based alerts
        for process in processes.values():
            threshold_alert = self._check_threshold_alert(process)
            if threshold_alert:
                alerts.append(threshold_alert)
        
        # Anomaly-based alerts
        for anomaly in anomalies:
            anomaly_alert = self._create_anomaly_alert(anomaly)
            alerts.append(anomaly_alert)
        
        # Causal change alerts
        causal_alerts = await self._check_causal_change_alerts(causal_relationships)
        alerts.extend(causal_alerts)
        
        # Predictive alerts
        predictive_alerts = await self._generate_predictive_alerts(processes, causal_relationships)
        alerts.extend(predictive_alerts)
        
        return alerts
    
    def _check_threshold_alert(self, process: GeologicalProcess) -> Optional[ProcessAlert]:
        """Check if process exceeds defined thresholds"""
        thresholds = process.thresholds
        current_value = process.current_value
        
        if 'critical' in thresholds and current_value > thresholds['critical']:
            return ProcessAlert(
                id=f"threshold_{process.id}_{int(time.time())}",
                process_id=process.id,
                alert_type=AlertType.THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                message=f"{process.name} exceeded critical threshold: {current_value} {process.unit}",
                timestamp=datetime.now(),
                causal_context=[],
                metadata={'threshold_type': 'critical', 'threshold_value': thresholds['critical']}
            )
        elif 'high' in thresholds and current_value > thresholds['high']:
            return ProcessAlert(
                id=f"threshold_{process.id}_{int(time.time())}",
                process_id=process.id,
                alert_type=AlertType.THRESHOLD,
                severity=AlertSeverity.HIGH,
                message=f"{process.name} exceeded high threshold: {current_value} {process.unit}",
                timestamp=datetime.now(),
                causal_context=[],
                metadata={'threshold_type': 'high', 'threshold_value': thresholds['high']}
            )
        
        return None
    
    def _create_anomaly_alert(self, anomaly: AnomalyDetection) -> ProcessAlert:
        """Create alert from anomaly detection"""
        severity = AlertSeverity.HIGH if anomaly.confidence > 0.8 else AlertSeverity.MEDIUM
        
        return ProcessAlert(
            id=f"anomaly_{anomaly.process_id}_{int(time.time())}",
            process_id=anomaly.process_id,
            alert_type=AlertType.ANOMALY,
            severity=severity,
            message=f"Anomaly detected in {anomaly.process_id}: score {anomaly.anomaly_score:.3f}",
            timestamp=anomaly.timestamp,
            causal_context=anomaly.causal_pathway,
            metadata={
                'anomaly_score': anomaly.anomaly_score,
                'threshold': anomaly.threshold,
                'detection_method': anomaly.detection_method
            }
        )
    
    async def _check_causal_change_alerts(self, 
                                        causal_relationships: List[CausalRelationship]) -> List[ProcessAlert]:
        """Check for significant changes in causal relationships"""
        alerts = []
        
        # Compare with historical causal strengths
        for relationship in causal_relationships:
            # This would compare with stored historical values
            # For now, we'll flag relationships with very high or very low confidence
            if relationship.confidence > 0.95:
                alerts.append(ProcessAlert(
                    id=f"causal_strong_{relationship.source_process}_{relationship.target_process}_{int(time.time())}",
                    process_id=relationship.target_process,
                    alert_type=AlertType.CAUSAL_CHANGE,
                    severity=AlertSeverity.MEDIUM,
                    message=f"Strong causal relationship detected: {relationship.source_process} → {relationship.target_process}",
                    timestamp=datetime.now(),
                    causal_context=[relationship.source_process],
                    metadata={'causal_strength': relationship.strength, 'confidence': relationship.confidence}
                ))
        
        return alerts
    
    async def _generate_predictive_alerts(self, 
                                        processes: Dict[str, GeologicalProcess],
                                        causal_relationships: List[CausalRelationship]) -> List[ProcessAlert]:
        """Generate predictive alerts based on causal models"""
        alerts = []
        
        # Simple predictive model based on causal influences
        for process in processes.values():
            # Find processes that causally influence this one
            influences = [
                rel for rel in causal_relationships
                if rel.target_process == process.id and rel.confidence > 0.6
            ]
            
            if influences:
                # Calculate predicted risk based on causal influences
                total_influence = sum(rel.strength * rel.confidence for rel in influences)
                
                if total_influence > 0.8:  # High causal influence threshold
                    alerts.append(ProcessAlert(
                        id=f"predictive_{process.id}_{int(time.time())}",
                        process_id=process.id,
                        alert_type=AlertType.PREDICTIVE,
                        severity=AlertSeverity.MEDIUM,
                        message=f"High causal influence detected for {process.name} - elevated risk predicted",
                        timestamp=datetime.now(),
                        causal_context=[rel.source_process for rel in influences],
                        metadata={'total_influence': total_influence, 'influence_count': len(influences)}
                    ))
        
        return alerts

class RealTimeMonitorService:
    """Main service class for real-time geological process monitoring"""
    
    def __init__(self):
        self.processes: Dict[str, GeologicalProcess] = {}
        self.streaming_data: Dict[str, List[StreamingDataPoint]] = {}
        self.causal_engine = CausalAnalysisEngine()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.connected_clients: Set[WebSocket] = set()
        self.redis_client = None
        self.db_pool = None
        
        # Data retention settings
        self.max_data_points = 1000
        self.data_retention_hours = 24
        
    async def initialize(self):
        """Initialize service connections and data"""
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Initialize PostgreSQL connection pool
            self.db_pool = await asyncpg.create_pool(
                "postgresql://prometheus:prometheus123@localhost:5432/prometheus"
            )
            
            # Load initial process configurations
            await self.load_process_configurations()
            
            logger.info("Real-time monitor service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise
    
    async def load_process_configurations(self):
        """Load geological process configurations from database"""
        # Mock process configurations for demonstration
        mock_processes = [
            GeologicalProcess(
                id="seismic_001",
                name="Seismic Activity - Mt. Vesuvius",
                type=ProcessType.SEISMIC,
                location={"lat": 40.8214, "lon": 14.4260, "depth": 0},
                current_value=2.3,
                unit="magnitude",
                trend="increasing",
                alert_level=AlertLevel.ELEVATED,
                causal_strength=0.75,
                last_update=datetime.now(),
                thresholds={"high": 4.0, "critical": 5.5},
                quality_score=0.95
            ),
            GeologicalProcess(
                id="deform_001",
                name="Ground Deformation - Campi Flegrei",
                type=ProcessType.DEFORMATION,
                location={"lat": 40.8270, "lon": 14.1390, "depth": 0},
                current_value=15.2,
                unit="mm/month",
                trend="increasing",
                alert_level=AlertLevel.HIGH,
                causal_strength=0.82,
                last_update=datetime.now(),
                thresholds={"high": 20.0, "critical": 30.0},
                quality_score=0.88
            ),
            GeologicalProcess(
                id="thermal_001",
                name="Thermal Anomaly - Stromboli",
                type=ProcessType.THERMAL,
                location={"lat": 38.7890, "lon": 15.2130, "depth": 0},
                current_value=850,
                unit="°C",
                trend="volatile",
                alert_level=AlertLevel.CRITICAL,
                causal_strength=0.68,
                last_update=datetime.now(),
                thresholds={"high": 800.0, "critical": 900.0},
                quality_score=0.92
            )
        ]
        
        for process in mock_processes:
            self.processes[process.id] = process
            self.streaming_data[process.id] = []
    
    async def add_streaming_data(self, data_point: StreamingDataPoint):
        """Add new streaming data point and maintain data retention"""
        process_id = data_point.process_id
        
        if process_id not in self.streaming_data:
            self.streaming_data[process_id] = []
        
        self.streaming_data[process_id].append(data_point)
        
        # Maintain data retention
        cutoff_time = datetime.now() - timedelta(hours=self.data_retention_hours)
        self.streaming_data[process_id] = [
            point for point in self.streaming_data[process_id]
            if point.timestamp > cutoff_time
        ]
        
        # Limit number of data points
        if len(self.streaming_data[process_id]) > self.max_data_points:
            self.streaming_data[process_id] = self.streaming_data[process_id][-self.max_data_points:]
        
        # Update process current value
        if process_id in self.processes:
            self.processes[process_id].current_value = data_point.value
            self.processes[process_id].last_update = data_point.timestamp
    
    async def process_real_time_analysis(self):
        """Perform real-time causal analysis and anomaly detection"""
        try:
            # Analyze causal relationships
            causal_relationships = await self.causal_engine.analyze_causal_relationships(
                self.streaming_data
            )
            
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(
                self.streaming_data, causal_relationships
            )
            
            # Generate alerts
            alerts = await self.alert_manager.generate_alerts(
                self.processes, anomalies, causal_relationships
            )
            
            # Broadcast updates to connected clients
            await self.broadcast_updates({
                'causal_relationships': causal_relationships,
                'anomalies': anomalies,
                'alerts': alerts
            })
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
    
    async def broadcast_updates(self, updates: Dict[str, Any]):
        """Broadcast updates to all connected WebSocket clients"""
        if not self.connected_clients:
            return
        
        # Prepare messages for different update types
        messages = []
        
        if 'causal_relationships' in updates:
            messages.append({
                'type': 'causal_relationship_update',
                'payload': [asdict(rel) for rel in updates['causal_relationships']]
            })
        
        if 'anomalies' in updates:
            messages.append({
                'type': 'anomaly_detection',
                'payload': [asdict(anomaly) for anomaly in updates['anomalies']]
            })
        
        if 'alerts' in updates:
            for alert in updates['alerts']:
                messages.append({
                    'type': 'alert',
                    'payload': asdict(alert)
                })
        
        # Send messages to all connected clients
        disconnected_clients = set()
        for client in self.connected_clients:
            try:
                for message in messages:
                    await client.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected_clients
    
    async def simulate_streaming_data(self):
        """Simulate streaming geological data for demonstration"""
        while True:
            try:
                current_time = datetime.now()
                
                for process_id, process in self.processes.items():
                    # Generate realistic data with some noise and trends
                    base_value = process.current_value
                    
                    # Add trend-based variation
                    if process.trend == "increasing":
                        trend_factor = 1.001 + np.random.normal(0, 0.0005)
                    elif process.trend == "decreasing":
                        trend_factor = 0.999 + np.random.normal(0, 0.0005)
                    elif process.trend == "volatile":
                        trend_factor = 1.0 + np.random.normal(0, 0.01)
                    else:  # stable
                        trend_factor = 1.0 + np.random.normal(0, 0.001)
                    
                    new_value = base_value * trend_factor
                    
                    # Calculate causal influence (simplified)
                    causal_influence = np.random.beta(2, 5)  # Skewed towards lower values
                    
                    # Generate quality score
                    quality = max(0.5, min(1.0, np.random.normal(0.9, 0.1)))
                    
                    data_point = StreamingDataPoint(
                        timestamp=current_time,
                        process_id=process_id,
                        value=new_value,
                        quality=quality,
                        causal_influence=causal_influence,
                        metadata={"source": "simulation", "sensor_id": f"sensor_{process_id}"}
                    )
                    
                    await self.add_streaming_data(data_point)
                    
                    # Broadcast streaming data to clients
                    await self.broadcast_updates({
                        'streaming_data': [data_point]
                    })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in data simulation: {e}")
                await asyncio.sleep(10)

# Global service instance
monitor_service = RealTimeMonitorService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await monitor_service.initialize()
    
    # Start background tasks
    analysis_task = asyncio.create_task(
        periodic_analysis()
    )
    simulation_task = asyncio.create_task(
        monitor_service.simulate_streaming_data()
    )
    
    yield
    
    # Shutdown
    analysis_task.cancel()
    simulation_task.cancel()

async def periodic_analysis():
    """Periodic causal analysis and anomaly detection"""
    while True:
        try:
            await monitor_service.process_real_time_analysis()
            await asyncio.sleep(30)  # Run analysis every 30 seconds
        except Exception as e:
            logger.error(f"Error in periodic analysis: {e}")
            await asyncio.sleep(60)

# FastAPI application
app = FastAPI(
    title="Real-Time Geological Process Monitor",
    description="Advanced real-time monitoring system for geological processes with causal analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/geological-monitoring")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring data"""
    await websocket.accept()
    monitor_service.connected_clients.add(websocket)
    
    try:
        # Send initial data
        initial_data = {
            'type': 'initial_data',
            'payload': {
                'processes': [asdict(process) for process in monitor_service.processes.values()],
                'streaming_data': {
                    process_id: [asdict(point) for point in points[-50:]]  # Last 50 points
                    for process_id, points in monitor_service.streaming_data.items()
                }
            }
        }
        await websocket.send_text(json.dumps(initial_data, default=str))
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong, configuration updates, etc.)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(message)
                
                # Handle client messages
                if data.get('type') == 'ping':
                    await websocket.send_text(json.dumps({'type': 'pong'}))
                
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_text(json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()}))
            
    except WebSocketDisconnect:
        monitor_service.connected_clients.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        monitor_service.connected_clients.discard(websocket)

@app.get("/api/v1/processes")
async def get_processes():
    """Get all geological processes"""
    return [asdict(process) for process in monitor_service.processes.values()]

@app.get("/api/v1/processes/{process_id}/data")
async def get_process_data(process_id: str, limit: int = 100):
    """Get streaming data for a specific process"""
    if process_id not in monitor_service.streaming_data:
        raise HTTPException(status_code=404, detail="Process not found")
    
    data = monitor_service.streaming_data[process_id][-limit:]
    return [asdict(point) for point in data]

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_clients": len(monitor_service.connected_clients),
        "active_processes": len(monitor_service.processes),
        "total_data_points": sum(len(points) for points in monitor_service.streaming_data.values())
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )