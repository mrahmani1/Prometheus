"""Stream processing pipeline for multi-sensor data fusion"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import structlog
import numpy as np
from collections import defaultdict, deque

from .kafka_producer import KafkaDataProducer
from .kafka_consumer import KafkaDataConsumer
from .redis_streams import RedisStreamsManager
from models.geological_data import (
    SeismicEvent, ThermalMeasurement, StrainMeasurement,
    DataQuality, ProcessedDataBatch
)

logger = structlog.get_logger()

class StreamProcessor:
    """Stream processing pipeline for geological data fusion and analysis"""
    
    def __init__(self, config: Any, data_processor: Any):
        self.config = config
        self.data_processor = data_processor
        self.is_initialized = False
        self.is_processing = False
        
        # Streaming components
        self.kafka_producer = KafkaDataProducer(config)
        self.kafka_consumer = KafkaDataConsumer(config)
        self.redis_streams = RedisStreamsManager(config)
        
        # Processing windows for data fusion
        self.fusion_windows = {
            "seismic": deque(maxlen=1000),      # Last 1000 seismic events
            "thermal": deque(maxlen=500),       # Last 500 thermal measurements
            "strain": deque(maxlen=2000),       # Last 2000 strain measurements
            "deformation": deque(maxlen=200)    # Last 200 deformation measurements
        }
        
        # Time windows for correlation analysis
        self.correlation_window = timedelta(minutes=5)  # 5-minute correlation window
        self.alert_threshold_window = timedelta(minutes=1)  # 1-minute alert window
        
        # Alert thresholds
        self.alert_thresholds = {
            "seismic_magnitude": 4.0,           # Magnitude 4.0+ earthquakes
            "thermal_anomaly": 50.0,            # 50K above baseline
            "strain_rate": 1e-6,                # High strain rate threshold
            "correlation_strength": 0.7         # Strong correlation threshold
        }
        
        # Processing statistics
        self.events_processed = 0
        self.alerts_generated = 0
        self.correlations_detected = 0
        
        # Background tasks
        self.processing_tasks = []
        
    async def initialize(self):
        """Initialize stream processing pipeline"""
        try:
            # Initialize streaming components
            await self.kafka_producer.initialize()
            await self.kafka_consumer.initialize()
            await self.redis_streams.initialize()
            
            # Register message handlers
            self._register_message_handlers()
            
            self.is_initialized = True
            logger.info("Stream processor initialized")
            
        except Exception as e:
            logger.error("Failed to initialize stream processor", error=str(e))
            raise
    
    def _register_message_handlers(self):
        """Register Kafka message handlers"""
        
        # Seismic events handler
        self.kafka_consumer.register_handler(
            f"{self.config.kafka_topic_prefix}.seismic.events",
            self._handle_seismic_event
        )
        
        # Thermal measurements handler
        self.kafka_consumer.register_handler(
            f"{self.config.kafka_topic_prefix}.thermal.measurements",
            self._handle_thermal_measurement
        )
        
        # Strain measurements handler
        self.kafka_consumer.register_handler(
            f"{self.config.kafka_topic_prefix}.strain.measurements",
            self._handle_strain_measurement
        )
        
        # Processing batches handler
        self.kafka_consumer.register_handler(
            f"{self.config.kafka_topic_prefix}.processing.batches",
            self._handle_processing_batch
        )
    
    async def start_processing(self):
        """Start stream processing"""
        if not self.is_initialized:
            raise RuntimeError("Stream processor not initialized")
        
        self.is_processing = True
        logger.info("Starting stream processing")
        
        # Start background tasks
        self.processing_tasks = [
            asyncio.create_task(self._kafka_consumer_task()),
            asyncio.create_task(self._correlation_analysis_task()),
            asyncio.create_task(self._alert_monitoring_task()),
            asyncio.create_task(self._data_fusion_task())
        ]
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.processing_tasks)
        except Exception as e:
            logger.error("Error in stream processing", error=str(e))
        finally:
            self.is_processing = False
    
    async def _kafka_consumer_task(self):
        """Kafka consumer background task"""
        try:
            await self.kafka_consumer.start_consuming()
        except Exception as e:
            logger.error("Error in Kafka consumer task", error=str(e))
    
    async def _handle_seismic_event(self, message_context: Dict[str, Any]):
        """Handle seismic event message"""
        try:
            event_data = message_context["value"]
            
            # Add to fusion window
            self.fusion_windows["seismic"].append({
                "timestamp": datetime.fromisoformat(event_data.get("timestamp")),
                "data": event_data,
                "received_at": message_context["received_at"]
            })
            
            # Add to Redis streams for immediate analysis
            await self.redis_streams.add_seismic_event(
                event_data, 
                event_id=event_data.get("event_id")
            )
            
            # Check for immediate alerts
            magnitude = event_data.get("magnitude", 0.0)
            if magnitude >= self.alert_thresholds["seismic_magnitude"]:
                await self._generate_alert({
                    "type": "seismic_alert",
                    "magnitude": magnitude,
                    "location": event_data.get("location"),
                    "timestamp": event_data.get("timestamp"),
                    "severity": "high" if magnitude >= 6.0 else "medium"
                })
            
            self.events_processed += 1
            
            logger.debug(
                "Processed seismic event",
                event_id=event_data.get("event_id"),
                magnitude=magnitude
            )
            
        except Exception as e:
            logger.error("Error handling seismic event", error=str(e))
    
    async def _handle_thermal_measurement(self, message_context: Dict[str, Any]):
        """Handle thermal measurement message"""
        try:
            measurement_data = message_context["value"]
            
            # Add to fusion window
            self.fusion_windows["thermal"].append({
                "timestamp": datetime.fromisoformat(measurement_data.get("timestamp")),
                "data": measurement_data,
                "received_at": message_context["received_at"]
            })
            
            # Add to Redis streams
            await self.redis_streams.add_thermal_measurement(measurement_data)
            
            # Check for thermal anomalies
            temperature = measurement_data.get("temperature", 0.0)
            baseline_temp = 300.0  # 27°C in Kelvin
            
            if temperature > baseline_temp + self.alert_thresholds["thermal_anomaly"]:
                await self._generate_alert({
                    "type": "thermal_anomaly",
                    "temperature": temperature,
                    "baseline": baseline_temp,
                    "anomaly_magnitude": temperature - baseline_temp,
                    "location": measurement_data.get("location"),
                    "timestamp": measurement_data.get("timestamp"),
                    "severity": "high" if temperature > baseline_temp + 100 else "medium"
                })
            
            self.events_processed += 1
            
            logger.debug(
                "Processed thermal measurement",
                temperature=temperature,
                location=measurement_data.get("location")
            )
            
        except Exception as e:
            logger.error("Error handling thermal measurement", error=str(e))
    
    async def _handle_strain_measurement(self, message_context: Dict[str, Any]):
        """Handle strain measurement message"""
        try:
            measurement_data = message_context["value"]
            
            # Add to fusion window
            self.fusion_windows["strain"].append({
                "timestamp": datetime.fromisoformat(measurement_data.get("timestamp")),
                "data": measurement_data,
                "received_at": message_context["received_at"]
            })
            
            # Add to Redis streams
            await self.redis_streams.add_strain_measurement(measurement_data)
            
            # Check for high strain rates
            strain_rate = abs(measurement_data.get("strain_rate", 0.0))
            
            if strain_rate > self.alert_thresholds["strain_rate"]:
                await self._generate_alert({
                    "type": "high_strain_rate",
                    "strain_rate": strain_rate,
                    "channel": measurement_data.get("channel_index"),
                    "location": measurement_data.get("location"),
                    "timestamp": measurement_data.get("timestamp"),
                    "severity": "high" if strain_rate > self.alert_thresholds["strain_rate"] * 10 else "medium"
                })\n            \n            self.events_processed += 1\n            \n            logger.debug(\n                \"Processed strain measurement\",\n                strain_rate=strain_rate,\n                channel=measurement_data.get(\"channel_index\")\n            )\n            \n        except Exception as e:\n            logger.error(\"Error handling strain measurement\", error=str(e))\n    \n    async def _handle_processing_batch(self, message_context: Dict[str, Any]):\n        \"\"\"Handle processing batch message\"\"\"\n        try:\n            batch_data = message_context[\"value\"]\n            \n            # Extract quality metrics\n            quality_metrics = batch_data.get(\"quality_metrics\", {})\n            \n            # Add quality metrics to Redis streams\n            await self.redis_streams.add_quality_metrics({\n                \"batch_id\": batch_data.get(\"batch_id\"),\n                \"data_type\": batch_data.get(\"data_type\"),\n                \"source_connector\": batch_data.get(\"source_connector\"),\n                \"metrics\": quality_metrics,\n                \"timestamp\": batch_data.get(\"timestamp\")\n            })\n            \n            logger.debug(\n                \"Processed batch metadata\",\n                batch_id=batch_data.get(\"batch_id\"),\n                data_type=batch_data.get(\"data_type\")\n            )\n            \n        except Exception as e:\n            logger.error(\"Error handling processing batch\", error=str(e))\n    \n    async def _correlation_analysis_task(self):\n        \"\"\"Background task for correlation analysis\"\"\"\n        while self.is_processing:\n            try:\n                await self._analyze_correlations()\n                await asyncio.sleep(30)  # Run every 30 seconds\n            except Exception as e:\n                logger.error(\"Error in correlation analysis\", error=str(e))\n                await asyncio.sleep(60)  # Wait longer on error\n    \n    async def _analyze_correlations(self):\n        \"\"\"Analyze correlations between different data types\"\"\"\n        try:\n            current_time = datetime.utcnow()\n            window_start = current_time - self.correlation_window\n            \n            # Get recent data from each type\n            recent_seismic = [\n                item for item in self.fusion_windows[\"seismic\"]\n                if item[\"timestamp\"] >= window_start\n            ]\n            \n            recent_thermal = [\n                item for item in self.fusion_windows[\"thermal\"]\n                if item[\"timestamp\"] >= window_start\n            ]\n            \n            recent_strain = [\n                item for item in self.fusion_windows[\"strain\"]\n                if item[\"timestamp\"] >= window_start\n            ]\n            \n            # Analyze seismic-thermal correlations\n            if len(recent_seismic) > 0 and len(recent_thermal) > 0:\n                correlation = await self._calculate_spatiotemporal_correlation(\n                    recent_seismic, recent_thermal, \"seismic-thermal\"\n                )\n                \n                if correlation > self.alert_thresholds[\"correlation_strength\"]:\n                    await self._generate_correlation_alert(\n                        \"seismic-thermal\", correlation, recent_seismic, recent_thermal\n                    )\n            \n            # Analyze seismic-strain correlations\n            if len(recent_seismic) > 0 and len(recent_strain) > 0:\n                correlation = await self._calculate_spatiotemporal_correlation(\n                    recent_seismic, recent_strain, \"seismic-strain\"\n                )\n                \n                if correlation > self.alert_thresholds[\"correlation_strength\"]:\n                    await self._generate_correlation_alert(\n                        \"seismic-strain\", correlation, recent_seismic, recent_strain\n                    )\n            \n            # Analyze thermal-strain correlations\n            if len(recent_thermal) > 0 and len(recent_strain) > 0:\n                correlation = await self._calculate_spatiotemporal_correlation(\n                    recent_thermal, recent_strain, \"thermal-strain\"\n                )\n                \n                if correlation > self.alert_thresholds[\"correlation_strength\"]:\n                    await self._generate_correlation_alert(\n                        \"thermal-strain\", correlation, recent_thermal, recent_strain\n                    )\n            \n            logger.debug(\n                \"Correlation analysis completed\",\n                seismic_count=len(recent_seismic),\n                thermal_count=len(recent_thermal),\n                strain_count=len(recent_strain)\n            )\n            \n        except Exception as e:\n            logger.error(\"Error in correlation analysis\", error=str(e))\n    \n    async def _calculate_spatiotemporal_correlation(\n        self, \n        data1: List[Dict[str, Any]], \n        data2: List[Dict[str, Any]],\n        correlation_type: str\n    ) -> float:\n        \"\"\"Calculate spatiotemporal correlation between two datasets\"\"\"\n        try:\n            if len(data1) < 2 or len(data2) < 2:\n                return 0.0\n            \n            # Simple correlation based on temporal proximity and spatial clustering\n            correlation_score = 0.0\n            matches = 0\n            \n            for item1 in data1:\n                timestamp1 = item1[\"timestamp\"]\n                location1 = item1[\"data\"].get(\"location\", {})\n                lat1 = location1.get(\"latitude\", 0.0)\n                lon1 = location1.get(\"longitude\", 0.0)\n                \n                for item2 in data2:\n                    timestamp2 = item2[\"timestamp\"]\n                    location2 = item2[\"data\"].get(\"location\", {})\n                    lat2 = location2.get(\"latitude\", 0.0)\n                    lon2 = location2.get(\"longitude\", 0.0)\n                    \n                    # Temporal correlation (within 5 minutes)\n                    time_diff = abs((timestamp1 - timestamp2).total_seconds())\n                    if time_diff <= 300:  # 5 minutes\n                        temporal_score = max(0, 1 - time_diff / 300)\n                        \n                        # Spatial correlation (within 50 km)\n                        spatial_distance = self._calculate_distance(lat1, lon1, lat2, lon2)\n                        if spatial_distance <= 50:  # 50 km\n                            spatial_score = max(0, 1 - spatial_distance / 50)\n                            \n                            # Combined score\n                            combined_score = (temporal_score + spatial_score) / 2\n                            correlation_score += combined_score\n                            matches += 1\n            \n            # Normalize by number of potential matches\n            if matches > 0:\n                correlation_score = correlation_score / matches\n            \n            logger.debug(\n                \"Calculated correlation\",\n                type=correlation_type,\n                score=correlation_score,\n                matches=matches\n            )\n            \n            return correlation_score\n            \n        except Exception as e:\n            logger.error(\"Error calculating correlation\", error=str(e))\n            return 0.0\n    \n    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:\n        \"\"\"Calculate distance between two points in kilometers\"\"\"\n        # Simple approximation for small distances\n        lat_diff = lat1 - lat2\n        lon_diff = lon1 - lon2\n        return np.sqrt(lat_diff**2 + lon_diff**2) * 111.0  # Rough km conversion\n    \n    async def _generate_correlation_alert(\n        self, \n        correlation_type: str, \n        correlation_strength: float,\n        data1: List[Dict[str, Any]], \n        data2: List[Dict[str, Any]]\n    ):\n        \"\"\"Generate alert for detected correlation\"\"\"\n        try:\n            alert = {\n                \"type\": \"correlation_detected\",\n                \"correlation_type\": correlation_type,\n                \"correlation_strength\": correlation_strength,\n                \"data1_count\": len(data1),\n                \"data2_count\": len(data2),\n                \"timestamp\": datetime.utcnow().isoformat(),\n                \"severity\": \"high\" if correlation_strength > 0.9 else \"medium\",\n                \"window_minutes\": self.correlation_window.total_seconds() / 60\n            }\n            \n            await self._generate_alert(alert)\n            self.correlations_detected += 1\n            \n            logger.info(\n                \"Correlation detected\",\n                type=correlation_type,\n                strength=correlation_strength\n            )\n            \n        except Exception as e:\n            logger.error(\"Error generating correlation alert\", error=str(e))\n    \n    async def _alert_monitoring_task(self):\n        \"\"\"Background task for alert monitoring\"\"\"\n        while self.is_processing:\n            try:\n                await self._monitor_alert_conditions()\n                await asyncio.sleep(10)  # Check every 10 seconds\n            except Exception as e:\n                logger.error(\"Error in alert monitoring\", error=str(e))\n                await asyncio.sleep(30)  # Wait longer on error\n    \n    async def _monitor_alert_conditions(self):\n        \"\"\"Monitor for alert conditions\"\"\"\n        try:\n            current_time = datetime.utcnow()\n            alert_window_start = current_time - self.alert_threshold_window\n            \n            # Check for rapid sequence of events\n            recent_seismic = [\n                item for item in self.fusion_windows[\"seismic\"]\n                if item[\"timestamp\"] >= alert_window_start\n            ]\n            \n            if len(recent_seismic) >= 5:  # 5+ events in 1 minute\n                await self._generate_alert({\n                    \"type\": \"seismic_swarm\",\n                    \"event_count\": len(recent_seismic),\n                    \"time_window_minutes\": self.alert_threshold_window.total_seconds() / 60,\n                    \"timestamp\": current_time.isoformat(),\n                    \"severity\": \"high\"\n                })\n            \n            # Check for sustained high strain rates\n            recent_strain = [\n                item for item in self.fusion_windows[\"strain\"]\n                if item[\"timestamp\"] >= alert_window_start\n            ]\n            \n            high_strain_count = sum(\n                1 for item in recent_strain\n                if abs(item[\"data\"].get(\"strain_rate\", 0.0)) > self.alert_thresholds[\"strain_rate\"]\n            )\n            \n            if high_strain_count >= 10:  # 10+ high strain measurements in 1 minute\n                await self._generate_alert({\n                    \"type\": \"sustained_high_strain\",\n                    \"high_strain_count\": high_strain_count,\n                    \"total_measurements\": len(recent_strain),\n                    \"time_window_minutes\": self.alert_threshold_window.total_seconds() / 60,\n                    \"timestamp\": current_time.isoformat(),\n                    \"severity\": \"medium\"\n                })\n            \n        except Exception as e:\n            logger.error(\"Error monitoring alert conditions\", error=str(e))\n    \n    async def _data_fusion_task(self):\n        \"\"\"Background task for data fusion\"\"\"\n        while self.is_processing:\n            try:\n                await self._perform_data_fusion()\n                await asyncio.sleep(60)  # Run every minute\n            except Exception as e:\n                logger.error(\"Error in data fusion\", error=str(e))\n                await asyncio.sleep(120)  # Wait longer on error\n    \n    async def _perform_data_fusion(self):\n        \"\"\"Perform multi-sensor data fusion\"\"\"\n        try:\n            current_time = datetime.utcnow()\n            fusion_window_start = current_time - timedelta(minutes=10)  # 10-minute fusion window\n            \n            # Get recent data from all sensors\n            recent_data = {}\n            for data_type, window in self.fusion_windows.items():\n                recent_data[data_type] = [\n                    item for item in window\n                    if item[\"timestamp\"] >= fusion_window_start\n                ]\n            \n            # Perform spatial clustering\n            clusters = await self._spatial_clustering(recent_data)\n            \n            # Analyze each cluster\n            for cluster_id, cluster_data in clusters.items():\n                fusion_result = await self._analyze_cluster(cluster_id, cluster_data)\n                \n                if fusion_result:\n                    # Add fusion result to Redis streams\n                    await self.redis_streams.add_real_time_analysis(\n                        fusion_result,\n                        analysis_id=f\"fusion_{cluster_id}_{int(current_time.timestamp())}\"\n                    )\n            \n            logger.debug(\n                \"Data fusion completed\",\n                clusters=len(clusters),\n                total_events=sum(len(data) for data in recent_data.values())\n            )\n            \n        except Exception as e:\n            logger.error(\"Error in data fusion\", error=str(e))\n    \n    async def _spatial_clustering(self, recent_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:\n        \"\"\"Perform spatial clustering of multi-sensor data\"\"\"\n        try:\n            clusters = defaultdict(lambda: defaultdict(list))\n            cluster_id = 0\n            \n            # Simple spatial clustering based on geographic proximity\n            all_locations = []\n            \n            # Collect all locations with metadata\n            for data_type, data_list in recent_data.items():\n                for item in data_list:\n                    location = item[\"data\"].get(\"location\", {})\n                    lat = location.get(\"latitude\", 0.0)\n                    lon = location.get(\"longitude\", 0.0)\n                    \n                    all_locations.append({\n                        \"lat\": lat,\n                        \"lon\": lon,\n                        \"data_type\": data_type,\n                        \"item\": item,\n                        \"cluster_id\": None\n                    })\n            \n            # Simple clustering algorithm (can be improved with proper clustering)\n            cluster_radius = 25.0  # 25 km radius\n            \n            for i, location in enumerate(all_locations):\n                if location[\"cluster_id\"] is not None:\n                    continue\n                \n                # Start new cluster\n                current_cluster_id = cluster_id\n                cluster_id += 1\n                \n                location[\"cluster_id\"] = current_cluster_id\n                clusters[current_cluster_id][location[\"data_type\"]].append(location[\"item\"])\n                \n                # Find nearby locations\n                for j, other_location in enumerate(all_locations[i+1:], i+1):\n                    if other_location[\"cluster_id\"] is not None:\n                        continue\n                    \n                    distance = self._calculate_distance(\n                        location[\"lat\"], location[\"lon\"],\n                        other_location[\"lat\"], other_location[\"lon\"]\n                    )\n                    \n                    if distance <= cluster_radius:\n                        other_location[\"cluster_id\"] = current_cluster_id\n                        clusters[current_cluster_id][other_location[\"data_type\"]].append(other_location[\"item\"])\n            \n            return dict(clusters)\n            \n        except Exception as e:\n            logger.error(\"Error in spatial clustering\", error=str(e))\n            return {}\n    \n    async def _analyze_cluster(self, cluster_id: str, cluster_data: Dict[str, List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:\n        \"\"\"Analyze a spatial cluster of multi-sensor data\"\"\"\n        try:\n            # Calculate cluster statistics\n            total_events = sum(len(data_list) for data_list in cluster_data.values())\n            \n            if total_events < 2:\n                return None  # Not enough data for meaningful analysis\n            \n            # Calculate center of cluster\n            all_lats = []\n            all_lons = []\n            \n            for data_list in cluster_data.values():\n                for item in data_list:\n                    location = item[\"data\"].get(\"location\", {})\n                    all_lats.append(location.get(\"latitude\", 0.0))\n                    all_lons.append(location.get(\"longitude\", 0.0))\n            \n            center_lat = np.mean(all_lats) if all_lats else 0.0\n            center_lon = np.mean(all_lons) if all_lons else 0.0\n            \n            # Analyze data types present\n            data_types_present = list(cluster_data.keys())\n            \n            # Calculate significance score\n            significance_score = 0.0\n            \n            # Higher score for multiple data types\n            significance_score += len(data_types_present) * 0.2\n            \n            # Higher score for more events\n            significance_score += min(total_events * 0.1, 1.0)\n            \n            # Check for high-magnitude events\n            if \"seismic\" in cluster_data:\n                max_magnitude = max(\n                    item[\"data\"].get(\"magnitude\", 0.0)\n                    for item in cluster_data[\"seismic\"]\n                )\n                if max_magnitude >= 4.0:\n                    significance_score += 0.5\n            \n            # Check for thermal anomalies\n            if \"thermal\" in cluster_data:\n                max_temp = max(\n                    item[\"data\"].get(\"temperature\", 0.0)\n                    for item in cluster_data[\"thermal\"]\n                )\n                if max_temp > 350.0:  # 77°C\n                    significance_score += 0.3\n            \n            fusion_result = {\n                \"cluster_id\": cluster_id,\n                \"center_location\": {\n                    \"latitude\": center_lat,\n                    \"longitude\": center_lon\n                },\n                \"data_types_present\": data_types_present,\n                \"total_events\": total_events,\n                \"event_counts\": {dt: len(data_list) for dt, data_list in cluster_data.items()},\n                \"significance_score\": significance_score,\n                \"analysis_timestamp\": datetime.utcnow().isoformat(),\n                \"time_span_minutes\": 10  # Analysis window\n            }\n            \n            # Add type-specific analysis\n            if \"seismic\" in cluster_data:\n                seismic_events = cluster_data[\"seismic\"]\n                fusion_result[\"seismic_analysis\"] = {\n                    \"max_magnitude\": max(item[\"data\"].get(\"magnitude\", 0.0) for item in seismic_events),\n                    \"avg_magnitude\": np.mean([item[\"data\"].get(\"magnitude\", 0.0) for item in seismic_events]),\n                    \"depth_range\": [\n                        min(item[\"data\"].get(\"depth\", 0.0) for item in seismic_events),\n                        max(item[\"data\"].get(\"depth\", 0.0) for item in seismic_events)\n                    ]\n                }\n            \n            if \"thermal\" in cluster_data:\n                thermal_measurements = cluster_data[\"thermal\"]\n                fusion_result[\"thermal_analysis\"] = {\n                    \"max_temperature\": max(item[\"data\"].get(\"temperature\", 0.0) for item in thermal_measurements),\n                    \"avg_temperature\": np.mean([item[\"data\"].get(\"temperature\", 0.0) for item in thermal_measurements]),\n                    \"temperature_std\": np.std([item[\"data\"].get(\"temperature\", 0.0) for item in thermal_measurements])\n                }\n            \n            if \"strain\" in cluster_data:\n                strain_measurements = cluster_data[\"strain\"]\n                strain_rates = [abs(item[\"data\"].get(\"strain_rate\", 0.0)) for item in strain_measurements]\n                fusion_result[\"strain_analysis\"] = {\n                    \"max_strain_rate\": max(strain_rates),\n                    \"avg_strain_rate\": np.mean(strain_rates),\n                    \"strain_rate_std\": np.std(strain_rates)\n                }\n            \n            return fusion_result\n            \n        except Exception as e:\n            logger.error(\"Error analyzing cluster\", cluster_id=cluster_id, error=str(e))\n            return None\n    \n    async def _generate_alert(self, alert_data: Dict[str, Any]):\n        \"\"\"Generate and distribute alert\"\"\"\n        try:\n            # Add alert metadata\n            alert = {\n                **alert_data,\n                \"alert_id\": f\"alert_{int(datetime.utcnow().timestamp())}_{alert_data.get('type', 'unknown')}\",\n                \"generated_at\": datetime.utcnow().isoformat(),\n                \"source\": \"stream_processor\"\n            }\n            \n            # Send to Kafka\n            await self.kafka_producer.send_alert(alert)\n            \n            # Add to Redis streams\n            await self.redis_streams.add_alert(alert)\n            \n            self.alerts_generated += 1\n            \n            logger.info(\n                \"Alert generated\",\n                alert_type=alert_data.get(\"type\"),\n                severity=alert_data.get(\"severity\"),\n                alert_id=alert[\"alert_id\"]\n            )\n            \n        except Exception as e:\n            logger.error(\"Error generating alert\", error=str(e))\n    \n    async def get_statistics(self) -> Dict[str, Any]:\n        \"\"\"Get stream processor statistics\"\"\"\n        stats = {\n            \"is_initialized\": self.is_initialized,\n            \"is_processing\": self.is_processing,\n            \"events_processed\": self.events_processed,\n            \"alerts_generated\": self.alerts_generated,\n            \"correlations_detected\": self.correlations_detected,\n            \"fusion_window_sizes\": {\n                data_type: len(window)\n                for data_type, window in self.fusion_windows.items()\n            },\n            \"processing_tasks_running\": len([\n                task for task in self.processing_tasks\n                if not task.done()\n            ])\n        }\n        \n        # Add component statistics\n        if self.is_initialized:\n            stats[\"kafka_producer\"] = await self.kafka_producer.get_statistics()\n            stats[\"kafka_consumer\"] = await self.kafka_consumer.get_statistics()\n            stats[\"redis_streams\"] = await self.redis_streams.get_statistics()\n        \n        return stats\n    \n    async def stop_processing(self):\n        \"\"\"Stop stream processing\"\"\"\n        self.is_processing = False\n        \n        # Cancel processing tasks\n        for task in self.processing_tasks:\n            if not task.done():\n                task.cancel()\n        \n        # Wait for tasks to complete\n        if self.processing_tasks:\n            await asyncio.gather(*self.processing_tasks, return_exceptions=True)\n        \n        logger.info(\"Stream processing stopped\")\n    \n    async def cleanup(self):\n        \"\"\"Cleanup stream processor\"\"\"\n        await self.stop_processing()\n        \n        # Cleanup components\n        if self.kafka_producer:\n            await self.kafka_producer.cleanup()\n        if self.kafka_consumer:\n            await self.kafka_consumer.cleanup()\n        if self.redis_streams:\n            await self.redis_streams.cleanup()\n        \n        self.is_initialized = False\n        logger.info(\"Stream processor cleanup completed\")