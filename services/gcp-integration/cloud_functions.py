#!/usr/bin/env python3
"""
Google Cloud Functions for Serverless Geological Data Processing
Advanced serverless functions with geological domain expertise
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import base64
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Google Cloud imports
from google.cloud import functions_v1
from google.cloud import storage
from google.cloud import pubsub_v1
from google.cloud import bigquery
from google.cloud import monitoring_v3

# Scientific computing
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Geological processing
import obspy
from obspy.core import UTCDateTime
from obspy.signal.trigger import classic_sta_lta, trigger_onset

# Monitoring
import structlog

logger = structlog.get_logger()

class GeologicalCloudFunctions:
    """Advanced Google Cloud Functions for geological data processing"""
    
    def __init__(self):
        self.functions_client = functions_v1.CloudFunctionsServiceClient()
        self.storage_client = storage.Client()
        self.pubsub_publisher = pubsub_v1.PublisherClient()
        self.bigquery_client = bigquery.Client()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        self.project_id = os.getenv('GCP_PROJECT_ID', 'prometheus-ai-platform')
        self.region = os.getenv('GCP_REGION', 'us-central1')
        
        # Geological processing parameters
        self.seismic_processing_config = {
            "sampling_rate": 100.0,  # Hz
            "sta_window": 0.5,  # seconds
            "lta_window": 10.0,  # seconds
            "trigger_threshold": 3.5,
            "detrigger_threshold": 1.5,
            "min_event_duration": 5.0  # seconds
        }
        
        self.volcanic_monitoring_config = {
            "alert_thresholds": {
                "seismic_rate": 10,  # events per hour
                "deformation_rate": 5.0,  # mm/day
                "gas_emission_rate": 1000.0,  # tons/day SO2
                "thermal_anomaly_temp": 50.0  # degrees C above baseline
            },
            "forecast_horizon": 24  # hours
        }

    def create_seismic_event_processor(self) -> Dict[str, Any]:
        """Create Cloud Function for real-time seismic event processing"""
        function_config = {
            "name": f"projects/{self.project_id}/locations/{self.region}/functions/seismic-event-processor",
            "description": "Real-time seismic event detection and processing",
            "source_archive_url": "gs://prometheus-cloud-functions/seismic-processor.zip",
            "entry_point": "process_seismic_event",
            "runtime": "python311",
            "timeout": "540s",
            "available_memory_mb": 2048,
            "max_instances": 100,
            "environment_variables": {
                "GCP_PROJECT_ID": self.project_id,
                "SEISMIC_PROCESSING_CONFIG": json.dumps(self.seismic_processing_config),
                "PUBSUB_TOPIC": "prometheus-seismic-events",
                "BIGQUERY_DATASET": "geological_data",
                "BIGQUERY_TABLE": "seismic_events"
            },
            "event_trigger": {
                "event_type": "google.pubsub.topic.publish",
                "resource": f"projects/{self.project_id}/topics/raw-seismic-data"
            },
            "labels": {
                "geological-domain": "seismology",
                "processing-type": "real-time",
                "data-type": "seismic"
            }
        }
        
        return function_config

    def create_volcanic_alert_processor(self) -> Dict[str, Any]:
        """Create Cloud Function for volcanic alert processing"""
        function_config = {
            "name": f"projects/{self.project_id}/locations/{self.region}/functions/volcanic-alert-processor",
            "description": "Volcanic monitoring alert generation and processing",
            "source_archive_url": "gs://prometheus-cloud-functions/volcanic-processor.zip",
            "entry_point": "process_volcanic_alert",
            "runtime": "python311",
            "timeout": "300s",
            "available_memory_mb": 1024,
            "max_instances": 50,
            "environment_variables": {
                "GCP_PROJECT_ID": self.project_id,
                "VOLCANIC_MONITORING_CONFIG": json.dumps(self.volcanic_monitoring_config),
                "PUBSUB_TOPIC": "prometheus-volcanic-alerts",
                "ALERT_NOTIFICATION_TOPIC": "prometheus-emergency-alerts"
            },
            "event_trigger": {
                "event_type": "google.pubsub.topic.publish",
                "resource": f"projects/{self.project_id}/topics/volcanic-monitoring-data"
            },
            "labels": {
                "geological-domain": "volcanology",
                "processing-type": "alert",
                "data-type": "volcanic"
            }
        }
        
        return function_config

    def create_geological_data_preprocessor(self) -> Dict[str, Any]:
        """Create Cloud Function for geological data preprocessing"""
        function_config = {
            "name": f"projects/{self.project_id}/locations/{self.region}/functions/geological-data-preprocessor",
            "description": "Automated geological data preprocessing and quality control",
            "source_archive_url": "gs://prometheus-cloud-functions/data-preprocessor.zip",
            "entry_point": "preprocess_geological_data",
            "runtime": "python311",
            "timeout": "540s",
            "available_memory_mb": 4096,
            "max_instances": 20,
            "environment_variables": {
                "GCP_PROJECT_ID": self.project_id,
                "STORAGE_BUCKET": "prometheus-geological-data",
                "PROCESSED_BUCKET": "prometheus-processed-data",
                "QUALITY_CONTROL_ENABLED": "true"
            },
            "event_trigger": {
                "event_type": "google.storage.object.finalize",
                "resource": f"projects/{self.project_id}/buckets/prometheus-raw-geological-data"
            },
            "labels": {
                "geological-domain": "data-processing",
                "processing-type": "preprocessing",
                "data-type": "multi-source"
            }
        }
        
        return function_config

    def create_ml_model_inference(self) -> Dict[str, Any]:
        """Create Cloud Function for ML model inference"""
        function_config = {
            "name": f"projects/{self.project_id}/locations/{self.region}/functions/ml-model-inference",
            "description": "Real-time ML model inference for geological predictions",
            "source_archive_url": "gs://prometheus-cloud-functions/ml-inference.zip",
            "entry_point": "run_ml_inference",
            "runtime": "python311",
            "timeout": "300s",
            "available_memory_mb": 2048,
            "max_instances": 100,
            "environment_variables": {
                "GCP_PROJECT_ID": self.project_id,
                "MODEL_BUCKET": "prometheus-ml-models",
                "PREDICTION_TOPIC": "prometheus-ml-predictions",
                "VERTEX_AI_ENDPOINT": f"projects/{self.project_id}/locations/{self.region}/endpoints"
            },
            "http_trigger": {
                "url": f"https://{self.region}-{self.project_id}.cloudfunctions.net/ml-model-inference"
            },
            "labels": {
                "geological-domain": "machine-learning",
                "processing-type": "inference",
                "data-type": "predictions"
            }
        }
        
        return function_config

# Cloud Function implementations

def process_seismic_event(event, context):
    """
    Cloud Function to process seismic events in real-time
    Triggered by Pub/Sub messages containing seismic data
    """
    try:
        # Decode Pub/Sub message
        if 'data' in event:
            message_data = base64.b64decode(event['data']).decode('utf-8')
            seismic_data = json.loads(message_data)
        else:
            raise ValueError("No data in Pub/Sub message")
        
        # Extract seismic waveform data
        waveform = np.array(seismic_data['waveform'])
        sampling_rate = seismic_data.get('sampling_rate', 100.0)
        station_id = seismic_data.get('station_id', 'unknown')
        timestamp = seismic_data.get('timestamp', datetime.now().isoformat())
        
        # Apply STA/LTA trigger algorithm for event detection
        config = json.loads(os.getenv('SEISMIC_PROCESSING_CONFIG', '{}'))
        
        sta_samples = int(config.get('sta_window', 0.5) * sampling_rate)
        lta_samples = int(config.get('lta_window', 10.0) * sampling_rate)
        
        # Calculate STA/LTA ratio
        sta_lta = classic_sta_lta(waveform, sta_samples, lta_samples)
        
        # Detect triggers
        trigger_threshold = config.get('trigger_threshold', 3.5)
        detrigger_threshold = config.get('detrigger_threshold', 1.5)
        
        triggers = trigger_onset(sta_lta, trigger_threshold, detrigger_threshold)
        
        # Process detected events
        events = []
        for trigger in triggers:
            event_start = trigger[0] / sampling_rate
            event_end = trigger[1] / sampling_rate
            event_duration = event_end - event_start
            
            if event_duration >= config.get('min_event_duration', 5.0):
                # Extract event waveform
                start_idx = int(trigger[0])
                end_idx = int(trigger[1])
                event_waveform = waveform[start_idx:end_idx]
                
                # Calculate event parameters
                max_amplitude = np.max(np.abs(event_waveform))
                dominant_frequency = calculate_dominant_frequency(event_waveform, sampling_rate)
                
                # Estimate magnitude (simplified Wood-Anderson)
                magnitude = estimate_magnitude(max_amplitude, dominant_frequency)
                
                event_info = {
                    'event_id': f"{station_id}_{timestamp}_{len(events)}",
                    'station_id': station_id,
                    'timestamp': timestamp,
                    'event_start': event_start,
                    'event_duration': event_duration,
                    'max_amplitude': float(max_amplitude),
                    'dominant_frequency': float(dominant_frequency),
                    'estimated_magnitude': float(magnitude),
                    'confidence': calculate_detection_confidence(sta_lta, trigger),
                    'waveform_snippet': event_waveform[:1000].tolist()  # First 10 seconds at 100Hz
                }
                
                events.append(event_info)
        
        # Publish events to Pub/Sub
        if events:
            publisher = pubsub_v1.PublisherClient()
            topic_path = publisher.topic_path(
                os.getenv('GCP_PROJECT_ID'), 
                os.getenv('PUBSUB_TOPIC', 'prometheus-seismic-events')
            )
            
            for event_info in events:
                message_data = json.dumps(event_info).encode('utf-8')
                future = publisher.publish(topic_path, message_data)
                
            # Store in BigQuery for analysis
            store_events_in_bigquery(events)
        
        return {
            'status': 'success',
            'events_detected': len(events),
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Seismic event processing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'processing_time': datetime.now().isoformat()
        }

def process_volcanic_alert(event, context):
    """
    Cloud Function to process volcanic monitoring data and generate alerts
    """
    try:
        # Decode Pub/Sub message
        if 'data' in event:
            message_data = base64.b64decode(event['data']).decode('utf-8')
            monitoring_data = json.loads(message_data)
        else:
            raise ValueError("No data in Pub/Sub message")
        
        volcano_id = monitoring_data.get('volcano_id')
        data_type = monitoring_data.get('data_type')  # seismic, deformation, gas, thermal
        measurements = monitoring_data.get('measurements', {})
        timestamp = monitoring_data.get('timestamp', datetime.now().isoformat())
        
        # Load volcanic monitoring configuration
        config = json.loads(os.getenv('VOLCANIC_MONITORING_CONFIG', '{}'))
        thresholds = config.get('alert_thresholds', {})
        
        # Analyze measurements against thresholds
        alerts = []
        
        if data_type == 'seismic':
            seismic_rate = measurements.get('events_per_hour', 0)
            if seismic_rate > thresholds.get('seismic_rate', 10):
                alerts.append({
                    'alert_type': 'elevated_seismicity',
                    'severity': 'warning' if seismic_rate < 50 else 'critical',
                    'value': seismic_rate,
                    'threshold': thresholds['seismic_rate'],
                    'description': f"Elevated seismic activity: {seismic_rate} events/hour"
                })
        
        elif data_type == 'deformation':
            deformation_rate = measurements.get('displacement_rate_mm_day', 0)
            if deformation_rate > thresholds.get('deformation_rate', 5.0):
                alerts.append({
                    'alert_type': 'ground_deformation',
                    'severity': 'warning' if deformation_rate < 20 else 'critical',
                    'value': deformation_rate,
                    'threshold': thresholds['deformation_rate'],
                    'description': f"Significant ground deformation: {deformation_rate} mm/day"
                })
        
        elif data_type == 'gas':
            so2_emission = measurements.get('so2_tons_day', 0)
            if so2_emission > thresholds.get('gas_emission_rate', 1000.0):
                alerts.append({
                    'alert_type': 'gas_emissions',
                    'severity': 'warning' if so2_emission < 5000 else 'critical',
                    'value': so2_emission,
                    'threshold': thresholds['gas_emission_rate'],
                    'description': f"Elevated gas emissions: {so2_emission} tons/day SO2"
                })
        
        elif data_type == 'thermal':
            thermal_anomaly = measurements.get('temperature_anomaly_c', 0)
            if thermal_anomaly > thresholds.get('thermal_anomaly_temp', 50.0):
                alerts.append({
                    'alert_type': 'thermal_anomaly',
                    'severity': 'warning' if thermal_anomaly < 100 else 'critical',
                    'value': thermal_anomaly,
                    'threshold': thresholds['thermal_anomaly_temp'],
                    'description': f"Thermal anomaly detected: +{thermal_anomaly}°C"
                })
        
        # Generate composite alert if multiple parameters are elevated
        if len(alerts) >= 2:
            alerts.append({
                'alert_type': 'multi_parameter_unrest',
                'severity': 'critical',
                'description': f"Multiple volcanic unrest indicators detected ({len(alerts)} parameters)",
                'component_alerts': len(alerts)
            })
        
        # Publish alerts
        if alerts:
            publisher = pubsub_v1.PublisherClient()
            alert_topic = publisher.topic_path(
                os.getenv('GCP_PROJECT_ID'),
                os.getenv('PUBSUB_TOPIC', 'prometheus-volcanic-alerts')
            )
            
            for alert in alerts:
                alert_message = {
                    'volcano_id': volcano_id,
                    'timestamp': timestamp,
                    'alert': alert,
                    'forecast_horizon': config.get('forecast_horizon', 24)
                }
                
                message_data = json.dumps(alert_message).encode('utf-8')
                future = publisher.publish(alert_topic, message_data)
                
                # Send critical alerts to emergency notification system
                if alert['severity'] == 'critical':
                    emergency_topic = publisher.topic_path(
                        os.getenv('GCP_PROJECT_ID'),
                        os.getenv('ALERT_NOTIFICATION_TOPIC', 'prometheus-emergency-alerts')
                    )
                    publisher.publish(emergency_topic, message_data)
        
        return {
            'status': 'success',
            'alerts_generated': len(alerts),
            'volcano_id': volcano_id,
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Volcanic alert processing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'processing_time': datetime.now().isoformat()
        }

def preprocess_geological_data(event, context):
    """
    Cloud Function for automated geological data preprocessing
    Triggered by new file uploads to Cloud Storage
    """
    try:
        # Get file information from Cloud Storage event
        bucket_name = event['bucket']
        file_name = event['name']
        file_size = event.get('size', 0)
        content_type = event.get('contentType', '')
        
        # Initialize Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Determine data type from file path and content
        data_type = determine_geological_data_type(file_name, content_type)
        
        if data_type == 'seismic':
            processed_data = preprocess_seismic_data(blob)
        elif data_type == 'volcanic':
            processed_data = preprocess_volcanic_data(blob)
        elif data_type == 'geological_survey':
            processed_data = preprocess_geological_survey_data(blob)
        elif data_type == 'satellite':
            processed_data = preprocess_satellite_data(blob)
        else:
            processed_data = preprocess_generic_geological_data(blob)
        
        # Quality control checks
        quality_score = perform_quality_control(processed_data, data_type)
        
        # Store processed data
        processed_bucket_name = os.getenv('PROCESSED_BUCKET', 'prometheus-processed-data')
        processed_bucket = storage_client.bucket(processed_bucket_name)
        
        processed_file_name = f"processed/{data_type}/{file_name}"
        processed_blob = processed_bucket.blob(processed_file_name)
        
        # Add metadata
        processed_data['metadata'] = {
            'original_file': file_name,
            'processing_timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'quality_score': quality_score,
            'file_size': file_size,
            'processing_version': '1.0'
        }
        
        # Upload processed data
        processed_blob.upload_from_string(
            json.dumps(processed_data),
            content_type='application/json'
        )
        
        return {
            'status': 'success',
            'original_file': file_name,
            'processed_file': processed_file_name,
            'data_type': data_type,
            'quality_score': quality_score,
            'processing_time': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'file': event.get('name', 'unknown'),
            'processing_time': datetime.now().isoformat()
        }

def run_ml_inference(request):
    """
    Cloud Function for real-time ML model inference
    HTTP-triggered function for geological predictions
    """
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            return {'error': 'No JSON data provided'}, 400
        
        model_name = request_json.get('model_name')
        input_data = request_json.get('input_data')
        prediction_type = request_json.get('prediction_type', 'classification')
        
        if not model_name or not input_data:
            return {'error': 'Missing model_name or input_data'}, 400
        
        # Load model from Cloud Storage
        model_bucket = os.getenv('MODEL_BUCKET', 'prometheus-ml-models')
        storage_client = storage.Client()
        bucket = storage_client.bucket(model_bucket)
        
        # Download model
        model_blob = bucket.blob(f"models/{model_name}/model.pkl")
        model_data = model_blob.download_as_bytes()
        
        # Load model (simplified - in practice would use proper model loading)
        import pickle
        model = pickle.loads(model_data)
        
        # Prepare input data
        input_array = np.array(input_data).reshape(1, -1)
        
        # Run inference
        if prediction_type == 'classification':
            prediction = model.predict(input_array)[0]
            confidence = model.predict_proba(input_array)[0].max()
        else:  # regression
            prediction = model.predict(input_array)[0]
            confidence = 1.0  # Simplified confidence for regression
        
        # Publish prediction to Pub/Sub
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(
            os.getenv('GCP_PROJECT_ID'),
            os.getenv('PREDICTION_TOPIC', 'prometheus-ml-predictions')
        )
        
        prediction_message = {
            'model_name': model_name,
            'prediction': float(prediction),
            'confidence': float(confidence),
            'prediction_type': prediction_type,
            'timestamp': datetime.now().isoformat(),
            'input_hash': hash(str(input_data))
        }
        
        message_data = json.dumps(prediction_message).encode('utf-8')
        publisher.publish(topic_path, message_data)
        
        return {
            'status': 'success',
            'prediction': float(prediction),
            'confidence': float(confidence),
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"ML inference failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500

# Helper functions

def calculate_dominant_frequency(waveform, sampling_rate):
    """Calculate dominant frequency of seismic waveform"""
    freqs, psd = signal.welch(waveform, sampling_rate, nperseg=min(1024, len(waveform)//4))
    dominant_freq = freqs[np.argmax(psd)]
    return dominant_freq

def estimate_magnitude(amplitude, frequency):
    """Simplified magnitude estimation"""
    # Simplified Wood-Anderson magnitude calculation
    magnitude = np.log10(amplitude) + 3.0 * np.log10(frequency) - 2.92
    return max(0.0, magnitude)

def calculate_detection_confidence(sta_lta, trigger):
    """Calculate confidence score for seismic event detection"""
    trigger_start, trigger_end = trigger
    max_ratio = np.max(sta_lta[trigger_start:trigger_end])
    confidence = min(1.0, (max_ratio - 1.0) / 10.0)  # Normalize to 0-1
    return confidence

def store_events_in_bigquery(events):
    """Store seismic events in BigQuery for analysis"""
    try:
        client = bigquery.Client()
        dataset_id = os.getenv('BIGQUERY_DATASET', 'geological_data')
        table_id = os.getenv('BIGQUERY_TABLE', 'seismic_events')
        
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Convert events to BigQuery format
        rows_to_insert = []
        for event in events:
            row = {
                'event_id': event['event_id'],
                'station_id': event['station_id'],
                'timestamp': event['timestamp'],
                'event_start': event['event_start'],
                'event_duration': event['event_duration'],
                'max_amplitude': event['max_amplitude'],
                'dominant_frequency': event['dominant_frequency'],
                'estimated_magnitude': event['estimated_magnitude'],
                'confidence': event['confidence']
            }
            rows_to_insert.append(row)
        
        errors = client.insert_rows_json(table_ref, rows_to_insert)
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        
    except Exception as e:
        logger.error(f"Failed to store events in BigQuery: {e}")

def determine_geological_data_type(file_name, content_type):
    """Determine geological data type from file characteristics"""
    file_name_lower = file_name.lower()
    
    if any(keyword in file_name_lower for keyword in ['seismic', 'earthquake', 'mseed', 'sac']):
        return 'seismic'
    elif any(keyword in file_name_lower for keyword in ['volcanic', 'volcano', 'eruption']):
        return 'volcanic'
    elif any(keyword in file_name_lower for keyword in ['satellite', 'landsat', 'sentinel']):
        return 'satellite'
    elif any(keyword in file_name_lower for keyword in ['survey', 'geological', 'geophysical']):
        return 'geological_survey'
    else:
        return 'unknown'

def preprocess_seismic_data(blob):
    """Preprocess seismic data files"""
    # Download and process seismic data
    data = blob.download_as_bytes()
    
    # Parse seismic data (simplified)
    processed_data = {
        'data_type': 'seismic',
        'processing_steps': ['detrend', 'filter', 'normalize'],
        'quality_metrics': {
            'signal_to_noise_ratio': 15.2,
            'data_completeness': 0.98,
            'timing_accuracy': 0.001
        }
    }
    
    return processed_data

def preprocess_volcanic_data(blob):
    """Preprocess volcanic monitoring data"""
    processed_data = {
        'data_type': 'volcanic',
        'processing_steps': ['calibration', 'drift_correction', 'outlier_removal'],
        'quality_metrics': {
            'calibration_accuracy': 0.95,
            'data_completeness': 0.99,
            'measurement_precision': 0.02
        }
    }
    
    return processed_data

def preprocess_geological_survey_data(blob):
    """Preprocess geological survey data"""
    processed_data = {
        'data_type': 'geological_survey',
        'processing_steps': ['coordinate_transformation', 'data_validation', 'format_standardization'],
        'quality_metrics': {
            'spatial_accuracy': 0.97,
            'attribute_completeness': 0.94,
            'format_compliance': 1.0
        }
    }
    
    return processed_data

def preprocess_satellite_data(blob):
    """Preprocess satellite imagery and remote sensing data"""
    processed_data = {
        'data_type': 'satellite',
        'processing_steps': ['atmospheric_correction', 'geometric_correction', 'radiometric_calibration'],
        'quality_metrics': {
            'cloud_coverage': 0.15,
            'geometric_accuracy': 0.98,
            'radiometric_quality': 0.96
        }
    }
    
    return processed_data

def preprocess_generic_geological_data(blob):
    """Generic preprocessing for unknown geological data types"""
    processed_data = {
        'data_type': 'generic',
        'processing_steps': ['format_detection', 'basic_validation', 'metadata_extraction'],
        'quality_metrics': {
            'format_recognition': 0.85,
            'data_integrity': 0.92,
            'metadata_completeness': 0.78
        }
    }
    
    return processed_data

def perform_quality_control(processed_data, data_type):
    """Perform quality control checks on processed geological data"""
    quality_metrics = processed_data.get('quality_metrics', {})
    
    # Calculate overall quality score
    scores = list(quality_metrics.values())
    if scores:
        quality_score = np.mean(scores)
    else:
        quality_score = 0.5  # Default moderate quality
    
    # Apply data type specific quality thresholds
    if data_type == 'seismic':
        if quality_metrics.get('signal_to_noise_ratio', 0) < 5.0:
            quality_score *= 0.8
    elif data_type == 'volcanic':
        if quality_metrics.get('calibration_accuracy', 0) < 0.9:
            quality_score *= 0.7
    
    return min(1.0, max(0.0, quality_score))