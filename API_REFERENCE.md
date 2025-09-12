# Prometheus AI Platform - API Reference
## Comprehensive REST API Documentation

---

## 🌐 Microservices API Architecture

### Service Endpoints (✅ **IMPLEMENTED**)
```
API Gateway:           http://localhost:3000    # Node.js Express with service discovery
Helios Core Service:   http://localhost:8001    # Python FastAPI with PINN engine
Data Ingestion:        http://localhost:8002    # Python FastAPI with multi-source processing
Volcano Watchtower:    http://localhost:8003    # Python FastAPI with real-time monitoring
GCP Integration:       http://localhost:8004    # ✅ ENHANCED: Google Cloud Platform integration with advanced Dataflow processing
Advanced ML Pipeline:  http://localhost:8001/ml # ✅ NEW: Advanced ML capabilities integrated in Helios Core
```

### Infrastructure Services (✅ **IMPLEMENTED**)
```
Consul (Service Discovery): http://localhost:8500
PostgreSQL + PostGIS:       localhost:5432
InfluxDB (Time-series):     http://localhost:8086
Neo4j (Graph Database):     http://localhost:7474
Redis (Cache):              localhost:6379
Kafka (Message Streaming):  localhost:9092
```

### Production URLs
```
API Gateway:           https://api.prometheus-ai.com
Helios Core:           https://api.prometheus-ai.com/api/v1/helios
Data Ingestion:        https://api.prometheus-ai.com/api/v1/data
Volcano Watchtower:    https://api.prometheus-ai.com/api/v1/volcano
GCP Integration:       https://api.prometheus-ai.com/api/v1/gcp
```

### Future VR/AR Services (🔄 **IN DEVELOPMENT**)
```
VR Geological Explorer:    http://localhost:8007    # Virtual reality geological exploration
AR Field Assistant:        http://localhost:8008    # Augmented reality field data overlay
Haptic Feedback Service:   http://localhost:8009    # Tactile geological exploration
Collaborative VR/AR:       http://localhost:8010    # Multi-user immersive environments
Gesture Control Service:   http://localhost:8011    # Natural interaction interfaces
Immersive Analytics:       http://localhost:8012    # 3D/4D data visualization in VR/AR
```

### Authentication (✅ **IMPLEMENTED**)
All API endpoints require JWT authentication through the API Gateway.

```bash
# Login through API Gateway to get access token
curl -X POST http://localhost:3000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "prometheus123"}'

# Response includes JWT token
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {"id": 1, "username": "admin", "role": "admin"}
}

# Use token in subsequent requests to any service
curl -H "Authorization: Bearer <access_token>" \
     http://localhost:3000/api/v1/helios/analyze/geological
```

---

## 🚪 API Gateway Endpoints (Port 3000) ✅ **IMPLEMENTED**

### Authentication & Service Management

#### POST `/auth/login`
Authenticate user and receive JWT token for accessing all services.

**Request:**
```json
{
  "username": "admin",
  "password": "prometheus123"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": 1,
    "username": "admin", 
    "role": "admin"
  }
}
```

#### GET `/health`
Check API Gateway health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": ["prometheus-helios-core", "prometheus-data-ingestion", "prometheus-volcano-watchtower"]
}
```

#### GET `/api/services`
Get status of all registered microservices.

**Response:**
```json
{
  "services": [
    {
      "name": "prometheus-helios-core",
      "instances": 3,
      "healthy": true
    },
    {
      "name": "prometheus-data-ingestion", 
      "instances": 2,
      "healthy": true
    }
  ]
}
```

### Service Proxying
The API Gateway automatically routes requests to backend services:
- `/api/v1/helios/*` → Helios Core Service (Port 8001)
- `/api/v1/data/*` → Data Ingestion Service (Port 8002)
- `/api/v1/volcano/*` → Volcano Watchtower Service (Port 8003)

---

## 🧠 Helios Core Service API (Port 8001) ✅ **IMPLEMENTED**

### Geological Analysis & Physics Simulations

#### POST `/analyze/geological`
Perform geological data analysis using Physics-Informed Neural Networks.

**Request:**
```json
{
  "data_type": "seismic_tomography",
  "spatial_extent": {
    "min_lat": 37.0,
    "max_lat": 38.0,
    "min_lon": -123.0,
    "max_lon": -122.0
  },
  "temporal_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-31T23:59:59Z"
  },
  "parameters": {
    "resolution": "high",
    "uncertainty_quantification": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "results": {
    "velocity_model": {...},
    "uncertainty_bounds": {...},
    "processing_time": 45.2
  },
  "metadata": {
    "processing_time": 45.2,
    "model_version": "1.0.0",
    "confidence": 0.87
  },
  "correlation_id": "req_123456"
}
```

#### POST `/simulate/physics`
Run physics-informed simulations for geological processes.

**Request:**
```json
{
  "simulation_type": "mantle_convection",
  "initial_conditions": {
    "temperature_profile": "adiabatic",
    "viscosity_model": "temperature_dependent"
  },
  "boundary_conditions": {
    "surface_temperature": 273,
    "core_temperature": 4000
  },
  "time_span": 1000000,
  "parameters": {
    "resolution": "medium",
    "adaptive_timestep": true
  }
}
```

#### GET `/models/status`
Get status of loaded AI/ML models.

**Response:**
```json
{
  "status": "success",
  "models": {
    "geological_pinn": {
      "status": "loaded",
      "version": "2.1.0",
      "last_updated": "2024-01-15T08:00:00Z"
    },
    "physics_coupling": {
      "status": "loaded", 
      "version": "1.5.0",
      "last_updated": "2024-01-14T12:00:00Z"
    }
  }
}
```

---

## 📊 Data Ingestion Service API (Port 8002) ✅ **IMPLEMENTED**

### Multi-Source Data Processing

#### POST `/ingest`
Ingest data from multiple geological data sources.

**Request:**
```json
{
  "source": "IRIS",
  "data_type": "seismic_waveforms",
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-02T00:00:00Z"
  },
  "stations": ["BK.BRK", "BK.CMB", "BK.MNRC"],
  "parameters": {
    "sample_rate": 100,
    "format": "mseed"
  }
}
```

#### POST `/validate`
Validate geological data quality and consistency.

**Request:**
```json
{
  "data_id": "dataset_12345",
  "validation_rules": ["completeness", "accuracy", "consistency"],
  "quality_threshold": 0.8
}
```

#### GET `/sources`
List available data sources and their status.

**Response:**
```json
{
  "sources": [
    {
      "name": "IRIS",
      "type": "seismic",
      "status": "online",
      "last_sync": "2024-01-15T10:00:00Z"
    },
    {
      "name": "USGS",
      "type": "earthquake_catalog",
      "status": "online", 
      "last_sync": "2024-01-15T09:45:00Z"
    }
  ]
}
```

---

## 🌋 Volcano Watchtower Service API (Port 8003) ✅ **IMPLEMENTED**

### Real-Time Volcanic Monitoring

#### POST `/monitor`
Start monitoring a specific volcano with real-time data processing.

**Request:**
```json
{
  "volcano_id": "MSH",
  "monitoring_parameters": {
    "seismic_threshold": 2.0,
    "deformation_threshold": 5.0,
    "gas_emission_threshold": 1000
  },
  "alert_settings": {
    "email_notifications": true,
    "webhook_url": "https://alerts.example.com/volcano"
  }
}
```

#### POST `/forecast`
Generate eruption probability forecast for a volcano.

**Request:**
```json
{
  "volcano_id": "MSH",
  "forecast_horizon": "30_days",
  "include_uncertainty": true,
  "model_ensemble": true
}
```

**Response:**
```json
{
  "volcano_id": "MSH",
  "forecast": {
    "eruption_probability": 0.15,
    "confidence_interval": [0.08, 0.22],
    "vei_prediction": {
      "most_likely": 2,
      "range": [1, 4]
    }
  },
  "contributing_factors": {
    "seismic_activity": 0.3,
    "ground_deformation": 0.4,
    "gas_emissions": 0.2
  }
}
```

#### GET `/volcanoes/{id}/status`
Get current status and recent activity for a specific volcano.

**Response:**
```json
{
  "volcano_id": "MSH",
  "current_status": {
    "alert_level": "yellow",
    "last_updated": "2024-01-15T10:30:00Z",
    "seismic_events_24h": 12,
    "deformation_rate": "2.3 cm/year",
    "gas_emission_so2": "150 tons/day"
  },
  "recent_activity": [...],
  "monitoring_stations": 15
}
```

#### WebSocket `/ws/{volcano_id}`
Real-time WebSocket connection for live volcanic monitoring data.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8003/ws/MSH');
ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Real-time update:', data);
};
```

**Real-time Data Format:**
```json
{
  "timestamp": "2024-01-15T10:30:15Z",
  "volcano_id": "MSH",
  "data_type": "seismic_event",
  "magnitude": 2.1,
  "depth": 5.2,
  "location": [-122.1944, 46.1912]
}
```

---

## 🤖 MLOps Pipeline API (✅ **IMPLEMENTED**)

### Model Lifecycle Management

#### POST `/mlops/models/register`
Register a new ML model in the comprehensive model database system.

**Request:**
```json
{
  "model_name": "volcanic_eruption_predictor_v3",
  "model_type": "pytorch",
  "framework": "pytorch",
  "format": "pytorch",
  "description": "Advanced volcanic eruption prediction with attention mechanisms and uncertainty quantification",
  "geological_domain": "volcanic",
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "hidden_sizes": [256, 128, 64],
    "use_attention": true,
    "uncertainty_estimation": true
  },
  "performance_metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "f1_score": 0.89,
    "auc_roc": 0.94
  },
  "storage_backend": "multi_cloud",
  "tags": ["volcanic", "prediction", "attention", "uncertainty"]
}
```

**Response:**
```json
{
  "model_id": "model_12345",
  "registration_status": "success",
  "storage_locations": {
    "local": "./models/volcanic/eruption_probability/model_12345/v3.0.0",
    "aws_s3": "s3://prometheus-models/volcanic/eruption_probability/model_12345/v3.0.0",
    "azure_blob": "https://prometheusmodels.blob.core.windows.net/volcanic-models/model_12345/v3.0.0",
    "google_cloud": "gs://prometheus-gcp-models/volcanic/eruption_probability/model_12345/v3.0.0"
  },
  "checksum": "sha256:a1b2c3d4e5f6...",
  "model_size_bytes": 15728640,
  "validation_results": {
    "format_valid": true,
    "performance_verified": true,
    "inference_test_passed": true
  }
}
```

#### GET `/mlops/models/{model_id}`
Retrieve model metadata and download links.

**Response:**
```json
{
  "model_id": "model_12345",
  "name": "volcanic_eruption_predictor_v3",
  "version": "3.0.0",
  "status": "deployed",
  "performance_metrics": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "f1_score": 0.89
  },
  "deployment_info": {
    "deployed_at": "2025-01-11T10:30:00Z",
    "deployment_strategy": "gradual_rollout",
    "current_traffic_percentage": 100,
    "endpoints": [
      "https://api.prometheus-ai.com/api/v1/models/model_12345/predict"
    ]
  },
  "monitoring": {
    "prediction_count_24h": 15420,
    "average_latency_ms": 45.2,
    "error_rate": 0.002,
    "drift_detected": false
  }
}
```

### Model Training and Deployment

#### POST `/mlops/training/start`
Start automated model training with multi-cloud support.

**Request:**
```json
{
  "model_config": {
    "model_type": "volcanic_prediction",
    "architecture": "VolcanicPredictionNet",
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 100,
      "use_attention": true,
      "uncertainty_estimation": true
    }
  },
  "training_data": {
    "dataset_path": "s3://prometheus-data/volcanic_training_2025.csv",
    "validation_split": 0.2,
    "features": ["seismic_activity", "deformation", "gas_emissions", "thermal_anomalies"],
    "target": "eruption_probability"
  },
  "cloud_providers": ["azure", "gcp", "aws"],
  "deployment_config": {
    "auto_deploy": true,
    "deployment_strategy": "gradual_rollout",
    "success_criteria": {
      "min_accuracy": 0.85,
      "max_error_rate": 0.05
    }
  }
}
```

**Response:**
```json
{
  "training_job_id": "train_67890",
  "status": "initiated",
  "cloud_jobs": {
    "azure": {
      "job_id": "azure_job_12345",
      "status": "queued",
      "estimated_duration": "2h 30m"
    },
    "gcp": {
      "job_id": "vertex_ai_job_54321",
      "status": "running",
      "progress": 0.15
    },
    "aws": {
      "job_id": "sagemaker_job_98765",
      "status": "queued"
    }
  },
  "estimated_completion": "2025-01-11T14:30:00Z"
}
```

#### GET `/mlops/training/{job_id}/status`
Get training job status and progress.

**Response:**
```json
{
  "training_job_id": "train_67890",
  "overall_status": "training",
  "progress": 0.67,
  "best_model": {
    "cloud_provider": "gcp",
    "accuracy": 0.91,
    "loss": 0.18
  },
  "cloud_jobs": {
    "azure": {
      "status": "completed",
      "final_metrics": {
        "accuracy": 0.89,
        "loss": 0.23
      }
    },
    "gcp": {
      "status": "training",
      "current_epoch": 75,
      "current_metrics": {
        "accuracy": 0.91,
        "loss": 0.18
      }
    },
    "aws": {
      "status": "training",
      "current_epoch": 68,
      "current_metrics": {
        "accuracy": 0.88,
        "loss": 0.25
      }
    }
  }
}
```

### Model Monitoring and Drift Detection

#### POST `/mlops/monitoring/drift/detect`
Detect data drift in model inputs.

**Request:**
```json
{
  "model_id": "model_12345",
  "current_data": {
    "features": ["seismic_activity", "deformation", "gas_emissions"],
    "data": [
      [2.1, 0.5, 150],
      [1.8, 0.3, 120],
      [2.3, 0.7, 180]
    ]
  },
  "detection_method": "ks_test",
  "threshold": 0.05
}
```

**Response:**
```json
{
  "drift_report": {
    "overall_drift_detected": true,
    "drift_score": 0.23,
    "feature_reports": [
      {
        "feature_name": "seismic_activity",
        "drift_detected": false,
        "drift_score": 0.02,
        "p_value": 0.15
      },
      {
        "feature_name": "deformation",
        "drift_detected": true,
        "drift_score": 0.45,
        "p_value": 0.001
      },
      {
        "feature_name": "gas_emissions",
        "drift_detected": false,
        "drift_score": 0.08,
        "p_value": 0.12
      }
    ],
    "recommendations": [
      "Retrain model due to significant drift in deformation feature",
      "Investigate data quality for deformation measurements",
      "Consider updating reference data distribution"
    ]
  }
}
```

#### GET `/mlops/monitoring/{model_id}/performance`
Get model performance metrics and monitoring data.

**Response:**
```json
{
  "model_id": "model_12345",
  "monitoring_period": "24h",
  "performance_metrics": {
    "prediction_count": 15420,
    "average_latency_ms": 45.2,
    "p95_latency_ms": 89.5,
    "error_rate": 0.002,
    "accuracy": 0.89,
    "throughput_per_second": 178.5
  },
  "resource_usage": {
    "cpu_usage_percent": 65.2,
    "memory_usage_mb": 1024,
    "gpu_usage_percent": 78.5
  },
  "alerts": [
    {
      "type": "performance_degradation",
      "level": "warning",
      "message": "Accuracy dropped to 0.87 (below threshold of 0.88)",
      "timestamp": "2025-01-11T09:15:00Z"
    }
  ],
  "trends": {
    "accuracy_trend": "declining",
    "latency_trend": "stable",
    "error_rate_trend": "stable"
  }
}
```

### A/B Testing and Gradual Rollout

#### POST `/mlops/ab-test/create`
Create A/B test for model comparison.

**Request:**
```json
{
  "test_name": "eruption_predictor_v3_vs_v2",
  "model_a_id": "model_12345",
  "model_b_id": "model_67890",
  "traffic_split": [0.5, 0.5],
  "duration_hours": 48,
  "success_metrics": ["accuracy", "latency", "user_satisfaction"],
  "statistical_significance_threshold": 0.05
}
```

**Response:**
```json
{
  "test_id": "ab_test_54321",
  "status": "active",
  "start_time": "2025-01-11T10:00:00Z",
  "end_time": "2025-01-13T10:00:00Z",
  "routing_endpoint": "/api/v1/models/ab-test/ab_test_54321/predict",
  "monitoring_dashboard": "https://monitoring.prometheus-ai.com/ab-tests/ab_test_54321"
}
```

#### GET `/mlops/ab-test/{test_id}/results`
Get A/B test results and statistical analysis.

**Response:**
```json
{
  "test_id": "ab_test_54321",
  "test_name": "eruption_predictor_v3_vs_v2",
  "status": "completed",
  "duration_hours": 48,
  "total_requests": 25680,
  "model_a_results": {
    "requests": 12840,
    "accuracy": 0.91,
    "average_latency_ms": 42.3,
    "error_rate": 0.001
  },
  "model_b_results": {
    "requests": 12840,
    "accuracy": 0.89,
    "average_latency_ms": 45.2,
    "error_rate": 0.002
  },
  "statistical_analysis": {
    "accuracy_difference": 0.02,
    "accuracy_p_value": 0.003,
    "accuracy_significant": true,
    "latency_difference": -2.9,
    "latency_p_value": 0.12,
    "latency_significant": false
  },
  "recommendation": {
    "winner": "model_a",
    "confidence": 0.95,
    "action": "promote_to_production"
  }
}
```

#### POST `/mlops/rollout/gradual`
Start gradual rollout of new model version.

**Request:**
```json
{
  "rollout_name": "eruption_predictor_v3_rollout",
  "old_model_id": "model_67890",
  "new_model_id": "model_12345",
  "rollout_stages": [
    {"traffic_percentage": 5, "duration_minutes": 30},
    {"traffic_percentage": 25, "duration_minutes": 60},
    {"traffic_percentage": 50, "duration_minutes": 120},
    {"traffic_percentage": 100, "duration_minutes": 0}
  ],
  "success_criteria": {
    "max_error_rate": 0.05,
    "min_accuracy": 0.85,
    "max_latency_p95": 100
  },
  "auto_rollback": true
}
```

**Response:**
```json
{
  "rollout_id": "rollout_98765",
  "status": "active",
  "current_stage": 0,
  "current_traffic_percentage": 5,
  "next_stage_time": "2025-01-11T10:30:00Z",
  "monitoring_dashboard": "https://monitoring.prometheus-ai.com/rollouts/rollout_98765"
}
```

---

## ☁️ GCP Integration Service API (Port 8004) ✅ **ENHANCED**

### Advanced Dataflow Processing

#### POST `/api/v1/dataflow/process-stream`
Process geological data streams using Google Cloud Dataflow with specialized pipelines.

**Request:**
```json
{
  "stream_name": "volcanic-monitoring-stream",
  "processing_type": "volcanic",
  "parameters": {
    "deformation_threshold": 5.0,
    "gas_flux_threshold": 500.0,
    "thermal_threshold": 25.0,
    "volcano_ids": ["mount_st_helens", "yellowstone"]
  }
}
```

**Response:**
```json
{
  "processing_type": "volcanic",
  "job_id": "dataflow-job-12345",
  "job_name": "volcanic-monitoring-1642234567",
  "status": "running",
  "parameters": {
    "deformation_threshold": 5.0,
    "gas_flux_threshold": 500.0,
    "thermal_threshold": 25.0,
    "volcano_ids": ["mount_st_helens", "yellowstone"]
  },
  "created_at": "2025-01-11T10:30:00Z"
}
```

#### POST `/api/v1/dataflow/process-stream` (Seismic Processing)
Process seismic data streams with real-time event detection.

**Request:**
```json
{
  "stream_name": "seismic-data-stream",
  "processing_type": "seismic",
  "parameters": {
    "magnitude_threshold": 4.0,
    "depth_threshold": 100.0,
    "region_filter": "california"
  }
}
```

**Response:**
```json
{
  "processing_type": "seismic",
  "job_id": "dataflow-job-67890",
  "job_name": "seismic-processing-1642234567",
  "status": "running",
  "parameters": {
    "magnitude_threshold": 4.0,
    "depth_threshold": 100.0,
    "region_filter": "california"
  },
  "created_at": "2025-01-11T10:30:00Z"
}
```

#### POST `/api/v1/dataflow/process-stream` (Real-time Alerts)
Process real-time alert streams with intelligent prioritization and routing.

**Request:**
```json
{
  "stream_name": "geological-alerts-stream",
  "processing_type": "real_time_alerts",
  "parameters": {
    "alert_types": ["seismic", "volcanic", "landslide"],
    "priority_levels": ["critical", "high", "medium"],
    "notification_channels": ["email", "sms", "webhook"]
  }
}
```

**Response:**
```json
{
  "processing_type": "real_time_alerts",
  "job_id": "dataflow-job-13579",
  "job_name": "real-time-alerts-1642234567",
  "status": "running",
  "parameters": {
    "alert_types": ["seismic", "volcanic", "landslide"],
    "priority_levels": ["critical", "high", "medium"],
    "notification_channels": ["email", "sms", "webhook"]
  },
  "created_at": "2025-01-11T10:30:00Z"
}
```

#### GET `/api/v1/dataflow/jobs/{job_id}`
Get Dataflow job status and details.

**Response:**
```json
{
  "job_id": "dataflow-job-12345",
  "job_name": "volcanic-monitoring-1642234567",
  "status": "JOB_STATE_RUNNING",
  "create_time": "2025-01-11T10:30:00Z",
  "start_time": "2025-01-11T10:32:00Z",
  "job_type": "JOB_TYPE_STREAMING",
  "pipeline_description": {
    "display_data": [
      {
        "key": "inputSubscription",
        "value": "projects/prometheus-ai/subscriptions/volcanic-monitoring-stream"
      },
      {
        "key": "outputTable", 
        "value": "prometheus-ai:geological_analytics.volcanic_monitoring"
      }
    ]
  }
}
```

#### GET `/api/v1/dataflow/jobs`
List all active Dataflow jobs.

**Response:**
```json
{
  "active_jobs": [
    {
      "job_id": "dataflow-job-12345",
      "job_name": "volcanic-monitoring-1642234567",
      "status": "JOB_STATE_RUNNING",
      "create_time": "2025-01-11T10:30:00Z",
      "job_type": "JOB_TYPE_STREAMING"
    },
    {
      "job_id": "dataflow-job-67890",
      "job_name": "seismic-processing-1642234567",
      "status": "JOB_STATE_RUNNING",
      "create_time": "2025-01-11T10:25:00Z",
      "job_type": "JOB_TYPE_STREAMING"
    }
  ]
}
```

#### DELETE `/api/v1/dataflow/jobs/{job_id}`
Cancel a running Dataflow job.

**Response:**
```json
{
  "job_id": "dataflow-job-12345",
  "status": "JOB_STATE_CANCELLED",
  "message": "Job cancellation requested"
}
```

### Dataflow Pipeline Templates

The GCP Integration service supports 4 specialized Dataflow pipeline templates:

1. **Seismic Event Detector** (`seismic_event_detector`)
   - Real-time seismic event detection and classification
   - Magnitude and depth threshold filtering
   - Regional filtering capabilities
   - Automated alert generation

2. **Volcanic Alert System** (`volcanic_alert_system`)
   - Multi-parameter volcanic monitoring
   - Deformation, gas flux, and thermal threshold monitoring
   - Volcano-specific configuration
   - Real-time eruption forecasting

3. **Geological Stream Processor** (`geological_stream_processor`)
   - General geological data stream processing
   - Configurable windowing and data validation
   - Multi-format data support
   - Quality control and anomaly detection

4. **Mineral Data Enricher** (`mineral_data_enricher`)
   - Geochemical data processing and enrichment
   - Element filtering and sample type classification
   - Quality threshold validation
   - Mineral deposit correlation

### Processing Configuration Types

#### Seismic Processing Configuration
```json
{
  "pipeline_type": "streaming",
  "input_source": "pubsub",
  "output_sink": "bigquery",
  "processing_functions": [
    "seismic_event_detection",
    "magnitude_classification", 
    "location_validation",
    "alert_generation"
  ],
  "windowing": {
    "type": "sliding",
    "size": "60s",
    "period": "10s"
  }
}
```

#### Real-time Alerts Configuration
```json
{
  "pipeline_type": "streaming",
  "input_source": "pubsub",
  "output_sink": "pubsub",
  "processing_functions": [
    "threshold_monitoring",
    "alert_prioritization",
    "notification_routing",
    "escalation_management"
  ],
  "windowing": {
    "type": "global"
  }
}
```

---

## 🤖 Advanced ML Pipeline API (✅ **PRODUCTION-READY**)

### Neural Architecture Search (NAS)

#### POST `/ml-pipeline/nas/search`
Perform automated neural architecture search for optimal geological model design.

**Request:**
```json
{
  "search_config": {
    "max_trials": 100,
    "max_epochs_per_trial": 50,
    "pruning_enabled": true,
    "multi_objective": true,
    "objectives": ["accuracy", "latency", "memory"]
  },
  "model_constraints": {
    "max_parameters": 10000000,
    "max_layers": 8,
    "activation_functions": ["relu", "gelu", "swish"],
    "dropout_range": [0.1, 0.5]
  },
  "training_data": {
    "dataset_path": "s3://prometheus-data/geological_training.csv",
    "input_dim": 128,
    "output_dim": 5,
    "validation_split": 0.2
  }
}
```

**Response:**
```json
{
  "search_id": "nas_search_12345",
  "status": "completed",
  "best_architecture": {
    "trial_id": 67,
    "parameters": {
      "n_layers": 6,
      "hidden_dim": 256,
      "dropout_rate": 0.3,
      "activation": "gelu"
    },
    "performance": {
      "accuracy": 0.94,
      "latency_ms": 45.2,
      "memory_mb": 128,
      "parameters": 8500000
    }
  },
  "optimization_history": [...],
  "pareto_front": [...]
}
```

### Ensemble Learning

#### POST `/ml-pipeline/ensemble/create`
Create ensemble of diverse geological models with uncertainty quantification.

**Request:**
```json
{
  "ensemble_config": {
    "base_models": ["cnn", "transformer", "gnn", "pinn"],
    "stacking_method": "neural_stacking",
    "uncertainty_method": "deep_ensemble",
    "num_ensemble_members": 5,
    "diversity_regularization": 0.1
  },
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "early_stopping": true
  }
}
```

**Response:**
```json
{
  "ensemble_id": "ensemble_67890",
  "status": "training",
  "base_models": [
    {
      "model_type": "cnn",
      "status": "completed",
      "accuracy": 0.89
    },
    {
      "model_type": "transformer", 
      "status": "training",
      "progress": 0.75
    }
  ],
  "meta_model_status": "pending",
  "estimated_completion": "2025-01-11T16:30:00Z"
}
```

#### POST `/ml-pipeline/ensemble/{ensemble_id}/predict`
Make predictions with uncertainty quantification using ensemble models.

**Request:**
```json
{
  "input_data": {
    "seismic_features": [2.1, 1.8, 2.3],
    "deformation_data": [0.5, 0.3, 0.7],
    "geochemical_data": [150, 120, 180]
  },
  "uncertainty_options": {
    "confidence_intervals": [0.68, 0.95],
    "monte_carlo_samples": 100
  }
}
```

**Response:**
```json
{
  "predictions": {
    "mean": [0.15, 0.25, 0.35, 0.20, 0.05],
    "std": [0.03, 0.04, 0.05, 0.03, 0.01],
    "confidence_intervals": {
      "68%": [[0.12, 0.18], [0.21, 0.29], ...],
      "95%": [[0.09, 0.21], [0.17, 0.33], ...]
    }
  },
  "ensemble_agreement": 0.87,
  "model_contributions": {
    "cnn": 0.25,
    "transformer": 0.30,
    "gnn": 0.20,
    "pinn": 0.25
  }
}
```

### Transfer Learning

#### POST `/ml-pipeline/transfer/adapt`
Adapt pre-trained geological models to new domains with physics constraints.

**Request:**
```json
{
  "transfer_config": {
    "source_domain": "general_geology",
    "target_domain": "volcanic_systems",
    "adaptation_method": "domain_adversarial",
    "physics_constraints": true,
    "fine_tuning_strategy": "gradual_unfreezing"
  },
  "source_model_id": "model_12345",
  "target_data": {
    "dataset_path": "s3://prometheus-data/volcanic_target.csv",
    "validation_split": 0.2
  },
  "adaptation_params": {
    "adaptation_epochs": 50,
    "learning_rate": 0.0001,
    "domain_loss_weight": 0.1
  }
}
```

**Response:**
```json
{
  "adaptation_id": "adapt_54321",
  "status": "completed",
  "adapted_model_id": "model_adapted_67890",
  "performance": {
    "source_domain_accuracy": 0.89,
    "target_domain_accuracy": 0.85,
    "domain_adaptation_loss": 0.12
  },
  "physics_constraints_satisfied": true,
  "adaptation_time": "45 minutes"
}
```

### Few-Shot Learning

#### POST `/ml-pipeline/few-shot/train`
Train models for rare geological phenomena using meta-learning approaches.

**Request:**
```json
{
  "few_shot_config": {
    "support_shots": 5,
    "query_shots": 15,
    "meta_learning_algorithm": "maml",
    "adaptation_steps": 5,
    "adaptation_lr": 0.01
  },
  "task_distribution": {
    "rare_mineral_identification": 0.3,
    "unusual_seismic_patterns": 0.4,
    "exotic_volcanic_processes": 0.3
  },
  "meta_training_episodes": 1000
}
```

**Response:**
```json
{
  "few_shot_model_id": "few_shot_98765",
  "meta_training_status": "completed",
  "performance": {
    "meta_accuracy": 0.78,
    "adaptation_speed": "3 gradient steps",
    "generalization_score": 0.82
  },
  "supported_tasks": [
    "rare_mineral_identification",
    "unusual_seismic_patterns", 
    "exotic_volcanic_processes"
  ]
}
```

### Continual Learning

#### POST `/ml-pipeline/continual/update`
Update models with new geological data while preventing catastrophic forgetting.

**Request:**
```json
{
  "continual_config": {
    "memory_size": 1000,
    "rehearsal_method": "experience_replay",
    "regularization_method": "elastic_weight_consolidation",
    "plasticity_stability_tradeoff": 0.5
  },
  "new_data": {
    "dataset_path": "s3://prometheus-data/new_geological_data.csv",
    "task_type": "volcanic_eruption_prediction"
  },
  "base_model_id": "model_12345"
}
```

**Response:**
```json
{
  "continual_model_id": "continual_13579",
  "update_status": "completed",
  "performance": {
    "new_task_accuracy": 0.87,
    "old_task_retention": 0.91,
    "forgetting_measure": 0.02
  },
  "memory_usage": {
    "replay_buffer_size": 1000,
    "importance_weights_stored": 50000
  }
}
```

### Generative Models

#### POST `/ml-pipeline/generative/synthesize`
Generate realistic geological data using physics-informed GANs.

**Request:**
```json
{
  "generative_config": {
    "model_type": "physics_informed_gan",
    "latent_dim": 128,
    "physics_loss_weight": 1.0,
    "adversarial_loss_weight": 1.0
  },
  "synthesis_params": {
    "num_samples": 1000,
    "geological_constraints": {
      "rock_type": "volcanic",
      "age_range": [0, 100],
      "location_bounds": {
        "min_lat": 40.0,
        "max_lat": 45.0,
        "min_lon": -125.0,
        "max_lon": -120.0
      }
    }
  }
}
```

**Response:**
```json
{
  "synthesis_id": "synth_24680",
  "generated_data": {
    "dataset_path": "s3://prometheus-data/synthetic_geological_data.csv",
    "num_samples": 1000,
    "data_quality_score": 0.89
  },
  "physics_validation": {
    "conservation_laws_satisfied": true,
    "thermodynamic_consistency": 0.94,
    "geological_realism_score": 0.91
  },
  "generation_time": "12 minutes"
}
```

### Causal Inference

#### POST `/ml-pipeline/causal/discover`
Discover causal relationships in geological processes using advanced algorithms.

**Request:**
```json
{
  "causal_config": {
    "method": "pc_algorithm",
    "independence_test": "partial_correlation",
    "significance_level": 0.05,
    "max_conditioning_set_size": 3
  },
  "data": {
    "dataset_path": "s3://prometheus-data/geological_processes.csv",
    "variables": [
      "seismic_activity",
      "ground_deformation", 
      "gas_emissions",
      "temperature_anomalies",
      "eruption_occurrence"
    ]
  }
}
```

**Response:**
```json
{
  "causal_graph": {
    "nodes": ["seismic_activity", "ground_deformation", "gas_emissions", "temperature_anomalies", "eruption_occurrence"],
    "edges": [
      {
        "from": "seismic_activity",
        "to": "ground_deformation",
        "strength": 0.78,
        "confidence": 0.95
      },
      {
        "from": "gas_emissions",
        "to": "eruption_occurrence", 
        "strength": 0.85,
        "confidence": 0.98
      }
    ]
  },
  "causal_insights": [
    "Seismic activity is a strong causal driver of ground deformation",
    "Gas emissions are the strongest predictor of eruption occurrence",
    "Temperature anomalies show weak causal relationships"
  ],
  "statistical_tests": {
    "independence_tests_performed": 45,
    "significant_relationships": 12,
    "false_discovery_rate": 0.05
  }
}
```

---

## 🗄️ Database Schema Integration API

### Geological Database Management

#### GET `/database/geological-structures`
Retrieve geological structures with comprehensive age information and metadata.

**Query Parameters:**
- `location` (optional): Geographic bounding box in WKT format
- `age_min` (optional): Minimum age in Ma
- `age_max` (optional): Maximum age in Ma
- `lithology` (optional): Filter by lithology type
- `data_quality` (optional): Filter by data quality grade (A, B, C, D)
- `confidence_level` (optional): Minimum confidence level (0.0-1.0)

**Response:**
```json
{
  "structures": [
    {
      "id": 1,
      "structure_id": "STRUCT_001",
      "location": {
        "type": "Point",
        "coordinates": [-122.4194, 37.7749, 100]
      },
      "structure_type": "intrusive_pluton",
      "lithology": "granite",
      "age": 150.5,
      "properties": {
        "formation_age": 150.5,
        "radiometric_age": 148.2,
        "stratigraphic_age": "Jurassic",
        "age_uncertainty": 2.5,
        "dating_method": "U-Pb zircon",
        "confidence_level": 0.85,
        "geological_unit": "Sierra Nevada Batholith"
      },
      "data_quality_grade": "A",
      "confidence_level": 0.85
    }
  ],
  "total_count": 1,
  "page": 1,
  "per_page": 50
}
```

#### POST `/database/rock-mechanical-properties`
Add rock mechanical properties data to the database.

**Request Body:**
```json
{
  "formation_id": "FORM_001",
  "lithology": "granite",
  "unconfined_compressive_strength_mpa": 120.5,
  "tensile_strength_mpa": 8.2,
  "cohesion_mpa": 15.0,
  "friction_angle_deg": 35.0,
  "elastic_modulus_gpa": 45.0,
  "poisson_ratio": 0.25,
  "permeability_m2": 1e-18,
  "thermal_conductivity_w_m_k": 2.8,
  "thermal_expansion_per_k": 8e-6,
  "porosity": 0.02,
  "fracture_density_per_m3": 0.1,
  "data_quality_grade": "A",
  "test_standard": "ISRM",
  "sample_location": {
    "type": "Point",
    "coordinates": [-122.4194, 37.7749, 100]
  },
  "collected_date": "2024-01-15"
}
```

#### GET `/database/volcanic-chambers`
Retrieve volcanic chamber data with comprehensive geometric and thermodynamic properties.

**Response:**
```json
{
  "chambers": [
    {
      "id": 1,
      "chamber_id": "CHAMBER_001",
      "location": {
        "type": "Point",
        "coordinates": [-122.1817, 37.7217, -5000]
      },
      "chamber_volume_km3": 50.0,
      "chamber_depth_km": 5.0,
      "chamber_shape": "ellipsoid",
      "melt_fraction": 0.3,
      "temperature_c": 850.0,
      "pressure_mpa": 150.0,
      "volatile_content": {
        "H2O": 4.0,
        "CO2": 0.5,
        "SO2": 0.1
      },
      "confidence_level": 0.75
    }
  ]
}
```

### Model Database System API

#### POST `/models/register`
Register a new ML model in the model database system.

**Request Body:**
```json
{
  "model_name": "eruption_predictor_v2",
  "model_type": "neural_network",
  "framework": "pytorch",
  "version": "2.1.0",
  "description": "Advanced eruption prediction model with transformer architecture",
  "metadata": {
    "architecture": "transformer",
    "input_features": 128,
    "output_classes": 5,
    "training_dataset": "global_volcanic_events_2024",
    "performance_metrics": {
      "accuracy": 0.89,
      "precision": 0.87,
      "recall": 0.91,
      "f1_score": 0.89
    }
  },
  "storage_backend": "s3",
  "model_path": "s3://prometheus-models/eruption_predictor_v2.pth",
  "validation_status": "validated",
  "tags": ["volcanic", "prediction", "transformer", "production"]
}
```

**Response:**
```json
{
  "model_id": "model_12345",
  "registration_status": "success",
  "model_version_id": "version_67890",
  "storage_location": "s3://prometheus-models/eruption_predictor_v2.pth",
  "validation_results": {
    "format_valid": true,
    "metadata_complete": true,
    "performance_verified": true
  },
  "created_at": "2025-01-11T10:30:00Z"
}
```

### Advanced Model Manager API (Task 2.4 - ✅ COMPLETED)

The Advanced Model Manager provides comprehensive ML operations with multi-cloud integration, advanced neural network architectures, physics-informed constraints, uncertainty quantification, and cross-cloud model synchronization.

#### POST `/models/train/multi-cloud`
Train models across multiple cloud providers with advanced MLOps integration.

**Request Body:**
```json
{
  "model_config": {
    "model_type": "volcanic_prediction",
    "hyperparameters": {
      "learning_rate": 0.001,
      "batch_size": 32,
      "epochs": 100,
      "hidden_sizes": [256, 128, 64],
      "use_attention": true,
      "uncertainty_estimation": true
    },
    "multi_cloud_training": true,
    "enable_vertex_ai": true,
    "enable_sagemaker": true,
    "enable_azure_ml": true,
    "auto_deploy": true
  },
  "training_data": {
    "dataset_path": "s3://prometheus-data/volcanic_training_2024.csv",
    "validation_split": 0.2,
    "features": ["seismic_activity", "deformation", "gas_emissions", "thermal_anomalies"],
    "targets": ["eruption_probability", "vei", "time_to_event"]
  },
  "cloud_providers": {
    "azure": {
      "vm_endpoint": "https://prometheus-azure.eastus.cloudapp.azure.com",
      "storage_account": "prometheusmodels",
      "container": "volcanic-models"
    },
    "gcp": {
      "project_id": "prometheus-ai-platform",
      "region": "us-central1",
      "bucket": "prometheus-gcp-models"
    },
    "aws": {
      "region": "us-west-2",
      "bucket": "prometheus-aws-models"
    }
  }
}
```

**Response:**
```json
{
  "training_job_id": "train_multi_12345",
  "status": "initiated",
  "cloud_jobs": {
    "azure": {
      "job_id": "azure_job_67890",
      "status": "queued",
      "endpoint": "https://prometheus-azure.eastus.cloudapp.azure.com/jobs/67890"
    },
    "gcp": {
      "job_id": "gcp_job_54321",
      "status": "running",
      "vertex_ai_job": "projects/prometheus-ai-platform/locations/us-central1/customJobs/54321"
    },
    "aws": {
      "job_id": "aws_job_98765",
      "status": "queued",
      "sagemaker_job": "prometheus-training-2025-01-11-10-30-00"
    }
  },
  "estimated_completion": "2025-01-11T14:30:00Z",
  "model_architectures": {
    "volcanic_prediction_net": {
      "type": "VolcanicPredictionNet",
      "features": ["attention_mechanism", "uncertainty_quantification", "physics_constraints"]
    },
    "multi_task_net": {
      "type": "MultiTaskVolcanicNet",
      "tasks": ["eruption_probability", "vei_distribution", "time_to_event"]
    }
  }
}
```

#### GET `/models/train/{training_job_id}/status`
Get the status of a multi-cloud training job.

**Response:**
```json
{
  "training_job_id": "train_multi_12345",
  "overall_status": "training",
  "progress": 0.67,
  "estimated_completion": "2025-01-11T14:30:00Z",
  "cloud_jobs": {
    "azure": {
      "status": "completed",
      "progress": 1.0,
      "metrics": {
        "accuracy": 0.89,
        "loss": 0.23,
        "training_time": "2h 15m"
      },
      "model_location": "https://prometheusmodels.blob.core.windows.net/volcanic-models/model_v2.1.0.pth"
    },
    "gcp": {
      "status": "training",
      "progress": 0.75,
      "current_epoch": 75,
      "metrics": {
        "accuracy": 0.87,
        "loss": 0.25
      }
    },
    "aws": {
      "status": "training",
      "progress": 0.68,
      "current_epoch": 68,
      "metrics": {
        "accuracy": 0.86,
        "loss": 0.27
      }
    }
  },
  "best_performing_model": {
    "cloud_provider": "azure",
    "accuracy": 0.89,
    "model_path": "https://prometheusmodels.blob.core.windows.net/volcanic-models/model_v2.1.0.pth"
  }
}
```

#### POST `/models/predict/volcanic`
Make predictions using advanced volcanic prediction models with uncertainty quantification.

**Request Body:**
```json
{
  "model_version": "v2.1.0",
  "input_data": {
    "seismic_activity": [2.1, 1.8, 2.3, 1.9, 2.5],
    "deformation_rate": 0.5,
    "gas_emissions": {
      "SO2": 150,
      "CO2": 8500,
      "H2S": 25
    },
    "thermal_anomalies": 3,
    "location": {
      "latitude": 44.428,
      "longitude": -110.588,
      "volcano_id": "yellowstone"
    }
  },
  "prediction_options": {
    "uncertainty_quantification": true,
    "monte_carlo_samples": 100,
    "confidence_intervals": [0.68, 0.95],
    "physics_constraints": true
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_24680",
  "model_version": "v2.1.0",
  "predictions": {
    "eruption_probability": {
      "mean": 0.15,
      "std": 0.03,
      "confidence_intervals": {
        "68%": [0.12, 0.18],
        "95%": [0.09, 0.21]
      }
    },
    "vei_distribution": {
      "VEI_0": 0.05,
      "VEI_1": 0.15,
      "VEI_2": 0.25,
      "VEI_3": 0.30,
      "VEI_4": 0.20,
      "VEI_5": 0.04,
      "VEI_6": 0.01
    },
    "time_to_event": {
      "mean_days": 172,
      "std_days": 45,
      "confidence_intervals": {
        "68%": [127, 217],
        "95%": [82, 262]
      }
    }
  },
  "model_performance": {
    "accuracy": 0.89,
    "precision": 0.87,
    "recall": 0.91,
    "uncertainty_calibration": 0.94
  },
  "physics_constraints_applied": {
    "magma_chamber_pressure": "validated",
    "thermal_equilibrium": "enforced",
    "gas_solubility": "constrained"
  },
  "feature_importance": {
    "seismic_activity": 0.35,
    "deformation_rate": 0.28,
    "gas_emissions": 0.22,
    "thermal_anomalies": 0.15
  }
}
```

#### POST `/models/sync/cross-cloud`
Synchronize models across all cloud providers.

**Request Body:**
```json
{
  "sync_options": {
    "direction": "bidirectional",
    "include_metadata": true,
    "include_performance_metrics": true,
    "conflict_resolution": "latest_version"
  },
  "cloud_providers": ["azure", "gcp", "aws"],
  "model_filters": {
    "model_type": "volcanic_prediction",
    "min_accuracy": 0.8,
    "status": "validated"
  }
}
```

**Response:**
```json
{
  "sync_job_id": "sync_13579",
  "status": "completed",
  "synchronized_models": [
    {
      "model_id": "model_12345",
      "model_name": "eruption_predictor_v2",
      "sync_status": "success",
      "locations": {
        "azure": "https://prometheusmodels.blob.core.windows.net/volcanic-models/model_v2.1.0.pth",
        "gcp": "gs://prometheus-gcp-models/volcanic-models/model_v2.1.0.pth",
        "aws": "s3://prometheus-aws-models/volcanic-models/model_v2.1.0.pth"
      }
    }
  ],
  "sync_summary": {
    "total_models": 15,
    "successful_syncs": 14,
    "failed_syncs": 1,
    "conflicts_resolved": 2
  }
}
```

#### GET `/models/{model_id}/versions`
Retrieve all versions of a specific model with metadata and performance tracking.

**Response:**
```json
{
  "model_id": "model_12345",
  "model_name": "eruption_predictor",
  "versions": [
    {
      "version_id": "version_67890",
      "version": "2.1.0",
      "created_at": "2025-01-11T10:30:00Z",
      "parent_version": "version_67889",
      "performance_metrics": {
        "accuracy": 0.89,
        "precision": 0.87,
        "recall": 0.91
      },
      "deployment_status": "production",
      "validation_status": "validated",
      "change_summary": "Improved transformer architecture with attention mechanisms"
    }
  ],
  "total_versions": 15,
  "latest_version": "2.1.0"
}
```

### Advanced Data Integration Pipeline API (Task 2.3 - ✅ COMPLETED)

The Advanced Data Integration Pipeline provides comprehensive multi-source geological data processing capabilities with real-time volcanic data enhancement, automated feature engineering, comprehensive quality assessment, and anomaly detection systems.

#### POST `/data-integration/process`
Process multi-source geological data through the advanced integration pipeline.

**Request Body:**
```json
{
  "data_sources": [
    {
      "source_type": "seismic",
      "source_url": "https://service.iris.edu/fdsnws/event/1/",
      "parameters": {
        "starttime": "2024-01-01",
        "endtime": "2024-12-31",
        "minmagnitude": 4.0,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "maxradius": 5.0
      }
    },
    {
      "source_type": "insar",
      "source_url": "https://asf.alaska.edu/api/",
      "parameters": {
        "platform": "SENTINEL-1",
        "processingLevel": "L1",
        "bbox": "-122.5,-122.3,37.7,37.8"
      }
    },
    {
      "source_type": "geophysical",
      "source_url": "https://geomag.usgs.gov/api/",
      "parameters": {
        "data_type": "gravity",
        "survey_area": "yellowstone",
        "resolution": "high"
      }
    },
    {
      "source_type": "geochemical",
      "source_url": "https://earthchem.org/api/",
      "parameters": {
        "sample_type": "gas_emission",
        "location": "volcanic_vent",
        "compounds": ["SO2", "CO2", "H2S"]
      }
    }
  ],
  "processing_options": {
    "quality_assessment": true,
    "anomaly_detection": true,
    "feature_engineering": true,
    "real_time_enhancement": true,
    "geological_context_integration": true,
    "multi_parameter_fusion": true
  },
  "output_format": "geojson"
}
```

**Response:**
```json
{
  "processing_id": "proc_12345",
  "status": "completed",
  "processed_data": {
    "seismic_events": 45,
    "insar_points": 12500,
    "geophysical_measurements": 850,
    "geochemical_samples": 23,
    "quality_score": 0.87,
    "anomalies_detected": 3
  },
  "enhanced_features": {
    "temporal_patterns": {
      "seismic_swarms": 2,
      "deformation_trends": "accelerating",
      "gas_emission_changes": "increasing"
    },
    "spatial_correlations": {
      "seismic_deformation_correlation": 0.78,
      "thermal_gas_correlation": 0.92
    },
    "geological_context": {
      "magma_chamber_proximity": 0.85,
      "fault_system_interaction": 0.67,
      "hydrothermal_activity": 0.73
    }
  },
  "data_quality_assessment": {
    "completeness": 0.92,
    "accuracy": 0.89,
    "consistency": 0.94,
    "timeliness": 0.96
  },
  "real_time_enhancements": {
    "volcanic_data_fusion": "active",
    "predictive_modeling": "enabled",
    "alert_generation": "configured"
  },
  "output_location": "s3://prometheus-data/processed/proc_12345.geojson"
}
```

#### GET `/data-integration/jobs/{processing_id}/status`
Get the status of a data integration job.

**Response:**
```json
{
  "processing_id": "proc_12345",
  "status": "processing",
  "progress": 0.67,
  "estimated_completion": "2025-01-09T16:45:00Z",
  "current_stage": "feature_engineering",
  "stages_completed": [
    "data_ingestion",
    "quality_assessment",
    "anomaly_detection"
  ],
  "stages_remaining": [
    "geological_context_integration",
    "output_generation"
  ]
}
```

#### POST `/data-integration/quality-assessment`
Perform comprehensive data quality assessment on geological datasets.

**Request Body:**
```json
{
  "dataset_id": "dataset_12345",
  "assessment_criteria": {
    "completeness_check": true,
    "accuracy_validation": true,
    "consistency_analysis": true,
    "timeliness_evaluation": true,
    "geological_validity": true
  },
  "reference_standards": ["USGS", "IRIS", "GVP"]
}
```

**Response:**
```json
{
  "assessment_id": "qa_67890",
  "overall_quality_score": 0.87,
  "detailed_assessment": {
    "completeness": {
      "score": 0.92,
      "missing_data_percentage": 8.0,
      "gaps_identified": 3
    },
    "accuracy": {
      "score": 0.89,
      "validation_errors": 12,
      "outliers_detected": 5
    },
    "consistency": {
      "score": 0.94,
      "inconsistencies": 2,
      "cross_validation_passed": true
    },
    "timeliness": {
      "score": 0.96,
      "data_latency": "2.3 hours",
      "update_frequency": "real-time"
    }
  },
  "recommendations": [
    "Fill temporal gaps using interpolation",
    "Validate outlier measurements",
    "Implement additional quality controls"
  ]
}
```

---

## 🌋 Volcanic Systems API

### Multi-Chamber Volcanic System Simulation

#### POST `/volcanic/multi-chamber/simulate`
Simulate multi-chamber volcanic system with physics-informed neural networks.

**Request Body:**
```json
{
  "chambers": [
    {
      "id": "chamber_1",
      "depth": 5000,
      "volume": 1e9,
      "temperature": 1200,
      "pressure": 150e6,
      "composition": {
        "SiO2": 65.0,
        "H2O": 4.0,
        "CO2": 0.5
      }
    }
  ],
  "connections": [
    {
      "from": "chamber_1",
      "to": "chamber_2",
      "conductivity": 0.1,
      "cross_section": 100
    }
  ],
  "time_range": {
    "start": 0,
    "end": 86400,
    "step": 3600
  },
  "parameters": {
    "physics_model": "thermodynamic",
    "ml_enhancement": true,
    "uncertainty_quantification": true
  }
}
```

**Response:**
```json
{
  "simulation_id": "sim_12345",
  "status": "completed",
  "results": {
    "temperature_evolution": [...],
    "pressure_evolution": [...],
    "volume_changes": [...],
    "gas_emissions": [...]
  },
  "ml_predictions": {
    "eruption_probability": 0.15,
    "time_to_eruption": 172800,
    "expected_vei": 3,
    "confidence": 0.87
  },
  "uncertainty": {
    "eruption_probability_bounds": [0.08, 0.23],
    "vei_distribution": {...}
  }
}
```

### Real-time Volcanic Monitoring

#### GET `/volcanic/monitoring/{volcano_id}/status`
Get current monitoring status for a specific volcano.

**Parameters:**
- `volcano_id`: Volcano identifier (e.g., "yellowstone", "vesuvius")

**Response:**
```json
{
  "volcano_id": "yellowstone",
  "name": "Yellowstone Caldera",
  "location": {
    "latitude": 44.428,
    "longitude": -110.588,
    "elevation": 2805
  },
  "current_status": {
    "alert_level": "advisory",
    "last_updated": "2025-01-09T15:30:00Z",
    "seismic_activity": {
      "events_24h": 12,
      "max_magnitude": 2.1,
      "swarm_detected": false
    },
    "deformation": {
      "uplift_rate": 0.5,
      "max_displacement": 2.3,
      "trend": "stable"
    },
    "gas_emissions": {
      "so2_flux": 150,
      "co2_flux": 8500,
      "temperature": 85.2
    }
  },
  "ml_analysis": {
    "anomaly_score": 0.23,
    "risk_assessment": "low",
    "precursor_patterns": []
  }
}
```

#### WebSocket `/ws/volcanic/monitoring/{volcano_id}`
Real-time streaming of volcanic monitoring data.

**Message Format:**
```json
{
  "type": "seismic_update",
  "timestamp": "2025-01-09T15:30:00Z",
  "volcano_id": "yellowstone",
  "data": {
    "magnitude": 1.8,
    "depth": 8.5,
    "location": [44.435, -110.592],
    "ml_classification": "volcano-tectonic"
  }
}
```

---

## 🌍 Seismic Analysis API

### Advanced Seismic Processing

#### POST `/seismic/analyze/waveform`
Analyze seismic waveforms using transformer-based ML models.

**Request Body:**
```json
{
  "waveform_data": {
    "stations": ["YNM", "LKWY", "B944"],
    "sampling_rate": 100,
    "duration": 300,
    "data": [...] // Base64 encoded waveform data
  },
  "analysis_options": {
    "phase_picking": true,
    "magnitude_estimation": true,
    "source_mechanism": true,
    "uncertainty_quantification": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "seismic_67890",
  "results": {
    "phase_picks": {
      "p_arrival": "2025-01-09T15:30:12.450Z",
      "s_arrival": "2025-01-09T15:30:18.230Z",
      "confidence": 0.94
    },
    "magnitude": {
      "ml": 3.2,
      "mw": 3.1,
      "uncertainty": 0.2
    },
    "location": {
      "latitude": 44.428,
      "longitude": -110.588,
      "depth": 8.5,
      "uncertainty": {
        "horizontal": 0.8,
        "vertical": 1.2
      }
    },
    "source_mechanism": {
      "strike": 45,
      "dip": 60,
      "rake": -90,
      "moment_tensor": [...]
    }
  }
}
```

### Ambient Noise Analysis

#### GET `/seismic/ambient-noise/{station_id}/velocity-changes`
Get velocity changes from ambient noise analysis.

**Parameters:**
- `station_id`: Seismic station identifier
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)

**Response:**
```json
{
  "station_id": "YNM",
  "time_series": [
    {
      "date": "2025-01-01",
      "velocity_change": -0.02,
      "uncertainty": 0.005,
      "coherence": 0.85
    }
  ],
  "trends": {
    "linear_trend": -0.001,
    "seasonal_component": true,
    "anomalies": [
      {
        "date": "2025-01-05",
        "magnitude": -0.05,
        "significance": 0.99
      }
    ]
  }
}
```

---

## 🛰️ InSAR Processing API

### Advanced InSAR Analysis

#### POST `/insar/process/interferogram`
Process InSAR data with ML-enhanced algorithms.

**Request Body:**
```json
{
  "satellite_data": {
    "mission": "Sentinel-1",
    "track": 64,
    "frame": 431,
    "master_date": "2025-01-01",
    "slave_date": "2025-01-13"
  },
  "processing_options": {
    "algorithm": "SBAS",
    "atmospheric_correction": true,
    "ml_enhancement": true,
    "coherence_threshold": 0.3
  },
  "area_of_interest": {
    "type": "Polygon",
    "coordinates": [[...]]
  }
}
```

**Response:**
```json
{
  "processing_id": "insar_54321",
  "status": "completed",
  "results": {
    "interferogram": {
      "url": "/data/interferograms/insar_54321.tif",
      "coherence_map": "/data/coherence/insar_54321.tif",
      "unwrapped_phase": "/data/unwrapped/insar_54321.tif"
    },
    "deformation_analysis": {
      "max_displacement": 15.2,
      "displacement_rate": 2.3,
      "deformation_pattern": "radial",
      "source_model": {
        "type": "mogi",
        "depth": 4500,
        "volume_change": 1.2e6,
        "confidence": 0.89
      }
    },
    "quality_metrics": {
      "mean_coherence": 0.67,
      "phase_unwrapping_errors": 0.02,
      "atmospheric_correction_quality": 0.85
    }
  }
}
```

### Deformation Time Series

#### GET `/insar/deformation/{location}/time-series`
Get deformation time series for a specific location.

**Parameters:**
- `location`: Coordinates (lat,lon) or location name
- `start_date`: Start date for time series
- `end_date`: End date for time series

**Response:**
```json
{
  "location": {
    "latitude": 44.428,
    "longitude": -110.588,
    "name": "Yellowstone Caldera"
  },
  "time_series": [
    {
      "date": "2025-01-01",
      "los_displacement": 2.3,
      "vertical_displacement": 2.1,
      "east_displacement": 0.5,
      "north_displacement": -0.2,
      "uncertainty": 0.3
    }
  ],
  "analysis": {
    "trend": {
      "rate": 2.1,
      "acceleration": 0.05,
      "r_squared": 0.94
    },
    "anomalies": [],
    "forecast": {
      "next_30_days": 4.2,
      "confidence_interval": [3.8, 4.6]
    }
  }
}
```

---

## 🏔️ Mantle Dynamics API

### 3D Mantle Convection

#### POST `/mantle/convection/simulate`
Run 3D mantle convection simulation.

**Request Body:**
```json
{
  "parameters": {
    "rayleigh_number": 1e6,
    "viscosity_contrast": 1000,
    "temperature_contrast": 2000,
    "boundary_conditions": "free_slip"
  },
  "grid": {
    "resolution": [128, 128, 64],
    "adaptive_refinement": true
  },
  "time_integration": {
    "total_time": 1e9,
    "time_step": 1e6,
    "output_frequency": 1e7
  }
}
```

**Response:**
```json
{
  "simulation_id": "mantle_98765",
  "status": "running",
  "progress": 0.15,
  "estimated_completion": "2025-01-09T18:00:00Z",
  "preliminary_results": {
    "convection_cells": 4,
    "heat_flux": {
      "surface": 0.087,
      "cmb": 0.045
    },
    "velocity_field": {
      "max_velocity": 0.12,
      "rms_velocity": 0.034
    }
  }
}
```

### Plate Evolution

#### GET `/mantle/plates/reconstruction/{age}`
Get plate reconstruction for a specific geological age.

**Parameters:**
- `age`: Geological age in Ma (million years ago)

**Response:**
```json
{
  "age": 100,
  "plates": [
    {
      "plate_id": "pacific",
      "name": "Pacific Plate",
      "boundaries": {
        "type": "Polygon",
        "coordinates": [[...]]
      },
      "motion": {
        "velocity": [5.2, -2.1],
        "angular_velocity": 0.65,
        "euler_pole": [60.2, -85.4]
      }
    }
  ],
  "hotspots": [
    {
      "name": "Hawaii",
      "location": [19.5, -155.5],
      "age": 85,
      "flux": 0.15
    }
  ]
}
```

---

## 💎 Resource Exploration API

### Mineral Prospectivity Mapping

#### POST `/resources/prospectivity/analyze`
Analyze mineral prospectivity using AI-powered models.

**Request Body:**
```json
{
  "area_of_interest": {
    "type": "Polygon",
    "coordinates": [[...]]
  },
  "target_commodities": ["gold", "copper", "lithium"],
  "data_layers": {
    "geological_maps": true,
    "geophysical_grids": true,
    "geochemical_surveys": true,
    "remote_sensing": true
  },
  "analysis_options": {
    "ml_models": ["random_forest", "neural_network", "gradient_boosting"],
    "uncertainty_quantification": true,
    "feature_importance": true
  }
}
```

**Response:**
```json
{
  "analysis_id": "prospect_13579",
  "results": {
    "prospectivity_maps": {
      "gold": "/data/prospectivity/gold_13579.tif",
      "copper": "/data/prospectivity/copper_13579.tif",
      "lithium": "/data/prospectivity/lithium_13579.tif"
    },
    "targets": [
      {
        "target_id": "T001",
        "commodity": "gold",
        "location": [-31.5, 121.2],
        "prospectivity_score": 0.87,
        "confidence": 0.92,
        "area": 2.5,
        "economic_potential": "high"
      }
    ],
    "model_performance": {
      "accuracy": 0.89,
      "precision": 0.85,
      "recall": 0.91,
      "auc_score": 0.93
    }
  }
}
```

### Geochemical Analysis

#### POST `/resources/geochemistry/analyze`
Perform advanced geochemical analysis with ML enhancement.

**Request Body:**
```json
{
  "samples": [
    {
      "sample_id": "GC001",
      "location": [-31.5, 121.2],
      "elements": {
        "SiO2": 65.2,
        "Al2O3": 15.8,
        "Fe2O3": 4.2,
        "Au": 0.15,
        "Cu": 125
      }
    }
  ],
  "analysis_type": ["classification", "anomaly_detection", "source_discrimination"],
  "reference_database": "global_geochemistry"
}
```

**Response:**
```json
{
  "analysis_id": "geochem_24680",
  "results": {
    "sample_classifications": [
      {
        "sample_id": "GC001",
        "rock_type": "granodiorite",
        "confidence": 0.94,
        "tectonic_setting": "volcanic_arc",
        "alteration_type": "phyllic",
        "anomaly_score": 0.78
      }
    ],
    "geochemical_signatures": {
      "pathfinder_elements": ["As", "Sb", "Hg"],
      "metal_associations": ["Au-Cu-Mo"],
      "alteration_minerals": ["sericite", "pyrite", "quartz"]
    },
    "exploration_recommendations": [
      "Follow up with detailed soil sampling",
      "Consider IP geophysical survey",
      "Target depth: 50-150m"
    ]
  }
}
```

---

## 📊 Data Management API

### Data Integration Pipeline

#### POST `/data/integrate`
Integrate multi-source geological data.

**Request Body:**
```json
{
  "data_sources": [
    {
      "type": "seismic_catalog",
      "source": "USGS",
      "format": "QuakeML",
      "url": "https://earthquake.usgs.gov/fdsnws/event/1/query"
    },
    {
      "type": "satellite_imagery",
      "source": "Copernicus",
      "mission": "Sentinel-1",
      "processing_level": "L1"
    }
  ],
  "integration_options": {
    "spatial_interpolation": "kriging",
    "temporal_alignment": true,
    "quality_control": true,
    "feature_engineering": true
  }
}
```

**Response:**
```json
{
  "integration_id": "data_int_11111",
  "status": "processing",
  "progress": 0.45,
  "integrated_datasets": [
    {
      "dataset_id": "seismic_2025_01",
      "records": 1247,
      "quality_score": 0.92,
      "spatial_coverage": "regional",
      "temporal_range": ["2025-01-01", "2025-01-09"]
    }
  ]
}
```

### External Data Connectors

#### GET `/data/external/{source}/available`
List available data from external sources.

**Parameters:**
- `source`: Data source (e.g., "iris", "usgs", "copernicus", "asf")

**Response:**
```json
{
  "source": "iris",
  "available_data": [
    {
      "network": "US",
      "station": "ANMO",
      "location": "00",
      "channel": "BHZ",
      "start_time": "2020-01-01T00:00:00Z",
      "end_time": "2025-01-09T23:59:59Z",
      "sample_rate": 20.0
    }
  ],
  "query_capabilities": {
    "time_range": true,
    "geographic_bounds": true,
    "magnitude_range": true,
    "depth_range": true
  }
}
```

---

## 🔧 System Management API

### Health Monitoring

#### GET `/system/health`
Get overall system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-09T15:30:00Z",
  "services": {
    "api_gateway": {
      "status": "healthy",
      "response_time": 45,
      "cpu_usage": 0.23,
      "memory_usage": 0.67
    },
    "helios_core": {
      "status": "healthy",
      "active_simulations": 3,
      "queue_length": 12,
      "gpu_utilization": 0.78
    },
    "databases": {
      "postgresql": "healthy",
      "influxdb": "healthy",
      "neo4j": "healthy",
      "redis": "healthy"
    }
  },
  "performance_metrics": {
    "requests_per_second": 156,
    "average_response_time": 89,
    "error_rate": 0.002
  }
}
```

### Performance Metrics

#### GET `/system/metrics`
Get detailed performance metrics.

**Response:**
```json
{
  "timestamp": "2025-01-09T15:30:00Z",
  "api_metrics": {
    "total_requests": 1247893,
    "successful_requests": 1244651,
    "error_rate": 0.0026,
    "average_response_time": 89,
    "p95_response_time": 234,
    "p99_response_time": 567
  },
  "resource_usage": {
    "cpu_cores": 32,
    "cpu_usage": 0.67,
    "memory_total": "128GB",
    "memory_used": "89GB",
    "disk_usage": 0.45,
    "network_io": {
      "bytes_in": 1.2e9,
      "bytes_out": 2.3e9
    }
  },
  "ml_model_performance": {
    "active_models": 15,
    "inference_time": 23,
    "model_accuracy": 0.91,
    "training_jobs": 2
  }
}
```

---

## 🔐 Authentication & Authorization

### User Management

#### POST `/auth/register`
Register a new user account.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secure_password",
  "first_name": "John",
  "last_name": "Doe",
  "organization": "University Research",
  "role": "researcher"
}
```

#### POST `/auth/login`
Authenticate user and get access tokens.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 3600,
  "user": {
    "id": "user_12345",
    "email": "user@example.com",
    "role": "researcher",
    "permissions": ["view:data", "run:simulation"]
  }
}
```

### Role-Based Access Control

#### Permissions
- `view:data` - View geological and monitoring data
- `edit:data` - Modify data and configurations
- `run:simulation` - Execute simulations and analysis
- `admin:all` - Full administrative access

#### Protected Endpoints
Most endpoints require specific permissions. Include the JWT token in the Authorization header:

```bash
curl -H "Authorization: Bearer <access_token>" \
     -X GET /api/v1/volcanic/systems
```

---

## 📈 Rate Limiting

### Rate Limits
- **Standard Users**: 1000 requests/hour
- **Premium Users**: 10000 requests/hour
- **Enterprise**: Unlimited

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1641811200
```

---

## 🚨 Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "chambers.depth",
      "issue": "Value must be positive"
    },
    "request_id": "req_67890",
    "timestamp": "2025-01-09T15:30:00Z"
  }
}
```

### Common Error Codes
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error (system error)
- `503` - Service Unavailable (maintenance mode)

---

## 📚 SDK and Libraries

### Python SDK
```python
from prometheus_ai import PrometheusClient

client = PrometheusClient(
    base_url="https://api.prometheus-ai.org",
    api_key="your_api_key"
)

# Simulate volcanic system
result = client.volcanic.simulate_multi_chamber(
    chambers=[...],
    connections=[...],
    parameters={...}
)
```

### JavaScript SDK
```javascript
import { PrometheusAPI } from '@prometheus-ai/sdk';

const client = new PrometheusAPI({
  baseURL: 'https://api.prometheus-ai.org',
  apiKey: 'your_api_key'
});

// Analyze seismic data
const analysis = await client.seismic.analyzeWaveform({
  waveformData: {...},
  analysisOptions: {...}
});
```

---

## 🔄 Webhooks

### Webhook Events
- `simulation.completed` - Simulation finished
- `alert.triggered` - Monitoring alert triggered
- `data.updated` - New data available
- `system.maintenance` - Maintenance scheduled

### Webhook Payload
```json
{
  "event": "simulation.completed",
  "timestamp": "2025-01-09T15:30:00Z",
  "data": {
    "simulation_id": "sim_12345",
    "status": "completed",
    "results_url": "/api/v1/simulations/sim_12345/results"
  }
}
```

---

## 🧪 Geological Validation Testing API (✅ **COMPLETED**)

### Overview
The Geological Validation Testing API provides comprehensive validation of geological algorithms and models using real geological data from peer-reviewed studies and authoritative sources.

### Base URL
```
Testing Framework: python tests/geological_validation/run_validation_tests.py
```

### Validation Suites

#### Run All Validation Tests
```bash
python tests/geological_validation/run_validation_tests.py --suite all
```

**Response:**
```json
{
  "validation_id": "val_20250111_143022",
  "status": "completed",
  "overall_result": "PASS",
  "execution_time": "45.2s",
  "suites": {
    "seismic": {"status": "PASS", "accuracy": 0.92, "tests": 15},
    "volcanic": {"status": "PASS", "accuracy": 0.87, "tests": 12},
    "geochemical": {"status": "PASS", "accuracy": 0.95, "tests": 18},
    "visualization": {"status": "PASS", "accuracy": 0.94, "tests": 10},
    "data_integrity": {"status": "PASS", "reliability": 0.98, "tests": 8},
    "physics": {"status": "PASS", "conservation_compliance": 0.99, "tests": 20}
  }
}
```

#### Seismic Analysis Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite seismic
```

**Validates:**
- Phase picking accuracy using Loma Prieta Earthquake (1989) data
- Magnitude estimation with global earthquake catalogs
- Earthquake location determination accuracy
- Event detection and classification
- Moment tensor inversion validation

**Accuracy Standards:**
- Phase picking: >90% accuracy
- Magnitude error: <0.3 RMS units
- Location accuracy: <5km for regional events

#### Volcanic Prediction Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite volcanic
```

**Validates:**
- Eruption forecasting using Mount St. Helens 1980 eruption data
- VEI (Volcanic Explosivity Index) prediction accuracy
- Precursor pattern recognition and analysis
- Caldera system classification
- Supereruption risk assessment

**Accuracy Standards:**
- Eruption prediction: >75% accuracy
- VEI prediction: <1 unit error
- Precursor detection: >80% sensitivity

#### Geochemical Analysis Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite geochemical
```

**Validates:**
- Major/trace element analysis with certified reference materials
- Rock classification using geochemical compositions
- Thermobarometry calculations validation
- Isotope ratio analysis and interpretation
- Analytical precision and accuracy testing

**Uses Real Data:**
- Yellowstone geochemical samples
- USGS certified reference materials
- Laboratory-verified analytical results

#### 3D Visualization Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite visualization
```

**Validates:**
- Spatial coordinate accuracy against San Andreas Fault 3D model
- Geological unit boundary accuracy
- 3D volume calculation validation
- Rendering fidelity and visual accuracy
- Cross-sectional accuracy testing

#### Data Integrity Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite data_integrity
```

**Validates:**
- Production-scale performance (>1000 records/second)
- Data completeness and quality assessment
- Cross-dataset consistency validation
- Real-time processing capabilities (<100ms response)
- Fault tolerance and recovery testing

#### Physics-Informed Validation
```bash
python tests/geological_validation/run_validation_tests.py --suite physics
```

**Validates:**
- Physics-informed neural network accuracy
- Conservation law compliance testing
- Numerical stability and convergence
- Geological constraint enforcement
- Thermodynamic consistency validation

### Continuous Validation Framework
```bash
python tests/geological_validation/run_validation_tests.py --continuous
```

**Features:**
- Automated validation scheduling
- Research publication monitoring
- Impact assessment of new discoveries
- Performance degradation alerts
- Trend analysis and reporting

### HTML Report Generation
```bash
python tests/geological_validation/run_validation_tests.py --suite all --html-report validation_report.html
```

**Report Includes:**
- Executive summary with pass/fail status
- Detailed test results by validation suite
- Performance trends and analysis
- Recommendations for improvements
- Statistical significance testing results

### Configuration
Edit `tests/geological_validation/config.json` to customize validation parameters:

```json
{
  "validation_thresholds": {
    "seismic": {
      "phase_pick_accuracy": 0.90,
      "magnitude_error_rms": 0.3,
      "location_accuracy_km": 5.0
    },
    "volcanic": {
      "eruption_prediction_accuracy": 0.75,
      "vei_prediction_error": 1.0,
      "precursor_sensitivity": 0.80
    },
    "geochemical": {
      "analytical_accuracy": 0.95,
      "precision_threshold": 0.02
    },
    "visualization": {
      "spatial_accuracy": 0.95,
      "volume_calculation_error": 0.05
    },
    "data_integrity": {
      "throughput_records_per_second": 1000,
      "response_time_ms": 100,
      "completeness_threshold": 0.98
    },
    "physics": {
      "conservation_compliance": 0.99,
      "numerical_stability": 0.95
    }
  }
}
```

### Benchmark Datasets
The validation framework uses real geological datasets:

1. **Loma Prieta Earthquake (1989)** - Seismic analysis validation
2. **Mount St. Helens 1980 Eruption** - Volcanic prediction validation  
3. **Yellowstone Geochemical Samples** - Geochemical analysis validation
4. **San Andreas Fault 3D Model** - Visualization validation
5. **ISC-GEM Global Earthquake Catalog** - Large-scale data validation

### Error Codes
- `VAL_001` - Validation suite not found
- `VAL_002` - Benchmark dataset unavailable
- `VAL_003` - Accuracy threshold not met
- `VAL_004` - Performance requirement failed
- `VAL_005` - Data integrity check failed

---

## 🔒 Security Testing API (✅ **COMPLETED**)

### Security Testing Framework Endpoints

The security testing framework provides comprehensive security validation endpoints for testing authentication, input validation, SSL/TLS configuration, network security, and privacy compliance.

#### POST `/api/v1/security/test/comprehensive`
Run comprehensive security testing suite with OWASP ZAP integration.

**Request:**
```json
{
  "target_url": "https://localhost:3000",
  "test_categories": ["auth", "input_validation", "ssl_tls", "network_security", "privacy_compliance"],
  "zap_config": {
    "proxy_url": "http://127.0.0.1:8080",
    "api_key": "your-zap-api-key"
  },
  "reporting": {
    "generate_html": true,
    "include_screenshots": true,
    "detailed_findings": true
  }
}
```

**Response:**
```json
{
  "test_id": "sec_test_20240115_103000",
  "status": "completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "target_url": "https://localhost:3000",
  "summary": {
    "overall_score": 92,
    "risk_level": "low",
    "critical_issues": 0,
    "high_issues": 1,
    "medium_issues": 3,
    "low_issues": 5
  },
  "tests": {
    "auth_testing": {
      "status": "completed",
      "score": 95,
      "brute_force_protection": true,
      "jwt_validation": true,
      "session_management": true
    },
    "input_validation": {
      "status": "completed",
      "score": 90,
      "xss_protection": true,
      "sql_injection_protection": true,
      "file_upload_security": true
    },
    "ssl_tls": {
      "status": "completed",
      "score": 98,
      "certificate_valid": true,
      "strong_ciphers": true,
      "hsts_enabled": true
    },
    "network_security": {
      "status": "completed",
      "score": 85,
      "open_ports": 3,
      "services_found": 2,
      "firewall_effective": true
    },
    "privacy_compliance": {
      "status": "completed",
      "score": 92,
      "cookie_compliance": true,
      "gdpr_endpoints": 4,
      "privacy_policy_found": true
    }
  },
  "reports": {
    "json_report": "/api/v1/security/reports/sec_test_20240115_103000.json",
    "html_dashboard": "/api/v1/security/reports/sec_test_20240115_103000.html"
  }
}
```

#### GET `/api/v1/security/test/{test_id}`
Retrieve security test results by test ID.

**Response:**
```json
{
  "test_id": "sec_test_20240115_103000",
  "status": "completed",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:45:00Z",
  "duration_seconds": 900,
  "results": {
    "summary": { "overall_score": 92, "risk_level": "low" },
    "detailed_findings": [
      {
        "category": "ssl_tls",
        "finding": "Weak cipher suite detected",
        "severity": "medium",
        "cvss_score": 5.3,
        "remediation": "Disable weak cipher suites in server configuration"
      }
    ]
  }
}
```

#### POST `/api/v1/security/test/auth`
Run authentication and authorization security tests.

**Request:**
```json
{
  "target_url": "https://localhost:3000",
  "test_credentials": {
    "username": "testuser",
    "password": "testpassword"
  },
  "test_scenarios": ["brute_force", "jwt_validation", "session_management", "rbac"]
}
```

#### POST `/api/v1/security/test/input-validation`
Run input validation and sanitization tests.

**Request:**
```json
{
  "target_url": "https://localhost:3000",
  "test_payloads": {
    "xss": ["<script>alert(1)</script>", "javascript:alert(1)"],
    "sql_injection": ["'; DROP TABLE users; --", "' OR '1'='1"],
    "file_upload": ["malicious.php", "../../etc/passwd"]
  }
}
```

#### POST `/api/v1/security/test/privacy-compliance`
Run GDPR and privacy compliance tests.

**Request:**
```json
{
  "target_url": "https://localhost:3000",
  "compliance_standards": ["gdpr", "ccpa"],
  "test_categories": ["cookie_compliance", "data_subject_rights", "consent_mechanisms", "data_collection_audit"]
}
```

#### GET `/api/v1/security/reports/{report_id}`
Download security test reports in various formats.

**Query Parameters:**
- `format`: `json`, `html`, `pdf`, `csv`
- `include_screenshots`: `true`, `false`
- `detailed_findings`: `true`, `false`

### Privacy and GDPR Compliance Endpoints

#### GET `/api/v1/privacy/policy`
Retrieve privacy policy information.

**Response:**
```json
{
  "policy_url": "/privacy-policy",
  "last_updated": "2024-01-01T00:00:00Z",
  "version": "1.2",
  "compliance_standards": ["gdpr", "ccpa"],
  "contact_email": "privacy@prometheus-ai.com"
}
```

#### GET `/api/v1/privacy/data-subject-rights`
List available data subject rights and endpoints.

**Response:**
```json
{
  "rights": [
    {
      "right": "access",
      "endpoint": "/api/v1/privacy/access",
      "description": "Right to access personal data"
    },
    {
      "right": "rectification",
      "endpoint": "/api/v1/privacy/rectification",
      "description": "Right to rectify inaccurate personal data"
    },
    {
      "right": "erasure",
      "endpoint": "/api/v1/privacy/erasure",
      "description": "Right to erasure (deletion) of personal data"
    },
    {
      "right": "portability",
      "endpoint": "/api/v1/privacy/portability",
      "description": "Right to data portability"
    },
    {
      "right": "object",
      "endpoint": "/api/v1/privacy/object",
      "description": "Right to object to processing"
    }
  ]
}
```

#### POST `/api/v1/privacy/consent`
Manage user consent for data processing.

**Request:**
```json
{
  "user_id": "user123",
  "consent_type": "data_processing",
  "consent_given": true,
  "consent_timestamp": "2024-01-15T10:30:00Z",
  "consent_method": "explicit"
}
```

### Security Monitoring Endpoints

#### GET `/api/v1/security/metrics`
Retrieve security metrics and monitoring data.

**Response:**
```json
{
  "security_score": 92,
  "last_scan": "2024-01-15T10:30:00Z",
  "vulnerabilities": {
    "critical": 0,
    "high": 1,
    "medium": 3,
    "low": 5
  },
  "compliance_status": {
    "gdpr": "compliant",
    "iso27001": "compliant",
    "soc2": "in_progress"
  },
  "recent_incidents": 0,
  "security_events": 15
}
```

#### GET `/api/v1/security/alerts`
Retrieve security alerts and notifications.

**Response:**
```json
{
  "alerts": [
    {
      "id": "alert_001",
      "severity": "medium",
      "type": "vulnerability_detected",
      "message": "Weak cipher suite detected in SSL configuration",
      "timestamp": "2024-01-15T10:30:00Z",
      "status": "open",
      "remediation": "Update SSL configuration to disable weak ciphers"
    }
  ],
  "total_alerts": 1,
  "unresolved_alerts": 1
}
```

### Error Codes

Security testing specific error codes:

- `SEC_001` - Security test configuration invalid
- `SEC_002` - OWASP ZAP connection failed
- `SEC_003` - Authentication test failed
- `SEC_004` - SSL/TLS test failed
- `SEC_005` - Network security scan failed
- `SEC_006` - Privacy compliance test failed
- `SEC_007` - Security report generation failed
- `SEC_008` - Insufficient permissions for security testing
- `SEC_009` - Security test timeout
- `SEC_010` - Invalid security test parameters

---

## 📞 Support

### API Support
- **Documentation**: https://docs.prometheus-ai.org
- **Status Page**: https://status.prometheus-ai.org
- **Support Email**: api-support@prometheus-ai.org
- **Community Forum**: https://community.prometheus-ai.org

### Rate Limit Increases
Contact support for rate limit increases or enterprise access.

---

*Last Updated: January 10, 2025*
*API Version: v1.0*

---

## 🔄 Current Development Status

**Task 1: Core Module Integration Framework** is currently **In Progress** with active development focusing on:

- **Module Integration**: Actively refining unified module loader and dependency management system
- **Microservices Development**: Active transition from monolithic Flask API to distributed services architecture
- **API Standardization**: Ensuring consistent API patterns across all 100+ geological modules
- **Performance Optimization**: Enhancing response times and resource utilization

**Task 2: Database Schema Integration and Advanced Data Management** is currently **In Progress** with critical priority development focusing on:

- **Comprehensive Geological Database Schema**: Implementing 50+ specialized geological tables including rock mechanical properties, volcanic chambers, and critical failure zones
- **Model Database System**: Building ML model lifecycle management with versioning, validation, and multi-backend storage
- **Advanced Data Integration Pipeline (Task 2.3)**: ✅ **COMPLETED** - Multi-source geological data processing with automated feature engineering, real-time volcanic data enhancement, comprehensive data quality assessment, and anomaly detection systems
- **Database API Endpoints**: Developing RESTful APIs for geological database management and model operations

The API endpoints documented above represent the target architecture. Current implementation uses a Flask-based monolithic API in `backend/api.py` with active migration to the microservices architecture and enhanced database integration described in this documentation.