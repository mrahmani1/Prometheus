import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import consul
import redis.asyncio as redis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import json

# Add backend modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

from connectors.iris_connector import IRISConnector
from connectors.copernicus_connector import CopernicusConnector
from connectors.asf_connector import ASFConnector
from connectors.usgs_connector import USGSConnector
from connectors.nasa_connector import NASAConnector
from processors.data_validator import DataValidator
from processors.quality_controller import QualityController
from processors.data_transformer import DataTransformer

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Configuration
CONSUL_HOST = os.getenv('CONSUL_HOST', 'localhost')
CONSUL_PORT = int(os.getenv('CONSUL_PORT', '8500'))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
SERVICE_NAME = 'prometheus-data-ingestion'
SERVICE_PORT = int(os.getenv('PORT', '8002'))
SERVICE_HOST = os.getenv('SERVICE_HOST', 'localhost')

# Global instances
consul_client = None
redis_client = None
kafka_producer = None
data_connectors = {}
data_processors = {}

# Pydantic models
class DataIngestionRequest(BaseModel):
    source: str = Field(..., description="Data source identifier")
    data_type: str = Field(..., description="Type of data to ingest")
    spatial_filter: Optional[Dict[str, float]] = Field(None, description="Spatial filtering parameters")
    temporal_filter: Optional[Dict[str, str]] = Field(None, description="Temporal filtering parameters")
    quality_requirements: Optional[Dict[str, Any]] = Field(None, description="Data quality requirements")
    processing_options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")

class StreamingConfig(BaseModel):
    topic: str = Field(..., description="Kafka topic name")
    batch_size: int = Field(default=100, description="Batch size for streaming")
    interval_seconds: int = Field(default=60, description="Streaming interval in seconds")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Data filters")

class DataValidationRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data to validate")
    validation_rules: List[str] = Field(..., description="Validation rules to apply")
    strict_mode: bool = Field(default=False, description="Enable strict validation mode")

class QualityControlRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="Data for quality control")
    quality_metrics: List[str] = Field(..., description="Quality metrics to compute")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Quality thresholds")

class IngestionResponse(BaseModel):
    status: str
    message: str
    data_id: Optional[str] = None
    records_processed: int = 0
    quality_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global consul_client, redis_client, kafka_producer, data_connectors, data_processors
    
    try:
        # Initialize external connections
        consul_client = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)
        redis_client = redis.from_url(REDIS_URL)
        
        # Initialize Kafka producer
        kafka_producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        await kafka_producer.start()
        
        # Test connections
        await redis_client.ping()
        consul_client.agent.self()
        
        logger.info("External connections established")
        
        # Initialize data connectors
        data_connectors = {
            'iris': IRISConnector(),
            'copernicus': CopernicusConnector(),
            'asf': ASFConnector(),
            'usgs': USGSConnector(),
            'nasa': NASAConnector()
        }
        
        # Initialize data processors
        data_processors = {
            'validator': DataValidator(),
            'quality_controller': QualityController(),
            'transformer': DataTransformer()
        }
        
        # Initialize all connectors and processors
        for name, connector in data_connectors.items():
            await connector.initialize()
            logger.info(f"Initialized {name} connector")
        
        for name, processor in data_processors.items():
            await processor.initialize()
            logger.info(f"Initialized {name} processor")
        
        logger.info("Data connectors and processors initialized")
        
        # Register service with Consul
        service_id = f"{SERVICE_NAME}-{SERVICE_HOST}-{SERVICE_PORT}"
        consul_client.agent.service.register(
            name=SERVICE_NAME,
            service_id=service_id,
            address=SERVICE_HOST,
            port=SERVICE_PORT,
            check=consul.Check.http(f"http://{SERVICE_HOST}:{SERVICE_PORT}/health", interval="10s"),
            tags=['data', 'ingestion', 'kafka', 'prometheus']
        )
        
        logger.info("Service registered with Consul", service_id=service_id)
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize service", error=str(e))
        raise
    finally:
        # Cleanup
        if kafka_producer:
            await kafka_producer.stop()
        
        if consul_client:
            try:
                consul_client.agent.service.deregister(service_id)
                logger.info("Service deregistered from Consul")
            except Exception as e:
                logger.error("Failed to deregister service", error=str(e))
        
        if redis_client:
            await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Data Ingestion Service",
    description="Multi-source geological data ingestion and processing service",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Dependencies
async def get_correlation_id(x_correlation_id: Optional[str] = Header(None)) -> str:
    return x_correlation_id or f"data-ingestion-{datetime.now().isoformat()}"

async def get_user_context(
    x_user_id: Optional[str] = Header(None),
    x_user_role: Optional[str] = Header(None)
) -> Dict[str, str]:
    return {
        "user_id": x_user_id or "anonymous",
        "user_role": x_user_role or "user"
    }

# Health endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "connectors": list(data_connectors.keys()),
        "processors": list(data_processors.keys())
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check Redis connection
        await redis_client.ping()
        
        # Check Kafka producer
        if not kafka_producer or kafka_producer._closed:
            raise HTTPException(status_code=503, detail="Kafka producer not ready")
        
        # Check connectors
        connector_status = {}
        for name, connector in data_connectors.items():
            status = await connector.get_status()
            connector_status[name] = status
            if status.get('status') != 'ready':
                raise HTTPException(status_code=503, detail=f"Connector {name} not ready")
        
        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "redis": "connected",
                "kafka": "connected",
                "consul": "connected"
            },
            "connectors": connector_status
        }
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

# Data ingestion endpoints
@app.post("/ingest", response_model=IngestionResponse)
async def ingest_data(
    request: DataIngestionRequest,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id),
    user_context: Dict[str, str] = Depends(get_user_context)
):
    """Ingest data from specified source"""
    try:
        logger.info("Data ingestion requested",
                   correlation_id=correlation_id,
                   source=request.source,
                   data_type=request.data_type,
                   user_id=user_context["user_id"])
        
        # Get appropriate connector
        if request.source not in data_connectors:
            raise HTTPException(status_code=400, detail=f"Unknown data source: {request.source}")
        
        connector = data_connectors[request.source]
        
        # Fetch data from source
        raw_data = await connector.fetch_data(
            data_type=request.data_type,
            spatial_filter=request.spatial_filter,
            temporal_filter=request.temporal_filter,
            **request.processing_options
        )
        
        # Validate data
        validator = data_processors['validator']
        validation_result = await validator.validate_data(
            data=raw_data,
            source=request.source,
            data_type=request.data_type
        )
        
        if not validation_result['valid']:
            logger.warning("Data validation failed",
                          correlation_id=correlation_id,
                          errors=validation_result['errors'])
            return IngestionResponse(
                status="validation_failed",
                message="Data validation failed",
                metadata={"validation_errors": validation_result['errors']}
            )
        
        # Quality control
        quality_controller = data_processors['quality_controller']
        quality_result = await quality_controller.assess_quality(
            data=raw_data,
            requirements=request.quality_requirements
        )
        
        # Transform data
        transformer = data_processors['transformer']
        processed_data = await transformer.transform_data(
            data=raw_data,
            target_format="standard_geological",
            processing_options=request.processing_options
        )
        
        # Store in cache
        data_id = f"{request.source}_{request.data_type}_{correlation_id}"
        await redis_client.setex(
            f"ingested_data:{data_id}",
            3600,  # 1 hour TTL
            json.dumps(processed_data, default=str)
        )
        
        # Stream to Kafka
        background_tasks.add_task(
            stream_to_kafka,
            topic=f"geological_data_{request.data_type}",
            data=processed_data,
            metadata={
                "source": request.source,
                "correlation_id": correlation_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info("Data ingestion completed",
                   correlation_id=correlation_id,
                   data_id=data_id,
                   quality_score=quality_result['score'])
        
        return IngestionResponse(
            status="success",
            message="Data ingested successfully",
            data_id=data_id,
            records_processed=len(processed_data) if isinstance(processed_data, list) else 1,
            quality_score=quality_result['score'],
            metadata={
                "source": request.source,
                "processing_time": processed_data.get('processing_time', 0),
                "validation_passed": validation_result['valid']
            }
        )
        
    except Exception as e:
        logger.error("Data ingestion failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/validate", response_model=Dict[str, Any])
async def validate_data(
    request: DataValidationRequest,
    correlation_id: str = Depends(get_correlation_id)
):
    """Validate data against specified rules"""
    try:
        validator = data_processors['validator']
        result = await validator.validate_with_rules(
            data=request.data,
            rules=request.validation_rules,
            strict_mode=request.strict_mode
        )
        
        return {
            "status": "success",
            "validation_result": result,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Data validation failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/quality-control", response_model=Dict[str, Any])
async def quality_control(
    request: QualityControlRequest,
    correlation_id: str = Depends(get_correlation_id)
):
    """Perform quality control assessment"""
    try:
        quality_controller = data_processors['quality_controller']
        result = await quality_controller.detailed_assessment(
            data=request.data,
            metrics=request.quality_metrics,
            thresholds=request.thresholds
        )
        
        return {
            "status": "success",
            "quality_assessment": result,
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Quality control failed",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Quality control failed: {str(e)}")

@app.post("/stream/start")
async def start_streaming(
    config: StreamingConfig,
    background_tasks: BackgroundTasks,
    correlation_id: str = Depends(get_correlation_id)
):
    """Start streaming data to Kafka topic"""
    try:
        # Start background streaming task
        background_tasks.add_task(
            continuous_streaming,
            config=config,
            correlation_id=correlation_id
        )
        
        return {
            "status": "success",
            "message": f"Streaming started to topic {config.topic}",
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Failed to start streaming",
                    correlation_id=correlation_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")

@app.get("/sources")
async def list_sources():
    """List available data sources"""
    sources = {}
    for name, connector in data_connectors.items():
        status = await connector.get_status()
        sources[name] = {
            "status": status.get('status', 'unknown'),
            "capabilities": status.get('capabilities', []),
            "data_types": status.get('supported_data_types', [])
        }
    
    return {"sources": sources}

# Background tasks
async def stream_to_kafka(topic: str, data: Dict[str, Any], metadata: Dict[str, Any]):
    """Stream data to Kafka topic"""
    try:
        message = {
            "data": data,
            "metadata": metadata
        }
        
        await kafka_producer.send(topic, message)
        logger.info("Data streamed to Kafka", topic=topic, metadata=metadata)
        
    except Exception as e:
        logger.error("Failed to stream to Kafka", topic=topic, error=str(e))

async def continuous_streaming(config: StreamingConfig, correlation_id: str):
    """Continuous streaming background task"""
    try:
        logger.info("Starting continuous streaming",
                   topic=config.topic,
                   interval=config.interval_seconds,
                   correlation_id=correlation_id)
        
        while True:
            # This would implement actual continuous data fetching
            # For now, just log the activity
            logger.info("Streaming batch processed",
                       topic=config.topic,
                       correlation_id=correlation_id)
            
            await asyncio.sleep(config.interval_seconds)
            
    except Exception as e:
        logger.error("Continuous streaming failed",
                    correlation_id=correlation_id,
                    error=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_config=None
    )