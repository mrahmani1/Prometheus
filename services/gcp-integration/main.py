#!/usr/bin/env python3
"""
Google Cloud Platform Data Analytics Integration Service
Comprehensive GCP data analytics and BigQuery integration for geological data processing
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from core.bigquery_integration import BigQueryGeologicalAnalytics
from core.dataflow_processor import DataflowGeologicalProcessor
from core.dataproc_analytics import DataprocSparkAnalytics
from core.composer_orchestrator import ComposerWorkflowOrchestrator
from core.data_fusion_pipeline import DataFusionPipelineManager
from core.looker_visualization import LookerGeologicalBI
from core.data_catalog_manager import DataCatalogGeologicalManager
from core.dataplex_manager import DataplexUnifiedManager
from core.earth_engine_integration import EarthEngineGeologicalAnalysis
from models.gcp_models import *
from api.health import router as health_router
from api.bigquery import router as bigquery_router
from api.dataflow import router as dataflow_router
from api.dataproc import router as dataproc_router
from api.composer import router as composer_router
from api.data_fusion import router as data_fusion_router
from api.looker import router as looker_router
from api.data_catalog import router as data_catalog_router
from api.dataplex import router as dataplex_router
from api.earth_engine import router as earth_engine_router

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

logger = structlog.get_logger(__name__)

# Global service instances
bigquery_service: Optional[BigQueryGeologicalAnalytics] = None
dataflow_service: Optional[DataflowGeologicalProcessor] = None
dataproc_service: Optional[DataprocSparkAnalytics] = None
composer_service: Optional[ComposerWorkflowOrchestrator] = None
data_fusion_service: Optional[DataFusionPipelineManager] = None
looker_service: Optional[LookerGeologicalBI] = None
data_catalog_service: Optional[DataCatalogGeologicalManager] = None
dataplex_service: Optional[DataplexUnifiedManager] = None
earth_engine_service: Optional[EarthEngineGeologicalAnalysis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global (bigquery_service, dataflow_service, dataproc_service, composer_service,
            data_fusion_service, looker_service, data_catalog_service, dataplex_service,
            earth_engine_service)
    
    logger.info("Starting GCP Data Analytics Integration Service")
    
    try:
        # Initialize all GCP services
        bigquery_service = BigQueryGeologicalAnalytics()
        await bigquery_service.initialize()
        
        dataflow_service = DataflowGeologicalProcessor()
        await dataflow_service.initialize()
        
        dataproc_service = DataprocSparkAnalytics()
        await dataproc_service.initialize()
        
        composer_service = ComposerWorkflowOrchestrator()
        await composer_service.initialize()
        
        data_fusion_service = DataFusionPipelineManager()
        await data_fusion_service.initialize()
        
        looker_service = LookerGeologicalBI()
        await looker_service.initialize()
        
        data_catalog_service = DataCatalogGeologicalManager()
        await data_catalog_service.initialize()
        
        dataplex_service = DataplexUnifiedManager()
        await dataplex_service.initialize()
        
        earth_engine_service = EarthEngineGeologicalAnalysis()
        await earth_engine_service.initialize()
        
        logger.info("All GCP services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to initialize GCP services", error=str(e))
        raise
    finally:
        logger.info("Shutting down GCP Data Analytics Integration Service")
        
        # Cleanup services
        services = [
            bigquery_service, dataflow_service, dataproc_service, composer_service,
            data_fusion_service, looker_service, data_catalog_service, 
            dataplex_service, earth_engine_service
        ]
        
        for service in services:
            if service:
                try:
                    await service.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up service {service.__class__.__name__}", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="GCP Data Analytics Integration Service",
    description="Comprehensive Google Cloud Platform data analytics and BigQuery integration for geological data processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(bigquery_router, prefix="/api/v1/bigquery", tags=["bigquery"])
app.include_router(dataflow_router, prefix="/api/v1/dataflow", tags=["dataflow"])
app.include_router(dataproc_router, prefix="/api/v1/dataproc", tags=["dataproc"])
app.include_router(composer_router, prefix="/api/v1/composer", tags=["composer"])
app.include_router(data_fusion_router, prefix="/api/v1/data-fusion", tags=["data-fusion"])
app.include_router(looker_router, prefix="/api/v1/looker", tags=["looker"])
app.include_router(data_catalog_router, prefix="/api/v1/data-catalog", tags=["data-catalog"])
app.include_router(dataplex_router, prefix="/api/v1/dataplex", tags=["dataplex"])
app.include_router(earth_engine_router, prefix="/api/v1/earth-engine", tags=["earth-engine"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "GCP Data Analytics Integration Service",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "capabilities": [
            "BigQuery Geological Analytics",
            "BigQuery ML for Geological Data",
            "Dataflow Real-time Processing",
            "Dataproc Spark Analytics",
            "Composer Workflow Orchestration",
            "Data Fusion Visual Pipelines",
            "Looker Geological BI",
            "Data Catalog Management",
            "Dataplex Unified Data Management",
            "Google Earth Engine Integration"
        ]
    }


@app.get("/api/v1/status")
async def get_service_status():
    """Get comprehensive service status"""
    try:
        status = {
            "service": "GCP Data Analytics Integration",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {}
        }
        
        # Check all service statuses
        services = {
            "bigquery": bigquery_service,
            "dataflow": dataflow_service,
            "dataproc": dataproc_service,
            "composer": composer_service,
            "data_fusion": data_fusion_service,
            "looker": looker_service,
            "data_catalog": data_catalog_service,
            "dataplex": dataplex_service,
            "earth_engine": earth_engine_service
        }
        
        for name, service in services.items():
            if service:
                try:
                    service_status = await service.get_status()
                    status["services"][name] = service_status
                except Exception as e:
                    status["services"][name] = {"status": "error", "error": str(e)}
            else:
                status["services"][name] = {"status": "not_initialized"}
        
        return status
        
    except Exception as e:
        logger.error("Error getting service status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting service status: {str(e)}")


@app.post("/api/v1/analyze/geological-dataset")
async def analyze_geological_dataset(
    request: GeologicalAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Comprehensive geological dataset analysis using all GCP services"""
    try:
        logger.info("Starting comprehensive geological dataset analysis", 
                   dataset_id=request.dataset_id)
        
        # Create analysis job
        analysis_job = GeologicalAnalysisJob(
            job_id=f"analysis_{request.dataset_id}_{int(datetime.utcnow().timestamp())}",
            dataset_id=request.dataset_id,
            analysis_type=request.analysis_type,
            parameters=request.parameters,
            status="started",
            created_at=datetime.utcnow()
        )
        
        # Start background analysis
        background_tasks.add_task(
            run_comprehensive_analysis,
            analysis_job
        )
        
        return {
            "job_id": analysis_job.job_id,
            "status": "started",
            "message": "Comprehensive geological analysis started",
            "estimated_completion": (datetime.utcnow() + timedelta(hours=2)).isoformat()
        }
        
    except Exception as e:
        logger.error("Error starting geological dataset analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


async def run_comprehensive_analysis(analysis_job: GeologicalAnalysisJob):
    """Run comprehensive geological analysis using all GCP services"""
    try:
        logger.info("Running comprehensive geological analysis", job_id=analysis_job.job_id)
        
        # 1. BigQuery analysis
        if bigquery_service:
            bigquery_results = await bigquery_service.analyze_geological_dataset(
                analysis_job.dataset_id,
                analysis_job.analysis_type,
                analysis_job.parameters
            )
            analysis_job.results["bigquery"] = bigquery_results
        
        # 2. Dataflow processing
        if dataflow_service:
            dataflow_results = await dataflow_service.process_geological_stream(
                analysis_job.dataset_id,
                analysis_job.parameters
            )
            analysis_job.results["dataflow"] = dataflow_results
        
        # 3. Dataproc Spark analytics
        if dataproc_service:
            spark_results = await dataproc_service.run_geological_analytics(
                analysis_job.dataset_id,
                analysis_job.analysis_type
            )
            analysis_job.results["dataproc"] = spark_results
        
        # 4. Earth Engine analysis
        if earth_engine_service:
            ee_results = await earth_engine_service.analyze_satellite_data(
                analysis_job.dataset_id,
                analysis_job.parameters.get("region", {})
            )
            analysis_job.results["earth_engine"] = ee_results
        
        # 5. Create Looker dashboard
        if looker_service:
            dashboard_results = await looker_service.create_geological_dashboard(
                analysis_job.job_id,
                analysis_job.results
            )
            analysis_job.results["looker"] = dashboard_results
        
        analysis_job.status = "completed"
        analysis_job.completed_at = datetime.utcnow()
        
        logger.info("Comprehensive geological analysis completed", job_id=analysis_job.job_id)
        
    except Exception as e:
        logger.error("Error in comprehensive geological analysis", 
                    job_id=analysis_job.job_id, error=str(e))
        analysis_job.status = "failed"
        analysis_job.error = str(e)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("ENVIRONMENT") == "development",
        log_config=None  # Use structlog configuration
    )