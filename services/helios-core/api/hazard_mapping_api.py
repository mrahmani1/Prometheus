"""
Hazard Mapping API Service

This module provides REST API endpoints for the comprehensive hazard mapping system,
including multi-hazard risk assessment, real-time updates, and communication products.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
import json

from core.hazard_mapping import (
    HazardMappingEngine, GeographicBounds, HazardType, RiskLevel,
    MultiHazardAssessment
)
from core.realtime_hazard_updates import (
    RealTimeHazardUpdater, MonitoringDataStream, HazardAlert
)
from core.hazard_communication import (
    HazardCommunicationEngine, AudienceType, CommunicationFormat,
    CommunicationProduct
)

logger = logging.getLogger(__name__)

# Initialize components
hazard_engine = HazardMappingEngine()
realtime_updater = None  # Will be initialized on startup
communication_engine = HazardCommunicationEngine()

router = APIRouter(prefix="/api/v1/hazard-mapping", tags=["hazard-mapping"])


# Pydantic models for API requests/responses
class GeographicBoundsModel(BaseModel):
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    resolution: float = Field(0.01, gt=0, le=1, description="Grid resolution in degrees")


class MonitoringDataModel(BaseModel):
    stream_id: str
    data_type: str
    timestamp: datetime
    location: Dict[str, float]  # lat, lon, elevation
    value: float
    unit: str
    quality_flag: str = Field(..., regex="^(good|questionable|bad)$")
    sensor_id: str


class HazardAssessmentRequest(BaseModel):
    geographic_bounds: GeographicBoundsModel
    monitoring_data: Optional[Dict[str, Any]] = None
    time_horizon_days: int = Field(30, ge=1, le=365, description="Assessment time horizon in days")
    include_uncertainty: bool = True
    include_cascading_effects: bool = True


class CommunicationRequest(BaseModel):
    assessment_id: str
    audience_type: str = Field(..., regex="^(scientific|emergency_management|general_public|media|government|industry)$")
    format_type: str = Field(..., regex="^(technical_report|executive_summary|public_advisory|emergency_alert|infographic|interactive_map|social_media)$")
    custom_config: Optional[Dict[str, Any]] = None


class HazardAssessmentResponse(BaseModel):
    assessment_id: str
    timestamp: datetime
    geographic_bounds: GeographicBoundsModel
    individual_hazards: Dict[str, Dict[str, Any]]
    combined_risk_summary: Dict[str, float]
    uncertainty_summary: Dict[str, float]
    cascading_effects: Dict[str, List[str]]
    population_at_risk: int
    economic_impact_estimate: float
    status: str


class AlertResponse(BaseModel):
    alert_id: str
    timestamp: datetime
    hazard_type: str
    risk_level: str
    affected_area: GeographicBoundsModel
    population_at_risk: int
    recommended_actions: List[str]
    confidence: float
    expires_at: datetime
    escalation_level: int


async def get_hazard_engine():
    """Dependency to get initialized hazard engine"""
    if not hasattr(hazard_engine, '_initialized'):
        await hazard_engine.initialize_hazard_models()
        hazard_engine._initialized = True
    return hazard_engine


async def get_realtime_updater():
    """Dependency to get initialized real-time updater"""
    global realtime_updater
    if realtime_updater is None:
        engine = await get_hazard_engine()
        realtime_updater = RealTimeHazardUpdater(engine)
        # Start monitoring in background
        asyncio.create_task(realtime_updater.start_monitoring())
    return realtime_updater


@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing hazard mapping services...")
    await get_hazard_engine()
    await get_realtime_updater()
    logger.info("Hazard mapping services initialized successfully")


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down hazard mapping services...")
    if realtime_updater:
        await realtime_updater.stop_monitoring()
    logger.info("Hazard mapping services shut down")


@router.post("/assess", response_model=HazardAssessmentResponse)
async def assess_multi_hazard_risk(
    request: HazardAssessmentRequest,
    background_tasks: BackgroundTasks,
    engine: HazardMappingEngine = Depends(get_hazard_engine)
):
    """
    Perform comprehensive multi-hazard risk assessment
    
    This endpoint analyzes multiple geological hazards simultaneously,
    providing probabilistic risk estimates with uncertainty quantification.
    """
    try:
        logger.info(f"Starting multi-hazard assessment for bounds: {request.geographic_bounds}")
        
        # Convert Pydantic model to internal format
        bounds = GeographicBounds(
            min_lat=request.geographic_bounds.min_lat,
            max_lat=request.geographic_bounds.max_lat,
            min_lon=request.geographic_bounds.min_lon,
            max_lon=request.geographic_bounds.max_lon,
            resolution=request.geographic_bounds.resolution
        )
        
        # Prepare monitoring data
        monitoring_data = request.monitoring_data or {}
        
        # Perform assessment
        assessment = await engine.assess_multi_hazard_risk(
            geographic_bounds=bounds,
            monitoring_data=monitoring_data,
            time_horizon=timedelta(days=request.time_horizon_days)
        )
        
        # Convert to response format
        response = HazardAssessmentResponse(
            assessment_id=assessment.assessment_id,
            timestamp=assessment.timestamp,
            geographic_bounds=GeographicBoundsModel(
                min_lat=assessment.geographic_bounds.min_lat,
                max_lat=assessment.geographic_bounds.max_lat,
                min_lon=assessment.geographic_bounds.min_lon,
                max_lon=assessment.geographic_bounds.max_lon,
                resolution=assessment.geographic_bounds.resolution
            ),
            individual_hazards={
                hazard_type.value: {
                    "probability": hazard_prob.probability,
                    "confidence_interval": hazard_prob.confidence_interval,
                    "uncertainty": hazard_prob.uncertainty,
                    "contributing_factors": hazard_prob.contributing_factors
                }
                for hazard_type, hazard_prob in assessment.individual_hazards.items()
            },
            combined_risk_summary={
                "max_risk": float(assessment.combined_risk_map.max()),
                "mean_risk": float(assessment.combined_risk_map.mean()),
                "area_above_moderate_risk": int((assessment.combined_risk_map > 0.3).sum())
            },
            uncertainty_summary={
                "max_uncertainty": float(assessment.uncertainty_map.max()),
                "mean_uncertainty": float(assessment.uncertainty_map.mean())
            },
            cascading_effects={
                primary.value: [secondary.value for secondary in secondaries]
                for primary, secondaries in assessment.cascading_effects.items()
            },
            population_at_risk=assessment.total_population_at_risk,
            economic_impact_estimate=assessment.economic_impact_estimate,
            status="completed"
        )
        
        logger.info(f"Multi-hazard assessment completed: {assessment.assessment_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in multi-hazard assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@router.get("/assessments/{assessment_id}")
async def get_assessment(
    assessment_id: str,
    updater: RealTimeHazardUpdater = Depends(get_realtime_updater)
):
    """Get details of a specific hazard assessment"""
    try:
        assessments = updater.get_active_assessments()
        
        if assessment_id not in assessments:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = assessments[assessment_id]
        
        return {
            "assessment_id": assessment.assessment_id,
            "timestamp": assessment.timestamp,
            "geographic_bounds": {
                "min_lat": assessment.geographic_bounds.min_lat,
                "max_lat": assessment.geographic_bounds.max_lat,
                "min_lon": assessment.geographic_bounds.min_lon,
                "max_lon": assessment.geographic_bounds.max_lon,
                "resolution": assessment.geographic_bounds.resolution
            },
            "combined_risk_map": assessment.combined_risk_map.tolist(),
            "uncertainty_map": assessment.uncertainty_map.tolist(),
            "individual_hazards": {
                hazard_type.value: {
                    "probability": hazard_prob.probability,
                    "confidence_interval": hazard_prob.confidence_interval,
                    "uncertainty": hazard_prob.uncertainty,
                    "time_window_days": hazard_prob.time_window.days,
                    "contributing_factors": hazard_prob.contributing_factors
                }
                for hazard_type, hazard_prob in assessment.individual_hazards.items()
            },
            "cascading_effects": {
                primary.value: [secondary.value for secondary in secondaries]
                for primary, secondaries in assessment.cascading_effects.items()
            },
            "population_at_risk": assessment.total_population_at_risk,
            "economic_impact_estimate": assessment.economic_impact_estimate
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving assessment {assessment_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve assessment: {str(e)}")


@router.post("/monitoring-data")
async def ingest_monitoring_data(
    data: MonitoringDataModel,
    updater: RealTimeHazardUpdater = Depends(get_realtime_updater)
):
    """
    Ingest real-time monitoring data
    
    This endpoint accepts monitoring data from various sensors and triggers
    hazard map updates when significant changes are detected.
    """
    try:
        # Convert to internal format
        data_stream = MonitoringDataStream(
            stream_id=data.stream_id,
            data_type=data.data_type,
            timestamp=data.timestamp,
            location=data.location,
            value=data.value,
            unit=data.unit,
            quality_flag=data.quality_flag,
            sensor_id=data.sensor_id
        )
        
        # Ingest data and check for triggers
        await updater.ingest_monitoring_data(data_stream)
        
        return {
            "status": "success",
            "message": "Monitoring data ingested successfully",
            "stream_id": data.stream_id,
            "timestamp": data.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error ingesting monitoring data: {e}")
        raise HTTPException(status_code=500, detail=f"Data ingestion failed: {str(e)}")


@router.get("/monitoring-status")
async def get_monitoring_status(
    updater: RealTimeHazardUpdater = Depends(get_realtime_updater)
):
    """Get current monitoring system status"""
    try:
        active_assessments = updater.get_active_assessments()
        monitoring_summary = updater.get_monitoring_data_summary()
        
        return {
            "status": "operational" if updater.is_running else "stopped",
            "active_assessments": len(active_assessments),
            "monitoring_streams": len(monitoring_summary),
            "data_streams": monitoring_summary,
            "last_update": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting monitoring status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(
    risk_level: Optional[str] = Query(None, regex="^(VERY_LOW|LOW|MODERATE|HIGH|VERY_HIGH|EXTREME)$"),
    hazard_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """Get active hazard alerts"""
    try:
        # In a real implementation, this would query a database of active alerts
        # For now, return empty list as placeholder
        alerts = []
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve alerts: {str(e)}")


@router.post("/communication")
async def generate_communication_product(
    request: CommunicationRequest,
    updater: RealTimeHazardUpdater = Depends(get_realtime_updater)
):
    """
    Generate hazard communication product for specific audience
    
    This endpoint creates tailored communication products (reports, advisories,
    alerts) for different audiences including scientists, emergency managers,
    and the general public.
    """
    try:
        # Get assessment
        assessments = updater.get_active_assessments()
        if request.assessment_id not in assessments:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        assessment = assessments[request.assessment_id]
        
        # Convert string enums
        try:
            audience_type = AudienceType(request.audience_type)
            format_type = CommunicationFormat(request.format_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid audience or format type: {e}")
        
        # Generate communication product
        product = await communication_engine.generate_communication_product(
            assessment=assessment,
            audience_type=audience_type,
            format_type=format_type,
            alert=None,  # Could be enhanced to include active alerts
            custom_config=request.custom_config
        )
        
        return {
            "product_id": product.product_id,
            "timestamp": product.timestamp,
            "audience_type": product.audience_type.value,
            "format_type": product.format_type.value,
            "title": product.title,
            "content": product.content,
            "visual_elements": product.visual_elements,
            "urgency_level": product.urgency_level,
            "distribution_channels": product.distribution_channels,
            "language": product.language
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating communication product: {e}")
        raise HTTPException(status_code=500, detail=f"Communication generation failed: {str(e)}")


@router.get("/hazard-types")
async def get_supported_hazard_types():
    """Get list of supported hazard types"""
    return {
        "hazard_types": [
            {
                "type": hazard_type.value,
                "name": hazard_type.value.replace('_', ' ').title(),
                "description": _get_hazard_description(hazard_type)
            }
            for hazard_type in HazardType
        ]
    }


@router.get("/risk-levels")
async def get_risk_levels():
    """Get list of risk levels"""
    return {
        "risk_levels": [
            {
                "level": risk_level.value,
                "name": risk_level.name,
                "description": _get_risk_level_description(risk_level)
            }
            for risk_level in RiskLevel
        ]
    }


@router.get("/audience-types")
async def get_audience_types():
    """Get list of supported audience types for communication"""
    return {
        "audience_types": [
            {
                "type": audience_type.value,
                "name": audience_type.value.replace('_', ' ').title(),
                "description": _get_audience_description(audience_type)
            }
            for audience_type in AudienceType
        ]
    }


@router.get("/communication-formats")
async def get_communication_formats():
    """Get list of supported communication formats"""
    return {
        "communication_formats": [
            {
                "format": format_type.value,
                "name": format_type.value.replace('_', ' ').title(),
                "description": _get_format_description(format_type)
            }
            for format_type in CommunicationFormat
        ]
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check if services are initialized
        engine_status = hasattr(hazard_engine, '_initialized')
        updater_status = realtime_updater is not None and realtime_updater.is_running
        
        return {
            "status": "healthy" if engine_status and updater_status else "degraded",
            "timestamp": datetime.now(),
            "services": {
                "hazard_engine": "initialized" if engine_status else "not_initialized",
                "realtime_updater": "running" if updater_status else "stopped",
                "communication_engine": "ready"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(),
                "error": str(e)
            }
        )


def _get_hazard_description(hazard_type: HazardType) -> str:
    """Get description for hazard type"""
    descriptions = {
        HazardType.VOLCANIC_ERUPTION: "Explosive or effusive volcanic activity",
        HazardType.SEISMIC_EARTHQUAKE: "Ground shaking from tectonic activity",
        HazardType.LANDSLIDE: "Mass movement of rock, debris, or earth",
        HazardType.TSUNAMI: "Ocean waves generated by seismic activity",
        HazardType.PYROCLASTIC_FLOW: "Fast-moving volcanic gas and rock flows",
        HazardType.LAHAR: "Volcanic mudflows and debris flows",
        HazardType.ASH_FALL: "Airborne volcanic ash deposition",
        HazardType.LAVA_FLOW: "Molten rock flows from volcanic vents",
        HazardType.GROUND_DEFORMATION: "Surface displacement from subsurface processes",
        HazardType.LIQUEFACTION: "Soil behavior change during seismic shaking"
    }
    return descriptions.get(hazard_type, "Geological hazard")


def _get_risk_level_description(risk_level: RiskLevel) -> str:
    """Get description for risk level"""
    descriptions = {
        RiskLevel.VERY_LOW: "Minimal risk, normal conditions",
        RiskLevel.LOW: "Low risk, heightened awareness recommended",
        RiskLevel.MODERATE: "Moderate risk, precautionary measures advised",
        RiskLevel.HIGH: "High risk, protective actions recommended",
        RiskLevel.VERY_HIGH: "Very high risk, immediate action required",
        RiskLevel.EXTREME: "Extreme risk, emergency response activated"
    }
    return descriptions.get(risk_level, "Risk level")


def _get_audience_description(audience_type: AudienceType) -> str:
    """Get description for audience type"""
    descriptions = {
        AudienceType.SCIENTIFIC: "Researchers and scientists requiring technical details",
        AudienceType.EMERGENCY_MANAGEMENT: "Emergency managers and first responders",
        AudienceType.GENERAL_PUBLIC: "General public requiring clear, actionable information",
        AudienceType.MEDIA: "News media and journalists",
        AudienceType.GOVERNMENT: "Government officials and policy makers",
        AudienceType.INDUSTRY: "Industrial and commercial stakeholders"
    }
    return descriptions.get(audience_type, "Target audience")


def _get_format_description(format_type: CommunicationFormat) -> str:
    """Get description for communication format"""
    descriptions = {
        CommunicationFormat.TECHNICAL_REPORT: "Detailed technical analysis with scientific data",
        CommunicationFormat.EXECUTIVE_SUMMARY: "Concise summary for decision makers",
        CommunicationFormat.PUBLIC_ADVISORY: "Public safety information in plain language",
        CommunicationFormat.EMERGENCY_ALERT: "Urgent alert for immediate action",
        CommunicationFormat.INFOGRAPHIC: "Visual summary with key information",
        CommunicationFormat.INTERACTIVE_MAP: "Interactive visualization of hazard data",
        CommunicationFormat.SOCIAL_MEDIA: "Brief update suitable for social media platforms"
    }
    return descriptions.get(format_type, "Communication format")