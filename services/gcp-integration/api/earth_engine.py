"""
Earth Engine API Router
REST API endpoints for Google Earth Engine geological analysis
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *
from ..core.earth_engine_integration import EarthEngineGeologicalAnalysis

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get Earth Engine service
async def get_earth_engine_service() -> EarthEngineGeologicalAnalysis:
    from ..main import earth_engine_service
    if not earth_engine_service:
        raise HTTPException(status_code=503, detail="Earth Engine service not available")
    return earth_engine_service


@router.get("/status")
async def get_earth_engine_status(
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Get Earth Engine service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="Google Earth Engine Geological Analysis",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting Earth Engine status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/analyze")
async def analyze_satellite_data(
    request: EarthEngineAnalysisRequest,
    background_tasks: BackgroundTasks,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Analyze satellite data using Google Earth Engine"""
    try:
        logger.info("Starting Earth Engine analysis", 
                   analysis_id=request.analysis_id, analysis_type=request.analysis_type)
        
        # Convert region to dictionary
        region_dict = {}
        if request.region.coordinates:
            region_dict["coordinates"] = request.region.coordinates
        elif request.region.bounds:
            region_dict["bounds"] = request.region.bounds
        elif request.region.center and request.region.radius:
            region_dict["center"] = request.region.center
            region_dict["radius"] = request.region.radius
        
        result = await service.analyze_satellite_data(
            request.analysis_id,
            region_dict,
            request.analysis_type,
            request.date_range,
            request.parameters
        )
        
        return AnalysisResponse(
            job_id=request.analysis_id,
            status="completed",
            message="Earth Engine analysis completed successfully",
            results=result
        )
        
    except Exception as e:
        logger.error("Error in Earth Engine analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.get("/datasets")
async def list_available_datasets(
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """List available Earth Engine datasets"""
    try:
        datasets = await service.list_available_datasets()
        return datasets
        
    except Exception as e:
        logger.error("Error listing datasets", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.get("/collections")
async def get_earth_engine_collections(
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Get Earth Engine collections information"""
    try:
        return {
            "collections": service.ee_collections,
            "total_collections": len(service.ee_collections)
        }
        
    except Exception as e:
        logger.error("Error getting collections", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting collections: {str(e)}")


@router.get("/algorithms")
async def list_analysis_algorithms(
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """List available analysis algorithms"""
    try:
        return {
            "algorithms": list(service.analysis_algorithms.keys()),
            "algorithm_descriptions": {
                "thermal_anomaly": "Detect thermal anomalies using MODIS and Landsat thermal data",
                "vegetation_stress": "Analyze vegetation stress for geological applications",
                "mineral_mapping": "Mineral mapping using multispectral satellite data",
                "landslide_detection": "Landslide detection using optical and radar data",
                "volcanic_monitoring": "Volcanic monitoring using thermal and deformation analysis",
                "deforestation": "Deforestation analysis and monitoring",
                "water_quality": "Water quality assessment using satellite data",
                "urban_expansion": "Urban expansion analysis and monitoring",
                "coastal_erosion": "Coastal erosion monitoring and analysis",
                "glacier_monitoring": "Glacier monitoring and change detection"
            }
        }
        
    except Exception as e:
        logger.error("Error listing algorithms", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing algorithms: {str(e)}")


@router.get("/export-tasks/{task_id}")
async def get_export_status(
    task_id: str,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Get Earth Engine export task status"""
    try:
        status = await service.get_export_status(task_id)
        return ExportTaskResponse(
            task_id=status["task_id"],
            status=status["status"],
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Error getting export status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting export status: {str(e)}")


@router.post("/thermal-anomaly")
async def detect_thermal_anomalies(
    region: GeographicRegion,
    date_range: Optional[Dict[str, str]] = None,
    threshold_multiplier: float = 2.0,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Detect thermal anomalies in specified region"""
    try:
        analysis_id = f"thermal_anomaly_{int(datetime.utcnow().timestamp())}"
        
        region_dict = {}
        if region.coordinates:
            region_dict["coordinates"] = region.coordinates
        elif region.bounds:
            region_dict["bounds"] = region.bounds
        elif region.center and region.radius:
            region_dict["center"] = region.center
            region_dict["radius"] = region.radius
        
        result = await service.analyze_satellite_data(
            analysis_id,
            region_dict,
            "thermal_anomaly",
            date_range,
            {"threshold_multiplier": threshold_multiplier}
        )
        
        return result
        
    except Exception as e:
        logger.error("Error detecting thermal anomalies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Thermal anomaly detection error: {str(e)}")


@router.post("/mineral-mapping")
async def perform_mineral_mapping(
    region: GeographicRegion,
    date_range: Optional[Dict[str, str]] = None,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Perform mineral mapping analysis"""
    try:
        analysis_id = f"mineral_mapping_{int(datetime.utcnow().timestamp())}"
        
        region_dict = {}
        if region.coordinates:
            region_dict["coordinates"] = region.coordinates
        elif region.bounds:
            region_dict["bounds"] = region.bounds
        elif region.center and region.radius:
            region_dict["center"] = region.center
            region_dict["radius"] = region.radius
        
        result = await service.analyze_satellite_data(
            analysis_id,
            region_dict,
            "mineral_mapping",
            date_range,
            {}
        )
        
        return result
        
    except Exception as e:
        logger.error("Error in mineral mapping", error=str(e))
        raise HTTPException(status_code=500, detail=f"Mineral mapping error: {str(e)}")


@router.post("/volcanic-monitoring")
async def monitor_volcanic_activity(
    region: GeographicRegion,
    date_range: Optional[Dict[str, str]] = None,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Monitor volcanic activity using satellite data"""
    try:
        analysis_id = f"volcanic_monitoring_{int(datetime.utcnow().timestamp())}"
        
        region_dict = {}
        if region.coordinates:
            region_dict["coordinates"] = region.coordinates
        elif region.bounds:
            region_dict["bounds"] = region.bounds
        elif region.center and region.radius:
            region_dict["center"] = region.center
            region_dict["radius"] = region.radius
        
        result = await service.analyze_satellite_data(
            analysis_id,
            region_dict,
            "volcanic_monitoring",
            date_range,
            {}
        )
        
        return result
        
    except Exception as e:
        logger.error("Error in volcanic monitoring", error=str(e))
        raise HTTPException(status_code=500, detail=f"Volcanic monitoring error: {str(e)}")


@router.post("/landslide-detection")
async def detect_landslides(
    region: GeographicRegion,
    date_range: Optional[Dict[str, str]] = None,
    slope_threshold: float = 30.0,
    ndvi_change_threshold: float = -0.3,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Detect potential landslides using satellite data"""
    try:
        analysis_id = f"landslide_detection_{int(datetime.utcnow().timestamp())}"
        
        region_dict = {}
        if region.coordinates:
            region_dict["coordinates"] = region.coordinates
        elif region.bounds:
            region_dict["bounds"] = region.bounds
        elif region.center and region.radius:
            region_dict["center"] = region.center
            region_dict["radius"] = region.radius
        
        result = await service.analyze_satellite_data(
            analysis_id,
            region_dict,
            "landslide_detection",
            date_range,
            {
                "slope_threshold": slope_threshold,
                "ndvi_change_threshold": ndvi_change_threshold
            }
        )
        
        return result
        
    except Exception as e:
        logger.error("Error in landslide detection", error=str(e))
        raise HTTPException(status_code=500, detail=f"Landslide detection error: {str(e)}")


@router.post("/vegetation-stress")
async def analyze_vegetation_stress(
    region: GeographicRegion,
    date_range: Optional[Dict[str, str]] = None,
    stress_threshold: float = 0.3,
    variability_threshold: float = 0.1,
    service: EarthEngineGeologicalAnalysis = Depends(get_earth_engine_service)
):
    """Analyze vegetation stress for geological applications"""
    try:
        analysis_id = f"vegetation_stress_{int(datetime.utcnow().timestamp())}"
        
        region_dict = {}
        if region.coordinates:
            region_dict["coordinates"] = region.coordinates
        elif region.bounds:
            region_dict["bounds"] = region.bounds
        elif region.center and region.radius:
            region_dict["center"] = region.center
            region_dict["radius"] = region.radius
        
        result = await service.analyze_satellite_data(
            analysis_id,
            region_dict,
            "vegetation_stress",
            date_range,
            {
                "stress_threshold": stress_threshold,
                "variability_threshold": variability_threshold
            }
        )
        
        return result
        
    except Exception as e:
        logger.error("Error in vegetation stress analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Vegetation stress analysis error: {str(e)}")