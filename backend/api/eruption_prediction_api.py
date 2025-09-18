"""
FastAPI endpoints for Eruption Location Prediction System

This module provides REST API endpoints for the advanced eruption location prediction
system, enabling integration with the volcanic dashboard frontend and external systems.

Author: Prometheus AI Platform
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
from uuid import UUID, uuid4

# Import the main prediction system
from backend.modules.eruption_location_prediction_system import (
    EruptionLocationPredictionSystem,
    SillGeometry,
    StressFieldParameters,
    FailureCriterion,
    TopographicControls,
    EruptionPrediction,
    HazardZone
)

logger = logging.getLogger(__name__)

# Pydantic models for API requests and responses
class SillGeometryRequest(BaseModel):
    sill_id: str
    center_coordinates: Tuple[float, float, float] = Field(..., description="X, Y, Z coordinates in meters")
    length: float = Field(..., gt=0, description="Sill length in meters")
    width: float = Field(..., gt=0, description="Sill width in meters") 
    thickness: float = Field(..., gt=0, description="Sill thickness in meters")
    dip: float = Field(..., ge=0, le=90, description="Sill dip angle in degrees")
    strike: float = Field(..., ge=0, lt=360, description="Sill strike angle in degrees")
    volume: float = Field(..., gt=0, description="Sill volume in cubic meters")
    aspect_ratio: float = Field(..., gt=0, description="Length to width ratio")
    shape_factor: float = Field(..., ge=0, le=1, description="Geometric shape factor")
    connectivity_index: float = Field(..., ge=0, le=1, description="Connection to other sills")

class StressFieldRequest(BaseModel):
    principal_stress_1: float = Field(..., description="Maximum principal stress in Pa")
    principal_stress_2: float = Field(..., description="Intermediate principal stress in Pa")
    principal_stress_3: float = Field(..., description="Minimum principal stress in Pa")
    stress_orientation: Tuple[float, float, float] = Field(..., description="Azimuth, plunge, rake")
    analysis_grid: Tuple[int, int, int] = Field(default=(100, 100, 50), description="Grid resolution")

class RockPropertiesRequest(BaseModel):
    cohesion: float = Field(default=10e6, description="Cohesive strength in Pa")
    friction_angle: float = Field(default=30.0, description="Internal friction angle in degrees")
    tensile_strength: float = Field(default=5e6, description="Tensile strength in Pa")
    ucs: float = Field(default=100e6, description="Unconfined compressive strength in Pa")
    hoek_brown_mi: float = Field(default=10.0, description="Hoek-Brown material constant")
    hoek_brown_gsi: float = Field(default=50.0, description="Geological Strength Index")

class TopographicAnalysisRequest(BaseModel):
    analysis_area: Tuple[float, float, float, float] = Field(..., description="min_x, min_y, max_x, max_y")
    dem_resolution: float = Field(default=30.0, description="DEM resolution in meters")

class MagmaPropertiesRequest(BaseModel):
    viscosity: float = Field(default=1e4, description="Magma viscosity in Pa·s")
    gas_content: float = Field(default=0.03, ge=0, le=1, description="Gas content by weight fraction")
    temperature: float = Field(default=1200, description="Temperature in Celsius")
    silica_content: float = Field(default=0.65, ge=0, le=1, description="Silica content fraction")
    density: float = Field(default=2500, description="Magma density in kg/m³")

class EruptionPredictionRequest(BaseModel):
    sill_geometries: List[SillGeometryRequest]
    stress_field_params: StressFieldRequest
    rock_properties: RockPropertiesRequest
    topographic_analysis: TopographicAnalysisRequest
    magma_properties: MagmaPropertiesRequest
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    time_horizon_days: float = Field(default=365.0, gt=0)

class EruptionPredictionResponse(BaseModel):
    prediction_id: str
    analysis_timestamp: datetime
    stress_field_results: Dict[str, Any]
    failure_analysis_results: Dict[str, Any]
    topographic_controls: Dict[str, Any]
    probabilistic_mapping: Dict[str, Any]
    temporal_evolution: Dict[str, Any]
    eruption_style_predictions: Dict[str, Any]
    hazard_zone_mapping: Dict[str, Any]
    validation_results: Dict[str, Any]
    overall_confidence: float
    processing_time_seconds: float

class HazardZoneResponse(BaseModel):
    zone_id: str
    hazard_level: str
    hazard_type: str
    probability: float
    impact_radius: float
    population_at_risk: int
    evacuation_time: float
    monitoring_priority: int

# Initialize router
router = APIRouter(prefix="/api/v1/eruption-prediction", tags=["Eruption Prediction"])

# Global prediction system instance
prediction_system: Optional[EruptionLocationPredictionSystem] = None

async def get_prediction_system():
    """Dependency to get the prediction system instance"""
    global prediction_system
    if prediction_system is None:
        # Initialize with database configuration
        db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'prometheus_volcanic',
            'user': 'prometheus',
            'password': 'prometheus123',
            'redis_host': 'localhost',
            'redis_port': 6379
        }
        prediction_system = EruptionLocationPredictionSystem(db_config)
        await prediction_system.initialize()
    return prediction_system

@router.post("/analyze/complete", response_model=EruptionPredictionResponse)
async def complete_eruption_analysis(
    request: EruptionPredictionRequest,
    background_tasks: BackgroundTasks,
    system: EruptionLocationPredictionSystem = Depends(get_prediction_system)
):
    """
    Perform complete eruption location prediction analysis
    """
    try:
        start_time = datetime.now()
        prediction_id = str(uuid4())
        
        logger.info(f"Starting complete eruption analysis {prediction_id}")
        
        # Convert request models to system models
        sill_geometries = [
            SillGeometry(
                sill_id=sg.sill_id,
                center_coordinates=sg.center_coordinates,
                length=sg.length,
                width=sg.width,
                thickness=sg.thickness,
                dip=sg.dip,
                strike=sg.strike,
                volume=sg.volume,
                aspect_ratio=sg.aspect_ratio,
                shape_factor=sg.shape_factor,
                connectivity_index=sg.connectivity_index
            ) for sg in request.sill_geometries
        ]
        
        regional_stress = StressFieldParameters(
            principal_stress_1=request.stress_field_params.principal_stress_1,
            principal_stress_2=request.stress_field_params.principal_stress_2,
            principal_stress_3=request.stress_field_params.principal_stress_3,
            stress_orientation=request.stress_field_params.stress_orientation,
            differential_stress=request.stress_field_params.principal_stress_1 - request.stress_field_params.principal_stress_3,
            mean_stress=(request.stress_field_params.principal_stress_1 + 
                        request.stress_field_params.principal_stress_2 + 
                        request.stress_field_params.principal_stress_3) / 3,
            stress_ratio=((request.stress_field_params.principal_stress_2 - request.stress_field_params.principal_stress_3) /
                         (request.stress_field_params.principal_stress_1 - request.stress_field_params.principal_stress_3)),
            deviatoric_stress=None,  # Will be calculated
            stress_invariants=(0, 0, 0),  # Will be calculated
            von_mises_stress=0  # Will be calculated
        )
        
        # Step 1: Stress field analysis
        stress_field_results = await system.analyze_stress_field_around_sill_complex(
            sill_geometries, regional_stress, request.stress_field_params.analysis_grid
        )
        
        # Step 2: Failure criterion analysis
        rock_props = {
            'cohesion': request.rock_properties.cohesion,
            'friction_angle': request.rock_properties.friction_angle * 3.14159 / 180,  # Convert to radians
            'tensile_strength': request.rock_properties.tensile_strength,
            'ucs': request.rock_properties.ucs,
            'hoek_brown_mi': request.rock_properties.hoek_brown_mi,
            'hoek_brown_gsi': request.rock_properties.hoek_brown_gsi
        }
        
        failure_analysis_results = await system.analyze_failure_criteria(
            stress_field_results, rock_props
        )
        
        # Step 3: Topographic and structural controls
        topographic_controls = await system.analyze_topographic_structural_controls(
            request.topographic_analysis.analysis_area,
            request.topographic_analysis.dem_resolution
        )
        
        # Step 4: Probabilistic mapping
        probabilistic_mapping = await system.create_probabilistic_eruption_location_mapping(
            stress_field_results,
            failure_analysis_results,
            topographic_controls,
            sill_geometries,
            request.confidence_level
        )
        
        # Step 5: Temporal evolution modeling
        growth_rates = {sg.sill_id: 1000.0 for sg in sill_geometries}  # Default growth rates
        temporal_evolution = await system.model_temporal_evolution_eruption_potential(
            sill_geometries, growth_rates, request.time_horizon_days
        )
        
        # Step 6: Eruption style prediction
        magma_props = {
            'viscosity': request.magma_properties.viscosity,
            'gas_content': request.magma_properties.gas_content,
            'temperature': request.magma_properties.temperature,
            'silica_content': request.magma_properties.silica_content,
            'density': request.magma_properties.density
        }
        
        eruption_style_predictions = await system.predict_eruption_style(
            sill_geometries, magma_props, stress_field_results
        )
        
        # Step 7: Hazard zone mapping
        population_data = {'total': 100000, 'density': 500}  # Mock data - would be loaded from database
        infrastructure_data = ['hospital', 'school', 'power_plant']  # Mock data
        
        hazard_zone_mapping = await system.create_hazard_zone_mapping(
            probabilistic_mapping, eruption_style_predictions, 
            population_data, infrastructure_data
        )
        
        # Step 8: Validation against historical data
        validation_results = await system.validate_against_historical_eruptions(
            {
                'surface_coordinates': probabilistic_mapping['surface_coordinates'],
                'eruption_probability': probabilistic_mapping['eruption_probability'],
                'eruption_forecasts': temporal_evolution.get('eruption_forecasts', {}),
                'eruption_style_predictions': eruption_style_predictions
            }
        )
        
        # Calculate overall confidence
        overall_confidence = min(1.0, (
            validation_results.get('model_performance_score', 0.7) * 0.4 +
            probabilistic_mapping.get('confidence_level', 0.95) * 0.3 +
            stress_field_results.get('validation_score', 0.8) * 0.3
        ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Schedule background task for additional processing
        background_tasks.add_task(
            _post_process_analysis, prediction_id, stress_field_results, system
        )
        
        response = EruptionPredictionResponse(
            prediction_id=prediction_id,
            analysis_timestamp=start_time,
            stress_field_results=stress_field_results,
            failure_analysis_results=failure_analysis_results,
            topographic_controls=topographic_controls,
            probabilistic_mapping=probabilistic_mapping,
            temporal_evolution=temporal_evolution,
            eruption_style_predictions=eruption_style_predictions,
            hazard_zone_mapping=hazard_zone_mapping,
            validation_results=validation_results,
            overall_confidence=overall_confidence,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"Completed eruption analysis {prediction_id} in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in complete eruption analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze/stress-field")
async def analyze_stress_field(
    sill_geometries: List[SillGeometryRequest],
    stress_params: StressFieldRequest,
    system: EruptionLocationPredictionSystem = Depends(get_prediction_system)
):
    """Analyze stress field around sill complex geometries"""
    try:
        # Convert to system models
        sills = [SillGeometry(**sg.dict()) for sg in sill_geometries]
        regional_stress = StressFieldParameters(
            principal_stress_1=stress_params.principal_stress_1,
            principal_stress_2=stress_params.principal_stress_2,
            principal_stress_3=stress_params.principal_stress_3,
            stress_orientation=stress_params.stress_orientation,
            differential_stress=stress_params.principal_stress_1 - stress_params.principal_stress_3,
            mean_stress=(stress_params.principal_stress_1 + stress_params.principal_stress_2 + stress_params.principal_stress_3) / 3,
            stress_ratio=0,  # Will be calculated
            deviatoric_stress=None,
            stress_invariants=(0, 0, 0),
            von_mises_stress=0
        )
        
        results = await system.analyze_stress_field_around_sill_complex(
            sills, regional_stress, stress_params.analysis_grid
        )
        
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Error in stress field analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/{prediction_id}")
async def get_prediction_results(
    prediction_id: str,
    system: EruptionLocationPredictionSystem = Depends(get_prediction_system)
):
    """Retrieve stored prediction results by ID"""
    try:
        # Query database for prediction results
        async with system.engine.begin() as conn:
            result = await conn.execute(
                "SELECT * FROM probabilistic_eruption_mapping WHERE mapping_id = %s",
                (prediction_id,)
            )
            prediction_data = result.fetchone()
            
        if not prediction_data:
            raise HTTPException(status_code=404, detail="Prediction not found")
            
        return JSONResponse(content=dict(prediction_data))
        
    except Exception as e:
        logger.error(f"Error retrieving prediction {prediction_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hazard-zones/active", response_model=List[HazardZoneResponse])
async def get_active_hazard_zones(
    hazard_level: Optional[str] = Query(None, description="Filter by hazard level"),
    min_probability: Optional[float] = Query(0.0, description="Minimum probability threshold"),
    system: EruptionLocationPredictionSystem = Depends(get_prediction_system)
):
    """Get currently active hazard zones"""
    try:
        # Query active hazard zones from database
        query = """
        SELECT hz.zone_id, hz.hazard_level, hz.hazard_type, hz.occurrence_probability,
               hz.impact_radius_m, hz.population_at_risk, hz.evacuation_time_hours,
               hz.monitoring_priority
        FROM hazard_zones hz
        JOIN probabilistic_eruption_mapping pem ON hz.mapping_id = pem.mapping_id
        WHERE pem.analysis_timestamp > NOW() - INTERVAL '7 days'
        """
        
        params = []
        if hazard_level:
            query += " AND hz.hazard_level = %s"
            params.append(hazard_level)
        if min_probability > 0:
            query += " AND hz.occurrence_probability >= %s"
            params.append(min_probability)
            
        query += " ORDER BY hz.occurrence_probability DESC, hz.population_at_risk DESC"
        
        async with system.engine.begin() as conn:
            result = await conn.execute(query, params)
            hazard_zones = result.fetchall()
            
        response = [
            HazardZoneResponse(
                zone_id=hz['zone_id'],
                hazard_level=hz['hazard_level'],
                hazard_type=hz['hazard_type'],
                probability=hz['occurrence_probability'],
                impact_radius=hz['impact_radius_m'],
                population_at_risk=hz['population_at_risk'],
                evacuation_time=hz['evacuation_time_hours'],
                monitoring_priority=hz['monitoring_priority']
            ) for hz in hazard_zones
        ]
        
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving hazard zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/validation/historical")
async def get_historical_validation_metrics(
    days_back: int = Query(30, description="Days of historical data to include"),
    system: EruptionLocationPredictionSystem = Depends(get_prediction_system)
):
    """Get historical validation metrics for model performance"""
    try:
        async with system.engine.begin() as conn:
            result = await conn.execute("""
                SELECT validation_timestamp, model_performance_score, 
                       spatial_accuracy, probability_calibration, overall_performance
                FROM model_validation_results
                WHERE validation_timestamp > NOW() - INTERVAL '%s days'
                ORDER BY validation_timestamp DESC
            """, (days_back,))
            
            validation_data = [dict(row) for row in result.fetchall()]
            
        return JSONResponse(content={
            'validation_history': validation_data,
            'summary_statistics': {
                'mean_performance_score': sum(v['model_performance_score'] for v in validation_data) / len(validation_data) if validation_data else 0,
                'total_validations': len(validation_data),
                'date_range': f"Last {days_back} days"
            }
        })
        
    except Exception as e:
        logger.error(f"Error retrieving validation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _post_process_analysis(
    prediction_id: str, 
    stress_results: Dict[str, Any], 
    system: EruptionLocationPredictionSystem
):
    """Background task for additional post-processing"""
    try:
        # Perform additional analysis or cleanup
        logger.info(f"Post-processing analysis {prediction_id}")
        
        # Could include:
        # - Additional validation
        # - Data archiving
        # - Notification sending
        # - Report generation
        
        await asyncio.sleep(1)  # Simulate processing
        logger.info(f"Completed post-processing for {prediction_id}")
        
    except Exception as e:
        logger.error(f"Error in post-processing {prediction_id}: {e}")

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}