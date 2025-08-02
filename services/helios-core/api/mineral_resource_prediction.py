"""
API endpoints for mineral resource prediction functionality.

This module provides REST API endpoints for training and using mineral resource
prediction models, including ore deposit pattern recognition, geochemical signature
analysis, and geological process modeling.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch.nn as nn
import torch.nn.functional as F

from core.config import get_settings
from core.mineral_resource_prediction import (DepositType, FormationProcess,
                                              MineralResourcePredictor,
                                              ResourceDeposit, ResourceType)
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from models.geological import GeochemicalSample, GeologicalLocation
from pydantic import BaseModel, Field

# Initialize router
router = APIRouter(prefix="/api/v1/mineral-resources", tags=["mineral-resources"])
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

def get_predictor() -> MineralResourcePredictor:
    """Get or create the mineral resource predictor instance"""
    global predictor
    if predictor is None:
        settings = get_settings()
        predictor = MineralResourcePredictor(
            model_storage_path=settings.model_storage_path
        )
        # Try to load existing models
        try:
            predictor.load_models()
            logger.info("Loaded existing mineral resource prediction models")
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
    return predictor

# Pydantic models for API requests/responses
class LocationRequest(BaseModel):
    """Request model for geological location"""

    latitude: float = Field(
        ..., ge=-90, le=90, description="Latitude in decimal degrees"
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Longitude in decimal degrees"
    )
    elevation: float = Field(..., description="Elevation above sea level in meters")
    depth: Optional[float] = Field(
        None, ge=0, description="Depth below surface in meters"
    )

class GeochemicalDataRequest(BaseModel):
    """Request model for geochemical data"""

    composition: Dict[str, float] = Field(
        ..., description="Element/compound concentrations"
    )
    isotope_ratios: Optional[Dict[str, float]] = Field(
        None, description="Isotopic ratios"
    )
    temperature: Optional[float] = Field(
        None, description="Sample temperature in Celsius"
    )
    pressure: Optional[float] = Field(None, description="Sample pressure in Pa")
    ph: Optional[float] = Field(None, ge=0, le=14, description="pH value")

class GeologicalSettingRequest(BaseModel):
    """Request model for geological setting"""

    rock_types: List[str] = Field(..., description="Present rock types")
    fault_density: float = Field(0.0, ge=0, description="Fault density (faults/km²)")
    fracture_intensity: float = Field(0.0, ge=0, description="Fracture intensity")
    fold_intensity: float = Field(0.0, ge=0, description="Fold intensity")
    intrusion_distance: float = Field(
        1000.0, ge=0, description="Distance to nearest intrusion (m)"
    )
    volcanic_activity: float = Field(
        0.0, ge=0, le=1, description="Volcanic activity level (0-1)"
    )
    hydrothermal_alteration: float = Field(
        0.0, ge=0, le=1, description="Hydrothermal alteration intensity (0-1)"
    )

class ProcessConditionsRequest(BaseModel):
    """Request model for process modeling conditions"""

    hydrothermal: Optional[Dict[str, Any]] = Field(
        None, description="Hydrothermal process conditions"
    )
    magmatic: Optional[Dict[str, Any]] = Field(
        None, description="Magmatic process conditions"
    )
    sedimentary: Optional[Dict[str, Any]] = Field(
        None, description="Sedimentary process conditions"
    )

class ResourcePredictionRequest(BaseModel):
    """Request model for resource prediction"""

    location: LocationRequest
    geological_setting: GeologicalSettingRequest
    geochemical_data: Optional[GeochemicalDataRequest] = None
    process_conditions: Optional[ProcessConditionsRequest] = None

class TrainingDepositRequest(BaseModel):
    """Request model for training deposit data"""

    location: LocationRequest
    resource_type: ResourceType
    deposit_type: DepositType
    formation_process: FormationProcess
    grade: float = Field(..., gt=0, description="Resource grade")
    tonnage: float = Field(..., gt=0, description="Resource tonnage")
    geochemical_signature: Dict[str, float]
    geological_setting: GeologicalSettingRequest
    confidence: float = Field(
        1.0, ge=0, le=1, description="Confidence in deposit parameters"
    )
    deposit_id: Optional[str] = None

class TrainingDataRequest(BaseModel):
    """Request model for adding training data"""

    deposits: List[TrainingDepositRequest]
    geochemical_samples: List[GeochemicalDataRequest]

class ResourceMapRequest(BaseModel):
    """Request model for generating resource maps"""

    locations: List[LocationRequest]
    geological_settings: List[GeologicalSettingRequest]
    resource_type: Optional[ResourceType] = None

class ResourcePredictionResponse(BaseModel):
    """Response model for resource predictions"""

    location: LocationRequest
    resource_type: ResourceType
    probability: float
    predicted_grade: float
    predicted_tonnage: float
    uncertainty_bounds: Dict[str, List[float]]
    formation_likelihood: Dict[str, float]
    geochemical_indicators: Dict[str, float]
    geological_favorability: float
    model_confidence: float
    prediction_id: Optional[str] = None
    timestamp: str

class TrainingResultsResponse(BaseModel):
    """Response model for training results"""

    signature_analysis: Dict[str, Any]
    pattern_recognition: Dict[str, Any]
    training_timestamp: str
    status: str

class ResourceMapResponse(BaseModel):
    """Response model for resource maps"""

    locations: List[List[float]]
    probabilities: List[float]
    resource_types: List[str]
    predicted_grades: List[float]
    predicted_tonnages: List[float]
    confidence_levels: List[float]
    uncertainty_bounds: List[Dict[str, List[float]]]
    metadata: Dict[str, Any]

@router.post("/predict", response_model=ResourcePredictionResponse)
async def predict_resource_potential(
    request: ResourcePredictionRequest,
    predictor: MineralResourcePredictor = Depends(get_predictor),
):
    """
    Predict mineral resource potential at a specific location.

    This endpoint combines pattern recognition, geochemical signature analysis,
    and geological process modeling to provide comprehensive resource predictions
    with uncertainty quantification.
    """
    try:
        # Convert request to internal models
        location = GeologicalLocation(
            latitude=request.location.latitude,
            longitude=request.location.longitude,
            elevation=request.location.elevation,
            depth=request.location.depth,
        )

        geological_setting = {
            "rock_types": request.geological_setting.rock_types,
            "fault_density": request.geological_setting.fault_density,
            "fracture_intensity": request.geological_setting.fracture_intensity,
            "fold_intensity": request.geological_setting.fold_intensity,
            "intrusion_distance": request.geological_setting.intrusion_distance,
            "volcanic_activity": request.geological_setting.volcanic_activity,
            "hydrothermal_alteration": request.geological_setting.hydrothermal_alteration,
        }

        geochemical_data = None
        if request.geochemical_data:
            geochemical_data = request.geochemical_data.composition

        process_conditions = None
        if request.process_conditions:
            process_conditions = request.process_conditions.dict(exclude_none=True)

        # Make prediction
        prediction = predictor.predict_resource_potential(
            location=location,
            geological_setting=geological_setting,
            geochemical_data=geochemical_data,
            process_conditions=process_conditions,
        )

        # Convert uncertainty bounds format
        uncertainty_bounds_formatted = {}
        for key, (lower, upper) in prediction.uncertainty_bounds.items():
            uncertainty_bounds_formatted[key] = [lower, upper]

        # Convert formation likelihood
        formation_likelihood_formatted = {
            process.value: likelihood
            for process, likelihood in prediction.formation_likelihood.items()
        }

        # Create response
        response = ResourcePredictionResponse(
            location=request.location,
            resource_type=prediction.resource_type,
            probability=prediction.probability,
            predicted_grade=prediction.predicted_grade,
            predicted_tonnage=prediction.predicted_tonnage,
            uncertainty_bounds=uncertainty_bounds_formatted,
            formation_likelihood=formation_likelihood_formatted,
            geochemical_indicators=prediction.geochemical_indicators,
            geological_favorability=prediction.geological_favorability,
            model_confidence=prediction.model_confidence,
            prediction_id=prediction.prediction_id,
            timestamp=prediction.timestamp.isoformat(),
        )

        logger.info(
            f"Resource prediction completed for location {location.latitude}, {location.longitude}"
        )
        return response

    except Exception as e:
        logger.error(f"Error in resource prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/train", response_model=TrainingResultsResponse)
async def train_models(
    request: TrainingDataRequest,
    background_tasks: BackgroundTasks,
    predictor: MineralResourcePredictor = Depends(get_predictor),
):
    """
    Train mineral resource prediction models with provided data.

    This endpoint trains the geochemical signature analyzer and pattern recognition
    models using the provided training dataset of known deposits and samples.
    """
    try:
        # Convert training deposits
        deposits = []
        for deposit_req in request.deposits:
            location = GeologicalLocation(
                latitude=deposit_req.location.latitude,
                longitude=deposit_req.location.longitude,
                elevation=deposit_req.location.elevation,
                depth=deposit_req.location.depth,
            )

            geological_setting = {
                "rock_types": deposit_req.geological_setting.rock_types,
                "fault_density": deposit_req.geological_setting.fault_density,
                "fracture_intensity": deposit_req.geological_setting.fracture_intensity,
                "fold_intensity": deposit_req.geological_setting.fold_intensity,
                "intrusion_distance": deposit_req.geological_setting.intrusion_distance,
                "volcanic_activity": deposit_req.geological_setting.volcanic_activity,
                "hydrothermal_alteration": deposit_req.geological_setting.hydrothermal_alteration,
            }

            deposit = ResourceDeposit(
                location=location,
                resource_type=deposit_req.resource_type,
                deposit_type=deposit_req.deposit_type,
                formation_process=deposit_req.formation_process,
                grade=deposit_req.grade,
                tonnage=deposit_req.tonnage,
                geochemical_signature=deposit_req.geochemical_signature,
                geological_setting=geological_setting,
                confidence=deposit_req.confidence,
                deposit_id=deposit_req.deposit_id,
            )
            deposits.append(deposit)

        # Convert geochemical samples
        samples = []
        for i, sample_req in enumerate(request.geochemical_samples):
            # Use first deposit location as default for samples
            sample_location = (
                deposits[0].location
                if deposits
                else GeologicalLocation(latitude=0.0, longitude=0.0, elevation=0.0)
            )

            sample = GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=sample_location,
                sample_type="rock",  # Default
                composition=sample_req.composition,
                isotope_ratios=sample_req.isotope_ratios,
                temperature=sample_req.temperature,
                pressure=sample_req.pressure,
                ph=sample_req.ph,
            )
            samples.append(sample)

        # Add training data
        predictor.add_training_data(deposits, samples)

        # Train models
        results = predictor.train_models()

        response = TrainingResultsResponse(
            signature_analysis=results.get("signature_analysis", {}),
            pattern_recognition=results.get("pattern_recognition", {}),
            training_timestamp=datetime.utcnow().isoformat(),
            status="completed",
        )

        logger.info(
            f"Model training completed with {len(deposits)} deposits and {len(samples)} samples"
        )
        return response

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/generate-map", response_model=ResourceMapResponse)
async def generate_resource_map(
    request: ResourceMapRequest,
    predictor: MineralResourcePredictor = Depends(get_predictor),
):
    """
    Generate a resource potential map for multiple locations.

    This endpoint evaluates resource potential across multiple locations and
    returns a comprehensive map with predictions, uncertainties, and metadata.
    """
    try:
        if len(request.locations) != len(request.geological_settings):
            raise HTTPException(
                status_code=400,
                detail="Number of locations must match number of geological settings",
            )

        # Convert locations
        locations = []
        for loc_req in request.locations:
            location = GeologicalLocation(
                latitude=loc_req.latitude,
                longitude=loc_req.longitude,
                elevation=loc_req.elevation,
                depth=loc_req.depth,
            )
            locations.append(location)

        # Convert geological settings
        geological_settings = []
        for setting_req in request.geological_settings:
            setting = {
                "rock_types": setting_req.rock_types,
                "fault_density": setting_req.fault_density,
                "fracture_intensity": setting_req.fracture_intensity,
                "fold_intensity": setting_req.fold_intensity,
                "intrusion_distance": setting_req.intrusion_distance,
                "volcanic_activity": setting_req.volcanic_activity,
                "hydrothermal_alteration": setting_req.hydrothermal_alteration,
            }
            geological_settings.append(setting)

        # Generate map
        map_data = predictor.generate_resource_map(
            locations=locations,
            geological_settings=geological_settings,
            resource_type=request.resource_type,
        )

        # Format uncertainty bounds for response
        formatted_uncertainty_bounds = []
        for bounds in map_data["uncertainty_bounds"]:
            formatted_bounds = {}
            for key, (lower, upper) in bounds.items():
                formatted_bounds[key] = [lower, upper]
            formatted_uncertainty_bounds.append(formatted_bounds)

        response = ResourceMapResponse(
            locations=map_data["locations"],
            probabilities=map_data["probabilities"],
            resource_types=map_data["resource_types"],
            predicted_grades=map_data["predicted_grades"],
            predicted_tonnages=map_data["predicted_tonnages"],
            confidence_levels=map_data["confidence_levels"],
            uncertainty_bounds=formatted_uncertainty_bounds,
            metadata=map_data["metadata"],
        )

        logger.info(f"Resource map generated for {len(locations)} locations")
        return response

    except Exception as e:
        logger.error(f"Error in resource map generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Map generation failed: {str(e)}")

@router.get("/resource-types")
async def get_resource_types():
    """Get available resource types for prediction."""
    return {
        "resource_types": [rt.value for rt in ResourceType],
        "description": "Available mineral resource types for prediction",
    }

@router.get("/deposit-types")
async def get_deposit_types():
    """Get available deposit types for classification."""
    return {
        "deposit_types": [dt.value for dt in DepositType],
        "description": "Available ore deposit types for classification",
    }

@router.get("/formation-processes")
async def get_formation_processes():
    """Get available formation processes for modeling."""
    return {
        "formation_processes": [fp.value for fp in FormationProcess],
        "description": "Available ore formation processes for modeling",
    }

@router.get("/model-status")
async def get_model_status(
    predictor: MineralResourcePredictor = Depends(get_predictor),
):
    """Get current status of trained models."""
    try:
        # Check if models are trained
        signature_trained = bool(predictor.signature_analyzer.signature_models)
        pattern_trained = bool(predictor.pattern_recognizer.pattern_models)

        # Get training data statistics
        n_deposits = len(predictor.training_deposits)
        n_samples = len(predictor.training_samples)

        return {
            "signature_analyzer_trained": signature_trained,
            "pattern_recognizer_trained": pattern_trained,
            "training_deposits": n_deposits,
            "training_samples": n_samples,
            "models_ready": signature_trained and pattern_trained,
            "status": (
                "ready" if (signature_trained and pattern_trained) else "needs_training"
            ),
        }

    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/models")
async def reset_models(predictor: MineralResourcePredictor = Depends(get_predictor)):
    """Reset all trained models and clear training data."""
    try:
        # Clear models
        predictor.signature_analyzer.signature_models.clear()
        predictor.pattern_recognizer.pattern_models.clear()

        # Clear training data
        predictor.training_deposits.clear()
        predictor.training_samples.clear()

        logger.info("All models and training data cleared")
        return {"status": "success", "message": "All models and training data cleared"}

    except Exception as e:
        logger.error(f"Error resetting models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
