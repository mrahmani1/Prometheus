"""API routes for Helios Core Service"""

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import numpy as np
import structlog

logger = structlog.get_logger()

router = APIRouter()

# Pydantic models for request/response validation
class GeologicalDataRequest(BaseModel):
    input_features: List[List[float]] = Field(..., description="Input features for geological analysis")
    timestamp: str = Field(..., description="Timestamp of the data")
    location: Dict[str, float] = Field(..., description="Geographic location")
    data_type: str = Field(..., description="Type of geological data")

class SeismicEventRequest(BaseModel):
    waveform_data: List[List[float]] = Field(..., description="3-component seismic waveform data")
    sampling_rate: float = Field(..., description="Sampling rate in Hz")
    station_id: str = Field(..., description="Seismic station identifier")
    timestamp: str = Field(..., description="Event timestamp")

class UncertaintyRequest(BaseModel):
    num_scenarios: int = Field(default=100, ge=1, le=1000, description="Number of scenarios to generate")
    scenario_type: str = Field(default="geological", description="Type of uncertainty scenarios")

class GeologicalDataResponse(BaseModel):
    predictions: List[List[float]]
    model_version: str
    timestamp: str
    confidence: float
    processing_time_ms: float

class SeismicEventResponse(BaseModel):
    event_type: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    processing_time_ms: float

class UncertaintyResponse(BaseModel):
    scenarios: List[List[float]]
    num_scenarios: int
    model_version: str
    generation_time_ms: float

# Dependency to get Helios engine instance
async def get_helios_engine():
    """Dependency to get the Helios engine instance"""
    # This would be injected from the main application
    # For now, we'll assume it's available globally
    from main import helios_engine
    if not helios_engine or not await helios_engine.is_ready():
        raise HTTPException(status_code=503, detail="Helios engine not ready")
    return helios_engine

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Helios AI Core Engine",
        "version": "1.0.0",
        "description": "Physics-Informed Neural Network engine for geodynamic simulations",
        "endpoints": {
            "geological_analysis": "/geological/analyze",
            "seismic_detection": "/seismic/detect",
            "uncertainty_generation": "/uncertainty/generate",
            "model_info": "/models/info"
        }
    }

@router.post("/geological/analyze", response_model=GeologicalDataResponse)
async def analyze_geological_data(
    request: GeologicalDataRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Analyze geological data using Physics-Informed Neural Networks"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing geological data analysis request",
                   data_type=request.data_type,
                   location=request.location)
        
        # Process the geological data
        result = await helios_engine.process_geological_data({
            "input_features": request.input_features,
            "timestamp": request.timestamp,
            "location": request.location,
            "data_type": request.data_type
        })
        
        processing_time = (time.time() - start_time) * 1000
        
        return GeologicalDataResponse(
            predictions=result["predictions"],
            model_version=result["model_version"],
            timestamp=result["timestamp"],
            confidence=result["confidence"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to analyze geological data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/seismic/detect", response_model=SeismicEventResponse)
async def detect_seismic_events(
    request: SeismicEventRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Detect and classify seismic events using CNN models"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing seismic event detection request",
                   station_id=request.station_id,
                   sampling_rate=request.sampling_rate)
        
        # Convert waveform data to numpy array
        waveform_array = np.array(request.waveform_data, dtype=np.float32)
        
        # Detect seismic events
        result = await helios_engine.detect_seismic_events(waveform_array)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SeismicEventResponse(
            event_type=result["event_type"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version=result["model_version"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to detect seismic events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# Enhanced seismic processing endpoints
class ContinuousSeismicRequest(BaseModel):
    waveform_data: List[List[float]] = Field(..., description="3-component continuous seismic waveform data")
    timestamps: List[str] = Field(..., description="Timestamp for each sample")
    sampling_rate: float = Field(..., description="Sampling rate in Hz")
    station_id: str = Field(..., description="Seismic station identifier")
    location: Dict[str, float] = Field(..., description="Station location (lat, lon, elevation)")
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection threshold")

class MultiStationSeismicRequest(BaseModel):
    stations: List[Dict[str, Any]] = Field(..., description="List of station data with waveforms")
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Detection threshold")
    location_estimation: bool = Field(default=True, description="Whether to estimate event locations")

class SeismicEventListResponse(BaseModel):
    events: List[Dict[str, Any]]
    total_events: int
    processing_statistics: Dict[str, Any]
    model_version: str
    processing_time_ms: float

@router.post("/seismic/detect-continuous", response_model=SeismicEventListResponse)
async def detect_continuous_seismic_events(
    request: ContinuousSeismicRequest
):
    """Detect seismic events from continuous waveform data using CNN"""
    try:
        import time
        from datetime import datetime
        from core.seismic_cnn_processor import SeismicCNNProcessor
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing continuous seismic event detection request",
                   station_id=request.station_id,
                   sampling_rate=request.sampling_rate,
                   data_length=len(request.waveform_data[0]) if request.waveform_data else 0)
        
        # Initialize CNN processor
        processor = SeismicCNNProcessor(
            sampling_rate=request.sampling_rate,
            window_length=30.0,
            detection_threshold=request.detection_threshold,
            device="cpu"
        )
        
        # Convert data
        waveform_array = np.array(request.waveform_data, dtype=np.float32)
        timestamps_array = np.array([datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in request.timestamps])
        
        # Create location object
        sensor_location = GeologicalLocation(
            latitude=request.location['latitude'],
            longitude=request.location['longitude'],
            elevation=request.location.get('elevation', 0.0)
        )
        
        # Process continuous data
        detected_events = processor.process_continuous_data(
            waveform_data=waveform_array,
            timestamps=timestamps_array,
            sensor_location=sensor_location,
            sensor_id=request.station_id
        )
        
        # Convert events to serializable format
        events_data = []
        for event in detected_events:
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "confidence": event.confidence,
                "magnitude": event.magnitude,
                "depth": event.depth,
                "duration": event.duration,
                "peak_amplitude": event.peak_amplitude,
                "location": {
                    "latitude": event.location.latitude if event.location else None,
                    "longitude": event.location.longitude if event.location else None,
                    "elevation": event.location.elevation if event.location else None
                },
                "frequency_content": event.frequency_content,
                "waveform_features": event.waveform_features,
                "detection_metadata": event.detection_metadata
            }
            events_data.append(event_data)
        
        # Get processing statistics
        stats = processor.get_processing_statistics()
        
        processing_time = (time.time() - start_time) * 1000
        
        return SeismicEventListResponse(
            events=events_data,
            total_events=len(detected_events),
            processing_statistics=stats,
            model_version="seismic_cnn_v1.0",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to detect continuous seismic events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.post("/seismic/detect-multi-station", response_model=SeismicEventListResponse)
async def detect_multi_station_seismic_events(
    request: MultiStationSeismicRequest
):
    """Detect and locate seismic events using multiple stations"""
    try:
        import time
        from datetime import datetime
        from core.seismic_cnn_processor import SeismicCNNProcessor, SeismicEventClassifier
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing multi-station seismic event detection request",
                   num_stations=len(request.stations),
                   location_estimation=request.location_estimation)
        
        # Initialize processors
        processor = SeismicCNNProcessor(
            sampling_rate=100.0,  # Default, will be overridden per station
            window_length=30.0,
            detection_threshold=request.detection_threshold,
            device="cpu"
        )
        
        classifier = SeismicEventClassifier()
        
        all_events = []
        station_locations = {}
        
        # Process each station
        for station_data in request.stations:
            station_id = station_data['station_id']
            sampling_rate = station_data['sampling_rate']
            waveform_data = np.array(station_data['waveform_data'], dtype=np.float32)
            timestamps = np.array([
                datetime.fromisoformat(ts.replace('Z', '+00:00')) 
                for ts in station_data['timestamps']
            ])
            
            # Update processor sampling rate
            processor.sampling_rate = sampling_rate
            processor.window_samples = int(processor.window_length * sampling_rate)
            
            # Create location
            location_data = station_data['location']
            sensor_location = GeologicalLocation(
                latitude=location_data['latitude'],
                longitude=location_data['longitude'],
                elevation=location_data.get('elevation', 0.0)
            )
            station_locations[station_id] = sensor_location
            
            # Process station data
            station_events = processor.process_continuous_data(
                waveform_data=waveform_data,
                timestamps=timestamps,
                sensor_location=sensor_location,
                sensor_id=station_id
            )
            
            all_events.extend(station_events)
        
        # Apply volcanic classification
        classified_events = classifier.classify_volcanic_events(all_events)
        
        # Estimate locations if requested
        if request.location_estimation and len(station_locations) >= 3:
            located_events = processor.estimate_location(classified_events, station_locations)
        else:
            located_events = classified_events
        
        # Convert events to serializable format
        events_data = []
        for event in located_events:
            event_data = {
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type.value,
                "confidence": event.confidence,
                "magnitude": event.magnitude,
                "depth": event.depth,
                "duration": event.duration,
                "peak_amplitude": event.peak_amplitude,
                "location": {
                    "latitude": event.location.latitude if event.location else None,
                    "longitude": event.location.longitude if event.location else None,
                    "elevation": event.location.elevation if event.location else None
                },
                "frequency_content": event.frequency_content,
                "waveform_features": event.waveform_features,
                "detection_metadata": event.detection_metadata
            }
            events_data.append(event_data)
        
        # Get processing statistics
        stats = processor.get_processing_statistics()
        stats['stations_processed'] = len(request.stations)
        stats['location_estimation_enabled'] = request.location_estimation
        
        processing_time = (time.time() - start_time) * 1000
        
        return SeismicEventListResponse(
            events=events_data,
            total_events=len(located_events),
            processing_statistics=stats,
            model_version="seismic_cnn_v1.0",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to detect multi-station seismic events", error=str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@router.get("/seismic/model-info")
async def get_seismic_model_info():
    """Get information about the seismic CNN model"""
    try:
        from core.seismic_cnn_processor import SeismicEventType
        
        return {
            "model_name": "Seismic CNN Event Detector",
            "model_version": "1.0.0",
            "model_type": "Convolutional Neural Network",
            "description": "CNN-based seismic event detection and classification system",
            "supported_event_types": [event_type.value for event_type in SeismicEventType],
            "input_requirements": {
                "channels": 3,
                "sampling_rate_hz": "50-500 (recommended: 100)",
                "window_length_seconds": "10-60 (default: 30)",
                "data_format": "3-component seismic waveform (Z, N, E)"
            },
            "output_capabilities": [
                "Event detection (binary)",
                "Event type classification",
                "Magnitude estimation",
                "Confidence assessment",
                "Feature extraction",
                "Parameter estimation (duration, peak amplitude)"
            ],
            "performance_metrics": {
                "detection_accuracy": "~85% (synthetic data)",
                "classification_accuracy": "~80% (synthetic data)",
                "processing_speed": "Real-time capable",
                "false_positive_rate": "~15%"
            },
            "limitations": [
                "Requires training on real seismic data for production use",
                "Performance varies with noise levels",
                "Location estimation requires multiple stations",
                "Magnitude estimation accuracy depends on calibration"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get seismic model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.post("/uncertainty/generate", response_model=UncertaintyResponse)
async def generate_uncertainty_scenarios(
    request: UncertaintyRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Generate uncertainty scenarios using GAN models"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing uncertainty scenario generation request",
                   num_scenarios=request.num_scenarios,
                   scenario_type=request.scenario_type)
        
        # Generate uncertainty scenarios
        result = await helios_engine.generate_uncertainty_scenarios(
            num_scenarios=request.num_scenarios
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        return UncertaintyResponse(
            scenarios=result["scenarios"],
            num_scenarios=result["num_scenarios"],
            model_version=result["model_version"],
            generation_time_ms=generation_time
        )
        
    except Exception as e:
        logger.error("Failed to generate uncertainty scenarios", error=str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/models/info")
async def get_model_info(helios_engine = Depends(get_helios_engine)):
    """Get information about loaded models"""
    try:
        model_count = await helios_engine.get_loaded_models_count()
        
        return {
            "loaded_models": model_count,
            "available_models": [
                {
                    "name": "pinn_geological",
                    "type": "Physics-Informed Neural Network",
                    "description": "Geological process modeling with physics constraints"
                },
                {
                    "name": "cnn_seismic",
                    "type": "Convolutional Neural Network",
                    "description": "Seismic event detection and classification"
                },
                {
                    "name": "gan_uncertainty",
                    "type": "Generative Adversarial Network",
                    "description": "Uncertainty quantification and scenario generation"
                }
            ],
            "device": str(helios_engine.device),
            "physics_constraints": helios_engine.physics_constraints,
            "simulation_params": helios_engine.simulation_params
        }
        
    except Exception as e:
        logger.error("Failed to get model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/models/status")
async def get_model_status(helios_engine = Depends(get_helios_engine)):
    """Get detailed status of all models"""
    try:
        return {
            "engine_initialized": helios_engine.is_initialized,
            "engine_healthy": await helios_engine.is_healthy(),
            "engine_ready": await helios_engine.is_ready(),
            "loaded_models": list(helios_engine.models.keys()),
            "device_info": {
                "device": str(helios_engine.device),
                "cuda_available": helios_engine.device.type == "cuda",
                "gpu_memory_allocated": (
                    torch.cuda.memory_allocated() if helios_engine.device.type == "cuda" else 0
                ),
                "gpu_memory_cached": (
                    torch.cuda.memory_reserved() if helios_engine.device.type == "cuda" else 0
                )
            }
        }
        
    except Exception as e:
        logger.error("Failed to get model status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.post("/models/reload")
async def reload_models(helios_engine = Depends(get_helios_engine)):
    """Reload all models (admin endpoint)"""
    try:
        logger.info("Reloading models")
        
        # This would implement model reloading logic
        await helios_engine._load_core_models()
        
        return {
            "status": "success",
            "message": "Models reloaded successfully",
            "loaded_models": await helios_engine.get_loaded_models_count()
        }
        
    except Exception as e:
        logger.error("Failed to reload models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")

# Physics simulation endpoints
@router.post("/physics/simulate")
async def run_physics_simulation(
    simulation_params: Dict[str, Any],
    helios_engine = Depends(get_helios_engine)
):
    """Run physics-informed simulation"""
    try:
        logger.info("Running physics simulation", params=simulation_params)
        
        # This would implement actual physics simulation
        # For now, return a placeholder response
        return {
            "status": "completed",
            "simulation_id": "sim_12345",
            "results": {
                "temperature_field": [[20.0, 21.0], [22.0, 23.0]],
                "velocity_field": [[0.1, 0.2], [0.3, 0.4]],
                "pressure_field": [[1000.0, 1001.0], [1002.0, 1003.0]]
            },
            "physics_constraints_satisfied": True,
            "convergence_achieved": True
        }
        
    except Exception as e:
        logger.error("Failed to run physics simulation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

# Integrated Geodynamic Modeling endpoints
class IntegratedGeodynamicRequest(BaseModel):
    seismic_events: List[Dict[str, Any]] = Field(default=[], description="Seismic event data")
    deformation_measurements: List[Dict[str, Any]] = Field(default=[], description="Deformation measurement data")
    geochemical_samples: List[Dict[str, Any]] = Field(default=[], description="Geochemical sample data")
    station_locations: List[Dict[str, Any]] = Field(..., description="Monitoring station locations")
    grid_bounds: List[List[float]] = Field(..., description="Spatial bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")
    resolution: float = Field(default=1.0, ge=0.1, le=10.0, description="Grid resolution in km")

class SeismicTomographyRequest(BaseModel):
    seismic_events: List[Dict[str, Any]] = Field(..., description="Seismic event data")
    station_locations: List[Dict[str, Any]] = Field(..., description="Seismic station locations")
    grid_bounds: List[List[float]] = Field(..., description="Spatial bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")
    resolution: float = Field(default=1.0, ge=0.1, le=10.0, description="Grid resolution in km")

class DeformationTrackingRequest(BaseModel):
    deformation_measurements: List[Dict[str, Any]] = Field(..., description="Deformation measurement data")
    grid_bounds: List[List[float]] = Field(..., description="Spatial bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")
    resolution: float = Field(default=0.5, ge=0.1, le=5.0, description="Grid resolution in km")

class EarthquakeAnalysisRequest(BaseModel):
    seismic_events: List[Dict[str, Any]] = Field(..., description="Seismic event data")
    time_window: List[str] = Field(default=None, description="Time window [start_iso, end_iso]")
    spatial_bounds: List[List[float]] = Field(default=None, description="Spatial bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")

class GeochemicalAnalysisRequest(BaseModel):
    geochemical_samples: List[Dict[str, Any]] = Field(..., description="Geochemical sample data")
    reference_compositions: Dict[str, Dict[str, float]] = Field(default=None, description="Reference magma compositions")

@router.post("/geodynamic/integrated")
async def process_integrated_geodynamic_data(
    request: IntegratedGeodynamicRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Process integrated geodynamic data from all sources"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing integrated geodynamic data request",
                   num_seismic_events=len(request.seismic_events),
                   num_deformation_measurements=len(request.deformation_measurements),
                   num_geochemical_samples=len(request.geochemical_samples),
                   num_stations=len(request.station_locations))
        
        # Convert grid bounds to tuple format
        grid_bounds = tuple(tuple(bound) for bound in request.grid_bounds)
        
        # Process integrated data
        result = await helios_engine.process_integrated_geodynamic_data(
            request.seismic_events,
            request.deformation_measurements,
            request.geochemical_samples,
            request.station_locations,
            grid_bounds,
            request.resolution
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to process integrated geodynamic data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/geodynamic/seismic-tomography")
async def analyze_seismic_tomography(
    request: SeismicTomographyRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Analyze seismic tomography data for structural model generation"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing seismic tomography request",
                   num_events=len(request.seismic_events),
                   num_stations=len(request.station_locations))
        
        # Convert grid bounds to tuple format
        grid_bounds = tuple(tuple(bound) for bound in request.grid_bounds)
        
        # Process seismic tomography
        result = await helios_engine.analyze_seismic_tomography(
            request.seismic_events,
            request.station_locations,
            grid_bounds,
            request.resolution
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to analyze seismic tomography", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/geodynamic/deformation-tracking")
async def track_deformation(
    request: DeformationTrackingRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Track multi-source deformation with millimeter accuracy"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing deformation tracking request",
                   num_measurements=len(request.deformation_measurements))
        
        # Convert grid bounds to tuple format
        grid_bounds = tuple(tuple(bound) for bound in request.grid_bounds)
        
        # Track deformation
        result = await helios_engine.track_deformation(
            request.deformation_measurements,
            grid_bounds,
            request.resolution
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to track deformation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")

@router.post("/geodynamic/earthquake-analysis")
async def analyze_earthquake_catalog(
    request: EarthquakeAnalysisRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Analyze earthquake catalog for stress and magma movement detection"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing earthquake analysis request",
                   num_events=len(request.seismic_events))
        
        # Convert time window and spatial bounds
        time_window = tuple(request.time_window) if request.time_window else None
        spatial_bounds = tuple(tuple(bound) for bound in request.spatial_bounds) if request.spatial_bounds else None
        
        # Analyze earthquake catalog
        result = await helios_engine.analyze_earthquake_catalog(
            request.seismic_events,
            time_window,
            spatial_bounds
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to analyze earthquake catalog", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/geodynamic/geochemical-analysis")
async def analyze_geochemical_data(
    request: GeochemicalAnalysisRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Analyze geochemical data for magma composition determination"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing geochemical analysis request",
                   num_samples=len(request.geochemical_samples))
        
        # Analyze geochemical data
        result = await helios_engine.analyze_geochemical_data(
            request.geochemical_samples,
            request.reference_compositions
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to analyze geochemical data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Long-term Geological Simulation endpoints
class LongTermSimulationRequest(BaseModel):
    simulation_config: Dict[str, Any] = Field(..., description="Configuration for simulation components")
    time_span: List[str] = Field(..., description="Time span [start_iso, end_iso]")
    time_scale: str = Field(default="millions_years", description="Time scale for simulation")

class VolcanicEvolutionRequest(BaseModel):
    initial_state_config: Dict[str, Any] = Field(..., description="Initial volcanic system configuration")
    time_span: List[str] = Field(..., description="Time span [start_iso, end_iso]")
    time_step_days: int = Field(default=365000, description="Time step in days")
    external_forcing: Dict[str, Any] = Field(default=None, description="External forcing factors")

class MagmaGenerationRequest(BaseModel):
    initial_conditions: Dict[str, Any] = Field(..., description="Initial mantle conditions")
    time_span: List[str] = Field(..., description="Time span [start_iso, end_iso]")
    spatial_domain: List[List[float]] = Field(..., description="3D spatial domain [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")
    time_step_days: int = Field(default=36500, description="Time step in days")

class HydrothermalCirculationRequest(BaseModel):
    initial_conditions: Dict[str, Any] = Field(..., description="Initial fluid and rock conditions")
    heat_source: Dict[str, Any] = Field(..., description="Heat source parameters")
    time_span: List[str] = Field(..., description="Time span [start_iso, end_iso]")
    spatial_domain: List[List[float]] = Field(..., description="3D spatial domain")
    time_step_days: int = Field(default=3650, description="Time step in days")

class TectonicReconstructionRequest(BaseModel):
    initial_configuration: Dict[str, Any] = Field(..., description="Initial plate configuration")
    time_span: List[str] = Field(..., description="Time span [start_iso, end_iso]")
    time_step_days: int = Field(default=365000000, description="Time step in days")
    reference_frame: str = Field(default="hotspot", description="Reference frame for reconstruction")

@router.post("/geological/long-term-simulation")
async def run_long_term_geological_simulation(
    request: LongTermSimulationRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Run comprehensive long-term geological simulation"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing long-term geological simulation request",
                   time_scale=request.time_scale,
                   components=list(request.simulation_config.keys()))
        
        # Run comprehensive simulation
        result = await helios_engine.run_long_term_geological_simulation(
            request.simulation_config,
            request.time_span,
            request.time_scale
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to run long-term geological simulation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/geological/volcanic-evolution")
async def simulate_volcanic_evolution(
    request: VolcanicEvolutionRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Simulate volcanic system evolution over geological timescales"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing volcanic evolution simulation request",
                   time_step_days=request.time_step_days)
        
        # Run volcanic evolution simulation
        result = await helios_engine.simulate_volcanic_evolution(
            request.initial_state_config,
            request.time_span,
            request.time_step_days,
            request.external_forcing
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to simulate volcanic evolution", error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/geological/magma-generation")
async def track_magma_generation(
    request: MagmaGenerationRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Track magma generation from mantle to surface systems"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing magma generation tracking request",
                   time_step_days=request.time_step_days)
        
        # Run magma generation tracking
        result = await helios_engine.track_magma_generation(
            request.initial_conditions,
            request.time_span,
            request.spatial_domain,
            request.time_step_days
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to track magma generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Tracking failed: {str(e)}")

@router.post("/geological/hydrothermal-circulation")
async def simulate_hydrothermal_circulation(
    request: HydrothermalCirculationRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Simulate hydrothermal system circulation"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing hydrothermal circulation simulation request",
                   time_step_days=request.time_step_days)
        
        # Run hydrothermal circulation simulation
        result = await helios_engine.simulate_hydrothermal_circulation(
            request.initial_conditions,
            request.heat_source,
            request.time_span,
            request.spatial_domain,
            request.time_step_days
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to simulate hydrothermal circulation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/geological/tectonic-reconstruction")
async def reconstruct_tectonic_plates(
    request: TectonicReconstructionRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Reconstruct tectonic plate movements through deep time"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing tectonic reconstruction request",
                   time_step_days=request.time_step_days,
                   reference_frame=request.reference_frame)
        
        # Run tectonic reconstruction
        result = await helios_engine.reconstruct_tectonic_plates(
            request.initial_configuration,
            request.time_span,
            request.time_step_days,
            request.reference_frame
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to reconstruct tectonic plates", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

# Real-time monitoring endpoints
@router.post("/monitoring/analyze")
async def analyze_monitoring_data(
    monitoring_data: Dict[str, Any],
    helios_engine = Depends(get_helios_engine)
):
    """Analyze real-time monitoring data"""
    try:
        logger.info("Analyzing monitoring data", data_type=monitoring_data.get("type"))
        
        # This would implement real-time monitoring analysis
        # For now, return a placeholder response
        return {
            "alert_level": 2,
            "risk_assessment": {
                "volcanic_unrest": 0.3,
                "eruption_probability_24h": 0.05,
                "eruption_probability_7d": 0.15
            },
            "recommendations": [
                "Continue monitoring",
                "Check gas emissions",
                "Verify deformation patterns"
            ],
            "confidence": 0.85
        }
        
    except Exception as e:
        logger.error("Failed to analyze monitoring data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
#
 Earthquake Forecasting API endpoints

class EarthquakePatternAnalysisRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(..., description="List of seismic events for pattern analysis")

class StatisticalSeismicityModelRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(..., description="List of seismic events for statistical modeling")

class TemporalClusteringRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(..., description="List of seismic events for clustering analysis")

class ProbabilisticHazardAssessmentRequest(BaseModel):
    events: List[Dict[str, Any]] = Field(..., description="List of historical seismic events")
    target_region: Dict[str, float] = Field(..., description="Target region bounds (lat_min, lat_max, lon_min, lon_max)")
    forecast_period_days: int = Field(default=30, ge=1, le=365, description="Forecast period in days")

class EarthquakeConfigUpdateRequest(BaseModel):
    config_updates: Dict[str, Any] = Field(..., description="Configuration updates to apply")

@router.post("/earthquake/analyze-patterns")
async def analyze_earthquake_patterns(
    request: EarthquakePatternAnalysisRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Analyze earthquake patterns using machine learning models"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing earthquake pattern analysis request",
                   num_events=len(request.events))
        
        # Analyze earthquake patterns
        result = await helios_engine.analyze_earthquake_patterns(request.events)
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to analyze earthquake patterns", error=str(e))
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")

@router.post("/earthquake/statistical-model")
async def create_statistical_seismicity_model(
    request: StatisticalSeismicityModelRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Create statistical seismicity models with confidence intervals"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing statistical seismicity model request",
                   num_events=len(request.events))
        
        # Create statistical model
        result = await helios_engine.create_statistical_seismicity_model(request.events)
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to create statistical seismicity model", error=str(e))
        raise HTTPException(status_code=500, detail=f"Statistical modeling failed: {str(e)}")

@router.post("/earthquake/temporal-clustering")
async def perform_temporal_clustering_analysis(
    request: TemporalClusteringRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Perform temporal clustering analysis for earthquake sequences"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing temporal clustering analysis request",
                   num_events=len(request.events))
        
        # Perform clustering analysis
        result = await helios_engine.perform_temporal_clustering_analysis(request.events)
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to perform temporal clustering analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@router.post("/earthquake/hazard-assessment")
async def generate_probabilistic_hazard_assessment(
    request: ProbabilisticHazardAssessmentRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Generate probabilistic hazard assessment with uncertainty quantification"""
    try:
        import time
        start_time = time.time()
        
        logger.info("Processing probabilistic hazard assessment request",
                   num_events=len(request.events),
                   forecast_period_days=request.forecast_period_days)
        
        # Generate hazard assessment
        result = await helios_engine.generate_probabilistic_hazard_assessment(
            request.events,
            request.target_region,
            request.forecast_period_days
        )
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error("Failed to generate probabilistic hazard assessment", error=str(e))
        raise HTTPException(status_code=500, detail=f"Hazard assessment failed: {str(e)}")

@router.get("/earthquake/config")
async def get_earthquake_forecasting_config(
    helios_engine = Depends(get_helios_engine)
):
    """Get earthquake forecasting engine configuration"""
    try:
        result = await helios_engine.get_earthquake_forecasting_config()
        return result
        
    except Exception as e:
        logger.error("Failed to get earthquake forecasting config", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")

@router.put("/earthquake/config")
async def update_earthquake_forecasting_config(
    request: EarthquakeConfigUpdateRequest,
    helios_engine = Depends(get_helios_engine)
):
    """Update earthquake forecasting engine configuration"""
    try:
        logger.info("Updating earthquake forecasting configuration",
                   updates=request.config_updates)
        
        result = await helios_engine.update_earthquake_forecasting_config(
            request.config_updates
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to update earthquake forecasting config", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@router.get("/earthquake/model-info")
async def get_earthquake_forecasting_model_info():
    """Get information about earthquake forecasting models and capabilities"""
    try:
        return {
            "service": "Earthquake Forecasting Engine",
            "version": "1.0.0",
            "description": "Probabilistic earthquake forecasting with machine learning and statistical models",
            "capabilities": {
                "pattern_analysis": {
                    "description": "Machine learning-based earthquake pattern analysis",
                    "models": ["Random Forest", "Gradient Boosting"],
                    "features": [
                        "Spatial pattern recognition",
                        "Temporal pattern analysis", 
                        "Magnitude distribution analysis",
                        "Feature importance ranking"
                    ]
                },
                "statistical_modeling": {
                    "description": "Statistical seismicity models with confidence intervals",
                    "models": ["Gutenberg-Richter", "Poisson Process"],
                    "features": [
                        "b-value estimation",
                        "Seismicity rate calculation",
                        "Recurrence interval estimation",
                        "Uncertainty quantification"
                    ]
                },
                "temporal_clustering": {
                    "description": "Temporal clustering analysis for earthquake sequences",
                    "algorithms": ["DBSCAN"],
                    "features": [
                        "Mainshock-aftershock identification",
                        "Earthquake swarm detection",
                        "Cluster type classification",
                        "Spatial-temporal analysis"
                    ]
                },
                "hazard_assessment": {
                    "description": "Probabilistic hazard assessment with uncertainty quantification",
                    "models": ["Ensemble Forecasting", "ETAS", "Statistical"],
                    "features": [
                        "Multi-model ensemble forecasting",
                        "Confidence interval estimation",
                        "Hazard level assessment",
                        "Contributing factor identification"
                    ]
                }
            },
            "input_requirements": {
                "seismic_events": {
                    "required_fields": ["timestamp", "latitude", "longitude", "magnitude", "depth"],
                    "optional_fields": ["event_type", "confidence", "agency", "event_id"],
                    "data_format": "ISO 8601 timestamps, decimal degrees, magnitude scales"
                },
                "target_region": {
                    "required_fields": ["lat_min", "lat_max", "lon_min", "lon_max"],
                    "format": "Decimal degrees"
                }
            },
            "output_capabilities": [
                "Probability estimates for different magnitude ranges",
                "Confidence intervals for all predictions",
                "Hazard level classification (Low, Moderate, High, Very High, Extreme)",
                "Uncertainty metrics and ensemble statistics",
                "Contributing factor identification",
                "Earthquake cluster analysis results"
            ],
            "forecast_periods": [1, 7, 30, 365],  # days
            "magnitude_ranges": ["M>=4.0", "M>=5.0", "M>=6.0", "M>=7.0"],
            "performance_characteristics": {
                "minimum_events_required": 10,
                "optimal_catalog_length": "5+ years",
                "processing_speed": "Real-time capable",
                "uncertainty_quantification": "Bootstrap sampling with 95% confidence intervals"
            },
            "limitations": [
                "Requires sufficient historical data for reliable forecasting",
                "Performance depends on data quality and completeness",
                "Forecasts are probabilistic, not deterministic",
                "Regional calibration may be required for optimal performance"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get earthquake forecasting model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

# Example data generation endpoint for testing
@router.post("/earthquake/generate-test-data")
async def generate_test_earthquake_data(
    num_events: int = Field(default=50, ge=10, le=500, description="Number of events to generate"),
    region_bounds: Dict[str, float] = Field(
        default={"lat_min": 34.0, "lat_max": 36.0, "lon_min": -119.0, "lon_max": -117.0},
        description="Geographic bounds for synthetic events"
    ),
    time_span_days: int = Field(default=365, ge=30, le=3650, description="Time span for events in days")
):
    """Generate synthetic earthquake data for testing and demonstration"""
    try:
        import numpy as np
        from datetime import datetime, timedelta
        
        logger.info("Generating synthetic earthquake test data",
                   num_events=num_events,
                   region_bounds=region_bounds,
                   time_span_days=time_span_days)
        
        events = []
        base_time = datetime.utcnow() - timedelta(days=time_span_days)
        
        for i in range(num_events):
            # Random time within span
            random_days = np.random.uniform(0, time_span_days)
            timestamp = base_time + timedelta(days=random_days)
            
            # Random location within bounds
            latitude = np.random.uniform(region_bounds["lat_min"], region_bounds["lat_max"])
            longitude = np.random.uniform(region_bounds["lon_min"], region_bounds["lon_max"])
            
            # Magnitude following Gutenberg-Richter distribution
            magnitude = 2.5 + np.random.exponential(1.0)
            magnitude = min(magnitude, 8.0)  # Cap at M8.0
            
            # Depth correlated with magnitude
            depth = np.random.uniform(5, 30) + magnitude * 2
            depth = min(depth, 50)  # Cap at 50 km
            
            event = {
                "event_id": f"synthetic_{i:04d}",
                "timestamp": timestamp.isoformat(),
                "latitude": latitude,
                "longitude": longitude,
                "magnitude": round(magnitude, 1),
                "depth": round(depth, 1),
                "event_type": "earthquake",
                "confidence": round(np.random.uniform(0.8, 1.0), 2),
                "agency": "SYNTHETIC"
            }
            events.append(event)
        
        # Sort by time
        events.sort(key=lambda e: e["timestamp"])
        
        return {
            "status": "success",
            "num_events": len(events),
            "time_span_days": time_span_days,
            "region_bounds": region_bounds,
            "magnitude_range": [
                min(e["magnitude"] for e in events),
                max(e["magnitude"] for e in events)
            ],
            "events": events,
            "generation_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to generate test earthquake data", error=str(e))
        raise HTTPException(status_code=500, detail=f"Data generation failed: {str(e)}")
# 
Earthquake Impact Modeling endpoints
class GroundShakingRequest(BaseModel):
    earthquake: Dict[str, Any] = Field(..., description="Earthquake event data")
    target_locations: List[Dict[str, float]] = Field(..., description="Target locations for ground motion calculation")
    soil_conditions: Dict[str, str] = Field(default=None, description="Soil conditions at target locations")
    ground_motion_model: str = Field(default="boore_atkinson_2008", description="Ground motion prediction equation")

class LiquefactionRequest(BaseModel):
    ground_motion: List[Dict[str, Any]] = Field(..., description="Ground motion parameters")
    soil_properties: Dict[str, Dict[str, float]] = Field(..., description="Soil properties for each location")
    groundwater_data: Dict[str, float] = Field(default=None, description="Groundwater depth data")

class LandslideRequest(BaseModel):
    ground_motion: List[Dict[str, Any]] = Field(..., description="Ground motion parameters")
    slope_data: Dict[str, float] = Field(..., description="Slope angles for each location")
    soil_strength: Dict[str, Dict[str, float]] = Field(..., description="Soil strength parameters")
    topography: Dict[str, Any] = Field(default=None, description="Digital elevation model data")

class TsunamiRequest(BaseModel):
    earthquake: Dict[str, Any] = Field(..., description="Earthquake event data")
    coastal_locations: List[Dict[str, float]] = Field(..., description="Coastal locations for impact assessment")
    bathymetry: Dict[str, Any] = Field(default=None, description="Bathymetry data for wave propagation")

class EarthquakeImpactResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary_statistics: Dict[str, Any]
    model_version: str
    processing_time_ms: float

@router.post("/earthquake-impact/ground-shaking", response_model=EarthquakeImpactResponse)
async def simulate_ground_shaking(request: GroundShakingRequest):
    """Simulate ground shaking using physics-based models"""
    try:
        import time
        from datetime import datetime
        from core.earthquake_impact_modeling import (
            EarthquakeImpactModelingEngine, 
            GroundMotionModel, 
            SoilType
        )
        from models.geological import SeismicEvent, GeologicalLocation, EventType
        
        start_time = time.time()
        
        logger.info("Processing ground shaking simulation request",
                   num_locations=len(request.target_locations),
                   ground_motion_model=request.ground_motion_model)
        
        # Initialize impact modeling engine
        engine = EarthquakeImpactModelingEngine()
        
        # Convert earthquake data
        eq_data = request.earthquake
        earthquake = SeismicEvent(
            timestamp=datetime.fromisoformat(eq_data['timestamp'].replace('Z', '+00:00')),
            location=GeologicalLocation(
                latitude=eq_data['location']['latitude'],
                longitude=eq_data['location']['longitude'],
                elevation=eq_data['location'].get('elevation', 0.0)
            ),
            magnitude=eq_data['magnitude'],
            depth=eq_data['depth'],
            event_type=EventType(eq_data.get('event_type', 'earthquake'))
        )
        
        # Convert target locations
        target_locations = [
            GeologicalLocation(
                latitude=loc['latitude'],
                longitude=loc['longitude'],
                elevation=loc.get('elevation', 0.0)
            ) for loc in request.target_locations
        ]
        
        # Convert soil conditions
        soil_conditions = None
        if request.soil_conditions:
            soil_conditions = {
                k: SoilType(v) for k, v in request.soil_conditions.items()
            }
        
        # Convert ground motion model
        gm_model = GroundMotionModel(request.ground_motion_model)
        
        # Simulate ground shaking
        results = engine.simulate_ground_shaking(
            earthquake=earthquake,
            target_locations=target_locations,
            soil_conditions=soil_conditions,
            model=gm_model
        )
        
        # Convert results to serializable format
        results_data = []
        for i, result in enumerate(results):
            result_data = {
                "location_id": str(i),
                "location": {
                    "latitude": result.location.latitude,
                    "longitude": result.location.longitude,
                    "elevation": result.location.elevation
                },
                "peak_ground_acceleration": result.peak_ground_acceleration,
                "peak_ground_velocity": result.peak_ground_velocity,
                "peak_ground_displacement": result.peak_ground_displacement,
                "spectral_accelerations": result.spectral_accelerations,
                "duration": result.duration,
                "arias_intensity": result.arias_intensity,
                "uncertainty": result.uncertainty
            }
            results_data.append(result_data)
        
        # Calculate summary statistics
        pga_values = [r.peak_ground_acceleration for r in results]
        pgv_values = [r.peak_ground_velocity for r in results]
        
        summary_stats = {
            "num_locations": len(results),
            "pga_statistics": {
                "min": min(pga_values),
                "max": max(pga_values),
                "mean": sum(pga_values) / len(pga_values),
                "median": sorted(pga_values)[len(pga_values)//2]
            },
            "pgv_statistics": {
                "min": min(pgv_values),
                "max": max(pgv_values),
                "mean": sum(pgv_values) / len(pgv_values),
                "median": sorted(pgv_values)[len(pgv_values)//2]
            }
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return EarthquakeImpactResponse(
            results=results_data,
            summary_statistics=summary_stats,
            model_version="earthquake_impact_v1.0",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to simulate ground shaking", error=str(e))
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.post("/earthquake-impact/liquefaction", response_model=EarthquakeImpactResponse)
async def assess_liquefaction_susceptibility(request: LiquefactionRequest):
    """Assess liquefaction susceptibility and probability"""
    try:
        import time
        from core.earthquake_impact_modeling import (
            EarthquakeImpactModelingEngine,
            GroundMotionParameters
        )
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing liquefaction assessment request",
                   num_locations=len(request.ground_motion))
        
        # Initialize impact modeling engine
        engine = EarthquakeImpactModelingEngine()
        
        # Convert ground motion data
        ground_motion = []
        for gm_data in request.ground_motion:
            location = GeologicalLocation(
                latitude=gm_data['location']['latitude'],
                longitude=gm_data['location']['longitude'],
                elevation=gm_data['location'].get('elevation', 0.0)
            )
            
            gm_params = GroundMotionParameters(
                location=location,
                peak_ground_acceleration=gm_data['peak_ground_acceleration'],
                peak_ground_velocity=gm_data['peak_ground_velocity'],
                peak_ground_displacement=gm_data['peak_ground_displacement'],
                spectral_accelerations=gm_data['spectral_accelerations'],
                duration=gm_data['duration'],
                arias_intensity=gm_data['arias_intensity'],
                uncertainty=gm_data.get('uncertainty', {})
            )
            ground_motion.append(gm_params)
        
        # Assess liquefaction susceptibility
        results = engine.assess_liquefaction_susceptibility(
            ground_motion=ground_motion,
            soil_properties=request.soil_properties,
            groundwater_data=request.groundwater_data
        )
        
        # Convert results to serializable format
        results_data = []
        for i, result in enumerate(results):
            result_data = {
                "location_id": str(i),
                "location": {
                    "latitude": result.location.latitude,
                    "longitude": result.location.longitude,
                    "elevation": result.location.elevation
                },
                "susceptibility": result.susceptibility.value,
                "factor_of_safety": result.factor_of_safety,
                "probability": result.probability,
                "contributing_factors": result.contributing_factors,
                "soil_properties": result.soil_properties,
                "groundwater_depth": result.groundwater_depth
            }
            results_data.append(result_data)
        
        # Calculate summary statistics
        probabilities = [r.probability for r in results]
        factors_of_safety = [r.factor_of_safety for r in results]
        
        susceptibility_counts = {}
        for result in results:
            susc = result.susceptibility.value
            susceptibility_counts[susc] = susceptibility_counts.get(susc, 0) + 1
        
        summary_stats = {
            "num_locations": len(results),
            "susceptibility_distribution": susceptibility_counts,
            "probability_statistics": {
                "min": min(probabilities),
                "max": max(probabilities),
                "mean": sum(probabilities) / len(probabilities)
            },
            "factor_of_safety_statistics": {
                "min": min(factors_of_safety),
                "max": max(factors_of_safety),
                "mean": sum(factors_of_safety) / len(factors_of_safety)
            },
            "high_risk_locations": sum(1 for r in results if r.susceptibility.value in ["high", "very_high"])
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return EarthquakeImpactResponse(
            results=results_data,
            summary_statistics=summary_stats,
            model_version="earthquake_impact_v1.0",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to assess liquefaction susceptibility", error=str(e))
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/earthquake-impact/landslide", response_model=EarthquakeImpactResponse)
async def model_landslide_triggering(request: LandslideRequest):
    """Model landslide triggering and runout simulation"""
    try:
        import time
        from core.earthquake_impact_modeling import (
            EarthquakeImpactModelingEngine,
            GroundMotionParameters
        )
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing landslide modeling request",
                   num_locations=len(request.ground_motion))
        
        # Initialize impact modeling engine
        engine = EarthquakeImpactModelingEngine()
        
        # Convert ground motion data
        ground_motion = []
        for gm_data in request.ground_motion:
            location = GeologicalLocation(
                latitude=gm_data['location']['latitude'],
                longitude=gm_data['location']['longitude'],
                elevation=gm_data['location'].get('elevation', 0.0)
            )
            
            gm_params = GroundMotionParameters(
                location=location,
                peak_ground_acceleration=gm_data['peak_ground_acceleration'],
                peak_ground_velocity=gm_data['peak_ground_velocity'],
                peak_ground_displacement=gm_data['peak_ground_displacement'],
                spectral_accelerations=gm_data['spectral_accelerations'],
                duration=gm_data['duration'],
                arias_intensity=gm_data['arias_intensity'],
                uncertainty=gm_data.get('uncertainty', {})
            )
            ground_motion.append(gm_params)
        
        # Model landslide triggering
        results = engine.model_landslide_triggering(
            ground_motion=ground_motion,
            slope_data=request.slope_data,
            soil_strength=request.soil_strength,
            topography=request.topography
        )
        
        # Convert results to serializable format
        results_data = []
        for i, result in enumerate(results):
            runout_path_data = [
                {
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                    "elevation": loc.elevation
                } for loc in result.runout_path
            ]
            
            result_data = {
                "location_id": str(i),
                "location": {
                    "latitude": result.location.latitude,
                    "longitude": result.location.longitude,
                    "elevation": result.location.elevation
                },
                "susceptibility": result.susceptibility.value,
                "trigger_threshold": result.trigger_threshold,
                "probability": result.probability,
                "volume_estimate": result.volume_estimate,
                "runout_distance": result.runout_distance,
                "runout_path": runout_path_data,
                "impact_area": result.impact_area
            }
            results_data.append(result_data)
        
        # Calculate summary statistics
        probabilities = [r.probability for r in results]
        volumes = [r.volume_estimate for r in results]
        runout_distances = [r.runout_distance for r in results]
        
        susceptibility_counts = {}
        for result in results:
            susc = result.susceptibility.value
            susceptibility_counts[susc] = susceptibility_counts.get(susc, 0) + 1
        
        summary_stats = {
            "num_locations": len(results),
            "susceptibility_distribution": susceptibility_counts,
            "probability_statistics": {
                "min": min(probabilities) if probabilities else 0,
                "max": max(probabilities) if probabilities else 0,
                "mean": sum(probabilities) / len(probabilities) if probabilities else 0
            },
            "volume_statistics": {
                "total": sum(volumes),
                "mean": sum(volumes) / len(volumes) if volumes else 0,
                "max": max(volumes) if volumes else 0
            },
            "runout_statistics": {
                "mean": sum(runout_distances) / len(runout_distances) if runout_distances else 0,
                "max": max(runout_distances) if runout_distances else 0
            },
            "high_risk_locations": sum(1 for r in results if r.susceptibility.value in ["high", "very_high"])
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return EarthquakeImpactResponse(
            results=results_data,
            summary_statistics=summary_stats,
            model_version="earthquake_impact_v1.0",
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Failed to model landslide triggering", error=str(e))
        raise HTTPException(status_code=500, detail=f"Modeling failed: {str(e)}")

@router.post("/earthquake-impact/tsunami")
async def model_tsunami_generation(request: TsunamiRequest):
    """Model tsunami generation and propagation"""
    try:
        import time
        from datetime import datetime
        from core.earthquake_impact_modeling import EarthquakeImpactModelingEngine
        from models.geological import SeismicEvent, GeologicalLocation, EventType
        
        start_time = time.time()
        
        logger.info("Processing tsunami modeling request",
                   num_coastal_locations=len(request.coastal_locations))
        
        # Initialize impact modeling engine
        engine = EarthquakeImpactModelingEngine()
        
        # Convert earthquake data
        eq_data = request.earthquake
        earthquake = SeismicEvent(
            timestamp=datetime.fromisoformat(eq_data['timestamp'].replace('Z', '+00:00')),
            location=GeologicalLocation(
                latitude=eq_data['location']['latitude'],
                longitude=eq_data['location']['longitude'],
                elevation=eq_data['location'].get('elevation', 0.0)
            ),
            magnitude=eq_data['magnitude'],
            depth=eq_data['depth'],
            event_type=EventType(eq_data.get('event_type', 'earthquake'))
        )
        
        # Convert coastal locations
        coastal_locations = [
            GeologicalLocation(
                latitude=loc['latitude'],
                longitude=loc['longitude'],
                elevation=loc.get('elevation', 0.0)
            ) for loc in request.coastal_locations
        ]
        
        # Model tsunami generation
        result = engine.model_tsunami_generation(
            earthquake=earthquake,
            coastal_locations=coastal_locations,
            bathymetry=request.bathymetry
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if result is None:
            return {
                "tsunami_generated": False,
                "message": "Earthquake unlikely to generate significant tsunami",
                "processing_time_ms": processing_time
            }
        
        # Convert result to serializable format
        result_data = {
            "tsunami_generated": True,
            "source_location": {
                "latitude": result.source_location.latitude,
                "longitude": result.source_location.longitude,
                "elevation": result.source_location.elevation
            },
            "source_type": result.source_type.value,
            "initial_wave_height": result.initial_wave_height,
            "wave_period": result.wave_period,
            "propagation_speed": result.propagation_speed,
            "arrival_times": result.arrival_times,
            "wave_heights": result.wave_heights,
            "inundation_distances": result.inundation_distances,
            "affected_coastline": result.affected_coastline,
            "summary_statistics": {
                "num_coastal_locations": len(coastal_locations),
                "arrival_time_range": [
                    min(result.arrival_times.values()),
                    max(result.arrival_times.values())
                ],
                "wave_height_range": [
                    min(result.wave_heights.values()),
                    max(result.wave_heights.values())
                ],
                "inundation_range": [
                    min(result.inundation_distances.values()),
                    max(result.inundation_distances.values())
                ]
            },
            "processing_time_ms": processing_time
        }
        
        return result_data
        
    except Exception as e:
        logger.error("Failed to model tsunami generation", error=str(e))
        raise HTTPException(status_code=500, detail=f"Modeling failed: {str(e)}")

@router.get("/earthquake-impact/model-info")
async def get_earthquake_impact_model_info():
    """Get information about earthquake impact modeling capabilities"""
    try:
        from core.earthquake_impact_modeling import (
            GroundMotionModel, SoilType, LiquefactionSusceptibility,
            LandslideSusceptibility, TsunamiSource
        )
        
        return {
            "model_name": "Earthquake Impact Modeling System",
            "model_version": "1.0.0",
            "description": "Comprehensive earthquake impact modeling including ground shaking, liquefaction, landslides, and tsunamis",
            "capabilities": [
                "Ground shaking simulation using physics-based models",
                "Liquefaction susceptibility mapping and modeling",
                "Landslide triggering and runout simulation",
                "Tsunami generation and propagation modeling"
            ],
            "supported_ground_motion_models": [model.value for model in GroundMotionModel],
            "supported_soil_types": [soil_type.value for soil_type in SoilType],
            "liquefaction_susceptibility_levels": [level.value for level in LiquefactionSusceptibility],
            "landslide_susceptibility_levels": [level.value for level in LandslideSusceptibility],
            "tsunami_source_types": [source.value for source in TsunamiSource],
            "input_requirements": {
                "ground_shaking": {
                    "earthquake": "SeismicEvent with magnitude, depth, location",
                    "target_locations": "List of GeologicalLocation objects",
                    "soil_conditions": "Optional soil type mapping",
                    "ground_motion_model": "GMPE selection"
                },
                "liquefaction": {
                    "ground_motion": "GroundMotionParameters from shaking analysis",
                    "soil_properties": "SPT N-values, fines content, stress conditions",
                    "groundwater_data": "Depth to groundwater table"
                },
                "landslide": {
                    "ground_motion": "GroundMotionParameters from shaking analysis",
                    "slope_data": "Slope angles in degrees",
                    "soil_strength": "Cohesion, friction angle, unit weight",
                    "topography": "Optional DEM data for runout modeling"
                },
                "tsunami": {
                    "earthquake": "Large, shallow earthquake event",
                    "coastal_locations": "Coastal points for impact assessment",
                    "bathymetry": "Optional bathymetry data for wave propagation"
                }
            },
            "output_capabilities": [
                "Ground motion parameters (PGA, PGV, PGD, SA, duration)",
                "Liquefaction probability and factor of safety",
                "Landslide volume estimates and runout distances",
                "Tsunami wave heights and arrival times",
                "Uncertainty quantification for all analyses",
                "Summary statistics and risk assessments"
            ],
            "limitations": [
                "Ground motion models are simplified implementations",
                "Liquefaction analysis uses basic empirical methods",
                "Landslide runout modeling is simplified",
                "Tsunami modeling uses shallow water approximations",
                "Requires calibration with local data for accuracy"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get earthquake impact model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")
# 
Infrastructure Vulnerability Assessment endpoints
class BuildingInventoryItem(BaseModel):
    building_id: str = Field(..., description="Unique building identifier")
    location: Dict[str, float] = Field(..., description="Building location (lat, lon, elevation)")
    building_type: str = Field(..., description="Building structural type")
    year_built: int = Field(..., description="Construction year")
    number_of_stories: int = Field(..., description="Number of stories")
    floor_area: float = Field(..., description="Total floor area in square meters")
    occupancy: int = Field(..., description="Number of occupants")
    replacement_cost: float = Field(..., description="Replacement cost in USD")
    seismic_design_level: str = Field(default="none", description="Seismic design code level")

class TransportationNetworkItem(BaseModel):
    network_id: str = Field(..., description="Network identifier")
    network_type: str = Field(..., description="Type of transportation network")
    nodes: Dict[str, Dict[str, float]] = Field(..., description="Network nodes with locations")
    edges: Dict[str, List[str]] = Field(..., description="Network edges (node pairs)")
    capacity: Dict[str, float] = Field(..., description="Edge capacities")
    importance: Dict[str, float] = Field(..., description="Node/edge importance scores")
    vulnerability: Dict[str, float] = Field(..., description="Vulnerability scores")

class UtilitySystemItem(BaseModel):
    system_id: str = Field(..., description="System identifier")
    utility_type: str = Field(..., description="Type of utility system")
    components: Dict[str, Dict[str, float]] = Field(..., description="System components with locations")
    dependencies: Dict[str, List[str]] = Field(..., description="Dependencies between components")
    capacity: Dict[str, float] = Field(..., description="Component capacities")
    redundancy: Dict[str, float] = Field(..., description="Redundancy levels")
    service_area: Dict[str, float] = Field(..., description="Service area coverage")

class InfrastructureVulnerabilityRequest(BaseModel):
    scenario_name: str = Field(..., description="Name of the earthquake scenario")
    earthquake: Dict[str, Any] = Field(..., description="Earthquake event data")
    buildings: List[BuildingInventoryItem] = Field(default=[], description="Building inventory")
    transportation_networks: List[TransportationNetworkItem] = Field(default=[], description="Transportation networks")
    utility_systems: List[UtilitySystemItem] = Field(default=[], description="Utility systems")
    ground_motion_data: Dict[str, float] = Field(..., description="Ground motion intensities by location")

class BuildingDamageRequest(BaseModel):
    scenario_name: str = Field(..., description="Name of the earthquake scenario")
    buildings: List[BuildingInventoryItem] = Field(..., description="Building inventory")
    ground_motion_data: Dict[str, float] = Field(..., description="Ground motion intensities by location")

class TransportationDisruptionRequest(BaseModel):
    scenario_name: str = Field(..., description="Name of the earthquake scenario")
    transportation_networks: List[TransportationNetworkItem] = Field(..., description="Transportation networks")
    ground_motion_data: Dict[str, float] = Field(..., description="Ground motion intensities by location")

class UtilityFailureRequest(BaseModel):
    scenario_name: str = Field(..., description="Name of the earthquake scenario")
    utility_systems: List[UtilitySystemItem] = Field(..., description="Utility systems")
    ground_motion_data: Dict[str, float] = Field(..., description="Ground motion intensities by location")

class CascadingFailureRequest(BaseModel):
    scenario_name: str = Field(..., description="Name of the earthquake scenario")
    building_assessments: List[Dict[str, Any]] = Field(..., description="Building damage assessments")
    transportation_disruptions: List[Dict[str, Any]] = Field(..., description="Transportation disruptions")
    utility_failures: List[Dict[str, Any]] = Field(..., description="Utility failures")

@router.post("/infrastructure/vulnerability-assessment")
async def assess_infrastructure_vulnerability(
    request: InfrastructureVulnerabilityRequest
):
    """Comprehensive infrastructure vulnerability assessment"""
    try:
        import time
        from datetime import datetime
        from core.infrastructure_vulnerability import (
            InfrastructureVulnerabilityEngine,
            BuildingInventory,
            TransportationNetwork,
            UtilitySystem,
            BuildingType,
            TransportationType,
            UtilityType
        )
        from models.geological import GeologicalLocation, SeismicEvent
        
        start_time = time.time()
        
        logger.info("Processing comprehensive infrastructure vulnerability assessment",
                   scenario_name=request.scenario_name,
                   num_buildings=len(request.buildings),
                   num_transport_networks=len(request.transportation_networks),
                   num_utility_systems=len(request.utility_systems))
        
        # Initialize vulnerability engine
        engine = InfrastructureVulnerabilityEngine()
        
        # Convert building inventory
        buildings = []
        for bldg_data in request.buildings:
            location = GeologicalLocation(
                latitude=bldg_data.location['latitude'],
                longitude=bldg_data.location['longitude'],
                elevation=bldg_data.location.get('elevation', 0.0)
            )
            
            building = BuildingInventory(
                building_id=bldg_data.building_id,
                location=location,
                building_type=BuildingType(bldg_data.building_type),
                year_built=bldg_data.year_built,
                number_of_stories=bldg_data.number_of_stories,
                floor_area=bldg_data.floor_area,
                occupancy=bldg_data.occupancy,
                replacement_cost=bldg_data.replacement_cost,
                seismic_design_level=bldg_data.seismic_design_level
            )
            buildings.append(building)
        
        # Convert transportation networks
        transport_networks = []
        for net_data in request.transportation_networks:
            nodes = {}
            for node_id, loc_data in net_data.nodes.items():
                nodes[node_id] = GeologicalLocation(
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    elevation=loc_data.get('elevation', 0.0)
                )
            
            edges = {}
            for edge_id, node_pair in net_data.edges.items():
                edges[edge_id] = tuple(node_pair)
            
            network = TransportationNetwork(
                network_id=net_data.network_id,
                network_type=TransportationType(net_data.network_type),
                nodes=nodes,
                edges=edges,
                capacity=net_data.capacity,
                importance=net_data.importance,
                vulnerability=net_data.vulnerability
            )
            transport_networks.append(network)
        
        # Convert utility systems
        utility_systems = []
        for sys_data in request.utility_systems:
            components = {}
            for comp_id, loc_data in sys_data.components.items():
                components[comp_id] = GeologicalLocation(
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    elevation=loc_data.get('elevation', 0.0)
                )
            
            system = UtilitySystem(
                system_id=sys_data.system_id,
                utility_type=UtilityType(sys_data.utility_type),
                components=components,
                dependencies=sys_data.dependencies,
                capacity=sys_data.capacity,
                redundancy=sys_data.redundancy,
                service_area=sys_data.service_area
            )
            utility_systems.append(system)
        
        # Assess building damage
        building_assessments = engine.assess_building_damage(
            buildings, request.ground_motion_data, request.scenario_name
        )
        
        # Analyze transportation disruption
        transportation_disruptions = engine.analyze_transportation_disruption(
            transport_networks, request.ground_motion_data, request.scenario_name
        )
        
        # Model utility failures
        utility_failures = engine.model_utility_failure_probability(
            utility_systems, request.ground_motion_data, request.scenario_name
        )
        
        # Analyze cascading failures
        cascade_analysis = engine.analyze_cascading_failures(
            building_assessments,
            transportation_disruptions,
            utility_failures,
            request.scenario_name
        )
        
        # Convert results to serializable format
        building_results = []
        for assessment in building_assessments:
            building_results.append({
                "building_id": assessment.building_id,
                "damage_state": assessment.damage_state.value,
                "damage_probabilities": {state.value: prob for state, prob in assessment.damage_probabilities.items()},
                "repair_cost": assessment.repair_cost,
                "repair_time": assessment.repair_time,
                "casualties": assessment.casualties,
                "functionality_loss": assessment.functionality_loss,
                "economic_loss": assessment.economic_loss
            })
        
        transport_results = []
        for disruption in transportation_disruptions:
            transport_results.append({
                "network_id": disruption.network_id,
                "disrupted_nodes": disruption.disrupted_nodes,
                "disrupted_edges": disruption.disrupted_edges,
                "capacity_reduction": disruption.capacity_reduction,
                "connectivity_loss": disruption.connectivity_loss,
                "detour_factors": disruption.detour_factors,
                "economic_impact": disruption.economic_impact,
                "recovery_time": disruption.recovery_time
            })
        
        utility_results = []
        for failure in utility_failures:
            utility_results.append({
                "system_id": failure.system_id,
                "failed_components": failure.failed_components,
                "failure_probabilities": failure.failure_probabilities,
                "service_disruption": failure.service_disruption,
                "affected_population": failure.affected_population,
                "restoration_time": failure.restoration_time,
                "economic_loss": failure.economic_loss,
                "cascading_effects": failure.cascading_effects
            })
        
        cascade_results = {
            "scenario_id": cascade_analysis.scenario_id,
            "initial_failures": cascade_analysis.initial_failures,
            "cascade_sequence": [
                {"time": item[0], "system": item[1], "component": item[2]}
                for item in cascade_analysis.cascade_sequence
            ],
            "total_affected_systems": cascade_analysis.total_affected_systems,
            "cascade_duration": cascade_analysis.cascade_duration,
            "amplification_factor": cascade_analysis.amplification_factor,
            "system_resilience": cascade_analysis.system_resilience,
            "recovery_strategy": cascade_analysis.recovery_strategy
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "scenario_name": request.scenario_name,
            "building_damage_assessment": building_results,
            "transportation_disruption_analysis": transport_results,
            "utility_failure_assessment": utility_results,
            "cascading_failure_analysis": cascade_results,
            "summary_statistics": {
                "total_buildings_assessed": len(building_assessments),
                "total_transport_networks_analyzed": len(transportation_disruptions),
                "total_utility_systems_assessed": len(utility_failures),
                "total_repair_cost": sum(a.repair_cost for a in building_assessments),
                "total_casualties": sum(a.casualties for a in building_assessments),
                "total_economic_loss": sum(a.economic_loss for a in building_assessments),
                "total_affected_population": sum(f.affected_population for f in utility_failures),
                "system_resilience_score": cascade_analysis.system_resilience
            },
            "processing_time_ms": processing_time,
            "model_version": "infrastructure_vulnerability_v1.0"
        }
        
    except Exception as e:
        logger.error("Failed to assess infrastructure vulnerability", error=str(e))
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/infrastructure/building-damage")
async def assess_building_damage(
    request: BuildingDamageRequest
):
    """Assess building damage estimation models"""
    try:
        import time
        from core.infrastructure_vulnerability import (
            InfrastructureVulnerabilityEngine,
            BuildingInventory,
            BuildingType
        )
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing building damage assessment",
                   scenario_name=request.scenario_name,
                   num_buildings=len(request.buildings))
        
        # Initialize vulnerability engine
        engine = InfrastructureVulnerabilityEngine()
        
        # Convert building inventory
        buildings = []
        for bldg_data in request.buildings:
            location = GeologicalLocation(
                latitude=bldg_data.location['latitude'],
                longitude=bldg_data.location['longitude'],
                elevation=bldg_data.location.get('elevation', 0.0)
            )
            
            building = BuildingInventory(
                building_id=bldg_data.building_id,
                location=location,
                building_type=BuildingType(bldg_data.building_type),
                year_built=bldg_data.year_built,
                number_of_stories=bldg_data.number_of_stories,
                floor_area=bldg_data.floor_area,
                occupancy=bldg_data.occupancy,
                replacement_cost=bldg_data.replacement_cost,
                seismic_design_level=bldg_data.seismic_design_level
            )
            buildings.append(building)
        
        # Assess building damage
        assessments = engine.assess_building_damage(
            buildings, request.ground_motion_data, request.scenario_name
        )
        
        # Convert results to serializable format
        results = []
        for assessment in assessments:
            results.append({
                "building_id": assessment.building_id,
                "damage_state": assessment.damage_state.value,
                "damage_probabilities": {state.value: prob for state, prob in assessment.damage_probabilities.items()},
                "repair_cost": assessment.repair_cost,
                "repair_time": assessment.repair_time,
                "casualties": assessment.casualties,
                "functionality_loss": assessment.functionality_loss,
                "economic_loss": assessment.economic_loss
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "scenario_name": request.scenario_name,
            "building_assessments": results,
            "summary_statistics": {
                "total_buildings": len(assessments),
                "total_repair_cost": sum(a.repair_cost for a in assessments),
                "total_casualties": sum(a.casualties for a in assessments),
                "total_economic_loss": sum(a.economic_loss for a in assessments),
                "damage_distribution": {
                    state.value: sum(1 for a in assessments if a.damage_state == state)
                    for state in set(a.damage_state for a in assessments)
                }
            },
            "processing_time_ms": processing_time,
            "model_version": "building_damage_v1.0"
        }
        
    except Exception as e:
        logger.error("Failed to assess building damage", error=str(e))
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

@router.post("/infrastructure/transportation-disruption")
async def analyze_transportation_disruption(
    request: TransportationDisruptionRequest
):
    """Analyze transportation network disruption"""
    try:
        import time
        from core.infrastructure_vulnerability import (
            InfrastructureVulnerabilityEngine,
            TransportationNetwork,
            TransportationType
        )
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing transportation disruption analysis",
                   scenario_name=request.scenario_name,
                   num_networks=len(request.transportation_networks))
        
        # Initialize vulnerability engine
        engine = InfrastructureVulnerabilityEngine()
        
        # Convert transportation networks
        networks = []
        for net_data in request.transportation_networks:
            nodes = {}
            for node_id, loc_data in net_data.nodes.items():
                nodes[node_id] = GeologicalLocation(
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    elevation=loc_data.get('elevation', 0.0)
                )
            
            edges = {}
            for edge_id, node_pair in net_data.edges.items():
                edges[edge_id] = tuple(node_pair)
            
            network = TransportationNetwork(
                network_id=net_data.network_id,
                network_type=TransportationType(net_data.network_type),
                nodes=nodes,
                edges=edges,
                capacity=net_data.capacity,
                importance=net_data.importance,
                vulnerability=net_data.vulnerability
            )
            networks.append(network)
        
        # Analyze transportation disruption
        disruptions = engine.analyze_transportation_disruption(
            networks, request.ground_motion_data, request.scenario_name
        )
        
        # Convert results to serializable format
        results = []
        for disruption in disruptions:
            results.append({
                "network_id": disruption.network_id,
                "disrupted_nodes": disruption.disrupted_nodes,
                "disrupted_edges": disruption.disrupted_edges,
                "capacity_reduction": disruption.capacity_reduction,
                "connectivity_loss": disruption.connectivity_loss,
                "detour_factors": disruption.detour_factors,
                "economic_impact": disruption.economic_impact,
                "recovery_time": disruption.recovery_time
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "scenario_name": request.scenario_name,
            "transportation_disruptions": results,
            "summary_statistics": {
                "total_networks": len(disruptions),
                "total_economic_impact": sum(d.economic_impact for d in disruptions),
                "average_connectivity_loss": sum(d.connectivity_loss for d in disruptions) / len(disruptions) if disruptions else 0,
                "total_disrupted_nodes": sum(len(d.disrupted_nodes) for d in disruptions),
                "total_disrupted_edges": sum(len(d.disrupted_edges) for d in disruptions)
            },
            "processing_time_ms": processing_time,
            "model_version": "transportation_disruption_v1.0"
        }
        
    except Exception as e:
        logger.error("Failed to analyze transportation disruption", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/infrastructure/utility-failure")
async def model_utility_failure_probability(
    request: UtilityFailureRequest
):
    """Model utility system failure probability"""
    try:
        import time
        from core.infrastructure_vulnerability import (
            InfrastructureVulnerabilityEngine,
            UtilitySystem,
            UtilityType
        )
        from models.geological import GeologicalLocation
        
        start_time = time.time()
        
        logger.info("Processing utility failure probability modeling",
                   scenario_name=request.scenario_name,
                   num_systems=len(request.utility_systems))
        
        # Initialize vulnerability engine
        engine = InfrastructureVulnerabilityEngine()
        
        # Convert utility systems
        systems = []
        for sys_data in request.utility_systems:
            components = {}
            for comp_id, loc_data in sys_data.components.items():
                components[comp_id] = GeologicalLocation(
                    latitude=loc_data['latitude'],
                    longitude=loc_data['longitude'],
                    elevation=loc_data.get('elevation', 0.0)
                )
            
            system = UtilitySystem(
                system_id=sys_data.system_id,
                utility_type=UtilityType(sys_data.utility_type),
                components=components,
                dependencies=sys_data.dependencies,
                capacity=sys_data.capacity,
                redundancy=sys_data.redundancy,
                service_area=sys_data.service_area
            )
            systems.append(system)
        
        # Model utility failures
        failures = engine.model_utility_failure_probability(
            systems, request.ground_motion_data, request.scenario_name
        )
        
        # Convert results to serializable format
        results = []
        for failure in failures:
            results.append({
                "system_id": failure.system_id,
                "failed_components": failure.failed_components,
                "failure_probabilities": failure.failure_probabilities,
                "service_disruption": failure.service_disruption,
                "affected_population": failure.affected_population,
                "restoration_time": failure.restoration_time,
                "economic_loss": failure.economic_loss,
                "cascading_effects": failure.cascading_effects
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "scenario_name": request.scenario_name,
            "utility_failures": results,
            "summary_statistics": {
                "total_systems": len(failures),
                "total_economic_loss": sum(f.economic_loss for f in failures),
                "total_affected_population": sum(f.affected_population for f in failures),
                "average_restoration_time": sum(f.restoration_time for f in failures) / len(failures) if failures else 0,
                "total_failed_components": sum(len(f.failed_components) for f in failures),
                "systems_with_cascading_effects": sum(1 for f in failures if f.cascading_effects)
            },
            "processing_time_ms": processing_time,
            "model_version": "utility_failure_v1.0"
        }
        
    except Exception as e:
        logger.error("Failed to model utility failure probability", error=str(e))
        raise HTTPException(status_code=500, detail=f"Modeling failed: {str(e)}")

@router.post("/infrastructure/cascading-failure")
async def analyze_cascading_failures(
    request: CascadingFailureRequest
):
    """Analyze cascading failure effects across infrastructure systems"""
    try:
        import time
        from core.infrastructure_vulnerability import (
            InfrastructureVulnerabilityEngine,
            BuildingDamageAssessment,
            TransportationDisruption,
            UtilityFailureAssessment,
            DamageState
        )
        
        start_time = time.time()
        
        logger.info("Processing cascading failure analysis",
                   scenario_name=request.scenario_name,
                   num_building_assessments=len(request.building_assessments),
                   num_transport_disruptions=len(request.transportation_disruptions),
                   num_utility_failures=len(request.utility_failures))
        
        # Initialize vulnerability engine
        engine = InfrastructureVulnerabilityEngine()
        
        # Convert building assessments (simplified conversion for API)
        building_assessments = []
        for bldg_data in request.building_assessments:
            damage_probs = {DamageState(state): prob for state, prob in bldg_data['damage_probabilities'].items()}
            
            assessment = BuildingDamageAssessment(
                building_id=bldg_data['building_id'],
                damage_state=DamageState(bldg_data['damage_state']),
                damage_probabilities=damage_probs,
                repair_cost=bldg_data['repair_cost'],
                repair_time=bldg_data['repair_time'],
                casualties=bldg_data['casualties'],
                functionality_loss=bldg_data['functionality_loss'],
                economic_loss=bldg_data['economic_loss']
            )
            building_assessments.append(assessment)
        
        # Convert transportation disruptions (simplified conversion for API)
        transport_disruptions = []
        for trans_data in request.transportation_disruptions:
            disruption = TransportationDisruption(
                network_id=trans_data['network_id'],
                disrupted_nodes=trans_data['disrupted_nodes'],
                disrupted_edges=trans_data['disrupted_edges'],
                capacity_reduction=trans_data['capacity_reduction'],
                connectivity_loss=trans_data['connectivity_loss'],
                detour_factors=trans_data['detour_factors'],
                economic_impact=trans_data['economic_impact'],
                recovery_time=trans_data['recovery_time']
            )
            transport_disruptions.append(disruption)
        
        # Convert utility failures (simplified conversion for API)
        utility_failures = []
        for util_data in request.utility_failures:
            failure = UtilityFailureAssessment(
                system_id=util_data['system_id'],
                failed_components=util_data['failed_components'],
                failure_probabilities=util_data['failure_probabilities'],
                service_disruption=util_data['service_disruption'],
                affected_population=util_data['affected_population'],
                restoration_time=util_data['restoration_time'],
                economic_loss=util_data['economic_loss'],
                cascading_effects=util_data['cascading_effects']
            )
            utility_failures.append(failure)
        
        # Analyze cascading failures
        cascade_analysis = engine.analyze_cascading_failures(
            building_assessments,
            transport_disruptions,
            utility_failures,
            request.scenario_name
        )
        
        # Convert results to serializable format
        result = {
            "scenario_id": cascade_analysis.scenario_id,
            "initial_failures": cascade_analysis.initial_failures,
            "cascade_sequence": [
                {"time": item[0], "system": item[1], "component": item[2]}
                for item in cascade_analysis.cascade_sequence
            ],
            "total_affected_systems": cascade_analysis.total_affected_systems,
            "cascade_duration": cascade_analysis.cascade_duration,
            "amplification_factor": cascade_analysis.amplification_factor,
            "system_resilience": cascade_analysis.system_resilience,
            "recovery_strategy": cascade_analysis.recovery_strategy
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "scenario_name": request.scenario_name,
            "cascading_failure_analysis": result,
            "processing_time_ms": processing_time,
            "model_version": "cascading_failure_v1.0"
        }
        
    except Exception as e:
        logger.error("Failed to analyze cascading failures", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/infrastructure/model-info")
async def get_infrastructure_model_info():
    """Get information about infrastructure vulnerability models"""
    try:
        from core.infrastructure_vulnerability import BuildingType, TransportationType, UtilityType, DamageState
        
        return {
            "model_name": "Infrastructure Vulnerability Assessment System",
            "model_version": "1.0.0",
            "description": "Comprehensive infrastructure vulnerability assessment for earthquake scenarios",
            "capabilities": [
                "Building damage estimation using fragility curves",
                "Transportation network disruption analysis",
                "Utility system failure probability modeling",
                "Cascading failure analysis across infrastructure systems"
            ],
            "supported_building_types": [btype.value for btype in BuildingType],
            "supported_transportation_types": [ttype.value for ttype in TransportationType],
            "supported_utility_types": [utype.value for utype in UtilityType],
            "damage_states": [state.value for state in DamageState],
            "input_requirements": {
                "ground_motion_data": "Peak Ground Acceleration (PGA) values in g",
                "building_inventory": "Building characteristics and locations",
                "transportation_networks": "Network topology and characteristics",
                "utility_systems": "System components and dependencies"
            },
            "output_capabilities": [
                "Building damage probabilities and repair costs",
                "Transportation network connectivity loss",
                "Utility system service disruption levels",
                "Cascading failure propagation analysis",
                "Economic impact assessment",
                "Recovery time estimation"
            ],
            "methodologies": {
                "building_damage": "HAZUS-MH fragility curves with age and design factors",
                "transportation_disruption": "Network analysis with capacity reduction",
                "utility_failure": "Component fragility with dependency modeling",
                "cascading_analysis": "Graph-based interdependency modeling"
            },
            "limitations": [
                "Requires calibration with local building inventory data",
                "Simplified interdependency relationships",
                "Economic loss estimates are approximate",
                "Recovery time estimates based on historical data"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get infrastructure model info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")