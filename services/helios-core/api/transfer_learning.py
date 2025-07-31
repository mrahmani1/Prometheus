"""
Transfer Learning API Endpoints

This module provides REST API endpoints for domain adaptation, few-shot learning,
and model fine-tuning capabilities of the Helios AI engine.

Requirements addressed: 5.1, 5.2
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import torch
import numpy as np
import logging
from datetime import datetime
import asyncio
import json
import io

from ..core.transfer_learning_engine import (
    TransferLearningEngine, GeologicalDomain, AdaptationConfig, AdaptationResult
)
from ..core.domain_adaptation_utils import (
    GeologicalDomainRegistry, DomainDataPreprocessor, AdaptationStrategySelector,
    GeologicalSetting, TectonicContext, DataCharacteristics
)
from ..core.helios_engine import HeliosEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/transfer-learning", tags=["transfer-learning"])

# Global instances
transfer_engine = TransferLearningEngine()
domain_registry = GeologicalDomainRegistry()
data_preprocessor = DomainDataPreprocessor()
strategy_selector = AdaptationStrategySelector()
helios_engine = HeliosEngine()


# Pydantic models for API
class GeologicalDomainRequest(BaseModel):
    domain_id: str = Field(..., description="Unique identifier for the domain")
    region_name: str = Field(..., description="Human-readable region name")
    geological_setting: str = Field(..., description="Type of geological setting")
    tectonic_context: str = Field(..., description="Tectonic context")
    data_characteristics: Dict[str, Any] = Field(..., description="Data characteristics")
    available_data_types: List[str] = Field(..., description="Available data types")
    data_sparsity_level: float = Field(..., ge=0.0, le=1.0, description="Data sparsity level")


class AdaptationConfigRequest(BaseModel):
    target_domain_id: str = Field(..., description="Target domain ID")
    adaptation_method: Optional[str] = Field(None, description="Adaptation method (auto-select if None)")
    learning_rate: float = Field(1e-4, gt=0, description="Learning rate")
    num_epochs: int = Field(100, gt=0, description="Number of training epochs")
    batch_size: int = Field(32, gt=0, description="Batch size")
    regularization_strength: float = Field(0.01, ge=0, description="Regularization strength")
    few_shot_samples: int = Field(10, gt=0, description="Number of few-shot samples")
    validation_split: float = Field(0.2, gt=0, lt=1, description="Validation split ratio")


class AdaptationRequest(BaseModel):
    source_model_id: str = Field(..., description="Source model identifier")
    config: AdaptationConfigRequest = Field(..., description="Adaptation configuration")
    use_few_shot: bool = Field(False, description="Use few-shot learning")


class DomainSimilarityRequest(BaseModel):
    target_domain_id: str = Field(..., description="Target domain ID")
    top_k: int = Field(3, gt=0, le=10, description="Number of similar domains to return")


class ModelValidationRequest(BaseModel):
    adapted_model_id: str = Field(..., description="Adapted model ID")
    validation_domain_id: str = Field(..., description="Domain for validation")


# API Endpoints

@router.get("/domains", response_model=List[str])
async def list_domains():
    """List all registered geological domains"""
    try:
        domains = domain_registry.list_domains()
        logger.info(f"Retrieved {len(domains)} registered domains")
        return domains
    except Exception as e:
        logger.error(f"Error listing domains: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/domains/{domain_id}")
async def get_domain(domain_id: str):
    """Get details of a specific geological domain"""
    try:
        domain = domain_registry.get_domain(domain_id)
        if not domain:
            raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
        
        return {
            "domain_id": domain.domain_id,
            "region_name": domain.region_name,
            "geological_setting": domain.geological_setting,
            "tectonic_context": domain.tectonic_context,
            "data_characteristics": domain.data_characteristics,
            "available_data_types": domain.available_data_types,
            "data_sparsity_level": domain.data_sparsity_level
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving domain {domain_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/domains")
async def register_domain(domain_request: GeologicalDomainRequest):
    """Register a new geological domain"""
    try:
        # Validate geological setting and tectonic context
        try:
            geological_setting = GeologicalSetting(domain_request.geological_setting)
            tectonic_context = TectonicContext(domain_request.tectonic_context)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid geological setting or tectonic context: {str(e)}")
        
        # Create data characteristics object
        data_char = DataCharacteristics(**domain_request.data_characteristics)
        
        # Register domain
        domain_registry.register_domain(
            domain_id=domain_request.domain_id,
            region_name=domain_request.region_name,
            geological_setting=geological_setting,
            tectonic_context=tectonic_context,
            data_characteristics=data_char,
            available_data_types=domain_request.available_data_types,
            data_sparsity_level=domain_request.data_sparsity_level
        )
        
        logger.info(f"Successfully registered domain: {domain_request.domain_id}")
        return {"message": f"Domain {domain_request.domain_id} registered successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering domain: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/domains/{domain_id}/similarity")
async def find_similar_domains(domain_id: str, request: DomainSimilarityRequest):
    """Find domains similar to the specified domain"""
    try:
        similar_domains = domain_registry.get_similar_domains(
            request.target_domain_id, request.top_k
        )
        
        if not similar_domains:
            
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import asyncio
from datetime import datetime, timedelta
import uuid

class AdvancedGeologicalAPI:
    """Advanced API endpoints for geological data and analysis"""
    
    def __init__(self):
        self.app = FastAPI(
            title="Prometheus Geological Analysis API",
            description="Advanced geological simulation and monitoring API",
            version="2.0.0"
        )
        self.security = HTTPBearer()
        self._setup_routes()
        self._setup_middleware()
    
    def _setup_middleware(self):
        """Setup API middleware for logging, CORS, rate limiting"""
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/v2/seismic/analyze", response_model=SeismicAnalysisResponse)
        async def analyze_seismic_data(
            request: SeismicAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced seismic data analysis endpoint"""
            
            # Validate authentication
            user_id = await self._validate_token(credentials.credentials)
            
            # Generate analysis ID
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize seismic processor
                processor = AdvancedSeismicProcessor(sampling_rate=request.sampling_rate)
                
                # Process seismic data
                analysis_result = processor.process_seismic_stream(
                    data=np.array(request.waveform_data),
                    station_info=request.station_info.dict()
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat(),
                    'processing_time_ms': 0  # Would be calculated
                })
                
                # Store results asynchronously
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'seismic'
                )
                
                return SeismicAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.post("/api/v2/volcanic/monitor", response_model=VolcanicMonitoringResponse)
        async def monitor_volcanic_activity(
            request: VolcanicMonitoringRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Comprehensive volcanic monitoring endpoint"""
            
            user_id = await self._validate_token(credentials.credentials)
            monitoring_id = str(uuid.uuid4())
            
            try:
                # Initialize volcanic monitor
                monitor = AdvancedVolcanicMonitor()
                
                # Perform comprehensive assessment
                assessment = monitor.comprehensive_volcanic_assessment(
                    monitoring_data=request.monitoring_data,
                    volcano_info=request.volcano_info.dict()
                )
                
                # Add metadata
                assessment.update({
                    'monitoring_id': monitoring_id,
                    'user_id': user_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results and trigger alerts if necessary
                background_tasks.add_task(
                    self._process_volcanic_assessment,
                    monitoring_id, assessment
                )
                
                return VolcanicMonitoringResponse(**assessment)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Monitoring failed: {str(e)}")
        
        @self.app.post("/api/v2/geochemical/analyze", response_model=GeochemicalAnalysisResponse)
        async def analyze_geochemical_composition(
            request: GeochemicalAnalysisRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Advanced geochemical composition analysis"""
            
            user_id = await self._validate_token(credentials.credentials)
            analysis_id = str(uuid.uuid4())
            
            try:
                # Initialize geochemical analyzer
                analyzer = AdvancedGeochemicalAnalyzer()
                
                # Perform analysis
                analysis_result = analyzer.analyze_rock_composition(
                    composition=request.composition,
                    temperature=request.temperature,
                    pressure=request.pressure
                )
                
                # Add metadata
                analysis_result.update({
                    'analysis_id': analysis_id,
                    'user_id': user_id,
                    'sample_id': request.sample_id,
                    'request_timestamp': datetime.now().isoformat()
                })
                
                # Store results
                background_tasks.add_task(
                    self._store_analysis_results,
                    analysis_id, analysis_result, 'geochemical'
                )
                
                return GeochemicalAnalysisResponse(**analysis_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Geochemical analysis failed: {str(e)}")
        
        @self.app.get("/api/v2/analysis/{analysis_id}", response_model=AnalysisStatusResponse)
        async def get_analysis_status(
            analysis_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Get analysis status and results"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Retrieve analysis from database/cache
                analysis_data = await self._get_analysis_results(analysis_id, user_id)
                
                if not analysis_data:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                return AnalysisStatusResponse(**analysis_data)
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")
        
        @self.app.get("/api/v2/spatial/query", response_model=SpatialQueryResponse)
        async def spatial_geological_query(
            bbox: str = Query(..., description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
            start_time: str = Query(..., description="Start time (ISO format)"),
            end_time: str = Query(..., description="End time (ISO format)"),
            data_types: List[str] = Query(["seismic"], description="Data types to include"),
            min_magnitude: float = Query(0.0, description="Minimum magnitude"),
            max_magnitude: float = Query(10.0, description="Maximum magnitude"),
            limit: int = Query(1000, le=10000, description="Maximum results"),
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """Spatial query for geological data"""
            
            user_id = await self._validate_token(credentials.credentials)
            
            try:
                # Parse bounding box
                bbox_coords = [float(x) for x in bbox.split(',')]
                if len(bbox_coords) != 4:
                    raise ValueError("Invalid bounding box format")
                
                # Initialize database
                db = AdvancedGeologicalDatabase({
                    'postgres': {'host': 'localhost', 'database': 'geological_data'},
                    'influxdb': {'url': 'http://localhost:8086'}
                })
                
                # Perform spatial query
                results = db.query_spatial_events(
                    bbox=tuple(bbox_coords),
                    time_range=(start_time, end_time),
                    magnitude_range=(min_magnitude, max_magnitude)
                )
                
                # Limit results
                results = results[:limit]
                
                return SpatialQueryResponse(
                    query_id=str(uuid.uuid4()),
                    user_id=user_id,
                    bbox=bbox_coords,
                    time_range=[start_time, end_time],
                    result_count=len(results),
                    results=results,
                    query_timestamp=datetime.now().isoformat()
                )
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Spatial query failed: {str(e)}")
    
    async def _validate_token(self, token: str) -> str:
        """Validate JWT token and return user ID"""
        # Simplified token validation - implement proper JWT validation
        if not token or len(token) < 10:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Extract user ID from token (simplified)
        return f"user_{hash(token) % 10000}"
    
    async def _store_analysis_results(self, analysis_id: str, results: Dict[str, Any], analysis_type: str):
        """Store analysis results in database"""
        try:
            # Initialize database connection
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            # Cache results
            db.cache_analysis_results(analysis_id, results, expiration_hours=24)
            
            logger.info(f"Stored {analysis_type} analysis results for {analysis_id}")
            
        except Exception as e:
            logger.error(f"Failed to store analysis results: {e}")
    
    async def _get_analysis_results(self, analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from database"""
        try:
            db = AdvancedGeologicalDatabase({
                'redis': {'host': 'localhost', 'port': 6379}
            })
            
            results = db.get_cached_results(analysis_id)
            
            # Verify user has access to this analysis
            if results and results.get('user_id') == user_id:
                return results
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve analysis results: {e}")
            return None

# Pydantic models for request/response validation
class StationInfo(BaseModel):
    station_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: float
    network: str

class SeismicAnalysisRequest(BaseModel):
    waveform_data: List[List[float]]
    sampling_rate: float = Field(..., gt=0, le=1000)
    station_info: StationInfo
    analysis_parameters: Optional[Dict[str, Any]] = {}

class SeismicAnalysisResponse(BaseModel):
    analysis_id: str
    station_id: str
    processing_timestamp: str
    components: List[Dict[str, Any]]
    cross_component_analysis: Dict[str, Any]
    overall_quality: float

class VolcanicMonitoringRequest(BaseModel):
    volcano_info: Dict[str, Any]
    monitoring_data: Dict[str, Any]
    analysis_parameters: Optional[Dict[str, Any]] = {}

class VolcanicMonitoringResponse(BaseModel):
    monitoring_id: str
    volcano_id: str
    assessment_timestamp: str
    current_alert_level: str
    eruption_probability: Dict[str, float]
    recommendations: List[str]

        
        # Format response with domain details
        result = []
        for similar_id, similarity_score in similar_domains:
            domain = domain_registry.get_domain(similar_id)
            if domain:
                result.append({
                    "domain_id": similar_id,
                    "region_name": domain.region_name,
                    "similarity_score": similarity_score,
                    "geological_setting": domain.geological_setting,
                    "data_sparsity_level": domain.data_sparsity_level
                })
        
        logger.info(f"Found {len(result)} similar domains for {request.target_domain_id}")
        return {"similar_domains": result}
        
    except Exception as e:
        logger.error(f"Error finding similar domains: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptation/strategy-recommendation")
async def get_adaptation_strategy_recommendation(domain_id: str):
    """Get adaptation strategy recommendations for a domain"""
    try:
        domain = domain_registry.get_domain(domain_id)
        if not domain:
            raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
        
        # Get strategy recommendations
        recommendations = strategy_selector.get_strategy_recommendations(domain)
        
        # Get general adaptation recommendations
        general_recommendations = transfer_engine.get_adaptation_recommendations(domain)
        
        return {
            "domain_id": domain_id,
            "region_name": domain.region_name,
            "strategy_recommendations": recommendations,
            "general_recommendations": general_recommendations,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/adaptation/adapt")
async def adapt_model(request: AdaptationRequest, background_tasks: BackgroundTasks):
    """Adapt a model to a target geological domain"""
    try:
        # Get target domain
        target_domain = domain_registry.get_domain(request.config.target_domain_id)
        if not target_domain:
            raise HTTPException(
                status_code=404, 
                detail=f"Target domain {request.config.target_domain_id} not found"
            )
        
        # Load source model (simplified - would load from model registry)
        source_model = helios_engine.get_pinn_model()  # Get current PINN model
        
        # Create adaptation configuration
        adaptation_config = AdaptationConfig(
            source_domain=None,  # Would be determined from model metadata
            target_domain=target_domain,
            adaptation_method=request.config.adaptation_method or strategy_selector.select_strategy(target_domain),
            learning_rate=request.config.learning_rate,
            num_epochs=request.config.num_epochs,
            batch_size=request.config.batch_size,
            regularization_strength=request.config.regularization_strength,
            few_shot_samples=request.config.few_shot_samples,
            validation_split=request.config.validation_split
        )
        
        # Generate dummy target data for demonstration
        # In practice, this would come from the data ingestion service
        target_data = _generate_dummy_target_data(target_domain, request.config.batch_size)
        
        # Start adaptation process in background
        adaptation_id = f"adaptation_{request.config.target_domain_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        background_tasks.add_task(
            _run_adaptation_task,
            adaptation_id,
            source_model,
            target_domain,
            target_data,
            adaptation_config,
            request.use_few_shot
        )
        
        logger.info(f"Started adaptation task: {adaptation_id}")
        return {
            "adaptation_id": adaptation_id,
            "status": "started",
            "target_domain": request.config.target_domain_id,
            "adaptation_method": adaptation_config.adaptation_method,
            "message": "Adaptation process started in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting adaptation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptation/{adaptation_id}/status")
async def get_adaptation_status(adaptation_id: str):
    """Get status of an adaptation process"""
    try:
        # In practice, this would check a task queue or database
        # For now, return a mock status
        return {
            "adaptation_id": adaptation_id,
            "status": "completed",  # or "running", "failed"
            "progress": 100,
            "estimated_completion": None,
            "current_epoch": 100,
            "total_epochs": 100,
            "current_loss": 0.0234,
            "message": "Adaptation completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error getting adaptation status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validation/validate-model")
async def validate_adapted_model(request: ModelValidationRequest):
    """Validate an adapted model's performance"""
    try:
        # Get validation domain
        validation_domain = domain_registry.get_domain(request.validation_domain_id)
        if not validation_domain:
            raise HTTPException(
                status_code=404,
                detail=f"Validation domain {request.validation_domain_id} not found"
            )
        
        # Load adapted model (simplified)
        adapted_model = helios_engine.get_pinn_model()  # Would load specific adapted model
        original_model = helios_engine.get_pinn_model()  # Would load original model
        
        # Generate validation data
        validation_data = _generate_dummy_target_data(validation_domain, 32)
        
        # Perform validation
        validation_scores = transfer_engine.validator.validate_adapted_model(
            adapted_model, validation_data, original_model, validation_domain
        )
        
        logger.info(f"Model validation completed for {request.adapted_model_id}")
        return {
            "adapted_model_id": request.adapted_model_id,
            "validation_domain": request.validation_domain_id,
            "validation_scores": validation_scores,
            "timestamp": datetime.utcnow().isoformat(),
            "validation_summary": {
                "overall_performance": "good" if validation_scores.get('mse_improvement', 0) > 0 else "needs_improvement",
                "physics_consistency": validation_scores.get('physics_consistency', 0),
                "geological_plausibility": validation_scores.get('geological_plausibility', 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/upload-domain-data")
async def upload_domain_data(
    domain_id: str,
    data_file: UploadFile = File(...),
    data_type: str = "geological_samples"
):
    """Upload training data for a specific domain"""
    try:
        # Validate domain exists
        domain = domain_registry.get_domain(domain_id)
        if not domain:
            raise HTTPException(status_code=404, detail=f"Domain {domain_id} not found")
        
        # Read uploaded file
        content = await data_file.read()
        
        # Process based on file type
        if data_file.filename.endswith('.json'):
            data = json.loads(content.decode('utf-8'))
        elif data_file.filename.endswith('.csv'):
            # Would use pandas to read CSV
            data = {"message": "CSV processing not implemented in this demo"}
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store data (simplified - would use proper data storage)
        storage_path = f"domain_data/{domain_id}/{data_type}/{data_file.filename}"
        
        logger.info(f"Uploaded data for domain {domain_id}: {data_file.filename}")
        return {
            "domain_id": domain_id,
            "data_type": data_type,
            "filename": data_file.filename,
            "storage_path": storage_path,
            "file_size": len(content),
            "message": "Data uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading domain data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adaptation/history")
async def get_adaptation_history():
    """Get history of all adaptation processes"""
    try:
        # Return adaptation history from transfer engine
        history_data = []
        for entry in transfer_engine.adaptation_history:
            history_data.append({
                "target_domain": entry['target_domain'],
                "strategy": entry['strategy'],
                "validation_scores": entry['result'].validation_scores,
                "adaptation_metrics": entry['result'].adaptation_metrics,
                "confidence_score": entry['result'].confidence_score
            })
        
        return {
            "adaptation_history": history_data,
            "total_adaptations": len(history_data)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving adaptation history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export/domains")
async def export_domain_registry():
    """Export the domain registry to JSON"""
    try:
        export_path = "domain_registry_export.json"
        domain_registry.export_domains(export_path)
        
        return {
            "export_path": export_path,
            "exported_domains": len(domain_registry.domains),
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Domain registry exported successfully"
        }
        
    except Exception as e:
        logger.error(f"Error exporting domain registry: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions

def _generate_dummy_target_data(domain: GeologicalDomain, batch_size: int):
    """Generate dummy target data for demonstration"""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Generate synthetic geological data based on domain characteristics
    num_samples = batch_size * 10
    num_features = len(domain.available_data_types) * 10  # 10 features per data type
    
    # Add noise based on domain data quality
    noise_level = 1.0 - domain.data_characteristics.get('data_quality', 0.8)
    
    data = torch.randn(num_samples, num_features) + noise_level * torch.randn(num_samples, num_features)
    targets = torch.randn(num_samples, 1)  
class AdvancedDataProcessor:
    """Advanced data processing pipeline for geological datasets"""
    
    def __init__(self):
        self.processors = {
            'time_series': TimeSeriesProcessor(),
            'spatial': SpatialDataProcessor(),
            'spectral': SpectralAnalysisProcessor(),
            'statistical': StatisticalProcessor(),
            'ml_preprocessing': MLPreprocessor()
        }
    
    def process_geological_dataset(self, dataset: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process geological dataset with configurable pipeline"""
        
        results = {
            'dataset_id': dataset.get('id', str(uuid.uuid4())),
            'processing_timestamp': datetime.now().isoformat(),
            'original_size': len(dataset.get('data', [])),
            'processing_steps': []
        }
        
        data = dataset['data']
        
        # Apply processing steps based on configuration
        for step_name, step_config in processing_config.items():
            if step_name in self.processors:
                try:
                    processor = self.processors[step_name]
                    processed_data, step_results = processor.process(data, step_config)
                    
                    data = processed_data
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'success',
                        'results': step_results,
                        'output_size': len(data) if isinstance(data, list) else 'N/A'
                    })
                    
                except Exception as e:
                    results['processing_steps'].append({
                        'step': step_name,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        results['processed_data'] = data
        results['final_size'] = len(data) if isinstance(data, list) else 'N/A'
        
        return results

class TimeSeriesProcessor:
    """Advanced time series processing for geological data"""
    
    def process(self, data: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process time series data"""
        
        if not isinstance(data, (list, np.ndarray)):
            return data, {'error': 'Invalid data type for time series processing'}
        
        data_array = np.array(data)
        results = {}
        
        # Detrending
        if config.get('detrend', False):
            from scipy import signal
            data_array = signal.detrend(data_array, type=config.get('detrend_type', 'linear'))
            results['detrended'] = True
        
        # Filtering
        if 'filter' in config:
            filter_config = config['filter']
            data_array = self._apply_filter(data_array, filter_config)
            results['filtered'] = filter_config
        
        # Resampling
        if 'resample' in config:
            resample_config = config['resample']
            data_array = self._resample_data(data_array, resample_config)
            results['resampled'] = resample_config
        
        # Anomaly detection
        if config.get('detect_anomalies', False):
            anomalies = self._detect_anomalies(data_array, config.get('anomaly_config', {}))
            results['anomalies'] = anomalies
        
        return data_array.tolist(), results
    
    def _apply_filter(self, data: np.ndarray, filter_config: Dict[str, Any]) -> np.ndarray:
        """Apply digital filter to data"""
        from scipy import signal
        
        filter_type = filter_config.get('type', 'bandpass')
        sampling_rate = filter_config.get('sampling_rate', 100.0)
        
        if filter_type == 'bandpass':
            low_freq = filter_config.get('low_freq', 1.0)
            high_freq = filter_config.get('high_freq', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                [low_freq/nyquist, high_freq/nyquist],
                btype='band'
            )
            return signal.filtfilt(b, a, data)
        
        elif filter_type == 'lowpass':
            cutoff = filter_config.get('cutoff', 25.0)
            nyquist = sampling_rate / 2
            
            b, a = signal.butter(
                filter_config.get('order', 4),
                cutoff/nyquist,
                btype='low'
            )
            return signal.filtfilt(b, a, data)
        
        return data
    
    def _resample_data(self, data: np.ndarray, resample_config: Dict[str, Any]) -> np.ndarray:
        """Resample data to different sampling rate"""
        from scipy import signal
        
        original_rate = resample_config.get('original_rate', 100.0)
        target_rate = resample_config.get('target_rate', 50.0)
        
        if original_rate != target_rate:
            num_samples = int(len(data) * target_rate / original_rate)
            return signal.resample(data, num_samples)
        
        return data
    
    def _detect_anomalies(self, data: np.ndarray, anomaly_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in time series data"""
        
        method = anomaly_config.get('method', 'statistical')
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.mean(data)
            std_val = np.std(data)
            threshold = anomaly_config.get('threshold', 3.0)
            
            anomaly_indices = np.where(np.abs(data - mean_val) > threshold * std_val)[0]
            
            return {
                'method': 'statistical',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'threshold_used': threshold
            }
        
        elif method == 'isolation_forest':
            # Machine learning based anomaly detection
            from sklearn.ensemble import IsolationForest
            
            contamination = anomaly_config.get('contamination', 0.1)
            clf = IsolationForest(contamination=contamination, random_state=42)
            
            data_reshaped = data.reshape(-1, 1)
            anomaly_labels = clf.fit_predict(data_reshaped)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'method': 'isolation_forest',
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'contamination': contamination
            }
        
        return {'method': 'none', 'anomaly_count': 0}

    
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


async def _run_adaptation_task(adaptation_id: str, source_model, target_domain, 
                             target_data, config: AdaptationConfig, use_few_shot: bool):
    """Run adaptation task in background"""
    try:
        logger.info(f"Running adaptation task: {adaptation_id}")
        
        if use_few_shot:
            result = transfer_engine.few_shot_adapt(source_model, target_domain, target_data, config)
        else:
            result = transfer_engine.adapt_to_domain(source_model, target_domain, target_data, config)
        
        # Save adapted model (simplified)
        model_path = f"adapted_models/{adaptation_id}.pth"
        torch.save(result.adapted_model.state_dict(), model_path)
        
        logger.info(f"Adaptation task completed: {adaptation_id}")
        
    except Exception as e:
        logger.error(f"Adaptation task failed: {adaptation_id}, Error: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for transfer learning service"""
    return {
        "status": "healthy",
        "service": "transfer-learning",
        "registered_domains": len(domain_registry.domains),
        "adaptation_history": len(transfer_engine.adaptation_history),
        "timestamp": datetime.utcnow().isoformat()
    }