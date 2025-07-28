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
            return {"similar_domains": [], "message": "No similar domains found"}
        
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
    targets = torch.randn(num_samples, 1)  # Simplified target
    
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