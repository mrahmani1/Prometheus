"""MongoDB API endpoints for unstructured geological data"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field
import structlog

from storage.mongodb_storage import MongoDBStorage
from models.geological_data import GeochemicalSample, AtmosphericMeasurement
from models.unstructured_geological_data import (
    RockSample, MineralAnalysis, FluidSample, CoreSample,
    GeologicalSurvey, FieldObservation, LaboratoryResult
)

logger = structlog.get_logger()

router = APIRouter(prefix="/mongodb", tags=["MongoDB Storage"])

# Pydantic models for API requests/responses
class LocationQuery(BaseModel):
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)
    radius_km: float = Field(..., gt=0, le=1000)

class TimeRangeQuery(BaseModel):
    start_time: datetime
    end_time: datetime

class BulkInsertRequest(BaseModel):
    documents: List[Dict[str, Any]]
    collection_name: str

class AggregationRequest(BaseModel):
    collection_name: str
    pipeline: List[Dict[str, Any]]

class DataQualityResponse(BaseModel):
    total_count: List[Dict[str, int]]
    quality_distribution: List[Dict[str, Any]]
    validation_issues: List[Dict[str, Any]]
    temporal_coverage: List[Dict[str, Any]]
    source_distribution: List[Dict[str, Any]]

# Dependency to get MongoDB storage instance
async def get_mongodb_storage() -> MongoDBStorage:
    # This would typically be injected from the main application
    # For now, we'll create a placeholder
    storage = MongoDBStorage("mongodb://localhost:27017")
    if not storage.is_initialized:
        await storage.initialize()
    return storage

@router.post("/geochemical-samples", response_model=Dict[str, str])
async def store_geochemical_sample(
    sample: GeochemicalSample,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store a geochemical sample in MongoDB"""
    try:
        document_id = await storage.store_geochemical_sample(sample)
        return {"id": document_id, "status": "stored"}
    except Exception as e:
        logger.error("Failed to store geochemical sample", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/atmospheric-measurements", response_model=Dict[str, str])
async def store_atmospheric_measurement(
    measurement: AtmosphericMeasurement,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store an atmospheric measurement in MongoDB"""
    try:
        document_id = await storage.store_atmospheric_measurement(measurement)
        return {"id": document_id, "status": "stored"}
    except Exception as e:
        logger.error("Failed to store atmospheric measurement", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unstructured-data", response_model=Dict[str, str])
async def store_unstructured_data(
    data: Dict[str, Any] = Body(...),
    collection_name: str = Query(...),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Store arbitrary unstructured geological data"""
    try:
        document_id = await storage.store_unstructured_data(data, collection_name)
        return {"id": document_id, "status": "stored", "collection": collection_name}
    except Exception as e:
        logger.error("Failed to store unstructured data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/bulk-insert", response_model=Dict[str, Any])
async def bulk_insert_documents(
    request: BulkInsertRequest,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Bulk insert multiple documents"""
    try:
        document_ids = await storage.bulk_insert(request.documents, request.collection_name)
        return {
            "inserted_ids": document_ids,
            "count": len(document_ids),
            "collection": request.collection_name,
            "status": "completed"
        }
    except Exception as e:
        logger.error("Failed bulk insert", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/location", response_model=List[Dict[str, Any]])
async def query_by_location(
    query: LocationQuery,
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Query documents by geographic location"""
    try:
        results = await storage.query_by_location(
            query.longitude, query.latitude, query.radius_km, collection_name, limit
        )
        return results
    except Exception as e:
        logger.error("Failed location query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/time-range", response_model=List[Dict[str, Any]])
async def query_by_time_range(
    query: TimeRangeQuery,
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Query documents by time range"""
    try:
        results = await storage.query_by_time_range(
            query.start_time, query.end_time, collection_name, limit
        )
        return results
    except Exception as e:
        logger.error("Failed time range query", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search", response_model=List[Dict[str, Any]])
async def text_search(
    q: str = Query(..., min_length=1),
    collection_name: str = Query(...),
    limit: int = Query(100, ge=1, le=1000),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Perform full-text search across documents"""
    try:
        results = await storage.text_search(q, collection_name, limit)
        return results
    except Exception as e:
        logger.error("Failed text search", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/geochemical-composition", response_model=Dict[str, Any])
async def aggregate_geochemical_composition(
    sample_type: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate geochemical composition statistics"""
    try:
        time_range = None
        if start_time and end_time:
            time_range = (start_time, end_time)
        
        results = await storage.aggregate_geochemical_composition(sample_type, time_range)
        return results
    except Exception as e:
        logger.error("Failed geochemical composition aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/temporal-trends", response_model=Dict[str, Any])
async def aggregate_temporal_trends(
    collection_name: str = Query(...),
    field_name: str = Query(...),
    time_interval: str = Query("day", regex="^(hour|day|month|year)$"),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate temporal trends for a specific field"""
    try:
        results = await storage.aggregate_temporal_trends(
            collection_name, field_name, time_interval
        )
        return results
    except Exception as e:
        logger.error("Failed temporal trends aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/aggregate/spatial-distribution", response_model=Dict[str, Any])
async def aggregate_spatial_distribution(
    collection_name: str = Query(...),
    grid_size: float = Query(0.1, gt=0, le=10),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Aggregate spatial distribution of data points"""
    try:
        results = await storage.aggregate_spatial_distribution(collection_name, grid_size)
        return results
    except Exception as e:
        logger.error("Failed spatial distribution aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/aggregate/custom", response_model=Dict[str, Any])
async def custom_aggregation(
    request: AggregationRequest,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Execute custom aggregation pipeline"""
    try:
        collection = storage.collections.get(request.collection_name, request.collection_name)
        cursor = storage.db[collection].aggregate(request.pipeline)
        results = await cursor.to_list(length=None)
        
        # Convert ObjectId to string for JSON serialization
        for result in results:
            if '_id' in result and hasattr(result['_id'], '__str__'):
                result['_id'] = str(result['_id'])
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error("Failed custom aggregation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quality-report/{collection_name}", response_model=DataQualityResponse)
async def get_data_quality_report(
    collection_name: str,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Generate comprehensive data quality report"""
    try:
        report = await storage.get_data_quality_report(collection_name)
        return DataQualityResponse(**report)
    except Exception as e:
        logger.error("Failed to generate data quality report", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/cleanup/{collection_name}")
async def cleanup_old_data(
    collection_name: str,
    days_to_keep: int = Query(365, ge=1, le=3650),
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Clean up old data based on retention policy"""
    try:
        deleted_count = await storage.cleanup_old_data(collection_name, days_to_keep)
        return {
            "collection": collection_name,
            "deleted_count": deleted_count,
            "days_kept": days_to_keep,
            "status": "completed"
        }
    except Exception as e:
        logger.error("Failed to cleanup old data", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections", response_model=Dict[str, List[str]])
async def list_collections(
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """List available collections"""
    try:
        collections = list(storage.collections.keys())
        return {"collections": collections}
    except Exception as e:
        logger.error("Failed to list collections", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Check MongoDB storage health"""
    try:
        is_healthy = await storage.is_healthy()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow(),
            "database": storage.database_name
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{collection_name}")
async def get_collection_stats(
    collection_name: str,
    storage: MongoDBStorage = Depends(get_mongodb_storage)
):
    """Get collection statistics"""
    try:
        collection = storage.collections.get(collection_name, collection_name)
        stats = await storage.db.command("collStats", collection)
        
        # Clean up the stats for JSON serialization
        clean_stats = {
            "collection": collection_name,
            "count": stats.get("count", 0),
            "size": stats.get("size", 0),
            "avgObjSize": stats.get("avgObjSize", 0),
            "storageSize": stats.get("storageSize", 0),
            "totalIndexSize": stats.get("totalIndexSize", 0),
            "indexSizes": stats.get("indexSizes", {}),
            "timestamp": datetime.utcnow()
        }
        
        return clean_stats
    except Exception as e:
        logger.error("Failed to get collection stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))