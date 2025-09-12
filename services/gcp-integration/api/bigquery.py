"""
BigQuery API Router
REST API endpoints for BigQuery geological analytics
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Optional, Any
import structlog

from ..models.gcp_models import *
from ..core.bigquery_integration import BigQueryGeologicalAnalytics

logger = structlog.get_logger(__name__)
router = APIRouter()

# Dependency to get BigQuery service
async def get_bigquery_service() -> BigQueryGeologicalAnalytics:
    from ..main import bigquery_service
    if not bigquery_service:
        raise HTTPException(status_code=503, detail="BigQuery service not available")
    return bigquery_service


@router.get("/status")
async def get_bigquery_status(
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Get BigQuery service status"""
    try:
        status = await service.get_status()
        return ServiceStatusResponse(
            service="BigQuery Geological Analytics",
            status=status["status"],
            timestamp=status["timestamp"],
            details=status
        )
    except Exception as e:
        logger.error("Error getting BigQuery status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")


@router.post("/analyze")
async def analyze_geological_dataset(
    request: GeologicalAnalysisRequest,
    background_tasks: BackgroundTasks,
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Analyze geological dataset using BigQuery"""
    try:
        logger.info("Starting BigQuery geological analysis", 
                   dataset_id=request.dataset_id, analysis_type=request.analysis_type)
        
        # Start analysis
        results = await service.analyze_geological_dataset(
            request.dataset_id,
            request.analysis_type.value,
            request.parameters
        )
        
        return AnalysisResponse(
            job_id=f"bq_{request.dataset_id}_{request.analysis_type.value}",
            status="completed",
            message="BigQuery analysis completed successfully",
            results=results
        )
        
    except Exception as e:
        logger.error("Error in BigQuery geological analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@router.post("/query")
async def execute_bigquery_query(
    request: BigQueryQueryRequest,
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Execute custom BigQuery query"""
    try:
        logger.info("Executing BigQuery query")
        
        # Execute query using BigQuery client
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(k, "STRING", v) 
                for k, v in (request.parameters or {}).items()
            ],
            use_legacy_sql=request.use_legacy_sql,
            dry_run=request.dry_run
        )
        
        query_job = service.client.query(request.query, job_config=job_config)
        
        if request.dry_run:
            return QueryResultResponse(
                query_id=query_job.job_id,
                status="dry_run_completed",
                total_rows=0,
                bytes_processed=query_job.total_bytes_processed
            )
        
        results = query_job.result()
        rows = [dict(row) for row in results]
        
        return QueryResultResponse(
            query_id=query_job.job_id,
            status="completed",
            rows=rows,
            total_rows=results.total_rows,
            bytes_processed=query_job.total_bytes_processed,
            execution_time_ms=int((query_job.ended - query_job.started).total_seconds() * 1000)
        )
        
    except Exception as e:
        logger.error("Error executing BigQuery query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@router.get("/datasets")
async def list_datasets(
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """List available BigQuery datasets"""
    try:
        datasets = list(service.client.list_datasets())
        
        dataset_list = []
        for dataset in datasets:
            dataset_list.append({
                "dataset_id": dataset.dataset_id,
                "project_id": dataset.project,
                "location": dataset.location,
                "description": dataset.description,
                "created": dataset.created.isoformat() if dataset.created else None,
                "modified": dataset.modified.isoformat() if dataset.modified else None
            })
        
        return {"datasets": dataset_list}
        
    except Exception as e:
        logger.error("Error listing datasets", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@router.get("/datasets/{dataset_id}/tables")
async def list_tables(
    dataset_id: str,
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """List tables in a dataset"""
    try:
        dataset_ref = service.client.dataset(dataset_id)
        tables = list(service.client.list_tables(dataset_ref))
        
        table_list = []
        for table in tables:
            table_list.append({
                "table_id": table.table_id,
                "table_type": table.table_type,
                "num_rows": table.num_rows,
                "num_bytes": table.num_bytes,
                "created": table.created.isoformat() if table.created else None,
                "modified": table.modified.isoformat() if table.modified else None
            })
        
        return {"tables": table_list}
        
    except Exception as e:
        logger.error("Error listing tables", dataset_id=dataset_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")


@router.get("/datasets/{dataset_id}/tables/{table_id}/schema")
async def get_table_schema(
    dataset_id: str,
    table_id: str,
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Get table schema"""
    try:
        table_ref = service.client.dataset(dataset_id).table(table_id)
        table = service.client.get_table(table_ref)
        
        schema = []
        for field in table.schema:
            schema.append({
                "name": field.name,
                "field_type": field.field_type,
                "mode": field.mode,
                "description": field.description
            })
        
        return {
            "table_id": table.table_id,
            "schema": schema,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes
        }
        
    except Exception as e:
        logger.error("Error getting table schema", 
                    dataset_id=dataset_id, table_id=table_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Error getting schema: {str(e)}")


@router.get("/ml-models")
async def list_ml_models(
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """List BigQuery ML models"""
    try:
        models_query = f"""
        SELECT
            model_name,
            model_type,
            creation_time,
            last_modified_time,
            feature_columns,
            label_columns
        FROM `{service.project_id}.{service.ml_dataset_id}.INFORMATION_SCHEMA.ML_MODELS`
        ORDER BY creation_time DESC
        """
        
        query_job = service.client.query(models_query)
        results = query_job.result()
        
        models = []
        for row in results:
            models.append({
                "model_name": row.model_name,
                "model_type": row.model_type,
                "creation_time": row.creation_time.isoformat() if row.creation_time else None,
                "last_modified_time": row.last_modified_time.isoformat() if row.last_modified_time else None,
                "feature_columns": row.feature_columns,
                "label_columns": row.label_columns
            })
        
        return {"models": models}
        
    except Exception as e:
        logger.error("Error listing ML models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@router.post("/ml-models/{model_name}/predict")
async def predict_with_ml_model(
    model_name: str,
    prediction_data: Dict[str, Any],
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Make predictions using BigQuery ML model"""
    try:
        # Construct prediction query
        feature_values = ", ".join([
            f"{k} AS {k}" for k, v in prediction_data.items()
        ])
        
        prediction_query = f"""
        SELECT
            *
        FROM ML.PREDICT(
            MODEL `{service.project_id}.{service.ml_dataset_id}.{model_name}`,
            (SELECT {feature_values})
        )
        """
        
        query_job = service.client.query(prediction_query)
        results = query_job.result()
        
        predictions = [dict(row) for row in results]
        
        return {
            "model_name": model_name,
            "predictions": predictions,
            "input_data": prediction_data
        }
        
    except Exception as e:
        logger.error("Error making ML prediction", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.get("/ml-models/{model_name}/evaluate")
async def evaluate_ml_model(
    model_name: str,
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Evaluate BigQuery ML model"""
    try:
        evaluation_query = f"""
        SELECT
            *
        FROM ML.EVALUATE(
            MODEL `{service.project_id}.{service.ml_dataset_id}.{model_name}`
        )
        """
        
        query_job = service.client.query(evaluation_query)
        results = query_job.result()
        
        evaluation_metrics = {}
        for row in results:
            evaluation_metrics.update(dict(row))
        
        return {
            "model_name": model_name,
            "evaluation_metrics": evaluation_metrics
        }
        
    except Exception as e:
        logger.error("Error evaluating ML model", model_name=model_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@router.post("/export")
async def export_query_results(
    query: str,
    destination_uri: str,
    format: str = "CSV",
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Export query results to Cloud Storage"""
    try:
        # Execute query first
        query_job = service.client.query(query)
        query_job.result()  # Wait for completion
        
        # Export results
        destination_uri_template = f"{destination_uri}/*.{format.lower()}"
        
        extract_job = service.client.extract_table(
            query_job.destination,
            destination_uri_template,
            job_config=bigquery.ExtractJobConfig(
                destination_format=getattr(bigquery.DestinationFormat, format)
            )
        )
        
        extract_job.result()  # Wait for completion
        
        return ExportTaskResponse(
            task_id=extract_job.job_id,
            status="completed",
            export_url=destination_uri,
            created_at=extract_job.created.isoformat()
        )
        
    except Exception as e:
        logger.error("Error exporting query results", error=str(e))
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")


@router.get("/statistics")
async def get_dataset_statistics(
    dataset_id: Optional[str] = Query(None, description="Dataset ID"),
    service: BigQueryGeologicalAnalytics = Depends(get_bigquery_service)
):
    """Get dataset statistics"""
    try:
        if dataset_id:
            # Statistics for specific dataset
            stats_query = f"""
            SELECT
                table_name,
                row_count,
                size_bytes,
                TIMESTAMP_MILLIS(creation_time) as creation_time,
                TIMESTAMP_MILLIS(last_modified_time) as last_modified_time
            FROM `{service.project_id}.{dataset_id}.__TABLES__`
            ORDER BY size_bytes DESC
            """
        else:
            # Statistics for all geological datasets
            stats_query = f"""
            SELECT
                table_schema as dataset_id,
                COUNT(*) as table_count,
                SUM(CAST(row_count AS INT64)) as total_rows,
                SUM(size_bytes) as total_size_bytes
            FROM `{service.project_id}.INFORMATION_SCHEMA.TABLE_STORAGE`
            WHERE table_schema IN ('{service.dataset_id}', '{service.ml_dataset_id}')
            GROUP BY table_schema
            """
        
        query_job = service.client.query(stats_query)
        results = query_job.result()
        
        statistics = [dict(row) for row in results]
        
        return {"statistics": statistics}
        
    except Exception as e:
        logger.error("Error getting dataset statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Statistics error: {str(e)}")