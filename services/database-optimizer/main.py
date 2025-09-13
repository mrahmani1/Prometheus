"""
Database Optimizer Service - Main Entry Point

This is the main FastAPI service for the Advanced Database Optimization system.
It provides REST API endpoints for database optimization, migration management,
performance monitoring, and geological domain-aware database operations.

Features:
- Database optimization recommendations and execution
- Migration management with geological context
- Performance monitoring and tuning
- Connection pool optimization
- Automated backup and recovery
- Schema evolution and version control
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.database_optimizer import (
    DatabaseOptimizer, DatabaseOptimizerConfig, OptimizationType,
    OptimizationRecommendation, create_database_optimizer
)
from core.migration_manager import (
    MigrationManager, MigrationManagerConfig, MigrationType,
    MigrationStrategy, MigrationStatus, create_migration_manager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global service instances
database_optimizer: Optional[DatabaseOptimizer] = None
migration_manager: Optional[MigrationManager] = None


# Pydantic models for API
class OptimizationRequest(BaseModel):
    database_name: str
    geological_context: Optional[Dict[str, Any]] = None


class ApplyOptimizationRequest(BaseModel):
    database_name: str
    recommendation_id: str


class CreateMigrationRequest(BaseModel):
    name: str
    description: str
    migration_type: str
    up_sql: List[str]
    down_sql: List[str]
    geological_context: Optional[Dict[str, Any]] = None
    strategy: str = "immediate"
    requires_maintenance: bool = False
    affects_geological_data: bool = False


class ExecuteMigrationRequest(BaseModel):
    database_name: str
    migration_id: str


class GeologicalOptimizationRequest(BaseModel):
    table_name: str
    optimization_type: str
    geological_context: Dict[str, Any]


# Service configuration
class ServiceConfig(BaseModel):
    """Service configuration"""
    database_optimizer: DatabaseOptimizerConfig = Field(default_factory=DatabaseOptimizerConfig)
    migration_manager: MigrationManagerConfig = Field(default_factory=MigrationManagerConfig)
    
    # Service settings
    host: str = Field(default="0.0.0.0", description="Service host")
    port: int = Field(default=8081, description="Service port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")


# Load configuration
def load_config() -> ServiceConfig:
    """Load service configuration from environment or defaults"""
    config = ServiceConfig()
    
    # Override with environment variables
    if os.getenv("DEBUG"):
        config.debug = os.getenv("DEBUG").lower() == "true"
    
    if os.getenv("PORT"):
        config.port = int(os.getenv("PORT"))
    
    # Database configurations
    if os.getenv("POSTGRES_URL"):
        config.database_optimizer.databases["prometheus_main"] = {
            "type": "postgresql",
            "url": os.getenv("POSTGRES_URL"),
            "pool_size": 20,
            "max_overflow": 30,
            "geological_domain": "volcanic_monitoring"
        }
        config.migration_manager.databases["prometheus_main"] = {
            "type": "postgresql",
            "url": os.getenv("POSTGRES_URL")
        }
    
    return config


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global database_optimizer, migration_manager
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize database optimizer
        logger.info("Initializing database optimizer...")
        database_optimizer = await create_database_optimizer(config.database_optimizer)
        
        # Initialize migration manager
        logger.info("Initializing migration manager...")
        migration_manager = await create_migration_manager(config.migration_manager)
        
        logger.info("Database Optimizer Service initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    finally:
        logger.info("Shutting down Database Optimizer Service")


# Create FastAPI application
config = load_config()
app = FastAPI(
    title="Database Optimizer Service",
    description="Advanced database optimization and migration service for Prometheus AI Platform",
    version="1.0.0",
    lifespan=lifespan,
    debug=config.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
async def get_database_optimizer() -> DatabaseOptimizer:
    """Get database optimizer instance"""
    if database_optimizer is None:
        raise HTTPException(status_code=503, detail="Database optimizer not initialized")
    return database_optimizer


async def get_migration_manager() -> MigrationManager:
    """Get migration manager instance"""
    if migration_manager is None:
        raise HTTPException(status_code=503, detail="Migration manager not initialized")
    return migration_manager


# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    try:
        # Check if all services are initialized
        if not all([database_optimizer, migration_manager]):
            raise HTTPException(status_code=503, detail="Services not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


# Database optimization endpoints
@app.post("/api/v1/optimize/analyze")
async def analyze_database_optimization(
    request: OptimizationRequest,
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Analyze database for optimization opportunities"""
    try:
        recommendations = await optimizer.optimize_database(
            request.database_name,
            request.geological_context
        )
        
        return {
            "database_name": request.database_name,
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "optimization_type": rec.optimization_type.value,
                    "priority": rec.priority,
                    "description": rec.description,
                    "expected_improvement": rec.expected_improvement,
                    "implementation_effort": rec.implementation_effort,
                    "geological_impact": rec.geological_impact,
                    "cost_impact": rec.cost_impact,
                    "affected_tables": rec.affected_tables
                }
                for rec in recommendations
            ]
        }
        
    except Exception as e:
        logger.error(f"Database optimization analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/optimize/apply")
async def apply_optimization(
    request: ApplyOptimizationRequest,
    background_tasks: BackgroundTasks,
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Apply optimization recommendation"""
    try:
        # Find recommendation
        recommendation = None
        for rec in optimizer.optimization_history:
            if rec.recommendation_id == request.recommendation_id:
                recommendation = rec
                break
        
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Apply optimization in background
        background_tasks.add_task(
            optimizer.apply_optimization,
            request.database_name,
            recommendation
        )
        
        return {
            "recommendation_id": request.recommendation_id,
            "status": "applying",
            "message": "Optimization is being applied in the background"
        }
        
    except Exception as e:
        logger.error(f"Optimization application error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/optimize/summary/{database_name}")
async def get_optimization_summary(
    database_name: str,
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Get optimization summary for database"""
    try:
        summary = optimizer.get_optimization_summary()
        
        # Filter for specific database if needed
        database_summary = {
            "database_name": database_name,
            "total_recommendations": summary["total_recommendations"],
            "recommendations_by_type": summary["recommendations_by_type"],
            "recent_recommendations": summary["recent_recommendations"]
        }
        
        # Add database-specific metrics
        if database_name in summary["database_metrics"]:
            database_summary["metrics"] = {
                "connection_count": summary["database_metrics"][database_name].connection_count,
                "avg_query_time": summary["database_metrics"][database_name].avg_query_time,
                "cache_hit_rate": summary["database_metrics"][database_name].cache_hit_rate,
                "storage_usage": summary["database_metrics"][database_name].storage_usage
            }
        
        return database_summary
        
    except Exception as e:
        logger.error(f"Optimization summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Migration management endpoints
@app.post("/api/v1/migrations/create")
async def create_migration(
    request: CreateMigrationRequest,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Create new database migration"""
    try:
        migration_type = MigrationType(request.migration_type)
        strategy = MigrationStrategy(request.strategy)
        
        migration = manager.create_migration(
            name=request.name,
            description=request.description,
            migration_type=migration_type,
            up_sql=request.up_sql,
            down_sql=request.down_sql,
            geological_context=request.geological_context,
            strategy=strategy
        )
        
        # Update migration properties
        migration.requires_maintenance = request.requires_maintenance
        migration.affects_geological_data = request.affects_geological_data
        
        return {
            "migration_id": migration.migration_id,
            "name": migration.name,
            "version": migration.version,
            "status": migration.status.value,
            "created_at": migration.created_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid migration type or strategy: {e}")
    except Exception as e:
        logger.error(f"Migration creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/migrations/execute")
async def execute_migration(
    request: ExecuteMigrationRequest,
    background_tasks: BackgroundTasks,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Execute database migration"""
    try:
        # Execute migration in background
        background_tasks.add_task(
            manager.execute_migration,
            request.database_name,
            request.migration_id
        )
        
        return {
            "migration_id": request.migration_id,
            "database_name": request.database_name,
            "status": "executing",
            "message": "Migration is being executed in the background"
        }
        
    except Exception as e:
        logger.error(f"Migration execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/migrations/execute-pending/{database_name}")
async def execute_pending_migrations(
    database_name: str,
    background_tasks: BackgroundTasks,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Execute all pending migrations for database"""
    try:
        # Execute pending migrations in background
        background_tasks.add_task(
            manager.execute_pending_migrations,
            database_name
        )
        
        return {
            "database_name": database_name,
            "status": "executing",
            "message": "Pending migrations are being executed in the background"
        }
        
    except Exception as e:
        logger.error(f"Pending migrations execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/migrations/status/{database_name}")
async def get_migration_status(
    database_name: str,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Get migration status for database"""
    try:
        status = manager.get_migration_status(database_name)
        return status
        
    except Exception as e:
        logger.error(f"Migration status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/migrations/list")
async def list_migrations(
    manager: MigrationManager = Depends(get_migration_manager)
):
    """List all available migrations"""
    try:
        migrations = []
        
        for migration in manager.migrations.values():
            migrations.append({
                "migration_id": migration.migration_id,
                "name": migration.name,
                "description": migration.description,
                "migration_type": migration.migration_type.value,
                "strategy": migration.strategy.value,
                "version": migration.version,
                "status": migration.status.value,
                "requires_maintenance": migration.requires_maintenance,
                "affects_geological_data": migration.affects_geological_data,
                "created_at": migration.created_at.isoformat(),
                "applied_at": migration.applied_at.isoformat() if migration.applied_at else None
            })
        
        # Sort by version
        migrations.sort(key=lambda x: x["version"], reverse=True)
        
        return {"migrations": migrations}
        
    except Exception as e:
        logger.error(f"Migration list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Geological optimization endpoints
@app.post("/api/v1/geological/optimize")
async def create_geological_optimization(
    request: GeologicalOptimizationRequest,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Create geological optimization migration"""
    try:
        migration = manager.generate_geological_migration(
            request.table_name,
            request.optimization_type,
            request.geological_context
        )
        
        return {
            "migration_id": migration.migration_id,
            "name": migration.name,
            "description": migration.description,
            "optimization_type": request.optimization_type,
            "table_name": request.table_name,
            "geological_context": request.geological_context,
            "up_sql": migration.up_sql,
            "down_sql": migration.down_sql
        }
        
    except Exception as e:
        logger.error(f"Geological optimization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/geological/recommendations/{database_name}")
async def get_geological_recommendations(
    database_name: str,
    domain: Optional[str] = None,
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Get geological optimization recommendations"""
    try:
        geological_context = {"domain": domain} if domain else {}
        
        recommendations = await optimizer.optimize_database(
            database_name,
            geological_context
        )
        
        # Filter for geological optimizations
        geological_recs = [
            rec for rec in recommendations
            if rec.optimization_type in [
                OptimizationType.GEOLOGICAL_OPTIMIZATION,
                OptimizationType.INDEXING,
                OptimizationType.PARTITIONING
            ]
        ]
        
        return {
            "database_name": database_name,
            "domain": domain,
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "optimization_type": rec.optimization_type.value,
                    "priority": rec.priority,
                    "description": rec.description,
                    "geological_impact": rec.geological_impact,
                    "affected_tables": rec.affected_tables,
                    "expected_improvement": rec.expected_improvement
                }
                for rec in geological_recs
            ]
        }
        
    except Exception as e:
        logger.error(f"Geological recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance monitoring endpoints
@app.get("/api/v1/performance/metrics/{database_name}")
async def get_performance_metrics(
    database_name: str,
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Get database performance metrics"""
    try:
        if database_name not in optimizer.performance_metrics:
            raise HTTPException(status_code=404, detail="Database metrics not found")
        
        metrics = optimizer.performance_metrics[database_name]
        
        return {
            "database_name": database_name,
            "metrics": {
                "connection_count": metrics.connection_count,
                "active_queries": metrics.active_queries,
                "avg_query_time": metrics.avg_query_time,
                "cache_hit_rate": metrics.cache_hit_rate,
                "storage_usage": metrics.storage_usage,
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "table_sizes": metrics.table_sizes,
                "index_usage": metrics.index_usage
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/performance/pool-metrics")
async def get_pool_metrics(
    optimizer: DatabaseOptimizer = Depends(get_database_optimizer)
):
    """Get connection pool metrics"""
    try:
        pool_metrics = optimizer.connection_manager.get_pool_metrics()
        
        return {
            "pool_metrics": pool_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Pool metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backup and recovery endpoints
@app.post("/api/v1/backup/create/{database_name}")
async def create_backup(
    database_name: str,
    background_tasks: BackgroundTasks,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Create database backup"""
    try:
        if database_name not in manager.database_configs:
            raise HTTPException(status_code=404, detail="Database configuration not found")
        
        # Create backup in background
        backup_id = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        background_tasks.add_task(
            manager.backup_manager.create_backup,
            manager.database_configs[database_name],
            backup_id
        )
        
        return {
            "database_name": database_name,
            "backup_id": backup_id,
            "status": "creating",
            "message": "Backup is being created in the background"
        }
        
    except Exception as e:
        logger.error(f"Backup creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/backup/cleanup")
async def cleanup_old_backups(
    background_tasks: BackgroundTasks,
    retention_days: int = 30,
    manager: MigrationManager = Depends(get_migration_manager)
):
    """Clean up old backup files"""
    try:
        background_tasks.add_task(
            manager.backup_manager.cleanup_old_backups,
            retention_days
        )
        
        return {
            "retention_days": retention_days,
            "status": "cleaning",
            "message": "Old backups are being cleaned up in the background"
        }
        
    except Exception as e:
        logger.error(f"Backup cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


# Main entry point
if __name__ == "__main__":
    config = load_config()
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug"
    )