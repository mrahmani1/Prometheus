"""
Workflow Automation API Routes
Advanced workflow management and execution endpoints
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc, func
from pydantic import BaseModel, Field

from ..main import (
    get_db, get_current_user, WorkflowDefinition, WorkflowExecution, WorkflowTemplate, WorkflowSchedule,
    CreateWorkflowRequest, ExecuteWorkflowRequest, WorkflowResponse, ExecutionResponse,
    ScheduleRequest, WorkflowGraph, WorkflowNode, WorkflowEdge, workflow_engine
)
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/workflow", tags=["workflow-automation"])

# Request/Response Models
class UpdateWorkflowRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_graph: Optional[WorkflowGraph] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    triggers: Optional[List[Dict[str, Any]]] = None
    schedule: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    retry_policy: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None

class WorkflowTemplateRequest(BaseModel):
    template_name: str = Field(..., min_length=1, max_length=255)
    category: str
    description: Optional[str] = None
    template_definition: WorkflowGraph
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_public: bool = False

class WorkflowTemplateResponse(BaseModel):
    id: str
    template_name: str
    category: str
    description: Optional[str]
    parameters: Dict[str, Any]
    tags: List[str]
    usage_count: int
    rating: float
    is_public: bool
    created_by: str
    created_at: datetime

class ScheduleResponse(BaseModel):
    id: str
    workflow_id: str
    schedule_name: str
    cron_expression: str
    timezone: str
    is_active: bool
    next_run_time: Optional[datetime]
    last_run_time: Optional[datetime]
    run_count: int
    max_runs: Optional[int]

class WorkflowAnalytics(BaseModel):
    total_workflows: int
    active_workflows: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    success_rate: float
    avg_execution_time: float
    total_schedules: int
    active_schedules: int
    execution_trends: List[Dict[str, Any]]

class ExecutionLogEntry(BaseModel):
    timestamp: datetime
    level: str
    message: str
    step_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

# Workflow Management
@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create new workflow"""
    try:
        workflow = await workflow_engine.create_workflow(request, current_user)
        
        return WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            status="active" if workflow.is_active else "inactive",
            created_by=str(workflow.created_by),
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            execution_count=0,  # Will be calculated
            last_execution=None,
            tags=workflow.tags or []
        )
        
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create workflow")

@router.get("/workflows", response_model=List[WorkflowResponse])
async def list_workflows(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    is_active: Optional[bool] = Query(None),
    tags: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List user's workflows"""
    try:
        query = select(WorkflowDefinition).where(WorkflowDefinition.created_by == UUID(current_user))
        
        if is_active is not None:
            query = query.where(WorkflowDefinition.is_active == is_active)
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            query = query.where(WorkflowDefinition.tags.op("&&")(tag_list))
        
        query = query.offset(offset).limit(limit).order_by(desc(WorkflowDefinition.updated_at))
        
        result = await db.execute(query)
        workflows = result.scalars().all()
        
        workflow_responses = []
        for workflow in workflows:
            # Count executions
            exec_query = select(func.count(WorkflowExecution.id)).where(
                WorkflowExecution.workflow_id == workflow.id
            )
            exec_result = await db.execute(exec_query)
            execution_count = exec_result.scalar() or 0
            
            # Get last execution
            last_exec_query = select(WorkflowExecution.started_at).where(
                WorkflowExecution.workflow_id == workflow.id
            ).order_by(desc(WorkflowExecution.started_at)).limit(1)
            last_exec_result = await db.execute(last_exec_query)
            last_execution = last_exec_result.scalar()
            
            workflow_responses.append(WorkflowResponse(
                id=str(workflow.id),
                name=workflow.name,
                description=workflow.description,
                version=workflow.version,
                status="active" if workflow.is_active else "inactive",
                created_by=str(workflow.created_by),
                created_at=workflow.created_at,
                updated_at=workflow.updated_at,
                execution_count=execution_count,
                last_execution=last_execution,
                tags=workflow.tags or []
            ))
        
        return workflow_responses
        
    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list workflows")

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get workflow details"""
    try:
        query = select(WorkflowDefinition).where(
            and_(
                WorkflowDefinition.id == UUID(workflow_id),
                WorkflowDefinition.created_by == UUID(current_user)
            )
        )
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Count executions
        exec_query = select(func.count(WorkflowExecution.id)).where(
            WorkflowExecution.workflow_id == workflow.id
        )
        exec_result = await db.execute(exec_query)
        execution_count = exec_result.scalar() or 0
        
        # Get last execution
        last_exec_query = select(WorkflowExecution.started_at).where(
            WorkflowExecution.workflow_id == workflow.id
        ).order_by(desc(WorkflowExecution.started_at)).limit(1)
        last_exec_result = await db.execute(last_exec_query)
        last_execution = last_exec_result.scalar()
        
        return WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            status="active" if workflow.is_active else "inactive",
            created_by=str(workflow.created_by),
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            execution_count=execution_count,
            last_execution=last_execution,
            tags=workflow.tags or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get workflow")

@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    request: UpdateWorkflowRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update workflow"""
    try:
        query = select(WorkflowDefinition).where(
            and_(
                WorkflowDefinition.id == UUID(workflow_id),
                WorkflowDefinition.created_by == UUID(current_user)
            )
        )
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Update fields
        if request.name is not None:
            workflow.name = request.name
        if request.description is not None:
            workflow.description = request.description
        if request.workflow_graph is not None:
            workflow.workflow_graph = request.workflow_graph.dict()
        if request.input_schema is not None:
            workflow.input_schema = request.input_schema
        if request.output_schema is not None:
            workflow.output_schema = request.output_schema
        if request.triggers is not None:
            workflow.triggers = request.triggers
        if request.schedule is not None:
            workflow.schedule = request.schedule
        if request.timeout_seconds is not None:
            workflow.timeout_seconds = request.timeout_seconds
        if request.retry_policy is not None:
            workflow.retry_policy = request.retry_policy
        if request.tags is not None:
            workflow.tags = request.tags
        if request.is_active is not None:
            workflow.is_active = request.is_active
        
        workflow.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return WorkflowResponse(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            version=workflow.version,
            status="active" if workflow.is_active else "inactive",
            created_by=str(workflow.created_by),
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            execution_count=0,  # Would need to recalculate
            last_execution=None,
            tags=workflow.tags or []
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update workflow")

@router.delete("/workflows/{workflow_id}")
async def delete_workflow(
    workflow_id: str,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete workflow"""
    try:
        query = select(WorkflowDefinition).where(
            and_(
                WorkflowDefinition.id == UUID(workflow_id),
                WorkflowDefinition.created_by == UUID(current_user)
            )
        )
        result = await db.execute(query)
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check for active executions
        active_exec_query = select(func.count(WorkflowExecution.id)).where(
            and_(
                WorkflowExecution.workflow_id == workflow.id,
                WorkflowExecution.status.in_(["pending", "running"])
            )
        )
        active_exec_result = await db.execute(active_exec_query)
        active_executions = active_exec_result.scalar() or 0
        
        if active_executions > 0:
            raise HTTPException(status_code=400, detail="Cannot delete workflow with active executions")
        
        await db.delete(workflow)
        await db.commit()
        
        logger.info("Workflow deleted", workflow_id=workflow_id, user_id=current_user)
        
        return {"message": "Workflow deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete workflow")

# Workflow Execution
@router.post("/workflows/{workflow_id}/execute", response_model=ExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    input_data: Dict[str, Any] = {},
    execution_name: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Execute workflow"""
    try:
        request = ExecuteWorkflowRequest(
            workflow_id=workflow_id,
            execution_name=execution_name,
            input_data=input_data,
            trigger_type="manual"
        )
        
        execution = await workflow_engine.execute_workflow(request, current_user)
        
        return ExecutionResponse(
            id=str(execution.id),
            workflow_id=str(execution.workflow_id),
            execution_name=execution.execution_name,
            status=execution.status,
            trigger_type=execution.trigger_type,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            duration_seconds=execution.duration_seconds,
            progress_percentage=execution.progress_percentage,
            error_message=execution.error_message
        )
        
    except Exception as e:
        logger.error("Failed to execute workflow", workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to execute workflow")

@router.get("/executions", response_model=List[ExecutionResponse])
async def list_executions(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    workflow_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """List workflow executions"""
    try:
        query = select(WorkflowExecution).where(WorkflowExecution.executed_by == UUID(current_user))
        
        if workflow_id:
            query = query.where(WorkflowExecution.workflow_id == UUID(workflow_id))
        
        if status:
            query = query.where(WorkflowExecution.status == status)
        
        query = query.offset(offset).limit(limit).order_by(desc(WorkflowExecution.started_at))
        
        result = await db.execute(query)
        executions = result.scalars().all()
        
        return [
            ExecutionResponse(
                id=str(execution.id),
                workflow_id=str(execution.workflow_id),
                execution_name=execution.execution_name,
                status=execution.status,
                trigger_type=execution.trigger_type,
                started_at=execution.started_at,
                completed_at=execution.completed_at,
                duration_seconds=execution.duration_seconds,
                progress_percentage=execution.progress_percentage,
                error_message=execution.error_message
            )
            for execution in executions
        ]
        
    except Exception as e:
        logger.error("Failed to list executions", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list executions")

@router.get("/executions/{execution_id}/status")
async def get_execution_status(
    execution_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get detailed execution status"""
    try:
        status = await workflow_engine.get_execution_status(execution_id)
        return status
        
    except Exception as e:
        logger.error("Failed to get execution status", execution_id=execution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get execution status")

@router.post("/executions/{execution_id}/cancel")
async def cancel_execution(
    execution_id: str,
    current_user: str = Depends(get_current_user)
):
    """Cancel workflow execution"""
    try:
        success = await workflow_engine.cancel_execution(execution_id, current_user)
        
        if success:
            return {"message": "Execution cancelled successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel execution")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel execution", execution_id=execution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel execution")

# Workflow Templates
@router.post("/templates", response_model=WorkflowTemplateResponse)
async def create_workflow_template(
    request: WorkflowTemplateRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create workflow template"""
    try:
        template = WorkflowTemplate(
            template_name=request.template_name,
            category=request.category,
            description=request.description,
            template_definition=request.template_definition.dict(),
            parameters=request.parameters,
            tags=request.tags,
            is_public=request.is_public,
            created_by=UUID(current_user)
        )
        
        db.add(template)
        await db.commit()
        await db.refresh(template)
        
        return WorkflowTemplateResponse(
            id=str(template.id),
            template_name=template.template_name,
            category=template.category,
            description=template.description,
            parameters=template.parameters,
            tags=template.tags or [],
            usage_count=template.usage_count,
            rating=template.rating,
            is_public=template.is_public,
            created_by=str(template.created_by),
            created_at=template.created_at
        )
        
    except Exception as e:
        logger.error("Failed to create template", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create template")

@router.get("/templates", response_model=List[WorkflowTemplateResponse])
async def list_workflow_templates(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    category: Optional[str] = Query(None),
    is_public: Optional[bool] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List workflow templates"""
    try:
        query = select(WorkflowTemplate).where(
            or_(
                WorkflowTemplate.created_by == UUID(current_user),
                WorkflowTemplate.is_public == True
            )
        )
        
        if category:
            query = query.where(WorkflowTemplate.category == category)
        
        if is_public is not None:
            query = query.where(WorkflowTemplate.is_public == is_public)
        
        query = query.offset(offset).limit(limit).order_by(desc(WorkflowTemplate.usage_count))
        
        result = await db.execute(query)
        templates = result.scalars().all()
        
        return [
            WorkflowTemplateResponse(
                id=str(template.id),
                template_name=template.template_name,
                category=template.category,
                description=template.description,
                parameters=template.parameters,
                tags=template.tags or [],
                usage_count=template.usage_count,
                rating=template.rating,
                is_public=template.is_public,
                created_by=str(template.created_by),
                created_at=template.created_at
            )
            for template in templates
        ]
        
    except Exception as e:
        logger.error("Failed to list templates", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list templates")

# Workflow Scheduling
@router.post("/schedules", response_model=ScheduleResponse)
async def create_schedule(
    request: ScheduleRequest,
    current_user: str = Depends(get_current_user)
):
    """Create workflow schedule"""
    try:
        schedule = await workflow_engine.create_schedule(request, current_user)
        
        return ScheduleResponse(
            id=str(schedule.id),
            workflow_id=str(schedule.workflow_id),
            schedule_name=schedule.schedule_name,
            cron_expression=schedule.cron_expression,
            timezone=schedule.timezone,
            is_active=schedule.is_active,
            next_run_time=schedule.next_run_time,
            last_run_time=schedule.last_run_time,
            run_count=schedule.run_count,
            max_runs=schedule.max_runs
        )
        
    except Exception as e:
        logger.error("Failed to create schedule", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create schedule")

@router.get("/schedules", response_model=List[ScheduleResponse])
async def list_schedules(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    workflow_id: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """List workflow schedules"""
    try:
        # Get workflows owned by user
        workflow_query = select(WorkflowDefinition.id).where(
            WorkflowDefinition.created_by == UUID(current_user)
        )
        workflow_result = await db.execute(workflow_query)
        user_workflow_ids = [row[0] for row in workflow_result.fetchall()]
        
        query = select(WorkflowSchedule).where(
            WorkflowSchedule.workflow_id.in_(user_workflow_ids)
        )
        
        if workflow_id:
            query = query.where(WorkflowSchedule.workflow_id == UUID(workflow_id))
        
        if is_active is not None:
            query = query.where(WorkflowSchedule.is_active == is_active)
        
        query = query.offset(offset).limit(limit).order_by(desc(WorkflowSchedule.created_at))
        
        result = await db.execute(query)
        schedules = result.scalars().all()
        
        return [
            ScheduleResponse(
                id=str(schedule.id),
                workflow_id=str(schedule.workflow_id),
                schedule_name=schedule.schedule_name,
                cron_expression=schedule.cron_expression,
                timezone=schedule.timezone,
                is_active=schedule.is_active,
                next_run_time=schedule.next_run_time,
                last_run_time=schedule.last_run_time,
                run_count=schedule.run_count,
                max_runs=schedule.max_runs
            )
            for schedule in schedules
        ]
        
    except Exception as e:
        logger.error("Failed to list schedules", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list schedules")

# Workflow Validation
@router.post("/workflows/validate", response_model=WorkflowValidationResult)
async def validate_workflow(
    workflow_graph: WorkflowGraph,
    current_user: str = Depends(get_current_user)
):
    """Validate workflow graph"""
    try:
        # Perform validation
        errors = []
        warnings = []
        suggestions = []
        
        # Check for start and end nodes
        start_nodes = [n for n in workflow_graph.nodes if n.type == "start"]
        end_nodes = [n for n in workflow_graph.nodes if n.type == "end"]
        
        if len(start_nodes) == 0:
            errors.append("Workflow must have a start node")
        elif len(start_nodes) > 1:
            warnings.append("Multiple start nodes found, only one will be used")
        
        if len(end_nodes) == 0:
            errors.append("Workflow must have at least one end node")
        
        # Check for disconnected nodes
        node_ids = {n.id for n in workflow_graph.nodes}
        connected_nodes = set()
        
        for edge in workflow_graph.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        disconnected = node_ids - connected_nodes
        if disconnected:
            warnings.append(f"Disconnected nodes found: {', '.join(disconnected)}")
        
        # Check for cycles (simplified)
        # In a real implementation, you'd use a proper cycle detection algorithm
        
        # Performance suggestions
        if len(workflow_graph.nodes) > 50:
            suggestions.append("Consider breaking large workflows into smaller sub-workflows")
        
        is_valid = len(errors) == 0
        
        return WorkflowValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error("Failed to validate workflow", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to validate workflow")

# Analytics and Monitoring
@router.get("/analytics", response_model=WorkflowAnalytics)
async def get_workflow_analytics(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    days: int = Query(30, ge=1, le=365)
):
    """Get workflow analytics"""
    try:
        since_date = datetime.utcnow() - timedelta(days=days)
        
        # Count workflows
        workflow_query = select(func.count(WorkflowDefinition.id)).where(
            WorkflowDefinition.created_by == UUID(current_user)
        )
        workflow_result = await db.execute(workflow_query)
        total_workflows = workflow_result.scalar() or 0
        
        # Count active workflows
        active_workflow_query = select(func.count(WorkflowDefinition.id)).where(
            and_(
                WorkflowDefinition.created_by == UUID(current_user),
                WorkflowDefinition.is_active == True
            )
        )
        active_workflow_result = await db.execute(active_workflow_query)
        active_workflows = active_workflow_result.scalar() or 0
        
        # Count executions
        execution_query = select(func.count(WorkflowExecution.id)).where(
            and_(
                WorkflowExecution.executed_by == UUID(current_user),
                WorkflowExecution.started_at >= since_date
            )
        )
        execution_result = await db.execute(execution_query)
        total_executions = execution_result.scalar() or 0
        
        # Count successful executions
        successful_query = select(func.count(WorkflowExecution.id)).where(
            and_(
                WorkflowExecution.executed_by == UUID(current_user),
                WorkflowExecution.status == "completed",
                WorkflowExecution.started_at >= since_date
            )
        )
        successful_result = await db.execute(successful_query)
        successful_executions = successful_result.scalar() or 0
        
        # Count failed executions
        failed_query = select(func.count(WorkflowExecution.id)).where(
            and_(
                WorkflowExecution.executed_by == UUID(current_user),
                WorkflowExecution.status == "failed",
                WorkflowExecution.started_at >= since_date
            )
        )
        failed_result = await db.execute(failed_query)
        failed_executions = failed_result.scalar() or 0
        
        # Calculate success rate
        success_rate = 0.0
        if total_executions > 0:
            success_rate = successful_executions / total_executions
        
        # Calculate average execution time
        avg_time_query = select(func.avg(WorkflowExecution.duration_seconds)).where(
            and_(
                WorkflowExecution.executed_by == UUID(current_user),
                WorkflowExecution.status == "completed",
                WorkflowExecution.started_at >= since_date
            )
        )
        avg_time_result = await db.execute(avg_time_query)
        avg_execution_time = float(avg_time_result.scalar() or 0.0)
        
        # Count schedules
        schedule_query = select(func.count(WorkflowSchedule.id)).where(
            WorkflowSchedule.workflow_id.in_(
                select(WorkflowDefinition.id).where(
                    WorkflowDefinition.created_by == UUID(current_user)
                )
            )
        )
        schedule_result = await db.execute(schedule_query)
        total_schedules = schedule_result.scalar() or 0
        
        # Count active schedules
        active_schedule_query = select(func.count(WorkflowSchedule.id)).where(
            and_(
                WorkflowSchedule.workflow_id.in_(
                    select(WorkflowDefinition.id).where(
                        WorkflowDefinition.created_by == UUID(current_user)
                    )
                ),
                WorkflowSchedule.is_active == True
            )
        )
        active_schedule_result = await db.execute(active_schedule_query)
        active_schedules = active_schedule_result.scalar() or 0
        
        # Execution trends (simplified)
        execution_trends = []
        for i in range(7):  # Last 7 days
            day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            
            day_query = select(func.count(WorkflowExecution.id)).where(
                and_(
                    WorkflowExecution.executed_by == UUID(current_user),
                    WorkflowExecution.started_at >= day_start,
                    WorkflowExecution.started_at < day_end
                )
            )
            day_result = await db.execute(day_query)
            day_count = day_result.scalar() or 0
            
            execution_trends.append({
                "date": day_start.strftime("%Y-%m-%d"),
                "executions": day_count
            })
        
        return WorkflowAnalytics(
            total_workflows=total_workflows,
            active_workflows=active_workflows,
            total_executions=total_executions,
            successful_executions=successful_executions,
            failed_executions=failed_executions,
            success_rate=round(success_rate, 3),
            avg_execution_time=round(avg_execution_time, 2),
            total_schedules=total_schedules,
            active_schedules=active_schedules,
            execution_trends=execution_trends
        )
        
    except Exception as e:
        logger.error("Failed to get workflow analytics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get workflow analytics")

# Real-time Execution Monitoring
@router.websocket("/executions/{execution_id}/monitor")
async def monitor_execution(websocket: WebSocket, execution_id: str):
    """Monitor workflow execution in real-time"""
    await websocket.accept()
    
    try:
        while True:
            # Get current execution status
            try:
                status = await workflow_engine.get_execution_status(execution_id)
                await websocket.send_json(status)
                
                # If execution is complete, close connection
                if status["execution"]["status"] in ["completed", "failed", "cancelled"]:
                    break
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                break
            
            # Wait before next update
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected", execution_id=execution_id)
    except Exception as e:
        logger.error("WebSocket error", execution_id=execution_id, error=str(e))