"""
Advanced Threat Detection and Response Service
Main service entry point for comprehensive security monitoring and incident response
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from siem_engine import SIEMEngine, SecurityEvent, EventType, ThreatLevel
from automated_response import SOAREngine, ResponseAction, IncidentResponse
from vulnerability_scanner import VulnerabilityManager, ScanTarget, ScanType
from penetration_testing import PenetrationTestingFramework, PentestTarget, PentestSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Advanced Threat Detection and Response Service",
    description="Comprehensive security monitoring, incident response, and vulnerability management",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global service instances
siem_engine: Optional[SIEMEngine] = None
soar_engine: Optional[SOAREngine] = None
vulnerability_manager: Optional[VulnerabilityManager] = None
pentest_framework: Optional[PenetrationTestingFramework] = None

# Configuration
CONFIG = {
    'redis_url': 'redis://localhost:6379',
    'database_url': 'postgresql://user:pass@localhost/threat_detection',
    'elasticsearch_url': 'http://localhost:9200',
    'firewall_api_url': 'http://localhost:8080',
    'firewall_api_key': 'test-key',
    'identity_api_url': 'http://localhost:8081',
    'identity_api_key': 'test-key',
    'zap_api_url': 'http://localhost:8080',
    'zap_api_key': 'test-key',
    'nmap_path': 'nmap',
    'trivy_path': 'trivy'
}

# Pydantic models
class SecurityEventRequest(BaseModel):
    event_type: str = Field(..., description="Type of security event")
    source_ip: str = Field(..., description="Source IP address")
    user_id: Optional[str] = Field(None, description="User ID if applicable")
    resource: str = Field(..., description="Resource accessed")
    action: str = Field(..., description="Action performed")
    outcome: str = Field(..., description="Outcome of the action")
    severity: int = Field(..., ge=1, le=10, description="Severity level (1-10)")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw event data")
    geological_context: Optional[Dict[str, Any]] = Field(None, description="Geological context if applicable")

class ScanTargetRequest(BaseModel):
    name: str = Field(..., description="Target name")
    target_type: str = Field(..., description="Target type (ip, domain, container, repository)")
    target_value: str = Field(..., description="Target value (IP, domain, image name, etc.)")
    scan_types: List[str] = Field(..., description="Types of scans to perform")
    geological_service: bool = Field(False, description="Whether this is a geological service")
    scan_frequency: str = Field("weekly", description="Scan frequency")

class PentestTargetRequest(BaseModel):
    name: str = Field(..., description="Target name")
    target_type: str = Field(..., description="Target type")
    target_scope: List[str] = Field(..., description="Target scope (IPs, domains, etc.)")
    geological_context: bool = Field(False, description="Whether target has geological context")
    authorized_by: str = Field(..., description="Authorization contact")
    restrictions: Optional[List[str]] = Field(None, description="Testing restrictions")

class PentestSessionRequest(BaseModel):
    name: str = Field(..., description="Session name")
    description: str = Field(..., description="Session description")
    target_ids: List[str] = Field(..., description="Target IDs to test")
    phases: List[str] = Field(..., description="Testing phases to execute")
    tester_id: str = Field(..., description="Tester ID")
    authorization_ref: str = Field(..., description="Authorization reference")
    geological_focus: bool = Field(False, description="Focus on geological vulnerabilities")

class ResponseApprovalRequest(BaseModel):
    response_id: str = Field(..., description="Incident response ID")
    approved_by: str = Field(..., description="Approver ID")

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and return user info"""
    # In production, implement proper JWT validation
    return {"user_id": "test_user", "role": "security_analyst"}

@app.on_event("startup")
async def startup_event():
    """Initialize all security services"""
    global siem_engine, soar_engine, vulnerability_manager, pentest_framework
    
    try:
        # Initialize SIEM Engine
        siem_engine = SIEMEngine(CONFIG)
        await siem_engine.initialize()
        logger.info("SIEM Engine initialized")
        
        # Initialize SOAR Engine
        soar_engine = SOAREngine(CONFIG)
        await soar_engine.initialize()
        logger.info("SOAR Engine initialized")
        
        # Initialize Vulnerability Manager
        vulnerability_manager = VulnerabilityManager(CONFIG)
        await vulnerability_manager.initialize()
        logger.info("Vulnerability Manager initialized")
        
        # Initialize Penetration Testing Framework
        pentest_framework = PenetrationTestingFramework(CONFIG)
        await pentest_framework.initialize()
        logger.info("Penetration Testing Framework initialized")
        
        logger.info("All threat detection services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    logger.info("Shutting down threat detection services")

# SIEM Endpoints
@app.post("/api/v1/security/events", response_model=Dict[str, Any])
async def submit_security_event(
    event_request: SecurityEventRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Submit security event for analysis"""
    try:
        # Create security event
        security_event = SecurityEvent(
            id=f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(event_request.source_ip) % 10000}",
            timestamp=datetime.utcnow(),
            event_type=EventType(event_request.event_type),
            source_ip=event_request.source_ip,
            user_id=event_request.user_id,
            resource=event_request.resource,
            action=event_request.action,
            outcome=event_request.outcome,
            severity=event_request.severity,
            raw_data=event_request.raw_data,
            geological_context=event_request.geological_context
        )
        
        # Process event asynchronously
        background_tasks.add_task(process_security_event, security_event)
        
        return {
            "status": "accepted",
            "event_id": security_event.id,
            "message": "Security event submitted for analysis"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit security event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_security_event(event: SecurityEvent):
    """Process security event through SIEM engine"""
    try:
        incident = await siem_engine.process_event(event)
        
        if incident:
            logger.info(f"Security incident created: {incident.id}")
            
            # Trigger automated response if configured
            incident_data = {
                'incident_type': incident.title.lower().replace(' ', '_'),
                'threat_level': incident.threat_level.value
            }
            
            response = await soar_engine.trigger_response(incident.id, incident_data)
            if response:
                logger.info(f"Automated response triggered: {response.id}")
                
    except Exception as e:
        logger.error(f"Failed to process security event: {str(e)}")

@app.get("/api/v1/security/dashboard", response_model=Dict[str, Any])
async def get_security_dashboard(current_user: Dict = Depends(get_current_user)):
    """Get security dashboard data"""
    try:
        dashboard_data = await siem_engine.get_incident_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# SOAR Endpoints
@app.post("/api/v1/security/response/approve", response_model=Dict[str, Any])
async def approve_incident_response(
    approval_request: ResponseApprovalRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Approve incident response for execution"""
    try:
        success = await soar_engine.approve_response(
            approval_request.response_id,
            approval_request.approved_by
        )
        
        if success:
            return {
                "status": "approved",
                "response_id": approval_request.response_id,
                "message": "Incident response approved and execution started"
            }
        else:
            raise HTTPException(status_code=404, detail="Incident response not found")
            
    except Exception as e:
        logger.error(f"Failed to approve incident response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/security/response/{response_id}", response_model=Dict[str, Any])
async def get_response_status(
    response_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get incident response status"""
    try:
        status = await soar_engine.get_response_status(response_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Incident response not found")
            
    except Exception as e:
        logger.error(f"Failed to get response status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Vulnerability Management Endpoints
@app.post("/api/v1/security/scan-targets", response_model=Dict[str, Any])
async def add_scan_target(
    target_request: ScanTargetRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Add new vulnerability scan target"""
    try:
        target = ScanTarget(
            id=f"tgt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(target_request.target_value) % 10000}",
            name=target_request.name,
            target_type=target_request.target_type,
            target_value=target_request.target_value,
            scan_types=[ScanType(st) for st in target_request.scan_types],
            geological_service=target_request.geological_service,
            scan_frequency=target_request.scan_frequency
        )
        
        await vulnerability_manager.add_scan_target(target)
        
        return {
            "status": "created",
            "target_id": target.id,
            "message": "Scan target added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add scan target: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/security/scans/{target_id}/{scan_type}", response_model=Dict[str, Any])
async def start_vulnerability_scan(
    target_id: str,
    scan_type: str,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Start vulnerability scan"""
    try:
        scan_id = await vulnerability_manager.start_scan(target_id, ScanType(scan_type))
        
        return {
            "status": "started",
            "scan_id": scan_id,
            "message": "Vulnerability scan started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start vulnerability scan: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/security/vulnerabilities/dashboard", response_model=Dict[str, Any])
async def get_vulnerability_dashboard(current_user: Dict = Depends(get_current_user)):
    """Get vulnerability dashboard data"""
    try:
        dashboard_data = await vulnerability_manager.get_vulnerability_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Failed to get vulnerability dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Penetration Testing Endpoints
@app.post("/api/v1/security/pentest/targets", response_model=Dict[str, Any])
async def add_pentest_target(
    target_request: PentestTargetRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Add new penetration testing target"""
    try:
        target = PentestTarget(
            id=f"pt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(target_request.name) % 10000}",
            name=target_request.name,
            target_type=target_request.target_type,
            target_scope=target_request.target_scope,
            geological_context=target_request.geological_context,
            authorized_by=target_request.authorized_by,
            authorization_date=datetime.utcnow(),
            restrictions=target_request.restrictions
        )
        
        # Store target (would be implemented in pentest framework)
        # await pentest_framework.add_target(target)
        
        return {
            "status": "created",
            "target_id": target.id,
            "message": "Penetration testing target added successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to add pentest target: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/security/pentest/sessions", response_model=Dict[str, Any])
async def start_pentest_session(
    session_request: PentestSessionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Start penetration testing session"""
    try:
        from penetration_testing import PentestPhase, PentestStatus
        
        session = PentestSession(
            id=f"pts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(session_request.name) % 10000}",
            name=session_request.name,
            description=session_request.description,
            target_ids=session_request.target_ids,
            phases=[PentestPhase(phase) for phase in session_request.phases],
            status=PentestStatus.PLANNED,
            started_at=datetime.utcnow(),
            tester_id=session_request.tester_id,
            authorization_ref=session_request.authorization_ref,
            geological_focus=session_request.geological_focus
        )
        
        session_id = await pentest_framework.start_pentest_session(session)
        
        return {
            "status": "started",
            "session_id": session_id,
            "message": "Penetration testing session started"
        }
        
    except Exception as e:
        logger.error(f"Failed to start pentest session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health Check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "siem_engine": siem_engine is not None,
            "soar_engine": soar_engine is not None,
            "vulnerability_manager": vulnerability_manager is not None,
            "pentest_framework": pentest_framework is not None
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint"""
    services_ready = all([
        siem_engine is not None,
        soar_engine is not None,
        vulnerability_manager is not None,
        pentest_framework is not None
    ])
    
    if services_ready:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Services not ready")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )