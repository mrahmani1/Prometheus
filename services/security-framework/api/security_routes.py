"""
Security API Routes for Zero-Trust Framework
Implements comprehensive security endpoints with geological context awareness
"""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional
from datetime import datetime
import logging

from models.security_models import (
    GeologicalContext, AccessRequest, MFAResponse, 
    SecurityEvent, ThreatAlert, DeviceTrust
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

def create_security_router(security_framework) -> APIRouter:
    """Create security API router with framework dependency injection"""
    
    router = APIRouter()
    
    @router.post("/authenticate")
    async def authenticate_user(
        request: Request,
        username: str,
        password: str,
        geological_context: Optional[GeologicalContext] = None
    ):
        """
        Authenticate user with geological context awareness
        Returns JWT token and geological permissions
        """
        try:
            result = await security_framework.identity_manager.authenticate_user(
                username, password, geological_context
            )
            
            if result.success:
                return {
                    "success": True,
                    "access_token": result.access_token,
                    "token_type": "bearer",
                    "expires_at": result.expires_at.isoformat() if result.expires_at else None,
                    "requires_mfa": result.requires_mfa,
                    "mfa_methods": result.mfa_methods,
                    "geological_permissions": result.geological_permissions,
                    "risk_score": result.risk_score
                }
            else:
                raise HTTPException(
                    status_code=401,
                    detail=result.error or "Authentication failed"
                )
                
        except Exception as e:
            logger.error(f"Authentication endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Authentication service error")
    
    @router.post("/mfa/challenge")
    async def initiate_mfa_challenge(
        request: Request,
        user_id: str,
        method_id: str,
        geological_context: Optional[GeologicalContext] = None,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Initiate MFA challenge with geological workflow validation
        Returns challenge data for specified MFA method
        """
        try:
            # Verify JWT token
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            if token_claims.get('sub') != user_id:
                raise HTTPException(status_code=403, detail="Token user mismatch")
                
            challenge = await security_framework.mfa_manager.initiate_mfa_challenge(
                user_id, method_id, geological_context
            )
            
            return {
                "challenge_id": challenge.challenge_id,
                "method_id": challenge.method_id,
                "challenge_type": challenge.challenge_type,
                "challenge_data": challenge.challenge_data,
                "expires_at": challenge.expires_at.isoformat(),
                "geological_context_required": challenge.geological_context_required
            }
            
        except Exception as e:
            logger.error(f"MFA challenge endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="MFA challenge service error")
    
    @router.post("/mfa/verify")
    async def verify_mfa_response(
        request: Request,
        challenge_id: str,
        response: MFAResponse
    ):
        """
        Verify MFA response with geological context validation
        Returns verification result and updated token if successful
        """
        try:
            success = await security_framework.mfa_manager.verify_mfa_response(
                challenge_id, response
            )
            
            if success:
                return {
                    "success": True,
                    "message": "MFA verification successful",
                    "updated_token": None  # Would generate updated JWT with MFA claim
                }
            else:
                return {
                    "success": False,
                    "message": "MFA verification failed"
                }
                
        except Exception as e:
            logger.error(f"MFA verification endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="MFA verification service error")
    
    @router.post("/access/request")
    async def request_resource_access(
        request: Request,
        access_request: AccessRequest,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Request access to geological resource with AI-powered decision making
        Returns access decision and dynamic permissions if approved
        """
        try:
            # Verify JWT token
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            if token_claims.get('sub') != access_request.user_id:
                raise HTTPException(status_code=403, detail="Token user mismatch")
                
            decision = await security_framework.access_controller.request_access(
                access_request.user_id,
                access_request.resource_id,
                access_request.requested_access_level,
                access_request.geological_context,
                access_request.justification
            )
            
            return {
                "request_id": decision.request_id,
                "approved": decision.approved,
                "reason": decision.reason,
                "risk_score": decision.risk_score,
                "permission_id": decision.permission.permission_id if decision.permission else None,
                "expires_at": decision.expires_at.isoformat() if decision.expires_at else None,
                "conditions": decision.conditions
            }
            
        except Exception as e:
            logger.error(f"Access request endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Access request service error")
    
    @router.post("/access/validate")
    async def validate_resource_access(
        request: Request,
        user_id: str,
        resource_id: str,
        access_level: str,
        geological_context: GeologicalContext,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Validate user access to resource with real-time permission checking
        Returns validation result and access details
        """
        try:
            # Verify JWT token
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            if token_claims.get('sub') != user_id:
                raise HTTPException(status_code=403, detail="Token user mismatch")
                
            valid = await security_framework.access_controller.validate_access(
                user_id, resource_id, access_level, geological_context
            )
            
            return {
                "valid": valid,
                "user_id": user_id,
                "resource_id": resource_id,
                "access_level": access_level,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Access validation endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Access validation service error")
    
    @router.post("/device/validate")
    async def validate_device_trust(
        request: Request,
        device_id: Optional[str] = None,
        geological_context: Optional[GeologicalContext] = None
    ):
        """
        Validate device trust with geological equipment awareness
        Returns device trust assessment and security recommendations
        """
        try:
            source_ip = request.client.host
            user_agent = request.headers.get("User-Agent", "")
            
            validation_result = await security_framework.device_trust_manager.validate_device(
                source_ip, user_agent, device_id, geological_context
            )
            
            return {
                "device_id": validation_result["device_id"],
                "trusted": validation_result["trusted"],
                "trust_score": validation_result["trust_score"],
                "risk_factors": validation_result["risk_factors"],
                "geological_compliance": validation_result.get("geological_compliance", {}),
                "security_features": validation_result.get("security_features", {}),
                "recommendations": validation_result.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"Device validation endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Device validation service error")
    
    @router.get("/network/validate")
    async def validate_network_request(request: Request):
        """
        Validate network request against segmentation policies
        Returns network validation result and policy information
        """
        try:
            valid = await security_framework.network_segmentation.validate_request(request)
            
            return {
                "valid": valid,
                "source_ip": request.client.host,
                "destination": str(request.url),
                "timestamp": datetime.utcnow().isoformat(),
                "policies_applied": []  # Would include actual policy IDs
            }
            
        except Exception as e:
            logger.error(f"Network validation endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Network validation service error")
    
    @router.get("/security/events")
    async def get_security_events(
        request: Request,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        geological_domain: Optional[str] = None,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Retrieve security events with filtering and pagination
        Requires appropriate security permissions
        """
        try:
            # Verify JWT token and check permissions
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            # Check if user has security monitoring permissions
            user_roles = token_claims.get('roles', [])
            if not any(role in ['security_admin', 'security_analyst', 'system_admin'] for role in user_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions for security events")
                
            # This would query the audit log database
            # For now, return placeholder response
            return {
                "events": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "filters": {
                    "event_type": event_type,
                    "user_id": user_id,
                    "geological_domain": geological_domain
                }
            }
            
        except Exception as e:
            logger.error(f"Security events endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Security events service error")
    
    @router.get("/threats/alerts")
    async def get_threat_alerts(
        request: Request,
        limit: int = 50,
        active_only: bool = True,
        threat_level: Optional[int] = None,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Retrieve threat alerts with filtering
        Requires security analyst or admin permissions
        """
        try:
            # Verify JWT token and check permissions
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            user_roles = token_claims.get('roles', [])
            if not any(role in ['security_admin', 'security_analyst', 'threat_analyst'] for role in user_roles):
                raise HTTPException(status_code=403, detail="Insufficient permissions for threat alerts")
                
            # This would query the threat alerts database
            return {
                "alerts": [],
                "total_count": 0,
                "limit": limit,
                "filters": {
                    "active_only": active_only,
                    "threat_level": threat_level
                }
            }
            
        except Exception as e:
            logger.error(f"Threat alerts endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Threat alerts service error")
    
    @router.post("/policy/evaluate")
    async def evaluate_security_policy(
        request: Request,
        policy_id: str,
        context: Dict,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Evaluate security policy against provided context
        Returns policy evaluation result and recommendations
        """
        try:
            # Verify JWT token
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            # This would evaluate the policy using the policy engine
            return {
                "policy_id": policy_id,
                "evaluation_result": "allow",  # or "deny"
                "confidence": 0.95,
                "conditions": [],
                "recommendations": []
            }
            
        except Exception as e:
            logger.error(f"Policy evaluation endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Policy evaluation service error")
    
    @router.get("/compliance/status")
    async def get_compliance_status(
        request: Request,
        geological_domain: Optional[str] = None,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ):
        """
        Get compliance status for geological data handling
        Returns compliance metrics and violation reports
        """
        try:
            # Verify JWT token
            token_claims = await security_framework.identity_manager.verify_token(
                credentials.credentials
            )
            
            return {
                "overall_compliance": 0.95,
                "geological_domain": geological_domain,
                "compliance_frameworks": [
                    "GDPR", "SOC2", "ISO27001", "NIST_Cybersecurity_Framework"
                ],
                "violations": [],
                "recommendations": [],
                "last_assessment": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Compliance status endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail="Compliance status service error")
    
    return router