#!/usr/bin/env python3
"""
Automated Incident Response and Escalation System
Handles automated responses to system incidents and manages escalation procedures
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import redis.asyncio as redis
import structlog
from kubernetes import client

logger = structlog.get_logger()

class IncidentSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class ResponseAction(Enum):
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    FAILOVER = "failover"
    ISOLATE_SERVICE = "isolate_service"
    NOTIFY_ONCALL = "notify_oncall"
    ESCALATE = "escalate"
    RUN_DIAGNOSTIC = "run_diagnostic"

@dataclass
class IncidentResponse:
    incident_id: str
    severity: IncidentSeverity
    service: str
    actions: List[ResponseAction]
    executed_actions: List[str]
    escalation_level: int
    created_at: datetime
    last_action_at: Optional[datetime] = None
    resolved: bool = False

@dataclass
class AutomationRule:
    name: str
    condition: Dict[str, Any]
    actions: List[ResponseAction]
    cooldown_minutes: int
    max_executions: int
    escalation_threshold: int

class IncidentResponseSystem:
    def __init__(self, k8s_client: client.AppsV1Api, redis_client: redis.Redis):
        self.k8s_client = k8s_client
        self.redis_client = redis_client
        self.active_incidents: Dict[str, IncidentResponse] = {}
        self.automation_rules = self._load_automation_rules()
        self.action_history: Dict[str, List[datetime]] = {}
        
    def _load_automation_rules(self) -> List[AutomationRule]:
        """Load automation rules for incident response"""
        return [
            # Service down rules
            AutomationRule(
                name="service_down_restart",
                condition={
                    "alert_type": "service_down",
                    "severity": ["high", "critical"]
                },
                actions=[ResponseAction.RESTART_SERVICE, ResponseAction.NOTIFY_ONCALL],
                cooldown_minutes=5,
                max_executions=3,
                escalation_threshold=2
            ),
            
            # High latency rules
            AutomationRule(
                name="high_latency_scale",
                condition={
                    "alert_type": "high_latency",
                    "response_time": ">5.0"
                },
                actions=[ResponseAction.SCALE_UP, ResponseAction.RUN_DIAGNOSTIC],
                cooldown_minutes=10,
                max_executions=2,
                escalation_threshold=1
            ),
            
            # Memory pressure rules
            AutomationRule(
                name="memory_pressure_restart",
                condition={
                    "alert_type": "high_memory_usage",
                    "memory_usage": ">90"
                },
                actions=[ResponseAction.RESTART_SERVICE, ResponseAction.SCALE_UP],
                cooldown_minutes=15,
                max_executions=2,
                escalation_threshold=1
            ),
            
            # Volcanic monitoring emergency
            AutomationRule(
                name="volcanic_system_emergency",
                condition={
                    "service": "volcano-watchtower",
                    "severity": "emergency"
                },
                actions=[
                    ResponseAction.RESTART_SERVICE,
                    ResponseAction.FAILOVER,
                    ResponseAction.NOTIFY_ONCALL,
                    ResponseAction.ESCALATE
                ],
                cooldown_minutes=1,
                max_executions=5,
                escalation_threshold=0  # Immediate escalation
            ),
            
            # Database connection issues
            AutomationRule(
                name="database_connection_failure",
                condition={
                    "alert_type": "database_connection_failure"
                },
                actions=[ResponseAction.RESTART_SERVICE, ResponseAction.RUN_DIAGNOSTIC],
                cooldown_minutes=5,
                max_executions=3,
                escalation_threshold=2
            ),
            
            # Kafka consumer lag
            AutomationRule(
                name="kafka_consumer_lag",
                condition={
                    "alert_type": "kafka_consumer_lag",
                    "lag": ">10000"
                },
                actions=[ResponseAction.SCALE_UP, ResponseAction.RUN_DIAGNOSTIC],
                cooldown_minutes=10,
                max_executions=2,
                escalation_threshold=1
            )
        ]

    async def process_incident(self, alert_data: Dict[str, Any]) -> Optional[IncidentResponse]:
        """Process an incoming alert and determine if automated response is needed"""
        try:
            # Check if this matches any automation rules
            matching_rules = self._find_matching_rules(alert_data)
            
            if not matching_rules:
                logger.info("No automation rules matched for alert", alert_data=alert_data)
                return None
            
            # Create incident response
            incident_id = f"incident-{alert_data.get('service', 'unknown')}-{int(time.time())}"
            
            # Use the most specific rule (first match)
            rule = matching_rules[0]
            
            incident = IncidentResponse(
                incident_id=incident_id,
                severity=IncidentSeverity(alert_data.get('severity', 'medium')),
                service=alert_data.get('service', 'unknown'),
                actions=rule.actions.copy(),
                executed_actions=[],
                escalation_level=0,
                created_at=datetime.utcnow()
            )
            
            self.active_incidents[incident_id] = incident
            
            logger.info(
                "Created incident response",
                incident_id=incident_id,
                rule_name=rule.name,
                actions=[action.value for action in rule.actions]
            )
            
            # Execute automated actions
            await self._execute_incident_actions(incident, rule)
            
            return incident
            
        except Exception as e:
            logger.error("Failed to process incident", error=str(e), alert_data=alert_data)
            return None

    def _find_matching_rules(self, alert_data: Dict[str, Any]) -> List[AutomationRule]:
        """Find automation rules that match the alert conditions"""
        matching_rules = []
        
        for rule in self.automation_rules:
            if self._rule_matches_alert(rule, alert_data):
                matching_rules.append(rule)
        
        # Sort by specificity (more conditions = more specific)
        matching_rules.sort(key=lambda r: len(r.condition), reverse=True)
        
        return matching_rules

    def _rule_matches_alert(self, rule: AutomationRule, alert_data: Dict[str, Any]) -> bool:
        """Check if a rule matches the alert data"""
        for condition_key, condition_value in rule.condition.items():
            alert_value = alert_data.get(condition_key)
            
            if alert_value is None:
                return False
            
            # Handle different condition types
            if isinstance(condition_value, list):
                if alert_value not in condition_value:
                    return False
            elif isinstance(condition_value, str):
                if condition_value.startswith('>'):
                    threshold = float(condition_value[1:])
                    if float(alert_value) <= threshold:
                        return False
                elif condition_value.startswith('<'):
                    threshold = float(condition_value[1:])
                    if float(alert_value) >= threshold:
                        return False
                else:
                    if str(alert_value) != condition_value:
                        return False
            else:
                if alert_value != condition_value:
                    return False
        
        return True

    async def _execute_incident_actions(self, incident: IncidentResponse, rule: AutomationRule):
        """Execute automated actions for an incident"""
        try:
            for action in incident.actions:
                # Check cooldown
                if not self._check_action_cooldown(incident.service, action, rule.cooldown_minutes):
                    logger.info(
                        "Action skipped due to cooldown",
                        incident_id=incident.incident_id,
                        action=action.value
                    )
                    continue
                
                # Check max executions
                if not self._check_max_executions(incident.service, action, rule.max_executions):
                    logger.info(
                        "Action skipped due to max executions",
                        incident_id=incident.incident_id,
                        action=action.value
                    )
                    continue
                
                # Execute the action
                success = await self._execute_action(incident, action)
                
                if success:
                    incident.executed_actions.append(action.value)
                    incident.last_action_at = datetime.utcnow()
                    
                    # Record action execution
                    action_key = f"{incident.service}:{action.value}"
                    if action_key not in self.action_history:
                        self.action_history[action_key] = []
                    self.action_history[action_key].append(datetime.utcnow())
                    
                    logger.info(
                        "Action executed successfully",
                        incident_id=incident.incident_id,
                        action=action.value
                    )
                else:
                    logger.error(
                        "Action execution failed",
                        incident_id=incident.incident_id,
                        action=action.value
                    )
                
                # Check if escalation is needed
                if len(incident.executed_actions) >= rule.escalation_threshold:
                    await self._escalate_incident(incident)
                
                # Wait between actions
                await asyncio.sleep(2)
                
        except Exception as e:
            logger.error(
                "Failed to execute incident actions",
                incident_id=incident.incident_id,
                error=str(e)
            )

    def _check_action_cooldown(self, service: str, action: ResponseAction, cooldown_minutes: int) -> bool:
        """Check if action is within cooldown period"""
        action_key = f"{service}:{action.value}"
        
        if action_key not in self.action_history:
            return True
        
        last_execution = max(self.action_history[action_key])
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.utcnow() - last_execution > cooldown_period

    def _check_max_executions(self, service: str, action: ResponseAction, max_executions: int) -> bool:
        """Check if action has exceeded maximum executions in the last hour"""
        action_key = f"{service}:{action.value}"
        
        if action_key not in self.action_history:
            return True
        
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_executions = [
            execution for execution in self.action_history[action_key]
            if execution > one_hour_ago
        ]
        
        return len(recent_executions) < max_executions

    async def _execute_action(self, incident: IncidentResponse, action: ResponseAction) -> bool:
        """Execute a specific automated action"""
        try:
            if action == ResponseAction.RESTART_SERVICE:
                return await self._restart_service(incident.service)
            
            elif action == ResponseAction.SCALE_UP:
                return await self._scale_service(incident.service, scale_up=True)
            
            elif action == ResponseAction.SCALE_DOWN:
                return await self._scale_service(incident.service, scale_up=False)
            
            elif action == ResponseAction.FAILOVER:
                return await self._failover_service(incident.service)
            
            elif action == ResponseAction.ISOLATE_SERVICE:
                return await self._isolate_service(incident.service)
            
            elif action == ResponseAction.NOTIFY_ONCALL:
                return await self._notify_oncall(incident)
            
            elif action == ResponseAction.ESCALATE:
                return await self._escalate_incident(incident)
            
            elif action == ResponseAction.RUN_DIAGNOSTIC:
                return await self._run_diagnostic(incident.service)
            
            else:
                logger.warning("Unknown action type", action=action.value)
                return False
                
        except Exception as e:
            logger.error(
                "Action execution failed",
                action=action.value,
                service=incident.service,
                error=str(e)
            )
            return False

    async def _restart_service(self, service_name: str) -> bool:
        """Restart a Kubernetes deployment"""
        try:
            deployment_name = f"{service_name}-deployment"
            
            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai"
            )
            
            # Add restart annotation to trigger rolling restart
            if not deployment.spec.template.metadata.annotations:
                deployment.spec.template.metadata.annotations = {}
            
            deployment.spec.template.metadata.annotations["kubectl.kubernetes.io/restartedAt"] = \
                datetime.utcnow().isoformat()
            
            # Update the deployment
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai",
                body=deployment
            )
            
            logger.info("Service restart initiated", service=service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to restart service", service=service_name, error=str(e))
            return False

    async def _scale_service(self, service_name: str, scale_up: bool) -> bool:
        """Scale a service up or down"""
        try:
            deployment_name = f"{service_name}-deployment"
            
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai"
            )
            
            current_replicas = deployment.spec.replicas or 1
            
            if scale_up:
                new_replicas = min(current_replicas + 1, 20)  # Max 20 replicas
            else:
                new_replicas = max(current_replicas - 1, 1)   # Min 1 replica
            
            if new_replicas != current_replicas:
                deployment.spec.replicas = new_replicas
                
                self.k8s_client.patch_namespaced_deployment(
                    name=deployment_name,
                    namespace="prometheus-ai",
                    body=deployment
                )
                
                direction = "up" if scale_up else "down"
                logger.info(
                    f"Service scaled {direction}",
                    service=service_name,
                    from_replicas=current_replicas,
                    to_replicas=new_replicas
                )
                return True
            else:
                logger.info("Service already at scaling limit", service=service_name)
                return False
                
        except Exception as e:
            logger.error("Failed to scale service", service=service_name, error=str(e))
            return False

    async def _failover_service(self, service_name: str) -> bool:
        """Implement failover for critical services"""
        try:
            # For volcanic monitoring, this is critical
            if service_name == "volcano-watchtower":
                # Try to start backup instance
                backup_deployment_name = f"{service_name}-backup-deployment"
                
                try:
                    backup_deployment = self.k8s_client.read_namespaced_deployment(
                        name=backup_deployment_name,
                        namespace="prometheus-ai"
                    )
                    
                    # Scale up backup if it exists
                    backup_deployment.spec.replicas = 2
                    self.k8s_client.patch_namespaced_deployment(
                        name=backup_deployment_name,
                        namespace="prometheus-ai",
                        body=backup_deployment
                    )
                    
                    logger.info("Backup service activated", service=service_name)
                    return True
                    
                except client.ApiException:
                    # Backup deployment doesn't exist, create emergency instance
                    logger.warning("No backup deployment found, creating emergency instance")
                    return await self._create_emergency_instance(service_name)
            
            return False
            
        except Exception as e:
            logger.error("Failed to failover service", service=service_name, error=str(e))
            return False

    async def _isolate_service(self, service_name: str) -> bool:
        """Isolate a problematic service"""
        try:
            # Scale down to 0 replicas to isolate
            deployment_name = f"{service_name}-deployment"
            
            deployment = self.k8s_client.read_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai"
            )
            
            # Store original replica count for later restoration
            original_replicas = deployment.spec.replicas or 1
            await self.redis_client.set(
                f"isolated_service:{service_name}:original_replicas",
                str(original_replicas),
                ex=3600  # Expire in 1 hour
            )
            
            deployment.spec.replicas = 0
            
            self.k8s_client.patch_namespaced_deployment(
                name=deployment_name,
                namespace="prometheus-ai",
                body=deployment
            )
            
            logger.info("Service isolated", service=service_name, original_replicas=original_replicas)
            return True
            
        except Exception as e:
            logger.error("Failed to isolate service", service=service_name, error=str(e))
            return False

    async def _notify_oncall(self, incident: IncidentResponse) -> bool:
        """Send notification to on-call personnel"""
        try:
            notification = {
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "service": incident.service,
                "executed_actions": incident.executed_actions,
                "timestamp": datetime.utcnow().isoformat(),
                "escalation_level": incident.escalation_level
            }
            
            # Store in Redis for external notification systems
            await self.redis_client.lpush("notifications:oncall", json.dumps(notification))
            
            logger.info("On-call notification sent", incident_id=incident.incident_id)
            return True
            
        except Exception as e:
            logger.error("Failed to send on-call notification", error=str(e))
            return False

    async def _escalate_incident(self, incident: IncidentResponse) -> bool:
        """Escalate an incident to higher level"""
        try:
            incident.escalation_level += 1
            
            escalation_data = {
                "incident_id": incident.incident_id,
                "severity": incident.severity.value,
                "service": incident.service,
                "escalation_level": incident.escalation_level,
                "executed_actions": incident.executed_actions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store escalation in Redis
            await self.redis_client.lpush("escalations", json.dumps(escalation_data))
            
            logger.warning(
                "Incident escalated",
                incident_id=incident.incident_id,
                escalation_level=incident.escalation_level
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to escalate incident", error=str(e))
            return False

    async def _run_diagnostic(self, service_name: str) -> bool:
        """Run diagnostic checks on a service"""
        try:
            diagnostic_data = {
                "service": service_name,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": []
            }
            
            # Check pod status
            try:
                pods = client.CoreV1Api().list_namespaced_pod(
                    namespace="prometheus-ai",
                    label_selector=f"app={service_name}"
                )
                
                pod_statuses = []
                for pod in pods.items:
                    pod_statuses.append({
                        "name": pod.metadata.name,
                        "phase": pod.status.phase,
                        "ready": all(condition.status == "True" 
                                   for condition in pod.status.conditions or []
                                   if condition.type == "Ready")
                    })
                
                diagnostic_data["checks"].append({
                    "type": "pod_status",
                    "result": pod_statuses
                })
                
            except Exception as e:
                diagnostic_data["checks"].append({
                    "type": "pod_status",
                    "error": str(e)
                })
            
            # Store diagnostic results
            await self.redis_client.set(
                f"diagnostics:{service_name}:{int(time.time())}",
                json.dumps(diagnostic_data),
                ex=3600  # Expire in 1 hour
            )
            
            logger.info("Diagnostic completed", service=service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to run diagnostic", service=service_name, error=str(e))
            return False

    async def _create_emergency_instance(self, service_name: str) -> bool:
        """Create an emergency instance of a critical service"""
        try:
            # This is a simplified emergency deployment
            # In production, you'd have pre-configured emergency templates
            
            emergency_deployment = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{service_name}-emergency",
                    "namespace": "prometheus-ai",
                    "labels": {
                        "app": f"{service_name}-emergency",
                        "emergency": "true"
                    }
                },
                "spec": {
                    "replicas": 1,
                    "selector": {
                        "matchLabels": {
                            "app": f"{service_name}-emergency"
                        }
                    },
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": f"{service_name}-emergency"
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": service_name,
                                "image": f"prometheus-ai/{service_name}:latest",
                                "ports": [{"containerPort": 8000}],
                                "env": [
                                    {"name": "EMERGENCY_MODE", "value": "true"}
                                ]
                            }]
                        }
                    }
                }
            }
            
            self.k8s_client.create_namespaced_deployment(
                namespace="prometheus-ai",
                body=emergency_deployment
            )
            
            logger.info("Emergency instance created", service=service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to create emergency instance", service=service_name, error=str(e))
            return False

    async def resolve_incident(self, incident_id: str):
        """Mark an incident as resolved"""
        if incident_id in self.active_incidents:
            self.active_incidents[incident_id].resolved = True
            logger.info("Incident resolved", incident_id=incident_id)

    async def get_active_incidents(self) -> List[Dict[str, Any]]:
        """Get all active incidents"""
        return [asdict(incident) for incident in self.active_incidents.values() if not incident.resolved]

    async def cleanup_old_incidents(self):
        """Clean up old resolved incidents"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        resolved_incidents = [
            incident_id for incident_id, incident in self.active_incidents.items()
            if incident.resolved and incident.created_at < cutoff_time
        ]
        
        for incident_id in resolved_incidents:
            del self.active_incidents[incident_id]
        
        logger.info("Cleaned up old incidents", count=len(resolved_incidents))