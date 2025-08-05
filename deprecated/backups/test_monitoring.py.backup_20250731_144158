#!/usr/bin/env python3
"""
Test suite for the System Monitoring Service
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import redis.asyncio as redis
from kubernetes import client

# Import the modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MonitoringService, ServiceStatus, AlertSeverity, Alert, HealthCheckResult, SystemMetrics
from incident_response import IncidentResponseSystem, IncidentSeverity, ResponseAction, IncidentResponse
from notification_service import NotificationService, NotificationRequest, NotificationType

class TestMonitoringService:
    """Test cases for the main monitoring service"""
    
    @pytest.fixture
    async def monitoring_service(self):
        """Create a monitoring service instance for testing"""
        service = MonitoringService()
        service.redis_client = AsyncMock()
        service.db_pool = AsyncMock()
        service.k8s_client = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_health_check_healthy_service(self, monitoring_service):
        """Test health check for a healthy service"""
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "healthy", "uptime": 3600})
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            # Mock dependency checks
            monitoring_service._check_dependencies = AsyncMock(return_value={"all_healthy": True, "dependencies": {}})
            
            results = await monitoring_service.perform_health_checks()
            
            assert len(results) > 0
            # Find the api-gateway result
            api_gateway_result = next((r for r in results if r.service_name == "api-gateway"), None)
            assert api_gateway_result is not None
            assert api_gateway_result.status == ServiceStatus.HEALTHY
            assert api_gateway_result.response_time > 0
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_service(self, monitoring_service):
        """Test health check for an unhealthy service"""
        # Mock HTTP response with error
        mock_response = AsyncMock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            monitoring_service._check_dependencies = AsyncMock(return_value={"all_healthy": True, "dependencies": {}})
            
            results = await monitoring_service.perform_health_checks()
            
            api_gateway_result = next((r for r in results if r.service_name == "api-gateway"), None)
            assert api_gateway_result is not None
            assert api_gateway_result.status == ServiceStatus.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_health_check_service_down(self, monitoring_service):
        """Test health check for a service that's completely down"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.get.side_effect = aiohttp.ClientError("Connection refused")
            
            results = await monitoring_service.perform_health_checks()
            
            api_gateway_result = next((r for r in results if r.service_name == "api-gateway"), None)
            assert api_gateway_result is not None
            assert api_gateway_result.status == ServiceStatus.DOWN
    
    @pytest.mark.asyncio
    async def test_dependency_check_redis_healthy(self, monitoring_service):
        """Test dependency check for healthy Redis"""
        monitoring_service.redis_client.ping = AsyncMock(return_value=True)
        
        result = await monitoring_service._check_dependencies(["redis"])
        
        assert result["all_healthy"] is True
        assert result["dependencies"]["redis"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_dependency_check_redis_unhealthy(self, monitoring_service):
        """Test dependency check for unhealthy Redis"""
        monitoring_service.redis_client.ping = AsyncMock(side_effect=redis.ConnectionError("Connection failed"))
        
        result = await monitoring_service._check_dependencies(["redis"])
        
        assert result["all_healthy"] is False
        assert "unhealthy" in result["dependencies"]["redis"]
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, monitoring_service):
        """Test system metrics collection"""
        # Mock Kubernetes API response
        mock_pod = MagicMock()
        mock_pod.status.phase = "Running"
        mock_pod.spec.containers = [MagicMock()]
        mock_pod.spec.containers[0].resources.requests = {"cpu": "100m", "memory": "256Mi"}
        
        mock_pods = MagicMock()
        mock_pods.items = [mock_pod, mock_pod]  # Two running pods
        
        monitoring_service.k8s_client.list_namespaced_pod.return_value = mock_pods
        monitoring_service.redis_client.keys.return_value = ["queue:test1", "queue:test2"]
        monitoring_service.redis_client.llen.return_value = 5
        
        metrics = await monitoring_service.collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.active_connections == 2
        assert metrics.queue_depth == 10  # 2 queues * 5 items each
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
    
    @pytest.mark.asyncio
    async def test_scaling_decision_scale_up(self, monitoring_service):
        """Test scaling decision for scale up"""
        # Mock high response time
        health_result = HealthCheckResult(
            service_name="api-gateway",
            status=ServiceStatus.HEALTHY,
            response_time=3.0,  # High response time
            timestamp=datetime.utcnow(),
            details={},
            dependencies=[]
        )
        
        monitoring_service.perform_health_checks = AsyncMock(return_value=[health_result])
        
        should_scale = await monitoring_service._should_scale_up("api-gateway", {"max_replicas": 10})
        
        assert should_scale is True
    
    @pytest.mark.asyncio
    async def test_scaling_decision_scale_down(self, monitoring_service):
        """Test scaling decision for scale down"""
        # Mock low response time
        health_result = HealthCheckResult(
            service_name="api-gateway",
            status=ServiceStatus.HEALTHY,
            response_time=0.05,  # Very low response time
            timestamp=datetime.utcnow(),
            details={},
            dependencies=[]
        )
        
        monitoring_service.perform_health_checks = AsyncMock(return_value=[health_result])
        monitoring_service.redis_client.llen.return_value = 2  # Low queue depth
        
        should_scale = await monitoring_service._should_scale_down("api-gateway", {"min_replicas": 1})
        
        assert should_scale is True
    
    @pytest.mark.asyncio
    async def test_alert_generation_service_down(self, monitoring_service):
        """Test alert generation when service goes down"""
        health_result = HealthCheckResult(
            service_name="volcano-watchtower",
            status=ServiceStatus.DOWN,
            response_time=0.0,
            timestamp=datetime.utcnow(),
            details={"error": "Connection refused"},
            dependencies=[]
        )
        
        monitoring_service._process_alert = AsyncMock()
        
        await monitoring_service._analyze_health_results([health_result])
        
        # Verify alert was generated
        monitoring_service._process_alert.assert_called_once()
        alert_call = monitoring_service._process_alert.call_args[0][0]
        assert isinstance(alert_call, Alert)
        assert alert_call.severity == AlertSeverity.EMERGENCY  # Volcanic monitoring is emergency
        assert alert_call.service == "volcano-watchtower"

class TestIncidentResponseSystem:
    """Test cases for the incident response system"""
    
    @pytest.fixture
    def incident_response(self):
        """Create an incident response system for testing"""
        k8s_client = AsyncMock()
        redis_client = AsyncMock()
        return IncidentResponseSystem(k8s_client, redis_client)
    
    @pytest.mark.asyncio
    async def test_incident_creation_service_down(self, incident_response):
        """Test incident creation for service down alert"""
        alert_data = {
            "service": "api-gateway",
            "severity": "critical",
            "alert_type": "service_down",
            "message": "Service is not responding"
        }
        
        incident = await incident_response.process_incident(alert_data)
        
        assert incident is not None
        assert incident.service == "api-gateway"
        assert incident.severity == IncidentSeverity.CRITICAL
        assert ResponseAction.RESTART_SERVICE in incident.actions
        assert ResponseAction.NOTIFY_ONCALL in incident.actions
    
    @pytest.mark.asyncio
    async def test_volcanic_emergency_incident(self, incident_response):
        """Test incident creation for volcanic monitoring emergency"""
        alert_data = {
            "service": "volcano-watchtower",
            "severity": "emergency",
            "alert_type": "service_down",
            "message": "Volcanic monitoring system is down"
        }
        
        incident = await incident_response.process_incident(alert_data)
        
        assert incident is not None
        assert incident.severity == IncidentSeverity.EMERGENCY
        assert ResponseAction.RESTART_SERVICE in incident.actions
        assert ResponseAction.FAILOVER in incident.actions
        assert ResponseAction.ESCALATE in incident.actions
    
    @pytest.mark.asyncio
    async def test_restart_service_action(self, incident_response):
        """Test service restart action"""
        # Mock Kubernetes deployment
        mock_deployment = MagicMock()
        mock_deployment.spec.template.metadata.annotations = {}
        
        incident_response.k8s_client.read_namespaced_deployment.return_value = mock_deployment
        incident_response.k8s_client.patch_namespaced_deployment.return_value = None
        
        success = await incident_response._restart_service("api-gateway")
        
        assert success is True
        incident_response.k8s_client.patch_namespaced_deployment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_scale_service_up(self, incident_response):
        """Test scaling service up"""
        mock_deployment = MagicMock()
        mock_deployment.spec.replicas = 2
        
        incident_response.k8s_client.read_namespaced_deployment.return_value = mock_deployment
        incident_response.k8s_client.patch_namespaced_deployment.return_value = None
        
        success = await incident_response._scale_service("api-gateway", scale_up=True)
        
        assert success is True
        assert mock_deployment.spec.replicas == 3
    
    @pytest.mark.asyncio
    async def test_cooldown_check(self, incident_response):
        """Test action cooldown checking"""
        service = "api-gateway"
        action = ResponseAction.RESTART_SERVICE
        
        # First execution should be allowed
        assert incident_response._check_action_cooldown(service, action, 5) is True
        
        # Record an execution
        action_key = f"{service}:{action.value}"
        incident_response.action_history[action_key] = [datetime.utcnow()]
        
        # Second execution within cooldown should be blocked
        assert incident_response._check_action_cooldown(service, action, 5) is False
        
        # Execution after cooldown should be allowed
        incident_response.action_history[action_key] = [datetime.utcnow() - timedelta(minutes=10)]
        assert incident_response._check_action_cooldown(service, action, 5) is True

class TestNotificationService:
    """Test cases for the notification service"""
    
    @pytest.fixture
    def notification_service(self):
        """Create a notification service for testing"""
        redis_client = AsyncMock()
        return NotificationService(redis_client)
    
    @pytest.mark.asyncio
    async def test_email_notification(self, notification_service):
        """Test email notification sending"""
        request = NotificationRequest(
            id="test-notification-1",
            alert_id="alert-123",
            incident_id=None,
            severity="critical",
            service="api-gateway",
            message="Service is down",
            details={"error": "Connection refused"},
            channels=["ops_email"],
            timestamp=datetime.utcnow()
        )
        
        with patch('smtplib.SMTP') as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value = mock_server
            
            results = await notification_service.send_notification(request)
            
            assert results["ops_email"] is True
            mock_server.sendmail.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_slack_notification(self, notification_service):
        """Test Slack notification sending"""
        request = NotificationRequest(
            id="test-notification-2",
            alert_id="alert-456",
            incident_id=None,
            severity="high",
            service="helios-core",
            message="High CPU usage detected",
            details={"cpu_usage": "85%"},
            channels=["slack_ops"],
            timestamp=datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            results = await notification_service.send_notification(request)
            
            assert results["slack_ops"] is True
    
    @pytest.mark.asyncio
    async def test_pagerduty_notification(self, notification_service):
        """Test PagerDuty notification sending"""
        request = NotificationRequest(
            id="test-notification-3",
            alert_id="alert-789",
            incident_id="incident-123",
            severity="emergency",
            service="volcano-watchtower",
            message="Volcanic monitoring system failure",
            details={"system_status": "down"},
            channels=["pagerduty_emergency"],
            timestamp=datetime.utcnow()
        )
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 202  # PagerDuty success status
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            results = await notification_service.send_notification(request)
            
            assert results["pagerduty_emergency"] is True
    
    def test_rate_limiting(self, notification_service):
        """Test notification rate limiting"""
        channel_name = "ops_email"
        
        # Should allow first notification
        assert notification_service._check_rate_limit(channel_name, 10) is True
        
        # Record 10 notifications in the last hour
        now = datetime.utcnow()
        notification_service.rate_limits[channel_name] = [now - timedelta(minutes=i) for i in range(10)]
        
        # Should block 11th notification
        assert notification_service._check_rate_limit(channel_name, 10) is False
        
        # Should allow notification after old ones expire
        notification_service.rate_limits[channel_name] = [now - timedelta(hours=2)]
        assert notification_service._check_rate_limit(channel_name, 10) is True
    
    @pytest.mark.asyncio
    async def test_escalation_processing(self, notification_service):
        """Test escalation processing"""
        alert_data = {
            "service": "volcano-watchtower",
            "severity": "emergency",
            "message": "System failure",
            "details": {"error": "Critical failure"}
        }
        
        notification_service.redis_client.get.return_value = json.dumps(alert_data)
        
        with patch.object(notification_service, 'send_notification') as mock_send:
            mock_send.return_value = {"emergency_email": True, "pagerduty_emergency": True}
            
            success = await notification_service.process_escalation("alert-123")
            
            assert success is True
            mock_send.assert_called_once()

class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_alert_flow(self):
        """Test complete alert flow from detection to notification"""
        # This would be a more complex integration test
        # that tests the entire flow from health check failure
        # to incident response to notification
        pass
    
    @pytest.mark.asyncio
    async def test_volcanic_emergency_response(self):
        """Test complete volcanic emergency response flow"""
        # Test the critical path for volcanic monitoring failures
        pass

# Performance tests
class TestPerformance:
    """Performance tests for the monitoring system"""
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance under load"""
        # Test that health checks complete within acceptable time limits
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_incident_processing(self):
        """Test concurrent incident processing"""
        # Test that multiple incidents can be processed simultaneously
        pass

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])