#!/usr/bin/env python3
"""
Notification and Escalation Service
Handles alert notifications, escalations, and communication with external systems
"""

import json
import smtplib
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
import redis.asyncio as redis
import structlog
import torch.nn as nn
import torch.nn.functional as F
from jinja2 import Template

logger = structlog.get_logger()

class NotificationType(Enum):
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"

class EscalationLevel(Enum):
    LEVEL_1 = 1  # Team notification
    LEVEL_2 = 2  # On-call notification
    LEVEL_3 = 3  # Management notification
    LEVEL_4 = 4  # Executive notification

@dataclass
class NotificationChannel:
    name: str
    type: NotificationType
    config: Dict[str, Any]
    enabled: bool = True
    rate_limit_per_hour: int = 100

@dataclass
class EscalationRule:
    service: str
    severity: str
    level: EscalationLevel
    timeout_minutes: int
    channels: List[str]
    conditions: Dict[str, Any]

@dataclass
class NotificationRequest:
    id: str
    alert_id: Optional[str]
    incident_id: Optional[str]
    severity: str
    service: str
    message: str
    details: Dict[str, Any]
    channels: List[str]
    timestamp: datetime
    escalation_level: int = 1

class NotificationService:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.channels = self._load_notification_channels()
        self.escalation_rules = self._load_escalation_rules()
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.templates = self._load_templates()

    def _load_notification_channels(self) -> Dict[str, NotificationChannel]:
        """Load notification channel configurations"""
        return {
            "ops_email": NotificationChannel(
                name="Operations Team Email",
                type=NotificationType.EMAIL,
                config={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@prometheus-ai.com",
                    "password": "app-password",  # Use environment variable in production
                    "recipients": ["ops-team@prometheus-ai.com"],
                },
                rate_limit_per_hour=50,
            ),
            "critical_email": NotificationChannel(
                name="Critical Alerts Email",
                type=NotificationType.EMAIL,
                config={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@prometheus-ai.com",
                    "password": "app-password",
                    "recipients": [
                        "critical-alerts@prometheus-ai.com",
                        "cto@prometheus-ai.com",
                    ],
                },
                rate_limit_per_hour=100,
            ),
            "emergency_email": NotificationChannel(
                name="Emergency Alerts Email",
                type=NotificationType.EMAIL,
                config={
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "alerts@prometheus-ai.com",
                    "password": "app-password",
                    "recipients": [
                        "emergency@prometheus-ai.com",
                        "ceo@prometheus-ai.com",
                        "cto@prometheus-ai.com",
                        "head-of-operations@prometheus-ai.com",
                    ],
                },
                rate_limit_per_hour=200,
            ),
            "slack_ops": NotificationChannel(
                name="Operations Slack Channel",
                type=NotificationType.SLACK,
                config={
                    "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
                    "channel": "#prometheus-ai-alerts",
                    "username": "Prometheus AI Monitor",
                },
                rate_limit_per_hour=100,
            ),
            "slack_critical": NotificationChannel(
                name="Critical Alerts Slack Channel",
                type=NotificationType.SLACK,
                config={
                    "webhook_url": "https://hooks.slack.com/services/YOUR/CRITICAL/WEBHOOK",
                    "channel": "#prometheus-ai-critical",
                    "username": "Prometheus AI Critical",
                },
                rate_limit_per_hour=200,
            ),
            "pagerduty_oncall": NotificationChannel(
                name="PagerDuty On-Call",
                type=NotificationType.PAGERDUTY,
                config={
                    "integration_key": "YOUR_PAGERDUTY_INTEGRATION_KEY",
                    "api_url": "https://events.pagerduty.com/v2/enqueue",
                },
                rate_limit_per_hour=50,
            ),
            "pagerduty_emergency": NotificationChannel(
                name="PagerDuty Emergency",
                type=NotificationType.PAGERDUTY,
                config={
                    "integration_key": "YOUR_EMERGENCY_PAGERDUTY_KEY",
                    "api_url": "https://events.pagerduty.com/v2/enqueue",
                },
                rate_limit_per_hour=100,
            ),
            "webhook_external": NotificationChannel(
                name="External Systems Webhook",
                type=NotificationType.WEBHOOK,
                config={
                    "url": "https://external-monitoring.example.com/webhook",
                    "headers": {
                        "Authorization": "Bearer YOUR_API_TOKEN",
                        "Content-Type": "application/json",
                    },
                },
                rate_limit_per_hour=200,
            ),
        }

    def _load_escalation_rules(self) -> List[EscalationRule]:
        """Load escalation rules for different services and severities"""
        return [
            # Volcanic monitoring emergency rules
            EscalationRule(
                service="volcano-watchtower",
                severity="emergency",
                level=EscalationLevel.LEVEL_4,
                timeout_minutes=0,  # Immediate escalation
                channels=["emergency_email", "pagerduty_emergency", "slack_critical"],
                conditions={"immediate": True},
            ),
            EscalationRule(
                service="volcano-watchtower",
                severity="critical",
                level=EscalationLevel.LEVEL_3,
                timeout_minutes=5,
                channels=["critical_email", "pagerduty_oncall", "slack_critical"],
                conditions={},
            ),
            # General critical service rules
            EscalationRule(
                service="*",  # All services
                severity="critical",
                level=EscalationLevel.LEVEL_2,
                timeout_minutes=15,
                channels=["critical_email", "pagerduty_oncall", "slack_critical"],
                conditions={"critical_service": True},
            ),
            # High severity rules
            EscalationRule(
                service="*",
                severity="high",
                level=EscalationLevel.LEVEL_2,
                timeout_minutes=30,
                channels=["ops_email", "slack_ops"],
                conditions={},
            ),
            # Medium severity rules
            EscalationRule(
                service="*",
                severity="medium",
                level=EscalationLevel.LEVEL_1,
                timeout_minutes=60,
                channels=["ops_email", "slack_ops"],
                conditions={},
            ),
            # Low severity rules
            EscalationRule(
                service="*",
                severity="low",
                level=EscalationLevel.LEVEL_1,
                timeout_minutes=120,
                channels=["slack_ops"],
                conditions={},
            ),
        ]

    def _load_templates(self) -> Dict[str, Template]:
        """Load notification templates"""
        return {
            "email_alert": Template(
                """
Subject: [{{ severity.upper() }}] Prometheus AI Alert: {{ service }}

Alert Details:
    - Service: {{ service }}
- Severity: {{ severity }}
- Message: {{ message }}
- Timestamp: {{ timestamp }}
- Alert ID: {{ alert_id }}
{% if incident_id %}
- Incident ID: {{ incident_id }}
{% endif %}

{% if details %}
Additional Details:
    {% for key, value in details.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if escalation_level > 1 %}
⚠️  This alert has been escalated to Level {{ escalation_level }}
{% endif %}

Dashboard: https://monitoring.prometheus-ai.com/alerts/{{ alert_id }}

This is an automated message from Prometheus AI Monitoring System.
            """
            ),
            "slack_alert": Template(
                """
{
    "channel": "{{ channel }}",
    "username": "{{ username }}",
    "icon_emoji": ":warning:",
    "attachments": [
        {
            "color": "{{ color }}",
            "title": "{{ severity.upper() }} Alert: {{ service }}",
            "text": "{{ message }}",
            "fields": [
                {
                    "title": "Service",
                    "value": "{{ service }}",
                    "short": true
                },
                {
                    "title": "Severity",
                    "value": "{{ severity }}",
                    "short": true
                },
                {
                    "title": "Timestamp",
                    "value": "{{ timestamp }}",
                    "short": true
                }
                {% if alert_id %},
                {
                    "title": "Alert ID",
                    "value": "{{ alert_id }}",
                    "short": true
                }
                {% endif %}
                {% if incident_id %},
                {
                    "title": "Incident ID",
                    "value": "{{ incident_id }}",
                    "short": true
                }
                {% endif %}
            ],
            "actions": [
                {
                    "type": "button",
                    "text": "View Dashboard",
                    "url": "https://monitoring.prometheus-ai.com/alerts/{{ alert_id }}"
                }
                {% if alert_id %},
                {
                    "type": "button",
                    "text": "Acknowledge",
                    "url": "https://monitoring.prometheus-ai.com/alerts/{{ alert_id }}/ack"
                }
                {% endif %}
            ]
        }
    ]
}
            """
            ),
            "pagerduty_alert": Template(
                """
{
    "routing_key": "{{ integration_key }}",
    "event_action": "trigger",
    "dedup_key": "{{ alert_id or incident_id }}",
    "payload": {
        "summary": "{{ severity.upper() }}: {{ service }} - {{ message }}",
        "source": "prometheus-ai-monitoring",
        "severity": "{{ pagerduty_severity }}",
        "component": "{{ service }}",
        "group": "prometheus-ai",
        "class": "{{ severity }}",
        "custom_details": {{ details | tojson }}
    },
    "links": [
        {
            "hre": "https://monitoring.prometheus-ai.com/alerts/{{ alert_id }}",
            "text": "View in Dashboard"
        }
    ]
}
            """
            ),
            "webhook_alert": Template(
                """
{
    "alert_id": "{{ alert_id }}",
    "incident_id": "{{ incident_id }}",
    "service": "{{ service }}",
    "severity": "{{ severity }}",
    "message": "{{ message }}",
    "timestamp": "{{ timestamp }}",
    "escalation_level": {{ escalation_level }},
    "details": {{ details | tojson }},
    "source": "prometheus-ai-monitoring"
}
            """
            ),
        }

    async def send_notification(self, request: NotificationRequest) -> Dict[str, bool]:
        """Send notification through specified channels"""
        results = {}

        for channel_name in request.channels:
            if channel_name not in self.channels:
                logger.warning(f"Unknown notification channel: {channel_name}")
                results[channel_name] = False
                continue

            channel = self.channels[channel_name]

            if not channel.enabled:
                logger.info(f"Channel {channel_name} is disabled")
                results[channel_name] = False
                continue

            # Check rate limits
            if not self._check_rate_limit(channel_name, channel.rate_limit_per_hour):
                logger.warning(f"Rate limit exceeded for channel {channel_name}")
                results[channel_name] = False
                continue

            try:
                success = await self._send_to_channel(request, channel)
                results[channel_name] = success

                if success:
                    self._record_rate_limit(channel_name)
                    logger.info(
                        "Notification sent successfully",
                        channel=channel_name,
                        alert_id=request.alert_id,
                        incident_id=request.incident_id,
                    )
                else:
                    logger.error(
                        "Failed to send notification",
                        channel=channel_name,
                        alert_id=request.alert_id,
                        incident_id=request.incident_id,
                    )

            except Exception as e:
                logger.error(
                    "Exception sending notification",
                    channel=channel_name,
                    error=str(e),
                    alert_id=request.alert_id,
                    incident_id=request.incident_id,
                )
                results[channel_name] = False

        return results

    def _check_rate_limit(self, channel_name: str, limit_per_hour: int) -> bool:
        """Check if channel is within rate limits"""
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)

        if channel_name not in self.rate_limits:
            self.rate_limits[channel_name] = []

        # Clean old entries
        self.rate_limits[channel_name] = [
            timestamp
            for timestamp in self.rate_limits[channel_name]
            if timestamp > one_hour_ago
        ]

        return len(self.rate_limits[channel_name]) < limit_per_hour

    def _record_rate_limit(self, channel_name: str):
        """Record a notification for rate limiting"""
        if channel_name not in self.rate_limits:
            self.rate_limits[channel_name] = []

        self.rate_limits[channel_name].append(datetime.utcnow())

    async def _send_to_channel(
        self, request: NotificationRequest, channel: NotificationChannel
    ) -> bool:
        """Send notification to a specific channel"""
        try:
            if channel.type == NotificationType.EMAIL:
                return await self._send_email(request, channel)
            elif channel.type == NotificationType.SLACK:
                return await self._send_slack(request, channel)
            elif channel.type == NotificationType.PAGERDUTY:
                return await self._send_pagerduty(request, channel)
            elif channel.type == NotificationType.WEBHOOK:
                return await self._send_webhook(request, channel)
            else:
                logger.warning(f"Unsupported notification type: {channel.type}")
                return False

        except Exception as e:
            logger.error(f"Failed to send to {channel.type.value}", error=str(e))
            return False

    async def _send_email(
        self, request: NotificationRequest, channel: NotificationChannel
    ) -> bool:
        """Send email notification"""
        try:
            config = channel.config

            # Render email template
            template_data = {
                **asdict(request),
                "timestamp": request.timestamp.isoformat(),
            }

            email_content = self.templates["email_alert"].render(**template_data)

            # Parse subject and body
            lines = email_content.strip().split("\n")
            subject = lines[0].replace("Subject: ", "")
            body = "\n".join(lines[2:])  # Skip subject and empty line

            # Create message
            msg = MIMEMultipart()
            msg["From"] = config["username"]
            msg["To"] = ", ".join(config["recipients"])
            msg["Subject"] = subject

            msg.attach(MIMEText(body, "plain"))

            # Send email
            server = smtplib.SMTP(config["smtp_server"], config["smtp_port"])
            server.starttls()
            server.login(config["username"], config["password"])

            text = msg.as_string()
            server.sendmail(config["username"], config["recipients"], text)
            server.quit()

            return True

        except Exception as e:
            logger.error("Failed to send email", error=str(e))
            return False

    async def _send_slack(
        self, request: NotificationRequest, channel: NotificationChannel
    ) -> bool:
        """Send Slack notification"""
        try:
            config = channel.config

            # Determine color based on severity
            color_map = {
                "low": "good",
                "medium": "warning",
                "high": "warning",
                "critical": "danger",
                "emergency": "danger",
            }

            template_data = {
                **asdict(request),
                "timestamp": request.timestamp.isoformat(),
                "channel": config["channel"],
                "username": config["username"],
                "color": color_map.get(request.severity, "warning"),
            }

            payload = json.loads(self.templates["slack_alert"].render(**template_data))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["webhook_url"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error("Failed to send Slack notification", error=str(e))
            return False

    async def _send_pagerduty(
        self, request: NotificationRequest, channel: NotificationChannel
    ) -> bool:
        """Send PagerDuty notification"""
        try:
            config = channel.config

            # Map severity to PagerDuty severity
            severity_map = {
                "low": "info",
                "medium": "warning",
                "high": "error",
                "critical": "critical",
                "emergency": "critical",
            }

            template_data = {
                **asdict(request),
                "timestamp": request.timestamp.isoformat(),
                "integration_key": config["integration_key"],
                "pagerduty_severity": severity_map.get(request.severity, "error"),
            }

            payload = json.loads(
                self.templates["pagerduty_alert"].render(**template_data)
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["api_url"],
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status == 202  # PagerDuty returns 202 for success

        except Exception as e:
            logger.error("Failed to send PagerDuty notification", error=str(e))
            return False

    async def _send_webhook(
        self, request: NotificationRequest, channel: NotificationChannel
    ) -> bool:
        """Send webhook notification"""
        try:
            config = channel.config

            template_data = {
                **asdict(request),
                "timestamp": request.timestamp.isoformat(),
            }

            payload = json.loads(
                self.templates["webhook_alert"].render(**template_data)
            )

            headers = config.get("headers", {})

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config["url"],
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return 200 <= response.status < 300

        except Exception as e:
            logger.error("Failed to send webhook notification", error=str(e))
            return False

    async def process_escalation(
        self, alert_id: str, incident_id: Optional[str] = None
    ) -> bool:
        """Process escalation for an alert or incident"""
        try:
            # Get alert/incident details from Redis
            if alert_id:
                alert_data = await self.redis_client.get(f"alert:{alert_id}")
                if not alert_data:
                    logger.warning(f"Alert {alert_id} not found for escalation")
                    return False

                alert_info = json.loads(alert_data)
            else:
                logger.warning("No alert_id provided for escalation")
                return False

            # Find matching escalation rules
            matching_rules = self._find_escalation_rules(
                alert_info["service"], alert_info["severity"]
            )

            if not matching_rules:
                logger.info(
                    f"No escalation rules found for {alert_info['service']}/{alert_info['severity']}"
                )
                return False

            # Apply the first matching rule
            rule = matching_rules[0]

            # Create escalation notification
            escalation_request = NotificationRequest(
                id=f"escalation-{alert_id}-{int(time.time())}",
                alert_id=alert_id,
                incident_id=incident_id,
                severity=alert_info["severity"],
                service=alert_info["service"],
                message=f"ESCALATED: {alert_info['message']}",
                details={
                    **alert_info.get("details", {}),
                    "escalation_level": rule.level.value,
                    "escalation_reason": "Timeout exceeded",
                },
                channels=rule.channels,
                timestamp=datetime.utcnow(),
                escalation_level=rule.level.value,
            )

            # Send escalation notifications
            results = await self.send_notification(escalation_request)

            # Log escalation
            logger.warning(
                "Alert escalated",
                alert_id=alert_id,
                incident_id=incident_id,
                escalation_level=rule.level.value,
                channels=rule.channels,
                results=results,
            )

            return any(results.values())

        except Exception as e:
            logger.error("Failed to process escalation", error=str(e))
            return False

    def _find_escalation_rules(
        self, service: str, severity: str
    ) -> List[EscalationRule]:
        """Find matching escalation rules"""
        matching_rules = []

        for rule in self.escalation_rules:
            if (
                rule.service == "*" or rule.service == service
            ) and rule.severity == severity:
                matching_rules.append(rule)

        # Sort by escalation level (higher levels first for emergency situations)
        matching_rules.sort(key=lambda r: r.level.value, reverse=True)

        return matching_rules

    async def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        stats = {"channels": {}, "rate_limits": {}, "total_sent_last_hour": 0}

        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)

        for channel_name, timestamps in self.rate_limits.items():
            recent_notifications = [t for t in timestamps if t > one_hour_ago]
            stats["channels"][channel_name] = {
                "sent_last_hour": len(recent_notifications),
                "rate_limit": self.channels[channel_name].rate_limit_per_hour,
                "enabled": self.channels[channel_name].enabled,
            }
            stats["total_sent_last_hour"] += len(recent_notifications)

        return stats
