"""
Alert Level Management and Notification System
Handles alert level determination, escalation, and emergency response triggers
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

import structlog
from models.volcanic_data import AlertLevel, VolcanicRiskAssessment

logger = structlog.get_logger()

class AlertPriority(Enum):
    """Alert priority levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AlertHistory:
    """Historical alert information"""

    volcano_id: str
    timestamp: datetime
    alert_level: AlertLevel
    duration: timedelta
    escalated: bool = False
    acknowledged: bool = False

class AlertLevelManager:
    """
    Manages volcanic alert levels and escalation logic

    Handles:
    - Alert level determination and updates
    - Alert escalation based on duration and conditions
    - Alert acknowledgment and resolution
    - Historical alert tracking
    """

    def __init__(self):
        # Current alert levels for each volcano
        self.current_alerts: Dict[str, AlertLevel] = {}

        # Alert history
        self.alert_history: Dict[str, List[AlertHistory]] = {}

        # Escalation rules
        self.escalation_rules = {
            1: {"duration_hours": 24, "escalate_to": 2},  # Advisory -> Watch after 24h
            2: {"duration_hours": 12, "escalate_to": 3},  # Watch -> Warning after 12h
            3: {"duration_hours": 6, "escalate_to": 4},  # Warning -> Alert after 6h
            4: {"duration_hours": 2, "escalate_to": 5},  # Alert -> Emergency after 2h
        }

        # Alert acknowledgment tracking
        self.acknowledged_alerts: Set[str] = set()

        logger.info("Alert Level Manager initialized")

    async def get_current_alert_level(self, volcano_id: str) -> AlertLevel:
        """Get current alert level for a volcano"""
        return self.current_alerts.get(
            volcano_id,
            AlertLevel(
                level=0,
                description="Normal",
                recommended_actions=["Continue routine monitoring"],
                validity_period=(
                    datetime.utcnow(),
                    datetime.utcnow() + timedelta(hours=24),
                ),
                confidence=1.0,
            ),
        )

    async def update_alert_level(self, volcano_id: str, new_alert: AlertLevel) -> bool:
        """
        Update alert level for a volcano

        Args:
            volcano_id: Volcano identifier
            new_alert: New alert level

        Returns:
            bool: True if alert level changed
        """
        current_alert = await self.get_current_alert_level(volcano_id)

        # Check if alert level has changed
        if current_alert.level != new_alert.level:
            logger.info(
                "Alert level changed",
                volcano_id=volcano_id,
                old_level=current_alert.level,
                new_level=new_alert.level,
            )

            # Archive current alert if it exists
            if volcano_id in self.current_alerts:
                await self._archive_alert(volcano_id, current_alert)

            # Set new alert
            self.current_alerts[volcano_id] = new_alert

            # Initialize alert history if needed
            if volcano_id not in self.alert_history:
                self.alert_history[volcano_id] = []

            # Add to history
            self.alert_history[volcano_id].append(
                AlertHistory(
                    volcano_id=volcano_id,
                    timestamp=datetime.utcnow(),
                    alert_level=new_alert,
                    duration=timedelta(0),
                )
            )

            return True

        return False

    async def check_escalation_candidates(self) -> Dict[str, Dict]:
        """
        Check for alerts that need escalation based on duration

        Returns:
            Dict mapping volcano_id to escalation info
        """
        escalation_candidates = {}

        for volcano_id, current_alert in self.current_alerts.items():
            if current_alert.level in self.escalation_rules:
                # Get alert start time
                alert_start = await self._get_alert_start_time(
                    volcano_id, current_alert.level
                )

                if alert_start:
                    duration = datetime.utcnow() - alert_start
                    escalation_rule = self.escalation_rules[current_alert.level]

                    if (
                        duration.total_seconds()
                        >= escalation_rule["duration_hours"] * 3600
                    ):
                        escalation_candidates[volcano_id] = {
                            "current_level": current_alert.level,
                            "escalate_to": escalation_rule["escalate_to"],
                            "duration": duration,
                            "alert_info": current_alert,
                        }

                        logger.warning(
                            "Alert escalation candidate identified",
                            volcano_id=volcano_id,
                            current_level=current_alert.level,
                            duration_hours=duration.total_seconds() / 3600,
                        )

        return escalation_candidates

    async def escalate_alert(
        self, volcano_id: str, escalation_info: Dict
    ) -> AlertLevel:
        """
        Escalate alert level for a volcano

        Args:
            volcano_id: Volcano identifier
            escalation_info: Escalation information from check_escalation_candidates

        Returns:
            AlertLevel: New escalated alert level
        """
        new_level = escalation_info["escalate_to"]
        current_alert = escalation_info["alert_info"]

        # Create escalated alert
        escalated_alert = AlertLevel(
            level=new_level,
            description=self._get_alert_description(new_level),
            recommended_actions=self._get_recommended_actions(new_level),
            validity_period=(
                datetime.utcnow(),
                datetime.utcnow()
                + timedelta(hours=self._get_alert_validity_hours(new_level)),
            ),
            confidence=current_alert.confidence
            * 0.9,  # Slightly reduce confidence for escalated alerts
        )

        # Update alert level
        await self.update_alert_level(volcano_id, escalated_alert)

        # Mark as escalated in history
        if volcano_id in self.alert_history and self.alert_history[volcano_id]:
            self.alert_history[volcano_id][-1].escalated = True

        logger.warning(
            "Alert escalated",
            volcano_id=volcano_id,
            old_level=escalation_info["current_level"],
            new_level=new_level,
        )

        return escalated_alert

    async def acknowledge_alert(self, volcano_id: str, acknowledger: str) -> bool:
        """
        Acknowledge an alert

        Args:
            volcano_id: Volcano identifier
            acknowledger: Person/system acknowledging the alert

        Returns:
            bool: True if acknowledgment successful
        """
        alert_key = f"{volcano_id}_{datetime.utcnow().date()}"

        if alert_key not in self.acknowledged_alerts:
            self.acknowledged_alerts.add(alert_key)

            # Mark in history
            if volcano_id in self.alert_history and self.alert_history[volcano_id]:
                self.alert_history[volcano_id][-1].acknowledged = True

            logger.info(
                "Alert acknowledged", volcano_id=volcano_id, acknowledger=acknowledger
            )

            return True

        return False

    async def get_alert_statistics(self, volcano_id: str, days: int = 30) -> Dict:
        """Get alert statistics for a volcano over specified period"""
        if volcano_id not in self.alert_history:
            return {
                "total_alerts": 0,
                "escalated_alerts": 0,
                "average_duration": 0,
                "max_level_reached": 0,
            }

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_alerts = [
            alert
            for alert in self.alert_history[volcano_id]
            if alert.timestamp >= cutoff_date
        ]

        if not recent_alerts:
            return {
                "total_alerts": 0,
                "escalated_alerts": 0,
                "average_duration": 0,
                "max_level_reached": 0,
            }

        total_alerts = len(recent_alerts)
        escalated_alerts = sum(1 for alert in recent_alerts if alert.escalated)
        average_duration = (
            sum(alert.duration.total_seconds() for alert in recent_alerts)
            / total_alerts
            / 3600
        )  # Convert to hours
        max_level = max(alert.alert_level.level for alert in recent_alerts)

        return {
            "total_alerts": total_alerts,
            "escalated_alerts": escalated_alerts,
            "average_duration": average_duration,
            "max_level_reached": max_level,
        }

    async def _get_alert_start_time(
        self, volcano_id: str, alert_level: int
    ) -> Optional[datetime]:
        """Get the start time of the current alert level"""
        if volcano_id not in self.alert_history:
            return None

        # Find the most recent alert with this level
        for alert in reversed(self.alert_history[volcano_id]):
            if alert.alert_level.level == alert_level:
                return alert.timestamp

        return None

    async def _archive_alert(self, volcano_id: str, alert: AlertLevel):
        """Archive an alert when it's replaced"""
        if volcano_id in self.alert_history and self.alert_history[volcano_id]:
            # Update duration of the last alert
            last_alert = self.alert_history[volcano_id][-1]
            last_alert.duration = datetime.utcnow() - last_alert.timestamp

    def _get_alert_description(self, level: int) -> str:
        """Get description for alert level"""
        descriptions = {
            0: "Normal",
            1: "Advisory",
            2: "Watch",
            3: "Warning",
            4: "Alert",
            5: "Emergency",
        }
        return descriptions.get(level, "Unknown")

    def _get_recommended_actions(self, level: int) -> List[str]:
        """Get recommended actions for alert level"""
        actions = {
            0: ["Continue routine monitoring"],
            1: ["Increase monitoring frequency", "Review evacuation plans"],
            2: [
                "Enhanced monitoring",
                "Prepare emergency services",
                "Public information updates",
            ],
            3: [
                "Continuous monitoring",
                "Evacuate high-risk areas",
                "Close access roads",
            ],
            4: [
                "Emergency response activation",
                "Full evacuation",
                "Aviation warnings",
            ],
            5: [
                "Maximum emergency response",
                "Complete evacuation",
                "International assistance",
            ],
        }
        return actions.get(level, ["Assess situation"])

    def _get_alert_validity_hours(self, level: int) -> int:
        """Get validity period in hours for alert level"""
        validity_hours = {
            0: 24,  # Normal - 24 hours
            1: 12,  # Advisory - 12 hours
            2: 8,  # Watch - 8 hours
            3: 6,  # Warning - 6 hours
            4: 4,  # Alert - 4 hours
            5: 2,  # Emergency - 2 hours
        }
        return validity_hours.get(level, 6)

class NotificationSystem:
    """
    Handles automated notifications and emergency response triggers

    Manages:
    - Multi-channel notification delivery
    - Emergency response coordination
    - Notification escalation and retry logic
    - Notification history and acknowledgments
    """

    def __init__(self):
        # Notification targets by alert level
        self.notification_targets = {
            1: ["observatory_sta", "local_authorities"],
            2: ["observatory_sta", "local_authorities", "emergency_services"],
            3: [
                "observatory_sta",
                "local_authorities",
                "emergency_services",
                "media",
            ],
            4: ["all_stakeholders", "aviation_authorities", "international_partners"],
            5: [
                "all_stakeholders",
                "aviation_authorities",
                "international_partners",
                "disaster_response",
            ],
        }

        # Notification channels
        self.notification_channels = {
            "email": {"enabled": True, "priority": 1},
            "sms": {"enabled": True, "priority": 2},
            "push": {"enabled": True, "priority": 3},
            "webhook": {"enabled": True, "priority": 4},
            "emergency_broadcast": {"enabled": True, "priority": 5},
        }

        # Retry configuration
        self.retry_config = {
            "max_retries": 3,
            "retry_delay_seconds": 300,  # 5 minutes
            "escalation_delay_seconds": 1800,  # 30 minutes
        }

        # Notification history
        self.notification_history: List[Dict] = []

        logger.info("Notification System initialized")

    async def send_alert_notification(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ) -> bool:
        """
        Send alert notification for volcanic risk assessment

        Args:
            volcano_id: Volcano identifier
            risk_assessment: Current risk assessment

        Returns:
            bool: True if notification sent successfully
        """
        alert_level = risk_assessment.alert_level.level

        logger.info(
            "Sending alert notification", volcano_id=volcano_id, alert_level=alert_level
        )

        try:
            # Determine notification targets
            targets = self._get_notification_targets(alert_level)

            # Create notification message
            message = await self._create_alert_message(volcano_id, risk_assessment)

            # Determine priority
            priority = self._get_notification_priority(alert_level)

            # Send notifications through all channels
            success = await self._send_multi_channel_notification(
                targets, message, priority, volcano_id
            )

            # Log notification
            await self._log_notification(volcano_id, alert_level, targets, success)

            # Trigger emergency response if needed
            if alert_level >= 4:  # Alert level or higher
                await self._trigger_emergency_response(volcano_id, risk_assessment)

            return success

        except Exception as e:
            logger.error(
                "Error sending alert notification", volcano_id=volcano_id, error=str(e)
            )
            return False

    async def escalate_alert(self, volcano_id: str, escalation_info: Dict) -> bool:
        """
        Send escalation notification

        Args:
            volcano_id: Volcano identifier
            escalation_info: Escalation information

        Returns:
            bool: True if escalation notification sent successfully
        """
        logger.warning(
            "Sending alert escalation notification",
            volcano_id=volcano_id,
            escalation_info=escalation_info,
        )

        try:
            # Create escalation message
            message = await self._create_escalation_message(volcano_id, escalation_info)

            # Send to all high-priority targets
            targets = ["all_stakeholders", "emergency_services", "media"]
            priority = AlertPriority.CRITICAL

            success = await self._send_multi_channel_notification(
                targets, message, priority, volcano_id
            )

            # Log escalation
            await self._log_notification(
                volcano_id,
                escalation_info["escalate_to"],
                targets,
                success,
                notification_type="escalation",
            )

            return success

        except Exception as e:
            logger.error(
                "Error sending escalation notification",
                volcano_id=volcano_id,
                error=str(e),
            )
            return False

    async def send_emergency_broadcast(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ) -> bool:
        """
        Send emergency broadcast for critical situations

        Args:
            volcano_id: Volcano identifier
            risk_assessment: Current risk assessment

        Returns:
            bool: True if broadcast sent successfully
        """
        logger.critical(
            "Sending emergency broadcast",
            volcano_id=volcano_id,
            alert_level=risk_assessment.alert_level.level,
        )

        try:
            # Create emergency message
            message = await self._create_emergency_message(volcano_id, risk_assessment)

            # Send through emergency broadcast system
            success = await self._send_emergency_broadcast(message, volcano_id)

            # Also send through all other channels
            targets = ["all_stakeholders"]
            await self._send_multi_channel_notification(
                targets, message, AlertPriority.EMERGENCY, volcano_id
            )

            # Log emergency broadcast
            await self._log_notification(
                volcano_id,
                risk_assessment.alert_level.level,
                ["emergency_broadcast"],
                success,
                notification_type="emergency_broadcast",
            )

            return success

        except Exception as e:
            logger.error(
                "Error sending emergency broadcast", volcano_id=volcano_id, error=str(e)
            )
            return False

    async def _send_multi_channel_notification(
        self,
        targets: List[str],
        message: Dict,
        priority: AlertPriority,
        volcano_id: str,
    ) -> bool:
        """Send notification through multiple channels"""

        success_count = 0
        total_channels = len(self.notification_channels)

        for channel, config in self.notification_channels.items():
            if not config["enabled"]:
                continue

            try:
                channel_success = await self._send_channel_notification(
                    channel, targets, message, priority, volcano_id
                )

                if channel_success:
                    success_count += 1

            except Exception as e:
                logger.error(
                    "Error sending notification through channel",
                    channel=channel,
                    volcano_id=volcano_id,
                    error=str(e),
                )

        # Consider successful if at least 50% of channels succeeded
        return success_count >= (total_channels * 0.5)

    async def _send_channel_notification(
        self,
        channel: str,
        targets: List[str],
        message: Dict,
        priority: AlertPriority,
        volcano_id: str,
    ) -> bool:
        """Send notification through specific channel"""

        # This would integrate with actual notification services
        # For now, simulate notification sending

        logger.info(
            "Sending notification through channel",
            channel=channel,
            targets=targets,
            priority=priority.value,
            volcano_id=volcano_id,
        )

        # Simulate different success rates by channel
        success_rates = {
            "email": 0.95,
            "sms": 0.90,
            "push": 0.85,
            "webhook": 0.98,
            "emergency_broadcast": 0.99,
        }

        import random

        success_rate = success_rates.get(channel, 0.90)
        success = random.random() < success_rate

        if success:
            logger.info(f"Notification sent successfully via {channel}")
        else:
            logger.warning(f"Notification failed via {channel}")

        return success

    async def _send_emergency_broadcast(self, message: Dict, volcano_id: str) -> bool:
        """Send emergency broadcast through emergency alert system"""

        logger.critical(
            "Activating emergency broadcast system",
            volcano_id=volcano_id,
            message_type="volcanic_emergency",
        )

        # This would integrate with emergency broadcast systems
        # For now, simulate emergency broadcast

        # Emergency broadcasts have very high reliability
        success = random.random() < 0.99

        if success:
            logger.info("Emergency broadcast sent successfully")
        else:
            logger.error("Emergency broadcast failed")

        return success

    async def _trigger_emergency_response(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ):
        """Trigger automated emergency response procedures"""

        logger.critical(
            "Triggering emergency response procedures",
            volcano_id=volcano_id,
            alert_level=risk_assessment.alert_level.level,
        )

        # Emergency response actions based on alert level
        if risk_assessment.alert_level.level >= 4:
            # Alert level - activate emergency services
            await self._activate_emergency_services(volcano_id, risk_assessment)

        if risk_assessment.alert_level.level >= 5:
            # Emergency level - full emergency response
            await self._activate_full_emergency_response(volcano_id, risk_assessment)

    async def _activate_emergency_services(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ):
        """Activate emergency services"""

        logger.critical("Activating emergency services", volcano_id=volcano_id)

        # This would integrate with emergency service APIs
        # For now, log the activation

        emergency_actions = [
            "Evacuate high-risk areas",
            "Close access roads",
            "Activate emergency shelters",
            "Deploy emergency personnel",
            "Issue aviation warnings",
        ]

        for action in emergency_actions:
            logger.info(f"Emergency action: {action}", volcano_id=volcano_id)

    async def _activate_full_emergency_response(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ):
        """Activate full emergency response"""

        logger.critical("Activating full emergency response", volcano_id=volcano_id)

        emergency_actions = [
            "Complete evacuation of danger zones",
            "International assistance request",
            "Military support activation",
            "Mass casualty response preparation",
            "Media coordination center activation",
        ]

        for action in emergency_actions:
            logger.critical(f"Full emergency action: {action}", volcano_id=volcano_id)

    def _get_notification_targets(self, alert_level: int) -> List[str]:
        """Get notification targets for alert level"""
        return self.notification_targets.get(alert_level, ["observatory_sta"])

    def _get_notification_priority(self, alert_level: int) -> AlertPriority:
        """Get notification priority for alert level"""
        priority_map = {
            0: AlertPriority.LOW,
            1: AlertPriority.LOW,
            2: AlertPriority.MEDIUM,
            3: AlertPriority.HIGH,
            4: AlertPriority.CRITICAL,
            5: AlertPriority.EMERGENCY,
        }
        return priority_map.get(alert_level, AlertPriority.MEDIUM)

    async def _create_alert_message(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ) -> Dict:
        """Create alert notification message"""

        alert_level = risk_assessment.alert_level

        message = {
            "type": "volcanic_alert",
            "volcano_id": volcano_id,
            "timestamp": datetime.utcnow().isoformat(),
            "alert_level": alert_level.level,
            "alert_description": alert_level.description,
            "eruption_probability": risk_assessment.eruption_probability,
            "recommended_actions": alert_level.recommended_actions,
            "contributing_factors": risk_assessment.contributing_factors,
            "confidence": risk_assessment.estimation_confidence,
            "validity_period": {
                "start": alert_level.validity_period[0].isoformat(),
                "end": alert_level.validity_period[1].isoformat(),
            },
        }

        return message

    async def _create_escalation_message(
        self, volcano_id: str, escalation_info: Dict
    ) -> Dict:
        """Create escalation notification message"""

        message = {
            "type": "alert_escalation",
            "volcano_id": volcano_id,
            "timestamp": datetime.utcnow().isoformat(),
            "previous_level": escalation_info["current_level"],
            "new_level": escalation_info["escalate_to"],
            "duration": str(escalation_info["duration"]),
            "reason": "Automatic escalation due to sustained alert conditions",
        }

        return message

    async def _create_emergency_message(
        self, volcano_id: str, risk_assessment: VolcanicRiskAssessment
    ) -> Dict:
        """Create emergency broadcast message"""

        message = {
            "type": "volcanic_emergency",
            "volcano_id": volcano_id,
            "timestamp": datetime.utcnow().isoformat(),
            "alert_level": risk_assessment.alert_level.level,
            "emergency_actions": risk_assessment.alert_level.recommended_actions,
            "immediate_threat": True,
            "evacuation_required": risk_assessment.alert_level.level >= 4,
        }

        return message

    async def _log_notification(
        self,
        volcano_id: str,
        alert_level: int,
        targets: List[str],
        success: bool,
        notification_type: str = "alert",
    ):
        """Log notification attempt"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "volcano_id": volcano_id,
            "alert_level": alert_level,
            "notification_type": notification_type,
            "targets": targets,
            "success": success,
        }

        self.notification_history.append(log_entry)

        # Keep only last 1000 entries
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]

    async def get_notification_statistics(self, days: int = 30) -> Dict:
        """Get notification statistics over specified period"""

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        recent_notifications = [
            notif
            for notif in self.notification_history
            if datetime.fromisoformat(notif["timestamp"]) >= cutoff_date
        ]

        if not recent_notifications:
            return {
                "total_notifications": 0,
                "success_rate": 0.0,
                "emergency_broadcasts": 0,
                "escalations": 0,
            }

        total_notifications = len(recent_notifications)
        successful_notifications = sum(1 for n in recent_notifications if n["success"])
        success_rate = successful_notifications / total_notifications

        emergency_broadcasts = sum(
            1
            for n in recent_notifications
            if n["notification_type"] == "emergency_broadcast"
        )

        escalations = sum(
            1 for n in recent_notifications if n["notification_type"] == "escalation"
        )

        return {
            "total_notifications": total_notifications,
            "success_rate": success_rate,
            "emergency_broadcasts": emergency_broadcasts,
            "escalations": escalations,
        }

    async def shutdown(self):
        """Shutdown notification system"""
        logger.info("Shutting down notification system")
        # Close any open connections, cleanup resources
