"""
Offline alert generation using pre-computed thresholds
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    data_type: str
    condition: str  # threshold condition
    threshold_value: float
    severity: AlertSeverity
    cooldown_minutes: int = 15
    enabled: bool = True
    metadata: Dict[str, Any] = None


@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    data_source: str
    trigger_value: float
    threshold_value: float
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class OfflineAlertGenerator:
    """Generates alerts using pre-computed thresholds for offline operation"""
    
    def __init__(self, config_path: str = "./alert_config.json"):
        self.config_path = config_path
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Alert callbacks for different actions
        self.alert_callbacks: Dict[str, List[Callable]] = {
            "new_alert": [],
            "alert_acknowledged": [],
            "alert_resolved": []
        }
        
        # Statistics
        self.stats = {
            "total_alerts": 0,
            "alerts_by_severity": {severity.value: 0 for severity in AlertSeverity},
            "alerts_by_type": {},
            "last_alert_time": None
        }
        
        self._load_alert_rules()
    
    def _load_alert_rules(self):
        """Load alert rules from configuration"""
        try:
            # Default alert rules for geological monitoring
            default_rules = {
                # Seismic alerts
                "seismic_magnitude_high": AlertRule(
                    rule_id="seismic_magnitude_high",
                    name="High Magnitude Seismic Event",
                    description="Seismic event with magnitude >= 4.5",
                    data_type="seismic",
                    condition="magnitude >= threshold",
                    threshold_value=4.5,
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=5
                ),
                "seismic_magnitude_medium": AlertRule(
                    rule_id="seismic_magnitude_medium",
                    name="Moderate Seismic Event",
                    description="Seismic event with magnitude >= 3.0",
                    data_type="seismic",
                    condition="magnitude >= threshold",
                    threshold_value=3.0,
                    severity=AlertSeverity.MEDIUM,
                    cooldown_minutes=10
                ),
                "seismic_swarm": AlertRule(
                    rule_id="seismic_swarm",
                    name="Seismic Swarm Detected",
                    description="Multiple seismic events in short time period",
                    data_type="seismic",
                    condition="event_count >= threshold",
                    threshold_value=10,
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=30
                ),
                
                # Deformation alerts
                "deformation_high": AlertRule(
                    rule_id="deformation_high",
                    name="Significant Ground Deformation",
                    description="Ground displacement >= 5cm",
                    data_type="deformation",
                    condition="displacement >= threshold",
                    threshold_value=0.05,  # 5cm in meters
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=15
                ),
                "deformation_medium": AlertRule(
                    rule_id="deformation_medium",
                    name="Ground Deformation Detected",
                    description="Ground displacement >= 1cm",
                    data_type="deformation",
                    condition="displacement >= threshold",
                    threshold_value=0.01,  # 1cm in meters
                    severity=AlertSeverity.MEDIUM,
                    cooldown_minutes=30
                ),
                "deformation_velocity": AlertRule(
                    rule_id="deformation_velocity",
                    name="Rapid Ground Movement",
                    description="Ground movement velocity >= 1cm/day",
                    data_type="deformation",
                    condition="velocity >= threshold",
                    threshold_value=0.01,  # 1cm/day in m/day
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=60
                ),
                
                # Thermal alerts
                "thermal_high": AlertRule(
                    rule_id="thermal_high",
                    name="High Temperature Anomaly",
                    description="Temperature anomaly >= 100°C above baseline",
                    data_type="thermal",
                    condition="temperature_anomaly >= threshold",
                    threshold_value=100.0,
                    severity=AlertSeverity.CRITICAL,
                    cooldown_minutes=10
                ),
                "thermal_medium": AlertRule(
                    rule_id="thermal_medium",
                    name="Temperature Anomaly",
                    description="Temperature anomaly >= 50°C above baseline",
                    data_type="thermal",
                    condition="temperature_anomaly >= threshold",
                    threshold_value=50.0,
                    severity=AlertSeverity.MEDIUM,
                    cooldown_minutes=20
                ),
                
                # Gas composition alerts
                "gas_co2_high": AlertRule(
                    rule_id="gas_co2_high",
                    name="High CO2 Concentration",
                    description="CO2 concentration >= 5000 ppm",
                    data_type="gas",
                    condition="co2 >= threshold",
                    threshold_value=5000.0,
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=15
                ),
                "gas_so2_high": AlertRule(
                    rule_id="gas_so2_high",
                    name="High SO2 Concentration",
                    description="SO2 concentration >= 500 ppm",
                    data_type="gas",
                    condition="so2 >= threshold",
                    threshold_value=500.0,
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=15
                ),
                "gas_h2s_critical": AlertRule(
                    rule_id="gas_h2s_critical",
                    name="Critical H2S Concentration",
                    description="H2S concentration >= 200 ppm (dangerous levels)",
                    data_type="gas",
                    condition="h2s >= threshold",
                    threshold_value=200.0,
                    severity=AlertSeverity.CRITICAL,
                    cooldown_minutes=5
                ),
                
                # System health alerts
                "system_cpu_high": AlertRule(
                    rule_id="system_cpu_high",
                    name="High CPU Usage",
                    description="CPU usage >= 90%",
                    data_type="system",
                    condition="cpu_usage >= threshold",
                    threshold_value=90.0,
                    severity=AlertSeverity.MEDIUM,
                    cooldown_minutes=10
                ),
                "system_memory_high": AlertRule(
                    rule_id="system_memory_high",
                    name="High Memory Usage",
                    description="Memory usage >= 85%",
                    data_type="system",
                    condition="memory_usage >= threshold",
                    threshold_value=85.0,
                    severity=AlertSeverity.MEDIUM,
                    cooldown_minutes=10
                ),
                "system_storage_critical": AlertRule(
                    rule_id="system_storage_critical",
                    name="Critical Storage Usage",
                    description="Storage usage >= 95%",
                    data_type="system",
                    condition="storage_usage >= threshold",
                    threshold_value=95.0,
                    severity=AlertSeverity.CRITICAL,
                    cooldown_minutes=5
                )
            }
            
            self.alert_rules = default_rules
            
            # Try to load custom rules from file
            try:
                with open(self.config_path, 'r') as f:
                    custom_config = json.load(f)
                    
                # Override default rules with custom ones
                for rule_data in custom_config.get("alert_rules", []):
                    rule = AlertRule(**rule_data)
                    self.alert_rules[rule.rule_id] = rule
                    
                logger.info(f"Loaded {len(self.alert_rules)} alert rules")
                
            except FileNotFoundError:
                # Save default rules to file
                self._save_alert_rules()
                logger.info("Created default alert rules configuration")
                
        except Exception as e:
            logger.error(f"Error loading alert rules: {e}")
    
    def _save_alert_rules(self):
        """Save alert rules to configuration file"""
        try:
            config = {
                "alert_rules": [
                    {
                        "rule_id": rule.rule_id,
                        "name": rule.name,
                        "description": rule.description,
                        "data_type": rule.data_type,
                        "condition": rule.condition,
                        "threshold_value": rule.threshold_value,
                        "severity": rule.severity.value,
                        "cooldown_minutes": rule.cooldown_minutes,
                        "enabled": rule.enabled,
                        "metadata": rule.metadata or {}
                    }
                    for rule in self.alert_rules.values()
                ]
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving alert rules: {e}")
    
    async def evaluate_data(self, data_type: str, data: Dict[str, Any], location: Optional[Dict[str, float]] = None) -> List[Alert]:
        """Evaluate data against alert rules and generate alerts"""
        generated_alerts = []
        
        try:
            # Get relevant rules for this data type
            relevant_rules = [rule for rule in self.alert_rules.values() 
                           if rule.data_type == data_type and rule.enabled]
            
            for rule in relevant_rules:
                # Check cooldown
                if self._is_in_cooldown(rule.rule_id):
                    continue
                
                # Evaluate rule condition
                triggered = await self._evaluate_rule_condition(rule, data)
                
                if triggered:
                    # Get trigger value for context
                    trigger_value = self._extract_trigger_value(rule, data)
                    
                    # Generate alert
                    alert = await self._create_alert(rule, data, trigger_value, location)
                    generated_alerts.append(alert)
                    
                    # Add to active alerts
                    self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)
                    
                    # Update cooldown
                    self.cooldown_tracker[rule.rule_id] = datetime.utcnow()
                    
                    # Update statistics
                    self._update_statistics(alert)
                    
                    # Trigger callbacks
                    await self._trigger_callbacks("new_alert", alert)
                    
                    logger.warning(f"ALERT GENERATED: {alert.title} - {alert.message}")
            
        except Exception as e:
            logger.error(f"Error evaluating data for alerts: {e}")
        
        return generated_alerts
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.cooldown_tracker:
            return False
        
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        last_trigger = self.cooldown_tracker[rule_id]
        cooldown_end = last_trigger + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.utcnow() < cooldown_end
    
    async def _evaluate_rule_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate if rule condition is met"""
        try:
            if rule.data_type == "seismic":
                return await self._evaluate_seismic_condition(rule, data)
            elif rule.data_type == "deformation":
                return await self._evaluate_deformation_condition(rule, data)
            elif rule.data_type == "thermal":
                return await self._evaluate_thermal_condition(rule, data)
            elif rule.data_type == "gas":
                return await self._evaluate_gas_condition(rule, data)
            elif rule.data_type == "system":
                return await self._evaluate_system_condition(rule, data)
            else:
                logger.warning(f"Unknown data type for rule evaluation: {rule.data_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating rule condition {rule.rule_id}: {e}")
            return False
    
    async def _evaluate_seismic_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate seismic data conditions"""
        if "magnitude >= threshold" in rule.condition:
            magnitude = data.get("magnitude", 0.0)
            return magnitude >= rule.threshold_value
        elif "event_count >= threshold" in rule.condition:
            event_count = data.get("event_count", 0)
            return event_count >= rule.threshold_value
        
        return False
    
    async def _evaluate_deformation_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate deformation data conditions"""
        if "displacement >= threshold" in rule.condition:
            displacement = data.get("displacement_magnitude", 0.0)
            return displacement >= rule.threshold_value
        elif "velocity >= threshold" in rule.condition:
            velocity = data.get("velocity_magnitude", 0.0)
            return velocity >= rule.threshold_value
        
        return False
    
    async def _evaluate_thermal_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate thermal data conditions"""
        if "temperature_anomaly >= threshold" in rule.condition:
            temp_anomaly = data.get("temperature_anomaly", 0.0)
            return temp_anomaly >= rule.threshold_value
        elif "temperature >= threshold" in rule.condition:
            temperature = data.get("temperature", 0.0)
            return temperature >= rule.threshold_value
        
        return False
    
    async def _evaluate_gas_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate gas composition conditions"""
        composition = data.get("composition", {})
        
        if "co2 >= threshold" in rule.condition:
            co2 = composition.get("CO2", 0.0)
            return co2 >= rule.threshold_value
        elif "so2 >= threshold" in rule.condition:
            so2 = composition.get("SO2", 0.0)
            return so2 >= rule.threshold_value
        elif "h2s >= threshold" in rule.condition:
            h2s = composition.get("H2S", 0.0)
            return h2s >= rule.threshold_value
        
        return False
    
    async def _evaluate_system_condition(self, rule: AlertRule, data: Dict[str, Any]) -> bool:
        """Evaluate system health conditions"""
        if "cpu_usage >= threshold" in rule.condition:
            cpu_usage = data.get("cpu_usage_percent", 0.0)
            return cpu_usage >= rule.threshold_value
        elif "memory_usage >= threshold" in rule.condition:
            memory_usage = data.get("memory_usage_percent", 0.0)
            return memory_usage >= rule.threshold_value
        elif "storage_usage >= threshold" in rule.condition:
            storage_usage = data.get("storage_usage_percent", 0.0)
            return storage_usage >= rule.threshold_value
        
        return False
    
    def _extract_trigger_value(self, rule: AlertRule, data: Dict[str, Any]) -> float:
        """Extract the value that triggered the alert"""
        if rule.data_type == "seismic":
            if "magnitude" in rule.condition:
                return data.get("magnitude", 0.0)
            elif "event_count" in rule.condition:
                return data.get("event_count", 0)
        elif rule.data_type == "deformation":
            if "displacement" in rule.condition:
                return data.get("displacement_magnitude", 0.0)
            elif "velocity" in rule.condition:
                return data.get("velocity_magnitude", 0.0)
        elif rule.data_type == "thermal":
            if "temperature_anomaly" in rule.condition:
                return data.get("temperature_anomaly", 0.0)
            elif "temperature" in rule.condition:
                return data.get("temperature", 0.0)
        elif rule.data_type == "gas":
            composition = data.get("composition", {})
            if "co2" in rule.condition:
                return composition.get("CO2", 0.0)
            elif "so2" in rule.condition:
                return composition.get("SO2", 0.0)
            elif "h2s" in rule.condition:
                return composition.get("H2S", 0.0)
        elif rule.data_type == "system":
            if "cpu_usage" in rule.condition:
                return data.get("cpu_usage_percent", 0.0)
            elif "memory_usage" in rule.condition:
                return data.get("memory_usage_percent", 0.0)
            elif "storage_usage" in rule.condition:
                return data.get("storage_usage_percent", 0.0)
        
        return 0.0
    
    async def _create_alert(self, rule: AlertRule, data: Dict[str, Any], trigger_value: float, location: Optional[Dict[str, float]]) -> Alert:
        """Create alert instance"""
        alert_id = f"{rule.rule_id}_{datetime.utcnow().timestamp()}"
        
        # Create contextual message
        message = f"{rule.description}. Trigger value: {trigger_value:.2f}, Threshold: {rule.threshold_value:.2f}"
        
        # Add location context if available
        if location:
            lat, lon = location.get("latitude", 0), location.get("longitude", 0)
            message += f" at location ({lat:.4f}, {lon:.4f})"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=datetime.utcnow(),
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=rule.name,
            message=message,
            data_source=rule.data_type,
            trigger_value=trigger_value,
            threshold_value=rule.threshold_value,
            location=location,
            metadata={
                "rule_description": rule.description,
                "data_snapshot": data.copy(),
                "cooldown_minutes": rule.cooldown_minutes
            }
        )
        
        return alert
    
    def _update_statistics(self, alert: Alert):
        """Update alert statistics"""
        self.stats["total_alerts"] += 1
        self.stats["alerts_by_severity"][alert.severity.value] += 1
        
        if alert.data_source not in self.stats["alerts_by_type"]:
            self.stats["alerts_by_type"][alert.data_source] = 0
        self.stats["alerts_by_type"][alert.data_source] += 1
        
        self.stats["last_alert_time"] = alert.timestamp
    
    async def _trigger_callbacks(self, event_type: str, alert: Alert):
        """Trigger registered callbacks for alert events"""
        callbacks = self.alert_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for alert events"""
        if event_type not in self.alert_callbacks:
            self.alert_callbacks[event_type] = []
        self.alert_callbacks[event_type].append(callback)
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.metadata["acknowledged_by"] = acknowledged_by
                
                await self._trigger_callbacks("alert_acknowledged", alert)
                logger.info(f"Alert acknowledged: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                alert.metadata["resolved_by"] = resolved_by
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                await self._trigger_callbacks("alert_resolved", alert)
                logger.info(f"Alert resolved: {alert_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, limit: int = 100, since: Optional[datetime] = None) -> List[Alert]:
        """Get alert history"""
        alerts = self.alert_history.copy()
        
        if since:
            alerts = [alert for alert in alerts if alert.timestamp >= since]
        
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        return {
            **self.stats,
            "active_alerts_count": len(self.active_alerts),
            "rules_count": len(self.alert_rules),
            "enabled_rules_count": len([r for r in self.alert_rules.values() if r.enabled])
        }
    
    def update_rule_threshold(self, rule_id: str, new_threshold: float) -> bool:
        """Update threshold for an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].threshold_value = new_threshold
                self._save_alert_rules()
                logger.info(f"Updated threshold for rule {rule_id}: {new_threshold}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating rule threshold: {e}")
            return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = True
                self._save_alert_rules()
                logger.info(f"Enabled alert rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error enabling rule: {e}")
            return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule"""
        try:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = False
                self._save_alert_rules()
                logger.info(f"Disabled alert rule: {rule_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error disabling rule: {e}")
            return False
    
    async def cleanup_old_alerts(self, retention_days: int = 30):
        """Clean up old resolved alerts"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            
            # Remove old resolved alerts from history
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp >= cutoff_date or alert.status == AlertStatus.ACTIVE
            ]
            
            logger.info(f"Cleaned up alerts older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Error cleaning up old alerts: {e}")