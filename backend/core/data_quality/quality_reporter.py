"""
Automated Quality Reporting and Alerting System

This module implements automated quality reporting and alerting for geological
data quality issues, including real-time monitoring, alert generation, and
comprehensive quality dashboards.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from .validation_framework import (
    GeologicalObservation, ValidationResult, QualityMetrics, QualityFlag
)
from .cross_source_validator import CrossSourceComparison
from .completeness_analyzer import CompletenessReport, TemporalGap, SpatialGap

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of quality alerts"""
    DATA_QUALITY_DEGRADATION = "data_quality_degradation"
    TEMPORAL_GAP = "temporal_gap"
    SPATIAL_GAP = "spatial_gap"
    SOURCE_FAILURE = "source_failure"
    ANOMALY_DETECTED = "anomaly_detected"
    CONSISTENCY_VIOLATION = "consistency_violation"
    COMPLETENESS_THRESHOLD = "completeness_threshold"


@dataclass
class QualityAlert:
    """Quality alert notification"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    timestamp: datetime
    title: str
    description: str
    affected_systems: List[str]
    data_type: Optional[str]
    location: Optional[Tuple[float, float, float]]
    metrics: Dict[str, Any]
    recommendations: List[str]
    auto_resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    report_id: str
    generation_timestamp: datetime
    report_period: Tuple[datetime, datetime]
    report_type: str  # 'daily', 'weekly', 'monthly', 'on_demand'
    
    # Summary metrics
    overall_quality_score: float
    total_observations: int
    quality_distribution: Dict[str, int]
    
    # Detailed analysis
    validation_results: List[ValidationResult]
    cross_source_comparisons: List[CrossSourceComparison]
    completeness_analysis: Optional[CompletenessReport]
    
    # Alerts and issues
    active_alerts: List[QualityAlert]
    resolved_alerts: List[QualityAlert]
    trending_issues: List[Dict[str, Any]]
    
    # Visualizations
    charts: Dict[str, str]  # Chart name -> file path
    
    # Recommendations
    priority_actions: List[str]
    improvement_suggestions: List[str]


class QualityReporter:
    """Automated quality reporting and alerting system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.active_alerts = {}
        self.alert_history = []
        self.subscribers = {}
        self.report_templates = self._load_report_templates()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for quality reporter"""
        return {
            'alerting': {
                'quality_thresholds': {
                    'critical': 0.5,
                    'warning': 0.7,
                    'info': 0.9
                },
                'gap_thresholds': {
                    'critical_hours': 24,
                    'warning_hours': 6,
                    'info_hours': 1
                },
                'anomaly_thresholds': {
                    'critical': 0.8,
                    'warning': 0.5,
                    'info': 0.2
                }
            },
            'reporting': {
                'daily_report_time': '08:00',
                'weekly_report_day': 'monday',
                'monthly_report_day': 1,
                'retention_days': 90
            },
            'notifications': {
                'email_enabled': True,
                'sms_enabled': False,
                'webhook_enabled': True,
                'slack_enabled': False
            }
        }
    
    def _load_report_templates(self) -> Dict[str, Template]:
        """Load report templates"""
        templates = {}
        
        # HTML email template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Geological Data Quality Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #2c3e50; color: white; padding: 20px; }
                .summary { background-color: #ecf0f1; padding: 15px; margin: 10px 0; }
                .alert { padding: 10px; margin: 5px 0; border-left: 4px solid; }
                .critical { border-color: #e74c3c; background-color: #fadbd8; }
                .warning { border-color: #f39c12; background-color: #fdeaa7; }
                .info { border-color: #3498db; background-color: #d6eaf8; }
                .metrics { display: flex; justify-content: space-around; }
                .metric { text-align: center; padding: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Geological Data Quality Report</h1>
                <p>Generated: {{ generation_time }}</p>
                <p>Period: {{ start_time }} to {{ end_time }}</p>
            </div>
            
            <div class="summary">
                <h2>Quality Summary</h2>
                <div class="metrics">
                    <div class="metric">
                        <h3>{{ overall_quality }}%</h3>
                        <p>Overall Quality</p>
                    </div>
                    <div class="metric">
                        <h3>{{ total_observations }}</h3>
                        <p>Total Observations</p>
                    </div>
                    <div class="metric">
                        <h3>{{ active_alerts }}</h3>
                        <p>Active Alerts</p>
                    </div>
                </div>
            </div>
            
            {% if alerts %}
            <div class="alerts">
                <h2>Active Alerts</h2>
                {% for alert in alerts %}
                <div class="alert {{ alert.severity }}">
                    <h4>{{ alert.title }}</h4>
                    <p>{{ alert.description }}</p>
                    <p><strong>Severity:</strong> {{ alert.severity }}</p>
                    <p><strong>Time:</strong> {{ alert.timestamp }}</p>
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            <div class="recommendations">
                <h2>Priority Actions</h2>
                <ul>
                {% for action in priority_actions %}
                    <li>{{ action }}</li>
                {% endfor %}
                </ul>
            </div>
        </body>
        </html>
        """
        
        templates['html_email'] = Template(html_template)
        
        # Text template for simple notifications
        text_template = """
        Geological Data Quality Alert
        
        Alert: {{ title }}
        Severity: {{ severity }}
        Time: {{ timestamp }}
        
        Description: {{ description }}
        
        Affected Systems: {{ affected_systems }}
        
        Recommendations:
        {% for rec in recommendations %}
        - {{ rec }}
        {% endfor %}
        """
        
        templates['text_alert'] = Template(text_template)
        
        return templates
    
    async def monitor_quality_continuously(
        self,
        validation_callback: Callable,
        check_interval: timedelta = timedelta(minutes=5)
    ):
        """Continuously monitor data quality and generate alerts"""
        logger.info("Starting continuous quality monitoring")
        
        while True:
            try:
                # Get latest validation results
                validation_results = await validation_callback()
                
                # Check for quality issues
                await self._check_quality_thresholds(validation_results)
                
                # Check for temporal gaps
                await self._check_temporal_gaps(validation_results)
                
                # Check for anomalies
                await self._check_anomaly_patterns(validation_results)
                
                # Process any new alerts
                await self._process_new_alerts()
                
                # Clean up resolved alerts
                await self._cleanup_resolved_alerts()
                
            except Exception as e:
                logger.error(f"Error in quality monitoring: {e}")
            
            await asyncio.sleep(check_interval.total_seconds())
    
    async def _check_quality_thresholds(self, validation_results: List[ValidationResult]):
        """Check if quality metrics fall below thresholds"""
        thresholds = self.config['alerting']['quality_thresholds']
        
        for result in validation_results:
            quality_score = result.quality_metrics.overall_quality
            
            # Determine severity based on thresholds
            if quality_score < thresholds['critical']:
                severity = AlertSeverity.CRITICAL
            elif quality_score < thresholds['warning']:
                severity = AlertSeverity.WARNING
            elif quality_score < thresholds['info']:
                severity = AlertSeverity.INFO
            else:
                continue  # Quality is acceptable
            
            # Create alert
            alert_id = f"quality_threshold_{result.details.get('observation_id', 'unknown')}_{datetime.utcnow().isoformat()}"
            
            if alert_id not in self.active_alerts:
                alert = QualityAlert(
                    alert_id=alert_id,
                    alert_type=AlertType.DATA_QUALITY_DEGRADATION,
                    severity=severity,
                    timestamp=datetime.utcnow(),
                    title=f"Data Quality Below Threshold",
                    description=f"Quality score {quality_score:.2f} is below {severity.value} threshold",
                    affected_systems=[result.details.get('source_system', 'unknown')],
                    data_type=result.details.get('data_type'),
                    location=None,
                    metrics={
                        'quality_score': quality_score,
                        'threshold': thresholds[severity.value],
                        'completeness': result.quality_metrics.completeness_score,
                        'accuracy': result.quality_metrics.accuracy_score,
                        'consistency': result.quality_metrics.consistency_score
                    },
                    recommendations=result.recommendations
                )
                
                self.active_alerts[alert_id] = alert
                await self._send_alert_notification(alert)
    
    async def _check_temporal_gaps(self, validation_results: List[ValidationResult]):
        """Check for temporal gaps in data"""
        # This would typically analyze timestamps in validation results
        # For now, we'll implement a simplified version
        
        # Group results by source and data type
        source_timestamps = {}
        for result in validation_results:
            source_key = f"{result.details.get('source_system')}_{result.details.get('data_type')}"
            if source_key not in source_timestamps:
                source_timestamps[source_key] = []
            source_timestamps[source_key].append(result.validation_timestamp)
        
        # Check for gaps
        gap_thresholds = self.config['alerting']['gap_thresholds']
        
        for source_key, timestamps in source_timestamps.items():
            if len(timestamps) < 2:
                continue
            
            timestamps.sort()
            for i in range(1, len(timestamps)):
                gap_duration = timestamps[i] - timestamps[i-1]
                gap_hours = gap_duration.total_seconds() / 3600
                
                # Determine severity
                if gap_hours >= gap_thresholds['critical_hours']:
                    severity = AlertSeverity.CRITICAL
                elif gap_hours >= gap_thresholds['warning_hours']:
                    severity = AlertSeverity.WARNING
                elif gap_hours >= gap_thresholds['info_hours']:
                    severity = AlertSeverity.INFO
                else:
                    continue
                
                # Create alert
                alert_id = f"temporal_gap_{source_key}_{timestamps[i-1].isoformat()}"
                
                if alert_id not in self.active_alerts:
                    alert = QualityAlert(
                        alert_id=alert_id,
                        alert_type=AlertType.TEMPORAL_GAP,
                        severity=severity,
                        timestamp=datetime.utcnow(),
                        title=f"Temporal Gap Detected",
                        description=f"Gap of {gap_hours:.1f} hours detected in {source_key}",
                        affected_systems=[source_key.split('_')[0]],
                        data_type=source_key.split('_')[1] if '_' in source_key else None,
                        location=None,
                        metrics={
                            'gap_duration_hours': gap_hours,
                            'gap_start': timestamps[i-1].isoformat(),
                            'gap_end': timestamps[i].isoformat()
                        },
                        recommendations=[
                            f"Investigate data source {source_key.split('_')[0]}",
                            "Check network connectivity and system status",
                            "Consider backup data sources if available"
                        ]
                    )
                    
                    self.active_alerts[alert_id] = alert
                    await self._send_alert_notification(alert)
    
    async def _check_anomaly_patterns(self, validation_results: List[ValidationResult]):
        """Check for anomaly patterns in validation results"""
        anomaly_thresholds = self.config['alerting']['anomaly_thresholds']
        
        for result in validation_results:
            anomaly_score = result.quality_metrics.anomaly_score
            
            # Determine severity
            if anomaly_score >= anomaly_thresholds['critical']:
                severity = AlertSeverity.CRITICAL
            elif anomaly_score >= anomaly_thresholds['warning']:
                severity = AlertSeverity.WARNING
            elif anomaly_score >= anomaly_thresholds['info']:
                severity = AlertSeverity.INFO
            else:
                continue
            
            # Create alert
            alert_id = f"anomaly_{result.details.get('observation_id', 'unknown')}_{datetime.utcnow().isoformat()}"
            
            if alert_id not in self.active_alerts:
                alert = QualityAlert(
                    alert_id=alert_id,
                    alert_type=AlertType.ANOMALY_DETECTED,
                    severity=severity,
                    timestamp=datetime.utcnow(),
                    title=f"Data Anomaly Detected",
                    description=f"Anomaly score {anomaly_score:.2f} indicates potential data issue",
                    affected_systems=[result.details.get('source_system', 'unknown')],
                    data_type=result.details.get('data_type'),
                    location=None,
                    metrics={
                        'anomaly_score': anomaly_score,
                        'threshold': anomaly_thresholds[severity.value]
                    },
                    recommendations=[
                        "Investigate data source for potential issues",
                        "Cross-validate with other data sources",
                        "Consider excluding anomalous data from analysis"
                    ]
                )
                
                self.active_alerts[alert_id] = alert
                await self._send_alert_notification(alert)
    
    async def _process_new_alerts(self):
        """Process any new alerts that need attention"""
        # Group alerts by severity for processing
        critical_alerts = [a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]
        
        if critical_alerts:
            # Send immediate notifications for critical alerts
            for alert in critical_alerts:
                await self._escalate_critical_alert(alert)
    
    async def _cleanup_resolved_alerts(self):
        """Clean up alerts that have been resolved"""
        current_time = datetime.utcnow()
        resolved_alerts = []
        
        for alert_id, alert in list(self.active_alerts.items()):
            # Check if alert conditions have been resolved
            if await self._is_alert_resolved(alert):
                alert.auto_resolved = True
                alert.resolution_timestamp = current_time
                resolved_alerts.append(alert)
                del self.active_alerts[alert_id]
        
        # Add to history
        self.alert_history.extend(resolved_alerts)
        
        # Clean up old history
        cutoff_time = current_time - timedelta(days=self.config['reporting']['retention_days'])
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff_time]
    
    async def _is_alert_resolved(self, alert: QualityAlert) -> bool:
        """Check if an alert condition has been resolved"""
        # This would implement logic to check if the underlying issue has been resolved
        # For now, we'll use a simple time-based resolution for demonstration
        
        resolution_timeout = {
            AlertSeverity.INFO: timedelta(hours=1),
            AlertSeverity.WARNING: timedelta(hours=6),
            AlertSeverity.ERROR: timedelta(hours=24),
            AlertSeverity.CRITICAL: timedelta(days=7)
        }
        
        timeout = resolution_timeout.get(alert.severity, timedelta(hours=24))
        return datetime.utcnow() - alert.timestamp > timeout
    
    async def _send_alert_notification(self, alert: QualityAlert):
        """Send alert notification to subscribers"""
        logger.info(f"Sending alert notification: {alert.title}")
        
        # Email notification
        if self.config['notifications']['email_enabled']:
            await self._send_email_alert(alert)
        
        # Webhook notification
        if self.config['notifications']['webhook_enabled']:
            await self._send_webhook_alert(alert)
        
        # Add to alert history
        self.alert_history.append(alert)
    
    async def _send_email_alert(self, alert: QualityAlert):
        """Send email alert notification"""
        try:
            template = self.report_templates['text_alert']
            
            message_body = template.render(
                title=alert.title,
                severity=alert.severity.value,
                timestamp=alert.timestamp.isoformat(),
                description=alert.description,
                affected_systems=', '.join(alert.affected_systems),
                recommendations=alert.recommendations
            )
            
            # This would use actual email configuration
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_webhook_alert(self, alert: QualityAlert):
        """Send webhook alert notification"""
        try:
            # This would send HTTP POST to configured webhook URLs
            webhook_payload = {
                'alert_id': alert.alert_id,
                'type': alert.alert_type.value,
                'severity': alert.severity.value,
                'timestamp': alert.timestamp.isoformat(),
                'title': alert.title,
                'description': alert.description,
                'metrics': alert.metrics
            }
            
            logger.info(f"Webhook alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    async def _escalate_critical_alert(self, alert: QualityAlert):
        """Escalate critical alerts for immediate attention"""
        logger.warning(f"Escalating critical alert: {alert.title}")
        
        # This would implement escalation procedures such as:
        # - SMS notifications to on-call personnel
        # - Phone calls for critical system failures
        # - Integration with incident management systems
        
        # For now, just log the escalation
        logger.critical(f"CRITICAL ALERT ESCALATED: {alert.description}")
    
    async def generate_quality_report(
        self,
        report_type: str = 'daily',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        validation_results: Optional[List[ValidationResult]] = None,
        completeness_analysis: Optional[CompletenessReport] = None
    ) -> QualityReport:
        """Generate comprehensive quality report"""
        
        if end_time is None:
            end_time = datetime.utcnow()
        
        if start_time is None:
            if report_type == 'daily':
                start_time = end_time - timedelta(days=1)
            elif report_type == 'weekly':
                start_time = end_time - timedelta(weeks=1)
            elif report_type == 'monthly':
                start_time = end_time - timedelta(days=30)
            else:
                start_time = end_time - timedelta(days=1)
        
        # Calculate summary metrics
        if validation_results:
            overall_quality = np.mean([r.quality_metrics.overall_quality for r in validation_results])
            total_observations = len(validation_results)
            
            # Quality distribution
            quality_distribution = {
                'excellent': sum(1 for r in validation_results if r.quality_metrics.overall_quality >= 0.9),
                'good': sum(1 for r in validation_results if 0.7 <= r.quality_metrics.overall_quality < 0.9),
                'fair': sum(1 for r in validation_results if 0.5 <= r.quality_metrics.overall_quality < 0.7),
                'poor': sum(1 for r in validation_results if r.quality_metrics.overall_quality < 0.5)
            }
        else:
            overall_quality = 0.0
            total_observations = 0
            quality_distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        # Get alerts in time range
        period_alerts = [
            alert for alert in self.alert_history + list(self.active_alerts.values())
            if start_time <= alert.timestamp <= end_time
        ]
        
        active_alerts = [alert for alert in period_alerts if not alert.auto_resolved]
        resolved_alerts = [alert for alert in period_alerts if alert.auto_resolved]
        
        # Generate trending analysis
        trending_issues = self._analyze_trending_issues(period_alerts)
        
        # Generate visualizations
        charts = await self._generate_quality_charts(validation_results, period_alerts)
        
        # Generate recommendations
        priority_actions = self._generate_priority_actions(active_alerts, overall_quality)
        improvement_suggestions = self._generate_improvement_suggestions(
            validation_results, completeness_analysis
        )
        
        report = QualityReport(
            report_id=f"quality_report_{report_type}_{end_time.strftime('%Y%m%d_%H%M%S')}",
            generation_timestamp=datetime.utcnow(),
            report_period=(start_time, end_time),
            report_type=report_type,
            overall_quality_score=overall_quality,
            total_observations=total_observations,
            quality_distribution=quality_distribution,
            validation_results=validation_results or [],
            cross_source_comparisons=[],  # Would be populated with actual data
            completeness_analysis=completeness_analysis,
            active_alerts=active_alerts,
            resolved_alerts=resolved_alerts,
            trending_issues=trending_issues,
            charts=charts,
            priority_actions=priority_actions,
            improvement_suggestions=improvement_suggestions
        )
        
        return report
    
    def _analyze_trending_issues(self, alerts: List[QualityAlert]) -> List[Dict[str, Any]]:
        """Analyze trending issues from alert history"""
        # Group alerts by type and analyze trends
        alert_counts = {}
        for alert in alerts:
            key = f"{alert.alert_type.value}_{alert.data_type or 'unknown'}"
            if key not in alert_counts:
                alert_counts[key] = []
            alert_counts[key].append(alert.timestamp)
        
        trending = []
        for issue_type, timestamps in alert_counts.items():
            if len(timestamps) >= 3:  # Need at least 3 occurrences to identify trend
                # Calculate frequency
                time_span = max(timestamps) - min(timestamps)
                frequency = len(timestamps) / max(1, time_span.total_seconds() / 3600)  # per hour
                
                trending.append({
                    'issue_type': issue_type,
                    'occurrences': len(timestamps),
                    'frequency_per_hour': frequency,
                    'first_occurrence': min(timestamps),
                    'last_occurrence': max(timestamps),
                    'trend': 'increasing' if frequency > 0.1 else 'stable'
                })
        
        return sorted(trending, key=lambda x: x['frequency_per_hour'], reverse=True)
    
    async def _generate_quality_charts(
        self, 
        validation_results: Optional[List[ValidationResult]], 
        alerts: List[QualityAlert]
    ) -> Dict[str, str]:
        """Generate quality visualization charts"""
        charts = {}
        
        if validation_results:
            # Quality score distribution
            quality_scores = [r.quality_metrics.overall_quality for r in validation_results]
            
            plt.figure(figsize=(10, 6))
            plt.hist(quality_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Quality Score')
            plt.ylabel('Frequency')
            plt.title('Data Quality Score Distribution')
            plt.grid(True, alpha=0.3)
            
            chart_path = f"/tmp/quality_distribution_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['quality_distribution'] = chart_path
        
        if alerts:
            # Alert severity distribution
            severity_counts = {}
            for alert in alerts:
                severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            
            plt.figure(figsize=(8, 6))
            plt.pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%')
            plt.title('Alert Severity Distribution')
            
            chart_path = f"/tmp/alert_severity_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['alert_severity'] = chart_path
        
        return charts
    
    def _generate_priority_actions(
        self, 
        active_alerts: List[QualityAlert], 
        overall_quality: float
    ) -> List[str]:
        """Generate priority actions based on current state"""
        actions = []
        
        # Critical alerts
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            actions.append(f"Address {len(critical_alerts)} critical alerts immediately")
        
        # Overall quality
        if overall_quality < 0.5:
            actions.append("Overall quality is critically low - investigate system-wide issues")
        elif overall_quality < 0.7:
            actions.append("Overall quality is below acceptable threshold - review data sources")
        
        # Alert patterns
        alert_types = {}
        for alert in active_alerts:
            alert_types[alert.alert_type.value] = alert_types.get(alert.alert_type.value, 0) + 1
        
        if alert_types.get('temporal_gap', 0) > 5:
            actions.append("Multiple temporal gaps detected - check data ingestion systems")
        
        if alert_types.get('anomaly_detected', 0) > 10:
            actions.append("High number of anomalies - review data validation parameters")
        
        return actions
    
    def _generate_improvement_suggestions(
        self,
        validation_results: Optional[List[ValidationResult]],
        completeness_analysis: Optional[CompletenessReport]
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if validation_results:
            # Analyze common quality issues
            low_accuracy_count = sum(1 for r in validation_results if r.quality_metrics.accuracy_score < 0.7)
            if low_accuracy_count > len(validation_results) * 0.2:
                suggestions.append("Consider calibrating instruments - accuracy issues detected in >20% of data")
            
            low_consistency_count = sum(1 for r in validation_results if r.quality_metrics.consistency_score < 0.7)
            if low_consistency_count > len(validation_results) * 0.15:
                suggestions.append("Review cross-source validation - consistency issues in >15% of data")
        
        if completeness_analysis:
            if completeness_analysis.overall_completeness < 0.8:
                suggestions.append("Improve data collection coverage - completeness below 80%")
            
            if completeness_analysis.temporal_gaps:
                critical_gaps = [g for g in completeness_analysis.temporal_gaps if g.severity == 'critical']
                if critical_gaps:
                    suggestions.append(f"Address {len(critical_gaps)} critical temporal gaps")
        
        return suggestions