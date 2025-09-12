"""
Comprehensive Compliance Dashboard for Prometheus AI Platform

This module provides a unified compliance dashboard with real-time metrics,
automated reporting, and interactive visualizations for all compliance frameworks.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
import aioredis
import asyncpg
import pandas as pd
import numpy as np
from collections import defaultdict

# Configure structured logging
logger = logging.getLogger(__name__)

class DashboardMetricType(Enum):
    """Types of compliance dashboard metrics"""
    COMPLIANCE_SCORE = "compliance_score"
    VIOLATION_COUNT = "violation_count"
    RISK_LEVEL = "risk_level"
    AUDIT_COVERAGE = "audit_coverage"
    POLICY_EFFECTIVENESS = "policy_effectiveness"
    INCIDENT_RESPONSE_TIME = "incident_response_time"
    DATA_RETENTION_COMPLIANCE = "data_retention_compliance"

class TimeRange(Enum):
    """Time range options for dashboard metrics"""
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert levels for compliance dashboard"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str
    title: str
    description: str
    metric_type: DashboardMetricType
    time_range: TimeRange
    filters: Dict[str, Any]
    visualization_config: Dict[str, Any]
    refresh_interval: int  # seconds
    position: Dict[str, int]  # x, y, width, height
    permissions: List[str]

@dataclass
class ComplianceAlert:
    """Compliance dashboard alert"""
    alert_id: str
    alert_type: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    framework: str
    affected_resources: List[str]
    recommended_actions: List[str]
    acknowledged: bool
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]

@dataclass
class DashboardMetric:
    """Dashboard metric data point"""
    metric_id: str
    metric_type: DashboardMetricType
    timestamp: datetime
    value: float
    target_value: Optional[float]
    threshold_warning: Optional[float]
    threshold_critical: Optional[float]
    framework: str
    context: Dict[str, Any]
    trend: str  # up, down, stable

class ComplianceDashboard:
    """
    Comprehensive compliance dashboard providing real-time visibility into
    compliance posture across all frameworks (GDPR, SOC2, ISO27001) with
    interactive visualizations and automated alerting.
    """
    
    def __init__(self, db_pool: asyncpg.Pool, redis_client: aioredis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.dashboard_widgets = {}
        self.active_alerts = {}
        self.metric_calculators = {}
        self._initialize_metric_calculators()
        
    def _initialize_metric_calculators(self):
        """Initialize metric calculation functions"""
        self.metric_calculators = {
            DashboardMetricType.COMPLIANCE_SCORE: self._calculate_compliance_score,
            DashboardMetricType.VIOLATION_COUNT: self._calculate_violation_count,
            DashboardMetricType.RISK_LEVEL: self._calculate_risk_level,
            DashboardMetricType.AUDIT_COVERAGE: self._calculate_audit_coverage,
            DashboardMetricType.POLICY_EFFECTIVENESS: self._calculate_policy_effectiveness,
            DashboardMetricType.INCIDENT_RESPONSE_TIME: self._calculate_incident_response_time,
            DashboardMetricType.DATA_RETENTION_COMPLIANCE: self._calculate_retention_compliance
        }
    
    async def initialize_dashboard_widgets(self):
        """Initialize default dashboard widgets for compliance monitoring"""
        widgets = [
            # Overall Compliance Score Widget
            DashboardWidget(
                widget_id="overall_compliance_score",
                widget_type="gauge",
                title="Overall Compliance Score",
                description="Aggregate compliance score across all frameworks",
                metric_type=DashboardMetricType.COMPLIANCE_SCORE,
                time_range=TimeRange.LAST_30_DAYS,
                filters={"frameworks": ["GDPR", "SOC2", "ISO27001"]},
                visualization_config={
                    "gauge_type": "radial",
                    "min_value": 0,
                    "max_value": 100,
                    "color_ranges": [
                        {"min": 0, "max": 60, "color": "#ff4444"},
                        {"min": 60, "max": 80, "color": "#ffaa00"},
                        {"min": 80, "max": 100, "color": "#44ff44"}
                    ]
                },
                refresh_interval=300,  # 5 minutes
                position={"x": 0, "y": 0, "width": 4, "height": 3},
                permissions=["compliance_viewer", "compliance_admin"]
            ),
            
            # GDPR Compliance Widget
            DashboardWidget(
                widget_id="gdpr_compliance",
                widget_type="scorecard",
                title="GDPR Compliance",
                description="GDPR compliance metrics and data subject rights",
                metric_type=DashboardMetricType.COMPLIANCE_SCORE,
                time_range=TimeRange.LAST_30_DAYS,
                filters={"frameworks": ["GDPR"]},
                visualization_config={
                    "show_trend": True,
                    "show_target": True,
                    "target_value": 95
                },
                refresh_interval=600,  # 10 minutes
                position={"x": 4, "y": 0, "width": 4, "height": 3},
                permissions=["compliance_viewer", "compliance_admin", "dpo"]
            ),
            
            # SOC2 Controls Widget
            DashboardWidget(
                widget_id="soc2_controls",
                widget_type="heatmap",
                title="SOC2 Control Effectiveness",
                description="SOC2 trust service criteria control effectiveness",
                metric_type=DashboardMetricType.POLICY_EFFECTIVENESS,
                time_range=TimeRange.LAST_30_DAYS,
                filters={"frameworks": ["SOC2"]},
                visualization_config={
                    "categories": ["Security", "Availability", "Processing Integrity", "Confidentiality", "Privacy"],
                    "color_scale": "RdYlGn"
                },
                refresh_interval=900,  # 15 minutes
                position={"x": 8, "y": 0, "width": 4, "height": 3},
                permissions=["compliance_viewer", "compliance_admin", "auditor"]
            ),
            
            # Security Incidents Timeline
            DashboardWidget(
                widget_id="security_incidents_timeline",
                widget_type="timeline",
                title="Security Incidents",
                description="Timeline of security incidents and compliance violations",
                metric_type=DashboardMetricType.VIOLATION_COUNT,
                time_range=TimeRange.LAST_7_DAYS,
                filters={"severity": ["high", "critical"]},
                visualization_config={
                    "group_by": "severity",
                    "show_details": True
                },
                refresh_interval=300,  # 5 minutes
                position={"x": 0, "y": 3, "width": 8, "height": 4},
                permissions=["compliance_viewer", "compliance_admin", "security_team"]
            ),
            
            # Risk Assessment Matrix
            DashboardWidget(
                widget_id="risk_assessment_matrix",
                widget_type="matrix",
                title="Risk Assessment Matrix",
                description="Current risk levels across different categories",
                metric_type=DashboardMetricType.RISK_LEVEL,
                time_range=TimeRange.LAST_30_DAYS,
                filters={},
                visualization_config={
                    "x_axis": "likelihood",
                    "y_axis": "impact",
                    "bubble_size": "risk_score"
                },
                refresh_interval=1800,  # 30 minutes
                position={"x": 8, "y": 3, "width": 4, "height": 4},
                permissions=["compliance_viewer", "compliance_admin", "risk_manager"]
            ),
            
            # Data Retention Compliance
            DashboardWidget(
                widget_id="data_retention_compliance",
                widget_type="bar_chart",
                title="Data Retention Compliance",
                description="Data retention policy compliance by category",
                metric_type=DashboardMetricType.DATA_RETENTION_COMPLIANCE,
                time_range=TimeRange.LAST_30_DAYS,
                filters={},
                visualization_config={
                    "group_by": "data_category",
                    "show_targets": True
                },
                refresh_interval=3600,  # 1 hour
                position={"x": 0, "y": 7, "width": 6, "height": 3},
                permissions=["compliance_viewer", "compliance_admin", "data_manager"]
            ),
            
            # Audit Coverage Widget
            DashboardWidget(
                widget_id="audit_coverage",
                widget_type="progress_bar",
                title="Audit Coverage",
                description="Audit log coverage and integrity status",
                metric_type=DashboardMetricType.AUDIT_COVERAGE,
                time_range=TimeRange.LAST_24_HOURS,
                filters={},
                visualization_config={
                    "show_percentage": True,
                    "show_absolute": True
                },
                refresh_interval=300,  # 5 minutes
                position={"x": 6, "y": 7, "width": 6, "height": 3},
                permissions=["compliance_viewer", "compliance_admin", "auditor"]
            )
        ]
        
        for widget in widgets:
            self.dashboard_widgets[widget.widget_id] = widget
            await self._store_widget_config(widget)
    
    async def get_dashboard_data(self, user_permissions: List[str], 
                               time_range: Optional[TimeRange] = None) -> Dict[str, Any]:
        """
        Get complete dashboard data for user with specified permissions
        
        Args:
            user_permissions: List of user permissions
            time_range: Optional time range override
            
        Returns:
            Dict containing complete dashboard data
        """
        dashboard_data = {
            "metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_permissions": user_permissions,
                "time_range": time_range.value if time_range else "default"
            },
            "widgets": {},
            "alerts": [],
            "summary_metrics": {},
            "trends": {}
        }
        
        # Get widget data for authorized widgets
        authorized_widgets = [
            widget for widget in self.dashboard_widgets.values()
            if any(perm in user_permissions for perm in widget.permissions)
        ]
        
        for widget in authorized_widgets:
            try:
                widget_time_range = time_range or widget.time_range
                widget_data = await self._get_widget_data(widget, widget_time_range)
                dashboard_data["widgets"][widget.widget_id] = widget_data
            except Exception as e:
                logger.error(f"Failed to get data for widget {widget.widget_id}: {str(e)}")
                dashboard_data["widgets"][widget.widget_id] = {
                    "error": f"Failed to load widget data: {str(e)}"
                }
        
        # Get active alerts
        dashboard_data["alerts"] = await self._get_active_alerts(user_permissions)
        
        # Get summary metrics
        dashboard_data["summary_metrics"] = await self._get_summary_metrics()
        
        # Get trend data
        dashboard_data["trends"] = await self._get_trend_data()
        
        return dashboard_data
    
    async def _get_widget_data(self, widget: DashboardWidget, time_range: TimeRange) -> Dict[str, Any]:
        """Get data for a specific dashboard widget"""
        # Calculate time range boundaries
        end_time = datetime.utcnow()
        if time_range == TimeRange.LAST_24_HOURS:
            start_time = end_time - timedelta(hours=24)
        elif time_range == TimeRange.LAST_7_DAYS:
            start_time = end_time - timedelta(days=7)
        elif time_range == TimeRange.LAST_30_DAYS:
            start_time = end_time - timedelta(days=30)
        elif time_range == TimeRange.LAST_90_DAYS:
            start_time = end_time - timedelta(days=90)
        elif time_range == TimeRange.LAST_YEAR:
            start_time = end_time - timedelta(days=365)
        else:
            start_time = end_time - timedelta(days=30)  # Default
        
        # Get metric calculator
        calculator = self.metric_calculators.get(widget.metric_type)
        if not calculator:
            raise ValueError(f"No calculator found for metric type: {widget.metric_type}")
        
        # Calculate metric data
        metric_data = await calculator(start_time, end_time, widget.filters)
        
        # Format data according to widget type
        widget_data = {
            "widget_config": asdict(widget),
            "data": metric_data,
            "last_updated": datetime.utcnow().isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
        
        return widget_data
    
    async def _calculate_compliance_score(self, start_time: datetime, end_time: datetime, 
                                        filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compliance score metric"""
        frameworks = filters.get("frameworks", ["GDPR", "SOC2", "ISO27001"])
        
        async with self.db_pool.acquire() as conn:
            # Get total policies and violations for each framework
            framework_scores = {}
            
            for framework in frameworks:
                # Get policy count
                policy_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM compliance_policies 
                    WHERE framework = $1 AND active = true
                """, framework)
                
                # Get violation count in time range
                violation_count = await conn.fetchval("""
                    SELECT COUNT(*) FROM compliance_violations v
                    JOIN compliance_policies p ON v.policy_id = p.policy_id
                    WHERE p.framework = $1 
                    AND v.detection_time BETWEEN $2 AND $3
                    AND v.status IN ('open', 'investigating')
                """, framework, start_time, end_time)
                
                # Calculate framework score (100% - violation penalty)
                if policy_count > 0:
                    violation_penalty = min((violation_count / policy_count) * 20, 80)
                    framework_score = max(20, 100 - violation_penalty)
                else:
                    framework_score = 100
                
                framework_scores[framework] = {
                    "score": round(framework_score, 2),
                    "policy_count": policy_count,
                    "violation_count": violation_count or 0
                }
            
            # Calculate overall score (weighted average)
            if framework_scores:
                overall_score = sum(data["score"] for data in framework_scores.values()) / len(framework_scores)
            else:
                overall_score = 0
        
        return {
            "overall_score": round(overall_score, 2),
            "framework_scores": framework_scores,
            "target_score": 95,
            "status": "compliant" if overall_score >= 80 else "non_compliant"
        }
    
    async def _calculate_violation_count(self, start_time: datetime, end_time: datetime, 
                                       filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate violation count metrics"""
        severity_filter = filters.get("severity", [])
        
        async with self.db_pool.acquire() as conn:
            # Base query
            where_conditions = ["detection_time BETWEEN $1 AND $2"]
            params = [start_time, end_time]
            
            if severity_filter:
                where_conditions.append(f"severity = ANY(${len(params) + 1})")
                params.append(severity_filter)
            
            where_clause = " AND ".join(where_conditions)
            
            # Get violation counts by severity
            violations_by_severity = await conn.fetch(f"""
                SELECT severity, COUNT(*) as count
                FROM compliance_violations 
                WHERE {where_clause}
                GROUP BY severity
            """, *params)
            
            # Get violations over time (daily buckets)
            violations_timeline = await conn.fetch(f"""
                SELECT 
                    DATE_TRUNC('day', detection_time) as date,
                    severity,
                    COUNT(*) as count
                FROM compliance_violations 
                WHERE {where_clause}
                GROUP BY DATE_TRUNC('day', detection_time), severity
                ORDER BY date
            """, *params)
            
            # Get total count
            total_violations = await conn.fetchval(f"""
                SELECT COUNT(*) FROM compliance_violations 
                WHERE {where_clause}
            """, *params)
        
        return {
            "total_violations": total_violations or 0,
            "by_severity": {row['severity']: row['count'] for row in violations_by_severity},
            "timeline": [
                {
                    "date": row['date'].isoformat(),
                    "severity": row['severity'],
                    "count": row['count']
                }
                for row in violations_timeline
            ]
        }
    
    async def _calculate_risk_level(self, start_time: datetime, end_time: datetime, 
                                  filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk level metrics"""
        async with self.db_pool.acquire() as conn:
            # Get current risk levels from risk register
            risk_data = await conn.fetch("""
                SELECT 
                    asset_id,
                    threat_id,
                    likelihood_level,
                    impact_level,
                    residual_risk_level,
                    risk_treatment
                FROM isms_risk_register 
                WHERE status = 'active'
            """)
            
            # Calculate risk distribution
            risk_distribution = defaultdict(int)
            for row in risk_data:
                risk_level = row['residual_risk_level']
                if risk_level <= 2:
                    risk_distribution['low'] += 1
                elif risk_level <= 3:
                    risk_distribution['medium'] += 1
                elif risk_level <= 4:
                    risk_distribution['high'] += 1
                else:
                    risk_distribution['very_high'] += 1
            
            # Calculate average risk score
            if risk_data:
                avg_risk = sum(row['residual_risk_level'] for row in risk_data) / len(risk_data)
            else:
                avg_risk = 0
        
        return {
            "average_risk_score": round(avg_risk, 2),
            "total_risks": len(risk_data),
            "risk_distribution": dict(risk_distribution),
            "high_risk_count": risk_distribution['high'] + risk_distribution['very_high']
        }
    
    async def create_compliance_alert(self, alert: ComplianceAlert) -> bool:
        """
        Create new compliance alert for dashboard
        
        Args:
            alert: Compliance alert to create
            
        Returns:
            bool: Success status
        """
        try:
            # Store alert in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO compliance_alerts 
                    (alert_id, alert_type, level, title, message, timestamp,
                     framework, affected_resources, recommended_actions, acknowledged)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                alert.alert_id, alert.alert_type, alert.level.value, alert.title,
                alert.message, alert.timestamp, alert.framework, alert.affected_resources,
                alert.recommended_actions, alert.acknowledged
                )
            
            # Cache alert for real-time display
            await self.redis_client.lpush(
                "compliance_alerts",
                json.dumps(asdict(alert), default=str)
            )
            await self.redis_client.ltrim("compliance_alerts", 0, 99)  # Keep last 100
            
            # Add to active alerts
            self.active_alerts[alert.alert_id] = alert
            
            logger.info(f"Compliance alert created: {alert.alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create compliance alert: {str(e)}")
            return False
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge compliance alert
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging the alert
            
        Returns:
            bool: Success status
        """
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE compliance_alerts 
                    SET acknowledged = true, acknowledged_by = $1, acknowledged_at = $2
                    WHERE alert_id = $3
                """, acknowledged_by, datetime.utcnow(), alert_id)
            
            # Update active alert
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.utcnow()
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {str(e)}")
            return False
    
    async def generate_compliance_report_pdf(self, report_config: Dict[str, Any]) -> bytes:
        """
        Generate PDF compliance report from dashboard data
        
        Args:
            report_config: Report configuration parameters
            
        Returns:
            bytes: PDF report content
        """
        # This would integrate with a PDF generation library like ReportLab
        # For now, return placeholder
        report_content = f"""
        Compliance Report Generated: {datetime.utcnow().isoformat()}
        
        Configuration: {json.dumps(report_config, indent=2)}
        
        This would contain:
        - Executive Summary
        - Compliance Scores by Framework
        - Violation Analysis
        - Risk Assessment
        - Recommendations
        - Detailed Metrics and Charts
        """
        
        return report_content.encode('utf-8')
    
    async def export_dashboard_data(self, format_type: str, filters: Dict[str, Any]) -> bytes:
        """
        Export dashboard data in specified format
        
        Args:
            format_type: Export format (csv, json, excel)
            filters: Data filters to apply
            
        Returns:
            bytes: Exported data
        """
        # Get dashboard data
        dashboard_data = await self.get_dashboard_data(["compliance_admin"])
        
        if format_type.lower() == "json":
            return json.dumps(dashboard_data, indent=2, default=str).encode('utf-8')
        elif format_type.lower() == "csv":
            # Convert to CSV format (simplified)
            csv_content = "metric,value,timestamp,framework\n"
            for widget_id, widget_data in dashboard_data["widgets"].items():
                if "data" in widget_data:
                    data = widget_data["data"]
                    if isinstance(data, dict) and "overall_score" in data:
                        csv_content += f"compliance_score,{data['overall_score']},{datetime.utcnow().isoformat()},overall\n"
            return csv_content.encode('utf-8')
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

# Database schema for compliance dashboard
COMPLIANCE_DASHBOARD_SCHEMA_SQL = """
-- Dashboard Widget Configurations
CREATE TABLE IF NOT EXISTS dashboard_widgets (
    widget_id VARCHAR(255) PRIMARY KEY,
    widget_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    metric_type VARCHAR(50) NOT NULL,
    time_range VARCHAR(20) NOT NULL,
    filters JSONB,
    visualization_config JSONB,
    refresh_interval INTEGER DEFAULT 300,
    position JSONB,
    permissions TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Compliance Alerts
CREATE TABLE IF NOT EXISTS compliance_alerts (
    alert_id VARCHAR(255) PRIMARY KEY,
    alert_type VARCHAR(100) NOT NULL,
    level VARCHAR(20) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    framework VARCHAR(50) NOT NULL,
    affected_resources TEXT[],
    recommended_actions TEXT[],
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Dashboard Metrics History
CREATE TABLE IF NOT EXISTS dashboard_metrics_history (
    metric_id VARCHAR(255) PRIMARY KEY,
    metric_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    target_value DECIMAL(10,2),
    threshold_warning DECIMAL(10,2),
    threshold_critical DECIMAL(10,2),
    framework VARCHAR(50),
    context JSONB,
    trend VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User Dashboard Preferences
CREATE TABLE IF NOT EXISTS user_dashboard_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    widget_layout JSONB,
    default_time_range VARCHAR(20) DEFAULT 'last_30_days',
    refresh_interval INTEGER DEFAULT 300,
    alert_preferences JSONB,
    theme VARCHAR(20) DEFAULT 'light',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_dashboard_widgets_type ON dashboard_widgets(widget_type);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_timestamp ON compliance_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_level ON compliance_alerts(level);
CREATE INDEX IF NOT EXISTS idx_compliance_alerts_acknowledged ON compliance_alerts(acknowledged);
CREATE INDEX IF NOT EXISTS idx_metrics_history_timestamp ON dashboard_metrics_history(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_history_type ON dashboard_metrics_history(metric_type);
"""