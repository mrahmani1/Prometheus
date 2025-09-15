"""
Data Quality Monitoring and Validation Pipeline System

This module provides comprehensive data quality monitoring, validation, and
alerting capabilities for geological data across all database systems in the
Prometheus AI Platform with advanced anomaly detection and automated remediation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import statistics
from collections import defaultdict, deque

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import asyncpg
import redis.asyncio as redis
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class QualityDimension(Enum):
    """Data quality dimensions to monitor"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    INTEGRITY = "integrity"
    CONFORMITY = "conformity"


class QualityRuleType(Enum):
    """Types of data quality rules"""
    RANGE_CHECK = "range_check"
    NULL_CHECK = "null_check"
    FORMAT_CHECK = "format_check"
    UNIQUENESS_CHECK = "uniqueness_check"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    STATISTICAL_OUTLIER = "statistical_outlier"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    CROSS_FIELD_VALIDATION = "cross_field_validation"
    BUSINESS_RULE = "business_rule"


class AlertSeverity(Enum):
    """Severity levels for quality alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityRule:
    """Definition of a data quality rule"""
    rule_id: str
    rule_name: str
    rule_type: QualityRuleType
    dimension: QualityDimension
    database_name: str
    table_name: str
    column_name: Optional[str]
    rule_definition: Dict[str, Any]
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    created_by: str = "system"
    created_timestamp: datetime = None
    
    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = datetime.utcnow()


@dataclass
class QualityMetric:
    """Data quality metric measurement"""
    metric_id: str
    rule_id: str
    database_name: str
    table_name: str
    column_name: Optional[str]
    dimension: QualityDimension
    measured_value: float
    threshold: float
    passed: bool
    measurement_timestamp: datetime
    sample_size: int
    metadata: Dict[str, Any]


@dataclass
class QualityAlert:
    """Data quality alert"""
    alert_id: str
    rule_id: str
    metric_id: str
    severity: AlertSeverity
    message: str
    database_name: str
    table_name: str
    column_name: Optional[str]
    measured_value: float
    threshold: float
    deviation: float
    alert_timestamp: datetime
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    report_id: str
    database_name: str
    table_name: str
    report_timestamp: datetime
    overall_quality_score: float
    dimension_scores: Dict[QualityDimension, float]
    total_rules_evaluated: int
    rules_passed: int
    rules_failed: int
    alerts_generated: int
    critical_issues: List[str]
    recommendations: List[str]


class GeologicalDataQualityRules:
    """Predefined quality rules for geological data"""
    
    @staticmethod
    def get_seismic_data_rules() -> List[QualityRule]:
        """Get quality rules for seismic data"""
        return [
            QualityRule(
                rule_id="seismic_magnitude_range",
                rule_name="Seismic Magnitude Range Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="seismic_events",
                column_name="magnitude",
                rule_definition={
                    "min_value": -2.0,
                    "max_value": 10.0
                },
                threshold=0.95,
                severity=AlertSeverity.ERROR
            ),
            QualityRule(
                rule_id="seismic_depth_validity",
                rule_name="Seismic Depth Validity Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="seismic_events",
                column_name="depth_km",
                rule_definition={
                    "min_value": 0.0,
                    "max_value": 700.0
                },
                threshold=0.98,
                severity=AlertSeverity.ERROR
            ),
            QualityRule(
                rule_id="seismic_coordinates_validity",
                rule_name="Seismic Coordinates Validity Check",
                rule_type=QualityRuleType.CROSS_FIELD_VALIDATION,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="seismic_events",
                column_name=None,
                rule_definition={
                    "latitude_range": [-90.0, 90.0],
                    "longitude_range": [-180.0, 180.0]
                },
                threshold=0.99,
                severity=AlertSeverity.CRITICAL
            ),
            QualityRule(
                rule_id="seismic_timestamp_timeliness",
                rule_name="Seismic Data Timeliness Check",
                rule_type=QualityRuleType.TEMPORAL_CONSISTENCY,
                dimension=QualityDimension.TIMELINESS,
                database_name="postgresql",
                table_name="seismic_events",
                column_name="timestamp",
                rule_definition={
                    "max_delay_hours": 2.0,
                    "future_tolerance_minutes": 5.0
                },
                threshold=0.90,
                severity=AlertSeverity.WARNING
            ),
            QualityRule(
                rule_id="seismic_completeness_check",
                rule_name="Seismic Data Completeness Check",
                rule_type=QualityRuleType.NULL_CHECK,
                dimension=QualityDimension.COMPLETENESS,
                database_name="postgresql",
                table_name="seismic_events",
                column_name=None,
                rule_definition={
                    "required_fields": ["timestamp", "station_id", "magnitude", "latitude", "longitude"]
                },
                threshold=0.95,
                severity=AlertSeverity.ERROR
            )
        ]
    
    @staticmethod
    def get_deformation_data_rules() -> List[QualityRule]:
        """Get quality rules for deformation data"""
        return [
            QualityRule(
                rule_id="deformation_displacement_range",
                rule_name="Deformation Displacement Range Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="mtsar_interferograms",
                column_name="mean_displacement_mm",
                rule_definition={
                    "min_value": -1000.0,
                    "max_value": 1000.0
                },
                threshold=0.95,
                severity=AlertSeverity.ERROR
            ),
            QualityRule(
                rule_id="deformation_coherence_quality",
                rule_name="MT-SAR Coherence Quality Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.ACCURACY,
                database_name="postgresql",
                table_name="mtsar_interferograms",
                column_name="average_coherence",
                rule_definition={
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "quality_threshold": 0.3
                },
                threshold=0.80,
                severity=AlertSeverity.WARNING
            ),
            QualityRule(
                rule_id="deformation_temporal_baseline",
                rule_name="Temporal Baseline Validity Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="mtsar_interferograms",
                column_name="temporal_baseline_days",
                rule_definition={
                    "min_value": 1,
                    "max_value": 365
                },
                threshold=0.98,
                severity=AlertSeverity.ERROR
            )
        ]
    
    @staticmethod
    def get_geochemical_data_rules() -> List[QualityRule]:
        """Get quality rules for geochemical data"""
        return [
            QualityRule(
                rule_id="geochemical_sio2_range",
                rule_name="SiO2 Content Range Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.VALIDITY,
                database_name="postgresql",
                table_name="geochemical_analyses",
                column_name="sio2_content",
                rule_definition={
                    "min_value": 35.0,
                    "max_value": 80.0
                },
                threshold=0.95,
                severity=AlertSeverity.ERROR
            ),
            QualityRule(
                rule_id="geochemical_total_oxides",
                rule_name="Total Oxides Sum Check",
                rule_type=QualityRuleType.BUSINESS_RULE,
                dimension=QualityDimension.CONSISTENCY,
                database_name="postgresql",
                table_name="geochemical_analyses",
                column_name=None,
                rule_definition={
                    "oxide_fields": ["sio2_content", "al2o3_content", "mgo_content", "cao_content", "na2o_content", "k2o_content"],
                    "expected_sum_range": [95.0, 105.0]
                },
                threshold=0.90,
                severity=AlertSeverity.WARNING
            ),
            QualityRule(
                rule_id="geochemical_analytical_uncertainty",
                rule_name="Analytical Uncertainty Check",
                rule_type=QualityRuleType.RANGE_CHECK,
                dimension=QualityDimension.ACCURACY,
                database_name="postgresql",
                table_name="geochemical_analyses",
                column_name="analytical_uncertainty",
                rule_definition={
                    "max_uncertainty_percent": 10.0
                },
                threshold=0.85,
                severity=AlertSeverity.WARNING
            )
        ]


class DataQualityMonitor:
    """Advanced data quality monitoring system"""
    
    def __init__(self,
                 postgres_connection_string: str,
                 influxdb_url: str,
                 influxdb_token: str,
                 influxdb_org: str,
                 redis_url: str):
        self.postgres_connection_string = postgres_connection_string
        self.influxdb_url = influxdb_url
        self.influxdb_token = influxdb_token
        self.influxdb_org = influxdb_org
        self.redis_url = redis_url
        
        self.postgres_conn = None
        self.influxdb_client = None
        self.redis_client = None
        
        # Quality rules registry
        self.quality_rules: Dict[str, QualityRule] = {}
        
        # Anomaly detection models
        self.anomaly_detectors: Dict[str, IsolationForest] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable[[QualityAlert], None]] = []
        
        # Monitoring statistics
        self.monitoring_stats = {
            'rules_evaluated': 0,
            'alerts_generated': 0,
            'quality_checks_performed': 0,
            'anomalies_detected': 0
        }
    
    async def initialize(self) -> None:
        """Initialize database connections and load quality rules"""
        try:
            # Initialize database connections
            self.postgres_conn = await asyncpg.connect(self.postgres_connection_string)
            self.influxdb_client = InfluxDBClientAsync(
                url=self.influxdb_url,
                token=self.influxdb_token,
                org=self.influxdb_org
            )
            self.redis_client = redis.from_url(self.redis_url)
            
            # Create database schema
            await self._create_quality_schema()
            
            # Load predefined quality rules
            await self._load_predefined_rules()
            
            # Initialize anomaly detection models
            await self._initialize_anomaly_detectors()
            
            logger.info("Data quality monitor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize data quality monitor", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        try:
            if self.postgres_conn:
                await self.postgres_conn.close()
            
            if self.influxdb_client:
                await self.influxdb_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Data quality monitor connections closed")
            
        except Exception as e:
            logger.error("Failed to close connections", error=str(e))
    
    async def _create_quality_schema(self) -> None:
        """Create database schema for quality monitoring"""
        try:
            # Create quality rules table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_rules (
                    rule_id VARCHAR(100) PRIMARY KEY,
                    rule_name VARCHAR(255) NOT NULL,
                    rule_type VARCHAR(50) NOT NULL,
                    dimension VARCHAR(50) NOT NULL,
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    column_name VARCHAR(100),
                    rule_definition JSONB NOT NULL,
                    threshold FLOAT NOT NULL,
                    severity VARCHAR(20) NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_by VARCHAR(100),
                    created_timestamp TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create quality metrics table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    rule_id VARCHAR(100) REFERENCES quality_rules(rule_id),
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    column_name VARCHAR(100),
                    dimension VARCHAR(50) NOT NULL,
                    measured_value FLOAT NOT NULL,
                    threshold FLOAT NOT NULL,
                    passed BOOLEAN NOT NULL,
                    measurement_timestamp TIMESTAMP DEFAULT NOW(),
                    sample_size INTEGER,
                    metadata JSONB
                )
            """)
            
            # Create quality alerts table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_alerts (
                    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    rule_id VARCHAR(100) REFERENCES quality_rules(rule_id),
                    metric_id UUID REFERENCES quality_metrics(metric_id),
                    severity VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    column_name VARCHAR(100),
                    measured_value FLOAT NOT NULL,
                    threshold FLOAT NOT NULL,
                    deviation FLOAT NOT NULL,
                    alert_timestamp TIMESTAMP DEFAULT NOW(),
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_timestamp TIMESTAMP,
                    resolution_notes TEXT
                )
            """)
            
            # Create quality reports table
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS quality_reports (
                    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    database_name VARCHAR(100) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    report_timestamp TIMESTAMP DEFAULT NOW(),
                    overall_quality_score FLOAT NOT NULL,
                    dimension_scores JSONB NOT NULL,
                    total_rules_evaluated INTEGER NOT NULL,
                    rules_passed INTEGER NOT NULL,
                    rules_failed INTEGER NOT NULL,
                    alerts_generated INTEGER NOT NULL,
                    critical_issues TEXT[],
                    recommendations TEXT[]
                )
            """)
            
            # Create indexes for performance
            await self.postgres_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_metrics_timestamp ON quality_metrics(measurement_timestamp);
                CREATE INDEX IF NOT EXISTS idx_quality_metrics_table ON quality_metrics(database_name, table_name);
                CREATE INDEX IF NOT EXISTS idx_quality_alerts_timestamp ON quality_alerts(alert_timestamp);
                CREATE INDEX IF NOT EXISTS idx_quality_alerts_severity ON quality_alerts(severity);
                CREATE INDEX IF NOT EXISTS idx_quality_reports_timestamp ON quality_reports(report_timestamp);
            """)
            
            logger.info("Quality monitoring database schema created")
            
        except Exception as e:
            logger.error("Failed to create quality schema", error=str(e))
            raise
    
    async def _load_predefined_rules(self) -> None:
        """Load predefined quality rules for geological data"""
        try:
            # Get all predefined rules
            all_rules = []
            all_rules.extend(GeologicalDataQualityRules.get_seismic_data_rules())
            all_rules.extend(GeologicalDataQualityRules.get_deformation_data_rules())
            all_rules.extend(GeologicalDataQualityRules.get_geochemical_data_rules())
            
            # Store rules in database and memory
            for rule in all_rules:
                await self._store_quality_rule(rule)
                self.quality_rules[rule.rule_id] = rule
            
            logger.info("Loaded predefined quality rules", 
                       total_rules=len(all_rules))
            
        except Exception as e:
            logger.error("Failed to load predefined rules", error=str(e))
            raise
    
    async def _store_quality_rule(self, rule: QualityRule) -> None:
        """Store quality rule in database"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO quality_rules 
                (rule_id, rule_name, rule_type, dimension, database_name, table_name,
                 column_name, rule_definition, threshold, severity, enabled, created_by)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (rule_id) DO UPDATE SET
                    rule_name = EXCLUDED.rule_name,
                    rule_definition = EXCLUDED.rule_definition,
                    threshold = EXCLUDED.threshold,
                    enabled = EXCLUDED.enabled
            """,
                rule.rule_id, rule.rule_name, rule.rule_type.value,
                rule.dimension.value, rule.database_name, rule.table_name,
                rule.column_name, json.dumps(rule.rule_definition),
                rule.threshold, rule.severity.value, rule.enabled, rule.created_by
            )
            
        except Exception as e:
            logger.error("Failed to store quality rule", 
                        rule_id=rule.rule_id,
                        error=str(e))
            raise
    
    async def _initialize_anomaly_detectors(self) -> None:
        """Initialize anomaly detection models for different data types"""
        try:
            # Initialize isolation forest models for different geological data types
            self.anomaly_detectors['seismic_magnitude'] = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            self.anomaly_detectors['deformation_rate'] = IsolationForest(
                contamination=0.05,
                random_state=42
            )
            
            self.anomaly_detectors['geochemical_composition'] = IsolationForest(
                contamination=0.08,
                random_state=42
            )
            
            # Train models with historical data if available
            await self._train_anomaly_detectors()
            
            logger.info("Initialized anomaly detection models")
            
        except Exception as e:
            logger.error("Failed to initialize anomaly detectors", error=str(e))
    
    async def _train_anomaly_detectors(self) -> None:
        """Train anomaly detection models with historical data"""
        try:
            # Train seismic magnitude anomaly detector
            seismic_data = await self.postgres_conn.fetch("""
                SELECT magnitude, depth_km, latitude, longitude
                FROM seismic_events
                WHERE timestamp >= NOW() - INTERVAL '30 days'
                AND magnitude IS NOT NULL
                LIMIT 10000
            """)
            
            if seismic_data:
                seismic_df = pd.DataFrame(seismic_data)
                seismic_features = seismic_df[['magnitude', 'depth_km', 'latitude', 'longitude']].fillna(0)
                
                if len(seismic_features) > 100:
                    scaler = StandardScaler()
                    seismic_scaled = scaler.fit_transform(seismic_features)
                    self.anomaly_detectors['seismic_magnitude'].fit(seismic_scaled)
                    
                    logger.info("Trained seismic anomaly detector", 
                               samples=len(seismic_features))
            
            # Train deformation anomaly detector
            deformation_data = await self.postgres_conn.fetch("""
                SELECT mean_displacement_mm, std_displacement_mm, 
                       deformation_rate_mm_year, valid_pixel_percentage
                FROM processed_mtsar_deformation
                WHERE processing_timestamp >= NOW() - INTERVAL '30 days'
                LIMIT 5000
            """)
            
            if deformation_data:
                deformation_df = pd.DataFrame(deformation_data)
                deformation_features = deformation_df.fillna(0)
                
                if len(deformation_features) > 50:
                    scaler = StandardScaler()
                    deformation_scaled = scaler.fit_transform(deformation_features)
                    self.anomaly_detectors['deformation_rate'].fit(deformation_scaled)
                    
                    logger.info("Trained deformation anomaly detector", 
                               samples=len(deformation_features))
            
        except Exception as e:
            logger.error("Failed to train anomaly detectors", error=str(e))


# Continue with quality evaluation methods in next part...
    async def evaluate_quality_rules(self, 
                                    database_name: str,
                                    table_name: str,
                                    time_window_hours: int = 24) -> QualityReport:
        """Evaluate all quality rules for a specific table"""
        try:
            report_id = str(uuid.uuid4())
            report_timestamp = datetime.utcnow()
            
            # Get applicable rules
            applicable_rules = [
                rule for rule in self.quality_rules.values()
                if (rule.database_name == database_name and 
                    rule.table_name == table_name and 
                    rule.enabled)
            ]
            
            if not applicable_rules:
                logger.warning("No quality rules found", 
                             database=database_name,
                             table=table_name)
                return self._create_empty_report(report_id, database_name, table_name, report_timestamp)
            
            # Evaluate each rule
            metrics = []
            alerts = []
            dimension_scores = defaultdict(list)
            
            for rule in applicable_rules:
                try:
                    metric = await self._evaluate_single_rule(rule, time_window_hours)
                    metrics.append(metric)
                    
                    # Track dimension scores
                    dimension_scores[rule.dimension].append(1.0 if metric.passed else 0.0)
                    
                    # Generate alert if rule failed
                    if not metric.passed:
                        alert = await self._generate_quality_alert(rule, metric)
                        alerts.append(alert)
                        
                        # Notify alert handlers
                        for handler in self.alert_handlers:
                            try:
                                await handler(alert)
                            except Exception as e:
                                logger.error("Alert handler failed", error=str(e))
                    
                    self.monitoring_stats['rules_evaluated'] += 1
                    
                except Exception as e:
                    logger.error("Failed to evaluate quality rule", 
                               rule_id=rule.rule_id,
                               error=str(e))
                    continue
            
            # Calculate overall scores
            overall_quality_score = sum(1.0 if m.passed else 0.0 for m in metrics) / len(metrics) if metrics else 0.0
            
            dimension_score_averages = {
                dimension: statistics.mean(scores) if scores else 0.0
                for dimension, scores in dimension_scores.items()
            }
            
            # Generate recommendations
            recommendations = await self._generate_quality_recommendations(
                metrics, alerts, dimension_score_averages
            )
            
            # Identify critical issues
            critical_issues = [
                alert.message for alert in alerts 
                if alert.severity == AlertSeverity.CRITICAL
            ]
            
            # Create quality report
            report = QualityReport(
                report_id=report_id,
                database_name=database_name,
                table_name=table_name,
                report_timestamp=report_timestamp,
                overall_quality_score=overall_quality_score,
                dimension_scores=dimension_score_averages,
                total_rules_evaluated=len(applicable_rules),
                rules_passed=sum(1 for m in metrics if m.passed),
                rules_failed=sum(1 for m in metrics if not m.passed),
                alerts_generated=len(alerts),
                critical_issues=critical_issues,
                recommendations=recommendations
            )
            
            # Store report
            await self._store_quality_report(report)
            
            # Store metrics and alerts
            for metric in metrics:
                await self._store_quality_metric(metric)
            
            for alert in alerts:
                await self._store_quality_alert(alert)
            
            self.monitoring_stats['quality_checks_performed'] += 1
            self.monitoring_stats['alerts_generated'] += len(alerts)
            
            logger.info("Quality evaluation completed", 
                       database=database_name,
                       table=table_name,
                       overall_score=overall_quality_score,
                       alerts=len(alerts))
            
            return report
            
        except Exception as e:
            logger.error("Quality evaluation failed", 
                        database=database_name,
                        table=table_name,
                        error=str(e))
            raise
    
    async def _evaluate_single_rule(self, 
                                   rule: QualityRule,
                                   time_window_hours: int) -> QualityMetric:
        """Evaluate a single quality rule"""
        try:
            metric_id = str(uuid.uuid4())
            measurement_timestamp = datetime.utcnow()
            
            if rule.rule_type == QualityRuleType.RANGE_CHECK:
                measured_value, sample_size, metadata = await self._evaluate_range_check(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.NULL_CHECK:
                measured_value, sample_size, metadata = await self._evaluate_null_check(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.UNIQUENESS_CHECK:
                measured_value, sample_size, metadata = await self._evaluate_uniqueness_check(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.TEMPORAL_CONSISTENCY:
                measured_value, sample_size, metadata = await self._evaluate_temporal_consistency(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.CROSS_FIELD_VALIDATION:
                measured_value, sample_size, metadata = await self._evaluate_cross_field_validation(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.BUSINESS_RULE:
                measured_value, sample_size, metadata = await self._evaluate_business_rule(rule, time_window_hours)
            elif rule.rule_type == QualityRuleType.STATISTICAL_OUTLIER:
                measured_value, sample_size, metadata = await self._evaluate_statistical_outlier(rule, time_window_hours)
            else:
                raise ValueError(f"Unsupported rule type: {rule.rule_type}")
            
            # Determine if rule passed
            passed = measured_value >= rule.threshold
            
            metric = QualityMetric(
                metric_id=metric_id,
                rule_id=rule.rule_id,
                database_name=rule.database_name,
                table_name=rule.table_name,
                column_name=rule.column_name,
                dimension=rule.dimension,
                measured_value=measured_value,
                threshold=rule.threshold,
                passed=passed,
                measurement_timestamp=measurement_timestamp,
                sample_size=sample_size,
                metadata=metadata
            )
            
            return metric
            
        except Exception as e:
            logger.error("Failed to evaluate single rule", 
                        rule_id=rule.rule_id,
                        error=str(e))
            raise
    
    async def _evaluate_range_check(self, 
                                   rule: QualityRule,
                                   time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate range check rule"""
        try:
            min_value = rule.rule_definition.get('min_value')
            max_value = rule.rule_definition.get('max_value')
            
            # Build query
            time_filter = f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'" if 'timestamp' in await self._get_table_columns(rule.table_name) else "TRUE"
            
            query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN {rule.column_name} IS NOT NULL THEN 1 END) as non_null_count,
                    COUNT(CASE WHEN {rule.column_name} >= $1 AND {rule.column_name} <= $2 THEN 1 END) as valid_count,
                    MIN({rule.column_name}) as min_observed,
                    MAX({rule.column_name}) as max_observed,
                    AVG({rule.column_name}) as avg_value
                FROM {rule.table_name}
                WHERE {time_filter}
            """
            
            result = await self.postgres_conn.fetchrow(query, min_value, max_value)
            
            total_count = result['total_count']
            non_null_count = result['non_null_count']
            valid_count = result['valid_count']
            
            # Calculate compliance rate
            compliance_rate = valid_count / non_null_count if non_null_count > 0 else 0.0
            
            metadata = {
                'total_records': total_count,
                'non_null_records': non_null_count,
                'valid_records': valid_count,
                'min_observed': float(result['min_observed']) if result['min_observed'] else None,
                'max_observed': float(result['max_observed']) if result['max_observed'] else None,
                'avg_value': float(result['avg_value']) if result['avg_value'] else None,
                'expected_range': [min_value, max_value]
            }
            
            return compliance_rate, total_count, metadata
            
        except Exception as e:
            logger.error("Range check evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_null_check(self, 
                                  rule: QualityRule,
                                  time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate null check rule (completeness)"""
        try:
            required_fields = rule.rule_definition.get('required_fields', [])
            
            if not required_fields:
                return 1.0, 0, {}
            
            # Build query to check completeness
            time_filter = f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'" if 'timestamp' in await self._get_table_columns(rule.table_name) else "TRUE"
            
            # Check each required field
            field_completeness = {}
            total_records = 0
            
            for field in required_fields:
                query = f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT({field}) as non_null_count
                    FROM {rule.table_name}
                    WHERE {time_filter}
                """
                
                result = await self.postgres_conn.fetchrow(query)
                total_records = result['total_count']
                non_null_count = result['non_null_count']
                
                completeness = non_null_count / total_records if total_records > 0 else 0.0
                field_completeness[field] = completeness
            
            # Calculate overall completeness
            overall_completeness = statistics.mean(field_completeness.values()) if field_completeness else 0.0
            
            metadata = {
                'total_records': total_records,
                'field_completeness': field_completeness,
                'required_fields': required_fields
            }
            
            return overall_completeness, total_records, metadata
            
        except Exception as e:
            logger.error("Null check evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_uniqueness_check(self, 
                                       rule: QualityRule,
                                       time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate uniqueness check rule"""
        try:
            time_filter = f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'" if 'timestamp' in await self._get_table_columns(rule.table_name) else "TRUE"
            
            query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(DISTINCT {rule.column_name}) as unique_count
                FROM {rule.table_name}
                WHERE {time_filter} AND {rule.column_name} IS NOT NULL
            """
            
            result = await self.postgres_conn.fetchrow(query)
            
            total_count = result['total_count']
            unique_count = result['unique_count']
            
            # Calculate uniqueness rate
            uniqueness_rate = unique_count / total_count if total_count > 0 else 0.0
            
            metadata = {
                'total_records': total_count,
                'unique_records': unique_count,
                'duplicate_records': total_count - unique_count,
                'uniqueness_rate': uniqueness_rate
            }
            
            return uniqueness_rate, total_count, metadata
            
        except Exception as e:
            logger.error("Uniqueness check evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_temporal_consistency(self, 
                                           rule: QualityRule,
                                           time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate temporal consistency rule"""
        try:
            max_delay_hours = rule.rule_definition.get('max_delay_hours', 24.0)
            future_tolerance_minutes = rule.rule_definition.get('future_tolerance_minutes', 5.0)
            
            query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN {rule.column_name} <= NOW() + INTERVAL '{future_tolerance_minutes} minutes' 
                               AND {rule.column_name} >= NOW() - INTERVAL '{max_delay_hours} hours' 
                          THEN 1 END) as timely_count,
                    COUNT(CASE WHEN {rule.column_name} > NOW() + INTERVAL '{future_tolerance_minutes} minutes' 
                          THEN 1 END) as future_count,
                    COUNT(CASE WHEN {rule.column_name} < NOW() - INTERVAL '{max_delay_hours} hours' 
                          THEN 1 END) as delayed_count
                FROM {rule.table_name}
                WHERE {rule.column_name} >= NOW() - INTERVAL '{time_window_hours} hours'
                AND {rule.column_name} IS NOT NULL
            """
            
            result = await self.postgres_conn.fetchrow(query)
            
            total_count = result['total_count']
            timely_count = result['timely_count']
            future_count = result['future_count']
            delayed_count = result['delayed_count']
            
            # Calculate timeliness rate
            timeliness_rate = timely_count / total_count if total_count > 0 else 0.0
            
            metadata = {
                'total_records': total_count,
                'timely_records': timely_count,
                'future_records': future_count,
                'delayed_records': delayed_count,
                'max_delay_hours': max_delay_hours,
                'future_tolerance_minutes': future_tolerance_minutes
            }
            
            return timeliness_rate, total_count, metadata
            
        except Exception as e:
            logger.error("Temporal consistency evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_cross_field_validation(self, 
                                             rule: QualityRule,
                                             time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate cross-field validation rule"""
        try:
            if rule.rule_id == "seismic_coordinates_validity":
                # Validate latitude and longitude ranges
                lat_range = rule.rule_definition.get('latitude_range', [-90, 90])
                lon_range = rule.rule_definition.get('longitude_range', [-180, 180])
                
                time_filter = f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'" if 'timestamp' in await self._get_table_columns(rule.table_name) else "TRUE"
                
                query = f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT(CASE WHEN latitude BETWEEN $1 AND $2 
                                   AND longitude BETWEEN $3 AND $4 
                              THEN 1 END) as valid_coordinates
                    FROM {rule.table_name}
                    WHERE {time_filter}
                    AND latitude IS NOT NULL 
                    AND longitude IS NOT NULL
                """
                
                result = await self.postgres_conn.fetchrow(
                    query, lat_range[0], lat_range[1], lon_range[0], lon_range[1]
                )
                
                total_count = result['total_count']
                valid_count = result['valid_coordinates']
                
                validity_rate = valid_count / total_count if total_count > 0 else 0.0
                
                metadata = {
                    'total_records': total_count,
                    'valid_coordinates': valid_count,
                    'invalid_coordinates': total_count - valid_count,
                    'latitude_range': lat_range,
                    'longitude_range': lon_range
                }
                
                return validity_rate, total_count, metadata
            
            return 1.0, 0, {}
            
        except Exception as e:
            logger.error("Cross-field validation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_business_rule(self, 
                                    rule: QualityRule,
                                    time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate business rule"""
        try:
            if rule.rule_id == "geochemical_total_oxides":
                # Check if total oxides sum is within expected range
                oxide_fields = rule.rule_definition.get('oxide_fields', [])
                expected_range = rule.rule_definition.get('expected_sum_range', [95, 105])
                
                if not oxide_fields:
                    return 1.0, 0, {}
                
                # Build sum expression
                sum_expression = ' + '.join([f'COALESCE({field}, 0)' for field in oxide_fields])
                
                time_filter = f"collection_date >= NOW() - INTERVAL '{time_window_hours} hours'" if 'collection_date' in await self._get_table_columns(rule.table_name) else "TRUE"
                
                query = f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT(CASE WHEN ({sum_expression}) BETWEEN $1 AND $2 THEN 1 END) as valid_sum_count,
                        AVG({sum_expression}) as avg_sum,
                        MIN({sum_expression}) as min_sum,
                        MAX({sum_expression}) as max_sum
                    FROM {rule.table_name}
                    WHERE {time_filter}
                """
                
                result = await self.postgres_conn.fetchrow(query, expected_range[0], expected_range[1])
                
                total_count = result['total_count']
                valid_count = result['valid_sum_count']
                
                compliance_rate = valid_count / total_count if total_count > 0 else 0.0
                
                metadata = {
                    'total_records': total_count,
                    'valid_sum_records': valid_count,
                    'invalid_sum_records': total_count - valid_count,
                    'expected_range': expected_range,
                    'avg_sum': float(result['avg_sum']) if result['avg_sum'] else None,
                    'min_sum': float(result['min_sum']) if result['min_sum'] else None,
                    'max_sum': float(result['max_sum']) if result['max_sum'] else None,
                    'oxide_fields': oxide_fields
                }
                
                return compliance_rate, total_count, metadata
            
            return 1.0, 0, {}
            
        except Exception as e:
            logger.error("Business rule evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _evaluate_statistical_outlier(self, 
                                          rule: QualityRule,
                                          time_window_hours: int) -> Tuple[float, int, Dict[str, Any]]:
        """Evaluate statistical outlier detection rule"""
        try:
            time_filter = f"timestamp >= NOW() - INTERVAL '{time_window_hours} hours'" if 'timestamp' in await self._get_table_columns(rule.table_name) else "TRUE"
            
            # Get data for outlier detection
            query = f"""
                SELECT {rule.column_name}
                FROM {rule.table_name}
                WHERE {time_filter} AND {rule.column_name} IS NOT NULL
                ORDER BY {rule.column_name}
            """
            
            records = await self.postgres_conn.fetch(query)
            
            if len(records) < 10:  # Need minimum data for statistical analysis
                return 1.0, len(records), {'insufficient_data': True}
            
            values = [float(record[rule.column_name]) for record in records]
            
            # Use IQR method for outlier detection
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            outlier_count = len(outliers)
            
            # Calculate non-outlier rate
            non_outlier_rate = (len(values) - outlier_count) / len(values)
            
            metadata = {
                'total_records': len(values),
                'outlier_count': outlier_count,
                'non_outlier_rate': non_outlier_rate,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers[:10]  # Limit to first 10 outliers
            }
            
            return non_outlier_rate, len(values), metadata
            
        except Exception as e:
            logger.error("Statistical outlier evaluation failed", error=str(e))
            return 0.0, 0, {}
    
    async def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        try:
            query = """
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = $1
            """
            
            records = await self.postgres_conn.fetch(query, table_name)
            return [record['column_name'] for record in records]
            
        except Exception as e:
            logger.error("Failed to get table columns", 
                        table=table_name,
                        error=str(e))
            return []
    
    async def _generate_quality_alert(self, 
                                    rule: QualityRule,
                                    metric: QualityMetric) -> QualityAlert:
        """Generate quality alert for failed rule"""
        try:
            alert_id = str(uuid.uuid4())
            deviation = abs(metric.measured_value - metric.threshold)
            
            # Generate alert message
            message = f"Quality rule '{rule.rule_name}' failed. "
            message += f"Measured value: {metric.measured_value:.3f}, "
            message += f"Threshold: {metric.threshold:.3f}, "
            message += f"Deviation: {deviation:.3f}"
            
            if metric.metadata:
                if 'total_records' in metric.metadata:
                    message += f", Records analyzed: {metric.metadata['total_records']}"
            
            alert = QualityAlert(
                alert_id=alert_id,
                rule_id=rule.rule_id,
                metric_id=metric.metric_id,
                severity=rule.severity,
                message=message,
                database_name=rule.database_name,
                table_name=rule.table_name,
                column_name=rule.column_name,
                measured_value=metric.measured_value,
                threshold=metric.threshold,
                deviation=deviation,
                alert_timestamp=datetime.utcnow()
            )
            
            return alert
            
        except Exception as e:
            logger.error("Failed to generate quality alert", error=str(e))
            raise
    
    async def _generate_quality_recommendations(self,
                                              metrics: List[QualityMetric],
                                              alerts: List[QualityAlert],
                                              dimension_scores: Dict[QualityDimension, float]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        try:
            # Analyze dimension scores
            for dimension, score in dimension_scores.items():
                if score < 0.7:  # Low quality threshold
                    if dimension == QualityDimension.COMPLETENESS:
                        recommendations.append(
                            "Improve data completeness by implementing validation at data ingestion points."
                        )
                    elif dimension == QualityDimension.ACCURACY:
                        recommendations.append(
                            "Enhance data accuracy through improved calibration and validation procedures."
                        )
                    elif dimension == QualityDimension.TIMELINESS:
                        recommendations.append(
                            "Optimize data processing pipelines to reduce latency and improve timeliness."
                        )
                    elif dimension == QualityDimension.VALIDITY:
                        recommendations.append(
                            "Implement stricter validation rules and range checks at data entry points."
                        )
            
            # Analyze critical alerts
            critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
            if critical_alerts:
                recommendations.append(
                    f"Address {len(critical_alerts)} critical data quality issues immediately."
                )
            
            # Analyze failed metrics
            failed_metrics = [m for m in metrics if not m.passed]
            if len(failed_metrics) > len(metrics) * 0.3:  # More than 30% failed
                recommendations.append(
                    "Consider reviewing and adjusting quality thresholds as many rules are failing."
                )
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("Data quality is within acceptable ranges. Continue monitoring.")
            
            recommendations.append("Schedule regular data quality reviews and rule updates.")
            
        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))
            recommendations.append("Unable to generate specific recommendations. Manual review required.")
        
        return recommendations
    
    def _create_empty_report(self, 
                           report_id: str,
                           database_name: str,
                           table_name: str,
                           report_timestamp: datetime) -> QualityReport:
        """Create empty quality report when no rules are found"""
        return QualityReport(
            report_id=report_id,
            database_name=database_name,
            table_name=table_name,
            report_timestamp=report_timestamp,
            overall_quality_score=1.0,
            dimension_scores={},
            total_rules_evaluated=0,
            rules_passed=0,
            rules_failed=0,
            alerts_generated=0,
            critical_issues=[],
            recommendations=["No quality rules configured for this table."]
        )
    
    async def _store_quality_report(self, report: QualityReport) -> None:
        """Store quality report in database"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO quality_reports 
                (report_id, database_name, table_name, overall_quality_score,
                 dimension_scores, total_rules_evaluated, rules_passed, rules_failed,
                 alerts_generated, critical_issues, recommendations)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                report.report_id, report.database_name, report.table_name,
                report.overall_quality_score, 
                json.dumps({k.value: v for k, v in report.dimension_scores.items()}),
                report.total_rules_evaluated, report.rules_passed, report.rules_failed,
                report.alerts_generated, report.critical_issues, report.recommendations
            )
            
        except Exception as e:
            logger.error("Failed to store quality report", error=str(e))
    
    async def _store_quality_metric(self, metric: QualityMetric) -> None:
        """Store quality metric in database"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO quality_metrics 
                (metric_id, rule_id, database_name, table_name, column_name,
                 dimension, measured_value, threshold, passed, sample_size, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                metric.metric_id, metric.rule_id, metric.database_name,
                metric.table_name, metric.column_name, metric.dimension.value,
                metric.measured_value, metric.threshold, metric.passed,
                metric.sample_size, json.dumps(metric.metadata)
            )
            
        except Exception as e:
            logger.error("Failed to store quality metric", error=str(e))
    
    async def _store_quality_alert(self, alert: QualityAlert) -> None:
        """Store quality alert in database"""
        try:
            await self.postgres_conn.execute("""
                INSERT INTO quality_alerts 
                (alert_id, rule_id, metric_id, severity, message, database_name,
                 table_name, column_name, measured_value, threshold, deviation)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
                alert.alert_id, alert.rule_id, alert.metric_id, alert.severity.value,
                alert.message, alert.database_name, alert.table_name,
                alert.column_name, alert.measured_value, alert.threshold, alert.deviation
            )
            
        except Exception as e:
            logger.error("Failed to store quality alert", error=str(e))
    
    def register_alert_handler(self, handler: Callable[[QualityAlert], None]) -> None:
        """Register alert handler for quality alerts"""
        self.alert_handlers.append(handler)
        logger.info("Registered quality alert handler")
    
    async def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """Get data for quality monitoring dashboard"""
        try:
            # Get recent quality scores
            recent_reports = await self.postgres_conn.fetch("""
                SELECT database_name, table_name, overall_quality_score, report_timestamp
                FROM quality_reports
                WHERE report_timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY report_timestamp DESC
            """)
            
            # Get active alerts
            active_alerts = await self.postgres_conn.fetch("""
                SELECT severity, COUNT(*) as count
                FROM quality_alerts
                WHERE alert_timestamp >= NOW() - INTERVAL '24 hours'
                AND resolved = FALSE
                GROUP BY severity
            """)
            
            # Get quality trends
            quality_trends = await self.postgres_conn.fetch("""
                SELECT 
                    DATE_TRUNC('hour', report_timestamp) as hour,
                    AVG(overall_quality_score) as avg_quality_score
                FROM quality_reports
                WHERE report_timestamp >= NOW() - INTERVAL '7 days'
                GROUP BY DATE_TRUNC('hour', report_timestamp)
                ORDER BY hour
            """)
            
            dashboard_data = {
                'recent_reports': [
                    {
                        'database': record['database_name'],
                        'table': record['table_name'],
                        'quality_score': record['overall_quality_score'],
                        'timestamp': record['report_timestamp'].isoformat()
                    }
                    for record in recent_reports
                ],
                'active_alerts': {
                    record['severity']: record['count']
                    for record in active_alerts
                },
                'quality_trends': [
                    {
                        'hour': record['hour'].isoformat(),
                        'avg_quality_score': float(record['avg_quality_score'])
                    }
                    for record in quality_trends
                ],
                'monitoring_stats': self.monitoring_stats,
                'dashboard_timestamp': datetime.utcnow().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to get dashboard data", error=str(e))
            return {}


# Example usage and testing
async def example_quality_monitoring():
    """Example of comprehensive data quality monitoring"""
    
    # Initialize quality monitor
    quality_monitor = DataQualityMonitor(
        postgres_connection_string='postgresql://prometheus:prometheus123@localhost:5432/prometheus',
        influxdb_url='http://localhost:8086',
        influxdb_token='prometheus-token',
        influxdb_org='prometheus',
        redis_url='redis://localhost:6379'
    )
    
    await quality_monitor.initialize()
    
    try:
        # Evaluate quality for seismic data
        seismic_report = await quality_monitor.evaluate_quality_rules(
            database_name='postgresql',
            table_name='seismic_events',
            time_window_hours=24
        )
        
        # Evaluate quality for deformation data
        deformation_report = await quality_monitor.evaluate_quality_rules(
            database_name='postgresql',
            table_name='mtsar_interferograms',
            time_window_hours=24
        )
        
        # Get dashboard data
        dashboard_data = await quality_monitor.get_quality_dashboard_data()
        
        logger.info("Quality monitoring example completed",
                   seismic_score=seismic_report.overall_quality_score,
                   deformation_score=deformation_report.overall_quality_score,
                   total_alerts=seismic_report.alerts_generated + deformation_report.alerts_generated)
        
    finally:
        await quality_monitor.close()


if __name__ == "__main__":
    asyncio.run(example_quality_monitoring())