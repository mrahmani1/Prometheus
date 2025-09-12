"""
Disaster Recovery and Business Continuity Data Models
Comprehensive data models for disaster recovery operations, backup management,
replication configuration, and business continuity planning.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import uuid

class BackupType(Enum):
    """Types of backup operations"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    TRANSACTION_LOG = "transaction_log"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VERIFYING = "verifying"
    VERIFIED = "verified"

class ReplicationStatus(Enum):
    """Replication status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SYNCING = "syncing"
    FAILED = "failed"
    PAUSED = "paused"
    DEGRADED = "degraded"

class RecoveryStatus(Enum):
    """Recovery operation status"""
    PLANNED = "planned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    TESTING = "testing"

class DisasterType(Enum):
    """Types of disasters"""
    HARDWARE_FAILURE = "hardware_failure"
    NETWORK_OUTAGE = "network_outage"
    DATA_CORRUPTION = "data_corruption"
    CYBER_ATTACK = "cyber_attack"
    NATURAL_DISASTER = "natural_disaster"
    HUMAN_ERROR = "human_error"
    SOFTWARE_FAILURE = "software_failure"

class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    INFLUXDB = "influxdb"
    NEO4J = "neo4j"
    REDIS = "redis"
    KAFKA = "kafka"

@dataclass
class BackupConfiguration:
    """Backup configuration settings"""
    database_type: DatabaseType
    backup_type: BackupType
    schedule_cron: str
    retention_days: int
    compression_enabled: bool = True
    encryption_enabled: bool = True
    verification_enabled: bool = True
    storage_locations: List[str] = field(default_factory=list)
    max_parallel_backups: int = 3
    backup_window_hours: List[int] = field(default_factory=lambda: [2, 3, 4])  # 2-4 AM
    
    def __post_init__(self):
        if not self.storage_locations:
            self.storage_locations = ["local", "s3", "gcs"]

@dataclass
class BackupMetadata:
    """Backup metadata information"""
    backup_id: str
    database_name: str
    database_type: DatabaseType
    backup_type: BackupType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    size_bytes: Optional[int] = None
    compressed_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    storage_location: Optional[str] = None
    encryption_key_id: Optional[str] = None
    parent_backup_id: Optional[str] = None  # For incremental backups
    error_message: Optional[str] = None
    verification_status: Optional[str] = None
    
    def __post_init__(self):
        if not self.backup_id:
            self.backup_id = str(uuid.uuid4())

@dataclass
class ReplicationConfiguration:
    """Cross-region replication configuration"""
    replication_id: str
    source_region: str
    target_regions: List[str]
    database_type: DatabaseType
    replication_mode: str  # "async", "sync", "semi_sync"
    lag_threshold_seconds: int = 300  # 5 minutes
    auto_failover_enabled: bool = False
    conflict_resolution_strategy: str = "source_wins"
    bandwidth_limit_mbps: Optional[int] = None
    compression_enabled: bool = True
    encryption_enabled: bool = True
    
    def __post_init__(self):
        if not self.replication_id:
            self.replication_id = str(uuid.uuid4())

@dataclass
class ReplicationStatus:
    """Replication status information"""
    replication_id: str
    source_region: str
    target_region: str
    status: ReplicationStatus
    last_sync_time: Optional[datetime] = None
    lag_seconds: Optional[int] = None
    bytes_replicated: Optional[int] = None
    error_count: int = 0
    last_error: Optional[str] = None
    health_score: float = 1.0  # 0.0 to 1.0

@dataclass
class RecoveryObjective:
    """Recovery Time and Point Objectives"""
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    service_tier: str  # "critical", "important", "standard"
    max_data_loss_minutes: int = 0
    
    def __post_init__(self):
        if self.max_data_loss_minutes == 0:
            self.max_data_loss_minutes = self.rpo_minutes

@dataclass
class RecoveryPlan:
    """Disaster recovery plan"""
    plan_id: str
    name: str
    description: str
    disaster_types: List[DisasterType]
    affected_services: List[str]
    recovery_steps: List[Dict[str, Any]]
    recovery_objectives: RecoveryObjective
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_minutes: int = 60
    last_tested: Optional[datetime] = None
    test_results: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())

@dataclass
class RecoveryExecution:
    """Recovery plan execution tracking"""
    execution_id: str
    plan_id: str
    disaster_type: DisasterType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: RecoveryStatus = RecoveryStatus.PLANNED
    current_step: int = 0
    total_steps: int = 0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    rollback_required: bool = False
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())

@dataclass
class BusinessContinuityPlan:
    """Business continuity planning"""
    plan_id: str
    name: str
    description: str
    business_functions: List[str]
    critical_dependencies: List[str]
    alternative_procedures: List[Dict[str, Any]]
    communication_plan: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    recovery_priorities: List[str]
    testing_schedule: str
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.plan_id:
            self.plan_id = str(uuid.uuid4())

@dataclass
class DisasterRecoveryMetrics:
    """Disaster recovery monitoring metrics"""
    timestamp: datetime
    rto_actual_minutes: Optional[int] = None
    rpo_actual_minutes: Optional[int] = None
    backup_success_rate: float = 0.0
    replication_health_score: float = 0.0
    recovery_test_success_rate: float = 0.0
    data_integrity_score: float = 0.0
    system_availability: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

@dataclass
class BackupStrategy:
    """Comprehensive backup strategy"""
    strategy_id: str
    name: str
    databases: List[DatabaseType]
    backup_configurations: List[BackupConfiguration]
    retention_policy: Dict[str, int]  # backup_type -> retention_days
    storage_strategy: Dict[str, Any]
    verification_strategy: Dict[str, Any]
    encryption_strategy: Dict[str, Any]
    
    def __post_init__(self):
        if not self.strategy_id:
            self.strategy_id = str(uuid.uuid4())

@dataclass
class FailoverConfiguration:
    """Automated failover configuration"""
    config_id: str
    trigger_conditions: List[Dict[str, Any]]
    failover_sequence: List[Dict[str, Any]]
    rollback_sequence: List[Dict[str, Any]]
    notification_settings: Dict[str, Any]
    auto_failback_enabled: bool = False
    failback_conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.config_id:
            self.config_id = str(uuid.uuid4())

@dataclass
class DataArchivalPolicy:
    """Data archival and retention policy"""
    policy_id: str
    name: str
    database_type: DatabaseType
    archival_rules: List[Dict[str, Any]]
    retention_periods: Dict[str, int]  # data_type -> retention_days
    storage_tiers: List[str]  # ["hot", "warm", "cold", "archive"]
    compression_settings: Dict[str, Any]
    encryption_settings: Dict[str, Any]
    
    def __post_init__(self):
        if not self.policy_id:
            self.policy_id = str(uuid.uuid4())

@dataclass
class IntegrityCheckResult:
    """Backup integrity check result"""
    check_id: str
    backup_id: str
    check_type: str  # "checksum", "restore_test", "data_validation"
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # "pending", "running", "passed", "failed"
    issues_found: List[Dict[str, Any]] = field(default_factory=list)
    repair_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.check_id:
            self.check_id = str(uuid.uuid4())

@dataclass
class GeoDistributedStorage:
    """Geo-distributed backup storage configuration"""
    storage_id: str
    primary_location: str
    replica_locations: List[str]
    storage_type: str  # "object_storage", "block_storage", "file_storage"
    replication_strategy: str  # "sync", "async", "eventual_consistency"
    consistency_level: str  # "strong", "eventual", "weak"
    access_patterns: Dict[str, Any]
    cost_optimization: Dict[str, Any]
    
    def __post_init__(self):
        if not self.storage_id:
            self.storage_id = str(uuid.uuid4())

@dataclass
class DisasterRecoveryTest:
    """Disaster recovery test execution"""
    test_id: str
    test_name: str
    test_type: str  # "backup_restore", "failover", "full_dr", "tabletop"
    scope: List[str]  # Services/databases to test
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "planned"  # "planned", "running", "completed", "failed"
    test_results: Dict[str, Any] = field(default_factory=dict)
    issues_identified: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.test_id:
            self.test_id = str(uuid.uuid4())

# Utility functions for model validation and conversion

def validate_backup_configuration(config: BackupConfiguration) -> bool:
    """Validate backup configuration"""
    if config.retention_days <= 0:
        return False
    if not config.schedule_cron:
        return False
    if config.max_parallel_backups <= 0:
        return False
    return True

def validate_recovery_objectives(objectives: RecoveryObjective) -> bool:
    """Validate recovery objectives"""
    if objectives.rto_minutes <= 0 or objectives.rpo_minutes < 0:
        return False
    if objectives.rpo_minutes > objectives.rto_minutes:
        return False  # RPO cannot be greater than RTO
    return True

def calculate_backup_retention_cost(
    backup_size_gb: float,
    retention_days: int,
    storage_cost_per_gb_month: float
) -> float:
    """Calculate backup retention cost"""
    monthly_cost = backup_size_gb * storage_cost_per_gb_month
    return monthly_cost * (retention_days / 30.0)

def estimate_recovery_time(
    data_size_gb: float,
    network_bandwidth_mbps: float,
    processing_overhead_factor: float = 1.5
) -> int:
    """Estimate recovery time in minutes"""
    transfer_time_seconds = (data_size_gb * 8 * 1024) / network_bandwidth_mbps
    total_time_seconds = transfer_time_seconds * processing_overhead_factor
    return int(total_time_seconds / 60)