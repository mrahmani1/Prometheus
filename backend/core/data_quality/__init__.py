"""
Data Quality Validation Framework

This package provides comprehensive data quality validation for geological observations,
including validation algorithms, cross-source validation, completeness analysis,
and automated quality reporting.
"""

from .validation_framework import (
    DataQualityValidator,
    GeologicalObservation,
    ValidationResult,
    QualityMetrics,
    QualityFlag,
    DataType,
    ProvenanceRecord
)

from .cross_source_validator import (
    CrossSourceValidator,
    CrossSourceComparison,
    ConsensusResult
)

from .completeness_analyzer import (
    CompletenessAnalyzer,
    CompletenessReport,
    TemporalGap,
    SpatialGap,
    DataAvailabilityMetrics
)

from .quality_reporter import (
    QualityReporter,
    QualityAlert,
    QualityReport,
    AlertSeverity,
    AlertType
)

__all__ = [
    # Main validation framework
    'DataQualityValidator',
    'GeologicalObservation',
    'ValidationResult',
    'QualityMetrics',
    'QualityFlag',
    'DataType',
    'ProvenanceRecord',
    
    # Cross-source validation
    'CrossSourceValidator',
    'CrossSourceComparison',
    'ConsensusResult',
    
    # Completeness analysis
    'CompletenessAnalyzer',
    'CompletenessReport',
    'TemporalGap',
    'SpatialGap',
    'DataAvailabilityMetrics',
    
    # Quality reporting
    'QualityReporter',
    'QualityAlert',
    'QualityReport',
    'AlertSeverity',
    'AlertType'
]

__version__ = '1.0.0'