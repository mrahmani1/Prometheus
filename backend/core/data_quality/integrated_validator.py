"""
Integrated Data Quality Validation System

This module provides a unified interface for all data quality validation components,
orchestrating validation workflows, cross-source validation, completeness analysis,
and automated reporting.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .validation_framework import (
    DataQualityValidator, GeologicalObservation, ValidationResult, 
    QualityMetrics, DataType
)
from .cross_source_validator import CrossSourceValidator, CrossSourceComparison
from .completeness_analyzer import CompletenessAnalyzer, CompletenessReport
from .quality_reporter import QualityReporter, QualityReport, QualityAlert

logger = logging.getLogger(__name__)


@dataclass
class ValidationWorkflowConfig:
    """Configuration for validation workflow"""
    enable_cross_source_validation: bool = True
    enable_completeness_analysis: bool = True
    enable_automated_reporting: bool = True
    enable_real_time_monitoring: bool = True
    
    # Validation parameters
    cross_source_min_references: int = 2
    completeness_analysis_window_hours: int = 24
    quality_report_frequency_hours: int = 24
    
    # Alert thresholds
    critical_quality_threshold: float = 0.5
    warning_quality_threshold: float = 0.7
    
    # Performance settings
    batch_size: int = 1000
    max_concurrent_validations: int = 10


@dataclass
class IntegratedValidationResult:
    """Comprehensive validation result combining all validation components"""
    observation: GeologicalObservation
    primary_validation: ValidationResult
    cross_source_validation: Optional[CrossSourceComparison]
    completeness_context: Dict[str, Any]
    
    # Aggregated metrics
    overall_confidence: float
    validation_timestamp: datetime
    processing_time_ms: float
    
    # Quality flags and recommendations
    consolidated_flags: List[str]
    priority_recommendations: List[str]


class IntegratedDataQualityValidator:
    """
    Integrated data quality validation system that orchestrates all validation components
    """
    
    def __init__(self, config: Optional[ValidationWorkflowConfig] = None):
        self.config = config or ValidationWorkflowConfig()
        
        # Initialize component validators
        self.primary_validator = DataQualityValidator()
        self.cross_source_validator = CrossSourceValidator()
        self.completeness_analyzer = CompletenessAnalyzer()
        self.quality_reporter = QualityReporter()
        
        # Internal state
        self.validation_cache = {}
        self.reference_data_cache = {}
        self.monitoring_active = False
        
        logger.info("Integrated Data Quality Validator initialized")
    
    async def validate_observation_comprehensive(
        self,
        observation: GeologicalObservation,
        reference_observations: Optional[List[GeologicalObservation]] = None,
        include_completeness_context: bool = True
    ) -> IntegratedValidationResult:
        """
        Perform comprehensive validation of a geological observation using all available methods
        
        Args:
            observation: The observation to validate
            reference_observations: Optional reference observations for cross-validation
            include_completeness_context: Whether to include completeness analysis context
            
        Returns:
            IntegratedValidationResult with comprehensive validation information
        """
        start_time = datetime.utcnow()
        
        logger.info(f"Starting comprehensive validation for observation {observation.observation_id}")
        
        # Step 1: Primary validation
        primary_validation = await self.primary_validator.validate_observation(
            observation, reference_observations
        )
        
        # Step 2: Cross-source validation (if enabled and references available)
        cross_source_validation = None
        if (self.config.enable_cross_source_validation and 
            reference_observations and 
            len(reference_observations) >= self.config.cross_source_min_references):
            
            cross_source_validation = await self.cross_source_validator.validate_cross_source_consistency(
                observation, reference_observations
            )
        
        # Step 3: Completeness context analysis (if enabled)
        completeness_context = {}
        if self.config.enable_completeness_analysis and include_completeness_context:
            completeness_context = await self._analyze_completeness_context(observation)
        
        # Step 4: Consolidate results
        overall_confidence = self._calculate_overall_confidence(
            primary_validation, cross_source_validation, completeness_context
        )
        
        consolidated_flags = self._consolidate_quality_flags(
            primary_validation, cross_source_validation
        )
        
        priority_recommendations = self._generate_priority_recommendations(
            primary_validation, cross_source_validation, completeness_context
        )
        
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        result = IntegratedValidationResult(
            observation=observation,
            primary_validation=primary_validation,
            cross_source_validation=cross_source_validation,
            completeness_context=completeness_context,
            overall_confidence=overall_confidence,
            validation_timestamp=datetime.utcnow(),
            processing_time_ms=processing_time,
            consolidated_flags=consolidated_flags,
            priority_recommendations=priority_recommendations
        )
        
        logger.info(f"Comprehensive validation completed in {processing_time:.1f}ms")
        return result
    
    async def validate_batch_observations(
        self,
        observations: List[GeologicalObservation],
        reference_data: Optional[Dict[str, List[GeologicalObservation]]] = None
    ) -> List[IntegratedValidationResult]:
        """
        Validate a batch of observations efficiently using parallel processing
        
        Args:
            observations: List of observations to validate
            reference_data: Optional dictionary mapping observation IDs to reference data
            
        Returns:
            List of IntegratedValidationResult objects
        """
        logger.info(f"Starting batch validation of {len(observations)} observations")
        
        # Process in batches to manage memory and concurrency
        batch_size = self.config.batch_size
        max_concurrent = self.config.max_concurrent_validations
        
        results = []
        
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i + batch_size]
            
            # Create semaphore to limit concurrent validations
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def validate_with_semaphore(obs):
                async with semaphore:
                    refs = reference_data.get(obs.observation_id, []) if reference_data else []
                    return await self.validate_observation_comprehensive(obs, refs)
            
            # Process batch concurrently
            batch_tasks = [validate_with_semaphore(obs) for obs in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Validation failed for observation {batch[j].observation_id}: {result}")
                    # Create a minimal result for failed validations
                    result = self._create_failed_validation_result(batch[j], str(result))
                
                results.append(result)
        
        logger.info(f"Batch validation completed: {len(results)} results")
        return results
    
    async def analyze_dataset_completeness(
        self,
        observations: List[GeologicalObservation],
        time_range: Optional[Tuple[datetime, datetime]] = None,
        spatial_bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> CompletenessReport:
        """
        Analyze completeness of an entire dataset
        
        Args:
            observations: List of observations to analyze
            time_range: Optional time range for analysis
            spatial_bounds: Optional spatial bounds for analysis
            
        Returns:
            CompletenessReport with detailed analysis
        """
        logger.info(f"Analyzing completeness for dataset with {len(observations)} observations")
        
        if time_range is None and observations:
            # Determine time range from observations
            timestamps = [obs.timestamp for obs in observations]
            time_range = (min(timestamps), max(timestamps))
        
        return await self.completeness_analyzer.analyze_completeness(
            observations, time_range, spatial_bounds
        )
    
    async def generate_quality_dashboard_report(
        self,
        observations: List[GeologicalObservation],
        validation_results: Optional[List[IntegratedValidationResult]] = None,
        report_type: str = 'comprehensive'
    ) -> QualityReport:
        """
        Generate a comprehensive quality report for dashboard display
        
        Args:
            observations: List of observations to include in report
            validation_results: Optional pre-computed validation results
            report_type: Type of report to generate
            
        Returns:
            QualityReport suitable for dashboard display
        """
        logger.info(f"Generating quality dashboard report for {len(observations)} observations")
        
        # If validation results not provided, compute them
        if validation_results is None:
            validation_results = await self.validate_batch_observations(observations)
        
        # Extract primary validation results for reporting
        primary_validations = [r.primary_validation for r in validation_results if r.primary_validation]
        
        # Generate completeness analysis
        completeness_analysis = None
        if self.config.enable_completeness_analysis:
            completeness_analysis = await self.analyze_dataset_completeness(observations)
        
        # Generate quality report
        report = await self.quality_reporter.generate_quality_report(
            report_type=report_type,
            validation_results=primary_validations,
            completeness_analysis=completeness_analysis
        )
        
        # Add integrated validation metrics
        report = self._enhance_report_with_integrated_metrics(report, validation_results)
        
        return report
    
    async def start_real_time_monitoring(
        self,
        data_stream_callback: callable,
        alert_callback: Optional[callable] = None
    ):
        """
        Start real-time quality monitoring of data streams
        
        Args:
            data_stream_callback: Callback function that returns new observations
            alert_callback: Optional callback for handling quality alerts
        """
        if self.monitoring_active:
            logger.warning("Real-time monitoring already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting real-time quality monitoring")
        
        async def monitoring_loop():
            while self.monitoring_active:
                try:
                    # Get new observations from stream
                    new_observations = await data_stream_callback()
                    
                    if new_observations:
                        # Validate new observations
                        validation_results = await self.validate_batch_observations(new_observations)
                        
                        # Check for quality issues
                        alerts = self._check_for_quality_alerts(validation_results)
                        
                        # Handle alerts
                        for alert in alerts:
                            await self.quality_reporter._send_alert_notification(alert)
                            if alert_callback:
                                await alert_callback(alert)
                    
                    # Wait before next check
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in real-time monitoring: {e}")
                    await asyncio.sleep(60)  # Wait before retrying
        
        # Start monitoring in background
        asyncio.create_task(monitoring_loop())
    
    def stop_real_time_monitoring(self):
        """Stop real-time quality monitoring"""
        self.monitoring_active = False
        logger.info("Real-time quality monitoring stopped")
    
    async def _analyze_completeness_context(
        self, 
        observation: GeologicalObservation
    ) -> Dict[str, Any]:
        """Analyze completeness context for a single observation"""
        # This would typically analyze the observation in the context of expected data patterns
        # For now, we'll provide a simplified implementation
        
        context = {
            'temporal_context': {
                'expected_frequency': self.completeness_analyzer.expected_frequencies.get(
                    observation.data_type, timedelta(hours=1)
                ).total_seconds() / 3600,  # Convert to hours
                'observation_age_hours': (datetime.utcnow() - observation.timestamp).total_seconds() / 3600
            },
            'spatial_context': {
                'location': observation.location,
                'in_critical_region': self._is_in_critical_region(observation.location)
            },
            'source_context': {
                'source_system': observation.source_system,
                'data_type': observation.data_type.value
            }
        }
        
        return context
    
    def _is_in_critical_region(self, location: Tuple[float, float, float]) -> bool:
        """Check if location is in a critical monitoring region"""
        lat, lon, _ = location
        
        # Check against critical regions from completeness analyzer
        for region in self.completeness_analyzer.critical_regions:
            bounds = region['bounds']
            if (bounds[0] <= lat <= bounds[1] and 
                bounds[2] <= lon <= bounds[3]):
                return True
        
        return False
    
    def _calculate_overall_confidence(
        self,
        primary_validation: ValidationResult,
        cross_source_validation: Optional[CrossSourceComparison],
        completeness_context: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score combining all validation results"""
        
        # Start with primary validation confidence
        base_confidence = primary_validation.quality_metrics.overall_quality
        
        # Adjust based on cross-source validation
        if cross_source_validation:
            cross_source_weight = 0.3
            cross_source_confidence = cross_source_validation.confidence_level
            base_confidence = (base_confidence * 0.7 + cross_source_confidence * cross_source_weight)
        
        # Adjust based on completeness context
        if completeness_context:
            # Reduce confidence for very old observations
            age_hours = completeness_context.get('temporal_context', {}).get('observation_age_hours', 0)
            if age_hours > 24:  # Older than 24 hours
                age_penalty = min(0.2, age_hours / 168)  # Up to 20% penalty over a week
                base_confidence *= (1.0 - age_penalty)
            
            # Increase confidence for observations in critical regions
            if completeness_context.get('spatial_context', {}).get('in_critical_region', False):
                base_confidence = min(1.0, base_confidence * 1.1)  # 10% bonus
        
        return max(0.0, min(1.0, base_confidence))
    
    def _consolidate_quality_flags(
        self,
        primary_validation: ValidationResult,
        cross_source_validation: Optional[CrossSourceComparison]
    ) -> List[str]:
        """Consolidate quality flags from all validation sources"""
        flags = [flag.value for flag in primary_validation.quality_flags]
        
        if cross_source_validation:
            if cross_source_validation.agreement_score < 0.7:
                flags.append('cross_source_disagreement')
            
            if cross_source_validation.disagreement_details:
                flags.append('source_inconsistency')
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(flags))
    
    def _generate_priority_recommendations(
        self,
        primary_validation: ValidationResult,
        cross_source_validation: Optional[CrossSourceComparison],
        completeness_context: Dict[str, Any]
    ) -> List[str]:
        """Generate priority recommendations based on all validation results"""
        recommendations = []
        
        # Add primary validation recommendations
        recommendations.extend(primary_validation.recommendations)
        
        # Add cross-source validation recommendations
        if cross_source_validation and cross_source_validation.agreement_score < 0.7:
            recommendations.append("Investigate cross-source disagreements")
        
        # Add completeness-based recommendations
        if completeness_context:
            age_hours = completeness_context.get('temporal_context', {}).get('observation_age_hours', 0)
            if age_hours > 48:  # Very old observation
                recommendations.append("Consider data freshness - observation is over 48 hours old")
        
        # Remove duplicates and prioritize
        unique_recommendations = list(dict.fromkeys(recommendations))
        
        # Sort by priority (critical issues first)
        priority_keywords = ['critical', 'immediate', 'urgent', 'investigate']
        prioritized = []
        
        for keyword in priority_keywords:
            for rec in unique_recommendations:
                if keyword.lower() in rec.lower() and rec not in prioritized:
                    prioritized.append(rec)
        
        # Add remaining recommendations
        for rec in unique_recommendations:
            if rec not in prioritized:
                prioritized.append(rec)
        
        return prioritized[:5]  # Return top 5 recommendations
    
    def _create_failed_validation_result(
        self, 
        observation: GeologicalObservation, 
        error_message: str
    ) -> IntegratedValidationResult:
        """Create a minimal validation result for failed validations"""
        
        # Create minimal quality metrics
        failed_metrics = QualityMetrics(
            completeness_score=0.0,
            accuracy_score=0.0,
            consistency_score=0.0,
            timeliness_score=0.0,
            uncertainty_level=1.0,
            anomaly_score=1.0,
            confidence_interval=(0.0, 0.0)
        )
        
        # Create failed validation result
        failed_validation = ValidationResult(
            is_valid=False,
            quality_metrics=failed_metrics,
            quality_flags=[],
            validation_timestamp=datetime.utcnow(),
            validation_method="failed_validation",
            details={'error': error_message},
            recommendations=[f"Validation failed: {error_message}"]
        )
        
        return IntegratedValidationResult(
            observation=observation,
            primary_validation=failed_validation,
            cross_source_validation=None,
            completeness_context={},
            overall_confidence=0.0,
            validation_timestamp=datetime.utcnow(),
            processing_time_ms=0.0,
            consolidated_flags=['validation_failed'],
            priority_recommendations=[f"Fix validation error: {error_message}"]
        )
    
    def _check_for_quality_alerts(
        self, 
        validation_results: List[IntegratedValidationResult]
    ) -> List[QualityAlert]:
        """Check validation results for conditions that should trigger alerts"""
        alerts = []
        
        # Check for low quality scores
        low_quality_count = sum(
            1 for result in validation_results 
            if result.overall_confidence < self.config.critical_quality_threshold
        )
        
        if low_quality_count > len(validation_results) * 0.2:  # More than 20% low quality
            alert = QualityAlert(
                alert_id=f"batch_quality_degradation_{datetime.utcnow().isoformat()}",
                alert_type=self.quality_reporter.AlertType.DATA_QUALITY_DEGRADATION,
                severity=self.quality_reporter.AlertSeverity.WARNING,
                timestamp=datetime.utcnow(),
                title="Batch Quality Degradation Detected",
                description=f"{low_quality_count} out of {len(validation_results)} observations have low quality scores",
                affected_systems=[],
                data_type=None,
                location=None,
                metrics={'low_quality_percentage': low_quality_count / len(validation_results)},
                recommendations=["Investigate data sources for systematic issues"]
            )
            alerts.append(alert)
        
        return alerts
    
    def _enhance_report_with_integrated_metrics(
        self, 
        report: QualityReport, 
        validation_results: List[IntegratedValidationResult]
    ) -> QualityReport:
        """Enhance quality report with integrated validation metrics"""
        
        if validation_results:
            # Calculate integrated metrics
            avg_confidence = np.mean([r.overall_confidence for r in validation_results])
            avg_processing_time = np.mean([r.processing_time_ms for r in validation_results])
            
            cross_source_validations = [
                r.cross_source_validation for r in validation_results 
                if r.cross_source_validation is not None
            ]
            
            cross_source_coverage = len(cross_source_validations) / len(validation_results)
            
            # Add to report details
            if not hasattr(report, 'integrated_metrics'):
                report.integrated_metrics = {}
            
            report.integrated_metrics.update({
                'average_confidence': avg_confidence,
                'average_processing_time_ms': avg_processing_time,
                'cross_source_validation_coverage': cross_source_coverage,
                'total_integrated_validations': len(validation_results)
            })
        
        return report