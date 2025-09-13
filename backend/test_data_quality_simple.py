"""
Simple test for the Data Quality Validation Framework

This test verifies the core functionality without complex dependencies.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import our data quality modules directly
from core.data_quality.validation_framework import (
    DataQualityValidator, GeologicalObservation, DataType, QualityFlag
)
from core.data_quality.cross_source_validator import CrossSourceValidator
from core.data_quality.completeness_analyzer import CompletenessAnalyzer
from core.data_quality.quality_reporter import QualityReporter
from core.data_quality.integrated_validator import IntegratedDataQualityValidator, ValidationWorkflowConfig


async def test_basic_validation():
    """Test basic data quality validation"""
    print("Testing basic data quality validation...")
    
    # Create a validator
    validator = DataQualityValidator()
    
    # Create a test observation
    observation = GeologicalObservation(
        observation_id="test_001",
        timestamp=datetime.utcnow(),
        location=(40.7128, -74.0060, 0.0),  # New York coordinates
        data_type=DataType.SEISMIC,
        value=4.2,  # Magnitude
        uncertainty=0.1,
        source_system="IRIS"
    )
    
    # Validate the observation
    result = await validator.validate_observation(observation)
    
    print(f"✓ Validation completed")
    print(f"  - Valid: {result.is_valid}")
    print(f"  - Overall quality: {result.quality_metrics.overall_quality:.3f}")
    print(f"  - Completeness: {result.quality_metrics.completeness_score:.3f}")
    print(f"  - Accuracy: {result.quality_metrics.accuracy_score:.3f}")
    print(f"  - Quality flags: {[flag.value for flag in result.quality_flags]}")
    
    assert result.quality_metrics.overall_quality > 0.0
    assert len(result.quality_flags) > 0
    
    return True


async def test_cross_source_validation():
    """Test cross-source validation"""
    print("\nTesting cross-source validation...")
    
    validator = CrossSourceValidator()
    
    # Create target observation
    target = GeologicalObservation(
        observation_id="target_001",
        timestamp=datetime.utcnow(),
        location=(40.7128, -74.0060, 0.0),
        data_type=DataType.SEISMIC,
        value=4.2,
        uncertainty=0.1,
        source_system="IRIS"
    )
    
    # Create reference observations
    references = [
        GeologicalObservation(
            observation_id="ref_001",
            timestamp=datetime.utcnow() + timedelta(minutes=2),
            location=(40.7130, -74.0058, 0.0),
            data_type=DataType.SEISMIC,
            value=4.1,  # Close agreement
            uncertainty=0.1,
            source_system="USGS"
        ),
        GeologicalObservation(
            observation_id="ref_002",
            timestamp=datetime.utcnow() + timedelta(minutes=5),
            location=(40.7125, -74.0062, 0.0),
            data_type=DataType.SEISMIC,
            value=4.3,  # Close agreement
            uncertainty=0.1,
            source_system="local_station"
        )
    ]
    
    # Perform cross-source validation
    result = await validator.validate_cross_source_consistency(target, references)
    
    print(f"✓ Cross-source validation completed")
    print(f"  - Agreement score: {result.agreement_score:.3f}")
    print(f"  - Confidence level: {result.confidence_level:.3f}")
    print(f"  - Reference sources: {len(result.reference_observations)}")
    
    assert 0.0 <= result.agreement_score <= 1.0
    assert 0.0 <= result.confidence_level <= 1.0
    
    return True


async def test_completeness_analysis():
    """Test completeness analysis"""
    print("\nTesting completeness analysis...")
    
    analyzer = CompletenessAnalyzer()
    
    # Create observations with some gaps
    observations = []
    base_time = datetime.utcnow() - timedelta(hours=12)
    
    for i in range(12):
        if i not in [3, 4, 8]:  # Create gaps at hours 3-4 and 8
            obs = GeologicalObservation(
                observation_id=f"obs_{i:03d}",
                timestamp=base_time + timedelta(hours=i),
                location=(40.0 + i * 0.01, -120.0 + i * 0.01, 0.0),
                data_type=DataType.SEISMIC,
                value=3.0 + np.random.normal(0, 0.5),
                uncertainty=0.1,
                source_system="IRIS"
            )
            observations.append(obs)
    
    time_range = (base_time, base_time + timedelta(hours=12))
    
    # Analyze completeness
    report = await analyzer.analyze_completeness(observations, time_range)
    
    print(f"✓ Completeness analysis completed")
    print(f"  - Overall completeness: {report.overall_completeness:.3f}")
    print(f"  - Temporal completeness: {report.temporal_completeness:.3f}")
    print(f"  - Spatial completeness: {report.spatial_completeness:.3f}")
    print(f"  - Temporal gaps found: {len(report.temporal_gaps)}")
    print(f"  - Spatial gaps found: {len(report.spatial_gaps)}")
    
    assert 0.0 <= report.overall_completeness <= 1.0
    assert len(report.temporal_gaps) > 0  # Should detect the intentional gaps
    
    return True


async def test_quality_reporting():
    """Test quality reporting"""
    print("\nTesting quality reporting...")
    
    reporter = QualityReporter()
    
    # Create some mock validation results
    from core.data_quality.validation_framework import ValidationResult, QualityMetrics
    
    validation_results = []
    for i in range(5):
        quality_score = 0.9 - (i * 0.1)  # Decreasing quality
        
        metrics = QualityMetrics(
            completeness_score=quality_score,
            accuracy_score=quality_score,
            consistency_score=quality_score,
            timeliness_score=quality_score,
            uncertainty_level=1.0 - quality_score,
            anomaly_score=1.0 - quality_score,
            confidence_interval=(quality_score - 0.1, quality_score + 0.1)
        )
        
        result = ValidationResult(
            is_valid=quality_score > 0.5,
            quality_metrics=metrics,
            quality_flags=[QualityFlag.VALID if quality_score > 0.7 else QualityFlag.SUSPICIOUS],
            validation_timestamp=datetime.utcnow(),
            validation_method="test_validation",
            details={'observation_id': f'test_{i}', 'source_system': 'test_system'},
            recommendations=[]
        )
        
        validation_results.append(result)
    
    # Generate quality report
    report = await reporter.generate_quality_report(
        report_type='test',
        validation_results=validation_results
    )
    
    print(f"✓ Quality report generated")
    print(f"  - Overall quality score: {report.overall_quality_score:.3f}")
    print(f"  - Total observations: {report.total_observations}")
    print(f"  - Quality distribution: {report.quality_distribution}")
    print(f"  - Priority actions: {len(report.priority_actions)}")
    
    assert report.overall_quality_score >= 0.0
    assert report.total_observations == len(validation_results)
    
    return True


async def test_integrated_validation():
    """Test integrated validation system"""
    print("\nTesting integrated validation system...")
    
    config = ValidationWorkflowConfig(
        enable_cross_source_validation=True,
        enable_completeness_analysis=True,
        enable_automated_reporting=True
    )
    
    validator = IntegratedDataQualityValidator(config)
    
    # Create test observations
    observations = []
    base_time = datetime.utcnow()
    
    for i in range(3):
        obs = GeologicalObservation(
            observation_id=f"integrated_test_{i}",
            timestamp=base_time + timedelta(minutes=i * 10),
            location=(40.0 + i * 0.01, -120.0 + i * 0.01, 0.0),
            data_type=DataType.SEISMIC,
            value=3.0 + i * 0.2,
            uncertainty=0.1,
            source_system=f"source_{i % 2}"
        )
        observations.append(obs)
    
    # Test comprehensive validation
    target_obs = observations[0]
    reference_obs = observations[1:]
    
    result = await validator.validate_observation_comprehensive(
        target_obs, reference_obs
    )
    
    print(f"✓ Integrated validation completed")
    print(f"  - Overall confidence: {result.overall_confidence:.3f}")
    print(f"  - Processing time: {result.processing_time_ms:.1f}ms")
    print(f"  - Consolidated flags: {result.consolidated_flags}")
    print(f"  - Recommendations: {len(result.priority_recommendations)}")
    
    assert 0.0 <= result.overall_confidence <= 1.0
    assert result.processing_time_ms >= 0
    
    # Test batch validation
    batch_results = await validator.validate_batch_observations(observations)
    
    print(f"✓ Batch validation completed")
    print(f"  - Processed observations: {len(batch_results)}")
    
    assert len(batch_results) == len(observations)
    
    return True


async def run_all_tests():
    """Run all data quality validation tests"""
    print("=" * 80)
    print("DATA QUALITY VALIDATION FRAMEWORK TESTS")
    print("=" * 80)
    
    tests = [
        test_basic_validation,
        test_cross_source_validation,
        test_completeness_analysis,
        test_quality_reporting,
        test_integrated_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("🎉 All tests passed! Data Quality Validation Framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    # Run all tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)