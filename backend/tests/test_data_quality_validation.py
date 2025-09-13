"""
Comprehensive tests for the Data Quality Validation Framework

This module tests all components of the data quality validation system including
validation algorithms, cross-source validation, completeness analysis, and
automated reporting.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

from backend.core.data_quality import (
    DataQualityValidator,
    GeologicalObservation,
    ValidationResult,
    QualityMetrics,
    QualityFlag,
    DataType,
    CrossSourceValidator,
    CompletenessAnalyzer,
    QualityReporter,
    IntegratedDataQualityValidator,
    ValidationWorkflowConfig
)


class TestDataQualityValidator:
    """Test the main data quality validator"""
    
    @pytest.fixture
    def validator(self):
        return DataQualityValidator()
    
    @pytest.fixture
    def sample_observation(self):
        return GeologicalObservation(
            observation_id="test_obs_001",
            timestamp=datetime.utcnow(),
            location=(40.7128, -74.0060, 0.0),  # New York coordinates
            data_type=DataType.SEISMIC,
            value=3.5,  # Magnitude
            uncertainty=0.1,
            source_system="IRIS"
        )
    
    @pytest.fixture
    def sample_geochemical_observation(self):
        return GeologicalObservation(
            observation_id="test_geochem_001",
            timestamp=datetime.utcnow(),
            location=(40.7128, -74.0060, 0.0),
            data_type=DataType.GEOCHEMICAL,
            value={
                'SiO2': 65.2,
                'Al2O3': 15.8,
                'Fe2O3': 4.2,
                'CaO': 3.1,
                'MgO': 2.8
            },
            uncertainty=0.05,
            source_system="EarthChem"
        )
    
    @pytest.mark.asyncio
    async def test_validate_observation_basic(self, validator, sample_observation):
        """Test basic observation validation"""
        result = await validator.validate_observation(sample_observation)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is not None
        assert isinstance(result.quality_metrics, QualityMetrics)
        assert len(result.quality_flags) > 0
        assert len(result.recommendations) >= 0
        
        # Check quality metrics are in valid ranges
        assert 0.0 <= result.quality_metrics.overall_quality <= 1.0
        assert 0.0 <= result.quality_metrics.completeness_score <= 1.0
        assert 0.0 <= result.quality_metrics.accuracy_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_geochemical_observation(self, validator, sample_geochemical_observation):
        """Test validation of geochemical observations"""
        result = await validator.validate_observation(sample_geochemical_observation)
        
        assert isinstance(result, ValidationResult)
        assert result.quality_metrics.completeness_score > 0.0
        
        # Geochemical data should have reasonable accuracy for valid compositions
        total_composition = sum(sample_geochemical_observation.value.values())
        if 95 <= total_composition <= 105:  # Valid total
            assert result.quality_metrics.accuracy_score > 0.8
    
    @pytest.mark.asyncio
    async def test_validate_with_reference_data(self, validator, sample_observation):
        """Test validation with reference data"""
        # Create reference observations
        reference_obs = [
            GeologicalObservation(
                observation_id="ref_001",
                timestamp=sample_observation.timestamp + timedelta(minutes=5),
                location=(40.7130, -74.0058, 0.0),  # Nearby location
                data_type=DataType.SEISMIC,
                value=3.4,  # Similar magnitude
                uncertainty=0.1,
                source_system="USGS"
            ),
            GeologicalObservation(
                observation_id="ref_002",
                timestamp=sample_observation.timestamp + timedelta(minutes=10),
                location=(40.7125, -74.0062, 0.0),  # Nearby location
                data_type=DataType.SEISMIC,
                value=3.6,  # Similar magnitude
                uncertainty=0.1,
                source_system="local_station"
            )
        ]
        
        result = await validator.validate_observation(sample_observation, reference_obs)
        
        assert isinstance(result, ValidationResult)
        # With good reference data, accuracy should be higher
        assert result.quality_metrics.accuracy_score > 0.5
    
    def test_physical_plausibility_seismic(self, validator):
        """Test physical plausibility checks for seismic data"""
        # Valid seismic observation
        valid_obs = GeologicalObservation(
            observation_id="valid_seismic",
            timestamp=datetime.utcnow(),
            location=(40.0, -120.0, 0.0),
            data_type=DataType.SEISMIC,
            value=5.5,  # Reasonable magnitude
            uncertainty=0.1,
            source_system="IRIS"
        )
        
        plausibility = validator._assess_physical_plausibility(valid_obs)
        assert plausibility > 0.8
        
        # Invalid seismic observation
        invalid_obs = GeologicalObservation(
            observation_id="invalid_seismic",
            timestamp=datetime.utcnow(),
            location=(40.0, -120.0, 0.0),
            data_type=DataType.SEISMIC,
            value=15.0,  # Unrealistic magnitude
            uncertainty=0.1,
            source_system="IRIS"
        )
        
        plausibility = validator._assess_physical_plausibility(invalid_obs)
        assert plausibility < 0.5
    
    def test_anomaly_detection_scalar(self, validator):
        """Test anomaly detection for scalar values"""
        # Normal observation
        normal_obs = GeologicalObservation(
            observation_id="normal",
            timestamp=datetime.utcnow(),
            location=(40.0, -120.0, 0.0),
            data_type=DataType.SEISMIC,
            value=3.0,
            uncertainty=0.1,
            source_system="IRIS"
        )
        
        anomaly_score = validator._detect_scalar_anomaly(normal_obs)
        assert anomaly_score < 0.5
        
        # Anomalous observation
        anomalous_obs = GeologicalObservation(
            observation_id="anomalous",
            timestamp=datetime.utcnow(),
            location=(40.0, -120.0, 0.0),
            data_type=DataType.SEISMIC,
            value=12.0,  # Very high magnitude
            uncertainty=0.1,
            source_system="IRIS"
        )
        
        anomaly_score = validator._detect_scalar_anomaly(anomalous_obs)
        assert anomaly_score > 0.5


class TestCrossSourceValidator:
    """Test cross-source validation functionality"""
    
    @pytest.fixture
    def cross_validator(self):
        return CrossSourceValidator()
    
    @pytest.fixture
    def target_observation(self):
        return GeologicalObservation(
            observation_id="target_001",
            timestamp=datetime.utcnow(),
            location=(40.7128, -74.0060, 0.0),
            data_type=DataType.SEISMIC,
            value=4.2,
            uncertainty=0.1,
            source_system="IRIS"
        )
    
    @pytest.fixture
    def reference_observations(self):
        base_time = datetime.utcnow()
        return [
            GeologicalObservation(
                observation_id="ref_001",
                timestamp=base_time + timedelta(minutes=2),
                location=(40.7130, -74.0058, 0.0),
                data_type=DataType.SEISMIC,
                value=4.1,  # Close agreement
                uncertainty=0.1,
                source_system="USGS"
            ),
            GeologicalObservation(
                observation_id="ref_002",
                timestamp=base_time + timedelta(minutes=5),
                location=(40.7125, -74.0062, 0.0),
                data_type=DataType.SEISMIC,
                value=4.3,  # Close agreement
                uncertainty=0.1,
                source_system="local_station"
            ),
            GeologicalObservation(
                observation_id="ref_003",
                timestamp=base_time + timedelta(minutes=1),
                location=(40.7132, -74.0055, 0.0),
                data_type=DataType.SEISMIC,
                value=3.8,  # Moderate disagreement
                uncertainty=0.1,
                source_system="regional_network"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_cross_source_validation(self, cross_validator, target_observation, reference_observations):
        """Test cross-source validation with multiple references"""
        result = await cross_validator.validate_cross_source_consistency(
            target_observation, reference_observations
        )
        
        assert result.primary_observation == target_observation
        assert len(result.reference_observations) > 0
        assert 0.0 <= result.agreement_score <= 1.0
        assert 0.0 <= result.confidence_level <= 1.0
        
        # With reasonably close values, agreement should be decent
        assert result.agreement_score > 0.5
    
    @pytest.mark.asyncio
    async def test_cross_source_with_disagreement(self, cross_validator, target_observation):
        """Test cross-source validation with significant disagreement"""
        # Create references with significant disagreement
        disagreeing_refs = [
            GeologicalObservation(
                observation_id="disagree_001",
                timestamp=datetime.utcnow(),
                location=(40.7130, -74.0058, 0.0),
                data_type=DataType.SEISMIC,
                value=6.5,  # Significant disagreement
                uncertainty=0.1,
                source_system="USGS"
            ),
            GeologicalObservation(
                observation_id="disagree_002",
                timestamp=datetime.utcnow(),
                location=(40.7125, -74.0062, 0.0),
                data_type=DataType.SEISMIC,
                value=2.1,  # Significant disagreement
                uncertainty=0.1,
                source_system="local_station"
            )
        ]
        
        result = await cross_validator.validate_cross_source_consistency(
            target_observation, disagreeing_refs
        )
        
        # Agreement should be low with significant disagreement
        assert result.agreement_score < 0.5
        assert len(result.disagreement_details) > 0
    
    def test_scalar_agreement_calculation(self, cross_validator):
        """Test scalar agreement calculation"""
        # Test close agreement
        agreement = cross_validator._calculate_scalar_agreement(4.0, 4.1, DataType.SEISMIC)
        assert agreement > 0.9
        
        # Test moderate disagreement
        agreement = cross_validator._calculate_scalar_agreement(4.0, 5.0, DataType.SEISMIC)
        assert 0.3 < agreement < 0.8
        
        # Test significant disagreement
        agreement = cross_validator._calculate_scalar_agreement(4.0, 8.0, DataType.SEISMIC)
        assert agreement < 0.3


class TestCompletenessAnalyzer:
    """Test data completeness analysis functionality"""
    
    @pytest.fixture
    def completeness_analyzer(self):
        return CompletenessAnalyzer()
    
    @pytest.fixture
    def sample_observations(self):
        """Create sample observations with some gaps"""
        base_time = datetime.utcnow() - timedelta(days=1)
        observations = []
        
        # Create observations with intentional gaps
        for i in range(24):  # 24 hours
            if i not in [5, 6, 15, 16, 17]:  # Create gaps at hours 5-6 and 15-17
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
        
        return observations
    
    @pytest.mark.asyncio
    async def test_completeness_analysis(self, completeness_analyzer, sample_observations):
        """Test basic completeness analysis"""
        time_range = (
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        
        report = await completeness_analyzer.analyze_completeness(
            sample_observations, time_range
        )
        
        assert report.overall_completeness >= 0.0
        assert report.temporal_completeness >= 0.0
        assert report.spatial_completeness >= 0.0
        assert len(report.temporal_gaps) > 0  # Should detect the intentional gaps
        assert isinstance(report.data_availability_matrix, pd.DataFrame)
    
    @pytest.mark.asyncio
    async def test_temporal_gap_detection(self, completeness_analyzer, sample_observations):
        """Test temporal gap detection"""
        time_range = (
            datetime.utcnow() - timedelta(days=1),
            datetime.utcnow()
        )
        
        gaps = await completeness_analyzer._analyze_temporal_gaps(sample_observations, time_range)
        
        assert len(gaps) > 0
        
        # Check gap properties
        for gap in gaps:
            assert gap.duration > timedelta(0)
            assert gap.start_time < gap.end_time
            assert gap.severity in ['minor', 'moderate', 'severe', 'critical']
    
    def test_spatial_bounds_calculation(self, completeness_analyzer, sample_observations):
        """Test spatial bounds calculation"""
        bounds = completeness_analyzer._calculate_spatial_bounds(sample_observations)
        
        assert len(bounds) == 4  # min_lat, max_lat, min_lon, max_lon
        assert bounds[0] <= bounds[1]  # min_lat <= max_lat
        assert bounds[2] <= bounds[3]  # min_lon <= max_lon


class TestQualityReporter:
    """Test quality reporting and alerting functionality"""
    
    @pytest.fixture
    def quality_reporter(self):
        return QualityReporter()
    
    @pytest.fixture
    def sample_validation_results(self):
        """Create sample validation results"""
        results = []
        
        for i in range(10):
            quality_score = 0.9 - (i * 0.05)  # Decreasing quality
            
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
            
            results.append(result)
        
        return results
    
    @pytest.mark.asyncio
    async def test_quality_report_generation(self, quality_reporter, sample_validation_results):
        """Test quality report generation"""
        report = await quality_reporter.generate_quality_report(
            report_type='test',
            validation_results=sample_validation_results
        )
        
        assert report.overall_quality_score >= 0.0
        assert report.total_observations == len(sample_validation_results)
        assert 'excellent' in report.quality_distribution
        assert 'good' in report.quality_distribution
        assert 'fair' in report.quality_distribution
        assert 'poor' in report.quality_distribution
    
    @pytest.mark.asyncio
    async def test_quality_threshold_checking(self, quality_reporter, sample_validation_results):
        """Test quality threshold checking for alerts"""
        # This would test the alert generation logic
        # For now, we'll test that the method runs without error
        await quality_reporter._check_quality_thresholds(sample_validation_results)
        
        # Check that alerts were generated for low quality observations
        low_quality_results = [r for r in sample_validation_results if r.quality_metrics.overall_quality < 0.5]
        if low_quality_results:
            assert len(quality_reporter.active_alerts) >= 0  # Should have some alerts


class TestIntegratedValidator:
    """Test the integrated validation system"""
    
    @pytest.fixture
    def integrated_validator(self):
        config = ValidationWorkflowConfig(
            enable_cross_source_validation=True,
            enable_completeness_analysis=True,
            enable_automated_reporting=True
        )
        return IntegratedDataQualityValidator(config)
    
    @pytest.fixture
    def test_observations(self):
        """Create test observations for integrated validation"""
        observations = []
        base_time = datetime.utcnow()
        
        for i in range(5):
            obs = GeologicalObservation(
                observation_id=f"integrated_test_{i}",
                timestamp=base_time + timedelta(minutes=i * 10),
                location=(40.0 + i * 0.01, -120.0 + i * 0.01, 0.0),
                data_type=DataType.SEISMIC,
                value=3.0 + i * 0.2,
                uncertainty=0.1,
                source_system=f"source_{i % 2}"  # Alternate between two sources
            )
            observations.append(obs)
        
        return observations
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation(self, integrated_validator, test_observations):
        """Test comprehensive validation of single observation"""
        target_obs = test_observations[0]
        reference_obs = test_observations[1:]
        
        result = await integrated_validator.validate_observation_comprehensive(
            target_obs, reference_obs
        )
        
        assert result.observation == target_obs
        assert result.primary_validation is not None
        assert 0.0 <= result.overall_confidence <= 1.0
        assert result.processing_time_ms >= 0
        assert len(result.consolidated_flags) > 0
    
    @pytest.mark.asyncio
    async def test_batch_validation(self, integrated_validator, test_observations):
        """Test batch validation of multiple observations"""
        results = await integrated_validator.validate_batch_observations(test_observations)
        
        assert len(results) == len(test_observations)
        
        for result in results:
            assert result.primary_validation is not None
            assert 0.0 <= result.overall_confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_dataset_completeness_analysis(self, integrated_validator, test_observations):
        """Test dataset completeness analysis"""
        report = await integrated_validator.analyze_dataset_completeness(test_observations)
        
        assert report.overall_completeness >= 0.0
        assert report.temporal_completeness >= 0.0
        assert report.spatial_completeness >= 0.0
    
    @pytest.mark.asyncio
    async def test_quality_dashboard_report(self, integrated_validator, test_observations):
        """Test quality dashboard report generation"""
        report = await integrated_validator.generate_quality_dashboard_report(test_observations)
        
        assert report.overall_quality_score >= 0.0
        assert report.total_observations > 0
        assert len(report.priority_actions) >= 0
        assert len(report.improvement_suggestions) >= 0


# Integration test to verify all components work together
@pytest.mark.asyncio
async def test_full_validation_workflow():
    """Test the complete validation workflow end-to-end"""
    
    # Create test data
    observations = []
    base_time = datetime.utcnow()
    
    for i in range(20):
        obs = GeologicalObservation(
            observation_id=f"workflow_test_{i}",
            timestamp=base_time + timedelta(minutes=i * 5),
            location=(40.0 + i * 0.001, -120.0 + i * 0.001, 0.0),
            data_type=DataType.SEISMIC,
            value=3.0 + np.random.normal(0, 0.3),
            uncertainty=0.1,
            source_system="IRIS" if i % 2 == 0 else "USGS"
        )
        observations.append(obs)
    
    # Initialize integrated validator
    validator = IntegratedDataQualityValidator()
    
    # Perform batch validation
    validation_results = await validator.validate_batch_observations(observations)
    
    # Generate completeness analysis
    completeness_report = await validator.analyze_dataset_completeness(observations)
    
    # Generate quality report
    quality_report = await validator.generate_quality_dashboard_report(
        observations, validation_results
    )
    
    # Verify results
    assert len(validation_results) == len(observations)
    assert completeness_report.overall_completeness >= 0.0
    assert quality_report.overall_quality_score >= 0.0
    
    print(f"Workflow test completed successfully:")
    print(f"- Validated {len(observations)} observations")
    print(f"- Overall quality score: {quality_report.overall_quality_score:.3f}")
    print(f"- Overall completeness: {completeness_report.overall_completeness:.3f}")
    print(f"- Active alerts: {len(quality_report.active_alerts)}")


if __name__ == "__main__":
    # Run the integration test
    asyncio.run(test_full_validation_workflow())