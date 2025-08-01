"""
Tests for integrated geodynamic process modeling

Tests for seismic tomography, deformation tracking, earthquake analysis,
and geochemical analysis components.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.geodynamic_modeling import (
    DeformationField,
    DeformationTracker,
    EarthquakeAnalysisResult,
    EarthquakeAnalyzer,
    GeochemicalAnalysisResult,
    GeochemicalAnalyzer,
    IntegratedGeodynamicModeler,
    SeismicTomographyProcessor,
    SeismicTomographyResult,
)
from core.pinn_engine import PINNEngine
from models.geological import (
    DeformationMeasurement,
    EventType,
    GeochemicalSample,
    GeologicalLocation,
    MeasurementType,
    SampleType,
    SeismicEvent,
)

@pytest.fixture
def device():
    """Test device fixture"""
    return torch.device("cpu")

@pytest.fixture
def mock_pinn_engine():
    """Mock PINN engine for testing"""
    engine = Mock(spec=PINNEngine)
    engine.create_model.return_value = Mock()
    engine.get_trainer.return_value = Mock()
    engine.predict.return_value = torch.randn(100, 1)
    return engine

@pytest.fixture
def sample_seismic_events():
    """Sample seismic events for testing"""
    events = []
    base_time = datetime.utcnow()

    for i in range(10):
        event = SeismicEvent(
            timestamp=base_time + timedelta(hours=i),
            location=GeologicalLocation(
                latitude=45.0 + i * 0.01,
                longitude=-122.0 + i * 0.01,
                elevation=1000.0,
                depth=5.0 + i * 0.5,
            ),
            magnitude=2.0 + i * 0.2,
            depth=5.0 + i * 0.5,
            event_type=EventType.EARTHQUAKE,
            confidence=0.9,
        )
        events.append(event)

    return events

@pytest.fixture
def sample_deformation_measurements():
    """Sample deformation measurements for testing"""
    measurements = []
    base_time = datetime.utcnow()

    for i in range(5):
        measurement = DeformationMeasurement(
            timestamp=base_time + timedelta(days=i),
            location=GeologicalLocation(
                latitude=45.0 + i * 0.01, longitude=-122.0 + i * 0.01, elevation=1000.0
            ),
            displacement_vector=np.array([0.001 * i, 0.002 * i, 0.0005 * i]),
            measurement_type=MeasurementType.GPS,
            uncertainty=np.array([0.001, 0.001, 0.002]),
            velocity_vector=np.array([0.01, 0.02, 0.005]),
        )
        measurements.append(measurement)

    return measurements

@pytest.fixture
def sample_geochemical_samples():
    """Sample geochemical samples for testing"""
    samples = []
    base_time = datetime.utcnow()

    compositions = [
        {"SiO2": 50.0, "MgO": 8.0, "Al2O3": 15.0, "CaO": 10.0, "H2O": 2.0},
        {"SiO2": 52.0, "MgO": 7.5, "Al2O3": 15.5, "CaO": 9.5, "H2O": 1.8},
        {"SiO2": 54.0, "MgO": 7.0, "Al2O3": 16.0, "CaO": 9.0, "H2O": 1.5},
    ]

    for i, composition in enumerate(compositions):
        sample = GeochemicalSample(
            timestamp=base_time + timedelta(days=i),
            location=GeologicalLocation(
                latitude=45.0, longitude=-122.0, elevation=1000.0
            ),
            sample_type=SampleType.LAVA,
            composition=composition,
            temperature=1100.0 + i * 10,
            pressure=1e8 + i * 1e7,
        )
        samples.append(sample)

    return samples

@pytest.fixture
def sample_station_locations():
    """Sample seismic station locations for testing"""
    stations = []

    for i in range(5):
        station = GeologicalLocation(
            latitude=45.0 + i * 0.1, longitude=-122.0 + i * 0.1, elevation=500.0
        )
        stations.append(station)

    return stations

class TestSeismicTomographyProcessor:
    """Tests for seismic tomography processor"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test processor initialization"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    processor = SeismicTomographyProcessor(mock_pinn_engine, device)

                    assert processor.pinn_engine == mock_pinn_engine
                    assert processor.device == device
                    assert isinstance(processor.velocity_model_cache, dict)

    def test_process_seismic_data(
        self, mock_pinn_engine, device, sample_seismic_events, sample_station_locations
    ):
        """Test seismic data processing"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    processor = SeismicTomographyProcessor(mock_pinn_engine, device)

                    grid_bounds = ((-123.0, -121.0), (44.0, 46.0), (0.0, 20.0))
                    resolution = 1.0

                    result = processor.process_seismic_data(
                        sample_seismic_events,
                        sample_station_locations,
                        grid_bounds,
                        resolution,
                    )

                    assert isinstance(result, SeismicTomographyResult)
                    assert result.velocity_model.shape == (2, 2, 20)
                    assert result.resolution == resolution
                    assert isinstance(result.quality_metrics, dict)

    def test_prepare_seismic_training_data(
        self, mock_pinn_engine, device, sample_seismic_events, sample_station_locations
    ):
        """Test seismic training data preparation"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    processor = SeismicTomographyProcessor(mock_pinn_engine, device)

                    training_data = processor._prepare_seismic_training_data(
                        sample_seismic_events, sample_station_locations
                    )

                    assert "inputs" in training_data
                    assert "targets" in training_data
                    assert (
                        training_data["inputs"].shape[1] == 6
                    )  # event + station coordinates
                    assert training_data["targets"].shape[1] == 1  # travel time

class TestDeformationTracker:
    """Tests for deformation tracker"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test tracker initialization"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    tracker = DeformationTracker(mock_pinn_engine, device)

                    assert tracker.pinn_engine == mock_pinn_engine
                    assert tracker.device == device
                    assert tracker.accuracy_threshold == 0.001

    def test_track_deformation(
        self, mock_pinn_engine, device, sample_deformation_measurements
    ):
        """Test deformation tracking"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    tracker = DeformationTracker(mock_pinn_engine, device)

                    grid_bounds = ((-123.0, -121.0), (44.0, 46.0), (0.0, 10.0))
                    resolution = 0.5

                    result = tracker.track_deformation(
                        sample_deformation_measurements, grid_bounds, resolution
                    )

                    assert isinstance(result, DeformationField)
                    assert result.displacement_field.shape[-1] == 3  # 3D displacement
                    assert result.velocity_field.shape[-1] == 3  # 3D velocity
                    assert result.strain_tensor.shape[-1] == 6  # 6 strain components
                    assert result.accuracy <= tracker.accuracy_threshold

    def test_compute_strain_tensor(self, mock_pinn_engine, device):
        """Test strain tensor computation"""
        with patch("models.ai_models.PINNModelConfig"):
            with patch("models.ai_models.NetworkArchitecture"):
                with patch("models.ai_models.PhysicsConstraint"):
                    tracker = DeformationTracker(mock_pinn_engine, device)

                    # Create test displacement field
                    displacement_field = np.random.randn(10, 10, 5, 3)
                    resolution = 1.0

                    strain_tensor = tracker._compute_strain_tensor(
                        displacement_field, resolution
                    )

                    assert strain_tensor.shape == (10, 10, 5, 6)
                    assert not np.isnan(strain_tensor).any()

class TestEarthquakeAnalyzer:
    """Tests for earthquake analyzer"""

    def test_initialization(self, device):
        """Test analyzer initialization"""
        analyzer = EarthquakeAnalyzer(device)

        assert analyzer.device == device
        assert hasattr(analyzer, "clustering_algorithm")
        assert hasattr(analyzer, "scaler")

    def test_analyze_earthquake_catalog(self, device, sample_seismic_events):
        """Test earthquake catalog analysis"""
        analyzer = EarthquakeAnalyzer(device)

        result = analyzer.analyze_earthquake_catalog(sample_seismic_events)

        assert isinstance(result, EarthquakeAnalysisResult)
        assert result.stress_changes.shape[-1] == 6  # 6 stress components
        assert isinstance(result.magma_movement_indicators, dict)
        assert isinstance(result.clustering_results, dict)
        assert isinstance(result.temporal_patterns, dict)
        assert isinstance(result.magnitude_frequency, dict)

    def test_detect_magma_movement(self, device, sample_seismic_events):
        """Test magma movement detection"""
        analyzer = EarthquakeAnalyzer(device)

        indicators = analyzer._detect_magma_movement(sample_seismic_events)

        assert "depth_migration_rate" in indicators
        assert "magnitude_trend" in indicators
        assert "spatial_migration" in indicators
        assert "event_rate" in indicators
        assert "volcanic_probability" in indicators

    def test_calculate_magnitude_frequency(self, device, sample_seismic_events):
        """Test magnitude-frequency calculation"""
        analyzer = EarthquakeAnalyzer(device)

        mag_freq = analyzer._calculate_magnitude_frequency(sample_seismic_events)

        assert "b_value" in mag_freq
        assert "a_value" in mag_freq
        assert "total_events" in mag_freq
        assert mag_freq["total_events"] == len(sample_seismic_events)

    def test_perform_clustering(self, device, sample_seismic_events):
        """Test spatial-temporal clustering"""
        analyzer = EarthquakeAnalyzer(device)

        clustering_results = analyzer._perform_clustering(sample_seismic_events)

        assert "clusters" in clustering_results
        assert "num_clusters" in clustering_results
        assert isinstance(clustering_results["clusters"], list)

class TestGeochemicalAnalyzer:
    """Tests for geochemical analyzer"""

    def test_initialization(self, device):
        """Test analyzer initialization"""
        analyzer = GeochemicalAnalyzer(device)

        assert analyzer.device == device
        assert isinstance(analyzer.composition_models, dict)

    def test_analyze_geochemical_data(self, device, sample_geochemical_samples):
        """Test geochemical data analysis"""
        analyzer = GeochemicalAnalyzer(device)

        result = analyzer.analyze_geochemical_data(sample_geochemical_samples)

        assert isinstance(result, GeochemicalAnalysisResult)
        assert isinstance(result.magma_composition, dict)
        assert isinstance(result.temperature_estimates, dict)
        assert isinstance(result.pressure_estimates, dict)
        assert isinstance(result.volatile_content, dict)
        assert isinstance(result.crystallinity, float)
        assert isinstance(result.evolution_pathway, list)
        assert isinstance(result.uncertainty_estimates, dict)

    def test_calculate_magma_composition(self, device, sample_geochemical_samples):
        """Test magma composition calculation"""
        analyzer = GeochemicalAnalyzer(device)

        composition = analyzer._calculate_magma_composition(sample_geochemical_samples)

        assert isinstance(composition, dict)
        assert "SiO2" in composition
        assert "MgO" in composition
        assert all(isinstance(v, float) for v in composition.values())

    def test_estimate_temperature(self, device, sample_geochemical_samples):
        """Test temperature estimation"""
        analyzer = GeochemicalAnalyzer(device)

        temperatures = analyzer._estimate_temperature(sample_geochemical_samples)

        assert isinstance(temperatures, dict)
        assert "measured" in temperatures
        assert temperatures["measured"] > 1000  # Reasonable magma temperature

    def test_determine_evolution_pathway(self, device, sample_geochemical_samples):
        """Test evolution pathway determination"""
        analyzer = GeochemicalAnalyzer(device)

        pathway = analyzer._determine_evolution_pathway(sample_geochemical_samples)

        assert isinstance(pathway, list)
        if len(pathway) > 0:
            step = pathway[0]
            assert "step" in step
            assert "process_type" in step
            assert "compositional_changes" in step

class TestIntegratedGeodynamicModeler:
    """Tests for integrated geodynamic modeler"""

    def test_initialization(self, mock_pinn_engine, device):
        """Test modeler initialization"""
        with patch("core.geodynamic_modeling.SeismicTomographyProcessor"):
            with patch("core.geodynamic_modeling.DeformationTracker"):
                with patch("core.geodynamic_modeling.EarthquakeAnalyzer"):
                    with patch("core.geodynamic_modeling.GeochemicalAnalyzer"):
                        modeler = IntegratedGeodynamicModeler(mock_pinn_engine, device)

                        assert modeler.pinn_engine == mock_pinn_engine
                        assert modeler.device == device
                        assert hasattr(modeler, "tomography_processor")
                        assert hasattr(modeler, "deformation_tracker")
                        assert hasattr(modeler, "earthquake_analyzer")
                        assert hasattr(modeler, "geochemical_analyzer")

    def test_process_integrated_data(
        self,
        mock_pinn_engine,
        device,
        sample_seismic_events,
        sample_deformation_measurements,
        sample_geochemical_samples,
        sample_station_locations,
    ):
        """Test integrated data processing"""
        with patch("core.geodynamic_modeling.SeismicTomographyProcessor") as mock_tomo:
            with patch("core.geodynamic_modeling.DeformationTracker") as mock_deform:
                with patch("core.geodynamic_modeling.EarthquakeAnalyzer") as mock_eq:
                    with patch(
                        "core.geodynamic_modeling.GeochemicalAnalyzer"
                    ) as mock_geochem:

                        # Mock the processors
                        mock_tomo.return_value.process_seismic_data.return_value = Mock(
                            spec=SeismicTomographyResult
                        )
                        mock_deform.return_value.track_deformation.return_value = Mock(
                            spec=DeformationField
                        )
                        mock_eq.return_value.analyze_earthquake_catalog.return_value = (
                            Mock(spec=EarthquakeAnalysisResult)
                        )
                        mock_geochem.return_value.analyze_geochemical_data.return_value = Mock(
                            spec=GeochemicalAnalysisResult
                        )

                        modeler = IntegratedGeodynamicModeler(mock_pinn_engine, device)

                        grid_bounds = ((-123.0, -121.0), (44.0, 46.0), (0.0, 20.0))
                        resolution = 1.0

                        results = modeler.process_integrated_data(
                            sample_seismic_events,
                            sample_deformation_measurements,
                            sample_geochemical_samples,
                            sample_station_locations,
                            grid_bounds,
                            resolution,
                        )

                        assert isinstance(results, dict)
                        assert "seismic_tomography" in results
                        assert "deformation_field" in results
                        assert "earthquake_analysis" in results
                        assert "geochemical_analysis" in results
                        assert "integrated_model" in results

    def test_integrate_analysis_results(self, mock_pinn_engine, device):
        """Test analysis results integration"""
        with patch("core.geodynamic_modeling.SeismicTomographyProcessor"):
            with patch("core.geodynamic_modeling.DeformationTracker"):
                with patch("core.geodynamic_modeling.EarthquakeAnalyzer"):
                    with patch("core.geodynamic_modeling.GeochemicalAnalyzer"):
                        modeler = IntegratedGeodynamicModeler(mock_pinn_engine, device)

                        # Mock results
                        mock_results = {
                            "seismic_tomography": Mock(spec=SeismicTomographyResult),
                            "deformation_field": Mock(spec=DeformationField),
                            "earthquake_analysis": Mock(spec=EarthquakeAnalysisResult),
                            "geochemical_analysis": Mock(
                                spec=GeochemicalAnalysisResult
                            ),
                        }

                        # Mock the analysis results with required attributes
                        mock_results[
                            "earthquake_analysis"
                        ].magma_movement_indicators = {
                            "volcanic_probability": np.array([0.7]),
                            "depth_migration_rate": np.array([0.1]),
                        }
                        mock_results["geochemical_analysis"].temperature_estimates = {
                            "measured": 1150
                        }
                        mock_results["geochemical_analysis"].pressure_estimates = {
                            "measured": 1.5e8
                        }

                        integrated = modeler._integrate_analysis_results(mock_results)

                        assert isinstance(integrated, dict)
                        assert "processing_timestamp" in integrated
                        assert "data_sources" in integrated
                        assert "integration_summary" in integrated

                        summary = integrated["integration_summary"]
                        assert "stress_consistency" in summary
                        assert "magma_indicators" in summary
                        assert "structural_consistency" in summary

class TestDataStructures:
    """Tests for data structure classes"""

    def test_seismic_tomography_result(self):
        """Test SeismicTomographyResult data structure"""
        velocity_model = np.random.randn(10, 10, 5)
        coordinates = np.random.randn(10, 10, 5, 3)
        uncertainty = np.random.randn(10, 10, 5)

        result = SeismicTomographyResult(
            velocity_model=velocity_model,
            coordinates=coordinates,
            uncertainty=uncertainty,
            resolution=1.0,
            quality_metrics={"resolution": 10.0},
            processing_timestamp=datetime.utcnow(),
        )

        assert np.array_equal(result.velocity_model, velocity_model)
        assert np.array_equal(result.coordinates, coordinates)
        assert np.array_equal(result.uncertainty, uncertainty)
        assert result.resolution == 1.0

    def test_deformation_field(self):
        """Test DeformationField data structure"""
        displacement_field = np.random.randn(10, 10, 5, 3)
        velocity_field = np.random.randn(10, 10, 5, 3)
        strain_tensor = np.random.randn(10, 10, 5, 6)
        coordinates = np.random.randn(10, 10, 5, 3)
        uncertainty = np.random.randn(10, 10, 5, 3)

        result = DeformationField(
            displacement_field=displacement_field,
            velocity_field=velocity_field,
            strain_tensor=strain_tensor,
            coordinates=coordinates,
            uncertainty=uncertainty,
            accuracy=0.001,
            processing_timestamp=datetime.utcnow(),
        )

        assert np.array_equal(result.displacement_field, displacement_field)
        assert np.array_equal(result.velocity_field, velocity_field)
        assert np.array_equal(result.strain_tensor, strain_tensor)
        assert result.accuracy == 0.001

    def test_earthquake_analysis_result(self):
        """Test EarthquakeAnalysisResult data structure"""
        stress_changes = np.random.randn(10, 10, 5, 6)
        magma_indicators = {"volcanic_probability": np.array([0.7])}

        result = EarthquakeAnalysisResult(
            stress_changes=stress_changes,
            magma_movement_indicators=magma_indicators,
            clustering_results={"num_clusters": 3},
            temporal_patterns={"mean_inter_event_time": 3600},
            magnitude_frequency={"b_value": 1.0},
            processing_timestamp=datetime.utcnow(),
        )

        assert np.array_equal(result.stress_changes, stress_changes)
        assert result.magma_movement_indicators == magma_indicators
        assert result.clustering_results["num_clusters"] == 3

    def test_geochemical_analysis_result(self):
        """Test GeochemicalAnalysisResult data structure"""
        result = GeochemicalAnalysisResult(
            magma_composition={"SiO2": 52.0, "MgO": 7.5},
            temperature_estimates={"measured": 1150.0},
            pressure_estimates={"measured": 1.5e8},
            volatile_content={"H2O": 2.0, "CO2": 0.5},
            crystallinity=0.3,
            evolution_pathway=[
                {"step": 1, "process_type": "fractional_crystallization"}
            ],
            uncertainty_estimates={"temperature_uncertainty": 25.0},
            processing_timestamp=datetime.utcnow(),
        )

        assert result.magma_composition["SiO2"] == 52.0
        assert result.temperature_estimates["measured"] == 1150.0
        assert result.crystallinity == 0.3
        assert len(result.evolution_pathway) == 1

if __name__ == "__main__":
    pytest.main([__file__])
