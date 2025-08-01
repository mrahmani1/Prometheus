"""
Unit tests for mineral resource prediction functionality.

Tests the machine learning models for ore deposit pattern recognition,
geochemical signature analysis, and geological process modeling.
"""

import shutil
import tempfile
from datetime import datetime

import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
from core.mineral_resource_prediction import (
    DepositType,
    FormationProcess,
    GeochemicalSignatureAnalyzer,
    GeologicalProcessModeler,
    MineralResourcePredictor,
    OreDepositPatternRecognizer,
    ResourceDeposit,
    ResourcePrediction,
    ResourceType,
)
from models.geological import GeochemicalSample, GeologicalLocation

class TestResourceDeposit:
    """Test ResourceDeposit data model"""

    def test_valid_resource_deposit(self):
        """Test creating a valid resource deposit"""
        location = GeologicalLocation(
            latitude=45.0, longitude=-120.0, elevation=1000.0, depth=500.0
        )

        deposit = ResourceDeposit(
            location=location,
            resource_type=ResourceType.COPPER,
            deposit_type=DepositType.PORPHYRY,
            formation_process=FormationProcess.HYDROTHERMAL,
            grade=1.5,
            tonnage=1000000.0,
            geochemical_signature={"Cu": 15000, "Mo": 100, "Au": 0.5},
            geological_setting={"rock_types": ["igneous"], "fault_density": 5.0},
            confidence=0.9,
        )

        assert deposit.resource_type == ResourceType.COPPER
        assert deposit.grade == 1.5
        assert deposit.tonnage == 1000000.0
        assert deposit.confidence == 0.9

    def test_invalid_grade(self):
        """Test that invalid grade raises ValueError"""
        location = GeologicalLocation(45.0, -120.0, 1000.0)

        with pytest.raises(ValueError, match="Grade must be positive"):
            ResourceDeposit(
                location=location,
                resource_type=ResourceType.COPPER,
                deposit_type=DepositType.PORPHYRY,
                formation_process=FormationProcess.HYDROTHERMAL,
                grade=-1.0,  # Invalid
                tonnage=1000000.0,
                geochemical_signature={},
                geological_setting={},
            )

    def test_invalid_tonnage(self):
        """Test that invalid tonnage raises ValueError"""
        location = GeologicalLocation(45.0, -120.0, 1000.0)

        with pytest.raises(ValueError, match="Tonnage must be positive"):
            ResourceDeposit(
                location=location,
                resource_type=ResourceType.COPPER,
                deposit_type=DepositType.PORPHYRY,
                formation_process=FormationProcess.HYDROTHERMAL,
                grade=1.5,
                tonnage=0.0,  # Invalid
                geochemical_signature={},
                geological_setting={},
            )

class TestGeochemicalSignatureAnalyzer:
    """Test GeochemicalSignatureAnalyzer functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing"""
        return GeochemicalSignatureAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Create sample geochemical data for testing"""
        location = GeologicalLocation(45.0, -120.0, 1000.0)

        samples = [
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type="rock",
                composition={
                    "SiO2": 65.0,
                    "Al2O3": 15.0,
                    "Fe2O3": 5.0,
                    "Cu": 5000,
                    "Mo": 100,
                    "Au": 2.0,
                },
                temperature=300.0,
                pressure=1e6,
            ),
            GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type="rock",
                composition={
                    "SiO2": 70.0,
                    "Al2O3": 12.0,
                    "Fe2O3": 3.0,
                    "Pb": 1000,
                    "Zn": 5000,
                    "Ag": 50,
                },
                temperature=250.0,
                pressure=8e5,
            ),
        ]

        deposit_types = [DepositType.PORPHYRY, DepositType.EPITHERMAL]

        return samples, deposit_types

    def test_extract_signature_features(self, analyzer, sample_data):
        """Test feature extraction from geochemical samples"""
        samples, _ = sample_data

        features = analyzer.extract_signature_features(samples)

        assert features.shape[0] == 2  # Two samples
        assert features.shape[1] > 18  # At least base elements + ratios
        assert not np.any(np.isnan(features))  # No NaN values

    def test_train_signature_classifier(self, analyzer, sample_data):
        """Test training of signature classifier"""
        samples, deposit_types = sample_data

        # Add more samples for better training
        location = GeologicalLocation(45.0, -120.0, 1000.0)
        for i in range(10):
            samples.append(
                GeochemicalSample(
                    timestamp=datetime.utcnow(),
                    location=location,
                    sample_type="rock",
                    composition={
                        "SiO2": 60.0 + i,
                        "Al2O3": 15.0,
                        "Cu": 1000 + i * 100,
                        "Au": 1.0 + i * 0.1,
                    },
                )
            )
            deposit_types.append(
                DepositType.PORPHYRY if i % 2 == 0 else DepositType.EPITHERMAL
            )

        results = analyzer.train_signature_classifier(samples, deposit_types)

        assert "train_accuracy" in results
        assert "test_accuracy" in results
        assert "cv_mean" in results
        assert "feature_importance" in results
        assert results["n_samples"] == len(samples)
        assert "deposit_classifier" in analyzer.signature_models

    def test_predict_deposit_type(self, analyzer, sample_data):
        """Test deposit type prediction"""
        samples, deposit_types = sample_data

        # Add more samples and train
        location = GeologicalLocation(45.0, -120.0, 1000.0)
        for i in range(10):
            samples.append(
                GeochemicalSample(
                    timestamp=datetime.utcnow(),
                    location=location,
                    sample_type="rock",
                    composition={
                        "SiO2": 60.0 + i,
                        "Al2O3": 15.0,
                        "Cu": 1000 + i * 100,
                        "Au": 1.0 + i * 0.1,
                    },
                )
            )
            deposit_types.append(
                DepositType.PORPHYRY if i % 2 == 0 else DepositType.EPITHERMAL
            )

        analyzer.train_signature_classifier(samples, deposit_types)

        # Test prediction
        test_sample = samples[0]
        predictions = analyzer.predict_deposit_type(test_sample)

        assert isinstance(predictions, dict)
        assert all(isinstance(k, DepositType) for k in predictions.keys())
        assert all(0 <= v <= 1 for v in predictions.values())
        assert abs(sum(predictions.values()) - 1.0) < 1e-6  # Probabilities sum to 1

    def test_predict_without_training(self, analyzer, sample_data):
        """Test that prediction without training raises error"""
        samples, _ = sample_data

        with pytest.raises(ValueError, match="Signature classifier not trained"):
            analyzer.predict_deposit_type(samples[0])

class TestOreDepositPatternRecognizer:
    """Test OreDepositPatternRecognizer functionality"""

    @pytest.fixture
    def recognizer(self):
        """Create recognizer instance for testing"""
        return OreDepositPatternRecognizer()

    @pytest.fixture
    def deposit_data(self):
        """Create sample deposit data for testing"""
        deposits = []

        for i in range(20):
            location = GeologicalLocation(
                latitude=45.0 + i * 0.1,
                longitude=-120.0 + i * 0.1,
                elevation=1000.0 + i * 10,
                depth=500.0 + i * 50,
            )

            deposit = ResourceDeposit(
                location=location,
                resource_type=ResourceType.COPPER if i % 2 == 0 else ResourceType.GOLD,
                deposit_type=(
                    DepositType.PORPHYRY if i % 2 == 0 else DepositType.EPITHERMAL
                ),
                formation_process=FormationProcess.HYDROTHERMAL,
                grade=1.0 + i * 0.1,
                tonnage=1000000.0 + i * 100000,
                geochemical_signature={
                    "Cu": 1000 + i * 100,
                    "Au": 1.0 + i * 0.1,
                    "Mo": 50 + i * 5,
                },
                geological_setting={
                    "rock_types": ["igneous"],
                    "fault_density": 5.0 + i * 0.5,
                    "fracture_intensity": 3.0,
                    "intrusion_distance": 1000.0 - i * 20,
                    "volcanic_activity": 0.5,
                    "hydrothermal_alteration": 0.7,
                },
                confidence=0.8 + i * 0.01,
            )
            deposits.append(deposit)

        return deposits

    def test_extract_spatial_features(self, recognizer, deposit_data):
        """Test spatial feature extraction"""
        features = recognizer.extract_spatial_features(deposit_data)

        assert features.shape[0] == len(deposit_data)
        assert features.shape[1] > 10  # Multiple feature types
        assert not np.any(np.isnan(features))

    def test_train_pattern_models(self, recognizer, deposit_data):
        """Test training of pattern recognition models"""
        results = recognizer.train_pattern_models(deposit_data)

        assert "resource_type" in results
        assert "deposit_type" in results
        assert "grade" in results
        assert "tonnage" in results

        # Check that models were created
        assert "resource_type_classifier" in recognizer.pattern_models
        assert "deposit_type_classifier" in recognizer.pattern_models
        assert "grade_predictor" in recognizer.pattern_models
        assert "tonnage_predictor" in recognizer.pattern_models

    def test_predict_resource_potential(self, recognizer, deposit_data):
        """Test resource potential prediction"""
        # Train models first
        recognizer.train_pattern_models(deposit_data)

        # Test prediction
        location = GeologicalLocation(45.5, -119.5, 1050.0, 600.0)
        geological_setting = {
            "rock_types": ["igneous"],
            "fault_density": 6.0,
            "fracture_intensity": 3.5,
            "fold_intensity": 2.0,
            "intrusion_distance": 800.0,
            "volcanic_activity": 0.6,
            "hydrothermal_alteration": 0.8,
        }
        geochemical_data = {"Cu": 2000, "Au": 1.5, "Mo": 75}

        prediction = recognizer.predict_resource_potential(
            location, geological_setting, geochemical_data
        )

        assert isinstance(prediction, ResourcePrediction)
        assert isinstance(prediction.resource_type, ResourceType)
        assert 0 <= prediction.probability <= 1
        assert prediction.predicted_grade >= 0
        assert prediction.predicted_tonnage >= 0
        assert 0 <= prediction.geological_favorability <= 1
        assert 0 <= prediction.model_confidence <= 1

class TestGeologicalProcessModeler:
    """Test GeologicalProcessModeler functionality"""

    @pytest.fixture
    def modeler(self):
        """Create modeler instance for testing"""
        return GeologicalProcessModeler()

    def test_model_hydrothermal_process(self, modeler):
        """Test hydrothermal process modeling"""
        temperature = 350.0
        pressure = 1e7
        fluid_composition = {"NaCl": 15.0, "pH": 6.0, "S": 0.1, "Cu": 100, "Au": 0.5}
        host_rock = {"reactivity": 0.7, "permeability": 1e-12}

        result = modeler.model_hydrothermal_process(
            temperature, pressure, fluid_composition, host_rock
        )

        assert result["formation_process"] == FormationProcess.HYDROTHERMAL
        assert "metal_solubility" in result
        assert "precipitation_temperature" in result
        assert "alteration_minerals" in result
        assert "estimated_grade" in result
        assert "favorability_score" in result
        assert 0 <= result["favorability_score"] <= 1

    def test_model_magmatic_process(self, modeler):
        """Test magmatic process modeling"""
        magma_composition = {
            "SiO2": 55.0,
            "MgO": 8.0,
            "FeO": 10.0,
            "S": 0.2,
            "Cu": 200,
            "Ni": 150,
        }
        crystallization_conditions = {
            "temperature": 1200.0,
            "pressure": 2e8,
            "oxygen_fugacity": -10.0,
        }

        result = modeler.model_magmatic_process(
            magma_composition, crystallization_conditions
        )

        assert result["formation_process"] == FormationProcess.MAGMATIC
        assert "sulfide_saturation" in result
        assert "metal_partitioning" in result
        assert "ore_minerals" in result
        assert "favorability_score" in result
        assert 0 <= result["favorability_score"] <= 1

    def test_model_sedimentary_process(self, modeler):
        """Test sedimentary process modeling"""
        sediment_composition = {
            "SiO2": 60.0,
            "Al2O3": 15.0,
            "Cu": 50,
            "Pb": 20,
            "Zn": 100,
            "U": 5,
        }
        depositional_environment = {
            "oxygen_level": "anoxic",
            "water_depth": 300,
            "organic_content": 0.15,
            "clay_content": 0.25,
            "evaporation_rate": 0.3,
        }

        result = modeler.model_sedimentary_process(
            sediment_composition, depositional_environment
        )

        assert result["formation_process"] == FormationProcess.SEDIMENTARY
        assert "redox_conditions" in result
        assert "metal_mobility" in result
        assert "concentration_mechanisms" in result
        assert "favorability_score" in result
        assert 0 <= result["favorability_score"] <= 1

class TestMineralResourcePredictor:
    """Test main MineralResourcePredictor class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def predictor(self, temp_dir):
        """Create predictor instance for testing"""
        return MineralResourcePredictor(model_storage_path=temp_dir)

    @pytest.fixture
    def training_data(self):
        """Create training data for testing"""
        deposits = []
        samples = []

        for i in range(15):
            location = GeologicalLocation(
                latitude=45.0 + i * 0.1,
                longitude=-120.0 + i * 0.1,
                elevation=1000.0,
                depth=500.0,
            )

            deposit = ResourceDeposit(
                location=location,
                resource_type=ResourceType.COPPER if i % 2 == 0 else ResourceType.GOLD,
                deposit_type=(
                    DepositType.PORPHYRY if i % 2 == 0 else DepositType.EPITHERMAL
                ),
                formation_process=FormationProcess.HYDROTHERMAL,
                grade=1.0 + i * 0.1,
                tonnage=1000000.0,
                geochemical_signature={"Cu": 1000 + i * 100, "Au": 1.0 + i * 0.1},
                geological_setting={
                    "rock_types": ["igneous"],
                    "fault_density": 5.0,
                    "intrusion_distance": 1000.0,
                },
            )
            deposits.append(deposit)

            sample = GeochemicalSample(
                timestamp=datetime.utcnow(),
                location=location,
                sample_type="rock",
                composition={"Cu": 1000 + i * 100, "Au": 1.0 + i * 0.1, "SiO2": 65.0},
            )
            samples.append(sample)

        return deposits, samples

    def test_add_training_data(self, predictor, training_data):
        """Test adding training data"""
        deposits, samples = training_data

        predictor.add_training_data(deposits, samples)

        assert len(predictor.training_deposits) == len(deposits)
        assert len(predictor.training_samples) == len(samples)

    def test_train_models(self, predictor, training_data):
        """Test model training"""
        deposits, samples = training_data

        predictor.add_training_data(deposits, samples)
        results = predictor.train_models()

        assert "signature_analysis" in results
        assert "pattern_recognition" in results
        assert len(predictor.signature_analyzer.signature_models) > 0
        assert len(predictor.pattern_recognizer.pattern_models) > 0

    def test_predict_resource_potential(self, predictor, training_data):
        """Test resource potential prediction"""
        deposits, samples = training_data

        # Train models
        predictor.add_training_data(deposits, samples)
        predictor.train_models()

        # Test prediction
        location = GeologicalLocation(45.5, -119.5, 1050.0)
        geological_setting = {
            "rock_types": ["igneous"],
            "fault_density": 6.0,
            "fracture_intensity": 3.0,
            "intrusion_distance": 800.0,
            "volcanic_activity": 0.6,
            "hydrothermal_alteration": 0.8,
        }
        geochemical_data = {"Cu": 2000, "Au": 1.5, "SiO2": 65.0}

        prediction = predictor.predict_resource_potential(
            location, geological_setting, geochemical_data
        )

        assert isinstance(prediction, ResourcePrediction)
        assert prediction.location == location
        assert isinstance(prediction.resource_type, ResourceType)
        assert 0 <= prediction.probability <= 1
        assert prediction.predicted_grade >= 0
        assert prediction.predicted_tonnage >= 0
        assert "grade" in prediction.uncertainty_bounds
        assert "tonnage" in prediction.uncertainty_bounds

    def test_generate_resource_map(self, predictor, training_data):
        """Test resource map generation"""
        deposits, samples = training_data

        # Train models
        predictor.add_training_data(deposits, samples)
        predictor.train_models()

        # Test map generation
        locations = [
            GeologicalLocation(45.1, -119.9, 1000.0),
            GeologicalLocation(45.2, -119.8, 1100.0),
            GeologicalLocation(45.3, -119.7, 1200.0),
        ]
        geological_settings = [
            {
                "rock_types": ["igneous"],
                "fault_density": 5.0,
                "intrusion_distance": 1000.0,
                "volcanic_activity": 0.5,
                "hydrothermal_alteration": 0.7,
            }
        ] * 3

        map_data = predictor.generate_resource_map(
            locations, geological_settings, ResourceType.COPPER
        )

        assert "locations" in map_data
        assert "probabilities" in map_data
        assert "resource_types" in map_data
        assert "predicted_grades" in map_data
        assert "predicted_tonnages" in map_data
        assert "metadata" in map_data
        assert len(map_data["locations"]) <= len(
            locations
        )  # May filter by resource type

    def test_save_and_load_models(self, predictor, training_data, temp_dir):
        """Test model saving and loading"""
        deposits, samples = training_data

        # Train and save models
        predictor.add_training_data(deposits, samples)
        predictor.train_models()

        # Create new predictor and load models
        new_predictor = MineralResourcePredictor(model_storage_path=temp_dir)
        new_predictor.load_models()

        # Check that models were loaded
        assert len(new_predictor.signature_analyzer.signature_models) > 0
        assert len(new_predictor.pattern_recognizer.pattern_models) > 0

    def test_evaluate_model_performance(self, predictor, training_data):
        """Test model performance evaluation"""
        deposits, samples = training_data

        # Train models
        predictor.add_training_data(deposits, samples)
        predictor.train_models()

        # Use some deposits as test data
        test_deposits = deposits[:5]

        metrics = predictor.evaluate_model_performance(test_deposits)

        assert "resource_type_accuracy" in metrics
        assert "mean_grade_error" in metrics
        assert "mean_tonnage_error" in metrics
        assert "probability_precision" in metrics
        assert "n_predictions" in metrics
        assert metrics["n_predictions"] == len(test_deposits)

    def test_train_without_data(self, predictor):
        """Test that training without data raises error"""
        with pytest.raises(ValueError, match="No training data available"):
            predictor.train_models()

if __name__ == "__main__":
    pytest.main([__file__])
