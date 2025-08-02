"""
Unit tests for AI model data structures

Tests all AI model configurations, predictions, and uncertainty quantification.
"""

from datetime import datetime

import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
from models.ai_models import (
    ActivationFunction,
    ModelMetadata,
    ModelPrediction,
    ModelRegistry,
    NetworkArchitecture,
    PhysicsConstraint,
    PhysicsConstraintType,
    PINNModelConfig,
    SimulationParameters,
    UncertaintyMethod,
    UncertaintyQuantification,
)


class TestNetworkArchitecture:
    """Test NetworkArchitecture data model"""

    def test_valid_architecture_creation(self):
        """Test creating a valid network architecture"""
        architecture = NetworkArchitecture(
            layer_sizes=[10, 64, 32, 1],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.SIGMOID,
            ],
            dropout_rates=[0.1, 0.2, 0.0],
            batch_normalization=True,
        )

        assert architecture.layer_sizes == [10, 64, 32, 1]
        assert len(architecture.activation_functions) == 3
        assert architecture.dropout_rates == [0.1, 0.2, 0.0]
        assert architecture.batch_normalization is True

    def test_default_dropout_rates(self):
        """Test default dropout rates are set correctly"""
        architecture = NetworkArchitecture(
            layer_sizes=[5, 10, 1],
            activation_functions=[ActivationFunction.RELU, ActivationFunction.SIGMOID],
        )

        assert architecture.dropout_rates == [0.0, 0.0]

    def test_invalid_layer_count(self):
        """Test validation of minimum layer count"""
        with pytest.raises(
            ValueError, match="Network must have at least input and output layers"
        ):
            NetworkArchitecture(
                layer_sizes=[10], activation_functions=[]  # Only one layer
            )

    def test_mismatched_activation_functions(self):
        """Test validation of activation function count"""
        with pytest.raises(
            ValueError,
            match="Number of activation functions must equal number of layer transitions",
        ):
            NetworkArchitecture(
                layer_sizes=[10, 20, 1],
                activation_functions=[ActivationFunction.RELU],  # Should be 2
            )

    def test_invalid_dropout_rate(self):
        """Test validation of dropout rate range"""
        with pytest.raises(
            ValueError, match="Dropout rate must be between 0.0 and 1.0"
        ):
            NetworkArchitecture(
                layer_sizes=[10, 20, 1],
                activation_functions=[
                    ActivationFunction.RELU,
                    ActivationFunction.SIGMOID,
                ],
                dropout_rates=[0.5, 1.5],  # Invalid rate > 1.0
            )

    def test_serialization(self):
        """Test to_dict and from_dict methods"""
        architecture = NetworkArchitecture(
            layer_sizes=[10, 64, 32, 1],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.RELU,
                ActivationFunction.SIGMOID,
            ],
            dropout_rates=[0.1, 0.2, 0.0],
            batch_normalization=True,
            residual_connections=True,
        )

        # Test serialization
        data = architecture.to_dict()
        assert data["layer_sizes"] == [10, 64, 32, 1]
        assert data["activation_functions"] == ["tanh", "relu", "sigmoid"]

        # Test deserialization
        restored_architecture = NetworkArchitecture.from_dict(data)
        assert restored_architecture.layer_sizes == architecture.layer_sizes
        assert (
            restored_architecture.activation_functions
            == architecture.activation_functions
        )
        assert (
            restored_architecture.batch_normalization
            == architecture.batch_normalization
        )


class TestPhysicsConstraint:
    """Test PhysicsConstraint data model"""

    def test_valid_navier_stokes_constraint(self):
        """Test creating a valid Navier-Stokes constraint"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.NAVIER_STOKES,
            equation_parameters={"viscosity": 1e-6, "density": 1000.0},
            weight=1.5,
            tolerance=1e-8,
        )

        assert constraint.constraint_type == PhysicsConstraintType.NAVIER_STOKES
        assert constraint.equation_parameters["viscosity"] == 1e-6
        assert constraint.weight == 1.5
        assert constraint.tolerance == 1e-8

    def test_missing_required_parameters(self):
        """Test validation of required equation parameters"""
        with pytest.raises(
            ValueError,
            match="Missing required parameters for PhysicsConstraintType.NAVIER_STOKES",
        ):
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.NAVIER_STOKES,
                equation_parameters={"viscosity": 1e-6},  # Missing density
            )

    def test_invalid_weight(self):
        """Test validation of constraint weight"""
        with pytest.raises(ValueError, match="Constraint weight must be positive"):
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.MASS_CONSERVATION,
                equation_parameters={},
                weight=-1.0,  # Invalid negative weight
            )

    def test_invalid_tolerance(self):
        """Test validation of tolerance"""
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            PhysicsConstraint(
                constraint_type=PhysicsConstraintType.MASS_CONSERVATION,
                equation_parameters={},
                tolerance=0.0,  # Invalid zero tolerance
            )

    def test_heat_transfer_constraint(self):
        """Test heat transfer constraint with required parameters"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.HEAT_TRANSFER,
            equation_parameters={
                "thermal_conductivity": 2.5,
                "specific_heat": 1000.0,
                "density": 2700.0,
            },
        )

        assert constraint.constraint_type == PhysicsConstraintType.HEAT_TRANSFER
        assert len(constraint.equation_parameters) == 3

    def test_serialization(self):
        """Test constraint serialization"""
        constraint = PhysicsConstraint(
            constraint_type=PhysicsConstraintType.ELASTIC_DEFORMATION,
            equation_parameters={"youngs_modulus": 2e11, "poisson_ratio": 0.3},
            weight=2.0,
            boundary_conditions={"type": "fixed", "value": 0.0},
            domain_bounds={"x": (0, 100), "y": (0, 50)},
        )

        # Test serialization
        data = constraint.to_dict()
        assert data["constraint_type"] == "elastic_deformation"
        assert data["boundary_conditions"]["type"] == "fixed"

        # Test deserialization
        restored_constraint = PhysicsConstraint.from_dict(data)
        assert restored_constraint.constraint_type == constraint.constraint_type
        assert restored_constraint.equation_parameters == constraint.equation_parameters


class TestPINNModelConfig:
    """Test PINNModelConfig data model"""

    def create_valid_architecture(self) -> NetworkArchitecture:
        """Helper to create valid architecture"""
        return NetworkArchitecture(
            layer_sizes=[10, 64, 32, 1],
            activation_functions=[
                ActivationFunction.TANH,
                ActivationFunction.TANH,
                ActivationFunction.SIGMOID,
            ],
        )

    def create_valid_constraint(self) -> PhysicsConstraint:
        """Helper to create valid physics constraint"""
        return PhysicsConstraint(
            constraint_type=PhysicsConstraintType.MASS_CONSERVATION,
            equation_parameters={},
        )

    def test_valid_pinn_config(self):
        """Test creating a valid PINN configuration"""
        architecture = self.create_valid_architecture()
        constraint = self.create_valid_constraint()

        config = PINNModelConfig(
            model_id="test_pinn_001",
            network_architecture=architecture,
            physics_constraints=[constraint],
            optimization_config={
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
            },
            training_config={
                "max_epochs": 1000,
                "batch_size": 32,
                "validation_split": 0.2,
            },
        )

        assert config.model_id == "test_pinn_001"
        assert len(config.physics_constraints) == 1
        assert config.optimization_config["learning_rate"] == 0.001
        assert config.training_config["batch_size"] == 32

    def test_empty_model_id(self):
        """Test validation of empty model ID"""
        architecture = self.create_valid_architecture()
        constraint = self.create_valid_constraint()

        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            PINNModelConfig(
                model_id="",  # Empty ID
                network_architecture=architecture,
                physics_constraints=[constraint],
                optimization_config={"optimizer": "adam", "learning_rate": 0.001},
                training_config={"max_epochs": 1000, "batch_size": 32},
            )

    def test_no_physics_constraints(self):
        """Test validation of physics constraints requirement"""
        architecture = self.create_valid_architecture()

        with pytest.raises(
            ValueError, match="PINN models must have at least one physics constraint"
        ):
            PINNModelConfig(
                model_id="test_pinn_001",
                network_architecture=architecture,
                physics_constraints=[],  # Empty constraints
                optimization_config={"optimizer": "adam", "learning_rate": 0.001},
                training_config={"max_epochs": 1000, "batch_size": 32},
            )

    def test_missing_optimization_params(self):
        """Test validation of required optimization parameters"""
        architecture = self.create_valid_architecture()
        constraint = self.create_valid_constraint()

        with pytest.raises(
            ValueError, match="Missing required optimization parameters"
        ):
            PINNModelConfig(
                model_id="test_pinn_001",
                network_architecture=architecture,
                physics_constraints=[constraint],
                optimization_config={"optimizer": "adam"},  # Missing learning_rate
                training_config={"max_epochs": 1000, "batch_size": 32},
            )

    def test_invalid_learning_rate(self):
        """Test validation of learning rate range"""
        architecture = self.create_valid_architecture()
        constraint = self.create_valid_constraint()

        with pytest.raises(ValueError, match="Learning rate must be between 0 and 1"):
            PINNModelConfig(
                model_id="test_pinn_001",
                network_architecture=architecture,
                physics_constraints=[constraint],
                optimization_config={
                    "optimizer": "adam",
                    "learning_rate": 1.5,
                },  # Invalid LR
                training_config={"max_epochs": 1000, "batch_size": 32},
            )

    def test_invalid_batch_size(self):
        """Test validation of batch size"""
        architecture = self.create_valid_architecture()
        constraint = self.create_valid_constraint()

        with pytest.raises(ValueError, match="Batch size must be positive"):
            PINNModelConfig(
                model_id="test_pinn_001",
                network_architecture=architecture,
                physics_constraints=[constraint],
                optimization_config={"optimizer": "adam", "learning_rate": 0.001},
                training_config={
                    "max_epochs": 1000,
                    "batch_size": 0,
                },  # Invalid batch size
            )


class TestSimulationParameters:
    """Test SimulationParameters data model"""

    def test_valid_simulation_parameters(self):
        """Test creating valid simulation parameters"""
        initial_conditions = {
            "temperature": np.array([300.0, 350.0, 400.0]),
            "pressure": np.array([1e5, 1.5e5, 2e5]),
        }

        params = SimulationParameters(
            simulation_id="sim_001",
            initial_conditions=initial_conditions,
            time_span=(0.0, 86400.0),  # 1 day in seconds
            spatial_domain={"x": (0, 1000), "y": (0, 1000), "z": (0, 100)},
            temporal_resolution=60.0,  # 1 minute
            spatial_resolution={"x": 10.0, "y": 10.0, "z": 5.0},
            physics_models=["heat_transfer", "fluid_flow"],
            boundary_conditions={"type": "dirichlet", "value": 300.0},
            material_properties={"rock": {"density": 2700.0, "porosity": 0.1}},
        )

        assert params.simulation_id == "sim_001"
        assert params.time_span == (0.0, 86400.0)
        assert len(params.physics_models) == 2
        assert params.uncertainty_quantification is True  # Default

    def test_invalid_time_span(self):
        """Test validation of time span"""
        with pytest.raises(ValueError, match="Start time must be less than end time"):
            SimulationParameters(
                simulation_id="sim_001",
                initial_conditions={"temp": np.array([300.0])},
                time_span=(100.0, 50.0),  # Start > End
                spatial_domain={"x": (0, 100), "y": (0, 100), "z": (0, 10)},
                temporal_resolution=1.0,
                spatial_resolution={"x": 1.0, "y": 1.0, "z": 1.0},
                physics_models=["heat_transfer"],
                boundary_conditions={},
                material_properties={},
            )

    def test_missing_spatial_dimensions(self):
        """Test validation of spatial domain completeness"""
        with pytest.raises(ValueError, match="Missing spatial dimensions"):
            SimulationParameters(
                simulation_id="sim_001",
                initial_conditions={"temp": np.array([300.0])},
                time_span=(0.0, 100.0),
                spatial_domain={"x": (0, 100), "y": (0, 100)},  # Missing z
                temporal_resolution=1.0,
                spatial_resolution={"x": 1.0, "y": 1.0, "z": 1.0},
                physics_models=["heat_transfer"],
                boundary_conditions={},
                material_properties={},
            )

    def test_invalid_spatial_bounds(self):
        """Test validation of spatial bounds"""
        with pytest.raises(ValueError, match="Minimum x must be less than maximum x"):
            SimulationParameters(
                simulation_id="sim_001",
                initial_conditions={"temp": np.array([300.0])},
                time_span=(0.0, 100.0),
                spatial_domain={
                    "x": (100, 50),
                    "y": (0, 100),
                    "z": (0, 10),
                },  # Invalid x bounds
                temporal_resolution=1.0,
                spatial_resolution={"x": 1.0, "y": 1.0, "z": 1.0},
                physics_models=["heat_transfer"],
                boundary_conditions={},
                material_properties={},
            )

    def test_serialization_with_numpy_arrays(self):
        """Test serialization with numpy arrays"""
        initial_conditions = {
            "temperature": np.array([300.0, 350.0, 400.0]),
            "velocity": np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        }

        params = SimulationParameters(
            simulation_id="sim_001",
            initial_conditions=initial_conditions,
            time_span=(0.0, 100.0),
            spatial_domain={"x": (0, 100), "y": (0, 100), "z": (0, 10)},
            temporal_resolution=1.0,
            spatial_resolution={"x": 1.0, "y": 1.0, "z": 1.0},
            physics_models=["heat_transfer"],
            boundary_conditions={},
            material_properties={},
        )

        # Test serialization
        data = params.to_dict()
        assert isinstance(data["initial_conditions"]["temperature"], list)

        # Test deserialization
        restored_params = SimulationParameters.from_dict(data)
        assert isinstance(restored_params.initial_conditions["temperature"], np.ndarray)
        assert np.array_equal(
            restored_params.initial_conditions["temperature"],
            initial_conditions["temperature"],
        )


class TestUncertaintyQuantification:
    """Test UncertaintyQuantification data model"""

    def test_valid_uncertainty_quantification(self):
        """Test creating valid uncertainty quantification"""
        epistemic = np.array([0.1, 0.2, 0.15])
        aleatoric = np.array([0.05, 0.1, 0.08])
        total = np.array([0.15, 0.3, 0.23])

        uq = UncertaintyQuantification(
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            confidence_intervals={"95%": (0.05, 0.95), "68%": (0.16, 0.84)},
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
            calibration_metrics={"ECE": 0.05, "MCE": 0.12},
        )

        assert uq.method == UncertaintyMethod.MONTE_CARLO_DROPOUT
        assert len(uq.confidence_intervals) == 2
        assert np.array_equal(uq.epistemic_uncertainty, epistemic)

    def test_mismatched_uncertainty_shapes(self):
        """Test validation of uncertainty array shapes"""
        with pytest.raises(
            ValueError,
            match="Epistemic and aleatoric uncertainty arrays must have same shape",
        ):
            UncertaintyQuantification(
                method=UncertaintyMethod.ENSEMBLE_VARIANCE,
                confidence_intervals={"95%": (0.05, 0.95)},
                epistemic_uncertainty=np.array([0.1, 0.2]),
                aleatoric_uncertainty=np.array([0.05, 0.1, 0.08]),  # Different shape
                total_uncertainty=np.array([0.15, 0.3]),
            )

    def test_negative_uncertainty_values(self):
        """Test validation of non-negative uncertainty values"""
        with pytest.raises(
            ValueError, match="Epistemic uncertainty values must be non-negative"
        ):
            UncertaintyQuantification(
                method=UncertaintyMethod.BAYESIAN_NEURAL_NETWORK,
                confidence_intervals={"95%": (0.05, 0.95)},
                epistemic_uncertainty=np.array([-0.1, 0.2]),  # Negative value
                aleatoric_uncertainty=np.array([0.05, 0.1]),
                total_uncertainty=np.array([0.15, 0.3]),
            )

    def test_serialization(self):
        """Test uncertainty quantification serialization"""
        epistemic = np.array([0.1, 0.2])
        aleatoric = np.array([0.05, 0.1])
        total = np.array([0.15, 0.3])

        uq = UncertaintyQuantification(
            method=UncertaintyMethod.DEEP_ENSEMBLES,
            confidence_intervals={"95%": (0.05, 0.95)},
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total,
        )

        # Test serialization
        data = uq.to_dict()
        assert data["method"] == "deep_ensembles"
        assert isinstance(data["epistemic_uncertainty"], list)

        # Test deserialization
        restored_uq = UncertaintyQuantification.from_dict(data)
        assert restored_uq.method == uq.method
        assert np.array_equal(restored_uq.epistemic_uncertainty, epistemic)


class TestModelPrediction:
    """Test ModelPrediction data model"""

    def create_valid_uncertainty(self) -> UncertaintyQuantification:
        """Helper to create valid uncertainty quantification"""
        return UncertaintyQuantification(
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            confidence_intervals={"95%": (0.05, 0.95)},
            epistemic_uncertainty=np.array([0.1, 0.2]),
            aleatoric_uncertainty=np.array([0.05, 0.1]),
            total_uncertainty=np.array([0.15, 0.3]),
        )

    def test_valid_model_prediction(self):
        """Test creating a valid model prediction"""
        uncertainty = self.create_valid_uncertainty()
        values = np.array([1.5, 2.3])

        prediction = ModelPrediction(
            prediction_id="pred_001",
            timestamp=datetime.utcnow(),
            model_id="model_001",
            model_version="v1.0.0",
            prediction_type="regression",
            input_data_hash="abc123def456",
            values=values,
            uncertainty=uncertainty,
            confidence_score=0.85,
            metadata={"input_features": 10, "processing_time": 0.05},
        )

        assert prediction.prediction_id == "pred_001"
        assert prediction.model_id == "model_001"
        assert prediction.confidence_score == 0.85
        assert np.array_equal(prediction.values, values)

    def test_empty_prediction_id(self):
        """Test validation of empty prediction ID"""
        uncertainty = self.create_valid_uncertainty()

        with pytest.raises(ValueError, match="Prediction ID cannot be empty"):
            ModelPrediction(
                prediction_id="",  # Empty ID
                timestamp=datetime.utcnow(),
                model_id="model_001",
                model_version="v1.0.0",
                prediction_type="regression",
                input_data_hash="abc123",
                values=np.array([1.5, 2.3]),
                uncertainty=uncertainty,
                confidence_score=0.85,
            )

    def test_invalid_confidence_score(self):
        """Test validation of confidence score range"""
        uncertainty = self.create_valid_uncertainty()

        with pytest.raises(
            ValueError, match="Confidence score must be between 0 and 1"
        ):
            ModelPrediction(
                prediction_id="pred_001",
                timestamp=datetime.utcnow(),
                model_id="model_001",
                model_version="v1.0.0",
                prediction_type="regression",
                input_data_hash="abc123",
                values=np.array([1.5, 2.3]),
                uncertainty=uncertainty,
                confidence_score=1.5,  # Invalid score > 1
            )

    def test_mismatched_prediction_uncertainty_shapes(self):
        """Test validation of prediction and uncertainty shape matching"""
        # Create uncertainty with different shape
        uncertainty = UncertaintyQuantification(
            method=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            confidence_intervals={"95%": (0.05, 0.95)},
            epistemic_uncertainty=np.array([0.1, 0.2, 0.15]),  # Shape (3,)
            aleatoric_uncertainty=np.array([0.05, 0.1, 0.08]),
            total_uncertainty=np.array([0.15, 0.3, 0.23]),
        )

        with pytest.raises(
            ValueError, match="Prediction values and uncertainty must have same shape"
        ):
            ModelPrediction(
                prediction_id="pred_001",
                timestamp=datetime.utcnow(),
                model_id="model_001",
                model_version="v1.0.0",
                prediction_type="regression",
                input_data_hash="abc123",
                values=np.array([1.5, 2.3]),  # Shape (2,) - mismatch
                uncertainty=uncertainty,
                confidence_score=0.85,
            )

    def test_serialization(self):
        """Test model prediction serialization"""
        uncertainty = self.create_valid_uncertainty()
        values = np.array([1.5, 2.3])
        timestamp = datetime(2023, 1, 1, 12, 0, 0)

        prediction = ModelPrediction(
            prediction_id="pred_001",
            timestamp=timestamp,
            model_id="model_001",
            model_version="v1.0.0",
            prediction_type="regression",
            input_data_hash="abc123def456",
            values=values,
            uncertainty=uncertainty,
            confidence_score=0.85,
        )

        # Test serialization
        data = prediction.to_dict()
        assert data["timestamp"] == timestamp.isoformat()
        assert isinstance(data["values"], list)

        # Test deserialization
        restored_prediction = ModelPrediction.from_dict(data)
        assert restored_prediction.timestamp == timestamp
        assert np.array_equal(restored_prediction.values, values)


class TestModelMetadata:
    """Test ModelMetadata data model"""

    def test_valid_model_metadata(self):
        """Test creating valid model metadata"""
        creation_time = datetime.utcnow()

        metadata = ModelMetadata(
            model_id="model_001",
            version="v1.0.0",
            creation_timestamp=creation_time,
            training_data_hash="data_hash_123",
            performance_metrics={"mse": 0.05, "r2": 0.95, "mae": 0.02},
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            training_history={
                "loss": [0.1, 0.08, 0.05],
                "val_loss": [0.12, 0.09, 0.06],
            },
            deployment_status="production",
            tags=["geological", "temperature", "pinn"],
            description="Temperature prediction model for volcanic systems",
        )

        assert metadata.model_id == "model_001"
        assert metadata.version == "v1.0.0"
        assert metadata.deployment_status == "production"
        assert len(metadata.tags) == 3
        assert metadata.performance_metrics["r2"] == 0.95

    def test_invalid_deployment_status(self):
        """Test validation of deployment status"""
        with pytest.raises(ValueError, match="Deployment status must be one o"):
            ModelMetadata(
                model_id="model_001",
                version="v1.0.0",
                creation_timestamp=datetime.utcnow(),
                training_data_hash="data_hash_123",
                performance_metrics={"mse": 0.05},
                hyperparameters={"learning_rate": 0.001},
                deployment_status="invalid_status",  # Invalid status
            )

    def test_negative_performance_metric(self):
        """Test validation of non-negative performance metrics"""
        with pytest.raises(
            ValueError, match="Performance metric mse cannot be negative"
        ):
            ModelMetadata(
                model_id="model_001",
                version="v1.0.0",
                creation_timestamp=datetime.utcnow(),
                training_data_hash="data_hash_123",
                performance_metrics={"mse": -0.05},  # Negative metric
                hyperparameters={"learning_rate": 0.001},
            )


class TestModelRegistry:
    """Test ModelRegistry data model"""

    def create_valid_metadata(self, model_id: str = "model_001") -> ModelMetadata:
        """Helper to create valid model metadata"""
        return ModelMetadata(
            model_id=model_id,
            version="v1.0.0",
            creation_timestamp=datetime.utcnow(),
            training_data_hash="data_hash_123",
            performance_metrics={"mse": 0.05},
            hyperparameters={"learning_rate": 0.001},
            tags=["geological", "temperature"],
        )

    def test_valid_model_registry(self):
        """Test creating and using model registry"""
        registry = ModelRegistry(registry_id="main_registry")

        # Add models
        model1 = self.create_valid_metadata("model_001")
        model2 = self.create_valid_metadata("model_002")

        registry.add_model(model1)
        registry.add_model(model2)

        assert len(registry.models) == 2
        assert "model_001" in registry.models
        assert "model_002" in registry.models

    def test_activate_deactivate_models(self):
        """Test model activation and deactivation"""
        registry = ModelRegistry(registry_id="main_registry")
        model = self.create_valid_metadata("model_001")
        registry.add_model(model)

        # Activate model
        registry.activate_model("model_001")
        assert "model_001" in registry.active_models

        # Deactivate model
        registry.deactivate_model("model_001")
        assert "model_001" not in registry.active_models

    def test_set_default_model(self):
        """Test setting default model"""
        registry = ModelRegistry(registry_id="main_registry")
        model = self.create_valid_metadata("model_001")
        registry.add_model(model)

        registry.set_default_model("model_001")
        assert registry.default_model == "model_001"
        assert "model_001" in registry.active_models  # Should be auto-activated

    def test_activate_nonexistent_model(self):
        """Test error when activating non-existent model"""
        registry = ModelRegistry(registry_id="main_registry")

        with pytest.raises(ValueError, match="Model nonexistent not found in registry"):
            registry.activate_model("nonexistent")

    def test_get_models_by_tags(self):
        """Test filtering models by tags"""
        registry = ModelRegistry(registry_id="main_registry")

        # Create models with different tags
        model1 = ModelMetadata(
            model_id="model_001",
            version="v1.0.0",
            creation_timestamp=datetime.utcnow(),
            training_data_hash="hash1",
            performance_metrics={"mse": 0.05},
            hyperparameters={"lr": 0.001},
            tags=["geological", "temperature", "pinn"],
        )

        model2 = ModelMetadata(
            model_id="model_002",
            version="v1.0.0",
            creation_timestamp=datetime.utcnow(),
            training_data_hash="hash2",
            performance_metrics={"mse": 0.03},
            hyperparameters={"lr": 0.001},
            tags=["geological", "pressure", "cnn"],
        )

        registry.add_model(model1)
        registry.add_model(model2)

        # Filter by tags
        geological_models = registry.get_model_by_tags(["geological"])
        assert len(geological_models) == 2

        temperature_models = registry.get_model_by_tags(["geological", "temperature"])
        assert len(temperature_models) == 1
        assert temperature_models[0].model_id == "model_001"

    def test_serialization(self):
        """Test registry serialization"""
        registry = ModelRegistry(registry_id="main_registry")
        model = self.create_valid_metadata("model_001")
        registry.add_model(model)
        registry.set_default_model("model_001")

        # Test serialization
        data = registry.to_dict()
        assert data["registry_id"] == "main_registry"
        assert data["default_model"] == "model_001"
        assert "model_001" in data["models"]

        # Test deserialization
        restored_registry = ModelRegistry.from_dict(data)
        assert restored_registry.registry_id == registry.registry_id
        assert restored_registry.default_model == registry.default_model
        assert len(restored_registry.models) == 1


if __name__ == "__main__":
    pytest.main([__file__])
