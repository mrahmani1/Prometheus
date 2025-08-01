"""
AI Model Data Structures and Configuration

Data structures for Physics-Informed Neural Networks (PINNs), model configurations,
predictions, and uncertainty quantification systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class PhysicsConstraintType(str, Enum):
    """Types of physics constraints for PINNs"""

    NAVIER_STOKES = "navier_stokes"
    HEAT_TRANSFER = "heat_transfer"
    ELASTIC_DEFORMATION = "elastic_deformation"
    MASS_CONSERVATION = "mass_conservation"
    MOMENTUM_CONSERVATION = "momentum_conservation"
    ENERGY_CONSERVATION = "energy_conservation"
    CONTINUITY_EQUATION = "continuity_equation"
    DIFFUSION_EQUATION = "diffusion_equation"
    WAVE_EQUATION = "wave_equation"

class ActivationFunction(str, Enum):
    """Supported activation functions"""

    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    GELU = "gelu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"

class OptimizationType(str, Enum):
    """Optimization algorithms"""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    LBFGS = "lbfgs"

class GeologicalProcess(str, Enum):
    """Types of geological processes"""

    VOLCANIC_ERUPTION = "volcanic_eruption"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    TSUNAMI = "tsunami"
    CRUSTAL_DEFORMATION = "crustal_deformation"
    MAGMA_INTRUSION = "magma_intrusion"
    HYDROTHERMAL_CIRCULATION = "hydrothermal_circulation"
    TECTONIC_MOVEMENT = "tectonic_movement"
    MANTLE_CONVECTION = "mantle_convection"
    GEOCHEMICAL_REACTION = "geochemical_reaction"

class ModelType(str, Enum):
    """Types of AI models"""

    PINN = "physics_informed_neural_network"
    GAN = "generative_adversarial_network"
    CNN = "convolutional_neural_network"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods"""

    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    BAYESIAN_NEURAL_NETWORK = "bayesian_neural_network"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    CONFORMAL_PREDICTION = "conformal_prediction"
    DEEP_ENSEMBLES = "deep_ensembles"

@dataclass
class NetworkArchitecture:
    """
    Neural network architecture specification.

    Attributes:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        activation_functions: Activation function for each layer
        dropout_rates: Dropout rates for each layer (0.0 = no dropout)
        batch_normalization: Whether to use batch normalization
        residual_connections: Whether to use residual connections
        attention_mechanism: Whether to include attention layers
    """

    layer_sizes: List[int]
    activation_functions: List[ActivationFunction]
    dropout_rates: List[float] = field(default_factory=list)
    batch_normalization: bool = False
    residual_connections: bool = False
    attention_mechanism: bool = False

    def __post_init__(self):
        """Validate architecture parameters"""
        if len(self.layer_sizes) < 2:
            raise ValueError("Network must have at least input and output layers")

        if len(self.activation_functions) != len(self.layer_sizes) - 1:
            raise ValueError(
                "Number of activation functions must equal number of layer transitions"
            )

        if self.dropout_rates and len(self.dropout_rates) != len(self.layer_sizes) - 1:
            raise ValueError("Dropout rates must match number of layers or be empty")

        # Set default dropout rates if not provided
        if not self.dropout_rates:
            self.dropout_rates = [0.0] * (len(self.layer_sizes) - 1)

        # Validate dropout rates
        for rate in self.dropout_rates:
            if not 0.0 <= rate <= 1.0:
                raise ValueError(
                    f"Dropout rate must be between 0.0 and 1.0, got {rate}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "layer_sizes": self.layer_sizes,
            "activation_functions": [af.value for af in self.activation_functions],
            "dropout_rates": self.dropout_rates,
            "batch_normalization": self.batch_normalization,
            "residual_connections": self.residual_connections,
            "attention_mechanism": self.attention_mechanism,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NetworkArchitecture":
        """Create from dictionary"""
        return cls(
            layer_sizes=data["layer_sizes"],
            activation_functions=[
                ActivationFunction(af) for af in data["activation_functions"]
            ],
            dropout_rates=data.get("dropout_rates", []),
            batch_normalization=data.get("batch_normalization", False),
            residual_connections=data.get("residual_connections", False),
            attention_mechanism=data.get("attention_mechanism", False),
        )

@dataclass
class PhysicsConstraint:
    """
    Physics constraint specification for PINNs.

    Attributes:
        constraint_type: Type of physics constraint
        equation_parameters: Parameters for the physics equation
        weight: Weight of this constraint in the loss function
        boundary_conditions: Boundary condition specifications
        domain_bounds: Spatial/temporal domain bounds for constraint
        tolerance: Numerical tolerance for constraint satisfaction
    """

    constraint_type: PhysicsConstraintType
    equation_parameters: Dict[str, float]
    weight: float = 1.0
    boundary_conditions: Optional[Dict[str, Any]] = None
    domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    tolerance: float = 1e-6

    def __post_init__(self):
        """Validate physics constraint parameters"""
        if self.weight <= 0:
            raise ValueError(f"Constraint weight must be positive, got {self.weight}")

        if self.tolerance <= 0:
            raise ValueError(f"Tolerance must be positive, got {self.tolerance}")

        # Validate equation parameters based on constraint type
        self._validate_equation_parameters()

    def _validate_equation_parameters(self):
        """Validate equation parameters based on constraint type"""
        required_params = {
            PhysicsConstraintType.NAVIER_STOKES: ["viscosity", "density"],
            PhysicsConstraintType.HEAT_TRANSFER: [
                "thermal_conductivity",
                "specific_heat",
                "density",
            ],
            PhysicsConstraintType.ELASTIC_DEFORMATION: [
                "youngs_modulus",
                "poisson_ratio",
            ],
            PhysicsConstraintType.MASS_CONSERVATION: [],
            PhysicsConstraintType.DIFFUSION_EQUATION: ["diffusion_coefficient"],
        }

        if self.constraint_type in required_params:
            required = required_params[self.constraint_type]
            missing = [
                param for param in required if param not in self.equation_parameters
            ]
            if missing:
                raise ValueError(
                    f"Missing required parameters for {self.constraint_type}: {missing}"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "constraint_type": self.constraint_type.value,
            "equation_parameters": self.equation_parameters,
            "weight": self.weight,
            "boundary_conditions": self.boundary_conditions,
            "domain_bounds": self.domain_bounds,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhysicsConstraint":
        """Create from dictionary"""
        return cls(
            constraint_type=PhysicsConstraintType(data["constraint_type"]),
            equation_parameters=data["equation_parameters"],
            weight=data.get("weight", 1.0),
            boundary_conditions=data.get("boundary_conditions"),
            domain_bounds=data.get("domain_bounds"),
            tolerance=data.get("tolerance", 1e-6),
        )

@dataclass
class PINNModelConfig:
    """
    Configuration for Physics-Informed Neural Networks.

    Attributes:
        model_id: Unique identifier for the model
        network_architecture: Neural network architecture specification
        physics_constraints: List of physics constraints to enforce
        optimization_config: Optimization algorithm configuration
        training_config: Training parameters and settings
        uncertainty_config: Uncertainty quantification configuration
        metadata: Additional model metadata
    """

    model_id: str
    network_architecture: NetworkArchitecture
    physics_constraints: List[PhysicsConstraint]
    optimization_config: Dict[str, Any]
    training_config: Dict[str, Any]
    uncertainty_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate PINN model configuration"""
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")

        if not self.physics_constraints:
            raise ValueError("PINN models must have at least one physics constraint")

        # Validate optimization config
        required_opt_params = ["optimizer", "learning_rate"]
        missing_opt = [
            param
            for param in required_opt_params
            if param not in self.optimization_config
        ]
        if missing_opt:
            raise ValueError(f"Missing required optimization parameters: {missing_opt}")

        # Validate training config
        required_train_params = ["max_epochs", "batch_size"]
        missing_train = [
            param
            for param in required_train_params
            if param not in self.training_config
        ]
        if missing_train:
            raise ValueError(f"Missing required training parameters: {missing_train}")

        # Validate learning rate
        lr = self.optimization_config.get("learning_rate", 0)
        if not 0 < lr < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {lr}")

        # Validate batch size
        batch_size = self.training_config.get("batch_size", 0)
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_id": self.model_id,
            "network_architecture": self.network_architecture.to_dict(),
            "physics_constraints": [pc.to_dict() for pc in self.physics_constraints],
            "optimization_config": self.optimization_config,
            "training_config": self.training_config,
            "uncertainty_config": self.uncertainty_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PINNModelConfig":
        """Create from dictionary"""
        return cls(
            model_id=data["model_id"],
            network_architecture=NetworkArchitecture.from_dict(
                data["network_architecture"]
            ),
            physics_constraints=[
                PhysicsConstraint.from_dict(pc) for pc in data["physics_constraints"]
            ],
            optimization_config=data["optimization_config"],
            training_config=data["training_config"],
            uncertainty_config=data.get("uncertainty_config"),
            metadata=data.get("metadata"),
        )

@dataclass
class SimulationParameters:
    """
    Parameters for geological simulations.

    Attributes:
        simulation_id: Unique identifier for the simulation
        initial_conditions: Initial state variables and values
        time_span: Simulation time range (start_time, end_time)
        spatial_domain: Spatial bounds {x: (min, max), y: (min, max), z: (min, max)}
        temporal_resolution: Time step size in seconds
        spatial_resolution: Spatial grid resolution in meters
        physics_models: List of active physics models
        boundary_conditions: Boundary condition specifications
        material_properties: Material property definitions
        uncertainty_quantification: Whether to include uncertainty analysis
    """

    simulation_id: str
    initial_conditions: Dict[str, np.ndarray]
    time_span: Tuple[float, float]
    spatial_domain: Dict[str, Tuple[float, float]]
    temporal_resolution: float
    spatial_resolution: Dict[str, float]
    physics_models: List[str]
    boundary_conditions: Dict[str, Any]
    material_properties: Dict[str, Dict[str, float]]
    uncertainty_quantification: bool = True

    def __post_init__(self):
        """Validate simulation parameters"""
        if not self.simulation_id:
            raise ValueError("Simulation ID cannot be empty")

        # Validate time span
        if len(self.time_span) != 2:
            raise ValueError("Time span must contain exactly 2 values (start, end)")

        start_time, end_time = self.time_span
        if start_time >= end_time:
            raise ValueError("Start time must be less than end time")

        # Validate temporal resolution
        if self.temporal_resolution <= 0:
            raise ValueError(
                f"Temporal resolution must be positive, got {self.temporal_resolution}"
            )

        # Validate spatial domain
        required_dims = ["x", "y", "z"]
        missing_dims = [dim for dim in required_dims if dim not in self.spatial_domain]
        if missing_dims:
            raise ValueError(f"Missing spatial dimensions: {missing_dims}")

        # Validate spatial bounds
        for dim, (min_val, max_val) in self.spatial_domain.items():
            if min_val >= max_val:
                raise ValueError(f"Minimum {dim} must be less than maximum {dim}")

        # Validate spatial resolution
        for dim, resolution in self.spatial_resolution.items():
            if resolution <= 0:
                raise ValueError(
                    f"Spatial resolution for {dim} must be positive, got {resolution}"
                )

        # Validate physics models
        if not self.physics_models:
            raise ValueError("At least one physics model must be specified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Convert numpy arrays to lists for JSON serialization
        initial_conditions_serializable = {}
        for key, value in self.initial_conditions.items():
            if isinstance(value, np.ndarray):
                initial_conditions_serializable[key] = value.tolist()
            else:
                initial_conditions_serializable[key] = value

        return {
            "simulation_id": self.simulation_id,
            "initial_conditions": initial_conditions_serializable,
            "time_span": self.time_span,
            "spatial_domain": self.spatial_domain,
            "temporal_resolution": self.temporal_resolution,
            "spatial_resolution": self.spatial_resolution,
            "physics_models": self.physics_models,
            "boundary_conditions": self.boundary_conditions,
            "material_properties": self.material_properties,
            "uncertainty_quantification": self.uncertainty_quantification,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationParameters":
        """Create from dictionary"""
        # Convert lists back to numpy arrays
        initial_conditions = {}
        for key, value in data["initial_conditions"].items():
            if isinstance(value, list):
                initial_conditions[key] = np.array(value)
            else:
                initial_conditions[key] = value

        return cls(
            simulation_id=data["simulation_id"],
            initial_conditions=initial_conditions,
            time_span=tuple(data["time_span"]),
            spatial_domain=data["spatial_domain"],
            temporal_resolution=data["temporal_resolution"],
            spatial_resolution=data["spatial_resolution"],
            physics_models=data["physics_models"],
            boundary_conditions=data["boundary_conditions"],
            material_properties=data["material_properties"],
            uncertainty_quantification=data.get("uncertainty_quantification", True),
        )

@dataclass
class UncertaintyQuantification:
    """
    Uncertainty quantification results and metadata.

    Attributes:
        method: Uncertainty quantification method used
        confidence_intervals: Confidence intervals for predictions
        epistemic_uncertainty: Model uncertainty (reducible with more data)
        aleatoric_uncertainty: Data uncertainty (irreducible noise)
        total_uncertainty: Combined uncertainty
        calibration_metrics: Uncertainty calibration quality metrics
    """

    method: UncertaintyMethod
    confidence_intervals: Dict[str, Tuple[float, float]]
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    total_uncertainty: np.ndarray
    calibration_metrics: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate uncertainty quantification data"""
        # Check array shapes match
        if self.epistemic_uncertainty.shape != self.aleatoric_uncertainty.shape:
            raise ValueError(
                "Epistemic and aleatoric uncertainty arrays must have same shape"
            )

        if self.epistemic_uncertainty.shape != self.total_uncertainty.shape:
            raise ValueError("Uncertainty arrays must have same shape")

        # Check uncertainty values are non-negative
        if np.any(self.epistemic_uncertainty < 0):
            raise ValueError("Epistemic uncertainty values must be non-negative")

        if np.any(self.aleatoric_uncertainty < 0):
            raise ValueError("Aleatoric uncertainty values must be non-negative")

        if np.any(self.total_uncertainty < 0):
            raise ValueError("Total uncertainty values must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "method": self.method.value,
            "confidence_intervals": self.confidence_intervals,
            "epistemic_uncertainty": self.epistemic_uncertainty.tolist(),
            "aleatoric_uncertainty": self.aleatoric_uncertainty.tolist(),
            "total_uncertainty": self.total_uncertainty.tolist(),
            "calibration_metrics": self.calibration_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UncertaintyQuantification":
        """Create from dictionary"""
        return cls(
            method=UncertaintyMethod(data["method"]),
            confidence_intervals=data["confidence_intervals"],
            epistemic_uncertainty=np.array(data["epistemic_uncertainty"]),
            aleatoric_uncertainty=np.array(data["aleatoric_uncertainty"]),
            total_uncertainty=np.array(data["total_uncertainty"]),
            calibration_metrics=data.get("calibration_metrics"),
        )

@dataclass
class ModelPrediction:
    """
    AI model prediction with uncertainty quantification.

    Attributes:
        prediction_id: Unique identifier for the prediction
        timestamp: When the prediction was made
        model_id: ID of the model that made the prediction
        model_version: Version of the model
        prediction_type: Type of prediction (classification, regression, etc.)
        input_data_hash: Hash of input data for reproducibility
        values: Predicted values
        uncertainty: Uncertainty quantification results
        confidence_score: Overall confidence in prediction (0-1)
        metadata: Additional prediction metadata
    """

    prediction_id: str
    timestamp: datetime
    model_id: str
    model_version: str
    prediction_type: str
    input_data_hash: str
    values: np.ndarray
    uncertainty: UncertaintyQuantification
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate model prediction data"""
        if not self.prediction_id:
            raise ValueError("Prediction ID cannot be empty")

        if not self.model_id:
            raise ValueError("Model ID cannot be empty")

        if not self.model_version:
            raise ValueError("Model version cannot be empty")

        if not 0 <= self.confidence_score <= 1:
            raise ValueError(
                f"Confidence score must be between 0 and 1, got {self.confidence_score}"
            )

        if not self.input_data_hash:
            raise ValueError("Input data hash cannot be empty")

        # Check that uncertainty shape matches prediction values
        if self.values.shape != self.uncertainty.total_uncertainty.shape:
            raise ValueError("Prediction values and uncertainty must have same shape")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat(),
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_type": self.prediction_type,
            "input_data_hash": self.input_data_hash,
            "values": self.values.tolist(),
            "uncertainty": self.uncertainty.to_dict(),
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPrediction":
        """Create from dictionary"""
        return cls(
            prediction_id=data["prediction_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_id=data["model_id"],
            model_version=data["model_version"],
            prediction_type=data["prediction_type"],
            input_data_hash=data["input_data_hash"],
            values=np.array(data["values"]),
            uncertainty=UncertaintyQuantification.from_dict(data["uncertainty"]),
            confidence_score=data["confidence_score"],
            metadata=data.get("metadata"),
        )

@dataclass
class ModelMetadata:
    """
    Model versioning and metadata tracking.

    Attributes:
        model_id: Unique model identifier
        version: Model version string
        creation_timestamp: When the model was created
        training_data_hash: Hash of training data
        performance_metrics: Model performance on validation data
        hyperparameters: Model hyperparameters
        training_history: Training loss and metrics history
        deployment_status: Current deployment status
        tags: Searchable tags for model organization
        description: Human-readable model description
    """

    model_id: str
    version: str
    creation_timestamp: datetime
    training_data_hash: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_history: Optional[Dict[str, List[float]]] = None
    deployment_status: str = "development"
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None

    def __post_init__(self):
        """Validate model metadata"""
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")

        if not self.version:
            raise ValueError("Model version cannot be empty")

        if not self.training_data_hash:
            raise ValueError("Training data hash cannot be empty")

        valid_statuses = [
            "development",
            "testing",
            "staging",
            "production",
            "deprecated",
        ]
        if self.deployment_status not in valid_statuses:
            raise ValueError(f"Deployment status must be one of {valid_statuses}")

        # Validate performance metrics are non-negative
        for metric, value in self.performance_metrics.items():
            if value < 0:
                raise ValueError(f"Performance metric {metric} cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "training_data_hash": self.training_data_hash,
            "performance_metrics": self.performance_metrics,
            "hyperparameters": self.hyperparameters,
            "training_history": self.training_history,
            "deployment_status": self.deployment_status,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary"""
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            creation_timestamp=datetime.fromisoformat(data["creation_timestamp"]),
            training_data_hash=data["training_data_hash"],
            performance_metrics=data["performance_metrics"],
            hyperparameters=data["hyperparameters"],
            training_history=data.get("training_history"),
            deployment_status=data.get("deployment_status", "development"),
            tags=data.get("tags", []),
            description=data.get("description"),
        )

@dataclass
class ModelRegistry:
    """
    Registry for managing multiple AI models and their configurations.

    Attributes:
        registry_id: Unique registry identifier
        models: Dictionary of model_id -> ModelMetadata
        active_models: List of currently active model IDs
        default_model: Default model ID for predictions
        last_updated: When the registry was last updated
    """

    registry_id: str
    models: Dict[str, ModelMetadata] = field(default_factory=dict)
    active_models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def add_model(self, metadata: ModelMetadata):
        """Add a model to the registry"""
        self.models[metadata.model_id] = metadata
        self.last_updated = datetime.utcnow()

    def activate_model(self, model_id: str):
        """Activate a model for use"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        if model_id not in self.active_models:
            self.active_models.append(model_id)

        self.last_updated = datetime.utcnow()

    def deactivate_model(self, model_id: str):
        """Deactivate a model"""
        if model_id in self.active_models:
            self.active_models.remove(model_id)

        if self.default_model == model_id:
            self.default_model = None

        self.last_updated = datetime.utcnow()

    def set_default_model(self, model_id: str):
        """Set the default model for predictions"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found in registry")

        self.default_model = model_id
        self.activate_model(model_id)  # Ensure default model is active

    def get_model_by_tags(self, tags: List[str]) -> List[ModelMetadata]:
        """Get models that have all specified tags"""
        matching_models = []
        for model in self.models.values():
            if all(tag in model.tags for tag in tags):
                matching_models.append(model)
        return matching_models

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "registry_id": self.registry_id,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "active_models": self.active_models,
            "default_model": self.default_model,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelRegistry":
        """Create from dictionary"""
        registry = cls(
            registry_id=data["registry_id"],
            active_models=data.get("active_models", []),
            default_model=data.get("default_model"),
            last_updated=datetime.fromisoformat(
                data.get("last_updated", datetime.utcnow().isoformat())
            ),
        )

        # Load models
        for model_id, model_data in data.get("models", {}).items():
            registry.models[model_id] = ModelMetadata.from_dict(model_data)

        return registry
