"""
Prometheus AI Geodynamic Platform - Data Models

This module contains all data models for geological, AI, and monitoring systems.
"""

import torch.nn.functional as F

from .ai_models import (
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
from .geological import (
    DeformationMeasurement,
    GeochemicalSample,
    GeologicalLocation,
    MagmaReservoir,
    SeismicEvent,
    ThermalMeasurement,
)
from .validation import GeologicalDataValidator, ValidationError, ValidationResult

__all__ = [
    # Geological models
    "GeologicalLocation",
    "SeismicEvent",
    "DeformationMeasurement",
    "GeochemicalSample",
    "ThermalMeasurement",
    "MagmaReservoir",
    # AI models
    "NetworkArchitecture",
    "PhysicsConstraint",
    "PINNModelConfig",
    "SimulationParameters",
    "UncertaintyQuantification",
    "ModelPrediction",
    "ModelMetadata",
    "ModelRegistry",
    "PhysicsConstraintType",
    "ActivationFunction",
    "UncertaintyMethod",
    # Validation
    "GeologicalDataValidator",
    "ValidationError",
    "ValidationResult",
]
