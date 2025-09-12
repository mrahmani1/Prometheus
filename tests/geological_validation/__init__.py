"""
Geological Validation Testing Framework
======================================

Comprehensive validation testing framework for geological algorithms and models
using real geological data and production-grade algorithms.

This framework validates:
- Physics-informed neural networks against known geological phenomena
- Seismic analysis accuracy using historical earthquake data
- Volcanic prediction models with real eruption datasets
- Geochemical analysis with laboratory-verified samples
- 3D geological modeling against ground-truth surveys
- Data integrity and consistency with production-scale datasets

All validation uses real geological data - no synthetic or mock data allowed.
"""

from .benchmark_datasets import BenchmarkDatasetManager
from .geological_accuracy_validator import GeologicalAccuracyValidator
from .seismic_validation import SeismicValidationSuite
from .volcanic_validation import VolcanicValidationSuite
from .geochemical_validation import GeochemicalValidationSuite
from .visualization_validation import VisualizationValidationSuite
from .data_integrity_validator import DataIntegrityValidator
from .continuous_validation import ContinuousValidationFramework

__all__ = [
    'BenchmarkDatasetManager',
    'GeologicalAccuracyValidator',
    'SeismicValidationSuite',
    'VolcanicValidationSuite',
    'GeochemicalValidationSuite',
    'VisualizationValidationSuite',
    'DataIntegrityValidator',
    'ContinuousValidationFramework'
]

__version__ = "1.0.0"