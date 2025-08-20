"""
Prometheus Backend Modules
Advanced 3D Geological Modeling System - Modular Architecture

This package contains the modular components of the advanced geological modeling system:

- geological_data_structures: Core data classes and structures
- geological_statistics: Advanced statistical analysis methods
- core_geological_modeler: Main geological modeling functionality  
- geological_constraints: Constraint satisfaction and ML optimization
- fracture_network_analysis: Fracture network modeling (to be implemented)
- implicit_surface_modeling: Surface modeling and marching cubes (to be implemented)
- uncertainty_analysis: Uncertainty quantification (to be implemented)
"""

# Core imports
try:
    # Try relative imports first (when used as package)
    from .geological_data_structures import (
        DrillholeData,
        StructuralFeature, 
        LithologyUnit,
        GeologicalSample,
        GeologicalModel,
        GeologicalDatabase,
        create_drillhole_from_survey,
        create_structural_feature_from_points,
        create_geological_model
    )

    from .geological_statistics import (
        CircularStatistics,
        HypothesisTesting,
        OrientationAnalysis,
        StabilityAnalysis,
        calculate_skewness,
        calculate_kurtosis
    )

    from .core_geological_modeler import (
        VizierGaussianProcessRegressor,
        GeologicalModeler,
        AdvancedGeologicalModeler,
        create_example_geological_model
    )

    from .geological_constraints import (
        GeologicalConstraints,
        MLConstraintOptimizer,
        AdaptiveConstraintSystem,
        create_example_constraint_system
    )
except ImportError:
    # Fallback to absolute imports (when used as standalone)
    from geological_data_structures import (
        DrillholeData,
        StructuralFeature, 
        LithologyUnit,
        GeologicalSample,
        GeologicalModel,
        GeologicalDatabase,
        create_drillhole_from_survey,
        create_structural_feature_from_points,
        create_geological_model
    )

    from geological_statistics import (
        CircularStatistics,
        HypothesisTesting,
        OrientationAnalysis,
        StabilityAnalysis,
        calculate_skewness,
        calculate_kurtosis
    )

    from core_geological_modeler import (
        VizierGaussianProcessRegressor,
        GeologicalModeler,
        AdvancedGeologicalModeler,
        create_example_geological_model
    )

    from geological_constraints import (
        GeologicalConstraints,
        MLConstraintOptimizer,
        AdaptiveConstraintSystem,
        create_example_constraint_system
    )

# Version information
__version__ = "2.0.0"
__author__ = "Advanced Geological Modeling Team"
__description__ = "Advanced 3D Geological Modeling System with ML and Statistical Analysis"

# Module metadata
__all__ = [
    # Data structures
    'DrillholeData',
    'StructuralFeature', 
    'LithologyUnit',
    'GeologicalSample',
    'GeologicalModel',
    'GeologicalDatabase',
    
    # Statistics
    'CircularStatistics',
    'HypothesisTesting', 
    'OrientationAnalysis',
    'StabilityAnalysis',
    
    # Core modeling
    'VizierGaussianProcessRegressor',
    'GeologicalModeler',
    'AdvancedGeologicalModeler',
    
    # Constraints
    'GeologicalConstraints',
    'MLConstraintOptimizer',
    'AdaptiveConstraintSystem',
    
    # Utilities
    'create_drillhole_from_survey',
    'create_structural_feature_from_points', 
    'create_geological_model',
    'create_example_geological_model',
    'create_example_constraint_system',
    'calculate_skewness',
    'calculate_kurtosis'
]

# Package-level configuration
import warnings
import logging

# Configure logging for the package
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_package_info():
    """Get information about the geological modeling package"""
    return {
        'version': __version__,
        'author': __author__, 
        'description': __description__,
        'modules': [
            'geological_data_structures',
            'geological_statistics', 
            'core_geological_modeler',
            'geological_constraints'
        ],
        'capabilities': [
            'Advanced orientation statistics',
            'Multi-scale stability analysis',
            'ML-based constraint optimization',
            'Gaussian Process modeling',
            'Comprehensive data quality assessment',
            'Regional constraint adaptation'
        ]
    }

def create_complete_example_model():
    """Create a complete example geological model with all components"""
    logger.info("Creating complete example geological model...")
    
    # Create advanced modeler
    modeler = create_example_geological_model()
    
    # Add constraint system
    constraint_system = create_example_constraint_system()
    
    # Generate comprehensive analysis
    orientation_analysis = modeler.analyze_structural_orientations()
    stability_analysis = modeler.assess_data_stability()
    model_stats = modeler.calculate_model_statistics()
    
    example_model = {
        'modeler': modeler,
        'constraint_system': constraint_system,
        'analyses': {
            'orientation_analysis': orientation_analysis,
            'stability_analysis': stability_analysis,
            'model_statistics': model_stats
        },
        'package_info': get_package_info()
    }
    
    logger.info("Complete example model created successfully")
    return example_model

# Package initialization message
logger.info(f"Prometheus Geological Modeling Package v{__version__} initialized")
logger.info(f"Available modules: {', '.join(__all__[:6])}...")

def run_package_tests():
    """Run basic package functionality tests"""
    logger.info("Running package functionality tests...")
    
    try:
        # Test data structures
        dh = DrillholeData('TEST001', 0, 0, 0, 100, 0, -90)
        assert dh.hole_id == 'TEST001'
        logger.info("✓ Data structures test passed")
        
        # Test statistics
        stats = CircularStatistics()
        test_angles = [0, 45, 90, 135]
        result = stats.compute_advanced_circular_statistics(np.array(test_angles) * np.pi / 180)
        assert 'mean_direction_rad' in result
        logger.info("✓ Statistics test passed")
        
        # Test modeler
        modeler = GeologicalModeler()
        assert hasattr(modeler, 'analyze_structural_orientations')
        logger.info("✓ Core modeler test passed")
        
        # Test constraints
        constraints = GeologicalConstraints()
        validation = constraints.validate_orientation('fault', 45, 60)
        assert 'is_valid' in validation
        logger.info("✓ Constraints test passed")
        
        logger.info("All package tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Package test failed: {e}")
        return False

# Optional: Run tests on import (can be disabled)
if __name__ == "__main__":
    run_package_tests()