#!/usr/bin/env python3
"""
Demonstration of the Modular Geological Modeling System
Shows how to use the split modules effectively
"""

import numpy as np
import sys
from pathlib import Path

# Add the modules to the path
sys.path.append(str(Path(__file__).parent))

# Import the modular system
from geological_data_structures import (
    DrillholeData, StructuralFeature, LithologyUnit, 
    create_geological_model
)
from geological_statistics import OrientationAnalysis, StabilityAnalysis
from core_geological_modeler import AdvancedGeologicalModeler
from geological_constraints import AdaptiveConstraintSystem

def demonstrate_modular_system():
    """Comprehensive demonstration of the modular geological system"""
    
    print("=" * 60)
    print("MODULAR GEOLOGICAL MODELING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # 1. Create geological data structures
    print("\n1. Creating Geological Data Structures...")
    
    # Create drillhole data
    drillhole_1 = DrillholeData(
        hole_id='DH-001',
        collar_x=1000.0,
        collar_y=2000.0,
        collar_z=300.0,
        end_of_hole=200.0,
        azimuth=45.0,
        dip=-60.0
    )
    
    drillhole_2 = DrillholeData(
        hole_id='DH-002', 
        collar_x=1100.0,
        collar_y=2100.0,
        collar_z=290.0,
        end_of_hole=180.0,
        azimuth=50.0,
        dip=-65.0
    )
    
    print(f"   [OK] Created drillholes: {drillhole_1.hole_id}, {drillhole_2.hole_id}")
    
    # Create structural features
    fault_points = np.array([
        [1000, 2000, 250],
        [1050, 2050, 240],
        [1100, 2100, 230],
        [1150, 2150, 220]
    ])
    
    fault_feature = StructuralFeature(
        feature_id='FAULT-001',
        feature_type='fault',
        points=fault_points
    )
    
    # Calculate best-fit plane
    fault_feature.calculate_best_fit_plane()
    
    print(f"   [OK] Created fault: {fault_feature.feature_id}")
    print(f"     Strike: {fault_feature.strike:.1f} deg, Dip: {fault_feature.dip:.1f} deg")
    
    # Create lithology unit
    granite_unit = LithologyUnit(
        unit_id='GRANITE-001',
        unit_name='Biotite Granite',
        rock_type='igneous',
        age='Devonian',
        formation='Granite Mountain Formation'
    )
    
    print(f"   [OK] Created lithology: {granite_unit.unit_name}")
    
    # 2. Advanced Statistical Analysis
    print("\n2. Advanced Statistical Analysis...")
    
    orientation_analyzer = OrientationAnalysis()
    
    # Sample strike measurements for analysis
    strike_measurements = [45, 50, 48, 52, 47, 49, 46, 51, 53, 44]
    
    # Perform circular statistics analysis
    circular_analysis = orientation_analyzer.analyze_circular_dispersion(strike_measurements)
    
    print(f"   [OK] Circular Statistics Analysis:")
    print(f"     Mean Strike: {circular_analysis['circular_statistics']['mean_direction_deg']:.1f} deg")
    print(f"     Dispersion: {circular_analysis['dispersion_degrees']:.1f} deg")
    print(f"     Analysis Quality: {circular_analysis['analysis_quality']}")
    print(f"     Best Distribution: {circular_analysis['distribution_analysis']['best_model']}")
    
    # Stability analysis
    stability_analyzer = StabilityAnalysis()
    
    # Create sample point cloud for stability analysis
    sample_points = np.random.normal([1075, 2075, 235], [25, 25, 10], (15, 3))
    
    stability_result = stability_analyzer.assess_orientation_stability(sample_points)
    
    print(f"   ✓ Stability Analysis:")
    print(f"     Overall Stability: {stability_result['overall_stability']:.3f}")
    print(f"     Stability Grade: {stability_result['stability_grade']['grade']}")
    
    # 3. Core Geological Modeling
    print("\n3. Core Geological Modeling...")
    
    # Create advanced modeler
    modeler = AdvancedGeologicalModeler()
    
    # Add data to modeler
    modeler.add_drillhole_data(drillhole_1)
    modeler.add_drillhole_data(drillhole_2)
    modeler.add_structural_feature(fault_feature)
    modeler.add_lithology_unit(granite_unit)
    
    print(f"   ✓ Added data to modeler:")
    print(f"     Drillholes: {len(modeler.drillholes)}")
    print(f"     Structural Features: {len(modeler.structural_features)}")
    print(f"     Lithology Units: {len(modeler.lithology_units)}")
    
    # Perform comprehensive analysis
    model_stats = modeler.calculate_model_statistics()
    
    print(f"   ✓ Model Statistics:")
    if 'spatial_extent' in model_stats:
        extent = model_stats['spatial_extent']
        print(f"     Spatial Extent: {extent['dimensions'][0]:.0f} x {extent['dimensions'][1]:.0f} x {extent['dimensions'][2]:.0f} m")
        print(f"     Model Volume: {extent['volume']:.0f} m³")
    
    print(f"     Data Quality: {model_stats['data_quality']['overall_quality_grade']}")
    
    # 4. Advanced Constraint System
    print("\n4. Advanced Constraint System...")
    
    constraint_system = AdaptiveConstraintSystem()
    
    # Test constraint validation
    validation_result = constraint_system.base_constraints.validate_orientation(
        'fault', fault_feature.strike, fault_feature.dip
    )
    
    print(f"   ✓ Constraint Validation:")
    print(f"     Fault Orientation Valid: {validation_result['is_valid']}")
    print(f"     Confidence: {validation_result['confidence']:.3f}")
    if validation_result['recommendations']:
        print(f"     Recommendations: {validation_result['recommendations'][0][:50]}...")
    
    # Regional adaptation example
    regional_data = {
        'faults': [
            {'dip': fault_feature.dip, 'strike': fault_feature.strike, 'type': 'normal'}
        ]
    }
    
    adaptation_result = constraint_system.adapt_constraints_to_region(
        'demo_region', regional_data
    )
    
    print(f"   ✓ Regional Adaptation:")
    print(f"     Stress Regime: {adaptation_result['stress_regime']}")
    print(f"     Adaptation Successful: {adaptation_result['adaptation_successful']}")
    
    # 5. Gaussian Process Modeling (if data available)
    print("\n5. Gaussian Process Modeling...")
    
    # Create synthetic training data for GP
    np.random.seed(42)
    n_points = 20
    X_train = np.random.uniform([950, 1950, 200], [1200, 2200, 300], (n_points, 3))
    y_train = np.random.normal(0, 1, n_points)  # Synthetic property values
    
    # Fit GP model
    gp_result = modeler.fit_gaussian_process_model(X_train, y_train, kernel_type='matern52')
    
    print(f"   ✓ Gaussian Process Model:")
    print(f"     Kernel Type: {gp_result['kernel_type']}")
    print(f"     Training Points: {gp_result['n_training_points']}")
    print(f"     Cross-validation R²: {gp_result['cross_validation']['mean_r2']:.3f}")
    
    # Make predictions with uncertainty
    X_query = np.array([[1050, 2050, 250], [1075, 2075, 235]])
    predictions = modeler.predict_with_uncertainty(X_query)
    
    print(f"   ✓ Predictions with Uncertainty:")
    for i, point in enumerate(X_query):
        mean_pred = predictions['mean_prediction'][i]
        std_pred = predictions['std_prediction'][i]
        print(f"     Point {i+1}: {mean_pred:.3f} ± {std_pred:.3f}")
    
    # 6. Generate Comprehensive Report
    print("\n6. Generating Comprehensive Report...")
    
    comprehensive_report = modeler.generate_comprehensive_report()
    
    print(f"   ✓ Report Generated:")
    print(f"     Report Timestamp: {comprehensive_report['timestamp']}")
    print(f"     Model Version: {comprehensive_report['model_version']}")
    print(f"     Total Sections: {len(comprehensive_report.keys())}")
    
    # Display key metrics
    if 'statistical_analysis' in comprehensive_report:
        stats_analysis = comprehensive_report['statistical_analysis']
        if 'model_statistics' in stats_analysis:
            overall_quality = stats_analysis['model_statistics']['data_quality']['overall_quality_score']
            print(f"     Overall Data Quality Score: {overall_quality:.3f}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nKey Benefits of Modular Architecture:")
    print("• Clear separation of concerns")
    print("• Easy maintenance and testing") 
    print("• Reusable components")
    print("• Scalable design")
    print("• Advanced statistical capabilities")
    print("• Machine learning integration")
    print("• Comprehensive uncertainty quantification")
    
    return {
        'modeler': modeler,
        'constraint_system': constraint_system,
        'comprehensive_report': comprehensive_report
    }

def demonstrate_individual_modules():
    """Demonstrate individual module capabilities"""
    
    print("\n" + "=" * 40)
    print("INDIVIDUAL MODULE DEMONSTRATIONS")
    print("=" * 40)
    
    # Data Structures Module
    print("\n📁 Geological Data Structures Module:")
    model = create_geological_model('DEMO-001', 'Demonstration Model')
    print(f"   • Created geological model: {model.model_name}")
    print(f"   • Model ID: {model.model_id}")
    print(f"   • Timestamp: {model.metadata['created_date']}")
    
    # Statistics Module
    print("\n📊 Geological Statistics Module:")
    from geological_statistics import CircularStatistics, HypothesisTesting
    
    # Test circular statistics
    test_angles = np.random.vonmises(np.pi/4, 2, 25)  # Concentrated around 45°
    stats = CircularStatistics.compute_advanced_circular_statistics(test_angles)
    print(f"   • Mean Direction: {stats['mean_direction_deg']:.1f}°")
    print(f"   • Concentration (κ): {stats['von_mises_kappa']:.2f}")
    
    # Test hypothesis testing
    rayleigh_stat, rayleigh_p = HypothesisTesting.rayleigh_test(test_angles)
    print(f"   • Rayleigh Test p-value: {rayleigh_p:.4f}")
    print(f"   • Distribution: {'Non-uniform' if rayleigh_p < 0.05 else 'Possibly uniform'}")
    
    # Constraints Module
    print("\n⚙️ Geological Constraints Module:")
    from geological_constraints import GeologicalConstraints, MLConstraintOptimizer
    
    constraints = GeologicalConstraints()
    # Test constraint bounds
    fault_bounds = constraints.get_constraint_bounds('fault', 'dip')
    print(f"   • Fault dip constraints: {fault_bounds[0]}° - {fault_bounds[1]}°")
    
    # Test ML optimizer (simplified)
    ml_optimizer = MLConstraintOptimizer()
    print(f"   • ML Optimizer initialized")
    print(f"   • Available models: {len(ml_optimizer.models)}")
    
    print("\n✅ All individual modules demonstrated successfully!")

if __name__ == "__main__":
    # Run the comprehensive demonstration
    demo_results = demonstrate_modular_system()
    
    # Run individual module demonstrations
    demonstrate_individual_modules()
    
    print(f"\n🎯 Demonstration completed! Results stored in demo_results variable.")
    print(f"📝 Access components via:")
    print(f"   • demo_results['modeler'] - Advanced geological modeler")
    print(f"   • demo_results['constraint_system'] - Adaptive constraints") 
    print(f"   • demo_results['comprehensive_report'] - Full analysis report")