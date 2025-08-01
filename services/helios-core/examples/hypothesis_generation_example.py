"""
Example usage of the Hypothesis Generation and Validation System

This example demonstrates how to use the automated hypothesis generation system
to create testable scientific hypotheses from discovered causal relationships,
formulate mathematical laws, assess confidence, and design validation experiments.

Requirements implemented: 15.3, 15.7, 15.8
"""

import json
import os
import sys
from datetime import datetime

import numpy as np

# Add the parent directory to the path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hypothesis_generation_engine import (
    CausalRelationship,
    HypothesisValidationSystem,
)

def create_sample_causal_relationships():
    """Create sample causal relationships for demonstration"""

    relationships = [
        CausalRelationship(
            cause="magma_temperature",
            effect="gas_pressure",
            strength=0.85,
            confidence=0.92,
            mechanism="thermal_expansion_of_volatiles",
            evidence=[
                "laboratory_experiments",
                "thermodynamic_modeling",
                "field_observations",
            ],
        ),
        CausalRelationship(
            cause="gas_pressure",
            effect="seismic_activity",
            strength=0.73,
            confidence=0.88,
            mechanism="pressure_induced_fracturing",
            evidence=["seismic_monitoring", "pressure_measurements"],
        ),
        CausalRelationship(
            cause="seismic_activity",
            effect="ground_deformation",
            strength=0.67,
            confidence=0.85,
            mechanism="elastic_stress_release",
            evidence=["GPS_measurements", "InSAR_data", "tiltmeter_data"],
        ),
        CausalRelationship(
            cause="magma_temperature",
            effect="thermal_emission",
            strength=0.91,
            confidence=0.95,
            mechanism="blackbody_radiation",
            evidence=["thermal_infrared_imagery", "temperature_measurements"],
        ),
        CausalRelationship(
            cause="ground_deformation",
            effect="stress_field_changes",
            strength=0.78,
            confidence=0.83,
            mechanism="elastic_deformation_coupling",
            evidence=["stress_modeling", "structural_analysis"],
        ),
    ]

    return relationships

def create_sample_geological_data():
    """Create sample geological data patterns"""

    # Simulate temperature-pressure relationship data
    temperatures = np.linspace(800, 1200, 50)  # Celsius
    pressures = 2.5 * temperatures + np.random.normal(0, 50, 50)  # MPa
    temp_pressure_data = np.column_stack([temperatures, pressures])

    # Simulate pressure-seismicity relationship
    pressures_seis = np.linspace(2000, 3500, 40)  # MPa
    seismic_events = 0.01 * pressures_seis**1.2 + np.random.normal(0, 5, 40)
    pressure_seismic_data = np.column_stack([pressures_seis, seismic_events])

    # Simulate deformation-stress relationship
    deformation = np.linspace(0, 0.05, 30)  # strain
    stress = 30e9 * deformation + np.random.normal(0, 1e6, 30)  # Pa
    deformation_stress_data = np.column_stack([deformation, stress])

    return {
        "temperature_pressure": temp_pressure_data,
        "pressure_seismicity": pressure_seismic_data,
        "deformation_stress": deformation_stress_data,
        "p_values": [0.001, 0.01, 0.005, 0.002, 0.008],
        "correlation_coefficients": [0.85, 0.73, 0.67, 0.91, 0.78],
        "sample_size": 150,
        "completeness": 0.92,
        "accuracy": 0.88,
        "temporal_coverage": 0.85,
    }

def create_geological_context():
    """Create geological context for the analysis"""

    return {
        "geological_setting": "active_stratovolcano",
        "location": "Mount_Prometheus_Observatory",
        "temporal_scale": "real_time_monitoring",
        "spatial_scale": "local_volcanic_system",
        "volcano_type": "andesitic_stratovolcano",
        "tectonic_setting": "subduction_zone",
        "monitoring_network": "comprehensive",
        "historical_activity": "frequent_unrest_episodes",
    }

def create_available_resources():
    """Define available resources for validation experiments"""

    return {
        "budget": "medium_to_high",
        "time": "months_to_years",
        "equipment": [
            "seismic_network",
            "GPS_stations",
            "thermal_cameras",
            "gas_sensors",
            "tiltmeters",
            "laboratory_facilities",
            "high_performance_computing",
            "field_vehicles",
        ],
        "personnel": 8,
        "institutional_support": "university_volcano_observatory",
        "international_collaboration": True,
    }

def demonstrate_hypothesis_generation():
    """Demonstrate the complete hypothesis generation workflow"""

    print("=" * 80)
    print("PROMETHEUS AI HYPOTHESIS GENERATION AND VALIDATION SYSTEM")
    print("=" * 80)
    print()

    # Initialize the system
    print("1. Initializing Hypothesis Validation System...")
    hypothesis_system = HypothesisValidationSystem()
    print("   ✓ System initialized successfully")
    print()

    # Prepare input data
    print("2. Preparing input data...")
    causal_relationships = create_sample_causal_relationships()
    geological_data = create_sample_geological_data()
    geological_context = create_geological_context()
    available_resources = create_available_resources()

    print(f"   ✓ {len(causal_relationships)} causal relationships prepared")
    print(f"   ✓ {len(geological_data)} data patterns prepared")
    print(f"   ✓ Geological context: {geological_context['geological_setting']}")
    print()

    # Display input causal relationships
    print("3. Input Causal Relationships:")
    for i, rel in enumerate(causal_relationships, 1):
        print(f"   {i}. {rel.cause} → {rel.effect}")
        print(f"      Strength: {rel.strength:.3f}, Confidence: {rel.confidence:.3f}")
        print(f"      Mechanism: {rel.mechanism}")
        print()

    # Generate hypotheses and laws
    print("4. Generating hypotheses and formulating laws...")
    start_time = datetime.now()

    results = hypothesis_system.generate_and_validate_hypotheses(
        causal_relationships=causal_relationships,
        geological_data=geological_data,
        geological_context=geological_context,
        available_resources=available_resources,
    )

    processing_time = (datetime.now() - start_time).total_seconds()
    print(f"   ✓ Processing completed in {processing_time:.2f} seconds")
    print()

    # Display results summary
    print("5. Results Summary:")
    print(f"   • Total hypotheses generated: {results['total_hypotheses']}")
    print(f"   • High-confidence hypotheses: {results['high_confidence_hypotheses']}")
    print(f"   • Testable hypotheses: {results['testable_hypotheses']}")
    print(f"   • Formulated laws: {len(results['formulated_laws'])}")
    print()

    # Display top hypotheses
    print("6. Top Generated Hypotheses:")
    top_hypotheses = sorted(
        results["generated_hypotheses"],
        key=lambda h: h.confidence_score * h.novelty_score,
        reverse=True,
    )[:5]

    for i, hypothesis in enumerate(top_hypotheses, 1):
        print(f"\n   Hypothesis {i}:")
        print(f"   ID: {hypothesis.id}")
        print(f"   Statement: {hypothesis.statement}")
        print(f"   Confidence: {hypothesis.confidence_score:.3f}")
        print(f"   Novelty: {hypothesis.novelty_score:.3f}")
        print(f"   Testability: {hypothesis.testability_score:.3f}")

        if hypothesis.predicted_outcomes:
            print("   Predicted outcomes:")
            for outcome in hypothesis.predicted_outcomes[:3]:
                print(f"     • {outcome}")

        if hypothesis.experimental_design:
            print(
                f"   Validation experiments: {len(hypothesis.experimental_design)} designed"
            )
            for exp in hypothesis.experimental_design[:2]:
                print(
                    f"     • {exp.get('experiment_type', 'Unknown')}: {exp.get('title', 'No title')[:60]}..."
                )

    print()

    # Display formulated laws
    print("7. Formulated Geological Laws:")
    for i, law in enumerate(results["formulated_laws"], 1):
        print(f"\n   Law {i}: {law.name}")
        print(f"   Mathematical expression: {law.mathematical_expression}")
        print(f"   Variables: {', '.join(law.variables)}")
        print(f"   Confidence: {law.confidence:.3f}")
        print(f"   Domain: {', '.join(law.domain_applicability)}")
        print(f"   Validation experiments: {len(law.validation_experiments)} designed")

    print()

    # Display summary insights
    print("8. Summary Insights:")
    summary = results["summary"]
    print(f"   • Average confidence: {summary['average_confidence']:.3f}")
    print(f"   • Average novelty: {summary['average_novelty']:.3f}")
    print(f"   • Average testability: {summary['average_testability']:.3f}")

    if "top_hypothesis" in summary:
        top_hyp = summary["top_hypothesis"]
        print(f"   • Top hypothesis: {top_hyp['statement'][:80]}...")
        print(
            f"     Confidence: {top_hyp['confidence']:.3f}, Novelty: {top_hyp['novelty']:.3f}"
        )

    print()

    # Display recommendations
    print("9. Research Recommendations:")
    if "recommendations" in summary:
        for i, rec in enumerate(summary["recommendations"], 1):
            print(f"   {i}. {rec}")

    print()

    # Save results to file
    print("10. Saving results...")
    output_file = (
        f"hypothesis_generation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    # Convert results to JSON-serializable format
    json_results = {
        "metadata": {
            "generation_timestamp": results["generation_timestamp"].isoformat(),
            "processing_time_seconds": processing_time,
            "input_relationships": len(causal_relationships),
            "geological_context": geological_context,
        },
        "hypotheses": [
            {
                "id": h.id,
                "statement": h.statement,
                "confidence_score": h.confidence_score,
                "novelty_score": h.novelty_score,
                "testability_score": h.testability_score,
                "predicted_outcomes": h.predicted_outcomes,
                "has_experimental_design": h.experimental_design is not None,
                "experiment_count": (
                    len(h.experimental_design) if h.experimental_design else 0
                ),
            }
            for h in results["generated_hypotheses"]
        ],
        "laws": [
            {
                "name": law.name,
                "mathematical_expression": law.mathematical_expression,
                "variables": law.variables,
                "confidence": law.confidence,
                "domain_applicability": law.domain_applicability,
            }
            for law in results["formulated_laws"]
        ],
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"   ✓ Results saved to {output_file}")
    print()

    print("=" * 80)
    print("HYPOTHESIS GENERATION DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return results

def demonstrate_individual_components():
    """Demonstrate individual system components"""

    print("\n" + "=" * 60)
    print("INDIVIDUAL COMPONENT DEMONSTRATIONS")
    print("=" * 60)

    # Test hypothesis generator
    print("\nA. Hypothesis Generator:")
    from core.hypothesis_generation_engine import HypothesisGenerator

    generator = HypothesisGenerator()
    sample_rels = create_sample_causal_relationships()[:2]
    context = create_geological_context()

    hypotheses = generator.generate_hypotheses_from_causal_relationships(
        sample_rels, context
    )

    print(
        f"   Generated {len(hypotheses)} hypotheses from {len(sample_rels)} relationships"
    )
    if hypotheses:
        print(f"   Example: {hypotheses[0].statement[:100]}...")

    # Test mathematical formulator
    print("\nB. Mathematical Formulation Engine:")
    from core.hypothesis_generation_engine import MathematicalFormulationEngine

    formulator = MathematicalFormulationEngine()
    data_patterns = create_sample_geological_data()

    law = formulator.formulate_geological_law(sample_rels, data_patterns, context)

    print(f"   Formulated law: {law.name}")
    print(f"   Expression: {law.mathematical_expression}")
    print(f"   Confidence: {law.confidence:.3f}")

    # Test confidence assessor
    print("\nC. Confidence Assessment Engine:")
    from core.hypothesis_generation_engine import ConfidenceAssessmentEngine

    assessor = ConfidenceAssessmentEngine()
    if hypotheses:
        confidence = assessor.assess_hypothesis_confidence(hypotheses[0], data_patterns)
        print(f"   Assessed confidence: {confidence:.3f}")

    # Test experiment designer
    print("\nD. Experimental Design Engine:")
    from core.hypothesis_generation_engine import ExperimentalDesignEngine

    designer = ExperimentalDesignEngine()
    resources = create_available_resources()

    if hypotheses:
        experiments = designer.design_validation_experiments(
            hypotheses[0], resources, "volcanic_systems"
        )
        print(f"   Designed {len(experiments)} validation experiments")
        if experiments:
            print(f"   Example: {experiments[0].get('title', 'No title')[:80]}...")

if __name__ == "__main__":
    try:
        # Run the main demonstration
        results = demonstrate_hypothesis_generation()

        # Run individual component demonstrations
        demonstrate_individual_components()

        print("\n✓ All demonstrations completed successfully!")
        print(
            f"Generated {results['total_hypotheses']} hypotheses and {len(results['formulated_laws'])} laws"
        )

    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        import traceback

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

        traceback.print_exc()
        sys.exit(1)
