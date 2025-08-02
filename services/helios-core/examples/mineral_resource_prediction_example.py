"""
Example script demonstrating mineral resource prediction functionality.

This script shows how to use the mineral resource prediction models for:
    1. Training models with known deposit data
2. Predicting resource potential at new locations
3. Generating resource maps
4. Evaluating model performance

Requirements addressed:
    - 6.1: Identify patterns in known resource locations
- 6.3: Provide confidence intervals and uncertainty estimates
- 2.3.7: Model hydrothermal, magmatic, and sedimentary ore-forming processes
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from core.mineral_resource_prediction import (
    DepositType,
    FormationProcess,
    MineralResourcePredictor,
    ResourceDeposit,
    ResourceType,
)
from models.geological import GeochemicalSample, GeologicalLocation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_training_data():
    """
    Create synthetic training data for demonstration.

    In practice, this would come from geological databases and field surveys.
    """
    logger.info("Creating synthetic training data...")

    deposits = []
    samples = []

    # Create copper porphyry deposits
    for i in range(25):
        location = GeologicalLocation(
            latitude=45.0 + np.random.normal(0, 0.5),
            longitude=-120.0 + np.random.normal(0, 0.5),
            elevation=1000.0 + np.random.normal(0, 200),
            depth=500.0 + np.random.normal(0, 100),
        )

        # Porphyry copper characteristics
        grade = np.random.lognormal(np.log(0.8), 0.3)  # ~0.8% Cu average
        tonnage = np.random.lognormal(np.log(100e6), 0.5)  # ~100 Mt average

        geochemical_signature = {
            "Cu": np.random.lognormal(np.log(8000), 0.3),  # ppm
            "Mo": np.random.lognormal(np.log(100), 0.5),
            "Au": np.random.lognormal(np.log(0.3), 0.4),
            "Ag": np.random.lognormal(np.log(3), 0.3),
            "SiO2": np.random.normal(65, 5),
            "Al2O3": np.random.normal(15, 2),
            "Fe2O3": np.random.normal(5, 1),
            "K2O": np.random.normal(3, 0.5),
        }

        geological_setting = {
            "rock_types": ["igneous"],
            "fault_density": np.random.uniform(3, 8),
            "fracture_intensity": np.random.uniform(2, 6),
            "fold_intensity": np.random.uniform(1, 4),
            "intrusion_distance": np.random.uniform(100, 2000),
            "volcanic_activity": np.random.uniform(0.3, 0.8),
            "hydrothermal_alteration": np.random.uniform(0.5, 0.9),
        }

        deposit = ResourceDeposit(
            location=location,
            resource_type=ResourceType.COPPER,
            deposit_type=DepositType.PORPHYRY,
            formation_process=FormationProcess.HYDROTHERMAL,
            grade=grade,
            tonnage=tonnage,
            geochemical_signature=geochemical_signature,
            geological_setting=geological_setting,
            confidence=np.random.uniform(0.7, 0.95),
            deposit_id=f"CU_PORPHYRY_{i:03d}",
        )
        deposits.append(deposit)

        # Corresponding geochemical sample
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type="rock",
            composition=geochemical_signature,
            temperature=np.random.uniform(250, 400),
            pressure=np.random.uniform(5e6, 2e7),
        )
        samples.append(sample)

    # Create epithermal gold deposits
    for i in range(20):
        location = GeologicalLocation(
            latitude=46.0 + np.random.normal(0, 0.3),
            longitude=-121.0 + np.random.normal(0, 0.3),
            elevation=1500.0 + np.random.normal(0, 300),
            depth=200.0 + np.random.normal(0, 50),
        )

        # Epithermal gold characteristics
        grade = np.random.lognormal(np.log(5), 0.5)  # g/t Au
        tonnage = np.random.lognormal(np.log(10e6), 0.6)  # ~10 Mt average

        geochemical_signature = {
            "Au": np.random.lognormal(np.log(5), 0.4),  # g/t
            "Ag": np.random.lognormal(np.log(50), 0.5),
            "Cu": np.random.lognormal(np.log(200), 0.4),
            "Pb": np.random.lognormal(np.log(500), 0.3),
            "Zn": np.random.lognormal(np.log(300), 0.3),
            "As": np.random.lognormal(np.log(100), 0.4),
            "Sb": np.random.lognormal(np.log(20), 0.5),
            "SiO2": np.random.normal(75, 3),
            "Al2O3": np.random.normal(12, 1),
            "K2O": np.random.normal(4, 0.8),
        }

        geological_setting = {
            "rock_types": ["volcanic"],
            "fault_density": np.random.uniform(5, 12),
            "fracture_intensity": np.random.uniform(4, 8),
            "fold_intensity": np.random.uniform(1, 3),
            "intrusion_distance": np.random.uniform(500, 3000),
            "volcanic_activity": np.random.uniform(0.6, 0.9),
            "hydrothermal_alteration": np.random.uniform(0.4, 0.8),
        }

        deposit = ResourceDeposit(
            location=location,
            resource_type=ResourceType.GOLD,
            deposit_type=DepositType.EPITHERMAL,
            formation_process=FormationProcess.HYDROTHERMAL,
            grade=grade,
            tonnage=tonnage,
            geochemical_signature=geochemical_signature,
            geological_setting=geological_setting,
            confidence=np.random.uniform(0.6, 0.9),
            deposit_id=f"AU_EPITHERMAL_{i:03d}",
        )
        deposits.append(deposit)

        # Corresponding geochemical sample
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type="rock",
            composition=geochemical_signature,
            temperature=np.random.uniform(150, 300),
            pressure=np.random.uniform(1e6, 1e7),
        )
        samples.append(sample)

    # Create VMS zinc deposits
    for i in range(15):
        location = GeologicalLocation(
            latitude=47.0 + np.random.normal(0, 0.2),
            longitude=-122.0 + np.random.normal(0, 0.2),
            elevation=800.0 + np.random.normal(0, 150),
            depth=800.0 + np.random.normal(0, 200),
        )

        # VMS characteristics
        grade = np.random.lognormal(np.log(8), 0.3)  # % Zn
        tonnage = np.random.lognormal(np.log(5e6), 0.4)  # ~5 Mt average

        geochemical_signature = {
            "Zn": np.random.lognormal(np.log(80000), 0.3),  # ppm
            "Pb": np.random.lognormal(np.log(20000), 0.4),
            "Cu": np.random.lognormal(np.log(10000), 0.4),
            "Ag": np.random.lognormal(np.log(100), 0.3),
            "Au": np.random.lognormal(np.log(2), 0.5),
            "SiO2": np.random.normal(50, 5),
            "Al2O3": np.random.normal(14, 2),
            "Fe2O3": np.random.normal(12, 3),
            "MgO": np.random.normal(6, 2),
        }

        geological_setting = {
            "rock_types": ["volcanic", "sedimentary"],
            "fault_density": np.random.uniform(4, 10),
            "fracture_intensity": np.random.uniform(3, 7),
            "fold_intensity": np.random.uniform(2, 5),
            "intrusion_distance": np.random.uniform(1000, 5000),
            "volcanic_activity": np.random.uniform(0.7, 1.0),
            "hydrothermal_alteration": np.random.uniform(0.6, 0.9),
        }

        deposit = ResourceDeposit(
            location=location,
            resource_type=ResourceType.ZINC,
            deposit_type=DepositType.VMS,
            formation_process=FormationProcess.HYDROTHERMAL,
            grade=grade,
            tonnage=tonnage,
            geochemical_signature=geochemical_signature,
            geological_setting=geological_setting,
            confidence=np.random.uniform(0.7, 0.9),
            deposit_id=f"ZN_VMS_{i:03d}",
        )
        deposits.append(deposit)

        # Corresponding geochemical sample
        sample = GeochemicalSample(
            timestamp=datetime.utcnow(),
            location=location,
            sample_type="rock",
            composition=geochemical_signature,
            temperature=np.random.uniform(200, 350),
            pressure=np.random.uniform(2e7, 5e7),
        )
        samples.append(sample)

    logger.info(f"Created {len(deposits)} deposits and {len(samples)} samples")
    return deposits, samples


def demonstrate_model_training(predictor, deposits, samples):
    """Demonstrate model training process"""
    logger.info("Demonstrating model training...")

    # Add training data
    predictor.add_training_data(deposits, samples)

    # Train models
    results = predictor.train_models()

    # Display training results
    print("\n=== MODEL TRAINING RESULTS ===")
    print("Signature Analysis:")
    sig_results = results["signature_analysis"]
    print(f"  - Training Accuracy: {sig_results['train_accuracy']:.3f}")
    print(f"  - Test Accuracy: {sig_results['test_accuracy']:.3f}")
    print(
        f"  - Cross-validation: {sig_results['cv_mean']:.3f} ± {sig_results['cv_std']:.3f}"
    )
    print(f"  - Number of samples: {sig_results['n_samples']}")

    print("\nPattern Recognition:")
    pat_results = results["pattern_recognition"]
    print(
        f"  - Resource Type Classifier: {pat_results['resource_type']['test_accuracy']:.3f}"
    )
    print(
        f"  - Deposit Type Classifier: {pat_results['deposit_type']['test_accuracy']:.3f}"
    )
    print(f"  - Grade Predictor R²: {pat_results['grade']['test_r2']:.3f}")
    print(f"  - Tonnage Predictor R²: {pat_results['tonnage']['test_r2']:.3f}")

    # Display feature importance
    print("\nTop 5 Most Important Geochemical Features:")
    importance = sig_results["feature_importance"]
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance_val in sorted_features[:5]:
        print(f"  - {feature}: {importance_val:.3f}")


def demonstrate_resource_prediction(predictor):
    """Demonstrate resource prediction at specific locations"""
    logger.info("Demonstrating resource prediction...")

    # Define test locations with different characteristics
    test_cases = [
        {
            "name": "Porphyry-favorable location",
            "location": GeologicalLocation(45.2, -119.8, 1200.0, 600.0),
            "geological_setting": {
                "rock_types": ["igneous"],
                "fault_density": 6.0,
                "fracture_intensity": 4.0,
                "fold_intensity": 2.0,
                "intrusion_distance": 500.0,
                "volcanic_activity": 0.7,
                "hydrothermal_alteration": 0.8,
            },
            "geochemical_data": {
                "Cu": 5000,
                "Mo": 150,
                "Au": 0.8,
                "SiO2": 63.0,
                "K2O": 3.5,
            },
        },
        {
            "name": "Epithermal-favorable location",
            "location": GeologicalLocation(46.1, -120.9, 1600.0, 300.0),
            "geological_setting": {
                "rock_types": ["volcanic"],
                "fault_density": 8.0,
                "fracture_intensity": 6.0,
                "fold_intensity": 1.5,
                "intrusion_distance": 1500.0,
                "volcanic_activity": 0.8,
                "hydrothermal_alteration": 0.6,
            },
            "geochemical_data": {
                "Au": 8.0,
                "Ag": 80,
                "As": 200,
                "Sb": 50,
                "SiO2": 76.0,
            },
        },
        {
            "name": "Low-potential location",
            "location": GeologicalLocation(44.5, -118.5, 800.0, 100.0),
            "geological_setting": {
                "rock_types": ["sedimentary"],
                "fault_density": 2.0,
                "fracture_intensity": 1.0,
                "fold_intensity": 3.0,
                "intrusion_distance": 5000.0,
                "volcanic_activity": 0.1,
                "hydrothermal_alteration": 0.2,
            },
            "geochemical_data": {"Cu": 50, "Au": 0.01, "SiO2": 70.0, "Al2O3": 15.0},
        },
    ]

    print("\n=== RESOURCE PREDICTIONS ===")

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(
            f"Location: {test_case['location'].latitude:.2f}°N, {test_case['location'].longitude:.2f}°W"
        )

        prediction = predictor.predict_resource_potential(
            location=test_case["location"],
            geological_setting=test_case["geological_setting"],
            geochemical_data=test_case["geochemical_data"],
        )

        print(f"  - Resource Type: {prediction.resource_type.value}")
        print(f"  - Probability: {prediction.probability:.3f}")
        print(f"  - Predicted Grade: {prediction.predicted_grade:.2f}")
        print(f"  - Predicted Tonnage: {prediction.predicted_tonnage:.0f} tonnes")
        print(f"  - Geological Favorability: {prediction.geological_favorability:.3f}")
        print(f"  - Model Confidence: {prediction.model_confidence:.3f}")

        # Display uncertainty bounds
        print(
            f"  - Grade Uncertainty: {prediction.uncertainty_bounds['grade'][0]:.2f} - {prediction.uncertainty_bounds['grade'][1]:.2f}"
        )
        print(
            f"  - Tonnage Uncertainty: {prediction.uncertainty_bounds['tonnage'][0]:.0f} - {prediction.uncertainty_bounds['tonnage'][1]:.0f}"
        )


def demonstrate_resource_mapping(predictor):
    """Demonstrate resource map generation"""
    logger.info("Demonstrating resource map generation...")

    # Create a grid of locations
    lat_range = np.linspace(45.0, 47.0, 10)
    lon_range = np.linspace(-122.0, -119.0, 10)

    locations = []
    geological_settings = []

    for lat in lat_range:
        for lon in lon_range:
            location = GeologicalLocation(
                latitude=lat,
                longitude=lon,
                elevation=1000.0 + np.random.normal(0, 200),
                depth=500.0 + np.random.normal(0, 100),
            )
            locations.append(location)

            # Vary geological settings based on location
            setting = {
                "rock_types": ["igneous"] if lat < 46.0 else ["volcanic"],
                "fault_density": 3.0 + (lat - 45.0) * 2.0,
                "fracture_intensity": 2.0 + abs(lon + 120.5) * 2.0,
                "fold_intensity": 2.0,
                "intrusion_distance": 1000.0 + abs(lon + 120.5) * 500.0,
                "volcanic_activity": 0.3 + (lat - 45.0) * 0.2,
                "hydrothermal_alteration": 0.4 + abs(lon + 120.5) * 0.2,
            }
            geological_settings.append(setting)

    # Generate resource map
    map_data = predictor.generate_resource_map(
        locations=locations,
        geological_settings=geological_settings,
        resource_type=ResourceType.COPPER,
    )

    print("\n=== RESOURCE MAP RESULTS ===")
    print(f"Total locations evaluated: {map_data['metadata']['total_locations']}")
    print(f"Successful predictions: {map_data['metadata']['successful_predictions']}")
    print(f"Resource type filter: {map_data['metadata']['resource_type_filter']}")

    # Display statistics
    probabilities = np.array(map_data["probabilities"])
    grades = np.array(map_data["predicted_grades"])

    print("\nProbability Statistics:")
    print(f"  - Mean: {probabilities.mean():.3f}")
    print(f"  - Std: {probabilities.std():.3f}")
    print(f"  - Max: {probabilities.max():.3f}")
    print(f"  - High potential locations (>0.7): {(probabilities > 0.7).sum()}")

    print("\nGrade Statistics:")
    print(f"  - Mean: {grades.mean():.2f}")
    print(f"  - Std: {grades.std():.2f}")
    print(f"  - Max: {grades.max():.2f}")

    return map_data


def demonstrate_process_modeling(predictor):
    """Demonstrate geological process modeling"""
    logger.info("Demonstrating geological process modeling...")

    print("\n=== GEOLOGICAL PROCESS MODELING ===")

    # Test hydrothermal process modeling
    location = GeologicalLocation(45.5, -120.0, 1300.0, 700.0)
    geological_setting = {
        "rock_types": ["igneous"],
        "fault_density": 7.0,
        "fracture_intensity": 5.0,
        "intrusion_distance": 300.0,
        "volcanic_activity": 0.8,
        "hydrothermal_alteration": 0.9,
    }

    process_conditions = {
        "hydrothermal": {
            "temperature": 350.0,
            "pressure": 1.5e7,
            "fluid_composition": {
                "NaCl": 20.0,
                "pH": 5.5,
                "S": 0.15,
                "Cu": 200,
                "Au": 1.0,
                "Mo": 50,
            },
            "host_rock": {"reactivity": 0.8, "permeability": 1e-13},
        }
    }

    prediction = predictor.predict_resource_potential(
        location=location,
        geological_setting=geological_setting,
        process_conditions=process_conditions,
    )

    print("Hydrothermal Process Modeling Results:")
    print(f"  - Resource Type: {prediction.resource_type.value}")
    print(f"  - Probability: {prediction.probability:.3f}")
    print(f"  - Geological Favorability: {prediction.geological_favorability:.3f}")
    print("  - Formation Process Likelihood:")
    for process, likelihood in prediction.formation_likelihood.items():
        print(f"    - {process.value}: {likelihood:.3f}")


def demonstrate_model_evaluation(predictor, deposits):
    """Demonstrate model performance evaluation"""
    logger.info("Demonstrating model evaluation...")

    # Use a subset of deposits for evaluation
    test_deposits = deposits[-10:]  # Last 10 deposits

    metrics = predictor.evaluate_model_performance(test_deposits)

    print("\n=== MODEL PERFORMANCE EVALUATION ===")
    print(f"Test dataset size: {metrics['n_predictions']}")
    print(f"Resource type accuracy: {metrics['resource_type_accuracy']:.3f}")
    print(f"Mean grade error: {metrics['mean_grade_error']:.3f}")
    print(f"Mean tonnage error: {metrics['mean_tonnage_error']:.3f}")
    print(f"High probability precision: {metrics['probability_precision']:.3f}")


def visualize_results(map_data):
    """Create visualizations of the results"""
    logger.info("Creating visualizations...")

    try:
        # Extract data for plotting
        locations = np.array(map_data["locations"])
        probabilities = np.array(map_data["probabilities"])
        grades = np.array(map_data["predicted_grades"])

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Resource probability map
        scatter1 = ax1.scatter(
            locations[:, 1],
            locations[:, 0],
            c=probabilities,
            cmap="viridis",
            s=50,
            alpha=0.7,
        )
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title("Resource Probability Map")
        plt.colorbar(scatter1, ax=ax1, label="Probability")

        # Plot 2: Predicted grade map
        scatter2 = ax2.scatter(
            locations[:, 1], locations[:, 0], c=grades, cmap="plasma", s=50, alpha=0.7
        )
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")
        ax2.set_title("Predicted Grade Map")
        plt.colorbar(scatter2, ax=ax2, label="Grade")

        # Plot 3: Probability histogram
        ax3.hist(probabilities, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
        ax3.set_xlabel("Probability")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Probability Distribution")
        ax3.axvline(
            probabilities.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {probabilities.mean():.3f}",
        )
        ax3.legend()

        # Plot 4: Grade histogram
        ax4.hist(grades, bins=20, alpha=0.7, color="lightcoral", edgecolor="black")
        ax4.set_xlabel("Predicted Grade")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Grade Distribution")
        ax4.axvline(
            grades.mean(),
            color="blue",
            linestyle="--",
            label=f"Mean: {grades.mean():.2f}",
        )
        ax4.legend()

        plt.tight_layout()

        # Save the plot
        output_path = Path("mineral_resource_prediction_results.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Visualization saved to {output_path}")

        plt.show()

    except ImportError:
        logger.warning("Matplotlib not available, skipping visualization")
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")


def main():
    """Main demonstration function"""
    print("=== MINERAL RESOURCE PREDICTION DEMONSTRATION ===")
    print(
        "This example demonstrates the complete mineral resource prediction workflow."
    )
    print("Requirements addressed: 6.1, 6.3, 2.3.7")

    # Initialize predictor
    predictor = MineralResourcePredictor(model_storage_path="example_models")

    # Create synthetic training data
    deposits, samples = create_synthetic_training_data()

    # Demonstrate model training
    demonstrate_model_training(predictor, deposits, samples)

    # Demonstrate resource prediction
    demonstrate_resource_prediction(predictor)

    # Demonstrate resource mapping
    map_data = demonstrate_resource_mapping(predictor)

    # Demonstrate process modeling
    demonstrate_process_modeling(predictor)

    # Demonstrate model evaluation
    demonstrate_model_evaluation(predictor, deposits)

    # Create visualizations
    visualize_results(map_data)

    print("\n=== DEMONSTRATION COMPLETE ===")
    print("The mineral resource prediction system has been successfully demonstrated.")
    print("Key capabilities shown:")
    print("- Machine learning models for ore deposit pattern recognition")
    print("- Geochemical signature analysis for resource identification")
    print("- Geological process modeling for resource formation")
    print("- Uncertainty quantification for resource predictions")
    print("- Resource potential mapping")
    print("- Model performance evaluation")


if __name__ == "__main__":
    main()
