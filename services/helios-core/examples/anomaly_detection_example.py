"""
Example usage of the comprehensive anomaly detection system for geological phenomena discovery

This example demonstrates how to use the anomaly detection system to identify
new geological phenomena through multi-variate, contextual, and collective analysis.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import pandas as pd
import torch.nn.functional as F

# Import anomaly detection components
from core.anomaly_detection.anomaly_manager import AnomalyDetectionManager
from core.anomaly_detection.collective_detector import GeologicalEventSequence
from core.anomaly_detection.contextual_detector import GeologicalContext
from core.anomaly_detection.continuous_learner import LearningUpdate

class GeologicalDataSimulator:
    """Simulate realistic geological data with embedded anomalies"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.variable_names = [
            "seismic_magnitude",
            "ground_deformation_rate",
            "so2_gas_flux",
            "thermal_anomaly_intensity",
            "magnetic_field_variation",
            "groundwater_level_change",
            "atmospheric_pressure_variation",
        ]

        # Normal parameter ranges and correlations
        self.normal_means = np.array([2.5, 0.01, 100, 0.5, 0.1, 0.0, 0.0])
        self.normal_stds = np.array([0.8, 0.005, 50, 0.2, 0.05, 0.02, 0.01])

        # Correlation matrix for normal conditions
        self.correlation_matrix = np.array(
            [
                [1.0, 0.3, 0.2, 0.1, 0.0, 0.1, 0.0],
                [0.3, 1.0, 0.4, 0.3, 0.1, 0.2, 0.1],
                [0.2, 0.4, 1.0, 0.6, 0.0, 0.1, 0.0],
                [0.1, 0.3, 0.6, 1.0, 0.2, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.2, 1.0, 0.0, 0.1],
                [0.1, 0.2, 0.1, 0.0, 0.0, 1.0, 0.3],
                [0.0, 0.1, 0.0, 0.0, 0.1, 0.3, 1.0],
            ]
        )

        # Covariance matrix
        self.covariance_matrix = (
            np.outer(self.normal_stds, self.normal_stds) * self.correlation_matrix
        )

    def generate_normal_data(self, n_samples: int) -> np.ndarray:
        """Generate normal geological data"""
        return np.random.multivariate_normal(
            self.normal_means, self.covariance_matrix, n_samples
        )

    def generate_volcanic_unrest_anomaly(self, n_samples: int) -> np.ndarray:
        """Generate data representing volcanic unrest"""
        # Elevated gas emissions, thermal anomalies, and ground deformation
        anomaly_means = self.normal_means.copy()
        anomaly_means[1] *= 5  # Increased deformation
        anomaly_means[2] *= 3  # Increased SO2
        anomaly_means[3] *= 4  # Increased thermal activity

        return np.random.multivariate_normal(
            anomaly_means, self.covariance_matrix * 1.5, n_samples
        )

    def generate_earthquake_swarm_anomaly(self, n_samples: int) -> np.ndarray:
        """Generate data representing earthquake swarm"""
        # Increased seismic activity with correlated deformation
        anomaly_means = self.normal_means.copy()
        anomaly_means[0] += 1.5  # Higher magnitude events
        anomaly_means[1] *= 3  # Increased deformation
        anomaly_means[4] *= 2  # Magnetic field changes

        return np.random.multivariate_normal(
            anomaly_means, self.covariance_matrix * 2.0, n_samples
        )

    def generate_novel_phenomenon_anomaly(self, n_samples: int) -> np.ndarray:
        """Generate data representing a novel geological phenomenon"""
        # Unusual combination of parameters not seen in training
        anomaly_data = np.zeros((n_samples, len(self.variable_names)))

        for i in range(n_samples):
            # Create non-linear relationships
            base_intensity = np.random.uniform(0.5, 2.0)

            # Seismic activity inversely correlated with gas emissions (unusual)
            anomaly_data[i, 0] = self.normal_means[0] + base_intensity
            anomaly_data[i, 2] = self.normal_means[2] - base_intensity * 50

            # Thermal anomalies with magnetic field coupling (novel)
            thermal_intensity = np.random.uniform(1.0, 3.0)
            anomaly_data[i, 3] = self.normal_means[3] + thermal_intensity
            anomaly_data[i, 4] = self.normal_means[4] + thermal_intensity * 0.1

            # Other parameters show complex patterns
            anomaly_data[i, 1] = self.normal_means[1] + np.sin(i * 0.5) * 0.02
            anomaly_data[i, 5] = self.normal_means[5] + np.cos(i * 0.3) * 0.05
            anomaly_data[i, 6] = self.normal_means[6] + np.random.normal(0, 0.02)

        return anomaly_data

    def generate_contexts(
        self, n_samples: int, setting: str = "subduction_zone"
    ) -> List[GeologicalContext]:
        """Generate geological contexts for the data"""
        contexts = []

        seasons = ["winter", "spring", "summer", "autumn"]
        times = ["morning", "afternoon", "evening", "night"]

        for i in range(n_samples):
            context = GeologicalContext(
                location=(
                    40.0 + np.random.normal(0, 0.1),
                    -120.0 + np.random.normal(0, 0.1),
                    1000.0 + np.random.normal(0, 100),
                ),
                geological_setting=setting,
                tectonic_environment=(
                    "convergent" if setting == "subduction_zone" else "extensional"
                ),
                volcanic_system_type="stratovolcano",
                season=seasons[i % 4],
                time_of_day=times[i % 4],
                weather_conditions={
                    "temperature": 20.0 + np.random.normal(0, 5),
                    "pressure": 1013.25 + np.random.normal(0, 10),
                    "humidity": 50.0 + np.random.normal(0, 15),
                    "wind_speed": np.random.uniform(0, 20),
                },
                recent_events=[],
                monitoring_network_status={
                    "seismic": "operational",
                    "gps": "operational",
                    "gas": "operational",
                },
            )
            contexts.append(context)

        return contexts

    def generate_event_sequences(
        self, n_sequences: int
    ) -> List[GeologicalEventSequence]:
        """Generate geological event sequences"""
        sequences = []

        for i in range(n_sequences):
            n_events = np.random.randint(3, 15)
            events = []

            start_time = i * 1000

            for j in range(n_events):
                event_time = start_time + j * np.random.uniform(10, 100)

                event = {
                    "timestamp": event_time,
                    "type": np.random.choice(
                        ["earthquake", "gas_emission", "deformation"]
                    ),
                    "magnitude": np.random.uniform(1.0, 5.0),
                    "location": (
                        40.0 + np.random.normal(0, 0.01),
                        -120.0 + np.random.normal(0, 0.01),
                        np.random.uniform(1, 20),
                    ),
                    "intensity": np.random.uniform(0.1, 2.0),
                }
                events.append(event)

            sequence = GeologicalEventSequence(
                events=events,
                start_time=start_time,
                end_time=start_time + (n_events - 1) * 100,
                location=(40.0, -120.0, 10.0),
                sequence_type=np.random.choice(["swarm", "precursory", "aftershock"]),
                metadata={"region": "test_region"},
            )
            sequences.append(sequence)

        return sequences

async def demonstrate_anomaly_detection():
    """Demonstrate comprehensive anomaly detection capabilities"""

    logger.info("Starting geological anomaly detection demonstration")

    # Initialize data simulator
    simulator = GeologicalDataSimulator()

    # Configuration for anomaly detection system
    config = {
        "multivariate": {
            "contamination": 0.1,
            "hidden_dims": [64, 32, 16],
            "dropout_rate": 0.1,
            "anomaly_threshold": 0.7,
            "ensemble_weights": {
                "isolation_forest": 0.25,
                "elliptic_envelope": 0.25,
                "autoencoder": 0.25,
                "vae": 0.25,
            },
        },
        "contextual": {
            "embedding_dim": 32,
            "similar_context_count": 15,
            "contextual_threshold": 0.6,
            "lower_percentile": 10,
            "upper_percentile": 90,
        },
        "collective": {"hidden_dim": 64, "num_layers": 2, "collective_threshold": 0.7},
        "causal_reinforcement": {
            "state_dim": 128,
            "action_dim": 32,
            "hidden_dims": [256, 128, 64],
            "learning_rate": 0.001,
            "buffer_size": 5000,
            "batch_size": 64,
        },
        "continuous_learning": {
            "input_dim": 7,
            "hidden_dims": [128, 64, 32],
            "output_dim": 1,
            "learning_rate": 0.001,
            "memory_size": 2000,
            "embedding_dim": 32,
        },
        "explanation": {
            "confidence_thresholds": {"high": 0.8, "medium": 0.6, "low": 0.4}
        },
        "significance": {
            "statistical_thresholds": {
                "p_value_critical": 0.001,
                "p_value_high": 0.01,
                "p_value_moderate": 0.05,
                "p_value_low": 0.1,
            }
        },
        "ensemble_weights": {
            "multivariate": 0.3,
            "contextual": 0.3,
            "collective": 0.2,
            "causal": 0.2,
        },
        "parallel_processing": True,
        "max_workers": 4,
    }

    # Initialize anomaly detection manager
    manager = AnomalyDetectionManager(config)

    # Generate training data (normal conditions)
    logger.info("Generating training data...")
    n_training = 500
    training_data_normal = simulator.generate_normal_data(n_training)
    training_contexts = simulator.generate_contexts(n_training)
    training_sequences = simulator.generate_event_sequences(20)

    training_data = {
        "data": training_data_normal,
        "variable_names": simulator.variable_names,
        "contexts": training_contexts,
        "sequences": training_sequences,
        "labels": np.zeros(n_training),  # All normal for training
    }

    # Fit all detectors
    logger.info("Fitting anomaly detection models...")
    await manager.fit_all_detectors(training_data)
    logger.info("Model fitting completed")

    # Generate test data with various types of anomalies
    logger.info("Generating test data with embedded anomalies...")

    # Normal data
    normal_test = simulator.generate_normal_data(50)

    # Known anomaly types
    volcanic_unrest = simulator.generate_volcanic_unrest_anomaly(10)
    earthquake_swarm = simulator.generate_earthquake_swarm_anomaly(8)

    # Novel phenomenon (unknown to training data)
    novel_phenomenon = simulator.generate_novel_phenomenon_anomaly(5)

    # Combine all test data
    test_data = np.vstack(
        [normal_test, volcanic_unrest, earthquake_swarm, novel_phenomenon]
    )

    # Generate contexts and timestamps
    n_test = len(test_data)
    test_contexts = simulator.generate_contexts(n_test)
    test_timestamps = np.arange(n_test, dtype=float)
    test_sequences = simulator.generate_event_sequences(5)

    # Create detection data
    detection_data = {
        "data": test_data,
        "timestamps": test_timestamps,
        "contexts": test_contexts,
        "sequences": test_sequences,
        "data_sources": ["simulated_monitoring_network"],
        "context": {
            "geological_setting": "subduction_zone",
            "monitoring_campaign": "volcanic_unrest_study",
        },
    }

    # Perform comprehensive anomaly detection
    logger.info("Performing comprehensive anomaly detection...")
    results = await manager.detect_anomalies(detection_data)

    # Analyze results
    logger.info("Analyzing detection results...")

    # Separate results by data type
    normal_results = results[:50]
    volcanic_results = results[50:60]
    earthquake_results = results[60:68]
    novel_results = results[68:73]

    # Print summary statistics
    def print_detection_summary(results: List, label: str):
        anomaly_count = sum(r.is_anomaly for r in results)
        avg_confidence = np.mean([r.confidence for r in results])
        max_confidence = max([r.confidence for r in results])

        print(f"\n{label}:")
        print(f"  Samples: {len(results)}")
        print(f"  Anomalies detected: {anomaly_count}")
        print(f"  Detection rate: {anomaly_count/len(results)*100:.1f}%")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Maximum confidence: {max_confidence:.3f}")

        # Show anomaly types
        anomaly_types = [r.anomaly_type for r in results if r.is_anomaly]
        if anomaly_types:
            unique_types = list(set(anomaly_types))
            print(f"  Anomaly types: {unique_types}")

    print_detection_summary(normal_results, "Normal Data")
    print_detection_summary(volcanic_results, "Volcanic Unrest")
    print_detection_summary(earthquake_results, "Earthquake Swarm")
    print_detection_summary(novel_results, "Novel Phenomenon")

    # Detailed analysis of novel phenomena
    logger.info("\nDetailed analysis of novel phenomena:")
    for i, result in enumerate(novel_results):
        if result.is_anomaly:
            print(f"\nNovel Anomaly {i+1}:")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Severity: {result.severity_level}")
            print(f"  Urgency: {result.urgency_level}")

            if result.explanation:
                print(f"  Explanation: {result.explanation.explanation_text}")
                print(f"  Primary cause: {result.explanation.primary_cause}")
                print(
                    f"  Contributing factors: {result.explanation.contributing_factors}"
                )

            if result.significance_assessment:
                print(
                    f"  Significance: {result.significance_assessment.overall_significance.value}"
                )
                print(f"  Risk level: {result.significance_assessment.risk_level}")
                print(
                    f"  Recommended response: {result.significance_assessment.recommended_response_level}"
                )

            print(f"  Recommended actions: {result.recommended_actions}")

    # Demonstrate continuous learning
    logger.info("\nDemonstrating continuous learning...")

    # Simulate new data arriving
    new_data = simulator.generate_volcanic_unrest_anomaly(20)
    new_labels = np.ones(20)  # All anomalies

    update = LearningUpdate(
        new_data=new_data,
        new_labels=new_labels,
        data_source="real_time_monitoring",
        timestamp=datetime.now().timestamp(),
        confidence=0.85,
        update_type="incremental",
    )

    # Update continuous learner
    update_result = manager.continuous_learner.continuous_update(update)

    print(
        f"Continuous learning update successful: {update_result['update_successful']}"
    )
    print(f"New patterns discovered: {update_result.get('new_patterns_found', 0)}")
    print(
        f"Task boundary detected: {update_result.get('task_boundary_detected', False)}"
    )

    # Get system statistics
    stats = manager.get_processing_statistics()
    print("\nProcessing Statistics:")
    print(f"  Total samples processed: {stats['total_processed']}")
    print(f"  Anomalies detected: {stats['anomalies_detected']}")
    print(f"  Anomaly rate: {stats.get('anomaly_rate', 0)*100:.1f}%")
    print(f"  Average processing time: {stats.get('average_processing_time', 0):.3f}s")

    # Demonstrate causal learning insights
    logger.info("\nCausal learning insights:")
    causal_policies = manager.causal_learner.get_learned_policies()

    print(
        f"Discovered causal relationships: {len(causal_policies['causal_relationships'])}"
    )
    for rel in causal_policies["causal_relationships"][:5]:  # Show top 5
        print(f"  {rel['cause']} → {rel['effect']} (strength: {rel['strength']:.3f})")

    print(f"Action preferences: {causal_policies['action_preferences']}")
    print(f"Learning effectiveness: {causal_policies['effectiveness_metrics']}")

    # Export results for further analysis
    logger.info("Exporting results...")
    manager.export_results(results, "anomaly_detection_results.json")

    # Visualization (if matplotlib is available)
    try:
        visualize_results(test_data, results, simulator.variable_names)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    logger.info("Anomaly detection demonstration completed successfully!")

def visualize_results(data: np.ndarray, results: List, variable_names: List[str]):
    """Visualize anomaly detection results"""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Geological Anomaly Detection Results", fontsize=16)

    # Extract anomaly information
    anomaly_scores = [r.confidence for r in results]
    is_anomaly = [r.is_anomaly for r in results]
    timestamps = range(len(results))

    # Plot 1: Anomaly scores over time
    ax1 = axes[0, 0]
    colors = ["red" if a else "blue" for a in is_anomaly]
    ax1.scatter(timestamps, anomaly_scores, c=colors, alpha=0.6)
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Anomaly Score")
    ax1.set_title("Anomaly Scores Over Time")
    ax1.legend(["Normal", "Anomaly"])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter correlation for anomalies
    ax2 = axes[0, 1]
    anomaly_indices = [i for i, a in enumerate(is_anomaly) if a]
    if anomaly_indices:
        anomaly_data = data[anomaly_indices]
        correlation_matrix = np.corrcoef(anomaly_data.T)
        im = ax2.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        ax2.set_xticks(range(len(variable_names)))
        ax2.set_yticks(range(len(variable_names)))
        ax2.set_xticklabels([name[:10] for name in variable_names], rotation=45)
        ax2.set_yticklabels([name[:10] for name in variable_names])
        ax2.set_title("Parameter Correlations in Anomalies")
        plt.colorbar(im, ax=ax2)

    # Plot 3: Anomaly types distribution
    ax3 = axes[1, 0]
    anomaly_types = [r.anomaly_type for r in results if r.is_anomaly]
    if anomaly_types:
        unique_types, counts = np.unique(anomaly_types, return_counts=True)
        ax3.bar(range(len(unique_types)), counts)
        ax3.set_xticks(range(len(unique_types)))
        ax3.set_xticklabels(unique_types, rotation=45)
        ax3.set_ylabel("Count")
        ax3.set_title("Distribution of Anomaly Types")

    # Plot 4: Severity vs Confidence
    ax4 = axes[1, 1]
    severity_levels = [r.severity_level for r in results if r.is_anomaly]
    anomaly_confidences = [r.confidence for r in results if r.is_anomaly]

    if severity_levels and anomaly_confidences:
        severity_mapping = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
        severity_numeric = [severity_mapping.get(s, 0) for s in severity_levels]

        ax4.scatter(anomaly_confidences, severity_numeric, alpha=0.7)
        ax4.set_xlabel("Confidence")
        ax4.set_ylabel("Severity Level")
        ax4.set_yticks([1, 2, 3, 4])
        ax4.set_yticklabels(["Low", "Moderate", "High", "Critical"])
        ax4.set_title("Anomaly Severity vs Confidence")
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("anomaly_detection_visualization.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Visualization saved as 'anomaly_detection_visualization.png'")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_anomaly_detection())
