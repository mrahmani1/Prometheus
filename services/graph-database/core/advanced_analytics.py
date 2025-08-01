"""
Advanced analytics for geological graph networks
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .database import GraphDatabaseManager

logger = logging.getLogger(__name__)

class AdvancedGraphAnalytics:
    """Advanced analytics for geological process networks"""

    def __init__(self, db_manager: GraphDatabaseManager):
        self.db_manager = db_manager

    async def analyze_temporal_evolution(
        self, process_ids: List[str], time_window_days: int = 365
    ) -> Dict[str, Any]:
        """Analyze how the network structure evolves over time"""

        # Get temporal data for processes
        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        AND r.created_at >= datetime() - duration({days: $time_window_days})
        RETURN p.process_id as process_id,
               p.created_at as process_created,
               collect({
                   relationship: r,
                   connected_id: connected.process_id,
                   created_at: r.created_at
               }) as temporal_relationships
        """

        parameters = {"process_ids": process_ids, "time_window_days": time_window_days}

        results = await self.db_manager.execute_query(query, parameters)

        # Analyze temporal patterns
        temporal_analysis = {
            "time_window_days": time_window_days,
            "network_growth": {},
            "relationship_formation_rate": {},
            "process_activation_timeline": [],
            "network_stability_metrics": {},
        }

        # Track network growth over time
        relationship_timeline = []
        for record in results:
            process_id = record["process_id"]
            process_created = record["process_created"]

            if process_created:
                temporal_analysis["process_activation_timeline"].append(
                    {
                        "process_id": process_id,
                        "activation_time": process_created,
                        "type": "process_creation",
                    }
                )

            for rel_data in record["temporal_relationships"]:
                if rel_data["relationship"] and rel_data["created_at"]:
                    relationship_timeline.append(
                        {
                            "source": process_id,
                            "target": rel_data["connected_id"],
                            "created_at": rel_data["created_at"],
                            "strength": rel_data["relationship"]["strength"],
                        }
                    )

        # Sort by creation time
        relationship_timeline.sort(key=lambda x: x["created_at"])

        # Calculate network growth metrics
        cumulative_relationships = 0
        for i, rel in enumerate(relationship_timeline):
            cumulative_relationships += 1
            temporal_analysis["network_growth"][rel["created_at"]] = {
                "total_relationships": cumulative_relationships,
                "growth_rate": cumulative_relationships / (i + 1) if i > 0 else 1.0,
            }

        return temporal_analysis

    async def detect_anomalous_patterns(
        self, process_ids: List[str], threshold_std_dev: float = 2.0
    ) -> Dict[str, Any]:
        """Detect anomalous patterns in the geological network"""

        # Get network metrics for anomaly detection
        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p.process_id as process_id,
               p.process_type as process_type,
               p.magnitude as magnitude,
               count(DISTINCT connected) as degree,
               avg(r.strength) as avg_relationship_strength,
               stddev(r.strength) as strength_std_dev,
               collect(DISTINCT {
                   connected_id: connected.process_id,
                   strength: r.strength,
                   confidence: r.confidence
               }) as connections
        """

        results = await self.db_manager.execute_query(
            query, {"process_ids": process_ids}
        )

        # Calculate network-wide statistics
        degrees = []
        magnitudes = []
        avg_strengths = []

        process_metrics = {}

        for record in results:
            process_id = record["process_id"]
            degree = record["degree"] or 0
            magnitude = record["magnitude"] or 0
            avg_strength = record["avg_relationship_strength"] or 0

            degrees.append(degree)
            magnitudes.append(magnitude)
            avg_strengths.append(avg_strength)

            process_metrics[process_id] = {
                "degree": degree,
                "magnitude": magnitude,
                "avg_strength": avg_strength,
                "process_type": record["process_type"],
                "connections": record["connections"],
            }

        # Calculate statistical thresholds
        degree_mean = np.mean(degrees) if degrees else 0
        degree_std = np.std(degrees) if degrees else 0
        magnitude_mean = np.mean(magnitudes) if magnitudes else 0
        magnitude_std = np.std(magnitudes) if magnitudes else 0
        strength_mean = np.mean(avg_strengths) if avg_strengths else 0
        strength_std = np.std(avg_strengths) if avg_strengths else 0

        # Detect anomalies
        anomalies = {
            "high_degree_anomalies": [],
            "high_magnitude_anomalies": [],
            "unusual_strength_patterns": [],
            "isolated_processes": [],
            "statistical_thresholds": {
                "degree": {"mean": degree_mean, "std": degree_std},
                "magnitude": {"mean": magnitude_mean, "std": magnitude_std},
                "strength": {"mean": strength_mean, "std": strength_std},
            },
        }

        for process_id, metrics in process_metrics.items():
            # High degree anomaly
            if (
                degree_std > 0
                and metrics["degree"] > degree_mean + threshold_std_dev * degree_std
            ):
                anomalies["high_degree_anomalies"].append(
                    {
                        "process_id": process_id,
                        "degree": metrics["degree"],
                        "z_score": (metrics["degree"] - degree_mean) / degree_std,
                        "process_type": metrics["process_type"],
                    }
                )

            # High magnitude anomaly
            if (
                magnitude_std > 0
                and metrics["magnitude"]
                > magnitude_mean + threshold_std_dev * magnitude_std
            ):
                anomalies["high_magnitude_anomalies"].append(
                    {
                        "process_id": process_id,
                        "magnitude": metrics["magnitude"],
                        "z_score": (metrics["magnitude"] - magnitude_mean)
                        / magnitude_std,
                        "process_type": metrics["process_type"],
                    }
                )

            # Unusual strength patterns
            if (
                strength_std > 0
                and abs(metrics["avg_strength"] - strength_mean)
                > threshold_std_dev * strength_std
            ):
                anomalies["unusual_strength_patterns"].append(
                    {
                        "process_id": process_id,
                        "avg_strength": metrics["avg_strength"],
                        "z_score": (metrics["avg_strength"] - strength_mean)
                        / strength_std,
                        "pattern_type": (
                            "high" if metrics["avg_strength"] > strength_mean else "low"
                        ),
                    }
                )

            # Isolated processes
            if metrics["degree"] == 0:
                anomalies["isolated_processes"].append(
                    {
                        "process_id": process_id,
                        "process_type": metrics["process_type"],
                        "magnitude": metrics["magnitude"],
                    }
                )

        return anomalies

    async def predict_cascade_probability(
        self,
        trigger_process_id: str,
        target_process_ids: List[str],
        time_horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """Predict probability of cascade effects reaching target processes"""

        # Get historical cascade patterns
        historical_query = """
        MATCH path = (trigger:GeologicalProcess {process_id: $trigger_id})-[r:INFLUENCES*1..5]->(target:GeologicalProcess)
        WHERE target.process_id IN $target_ids
        AND ALL(rel in r WHERE rel.relationship_type IN ['causal', 'triggers'])
        RETURN target.process_id as target_id,
               length(path) as path_length,
               [rel in r | rel.strength] as strengths,
               [rel in r | rel.confidence] as confidences,
               [rel in r | rel.time_delay] as time_delays
        """

        parameters = {
            "trigger_id": trigger_process_id,
            "target_ids": target_process_ids,
        }

        historical_results = await self.db_manager.execute_query(
            historical_query, parameters
        )

        # Calculate cascade probabilities
        cascade_predictions = {
            "trigger_process_id": trigger_process_id,
            "time_horizon_days": time_horizon_days,
            "target_predictions": {},
            "overall_cascade_probability": 0.0,
            "expected_cascade_time": None,
        }

        total_probability = 0.0
        cascade_times = []

        for record in historical_results:
            target_id = record["target_id"]
            path_length = record["path_length"]
            strengths = record["strengths"] or []
            confidences = record["confidences"] or []
            time_delays = record["time_delays"] or []

            # Calculate cascade probability based on path strength and confidence
            path_strength = np.prod(strengths) if strengths else 0.0
            path_confidence = np.mean(confidences) if confidences else 0.0
            total_time_delay = sum([td for td in time_delays if td is not None])

            # Probability decreases with path length and time delay
            distance_factor = 1.0 / (1.0 + path_length * 0.2)
            time_factor = (
                1.0 / (1.0 + total_time_delay / time_horizon_days)
                if total_time_delay
                else 1.0
            )

            cascade_probability = (
                path_strength * path_confidence * distance_factor * time_factor
            )

            cascade_predictions["target_predictions"][target_id] = {
                "cascade_probability": cascade_probability,
                "expected_time_days": total_time_delay,
                "path_length": path_length,
                "path_strength": path_strength,
                "confidence": path_confidence,
            }

            total_probability += cascade_probability
            if total_time_delay:
                cascade_times.append(total_time_delay)

        cascade_predictions["overall_cascade_probability"] = min(total_probability, 1.0)
        cascade_predictions["expected_cascade_time"] = (
            np.mean(cascade_times) if cascade_times else None
        )

        return cascade_predictions

    async def analyze_network_resilience(
        self, process_ids: List[str], failure_scenarios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze network resilience to process failures"""

        if not failure_scenarios:
            # Default failure scenarios: remove most critical processes
            from .graph_algorithms import GeologicalGraphAlgorithms
            algorithms = GeologicalGraphAlgorithms(self.db_manager)
            critical_processes = await algorithms.find_critical_processes(
                process_ids, criteria="centrality"
            )
            failure_scenarios = [proc["process_id"] for proc in critical_processes[:3]]

        resilience_analysis = {
            "baseline_network": {},
            "failure_scenarios": {},
            "resilience_metrics": {},
        }

        # Analyze baseline network
        baseline_network = await self.db_manager.analyze_process_network(process_ids)
        resilience_analysis["baseline_network"] = {
            "node_count": baseline_network.node_count,
            "edge_count": baseline_network.edge_count,
            "density": baseline_network.density,
        }

        # Analyze each failure scenario
        for failed_process in failure_scenarios:
            remaining_processes = [pid for pid in process_ids if pid != failed_process]

            if remaining_processes:
                # Analyze network without the failed process
                reduced_network = await self.db_manager.analyze_process_network(
                    remaining_processes
                )

                # Calculate impact metrics
                node_loss = baseline_network.node_count - reduced_network.node_count
                edge_loss = baseline_network.edge_count - reduced_network.edge_count
                density_change = reduced_network.density - baseline_network.density

                resilience_analysis["failure_scenarios"][failed_process] = {
                    "remaining_nodes": reduced_network.node_count,
                    "remaining_edges": reduced_network.edge_count,
                    "new_density": reduced_network.density,
                    "node_loss": node_loss,
                    "edge_loss": edge_loss,
                    "density_change": density_change,
                    "relative_impact": (
                        edge_loss / baseline_network.edge_count
                        if baseline_network.edge_count > 0
                        else 0
                    ),
                }

        # Calculate overall resilience metrics
        if resilience_analysis["failure_scenarios"]:
            impacts = [
                scenario["relative_impact"]
                for scenario in resilience_analysis["failure_scenarios"].values()
            ]
            resilience_analysis["resilience_metrics"] = {
                "average_impact": np.mean(impacts),
                "max_impact": np.max(impacts),
                "resilience_score": 1.0
                - np.mean(impacts),  # Higher score = more resilient
                "critical_failure_threshold": np.max(impacts),
            }

        return resilience_analysis

    async def identify_process_clusters(
        self, process_ids: List[str], clustering_method: str = "spatial_temporal"
    ) -> Dict[str, Any]:
        """Identify clusters of related geological processes"""

        if clustering_method == "spatial_temporal":
            return await self._spatial_temporal_clustering(process_ids)
        elif clustering_method == "process_type":
            return await self._process_type_clustering(process_ids)
        elif clustering_method == "strength_based":
            return await self._strength_based_clustering(process_ids)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")

    async def _spatial_temporal_clustering(
        self, process_ids: List[str]
    ) -> Dict[str, Any]:
        """Cluster processes based on spatial and temporal characteristics"""

        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (l:Location {location_id: p.location_id})
        RETURN p.process_id as process_id,
               p.process_type as process_type,
               p.timescale as timescale,
               p.depth_range as depth_range,
               l.latitude as latitude,
               l.longitude as longitude,
               l.elevation as elevation
        """

        results = await self.db_manager.execute_query(
            query, {"process_ids": process_ids}
        )

        # Group by spatial proximity and temporal scale
        spatial_clusters = defaultdict(list)
        temporal_clusters = defaultdict(list)

        for record in results:
            process_id = record["process_id"]
            timescale = record["timescale"]
            latitude = record["latitude"]
            longitude = record["longitude"]

            # Temporal clustering
            if timescale:
                temporal_clusters[timescale].append(process_id)

            # Spatial clustering (simplified - could use more sophisticated algorithms)
            if latitude is not None and longitude is not None:
                # Round to create spatial bins
                lat_bin = round(latitude, 1)
                lon_bin = round(longitude, 1)
                spatial_key = f"{lat_bin}_{lon_bin}"
                spatial_clusters[spatial_key].append(process_id)

        return {
            "clustering_method": "spatial_temporal",
            "spatial_clusters": dict(spatial_clusters),
            "temporal_clusters": dict(temporal_clusters),
            "cluster_count": len(spatial_clusters) + len(temporal_clusters),
        }

    async def _process_type_clustering(self, process_ids: List[str]) -> Dict[str, Any]:
        """Cluster processes by geological process type"""

        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        RETURN p.process_id as process_id,
               p.process_type as process_type,
               p.magnitude as magnitude
        """

        results = await self.db_manager.execute_query(
            query, {"process_ids": process_ids}
        )

        type_clusters = defaultdict(list)

        for record in results:
            process_type = record["process_type"]
            process_id = record["process_id"]
            type_clusters[process_type].append(
                {"process_id": process_id, "magnitude": record["magnitude"]}
            )

        return {
            "clustering_method": "process_type",
            "clusters": dict(type_clusters),
            "cluster_count": len(type_clusters),
        }

    async def _strength_based_clustering(
        self, process_ids: List[str]
    ) -> Dict[str, Any]:
        """Cluster processes based on relationship strength patterns"""

        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p.process_id as process_id,
               avg(r.strength) as avg_strength,
               count(r) as relationship_count,
               collect(DISTINCT {
                   connected_id: connected.process_id,
                   strength: r.strength
               }) as connections
        """

        results = await self.db_manager.execute_query(
            query, {"process_ids": process_ids}
        )

        # Cluster by strength ranges
        strength_clusters = {
            "high_strength": [],  # avg_strength > 0.7
            "medium_strength": [],  # 0.3 <= avg_strength <= 0.7
            "low_strength": [],  # avg_strength < 0.3
            "isolated": [],  # no connections
        }

        for record in results:
            process_id = record["process_id"]
            avg_strength = record["avg_strength"] or 0
            relationship_count = record["relationship_count"] or 0

            if relationship_count == 0:
                strength_clusters["isolated"].append(process_id)
            elif avg_strength > 0.7:
                strength_clusters["high_strength"].append(process_id)
            elif avg_strength >= 0.3:
                strength_clusters["medium_strength"].append(process_id)
            else:
                strength_clusters["low_strength"].append(process_id)

        return {
            "clustering_method": "strength_based",
            "clusters": strength_clusters,
            "cluster_count": sum(
                1 for cluster in strength_clusters.values() if cluster
            ),
        }

    async def generate_network_report(
        self,
        process_ids: List[str],
        include_predictions: bool = True,
        include_anomalies: bool = True,
        include_resilience: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive network analysis report"""

        report = {
            "report_id": f"network_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "process_count": len(process_ids),
            "analysis_components": [],
        }

        # Basic network analysis
        network_analysis = await self.db_manager.analyze_process_network(process_ids)
        report["basic_network_metrics"] = {
            "node_count": network_analysis.node_count,
            "edge_count": network_analysis.edge_count,
            "density": network_analysis.density,
            "centrality_measures": network_analysis.centrality_measures,
        }
        report["analysis_components"].append("basic_metrics")

        # Temporal evolution analysis
        try:
            temporal_analysis = await self.analyze_temporal_evolution(process_ids)
            report["temporal_evolution"] = temporal_analysis
            report["analysis_components"].append("temporal_evolution")
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")

        # Anomaly detection
        if include_anomalies:
            try:
                anomalies = await self.detect_anomalous_patterns(process_ids)
                report["anomaly_detection"] = anomalies
                report["analysis_components"].append("anomaly_detection")
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        # Resilience analysis
        if include_resilience:
            try:
                resilience = await self.analyze_network_resilience(process_ids)
                report["resilience_analysis"] = resilience
                report["analysis_components"].append("resilience_analysis")
            except Exception as e:
                logger.warning(f"Resilience analysis failed: {e}")

        # Process clustering
        try:
            clusters = await self.identify_process_clusters(process_ids)
            report["process_clustering"] = clusters
            report["analysis_components"].append("process_clustering")
        except Exception as e:
            logger.warning(f"Process clustering failed: {e}")

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results"""

        recommendations = []

        # Basic network recommendations
        if "basic_network_metrics" in report:
            metrics = report["basic_network_metrics"]

            if metrics["density"] < 0.1:
                recommendations.append(
                    "Network density is low - consider identifying missing relationships"
                )

            if metrics["edge_count"] == 0:
                recommendations.append(
                    "No relationships found - network may need relationship discovery"
                )

        # Anomaly-based recommendations
        if "anomaly_detection" in report:
            anomalies = report["anomaly_detection"]

            if anomalies.get("isolated_processes"):
                recommendations.append(
                    f"Found {len(anomalies['isolated_processes'])} isolated processes - investigate potential connections"
                )

            if anomalies.get("high_degree_anomalies"):
                recommendations.append(
                    "High-degree processes detected - monitor for potential cascade effects"
                )

        # Resilience recommendations
        if "resilience_analysis" in report:
            resilience = report["resilience_analysis"]

            if (
                "resilience_metrics" in resilience
                and resilience["resilience_metrics"]["resilience_score"] < 0.5
            ):
                recommendations.append(
                    "Network shows low resilience - consider redundant pathways"
                )

        # Clustering recommendations
        if "process_clustering" in report:
            clustering = report["process_clustering"]
            node_count = report.get("basic_network_metrics", {}).get("node_count", 0)

            if clustering["cluster_count"] > node_count * 0.5:
                recommendations.append(
                    "High cluster fragmentation - consider consolidation strategies"
                )

        return recommendations
