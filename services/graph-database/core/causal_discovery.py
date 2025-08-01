"""
Advanced causal relationship discovery for geological processes
"""

import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .database import GraphDatabaseManager

logger = logging.getLogger(__name__)


class CausalDiscoveryEngine:
    """Advanced causal relationship discovery for geological processes"""

    def __init__(self, db_manager: GraphDatabaseManager):
        self.db_manager = db_manager

    async def discover_hidden_causality(
        self,
        process_ids: List[str],
        confidence_threshold: float = 0.7,
        strength_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Discover hidden causal relationships using graph analysis"""

        # Get all existing relationships
        existing_query = """
        MATCH (p1:GeologicalProcess)-[r:INFLUENCES]->(p2:GeologicalProcess)
        WHERE p1.process_id IN $process_ids AND p2.process_id IN $process_ids
        AND r.relationship_type = 'causal'
        RETURN p1.process_id as source, p2.process_id as target, r.strength as strength
        """

        existing_relationships = await self.db_manager.execute_query(
            existing_query, {"process_ids": process_ids}
        )

        # Build existing causal graph
        existing_causality = set()
        for rel in existing_relationships:
            existing_causality.add((rel["source"], rel["target"]))

        # Discover potential hidden relationships
        hidden_relationships = []

        # Method 1: Transitive causality discovery
        transitive_relationships = await self._discover_transitive_causality(
            process_ids, existing_causality, confidence_threshold
        )
        hidden_relationships.extend(transitive_relationships)

        # Method 2: Common cause discovery
        common_cause_relationships = await self._discover_common_causes(
            process_ids, existing_causality, strength_threshold
        )
        hidden_relationships.extend(common_cause_relationships)

        # Method 3: Temporal pattern analysis
        temporal_relationships = await self._discover_temporal_causality(
            process_ids, existing_causality
        )
        hidden_relationships.extend(temporal_relationships)

        return {
            "existing_causal_relationships": len(existing_relationships),
            "discovered_hidden_relationships": len(hidden_relationships),
            "hidden_relationships": hidden_relationships,
            "discovery_methods": ["transitive", "common_cause", "temporal"],
            "confidence_threshold": confidence_threshold,
            "strength_threshold": strength_threshold,
        }

    async def _discover_transitive_causality(
        self,
        process_ids: List[str],
        existing_causality: Set[Tuple[str, str]],
        confidence_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Discover transitive causal relationships (A->B->C implies A->C)"""

        # Find potential transitive relationships
        transitive_query = """
        MATCH (a:GeologicalProcess)-[r1:INFLUENCES]->(b:GeologicalProcess)-[r2:INFLUENCES]->(c:GeologicalProcess)
        WHERE a.process_id IN $process_ids
        AND b.process_id IN $process_ids
        AND c.process_id IN $process_ids
        AND r1.relationship_type = 'causal'
        AND r2.relationship_type = 'causal'
        AND a.process_id <> c.process_id
        RETURN a.process_id as source,
               c.process_id as target,
               b.process_id as intermediate,
               r1.strength as strength1,
               r2.strength as strength2,
               r1.confidence as confidence1,
               r2.confidence as confidence2,
               r1.time_delay as delay1,
               r2.time_delay as delay2
        """

        results = await self.db_manager.execute_query(
            transitive_query, {"process_ids": process_ids}
        )

        transitive_relationships = []

        for record in results:
            source = record["source"]
            target = record["target"]
            intermediate = record["intermediate"]

            # Skip if direct relationship already exists
            if (source, target) in existing_causality:
                continue

            # Calculate transitive strength and confidence
            strength1 = record["strength1"] or 0
            strength2 = record["strength2"] or 0
            confidence1 = record["confidence1"] or 0
            confidence2 = record["confidence2"] or 0

            # Transitive strength is the product of individual strengths
            transitive_strength = strength1 * strength2
            transitive_confidence = min(
                confidence1, confidence2
            )  # Conservative estimate

            # Calculate total time delay
            delay1 = record["delay1"] or 0
            delay2 = record["delay2"] or 0
            total_delay = delay1 + delay2

            if transitive_confidence >= confidence_threshold:
                transitive_relationships.append(
                    {
                        "source_process_id": source,
                        "target_process_id": target,
                        "relationship_type": "transitive_causal",
                        "strength": transitive_strength,
                        "confidence": transitive_confidence,
                        "discovery_method": "transitive",
                        "intermediate_process": intermediate,
                        "total_time_delay": total_delay,
                        "evidence": {
                            "path_strength1": strength1,
                            "path_strength2": strength2,
                            "path_confidence1": confidence1,
                            "path_confidence2": confidence2,
                        },
                    }
                )

        return transitive_relationships

    async def _discover_common_causes(
        self,
        process_ids: List[str],
        existing_causality: Set[Tuple[str, str]],
        strength_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Discover relationships through common causes"""

        # Find processes with common causes
        common_cause_query = """
        MATCH (cause:GeologicalProcess)-[r1:INFLUENCES]->(effect1:GeologicalProcess),
              (cause)-[r2:INFLUENCES]->(effect2:GeologicalProcess)
        WHERE cause.process_id IN $process_ids
        AND effect1.process_id IN $process_ids
        AND effect2.process_id IN $process_ids
        AND effect1.process_id <> effect2.process_id
        AND r1.relationship_type = 'causal'
        AND r2.relationship_type = 'causal'
        RETURN cause.process_id as common_cause,
               effect1.process_id as effect1,
               effect2.process_id as effect2,
               r1.strength as strength1,
               r2.strength as strength2,
               r1.confidence as confidence1,
               r2.confidence as confidence2,
               effect1.process_type as type1,
               effect2.process_type as type2
        """

        results = await self.db_manager.execute_query(
            common_cause_query, {"process_ids": process_ids}
        )

        common_cause_relationships = []

        for record in results:
            effect1 = record["effect1"]
            effect2 = record["effect2"]
            common_cause = record["common_cause"]

            # Skip if direct relationship already exists
            if (effect1, effect2) in existing_causality or (
                effect2,
                effect1,
            ) in existing_causality:
                continue

            strength1 = record["strength1"] or 0
            strength2 = record["strength2"] or 0
            confidence1 = record["confidence1"] or 0
            confidence2 = record["confidence2"] or 0

            # Calculate correlation strength based on common cause
            correlation_strength = min(strength1, strength2)
            correlation_confidence = min(confidence1, confidence2)

            if correlation_strength >= strength_threshold:
                # Determine direction based on process types and strengths
                if strength1 > strength2:
                    source, target = effect1, effect2
                    relationship_strength = (
                        correlation_strength * 0.8
                    )  # Reduced for indirect relationship
                else:
                    source, target = effect2, effect1
                    relationship_strength = correlation_strength * 0.8

                common_cause_relationships.append(
                    {
                        "source_process_id": source,
                        "target_process_id": target,
                        "relationship_type": "common_cause_correlation",
                        "strength": relationship_strength,
                        "confidence": correlation_confidence,
                        "discovery_method": "common_cause",
                        "common_cause_process": common_cause,
                        "evidence": {
                            "cause_to_source_strength": (
                                strength1 if source == effect1 else strength2
                            ),
                            "cause_to_target_strength": (
                                strength2 if source == effect1 else strength1
                            ),
                            "cause_to_source_confidence": (
                                confidence1 if source == effect1 else confidence2
                            ),
                            "cause_to_target_confidence": (
                                confidence2 if source == effect1 else confidence1
                            ),
                        },
                    }
                )

        return common_cause_relationships

    async def _discover_temporal_causality(
        self, process_ids: List[str], existing_causality: Set[Tuple[str, str]]
    ) -> List[Dict[str, Any]]:
        """Discover causal relationships based on temporal patterns"""

        # Get temporal information about processes
        temporal_query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p.process_id as process_id,
               p.timescale as timescale,
               p.created_at as created_at,
               p.process_type as process_type,
               collect({
                   connected_id: connected.process_id,
                   time_delay: r.time_delay,
                   relationship_type: r.relationship_type
               }) as temporal_connections
        """

        results = await self.db_manager.execute_query(
            temporal_query, {"process_ids": process_ids}
        )

        # Analyze temporal patterns
        temporal_relationships = []
        process_temporal_data = {}

        for record in results:
            process_id = record["process_id"]
            process_temporal_data[process_id] = {
                "timescale": record["timescale"],
                "created_at": record["created_at"],
                "process_type": record["process_type"],
                "connections": record["temporal_connections"],
            }

        # Find temporal causality patterns
        for process1_id, data1 in process_temporal_data.items():
            for process2_id, data2 in process_temporal_data.items():
                if process1_id == process2_id:
                    continue

                # Skip if relationship already exists
                if (process1_id, process2_id) in existing_causality:
                    continue

                # Analyze temporal precedence
                temporal_score = self._calculate_temporal_causality_score(data1, data2)

                if temporal_score > 0.6:  # Threshold for temporal causality
                    temporal_relationships.append(
                        {
                            "source_process_id": process1_id,
                            "target_process_id": process2_id,
                            "relationship_type": "temporal_causal",
                            "strength": temporal_score * 0.7,  # Conservative strength
                            "confidence": temporal_score,
                            "discovery_method": "temporal",
                            "evidence": {
                                "temporal_score": temporal_score,
                                "source_timescale": data1["timescale"],
                                "target_timescale": data2["timescale"],
                            },
                        }
                    )

        return temporal_relationships

    def _calculate_temporal_causality_score(
        self, data1: Dict[str, Any], data2: Dict[str, Any]
    ) -> float:
        """Calculate temporal causality score between two processes"""

        score = 0.0

        # Timescale analysis
        timescale1 = data1.get("timescale", "")
        timescale2 = data2.get("timescale", "")

        # Define timescale hierarchy (faster -> slower)
        timescale_hierarchy = {
            "seconds": 1,
            "minutes": 2,
            "hours": 3,
            "days": 4,
            "weeks": 5,
            "months": 6,
            "years": 7,
            "decades": 8,
            "centuries": 9,
            "thousands_of_years": 10,
            "millions_of_years": 11,
        }

        scale1 = timescale_hierarchy.get(timescale1, 5)
        scale2 = timescale_hierarchy.get(timescale2, 5)

        # Faster processes can cause slower processes
        if scale1 < scale2:
            score += 0.4
        elif scale1 == scale2:
            score += 0.2

        # Creation time analysis
        created1 = data1.get("created_at")
        created2 = data2.get("created_at")

        if created1 and created2:
            if created1 < created2:
                score += 0.3
            elif created1 == created2:
                score += 0.1

        # Process type compatibility
        type1 = data1.get("process_type", "")
        type2 = data2.get("process_type", "")

        # Define causal compatibility between process types
        causal_compatibility = {
            ("tectonic", "seismic"): 0.3,
            ("magmatic", "volcanic"): 0.3,
            ("volcanic", "hydrothermal"): 0.3,
            ("seismic", "deformational"): 0.2,
            ("hydrothermal", "geochemical"): 0.2,
            ("tectonic", "volcanic"): 0.2,
            ("magmatic", "seismic"): 0.2,
        }

        compatibility = causal_compatibility.get((type1, type2), 0.0)
        score += compatibility

        return min(score, 1.0)

    async def validate_causal_relationships(
        self,
        proposed_relationships: List[Dict[str, Any]],
        validation_method: str = "statistical",
    ) -> Dict[str, Any]:
        """Validate proposed causal relationships"""

        if validation_method == "statistical":
            return await self._statistical_validation(proposed_relationships)
        elif validation_method == "domain_knowledge":
            return await self._domain_knowledge_validation(proposed_relationships)
        else:
            raise ValueError(f"Unsupported validation method: {validation_method}")

    async def _statistical_validation(
        self, proposed_relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate relationships using statistical methods"""

        validation_results = {
            "validation_method": "statistical",
            "total_relationships": len(proposed_relationships),
            "validated_relationships": [],
            "rejected_relationships": [],
            "validation_metrics": {},
        }

        for relationship in proposed_relationships:
            source_id = relationship["source_process_id"]
            target_id = relationship["target_process_id"]

            # Get statistical data for validation
            stats_query = """
            MATCH (source:GeologicalProcess {process_id: $source_id})
            MATCH (target:GeologicalProcess {process_id: $target_id})
            OPTIONAL MATCH (source)-[existing:INFLUENCES]->(target)
            RETURN source.magnitude as source_magnitude,
                   target.magnitude as target_magnitude,
                   source.confidence_level as source_confidence,
                   target.confidence_level as target_confidence,
                   existing.strength as existing_strength
            """

            stats_results = await self.db_manager.execute_query(
                stats_query, {"source_id": source_id, "target_id": target_id}
            )

            if stats_results:
                stats = stats_results[0]

                # Calculate validation score
                validation_score = self._calculate_statistical_validation_score(
                    relationship, stats
                )

                relationship["validation_score"] = validation_score
                relationship["validation_evidence"] = stats

                if validation_score > 0.6:
                    validation_results["validated_relationships"].append(relationship)
                else:
                    validation_results["rejected_relationships"].append(relationship)

        # Calculate overall validation metrics
        if validation_results["validated_relationships"]:
            avg_validation_score = np.mean(
                [
                    rel["validation_score"]
                    for rel in validation_results["validated_relationships"]
                ]
            )
            validation_results["validation_metrics"][
                "average_validation_score"
            ] = avg_validation_score

        validation_results["validation_metrics"]["validation_rate"] = (
            len(validation_results["validated_relationships"])
            / len(proposed_relationships)
            if proposed_relationships
            else 0
        )

        return validation_results

    def _calculate_statistical_validation_score(
        self, relationship: Dict[str, Any], stats: Dict[str, Any]
    ) -> float:
        """Calculate statistical validation score for a relationship"""

        score = 0.0

        # Magnitude correlation
        source_mag = stats.get("source_magnitude", 0) or 0
        target_mag = stats.get("target_magnitude", 0) or 0

        if source_mag > 0 and target_mag > 0:
            # Higher source magnitude should correlate with relationship strength
            magnitude_factor = min(source_mag / 10.0, 1.0)  # Normalize to 0-1
            score += magnitude_factor * 0.3

        # Confidence levels
        source_conf = stats.get("source_confidence", 0) or 0
        target_conf = stats.get("target_confidence", 0) or 0

        avg_confidence = (source_conf + target_conf) / 2
        score += avg_confidence * 0.3

        # Relationship strength consistency
        proposed_strength = relationship.get("strength", 0)
        existing_strength = stats.get("existing_strength")

        if existing_strength is not None:
            # If there's an existing relationship, check consistency
            strength_diff = abs(proposed_strength - existing_strength)
            consistency_score = max(0, 1.0 - strength_diff)
            score += consistency_score * 0.2
        else:
            # No existing relationship, use proposed strength
            score += proposed_strength * 0.2

        # Discovery method reliability
        method = relationship.get("discovery_method", "")
        method_reliability = {"transitive": 0.2, "common_cause": 0.15, "temporal": 0.1}

        score += method_reliability.get(method, 0.05)

        return min(score, 1.0)

    async def _domain_knowledge_validation(
        self, proposed_relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate relationships using geological domain knowledge"""

        # Define domain knowledge rules
        domain_rules = {
            # Process type compatibility rules
            "process_compatibility": {
                ("tectonic", "seismic"): 0.9,
                ("magmatic", "volcanic"): 0.9,
                ("volcanic", "hydrothermal"): 0.8,
                ("seismic", "deformational"): 0.7,
                ("hydrothermal", "geochemical"): 0.7,
                ("tectonic", "volcanic"): 0.6,
                ("magmatic", "seismic"): 0.6,
                ("volcanic", "seismic"): 0.5,
                ("deformational", "hydrothermal"): 0.4,
            },
            # Timescale compatibility
            "timescale_compatibility": {
                ("millions_of_years", "thousands_of_years"): 0.8,
                ("thousands_of_years", "years"): 0.7,
                ("years", "months"): 0.6,
                ("months", "days"): 0.5,
            },
        }

        validation_results = {
            "validation_method": "domain_knowledge",
            "total_relationships": len(proposed_relationships),
            "validated_relationships": [],
            "rejected_relationships": [],
            "domain_rules_applied": list(domain_rules.keys()),
        }

        for relationship in proposed_relationships:
            # Get process information for domain validation
            source_id = relationship["source_process_id"]
            target_id = relationship["target_process_id"]

            domain_query = """
            MATCH (source:GeologicalProcess {process_id: $source_id})
            MATCH (target:GeologicalProcess {process_id: $target_id})
            RETURN source.process_type as source_type,
                   target.process_type as target_type,
                   source.timescale as source_timescale,
                   target.timescale as target_timescale
            """

            domain_results = await self.db_manager.execute_query(
                domain_query, {"source_id": source_id, "target_id": target_id}
            )

            if domain_results:
                domain_data = domain_results[0]

                # Apply domain knowledge rules
                domain_score = self._apply_domain_rules(
                    relationship, domain_data, domain_rules
                )

                relationship["domain_validation_score"] = domain_score
                relationship["domain_evidence"] = domain_data

                if domain_score > 0.5:
                    validation_results["validated_relationships"].append(relationship)
                else:
                    validation_results["rejected_relationships"].append(relationship)

        return validation_results

    def _apply_domain_rules(
        self,
        relationship: Dict[str, Any],
        domain_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> float:
        """Apply domain knowledge rules to validate a relationship"""

        score = 0.0

        # Process type compatibility
        source_type = domain_data.get("source_type", "")
        target_type = domain_data.get("target_type", "")

        process_compatibility = domain_rules["process_compatibility"]
        compatibility_score = process_compatibility.get((source_type, target_type), 0.0)
        score += compatibility_score * 0.6

        # Timescale compatibility
        source_timescale = domain_data.get("source_timescale", "")
        target_timescale = domain_data.get("target_timescale", "")

        timescale_compatibility = domain_rules["timescale_compatibility"]
        timescale_score = timescale_compatibility.get(
            (source_timescale, target_timescale), 0.0
        )
        score += timescale_score * 0.4

        return min(score, 1.0)
