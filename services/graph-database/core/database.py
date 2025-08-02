"""
Neo4j Graph Database Manager for Geological Relationships
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from models.geological_models import (CausalRelationship, GeologicalProcess,
                                      GeologicalRelationship,
                                      NetworkAnalysisResult)
from neo4j import AsyncDriver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from .config import settings

logger = logging.getLogger(__name__)


class GraphDatabaseManager:
    """Manages Neo4j database connections and operations"""

    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self.connected = False

    async def connect(self) -> None:
        """Establish connection to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_lifetime=settings.max_connection_lifetime,
                max_connection_pool_size=settings.max_connection_pool_size,
                connection_acquisition_timeout=settings.connection_acquisition_timeout,
            )

            # Verify connectivity
            await self.driver.verify_connectivity()
            self.connected = True
            logger.info("Successfully connected to Neo4j database")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close database connection"""
        if self.driver:
            await self.driver.close()
            self.connected = False
            logger.info("Neo4j connection closed")

    async def is_connected(self) -> bool:
        """Check if database is connected"""
        if not self.driver or not self.connected:
            return False

        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            self.connected = False
            return False

    async def is_ready(self) -> bool:
        """Check if database is ready for operations"""
        if not await self.is_connected():
            return False

        try:
            async with self.driver.session(database=settings.neo4j_database) as session:
                result = await session.run("RETURN 1 as test")
                await result.single()
                return True
        except Exception:
            return False

    async def initialize_schema(self) -> None:
        """Initialize graph database schema and constraints"""
        schema_queries = [
            # Create constraints for unique identifiers
            "CREATE CONSTRAINT process_id_unique IF NOT EXISTS FOR (p:GeologicalProcess) REQUIRE p.process_id IS UNIQUE",
            "CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            # Create indexes for performance
            "CREATE INDEX process_type_index IF NOT EXISTS FOR (p:GeologicalProcess) ON (p.process_type)",
            "CREATE INDEX location_coords_index IF NOT EXISTS FOR (l:Location) ON (l.latitude, l.longitude)",
            "CREATE INDEX event_timestamp_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            "CREATE INDEX relationship_strength_index IF NOT EXISTS FOR ()-[r:INFLUENCES]-() ON (r.strength)",
            # Create full-text search indexes
            "CREATE FULLTEXT INDEX process_search_index IF NOT EXISTS FOR (p:GeologicalProcess) ON EACH [p.name, p.description]",
            "CREATE FULLTEXT INDEX location_search_index IF NOT EXISTS FOR (l:Location) ON EACH [l.name, l.region]",
        ]

        async with self.driver.session(database=settings.neo4j_database) as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                    logger.info(f"Executed schema query: {query}")
                except Exception as e:
                    logger.warning(
                        f"Schema query failed (may already exist): {query} - {e}"
                    )

    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results"""
        if not self.driver:
            raise RuntimeError("Database not connected")

        timeout = timeout or settings.query_timeout
        parameters = parameters or {}

        async with self.driver.session(database=settings.neo4j_database) as session:
            try:
                result = await session.run(query, parameters, timeout=timeout)
                records = await result.data()
                return records
            except (ServiceUnavailable, TransientError) as e:
                logger.error(f"Transient database error: {e}")
                raise
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise

    async def execute_write_transaction(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a write transaction"""
        if not self.driver:
            raise RuntimeError("Database not connected")

        parameters = parameters or {}

        async with self.driver.session(database=settings.neo4j_database) as session:
            try:
                result = await session.execute_write(
                    self._write_transaction, query, parameters
                )
                return result
            except Exception as e:
                logger.error(f"Write transaction failed: {e}")
                raise

    @staticmethod
    async def _write_transaction(tx, query: str, parameters: Dict[str, Any]):
        """Helper method for write transactions"""
        result = await tx.run(query, parameters)
        return await result.single()

    async def create_geological_process(self, process: GeologicalProcess) -> str:
        """Create a geological process node"""
        query = """
        CREATE (p:GeologicalProcess {
            process_id: $process_id,
            name: $name,
            process_type: $process_type,
            description: $description,
            location_id: $location_id,
            depth_range: $depth_range,
            temperature_range: $temperature_range,
            pressure_range: $pressure_range,
            timescale: $timescale,
            created_at: datetime(),
            updated_at: datetime()
        })
        RETURN p.process_id as process_id
        """

        parameters = {
            "process_id": process.process_id,
            "name": process.name,
            "process_type": process.process_type,
            "description": process.description,
            "location_id": process.location_id,
            "depth_range": process.depth_range,
            "temperature_range": process.temperature_range,
            "pressure_range": process.pressure_range,
            "timescale": process.timescale,
        }

        result = await self.execute_write_transaction(query, parameters)
        return result["process_id"]

    async def create_relationship(self, relationship: GeologicalRelationship) -> str:
        """Create a relationship between geological processes"""
        query = """
        MATCH (source:GeologicalProcess {process_id: $source_id})
        MATCH (target:GeologicalProcess {process_id: $target_id})
        CREATE (source)-[r:INFLUENCES {
            relationship_id: $relationship_id,
            relationship_type: $relationship_type,
            strength: $strength,
            confidence: $confidence,
            mechanism: $mechanism,
            time_delay: $time_delay,
            spatial_scale: $spatial_scale,
            created_at: datetime(),
            updated_at: datetime()
        }]->(target)
        RETURN r.relationship_id as relationship_id
        """

        parameters = {
            "source_id": relationship.source_process_id,
            "target_id": relationship.target_process_id,
            "relationship_id": relationship.relationship_id,
            "relationship_type": relationship.relationship_type,
            "strength": relationship.strength,
            "confidence": relationship.confidence,
            "mechanism": relationship.mechanism,
            "time_delay": relationship.time_delay,
            "spatial_scale": relationship.spatial_scale,
        }

        result = await self.execute_write_transaction(query, parameters)
        return result["relationship_id"]

    async def find_causal_chains(
        self, start_process_id: str, max_depth: int = 5
    ) -> List[CausalRelationship]:
        """Find causal chains starting from a specific process"""
        query = """
        MATCH path = (start:GeologicalProcess {process_id: $start_id})-[r:INFLUENCES*1..$max_depth]->(end:GeologicalProcess)
        WHERE ALL(rel in r WHERE rel.relationship_type = 'causal')
        RETURN path,
               [rel in r | rel.strength] as strengths,
               [rel in r | rel.confidence] as confidences,
               length(path) as chain_length
        ORDER BY chain_length, reduce(s = 1.0, strength in strengths | s * strength) DESC
        LIMIT 100
        """

        parameters = {"start_id": start_process_id, "max_depth": max_depth}

        results = await self.execute_query(query, parameters)

        causal_chains = []
        for record in results:
            path = record["path"]
            strengths = record["strengths"]
            confidences = record["confidences"]

            # Extract process chain
            processes = []
            relationships = []

            for i, node in enumerate(path.nodes):
                processes.append(
                    {
                        "process_id": node["process_id"],
                        "name": node["name"],
                        "process_type": node["process_type"],
                    }
                )

            for i, rel in enumerate(path.relationships):
                relationships.append(
                    {
                        "relationship_id": rel["relationship_id"],
                        "strength": strengths[i],
                        "confidence": confidences[i],
                        "mechanism": rel["mechanism"],
                    }
                )

            causal_chains.append(
                CausalRelationship(
                    chain_id=f"chain_{start_process_id}_{len(causal_chains)}",
                    process_chain=processes,
                    relationships=relationships,
                    total_strength=(
                        sum(strengths) / len(strengths) if strengths else 0.0
                    ),
                    confidence_score=(
                        sum(confidences) / len(confidences) if confidences else 0.0
                    ),
                )
            )

        return causal_chains

    async def analyze_process_network(
        self, process_ids: List[str]
    ) -> NetworkAnalysisResult:
        """Analyze network properties of geological processes"""
        # Get network structure
        network_query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p, collect(DISTINCT {
            relationship: r,
            connected_process: connected
        }) as connections
        """

        # Calculate centrality measures
        centrality_query = """
        CALL gds.graph.project(
            'geological_network',
            'GeologicalProcess',
            'INFLUENCES',
            {
                nodeProperties: ['process_type'],
                relationshipProperties: ['strength']
            }
        )
        YIELD graphName, nodeCount, relationshipCount

        CALL gds.pageRank.stream('geological_network')
        YIELD nodeId, score as pagerank_score

        CALL gds.betweenness.stream('geological_network')
        YIELD nodeId, score as betweenness_score

        CALL gds.closeness.stream('geological_network')
        YIELD nodeId, score as closeness_score

        MATCH (n) WHERE id(n) = nodeId
        RETURN n.process_id as process_id,
               pagerank_score,
               betweenness_score,
               closeness_score
        """

        parameters = {"process_ids": process_ids}

        try:
            # Get network structure
            network_results = await self.execute_query(network_query, parameters)

            # Calculate centrality measures (if GDS plugin is available)
            try:
                centrality_results = await self.execute_query(centrality_query)
                centrality_map = {
                    r["process_id"]: {
                        "pagerank": r["pagerank_score"],
                        "betweenness": r["betweenness_score"],
                        "closeness": r["closeness_score"],
                    }
                    for r in centrality_results
                }
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {e}")
                centrality_map = {}

            # Build network analysis result
            nodes = []
            edges = []

            for record in network_results:
                process = record["p"]
                connections = record["connections"]

                # Add node with centrality measures
                node_data = {
                    "process_id": process["process_id"],
                    "name": process["name"],
                    "process_type": process["process_type"],
                    "centrality_measures": centrality_map.get(
                        process["process_id"], {}
                    ),
                }
                nodes.append(node_data)

                # Add edges
                for conn in connections:
                    if conn["relationship"] and conn["connected_process"]:
                        edge_data = {
                            "source": process["process_id"],
                            "target": conn["connected_process"]["process_id"],
                            "relationship_type": conn["relationship"][
                                "relationship_type"
                            ],
                            "strength": conn["relationship"]["strength"],
                            "confidence": conn["relationship"]["confidence"],
                        }
                        edges.append(edge_data)

            return NetworkAnalysisResult(
                network_id=f"network_{datetime.now().isoformat()}",
                nodes=nodes,
                edges=edges,
                node_count=len(nodes),
                edge_count=len(edges),
                density=(
                    len(edges) / (len(nodes) * (len(nodes) - 1))
                    if len(nodes) > 1
                    else 0.0
                ),
                centrality_measures=centrality_map,
                analysis_timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            raise

    async def get_process_by_id(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get a geological process by ID"""
        query = """
        MATCH (p:GeologicalProcess {process_id: $process_id})
        RETURN p
        """

        results = await self.execute_query(query, {"process_id": process_id})
        return results[0]["p"] if results else None

    async def search_processes(
        self, search_term: str, process_type: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search geological processes using full-text search"""
        if process_type:
            query = """
            CALL db.index.fulltext.queryNodes('process_search_index', $search_term)
            YIELD node, score
            WHERE node.process_type = $process_type
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            parameters = {
                "search_term": search_term,
                "process_type": process_type,
                "limit": limit,
            }
        else:
            query = """
            CALL db.index.fulltext.queryNodes('process_search_index', $search_term)
            YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            LIMIT $limit
            """
            parameters = {"search_term": search_term, "limit": limit}

        results = await self.execute_query(query, parameters)
        return [
            {"process": record["node"], "relevance_score": record["score"]}
            for record in results
        ]
