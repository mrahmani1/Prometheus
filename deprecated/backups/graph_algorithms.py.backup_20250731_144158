"""
Graph algorithms for geological network analysis
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import math

from .database import GraphDatabaseManager
from models.geological_models import NetworkAnalysisResult, CausalRelationship

logger = logging.getLogger(__name__)


class GeologicalGraphAlgorithms:
    """Graph algorithms specialized for geological process networks"""
    
    def __init__(self, db_manager: GraphDatabaseManager):
        self.db_manager = db_manager
    
    async def find_shortest_paths(
        self, 
        source_id: str, 
        target_id: str,
        relationship_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two geological processes"""
        
        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH (source:GeologicalProcess {{process_id: $source_id}})
        MATCH (target:GeologicalProcess {{process_id: $target_id}})
        MATCH path = shortestPath((source)-[r{rel_filter}*]-(target))
        RETURN path,
               length(path) as path_length,
               [rel in relationships(path) | rel.strength] as strengths,
               [rel in relationships(path) | rel.confidence] as confidences,
               reduce(s = 1.0, strength in [rel in relationships(path) | rel.strength] | s * strength) as total_strength
        ORDER BY path_length, total_strength DESC
        LIMIT 10
        """
        
        parameters = {
            "source_id": source_id,
            "target_id": target_id
        }
        
        results = await self.db_manager.execute_query(query, parameters)
        
        paths = []
        for record in results:
            path = record["path"]
            path_data = {
                "path_id": f"path_{source_id}_{target_id}_{len(paths)}",
                "source_id": source_id,
                "target_id": target_id,
                "length": record["path_length"],
                "total_strength": record["total_strength"],
                "average_confidence": sum(record["confidences"]) / len(record["confidences"]) if record["confidences"] else 0.0,
                "nodes": [{"process_id": node["process_id"], "name": node["name"]} for node in path.nodes],
                "relationships": [
                    {
                        "relationship_type": rel["relationship_type"],
                        "strength": rel["strength"],
                        "confidence": rel["confidence"]
                    }
                    for rel in path.relationships
                ]
            }
            paths.append(path_data)
        
        return paths
    
    async def detect_communities(
        self, 
        process_ids: List[str],
        algorithm: str = "louvain"
    ) -> Dict[str, List[str]]:
        """Detect communities in the geological process network"""
        
        if algorithm == "louvain":
            return await self._louvain_community_detection(process_ids)
        elif algorithm == "label_propagation":
            return await self._label_propagation_communities(process_ids)
        else:
            raise ValueError(f"Unsupported community detection algorithm: {algorithm}")
    
    async def _louvain_community_detection(self, process_ids: List[str]) -> Dict[str, List[str]]:
        """Louvain community detection using Neo4j GDS"""
        
        # Create temporary graph projection
        projection_query = """
        CALL gds.graph.project(
            'temp_geological_network',
            {
                GeologicalProcess: {
                    filter: "n.process_id IN $process_ids"
                }
            },
            {
                INFLUENCES: {
                    properties: ['strength']
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        
        # Run Louvain algorithm
        louvain_query = """
        CALL gds.louvain.stream('temp_geological_network', {
            relationshipWeightProperty: 'strength'
        })
        YIELD nodeId, communityId
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, communityId
        """
        
        # Cleanup query
        cleanup_query = "CALL gds.graph.drop('temp_geological_network')"
        
        try:
            # Create projection
            await self.db_manager.execute_query(projection_query, {"process_ids": process_ids})
            
            # Run community detection
            results = await self.db_manager.execute_query(louvain_query)
            
            # Group processes by community
            communities = defaultdict(list)
            for record in results:
                community_id = f"community_{record['communityId']}"
                communities[community_id].append(record["process_id"])
            
            return dict(communities)
            
        except Exception as e:
            logger.warning(f"Louvain algorithm failed, falling back to simple clustering: {e}")
            return await self._simple_clustering(process_ids)
        
        finally:
            # Cleanup
            try:
                await self.db_manager.execute_query(cleanup_query)
            except Exception:
                pass
    
    async def _label_propagation_communities(self, process_ids: List[str]) -> Dict[str, List[str]]:
        """Label propagation community detection"""
        
        projection_query = """
        CALL gds.graph.project(
            'temp_lp_network',
            {
                GeologicalProcess: {
                    filter: "n.process_id IN $process_ids"
                }
            },
            'INFLUENCES'
        )
        """
        
        lp_query = """
        CALL gds.labelPropagation.stream('temp_lp_network')
        YIELD nodeId, communityId
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, communityId
        """
        
        cleanup_query = "CALL gds.graph.drop('temp_lp_network')"
        
        try:
            await self.db_manager.execute_query(projection_query, {"process_ids": process_ids})
            results = await self.db_manager.execute_query(lp_query)
            
            communities = defaultdict(list)
            for record in results:
                community_id = f"community_{record['communityId']}"
                communities[community_id].append(record["process_id"])
            
            return dict(communities)
            
        except Exception as e:
            logger.warning(f"Label propagation failed, falling back to simple clustering: {e}")
            return await self._simple_clustering(process_ids)
        
        finally:
            try:
                await self.db_manager.execute_query(cleanup_query)
            except Exception:
                pass
    
    async def _simple_clustering(self, process_ids: List[str]) -> Dict[str, List[str]]:
        """Simple clustering based on process types and spatial proximity"""
        
        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p, collect(DISTINCT connected.process_id) as connected_ids
        """
        
        results = await self.db_manager.execute_query(query, {"process_ids": process_ids})
        
        # Build adjacency list
        adjacency = {}
        process_types = {}
        
        for record in results:
            process = record["p"]
            process_id = process["process_id"]
            adjacency[process_id] = record["connected_ids"]
            process_types[process_id] = process["process_type"]
        
        # Simple clustering: group by process type and connectivity
        communities = defaultdict(list)
        visited = set()
        
        for process_id in process_ids:
            if process_id in visited:
                continue
            
            # Start new community
            community_id = f"community_{process_types[process_id]}_{len(communities)}"
            queue = deque([process_id])
            community_processes = []
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                community_processes.append(current)
                
                # Add connected processes of same type
                for connected in adjacency.get(current, []):
                    if (connected not in visited and 
                        process_types.get(connected) == process_types[process_id]):
                        queue.append(connected)
            
            communities[community_id] = community_processes
        
        return dict(communities)
    
    async def calculate_centrality_measures(
        self, 
        process_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for geological processes"""
        
        centrality_measures = {}
        
        # Try to use Neo4j GDS algorithms
        try:
            centrality_measures = await self._gds_centrality_measures(process_ids)
        except Exception as e:
            logger.warning(f"GDS centrality calculation failed: {e}")
            centrality_measures = await self._manual_centrality_measures(process_ids)
        
        return centrality_measures
    
    async def _gds_centrality_measures(self, process_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures using Neo4j GDS"""
        
        # Create graph projection
        projection_query = """
        CALL gds.graph.project(
            'centrality_network',
            {
                GeologicalProcess: {
                    filter: "n.process_id IN $process_ids"
                }
            },
            {
                INFLUENCES: {
                    properties: ['strength']
                }
            }
        )
        """
        
        # PageRank centrality
        pagerank_query = """
        CALL gds.pageRank.stream('centrality_network', {
            relationshipWeightProperty: 'strength'
        })
        YIELD nodeId, score
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, score as pagerank
        """
        
        # Betweenness centrality
        betweenness_query = """
        CALL gds.betweenness.stream('centrality_network')
        YIELD nodeId, score
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, score as betweenness
        """
        
        # Closeness centrality
        closeness_query = """
        CALL gds.closeness.stream('centrality_network')
        YIELD nodeId, score
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, score as closeness
        """
        
        # Degree centrality
        degree_query = """
        CALL gds.degree.stream('centrality_network')
        YIELD nodeId, score
        MATCH (n:GeologicalProcess) WHERE id(n) = nodeId
        RETURN n.process_id as process_id, score as degree
        """
        
        cleanup_query = "CALL gds.graph.drop('centrality_network')"
        
        try:
            # Create projection
            await self.db_manager.execute_query(projection_query, {"process_ids": process_ids})
            
            # Calculate different centrality measures
            pagerank_results = await self.db_manager.execute_query(pagerank_query)
            betweenness_results = await self.db_manager.execute_query(betweenness_query)
            closeness_results = await self.db_manager.execute_query(closeness_query)
            degree_results = await self.db_manager.execute_query(degree_query)
            
            # Combine results
            centrality_measures = {}
            
            # Initialize with PageRank
            for record in pagerank_results:
                process_id = record["process_id"]
                centrality_measures[process_id] = {"pagerank": record["pagerank"]}
            
            # Add betweenness
            for record in betweenness_results:
                process_id = record["process_id"]
                if process_id in centrality_measures:
                    centrality_measures[process_id]["betweenness"] = record["betweenness"]
            
            # Add closeness
            for record in closeness_results:
                process_id = record["process_id"]
                if process_id in centrality_measures:
                    centrality_measures[process_id]["closeness"] = record["closeness"]
            
            # Add degree
            for record in degree_results:
                process_id = record["process_id"]
                if process_id in centrality_measures:
                    centrality_measures[process_id]["degree"] = record["degree"]
            
            return centrality_measures
            
        finally:
            try:
                await self.db_manager.execute_query(cleanup_query)
            except Exception:
                pass
    
    async def _manual_centrality_measures(self, process_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate centrality measures manually when GDS is not available"""
        
        # Get network structure
        query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        RETURN p.process_id as process_id,
               collect(DISTINCT {
                   connected_id: connected.process_id,
                   strength: r.strength
               }) as connections
        """
        
        results = await self.db_manager.execute_query(query, {"process_ids": process_ids})
        
        # Build adjacency structure
        adjacency = {}
        for record in results:
            process_id = record["process_id"]
            connections = [conn["connected_id"] for conn in record["connections"] if conn["connected_id"]]
            adjacency[process_id] = connections
        
        centrality_measures = {}
        
        # Calculate degree centrality
        for process_id in process_ids:
            degree = len(adjacency.get(process_id, []))
            max_possible_degree = len(process_ids) - 1
            normalized_degree = degree / max_possible_degree if max_possible_degree > 0 else 0.0
            
            centrality_measures[process_id] = {
                "degree": normalized_degree,
                "pagerank": 0.0,  # Placeholder
                "betweenness": 0.0,  # Placeholder
                "closeness": 0.0  # Placeholder
            }
        
        return centrality_measures
    
    async def find_critical_processes(
        self, 
        process_ids: List[str],
        criteria: str = "centrality"
    ) -> List[Dict[str, Any]]:
        """Find critical processes in the geological network"""
        
        if criteria == "centrality":
            centrality_measures = await self.calculate_centrality_measures(process_ids)
            
            # Rank processes by combined centrality score
            critical_processes = []
            for process_id, measures in centrality_measures.items():
                combined_score = (
                    measures.get("pagerank", 0.0) * 0.3 +
                    measures.get("betweenness", 0.0) * 0.3 +
                    measures.get("closeness", 0.0) * 0.2 +
                    measures.get("degree", 0.0) * 0.2
                )
                
                critical_processes.append({
                    "process_id": process_id,
                    "criticality_score": combined_score,
                    "centrality_measures": measures
                })
            
            # Sort by criticality score
            critical_processes.sort(key=lambda x: x["criticality_score"], reverse=True)
            
            return critical_processes[:10]  # Return top 10 critical processes
        
        elif criteria == "connectivity":
            # Find processes with highest connectivity
            query = """
            MATCH (p:GeologicalProcess)
            WHERE p.process_id IN $process_ids
            OPTIONAL MATCH (p)-[:INFLUENCES]-(connected:GeologicalProcess)
            WHERE connected.process_id IN $process_ids
            RETURN p.process_id as process_id,
                   p.name as name,
                   p.process_type as process_type,
                   count(DISTINCT connected) as connection_count
            ORDER BY connection_count DESC
            LIMIT 10
            """
            
            results = await self.db_manager.execute_query(query, {"process_ids": process_ids})
            
            return [
                {
                    "process_id": record["process_id"],
                    "name": record["name"],
                    "process_type": record["process_type"],
                    "connection_count": record["connection_count"],
                    "criticality_score": record["connection_count"] / len(process_ids)
                }
                for record in results
            ]
        
        else:
            raise ValueError(f"Unsupported criticality criteria: {criteria}")
    
    async def analyze_cascade_effects(
        self, 
        trigger_process_id: str,
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """Analyze potential cascade effects from a trigger process"""
        
        query = """
        MATCH path = (trigger:GeologicalProcess {process_id: $trigger_id})-[:INFLUENCES*1..$max_depth]->(affected:GeologicalProcess)
        WHERE ALL(rel in relationships(path) WHERE rel.relationship_type IN ['causal', 'triggers'])
        RETURN path,
               length(path) as cascade_depth,
               [rel in relationships(path) | rel.strength] as strengths,
               [rel in relationships(path) | rel.confidence] as confidences,
               affected.process_type as affected_type,
               affected.magnitude as potential_magnitude
        ORDER BY cascade_depth, reduce(s = 1.0, strength in strengths | s * strength) DESC
        """
        
        parameters = {
            "trigger_id": trigger_process_id,
            "max_depth": max_depth
        }
        
        results = await self.db_manager.execute_query(query, parameters)
        
        # Analyze cascade patterns
        cascade_levels = defaultdict(list)
        affected_types = defaultdict(int)
        total_cascade_strength = 0.0
        
        for record in results:
            depth = record["cascade_depth"]
            strengths = record["strengths"]
            confidences = record["confidences"]
            affected_type = record["affected_type"]
            
            cascade_strength = sum(strengths) / len(strengths) if strengths else 0.0
            cascade_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            cascade_levels[depth].append({
                "path": record["path"],
                "cascade_strength": cascade_strength,
                "cascade_confidence": cascade_confidence,
                "affected_type": affected_type
            })
            
            affected_types[affected_type] += 1
            total_cascade_strength += cascade_strength
        
        return {
            "trigger_process_id": trigger_process_id,
            "cascade_levels": dict(cascade_levels),
            "affected_process_types": dict(affected_types),
            "total_cascade_strength": total_cascade_strength,
            "max_cascade_depth": max(cascade_levels.keys()) if cascade_levels else 0,
            "cascade_breadth": len(results)
        }