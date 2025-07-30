"""
API routes for the Graph Database Service
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse

from core.database import GraphDatabaseManager
from core.graph_algorithms import GeologicalGraphAlgorithms
from core.advanced_analytics import AdvancedGraphAnalytics
from core.causal_discovery import CausalDiscoveryEngine
from models.geological_models import (
    GeologicalProcess, GeologicalRelationship, GraphQuery,
    NetworkAnalysisResult, CausalRelationship, GraphVisualizationConfig
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency to get database manager
async def get_db_manager() -> GraphDatabaseManager:
    """Get database manager instance"""
    from main import db_manager
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not available")
    return db_manager


@router.post("/processes", response_model=Dict[str, str])
async def create_geological_process(
    process: GeologicalProcess,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create a new geological process"""
    try:
        process_id = await db.create_geological_process(process)
        return {"process_id": process_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create geological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes/{process_id}")
async def get_geological_process(
    process_id: str,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get a geological process by ID"""
    try:
        process = await db.get_process_by_id(process_id)
        if not process:
            raise HTTPException(status_code=404, detail="Process not found")
        return process
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get geological process: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processes")
async def search_geological_processes(
    search_term: Optional[str] = Query(None, description="Search term for process names/descriptions"),
    process_type: Optional[str] = Query(None, description="Filter by process type"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of results"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Search geological processes"""
    try:
        if search_term:
            results = await db.search_processes(search_term, process_type, limit)
        else:
            # Get all processes with optional type filter
            query = """
            MATCH (p:GeologicalProcess)
            """ + (f"WHERE p.process_type = '{process_type}'" if process_type else "") + """
            RETURN p
            ORDER BY p.name
            LIMIT $limit
            """
            results = await db.execute_query(query, {"limit": limit})
            results = [{"process": record["p"], "relevance_score": 1.0} for record in results]
        
        return {"processes": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Failed to search geological processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/relationships", response_model=Dict[str, str])
async def create_geological_relationship(
    relationship: GeologicalRelationship,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create a relationship between geological processes"""
    try:
        relationship_id = await db.create_relationship(relationship)
        return {"relationship_id": relationship_id, "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create geological relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-chains/{process_id}")
async def find_causal_chains(
    process_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum chain depth"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find causal chains starting from a specific process"""
    try:
        causal_chains = await db.find_causal_chains(process_id, max_depth)
        return {
            "start_process_id": process_id,
            "causal_chains": [chain.dict() for chain in causal_chains],
            "count": len(causal_chains)
        }
    except Exception as e:
        logger.error(f"Failed to find causal chains: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/network-analysis")
async def analyze_process_network(
    process_ids: List[str],
    include_communities: bool = Query(True, description="Include community detection"),
    include_centrality: bool = Query(True, description="Include centrality measures"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze network properties of geological processes"""
    try:
        # Basic network analysis
        network_result = await db.analyze_process_network(process_ids)
        
        # Enhanced analysis with algorithms
        algorithms = GeologicalGraphAlgorithms(db)
        
        enhanced_result = network_result.dict()
        
        if include_communities:
            try:
                communities = await algorithms.detect_communities(process_ids)
                enhanced_result["communities"] = communities
            except Exception as e:
                logger.warning(f"Community detection failed: {e}")
                enhanced_result["communities"] = {}
        
        if include_centrality:
            try:
                centrality_measures = await algorithms.calculate_centrality_measures(process_ids)
                enhanced_result["enhanced_centrality_measures"] = centrality_measures
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {e}")
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Failed to analyze process network: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/shortest-paths")
async def find_shortest_paths(
    source_id: str = Query(..., description="Source process ID"),
    target_id: str = Query(..., description="Target process ID"),
    relationship_types: Optional[List[str]] = Query(None, description="Filter by relationship types"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find shortest paths between two geological processes"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        paths = await algorithms.find_shortest_paths(source_id, target_id, relationship_types)
        
        return {
            "source_id": source_id,
            "target_id": target_id,
            "paths": paths,
            "count": len(paths)
        }
    except Exception as e:
        logger.error(f"Failed to find shortest paths: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/critical-processes")
async def find_critical_processes(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    criteria: str = Query("centrality", description="Criticality criteria (centrality, connectivity)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Find critical processes in the geological network"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        critical_processes = await algorithms.find_critical_processes(process_ids, criteria)
        
        return {
            "criteria": criteria,
            "critical_processes": critical_processes,
            "count": len(critical_processes)
        }
    except Exception as e:
        logger.error(f"Failed to find critical processes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cascade-analysis/{trigger_process_id}")
async def analyze_cascade_effects(
    trigger_process_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum cascade depth"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze potential cascade effects from a trigger process"""
    try:
        algorithms = GeologicalGraphAlgorithms(db)
        cascade_analysis = await algorithms.analyze_cascade_effects(trigger_process_id, max_depth)
        
        return cascade_analysis
    except Exception as e:
        logger.error(f"Failed to analyze cascade effects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualization/config")
async def create_visualization_config(
    config: GraphVisualizationConfig,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Create visualization configuration for graph data"""
    try:
        # Store visualization config in database
        query = """
        CREATE (v:VisualizationConfig {
            visualization_id: $visualization_id,
            layout_algorithm: $layout_algorithm,
            node_size_property: $node_size_property,
            node_color_property: $node_color_property,
            edge_width_property: $edge_width_property,
            show_node_labels: $show_node_labels,
            show_edge_labels: $show_edge_labels,
            min_edge_strength: $min_edge_strength,
            export_format: $export_format,
            include_metadata: $include_metadata,
            created_at: datetime()
        })
        RETURN v.visualization_id as visualization_id
        """
        
        parameters = config.dict()
        result = await db.execute_write_transaction(query, parameters)
        
        return {"visualization_id": result["visualization_id"], "status": "created"}
    except Exception as e:
        logger.error(f"Failed to create visualization config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization/{visualization_id}/data")
async def get_visualization_data(
    visualization_id: str,
    process_ids: List[str] = Query(..., description="Process IDs to visualize"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get graph data formatted for visualization"""
    try:
        # Get visualization config
        config_query = """
        MATCH (v:VisualizationConfig {visualization_id: $visualization_id})
        RETURN v
        """
        config_results = await db.execute_query(config_query, {"visualization_id": visualization_id})
        
        if not config_results:
            raise HTTPException(status_code=404, detail="Visualization config not found")
        
        config = config_results[0]["v"]
        
        # Get network data
        network_query = """
        MATCH (p:GeologicalProcess)
        WHERE p.process_id IN $process_ids
        OPTIONAL MATCH (p)-[r:INFLUENCES]-(connected:GeologicalProcess)
        WHERE connected.process_id IN $process_ids
        AND r.strength >= $min_edge_strength
        RETURN p, collect(DISTINCT {
            relationship: r,
            connected_process: connected
        }) as connections
        """
        
        parameters = {
            "process_ids": process_ids,
            "min_edge_strength": config.get("min_edge_strength", 0.1)
        }
        
        network_results = await db.execute_query(network_query, parameters)
        
        # Format data for visualization
        nodes = []
        edges = []
        edge_set = set()  # To avoid duplicate edges
        
        for record in network_results:
            process = record["p"]
            connections = record["connections"]
            
            # Add node
            node_data = {
                "id": process["process_id"],
                "label": process["name"] if config.get("show_node_labels", True) else "",
                "type": process["process_type"],
                "size": process.get(config.get("node_size_property", "magnitude"), 10),
                "color": process["process_type"],  # Color by type by default
                "metadata": {
                    "description": process.get("description", ""),
                    "location_id": process.get("location_id", ""),
                    "depth_range": process.get("depth_range", []),
                    "temperature_range": process.get("temperature_range", [])
                } if config.get("include_metadata", True) else {}
            }
            nodes.append(node_data)
            
            # Add edges
            for conn in connections:
                if conn["relationship"] and conn["connected_process"]:
                    rel = conn["relationship"]
                    target = conn["connected_process"]
                    
                    edge_key = tuple(sorted([process["process_id"], target["process_id"]]))
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        
                        edge_data = {
                            "source": process["process_id"],
                            "target": target["process_id"],
                            "label": rel["relationship_type"] if config.get("show_edge_labels", False) else "",
                            "weight": rel["strength"],
                            "width": rel.get(config.get("edge_width_property", "strength"), 1),
                            "type": rel["relationship_type"],
                            "metadata": {
                                "confidence": rel.get("confidence", 1.0),
                                "mechanism": rel.get("mechanism", ""),
                                "time_delay": rel.get("time_delay", None)
                            } if config.get("include_metadata", True) else {}
                        }
                        edges.append(edge_data)
        
        visualization_data = {
            "visualization_id": visualization_id,
            "config": config,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "generated_at": datetime.now().isoformat()
        }
        
        return visualization_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get visualization data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def execute_graph_query(
    query: GraphQuery,
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Execute a complex graph query"""
    try:
        if query.query_type == "causal_chain":
            if not query.start_processes:
                raise HTTPException(status_code=400, detail="Start processes required for causal chain query")
            
            results = []
            for start_process in query.start_processes:
                chains = await db.find_causal_chains(start_process, query.max_depth or 5)
                results.extend([chain.dict() for chain in chains])
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": results,
                "count": len(results)
            }
        
        elif query.query_type == "network_analysis":
            if not query.start_processes:
                raise HTTPException(status_code=400, detail="Process IDs required for network analysis")
            
            network_result = await db.analyze_process_network(query.start_processes)
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": network_result.dict(),
                "count": 1
            }
        
        elif query.query_type == "pattern_search":
            # Custom pattern search implementation
            cypher_query = await _build_pattern_search_query(query)
            results = await db.execute_query(cypher_query, query.dict())
            
            return {
                "query_id": query.query_id,
                "query_type": query.query_type,
                "results": results,
                "count": len(results)
            }
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported query type: {query.query_type}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute graph query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _build_pattern_search_query(query: GraphQuery) -> str:
    """Build Cypher query for pattern search"""
    
    # Base query structure
    cypher_parts = ["MATCH (p:GeologicalProcess)"]
    
    # Add filters
    where_conditions = []
    
    if query.process_types:
        process_types = "', '".join(query.process_types)
        where_conditions.append(f"p.process_type IN ['{process_types}']")
    
    if query.spatial_bounds:
        # Assuming spatial bounds are provided as lat/lon bounds
        bounds = query.spatial_bounds
        if all(key in bounds for key in ["min_lat", "max_lat", "min_lon", "max_lon"]):
            where_conditions.append(
                f"p.latitude >= {bounds['min_lat']} AND p.latitude <= {bounds['max_lat']} "
                f"AND p.longitude >= {bounds['min_lon']} AND p.longitude <= {bounds['max_lon']}"
            )
    
    # Add relationship patterns if specified
    if query.relationship_types:
        rel_types = "|".join(query.relationship_types)
        cypher_parts.append(f"OPTIONAL MATCH (p)-[r:{rel_types}]-(connected:GeologicalProcess)")
        
        if query.min_strength:
            where_conditions.append(f"r.strength >= {query.min_strength}")
        
        if query.min_confidence:
            where_conditions.append(f"r.confidence >= {query.min_confidence}")
    
    # Add WHERE clause if conditions exist
    if where_conditions:
        cypher_parts.append("WHERE " + " AND ".join(where_conditions))
    
    # Add RETURN clause
    if query.relationship_types:
        cypher_parts.append("RETURN p, collect(DISTINCT {relationship: r, connected: connected}) as relationships")
    else:
        cypher_parts.append("RETURN p")
    
    # Add LIMIT
    cypher_parts.append(f"LIMIT {query.max_results}")
    
    return " ".join(cypher_parts)


@router.get("/statistics")
async def get_graph_statistics(
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Get overall graph database statistics"""
    try:
        stats_query = """
        MATCH (p:GeologicalProcess)
        OPTIONAL MATCH (p)-[r:INFLUENCES]-()
        RETURN 
            count(DISTINCT p) as total_processes,
            count(DISTINCT r) as total_relationships,
            collect(DISTINCT p.process_type) as process_types,
            avg(r.strength) as avg_relationship_strength,
            avg(r.confidence) as avg_relationship_confidence
        """
        
        results = await db.execute_query(stats_query)
        
        if results:
            stats = results[0]
            return {
                "total_processes": stats["total_processes"],
                "total_relationships": stats["total_relationships"],
                "process_types": stats["process_types"],
                "average_relationship_strength": stats["avg_relationship_strength"],
                "average_relationship_confidence": stats["avg_relationship_confidence"],
                "generated_at": datetime.now().isoformat()
            }
        else:
            return {
                "total_processes": 0,
                "total_relationships": 0,
                "process_types": [],
                "average_relationship_strength": 0.0,
                "average_relationship_confidence": 0.0,
                "generated_at": datetime.now().isoformat()
            }
    
    except Exception as e:
        logger.error(f"Failed to get graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/temporal-evolution")
async def analyze_temporal_evolution(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    time_window_days: int = Query(365, description="Time window in days"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze temporal evolution of the geological network"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        temporal_analysis = await analytics.analyze_temporal_evolution(process_ids, time_window_days)
        return temporal_analysis
    except Exception as e:
        logger.error(f"Failed to analyze temporal evolution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/anomaly-detection")
async def detect_anomalous_patterns(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    threshold_std_dev: float = Query(2.0, description="Standard deviation threshold for anomaly detection"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Detect anomalous patterns in the geological network"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        anomalies = await analytics.detect_anomalous_patterns(process_ids, threshold_std_dev)
        return anomalies
    except Exception as e:
        logger.error(f"Failed to detect anomalous patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/cascade-prediction/{trigger_process_id}")
async def predict_cascade_probability(
    trigger_process_id: str,
    target_process_ids: List[str] = Query(..., description="Target process IDs"),
    time_horizon_days: int = Query(30, description="Time horizon for prediction in days"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Predict probability of cascade effects reaching target processes"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        predictions = await analytics.predict_cascade_probability(
            trigger_process_id, target_process_ids, time_horizon_days
        )
        return predictions
    except Exception as e:
        logger.error(f"Failed to predict cascade probability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/resilience-analysis")
async def analyze_network_resilience(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    failure_scenarios: Optional[List[str]] = Query(None, description="Specific failure scenarios to test"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Analyze network resilience to process failures"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        resilience = await analytics.analyze_network_resilience(process_ids, failure_scenarios)
        return resilience
    except Exception as e:
        logger.error(f"Failed to analyze network resilience: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/advanced-analytics/process-clustering")
async def identify_process_clusters(
    process_ids: List[str] = Query(..., description="Process IDs to cluster"),
    clustering_method: str = Query("spatial_temporal", description="Clustering method (spatial_temporal, process_type, strength_based)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Identify clusters of related geological processes"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        clusters = await analytics.identify_process_clusters(process_ids, clustering_method)
        return clusters
    except Exception as e:
        logger.error(f"Failed to identify process clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advanced-analytics/comprehensive-report")
async def generate_comprehensive_report(
    process_ids: List[str],
    include_predictions: bool = Query(True, description="Include cascade predictions"),
    include_anomalies: bool = Query(True, description="Include anomaly detection"),
    include_resilience: bool = Query(True, description="Include resilience analysis"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Generate comprehensive network analysis report"""
    try:
        analytics = AdvancedGraphAnalytics(db)
        report = await analytics.generate_network_report(
            process_ids, include_predictions, include_anomalies, include_resilience
        )
        return report
    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal-discovery/hidden-relationships")
async def discover_hidden_causality(
    process_ids: List[str] = Query(..., description="Process IDs to analyze"),
    confidence_threshold: float = Query(0.7, description="Confidence threshold for discovery"),
    strength_threshold: float = Query(0.5, description="Strength threshold for discovery"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Discover hidden causal relationships using advanced graph analysis"""
    try:
        causal_engine = CausalDiscoveryEngine(db)
        hidden_relationships = await causal_engine.discover_hidden_causality(
            process_ids, confidence_threshold, strength_threshold
        )
        return hidden_relationships
    except Exception as e:
        logger.error(f"Failed to discover hidden causality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/causal-discovery/validate-relationships")
async def validate_causal_relationships(
    proposed_relationships: List[Dict[str, Any]],
    validation_method: str = Query("statistical", description="Validation method (statistical, domain_knowledge)"),
    db: GraphDatabaseManager = Depends(get_db_manager)
):
    """Validate proposed causal relationships"""
    try:
        causal_engine = CausalDiscoveryEngine(db)
        validation_results = await causal_engine.validate_causal_relationships(
            proposed_relationships, validation_method
        )
        return validation_results
    except Exception as e:
        logger.error(f"Failed to validate causal relationships: {e}")
        raise HTTPException(status_code=500, detail=str(e))