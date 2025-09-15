"""
Data Lineage Tracking and Impact Analysis System

This module provides comprehensive data lineage tracking across all geological
databases with advanced impact analysis, dependency mapping, and data governance
capabilities for the Prometheus AI Platform.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque

import networkx as nx
import pandas as pd
import numpy as np
from neo4j import AsyncGraphDatabase
import asyncpg
import redis.asyncio as redis
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class LineageNodeType(Enum):
    """Types of nodes in data lineage graph"""
    DATA_SOURCE = "data_source"
    DATABASE_TABLE = "database_table"
    TRANSFORMATION = "transformation"
    ANALYSIS_RESULT = "analysis_result"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    VISUALIZATION = "visualization"
    REPORT = "report"
    API_ENDPOINT = "api_endpoint"


class LineageEdgeType(Enum):
    """Types of edges in data lineage graph"""
    DERIVES_FROM = "derives_from"
    TRANSFORMS_TO = "transforms_to"
    AGGREGATES_FROM = "aggregates_from"
    JOINS_WITH = "joins_with"
    FILTERS_FROM = "filters_from"
    ENRICHES = "enriches"
    VALIDATES = "validates"
    TRIGGERS = "triggers"


class ImpactLevel(Enum):
    """Levels of impact for change propagation"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph"""
    node_id: str
    node_type: LineageNodeType
    name: str
    description: str
    database_name: Optional[str]
    table_name: Optional[str]
    schema_info: Dict[str, Any]
    metadata: Dict[str, Any]
    created_timestamp: datetime
    last_modified: datetime
    owner: str
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['node_type'] = self.node_type.value
        data['created_timestamp'] = self.created_timestamp.isoformat()
        data['last_modified'] = self.last_modified.isoformat()
        return data


@dataclass
class LineageEdge:
    """Represents an edge in the data lineage graph"""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: LineageEdgeType
    transformation_logic: Optional[str]
    field_mappings: Dict[str, str]
    metadata: Dict[str, Any]
    created_timestamp: datetime
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['edge_type'] = self.edge_type.value
        data['created_timestamp'] = self.created_timestamp.isoformat()
        return data


@dataclass
class ImpactAnalysisResult:
    """Result of impact analysis for a change"""
    analysis_id: str
    source_node_id: str
    affected_nodes: List[str]
    impact_paths: List[List[str]]
    impact_levels: Dict[str, ImpactLevel]
    estimated_propagation_time: Dict[str, timedelta]
    recommendations: List[str]
    analysis_timestamp: datetime


@dataclass
class DataQualityMetrics:
    """Data quality metrics for lineage nodes"""
    node_id: str
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    timeliness_score: float
    validity_score: float
    overall_quality_score: float
    measurement_timestamp: datetime
    quality_issues: List[str]


class DataLineageTracker:
    """Advanced data lineage tracking system"""
    
    def __init__(self, 
                 neo4j_uri: str,
                 neo4j_username: str,
                 neo4j_password: str,
                 postgres_connection_string: str,
                 redis_url: str):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.postgres_connection_string = postgres_connection_string
        self.redis_url = redis_url
        
        self.neo4j_driver = None
        self.postgres_conn = None
        self.redis_client = None
        
        # In-memory lineage graph for fast operations
        self.lineage_graph = nx.DiGraph()
        
        # Cache for frequently accessed lineage information
        self.lineage_cache = {}
        
    async def initialize(self) -> None:
        """Initialize database connections and create schema"""
        try:
            # Initialize Neo4j connection
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            # Initialize PostgreSQL connection
            self.postgres_conn = await asyncpg.connect(self.postgres_connection_string)
            
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            
            # Create database schema
            await self._create_lineage_schema()
            
            # Load existing lineage into memory
            await self._load_lineage_graph()
            
            logger.info("Data lineage tracker initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize data lineage tracker", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close database connections"""
        try:
            if self.neo4j_driver:
                await self.neo4j_driver.close()
            
            if self.postgres_conn:
                await self.postgres_conn.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Data lineage tracker connections closed")
            
        except Exception as e:
            logger.error("Failed to close connections", error=str(e))
    
    async def _create_lineage_schema(self) -> None:
        """Create database schema for lineage tracking"""
        try:
            # Create PostgreSQL tables for lineage metadata
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_nodes (
                    node_id UUID PRIMARY KEY,
                    node_type VARCHAR(50) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    database_name VARCHAR(100),
                    table_name VARCHAR(100),
                    schema_info JSONB,
                    metadata JSONB,
                    created_timestamp TIMESTAMP DEFAULT NOW(),
                    last_modified TIMESTAMP DEFAULT NOW(),
                    owner VARCHAR(100),
                    tags TEXT[]
                )
            """)
            
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS lineage_edges (
                    edge_id UUID PRIMARY KEY,
                    source_node_id UUID REFERENCES lineage_nodes(node_id),
                    target_node_id UUID REFERENCES lineage_nodes(node_id),
                    edge_type VARCHAR(50) NOT NULL,
                    transformation_logic TEXT,
                    field_mappings JSONB,
                    metadata JSONB,
                    created_timestamp TIMESTAMP DEFAULT NOW(),
                    confidence_score FLOAT DEFAULT 1.0
                )
            """)
            
            await self.postgres_conn.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    node_id UUID REFERENCES lineage_nodes(node_id),
                    completeness_score FLOAT,
                    accuracy_score FLOAT,
                    consistency_score FLOAT,
                    timeliness_score FLOAT,
                    validity_score FLOAT,
                    overall_quality_score FLOAT,
                    measurement_timestamp TIMESTAMP DEFAULT NOW(),
                    quality_issues TEXT[],
                    PRIMARY KEY (node_id, measurement_timestamp)
                )
            """)
            
            # Create indexes for performance
            await self.postgres_conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_lineage_nodes_type ON lineage_nodes(node_type);
                CREATE INDEX IF NOT EXISTS idx_lineage_nodes_database ON lineage_nodes(database_name, table_name);
                CREATE INDEX IF NOT EXISTS idx_lineage_edges_source ON lineage_edges(source_node_id);
                CREATE INDEX IF NOT EXISTS idx_lineage_edges_target ON lineage_edges(target_node_id);
                CREATE INDEX IF NOT EXISTS idx_quality_metrics_timestamp ON data_quality_metrics(measurement_timestamp);
            """)
            
            # Create Neo4j constraints and indexes
            async with self.neo4j_driver.session() as session:
                # Create constraints
                await session.run("""
                    CREATE CONSTRAINT lineage_node_id IF NOT EXISTS
                    FOR (n:LineageNode) REQUIRE n.node_id IS UNIQUE
                """)
                
                await session.run("""
                    CREATE CONSTRAINT lineage_edge_id IF NOT EXISTS
                    FOR ()-[r:LINEAGE_EDGE]-() REQUIRE r.edge_id IS UNIQUE
                """)
                
                # Create indexes
                await session.run("""
                    CREATE INDEX lineage_node_type IF NOT EXISTS
                    FOR (n:LineageNode) ON (n.node_type)
                """)
                
                await session.run("""
                    CREATE INDEX lineage_database_table IF NOT EXISTS
                    FOR (n:LineageNode) ON (n.database_name, n.table_name)
                """)
            
            logger.info("Lineage database schema created successfully")
            
        except Exception as e:
            logger.error("Failed to create lineage schema", error=str(e))
            raise
    
    async def _load_lineage_graph(self) -> None:
        """Load existing lineage graph into memory"""
        try:
            # Load nodes
            nodes_query = "SELECT * FROM lineage_nodes"
            node_records = await self.postgres_conn.fetch(nodes_query)
            
            for record in node_records:
                node = LineageNode(
                    node_id=str(record['node_id']),
                    node_type=LineageNodeType(record['node_type']),
                    name=record['name'],
                    description=record['description'] or '',
                    database_name=record['database_name'],
                    table_name=record['table_name'],
                    schema_info=record['schema_info'] or {},
                    metadata=record['metadata'] or {},
                    created_timestamp=record['created_timestamp'],
                    last_modified=record['last_modified'],
                    owner=record['owner'] or '',
                    tags=record['tags'] or []
                )
                
                self.lineage_graph.add_node(node.node_id, **node.to_dict())
            
            # Load edges
            edges_query = "SELECT * FROM lineage_edges"
            edge_records = await self.postgres_conn.fetch(edges_query)
            
            for record in edge_records:
                edge = LineageEdge(
                    edge_id=str(record['edge_id']),
                    source_node_id=str(record['source_node_id']),
                    target_node_id=str(record['target_node_id']),
                    edge_type=LineageEdgeType(record['edge_type']),
                    transformation_logic=record['transformation_logic'],
                    field_mappings=record['field_mappings'] or {},
                    metadata=record['metadata'] or {},
                    created_timestamp=record['created_timestamp'],
                    confidence_score=record['confidence_score']
                )
                
                self.lineage_graph.add_edge(
                    edge.source_node_id,
                    edge.target_node_id,
                    **edge.to_dict()
                )
            
            logger.info("Loaded lineage graph into memory", 
                       nodes=len(self.lineage_graph.nodes),
                       edges=len(self.lineage_graph.edges))
            
        except Exception as e:
            logger.error("Failed to load lineage graph", error=str(e))
            raise
    
    async def register_data_source(self, 
                                 source_name: str,
                                 source_type: str,
                                 database_name: str,
                                 table_name: str,
                                 schema_info: Dict[str, Any],
                                 owner: str,
                                 tags: List[str] = None) -> str:
        """Register a new data source in the lineage system"""
        try:
            node_id = str(uuid.uuid4())
            
            node = LineageNode(
                node_id=node_id,
                node_type=LineageNodeType.DATA_SOURCE,
                name=source_name,
                description=f"{source_type} data source: {source_name}",
                database_name=database_name,
                table_name=table_name,
                schema_info=schema_info,
                metadata={
                    'source_type': source_type,
                    'registration_timestamp': datetime.utcnow().isoformat()
                },
                created_timestamp=datetime.utcnow(),
                last_modified=datetime.utcnow(),
                owner=owner,
                tags=tags or []
            )
            
            # Store in PostgreSQL
            await self.postgres_conn.execute("""
                INSERT INTO lineage_nodes 
                (node_id, node_type, name, description, database_name, table_name, 
                 schema_info, metadata, owner, tags)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, node_id, node.node_type.value, node.name, node.description,
                node.database_name, node.table_name, json.dumps(node.schema_info),
                json.dumps(node.metadata), node.owner, node.tags)
            
            # Store in Neo4j
            await self._store_node_in_neo4j(node)
            
            # Add to in-memory graph
            self.lineage_graph.add_node(node_id, **node.to_dict())
            
            logger.info("Registered data source", 
                       node_id=node_id,
                       source_name=source_name)
            
            return node_id
            
        except Exception as e:
            logger.error("Failed to register data source", 
                        source_name=source_name,
                        error=str(e))
            raise
    
    async def register_transformation(self,
                                    transformation_name: str,
                                    description: str,
                                    input_nodes: List[str],
                                    output_node: str,
                                    transformation_logic: str,
                                    field_mappings: Dict[str, str],
                                    owner: str) -> str:
        """Register a data transformation in the lineage system"""
        try:
            transformation_id = str(uuid.uuid4())
            
            # Create transformation node
            transformation_node = LineageNode(
                node_id=transformation_id,
                node_type=LineageNodeType.TRANSFORMATION,
                name=transformation_name,
                description=description,
                database_name=None,
                table_name=None,
                schema_info={},
                metadata={
                    'transformation_logic': transformation_logic,
                    'input_count': len(input_nodes),
                    'registration_timestamp': datetime.utcnow().isoformat()
                },
                created_timestamp=datetime.utcnow(),
                last_modified=datetime.utcnow(),
                owner=owner,
                tags=['transformation']
            )
            
            # Store transformation node
            await self._store_node(transformation_node)
            
            # Create edges from input nodes to transformation
            for input_node_id in input_nodes:
                edge_id = str(uuid.uuid4())
                
                edge = LineageEdge(
                    edge_id=edge_id,
                    source_node_id=input_node_id,
                    target_node_id=transformation_id,
                    edge_type=LineageEdgeType.TRANSFORMS_TO,
                    transformation_logic=transformation_logic,
                    field_mappings=field_mappings,
                    metadata={},
                    created_timestamp=datetime.utcnow(),
                    confidence_score=1.0
                )
                
                await self._store_edge(edge)
            
            # Create edge from transformation to output node
            output_edge_id = str(uuid.uuid4())
            
            output_edge = LineageEdge(
                edge_id=output_edge_id,
                source_node_id=transformation_id,
                target_node_id=output_node,
                edge_type=LineageEdgeType.DERIVES_FROM,
                transformation_logic=transformation_logic,
                field_mappings=field_mappings,
                metadata={},
                created_timestamp=datetime.utcnow(),
                confidence_score=1.0
            )
            
            await self._store_edge(output_edge)
            
            logger.info("Registered transformation", 
                       transformation_id=transformation_id,
                       transformation_name=transformation_name,
                       input_nodes=len(input_nodes))
            
            return transformation_id
            
        except Exception as e:
            logger.error("Failed to register transformation", 
                        transformation_name=transformation_name,
                        error=str(e))
            raise
    
    async def _store_node(self, node: LineageNode) -> None:
        """Store lineage node in all storage systems"""
        # Store in PostgreSQL
        await self.postgres_conn.execute("""
            INSERT INTO lineage_nodes 
            (node_id, node_type, name, description, database_name, table_name, 
             schema_info, metadata, owner, tags)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (node_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                last_modified = NOW()
        """, node.node_id, node.node_type.value, node.name, node.description,
            node.database_name, node.table_name, json.dumps(node.schema_info),
            json.dumps(node.metadata), node.owner, node.tags)
        
        # Store in Neo4j
        await self._store_node_in_neo4j(node)
        
        # Add to in-memory graph
        self.lineage_graph.add_node(node.node_id, **node.to_dict())
    
    async def _store_edge(self, edge: LineageEdge) -> None:
        """Store lineage edge in all storage systems"""
        # Store in PostgreSQL
        await self.postgres_conn.execute("""
            INSERT INTO lineage_edges 
            (edge_id, source_node_id, target_node_id, edge_type, 
             transformation_logic, field_mappings, metadata, confidence_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (edge_id) DO UPDATE SET
                transformation_logic = EXCLUDED.transformation_logic,
                field_mappings = EXCLUDED.field_mappings,
                confidence_score = EXCLUDED.confidence_score
        """, edge.edge_id, edge.source_node_id, edge.target_node_id,
            edge.edge_type.value, edge.transformation_logic,
            json.dumps(edge.field_mappings), json.dumps(edge.metadata),
            edge.confidence_score)
        
        # Store in Neo4j
        await self._store_edge_in_neo4j(edge)
        
        # Add to in-memory graph
        self.lineage_graph.add_edge(
            edge.source_node_id,
            edge.target_node_id,
            **edge.to_dict()
        )
    
    async def _store_node_in_neo4j(self, node: LineageNode) -> None:
        """Store lineage node in Neo4j"""
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MERGE (n:LineageNode {node_id: $node_id})
                SET n.node_type = $node_type,
                    n.name = $name,
                    n.description = $description,
                    n.database_name = $database_name,
                    n.table_name = $table_name,
                    n.owner = $owner,
                    n.tags = $tags,
                    n.created_timestamp = $created_timestamp,
                    n.last_modified = $last_modified
            """, 
                node_id=node.node_id,
                node_type=node.node_type.value,
                name=node.name,
                description=node.description,
                database_name=node.database_name,
                table_name=node.table_name,
                owner=node.owner,
                tags=node.tags,
                created_timestamp=node.created_timestamp.isoformat(),
                last_modified=node.last_modified.isoformat()
            )
    
    async def _store_edge_in_neo4j(self, edge: LineageEdge) -> None:
        """Store lineage edge in Neo4j"""
        async with self.neo4j_driver.session() as session:
            await session.run("""
                MATCH (source:LineageNode {node_id: $source_node_id})
                MATCH (target:LineageNode {node_id: $target_node_id})
                MERGE (source)-[r:LINEAGE_EDGE {edge_id: $edge_id}]->(target)
                SET r.edge_type = $edge_type,
                    r.transformation_logic = $transformation_logic,
                    r.confidence_score = $confidence_score,
                    r.created_timestamp = $created_timestamp
            """,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                edge_id=edge.edge_id,
                edge_type=edge.edge_type.value,
                transformation_logic=edge.transformation_logic,
                confidence_score=edge.confidence_score,
                created_timestamp=edge.created_timestamp.isoformat()
            )
    
    async def trace_lineage_upstream(self, node_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Trace data lineage upstream from a given node"""
        try:
            if node_id not in self.lineage_graph:
                raise ValueError(f"Node {node_id} not found in lineage graph")
            
            # Use BFS to trace upstream
            visited = set()
            queue = deque([(node_id, 0)])
            lineage_tree = {
                'root_node': node_id,
                'upstream_nodes': {},
                'paths': [],
                'depth': 0
            }
            
            while queue and len(visited) < 1000:  # Prevent infinite loops
                current_node, depth = queue.popleft()
                
                if current_node in visited or depth > max_depth:
                    continue
                
                visited.add(current_node)
                lineage_tree['depth'] = max(lineage_tree['depth'], depth)
                
                # Get node information
                node_data = self.lineage_graph.nodes[current_node]
                lineage_tree['upstream_nodes'][current_node] = {
                    'node_data': node_data,
                    'depth': depth,
                    'predecessors': []
                }
                
                # Get predecessors (upstream nodes)
                predecessors = list(self.lineage_graph.predecessors(current_node))
                lineage_tree['upstream_nodes'][current_node]['predecessors'] = predecessors
                
                # Add predecessors to queue
                for pred in predecessors:
                    if pred not in visited:
                        queue.append((pred, depth + 1))
                        
                        # Record path
                        lineage_tree['paths'].append({
                            'from': pred,
                            'to': current_node,
                            'depth': depth,
                            'edge_data': self.lineage_graph.edges[pred, current_node]
                        })
            
            # Cache result
            cache_key = f"upstream_lineage:{node_id}:{max_depth}"
            await self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps(lineage_tree, default=str)
            )
            
            logger.info("Traced upstream lineage", 
                       node_id=node_id,
                       upstream_nodes=len(lineage_tree['upstream_nodes']),
                       max_depth=lineage_tree['depth'])
            
            return lineage_tree
            
        except Exception as e:
            logger.error("Failed to trace upstream lineage", 
                        node_id=node_id,
                        error=str(e))
            raise
    
    async def trace_lineage_downstream(self, node_id: str, max_depth: int = 10) -> Dict[str, Any]:
        """Trace data lineage downstream from a given node"""
        try:
            if node_id not in self.lineage_graph:
                raise ValueError(f"Node {node_id} not found in lineage graph")
            
            # Use BFS to trace downstream
            visited = set()
            queue = deque([(node_id, 0)])
            lineage_tree = {
                'root_node': node_id,
                'downstream_nodes': {},
                'paths': [],
                'depth': 0
            }
            
            while queue and len(visited) < 1000:  # Prevent infinite loops
                current_node, depth = queue.popleft()
                
                if current_node in visited or depth > max_depth:
                    continue
                
                visited.add(current_node)
                lineage_tree['depth'] = max(lineage_tree['depth'], depth)
                
                # Get node information
                node_data = self.lineage_graph.nodes[current_node]
                lineage_tree['downstream_nodes'][current_node] = {
                    'node_data': node_data,
                    'depth': depth,
                    'successors': []
                }
                
                # Get successors (downstream nodes)
                successors = list(self.lineage_graph.successors(current_node))
                lineage_tree['downstream_nodes'][current_node]['successors'] = successors
                
                # Add successors to queue
                for succ in successors:
                    if succ not in visited:
                        queue.append((succ, depth + 1))
                        
                        # Record path
                        lineage_tree['paths'].append({
                            'from': current_node,
                            'to': succ,
                            'depth': depth,
                            'edge_data': self.lineage_graph.edges[current_node, succ]
                        })
            
            # Cache result
            cache_key = f"downstream_lineage:{node_id}:{max_depth}"
            await self.redis_client.setex(
                cache_key, 
                3600,  # 1 hour TTL
                json.dumps(lineage_tree, default=str)
            )
            
            logger.info("Traced downstream lineage", 
                       node_id=node_id,
                       downstream_nodes=len(lineage_tree['downstream_nodes']),
                       max_depth=lineage_tree['depth'])
            
            return lineage_tree
            
        except Exception as e:
            logger.error("Failed to trace downstream lineage", 
                        node_id=node_id,
                        error=str(e))
            raise


# Continue with impact analysis in next part...
    async def perform_impact_analysis(self, 
                                     node_id: str, 
                                     change_type: str = 'schema_change',
                                     severity: str = 'medium') -> ImpactAnalysisResult:
        """Perform comprehensive impact analysis for a node change"""
        try:
            analysis_id = str(uuid.uuid4())
            
            # Get downstream lineage
            downstream_lineage = await self.trace_lineage_downstream(node_id, max_depth=15)
            
            affected_nodes = list(downstream_lineage['downstream_nodes'].keys())
            impact_paths = []
            impact_levels = {}
            propagation_times = {}
            recommendations = []
            
            # Analyze impact for each affected node
            for affected_node_id in affected_nodes:
                if affected_node_id == node_id:
                    continue
                
                # Calculate impact level
                impact_level = await self._calculate_impact_level(
                    node_id, affected_node_id, change_type, severity
                )
                impact_levels[affected_node_id] = impact_level
                
                # Estimate propagation time
                propagation_time = await self._estimate_propagation_time(
                    node_id, affected_node_id
                )
                propagation_times[affected_node_id] = propagation_time
                
                # Find impact paths
                paths = await self._find_impact_paths(node_id, affected_node_id)
                impact_paths.extend(paths)
            
            # Generate recommendations
            recommendations = await self._generate_impact_recommendations(
                node_id, affected_nodes, impact_levels, change_type
            )
            
            # Create impact analysis result
            result = ImpactAnalysisResult(
                analysis_id=analysis_id,
                source_node_id=node_id,
                affected_nodes=affected_nodes,
                impact_paths=impact_paths,
                impact_levels=impact_levels,
                estimated_propagation_time=propagation_times,
                recommendations=recommendations,
                analysis_timestamp=datetime.utcnow()
            )
            
            # Store impact analysis result
            await self._store_impact_analysis(result)
            
            logger.info("Completed impact analysis", 
                       analysis_id=analysis_id,
                       source_node=node_id,
                       affected_nodes=len(affected_nodes),
                       high_impact_nodes=len([n for n, l in impact_levels.items() 
                                            if l in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]]))
            
            return result
            
        except Exception as e:
            logger.error("Impact analysis failed", 
                        node_id=node_id,
                        error=str(e))
            raise
    
    async def _calculate_impact_level(self, 
                                    source_node_id: str,
                                    target_node_id: str,
                                    change_type: str,
                                    severity: str) -> ImpactLevel:
        """Calculate impact level for a target node"""
        try:
            # Get node information
            source_node = self.lineage_graph.nodes[source_node_id]
            target_node = self.lineage_graph.nodes[target_node_id]
            
            # Base impact score
            impact_score = 0.0
            
            # Factor 1: Node type criticality
            node_type_weights = {
                LineageNodeType.DATA_SOURCE: 0.9,
                LineageNodeType.DATABASE_TABLE: 0.8,
                LineageNodeType.TRANSFORMATION: 0.6,
                LineageNodeType.ANALYSIS_RESULT: 0.4,
                LineageNodeType.CAUSAL_RELATIONSHIP: 0.7,
                LineageNodeType.VISUALIZATION: 0.3,
                LineageNodeType.REPORT: 0.2,
                LineageNodeType.API_ENDPOINT: 0.5
            }
            
            target_type = LineageNodeType(target_node['node_type'])
            impact_score += node_type_weights.get(target_type, 0.5)
            
            # Factor 2: Distance in lineage graph
            try:
                path_length = nx.shortest_path_length(
                    self.lineage_graph, source_node_id, target_node_id
                )
                # Closer nodes have higher impact
                distance_factor = max(0.1, 1.0 - (path_length - 1) * 0.2)
                impact_score += distance_factor
            except nx.NetworkXNoPath:
                impact_score += 0.1  # Minimal impact if no direct path
            
            # Factor 3: Change type severity
            change_type_weights = {
                'schema_change': 0.8,
                'data_format_change': 0.7,
                'business_logic_change': 0.6,
                'performance_change': 0.4,
                'metadata_change': 0.2
            }
            
            impact_score += change_type_weights.get(change_type, 0.5)
            
            # Factor 4: Severity multiplier
            severity_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5,
                'critical': 2.0
            }
            
            impact_score *= severity_multipliers.get(severity, 1.0)
            
            # Factor 5: Node usage frequency (from metadata)
            usage_frequency = target_node.get('metadata', {}).get('usage_frequency', 1.0)
            impact_score += min(0.5, usage_frequency / 100.0)
            
            # Convert score to impact level
            if impact_score >= 2.5:
                return ImpactLevel.CRITICAL
            elif impact_score >= 2.0:
                return ImpactLevel.HIGH
            elif impact_score >= 1.0:
                return ImpactLevel.MEDIUM
            else:
                return ImpactLevel.LOW
                
        except Exception as e:
            logger.error("Failed to calculate impact level", error=str(e))
            return ImpactLevel.MEDIUM
    
    async def _estimate_propagation_time(self, 
                                       source_node_id: str,
                                       target_node_id: str) -> timedelta:
        """Estimate time for change to propagate to target node"""
        try:
            # Get shortest path
            try:
                path = nx.shortest_path(self.lineage_graph, source_node_id, target_node_id)
            except nx.NetworkXNoPath:
                return timedelta(hours=24)  # Default if no path
            
            total_time = timedelta()
            
            # Estimate time for each hop in the path
            for i in range(len(path) - 1):
                current_node = path[i]
                next_node = path[i + 1]
                
                # Get edge information
                edge_data = self.lineage_graph.edges[current_node, next_node]
                edge_type = LineageEdgeType(edge_data['edge_type'])
                
                # Estimate processing time based on edge type
                processing_times = {
                    LineageEdgeType.DERIVES_FROM: timedelta(minutes=5),
                    LineageEdgeType.TRANSFORMS_TO: timedelta(minutes=15),
                    LineageEdgeType.AGGREGATES_FROM: timedelta(minutes=30),
                    LineageEdgeType.JOINS_WITH: timedelta(minutes=10),
                    LineageEdgeType.FILTERS_FROM: timedelta(minutes=2),
                    LineageEdgeType.ENRICHES: timedelta(minutes=20),
                    LineageEdgeType.VALIDATES: timedelta(minutes=5),
                    LineageEdgeType.TRIGGERS: timedelta(seconds=30)
                }
                
                hop_time = processing_times.get(edge_type, timedelta(minutes=10))
                
                # Adjust based on node type
                next_node_data = self.lineage_graph.nodes[next_node]
                node_type = LineageNodeType(next_node_data['node_type'])
                
                if node_type == LineageNodeType.DATABASE_TABLE:
                    hop_time += timedelta(minutes=5)  # Database write time
                elif node_type == LineageNodeType.ANALYSIS_RESULT:
                    hop_time += timedelta(minutes=30)  # Analysis computation time
                elif node_type == LineageNodeType.REPORT:
                    hop_time += timedelta(hours=1)  # Report generation time
                
                total_time += hop_time
            
            return total_time
            
        except Exception as e:
            logger.error("Failed to estimate propagation time", error=str(e))
            return timedelta(hours=2)  # Default estimate
    
    async def _find_impact_paths(self, 
                               source_node_id: str,
                               target_node_id: str) -> List[List[str]]:
        """Find all paths from source to target node"""
        try:
            # Find all simple paths (no cycles)
            paths = list(nx.all_simple_paths(
                self.lineage_graph, 
                source_node_id, 
                target_node_id,
                cutoff=10  # Limit path length
            ))
            
            return paths[:5]  # Return up to 5 paths
            
        except Exception as e:
            logger.error("Failed to find impact paths", error=str(e))
            return []
    
    async def _generate_impact_recommendations(self,
                                             source_node_id: str,
                                             affected_nodes: List[str],
                                             impact_levels: Dict[str, ImpactLevel],
                                             change_type: str) -> List[str]:
        """Generate recommendations for managing impact"""
        recommendations = []
        
        try:
            # Count impact levels
            critical_nodes = [n for n, l in impact_levels.items() if l == ImpactLevel.CRITICAL]
            high_impact_nodes = [n for n, l in impact_levels.items() if l == ImpactLevel.HIGH]
            
            # General recommendations
            if critical_nodes:
                recommendations.append(
                    f"CRITICAL: {len(critical_nodes)} nodes will be critically impacted. "
                    "Consider implementing the change during maintenance window."
                )
                
                recommendations.append(
                    "Create rollback plan and test thoroughly in staging environment."
                )
            
            if high_impact_nodes:
                recommendations.append(
                    f"HIGH IMPACT: {len(high_impact_nodes)} nodes will be significantly affected. "
                    "Notify downstream system owners before implementing change."
                )
            
            # Change type specific recommendations
            if change_type == 'schema_change':
                recommendations.append(
                    "Schema change detected. Ensure backward compatibility or coordinate "
                    "with all downstream consumers."
                )
                
                recommendations.append(
                    "Update data contracts and API documentation."
                )
            
            elif change_type == 'data_format_change':
                recommendations.append(
                    "Data format change may require updates to transformation logic "
                    "in downstream systems."
                )
            
            elif change_type == 'business_logic_change':
                recommendations.append(
                    "Business logic change may affect analytical results. "
                    "Validate outputs after implementation."
                )
            
            # Node-specific recommendations
            source_node = self.lineage_graph.nodes[source_node_id]
            source_type = LineageNodeType(source_node['node_type'])
            
            if source_type == LineageNodeType.DATA_SOURCE:
                recommendations.append(
                    "Data source change detected. Monitor data quality metrics "
                    "closely after implementation."
                )
            
            # Add monitoring recommendations
            recommendations.append(
                "Set up monitoring alerts for affected downstream systems."
            )
            
            recommendations.append(
                "Schedule post-implementation validation of all affected nodes."
            )
            
        except Exception as e:
            logger.error("Failed to generate recommendations", error=str(e))
            recommendations.append("Unable to generate specific recommendations. Manual review required.")
        
        return recommendations
    
    async def _store_impact_analysis(self, result: ImpactAnalysisResult) -> None:
        """Store impact analysis result"""
        try:
            # Store in PostgreSQL
            await self.postgres_conn.execute("""
                INSERT INTO impact_analysis_results 
                (analysis_id, source_node_id, affected_nodes, impact_levels, 
                 estimated_propagation_time, recommendations, analysis_timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """, 
                result.analysis_id,
                result.source_node_id,
                result.affected_nodes,
                json.dumps({k: v.value for k, v in result.impact_levels.items()}),
                json.dumps({k: v.total_seconds() for k, v in result.estimated_propagation_time.items()}),
                result.recommendations,
                result.analysis_timestamp
            )
            
            # Cache in Redis
            cache_key = f"impact_analysis:{result.analysis_id}"
            await self.redis_client.setex(
                cache_key,
                86400,  # 24 hours TTL
                json.dumps(asdict(result), default=str)
            )
            
        except Exception as e:
            logger.error("Failed to store impact analysis", error=str(e))
    
    async def record_data_quality_metrics(self, 
                                        node_id: str,
                                        quality_metrics: DataQualityMetrics) -> None:
        """Record data quality metrics for a lineage node"""
        try:
            # Store in PostgreSQL
            await self.postgres_conn.execute("""
                INSERT INTO data_quality_metrics 
                (node_id, completeness_score, accuracy_score, consistency_score,
                 timeliness_score, validity_score, overall_quality_score,
                 quality_issues)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
                quality_metrics.node_id,
                quality_metrics.completeness_score,
                quality_metrics.accuracy_score,
                quality_metrics.consistency_score,
                quality_metrics.timeliness_score,
                quality_metrics.validity_score,
                quality_metrics.overall_quality_score,
                quality_metrics.quality_issues
            )
            
            # Update node metadata with latest quality score
            await self.postgres_conn.execute("""
                UPDATE lineage_nodes 
                SET metadata = metadata || $2
                WHERE node_id = $1
            """, 
                node_id,
                json.dumps({
                    'latest_quality_score': quality_metrics.overall_quality_score,
                    'quality_measurement_timestamp': quality_metrics.measurement_timestamp.isoformat()
                })
            )
            
            logger.info("Recorded data quality metrics", 
                       node_id=node_id,
                       overall_score=quality_metrics.overall_quality_score)
            
        except Exception as e:
            logger.error("Failed to record quality metrics", 
                        node_id=node_id,
                        error=str(e))
            raise
    
    async def get_lineage_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the lineage system"""
        try:
            # Count nodes by type
            node_counts = defaultdict(int)
            for node_id, node_data in self.lineage_graph.nodes(data=True):
                node_type = node_data.get('node_type', 'unknown')
                node_counts[node_type] += 1
            
            # Count edges by type
            edge_counts = defaultdict(int)
            for source, target, edge_data in self.lineage_graph.edges(data=True):
                edge_type = edge_data.get('edge_type', 'unknown')
                edge_counts[edge_type] += 1
            
            # Calculate graph metrics
            total_nodes = len(self.lineage_graph.nodes)
            total_edges = len(self.lineage_graph.edges)
            
            # Find nodes with highest in-degree (most dependencies)
            in_degrees = dict(self.lineage_graph.in_degree())
            most_dependent_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Find nodes with highest out-degree (most dependents)
            out_degrees = dict(self.lineage_graph.out_degree())
            most_influential_nodes = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Get recent quality metrics
            recent_quality = await self.postgres_conn.fetch("""
                SELECT node_id, overall_quality_score, measurement_timestamp
                FROM data_quality_metrics
                WHERE measurement_timestamp >= NOW() - INTERVAL '24 hours'
                ORDER BY measurement_timestamp DESC
                LIMIT 10
            """)
            
            summary = {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'node_counts_by_type': dict(node_counts),
                'edge_counts_by_type': dict(edge_counts),
                'most_dependent_nodes': [
                    {'node_id': node_id, 'dependencies': count}
                    for node_id, count in most_dependent_nodes
                ],
                'most_influential_nodes': [
                    {'node_id': node_id, 'dependents': count}
                    for node_id, count in most_influential_nodes
                ],
                'recent_quality_metrics': [
                    {
                        'node_id': str(record['node_id']),
                        'quality_score': record['overall_quality_score'],
                        'timestamp': record['measurement_timestamp'].isoformat()
                    }
                    for record in recent_quality
                ],
                'graph_density': nx.density(self.lineage_graph),
                'is_dag': nx.is_directed_acyclic_graph(self.lineage_graph),
                'summary_timestamp': datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error("Failed to generate lineage summary", error=str(e))
            raise
    
    async def detect_lineage_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the data lineage graph"""
        anomalies = []
        
        try:
            # Detect orphaned nodes (no incoming or outgoing edges)
            for node_id, node_data in self.lineage_graph.nodes(data=True):
                in_degree = self.lineage_graph.in_degree(node_id)
                out_degree = self.lineage_graph.out_degree(node_id)
                
                if in_degree == 0 and out_degree == 0:
                    anomalies.append({
                        'type': 'orphaned_node',
                        'node_id': node_id,
                        'description': f"Node {node_data.get('name', node_id)} has no connections",
                        'severity': 'medium'
                    })
            
            # Detect cycles (should not exist in proper lineage)
            try:
                cycles = list(nx.simple_cycles(self.lineage_graph))
                for cycle in cycles:
                    anomalies.append({
                        'type': 'circular_dependency',
                        'nodes': cycle,
                        'description': f"Circular dependency detected: {' -> '.join(cycle)}",
                        'severity': 'high'
                    })
            except:
                pass  # Graph might be too large for cycle detection
            
            # Detect nodes with unusually high fan-out
            for node_id, out_degree in self.lineage_graph.out_degree():
                if out_degree > 50:  # Threshold for high fan-out
                    node_data = self.lineage_graph.nodes[node_id]
                    anomalies.append({
                        'type': 'high_fan_out',
                        'node_id': node_id,
                        'out_degree': out_degree,
                        'description': f"Node {node_data.get('name', node_id)} has unusually high fan-out ({out_degree})",
                        'severity': 'low'
                    })
            
            # Detect nodes with low quality scores
            recent_quality = await self.postgres_conn.fetch("""
                SELECT DISTINCT ON (node_id) node_id, overall_quality_score
                FROM data_quality_metrics
                ORDER BY node_id, measurement_timestamp DESC
            """)
            
            for record in recent_quality:
                if record['overall_quality_score'] < 0.5:  # Low quality threshold
                    node_data = self.lineage_graph.nodes.get(str(record['node_id']), {})
                    anomalies.append({
                        'type': 'low_quality_score',
                        'node_id': str(record['node_id']),
                        'quality_score': record['overall_quality_score'],
                        'description': f"Node {node_data.get('name', record['node_id'])} has low quality score ({record['overall_quality_score']:.2f})",
                        'severity': 'medium'
                    })
            
            logger.info("Detected lineage anomalies", 
                       total_anomalies=len(anomalies),
                       high_severity=len([a for a in anomalies if a['severity'] == 'high']))
            
            return anomalies
            
        except Exception as e:
            logger.error("Failed to detect lineage anomalies", error=str(e))
            return []


# Example usage and testing
async def example_lineage_tracking():
    """Example of comprehensive data lineage tracking"""
    
    # Initialize lineage tracker
    lineage_tracker = DataLineageTracker(
        neo4j_uri='bolt://localhost:7687',
        neo4j_username='neo4j',
        neo4j_password='prometheus123',
        postgres_connection_string='postgresql://prometheus:prometheus123@localhost:5432/prometheus',
        redis_url='redis://localhost:6379'
    )
    
    await lineage_tracker.initialize()
    
    try:
        # Register data sources
        seismic_source_id = await lineage_tracker.register_data_source(
            source_name='IRIS Seismic Network',
            source_type='real_time_stream',
            database_name='postgresql',
            table_name='seismic_events',
            schema_info={
                'fields': {
                    'timestamp': 'datetime',
                    'station_id': 'string',
                    'magnitude': 'float',
                    'depth_km': 'float',
                    'latitude': 'float',
                    'longitude': 'float'
                }
            },
            owner='seismic_team',
            tags=['seismic', 'real_time', 'iris']
        )
        
        # Register transformation
        transformation_id = await lineage_tracker.register_transformation(
            transformation_name='Seismic Event Processing',
            description='Process raw seismic events and calculate derived metrics',
            input_nodes=[seismic_source_id],
            output_node=seismic_source_id,  # Simplified for example
            transformation_logic='SELECT *, magnitude * 1.5 as adjusted_magnitude FROM seismic_events',
            field_mappings={'magnitude': 'adjusted_magnitude'},
            owner='data_engineering_team'
        )
        
        # Trace lineage
        upstream_lineage = await lineage_tracker.trace_lineage_upstream(transformation_id)
        downstream_lineage = await lineage_tracker.trace_lineage_downstream(seismic_source_id)
        
        # Perform impact analysis
        impact_analysis = await lineage_tracker.perform_impact_analysis(
            seismic_source_id,
            change_type='schema_change',
            severity='high'
        )
        
        # Record quality metrics
        quality_metrics = DataQualityMetrics(
            node_id=seismic_source_id,
            completeness_score=0.95,
            accuracy_score=0.92,
            consistency_score=0.88,
            timeliness_score=0.97,
            validity_score=0.94,
            overall_quality_score=0.93,
            measurement_timestamp=datetime.utcnow(),
            quality_issues=['minor_timestamp_gaps']
        )
        
        await lineage_tracker.record_data_quality_metrics(seismic_source_id, quality_metrics)
        
        # Get summary and detect anomalies
        summary = await lineage_tracker.get_lineage_summary()
        anomalies = await lineage_tracker.detect_lineage_anomalies()
        
        logger.info("Lineage tracking example completed", 
                   summary=summary,
                   anomalies_detected=len(anomalies))
        
    finally:
        await lineage_tracker.close()


if __name__ == "__main__":
    asyncio.run(example_lineage_tracking())