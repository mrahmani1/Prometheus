"""
Knowledge Graph Manager for dynamic updates and integration
Manages the scientific knowledge graph with real-time updates
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

import aioredis
import asyncpg
import numpy as np
import structlog
from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
from neo4j import AsyncGraphDatabase

from .config import settings
from .knowledge_extractor import ExtractedKnowledge
from .semantic_reasoner import (InferredRelationship, SemanticReasoner,
                                SemanticTriple)

logger = structlog.get_logger()


@dataclass
class KnowledgeUpdate:
    """Knowledge graph update event"""

    update_id: str
    update_type: str  # "addition", "modification", "deletion"
    affected_concepts: List[str]
    confidence_change: float
    source: str
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class DiscoveryEvent:
    """New scientific discovery event"""

    discovery_id: str
    discovery_type: str
    concepts_involved: List[str]
    relationships_discovered: List[Dict[str, Any]]
    confidence_score: float
    supporting_evidence: List[str]
    timestamp: datetime


class KnowledgeGraphManager:
    """Manages the scientific knowledge graph with dynamic updates"""

    def __init__(self):
        self.neo4j_driver = None
        self.redis_client = None
        self.postgres_pool = None
        self.influxdb_client = None
        self.semantic_reasoner = None

        # Update tracking
        self.pending_updates = []
        self.update_history = []
        self.discovery_events = []

        # Integration status
        self.external_integrations = {
            "semantic_scholar": False,
            "crossre": False,
            "pubmed": False,
            "geonames": False,
        }

    async def initialize(self):
        """Initialize knowledge graph manager and connections"""
        logger.info("Initializing Knowledge Graph Manager")

        try:
            # Initialize database connections
            await self._initialize_neo4j()
            await self._initialize_redis()
            await self._initialize_postgres()
            await self._initialize_influxdb()

            # Initialize semantic reasoner
            self.semantic_reasoner = SemanticReasoner()
            await self.semantic_reasoner.initialize()

            # Setup knowledge graph schema
            await self._setup_knowledge_graph_schema()

            # Initialize external integrations
            await self._initialize_external_integrations()

            logger.info("Knowledge Graph Manager initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize Knowledge Graph Manager", error=str(e))
            raise

    async def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        self.neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )

        # Test connection
        async with self.neo4j_driver.session() as session:
            result = await session.run("RETURN 1 as test")
            await result.single()

        logger.info("Neo4j connection established")

    async def _initialize_redis(self):
        """Initialize Redis connection"""
        redis_url = "redis://localhost:6379"
        self.redis_client = await aioredis.from_url(redis_url)

        # Test connection
        await self.redis_client.ping()
        logger.info("Redis connection established")

    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection pool"""
        self.postgres_pool = await asyncpg.create_pool(settings.postgres_uri)
        logger.info("PostgreSQL connection pool established")

    async def _initialize_influxdb(self):
        """Initialize InfluxDB connection"""
        self.influxdb_client = InfluxDBClientAsync(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
        )

        # Test connection
        health = await self.influxdb_client.health()
        if health.status != "pass":
            raise Exception("InfluxDB health check failed")

        logger.info("InfluxDB connection established")

    async def _setup_knowledge_graph_schema(self):
        """Setup Neo4j schema for knowledge graph"""
        schema_queries = [
            # Create constraints
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT relationship_id IF NOT EXISTS FOR (r:Relationship) REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT paper_doi IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
            # Create indexes
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR (r:Relationship) ON (r.type)",
            "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX temporal_index IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
            # Create full-text indexes
            "CALL db.index.fulltext.createNodeIndex('conceptSearch', ['Concept'], ['name', 'definition', 'description'])",
            "CALL db.index.fulltext.createNodeIndex('paperSearch', ['Paper'], ['title', 'abstract', 'keywords'])",
        ]

        async with self.neo4j_driver.session() as session:
            for query in schema_queries:
                try:
                    await session.run(query)
                except Exception as e:
                    # Ignore if constraint/index already exists
                    if "already exists" not in str(e).lower():
                        logger.warning(
                            "Failed to create schema element", query=query, error=str(e)
                        )

        logger.info("Knowledge graph schema setup completed")

    async def _initialize_external_integrations(self):
        """Initialize connections to external scientific databases"""
        # This would initialize connections to external APIs
        # For now, mark as initialized if API keys are available

        if settings.semantic_scholar_api_key:
            self.external_integrations["semantic_scholar"] = True

        if settings.crossref_api_key:
            self.external_integrations["crossre"] = True

        if settings.pubmed_api_key:
            self.external_integrations["pubmed"] = True

        # GeoNames doesn't require API key for basic usage
        self.external_integrations["geonames"] = True

        logger.info(
            "External integrations initialized", integrations=self.external_integrations
        )

    async def add_extracted_knowledge(
        self, extracted_knowledge: ExtractedKnowledge
    ) -> str:
        """Add extracted knowledge to the graph"""
        logger.info("Adding extracted knowledge to graph")

        update_id = f"update_{datetime.utcnow().isoformat()}"

        try:
            async with self.neo4j_driver.session() as session:
                # Add entities as concepts
                for entity in extracted_knowledge.entities:
                    await self._add_concept(
                        session, entity, extracted_knowledge.source_metadata
                    )

                # Add relationships
                for relationship in extracted_knowledge.relationships:
                    await self._add_relationship(
                        session, relationship, extracted_knowledge.source_metadata
                    )

                # Add concepts with definitions
                for concept in extracted_knowledge.concepts:
                    await self._add_concept_definition(
                        session, concept, extracted_knowledge.source_metadata
                    )

                # Add temporal facts
                for temporal_fact in extracted_knowledge.temporal_facts:
                    await self._add_temporal_fact(
                        session, temporal_fact, extracted_knowledge.source_metadata
                    )

                # Add source paper
                await self._add_source_paper(
                    session, extracted_knowledge.source_metadata
                )

            # Update semantic reasoner
            await self._update_semantic_reasoner(extracted_knowledge)

            # Record update event
            update_event = KnowledgeUpdate(
                update_id=update_id,
                update_type="addition",
                affected_concepts=[e["text"] for e in extracted_knowledge.entities],
                confidence_change=extracted_knowledge.confidence_scores["overall"],
                source=extracted_knowledge.source_metadata["doi"],
                timestamp=datetime.utcnow(),
                metadata=extracted_knowledge.source_metadata,
            )

            self.update_history.append(update_event)

            # Cache update in Redis
            await self.redis_client.setex(
                f"knowledge_update:{update_id}",
                3600,  # 1 hour TTL
                json.dumps(asdict(update_event), default=str),
            )

            logger.info("Knowledge successfully added to graph", update_id=update_id)
            return update_id

        except Exception as e:
            logger.error("Failed to add knowledge to graph", error=str(e))
            raise

    async def _add_concept(
        self, session, entity: Dict[str, Any], source_metadata: Dict[str, Any]
    ):
        """Add a concept node to the graph"""
        query = """
        MERGE (c:Concept {name: $name})
        ON CREATE SET
            c.id = randomUUID(),
            c.type = $type,
            c.confidence = $confidence,
            c.created_at = datetime(),
            c.sources = [$source]
        ON MATCH SET
            c.confidence = (c.confidence + $confidence) / 2,
            c.sources = c.sources + $source,
            c.updated_at = datetime()
        RETURN c.id as concept_id
        """

        await session.run(
            query,
            {
                "name": entity["text"],
                "type": entity["label"],
                "confidence": entity["confidence"],
                "source": source_metadata["doi"],
            },
        )

    async def _add_relationship(
        self, session, relationship: Dict[str, Any], source_metadata: Dict[str, Any]
    ):
        """Add a relationship between concepts"""
        query = """
        MATCH (source:Concept {name: $source_name})
        MATCH (target:Concept {name: $target_name})
        MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
        ON CREATE SET
            r.id = randomUUID(),
            r.confidence = $confidence,
            r.context = $context,
            r.created_at = datetime(),
            r.sources = [$source]
        ON MATCH SET
            r.confidence = (r.confidence + $confidence) / 2,
            r.sources = r.sources + $source,
            r.updated_at = datetime()
        """

        await session.run(
            query,
            {
                "source_name": relationship["source"],
                "target_name": relationship["target"],
                "rel_type": relationship["relationship_type"],
                "confidence": relationship["confidence"],
                "context": relationship["context"],
                "source": source_metadata["doi"],
            },
        )

    async def _add_concept_definition(
        self, session, concept: Dict[str, Any], source_metadata: Dict[str, Any]
    ):
        """Add concept definition"""
        query = """
        MERGE (c:Concept {name: $name})
        ON CREATE SET
            c.id = randomUUID(),
            c.definition = $definition,
            c.confidence = $confidence,
            c.created_at = datetime(),
            c.sources = [$source]
        ON MATCH SET
            c.definition = CASE
                WHEN c.definition IS NULL THEN $definition
                ELSE c.definition + "; " + $definition
            END,
            c.confidence = (c.confidence + $confidence) / 2,
            c.sources = c.sources + $source,
            c.updated_at = datetime()
        """

        await session.run(
            query,
            {
                "name": concept["name"],
                "definition": concept["definition"],
                "confidence": concept["confidence"],
                "source": source_metadata["doi"],
            },
        )

    async def _add_temporal_fact(
        self, session, temporal_fact: Dict[str, Any], source_metadata: Dict[str, Any]
    ):
        """Add temporal fact to the graph"""
        query = """
        CREATE (tf:TemporalFact {
            id: randomUUID(),
            temporal_marker: $temporal_marker,
            event_description: $event_description,
            context: $context,
            confidence: $confidence,
            source: $source,
            created_at: datetime()
        })
        """

        await session.run(
            query,
            {
                "temporal_marker": temporal_fact["temporal_marker"],
                "event_description": temporal_fact["event_description"],
                "context": temporal_fact["context"],
                "confidence": temporal_fact["confidence"],
                "source": source_metadata["doi"],
            },
        )

    async def _add_source_paper(self, session, source_metadata: Dict[str, Any]):
        """Add source paper to the graph"""
        query = """
        MERGE (p:Paper {doi: $doi})
        ON CREATE SET
            p.id = randomUUID(),
            p.title = $title,
            p.authors = $authors,
            p.publication_date = $publication_date,
            p.journal = $journal,
            p.created_at = datetime()
        """

        await session.run(
            query,
            {
                "doi": source_metadata["doi"],
                "title": source_metadata["title"],
                "authors": source_metadata["authors"],
                "publication_date": source_metadata["publication_date"],
                "journal": source_metadata["journal"],
            },
        )

    async def _update_semantic_reasoner(self, extracted_knowledge: ExtractedKnowledge):
        """Update semantic reasoner with new knowledge"""
        # Add semantic triples to reasoner
        for entity in extracted_knowledge.entities:
            triple = SemanticTriple(
                subject=entity["text"],
                predicate="rdf:type",
                object=entity["label"],
                confidence=entity["confidence"],
                source=extracted_knowledge.source_metadata["doi"],
                timestamp=datetime.utcnow(),
            )
            await self.semantic_reasoner.add_knowledge_triple(triple)

        for relationship in extracted_knowledge.relationships:
            triple = SemanticTriple(
                subject=relationship["source"],
                predicate=relationship["relationship_type"],
                object=relationship["target"],
                confidence=relationship["confidence"],
                source=extracted_knowledge.source_metadata["doi"],
                timestamp=datetime.utcnow(),
            )
            await self.semantic_reasoner.add_knowledge_triple(triple)

    async def discover_new_relationships(
        self, concepts: List[str]
    ) -> List[InferredRelationship]:
        """Discover new relationships using semantic reasoning"""
        logger.info("Discovering new relationships", concept_count=len(concepts))

        # Use semantic reasoner to infer relationships
        inferred_relationships = await self.semantic_reasoner.infer_relationships(
            concepts
        )

        # Filter high-confidence relationships
        high_confidence_relationships = [
            rel
            for rel in inferred_relationships
            if rel.confidence >= settings.causal_discovery_confidence_threshold
        ]

        # Add high-confidence relationships to the graph
        async with self.neo4j_driver.session() as session:
            for relationship in high_confidence_relationships:
                await self._add_inferred_relationship(session, relationship)

        # Record discovery event
        if high_confidence_relationships:
            discovery_event = DiscoveryEvent(
                discovery_id=f"discovery_{datetime.utcnow().isoformat()}",
                discovery_type="semantic_inference",
                concepts_involved=concepts,
                relationships_discovered=[
                    asdict(rel) for rel in high_confidence_relationships
                ],
                confidence_score=np.mean(
                    [rel.confidence for rel in high_confidence_relationships]
                ),
                supporting_evidence=[
                    rel.reasoning_chain[0] for rel in high_confidence_relationships
                ],
                timestamp=datetime.utcnow(),
            )

            self.discovery_events.append(discovery_event)

            # Cache discovery in Redis
            await self.redis_client.setex(
                f"discovery:{discovery_event.discovery_id}",
                86400,  # 24 hours TTL
                json.dumps(asdict(discovery_event), default=str),
            )

        logger.info(
            "Relationship discovery completed",
            discovered=len(high_confidence_relationships),
        )
        return inferred_relationships

    async def _add_inferred_relationship(
        self, session, relationship: InferredRelationship
    ):
        """Add inferred relationship to the graph"""
        query = """
        MATCH (source:Concept {name: $source_name})
        MATCH (target:Concept {name: $target_name})
        MERGE (source)-[r:INFERRED {type: $rel_type}]->(target)
        ON CREATE SET
            r.id = randomUUID(),
            r.confidence = $confidence,
            r.reasoning_chain = $reasoning_chain,
            r.supporting_evidence = $supporting_evidence,
            r.created_at = datetime()
        ON MATCH SET
            r.confidence = (r.confidence + $confidence) / 2,
            r.updated_at = datetime()
        """

        await session.run(
            query,
            {
                "source_name": relationship.source_concept,
                "target_name": relationship.target_concept,
                "rel_type": relationship.relationship_type,
                "confidence": relationship.confidence,
                "reasoning_chain": relationship.reasoning_chain,
                "supporting_evidence": relationship.supporting_evidence,
            },
        )

    async def update_from_new_discoveries(self, discoveries: List[Dict[str, Any]]):
        """Update knowledge graph from new scientific discoveries"""
        logger.info(
            "Updating knowledge graph from new discoveries", count=len(discoveries)
        )

        for discovery in discoveries:
            try:
                # Process discovery and extract knowledge
                extracted_knowledge = await self._process_discovery(discovery)

                # Add to knowledge graph
                await self.add_extracted_knowledge(extracted_knowledge)

                # Trigger relationship discovery
                concepts = [e["text"] for e in extracted_knowledge.entities]
                await self.discover_new_relationships(concepts)

                logger.info(
                    "Discovery processed successfully", discovery_id=discovery.get("id")
                )

            except Exception as e:
                logger.error(
                    "Failed to process discovery",
                    discovery_id=discovery.get("id"),
                    error=str(e),
                )

    async def _process_discovery(self, discovery: Dict[str, Any]) -> ExtractedKnowledge:
        """Process a discovery into extracted knowledge format"""
        # This would convert discovery data into ExtractedKnowledge format
        # For now, create a placeholder implementation

        return ExtractedKnowledge(
            entities=[],
            relationships=[],
            concepts=[],
            temporal_facts=[],
            confidence_scores={"overall": 0.8},
            source_metadata={
                "doi": discovery.get("doi", "unknown"),
                "title": discovery.get("title", "Unknown Discovery"),
                "authors": discovery.get("authors", []),
                "publication_date": datetime.utcnow().isoformat(),
                "journal": discovery.get("journal", "Unknown"),
                "extraction_timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        async with self.neo4j_driver.session() as session:
            stats_query = """
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[r:RELATED]->()
            OPTIONAL MATCH (c)-[ir:INFERRED]->()
            RETURN
                count(DISTINCT c) as concept_count,
                count(DISTINCT r) as relationship_count,
                count(DISTINCT ir) as inferred_relationship_count,
                avg(c.confidence) as avg_concept_confidence
            """

            result = await session.run(stats_query)
            stats = await result.single()

            return {
                "concepts": stats["concept_count"],
                "relationships": stats["relationship_count"],
                "inferred_relationships": stats["inferred_relationship_count"],
                "average_confidence": float(stats["avg_concept_confidence"] or 0),
                "total_updates": len(self.update_history),
                "total_discoveries": len(self.discovery_events),
                "external_integrations": self.external_integrations,
            }

    async def search_knowledge(
        self, query: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search knowledge graph using full-text search"""
        async with self.neo4j_driver.session() as session:
            search_query = """
            CALL db.index.fulltext.queryNodes('conceptSearch', $query)
            YIELD node, score
            RETURN node.name as name, node.definition as definition,
                   node.confidence as confidence, score
            ORDER BY score DESC
            LIMIT $limit
            """

            results = await session.run(search_query, {"query": query, "limit": limit})

            search_results = []
            async for record in results:
                search_results.append(
                    {
                        "name": record["name"],
                        "definition": record["definition"],
                        "confidence": record["confidence"],
                        "relevance_score": record["score"],
                    }
                )

            return search_results

    async def is_healthy(self) -> bool:
        """Check if the knowledge graph manager is healthy"""
        try:
            # Test Neo4j connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")

            # Test Redis connection
            await self.redis_client.ping()

            return True
        except Exception:
            return False

    async def close(self):
        """Close all connections"""
        logger.info("Closing Knowledge Graph Manager")

        if self.neo4j_driver:
            await self.neo4j_driver.close()

        if self.redis_client:
            await self.redis_client.close()

        if self.postgres_pool:
            await self.postgres_pool.close()

        if self.influxdb_client:
            await self.influxdb_client.close()

        if self.semantic_reasoner:
            await self.semantic_reasoner.close()
