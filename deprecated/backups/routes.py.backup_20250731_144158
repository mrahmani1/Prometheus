"""
API routes for the Scientific Knowledge Graph Service
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import structlog

from core.knowledge_extractor import ScientificPaper, ExtractedKnowledge
from core.semantic_reasoner import InferredRelationship
from core.temporal_causal_engine import TemporalCausalRelationship
from core.federated_discovery import DatasetMetadata, FederatedCausalResult

logger = structlog.get_logger()

router = APIRouter()

# Pydantic models for API requests/responses

class PaperRequest(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    full_text: Optional[str] = None
    doi: str
    publication_date: datetime
    journal: str
    keywords: List[str] = []
    citations: List[str] = []

class KnowledgeExtractionResponse(BaseModel):
    extraction_id: str
    entities_count: int
    relationships_count: int
    concepts_count: int
    temporal_facts_count: int
    confidence_scores: Dict[str, float]
    processing_time: float

class CausalDiscoveryRequest(BaseModel):
    time_series_data: Dict[str, List[Dict[str, Any]]]
    max_lag: int = Field(default=50, ge=1, le=200)
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    methods: List[str] = ["granger_causality", "transfer_entropy"]

class CausalDiscoveryResponse(BaseModel):
    discovery_id: str
    relationships: List[Dict[str, Any]]
    causal_network: Dict[str, Any]
    processing_time: float

class FederatedDiscoveryRequest(BaseModel):
    target_variables: List[str]
    max_lag: int = Field(default=50, ge=1, le=200)
    confidence_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    privacy_level: str = Field(default="differential_privacy")
    aggregation_methods: List[str] = ["weighted_average", "median_consensus"]

class DatasetRegistrationRequest(BaseModel):
    dataset_id: str
    institution: str
    location: str
    temporal_range_start: datetime
    temporal_range_end: datetime
    variables: List[str]
    sample_count: int
    data_quality_score: float = Field(ge=0.0, le=1.0)
    privacy_level: str = Field(default="public")
    contact_info: str
    data_endpoint: str

class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    search_type: str = Field(default="concepts")  # "concepts", "relationships", "papers"

class SemanticReasoningRequest(BaseModel):
    concepts: List[str]
    reasoning_depth: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.6, ge=0.1, le=1.0)

# Dependency to get service components
async def get_service_components():
    from main import service_components
    return service_components

# Knowledge extraction endpoints

@router.post("/extract/paper", response_model=KnowledgeExtractionResponse)
async def extract_knowledge_from_paper(
    paper_request: PaperRequest,
    background_tasks: BackgroundTasks,
    components = Depends(get_service_components)
):
    """Extract knowledge from a scientific paper"""
    
    try:
        start_time = datetime.utcnow()
        
        # Convert request to ScientificPaper
        paper = ScientificPaper(
            title=paper_request.title,
            authors=paper_request.authors,
            abstract=paper_request.abstract,
            full_text=paper_request.full_text,
            doi=paper_request.doi,
            publication_date=paper_request.publication_date,
            journal=paper_request.journal,
            keywords=paper_request.keywords,
            citations=paper_request.citations
        )
        
        # Extract knowledge
        knowledge_extractor = components["knowledge_extractor"]
        extracted_knowledge = await knowledge_extractor.extract_from_paper(paper)
        
        # Add to knowledge graph in background
        kg_manager = components["knowledge_graph_manager"]
        background_tasks.add_task(
            kg_manager.add_extracted_knowledge, 
            extracted_knowledge
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return KnowledgeExtractionResponse(
            extraction_id=f"extract_{datetime.utcnow().isoformat()}",
            entities_count=len(extracted_knowledge.entities),
            relationships_count=len(extracted_knowledge.relationships),
            concepts_count=len(extracted_knowledge.concepts),
            temporal_facts_count=len(extracted_knowledge.temporal_facts),
            confidence_scores=extracted_knowledge.confidence_scores,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Knowledge extraction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Knowledge extraction failed: {str(e)}")

@router.post("/extract/batch")
async def extract_knowledge_batch(
    papers: List[PaperRequest],
    background_tasks: BackgroundTasks,
    components = Depends(get_service_components)
):
    """Extract knowledge from multiple papers"""
    
    try:
        start_time = datetime.utcnow()
        
        # Convert requests to ScientificPaper objects
        paper_objects = []
        for paper_request in papers:
            paper = ScientificPaper(
                title=paper_request.title,
                authors=paper_request.authors,
                abstract=paper_request.abstract,
                full_text=paper_request.full_text,
                doi=paper_request.doi,
                publication_date=paper_request.publication_date,
                journal=paper_request.journal,
                keywords=paper_request.keywords,
                citations=paper_request.citations
            )
            paper_objects.append(paper)
        
        # Extract knowledge from all papers
        knowledge_extractor = components["knowledge_extractor"]
        extracted_knowledge_list = await knowledge_extractor.extract_from_multiple_sources(paper_objects)
        
        # Add all to knowledge graph in background
        kg_manager = components["knowledge_graph_manager"]
        for extracted_knowledge in extracted_knowledge_list:
            background_tasks.add_task(
                kg_manager.add_extracted_knowledge, 
                extracted_knowledge
            )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "batch_id": f"batch_{datetime.utcnow().isoformat()}",
            "processed_papers": len(extracted_knowledge_list),
            "total_entities": sum(len(ek.entities) for ek in extracted_knowledge_list),
            "total_relationships": sum(len(ek.relationships) for ek in extracted_knowledge_list),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error("Batch knowledge extraction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch extraction failed: {str(e)}")

# Causal discovery endpoints

@router.post("/causal/discover", response_model=CausalDiscoveryResponse)
async def discover_temporal_causality(
    request: CausalDiscoveryRequest,
    components = Depends(get_service_components)
):
    """Discover temporal causal relationships in time-series data"""
    
    try:
        start_time = datetime.utcnow()
        
        # Convert request data to pandas DataFrames
        import pandas as pd
        time_series_data = {}
        
        for variable_name, data_points in request.time_series_data.items():
            df = pd.DataFrame(data_points)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            time_series_data[variable_name] = df
        
        # Perform causal discovery
        temporal_causal_engine = components["temporal_causal_engine"]
        causal_relationships = await temporal_causal_engine.discover_temporal_causality(
            time_series_data,
            request.max_lag,
            request.confidence_threshold
        )
        
        # Build causal network
        causal_network = await temporal_causal_engine.build_causal_network(causal_relationships)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Convert relationships to dict format
        relationships_dict = []
        for rel in causal_relationships:
            relationships_dict.append({
                "cause_variable": rel.cause_variable,
                "effect_variable": rel.effect_variable,
                "time_lag": rel.time_lag,
                "causal_strength": rel.causal_strength,
                "confidence": rel.confidence,
                "p_value": rel.p_value,
                "method": rel.method,
                "supporting_evidence": rel.supporting_evidence
            })
        
        return CausalDiscoveryResponse(
            discovery_id=f"causal_{datetime.utcnow().isoformat()}",
            relationships=relationships_dict,
            causal_network={
                "nodes": causal_network.nodes,
                "edge_count": len(causal_network.edges),
                "network_confidence": causal_network.network_confidence,
                "temporal_resolution": causal_network.temporal_resolution
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error("Causal discovery failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Causal discovery failed: {str(e)}")

# Federated discovery endpoints

@router.post("/federated/register")
async def register_federated_dataset(
    request: DatasetRegistrationRequest,
    components = Depends(get_service_components)
):
    """Register a dataset for federated causal discovery"""
    
    try:
        # Create dataset metadata
        metadata = DatasetMetadata(
            dataset_id=request.dataset_id,
            institution=request.institution,
            location=request.location,
            temporal_range=(request.temporal_range_start, request.temporal_range_end),
            variables=request.variables,
            sample_count=request.sample_count,
            data_quality_score=request.data_quality_score,
            privacy_level=request.privacy_level,
            contact_info=request.contact_info
        )
        
        # Register dataset
        federated_discovery = components["federated_discovery"]
        await federated_discovery.register_dataset(metadata, request.data_endpoint)
        
        return {
            "status": "success",
            "dataset_id": request.dataset_id,
            "registration_time": datetime.utcnow(),
            "message": "Dataset registered successfully for federated discovery"
        }
        
    except Exception as e:
        logger.error("Dataset registration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dataset registration failed: {str(e)}")

@router.post("/federated/discover")
async def discover_federated_causality(
    request: FederatedDiscoveryRequest,
    components = Depends(get_service_components)
):
    """Discover causal relationships across federated datasets"""
    
    try:
        start_time = datetime.utcnow()
        
        # Perform federated causal discovery
        federated_discovery = components["federated_discovery"]
        federated_results = await federated_discovery.discover_federated_causality(
            request.target_variables,
            request.max_lag,
            request.confidence_threshold,
            request.privacy_level
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Convert results to dict format
        results_dict = []
        for result in federated_results:
            results_dict.append({
                "relationship": {
                    "cause_variable": result.relationship.cause_variable,
                    "effect_variable": result.relationship.effect_variable,
                    "time_lag": result.relationship.time_lag,
                    "causal_strength": result.relationship.causal_strength,
                    "confidence": result.relationship.confidence,
                    "method": result.relationship.method
                },
                "participating_datasets": result.participating_datasets,
                "consensus_score": result.consensus_score,
                "privacy_preserved": result.privacy_preserved,
                "aggregation_method": result.aggregation_method
            })
        
        return {
            "discovery_id": f"federated_{datetime.utcnow().isoformat()}",
            "federated_results": results_dict,
            "total_relationships": len(federated_results),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error("Federated causal discovery failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Federated discovery failed: {str(e)}")

@router.get("/federated/status")
async def get_federation_status(components = Depends(get_service_components)):
    """Get status of the federated discovery system"""
    
    try:
        federated_discovery = components["federated_discovery"]
        status = await federated_discovery.get_federation_status()
        return status
        
    except Exception as e:
        logger.error("Failed to get federation status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get federation status: {str(e)}")

# Semantic reasoning endpoints

@router.post("/semantic/infer")
async def infer_semantic_relationships(
    request: SemanticReasoningRequest,
    components = Depends(get_service_components)
):
    """Infer semantic relationships between concepts"""
    
    try:
        start_time = datetime.utcnow()
        
        # Perform semantic reasoning
        semantic_reasoner = components["semantic_reasoner"]
        inferred_relationships = await semantic_reasoner.infer_relationships(request.concepts)
        
        # Filter by confidence threshold
        filtered_relationships = [
            rel for rel in inferred_relationships 
            if rel.confidence >= request.confidence_threshold
        ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Convert to dict format
        relationships_dict = []
        for rel in filtered_relationships:
            relationships_dict.append({
                "source_concept": rel.source_concept,
                "target_concept": rel.target_concept,
                "relationship_type": rel.relationship_type,
                "confidence": rel.confidence,
                "reasoning_chain": rel.reasoning_chain,
                "supporting_evidence": rel.supporting_evidence
            })
        
        return {
            "inference_id": f"semantic_{datetime.utcnow().isoformat()}",
            "inferred_relationships": relationships_dict,
            "total_inferences": len(filtered_relationships),
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error("Semantic reasoning failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Semantic reasoning failed: {str(e)}")

@router.get("/semantic/concept/{concept_name}")
async def get_concept_hierarchy(
    concept_name: str,
    components = Depends(get_service_components)
):
    """Get hierarchical relationships for a concept"""
    
    try:
        semantic_reasoner = components["semantic_reasoner"]
        hierarchy = await semantic_reasoner.get_concept_hierarchy(concept_name)
        return hierarchy
        
    except Exception as e:
        logger.error("Failed to get concept hierarchy", concept=concept_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get concept hierarchy: {str(e)}")

@router.post("/semantic/validate")
async def validate_knowledge_consistency(components = Depends(get_service_components)):
    """Validate the consistency of the knowledge graph"""
    
    try:
        semantic_reasoner = components["semantic_reasoner"]
        validation_results = await semantic_reasoner.validate_knowledge_consistency()
        return validation_results
        
    except Exception as e:
        logger.error("Knowledge validation failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Knowledge validation failed: {str(e)}")

# Search and query endpoints

@router.post("/search")
async def search_knowledge_graph(
    request: SearchRequest,
    components = Depends(get_service_components)
):
    """Search the knowledge graph"""
    
    try:
        kg_manager = components["knowledge_graph_manager"]
        
        if request.search_type == "concepts":
            results = await kg_manager.search_knowledge(request.query, request.limit)
        else:
            # For other search types, use the same method for now
            results = await kg_manager.search_knowledge(request.query, request.limit)
        
        return {
            "query": request.query,
            "search_type": request.search_type,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("Knowledge search failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {str(e)}")

@router.post("/query/sparql")
async def execute_sparql_query(
    sparql_query: str,
    components = Depends(get_service_components)
):
    """Execute SPARQL query on the knowledge graph"""
    
    try:
        semantic_reasoner = components["semantic_reasoner"]
        results = await semantic_reasoner.query_knowledge(sparql_query)
        
        return {
            "query": sparql_query,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error("SPARQL query failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"SPARQL query failed: {str(e)}")

# Statistics and monitoring endpoints

@router.get("/statistics")
async def get_knowledge_graph_statistics(components = Depends(get_service_components)):
    """Get statistics about the knowledge graph"""
    
    try:
        kg_manager = components["knowledge_graph_manager"]
        stats = await kg_manager.get_knowledge_statistics()
        return stats
        
    except Exception as e:
        logger.error("Failed to get statistics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/sync/federated")
async def synchronize_federated_knowledge(
    background_tasks: BackgroundTasks,
    components = Depends(get_service_components)
):
    """Synchronize knowledge across federated datasets"""
    
    try:
        federated_discovery = components["federated_discovery"]
        
        # Run synchronization in background
        background_tasks.add_task(
            federated_discovery.synchronize_federated_knowledge
        )
        
        return {
            "status": "started",
            "message": "Federated knowledge synchronization started in background",
            "start_time": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Failed to start federated sync", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start federated sync: {str(e)}")

@router.post("/discover/relationships")
async def discover_new_relationships(
    concepts: List[str],
    background_tasks: BackgroundTasks,
    components = Depends(get_service_components)
):
    """Discover new relationships between concepts"""
    
    try:
        kg_manager = components["knowledge_graph_manager"]
        
        # Run relationship discovery in background
        background_tasks.add_task(
            kg_manager.discover_new_relationships,
            concepts
        )
        
        return {
            "status": "started",
            "concepts": concepts,
            "message": "Relationship discovery started in background",
            "start_time": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Failed to start relationship discovery", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start relationship discovery: {str(e)}")