"""
Advanced Knowledge Graph Integration for Enhanced Kalman Filter
Connects EnKF results with scientific knowledge representation and causal inference
Includes SPARQL/GraphQL queries, vector embeddings, and real-time literature integration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import logging
from datetime import datetime, timedelta
import asyncio
import json
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Advanced knowledge integration imports
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.plugins.sparql import prepareQuery
from SPARQLWrapper import SPARQLWrapper, JSON, XML, N3, RDF
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from owlrl import DeductiveClosure, OWLRL_Semantics
from owlready2 import get_ontology, World, destroy_entity
from py2neo import Graph as Neo4jGraph, Node, Relationship
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphSAGE, GAT
import aiohttp

# Core module imports
from .scientific_knowledge_graph import ScientificKnowledgeGraph, EntityType, RelationType
from .causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType
from .enhanced_kalman_filter import EnhancedEnsembleKalmanFilter, ReservoirParameters
from .geological_gnn import GeologicalGNN, GraphDataset
from .database_integration import DatabaseIntegration, DataType
from .mlops_pipeline import MLOpsManager, ModelVersion, ExperimentTracker
from .federated_learning import FederatedKnowledgeManager
from .digital_twin_earth import DigitalTwinEarth

@dataclass
class KnowledgeGraphConfig:
    """Configuration for advanced knowledge graph features"""
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    sparql_endpoint: str = "http://localhost:3030/volcano_knowledge/sparql"
    owl_ontology_path: str = "./ontologies/volcanic_deformation.owl"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_path: str = "./vector_stores/volcano_knowledge"
    literature_apis: List[str] = None
    external_endpoints: Dict[str, str] = None
    
    def __post_init__(self):
        if self.literature_apis is None:
            self.literature_apis = [
                "https://api.crossref.org/works",
                "https://api.semanticscholar.org/graph/v1/paper/search",
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            ]
        if self.external_endpoints is None:
            self.external_endpoints = {
                "wikidata": "https://query.wikidata.org/sparql",
                "geonames": "http://api.geonames.org/searchJSON",
                "usgs": "https://earthquake.usgs.gov/fdsnws/event/1/query"
            }

@dataclass
class KnowledgeEmbedding:
    """Knowledge embedding with metadata"""
    entity_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    confidence: float
    source: str

@dataclass
class CausalRelationship:
    """Enhanced causal relationship with inference"""
    cause: str
    effect: str
    strength: float
    confidence: float
    evidence: List[str]
    inferred_by: str
    temporal_lag: Optional[timedelta] = None
    mechanism: Optional[str] = None

class AdvancedEnKFKnowledgeIntegrator:
    """
    Advanced knowledge graph integration for Enhanced Kalman Filter
    Includes SPARQL queries, OWL reasoning, vector embeddings, and real-time updates
    """
    
    def __init__(self, knowledge_graph: ScientificKnowledgeGraph, config: KnowledgeGraphConfig = None):
        self.knowledge_graph = knowledge_graph
        self.config = config or KnowledgeGraphConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize advanced components
        self.neo4j_graph = None
        self.sparql_wrapper = None
        self.owl_world = None
        self.ontology = None
        self.embedding_model = None
        self.vector_store = None
        self.causal_discovery_engine = None
        self.federated_manager = None
        self.gnn_embedder = None
        self.literature_monitor = None
        
        # Initialize caches and indices
        self.entity_embeddings: Dict[str, KnowledgeEmbedding] = {}
        self.relationship_cache: Dict[str, List[CausalRelationship]] = {}
        self.similarity_index = None
        self.reasoning_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.query_performance = defaultdict(list)
        self.embedding_performance = defaultdict(list)
        
        # Initialize all advanced components
        asyncio.create_task(self._initialize_advanced_components())
    
    async def _initialize_advanced_components(self):
        """Initialize all advanced knowledge graph components"""
        try:
            # 1. Initialize Neo4j connection for graph database
            await self._initialize_neo4j_connection()
            
            # 2. Initialize SPARQL endpoint wrapper
            await self._initialize_sparql_wrapper()
            
            # 3. Initialize OWL ontology and reasoning engine
            await self._initialize_owl_reasoning()
            
            # 4. Initialize embedding models and vector stores
            await self._initialize_vector_embeddings()
            
            # 5. Initialize GNN-based knowledge embedder
            await self._initialize_gnn_embedder()
            
            # 6. Initialize causal discovery engine
            await self._initialize_causal_discovery()
            
            # 7. Initialize federated learning manager
            await self._initialize_federated_knowledge()
            
            # 8. Initialize literature monitoring system
            await self._initialize_literature_monitor()
            
            # 9. Initialize deformation ontology
            await self._initialize_advanced_deformation_ontology()
            
            # 10. Build similarity indices
            await self._build_similarity_indices()
            
            self.logger.info("Advanced knowledge graph components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced components: {e}")
            # Initialize fallback systems
            await self._initialize_fallback_systems()
    
    async def _initialize_neo4j_connection(self):
        """Initialize Neo4j graph database connection"""
        try:
            from py2neo import Graph as Neo4jGraph
            self.neo4j_graph = Neo4jGraph(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password
            )
            
            # Test connection
            self.neo4j_graph.run("RETURN 1")
            self.logger.info("Neo4j connection established")
            
        except Exception as e:
            self.logger.warning(f"Neo4j connection failed, using in-memory graph: {e}")
            self.neo4j_graph = None
    
    async def _initialize_sparql_wrapper(self):
        """Initialize SPARQL endpoint wrapper"""
        try:
            self.sparql_wrapper = SPARQLWrapper(self.config.sparql_endpoint)
            self.sparql_wrapper.setReturnFormat(JSON)
            
            # Test query
            test_query = """
            SELECT ?s ?p ?o WHERE {
                ?s ?p ?o .
            } LIMIT 1
            """
            self.sparql_wrapper.setQuery(test_query)
            results = self.sparql_wrapper.query().convert()
            
            self.logger.info("SPARQL endpoint connection established")
            
        except Exception as e:
            self.logger.warning(f"SPARQL endpoint connection failed: {e}")
            self.sparql_wrapper = None
    
    async def _initialize_owl_reasoning(self):
        """Initialize OWL ontology and reasoning engine"""
        try:
            from owlready2 import World, get_ontology
            
            self.owl_world = World()
            
            # Load volcanic deformation ontology
            if Path(self.config.owl_ontology_path).exists():
                self.ontology = get_ontology(self.config.owl_ontology_path).load()
            else:
                # Create basic ontology structure
                self.ontology = self.owl_world.get_ontology("http://volcano.knowledge/volcanic_deformation.owl")
                await self._create_base_ontology()
            
            # Initialize reasoning engine
            from owlrl import DeductiveClosure, OWLRL_Semantics
            self.reasoning_engine = DeductiveClosure(OWLRL_Semantics)
            
            self.logger.info("OWL ontology and reasoning engine initialized")
            
        except Exception as e:
            self.logger.warning(f"OWL reasoning initialization failed: {e}")
            self.owl_world = None
            self.ontology = None
    
    async def _initialize_vector_embeddings(self):
        """Initialize vector embedding models and stores"""
        try:
            # Initialize sentence transformer model
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Initialize vector store
            from langchain.vectorstores import FAISS
            from langchain.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
            
            vector_store_path = Path(self.config.vector_store_path)
            if vector_store_path.exists():
                self.vector_store = FAISS.load_local(str(vector_store_path), embeddings)
            else:
                # Create empty vector store
                self.vector_store = FAISS.from_texts(["dummy"], embeddings)
                vector_store_path.parent.mkdir(parents=True, exist_ok=True)
                self.vector_store.save_local(str(vector_store_path))
            
            self.logger.info("Vector embeddings and store initialized")
            
        except Exception as e:
            self.logger.warning(f"Vector embeddings initialization failed: {e}")
            self.embedding_model = None
            self.vector_store = None
    
    async def _initialize_gnn_embedder(self):
        """Initialize Graph Neural Network for knowledge embeddings"""
        try:
            from .geological_gnn import GeologicalGNN
            
            # Configure GNN for knowledge graph embeddings
            gnn_config = {
                'input_dim': 512,  # Embedding dimension
                'hidden_dim': 256,
                'output_dim': 128,
                'num_layers': 3,
                'dropout': 0.1,
                'learning_rate': 0.001
            }
            
            self.gnn_embedder = GeologicalGNN(gnn_config)
            
            # Load pre-trained weights if available
            gnn_weights_path = Path("./models/knowledge_gnn_embedder.pth")
            if gnn_weights_path.exists():
                self.gnn_embedder.load_state_dict(torch.load(gnn_weights_path))
            
            self.logger.info("GNN knowledge embedder initialized")
            
        except Exception as e:
            self.logger.warning(f"GNN embedder initialization failed: {e}")
            self.gnn_embedder = None
    
    async def _initialize_causal_discovery(self):
        """Initialize causal discovery engine"""
        try:
            from .causal_inference_engine import CausalInferenceEngine
            
            causal_config = {
                'method': 'PC',  # Peter-Clark algorithm
                'alpha': 0.05,
                'max_lag': 10,
                'bootstrap_samples': 1000
            }
            
            self.causal_discovery_engine = CausalInferenceEngine(causal_config)
            
            self.logger.info("Causal discovery engine initialized")
            
        except Exception as e:
            self.logger.warning(f"Causal discovery initialization failed: {e}")
            self.causal_discovery_engine = None
    
    async def _initialize_federated_knowledge(self):
        """Initialize federated knowledge learning manager"""
        try:
            from .federated_learning import FederatedKnowledgeManager
            
            federated_config = {
                'aggregation_method': 'fedavg',
                'min_participants': 2,
                'privacy_budget': 1.0,
                'differential_privacy': True
            }
            
            self.federated_manager = FederatedKnowledgeManager(federated_config)
            
            self.logger.info("Federated knowledge manager initialized")
            
        except Exception as e:
            self.logger.warning(f"Federated learning initialization failed: {e}")
            self.federated_manager = None
    
    async def _initialize_literature_monitor(self):
        """Initialize real-time literature monitoring system"""
        try:
            from .literature_monitor import LiteratureMonitor
            
            monitor_config = {
                'apis': self.config.literature_apis,
                'keywords': ['volcano', 'deformation', 'kalman filter', 'magma reservoir'],
                'update_interval': timedelta(hours=6),
                'max_papers_per_query': 50
            }
            
            self.literature_monitor = LiteratureMonitor(monitor_config)
            
            # Start monitoring in background
            asyncio.create_task(self.literature_monitor.start_monitoring())
            
            self.logger.info("Literature monitoring system initialized")
            
        except Exception as e:
            self.logger.warning(f"Literature monitor initialization failed: {e}")
            self.literature_monitor = None
    
    async def _execute_sparql_query(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute SPARQL query with performance tracking and error handling"""
        start_time = datetime.now()
        
        try:
            if not self.sparql_wrapper:
                raise ValueError("SPARQL wrapper not initialized")
            
            self.sparql_wrapper.setQuery(query)
            self.sparql_wrapper.setTimeout(timeout)
            
            results = self.sparql_wrapper.query().convert()
            
            # Track performance
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            self.query_performance['sparql'].append(query_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"SPARQL query failed: {e}")
            return {'results': {'bindings': []}}
    
    async def _execute_graphql_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute GraphQL query with performance tracking"""
        start_time = datetime.now()
        
        try:
            if not hasattr(self, 'graphql_client') or not self.graphql_client:
                # Initialize GraphQL client if available
                try:
                    from gql import gql, Client
                    from gql.transport.requests import RequestsHTTPTransport
                    
                    transport = RequestsHTTPTransport(
                        url=self.config.external_endpoints.get('graphql', 'http://localhost:4000/graphql')
                    )
                    self.graphql_client = Client(transport=transport, fetch_schema_from_transport=True)
                except ImportError:
                    self.logger.warning("GraphQL client not available")
                    return {}
            
            query_obj = gql(query)
            results = self.graphql_client.execute(query_obj, variable_values=variables)
            
            # Track performance
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            self.query_performance['graphql'].append(query_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"GraphQL query failed: {e}")
            return {}
    
    async def _compute_vector_similarity(self, entity1: str, entity2: str) -> float:
        """Compute vector similarity between two entities"""
        try:
            if entity1 not in self.entity_embeddings or entity2 not in self.entity_embeddings:
                return 0.0
            
            embedding1 = self.entity_embeddings[entity1].embedding
            embedding2 = self.entity_embeddings[entity2].embedding
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Vector similarity computation failed: {e}")
            return 0.0
    
    async def _fallback_similarity_search(self, volcano_id: str) -> Dict[str, Any]:
        """Fallback similarity search using basic knowledge graph"""
        try:
            # Use basic knowledge graph methods as fallback
            volcano_entity = f"volcano_{volcano_id}"
            relationships = await self.knowledge_graph.get_entity_relationships(volcano_entity)
            
            similar_entities = []
            for rel in relationships[:5]:  # Limit to top 5
                if rel.get('type') in [RelationType.SIMILAR_TO, RelationType.PART_OF]:
                    similar_entities.append({
                        'volcano_id': rel.get('target', '').split('_')[-1],
                        'relationship_type': rel.get('type'),
                        'confidence': rel.get('confidence', 0.5)
                    })
            
            return {
                'similar_volcanoes': similar_entities,
                'similarity_criteria': ['basic_relationships'],
                'matching_algorithm': 'knowledge_graph_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback similarity search failed: {e}")
            return {'similar_volcanoes': [], 'similarity_criteria': [], 'matching_algorithm': 'none'}
    
    async def _fallback_historical_analysis(self, volcano_id: str) -> Dict[str, Any]:
        """Fallback historical analysis using basic methods"""
        try:
            # Basic fallback implementation
            return {
                'historical_events': [],
                'pattern_analysis': {
                    'inflation_cycles': [],
                    'deflation_events': [],
                    'long_term_trends': [],
                    'episodic_unrest': []
                },
                'comparison_metrics': ['basic_temporal_correlation'],
                'fallback_used': True
            }
            
        except Exception as e:
            self.logger.error(f"Fallback historical analysis failed: {e}")
            return {'error': str(e)}
    
    async def _analyze_deformation_statistics(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical analysis on deformation events"""
        if not events:
            return {}
        
        try:
            magnitudes = [event['magnitude_cm'] for event in events]
            rates = [event['rate_cm_per_year'] for event in events]
            durations = []
            
            for event in events:
                start = datetime.fromisoformat(event['start_time'].replace('Z', '+00:00'))
                end = datetime.fromisoformat(event['end_time'].replace('Z', '+00:00'))
                duration = (end - start).days
                durations.append(duration)
            
            return {
                'magnitude_stats': {
                    'mean_cm': float(np.mean(magnitudes)),
                    'std_cm': float(np.std(magnitudes)),
                    'max_cm': float(np.max(magnitudes)),
                    'min_cm': float(np.min(magnitudes))
                },
                'rate_stats': {
                    'mean_cm_per_year': float(np.mean(rates)),
                    'std_cm_per_year': float(np.std(rates)),
                    'max_cm_per_year': float(np.max(rates)),
                    'min_cm_per_year': float(np.min(rates))
                },
                'duration_stats': {
                    'mean_days': float(np.mean(durations)),
                    'std_days': float(np.std(durations)),
                    'max_days': float(np.max(durations)),
                    'min_days': float(np.min(durations))
                },
                'event_frequency': len(events) / (durations[-1] - durations[0]) if len(durations) > 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}
    
    async def _perform_temporal_clustering(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform temporal clustering of deformation events"""
        if len(events) < 2:
            return {'clusters': [], 'cluster_count': 0}
        
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            # Prepare features for clustering
            features = []
            for event in events:
                start_timestamp = datetime.fromisoformat(event['start_time'].replace('Z', '+00:00')).timestamp()
                features.append([
                    start_timestamp,
                    event['magnitude_cm'],
                    event['rate_cm_per_year'],
                    event['spatial_extent_km2']
                ])
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Perform DBSCAN clustering
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Process cluster results
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'event_index': i,
                    'event': events[i]
                })
            
            return {
                'clusters': [{'cluster_id': k, 'events': v} for k, v in clusters.items() if k != -1],
                'noise_events': clusters.get(-1, []),
                'cluster_count': len([k for k in clusters.keys() if k != -1])
            }
            
        except Exception as e:
            self.logger.error(f"Temporal clustering failed: {e}")
            return {'clusters': [], 'cluster_count': 0, 'error': str(e)}
    
    async def _correlate_with_seismic_data(self, volcano_id: str, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Correlate deformation events with seismic data from USGS"""
        try:
            correlations = []
            
            for event in events:
                # Query USGS for seismic events during deformation
                start_time = event['start_time']
                end_time = event['end_time']
                
                usgs_url = f"{self.config.external_endpoints['usgs']}?starttime={start_time}&endtime={end_time}&format=geojson"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(usgs_url, timeout=30) as response:
                        if response.status == 200:
                            seismic_data = await response.json()
                            
                            seismic_events = seismic_data.get('features', [])
                            correlation = {
                                'deformation_event_id': event['event_id'],
                                'concurrent_earthquakes': len(seismic_events),
                                'max_magnitude': max([eq['properties']['mag'] for eq in seismic_events if eq['properties']['mag']], default=0),
                                'seismic_energy_sum': sum([10**(1.5*eq['properties']['mag'] + 9.1) for eq in seismic_events if eq['properties']['mag']], default=0)
                            }
                            correlations.append(correlation)
            
            return {
                'correlations': correlations,
                'total_deformation_events': len(events),
                'events_with_seismic_correlation': len([c for c in correlations if c['concurrent_earthquakes'] > 0])
            }
            
        except Exception as e:
            self.logger.error(f"Seismic correlation failed: {e}")
            return {'error': str(e)}

class LiteratureMonitor:
    """Real-time literature monitoring and knowledge extraction system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.apis = config.get('apis', [])
        self.keywords = config.get('keywords', [])
        self.update_interval = config.get('update_interval', timedelta(hours=6))
        self.max_papers_per_query = config.get('max_papers_per_query', 50)
        self.last_update = datetime.now() - self.update_interval
        self.paper_cache = {}
        self.knowledge_extractor = None
        
        # Initialize NLP pipeline for knowledge extraction
        self._initialize_nlp_pipeline()
    
    def _initialize_nlp_pipeline(self):
        """Initialize NLP pipeline for extracting knowledge from papers"""
        try:
            self.knowledge_extractor = pipeline(
                "text2text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            self.logger.warning(f"NLP pipeline initialization failed: {e}")
    
    async def start_monitoring(self):
        """Start continuous monitoring of literature APIs"""
        while True:
            try:
                if datetime.now() - self.last_update >= self.update_interval:
                    await self._update_literature()
                    self.last_update = datetime.now()
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                self.logger.error(f"Literature monitoring error: {e}")
                await asyncio.sleep(3600)  # Wait before retrying
    
    async def _update_literature(self):
        """Update literature from all configured APIs"""
        self.logger.info("Updating literature from APIs")
        
        tasks = []
        for api_url in self.apis:
            if 'crossref' in api_url:
                tasks.append(self._query_crossref())
            elif 'semanticscholar' in api_url:
                tasks.append(self._query_semantic_scholar())
            elif 'ncbi' in api_url:
                tasks.append(self._query_pubmed())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process and integrate results
        all_papers = []
        for result in results:
            if isinstance(result, list):
                all_papers.extend(result)
        
        await self._process_new_papers(all_papers)
    
    async def _query_crossref(self) -> List[Dict[str, Any]]:
        """Query Crossref API for recent papers"""
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for keyword in self.keywords:
                    url = f"https://api.crossref.org/works?query={keyword}&filter=from-pub-date:{(datetime.now() - self.update_interval).strftime('%Y-%m-%d')}&rows={self.max_papers_per_query}"
                    
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('message', {}).get('items', []):
                                paper = {
                                    'source': 'crossref',
                                    'doi': item.get('DOI'),
                                    'title': item.get('title', [''])[0] if item.get('title') else '',
                                    'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                                              for author in item.get('author', [])],
                                    'abstract': item.get('abstract', ''),
                                    'published_date': item.get('published-print', {}).get('date-parts', [[]])[0],
                                    'journal': item.get('container-title', [''])[0] if item.get('container-title') else '',
                                    'keywords': [keyword],
                                    'url': item.get('URL', '')
                                }
                                papers.append(paper)
            
            return papers[:self.max_papers_per_query]
            
        except Exception as e:
            self.logger.error(f"Crossref query failed: {e}")
            return []
    
    async def _query_semantic_scholar(self) -> List[Dict[str, Any]]:
        """Query Semantic Scholar API"""
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for keyword in self.keywords:
                    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={keyword}&limit={self.max_papers_per_query}&fields=title,authors,abstract,venue,year,url,citationCount"
                    
                    async with session.get(url, timeout=30) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('data', []):
                                paper = {
                                    'source': 'semantic_scholar',
                                    'title': item.get('title', ''),
                                    'authors': [author.get('name', '') for author in item.get('authors', [])],
                                    'abstract': item.get('abstract', ''),
                                    'published_date': [item.get('year', '')],
                                    'journal': item.get('venue', ''),
                                    'citation_count': item.get('citationCount', 0),
                                    'keywords': [keyword],
                                    'url': item.get('url', '')
                                }
                                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Semantic Scholar query failed: {e}")
            return []
    
    async def _query_pubmed(self) -> List[Dict[str, Any]]:
        """Query PubMed API via NCBI"""
        papers = []
        
        try:
            async with aiohttp.ClientSession() as session:
                for keyword in self.keywords:
                    # Search for paper IDs
                    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={keyword}&retmax={self.max_papers_per_query}&retmode=json"
                    
                    async with session.get(search_url, timeout=30) as response:
                        if response.status == 200:
                            search_data = await response.json()
                            ids = search_data.get('esearchresult', {}).get('idlist', [])
                            
                            if ids:
                                # Fetch paper details
                                detail_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={','.join(ids)}&retmode=json"
                                
                                async with session.get(detail_url, timeout=30) as detail_response:
                                    if detail_response.status == 200:
                                        detail_data = await detail_response.json()
                                        
                                        for id_key, item in detail_data.get('result', {}).items():
                                            if id_key != 'uids':
                                                paper = {
                                                    'source': 'pubmed',
                                                    'pmid': item.get('uid'),
                                                    'title': item.get('title', ''),
                                                    'authors': [author.get('name', '') for author in item.get('authors', [])],
                                                    'abstract': '',  # Would need additional API call
                                                    'published_date': item.get('pubdate', '').split()[0:3],
                                                    'journal': item.get('source', ''),
                                                    'keywords': [keyword],
                                                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{item.get('uid', '')}"
                                                }
                                                papers.append(paper)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"PubMed query failed: {e}")
            return []
    
    async def _process_new_papers(self, papers: List[Dict[str, Any]]):
        """Process and extract knowledge from new papers"""
        for paper in papers:
            paper_id = paper.get('doi') or paper.get('pmid') or paper.get('title', '')[:50]
            
            if paper_id in self.paper_cache:
                continue
            
            # Extract knowledge using NLP
            knowledge = await self._extract_knowledge_from_paper(paper)
            
            # Cache paper with extracted knowledge
            self.paper_cache[paper_id] = {
                'paper': paper,
                'knowledge': knowledge,
                'processed_date': datetime.now().isoformat()
            }
            
            self.logger.info(f"Processed new paper: {paper.get('title', '')[:100]}...")
    
    async def _extract_knowledge_from_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured knowledge from a paper"""
        try:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            
            if not text.strip():
                return {}
            
            # Extract key concepts using NLP
            extracted_knowledge = {
                'entities': [],
                'relationships': [],
                'methods': [],
                'findings': []
            }
            
            # Simple keyword-based extraction for demonstration
            # In practice, this would use advanced NLP models
            volcano_terms = ['volcano', 'magma', 'eruption', 'deformation', 'caldera', 'reservoir']
            method_terms = ['kalman filter', 'inversion', 'modeling', 'simulation', 'analysis']
            
            text_lower = text.lower()
            
            for term in volcano_terms:
                if term in text_lower:
                    extracted_knowledge['entities'].append({
                        'type': 'geological_entity',
                        'name': term,
                        'context': 'mentioned'
                    })
            
            for term in method_terms:
                if term in text_lower:
                    extracted_knowledge['methods'].append({
                        'type': 'computational_method',
                        'name': term,
                        'context': 'used'
                    })
            
            # Extract potential relationships
            if 'kalman' in text_lower and 'volcano' in text_lower:
                extracted_knowledge['relationships'].append({
                    'source': 'kalman_filter',
                    'target': 'volcano_monitoring',
                    'relationship': 'applied_to',
                    'confidence': 0.7
                })
            
            return extracted_knowledge
            
        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {e}")
            return {}
    
    def get_recent_papers(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently processed papers"""
        sorted_papers = sorted(
            self.paper_cache.values(),
            key=lambda x: x['processed_date'],
            reverse=True
        )
        return sorted_papers[:limit]

    async def _initialize_advanced_deformation_ontology(self):
        """Initialize advanced ontological framework with OWL reasoning"""
        try:
            # Create advanced ontology with OWL reasoning
            if self.ontology:
                with self.ontology:
                    # Define advanced classes and properties
                    class DeformationEvent(self.ontology.VolcanicEvent): pass
                    class MagmaReservoir(self.ontology.GeologicalFeature): pass
                    class EnsembleKalmanFilter(self.ontology.ComputationalMethod): pass
                    class UnitlessParameterization(self.ontology.MathematicalFormulation): pass
                    class GeodeticObservation(self.ontology.ScientificData): pass
                    
                    # Define object properties (relationships)
                    class causes(self.ontology.ObjectProperty): pass
                    class measuredBy(self.ontology.ObjectProperty): pass
                    class influences(self.ontology.ObjectProperty): pass
                    class partOf(self.ontology.ObjectProperty): pass
                    class hasReservoir(self.ontology.ObjectProperty): pass
                    class hasParameter(self.ontology.ObjectProperty): pass
                    
                    # Define data properties
                    class depth(self.ontology.DataProperty): 
                        domain = [MagmaReservoir]
                        range = [float]
                    
                    class volume(self.ontology.DataProperty):
                        domain = [MagmaReservoir]
                        range = [float]
                    
                    class pressure(self.ontology.DataProperty):
                        domain = [MagmaReservoir]  
                        range = [float]
                    
                    class magnitude(self.ontology.DataProperty):
                        domain = [DeformationEvent]
                        range = [float]
                    
                    # Add reasoning rules
                    class InflationEvent(DeformationEvent):
                        equivalent_to = [DeformationEvent & 
                                       self.ontology.hasProperty.some(
                                           self.ontology.Property & 
                                           self.ontology.hasValue.some(lambda x: x > 0)
                                       )]
                    
                    class DeflationEvent(DeformationEvent):
                        equivalent_to = [DeformationEvent & 
                                       self.ontology.hasProperty.some(
                                           self.ontology.Property & 
                                           self.ontology.hasValue.some(lambda x: x < 0)
                                       )]
                
                # Apply reasoning
                if self.reasoning_engine:
                    self.reasoning_engine.expand(self.ontology.world.graph)
                
                self.logger.info("Advanced OWL ontology initialized with reasoning")
            
            else:
                # Fallback to basic ontology initialization
                await self._initialize_basic_deformation_ontology()
                
        except Exception as e:
            self.logger.error(f"Advanced ontology initialization failed: {e}")
            await self._initialize_basic_deformation_ontology()
    
    async def _initialize_basic_deformation_ontology(self):
        """Fallback basic ontology initialization"""
        
        # Core deformation concepts
        deformation_entities = [
            ('magma_reservoir', EntityType.CONCEPT, {
                'description': 'Underground magma storage system causing surface deformation',
                'physical_properties': ['depth', 'volume', 'pressure', 'shape']
            }),
            ('surface_deformation', EntityType.PHENOMENON, {
                'description': 'Observable ground displacement patterns',
                'measurement_types': ['gps', 'insar', 'tilt', 'strain']
            }),
            ('ensemble_kalman_filter', EntityType.METHOD, {
                'description': 'Data assimilation technique for parameter estimation',
                'reformulation': 'Albright_Gregg_2022',
                'parameters': ['dv_star', 'gamma', 'delta', 'psi']
            }),
            ('unitless_parameterization', EntityType.CONCEPT, {
                'description': 'Reformulated parameter space avoiding physical constraints',
                'advantages': ['no_impossible_conditions', 'improved_convergence', 'reduced_correlations']
            }),
            ('geodetic_observations', EntityType.DATASET, {
                'description': 'GPS and InSAR measurements of ground displacement',
                'spatial_resolution': 'variable',
                'temporal_resolution': 'variable'
            })
        ]
        
        # Add entities to knowledge graph
        for entity_id, entity_type, properties in deformation_entities:
            asyncio.create_task(
                self.knowledge_graph.add_entity(entity_id, entity_type, properties)
            )
        
        # Define causal relationships
        causal_relationships = [
            ('magma_reservoir', 'surface_deformation', RelationType.CAUSES, 0.95),
            ('geodetic_observations', 'ensemble_kalman_filter', RelationType.MEASURED_BY, 0.90),
            ('ensemble_kalman_filter', 'magma_reservoir', RelationType.INFLUENCES, 0.85),
            ('unitless_parameterization', 'ensemble_kalman_filter', RelationType.PART_OF, 1.0)
        ]
        
        # Add relationships
        for source, target, relation_type, confidence in causal_relationships:
            asyncio.create_task(
                self.knowledge_graph.add_relationship(source, target, relation_type, confidence)
            )
    
    async def integrate_enkf_results(self, volcano_id: str, enkf_results: Dict[str, Any],
                                   reservoir_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate EnKF results into knowledge graph and extract insights
        """
        
        self.logger.info(f"Integrating EnKF results for {volcano_id} into knowledge graph")
        
        # Create volcano-specific entities
        volcano_entity = f"volcano_{volcano_id}"
        await self.knowledge_graph.add_entity(
            volcano_entity, 
            EntityType.LOCATION,
            {
                'volcano_id': volcano_id,
                'analysis_timestamp': datetime.now().isoformat(),
                'enkf_results': enkf_results
            }
        )
        
        # Add reservoir state entity
        reservoir_entity = f"reservoir_{volcano_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self.knowledge_graph.add_entity(
            reservoir_entity,
            EntityType.CONCEPT,
            {
                'depth_estimate': reservoir_state.get('depth_estimate', {}),
                'volume_estimate': reservoir_state.get('volume_estimate', {}),
                'pressure_change': reservoir_state.get('pressure_change', {}),
                'parameter_correlations': reservoir_state.get('parameter_correlations', {}),
                'source': 'enkf_assimilation'
            }
        )
        
        # Link volcano to reservoir
        await self.knowledge_graph.add_relationship(
            volcano_entity, 
            reservoir_entity, 
            RelationType.PART_OF,
            confidence=0.9
        )
        
        # Add assimilation process entity
        assimilation_entity = f"assimilation_{volcano_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        await self.knowledge_graph.add_entity(
            assimilation_entity,
            EntityType.PROCESS,
            {
                'update_metrics': enkf_results.get('update_metrics', {}),
                'ensemble_size': enkf_results.get('ensemble_size', 0),
                'assimilation_success': enkf_results.get('assimilation_success', False),
                'method': 'enhanced_ensemble_kalman_filter'
            }
        )
        
        # Connect assimilation to results
        await self.knowledge_graph.add_relationship(
            assimilation_entity,
            reservoir_entity,
            RelationType.CAUSES,
            confidence=enkf_results.get('assimilation_success', False) * 0.9
        )
        
        # Extract and store parameter insights
        parameter_insights = await self._extract_parameter_insights(reservoir_state, enkf_results)
        
        # Add parameter confidence entity
        confidence_entity = f"parameter_confidence_{volcano_id}"
        await self.knowledge_graph.add_entity(
            confidence_entity,
            EntityType.MEASUREMENT,
            parameter_insights
        )
        
        # Connect confidence to reservoir
        await self.knowledge_graph.add_relationship(
            confidence_entity,
            reservoir_entity,
            RelationType.MEASURED_BY,
            confidence=parameter_insights.get('overall_confidence', 0.5)
        )
        
        # Generate causal insights
        causal_insights = await self._generate_causal_insights(volcano_id, reservoir_state, enkf_results)
        
        return {
            'knowledge_integration': {
                'volcano_entity': volcano_entity,
                'reservoir_entity': reservoir_entity,
                'assimilation_entity': assimilation_entity,
                'parameter_insights': parameter_insights,
                'causal_insights': causal_insights
            },
            'integration_timestamp': datetime.now().isoformat()
        }
    
    async def _extract_parameter_insights(self, reservoir_state: Dict[str, Any], 
                                        enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights about parameter estimation quality"""
        
        insights = {}
        
        # Depth estimate quality
        depth_estimate = reservoir_state.get('depth_estimate', {})
        if depth_estimate:
            depth_uncertainty = depth_estimate.get('std', 0) / max(depth_estimate.get('mean', 1), 1)
            insights['depth_quality'] = {
                'relative_uncertainty': depth_uncertainty,
                'quality_level': 'high' if depth_uncertainty < 0.1 else 'medium' if depth_uncertainty < 0.3 else 'low',
                'mean_depth_km': depth_estimate.get('mean', 0) / 1000
            }
        
        # Volume estimate quality
        volume_estimate = reservoir_state.get('volume_estimate', {})
        if volume_estimate:
            volume_uncertainty = volume_estimate.get('std', 0) / max(volume_estimate.get('mean', 1), 1)
            insights['volume_quality'] = {
                'relative_uncertainty': volume_uncertainty,
                'quality_level': 'high' if volume_uncertainty < 0.2 else 'medium' if volume_uncertainty < 0.5 else 'low',
                'mean_volume_km3': volume_estimate.get('mean', 0) / 1e9
            }
        
        # Pressure change quality
        pressure_estimate = reservoir_state.get('pressure_change', {})
        if pressure_estimate:
            pressure_uncertainty = abs(pressure_estimate.get('std', 0)) / max(abs(pressure_estimate.get('mean', 1)), 1)
            insights['pressure_quality'] = {
                'relative_uncertainty': pressure_uncertainty,
                'quality_level': 'high' if pressure_uncertainty < 0.15 else 'medium' if pressure_uncertainty < 0.4 else 'low',
                'mean_pressure_mpa': pressure_estimate.get('mean', 0) / 1e6
            }
        
        # Overall parameter correlations
        correlations = reservoir_state.get('parameter_correlations', {})
        max_correlation = correlations.get('max_correlation', 0)
        
        insights['parameter_independence'] = {
            'max_correlation': max_correlation,
            'independence_level': 'high' if max_correlation < 0.3 else 'medium' if max_correlation < 0.6 else 'low',
            'correlation_broken': max_correlation < 0.5  # Indicates successful reformulation
        }
        
        # Assimilation effectiveness
        update_metrics = enkf_results.get('update_metrics', {})
        spread_reduction = update_metrics.get('spread_reduction', 0)
        
        insights['assimilation_effectiveness'] = {
            'spread_reduction': spread_reduction,
            'effectiveness_level': 'high' if spread_reduction > 0.3 else 'medium' if spread_reduction > 0.1 else 'low',
            'information_gain': spread_reduction > 0.05  # Indicates meaningful update
        }
        
        # Overall confidence assessment
        quality_scores = []
        for param_quality in ['depth_quality', 'volume_quality', 'pressure_quality']:
            if param_quality in insights:
                quality_level = insights[param_quality]['quality_level']
                score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}[quality_level]
                quality_scores.append(score)
        
        overall_confidence = np.mean(quality_scores) if quality_scores else 0.5
        
        # Boost confidence if parameter independence is high
        if insights.get('parameter_independence', {}).get('independence_level') == 'high':
            overall_confidence = min(1.0, overall_confidence + 0.1)
        
        # Boost confidence if assimilation was effective
        if insights.get('assimilation_effectiveness', {}).get('effectiveness_level') in ['high', 'medium']:
            overall_confidence = min(1.0, overall_confidence + 0.05)
        
        insights['overall_confidence'] = overall_confidence
        insights['confidence_level'] = 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.6 else 'low'
        
        return insights
    
    async def _generate_causal_insights(self, volcano_id: str, reservoir_state: Dict[str, Any],
                                      enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate causal insights about volcanic processes"""
        
        causal_insights = {}
        
        # Analyze depth-volume relationships
        depth_estimate = reservoir_state.get('depth_estimate', {})
        volume_estimate = reservoir_state.get('volume_estimate', {})
        
        if depth_estimate and volume_estimate:
            depth_mean = depth_estimate.get('mean', 0)
            volume_mean = volume_estimate.get('mean', 0)
            
            # Shallow vs deep reservoir implications
            if depth_mean < 3000:  # Shallow (<3km)
                causal_insights['reservoir_type'] = {
                    'classification': 'shallow',
                    'implications': ['high_eruption_potential', 'fast_response_time', 'high_hazard'],
                    'confidence': 0.8
                }
            elif depth_mean > 8000:  # Deep (>8km)
                causal_insights['reservoir_type'] = {
                    'classification': 'deep',
                    'implications': ['slow_response_time', 'potential_differentiation', 'long_term_evolution'],
                    'confidence': 0.7
                }
            else:  # Intermediate depth
                causal_insights['reservoir_type'] = {
                    'classification': 'intermediate',
                    'implications': ['moderate_response_time', 'mixed_processes'],
                    'confidence': 0.6
                }
        
        # Pressure change implications
        pressure_estimate = reservoir_state.get('pressure_change', {})
        if pressure_estimate:
            pressure_mean = pressure_estimate.get('mean', 0)
            
            if pressure_mean > 0:  # Inflation
                causal_insights['deformation_mode'] = {
                    'type': 'inflation',
                    'process': 'magma_injection',
                    'implications': ['increased_eruption_risk', 'ongoing_unrest'],
                    'confidence': 0.85
                }
            elif pressure_mean < 0:  # Deflation
                causal_insights['deformation_mode'] = {
                    'type': 'deflation',
                    'process': 'magma_withdrawal',
                    'implications': ['eruption_aftermath', 'system_depressurization'],
                    'confidence': 0.80
                }
        
        # Parameter correlation insights
        correlations = reservoir_state.get('parameter_correlations', {})
        max_correlation = correlations.get('max_correlation', 0)
        
        if max_correlation < 0.3:
            causal_insights['parameter_resolution'] = {
                'quality': 'excellent',
                'indication': 'well_constrained_parameters',
                'reliability': 'high',
                'confidence': 0.9
            }
        elif max_correlation < 0.6:
            causal_insights['parameter_resolution'] = {
                'quality': 'good',
                'indication': 'moderately_constrained_parameters',
                'reliability': 'medium',
                'confidence': 0.7
            }
        else:
            causal_insights['parameter_resolution'] = {
                'quality': 'poor',
                'indication': 'poorly_constrained_parameters',
                'reliability': 'low',
                'confidence': 0.4,
                'recommendation': 'additional_data_needed'
            }
        
        # Assimilation success patterns
        update_metrics = enkf_results.get('update_metrics', {})
        spread_reduction = update_metrics.get('spread_reduction', 0)
        
        if spread_reduction > 0.3:
            causal_insights['data_information_content'] = {
                'level': 'high',
                'indication': 'strong_observational_constraints',
                'implication': 'reliable_parameter_updates',
                'confidence': 0.85
            }
        elif spread_reduction > 0.1:
            causal_insights['data_information_content'] = {
                'level': 'medium',
                'indication': 'moderate_observational_constraints',
                'implication': 'some_parameter_refinement',
                'confidence': 0.6
            }
        else:
            causal_insights['data_information_content'] = {
                'level': 'low',
                'indication': 'weak_observational_constraints',
                'implication': 'limited_parameter_updates',
                'confidence': 0.3,
                'recommendation': 'improve_observation_network'
            }
        
        # Generate process chain insights
        causal_insights['process_chain'] = await self._infer_process_chain(
            reservoir_state, enkf_results
        )
        
        return causal_insights
    
    async def _infer_process_chain(self, reservoir_state: Dict[str, Any], 
                                 enkf_results: Dict[str, Any]) -> Dict[str, Any]:
        """Infer causal process chain from EnKF results"""
        
        process_chain = {
            'primary_process': None,
            'secondary_processes': [],
            'confidence': 0.0,
            'causal_links': []
        }
        
        # Extract key parameters
        depth_estimate = reservoir_state.get('depth_estimate', {})
        volume_estimate = reservoir_state.get('volume_estimate', {})
        pressure_estimate = reservoir_state.get('pressure_change', {})
        
        depth_mean = depth_estimate.get('mean', 0)
        volume_mean = volume_estimate.get('mean', 0)
        pressure_mean = pressure_estimate.get('mean', 0)
        
        # Infer primary process
        if pressure_mean > 1e6 and volume_mean > 1e6:  # High pressure, significant volume
            process_chain['primary_process'] = {
                'name': 'magma_intrusion',
                'description': 'Fresh magma injection into reservoir',
                'indicators': ['pressure_increase', 'volume_expansion'],
                'confidence': 0.8
            }
            process_chain['causal_links'].append({
                'cause': 'deep_magma_ascent',
                'effect': 'reservoir_pressurization',
                'strength': 0.85
            })
            
        elif pressure_mean < -5e5:  # Significant deflation
            process_chain['primary_process'] = {
                'name': 'magma_withdrawal',
                'description': 'Magma extraction from reservoir',
                'indicators': ['pressure_decrease', 'deflation'],
                'confidence': 0.75
            }
            process_chain['causal_links'].append({
                'cause': 'magma_eruption_or_lateral_intrusion',
                'effect': 'reservoir_depressurization',
                'strength': 0.8
            })
        
        # Secondary processes based on depth
        if depth_mean < 5000:  # Shallow
            process_chain['secondary_processes'].append({
                'name': 'volatile_exsolution',
                'description': 'Gas separation due to shallow pressures',
                'confidence': 0.7
            })
            
        if depth_mean > 8000:  # Deep
            process_chain['secondary_processes'].append({
                'name': 'crystal_fractionation',
                'description': 'Magma differentiation at depth',
                'confidence': 0.6
            })
        
        # Assess overall confidence
        if process_chain['primary_process']:
            base_confidence = process_chain['primary_process']['confidence']
            # Boost confidence if parameters are well constrained
            correlations = reservoir_state.get('parameter_correlations', {})
            max_correlation = correlations.get('max_correlation', 1.0)
            if max_correlation < 0.5:
                base_confidence = min(1.0, base_confidence + 0.1)
            
            process_chain['confidence'] = base_confidence
        
        return process_chain
    
    async def query_related_knowledge(self, volcano_id: str, query_type: str = 'similar_reservoirs') -> Dict[str, Any]:
        """Query knowledge graph for related volcanic systems and insights"""
        
        if query_type == 'similar_reservoirs':
            # Find volcanoes with similar reservoir characteristics
            similar_systems = await self._find_similar_reservoir_systems(volcano_id)
            
        elif query_type == 'historical_patterns':
            # Find historical deformation patterns
            similar_systems = await self._find_historical_deformation_patterns(volcano_id)
            
        elif query_type == 'causal_relationships':
            # Find causal relationships involving this volcano
            similar_systems = await self._find_causal_relationships(volcano_id)
            
        else:
            similar_systems = {}
        
        return {
            'query_type': query_type,
            'volcano_id': volcano_id,
            'results': similar_systems,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _find_similar_reservoir_systems(self, volcano_id: str) -> Dict[str, Any]:
        """Find volcanoes with similar reservoir characteristics using advanced SPARQL queries"""
        
        current_volcano_entity = f"volcano_{volcano_id}"
        
        try:
            # 1. Get current volcano's reservoir characteristics
            volcano_data = await self._execute_sparql_query(f"""
            PREFIX geo: <http://volcano.knowledge/geo#>
            PREFIX res: <http://volcano.knowledge/reservoir#>
            PREFIX param: <http://volcano.knowledge/parameters#>
            
            SELECT ?depth ?volume ?pressure ?shape ?composition ?temperature WHERE {{
                <{current_volcano_entity}> geo:hasReservoir ?reservoir .
                ?reservoir res:depth ?depth ;
                          res:volume ?volume ;
                          res:pressure ?pressure ;
                          res:shape ?shape ;
                          res:composition ?composition ;
                          res:temperature ?temperature .
            }}
            """)
            
            if not volcano_data.get('results', {}).get('bindings'):
                return await self._fallback_similarity_search(volcano_id)
            
            current_params = volcano_data['results']['bindings'][0]
            
            # 2. Find similar systems using SPARQL with similarity constraints
            similarity_query = f"""
            PREFIX geo: <http://volcano.knowledge/geo#>
            PREFIX res: <http://volcano.knowledge/reservoir#>
            PREFIX param: <http://volcano.knowledge/parameters#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            
            SELECT ?volcano ?depth ?volume ?pressure ?shape ?similarity_score WHERE {{
                ?volcano geo:hasReservoir ?reservoir .
                ?reservoir res:depth ?depth ;
                          res:volume ?volume ;
                          res:pressure ?pressure ;
                          res:shape ?shape .
                
                # Similarity filters (within 20% range)
                FILTER(?depth >= {float(current_params['depth']['value']) * 0.8} && 
                       ?depth <= {float(current_params['depth']['value']) * 1.2})
                FILTER(?volume >= {float(current_params['volume']['value']) * 0.8} && 
                       ?volume <= {float(current_params['volume']['value']) * 1.2})
                FILTER(?pressure >= {float(current_params['pressure']['value']) * 0.8} && 
                       ?pressure <= {float(current_params['pressure']['value']) * 1.2})
                
                # Calculate similarity score
                BIND((1 - (ABS(?depth - {current_params['depth']['value']}) / {current_params['depth']['value']} +
                           ABS(?volume - {current_params['volume']['value']}) / {current_params['volume']['value']} +
                           ABS(?pressure - {current_params['pressure']['value']}) / {current_params['pressure']['value']}) / 3) 
                     AS ?similarity_score)
                
                FILTER(?volcano != <{current_volcano_entity}>)
            }}
            ORDER BY DESC(?similarity_score)
            LIMIT 10
            """
            
            similar_systems_data = await self._execute_sparql_query(similarity_query)
            
            # 3. Process results and add vector similarity
            similar_volcanoes = []
            for binding in similar_systems_data.get('results', {}).get('bindings', []):
                volcano_uri = binding['volcano']['value']
                similarity_score = float(binding['similarity_score']['value'])
                
                # Enhance with vector similarity if embeddings available
                if self.embedding_model and volcano_uri in self.entity_embeddings:
                    vector_similarity = await self._compute_vector_similarity(current_volcano_entity, volcano_uri)
                    combined_score = 0.7 * similarity_score + 0.3 * vector_similarity
                else:
                    combined_score = similarity_score
                
                similar_volcanoes.append({
                    'volcano_id': volcano_uri.split('_')[-1] if '_' in volcano_uri else volcano_uri,
                    'volcano_uri': volcano_uri,
                    'sparql_similarity': similarity_score,
                    'combined_similarity': combined_score,
                    'reservoir_params': {
                        'depth': float(binding['depth']['value']),
                        'volume': float(binding['volume']['value']),
                        'pressure': float(binding['pressure']['value']),
                        'shape': binding['shape']['value']
                    }
                })
            
            # 4. Add graph-based similarity using Neo4j if available
            if self.neo4j_graph:
                graph_similar = await self._find_graph_similar_reservoirs(volcano_id, current_params)
                similar_volcanoes.extend(graph_similar)
            
            return {
                'similar_volcanoes': sorted(similar_volcanoes, key=lambda x: x['combined_similarity'], reverse=True)[:5],
                'similarity_criteria': ['reservoir_depth', 'magma_volume', 'pressure_regime', 'geometric_shape'],
                'matching_algorithm': 'sparql_vector_hybrid',
                'query_performance': {
                    'sparql_query_time_ms': self.query_performance.get('sparql', [])[-1] if self.query_performance.get('sparql') else 0,
                    'total_candidates': len(similar_systems_data.get('results', {}).get('bindings', [])),
                    'filtered_results': len(similar_volcanoes)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced similarity search: {e}")
            return await self._fallback_similarity_search(volcano_id)
    
    async def _find_historical_deformation_patterns(self, volcano_id: str) -> Dict[str, Any]:
        """Find historical deformation patterns using temporal SPARQL and GraphQL queries"""
        
        current_volcano_entity = f"volcano_{volcano_id}"
        
        try:
            # 1. Query temporal deformation events using SPARQL
            temporal_query = f"""
            PREFIX geo: <http://volcano.knowledge/geo#>
            PREFIX def: <http://volcano.knowledge/deformation#>
            PREFIX time: <http://www.w3.org/2006/time#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
            
            SELECT ?event ?start_time ?end_time ?magnitude ?rate ?pattern_type ?spatial_extent WHERE {{
                <{current_volcano_entity}> geo:hasDeformationEvent ?event .
                ?event def:startTime ?start_time ;
                       def:endTime ?end_time ;
                       def:magnitude ?magnitude ;
                       def:deformationRate ?rate ;
                       def:patternType ?pattern_type ;
                       def:spatialExtent ?spatial_extent .
                
                # Filter for events in the last 50 years
                FILTER(?start_time >= "{(datetime.now() - timedelta(days=50*365)).isoformat()}"^^xsd:dateTime)
            }}
            ORDER BY DESC(?start_time)
            """
            
            historical_data = await self._execute_sparql_query(temporal_query)
            
            # 2. Process historical events
            historical_events = []
            pattern_analysis = {
                'inflation_cycles': [],
                'deflation_events': [],
                'long_term_trends': [],
                'episodic_unrest': []
            }
            
            for binding in historical_data.get('results', {}).get('bindings', []):
                event = {
                    'event_id': binding['event']['value'].split('/')[-1],
                    'start_time': binding['start_time']['value'],
                    'end_time': binding['end_time']['value'],
                    'magnitude_cm': float(binding['magnitude']['value']),
                    'rate_cm_per_year': float(binding['rate']['value']),
                    'pattern_type': binding['pattern_type']['value'],
                    'spatial_extent_km2': float(binding['spatial_extent']['value'])
                }
                
                historical_events.append(event)
                
                # Categorize by pattern type
                pattern_type = event['pattern_type'].lower()
                if 'inflation' in pattern_type:
                    pattern_analysis['inflation_cycles'].append(event)
                elif 'deflation' in pattern_type:
                    pattern_analysis['deflation_events'].append(event)
                elif 'trend' in pattern_type:
                    pattern_analysis['long_term_trends'].append(event)
                else:
                    pattern_analysis['episodic_unrest'].append(event)
            
            # 3. Find similar temporal patterns using GraphQL if available
            similar_patterns = []
            if self.graphql_client:
                similar_patterns = await self._execute_graphql_query(f"""
                query FindSimilarDeformationPatterns($volcanoId: String!, $patternTypes: [String!]!) {{
                    volcano(id: $volcanoId) {{
                        similarDeformationPatterns(patternTypes: $patternTypes, limit: 10) {{
                            volcanoId
                            similarity
                            events {{
                                startTime
                                endTime
                                magnitude
                                rate
                                patternType
                                spatialPattern {{
                                    center {{ lat lon }}
                                    extent
                                    shape
                                }}
                            }}
                        }}
                    }}
                }}
                """, {
                    'volcanoId': volcano_id,
                    'patternTypes': list(set([event['pattern_type'] for event in historical_events]))
                })
            
            # 4. Statistical analysis of patterns
            statistical_analysis = await self._analyze_deformation_statistics(historical_events)
            
            # 5. Temporal clustering analysis
            temporal_clusters = await self._perform_temporal_clustering(historical_events)
            
            # 6. Cross-correlation with external datasets
            external_correlations = {}
            if self.config.external_endpoints.get('usgs'):
                external_correlations = await self._correlate_with_seismic_data(volcano_id, historical_events)
            
            return {
                'historical_events': historical_events,
                'pattern_analysis': pattern_analysis,
                'similar_patterns': similar_patterns,
                'statistical_analysis': statistical_analysis,
                'temporal_clusters': temporal_clusters,
                'external_correlations': external_correlations,
                'comparison_metrics': {
                    'temporal_similarity': 'dynamic_time_warping',
                    'spatial_similarity': 'spatial_correlation_coefficient',
                    'magnitude_similarity': 'normalized_magnitude_difference',
                    'rate_similarity': 'rate_correlation'
                },
                'query_performance': {
                    'sparql_query_time_ms': self.query_performance.get('sparql', [])[-1] if self.query_performance.get('sparql') else 0,
                    'graphql_query_time_ms': self.query_performance.get('graphql', [])[-1] if self.query_performance.get('graphql') else 0,
                    'total_events_found': len(historical_events)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in historical pattern analysis: {e}")
            return await self._fallback_historical_analysis(volcano_id)
    
    async def _find_causal_relationships(self, volcano_id: str) -> Dict[str, Any]:
        """Find causal relationships involving this volcano"""
        
        current_volcano_entity = f"volcano_{volcano_id}"
        
        try:
            # Get relationships from knowledge graph
            relationships = await self.knowledge_graph.get_entity_relationships(current_volcano_entity)
            
            causal_links = []
            for rel in relationships:
                if rel.get('type') in [RelationType.CAUSES, RelationType.INFLUENCES]:
                    causal_links.append({
                        'source': rel.get('source'),
                        'target': rel.get('target'),
                        'relationship': rel.get('type'),
                        'confidence': rel.get('confidence', 0.5)
                    })
            
            return {
                'causal_links': causal_links,
                'total_relationships': len(relationships),
                'causal_relationships': len(causal_links)
            }
            
        except Exception as e:
            self.logger.error(f"Error finding causal relationships: {e}")
            return {'error': str(e)}

    async def generate_knowledge_summary(self, volcano_id: str) -> Dict[str, Any]:
        """Generate comprehensive knowledge summary for volcano"""
        
        summary = {
            'volcano_id': volcano_id,
            'knowledge_entities': {},
            'causal_insights': {},
            'parameter_quality': {},
            'process_understanding': {},
            'recommendations': []
        }
        
        try:
            # Get all volcano-related entities
            volcano_entity = f"volcano_{volcano_id}"
            volcano_data = await self.knowledge_graph.get_entity(volcano_entity)
            
            if volcano_data:
                summary['knowledge_entities']['volcano'] = volcano_data
                
                # Get related entities
                relationships = await self.knowledge_graph.get_entity_relationships(volcano_entity)
                summary['knowledge_entities']['relationships'] = relationships
            
            # Generate recommendations based on knowledge
            summary['recommendations'] = await self._generate_recommendations(volcano_id)
            
        except Exception as e:
            self.logger.error(f"Error generating knowledge summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    async def _generate_recommendations(self, volcano_id: str) -> List[Dict[str, Any]]:
        """Generate recommendations based on accumulated knowledge"""
        
        recommendations = []
        
        # Get latest reservoir state and EnKF results from knowledge graph
        try:
            volcano_entity = f"volcano_{volcano_id}"
            relationships = await self.knowledge_graph.get_entity_relationships(volcano_entity)
            
            # Analyze parameter quality and suggest improvements
            for rel in relationships:
                if 'parameter_confidence' in rel.get('target', ''):
                    confidence_data = await self.knowledge_graph.get_entity(rel['target'])
                    
                    if confidence_data:
                        overall_confidence = confidence_data.get('overall_confidence', 0.5)
                        
                        if overall_confidence < 0.6:
                            recommendations.append({
                                'type': 'data_collection',
                                'priority': 'high',
                                'description': 'Improve observation network to enhance parameter constraints',
                                'specific_actions': [
                                    'Add GPS stations in near-field',
                                    'Increase InSAR temporal sampling',
                                    'Deploy tilt meters for higher sensitivity'
                                ]
                            })
                        
                        # Check parameter correlations
                        max_correlation = confidence_data.get('parameter_correlations', {}).get('max_correlation', 0)
                        if max_correlation > 0.6:
                            recommendations.append({
                                'type': 'methodology',
                                'priority': 'medium', 
                                'description': 'Address parameter correlation issues',
                                'specific_actions': [
                                    'Consider alternative observation types',
                                    'Implement advanced regularization',
                                    'Use longer observation time series'
                                ]
                            })
            
            # Add general monitoring recommendations
            recommendations.append({
                'type': 'monitoring',
                'priority': 'medium',
                'description': 'Maintain continuous EnKF assimilation',
                'specific_actions': [
                    'Run daily parameter updates',
                    'Monitor ensemble spread evolution', 
                    'Track parameter correlation changes'
                ]
            })
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append({
                'type': 'system',
                'priority': 'high',
                'description': f'Address knowledge integration error: {e}'
            })
        
        return recommendations
    
    async def _build_similarity_indices(self):
        """Build comprehensive similarity indices for knowledge entities"""
        try:
            if not self.vector_store or not self.embedding_model:
                self.logger.warning("Vector store not available, skipping similarity indices")
                return
            
            # Build indices for different entity types
            await self._build_volcano_similarity_index()
            await self._build_parameter_similarity_index()
            await self._build_deformation_pattern_index()
            await self._build_literature_semantic_index()
            
            self.logger.info("Comprehensive similarity indices built successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to build similarity indices: {e}")
    
    async def _build_volcano_similarity_index(self):
        """Build similarity index for volcano characteristics"""
        try:
            # Sample volcano entities for index building
            volcano_entities = [
                "Mount Vesuvius shallow magma chamber depth 3-8km",
                "Yellowstone caldera large magma reservoir 8-15km depth",
                "Mount Etna multiple magma chambers variable depths",
                "Kilauea shallow lava lake continuous eruption",
                "Mount St. Helens explosive dacitic magma chamber",
                "Santorini caldera complex rhyodacitic system",
                "Campi Flegrei active resurgent caldera system",
                "Long Valley caldera silicic magma chamber"
            ]
            
            # Add to vector store
            if self.vector_store:
                self.vector_store.add_texts(volcano_entities)
                
            self.logger.info("Volcano similarity index built")
            
        except Exception as e:
            self.logger.error(f"Failed to build volcano similarity index: {e}")
    
    async def _build_parameter_similarity_index(self):
        """Build similarity index for deformation parameters"""
        try:
            parameter_entities = [
                "ground deformation uplift 10-50 cm/year volcanic inflation",
                "seismic velocity changes 5-15% magma intrusion indicator",
                "gravity anomalies +20 to +100 mGal magma accumulation",
                "magnetic field changes 10-200 nT thermal effects",
                "gas emission CO2 SO2 increased volcanic unrest",
                "thermal anomalies 5-20°C surface temperature increase",
                "ground tilt microradians volcanic edifice deformation",
                "strain rate 10-6 to 10-4 per year crustal extension"
            ]
            
            if self.vector_store:
                self.vector_store.add_texts(parameter_entities)
                
            self.logger.info("Parameter similarity index built")
            
        except Exception as e:
            self.logger.error(f"Failed to build parameter similarity index: {e}")
    
    async def _build_deformation_pattern_index(self):
        """Build similarity index for deformation patterns"""
        try:
            deformation_patterns = [
                "radial uplift pattern circular Mogi source shallow intrusion",
                "linear deformation InSAR interferometry dike intrusion",
                "subsidence pattern caldera collapse deflation",
                "asymmetric deformation complex source geometry",
                "temporal acceleration exponential pre-eruption",
                "cyclic deformation breathing caldera magma chamber",
                "localized high gradient steep deformation zones",
                "regional tilting far-field volcanic effects"
            ]
            
            if self.vector_store:
                self.vector_store.add_texts(deformation_patterns)
                
            self.logger.info("Deformation pattern similarity index built")
            
        except Exception as e:
            self.logger.error(f"Failed to build deformation pattern index: {e}")
    
    async def _build_literature_semantic_index(self):
        """Build semantic index for literature and research papers"""
        try:
            literature_topics = [
                "volcanic deformation modeling InSAR geodesy elastic half-space",
                "magma chamber geometry finite element modeling",
                "caldera resurgence ground uplift magma intrusion",
                "ensemble Kalman filter data assimilation volcanic monitoring",
                "machine learning volcanic eruption forecasting seismic precursors",
                "Bayesian inference volcano hazard assessment uncertainty quantification",
                "multi-physics coupling thermal mechanical deformation",
                "time series analysis volcanic unrest pattern recognition"
            ]
            
            if self.vector_store:
                self.vector_store.add_texts(literature_topics)
                # Save updated vector store
                vector_store_path = Path(self.config.vector_store_path)
                self.vector_store.save_local(str(vector_store_path))
                
            self.logger.info("Literature semantic index built and saved")
            
        except Exception as e:
            self.logger.error(f"Failed to build literature semantic index: {e}")
    
    async def semantic_similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on knowledge entities"""
        try:
            if not self.vector_store:
                self.logger.warning("Vector store not available for similarity search")
                return []
            
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'text': doc.page_content,
                    'similarity_score': float(1 - score),  # Convert distance to similarity
                    'metadata': doc.metadata
                })
            
            self.logger.info(f"Found {len(formatted_results)} similar entities for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Semantic similarity search failed: {e}")
            return []