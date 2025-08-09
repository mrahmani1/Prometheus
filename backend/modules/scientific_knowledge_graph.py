"""
Scientific Knowledge Graph System for Geological Research
Automated knowledge extraction, semantic reasoning, and federated causal discovery
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import re
import requests
from datetime import datetime, timedelta
import urllib.parse
from pathlib import Path
import hashlib
import sqlite3
from abc import ABC, abstractmethod

# Graph and NLP imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("NetworkX not available. Graph operations will be limited.")

try:
    import spacy
    from spacy import displacy
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy not available. NLP features will be limited.")

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Advanced NLP features will be limited.")

try:
    import rdflib
    from rdflib import Graph, Literal, RDF, URIRef, Namespace
    from rdflib.namespace import FOAF, DC
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    warnings.warn("RDFLib not available. Ontology features will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. Text analysis will be limited.")

# Import causal discovery components
from .causal_inference_engine import CausalGraph, CausalEdge, GeologicalCausalRelationType, CausalDiscoveryEngine
from .hypothesis_generation_system import GeneratedHypothesis, HypothesisGenerator

warnings.filterwarnings('ignore')

class EntityType(Enum):
    CONCEPT = "concept"
    PROCESS = "process"
    LOCATION = "location"
    MINERAL = "mineral"
    ROCK_TYPE = "rock_type"
    MEASUREMENT = "measurement"
    PHENOMENON = "phenomenon"
    RESEARCHER = "researcher"
    PUBLICATION = "publication"
    DATASET = "dataset"
    INSTRUMENT = "instrument"
    METHOD = "method"

class RelationType(Enum):
    PART_OF = "part_of"
    CAUSES = "causes"
    INFLUENCES = "influences"
    OCCURS_IN = "occurs_in"
    MEASURED_BY = "measured_by"
    STUDIED_BY = "studied_by"
    PUBLISHED_IN = "published_in"
    SIMILAR_TO = "similar_to"
    COMPOSED_OF = "composed_of"
    PRECEDES = "precedes"
    CORRELATES_WITH = "correlates_with"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    TEMPORAL_BEFORE = "temporal_before"
    TEMPORAL_AFTER = "temporal_after"
    CAUSAL_TEMPORAL = "causal_temporal"

class DataSourceType(Enum):
    LITERATURE = "literature"
    DATABASE = "database"
    SENSOR_DATA = "sensor_data"
    EXPERIMENTAL = "experimental"
    SIMULATION = "simulation"
    FIELD_OBSERVATION = "field_observation"

@dataclass
class KnowledgeEntity:
    """Entity in the knowledge graph"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_references: List[str] = field(default_factory=list)
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeRelation:
    """Relation between entities in the knowledge graph"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    temporal_context: Optional[Dict[str, Any]] = None
    causal_strength: Optional[float] = None
    statistical_significance: Optional[float] = None
    source_references: List[str] = field(default_factory=list)
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class LiteratureDocument:
    """Document extracted from geological literature"""
    id: str
    title: str
    authors: List[str]
    abstract: str
    full_text: str
    publication_year: int
    journal: str
    doi: Optional[str]
    keywords: List[str]
    extracted_entities: List[KnowledgeEntity] = field(default_factory=list)
    extracted_relations: List[KnowledgeRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TemporalCausalRelation:
    """Temporal causal relationship with time-series evidence"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    temporal_lag: float  # Time lag in specified units
    temporal_units: str  # years, months, days, etc.
    causal_strength: float
    temporal_evidence: pd.DataFrame  # Time-series data
    seasonality: Optional[Dict[str, Any]] = None
    trend_analysis: Optional[Dict[str, Any]] = None
    granger_causality_pvalue: Optional[float] = None
    ccf_analysis: Optional[Dict[str, Any]] = None  # Cross-correlation function
    confidence_intervals: Optional[Tuple[float, float]] = None

class ExternalDataSource(ABC):
    """Abstract base class for external data sources"""
    
    @abstractmethod
    async def query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query the external data source"""
        pass
    
    @abstractmethod
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get ontology mapping for entities"""
        pass

class GeoScienceWorldAPI(ExternalDataSource):
    """Interface to GeoScienceWorld database"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://pubs.geoscienceworld.org/api/v1"
        
    async def query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query GeoScienceWorld database"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "title": f"Geological study {i}",
                "abstract": f"Abstract about {query}",
                "authors": ["Author A", "Author B"],
                "year": 2020 + i % 5,
                "doi": f"10.1000/example.{i}"
            }
            for i in range(min(limit, 10))
        ]
    
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get geological ontology mapping"""
        return {
            "igneous_rock": "http://ontology.geoscienceworld.org/IgneousRock",
            "metamorphic_rock": "http://ontology.geoscienceworld.org/MetamorphicRock",
            "sedimentary_rock": "http://ontology.geoscienceworld.org/SedimentaryRock"
        }

class USGSDataAPI(ExternalDataSource):
    """Interface to USGS data services"""
    
    def __init__(self):
        self.base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
    async def query(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Query USGS earthquake database"""
        # Mock implementation - replace with actual API calls
        return [
            {
                "magnitude": 4.5 + i * 0.1,
                "location": f"Location {i}",
                "depth": 10 + i,
                "time": datetime.now() - timedelta(days=i)
            }
            for i in range(min(limit, 20))
        ]
    
    async def get_ontology_mapping(self) -> Dict[str, str]:
        """Get seismological ontology mapping"""
        return {
            "earthquake": "http://ontology.usgs.gov/Earthquake",
            "magnitude": "http://ontology.usgs.gov/Magnitude",
            "epicenter": "http://ontology.usgs.gov/Epicenter"
        }

class LiteratureExtractor:
    """Extract knowledge from geological literature"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLP models
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
            
        # Initialize transformer model for embeddings
        if TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = pipeline(
                    "feature-extraction",
                    model="sentence-transformers/all-MiniLM-L6-v2",
                    return_tensors="np"
                )
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # Geological term patterns
        self.geological_patterns = self._initialize_geological_patterns()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for literature extraction"""
        return {
            "max_document_length": 50000,
            "entity_confidence_threshold": 0.7,
            "relation_confidence_threshold": 0.6,
            "enable_coreference_resolution": True,
            "enable_temporal_extraction": True,
            "geological_vocabularies": [
                "minerals", "rocks", "processes", "locations", "measurements"
            ],
            "language_models": {
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
                "ner": "en_core_web_sm"
            }
        }
    
    def _initialize_geological_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for geological entity recognition"""
        return {
            "minerals": [
                "quartz", "feldspar", "mica", "calcite", "dolomite", "gypsum", "halite",
                "olivine", "pyroxene", "amphibole", "garnet", "epidote", "chlorite",
                "magnetite", "hematite", "pyrite", "galena", "sphalerite"
            ],
            "rock_types": [
                "granite", "basalt", "andesite", "rhyolite", "gabbro", "peridotite",
                "sandstone", "limestone", "shale", "mudstone", "conglomerate",
                "gneiss", "schist", "slate", "marble", "quartzite", "serpentinite"
            ],
            "processes": [
                "crystallization", "metamorphism", "weathering", "erosion", "deposition",
                "diagenesis", "melting", "solidification", "deformation", "fracturing",
                "faulting", "folding", "uplift", "subsidence", "volcanism", "plutonism"
            ],
            "measurements": [
                "temperature", "pressure", "density", "viscosity", "porosity", "permeability",
                "magnetic susceptibility", "electrical conductivity", "seismic velocity",
                "age", "composition", "concentration", "flow rate", "stress", "strain"
            ],
            "phenomena": [
                "earthquake", "volcanic eruption", "landslide", "tsunami", "flood",
                "mountain building", "seafloor spreading", "subduction", "collision",
                "rifting", "hotspot", "plume", "convection", "differentiation"
            ]
        }
    
    async def extract_from_document(self, document: LiteratureDocument) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """Extract entities and relations from a literature document"""
        
        entities = []
        relations = []
        
        try:
            # Combine title, abstract, and full text for processing
            full_content = f"{document.title}\n\n{document.abstract}\n\n{document.full_text}"
            
            if len(full_content) > self.config["max_document_length"]:
                full_content = full_content[:self.config["max_document_length"]]
            
            # Extract entities
            extracted_entities = await self._extract_entities(full_content, document.id)
            entities.extend(extracted_entities)
            
            # Extract relations
            extracted_relations = await self._extract_relations(
                full_content, extracted_entities, document.id
            )
            relations.extend(extracted_relations)
            
            # Extract temporal relations if enabled
            if self.config["enable_temporal_extraction"]:
                temporal_relations = await self._extract_temporal_relations(
                    full_content, extracted_entities, document.id
                )
                relations.extend(temporal_relations)
            
            # Update document with extracted knowledge
            document.extracted_entities = entities
            document.extracted_relations = relations
            
        except Exception as e:
            self.logger.error(f"Failed to extract knowledge from document {document.id}: {str(e)}")
        
        return entities, relations
    
    async def _extract_entities(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract geological entities from text"""
        
        entities = []
        entity_mentions = {}  # Track unique entities
        
        # Pattern-based extraction
        pattern_entities = self._extract_entities_by_patterns(text, source_id)
        for entity in pattern_entities:
            entity_key = (entity.name.lower(), entity.entity_type)
            if entity_key not in entity_mentions:
                entity_mentions[entity_key] = entity
                entities.append(entity)
        
        # NLP-based extraction if spaCy is available
        if self.nlp:
            nlp_entities = await self._extract_entities_with_nlp(text, source_id)
            for entity in nlp_entities:
                entity_key = (entity.name.lower(), entity.entity_type)
                if entity_key not in entity_mentions:
                    entity_mentions[entity_key] = entity
                    entities.append(entity)
                else:
                    # Merge information
                    existing_entity = entity_mentions[entity_key]
                    existing_entity.confidence = max(existing_entity.confidence, entity.confidence)
                    existing_entity.properties.update(entity.properties)
        
        # Generate embeddings if transformer model is available
        if self.embedding_model:
            for entity in entities:
                try:
                    embedding = self.embedding_model(entity.name)[0][0]  # Get first token embedding
                    entity.embedding_vector = np.mean(embedding, axis=0)  # Average over tokens
                except Exception as e:
                    self.logger.warning(f"Failed to generate embedding for {entity.name}: {str(e)}")
        
        return entities
    
    def _extract_entities_by_patterns(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract entities using pattern matching"""
        
        entities = []
        text_lower = text.lower()
        
        for category, terms in self.geological_patterns.items():
            for term in terms:
                # Find all occurrences of the term
                pattern = r'\b' + re.escape(term.lower()) + r'\b'
                matches = re.finditer(pattern, text_lower)
                
                for match in matches:
                    # Get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Determine entity type based on category
                    if category == "minerals":
                        entity_type = EntityType.MINERAL
                    elif category == "rock_types":
                        entity_type = EntityType.ROCK_TYPE
                    elif category == "processes":
                        entity_type = EntityType.PROCESS
                    elif category == "measurements":
                        entity_type = EntityType.MEASUREMENT
                    elif category == "phenomena":
                        entity_type = EntityType.PHENOMENON
                    else:
                        entity_type = EntityType.CONCEPT
                    
                    # Create entity
                    entity_id = hashlib.md5(f"{term}_{entity_type.value}_{source_id}".encode()).hexdigest()[:16]
                    
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=term,
                        entity_type=entity_type,
                        properties={
                            "category": category,
                            "context": context,
                            "mention_count": 1
                        },
                        confidence=0.8,  # Pattern-based confidence
                        source_references=[source_id]
                    )
                    
                    entities.append(entity)
                    break  # Only create one entity per term per document
        
        return entities
    
    async def _extract_entities_with_nlp(self, text: str, source_id: str) -> List[KnowledgeEntity]:
        """Extract entities using NLP (spaCy)"""
        
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                entity_type = self._map_spacy_entity_type(ent.label_)
                
                if entity_type:  # Only include relevant geological entities
                    entity_id = hashlib.md5(f"{ent.text}_{entity_type.value}_{source_id}".encode()).hexdigest()[:16]
                    
                    # Get surrounding context
                    start_token = max(0, ent.start - 10)
                    end_token = min(len(doc), ent.end + 10)
                    context = doc[start_token:end_token].text
                    
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=ent.text,
                        entity_type=entity_type,
                        properties={
                            "spacy_label": ent.label_,
                            "context": context,
                            "start_char": ent.start_char,
                            "end_char": ent.end_char
                        },
                        confidence=self._calculate_entity_confidence(ent),
                        source_references=[source_id]
                    )
                    
                    entities.append(entity)
            
            # Extract additional entities using custom patterns
            custom_entities = self._extract_custom_geological_entities(doc, source_id)
            entities.extend(custom_entities)
            
        except Exception as e:
            self.logger.warning(f"NLP entity extraction failed: {str(e)}")
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to geological entity types"""
        
        mapping = {
            "PERSON": EntityType.RESEARCHER,
            "ORG": EntityType.PUBLICATION,  # Journals, institutions
            "GPE": EntityType.LOCATION,      # Countries, cities, states
            "LOC": EntityType.LOCATION,      # Locations, mountain ranges, etc.
            "PRODUCT": EntityType.INSTRUMENT, # Scientific instruments
            "EVENT": EntityType.PHENOMENON,  # Geological events
            "FAC": EntityType.LOCATION,      # Facilities, landmarks
        }
        
        return mapping.get(spacy_label)
    
    def _calculate_entity_confidence(self, spacy_entity) -> float:
        """Calculate confidence score for spaCy entity"""
        
        base_confidence = 0.7
        
        # Increase confidence for longer entities
        if len(spacy_entity.text) > 10:
            base_confidence += 0.1
        
        # Increase confidence for capitalized entities
        if spacy_entity.text.istitle():
            base_confidence += 0.1
        
        # Check if entity matches known geological terms
        entity_lower = spacy_entity.text.lower()
        for category, terms in self.geological_patterns.items():
            if any(term in entity_lower for term in terms):
                base_confidence += 0.2
                break
        
        return min(1.0, base_confidence)
    
    def _extract_custom_geological_entities(self, doc, source_id: str) -> List[KnowledgeEntity]:
        """Extract geological entities using custom rules"""
        
        entities = []
        
        # Look for measurement patterns (number + unit)
        measurement_pattern = r'(\d+\.?\d*)\s*(°C|°F|K|Pa|MPa|GPa|m/s|km/s|g/cm³|%|ppm|Ma|Ga)'
        
        for match in re.finditer(measurement_pattern, doc.text):
            value = match.group(1)
            unit = match.group(2)
            
            entity_id = hashlib.md5(f"measurement_{value}_{unit}_{source_id}".encode()).hexdigest()[:16]
            
            entity = KnowledgeEntity(
                id=entity_id,
                name=f"{value} {unit}",
                entity_type=EntityType.MEASUREMENT,
                properties={
                    "value": float(value),
                    "unit": unit,
                    "measurement_type": self._classify_measurement_unit(unit)
                },
                confidence=0.9,
                source_references=[source_id]
            )
            
            entities.append(entity)
        
        return entities
    
    def _classify_measurement_unit(self, unit: str) -> str:
        """Classify measurement unit type"""
        
        unit_types = {
            "temperature": ["°C", "°F", "K"],
            "pressure": ["Pa", "MPa", "GPa", "bar"],
            "velocity": ["m/s", "km/s", "cm/s"],
            "density": ["g/cm³", "kg/m³"],
            "concentration": ["%", "ppm", "ppb"],
            "time": ["Ma", "Ga", "ka", "years"],
            "length": ["m", "km", "cm", "mm"]
        }
        
        for unit_type, units in unit_types.items():
            if unit in units:
                return unit_type
        
        return "unknown"
    
    async def _extract_relations(self, text: str, entities: List[KnowledgeEntity], 
                               source_id: str) -> List[KnowledgeRelation]:
        """Extract relations between entities"""
        
        relations = []
        
        # Pattern-based relation extraction
        pattern_relations = self._extract_relations_by_patterns(text, entities, source_id)
        relations.extend(pattern_relations)
        
        # Dependency parsing-based extraction if spaCy is available
        if self.nlp:
            dependency_relations = await self._extract_relations_with_dependencies(
                text, entities, source_id
            )
            relations.extend(dependency_relations)
        
        return relations
    
    def _extract_relations_by_patterns(self, text: str, entities: List[KnowledgeEntity],
                                     source_id: str) -> List[KnowledgeRelation]:
        """Extract relations using linguistic patterns"""
        
        relations = []
        text_lower = text.lower()
        
        # Create entity name mapping
        entity_names = {entity.name.lower(): entity for entity in entities}
        
        # Causal relation patterns
        causal_patterns = [
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+leads?\s+to\s+(\w+)',
            r'(\w+)\s+results?\s+in\s+(\w+)',
            r'(\w+)\s+triggers?\s+(\w+)',
            r'(\w+)\s+induces?\s+(\w+)'
        ]
        
        for pattern in causal_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                source_term = match.group(1)
                target_term = match.group(2)
                
                # Find matching entities
                source_entity = self._find_matching_entity(source_term, entity_names)
                target_entity = self._find_matching_entity(target_term, entity_names)
                
                if source_entity and target_entity:
                    relation_id = hashlib.md5(
                        f"{source_entity.id}_{target_entity.id}_causes_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relation_type=RelationType.CAUSES,
                        properties={
                            "pattern_matched": pattern,
                            "context": match.group(0)
                        },
                        confidence=0.7,
                        evidence=[match.group(0)],
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        # Compositional patterns
        compositional_patterns = [
            r'(\w+)\s+contains?\s+(\w+)',
            r'(\w+)\s+composed\s+of\s+(\w+)',
            r'(\w+)\s+made\s+of\s+(\w+)',
            r'(\w+)\s+consists?\s+of\s+(\w+)'
        ]
        
        for pattern in compositional_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                container = match.group(1)
                component = match.group(2)
                
                container_entity = self._find_matching_entity(container, entity_names)
                component_entity = self._find_matching_entity(component, entity_names)
                
                if container_entity and component_entity:
                    relation_id = hashlib.md5(
                        f"{container_entity.id}_{component_entity.id}_composed_of_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=container_entity.id,
                        target_entity_id=component_entity.id,
                        relation_type=RelationType.COMPOSED_OF,
                        confidence=0.8,
                        evidence=[match.group(0)],
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        return relations
    
    def _find_matching_entity(self, term: str, entity_mapping: Dict[str, KnowledgeEntity]) -> Optional[KnowledgeEntity]:
        """Find matching entity for a term"""
        
        term_lower = term.lower()
        
        # Exact match
        if term_lower in entity_mapping:
            return entity_mapping[term_lower]
        
        # Partial match
        for entity_name, entity in entity_mapping.items():
            if term_lower in entity_name or entity_name in term_lower:
                return entity
        
        return None
    
    async def _extract_relations_with_dependencies(self, text: str, entities: List[KnowledgeEntity],
                                                 source_id: str) -> List[KnowledgeRelation]:
        """Extract relations using dependency parsing"""
        
        relations = []
        
        if not self.nlp:
            return relations
        
        try:
            doc = self.nlp(text)
            entity_spans = {entity.name.lower(): entity for entity in entities}
            
            # Process each sentence
            for sent in doc.sents:
                sentence_relations = self._extract_sentence_relations(
                    sent, entity_spans, source_id
                )
                relations.extend(sentence_relations)
                
        except Exception as e:
            self.logger.warning(f"Dependency parsing failed: {str(e)}")
        
        return relations
    
    def _extract_sentence_relations(self, sentence, entity_spans: Dict[str, KnowledgeEntity],
                                  source_id: str) -> List[KnowledgeRelation]:
        """Extract relations from a single sentence using dependency parsing"""
        
        relations = []
        
        # Find verbs that might indicate relationships
        relationship_verbs = {
            "cause", "causes", "caused": RelationType.CAUSES,
            "influence", "influences", "influenced": RelationType.INFLUENCES,
            "occur", "occurs", "occurred": RelationType.OCCURS_IN,
            "contain", "contains", "contained": RelationType.COMPOSED_OF,
            "precede", "precedes", "preceded": RelationType.PRECEDES,
            "correlate", "correlates", "correlated": RelationType.CORRELATES_WITH
        }
        
        for token in sentence:
            if token.lemma_ in relationship_verbs:
                relation_type = relationship_verbs[token.lemma_]
                
                # Find subject and object
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child
                
                if subject and obj:
                    # Try to match with extracted entities
                    subject_entity = self._find_matching_entity(subject.text.lower(), entity_spans)
                    object_entity = self._find_matching_entity(obj.text.lower(), entity_spans)
                    
                    if subject_entity and object_entity:
                        relation_id = hashlib.md5(
                            f"{subject_entity.id}_{object_entity.id}_{relation_type.value}_{source_id}".encode()
                        ).hexdigest()[:16]
                        
                        relation = KnowledgeRelation(
                            id=relation_id,
                            source_entity_id=subject_entity.id,
                            target_entity_id=object_entity.id,
                            relation_type=relation_type,
                            properties={
                                "verb": token.lemma_,
                                "dependency_pattern": f"{subject.dep_}-{token.lemma_}-{obj.dep_}"
                            },
                            confidence=0.6,
                            evidence=[sentence.text],
                            source_references=[source_id]
                        )
                        
                        relations.append(relation)
        
        return relations
    
    async def _extract_temporal_relations(self, text: str, entities: List[KnowledgeEntity],
                                        source_id: str) -> List[KnowledgeRelation]:
        """Extract temporal relations from text"""
        
        relations = []
        
        # Temporal patterns
        temporal_patterns = [
            r'(\w+)\s+before\s+(\w+)',
            r'(\w+)\s+after\s+(\w+)',
            r'(\w+)\s+during\s+(\w+)',
            r'(\w+)\s+precedes?\s+(\w+)',
            r'(\w+)\s+follows?\s+(\w+)'
        ]
        
        entity_names = {entity.name.lower(): entity for entity in entities}
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                first_term = match.group(1)
                second_term = match.group(2)
                
                first_entity = self._find_matching_entity(first_term, entity_names)
                second_entity = self._find_matching_entity(second_term, entity_names)
                
                if first_entity and second_entity:
                    # Determine relation type based on pattern
                    if "before" in pattern or "precedes" in pattern:
                        relation_type = RelationType.TEMPORAL_BEFORE
                        source_entity = first_entity
                        target_entity = second_entity
                    elif "after" in pattern or "follows" in pattern:
                        relation_type = RelationType.TEMPORAL_AFTER
                        source_entity = second_entity
                        target_entity = first_entity
                    else:
                        relation_type = RelationType.TEMPORAL_BEFORE
                        source_entity = first_entity
                        target_entity = second_entity
                    
                    relation_id = hashlib.md5(
                        f"{source_entity.id}_{target_entity.id}_{relation_type.value}_{source_id}".encode()
                    ).hexdigest()[:16]
                    
                    relation = KnowledgeRelation(
                        id=relation_id,
                        source_entity_id=source_entity.id,
                        target_entity_id=target_entity.id,
                        relation_type=relation_type,
                        properties={
                            "temporal_pattern": pattern,
                            "context": match.group(0)
                        },
                        confidence=0.6,
                        evidence=[match.group(0)],
                        temporal_context={
                            "extraction_method": "pattern_matching",
                            "temporal_indicator": "before/after"
                        },
                        source_references=[source_id]
                    )
                    
                    relations.append(relation)
        
        return relations

class TemporalCausalAnalyzer:
    """Analyze temporal causal relationships in time-series data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "max_lag": 12,  # Maximum lag periods to test
            "significance_threshold": 0.05,
            "min_observations": 30,
            "detrend_method": "linear",
            "seasonal_decomposition": True,
            "granger_test": True,
            "ccf_analysis": True,
            "bootstrap_samples": 1000
        }
    
    async def analyze_temporal_causality(self, source_series: pd.Series, 
                                       target_series: pd.Series,
                                       entity_ids: Tuple[str, str]) -> Optional[TemporalCausalRelation]:
        """Analyze temporal causal relationship between two time series"""
        
        if len(source_series) < self.config["min_observations"]:
            self.logger.warning("Insufficient observations for temporal causal analysis")
            return None
        
        try:
            # Align time series
            aligned_data = pd.concat([source_series, target_series], axis=1, join='inner')
            if len(aligned_data) < self.config["min_observations"]:
                return None
            
            source_col = aligned_data.columns[0]
            target_col = aligned_data.columns[1]
            
            # Prepare data
            prepared_data = self._prepare_time_series_data(aligned_data)
            
            # Cross-correlation analysis
            ccf_results = None
            if self.config["ccf_analysis"]:
                ccf_results = self._cross_correlation_analysis(
                    prepared_data[source_col], prepared_data[target_col]
                )
            
            # Granger causality test
            granger_pvalue = None
            optimal_lag = 1
            if self.config["granger_test"]:
                granger_results = self._granger_causality_test(
                    prepared_data[source_col], prepared_data[target_col]
                )
                granger_pvalue = granger_results["p_value"]
                optimal_lag = granger_results["optimal_lag"]
            
            # Calculate causal strength
            causal_strength = self._calculate_causal_strength(
                prepared_data[source_col], prepared_data[target_col], optimal_lag
            )
            
            # Seasonality analysis
            seasonality = None
            if self.config["seasonal_decomposition"]:
                seasonality = self._analyze_seasonality(prepared_data)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(prepared_data)
            
            # Bootstrap confidence intervals
            confidence_intervals = await self._bootstrap_confidence_intervals(
                prepared_data[source_col], prepared_data[target_col], optimal_lag
            )
            
            # Create temporal causal relation
            relation_id = hashlib.md5(
                f"temporal_{entity_ids[0]}_{entity_ids[1]}".encode()
            ).hexdigest()[:16]
            
            temporal_relation = TemporalCausalRelation(
                relation_id=relation_id,
                source_entity_id=entity_ids[0],
                target_entity_id=entity_ids[1],
                temporal_lag=optimal_lag,
                temporal_units="periods",  # Adjust based on data frequency
                causal_strength=causal_strength,
                temporal_evidence=prepared_data,
                seasonality=seasonality,
                trend_analysis=trend_analysis,
                granger_causality_pvalue=granger_pvalue,
                ccf_analysis=ccf_results,
                confidence_intervals=confidence_intervals
            )
            
            return temporal_relation
            
        except Exception as e:
            self.logger.error(f"Temporal causal analysis failed: {str(e)}")
            return None
    
    def _prepare_time_series_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare time series data for analysis"""
        
        prepared = data.copy()
        
        # Handle missing values
        prepared = prepared.fillna(method='ffill').fillna(method='bfill')
        
        # Detrend if specified
        if self.config["detrend_method"] == "linear":
            for col in prepared.columns:
                if prepared[col].dtype in [np.float64, np.int64]:
                    # Linear detrending
                    x = np.arange(len(prepared))
                    coeffs = np.polyfit(x, prepared[col], 1)
                    trend = np.polyval(coeffs, x)
                    prepared[col] = prepared[col] - trend
        
        return prepared
    
    def _cross_correlation_analysis(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-correlation analysis"""
        
        try:
            # Calculate cross-correlation
            max_lag = min(self.config["max_lag"], len(x) // 4)
            
            correlations = []
            lags = []
            
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    corr = x.corr(y)
                elif lag > 0:
                    # y leads x by lag periods
                    if len(x) > lag:
                        corr = x[lag:].corr(y[:-lag])
                    else:
                        corr = np.nan
                else:
                    # x leads y by |lag| periods
                    abs_lag = abs(lag)
                    if len(y) > abs_lag:
                        corr = x[:-abs_lag].corr(y[abs_lag:])
                    else:
                        corr = np.nan
                
                if not np.isnan(corr):
                    correlations.append(corr)
                    lags.append(lag)
            
            # Find optimal lag
            max_corr_idx = np.argmax(np.abs(correlations))
            optimal_lag = lags[max_corr_idx]
            max_correlation = correlations[max_corr_idx]
            
            return {
                "correlations": correlations,
                "lags": lags,
                "optimal_lag": optimal_lag,
                "max_correlation": max_correlation,
                "significant_lags": [lag for i, lag in enumerate(lags) 
                                   if abs(correlations[i]) > 2/np.sqrt(len(x))]
            }
            
        except Exception as e:
            self.logger.warning(f"Cross-correlation analysis failed: {str(e)}")
            return {"optimal_lag": 1, "max_correlation": 0.0}
    
    def _granger_causality_test(self, x: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Perform Granger causality test"""
        
        try:
            # Simple implementation - in practice, use statsmodels
            from scipy import stats
            
            best_lag = 1
            best_pvalue = 1.0
            
            for lag in range(1, min(self.config["max_lag"], len(x) // 4)):
                try:
                    # Create lagged variables
                    y_lagged = y[lag:].values
                    x_lagged = x[:-lag].values
                    y_current = y[lag:].values
                    
                    if len(y_lagged) < 10:
                        continue
                    
                    # Restricted model: y ~ y_lagged
                    y_lag_matrix = np.column_stack([y_lagged[:-1]])
                    y_target = y_current[1:]
                    
                    # Unrestricted model: y ~ y_lagged + x_lagged
                    xy_lag_matrix = np.column_stack([y_lagged[:-1], x_lagged[:-1]])
                    
                    # Calculate RSS for both models
                    try:
                        from sklearn.linear_model import LinearRegression
                        
                        # Restricted model
                        reg_restricted = LinearRegression().fit(y_lag_matrix, y_target)
                        y_pred_restricted = reg_restricted.predict(y_lag_matrix)
                        rss_restricted = np.sum((y_target - y_pred_restricted) ** 2)
                        
                        # Unrestricted model  
                        reg_unrestricted = LinearRegression().fit(xy_lag_matrix, y_target)
                        y_pred_unrestricted = reg_unrestricted.predict(xy_lag_matrix)
                        rss_unrestricted = np.sum((y_target - y_pred_unrestricted) ** 2)
                        
                        # F-statistic
                        n = len(y_target)
                        k = xy_lag_matrix.shape[1]
                        
                        f_stat = ((rss_restricted - rss_unrestricted) / 1) / (rss_unrestricted / (n - k - 1))
                        p_value = 1 - stats.f.cdf(f_stat, 1, n - k - 1)
                        
                        if p_value < best_pvalue:
                            best_pvalue = p_value
                            best_lag = lag
                            
                    except Exception:
                        continue
                        
                except Exception:
                    continue
            
            return {
                "optimal_lag": best_lag,
                "p_value": best_pvalue,
                "is_significant": best_pvalue < self.config["significance_threshold"]
            }
            
        except Exception as e:
            self.logger.warning(f"Granger causality test failed: {str(e)}")
            return {"optimal_lag": 1, "p_value": 1.0, "is_significant": False}
    
    def _calculate_causal_strength(self, x: pd.Series, y: pd.Series, lag: int) -> float:
        """Calculate causal strength based on predictive improvement"""
        
        try:
            if lag >= len(x) or lag <= 0:
                return 0.0
            
            # Create lagged variables
            y_current = y[lag:].values
            y_lagged = y[:-lag].values
            x_lagged = x[:-lag].values
            
            if len(y_current) < 10:
                return 0.0
            
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Baseline model: y ~ y_lagged
            reg_baseline = LinearRegression().fit(y_lagged.reshape(-1, 1), y_current)
            y_pred_baseline = reg_baseline.predict(y_lagged.reshape(-1, 1))
            r2_baseline = r2_score(y_current, y_pred_baseline)
            
            # Enhanced model: y ~ y_lagged + x_lagged
            X_enhanced = np.column_stack([y_lagged, x_lagged])
            reg_enhanced = LinearRegression().fit(X_enhanced, y_current)
            y_pred_enhanced = reg_enhanced.predict(X_enhanced)
            r2_enhanced = r2_score(y_current, y_pred_enhanced)
            
            # Causal strength as improvement in R²
            causal_strength = max(0.0, r2_enhanced - r2_baseline)
            
            return causal_strength
            
        except Exception as e:
            self.logger.warning(f"Causal strength calculation failed: {str(e)}")
            return 0.0
    
    def _analyze_seasonality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonality in time series"""
        
        try:
            seasonality_results = {}
            
            for col in data.columns:
                if data[col].dtype in [np.float64, np.int64]:
                    # Simple seasonality detection using FFT
                    series = data[col].dropna()
                    
                    if len(series) < 24:  # Need enough data points
                        continue
                    
                    # FFT to detect dominant frequencies
                    fft = np.fft.fft(series.values)
                    freqs = np.fft.fftfreq(len(series))
                    
                    # Find dominant frequencies (excluding DC component)
                    power = np.abs(fft[1:len(fft)//2])
                    freqs_positive = freqs[1:len(freqs)//2]
                    
                    if len(power) > 0:
                        dominant_freq_idx = np.argmax(power)
                        dominant_freq = freqs_positive[dominant_freq_idx]
                        dominant_period = 1 / abs(dominant_freq) if dominant_freq != 0 else np.inf
                        
                        seasonality_results[col] = {
                            "dominant_frequency": dominant_freq,
                            "dominant_period": dominant_period,
                            "seasonal_strength": power[dominant_freq_idx] / np.sum(power)
                        }
            
            return seasonality_results
            
        except Exception as e:
            self.logger.warning(f"Seasonality analysis failed: {str(e)}")
            return {}
    
    def _analyze_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trends in time series"""
        
        trend_results = {}
        
        for col in data.columns:
            if data[col].dtype in [np.float64, np.int64]:
                series = data[col].dropna()
                
                if len(series) < 10:
                    continue
                
                # Linear trend
                x = np.arange(len(series))
                coeffs = np.polyfit(x, series.values, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # Trend significance test
                from scipy import stats
                corr, p_value = stats.pearsonr(x, series.values)
                
                trend_results[col] = {
                    "linear_slope": slope,
                    "linear_intercept": intercept,
                    "trend_correlation": corr,
                    "trend_p_value": p_value,
                    "trend_significant": p_value < 0.05,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
                }
        
        return trend_results
    
    async def _bootstrap_confidence_intervals(self, x: pd.Series, y: pd.Series, 
                                            lag: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate bootstrap confidence intervals for causal strength"""
        
        try:
            n_bootstrap = self.config["bootstrap_samples"]
            bootstrap_strengths = []
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(x), len(x), replace=True)
                x_boot = x.iloc[indices].reset_index(drop=True)
                y_boot = y.iloc[indices].reset_index(drop=True)
                
                # Calculate causal strength
                strength = self._calculate_causal_strength(x_boot, y_boot, lag)
                bootstrap_strengths.append(strength)
            
            # Calculate confidence interval
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            ci_lower = np.percentile(bootstrap_strengths, lower_percentile)
            ci_upper = np.percentile(bootstrap_strengths, upper_percentile)
            
            return (ci_lower, ci_upper)
            
        except Exception as e:
            self.logger.warning(f"Bootstrap confidence interval calculation failed: {str(e)}")
            return (0.0, 1.0)

class FederatedCausalDiscovery:
    """Federated causal relationship discovery across multiple datasets"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        self.registered_datasets = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "min_datasets": 2,
            "consistency_threshold": 0.6,
            "meta_analysis_method": "random_effects",
            "heterogeneity_threshold": 0.7,
            "privacy_preserving": True,
            "differential_privacy_epsilon": 1.0,
            "secure_aggregation": False
        }
    
    def register_dataset(self, dataset_id: str, data: pd.DataFrame, 
                        metadata: Dict[str, Any] = None):
        """Register a dataset for federated analysis"""
        
        self.registered_datasets[dataset_id] = {
            "data": data,
            "metadata": metadata or {},
            "local_results": None,
            "summary_statistics": self._calculate_summary_statistics(data)
        }
        
        self.logger.info(f"Registered dataset {dataset_id} with {len(data)} observations")
    
    def _calculate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate privacy-preserving summary statistics"""
        
        stats = {}
        
        for col in data.select_dtypes(include=[np.number]).columns:
            series = data[col].dropna()
            
            if len(series) > 0:
                # Add noise for differential privacy if enabled
                if self.config["privacy_preserving"]:
                    epsilon = self.config["differential_privacy_epsilon"]
                    noise_scale = 1.0 / epsilon
                    
                    stats[col] = {
                        "mean": series.mean() + np.random.laplace(0, noise_scale),
                        "std": series.std() + np.random.laplace(0, noise_scale),
                        "count": len(series),
                        "min": series.min() + np.random.laplace(0, noise_scale),
                        "max": series.max() + np.random.laplace(0, noise_scale)
                    }
                else:
                    stats[col] = {
                        "mean": series.mean(),
                        "std": series.std(),
                        "count": len(series),
                        "min": series.min(),
                        "max": series.max()
                    }
        
        return stats
    
    async def discover_federated_causal_relationships(self, 
                                                    target_variables: List[str] = None) -> Dict[str, Any]:
        """Discover causal relationships across federated datasets"""
        
        if len(self.registered_datasets) < self.config["min_datasets"]:
            raise ValueError(f"Need at least {self.config['min_datasets']} datasets for federated analysis")
        
        # Step 1: Local causal discovery on each dataset
        local_results = await self._perform_local_causal_discovery(target_variables)
        
        # Step 2: Aggregate local results
        federated_results = await self._aggregate_local_results(local_results)
        
        # Step 3: Meta-analysis of causal relationships
        meta_analysis_results = await self._perform_meta_analysis(local_results)
        
        # Step 4: Consistency analysis across datasets
        consistency_analysis = await self._analyze_consistency(local_results)
        
        return {
            "local_results": local_results,
            "federated_causal_graph": federated_results,
            "meta_analysis": meta_analysis_results,
            "consistency_analysis": consistency_analysis,
            "summary": self._generate_federated_summary(
                federated_results, meta_analysis_results, consistency_analysis
            )
        }
    
    async def _perform_local_causal_discovery(self, target_variables: List[str] = None) -> Dict[str, Any]:
        """Perform causal discovery on each local dataset"""
        
        local_results = {}
        
        for dataset_id, dataset_info in self.registered_datasets.items():
            try:
                self.logger.info(f"Performing local causal discovery on dataset {dataset_id}")
                
                data = dataset_info["data"]
                
                # Filter target variables if specified
                if target_variables:
                    available_vars = [var for var in target_variables if var in data.columns]
                    if len(available_vars) < 2:
                        self.logger.warning(f"Dataset {dataset_id} has insufficient target variables")
                        continue
                    data = data[available_vars]
                
                # Initialize causal discovery engine
                causal_engine = CausalDiscoveryEngine()
                
                # Discover causal relationships
                causal_graphs = await causal_engine.discover_causal_structure(data)
                
                # Extract summary information for federation
                local_summary = self._extract_local_summary(causal_graphs, data)
                
                local_results[dataset_id] = {
                    "causal_graphs": causal_graphs,
                    "summary": local_summary,
                    "dataset_metadata": dataset_info["metadata"],
                    "n_observations": len(data),
                    "n_variables": len(data.columns)
                }
                
                # Store in dataset info
                dataset_info["local_results"] = local_results[dataset_id]
                
            except Exception as e:
                self.logger.error(f"Local causal discovery failed for dataset {dataset_id}: {str(e)}")
                continue
        
        return local_results
    
    def _extract_local_summary(self, causal_graphs: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract summary information from local causal discovery"""
        
        summary = {
            "discovered_relationships": [],
            "variable_statistics": {},
            "graph_properties": {}
        }
        
        # Extract relationships from all discovered graphs
        all_edges = []
        for algorithm, graph in causal_graphs.items():
            for edge in graph.edges:
                edge_info = {
                    "source": edge.source,
                    "target": edge.target,
                    "causal_type": edge.causal_type.value,
                    "strength": edge.strength,
                    "p_value": edge.p_value,
                    "effect_size": edge.effect_size,
                    "algorithm": algorithm.value
                }
                all_edges.append(edge_info)
        
        summary["discovered_relationships"] = all_edges
        
        # Variable statistics
        for col in data.select_dtypes(include=[np.number]).columns:
            summary["variable_statistics"][col] = {
                "mean": data[col].mean(),
                "std": data[col].std(),
                "n_obs": data[col].count()
            }
        
        # Graph properties
        if causal_graphs:
            first_graph = list(causal_graphs.values())[0]
            summary["graph_properties"] = {
                "n_nodes": len(first_graph.nodes),
                "n_edges": len(first_graph.edges),
                "edge_density": len(first_graph.edges) / max(1, len(first_graph.nodes) * (len(first_graph.nodes) - 1))
            }
        
        return summary
    
    async def _aggregate_local_results(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate local causal discovery results"""
        
        # Collect all discovered relationships
        all_relationships = []
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                relationship_with_source = relationship.copy()
                relationship_with_source["dataset_id"] = dataset_id
                relationship_with_source["n_observations"] = results["n_observations"]
                all_relationships.append(relationship_with_source)
        
        # Group relationships by source-target pair
        relationship_groups = defaultdict(list)
        
        for rel in all_relationships:
            key = (rel["source"], rel["target"])
            relationship_groups[key].append(rel)
        
        # Aggregate each relationship group
        federated_relationships = []
        
        for (source, target), relationships in relationship_groups.items():
            if len(relationships) >= self.config["min_datasets"]:
                aggregated_rel = self._aggregate_relationship_group(relationships)
                if aggregated_rel:
                    federated_relationships.append(aggregated_rel)
        
        # Build federated causal graph
        federated_graph = {
            "relationships": federated_relationships,
            "consensus_threshold": self.config["consistency_threshold"],
            "participating_datasets": list(local_results.keys()),
            "aggregation_method": "weighted_average"
        }
        
        return federated_graph
    
    def _aggregate_relationship_group(self, relationships: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Aggregate a group of relationships from different datasets"""
        
        if not relationships:
            return None
        
        # Extract values
        strengths = [rel["strength"] for rel in relationships]
        p_values = [rel["p_value"] for rel in relationships]
        effect_sizes = [rel["effect_size"] for rel in relationships]
        n_observations = [rel["n_observations"] for rel in relationships]
        
        # Weighted aggregation (weight by sample size)
        total_n = sum(n_observations)
        weights = [n / total_n for n in n_observations]
        
        aggregated_strength = np.average(strengths, weights=weights)
        aggregated_effect_size = np.average(effect_sizes, weights=weights)
        
        # Meta-analysis of p-values using Fisher's method
        from scipy import stats
        
        try:
            chi2_stat, meta_p_value = stats.combine_pvalues(p_values, method='fisher')
        except:
            meta_p_value = np.mean(p_values)
        
        # Calculate heterogeneity
        heterogeneity = self._calculate_heterogeneity(strengths, weights)
        
        # Only include if heterogeneity is acceptable
        if heterogeneity > self.config["heterogeneity_threshold"]:
            return None
        
        aggregated_relationship = {
            "source": relationships[0]["source"],
            "target": relationships[0]["target"],
            "aggregated_strength": aggregated_strength,
            "aggregated_effect_size": aggregated_effect_size,
            "meta_p_value": meta_p_value,
            "heterogeneity": heterogeneity,
            "n_datasets": len(relationships),
            "total_observations": total_n,
            "individual_results": relationships
        }
        
        return aggregated_relationship
    
    def _calculate_heterogeneity(self, values: List[float], weights: List[float]) -> float:
        """Calculate heterogeneity measure (I² statistic)"""
        
        try:
            # Calculate weighted mean
            weighted_mean = np.average(values, weights=weights)
            
            # Calculate Q statistic
            q_stat = sum(w * (v - weighted_mean)**2 for v, w in zip(values, weights))
            
            # Degrees of freedom
            df = len(values) - 1
            
            # I² statistic
            if df > 0 and q_stat > df:
                i_squared = (q_stat - df) / q_stat
            else:
                i_squared = 0.0
            
            return i_squared
            
        except Exception:
            return 1.0  # High heterogeneity if calculation fails
    
    async def _perform_meta_analysis(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-analysis of causal relationships"""
        
        meta_analysis_results = {
            "method": self.config["meta_analysis_method"],
            "combined_effects": [],
            "publication_bias_tests": {},
            "sensitivity_analysis": {}
        }
        
        # Collect effect sizes for meta-analysis
        effect_data = defaultdict(list)
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                key = (relationship["source"], relationship["target"])
                
                effect_data[key].append({
                    "effect_size": relationship["effect_size"],
                    "standard_error": self._estimate_standard_error(
                        relationship, results["n_observations"]
                    ),
                    "dataset_id": dataset_id,
                    "n": results["n_observations"]
                })
        
        # Perform meta-analysis for each relationship
        for (source, target), effects in effect_data.items():
            if len(effects) >= self.config["min_datasets"]:
                meta_result = self._meta_analyze_effects(effects)
                meta_result["source"] = source
                meta_result["target"] = target
                meta_analysis_results["combined_effects"].append(meta_result)
        
        return meta_analysis_results
    
    def _estimate_standard_error(self, relationship: Dict[str, Any], n_obs: int) -> float:
        """Estimate standard error for effect size"""
        
        # Simplified standard error estimation
        effect_size = abs(relationship["effect_size"])
        
        # For correlation-like effects
        if effect_size <= 1.0:
            # Fisher's z-transformation standard error
            se = 1.0 / np.sqrt(max(3, n_obs - 3))
        else:
            # For other effect sizes, use a simple approximation
            se = effect_size / np.sqrt(max(10, n_obs))
        
        return se
    
    def _meta_analyze_effects(self, effects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform meta-analysis on a set of effect sizes"""
        
        effect_sizes = [e["effect_size"] for e in effects]
        standard_errors = [e["standard_error"] for e in effects]
        sample_sizes = [e["n"] for e in effects]
        
        # Fixed effects meta-analysis
        weights = [1 / (se**2) for se in standard_errors]
        total_weight = sum(weights)
        
        combined_effect = sum(es * w for es, w in zip(effect_sizes, weights)) / total_weight
        combined_se = 1 / np.sqrt(total_weight)
        
        # Confidence interval
        ci_lower = combined_effect - 1.96 * combined_se
        ci_upper = combined_effect + 1.96 * combined_se
        
        # Test for overall effect
        z_score = combined_effect / combined_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Heterogeneity test (Q statistic)
        q_stat = sum(w * (es - combined_effect)**2 for es, w in zip(effect_sizes, weights))
        df = len(effects) - 1
        heterogeneity_p = 1 - stats.chi2.cdf(q_stat, df) if df > 0 else 1.0
        
        return {
            "combined_effect": combined_effect,
            "standard_error": combined_se,
            "confidence_interval": (ci_lower, ci_upper),
            "z_score": z_score,
            "p_value": p_value,
            "q_statistic": q_stat,
            "heterogeneity_p": heterogeneity_p,
            "n_studies": len(effects),
            "total_n": sum(sample_sizes)
        }
    
    async def _analyze_consistency(self, local_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze consistency of findings across datasets"""
        
        consistency_results = {
            "overall_consistency": 0.0,
            "relationship_consistency": {},
            "inconsistent_findings": [],
            "dataset_similarity": {}
        }
        
        # Analyze relationship consistency
        all_relationships = defaultdict(list)
        
        for dataset_id, results in local_results.items():
            for relationship in results["summary"]["discovered_relationships"]:
                key = (relationship["source"], relationship["target"])
                all_relationships[key].append({
                    "dataset_id": dataset_id,
                    "strength": relationship["strength"],
                    "p_value": relationship["p_value"],
                    "effect_size": relationship["effect_size"]
                })
        
        consistent_relationships = 0
        total_relationships = 0
        
        for (source, target), relationship_list in all_relationships.items():
            if len(relationship_list) >= 2:  # Need at least 2 datasets
                total_relationships += 1
                
                # Calculate consistency metrics
                strengths = [r["strength"] for r in relationship_list]
                effect_sizes = [r["effect_size"] for r in relationship_list]
                
                # Consistency based on coefficient of variation
                strength_cv = np.std(strengths) / (np.mean(strengths) + 1e-8)
                effect_cv = np.std(effect_sizes) / (abs(np.mean(effect_sizes)) + 1e-8)
                
                consistency_score = 1.0 / (1.0 + strength_cv + effect_cv)
                
                consistency_results["relationship_consistency"][(source, target)] = {
                    "consistency_score": consistency_score,
                    "strength_cv": strength_cv,
                    "effect_cv": effect_cv,
                    "n_datasets": len(relationship_list),
                    "mean_strength": np.mean(strengths),
                    "mean_effect": np.mean(effect_sizes)
                }
                
                if consistency_score > self.config["consistency_threshold"]:
                    consistent_relationships += 1
                else:
                    consistency_results["inconsistent_findings"].append({
                        "relationship": (source, target),
                        "consistency_score": consistency_score,
                        "details": relationship_list
                    })
        
        # Overall consistency
        if total_relationships > 0:
            consistency_results["overall_consistency"] = consistent_relationships / total_relationships
        
        # Dataset similarity analysis
        dataset_ids = list(local_results.keys())
        for i, dataset1 in enumerate(dataset_ids):
            for j, dataset2 in enumerate(dataset_ids[i+1:], i+1):
                similarity = self._calculate_dataset_similarity(
                    local_results[dataset1], local_results[dataset2]
                )
                consistency_results["dataset_similarity"][(dataset1, dataset2)] = similarity
        
        return consistency_results
    
    def _calculate_dataset_similarity(self, results1: Dict[str, Any], 
                                    results2: Dict[str, Any]) -> float:
        """Calculate similarity between two datasets' causal discovery results"""
        
        # Extract relationship sets
        rels1 = set((r["source"], r["target"]) for r in results1["summary"]["discovered_relationships"])
        rels2 = set((r["source"], r["target"]) for r in results2["summary"]["discovered_relationships"])
        
        # Jaccard similarity
        intersection = len(rels1.intersection(rels2))
        union = len(rels1.union(rels2))
        
        jaccard_similarity = intersection / max(1, union)
        
        return jaccard_similarity
    
    def _generate_federated_summary(self, federated_results: Dict[str, Any],
                                  meta_analysis: Dict[str, Any],
                                  consistency_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of federated causal discovery"""
        
        summary = {
            "total_datasets": len(self.registered_datasets),
            "total_relationships_discovered": len(federated_results["relationships"]),
            "meta_analysis_relationships": len(meta_analysis["combined_effects"]),
            "overall_consistency": consistency_analysis["overall_consistency"],
            "key_findings": [],
            "recommendations": []
        }
        
        # Identify key findings
        for rel in federated_results["relationships"]:
            if (rel["aggregated_strength"] > 0.7 and 
                rel["meta_p_value"] < 0.01 and 
                rel["heterogeneity"] < 0.3):
                
                summary["key_findings"].append({
                    "relationship": f"{rel['source']} → {rel['target']}",
                    "strength": rel["aggregated_strength"],
                    "significance": rel["meta_p_value"],
                    "datasets": rel["n_datasets"],
                    "observations": rel["total_observations"]
                })
        
        # Generate recommendations
        if consistency_analysis["overall_consistency"] < 0.5:
            summary["recommendations"].append(
                "Low consistency across datasets suggests need for more standardized data collection"
            )
        
        if len(summary["key_findings"]) == 0:
            summary["recommendations"].append(
                "No strong consistent relationships found - consider larger sample sizes or different variables"
            )
        
        return summary

class ScientificKnowledgeGraph:
    """Main scientific knowledge graph system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.literature_extractor = LiteratureExtractor()
        self.temporal_analyzer = TemporalCausalAnalyzer()
        self.federated_discovery = FederatedCausalDiscovery()
        
        # Knowledge graph storage
        self.entities = {}  # entity_id -> KnowledgeEntity
        self.relations = {}  # relation_id -> KnowledgeRelation
        self.temporal_relations = {}  # relation_id -> TemporalCausalRelation
        
        # External data sources
        self.external_sources = {
            "geoscienceworld": GeoScienceWorldAPI(),
            "usgs": USGSDataAPI()
        }
        
        # Graph representation
        if NETWORKX_AVAILABLE:
            self.graph = nx.MultiDiGraph()
        else:
            self.graph = None
        
        # RDF store
        if RDF_AVAILABLE:
            self.rdf_graph = Graph()
            self.geo_namespace = Namespace("http://ontology.prometheus.ai/geological/")
        else:
            self.rdf_graph = None
            self.geo_namespace = None
        
        # Initialize database
        self._initialize_database()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "database_path": "knowledge_graph.db",
            "enable_literature_extraction": True,
            "enable_temporal_analysis": True,
            "enable_federated_discovery": True,
            "auto_update_interval": 24 * 3600,  # 24 hours
            "confidence_threshold": 0.6,
            "semantic_similarity_threshold": 0.8,
            "max_literature_documents": 1000,
            "external_api_rate_limit": 10  # requests per second
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        
        db_path = self.config["database_path"]
        
        try:
            self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # Create tables
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    entity_type TEXT,
                    properties TEXT,
                    confidence REAL,
                    created_timestamp TEXT,
                    last_updated TEXT
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT,
                    target_entity_id TEXT,
                    relation_type TEXT,
                    properties TEXT,
                    confidence REAL,
                    created_timestamp TEXT,
                    last_updated TEXT,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id)
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS temporal_relations (
                    id TEXT PRIMARY KEY,
                    source_entity_id TEXT,
                    target_entity_id TEXT,
                    temporal_lag REAL,
                    causal_strength REAL,
                    granger_p_value REAL,
                    created_timestamp TEXT,
                    FOREIGN KEY (source_entity_id) REFERENCES entities (id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities (id)
                )
            """)
            
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS literature_documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    publication_year INTEGER,
                    journal TEXT,
                    doi TEXT,
                    processed BOOLEAN DEFAULT FALSE,
                    created_timestamp TEXT
                )
            """)
            
            self.db_conn.commit()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            self.db_conn = None
    
    async def add_literature_document(self, document: LiteratureDocument) -> bool:
        """Add literature document and extract knowledge"""
        
        try:
            # Store document in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO literature_documents 
                    (id, title, authors, abstract, publication_year, journal, doi, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document.id, document.title, json.dumps(document.authors),
                    document.abstract, document.publication_year, document.journal,
                    document.doi, document.metadata.get("created_timestamp", datetime.now().isoformat())
                ))
                self.db_conn.commit()
            
            # Extract knowledge if enabled
            if self.config["enable_literature_extraction"]:
                entities, relations = await self.literature_extractor.extract_from_document(document)
                
                # Add entities and relations to knowledge graph
                for entity in entities:
                    await self.add_entity(entity)
                
                for relation in relations:
                    await self.add_relation(relation)
                
                self.logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from {document.id}")
                
                # Mark document as processed
                if self.db_conn:
                    self.db_conn.execute(
                        "UPDATE literature_documents SET processed = TRUE WHERE id = ?",
                        (document.id,)
                    )
                    self.db_conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add literature document {document.id}: {str(e)}")
            return False
    
    async def add_entity(self, entity: KnowledgeEntity) -> bool:
        """Add entity to knowledge graph"""
        
        try:
            # Check for existing similar entities
            similar_entities = await self._find_similar_entities(entity)
            
            if similar_entities:
                # Merge with most similar entity
                most_similar = similar_entities[0]
                merged_entity = await self._merge_entities(most_similar, entity)
                entity = merged_entity
            
            # Store in memory
            self.entities[entity.id] = entity
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, name, entity_type, properties, confidence, created_timestamp, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id, entity.name, entity.entity_type.value,
                    json.dumps(entity.properties), entity.confidence,
                    entity.created_timestamp, entity.last_updated
                ))
                self.db_conn.commit()
            
            # Add to NetworkX graph
            if self.graph is not None:
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    entity_type=entity.entity_type.value,
                    confidence=entity.confidence
                )
            
            # Add to RDF graph
            if self.rdf_graph is not None:
                entity_uri = self.geo_namespace[entity.id]
                self.rdf_graph.add((entity_uri, RDF.type, self.geo_namespace[entity.entity_type.value]))
                self.rdf_graph.add((entity_uri, self.geo_namespace.name, Literal(entity.name)))
                self.rdf_graph.add((entity_uri, self.geo_namespace.confidence, Literal(entity.confidence)))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add entity {entity.id}: {str(e)}")
            return False
    
    async def add_relation(self, relation: KnowledgeRelation) -> bool:
        """Add relation to knowledge graph"""
        
        try:
            # Verify entities exist
            if (relation.source_entity_id not in self.entities or 
                relation.target_entity_id not in self.entities):
                self.logger.warning(f"Relation {relation.id} references non-existent entities")
                return False
            
            # Store in memory
            self.relations[relation.id] = relation
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO relations 
                    (id, source_entity_id, target_entity_id, relation_type, properties, 
                     confidence, created_timestamp, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    relation.id, relation.source_entity_id, relation.target_entity_id,
                    relation.relation_type.value, json.dumps(relation.properties),
                    relation.confidence, relation.created_timestamp, relation.last_updated
                ))
                self.db_conn.commit()
            
            # Add to NetworkX graph
            if self.graph is not None:
                self.graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    key=relation.id,
                    relation_type=relation.relation_type.value,
                    confidence=relation.confidence
                )
            
            # Add to RDF graph
            if self.rdf_graph is not None:
                source_uri = self.geo_namespace[relation.source_entity_id]
                target_uri = self.geo_namespace[relation.target_entity_id]
                relation_uri = self.geo_namespace[relation.relation_type.value]
                
                self.rdf_graph.add((source_uri, relation_uri, target_uri))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add relation {relation.id}: {str(e)}")
            return False
    
    async def add_temporal_relation(self, temporal_relation: TemporalCausalRelation) -> bool:
        """Add temporal causal relation to knowledge graph"""
        
        try:
            # Store in memory
            self.temporal_relations[temporal_relation.relation_id] = temporal_relation
            
            # Store in database
            if self.db_conn:
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO temporal_relations 
                    (id, source_entity_id, target_entity_id, temporal_lag, 
                     causal_strength, granger_p_value, created_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    temporal_relation.relation_id,
                    temporal_relation.source_entity_id,
                    temporal_relation.target_entity_id,
                    temporal_relation.temporal_lag,
                    temporal_relation.causal_strength,
                    temporal_relation.granger_causality_pvalue,
                    datetime.now().isoformat()
                ))
                self.db_conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add temporal relation {temporal_relation.relation_id}: {str(e)}")
            return False
    
    async def _find_similar_entities(self, entity: KnowledgeEntity) -> List[KnowledgeEntity]:
        """Find similar entities in the knowledge graph"""
        
        similar_entities = []
        
        for existing_entity in self.entities.values():
            if existing_entity.entity_type == entity.entity_type:
                # Check name similarity
                similarity = self._calculate_string_similarity(entity.name, existing_entity.name)
                
                if similarity > self.config["semantic_similarity_threshold"]:
                    similar_entities.append((existing_entity, similarity))
        
        # Sort by similarity and return entities
        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return [entity for entity, _ in similar_entities]
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Jaccard similarity"""
        
        # Simple Jaccard similarity on word level
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(1, union)
    
    async def _merge_entities(self, entity1: KnowledgeEntity, 
                            entity2: KnowledgeEntity) -> KnowledgeEntity:
        """Merge two similar entities"""
        
        # Keep the entity with higher confidence as primary
        if entity1.confidence >= entity2.confidence:
            primary, secondary = entity1, entity2
        else:
            primary, secondary = entity2, entity1
        
        # Merge properties
        merged_properties = primary.properties.copy()
        merged_properties.update(secondary.properties)
        
        # Merge aliases
        merged_aliases = list(set(primary.aliases + secondary.aliases + [secondary.name]))
        
        # Merge source references
        merged_references = list(set(primary.source_references + secondary.source_references))
        
        # Create merged entity
        merged_entity = KnowledgeEntity(
            id=primary.id,  # Keep primary ID
            name=primary.name,
            entity_type=primary.entity_type,
            properties=merged_properties,
            aliases=merged_aliases,
            confidence=max(primary.confidence, secondary.confidence),
            source_references=merged_references,
            created_timestamp=min(primary.created_timestamp, secondary.created_timestamp),
            last_updated=datetime.now().isoformat(),
            embedding_vector=primary.embedding_vector  # Keep primary embedding
        )
        
        return merged_entity
    
    async def query_knowledge_graph(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge graph using natural language"""
        
        results = []
        
        try:
            # Simple keyword-based search (can be enhanced with more sophisticated NLP)
            query_words = set(query.lower().split())
            
            # Search entities
            for entity in self.entities.values():
                entity_words = set((entity.name + " " + " ".join(entity.aliases)).lower().split())
                
                overlap = len(query_words.intersection(entity_words))
                if overlap > 0:
                    results.append({
                        "type": "entity",
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type.value,
                        "confidence": entity.confidence,
                        "relevance_score": overlap / len(query_words)
                    })
            
            # Search relations
            for relation in self.relations.values():
                source_entity = self.entities.get(relation.source_entity_id)
                target_entity = self.entities.get(relation.target_entity_id)
                
                if source_entity and target_entity:
                    relation_text = f"{source_entity.name} {relation.relation_type.value} {target_entity.name}"
                    relation_words = set(relation_text.lower().split())
                    
                    overlap = len(query_words.intersection(relation_words))
                    if overlap > 0:
                        results.append({
                            "type": "relation",
                            "id": relation.id,
                            "source": source_entity.name,
                            "relation": relation.relation_type.value,
                            "target": target_entity.name,
                            "confidence": relation.confidence,
                            "relevance_score": overlap / len(query_words)
                        })
            
            # Sort by relevance and limit results
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            results = results[:limit]
            
        except Exception as e:
            self.logger.error(f"Knowledge graph query failed: {str(e)}")
        
        return results
    
    async def update_from_external_sources(self) -> Dict[str, Any]:
        """Update knowledge graph from external scientific databases"""
        
        update_results = {
            "sources_queried": [],
            "new_entities": 0,
            "new_relations": 0,
            "errors": []
        }
        
        for source_name, source in self.external_sources.items():
            try:
                self.logger.info(f"Querying external source: {source_name}")
                
                # Query recent geological research
                query_terms = ["earthquake", "volcano", "geology", "geophysics", "tectonics"]
                
                for term in query_terms[:2]:  # Limit to avoid rate limits
                    documents = await source.query(term, limit=10)
                    
                    for doc_data in documents:
                        # Create literature document
                        doc_id = hashlib.md5(f"{source_name}_{doc_data.get('doi', doc_data['title'])}".encode()).hexdigest()[:16]
                        
                        document = LiteratureDocument(
                            id=doc_id,
                            title=doc_data["title"],
                            authors=doc_data.get("authors", []),
                            abstract=doc_data.get("abstract", ""),
                            full_text=doc_data.get("full_text", ""),
                            publication_year=doc_data.get("year", 2023),
                            journal=doc_data.get("journal", source_name),
                            doi=doc_data.get("doi"),
                            keywords=doc_data.get("keywords", [])
                        )
                        
                        # Add to knowledge graph
                        success = await self.add_literature_document(document)
                        if success:
                            update_results["new_entities"] += len(document.extracted_entities)
                            update_results["new_relations"] += len(document.extracted_relations)
                
                update_results["sources_queried"].append(source_name)
                
                # Rate limiting
                await asyncio.sleep(1 / self.config["external_api_rate_limit"])
                
            except Exception as e:
                error_msg = f"Failed to update from {source_name}: {str(e)}"
                self.logger.error(error_msg)
                update_results["errors"].append(error_msg)
        
        return update_results
    
    async def discover_temporal_causal_relationships(self, time_series_data: Dict[str, pd.Series]) -> List[TemporalCausalRelation]:
        """Discover temporal causal relationships from time-series data"""
        
        temporal_relations = []
        
        if not self.config["enable_temporal_analysis"]:
            return temporal_relations
        
        # Get all pairs of time series
        variables = list(time_series_data.keys())
        
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                try:
                    # Find corresponding entities
                    entity1_id = await self._find_entity_by_name(var1)
                    entity2_id = await self._find_entity_by_name(var2)
                    
                    if entity1_id and entity2_id:
                        # Analyze temporal causality in both directions
                        relation1 = await self.temporal_analyzer.analyze_temporal_causality(
                            time_series_data[var1], time_series_data[var2], (entity1_id, entity2_id)
                        )
                        
                        relation2 = await self.temporal_analyzer.analyze_temporal_causality(
                            time_series_data[var2], time_series_data[var1], (entity2_id, entity1_id)
                        )
                        
                        # Keep the stronger relationship
                        if relation1 and relation2:
                            if relation1.causal_strength > relation2.causal_strength:
                                temporal_relations.append(relation1)
                            else:
                                temporal_relations.append(relation2)
                        elif relation1:
                            temporal_relations.append(relation1)
                        elif relation2:
                            temporal_relations.append(relation2)
                
                except Exception as e:
                    self.logger.warning(f"Temporal analysis failed for {var1}-{var2}: {str(e)}")
                    continue
        
        # Add temporal relations to knowledge graph
        for relation in temporal_relations:
            await self.add_temporal_relation(relation)
        
        return temporal_relations
    
    async def _find_entity_by_name(self, name: str) -> Optional[str]:
        """Find entity ID by name"""
        
        name_lower = name.lower()
        
        for entity_id, entity in self.entities.items():
            if (entity.name.lower() == name_lower or 
                name_lower in [alias.lower() for alias in entity.aliases]):
                return entity_id
        
        return None
    
    async def perform_federated_discovery(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Perform federated causal discovery across multiple datasets"""
        
        if not self.config["enable_federated_discovery"]:
            return {"error": "Federated discovery disabled"}
        
        # Register datasets
        for dataset_id, data in datasets.items():
            self.federated_discovery.register_dataset(dataset_id, data)
        
        # Discover federated causal relationships
        federated_results = await self.federated_discovery.discover_federated_causal_relationships()
        
        # Integrate results into knowledge graph
        await self._integrate_federated_results(federated_results)
        
        return federated_results
    
    async def _integrate_federated_results(self, federated_results: Dict[str, Any]):
        """Integrate federated discovery results into knowledge graph"""
        
        try:
            for relationship in federated_results["federated_causal_graph"]["relationships"]:
                source_name = relationship["source"]
                target_name = relationship["target"]
                
                # Find or create entities
                source_entity_id = await self._find_entity_by_name(source_name)
                if not source_entity_id:
                    # Create new entity
                    entity_id = hashlib.md5(f"federated_{source_name}".encode()).hexdigest()[:16]
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=source_name,
                        entity_type=EntityType.CONCEPT,
                        confidence=0.8,
                        source_references=["federated_discovery"]
                    )
                    await self.add_entity(entity)
                    source_entity_id = entity_id
                
                target_entity_id = await self._find_entity_by_name(target_name)
                if not target_entity_id:
                    # Create new entity
                    entity_id = hashlib.md5(f"federated_{target_name}".encode()).hexdigest()[:16]
                    entity = KnowledgeEntity(
                        id=entity_id,
                        name=target_name,
                        entity_type=EntityType.CONCEPT,
                        confidence=0.8,
                        source_references=["federated_discovery"]
                    )
                    await self.add_entity(entity)
                    target_entity_id = entity_id
                
                # Create federated causal relation
                relation_id = hashlib.md5(
                    f"federated_{source_entity_id}_{target_entity_id}".encode()
                ).hexdigest()[:16]
                
                relation = KnowledgeRelation(
                    id=relation_id,
                    source_entity_id=source_entity_id,
                    target_entity_id=target_entity_id,
                    relation_type=RelationType.CAUSES,
                    properties={
                        "federated_discovery": True,
                        "aggregated_strength": relationship["aggregated_strength"],
                        "meta_p_value": relationship["meta_p_value"],
                        "n_datasets": relationship["n_datasets"],
                        "heterogeneity": relationship["heterogeneity"]
                    },
                    confidence=relationship["aggregated_strength"],
                    evidence=[f"Federated analysis across {relationship['n_datasets']} datasets"],
                    source_references=["federated_discovery"]
                )
                
                await self.add_relation(relation)
                
        except Exception as e:
            self.logger.error(f"Failed to integrate federated results: {str(e)}")
    
    def export_knowledge_graph(self, format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export knowledge graph in specified format"""
        
        if format == "json":
            return {
                "entities": [
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type.value,
                        "properties": entity.properties,
                        "confidence": entity.confidence,
                        "aliases": entity.aliases
                    }
                    for entity in self.entities.values()
                ],
                "relations": [
                    {
                        "id": relation.id,
                        "source": relation.source_entity_id,
                        "target": relation.target_entity_id,
                        "type": relation.relation_type.value,
                        "properties": relation.properties,
                        "confidence": relation.confidence
                    }
                    for relation in self.relations.values()
                ],
                "temporal_relations": [
                    {
                        "id": relation.relation_id,
                        "source": relation.source_entity_id,
                        "target": relation.target_entity_id,
                        "temporal_lag": relation.temporal_lag,
                        "causal_strength": relation.causal_strength
                    }
                    for relation in self.temporal_relations.values()
                ]
            }
        
        elif format == "rdf" and self.rdf_graph:
            return self.rdf_graph.serialize(format="turtle")
        
        elif format == "networkx" and self.graph:
            return nx.node_link_data(self.graph)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_temporal_relations": len(self.temporal_relations),
            "entity_types": {
                entity_type.value: sum(1 for e in self.entities.values() 
                                     if e.entity_type == entity_type)
                for entity_type in EntityType
            },
            "relation_types": {
                relation_type.value: sum(1 for r in self.relations.values() 
                                       if r.relation_type == relation_type)
                for relation_type in RelationType
            },
            "average_entity_confidence": np.mean([e.confidence for e in self.entities.values()]) if self.entities else 0.0,
            "average_relation_confidence": np.mean([r.confidence for r in self.relations.values()]) if self.relations else 0.0
        }

# Example usage and testing
async def example_knowledge_graph():
    """Example of scientific knowledge graph system"""
    
    # Initialize knowledge graph
    kg = ScientificKnowledgeGraph()
    
    print("Scientific Knowledge Graph System initialized")
    print(f"Components available:")
    print(f"  - NetworkX: {NETWORKX_AVAILABLE}")
    print(f"  - spaCy: {SPACY_AVAILABLE}")
    print(f"  - Transformers: {TRANSFORMERS_AVAILABLE}")
    print(f"  - RDFLib: {RDF_AVAILABLE}")
    
    # Add sample literature document
    sample_document = LiteratureDocument(
        id="sample_doc_001",
        title="Volcanic Eruption Triggers and Earthquake Relationships",
        authors=["Dr. Smith", "Dr. Johnson"],
        abstract="This study investigates the relationship between volcanic eruptions and seismic activity, finding strong correlations between magma pressure and earthquake frequency.",
        full_text="Volcanic eruptions are complex phenomena that often correlate with increased seismic activity. Our analysis of 10 years of data from Mount Example shows that increased magma pressure leads to higher earthquake frequency. The crystallization process in magma chambers causes stress buildup, which triggers fault activation in the surrounding rock formations.",
        publication_year=2023,
        journal="Journal of Geological Sciences",
        doi="10.1000/sample.2023.001",
        keywords=["volcano", "earthquake", "magma", "seismicity", "crystallization"]
    )
    
    print(f"\nAdding literature document: {sample_document.title}")
    await kg.add_literature_document(sample_document)
    
    # Create sample time-series data for temporal analysis
    print("\nGenerating sample time-series data for temporal causal analysis...")
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    
    # Create synthetic correlated time series
    np.random.seed(42)
    n_periods = len(dates)
    
    # Magma pressure (leading indicator)
    magma_pressure = 5.0 + 2.0 * np.sin(np.arange(n_periods) * 2 * np.pi / 12) + np.random.normal(0, 0.5, n_periods)
    
    # Earthquake frequency (lagged response to magma pressure)
    earthquake_freq = np.zeros(n_periods)
    for i in range(1, n_periods):
        earthquake_freq[i] = 2.0 + 0.3 * magma_pressure[i-1] + np.random.normal(0, 0.3)
    
    time_series_data = {
        "magma_pressure": pd.Series(magma_pressure, index=dates),
        "earthquake_frequency": pd.Series(earthquake_freq, index=dates)
    }
    
    # Discover temporal causal relationships
    temporal_relations = await kg.discover_temporal_causal_relationships(time_series_data)
    
    print(f"Discovered {len(temporal_relations)} temporal causal relationships")
    for relation in temporal_relations:
        if relation.granger_causality_pvalue and relation.granger_causality_pvalue < 0.05:
            print(f"  Significant temporal causality: {relation.source_entity_id} → {relation.target_entity_id}")
            print(f"    Lag: {relation.temporal_lag} {relation.temporal_units}")
            print(f"    Causal strength: {relation.causal_strength:.3f}")
            print(f"    Granger p-value: {relation.granger_causality_pvalue:.3f}")
    
    # Create sample datasets for federated discovery
    print("\nPreparing sample datasets for federated causal discovery...")
    
    # Dataset 1: Volcanic region A
    np.random.seed(123)
    n_samples = 200
    dataset1 = pd.DataFrame({
        "temperature": np.random.normal(800, 100, n_samples),
        "pressure": np.random.normal(5, 1, n_samples),
        "viscosity": np.zeros(n_samples)
    })
    
    # Create causal relationship: temperature and pressure affect viscosity
    for i in range(n_samples):
        dataset1.loc[i, "viscosity"] = (1e6 * np.exp(-dataset1.loc[i, "temperature"]/1000) * 
                                       (dataset1.loc[i, "pressure"]**0.3) + 
                                       np.random.lognormal(0, 0.2))
    
    # Dataset 2: Volcanic region B (similar but with different parameters)
    np.random.seed(456)
    dataset2 = pd.DataFrame({
        "temperature": np.random.normal(750, 120, n_samples),
        "pressure": np.random.normal(4.5, 1.2, n_samples),
        "viscosity": np.zeros(n_samples)
    })
    
    for i in range(n_samples):
        dataset2.loc[i, "viscosity"] = (1.2e6 * np.exp(-dataset2.loc[i, "temperature"]/950) * 
                                       (dataset2.loc[i, "pressure"]**0.35) + 
                                       np.random.lognormal(0, 0.25))
    
    datasets = {
        "volcanic_region_A": dataset1,
        "volcanic_region_B": dataset2
    }
    
    # Perform federated causal discovery
    print("\nPerforming federated causal discovery...")
    federated_results = await kg.perform_federated_discovery(datasets)
    
    print("Federated Discovery Results:")
    if "summary" in federated_results:
        summary = federated_results["summary"]
        print(f"  Total datasets: {summary.get('total_datasets', 0)}")
        print(f"  Relationships discovered: {summary.get('total_relationships_discovered', 0)}")
        print(f"  Overall consistency: {summary.get('overall_consistency', 0):.3f}")
        
        if "key_findings" in summary:
            print(f"  Key findings:")
            for finding in summary["key_findings"][:3]:
                print(f"    {finding['relationship']}: strength={finding['strength']:.3f}, p={finding['significance']:.3f}")
    
    # Query knowledge graph
    print("\nQuerying knowledge graph...")
    queries = [
        "volcano earthquake relationship",
        "magma pressure crystallization",
        "temperature viscosity"
    ]
    
    for query in queries:
        results = await kg.query_knowledge_graph(query, limit=5)
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:")
        for result in results[:3]:
            if result["type"] == "entity":
                print(f"  Entity: {result['name']} ({result['entity_type']}) - confidence: {result['confidence']:.3f}")
            elif result["type"] == "relation":
                print(f"  Relation: {result['source']} {result['relation']} {result['target']} - confidence: {result['confidence']:.3f}")
    
    # Update from external sources (mock)
    print("\nUpdating from external sources...")
    update_results = await kg.update_from_external_sources()
    print(f"Sources queried: {len(update_results['sources_queried'])}")
    print(f"New entities: {update_results['new_entities']}")
    print(f"New relations: {update_results['new_relations']}")
    if update_results['errors']:
        print(f"Errors encountered: {len(update_results['errors'])}")
    
    # Export knowledge graph
    print("\nExporting knowledge graph...")
    export_data = kg.export_knowledge_graph(format="json")
    print(f"Exported {len(export_data['entities'])} entities and {len(export_data['relations'])} relations")
    
    # Get statistics
    stats = kg.get_statistics()
    print(f"\nKnowledge Graph Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Total relations: {stats['total_relations']}")
    print(f"  Total temporal relations: {stats['total_temporal_relations']}")
    print(f"  Average entity confidence: {stats['average_entity_confidence']:.3f}")
    print(f"  Average relation confidence: {stats['average_relation_confidence']:.3f}")
    
    print(f"\nEntity types:")
    for entity_type, count in stats['entity_types'].items():
        if count > 0:
            print(f"    {entity_type}: {count}")
    
    print(f"\nRelation types:")
    for relation_type, count in stats['relation_types'].items():
        if count > 0:
            print(f"    {relation_type}: {count}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_knowledge_graph())