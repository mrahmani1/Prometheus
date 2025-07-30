"""
Scientific Knowledge Extractor for automated literature processing
Extracts geological concepts, relationships, and facts from scientific literature
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import structlog

import spacy
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.cluster import DBSCAN
import numpy as np

from .config import settings

logger = structlog.get_logger()


@dataclass
class ExtractedKnowledge:
    """Container for extracted scientific knowledge"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    temporal_facts: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    source_metadata: Dict[str, Any]


@dataclass
class ScientificPaper:
    """Scientific paper metadata and content"""
    title: str
    authors: List[str]
    abstract: str
    full_text: Optional[str]
    doi: str
    publication_date: datetime
    journal: str
    keywords: List[str]
    citations: List[str]


class ScientificKnowledgeExtractor:
    """Automated knowledge extraction from geological literature"""
    
    def __init__(self):
        self.nlp = None
        self.tokenizer = None
        self.model = None
        self.sentence_transformer = None
        self.ner_pipeline = None
        self.relation_extractor = None
        
        # Geological entity patterns
        self.geological_patterns = {
            'minerals': r'\b(?:quartz|feldspar|mica|olivine|pyroxene|amphibole|calcite|dolomite|gypsum|halite)\b',
            'rocks': r'\b(?:granite|basalt|limestone|sandstone|shale|marble|quartzite|schist|gneiss|slate)\b',
            'processes': r'\b(?:subduction|volcanism|metamorphism|weathering|erosion|sedimentation|crystallization)\b',
            'structures': r'\b(?:fault|fold|joint|fracture|dike|sill|batholith|pluton|volcano|crater)\b',
            'time_periods': r'\b(?:Precambrian|Paleozoic|Mesozoic|Cenozoic|Quaternary|Holocene|Pleistocene)\b'
        }
        
        # Relationship patterns
        self.relationship_patterns = {
            'causal': r'\b(?:causes?|results? in|leads? to|triggers?|induces?|produces?)\b',
            'temporal': r'\b(?:before|after|during|following|preceding|simultaneous)\b',
            'spatial': r'\b(?:above|below|within|adjacent to|near|distant from)\b',
            'compositional': r'\b(?:contains?|comprises?|consists? of|includes?)\b'
        }
    
    async def initialize(self):
        """Initialize NLP models and components"""
        logger.info("Initializing Scientific Knowledge Extractor")
        
        try:
            # Load spaCy model for geological NER
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add custom geological entity patterns
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            patterns = []
            for entity_type, pattern in self.geological_patterns.items():
                patterns.append({"label": entity_type.upper(), "pattern": [{"TEXT": {"REGEX": pattern}}]})
            ruler.add_patterns(patterns)
            
            # Load transformer models for embeddings and relation extraction
            model_name = "allenai/scibert_scivocab_uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Sentence transformer for semantic similarity
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Named entity recognition pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Relation extraction pipeline
            self.relation_extractor = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium"
            )
            
            logger.info("Scientific Knowledge Extractor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize knowledge extractor", error=str(e))
            raise
    
    async def extract_from_paper(self, paper: ScientificPaper) -> ExtractedKnowledge:
        """Extract knowledge from a scientific paper"""
        logger.info("Extracting knowledge from paper", title=paper.title)
        
        try:
            # Combine title, abstract, and full text
            full_content = f"{paper.title}. {paper.abstract}"
            if paper.full_text:
                full_content += f" {paper.full_text}"
            
            # Extract entities
            entities = await self._extract_entities(full_content)
            
            # Extract relationships
            relationships = await self._extract_relationships(full_content, entities)
            
            # Extract concepts and their definitions
            concepts = await self._extract_concepts(full_content)
            
            # Extract temporal facts
            temporal_facts = await self._extract_temporal_facts(full_content)
            
            # Calculate confidence scores
            confidence_scores = await self._calculate_confidence_scores(
                entities, relationships, concepts, temporal_facts
            )
            
            # Create source metadata
            source_metadata = {
                "doi": paper.doi,
                "title": paper.title,
                "authors": paper.authors,
                "publication_date": paper.publication_date.isoformat(),
                "journal": paper.journal,
                "extraction_timestamp": datetime.utcnow().isoformat()
            }
            
            return ExtractedKnowledge(
                entities=entities,
                relationships=relationships,
                concepts=concepts,
                temporal_facts=temporal_facts,
                confidence_scores=confidence_scores,
                source_metadata=source_metadata
            )
            
        except Exception as e:
            logger.error("Failed to extract knowledge from paper", error=str(e))
            raise
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract geological entities from text"""
        entities = []
        
        # Use spaCy for custom geological entities
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.8  # Default confidence for pattern-based extraction
            })
        
        # Use transformer-based NER for additional entities
        ner_results = self.ner_pipeline(text)
        for result in ner_results:
            entities.append({
                "text": result["word"],
                "label": result["entity_group"],
                "start": result["start"],
                "end": result["end"],
                "confidence": result["score"]
            })
        
        # Deduplicate entities
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    async def _extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        for sentence in sentences:
            # Find entities in this sentence
            sentence_entities = []
            for entity in entities:
                if entity["text"].lower() in sentence.lower():
                    sentence_entities.append(entity)
            
            # Extract relationships between entity pairs
            if len(sentence_entities) >= 2:
                for i, entity1 in enumerate(sentence_entities):
                    for entity2 in sentence_entities[i+1:]:
                        relationship = await self._extract_relationship_between_entities(
                            sentence, entity1, entity2
                        )
                        if relationship:
                            relationships.append(relationship)
        
        return relationships
    
    async def _extract_relationship_between_entities(
        self, sentence: str, entity1: Dict[str, Any], entity2: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract relationship between two specific entities in a sentence"""
        
        # Check for relationship patterns
        for rel_type, pattern in self.relationship_patterns.items():
            if re.search(pattern, sentence, re.IGNORECASE):
                # Determine direction based on entity positions
                e1_pos = sentence.lower().find(entity1["text"].lower())
                e2_pos = sentence.lower().find(entity2["text"].lower())
                
                if e1_pos < e2_pos:
                    source, target = entity1, entity2
                else:
                    source, target = entity2, entity1
                
                return {
                    "source": source["text"],
                    "target": target["text"],
                    "relationship_type": rel_type,
                    "context": sentence,
                    "confidence": 0.7,
                    "source_entity_type": source["label"],
                    "target_entity_type": target["label"]
                }
        
        return None
    
    async def _extract_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Extract geological concepts and their definitions"""
        concepts = []
        
        # Pattern for definitions (e.g., "X is defined as Y", "X refers to Y")
        definition_patterns = [
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+is\s+defined\s+as\s+([^.]+)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+refers\s+to\s+([^.]+)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s+means\s+([^.]+)',
            r'([A-Z][a-z]+(?:\s+[a-z]+)*)\s*:\s*([^.]+)'
        ]
        
        for pattern in definition_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_name = match.group(1).strip()
                definition = match.group(2).strip()
                
                # Check if it's a geological concept
                if any(re.search(geo_pattern, concept_name, re.IGNORECASE) 
                       for geo_pattern in self.geological_patterns.values()):
                    concepts.append({
                        "name": concept_name,
                        "definition": definition,
                        "context": match.group(0),
                        "confidence": 0.8
                    })
        
        return concepts
    
    async def _extract_temporal_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract temporal facts and sequences"""
        temporal_facts = []
        
        # Pattern for temporal sequences
        temporal_patterns = [
            r'(\d+(?:\.\d+)?)\s*(Ma|Ga|ka|years?\s+ago)\s+([^.]+)',
            r'(during|in)\s+the\s+([A-Z][a-z]+)\s+([^.]+)',
            r'(before|after)\s+([^,]+),\s*([^.]+)'
        ]
        
        for pattern in temporal_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                temporal_facts.append({
                    "temporal_marker": match.group(1),
                    "event_description": match.group(-1),  # Last group
                    "context": match.group(0),
                    "confidence": 0.6
                })
        
        return temporal_facts
    
    async def _calculate_confidence_scores(
        self, entities: List[Dict], relationships: List[Dict], 
        concepts: List[Dict], temporal_facts: List[Dict]
    ) -> Dict[str, float]:
        """Calculate overall confidence scores for extracted knowledge"""
        
        entity_confidence = np.mean([e["confidence"] for e in entities]) if entities else 0.0
        relationship_confidence = np.mean([r["confidence"] for r in relationships]) if relationships else 0.0
        concept_confidence = np.mean([c["confidence"] for c in concepts]) if concepts else 0.0
        temporal_confidence = np.mean([t["confidence"] for t in temporal_facts]) if temporal_facts else 0.0
        
        return {
            "entities": float(entity_confidence),
            "relationships": float(relationship_confidence),
            "concepts": float(concept_confidence),
            "temporal_facts": float(temporal_confidence),
            "overall": float(np.mean([
                entity_confidence, relationship_confidence, 
                concept_confidence, temporal_confidence
            ]))
        }
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on text similarity"""
        if not entities:
            return entities
        
        # Create embeddings for entity texts
        texts = [entity["text"] for entity in entities]
        embeddings = self.sentence_transformer.encode(texts)
        
        # Cluster similar entities
        clustering = DBSCAN(eps=0.3, min_samples=1, metric='cosine')
        clusters = clustering.fit_predict(embeddings)
        
        # Keep the highest confidence entity from each cluster
        deduplicated = []
        for cluster_id in set(clusters):
            cluster_entities = [entities[i] for i, c in enumerate(clusters) if c == cluster_id]
            best_entity = max(cluster_entities, key=lambda x: x["confidence"])
            deduplicated.append(best_entity)
        
        return deduplicated
    
    async def extract_from_multiple_sources(
        self, papers: List[ScientificPaper]
    ) -> List[ExtractedKnowledge]:
        """Extract knowledge from multiple papers concurrently"""
        logger.info("Extracting knowledge from multiple sources", count=len(papers))
        
        tasks = [self.extract_from_paper(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        extracted_knowledge = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Failed to extract knowledge from paper",
                    paper_title=papers[i].title,
                    error=str(result)
                )
            else:
                extracted_knowledge.append(result)
        
        logger.info("Knowledge extraction completed", successful=len(extracted_knowledge))
        return extracted_knowledge
    
    async def close(self):
        """Cleanup resources"""
        logger.info("Closing Scientific Knowledge Extractor")
        # Cleanup any resources if needed