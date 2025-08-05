"""
Tests for the Scientific Knowledge Extractor
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from core.knowledge_extractor import ScientificKnowledgeExtractor, ScientificPaper, ExtractedKnowledge


@pytest.fixture
async def knowledge_extractor():
    """Create a knowledge extractor instance for testing"""
    extractor = ScientificKnowledgeExtractor()
    
    # Mock the initialization to avoid loading actual models
    with patch.object(extractor, 'initialize') as mock_init:
        mock_init.return_value = None
        
        # Mock the NLP components
        extractor.nlp = Mock()
        extractor.tokenizer = Mock()
        extractor.model = Mock()
        extractor.sentence_transformer = Mock()
        extractor.ner_pipeline = Mock()
        extractor.relation_extractor = Mock()
        
        await extractor.initialize()
    
    return extractor


@pytest.fixture
def sample_paper():
    """Create a sample scientific paper for testing"""
    return ScientificPaper(
        title="Volcanic Eruption Prediction Using Seismic Precursors",
        authors=["Dr. Jane Smith", "Prof. John Doe"],
        abstract="This study investigates the relationship between seismic activity and volcanic eruptions.",
        full_text="Introduction: Volcanic eruptions pose significant hazards. Methods: We analyzed seismic data. Results: Earthquake frequency increases before eruptions. Conclusion: Seismic precursors provide reliable indicators.",
        doi="10.1000/example.2023.001",
        publication_date=datetime(2023, 6, 15),
        journal="Journal of Volcanology",
        keywords=["volcanic eruption", "seismic precursors"],
        citations=["10.1000/ref.2022.001"]
    )


class TestScientificKnowledgeExtractor:
    """Test cases for ScientificKnowledgeExtractor"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test knowledge extractor initialization"""
        extractor = ScientificKnowledgeExtractor()
        
        # Test that geological patterns are defined
        assert "minerals" in extractor.geological_patterns
        assert "rocks" in extractor.geological_patterns
        assert "processes" in extractor.geological_patterns
        
        # Test that relationship patterns are defined
        assert "causal" in extractor.relationship_patterns
        assert "temporal" in extractor.relationship_patterns
        assert "spatial" in extractor.relationship_patterns
    
    @pytest.mark.asyncio
    async def test_extract_entities(self, knowledge_extractor):
        """Test entity extraction from text"""
        text = "The granite contains quartz and feldspar minerals. Volcanic eruption occurred."
        
        # Mock spaCy NLP results
        mock_doc = Mock()
        mock_ent1 = Mock()
        mock_ent1.text = "granite"
        mock_ent1.label_ = "ROCKS"
        mock_ent1.start_char = 4
        mock_ent1.end_char = 11
        
        mock_ent2 = Mock()
        mock_ent2.text = "quartz"
        mock_ent2.label_ = "MINERALS"
        mock_ent2.start_char = 21
        mock_ent2.end_char = 27
        
        mock_doc.ents = [mock_ent1, mock_ent2]
        knowledge_extractor.nlp.return_value = mock_doc
        
        # Mock NER pipeline results
        knowledge_extractor.ner_pipeline.return_value = [
            {
                "word": "volcanic",
                "entity_group": "PROCESS",
                "start": 50,
                "end": 58,
                "score": 0.9
            }
        ]
        
        # Mock sentence transformer for deduplication
        knowledge_extractor.sentence_transformer.encode.return_value = [
            [0.1, 0.2, 0.3],  # granite embedding
            [0.4, 0.5, 0.6],  # quartz embedding
            [0.7, 0.8, 0.9]   # volcanic embedding
        ]
        
        entities = await knowledge_extractor._extract_entities(text)
        
        assert len(entities) >= 2
        assert any(entity["text"] == "granite" for entity in entities)
        assert any(entity["text"] == "quartz" for entity in entities)
    
    @pytest.mark.asyncio
    async def test_extract_concepts(self, knowledge_extractor):
        """Test concept extraction from text"""
        text = "Magma is defined as molten rock beneath the Earth's surface. Lava refers to magma that has reached the surface."
        
        concepts = await knowledge_extractor._extract_concepts(text)
        
        assert len(concepts) >= 1
        # Should find at least one concept definition
        concept_names = [concept["name"] for concept in concepts]
        assert any("Magma" in name or "Lava" in name for name in concept_names)
    
    @pytest.mark.asyncio
    async def test_extract_temporal_facts(self, knowledge_extractor):
        """Test temporal fact extraction"""
        text = "The eruption occurred 2.5 Ma ago. During the Pleistocene, volcanic activity increased."
        
        temporal_facts = await knowledge_extractor._extract_temporal_facts(text)
        
        assert len(temporal_facts) >= 1
        # Should find temporal markers
        temporal_markers = [fact["temporal_marker"] for fact in temporal_facts]
        assert any("2.5" in marker or "Pleistocene" in marker for marker in temporal_markers)
    
    @pytest.mark.asyncio
    async def test_extract_from_paper(self, knowledge_extractor, sample_paper):
        """Test complete knowledge extraction from a paper"""
        
        # Mock all the extraction methods
        knowledge_extractor._extract_entities = Mock(return_value=[
            {"text": "volcano", "label": "GEOLOGICAL_FEATURE", "confidence": 0.9}
        ])
        knowledge_extractor._extract_relationships = Mock(return_value=[
            {"source": "seismic", "target": "eruption", "relationship_type": "causal", "confidence": 0.8}
        ])
        knowledge_extractor._extract_concepts = Mock(return_value=[
            {"name": "seismic precursor", "definition": "earthquake activity before eruption", "confidence": 0.7}
        ])
        knowledge_extractor._extract_temporal_facts = Mock(return_value=[
            {"temporal_marker": "2-3 weeks", "event_description": "before eruption", "confidence": 0.6}
        ])
        knowledge_extractor._calculate_confidence_scores = Mock(return_value={
            "entities": 0.9, "relationships": 0.8, "concepts": 0.7, "temporal_facts": 0.6, "overall": 0.75
        })
        
        extracted_knowledge = await knowledge_extractor.extract_from_paper(sample_paper)
        
        assert isinstance(extracted_knowledge, ExtractedKnowledge)
        assert len(extracted_knowledge.entities) == 1
        assert len(extracted_knowledge.relationships) == 1
        assert len(extracted_knowledge.concepts) == 1
        assert len(extracted_knowledge.temporal_facts) == 1
        assert extracted_knowledge.confidence_scores["overall"] == 0.75
        assert extracted_knowledge.source_metadata["doi"] == sample_paper.doi
    
    @pytest.mark.asyncio
    async def test_extract_from_multiple_sources(self, knowledge_extractor):
        """Test extraction from multiple papers"""
        
        papers = [
            ScientificPaper(
                title="Paper 1", authors=["Author 1"], abstract="Abstract 1",
                full_text="Text 1", doi="10.1000/paper1", publication_date=datetime.now(),
                journal="Journal 1", keywords=[], citations=[]
            ),
            ScientificPaper(
                title="Paper 2", authors=["Author 2"], abstract="Abstract 2",
                full_text="Text 2", doi="10.1000/paper2", publication_date=datetime.now(),
                journal="Journal 2", keywords=[], citations=[]
            )
        ]
        
        # Mock the extract_from_paper method
        mock_extraction = ExtractedKnowledge(
            entities=[], relationships=[], concepts=[], temporal_facts=[],
            confidence_scores={"overall": 0.8}, source_metadata={"doi": "test"}
        )
        knowledge_extractor.extract_from_paper = Mock(return_value=mock_extraction)
        
        results = await knowledge_extractor.extract_from_multiple_sources(papers)
        
        assert len(results) == 2
        assert all(isinstance(result, ExtractedKnowledge) for result in results)
    
    def test_deduplicate_entities(self, knowledge_extractor):
        """Test entity deduplication"""
        entities = [
            {"text": "volcano", "confidence": 0.9},
            {"text": "volcanic", "confidence": 0.8},  # Similar to volcano
            {"text": "earthquake", "confidence": 0.7}
        ]
        
        # Mock sentence transformer
        knowledge_extractor.sentence_transformer.encode.return_value = [
            [0.1, 0.2, 0.3],  # volcano
            [0.1, 0.2, 0.35], # volcanic (similar)
            [0.8, 0.9, 1.0]   # earthquake (different)
        ]
        
        deduplicated = knowledge_extractor._deduplicate_entities(entities)
        
        # Should keep the highest confidence entity from similar cluster
        assert len(deduplicated) <= len(entities)
        confidences = [entity["confidence"] for entity in deduplicated]
        assert max(confidences) == 0.9  # Should keep the volcano entity


if __name__ == "__main__":
    pytest.main([__file__])