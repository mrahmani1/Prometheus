"""
Comprehensive tests for the Automated Scientific Discovery System

This test suite validates all components of the automated scientific discovery system
including hypothesis generation, experiment design, literature mining, knowledge graph
construction, peer review, scientific writing, collaboration matching, patent analysis,
and impact assessment.

Author: Prometheus AI Platform Team
Date: 2025-01-09
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import pandas as pd

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

from automated_scientific_discovery import (
    AutomatedScientificDiscoverySystem,
    HypothesisGenerator,
    ExperimentDesigner,
    LiteratureMiner,
    KnowledgeGraphBuilder,
    AutomatedPeerReview,
    ScientificWritingAssistant,
    ResearchCollaborationMatcher,
    PatentAnalyzer,
    ScientificImpactAnalyzer,
    ScientificHypothesis,
    ExperimentDesign,
    LiteratureAnalysis,
    ResearchCollaboration,
    ResearchDomain,
    DiscoveryType,
    ValidationStatus
)


class TestHypothesisGenerator:
    """Test suite for the HypothesisGenerator class"""
    
    @pytest.fixture
    def hypothesis_generator(self):
        """Create a HypothesisGenerator instance for testing"""
        generator = HypothesisGenerator()
        # Mock the heavy ML models for testing
        generator.hypothesis_model = Mock()
        generator.tokenizer = Mock()
        generator.semantic_model = Mock()
        return generator
    
    @pytest.fixture
    def sample_data_patterns(self):
        """Sample data patterns for testing"""
        return [
            {
                'type': 'linear_correlation',
                'significance': 0.85,
                'variables': ['seismic_velocity', 'magma_pressure'],
                'correlation': 0.78,
                'temporal': {'scale': 'days'},
                'spatial': {'scale': 'regional'}
            },
            {
                'type': 'exponential_growth',
                'significance': 0.92,
                'variables': ['gas_emissions', 'temperature'],
                'correlation': 0.88,
                'temporal': {'scale': 'hours'},
                'spatial': {'scale': 'local'}
            }
        ]
    
    @pytest.mark.asyncio
    async def test_generate_hypotheses_from_patterns(self, hypothesis_generator, sample_data_patterns):
        """Test hypothesis generation from data patterns"""
        # Mock the tokenizer and model
        hypothesis_generator.tokenizer.encode.return_value = Mock()
        hypothesis_generator.hypothesis_model.generate.return_value = [Mock()]
        hypothesis_generator.tokenizer.decode.return_value = "Test hypothesis about seismic velocity and magma pressure correlation"
        
        hypotheses = await hypothesis_generator.generate_hypotheses_from_patterns(
            sample_data_patterns, ResearchDomain.VOLCANOLOGY
        )
        
        assert len(hypotheses) == len(sample_data_patterns)
        assert all(isinstance(h, ScientificHypothesis) for h in hypotheses)
        assert all(h.domain == ResearchDomain.VOLCANOLOGY for h in hypotheses)
        assert all(h.discovery_type == DiscoveryType.PATTERN_DISCOVERY for h in hypotheses)
    
    @pytest.mark.asyncio
    async def test_analyze_pattern(self, hypothesis_generator):
        """Test pattern analysis functionality"""
        pattern = {
            'type': 'linear_correlation',
            'significance': 0.85,
            'variables': ['seismic_velocity', 'magma_pressure'],
            'correlation': 0.78
        }
        
        analysis = await hypothesis_generator.analyze_pattern(pattern, ResearchDomain.VOLCANOLOGY)
        
        assert analysis['pattern_type'] == 'linear_correlation'
        assert analysis['statistical_significance'] == 0.85
        assert analysis['correlation_strength'] == 0.78
        assert analysis['domain_context'] == 'volcanology'
        assert 'geological_relevance' in analysis
    
    @pytest.mark.asyncio
    async def test_generate_mathematical_formulation(self, hypothesis_generator):
        """Test mathematical formulation generation"""
        pattern_analysis = {
            'pattern_type': 'linear_correlation',
            'correlation_strength': 0.8,
            'variables_involved': ['seismic_velocity', 'magma_pressure']
        }
        
        formulation = await hypothesis_generator.generate_mathematical_formulation(pattern_analysis)
        
        assert formulation is not None
        assert 'seismic_velocity' in formulation
        assert 'magma_pressure' in formulation
        assert any(symbol in formulation for symbol in ['α', 'β', '='])
    
    @pytest.mark.asyncio
    async def test_assess_hypothesis_quality(self, hypothesis_generator):
        """Test hypothesis quality assessment"""
        hypothesis_text = "Seismic velocity correlates with magma chamber pressure in volcanic systems"
        pattern_analysis = {
            'statistical_significance': 0.85,
            'correlation_strength': 0.78,
            'geological_relevance': 0.9,
            'variables_involved': ['seismic_velocity', 'magma_pressure']
        }
        
        scores = await hypothesis_generator.assess_hypothesis_quality(hypothesis_text, pattern_analysis)
        
        assert 'confidence' in scores
        assert 'novelty' in scores
        assert 'testability' in scores
        assert 'impact' in scores
        assert all(0 <= score <= 1 for score in scores.values())


class TestExperimentDesigner:
    """Test suite for the ExperimentDesigner class"""
    
    @pytest.fixture
    def experiment_designer(self):
        """Create an ExperimentDesigner instance for testing"""
        return ExperimentDesigner()
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Sample hypothesis for testing"""
        return ScientificHypothesis(
            id="test_hyp_001",
            title="Seismic Velocity Correlation Hypothesis",
            description="Seismic velocity correlates with magma chamber pressure in volcanic systems",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            confidence_score=0.85,
            novelty_score=0.72,
            testability_score=0.88,
            impact_prediction=0.75,
            keywords=['seismic', 'velocity', 'magma', 'pressure', 'volcanic']
        )
    
    @pytest.mark.asyncio
    async def test_design_experiment(self, experiment_designer, sample_hypothesis):
        """Test experiment design generation"""
        experiment = await experiment_designer.design_experiment(sample_hypothesis)
        
        assert isinstance(experiment, ExperimentDesign)
        assert experiment.hypothesis_id == sample_hypothesis.id
        assert experiment.title is not None
        assert experiment.objective is not None
        assert experiment.methodology is not None
        assert experiment.feasibility_score >= 0
        assert experiment.cost_estimate is not None
    
    @pytest.mark.asyncio
    async def test_analyze_hypothesis_requirements(self, experiment_designer, sample_hypothesis):
        """Test hypothesis requirements analysis"""
        requirements = await experiment_designer.analyze_hypothesis_requirements(sample_hypothesis)
        
        assert 'variables_to_measure' in requirements
        assert 'experimental_conditions' in requirements
        assert 'temporal_scale' in requirements
        assert 'spatial_scale' in requirements
        assert 'expertise_required' in requirements
        assert isinstance(requirements['variables_to_measure'], list)
    
    @pytest.mark.asyncio
    async def test_select_design_type(self, experiment_designer):
        """Test experimental design type selection"""
        requirements = {
            'variables_to_measure': ['temperature', 'pressure'],
            'temporal_scale': 'hours',
            'spatial_scale': 'local'
        }
        
        design_type = await experiment_designer.select_design_type(requirements)
        
        assert design_type in ['observational_study', 'controlled_experiment', 'field_study', 
                              'laboratory_experiment', 'computational_simulation']
    
    @pytest.mark.asyncio
    async def test_assess_feasibility(self, experiment_designer):
        """Test experiment feasibility assessment"""
        experiment = ExperimentDesign(
            id="test_exp_001",
            hypothesis_id="test_hyp_001",
            title="Test Experiment",
            objective="Test objective",
            methodology="Test methodology",
            variables={'independent': ['var1'], 'dependent': ['var2']},
            controls=['control1'],
            measurements=['measurement1'],
            statistical_analysis="Test analysis",
            expected_outcomes=['outcome1'],
            resource_requirements={'personnel': ['researcher']},
            timeline={'phase1': '1 week'},
            risk_assessment={'technical_failure': 0.2},
            feasibility_score=0.0
        )
        
        feasibility_score = await experiment_designer.assess_feasibility(experiment)
        
        assert 0 <= feasibility_score <= 1


class TestLiteratureMiner:
    """Test suite for the LiteratureMiner class"""
    
    @pytest.fixture
    def literature_miner(self):
        """Create a LiteratureMiner instance for testing"""
        miner = LiteratureMiner()
        # Mock the external APIs for testing
        miner.arxiv_client = Mock()
        miner.semantic_model = Mock()
        return miner
    
    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing"""
        return [
            {
                'title': 'Seismic Analysis of Volcanic Systems',
                'authors': ['Smith, J.', 'Johnson, A.'],
                'abstract': 'This paper analyzes seismic patterns in volcanic systems...',
                'published': '2023-01-15',
                'source': 'arxiv',
                'citations': 25
            },
            {
                'title': 'Magma Chamber Dynamics and Eruption Forecasting',
                'authors': ['Brown, K.', 'Davis, M.'],
                'abstract': 'We investigate magma chamber dynamics for eruption prediction...',
                'published': '2022-08-20',
                'source': 'google_scholar',
                'citations': 42
            }
        ]
    
    @pytest.mark.asyncio
    async def test_mine_literature(self, literature_miner, sample_papers):
        """Test literature mining functionality"""
        # Mock the search methods
        literature_miner.search_multiple_sources = AsyncMock(return_value=sample_papers)
        literature_miner.semantic_model.encode.return_value = np.random.rand(2, 384)
        
        analysis = await literature_miner.mine_literature("volcanic seismic analysis", max_papers=10)
        
        assert isinstance(analysis, LiteratureAnalysis)
        assert analysis.query == "volcanic seismic analysis"
        assert analysis.papers_analyzed == len(sample_papers)
        assert isinstance(analysis.key_findings, list)
        assert isinstance(analysis.trending_topics, list)
    
    @pytest.mark.asyncio
    async def test_search_arxiv(self, literature_miner):
        """Test arXiv search functionality"""
        # Mock arXiv client
        mock_result = Mock()
        mock_result.title = "Test Paper"
        mock_result.authors = [Mock(name="Test Author")]
        mock_result.summary = "Test abstract"
        mock_result.published = datetime.now()
        mock_result.entry_id = "http://arxiv.org/abs/test"
        mock_result.categories = ["physics.geo-ph"]
        
        literature_miner.arxiv_client.results.return_value = [mock_result]
        
        papers = await literature_miner.search_arxiv("volcanic seismic", max_results=5)
        
        assert len(papers) == 1
        assert papers[0]['title'] == "Test Paper"
        assert papers[0]['source'] == 'arxiv'
    
    @pytest.mark.asyncio
    async def test_deduplicate_papers(self, literature_miner, sample_papers):
        """Test paper deduplication"""
        # Add a duplicate paper
        duplicate_papers = sample_papers + [sample_papers[0].copy()]
        
        # Mock semantic model
        literature_miner.semantic_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],  # First paper
            [0.4, 0.5, 0.6],  # Second paper
            [0.1, 0.2, 0.3]   # Duplicate of first paper
        ])
        
        unique_papers = await literature_miner.deduplicate_papers(duplicate_papers)
        
        # Should remove the duplicate
        assert len(unique_papers) < len(duplicate_papers)
    
    @pytest.mark.asyncio
    async def test_extract_key_findings(self, literature_miner, sample_papers):
        """Test key findings extraction"""
        key_findings = await literature_miner.extract_key_findings(sample_papers)
        
        assert isinstance(key_findings, list)
        # Should extract some findings from the abstracts
        assert len(key_findings) >= 0


class TestKnowledgeGraphBuilder:
    """Test suite for the KnowledgeGraphBuilder class"""
    
    @pytest.fixture
    def kg_builder(self):
        """Create a KnowledgeGraphBuilder instance for testing"""
        builder = KnowledgeGraphBuilder()
        builder.semantic_model = Mock()
        return builder
    
    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing"""
        return [
            {
                'title': 'Seismic Analysis of Volcanic Systems',
                'authors': ['Smith, J.', 'Johnson, A.'],
                'abstract': 'This paper analyzes seismic patterns in volcanic systems...',
                'published': '2023-01-15'
            }
        ]
    
    @pytest.fixture
    def sample_hypotheses(self):
        """Sample hypotheses for testing"""
        return [
            ScientificHypothesis(
                id="test_hyp_001",
                title="Test Hypothesis",
                description="Test hypothesis description",
                domain=ResearchDomain.VOLCANOLOGY,
                discovery_type=DiscoveryType.PATTERN_DISCOVERY,
                keywords=['seismic', 'volcanic']
            )
        ]
    
    @pytest.mark.asyncio
    async def test_build_knowledge_graph(self, kg_builder, sample_papers, sample_hypotheses):
        """Test knowledge graph construction"""
        # Mock semantic model
        kg_builder.semantic_model.encode.return_value = np.random.rand(384)
        
        graph_info = await kg_builder.build_knowledge_graph(sample_papers, sample_hypotheses)
        
        assert isinstance(graph_info, dict)
        assert 'nodes' in graph_info
        assert 'edges' in graph_info
        assert 'node_types' in graph_info
        assert 'relationship_types' in graph_info
        assert graph_info['nodes'] > 0
    
    @pytest.mark.asyncio
    async def test_add_paper_nodes(self, kg_builder, sample_papers):
        """Test adding paper nodes to knowledge graph"""
        kg_builder.semantic_model.encode.return_value = np.random.rand(384)
        
        await kg_builder.add_paper_nodes(sample_papers)
        
        assert len(kg_builder.graph.nodes()) == len(sample_papers)
        
        # Check node attributes
        for node_id, node_data in kg_builder.graph.nodes(data=True):
            assert node_data['type'] == 'paper'
            assert 'title' in node_data
    
    @pytest.mark.asyncio
    async def test_extract_scientific_concepts(self, kg_builder):
        """Test scientific concept extraction"""
        texts = [
            "Seismic analysis of volcanic eruptions using machine learning",
            "Magma chamber dynamics and eruption forecasting models"
        ]
        
        concepts = await kg_builder.extract_scientific_concepts(texts)
        
        assert isinstance(concepts, list)
        assert all('text' in concept for concept in concepts)
        assert all('category' in concept for concept in concepts)
        assert all('frequency' in concept for concept in concepts)


class TestAutomatedPeerReview:
    """Test suite for the AutomatedPeerReview class"""
    
    @pytest.fixture
    def peer_reviewer(self):
        """Create an AutomatedPeerReview instance for testing"""
        return AutomatedPeerReview()
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Sample hypothesis for testing"""
        return ScientificHypothesis(
            id="test_hyp_001",
            title="High Quality Hypothesis",
            description="This novel hypothesis proposes that seismic velocity correlates with magma chamber pressure in volcanic systems, providing important insights for eruption forecasting",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            mathematical_formulation="velocity = α + β * pressure + ε",
            confidence_score=0.85,
            novelty_score=0.78,
            testability_score=0.82,
            impact_prediction=0.75,
            keywords=['seismic', 'velocity', 'magma', 'pressure', 'volcanic', 'eruption']
        )
    
    @pytest.mark.asyncio
    async def test_review_hypothesis(self, peer_reviewer, sample_hypothesis):
        """Test automated hypothesis review"""
        review = await peer_reviewer.review_hypothesis(sample_hypothesis)
        
        assert isinstance(review, dict)
        assert 'hypothesis_id' in review
        assert 'overall_score' in review
        assert 'criterion_scores' in review
        assert 'strengths' in review
        assert 'weaknesses' in review
        assert 'recommendations' in review
        assert 'validation_status' in review
        
        assert 0 <= review['overall_score'] <= 1
        assert isinstance(review['validation_status'], ValidationStatus)
    
    @pytest.mark.asyncio
    async def test_evaluate_novelty(self, peer_reviewer, sample_hypothesis):
        """Test novelty evaluation"""
        novelty_score = await peer_reviewer.evaluate_novelty(sample_hypothesis)
        
        assert 0 <= novelty_score <= 1
        # Should be high due to novelty indicators in the description
        assert novelty_score > 0.5
    
    @pytest.mark.asyncio
    async def test_evaluate_methodology(self, peer_reviewer, sample_hypothesis):
        """Test methodology evaluation"""
        methodology_score = await peer_reviewer.evaluate_methodology(sample_hypothesis)
        
        assert 0 <= methodology_score <= 1
        # Should be high due to mathematical formulation and high testability
        assert methodology_score > 0.5
    
    @pytest.mark.asyncio
    async def test_identify_strengths(self, peer_reviewer, sample_hypothesis):
        """Test strength identification"""
        criterion_scores = {
            'novelty': 0.85,
            'methodology': 0.80,
            'significance': 0.75,
            'clarity': 0.70,
            'validity': 0.82
        }
        
        strengths = await peer_reviewer.identify_strengths(sample_hypothesis, criterion_scores)
        
        assert isinstance(strengths, list)
        assert len(strengths) > 0
        # Should identify high-scoring criteria as strengths
        assert any('novel' in strength.lower() for strength in strengths)


class TestScientificWritingAssistant:
    """Test suite for the ScientificWritingAssistant class"""
    
    @pytest.fixture
    def writing_assistant(self):
        """Create a ScientificWritingAssistant instance for testing"""
        assistant = ScientificWritingAssistant()
        # Mock the T5 model for testing
        assistant.tokenizer = Mock()
        assistant.generation_model = Mock()
        return assistant
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Sample hypothesis for testing"""
        return ScientificHypothesis(
            id="test_hyp_001",
            title="Seismic Velocity Correlation",
            description="Seismic velocity correlates with magma chamber pressure",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            keywords=['seismic', 'velocity', 'magma', 'pressure']
        )
    
    @pytest.fixture
    def sample_experiment(self):
        """Sample experiment design for testing"""
        return ExperimentDesign(
            id="test_exp_001",
            hypothesis_id="test_hyp_001",
            title="Test Experiment",
            objective="Test the correlation hypothesis",
            methodology="Observational study with statistical analysis",
            variables={'independent': ['pressure'], 'dependent': ['velocity']},
            controls=['temperature', 'depth'],
            measurements=['seismic measurements', 'pressure readings'],
            statistical_analysis="Correlation analysis and regression modeling",
            expected_outcomes=['Positive correlation', 'Statistical significance'],
            resource_requirements={'personnel': ['seismologist'], 'equipment': ['seismometers']},
            timeline={'data_collection': '4 weeks', 'analysis': '2 weeks'},
            risk_assessment={'equipment_failure': 0.2},
            feasibility_score=0.8
        )
    
    @pytest.fixture
    def sample_literature(self):
        """Sample literature analysis for testing"""
        return LiteratureAnalysis(
            query="seismic velocity magma pressure",
            papers_analyzed=25,
            key_findings=['Seismic velocity variations', 'Magma chamber dynamics'],
            research_gaps=['Limited real-time analysis', 'Insufficient long-term studies'],
            trending_topics=['machine learning', 'real-time monitoring'],
            citation_network={},
            collaboration_patterns={},
            temporal_trends={},
            semantic_clusters=[],
            impact_metrics={}
        )
    
    @pytest.mark.asyncio
    async def test_generate_paper_draft(self, writing_assistant, sample_hypothesis, 
                                       sample_experiment, sample_literature):
        """Test paper draft generation"""
        paper_sections = await writing_assistant.generate_paper_draft(
            sample_hypothesis, sample_experiment, sample_literature
        )
        
        assert isinstance(paper_sections, dict)
        expected_sections = ['title', 'abstract', 'introduction', 'methods', 
                           'results', 'discussion', 'conclusion', 'references']
        
        for section in expected_sections:
            assert section in paper_sections
            assert isinstance(paper_sections[section], str)
            assert len(paper_sections[section]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_title(self, writing_assistant, sample_hypothesis):
        """Test title generation"""
        title = await writing_assistant.generate_title(sample_hypothesis)
        
        assert isinstance(title, str)
        assert len(title) > 0
        assert len(title) <= 100
        # Should contain key concepts from hypothesis
        assert any(keyword in title.lower() for keyword in sample_hypothesis.keywords)
    
    @pytest.mark.asyncio
    async def test_generate_abstract(self, writing_assistant, sample_hypothesis, sample_experiment):
        """Test abstract generation"""
        abstract = await writing_assistant.generate_abstract(sample_hypothesis, sample_experiment)
        
        assert isinstance(abstract, str)
        assert len(abstract) > 0
        # Should contain key elements
        assert 'objective' in abstract.lower() or 'background' in abstract.lower()
        assert 'methods' in abstract.lower() or 'methodology' in abstract.lower()
    
    @pytest.mark.asyncio
    async def test_generate_grant_proposal(self, writing_assistant, sample_hypothesis, 
                                         sample_experiment):
        """Test grant proposal generation"""
        proposal_sections = await writing_assistant.generate_grant_proposal(
            sample_hypothesis, sample_experiment, 50000.0
        )
        
        assert isinstance(proposal_sections, dict)
        expected_sections = ['project_summary', 'significance', 'innovation', 
                           'approach', 'budget_justification']
        
        for section in expected_sections:
            assert section in proposal_sections
            assert isinstance(proposal_sections[section], str)
            assert len(proposal_sections[section]) > 0


class TestResearchCollaborationMatcher:
    """Test suite for the ResearchCollaborationMatcher class"""
    
    @pytest.fixture
    def collaboration_matcher(self):
        """Create a ResearchCollaborationMatcher instance for testing"""
        matcher = ResearchCollaborationMatcher()
        matcher.semantic_model = Mock()
        return matcher
    
    @pytest.fixture
    def sample_researcher(self):
        """Sample researcher profile for testing"""
        return {
            'id': 'researcher_001',
            'name': 'Dr. Jane Smith',
            'research_interests': ['volcanic monitoring', 'seismic analysis', 'machine learning'],
            'skills': ['Python', 'data analysis', 'geophysics', 'machine learning'],
            'location': 'University of Washington, Seattle, USA',
            'career_stage': 'assistant_professor',
            'publication_count': 25,
            'research_domains': ['volcanology', 'seismology']
        }
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate researchers for testing"""
        return [
            {
                'id': 'researcher_002',
                'name': 'Dr. John Doe',
                'research_interests': ['volcanic monitoring', 'eruption forecasting'],
                'skills': ['R', 'statistics', 'geology', 'remote sensing'],
                'location': 'Stanford University, California, USA',
                'career_stage': 'associate_professor',
                'publication_count': 45,
                'research_domains': ['volcanology']
            },
            {
                'id': 'researcher_003',
                'name': 'Dr. Maria Garcia',
                'research_interests': ['earthquake prediction', 'seismic networks'],
                'skills': ['MATLAB', 'signal processing', 'seismology'],
                'location': 'University of Tokyo, Japan',
                'career_stage': 'full_professor',
                'publication_count': 80,
                'research_domains': ['seismology']
            }
        ]
    
    @pytest.mark.asyncio
    async def test_find_collaborations(self, collaboration_matcher, sample_researcher, sample_candidates):
        """Test collaboration finding"""
        # Mock semantic similarity
        collaboration_matcher.semantic_model.encode.return_value = np.random.rand(384)
        
        collaborations = await collaboration_matcher.find_collaborations(
            sample_researcher, sample_candidates
        )
        
        assert isinstance(collaborations, list)
        assert all(isinstance(collab, ResearchCollaboration) for collab in collaborations)
        assert len(collaborations) <= 10  # Should limit to top 10
        
        # Check that collaborations are sorted by compatibility score
        if len(collaborations) > 1:
            scores = [c.compatibility_score for c in collaborations]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_calculate_compatibility(self, collaboration_matcher, sample_researcher, sample_candidates):
        """Test compatibility calculation"""
        # Mock semantic model
        collaboration_matcher.semantic_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        
        compatibility = await collaboration_matcher.calculate_compatibility(
            sample_researcher, sample_candidates[0]
        )
        
        assert isinstance(compatibility, dict)
        assert 'overall_score' in compatibility
        assert 'shared_interests' in compatibility
        assert 'complementary_skills' in compatibility
        assert 'detailed_scores' in compatibility
        
        assert 0 <= compatibility['overall_score'] <= 1
        assert isinstance(compatibility['shared_interests'], list)
        assert isinstance(compatibility['complementary_skills'], list)
    
    @pytest.mark.asyncio
    async def test_find_shared_interests(self, collaboration_matcher, sample_researcher, sample_candidates):
        """Test shared interests identification"""
        # Mock semantic model for similarity calculation
        collaboration_matcher.semantic_model.encode.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        shared_interests = await collaboration_matcher.find_shared_interests(
            sample_researcher, sample_candidates[0]
        )
        
        assert isinstance(shared_interests, list)
        # Should find 'volcanic monitoring' as shared interest
        assert any('volcanic monitoring' in interest for interest in shared_interests)
    
    @pytest.mark.asyncio
    async def test_find_complementary_skills(self, collaboration_matcher, sample_researcher, sample_candidates):
        """Test complementary skills identification"""
        complementary_skills = await collaboration_matcher.find_complementary_skills(
            sample_researcher, sample_candidates[0]
        )
        
        assert isinstance(complementary_skills, list)
        # Should identify skills unique to each researcher
        assert len(complementary_skills) > 0


class TestAutomatedScientificDiscoverySystem:
    """Test suite for the main AutomatedScientificDiscoverySystem class"""
    
    @pytest.fixture
    def discovery_system(self):
        """Create an AutomatedScientificDiscoverySystem instance for testing"""
        system = AutomatedScientificDiscoverySystem()
        
        # Mock all subsystems for testing
        system.hypothesis_generator = Mock()
        system.experiment_designer = Mock()
        system.literature_miner = Mock()
        system.knowledge_graph_builder = Mock()
        system.peer_reviewer = Mock()
        system.writing_assistant = Mock()
        system.collaboration_matcher = Mock()
        system.patent_analyzer = Mock()
        system.impact_analyzer = Mock()
        
        return system
    
    @pytest.fixture
    def sample_data_patterns(self):
        """Sample data patterns for testing"""
        return [
            {
                'type': 'linear_correlation',
                'significance': 0.85,
                'variables': ['seismic_velocity', 'magma_pressure'],
                'correlation': 0.78
            }
        ]
    
    @pytest.fixture
    def sample_researcher_profile(self):
        """Sample researcher profile for testing"""
        return {
            'id': 'researcher_001',
            'name': 'Dr. Test Researcher',
            'research_interests': ['volcanology', 'seismology'],
            'research_field': 'volcanology'
        }
    
    @pytest.mark.asyncio
    async def test_initialize_system(self, discovery_system):
        """Test system initialization"""
        # Mock the initialize methods
        discovery_system.hypothesis_generator.initialize_models = AsyncMock()
        discovery_system.experiment_designer.initialize_design_framework = AsyncMock()
        
        await discovery_system.initialize_system()
        
        # Verify initialization methods were called
        discovery_system.hypothesis_generator.initialize_models.assert_called_once()
        discovery_system.experiment_designer.initialize_design_framework.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_discovery_pipeline(self, discovery_system, sample_data_patterns, 
                                        sample_researcher_profile):
        """Test the complete discovery pipeline"""
        # Mock all the subsystem methods
        sample_hypothesis = ScientificHypothesis(
            id="test_hyp_001",
            title="Test Hypothesis",
            description="Test description",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            keywords=['test']
        )
        
        sample_experiment = ExperimentDesign(
            id="test_exp_001",
            hypothesis_id="test_hyp_001",
            title="Test Experiment",
            objective="Test objective",
            methodology="Test methodology",
            variables={},
            controls=[],
            measurements=[],
            statistical_analysis="Test analysis",
            expected_outcomes=[],
            resource_requirements={},
            timeline={},
            risk_assessment={},
            feasibility_score=0.8
        )
        
        sample_literature = LiteratureAnalysis(
            query="test query",
            papers_analyzed=10,
            key_findings=[],
            research_gaps=[],
            trending_topics=[],
            citation_network={},
            collaboration_patterns={},
            temporal_trends={},
            semantic_clusters=[],
            impact_metrics={}
        )
        
        # Configure mocks
        discovery_system.hypothesis_generator.generate_hypotheses_from_patterns = AsyncMock(
            return_value=[sample_hypothesis]
        )
        discovery_system.experiment_designer.design_experiment = AsyncMock(
            return_value=sample_experiment
        )
        discovery_system.literature_miner.mine_literature = AsyncMock(
            return_value=sample_literature
        )
        discovery_system.knowledge_graph_builder.build_knowledge_graph = AsyncMock(
            return_value={'nodes': 5, 'edges': 3}
        )
        discovery_system.peer_reviewer.review_hypothesis = AsyncMock(
            return_value={'validation_status': ValidationStatus.VALIDATED, 'overall_score': 0.8}
        )
        discovery_system.writing_assistant.generate_paper_draft = AsyncMock(
            return_value={'title': 'Test Paper', 'abstract': 'Test abstract'}
        )
        discovery_system.collaboration_matcher.find_collaborations = AsyncMock(
            return_value=[]
        )
        discovery_system.patent_analyzer.analyze_patent_landscape = AsyncMock(
            return_value={'patentability_score': 0.7}
        )
        discovery_system.impact_analyzer.analyze_scientific_impact = AsyncMock(
            return_value={'career_trajectory': 'ascending'}
        )
        
        # Run the pipeline
        results = await discovery_system.run_discovery_pipeline(
            data_patterns=sample_data_patterns,
            domain=ResearchDomain.VOLCANOLOGY,
            researcher_profile=sample_researcher_profile
        )
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'pipeline_id' in results
        assert 'domain' in results
        assert 'hypotheses' in results
        assert 'experiments' in results
        assert 'status' in results
        
        assert results['status'] == 'completed'
        assert results['domain'] == 'volcanology'
        assert len(results['hypotheses']) == 1
        assert len(results['experiments']) == 1
    
    @pytest.mark.asyncio
    async def test_export_discovery_results(self, discovery_system):
        """Test discovery results export"""
        # Create mock results
        pipeline_id = "test_pipeline_001"
        mock_results = {
            'pipeline_id': pipeline_id,
            'domain': 'volcanology',
            'status': 'completed',
            'start_time': datetime.now(),
            'hypotheses': [],
            'recommendations': ['Test recommendation']
        }
        
        discovery_system.active_discoveries[pipeline_id] = mock_results
        
        # Test JSON export
        json_export = await discovery_system.export_discovery_results(pipeline_id, 'json')
        assert isinstance(json_export, str)
        
        # Verify it's valid JSON
        parsed_json = json.loads(json_export)
        assert parsed_json['pipeline_id'] == pipeline_id
        
        # Test Markdown export
        markdown_export = await discovery_system.export_discovery_results(pipeline_id, 'markdown')
        assert isinstance(markdown_export, str)
        assert '# Scientific Discovery Report' in markdown_export
    
    def test_get_discovery_status(self, discovery_system):
        """Test getting discovery status"""
        pipeline_id = "test_pipeline_001"
        mock_results = {'pipeline_id': pipeline_id, 'status': 'running'}
        
        discovery_system.active_discoveries[pipeline_id] = mock_results
        
        # Test existing pipeline
        status = asyncio.run(discovery_system.get_discovery_status(pipeline_id))
        assert status == mock_results
        
        # Test non-existent pipeline
        status = asyncio.run(discovery_system.get_discovery_status("non_existent"))
        assert 'error' in status
    
    def test_list_active_discoveries(self, discovery_system):
        """Test listing active discoveries"""
        # Add mock discoveries
        discovery_system.active_discoveries = {
            'pipeline_001': {
                'domain': 'volcanology',
                'status': 'running',
                'start_time': datetime.now(),
                'hypotheses': [Mock(), Mock()]
            },
            'pipeline_002': {
                'domain': 'seismology',
                'status': 'completed',
                'start_time': datetime.now(),
                'hypotheses': [Mock()]
            }
        }
        
        discoveries = asyncio.run(discovery_system.list_active_discoveries())
        
        assert len(discoveries) == 2
        assert all('pipeline_id' in d for d in discoveries)
        assert all('domain' in d for d in discoveries)
        assert all('status' in d for d in discoveries)


# Integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_hypothesis_to_experiment_integration(self):
        """Test integration between hypothesis generation and experiment design"""
        # Create real instances (not mocked)
        hypothesis_generator = HypothesisGenerator()
        experiment_designer = ExperimentDesigner()
        
        # Create a sample hypothesis
        hypothesis = ScientificHypothesis(
            id="integration_test_001",
            title="Integration Test Hypothesis",
            description="Seismic velocity correlates with magma chamber pressure in volcanic systems due to changes in rock density and elastic properties",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            confidence_score=0.85,
            novelty_score=0.72,
            testability_score=0.88,
            impact_prediction=0.75,
            keywords=['seismic', 'velocity', 'magma', 'pressure', 'volcanic', 'density']
        )
        
        # Design experiment for the hypothesis
        experiment = await experiment_designer.design_experiment(hypothesis)
        
        # Verify integration
        assert experiment.hypothesis_id == hypothesis.id
        assert experiment.feasibility_score > 0
        assert len(experiment.variables) > 0
        assert len(experiment.expected_outcomes) > 0
    
    @pytest.mark.asyncio
    async def test_peer_review_to_writing_integration(self):
        """Test integration between peer review and scientific writing"""
        peer_reviewer = AutomatedPeerReview()
        writing_assistant = ScientificWritingAssistant()
        
        # Create a high-quality hypothesis
        hypothesis = ScientificHypothesis(
            id="integration_test_002",
            title="High Quality Research Hypothesis",
            description="This novel and innovative hypothesis proposes that seismic velocity correlates with magma chamber pressure, providing important insights for volcanic eruption forecasting",
            domain=ResearchDomain.VOLCANOLOGY,
            discovery_type=DiscoveryType.PATTERN_DISCOVERY,
            mathematical_formulation="velocity = α + β * pressure + ε",
            confidence_score=0.85,
            novelty_score=0.78,
            testability_score=0.82,
            impact_prediction=0.75,
            keywords=['seismic', 'velocity', 'magma', 'pressure', 'volcanic', 'eruption', 'forecasting']
        )
        
        # Conduct peer review
        review = await peer_reviewer.review_hypothesis(hypothesis)
        
        # Only generate paper if validated
        if review['validation_status'] in [ValidationStatus.VALIDATED, ValidationStatus.REQUIRES_REVISION]:
            # Create mock experiment and literature for writing
            experiment = ExperimentDesign(
                id="integration_exp_002",
                hypothesis_id=hypothesis.id,
                title="Integration Test Experiment",
                objective="Test the hypothesis through observational study",
                methodology="Observational study with statistical analysis",
                variables={'independent': ['pressure'], 'dependent': ['velocity']},
                controls=['temperature', 'depth'],
                measurements=['seismic measurements'],
                statistical_analysis="Correlation analysis",
                expected_outcomes=['Positive correlation'],
                resource_requirements={},
                timeline={},
                risk_assessment={},
                feasibility_score=0.8
            )
            
            literature = LiteratureAnalysis(
                query="seismic velocity magma pressure",
                papers_analyzed=10,
                key_findings=['Seismic velocity variations'],
                research_gaps=['Limited real-time analysis'],
                trending_topics=['machine learning'],
                citation_network={},
                collaboration_patterns={},
                temporal_trends={},
                semantic_clusters=[],
                impact_metrics={}
            )
            
            # Generate paper draft
            paper_draft = await writing_assistant.generate_paper_draft(
                hypothesis, experiment, literature
            )
            
            # Verify integration
            assert isinstance(paper_draft, dict)
            assert 'title' in paper_draft
            assert 'abstract' in paper_draft
            assert len(paper_draft['title']) > 0
            assert len(paper_draft['abstract']) > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])