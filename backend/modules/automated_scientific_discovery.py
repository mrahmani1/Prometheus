"""
Automated Scientific Discovery System for Geological Research

This module implements a comprehensive AI-powered scientific discovery system that automates
hypothesis generation, experiment design, literature analysis, and knowledge synthesis for
geological research. The system uses advanced NLP, machine learning, and knowledge graph
technologies to accelerate scientific discovery in geosciences.

Key Features:
- Automated hypothesis generation from data patterns and causal relationships
- Intelligent experiment design and execution planning
- Scientific literature mining with semantic analysis
- Dynamic knowledge graph construction and reasoning
- Automated peer review and validation systems
- AI-assisted scientific writing and paper generation
- Research collaboration matching and networking
- Grant proposal optimization and generation
- Patent analysis and prior art discovery
- Scientific impact assessment and citation analysis

Author: Prometheus AI Platform Team
Date: 2025-01-09
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, T5ForConditionalGeneration, T5Tokenizer
)
import spacy
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import arxiv
import scholarly
from pylatexenc.latex2text import LatexNodes2Text
import bibtexparser
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscoveryType(Enum):
    """Types of scientific discoveries"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    PATTERN_DISCOVERY = "pattern_discovery"
    CAUSAL_INFERENCE = "causal_inference"
    ANOMALY_DETECTION = "anomaly_detection"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    EXPERIMENTAL_DESIGN = "experimental_design"c
lass ResearchDomain(Enum):
    """Research domains for geological sciences"""
    VOLCANOLOGY = "volcanology"
    SEISMOLOGY = "seismology"
    GEODYNAMICS = "geodynamics"
    PETROLOGY = "petrology"
    GEOCHEMISTRY = "geochemistry"
    STRUCTURAL_GEOLOGY = "structural_geology"
    HYDROGEOLOGY = "hydrogeology"
    GEOPHYSICS = "geophysics"
    MINERALOGY = "mineralogy"
    TECTONICS = "tectonics"

class ValidationStatus(Enum):
    """Status of scientific validation"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REQUIRES_REVISION = "requires_revision"

@dataclass
class ScientificHypothesis:
    """Represents a generated scientific hypothesis"""
    id: str
    title: str
    description: str
    domain: ResearchDomain
    discovery_type: DiscoveryType
    mathematical_formulation: Optional[str] = None
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    novelty_score: float = 0.0
    testability_score: float = 0.0
    impact_prediction: float = 0.0
    generated_timestamp: datetime = field(default_factory=datetime.now)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    experimental_design: Optional[Dict[str, Any]] = None
    related_papers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

@dataclass
class ExperimentDesign:
    """Represents an automated experiment design"""
    id: str
    hypothesis_id: str
    title: str
    objective: str
    methodology: str
    variables: Dict[str, Any]
    controls: List[str]
    measurements: List[str]
    statistical_analysis: str
    expected_outcomes: List[str]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, str]
    risk_assessment: Dict[str, float]
    feasibility_score: float
    cost_estimate: Optional[float] = None

@dataclass
class LiteratureAnalysis:
    """Results of literature mining and analysis"""
    query: str
    papers_analyzed: int
    key_findings: List[str]
    research_gaps: List[str]
    trending_topics: List[str]
    citation_network: Dict[str, Any]
    collaboration_patterns: Dict[str, Any]
    temporal_trends: Dict[str, Any]
    semantic_clusters: List[Dict[str, Any]]
    impact_metrics: Dict[str, float]

@dataclass
class KnowledgeGraphNode:
    """Node in the scientific knowledge graph"""
    id: str
    type: str  # concept, researcher, paper, institution, etc.
    properties: Dict[str, Any]
    embeddings: Optional[np.ndarray] = None

@dataclass
class KnowledgeGraphEdge:
    """Edge in the scientific knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float
    properties: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchCollaboration:
    """Potential research collaboration match"""
    researcher1_id: str
    researcher2_id: str
    compatibility_score: float
    shared_interests: List[str]
    complementary_skills: List[str]
    collaboration_potential: str
    suggested_projects: List[str]
    contact_recommendation: strclass Hy
pothesisGenerator:
    """Advanced AI system for generating scientific hypotheses from data patterns"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.hypothesis_model = None
        self.causal_model = None
        self.pattern_detector = None
        self.geological_knowledge_base = {}
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize AI models for hypothesis generation"""
        try:
            # Initialize T5 model for text generation
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.hypothesis_model = T5ForConditionalGeneration.from_pretrained('t5-base')
            
            # Initialize pattern detection models
            self.pattern_detector = DBSCAN(eps=0.3, min_samples=5)
            
            # Load geological domain knowledge
            self.load_geological_knowledge()
            
            logger.info("Hypothesis generation models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def load_geological_knowledge(self):
        """Load geological domain knowledge and constraints"""
        self.geological_knowledge_base = {
            'physical_laws': [
                'Conservation of mass in geological systems',
                'Conservation of energy in thermal processes',
                'Conservation of momentum in tectonic processes',
                'Thermodynamic equilibrium in mineral assemblages',
                'Darcy\'s law for fluid flow in porous media'
            ],
            'geological_processes': [
                'Magma generation and differentiation',
                'Plate tectonics and continental drift',
                'Volcanic eruption mechanisms',
                'Seismic wave propagation',
                'Hydrothermal circulation',
                'Metamorphic reactions',
                'Sedimentary processes'
            ],
            'temporal_scales': {
                'seconds': ['Seismic events', 'Explosive eruptions'],
                'minutes_hours': ['Volcanic gas emissions', 'Landslides'],
                'days_months': ['Volcanic unrest', 'Earthquake swarms'],
                'years_decades': ['Volcanic cycles', 'Glacial cycles'],
                'millennia': ['Climate change', 'Landscape evolution'],
                'millions_years': ['Plate motion', 'Mountain building']
            },
            'spatial_scales': {
                'microscopic': ['Crystal structure', 'Mineral chemistry'],
                'hand_sample': ['Rock texture', 'Mineral assemblages'],
                'outcrop': ['Geological structures', 'Stratigraphic relations'],
                'regional': ['Fault systems', 'Volcanic fields'],
                'global': ['Plate boundaries', 'Mantle convection']
            }
        }
    
    async def generate_hypotheses_from_patterns(self, 
                                              data_patterns: List[Dict[str, Any]], 
                                              domain: ResearchDomain) -> List[ScientificHypothesis]:
        """Generate scientific hypotheses from identified data patterns"""
        hypotheses = []
        
        for pattern in data_patterns:
            try:
                # Analyze pattern characteristics
                pattern_analysis = await self.analyze_pattern(pattern, domain)
                
                # Generate hypothesis text
                hypothesis_text = await self.generate_hypothesis_text(pattern_analysis, domain)
                
                # Create mathematical formulation if applicable
                math_formulation = await self.generate_mathematical_formulation(pattern_analysis)
                
                # Assess hypothesis quality
                quality_scores = await self.assess_hypothesis_quality(hypothesis_text, pattern_analysis)
                
                # Create hypothesis object
                hypothesis = ScientificHypothesis(
                    id=f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(hypotheses)}",
                    title=await self.generate_hypothesis_title(hypothesis_text),
                    description=hypothesis_text,
                    domain=domain,
                    discovery_type=DiscoveryType.PATTERN_DISCOVERY,
                    mathematical_formulation=math_formulation,
                    confidence_score=quality_scores['confidence'],
                    novelty_score=quality_scores['novelty'],
                    testability_score=quality_scores['testability'],
                    impact_prediction=quality_scores['impact'],
                    keywords=await self.extract_keywords(hypothesis_text)
                )
                
                hypotheses.append(hypothesis)
                
            except Exception as e:
                logger.error(f"Error generating hypothesis from pattern: {e}")
                continue
        
        return hypotheses
    
    async def analyze_pattern(self, pattern: Dict[str, Any], domain: ResearchDomain) -> Dict[str, Any]:
        """Analyze data pattern characteristics for hypothesis generation"""
        analysis = {
            'pattern_type': pattern.get('type', 'unknown'),
            'statistical_significance': pattern.get('significance', 0.0),
            'temporal_characteristics': pattern.get('temporal', {}),
            'spatial_characteristics': pattern.get('spatial', {}),
            'variables_involved': pattern.get('variables', []),
            'correlation_strength': pattern.get('correlation', 0.0),
            'domain_context': domain.value,
            'geological_relevance': 0.0
        }
        
        # Assess geological relevance
        analysis['geological_relevance'] = await self.assess_geological_relevance(pattern, domain)
        
        return analysis
    
    async def generate_hypothesis_text(self, pattern_analysis: Dict[str, Any], domain: ResearchDomain) -> str:
        """Generate natural language hypothesis from pattern analysis"""
        # Create context-aware prompt
        prompt = self.create_hypothesis_prompt(pattern_analysis, domain)
        
        # Generate hypothesis using T5 model
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.hypothesis_model.generate(
                inputs, 
                max_length=200, 
                num_return_sequences=1, 
                temperature=0.7,
                do_sample=True
            )
        
        hypothesis = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process and validate hypothesis
        hypothesis = await self.post_process_hypothesis(hypothesis, domain)
        
        return hypothesis    def c
reate_hypothesis_prompt(self, pattern_analysis: Dict[str, Any], domain: ResearchDomain) -> str:
        """Create context-aware prompt for hypothesis generation"""
        domain_context = self.geological_knowledge_base.get('geological_processes', [])
        relevant_processes = [p for p in domain_context if domain.value.lower() in p.lower()]
        
        prompt = f"""
        Generate a scientific hypothesis based on the following geological data pattern:
        
        Domain: {domain.value}
        Pattern Type: {pattern_analysis['pattern_type']}
        Statistical Significance: {pattern_analysis['statistical_significance']}
        Variables: {', '.join(pattern_analysis['variables_involved'])}
        Correlation Strength: {pattern_analysis['correlation_strength']}
        
        Relevant Geological Processes: {', '.join(relevant_processes[:3])}
        
        The hypothesis should be:
        1. Scientifically plausible and testable
        2. Based on established geological principles
        3. Novel and potentially impactful
        4. Clearly stated with specific predictions
        
        Hypothesis:
        """
        
        return prompt.strip()
    
    async def generate_mathematical_formulation(self, pattern_analysis: Dict[str, Any]) -> Optional[str]:
        """Generate mathematical formulation for quantitative hypotheses"""
        if pattern_analysis['correlation_strength'] < 0.5:
            return None
        
        variables = pattern_analysis['variables_involved']
        if len(variables) < 2:
            return None
        
        # Generate basic mathematical relationships
        formulations = []
        
        if pattern_analysis['pattern_type'] == 'linear_correlation':
            formulations.append(f"{variables[0]} = α + β * {variables[1]} + ε")
        elif pattern_analysis['pattern_type'] == 'exponential_growth':
            formulations.append(f"{variables[0]} = A * exp(k * {variables[1]})")
        elif pattern_analysis['pattern_type'] == 'power_law':
            formulations.append(f"{variables[0]} = A * {variables[1]}^n")
        elif pattern_analysis['pattern_type'] == 'periodic':
            formulations.append(f"{variables[0]} = A * sin(ω * {variables[1]} + φ) + C")
        
        return formulations[0] if formulations else None
    
    async def assess_hypothesis_quality(self, hypothesis_text: str, pattern_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of generated hypothesis"""
        scores = {
            'confidence': 0.0,
            'novelty': 0.0,
            'testability': 0.0,
            'impact': 0.0
        }
        
        # Confidence based on statistical significance and pattern strength
        scores['confidence'] = min(1.0, pattern_analysis['statistical_significance'] * 
                                 pattern_analysis['correlation_strength'])
        
        # Novelty based on keyword uniqueness and literature gap analysis
        scores['novelty'] = await self.assess_novelty(hypothesis_text)
        
        # Testability based on measurable variables and experimental feasibility
        scores['testability'] = await self.assess_testability(hypothesis_text, pattern_analysis)
        
        # Impact based on domain relevance and potential applications
        scores['impact'] = await self.assess_impact_potential(hypothesis_text, pattern_analysis)
        
        return scores
    
    async def assess_novelty(self, hypothesis_text: str) -> float:
        """Assess novelty of hypothesis against existing literature"""
        # Extract key concepts
        doc = self.nlp(hypothesis_text)
        key_concepts = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
        
        # Simple novelty scoring based on concept uniqueness
        # In production, this would query literature databases
        novelty_score = min(1.0, len(set(key_concepts)) / max(1, len(key_concepts)) * 0.8)
        
        return novelty_score
    
    async def assess_testability(self, hypothesis_text: str, pattern_analysis: Dict[str, Any]) -> float:
        """Assess how testable the hypothesis is"""
        testability_indicators = [
            'measure', 'observe', 'experiment', 'test', 'predict', 'correlate',
            'temperature', 'pressure', 'concentration', 'rate', 'frequency'
        ]
        
        text_lower = hypothesis_text.lower()
        indicator_count = sum(1 for indicator in testability_indicators if indicator in text_lower)
        
        # Factor in number of measurable variables
        variable_factor = min(1.0, len(pattern_analysis['variables_involved']) / 5.0)
        
        testability_score = min(1.0, (indicator_count / len(testability_indicators)) + variable_factor) / 2
        
        return testability_score
    
    async def assess_impact_potential(self, hypothesis_text: str, pattern_analysis: Dict[str, Any]) -> float:
        """Assess potential scientific impact of hypothesis"""
        impact_keywords = [
            'breakthrough', 'novel', 'significant', 'important', 'critical',
            'hazard', 'risk', 'prediction', 'forecast', 'early warning'
        ]
        
        text_lower = hypothesis_text.lower()
        impact_count = sum(1 for keyword in impact_keywords if keyword in text_lower)
        
        # Factor in geological relevance
        relevance_factor = pattern_analysis.get('geological_relevance', 0.5)
        
        impact_score = min(1.0, (impact_count / len(impact_keywords)) + relevance_factor) / 2
        
        return impact_score
    
    async def assess_geological_relevance(self, pattern: Dict[str, Any], domain: ResearchDomain) -> float:
        """Assess how relevant the pattern is to geological sciences"""
        geological_terms = [
            'seismic', 'volcanic', 'tectonic', 'magma', 'earthquake', 'eruption',
            'fault', 'mineral', 'rock', 'geological', 'geophysical', 'geochemical'
        ]
        
        pattern_text = str(pattern).lower()
        term_count = sum(1 for term in geological_terms if term in pattern_text)
        
        relevance_score = min(1.0, term_count / len(geological_terms))
        
        return relevance_score
    
    async def generate_hypothesis_title(self, hypothesis_text: str) -> str:
        """Generate a concise title for the hypothesis"""
        # Extract key phrases using NLP
        doc = self.nlp(hypothesis_text)
        
        # Get noun phrases and entities
        key_phrases = [chunk.text for chunk in doc.noun_chunks][:3]
        entities = [ent.text for ent in doc.ents][:2]
        
        # Combine and create title
        title_elements = key_phrases + entities
        title = "Hypothesis: " + " - ".join(title_elements[:2])
        
        return title[:100]  # Limit title length
    
    async def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from hypothesis text"""
        doc = self.nlp(text)
        
        # Extract nouns, adjectives, and named entities
        keywords = []
        for token in doc:
            if (token.pos_ in ['NOUN', 'ADJ'] and 
                len(token.text) > 3 and 
                not token.is_stop and 
                token.is_alpha):
                keywords.append(token.lemma_.lower())
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                keywords.append(ent.text.lower())
        
        return list(set(keywords))[:10]  # Return top 10 unique keywords
    
    async def post_process_hypothesis(self, hypothesis: str, domain: ResearchDomain) -> str:
        """Post-process and validate generated hypothesis"""
        # Clean up the text
        hypothesis = hypothesis.strip()
        
        # Ensure it starts with a capital letter
        if hypothesis and not hypothesis[0].isupper():
            hypothesis = hypothesis[0].upper() + hypothesis[1:]
        
        # Ensure it ends with proper punctuation
        if hypothesis and hypothesis[-1] not in '.!?':
            hypothesis += '.'
        
        # Add domain context if missing
        domain_terms = {
            ResearchDomain.VOLCANOLOGY: ['volcanic', 'eruption', 'magma'],
            ResearchDomain.SEISMOLOGY: ['seismic', 'earthquake', 'fault'],
            ResearchDomain.GEODYNAMICS: ['tectonic', 'plate', 'mantle']
        }
        
        relevant_terms = domain_terms.get(domain, [])
        has_domain_context = any(term in hypothesis.lower() for term in relevant_terms)
        
        if not has_domain_context and relevant_terms:
            hypothesis = f"In {domain.value}, {hypothesis.lower()}"
        
        return hypothesisclass 
ExperimentDesigner:
    """Automated experiment design system for testing scientific hypotheses"""
    
    def __init__(self):
        self.design_templates = {}
        self.statistical_methods = {}
        self.resource_estimator = None
        self.initialize_design_framework()
    
    def initialize_design_framework(self):
        """Initialize experiment design framework"""
        self.design_templates = {
            'observational_study': {
                'description': 'Systematic observation and data collection',
                'methodology': 'Cross-sectional or longitudinal observation',
                'controls': ['Environmental factors', 'Measurement consistency'],
                'statistical_analysis': 'Descriptive statistics and correlation analysis'
            },
            'controlled_experiment': {
                'description': 'Manipulated variables with control groups',
                'methodology': 'Randomized controlled trial design',
                'controls': ['Control group', 'Randomization', 'Blinding'],
                'statistical_analysis': 'ANOVA, t-tests, regression analysis'
            },
            'field_study': {
                'description': 'Real-world geological field investigation',
                'methodology': 'Systematic field sampling and measurement',
                'controls': ['Sampling bias', 'Weather conditions', 'Equipment calibration'],
                'statistical_analysis': 'Spatial statistics and geostatistical analysis'
            },
            'laboratory_experiment': {
                'description': 'Controlled laboratory conditions',
                'methodology': 'Standardized laboratory protocols',
                'controls': ['Temperature', 'Pressure', 'Sample preparation'],
                'statistical_analysis': 'Experimental design analysis (factorial, response surface)'
            },
            'computational_simulation': {
                'description': 'Numerical modeling and simulation',
                'methodology': 'Physics-based computational models',
                'controls': ['Model parameters', 'Boundary conditions', 'Numerical stability'],
                'statistical_analysis': 'Sensitivity analysis and uncertainty quantification'
            }
        }
        
        self.statistical_methods = {
            'correlation_analysis': 'Pearson/Spearman correlation coefficients',
            'regression_analysis': 'Linear/nonlinear regression modeling',
            'time_series_analysis': 'ARIMA, spectral analysis, wavelet analysis',
            'spatial_analysis': 'Kriging, variogram analysis, spatial autocorrelation',
            'multivariate_analysis': 'PCA, factor analysis, cluster analysis',
            'hypothesis_testing': 't-tests, ANOVA, chi-square tests',
            'bayesian_analysis': 'Bayesian inference and model comparison',
            'machine_learning': 'Supervised/unsupervised learning algorithms'
        }
    
    async def design_experiment(self, hypothesis: ScientificHypothesis) -> ExperimentDesign:
        """Design an experiment to test the given hypothesis"""
        try:
            # Analyze hypothesis requirements
            requirements = await self.analyze_hypothesis_requirements(hypothesis)
            
            # Select appropriate experimental design
            design_type = await self.select_design_type(requirements)
            
            # Generate detailed experimental design
            design = await self.generate_detailed_design(hypothesis, design_type, requirements)
            
            # Assess feasibility and resources
            design.feasibility_score = await self.assess_feasibility(design)
            design.cost_estimate = await self.estimate_costs(design)
            
            return design
            
        except Exception as e:
            logger.error(f"Error designing experiment: {e}")
            raise
    
    async def analyze_hypothesis_requirements(self, hypothesis: ScientificHypothesis) -> Dict[str, Any]:
        """Analyze what the hypothesis requires for testing"""
        requirements = {
            'variables_to_measure': [],
            'experimental_conditions': [],
            'data_collection_methods': [],
            'statistical_requirements': [],
            'temporal_scale': 'unknown',
            'spatial_scale': 'unknown',
            'sample_size_estimate': 0,
            'equipment_needed': [],
            'expertise_required': []
        }
        
        # Extract variables from hypothesis description
        doc = spacy.load("en_core_web_sm")(hypothesis.description)
        
        # Identify measurable variables
        measurable_terms = [
            'temperature', 'pressure', 'velocity', 'acceleration', 'frequency',
            'magnitude', 'concentration', 'density', 'viscosity', 'conductivity',
            'displacement', 'strain', 'stress', 'flow rate', 'volume'
        ]
        
        for token in doc:
            if token.lemma_.lower() in measurable_terms:
                requirements['variables_to_measure'].append(token.text)
        
        # Determine temporal and spatial scales
        temporal_indicators = {
            'seconds': ['seismic', 'explosion', 'impact'],
            'minutes': ['eruption', 'landslide'],
            'hours': ['storm', 'tide'],
            'days': ['weather', 'daily'],
            'months': ['seasonal', 'monthly'],
            'years': ['annual', 'yearly', 'climate'],
            'decades': ['long-term', 'decadal'],
            'geological': ['geological', 'tectonic', 'evolutionary']
        }
        
        text_lower = hypothesis.description.lower()
        for scale, indicators in temporal_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                requirements['temporal_scale'] = scale
                break
        
        # Determine required expertise
        domain_expertise = {
            ResearchDomain.VOLCANOLOGY: ['volcanologist', 'geochemist', 'seismologist'],
            ResearchDomain.SEISMOLOGY: ['seismologist', 'geophysicist', 'structural geologist'],
            ResearchDomain.GEODYNAMICS: ['geodynamicist', 'tectonicist', 'geophysicist']
        }
        
        requirements['expertise_required'] = domain_expertise.get(hypothesis.domain, ['geologist'])
        
        return requirements
    
    async def select_design_type(self, requirements: Dict[str, Any]) -> str:
        """Select the most appropriate experimental design type"""
        # Decision logic for design type selection
        if len(requirements['variables_to_measure']) == 0:
            return 'observational_study'
        
        if requirements['temporal_scale'] in ['geological', 'decades', 'years']:
            return 'observational_study'
        
        if 'field' in str(requirements).lower() or 'outcrop' in str(requirements).lower():
            return 'field_study'
        
        if requirements['temporal_scale'] in ['seconds', 'minutes', 'hours']:
            return 'laboratory_experiment'
        
        if 'model' in str(requirements).lower() or 'simulation' in str(requirements).lower():
            return 'computational_simulation'
        
        return 'controlled_experiment'
    
    async def generate_detailed_design(self, 
                                     hypothesis: ScientificHypothesis, 
                                     design_type: str, 
                                     requirements: Dict[str, Any]) -> ExperimentDesign:
        """Generate detailed experimental design"""
        template = self.design_templates[design_type]
        
        design = ExperimentDesign(
            id=f"exp_{hypothesis.id}_{design_type}",
            hypothesis_id=hypothesis.id,
            title=f"Experimental Test of {hypothesis.title}",
            objective=f"To test the hypothesis: {hypothesis.description}",
            methodology=template['methodology'],
            variables={
                'independent': await self.identify_independent_variables(hypothesis),
                'dependent': requirements['variables_to_measure'],
                'controlled': await self.identify_control_variables(hypothesis, design_type)
            },
            controls=template['controls'],
            measurements=await self.design_measurement_protocol(requirements),
            statistical_analysis=await self.select_statistical_methods(requirements),
            expected_outcomes=await self.predict_outcomes(hypothesis),
            resource_requirements=await self.estimate_resources(design_type, requirements),
            timeline=await self.create_timeline(design_type, requirements),
            risk_assessment=await self.assess_risks(design_type, requirements),
            feasibility_score=0.0  # Will be calculated later
        )
        
        return design
    
    async def identify_independent_variables(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Identify independent variables from hypothesis"""
        # Parse mathematical formulation if available
        if hypothesis.mathematical_formulation:
            # Extract variables from mathematical expression
            import re
            variables = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', hypothesis.mathematical_formulation)
            return variables[:3]  # Return first 3 as independent variables
        
        # Extract from description using NLP
        doc = spacy.load("en_core_web_sm")(hypothesis.description)
        variables = []
        
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                token.dep_ in ['nsubj', 'dobj'] and 
                len(token.text) > 3):
                variables.append(token.text)
        
        return variables[:3]
    
    async def identify_control_variables(self, hypothesis: ScientificHypothesis, design_type: str) -> List[str]:
        """Identify variables that need to be controlled"""
        control_variables = []
        
        # Standard geological control variables
        geological_controls = [
            'temperature', 'pressure', 'humidity', 'elevation',
            'sample_preparation', 'measurement_time', 'equipment_calibration'
        ]
        
        # Add domain-specific controls
        domain_controls = {
            ResearchDomain.VOLCANOLOGY: ['gas_composition', 'magma_temperature', 'eruption_style'],
            ResearchDomain.SEISMOLOGY: ['station_location', 'instrument_response', 'noise_level'],
            ResearchDomain.GEODYNAMICS: ['boundary_conditions', 'material_properties', 'time_step']
        }
        
        control_variables.extend(geological_controls[:5])
        control_variables.extend(domain_controls.get(hypothesis.domain, [])[:3])
        
        return control_variables
    
    async def design_measurement_protocol(self, requirements: Dict[str, Any]) -> List[str]:
        """Design measurement protocol based on requirements"""
        protocols = []
        
        for variable in requirements['variables_to_measure']:
            if 'temperature' in variable.lower():
                protocols.append('Thermocouple measurements with ±0.1°C accuracy')
            elif 'pressure' in variable.lower():
                protocols.append('Pressure transducer measurements with ±0.01 bar accuracy')
            elif 'seismic' in variable.lower():
                protocols.append('Broadband seismometer recordings at 100 Hz sampling rate')
            elif 'chemical' in variable.lower():
                protocols.append('X-ray fluorescence spectroscopy for elemental analysis')
            else:
                protocols.append(f'Standardized measurement protocol for {variable}')
        
        return protocols
    
    async def select_statistical_methods(self, requirements: Dict[str, Any]) -> str:
        """Select appropriate statistical analysis methods"""
        methods = []
        
        num_variables = len(requirements['variables_to_measure'])
        
        if num_variables == 1:
            methods.append('Descriptive statistics and normality testing')
        elif num_variables == 2:
            methods.append('Correlation analysis and simple regression')
        else:
            methods.append('Multiple regression and multivariate analysis')
        
        # Add time series analysis if temporal data
        if requirements['temporal_scale'] not in ['unknown']:
            methods.append('Time series analysis and trend detection')
        
        # Add spatial analysis if spatial data
        if 'field' in str(requirements).lower():
            methods.append('Spatial statistics and geostatistical analysis')
        
        return '; '.join(methods)
    
    async def predict_outcomes(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Predict expected experimental outcomes"""
        outcomes = []
        
        if hypothesis.confidence_score > 0.7:
            outcomes.append('Strong support for the hypothesis with statistical significance')
        elif hypothesis.confidence_score > 0.5:
            outcomes.append('Moderate support for the hypothesis')
        else:
            outcomes.append('Inconclusive results requiring further investigation')
        
        # Add specific predictions based on hypothesis type
        if hypothesis.discovery_type == DiscoveryType.PATTERN_DISCOVERY:
            outcomes.append('Identification of consistent patterns in the data')
        elif hypothesis.discovery_type == DiscoveryType.CAUSAL_INFERENCE:
            outcomes.append('Evidence of causal relationships between variables')
        
        outcomes.append('Quantification of uncertainty and confidence intervals')
        outcomes.append('Recommendations for future research directions')
        
        return outcomes    a
sync def estimate_resources(self, design_type: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resource requirements for the experiment"""
        resources = {
            'personnel': [],
            'equipment': [],
            'materials': [],
            'facilities': [],
            'computational': {},
            'travel': {}
        }
        
        # Personnel requirements
        expertise_needed = requirements.get('expertise_required', [])
        resources['personnel'] = [f"{expert} (1 person-month)" for expert in expertise_needed]
        
        # Equipment requirements by design type
        equipment_by_type = {
            'field_study': ['GPS units', 'Geological hammers', 'Sample containers', 'Field notebooks'],
            'laboratory_experiment': ['Analytical instruments', 'Sample preparation equipment', 'Safety equipment'],
            'computational_simulation': ['High-performance computers', 'Software licenses', 'Data storage'],
            'observational_study': ['Monitoring instruments', 'Data loggers', 'Communication equipment']
        }
        
        resources['equipment'] = equipment_by_type.get(design_type, ['Standard geological equipment'])
        
        # Materials and consumables
        resources['materials'] = ['Sample containers', 'Reagents', 'Calibration standards', 'Documentation materials']
        
        # Facility requirements
        if design_type == 'laboratory_experiment':
            resources['facilities'] = ['Clean laboratory space', 'Fume hoods', 'Sample storage']
        elif design_type == 'field_study':
            resources['facilities'] = ['Field station access', 'Transportation', 'Accommodation']
        
        return resources
    
    async def create_timeline(self, design_type: str, requirements: Dict[str, Any]) -> Dict[str, str]:
        """Create project timeline based on design type and requirements"""
        timeline = {}
        
        # Base timeline templates
        timeline_templates = {
            'field_study': {
                'planning': '2 weeks',
                'preparation': '1 week',
                'data_collection': '2-4 weeks',
                'analysis': '4-6 weeks',
                'reporting': '2-3 weeks'
            },
            'laboratory_experiment': {
                'planning': '1 week',
                'setup': '1 week',
                'data_collection': '1-2 weeks',
                'analysis': '2-3 weeks',
                'reporting': '1-2 weeks'
            },
            'computational_simulation': {
                'model_development': '2-3 weeks',
                'validation': '1 week',
                'simulation_runs': '1-2 weeks',
                'analysis': '2-3 weeks',
                'reporting': '1-2 weeks'
            },
            'observational_study': {
                'planning': '1 week',
                'monitoring_setup': '1 week',
                'data_collection': '4-12 weeks',
                'analysis': '3-4 weeks',
                'reporting': '2-3 weeks'
            }
        }
        
        timeline = timeline_templates.get(design_type, timeline_templates['laboratory_experiment'])
        
        # Adjust based on temporal scale
        temporal_scale = requirements.get('temporal_scale', 'unknown')
        if temporal_scale in ['geological', 'decades', 'years']:
            timeline['data_collection'] = '6-12 months'
        
        return timeline
    
    async def assess_risks(self, design_type: str, requirements: Dict[str, Any]) -> Dict[str, float]:
        """Assess risks associated with the experimental design"""
        risks = {
            'technical_failure': 0.2,
            'data_quality': 0.3,
            'resource_availability': 0.2,
            'timeline_delays': 0.3,
            'safety_concerns': 0.1,
            'weather_dependency': 0.0,
            'equipment_failure': 0.2,
            'personnel_availability': 0.1
        }
        
        # Adjust risks based on design type
        if design_type == 'field_study':
            risks['weather_dependency'] = 0.4
            risks['safety_concerns'] = 0.3
            risks['timeline_delays'] = 0.5
        elif design_type == 'computational_simulation':
            risks['technical_failure'] = 0.4
            risks['computational_resources'] = 0.3
        elif design_type == 'laboratory_experiment':
            risks['equipment_failure'] = 0.3
            risks['data_quality'] = 0.2
        
        return risks
    
    async def assess_feasibility(self, design: ExperimentDesign) -> float:
        """Assess overall feasibility of the experimental design"""
        feasibility_factors = {
            'resource_availability': 0.8,  # Assume resources are generally available
            'technical_complexity': 0.7,   # Moderate technical complexity
            'timeline_realism': 0.8,       # Realistic timeline
            'cost_reasonableness': 0.7,    # Reasonable cost
            'safety_compliance': 0.9,      # High safety compliance
            'ethical_approval': 0.9        # Assume ethical approval obtainable
        }
        
        # Adjust based on risk assessment
        avg_risk = sum(design.risk_assessment.values()) / len(design.risk_assessment)
        feasibility_factors['risk_mitigation'] = 1.0 - avg_risk
        
        # Calculate weighted feasibility score
        feasibility_score = sum(feasibility_factors.values()) / len(feasibility_factors)
        
        return min(1.0, feasibility_score)
    
    async def estimate_costs(self, design: ExperimentDesign) -> float:
        """Estimate costs for the experimental design"""
        cost_factors = {
            'personnel': 5000,      # Base personnel cost
            'equipment': 2000,      # Base equipment cost
            'materials': 500,       # Base materials cost
            'travel': 1000,         # Base travel cost
            'facilities': 1000,     # Base facility cost
            'overhead': 0.3         # 30% overhead
        }
        
        # Adjust costs based on design complexity
        num_variables = len(design.variables.get('dependent', []))
        complexity_multiplier = 1.0 + (num_variables * 0.2)
        
        base_cost = sum(cost_factors[key] for key in cost_factors if key != 'overhead')
        total_cost = base_cost * complexity_multiplier * (1 + cost_factors['overhead'])
        
        return total_costcla
ss LiteratureMiner:
    """Advanced scientific literature mining and analysis system"""
    
    def __init__(self):
        self.arxiv_client = arxiv.Client()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.citation_network = nx.DiGraph()
        self.paper_embeddings = {}
        self.research_trends = {}
        
    async def mine_literature(self, query: str, max_papers: int = 100) -> LiteratureAnalysis:
        """Mine and analyze scientific literature for a given query"""
        try:
            # Search multiple sources
            papers = await self.search_multiple_sources(query, max_papers)
            
            # Analyze papers
            analysis = await self.analyze_papers(papers, query)
            
            # Build citation network
            await self.build_citation_network(papers)
            
            # Identify trends and gaps
            analysis.trending_topics = await self.identify_trending_topics(papers)
            analysis.research_gaps = await self.identify_research_gaps(papers)
            
            # Analyze collaboration patterns
            analysis.collaboration_patterns = await self.analyze_collaborations(papers)
            
            # Calculate impact metrics
            analysis.impact_metrics = await self.calculate_impact_metrics(papers)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error mining literature: {e}")
            raise
    
    async def search_multiple_sources(self, query: str, max_papers: int) -> List[Dict[str, Any]]:
        """Search multiple academic databases and sources"""
        papers = []
        
        # Search arXiv
        try:
            arxiv_papers = await self.search_arxiv(query, max_papers // 3)
            papers.extend(arxiv_papers)
        except Exception as e:
            logger.warning(f"ArXiv search failed: {e}")
        
        # Search Google Scholar (using scholarly library)
        try:
            scholar_papers = await self.search_google_scholar(query, max_papers // 3)
            papers.extend(scholar_papers)
        except Exception as e:
            logger.warning(f"Google Scholar search failed: {e}")
        
        # Search PubMed for geological/environmental papers
        try:
            pubmed_papers = await self.search_pubmed(query, max_papers // 3)
            papers.extend(pubmed_papers)
        except Exception as e:
            logger.warning(f"PubMed search failed: {e}")
        
        # Remove duplicates based on title similarity
        papers = await self.deduplicate_papers(papers)
        
        return papers[:max_papers]
    
    async def search_arxiv(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search arXiv for relevant papers"""
        papers = []
        
        # Construct arXiv search
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for result in self.arxiv_client.results(search):
            paper = {
                'title': result.title,
                'authors': [author.name for author in result.authors],
                'abstract': result.summary,
                'published': result.published,
                'url': result.entry_id,
                'source': 'arxiv',
                'categories': result.categories,
                'doi': getattr(result, 'doi', None)
            }
            papers.append(paper)
        
        return papers
    
    async def search_google_scholar(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search Google Scholar for relevant papers"""
        papers = []
        
        try:
            search_query = scholarly.search_pubs(query)
            
            for i, result in enumerate(search_query):
                if i >= max_results:
                    break
                
                paper = {
                    'title': result.get('bib', {}).get('title', ''),
                    'authors': result.get('bib', {}).get('author', []),
                    'abstract': result.get('bib', {}).get('abstract', ''),
                    'published': result.get('bib', {}).get('pub_year', ''),
                    'url': result.get('pub_url', ''),
                    'source': 'google_scholar',
                    'citations': result.get('num_citations', 0),
                    'venue': result.get('bib', {}).get('venue', '')
                }
                papers.append(paper)
                
        except Exception as e:
            logger.warning(f"Google Scholar search error: {e}")
        
        return papers
    
    async def search_pubmed(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search PubMed for relevant papers (geological/environmental focus)"""
        papers = []
        
        # Add geological terms to query
        geological_query = f"{query} AND (geology OR geophysics OR volcanology OR seismology)"
        
        # This would integrate with PubMed API in production
        # For now, return empty list as PubMed requires API key
        logger.info(f"PubMed search would be performed for: {geological_query}")
        
        return papers
    
    async def deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on title similarity"""
        if not papers:
            return papers
        
        # Calculate title embeddings
        titles = [paper.get('title', '') for paper in papers]
        embeddings = self.semantic_model.encode(titles)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Remove duplicates (similarity > 0.9)
        unique_papers = []
        used_indices = set()
        
        for i, paper in enumerate(papers):
            if i in used_indices:
                continue
            
            unique_papers.append(paper)
            
            # Mark similar papers as used
            for j in range(i + 1, len(papers)):
                if similarity_matrix[i][j] > 0.9:
                    used_indices.add(j)
        
        return unique_papers
    
    async def analyze_papers(self, papers: List[Dict[str, Any]], query: str) -> LiteratureAnalysis:
        """Analyze the collected papers"""
        analysis = LiteratureAnalysis(
            query=query,
            papers_analyzed=len(papers),
            key_findings=[],
            research_gaps=[],
            trending_topics=[],
            citation_network={},
            collaboration_patterns={},
            temporal_trends={},
            semantic_clusters=[],
            impact_metrics={}
        )
        
        if not papers:
            return analysis
        
        # Extract key findings from abstracts
        analysis.key_findings = await self.extract_key_findings(papers)
        
        # Perform semantic clustering
        analysis.semantic_clusters = await self.perform_semantic_clustering(papers)
        
        # Analyze temporal trends
        analysis.temporal_trends = await self.analyze_temporal_trends(papers)
        
        return analysis
    
    async def extract_key_findings(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from paper abstracts"""
        findings = []
        
        # Combine all abstracts
        abstracts = [paper.get('abstract', '') for paper in papers if paper.get('abstract')]
        
        if not abstracts:
            return findings
        
        # Use TF-IDF to find important phrases
        vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(2, 3),
            stop_words='english'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top phrases
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:]
            
            findings = [feature_names[i] for i in top_indices]
            
        except Exception as e:
            logger.warning(f"Error extracting key findings: {e}")
        
        return findings
    
    async def perform_semantic_clustering(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform semantic clustering of papers"""
        clusters = []
        
        if len(papers) < 3:
            return clusters
        
        # Get paper embeddings
        texts = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            texts.append(text)
        
        embeddings = self.semantic_model.encode(texts)
        
        # Perform clustering
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Group papers by cluster
        cluster_groups = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise points
                continue
            
            if label not in cluster_groups:
                cluster_groups[label] = []
            
            cluster_groups[label].append(papers[i])
        
        # Create cluster summaries
        for cluster_id, cluster_papers in cluster_groups.items():
            cluster_info = {
                'cluster_id': cluster_id,
                'paper_count': len(cluster_papers),
                'representative_titles': [p.get('title', '')[:100] for p in cluster_papers[:3]],
                'common_themes': await self.extract_common_themes(cluster_papers)
            }
            clusters.append(cluster_info)
        
        return clusters
    
    async def extract_common_themes(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from a cluster of papers"""
        # Combine titles and abstracts
        texts = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            texts.append(text)
        
        combined_text = ' '.join(texts)
        
        # Extract key terms using NLP
        doc = spacy.load("en_core_web_sm")(combined_text)
        
        # Get noun phrases and entities
        themes = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 5 and chunk.text.lower() not in ['paper', 'study', 'research']:
                themes.append(chunk.text)
        
        # Get most frequent themes
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Return top themes
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, count in sorted_themes[:5]]
    
    async def analyze_temporal_trends(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in the literature"""
        trends = {
            'publication_timeline': {},
            'research_evolution': {},
            'emerging_topics': []
        }
        
        # Group papers by year
        year_groups = {}
        for paper in papers:
            year = str(paper.get('published', ''))[:4]
            if year.isdigit():
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(paper)
        
        trends['publication_timeline'] = {year: len(papers) for year, papers in year_groups.items()}
        
        # Analyze topic evolution over time
        if len(year_groups) > 1:
            trends['research_evolution'] = await self.analyze_topic_evolution(year_groups)
        
        return trends
    
    async def analyze_topic_evolution(self, year_groups: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Analyze how research topics evolve over time"""
        evolution = {}
        
        for year, papers in year_groups.items():
            # Extract topics for this year
            year_topics = []
            for paper in papers:
                title_abstract = f"{paper.get('title', '')} {paper.get('abstract', '')}"
                doc = spacy.load("en_core_web_sm")(title_abstract)
                
                topics = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'EVENT', 'ORG']]
                year_topics.extend(topics)
            
            # Count topic frequencies
            topic_counts = {}
            for topic in year_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            # Store top topics for this year
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            evolution[year] = [topic for topic, count in sorted_topics[:5]]
        
        return evolution    async 
def build_citation_network(self, papers: List[Dict[str, Any]]):
        """Build citation network from papers"""
        # Clear existing network
        self.citation_network.clear()
        
        # Add nodes for each paper
        for paper in papers:
            paper_id = paper.get('url', paper.get('title', ''))
            self.citation_network.add_node(paper_id, **paper)
        
        # Add edges based on citations (simplified - would need actual citation data)
        # This is a placeholder implementation
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers):
                if i != j:
                    # Simple heuristic: papers by same authors or similar titles might cite each other
                    similarity = await self.calculate_paper_similarity(paper1, paper2)
                    if similarity > 0.7:
                        paper1_id = paper1.get('url', paper1.get('title', ''))
                        paper2_id = paper2.get('url', paper2.get('title', ''))
                        self.citation_network.add_edge(paper1_id, paper2_id, weight=similarity)
    
    async def calculate_paper_similarity(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> float:
        """Calculate similarity between two papers"""
        # Compare titles and abstracts
        text1 = f"{paper1.get('title', '')} {paper1.get('abstract', '')}"
        text2 = f"{paper2.get('title', '')} {paper2.get('abstract', '')}"
        
        if not text1 or not text2:
            return 0.0
        
        # Use sentence transformer for semantic similarity
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    async def identify_trending_topics(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Identify trending topics in the literature"""
        # Extract topics from recent papers (last 2 years)
        current_year = datetime.now().year
        recent_papers = []
        
        for paper in papers:
            pub_year = str(paper.get('published', ''))[:4]
            if pub_year.isdigit() and int(pub_year) >= current_year - 2:
                recent_papers.append(paper)
        
        if not recent_papers:
            recent_papers = papers  # Use all papers if no recent ones
        
        # Extract topics using NLP
        all_topics = []
        for paper in recent_papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            doc = spacy.load("en_core_web_sm")(text)
            
            # Extract noun phrases and named entities
            topics = []
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3:
                    topics.append(chunk.text.lower())
            
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'EVENT', 'ORG', 'GPE']:
                    topics.append(ent.text.lower())
            
            all_topics.extend(topics)
        
        # Count topic frequencies
        topic_counts = {}
        for topic in all_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Return top trending topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:10] if count > 1]
    
    async def identify_research_gaps(self, papers: List[Dict[str, Any]]) -> List[str]:
        """Identify potential research gaps in the literature"""
        gaps = []
        
        # Analyze what's missing by looking at common research questions
        research_questions = [
            "What causes",
            "How does",
            "Why do",
            "When will",
            "Where can",
            "Which factors"
        ]
        
        # Extract questions and statements from abstracts
        all_text = ""
        for paper in papers:
            abstract = paper.get('abstract', '')
            all_text += f" {abstract}"
        
        # Find gaps by identifying underexplored areas
        doc = spacy.load("en_core_web_sm")(all_text)
        
        # Look for uncertainty indicators
        uncertainty_patterns = [
            "unclear", "unknown", "uncertain", "poorly understood",
            "needs further research", "requires investigation",
            "not well established", "remains to be determined"
        ]
        
        sentences = [sent.text for sent in doc.sents]
        gap_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(pattern in sentence_lower for pattern in uncertainty_patterns):
                gap_sentences.append(sentence.strip())
        
        # Extract key gap areas
        for sentence in gap_sentences[:5]:  # Top 5 gap indicators
            # Extract the main topic of the gap
            doc_sent = spacy.load("en_core_web_sm")(sentence)
            key_phrases = [chunk.text for chunk in doc_sent.noun_chunks]
            if key_phrases:
                gap = f"Limited understanding of {key_phrases[0]}"
                gaps.append(gap)
        
        return gaps
    
    async def analyze_collaborations(self, papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collaboration patterns in the literature"""
        collaborations = {
            'author_networks': {},
            'institutional_collaborations': {},
            'international_collaborations': {},
            'collaboration_trends': {}
        }
        
        # Build author collaboration network
        author_pairs = {}
        institutions = {}
        
        for paper in papers:
            authors = paper.get('authors', [])
            if len(authors) > 1:
                # Record all author pairs in this paper
                for i in range(len(authors)):
                    for j in range(i + 1, len(authors)):
                        pair = tuple(sorted([authors[i], authors[j]]))
                        author_pairs[pair] = author_pairs.get(pair, 0) + 1
        
        # Find most frequent collaborations
        top_collaborations = sorted(author_pairs.items(), key=lambda x: x[1], reverse=True)
        collaborations['author_networks'] = {
            f"{pair[0]} - {pair[1]}": count 
            for pair, count in top_collaborations[:10]
        }
        
        return collaborations
    
    async def calculate_impact_metrics(self, papers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate impact metrics for the literature set"""
        metrics = {
            'total_papers': len(papers),
            'average_citations': 0.0,
            'h_index_estimate': 0.0,
            'recent_activity': 0.0,
            'international_diversity': 0.0
        }
        
        # Calculate average citations
        citations = [paper.get('citations', 0) for paper in papers if isinstance(paper.get('citations'), int)]
        if citations:
            metrics['average_citations'] = sum(citations) / len(citations)
            
            # Estimate h-index
            citations_sorted = sorted(citations, reverse=True)
            h_index = 0
            for i, cit_count in enumerate(citations_sorted):
                if cit_count >= i + 1:
                    h_index = i + 1
                else:
                    break
            metrics['h_index_estimate'] = h_index
        
        # Calculate recent activity (papers in last 2 years)
        current_year = datetime.now().year
        recent_papers = 0
        for paper in papers:
            pub_year = str(paper.get('published', ''))[:4]
            if pub_year.isdigit() and int(pub_year) >= current_year - 2:
                recent_papers += 1
        
        metrics['recent_activity'] = recent_papers / len(papers) if papers else 0.0
        
        return metricsc
lass KnowledgeGraphBuilder:
    """Advanced knowledge graph construction and reasoning system"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_embeddings = {}
        self.relationship_types = {
            'cites', 'authored_by', 'affiliated_with', 'studies', 'uses_method',
            'measures', 'correlates_with', 'causes', 'influences', 'part_of',
            'similar_to', 'contradicts', 'supports', 'extends', 'applies_to'
        }
        self.reasoning_engine = None
        self.initialize_reasoning_engine()
    
    def initialize_reasoning_engine(self):
        """Initialize the reasoning engine for knowledge graph inference"""
        # Initialize semantic similarity model
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize reasoning rules
        self.reasoning_rules = {
            'transitivity': ['causes', 'influences', 'part_of'],
            'symmetry': ['similar_to', 'correlates_with'],
            'inverse': {'causes': 'caused_by', 'influences': 'influenced_by'}
        }
    
    async def build_knowledge_graph(self, 
                                  papers: List[Dict[str, Any]], 
                                  hypotheses: List[ScientificHypothesis]) -> Dict[str, Any]:
        """Build comprehensive knowledge graph from papers and hypotheses"""
        try:
            # Clear existing graph
            self.graph.clear()
            self.node_embeddings.clear()
            
            # Add paper nodes and relationships
            await self.add_paper_nodes(papers)
            await self.add_paper_relationships(papers)
            
            # Add hypothesis nodes and relationships
            await self.add_hypothesis_nodes(hypotheses)
            await self.add_hypothesis_relationships(hypotheses, papers)
            
            # Extract concept nodes from text
            await self.extract_concept_nodes(papers, hypotheses)
            
            # Infer additional relationships
            await self.infer_relationships()
            
            # Calculate graph metrics
            graph_metrics = await self.calculate_graph_metrics()
            
            return {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'node_types': await self.get_node_type_distribution(),
                'relationship_types': await self.get_relationship_distribution(),
                'graph_metrics': graph_metrics,
                'central_concepts': await self.find_central_concepts(),
                'knowledge_clusters': await self.identify_knowledge_clusters()
            }
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise
    
    async def add_paper_nodes(self, papers: List[Dict[str, Any]]):
        """Add paper nodes to the knowledge graph"""
        for paper in papers:
            paper_id = f"paper_{hash(paper.get('title', ''))}"
            
            node_data = {
                'type': 'paper',
                'title': paper.get('title', ''),
                'authors': paper.get('authors', []),
                'abstract': paper.get('abstract', ''),
                'published': paper.get('published', ''),
                'source': paper.get('source', ''),
                'citations': paper.get('citations', 0),
                'url': paper.get('url', '')
            }
            
            self.graph.add_node(paper_id, **node_data)
            
            # Generate and store embeddings
            text_content = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            embedding = self.semantic_model.encode(text_content)
            self.node_embeddings[paper_id] = embedding
    
    async def add_paper_relationships(self, papers: List[Dict[str, Any]]):
        """Add relationships between papers"""
        paper_ids = [f"paper_{hash(paper.get('title', ''))}" for paper in papers]
        
        # Add author relationships
        for i, paper in enumerate(papers):
            paper_id = paper_ids[i]
            authors = paper.get('authors', [])
            
            for author in authors:
                author_id = f"author_{hash(author)}"
                
                # Add author node if not exists
                if not self.graph.has_node(author_id):
                    self.graph.add_node(author_id, type='author', name=author)
                
                # Add authorship relationship
                self.graph.add_edge(author_id, paper_id, relationship='authored')
        
        # Add citation relationships (simplified)
        for i, paper1 in enumerate(papers):
            for j, paper2 in enumerate(papers):
                if i != j:
                    similarity = await self.calculate_semantic_similarity(paper1, paper2)
                    if similarity > 0.8:  # High similarity might indicate citation
                        self.graph.add_edge(
                            paper_ids[i], 
                            paper_ids[j], 
                            relationship='cites',
                            weight=similarity
                        )
    
    async def add_hypothesis_nodes(self, hypotheses: List[ScientificHypothesis]):
        """Add hypothesis nodes to the knowledge graph"""
        for hypothesis in hypotheses:
            hypothesis_id = f"hypothesis_{hypothesis.id}"
            
            node_data = {
                'type': 'hypothesis',
                'title': hypothesis.title,
                'description': hypothesis.description,
                'domain': hypothesis.domain.value,
                'discovery_type': hypothesis.discovery_type.value,
                'confidence_score': hypothesis.confidence_score,
                'novelty_score': hypothesis.novelty_score,
                'testability_score': hypothesis.testability_score,
                'keywords': hypothesis.keywords
            }
            
            self.graph.add_node(hypothesis_id, **node_data)
            
            # Generate embeddings
            text_content = f"{hypothesis.title} {hypothesis.description}"
            embedding = self.semantic_model.encode(text_content)
            self.node_embeddings[hypothesis_id] = embedding
    
    async def add_hypothesis_relationships(self, 
                                        hypotheses: List[ScientificHypothesis], 
                                        papers: List[Dict[str, Any]]):
        """Add relationships between hypotheses and papers"""
        paper_ids = [f"paper_{hash(paper.get('title', ''))}" for paper in papers]
        
        for hypothesis in hypotheses:
            hypothesis_id = f"hypothesis_{hypothesis.id}"
            
            # Find related papers based on semantic similarity
            for i, paper in enumerate(papers):
                similarity = await self.calculate_hypothesis_paper_similarity(hypothesis, paper)
                
                if similarity > 0.6:  # Threshold for relatedness
                    self.graph.add_edge(
                        hypothesis_id,
                        paper_ids[i],
                        relationship='supported_by' if similarity > 0.8 else 'related_to',
                        weight=similarity
                    )
    
    async def extract_concept_nodes(self, 
                                  papers: List[Dict[str, Any]], 
                                  hypotheses: List[ScientificHypothesis]):
        """Extract concept nodes from papers and hypotheses"""
        # Combine all text content
        all_texts = []
        
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
            all_texts.append(text)
        
        for hypothesis in hypotheses:
            text = f"{hypothesis.title} {hypothesis.description}"
            all_texts.append(text)
        
        # Extract concepts using NLP
        concepts = await self.extract_scientific_concepts(all_texts)
        
        # Add concept nodes
        for concept in concepts:
            concept_id = f"concept_{hash(concept['text'])}"
            
            if not self.graph.has_node(concept_id):
                self.graph.add_node(
                    concept_id,
                    type='concept',
                    text=concept['text'],
                    category=concept['category'],
                    frequency=concept['frequency']
                )
                
                # Generate embeddings
                embedding = self.semantic_model.encode(concept['text'])
                self.node_embeddings[concept_id] = embedding
    
    async def extract_scientific_concepts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract scientific concepts from text using NLP"""
        concepts = {}
        
        # Load spaCy model
        nlp = spacy.load("en_core_web_sm")
        
        for text in texts:
            doc = nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PRODUCT', 'EVENT', 'ORG', 'GPE', 'PERSON']:
                    concept_text = ent.text.lower()
                    if concept_text not in concepts:
                        concepts[concept_text] = {
                            'text': ent.text,
                            'category': ent.label_,
                            'frequency': 0
                        }
                    concepts[concept_text]['frequency'] += 1
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text) > 3 and chunk.text.lower() not in ['paper', 'study', 'research']:
                    concept_text = chunk.text.lower()
                    if concept_text not in concepts:
                        concepts[concept_text] = {
                            'text': chunk.text,
                            'category': 'NOUN_PHRASE',
                            'frequency': 0
                        }
                    concepts[concept_text]['frequency'] += 1
        
        # Filter concepts by frequency
        filtered_concepts = [
            concept for concept in concepts.values() 
            if concept['frequency'] > 1
        ]
        
        return filtered_concepts[:100]  # Limit to top 100 concepts
    
    async def infer_relationships(self):
        """Infer additional relationships using reasoning rules"""
        # Apply transitivity rules
        for rel_type in self.reasoning_rules['transitivity']:
            await self.apply_transitivity(rel_type)
        
        # Apply symmetry rules
        for rel_type in self.reasoning_rules['symmetry']:
            await self.apply_symmetry(rel_type)
        
        # Infer semantic relationships
        await self.infer_semantic_relationships()
    
    async def apply_transitivity(self, relationship_type: str):
        """Apply transitivity rule for specific relationship type"""
        # Find all edges of the specified type
        edges_to_add = []
        
        for node1 in self.graph.nodes():
            for node2 in self.graph.successors(node1):
                if self.graph.has_edge(node1, node2):
                    edge_data = self.graph.get_edge_data(node1, node2)
                    if any(data.get('relationship') == relationship_type for data in edge_data.values()):
                        # Look for transitive connections
                        for node3 in self.graph.successors(node2):
                            if self.graph.has_edge(node2, node3):
                                edge_data2 = self.graph.get_edge_data(node2, node3)
                                if any(data.get('relationship') == relationship_type for data in edge_data2.values()):
                                    # Add transitive edge if not exists
                                    if not self.graph.has_edge(node1, node3):
                                        edges_to_add.append((node1, node3, relationship_type, 0.5))
        
        # Add inferred edges
        for node1, node3, rel_type, weight in edges_to_add:
            self.graph.add_edge(node1, node3, relationship=rel_type, weight=weight, inferred=True)
    
    async def apply_symmetry(self, relationship_type: str):
        """Apply symmetry rule for specific relationship type"""
        edges_to_add = []
        
        for node1, node2, edge_data in self.graph.edges(data=True):
            if edge_data.get('relationship') == relationship_type:
                # Add symmetric edge if not exists
                if not self.graph.has_edge(node2, node1):
                    edges_to_add.append((node2, node1, relationship_type, edge_data.get('weight', 1.0)))
        
        # Add symmetric edges
        for node1, node2, rel_type, weight in edges_to_add:
            self.graph.add_edge(node1, node2, relationship=rel_type, weight=weight, inferred=True)
    
    async def infer_semantic_relationships(self):
        """Infer relationships based on semantic similarity"""
        nodes = list(self.graph.nodes())
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if node1 in self.node_embeddings and node2 in self.node_embeddings:
                    # Calculate semantic similarity
                    similarity = cosine_similarity(
                        [self.node_embeddings[node1]], 
                        [self.node_embeddings[node2]]
                    )[0][0]
                    
                    # Add similarity relationship if high enough
                    if similarity > 0.7 and not self.graph.has_edge(node1, node2):
                        self.graph.add_edge(
                            node1, node2, 
                            relationship='similar_to', 
                            weight=similarity, 
                            inferred=True
                        )
    
    async def calculate_semantic_similarity(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> float:
        """Calculate semantic similarity between two papers"""
        text1 = f"{paper1.get('title', '')} {paper1.get('abstract', '')}"
        text2 = f"{paper2.get('title', '')} {paper2.get('abstract', '')}"
        
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    async def calculate_hypothesis_paper_similarity(self, 
                                                  hypothesis: ScientificHypothesis, 
                                                  paper: Dict[str, Any]) -> float:
        """Calculate similarity between hypothesis and paper"""
        hyp_text = f"{hypothesis.title} {hypothesis.description}"
        paper_text = f"{paper.get('title', '')} {paper.get('abstract', '')}"
        
        if not hyp_text or not paper_text:
            return 0.0
        
        embeddings = self.semantic_model.encode([hyp_text, paper_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)    asyn
c def calculate_graph_metrics(self) -> Dict[str, float]:
        """Calculate various graph metrics"""
        metrics = {}
        
        if len(self.graph.nodes()) == 0:
            return metrics
        
        # Basic metrics
        metrics['density'] = nx.density(self.graph)
        metrics['average_clustering'] = nx.average_clustering(self.graph.to_undirected())
        
        # Centrality measures
        try:
            degree_centrality = nx.degree_centrality(self.graph)
            metrics['max_degree_centrality'] = max(degree_centrality.values())
            metrics['avg_degree_centrality'] = sum(degree_centrality.values()) / len(degree_centrality)
            
            betweenness_centrality = nx.betweenness_centrality(self.graph)
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
            
            closeness_centrality = nx.closeness_centrality(self.graph)
            metrics['max_closeness_centrality'] = max(closeness_centrality.values())
            
        except Exception as e:
            logger.warning(f"Error calculating centrality metrics: {e}")
        
        # Connected components
        if self.graph.is_directed():
            metrics['strongly_connected_components'] = nx.number_strongly_connected_components(self.graph)
            metrics['weakly_connected_components'] = nx.number_weakly_connected_components(self.graph)
        else:
            metrics['connected_components'] = nx.number_connected_components(self.graph.to_undirected())
        
        return metrics
    
    async def get_node_type_distribution(self) -> Dict[str, int]:
        """Get distribution of node types in the graph"""
        type_counts = {}
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        return type_counts
    
    async def get_relationship_distribution(self) -> Dict[str, int]:
        """Get distribution of relationship types in the graph"""
        rel_counts = {}
        
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'unknown')
            rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
        
        return rel_counts
    
    async def find_central_concepts(self) -> List[Dict[str, Any]]:
        """Find the most central concepts in the knowledge graph"""
        if len(self.graph.nodes()) == 0:
            return []
        
        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Get top central nodes
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        central_concepts = []
        for node_id, centrality_score in sorted_nodes[:10]:
            node_data = self.graph.nodes[node_id]
            central_concepts.append({
                'id': node_id,
                'type': node_data.get('type', 'unknown'),
                'text': node_data.get('text', node_data.get('title', node_id)),
                'centrality_score': centrality_score,
                'connections': len(list(self.graph.neighbors(node_id)))
            })
        
        return central_concepts
    
    async def identify_knowledge_clusters(self) -> List[Dict[str, Any]]:
        """Identify clusters of related knowledge in the graph"""
        clusters = []
        
        if len(self.graph.nodes()) < 3:
            return clusters
        
        try:
            # Convert to undirected graph for community detection
            undirected_graph = self.graph.to_undirected()
            
            # Use simple community detection (in production, use more sophisticated algorithms)
            # For now, use connected components as clusters
            connected_components = list(nx.connected_components(undirected_graph))
            
            for i, component in enumerate(connected_components):
                if len(component) > 2:  # Only consider clusters with more than 2 nodes
                    cluster_nodes = []
                    for node_id in component:
                        node_data = self.graph.nodes[node_id]
                        cluster_nodes.append({
                            'id': node_id,
                            'type': node_data.get('type', 'unknown'),
                            'text': node_data.get('text', node_data.get('title', node_id))
                        })
                    
                    clusters.append({
                        'cluster_id': i,
                        'size': len(component),
                        'nodes': cluster_nodes[:10],  # Limit to first 10 nodes
                        'theme': await self.identify_cluster_theme(component)
                    })
            
        except Exception as e:
            logger.warning(f"Error identifying knowledge clusters: {e}")
        
        return clusters
    
    async def identify_cluster_theme(self, cluster_nodes: set) -> str:
        """Identify the main theme of a knowledge cluster"""
        # Collect text from all nodes in the cluster
        cluster_texts = []
        
        for node_id in cluster_nodes:
            node_data = self.graph.nodes[node_id]
            text = node_data.get('text', node_data.get('title', ''))
            if text:
                cluster_texts.append(text)
        
        if not cluster_texts:
            return "Unknown theme"
        
        # Use TF-IDF to find key terms
        try:
            vectorizer = TfidfVectorizer(max_features=5, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-3:]  # Top 3 terms
            
            theme_terms = [feature_names[i] for i in top_indices]
            return " + ".join(theme_terms)
            
        except Exception as e:
            logger.warning(f"Error identifying cluster theme: {e}")
            return "Mixed topics"class A
utomatedPeerReview:
    """Automated peer review and validation system"""
    
    def __init__(self):
        self.review_criteria = {}
        self.quality_metrics = {}
        self.validation_models = {}
        self.initialize_review_system()
    
    def initialize_review_system(self):
        """Initialize the automated peer review system"""
        self.review_criteria = {
            'novelty': {
                'weight': 0.25,
                'description': 'Originality and innovation of the research',
                'metrics': ['literature_gap', 'unique_approach', 'new_insights']
            },
            'methodology': {
                'weight': 0.25,
                'description': 'Soundness of research methodology',
                'metrics': ['experimental_design', 'statistical_validity', 'reproducibility']
            },
            'significance': {
                'weight': 0.20,
                'description': 'Scientific and practical significance',
                'metrics': ['impact_potential', 'theoretical_contribution', 'practical_applications']
            },
            'clarity': {
                'weight': 0.15,
                'description': 'Clarity of presentation and writing',
                'metrics': ['writing_quality', 'logical_structure', 'figure_quality']
            },
            'validity': {
                'weight': 0.15,
                'description': 'Validity of conclusions and interpretations',
                'metrics': ['evidence_support', 'logical_reasoning', 'limitation_acknowledgment']
            }
        }
    
    async def review_hypothesis(self, hypothesis: ScientificHypothesis) -> Dict[str, Any]:
        """Conduct automated peer review of a scientific hypothesis"""
        try:
            review_results = {
                'hypothesis_id': hypothesis.id,
                'overall_score': 0.0,
                'criterion_scores': {},
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'validation_status': ValidationStatus.PENDING,
                'review_timestamp': datetime.now(),
                'detailed_feedback': {}
            }
            
            # Evaluate each criterion
            total_weighted_score = 0.0
            
            for criterion, config in self.review_criteria.items():
                score = await self.evaluate_criterion(hypothesis, criterion, config)
                review_results['criterion_scores'][criterion] = score
                total_weighted_score += score * config['weight']
            
            review_results['overall_score'] = total_weighted_score
            
            # Generate feedback
            review_results['strengths'] = await self.identify_strengths(hypothesis, review_results['criterion_scores'])
            review_results['weaknesses'] = await self.identify_weaknesses(hypothesis, review_results['criterion_scores'])
            review_results['recommendations'] = await self.generate_recommendations(hypothesis, review_results)
            
            # Determine validation status
            review_results['validation_status'] = await self.determine_validation_status(review_results['overall_score'])
            
            return review_results
            
        except Exception as e:
            logger.error(f"Error in automated peer review: {e}")
            raise
    
    async def evaluate_criterion(self, 
                                hypothesis: ScientificHypothesis, 
                                criterion: str, 
                                config: Dict[str, Any]) -> float:
        """Evaluate a specific review criterion"""
        if criterion == 'novelty':
            return await self.evaluate_novelty(hypothesis)
        elif criterion == 'methodology':
            return await self.evaluate_methodology(hypothesis)
        elif criterion == 'significance':
            return await self.evaluate_significance(hypothesis)
        elif criterion == 'clarity':
            return await self.evaluate_clarity(hypothesis)
        elif criterion == 'validity':
            return await self.evaluate_validity(hypothesis)
        else:
            return 0.5  # Default neutral score
    
    async def evaluate_novelty(self, hypothesis: ScientificHypothesis) -> float:
        """Evaluate the novelty of the hypothesis"""
        # Use the pre-calculated novelty score
        base_score = hypothesis.novelty_score
        
        # Additional novelty checks
        novelty_indicators = [
            'novel', 'new', 'first', 'unprecedented', 'innovative',
            'breakthrough', 'original', 'unique', 'pioneering'
        ]
        
        text = f"{hypothesis.title} {hypothesis.description}".lower()
        indicator_count = sum(1 for indicator in novelty_indicators if indicator in text)
        
        # Boost score based on novelty indicators
        novelty_boost = min(0.3, indicator_count * 0.05)
        
        return min(1.0, base_score + novelty_boost)
    
    async def evaluate_methodology(self, hypothesis: ScientificHypothesis) -> float:
        """Evaluate the methodological soundness"""
        methodology_score = 0.0
        
        # Check for testability
        methodology_score += hypothesis.testability_score * 0.4
        
        # Check for mathematical formulation
        if hypothesis.mathematical_formulation:
            methodology_score += 0.3
        
        # Check for experimental design reference
        if hypothesis.experimental_design:
            methodology_score += 0.3
        
        return min(1.0, methodology_score)
    
    async def evaluate_significance(self, hypothesis: ScientificHypothesis) -> float:
        """Evaluate the scientific significance"""
        # Use pre-calculated impact prediction
        base_score = hypothesis.impact_prediction
        
        # Check for significance indicators
        significance_indicators = [
            'important', 'significant', 'critical', 'major', 'substantial',
            'breakthrough', 'transformative', 'paradigm', 'revolutionary'
        ]
        
        text = f"{hypothesis.title} {hypothesis.description}".lower()
        indicator_count = sum(1 for indicator in significance_indicators if indicator in text)
        
        significance_boost = min(0.2, indicator_count * 0.04)
        
        return min(1.0, base_score + significance_boost)
    
    async def evaluate_clarity(self, hypothesis: ScientificHypothesis) -> float:
        """Evaluate the clarity of presentation"""
        clarity_score = 0.0
        
        # Check description length (not too short, not too long)
        desc_length = len(hypothesis.description.split())
        if 20 <= desc_length <= 200:
            clarity_score += 0.3
        elif 10 <= desc_length <= 300:
            clarity_score += 0.2
        else:
            clarity_score += 0.1
        
        # Check for clear structure
        structure_indicators = ['because', 'therefore', 'thus', 'consequently', 'if', 'then']
        text = hypothesis.description.lower()
        structure_count = sum(1 for indicator in structure_indicators if indicator in text)
        
        clarity_score += min(0.3, structure_count * 0.1)
        
        # Check for technical terms (should have some but not too many)
        technical_terms = len(hypothesis.keywords)
        if 3 <= technical_terms <= 10:
            clarity_score += 0.4
        elif 1 <= technical_terms <= 15:
            clarity_score += 0.2
        
        return min(1.0, clarity_score)
    
    async def evaluate_validity(self, hypothesis: ScientificHypothesis) -> float:
        """Evaluate the validity of the hypothesis"""
        validity_score = 0.0
        
        # Use confidence score as base
        validity_score += hypothesis.confidence_score * 0.5
        
        # Check for supporting evidence
        if hypothesis.supporting_evidence:
            validity_score += min(0.3, len(hypothesis.supporting_evidence) * 0.1)
        
        # Check for acknowledgment of limitations
        limitation_indicators = ['however', 'limitation', 'constraint', 'assumption', 'caveat']
        text = hypothesis.description.lower()
        limitation_count = sum(1 for indicator in limitation_indicators if indicator in text)
        
        if limitation_count > 0:
            validity_score += 0.2
        
        return min(1.0, validity_score)
    
    async def identify_strengths(self, 
                               hypothesis: ScientificHypothesis, 
                               criterion_scores: Dict[str, float]) -> List[str]:
        """Identify strengths of the hypothesis"""
        strengths = []
        
        # Check high-scoring criteria
        for criterion, score in criterion_scores.items():
            if score >= 0.8:
                if criterion == 'novelty':
                    strengths.append("Highly novel and original approach")
                elif criterion == 'methodology':
                    strengths.append("Sound methodological foundation")
                elif criterion == 'significance':
                    strengths.append("High potential scientific impact")
                elif criterion == 'clarity':
                    strengths.append("Clear and well-structured presentation")
                elif criterion == 'validity':
                    strengths.append("Strong logical foundation and evidence")
        
        # Check specific hypothesis attributes
        if hypothesis.mathematical_formulation:
            strengths.append("Includes mathematical formulation for quantitative testing")
        
        if hypothesis.testability_score > 0.7:
            strengths.append("Highly testable with clear experimental predictions")
        
        if len(hypothesis.keywords) >= 5:
            strengths.append("Well-defined scope with relevant keywords")
        
        return strengths
    
    async def identify_weaknesses(self, 
                                hypothesis: ScientificHypothesis, 
                                criterion_scores: Dict[str, float]) -> List[str]:
        """Identify weaknesses of the hypothesis"""
        weaknesses = []
        
        # Check low-scoring criteria
        for criterion, score in criterion_scores.items():
            if score <= 0.4:
                if criterion == 'novelty':
                    weaknesses.append("Limited novelty - similar ideas may exist in literature")
                elif criterion == 'methodology':
                    weaknesses.append("Methodological approach needs strengthening")
                elif criterion == 'significance':
                    weaknesses.append("Scientific significance could be better articulated")
                elif criterion == 'clarity':
                    weaknesses.append("Presentation clarity needs improvement")
                elif criterion == 'validity':
                    weaknesses.append("Logical foundation or evidence support is weak")
        
        # Check specific issues
        if not hypothesis.mathematical_formulation and hypothesis.domain in [ResearchDomain.GEOPHYSICS, ResearchDomain.GEODYNAMICS]:
            weaknesses.append("Lacks mathematical formulation for quantitative domain")
        
        if hypothesis.testability_score < 0.3:
            weaknesses.append("Difficult to test experimentally")
        
        if len(hypothesis.supporting_evidence) == 0:
            weaknesses.append("No supporting evidence provided")
        
        if len(hypothesis.description.split()) < 15:
            weaknesses.append("Description too brief - needs more detail")
        
        return weaknesses
    
    async def generate_recommendations(self, 
                                     hypothesis: ScientificHypothesis, 
                                     review_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        overall_score = review_results['overall_score']
        criterion_scores = review_results['criterion_scores']
        
        # General recommendations based on overall score
        if overall_score < 0.5:
            recommendations.append("Major revision required - consider fundamental restructuring")
        elif overall_score < 0.7:
            recommendations.append("Moderate revision needed - address key weaknesses")
        else:
            recommendations.append("Minor revisions suggested for optimization")
        
        # Specific recommendations based on weak criteria
        if criterion_scores.get('novelty', 0) < 0.5:
            recommendations.append("Conduct more thorough literature review to establish novelty")
            recommendations.append("Highlight unique aspects and innovations more clearly")
        
        if criterion_scores.get('methodology', 0) < 0.5:
            recommendations.append("Develop more detailed experimental or analytical methodology")
            recommendations.append("Consider adding mathematical formulation if applicable")
        
        if criterion_scores.get('significance', 0) < 0.5:
            recommendations.append("Better articulate the potential impact and applications")
            recommendations.append("Connect to broader scientific or societal challenges")
        
        if criterion_scores.get('clarity', 0) < 0.5:
            recommendations.append("Improve writing clarity and logical structure")
            recommendations.append("Add more specific details and examples")
        
        if criterion_scores.get('validity', 0) < 0.5:
            recommendations.append("Strengthen logical reasoning and evidence support")
            recommendations.append("Acknowledge limitations and assumptions more explicitly")
        
        # Domain-specific recommendations
        if hypothesis.domain == ResearchDomain.VOLCANOLOGY:
            recommendations.append("Consider integration with real-time monitoring data")
        elif hypothesis.domain == ResearchDomain.SEISMOLOGY:
            recommendations.append("Validate against historical seismic datasets")
        
        return recommendations
    
    async def determine_validation_status(self, overall_score: float) -> ValidationStatus:
        """Determine validation status based on overall score"""
        if overall_score >= 0.8:
            return ValidationStatus.VALIDATED
        elif overall_score >= 0.6:
            return ValidationStatus.REQUIRES_REVISION
        elif overall_score >= 0.4:
            return ValidationStatus.IN_REVIEW
        else:
            return ValidationStatus.REJECTEDclass S
cientificWritingAssistant:
    """AI-powered scientific writing assistance and generation system"""
    
    def __init__(self):
        self.writing_models = {}
        self.style_guidelines = {}
        self.citation_manager = None
        self.initialize_writing_system()
    
    def initialize_writing_system(self):
        """Initialize the scientific writing system"""
        try:
            # Initialize T5 model for text generation
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.generation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
            
            # Initialize style guidelines
            self.style_guidelines = {
                'academic_paper': {
                    'structure': ['abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion'],
                    'tone': 'formal',
                    'person': 'third_person',
                    'tense': 'past_for_methods_results',
                    'citation_style': 'apa'
                },
                'research_proposal': {
                    'structure': ['abstract', 'background', 'objectives', 'methodology', 'timeline', 'budget'],
                    'tone': 'persuasive_formal',
                    'person': 'first_person_plural',
                    'tense': 'future_for_plans',
                    'citation_style': 'apa'
                },
                'grant_proposal': {
                    'structure': ['summary', 'significance', 'innovation', 'approach', 'team', 'budget'],
                    'tone': 'persuasive',
                    'person': 'first_person_plural',
                    'tense': 'mixed',
                    'citation_style': 'agency_specific'
                }
            }
            
            logger.info("Scientific writing system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing writing system: {e}")
    
    async def generate_paper_draft(self, 
                                 hypothesis: ScientificHypothesis, 
                                 experiment_design: ExperimentDesign,
                                 literature_analysis: LiteratureAnalysis) -> Dict[str, str]:
        """Generate a complete paper draft from hypothesis and supporting materials"""
        try:
            paper_sections = {}
            
            # Generate each section
            paper_sections['title'] = await self.generate_title(hypothesis)
            paper_sections['abstract'] = await self.generate_abstract(hypothesis, experiment_design)
            paper_sections['introduction'] = await self.generate_introduction(hypothesis, literature_analysis)
            paper_sections['methods'] = await self.generate_methods_section(experiment_design)
            paper_sections['results'] = await self.generate_results_placeholder(experiment_design)
            paper_sections['discussion'] = await self.generate_discussion(hypothesis, literature_analysis)
            paper_sections['conclusion'] = await self.generate_conclusion(hypothesis)
            paper_sections['references'] = await self.generate_references(literature_analysis)
            
            return paper_sections
            
        except Exception as e:
            logger.error(f"Error generating paper draft: {e}")
            raise
    
    async def generate_title(self, hypothesis: ScientificHypothesis) -> str:
        """Generate an appropriate scientific title"""
        # Extract key concepts from hypothesis
        key_concepts = hypothesis.keywords[:3]
        domain = hypothesis.domain.value
        
        # Create title templates
        title_templates = [
            f"{key_concepts[0].title()} {key_concepts[1] if len(key_concepts) > 1 else 'Analysis'} in {domain.title()}: {hypothesis.title.replace('Hypothesis: ', '')}",
            f"Novel Insights into {key_concepts[0].title()} through {domain.title()} Analysis",
            f"Investigating {key_concepts[0].title()} Patterns in {domain.title()} Systems",
            f"Advanced {domain.title()} Analysis of {key_concepts[0].title()} Phenomena"
        ]
        
        # Select best title based on length and clarity
        best_title = min(title_templates, key=len)
        
        # Ensure title is not too long
        if len(best_title) > 100:
            best_title = f"{key_concepts[0].title()} Analysis in {domain.title()}"
        
        return best_title
    
    async def generate_abstract(self, 
                              hypothesis: ScientificHypothesis, 
                              experiment_design: ExperimentDesign) -> str:
        """Generate abstract section"""
        abstract_template = f"""
        Background: {hypothesis.description}
        
        Objective: To test the hypothesis that {hypothesis.description.lower()}
        
        Methods: {experiment_design.methodology} {experiment_design.statistical_analysis}
        
        Expected Results: {' '.join(experiment_design.expected_outcomes[:2])}
        
        Significance: This research addresses a critical gap in {hypothesis.domain.value} 
        and has potential implications for {', '.join(hypothesis.keywords[:3])}.
        """
        
        # Clean up the abstract
        abstract = ' '.join(abstract_template.split())
        abstract = abstract.replace('  ', ' ').strip()
        
        return abstract
    
    async def generate_introduction(self, 
                                  hypothesis: ScientificHypothesis, 
                                  literature_analysis: LiteratureAnalysis) -> str:
        """Generate introduction section"""
        introduction_parts = []
        
        # Background context
        introduction_parts.append(f"The field of {hypothesis.domain.value} has seen significant advances in recent years.")
        
        # Literature context
        if literature_analysis.key_findings:
            introduction_parts.append(f"Recent research has identified {', '.join(literature_analysis.key_findings[:2])}.")
        
        # Research gap
        if literature_analysis.research_gaps:
            introduction_parts.append(f"However, {literature_analysis.research_gaps[0].lower()}.")
        
        # Hypothesis statement
        introduction_parts.append(f"We hypothesize that {hypothesis.description.lower()}")
        
        # Significance
        introduction_parts.append(f"This research is significant because it addresses fundamental questions in {hypothesis.domain.value} and has potential applications in {', '.join(hypothesis.keywords[:2])}.")
        
        return ' '.join(introduction_parts)
    
    async def generate_methods_section(self, experiment_design: ExperimentDesign) -> str:
        """Generate methods section"""
        methods_parts = []
        
        # Study design
        methods_parts.append(f"Study Design: {experiment_design.methodology}")
        
        # Variables
        if experiment_design.variables:
            indep_vars = experiment_design.variables.get('independent', [])
            dep_vars = experiment_design.variables.get('dependent', [])
            
            if indep_vars:
                methods_parts.append(f"Independent variables included {', '.join(indep_vars)}.")
            if dep_vars:
                methods_parts.append(f"Dependent variables measured were {', '.join(dep_vars)}.")
        
        # Measurements
        if experiment_design.measurements:
            methods_parts.append(f"Measurements: {' '.join(experiment_design.measurements[:2])}")
        
        # Statistical analysis
        methods_parts.append(f"Statistical Analysis: {experiment_design.statistical_analysis}")
        
        # Controls
        if experiment_design.controls:
            methods_parts.append(f"Controls: {', '.join(experiment_design.controls[:3])}")
        
        return ' '.join(methods_parts)
    
    async def generate_results_placeholder(self, experiment_design: ExperimentDesign) -> str:
        """Generate results section placeholder"""
        results_parts = []
        
        results_parts.append("[Results section to be completed after data collection and analysis]")
        
        # Expected results framework
        if experiment_design.expected_outcomes:
            results_parts.append(f"Expected findings include: {', '.join(experiment_design.expected_outcomes)}")
        
        results_parts.append("[Insert statistical results, figures, and tables here]")
        results_parts.append("[Include significance tests and confidence intervals]")
        
        return ' '.join(results_parts)
    
    async def generate_discussion(self, 
                                hypothesis: ScientificHypothesis, 
                                literature_analysis: LiteratureAnalysis) -> str:
        """Generate discussion section"""
        discussion_parts = []
        
        # Interpretation of results
        discussion_parts.append("[Interpretation of results in context of the hypothesis]")
        
        # Comparison with literature
        if literature_analysis.key_findings:
            discussion_parts.append(f"These findings are consistent with previous research showing {literature_analysis.key_findings[0]}.")
        
        # Implications
        discussion_parts.append(f"The implications of this research for {hypothesis.domain.value} include improved understanding of {', '.join(hypothesis.keywords[:2])}.")
        
        # Limitations
        discussion_parts.append("Limitations of this study include [insert specific limitations based on experimental design and data collection].")
        
        # Future research
        discussion_parts.append("Future research should investigate [insert specific future directions based on findings].")
        
        return ' '.join(discussion_parts)
    
    async def generate_conclusion(self, hypothesis: ScientificHypothesis) -> str:
        """Generate conclusion section"""
        conclusion_parts = []
        
        conclusion_parts.append(f"This study investigated {hypothesis.description.lower()}")
        conclusion_parts.append("[Summary of key findings]")
        conclusion_parts.append(f"The results contribute to our understanding of {hypothesis.domain.value} and have implications for {', '.join(hypothesis.keywords[:2])}.")
        conclusion_parts.append("[Statement about hypothesis support/rejection based on results]")
        
        return ' '.join(conclusion_parts)
    
    async def generate_references(self, literature_analysis: LiteratureAnalysis) -> str:
        """Generate references section placeholder"""
        references = []
        
        references.append("[References to be formatted according to journal guidelines]")
        references.append(f"[Approximately {literature_analysis.papers_analyzed} relevant papers identified in literature review]")
        references.append("[Include key papers related to methodology and theoretical framework]")
        
        return ' '.join(references)
    
    async def improve_writing_quality(self, text: str, section_type: str) -> str:
        """Improve the quality of scientific writing"""
        improvements = []
        
        # Check sentence length
        sentences = text.split('.')
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        if long_sentences:
            improvements.append("Consider breaking down long sentences for clarity")
        
        # Check passive voice (simplified check)
        passive_indicators = ['was', 'were', 'been', 'being']
        passive_count = sum(1 for indicator in passive_indicators if indicator in text.lower())
        if passive_count > len(text.split()) * 0.1:
            improvements.append("Consider reducing passive voice usage")
        
        # Check technical term density
        if section_type == 'abstract':
            word_count = len(text.split())
            if word_count > 300:
                improvements.append("Abstract may be too long - consider condensing")
        
        # Return improved text (placeholder - would use more sophisticated NLP)
        if improvements:
            improved_text = f"{text}\n\n[Writing suggestions: {'; '.join(improvements)}]"
        else:
            improved_text = text
        
        return improved_text
    
    async def generate_grant_proposal(self, 
                                    hypothesis: ScientificHypothesis,
                                    experiment_design: ExperimentDesign,
                                    budget_estimate: float) -> Dict[str, str]:
        """Generate grant proposal sections"""
        proposal_sections = {}
        
        proposal_sections['project_summary'] = await self.generate_project_summary(hypothesis, experiment_design)
        proposal_sections['significance'] = await self.generate_significance_section(hypothesis)
        proposal_sections['innovation'] = await self.generate_innovation_section(hypothesis)
        proposal_sections['approach'] = await self.generate_approach_section(experiment_design)
        proposal_sections['budget_justification'] = await self.generate_budget_justification(experiment_design, budget_estimate)
        
        return proposal_sections
    
    async def generate_project_summary(self, 
                                     hypothesis: ScientificHypothesis,
                                     experiment_design: ExperimentDesign) -> str:
        """Generate project summary for grant proposal"""
        summary_parts = []
        
        summary_parts.append(f"Project Title: {hypothesis.title}")
        summary_parts.append(f"Research Domain: {hypothesis.domain.value}")
        summary_parts.append(f"Objective: {experiment_design.objective}")
        summary_parts.append(f"Methodology: {experiment_design.methodology}")
        summary_parts.append(f"Expected Impact: This research will advance understanding of {', '.join(hypothesis.keywords[:3])} in {hypothesis.domain.value}.")
        
        return ' '.join(summary_parts)
    
    async def generate_significance_section(self, hypothesis: ScientificHypothesis) -> str:
        """Generate significance section for grant proposal"""
        significance_parts = []
        
        significance_parts.append(f"This research addresses a critical knowledge gap in {hypothesis.domain.value}.")
        significance_parts.append(f"The hypothesis that {hypothesis.description.lower()} represents a novel approach to understanding {', '.join(hypothesis.keywords[:2])}.")
        significance_parts.append(f"Potential applications include improved {hypothesis.domain.value} modeling and prediction capabilities.")
        
        return ' '.join(significance_parts)
    
    async def generate_innovation_section(self, hypothesis: ScientificHypothesis) -> str:
        """Generate innovation section for grant proposal"""
        innovation_parts = []
        
        innovation_parts.append(f"Innovation Score: {hypothesis.novelty_score:.2f}/1.0")
        innovation_parts.append("This research introduces novel methodological approaches and theoretical frameworks.")
        
        if hypothesis.mathematical_formulation:
            innovation_parts.append("The mathematical formulation provides a quantitative framework for testing the hypothesis.")
        
        innovation_parts.append(f"The interdisciplinary approach combining {hypothesis.domain.value} with advanced analytical methods represents a significant innovation.")
        
        return ' '.join(innovation_parts)
    
    async def generate_approach_section(self, experiment_design: ExperimentDesign) -> str:
        """Generate approach section for grant proposal"""
        approach_parts = []
        
        approach_parts.append(f"Research Approach: {experiment_design.methodology}")
        approach_parts.append(f"Timeline: {experiment_design.timeline}")
        approach_parts.append(f"Risk Mitigation: {experiment_design.risk_assessment}")
        approach_parts.append(f"Expected Outcomes: {', '.join(experiment_design.expected_outcomes)}")
        
        return ' '.join(approach_parts)
    
    async def generate_budget_justification(self, 
                                          experiment_design: ExperimentDesign,
                                          budget_estimate: float) -> str:
        """Generate budget justification for grant proposal"""
        budget_parts = []
        
        budget_parts.append(f"Total Budget Request: ${budget_estimate:,.2f}")
        budget_parts.append(f"Personnel: {experiment_design.resource_requirements.get('personnel', 'TBD')}")
        budget_parts.append(f"Equipment: {experiment_design.resource_requirements.get('equipment', 'TBD')}")
        budget_parts.append(f"Materials: {experiment_design.resource_requirements.get('materials', 'TBD')}")
        budget_parts.append("Budget justification based on standard rates and market prices for geological research equipment and personnel.")
        
        return ' '.join(budget_parts)class Re
searchCollaborationMatcher:
    """AI-powered research collaboration and matching system"""
    
    def __init__(self):
        self.researcher_profiles = {}
        self.collaboration_history = {}
        self.matching_algorithms = {}
        self.network_analyzer = None
        self.initialize_matching_system()
    
    def initialize_matching_system(self):
        """Initialize the collaboration matching system"""
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collaboration_network = nx.Graph()
        
        # Initialize matching criteria
        self.matching_criteria = {
            'research_interests': 0.3,
            'complementary_skills': 0.25,
            'geographic_proximity': 0.15,
            'career_stage_compatibility': 0.1,
            'publication_compatibility': 0.1,
            'collaboration_history': 0.1
        }
    
    async def find_collaborations(self, 
                                researcher_profile: Dict[str, Any], 
                                candidate_pool: List[Dict[str, Any]]) -> List[ResearchCollaboration]:
        """Find potential research collaborations for a researcher"""
        try:
            collaborations = []
            
            for candidate in candidate_pool:
                # Calculate compatibility score
                compatibility = await self.calculate_compatibility(researcher_profile, candidate)
                
                if compatibility['overall_score'] > 0.6:  # Threshold for potential collaboration
                    collaboration = ResearchCollaboration(
                        researcher1_id=researcher_profile['id'],
                        researcher2_id=candidate['id'],
                        compatibility_score=compatibility['overall_score'],
                        shared_interests=compatibility['shared_interests'],
                        complementary_skills=compatibility['complementary_skills'],
                        collaboration_potential=await self.assess_collaboration_potential(compatibility),
                        suggested_projects=await self.suggest_collaboration_projects(researcher_profile, candidate),
                        contact_recommendation=await self.generate_contact_recommendation(compatibility)
                    )
                    collaborations.append(collaboration)
            
            # Sort by compatibility score
            collaborations.sort(key=lambda x: x.compatibility_score, reverse=True)
            
            return collaborations[:10]  # Return top 10 matches
            
        except Exception as e:
            logger.error(f"Error finding collaborations: {e}")
            raise
    
    async def calculate_compatibility(self, 
                                   researcher1: Dict[str, Any], 
                                   researcher2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compatibility between two researchers"""
        compatibility = {
            'overall_score': 0.0,
            'shared_interests': [],
            'complementary_skills': [],
            'detailed_scores': {}
        }
        
        # Research interests similarity
        interests_score = await self.calculate_interests_similarity(researcher1, researcher2)
        compatibility['detailed_scores']['research_interests'] = interests_score
        compatibility['shared_interests'] = await self.find_shared_interests(researcher1, researcher2)
        
        # Complementary skills
        skills_score = await self.calculate_skills_complementarity(researcher1, researcher2)
        compatibility['detailed_scores']['complementary_skills'] = skills_score
        compatibility['complementary_skills'] = await self.find_complementary_skills(researcher1, researcher2)
        
        # Geographic proximity
        geo_score = await self.calculate_geographic_compatibility(researcher1, researcher2)
        compatibility['detailed_scores']['geographic_proximity'] = geo_score
        
        # Career stage compatibility
        career_score = await self.calculate_career_compatibility(researcher1, researcher2)
        compatibility['detailed_scores']['career_stage_compatibility'] = career_score
        
        # Publication compatibility
        pub_score = await self.calculate_publication_compatibility(researcher1, researcher2)
        compatibility['detailed_scores']['publication_compatibility'] = pub_score
        
        # Collaboration history
        history_score = await self.calculate_collaboration_history_score(researcher1, researcher2)
        compatibility['detailed_scores']['collaboration_history'] = history_score
        
        # Calculate weighted overall score
        overall_score = 0.0
        for criterion, weight in self.matching_criteria.items():
            score = compatibility['detailed_scores'].get(criterion, 0.0)
            overall_score += score * weight
        
        compatibility['overall_score'] = overall_score
        
        return compatibility
    
    async def calculate_interests_similarity(self, 
                                          researcher1: Dict[str, Any], 
                                          researcher2: Dict[str, Any]) -> float:
        """Calculate similarity in research interests"""
        interests1 = researcher1.get('research_interests', [])
        interests2 = researcher2.get('research_interests', [])
        
        if not interests1 or not interests2:
            return 0.0
        
        # Use semantic similarity
        interests1_text = ' '.join(interests1)
        interests2_text = ' '.join(interests2)
        
        embeddings = self.semantic_model.encode([interests1_text, interests2_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    async def find_shared_interests(self, 
                                  researcher1: Dict[str, Any], 
                                  researcher2: Dict[str, Any]) -> List[str]:
        """Find shared research interests"""
        interests1 = set(researcher1.get('research_interests', []))
        interests2 = set(researcher2.get('research_interests', []))
        
        # Direct overlap
        shared = list(interests1.intersection(interests2))
        
        # Semantic similarity for near-matches
        for interest1 in interests1:
            for interest2 in interests2:
                if interest1 != interest2:
                    embeddings = self.semantic_model.encode([interest1, interest2])
                    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                    
                    if similarity > 0.8:  # High semantic similarity
                        shared.append(f"{interest1} / {interest2}")
        
        return shared[:5]  # Return top 5 shared interests
    
    async def calculate_skills_complementarity(self, 
                                             researcher1: Dict[str, Any], 
                                             researcher2: Dict[str, Any]) -> float:
        """Calculate complementarity of skills"""
        skills1 = set(researcher1.get('skills', []))
        skills2 = set(researcher2.get('skills', []))
        
        # Calculate complementarity (skills that one has but the other doesn't)
        unique_skills1 = skills1 - skills2
        unique_skills2 = skills2 - skills1
        
        total_skills = len(skills1.union(skills2))
        complementary_skills = len(unique_skills1) + len(unique_skills2)
        
        if total_skills == 0:
            return 0.0
        
        # Normalize complementarity score
        complementarity_score = complementary_skills / total_skills
        
        return min(1.0, complementarity_score)
    
    async def find_complementary_skills(self, 
                                      researcher1: Dict[str, Any], 
                                      researcher2: Dict[str, Any]) -> List[str]:
        """Find complementary skills between researchers"""
        skills1 = set(researcher1.get('skills', []))
        skills2 = set(researcher2.get('skills', []))
        
        # Skills that researcher1 has but researcher2 doesn't
        unique_to_1 = list(skills1 - skills2)
        # Skills that researcher2 has but researcher1 doesn't
        unique_to_2 = list(skills2 - skills1)
        
        complementary = []
        complementary.extend([f"R1: {skill}" for skill in unique_to_1[:3]])
        complementary.extend([f"R2: {skill}" for skill in unique_to_2[:3]])
        
        return complementary
    
    async def calculate_geographic_compatibility(self, 
                                              researcher1: Dict[str, Any], 
                                              researcher2: Dict[str, Any]) -> float:
        """Calculate geographic compatibility"""
        location1 = researcher1.get('location', '')
        location2 = researcher2.get('location', '')
        
        if not location1 or not location2:
            return 0.5  # Neutral score if location unknown
        
        # Simple geographic compatibility (in production, use actual distance calculation)
        if location1 == location2:
            return 1.0  # Same location
        
        # Check if same country/region (simplified)
        location1_parts = location1.split(',')
        location2_parts = location2.split(',')
        
        if len(location1_parts) > 1 and len(location2_parts) > 1:
            if location1_parts[-1].strip() == location2_parts[-1].strip():  # Same country
                return 0.7
        
        return 0.3  # Different countries
    
    async def calculate_career_compatibility(self, 
                                           researcher1: Dict[str, Any], 
                                           researcher2: Dict[str, Any]) -> float:
        """Calculate career stage compatibility"""
        career1 = researcher1.get('career_stage', 'unknown')
        career2 = researcher2.get('career_stage', 'unknown')
        
        # Define career stage compatibility matrix
        compatibility_matrix = {
            ('graduate_student', 'graduate_student'): 0.8,
            ('graduate_student', 'postdoc'): 0.9,
            ('graduate_student', 'assistant_professor'): 0.7,
            ('graduate_student', 'associate_professor'): 0.5,
            ('graduate_student', 'full_professor'): 0.3,
            ('postdoc', 'postdoc'): 0.9,
            ('postdoc', 'assistant_professor'): 0.8,
            ('postdoc', 'associate_professor'): 0.7,
            ('postdoc', 'full_professor'): 0.5,
            ('assistant_professor', 'assistant_professor'): 0.9,
            ('assistant_professor', 'associate_professor'): 0.8,
            ('assistant_professor', 'full_professor'): 0.6,
            ('associate_professor', 'associate_professor'): 0.9,
            ('associate_professor', 'full_professor'): 0.8,
            ('full_professor', 'full_professor'): 0.9
        }
        
        # Make symmetric
        key = (career1, career2)
        reverse_key = (career2, career1)
        
        return compatibility_matrix.get(key, compatibility_matrix.get(reverse_key, 0.5))
    
    async def calculate_publication_compatibility(self, 
                                               researcher1: Dict[str, Any], 
                                               researcher2: Dict[str, Any]) -> float:
        """Calculate publication compatibility"""
        pubs1 = researcher1.get('publication_count', 0)
        pubs2 = researcher2.get('publication_count', 0)
        
        if pubs1 == 0 and pubs2 == 0:
            return 0.5  # Neutral if no publications
        
        # Calculate ratio similarity
        max_pubs = max(pubs1, pubs2)
        min_pubs = min(pubs1, pubs2)
        
        if max_pubs == 0:
            return 0.5
        
        ratio = min_pubs / max_pubs
        
        # Prefer similar publication levels
        return ratio
    
    async def calculate_collaboration_history_score(self, 
                                                  researcher1: Dict[str, Any], 
                                                  researcher2: Dict[str, Any]) -> float:
        """Calculate collaboration history score"""
        # Check if they have collaborated before
        researcher1_id = researcher1.get('id', '')
        researcher2_id = researcher2.get('id', '')
        
        # In production, this would check actual collaboration database
        # For now, return neutral score
        return 0.5
    
    async def assess_collaboration_potential(self, compatibility: Dict[str, Any]) -> str:
        """Assess the potential for successful collaboration"""
        overall_score = compatibility['overall_score']
        
        if overall_score >= 0.8:
            return "Excellent collaboration potential - highly recommended"
        elif overall_score >= 0.7:
            return "Good collaboration potential - recommended"
        elif overall_score >= 0.6:
            return "Moderate collaboration potential - worth exploring"
        else:
            return "Limited collaboration potential"
    
    async def suggest_collaboration_projects(self, 
                                          researcher1: Dict[str, Any], 
                                          researcher2: Dict[str, Any]) -> List[str]:
        """Suggest potential collaboration projects"""
        projects = []
        
        # Based on shared interests
        shared_interests = await self.find_shared_interests(researcher1, researcher2)
        for interest in shared_interests[:2]:
            projects.append(f"Joint research on {interest}")
        
        # Based on complementary skills
        complementary_skills = await self.find_complementary_skills(researcher1, researcher2)
        if complementary_skills:
            projects.append("Interdisciplinary project leveraging complementary expertise")
        
        # Domain-specific suggestions
        domains1 = researcher1.get('research_domains', [])
        domains2 = researcher2.get('research_domains', [])
        
        common_domains = set(domains1).intersection(set(domains2))
        for domain in list(common_domains)[:2]:
            projects.append(f"Collaborative study in {domain}")
        
        return projects[:5]  # Return top 5 project suggestions
    
    async def generate_contact_recommendation(self, compatibility: Dict[str, Any]) -> str:
        """Generate personalized contact recommendation"""
        overall_score = compatibility['overall_score']
        shared_interests = compatibility['shared_interests']
        
        if overall_score >= 0.8:
            recommendation = f"Highly recommended collaboration. Suggest initiating contact to discuss shared interests in {', '.join(shared_interests[:2])}."
        elif overall_score >= 0.7:
            recommendation = f"Good potential for collaboration. Consider reaching out to explore opportunities in {', '.join(shared_interests[:2])}."
        else:
            recommendation = f"Moderate compatibility. May be worth a brief discussion about potential collaboration areas."
        
        return recommendationc
lass PatentAnalyzer:
    """Patent analysis and prior art search system"""
    
    def __init__(self):
        self.patent_databases = {}
        self.prior_art_analyzer = None
        self.novelty_assessor = None
        self.initialize_patent_system()
    
    def initialize_patent_system(self):
        """Initialize patent analysis system"""
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Patent classification systems
        self.patent_classes = {
            'geological_instruments': ['G01V', 'G01N'],
            'seismic_analysis': ['G01V1', 'G01V3'],
            'geological_modeling': ['G06F17', 'G06T17'],
            'volcanic_monitoring': ['G01V1', 'G08B21'],
            'mineral_exploration': ['G01V3', 'G01V5']
        }
    
    async def analyze_patent_landscape(self, 
                                     hypothesis: ScientificHypothesis, 
                                     technology_description: str) -> Dict[str, Any]:
        """Analyze patent landscape for a given technology"""
        try:
            analysis = {
                'patent_search_results': [],
                'prior_art_assessment': {},
                'novelty_analysis': {},
                'patentability_score': 0.0,
                'freedom_to_operate': {},
                'competitive_landscape': {},
                'recommendations': []
            }
            
            # Search for relevant patents
            analysis['patent_search_results'] = await self.search_patents(hypothesis, technology_description)
            
            # Assess prior art
            analysis['prior_art_assessment'] = await self.assess_prior_art(
                technology_description, 
                analysis['patent_search_results']
            )
            
            # Analyze novelty
            analysis['novelty_analysis'] = await self.analyze_novelty(
                technology_description, 
                analysis['patent_search_results']
            )
            
            # Calculate patentability score
            analysis['patentability_score'] = await self.calculate_patentability_score(analysis)
            
            # Assess freedom to operate
            analysis['freedom_to_operate'] = await self.assess_freedom_to_operate(
                technology_description, 
                analysis['patent_search_results']
            )
            
            # Analyze competitive landscape
            analysis['competitive_landscape'] = await self.analyze_competitive_landscape(
                analysis['patent_search_results']
            )
            
            # Generate recommendations
            analysis['recommendations'] = await self.generate_patent_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing patent landscape: {e}")
            raise
    
    async def search_patents(self, 
                           hypothesis: ScientificHypothesis, 
                           technology_description: str) -> List[Dict[str, Any]]:
        """Search for relevant patents"""
        # In production, this would integrate with patent databases like USPTO, EPO, etc.
        # For now, simulate patent search results
        
        search_terms = hypothesis.keywords + [hypothesis.domain.value]
        
        # Simulated patent results
        simulated_patents = [
            {
                'patent_number': 'US10123456',
                'title': f'Method for {hypothesis.keywords[0]} analysis in geological systems',
                'abstract': f'A method for analyzing {hypothesis.keywords[0]} using advanced computational techniques...',
                'inventors': ['Smith, J.', 'Johnson, A.'],
                'assignee': 'GeoTech Corp',
                'filing_date': '2020-01-15',
                'publication_date': '2021-07-20',
                'patent_class': 'G01V1/00',
                'relevance_score': 0.85
            },
            {
                'patent_number': 'US10234567',
                'title': f'System for monitoring {hypothesis.domain.value} phenomena',
                'abstract': f'A monitoring system for real-time analysis of {hypothesis.domain.value} data...',
                'inventors': ['Brown, K.', 'Davis, M.'],
                'assignee': 'Monitoring Solutions Inc',
                'filing_date': '2019-05-10',
                'publication_date': '2020-11-15',
                'patent_class': 'G08B21/00',
                'relevance_score': 0.72
            }
        ]
        
        # Calculate actual relevance scores
        for patent in simulated_patents:
            patent_text = f"{patent['title']} {patent['abstract']}"
            relevance = await self.calculate_patent_relevance(technology_description, patent_text)
            patent['relevance_score'] = relevance
        
        # Sort by relevance
        simulated_patents.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return simulated_patents
    
    async def calculate_patent_relevance(self, technology_description: str, patent_text: str) -> float:
        """Calculate relevance between technology and patent"""
        if not technology_description or not patent_text:
            return 0.0
        
        embeddings = self.semantic_model.encode([technology_description, patent_text])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    async def assess_prior_art(self, 
                             technology_description: str, 
                             patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess prior art landscape"""
        assessment = {
            'relevant_patents_count': len([p for p in patents if p['relevance_score'] > 0.7]),
            'closest_prior_art': [],
            'prior_art_gaps': [],
            'novelty_indicators': []
        }
        
        # Identify closest prior art
        high_relevance_patents = [p for p in patents if p['relevance_score'] > 0.7]
        assessment['closest_prior_art'] = high_relevance_patents[:5]
        
        # Identify gaps in prior art
        if len(high_relevance_patents) < 3:
            assessment['prior_art_gaps'].append("Limited prior art in this specific area")
        
        # Check for novelty indicators
        technology_terms = set(technology_description.lower().split())
        
        for patent in patents:
            patent_terms = set(f"{patent['title']} {patent['abstract']}".lower().split())
            unique_tech_terms = technology_terms - patent_terms
            
            if len(unique_tech_terms) > len(technology_terms) * 0.3:
                assessment['novelty_indicators'].append(f"Novel aspects: {', '.join(list(unique_tech_terms)[:5])}")
        
        return assessment
    
    async def analyze_novelty(self, 
                            technology_description: str, 
                            patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze novelty of the technology"""
        novelty_analysis = {
            'novelty_score': 0.0,
            'novel_aspects': [],
            'similar_technologies': [],
            'differentiation_factors': []
        }
        
        if not patents:
            novelty_analysis['novelty_score'] = 0.9
            novelty_analysis['novel_aspects'] = ['No similar patents found']
            return novelty_analysis
        
        # Calculate novelty based on similarity to existing patents
        max_similarity = max([p['relevance_score'] for p in patents])
        novelty_analysis['novelty_score'] = 1.0 - max_similarity
        
        # Identify similar technologies
        similar_patents = [p for p in patents if p['relevance_score'] > 0.6]
        novelty_analysis['similar_technologies'] = [
            f"{p['patent_number']}: {p['title']}" for p in similar_patents[:3]
        ]
        
        # Identify differentiation factors
        technology_words = set(technology_description.lower().split())
        
        for patent in similar_patents:
            patent_words = set(f"{patent['title']} {patent['abstract']}".lower().split())
            unique_words = technology_words - patent_words
            
            if unique_words:
                novelty_analysis['differentiation_factors'].extend(list(unique_words)[:3])
        
        # Remove duplicates and limit
        novelty_analysis['differentiation_factors'] = list(set(novelty_analysis['differentiation_factors']))[:5]
        
        return novelty_analysis
    
    async def calculate_patentability_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall patentability score"""
        factors = {
            'novelty': analysis['novelty_analysis']['novelty_score'] * 0.4,
            'prior_art_gaps': min(0.3, len(analysis['prior_art_assessment']['prior_art_gaps']) * 0.1),
            'technical_complexity': 0.2,  # Assume moderate complexity
            'commercial_potential': 0.1   # Assume some commercial potential
        }
        
        patentability_score = sum(factors.values())
        return min(1.0, patentability_score)
    
    async def assess_freedom_to_operate(self, 
                                      technology_description: str, 
                                      patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess freedom to operate"""
        fto_assessment = {
            'risk_level': 'low',
            'blocking_patents': [],
            'expired_patents': [],
            'licensing_opportunities': [],
            'recommendations': []
        }
        
        # Check for potentially blocking patents
        current_year = datetime.now().year
        
        for patent in patents:
            if patent['relevance_score'] > 0.8:
                filing_year = int(patent['filing_date'][:4])
                
                # Patents typically expire 20 years from filing
                if current_year - filing_year < 20:
                    fto_assessment['blocking_patents'].append(patent)
                else:
                    fto_assessment['expired_patents'].append(patent)
        
        # Assess risk level
        if len(fto_assessment['blocking_patents']) > 2:
            fto_assessment['risk_level'] = 'high'
        elif len(fto_assessment['blocking_patents']) > 0:
            fto_assessment['risk_level'] = 'medium'
        
        # Generate recommendations
        if fto_assessment['risk_level'] == 'high':
            fto_assessment['recommendations'].append("Consider design-around strategies")
            fto_assessment['recommendations'].append("Explore licensing opportunities")
        elif fto_assessment['risk_level'] == 'medium':
            fto_assessment['recommendations'].append("Conduct detailed patent analysis")
        
        return fto_assessment
    
    async def analyze_competitive_landscape(self, patents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze competitive patent landscape"""
        landscape = {
            'key_players': {},
            'patent_trends': {},
            'technology_clusters': [],
            'white_spaces': []
        }
        
        # Identify key players (assignees)
        assignee_counts = {}
        for patent in patents:
            assignee = patent.get('assignee', 'Unknown')
            assignee_counts[assignee] = assignee_counts.get(assignee, 0) + 1
        
        landscape['key_players'] = dict(sorted(assignee_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Analyze patent trends over time
        year_counts = {}
        for patent in patents:
            year = patent['filing_date'][:4]
            year_counts[year] = year_counts.get(year, 0) + 1
        
        landscape['patent_trends'] = year_counts
        
        # Identify technology clusters (simplified)
        patent_classes = [patent.get('patent_class', '') for patent in patents]
        class_counts = {}
        for pc in patent_classes:
            if pc:
                main_class = pc.split('/')[0]  # Get main class
                class_counts[main_class] = class_counts.get(main_class, 0) + 1
        
        landscape['technology_clusters'] = list(class_counts.keys())
        
        # Identify potential white spaces
        if len(patents) < 5:
            landscape['white_spaces'].append("Limited patent activity - potential opportunity")
        
        return landscape
    
    async def generate_patent_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate patent strategy recommendations"""
        recommendations = []
        
        patentability_score = analysis['patentability_score']
        novelty_score = analysis['novelty_analysis']['novelty_score']
        fto_risk = analysis['freedom_to_operate']['risk_level']
        
        # Patentability recommendations
        if patentability_score > 0.7:
            recommendations.append("High patentability - recommend filing patent application")
        elif patentability_score > 0.5:
            recommendations.append("Moderate patentability - consider strengthening novel aspects")
        else:
            recommendations.append("Low patentability - focus on trade secret or other IP strategies")
        
        # Novelty recommendations
        if novelty_score > 0.8:
            recommendations.append("High novelty - strong patent potential")
        elif novelty_score < 0.4:
            recommendations.append("Limited novelty - consider design-around approaches")
        
        # Freedom to operate recommendations
        if fto_risk == 'high':
            recommendations.append("High FTO risk - conduct detailed clearance analysis")
        elif fto_risk == 'medium':
            recommendations.append("Moderate FTO risk - monitor key patents")
        
        # Strategic recommendations
        key_players = analysis['competitive_landscape']['key_players']
        if key_players:
            top_player = list(key_players.keys())[0]
            recommendations.append(f"Monitor {top_player} patent activity closely")
        
        return recommendationsclass
 ScientificImpactAnalyzer:
    """Scientific impact and citation analysis system"""
    
    def __init__(self):
        self.citation_networks = {}
        self.impact_metrics = {}
        self.prediction_models = {}
        self.initialize_impact_system()
    
    def initialize_impact_system(self):
        """Initialize scientific impact analysis system"""
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.citation_graph = nx.DiGraph()
        
        # Impact metrics definitions
        self.impact_metrics_definitions = {
            'citation_count': 'Total number of citations received',
            'h_index': 'H-index based on citation distribution',
            'i10_index': 'Number of papers with at least 10 citations',
            'field_weighted_citation_impact': 'Citations relative to field average',
            'altmetric_score': 'Alternative metrics (social media, news, etc.)',
            'collaboration_index': 'Breadth of research collaborations',
            'interdisciplinary_index': 'Cross-disciplinary impact measure'
        }
    
    async def analyze_scientific_impact(self, 
                                      researcher_profile: Dict[str, Any], 
                                      publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scientific impact of a researcher or research area"""
        try:
            impact_analysis = {
                'researcher_id': researcher_profile.get('id', 'unknown'),
                'basic_metrics': {},
                'advanced_metrics': {},
                'citation_network_analysis': {},
                'temporal_impact_trends': {},
                'field_comparison': {},
                'impact_prediction': {},
                'recommendations': []
            }
            
            # Calculate basic metrics
            impact_analysis['basic_metrics'] = await self.calculate_basic_metrics(publications)
            
            # Calculate advanced metrics
            impact_analysis['advanced_metrics'] = await self.calculate_advanced_metrics(
                researcher_profile, publications
            )
            
            # Analyze citation network
            impact_analysis['citation_network_analysis'] = await self.analyze_citation_network(publications)
            
            # Analyze temporal trends
            impact_analysis['temporal_impact_trends'] = await self.analyze_temporal_trends(publications)
            
            # Compare with field averages
            impact_analysis['field_comparison'] = await self.compare_with_field(
                researcher_profile, impact_analysis['basic_metrics']
            )
            
            # Predict future impact
            impact_analysis['impact_prediction'] = await self.predict_future_impact(
                impact_analysis['temporal_impact_trends']
            )
            
            # Generate recommendations
            impact_analysis['recommendations'] = await self.generate_impact_recommendations(impact_analysis)
            
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing scientific impact: {e}")
            raise
    
    async def calculate_basic_metrics(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate basic citation metrics"""
        metrics = {
            'total_publications': len(publications),
            'total_citations': 0,
            'average_citations_per_paper': 0.0,
            'h_index': 0,
            'i10_index': 0,
            'most_cited_paper': {},
            'recent_publications': 0
        }
        
        if not publications:
            return metrics
        
        # Extract citation counts
        citation_counts = []
        total_citations = 0
        
        current_year = datetime.now().year
        
        for pub in publications:
            citations = pub.get('citations', 0)
            citation_counts.append(citations)
            total_citations += citations
            
            # Count recent publications (last 3 years)
            pub_year = str(pub.get('published', ''))[:4]
            if pub_year.isdigit() and int(pub_year) >= current_year - 3:
                metrics['recent_publications'] += 1
        
        metrics['total_citations'] = total_citations
        metrics['average_citations_per_paper'] = total_citations / len(publications) if publications else 0
        
        # Calculate h-index
        citation_counts_sorted = sorted(citation_counts, reverse=True)
        h_index = 0
        for i, citations in enumerate(citation_counts_sorted):
            if citations >= i + 1:
                h_index = i + 1
            else:
                break
        metrics['h_index'] = h_index
        
        # Calculate i10-index
        metrics['i10_index'] = len([c for c in citation_counts if c >= 10])
        
        # Find most cited paper
        if citation_counts:
            max_citations = max(citation_counts)
            most_cited_idx = citation_counts.index(max_citations)
            metrics['most_cited_paper'] = {
                'title': publications[most_cited_idx].get('title', ''),
                'citations': max_citations,
                'year': publications[most_cited_idx].get('published', '')[:4]
            }
        
        return metrics
    
    async def calculate_advanced_metrics(self, 
                                       researcher_profile: Dict[str, Any], 
                                       publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate advanced impact metrics"""
        metrics = {
            'field_weighted_citation_impact': 0.0,
            'collaboration_index': 0.0,
            'interdisciplinary_index': 0.0,
            'innovation_index': 0.0,
            'international_collaboration_ratio': 0.0,
            'journal_impact_score': 0.0
        }
        
        if not publications:
            return metrics
        
        # Calculate collaboration index
        total_coauthors = 0
        unique_coauthors = set()
        
        for pub in publications:
            authors = pub.get('authors', [])
            if len(authors) > 1:
                coauthors = len(authors) - 1  # Exclude the researcher themselves
                total_coauthors += coauthors
                unique_coauthors.update(authors)
        
        metrics['collaboration_index'] = len(unique_coauthors) / len(publications) if publications else 0
        
        # Calculate interdisciplinary index (based on keyword diversity)
        all_keywords = []
        for pub in publications:
            # Extract keywords from title and abstract
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            doc = spacy.load("en_core_web_sm")(text)
            keywords = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'EVENT', 'ORG']]
            all_keywords.extend(keywords)
        
        unique_keywords = len(set(all_keywords))
        total_keywords = len(all_keywords)
        metrics['interdisciplinary_index'] = unique_keywords / max(1, total_keywords)
        
        # Calculate innovation index (based on novelty indicators)
        innovation_indicators = ['novel', 'new', 'first', 'innovative', 'breakthrough']
        innovation_count = 0
        
        for pub in publications:
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}".lower()
            innovation_count += sum(1 for indicator in innovation_indicators if indicator in text)
        
        metrics['innovation_index'] = innovation_count / len(publications) if publications else 0
        
        # Estimate international collaboration ratio
        international_collabs = 0
        for pub in publications:
            authors = pub.get('authors', [])
            # Simple heuristic: if more than 3 authors, likely international
            if len(authors) > 3:
                international_collabs += 1
        
        metrics['international_collaboration_ratio'] = international_collabs / len(publications) if publications else 0
        
        return metrics
    
    async def analyze_citation_network(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze citation network structure"""
        network_analysis = {
            'network_size': 0,
            'clustering_coefficient': 0.0,
            'centrality_measures': {},
            'influential_papers': [],
            'citation_clusters': []
        }
        
        # Build citation network (simplified)
        self.citation_graph.clear()
        
        # Add nodes for each publication
        for i, pub in enumerate(publications):
            pub_id = f"pub_{i}"
            self.citation_graph.add_node(pub_id, **pub)
        
        # Add edges based on citation relationships (simplified)
        # In production, this would use actual citation data
        for i, pub1 in enumerate(publications):
            for j, pub2 in enumerate(publications):
                if i != j:
                    # Simple heuristic: papers with similar keywords might cite each other
                    similarity = await self.calculate_publication_similarity(pub1, pub2)
                    if similarity > 0.7:
                        self.citation_graph.add_edge(f"pub_{i}", f"pub_{j}", weight=similarity)
        
        network_analysis['network_size'] = len(self.citation_graph.nodes())
        
        if len(self.citation_graph.nodes()) > 1:
            # Calculate clustering coefficient
            network_analysis['clustering_coefficient'] = nx.average_clustering(
                self.citation_graph.to_undirected()
            )
            
            # Calculate centrality measures
            try:
                degree_centrality = nx.degree_centrality(self.citation_graph)
                betweenness_centrality = nx.betweenness_centrality(self.citation_graph)
                
                network_analysis['centrality_measures'] = {
                    'max_degree_centrality': max(degree_centrality.values()) if degree_centrality else 0,
                    'max_betweenness_centrality': max(betweenness_centrality.values()) if betweenness_centrality else 0
                }
                
                # Identify influential papers
                top_degree_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:3]
                for node_id, centrality in top_degree_nodes:
                    node_data = self.citation_graph.nodes[node_id]
                    network_analysis['influential_papers'].append({
                        'title': node_data.get('title', ''),
                        'centrality_score': centrality
                    })
                    
            except Exception as e:
                logger.warning(f"Error calculating centrality measures: {e}")
        
        return network_analysis
    
    async def calculate_publication_similarity(self, pub1: Dict[str, Any], pub2: Dict[str, Any]) -> float:
        """Calculate similarity between two publications"""
        text1 = f"{pub1.get('title', '')} {pub1.get('abstract', '')}"
        text2 = f"{pub2.get('title', '')} {pub2.get('abstract', '')}"
        
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return float(similarity)
    
    async def analyze_temporal_trends(self, publications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal trends in impact"""
        trends = {
            'publication_timeline': {},
            'citation_timeline': {},
            'impact_trajectory': {},
            'career_phase_analysis': {}
        }
        
        # Group publications by year
        year_data = {}
        for pub in publications:
            year = str(pub.get('published', ''))[:4]
            if year.isdigit():
                if year not in year_data:
                    year_data[year] = {'count': 0, 'citations': 0}
                year_data[year]['count'] += 1
                year_data[year]['citations'] += pub.get('citations', 0)
        
        trends['publication_timeline'] = {year: data['count'] for year, data in year_data.items()}
        trends['citation_timeline'] = {year: data['citations'] for year, data in year_data.items()}
        
        # Calculate impact trajectory
        if year_data:
            years = sorted(year_data.keys())
            if len(years) > 1:
                early_years = years[:len(years)//2]
                recent_years = years[len(years)//2:]
                
                early_avg_citations = sum(year_data[y]['citations'] for y in early_years) / len(early_years)
                recent_avg_citations = sum(year_data[y]['citations'] for y in recent_years) / len(recent_years)
                
                trends['impact_trajectory'] = {
                    'early_career_avg_citations': early_avg_citations,
                    'recent_avg_citations': recent_avg_citations,
                    'trajectory': 'increasing' if recent_avg_citations > early_avg_citations else 'stable_or_decreasing'
                }
        
        return trends
    
    async def compare_with_field(self, 
                               researcher_profile: Dict[str, Any], 
                               basic_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare metrics with field averages"""
        field_comparison = {
            'field': researcher_profile.get('research_field', 'unknown'),
            'percentile_rankings': {},
            'field_averages': {},
            'relative_performance': {}
        }
        
        # Field averages (these would come from databases in production)
        field_averages = {
            'volcanology': {'h_index': 15, 'avg_citations_per_paper': 25, 'total_publications': 30},
            'seismology': {'h_index': 18, 'avg_citations_per_paper': 30, 'total_publications': 35},
            'geodynamics': {'h_index': 20, 'avg_citations_per_paper': 35, 'total_publications': 40},
            'default': {'h_index': 12, 'avg_citations_per_paper': 20, 'total_publications': 25}
        }
        
        field = researcher_profile.get('research_field', 'default')
        field_avg = field_averages.get(field, field_averages['default'])
        
        field_comparison['field_averages'] = field_avg
        
        # Calculate relative performance
        for metric in ['h_index', 'avg_citations_per_paper', 'total_publications']:
            researcher_value = basic_metrics.get(metric, 0)
            field_value = field_avg.get(metric, 1)
            
            relative_performance = researcher_value / field_value if field_value > 0 else 0
            field_comparison['relative_performance'][metric] = relative_performance
            
            # Estimate percentile (simplified)
            if relative_performance >= 1.5:
                percentile = 90
            elif relative_performance >= 1.2:
                percentile = 75
            elif relative_performance >= 1.0:
                percentile = 60
            elif relative_performance >= 0.8:
                percentile = 40
            else:
                percentile = 25
            
            field_comparison['percentile_rankings'][metric] = percentile
        
        return field_comparison
    
    async def predict_future_impact(self, temporal_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Predict future impact based on current trends"""
        prediction = {
            'predicted_h_index_growth': 0.0,
            'predicted_citation_growth': 0.0,
            'career_trajectory': 'stable',
            'recommendations_for_growth': []
        }
        
        # Simple trend analysis
        impact_trajectory = temporal_trends.get('impact_trajectory', {})
        
        if impact_trajectory.get('trajectory') == 'increasing':
            prediction['predicted_h_index_growth'] = 0.15  # 15% annual growth
            prediction['predicted_citation_growth'] = 0.20  # 20% annual growth
            prediction['career_trajectory'] = 'ascending'
        else:
            prediction['predicted_h_index_growth'] = 0.05  # 5% annual growth
            prediction['predicted_citation_growth'] = 0.10  # 10% annual growth
            prediction['career_trajectory'] = 'stable'
        
        # Generate growth recommendations
        if prediction['career_trajectory'] == 'stable':
            prediction['recommendations_for_growth'] = [
                "Increase collaboration with high-impact researchers",
                "Target higher-impact journals",
                "Develop interdisciplinary research projects"
            ]
        
        return prediction
    
    async def generate_impact_recommendations(self, impact_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving scientific impact"""
        recommendations = []
        
        basic_metrics = impact_analysis['basic_metrics']
        field_comparison = impact_analysis['field_comparison']
        
        # H-index recommendations
        h_index_percentile = field_comparison['percentile_rankings'].get('h_index', 50)
        if h_index_percentile < 50:
            recommendations.append("Focus on publishing in higher-impact journals to improve h-index")
        
        # Citation recommendations
        avg_citations = basic_metrics.get('avg_citations_per_paper', 0)
        if avg_citations < 10:
            recommendations.append("Improve paper visibility through conference presentations and social media")
        
        # Collaboration recommendations
        collaboration_index = impact_analysis['advanced_metrics'].get('collaboration_index', 0)
        if collaboration_index < 2:
            recommendations.append("Increase research collaborations to expand network and impact")
        
        # Interdisciplinary recommendations
        interdisciplinary_index = impact_analysis['advanced_metrics'].get('interdisciplinary_index', 0)
        if interdisciplinary_index < 0.3:
            recommendations.append("Explore interdisciplinary research opportunities")
        
        # Recent activity recommendations
        recent_pubs = basic_metrics.get('recent_publications', 0)
        if recent_pubs < 3:
            recommendations.append("Increase publication frequency to maintain research momentum")
        
        return recommendationsclass Au
tomatedScientificDiscoverySystem:
    """Main orchestrator for the automated scientific discovery system"""
    
    def __init__(self):
        self.hypothesis_generator = HypothesisGenerator()
        self.experiment_designer = ExperimentDesigner()
        self.literature_miner = LiteratureMiner()
        self.knowledge_graph_builder = KnowledgeGraphBuilder()
        self.peer_reviewer = AutomatedPeerReview()
        self.writing_assistant = ScientificWritingAssistant()
        self.collaboration_matcher = ResearchCollaborationMatcher()
        self.patent_analyzer = PatentAnalyzer()
        self.impact_analyzer = ScientificImpactAnalyzer()
        
        self.discovery_pipeline = {}
        self.active_discoveries = {}
        
    async def initialize_system(self):
        """Initialize the complete scientific discovery system"""
        try:
            logger.info("Initializing Automated Scientific Discovery System...")
            
            # Initialize all subsystems
            await asyncio.gather(
                self.hypothesis_generator.initialize_models(),
                self.experiment_designer.initialize_design_framework(),
                # Other subsystems are already initialized in their constructors
            )
            
            logger.info("Automated Scientific Discovery System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing discovery system: {e}")
            raise
    
    async def run_discovery_pipeline(self, 
                                   data_patterns: List[Dict[str, Any]], 
                                   domain: ResearchDomain,
                                   researcher_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the complete scientific discovery pipeline"""
        try:
            pipeline_id = f"discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            discovery_results = {
                'pipeline_id': pipeline_id,
                'domain': domain.value,
                'start_time': datetime.now(),
                'hypotheses': [],
                'experiments': [],
                'literature_analysis': None,
                'knowledge_graph': None,
                'peer_reviews': [],
                'paper_drafts': [],
                'collaborations': [],
                'patent_analysis': None,
                'impact_predictions': None,
                'recommendations': [],
                'status': 'running'
            }
            
            self.active_discoveries[pipeline_id] = discovery_results
            
            # Step 1: Generate hypotheses from data patterns
            logger.info(f"Generating hypotheses for {domain.value}...")
            hypotheses = await self.hypothesis_generator.generate_hypotheses_from_patterns(
                data_patterns, domain
            )
            discovery_results['hypotheses'] = hypotheses
            
            # Step 2: Design experiments for each hypothesis
            logger.info("Designing experiments...")
            experiments = []
            for hypothesis in hypotheses:
                experiment = await self.experiment_designer.design_experiment(hypothesis)
                experiments.append(experiment)
            discovery_results['experiments'] = experiments
            
            # Step 3: Mine relevant literature
            logger.info("Mining scientific literature...")
            if hypotheses:
                query = f"{domain.value} {' '.join(hypotheses[0].keywords[:3])}"
                literature_analysis = await self.literature_miner.mine_literature(query)
                discovery_results['literature_analysis'] = literature_analysis
            
            # Step 4: Build knowledge graph
            logger.info("Building knowledge graph...")
            if discovery_results['literature_analysis']:
                papers = discovery_results['literature_analysis'].__dict__.get('papers', [])
                knowledge_graph = await self.knowledge_graph_builder.build_knowledge_graph(
                    papers, hypotheses
                )
                discovery_results['knowledge_graph'] = knowledge_graph
            
            # Step 5: Conduct peer review
            logger.info("Conducting automated peer review...")
            peer_reviews = []
            for hypothesis in hypotheses:
                review = await self.peer_reviewer.review_hypothesis(hypothesis)
                peer_reviews.append(review)
            discovery_results['peer_reviews'] = peer_reviews
            
            # Step 6: Generate paper drafts for validated hypotheses
            logger.info("Generating paper drafts...")
            paper_drafts = []
            for i, hypothesis in enumerate(hypotheses):
                review = peer_reviews[i]
                if review['validation_status'] in [ValidationStatus.VALIDATED, ValidationStatus.REQUIRES_REVISION]:
                    experiment = experiments[i]
                    paper_draft = await self.writing_assistant.generate_paper_draft(
                        hypothesis, experiment, discovery_results['literature_analysis']
                    )
                    paper_drafts.append({
                        'hypothesis_id': hypothesis.id,
                        'sections': paper_draft
                    })
            discovery_results['paper_drafts'] = paper_drafts
            
            # Step 7: Find research collaborations (if researcher profile provided)
            if researcher_profile:
                logger.info("Finding research collaborations...")
                # This would typically involve a database of researchers
                candidate_pool = []  # Would be populated from researcher database
                collaborations = await self.collaboration_matcher.find_collaborations(
                    researcher_profile, candidate_pool
                )
                discovery_results['collaborations'] = collaborations
            
            # Step 8: Analyze patent landscape
            logger.info("Analyzing patent landscape...")
            if hypotheses:
                patent_analysis = await self.patent_analyzer.analyze_patent_landscape(
                    hypotheses[0], hypotheses[0].description
                )
                discovery_results['patent_analysis'] = patent_analysis
            
            # Step 9: Predict scientific impact
            logger.info("Predicting scientific impact...")
            if researcher_profile and discovery_results['literature_analysis']:
                publications = []  # Would extract from literature analysis
                impact_predictions = await self.impact_analyzer.analyze_scientific_impact(
                    researcher_profile, publications
                )
                discovery_results['impact_predictions'] = impact_predictions
            
            # Step 10: Generate final recommendations
            discovery_results['recommendations'] = await self.generate_discovery_recommendations(
                discovery_results
            )
            
            discovery_results['status'] = 'completed'
            discovery_results['end_time'] = datetime.now()
            
            logger.info(f"Discovery pipeline {pipeline_id} completed successfully")
            
            return discovery_results
            
        except Exception as e:
            logger.error(f"Error in discovery pipeline: {e}")
            if pipeline_id in self.active_discoveries:
                self.active_discoveries[pipeline_id]['status'] = 'failed'
                self.active_discoveries[pipeline_id]['error'] = str(e)
            raise
    
    async def generate_discovery_recommendations(self, discovery_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on discovery results"""
        recommendations = []
        
        # Hypothesis recommendations
        hypotheses = discovery_results.get('hypotheses', [])
        if hypotheses:
            validated_count = sum(1 for h in hypotheses if h.validation_status == ValidationStatus.VALIDATED)
            recommendations.append(f"Generated {len(hypotheses)} hypotheses, {validated_count} validated")
            
            if validated_count > 0:
                recommendations.append("Proceed with experimental validation of top hypotheses")
            else:
                recommendations.append("Refine hypotheses based on peer review feedback")
        
        # Experiment recommendations
        experiments = discovery_results.get('experiments', [])
        if experiments:
            high_feasibility = sum(1 for e in experiments if e.feasibility_score > 0.7)
            recommendations.append(f"{high_feasibility}/{len(experiments)} experiments have high feasibility")
            
            if high_feasibility > 0:
                recommendations.append("Start with highest feasibility experiments")
        
        # Literature recommendations
        literature_analysis = discovery_results.get('literature_analysis')
        if literature_analysis and literature_analysis.research_gaps:
            recommendations.append(f"Identified {len(literature_analysis.research_gaps)} research gaps to explore")
        
        # Collaboration recommendations
        collaborations = discovery_results.get('collaborations', [])
        if collaborations:
            high_compatibility = sum(1 for c in collaborations if c.compatibility_score > 0.8)
            recommendations.append(f"Found {high_compatibility} high-compatibility collaboration opportunities")
        
        # Patent recommendations
        patent_analysis = discovery_results.get('patent_analysis')
        if patent_analysis:
            patentability_score = patent_analysis.get('patentability_score', 0)
            if patentability_score > 0.7:
                recommendations.append("High patent potential - consider IP protection strategy")
            elif patentability_score > 0.5:
                recommendations.append("Moderate patent potential - strengthen novel aspects")
        
        # Impact recommendations
        impact_predictions = discovery_results.get('impact_predictions')
        if impact_predictions:
            trajectory = impact_predictions.get('career_trajectory', 'stable')
            if trajectory == 'ascending':
                recommendations.append("Strong impact trajectory - maintain current research direction")
            else:
                recommendations.append("Consider strategies to increase research impact")
        
        return recommendations
    
    async def get_discovery_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get status of a discovery pipeline"""
        if pipeline_id not in self.active_discoveries:
            return {'error': 'Pipeline not found'}
        
        return self.active_discoveries[pipeline_id]
    
    async def list_active_discoveries(self) -> List[Dict[str, Any]]:
        """List all active discovery pipelines"""
        return [
            {
                'pipeline_id': pid,
                'domain': results.get('domain'),
                'status': results.get('status'),
                'start_time': results.get('start_time'),
                'hypothesis_count': len(results.get('hypotheses', []))
            }
            for pid, results in self.active_discoveries.items()
        ]
    
    async def export_discovery_results(self, pipeline_id: str, format: str = 'json') -> str:
        """Export discovery results in specified format"""
        if pipeline_id not in self.active_discoveries:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        results = self.active_discoveries[pipeline_id]
        
        if format == 'json':
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self._make_serializable(results)
            return json.dumps(serializable_results, indent=2)
        elif format == 'markdown':
            return await self._generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj
    
    async def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate markdown report of discovery results"""
        report_lines = [
            f"# Scientific Discovery Report",
            f"",
            f"**Pipeline ID:** {results.get('pipeline_id')}",
            f"**Domain:** {results.get('domain')}",
            f"**Status:** {results.get('status')}",
            f"**Start Time:** {results.get('start_time')}",
            f"",
            f"## Hypotheses Generated",
            f""
        ]
        
        hypotheses = results.get('hypotheses', [])
        for i, hypothesis in enumerate(hypotheses, 1):
            report_lines.extend([
                f"### Hypothesis {i}: {hypothesis.title}",
                f"",
                f"**Description:** {hypothesis.description}",
                f"**Confidence Score:** {hypothesis.confidence_score:.2f}",
                f"**Novelty Score:** {hypothesis.novelty_score:.2f}",
                f"**Testability Score:** {hypothesis.testability_score:.2f}",
                f""
            ])
        
        # Add other sections...
        recommendations = results.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                f"## Recommendations",
                f""
            ])
            for rec in recommendations:
                report_lines.append(f"- {rec}")
        
        return "\n".join(report_lines)


# Example usage and testing functions
async def example_usage():
    """Example usage of the Automated Scientific Discovery System"""
    
    # Initialize the system
    discovery_system = AutomatedScientificDiscoverySystem()
    await discovery_system.initialize_system()
    
    # Example data patterns (would come from actual data analysis)
    data_patterns = [
        {
            'type': 'linear_correlation',
            'significance': 0.85,
            'variables': ['seismic_velocity', 'magma_chamber_pressure'],
            'correlation': 0.78,
            'temporal': {'scale': 'days'},
            'spatial': {'scale': 'regional'}
        },
        {
            'type': 'periodic',
            'significance': 0.72,
            'variables': ['ground_deformation', 'gas_emissions'],
            'correlation': 0.65,
            'temporal': {'scale': 'hours'},
            'spatial': {'scale': 'local'}
        }
    ]
    
    # Example researcher profile
    researcher_profile = {
        'id': 'researcher_001',
        'name': 'Dr. Jane Smith',
        'research_interests': ['volcanic monitoring', 'seismic analysis', 'eruption forecasting'],
        'skills': ['machine learning', 'geophysics', 'data analysis', 'Python programming'],
        'location': 'University of Washington, Seattle, USA',
        'career_stage': 'assistant_professor',
        'publication_count': 25,
        'research_field': 'volcanology'
    }
    
    # Run the discovery pipeline
    results = await discovery_system.run_discovery_pipeline(
        data_patterns=data_patterns,
        domain=ResearchDomain.VOLCANOLOGY,
        researcher_profile=researcher_profile
    )
    
    # Export results
    json_report = await discovery_system.export_discovery_results(
        results['pipeline_id'], 
        format='json'
    )
    
    markdown_report = await discovery_system.export_discovery_results(
        results['pipeline_id'], 
        format='markdown'
    )
    
    print("Discovery pipeline completed successfully!")
    print(f"Generated {len(results['hypotheses'])} hypotheses")
    print(f"Designed {len(results['experiments'])} experiments")
    print(f"Found {len(results.get('collaborations', []))} collaboration opportunities")
    
    return results


if __name__ == "__main__":
    # Run example usage
    asyncio.run(example_usage())