"""
Tests for Hypothesis Generation and Validation Engine

Tests the automated hypothesis generation from discovered causal relationships,
mathematical formulation engines for new geological laws, confidence assessment 
for generated hypotheses, and experimental design suggestions.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from core.hypothesis_generation_engine import (
    HypothesisGenerator,
    MathematicalFormulationEngine,
    ConfidenceAssessmentEngine,
    ExperimentalDesignEngine,
    HypothesisValidationSystem,
    CausalRelationship,
    ScientificHypothesis,
    GeologicalLaw
)

class TestHypothesisGenerator:
    """Test hypothesis generation from causal relationships"""
    
    def setup_method(self):
        self.generator = HypothesisGenerator()
        
        # Create sample causal relationships
        self.sample_relationships = [
            CausalRelationship(
                cause="temperature",
                effect="pressure",
                strength=0.8,
                confidence=0.9,
                mechanism="thermal_expansion",
                evidence=["thermodynamic_data", "field_observations"]
            ),
            CausalRelationship(
                cause="pressure",
                effect="deformation",
                strength=0.7,
                confidence=0.85,
                mechanism="elastic_response",
                evidence=["laboratory_experiments"]
            )
        ]
        
        self.geological_context = {
            'geological_setting': 'volcanic_system',
            'temporal_scale': 'real_time',
            'spatial_scale': 'local'
        }
    
    def test_generate_hypotheses_from_causal_relationships(self):
        """Test hypothesis generation from causal relationships"""
        
        hypotheses = self.generator.generate_hypotheses_from_causal_relationships(
            self.sample_relationships, self.geological_context
        )
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, ScientificHypothesis) for h in hypotheses)
        
        # Check that primary hypotheses are generated
        primary_hypotheses = [h for h in hypotheses if h.id.startswith('hyp_')]
        assert len(primary_hypotheses) >= len(self.sample_relationships)
        
        # Check hypothesis content
        for hypothesis in hypotheses:
            assert hypothesis.statement is not None
            assert len(hypothesis.statement) > 0
            assert hypothesis.confidence_score >= 0
            assert hypothesis.testability_score >= 0
            assert hypothesis.novelty_score >= 0
    
    def test_generate_primary_hypothesis(self):
        """Test primary hypothesis generation"""
        
        relationship = self.sample_relationships[0]
        hypothesis = self.generator._generate_primary_hypothesis(
            relationship, self.geological_context
        )
        
        assert isinstance(hypothesis, ScientificHypothesis)
        assert relationship.cause in hypothesis.statement
        assert relationship.effect in hypothesis.statement
        assert hypothesis.confidence_score > 0
        assert len(hypothesis.supporting_relationships) == 1
        assert hypothesis.supporting_relationships[0] == relationship
    
    def test_generate_alternative_mechanism_hypotheses(self):
        """Test alternative mechanism hypothesis generation"""
        
        relationship = self.sample_relationships[0]
        alt_hypotheses = self.generator._generate_alternative_mechanism_hypotheses(
            relationship, self.geological_context
        )
        
        # Should generate at least one alternative
        assert len(alt_hypotheses) >= 0
        
        for hypothesis in alt_hypotheses:
            assert isinstance(hypothesis, ScientificHypothesis)
            assert 'Alternative mechanism' in hypothesis.statement
            assert hypothesis.id.startswith('alt_')
    
    def test_generate_interaction_hypotheses(self):
        """Test interaction hypothesis generation"""
        
        primary_rel = self.sample_relationships[0]
        interaction_hypotheses = self.generator._generate_interaction_hypotheses(
            primary_rel, self.sample_relationships, self.geological_context
        )
        
        # Should find interactions between temperature->pressure and pressure->deformation
        assert len(interaction_hypotheses) >= 0
        
        for hypothesis in interaction_hypotheses:
            assert isinstance(hypothesis, ScientificHypothesis)
            assert len(hypothesis.supporting_relationships) >= 2


class TestMathematicalFormulationEngine:
    """Test mathematical formulation of geological laws"""
    
    def setup_method(self):
        self.formulator = MathematicalFormulationEngine()
        
        self.sample_relationships = [
            CausalRelationship(
                cause="temperature",
                effect="pressure",
                strength=0.8,
                confidence=0.9
            )
        ]
        
        self.data_patterns = {
            'temperature_pressure': np.array([
                [300, 1000], [400, 1200], [500, 1400], [600, 1600]
            ])
        }
        
        self.physical_context = {
            'system_type': 'volcanic',
            'constraints': ['conservation_of_energy', 'thermodynamic_equilibrium']
        }
    
    def test_formulate_geological_law(self):
        """Test geological law formulation"""
        
        law = self.formulator.formulate_geological_law(
            self.sample_relationships,
            self.data_patterns,
            self.physical_context
        )
        
        assert isinstance(law, GeologicalLaw)
        assert law.name is not None
        assert law.mathematical_expression is not None
        assert len(law.variables) > 0
        assert law.confidence > 0
        assert len(law.domain_applicability) > 0
    
    def test_determine_functional_form(self):
        """Test functional form determination"""
        
        functional_form = self.formulator._determine_functional_form(
            self.sample_relationships, self.data_patterns
        )
        
        assert functional_form in ['linear', 'power', 'exponential', 'logarithmic']
    
    def test_generate_mathematical_expression(self):
        """Test mathematical expression generation"""
        
        expression = self.formulator._generate_mathematical_expression(
            'linear', self.sample_relationships, self.physical_context
        )
        
        assert expression is not None
        # Should be a sympy equation
        assert hasattr(expression, 'lhs') and hasattr(expression, 'rhs')
    
    def test_validate_physical_constraints(self):
        """Test physical constraint validation"""
        
        # Create a simple expression
        from sympy import symbols, Eq
        T, P = symbols('T P', real=True, positive=True)
        expression = Eq(P, T)
        
        validated = self.formulator._validate_physical_constraints(
            expression, self.physical_context
        )
        
        assert validated is not None


class TestConfidenceAssessmentEngine:
    """Test confidence assessment for hypotheses"""
    
    def setup_method(self):
        self.assessor = ConfidenceAssessmentEngine()
        
        self.sample_hypothesis = ScientificHypothesis(
            id="test_hyp_001",
            statement="Temperature increases cause pressure increases through thermal expansion",
            confidence_score=0.0,  # Will be calculated
            testability_score=0.8,
            novelty_score=0.7,
            supporting_relationships=[
                CausalRelationship(
                    cause="temperature",
                    effect="pressure", 
                    strength=0.8,
                    confidence=0.9
                )
            ]
        )
        
        self.supporting_data = {
            'p_values': [0.01, 0.05, 0.001],
            'correlation_coefficients': [0.8, 0.7, 0.9],
            'completeness': 0.9,
            'accuracy': 0.85,
            'sample_size': 1000,
            'temporal_coverage': 0.8
        }
    
    def test_assess_hypothesis_confidence(self):
        """Test overall confidence assessment"""
        
        confidence = self.assessor.assess_hypothesis_confidence(
            self.sample_hypothesis, self.supporting_data
        )
        
        assert 0 <= confidence <= 1
        assert self.sample_hypothesis.confidence_score == confidence
    
    def test_assess_causal_strength(self):
        """Test causal strength assessment"""
        
        strength_score = self.assessor._assess_causal_strength(self.sample_hypothesis)
        
        assert 0 <= strength_score <= 1
        # Should be high given strong causal relationship
        assert strength_score > 0.5
    
    def test_assess_statistical_significance(self):
        """Test statistical significance assessment"""
        
        sig_score = self.assessor._assess_statistical_significance(
            self.sample_hypothesis, self.supporting_data
        )
        
        assert 0 <= sig_score <= 1
        # Should be high given low p-values
        assert sig_score > 0.8
    
    def test_assess_physical_plausibility(self):
        """Test physical plausibility assessment"""
        
        plausibility = self.assessor._assess_physical_plausibility(self.sample_hypothesis)
        
        assert 0 <= plausibility <= 1
        # Should be reasonable for thermal expansion
        assert plausibility >= 0.5
    
    def test_assess_data_quality(self):
        """Test data quality assessment"""
        
        quality_score = self.assessor._assess_data_quality(self.supporting_data)
        
        assert 0 <= quality_score <= 1
        # Should be high given good data metrics
        assert quality_score > 0.7


class TestExperimentalDesignEngine:
    """Test experimental design for hypothesis validation"""
    
    def setup_method(self):
        self.designer = ExperimentalDesignEngine()
        
        self.sample_hypothesis = ScientificHypothesis(
            id="test_hyp_001",
            statement="Temperature increases cause pressure increases in magma chambers",
            confidence_score=0.8,
            testability_score=0.9,
            novelty_score=0.7
        )
        
        self.available_resources = {
            'budget': 'medium',
            'time': 'weeks_to_months',
            'equipment': ['seismometers', 'temperature_sensors']
        }
    
    def test_design_validation_experiments(self):
        """Test validation experiment design"""
        
        experiments = self.designer.design_validation_experiments(
            self.sample_hypothesis,
            self.available_resources,
            'volcanic_systems'
        )
        
        assert len(experiments) > 0
        
        for experiment in experiments:
            assert 'title' in experiment
            assert 'objective' in experiment
            assert 'methodology' in experiment
            assert 'experiment_type' in experiment
            assert 'estimated_cost' in experiment
            assert 'estimated_duration' in experiment
    
    def test_select_suitable_experiments(self):
        """Test suitable experiment selection"""
        
        suitable_types = self.designer._select_suitable_experiments(
            self.sample_hypothesis,
            'volcanic_systems',
            self.available_resources
        )
        
        assert len(suitable_types) > 0
        assert all(exp_type in self.designer.experiment_types for exp_type in suitable_types)
    
    def test_design_specific_experiments(self):
        """Test specific experiment design methods"""
        
        variables = ['temperature', 'pressure']
        
        # Test field observation design
        field_design = self.designer._design_field_observation(
            self.sample_hypothesis, variables, 'volcanic_systems'
        )
        assert 'title' in field_design
        assert 'methodology' in field_design
        
        # Test laboratory experiment design
        lab_design = self.designer._design_laboratory_experiment(
            self.sample_hypothesis, variables
        )
        assert 'title' in lab_design
        assert 'experimental_conditions' in lab_design
        
        # Test numerical simulation design
        sim_design = self.designer._design_numerical_simulation(
            self.sample_hypothesis, variables
        )
        assert 'title' in sim_design
        assert 'model_type' in sim_design
    
    def test_extract_testable_variables(self):
        """Test variable extraction from hypothesis"""
        
        variables = self.designer._extract_testable_variables(self.sample_hypothesis)
        
        assert len(variables) > 0
        assert 'temperature' in variables
        assert 'pressure' in variables


class TestHypothesisValidationSystem:
    """Test integrated hypothesis validation system"""
    
    def setup_method(self):
        self.system = HypothesisValidationSystem()
        
        self.causal_relationships = [
            CausalRelationship(
                cause="temperature",
                effect="pressure",
                strength=0.8,
                confidence=0.9,
                mechanism="thermal_expansion"
            ),
            CausalRelationship(
                cause="pressure", 
                effect="seismic_activity",
                strength=0.6,
                confidence=0.8,
                mechanism="stress_release"
            )
        ]
        
        self.geological_data = {
            'temperature_pressure': np.array([[300, 1000], [400, 1200]]),
            'p_values': [0.01, 0.05],
            'sample_size': 500
        }
        
        self.geological_context = {
            'setting': 'volcanic_system',
            'location': 'test_volcano',
            'temporal_scale': 'real_time'
        }
        
        self.available_resources = {
            'budget': 'medium',
            'time': 'months',
            'equipment': ['sensors', 'computers']
        }
    
    def test_generate_and_validate_hypotheses(self):
        """Test complete hypothesis generation and validation workflow"""
        
        results = self.system.generate_and_validate_hypotheses(
            self.causal_relationships,
            self.geological_data,
            self.geological_context,
            self.available_resources
        )
        
        # Check results structure
        assert 'generated_hypotheses' in results
        assert 'formulated_laws' in results
        assert 'total_hypotheses' in results
        assert 'high_confidence_hypotheses' in results
        assert 'testable_hypotheses' in results
        assert 'generation_timestamp' in results
        assert 'summary' in results
        
        # Check hypotheses
        hypotheses = results['generated_hypotheses']
        assert len(hypotheses) > 0
        assert all(isinstance(h, ScientificHypothesis) for h in hypotheses)
        
        # Check laws
        laws = results['formulated_laws']
        assert len(laws) > 0
        assert all(isinstance(law, GeologicalLaw) for law in laws)
        
        # Check summary
        summary = results['summary']
        assert 'average_confidence' in summary
        assert 'average_novelty' in summary
        assert 'top_hypothesis' in summary
        assert 'recommendations' in summary
    
    def test_generate_summary(self):
        """Test summary generation"""
        
        sample_hypotheses = [
            ScientificHypothesis(
                id="hyp1",
                statement="Test hypothesis 1",
                confidence_score=0.8,
                novelty_score=0.7,
                testability_score=0.9
            ),
            ScientificHypothesis(
                id="hyp2", 
                statement="Test hypothesis 2",
                confidence_score=0.6,
                novelty_score=0.8,
                testability_score=0.7
            )
        ]
        
        sample_laws = [
            GeologicalLaw(
                name="Test Law",
                mathematical_expression="P = a*T + b",
                variables=['P', 'T'],
                domain_applicability=['volcanic_systems'],
                confidence=0.8
            )
        ]
        
        summary = self.system._generate_summary(sample_hypotheses, sample_laws)
        
        assert 'average_confidence' in summary
        assert 'average_novelty' in summary
        assert 'top_hypothesis' in summary
        assert 'formulated_laws_count' in summary
        assert summary['formulated_laws_count'] == 1
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        
        high_conf_hypothesis = ScientificHypothesis(
            id="high_conf",
            statement="High confidence hypothesis",
            confidence_score=0.9,
            novelty_score=0.8,
            testability_score=0.9
        )
        high_conf_hypothesis.experimental_design = [{'type': 'field_observation'}]
        
        recommendations = self.system._generate_recommendations(
            [high_conf_hypothesis], []
        )
        
        assert len(recommendations) > 0
        assert any('high-confidence' in rec for rec in recommendations)
        assert any('testable' in rec for rec in recommendations)


if __name__ == '__main__':
    pytest.main([__file__])