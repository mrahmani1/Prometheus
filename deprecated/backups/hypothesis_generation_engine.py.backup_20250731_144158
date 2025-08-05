"""
Hypothesis Generation and Validation Engine for Autonomous Scientific Discovery

This module implements automated hypothesis generation from discovered causal relationships,
mathematical formulation engines for new geological laws, confidence assessment for 
generated hypotheses, and experimental design suggestions for hypothesis validation.

Requirements: 15.3, 15.7, 15.8
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from scipy import stats
from sklearn.metrics import r2_score
import sympy as sp
from sympy import symbols, Function, Eq, diff, integrate
import networkx as nx
from itertools import combinations, permutations

logger = logging.getLogger(__name__)

@dataclass
class CausalRelationship:
    """Represents a discovered causal relationship between geological variables"""
    cause: str
    effect: str
    strength: float  # Causal strength coefficient
    confidence: float  # Statistical confidence
    mechanism: Optional[str] = None  # Proposed physical mechanism
    evidence: List[str] = field(default_factory=list)  # Supporting evidence
    
@dataclass
class ScientificHypothesis:
    """Represents a generated scientific hypothesis"""
    id: str
    statement: str
    mathematical_formulation: Optional[str] = None
    confidence_score: float = 0.0
    testability_score: float = 0.0
    novelty_score: float = 0.0
    supporting_relationships: List[CausalRelationship] = field(default_factory=list)
    predicted_outcomes: List[str] = field(default_factory=list)
    experimental_design: Optional[Dict[str, Any]] = None
    generated_timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class GeologicalLaw:
    """Represents a discovered or formulated geological law"""
    name: str
    mathematical_expression: str
    variables: List[str]
    domain_applicability: List[str]  # Geological contexts where law applies
    confidence: float
    validation_experiments: List[Dict[str, Any]] = field(default_factory=list)
    supporting_data: List[str] = field(default_factory=list)

class HypothesisGenerator:
    """
    Generates scientific hypotheses from discovered causal relationships
    Implements requirement 15.3: suggest new scientific hypotheses for human validation
    """
    
    def __init__(self):
        self.geological_knowledge_base = self._initialize_knowledge_base()
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.confidence_threshold = 0.7
        
    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize geological knowledge base for hypothesis generation"""
        return {
            'geological_processes': [
                'magma_intrusion', 'crustal_deformation', 'hydrothermal_circulation',
                'seismic_activity', 'volcanic_eruption', 'plate_tectonics',
                'mantle_convection', 'metamorphism', 'weathering', 'sedimentation'
            ],
            'physical_laws': [
                'conservation_of_mass', 'conservation_of_energy', 'conservation_of_momentum',
                'heat_transfer', 'fluid_dynamics', 'elastic_deformation', 'stress_strain'
            ],
            'geological_variables': [
                'temperature', 'pressure', 'density', 'viscosity', 'composition',
                'stress', 'strain', 'permeability', 'porosity', 'velocity'
            ],
            'temporal_scales': ['seconds', 'minutes', 'hours', 'days', 'years', 'millennia', 'geological_time'],
            'spatial_scales': ['microscopic', 'local', 'regional', 'continental', 'global']
        }
    
    def _load_hypothesis_templates(self) -> List[str]:
        """Load templates for hypothesis generation"""
        return [
            "If {cause} increases, then {effect} will {direction} due to {mechanism}",
            "When {condition} occurs, {process} will be enhanced/inhibited by {factor}",
            "The relationship between {var1} and {var2} is mediated by {mediator}",
            "Under conditions of {context}, {phenomenon} exhibits {behavior}",
            "The coupling between {system1} and {system2} results in {outcome}"
        ]
    
    def generate_hypotheses_from_causal_relationships(
        self, 
        causal_relationships: List[CausalRelationship],
        geological_context: Dict[str, Any]
    ) -> List[ScientificHypothesis]:
        """
        Generate testable hypotheses from discovered causal relationships
        
        Args:
            causal_relationships: List of discovered causal relationships
            geological_context: Context information about the geological setting
            
        Returns:
            List of generated scientific hypotheses
        """
        hypotheses = []
        
        for relationship in causal_relationships:
            if relationship.confidence >= self.confidence_threshold:
                # Generate primary hypothesis
                primary_hypothesis = self._generate_primary_hypothesis(relationship, geological_context)
                hypotheses.append(primary_hypothesis)
                
                # Generate secondary hypotheses (alternative mechanisms)
                secondary_hypotheses = self._generate_alternative_mechanism_hypotheses(
                    relationship, geological_context
                )
                hypotheses.extend(secondary_hypotheses)
                
                # Generate interaction hypotheses
                interaction_hypotheses = self._generate_interaction_hypotheses(
                    relationship, causal_relationships, geological_context
                )
                hypotheses.extend(interaction_hypotheses)
        
        # Remove duplicates and rank by novelty and testability
        unique_hypotheses = self._remove_duplicate_hypotheses(hypotheses)
        ranked_hypotheses = self._rank_hypotheses(unique_hypotheses)
        
        logger.info(f"Generated {len(ranked_hypotheses)} unique hypotheses from {len(causal_relationships)} causal relationships")
        
        return ranked_hypotheses
    
    def _generate_primary_hypothesis(
        self, 
        relationship: CausalRelationship, 
        context: Dict[str, Any]
    ) -> ScientificHypothesis:
        """Generate primary hypothesis from causal relationship"""
        
        # Determine causal direction and mechanism
        direction = "increase" if relationship.strength > 0 else "decrease"
        mechanism = relationship.mechanism or self._infer_mechanism(relationship, context)
        
        # Generate hypothesis statement
        statement = f"In {context.get('geological_setting', 'geological systems')}, " \
                   f"changes in {relationship.cause} causally influence {relationship.effect} " \
                   f"through {mechanism}, with a {direction} relationship of strength {relationship.strength:.3f}"
        
        # Calculate confidence and testability scores
        confidence_score = self._calculate_hypothesis_confidence(relationship, context)
        testability_score = self._calculate_testability_score(relationship, context)
        novelty_score = self._calculate_novelty_score(relationship, context)
        
        # Generate predicted outcomes
        predicted_outcomes = self._generate_predicted_outcomes(relationship, context)
        
        hypothesis = ScientificHypothesis(
            id=f"hyp_{relationship.cause}_{relationship.effect}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            statement=statement,
            confidence_score=confidence_score,
            testability_score=testability_score,
            novelty_score=novelty_score,
            supporting_relationships=[relationship],
            predicted_outcomes=predicted_outcomes
        )
        
        return hypothesis
    
    def _generate_alternative_mechanism_hypotheses(
        self, 
        relationship: CausalRelationship, 
        context: Dict[str, Any]
    ) -> List[ScientificHypothesis]:
        """Generate hypotheses with alternative causal mechanisms"""
        
        alternative_mechanisms = self._identify_alternative_mechanisms(relationship, context)
        hypotheses = []
        
        for mechanism in alternative_mechanisms:
            statement = f"Alternative mechanism: {relationship.cause} influences {relationship.effect} " \
                       f"through {mechanism} rather than direct causation"
            
            hypothesis = ScientificHypothesis(
                id=f"alt_{relationship.cause}_{relationship.effect}_{mechanism}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                statement=statement,
                confidence_score=relationship.confidence * 0.7,  # Lower confidence for alternatives
                testability_score=self._calculate_testability_score(relationship, context),
                novelty_score=self._calculate_novelty_score(relationship, context) * 1.2,  # Higher novelty
                supporting_relationships=[relationship],
                predicted_outcomes=self._generate_predicted_outcomes(relationship, context)
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_interaction_hypotheses(
        self, 
        primary_relationship: CausalRelationship,
        all_relationships: List[CausalRelationship],
        context: Dict[str, Any]
    ) -> List[ScientificHypothesis]:
        """Generate hypotheses about interactions between multiple causal relationships"""
        
        interaction_hypotheses = []
        
        # Find relationships that share variables with the primary relationship
        related_relationships = [
            rel for rel in all_relationships 
            if rel != primary_relationship and (
                rel.cause == primary_relationship.effect or 
                rel.effect == primary_relationship.cause or
                rel.cause == primary_relationship.cause or
                rel.effect == primary_relationship.effect
            )
        ]
        
        for related_rel in related_relationships:
            # Generate interaction hypothesis
            if related_rel.cause == primary_relationship.effect:
                # Chain interaction: A -> B -> C
                statement = f"Causal chain hypothesis: {primary_relationship.cause} influences " \
                           f"{related_rel.effect} through the intermediate variable {primary_relationship.effect}"
                
            elif related_rel.effect == primary_relationship.cause:
                # Reverse chain: C -> B -> A
                statement = f"Feedback loop hypothesis: {related_rel.cause} and {primary_relationship.effect} " \
                           f"form a feedback system through {primary_relationship.cause}"
                
            else:
                # Common cause or effect
                statement = f"Interaction hypothesis: {primary_relationship.cause} and {related_rel.cause} " \
                           f"interact to jointly influence geological processes"
            
            hypothesis = ScientificHypothesis(
                id=f"int_{primary_relationship.cause}_{related_rel.cause}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                statement=statement,
                confidence_score=min(primary_relationship.confidence, related_rel.confidence) * 0.8,
                testability_score=self._calculate_interaction_testability(primary_relationship, related_rel),
                novelty_score=self._calculate_novelty_score(primary_relationship, context) * 1.5,
                supporting_relationships=[primary_relationship, related_rel],
                predicted_outcomes=self._generate_interaction_outcomes(primary_relationship, related_rel)
            )
            interaction_hypotheses.append(hypothesis)
        
        return interaction_hypotheses


class MathematicalFormulationEngine:
    """
    Builds mathematical formulation engines for new geological laws
    Implements requirement 15.8: provide mathematical formulations and confidence assessments
    """
    
    def __init__(self):
        self.symbolic_variables = self._initialize_symbolic_variables()
        self.physical_constants = self._initialize_physical_constants()
        self.equation_templates = self._load_equation_templates()
        
    def _initialize_symbolic_variables(self) -> Dict[str, sp.Symbol]:
        """Initialize symbolic variables for mathematical formulations"""
        return {
            't': symbols('t', real=True, positive=True),  # time
            'T': symbols('T', real=True, positive=True),  # temperature
            'P': symbols('P', real=True, positive=True),  # pressure
            'rho': symbols('rho', real=True, positive=True),  # density
            'mu': symbols('mu', real=True, positive=True),  # viscosity
            'v': symbols('v', real=True),  # velocity
            'sigma': symbols('sigma', real=True),  # stress
            'epsilon': symbols('epsilon', real=True),  # strain
            'k': symbols('k', real=True, positive=True),  # permeability
            'phi': symbols('phi', real=True, positive=True),  # porosity
            'x': symbols('x', real=True),  # spatial coordinate x
            'y': symbols('y', real=True),  # spatial coordinate y
            'z': symbols('z', real=True),  # spatial coordinate z
        }
    
    def _initialize_physical_constants(self) -> Dict[str, float]:
        """Initialize physical constants for geological processes"""
        return {
            'g': 9.81,  # gravitational acceleration
            'R': 8.314,  # gas constant
            'k_B': 1.38e-23,  # Boltzmann constant
            'rho_mantle': 3300,  # mantle density (kg/m³)
            'rho_crust': 2700,  # crustal density (kg/m³)
            'alpha_thermal': 3e-5,  # thermal expansion coefficient
            'kappa_thermal': 1e-6,  # thermal diffusivity
        }
    
    def _load_equation_templates(self) -> Dict[str, str]:
        """Load templates for different types of geological equations"""
        return {
            'conservation': '{quantity}_t + div({flux}) = {source}',
            'diffusion': '{quantity}_t = {diffusivity} * laplacian({quantity}) + {source}',
            'advection': '{quantity}_t + {velocity} * grad({quantity}) = {diffusivity} * laplacian({quantity})',
            'constitutive': '{output} = {function}({inputs})',
            'equilibrium': '{force_sum} = 0',
            'rate_law': 'd{quantity}/dt = {rate_constant} * {concentration}^{order}',
            'power_law': '{dependent} = {coefficient} * {independent}^{exponent}',
            'exponential': '{dependent} = {coefficient} * exp({exponent} * {independent})',
            'logarithmic': '{dependent} = {coefficient} * log({independent})'
        }
    
    def formulate_geological_law(
        self, 
        causal_relationships: List[CausalRelationship],
        data_patterns: Dict[str, np.ndarray],
        physical_context: Dict[str, Any]
    ) -> GeologicalLaw:
        """
        Formulate mathematical expressions for discovered geological laws
        
        Args:
            causal_relationships: Discovered causal relationships
            data_patterns: Observed data patterns
            physical_context: Physical context and constraints
            
        Returns:
            Formulated geological law with mathematical expression
        """
        
        # Analyze the functional form of relationships
        functional_form = self._determine_functional_form(causal_relationships, data_patterns)
        
        # Generate mathematical expression
        mathematical_expression = self._generate_mathematical_expression(
            functional_form, causal_relationships, physical_context
        )
        
        # Validate against physical constraints
        validated_expression = self._validate_physical_constraints(
            mathematical_expression, physical_context
        )
        
        # Calculate confidence in the formulated law
        confidence = self._calculate_law_confidence(
            validated_expression, causal_relationships, data_patterns
        )
        
        # Determine domain of applicability
        domain_applicability = self._determine_domain_applicability(
            causal_relationships, physical_context
        )
        
        # Generate validation experiments
        validation_experiments = self._design_validation_experiments(
            validated_expression, domain_applicability
        )
        
        law = GeologicalLaw(
            name=self._generate_law_name(causal_relationships),
            mathematical_expression=str(validated_expression),
            variables=self._extract_variables(validated_expression),
            domain_applicability=domain_applicability,
            confidence=confidence,
            validation_experiments=validation_experiments,
            supporting_data=[str(rel) for rel in causal_relationships]
        )
        
        logger.info(f"Formulated geological law: {law.name} with confidence {confidence:.3f}")
        
        return law
    
    def _determine_functional_form(
        self, 
        relationships: List[CausalRelationship], 
        data_patterns: Dict[str, np.ndarray]
    ) -> str:
        """Determine the functional form of the relationship from data patterns"""
        
        # Analyze correlation patterns and fit different functional forms
        best_form = 'linear'
        best_r2 = 0.0
        
        for var_pair, data in data_patterns.items():
            # Convert to numpy array if it's a list
            if isinstance(data, list):
                try:
                    data = np.array(data)
                except:
                    continue
            
            # Skip non-numeric data
            if not hasattr(data, 'shape'):
                continue
                
            if len(data.shape) == 2 and data.shape[1] >= 2:
                x, y = data[:, 0], data[:, 1]
                
                # Test different functional forms
                forms_to_test = {
                    'linear': lambda x, a, b: a * x + b,
                    'power': lambda x, a, b: a * np.power(np.abs(x), b),
                    'exponential': lambda x, a, b: a * np.exp(b * x),
                    'logarithmic': lambda x, a, b: a * np.log(np.abs(x) + 1e-10) + b
                }
                
                for form_name, form_func in forms_to_test.items():
                    try:
                        # Simple parameter estimation
                        if form_name == 'linear':
                            coeffs = np.polyfit(x, y, 1)
                            y_pred = np.polyval(coeffs, x)
                        elif form_name == 'power':
                            log_x, log_y = np.log(np.abs(x) + 1e-10), np.log(np.abs(y) + 1e-10)
                            coeffs = np.polyfit(log_x, log_y, 1)
                            y_pred = np.exp(coeffs[1]) * np.power(np.abs(x), coeffs[0])
                        else:
                            continue  # Skip complex fitting for now
                        
                        r2 = r2_score(y, y_pred)
                        if r2 > best_r2:
                            best_r2 = r2
                            best_form = form_name
                            
                    except Exception as e:
                        logger.warning(f"Failed to fit {form_name} form: {e}")
                        continue
        
        return best_form
    
    def _generate_mathematical_expression(
        self, 
        functional_form: str, 
        relationships: List[CausalRelationship],
        context: Dict[str, Any]
    ) -> sp.Expr:
        """Generate symbolic mathematical expression"""
        
        # Get primary relationship
        primary_rel = max(relationships, key=lambda r: r.confidence)
        
        # Map variable names to symbols
        cause_var = self._get_symbol_for_variable(primary_rel.cause)
        effect_var = self._get_symbol_for_variable(primary_rel.effect)
        
        # Generate expression based on functional form
        if functional_form == 'linear':
            # Linear relationship: effect = a * cause + b
            a, b = symbols('a b', real=True)
            expression = Eq(effect_var, a * cause_var + b)
            
        elif functional_form == 'power':
            # Power law: effect = a * cause^n
            a, n = symbols('a n', real=True)
            expression = Eq(effect_var, a * cause_var**n)
            
        elif functional_form == 'exponential':
            # Exponential: effect = a * exp(b * cause)
            a, b = symbols('a b', real=True)
            expression = Eq(effect_var, a * sp.exp(b * cause_var))
            
        elif functional_form == 'logarithmic':
            # Logarithmic: effect = a * log(cause) + b
            a, b = symbols('a b', real=True)
            expression = Eq(effect_var, a * sp.log(cause_var) + b)
            
        else:
            # Default to linear
            a, b = symbols('a b', real=True)
            expression = Eq(effect_var, a * cause_var + b)
        
        # Add time dependence if temporal relationship exists
        if any('time' in rel.cause.lower() or 'time' in rel.effect.lower() for rel in relationships):
            t = self.symbolic_variables['t']
            # Add temporal derivative term
            expression = Eq(diff(effect_var, t), expression.rhs)
        
        return expression
    
    def _get_symbol_for_variable(self, variable_name: str) -> sp.Symbol:
        """Map geological variable names to symbolic variables"""
        
        variable_mapping = {
            'temperature': 'T',
            'pressure': 'P',
            'density': 'rho',
            'viscosity': 'mu',
            'velocity': 'v',
            'stress': 'sigma',
            'strain': 'epsilon',
            'permeability': 'k',
            'porosity': 'phi'
        }
        
        # Check for direct mapping
        for key, symbol_name in variable_mapping.items():
            if key in variable_name.lower():
                return self.symbolic_variables[symbol_name]
        
        # Create new symbol if not found
        return symbols(variable_name.replace(' ', '_'), real=True)
    
    def _validate_physical_constraints(
        self, 
        expression: sp.Expr, 
        context: Dict[str, Any]
    ) -> sp.Expr:
        """Validate mathematical expression against physical constraints"""
        
        # Check dimensional consistency
        if not self._check_dimensional_consistency(expression):
            logger.warning("Expression may have dimensional inconsistencies")
        
        # Check for conservation laws
        if not self._check_conservation_laws(expression, context):
            logger.warning("Expression may violate conservation laws")
        
        # Check for thermodynamic constraints
        if not self._check_thermodynamic_constraints(expression):
            logger.warning("Expression may violate thermodynamic constraints")
        
        return expression
    
    def _check_dimensional_consistency(self, expression: sp.Expr) -> bool:
        """Check if the expression is dimensionally consistent"""
        # Simplified dimensional analysis
        # In a full implementation, this would use a proper dimensional analysis library
        return True  # Placeholder
    
    def _check_conservation_laws(self, expression: sp.Expr, context: Dict[str, Any]) -> bool:
        """Check if expression respects conservation laws"""
        # Check for mass, energy, momentum conservation
        # This is a simplified check
        return True  # Placeholder
    
    def _check_thermodynamic_constraints(self, expression: sp.Expr) -> bool:
        """Check thermodynamic constraints like entropy increase"""
        return True  # Placeholder


class ConfidenceAssessmentEngine:
    """
    Implements confidence assessment for generated hypotheses
    Implements requirement 15.8: provide confidence assessments
    """
    
    def __init__(self):
        self.confidence_factors = self._initialize_confidence_factors()
        
    def _initialize_confidence_factors(self) -> Dict[str, float]:
        """Initialize factors that contribute to hypothesis confidence"""
        return {
            'causal_strength': 0.25,  # Strength of underlying causal relationships
            'statistical_significance': 0.20,  # Statistical significance of patterns
            'physical_plausibility': 0.20,  # Consistency with known physics
            'data_quality': 0.15,  # Quality and quantity of supporting data
            'prior_knowledge': 0.10,  # Consistency with existing knowledge
            'reproducibility': 0.10   # Reproducibility across different datasets
        }
    
    def assess_hypothesis_confidence(
        self, 
        hypothesis: ScientificHypothesis,
        supporting_data: Dict[str, Any],
        validation_results: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Assess confidence in a generated hypothesis
        
        Args:
            hypothesis: The hypothesis to assess
            supporting_data: Data supporting the hypothesis
            validation_results: Results from validation experiments
            
        Returns:
            Confidence score between 0 and 1
        """
        
        confidence_scores = {}
        
        # Assess causal strength
        confidence_scores['causal_strength'] = self._assess_causal_strength(hypothesis)
        
        # Assess statistical significance
        confidence_scores['statistical_significance'] = self._assess_statistical_significance(
            hypothesis, supporting_data
        )
        
        # Assess physical plausibility
        confidence_scores['physical_plausibility'] = self._assess_physical_plausibility(hypothesis)
        
        # Assess data quality
        confidence_scores['data_quality'] = self._assess_data_quality(supporting_data)
        
        # Assess consistency with prior knowledge
        confidence_scores['prior_knowledge'] = self._assess_prior_knowledge_consistency(hypothesis)
        
        # Assess reproducibility
        confidence_scores['reproducibility'] = self._assess_reproducibility(
            hypothesis, validation_results
        )
        
        # Calculate weighted confidence score
        total_confidence = sum(
            score * self.confidence_factors[factor]
            for factor, score in confidence_scores.items()
        )
        
        # Update hypothesis confidence
        hypothesis.confidence_score = total_confidence
        
        logger.info(f"Assessed confidence for hypothesis {hypothesis.id}: {total_confidence:.3f}")
        logger.debug(f"Confidence breakdown: {confidence_scores}")
        
        return total_confidence
    
    def _assess_causal_strength(self, hypothesis: ScientificHypothesis) -> float:
        """Assess the strength of causal relationships supporting the hypothesis"""
        if not hypothesis.supporting_relationships:
            return 0.0
        
        # Average causal strength weighted by confidence
        total_strength = sum(
            rel.strength * rel.confidence 
            for rel in hypothesis.supporting_relationships
        )
        total_weight = sum(rel.confidence for rel in hypothesis.supporting_relationships)
        
        if total_weight == 0:
            return 0.0
        
        return min(abs(total_strength / total_weight), 1.0)
    
    def _assess_statistical_significance(
        self, 
        hypothesis: ScientificHypothesis, 
        supporting_data: Dict[str, Any]
    ) -> float:
        """Assess statistical significance of patterns supporting the hypothesis"""
        
        if 'p_values' in supporting_data:
            p_values = supporting_data['p_values']
            # Convert p-values to confidence scores
            significance_scores = [max(0, 1 - p) for p in p_values if p is not None]
            return np.mean(significance_scores) if significance_scores else 0.5
        
        if 'correlation_coefficients' in supporting_data:
            correlations = supporting_data['correlation_coefficients']
            return np.mean([abs(r) for r in correlations if r is not None])
        
        return 0.5  # Default moderate confidence
    
    def _assess_physical_plausibility(self, hypothesis: ScientificHypothesis) -> float:
        """Assess physical plausibility of the hypothesis"""
        
        # Check for physical impossibilities
        implausible_terms = [
            'negative_temperature', 'infinite_velocity', 'negative_density',
            'faster_than_light', 'perpetual_motion'
        ]
        
        statement_lower = hypothesis.statement.lower()
        for term in implausible_terms:
            if term.replace('_', ' ') in statement_lower:
                return 0.1  # Very low confidence for physically implausible hypotheses
        
        # Check for consistency with known physical laws
        plausible_mechanisms = [
            'heat_transfer', 'fluid_flow', 'elastic_deformation', 'chemical_reaction',
            'phase_transition', 'diffusion', 'convection', 'conduction'
        ]
        
        plausibility_score = 0.5  # Default
        for mechanism in plausible_mechanisms:
            if mechanism.replace('_', ' ') in statement_lower:
                plausibility_score += 0.1
        
        return min(plausibility_score, 1.0)
    
    def _assess_data_quality(self, supporting_data: Dict[str, Any]) -> float:
        """Assess quality of data supporting the hypothesis"""
        
        quality_score = 0.0
        
        # Check data completeness
        if 'completeness' in supporting_data:
            quality_score += 0.3 * supporting_data['completeness']
        
        # Check data accuracy
        if 'accuracy' in supporting_data:
            quality_score += 0.3 * supporting_data['accuracy']
        
        # Check sample size
        if 'sample_size' in supporting_data:
            sample_size = supporting_data['sample_size']
            # Logarithmic scaling for sample size contribution
            size_score = min(np.log10(sample_size) / 4, 1.0) if sample_size > 0 else 0
            quality_score += 0.2 * size_score
        
        # Check temporal coverage
        if 'temporal_coverage' in supporting_data:
            quality_score += 0.2 * supporting_data['temporal_coverage']
        
        return min(quality_score, 1.0)
    
    def _assess_prior_knowledge_consistency(self, hypothesis: ScientificHypothesis) -> float:
        """Assess consistency with existing geological knowledge"""
        
        # This would integrate with a geological knowledge base
        # For now, use simple heuristics
        
        consistent_terms = [
            'magma', 'temperature', 'pressure', 'deformation', 'seismic',
            'volcanic', 'tectonic', 'thermal', 'mechanical', 'chemical'
        ]
        
        statement_lower = hypothesis.statement.lower()
        consistency_score = sum(
            0.1 for term in consistent_terms 
            if term in statement_lower
        )
        
        return min(consistency_score, 1.0)
    
    def _assess_reproducibility(
        self, 
        hypothesis: ScientificHypothesis,
        validation_results: Optional[Dict[str, Any]]
    ) -> float:
        """Assess reproducibility of hypothesis across different conditions"""
        
        if validation_results is None:
            return 0.5  # Default when no validation results available
        
        if 'cross_validation_scores' in validation_results:
            cv_scores = validation_results['cross_validation_scores']
            return np.mean(cv_scores) if cv_scores else 0.5
        
        if 'replication_success_rate' in validation_results:
            return validation_results['replication_success_rate']
        
        return 0.5


class ExperimentalDesignEngine:
    """
    Writes experimental design suggestions for hypothesis validation
    Implements requirement 15.7: suggest validation experiments
    """
    
    def __init__(self):
        self.experiment_types = self._initialize_experiment_types()
        self.geological_contexts = self._initialize_geological_contexts()
        
    def _initialize_experiment_types(self) -> Dict[str, Dict[str, Any]]:
        """Initialize types of experiments for geological hypothesis testing"""
        return {
            'field_observation': {
                'description': 'Direct field measurements and observations',
                'cost': 'medium',
                'duration': 'weeks_to_months',
                'precision': 'high',
                'applicability': ['surface_processes', 'structural_geology', 'volcanism']
            },
            'laboratory_experiment': {
                'description': 'Controlled laboratory experiments',
                'cost': 'low_to_medium',
                'duration': 'days_to_weeks',
                'precision': 'very_high',
                'applicability': ['material_properties', 'chemical_reactions', 'phase_transitions']
            },
            'numerical_simulation': {
                'description': 'Computational modeling and simulation',
                'cost': 'low',
                'duration': 'hours_to_days',
                'precision': 'variable',
                'applicability': ['complex_systems', 'long_timescales', 'extreme_conditions']
            },
            'analog_modeling': {
                'description': 'Physical analog experiments',
                'cost': 'medium',
                'duration': 'days_to_weeks',
                'precision': 'medium',
                'applicability': ['tectonic_processes', 'fluid_flow', 'deformation']
            },
            'remote_sensing': {
                'description': 'Satellite and airborne observations',
                'cost': 'medium_to_high',
                'duration': 'months_to_years',
                'precision': 'medium',
                'applicability': ['large_scale_processes', 'temporal_changes', 'inaccessible_areas']
            },
            'geophysical_survey': {
                'description': 'Seismic, gravity, magnetic, electrical surveys',
                'cost': 'high',
                'duration': 'weeks_to_months',
                'precision': 'high',
                'applicability': ['subsurface_structure', 'deep_processes', 'regional_studies']
            }
        }
    
    def _initialize_geological_contexts(self) -> List[str]:
        """Initialize geological contexts for experiment design"""
        return [
            'volcanic_systems', 'tectonic_boundaries', 'sedimentary_basins',
            'metamorphic_terranes', 'hydrothermal_systems', 'ore_deposits',
            'groundwater_systems', 'coastal_environments', 'mountain_belts'
        ]
    
    def design_validation_experiments(
        self, 
        hypothesis: ScientificHypothesis,
        available_resources: Dict[str, Any],
        geological_context: str
    ) -> List[Dict[str, Any]]:
        """
        Design experiments to validate a scientific hypothesis
        
        Args:
            hypothesis: The hypothesis to validate
            available_resources: Available resources (budget, time, equipment)
            geological_context: Geological context for the experiments
            
        Returns:
            List of experimental designs
        """
        
        experimental_designs = []
        
        # Determine appropriate experiment types
        suitable_experiments = self._select_suitable_experiments(
            hypothesis, geological_context, available_resources
        )
        
        for exp_type in suitable_experiments:
            design = self._design_specific_experiment(
                hypothesis, exp_type, geological_context, available_resources
            )
            experimental_designs.append(design)
        
        # Rank experiments by feasibility and expected information gain
        ranked_designs = self._rank_experimental_designs(
            experimental_designs, available_resources
        )
        
        logger.info(f"Designed {len(ranked_designs)} validation experiments for hypothesis {hypothesis.id}")
        
        return ranked_designs
    
    def _select_suitable_experiments(
        self, 
        hypothesis: ScientificHypothesis,
        geological_context: str,
        resources: Dict[str, Any]
    ) -> List[str]:
        """Select experiment types suitable for the hypothesis and context"""
        
        suitable_types = []
        
        # Analyze hypothesis content to determine suitable experiment types
        statement_lower = hypothesis.statement.lower()
        
        # Field observation suitable for surface processes
        if any(term in statement_lower for term in ['surface', 'outcrop', 'field', 'exposure']):
            suitable_types.append('field_observation')
        
        # Laboratory experiments for material properties
        if any(term in statement_lower for term in ['temperature', 'pressure', 'composition', 'reaction']):
            suitable_types.append('laboratory_experiment')
        
        # Numerical simulation for complex systems
        if any(term in statement_lower for term in ['system', 'interaction', 'feedback', 'coupling']):
            suitable_types.append('numerical_simulation')
        
        # Analog modeling for physical processes
        if any(term in statement_lower for term in ['deformation', 'flow', 'stress', 'strain']):
            suitable_types.append('analog_modeling')
        
        # Remote sensing for large-scale processes
        if any(term in statement_lower for term in ['regional', 'large-scale', 'temporal', 'evolution']):
            suitable_types.append('remote_sensing')
        
        # Geophysical surveys for subsurface processes
        if any(term in statement_lower for term in ['subsurface', 'deep', 'structure', 'seismic']):
            suitable_types.append('geophysical_survey')
        
        # Filter by available resources
        budget = resources.get('budget', 'medium')
        time_available = resources.get('time', 'medium')
        
        filtered_types = []
        for exp_type in suitable_types:
            exp_info = self.experiment_types[exp_type]
            if self._is_feasible(exp_info, budget, time_available):
                filtered_types.append(exp_type)
        
        return filtered_types or ['numerical_simulation']  # Default to simulation if nothing else feasible
    
    def _design_specific_experiment(
        self, 
        hypothesis: ScientificHypothesis,
        experiment_type: str,
        geological_context: str,
        resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design a specific experiment for hypothesis validation"""
        
        exp_info = self.experiment_types[experiment_type]
        
        # Extract variables from hypothesis
        variables = self._extract_testable_variables(hypothesis)
        
        # Design experiment based on type
        if experiment_type == 'field_observation':
            design = self._design_field_observation(hypothesis, variables, geological_context)
        elif experiment_type == 'laboratory_experiment':
            design = self._design_laboratory_experiment(hypothesis, variables)
        elif experiment_type == 'numerical_simulation':
            design = self._design_numerical_simulation(hypothesis, variables)
        elif experiment_type == 'analog_modeling':
            design = self._design_analog_modeling(hypothesis, variables)
        elif experiment_type == 'remote_sensing':
            design = self._design_remote_sensing(hypothesis, variables, geological_context)
        elif experiment_type == 'geophysical_survey':
            design = self._design_geophysical_survey(hypothesis, variables, geological_context)
        else:
            design = self._design_default_experiment(hypothesis, variables)
        
        # Add common experiment metadata
        design.update({
            'experiment_type': experiment_type,
            'hypothesis_id': hypothesis.id,
            'estimated_cost': exp_info['cost'],
            'estimated_duration': exp_info['duration'],
            'expected_precision': exp_info['precision'],
            'success_criteria': self._define_success_criteria(hypothesis),
            'risk_assessment': self._assess_experiment_risks(experiment_type, geological_context),
            'required_equipment': self._list_required_equipment(experiment_type, variables),
            'data_collection_protocol': self._design_data_collection_protocol(experiment_type, variables)
        })
        
        return design
    
    def _design_field_observation(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str], 
        context: str
    ) -> Dict[str, Any]:
        """Design field observation experiment"""
        return {
            'title': f'Field validation of {hypothesis.statement[:50]}...',
            'objective': f'Validate hypothesis through direct field measurements',
            'methodology': 'Systematic field measurements and observations',
            'target_variables': variables,
            'sampling_strategy': 'Stratified sampling across geological units',
            'measurement_techniques': self._select_field_techniques(variables),
            'site_selection_criteria': self._define_site_criteria(context),
            'quality_control': 'Duplicate measurements, calibration checks'
        }
    
    def _design_laboratory_experiment(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str]
    ) -> Dict[str, Any]:
        """Design laboratory experiment"""
        return {
            'title': f'Laboratory validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis under controlled laboratory conditions',
            'methodology': 'Controlled parameter variation experiments',
            'experimental_conditions': self._define_lab_conditions(variables),
            'control_variables': self._identify_control_variables(variables),
            'measurement_protocol': self._design_measurement_protocol(variables),
            'sample_preparation': 'Standard geological sample preparation techniques',
            'replication_strategy': 'Minimum 3 replicates per condition'
        }
    
    def _design_numerical_simulation(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str]
    ) -> Dict[str, Any]:
        """Design numerical simulation experiment"""
        return {
            'title': f'Numerical validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis through computational modeling',
            'methodology': 'Physics-based numerical simulation',
            'model_type': self._select_simulation_model(hypothesis),
            'boundary_conditions': self._define_boundary_conditions(variables),
            'parameter_space': self._define_parameter_space(variables),
            'validation_metrics': self._define_validation_metrics(hypothesis),
            'sensitivity_analysis': 'Monte Carlo parameter sensitivity analysis'
        }
    
    def _design_analog_modeling(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str]
    ) -> Dict[str, Any]:
        """Design analog modeling experiment"""
        return {
            'title': f'Analog validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis using physical analog models',
            'methodology': 'Scaled physical analog experiments',
            'scaling_relationships': self._define_scaling_laws(variables),
            'analog_materials': self._select_analog_materials(variables),
            'measurement_system': 'High-resolution imaging and sensors',
            'experimental_setup': self._design_analog_setup(variables)
        }
    
    def _design_remote_sensing(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str], 
        context: str
    ) -> Dict[str, Any]:
        """Design remote sensing experiment"""
        return {
            'title': f'Remote sensing validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis using satellite/airborne observations',
            'methodology': 'Multi-temporal remote sensing analysis',
            'sensor_requirements': self._select_sensors(variables),
            'temporal_resolution': self._define_temporal_requirements(hypothesis),
            'spatial_resolution': self._define_spatial_requirements(context),
            'processing_workflow': self._design_processing_workflow(variables),
            'validation_approach': 'Ground truth validation with field data'
        }
    
    def _design_geophysical_survey(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str], 
        context: str
    ) -> Dict[str, Any]:
        """Design geophysical survey experiment"""
        return {
            'title': f'Geophysical validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis using geophysical methods',
            'methodology': 'Multi-method geophysical survey',
            'survey_methods': self._select_geophysical_methods(variables),
            'survey_design': self._design_survey_geometry(context),
            'data_processing': 'Standard geophysical data processing workflows',
            'interpretation_approach': 'Integrated geophysical interpretation',
            'resolution_targets': self._define_resolution_targets(variables)
        }
    
    def _design_default_experiment(
        self, 
        hypothesis: ScientificHypothesis, 
        variables: List[str]
    ) -> Dict[str, Any]:
        """Design default experiment when specific type not available"""
        return {
            'title': f'General validation of {hypothesis.statement[:50]}...',
            'objective': 'Test hypothesis using available methods',
            'methodology': 'Multi-method approach combining available techniques',
            'target_variables': variables,
            'approach': 'Adaptive experimental design based on available resources'
        }
    
    # Helper methods for experiment design
    def _extract_testable_variables(self, hypothesis: ScientificHypothesis) -> List[str]:
        """Extract testable variables from hypothesis"""
        # Simple extraction based on common geological terms
        geological_terms = [
            'temperature', 'pressure', 'density', 'viscosity', 'velocity',
            'stress', 'strain', 'permeability', 'porosity', 'composition',
            'time', 'depth', 'distance', 'volume', 'mass'
        ]
        
        statement_lower = hypothesis.statement.lower()
        found_variables = [term for term in geological_terms if term in statement_lower]
        
        return found_variables or ['unknown_variable']
    
    def _is_feasible(self, exp_info: Dict[str, Any], budget: str, time: str) -> bool:
        """Check if experiment is feasible given constraints"""
        # Simplified feasibility check
        cost_levels = {'low': 1, 'low_to_medium': 2, 'medium': 3, 'medium_to_high': 4, 'high': 5}
        time_levels = {'hours_to_days': 1, 'days_to_weeks': 2, 'weeks_to_months': 3, 'months_to_years': 4}
        
        exp_cost = cost_levels.get(exp_info['cost'], 3)
        exp_time = time_levels.get(exp_info['duration'], 3)
        
        available_cost = cost_levels.get(budget, 3)
        available_time = time_levels.get(time, 3)
        
        return exp_cost <= available_cost and exp_time <= available_time
    
    def _rank_experimental_designs(
        self, 
        designs: List[Dict[str, Any]], 
        resources: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Rank experimental designs by feasibility and information gain"""
        
        def score_design(design):
            # Simple scoring based on cost, duration, and precision
            cost_score = {'low': 5, 'medium': 3, 'high': 1}.get(design.get('estimated_cost', 'medium'), 3)
            precision_score = {'very_high': 5, 'high': 4, 'medium': 3, 'low': 2}.get(design.get('expected_precision', 'medium'), 3)
            
            return cost_score + precision_score
        
        return sorted(designs, key=score_design, reverse=True)


class HypothesisValidationSystem:
    """
    Main system integrating all components for hypothesis generation and validation
    """
    
    def __init__(self):
        self.hypothesis_generator = HypothesisGenerator()
        self.math_formulator = MathematicalFormulationEngine()
        self.confidence_assessor = ConfidenceAssessmentEngine()
        self.experiment_designer = ExperimentalDesignEngine()
        
    def generate_and_validate_hypotheses(
        self,
        causal_relationships: List[CausalRelationship],
        geological_data: Dict[str, np.ndarray],
        geological_context: Dict[str, Any],
        available_resources: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete workflow for hypothesis generation and validation design
        
        Args:
            causal_relationships: Discovered causal relationships
            geological_data: Supporting geological data
            geological_context: Geological context information
            available_resources: Available resources for validation
            
        Returns:
            Complete analysis results including hypotheses, laws, and experiments
        """
        
        logger.info("Starting hypothesis generation and validation workflow")
        
        # Generate hypotheses from causal relationships
        hypotheses = self.hypothesis_generator.generate_hypotheses_from_causal_relationships(
            causal_relationships, geological_context
        )
        
        # Formulate mathematical laws
        geological_laws = []
        if len(causal_relationships) > 0:
            law = self.math_formulator.formulate_geological_law(
                causal_relationships, geological_data, geological_context
            )
            geological_laws.append(law)
        
        # Assess confidence for each hypothesis
        validated_hypotheses = []
        for hypothesis in hypotheses:
            confidence = self.confidence_assessor.assess_hypothesis_confidence(
                hypothesis, geological_data
            )
            
            # Design validation experiments for high-confidence hypotheses
            if confidence > 0.6:
                experiments = self.experiment_designer.design_validation_experiments(
                    hypothesis, 
                    available_resources, 
                    geological_context.get('setting', 'general')
                )
                hypothesis.experimental_design = experiments
            
            validated_hypotheses.append(hypothesis)
        
        # Sort hypotheses by confidence and novelty
        validated_hypotheses.sort(
            key=lambda h: h.confidence_score * h.novelty_score, 
            reverse=True
        )
        
        results = {
            'generated_hypotheses': validated_hypotheses,
            'formulated_laws': geological_laws,
            'total_hypotheses': len(validated_hypotheses),
            'high_confidence_hypotheses': len([h for h in validated_hypotheses if h.confidence_score > 0.7]),
            'testable_hypotheses': len([h for h in validated_hypotheses if h.experimental_design]),
            'generation_timestamp': datetime.now(),
            'summary': self._generate_summary(validated_hypotheses, geological_laws)
        }
        
        logger.info(f"Generated {len(validated_hypotheses)} hypotheses and {len(geological_laws)} laws")
        
        return results
    
    def _generate_summary(
        self, 
        hypotheses: List[ScientificHypothesis], 
        laws: List[GeologicalLaw]
    ) -> Dict[str, Any]:
        """Generate summary of hypothesis generation results"""
        
        if not hypotheses:
            return {'message': 'No hypotheses generated'}
        
        avg_confidence = np.mean([h.confidence_score for h in hypotheses])
        avg_novelty = np.mean([h.novelty_score for h in hypotheses])
        avg_testability = np.mean([h.testability_score for h in hypotheses])
        
        top_hypothesis = max(hypotheses, key=lambda h: h.confidence_score)
        
        return {
            'average_confidence': avg_confidence,
            'average_novelty': avg_novelty,
            'average_testability': avg_testability,
            'top_hypothesis': {
                'statement': top_hypothesis.statement,
                'confidence': top_hypothesis.confidence_score,
                'novelty': top_hypothesis.novelty_score
            },
            'formulated_laws_count': len(laws),
            'recommendations': self._generate_recommendations(hypotheses, laws)
        }
    
    def _generate_recommendations(
        self, 
        hypotheses: List[ScientificHypothesis], 
        laws: List[GeologicalLaw]
    ) -> List[str]:
        """Generate recommendations for further research"""
        
        recommendations = []
        
        # High confidence hypotheses
        high_conf_hypotheses = [h for h in hypotheses if h.confidence_score > 0.8]
        if high_conf_hypotheses:
            recommendations.append(
                f"Prioritize validation of {len(high_conf_hypotheses)} high-confidence hypotheses"
            )
        
        # Novel hypotheses
        novel_hypotheses = [h for h in hypotheses if h.novelty_score > 0.8]
        if novel_hypotheses:
            recommendations.append(
                f"Investigate {len(novel_hypotheses)} highly novel hypotheses for breakthrough potential"
            )
        
        # Testable hypotheses
        testable_hypotheses = [h for h in hypotheses if h.experimental_design]
        if testable_hypotheses:
            recommendations.append(
                f"Begin experimental validation of {len(testable_hypotheses)} testable hypotheses"
            )
        
        # Mathematical laws
        if laws:
            recommendations.append(
                f"Validate {len(laws)} formulated geological laws through independent datasets"
            )
        
        return recommendations

# Helper methods for HypothesisGenerator class (continued from earlier)
def _infer_mechanism(self, relationship: CausalRelationship, context: Dict[str, Any]) -> str:
    """Infer physical mechanism for causal relationship"""
    
    # Simple mechanism inference based on variable types
    cause_lower = relationship.cause.lower()
    effect_lower = relationship.effect.lower()
    
    if 'temperature' in cause_lower and 'pressure' in effect_lower:
        return 'thermal_expansion'
    elif 'pressure' in cause_lower and 'deformation' in effect_lower:
        return 'elastic_response'
    elif 'stress' in cause_lower and 'fracture' in effect_lower:
        return 'brittle_failure'
    elif 'fluid' in cause_lower and 'transport' in effect_lower:
        return 'advective_transport'
    else:
        return 'unknown_mechanism'

# Add remaining helper methods to HypothesisGenerator
HypothesisGenerator._infer_mechanism = _infer_mechanism

# Additional helper methods would be implemented here for completeness
# These are abbreviated for brevity but would include full implementations

# Helper methods for HypothesisGenerator class
def _calculate_hypothesis_confidence(self, relationship: CausalRelationship, context: Dict[str, Any]) -> float:
    """Calculate confidence score for hypothesis based on relationship and context"""
    base_confidence = relationship.confidence
    
    # Adjust based on mechanism clarity
    mechanism_bonus = 0.1 if relationship.mechanism else 0.0
    
    # Adjust based on evidence quantity
    evidence_bonus = min(len(relationship.evidence) * 0.05, 0.2)
    
    return min(base_confidence + mechanism_bonus + evidence_bonus, 1.0)

def _calculate_testability_score(self, relationship: CausalRelationship, context: Dict[str, Any]) -> float:
    """Calculate testability score for hypothesis"""
    # Base testability on variable types
    measurable_vars = ['temperature', 'pressure', 'density', 'velocity', 'stress', 'strain']
    cause_measurable = any(var in relationship.cause.lower() for var in measurable_vars)
    effect_measurable = any(var in relationship.effect.lower() for var in measurable_vars)
    
    testability = 0.5  # Base score
    if cause_measurable:
        testability += 0.2
    if effect_measurable:
        testability += 0.2
    if relationship.mechanism:
        testability += 0.1
    
    return min(testability, 1.0)

def _calculate_novelty_score(self, relationship: CausalRelationship, context: Dict[str, Any]) -> float:
    """Calculate novelty score for hypothesis"""
    # Simple novelty assessment based on relationship strength and mechanism
    novelty = 0.5  # Base novelty
    
    # Higher strength relationships might be more novel
    if abs(relationship.strength) > 0.8:
        novelty += 0.2
    
    # Uncommon mechanisms might be more novel
    common_mechanisms = ['thermal_expansion', 'elastic_response', 'pressure_change']
    if relationship.mechanism and relationship.mechanism not in common_mechanisms:
        novelty += 0.2
    
    return min(novelty, 1.0)

def _generate_predicted_outcomes(self, relationship: CausalRelationship, context: Dict[str, Any]) -> List[str]:
    """Generate predicted outcomes for hypothesis"""
    outcomes = []
    
    direction = "increase" if relationship.strength > 0 else "decrease"
    
    outcomes.append(f"When {relationship.cause} increases, {relationship.effect} should {direction}")
    
    if relationship.mechanism:
        outcomes.append(f"The {relationship.mechanism} mechanism should be observable")
    
    # Add context-specific outcomes
    if context.get('geological_setting') == 'volcanic_system':
        outcomes.append("Changes should be detectable through volcanic monitoring networks")
    
    return outcomes

def _identify_alternative_mechanisms(self, relationship: CausalRelationship, context: Dict[str, Any]) -> List[str]:
    """Identify alternative mechanisms for causal relationship"""
    alternatives = []
    
    # Based on variable types, suggest alternative mechanisms
    if 'temperature' in relationship.cause.lower() and 'pressure' in relationship.effect.lower():
        alternatives.extend(['gas_expansion', 'phase_transition', 'chemical_reaction'])
    elif 'pressure' in relationship.cause.lower() and 'deformation' in relationship.effect.lower():
        alternatives.extend(['plastic_deformation', 'brittle_failure', 'viscous_flow'])
    else:
        alternatives.extend(['indirect_coupling', 'feedback_mechanism', 'threshold_effect'])
    
    return alternatives[:3]  # Limit to 3 alternatives

def _generate_interaction_outcomes(self, rel1: CausalRelationship, rel2: CausalRelationship) -> List[str]:
    """Generate predicted outcomes for interaction hypotheses"""
    outcomes = []
    
    if rel1.effect == rel2.cause:
        outcomes.append(f"Chain effect: {rel1.cause} → {rel1.effect} → {rel2.effect}")
        outcomes.append(f"Combined strength should be approximately {rel1.strength * rel2.strength:.3f}")
    
    outcomes.append("Interaction effects should be measurable in field data")
    
    return outcomes

def _calculate_interaction_testability(self, rel1: CausalRelationship, rel2: CausalRelationship) -> float:
    """Calculate testability for interaction hypotheses"""
    # Interaction hypotheses are generally harder to test
    base_testability = min(
        self._calculate_testability_score(rel1, {}),
        self._calculate_testability_score(rel2, {})
    )
    
    return base_testability * 0.8  # Reduce by 20% for interaction complexity

def _remove_duplicate_hypotheses(self, hypotheses: List[ScientificHypothesis]) -> List[ScientificHypothesis]:
    """Remove duplicate hypotheses based on statement similarity"""
    unique_hypotheses = []
    seen_statements = set()
    
    for hypothesis in hypotheses:
        # Simple duplicate detection based on first 100 characters
        statement_key = hypothesis.statement[:100].lower()
        if statement_key not in seen_statements:
            unique_hypotheses.append(hypothesis)
            seen_statements.add(statement_key)
    
    return unique_hypotheses

def _rank_hypotheses(self, hypotheses: List[ScientificHypothesis]) -> List[ScientificHypothesis]:
    """Rank hypotheses by combined score of confidence, novelty, and testability"""
    def combined_score(h):
        return h.confidence_score * 0.4 + h.novelty_score * 0.3 + h.testability_score * 0.3
    
    return sorted(hypotheses, key=combined_score, reverse=True)

# Add these methods to the HypothesisGenerator class
HypothesisGenerator._calculate_hypothesis_confidence = _calculate_hypothesis_confidence
HypothesisGenerator._calculate_testability_score = _calculate_testability_score
HypothesisGenerator._calculate_novelty_score = _calculate_novelty_score
HypothesisGenerator._generate_predicted_outcomes = _generate_predicted_outcomes
HypothesisGenerator._identify_alternative_mechanisms = _identify_alternative_mechanisms
HypothesisGenerator._generate_interaction_outcomes = _generate_interaction_outcomes
HypothesisGenerator._calculate_interaction_testability = _calculate_interaction_testability
HypothesisGenerator._remove_duplicate_hypotheses = _remove_duplicate_hypotheses
HypothesisGenerator._rank_hypotheses = _rank_hypotheses

# Helper methods for MathematicalFormulationEngine
def _calculate_law_confidence(self, expression: sp.Expr, relationships: List[CausalRelationship], data_patterns: Dict[str, np.ndarray]) -> float:
    """Calculate confidence in formulated law"""
    # Base confidence on average relationship confidence
    if not relationships:
        return 0.5
    
    avg_rel_confidence = np.mean([rel.confidence for rel in relationships])
    
    # Adjust based on data quality
    data_quality_bonus = 0.0
    if data_patterns:
        # Simple data quality assessment
        total_samples = sum(len(data) for data in data_patterns.values() if hasattr(data, '__len__'))
        if total_samples > 100:
            data_quality_bonus = 0.1
        elif total_samples > 50:
            data_quality_bonus = 0.05
    
    return min(avg_rel_confidence + data_quality_bonus, 1.0)

def _determine_domain_applicability(self, relationships: List[CausalRelationship], context: Dict[str, Any]) -> List[str]:
    """Determine domain of applicability for geological law"""
    domains = []
    
    # Based on context
    if 'volcanic' in str(context).lower():
        domains.append('volcanic_systems')
    if 'tectonic' in str(context).lower():
        domains.append('tectonic_processes')
    if 'hydrothermal' in str(context).lower():
        domains.append('hydrothermal_systems')
    
    # Based on variables in relationships
    for rel in relationships:
        if 'temperature' in rel.cause.lower() or 'temperature' in rel.effect.lower():
            domains.append('thermal_processes')
        if 'pressure' in rel.cause.lower() or 'pressure' in rel.effect.lower():
            domains.append('pressure_systems')
        if 'deformation' in rel.cause.lower() or 'deformation' in rel.effect.lower():
            domains.append('deformation_processes')
    
    return list(set(domains)) or ['general_geological_systems']

def _design_validation_experiments(self, expression: sp.Expr, domains: List[str]) -> List[Dict[str, Any]]:
    """Design validation experiments for geological law"""
    experiments = []
    
    for domain in domains:
        if domain == 'volcanic_systems':
            experiments.append({
                'type': 'field_observation',
                'description': 'Validate law using volcanic monitoring data',
                'duration': 'months',
                'cost': 'medium'
            })
        elif domain == 'thermal_processes':
            experiments.append({
                'type': 'laboratory_experiment',
                'description': 'Test thermal relationships under controlled conditions',
                'duration': 'weeks',
                'cost': 'low'
            })
        else:
            experiments.append({
                'type': 'numerical_simulation',
                'description': f'Validate law through {domain} modeling',
                'duration': 'days',
                'cost': 'low'
            })
    
    return experiments

def _generate_law_name(self, relationships: List[CausalRelationship]) -> str:
    """Generate name for geological law"""
    if not relationships:
        return "Unknown_Geological_Law"
    
    primary_rel = max(relationships, key=lambda r: r.confidence)
    
    # Create name based on primary relationship
    cause_clean = primary_rel.cause.replace('_', ' ').title()
    effect_clean = primary_rel.effect.replace('_', ' ').title()
    
    return f"{cause_clean}_{effect_clean}_Relationship_Law"

def _extract_variables(self, expression: sp.Expr) -> List[str]:
    """Extract variables from mathematical expression"""
    if hasattr(expression, 'free_symbols'):
        return [str(symbol) for symbol in expression.free_symbols]
    else:
        return ['unknown_variable']

# Add these methods to the MathematicalFormulationEngine class
MathematicalFormulationEngine._calculate_law_confidence = _calculate_law_confidence
MathematicalFormulationEngine._determine_domain_applicability = _determine_domain_applicability
MathematicalFormulationEngine._design_validation_experiments = _design_validation_experiments
MathematicalFormulationEngine._generate_law_name = _generate_law_name
MathematicalFormulationEngine._extract_variables = _extract_variables

# Helper methods for ExperimentalDesignEngine
def _select_field_techniques(self, variables: List[str]) -> List[str]:
    """Select appropriate field measurement techniques"""
    techniques = []
    
    for var in variables:
        if 'temperature' in var.lower():
            techniques.extend(['thermal_infrared_camera', 'thermocouple_measurements'])
        elif 'pressure' in var.lower():
            techniques.extend(['pressure_sensors', 'barometric_measurements'])
        elif 'deformation' in var.lower():
            techniques.extend(['GPS_measurements', 'InSAR_analysis', 'tiltmeters'])
        elif 'seismic' in var.lower():
            techniques.extend(['seismometer_networks', 'accelerometers'])
    
    return list(set(techniques)) or ['general_field_measurements']

def _define_site_criteria(self, context: str) -> List[str]:
    """Define site selection criteria"""
    criteria = ['accessibility', 'safety', 'representative_conditions']
    
    if 'volcanic' in context.lower():
        criteria.extend(['active_volcanic_area', 'monitoring_infrastructure'])
    elif 'tectonic' in context.lower():
        criteria.extend(['fault_proximity', 'seismic_activity'])
    
    return criteria

def _define_lab_conditions(self, variables: List[str]) -> Dict[str, str]:
    """Define laboratory experimental conditions"""
    conditions = {
        'temperature_control': 'precise_temperature_regulation',
        'pressure_control': 'controlled_pressure_environment',
        'atmosphere': 'controlled_atmospheric_conditions'
    }
    
    for var in variables:
        if 'high_temperature' in var.lower():
            conditions['temperature_range'] = '800-1200°C'
        elif 'high_pressure' in var.lower():
            conditions['pressure_range'] = '1-10_GPa'
    
    return conditions

def _identify_control_variables(self, variables: List[str]) -> List[str]:
    """Identify control variables for experiments"""
    control_vars = []
    
    # Standard geological control variables
    standard_controls = ['temperature', 'pressure', 'composition', 'time', 'sample_size']
    
    for var in variables:
        for control in standard_controls:
            if control not in var.lower():
                control_vars.append(control)
    
    return list(set(control_vars))[:5]  # Limit to 5 control variables

def _design_measurement_protocol(self, variables: List[str]) -> Dict[str, str]:
    """Design measurement protocol"""
    protocol = {
        'sampling_frequency': 'continuous_or_high_frequency',
        'calibration': 'regular_calibration_checks',
        'quality_control': 'duplicate_measurements',
        'data_recording': 'automated_digital_recording'
    }
    
    return protocol

def _select_simulation_model(self, hypothesis: ScientificHypothesis) -> str:
    """Select appropriate simulation model type"""
    statement_lower = hypothesis.statement.lower()
    
    if 'thermal' in statement_lower or 'temperature' in statement_lower:
        return 'finite_element_thermal_model'
    elif 'fluid' in statement_lower or 'flow' in statement_lower:
        return 'computational_fluid_dynamics'
    elif 'deformation' in statement_lower or 'stress' in statement_lower:
        return 'finite_element_mechanical_model'
    else:
        return 'physics_informed_neural_network'

def _define_boundary_conditions(self, variables: List[str]) -> Dict[str, str]:
    """Define boundary conditions for simulation"""
    conditions = {}
    
    for var in variables:
        if 'temperature' in var.lower():
            conditions['thermal_boundary'] = 'fixed_temperature_or_heat_flux'
        elif 'pressure' in var.lower():
            conditions['pressure_boundary'] = 'fixed_pressure_or_stress'
        elif 'velocity' in var.lower():
            conditions['velocity_boundary'] = 'no_slip_or_free_slip'
    
    return conditions or {'general_boundary': 'physically_realistic_constraints'}

def _define_parameter_space(self, variables: List[str]) -> Dict[str, str]:
    """Define parameter space for simulation"""
    param_space = {}
    
    for var in variables:
        if 'temperature' in var.lower():
            param_space['temperature_range'] = '273-1500_K'
        elif 'pressure' in var.lower():
            param_space['pressure_range'] = '1e5-1e9_Pa'
        elif 'density' in var.lower():
            param_space['density_range'] = '1000-5000_kg_m3'
    
    return param_space or {'general_range': 'geologically_realistic_values'}

def _define_validation_metrics(self, hypothesis: ScientificHypothesis) -> List[str]:
    """Define validation metrics for simulation"""
    metrics = ['correlation_coefficient', 'root_mean_square_error', 'coefficient_of_determination']
    
    if 'prediction' in hypothesis.statement.lower():
        metrics.append('prediction_accuracy')
    if 'trend' in hypothesis.statement.lower():
        metrics.append('trend_analysis')
    
    return metrics

# Add remaining helper methods to ExperimentalDesignEngine
ExperimentalDesignEngine._select_field_techniques = _select_field_techniques
ExperimentalDesignEngine._define_site_criteria = _define_site_criteria
ExperimentalDesignEngine._define_lab_conditions = _define_lab_conditions
ExperimentalDesignEngine._identify_control_variables = _identify_control_variables
ExperimentalDesignEngine._design_measurement_protocol = _design_measurement_protocol
ExperimentalDesignEngine._select_simulation_model = _select_simulation_model
ExperimentalDesignEngine._define_boundary_conditions = _define_boundary_conditions
ExperimentalDesignEngine._define_parameter_space = _define_parameter_space
ExperimentalDesignEngine._define_validation_metrics = _define_validation_metrics

# Additional helper methods for remaining experimental design functions
def _define_scaling_laws(self, variables: List[str]) -> Dict[str, str]:
    """Define scaling laws for analog modeling"""
    return {
        'geometric_scaling': '1:10000_to_1:100000',
        'dynamic_scaling': 'froude_number_similarity',
        'material_scaling': 'density_and_viscosity_ratios'
    }

def _select_analog_materials(self, variables: List[str]) -> List[str]:
    """Select analog materials for physical modeling"""
    materials = ['silicone_putty', 'sand', 'clay', 'gelatin']
    
    for var in variables:
        if 'viscous' in var.lower():
            materials.append('honey_or_corn_syrup')
        elif 'brittle' in var.lower():
            materials.append('plaster_of_paris')
    
    return materials

def _design_analog_setup(self, variables: List[str]) -> Dict[str, str]:
    """Design analog experimental setup"""
    return {
        'apparatus': 'deformation_rig_with_imaging',
        'measurement_system': 'particle_image_velocimetry',
        'boundary_conditions': 'controlled_loading_system',
        'monitoring': 'high_speed_cameras_and_sensors'
    }

def _select_sensors(self, variables: List[str]) -> List[str]:
    """Select appropriate sensors for remote sensing"""
    sensors = []
    
    for var in variables:
        if 'temperature' in var.lower():
            sensors.extend(['thermal_infrared', 'MODIS', 'ASTER'])
        elif 'deformation' in var.lower():
            sensors.extend(['InSAR', 'GPS', 'LiDAR'])
        elif 'gas' in var.lower():
            sensors.extend(['hyperspectral', 'DOAS'])
    
    return sensors or ['multispectral_optical']

def _define_temporal_requirements(self, hypothesis: ScientificHypothesis) -> str:
    """Define temporal resolution requirements"""
    if 'rapid' in hypothesis.statement.lower() or 'sudden' in hypothesis.statement.lower():
        return 'high_temporal_resolution_daily_or_hourly'
    elif 'long-term' in hypothesis.statement.lower():
        return 'monthly_to_yearly_observations'
    else:
        return 'weekly_to_monthly_observations'

def _define_spatial_requirements(self, context: str) -> str:
    """Define spatial resolution requirements"""
    if 'local' in context.lower():
        return 'high_spatial_resolution_1-10_meters'
    elif 'regional' in context.lower():
        return 'medium_spatial_resolution_10-100_meters'
    else:
        return 'variable_spatial_resolution_based_on_process_scale'

def _design_processing_workflow(self, variables: List[str]) -> List[str]:
    """Design data processing workflow"""
    workflow = ['data_acquisition', 'preprocessing', 'analysis', 'validation']
    
    for var in variables:
        if 'deformation' in var.lower():
            workflow.insert(-1, 'interferometric_processing')
        elif 'temperature' in var.lower():
            workflow.insert(-1, 'thermal_calibration')
    
    return workflow

def _select_geophysical_methods(self, variables: List[str]) -> List[str]:
    """Select geophysical survey methods"""
    methods = []
    
    for var in variables:
        if 'seismic' in var.lower() or 'velocity' in var.lower():
            methods.append('seismic_reflection_refraction')
        elif 'density' in var.lower():
            methods.append('gravity_survey')
        elif 'magnetic' in var.lower():
            methods.append('magnetic_survey')
        elif 'electrical' in var.lower():
            methods.append('electrical_resistivity')
    
    return methods or ['multi_method_geophysical_survey']

def _design_survey_geometry(self, context: str) -> Dict[str, str]:
    """Design survey geometry"""
    return {
        'survey_lines': 'grid_or_profile_based_on_target',
        'station_spacing': 'optimized_for_target_resolution',
        'coverage': 'comprehensive_spatial_coverage',
        'logistics': 'accessible_and_safe_locations'
    }

def _define_resolution_targets(self, variables: List[str]) -> Dict[str, str]:
    """Define resolution targets for geophysical survey"""
    targets = {}
    
    for var in variables:
        if 'shallow' in var.lower():
            targets['depth_resolution'] = '0-100_meters'
        elif 'deep' in var.lower():
            targets['depth_resolution'] = '1-10_kilometers'
        else:
            targets['depth_resolution'] = '100_meters_to_1_kilometer'
    
    return targets

def _define_success_criteria(self, hypothesis: ScientificHypothesis) -> List[str]:
    """Define success criteria for experiment"""
    criteria = [
        'statistical_significance_p_less_than_0.05',
        'effect_size_greater_than_minimum_detectable',
        'reproducible_results_across_conditions'
    ]
    
    if 'correlation' in hypothesis.statement.lower():
        criteria.append('correlation_coefficient_greater_than_0.7')
    if 'prediction' in hypothesis.statement.lower():
        criteria.append('prediction_accuracy_greater_than_80_percent')
    
    return criteria

def _assess_experiment_risks(self, experiment_type: str, context: str) -> Dict[str, str]:
    """Assess risks for experiment"""
    risks = {
        'technical_risk': 'medium',
        'safety_risk': 'low_to_medium',
        'weather_dependency': 'medium',
        'equipment_failure': 'low'
    }
    
    if experiment_type == 'field_observation' and 'volcanic' in context.lower():
        risks['safety_risk'] = 'high'
        risks['weather_dependency'] = 'high'
    elif experiment_type == 'laboratory_experiment':
        risks['safety_risk'] = 'low'
        risks['weather_dependency'] = 'none'
    
    return risks

def _list_required_equipment(self, experiment_type: str, variables: List[str]) -> List[str]:
    """List required equipment for experiment"""
    equipment = []
    
    if experiment_type == 'field_observation':
        equipment.extend(['field_instruments', 'data_loggers', 'power_supply', 'communication_equipment'])
        for var in variables:
            if 'temperature' in var.lower():
                equipment.append('thermal_cameras')
            elif 'seismic' in var.lower():
                equipment.append('seismometers')
    elif experiment_type == 'laboratory_experiment':
        equipment.extend(['controlled_environment_chamber', 'measurement_instruments', 'sample_preparation_tools'])
    
    return equipment

def _design_data_collection_protocol(self, experiment_type: str, variables: List[str]) -> Dict[str, str]:
    """Design data collection protocol"""
    protocol = {
        'sampling_strategy': 'systematic_sampling',
        'quality_control': 'regular_calibration_and_validation',
        'data_storage': 'redundant_digital_storage',
        'documentation': 'comprehensive_metadata_recording'
    }
    
    if experiment_type == 'field_observation':
        protocol['timing'] = 'continuous_or_scheduled_measurements'
    elif experiment_type == 'laboratory_experiment':
        protocol['timing'] = 'controlled_experimental_schedule'
    
    return protocol

# Add remaining helper methods to ExperimentalDesignEngine
ExperimentalDesignEngine._define_scaling_laws = _define_scaling_laws
ExperimentalDesignEngine._select_analog_materials = _select_analog_materials
ExperimentalDesignEngine._design_analog_setup = _design_analog_setup
ExperimentalDesignEngine._select_sensors = _select_sensors
ExperimentalDesignEngine._define_temporal_requirements = _define_temporal_requirements
ExperimentalDesignEngine._define_spatial_requirements = _define_spatial_requirements
ExperimentalDesignEngine._design_processing_workflow = _design_processing_workflow
ExperimentalDesignEngine._select_geophysical_methods = _select_geophysical_methods
ExperimentalDesignEngine._design_survey_geometry = _design_survey_geometry
ExperimentalDesignEngine._define_resolution_targets = _define_resolution_targets
ExperimentalDesignEngine._define_success_criteria = _define_success_criteria
ExperimentalDesignEngine._assess_experiment_risks = _assess_experiment_risks
ExperimentalDesignEngine._list_required_equipment = _list_required_equipment
ExperimentalDesignEngine._design_data_collection_protocol = _design_data_collection_protocol